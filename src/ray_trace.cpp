#include "ray_trace.h"

#include <thread>

#include "window.hpp"

#include "vulkan.h"

#include <iostream>

extern "C"
__declspec(dllexport)
void ray_trace(
    uint32_t samples,
    uint32_t samplesPerRenderCall,
    bool storeRenderResult,
    uint32_t width,
    uint32_t height
) {

    window view_window{ width, height };

    // SETUP
    VulkanSettings settings = {
        .windowWidth = width,
        .windowHeight = height
    };


    std::vector<const char*> required_extensions;

    auto window_required_extensions = view_window.get_required_extensions();

    required_extensions.insert(required_extensions.end(), window_required_extensions.begin(), window_required_extensions.end());
    auto ray_trace_required_extensions = Vulkan::get_required_instance_extensions();
    required_extensions.insert(required_extensions.end(), ray_trace_required_extensions.begin(),
        ray_trace_required_extensions.end());

    auto instance = vulkan::create_instance(required_extensions);

    auto surface = view_window.create_surface(instance);

    auto physical_device = vulkan::pick_physical_device(instance, Vulkan::get_required_device_extensions());

    auto [compute_queue_family, present_queue_family] = vulkan::find_queue_family(physical_device, surface);

    auto [device, compute_queue, present_queue] = vulkan::create_device(instance, physical_device, compute_queue_family, present_queue_family, Vulkan::get_required_device_extensions());

    auto command_pool = device.createCommandPool({ .queueFamilyIndex = compute_queue_family });

    auto surface_capabilities = physical_device.getSurfaceCapabilitiesKHR(surface);
    auto swapchain_extent = surface_capabilities.currentExtent;
    if (UINT32_MAX == swapchain_extent.width) {
        swapchain_extent.width = settings.windowWidth;
        swapchain_extent.height = settings.windowHeight;
    }
    const vk::Format format = vk::Format::eR8G8B8A8Unorm;
    const vk::ColorSpaceKHR color_space = vk::ColorSpaceKHR::eSrgbNonlinear;
    vk::PresentModeKHR present_mode = vk::PresentModeKHR::eImmediate;
    auto image_count = std::max(surface_capabilities.minImageCount, 4u);
    auto swapchain = vulkan::create_swapchain(physical_device, surface, device,
        surface_capabilities.minImageCount, format, color_space, present_mode, swapchain_extent, surface_capabilities.currentTransform);

    auto swapchain_images = device.getSwapchainImagesKHR(swapchain);
    auto swapchain_image_count = swapchain_images.size();
    std::vector<vk::ImageView> swapchain_image_views(swapchain_images.size());
    std::ranges::transform(swapchain_images, swapchain_image_views.begin(),
        [device, format](auto swapChainImage) {
            return device.createImageView(
                {
                        .image = swapChainImage,
                        .viewType = vk::ImageViewType::e2D,
                        .format = format,
                        .subresourceRange = {
                                .aspectMask = vk::ImageAspectFlagBits::eColor,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1
                        }
                });
        }
    );

    vk::PhysicalDeviceMemoryProperties memory_properties = physical_device.getMemoryProperties();

    auto render_target_images = std::vector<VulkanImage>(swapchain_image_count);
    auto summed_images = std::vector<VulkanImage>(swapchain_image_count);
    {
        auto extent = vk::Extent3D{ swapchain_extent.width, swapchain_extent.height, 1 };
        std::ranges::generate(
            render_target_images,
            [device, extent, &memory_properties]() {
                return vulkan::create_image(
                    device, extent, vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc, memory_properties
                    );
            }
            );
        std::ranges::generate(
            summed_images,
            [device, extent, &memory_properties]() {
                return vulkan::create_image(
                    device, extent, vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst, memory_properties
                );
            }
        );
    }

    auto fences = vulkan::create_fences(device, swapchain_images.size());

    auto next_image_semaphores = vulkan::create_semaphores(device, swapchain_images.size() + 1);
    auto render_image_semaphores = vulkan::create_semaphores(device, swapchain_images.size());

    auto scene = generateRandomScene();

    std::vector<vk::AabbPositionsKHR> aabbs(scene.sphereAmount);

    auto aabb_buffer = vulkan::create_aabb_buffer(device, aabbs.size(), memory_properties);

    auto dynamicDispatchLoader = vk::detail::DispatchLoaderDynamic(instance, vkGetInstanceProcAddr, device);

    // ACCELERATION STRUCTURE META INFO
    vk::AccelerationStructureGeometryKHR aabbs_geometry = {
            .geometryType = vk::GeometryTypeKHR::eAabbs,
            .flags = vk::GeometryFlagBitsKHR::eOpaque
    };
    auto [bottom_accel, bottom_accel_build_info] = vulkan::createBottomAccelerationStructure(device, aabb_buffer, aabbs.size(), aabbs_geometry, memory_properties, dynamicDispatchLoader);
    // ACCELERATION STRUCTURE META INFO
    vk::AccelerationStructureGeometryKHR instances_geometry = {
            .geometryType = vk::GeometryTypeKHR::eInstances,
            .flags = vk::GeometryFlagBitsKHR::eOpaque
    };
    auto [top_accel, top_accel_build_info] = vulkan::createTopAccelerationStructure(device, bottom_accel.accelerationStructure, instances_geometry, memory_properties, dynamicDispatchLoader);

    auto rt_descriptor_set_layout = vulkan::create_descriptor_set_layout(device);
    auto rt_descriptor_pool = vulkan::create_descriptor_pool(device, swapchain_images.size());

    auto [sphere_buffer, sphere_buffer_size] = vulkan::create_sphere_buffer(device, memory_properties);
    auto render_call_info_buffers = vulkan::create_render_call_info_buffers(device, swapchain_images.size(), memory_properties);

    auto rt_descriptor_sets = vulkan::create_descriptor_set(device, swapchain_images.size(),
        rt_descriptor_set_layout, rt_descriptor_pool, render_target_images, top_accel.accelerationStructure, sphere_buffer, summed_images, render_call_info_buffers);

    auto rt_pipeline_layout = vulkan::create_pipeline_layout(device, rt_descriptor_set_layout);

    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelinePropertiesKhr = {};

    vk::PhysicalDeviceProperties2 physicalDeviceProperties2 = {
            .pNext = &rayTracingPipelinePropertiesKhr
    };

    physical_device.getProperties2(&physicalDeviceProperties2);

    auto rt_pipeline = vulkan::create_rt_pipeline(device, rayTracingPipelinePropertiesKhr.maxRayRecursionDepth, rt_pipeline_layout, dynamicDispatchLoader);

    auto [shader_binding_table_buffer, sbtRayGenAddressRegion, sbtMissAddressRegion, sbtHitAddressRegion] = vulkan::create_shader_binding_table_buffer(device, rt_pipeline, rayTracingPipelinePropertiesKhr, memory_properties, dynamicDispatchLoader);

    auto command_buffers = vulkan::create_command_buffers(
        device, command_pool, swapchain_images.size(), swapchain_images, compute_queue_family,
        render_target_images, summed_images, rt_pipeline, rt_descriptor_sets, rt_pipeline_layout,
        sbtRayGenAddressRegion, sbtMissAddressRegion, sbtHitAddressRegion,
        width, height, dynamicDispatchLoader);

    Vulkan vulkan(
        memory_properties,
        device, compute_queue, present_queue,
        { compute_queue_family, present_queue_family },
        command_pool,
        swapchain, swapchain_images, swapchain_extent,
        summed_images,
        fences,
        next_image_semaphores, render_image_semaphores,
        aabbs, aabb_buffer,
        bottom_accel_build_info,
        top_accel_build_info,
        sphere_buffer, sphere_buffer_size,
        render_call_info_buffers,
        command_buffers,
        dynamicDispatchLoader,
        settings, scene
    );
    while (!view_window.should_close()) {
        scene = generateRandomScene();
        std::ranges::transform(
            std::span{ scene.spheres, scene.sphereAmount },
            aabbs.begin(),
            [](auto& sphere) {
                auto getAABBFromSphere = [](const glm::vec4& geometry) {
                    return vk::AabbPositionsKHR{
                            .minX = geometry.x - geometry.w,
                            .minY = geometry.y - geometry.w,
                            .minZ = geometry.z - geometry.w,
                            .maxX = geometry.x + geometry.w,
                            .maxY = geometry.y + geometry.w,
                            .maxZ = geometry.z + geometry.w
                    };
                    };
                return getAABBFromSphere(sphere.geometry);
            }
        );
        vulkan::build_accel_structures(device, compute_queue, command_pool,
            aabbs, aabb_buffer, bottom_accel_build_info, top_accel_build_info,
            sphere_buffer, sphere_buffer_size, std::span{ scene.spheres, scene.sphereAmount },
            dynamicDispatchLoader);

        // RENDERING
        int requiredRenderCalls = samples / samplesPerRenderCall;

        uint32_t number = 0;
        while (number < requiredRenderCalls) {
            ++number;
            RenderCallInfo renderCallInfo = {
                .number = number,
                .samplesPerRenderCall = samplesPerRenderCall,
            };

            vulkan.render(renderCallInfo);

            view_window.poll_events();
            if (view_window.should_close()) {
                break;
            }
        }

        if (storeRenderResult) {
            std::cout << "Write to file:" << "render_result.png" << std::endl;
            vulkan.write_to_file("render_result.hdr");
            std::cout << "Write completes." << std::endl;
        }

    }

    // WINDOW
    while (!view_window.should_close()) {
        view_window.poll_events();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    device.waitIdle();

    vulkan::destroy_buffer(device, shader_binding_table_buffer);

    device.destroyPipeline(rt_pipeline);
    device.destroyPipelineLayout(rt_pipeline_layout);

    std::ranges::for_each(render_call_info_buffers, [device](auto buffer) {vulkan::destroy_buffer(device, buffer); });
    vulkan::destroy_buffer(device, sphere_buffer);

    device.destroyDescriptorPool(rt_descriptor_pool);
    device.destroyDescriptorSetLayout(rt_descriptor_set_layout);

    vulkan::destroy_acceleration_structure(device, top_accel, dynamicDispatchLoader);
    vulkan::destroy_acceleration_structure(device, bottom_accel, dynamicDispatchLoader);

    vulkan::destroy_buffer(device, aabb_buffer);

    std::ranges::for_each(next_image_semaphores, [device](auto semaphore) {device.destroySemaphore(semaphore); });
    std::ranges::for_each(render_image_semaphores, [device](auto semaphore) {device.destroySemaphore(semaphore); });
    std::ranges::for_each(fences, [device](auto fence) {device.destroyFence(fence); });

    std::ranges::for_each(render_target_images, [device](auto image) {vulkan::destroy_image(device, image); });
    std::ranges::for_each(summed_images, [device](auto image) {vulkan::destroy_image(device, image); });

    std::ranges::for_each(swapchain_image_views, [device](auto swapChainImageView) {device.destroyImageView(swapChainImageView); });
    device.destroySwapchainKHR(swapchain);
    device.destroyCommandPool(command_pool);
    device.destroy();
    instance.destroySurfaceKHR(surface);
    instance.destroy();
}