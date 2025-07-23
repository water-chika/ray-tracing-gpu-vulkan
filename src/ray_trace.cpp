#include "ray_trace.h"

#include <thread>

#include "window.hpp"

#include "vulkan.h"

#include "workload_tuner.hpp"

#include <iostream>
#include <algorithm>
#include <cstdint>
#include <execution>

extern "C"
#if WIN32
__declspec(dllexport)
#endif
void ray_trace(
    uint32_t samples,
    bool storeRenderResult,
    uint32_t width,
    uint32_t height,
    uint32_t gpu_count
) {
    auto window_system = window::init_window_system();

    // SETUP
    VulkanSettings settings = {
        .windowWidth = width,
        .windowHeight = height
    };


    std::vector<const char*> required_extensions;

    auto window_required_extensions = window::get_vulkan_required_extensions(window_system);

    required_extensions.insert(required_extensions.end(), window_required_extensions.begin(), window_required_extensions.end());
    auto ray_trace_required_extensions = Vulkan::get_required_instance_extensions();
    required_extensions.insert(required_extensions.end(), ray_trace_required_extensions.begin(),
        ray_trace_required_extensions.end());

    auto instance = vulkan::create_instance(required_extensions);

    auto physical_devices = vulkan::pick_physical_devices(instance, Vulkan::get_required_device_extensions());
    if (physical_devices.size() > gpu_count) {
        physical_devices.resize(gpu_count);
    }
    if (physical_devices.size() == 0) {
        throw std::runtime_error{ "No GPUs with required extensions" };
    }
    auto physical_device_indices = std::vector<uint32_t>(physical_devices.size());
    std::ranges::iota(physical_device_indices, 0);

    int test_physical_device_index = 0;

    auto physical_devices_memory_properties = std::vector<vk::PhysicalDeviceMemoryProperties>(physical_devices.size());
    std::ranges::transform(
        physical_devices,
        physical_devices_memory_properties.begin(),
        [](auto physical_device) {
            return physical_device.getMemoryProperties();
        }
    );

    auto physical_devices_window = std::vector<window::window>(physical_devices.size());
    std::ranges::transform(
        physical_device_indices,
        physical_devices_window.begin(),
        [&window_system, width=width/physical_devices.size(), height=height/physical_devices.size()](auto i) {
            return window::create_window(window_system, width, height);
        }
    );
    auto view_window = physical_devices_window[test_physical_device_index];

    auto physical_devices_render_extent = std::vector<glm::u32vec2>(physical_devices.size());
    std::ranges::generate(
        physical_devices_render_extent,
        [width, height, physical_device_count = physical_devices.size()]() {
            return glm::u32vec2{ width, height / physical_device_count };
        }
    );
    physical_devices_render_extent[0].y += height - physical_devices.size() * (height / physical_devices.size());

    tune::tuning_info tuning_info{};
    tune::init_tuning_info(tuning_info, height, physical_devices.size());

    uint32_t benchmark_frame_count = 100;

    while (!window::should_window_close(view_window)) {
        auto physical_devices_render_offset = std::vector<glm::u32vec2>(physical_devices.size());
        physical_devices_render_offset[0] = { 0, 0 };
        for (int i = 1; i < physical_devices.size(); i++) {
            physical_devices_render_offset[i] = { 0, physical_devices_render_offset[i - 1].y + physical_devices_render_extent[i - 1].y };

        }

        std::ranges::for_each(
            physical_device_indices,
            [&physical_devices_window, &physical_devices_render_offset, &physical_devices_render_extent](auto i) {
                auto& window = physical_devices_window[i];
                auto& offset = physical_devices_render_offset[i];
                auto& extent = physical_devices_render_extent[i];
                window::set_window_position(window, std::pair{ offset.x,offset.y });
                window::set_window_size(window, std::pair{ extent.x, extent.y });
            }
        );

        auto physical_devices_surface = std::vector<vk::SurfaceKHR>(physical_devices.size());
        std::ranges::transform(
            physical_devices_window,
            physical_devices_surface.begin(),
            [instance](auto& window) {
                return window::create_window_vulkan_surface(window, instance);
            }
        );

        auto compute_queue_families = std::vector<uint32_t>(physical_devices.size());
        auto present_queue_families = std::vector<uint32_t>(physical_devices.size());
        std::ranges::for_each(
            physical_device_indices,
            [&compute_queue_families, &present_queue_families, &physical_devices, &physical_devices_surface](auto i) {
                auto [compute_queue_family, present_queue_family] = vulkan::find_queue_family(physical_devices[i], physical_devices_surface[i]);
            }
        );


        auto devices = std::vector<vk::Device>(physical_devices.size());
        auto physical_devices_compute_queue = std::vector<vk::Queue>(physical_devices.size());
        auto physical_devices_present_queue = std::vector<vk::Queue>(physical_devices.size());

        std::ranges::for_each(
            physical_device_indices,
            [&devices, &physical_devices_compute_queue, &physical_devices_present_queue, instance, &physical_devices, &compute_queue_families, &present_queue_families](auto i) {
                auto [device, compute_queue, present_queue] = vulkan::create_device(instance, physical_devices[i], compute_queue_families[i], present_queue_families[i],
                    Vulkan::get_required_device_extensions());
                devices[i] = device;
                physical_devices_compute_queue[i] = compute_queue;
                physical_devices_present_queue[i] = present_queue;
            }
        );



        auto physical_devices_command_pool = std::vector<vk::CommandPool>(physical_devices.size());
        std::ranges::for_each(
            physical_device_indices,
            [&physical_devices_command_pool, &devices, &compute_queue_families](auto i) {
                auto command_pool = devices[i].createCommandPool({ .queueFamilyIndex = compute_queue_families[i] });
                physical_devices_command_pool[i] = command_pool;
            }
        );

        auto physical_devices_surface_capabilities = std::vector<vk::SurfaceCapabilitiesKHR>(physical_devices.size());
        std::ranges::transform(
            physical_device_indices,
            physical_devices_surface_capabilities.begin(),
            [&physical_devices, &physical_devices_surface](auto i) {
                return physical_devices[i].getSurfaceCapabilitiesKHR(physical_devices_surface[i]);
            }
        );

        auto physical_devices_swapchain_extent = std::vector<vk::Extent2D>(physical_devices.size());
        std::ranges::transform(
            physical_device_indices,
            physical_devices_swapchain_extent.begin(),
            [&physical_devices_render_extent, &physical_devices_surface_capabilities](auto i) {
                auto swapchain_extent = physical_devices_surface_capabilities[i].currentExtent;
                if (UINT32_MAX == swapchain_extent.width) {
                    swapchain_extent.width = physical_devices_render_extent[i].x;
                    swapchain_extent.height = physical_devices_render_extent[i].y;
                }
                return swapchain_extent;
            });

        const vk::Format format = vk::Format::eR8G8B8A8Unorm;
        const vk::ColorSpaceKHR color_space = vk::ColorSpaceKHR::eSrgbNonlinear;
        vk::PresentModeKHR present_mode = vk::PresentModeKHR::eImmediate;
        auto image_count = std::ranges::max(physical_devices_surface_capabilities, std::ranges::less{},
            [](auto& surface_capabilities) { return surface_capabilities.minImageCount; }
        ).minImageCount;
        auto surface_transform = physical_devices_surface_capabilities[0].currentTransform;

        auto physical_devices_swapchain = std::vector<vk::SwapchainKHR>(physical_devices.size());
        std::ranges::transform(
            physical_device_indices,
            physical_devices_swapchain.begin(),
            [&physical_devices, &physical_devices_surface, &devices, image_count, format, color_space, present_mode, &physical_devices_swapchain_extent, surface_transform](auto i) {
                return vulkan::create_swapchain(physical_devices[i], physical_devices_surface[i], devices[i],
                    image_count, format, color_space, present_mode, physical_devices_swapchain_extent[i], surface_transform);
            }
        );

        auto physical_devices_swapchain_images = std::vector<std::vector<vk::Image>>(physical_devices.size());
        std::ranges::transform(
            physical_device_indices,
            physical_devices_swapchain_images.begin(),
            [&devices, &physical_devices_swapchain](auto i) {
                return devices[i].getSwapchainImagesKHR(physical_devices_swapchain[i]);
            }
        );
        auto physical_devices_swapchain_image_count = std::vector<uint32_t>(physical_devices.size());
        std::ranges::transform(
            physical_devices_swapchain_images,
            physical_devices_swapchain_image_count.begin(),
            [](auto& images) {
                return images.size();
            }
        );

        auto physical_devices_swapchain_image_views = std::vector<std::vector<vk::ImageView>>(physical_devices.size());
        std::ranges::transform(
            physical_device_indices,
            physical_devices_swapchain_image_views.begin(),
            [&devices, &physical_devices_swapchain_images, physical_devices_swapchain_image_count, format](auto i) {
                std::vector<vk::ImageView> swapchain_image_views(physical_devices_swapchain_image_count[i]);
                std::ranges::transform(physical_devices_swapchain_images[i], swapchain_image_views.begin(),
                    [device = devices[i], format](auto swapChainImage) {
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
                return swapchain_image_views;
            }
        );

        auto physical_devices_render_image_count = physical_devices_swapchain_image_count;
        auto physical_devices_render_image_indices = std::vector<std::vector<uint32_t>>(physical_devices.size());
        std::ranges::transform(
            physical_devices_render_image_count,
            physical_devices_render_image_indices.begin(),
            [](auto render_image_count) {
                auto render_image_indices = std::vector<uint32_t>(render_image_count);
                std::ranges::iota(render_image_indices, 0);
                return render_image_indices;
            }
        );

        auto physical_devices_render_target_images = std::vector<std::vector<VulkanImage>>(devices.size());
        auto physical_devices_summed_images = std::vector<std::vector<VulkanImage>>(devices.size());

        std::ranges::for_each(
            physical_device_indices,
            [&physical_devices_render_target_images, &physical_devices_summed_images, physical_devices_render_image_count, physical_devices_swapchain_extent, &devices, &physical_devices_memory_properties](auto i) {
                auto render_target_images = std::vector<VulkanImage>(physical_devices_render_image_count[i]);
                auto summed_images = std::vector<VulkanImage>(physical_devices_render_image_count[i]);
                {
                    auto extent = vk::Extent3D{ physical_devices_swapchain_extent[i].width, physical_devices_swapchain_extent[i].height, 1 };
                    std::ranges::generate(
                        render_target_images,
                        [device = devices[i], extent, &memory_properties = physical_devices_memory_properties[i]]() {
                            return vulkan::create_image(
                                device, extent, vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc, memory_properties
                            );
                        }
                    );
                    std::ranges::generate(
                        summed_images,
                        [device = devices[i], extent, &memory_properties = physical_devices_memory_properties[i]]() {
                            return vulkan::create_image(
                                device, extent, vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst, memory_properties
                            );
                        }
                    );
                }
                physical_devices_render_target_images[i] = render_target_images;
                physical_devices_summed_images[i] = summed_images;
            });
        auto render_target_images = physical_devices_render_target_images[test_physical_device_index];
        auto summed_images = physical_devices_summed_images[test_physical_device_index];

        auto physical_devices_fences = std::vector<std::vector<vk::Fence>>(physical_devices.size());
        std::ranges::transform(
            physical_device_indices,
            physical_devices_fences.begin(),
            [&physical_devices_render_image_count, &devices](auto i) {
                auto fences = vulkan::create_fences(devices[i], physical_devices_render_image_count[i]);
                return fences;
            }
        );

        auto physical_devices_next_image_semaphores = std::vector<std::vector<vk::Semaphore>>(physical_devices.size());
        std::ranges::transform(
            physical_device_indices,
            physical_devices_next_image_semaphores.begin(),
            [&physical_devices_render_image_count, &devices](auto i) {
                auto next_image_semaphores = vulkan::create_semaphores(devices[i], physical_devices_render_image_count[i] + 1);
                return next_image_semaphores;
            }
        );

        auto physical_devices_render_image_semaphores = std::vector<std::vector<vk::Semaphore>>(physical_devices.size());
        std::ranges::transform(
            physical_device_indices,
            physical_devices_render_image_semaphores.begin(),
            [&physical_devices_render_image_count, &devices](auto i) {
                auto render_image_semaphores = vulkan::create_semaphores(devices[i], physical_devices_render_image_count[i]);
                return render_image_semaphores;
            }
        );

        auto scene = generateRandomScene();

        auto sphere_amount = scene.sphereAmount;

        auto physical_devices_aabb_buffers = std::vector<std::vector<VulkanBuffer>>(physical_devices.size());
        std::ranges::transform(
            physical_device_indices,
            physical_devices_aabb_buffers.begin(),
            [sphere_amount, &physical_devices_render_image_count, &devices, &physical_devices_memory_properties](auto i) {
                auto aabb_buffers = std::vector<VulkanBuffer>(physical_devices_render_image_count[i]);
                std::ranges::generate(
                    aabb_buffers,
                    [device = devices[i], sphere_amount, &memory_properties = physical_devices_memory_properties[i]]() {
                        return vulkan::create_aabb_buffer(device, sphere_amount, memory_properties);
                    }
                );
                return aabb_buffers;
            }
        );
        auto aabb_buffers = physical_devices_aabb_buffers[test_physical_device_index];

        std::vector<vk::AabbPositionsKHR> aabbs(sphere_amount);

        auto physical_devices_dynamic_dispatch_loader = std::vector<vk::detail::DispatchLoaderDynamic>(physical_devices.size());
        std::ranges::transform(
            devices,
            physical_devices_dynamic_dispatch_loader.begin(),
            [instance](auto device) {
                return vk::detail::DispatchLoaderDynamic(instance, vkGetInstanceProcAddr, device);
            }
        );
        auto dynamicDispatchLoader = physical_devices_dynamic_dispatch_loader[test_physical_device_index];

        auto physical_devices_aabbs_geometries = std::vector<std::vector<vk::AccelerationStructureGeometryKHR>>(physical_devices.size());
        auto physical_devices_bottom_accels = std::vector<std::vector<VulkanAccelerationStructure>>(physical_devices.size());
        auto physical_devices_bottom_accel_build_infos = std::vector<std::vector<vk::AccelerationStructureBuildGeometryInfoKHR>>(physical_devices.size());
        std::ranges::for_each(
            physical_device_indices,
            [&physical_devices_aabbs_geometries, &physical_devices_bottom_accels, &physical_devices_bottom_accel_build_infos, &physical_devices_render_image_count, &physical_devices_render_image_indices,
            &devices, sphere_amount, &physical_devices_aabb_buffers, &physical_devices_memory_properties, &physical_devices_dynamic_dispatch_loader](auto i) {
                auto render_image_count = physical_devices_render_image_count[i];
                auto& aabbs_geometries = physical_devices_aabbs_geometries[i];
                aabbs_geometries.resize(render_image_count);
                std::ranges::fill(aabbs_geometries, vk::AccelerationStructureGeometryKHR{ .geometryType = vk::GeometryTypeKHR::eAabbs, .flags = vk::GeometryFlagBitsKHR::eOpaque });

                auto bottom_accels = std::vector<VulkanAccelerationStructure>(render_image_count);
                auto bottom_accel_build_infos = std::vector<vk::AccelerationStructureBuildGeometryInfoKHR>(render_image_count);
                std::ranges::for_each(
                    physical_devices_render_image_indices[i],
                    [&bottom_accels, &bottom_accel_build_infos, &aabbs_geometries,
                    device = devices[i], &aabb_buffers = physical_devices_aabb_buffers[i], sphere_amount,
                    &memory_properties = physical_devices_memory_properties[i], &dynamicDispatchLoader = physical_devices_dynamic_dispatch_loader[i]](uint32_t i) {
                        auto [bottom_accel, bottom_accel_build_info] = vulkan::createBottomAccelerationStructure(device, aabb_buffers[i], sphere_amount, aabbs_geometries[i], memory_properties, dynamicDispatchLoader);
                        bottom_accels[i] = bottom_accel;
                        bottom_accel_build_infos[i] = bottom_accel_build_info;
                    }
                );
                physical_devices_bottom_accels[i] = bottom_accels;
                physical_devices_bottom_accel_build_infos[i] = bottom_accel_build_infos;
            }
        );
        auto& aabbs_geometries = physical_devices_aabbs_geometries[test_physical_device_index];
        auto& bottom_accels = physical_devices_bottom_accels[test_physical_device_index];
        auto& bottom_accel_build_infos = physical_devices_bottom_accel_build_infos[test_physical_device_index];


        auto physical_devices_instances_geometries = std::vector<std::vector<vk::AccelerationStructureGeometryKHR>>(physical_devices.size());
        auto physical_devices_top_accels = std::vector<std::vector<VulkanAccelerationStructure>>(physical_devices.size());
        auto physical_devices_top_accel_build_infos = std::vector<std::vector<vk::AccelerationStructureBuildGeometryInfoKHR>>(physical_devices.size());
        std::ranges::for_each(
            physical_device_indices,
            [&physical_devices_instances_geometries, &physical_devices_top_accels, &physical_devices_top_accel_build_infos,
            &physical_devices_render_image_indices, &physical_devices_render_image_count,
            &devices, &physical_devices_memory_properties, &physical_devices_bottom_accels, &physical_devices_dynamic_dispatch_loader](auto i) {
                auto render_image_count = physical_devices_render_image_count[i];
                auto& instances_geometries = physical_devices_instances_geometries[i];
                instances_geometries.resize(render_image_count);
                std::ranges::fill(instances_geometries, vk::AccelerationStructureGeometryKHR{ .geometryType = vk::GeometryTypeKHR::eInstances, .flags = vk::GeometryFlagBitsKHR::eOpaque });

                auto top_accels = std::vector<VulkanAccelerationStructure>(render_image_count);
                auto top_accel_build_infos = std::vector<vk::AccelerationStructureBuildGeometryInfoKHR>(render_image_count);
                std::ranges::for_each(
                    physical_devices_render_image_indices[i],
                    [&top_accels, &top_accel_build_infos, &instances_geometries,
                    device = devices[i], &bottom_accels = physical_devices_bottom_accels[i],
                    &memory_properties = physical_devices_memory_properties[i], &dynamicDispatchLoader = physical_devices_dynamic_dispatch_loader[i]](uint32_t i) {
                        auto [top_accel, top_accel_build_info] = vulkan::createTopAccelerationStructure(device, bottom_accels[i].accelerationStructure, instances_geometries[i], memory_properties, dynamicDispatchLoader);
                        top_accels[i] = top_accel;
                        top_accel_build_infos[i] = top_accel_build_info;
                    }
                );
                physical_devices_top_accels[i] = top_accels;
                physical_devices_top_accel_build_infos[i] = top_accel_build_infos;
            });
        auto instances_geometries = physical_devices_instances_geometries[test_physical_device_index];
        auto top_accels = physical_devices_top_accels[test_physical_device_index];
        auto top_accel_build_infos = physical_devices_top_accel_build_infos[test_physical_device_index];

        auto physical_devices_rt_descriptor_set_layout = std::vector<vk::DescriptorSetLayout>(physical_devices.size());
        std::ranges::transform(
            devices,
            physical_devices_rt_descriptor_set_layout.begin(),
            [](auto device) { return vulkan::create_descriptor_set_layout(device); }
        );
        auto rt_descriptor_set_layout = physical_devices_rt_descriptor_set_layout[test_physical_device_index];

        auto physical_devices_rt_descriptor_pool = std::vector<vk::DescriptorPool>(physical_devices.size());
        std::ranges::transform(
            physical_device_indices,
            physical_devices_rt_descriptor_pool.begin(),
            [&physical_devices_render_image_count, &devices](auto i) { return vulkan::create_descriptor_pool(devices[i], physical_devices_render_image_count[i]); }
        );
        auto rt_descriptor_pool = physical_devices_rt_descriptor_pool[test_physical_device_index];

        auto physical_devices_sphere_buffers = std::vector<std::vector<VulkanBuffer>>(physical_devices.size());
        std::ranges::transform(
            physical_device_indices,
            physical_devices_sphere_buffers.begin(),
            [&devices, &physical_devices_memory_properties, &physical_devices_render_image_count](auto i) {
                auto sphere_buffers = std::vector<VulkanBuffer>(physical_devices_render_image_count[i]);
                std::ranges::generate(
                    sphere_buffers,
                    [device = devices[i], &memory_properties = physical_devices_memory_properties[i]]() { return vulkan::create_sphere_buffer(device, memory_properties); }
                );
                return sphere_buffers;
            }
        );
        auto sphere_buffers = physical_devices_sphere_buffers[test_physical_device_index];

        auto physical_devices_render_call_info_buffers = std::vector<std::vector<VulkanBuffer>>(physical_devices.size());
        std::ranges::transform(
            physical_device_indices,
            physical_devices_render_call_info_buffers.begin(),
            [&devices, &physical_devices_memory_properties, &physical_devices_render_image_count](auto i) {
                return vulkan::create_render_call_info_buffers(devices[i], physical_devices_render_image_count[i], physical_devices_memory_properties[i]);
            });
        auto render_call_info_buffers = physical_devices_render_call_info_buffers[test_physical_device_index];

        auto physical_devices_rt_descriptor_sets = std::vector<std::vector<vk::DescriptorSet>>(physical_devices.size());
        std::ranges::transform(
            physical_device_indices,
            physical_devices_rt_descriptor_sets.begin(),
            [&devices, &physical_devices_render_image_count, &physical_devices_rt_descriptor_set_layout,
            &physical_devices_rt_descriptor_pool, &physical_devices_render_target_images,
            &physical_devices_top_accels, &physical_devices_sphere_buffers, &physical_devices_summed_images,
            &physical_devices_render_call_info_buffers](auto i) {
                return vulkan::create_descriptor_set(devices[i], physical_devices_render_image_count[i],
                    physical_devices_rt_descriptor_set_layout[i], physical_devices_rt_descriptor_pool[i], physical_devices_render_target_images[i],
                    physical_devices_top_accels[i], physical_devices_sphere_buffers[i], physical_devices_summed_images[i], physical_devices_render_call_info_buffers[i]);
            });
        auto rt_descriptor_sets = physical_devices_rt_descriptor_sets[test_physical_device_index];

        auto physical_devices_rt_pipeline_layout = std::vector<vk::PipelineLayout>(physical_devices.size());
        std::ranges::transform(
            physical_device_indices,
            physical_devices_rt_pipeline_layout.begin(),
            [&devices, &physical_devices_rt_descriptor_set_layout](auto i) {
                return vulkan::create_pipeline_layout(devices[i], physical_devices_rt_descriptor_set_layout[i]);
            }
        );
        auto rt_pipeline_layout = physical_devices_rt_pipeline_layout[test_physical_device_index];

        auto physical_devices_ray_tracing_pipeline_properties = std::vector<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>(physical_devices.size());
        std::ranges::transform(
            physical_devices,
            physical_devices_ray_tracing_pipeline_properties.begin(),
            [](auto physical_device) {
                vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelinePropertiesKhr = {};

                vk::PhysicalDeviceProperties2 physicalDeviceProperties2 = {
                        .pNext = &rayTracingPipelinePropertiesKhr
                };

                physical_device.getProperties2(&physicalDeviceProperties2);
                return rayTracingPipelinePropertiesKhr;
            }
        );
        auto rayTracingPipelinePropertiesKhr = physical_devices_ray_tracing_pipeline_properties[test_physical_device_index];

        auto max_ray_recursion_depth = std::ranges::min(physical_devices_ray_tracing_pipeline_properties,
            std::ranges::less{},
            [](auto& props) {
                return props.maxRayRecursionDepth;
            }).maxRayRecursionDepth;

        auto physical_devices_rt_pipeline = std::vector<vk::Pipeline>(physical_devices.size());
        std::ranges::transform(
            physical_device_indices,
            physical_devices_rt_pipeline.begin(),
            [&devices, max_ray_recursion_depth, &physical_devices_rt_pipeline_layout, &physical_devices_dynamic_dispatch_loader](auto i) {
                return vulkan::create_rt_pipeline(devices[i], max_ray_recursion_depth, physical_devices_rt_pipeline_layout[i], physical_devices_dynamic_dispatch_loader[i]);
            }
        );
        auto rt_pipeline = physical_devices_rt_pipeline[test_physical_device_index];

        auto physical_devices_shader_binding_table_buffer = std::vector<VulkanBuffer>(physical_devices.size());
        auto physical_devices_sbt_ray_gen_address_region = std::vector<vk::StridedDeviceAddressRegionKHR>(physical_devices.size());
        auto physical_devices_sbt_miss_address_region = std::vector<vk::StridedDeviceAddressRegionKHR>(physical_devices.size());
        auto physical_devices_sbt_hit_address_region = std::vector<vk::StridedDeviceAddressRegionKHR>(physical_devices.size());
        std::ranges::for_each(
            physical_device_indices,
            [&physical_devices_shader_binding_table_buffer, &physical_devices_sbt_ray_gen_address_region, &physical_devices_sbt_miss_address_region, &physical_devices_sbt_hit_address_region,
            &devices, &physical_devices_rt_pipeline, &physical_devices_ray_tracing_pipeline_properties,
            &physical_devices_memory_properties, &physical_devices_dynamic_dispatch_loader](auto i) {
                auto [shader_binding_table_buffer, sbtRayGenAddressRegion, sbtMissAddressRegion, sbtHitAddressRegion] =
                    vulkan::create_shader_binding_table_buffer(devices[i], physical_devices_rt_pipeline[i], physical_devices_ray_tracing_pipeline_properties[i],
                        physical_devices_memory_properties[i], physical_devices_dynamic_dispatch_loader[i]);
                physical_devices_shader_binding_table_buffer[i] = shader_binding_table_buffer;
                physical_devices_sbt_ray_gen_address_region[i] = sbtRayGenAddressRegion;
                physical_devices_sbt_miss_address_region[i] = sbtMissAddressRegion;
                physical_devices_sbt_hit_address_region[i] = sbtHitAddressRegion;
            }
        );
        auto shader_binding_table_buffer = physical_devices_shader_binding_table_buffer[test_physical_device_index];
        auto sbtRayGenAddressRegion = physical_devices_sbt_ray_gen_address_region[test_physical_device_index];
        auto sbtMissAddressRegion = physical_devices_sbt_miss_address_region[test_physical_device_index];
        auto sbtHitAddressRegion = physical_devices_sbt_hit_address_region[test_physical_device_index];



        auto physical_devices_command_buffers = std::vector<std::vector<vk::CommandBuffer>>(physical_devices.size());
        std::ranges::transform(
            physical_device_indices,
            physical_devices_command_buffers.begin(),
            [&devices, &physical_devices_command_pool, &physical_devices_render_image_count, &physical_devices_swapchain_images, compute_queue_families,
            &physical_devices_render_target_images, &physical_devices_summed_images, &physical_devices_rt_pipeline, &physical_devices_rt_descriptor_sets, &physical_devices_rt_pipeline_layout,
            &aabbs, &physical_devices_bottom_accel_build_infos, &physical_devices_bottom_accels,
            &physical_devices_top_accel_build_infos, &physical_devices_top_accels,
            &physical_devices_sbt_ray_gen_address_region, &physical_devices_sbt_miss_address_region, &physical_devices_sbt_hit_address_region,
            &physical_devices_render_extent, &physical_devices_swapchain_extent, &physical_devices_dynamic_dispatch_loader](auto i) {
                return vulkan::create_command_buffers(
                    devices[i], physical_devices_command_pool[i], physical_devices_render_image_count[i], physical_devices_swapchain_images[i], compute_queue_families[i],
                    physical_devices_render_target_images[i], physical_devices_summed_images[i], physical_devices_rt_pipeline[i], physical_devices_rt_descriptor_sets[i], physical_devices_rt_pipeline_layout[i],
                    aabbs, physical_devices_bottom_accel_build_infos[i], physical_devices_bottom_accels[i],
                    physical_devices_top_accel_build_infos[i], physical_devices_top_accels[i],
                    physical_devices_sbt_ray_gen_address_region[i], physical_devices_sbt_miss_address_region[i], physical_devices_sbt_hit_address_region[i],
                    physical_devices_render_extent[i].x, physical_devices_render_extent[i].y,
                    physical_devices_swapchain_extent[i],
                    physical_devices_dynamic_dispatch_loader[i]);
            }
        );

        auto physical_devices_next_image_semaphores_indices = std::vector<std::vector<uint32_t>>(physical_devices.size());
        std::ranges::transform(
            physical_device_indices,
            physical_devices_next_image_semaphores_indices.begin(),
            [&physical_devices_render_image_count](auto i) {
                auto next_image_semaphores_indices = std::vector<uint32_t>(physical_devices_render_image_count[i]);
                std::ranges::iota(next_image_semaphores_indices, 0);
                return next_image_semaphores_indices;
            }
        );

        auto physical_devices_next_image_free_semaphore_index = std::vector<uint32_t>(physical_devices.size());
        physical_devices_next_image_free_semaphore_index = physical_devices_render_image_count;

        while (!window::should_window_close(view_window)) {
            auto physical_devices_present_time = std::vector<std::chrono::steady_clock::time_point>(physical_devices.size());
            std::ranges::generate(
                physical_devices_present_time,
                []() {
                    return std::chrono::steady_clock::now();
                }
            );
            auto physical_devices_duration_of_gpu = std::vector<std::chrono::steady_clock::duration>(physical_devices.size());
            auto begin_time = std::chrono::steady_clock::now();
            uint32_t frame_index = 0;

            while (!window::should_window_close(view_window)
                && frame_index++ < benchmark_frame_count) {
                auto cursor_pos = window::get_window_cursor_position(view_window);
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

                auto [x, y] = cursor_pos;
                x /= 500.0;
                y /= 500.0;
                auto camera_dir = glm::vec3{ sin(x) * cos(y), -sin(y), cos(x) * cos(y) };

                {
                    auto spheres = std::span{ scene.spheres, scene.sphereAmount };

                    auto physical_devices_acquire_image_time = std::vector<std::chrono::steady_clock::time_point>(physical_devices.size());
                    auto physical_devices_swapchain_image_index = std::vector<uint32_t>(physical_devices.size());
                    auto physical_devices_acquire_image_semaphore = std::vector<vk::Semaphore>(physical_devices.size());
                    std::for_each(
                        std::execution::par_unseq,
                        physical_device_indices.begin(), physical_device_indices.end(),
                        [&physical_devices_swapchain_image_index,
                        &physical_devices_acquire_image_semaphore,
                        &devices,
                        &physical_devices_next_image_semaphores, &physical_devices_swapchain,
                        &physical_devices_next_image_free_semaphore_index, &physical_devices_next_image_semaphores_indices,
                        &physical_devices_acquire_image_time](auto i) {
                            uint32_t swapchain_image_index = 0;
                            auto acquire_image_semaphore = physical_devices_next_image_semaphores[i][physical_devices_next_image_free_semaphore_index[i]];
                            if (auto [result, index] = devices[i].acquireNextImageKHR(physical_devices_swapchain[i], UINT64_MAX, acquire_image_semaphore);
                                result == vk::Result::eSuccess || result == vk::Result::eSuboptimalKHR) {
                                swapchain_image_index = index;
                            }
                            else {
                                throw std::runtime_error{ "failed to acquire next image" };
                            }
                            physical_devices_acquire_image_time[i] = std::chrono::steady_clock::now();
                            std::swap(physical_devices_next_image_free_semaphore_index[i], physical_devices_next_image_semaphores_indices[i][swapchain_image_index]);
                            physical_devices_acquire_image_semaphore[i] = acquire_image_semaphore;
                            physical_devices_swapchain_image_index[i] = swapchain_image_index;
                        }
                    );
                    std::ranges::for_each(
                        physical_device_indices,
                        [&physical_devices_acquire_image_time,
                        &physical_devices_present_time,
                        &physical_devices_duration_of_gpu](auto i) {
                            auto duration = physical_devices_acquire_image_time[i] - physical_devices_present_time[i];
                            physical_devices_duration_of_gpu[i] += duration;
                        }
                    );

                    std::ranges::for_each(
                        physical_device_indices,
                        [&devices, &physical_devices_fences, &physical_devices_swapchain_image_index](auto i) {
                            auto fence = physical_devices_fences[i][physical_devices_swapchain_image_index[i]];
                            {
                                vk::Result res = devices[i].waitForFences(fence, true, UINT64_MAX);
                                if (res != vk::Result::eSuccess) {
                                    throw std::runtime_error{ "failed to wait fences" };
                                }
                            }
                            devices[i].resetFences(fence);
                        }
                    );

                    std::ranges::for_each(
                        physical_device_indices,
                        [&devices, &physical_devices_swapchain_image_index, samples, width, height, &physical_devices_render_offset,
                        &physical_devices_render_call_info_buffers, &camera_dir](auto i) {
                            RenderCallInfo renderCallInfo = {
                                .number = 0,
                                .samplesPerRenderCall = samples,
                                .offset = physical_devices_render_offset[i],
                                .image_size = {width, height},
                                .camera_pos = {13.0f, 11.0f, -3.0f, 0},
                                .camera_dir = {-13.0f, -11.0f, 3.0f, 0},
                            };
                            void* data = devices[i].mapMemory(physical_devices_render_call_info_buffers[i][physical_devices_swapchain_image_index[i]].memory, 0, sizeof(RenderCallInfo));
                            memcpy(data, &renderCallInfo, sizeof(RenderCallInfo));
                            devices[i].unmapMemory(physical_devices_render_call_info_buffers[i][physical_devices_swapchain_image_index[i]].memory);
                        }
                    );

                    std::ranges::for_each(
                        physical_device_indices,
                        [&devices, &aabbs, &physical_devices_aabb_buffers, &physical_devices_swapchain_image_index, &physical_devices_sphere_buffers, &spheres](auto i) {
                            vulkan::update_accel_structures_data(devices[i],
                                aabbs, physical_devices_aabb_buffers[i][physical_devices_swapchain_image_index[i]],
                                physical_devices_sphere_buffers[i][physical_devices_swapchain_image_index[i]], spheres.size_bytes(), spheres);
                        }
                    );

                    std::for_each(
                        std::execution::par_unseq,
                        physical_device_indices.begin(), physical_device_indices.end(),
                        [&physical_devices_compute_queue, &physical_devices_command_buffers,
                        &physical_devices_render_image_semaphores, &physical_devices_swapchain_image_index,
                        &physical_devices_acquire_image_semaphore, &physical_devices_fences](auto i) {
                            auto swapchain_image_index = physical_devices_swapchain_image_index[i];
                            auto render_image_semaphore = physical_devices_render_image_semaphores[i][swapchain_image_index];

                            auto wait_semaphores = std::array{ physical_devices_acquire_image_semaphore[i] };
                            auto  wait_stage_masks =
                                std::array<vk::PipelineStageFlags, 1>{ vk::PipelineStageFlagBits::eAllCommands };
                            auto signal_semaphores = std::array{ render_image_semaphore };
                            auto submitInfo = vk::SubmitInfo{}
                                .setCommandBuffers(physical_devices_command_buffers[i][swapchain_image_index])
                                .setWaitSemaphores(wait_semaphores)
                                .setWaitDstStageMask(wait_stage_masks)
                                .setSignalSemaphores(signal_semaphores);

                            auto res = physical_devices_compute_queue[i].submit(1, &submitInfo, physical_devices_fences[i][swapchain_image_index]);
                            if (res != vk::Result::eSuccess) {
                                throw std::runtime_error{ "failed to submit" };
                            }
                        }
                    );

                    std::for_each(
                        std::execution::par_unseq,
                        physical_device_indices.begin(), physical_device_indices.end(),
                        [&physical_devices_present_queue,
                        &physical_devices_render_image_semaphores, &physical_devices_swapchain, &physical_devices_swapchain_image_index,
                        &physical_devices_present_time](auto i) {
                            auto present_queue = physical_devices_present_queue[i];
                            auto swapchain_image_index = physical_devices_swapchain_image_index[i];
                            vk::PresentInfoKHR presentInfo = {
                                    .waitSemaphoreCount = 1,
                                    .pWaitSemaphores = &physical_devices_render_image_semaphores[i][swapchain_image_index],
                                    .swapchainCount = 1,
                                    .pSwapchains = &physical_devices_swapchain[i],
                                    .pImageIndices = &swapchain_image_index
                            };

                            auto res = present_queue.presentKHR(presentInfo);
                            if (res != vk::Result::eSuccess) {
                                std::cerr << "present return: " << res << std::endl;
                            }
                            physical_devices_present_time[i] = std::chrono::steady_clock::now();
                        }
                    );
                }

                window::poll_events(window_system);
            }

            auto end_time = std::chrono::steady_clock::now();
            auto duration = end_time - begin_time;
            auto frame_count = frame_index;
            auto duration_per_frame = duration / frame_count;
            std::cout << "duration_per_frame: " << duration_per_frame << std::endl;

            using namespace std::literals;
            benchmark_frame_count = (4s + 50*duration_per_frame) / duration_per_frame;

            auto frame_info = tune::frame_info{
                .workload_distribution = std::vector<uint32_t>(physical_devices.size()),
                .duration = duration_per_frame,
                .estimated_gpu_duration = std::vector<std::chrono::steady_clock::duration>(physical_devices.size())
            };
            std::ranges::for_each(
                physical_device_indices,
                [&frame_info, &physical_devices_render_extent, &physical_devices_duration_of_gpu, frame_count](auto i) {
                    frame_info.workload_distribution[i] = physical_devices_render_extent[i].y;
                    frame_info.estimated_gpu_duration[i] = physical_devices_duration_of_gpu[i] / frame_count;
                }
            );
            tune::add_frame_info(tuning_info, std::move(frame_info));

            auto opt_next_workload_distribution = tune::get_workload(tuning_info);
            if (opt_next_workload_distribution) {
                auto& next_workload_distribution = opt_next_workload_distribution.value();
                std::ranges::transform(
                    next_workload_distribution,
                    physical_devices_render_extent.begin(),
                    [width](auto w) {
                        return glm::uvec2{ width, w };
                    }
                );
                break;
            }
        }

        std::ranges::for_each(
            devices,
            [](auto& device) {
                device.waitIdle();
            }
        );

        std::ranges::for_each(
            physical_device_indices,
            [&devices, &physical_devices_shader_binding_table_buffer](auto i) {
                vulkan::destroy_buffer(devices[i], physical_devices_shader_binding_table_buffer[i]);
            });

        std::ranges::for_each(
            physical_device_indices,
            [&devices, &physical_devices_rt_pipeline](auto i) {
                devices[i].destroyPipeline(physical_devices_rt_pipeline[i]);
            });
        std::ranges::for_each(
            physical_device_indices,
            [&devices, &physical_devices_rt_pipeline_layout](auto i) {
                devices[i].destroyPipelineLayout(physical_devices_rt_pipeline_layout[i]);
            });

        std::ranges::for_each(
            physical_device_indices,
            [&devices, &physical_devices_render_call_info_buffers](auto i) {
                std::ranges::for_each(physical_devices_render_call_info_buffers[i], [device = devices[i]](auto buffer) {vulkan::destroy_buffer(device, buffer); });
            });
        std::ranges::for_each(
            physical_device_indices,
            [&devices, &physical_devices_sphere_buffers](auto i) {
                std::ranges::for_each(physical_devices_sphere_buffers[i], [device = devices[i]](auto& sphere_buffer) { vulkan::destroy_buffer(device, sphere_buffer); });
            });

        std::ranges::for_each(
            physical_device_indices,
            [&devices, &physical_devices_rt_descriptor_pool](auto i) {
                devices[i].destroyDescriptorPool(physical_devices_rt_descriptor_pool[i]);
            }
        );

        std::ranges::for_each(
            physical_device_indices,
            [&devices, &physical_devices_rt_descriptor_set_layout](auto i) {
                devices[i].destroyDescriptorSetLayout(physical_devices_rt_descriptor_set_layout[i]);
            }
        );

        std::ranges::for_each(
            physical_device_indices,
            [&devices, &physical_devices_top_accels, &physical_devices_dynamic_dispatch_loader](auto i) {
                std::ranges::for_each(physical_devices_top_accels[i],
                    [device = devices[i], &dynamicDispatchLoader = physical_devices_dynamic_dispatch_loader[i]](auto top_accel) {
                        vulkan::destroy_acceleration_structure(device, top_accel, dynamicDispatchLoader);
                    });
            });
        std::ranges::for_each(
            physical_device_indices,
            [&devices, &physical_devices_bottom_accels, &physical_devices_dynamic_dispatch_loader](auto i) {
                std::ranges::for_each(physical_devices_bottom_accels[i],
                    [device = devices[i], &dynamicDispatchLoader = physical_devices_dynamic_dispatch_loader[i]](auto bottom_accel) {
                        vulkan::destroy_acceleration_structure(device, bottom_accel, dynamicDispatchLoader);
                    });
            });

        std::ranges::for_each(
            physical_device_indices,
            [&devices, &physical_devices_aabb_buffers](auto i) {
                std::ranges::for_each(physical_devices_aabb_buffers[i],
                    [device = devices[i]](auto& aabb_buffer) {
                        vulkan::destroy_buffer(device, aabb_buffer);
                    });
            });

        std::ranges::for_each(
            physical_device_indices,
            [&physical_devices_next_image_semaphores, &devices](auto i) {
                auto& next_image_semaphores = physical_devices_next_image_semaphores[i];
                auto& device = devices[i];
                std::ranges::for_each(next_image_semaphores, [device](auto semaphore) {device.destroySemaphore(semaphore); });
            });
        std::ranges::for_each(
            physical_device_indices,
            [&physical_devices_render_image_semaphores, &devices](auto i) {
                auto& render_image_semaphores = physical_devices_render_image_semaphores[i];
                auto& device = devices[i];
                std::ranges::for_each(render_image_semaphores, [device](auto semaphore) {device.destroySemaphore(semaphore); });
            });
        std::ranges::for_each(
            physical_device_indices,
            [&physical_devices_fences, &devices](auto i) {
                auto& fences = physical_devices_fences[i];
                auto& device = devices[i];
                std::ranges::for_each(fences, [device](auto fence) {device.destroyFence(fence); });
            });

        std::ranges::for_each(
            physical_device_indices,
            [&devices, &physical_devices_render_target_images](auto i) {
                std::ranges::for_each(physical_devices_render_target_images[i], [device = devices[i]](auto& image) {vulkan::destroy_image(device, image); });
            });
        std::ranges::for_each(
            physical_device_indices,
            [&devices, &physical_devices_summed_images](auto i) {
                std::ranges::for_each(physical_devices_summed_images[i], [device = devices[i]](auto& image) {vulkan::destroy_image(device, image); });
            });

        std::ranges::for_each(
            physical_device_indices,
            [&physical_devices_swapchain_image_views, &devices](auto i) {
                auto& swapchain_image_views = physical_devices_swapchain_image_views[i];
                auto& device = devices[i];
                std::ranges::for_each(swapchain_image_views, [device](auto swapChainImageView) {device.destroyImageView(swapChainImageView); });
            });
        std::ranges::for_each(
            physical_device_indices,
            [&physical_devices_swapchain, &devices](auto i) {
                auto& swapchain = physical_devices_swapchain[i];
                auto& device = devices[i];
                device.destroySwapchainKHR(swapchain);
            });
        std::ranges::for_each(
            physical_device_indices,
            [&devices, &physical_devices_command_pool](auto i) {
                devices[i].destroyCommandPool(physical_devices_command_pool[i]);
            });
        std::ranges::for_each(
            physical_device_indices,
            [&devices](auto i) {
                devices[i].destroy();
            });
        std::ranges::for_each(
            physical_device_indices,
            [&instance, &physical_devices_surface](auto i) {
                auto& surface = physical_devices_surface[i];
                instance.destroySurfaceKHR(surface);
            });
    }

    window::destroy_window(view_window);

    window::destroy_window_system(window_system);

    instance.destroy();
}
