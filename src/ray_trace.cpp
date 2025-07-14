#include "ray_trace.h"

#include <thread>

#include "window.hpp"

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
    auto swapchain = vulkan::create_swapchain(physical_device, surface, device,
        surface_capabilities.minImageCount, format, color_space, present_mode, swapchain_extent, surface_capabilities.currentTransform);

    auto swapchain_images = device.getSwapchainImagesKHR(swapchain);
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

    auto [render_target_image, summed_image] = vulkan::create_images(device, vk::Extent3D{ swapchain_extent.width, swapchain_extent.height, 1 }, memory_properties);

    auto fences = vulkan::create_fences(device, swapchain_images.size());

    auto next_image_semaphores = vulkan::create_semaphores(device, swapchain_images.size() + 1);
    auto render_image_semaphores = vulkan::create_semaphores(device, swapchain_images.size());

    auto scene = generateRandomScene();

    std::vector<vk::AabbPositionsKHR> aabbs(scene.sphereAmount);

    auto aabb_buffer = vulkan::create_aabb_buffer(device, aabbs.size(), memory_properties);

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
        Vulkan vulkan(
            instance, surface,
            physical_device, memory_properties,
            device, compute_queue, present_queue,
            {compute_queue_family, present_queue_family},
            command_pool,
            swapchain,swapchain_images, swapchain_image_views,swapchain_extent,
            render_target_image, summed_image,
            fences,
            next_image_semaphores, render_image_semaphores,
            aabbs, aabb_buffer,
            settings, scene
        );

        // RENDERING

        auto renderBeginTime = std::chrono::steady_clock::now();
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

        vulkan.wait_render_complete();

        auto renderTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - renderBeginTime).count();

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

    vulkan::destroy_buffer(device, aabb_buffer);

    std::ranges::for_each(next_image_semaphores, [device](auto semaphore) {device.destroySemaphore(semaphore); });
    std::ranges::for_each(render_image_semaphores, [device](auto semaphore) {device.destroySemaphore(semaphore); });
    std::ranges::for_each(fences, [device](auto fence) {device.destroyFence(fence); });

    vulkan::destroy_image(device, render_target_image);
    vulkan::destroy_image(device, summed_image);

    std::ranges::for_each(swapchain_image_views, [device](auto swapChainImageView) {device.destroyImageView(swapChainImageView); });
    device.destroySwapchainKHR(swapchain);
    device.destroyCommandPool(command_pool);
    device.destroy();
    instance.destroySurfaceKHR(surface);
    instance.destroy();
}