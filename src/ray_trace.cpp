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


    Vulkan vulkan(instance, surface, settings, generateRandomScene()
        );

    // RENDERING
    std::cout << "Rendering started: " << samples << " samples with "
        << samplesPerRenderCall << " samples per render call" << std::endl;

    auto renderBeginTime = std::chrono::steady_clock::now();
    int requiredRenderCalls = samples / samplesPerRenderCall;

    uint32_t number = 0;
    while (number < requiredRenderCalls) {
        ++number;
        RenderCallInfo renderCallInfo = {
            .number = number,
            .samplesPerRenderCall = samplesPerRenderCall,
        };

        std::cout << "Render call " << number << " / " << requiredRenderCalls
            << " (" << (number * samplesPerRenderCall) << " / " << samples
            << " samples)" << std::endl;

        vulkan.render(renderCallInfo);

        view_window.poll_events();
        if (view_window.should_close()) {
            break;
        }
    }

    vulkan.wait_render_complete();

    auto renderTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - renderBeginTime).count();
    std::cout << "Rendering completed: " << number * samplesPerRenderCall << " samples rendered in "
        << renderTime << " ms" << std::endl << std::endl;

    if (storeRenderResult) {
        std::cout << "Write to file:" << "render_result.png" << std::endl;
        vulkan.write_to_file("render_result.hdr");
        std::cout << "Write completes." << std::endl;
    }

    // WINDOW
    while (!view_window.should_close()) {
        view_window.poll_events();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    instance.destroySurfaceKHR(surface);
    instance.destroy();
}