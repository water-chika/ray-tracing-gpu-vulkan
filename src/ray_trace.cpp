#include "ray_trace.h"

#include <thread>

extern "C"
__declspec(dllexport)
void ray_trace(
    uint32_t samples,
    uint32_t samplesPerRenderCall,
    bool storeRenderResult,
    uint32_t width,
    uint32_t height
) {


    // SETUP
    VulkanSettings settings = {
        .windowWidth = width,
        .windowHeight = height
    };

    Vulkan vulkan(settings, generateRandomScene());

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

        vulkan.update();
        if (vulkan.shouldExit()) {
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
    while (!vulkan.shouldExit()) {
        vulkan.update();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}