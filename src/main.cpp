#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <charconv>

#include "vulkan.h"

int main(int argc, const char** argv) {
    // COMMAND LINE ARGUMENTS
    uint32_t samples = 10000;
    uint32_t samplesPerRenderCall = 200;

    if (argc >= 2) {
        std::from_chars(argv[1], argv[1] + strlen(argv[1]), samples);
    }

    if (argc >= 3) {
        std::from_chars(argv[2], argv[2] + strlen(argv[2]), samplesPerRenderCall);
    }

    if (samples % samplesPerRenderCall != 0) {
        std::cerr << "'samples' (" << samples << ") has to be a multiple of "
            << "'samples per render call' (" << samplesPerRenderCall << ")" << std::endl;
        exit(1);
    }

    // SETUP
    VulkanSettings settings = { 
        .windowWidth = 1920, 
        .windowHeight = 1080 
    };

    Vulkan vulkan(settings, generateRandomScene());

    // RENDERING
    std::cout << "Rendering started: " << samples << " samples with "
        << samplesPerRenderCall << " samples per render call" << std::endl;

    auto renderBeginTime = std::chrono::steady_clock::now();
    int requiredRenderCalls = samples / samplesPerRenderCall;

    auto previousRenderCallTime = std::chrono::steady_clock::now();

    for (uint32_t number = 1; number <= requiredRenderCalls; number++) {
        RenderCallInfo renderCallInfo = {
            .number = number,
            .samplesPerRenderCall = samplesPerRenderCall,
        };

        std::cout << "Render call " << number << " / " << requiredRenderCalls
            << " (" << (number * samplesPerRenderCall) << " / " << samples
            << " samples)";

        vulkan.render(renderCallInfo);

        auto currentRenderCallTime = std::chrono::steady_clock::now();

        auto renderCallTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                currentRenderCallTime - previousRenderCallTime).count();
        previousRenderCallTime = currentRenderCallTime;
        std::cout << " - Completed in " << renderCallTime << " ms" << std::endl;

        vulkan.update();
    }

    auto renderTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - renderBeginTime).count();
    std::cout << "Rendering completed: " << samples << " samples rendered in "
        << renderTime << " ms" << std::endl << std::endl;

    std::cout << "Write to file:" << "render_result.png" << std::endl;
    vulkan.write_to_file("render_result.png");
    std::cout << "Write completes." << std::endl;

    // WINDOW
    while (!vulkan.shouldExit()) {
        vulkan.update();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}
