#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <charconv>

#include "ray_trace.h"

int main(int argc, const char** argv) {
    using namespace std::literals;
    // COMMAND LINE ARGUMENTS
    uint32_t samples = 10;
    uint32_t samplesPerRenderCall = 1;
    bool storeRenderResult = false;
    uint32_t width = 1920;
    uint32_t height = 1080;

    if (argc >= 2) {
        std::from_chars(argv[1], argv[1] + strlen(argv[1]), samples);
    }

    if (argc >= 3) {
        std::from_chars(argv[2], argv[2] + strlen(argv[2]), samplesPerRenderCall);
    }
    for (int i = 1; i < argc; i++) {
        if (argv[i] == "--help"s) {
            std::cout << "--help                            # Show this help infomation" << std::endl;
            std::cout << "--store                           # Store rendered image to file" << std::endl;
            std::cout << "--samples <count>                 # Total samples to render" << std::endl;
            std::cout << "--samples_per_render_call <count> # Samples every render call will render" << std::endl;
            std::cout << "--width <width>                   # Image width" << std::endl;
            std::cout << "--height <height>                 # Image height" << std::endl;
            exit(0);
        }
        else if (argv[i] == "--store"s) {
            storeRenderResult = true;
        }
        else if (argv[i] == "--samples"s) {
            std::from_chars(argv[i + 1], argv[i + 1] + strlen(argv[i + 1]), samples);
            ++i;
        }
        else if (argv[i] == "--samples_per_render_call"s) {
            std::from_chars(argv[i + 1], argv[i + 1] + strlen(argv[i + 1]), samplesPerRenderCall);
            ++i;
        }
        else if (argv[i] == "--width"s) {
            std::from_chars(argv[i + 1], argv[i + 1] + strlen(argv[i + 1]), width);
            ++i;
        }
        else if (argv[i] == "--height"s) {
            std::from_chars(argv[i + 1], argv[i + 1] + strlen(argv[i + 1]), height);
            ++i;
        }
        else {
            std::cerr << "unknown argument: " << argv[i] << std::endl;
        }
    }

    if (samples % samplesPerRenderCall != 0) {
        std::cerr << "'samples' (" << samples << ") has to be a multiple of "
            << "'samples per render call' (" << samplesPerRenderCall << ")" << std::endl;
        exit(1);
    }
    try {
        ray_trace(
            samples,
            samplesPerRenderCall,
            storeRenderResult,
            width,
            height);
    }
    catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}
