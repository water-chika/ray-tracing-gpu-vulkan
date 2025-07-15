#pragma once

#include <cstdint>

extern "C"
__declspec(dllexport)
void ray_trace(
    uint32_t samples = 10,
    uint32_t samplesPerRenderCall = 1,
    bool storeRenderResult = false,
    uint32_t width = 1920,
    uint32_t height = 1080
);