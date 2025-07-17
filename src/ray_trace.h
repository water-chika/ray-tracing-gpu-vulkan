#pragma once

#include <cstdint>

extern "C"
#if WIN32
__declspec(dllimport)
#endif
void __stdcall ray_trace(
    uint32_t samples = 10,
    bool storeRenderResult = false,
    uint32_t width = 1920,
    uint32_t height = 1080
);
