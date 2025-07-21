#pragma once

#include <memory>

struct RenderCallInfo {
    uint32_t number;
    uint32_t samplesPerRenderCall;
    glm::uvec2 offset;
    glm::uvec2 image_size;
    uint32_t t[2];
    glm::vec4 camera_pos;
    glm::vec4 camera_dir;
};
