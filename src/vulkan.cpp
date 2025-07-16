#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "vulkan.h"
#include <iostream>
#include <set>
#include <fstream>
#include <stb/stb_image_write.h>
#include "shader_path.hpp"
#include <algorithm>
#include <numeric>

void Vulkan::wait_render_complete() {
    device.waitIdle();
}

void Vulkan::updateRenderCallInfoBuffer(const RenderCallInfo &renderCallInfo, int index) {
    void* data = device.mapMemory(renderCallInfoBuffers[index].memory, 0, sizeof(RenderCallInfo));
    memcpy(data, &renderCallInfo, sizeof(RenderCallInfo));
    device.unmapMemory(renderCallInfoBuffers[index].memory);
}

void Vulkan::write_to_file(std::filesystem::path path) {
    auto width = settings.windowWidth;
    auto height = settings.windowHeight;
    int component_count = 4;
    size_t pixel_size = component_count * sizeof(float);
    vk::BufferCreateInfo bufferCreateInfo = {
            .size = width*height* pixel_size,
            .usage = vk::BufferUsageFlagBits::eTransferDst,
            .sharingMode = vk::SharingMode::eExclusive
    };

    vk::Buffer buffer = device.createBuffer(bufferCreateInfo);

    vk::MemoryRequirements memoryRequirements = device.getBufferMemoryRequirements(buffer);

    vk::MemoryAllocateFlagsInfo allocateFlagsInfo = {
    };

    vk::MemoryAllocateInfo allocateInfo = {
            .pNext = &allocateFlagsInfo,
            .allocationSize = memoryRequirements.size,
            .memoryTypeIndex = vulkan::findMemoryTypeIndex(
                m_memory_properties,
                    memoryRequirements.memoryTypeBits,
                    vk::MemoryPropertyFlagBits::eHostVisible|
                    vk::MemoryPropertyFlagBits::eHostCoherent)
    };

    vk::DeviceMemory memory = device.allocateMemory(allocateInfo);

    device.bindBufferMemory(buffer, memory, 0);

    vulkan::execute_single_time_command(
        device, computeQueue, commandPool,
        [this, buffer,width,height](vk::CommandBuffer cmd) {
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                vk::PipelineStageFlagBits::eAllCommands,
                {},
                {},
                {},
                vk::ImageMemoryBarrier{}
                .setSrcAccessMask(vk::AccessFlagBits::eMemoryWrite)
                .setSrcQueueFamilyIndex(computeQueueFamily)
                .setDstAccessMask(vk::AccessFlagBits::eMemoryRead)
                .setDstQueueFamilyIndex(computeQueueFamily)
                .setImage(m_summed_images[0].image)
                .setOldLayout(vk::ImageLayout::eTransferSrcOptimal)
                .setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
                .setSubresourceRange(
                    vk::ImageSubresourceRange{}
                    .setAspectMask(vk::ImageAspectFlagBits::eColor)
                    .setLayerCount(1)
                    .setLevelCount(1)
                )
            );
            auto region = vk::BufferImageCopy{}
                .setImageSubresource(
                        vk::ImageSubresourceLayers{}
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                        .setLayerCount(1)
                        )
                .setImageExtent(vk::Extent3D{width,height,1})
            ;
            cmd.copyImageToBuffer(
                m_summed_images[0].image,
                    vk::ImageLayout::eTransferSrcOptimal,
                    buffer,
                    1,
                    &region);
        });

    auto data = reinterpret_cast<const float*>(device.mapMemory(memory, 0, vk::WholeSize));

    stbi_write_hdr(path.string().c_str(), width, height, component_count, data);

    device.freeMemory(memory);
    device.destroyBuffer(buffer);
}

