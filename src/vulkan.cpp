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

void Vulkan::render(const RenderCallInfo& renderCallInfo) {
    uint32_t swapChainImageIndex = 0;
    auto acquire_image_semaphore = get_acquire_image_semaphore();
    if (auto [result, index] = device.acquireNextImageKHR(swapChain, UINT64_MAX, acquire_image_semaphore);
        result == vk::Result::eSuccess || result == vk::Result::eSuboptimalKHR) {
        swapChainImageIndex = index;
    }
    else {
        throw std::runtime_error{ "failed to acquire next image" };
    }
    free_acquire_image_semaphore(swapChainImageIndex);

    auto fence = get_fence(swapChainImageIndex);
    {
        vk::Result res = device.waitForFences(fence, true, UINT64_MAX);
        if (res != vk::Result::eSuccess) {
            throw std::runtime_error{ "failed to wait fences" };
        }
    }
    device.resetFences(fence);
    updateRenderCallInfoBuffer(renderCallInfo, swapChainImageIndex);

    auto render_image_semaphore = get_render_image_semaphore(swapChainImageIndex);

    auto wait_semaphores = std::array{ acquire_image_semaphore };
    auto  wait_stage_masks =
        std::array<vk::PipelineStageFlags, 1>{ vk::PipelineStageFlagBits::eAllCommands };
    auto signal_semaphores = std::array{ render_image_semaphore };
    auto submitInfo = vk::SubmitInfo{}
        .setCommandBuffers(commandBuffers[swapChainImageIndex])
        .setWaitSemaphores(wait_semaphores)
        .setWaitDstStageMask(wait_stage_masks)
        .setSignalSemaphores(signal_semaphores);

    computeQueue.submit(1, &submitInfo, fence);

    vk::PresentInfoKHR presentInfo = {
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &render_image_semaphore,
            .swapchainCount = 1,
            .pSwapchains = &swapChain,
            .pImageIndices = &swapChainImageIndex
    };

    presentQueue.presentKHR(presentInfo);
}






vk::ImageMemoryBarrier Vulkan::getImagePipelineBarrier(
        const vk::AccessFlags srcAccessFlags, const vk::AccessFlags dstAccessFlags,
        const vk::ImageLayout &oldLayout, const vk::ImageLayout &newLayout,
        const vk::Image &image) const {

    return {
            .srcAccessMask = srcAccessFlags,
            .dstAccessMask = dstAccessFlags,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = computeQueueFamily,
            .dstQueueFamilyIndex = computeQueueFamily,
            .image = image,
            .subresourceRange = {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1
            },
    };
}


void Vulkan::executeSingleTimeCommand(const std::function<void(const vk::CommandBuffer &singleTimeCommandBuffer)> &c) {
    vk::CommandBuffer singleTimeCommandBuffer = device.allocateCommandBuffers(
            {
                    .commandPool = commandPool,
                    .level = vk::CommandBufferLevel::ePrimary,
                    .commandBufferCount = 1
            }).front();

    vk::CommandBufferBeginInfo beginInfo = {
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
    };

    singleTimeCommandBuffer.begin(&beginInfo);

    c(singleTimeCommandBuffer);

    singleTimeCommandBuffer.end();


    vk::SubmitInfo submitInfo = {
            .commandBufferCount = 1,
            .pCommandBuffers = &singleTimeCommandBuffer
    };

    vk::Fence f = device.createFence({});
    computeQueue.submit(1, &submitInfo, f);
    device.waitForFences(1, &f, true, UINT64_MAX);

    device.destroyFence(f);
    device.freeCommandBuffers(commandPool, singleTimeCommandBuffer);
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

    executeSingleTimeCommand(
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

