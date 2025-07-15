#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "vulkan.h"
#include <iostream>
#include <set>
#include <fstream>
#include <stb/stb_image_write.h>
#include "shader_path.hpp"
#include <algorithm>
#include <numeric>

Vulkan::~Vulkan() {
    vulkan::destroy_buffer(device, shaderBindingTableBuffer);
}


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



void Vulkan::record_ray_tracing(vk::CommandBuffer commandBuffer, int index) {
    // RENDER TARGET IMAGE UNDEFINED -> GENERAL
    // Sync summed pixel color image with previous ray tracing.
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eRayTracingShaderKHR,
        vk::PipelineStageFlagBits::eRayTracingShaderKHR,
        vk::DependencyFlagBits::eByRegion, {}, {},
        std::array{
            getImagePipelineBarrier(
                vk::AccessFlagBits::eNoneKHR, vk::AccessFlagBits::eShaderWrite,
                vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, renderTargetImage.image),
            getImagePipelineBarrier(
                vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
                vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
                vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral, summedPixelColorImage.image)
        });

    // RAY TRACING
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, rtPipeline);

    std::vector<vk::DescriptorSet> descriptorSets = { rtDescriptorSets[index] };
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, rtPipelineLayout,
        0, descriptorSets, nullptr);

    commandBuffer.traceRaysKHR(sbtRayGenAddressRegion, sbtMissAddressRegion, sbtHitAddressRegion, {},
        settings.windowWidth, settings.windowHeight, 1, dynamicDispatchLoader);
}

void Vulkan::createCommandBuffer() {
    commandBuffers.resize(swapChainImages.size());
    for (int swapChainImageIndex = 0; swapChainImageIndex < swapChainImages.size(); swapChainImageIndex++) {
        auto& commandBuffer = commandBuffers[swapChainImageIndex];
        auto& swapChainImage = swapChainImages[swapChainImageIndex];
        commandBuffer = device.allocateCommandBuffers(
            {
                    .commandPool = commandPool,
                    .level = vk::CommandBufferLevel::ePrimary,
                    .commandBufferCount = 1
            }).front();

        vk::CommandBufferBeginInfo beginInfo = {};
        commandBuffer.begin(&beginInfo);

        record_ray_tracing(commandBuffer, swapChainImageIndex);

        // RENDER TARGET IMAGE: GENERAL -> TRANSFER SRC & SWAP CHAIN IMAGE: UNDEFINED -> TRANSFER DST
        vk::ImageMemoryBarrier imageBarriersToTransfer[2] = {
                getImagePipelineBarrier(
                        vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead,
                        vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal, renderTargetImage.image),
                getImagePipelineBarrier(
                        vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eTransferWrite,
                        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, swapChainImage)
        };

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
            vk::DependencyFlagBits::eByRegion, 0, nullptr,
            0, nullptr, 2, imageBarriersToTransfer);


        // COPY RENDER TARGET IMAGE TO SWAP CHAIN IMAGE
        vk::ImageSubresourceLayers subresourceLayers = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1
        };

        vk::ImageCopy imageCopy = {
                .srcSubresource = subresourceLayers,
                .srcOffset = {0, 0, 0},
                .dstSubresource = subresourceLayers,
                .dstOffset = {0, 0, 0},
                .extent = {
                        .width = swapChainExtent.width,
                        .height = swapChainExtent.height,
                        .depth = 1
                }
        };

        commandBuffer.copyImage(renderTargetImage.image, vk::ImageLayout::eTransferSrcOptimal, swapChainImage,
            vk::ImageLayout::eTransferDstOptimal, 1, &imageCopy);


        // SWAP CHAIN IMAGE: TRANSFER DST -> PRESENT
        vk::ImageMemoryBarrier barrierSwapChainToPresent = getImagePipelineBarrier(
            vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eMemoryRead,
            vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR, swapChainImage);

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
            vk::DependencyFlagBits::eByRegion, 0, nullptr,
            0, nullptr, 1, &barrierSwapChainToPresent);

        commandBuffer.end();
    }
    commandBuffersForNoPresent.resize(2);
    std::ranges::for_each(
        commandBuffersForNoPresent,
        [this](auto& commandBuffer) {
            commandBuffer = device.allocateCommandBuffers(
                {
                        .commandPool = commandPool,
                        .level = vk::CommandBufferLevel::ePrimary,
                        .commandBufferCount = 1
                }).front();

            vk::CommandBufferBeginInfo beginInfo = {};
            commandBuffer.begin(&beginInfo);

            record_ray_tracing(commandBuffer, 0);

            commandBuffer.end();

        }
    );
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




void Vulkan::destroyAccelerationStructure(const VulkanAccelerationStructure &accelerationStructure) {
    device.destroyAccelerationStructureKHR(accelerationStructure.accelerationStructure, nullptr, dynamicDispatchLoader);
    vulkan::destroy_buffer(device, accelerationStructure.structureBuffer);
    vulkan::destroy_buffer(device, accelerationStructure.scratchBuffer);
    vulkan::destroy_buffer(device, accelerationStructure.instancesBuffer);
}



void Vulkan::createShaderBindingTable() {
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties = getRayTracingProperties();
    uint32_t baseAlignment = rayTracingProperties.shaderGroupBaseAlignment;
    uint32_t handleSize = rayTracingProperties.shaderGroupHandleSize;


    const uint32_t shaderGroupCount = 3;
    vk::DeviceSize sbtBufferSize = baseAlignment * shaderGroupCount;

    shaderBindingTableBuffer = vulkan::create_buffer(device, sbtBufferSize,
                                            vk::BufferUsageFlagBits::eShaderBindingTableKHR |
                                            vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                            vk::MemoryPropertyFlagBits::eHostVisible |
                                            vk::MemoryPropertyFlagBits::eHostCoherent |
                                            vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_memory_properties);


    std::vector<uint8_t> handles = device.getRayTracingShaderGroupHandlesKHR<uint8_t>(
            rtPipeline, 0, shaderGroupCount, shaderGroupCount * handleSize, dynamicDispatchLoader);

    vk::DeviceAddress sbtAddress = device.getBufferAddress({.buffer = shaderBindingTableBuffer.buffer});

    vk::StridedDeviceAddressRegionKHR addressRegion = {
            .stride = baseAlignment,
            .size = handleSize
    };

    sbtRayGenAddressRegion = addressRegion;
    sbtRayGenAddressRegion.size = baseAlignment;
    sbtRayGenAddressRegion.deviceAddress = sbtAddress;

    sbtMissAddressRegion = addressRegion;
    sbtMissAddressRegion.deviceAddress = sbtAddress + baseAlignment;

    sbtHitAddressRegion = addressRegion;
    sbtHitAddressRegion.deviceAddress = sbtAddress + baseAlignment * 2;

    uint8_t* sbtBufferData = static_cast<uint8_t*>(device.mapMemory(shaderBindingTableBuffer.memory, 0, sbtBufferSize));

    memcpy(sbtBufferData, handles.data(), handleSize);
    memcpy(sbtBufferData + baseAlignment, handles.data() + handleSize, handleSize);
    memcpy(sbtBufferData + baseAlignment * 2, handles.data() + handleSize * 2, handleSize);

    device.unmapMemory(shaderBindingTableBuffer.memory);
}

vk::PhysicalDeviceRayTracingPipelinePropertiesKHR Vulkan::getRayTracingProperties() const {
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelinePropertiesKhr = {};

    vk::PhysicalDeviceProperties2 physicalDeviceProperties2 = {
            .pNext = &rayTracingPipelinePropertiesKhr
    };

    physicalDevice.getProperties2(&physicalDeviceProperties2);

    return rayTracingPipelinePropertiesKhr;
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
                .setImage(summedPixelColorImage.image)
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
                summedPixelColorImage.image,
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

