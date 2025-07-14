#pragma once

#define VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define VKFW_NO_STRUCT_CONSTRUCTORS

#include <vulkan/vulkan.hpp>

#include <functional>
#include "vulkan_settings.h"
#include "scene.h"
#include "render_call_info.h"

#include <map>
#include <filesystem>
#include <set>
#include <string>
#include <ranges>
#include <numeric>

struct VulkanImage {
    vk::Image image;
    vk::DeviceMemory memory;
    vk::ImageView imageView;
};

struct VulkanBuffer {
    vk::Buffer buffer;
    vk::DeviceMemory memory;
};

struct VulkanAccelerationStructure {
    vk::AccelerationStructureKHR accelerationStructure;
    VulkanBuffer structureBuffer;
    VulkanBuffer scratchBuffer;
    VulkanBuffer instancesBuffer;
};

namespace vulkan {
    vk::Instance create_instance(const auto& extensions) {
        vk::ApplicationInfo applicationInfo = {
        .pApplicationName = "Ray Tracing (Vulkan)",
        .applicationVersion = 1,
        .pEngineName = "Ray Tracing (Vulkan)",
        .engineVersion = 1,
        .apiVersion = VK_API_VERSION_1_3
        };



        std::vector<const char*> enabledLayers = { };

        vk::InstanceCreateInfo instanceCreateInfo = {
                .pApplicationInfo = &applicationInfo,
                .enabledLayerCount = static_cast<uint32_t>(enabledLayers.size()),
                .ppEnabledLayerNames = enabledLayers.data(),
                .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
                .ppEnabledExtensionNames = extensions.data(),
        };

#ifndef _DEBUG
        instanceCreateInfo.pNext = nullptr;
#endif

        return vk::createInstance(instanceCreateInfo);
    }

    vk::PhysicalDevice pick_physical_device(vk::Instance instance, const auto& extensions) {
        std::vector<vk::PhysicalDevice> allPhysicalDevices = instance.enumeratePhysicalDevices();

        if (allPhysicalDevices.empty()) {
            throw std::runtime_error("No GPU with Vulkan support found!");
        }

        std::vector<vk::PhysicalDevice> withRequiredExtensionsPhysicalDevices{};
        for (const vk::PhysicalDevice& d : allPhysicalDevices) {
            std::vector<vk::ExtensionProperties> availableExtensions = d.enumerateDeviceExtensionProperties();
            std::set<std::string> requiredExtensions(extensions.begin(), extensions.end());

            for (const vk::ExtensionProperties& extension : availableExtensions) {
                requiredExtensions.erase(extension.extensionName);
            }

            if (requiredExtensions.empty()) {
                withRequiredExtensionsPhysicalDevices.push_back(d);
            }
        }

        for (const vk::PhysicalDevice& d : withRequiredExtensionsPhysicalDevices) {
            if (d.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                return d;
            }
        }

        if (withRequiredExtensionsPhysicalDevices.size() > 0) {
            return withRequiredExtensionsPhysicalDevices[0];
        }

        throw std::runtime_error("No GPU supporting all required features found!");
    }

    inline auto find_queue_family(vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface) {
        uint32_t computeQueueFamily = 0, presentQueueFamily = 0;

        std::vector<vk::QueueFamilyProperties> queueFamilies = physicalDevice.getQueueFamilyProperties();

        bool computeFamilyFound = false;
        bool presentFamilyFound = false;

        for (uint32_t i = 0; i < queueFamilies.size(); i++) {
            bool supportsGraphics = (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics)
                == vk::QueueFlagBits::eGraphics;
            bool supportsCompute = (queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute)
                == vk::QueueFlagBits::eCompute;
            bool supportsPresenting = physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), surface);

            if (supportsCompute && !supportsGraphics && !computeFamilyFound) {
                computeQueueFamily = i;
                computeFamilyFound = true;
                continue;
            }

            if (supportsPresenting && !presentFamilyFound) {
                presentQueueFamily = i;
                presentFamilyFound = true;
            }

            if (computeFamilyFound && presentFamilyFound)
                break;
        }
        if (!computeFamilyFound || !presentFamilyFound) {
            throw std::runtime_error{ "can not find compute queue or present queue" };
        }

        return std::pair{ computeQueueFamily, presentQueueFamily };
    }


    auto create_device(
        vk::Instance instance,
        vk::PhysicalDevice physical_device,
        uint32_t computeQueueFamily, uint32_t presentQueueFamily,
        const auto& extensions
        ) {
        float queuePriority = 1.0f;
        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos = {
                {
                        .queueFamilyIndex = presentQueueFamily,
                        .queueCount = 1,
                        .pQueuePriorities = &queuePriority
                },
                {
                        .queueFamilyIndex = computeQueueFamily,
                        .queueCount = 1,
                        .pQueuePriorities = &queuePriority
                }
        };

        vk::PhysicalDeviceFeatures deviceFeatures = {
            .shaderFloat64 = true,
        };

        vk::PhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures = {
                .bufferDeviceAddress = true,
                .bufferDeviceAddressCaptureReplay = false,
                .bufferDeviceAddressMultiDevice = false
        };

        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures = {
                .pNext = &bufferDeviceAddressFeatures,
                .rayTracingPipeline = true
        };

        vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures = {
                .pNext = &rayTracingPipelineFeatures,
                .accelerationStructure = true,
                .accelerationStructureCaptureReplay = true,
                .accelerationStructureIndirectBuild = false,
                .accelerationStructureHostCommands = false,
                .descriptorBindingAccelerationStructureUpdateAfterBind = false
        };

        vk::DeviceCreateInfo deviceCreateInfo = {
                .pNext = &accelerationStructureFeatures,
                .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
                .pQueueCreateInfos = queueCreateInfos.data(),
                .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
                .ppEnabledExtensionNames = extensions.data(),
                .pEnabledFeatures = &deviceFeatures
        };

        auto device = physical_device.createDevice(deviceCreateInfo);

        auto computeQueue = device.getQueue(computeQueueFamily, 0);
        auto presentQueue = device.getQueue(presentQueueFamily, 0);

        return std::tuple{ device, computeQueue, presentQueue };
    }


    inline auto create_swapchain(
        vk::PhysicalDevice physicalDevice,
        vk::SurfaceKHR surface,
        vk::Device device,
        uint32_t min_image_count,
        vk::Format format,
        vk::ColorSpaceKHR color_space,
        vk::PresentModeKHR presentMode,
        vk::Extent2D swapchain_extent,
        vk::SurfaceTransformFlagBitsKHR pre_transform
    ) {
        auto present_modes = physicalDevice.getSurfacePresentModesKHR(surface);
        if (!std::ranges::contains(present_modes, presentMode)) {
            presentMode = present_modes[0];
        }

        vk::SwapchainCreateInfoKHR swapChainCreateInfo = {
                .surface = surface,
                .minImageCount = min_image_count,
                .imageFormat = format,
                .imageColorSpace = color_space,
                .imageExtent = swapchain_extent,
                .imageArrayLayers = 1,
                .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst,
                .imageSharingMode = vk::SharingMode::eExclusive,
                .preTransform = pre_transform,
                .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
                .presentMode = presentMode,
                .clipped = true,
                .oldSwapchain = nullptr
        };

        auto swapchain = device.createSwapchainKHR(swapChainCreateInfo);
        return swapchain;
    }

    inline uint32_t findMemoryTypeIndex(const vk::PhysicalDeviceMemoryProperties& memory_properties, const uint32_t& memoryTypeBits, const vk::MemoryPropertyFlags& properties) {
        for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
            if ((memoryTypeBits & (1 << i)) && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("Unable to find suitable memory type!");
    }

    inline VulkanImage create_image(vk::Device device, vk::Extent3D extent, const vk::Format format, const vk::Flags<vk::ImageUsageFlagBits> usageFlagBits,
        vk::PhysicalDeviceMemoryProperties memory_properties) {
        vk::ImageCreateInfo imageCreateInfo = {
                .imageType = vk::ImageType::e2D,
                .format = format,
                .extent = extent,
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = vk::SampleCountFlagBits::e1,
                .tiling = vk::ImageTiling::eOptimal,
                .usage = usageFlagBits,
                .sharingMode = vk::SharingMode::eExclusive,
                .initialLayout = vk::ImageLayout::eUndefined
        };

        vk::Image image = device.createImage(imageCreateInfo);

        vk::MemoryRequirements memoryRequirements = device.getImageMemoryRequirements(image);

        vk::MemoryAllocateInfo allocateInfo = {
                .allocationSize = memoryRequirements.size,
                .memoryTypeIndex = findMemoryTypeIndex(
                    memory_properties,
                    memoryRequirements.memoryTypeBits,
                    vk::MemoryPropertyFlagBits::eDeviceLocal)
        };

        vk::DeviceMemory memory = device.allocateMemory(allocateInfo);

        device.bindImageMemory(image, memory, 0);

        return {
                .image = image,
                .memory = memory,
                .imageView = device.createImageView(
                    {
                            .image = image,
                            .viewType = vk::ImageViewType::e2D,
                            .format = format,
                            .subresourceRange = {
                                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                                    .baseMipLevel = 0,
                                    .levelCount = 1,
                                    .baseArrayLayer = 0,
                                    .layerCount = 1
                            }
                    })
        };
    }

    inline auto create_images(vk::Device device, vk::Extent3D extent, vk::PhysicalDeviceMemoryProperties memory_properties) {
        auto renderTargetImage = create_image(
            device,
            extent,
            vk::Format::eR8G8B8A8Unorm,
            vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc, memory_properties);

        const vk::Format summedPixelColorImageFormat = vk::Format::eR32G32B32A32Sfloat;
        auto summedPixelColorImage = create_image(device, extent, summedPixelColorImageFormat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst, memory_properties);

        return std::tuple{ renderTargetImage, summedPixelColorImage };
    }

    inline void destroy_image(vk::Device device, const VulkanImage& image) {
        device.destroyImageView(image.imageView);
        device.destroyImage(image.image);
        device.freeMemory(image.memory);
    }

    inline auto create_fences(vk::Device device, uint32_t count) {
        std::vector<vk::Fence> fences(count);
        std::ranges::generate(
            fences,
            [device]() {
                return device.createFence(vk::FenceCreateInfo{}.setFlags(vk::FenceCreateFlagBits::eSignaled));
            }
        );
        return fences;
    }

    inline auto create_semaphores(vk::Device device, uint32_t count) {
        std::vector<vk::Semaphore> semaphores(count);
        std::ranges::generate(
            semaphores,
            [device]() {
                return device.createSemaphore(vk::SemaphoreCreateInfo{});
            }
        );
        return semaphores;
    }

    inline VulkanBuffer create_buffer(vk::Device device, const vk::DeviceSize& size, const vk::Flags<vk::BufferUsageFlagBits>& usage,
        const vk::Flags<vk::MemoryPropertyFlagBits>& memoryProperty, const vk::PhysicalDeviceMemoryProperties& memory_properties) {
        vk::BufferCreateInfo bufferCreateInfo = {
                .size = size,
                .usage = usage,
                .sharingMode = vk::SharingMode::eExclusive
        };

        vk::Buffer buffer = device.createBuffer(bufferCreateInfo);

        vk::MemoryRequirements memoryRequirements = device.getBufferMemoryRequirements(buffer);

        vk::MemoryAllocateFlagsInfo allocateFlagsInfo = {
                .flags = vk::MemoryAllocateFlagBits::eDeviceAddress
        };

        vk::MemoryAllocateInfo allocateInfo = {
                .pNext = &allocateFlagsInfo,
                .allocationSize = memoryRequirements.size,
                .memoryTypeIndex = vulkan::findMemoryTypeIndex(memory_properties, memoryRequirements.memoryTypeBits, memoryProperty)
        };

        vk::DeviceMemory memory = device.allocateMemory(allocateInfo);

        device.bindBufferMemory(buffer, memory, 0);

        return {
                .buffer = buffer,
                .memory = memory,
        };
    }

    inline void destroy_buffer(vk::Device device, const VulkanBuffer& buffer) {
        device.destroyBuffer(buffer.buffer);
        device.freeMemory(buffer.memory);
    }

    inline auto create_aabb_buffer(vk::Device device, uint32_t count, const vk::PhysicalDeviceMemoryProperties& memory_properties) {
        const vk::DeviceSize bufferSize = sizeof(vk::AabbPositionsKHR) * count;

        auto aabbBuffer = create_buffer(device, bufferSize,
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent |
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            memory_properties);
        return aabbBuffer;
    }

    inline auto createBottomAccelerationStructure(vk::Device device, VulkanBuffer& aabbBuffer, uint32_t max_primitive_count,
        vk::AccelerationStructureGeometryKHR& geometry,
        const vk::PhysicalDeviceMemoryProperties& memory_properties, vk::detail::DispatchLoaderDynamic& dynamicDispatchLoader) {

        geometry.geometry.aabbs.sType = vk::StructureType::eAccelerationStructureGeometryAabbsDataKHR;
        geometry.geometry.aabbs.stride = sizeof(vk::AabbPositionsKHR);
        geometry.geometry.aabbs.data.deviceAddress = device.getBufferAddress({ .buffer = aabbBuffer.buffer });


        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo = {
                .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
                .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
                .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
                .srcAccelerationStructure = nullptr,
                .dstAccelerationStructure = nullptr,
                .geometryCount = 1,
                .pGeometries = &geometry,
                .scratchData = {}
        };


        // CALCULATE REQUIRED SIZE FOR THE ACCELERATION STRUCTURE
        std::vector<uint32_t> maxPrimitiveCounts = { max_primitive_count };

        vk::AccelerationStructureBuildSizesInfoKHR buildSizesInfo = device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo, maxPrimitiveCounts, dynamicDispatchLoader);


        // ALLOCATE BUFFERS FOR ACCELERATION STRUCTURE
        VulkanAccelerationStructure bottomAccelerationStructure{};

        bottomAccelerationStructure.structureBuffer = vulkan::create_buffer(device, buildSizesInfo.accelerationStructureSize,
            vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eDeviceLocal, memory_properties);

        bottomAccelerationStructure.scratchBuffer = vulkan::create_buffer(device, buildSizesInfo.buildScratchSize,
            vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eDeviceLocal, memory_properties);

        // CREATE THE ACCELERATION STRUCTURE
        vk::AccelerationStructureCreateInfoKHR createInfo = {
                .buffer = bottomAccelerationStructure.structureBuffer.buffer,
                .offset = 0,
                .size = buildSizesInfo.accelerationStructureSize,
                .type = vk::AccelerationStructureTypeKHR::eBottomLevel
        };

        bottomAccelerationStructure.accelerationStructure =
            device.createAccelerationStructureKHR(createInfo, nullptr, dynamicDispatchLoader);


        // FILL IN THE REMAINING META INFO
        buildInfo.dstAccelerationStructure = bottomAccelerationStructure.accelerationStructure;
        buildInfo.scratchData.deviceAddress =
            device.getBufferAddress({ .buffer = bottomAccelerationStructure.scratchBuffer.buffer });
        return std::tuple{ bottomAccelerationStructure, buildInfo};
    }

    inline void destroy_acceleration_structure(vk::Device device, const VulkanAccelerationStructure& accelerationStructure, vk::detail::DispatchLoaderDynamic& dynamicDispatchLoader) {
        device.destroyAccelerationStructureKHR(accelerationStructure.accelerationStructure, nullptr, dynamicDispatchLoader);
        vulkan::destroy_buffer(device, accelerationStructure.structureBuffer);
        vulkan::destroy_buffer(device, accelerationStructure.scratchBuffer);
        vulkan::destroy_buffer(device, accelerationStructure.instancesBuffer);
    }
}


class Vulkan {
public:
    Vulkan(
        vk::Instance instance, vk::SurfaceKHR surface,
        vk::PhysicalDevice physical_device, vk::PhysicalDeviceMemoryProperties memory_properties,
        vk::Device device, vk::Queue compute_queue, vk::Queue present_queue,
        std::pair<uint32_t, uint32_t> compute_present_queue_families,
        vk::CommandPool command_pool,
        vk::SwapchainKHR swapchain, auto swapchain_images, auto swapchain_image_views,
        vk::Extent2D swapchain_extent,
        VulkanImage render_target_image, VulkanImage summed_image,
        auto fences,
        auto next_image_semaphores, auto render_image_semaphores,
        auto& aabbs, VulkanBuffer& aabb_buffer,
        VulkanAccelerationStructure bottom_accel, vk::AccelerationStructureBuildGeometryInfoKHR bottom_accel_build_info,
        VulkanSettings settings, Scene scene) :
        instance{instance}, surface{surface},
        physicalDevice{ physical_device }, m_memory_properties{memory_properties},
        device{ device }, computeQueue{compute_queue}, presentQueue{present_queue},
        computeQueueFamily{ compute_present_queue_families.first }, presentQueueFamily{ compute_present_queue_families.second },
        commandPool{command_pool},
        swapChain{ swapchain }, swapChainImages{ swapchain_images }, swapChainImageViews{ swapchain_image_views }, swapChainExtent{ swapchain_extent },
        renderTargetImage{render_target_image}, summedPixelColorImage{summed_image},
        m_fences{fences},
        m_next_image_semaphores{ next_image_semaphores }, m_render_image_semaphores{ render_image_semaphores },
        aabbs{aabbs}, aabbBuffer{ aabb_buffer },
        bottomAccelerationStructure{bottom_accel},
        settings(settings), scene(scene),
        m_width(settings.windowWidth), m_height(settings.windowHeight)
    {


        dynamicDispatchLoader = vk::detail::DispatchLoaderDynamic(instance, vkGetInstanceProcAddr, device);

        executeSingleTimeCommand(
            [this](vk::CommandBuffer commandBuffer) {
                commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                    vk::PipelineStageFlagBits::eTransfer,
                    vk::DependencyFlagBits::eByRegion,
                    {}, {},
                    getImagePipelineBarrier(
                        vk::AccessFlagBits::eNoneKHR, vk::AccessFlagBits::eTransferWrite,
                        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, summedPixelColorImage.image));
                commandBuffer.clearColorImage(summedPixelColorImage.image, vk::ImageLayout::eTransferDstOptimal,
                    vk::ClearColorValue{},
                    vk::ImageSubresourceRange{}
                    .setAspectMask(vk::ImageAspectFlagBits::eColor)
                    .setLayerCount(1)
                    .setLevelCount(1));
                commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                    vk::PipelineStageFlagBits::eRayTracingShaderKHR,
                    vk::DependencyFlagBits::eByRegion,
                    {}, {},
                    getImagePipelineBarrier(
                        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral, summedPixelColorImage.image));
            }
        );

        auto aabbs_buffer_size = sizeof(aabbs[0]) * aabbs.size();
        void* data = device.mapMemory(aabbBuffer.memory, 0, aabbs_buffer_size);
        memcpy(data, aabbs.data(), aabbs_buffer_size);
        device.unmapMemory(aabbBuffer.memory);


        // BUILD THE ACCELERATION STRUCTURE
        vk::AccelerationStructureBuildRangeInfoKHR buildRangeInfo = {
                .primitiveCount = static_cast<uint32_t>(aabbs.size()),
                .primitiveOffset = 0,
                .firstVertex = 0,
                .transformOffset = 0
        };

        const vk::AccelerationStructureBuildRangeInfoKHR* pBuildRangeInfos[] = { &buildRangeInfo };

        executeSingleTimeCommand([&](const vk::CommandBuffer& singleTimeCommandBuffer) {
            singleTimeCommandBuffer.buildAccelerationStructuresKHR(1, &bottom_accel_build_info, pBuildRangeInfos, dynamicDispatchLoader);
            });

        createTopAccelerationStructure();

        createSphereBuffer();
        createRenderCallInfoBuffer();

        createDescriptorSetLayout();
        createDescriptorPool();
        createDescriptorSet();
        createPipelineLayout();
        createRTPipeline();

        createShaderBindingTable();
        createCommandBuffer();

        m_next_image_semaphores_indices.resize(swapChainImages.size());
        std::iota(m_next_image_semaphores_indices.begin(), m_next_image_semaphores_indices.end(), 0);
        m_next_image_free_semaphore_index = swapChainImages.size();
    }

    ~Vulkan();

    static auto get_required_instance_extensions() {
        const std::vector<const char*> requiredInstanceExtensions = {
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
        };
        return requiredInstanceExtensions;
    }

    static auto get_required_device_extensions() {
        const std::vector<const char*> requiredDeviceExtensions = {
                VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
                VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
                VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
                VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
                VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
                VK_KHR_MAINTENANCE3_EXTENSION_NAME
        };
        return requiredDeviceExtensions;
    }

    void render(const RenderCallInfo &renderCallInfo);

    void wait_render_complete();

    void write_to_file(std::filesystem::path path);

private:
    VulkanSettings settings;
    Scene scene;
    std::vector<vk::AabbPositionsKHR> aabbs;

    vk::Instance instance;
    vk::SurfaceKHR surface;
    vk::PhysicalDevice physicalDevice;
    vk::PhysicalDeviceMemoryProperties m_memory_properties;
    vk::Device device;

    vk::detail::DispatchLoaderDynamic dynamicDispatchLoader;

    uint32_t presentQueueFamily = 0, computeQueueFamily = 0;
    vk::Queue presentQueue, computeQueue;

    vk::CommandPool commandPool;

    vk::SwapchainKHR swapChain;
    vk::Extent2D swapChainExtent;
    std::vector<vk::Image> swapChainImages;
    std::vector<vk::ImageView> swapChainImageViews;

    vk::DescriptorSetLayout rtDescriptorSetLayout;
    vk::DescriptorPool rtDescriptorPool;
    std::vector<vk::DescriptorSet> rtDescriptorSets;
    vk::PipelineLayout rtPipelineLayout;
    vk::Pipeline rtPipeline;

    std::vector<vk::CommandBuffer> commandBuffers;
    std::vector<vk::CommandBuffer> commandBuffersForNoPresent;

    std::vector<vk::Fence> m_fences;
    auto get_fence(uint32_t image_index) {
        return m_fences[image_index];
    }

    // There is swapchain image count + 1 semaphores
    // So that we could get a free semaphore.
    std::vector<vk::Semaphore> m_next_image_semaphores;
    uint32_t m_next_image_free_semaphore_index;
    std::vector<uint32_t> m_next_image_semaphores_indices;
    auto get_acquire_image_semaphore() {
        return m_next_image_semaphores[m_next_image_free_semaphore_index];
    }
    auto free_acquire_image_semaphore(uint32_t image_index) {
        std::swap(m_next_image_free_semaphore_index, m_next_image_semaphores_indices[image_index]);
    }

    // Semaphore for swapchain present.
    std::vector<vk::Semaphore> m_render_image_semaphores;
    auto get_render_image_semaphore(uint32_t image_index) {
        return m_render_image_semaphores[image_index];
    }

    VulkanImage renderTargetImage;
    VulkanImage summedPixelColorImage;

    VulkanBuffer aabbBuffer;

    VulkanAccelerationStructure bottomAccelerationStructure;
    VulkanAccelerationStructure topAccelerationStructure;

    VulkanBuffer shaderBindingTableBuffer;
    vk::StridedDeviceAddressRegionKHR sbtRayGenAddressRegion, sbtHitAddressRegion, sbtMissAddressRegion;

    VulkanBuffer sphereBuffer;
    std::vector<VulkanBuffer> renderCallInfoBuffers;

    int m_width;
    int m_height;

    void createDescriptorSetLayout();

    void createDescriptorPool();

    void createDescriptorSet();

    void createPipelineLayout();

    void createRTPipeline();

    [[nodiscard]] static std::vector<char> readBinaryFile(const std::string &path);

    void record_ray_tracing(vk::CommandBuffer cmd, int index);
    void createCommandBuffer();

    [[nodiscard]] vk::ImageMemoryBarrier getImagePipelineBarrier(
            const vk::AccessFlags srcAccessFlags, const vk::AccessFlags dstAccessFlags,
            const vk::ImageLayout &oldLayout, const vk::ImageLayout &newLayout, const vk::Image &image) const;

    void executeSingleTimeCommand(const std::function<void(const vk::CommandBuffer &singleTimeCommandBuffer)> &c);

    void createTopAccelerationStructure();

    void destroyAccelerationStructure(const VulkanAccelerationStructure &accelerationStructure);

    [[nodiscard]] vk::ShaderModule createShaderModule(const std::string &path) const;

    void createShaderBindingTable();

    [[nodiscard]] vk::PhysicalDeviceRayTracingPipelinePropertiesKHR getRayTracingProperties() const;

    void createSphereBuffer();

    void createRenderCallInfoBuffer();

    void updateRenderCallInfoBuffer(const RenderCallInfo &renderCallInfo, int index);

};
