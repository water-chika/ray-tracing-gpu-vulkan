#pragma once

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
#include <fstream>
#include <unordered_map>

#include "shader_path.hpp"

#include "vulkan.hpp"

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

    auto pick_physical_devices(vk::Instance instance, const auto& extensions) {
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

        std::unordered_map<uint32_t, vk::PhysicalDevice> id_device_map{};

        std::ranges::for_each(
            withRequiredExtensionsPhysicalDevices,
            [&id_device_map](vk::PhysicalDevice physical_device) {
                auto properties = physical_device.getProperties();
                id_device_map[properties.deviceID] = physical_device;
            }
        );

        auto unique_physical_devices = std::vector<vk::PhysicalDevice>(id_device_map.size());

        std::ranges::transform(
            id_device_map,
            unique_physical_devices.begin(),
            [](auto k_v) {
                return k_v.second;
            }
        );

        return unique_physical_devices;
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


    inline auto createTopAccelerationStructure(vk::Device device,
        vk::AccelerationStructureKHR bottom_accel,
        vk::AccelerationStructureGeometryKHR& geometry,
        const vk::PhysicalDeviceMemoryProperties& memory_properties,
        vk::detail::DispatchLoaderDynamic& dynamicDispatchLoader) {

        geometry.geometry.instances.sType = vk::StructureType::eAccelerationStructureGeometryInstancesDataKHR;
        geometry.geometry.instances.arrayOfPointers = false;


        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo = {
                .type = vk::AccelerationStructureTypeKHR::eTopLevel,
                .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
                .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
                .srcAccelerationStructure = nullptr,
                .dstAccelerationStructure = nullptr,
                .geometryCount = 1,
                .pGeometries = &geometry,
                .scratchData = {}
        };


        // CALCULATE REQUIRED SIZE FOR THE ACCELERATION STRUCTURE
        vk::AccelerationStructureBuildSizesInfoKHR buildSizesInfo = device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo, { 1 }, dynamicDispatchLoader);

        VulkanAccelerationStructure topAccelerationStructure{};
        // ALLOCATE BUFFERS FOR ACCELERATION STRUCTURE
        topAccelerationStructure.structureBuffer = vulkan::create_buffer(device, buildSizesInfo.accelerationStructureSize,
            vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR,
            vk::MemoryPropertyFlagBits::eDeviceLocal, memory_properties);

        topAccelerationStructure.scratchBuffer = vulkan::create_buffer(device, buildSizesInfo.buildScratchSize,
            vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eDeviceLocal, memory_properties);

        // CREATE THE ACCELERATION STRUCTURE
        vk::AccelerationStructureCreateInfoKHR createInfo = {
                .buffer = topAccelerationStructure.structureBuffer.buffer,
                .offset = 0,
                .size = buildSizesInfo.accelerationStructureSize,
                .type = vk::AccelerationStructureTypeKHR::eTopLevel
        };

        topAccelerationStructure.accelerationStructure =
            device.createAccelerationStructureKHR(createInfo, nullptr, dynamicDispatchLoader);


        // CREATE INSTANCE INFO & WRITE IN NEW BUFFER
        std::array<std::array<float, 4>, 3> matrix = {
                {
                        {1.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, 1.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 1.0f, 0.0f}
                } };

        vk::AccelerationStructureInstanceKHR accelerationStructureInstance = {
                .transform = {.matrix = matrix},
                .instanceCustomIndex = 0,
                .mask = 0xFF,
                .instanceShaderBindingTableRecordOffset = 0,
                .accelerationStructureReference = device.getAccelerationStructureAddressKHR(
                        {.accelerationStructure = bottom_accel},
                        dynamicDispatchLoader),
        };

        topAccelerationStructure.instancesBuffer = vulkan::create_buffer(
            device,
            sizeof(vk::AccelerationStructureInstanceKHR),
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostCoherent |
            vk::MemoryPropertyFlagBits::eHostVisible,
            memory_properties);

        void* pInstancesBuffer = device.mapMemory(topAccelerationStructure.instancesBuffer.memory, 0,
            sizeof(vk::AccelerationStructureInstanceKHR));
        memcpy(pInstancesBuffer, &accelerationStructureInstance, sizeof(vk::AccelerationStructureInstanceKHR));
        device.unmapMemory(topAccelerationStructure.instancesBuffer.memory);


        // FILL IN THE REMAINING META INFO
        buildInfo.dstAccelerationStructure = topAccelerationStructure.accelerationStructure;
        buildInfo.scratchData.deviceAddress = device.getBufferAddress(
            { .buffer = topAccelerationStructure.scratchBuffer.buffer });

        geometry.geometry.instances.data.deviceAddress = device.getBufferAddress(
            { .buffer = topAccelerationStructure.instancesBuffer.buffer });

        return std::tuple{ topAccelerationStructure, buildInfo };
    }


    inline auto create_descriptor_set_layout(vk::Device device) {
        std::vector<vk::DescriptorSetLayoutBinding> bindings = {
                {
                        .binding = 0,
                        .descriptorType = vk::DescriptorType::eStorageImage,
                        .descriptorCount = 1,
                        .stageFlags = vk::ShaderStageFlagBits::eRaygenKHR
                },
                {
                        .binding = 1,
                        .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
                        .descriptorCount = 1,
                        .stageFlags = vk::ShaderStageFlagBits::eRaygenKHR
                },
                {
                        .binding = 2,
                        .descriptorType = vk::DescriptorType::eUniformBuffer,
                        .descriptorCount = 1,
                        .stageFlags = vk::ShaderStageFlagBits::eIntersectionKHR |
                                      vk::ShaderStageFlagBits::eClosestHitKHR
                },
                {
                        .binding = 3,
                        .descriptorType = vk::DescriptorType::eStorageImage,
                        .descriptorCount = 1,
                        .stageFlags = vk::ShaderStageFlagBits::eRaygenKHR
                },
                {
                        .binding = 4,
                        .descriptorType = vk::DescriptorType::eUniformBuffer,
                        .descriptorCount = 1,
                        .stageFlags = vk::ShaderStageFlagBits::eRaygenKHR
                }
        };

        auto rtDescriptorSetLayout = device.createDescriptorSetLayout(
            {
                    .bindingCount = static_cast<uint32_t>(bindings.size()),
                    .pBindings = bindings.data()
            });
        return rtDescriptorSetLayout;
    }

    inline auto create_descriptor_pool(vk::Device device, uint32_t swapchain_image_count) {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
                {
                        .type = vk::DescriptorType::eStorageImage,
                        .descriptorCount = 2 * swapchain_image_count
                },
                {
                        .type = vk::DescriptorType::eAccelerationStructureKHR,
                        .descriptorCount = 1 * swapchain_image_count
                },
                {
                        .type = vk::DescriptorType::eUniformBuffer,
                        .descriptorCount = 2 * swapchain_image_count
                }
        };

        auto rtDescriptorPool = device.createDescriptorPool(
            {
                    .maxSets = swapchain_image_count,
                    .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                    .pPoolSizes = poolSizes.data()
            });
        return rtDescriptorPool;
    }


    inline auto create_sphere_buffer(vk::Device device, const vk::PhysicalDeviceMemoryProperties& memory_properties) {
        const vk::DeviceSize bufferSize = sizeof(Sphere) * MAX_SPHERE_AMOUNT;

        auto sphereBuffer = vulkan::create_buffer(device, bufferSize,
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent |
            vk::MemoryPropertyFlagBits::eDeviceLocal, memory_properties);
        return sphereBuffer;
    }

    inline auto create_render_call_info_buffers(vk::Device device, uint32_t swapchain_image_count, const vk::PhysicalDeviceMemoryProperties& memory_properties) {
        std::vector<VulkanBuffer> renderCallInfoBuffers(swapchain_image_count);
        std::ranges::generate(
            renderCallInfoBuffers,
            [device, &memory_properties]() {
                return vulkan::create_buffer(device, sizeof(RenderCallInfo), vk::BufferUsageFlagBits::eUniformBuffer,
                    vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent |
                    vk::MemoryPropertyFlagBits::eDeviceLocal,
                    memory_properties);
            });
        return renderCallInfoBuffers;
    }

    inline auto create_descriptor_set(vk::Device device, uint32_t swapchain_image_count,
        vk::DescriptorSetLayout rtDescriptorSetLayout,
        vk::DescriptorPool rtDescriptorPool,
        const auto& render_target_images,
        const auto& top_accelerations,
        const auto& sphereBuffers,
        const auto& summed_images,
        const auto& renderCallInfoBuffers) {
        std::vector<vk::DescriptorSetLayout> layouts(swapchain_image_count);
        std::ranges::fill(layouts, rtDescriptorSetLayout);
        auto rtDescriptorSets = device.allocateDescriptorSets(
            vk::DescriptorSetAllocateInfo{}
            .setDescriptorPool(rtDescriptorPool)
            .setSetLayouts(layouts)
        );

        auto render_target_image_infos = std::vector<vk::DescriptorImageInfo>(swapchain_image_count);
        std::ranges::transform(
            render_target_images,
            render_target_image_infos.begin(),
            [](auto& image) {
                return vk::DescriptorImageInfo{ .imageView = image.imageView, .imageLayout = vk::ImageLayout::eGeneral };
            }
        );

        auto acceleration_structure_infos = std::vector<vk::WriteDescriptorSetAccelerationStructureKHR>(swapchain_image_count);
        std::ranges::transform(
            top_accelerations,
            acceleration_structure_infos.begin(),
            [](auto& accel) {
                return vk::WriteDescriptorSetAccelerationStructureKHR{
                    .accelerationStructureCount = 1,
                    .pAccelerationStructures = &accel.accelerationStructure
                };
            }
        );

        auto sphere_buffer_infos = std::vector<vk::DescriptorBufferInfo>(swapchain_image_count);
        std::ranges::transform(
            sphereBuffers,
            sphere_buffer_infos.begin(),
            [](auto& sphere_buffer) {
                return  vk::DescriptorBufferInfo{
                    .buffer = sphere_buffer.buffer,
                    .offset = 0,
                    .range = sizeof(Sphere) * MAX_SPHERE_AMOUNT
                };
            }
        );

        auto summed_image_infos = std::vector<vk::DescriptorImageInfo>(swapchain_image_count);
        std::ranges::transform(
            summed_images,
            summed_image_infos.begin(),
            [](auto& image) {
                return vk::DescriptorImageInfo{ .imageView = image.imageView, .imageLayout = vk::ImageLayout::eGeneral };
            }
        );

        std::vector<vk::DescriptorBufferInfo> renderCallInfoBufferInfos(swapchain_image_count);

        std::vector<vk::WriteDescriptorSet> descriptorWrites{};
        for (int i = 0; i < swapchain_image_count; i++) {
            auto set = rtDescriptorSets[i];
            descriptorWrites.push_back(
                {
                        .dstSet = set,
                        .dstBinding = 0,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eStorageImage,
                        .pImageInfo = &render_target_image_infos[i]
                });
            descriptorWrites.push_back(
                {
                        .pNext = &acceleration_structure_infos[i],
                        .dstSet = set,
                        .dstBinding = 1,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eAccelerationStructureKHR
                });
            descriptorWrites.push_back(
                {
                        .dstSet = set,
                        .dstBinding = 2,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eUniformBuffer,
                        .pBufferInfo = &sphere_buffer_infos[i]
                });
            descriptorWrites.push_back(
                {
                        .dstSet = set,
                        .dstBinding = 3,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eStorageImage,
                        .pImageInfo = &summed_image_infos[i]
                });
            renderCallInfoBufferInfos[i] = vk::DescriptorBufferInfo{}
                .setBuffer(renderCallInfoBuffers[i].buffer)
                .setRange(vk::WholeSize);
            descriptorWrites.push_back(
                {
                        .dstSet = set,
                        .dstBinding = 4,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eUniformBuffer,
                        .pBufferInfo = &renderCallInfoBufferInfos[i]
                });
        };

        device.updateDescriptorSets(static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(),
            0, nullptr);

        return rtDescriptorSets;
    }

    inline auto create_pipeline_layout(vk::Device device, vk::DescriptorSetLayout rtDescriptorSetLayout) {
        auto rtPipelineLayout = device.createPipelineLayout(
            {
                    .setLayoutCount = 1,
                    .pSetLayouts = &rtDescriptorSetLayout,
                    .pushConstantRangeCount = 0,
                    .pPushConstantRanges = nullptr
            });
        return rtPipelineLayout;
    }

    inline std::vector<char> readBinaryFile(const std::string& path) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);

        if (!file.is_open())
            throw std::runtime_error("[Error] Failed to open file at '" + path + "'!");

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    inline vk::ShaderModule createShaderModule(vk::Device device, const std::string& path) {
        std::vector<char> shaderCode = readBinaryFile(path);

        vk::ShaderModuleCreateInfo shaderModuleCreateInfo = {
                .codeSize = shaderCode.size(),
                .pCode = reinterpret_cast<const uint32_t*>(shaderCode.data())
        };

        return device.createShaderModule(shaderModuleCreateInfo);
    }

    inline auto create_rt_pipeline(vk::Device device, uint32_t max_depth, vk::PipelineLayout pipeline_layout, vk::detail::DispatchLoaderDynamic& dynamicDispatchLoader) {
        vk::ShaderModule raygenModule = createShaderModule(device, rgen_shader_path);
        vk::ShaderModule intModule = createShaderModule(device, rint_shader_path);
        vk::ShaderModule chitModule = createShaderModule(device, rchit_shader_path);
        vk::ShaderModule missModule = createShaderModule(device, rmiss_shader_path);

        std::vector<vk::PipelineShaderStageCreateInfo> stages = {
                {
                        .stage = vk::ShaderStageFlagBits::eRaygenKHR,
                        .module = raygenModule,
                        .pName = "main"
                },
                {
                        .stage = vk::ShaderStageFlagBits::eIntersectionKHR,
                        .module = intModule,
                        .pName = "main"
                },
                {
                        .stage = vk::ShaderStageFlagBits::eMissKHR,
                        .module = missModule,
                        .pName = "main"
                },
                {
                        .stage = vk::ShaderStageFlagBits::eClosestHitKHR,
                        .module = chitModule,
                        .pName = "main"
                }
        };

        std::vector<vk::RayTracingShaderGroupCreateInfoKHR> groups = {
                {
                        .type = vk::RayTracingShaderGroupTypeKHR::eGeneral,
                        .generalShader = 0,
                        .closestHitShader = VK_SHADER_UNUSED_KHR,
                        .anyHitShader = VK_SHADER_UNUSED_KHR,
                        .intersectionShader = VK_SHADER_UNUSED_KHR
                },
                {
                        .type = vk::RayTracingShaderGroupTypeKHR::eGeneral,
                        .generalShader = 2,
                        .closestHitShader = VK_SHADER_UNUSED_KHR,
                        .anyHitShader = VK_SHADER_UNUSED_KHR,
                        .intersectionShader = VK_SHADER_UNUSED_KHR
                },
                {
                        .type = vk::RayTracingShaderGroupTypeKHR::eProceduralHitGroup,
                        .generalShader = VK_SHADER_UNUSED_KHR,
                        .closestHitShader = 3,
                        .anyHitShader = VK_SHADER_UNUSED_KHR,
                        .intersectionShader = 1
                }
        };

        vk::PipelineLibraryCreateInfoKHR libraryCreateInfo = { .libraryCount = 0 };

        vk::RayTracingPipelineCreateInfoKHR pipelineCreateInfo = {
                .stageCount = static_cast<uint32_t>(stages.size()),
                .pStages = stages.data(),
                .groupCount = static_cast<uint32_t>(groups.size()),
                .pGroups = groups.data(),
                .maxPipelineRayRecursionDepth = max_depth,
                .pLibraryInfo = &libraryCreateInfo,
                .pLibraryInterface = nullptr,
                .layout = pipeline_layout,
                .basePipelineHandle = VK_NULL_HANDLE,
                .basePipelineIndex = 0
        };

        auto rtPipeline = device.createRayTracingPipelineKHR(nullptr, nullptr, pipelineCreateInfo,
            nullptr, dynamicDispatchLoader).value;

        device.destroyShaderModule(raygenModule);
        device.destroyShaderModule(chitModule);
        device.destroyShaderModule(missModule);
        device.destroyShaderModule(intModule);

        return rtPipeline;
    }


    inline auto create_shader_binding_table_buffer(vk::Device device,
        vk::Pipeline rtPipeline,
        vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties,
        const vk::PhysicalDeviceMemoryProperties& memory_properties,
        vk::detail::DispatchLoaderDynamic& dynamicDispatchLoader) {

        uint32_t baseAlignment = rayTracingProperties.shaderGroupBaseAlignment;
        uint32_t handleSize = rayTracingProperties.shaderGroupHandleSize;


        const uint32_t shaderGroupCount = 3;
        vk::DeviceSize sbtBufferSize = baseAlignment * shaderGroupCount;

        auto shaderBindingTableBuffer = vulkan::create_buffer(device, sbtBufferSize,
            vk::BufferUsageFlagBits::eShaderBindingTableKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent |
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            memory_properties);


        std::vector<uint8_t> handles = device.getRayTracingShaderGroupHandlesKHR<uint8_t>(
            rtPipeline, 0, shaderGroupCount, shaderGroupCount * handleSize, dynamicDispatchLoader);

        vk::DeviceAddress sbtAddress = device.getBufferAddress({ .buffer = shaderBindingTableBuffer.buffer });

        vk::StridedDeviceAddressRegionKHR addressRegion = {
                .stride = baseAlignment,
                .size = handleSize
        };

        auto sbtRayGenAddressRegion = addressRegion;
        sbtRayGenAddressRegion.size = baseAlignment;
        sbtRayGenAddressRegion.deviceAddress = sbtAddress;

        auto sbtMissAddressRegion = addressRegion;
        sbtMissAddressRegion.deviceAddress = sbtAddress + baseAlignment;

        auto sbtHitAddressRegion = addressRegion;
        sbtHitAddressRegion.deviceAddress = sbtAddress + baseAlignment * 2;

        uint8_t* sbtBufferData = static_cast<uint8_t*>(device.mapMemory(shaderBindingTableBuffer.memory, 0, sbtBufferSize));

        memcpy(sbtBufferData, handles.data(), handleSize);
        memcpy(sbtBufferData + baseAlignment, handles.data() + handleSize, handleSize);
        memcpy(sbtBufferData + baseAlignment * 2, handles.data() + handleSize * 2, handleSize);

        device.unmapMemory(shaderBindingTableBuffer.memory);

        return std::tuple{ shaderBindingTableBuffer, sbtRayGenAddressRegion, sbtMissAddressRegion, sbtHitAddressRegion };
    }

    inline auto record_ray_tracing(vk::CommandBuffer commandBuffer, uint32_t queue_family, vk::Image render_target_image, vk::Image summed_image,
        vk::Pipeline pipeline, vk::DescriptorSet descriptor_set, vk::PipelineLayout pipeline_layout,
        vk::StridedDeviceAddressRegionKHR sbtRayGenAddressRegion, vk::StridedDeviceAddressRegionKHR sbtMissAddressRegion, vk::StridedDeviceAddressRegionKHR sbtHitAddressRegion,
        uint32_t width, uint32_t height, vk::detail::DispatchLoaderDynamic& dynamicDispatchLoader) {
        // RENDER TARGET IMAGE UNDEFINED -> GENERAL
        // Sync summed pixel color image with previous ray tracing.
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eRayTracingShaderKHR,
            vk::PipelineStageFlagBits::eRayTracingShaderKHR,
            vk::DependencyFlagBits::eByRegion, {}, {},
            std::array{
                vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eNoneKHR,
                .dstAccessMask = vk::AccessFlagBits::eShaderWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = queue_family,
                .dstQueueFamilyIndex = queue_family,
                .image = render_target_image,
                .subresourceRange = {
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                },
                },
                vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
                .dstAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = queue_family,
                .dstQueueFamilyIndex = queue_family,
                .image = summed_image,
                .subresourceRange = {
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                }
                }
            });

        // RAY TRACING
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, pipeline);

        std::vector<vk::DescriptorSet> descriptorSets = { descriptor_set };
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, pipeline_layout,
            0, descriptorSets, nullptr);

        commandBuffer.traceRaysKHR(sbtRayGenAddressRegion, sbtMissAddressRegion, sbtHitAddressRegion, {},
            width, height, 1, dynamicDispatchLoader);
    }

    inline auto create_command_buffers(vk::Device device, vk::CommandPool commandPool, uint32_t swapchain_images_count, const auto& swapchain_images,
        uint32_t queue_family, auto& render_target_images, auto& summed_images,
        vk::Pipeline pipeline, const auto& descriptor_sets, vk::PipelineLayout pipeline_layout,
        auto& aabbs, auto& bottom_accel_build_infos, auto& bottom_accels,
        auto& top_accel_build_infos, auto& top_accels,
        vk::StridedDeviceAddressRegionKHR sbtRayGenAddressRegion, vk::StridedDeviceAddressRegionKHR sbtMissAddressRegion, vk::StridedDeviceAddressRegionKHR sbtHitAddressRegion,
        uint32_t width, uint32_t height, vk::Extent2D image_extent, vk::detail::DispatchLoaderDynamic& dynamicDispatchLoader) {
        auto commandBuffers = std::vector<vk::CommandBuffer>(swapchain_images_count);
        for (int swapChainImageIndex = 0; swapChainImageIndex < swapchain_images_count; swapChainImageIndex++) {
            auto& commandBuffer = commandBuffers[swapChainImageIndex];
            auto& swapChainImage = swapchain_images[swapChainImageIndex];
            commandBuffer = device.allocateCommandBuffers(
                {
                        .commandPool = commandPool,
                        .level = vk::CommandBufferLevel::ePrimary,
                        .commandBufferCount = 1
                }).front();

            vk::CommandBufferBeginInfo beginInfo = {};
            commandBuffer.begin(&beginInfo);


            // BUILD THE ACCELERATION STRUCTURE
            vk::AccelerationStructureBuildRangeInfoKHR buildRangeInfo = {
                    .primitiveCount = static_cast<uint32_t>(aabbs.size()),
                    .primitiveOffset = 0,
                    .firstVertex = 0,
                    .transformOffset = 0
            };

            const vk::AccelerationStructureBuildRangeInfoKHR* pBuildRangeInfos[] = { &buildRangeInfo };
            commandBuffer.buildAccelerationStructuresKHR(1, &bottom_accel_build_infos[swapChainImageIndex], pBuildRangeInfos, dynamicDispatchLoader);
            commandBuffer.pipelineBarrier2(
                vk::DependencyInfo{}
                .setBufferMemoryBarriers(
                    vk::BufferMemoryBarrier2{}.setBuffer(bottom_accels[swapChainImageIndex].structureBuffer.buffer).setSrcAccessMask(vk::AccessFlagBits2::eMemoryWrite).setDstAccessMask(vk::AccessFlagBits2::eMemoryWrite)
                    .setSrcQueueFamilyIndex(queue_family).setDstQueueFamilyIndex(queue_family)
                    .setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands).setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
                    .setSize(vk::WholeSize)
                )
                );

            // BUILD THE ACCELERATION STRUCTURE
            vk::AccelerationStructureBuildRangeInfoKHR top_buildRangeInfo = {
                    .primitiveCount = 1,
                    .primitiveOffset = 0,
                    .firstVertex = 0,
                    .transformOffset = 0
            };

            const vk::AccelerationStructureBuildRangeInfoKHR* p_top_BuildRangeInfos[] = { &top_buildRangeInfo };
            commandBuffer.buildAccelerationStructuresKHR(1, &top_accel_build_infos[swapChainImageIndex], p_top_BuildRangeInfos, dynamicDispatchLoader);

            commandBuffer.pipelineBarrier2(
                vk::DependencyInfo{}
                .setBufferMemoryBarriers(
                    vk::BufferMemoryBarrier2{}.setBuffer(top_accels[swapChainImageIndex].structureBuffer.buffer).setSrcAccessMask(vk::AccessFlagBits2::eMemoryWrite).setDstAccessMask(vk::AccessFlagBits2::eMemoryWrite)
                    .setSrcQueueFamilyIndex(queue_family).setDstQueueFamilyIndex(queue_family)
                    .setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands).setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
                    .setSize(vk::WholeSize)
                )
            );

            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eTransfer,
                vk::DependencyFlagBits::eByRegion,
                {}, {},
                vk::ImageMemoryBarrier{
                    .srcAccessMask = vk::AccessFlagBits::eNoneKHR,
                    .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .oldLayout = vk::ImageLayout::eUndefined,
                    .newLayout = vk::ImageLayout::eTransferDstOptimal,
                    .srcQueueFamilyIndex = queue_family,
                    .dstQueueFamilyIndex = queue_family,
                    .image = summed_images[swapChainImageIndex].image,
                    .subresourceRange = {
                            .aspectMask = vk::ImageAspectFlagBits::eColor,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1
                    },
                });
            commandBuffer.clearColorImage(summed_images[swapChainImageIndex].image, vk::ImageLayout::eTransferDstOptimal,
                vk::ClearColorValue{},
                vk::ImageSubresourceRange{}
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setLayerCount(1)
                .setLevelCount(1));
            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eRayTracingShaderKHR,
                vk::DependencyFlagBits::eByRegion,
                {}, {},
                vk::ImageMemoryBarrier{
                    .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                    .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                    .newLayout = vk::ImageLayout::eGeneral,
                    .srcQueueFamilyIndex = queue_family,
                    .dstQueueFamilyIndex = queue_family,
                    .image = summed_images[swapChainImageIndex].image,
                    .subresourceRange = {
                            .aspectMask = vk::ImageAspectFlagBits::eColor,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1
                    },
                });

            record_ray_tracing(commandBuffer, queue_family, render_target_images[swapChainImageIndex].image, summed_images[swapChainImageIndex].image,
                pipeline, descriptor_sets[swapChainImageIndex], pipeline_layout,
                sbtRayGenAddressRegion, sbtMissAddressRegion, sbtHitAddressRegion,
                width, height, dynamicDispatchLoader);

            // RENDER TARGET IMAGE: GENERAL -> TRANSFER SRC & SWAP CHAIN IMAGE: UNDEFINED -> TRANSFER DST
            vk::ImageMemoryBarrier imageBarriersToTransfer[2] = {
                vk::ImageMemoryBarrier{
                    .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
                    .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                    .oldLayout = vk::ImageLayout::eGeneral,
                    .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                    .srcQueueFamilyIndex = queue_family,
                    .dstQueueFamilyIndex = queue_family,
                    .image = render_target_images[swapChainImageIndex].image,
                    .subresourceRange = {
                            .aspectMask = vk::ImageAspectFlagBits::eColor,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1
                    },
                },
                vk::ImageMemoryBarrier{
                    .srcAccessMask = vk::AccessFlagBits::eMemoryRead,
                    .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .oldLayout = vk::ImageLayout::eUndefined,
                    .newLayout = vk::ImageLayout::eTransferDstOptimal,
                    .srcQueueFamilyIndex = queue_family,
                    .dstQueueFamilyIndex = queue_family,
                    .image = swapChainImage,
                    .subresourceRange = {
                            .aspectMask = vk::ImageAspectFlagBits::eColor,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1
                    },
                }
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
                            .width = image_extent.width,
                            .height = image_extent.height,
                            .depth = 1
                    }
            };

            commandBuffer.copyImage(render_target_images[swapChainImageIndex].image, vk::ImageLayout::eTransferSrcOptimal, swapChainImage,
                vk::ImageLayout::eTransferDstOptimal, 1, &imageCopy);


            // SWAP CHAIN IMAGE: TRANSFER DST -> PRESENT
            vk::ImageMemoryBarrier barrierSwapChainToPresent = vk::ImageMemoryBarrier{
                    .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .dstAccessMask = vk::AccessFlagBits::eMemoryRead,
                    .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                    .newLayout = vk::ImageLayout::ePresentSrcKHR,
                    .srcQueueFamilyIndex = queue_family,
                    .dstQueueFamilyIndex = queue_family,
                    .image = swapChainImage,
                    .subresourceRange = {
                            .aspectMask = vk::ImageAspectFlagBits::eColor,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1
                    },
            };

            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
                vk::DependencyFlagBits::eByRegion, 0, nullptr,
                0, nullptr, 1, &barrierSwapChainToPresent);

            commandBuffer.end();
        }

        return commandBuffers;
    }


    inline auto execute_single_time_command(vk::Device device, vk::Queue queue, vk::CommandPool command_pool, const std::function<void(const vk::CommandBuffer& singleTimeCommandBuffer)>& c) {
        vk::CommandBuffer singleTimeCommandBuffer = device.allocateCommandBuffers(
            {
                    .commandPool = command_pool,
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
        queue.submit(1, &submitInfo, f);
        device.waitForFences(1, &f, true, UINT64_MAX);

        device.destroyFence(f);
        device.freeCommandBuffers(command_pool, singleTimeCommandBuffer);
    }

    inline auto update_accel_structures_data(vk::Device device,
        auto& aabbs, VulkanBuffer& aabb_buffer,
        VulkanBuffer sphere_buffer, uint32_t sphere_buffer_size,
        std::span<Sphere> spheres
    ) {
        auto aabbs_buffer_size = sizeof(aabbs[0]) * aabbs.size();
        void* data = device.mapMemory(aabb_buffer.memory, 0, aabbs_buffer_size);
        memcpy(data, aabbs.data(), aabbs_buffer_size);
        device.unmapMemory(aabb_buffer.memory);

        {
            void* data = device.mapMemory(sphere_buffer.memory, 0, sphere_buffer_size);
            memcpy(data, spheres.data(), sizeof(Sphere) * spheres.size());
            device.unmapMemory(sphere_buffer.memory);
        }
    }

}


class Vulkan {
public:
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

private:
    
    std::vector<vk::Fence> m_fences;
    vk::Fence get_fence(uint32_t image_index) {
        return m_fences[image_index];
    }

    // There is swapchain image count + 1 semaphores
    // So that we could get a free semaphore.
    std::vector<vk::Semaphore> m_next_image_semaphores;
    uint32_t m_next_image_free_semaphore_index;
    std::vector<uint32_t> m_next_image_semaphores_indices;
    vk::Semaphore get_acquire_image_semaphore() {
        return m_next_image_semaphores[m_next_image_free_semaphore_index];
    }
    void free_acquire_image_semaphore(uint32_t image_index) {
        std::swap(m_next_image_free_semaphore_index, m_next_image_semaphores_indices[image_index]);
    }

    // Semaphore for swapchain present.
    std::vector<vk::Semaphore> m_render_image_semaphores;
    vk::Semaphore get_render_image_semaphore(uint32_t image_index) {
        return m_render_image_semaphores[image_index];
    }
};
