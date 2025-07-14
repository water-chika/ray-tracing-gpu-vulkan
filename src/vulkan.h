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
}


class Vulkan {
public:
    Vulkan(
        vk::Instance instance, vk::SurfaceKHR surface, vk::PhysicalDevice physical_device,
        std::pair<uint32_t, uint32_t> compute_present_queue_families,
        VulkanSettings settings, Scene scene) :
        instance{instance}, surface{surface},
        physicalDevice{ physical_device },
        computeQueueFamily{ compute_present_queue_families.first }, presentQueueFamily{ compute_present_queue_families.second },
        settings(settings), scene(scene),
        m_width(settings.windowWidth), m_height(settings.windowHeight)
    {

        aabbs.reserve(scene.sphereAmount);
        for (int i = 0; i < scene.sphereAmount; i++) {
            aabbs.push_back(getAABBFromSphere(scene.spheres[i].geometry));
        }
        createLogicalDevice();

        dynamicDispatchLoader = vk::detail::DispatchLoaderDynamic(instance, vkGetInstanceProcAddr, device);

        createCommandPool();
        createSwapChain();
        createImages();

        createAABBBuffer();
        createBottomAccelerationStructure();
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

        createFence();
        createSemaphore();
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

    const vk::Format swapChainImageFormat = vk::Format::eR8G8B8A8Unorm;
    const vk::Format summedPixelColorImageFormat = vk::Format::eR32G32B32A32Sfloat;
    const vk::ColorSpaceKHR colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
    vk::PresentModeKHR presentMode = vk::PresentModeKHR::eImmediate;




    vk::Instance instance;
    vk::SurfaceKHR surface;
    vk::PhysicalDevice physicalDevice;
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

    void createSurface(auto&& create_surface) {
        surface = create_surface(instance);
    }

    void findQueueFamilies();

    void createLogicalDevice();

    void createCommandPool();

    void createSwapChain();

    [[nodiscard]] vk::ImageView createImageView(const vk::Image &image, const vk::Format &format) const;

    void createDescriptorSetLayout();

    void createDescriptorPool();

    void createDescriptorSet();

    void createPipelineLayout();

    void createRTPipeline();

    [[nodiscard]] static std::vector<char> readBinaryFile(const std::string &path);

    void record_ray_tracing(vk::CommandBuffer cmd, int index);
    void createCommandBuffer();

    void createFence();

    void createSemaphore();

    void createImages();

    [[nodiscard]] uint32_t findMemoryTypeIndex(const uint32_t &memoryTypeBits,
                                               const vk::MemoryPropertyFlags &properties);

    [[nodiscard]] vk::ImageMemoryBarrier getImagePipelineBarrier(
            const vk::AccessFlags srcAccessFlags, const vk::AccessFlags dstAccessFlags,
            const vk::ImageLayout &oldLayout, const vk::ImageLayout &newLayout, const vk::Image &image) const;

    [[nodiscard]] VulkanImage createImage(const vk::Format &format,
                                          const vk::Flags<vk::ImageUsageFlagBits> &usageFlagBits);

    void destroyImage(const VulkanImage &image) const;

    [[nodiscard]] VulkanBuffer createBuffer(const vk::DeviceSize &size, const vk::Flags<vk::BufferUsageFlagBits> &usage,
                                            const vk::Flags<vk::MemoryPropertyFlagBits> &memoryProperty);

    void destroyBuffer(const VulkanBuffer &buffer) const;

    void executeSingleTimeCommand(const std::function<void(const vk::CommandBuffer &singleTimeCommandBuffer)> &c);

    void createAABBBuffer();

    void createBottomAccelerationStructure();

    void createTopAccelerationStructure();

    void destroyAccelerationStructure(const VulkanAccelerationStructure &accelerationStructure);

    [[nodiscard]] vk::ShaderModule createShaderModule(const std::string &path) const;

    void createShaderBindingTable();

    [[nodiscard]] vk::PhysicalDeviceRayTracingPipelinePropertiesKHR getRayTracingProperties() const;

    void createSphereBuffer();

    [[nodiscard]] static vk::AabbPositionsKHR getAABBFromSphere(const glm::vec4 &geometry);

    void createRenderCallInfoBuffer();

    void updateRenderCallInfoBuffer(const RenderCallInfo &renderCallInfo, int index);

};
