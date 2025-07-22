#pragma once

#include "vulkan.hpp"

#include <GLFW/glfw3.h>

#include <span>

namespace window {
	struct window_system {};
	inline window_system init_window_system() {
		glfwInit();
		return {};
	}
	inline void destroy_window_system(window_system&) {
		glfwTerminate();
	}
	inline void poll_events(window_system&) {
		glfwPollEvents();
	}
	inline auto get_vulkan_required_extensions(window_system&) {
		uint32_t count{ 0 };
		const char** extensions = glfwGetRequiredInstanceExtensions(&count);
		return std::span{ extensions, count };
	}
	struct window {
		GLFWwindow* glfw_window;
	};
	inline window create_window(window_system&, uint32_t width, uint32_t height, uint32_t x = 0, uint32_t y = 0) {
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
		glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);
		glfwWindowHint(GLFW_POSITION_X, x);
		glfwWindowHint(GLFW_POSITION_Y, y);
		auto glfw_window = glfwCreateWindow(width, height, "GPU Ray Tracing (Vulkan)",
			nullptr, nullptr);
		return window{ glfw_window };
	}
	inline void destroy_window(window& window) {
		glfwDestroyWindow(window.glfw_window);
	}
	inline auto get_window_cursor_position(window& window) {
		double xpos, ypos;
		glfwGetCursorPos(window.glfw_window, &xpos, &ypos);
		return std::tuple{ xpos, ypos };
	}
	inline void set_window_position(window& window, auto pos) {
		auto [x, y] = pos;
		glfwSetWindowPos(window.glfw_window, x, y);
	}
	inline void set_window_size(window& window, auto size) {
		auto [width, height] = size;
		glfwSetWindowSize(window.glfw_window, width, height);
	}
	inline auto should_window_close(window& window) {
		return glfwWindowShouldClose(window.glfw_window);
	}
	inline auto create_window_vulkan_surface(window& window, VkInstance instance) {
		VkSurfaceKHR surface;
		auto res = glfwCreateWindowSurface(instance, window.glfw_window, nullptr, &surface);
		if (res != VK_SUCCESS) {
			throw std::runtime_error{ "failed to create surface" };
		}

		return surface;
	}
}
