#pragma once

#include "vulkan.hpp"

#include <GLFW/glfw3.h>

#include <span>

class window {
public:
	window(uint32_t width, uint32_t height) {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		m_window = glfwCreateWindow(width, height, "GPU Ray Tracing (Vulkan)",
			nullptr, nullptr);
	}
	~window() {
		glfwDestroyWindow(m_window);
		glfwTerminate();
	}

	void poll_events() {
		glfwPollEvents();
	}
private:
	static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {

	}
public:

	void set_cursor_event() {
		glfwSetCursorPosCallback(m_window, cursor_position_callback);
	}

	auto get_cursor_position() {
		double xpos, ypos;
		glfwGetCursorPos(m_window, &xpos, &ypos);
		return std::tuple{ xpos, ypos };
	}

	bool should_close() {
		return glfwWindowShouldClose(m_window);
	}

	auto get_required_extensions() {
		uint32_t count{ 0 };
		const char** extensions = glfwGetRequiredInstanceExtensions(&count);
		return std::span{ extensions, count };
	}
	auto create_surface(vk::Instance instance) {
		VkSurfaceKHR surface;
		auto res = glfwCreateWindowSurface(instance, m_window, nullptr, &surface);
		if (res != VK_SUCCESS) {
			throw std::runtime_error{ "failed to create surface" };
		}

		return vk::SurfaceKHR{ surface };
	}
private:
	GLFWwindow* m_window;
};
