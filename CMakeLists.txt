cmake_minimum_required(VERSION 3.20)
project(RayTracingGPUVulkan)

set(CMAKE_CXX_STANDARD 23)

find_package(Vulkan REQUIRED)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/glfw/CMakeLists.txt)
    add_subdirectory(lib/glfw)
else()
    find_library(glfw glfw REQUIRED)
endif()

add_executable(
        RayTracingGPUVulkan
        src/main.cpp
        src/vulkan_settings.h
        src/vulkan.h
        src/vulkan.cpp
        src/scene.h
        src/scene.cpp
        src/render_call_info.h
        ${CMAKE_CURRENT_BINARY_DIR}/include/shader_path.hpp
)

target_link_libraries(RayTracingGPUVulkan
    glfw
    Vulkan::Vulkan
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/glm/CMakeLists.txt)
    add_subdirectory(lib/glm)
    target_link_libraries(RayTracingGPUVulkan glm)
else()
    include(CheckIncludeFileCXX)
    CHECK_INCLUDE_FILE_CXX(glm/glm.hpp glm_exist)
    if(${glm_exist})
    else()
        message(FATAL_ERROR "glm not found")
    endif()
endif()


if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/stb/stb_image_write.h)
    target_include_directories(RayTracingGPUVulkan PUBLIC lib)
else()
    include(CheckIncludeFileCXX)
    CHECK_INCLUDE_FILE_CXX(stb/stb_image_write.h stb_exist)
    if(${stb_exist})
    else()
        message(FATAL_ERROR "stb not found")
    endif()
endif()


function(compile_glsl stage glsl_file spv_file)
add_custom_command(COMMENT "Compiling ${stage} shader"
                    OUTPUT ${spv_file}
                    COMMAND Vulkan::glslangValidator -V --target-env vulkan1.3 -S ${stage} -o ${spv_file}
                            ${glsl_file}
                    MAIN_DEPENDENCY ${glsl_file}
                    DEPENDS ${glsl_file} Vulkan::glslangValidator)
endfunction()
function(compile_glsl_help stage)
    compile_glsl(${stage}
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.${stage}
        ${CMAKE_CURRENT_BINARY_DIR}/shaders/shader.${stage}.spv
    )
    set(
        ${stage}_shader_path
        "shaders/shader.${stage}.spv"
        PARENT_SCOPE
    )
    target_sources(RayTracingGPUVulkan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.${stage} ${CMAKE_CURRENT_BINARY_DIR}/shaders/shader.${stage}.spv)
endfunction()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders)
compile_glsl_help(rgen)
compile_glsl_help(rint)
compile_glsl_help(rchit)
compile_glsl_help(rmiss)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader_path.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/include/shader_path.hpp
)

target_include_directories(RayTracingGPUVulkan PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)
