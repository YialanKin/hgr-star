# 指定最低 CMake 版本
cmake_minimum_required(VERSION 3.18)

# 设置项目元信息
project(HGRSTAR VERSION 1.0 LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 查找 PyTorch (LibTorch)
find_package(Torch REQUIRED)

# -----------------------------
# 设置 CUDA（如果项目使用 CUDA）
# -----------------------------
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "nvcc" CACHE STRING "CUDA compiler")
endif()

# -----------------------------
# 添加第三方依赖：nlohmann/json
# -----------------------------
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.2  # 使用稳定版本
)
FetchContent_MakeAvailable(nlohmann_json)

# -----------------------------
# 定义目标（可执行文件）
# -----------------------------
file(GLOB SOURCES
    CONFIGURE_DEPENDS  # 重要：当添加新文件时自动重新配置
    src/*.cpp
)
add_executable(HGRSTAR ${SOURCES})

# -----------------------------
# 设置目标属性和依赖
# -----------------------------

# 链接 LibTorch 与 nlohmann/json
target_link_libraries(HGRSTAR PRIVATE 
    "${TORCH_LIBRARIES}"
    nlohmann_json::nlohmann_json
)

# 包含目录：使用目标范围
target_include_directories(HGRSTAR PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  # 项目内头文件
    # nlohmann/json 已通过 FetchContent 提供，无需额外 include
)

# -----------------------------
# 编译器优化和警告设置
# -----------------------------
# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET HGRSTAR
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:HGRSTAR>)
    # 编译优化
    target_compile_options(HGRSTAR PRIVATE /O2)
    target_compile_options(HGRSTAR PRIVATE /Ob2 /Oi /Ot /Oy /GL)
    target_link_options(HGRSTAR PRIVATE /LTCG)
    # 禁用警告
    message(STATUS "Adding MSVC compile options: /w44503 /wd4267 /wd4819")
    target_compile_options(HGRSTAR PRIVATE /w44503)
    target_compile_options(HGRSTAR PRIVATE /wd4267)
    target_compile_options(HGRSTAR PRIVATE /wd4819)

endif (MSVC)