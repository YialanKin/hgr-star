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
if(MSVC)
    # MSVC 优化
    target_compile_options(HGRSTAR PRIVATE
        /O2                    # 全优化
        /Ob2 /Oi /Ot /Oy /GL   # 高级优化
    )
    target_link_options(HGRSTAR PRIVATE /LTCG)  # 链接时代码生成

    # 禁用特定警告
    target_compile_options(HGRSTAR PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4819>
    )
else()
    # GCC/Clang 优化
    target_compile_options(HGRSTAR PRIVATE
        -O3 -march=native -DNDEBUG
    )
    target_compile_options(HGRSTAR PRIVATE
        -Wall -Wextra -Wpedantic
        # 可选：-Werror
    )
endif()