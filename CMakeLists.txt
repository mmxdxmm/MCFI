cmake_minimum_required(VERSION 3.22)
project(mcfi C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT ANDROID)
    message(FATAL_ERROR "必须使用 Android NDK 工具链编译")
endif()
if(NOT ${ANDROID_ABI} STREQUAL "arm64-v8a")
    message(FATAL_ERROR "本模块仅支持 arm64-v8a")
endif()

# ---- Dobby（内联 hook 框架，静态链接） ----
set(DOBBY_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(DOBBY_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(GENERATE_SHARED OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/Dobby dobby_build)

include_directories(third_party third_party/Dobby/include src)

# ---- Zygisk 模块 so（含入口 + 补帧载荷，最终打包为 zygisk/arm64-v8a.so） ----
add_library(mcfi_zygisk SHARED
    src/entry.cpp
    src/payload.cpp
    src/payload_vk.cpp)
target_link_libraries(mcfi_zygisk dobby_static log EGL GLESv3 vulkan)
set_target_properties(mcfi_zygisk PROPERTIES
    OUTPUT_NAME "arm64-v8a"
    PREFIX ""
    SUFFIX ".so")
target_compile_options(mcfi_zygisk PRIVATE -Wall)
target_link_options(mcfi_zygisk PRIVATE -Wl,--exclude-libs,ALL)
# zygisk so 必须完全自包含：静态链接 C++ 运行时（zygote/zygiskd 没有 libc++_shared.so，
# 动态依赖会导致 dlopen 失败 -> "无法加载模块"）
target_link_options(mcfi_zygisk PRIVATE -static-libstdc++ -Wl,-z,now)

# ---- 守护进程（配置服务 + 控制面板 HTTP） ----
add_executable(mcfid src/daemon.cpp)
target_link_libraries(mcfid log)
set_target_properties(mcfid PROPERTIES OUTPUT_NAME "mcfid")
target_link_options(mcfid PRIVATE -static-libstdc++)
