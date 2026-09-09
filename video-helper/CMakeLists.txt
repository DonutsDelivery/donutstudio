# arbit-video-helper — GPL sidecar binary for Arbit video support.
#
# Links FFmpeg (libav*) and aubio, which must never be linked into the Arbit
# plugin/standalone binaries. All video decode, audio extraction, beat
# detection, thumbnailing, and export happens in this process; Arbit talks to
# it over JSON-RPC (stdin/stdout) + a shared-memory frame ring.
#
# License of this target: GPL-2.0-or-later (source published in-repo).

cmake_minimum_required(VERSION 3.22)
project(arbit-video-helper VERSION 0.1.0 LANGUAGES C CXX)  # C for vendored QuickJS-ng

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../tools/check_renderer_geometry_closure.py
            ${CMAKE_CURRENT_SOURCE_DIR}/..
    RESULT_VARIABLE ARBIT_RENDERER_GEOMETRY_CLOSURE_RESULT)
if(NOT ARBIT_RENDERER_GEOMETRY_CLOSURE_RESULT EQUAL 0)
    message(FATAL_ERROR "Renderer Geometry Core source closure contract failed")
endif()
execute_process(
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../tools/generate_shader_catalog_manifest.py
            ${CMAKE_CURRENT_SOURCE_DIR}/../content/shader-packs
            ${CMAKE_CURRENT_SOURCE_DIR}/../shared/ShaderCatalogManifest.h --check
    RESULT_VARIABLE ARBIT_SHADER_CATALOG_CHECK_RESULT)
if(NOT ARBIT_SHADER_CATALOG_CHECK_RESULT EQUAL 0)
    message(FATAL_ERROR "ShaderCatalogManifest.h is stale; regenerate it before building")
endif()
add_custom_target(arbit-shader-catalog-check ALL
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../tools/generate_shader_catalog_manifest.py
            ${CMAKE_CURRENT_SOURCE_DIR}/../content/shader-packs
            ${CMAKE_CURRENT_SOURCE_DIR}/../shared/ShaderCatalogManifest.h --check
    COMMENT "Checking generated shader catalog manifest"
    VERBATIM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    enable_language(OBJCXX)
    option(ARBIT_WITH_METAL "Enable the native sokol_gfx Metal backend" ON)
endif()

# Hardware accel: the decoder tries NVDEC/VAAPI (Linux), D3D11VA/NVDEC
# (Windows), VideoToolbox (macOS) at runtime and falls back to software.
# When linking a system FFmpeg nothing extra is needed here. When CI builds
# FFmpeg from source (Windows), configure it with nv-codec-headers
# (https://git.videolan.org/git/ffmpeg/nv-codec-headers.git, MIT, headers
# only — NVIDIA libs are dlopen'd from the user's driver) so NVDEC/NVENC
# support is compiled in: --enable-ffnvcodec.

find_package(PkgConfig REQUIRED)
pkg_check_modules(AV REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libavdevice libswscale libswresample libavfilter)
pkg_check_modules(AUBIO IMPORTED_TARGET aubio)

# Per-frame Lua hook (M8 "agent-programmable media machine"). Optional — without
# Lua the helper builds fine and a job's luaScript is silently inert (lua_hook.h
# compiles to no-ops). Lua is MIT, pure-C, no GPL entanglement. Distro packages
# name the .pc variously, so try the common ones.
pkg_check_modules(LUA IMPORTED_TARGET lua5.4)
if(NOT LUA_FOUND)
    pkg_check_modules(LUA IMPORTED_TARGET lua-5.4)
endif()
if(NOT LUA_FOUND)
    pkg_check_modules(LUA IMPORTED_TARGET lua5.3)
endif()
if(NOT LUA_FOUND)
    pkg_check_modules(LUA IMPORTED_TARGET lua)
endif()

find_package(nlohmann_json 3.10 QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Viewport window (GPU-resident preview): GLFW + OpenGL. Optional — without
# it the helper builds headless and the viewport_* methods return an error.
pkg_check_modules(GLFW3 IMPORTED_TARGET glfw3)
find_package(OpenGL QUIET)
option(ARBIT_REQUIRE_VIEWPORT "Fail configure when the production viewport cannot be built" OFF)
if(ARBIT_REQUIRE_VIEWPORT AND NOT (GLFW3_FOUND AND OpenGL_FOUND))
    message(FATAL_ERROR
        "ARBIT_REQUIRE_VIEWPORT=ON requires GLFW3 and OpenGL; refusing a headless shipping helper")
endif()

add_executable(arbit-video-helper
    src/main.cpp
    src/capture_device.cpp
    src/media.cpp
    src/gltf_glb.cpp
    src/animation_clip.cpp
    src/animation_playback_control.cpp
    src/deformation_contract.cpp
    src/animation_deformation_evaluation.cpp
    src/imported_animation_deformation_consumer.cpp
    src/native_animation_deformation.cpp
    src/native_animation_deformation_renderer.cpp
    src/glb_scene_adapter.cpp
    src/glb_native_render_seam.cpp
    src/imported_scene_payload_execution.cpp
    src/imported_animated_scene_payload_execution.cpp
    src/fixture_scene_renderer.cpp
    src/geometry_core_backend.cpp
    src/sdf_native_renderer.cpp
    src/optical_flow_native_executor.cpp
    src/export_visual_plan_session.cpp
    src/exporter_visual_plan_callbacks.cpp
    src/visual_plan_execution_telemetry.cpp
    src/visual_renderer_telemetry.cpp
    src/exporter.cpp
)
add_dependencies(arbit-video-helper arbit-shader-catalog-check)

if(APPLE AND ARBIT_WITH_METAL)
    target_sources(arbit-video-helper PRIVATE src/volume_native_renderer_metal.mm)
elseif(GLFW3_FOUND AND OpenGL_FOUND AND NOT APPLE)
    target_sources(arbit-video-helper PRIVATE src/volume_native_renderer_opengl.cpp)
else()
    target_sources(arbit-video-helper PRIVATE src/volume_native_renderer_stub.cpp)
endif()

# Visual-engine P6 foundation: a language-neutral native-backend seam plus a
# real compute/offscreen-render self-test.  FrameRenderer remains GL-backed
# until the subsequent migration slices route its passes through this seam.
if(APPLE AND ARBIT_WITH_METAL)
    target_sources(arbit-video-helper PRIVATE
        src/gpu_backend/backend_sokol.mm
        src/diffraction_material_metal.mm)
    target_include_directories(arbit-video-helper PRIVATE third_party/sokol)
    target_link_libraries(arbit-video-helper PRIVATE
        "-framework Metal" "-framework QuartzCore" "-framework Cocoa")
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_METAL_BACKEND=1)
    message(STATUS "arbit-video-helper: native sokol_gfx Metal backend enabled")
elseif(GLFW3_FOUND AND OpenGL_FOUND)
    target_sources(arbit-video-helper PRIVATE src/gpu_backend/backend_opengl.cpp)
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_OPENGL_SDF_BACKEND=1)
    message(STATUS "arbit-video-helper: strict native OpenGL SDF backend enabled")
else()
    target_sources(arbit-video-helper PRIVATE src/gpu_backend/backend_stub.cpp)
endif()

if(GLFW3_FOUND AND OpenGL_FOUND)
    target_sources(arbit-video-helper PRIVATE
        src/viewport.cpp
        src/renderer.cpp
        src/color_transform_gl.cpp
        src/shader_generator.cpp
        src/particle_engine.cpp
        src/diffraction_material_gl.cpp
        src/gl_loader.cpp
        src/lut_loader.cpp)
    target_link_libraries(arbit-video-helper PRIVATE PkgConfig::GLFW3 OpenGL::GL)
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_VIEWPORT=1)

    # Zero-copy docked viewport: exports the composited frame as shared GPU
    # buffers over a unix socket (SharedGpuSurfaceProtocol.h). One backend
    # per platform (gpu_export_common.h); actual capability (driver
    # extensions, context flavor) is probed at runtime and
    # viewport_open_shared fails cleanly without it.
    if(APPLE)
        target_sources(arbit-video-helper PRIVATE src/mac_helper_application.mm)
        target_link_libraries(arbit-video-helper PRIVATE "-framework Cocoa")
        if(ARBIT_WITH_METAL)
            # IOSurface-backed native Metal targets (global IOSurfaceIDs — no
            # fd passing). The shared path creates no OpenGL context.
            target_sources(arbit-video-helper PRIVATE
                src/mac_metal_viewport_surface.mm
                src/gpu_export_iosurface.cpp
                src/fd_socket.cpp)
            target_link_libraries(arbit-video-helper PRIVATE
                "-framework IOSurface" "-framework CoreFoundation")
            target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_IOSURFACE=1)
        endif()
    elseif(WIN32)
        # D3D11 keyed-mutex shared textures bridged into GL via
        # WGL_NV_DX_interop2; socket is AF_UNIX SOCK_STREAM (Winsock).
        target_sources(arbit-video-helper PRIVATE
            src/gpu_export_d3d.cpp
            src/fd_socket.cpp)
        target_link_libraries(arbit-video-helper PRIVATE d3d11 ws2_32)
        target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_D3D=1)
    else()
        # Two Linux backends may compile; gpu_export_common.h prefers VkFd.
        #  - VkFd (gpu_export_vk.cpp): Vulkan external memory (OPAQUE_FD) imported
        #    into GL via GL_EXT_memory_object_fd. Works on the NVIDIA proprietary
        #    driver under GLX (where EGL_MESA dmabuf export is unavailable) and is
        #    windowing-agnostic. Links libvulkan (hard dep when enabled, like
        #    ARBIT_NCNN_ZEROCOPY) — set ARBIT_WITH_VKFD=OFF for a vulkan-free build.
        #  - EGL dmabuf (gpu_export.cpp): Mesa/Wayland fallback when VkFd is off.
        # fd_socket.cpp is shared by both.
        option(ARBIT_WITH_VKFD "Zero-copy docked viewport via Vulkan OPAQUE_FD (Linux)" ON)
        set(_arbit_shared_any OFF)
        find_library(ARBIT_VKFD_LOADER NAMES vulkan vulkan-1)
        if(ARBIT_WITH_VKFD AND ARBIT_VKFD_LOADER)
            # The system ships no Vulkan headers (only the loader); fetch the same
            # pinned headers RIFE uses. FetchContent honors the first declare, so
            # the later ncnn-zerocopy block reuses this populated copy.
            include(FetchContent)
            FetchContent_Declare(VulkanHeaders
                GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers
                GIT_TAG d2d8ded679e53c9f27c0c7a18750e661745886eb) # pinned (Vulkan 1.4.x)
            FetchContent_MakeAvailable(VulkanHeaders)
            target_sources(arbit-video-helper PRIVATE src/gpu_export_vk.cpp)
            target_include_directories(arbit-video-helper PRIVATE ${vulkanheaders_SOURCE_DIR}/include)
            target_link_libraries(arbit-video-helper PRIVATE ${ARBIT_VKFD_LOADER})
            target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_VKFD=1)
            set(_arbit_shared_any ON)
        endif()
        pkg_check_modules(EGL IMPORTED_TARGET egl)
        if(EGL_FOUND)
            target_sources(arbit-video-helper PRIVATE src/gpu_export.cpp)
            target_link_libraries(arbit-video-helper PRIVATE PkgConfig::EGL)
            target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_DMABUF=1)
            set(_arbit_shared_any ON)
        endif()
        if(_arbit_shared_any)
            target_sources(arbit-video-helper PRIVATE src/fd_socket.cpp)
        endif()
    endif()
endif()

# RIFE optical-flow frame interpolation ("speed warp"): ONNX Runtime.
# Strictly optional and OFF by default — CI and packagers build without it
# and the helper degrades to nearest-frame (interpolation "auto") or reports
# "RIFE unavailable" (interpolation "rife"). The RIFE model binary is NOT
# part of the build; the helper downloads it on first use (see src/rife.cpp
# and PROTOCOL.md §RIFE).
option(ARBIT_WITH_ONNX "Enable RIFE frame interpolation via ONNX Runtime" OFF)
set(ARBIT_ONNXRUNTIME_ROOT "" CACHE PATH
    "Existing ONNX Runtime install (include/ + lib/); skips the prebuilt download")
if(ARBIT_WITH_ONNX)
    set(ARBIT_ORT_VERSION 1.26.0)
    if(ARBIT_ONNXRUNTIME_ROOT)
        set(ARBIT_ORT_ROOT "${ARBIT_ONNXRUNTIME_ROOT}")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux"
           AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
        # Official prebuilt, GPU flavor (CUDA 13 EP + CPU EP in one package).
        # The CUDA provider is dlopen'd at runtime; machines without
        # CUDA/cuDNN fall back to the CPU EP inside RifeEngine::init().
        include(FetchContent)
        FetchContent_Declare(onnxruntime_prebuilt
            URL https://github.com/microsoft/onnxruntime/releases/download/v${ARBIT_ORT_VERSION}/onnxruntime-linux-x64-gpu_cuda13-${ARBIT_ORT_VERSION}.tgz
            URL_HASH SHA256=aa619d5701bbe58046cc998b21e692d5b2aefac1479f375c4b988526cb80befa
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(onnxruntime_prebuilt)
        set(ARBIT_ORT_ROOT "${onnxruntime_prebuilt_SOURCE_DIR}")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin"
           AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
        # Official prebuilt for Apple Silicon. The Core ML EP is compiled into
        # the dylib (no separate provider lib), so RifeEngine can append it at
        # runtime on __APPLE__. Realtime tier 2 on macOS runs on the ANE / GPU.
        include(FetchContent)
        FetchContent_Declare(onnxruntime_prebuilt
            URL https://github.com/microsoft/onnxruntime/releases/download/v${ARBIT_ORT_VERSION}/onnxruntime-osx-arm64-${ARBIT_ORT_VERSION}.tgz
            URL_HASH SHA256=7a1280bbb1701ea514f71828765237e7896e0f2e1cd332f1f70dbd5c3e33aca3
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(onnxruntime_prebuilt)
        set(ARBIT_ORT_ROOT "${onnxruntime_prebuilt_SOURCE_DIR}")
    else()
        message(FATAL_ERROR "ARBIT_WITH_ONNX=ON: no prebuilt wired for this "
                            "platform — set ARBIT_ONNXRUNTIME_ROOT to an "
                            "ONNX Runtime install")
    endif()
    find_library(ARBIT_ORT_LIBRARY NAMES onnxruntime
                 HINTS "${ARBIT_ORT_ROOT}/lib" REQUIRED)
    # rife.cpp = the RIFE ONNX engine (shared by export + viewport). interp.cpp (the
    # live-viewport async RIFE worker, retime tier 2) is added below for EITHER backend.
    target_sources(arbit-video-helper PRIVATE src/rife.cpp)
    target_include_directories(arbit-video-helper PRIVATE "${ARBIT_ORT_ROOT}/include")
    target_link_libraries(arbit-video-helper PRIVATE "${ARBIT_ORT_LIBRARY}")
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_ONNX=1)
    # Dev/run-in-place: resolve libonnxruntime.so* from the fetched package.
    # Deployment ships the ORT libs next to the helper binary ($ORIGIN).
    set_target_properties(arbit-video-helper PROPERTIES
        BUILD_RPATH "${ARBIT_ORT_ROOT}/lib"
        INSTALL_RPATH "$ORIGIN;${ARBIT_ORT_ROOT}/lib")
endif()

# --- Bundled Vulkan loader discovery (Linux) -------------------------------
# The Linux release bundle ships libvulkan.so.1 next to the helper so the helper
# starts even on machines with NO system Vulkan loader — it then finds no GPU and
# degrades to the shm present path + Frame Blend, so basic (non-interpolated)
# video editing always works (the core pipeline is GL + FFmpeg, not Vulkan).
# $ORIGIN makes the helper find that bundled loader; it is harmless when absent
# (the dynamic linker falls through to the system paths) and the loader is a thin
# dispatch shim that still loads the user's own GPU driver/ICD at runtime.
# Deployment copies the build-tree binary directly (no cmake --install step), so
# bake the rpath into the build tree via BUILD_WITH_INSTALL_RPATH. ncnn is static
# and the helper has no other build-tree shared deps besides ORT (whose path stays
# in INSTALL_RPATH above), so this is safe.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_property(TARGET arbit-video-helper APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
    set_target_properties(arbit-video-helper PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# RIFE frame interpolation, portable path: ncnn + Vulkan ("rife-vulkan", tier 4).
# Unlike the ONNX/CUDA path above (NVIDIA-only, OFF by default), this runs on any
# conformant Vulkan driver (AMD/Intel/NVIDIA/Apple-MoltenVK) from one binary, so
# it ships ON by default — the flag exists only to allow a stripped build.
#
# Net source is vendored (src/rife_net/, nihui/rife-ncnn-vulkan, MIT) and ncnn
# (Tencent, BSD-3) is fetched. ncnn 20250503 is PINNED on purpose: it is the
# newest release that still defines the afpvec8 packed-fp16 shader type +
# Option::use_shader_pack8 that the 2022-era net source uses (both removed by
# ncnn 20251225). See artifacts/planning/missing-features/plans/media-machine/rife-ncnn-vulkan-plan.md.
# ncnn's SIMPLEVK path expects a Vulkan loader symbol at link time on macOS.
# MoltenVK satisfies that (see Apple branch below). Default ON everywhere so
# the shipped helper includes Speed Warp; configure -DARBIT_WITH_NCNN=OFF to
# strip RIFE. macOS CI installs molten-vk via Homebrew and asserts rife-vulkan.
set(_arbit_ncnn_default ON)
option(ARBIT_WITH_NCNN "Enable portable RIFE frame interpolation via ncnn + Vulkan" ${_arbit_ncnn_default})
# Zero-copy GPU RIFE output (Phase 1b): keep the RIFE result on the GPU and export
# it to OpenGL via OPAQUE_FD instead of bouncing it through CPU. Needs the REAL
# Vulkan headers (ncnn's in-house simplevk.h lacks the external-memory structs) and
# the Vulkan loader at link time, and makes libvulkan a HARD runtime dependency
# (vs the SIMPLEVK soft-dlopen). The export path itself (OPAQUE_FD -> GL) is
# Linux-specific (macOS would need IOSurface, Windows DXGI — not built). It remains
# fail-closed until ncnn exposes a legal asynchronous submission API; the portable
# synchronous CPU-readback RIFE path stays available.
set(_arbit_zerocopy_default OFF)
option(ARBIT_NCNN_ZEROCOPY "Zero-copy GPU RIFE output (Vulkan->GL OPAQUE_FD, Linux)" ${_arbit_zerocopy_default})
if(ARBIT_NCNN_ZEROCOPY)
    message(FATAL_ERROR
        "ARBIT_NCNN_ZEROCOPY is blocked: pinned ncnn 20250503 exposes only "
        "VkCompute::submit_and_wait(), so external wait/signal semaphores and "
        "per-submission completion cannot be attached legally. See "
        "artifacts/planning/research-only/media-machine/rife-ncnn-async-submission-blocker.md.")
endif()
if(ARBIT_WITH_NCNN)
    include(FetchContent)
    # ncnn build knobs: Vulkan on, static, no extras, bundle+build glslang.
    set(NCNN_VULKAN ON CACHE BOOL "" FORCE)
    set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(NCNN_SHARED_LIB OFF CACHE BOOL "" FORCE)
    set(NCNN_INSTALL_SDK OFF CACHE BOOL "" FORCE)
    set(NCNN_SYSTEM_GLSLANG OFF CACHE BOOL "" FORCE)

    if(ARBIT_NCNN_ZEROCOPY)
        # SIMPLEVK off => ncnn includes <vulkan/vulkan.h> (real external-memory
        # structs) and links the loader. Provide both before configuring ncnn.
        set(NCNN_SIMPLEVK OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(VulkanHeaders
            GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers
            GIT_TAG d2d8ded679e53c9f27c0c7a18750e661745886eb) # pinned (Vulkan 1.4.x)
        FetchContent_MakeAvailable(VulkanHeaders)
        find_library(ARBIT_VULKAN_LOADER NAMES vulkan vulkan-1)
        if(NOT ARBIT_VULKAN_LOADER)
            message(FATAL_ERROR
                "ARBIT_NCNN_ZEROCOPY=ON needs the Vulkan loader (libvulkan). Install "
                "vulkan-loader / libvulkan-dev, or configure -DARBIT_NCNN_ZEROCOPY=OFF "
                "for the CPU-readback-only RIFE path.")
        endif()
        set(Vulkan_INCLUDE_DIR ${vulkanheaders_SOURCE_DIR}/include CACHE PATH "" FORCE)
        set(Vulkan_LIBRARY ${ARBIT_VULKAN_LOADER} CACHE FILEPATH "" FORCE)
    elseif(APPLE)
        # ncnn's SimpleVK uses STATIC link-time Vulkan linkage on Apple platforms
        # (unlike the dlopen path used on Linux/Windows/Android) — see ncnn's
        # src/CMakeLists.txt "simplevk always use static vulkan linkage on apple
        # platform". If Vulkan_LIBRARY isn't defined there, ncnn's own CMake only
        # WARNS and links nothing, so the helper would build fine but never find a
        # GPU. MoltenVK is itself a complete Vulkan-API implementation (not just a
        # loader), so linking it directly satisfies vkGetInstanceProcAddr et al.
        # with no separate loader / ICD-JSON discovery step needed.
        find_library(ARBIT_MOLTENVK_LIBRARY NAMES MoltenVK)
        if(NOT ARBIT_MOLTENVK_LIBRARY)
            message(FATAL_ERROR
                "ARBIT_WITH_NCNN=ON needs MoltenVK on macOS (brew install molten-vk), "
                "or configure -DARBIT_WITH_NCNN=OFF to build without RIFE support.")
        endif()
        set(Vulkan_LIBRARY ${ARBIT_MOLTENVK_LIBRARY} CACHE FILEPATH "" FORCE)
    endif()

    # ncnn 20250503 only enables base VK_KHR_external_memory; the PATCH_COMMAND adds
    # the fd-based variants needed to export to GL (idempotent, inert unless the ncnn
    # object is compiled with ARBIT_RIFE_ZEROCOPY — see below).
    FetchContent_Declare(ncnn
        URL https://github.com/Tencent/ncnn/releases/download/20250503/ncnn-20250503-full-source.zip
        URL_HASH SHA256=b3190d1b6beae3aac30fa99f4d3fd89a2b47a010b888d2d2389189784743a690
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        PATCH_COMMAND ${CMAKE_COMMAND} -DNCNN_GPU_CPP=src/gpu.cpp
                      -P ${CMAKE_CURRENT_SOURCE_DIR}/patches/ncnn_enable_external_memory_fd.cmake)
    # ncnn's vendored glslang declares a pre-3.5 cmake_minimum_required; allow it
    # under CMake 4.x (harmless on older CMake).
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    FetchContent_MakeAvailable(ncnn)

    # GCC 13 can ICE in cfgcleanup while compiling ncnn's generated AVX packing
    # unit with ncnn's own -Ofast flags (GCC PR-style internal failure, not a
    # source diagnostic). Override that dependency target to -O2 only on the
    # affected compiler generation. GNU 12/14+, Clang, and DonutStudio's own
    # targets retain their existing release optimization settings.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
       AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13
       AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
        target_compile_options(ncnn PRIVATE -O2)
    endif()

    if(ARBIT_NCNN_ZEROCOPY)
        # activates the gpu.cpp patch block
        target_compile_definitions(ncnn PRIVATE ARBIT_RIFE_ZEROCOPY)
    endif()

    # Hex-encode the vendored net's custom-layer GLSL shaders into headers the
    # net source #includes (ncnn compiles them to SPIR-V online via glslang).
    set(ARBIT_RIFE_NET ${CMAKE_CURRENT_SOURCE_DIR}/src/rife_net)
    set(ARBIT_RIFE_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/rife_shaders)
    file(MAKE_DIRECTORY ${ARBIT_RIFE_SHADER_DIR})
    set(ARBIT_RIFE_SHADER_HEADERS)
    foreach(s
        rife_preproc rife_postproc rife_preproc_tta rife_postproc_tta
        rife_flow_tta_avg rife_v2_flow_tta_avg rife_v4_flow_tta_avg
        rife_flow_tta_temporal_avg rife_v2_flow_tta_temporal_avg rife_v4_flow_tta_temporal_avg
        rife_out_tta_temporal_avg rife_v4_timestep rife_v4_timestep_tta
        warp warp_pack4 warp_pack8)
        set(_hdr ${ARBIT_RIFE_SHADER_DIR}/${s}.comp.hex.h)
        add_custom_command(
            OUTPUT ${_hdr}
            COMMAND ${CMAKE_COMMAND}
                    -DSHADER_SRC=${ARBIT_RIFE_NET}/${s}.comp
                    -DSHADER_COMP_HEADER=${_hdr}
                    -P ${ARBIT_RIFE_NET}/generate_shader_comp_header.cmake
            DEPENDS ${ARBIT_RIFE_NET}/${s}.comp
            WORKING_DIRECTORY ${ARBIT_RIFE_SHADER_DIR}
            COMMENT "hex shader ${s}.comp"
            VERBATIM)
        list(APPEND ARBIT_RIFE_SHADER_HEADERS ${_hdr})
    endforeach()
    add_custom_target(arbit-rife-shaders DEPENDS ${ARBIT_RIFE_SHADER_HEADERS})

    target_sources(arbit-video-helper PRIVATE
        ${ARBIT_RIFE_NET}/rife.cpp
        ${ARBIT_RIFE_NET}/warp.cpp
        src/rife_ncnn.cpp)
    add_dependencies(arbit-video-helper arbit-rife-shaders)
    target_include_directories(arbit-video-helper PRIVATE
        ${ARBIT_RIFE_NET} ${ARBIT_RIFE_SHADER_DIR})
    target_link_libraries(arbit-video-helper PRIVATE ncnn)
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_NCNN=1)

    add_executable(rife-ncnn-inference-tests
        tests/rife_ncnn_inference_tests.cpp
        ${ARBIT_RIFE_NET}/rife.cpp
        ${ARBIT_RIFE_NET}/warp.cpp
        src/rife_ncnn.cpp)
    add_dependencies(rife-ncnn-inference-tests arbit-rife-shaders)
    target_include_directories(rife-ncnn-inference-tests PRIVATE
        src ${ARBIT_RIFE_NET} ${ARBIT_RIFE_SHADER_DIR})
    target_link_libraries(rife-ncnn-inference-tests PRIVATE ncnn PkgConfig::AV)
    target_compile_definitions(rife-ncnn-inference-tests PRIVATE ARBIT_HAVE_NCNN=1)
    if(NOT MSVC)
        target_compile_options(rife-ncnn-inference-tests PRIVATE -Wall -Wextra)
    endif()
    if(ARBIT_NCNN_ZEROCOPY)
        target_compile_definitions(rife-ncnn-inference-tests PRIVATE ARBIT_RIFE_ZEROCOPY=1)
        target_link_libraries(rife-ncnn-inference-tests PRIVATE Vulkan::Headers)
        # rife_ncnn.cpp's ExportableRing + rife.cpp's process_v4_to_buffer include
        # <vulkan/vulkan.h> (via ncnn's PUBLIC Vulkan::Vulkan) and use the external-
        # memory API. The GL import side is loaded at runtime via the viewport's GL
        # context (no extra link).
        target_compile_definitions(arbit-video-helper PRIVATE ARBIT_RIFE_ZEROCOPY=1)
        target_link_libraries(arbit-video-helper PRIVATE Vulkan::Headers)
    endif()
endif()

# The live-viewport async RIFE worker is backend-agnostic (picks ncnn-Vulkan when
# built, else the ONNX engine), so it compiles whenever EITHER backend is enabled.
if(ARBIT_WITH_NCNN OR ARBIT_WITH_ONNX)
    target_sources(arbit-video-helper PRIVATE src/interp.cpp)
endif()

target_include_directories(arbit-video-helper PRIVATE
    # VideoFrameSharedMemory.h is shared with the plugin (dependency-free header)
    ${CMAKE_CURRENT_SOURCE_DIR}/../plugin/Source
    # Typed 3D scene contracts are shared by the GLB adapter and renderer seam.
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared
)

target_link_libraries(arbit-video-helper PRIVATE
    PkgConfig::AV
    nlohmann_json::nlohmann_json
)

if(AUBIO_FOUND)
    target_link_libraries(arbit-video-helper PRIVATE PkgConfig::AUBIO)
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_AUBIO=1)
endif()

if(LUA_FOUND)
    target_link_libraries(arbit-video-helper PRIVATE PkgConfig::LUA)
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_LUA=1)
    message(STATUS "arbit-video-helper: Lua per-frame hook enabled (lua ${LUA_VERSION})")
else()
    message(STATUS "arbit-video-helper: Lua not found — per-frame hook disabled")
endif()

# Per-frame JavaScript hook (visual-engine P2 — QuickJS-ng front-end). Vendored
# under third_party/quickjs/ (MIT, single-tree, no external deps) so it always
# builds where the source is present; js_hook.h compiles to no-ops when
# ARBIT_HAVE_QUICKJS is off. JavaScript is the creative-coding + AI-agent lingua
# franca — Shadertoy-JS / generative-JS authoring with no browser. The compiled
# library is third-party C: warnings silenced so it never trips the helper flags.
set(ARBIT_QUICKJS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/quickjs)
if(EXISTS ${ARBIT_QUICKJS_DIR}/quickjs.c)
    add_library(arbit-quickjs STATIC
        ${ARBIT_QUICKJS_DIR}/quickjs.c
        ${ARBIT_QUICKJS_DIR}/cutils.c
        ${ARBIT_QUICKJS_DIR}/libregexp.c
        ${ARBIT_QUICKJS_DIR}/libunicode.c
        ${ARBIT_QUICKJS_DIR}/xsum.c)
    target_include_directories(arbit-quickjs PUBLIC ${ARBIT_QUICKJS_DIR})
    set_target_properties(arbit-quickjs PROPERTIES
        C_STANDARD 11 POSITION_INDEPENDENT_CODE ON)
    if(NOT MSVC)
        target_compile_options(arbit-quickjs PRIVATE -w)
    endif()
    if(NOT WIN32)
        target_link_libraries(arbit-quickjs PUBLIC m)
    endif()
    target_link_libraries(arbit-video-helper PRIVATE arbit-quickjs)
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_QUICKJS=1)
    message(STATUS "arbit-video-helper: QuickJS-ng per-frame JS hook enabled")
else()
    message(STATUS "arbit-video-helper: QuickJS-ng not vendored — JS hook disabled")
endif()

# SPIR-V → GLSL ingestion (visual-engine P3 — the "universal adapter"). Vendored
# SPIRV-Cross (third_party/spirv_cross/, Apache-2.0, GLSL backend only). Slang,
# HLSL and GLSL all compile to SPIR-V (slangc/dxc/glslang), so this one backend
# makes the helper language-agnostic. Gated: shader_lang.h returns a clean
# diagnostic without it. Third-party C++ → warnings silenced.
set(ARBIT_SPVCROSS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/spirv_cross)
if(EXISTS ${ARBIT_SPVCROSS_DIR}/spirv_glsl.cpp)
    add_library(arbit-spirv-cross STATIC
        ${ARBIT_SPVCROSS_DIR}/spirv_cross.cpp
        ${ARBIT_SPVCROSS_DIR}/spirv_glsl.cpp
        ${ARBIT_SPVCROSS_DIR}/spirv_cfg.cpp
        ${ARBIT_SPVCROSS_DIR}/spirv_parser.cpp
        ${ARBIT_SPVCROSS_DIR}/spirv_cross_parsed_ir.cpp
        ${ARBIT_SPVCROSS_DIR}/spirv_cross_util.cpp)
    target_include_directories(arbit-spirv-cross PUBLIC ${ARBIT_SPVCROSS_DIR})
    set_target_properties(arbit-spirv-cross PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(NOT MSVC)
        target_compile_options(arbit-spirv-cross PRIVATE -w)
    endif()
    target_link_libraries(arbit-video-helper PRIVATE arbit-spirv-cross)
    target_include_directories(arbit-video-helper PRIVATE ${ARBIT_SPVCROSS_DIR})
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_SPIRV_CROSS=1)
    message(STATUS "arbit-video-helper: SPIR-V ingestion enabled (SPIRV-Cross)")
else()
    message(STATUS "arbit-video-helper: SPIRV-Cross not vendored — SPIR-V ingestion disabled")
endif()

# Strict macOS generator path: compile the same wrapped GLSL contract to SPIR-V
# with glslang, convert it to MSL in-process, and render into private Metal
# textures sampled directly by the compositor. MoltenVK's converter is static,
# so the installed helper does not launch or depend on a developer tool.
if(APPLE AND ARBIT_WITH_METAL AND GLFW3_FOUND AND OpenGL_FOUND)
    if(NOT ARBIT_WITH_NCNN)
        message(FATAL_ERROR "The native Metal generator path requires bundled glslang (ARBIT_WITH_NCNN=ON)")
    endif()
    find_path(ARBIT_MVK_SHADER_CONVERTER_INCLUDE
        NAMES MoltenVKShaderConverter/SPIRVConversion.h)
    find_library(ARBIT_MVK_SHADER_CONVERTER_LIBRARY
        NAMES MoltenVKShaderConverter
        PATH_SUFFIXES
            Frameworks/MoltenVKShaderConverter.xcframework/macos-arm64
            Frameworks/MoltenVKShaderConverter.xcframework/macos-x86_64)
    find_program(ARBIT_MVK_SHADER_CONVERTER_TOOL NAMES MoltenVKShaderConverter)
    if(ARBIT_MVK_SHADER_CONVERTER_TOOL)
        get_filename_component(ARBIT_MVK_SHADER_CONVERTER_TOOL_REAL
            ${ARBIT_MVK_SHADER_CONVERTER_TOOL} REALPATH)
        get_filename_component(ARBIT_MVK_BIN_DIR
            ${ARBIT_MVK_SHADER_CONVERTER_TOOL_REAL} DIRECTORY)
        get_filename_component(ARBIT_MVK_PREFIX ${ARBIT_MVK_BIN_DIR} DIRECTORY)
    endif()
    find_library(ARBIT_MVK_SPIRV_CROSS_LIBRARY NAMES SPIRVCross
        HINTS ${ARBIT_MVK_PREFIX}/libexec/lib
        PATH_SUFFIXES libexec/lib)
    find_library(ARBIT_MVK_SPIRV_TOOLS_LIBRARY NAMES SPIRVTools
        HINTS ${ARBIT_MVK_PREFIX}/libexec/lib
        PATH_SUFFIXES libexec/lib)
    if(NOT ARBIT_MVK_SHADER_CONVERTER_INCLUDE
       OR NOT ARBIT_MVK_SHADER_CONVERTER_LIBRARY
       OR NOT ARBIT_MVK_SPIRV_CROSS_LIBRARY
       OR NOT ARBIT_MVK_SPIRV_TOOLS_LIBRARY)
        message(FATAL_ERROR
            "Strict Metal generators require MoltenVKShaderConverter and its static SPIR-V libraries")
    endif()

    target_sources(arbit-video-helper PRIVATE src/gpu_backend/metal_shader_generator.mm)
    target_include_directories(arbit-video-helper PRIVATE
        ${ARBIT_MVK_SHADER_CONVERTER_INCLUDE}
        ${ncnn_SOURCE_DIR}/glslang)
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_METAL_GENERATORS=1)
    target_link_libraries(arbit-video-helper PRIVATE
        glslang SPIRV glslang-default-resource-limits
        ${ARBIT_MVK_SHADER_CONVERTER_LIBRARY}
        ${ARBIT_MVK_SPIRV_CROSS_LIBRARY}
        ${ARBIT_MVK_SPIRV_TOOLS_LIBRARY}
        "-framework Foundation")
endif()

if(WIN32)
    # windows.h (pulled in by VideoFrameSharedMemory.h / gl_loader.h) defines
    # min/max macros that break std::min/std::max.
    target_compile_definitions(arbit-video-helper PRIVATE NOMINMAX)

    # FFmpeg, zimg, and the rest of the DLL closure come from MSYS2. Keep the
    # helper's own GCC C++ and GCC support runtimes inside the executable so a
    # different libstdc++ ABI cannot be selected to satisfy both the Ubuntu
    # cross-compiler and the MSYS2 dependency graph at process startup.
    target_link_options(arbit-video-helper PRIVATE -static-libstdc++ -static-libgcc)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(arbit-video-helper PRIVATE rt)   # shm_open (Linux only; macOS has it in libSystem)
endif()
if(NOT WIN32)
    target_link_libraries(arbit-video-helper PRIVATE pthread)
endif()

# --- Tests (dependency-free) -------------------------------------------------
# The shared render-graph param grammar (Stage 6) is header-only and links nothing
# (no FFmpeg/GL/Lua), so its unit test builds and runs anywhere — the guard-rail that
# replaced the old "KEEP IN SYNC" comments across exporter.cpp / viewport.cpp.
enable_testing()
add_executable(arbit-geometry-core-tests tests/geometry_core_admission_tests.cpp)
target_include_directories(arbit-geometry-core-tests PRIVATE src ../shared)
target_compile_features(arbit-geometry-core-tests PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(arbit-geometry-core-tests PRIVATE /W4 /WX)
else()
    target_compile_options(arbit-geometry-core-tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
add_test(NAME geometry_core_admission COMMAND arbit-geometry-core-tests)

add_executable(arbit-video-param-tests tests/param_grammar_tests.cpp)
target_include_directories(arbit-video-param-tests PRIVATE src)
if(NOT MSVC)
    target_compile_options(arbit-video-param-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME video_param_grammar COMMAND arbit-video-param-tests)

add_executable(arbit-beat-timeline-tests tests/beat_timeline_tests.cpp)
target_include_directories(arbit-beat-timeline-tests PRIVATE src)
if(NOT MSVC)
    target_compile_options(arbit-beat-timeline-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME beat_timeline COMMAND arbit-beat-timeline-tests)

add_executable(arbit-video-control-plan-tests tests/video_control_plan_tests.cpp)
target_include_directories(arbit-video-control-plan-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(arbit-video-control-plan-tests PRIVATE cxx_std_17)
if(MSVC)
  target_compile_options(arbit-video-control-plan-tests PRIVATE /W4)
else()
  target_compile_options(arbit-video-control-plan-tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME video_control_plan COMMAND arbit-video-control-plan-tests)

add_test(NAME depth_primitives_source_contract
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/depth_primitives_source_contract.py)
add_test(NAME renderer_geometry_closure_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../tools/check_renderer_geometry_closure.py
            ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_test(NAME fixture_scene_culling_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixture_scene_culling_source_contract.py)
add_test(NAME bundled_shader_coordinate_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/bundled_shader_coordinate_contract.py
            ${CMAKE_CURRENT_SOURCE_DIR}/../content/shader-packs)
add_test(NAME rife_ncnn_async_submission_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/rife_ncnn_async_submission_source_contract.py
            ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(arbit-source-binding-tests tests/source_binding_tests.cpp)
target_include_directories(arbit-source-binding-tests PRIVATE src)
if(NOT MSVC)
    target_compile_options(arbit-source-binding-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME source_binding COMMAND arbit-source-binding-tests)

add_executable(arbit-canonical-block-c-frame-identity-tests
    tests/canonical_block_c_frame_identity_tests.cpp)
target_include_directories(arbit-canonical-block-c-frame-identity-tests PRIVATE src)
target_link_libraries(arbit-canonical-block-c-frame-identity-tests PRIVATE
    nlohmann_json::nlohmann_json)
target_compile_features(arbit-canonical-block-c-frame-identity-tests PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(arbit-canonical-block-c-frame-identity-tests PRIVATE /W4 /WX)
else()
    target_compile_options(arbit-canonical-block-c-frame-identity-tests PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
endif()
add_test(NAME canonical_block_c_frame_identity
    COMMAND arbit-canonical-block-c-frame-identity-tests)

add_executable(arbit-score-renderer-tests tests/score_renderer_tests.cpp)
target_include_directories(arbit-score-renderer-tests PRIVATE src)
if(NOT MSVC)
    target_compile_options(arbit-score-renderer-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME score_renderer COMMAND arbit-score-renderer-tests)

add_executable(arbit-shader-operation-plan-tests tests/shader_operation_plan_tests.cpp)
target_include_directories(arbit-shader-operation-plan-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
target_link_libraries(arbit-shader-operation-plan-tests PRIVATE nlohmann_json::nlohmann_json)
if(NOT MSVC)
    target_compile_options(arbit-shader-operation-plan-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME shader_operation_plan COMMAND arbit-shader-operation-plan-tests)

add_executable(arbit-render-snapshot-tests tests/render_snapshot_tests.cpp)
target_include_directories(arbit-render-snapshot-tests PRIVATE src)
target_link_libraries(arbit-render-snapshot-tests PRIVATE nlohmann_json::nlohmann_json)
if(NOT MSVC)
    target_compile_options(arbit-render-snapshot-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME render_snapshot COMMAND arbit-render-snapshot-tests)

add_executable(arbit-gltf-glb-tests
    tests/gltf_glb_tests.cpp
    src/gltf_glb.cpp
    src/animation_clip.cpp
    src/animation_playback_control.cpp
    src/deformation_contract.cpp
    src/animation_deformation_evaluation.cpp
    src/imported_animation_deformation_consumer.cpp)
target_include_directories(arbit-gltf-glb-tests PRIVATE src)
target_link_libraries(arbit-gltf-glb-tests PRIVATE
    nlohmann_json::nlohmann_json
    PkgConfig::AV)
if(NOT MSVC)
    target_compile_options(arbit-gltf-glb-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME gltf_glb_admission COMMAND arbit-gltf-glb-tests)

add_executable(arbit-glb-scene-adapter-tests
    tests/glb_scene_adapter_tests.cpp
    src/glb_scene_adapter.cpp)
target_include_directories(arbit-glb-scene-adapter-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-glb-scene-adapter-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME glb_scene_adapter COMMAND arbit-glb-scene-adapter-tests)

add_executable(arbit-holographic-trading-card-asset-tests
    tests/holographic_trading_card_asset_tests.cpp
    tests/support/fixture_scene.cpp
    src/gltf_glb.cpp
    src/glb_scene_adapter.cpp
    src/animation_clip.cpp
    src/animation_playback_control.cpp
    src/deformation_contract.cpp
    src/animation_deformation_evaluation.cpp
    src/imported_animation_deformation_consumer.cpp)
target_include_directories(arbit-holographic-trading-card-asset-tests PRIVATE
    src tests ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
target_link_libraries(arbit-holographic-trading-card-asset-tests PRIVATE
    nlohmann_json::nlohmann_json PkgConfig::AV)
if(NOT MSVC)
    target_compile_options(arbit-holographic-trading-card-asset-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME holographic_trading_card_asset
    COMMAND arbit-holographic-trading-card-asset-tests)

add_executable(arbit-glb-native-render-seam-tests
    tests/glb_native_render_seam_tests.cpp
    src/gltf_glb.cpp
    src/glb_scene_adapter.cpp
    src/glb_native_render_seam.cpp
    src/imported_scene_payload_execution.cpp
    src/imported_animated_scene_payload_execution.cpp
    src/imported_animation_deformation_consumer.cpp
    src/fixture_scene_renderer.cpp
    src/animation_clip.cpp
    src/animation_playback_control.cpp
    src/animation_deformation_evaluation.cpp
    src/deformation_contract.cpp
    src/native_animation_deformation.cpp
    src/native_animation_deformation_renderer.cpp
    src/gpu_backend/backend_stub.cpp)
target_include_directories(arbit-glb-native-render-seam-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
target_link_libraries(arbit-glb-native-render-seam-tests PRIVATE
    nlohmann_json::nlohmann_json
    PkgConfig::AV)
if(NOT MSVC)
    target_compile_options(arbit-glb-native-render-seam-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME glb_native_render_seam COMMAND arbit-glb-native-render-seam-tests)

add_executable(arbit-model-payload-transport-tests
    tests/model_payload_transport_tests.cpp)
target_include_directories(arbit-model-payload-transport-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-model-payload-transport-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME model_payload_transport COMMAND arbit-model-payload-transport-tests)

add_executable(arbit-model-cache-contract-tests
    tests/model_cache_contract_tests.cpp)
target_include_directories(arbit-model-cache-contract-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-model-cache-contract-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME model_cache_contract COMMAND arbit-model-cache-contract-tests)

add_executable(arbit-animation-clip-tests
    tests/animation_clip_tests.cpp
    src/animation_clip.cpp)
target_include_directories(arbit-animation-clip-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-animation-clip-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME animation_clip_sampling COMMAND arbit-animation-clip-tests)

add_executable(arbit-animation-mix-tests
    tests/animation_mix_tests.cpp
    src/animation_mix.cpp
    src/animation_clip.cpp)
target_include_directories(arbit-animation-mix-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-animation-mix-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME animation_crossfade COMMAND arbit-animation-mix-tests)

add_executable(arbit-animation-playback-control-tests
    tests/animation_playback_control_tests.cpp
    src/animation_playback_control.cpp
    src/animation_clip.cpp)
target_include_directories(arbit-animation-playback-control-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-animation-playback-control-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME animation_playback_control
    COMMAND arbit-animation-playback-control-tests)

add_executable(arbit-animation-deformation-evaluation-tests
    tests/animation_deformation_evaluation_tests.cpp
    src/animation_deformation_evaluation.cpp
    src/animation_clip.cpp
    src/deformation_contract.cpp)
target_include_directories(arbit-animation-deformation-evaluation-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-animation-deformation-evaluation-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME animation_deformation_evaluation
    COMMAND arbit-animation-deformation-evaluation-tests)

add_executable(arbit-native-animation-deformation-tests
    tests/native_animation_deformation_tests.cpp
    tests/support/fixture_scene.cpp
    src/gltf_glb.cpp
    src/glb_scene_adapter.cpp
    src/native_animation_deformation.cpp
    src/native_animation_deformation_renderer.cpp
    src/animation_deformation_evaluation.cpp
    src/animation_clip.cpp
    src/deformation_contract.cpp
    src/gpu_backend/backend_stub.cpp)
target_include_directories(arbit-native-animation-deformation-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
target_link_libraries(arbit-native-animation-deformation-tests PRIVATE
    nlohmann_json::nlohmann_json PkgConfig::AV)
if(NOT MSVC)
    target_compile_options(arbit-native-animation-deformation-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME native_animation_deformation
    COMMAND arbit-native-animation-deformation-tests)

add_executable(arbit-sdf-ir-admission-tests tests/sdf_ir_admission_tests.cpp)
target_include_directories(arbit-sdf-ir-admission-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
target_compile_definitions(arbit-sdf-ir-admission-tests PRIVATE
    SDF_SCENE_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../content/shader-packs")
if(NOT MSVC)
    target_compile_options(arbit-sdf-ir-admission-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME sdf_ir_admission COMMAND arbit-sdf-ir-admission-tests)

add_executable(arbit-sdf-reference-evaluator-tests
    tests/sdf_reference_evaluator_tests.cpp
    tests/support/sdf_reference_evaluator.cpp)
target_include_directories(arbit-sdf-reference-evaluator-tests PRIVATE
    src
    tests
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-sdf-reference-evaluator-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME sdf_reference_evaluator COMMAND arbit-sdf-reference-evaluator-tests)

add_executable(arbit-surface-material-admission-tests
    tests/surface_material_admission_tests.cpp)
target_include_directories(arbit-surface-material-admission-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-surface-material-admission-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME surface_material_admission COMMAND arbit-surface-material-admission-tests)

add_executable(arbit-material-program-compiler-tests
    tests/material_program_compiler_tests.cpp)
target_include_directories(arbit-material-program-compiler-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-material-program-compiler-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME material_program_compiler COMMAND arbit-material-program-compiler-tests)

add_executable(arbit-surface-material-binding-admission-tests
    tests/surface_material_binding_admission_tests.cpp
    tests/support/fixture_scene.cpp
    src/gltf_glb.cpp
    src/glb_scene_adapter.cpp)
target_include_directories(arbit-surface-material-binding-admission-tests PRIVATE
    src
    tests
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
target_link_libraries(arbit-surface-material-binding-admission-tests PRIVATE
    nlohmann_json::nlohmann_json PkgConfig::AV)
if(NOT MSVC)
    target_compile_options(arbit-surface-material-binding-admission-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME surface_material_binding_admission
    COMMAND arbit-surface-material-binding-admission-tests)

add_executable(arbit-path-mask-contract-tests
    tests/path_mask_contract_tests.cpp)
target_include_directories(arbit-path-mask-contract-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-path-mask-contract-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME path_mask_contract COMMAND arbit-path-mask-contract-tests)

add_executable(arbit-diffraction-material-admission-tests
    tests/diffraction_material_admission_tests.cpp)
target_include_directories(arbit-diffraction-material-admission-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-diffraction-material-admission-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME diffraction_material_admission
    COMMAND arbit-diffraction-material-admission-tests)

add_executable(arbit-diffraction-product-plans-compile-tests
    tests/diffraction_product_plans_compile_tests.cpp)
target_include_directories(arbit-diffraction-product-plans-compile-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
target_compile_features(arbit-diffraction-product-plans-compile-tests PRIVATE cxx_std_17)
if(NOT MSVC)
    target_compile_options(arbit-diffraction-product-plans-compile-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME diffraction_product_plans_compile
    COMMAND arbit-diffraction-product-plans-compile-tests)

add_executable(arbit-diffraction-material-execution-tests
    tests/diffraction_material_execution_tests.cpp)
target_include_directories(arbit-diffraction-material-execution-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-diffraction-material-execution-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME diffraction_material_execution
    COMMAND arbit-diffraction-material-execution-tests)

add_executable(arbit-diffraction-order-bsdf-tests
    tests/diffraction_order_bsdf_tests.cpp
    src/diffraction_order_bsdf.cpp)
target_include_directories(arbit-diffraction-order-bsdf-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
target_compile_features(arbit-diffraction-order-bsdf-tests PRIVATE cxx_std_17)
if(NOT MSVC)
    target_compile_options(arbit-diffraction-order-bsdf-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME diffraction_order_bsdf
    COMMAND arbit-diffraction-order-bsdf-tests)

add_test(NAME diffraction_evaluation_backend_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/diffraction_evaluation_backend_source_contract.py
            ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(arbit-diffraction-reference-oracle-tests
    tests/diffraction_reference_oracle_tests.cpp
    tests/support/diffraction_reference_oracle.cpp)
target_include_directories(arbit-diffraction-reference-oracle-tests PRIVATE
    src
    tests
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-diffraction-reference-oracle-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME diffraction_reference_oracle
    COMMAND arbit-diffraction-reference-oracle-tests)
add_test(NAME diffraction_wavelength_reference_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/diffraction_wavelength_reference_source_contract.py)

add_executable(arbit-diffraction-published-grating-validation-tests
    tests/diffraction_published_grating_validation_tests.cpp
    tests/support/diffraction_reference_oracle.cpp)
target_include_directories(arbit-diffraction-published-grating-validation-tests PRIVATE
    src
    tests
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
target_compile_definitions(arbit-diffraction-published-grating-validation-tests PRIVATE
    DIFFRACTION_PUBLISHED_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/diffraction/arnold-2025-centered-rectangular.csv"
    DIFFRACTION_PUBLISHED_METADATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/diffraction/arnold-2025-centered-rectangular.metadata.json")
if(NOT MSVC)
    target_compile_options(arbit-diffraction-published-grating-validation-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME diffraction_published_grating_validation
    COMMAND arbit-diffraction-published-grating-validation-tests)

add_executable(arbit-diffractive-foil-tests
    tests/diffractive_foil_tests.cpp
    tests/support/diffraction_reference_oracle.cpp)
target_include_directories(arbit-diffractive-foil-tests PRIVATE
    src
    tests
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-diffractive-foil-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME diffractive_foil
    COMMAND arbit-diffractive-foil-tests)

if(GLFW3_FOUND AND OpenGL_FOUND AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(arbit-diffraction-material-gl-tests
        tests/diffraction_material_gl_tests.cpp
        tests/support/diffraction_reference_oracle.cpp
        src/diffraction_material_gl.cpp
        src/gl_loader.cpp)
    target_include_directories(arbit-diffraction-material-gl-tests PRIVATE
        src
        tests
        ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    target_compile_definitions(arbit-diffraction-material-gl-tests PRIVATE
        ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=0)
    target_link_libraries(arbit-diffraction-material-gl-tests PRIVATE
        PkgConfig::GLFW3 OpenGL::GL)
    target_compile_options(arbit-diffraction-material-gl-tests PRIVATE
        -Wall -Wextra -Werror)
    add_test(NAME diffraction_material_gl
        COMMAND arbit-diffraction-material-gl-tests)
    set_tests_properties(diffraction_material_gl PROPERTIES
        LABELS "native-physical-diffraction-runtime"
        TIMEOUT 600
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE)
endif()

add_executable(arbit-deformation-contract-tests
    tests/deformation_contract_tests.cpp
    src/deformation_contract.cpp)
target_include_directories(arbit-deformation-contract-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-deformation-contract-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME deformation_contract COMMAND arbit-deformation-contract-tests)

add_executable(arbit-vertex-modifier-ir-admission-tests
    tests/vertex_modifier_ir_admission_tests.cpp)
target_include_directories(arbit-vertex-modifier-ir-admission-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-vertex-modifier-ir-admission-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME vertex_modifier_ir_admission COMMAND arbit-vertex-modifier-ir-admission-tests)

add_executable(arbit-temporal-resource-contract-tests
    tests/temporal_resource_contract_tests.cpp)
target_include_directories(arbit-temporal-resource-contract-tests PRIVATE src)
if(NOT MSVC)
    target_compile_options(arbit-temporal-resource-contract-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME temporal_resource_contract COMMAND arbit-temporal-resource-contract-tests)

add_executable(arbit-volume-data-admission-tests
    tests/volume_data_admission_tests.cpp)
target_include_directories(arbit-volume-data-admission-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-volume-data-admission-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME volume_data_admission COMMAND arbit-volume-data-admission-tests)
add_test(NAME metal_volume_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/metal_volume_source_contract.py)
add_test(NAME metal_static_glb_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/metal_static_glb_source_contract.py)

add_executable(arbit-optical-flow-contract-tests
    tests/optical_flow_contract_tests.cpp)
target_include_directories(arbit-optical-flow-contract-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-optical-flow-contract-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME optical_flow_contract COMMAND arbit-optical-flow-contract-tests)

add_executable(arbit-optical-flow-native-executor-tests
    tests/optical_flow_native_executor_tests.cpp
    src/optical_flow_native_executor.cpp)
target_include_directories(arbit-optical-flow-native-executor-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-optical-flow-native-executor-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME optical_flow_native_executor
    COMMAND arbit-optical-flow-native-executor-tests)

add_executable(arbit-fixture-scene-reference-tests
    tests/fixture_scene_reference_tests.cpp
    tests/support/fixture_scene.cpp
    tests/support/fixture_scene_reference_renderer.cpp
    src/gltf_glb.cpp
    src/glb_scene_adapter.cpp)
target_include_directories(arbit-fixture-scene-reference-tests PRIVATE
    tests/support
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
target_link_libraries(arbit-fixture-scene-reference-tests PRIVATE
    nlohmann_json::nlohmann_json PkgConfig::AV)
if(NOT MSVC)
    target_compile_options(arbit-fixture-scene-reference-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME fixture_scene_reference COMMAND arbit-fixture-scene-reference-tests)

add_executable(arbit-fixture-scene-renderer-tests
    tests/fixture_scene_renderer_tests.cpp
    tests/support/fixture_scene.cpp
    src/fixture_scene_renderer.cpp
    src/gpu_backend/backend_stub.cpp
    src/gltf_glb.cpp
    src/glb_scene_adapter.cpp)
target_include_directories(arbit-fixture-scene-renderer-tests PRIVATE
    src
    tests
    tests/support
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
target_link_libraries(arbit-fixture-scene-renderer-tests PRIVATE
    nlohmann_json::nlohmann_json PkgConfig::AV)
if(NOT MSVC)
    target_compile_options(arbit-fixture-scene-renderer-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME fixture_scene_renderer COMMAND arbit-fixture-scene-renderer-tests)

add_executable(arbit-sdf-native-render-admission-tests
    tests/sdf_native_render_admission_tests.cpp
    src/gpu_backend/backend_stub.cpp)
target_include_directories(arbit-sdf-native-render-admission-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-sdf-native-render-admission-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME sdf_native_render_admission
    COMMAND arbit-sdf-native-render-admission-tests)

add_executable(arbit-sdf-native-renderer-dispatch-tests
    tests/sdf_native_renderer_tests.cpp
    src/sdf_native_renderer.cpp
    src/gpu_backend/backend_stub.cpp)
target_include_directories(arbit-sdf-native-renderer-dispatch-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
if(NOT MSVC)
    target_compile_options(arbit-sdf-native-renderer-dispatch-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME sdf_native_renderer_dispatch
    COMMAND arbit-sdf-native-renderer-dispatch-tests)

add_executable(arbit-color-transform-contract-tests
    ../plugin/Tests/ColorTransformContractTests.cpp)
if(NOT MSVC)
    target_compile_options(arbit-color-transform-contract-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME color_transform_contract COMMAND arbit-color-transform-contract-tests)

add_executable(arbit-color-transform-operation-execution-tests
    tests/color_transform_operation_execution_tests.cpp)
target_include_directories(arbit-color-transform-operation-execution-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(arbit-color-transform-operation-execution-tests PRIVATE
    nlohmann_json::nlohmann_json)
if(NOT MSVC)
    target_compile_options(arbit-color-transform-operation-execution-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME color_transform_operation_execution
    COMMAND arbit-color-transform-operation-execution-tests)

add_executable(arbit-render-pass-output-contract-tests
    ../plugin/Tests/RenderPassOutputContractTests.cpp)
if(NOT MSVC)
    target_compile_options(arbit-render-pass-output-contract-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME render_pass_output_contract COMMAND arbit-render-pass-output-contract-tests)

add_executable(arbit-render-pass-output-publication-tests
    tests/render_pass_output_publication_tests.cpp
    tests/support/fixture_scene.cpp
    src/gpu_backend/backend_stub.cpp
    src/gltf_glb.cpp
    src/glb_scene_adapter.cpp)
target_include_directories(arbit-render-pass-output-publication-tests PRIVATE
    src
    tests
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
target_link_libraries(arbit-render-pass-output-publication-tests PRIVATE
    nlohmann_json::nlohmann_json PkgConfig::AV)
if(NOT MSVC)
    target_compile_options(arbit-render-pass-output-publication-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME render_pass_output_publication COMMAND arbit-render-pass-output-publication-tests)

add_executable(arbit-frame-memory-budget-admission-tests
    tests/frame_memory_budget_admission_tests.cpp)
target_include_directories(arbit-frame-memory-budget-admission-tests PRIVATE src)
if(NOT MSVC)
    target_compile_options(arbit-frame-memory-budget-admission-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME frame_memory_budget_admission COMMAND arbit-frame-memory-budget-admission-tests)
add_test(NAME frame_memory_backend_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/frame_memory_backend_source_contract.py
            ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(arbit-aov-inspection-native-executor-tests
    tests/aov_inspection_native_executor_tests.cpp
    tests/support/fixture_scene.cpp
    src/gltf_glb.cpp
    src/glb_scene_adapter.cpp)
target_include_directories(arbit-aov-inspection-native-executor-tests PRIVATE
    src
    tests
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
target_link_libraries(arbit-aov-inspection-native-executor-tests PRIVATE
    nlohmann_json::nlohmann_json PkgConfig::AV)
if(NOT MSVC)
    target_compile_options(arbit-aov-inspection-native-executor-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME aov_inspection_native_executor
         COMMAND arbit-aov-inspection-native-executor-tests)

add_executable(programmable-admission-tests tests/programmable_admission_tests.cpp)
add_dependencies(programmable-admission-tests arbit-shader-catalog-check)
target_include_directories(programmable-admission-tests PRIVATE src)
target_compile_definitions(programmable-admission-tests PRIVATE
    SHADER_CATALOG_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../content/shader-packs")
target_link_libraries(programmable-admission-tests PRIVATE nlohmann_json::nlohmann_json)
if(NOT MSVC)
    target_compile_options(programmable-admission-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME programmable_admission COMMAND programmable-admission-tests)
set_tests_properties(programmable_admission PROPERTIES TIMEOUT 30)

option(ARBIT_PROGRAMMABLE_TEST_MODE "Enable private programmable-runtime test diagnostics" OFF)
if(ARBIT_PROGRAMMABLE_TEST_MODE)
    target_compile_definitions(arbit-video-helper PRIVATE ARBIT_PROGRAMMABLE_TEST_MODE=1)
endif()

if(UNIX)
    add_executable(programmable-private-payload-fixture tests/private_payload_fixture.cpp)
    target_include_directories(programmable-private-payload-fixture PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    add_executable(programmable-private-payload-tests tests/private_payload_tests.cpp)
    target_include_directories(programmable-private-payload-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    target_compile_definitions(programmable-private-payload-tests PRIVATE
        PRIVATE_PAYLOAD_FIXTURE="$<TARGET_FILE:programmable-private-payload-fixture>")
    add_dependencies(programmable-private-payload-tests programmable-private-payload-fixture)
    add_test(NAME programmable_private_payload COMMAND programmable-private-payload-tests)
    set_tests_properties(programmable_private_payload PROPERTIES TIMEOUT 30)
    # This matrix asserts compile-gated private diagnostics in addition to the
    # public grant/replay behavior.  Do not register it against production-mode
    # helpers, where those diagnostics intentionally do not exist.
    if(ARBIT_PROGRAMMABLE_TEST_MODE)
        add_test(NAME programmable_rpc_grant_matrix
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/programmable_rpc_grant_matrix.py
                    $<TARGET_FILE:arbit-video-helper>)
        set_tests_properties(programmable_rpc_grant_matrix PROPERTIES TIMEOUT 30 RUN_SERIAL TRUE)
    endif()

endif()

# Production lifecycle coverage deliberately links JUCE and drives the real
# process manager into the exact helper target rather than a fork-only fixture.
# It is also a Windows source/launch gate: that build exercises the production
# CreateProcess call site and proves required-private-channel launch is refused.
set(ARBIT_JUCE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../JUCE" CACHE PATH
    "JUCE source tree used by the production sidecar lifecycle test")
if(EXISTS "${ARBIT_JUCE_SOURCE_DIR}/CMakeLists.txt" AND NOT MINGW)
    add_subdirectory("${ARBIT_JUCE_SOURCE_DIR}" ${CMAKE_CURRENT_BINARY_DIR}/JUCE EXCLUDE_FROM_ALL)
    add_executable(programmable-production-lifecycle-tests
        tests/production_lifecycle_tests.cpp)
    target_include_directories(programmable-production-lifecycle-tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../plugin/Source)
    target_compile_definitions(programmable-production-lifecycle-tests PRIVATE
        PRIVATE_LIFECYCLE_HELPER="$<TARGET_FILE:arbit-video-helper>"
        AUTHORITY_CURATED_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/../content/shader-packs/arbit-essentials/shaders/aurora_drift.fs"
        ARBIT_PROGRAMMABLE_TEST_MODE=1 ARBIT_PROGRAMMABLE_AUTHORITY_TEST_SEAM=1
        JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(programmable-resource-limit-fixture
            tests/resource_limit_fixture.cpp)
        target_compile_definitions(programmable-production-lifecycle-tests PRIVATE
            RESOURCE_LIMIT_FIXTURE="$<TARGET_FILE:programmable-resource-limit-fixture>")
        add_dependencies(programmable-production-lifecycle-tests
            programmable-resource-limit-fixture)
    endif()
    target_link_libraries(programmable-production-lifecycle-tests PRIVATE
        juce::juce_core juce::juce_events)
    if(NOT MSVC)
        target_compile_options(programmable-production-lifecycle-tests PRIVATE -UNDEBUG)
    endif()
    add_dependencies(programmable-production-lifecycle-tests arbit-video-helper)
    if(ARBIT_PROGRAMMABLE_TEST_MODE)
        add_test(NAME programmable_production_lifecycle
            COMMAND programmable-production-lifecycle-tests)
        set_tests_properties(programmable_production_lifecycle PROPERTIES TIMEOUT 30 RUN_SERIAL TRUE)
    endif()
endif()

add_executable(programmable-platform-capability-tests tests/platform_capability_tests.cpp)
target_include_directories(programmable-platform-capability-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
add_test(NAME programmable_platform_capability COMMAND programmable-platform-capability-tests)
set_tests_properties(programmable_platform_capability PROPERTIES TIMEOUT 30)

add_executable(programmable-hook-limits-tests tests/programmable_hook_limits_tests.cpp)
target_include_directories(programmable-hook-limits-tests PRIVATE src)
if(LUA_FOUND)
    target_link_libraries(programmable-hook-limits-tests PRIVATE PkgConfig::LUA)
    target_compile_definitions(programmable-hook-limits-tests PRIVATE ARBIT_HAVE_LUA=1)
endif()
if(TARGET arbit-quickjs)
    target_link_libraries(programmable-hook-limits-tests PRIVATE arbit-quickjs)
    target_compile_definitions(programmable-hook-limits-tests PRIVATE ARBIT_HAVE_QUICKJS=1)
endif()
if(MSVC)
    target_compile_options(programmable-hook-limits-tests PRIVATE /UNDEBUG)
else()
    target_compile_options(programmable-hook-limits-tests PRIVATE -Wall -Wextra -UNDEBUG)
endif()
add_test(NAME programmable_hook_limits COMMAND programmable-hook-limits-tests)
set_tests_properties(programmable_hook_limits PROPERTIES TIMEOUT 30)

if(UNIX AND NOT WIN32)
    add_executable(arbit-depth-cache-tests tests/depth_cache_tests.cpp)
    target_include_directories(arbit-depth-cache-tests PRIVATE src)
    target_link_libraries(arbit-depth-cache-tests PRIVATE PkgConfig::AV nlohmann_json::nlohmann_json)
    target_compile_options(arbit-depth-cache-tests PRIVATE -Wall -Wextra -Werror)
    add_test(NAME depth_cache_admission COMMAND arbit-depth-cache-tests)
    set_tests_properties(depth_cache_admission PROPERTIES TIMEOUT 30)
endif()
add_test(NAME depth_cache_windows_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/depth_cache_windows_source_contract.py
            ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME blend_modes_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/blend_modes_source_contract.py)
add_test(NAME phase13_recovery_preview_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/phase13_recovery_preview_source_contract.py
            ${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME surface_material_starter_metal_source_gate
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/surface_material_starter_metal_source_gate.py
            ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME surface_material_starter_independent_oracle_freshness
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/generate_surface_material_starter_expected.py
            --check)

add_executable(arbit-surface-material-starter-oracle-tests
    tests/surface_material_starter_oracle_tests.cpp)
target_include_directories(arbit-surface-material-starter-oracle-tests PRIVATE src tests)
target_compile_definitions(arbit-surface-material-starter-oracle-tests PRIVATE
    SURFACE_MATERIAL_STARTER_ORACLE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/surface_material_starter_expected.json")
target_link_libraries(arbit-surface-material-starter-oracle-tests PRIVATE
    nlohmann_json::nlohmann_json)
if(NOT MSVC)
    target_compile_options(arbit-surface-material-starter-oracle-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME surface_material_starter_oracle_cross_language
    COMMAND arbit-surface-material-starter-oracle-tests)

add_executable(arbit-composite-probe-contract-tests tests/composite_probe_contract_tests.cpp)
target_include_directories(arbit-composite-probe-contract-tests PRIVATE src)
if(NOT MSVC)
    target_compile_options(arbit-composite-probe-contract-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME composite_probe_contract COMMAND arbit-composite-probe-contract-tests)

add_executable(arbit-visual-plan-executor-tests tests/visual_plan_executor_tests.cpp)
target_include_directories(arbit-visual-plan-executor-tests PRIVATE src)
target_link_libraries(arbit-visual-plan-executor-tests PRIVATE nlohmann_json::nlohmann_json)
if(NOT MSVC)
    target_compile_options(arbit-visual-plan-executor-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME visual_plan_executor COMMAND arbit-visual-plan-executor-tests)

add_executable(arbit-four-way-compositor-tests tests/four_way_compositor_tests.cpp)
target_include_directories(arbit-four-way-compositor-tests PRIVATE src)
target_link_libraries(arbit-four-way-compositor-tests PRIVATE nlohmann_json::nlohmann_json)
target_compile_features(arbit-four-way-compositor-tests PRIVATE cxx_std_17)
if(NOT MSVC)
    target_compile_options(arbit-four-way-compositor-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME four_way_compositor COMMAND arbit-four-way-compositor-tests)

add_executable(arbit-imported-animation-visual-plan-execution-tests
    tests/imported_animation_visual_plan_execution_tests.cpp)
target_include_directories(arbit-imported-animation-visual-plan-execution-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
target_link_libraries(arbit-imported-animation-visual-plan-execution-tests PRIVATE
    nlohmann_json::nlohmann_json)
if(NOT MSVC)
    target_compile_options(arbit-imported-animation-visual-plan-execution-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME imported_animation_visual_plan_execution
    COMMAND arbit-imported-animation-visual-plan-execution-tests)

add_executable(arbit-imported-scene-visual-plan-execution-tests
    tests/imported_scene_visual_plan_execution_tests.cpp)
target_include_directories(arbit-imported-scene-visual-plan-execution-tests PRIVATE
    src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
target_link_libraries(arbit-imported-scene-visual-plan-execution-tests PRIVATE
    nlohmann_json::nlohmann_json)
if(NOT MSVC)
    target_compile_options(arbit-imported-scene-visual-plan-execution-tests PRIVATE
        -Wall -Wextra -Werror)
endif()
add_test(NAME imported_scene_visual_plan_execution
    COMMAND arbit-imported-scene-visual-plan-execution-tests)
add_test(NAME imported_scene_product_route_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/imported_scene_product_route_source_contract.py
            ${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME sdf_product_route_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/sdf_product_route_source_contract.py
            ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME ordered_shader_authority_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/ordered_shader_authority_source_contract.py
            ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(arbit-visual-plan-resource-budget-tests tests/visual_plan_resource_budget_tests.cpp)
target_include_directories(arbit-visual-plan-resource-budget-tests PRIVATE src)
target_link_libraries(arbit-visual-plan-resource-budget-tests PRIVATE nlohmann_json::nlohmann_json)
if(NOT MSVC)
    target_compile_options(arbit-visual-plan-resource-budget-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME visual_plan_resource_budget COMMAND arbit-visual-plan-resource-budget-tests)

add_executable(arbit-flat-shader-bridge-tests tests/flat_shader_bridge_tests.cpp)
add_dependencies(arbit-flat-shader-bridge-tests arbit-shader-catalog-check)
target_include_directories(arbit-flat-shader-bridge-tests PRIVATE src)
target_compile_definitions(arbit-flat-shader-bridge-tests PRIVATE
    SHADER_CATALOG_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../content/shader-packs")
target_link_libraries(arbit-flat-shader-bridge-tests PRIVATE nlohmann_json::nlohmann_json)
if(NOT MSVC)
    target_compile_options(arbit-flat-shader-bridge-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME flat_shader_bridge_contract COMMAND arbit-flat-shader-bridge-tests)
set_tests_properties(flat_shader_bridge_contract PROPERTIES TIMEOUT 30)

if(UNIX AND NOT WIN32)
    add_executable(arbit-tracking-runtime-tests tests/tracking_runtime_tests.cpp)
    target_include_directories(arbit-tracking-runtime-tests PRIVATE src)
    target_compile_definitions(arbit-tracking-runtime-tests PRIVATE
        ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=0)
    target_link_libraries(arbit-tracking-runtime-tests PRIVATE nlohmann_json::nlohmann_json)
    target_compile_options(arbit-tracking-runtime-tests PRIVATE -Wall -Wextra -Werror)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 16)
        # GCC 16 misdiagnoses nlohmann::json's shared output adapter teardown
        # after inlining it through dump(); keep project warnings fatal.
        target_compile_options(arbit-tracking-runtime-tests PRIVATE -Wno-array-bounds)
    endif()
    add_test(NAME tracking_runtime COMMAND arbit-tracking-runtime-tests)
    set_tests_properties(tracking_runtime PROPERTIES TIMEOUT 30)
endif()

add_test(NAME generator_particle_telemetry_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/generator_particle_telemetry_source_contract.py)
add_test(NAME node_cost_telemetry_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/node_cost_telemetry_source_contract.py)

add_executable(arbit-viewport-telemetry-owner-tests tests/viewport_telemetry_owner_tests.cpp)
target_include_directories(arbit-viewport-telemetry-owner-tests PRIVATE src)
if(NOT MSVC)
    target_compile_options(arbit-viewport-telemetry-owner-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME viewport_telemetry_owner COMMAND arbit-viewport-telemetry-owner-tests)
add_test(NAME viewport_telemetry_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/viewport_telemetry_source_contract.py
            ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(arbit-export-telemetry-owner-tests tests/export_telemetry_owner_tests.cpp)
target_include_directories(arbit-export-telemetry-owner-tests PRIVATE src)
if(NOT MSVC)
    target_compile_options(arbit-export-telemetry-owner-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME export_telemetry_owner COMMAND arbit-export-telemetry-owner-tests)

add_executable(arbit-export-visual-plan-session-tests
    tests/export_visual_plan_session_tests.cpp
    src/export_visual_plan_session.cpp
    src/exporter_visual_plan_callbacks.cpp
    src/visual_plan_execution_telemetry.cpp
    src/visual_renderer_telemetry.cpp)
target_include_directories(arbit-export-visual-plan-session-tests PRIVATE src)
if(NOT MSVC)
    target_compile_options(arbit-export-visual-plan-session-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME export_visual_plan_session COMMAND arbit-export-visual-plan-session-tests)

add_test(NAME export_telemetry_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/export_telemetry_source_contract.py
            ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME viewport_backend_budget_source_contract
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/viewport_backend_budget_source_contract.py
            ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(arbit-export-encoder-fallback-tests
    tests/export_encoder_fallback_tests.cpp)
target_include_directories(arbit-export-encoder-fallback-tests PRIVATE src)
if(NOT MSVC)
    target_compile_options(arbit-export-encoder-fallback-tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME export_encoder_fallback COMMAND arbit-export-encoder-fallback-tests)

# M7 physical Linux acceptance: drive the production helper/exporter/FrameRenderer
# and FFmpeg handoff at exact 1080p then 4K. This is intentionally not part of
# the default test set: it needs a real OpenGL device/display and ffprobe and may
# take up to five minutes. Run explicitly with `ctest -L native-export-runtime`.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND GLFW3_FOUND AND OpenGL_FOUND)
    add_test(NAME native_linux_export_runtime
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/export_runtime_fixture.py
                $<TARGET_FILE:arbit-video-helper>)
    set_tests_properties(native_linux_export_runtime PROPERTIES
        LABELS "native-export-runtime"
        TIMEOUT 300
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE)
endif()

add_executable(arbit-viewport-plan-concurrency-tests tests/viewport_plan_concurrency_tests.cpp)
target_include_directories(arbit-viewport-plan-concurrency-tests PRIVATE src)
find_package(Threads REQUIRED)
target_link_libraries(arbit-viewport-plan-concurrency-tests PRIVATE Threads::Threads)
if(NOT MSVC)
    target_compile_options(arbit-viewport-plan-concurrency-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME viewport_plan_concurrency COMMAND arbit-viewport-plan-concurrency-tests)

if(GLFW3_FOUND AND OpenGL_FOUND)
    # Real Linux pixel acceptance for Depth Fog. This compiles the production
    # FrameRenderer and shader owners, creates a hidden native GL context, uploads
    # an actual GL_R16 gradient, and reads the composited pixels back. No CPU
    # renderer or replacement shader participates.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(arbit-depth-fog-gl-pixel-tests
            tests/depth_fog_gl_pixel_tests.cpp
            src/renderer.cpp
            src/visual_plan_execution_telemetry.cpp
            src/visual_renderer_telemetry.cpp
            src/geometry_core_backend.cpp
            src/color_transform_gl.cpp
            src/gpu_backend/backend_stub.cpp
            src/shader_generator.cpp
            src/particle_engine.cpp
            src/gl_loader.cpp
            src/lut_loader.cpp)
        target_include_directories(arbit-depth-fog-gl-pixel-tests PRIVATE src)
        target_compile_definitions(arbit-depth-fog-gl-pixel-tests PRIVATE
            ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=0)
        target_link_libraries(arbit-depth-fog-gl-pixel-tests PRIVATE
            PkgConfig::GLFW3 OpenGL::GL nlohmann_json::nlohmann_json PkgConfig::AV)
        target_compile_options(arbit-depth-fog-gl-pixel-tests PRIVATE -Wall -Wextra -Werror)
        add_test(NAME depth_fog_gl_pixels COMMAND arbit-depth-fog-gl-pixel-tests)
        set_tests_properties(depth_fog_gl_pixels PROPERTIES
            LABELS "native-depth-fog-runtime"
            TIMEOUT 600
            RUN_SERIAL TRUE)

        add_executable(arbit-temporal-frame-delay-gl-pixel-tests
            tests/temporal_frame_delay_gl_pixel_tests.cpp
            src/renderer.cpp
            src/visual_plan_execution_telemetry.cpp
            src/visual_renderer_telemetry.cpp
            src/geometry_core_backend.cpp
            src/color_transform_gl.cpp
            src/gpu_backend/backend_stub.cpp
            src/shader_generator.cpp
            src/particle_engine.cpp
            src/gl_loader.cpp
            src/lut_loader.cpp)
        target_include_directories(arbit-temporal-frame-delay-gl-pixel-tests PRIVATE src)
        target_compile_definitions(arbit-temporal-frame-delay-gl-pixel-tests PRIVATE
            ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=0)
        target_link_libraries(arbit-temporal-frame-delay-gl-pixel-tests PRIVATE
            PkgConfig::GLFW3 OpenGL::GL nlohmann_json::nlohmann_json PkgConfig::AV)
        target_compile_options(arbit-temporal-frame-delay-gl-pixel-tests PRIVATE
            -Wall -Wextra -Werror)
        add_test(NAME temporal_frame_delay_gl_pixels
            COMMAND arbit-temporal-frame-delay-gl-pixel-tests)
        set_tests_properties(temporal_frame_delay_gl_pixels PROPERTIES
            LABELS "native-temporal-runtime"
            TIMEOUT 600
            SKIP_RETURN_CODE 77
            RUN_SERIAL TRUE)

        add_executable(arbit-motion-blur-gl-pixel-tests
            tests/motion_blur_gl_pixel_tests.cpp
            src/renderer.cpp
            src/visual_plan_execution_telemetry.cpp
            src/visual_renderer_telemetry.cpp
            src/geometry_core_backend.cpp
            src/color_transform_gl.cpp
            src/gpu_backend/backend_opengl.cpp
            src/sdf_native_renderer.cpp
            src/deformation_contract.cpp
            src/native_animation_deformation.cpp
            src/shader_generator.cpp
            src/particle_engine.cpp
            src/gl_loader.cpp
            src/lut_loader.cpp)
        target_include_directories(arbit-motion-blur-gl-pixel-tests PRIVATE
            src
            ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
        target_compile_definitions(arbit-motion-blur-gl-pixel-tests PRIVATE
            ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=0
            ARBIT_HAVE_OPENGL_SDF_BACKEND=1)
        target_link_libraries(arbit-motion-blur-gl-pixel-tests PRIVATE
            PkgConfig::GLFW3 OpenGL::GL nlohmann_json::nlohmann_json PkgConfig::AV)
        target_compile_options(arbit-motion-blur-gl-pixel-tests PRIVATE
            -Wall -Wextra -Werror)
        add_test(NAME motion_blur_gl_pixels
            COMMAND arbit-motion-blur-gl-pixel-tests)
        set_tests_properties(motion_blur_gl_pixels PROPERTIES
            LABELS "native-temporal-runtime"
            TIMEOUT 600
            SKIP_RETURN_CODE 77
            RUN_SERIAL TRUE)

        add_executable(arbit-custom-shader-gl-pixel-tests
            tests/custom_shader_gl_pixel_tests.cpp
            src/renderer.cpp
            src/geometry_core_backend.cpp
            src/color_transform_gl.cpp
            src/gpu_backend/backend_opengl.cpp
            src/sdf_native_renderer.cpp
            src/deformation_contract.cpp
            src/native_animation_deformation.cpp
            src/visual_plan_execution_telemetry.cpp
            src/visual_renderer_telemetry.cpp
            src/shader_generator.cpp
            src/particle_engine.cpp
            src/gl_loader.cpp
            src/lut_loader.cpp)
        target_include_directories(arbit-custom-shader-gl-pixel-tests PRIVATE
            src
            ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
        target_compile_definitions(arbit-custom-shader-gl-pixel-tests PRIVATE
            ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=0
            ARBIT_HAVE_OPENGL_SDF_BACKEND=1
            ARBIT_SHADER_PACK_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../content/shader-packs")
        target_link_libraries(arbit-custom-shader-gl-pixel-tests PRIVATE
            PkgConfig::GLFW3 OpenGL::GL nlohmann_json::nlohmann_json PkgConfig::AV)
        target_compile_options(arbit-custom-shader-gl-pixel-tests PRIVATE
            -Wall -Wextra -Werror)
        add_test(NAME custom_shader_gl_pixels
            COMMAND arbit-custom-shader-gl-pixel-tests)
        set_tests_properties(custom_shader_gl_pixels PROPERTIES
            LABELS "native-custom-shader-runtime"
            TIMEOUT 600
            SKIP_RETURN_CODE 77
            RUN_SERIAL TRUE)
    endif()

    # The production particle GL path requires OpenGL 4.3 shader-storage
    # buffers. macOS exposes at most OpenGL 4.1 and uses the native Metal path.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(arbit-particle-engine-gl-tests
            tests/particle_engine_gl_tests.cpp src/particle_engine.cpp src/gl_loader.cpp)
        target_include_directories(arbit-particle-engine-gl-tests PRIVATE src)
        target_compile_definitions(arbit-particle-engine-gl-tests PRIVATE
            ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=0)
        target_link_libraries(arbit-particle-engine-gl-tests PRIVATE PkgConfig::GLFW3 OpenGL::GL)
        add_test(NAME particle_engine_gl COMMAND arbit-particle-engine-gl-tests)
        set_tests_properties(particle_engine_gl PROPERTIES
            LABELS "native-particle-runtime"
            TIMEOUT 600
            SKIP_RETURN_CODE 77
            RUN_SERIAL TRUE)
    endif()
endif()

add_executable(arbit-node-preview-presentation-tests tests/node_preview_presentation_tests.cpp)
target_include_directories(arbit-node-preview-presentation-tests PRIVATE src)
if(NOT MSVC)
    target_compile_options(arbit-node-preview-presentation-tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME node_preview_presentation COMMAND arbit-node-preview-presentation-tests)

if(NOT APPLE AND GLFW3_FOUND AND OpenGL_FOUND)
    add_executable(arbit-opengl-volume-backend-tests
        tests/opengl_volume_backend_tests.cpp
        src/volume_native_renderer_opengl.cpp
        src/gl_loader.cpp)
    target_include_directories(arbit-opengl-volume-backend-tests PRIVATE
        src
        ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    target_compile_definitions(arbit-opengl-volume-backend-tests PRIVATE
        ARBIT_HAVE_VIEWPORT=1)
    target_link_libraries(arbit-opengl-volume-backend-tests PRIVATE
        PkgConfig::GLFW3 OpenGL::GL)
    if(NOT MSVC)
        target_compile_options(arbit-opengl-volume-backend-tests PRIVATE
            -Wall -Wextra -Werror)
    endif()
    add_test(NAME opengl_volume_backend COMMAND arbit-opengl-volume-backend-tests)
    set_tests_properties(opengl_volume_backend PROPERTIES
        LABELS "native-volume-runtime"
        TIMEOUT 60
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE)

    add_executable(arbit-opengl-sdf-backend-tests
        tests/opengl_sdf_backend_tests.cpp
        tests/support/sdf_native_backend_parity.cpp
        tests/support/sdf_reference_evaluator.cpp
        src/sdf_native_renderer.cpp
        src/deformation_contract.cpp
        src/native_animation_deformation.cpp
        src/gpu_backend/backend_opengl.cpp
        src/gl_loader.cpp)
    target_include_directories(arbit-opengl-sdf-backend-tests PRIVATE
        src
        ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    target_compile_definitions(arbit-opengl-sdf-backend-tests PRIVATE
        ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=0)
    target_link_libraries(arbit-opengl-sdf-backend-tests PRIVATE
        PkgConfig::GLFW3 OpenGL::GL)
    if(NOT MSVC)
        target_compile_options(arbit-opengl-sdf-backend-tests PRIVATE
            -Wall -Wextra -Werror)
    endif()
    add_test(NAME opengl_sdf_backend COMMAND arbit-opengl-sdf-backend-tests)
    set_tests_properties(opengl_sdf_backend PROPERTIES
        LABELS "native-sdf-runtime"
        TIMEOUT 60
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE)

    # visual_plan_executor.h owns the production Geometry Core path, so its
    # focused test must link the same native backend it instantiates.
    target_sources(arbit-visual-plan-executor-tests PRIVATE
        src/geometry_core_backend.cpp
        src/visual_plan_execution_telemetry.cpp
        src/deformation_contract.cpp
        src/native_animation_deformation.cpp
        src/gpu_backend/backend_opengl.cpp
        src/gl_loader.cpp)
    target_include_directories(arbit-visual-plan-executor-tests PRIVATE
        tests ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    target_compile_definitions(arbit-visual-plan-executor-tests PRIVATE
        ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=0
        ARBIT_HAVE_OPENGL_SDF_BACKEND=1)
    target_link_libraries(arbit-visual-plan-executor-tests PRIVATE
        PkgConfig::GLFW3 OpenGL::GL)

    add_executable(arbit-opengl-geometry-core-backend-tests
        tests/opengl_geometry_core_backend_tests.cpp
        src/geometry_core_backend.cpp
        src/deformation_contract.cpp
        src/native_animation_deformation.cpp
        src/gpu_backend/backend_opengl.cpp
        src/gl_loader.cpp)
    target_include_directories(arbit-opengl-geometry-core-backend-tests PRIVATE
        src tests ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    target_compile_definitions(arbit-opengl-geometry-core-backend-tests PRIVATE
        ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=0
        ARBIT_HAVE_OPENGL_SDF_BACKEND=1)
    target_link_libraries(arbit-opengl-geometry-core-backend-tests PRIVATE
        PkgConfig::GLFW3 OpenGL::GL)
    target_compile_options(arbit-opengl-geometry-core-backend-tests PRIVATE
        -Wall -Wextra -Werror)
    add_test(NAME opengl_geometry_core_backend
        COMMAND arbit-opengl-geometry-core-backend-tests)
    set_tests_properties(opengl_geometry_core_backend PROPERTIES
        LABELS "native-geometry-core-runtime"
        TIMEOUT 60
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE)

    add_executable(arbit-opengl-fixture-scene-backend-tests
        tests/opengl_fixture_scene_backend_tests.cpp
        tests/support/fixture_scene.cpp
        src/gltf_glb.cpp
        src/glb_scene_adapter.cpp
        src/animation_clip.cpp
        src/animation_playback_control.cpp
        src/animation_deformation_evaluation.cpp
        src/imported_animation_deformation_consumer.cpp
        src/fixture_scene_renderer.cpp
        src/deformation_contract.cpp
        src/native_animation_deformation.cpp
        src/gpu_backend/backend_opengl.cpp
        src/gl_loader.cpp)
    target_include_directories(arbit-opengl-fixture-scene-backend-tests PRIVATE
        src
        tests
        ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    target_compile_definitions(arbit-opengl-fixture-scene-backend-tests PRIVATE
        ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=0
        ARBIT_HAVE_OPENGL_SDF_BACKEND=1
        SURFACE_MATERIAL_STARTER_ORACLE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/surface_material_starter_expected.json")
    target_link_libraries(arbit-opengl-fixture-scene-backend-tests PRIVATE
        PkgConfig::GLFW3 OpenGL::GL nlohmann_json::nlohmann_json PkgConfig::AV)
    if(NOT MSVC)
        target_compile_options(arbit-opengl-fixture-scene-backend-tests PRIVATE
            -Wall -Wextra -Werror)
    endif()
    add_test(NAME opengl_fixture_scene_backend
        COMMAND arbit-opengl-fixture-scene-backend-tests)
    set_tests_properties(opengl_fixture_scene_backend PROPERTIES
        LABELS "native-fixture-runtime;native-surface-material-runtime"
        TIMEOUT 60
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE)

    add_executable(arbit-opengl-packaged-multi-object-acceptance
        tests/opengl_fixture_scene_backend_tests.cpp
        tests/support/fixture_scene.cpp
        src/gltf_glb.cpp
        src/glb_scene_adapter.cpp
        src/fixture_scene_renderer.cpp
        src/deformation_contract.cpp
        src/native_animation_deformation.cpp
        src/gpu_backend/backend_opengl.cpp
        src/gl_loader.cpp)
    target_include_directories(arbit-opengl-packaged-multi-object-acceptance PRIVATE
        src tests ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    target_compile_definitions(arbit-opengl-packaged-multi-object-acceptance PRIVATE
        ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=0
        ARBIT_HAVE_OPENGL_SDF_BACKEND=1 ARBIT_STRICT_PACKAGED_ACCEPTANCE=1
        SURFACE_MATERIAL_STARTER_ORACLE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/surface_material_starter_expected.json")
    target_link_libraries(arbit-opengl-packaged-multi-object-acceptance PRIVATE
        PkgConfig::GLFW3 OpenGL::GL nlohmann_json::nlohmann_json PkgConfig::AV)
    if(NOT MSVC)
        target_compile_options(arbit-opengl-packaged-multi-object-acceptance PRIVATE
            -Wall -Wextra -Werror)
    endif()
    add_test(NAME opengl_packaged_multi_object_acceptance
        COMMAND arbit-opengl-packaged-multi-object-acceptance)
    set_tests_properties(opengl_packaged_multi_object_acceptance PROPERTIES
        LABELS "native-fixture-runtime;packaged-asset-runtime;physical-opengl"
        TIMEOUT 60
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE)

    add_executable(arbit-opengl-animation-deformation-backend-tests
        tests/opengl_animation_deformation_backend_tests.cpp
        tests/support/fixture_scene.cpp
        src/gltf_glb.cpp
        src/glb_scene_adapter.cpp
        src/animation_clip.cpp
        src/animation_deformation_evaluation.cpp
        src/deformation_contract.cpp
        src/native_animation_deformation.cpp
        src/gpu_backend/backend_opengl.cpp
        src/gl_loader.cpp)
    target_include_directories(arbit-opengl-animation-deformation-backend-tests PRIVATE
        src
        tests
        ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    target_compile_definitions(arbit-opengl-animation-deformation-backend-tests PRIVATE
        ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=0
        ARBIT_HAVE_OPENGL_SDF_BACKEND=1)
    target_link_libraries(arbit-opengl-animation-deformation-backend-tests PRIVATE
        PkgConfig::GLFW3 OpenGL::GL nlohmann_json::nlohmann_json PkgConfig::AV)
    if(NOT MSVC)
        target_compile_options(arbit-opengl-animation-deformation-backend-tests PRIVATE
            -Wall -Wextra -Werror)
    endif()
    add_test(NAME opengl_animation_deformation_backend
        COMMAND arbit-opengl-animation-deformation-backend-tests)
    set_tests_properties(opengl_animation_deformation_backend PROPERTIES
        LABELS "native-animation-deformation-runtime"
        TIMEOUT 60
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE)
endif()

if(APPLE AND ARBIT_WITH_METAL)
    add_executable(arbit-metal-geometry-core-backend-tests
        tests/metal_geometry_core_backend_tests.mm
        src/geometry_core_backend.cpp
        src/deformation_contract.cpp
        src/native_animation_deformation.cpp
        src/gpu_backend/backend_sokol.mm)
    target_include_directories(arbit-metal-geometry-core-backend-tests PRIVATE
        src tests third_party/sokol ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    target_compile_definitions(arbit-metal-geometry-core-backend-tests PRIVATE
        ARBIT_HAVE_VIEWPORT=0 ARBIT_HAVE_METAL_BACKEND=1)
    target_link_libraries(arbit-metal-geometry-core-backend-tests PRIVATE
        "-framework Metal" "-framework QuartzCore" "-framework Cocoa")
    target_compile_options(arbit-metal-geometry-core-backend-tests PRIVATE
        -Wall -Wextra -Werror)
    add_test(NAME metal_geometry_core_backend
        COMMAND arbit-metal-geometry-core-backend-tests)
    set_tests_properties(metal_geometry_core_backend PROPERTIES
        LABELS "native-geometry-core-runtime;physical-apple-silicon"
        TIMEOUT 600
        RUN_SERIAL TRUE)

    add_executable(arbit-metal-animation-deformation-backend-tests
        tests/metal_animation_deformation_backend_tests.mm
        tests/support/fixture_scene.cpp
        src/animation_clip.cpp
        src/animation_deformation_evaluation.cpp
        src/deformation_contract.cpp
        src/native_animation_deformation.cpp
        src/gpu_backend/backend_sokol.mm)
    target_include_directories(arbit-metal-animation-deformation-backend-tests PRIVATE
        src
        tests
        third_party/sokol
        ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    target_compile_definitions(arbit-metal-animation-deformation-backend-tests PRIVATE
        ARBIT_HAVE_VIEWPORT=0 ARBIT_HAVE_METAL_BACKEND=1)
    target_link_libraries(arbit-metal-animation-deformation-backend-tests PRIVATE
        "-framework Metal" "-framework QuartzCore" "-framework Cocoa")
    target_compile_options(arbit-metal-animation-deformation-backend-tests PRIVATE
        -Wall -Wextra -Werror)
    add_test(NAME metal_animation_deformation_backend
        COMMAND arbit-metal-animation-deformation-backend-tests)
    set_tests_properties(metal_animation_deformation_backend PROPERTIES
        LABELS "native-animation-deformation-runtime;physical-apple-silicon"
        TIMEOUT 600
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE)

    add_executable(arbit-metal-volume-backend-tests
        tests/metal_volume_backend_tests.mm
        src/volume_native_renderer_metal.mm
        src/deformation_contract.cpp
        src/native_animation_deformation.cpp
        src/gpu_backend/backend_sokol.mm)
    target_include_directories(arbit-metal-volume-backend-tests PRIVATE
        src
        third_party/sokol
        ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    target_compile_definitions(arbit-metal-volume-backend-tests PRIVATE
        ARBIT_HAVE_VIEWPORT=0 ARBIT_HAVE_METAL_BACKEND=1)
    target_link_libraries(arbit-metal-volume-backend-tests PRIVATE
        "-framework Metal" "-framework QuartzCore" "-framework Cocoa")
    target_compile_options(arbit-metal-volume-backend-tests PRIVATE
        -Wall -Wextra -Werror)
    add_test(NAME metal_volume_backend
        COMMAND arbit-metal-volume-backend-tests)
    set_tests_properties(metal_volume_backend PROPERTIES
        LABELS "native-volume-runtime;physical-apple-silicon"
        TIMEOUT 600
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE)

    add_executable(arbit-metal-backend-tests
        tests/metal_backend_tests.mm
        tests/support/fixture_scene.cpp
        tests/support/sdf_native_backend_parity.cpp
        tests/support/sdf_reference_evaluator.cpp
        src/sdf_native_renderer.cpp
        src/deformation_contract.cpp
        src/native_animation_deformation.cpp
        src/gpu_backend/backend_sokol.mm)
    target_include_directories(arbit-metal-backend-tests PRIVATE
        src
        third_party/sokol
        ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    target_link_libraries(arbit-metal-backend-tests PRIVATE
        "-framework Metal" "-framework QuartzCore" "-framework Cocoa")
    add_test(NAME metal_backend COMMAND arbit-metal-backend-tests)

    add_executable(arbit-optical-flow-native-metal-tests
        tests/optical_flow_native_metal_tests.mm
        src/optical_flow_native_executor.cpp
        src/gpu_backend/backend_sokol.mm)
    target_include_directories(arbit-optical-flow-native-metal-tests PRIVATE
        src
        third_party/sokol
        ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    target_compile_definitions(arbit-optical-flow-native-metal-tests PRIVATE
        ARBIT_HAVE_VIEWPORT=0 ARBIT_HAVE_METAL_BACKEND=1)
    target_link_libraries(arbit-optical-flow-native-metal-tests PRIVATE
        "-framework Metal" "-framework QuartzCore" "-framework Cocoa")
    target_compile_options(arbit-optical-flow-native-metal-tests PRIVATE
        -Wall -Wextra -Werror)
    add_test(NAME optical_flow_native_metal
        COMMAND arbit-optical-flow-native-metal-tests)
    set_tests_properties(optical_flow_native_metal PROPERTIES
        LABELS "native-optical-flow-runtime"
        TIMEOUT 60
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE)

    add_executable(arbit-diffraction-material-metal-tests
        tests/diffraction_material_metal_tests.cpp
        tests/support/diffraction_reference_oracle.cpp
        src/diffraction_material_metal.mm
        src/deformation_contract.cpp
        src/native_animation_deformation.cpp
        src/gpu_backend/backend_sokol.mm)
    target_include_directories(arbit-diffraction-material-metal-tests PRIVATE
        src
        tests
        third_party/sokol
        ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    target_compile_definitions(arbit-diffraction-material-metal-tests PRIVATE
        ARBIT_HAVE_VIEWPORT=0 ARBIT_HAVE_METAL_BACKEND=1)
    target_link_libraries(arbit-diffraction-material-metal-tests PRIVATE
        "-framework Metal" "-framework QuartzCore" "-framework Cocoa")
    target_compile_options(arbit-diffraction-material-metal-tests PRIVATE
        -Wall -Wextra -Werror)
    add_test(NAME diffraction_material_metal
        COMMAND arbit-diffraction-material-metal-tests)
    set_tests_properties(diffraction_material_metal PROPERTIES
        LABELS "native-physical-diffraction-runtime"
        TIMEOUT 600
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE)
    add_test(NAME diffraction_wavelength_validation_metal_strict
        COMMAND arbit-diffraction-material-metal-tests --strict)
    set_tests_properties(diffraction_wavelength_validation_metal_strict PROPERTIES
        LABELS "native-physical-diffraction-runtime;strict-metal-runtime"
        TIMEOUT 600
        RUN_SERIAL TRUE)

    add_executable(arbit-metal-fixture-diffraction-tests
        tests/metal_fixture_scene_backend_tests.mm
        tests/support/fixture_scene.cpp
        src/fixture_scene_renderer.cpp
        src/gltf_glb.cpp
        src/glb_scene_adapter.cpp
        src/animation_clip.cpp
        src/animation_playback_control.cpp
        src/animation_deformation_evaluation.cpp
        src/imported_animation_deformation_consumer.cpp
        src/deformation_contract.cpp
        src/native_animation_deformation.cpp
        src/gpu_backend/backend_sokol.mm)
    target_include_directories(arbit-metal-fixture-diffraction-tests PRIVATE
        src
        tests
        third_party/sokol
        ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    target_compile_definitions(arbit-metal-fixture-diffraction-tests PRIVATE
        ARBIT_HAVE_VIEWPORT=0 ARBIT_HAVE_METAL_BACKEND=1
        SURFACE_MATERIAL_STARTER_ORACLE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/surface_material_starter_expected.json")
    target_link_libraries(arbit-metal-fixture-diffraction-tests PRIVATE
        nlohmann_json::nlohmann_json PkgConfig::AV
        "-framework Metal" "-framework QuartzCore" "-framework Cocoa")
    target_compile_options(arbit-metal-fixture-diffraction-tests PRIVATE
        -Wall -Wextra -Werror)
    add_test(NAME metal_fixture_diffraction
        COMMAND arbit-metal-fixture-diffraction-tests)
    set_tests_properties(metal_fixture_diffraction PROPERTIES
        LABELS "native-physical-diffraction-runtime;native-surface-material-runtime;strict-static-glb-runtime;physical-apple-silicon"
        TIMEOUT 600
        RUN_SERIAL TRUE)

    add_executable(arbit-metal-rich-static-glb-tests
        tests/metal_rich_static_glb_backend_tests.mm
        src/gltf_glb.cpp
        src/glb_scene_adapter.cpp
        src/deformation_contract.cpp
        src/native_animation_deformation.cpp
        src/gpu_backend/backend_sokol.mm)
    target_include_directories(arbit-metal-rich-static-glb-tests PRIVATE
        src
        tests
        third_party/sokol
        ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    target_compile_definitions(arbit-metal-rich-static-glb-tests PRIVATE
        ARBIT_HAVE_VIEWPORT=0 ARBIT_HAVE_METAL_BACKEND=1)
    target_link_libraries(arbit-metal-rich-static-glb-tests PRIVATE
        nlohmann_json::nlohmann_json PkgConfig::AV
        "-framework Metal" "-framework QuartzCore" "-framework Cocoa")
    target_compile_options(arbit-metal-rich-static-glb-tests PRIVATE
        -Wall -Wextra -Werror)
    add_test(NAME metal_rich_static_glb
        COMMAND arbit-metal-rich-static-glb-tests)
    set_tests_properties(metal_rich_static_glb PROPERTIES
        LABELS "strict-static-glb-runtime;physical-apple-silicon"
        TIMEOUT 600
        RUN_SERIAL TRUE)
    add_executable(arbit-holographic-trading-card-metal-strict-tests
        tests/metal_fixture_scene_backend_tests.mm
        tests/support/fixture_scene.cpp
        src/fixture_scene_renderer.cpp
        src/gltf_glb.cpp
        src/glb_scene_adapter.cpp
        src/animation_clip.cpp
        src/animation_playback_control.cpp
        src/animation_deformation_evaluation.cpp
        src/imported_animation_deformation_consumer.cpp
        src/deformation_contract.cpp
        src/native_animation_deformation.cpp
        src/gpu_backend/backend_sokol.mm)
    target_include_directories(arbit-holographic-trading-card-metal-strict-tests PRIVATE
        src tests third_party/sokol ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
    target_compile_definitions(arbit-holographic-trading-card-metal-strict-tests PRIVATE
        ARBIT_HAVE_VIEWPORT=0 ARBIT_HAVE_METAL_BACKEND=1
        ARBIT_HOLOGRAPHIC_TRADING_CARD_STRICT=1
        SURFACE_MATERIAL_STARTER_ORACLE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/surface_material_starter_expected.json")
    target_link_libraries(arbit-holographic-trading-card-metal-strict-tests PRIVATE
        nlohmann_json::nlohmann_json PkgConfig::AV
        "-framework Metal" "-framework QuartzCore" "-framework Cocoa")
    target_compile_options(arbit-holographic-trading-card-metal-strict-tests PRIVATE
        -Wall -Wextra -Werror)
    add_test(NAME holographic_trading_card_metal_strict
        COMMAND arbit-holographic-trading-card-metal-strict-tests)
    set_tests_properties(holographic_trading_card_metal_strict PROPERTIES
        LABELS "native-physical-diffraction-runtime;strict-metal-card"
        TIMEOUT 600
        RUN_SERIAL TRUE)

    if(GLFW3_FOUND AND OpenGL_FOUND)
        add_executable(arbit-metal-iosurface-consumer
            tests/metal_iosurface_consumer.mm)
        target_link_libraries(arbit-metal-iosurface-consumer PRIVATE
            "-framework Metal" "-framework IOSurface" "-framework Foundation")

        add_executable(arbit-metal-particle-smoke
            tests/metal_particle_smoke_tests.cpp
            src/particle_engine.cpp
            src/gl_loader.cpp
            src/gpu_backend/backend_sokol.mm)
        target_include_directories(arbit-metal-particle-smoke PRIVATE src third_party/sokol)
        target_compile_definitions(arbit-metal-particle-smoke PRIVATE
            ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=1)
        target_link_libraries(arbit-metal-particle-smoke PRIVATE
            PkgConfig::GLFW3 OpenGL::GL
            "-framework Metal" "-framework QuartzCore" "-framework Cocoa"
            "-framework IOSurface")

        add_executable(arbit-metal-color-transform-runtime-tests
            tests/metal_color_transform_runtime_tests.cpp
            src/renderer.cpp
            src/geometry_core_backend.cpp
            src/color_transform_gl.cpp
            src/sdf_native_renderer.cpp
            src/deformation_contract.cpp
            src/gl_loader.cpp
            src/lut_loader.cpp
            src/native_animation_deformation.cpp
            src/particle_engine.cpp
            src/shader_generator.cpp
            src/gpu_backend/backend_sokol.mm)
        target_include_directories(arbit-metal-color-transform-runtime-tests PRIVATE
            src third_party/sokol ${CMAKE_CURRENT_SOURCE_DIR}/../shared)
        target_compile_definitions(arbit-metal-color-transform-runtime-tests PRIVATE
            ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=1)
        target_link_libraries(arbit-metal-color-transform-runtime-tests PRIVATE
            PkgConfig::GLFW3 OpenGL::GL PkgConfig::AV
            "-framework Metal" "-framework QuartzCore" "-framework Cocoa"
            "-framework IOSurface" "-framework Foundation"
            nlohmann_json::nlohmann_json)
        target_compile_options(arbit-metal-color-transform-runtime-tests PRIVATE
            -Wall -Wextra -Werror)
        add_test(NAME metal_color_transform_runtime
            COMMAND arbit-metal-color-transform-runtime-tests)
        set_tests_properties(metal_color_transform_runtime PROPERTIES
            LABELS "native-color-transform-runtime;physical-apple-silicon"
            TIMEOUT 600
            RUN_SERIAL TRUE)

        add_executable(arbit-metal-frame-compositor-smoke
            tests/metal_frame_compositor_smoke_tests.cpp
            src/gl_loader.cpp
            src/gpu_backend/metal_shader_generator.mm
            src/gpu_backend/backend_sokol.mm)
        target_include_directories(arbit-metal-frame-compositor-smoke PRIVATE
            src third_party/sokol
            ${ARBIT_MVK_SHADER_CONVERTER_INCLUDE}
            ${ncnn_SOURCE_DIR}/glslang)
        target_compile_definitions(arbit-metal-frame-compositor-smoke PRIVATE
            ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=1
            ARBIT_HAVE_METAL_GENERATORS=1)
        target_link_libraries(arbit-metal-frame-compositor-smoke PRIVATE
            PkgConfig::GLFW3 OpenGL::GL
            "-framework Metal" "-framework QuartzCore" "-framework Cocoa"
            "-framework IOSurface" "-framework Foundation"
            nlohmann_json::nlohmann_json
            glslang SPIRV glslang-default-resource-limits
            ${ARBIT_MVK_SHADER_CONVERTER_LIBRARY}
            ${ARBIT_MVK_SPIRV_CROSS_LIBRARY}
            ${ARBIT_MVK_SPIRV_TOOLS_LIBRARY})
        add_dependencies(arbit-metal-frame-compositor-smoke
            arbit-metal-iosurface-consumer)
        add_test(NAME metal_frame_compositor_smoke
                 COMMAND arbit-metal-frame-compositor-smoke)

        add_executable(arbit-custom-shader-metal-pixel-tests
                tests/custom_shader_gl_pixel_tests.cpp
                src/renderer.cpp
                src/geometry_core_backend.cpp
                src/color_transform_gl.cpp
                src/gpu_backend/backend_sokol.mm
                src/gpu_backend/metal_shader_generator.mm
                src/sdf_native_renderer.cpp
                src/deformation_contract.cpp
                src/native_animation_deformation.cpp
                src/visual_plan_execution_telemetry.cpp
                src/visual_renderer_telemetry.cpp
                src/shader_generator.cpp
                src/particle_engine.cpp
                src/gl_loader.cpp
                src/lut_loader.cpp)
            target_include_directories(arbit-custom-shader-metal-pixel-tests PRIVATE
                src third_party/sokol
                ${CMAKE_CURRENT_SOURCE_DIR}/../shared
                ${ARBIT_MVK_SHADER_CONVERTER_INCLUDE}
                ${ncnn_SOURCE_DIR}/glslang)
            target_compile_definitions(arbit-custom-shader-metal-pixel-tests PRIVATE
                ARBIT_HAVE_VIEWPORT=1 ARBIT_HAVE_METAL_BACKEND=1
                ARBIT_HAVE_METAL_GENERATORS=1 ARBIT_HAVE_OPENGL_SDF_BACKEND=0
                ARBIT_STRICT_METAL_FIXTURE=1
                ARBIT_SHADER_PACK_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../content/shader-packs")
            target_link_libraries(arbit-custom-shader-metal-pixel-tests PRIVATE
                PkgConfig::GLFW3 OpenGL::GL PkgConfig::AV
                "-framework Metal" "-framework QuartzCore" "-framework Cocoa"
                "-framework IOSurface" "-framework Foundation"
                nlohmann_json::nlohmann_json
                glslang SPIRV glslang-default-resource-limits
                ${ARBIT_MVK_SHADER_CONVERTER_LIBRARY}
                ${ARBIT_MVK_SPIRV_CROSS_LIBRARY}
                ${ARBIT_MVK_SPIRV_TOOLS_LIBRARY})
            target_compile_options(arbit-custom-shader-metal-pixel-tests PRIVATE
                -Wall -Wextra -Werror)
            add_test(NAME custom_shader_metal_pixels
                COMMAND arbit-custom-shader-metal-pixel-tests)
        set_tests_properties(custom_shader_metal_pixels PROPERTIES
            LABELS "native-custom-shader-runtime;strict-metal-runtime"
            TIMEOUT 600
            RUN_SERIAL TRUE)
    endif()
endif()

if(Python3_Interpreter_FOUND AND UNIX)
    add_test(NAME concurrent_recorder_smoke
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/concurrent_recorder_smoke.py
                $<TARGET_FILE:arbit-video-helper>)
    add_test(NAME capture_source_smoke
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/capture_source_smoke.py
                $<TARGET_FILE:arbit-video-helper>)
    set_tests_properties(concurrent_recorder_smoke capture_source_smoke PROPERTIES
        TIMEOUT 30
        RUN_SERIAL TRUE)
endif()
