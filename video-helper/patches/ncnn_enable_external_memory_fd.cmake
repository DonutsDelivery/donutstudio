# Idempotent ncnn patch (applied as a FetchContent PATCH_COMMAND).
#
# ncnn 20250503 enables the base VK_KHR_external_memory device extension but NOT
# the fd-based variants (VK_KHR_external_memory_fd / _semaphore / _semaphore_fd)
# that Arbit's Phase-1b zero-copy path needs to export a RIFE output VkBuffer to
# OpenGL via OPAQUE_FD. This inserts a probe-and-enable block into the device
# create path that enables ONLY the fd extensions the physical device actually
# advertises (so vkCreateDevice never fails on GPUs lacking them). The whole block
# is guarded by ARBIT_RIFE_ZEROCOPY, so a stock build (define unset) is unchanged.
#
# Invoked: cmake -DNCNN_GPU_CPP=<path/to/src/gpu.cpp> -P this.cmake
# Idempotent: re-running is a no-op once the marker is present.

if(NOT DEFINED NCNN_GPU_CPP)
    message(FATAL_ERROR "NCNN_GPU_CPP must be set")
endif()
if(NOT EXISTS "${NCNN_GPU_CPP}")
    message(FATAL_ERROR "ncnn gpu.cpp not found at ${NCNN_GPU_CPP}")
endif()

file(READ "${NCNN_GPU_CPP}" _src)

if(_src MATCHES "ARBIT_RIFE_ZEROCOPY external_memory_fd")
    message(STATUS "ncnn external-memory-fd patch already applied")
    return()
endif()

# Single-line anchor (no embedded newlines -> no escape ambiguity). Unique in gpu.cpp.
set(_anchor "        enabledExtensions.push_back(\"VK_KHR_external_memory\");")

# Replacement = the anchor line followed by the gated probe-and-enable block. This
# is a real multi-line string literal: the newlines below are preserved verbatim.
set(_inject "        enabledExtensions.push_back(\"VK_KHR_external_memory\");
#ifdef ARBIT_RIFE_ZEROCOPY
    // ARBIT_RIFE_ZEROCOPY external_memory_fd: enable the fd-based external memory/
    // semaphore extensions so a RIFE output buffer can be exported to OpenGL. Probe
    // the physical device and enable only what is supported (vkCreateDevice would
    // fail otherwise on GPUs without them).
    {
        uint32_t arbit_ext_count = 0;
        vkEnumerateDeviceExtensionProperties(info.physicalDevice(), 0, &arbit_ext_count, 0);
        std::vector<VkExtensionProperties> arbit_exts(arbit_ext_count);
        if (arbit_ext_count)
            vkEnumerateDeviceExtensionProperties(info.physicalDevice(), 0, &arbit_ext_count, arbit_exts.data());
        for (uint32_t ai = 0; ai < arbit_ext_count; ai++)
        {
            const char* en = arbit_exts[ai].extensionName;
            if (strcmp(en, \"VK_KHR_external_memory_fd\") == 0) enabledExtensions.push_back(\"VK_KHR_external_memory_fd\");
            else if (strcmp(en, \"VK_KHR_external_semaphore\") == 0) enabledExtensions.push_back(\"VK_KHR_external_semaphore\");
            else if (strcmp(en, \"VK_KHR_external_semaphore_fd\") == 0) enabledExtensions.push_back(\"VK_KHR_external_semaphore_fd\");
        }
    }
#endif")

string(FIND "${_src}" "${_anchor}" _pos)
if(_pos EQUAL -1)
    message(FATAL_ERROR "ncnn external-memory-fd patch: anchor not found in ${NCNN_GPU_CPP} (ncnn version changed?)")
endif()

string(REPLACE "${_anchor}" "${_inject}" _src "${_src}")
file(WRITE "${NCNN_GPU_CPP}" "${_src}")
message(STATUS "ncnn external-memory-fd patch applied to ${NCNN_GPU_CPP}")
