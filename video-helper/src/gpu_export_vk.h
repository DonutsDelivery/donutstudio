// gpu_export_vk.h — VkFdExporter: allocates FBO-attached RGBA8 textures whose
// storage is a Vulkan VkDeviceMemory exported as an OPAQUE_FD, for the
// zero-copy docked viewport (SharedGpuSurfaceProtocol.h, kHandleVkFd).
//
// This is the Linux backend that works on the NVIDIA proprietary driver under
// GLX, where the EGL_MESA dmabuf path (gpu_export.h) is unavailable. Vulkan's
// only role is to allocate exportable memory (GL cannot export an fd); both the
// producer (this helper's GL) and the consumer (Arbit's JUCE GL) import that
// memory into a texture with GL_EXT_memory_object_fd — core GL, no EGL, so the
// consumer keeps its GLX context.
//
// Per buffer: create a VkImage (RGBA8, OPTIMAL tiling, external OPAQUE_FD) +
// dedicated VkDeviceMemory; export one fd; import a copy into THIS GL context
// (glCreateMemoryObjectsEXT + glImportMemoryFdEXT + glTexStorageMem2DEXT) and
// attach it to an FBO we render into; ship a second exported fd to the consumer.
// Allocation is once per buffer; per frame only ownership moves (blit + announce).
//
// Self-contained: owns a minimal VkInstance/VkDevice on the same GPU as the GL
// context, so the docked surface works even when RIFE/ncnn is not built.
#pragma once

#if ARBIT_HAVE_VKFD

#include "gl_loader.h"

#include <cstdint>
#include <string>

namespace gpuexp
{

// The common ExportedBuffer field set the generic BUFFERS announcement in
// viewport.cpp reads (width/height/fourcc/modifier/planeCount/fds/strides/
// offsets/busy), plus this backend's private VK + GL-import handles. Only ONE
// gpuexp::ExportedBuffer compiles per build (gpu_export_common.h includes a
// single backend header), so these definitions never collide.
struct ExportedBuffer
{
    unsigned tex = 0, fbo = 0;     // GL texture (backed by the VK memory) + its FBO
    int width = 0, height = 0;

    uint32_t fourcc = 0;           // always gpusurf::kHandleVkFd
    uint64_t modifier = 0;         // allocation size (VkMemoryRequirements.size)
    int planeCount = 0;            // 1 (one OPAQUE_FD memory fd)
    int fds[4] = { -1, -1, -1, -1 };       // fds[0] = the fd shipped to the consumer
    uint32_t strides[4] = {}, offsets[4] = {};

    // Private to the VK backend (opaque to viewport.cpp). VK handles stored as
    // uint64_t so this header pulls no Vulkan headers.
    uint64_t vkImage = 0;          // VkImage
    uint64_t vkMemory = 0;         // VkDeviceMemory
    unsigned glMemObject = 0;      // GL_EXT_memory_object handle for tex

    // Announced via FRAME_READY and not yet FRAME_RELEASEd. Managed by the
    // viewport render loop, not the exporter.
    bool busy = false;
};

class VkFdExporter
{
public:
    VkFdExporter() = default;
    ~VkFdExporter() { shutdown(); }
    VkFdExporter (const VkFdExporter&) = delete;
    VkFdExporter& operator= (const VkFdExporter&) = delete;

    // Creates the Vulkan instance/device (matching the GL context's GPU) and
    // resolves the GL_EXT_memory_object_fd entry points. GL context must be
    // current. Returns false with errorOut set when unavailable (no Vulkan
    // loader, no external_memory_fd, GL lacks GL_EXT_memory_object_fd, ...).
    bool initialize (const arbitgl::GlFuncs* gl, std::string& errorOut);
    bool available() const { return ready_; }

    // Present-path name for viewport_info.presentPath.
    const char* pathName() const { return "vkfd-docked"; }

    // GPU identity string (VkPhysicalDeviceIDProperties UUID hex) for the
    // consumer's multi-GPU demotion check; empty when unavailable.
    const std::string& devicePath() const { return deviceUuid_; }

    // Allocates one w*h RGBA8 buffer (VK memory + GL texture + FBO + exported
    // fd). GL context current.
    bool allocate (ExportedBuffer& b, int width, int height, std::string& errorOut);
    void destroy (ExportedBuffer& b);  // GL context current; frees VK + GL + fds

    // Copies srcTexture (same size as b) into b's FBO via glBlitFramebuffer.
    bool blit (unsigned srcTexture, const ExportedBuffer& b);

    // v1 sync model: producer glFinish()es; no cross-process GL fence yet.
    bool fenceSyncAvailable() const { return false; }
    int createNativeFenceFd() { return -1; }

    void shutdown();  // destroys the read FBO + tears down the Vulkan device

private:
    const arbitgl::GlFuncs* gl_ = nullptr;
    bool ready_ = false;
    unsigned readFbo_ = 0;
    std::string deviceUuid_;

    // Vulkan handles (opaque; real types live in the .cpp). Stored as void* /
    // uint64_t to keep Vulkan headers out of this header.
    void* vkInstance_ = nullptr;       // VkInstance
    void* vkPhysical_ = nullptr;       // VkPhysicalDevice
    void* vkDevice_ = nullptr;         // VkDevice
    void* vkGetMemoryFd_ = nullptr;    // PFN_vkGetMemoryFdKHR
    uint32_t memTypeIndex_ = 0xffffffffu;

    // Resolved GL_EXT_memory_object_fd entry points.
    void* glCreateMemoryObjects_ = nullptr;
    void* glDeleteMemoryObjects_ = nullptr;
    void* glMemoryObjectParameteriv_ = nullptr;
    void* glImportMemoryFd_ = nullptr;
    void* glTexStorageMem2D_ = nullptr;
    void* blitFramebuffer_ = nullptr;
};

} // namespace gpuexp

#endif // ARBIT_HAVE_VKFD
