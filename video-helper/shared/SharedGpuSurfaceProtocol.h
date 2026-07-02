// SharedGpuSurfaceProtocol.h — wire protocol for the zero-copy GPU surface
// channel between arbit-video-helper (producer) and Arbit (consumer).
//
// Shared dependency-free header (like VideoFrameSharedMemory.h): plain POD
// structs, no JUCE/libav/GL. The channel is a unix domain socket created and
// listened on by the HELPER at the path returned from the
// `viewport_open_shared` RPC. Per-platform transport + buffer handles:
//
//   Linux   — SOCK_SEQPACKET; one sendmsg() == one datagram == one message.
//             Buffers are dmabufs: fds ride in SCM_RIGHTS ancillary data
//             (the stdio JSON-RPC channel cannot carry fds), BufferInfo
//             carries DRM fourcc/modifier/planes.
//   macOS   — SOCK_STREAM (no SOCK_SEQPACKET on Darwin); messages are framed
//             by MsgHeader.payloadBytes. Buffers are global IOSurfaces:
//             BufferInfo.modifier = IOSurfaceID (uint32, cross-process via
//             IOSurfaceLookup — nothing to attach), fourcc = kHandleIOSurface,
//             planeCount = 0, fdCount always 0.
//   Windows — AF_UNIX SOCK_STREAM (Win10 1803+), framed like macOS. Buffers
//             are D3D11 keyed-mutex shared textures: BufferInfo.modifier =
//             the legacy DXGI shared HANDLE value (global, opened with
//             ID3D11Device::OpenSharedResource — no handle duplication),
//             fourcc = kHandleD3D11, planeCount = 0, fdCount always 0.
//             Keyed-mutex discipline: producer AcquireSync(0) → write →
//             ReleaseSync(1); consumer AcquireSync(1) → sample (held while
//             displayed) → ReleaseSync(0) before FRAME_RELEASE.
//
// Message = one MsgHeader followed by `payloadBytes` of payload (same
// datagram on Linux, contiguous bytes on the stream transports). Helper
// restart closes the socket; EOF on either side invalidates all announced
// buffers.
//
// Flow:
//   helper → arbit : HELLO          (version handshake; no fds)
//   helper → arbit : BUFFERS        (N BufferInfo records; fds attached in
//                                    buffer order, planes within a buffer in
//                                    plane order)
//   helper → arbit : FRAME_READY    (buffer finished rendering; when
//                                    kFlagFenceSync is set one acquire-fence
//                                    fd rides along and the consumer GPU-waits
//                                    on it before sampling; otherwise the
//                                    producer glFinish()ed before sending)
//   arbit  → helper : FRAME_RELEASE (consumer done sampling the buffer)
//   helper → arbit : BUFFERS_GONE   (resize/teardown; consumer must drop all
//                                    imports, new BUFFERS follows if alive)
#pragma once

#include <cstdint>

namespace gpusurf
{

static constexpr uint32_t kMagic = 0x41475055;   // 'APGU' little-endian bytes "UPGA" — Arbit GPU surface
static constexpr uint32_t kProtocolVersion = 2;  // v2 adds the vkfd buffer kind (kHandleVkFd)

// Hello.flags / FrameReadyPayload.flags bit: producer attaches an
// EGL_ANDROID_native_fence_sync acquire-fence fd to FRAME_READY instead of
// glFinish()ing. Release direction (FRAME_RELEASE fences) remains v1.
// Linux-only — the macOS producer glFinish()es (IOSurface coherence) and the
// Windows producer orders writes through the keyed mutex.
static constexpr uint32_t kFlagFenceSync = 1u << 0;

// BufferInfo.fourcc tags for the non-dmabuf transports (DRM fourccs never
// collide with these — both are valid printable fourccs we mint ourselves).
static constexpr uint32_t kHandleIOSurface = 0x46534F49; // 'IOSF'
static constexpr uint32_t kHandleD3D11     = 0x48443344; // 'D3DH'
// Linux zero-copy via Vulkan external memory instead of EGL dmabuf — the path
// that works on the NVIDIA proprietary driver + GLX (where EGL_MESA dmabuf
// export is unavailable). The buffer is a VkDeviceMemory exported as an
// OPAQUE_FD; one fd rides in SCM_RIGHTS (planeCount = 1). Both producer and
// consumer import it into GL with GL_EXT_memory_object_fd (glImportMemoryFdEXT
// + glTexStorageMem2DEXT), which is core GL — no EGL, so JUCE keeps its GLX
// context. Encoding: fourcc = kHandleVkFd; modifier = the allocation size
// (VkMemoryRequirements.size) passed to glImportMemoryFdEXT; planes[0].offset =
// memory offset (0 for a dedicated allocation), planes[0].stride unused; format
// is implicitly RGBA8 with GL_OPTIMAL_TILING_EXT. Sync (v1): the producer
// glFinish()es before FRAME_READY; kFlagFenceSync is not used for vkfd v1.
static constexpr uint32_t kHandleVkFd      = 0x44464B56; // 'VKFD'

enum class MsgType : uint32_t
{
    Hello = 1,
    Buffers = 2,
    BuffersGone = 3,
    FrameReady = 4,
    FrameRelease = 5,
};

#pragma pack(push, 1)

struct MsgHeader
{
    uint32_t magic = kMagic;
    uint32_t type = 0;          // MsgType
    uint32_t fdCount = 0;       // fds attached via SCM_RIGHTS to this datagram
    uint32_t payloadBytes = 0;  // bytes following this header
};

struct HelloPayload
{
    uint32_t version = kProtocolVersion;
    uint32_t flags = 0;
};

static constexpr uint32_t kMaxPlanes = 4;
static constexpr uint32_t kMaxBuffers = 4;

struct PlaneInfo
{
    uint32_t strideBytes = 0;
    uint32_t offsetBytes = 0;
};

struct BufferInfo
{
    uint32_t index = 0;         // 0..bufferCount-1, referenced by FRAME_READY/RELEASE
    uint32_t fourcc = 0;        // DRM fourcc (e.g. DRM_FORMAT_ABGR8888), or
                                // kHandleIOSurface / kHandleD3D11 on mac/win
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t modifier = 0;      // Linux: DRM format modifier; macOS:
                                // IOSurfaceID; Windows: DXGI shared HANDLE
    uint32_t planeCount = 0;    // 0 on the handle transports (no fds)
    PlaneInfo planes[kMaxPlanes];
};

struct BuffersPayload
{
    uint32_t bufferCount = 0;
    // followed by bufferCount * BufferInfo; fds attached in buffer order,
    // planes within each buffer in plane order (sum of planeCount == fdCount)
};

struct FrameReadyPayload
{
    uint32_t bufferIndex = 0;
    uint32_t flags = 0;         // kFlagFenceSync ⇒ one acquire-fence fd attached
    uint64_t frameSeq = 0;      // monotonic presented-frame counter
    double displaySec = 0.0;    // timeline time this frame represents
    uint64_t frameHash = 0;     // FNV-1a of source frame (testing/verification)
};

struct FrameReleasePayload
{
    uint32_t bufferIndex = 0;
    uint32_t flags = 0;         // kFlagFenceSync ⇒ one release-fence fd attached
};

#pragma pack(pop)

// Present-path names reported by viewport_info.presentPath — the fallback
// ladder, best first. The zero-copy docked rung's name states the platform
// mechanism actually in use.
static constexpr const char* kPathDmabufDocked    = "dmabuf-docked";    // Linux (EGL dmabuf)
static constexpr const char* kPathVkFdDocked      = "vkfd-docked";      // Linux (Vulkan OPAQUE_FD)
static constexpr const char* kPathIOSurfaceDocked = "iosurface-docked"; // macOS
static constexpr const char* kPathD3D11Docked     = "d3d11-docked";     // Windows
static constexpr const char* kPathShmDocked       = "shm-docked";
static constexpr const char* kPathGlfwWindow      = "glfw-window";
static constexpr const char* kPathCpuShm          = "cpu-shm";

} // namespace gpusurf
