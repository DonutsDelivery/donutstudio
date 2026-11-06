#pragma once

// VA-API from GPU memory (Plan 3, export and encoders): each surface of the
// encoder's VA-API pool maps to DRM-PRIME once, and its two planes import into
// the export's GL context as EGLImage-backed textures (vaapi_drm_import.h).
// The GPU YUV pass renders into them, so a frame reaches the encoder with no
// readback and no upload. The export's GL context must be EGL-backed; a GLX
// context, or a build without EGL, names what is missing.

#include <cstdint>
#include <map>
#include <string>

struct AVBufferRef;
struct AVFrame;

namespace videohelper
{

class VaapiDrmFrameBridge
{
public:
    VaapiDrmFrameBridge() = default;
    ~VaapiDrmFrameBridge();
    VaapiDrmFrameBridge (const VaapiDrmFrameBridge&) = delete;
    VaapiDrmFrameBridge& operator= (const VaapiDrmFrameBridge&) = delete;

    // hwFrames is the encoder's VA-API AVHWFramesContext; bits is 8 for NV12
    // and 10 for P010. The export's GL context must be current. Returns false
    // with the reason when EGL or its dma-buf import is missing.
    bool open (AVBufferRef* hwFrames, int bits, std::string& error);
    bool isOpen() const noexcept { return hwFrames_ != nullptr; }

    // A frame from the pool, with its luma and chroma planes as GL textures to
    // render into. Call finish() after rendering and before encoding the
    // frame. The caller frees the frame. Returns nullptr with the reason.
    AVFrame* acquire (unsigned& lumaTexture, unsigned& chromaTexture, std::string& error);

    // Waits for the GL writes to the planes, so the encoder reads the frame.
    void finish();

    void close();

private:
    struct Surface
    {
        AVFrame* mapping = nullptr;  // the DRM-PRIME mapping, holding the dma-buf fds
        void* images[2] = {};        // EGLImageKHR
        unsigned textures[2] = {};
    };
    bool import (AVFrame* frame, Surface& surface, std::string& error);
    void release (Surface& surface);

    AVBufferRef* hwFrames_ = nullptr;
    int bits_ = 8;
    void* display_ = nullptr;  // EGLDisplay
    void* createImage_ = nullptr;
    void* destroyImage_ = nullptr;
    void* imageTargetTexture_ = nullptr;
    bool modifiers_ = false;
    std::map<std::uintptr_t, Surface> surfaces_;  // VASurfaceID -> imported planes
};

} // namespace videohelper
