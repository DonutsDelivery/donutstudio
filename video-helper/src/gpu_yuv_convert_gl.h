#pragma once

// The GPU half of yuv_encode_math.h: one GL pass writes a composite into a luma
// plane (R8 for NV12, R16 for P010) and a half-size chroma plane (RG8 or RG16)
// that CUDA-GL interop or a DRM-PRIME import hands to the encoder. Rows stay top
// first, as the composite's.

#include "gl_loader.h"
#include "yuv_encode_math.h"

#include <string>

namespace videorender
{
class YuvPlanesGl final
{
public:
    bool initialize (arbitgl::GlFuncs* gl, std::string& error);
    // Converts sourceTexture (width x height, even sizes) into the planes this
    // pass owns, reallocating them when the size or depth changes.
    bool render (unsigned sourceTexture, int width, int height,
                 const videohelper::yuvencode::Conversion& conversion, std::string& error);
    // Writes into planes the caller owns, such as a VA-API surface's imported
    // as GL textures. Each must match the pass's formats for the depth.
    bool renderInto (unsigned sourceTexture, unsigned lumaTexture, unsigned chromaTexture, int width, int height,
                     const videohelper::yuvencode::Conversion& conversion, std::string& error);
    unsigned lumaTexture() const noexcept { return luma_; }
    unsigned chromaTexture() const noexcept { return chroma_; }
    void shutdown() noexcept;
    bool ready() const noexcept { return gl_ != nullptr && program_ != 0 && vao_ != 0; }

private:
    bool draw (unsigned sourceTexture, unsigned target, int width, int height, int plane,
               const videohelper::yuvencode::ShaderConstants& constants, std::string& error);

    arbitgl::GlFuncs* gl_ = nullptr;
    unsigned program_ = 0;
    unsigned vao_ = 0;
    unsigned framebuffer_ = 0;
    unsigned luma_ = 0, chroma_ = 0;
    int planesWidth_ = 0, planesHeight_ = 0, planesBits_ = 0;
};
} // namespace videorender
