#pragma once

// VA-API from GPU memory (Plan 3, export and encoders). A VA-API surface from
// the encoder's pool maps to DRM-PRIME, and its luma and chroma planes import
// into the export's GL context as EGLImages (EGL_EXT_image_dma_buf_import),
// so the GPU YUV pass writes straight into the surface the encoder reads.
//
// This header reads FFmpeg's DRM descriptor into one import per plane and
// builds the EGL attribute list for each. It needs no EGL headers: the
// attribute names are the EGL_EXT_image_dma_buf_import(_modifiers) values.

extern "C" {
#include <libavutil/hwcontext_drm.h>
}

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace videohelper::vaapidrm
{

namespace fourcc
{
constexpr std::uint32_t code (char a, char b, char c, char d) noexcept
{
    return static_cast<std::uint32_t> (static_cast<unsigned char> (a))
         | static_cast<std::uint32_t> (static_cast<unsigned char> (b)) << 8
         | static_cast<std::uint32_t> (static_cast<unsigned char> (c)) << 16
         | static_cast<std::uint32_t> (static_cast<unsigned char> (d)) << 24;
}
constexpr std::uint32_t kR8 = code ('R', '8', ' ', ' ');
constexpr std::uint32_t kGr88 = code ('G', 'R', '8', '8');
constexpr std::uint32_t kR16 = code ('R', '1', '6', ' ');
constexpr std::uint32_t kGr1616 = code ('G', 'R', '3', '2');
constexpr std::uint32_t kNv12 = code ('N', 'V', '1', '2');
constexpr std::uint32_t kP010 = code ('P', '0', '1', '0');

inline std::string name (std::uint32_t value)
{
    std::string text;
    for (int shift = 0; shift < 32; shift += 8)
    {
        const char c = static_cast<char> ((value >> shift) & 0xff);
        text += c >= 32 && c < 127 ? c : '?';
    }
    while (! text.empty() && text.back() == ' ')
        text.pop_back();
    return text;
}
} // namespace fourcc

constexpr std::uint64_t kModifierInvalid = 0x00ffffffffffffffULL;
constexpr std::uint64_t kModifierLinear = 0;

namespace egl
{
constexpr std::int32_t kNone = 0x3038, kHeight = 0x3056, kWidth = 0x3057;
constexpr std::int32_t kLinuxDrmFourcc = 0x3271, kPlane0Fd = 0x3272, kPlane0Offset = 0x3273, kPlane0Pitch = 0x3274;
constexpr std::int32_t kPlane0ModifierLo = 0x3443, kPlane0ModifierHi = 0x3444;
} // namespace egl

// One plane as a single-plane EGLImage: R8 or R16 luma, GR88 or GR1616 chroma.
struct PlaneImport
{
    std::uint32_t fourcc = 0;
    int width = 0, height = 0;
    int fd = -1;
    std::ptrdiff_t offset = 0, pitch = 0;
    std::uint64_t modifier = kModifierInvalid;
};

// Luma, then chroma, from a frame of width x height. FFmpeg exports a VA-API
// NV12 surface as two single-plane layers (R8, GR88), or on older drivers as
// one NV12 layer of two planes; P010 likewise as R16 and GR1616, or P010.
inline bool planeImports (const AVDRMFrameDescriptor& descriptor, int width, int height, int bits,
                          std::array<PlaneImport, 2>& imports, std::string& error)
{
    const bool tenBit = bits == 10;
    const std::uint32_t expected[2] = { tenBit ? fourcc::kR16 : fourcc::kR8, tenBit ? fourcc::kGr1616 : fourcc::kGr88 };
    const std::uint32_t composed = tenBit ? fourcc::kP010 : fourcc::kNv12;

    const AVDRMPlaneDescriptor* planes[2] = {};
    if (descriptor.nb_layers == 2 && descriptor.layers[0].nb_planes == 1 && descriptor.layers[1].nb_planes == 1)
    {
        if (descriptor.layers[0].format != expected[0] || descriptor.layers[1].format != expected[1])
        {
            error = "VA-API exported its planes as " + fourcc::name (descriptor.layers[0].format) + " and "
                + fourcc::name (descriptor.layers[1].format) + ", not " + fourcc::name (expected[0]) + " and "
                + fourcc::name (expected[1]);
            return false;
        }
        planes[0] = &descriptor.layers[0].planes[0];
        planes[1] = &descriptor.layers[1].planes[0];
    }
    else if (descriptor.nb_layers == 1 && descriptor.layers[0].nb_planes == 2)
    {
        if (descriptor.layers[0].format != composed)
        {
            error = "VA-API exported a " + fourcc::name (descriptor.layers[0].format) + " surface, not "
                + fourcc::name (composed);
            return false;
        }
        planes[0] = &descriptor.layers[0].planes[0];
        planes[1] = &descriptor.layers[0].planes[1];
    }
    else
    {
        error = "VA-API exported " + std::to_string (descriptor.nb_layers) + " layers, not a luma and a chroma plane";
        return false;
    }

    for (int i = 0; i < 2; ++i)
    {
        const int object = planes[i]->object_index;
        if (object < 0 || object >= descriptor.nb_objects)
        {
            error = "VA-API exported a plane in object " + std::to_string (object) + " of "
                + std::to_string (descriptor.nb_objects);
            return false;
        }
        auto& import = imports[static_cast<std::size_t> (i)];
        import.fourcc = expected[i];
        import.width = i == 0 ? width : (width + 1) / 2;
        import.height = i == 0 ? height : (height + 1) / 2;
        import.fd = descriptor.objects[object].fd;
        import.offset = planes[i]->offset;
        import.pitch = planes[i]->pitch;
        import.modifier = descriptor.objects[object].format_modifier;
    }
    return true;
}

// The attribute list for eglCreateImageKHR (EGL_LINUX_DMA_BUF_EXT). A tiled
// surface needs the modifier extension; a linear one or one without a modifier
// imports without it.
inline bool imageAttributes (const PlaneImport& import, bool modifiersSupported,
                             std::vector<std::int32_t>& attributes, std::string& error)
{
    attributes = { egl::kWidth, import.width, egl::kHeight, import.height,
                   egl::kLinuxDrmFourcc, static_cast<std::int32_t> (import.fourcc),
                   egl::kPlane0Fd, import.fd,
                   egl::kPlane0Offset, static_cast<std::int32_t> (import.offset),
                   egl::kPlane0Pitch, static_cast<std::int32_t> (import.pitch) };
    if (import.modifier != kModifierInvalid)
    {
        if (modifiersSupported)
        {
            attributes.push_back (egl::kPlane0ModifierLo);
            attributes.push_back (static_cast<std::int32_t> (import.modifier & 0xffffffffu));
            attributes.push_back (egl::kPlane0ModifierHi);
            attributes.push_back (static_cast<std::int32_t> (import.modifier >> 32));
        }
        else if (import.modifier != kModifierLinear)
        {
            char hex[24] = {};
            std::snprintf (hex, sizeof (hex), "0x%llx", static_cast<unsigned long long> (import.modifier));
            error = std::string ("the VA-API surface is tiled (modifier ") + hex
                + ") and EGL lacks EGL_EXT_image_dma_buf_import_modifiers";
            attributes.clear();
            return false;
        }
    }
    attributes.push_back (egl::kNone);
    return true;
}

} // namespace videohelper::vaapidrm
