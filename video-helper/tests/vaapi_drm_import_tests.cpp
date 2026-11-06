// VA-API from GPU memory (Plan 3, export and encoders): a VA-API surface's
// DRM-PRIME descriptor reads into a luma and a chroma import, and each import
// into the EGL attribute list that makes it a GL texture the YUV pass renders
// into. No device needed: the descriptors are built here as FFmpeg exports them.

#include "vaapi_drm_import.h"

#include <iostream>
#include <string>

namespace
{

namespace drm = videohelper::vaapidrm;

int checks = 0;
int failures = 0;

void check (bool condition, const std::string& message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

bool contains (const std::string& text, const std::string& part)
{
    return text.find (part) != std::string::npos;
}

// Two single-plane layers in one object, as vaExportSurfaceHandle with
// separate layers gives a 1920 x 1080 NV12 surface padded to 1088 rows.
AVDRMFrameDescriptor separateLayers (std::uint32_t luma, std::uint32_t chroma, std::uint64_t modifier)
{
    AVDRMFrameDescriptor descriptor {};
    descriptor.nb_objects = 1;
    descriptor.objects[0].fd = 42;
    descriptor.objects[0].format_modifier = modifier;
    descriptor.nb_layers = 2;
    descriptor.layers[0].format = luma;
    descriptor.layers[0].nb_planes = 1;
    descriptor.layers[0].planes[0] = { 0, 0, 2048 };
    descriptor.layers[1].format = chroma;
    descriptor.layers[1].nb_planes = 1;
    descriptor.layers[1].planes[0] = { 0, 2048 * 1088, 2048 };
    return descriptor;
}

void planeTests()
{
    std::string error;
    std::array<drm::PlaneImport, 2> planes;
    auto descriptor = separateLayers (drm::fourcc::kR8, drm::fourcc::kGr88, drm::kModifierLinear);
    check (drm::planeImports (descriptor, 1920, 1080, 8, planes, error), "an NV12 surface imports: " + error);
    check (planes[0].fourcc == drm::fourcc::kR8 && planes[0].width == 1920 && planes[0].height == 1080
               && planes[0].fd == 42 && planes[0].offset == 0 && planes[0].pitch == 2048,
           "luma is a full-size R8 plane at the object's start");
    check (planes[1].fourcc == drm::fourcc::kGr88 && planes[1].width == 960 && planes[1].height == 540
               && planes[1].offset == 2048 * 1088 && planes[1].pitch == 2048,
           "chroma is a half-size GR88 plane after the padded luma");

    check (drm::planeImports (descriptor, 1919, 1079, 8, planes, error) && planes[1].width == 960
               && planes[1].height == 540,
           "odd sizes round chroma up");

    descriptor = separateLayers (drm::fourcc::kR16, drm::fourcc::kGr1616, drm::kModifierInvalid);
    check (drm::planeImports (descriptor, 1920, 1080, 10, planes, error)
               && planes[0].fourcc == drm::fourcc::kR16 && planes[1].fourcc == drm::fourcc::kGr1616,
           "a P010 surface imports as R16 and GR1616: " + error);

    // Older drivers export one NV12 layer of two planes.
    AVDRMFrameDescriptor composed {};
    composed.nb_objects = 1;
    composed.objects[0].fd = 7;
    composed.objects[0].format_modifier = drm::kModifierInvalid;
    composed.nb_layers = 1;
    composed.layers[0].format = drm::fourcc::kNv12;
    composed.layers[0].nb_planes = 2;
    composed.layers[0].planes[0] = { 0, 0, 1920 };
    composed.layers[0].planes[1] = { 0, 1920 * 1080, 1920 };
    check (drm::planeImports (composed, 1920, 1080, 8, planes, error) && planes[0].fourcc == drm::fourcc::kR8
               && planes[1].fourcc == drm::fourcc::kGr88 && planes[1].offset == 1920 * 1080 && planes[1].fd == 7,
           "a composed NV12 layer splits into R8 and GR88: " + error);

    descriptor = separateLayers (drm::fourcc::kR8, drm::fourcc::kGr88, drm::kModifierLinear);
    error.clear();
    check (! drm::planeImports (descriptor, 1920, 1080, 10, planes, error) && contains (error, "R8 and GR88, not R16 and GR32"),
           "an 8-bit surface for a 10-bit pool is refused by name: " + error);
    descriptor.layers[1].planes[0].object_index = 1;
    error.clear();
    check (! drm::planeImports (descriptor, 1920, 1080, 8, planes, error) && contains (error, "object 1 of 1"),
           "a plane in a missing object is refused: " + error);
    descriptor.nb_layers = 3;
    error.clear();
    check (! drm::planeImports (descriptor, 1920, 1080, 8, planes, error) && contains (error, "3 layers"),
           "an unknown layout is refused: " + error);
}

void attributeTests()
{
    drm::PlaneImport plane;
    plane.fourcc = drm::fourcc::kGr88;
    plane.width = 960;
    plane.height = 540;
    plane.fd = 42;
    plane.offset = 2048 * 1088;
    plane.pitch = 2048;
    plane.modifier = drm::kModifierInvalid;

    std::string error;
    std::vector<std::int32_t> attributes;
    check (drm::imageAttributes (plane, false, attributes, error), "a plane without a modifier imports: " + error);
    const std::vector<std::int32_t> expected = {
        drm::egl::kWidth, 960, drm::egl::kHeight, 540, drm::egl::kLinuxDrmFourcc,
        static_cast<std::int32_t> (drm::fourcc::kGr88), drm::egl::kPlane0Fd, 42,
        drm::egl::kPlane0Offset, 2048 * 1088, drm::egl::kPlane0Pitch, 2048, drm::egl::kNone };
    check (attributes == expected, "the attribute list names size, fourcc, fd, offset and pitch");

    plane.modifier = 0x0100000000000002ULL;  // a tiled layout
    check (drm::imageAttributes (plane, true, attributes, error) && attributes.size() == expected.size() + 4
               && attributes[12] == drm::egl::kPlane0ModifierLo && attributes[13] == 2
               && attributes[14] == drm::egl::kPlane0ModifierHi && attributes[15] == 0x01000000
               && attributes.back() == drm::egl::kNone,
           "a tiled plane passes its modifier in two halves");
    error.clear();
    check (! drm::imageAttributes (plane, false, attributes, error)
               && contains (error, "0x100000000000002") && contains (error, "import_modifiers"),
           "a tiled plane without the modifier extension is refused by name: " + error);

    plane.modifier = drm::kModifierLinear;
    check (drm::imageAttributes (plane, false, attributes, error) && attributes == expected,
           "a linear plane imports without the modifier extension");

    check (drm::fourcc::name (drm::fourcc::kR8) == "R8" && drm::fourcc::name (drm::fourcc::kGr1616) == "GR32",
           "fourccs read as their DRM names");
}

} // namespace

int main()
{
    planeTests();
    attributeTests();
    std::cout << (checks - failures) << "/" << checks << " VA-API DRM import checks passed\n";
    return failures == 0 ? 0 : 1;
}
