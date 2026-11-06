// The Metal Path Trace upload (cpu_scene_frame_metal.mm) on the sokol Metal
// device: rows arrive top first, SDR colour as BGRA8, frame layers and the pass
// resolve match the OpenGL passes, and the frame carries every property the
// Metal compositor and the imported scene layer check before they sample it.
//
// With "--ppm in.ppm out.ppm" it also uploads an 8-bit PPM as a Path Trace
// image and writes back what the compositor's texture holds.

#include "gpu_backend/cpu_scene_frame.h"
#include "gpu_backend/sokol_metal_context.h"

#import <Metal/Metal.h>

#define SOKOL_METAL
#include "sokol_gfx.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace
{
int failures = 0;

void check (bool condition, const std::string& what)
{
    if (condition) return;
    ++failures;
    std::printf ("FAIL: %s\n", what.c_str());
}

std::uint16_t halfBits (float value)
{
    const __fp16 half = static_cast<__fp16> (value);
    std::uint16_t bits = 0;
    std::memcpy (&bits, &half, sizeof bits);
    return bits;
}

float halfValue (std::uint16_t bits)
{
    __fp16 half;
    std::memcpy (&half, &bits, sizeof half);
    return static_cast<float> (half);
}

std::uintptr_t metalDevice() { return reinterpret_cast<std::uintptr_t> (arbitgpu::sokolmetal::device()); }

// The tight rows of a sokol image, through a blit on sokol's own queue so the
// copy runs after the passes the upload committed.
std::vector<std::uint8_t> readImage (std::uintptr_t imageId, std::size_t pixelBytes)
{
    std::lock_guard<std::mutex> lock (arbitgpu::sokolmetal::mutex());
    const sg_image image { static_cast<std::uint32_t> (imageId) };
    const auto info = sg_mtl_query_image_info (image);
    id<MTLTexture> texture = (__bridge id<MTLTexture>) info.tex[info.active_slot];
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
    const std::size_t rowBytes = pixelBytes * texture.width;
    id<MTLBuffer> buffer = [queue.device newBufferWithLength: rowBytes * texture.height
                                                     options: MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    [blit copyFromTexture: texture sourceSlice: 0 sourceLevel: 0 sourceOrigin: MTLOriginMake (0, 0, 0)
               sourceSize: MTLSizeMake (texture.width, texture.height, 1) toBuffer: buffer destinationOffset: 0
      destinationBytesPerRow: rowBytes destinationBytesPerImage: rowBytes * texture.height];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    std::vector<std::uint8_t> out (rowBytes * texture.height);
    std::memcpy (out.data(), buffer.contents, out.size());
    [buffer release];
    return out;
}

// The checks the Metal compositor (backend_sokol.mm, the layer admission) and
// imported_scene_visual_plan_execution.h apply to a scene frame.
bool compositorAccepts (const arbitgpu::NativeFixtureSceneFrame& frame, bool linear, bool outputs, std::string& why)
{
    using arbitgpu::NativeTexturePixelFormat;
    const auto color = frame.colorTextureDescriptor();
    const auto depth = frame.depthTextureDescriptor();
    const auto exact = [&] (const arbitgpu::NativeTextureViewDescriptor& d, NativeTexturePixelFormat format,
                            std::uintptr_t image, std::uintptr_t view)
    {
        return d.complete() && d.backend == "metal" && d.format == format && d.imageHandle == image
            && d.textureViewHandle == view && d.width == frame.width() && d.height == frame.height()
            && d.deviceOrContextIdentity == metalDevice() && d.rowOrder == arbitgpu::NativeTextureRowOrder::TopFirst;
    };
    if (frame.backend() != "metal") { why = "backend"; return false; }
    if (! exact (color, linear ? NativeTexturePixelFormat::Rgba16Float : NativeTexturePixelFormat::Bgra8Unorm,
                 frame.colorImageHandle(), frame.colorTextureViewHandle()))
    { why = "colour descriptor"; return false; }
    if (linear && ! arbitgpu::isLinearSceneColor (color)) { why = "linear scene colour"; return false; }
    if (! exact (depth, NativeTexturePixelFormat::R32Float, frame.depthImageHandle(), frame.depthTextureViewHandle()))
    { why = "depth descriptor"; return false; }
    std::lock_guard<std::mutex> lock (arbitgpu::sokolmetal::mutex());
    const sg_view view { static_cast<std::uint32_t> (color.textureViewHandle) };
    const sg_image image { static_cast<std::uint32_t> (color.imageHandle) };
    if (sg_query_view_state (view) != SG_RESOURCESTATE_VALID || sg_query_view_type (view) != SG_VIEWTYPE_TEXTURE
        || sg_query_image_state (image) != SG_RESOURCESTATE_VALID)
    { why = "colour resources"; return false; }
    const auto storage = sg_query_image_desc (image);
    if (sg_query_view_desc (view).texture.image.id != image.id || storage.type != SG_IMAGETYPE_2D
        || storage.width != static_cast<int> (frame.width()) || storage.height != static_cast<int> (frame.height())
        || storage.sample_count != 1
        || storage.pixel_format != (linear ? SG_PIXELFORMAT_RGBA16F : SG_PIXELFORMAT_BGRA8))
    { why = "colour storage"; return false; }
    if (sg_query_view_state (sg_view { static_cast<std::uint32_t> (depth.textureViewHandle) }) != SG_RESOURCESTATE_VALID)
    { why = "depth view"; return false; }
    if (outputs)
    {
        const NativeTexturePixelFormat formats[6] { NativeTexturePixelFormat::Rgba16Float,
            NativeTexturePixelFormat::Rgba16Float, NativeTexturePixelFormat::R8Unorm, NativeTexturePixelFormat::R32Uint,
            NativeTexturePixelFormat::R32Uint, NativeTexturePixelFormat::Rg16Float };
        for (std::size_t i = 0; i < 6; ++i)
        {
            const auto raw = frame.passTextureDescriptor (renderpasscomposite::kInputs[i + 2]);
            if (! exact (raw, formats[i], raw.imageHandle, raw.textureViewHandle)
                || raw.rendererGeneration != color.rendererGeneration
                || sg_query_view_state (sg_view { static_cast<std::uint32_t> (raw.textureViewHandle) })
                       != SG_RESOURCESTATE_VALID)
            { why = "raw pass " + std::to_string (i); return false; }
        }
    }
    return true;
}

// A 2x2 BGRA8 frame texture, top first, standing in for a material's frame.
class TestFrame final : public arbitgpu::NativeFixtureSceneFrame
{
public:
    explicit TestFrame (const std::uint8_t (&bgra)[16])
    {
        std::lock_guard<std::mutex> lock (arbitgpu::sokolmetal::mutex());
        sg_image_desc desc = {};
        desc.width = 2;
        desc.height = 2;
        desc.pixel_format = SG_PIXELFORMAT_BGRA8;
        desc.data.mip_levels[0] = { bgra, 16 };
        image_ = sg_make_image (&desc);
        sg_view_desc view = {};
        view.texture.image = image_;
        view_ = sg_make_view (&view);
    }
    ~TestFrame() override
    {
        std::lock_guard<std::mutex> lock (arbitgpu::sokolmetal::mutex());
        sg_destroy_view (view_);
        sg_destroy_image (image_);
    }
    const std::string& backend() const noexcept override { return backend_; }
    std::uint32_t width() const noexcept override { return 2; }
    std::uint32_t height() const noexcept override { return 2; }
    std::uintptr_t colorImageHandle() const noexcept override { return image_.id; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return view_.id; }
    arbitgpu::NativeTextureViewDescriptor colorTextureDescriptor() const noexcept override
    {
        return { backend_, arbitgpu::NativeTextureViewKind::Texture2D, arbitgpu::NativeTexturePixelFormat::Bgra8Unorm,
                 image_.id, view_.id, 2, 2, 1, true, metalDevice(), 1 };
    }

private:
    sg_image image_ {};
    sg_view view_ {};
    const std::string backend_ = "metal";
};

float srgbToLinear (float c) { return c <= 0.04045f ? c / 12.92f : std::pow ((c + 0.055f) / 1.055f, 2.4f); }

// Bottom-first source pixel of the top-first texture pixel (x, y).
std::size_t source (std::uint32_t x, std::uint32_t y, std::uint32_t width, std::uint32_t height)
{
    return static_cast<std::size_t> (height - 1 - y) * width + x;
}

void sdrNoLayers()
{
    const std::uint32_t w = 7, h = 5;
    std::vector<std::uint8_t> color (w * h * 4);
    std::vector<float> depth (w * h);
    for (std::size_t i = 0; i < w * h; ++i)
    {
        color[i * 4 + 0] = static_cast<std::uint8_t> (i);
        color[i * 4 + 1] = static_cast<std::uint8_t> (100 + i);
        color[i * 4 + 2] = static_cast<std::uint8_t> (200 - i);
        color[i * 4 + 3] = 255;
        depth[i] = 1.0f + static_cast<float> (i);
    }
    arbitgpu::CpuSceneImage image;
    image.width = w;
    image.height = h;
    image.color = &color;
    image.depth = &depth;
    std::string error;
    const auto frame = arbitgpu::uploadCpuSceneFrame ("metal", image, error);
    check (frame != nullptr, "SDR upload: " + error);
    if (! frame) return;
    std::string why;
    check (compositorAccepts (*frame, false, false, why), "SDR frame refused by the compositor checks: " + why);
    const auto texels = readImage (frame->colorImageHandle(), 4);
    const auto depths = readImage (frame->depthImageHandle(), 4);
    bool rows = true;
    for (std::uint32_t y = 0; y < h; ++y)
        for (std::uint32_t x = 0; x < w; ++x)
        {
            const auto s = source (x, y, w, h);
            const auto* t = &texels[(y * w + x) * 4];
            float d = 0.0f;
            std::memcpy (&d, &depths[(y * w + x) * 4], 4);
            rows = rows && t[0] == color[s * 4 + 2] && t[1] == color[s * 4 + 1] && t[2] == color[s * 4 + 0]
                && t[3] == color[s * 4 + 3] && d == depth[s];
        }
    check (rows, "SDR colour lands top first as BGRA8 and depth top first");
    check (arbitgpu::uploadCpuSceneFrame ("opengl", image, error) == nullptr, "the Metal build refuses another backend");
}

void linearFrameLayer()
{
    const std::uint32_t w = 6, h = 4;
    std::vector<std::uint8_t> color (w * h * 8);
    std::vector<float> depth (w * h, 2.0f);
    std::vector<std::uint8_t> light (w * h * 8);
    std::vector<float> uv (w * h * 3);
    const float base[4] { 0.1f, 0.2f, 0.3f, 1.0f }, lit[3] { 2.0f, 1.0f, 0.5f };
    for (std::size_t i = 0; i < w * h; ++i)
    {
        for (int c = 0; c < 4; ++c)
        {
            const auto bits = halfBits (base[c]);
            std::memcpy (&color[i * 8 + c * 2], &bits, 2);
            const auto lightBits = halfBits (c < 3 ? lit[c] : 1.0f);
            std::memcpy (&light[i * 8 + c * 2], &lightBits, 2);
        }
        // Texel centres of the 2x2 frame, so a linear sample is exact. Every
        // fourth pixel has no coverage and keeps the traced colour.
        uv[i * 3 + 0] = (i % 2) ? 0.75f : 0.25f;
        uv[i * 3 + 1] = ((i / 2) % 2) ? 0.75f : 0.25f;
        uv[i * 3 + 2] = (i % 4 == 3) ? 0.0f : 1.0f;
    }
    // BGRA8, top first: red opaque, green half alpha, blue opaque, white clear.
    const std::uint8_t frameTexels[16] { 0, 0, 255, 255, 0, 255, 0, 128, 255, 0, 0, 255, 255, 255, 255, 0 };
    const auto layerFrame = std::make_shared<TestFrame> (frameTexels);
    arbitgpu::CpuSceneImage image;
    image.width = w;
    image.height = h;
    image.linearColor = true;
    image.color = &color;
    image.depth = &depth;
    image.background[0] = 0.05f;
    image.background[1] = 0.06f;
    image.background[2] = 0.07f;
    image.frameLayers.push_back ({ layerFrame, &light, &uv });
    std::string error;
    const auto frame = arbitgpu::uploadCpuSceneFrame ("metal", image, error);
    check (frame != nullptr, "frame layer upload: " + error);
    if (! frame) return;
    std::string why;
    check (compositorAccepts (*frame, true, false, why), "layered frame refused by the compositor checks: " + why);
    const auto texels = readImage (frame->colorImageHandle(), 8);
    double worst = 0.0;
    for (std::uint32_t y = 0; y < h; ++y)
        for (std::uint32_t x = 0; x < w; ++x)
        {
            const auto s = source (x, y, w, h);
            const float u = uv[s * 3], v = uv[s * 3 + 1], coverage = uv[s * 3 + 2];
            const int fx = u > 0.5f ? 1 : 0, fy = v > 0.5f ? 1 : 0;
            const std::uint8_t* f = &frameTexels[(fy * 2 + fx) * 4];
            const float tint[3] { srgbToLinear (f[2] / 255.0f), srgbToLinear (f[1] / 255.0f), srgbToLinear (f[0] / 255.0f) };
            const float alpha = f[3] / 255.0f;
            for (int c = 0; c < 4; ++c)
            {
                float expected = base[c];
                if (coverage > 0.0f && c < 3)
                    expected += lit[c] * tint[c] * alpha + image.background[c] * coverage * (1.0f - alpha);
                std::uint16_t bits = 0;
                std::memcpy (&bits, &texels[(y * w + x) * 8 + c * 2], 2);
                worst = std::max (worst, static_cast<double> (std::fabs (halfValue (bits) - expected)));
            }
        }
    std::printf ("frame layer: largest difference from the OpenGL formula %.5f\n", worst);
    check (worst < 0.01, "the frame layer pass adds light * tint * alpha + background * coverage * (1 - alpha)");
}

void outputsAndResolve()
{
    const std::uint32_t w = 5, h = 3;
    const std::size_t n = w * h;
    std::vector<std::uint8_t> color (n * 4, 0);
    std::vector<float> depth (n);
    std::vector<std::uint16_t> normal (n * 4), emission (n * 4), motion (n * 2);
    std::vector<std::uint8_t> mask (n);
    std::vector<std::uint32_t> materialId (n), objectId (n);
    for (std::size_t i = 0; i < n; ++i)
    {
        color[i * 4 + 0] = static_cast<std::uint8_t> (10 * i);
        color[i * 4 + 3] = 255;
        depth[i] = 0.5f * static_cast<float> (i);
        const float nx = static_cast<float> (i) / n * 2.0f - 1.0f;
        normal[i * 4 + 0] = halfBits (nx);
        normal[i * 4 + 1] = halfBits (0.0f);
        normal[i * 4 + 2] = halfBits (1.0f);
        normal[i * 4 + 3] = halfBits (0.0f);
        for (int c = 0; c < 4; ++c) emission[i * 4 + c] = halfBits (0.0f);
        motion[i * 2] = halfBits (static_cast<float> (i));
        motion[i * 2 + 1] = halfBits (0.0f);
        mask[i] = i % 3 == 0 ? 0 : 255;
        materialId[i] = static_cast<std::uint32_t> (i);
        objectId[i] = static_cast<std::uint32_t> (100 + i);
    }
    arbitgpu::CpuSceneImage image;
    image.width = w;
    image.height = h;
    image.color = &color;
    image.depth = &depth;
    image.normal = &normal;
    image.emission = &emission;
    image.mask = &mask;
    image.materialId = &materialId;
    image.objectId = &objectId;
    image.motion = &motion;
    image.imageOutput = renderpassoutput::Output::Normal;
    image.resolveClear = { 0.0f, 0.0f, 0.0f, 0.0f };
    std::string error;
    const auto frame = arbitgpu::uploadCpuSceneFrame ("metal", image, error);
    check (frame != nullptr, "resolve upload: " + error);
    if (! frame) return;
    std::string why;
    check (compositorAccepts (*frame, false, true, why), "resolved frame refused by the compositor checks: " + why);
    const auto texels = readImage (frame->colorImageHandle(), 4);
    int worst = 0;
    for (std::uint32_t y = 0; y < h; ++y)
        for (std::uint32_t x = 0; x < w; ++x)
        {
            const auto s = source (x, y, w, h);
            const auto* t = &texels[(y * w + x) * 4];
            // BGRA8 memory: blue first. Normal (nx, 0, 1) shows as (nx / 2 + 0.5, 0.5, 1).
            const float nx = halfValue (normal[s * 4]) * 0.5f + 0.5f;
            const int r = mask[s] == 0 ? 0 : static_cast<int> (std::lround (nx * 255.0f));
            const int g = mask[s] == 0 ? 0 : 128;
            const int b = mask[s] == 0 ? 0 : 255;
            const int a = mask[s] == 0 ? 0 : 255;
            worst = std::max ({ worst, std::abs (t[2] - r), std::abs (t[1] - g), std::abs (t[0] - b), std::abs (t[3] - a) });
        }
    std::printf ("Normal view resolve: largest difference %d of 255\n", worst);
    check (worst <= 1, "the resolve shows the Normal output where the mask covers and the clear elsewhere");
    arbitgpu::NativeRawPassPixels raw;
    check (frame->readRawPass (renderpassoutput::Output::Color, raw, error)
               && raw.format == arbitgpu::NativeTexturePixelFormat::Bgra8Unorm && raw.bytes.size() == n * 4
               && raw.bytes[0] == color[source (0, 0, w, h) * 4 + 2] && raw.bytes[2] == color[source (0, 0, w, h) * 4],
           "raw Color reads the traced colour, top first, before the resolve");
    check (frame->readRawPass (renderpassoutput::Output::ObjectId, raw, error)
               && raw.format == arbitgpu::NativeTexturePixelFormat::R32Uint && raw.bytes.size() == n * 4
               && reinterpret_cast<const std::uint32_t*> (raw.bytes.data())[0] == objectId[source (0, 0, w, h)],
           "raw ObjectId reads top first");
}

void timing()
{
    const std::uint32_t w = 1920, h = 1080;
    const std::size_t n = static_cast<std::size_t> (w) * h;
    std::vector<std::uint8_t> color (n * 8, 0x3c);
    std::vector<float> depth (n, 3.0f);
    std::vector<std::uint16_t> normal (n * 4), emission (n * 4), motion (n * 2);
    std::vector<std::uint8_t> mask (n, 255);
    std::vector<std::uint32_t> materialId (n, 1), objectId (n, 2);
    arbitgpu::CpuSceneImage image;
    image.width = w;
    image.height = h;
    image.linearColor = true;
    image.color = &color;
    image.depth = &depth;
    std::string error;
    for (int outputs = 0; outputs < 2; ++outputs)
    {
        if (outputs)
        {
            image.normal = &normal;
            image.emission = &emission;
            image.mask = &mask;
            image.materialId = &materialId;
            image.objectId = &objectId;
            image.motion = &motion;
        }
        double best = 1e9;
        for (int run = 0; run < 5; ++run)
        {
            const auto start = std::chrono::steady_clock::now();
            const auto frame = arbitgpu::uploadCpuSceneFrame ("metal", image, error);
            const double ms = std::chrono::duration<double, std::milli> (std::chrono::steady_clock::now() - start).count();
            check (frame != nullptr, "1080p upload: " + error);
            best = std::min (best, ms);
        }
        std::printf ("1920x1080 RGBA16F upload%s: best of 5 %.2f ms\n", outputs ? " with the six outputs" : "", best);
    }
}

int ppm (const char* in, const char* out)
{
    std::ifstream file (in, std::ios::binary);
    std::string magic;
    unsigned w = 0, h = 0, top = 0;
    file >> magic >> w >> h >> top;
    file.get();
    if (magic != "P6" || top != 255 || w == 0 || h == 0)
    {
        std::printf ("FAIL: %s is not an 8-bit P6 PPM\n", in);
        return 1;
    }
    std::vector<std::uint8_t> rgb (static_cast<std::size_t> (w) * h * 3);
    file.read (reinterpret_cast<char*> (rgb.data()), static_cast<std::streamsize> (rgb.size()));
    // PPM rows are top first; the film is bottom first.
    std::vector<std::uint8_t> color (static_cast<std::size_t> (w) * h * 4);
    std::vector<float> depth (static_cast<std::size_t> (w) * h, 1.0f);
    for (unsigned y = 0; y < h; ++y)
        for (unsigned x = 0; x < w; ++x)
        {
            const auto* p = &rgb[(static_cast<std::size_t> (h - 1 - y) * w + x) * 3];
            auto* q = &color[(static_cast<std::size_t> (y) * w + x) * 4];
            q[0] = p[0];
            q[1] = p[1];
            q[2] = p[2];
            q[3] = 255;
        }
    arbitgpu::CpuSceneImage image;
    image.width = w;
    image.height = h;
    image.color = &color;
    image.depth = &depth;
    std::string error;
    const auto frame = arbitgpu::uploadCpuSceneFrame ("metal", image, error);
    std::string why;
    if (! frame || ! compositorAccepts (*frame, false, false, why))
    {
        std::printf ("FAIL: PPM upload: %s%s\n", error.c_str(), why.c_str());
        return 1;
    }
    const auto texels = readImage (frame->colorImageHandle(), 4);
    std::vector<std::uint8_t> back (rgb.size());
    for (std::size_t i = 0; i < static_cast<std::size_t> (w) * h; ++i)
    {
        back[i * 3 + 0] = texels[i * 4 + 2];
        back[i * 3 + 1] = texels[i * 4 + 1];
        back[i * 3 + 2] = texels[i * 4 + 0];
    }
    std::ofstream output (out, std::ios::binary);
    output << "P6\n" << w << " " << h << "\n255\n";
    output.write (reinterpret_cast<const char*> (back.data()), static_cast<std::streamsize> (back.size()));
    std::printf ("%s: the compositor texture %s the Path Trace image\n", out, back == rgb ? "matches" : "DIFFERS FROM");
    return back == rgb ? 0 : 1;
}
} // namespace

int main (int argc, char** argv)
{
    @autoreleasepool
    {
        {
            std::lock_guard<std::mutex> lock (arbitgpu::sokolmetal::mutex());
            if (! arbitgpu::sokolmetal::ensure())
            {
                std::printf ("SKIP: no sokol Metal device: %s\n", arbitgpu::sokolmetal::error().c_str());
                return 77;
            }
        }
        if (argc == 4 && std::string (argv[1]) == "--ppm")
            return ppm (argv[2], argv[3]);
        sdrNoLayers();
        linearFrameLayer();
        outputsAndResolve();
        timing();
    }
    std::printf (failures == 0 ? "pathtrace Metal upload tests passed\n" : "pathtrace Metal upload tests: %d failed\n",
                 failures);
    return failures == 0 ? 0 : 1;
}
