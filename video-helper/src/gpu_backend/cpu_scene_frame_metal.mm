#include "cpu_scene_frame.h"
#include "sokol_metal_context.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

#define SOKOL_METAL
#include "sokol_gfx.h"

// The Metal build's Path Trace upload. The film arrives as CPU rows, bottom
// first. Each input is written once, flipped to the top-first rows Metal frames
// use, into a CPU-visible MTLBuffer, and the frame's texture is a view of that
// buffer. On unified memory the compositor samples those same bytes: there is no
// second copy into a private texture. Frame layers and the pass resolve then run
// as sokol passes, the same work cpu_scene_frame_opengl.cpp does in OpenGL.

namespace arbitgpu
{
namespace
{
// Tagged so a CPU frame generation never equals a fixture preparation's.
std::uint64_t nextCpuFrameGeneration() noexcept
{
    static std::atomic<std::uint64_t> counter { 0 };
    return (std::uint64_t { 1 } << 62) | (counter.fetch_add (1, std::memory_order_relaxed) + 1);
}

// Raw pass order of the raster's frames (renderpasscomposite::kInputs[2..7]).
constexpr std::array<renderpassoutput::Output, 6> kPassOutputs {
    renderpassoutput::Output::Normal, renderpassoutput::Output::Emission, renderpassoutput::Output::Mask,
    renderpassoutput::Output::MaterialId, renderpassoutput::Output::ObjectId, renderpassoutput::Output::Motion };
constexpr std::array<NativeTexturePixelFormat, 6> kPassFormats {
    NativeTexturePixelFormat::Rgba16Float, NativeTexturePixelFormat::Rgba16Float, NativeTexturePixelFormat::R8Unorm,
    NativeTexturePixelFormat::R32Uint, NativeTexturePixelFormat::R32Uint, NativeTexturePixelFormat::Rg16Float };
constexpr std::array<sg_pixel_format, 6> kPassSokolFormats {
    SG_PIXELFORMAT_RGBA16F, SG_PIXELFORMAT_RGBA16F, SG_PIXELFORMAT_R8,
    SG_PIXELFORMAT_R32UI, SG_PIXELFORMAT_R32UI, SG_PIXELFORMAT_RG16F };
constexpr std::array<MTLPixelFormat, 6> kPassMetalFormats {
    MTLPixelFormatRGBA16Float, MTLPixelFormatRGBA16Float, MTLPixelFormatR8Unorm,
    MTLPixelFormatR32Uint, MTLPixelFormatR32Uint, MTLPixelFormatRG16Float };
constexpr std::array<std::size_t, 6> kPassPixelBytes { 8, 8, 1, 4, 4, 4 };

bool valid (sg_resource_state state) noexcept { return state == SG_RESOURCESTATE_VALID; }

void release (id object) noexcept
{
#if ! __has_feature(objc_arc)
    [object release];
#else
    (void) object;
#endif
}

// One sampled texture of the frame or of a pass input.
struct MetalImage final
{
    sg_image image {};
    sg_view texture {};
    sg_view attachment {};
    id<MTLBuffer> buffer = nil;  // the rows a buffer-backed texture views
    std::size_t rowBytes = 0;    // tight bytes per row
    std::size_t stride = 0;      // the buffer's bytes per row

    void destroy() noexcept
    {
        if (attachment.id != 0) sg_destroy_view (attachment);
        if (texture.id != 0) sg_destroy_view (texture);
        if (image.id != 0) sg_destroy_image (image);
        attachment = {};
        texture = {};
        image = {};
        release (buffer);
        buffer = nil;
    }

    // The tight top-first rows, from the buffer the texture views.
    std::vector<std::uint8_t> bytes (std::uint32_t height) const
    {
        std::vector<std::uint8_t> out (rowBytes * height);
        const auto* rows = static_cast<const std::uint8_t*> (buffer.contents);
        for (std::uint32_t row = 0; row < height; ++row)
            std::memcpy (out.data() + row * rowBytes, rows + row * stride, rowBytes);
        return out;
    }
};

// Writes bottom-first rows into a CPU-visible buffer as top-first rows and
// wraps the buffer in a texture. convert, when given, writes one row and may
// change the pixel layout (RGBA8 to BGRA8, RGB32F to RGBA32F).
template <typename Convert>
bool makeBufferImage (id<MTLDevice> device, MTLPixelFormat metalFormat, sg_pixel_format format,
                      std::size_t pixelBytes, std::uint32_t width, std::uint32_t height, const char* label,
                      Convert&& convert, MetalImage& out)
{
    out = {};
    out.rowBytes = pixelBytes * width;
    const std::size_t alignment = std::max<std::size_t> (1, [device minimumLinearTextureAlignmentForPixelFormat: metalFormat]);
    out.stride = (out.rowBytes + alignment - 1) / alignment * alignment;
    const MTLResourceOptions storage = device.hasUnifiedMemory ? MTLResourceStorageModeShared
                                                               : MTLResourceStorageModeManaged;
    out.buffer = [device newBufferWithLength: out.stride * height options: storage];
    if (out.buffer == nil) return false;
    auto* rows = static_cast<std::uint8_t*> (out.buffer.contents);
    for (std::uint32_t row = 0; row < height; ++row)
        convert (rows + static_cast<std::size_t> (row) * out.stride, height - 1 - row);
    if (! device.hasUnifiedMemory)
        [out.buffer didModifyRange: NSMakeRange (0, out.buffer.length)];
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat: metalFormat
                                                                                          width: width
                                                                                         height: height
                                                                                      mipmapped: NO];
    descriptor.storageMode = device.hasUnifiedMemory ? MTLStorageModeShared : MTLStorageModeManaged;
    descriptor.usage = MTLTextureUsageShaderRead;
    // sokol balances one retain when it stores an injected texture, which takes
    // over this +1.
    id<MTLTexture> texture = [out.buffer newTextureWithDescriptor: descriptor offset: 0 bytesPerRow: out.stride];
    if (texture == nil) return false;
    sg_image_desc desc = {};
    desc.usage.immutable = true;
    desc.width = static_cast<int> (width);
    desc.height = static_cast<int> (height);
    desc.pixel_format = format;
    desc.sample_count = 1;
    desc.mtl_textures[0] = (__bridge const void*) texture;
    desc.label = label;
    out.image = sg_make_image (&desc);
    sg_view_desc view = {};
    view.texture.image = out.image;
    out.texture = sg_make_view (&view);
    return valid (sg_query_image_state (out.image)) && valid (sg_query_view_state (out.texture));
}

// Copies one tight bottom-first row per call.
auto rowCopy (const void* bytes, std::size_t rowBytes)
{
    return [source = static_cast<const std::uint8_t*> (bytes), rowBytes] (std::uint8_t* target, std::uint32_t row)
    {
        std::memcpy (target, source + static_cast<std::size_t> (row) * rowBytes, rowBytes);
    };
}

bool makeTarget (sg_pixel_format format, std::uint32_t width, std::uint32_t height, const char* label,
                 MetalImage& out)
{
    out = {};
    sg_image_desc desc = {};
    desc.usage.color_attachment = true;
    desc.width = static_cast<int> (width);
    desc.height = static_cast<int> (height);
    desc.pixel_format = format;
    desc.sample_count = 1;
    desc.label = label;
    out.image = sg_make_image (&desc);
    sg_view_desc view = {};
    view.color_attachment.image = out.image;
    out.attachment = sg_make_view (&view);
    view = {};
    view.texture.image = out.image;
    out.texture = sg_make_view (&view);
    return valid (sg_query_image_state (out.image)) && valid (sg_query_view_state (out.attachment))
        && valid (sg_query_view_state (out.texture));
}

class MetalCpuSceneFrame final : public NativeFixtureSceneFrame
{
public:
    MetalCpuSceneFrame (std::uintptr_t device, std::array<MetalImage, 8> images, std::uint32_t width,
                        std::uint32_t height, bool linearColor, bool outputs) noexcept
        : device_ (device), images_ (images), width_ (width), height_ (height), linearColor_ (linearColor),
          outputs_ (outputs), generation_ (nextCpuFrameGeneration())
    {
    }

    ~MetalCpuSceneFrame() override
    {
        std::lock_guard<std::mutex> lock (sokolmetal::mutex());
        if (sg_isvalid())
            for (auto& image : images_)
                image.destroy();
        else
            for (auto& image : images_)
                release (image.buffer);
    }

    const std::string& backend() const noexcept override { return backend_; }
    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }
    std::uintptr_t colorImageHandle() const noexcept override { return images_[0].image.id; }
    std::uintptr_t colorTextureViewHandle() const noexcept override { return images_[0].texture.id; }
    std::uintptr_t depthImageHandle() const noexcept override { return images_[1].image.id; }
    std::uintptr_t depthTextureViewHandle() const noexcept override { return images_[1].texture.id; }
    NativeTextureViewDescriptor colorTextureDescriptor() const noexcept override
    {
        return { backend_, NativeTextureViewKind::Texture2D,
            linearColor_ ? NativeTexturePixelFormat::Rgba16Float : NativeTexturePixelFormat::Bgra8Unorm,
            images_[0].image.id, images_[0].texture.id, width_, height_, 1, true, device_, generation_,
            linearColor_ ? colortransform::ColorSpace::LinearSRGB : colortransform::ColorSpace::Unspecified,
            linearColor_ ? colortransform::TransferFunction::Linear : colortransform::TransferFunction::Unspecified };
    }
    NativeTextureViewDescriptor depthTextureDescriptor() const noexcept override
    {
        return { backend_, NativeTextureViewKind::Texture2D, NativeTexturePixelFormat::R32Float,
            images_[1].image.id, images_[1].texture.id, width_, height_, 1, true, device_, generation_ };
    }
    NativeTextureViewDescriptor passTextureDescriptor (renderpassoutput::Output output) const noexcept override
    {
        for (std::size_t i = 0; i < kPassOutputs.size(); ++i)
            if (kPassOutputs[i] == output && images_[i + 2].image.id != 0)
                return { backend_, NativeTextureViewKind::Texture2D, kPassFormats[i], images_[i + 2].image.id,
                    images_[i + 2].texture.id, width_, height_, 1, true, device_, generation_ };
        return NativeFixtureSceneFrame::passTextureDescriptor (output);
    }
    // Read from the buffers the textures view, so no GPU readback runs. Color is
    // the traced colour before frame layers or a resolve, as in OpenGL.
    bool readRawPass (renderpassoutput::Output output, NativeRawPassPixels& pixels, std::string& error) const override
    {
        pixels = {};
        const MetalImage* image = nullptr;
        NativeTexturePixelFormat format = NativeTexturePixelFormat::Invalid;
        if (outputs_ && output == renderpassoutput::Output::Color)
        {
            image = &traced_;
            format = linearColor_ ? NativeTexturePixelFormat::Rgba16Float : NativeTexturePixelFormat::Bgra8Unorm;
        }
        else if (outputs_ && output == renderpassoutput::Output::Depth)
        {
            image = &images_[1];
            format = NativeTexturePixelFormat::R32Float;
        }
        else if (outputs_)
            for (std::size_t i = 0; i < kPassOutputs.size(); ++i)
                if (kPassOutputs[i] == output)
                {
                    image = &images_[i + 2];
                    format = kPassFormats[i];
                }
        if (image == nullptr || image->buffer == nil)
        {
            error = "Path Trace frame has no " + std::string (renderpassoutput::token (output)) + " output";
            return false;
        }
        pixels.format = format;
        pixels.width = width_;
        pixels.height = height_;
        pixels.bytes = image->bytes (height_);
        return true;
    }

    // The traced colour's buffer, kept for the raw Color read when a pass replaced the Image.
    void keepTraced (id<MTLBuffer> buffer, std::size_t rowBytes, std::size_t stride) noexcept
    {
        traced_.buffer = buffer;
        traced_.rowBytes = rowBytes;
        traced_.stride = stride;
    }

private:
    std::uintptr_t device_ = 0;
    std::array<MetalImage, 8> images_ {};
    MetalImage traced_ {};
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    bool linearColor_ = false;
    bool outputs_ = false;
    std::uint64_t generation_ = 0;
    const std::string backend_ = "metal";
};

const char* const kVertexSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct VertexOut { float4 position [[position]]; };
vertex VertexOut fullscreen(uint vertexId [[vertex_id]]) {
    const float2 positions[3] = { float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f) };
    VertexOut out;
    out.position = float4(positions[vertexId], 0.0f, 1.0f);
    return out;
}
)metal";

// copyTraced writes the traced colour into the target. frameLayer is the OpenGL
// frame layer pass: the tint and coverage of the raster's frame-tinted materials
// added onto it. passResolve is the raster's resolve (backend_opengl.cpp,
// kDiffractionResolveFragmentShaderTail and kFixturePassCompositeShader).
const char* const kFragmentSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct VertexOut { float4 position [[position]]; };

fragment float4 copyTraced(VertexOut in [[stage_in]], texture2d<float> traced [[texture(0)]]) {
    return traced.read(uint2(in.position.xy));
}

struct LayerUniforms { float4 background; float4 control; };
float3 srgbToLinear(float3 c) {
    return select(pow((c + 0.055f) / 1.055f, float3(2.4f)), c / 12.92f, c < float3(0.04045f));
}
fragment float4 frameLayer(VertexOut in [[stage_in]], constant LayerUniforms& u [[buffer(0)]],
                           texture2d<float> light [[texture(0)]], texture2d<float> uvCoverage [[texture(1)]],
                           texture2d<float> frame [[texture(2)]], sampler frameSampler [[sampler(1)]]) {
    const uint2 pixel = uint2(in.position.xy);
    const float3 uvc = uvCoverage.read(pixel).xyz;
    if (uvc.z <= 0.0f) discard_fragment();
    const float2 uv = u.control.x > 0.5f ? float2(uvc.x, 1.0f - uvc.y) : uvc.xy;
    const float4 imagery = frame.sample(frameSampler, uv);
    const float3 tint = u.control.y > 0.5f ? imagery.rgb : srgbToLinear(imagery.rgb);
    const float alpha = clamp(imagery.a, 0.0f, 1.0f);
    return float4(light.read(pixel).rgb * tint * alpha + u.background.rgb * uvc.z * (1.0f - alpha), 0.0f);
}

struct ResolveUniforms {
    float4 passProgramControl;
    float4 passModes[8];
    float4 passColors[8];
    float4 passReferences[8];
    float4 passTransforms[8];
    float4 motionViewport;
    float4 imageOutput;
};
struct Raw {
    float4 color;
    float linearDepth;
    float4 normal;
    float4 emission;
    float mask;
    uint materialId;
    uint objectId;
    float2 motion;
};
float4 fixtureHalfRange(float4 color) {
    return select(clamp(color, float4(-65504.0f), float4(65504.0f)), color, isnan(color) || isinf(color));
}
float4 compositePass(Raw value, constant ResolveUniforms& u) {
    const int count = int(u.passProgramControl.x);
    if (count == 0 && u.passProgramControl.w > 0.5f) {
        const float3 rgb = clamp(value.color.rgb, 0.0f, 1.0f);
        return float4(select(1.055f * pow(rgb, float3(1.0f / 2.4f)) - 0.055f, rgb * 12.92f,
            rgb <= float3(0.0031308f)), clamp(value.color.a, 0.0f, 1.0f));
    }
    if (count == 0) return fixtureHalfRange(value.color);
    float4 values[8];
    for (int i = 0; i < 8; ++i) {
        if (i >= count) break;
        const float4 p = u.passModes[i];
        const int mode = int(p.x);
        const float depth = clamp((value.linearDepth - p.y) / (p.z - p.y), 0.0f, 1.0f);
        float3 fog = u.passColors[i].rgb;
        if (u.passTransforms[i].y > 0.5f)
            fog = select(pow((fog + 0.055f) / 1.055f, float3(2.4f)), fog / 12.92f, fog <= float3(0.04045f));
        const uint identity = uint(u.passReferences[i].x) | (uint(u.passReferences[i].y) << 16u);
        float4 result = float4(value.color.rgb, value.mask);
        if (mode == 1) result = float4(mix(value.color.rgb, fog, depth * p.w), value.mask);
        const float inspectionAlpha = u.passTransforms[i].w > 0.5f ? 1.0f : value.mask;
        if (mode == 2) result = float4(float3(u.passTransforms[i].w > 1.5f ? 1.0f - depth : depth), inspectionAlpha);
        if (mode == 3) result = float4(value.normal.xyz * 0.5f + 0.5f, inspectionAlpha);
        if (mode == 4) result = float4(value.emission.rgb, value.mask);
        if (mode >= 5 && mode <= 7) {
            float matte = value.mask;
            if (mode == 6) matte *= value.materialId == identity ? 1.0f : 0.0f;
            if (mode == 7) matte *= value.objectId == identity ? 1.0f : 0.0f;
            result = float4(float3(matte), 1.0f);
        }
        if (mode == 8) result = float4(clamp(float2(0.5f) + value.motion / u.motionViewport.xy
            * (1.0f + 63.0f * p.w), 0.0f, 1.0f), 0.5f, value.mask);
        if (mode == 9 || mode == 10) {
            const float4 a = values[int(u.passReferences[i].z) - 1];
            const float4 b = values[int(u.passReferences[i].w) - 1];
            result = mode == 9 ? mix(a, b, p.w) : float4(a.rgb + b.rgb * p.w, max(a.a, b.a));
        }
        result.rgb *= exp2(u.passTransforms[i].x);
        values[i] = result;
    }
    const int selected = int(u.passProgramControl.y);
    float4 result = values[selected];
    const int transform = int(u.passTransforms[selected].z);
    if (transform > 0 && u.passProgramControl.z < 0.5f) {
        float3 rgb = max(result.rgb, float3(0.0f));
        if (transform == 2) rgb /= float3(1.0f) + rgb;
        result.rgb = select(1.055f * pow(rgb, float3(1.0f / 2.4f)) - 0.055f, rgb * 12.92f, rgb <= float3(0.0031308f));
    }
    return u.passProgramControl.z > 0.5f ? fixtureHalfRange(result) : clamp(result, 0.0f, 1.0f);
}
fragment float4 passResolve(VertexOut in [[stage_in]], constant ResolveUniforms& u [[buffer(0)]],
                            texture2d<float> color [[texture(0)]], texture2d<float> linearDepth [[texture(1)]],
                            texture2d<float> normal [[texture(2)]], texture2d<float> emission [[texture(3)]],
                            texture2d<float> mask [[texture(4)]], texture2d<uint> materialId [[texture(5)]],
                            texture2d<uint> objectId [[texture(6)]], texture2d<float> motion [[texture(7)]]) {
    const uint2 pixel = uint2(in.position.xy);
    Raw value;
    value.mask = mask.read(pixel).r;
    if (value.mask <= 0.0f) discard_fragment();
    value.linearDepth = linearDepth.read(pixel).r;
    value.normal = normal.read(pixel);
    value.emission = emission.read(pixel);
    value.materialId = materialId.read(pixel).r;
    value.objectId = objectId.read(pixel).r;
    value.motion = motion.read(pixel).rg;
    value.color = color.read(pixel);
    if (any(isnan(value.color)) || any(isinf(value.color)))
        return float4(float3(as_type<float>(0x7fc00000u | (11u << 10u))), 1.0f);
    const int output = int(u.imageOutput.x);
    float4 result;
    if (output != 0) {
        float3 image = float3(0.0f);
        if (output == 1) image = float3(value.linearDepth);
        if (output == 2) image = value.normal.xyz * 0.5f + 0.5f;
        if (output == 3) image = float3(clamp(float2(0.5f) + value.motion / u.motionViewport.xy, 0.0f, 1.0f), 0.5f);
        if (output == 4) image = value.emission.rgb;
        if (output == 5) image = float3(value.mask);
        if (output == 6 || output == 7) {
            uint hash = (output == 6 ? value.materialId : value.objectId) * 0x9e3779b9u;
            hash ^= hash >> 16u;
            image = float3(32u + (hash & 191u), 32u + ((hash >> 8u) & 191u), 32u + ((hash >> 16u) & 191u)) / 255.0f;
        }
        result = float4(image, value.color.a);
    } else {
        result = compositePass(value, u);
    }
    if (any(isnan(result)) || any(isinf(result)))
        return float4(float3(as_type<float>(0x7fc00000u | (10u << 10u))), 1.0f);
    return result;
}
)metal";

struct LayerUniforms final
{
    float background[4];
    float control[4];  // x: the frame is bottom first, y: the frame is linear
};

struct ResolveUniforms final
{
    renderpasscomposite::GpuProgram program;
    float motionViewport[4];
    float imageOutput[4];
};
static_assert (sizeof (ResolveUniforms) == 560);

// Pipelines for one target format: index 0 is BGRA8, 1 is RGBA16F.
struct Programs final
{
    sg_shader copy {}, layer {}, resolve {};
    std::array<sg_pipeline, 2> copyPipeline {}, layerPipeline {}, resolvePipeline {};
    sg_sampler point {}, linear {};

    bool live() const noexcept
    {
        bool ok = valid (sg_query_shader_state (copy)) && valid (sg_query_shader_state (layer))
            && valid (sg_query_shader_state (resolve)) && valid (sg_query_sampler_state (point))
            && valid (sg_query_sampler_state (linear));
        for (std::size_t i = 0; i < 2; ++i)
            ok = ok && valid (sg_query_pipeline_state (copyPipeline[i]))
                && valid (sg_query_pipeline_state (layerPipeline[i]))
                && valid (sg_query_pipeline_state (resolvePipeline[i]));
        return ok;
    }
};

// sokol wants every texture in a texture-sampler pair. The shaders read texels,
// so a texture without a filtered sample pairs with the point sampler, slot 0.
void setTexture (sg_shader_desc& desc, int slot, sg_image_sample_type type, int sampler = 0)
{
    desc.views[slot].texture.stage = SG_SHADERSTAGE_FRAGMENT;
    desc.views[slot].texture.image_type = SG_IMAGETYPE_2D;
    desc.views[slot].texture.sample_type = type;
    desc.views[slot].texture.msl_texture_n = static_cast<std::uint8_t> (slot);
    desc.texture_sampler_pairs[slot] = { SG_SHADERSTAGE_FRAGMENT, static_cast<std::uint8_t> (slot),
                                         static_cast<std::uint8_t> (sampler), nullptr };
}

void setSampler (sg_shader_desc& desc, int slot, sg_sampler_type type)
{
    desc.samplers[slot].stage = SG_SHADERSTAGE_FRAGMENT;
    desc.samplers[slot].sampler_type = type;
    desc.samplers[slot].msl_sampler_n = static_cast<std::uint8_t> (slot);
}

bool buildPrograms (Programs& out, std::string& error)
{
    out = {};
    sg_shader_desc copy = {};
    copy.vertex_func.source = kVertexSource;
    copy.vertex_func.entry = "fullscreen";
    copy.fragment_func.source = kFragmentSource;
    copy.fragment_func.entry = "copyTraced";
    setTexture (copy, 0, SG_IMAGESAMPLETYPE_FLOAT);
    setSampler (copy, 0, SG_SAMPLERTYPE_NONFILTERING);
    copy.label = "arbit-pathtrace-copy";
    out.copy = sg_make_shader (&copy);

    sg_shader_desc layer = {};
    layer.vertex_func = copy.vertex_func;
    layer.fragment_func.source = kFragmentSource;
    layer.fragment_func.entry = "frameLayer";
    layer.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
    layer.uniform_blocks[0].size = sizeof (LayerUniforms);
    layer.uniform_blocks[0].msl_buffer_n = 0;
    setTexture (layer, 0, SG_IMAGESAMPLETYPE_FLOAT);
    setTexture (layer, 1, SG_IMAGESAMPLETYPE_UNFILTERABLE_FLOAT);
    setTexture (layer, 2, SG_IMAGESAMPLETYPE_FLOAT, 1);
    setSampler (layer, 0, SG_SAMPLERTYPE_NONFILTERING);
    setSampler (layer, 1, SG_SAMPLERTYPE_FILTERING);
    layer.label = "arbit-pathtrace-frame-layer";
    out.layer = sg_make_shader (&layer);

    sg_shader_desc resolve = {};
    resolve.vertex_func = copy.vertex_func;
    resolve.fragment_func.source = kFragmentSource;
    resolve.fragment_func.entry = "passResolve";
    resolve.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
    resolve.uniform_blocks[0].size = sizeof (ResolveUniforms);
    resolve.uniform_blocks[0].msl_buffer_n = 0;
    // Depth is R32F; the rest are filterable floats but the two identities.
    for (int slot = 0; slot < 8; ++slot)
        setTexture (resolve, slot, slot == 5 || slot == 6 ? SG_IMAGESAMPLETYPE_UINT
                                   : slot == 1 ? SG_IMAGESAMPLETYPE_UNFILTERABLE_FLOAT : SG_IMAGESAMPLETYPE_FLOAT);
    setSampler (resolve, 0, SG_SAMPLERTYPE_NONFILTERING);
    resolve.label = "arbit-pathtrace-pass-resolve";
    out.resolve = sg_make_shader (&resolve);

    sg_sampler_desc sampler = {};
    sampler.min_filter = SG_FILTER_NEAREST;
    sampler.mag_filter = SG_FILTER_NEAREST;
    sampler.wrap_u = SG_WRAP_CLAMP_TO_EDGE;
    sampler.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
    sampler.label = "arbit-pathtrace-point-sampler";
    out.point = sg_make_sampler (&sampler);
    sampler.min_filter = SG_FILTER_LINEAR;
    sampler.mag_filter = SG_FILTER_LINEAR;
    sampler.label = "arbit-pathtrace-frame-sampler";
    out.linear = sg_make_sampler (&sampler);

    const sg_pixel_format formats[2] { SG_PIXELFORMAT_BGRA8, SG_PIXELFORMAT_RGBA16F };
    for (std::size_t i = 0; i < 2; ++i)
    {
        sg_pipeline_desc pipeline = {};
        pipeline.colors[0].pixel_format = formats[i];
        pipeline.depth.pixel_format = SG_PIXELFORMAT_NONE;
        pipeline.primitive_type = SG_PRIMITIVETYPE_TRIANGLES;
        pipeline.sample_count = 1;
        pipeline.shader = out.copy;
        pipeline.label = "arbit-pathtrace-copy-pipeline";
        out.copyPipeline[i] = sg_make_pipeline (&pipeline);
        pipeline.shader = out.resolve;
        pipeline.label = "arbit-pathtrace-pass-resolve-pipeline";
        out.resolvePipeline[i] = sg_make_pipeline (&pipeline);
        // The layer adds light; the target's alpha stays.
        pipeline.shader = out.layer;
        pipeline.colors[0].blend.enabled = true;
        pipeline.colors[0].blend.src_factor_rgb = SG_BLENDFACTOR_ONE;
        pipeline.colors[0].blend.dst_factor_rgb = SG_BLENDFACTOR_ONE;
        pipeline.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_ZERO;
        pipeline.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE;
        pipeline.label = "arbit-pathtrace-frame-layer-pipeline";
        out.layerPipeline[i] = sg_make_pipeline (&pipeline);
    }
    if (out.live()) return true;
    error = "Path Trace Metal frame programs could not be built: " + sokolmetal::log();
    return false;
}

bool resolveRequested (const CpuSceneImage& image) noexcept
{
    return image.imageOutput != renderpassoutput::Output::Color || image.passProgram.count > 0;
}

// The frame layer's source must be a live texture on this Metal device.
bool liveMetalFrame (const NativeFixtureSceneFrame& frame, std::uintptr_t device) noexcept
{
    const auto descriptor = frame.colorTextureDescriptor();
    return frame.backend() == "metal" && descriptor.complete() && descriptor.backend == "metal"
        && descriptor.deviceOrContextIdentity == device
        && descriptor.textureViewHandle == frame.colorTextureViewHandle()
        && descriptor.textureViewHandle <= std::numeric_limits<std::uint32_t>::max()
        && valid (sg_query_view_state (sg_view { static_cast<std::uint32_t> (descriptor.textureViewHandle) }));
}
} // namespace

std::shared_ptr<const NativeFixtureSceneFrame> uploadCpuSceneFrame (
    const std::string& backend, const CpuSceneImage& image, std::string& error)
{
    if (backend != "metal")
    {
        error = "Path Trace upload requires the Metal fixture backend";
        return nullptr;
    }
    const std::size_t pixels = static_cast<std::size_t> (image.width) * image.height;
    if (image.width == 0 || image.height == 0
        || image.width > static_cast<std::uint32_t> (std::numeric_limits<int>::max())
        || image.height > static_cast<std::uint32_t> (std::numeric_limits<int>::max())
        || image.color == nullptr || image.depth == nullptr
        || image.color->size() != pixels * (image.linearColor ? 8u : 4u) || image.depth->size() != pixels)
    {
        error = "Path Trace image does not match its declared size and format";
        return nullptr;
    }
    for (const auto& layer : image.frameLayers)
        if (layer.frame == nullptr || layer.light == nullptr || layer.uvCoverage == nullptr
            || layer.light->size() != pixels * 8u || layer.uvCoverage->size() != pixels * 3u)
        {
            error = "Path Trace frame layer does not match its image";
            return nullptr;
        }
    const bool outputs = image.normal != nullptr;
    if (outputs
        && (image.emission == nullptr || image.mask == nullptr || image.materialId == nullptr
            || image.objectId == nullptr || image.motion == nullptr
            || image.normal->size() != pixels * 4u || image.emission->size() != pixels * 4u
            || image.mask->size() != pixels || image.materialId->size() != pixels
            || image.objectId->size() != pixels || image.motion->size() != pixels * 2u))
    {
        error = "Path Trace render outputs do not match their image";
        return nullptr;
    }
    const bool resolve = resolveRequested (image);
    if (! outputs && resolve)
    {
        error = "Path Trace Image view or pass composite has no render outputs to read";
        return nullptr;
    }

    std::lock_guard<std::mutex> lock (sokolmetal::mutex());
    @autoreleasepool
    {
        if (! sokolmetal::ensure() || sg_query_backend() != SG_BACKEND_METAL_MACOS)
        {
            error = "Path Trace upload has no Metal backend: " + sokolmetal::error();
            return nullptr;
        }
        id<MTLDevice> device = (__bridge id<MTLDevice>) sokolmetal::device();
        const auto deviceIdentity = reinterpret_cast<std::uintptr_t> (sokolmetal::device());
        for (const auto& layer : image.frameLayers)
            if (! liveMetalFrame (*layer.frame, deviceIdentity))
            {
                error = "Path Trace frame layer's frame is not a live texture on this Metal device";
                return nullptr;
            }
        static Programs programs;
        const bool passes = ! image.frameLayers.empty() || resolve;
        if (passes && ! programs.live() && ! buildPrograms (programs, error))
            return nullptr;

        // Color, Depth, the six outputs. Temporary inputs die after the passes are committed.
        std::array<MetalImage, 8> images {};
        std::vector<MetalImage> temporary;
        const auto fail = [&] (const char* message)
        {
            for (auto& entry : images) entry.destroy();
            for (auto& entry : temporary) entry.destroy();
            error = message;
            return nullptr;
        };
        const std::uint32_t w = image.width, h = image.height;
        const sg_pixel_format colorFormat = image.linearColor ? SG_PIXELFORMAT_RGBA16F : SG_PIXELFORMAT_BGRA8;
        bool made = false;
        if (image.linearColor)
            made = makeBufferImage (device, MTLPixelFormatRGBA16Float, colorFormat, 8, w, h, "arbit-pathtrace-color",
                                    rowCopy (image.color->data(), w * 8u), images[0]);
        else
        {
            // The compositor takes SDR Metal colour as BGRA8.
            const auto* source = image.color->data();
            made = makeBufferImage (device, MTLPixelFormatBGRA8Unorm, colorFormat, 4, w, h, "arbit-pathtrace-color",
                [source, w] (std::uint8_t* target, std::uint32_t row)
                {
                    const auto* from = source + static_cast<std::size_t> (row) * w * 4u;
                    for (std::uint32_t x = 0; x < w; ++x, from += 4, target += 4)
                    {
                        target[0] = from[2];
                        target[1] = from[1];
                        target[2] = from[0];
                        target[3] = from[3];
                    }
                }, images[0]);
        }
        made = made && makeBufferImage (device, MTLPixelFormatR32Float, SG_PIXELFORMAT_R32F, 4, w, h,
                                        "arbit-pathtrace-depth", rowCopy (image.depth->data(), w * 4u), images[1]);
        if (made && outputs)
        {
            const void* sources[6] { image.normal->data(), image.emission->data(), image.mask->data(),
                                     image.materialId->data(), image.objectId->data(), image.motion->data() };
            for (std::size_t i = 0; made && i < 6; ++i)
                made = makeBufferImage (device, kPassMetalFormats[i], kPassSokolFormats[i], kPassPixelBytes[i], w, h,
                                        "arbit-pathtrace-pass", rowCopy (sources[i], w * kPassPixelBytes[i]),
                                        images[i + 2]);
        }
        if (! made)
            return fail ("Path Trace upload could not allocate its Metal textures");

        const std::size_t formatIndex = image.linearColor ? 1u : 0u;
        MetalImage traced = images[0];
        bool replaced = false;
        if (! image.frameLayers.empty())
        {
            MetalImage target;
            const bool targetMade = makeTarget (colorFormat, w, h, "arbit-pathtrace-layered", target);
            temporary.push_back (target);
            if (! targetMade)
                return fail ("Path Trace frame layer target could not be allocated");
            // The light and uv of every layer, as textures for the pass.
            std::vector<std::pair<MetalImage, MetalImage>> inputs;
            for (const auto& layer : image.frameLayers)
            {
                MetalImage light, uv;
                const auto* coverage = layer.uvCoverage->data();
                const bool inputMade = makeBufferImage (device, MTLPixelFormatRGBA16Float, SG_PIXELFORMAT_RGBA16F, 8, w, h,
                                                        "arbit-pathtrace-layer-light",
                                                        rowCopy (layer.light->data(), w * 8u), light)
                    && makeBufferImage (device, MTLPixelFormatRGBA32Float, SG_PIXELFORMAT_RGBA32F, 16, w, h,
                        "arbit-pathtrace-layer-uv",
                        [coverage, w] (std::uint8_t* target, std::uint32_t row)
                        {
                            const float* from = coverage + static_cast<std::size_t> (row) * w * 3u;
                            auto* to = reinterpret_cast<float*> (target);
                            for (std::uint32_t x = 0; x < w; ++x, from += 3, to += 4)
                            {
                                to[0] = from[0];
                                to[1] = from[1];
                                to[2] = from[2];
                                to[3] = 0.0f;
                            }
                        }, uv);
                temporary.push_back (light);
                temporary.push_back (uv);
                if (! inputMade)
                    return fail ("Path Trace frame layer inputs could not be allocated");
                inputs.emplace_back (light, uv);
            }
            sg_pass pass = {};
            pass.attachments.colors[0] = target.attachment;
            pass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
            pass.action.colors[0].store_action = SG_STOREACTION_STORE;
            pass.label = "arbit-pathtrace-frame-layers";
            sg_begin_pass (&pass);
            sg_apply_pipeline (programs.copyPipeline[formatIndex]);
            sg_bindings bindings = {};
            bindings.views[0] = images[0].texture;
            bindings.samplers[0] = programs.point;
            sg_apply_bindings (&bindings);
            sg_draw (0, 3, 1);
            sg_apply_pipeline (programs.layerPipeline[formatIndex]);
            for (std::size_t index = 0; index < image.frameLayers.size(); ++index)
            {
                const auto& frame = *image.frameLayers[index].frame;
                const auto descriptor = frame.colorTextureDescriptor();
                LayerUniforms uniforms {};
                std::copy (image.background, image.background + 3, uniforms.background);
                uniforms.control[0] = descriptor.rowOrder == NativeTextureRowOrder::BottomFirst ? 1.0f : 0.0f;
                uniforms.control[1] = materialFrameIsLinear (descriptor) ? 1.0f : 0.0f;
                bindings = {};
                bindings.views[0] = inputs[index].first.texture;
                bindings.views[1] = inputs[index].second.texture;
                bindings.views[2] = sg_view { static_cast<std::uint32_t> (descriptor.textureViewHandle) };
                bindings.samplers[0] = programs.point;
                bindings.samplers[1] = programs.linear;
                sg_apply_bindings (&bindings);
                const sg_range range { &uniforms, sizeof (uniforms) };
                sg_apply_uniforms (0, &range);
                sg_draw (0, 3, 1);
            }
            sg_end_pass();
            std::swap (images[0], temporary.front());
            replaced = true;
        }
        // The Image view or composite replaces the colour on the Image; raw
        // Color still reads the traced one.
        if (resolve)
        {
            MetalImage target;
            const bool targetMade = makeTarget (colorFormat, w, h, "arbit-pathtrace-resolved", target);
            temporary.push_back (target);
            if (! targetMade)
                return fail ("Path Trace pass resolve target could not be allocated");
            ResolveUniforms uniforms {};
            uniforms.program = renderpasscomposite::gpuProgram (image.passProgram);
            uniforms.program.control[2] = image.linearColor ? 1.0f : 0.0f;
            uniforms.program.control[3] = 0.0f;
            uniforms.motionViewport[0] = static_cast<float> (w);
            uniforms.motionViewport[1] = static_cast<float> (h);
            uniforms.imageOutput[0] = static_cast<float> (static_cast<int> (image.imageOutput));
            sg_pass pass = {};
            pass.attachments.colors[0] = target.attachment;
            pass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
            pass.action.colors[0].clear_value = { image.resolveClear[0], image.resolveClear[1],
                                                  image.resolveClear[2], image.resolveClear[3] };
            pass.action.colors[0].store_action = SG_STOREACTION_STORE;
            pass.label = "arbit-pathtrace-pass-resolve";
            sg_begin_pass (&pass);
            sg_apply_pipeline (programs.resolvePipeline[formatIndex]);
            sg_bindings bindings = {};
            for (std::size_t slot = 0; slot < 8; ++slot)
                bindings.views[slot] = images[slot].texture;
            bindings.samplers[0] = programs.point;
            sg_apply_bindings (&bindings);
            const sg_range range { &uniforms, sizeof (uniforms) };
            sg_apply_uniforms (0, &range);
            sg_draw (0, 3, 1);
            sg_end_pass();
            std::swap (images[0], temporary.back());
            replaced = true;
        }
        if (passes)
            sg_commit();
        // The traced colour's buffer outlives its texture for the raw Color read.
        id<MTLBuffer> tracedBuffer = nil;
        if (replaced && outputs)
        {
            tracedBuffer = traced.buffer;
#if ! __has_feature(objc_arc)
            [tracedBuffer retain];
#endif
        }
        // sokol defers each release until the frames in flight that use it are done.
        for (auto& entry : temporary) entry.destroy();
        try
        {
            auto frame = std::make_shared<MetalCpuSceneFrame> (deviceIdentity, images, w, h, image.linearColor, outputs);
            if (tracedBuffer != nil)
                frame->keepTraced (tracedBuffer, traced.rowBytes, traced.stride);
            return frame;
        }
        catch (const std::bad_alloc&)
        {
            release (tracedBuffer);
            for (auto& entry : images) entry.destroy();
            error = "Path Trace frame allocation failed";
            return nullptr;
        }
    }
}

} // namespace arbitgpu
