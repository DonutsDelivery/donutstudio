#include "volume_native_renderer.h"
#include "gpu_backend/sokol_metal_context.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#define SOKOL_METAL
#include "sokol_gfx.h"

namespace videohelper::volume
{
namespace
{
constexpr std::size_t kMaximumVolumeBytes = 512u * 1024u * 1024u;
constexpr std::uint32_t kMaximumRenderExtent = 4096;
constexpr std::uint32_t kMaximumTexture3DExtent = 2048;
constexpr std::uint32_t kMaximumRaySteps = 256;
constexpr const char* kUnavailable = "native Metal volume execution is unavailable";

constexpr const char* kVertexShader = R"metal(
#include <metal_stdlib>
using namespace metal;
struct VertexOut { float4 position [[position]]; };
vertex VertexOut _main(uint vertexId [[vertex_id]])
{
    const float2 positions[3] = {
        float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)
    };
    VertexOut out;
    out.position = float4(positions[vertexId], 0.0f, 1.0f);
    return out;
}
)metal";

constexpr const char* kFragmentShader = R"metal(
#include <metal_stdlib>
using namespace metal;
struct VertexOut { float4 position [[position]]; };
struct VolumeUniforms {
    float4x4 worldToLocal;
    float4 boundsMinimum;
    float4 boundsMaximum;
    float4 cameraOrigin;
    float4 cameraTarget;
    float4 extentCameraSteps;
};

bool intersectBox(float3 origin, float3 direction,
                  float3 boundsMinimum, float3 boundsMaximum,
                  thread float& nearT, thread float& farT)
{
    const float3 inverseDirection = 1.0f / direction;
    const float3 first = (boundsMinimum - origin) * inverseDirection;
    const float3 second = (boundsMaximum - origin) * inverseDirection;
    const float3 nearAxis = min(first, second);
    const float3 farAxis = max(first, second);
    nearT = max(max(nearAxis.x, nearAxis.y), nearAxis.z);
    farT = min(min(farAxis.x, farAxis.y), farAxis.z);
    return farT >= max(nearT, 0.0f);
}

fragment float4 _main(VertexOut in [[stage_in]],
                      constant VolumeUniforms& u [[buffer(0)]],
                      texture3d<float> volume [[texture(0)]],
                      sampler volumeSampler [[sampler(0)]])
{
    // Match OpenGL's bottom-left gl_FragCoord convention. Metal fragment
    // positions are top-left, so flip only Y before applying the shared camera.
    const float2 fragmentPosition = float2(
        in.position.x, u.extentCameraSteps.y - in.position.y);
    const float2 extent = u.extentCameraSteps.xy;
    const float2 uv = (2.0f * fragmentPosition - extent) / extent.y;
    const float3 cameraOrigin = u.cameraOrigin.xyz;
    const float3 forward = u.cameraTarget.xyz - cameraOrigin;
    const float3 worldDirection = normalize(forward
        + float3(uv.x * u.extentCameraSteps.z,
                 uv.y * u.extentCameraSteps.z, 0.0f));
    const float3 localOrigin = (u.worldToLocal * float4(cameraOrigin, 1.0f)).xyz;
    const float3 localDirection = (u.worldToLocal * float4(worldDirection, 0.0f)).xyz;
    const float3 background = float3(7.0f / 255.0f, 10.0f / 255.0f, 18.0f / 255.0f);

    float nearT = 0.0f;
    float farT = 0.0f;
    if (!intersectBox(localOrigin, localDirection, u.boundsMinimum.xyz,
                      u.boundsMaximum.xyz, nearT, farT))
        return float4(background, 1.0f);

    nearT = max(nearT, 0.0f);
    const int stepCount = max(int(u.extentCameraSteps.w + 0.5f), 1);
    const float stepLength = max((farT - nearT) / float(stepCount), 0.000001f);
    const float3 localExtent = u.boundsMaximum.xyz - u.boundsMinimum.xyz;
    float3 accumulated = float3(0.0f);
    float transmittance = 1.0f;
    for (int step = 0; step < 256; ++step)
    {
        if (step >= stepCount || transmittance < 0.004f) break;
        const float t = nearT + (float(step) + 0.5f) * stepLength;
        const float3 localPoint = localOrigin + localDirection * t;
        const float3 texturePoint = (localPoint - u.boundsMinimum.xyz) / localExtent;
        const float density = max(volume.sample(volumeSampler,
            clamp(texturePoint, 0.0f, 1.0f)).r, 0.0f);
        const float alpha = 1.0f - exp(-density * 8.0f * stepLength);
        const float3 sampleColor = mix(float3(0.08f, 0.24f, 0.62f),
                                       float3(1.0f, 0.46f, 0.12f),
                                       clamp(density, 0.0f, 1.0f));
        accumulated += transmittance * alpha * sampleColor;
        transmittance *= 1.0f - alpha;
    }
    return float4(accumulated + transmittance * background, 1.0f);
}
)metal";

struct VolumeUniforms final
{
    float worldToLocal[16] {};
    float boundsMinimum[4] {};
    float boundsMaximum[4] {};
    float cameraOrigin[4] {};
    float cameraTarget[4] {};
    float extentCameraSteps[4] {};
};
static_assert (sizeof (VolumeUniforms) == 144,
               "Metal volume uniform layout changed");

bool resourceValid (sg_resource_state state) noexcept
{
    return state == SG_RESOURCESTATE_VALID;
}

bool textureFormat (videowire::VolumeVoxelFormat format,
                    sg_pixel_format& result) noexcept
{
    switch (format)
    {
        case videowire::VolumeVoxelFormat::densityU8:
            result = SG_PIXELFORMAT_R8;
            return true;
        case videowire::VolumeVoxelFormat::densityF16:
            result = SG_PIXELFORMAT_R16F;
            return true;
        case videowire::VolumeVoxelFormat::densityF32:
            result = SG_PIXELFORMAT_R32F;
            return true;
    }
    result = SG_PIXELFORMAT_NONE;
    return false;
}

bool expectedByteCount (const AdmittedVolume& volume,
                        std::size_t& expected) noexcept
{
    std::size_t bytesPerVoxel = 0;
    switch (volume.format())
    {
        case videowire::VolumeVoxelFormat::densityU8: bytesPerVoxel = 1; break;
        case videowire::VolumeVoxelFormat::densityF16: bytesPerVoxel = 2; break;
        case videowire::VolumeVoxelFormat::densityF32: bytesPerVoxel = 4; break;
    }
    const auto& dimensions = volume.dimensions();
    const auto width = static_cast<std::size_t> (dimensions.width);
    const auto height = static_cast<std::size_t> (dimensions.height);
    const auto depth = static_cast<std::size_t> (dimensions.depth);
    if (bytesPerVoxel == 0 || (width != 0 && height > SIZE_MAX / width)) return false;
    const auto plane = width * height;
    if (plane != 0 && depth > SIZE_MAX / plane) return false;
    const auto voxels = plane * depth;
    if (voxels != 0 && bytesPerVoxel > SIZE_MAX / voxels) return false;
    if (voxels != volume.voxelCount()) return false;
    expected = voxels * bytesPerVoxel;
    return true;
}

bool invertAffine (const std::array<float, 16>& source,
                   std::array<float, 16>& inverse) noexcept
{
    constexpr double tolerance = 1.0e-7;
    if (std::abs (source[3]) > tolerance || std::abs (source[7]) > tolerance
        || std::abs (source[11]) > tolerance || std::abs (source[15] - 1.0f) > tolerance)
        return false;

    double augmented[4][8] {};
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
        {
            augmented[row][column] = source[static_cast<std::size_t> (column * 4 + row)];
            augmented[row][column + 4] = row == column ? 1.0 : 0.0;
        }

    for (int column = 0; column < 4; ++column)
    {
        int pivot = column;
        for (int row = column + 1; row < 4; ++row)
            if (std::abs (augmented[row][column]) > std::abs (augmented[pivot][column]))
                pivot = row;
        if (! std::isfinite (augmented[pivot][column])
            || std::abs (augmented[pivot][column]) <= tolerance)
            return false;
        if (pivot != column)
            for (int value = 0; value < 8; ++value)
                std::swap (augmented[pivot][value], augmented[column][value]);

        const auto divisor = augmented[column][column];
        for (int value = 0; value < 8; ++value) augmented[column][value] /= divisor;
        for (int row = 0; row < 4; ++row)
        {
            if (row == column) continue;
            const auto multiplier = augmented[row][column];
            for (int value = 0; value < 8; ++value)
                augmented[row][value] -= multiplier * augmented[column][value];
        }
    }

    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
        {
            const auto value = augmented[row][column + 4];
            if (! std::isfinite (value)
                || value > std::numeric_limits<float>::max()
                || value < -std::numeric_limits<float>::max())
                return false;
            inverse[static_cast<std::size_t> (column * 4 + row)]
                = static_cast<float> (value);
        }
    return true;
}

std::array<float, 3> transformPoint (const std::array<float, 16>& matrix,
                                     float x, float y, float z) noexcept
{
    return {
        matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
        matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
        matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14]
    };
}

bool cameraForVolume (const AdmittedVolume& volume,
                      std::array<float, 3>& origin,
                      std::array<float, 3>& target,
                      float& halfHeight) noexcept
{
    const auto& bounds = volume.bounds();
    const auto& matrix = volume.transform().localToWorld;
    std::array<float, 3> minimum {
        std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    };
    std::array<float, 3> maximum {
        -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max()
    };
    for (int z = 0; z < 2; ++z)
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x)
            {
                const auto point = transformPoint (
                    matrix,
                    x == 0 ? bounds.minimum.x : bounds.maximum.x,
                    y == 0 ? bounds.minimum.y : bounds.maximum.y,
                    z == 0 ? bounds.minimum.z : bounds.maximum.z);
                for (int axis = 0; axis < 3; ++axis)
                {
                    minimum[axis] = std::min (minimum[axis], point[axis]);
                    maximum[axis] = std::max (maximum[axis], point[axis]);
                }
            }

    const auto extent = std::max ({ maximum[0] - minimum[0], maximum[1] - minimum[1],
                                    maximum[2] - minimum[2] });
    if (! std::isfinite (extent) || extent <= 1.0e-6f) return false;
    for (int axis = 0; axis < 3; ++axis)
    {
        target[axis] = (minimum[axis] + maximum[axis]) * 0.5f;
        if (! std::isfinite (target[axis])) return false;
    }
    halfHeight = extent * 0.62f;
    origin = { target[0], target[1], target[2] + extent * 2.0f };
    return std::all_of (origin.begin(), origin.end(),
                        [] (float value) { return std::isfinite (value); });
}

class MetalVolumeFrame final : public NativeVolumeFrame
{
public:
    ~MetalVolumeFrame() override
    {
        if (! hasResources()) return;
        std::lock_guard<std::mutex> lock (arbitgpu::sokolmetal::mutex());
        if (sg_isvalid()) destroyUnlocked();
    }

    const std::string& backend() const noexcept override { return backend_; }
    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }
    std::uintptr_t colorImageHandle() const noexcept override { return colorImage.id; }
    std::uintptr_t colorTextureViewHandle() const noexcept override
    {
        return colorTextureView.id;
    }

    bool hasResources() const noexcept
    {
        return colorImage.id != 0 || colorAttachmentView.id != 0
            || colorTextureView.id != 0 || volumeImage.id != 0
            || volumeTextureView.id != 0 || sampler.id != 0
            || shader.id != 0 || pipeline.id != 0;
    }

    void destroyUnlocked() noexcept
    {
        if (pipeline.id != 0) sg_destroy_pipeline (pipeline);
        if (shader.id != 0) sg_destroy_shader (shader);
        if (sampler.id != 0) sg_destroy_sampler (sampler);
        if (volumeTextureView.id != 0) sg_destroy_view (volumeTextureView);
        if (volumeImage.id != 0) sg_destroy_image (volumeImage);
        if (colorTextureView.id != 0) sg_destroy_view (colorTextureView);
        if (colorAttachmentView.id != 0) sg_destroy_view (colorAttachmentView);
        if (colorImage.id != 0) sg_destroy_image (colorImage);
        colorImage = {};
        colorAttachmentView = {};
        colorTextureView = {};
        volumeImage = {};
        volumeTextureView = {};
        sampler = {};
        shader = {};
        pipeline = {};
    }

    std::string backend_ = "metal";
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    sg_image colorImage = {};
    sg_view colorAttachmentView = {};
    sg_view colorTextureView = {};
    sg_image volumeImage = {};
    sg_view volumeTextureView = {};
    sg_sampler sampler = {};
    sg_shader shader = {};
    sg_pipeline pipeline = {};
};

class MetalVolumeExecutionBackend final : public NativeVolumeExecutionBackend
{
public:
    NativeVolumeExecutionCapabilities capabilities() const override
    {
        std::lock_guard<std::mutex> lock (arbitgpu::sokolmetal::mutex());
        return capabilitiesUnlocked();
    }

    NativeVolumeSubmission render (const NativeVolumeDrawRequest& request) override
    {
        NativeVolumeSubmission result;
        std::lock_guard<std::mutex> lock (arbitgpu::sokolmetal::mutex());
        const auto available = capabilitiesUnlocked();
        if (! available.volume.nativeGpuAvailable)
        {
            result.error = available.backend.empty() ? kUnavailable : "Metal volume capabilities are unavailable";
            return result;
        }
        if (request.volume == nullptr || request.width == 0 || request.height == 0
            || request.width > available.maxRenderExtent
            || request.height > available.maxRenderExtent
            || static_cast<std::uint64_t> (request.width) * request.height
                > available.maxRenderPixels
            || request.volume->bytes().size() > available.volume.maxVolumeBytes)
        {
            result.error = "Metal native volume request exceeds backend limits";
            return result;
        }
        if (request.volume->storage() != videowire::VolumeStorage::dense)
        {
            result.error = "Metal native volume execution currently admits dense volumes";
            return result;
        }

        const auto& dimensions = request.volume->dimensions();
        std::size_t expectedBytes = 0;
        if (dimensions.width == 0 || dimensions.height == 0 || dimensions.depth == 0
            || dimensions.width > available.volume.maxTexture3DDimension
            || dimensions.height > available.volume.maxTexture3DDimension
            || dimensions.depth > available.volume.maxTexture3DDimension)
        {
            result.error = "Metal native volume dimensions exceed the 3D texture limit";
            return result;
        }
        if (! expectedByteCount (*request.volume, expectedBytes)
            || expectedBytes != request.volume->bytes().size())
        {
            result.error = "Metal native volume payload does not match its format and dimensions";
            return result;
        }

        sg_pixel_format format = SG_PIXELFORMAT_NONE;
        std::array<float, 16> worldToLocal {};
        std::array<float, 3> cameraOrigin {};
        std::array<float, 3> cameraTarget {};
        float cameraHalfHeight = 0.0f;
        if (! textureFormat (request.volume->format(), format)
            || ! invertAffine (request.volume->transform().localToWorld, worldToLocal)
            || ! cameraForVolume (*request.volume, cameraOrigin, cameraTarget, cameraHalfHeight))
        {
            result.error = "Metal native volume requires an invertible finite affine transform";
            return result;
        }

        auto frame = std::make_shared<MetalVolumeFrame>();
        frame->width_ = request.width;
        frame->height_ = request.height;
        auto fail = [&] (const char* error)
        {
            frame->destroyUnlocked();
            result.error = error;
            return result;
        };

        sg_image_desc volumeDesc = {};
        volumeDesc.type = SG_IMAGETYPE_3D;
        volumeDesc.usage.immutable = true;
        volumeDesc.width = static_cast<int> (dimensions.width);
        volumeDesc.height = static_cast<int> (dimensions.height);
        volumeDesc.num_slices = static_cast<int> (dimensions.depth);
        volumeDesc.pixel_format = format;
        volumeDesc.data.mip_levels[0] = {
            request.volume->bytes().data(), request.volume->bytes().size() };
        volumeDesc.label = "arbit-metal-volume-density";
        frame->volumeImage = sg_make_image (&volumeDesc);
        sg_view_desc volumeViewDesc = {};
        volumeViewDesc.texture.image = frame->volumeImage;
        frame->volumeTextureView = sg_make_view (&volumeViewDesc);

        sg_sampler_desc samplerDesc = {};
        samplerDesc.min_filter = SG_FILTER_LINEAR;
        samplerDesc.mag_filter = SG_FILTER_LINEAR;
        samplerDesc.wrap_u = SG_WRAP_CLAMP_TO_EDGE;
        samplerDesc.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
        samplerDesc.wrap_w = SG_WRAP_CLAMP_TO_EDGE;
        samplerDesc.label = "arbit-metal-volume-sampler";
        frame->sampler = sg_make_sampler (&samplerDesc);

        sg_image_desc colorDesc = {};
        colorDesc.usage.color_attachment = true;
        colorDesc.width = static_cast<int> (request.width);
        colorDesc.height = static_cast<int> (request.height);
        colorDesc.pixel_format = SG_PIXELFORMAT_BGRA8;
        colorDesc.sample_count = 1;
        colorDesc.label = "arbit-metal-volume-color";
        frame->colorImage = sg_make_image (&colorDesc);
        sg_view_desc colorAttachmentDesc = {};
        colorAttachmentDesc.color_attachment.image = frame->colorImage;
        frame->colorAttachmentView = sg_make_view (&colorAttachmentDesc);
        sg_view_desc colorTextureDesc = {};
        colorTextureDesc.texture.image = frame->colorImage;
        frame->colorTextureView = sg_make_view (&colorTextureDesc);

        sg_shader_desc shaderDesc = {};
        shaderDesc.vertex_func.source = kVertexShader;
        shaderDesc.fragment_func.source = kFragmentShader;
        shaderDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        shaderDesc.uniform_blocks[0].size = sizeof (VolumeUniforms);
        shaderDesc.uniform_blocks[0].msl_buffer_n = 0;
        shaderDesc.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        shaderDesc.views[0].texture.image_type = SG_IMAGETYPE_3D;
        shaderDesc.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        shaderDesc.views[0].texture.msl_texture_n = 0;
        shaderDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        shaderDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        shaderDesc.samplers[0].msl_sampler_n = 0;
        shaderDesc.texture_sampler_pairs[0] = {
            SG_SHADERSTAGE_FRAGMENT, 0, 0, "volume" };
        shaderDesc.label = "arbit-metal-volume-raymarch-shader";
        frame->shader = sg_make_shader (&shaderDesc);

        sg_pipeline_desc pipelineDesc = {};
        pipelineDesc.shader = frame->shader;
        pipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_BGRA8;
        pipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        pipelineDesc.primitive_type = SG_PRIMITIVETYPE_TRIANGLES;
        pipelineDesc.sample_count = 1;
        pipelineDesc.label = "arbit-metal-volume-raymarch-pipeline";
        frame->pipeline = sg_make_pipeline (&pipelineDesc);

        const bool resourcesOk =
            resourceValid (sg_query_image_state (frame->volumeImage))
            && resourceValid (sg_query_view_state (frame->volumeTextureView))
            && resourceValid (sg_query_sampler_state (frame->sampler))
            && resourceValid (sg_query_image_state (frame->colorImage))
            && resourceValid (sg_query_view_state (frame->colorAttachmentView))
            && resourceValid (sg_query_view_state (frame->colorTextureView))
            && resourceValid (sg_query_shader_state (frame->shader))
            && resourceValid (sg_query_pipeline_state (frame->pipeline));
        if (! resourcesOk)
            return fail ("Metal native volume GPU resource creation failed");

        VolumeUniforms uniforms {};
        std::copy (worldToLocal.begin(), worldToLocal.end(), uniforms.worldToLocal);
        const auto& bounds = request.volume->bounds();
        uniforms.boundsMinimum[0] = bounds.minimum.x;
        uniforms.boundsMinimum[1] = bounds.minimum.y;
        uniforms.boundsMinimum[2] = bounds.minimum.z;
        uniforms.boundsMaximum[0] = bounds.maximum.x;
        uniforms.boundsMaximum[1] = bounds.maximum.y;
        uniforms.boundsMaximum[2] = bounds.maximum.z;
        std::copy (cameraOrigin.begin(), cameraOrigin.end(), uniforms.cameraOrigin);
        std::copy (cameraTarget.begin(), cameraTarget.end(), uniforms.cameraTarget);
        uniforms.extentCameraSteps[0] = static_cast<float> (request.width);
        uniforms.extentCameraSteps[1] = static_cast<float> (request.height);
        uniforms.extentCameraSteps[2] = cameraHalfHeight;
        uniforms.extentCameraSteps[3] = static_cast<float> (std::min<std::uint32_t> (
            kMaximumRaySteps,
            std::max ({ dimensions.width, dimensions.height, dimensions.depth })));

        sg_pass pass = {};
        pass.attachments.colors[0] = frame->colorAttachmentView;
        pass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
        pass.action.colors[0].store_action = SG_STOREACTION_STORE;
        pass.action.colors[0].clear_value = {
            7.0f / 255.0f, 10.0f / 255.0f, 18.0f / 255.0f, 1.0f };
        pass.label = "arbit-metal-volume-raymarch-pass";
        sg_begin_pass (&pass);
        sg_apply_pipeline (frame->pipeline);
        sg_bindings bindings = {};
        bindings.views[0] = frame->volumeTextureView;
        bindings.samplers[0] = frame->sampler;
        sg_apply_bindings (&bindings);
        const sg_range uniformRange = { &uniforms, sizeof (uniforms) };
        sg_apply_uniforms (0, &uniformRange);
        sg_draw (0, 3, 1);
        sg_end_pass();
        sg_commit();

        result.rendered = true;
        result.frame = std::move (frame);
        return result;
    }

private:
    static NativeVolumeExecutionCapabilities capabilitiesUnlocked()
    {
        NativeVolumeExecutionCapabilities result;
        if (! arbitgpu::sokolmetal::ensure()) return result;
        if (sg_query_backend() != SG_BACKEND_METAL_MACOS) return result;

        const auto limits = sg_query_limits();
        const auto color = sg_query_pixelformat (SG_PIXELFORMAT_BGRA8);
        const auto r8 = sg_query_pixelformat (SG_PIXELFORMAT_R8);
        const auto r16f = sg_query_pixelformat (SG_PIXELFORMAT_R16F);
        const auto r32f = sg_query_pixelformat (SG_PIXELFORMAT_R32F);
        if (limits.max_image_size_2d <= 0 || limits.max_image_size_3d <= 0
            || ! color.render || ! color.sample
            || ! r8.sample || ! r8.filter || ! r16f.sample || ! r16f.filter
            || ! r32f.sample || ! r32f.filter)
            return result;

        const auto maximumRenderExtent = static_cast<std::uint32_t> (
            std::min (limits.max_image_size_2d,
                      static_cast<int> (kMaximumRenderExtent)));
        result.volume.nativeGpuAvailable = true;
        result.volume.volumeRaymarch = true;
        result.volume.denseVolumeUpload = true;
        result.volume.sparseBrickUpload = false;
        result.volume.maxTexture3DDimension = static_cast<std::uint32_t> (
            std::min (limits.max_image_size_3d,
                      static_cast<int> (kMaximumTexture3DExtent)));
        result.volume.maxBrickEdge = result.volume.maxTexture3DDimension;
        result.volume.maxSparseBrickCount = 65536;
        result.volume.maxVolumeBytes = kMaximumVolumeBytes;
        result.backend = "metal";
        result.maxRenderExtent = maximumRenderExtent;
        result.maxRenderPixels = std::min<std::uint64_t> (
            static_cast<std::uint64_t> (maximumRenderExtent) * maximumRenderExtent,
            static_cast<std::uint64_t> (kMaximumRenderExtent) * kMaximumRenderExtent);
        return result;
    }
};
} // namespace

NativeVolumeExecutionBackend& nativeVolumeExecutionBackend()
{
    static MetalVolumeExecutionBackend backend;
    return backend;
}
} // namespace videohelper::volume
