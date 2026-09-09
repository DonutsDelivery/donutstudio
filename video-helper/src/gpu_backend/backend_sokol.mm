#include "backend.h"
#include "../geometry_core_admission.h"
#include "../geometry_core_diagnostic_colors.h"
#include "../sdf_native_program.h"
#include "sokol_metal_context.h"
#if ARBIT_HAVE_VIEWPORT
#include "particle_engine_metal.h"
#if ARBIT_HAVE_METAL_GENERATORS
#include "metal_shader_generator.h"
#endif
#include "../gl_loader.h"
#include "../particle_engine.h"
#include "../renderer.h"
#include "../shader_generator.h"
#include "../../../shared/ColorTransformGpuMath.h"
#include <atomic>
#endif

#import <CoreFoundation/CoreFoundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <OpenGL/CGLIOSurface.h>
#import <OpenGL/OpenGL.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

#define SOKOL_IMPL
#define SOKOL_METAL
#include "sokol_gfx.h"

namespace
{

// The viewport is owned by another translation unit and may be torn down during
// process-static destruction. Keep the backend lock alive until the process is
// gone so a late renderer shutdown can never lock a destroyed std::mutex.
std::mutex& sokolMutex()
{
    static auto* mutex = new std::mutex();
    return *mutex;
}

std::uint64_t nextMetalRendererGeneration() noexcept
{
    static std::uint64_t generation = 0;
    return ++generation;
}
id<MTLDevice> gMetalDevice = nil;
std::string gSokolError;
std::string gSokolLastLog;
std::mutex& sokolLogMutex()
{
    static auto* mutex = new std::mutex();
    return *mutex;
}

void captureSokolLog (const char*, std::uint32_t logLevel, std::uint32_t,
                      const char* message, std::uint32_t, const char*, void*)
{
    if (logLevel > 1 || message == nullptr)
        return;
    std::lock_guard<std::mutex> lock (sokolLogMutex());
    gSokolLastLog = message;
}

std::string lastSokolLog()
{
    std::lock_guard<std::mutex> lock (sokolLogMutex());
    return gSokolLastLog;
}

bool ensureSokolMetal()
{
    if (sg_isvalid())
        return true;
    if (! gSokolError.empty())
        return false;

    gMetalDevice = MTLCreateSystemDefaultDevice();
    if (gMetalDevice == nil)
    {
        gSokolError = "Metal returned no default device";
        return false;
    }

    sg_desc desc = {};
    desc.environment.metal.device = (__bridge const void*) gMetalDevice;
    desc.environment.defaults.color_format = SG_PIXELFORMAT_BGRA8;
    desc.environment.defaults.depth_format = SG_PIXELFORMAT_DEPTH_STENCIL;
    desc.environment.defaults.sample_count = 1;
    desc.logger.func = captureSokolLog;
    sg_setup (&desc);
    if (! sg_isvalid())
    {
        gSokolError = "sokol_gfx Metal initialization failed";
        return false;
    }
    return true;
}

uint32_t fnv1a (const void* data, size_t size)
{
    const auto* bytes = static_cast<const uint8_t*> (data);
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < size; ++i)
    {
        hash ^= bytes[i];
        hash *= 16777619u;
    }
    return hash;
}

std::string deviceName (id<MTLDevice> device)
{
    if (device == nil || device.name == nil)
        return {};
    return std::string ([device.name UTF8String]);
}

void readBackSubmittedWork (id<MTLDevice> device, id<MTLBuffer> computeBuffer,
                            id<MTLTexture> renderTexture,
                            std::array<uint32_t, 4>& computed,
                            std::array<uint8_t, 4 * 4 * 4>& pixels)
{
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
    id<MTLBuffer> textureReadback = [device newBufferWithLength:pixels.size()
                                                        options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (computeBuffer.storageMode == MTLStorageModeManaged)
        [blit synchronizeResource:computeBuffer];
    if (renderTexture != nil && textureReadback != nil)
    {
        [blit copyFromTexture:renderTexture
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:MTLOriginMake (0, 0, 0)
                   sourceSize:MTLSizeMake (4, 4, 1)
                     toBuffer:textureReadback
            destinationOffset:0
       destinationBytesPerRow:16
     destinationBytesPerImage:pixels.size()];
    }
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];

    if (computeBuffer != nil && computeBuffer.contents != nullptr)
        std::memcpy (computed.data(), computeBuffer.contents, sizeof (computed));
    if (textureReadback != nil && textureReadback.contents != nullptr)
        std::memcpy (pixels.data(), textureReadback.contents, pixels.size());

#if ! __has_feature(objc_arc)
    [textureReadback release];
#endif
}

bool resourceValid (sg_resource_state state)
{
    return state == SG_RESOURCESTATE_VALID;
}

bool compileMetalShaderStage (const char* source, const char* stage,
                              std::string& error)
{
    NSError* libraryError = nil;
    id<MTLLibrary> library = [gMetalDevice
        newLibraryWithSource:[NSString stringWithUTF8String:source]
        options:nil
        error:&libraryError];
    id<MTLFunction> function = [library newFunctionWithName:@"_main"];
    if (library != nil && function != nil)
        return true;

    error = "Metal fixture " + std::string (stage) + " shader compilation failed";
    if (libraryError != nil && libraryError.localizedDescription != nil)
        error += ": " + std::string (libraryError.localizedDescription.UTF8String);
    else if (library != nil)
        error += ": entry point _main was not found";
    return false;
}

struct FixtureUniforms
{
    float objectMatrix[16];
    float cameraRotation[4];
    float cameraTranslation[4];
    float projection[4];
    float baseColor[4];
    float materialParams[4];
    float timeMixEndColorAndTime[4];
    float emissive[4];
    float ambient[4];
    float lightRotations[16][4];
    float lightColors[16][4];
    float lightPositions[16][4];
    float lightCones[16][4];
    std::uint32_t lightKinds[16][4];
    std::uint32_t materialKind[4];
    diffractionmaterial::PhysicalDiffractionGpuParameters diffraction;
    float diffractionIncidentDirectionAndIntensity[4];
    std::uint32_t diffractionPathKindAndBounce[4];
    float diffractionFoilField[4][4];
    float diffractionOccupancyRectangles[5][4];
    std::uint32_t diffractionSpatialCounts[4];
    std::uint32_t diffractionEvaluationSchedule[4];
    float noteInstanceTransforms[visualnoteinstancing::kMaximumInstances][4];
    // Geometry Core and Block C remain independent instance domains. x selects
    // the Geometry Core matrix stream; y selects one Block C row for a geometry
    // draw, or UINT_MAX to use Metal's instance_id for an ordinary scene draw.
    std::uint32_t geometryInstanceControl[4];
};
static_assert (offsetof (FixtureUniforms, objectMatrix) == 0
               && offsetof (FixtureUniforms, materialKind) == 1472
               && offsetof (FixtureUniforms, diffraction) == 1488
               && offsetof (FixtureUniforms, diffractionIncidentDirectionAndIntensity) == 1888
               && offsetof (FixtureUniforms, diffractionPathKindAndBounce) == 1904
               && offsetof (FixtureUniforms, diffractionFoilField) == 1920
               && offsetof (FixtureUniforms, diffractionOccupancyRectangles) == 1984
               && offsetof (FixtureUniforms, diffractionSpatialCounts) == 2064
               && offsetof (FixtureUniforms, diffractionEvaluationSchedule) == 2080
               && offsetof (FixtureUniforms, noteInstanceTransforms) == 2096
               && offsetof (FixtureUniforms, geometryInstanceControl) == 4144
               && sizeof (FixtureUniforms) == 4160,
               "Metal fixture uniform layout changed");

struct MetalFixtureInstanceGpuRecord
{
    std::array<float, 16> matrix {};
    std::array<float, 4> identityColor {};
};
static_assert (sizeof (MetalFixtureInstanceGpuRecord) == 20 * sizeof (float),
               "Metal fixture instance record layout changed");

struct SceneAovUniforms
{
    float objectRotation[4];
    float objectTranslation[4];
    float objectScale[4];
    float cameraRotation[4];
    float cameraTranslation[4];
    float projection[4];
    float emission[4];
    std::uint32_t identifiers[4];
};
static_assert (sizeof (SceneAovUniforms) == 128,
               "Metal scene AOV uniform layout changed");

[[maybe_unused]] const char* kSceneAovMetalShader = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Uniforms {
    float4 objectRotation;
    float4 objectTranslation;
    float4 objectScale;
    float4 cameraRotation;
    float4 cameraTranslation;
    float4 projection;
    float4 emission;
    uint4 identifiers;
};
struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
};
struct VertexOut {
    float4 position [[position]];
    float3 normal;
    float depth;
};
float3 rotateQuaternion(float4 q, float3 value) {
    return value + 2.0f * cross(q.xyz, cross(q.xyz, value) + q.w * value);
}
vertex VertexOut sceneAovVertex(VertexIn in [[stage_in]],
                               constant Uniforms& u [[buffer(1)]]) {
    VertexOut out;
    const float3 world = rotateQuaternion(
        u.objectRotation, in.position * u.objectScale.xyz) + u.objectTranslation.xyz;
    out.normal = normalize(rotateQuaternion(
        u.objectRotation, in.normal / u.objectScale.xyz));
    const float4 inverseCamera = float4(-u.cameraRotation.xyz, u.cameraRotation.w);
    const float3 camera = rotateQuaternion(
        inverseCamera, world - u.cameraTranslation.xyz);
    const float distance = -camera.z;
    out.depth = distance;
    out.position = float4(camera.x / (u.projection.x * u.projection.y),
                          camera.y / u.projection.x,
                          distance * u.projection.w / (u.projection.w - u.projection.z)
                            - u.projection.z * u.projection.w
                              / (u.projection.w - u.projection.z), distance);
    return out;
}
fragment float sceneAovDepth(VertexOut in [[stage_in]]) { return in.depth; }
fragment float4 sceneAovNormal(VertexOut in [[stage_in]]) {
    return float4(normalize(in.normal), 1.0f);
}
fragment float4 sceneAovEmission(VertexOut, constant Uniforms& u [[buffer(1)]]) {
    return float4(u.emission.xyz, 1.0f);
}
fragment float sceneAovMask(VertexOut) { return 1.0f; }
fragment uint sceneAovMaterialId(VertexOut, constant Uniforms& u [[buffer(1)]]) {
    return u.identifiers.x;
}
fragment uint sceneAovObjectId(VertexOut, constant Uniforms& u [[buffer(1)]]) {
    return u.identifiers.y;
}
)metal";

class MetalFixtureSceneResources final : public arbitgpu::NativeFixtureSceneResources
{
public:
    ~MetalFixtureSceneResources() override
    {
        if (hasResources())
        {
            std::lock_guard<std::mutex> lock (sokolMutex());
            destroyUnlocked();
        }
    }

    const std::string& backend() const noexcept override { return backend_; }

    bool hasResources() const noexcept
    {
        return vertexBuffer.id != 0 || indexBuffer.id != 0 || !textureImages.empty()
            || !textureViews.empty() || !samplers.empty() || shader.id != 0
            || geometryInstanceBuffer.id != 0 || !pipelines.empty();
    }

    void destroyUnlocked() noexcept
    {
        for (const auto pipeline : pipelines)
            if (pipeline.id != 0) sg_destroy_pipeline (pipeline);
        if (shader.id != 0) sg_destroy_shader (shader);
        for (const auto sampler : samplers)
            if (sampler.id != 0) sg_destroy_sampler (sampler);
        for (const auto view : textureViews)
            if (view.id != 0) sg_destroy_view (view);
        for (const auto image : textureImages)
            if (image.id != 0) sg_destroy_image (image);
        if (geometryInstanceBuffer.id != 0) sg_destroy_buffer (geometryInstanceBuffer);
        if (indexBuffer.id != 0) sg_destroy_buffer (indexBuffer);
        if (vertexBuffer.id != 0) sg_destroy_buffer (vertexBuffer);
        vertexBuffer = {};
        indexBuffer = {};
        geometryInstanceBuffer = {};
        textureImages.clear();
        textureViews.clear();
        samplers.clear();
        shader = {};
        pipelines.clear();
    }

    std::string backend_ = "metal";
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> snapshot;
    std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> materialProgram;
    sg_buffer vertexBuffer = {};
    sg_buffer indexBuffer = {};
    sg_buffer geometryInstanceBuffer = {};
    bool instancedSharedGeometry = false;
    bool diagnosticInstanceIdentityColors = false;
    std::shared_ptr<const videohelper::geometry::AdmittedPlanValue> geometryAdmission;
    std::vector<sg_image> textureImages;
    std::vector<sg_view> textureViews;
    std::vector<sg_sampler> samplers;
    sg_shader shader = {};
    // Opaque/mask then blend, or diffraction, each with double-sided,
    // positive, and reflected winding.
    std::vector<sg_pipeline> pipelines;
    std::uint64_t rendererGeneration = 0;
};

class MetalFixtureSceneFrame final : public arbitgpu::NativeFixtureSceneFrame
{
public:
    ~MetalFixtureSceneFrame() override
    {
        if (hasResources())
        {
            std::lock_guard<std::mutex> lock (sokolMutex());
            destroyUnlocked();
        }
    }

    const std::string& backend() const noexcept override { return backend_; }
    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }
    std::uintptr_t colorImageHandle() const noexcept override { return colorImage.id; }
    std::uintptr_t colorTextureViewHandle() const noexcept override
    {
        return colorTextureView.id;
    }
    std::uintptr_t depthImageHandle() const noexcept override { return depthImage.id; }
    std::uintptr_t depthTextureViewHandle() const noexcept override
    {
        return depthTextureView.id;
    }
    std::uintptr_t nativeResourceCacheIdentity() const noexcept override
    {
        return reinterpret_cast<std::uintptr_t> (staticResources.get());
    }
    arbitgpu::NativeTextureViewDescriptor colorTextureDescriptor() const noexcept override
    {
        return { backend_, arbitgpu::NativeTextureViewKind::Texture2D,
                 arbitgpu::NativeTexturePixelFormat::Bgra8Unorm, colorImage.id, colorTextureView.id,
                 width_, height_, 1, true,
                 reinterpret_cast<std::uintptr_t>((__bridge void*)gMetalDevice), rendererGeneration_ };
    }
    arbitgpu::NativeTextureViewDescriptor depthTextureDescriptor() const noexcept override
    {
        return { backend_, arbitgpu::NativeTextureViewKind::Texture2D,
                 arbitgpu::NativeTexturePixelFormat::R32Float, depthImage.id, depthTextureView.id,
                 width_, height_, 1, true,
                 reinterpret_cast<std::uintptr_t>((__bridge void*)gMetalDevice), rendererGeneration_ };
    }

    bool hasResources() const noexcept
    {
        return colorImage.id != 0 || colorAttachmentView.id != 0
            || colorTextureView.id != 0 || depthImage.id != 0 || depthView.id != 0
            || depthTextureView.id != 0 || depthStencilImage.id != 0
            || depthStencilView.id != 0;
    }

    void destroyUnlocked() noexcept
    {
        if (depthTextureView.id != 0) sg_destroy_view (depthTextureView);
        if (depthView.id != 0) sg_destroy_view (depthView);
        if (depthImage.id != 0) sg_destroy_image (depthImage);
        if (depthStencilView.id != 0) sg_destroy_view (depthStencilView);
        if (depthStencilImage.id != 0) sg_destroy_image (depthStencilImage);
        if (colorTextureView.id != 0) sg_destroy_view (colorTextureView);
        if (colorAttachmentView.id != 0) sg_destroy_view (colorAttachmentView);
        if (colorImage.id != 0) sg_destroy_image (colorImage);
        colorImage = {};
        colorAttachmentView = {};
        colorTextureView = {};
        depthImage = {};
        depthView = {};
        depthTextureView = {};
        depthStencilImage = {};
        depthStencilView = {};
    }

    std::string backend_ = "metal";
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::shared_ptr<const MetalFixtureSceneResources> staticResources;
    std::uint64_t rendererGeneration_ = 0;
    sg_image colorImage = {};
    sg_view colorAttachmentView = {};
    sg_view colorTextureView = {};
    sg_image depthImage = {};
    sg_view depthView = {};
    sg_view depthTextureView = {};
    sg_image depthStencilImage = {};
    sg_view depthStencilView = {};
};

struct DeformationComputeUniforms
{
    std::uint32_t vertexCount = 0;
    std::uint32_t jointCount = 0;
    std::uint32_t morphTargetCount = 0;
    std::uint32_t padding = 0;
};
static_assert (sizeof (DeformationComputeUniforms) == 16,
               "Metal deformation uniform layout changed");

class MetalDeformationResources final : public arbitgpu::NativeDeformationResources
{
public:
    ~MetalDeformationResources() override
    {
        if (hasResources())
        {
            std::lock_guard<std::mutex> lock (sokolMutex());
            destroyUnlocked();
        }
    }

    const std::string& backend() const noexcept override { return backend_; }

    bool hasResources() const noexcept
    {
        return baseVertices.id != 0 || deformedVertices.id != 0
            || indexBuffer.id != 0 || jointIndices.id != 0
            || jointWeights.id != 0 || morphPositions.id != 0
            || morphNormals.id != 0 || jointPalette.id != 0
            || morphWeights.id != 0 || computeShader.id != 0
            || computePipeline.id != 0 || drawShader.id != 0
            || drawPipeline.id != 0;
    }

    void destroyUnlocked() noexcept
    {
        if (drawPipeline.id != 0) sg_destroy_pipeline (drawPipeline);
        if (drawShader.id != 0) sg_destroy_shader (drawShader);
        if (computePipeline.id != 0) sg_destroy_pipeline (computePipeline);
        if (computeShader.id != 0) sg_destroy_shader (computeShader);
        for (auto view : { morphWeightsView, jointPaletteView, morphNormalsView,
                           morphPositionsView, jointWeightsView, jointIndicesView,
                           deformedVerticesView, baseVerticesView })
            if (view.id != 0) sg_destroy_view (view);
        for (auto buffer : { morphWeights, jointPalette, morphNormals, morphPositions,
                             jointWeights, jointIndices, indexBuffer,
                             deformedVertices, baseVertices })
            if (buffer.id != 0) sg_destroy_buffer (buffer);
        baseVertices = {}; baseVerticesView = {};
        deformedVertices = {}; deformedVerticesView = {};
        indexBuffer = {};
        jointIndices = {}; jointIndicesView = {};
        jointWeights = {}; jointWeightsView = {};
        morphPositions = {}; morphPositionsView = {};
        morphNormals = {}; morphNormalsView = {};
        jointPalette = {}; jointPaletteView = {};
        morphWeights = {}; morphWeightsView = {};
        computeShader = {}; computePipeline = {};
        drawShader = {}; drawPipeline = {};
    }

    std::string backend_ = "metal";
    std::shared_ptr<const arbitgpu::NativeDeformationScene> source;
    std::shared_ptr<const MetalFixtureSceneResources> fixture;
    std::uint32_t vertexCount = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t jointCount = 0;
    std::uint32_t morphTargetCount = 0;
    sg_buffer baseVertices = {}; sg_view baseVerticesView = {};
    sg_buffer deformedVertices = {}; sg_view deformedVerticesView = {};
    sg_buffer indexBuffer = {};
    sg_buffer jointIndices = {}; sg_view jointIndicesView = {};
    sg_buffer jointWeights = {}; sg_view jointWeightsView = {};
    sg_buffer morphPositions = {}; sg_view morphPositionsView = {};
    sg_buffer morphNormals = {}; sg_view morphNormalsView = {};
    sg_buffer jointPalette = {}; sg_view jointPaletteView = {};
    sg_buffer morphWeights = {}; sg_view morphWeightsView = {};
    sg_shader computeShader = {}; sg_pipeline computePipeline = {};
    sg_shader drawShader = {}; sg_pipeline drawPipeline = {};
    mutable std::mutex submissionMutex;
};

class MetalDeformationFrame final : public arbitgpu::NativeFixtureSceneFrame
{
public:
    ~MetalDeformationFrame() override
    {
        if (hasResources())
        {
            std::lock_guard<std::mutex> lock (sokolMutex());
            destroyUnlocked();
        }
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
            || colorTextureView.id != 0 || depthImage.id != 0 || depthView.id != 0;
    }

    void destroyUnlocked() noexcept
    {
        if (depthView.id != 0) sg_destroy_view (depthView);
        if (depthImage.id != 0) sg_destroy_image (depthImage);
        if (colorTextureView.id != 0) sg_destroy_view (colorTextureView);
        if (colorAttachmentView.id != 0) sg_destroy_view (colorAttachmentView);
        if (colorImage.id != 0) sg_destroy_image (colorImage);
        colorImage = {}; colorAttachmentView = {}; colorTextureView = {};
        depthImage = {}; depthView = {};
    }

    std::string backend_ = "metal";
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::shared_ptr<const MetalDeformationResources> staticResources;
    sg_image colorImage = {};
    sg_view colorAttachmentView = {};
    sg_view colorTextureView = {};
    sg_image depthImage = {};
    sg_view depthView = {};
};

struct alignas(16) SdfGpuRecord
{
    std::uint32_t header[4];
    float parameters0[4];
    float parameters1[4];
};
static_assert (sizeof (SdfGpuRecord) == 48, "Metal SDF record layout changed");

struct alignas(16) SdfUniforms
{
    SdfGpuRecord records[arbitgpu::kNativeSdfMaximumRecords];
    float raymarch[4];
    std::uint32_t quality[4];
    std::uint32_t program[4];
};
static_assert (sizeof (SdfUniforms) == 3120, "Metal SDF uniform layout changed");

float sdfHalfToFloat (std::uint16_t value) noexcept
{
    const std::uint32_t sign = static_cast<std::uint32_t> (value & 0x8000u) << 16u;
    const std::uint32_t exponent = (value >> 10u) & 0x1fu;
    const std::uint32_t fraction = value & 0x03ffu;
    std::uint32_t bits = sign;
    if (exponent == 0)
    {
        if (fraction != 0)
        {
            std::uint32_t normalized = fraction;
            std::uint32_t shift = 0;
            while ((normalized & 0x0400u) == 0) { normalized <<= 1u; ++shift; }
            bits |= (113u - shift) << 23u;
            bits |= (normalized & 0x03ffu) << 13u;
        }
    }
    else if (exponent == 0x1fu)
    {
        bits |= 0x7f800000u | (fraction << 13u);
    }
    else
    {
        bits |= (exponent + 112u) << 23u;
        bits |= fraction << 13u;
    }
    float result = 0.0f;
    std::memcpy (&result, &bits, sizeof (result));
    return result;
}

class MetalSdfSceneFrame final : public arbitgpu::NativeSdfSceneFrame
{
public:
    ~MetalSdfSceneFrame() override
    {
        if (outputBackend_ != nullptr && lifecycle_.value != 0)
            outputBackend_->releaseRenderPassOutputs (lifecycle_);
    }

    const std::string& backend() const noexcept override { return backend_; }
    std::uint32_t width() const noexcept override { return width_; }
    std::uint32_t height() const noexcept override { return height_; }
    std::uintptr_t colorImageHandle() const noexcept override { return colorImage.id; }
    std::uintptr_t colorTextureViewHandle() const noexcept override
    {
        return colorTextureView.id;
    }
    const arbitgpu::FrameMemoryAdmission& frameMemoryAdmission() const noexcept override
    {
        return frameMemory_;
    }
    const arbitgpu::NativeSdfResourceReceipt& sdfResourceReceipt() const noexcept override
    {
        return receipt_;
    }

    bool readColorPixels (std::vector<std::uint8_t>& output) const override
    {
        std::lock_guard<std::mutex> lock (sokolMutex());
        @autoreleasepool
        {
            if (colorImage.id == 0 || width_ == 0 || height_ == 0)
                return false;
            const auto native = sg_mtl_query_image_info (colorImage);
            id<MTLTexture> texture = (__bridge id<MTLTexture>)
                native.tex[native.active_slot];
            id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)
                sg_mtl_command_queue();
            const auto sourceRowBytes = static_cast<NSUInteger> (width_) * 8u;
            const auto alignedRowBytes = (sourceRowBytes + 255u) & ~NSUInteger (255u);
            const auto bufferBytes = alignedRowBytes * static_cast<NSUInteger> (height_);
            id<MTLBuffer> readback = [gMetalDevice newBufferWithLength:bufferBytes
                                                               options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
            if (texture == nil || readback == nil || command == nil || blit == nil)
                return false;
            [blit copyFromTexture:texture
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake (0, 0, 0)
                       sourceSize:MTLSizeMake (width_, height_, 1)
                         toBuffer:readback
                destinationOffset:0
           destinationBytesPerRow:alignedRowBytes
         destinationBytesPerImage:bufferBytes];
            [blit endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted
                || readback.contents == nullptr)
                return false;
            output.resize (static_cast<std::size_t> (width_) * height_ * 4u);
            const auto* source = static_cast<const std::uint8_t*> (readback.contents);
            for (std::uint32_t row = 0; row < height_; ++row)
            {
                const auto* sourcePixels = reinterpret_cast<const std::uint16_t*> (
                    source + static_cast<std::size_t> (row) * alignedRowBytes);
                auto* destination = output.data()
                    + static_cast<std::size_t> (row) * width_ * 4u;
                for (std::uint32_t column = 0; column < width_; ++column)
                {
                    for (std::size_t channel = 0; channel < 4; ++channel)
                    {
                        const auto value = std::clamp (sdfHalfToFloat (
                            sourcePixels[static_cast<std::size_t> (column) * 4u + channel]),
                            0.0f, 1.0f);
                        destination[static_cast<std::size_t> (column) * 4u + channel]
                            = static_cast<std::uint8_t> (std::lround (value * 255.0f));
                    }
                }
            }
#if ! __has_feature(objc_arc)
            [readback release];
#endif
            return true;
        }
    }

    std::string backend_ = "metal";
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    sg_image colorImage = {};
    sg_view colorAttachmentView = {};
    sg_view colorTextureView = {};
    arbitgpu::RenderPassOutputBackend* outputBackend_ = nullptr;
    arbitgpu::RenderPassOutputLifecycleHandle lifecycle_ {};
    arbitgpu::FrameMemoryAdmission frameMemory_ {};
    arbitgpu::NativeSdfResourceReceipt receipt_ {};
};

const char* kSdfVertexShader = R"metal(
#include <metal_stdlib>
using namespace metal;
struct VertexOut { float4 position [[position]]; };
vertex VertexOut _main(uint vertexId [[vertex_id]]) {
    const float2 positions[3] = { float2(-1.0, -1.0),
                                  float2( 3.0, -1.0),
                                  float2(-1.0,  3.0) };
    VertexOut out;
    out.position = float4(positions[vertexId], 0.0, 1.0);
    return out;
}
)metal";

const char* kSdfFragmentShader = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Record { uint4 header; float4 parameters0; float4 parameters1; };
struct Uniforms {
    Record records[64];
    float4 raymarch;
    uint4 quality;
    uint4 program;
};
struct VertexOut { float4 position [[position]]; };
float componentAt(float3 v, int axis) { return axis == 0 ? v.x : (axis == 1 ? v.y : v.z); }
float3 withComponent(float3 v, int axis, float x) {
    if (axis == 0) v.x=x; else if (axis == 1) v.y=x; else v.z=x; return v;
}
float3 rotateAxis(float3 v, float3 axis, float radians) {
    axis=normalize(axis); float c=cos(radians), s=sin(radians);
    return v*c+cross(axis,v)*s+axis*dot(axis,v)*(1.0-c);
}
float3 rotatePair(float3 v, int first, int second, float radians) {
    float a=componentAt(v,first), b=componentAt(v,second), c=cos(radians), s=sin(radians);
    v=withComponent(v,first,c*a-s*b); return withComponent(v,second,s*a+c*b);
}
float boxDistance(float3 p,float3 e) {
    float3 q=abs(p)-e; return length(max(q,float3(0.0)))+min(max(q.x,max(q.y,q.z)),0.0);
}
float primitiveDistance(uint op,float3 p,float4 a,float4 b) {
    if(op==2u)return boxDistance(p,a.xyz);
    if(op==3u)return boxDistance(p,a.xyz-float3(a.w))-a.w;
    if(op==4u)return dot(p,normalize(a.xyz))+a.w;
    if(op==5u)return length(float2(length(p.xz)-a.x,p.y))-a.y;
    if(op==6u){float3 r=p-a.xyz,s=float3(a.w,b.x,b.y)-a.xyz;float t=clamp(dot(r,s)/dot(s,s),0.0,1.0);return length(r-s*t)-b.z;}
    if(op==7u){float2 q=abs(float2(length(p.xz),p.y))-a.xy;return min(max(q.x,q.y),0.0)+length(max(q,float2(0.0)));}
    if(op==8u){float2 q=float2(length(p.xz),p.y),base=float2(a.x,-a.y),side=float2(a.x,-2.0*a.y);float2 cap=float2(q.x-min(q.x,q.y<0.0?a.x:0.0),abs(q.y)-a.y);float2 slope=q-base+side*clamp(dot(base-q,side)/dot(side,side),0.0,1.0);return(slope.x<0.0&&cap.y<0.0?-1.0:1.0)*sqrt(min(dot(cap,cap),dot(slope,slope)));}
    if(op==9u){float3 cell=p*a.x;return abs(dot(sin(cell),cos(cell.zxy)))/a.x-a.y;}
    return length(p)-a.x;
}
float smoothUnionDistance(float a,float b,float radius){float h=clamp(0.5+0.5*(b-a)/radius,0.0,1.0);return mix(b,a,h)-radius*h*(1.0-h);}
float3 childPoint(uint op,float3 p,float4 a,float4 b){
    if(op==16u)return p-a.xyz;
    if(op==17u)return rotateAxis(p,a.xyz,-a.w);
    if(op==18u)return p/a.xyz;
    if(op==19u)return p+a.xyz*0.5-a.xyz*floor((p+a.xyz*0.5)/a.xyz)-a.xyz*0.5;
    if(op==20u){int axis=int(a.x),first=(axis+1)%3,second=(axis+2)%3;float x=componentAt(p,first),y=componentAt(p,second),radius=length(float2(x,y)),sector=6.283185307179586/a.y;float shifted=atan2(y,x)-a.z+sector*0.5;float angle=shifted-sector*floor(shifted/sector)-sector*0.5;p=withComponent(p,first,radius*cos(angle));return withComponent(p,second,radius*sin(angle));}
    if(op==21u)return mix(p,abs(p),a.xyz);
    if(op==22u){int axis=int(a.x);return rotatePair(p,(axis+1)%3,(axis+2)%3,-a.y*componentAt(p,axis));}
    if(op==23u){int axis=int(a.x);return rotatePair(p,(axis+1)%3,axis,-a.y*componentAt(p,axis));}
    if(op==24u){int axis=int(a.x);float factor=1.0-a.y*componentAt(p,axis);p=withComponent(p,(axis+1)%3,componentAt(p,(axis+1)%3)*factor);return withComponent(p,(axis+2)%3,componentAt(p,(axis+2)%3)*factor);}
    if(op==26u)return p+float3(a.x*sin(a.w*p.y),a.y*sin(b.x*p.z),a.z*sin(b.y*p.x));
    return p;
}
float sceneDistance(float3 point,constant Uniforms& u){
    int indices[32],stages[32];float3 points[32];float firstValues[32];int top=0;indices[0]=int(u.program.y);stages[0]=0;points[0]=point;float value=0.0;
    for(int iteration=0;iteration<768;++iteration){int index=indices[top];Record record=u.records[index];uint op=record.header.x;float4 a=record.parameters0,b=record.parameters1;
        if(op<=9u){value=primitiveDistance(op,points[top],a,b);if(top==0)return value;--top;continue;}
        bool binary=op>=10u&&op<=15u;
        if(binary){if(stages[top]==0){stages[top]=1;++top;indices[top]=int(record.header.y);stages[top]=0;points[top]=points[top-1];continue;}if(stages[top]==1){firstValues[top]=value;stages[top]=2;++top;indices[top]=int(record.header.z);stages[top]=0;points[top]=points[top-1];continue;}float first=firstValues[top],second=value;if(op==10u)value=min(first,second);else if(op==11u)value=max(first,second);else if(op==12u)value=max(first,-second);else if(op==13u)value=smoothUnionDistance(first,second,a.x);else if(op==14u)value=-smoothUnionDistance(-first,-second,a.x);else value=-smoothUnionDistance(-first,second,a.x);}
        else if(stages[top]==0){stages[top]=1;++top;indices[top]=int(record.header.y);stages[top]=0;points[top]=childPoint(op,points[top-1],a,b);continue;}
        else{if(op==18u)value*=min(abs(a.x),min(abs(a.y),abs(a.z)));else if(op==25u){float3 q=points[top];value+=a.x*sin(a.y*q.x)*sin(a.y*q.y)*sin(a.y*q.z);}}
        if(top==0)return value;--top;
    }return u.raymarch.y;
}
float qualityScale(uint q){if(q==0u)return 4.0;if(q==1u)return 2.0;if(q==2u)return 1.0;return 0.5;}
float3 sceneNormal(float3 p,constant Uniforms&u){float e=max(u.raymarch.x*qualityScale(u.quality.y),0.000001);return normalize(float3(sceneDistance(p+float3(e,0,0),u)-sceneDistance(p-float3(e,0,0),u),sceneDistance(p+float3(0,e,0),u)-sceneDistance(p-float3(0,e,0),u),sceneDistance(p+float3(0,0,e),u)-sceneDistance(p-float3(0,0,e),u)));}
float sceneShadow(float3 origin,float3 direction,constant Uniforms&u){int limit=u.quality.z==0u?8:(u.quality.z==1u?16:(u.quality.z==2u?32:64));float travel=u.raymarch.x*4.0,visibility=1.0;for(int step=0;step<64;++step){if(step>=limit)break;float field=sceneDistance(origin+direction*travel,u);if(field<u.raymarch.x)return 0.0;visibility=min(visibility,12.0*field/max(travel,u.raymarch.x));travel+=clamp(field,u.raymarch.x*2.0,0.25);if(travel>min(u.raymarch.y,8.0))break;}return clamp(visibility,0.0,1.0);}
fragment float4 _main(VertexOut in [[stage_in]],constant Uniforms&u [[buffer(0)]]){
    float2 extent=u.raymarch.zw,uv=(2.0*in.position.xy-extent)/extent.y;float3 origin=float3(0,0,3),direction=normalize(float3(uv,-1.8));float travel=0.0;bool hit=false;
    for(uint step=0u;step<512u;++step){if(step>=u.quality.x||travel>u.raymarch.y)break;float field=sceneDistance(origin+direction*travel,u);float threshold=max(u.raymarch.x*qualityScale(u.quality.w)*max(1.0,travel*0.05),0.000001);if(field<=threshold){hit=true;break;}travel+=field;}
    if(!hit)return u.program.x==1u?float4(1.0):float4(0.02745,0.03922,0.07059,1.0);float3 point=origin+direction*travel,normal=sceneNormal(point,u);if(u.program.x==1u)return float4(float3(clamp(travel/u.raymarch.y,0.0,1.0)),1.0);if(u.program.x==2u)return float4(normal*0.5+0.5,1.0);float3 light=normalize(float3(-0.45,0.75,0.6));float diffuse=max(dot(normal,light),0.0),shadow=sceneShadow(point+normal*u.raymarch.x*4.0,light,u),rim=pow(1.0-max(dot(normal,-direction),0.0),3.0);return float4(float3(0.12,0.42,0.88)*(0.12+0.88*diffuse*shadow)+float3(0.18,0.35,0.65)*rim,1.0);
}
)metal";

std::shared_ptr<const arbitgpu::NativeSdfCompiledProgram> resolveSdfProgram (
    const arbitgpu::NativeSdfDrawRequest& request, std::string& error)
{
    if (request.compiledProgram != nullptr)
    {
        if (! videohelper::sdf::validateNativeSdfProgram (
                *request.compiledProgram, request.geometry, error))
            return {};
        return request.compiledProgram;
    }
    return videohelper::sdf::compileNativeSdfProgram (request.geometry, error);
}

const char* kFixtureVertexShader = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Uniforms {
    float4x4 objectMatrix;
    float4 cameraRotation;
    float4 cameraTranslation;
    float4 projection;
    float4 baseColor;
    float4 materialParams;
    float4 timeMixEndColorAndTime;
    float4 emissive;
    float4 ambient;
    float4 lightRotations[16];
    float4 lightColors[16];
    float4 lightPositions[16];
    float4 lightCones[16];
    uint4 lightKinds[16];
    uint4 materialKind;
    float4 diffractionGeometry;
    float4 diffractionSecondaryGeometry;
    float4 diffractionMicrostructure;
    float4 diffractionControl;
    float4 diffractionIncident;
    float4 diffractionCoating;
    float4 diffractionRoughness;
    float4 diffractionGrooveField;
    float4 diffractionGrooveVariation;
    float4 diffractionSpectral[8];
    float4 diffractionSpectralZ[8];
    float4 diffractionIncidentDirectionAndIntensity;
    uint4 diffractionPathKindAndBounce;
    float4 diffractionFoilField[4];
    float4 diffractionOccupancyRectangles[5];
    uint4 diffractionSpatialCounts;
    uint4 diffractionEvaluationSchedule;
    float4 noteInstanceTransforms[128];
    uint4 geometryInstanceControl;
};
struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float2 uv [[attribute(2)]];
    float4 color [[attribute(3)]];
    float4 tangent [[attribute(4)]];
    float4 instanceMatrix0 [[attribute(5)]];
    float4 instanceMatrix1 [[attribute(6)]];
    float4 instanceMatrix2 [[attribute(7)]];
    float4 instanceMatrix3 [[attribute(8)]];
    float4 instanceIdentityColor [[attribute(9)]];
};
struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float3 baseColor;
    float3 litBase;
    float3 emissive;
    float3 worldPosition;
    float3 worldNormal;
    float3 worldTangent;
    float bitangentSign;
    float2 metallicRoughness;
    float linearDepth;
    float opacity;
};
float3 rotateQuaternion(float4 q, float3 value) {
    return value + 2.0f * cross(q.xyz, cross(q.xyz, value) + q.w * value);
}
vertex VertexOut _main(VertexIn in [[stage_in]],
                       constant Uniforms& u [[buffer(0)]],
                       uint instance [[instance_id]]) {
    VertexOut out;
    const bool useGeometryInstance = u.geometryInstanceControl.x != 0u;
    const uint noteIndex = useGeometryInstance
        ? u.geometryInstanceControl.y : instance;
    const float4 noteInstance = u.noteInstanceTransforms[noteIndex];
    const float4x4 objectMatrix = useGeometryInstance
        ? float4x4(in.instanceMatrix0, in.instanceMatrix1,
                   in.instanceMatrix2, in.instanceMatrix3)
        : u.objectMatrix;
    const float3 worldPosition = (objectMatrix
        * float4(in.position * noteInstance.w + noteInstance.xyz, 1.0f)).xyz;
    const float3x3 objectLinear = float3x3(
        objectMatrix[0].xyz, objectMatrix[1].xyz, objectMatrix[2].xyz);
    const float objectDeterminant = determinant(objectLinear);
    const float3x3 normalMatrix = float3x3(
        cross(objectLinear[1], objectLinear[2]),
        cross(objectLinear[2], objectLinear[0]),
        cross(objectLinear[0], objectLinear[1])) / objectDeterminant;
    const float3 worldNormal = normalize(normalMatrix * in.normal);
    const float4 inverseCamera = float4(-u.cameraRotation.xyz, u.cameraRotation.w);
    const float3 cameraPosition = rotateQuaternion(
        inverseCamera, worldPosition - u.cameraTranslation.xyz);
    const float distance = -cameraPosition.z;
    const float tanHalfFov = u.projection.x;
    const float aspect = u.projection.y;
    const float nearPlane = u.projection.z;
    const float farPlane = u.projection.w;
    out.position = float4(cameraPosition.x / (tanHalfFov * aspect),
                          cameraPosition.y / tanHalfFov,
                          distance * farPlane / (farPlane - nearPlane)
                              - nearPlane * farPlane / (farPlane - nearPlane), distance);
    out.uv = in.uv;
    out.baseColor = (u.materialParams.w > 0.5f
        ? mix(u.baseColor.xyz, u.timeMixEndColorAndTime.xyz,
              clamp(u.timeMixEndColorAndTime.w, 0.0f, 1.0f))
        : u.baseColor.xyz) * in.color.rgb
        * (useGeometryInstance
            ? mix(float3(1.0f), in.instanceIdentityColor.rgb,
                  step(0.5f, in.instanceIdentityColor.a))
            : float3(1.0f));
    out.litBase = out.baseColor * u.ambient.xyz;
    out.emissive = u.emissive.xyz;
    out.worldPosition = worldPosition;
    out.worldNormal = worldNormal;
    out.worldTangent = normalize(objectLinear * in.tangent.xyz);
    out.bitangentSign = in.tangent.w * sign(objectDeterminant);
    out.metallicRoughness = u.materialParams.xy;
    out.linearDepth = clamp((distance - nearPlane)
        / max(farPlane - nearPlane, 0.000001f), 0.0f, 1.0f);
    out.opacity = u.baseColor.w * in.color.a;
    return out;
}
)metal";

const char* kFixtureFragmentShader = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Uniforms {
    float4x4 objectMatrix;
    float4 cameraRotation;
    float4 cameraTranslation;
    float4 projection;
    float4 baseColor;
    float4 materialParams;
    float4 timeMixEndColorAndTime;
    float4 emissive;
    float4 ambient;
    float4 lightRotations[16];
    float4 lightColors[16];
    float4 lightPositions[16];
    float4 lightCones[16];
    uint4 lightKinds[16];
    uint4 materialKind;
    float4 diffractionGeometry;
    float4 diffractionSecondaryGeometry;
    float4 diffractionMicrostructure;
    float4 diffractionControl;
    float4 diffractionIncident;
    float4 diffractionCoating;
    float4 diffractionRoughness;
    float4 diffractionGrooveField;
    float4 diffractionGrooveVariation;
    float4 diffractionSpectral[8];
    float4 diffractionSpectralZ[8];
    float4 diffractionIncidentDirectionAndIntensity;
    uint4 diffractionPathKindAndBounce;
    float4 diffractionFoilField[4];
    float4 diffractionOccupancyRectangles[5];
    uint4 diffractionSpatialCounts;
    uint4 diffractionEvaluationSchedule;
    float4 noteInstanceTransforms[128];
    uint4 geometryInstanceControl;
};
struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float3 baseColor;
    float3 litBase;
    float3 emissive;
    float3 worldPosition;
    float3 worldNormal;
    float3 worldTangent;
    float bitangentSign;
    float2 metallicRoughness;
    float linearDepth;
    float opacity;
};

constant float pi = 3.14159265358979323846f;
constant float degreesToRadians = 0.01745329251994329577f;
constant uint materialDiffractionReflective = 1u;
constant int profileBinaryRectangular = 1;
constant int profileSinusoidal = 2;
constant int profileBlazedSawtooth = 3;
constant int coatingUncoated = 1;
constant int coatingIncoherentDielectric = 2;

float3 rotateQuaternion(float4 q, float3 value) {
    return value + 2.0f * cross(q.xyz, cross(q.xyz, value) + q.w * value);
}

float interfaceReflectance(float2 first, float2 second) {
    const float2 difference = first - second;
    const float2 sum = first + second;
    return clamp(dot(difference, difference) / dot(sum, sum), 0.0f, 1.0f);
}

float spectralReflectance(float wavelength, int coatingModel,
                          float substrateN, float substrateK,
                          constant Uniforms& u) {
    if (coatingModel == coatingUncoated)
        return interfaceReflectance(float2(1.0f, 0.0f),
                                    float2(substrateN, substrateK));
    if (coatingModel != coatingIncoherentDielectric)
        return -1.0f;
    const float2 layer = float2(u.diffractionCoating.y, u.diffractionCoating.z);
    const float r01 = interfaceReflectance(float2(1.0f, 0.0f), layer);
    const float r12 = interfaceReflectance(layer, float2(substrateN, substrateK));
    const float roundTrip = exp(-8.0f * pi * u.diffractionCoating.z
                                * u.diffractionCoating.x / wavelength);
    const float denominator = 1.0f - r01 * r12 * roundTrip;
    return clamp(r01 + (1.0f - r01) * (1.0f - r01)
                    * r12 * roundTrip / denominator, 0.0f, 1.0f);
}

float2 complexMultiply(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

float2 complexExponentialIntegral(float frequency, float begin, float end) {
    if (abs(frequency) < 1.0e-6f)
        return float2(end - begin, 0.0f);
    const float beginPhase = frequency * begin;
    const float endPhase = frequency * end;
    return float2((sin(endPhase) - sin(beginPhase)) / frequency,
                  -(cos(endPhase) - cos(beginPhase)) / frequency);
}

float besselJ(int order, float argument) {
    const int magnitude = order < 0 ? -order : order;
    const float halfArgument = 0.5f * argument;
    float term = 1.0f;
    for (int factor = 1; factor <= 8; ++factor)
        if (factor <= magnitude)
            term *= halfArgument / float(factor);
    float sum = term;
    for (int seriesIndex = 1; seriesIndex <= 24; ++seriesIndex) {
        term *= -(halfArgument * halfArgument)
              / (float(seriesIndex) * float(magnitude + seriesIndex));
        sum += term;
    }
    return sum;
}

float profileEfficiency(int profile, int signedOrder, float duty,
                        float phase, float2 terracePhase) {
    if (profile == profileBinaryRectangular) {
        if (signedOrder == 0) {
            const float2 amplitude = float2(1.0f - duty, 0.0f) + duty * terracePhase;
            return dot(amplitude, amplitude);
        }
        const float order = float(signedOrder);
        const float aperture = sin(pi * order * duty) / (pi * order);
        const float rotation = -pi * order * duty;
        const float2 amplitude = aperture * complexMultiply(
            terracePhase - float2(1.0f, 0.0f),
            float2(cos(rotation), sin(rotation)));
        return dot(amplitude, amplitude);
    }
    if (profile == profileSinusoidal) {
        const float amplitude = besselJ(signedOrder, 0.5f * phase);
        return amplitude * amplitude;
    }
    if (profile == profileBlazedSawtooth) {
        const float orderFrequency = -2.0f * pi * float(signedOrder);
        const float2 ramp = complexExponentialIntegral(
            phase / duty + orderFrequency, 0.0f, duty);
        const float2 land = complexExponentialIntegral(orderFrequency, duty, 1.0f);
        const float2 amplitude = ramp + land;
        return dot(amplitude, amplitude);
    }
    return 0.0f;
}

float diffractionOrderContribution(int primaryOrder,
                                   int secondaryOrder,
                                   bool crossedTwoDimensional,
                                   float wavelength,
                                   float3 incident,
                                   float3 outgoing,
                                   int profile,
                                   float duty,
                                   float sigma,
                                   float4 localGeometry,
                                   float4 localSecondaryGeometry,
                                   constant Uniforms& u) {
    const int firstOrder = int(round(u.diffractionMicrostructure.w));
    const int lastOrder = int(u.diffractionControl.x + 0.5f);
    const int primaryMagnitude = primaryOrder < 0 ? -primaryOrder : primaryOrder;
    const int secondaryMagnitude = secondaryOrder < 0 ? -secondaryOrder : secondaryOrder;
    if ((primaryOrder != 0
            && (primaryMagnitude < firstOrder || primaryMagnitude > lastOrder))
        || (crossedTwoDimensional && secondaryOrder != 0
            && (secondaryMagnitude < firstOrder || secondaryMagnitude > lastOrder)))
        return 0.0f;

    float2 expectedTangent = -incident.xy
        + float(primaryOrder) * wavelength / localGeometry.z
            * localGeometry.xy;
    if (crossedTwoDimensional)
        expectedTangent += float(secondaryOrder) * wavelength
            / localSecondaryGeometry.z
            * localSecondaryGeometry.xy;
    if (dot(expectedTangent, expectedTangent) >= 1.0f)
        return 0.0f;

    const float expectedOutgoingCosine = sqrt(max(
        0.0f, 1.0f - dot(expectedTangent, expectedTangent)));
    const float2 deviation = outgoing.xy - expectedTangent;
    const float gaussianNormalization = 1.0f / (2.0f * pi * sigma * sigma);
    const float lobe = gaussianNormalization
        * exp(-dot(deviation, deviation) / (2.0f * sigma * sigma));
    const float roughnessArgument = 2.0f * pi * u.diffractionRoughness.x
        * (incident.z + expectedOutgoingCosine) / wavelength;
    const float coherentFactor = exp(-(roughnessArgument * roughnessArgument));
    const float phase = 2.0f * pi * u.diffractionGeometry.w
        * (incident.z + expectedOutgoingCosine) / wavelength;
    const float2 terracePhase = float2(cos(phase), sin(phase));
    float efficiency = coherentFactor * profileEfficiency(
        profile, primaryOrder, duty, phase, terracePhase);
    if (crossedTwoDimensional)
        efficiency *= profileEfficiency(
            profile, secondaryOrder, duty, phase, terracePhase);
    efficiency *= expectedOutgoingCosine / incident.z;
    return efficiency * lobe;
}

float3 evaluateDiffraction(VertexOut in, constant Uniforms& u) {
    const float3 normal = normalize(in.worldNormal);
    const float3 dpdx = dfdx(in.worldPosition);
    const float3 dpdy = dfdy(in.worldPosition);
    const float2 duvdx = dfdx(in.uv);
    const float2 duvdy = dfdy(in.uv);
    const float determinant = duvdx.x * duvdy.y - duvdx.y * duvdy.x;
    if (abs(determinant) <= 1.0e-8f)
        return float3(0.0f);
    float3 tangent = normalize((dpdx * duvdy.y - dpdy * duvdx.y) / determinant);
    tangent = normalize(tangent - normal * dot(normal, tangent));
    float3 bitangent = normalize(cross(normal, tangent));
    if (dot(bitangent, (-dpdx * duvdy.x + dpdy * duvdx.x) / determinant) < 0.0f)
        bitangent = -bitangent;

    const float3 toLightWorld = normalize(rotateQuaternion(
        u.lightRotations[0], float3(0.0f, 0.0f, 1.0f)));
    const float3 toViewWorld = normalize(u.cameraTranslation.xyz - in.worldPosition);
    const float3 incident = u.diffractionIncidentDirectionAndIntensity.xyz;
    const float3 outgoing = float3(dot(toViewWorld, tangent),
                                  dot(toViewWorld, bitangent),
                                  dot(toViewWorld, normal));
    if (incident.z <= 0.0f || outgoing.z <= 0.0f)
        return float3(0.0f);

    const float duty = u.diffractionMicrostructure.x;
    const float substrateN = u.diffractionMicrostructure.y;
    const float substrateK = u.diffractionMicrostructure.z;
    const int profile = int(u.diffractionControl.y + 0.5f);
    const int coatingModel = int(u.diffractionControl.w + 0.5f);
    const int firstOrder = int(round(u.diffractionMicrostructure.w));
    const int lastOrder = int(u.diffractionControl.x + 0.5f);
    const bool crossedTwoDimensional
        = int(u.diffractionSecondaryGeometry.w + 0.5f) == 2;
    const float sigma = 2.0f * u.diffractionRoughness.y;
    float4 localGeometry = u.diffractionGeometry;
    float4 localSecondaryGeometry = u.diffractionSecondaryGeometry;
    if (u.diffractionSpatialCounts.x > 0u) {
        const float4 lower = mix(u.diffractionFoilField[0], u.diffractionFoilField[1], clamp(in.uv.x, 0.0f, 1.0f));
        const float4 upper = mix(u.diffractionFoilField[2], u.diffractionFoilField[3], clamp(in.uv.x, 0.0f, 1.0f));
        const float4 local = mix(lower, upper, clamp(in.uv.y, 0.0f, 1.0f));
        localGeometry.xyz = float3(normalize(local.xy), local.z);
    }
    const int fieldMode = int(u.diffractionGrooveField.x + 0.5f);
    if (u.diffractionSpatialCounts.x == 0u && fieldMode != 1) {
        const float2 offset = in.uv - u.diffractionGrooveField.yz;
        const float coordinate = fieldMode == 2
            ? dot(offset, u.diffractionGrooveVariation.xy) : length(offset);
        localGeometry.z += coordinate * u.diffractionGrooveVariation.z;
        localSecondaryGeometry.z += coordinate * u.diffractionGrooveVariation.w;
        const float angle = coordinate * u.diffractionGrooveField.w * degreesToRadians;
        const float2x2 rotation = float2x2(
            float2(cos(angle), sin(angle)), float2(-sin(angle), cos(angle)));
        localGeometry.xy = rotation * localGeometry.xy;
        localSecondaryGeometry.xy = rotation * localSecondaryGeometry.xy;
    }

    float3 xyz = float3(0.0f);
    float referenceWhiteY = 0.0f;

    for (int wavelengthIndex = 0; wavelengthIndex < 8; ++wavelengthIndex) {
        const float4 spectral = u.diffractionSpectral[wavelengthIndex];
        const float4 spectralZ = u.diffractionSpectralZ[wavelengthIndex];
        const float wavelength = spectral.x;
        const float quadrature = spectralZ.y;
        const float reflectance = spectralReflectance(
            wavelength, coatingModel, substrateN, substrateK, u);
        float wavelengthRadiance = diffractionOrderContribution(
            0, 0, crossedTwoDimensional, wavelength,
            incident, outgoing, profile, duty, sigma,
                        localGeometry, localSecondaryGeometry, u);
        if (crossedTwoDimensional) {
            for (int primaryOrder = -lastOrder; primaryOrder <= -firstOrder; ++primaryOrder) {
                for (int secondaryOrder = -lastOrder; secondaryOrder <= -firstOrder; ++secondaryOrder)
                    wavelengthRadiance += diffractionOrderContribution(
                        primaryOrder, secondaryOrder, true, wavelength,
                        incident, outgoing, profile, duty, sigma,
                        localGeometry, localSecondaryGeometry, u);
                for (int secondaryOrder = firstOrder; secondaryOrder <= lastOrder; ++secondaryOrder)
                    wavelengthRadiance += diffractionOrderContribution(
                        primaryOrder, secondaryOrder, true, wavelength,
                        incident, outgoing, profile, duty, sigma,
                        localGeometry, localSecondaryGeometry, u);
            }
            for (int primaryOrder = firstOrder; primaryOrder <= lastOrder; ++primaryOrder) {
                for (int secondaryOrder = -lastOrder; secondaryOrder <= -firstOrder; ++secondaryOrder)
                    wavelengthRadiance += diffractionOrderContribution(
                        primaryOrder, secondaryOrder, true, wavelength,
                        incident, outgoing, profile, duty, sigma,
                        localGeometry, localSecondaryGeometry, u);
                for (int secondaryOrder = firstOrder; secondaryOrder <= lastOrder; ++secondaryOrder)
                    wavelengthRadiance += diffractionOrderContribution(
                        primaryOrder, secondaryOrder, true, wavelength,
                        incident, outgoing, profile, duty, sigma,
                        localGeometry, localSecondaryGeometry, u);
            }
            for (int primaryOrder = -lastOrder; primaryOrder <= -firstOrder; ++primaryOrder)
                wavelengthRadiance += diffractionOrderContribution(
                    primaryOrder, 0, true, wavelength,
                    incident, outgoing, profile, duty, sigma,
                        localGeometry, localSecondaryGeometry, u);
            for (int primaryOrder = firstOrder; primaryOrder <= lastOrder; ++primaryOrder)
                wavelengthRadiance += diffractionOrderContribution(
                    primaryOrder, 0, true, wavelength,
                    incident, outgoing, profile, duty, sigma,
                        localGeometry, localSecondaryGeometry, u);
            for (int secondaryOrder = -lastOrder; secondaryOrder <= -firstOrder; ++secondaryOrder)
                wavelengthRadiance += diffractionOrderContribution(
                    0, secondaryOrder, true, wavelength,
                    incident, outgoing, profile, duty, sigma,
                        localGeometry, localSecondaryGeometry, u);
            for (int secondaryOrder = firstOrder; secondaryOrder <= lastOrder; ++secondaryOrder)
                wavelengthRadiance += diffractionOrderContribution(
                    0, secondaryOrder, true, wavelength,
                    incident, outgoing, profile, duty, sigma,
                        localGeometry, localSecondaryGeometry, u);
        } else {
            for (int primaryOrder = -lastOrder; primaryOrder <= -firstOrder; ++primaryOrder)
                wavelengthRadiance += diffractionOrderContribution(
                    primaryOrder, 0, false, wavelength,
                    incident, outgoing, profile, duty, sigma,
                        localGeometry, localSecondaryGeometry, u);
            for (int primaryOrder = firstOrder; primaryOrder <= lastOrder; ++primaryOrder)
                wavelengthRadiance += diffractionOrderContribution(
                    primaryOrder, 0, false, wavelength,
                    incident, outgoing, profile, duty, sigma,
                        localGeometry, localSecondaryGeometry, u);
        }
        const float pathTransport = u.diffractionPathKindAndBounce.x == 3u
            ? 1.0f / float(1u + u.diffractionPathKindAndBounce.y) : 1.0f;
        const float energy = u.diffractionIncidentDirectionAndIntensity.w
                           * spectral.y * pathTransport * quadrature * reflectance
                           * wavelengthRadiance * incident.z;
        xyz += energy * float3(spectral.z, spectral.w, spectralZ.x);
        referenceWhiteY += quadrature * spectral.w;
    }
    xyz /= max(referenceWhiteY, 1.0e-8f);
    return float3(3.2406f * xyz.x - 1.5372f * xyz.y - 0.4986f * xyz.z,
                  -0.9689f * xyz.x + 1.8758f * xyz.y + 0.0415f * xyz.z,
                  0.0557f * xyz.x - 0.2040f * xyz.y + 1.0570f * xyz.z);
}

struct FragmentOut {
    float4 color [[color(0)]];
    float linearDepth [[color(1)]];
};
fragment FragmentOut _main(VertexOut in [[stage_in]],
                           texture2d<float> baseTexture [[texture(0)]],
                           texture2d<float> metallicRoughnessTexture [[texture(1)]],
                           texture2d<float> normalTexture [[texture(2)]],
                           texture2d<float> occlusionTexture [[texture(3)]],
                           texture2d<float> emissiveTexture [[texture(4)]],
                           sampler baseSampler [[sampler(0)]],
                           sampler metallicRoughnessSampler [[sampler(1)]],
                           sampler normalSampler [[sampler(2)]],
                           sampler occlusionSampler [[sampler(3)]],
                           sampler emissiveSampler [[sampler(4)]],
                           constant Uniforms& u [[buffer(0)]]) {
    FragmentOut out;
    out.linearDepth = in.linearDepth;
    if (u.materialKind.x == materialDiffractionReflective)
    {
        const float coverage = u.diffractionSpatialCounts.y == 0u
            ? 1.0f : as_type<float>(u.diffractionPathKindAndBounce.w);
        float occupancy = u.diffractionSpatialCounts.y == 0u ? 1.0f : 0.0f;
        for (uint i = 0u; i < u.diffractionSpatialCounts.y; ++i) {
            const float4 r = u.diffractionOccupancyRectangles[i];
            occupancy = max(occupancy, in.uv.x >= r.x && in.uv.y >= r.y
                && in.uv.x <= r.z && in.uv.y <= r.w ? 1.0f : 0.0f);
        }
        if (u.diffractionSpatialCounts.x > 0u) {
            const float4 lower = mix(u.diffractionFoilField[0], u.diffractionFoilField[1], clamp(in.uv.x, 0.0f, 1.0f));
            const float4 upper = mix(u.diffractionFoilField[2], u.diffractionFoilField[3], clamp(in.uv.x, 0.0f, 1.0f));
            occupancy *= mix(lower.w, upper.w, clamp(in.uv.y, 0.0f, 1.0f));
        }
        const uint evaluationIndex = u.diffractionEvaluationSchedule.w
            * (u.diffractionEvaluationSchedule.x * u.diffractionEvaluationSchedule.y)
            + uint(in.position.y) * u.diffractionEvaluationSchedule.x
            + uint(in.position.x);
        const float3 diffraction = (u.diffractionSpatialCounts.x == 0u
                || evaluationIndex < u.diffractionEvaluationSchedule.z)
            ? max(evaluateDiffraction(in, u), float3(0.0f)) : in.litBase;
        out.color = float4(mix(in.litBase,
            diffraction,
            clamp(coverage * occupancy, 0.0f, 1.0f)), 1.0f);
    }
    else
    {
        const float4 baseTexel = baseTexture.sample(baseSampler, in.uv);
        const float alpha = in.opacity * baseTexel.a;
        if (u.materialKind.z == 1u && alpha < as_type<float>(u.materialKind.w))
            discard_fragment();
        const float4 metallicRoughnessTexel
            = metallicRoughnessTexture.sample(metallicRoughnessSampler, in.uv);
        const float roughness = in.metallicRoughness.y
            * (u.materialKind.y & 1u ? metallicRoughnessTexel.g : 1.0f);
        const float metallic = in.metallicRoughness.x
            * (u.materialKind.y & 1u ? metallicRoughnessTexel.b : 1.0f);
        float3 normal = normalize(in.worldNormal);
        if (u.materialKind.y & 2u) {
            const float3 tangent = normalize(in.worldTangent
                - normal * dot(normal, in.worldTangent));
            const float3 bitangent = normalize(cross(normal, tangent)) * in.bitangentSign;
            float3 tangentNormal = normalTexture.sample(normalSampler, in.uv).xyz * 2.0f - 1.0f;
            tangentNormal.xy *= u.materialParams.z;
            normal = normalize(float3x3(tangent, bitangent, normal) * tangentNormal);
        }
        float3 lighting = u.ambient.xyz;
        for (uint lightIndex = 0u; lightIndex < 16u; ++lightIndex) {
            if (lightIndex >= uint(u.ambient.w)) break;
            const uint kind = u.lightKinds[lightIndex].x;
            if (kind == 2u) {
                lighting += u.lightColors[lightIndex].rgb * u.lightColors[lightIndex].a;
                continue;
            }
            const float3 delta = u.lightPositions[lightIndex].xyz - in.worldPosition;
            const float3 toLight = kind == 0u
                ? rotateQuaternion(u.lightRotations[lightIndex], float3(0.0f, 0.0f, 1.0f))
                : normalize(delta);
            float attenuation = 1.0f;
            if (kind == 1u || kind == 3u) {
                const float lightDistance = length(delta);
                attenuation = 1.0f / max(lightDistance * lightDistance, 1.0f);
                const float range = u.lightPositions[lightIndex].w;
                if (range > 0.0f)
                    attenuation *= pow(clamp(1.0f - lightDistance / range, 0.0f, 1.0f), 2.0f);
                if (kind == 3u) {
                    const float3 spotDirection = rotateQuaternion(
                        u.lightRotations[lightIndex], float3(0.0f, 0.0f, -1.0f));
                    const float coneCosine = dot(spotDirection, normalize(
                        in.worldPosition - u.lightPositions[lightIndex].xyz));
                    attenuation *= smoothstep(u.lightCones[lightIndex].y,
                                               u.lightCones[lightIndex].x, coneCosine);
                }
            }
            lighting += u.lightColors[lightIndex].rgb * u.lightColors[lightIndex].a
                * max(dot(normal, normalize(toLight)), 0.0f) * attenuation;
        }
        const float diffuseWeight = (1.0f - metallic) * (1.0f - 0.5f * roughness);
        const float specularWeight = mix(0.04f, 1.0f, metallic) * (1.0f - roughness);
        const float occlusion = u.materialKind.y & 4u
            ? occlusionTexture.sample(occlusionSampler, in.uv).r : 1.0f;
        const float3 emissive = in.emissive * (u.materialKind.y & 8u
            ? emissiveTexture.sample(emissiveSampler, in.uv).rgb : float3(1.0f));
        out.color = float4(in.baseColor * baseTexel.rgb * lighting
            * (diffuseWeight + specularWeight) * occlusion + emissive, alpha);

    }
    return out;
}
)metal";

[[maybe_unused]] const char* kDeformationComputeShader = R"metal(
#include <metal_stdlib>
using namespace metal;
struct SceneVertex {
    packed_float3 position;
    packed_float3 normal;
    packed_float2 uv;
};
struct Params {
    uint vertexCount;
    uint jointCount;
    uint morphTargetCount;
    uint padding;
};
kernel void _main(const device SceneVertex* source [[buffer(8)]],
                  device SceneVertex* destination [[buffer(9)]],
                  const device uint* jointIndices [[buffer(10)]],
                  const device float* jointWeights [[buffer(11)]],
                  const device float* morphPositions [[buffer(12)]],
                  const device float* morphNormals [[buffer(13)]],
                  const device float4x4* jointPalette [[buffer(14)]],
                  const device float* morphWeights [[buffer(15)]],
                  constant Params& params [[buffer(0)]],
                  uint vertexIndex [[thread_position_in_grid]]) {
    if (vertexIndex >= params.vertexCount) return;
    const SceneVertex input = source[vertexIndex];
    float3 position = float3(input.position);
    float3 normal = float3(input.normal);
    for (uint target = 0; target < params.morphTargetCount; ++target) {
        const uint scalar = (target * params.vertexCount + vertexIndex) * 3u;
        const float weight = morphWeights[target];
        position += weight * float3(morphPositions[scalar],
                                    morphPositions[scalar + 1u],
                                    morphPositions[scalar + 2u]);
        normal += weight * float3(morphNormals[scalar],
                                  morphNormals[scalar + 1u],
                                  morphNormals[scalar + 2u]);
    }
    if (params.jointCount > 0u) {
        float3 skinnedPosition = float3(0.0f);
        float3 skinnedNormal = float3(0.0f);
        for (uint component = 0; component < 4u; ++component) {
            const uint influence = vertexIndex * 4u + component;
            const float weight = jointWeights[influence];
            const uint joint = jointIndices[influence];
            if (weight == 0.0f) continue;
            const float4x4 palette = jointPalette[joint];
            skinnedPosition += weight * (palette * float4(position, 1.0f)).xyz;
            skinnedNormal += weight * (palette * float4(normal, 0.0f)).xyz;
        }
        position = skinnedPosition;
        normal = skinnedNormal;
    }
    SceneVertex output;
    output.position = packed_float3(position);
    output.normal = packed_float3(normalize(normal));
    output.uv = input.uv;
    destination[vertexIndex] = output;
}
)metal";

[[maybe_unused]] const char* kDeformationVertexShader = R"metal(
#include <metal_stdlib>
using namespace metal;
struct SceneVertex {
    packed_float3 position;
    packed_float3 normal;
    packed_float2 uv;
};
struct Uniforms {
    float4 objectRotation;
    float4 objectTranslation;
    float4 objectScale;
    float4 cameraRotation;
    float4 cameraTranslation;
    float4 projection;
    float4 baseColor;
    float4 emissive;
    float4 ambient;
    float4 lightRotation;
    float4 lightColorIntensity;
};
struct VertexOut {
    float4 position [[position]];
    float3 color [[user(locn0)]];
};
float3 rotateQuaternion(float4 q, float3 value) {
    return value + 2.0f * cross(q.xyz, cross(q.xyz, value) + q.w * value);
}
vertex VertexOut _main(const device SceneVertex* vertices [[buffer(8)]],
                       constant Uniforms& u [[buffer(0)]],
                       uint vertex [[vertex_id]]) {
    const SceneVertex input = vertices[vertex];
    const float3 worldPosition = rotateQuaternion(
        u.objectRotation, float3(input.position) * u.objectScale.xyz)
        + u.objectTranslation.xyz;
    const float3 worldNormal = normalize(rotateQuaternion(
        u.objectRotation, float3(input.normal) / u.objectScale.xyz));
    const float4 inverseCamera = float4(-u.cameraRotation.xyz, u.cameraRotation.w);
    const float3 cameraPosition = rotateQuaternion(
        inverseCamera, worldPosition - u.cameraTranslation.xyz);
    const float distance = -cameraPosition.z;
    VertexOut out;
    out.position = float4(cameraPosition.x / (u.projection.x * u.projection.y),
                          cameraPosition.y / u.projection.x,
                          distance * u.projection.w / (u.projection.w - u.projection.z)
                              - u.projection.z * u.projection.w
                                  / (u.projection.w - u.projection.z),
                          distance);
    const float3 toLight = rotateQuaternion(u.lightRotation, float3(0.0f, 0.0f, 1.0f));
    const float diffuse = max(dot(worldNormal, normalize(toLight)), 0.0f);
    const float3 lighting = u.ambient.xyz
        + u.lightColorIntensity.xyz * u.lightColorIntensity.w * diffuse;
    out.color = u.baseColor.xyz * lighting + u.emissive.xyz;
    return out;
}
)metal";

[[maybe_unused]] const char* kDeformationFragmentShader = R"metal(
#include <metal_stdlib>
using namespace metal;
struct VertexOut {
    float4 position [[position]];
    float3 color [[user(locn0)]];
};
fragment float4 _main(VertexOut input [[stage_in]]) {
    return float4(input.color, 1.0f);
}
)metal";

void storeVec3 (float (&destination)[4],
                const HarmonicMIDI::grid::SceneVec3& source,
                float fourth = 0.0f) noexcept
{
    destination[0] = source.x;
    destination[1] = source.y;
    destination[2] = source.z;
    destination[3] = fourth;
}

void storeQuaternion (float (&destination)[4],
                      const HarmonicMIDI::grid::SceneQuaternion& source) noexcept
{
    destination[0] = source.x;
    destination[1] = source.y;
    destination[2] = source.z;
    destination[3] = source.w;
}

std::array<float, 16> metalFixtureTransformMatrix (
    const HarmonicMIDI::grid::SceneTransform3D& transform) noexcept
{
    const auto x = transform.rotation.x, y = transform.rotation.y;
    const auto z = transform.rotation.z, w = transform.rotation.w;
    const auto sx = transform.scale.x, sy = transform.scale.y, sz = transform.scale.z;
    return {
        (1.0f - 2.0f * (y * y + z * z)) * sx,
        (2.0f * (x * y + w * z)) * sx,
        (2.0f * (x * z - w * y)) * sx, 0.0f,
        (2.0f * (x * y - w * z)) * sy,
        (1.0f - 2.0f * (x * x + z * z)) * sy,
        (2.0f * (y * z + w * x)) * sy, 0.0f,
        (2.0f * (x * z + w * y)) * sz,
        (2.0f * (y * z - w * x)) * sz,
        (1.0f - 2.0f * (x * x + y * y)) * sz, 0.0f,
        transform.translation.x, transform.translation.y, transform.translation.z, 1.0f
    };
}

std::array<float, 16> multiplyMetalFixtureMatrices (
    const std::array<float, 16>& left, const std::array<float, 16>& right) noexcept
{
    std::array<float, 16> result {};
    for (std::size_t column = 0; column < 4; ++column)
        for (std::size_t row = 0; row < 4; ++row)
            for (std::size_t inner = 0; inner < 4; ++inner)
                result[column * 4 + row] += left[inner * 4 + row]
                    * right[column * 4 + inner];
    return result;
}

std::array<float, 16> metalFixtureWorldMatrix (
    const HarmonicMIDI::grid::Visual3DScene& scene,
    const HarmonicMIDI::grid::SceneObjectRecord& object) noexcept
{
    auto result = metalFixtureTransformMatrix (object.transform);
    auto parent = object.parent;
    for (std::size_t depth = 0; parent.isValid() && depth < scene.objectCount; ++depth)
    {
        const auto* parentObject = HarmonicMIDI::grid::visual3d_detail::findById (
            scene.objects, scene.objectCount, parent);
        if (parentObject == nullptr) break;
        result = multiplyMetalFixtureMatrices (
            metalFixtureTransformMatrix (parentObject->transform), result);
        parent = parentObject->parent;
    }
    return result;
}

float metalFixtureDeterminant3x3 (const std::array<float, 16>& matrix) noexcept
{
    return matrix[0] * (matrix[5] * matrix[10] - matrix[9] * matrix[6])
         - matrix[4] * (matrix[1] * matrix[10] - matrix[9] * matrix[2])
         + matrix[8] * (matrix[1] * matrix[6] - matrix[5] * matrix[2]);
}

} // namespace

namespace arbitgpu::sokolmetal
{
std::mutex& mutex() noexcept
{
    return sokolMutex();
}

bool ensure() noexcept
{
    return ensureSokolMetal();
}

void* device() noexcept
{
    return (__bridge void*) gMetalDevice;
}

const std::string& error() noexcept
{
    return gSokolError;
}

std::string log()
{
    return lastSokolLog();
}
} // namespace arbitgpu::sokolmetal

namespace arbitgpu
{

namespace
{
sg_pixel_format sokolPixelFormat (renderpassoutput::PixelFormat format) noexcept
{
    using renderpassoutput::PixelFormat;
    switch (format)
    {
        case PixelFormat::R8Unorm: return SG_PIXELFORMAT_R8;
        case PixelFormat::RG16Float: return SG_PIXELFORMAT_RG16F;
        case PixelFormat::RGBA16Float: return SG_PIXELFORMAT_RGBA16F;
        case PixelFormat::R32Float: return SG_PIXELFORMAT_R32F;
        case PixelFormat::R32Uint: return SG_PIXELFORMAT_R32UI;
        case PixelFormat::Invalid: break;
    }
    return SG_PIXELFORMAT_NONE;
}

RenderPassOutputCapabilities metalRenderPassOutputCapabilitiesUnlocked()
{
    RenderPassOutputCapabilities result;
    if (! ensureSokolMetal())
    {
        result.error = gSokolError;
        return result;
    }
    if (sg_query_backend() != SG_BACKEND_METAL_MACOS)
    {
        result.error = "sokol_gfx did not select the macOS Metal backend";
        return result;
    }

    const auto nativeLimits = sg_query_limits();
    if (nativeLimits.max_color_attachments <= 0 || nativeLimits.max_image_size_2d <= 0)
    {
        result.error = "Metal reported no render-pass output attachment capacity";
        return result;
    }

    const auto maximumDimension = static_cast<std::uint32_t> (
        std::min (nativeLimits.max_image_size_2d,
                  static_cast<int> (renderpassoutput::kMaximumWidth)));
    result.limits.attachments = std::min (
        static_cast<std::size_t> (nativeLimits.max_color_attachments),
        renderpassoutput::kMaximumAttachments);
    result.limits.width = maximumDimension;
    result.limits.height = std::min (
        maximumDimension, renderpassoutput::kMaximumHeight);
    result.limits.pixels = std::min (
        static_cast<std::uint64_t> (result.limits.width) * result.limits.height,
        renderpassoutput::kMaximumPixels);
    result.limits.totalBytes = renderpassoutput::kMaximumTotalBytes;

    for (const auto output : renderpassoutput::kOutputs)
    {
        const auto requirements = renderpassoutput::requirements (output);
        const auto format = sokolPixelFormat (requirements.format);
        if (format == SG_PIXELFORMAT_NONE)
            continue;
        const auto formatInfo = sg_query_pixelformat (format);
        const auto exactBytes = renderpassoutput::bytesPerPixel (requirements.format);
        result.supportedOutputs[static_cast<std::size_t> (output)] =
            formatInfo.render && formatInfo.sample
            && formatInfo.bytes_per_pixel == exactBytes;
    }

    result.available = std::any_of (
        result.supportedOutputs.begin(), result.supportedOutputs.end(),
        [] (bool supported) { return supported; });
    if (! result.available)
        result.error = "Metal supports none of the exact render-pass output formats";
    return result;
}

struct MetalRenderPassOutputAttachment
{
    renderpassoutput::Output output = renderpassoutput::Output::Count;
    sg_image image = {};
    sg_view attachmentView = {};
    sg_view textureView = {};
};

struct MetalRenderPassOutputAllocation
{
    std::vector<MetalRenderPassOutputAttachment> attachments;
};

void destroyRenderPassOutputAllocationUnlocked (
    MetalRenderPassOutputAllocation& allocation) noexcept
{
    for (auto found = allocation.attachments.rbegin();
         found != allocation.attachments.rend(); ++found)
    {
        if (found->textureView.id != 0)
            sg_destroy_view (found->textureView);
        if (found->attachmentView.id != 0)
            sg_destroy_view (found->attachmentView);
        if (found->image.id != 0)
            sg_destroy_image (found->image);
        *found = {};
    }
    allocation.attachments.clear();
}

class MetalRenderPassOutputBackend final : public RenderPassOutputBackend
{
public:
    RenderPassOutputCapabilities renderPassOutputCapabilities() const override
    {
        std::lock_guard<std::mutex> lock (sokolMutex());
        @autoreleasepool
        {
            return metalRenderPassOutputCapabilitiesUnlocked();
        }
    }

    RenderPassOutputAdmission admitRenderPassOutputs (
        const renderpassoutput::AdmittedOutputs& outputs) override
    {
        RenderPassOutputAdmission result;
        std::lock_guard<std::mutex> lock (sokolMutex());
        @autoreleasepool
        {
            const auto capabilities = metalRenderPassOutputCapabilitiesUnlocked();
            if (! capabilities.available)
            {
                result.error = capabilities.error;
                return result;
            }
            if (outputs.attachments().empty())
            {
                result.error = "Metal render-pass output admission requires an attachment";
                return result;
            }
            if (! admitRenderPassOutputFrameMemory (
                    outputs, capabilities.limits.totalBytes,
                    result.frameMemory, result.error))
                return result;

            const auto extent = outputs.extent();
            const auto pixels = static_cast<std::uint64_t> (extent.width) * extent.height;
            if (outputs.attachments().size() > capabilities.limits.attachments
                || extent.width > capabilities.limits.width
                || extent.height > capabilities.limits.height
                || pixels > capabilities.limits.pixels
                || outputs.totalByteCount() > capabilities.limits.totalBytes)
            {
                result.error = "Metal render-pass output request exceeds backend limits";
                return result;
            }

            MetalRenderPassOutputAllocation allocation;
            allocation.attachments.reserve (outputs.attachments().size());
            result.resources.reserve (outputs.attachments().size());
            for (const auto& attachment : outputs.attachments())
            {
                if (! capabilities.supports (attachment.output))
                {
                    result.error = "Metal does not support render-pass output ";
                    result.error += renderpassoutput::token (attachment.output);
                    destroyRenderPassOutputAllocationUnlocked (allocation);
                    result.resources.clear();
                    return result;
                }

                const auto format = sokolPixelFormat (attachment.format);
                MetalRenderPassOutputAttachment owned;
                owned.output = attachment.output;

                sg_image_desc imageDesc = {};
                imageDesc.usage.color_attachment = true;
                imageDesc.width = static_cast<int> (attachment.extent.width);
                imageDesc.height = static_cast<int> (attachment.extent.height);
                imageDesc.pixel_format = format;
                imageDesc.sample_count = 1;
                imageDesc.label = "arbit-metal-render-pass-output";
                owned.image = sg_make_image (&imageDesc);
                bool valid = owned.image.id != 0
                    && resourceValid (sg_query_image_state (owned.image));

                if (valid)
                {
                    sg_view_desc attachmentViewDesc = {};
                    attachmentViewDesc.color_attachment.image = owned.image;
                    owned.attachmentView = sg_make_view (&attachmentViewDesc);
                    valid = owned.attachmentView.id != 0
                        && resourceValid (sg_query_view_state (owned.attachmentView));
                }

                if (valid)
                {
                    sg_view_desc textureViewDesc = {};
                    textureViewDesc.texture.image = owned.image;
                    owned.textureView = sg_make_view (&textureViewDesc);
                    valid = owned.textureView.id != 0
                        && resourceValid (sg_query_view_state (owned.textureView));
                }

                allocation.attachments.push_back (owned);
                if (! valid)
                {
                    result.error = "Metal failed to allocate render-pass output ";
                    result.error += renderpassoutput::token (attachment.output);
                    destroyRenderPassOutputAllocationUnlocked (allocation);
                    result.resources.clear();
                    return result;
                }

                result.resources.push_back ({ attachment.output,
                                              owned.image.id,
                                              owned.attachmentView.id,
                                              owned.textureView.id });
            }

            RenderPassOutputLifecycleHandle lifecycle;
            do
            {
                lifecycle = { nextLifecycle_++ };
            }
            while (! lifecycle || allocations_.find (lifecycle.value) != allocations_.end());
            allocations_.emplace (lifecycle.value, std::move (allocation));
            result.lifecycle = lifecycle;
            return result;
        }
    }

    RenderPassColorAovExecution executeColorAovClear (
        RenderPassOutputLifecycleHandle lifecycle,
        const RenderPassColorAovClear& clear) override
    {
        RenderPassColorAovExecution result;
        std::lock_guard<std::mutex> lock (sokolMutex());
        @autoreleasepool
        {
            if (! lifecycle)
            {
                result.error = "Metal Color AOV execution requires a lifecycle";
                return result;
            }
            if (! std::all_of (clear.linearRgba.begin(), clear.linearRgba.end(),
                               [] (float value) { return std::isfinite (value); }))
            {
                result.error = "Metal Color AOV clear values must be finite";
                return result;
            }

            const auto allocation = allocations_.find (lifecycle.value);
            if (allocation == allocations_.end())
            {
                result.error = "Metal Color AOV execution rejected a stale lifecycle";
                return result;
            }
            const auto color = std::find_if (
                allocation->second.attachments.begin(), allocation->second.attachments.end(),
                [] (const MetalRenderPassOutputAttachment& attachment)
                {
                    return attachment.output == renderpassoutput::Output::Color;
                });
            if (color == allocation->second.attachments.end()
                || color->attachmentView.id == 0
                || ! resourceValid (sg_query_view_state (color->attachmentView)))
            {
                result.error = "Metal Color AOV execution requires an admitted Color attachment";
                return result;
            }

            sg_pass pass = {};
            pass.attachments.colors[0] = color->attachmentView;
            pass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
            pass.action.colors[0].store_action = SG_STOREACTION_STORE;
            pass.action.colors[0].clear_value = {
                clear.linearRgba[0], clear.linearRgba[1],
                clear.linearRgba[2], clear.linearRgba[3] };
            pass.label = "arbit-metal-color-aov-clear-pass";
            sg_begin_pass (&pass);
            sg_end_pass();
            sg_commit();

            do
            {
                result.submission = nextSubmission_++;
            }
            while (result.submission == 0);
            result.submitted = true;
            return result;
        }
    }

    RenderPassMotionAovExecution executeMotionAovClear (
        RenderPassOutputLifecycleHandle lifecycle,
        const RenderPassMotionAovClear& clear) override
    {
        RenderPassMotionAovExecution result;
        std::lock_guard<std::mutex> lock (sokolMutex());
        @autoreleasepool
        {
            if (! lifecycle)
            {
                result.error = "Metal Motion AOV execution requires a lifecycle";
                return result;
            }
            if (! std::all_of (clear.pixelDisplacement.begin(), clear.pixelDisplacement.end(),
                               [] (float value) { return std::isfinite (value); }))
            {
                result.error = "Metal Motion AOV clear values must be finite";
                return result;
            }

            const auto allocation = allocations_.find (lifecycle.value);
            if (allocation == allocations_.end())
            {
                result.error = "Metal Motion AOV execution rejected a stale lifecycle";
                return result;
            }
            const auto motion = std::find_if (
                allocation->second.attachments.begin(), allocation->second.attachments.end(),
                [] (const MetalRenderPassOutputAttachment& attachment)
                {
                    return attachment.output == renderpassoutput::Output::Motion;
                });
            if (motion == allocation->second.attachments.end()
                || motion->attachmentView.id == 0
                || ! resourceValid (sg_query_view_state (motion->attachmentView)))
            {
                result.error = "Metal Motion AOV execution requires an admitted Motion attachment";
                return result;
            }

            sg_pass pass = {};
            pass.attachments.colors[0] = motion->attachmentView;
            pass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
            pass.action.colors[0].store_action = SG_STOREACTION_STORE;
            pass.action.colors[0].clear_value = {
                clear.pixelDisplacement[0], clear.pixelDisplacement[1], 0.0f, 0.0f };
            pass.label = "arbit-metal-motion-aov-clear-pass";
            sg_begin_pass (&pass);
            sg_end_pass();
            sg_commit();

            do
            {
                result.submission = nextSubmission_++;
            }
            while (result.submission == 0);
            result.submitted = true;
            return result;
        }
    }

    RenderPassSceneAovExecution executeSceneAov (
        RenderPassOutputLifecycleHandle,
        const sceneaov::Payload&) override
    {
        RenderPassSceneAovExecution result;
        result.error = "native Metal scene AOV execution is not compiled in";
        return result;
    }

    void releaseRenderPassOutputs (
        RenderPassOutputLifecycleHandle lifecycle) noexcept override
    {
        if (! lifecycle)
            return;
        std::lock_guard<std::mutex> lock (sokolMutex());
        @autoreleasepool
        {
            const auto found = allocations_.find (lifecycle.value);
            if (found == allocations_.end())
                return;
            destroyRenderPassOutputAllocationUnlocked (found->second);
            allocations_.erase (found);
        }
    }

private:
    std::uint64_t nextLifecycle_ = 1;
    std::uint64_t nextSubmission_ = 1;
    std::unordered_map<std::uint64_t, MetalRenderPassOutputAllocation> allocations_;
};

static constexpr const char* kOpticalFlowKernel = R"METAL(
#include <metal_stdlib>
using namespace metal;

inline float flowLuma(half4 rgba)
{
    return dot(float3(rgba.rgb), float3(0.2126, 0.7152, 0.0722));
}

kernel void boundedBlockFlow(texture2d<half, access::read> first [[texture(0)]],
                             texture2d<half, access::read> second [[texture(1)]],
                             texture2d<half, access::write> output [[texture(2)]],
                             uint2 gid [[thread_position_in_grid]])
{
    const int width = int(output.get_width());
    const int height = int(output.get_height());
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;

    float bestScore = INFINITY;
    int2 best = int2(0);
    int bestMagnitude = 0;
    for (int dy = -4; dy <= 4; ++dy)
    {
        for (int dx = -4; dx <= 4; ++dx)
        {
            float score = 0.0;
            for (int py = -1; py <= 1; ++py)
            {
                for (int px = -1; px <= 1; ++px)
                {
                    const int2 a = clamp(int2(gid) + int2(px, py),
                                         int2(0), int2(width - 1, height - 1));
                    const int2 b = clamp(int2(gid) + int2(px + dx, py + dy),
                                         int2(0), int2(width - 1, height - 1));
                    const float difference = flowLuma(first.read(uint2(a)))
                                           - flowLuma(second.read(uint2(b)));
                    score += difference * difference;
                }
            }
            const int magnitude = dx * dx + dy * dy;
            const bool betterTie = score == bestScore
                && (magnitude < bestMagnitude
                    || (magnitude == bestMagnitude
                        && (dy < best.y || (dy == best.y && dx < best.x))));
            if (score < bestScore || betterTie)
            {
                bestScore = score;
                best = int2(dx, dy);
                bestMagnitude = magnitude;
            }
        }
    }
    output.write(half4(half(best.x), half(best.y), half(0.0), half(0.0)), gid);
}
)METAL";

class MetalOpticalFlowExecutionBackend final : public NativeOpticalFlowExecutionBackend
{
public:
    videoopticalflow::BackendCapabilities opticalFlowCapabilities() const override
    {
        std::lock_guard<std::mutex> lock (sokolMutex());
        @autoreleasepool
        {
            return capabilitiesUnlocked();
        }
    }

    NativeOpticalFlowSubmission executeOpticalFlow (
        const videoopticalflow::AdmittedRequest& request,
        const NativeOpticalFlowInputResource& first,
        const NativeOpticalFlowInputResource& second,
        bool resetTemporalState) override
    {
        NativeOpticalFlowSubmission result;
        (void) resetTemporalState; // The bounded pair kernel retains no history.
        std::lock_guard<std::mutex> lock (sokolMutex());
        @autoreleasepool
        {
            const auto capabilities = capabilitiesUnlocked();
            if (capabilities.kind != videoopticalflow::BackendKind::NativeGpu
                || ! capabilities.supportsOpticalFlow)
            {
                result.error = "Metal optical-flow compute backend is unavailable";
                return result;
            }
            if (request.backendImplementation() != capabilities.implementation
                || request.backendImplementationRevision()
                       != capabilities.implementationRevision)
            {
                result.error = "Metal optical-flow request was admitted for another backend revision";
                return result;
            }
            const auto sameDescriptor = [] (
                const videoopticalflow::FrameDescription& left,
                const videoopticalflow::FrameDescription& right)
            {
                return left.identity.stream == right.identity.stream
                    && left.identity.content == right.identity.content
                    && left.identity.frameIndex == right.identity.frameIndex
                    && left.timestamp.ticks == right.timestamp.ticks
                    && left.timestamp.timescale == right.timestamp.timescale
                    && left.extent == right.extent
                    && left.format == right.format
                    && left.colorSpace == right.colorSpace;
            };
            const bool exactResources = first.immutable && second.immutable
                && videoopticalflow::detail::anyNonZero (first.identity)
                && videoopticalflow::detail::anyNonZero (second.identity)
                && first.identity != second.identity
                && first.helperGeneration != 0
                && first.helperGeneration == second.helperGeneration
                && first.structuralRevision != 0
                && first.structuralRevision == second.structuralRevision
                && first.imageHandle != 0 && second.imageHandle != 0
                && first.imageHandle == first.textureViewHandle
                && second.imageHandle == second.textureViewHandle
                && first.imageHandle != second.imageHandle
                && sameDescriptor (first.descriptor, request.first())
                && sameDescriptor (second.descriptor, request.second());
            if (! exactResources)
            {
                result.error = "Metal optical-flow input resource receipts do not match admission";
                return result;
            }

            id<MTLTexture> firstTexture = (__bridge id<MTLTexture>)
                reinterpret_cast<void*> (first.textureViewHandle);
            id<MTLTexture> secondTexture = (__bridge id<MTLTexture>)
                reinterpret_cast<void*> (second.textureViewHandle);
            const auto extent = request.output().extent;
            const bool texturesMatch = firstTexture != nil && secondTexture != nil
                && firstTexture.device == gMetalDevice && secondTexture.device == gMetalDevice
                && firstTexture.textureType == MTLTextureType2D
                && secondTexture.textureType == MTLTextureType2D
                && firstTexture.pixelFormat == MTLPixelFormatRGBA16Float
                && secondTexture.pixelFormat == MTLPixelFormatRGBA16Float
                && firstTexture.width == extent.width && firstTexture.height == extent.height
                && secondTexture.width == extent.width && secondTexture.height == extent.height
                && (firstTexture.usage & MTLTextureUsageShaderRead) != 0
                && (secondTexture.usage & MTLTextureUsageShaderRead) != 0;
            if (! texturesMatch)
            {
                result.error = "Metal optical-flow inputs violate exact RGBA16F resource bindings";
                return result;
            }
            if (! ensurePipelineUnlocked (result.error))
                return result;

            MTLTextureDescriptor* descriptor =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG16Float
                                                                   width:extent.width
                                                                  height:extent.height
                                                               mipmapped:NO];
            descriptor.storageMode = MTLStorageModePrivate;
            descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            id<MTLTexture> output = [gMetalDevice newTextureWithDescriptor:descriptor];
            if (output == nil)
            {
                result.error = "Metal optical-flow output allocation failed";
                return result;
            }

            id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (queue == nil || command == nil || encoder == nil)
            {
                result.error = "Metal optical-flow command allocation failed";
#if ! __has_feature(objc_arc)
                [output release];
#endif
                return result;
            }
            [encoder setComputePipelineState:pipeline_];
            [encoder setTexture:firstTexture atIndex:0];
            [encoder setTexture:secondTexture atIndex:1];
            [encoder setTexture:output atIndex:2];
            const NSUInteger width = pipeline_.threadExecutionWidth;
            const NSUInteger height = std::max<NSUInteger> (
                1, pipeline_.maxTotalThreadsPerThreadgroup / width);
            [encoder dispatchThreads:MTLSizeMake(extent.width, extent.height, 1)
                 threadsPerThreadgroup:MTLSizeMake(width, height, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
            {
                result.error = command.error != nil
                    ? std::string ([[command.error localizedDescription] UTF8String])
                    : "Metal optical-flow command did not complete";
#if ! __has_feature(objc_arc)
                [output release];
#endif
                return result;
            }

            do result.lifecycle.value = nextLifecycle_++;
            while (! result.lifecycle || outputs_.find(result.lifecycle.value) != outputs_.end());
            outputs_.emplace(result.lifecycle.value, output);
            do result.submission = nextSubmission_++;
            while (result.submission == 0);
            result.result.requestCacheKey = request.cacheKey();
            result.result.backendImplementation = request.backendImplementation();
            result.result.backendImplementationRevision =
                request.backendImplementationRevision();
            result.result.motionVectors.helperGeneration = first.helperGeneration;
            result.result.motionVectors.descriptor = request.output();
            result.result.motionVectors.byteCount = request.footprint().outputBytes;
            std::memcpy(result.result.motionVectors.identity.data(),
                        &result.lifecycle.value, sizeof(result.lifecycle.value));
            std::memcpy(result.result.motionVectors.identity.data()
                            + sizeof(result.lifecycle.value),
                        &result.submission, sizeof(result.submission));
            result.imageHandle = reinterpret_cast<std::uintptr_t> ((__bridge void*) output);
            result.textureViewHandle = result.imageHandle;
            result.completed = true;
            return result;
        }
    }

    void releaseOpticalFlowOutput (
        NativeOpticalFlowOutputLifecycleHandle lifecycle) noexcept override
    {
        if (! lifecycle) return;
        std::lock_guard<std::mutex> lock (sokolMutex());
        const auto found = outputs_.find(lifecycle.value);
        if (found == outputs_.end()) return;
#if ! __has_feature(objc_arc)
        [found->second release];
#endif
        outputs_.erase(found);
    }

private:
    videoopticalflow::BackendCapabilities capabilitiesUnlocked() const
    {
        videoopticalflow::BackendCapabilities result;
        if (! ensureSokolMetal()) return result;
        result.kind = videoopticalflow::BackendKind::NativeGpu;
        result.supportsOpticalFlow = true;
        static constexpr std::array<std::uint8_t, 16> identity = {
            'M','T','L','B','l','o','c','k','F','l','o','w','0','0','0','1' };
        result.implementation = identity;
        result.implementationRevision = 1;
        result.limits = {};
        result.temporaryBytesPerPixel = 0;
        result.fixedTemporaryBytes = 0;
        return result;
    }

    bool ensurePipelineUnlocked (std::string& error)
    {
        if (pipeline_ != nil) return true;
        NSError* libraryError = nil;
        NSString* source = [NSString stringWithUTF8String:kOpticalFlowKernel];
        id<MTLLibrary> library = [gMetalDevice newLibraryWithSource:source
                                                            options:nil
                                                              error:&libraryError];
        if (library == nil)
        {
            error = libraryError != nil
                ? std::string ([[libraryError localizedDescription] UTF8String])
                : "Metal optical-flow shader compilation failed";
            return false;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"boundedBlockFlow"];
        NSError* pipelineError = nil;
        pipeline_ = [gMetalDevice newComputePipelineStateWithFunction:function
                                                                 error:&pipelineError];
#if ! __has_feature(objc_arc)
        [function release];
        [library release];
#endif
        if (pipeline_ == nil)
        {
            error = pipelineError != nil
                ? std::string ([[pipelineError localizedDescription] UTF8String])
                : "Metal optical-flow pipeline creation failed";
            return false;
        }
        return true;
    }

    id<MTLComputePipelineState> pipeline_ = nil;
    std::uint64_t nextLifecycle_ = 1;
    std::uint64_t nextSubmission_ = 1;
    std::unordered_map<std::uint64_t, id<MTLTexture>> outputs_;
};

class MetalSdfExecutionBackend final : public NativeSdfExecutionBackend
{
public:
    NativeSdfExecutionCapabilities capabilities() const override
    {
        std::lock_guard<std::mutex> lock (sokolMutex());
        @autoreleasepool
        {
            return capabilitiesUnlocked();
        }
    }

    NativeSdfSceneSubmission render (const NativeSdfDrawRequest& request) override
    {
        NativeSdfSceneSubmission result;
        std::string programError;
        const auto compiledProgram = resolveSdfProgram (request, programError);
        if (! compiledProgram)
        {
            result.error = programError.empty()
                ? "Metal native SDF program admission failed" : std::move (programError);
            return result;
        }
        const auto available = capabilities();
        {
            if (! available.available)
            {
                result.error = available.error;
                return result;
            }
            const auto validQuality = [] (NativeSdfQuality quality)
            {
                return static_cast<std::uint8_t> (quality)
                    < static_cast<std::uint8_t> (NativeSdfQuality::count);
            };
            if (request.width == 0 || request.height == 0
                || request.width > available.maxExtent
                || request.height > available.maxExtent
                || static_cast<std::uint64_t> (request.width) * request.height
                    > available.maxPixels)
            {
                result.error = "Metal native SDF render dimensions exceed backend limits";
                return result;
            }
            if (request.maximumSteps == 0 || request.maximumSteps > available.maxSteps
                || ! std::isfinite (request.epsilon)
                || request.epsilon < available.minEpsilon
                || request.epsilon > available.maxEpsilon
                || ! std::isfinite (request.maximumDistance)
                || request.maximumDistance < request.epsilon
                || request.maximumDistance > available.maxDistance
                || ! validQuality (request.adaptiveQuality)
                || ! validQuality (request.normalQuality)
                || ! validQuality (request.shadowQuality)
                || (request.output != NativeSdfOutput::color
                    && request.output != NativeSdfOutput::depth
                    && request.output != NativeSdfOutput::normal))
            {
                result.error = "Metal native SDF raymarch controls exceed backend limits";
                return result;
            }

            renderpassoutput::Description outputDescription;
            outputDescription.extent = { request.width, request.height };
            outputDescription.attachments.push_back ({
                renderpassoutput::Output::Color,
                renderpassoutput::PixelFormat::RGBA16Float,
                renderpassoutput::ColorSpace::LinearSRGB,
                outputDescription.extent });
            renderpassoutput::AdmissionFailure outputFailure;
            const auto admittedOutputs = renderpassoutput::admit (
                outputDescription, outputFailure);
            if (! admittedOutputs)
            {
                result.error = "Metal native SDF output admission failed: "
                    + std::string (renderpassoutput::token (outputFailure));
                return result;
            }
            auto& outputBackend = nativeRenderPassOutputBackend();
            auto outputAdmission = outputBackend.admitRenderPassOutputs (*admittedOutputs);
            if (outputAdmission.lifecycle.value == 0 || outputAdmission.resources.size() != 1)
            {
                if (outputAdmission.lifecycle.value != 0)
                    outputBackend.releaseRenderPassOutputs (outputAdmission.lifecycle);
                result.error = outputAdmission.error.empty()
                    ? "Metal native SDF output allocation failed"
                    : std::move (outputAdmission.error);
                return result;
            }

            std::unique_lock<std::mutex> lock (sokolMutex());
            @autoreleasepool
            {
            auto frame = std::make_shared<MetalSdfSceneFrame>();
            frame->width_ = request.width;
            frame->height_ = request.height;
            frame->outputBackend_ = &outputBackend;
            frame->lifecycle_ = outputAdmission.lifecycle;
            frame->frameMemory_ = outputAdmission.frameMemory;
            frame->receipt_.compiledRecordCount = static_cast<std::uint32_t> (
                compiledProgram->records().size());
            frame->receipt_.compiledRecordBytes = frame->receipt_.compiledRecordCount
                * sizeof (arbitgpu::NativeSdfCompiledRecord);
            frame->receipt_.geometryCacheBytes = request.geometryCacheBytes;
            frame->receipt_.backendProgramBytes = std::strlen (kSdfVertexShader)
                + std::strlen (kSdfFragmentShader);
            frame->receipt_.uniformBytes = sizeof (SdfUniforms);
            frame->receipt_.attachmentBytes = outputAdmission.frameMemory.requestedBytes;
            frame->receipt_.totalBytes = frame->receipt_.compiledRecordBytes
                + frame->receipt_.geometryCacheBytes + frame->receipt_.backendProgramBytes
                + frame->receipt_.uniformBytes + frame->receipt_.attachmentBytes;
            frame->colorImage.id = static_cast<std::uint32_t> (
                outputAdmission.resources[0].image);
            frame->colorAttachmentView.id = static_cast<std::uint32_t> (
                outputAdmission.resources[0].attachmentView);
            frame->colorTextureView.id = static_cast<std::uint32_t> (
                outputAdmission.resources[0].textureView);
            auto fail = [&] (std::string diagnostic)
            {
                result.error = std::move (diagnostic);
                lock.unlock();
                frame.reset();
                return result;
            };

            if (! ensurePipelineUnlocked (result.error))
                return fail (std::move (result.error));

            const bool resourcesOk =
                resourceValid (sg_query_image_state (frame->colorImage))
                && resourceValid (sg_query_view_state (frame->colorAttachmentView))
                && resourceValid (sg_query_view_state (frame->colorTextureView));
            if (! resourcesOk)
                return fail ("Metal native SDF GPU resource creation failed");

            SdfUniforms uniforms {};
            for (std::size_t index = 0; index < compiledProgram->records().size(); ++index)
            {
                const auto& source = compiledProgram->records()[index];
                auto& destination = uniforms.records[index];
                destination.header[0] = source.operation;
                destination.header[1] = source.input0;
                destination.header[2] = source.input1;
                destination.header[3] = source.parameterCount;
                std::copy_n (source.parameters.data(), 4, destination.parameters0);
                std::copy_n (source.parameters.data() + 4, 4, destination.parameters1);
            }
            uniforms.raymarch[0] = static_cast<float> (request.epsilon);
            uniforms.raymarch[1] = static_cast<float> (request.maximumDistance);
            uniforms.raymarch[2] = static_cast<float> (request.width);
            uniforms.raymarch[3] = static_cast<float> (request.height);
            uniforms.quality[0] = request.maximumSteps;
            uniforms.quality[1] = static_cast<std::uint32_t> (request.normalQuality);
            uniforms.quality[2] = static_cast<std::uint32_t> (request.shadowQuality);
            uniforms.quality[3] = static_cast<std::uint32_t> (request.adaptiveQuality);
            uniforms.program[0] = static_cast<std::uint32_t> (request.output);
            uniforms.program[1] = compiledProgram->rootIndex();
            uniforms.program[2] = static_cast<std::uint32_t> (compiledProgram->records().size());

            sg_pass pass = {};
            pass.attachments.colors[0] = frame->colorAttachmentView;
            pass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
            pass.action.colors[0].store_action = SG_STOREACTION_STORE;
            pass.action.colors[0].clear_value = {
                7.0f / 255.0f, 10.0f / 255.0f, 18.0f / 255.0f, 1.0f };
            pass.label = "arbit-metal-sdf-scene-pass";
            sg_begin_pass (&pass);
            sg_apply_pipeline (pipeline_);
            const sg_range uniformRange = { &uniforms, sizeof (uniforms) };
            sg_apply_uniforms (0, &uniformRange);
            sg_draw (0, 3, 1);
            sg_end_pass();
            sg_commit();

            result.rendered = true;
            result.frame = std::move (frame);
            return result;
        }
        }
    }

private:
    bool ensurePipelineUnlocked (std::string& error)
    {
        if (resourceValid (sg_query_shader_state (shader_))
            && resourceValid (sg_query_pipeline_state (pipeline_)))
            return true;

        sg_shader_desc shaderDesc = {};
        shaderDesc.vertex_func.source = kSdfVertexShader;
        shaderDesc.fragment_func.source = kSdfFragmentShader;
        shaderDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        shaderDesc.uniform_blocks[0].size = sizeof (SdfUniforms);
        shaderDesc.uniform_blocks[0].msl_buffer_n = 0;
        shaderDesc.label = "arbit-metal-sdf-scene-shader";
        shader_ = sg_make_shader (&shaderDesc);

        sg_pipeline_desc pipelineDesc = {};
        pipelineDesc.shader = shader_;
        pipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
        pipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        pipelineDesc.primitive_type = SG_PRIMITIVETYPE_TRIANGLES;
        pipelineDesc.sample_count = 1;
        pipelineDesc.label = "arbit-metal-sdf-scene-pipeline";
        pipeline_ = sg_make_pipeline (&pipelineDesc);
        if (! resourceValid (sg_query_shader_state (shader_))
            || ! resourceValid (sg_query_pipeline_state (pipeline_)))
        {
            if (pipeline_.id != 0) sg_destroy_pipeline (pipeline_);
            if (shader_.id != 0) sg_destroy_shader (shader_);
            pipeline_ = {};
            shader_ = {};
            error = "Metal native SDF shader or pipeline creation failed";
            return false;
        }
        return true;
    }

    static NativeSdfExecutionCapabilities capabilitiesUnlocked()
    {
        NativeSdfExecutionCapabilities result;
        if (! ensureSokolMetal())
        {
            result.error = gSokolError;
            return result;
        }
        result.backend = "metal";
        result.device = deviceName (gMetalDevice);
        if (sg_query_backend() != SG_BACKEND_METAL_MACOS)
        {
            result.error = "sokol_gfx did not select the macOS Metal backend";
            return result;
        }
        const auto nativeLimits = sg_query_limits();
        const auto format = sg_query_pixelformat (SG_PIXELFORMAT_RGBA16F);
        if (! format.render || ! format.sample || nativeLimits.max_image_size_2d <= 0)
        {
            result.error = "Metal does not support the native SDF RGBA16F target";
            return result;
        }
        result.available = true;
        for (std::uint32_t operation = static_cast<std::uint32_t> (
                 videowire::SdfOperation::sphere);
             operation <= static_cast<std::uint32_t> (videowire::SdfOperation::domainWarp);
             ++operation)
            result.supportedOperations[operation] = true;
        result.supportedOutputs[static_cast<std::size_t> (
            NativeSdfOutput::color)] = true;
        result.supportedOutputs[static_cast<std::size_t> (
            NativeSdfOutput::depth)] = true;
        result.supportedOutputs[static_cast<std::size_t> (
            NativeSdfOutput::normal)] = true;
        result.maxOperations = kNativeSdfMaximumRecords;
        result.maxDepth = kNativeSdfMaximumDepth;
        result.maxExtent = static_cast<std::uint32_t> (
            std::min (nativeLimits.max_image_size_2d, 4096));
        result.maxPixels = std::min (
            static_cast<std::uint64_t> (result.maxExtent) * result.maxExtent,
            4096ull * 4096ull);
        result.maxSteps = 512;
        result.minEpsilon = 0.000001;
        result.maxEpsilon = 0.1;
        result.maxDistance = 1000.0;
        return result;
    }

    sg_shader shader_ {};
    sg_pipeline pipeline_ {};
};

class MetalFixtureSceneBackend final : public NativeFixtureSceneBackend
{
public:
    BackendInfo info() const override
    {
        return queryNativeBackend();
    }

    GeometryCoreExecutionCapabilities geometryCoreCapabilities() const override
    {
        GeometryCoreExecutionCapabilities capabilities;
        const auto available = info();
        if (! available.available) return capabilities;
        capabilities.immutableSourceBuffers = true;
        capabilities.stableElementIds = true;
        capabilities.gpuInstancingWithoutMeshExpansion = true;
        capabilities.supportedCarriers = (1u << static_cast<unsigned> (
            videowire::geometry::CarrierKind::geometry3D))
            | (1u << static_cast<unsigned> (
                videowire::geometry::CarrierKind::instances3D));
        capabilities.maxVertices = HarmonicMIDI::grid::Visual3DScene::kMaxVertices;
        capabilities.maxIndices = HarmonicMIDI::grid::Visual3DScene::kMaxIndices;
        capabilities.maxPoints = 1;
        capabilities.maxCurvePoints = 1;
        capabilities.maxSplines = 1;
        capabilities.maxInstances = HarmonicMIDI::grid::Visual3DScene::kMaxObjects;
        capabilities.maxFieldElements = 1;
        capabilities.maxAttributes = videowire::geometry::kMaximumAttributes;
        capabilities.maxOperations = videowire::geometry::kMaximumOperations;
        capabilities.maxDispatches = videowire::geometry::kMaximumDispatches;
        capabilities.maxBufferBytes = 64u * 1024u * 1024u;
        return capabilities;
    }

    NativeFixtureScenePreparation prepare (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& requestedMaterialProgram) override
    {
        return prepareGeometryInstances(scene, requestedMaterialProgram, {}, false);
    }

    NativeFixtureScenePreparation prepareGeometryInstances (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& requestedMaterialProgram,
        const std::shared_ptr<const videohelper::geometry::AdmittedPlanValue>& geometryAdmission,
        bool diagnosticInstanceIdentityColors) override
    {
        using namespace HarmonicMIDI::grid;

        NativeFixtureScenePreparation result;
        const auto materialProgram = requestedMaterialProgram == nullptr
            ? std::shared_ptr<const NativeFixtureSurfaceMaterialProgram> {}
            : std::make_shared<const NativeFixtureSurfaceMaterialProgram> (
                *requestedMaterialProgram);
        if (scene == nullptr || ! validateVisual3DScene (*scene).valid()
            || scene->objectCount == 0 || scene->materialCount == 0
            || scene->lightCount > Visual3DScene::kMaxLights
            || scene->cameraCount == 0 || scene->cameraCount > Visual3DScene::kMaxCameras
            || visual3d_detail::findById (scene->cameras, scene->cameraCount,
                                          scene->activeCamera) == nullptr)
        {
            result.error = "Metal fixture preparation requires a bounded renderable scene";
            return result;
        }
        const auto& sceneMaterial = scene->materials[0];
        const auto& firstObject = scene->objects[0];
        bool instancedSharedGeometry = false;
        if (geometryAdmission != nullptr)
        {
            if (geometryAdmission->value().descriptor().carrier
                    != videowire::geometry::CarrierKind::instances3D)
            {
                result.error = "Metal Geometry Core preparation requires admitted Instances3D authority";
                return result;
            }
            const auto& admittedInstances = std::get<videowire::geometry::InstancesData>(
                geometryAdmission->value().descriptor().data).instances;
            instancedSharedGeometry = admittedInstances.size() == scene->objectCount
                && scene->objectCount > 1 && materialProgram == nullptr
                && !firstObject.parent.isValid()
                && firstObject.material == sceneMaterial.id
                && sceneMaterial.alphaMode != SceneAlphaMode::Blend;
            for (std::size_t index = 0;
                 instancedSharedGeometry && index < scene->objectCount; ++index)
            {
                const auto& candidate = scene->objects[index];
                instancedSharedGeometry = admittedInstances[index].stableId == candidate.id.value
                    && !candidate.parent.isValid() && candidate.material == firstObject.material
                    && candidate.firstVertex == firstObject.firstVertex
                    && candidate.vertexCount == firstObject.vertexCount
                    && candidate.firstIndex == firstObject.firstIndex
                    && candidate.indexCount == firstObject.indexCount;
            }
            if (!instancedSharedGeometry)
            {
                result.error = "Metal Geometry Core authority does not match the retained instance scene";
                return result;
            }
        }
        const bool materialKindValid = materialProgram == nullptr
            || (materialProgram->kind == NativeFixtureMaterialKind::SurfacePbr
                && materialProgram->parameters.identifiers[0] == sceneMaterial.id.value)
            || validNativeFixtureDiffractionProgram (
                *materialProgram, NativeFixtureMaterialBackend::Metal,
                scene->objects[0].id);
        if (materialProgram != nullptr
            && (materialProgram->layoutVersion
                    != NativeFixtureSurfaceMaterialProgram::kLayoutVersion
                || materialProgram->backend != NativeFixtureMaterialBackend::Metal
                || materialProgram->programIdentity.empty()
                || materialProgram->bindingDigest.empty()
                || materialProgram->object != scene->objects[0].id
                || ! materialKindValid))
        {
            result.error = "Metal fixture preparation requires an exact bounded material binding";
            return result;
        }

        std::lock_guard<std::mutex> lock (sokolMutex());
        @autoreleasepool
        {
            if (! ensureSokolMetal())
            {
                result.error = gSokolError;
                return result;
            }
            if (scene == nullptr || ! validateVisual3DScene (*scene).valid()
                || scene->objectCount == 0 || scene->materialCount == 0
                || scene->lightCount > Visual3DScene::kMaxLights
                || scene->cameraCount == 0 || scene->cameraCount > Visual3DScene::kMaxCameras
                || visual3d_detail::findById (scene->cameras, scene->cameraCount,
                                              scene->activeCamera) == nullptr)
            {
                result.error = "Metal fixture preparation requires a bounded renderable scene";
                return result;
            }
            const auto& sceneMaterial = scene->materials[0];
            const bool materialKindValid = materialProgram == nullptr
                || (materialProgram->kind == NativeFixtureMaterialKind::SurfacePbr
                    && materialProgram->parameters.identifiers[0] == sceneMaterial.id.value
                    && validNativeFixtureSurfaceParameters (materialProgram->parameters))
                || validNativeFixtureDiffractionProgram (
                    *materialProgram, NativeFixtureMaterialBackend::Metal,
                    scene->objects[0].id);
            if (materialProgram != nullptr
                && (materialProgram->layoutVersion
                        != NativeFixtureSurfaceMaterialProgram::kLayoutVersion
                    || materialProgram->backend
                        != NativeFixtureMaterialBackend::Metal
                    || materialProgram->programIdentity.empty()
                    || materialProgram->bindingDigest.empty()
                    || materialProgram->object != scene->objects[0].id
                    || ! materialKindValid))
            {
                result.error = "Metal fixture preparation requires an exact bounded material binding";
                return result;
            }
            auto resources = std::make_shared<MetalFixtureSceneResources>();
            resources->snapshot = scene;
            resources->materialProgram = materialProgram;
            resources->geometryAdmission = geometryAdmission;
            resources->instancedSharedGeometry = instancedSharedGeometry;
            resources->diagnosticInstanceIdentityColors = diagnosticInstanceIdentityColors;
            resources->rendererGeneration = nextMetalRendererGeneration();
            auto fail = [&] (const std::string& error)
            {
                resources->destroyUnlocked();
                result.error = error;
                return result;
            };

            static_assert (sizeof (SceneVertex) == 16 * sizeof (float),
                           "fixture vertex upload layout changed");
            static_assert (sizeof (SceneTexelRgba8) == 4,
                           "fixture texel upload layout changed");

            const auto vertexBytes = scene->vertexCount * sizeof (SceneVertex);
            sg_buffer_desc vertexDesc = {};
            vertexDesc.usage.vertex_buffer = true;
            vertexDesc.usage.storage_buffer = true;
            vertexDesc.data = {
                scene->vertices.data(),
                vertexBytes
            };
            vertexDesc.label = "arbit-metal-fixture-vertices";
            resources->vertexBuffer = sg_make_buffer (&vertexDesc);

            const auto indexBytes = scene->indexCount * sizeof (std::uint32_t);
            sg_buffer_desc indexDesc = {};
            indexDesc.usage.index_buffer = true;
            indexDesc.data = {
                scene->indices.data(),
                indexBytes
            };
            indexDesc.label = "arbit-metal-fixture-indices";
            resources->indexBuffer = sg_make_buffer (&indexDesc);

            sg_buffer_desc instanceDesc = {};
            instanceDesc.size = Visual3DScene::kMaxObjects
                * sizeof (MetalFixtureInstanceGpuRecord);
            instanceDesc.usage.vertex_buffer = true;
            instanceDesc.usage.dynamic_update = true;
            instanceDesc.label = instancedSharedGeometry
                ? "arbit-metal-geometry-core-instances"
                : "arbit-metal-fixture-unused-instance-attributes";
            resources->geometryInstanceBuffer = sg_make_buffer (&instanceDesc);

            const SceneTexelRgba8 whiteTexel { 255, 255, 255, 255 };
            std::size_t textureBytes = sizeof (whiteTexel);
            if (materialProgram != nullptr)
            {
                using Source = arbitgpu::NativeFixtureSurfaceMaterialProgram::BaseColorSource;
                if (materialProgram->baseColorSource == Source::ImportedSrgbTexture)
                {
                    if (! materialProgram->importedBaseColorTexture
                        || materialProgram->importedBaseColorTexture->texelCount == 0)
                    {
                        return fail ("Metal fixture material program has no owned imported texture");
                    }
                    const auto& imported = *materialProgram->importedBaseColorTexture;
                    textureBytes += imported.texelCount * sizeof (SceneTexelRgba8);
                }
            }
            // Upload separate linear and sRGB images because one glTF image can serve
            // both color and data roles. Each role also keeps its glTF sampler state.
            resources->textureImages.resize (1 + scene->textureCount * 2);
            resources->textureViews.resize (resources->textureImages.size());
            resources->samplers.resize (resources->textureImages.size());
            const auto filter = [] (std::uint32_t value)
            {
                return value == 9728 || value == 9984 || value == 9986
                    ? SG_FILTER_NEAREST : SG_FILTER_LINEAR;
            };
            const auto wrap = [] (std::uint32_t value)
            {
                if (value == 33071) return SG_WRAP_CLAMP_TO_EDGE;
                if (value == 33648) return SG_WRAP_MIRRORED_REPEAT;
                return SG_WRAP_REPEAT;
            };
            for (std::size_t slot = 0; slot < resources->textureImages.size(); ++slot)
            {
                const auto* source = slot > 0 ? &scene->textures[(slot - 1) / 2] : nullptr;
                const auto colorRole = source != nullptr && ((slot - 1) % 2 == 1);
                const void* data = &whiteTexel;
                std::size_t bytes = sizeof (whiteTexel);
                int textureWidth = 1, textureHeight = 1;
                if (source != nullptr)
                {
                    data = scene->textureTexels.data() + source->firstTexel;
                    bytes = static_cast<std::size_t> (source->width) * source->height
                        * sizeof (SceneTexelRgba8);
                    textureWidth = static_cast<int> (source->width);
                    textureHeight = static_cast<int> (source->height);
                    textureBytes += bytes;
                }
                else if (materialProgram != nullptr
                         && materialProgram->baseColorSource
                            == NativeFixtureSurfaceMaterialProgram::BaseColorSource::ImportedSrgbTexture)
                {
                    const auto& imported = *materialProgram->importedBaseColorTexture;
                    data = imported.texels.data();
                    bytes = imported.texelCount * sizeof (SceneTexelRgba8);
                    textureWidth = static_cast<int> (imported.width);
                    textureHeight = static_cast<int> (imported.height);
                }
                sg_image_desc textureDesc = {};
                textureDesc.width = textureWidth;
                textureDesc.height = textureHeight;
                textureDesc.num_mipmaps = source != nullptr && source->minFilter >= 9984 ? 0 : 1;
                textureDesc.pixel_format = colorRole || slot == 0
                    ? SG_PIXELFORMAT_SRGB8A8 : SG_PIXELFORMAT_RGBA8;
                textureDesc.data.mip_levels[0] = { data, bytes };
                textureDesc.label = colorRole ? "arbit-metal-fixture-srgb-texture"
                                              : "arbit-metal-fixture-linear-texture";
                resources->textureImages[slot] = sg_make_image (&textureDesc);
                sg_view_desc textureViewDesc = {};
                textureViewDesc.texture.image = resources->textureImages[slot];
                resources->textureViews[slot] = sg_make_view (&textureViewDesc);
                sg_sampler_desc samplerDesc = {};
                samplerDesc.min_filter = source != nullptr ? filter (source->minFilter) : SG_FILTER_NEAREST;
                samplerDesc.mag_filter = source != nullptr ? filter (source->magFilter) : SG_FILTER_NEAREST;
                samplerDesc.mipmap_filter = source != nullptr && source->minFilter >= 9984
                    ? filter (source->minFilter) : SG_FILTER_NEAREST;
                samplerDesc.wrap_u = source != nullptr ? wrap (source->wrapS) : SG_WRAP_REPEAT;
                samplerDesc.wrap_v = source != nullptr ? wrap (source->wrapT) : SG_WRAP_REPEAT;
                samplerDesc.label = "arbit-metal-fixture-sampler";
                resources->samplers[slot] = sg_make_sampler (&samplerDesc);
            }

            sg_shader_desc shaderDesc = {};
            shaderDesc.vertex_func.source = kFixtureVertexShader;
            shaderDesc.fragment_func.source = kFixtureFragmentShader;
            std::string shaderError;
            if (! compileMetalShaderStage (
                    kFixtureVertexShader, "vertex", shaderError)
                || ! compileMetalShaderStage (
                    kFixtureFragmentShader, "fragment", shaderError))
            {
                return fail (shaderError);
            }
            shaderDesc.attrs[0].base_type = SG_SHADERATTRBASETYPE_FLOAT;
            shaderDesc.attrs[1].base_type = SG_SHADERATTRBASETYPE_FLOAT;
            shaderDesc.attrs[2].base_type = SG_SHADERATTRBASETYPE_FLOAT;
            shaderDesc.attrs[3].base_type = SG_SHADERATTRBASETYPE_FLOAT;
            shaderDesc.attrs[4].base_type = SG_SHADERATTRBASETYPE_FLOAT;
            shaderDesc.attrs[5].base_type = SG_SHADERATTRBASETYPE_FLOAT;
            shaderDesc.attrs[6].base_type = SG_SHADERATTRBASETYPE_FLOAT;
            shaderDesc.attrs[7].base_type = SG_SHADERATTRBASETYPE_FLOAT;
            shaderDesc.attrs[8].base_type = SG_SHADERATTRBASETYPE_FLOAT;
            shaderDesc.attrs[9].base_type = SG_SHADERATTRBASETYPE_FLOAT;
            shaderDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
            shaderDesc.uniform_blocks[0].size = sizeof (FixtureUniforms);
            shaderDesc.uniform_blocks[0].msl_buffer_n = 0;
            shaderDesc.uniform_blocks[1].stage = SG_SHADERSTAGE_FRAGMENT;
            shaderDesc.uniform_blocks[1].size = sizeof (FixtureUniforms);
            shaderDesc.uniform_blocks[1].msl_buffer_n = 0;
            const char* textureNames[] = { "baseTexture", "metallicRoughnessTexture",
                "normalTexture", "occlusionTexture", "emissiveTexture" };
            for (int slot = 0; slot < 5; ++slot)
            {
                const auto shaderSlot = static_cast<std::uint8_t> (slot);
                shaderDesc.views[slot].texture.stage = SG_SHADERSTAGE_FRAGMENT;
                shaderDesc.views[slot].texture.image_type = SG_IMAGETYPE_2D;
                shaderDesc.views[slot].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
                shaderDesc.views[slot].texture.msl_texture_n = slot;
                shaderDesc.samplers[slot].stage = SG_SHADERSTAGE_FRAGMENT;
                shaderDesc.samplers[slot].sampler_type = SG_SAMPLERTYPE_FILTERING;
                shaderDesc.samplers[slot].msl_sampler_n = slot;
                shaderDesc.texture_sampler_pairs[slot] = {
                    SG_SHADERSTAGE_FRAGMENT, shaderSlot, shaderSlot, textureNames[slot] };
            }
            shaderDesc.label = "arbit-metal-fixture-shader";
            resources->shader = sg_make_shader (&shaderDesc);

            sg_pipeline_desc pipelineDesc = {};
            pipelineDesc.shader = resources->shader;
            pipelineDesc.layout.buffers[0].stride = sizeof (SceneVertex);
            pipelineDesc.layout.attrs[0] = {
                0, static_cast<int> (offsetof (SceneVertex, position)), SG_VERTEXFORMAT_FLOAT3 };
            pipelineDesc.layout.attrs[1] = {
                0, static_cast<int> (offsetof (SceneVertex, normal)), SG_VERTEXFORMAT_FLOAT3 };
            pipelineDesc.layout.attrs[2] = {
                0, static_cast<int> (offsetof (SceneVertex, uv)), SG_VERTEXFORMAT_FLOAT2 };
            pipelineDesc.layout.attrs[3] = {
                0, static_cast<int> (offsetof (SceneVertex, color)), SG_VERTEXFORMAT_FLOAT4 };
            pipelineDesc.layout.attrs[4] = {
                0, static_cast<int> (offsetof (SceneVertex, tangent)), SG_VERTEXFORMAT_FLOAT4 };
            pipelineDesc.layout.buffers[1].stride = sizeof (MetalFixtureInstanceGpuRecord);
            pipelineDesc.layout.buffers[1].step_func = SG_VERTEXSTEP_PER_INSTANCE;
            for (int column = 0; column < 4; ++column)
                pipelineDesc.layout.attrs[5 + column] = {
                    1, column * 4 * static_cast<int> (sizeof (float)), SG_VERTEXFORMAT_FLOAT4 };
            pipelineDesc.layout.attrs[9] = {
                1, static_cast<int> (offsetof (MetalFixtureInstanceGpuRecord, identityColor)),
                SG_VERTEXFORMAT_FLOAT4 };
            pipelineDesc.depth.pixel_format = SG_PIXELFORMAT_DEPTH_STENCIL;
            const bool diffractionProgram = resources->materialProgram != nullptr
                && resources->materialProgram->kind
                    == NativeFixtureMaterialKind::DiffractionReflective;
            pipelineDesc.depth.compare = diffractionProgram
                ? SG_COMPAREFUNC_LESS_EQUAL : SG_COMPAREFUNC_LESS;
            pipelineDesc.depth.write_enabled = true;
            pipelineDesc.color_count = 2;
            pipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_BGRA8;
            pipelineDesc.colors[0].blend.enabled = diffractionProgram;
            pipelineDesc.colors[0].blend.src_factor_rgb = SG_BLENDFACTOR_ONE;
            pipelineDesc.colors[0].blend.dst_factor_rgb = SG_BLENDFACTOR_ONE;
            pipelineDesc.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_ZERO;
            pipelineDesc.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE;
            pipelineDesc.colors[1].pixel_format = SG_PIXELFORMAT_R32F;
            pipelineDesc.primitive_type = SG_PRIMITIVETYPE_TRIANGLES;
            pipelineDesc.index_type = SG_INDEXTYPE_UINT32;
            pipelineDesc.sample_count = 1;
            resources->pipelines.resize (diffractionProgram ? 3 : 6);
            const auto modeCount = diffractionProgram ? 1 : 2;
            for (int blend = 0; blend < modeCount; ++blend)
                for (int winding = 0; winding < 3; ++winding)
                {
                    pipelineDesc.depth.write_enabled = blend == 0;
                    pipelineDesc.cull_mode = winding == 0 ? SG_CULLMODE_NONE : SG_CULLMODE_BACK;
                    pipelineDesc.face_winding = winding == 2 ? SG_FACEWINDING_CW
                                                             : SG_FACEWINDING_CCW;
                    pipelineDesc.colors[0].blend.enabled = diffractionProgram || blend != 0;
                    pipelineDesc.colors[0].blend.src_factor_rgb = diffractionProgram
                        ? SG_BLENDFACTOR_ONE : SG_BLENDFACTOR_SRC_ALPHA;
                    pipelineDesc.colors[0].blend.dst_factor_rgb = diffractionProgram
                        ? SG_BLENDFACTOR_ONE : SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
                    pipelineDesc.colors[0].blend.src_factor_alpha = diffractionProgram
                        ? SG_BLENDFACTOR_ZERO : SG_BLENDFACTOR_ONE;
                    pipelineDesc.colors[0].blend.dst_factor_alpha = diffractionProgram
                        ? SG_BLENDFACTOR_ONE : SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
                    pipelineDesc.label = diffractionProgram
                        ? "arbit-metal-fixture-diffraction-pipeline"
                        : (blend != 0 ? "arbit-metal-fixture-blend-pipeline"
                                      : "arbit-metal-fixture-opaque-pipeline");
                    resources->pipelines[blend * 3 + winding] = sg_make_pipeline (&pipelineDesc);
                }

            const char* failedResource = nullptr;
            if (! resourceValid (sg_query_buffer_state (resources->vertexBuffer)))
                failedResource = "vertex buffer";
            else if (! resourceValid (sg_query_buffer_state (resources->indexBuffer)))
                failedResource = "index buffer";
            else if (! resourceValid (sg_query_buffer_state (resources->geometryInstanceBuffer)))
                failedResource = "instance buffer";
            else if (! resourceValid (sg_query_shader_state (resources->shader)))
                failedResource = "shader";
            for (const auto image : resources->textureImages)
                if (failedResource == nullptr && !resourceValid (sg_query_image_state (image)))
                    failedResource = "texture image";
            for (const auto view : resources->textureViews)
                if (failedResource == nullptr && !resourceValid (sg_query_view_state (view)))
                    failedResource = "texture view";
            for (const auto sampler : resources->samplers)
                if (failedResource == nullptr && !resourceValid (sg_query_sampler_state (sampler)))
                    failedResource = "sampler";
            for (const auto pipeline : resources->pipelines)
                if (failedResource == nullptr && !resourceValid (sg_query_pipeline_state (pipeline)))
                    failedResource = "pipeline";
            if (failedResource != nullptr)
            {
                const auto log = lastSokolLog();
                auto message = std::string { "Metal fixture " } + failedResource
                    + " creation failed";
                if (! log.empty())
                    message += ": " + log;
                return fail (message.c_str());
            }

            result.prepared = true;
            result.resources = std::move (resources);
            result.stats.staticUploadCount = 1;
            result.stats.vertexBytes = vertexBytes;
            result.stats.indexBytes = indexBytes;
            result.stats.textureBytes = textureBytes;
            return result;
        }
    }

    NativeFixtureSceneSubmission render (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
        const std::shared_ptr<const NativeFixtureSceneResources>& nativeResources,
        std::uint32_t width,
        std::uint32_t height,
        NativeFixtureSceneRuntimeInputs runtimeInputs) override
    {
        using namespace HarmonicMIDI::grid;

        NativeFixtureSceneSubmission result;
        std::lock_guard<std::mutex> lock (sokolMutex());
        @autoreleasepool
        {
            if (! ensureSokolMetal())
            {
                result.error = gSokolError;
                return result;
            }

            const auto resources = std::dynamic_pointer_cast<const MetalFixtureSceneResources> (
                nativeResources);
            if (scene == nullptr || resources == nullptr || resources->snapshot != scene)
            {
                result.error = "Metal fixture render requires exact prepared scene resources";
                return result;
            }
            if (! nativeFixtureDimensionsWithinBounds (width, height))
            {
                result.error = "Metal fixture render dimensions exceed backend limits";
                return result;
            }
            if (resources->materialProgram != nullptr
                && ! nativeFixtureDiffractionWorkWithinBudget (
                    *resources->materialProgram, width, height))
            {
                result.error = "Metal fixture diffraction workload exceeds backend limits";
                return result;
            }
            diffractivefoil::EvaluationSchedule spatialSchedule;
            if (resources->materialProgram != nullptr
                && resources->materialProgram->diffractionFoilMaximumEvaluations != 0
                && !nativeFixtureSpatialFoilEvaluationSchedule(
                    *resources->materialProgram, width, height, spatialSchedule))
            {
                result.error = "Metal fixture diffraction workload exceeds backend limits";
                return result;
            }
            if (!validFixtureRuntimeInputs(runtimeInputs))
            {
                result.error = "Metal fixture scene modulation is non-finite or out of bounds";
                return result;
            }
            if (resources->materialProgram != nullptr
                && resources->materialProgram->baseColorSource
                    == NativeFixtureSurfaceMaterialProgram::BaseColorSource::TimeLinearMix
                && (! std::isfinite (runtimeInputs.timeSeconds)
                    || std::abs (runtimeInputs.timeSeconds)
                        > surfacematerial::kMaximumEvaluationMagnitude))
            {
                result.error = "Metal fixture material time input is non-finite or out of bounds";
                return result;
            }
            const auto noteInstances = prepareNativeNoteInstances(runtimeInputs);

            auto frame = std::make_shared<MetalFixtureSceneFrame>();
            frame->width_ = width;
            frame->height_ = height;
            frame->rendererGeneration_ = resources->rendererGeneration;
            frame->staticResources = resources;
            auto fail = [&] (const char* error)
            {
                frame->destroyUnlocked();
                result.error = error;
                return result;
            };

            const auto* selectedCamera = visual3d_detail::findById (
                scene->cameras, scene->cameraCount, scene->activeCamera);
            if (!runtimeInputs.cameraOverride.has_value() && selectedCamera == nullptr)
            {
                result.error = "Metal fixture render cannot resolve the active camera";
                return result;
            }
            const auto& camera = runtimeInputs.cameraOverride.has_value()
                ? *runtimeInputs.cameraOverride : *selectedCamera;

            sg_image_desc colorDesc = {};
            colorDesc.usage.color_attachment = true;
            colorDesc.width = static_cast<int> (width);
            colorDesc.height = static_cast<int> (height);
            colorDesc.pixel_format = SG_PIXELFORMAT_BGRA8;
            colorDesc.sample_count = 1;
            colorDesc.label = "arbit-metal-fixture-color";
            frame->colorImage = sg_make_image (&colorDesc);
            sg_view_desc colorAttachmentDesc = {};
            colorAttachmentDesc.color_attachment.image = frame->colorImage;
            frame->colorAttachmentView = sg_make_view (&colorAttachmentDesc);
            sg_view_desc colorTextureDesc = {};
            colorTextureDesc.texture.image = frame->colorImage;
            frame->colorTextureView = sg_make_view (&colorTextureDesc);

            const auto r32Capabilities = sg_query_pixelformat (SG_PIXELFORMAT_R32F);
            if (! r32Capabilities.render || ! r32Capabilities.sample
                || r32Capabilities.bytes_per_pixel != 4)
                return fail ("Metal does not support sampleable R32F fixture depth");
            sg_image_desc depthDesc = {};
            depthDesc.usage.color_attachment = true;
            depthDesc.width = static_cast<int> (width);
            depthDesc.height = static_cast<int> (height);
            depthDesc.pixel_format = SG_PIXELFORMAT_R32F;
            depthDesc.sample_count = 1;
            depthDesc.label = "arbit-metal-fixture-linear-depth";
            frame->depthImage = sg_make_image (&depthDesc);
            sg_view_desc depthViewDesc = {};
            depthViewDesc.color_attachment.image = frame->depthImage;
            frame->depthView = sg_make_view (&depthViewDesc);
            sg_view_desc depthTextureViewDesc = {};
            depthTextureViewDesc.texture.image = frame->depthImage;
            frame->depthTextureView = sg_make_view (&depthTextureViewDesc);

            sg_image_desc depthStencilDesc = {};
            depthStencilDesc.usage.depth_stencil_attachment = true;
            depthStencilDesc.width = static_cast<int> (width);
            depthStencilDesc.height = static_cast<int> (height);
            depthStencilDesc.pixel_format = SG_PIXELFORMAT_DEPTH_STENCIL;
            depthStencilDesc.sample_count = 1;
            depthStencilDesc.label = "arbit-metal-fixture-depth-stencil";
            frame->depthStencilImage = sg_make_image (&depthStencilDesc);
            sg_view_desc depthStencilViewDesc = {};
            depthStencilViewDesc.depth_stencil_attachment.image = frame->depthStencilImage;
            frame->depthStencilView = sg_make_view (&depthStencilViewDesc);

            const bool resourcesOk =
                resourceValid (sg_query_image_state (frame->colorImage))
                && resourceValid (sg_query_view_state (frame->colorAttachmentView))
                && resourceValid (sg_query_view_state (frame->colorTextureView))
                && resourceValid (sg_query_image_state (frame->depthImage))
                && resourceValid (sg_query_view_state (frame->depthView))
                && resourceValid (sg_query_view_state (frame->depthTextureView))
                && resourceValid (sg_query_image_state (frame->depthStencilImage))
                && resourceValid (sg_query_view_state (frame->depthStencilView));
            if (! resourcesOk)
                return fail ("Metal fixture GPU resource creation failed");

            FixtureUniforms uniforms {};
            std::memcpy(uniforms.noteInstanceTransforms,
                        noteInstances.transforms.data(),
                        sizeof(uniforms.noteInstanceTransforms));
            storeQuaternion (uniforms.cameraRotation, camera.transform.rotation);
            storeVec3 (uniforms.cameraTranslation, camera.transform.translation);
            for (std::size_t axis = 0; axis < 3; ++axis)
                uniforms.cameraTranslation[axis] += runtimeInputs.cameraTranslationOffset[axis];
            uniforms.projection[0] = std::tan (camera.verticalFovRadians * 0.5f);
            uniforms.projection[1] = static_cast<float> (width) / static_cast<float> (height);
            uniforms.projection[2] = camera.nearPlane;
            uniforms.projection[3] = camera.farPlane;
            if (resources->materialProgram != nullptr)
            {
                uniforms.materialKind[0] = static_cast<std::uint32_t> (
                    resources->materialProgram->kind);
                if (resources->materialProgram->kind
                    == NativeFixtureMaterialKind::DiffractionReflective)
                {
                    const auto& path = resources->materialProgram->diffractionPaths[0];
                    uniforms.diffraction = path.material;
                    std::memcpy(uniforms.diffractionIncidentDirectionAndIntensity,
                                &path.incidentDirectionAndIntensity, 4 * sizeof(float));
                    std::copy(path.kindBounceAndReserved.begin(),
                              path.kindBounceAndReserved.end(),
                              uniforms.diffractionPathKindAndBounce);
                    std::memcpy(uniforms.diffractionFoilField,
                                resources->materialProgram->diffractionFoilField.data(),
                                sizeof(uniforms.diffractionFoilField));
                    std::memcpy(uniforms.diffractionOccupancyRectangles,
                                resources->materialProgram->diffractionOccupancyRectangles.data(),
                                sizeof(uniforms.diffractionOccupancyRectangles));
                    uniforms.diffractionSpatialCounts[0]
                        = resources->materialProgram->diffractionFoilMaximumEvaluations > 0 ? 4u : 0u;
                    uniforms.diffractionSpatialCounts[1]
                        = resources->materialProgram->diffractionOccupancyRectangleCount;
                    uniforms.diffractionEvaluationSchedule[0] = spatialSchedule.width;
                    uniforms.diffractionEvaluationSchedule[1] = spatialSchedule.height;
                    uniforms.diffractionEvaluationSchedule[2]
                        = spatialSchedule.maximumEvaluations;
                }
                const auto& pbr = resources->materialProgram->parameters;
                std::copy (pbr.baseColorMetallic.begin(), pbr.baseColorMetallic.begin() + 3,
                           uniforms.baseColor);
                uniforms.baseColor[3] = pbr.normalOpacity[3];
                uniforms.materialParams[0] = pbr.baseColorMetallic[3];
                uniforms.materialParams[1] = pbr.emissionRoughness[3];
                uniforms.materialParams[2] = 1.0f;
                if (resources->materialProgram->baseColorSource
                    == NativeFixtureSurfaceMaterialProgram::BaseColorSource::TimeLinearMix)
                {
                    std::copy (resources->materialProgram->timeMixEndColor.begin(),
                               resources->materialProgram->timeMixEndColor.end(),
                               uniforms.timeMixEndColorAndTime);
                    uniforms.timeMixEndColorAndTime[3] = runtimeInputs.timeSeconds;
                    uniforms.materialParams[3] = 1.0f;
                }
                std::copy (pbr.emissionRoughness.begin(), pbr.emissionRoughness.begin() + 3,
                           uniforms.emissive);
            }
            for (std::size_t channel = 0; channel < 3; ++channel)
                uniforms.emissive[channel] *= runtimeInputs.emissionGain;
            storeVec3 (uniforms.ambient, scene->ambientColor,
                       static_cast<float> (scene->lightCount));
            for (std::size_t lightIndex = 0; lightIndex < scene->lightCount; ++lightIndex)
            {
                const auto& light = scene->lights[lightIndex];
                storeQuaternion (uniforms.lightRotations[lightIndex], light.transform.rotation);
                storeVec3 (uniforms.lightColors[lightIndex], light.color, light.intensity);
                storeVec3 (uniforms.lightPositions[lightIndex], light.transform.translation,
                           light.range);
                uniforms.lightCones[lightIndex][0] = std::cos (light.innerConeAngle);
                uniforms.lightCones[lightIndex][1] = std::cos (light.outerConeAngle);
                uniforms.lightKinds[lightIndex][0] = static_cast<std::uint32_t> (light.kind);
            }

            std::uint32_t instancedDrawCount = 0;
            std::uint32_t instanceBufferUploadCount = 0;
            std::uint32_t submittedInstanceCount = 0;
            std::uint32_t noteInstanceDrawCount = 0;
            std::uint32_t submittedNoteInstanceCount = 0;
            if (resources->instancedSharedGeometry)
            {
                std::array<MetalFixtureInstanceGpuRecord, Visual3DScene::kMaxObjects>
                    instanceRecords {};
                std::vector<videowire::geometry::StableId> stableIdentities;
                stableIdentities.reserve (scene->objectCount);
                for (std::size_t index = 0; index < scene->objectCount; ++index)
                    stableIdentities.push_back (scene->objects[index].id.value);
                std::sort (stableIdentities.begin(), stableIdentities.end());
                stableIdentities.erase (std::unique (stableIdentities.begin(),
                    stableIdentities.end()), stableIdentities.end());
                SceneTransform3D runtimeTransform {};
                runtimeTransform.translation = { runtimeInputs.objectTranslationOffset[0],
                    runtimeInputs.objectTranslationOffset[1],
                    runtimeInputs.objectTranslationOffset[2] };
                runtimeTransform = applyRuntimeObjectTransform (runtimeTransform,
                    runtimeInputs.objectRotationDegrees, runtimeInputs.objectScale);
                const auto runtimeMatrix = metalFixtureTransformMatrix (runtimeTransform);
                for (std::size_t index = 0; index < scene->objectCount; ++index)
                {
                    instanceRecords[index].matrix = multiplyMetalFixtureMatrices (
                        runtimeMatrix, metalFixtureWorldMatrix (*scene, scene->objects[index]));
                    const auto colorSlot = videohelper::geometry::diagnosticColorSlot (
                        stableIdentities, scene->objects[index].id.value);
                    instanceRecords[index].identityColor = {
                        static_cast<float>((colorSlot >> 16) & 0xffu) / 255.0f,
                        static_cast<float>((colorSlot >> 8) & 0xffu) / 255.0f,
                        static_cast<float>(colorSlot & 0xffu) / 255.0f,
                        resources->diagnosticInstanceIdentityColors ? 1.0f : 0.0f };
                }
                const sg_range instanceRange = { instanceRecords.data(),
                    scene->objectCount * sizeof (MetalFixtureInstanceGpuRecord) };
                sg_update_buffer (resources->geometryInstanceBuffer, &instanceRange);
                ++instanceBufferUploadCount;
            }

            sg_pass pass = {};
            pass.attachments.colors[0] = frame->colorAttachmentView;
            pass.attachments.colors[1] = frame->depthView;
            pass.attachments.depth_stencil = frame->depthStencilView;
            pass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
            pass.action.colors[0].store_action = SG_STOREACTION_STORE;
            pass.action.colors[0].clear_value = {
                7.0f / 255.0f, 10.0f / 255.0f, 18.0f / 255.0f, 1.0f };
            pass.action.colors[1].load_action = SG_LOADACTION_CLEAR;
            pass.action.colors[1].store_action = SG_STOREACTION_STORE;
            pass.action.colors[1].clear_value = { 1.0f, 0.0f, 0.0f, 1.0f };
            pass.action.depth.load_action = SG_LOADACTION_CLEAR;
            pass.action.depth.store_action = SG_STOREACTION_DONTCARE;
            pass.action.depth.clear_value = 1.0f;
            pass.action.stencil.load_action = SG_LOADACTION_CLEAR;
            pass.action.stencil.store_action = SG_STOREACTION_DONTCARE;
            pass.label = "arbit-metal-fixture-pass";
            sg_begin_pass (&pass);
            sg_bindings bindings = {};
            bindings.vertex_buffers[0] = resources->vertexBuffer;
            bindings.vertex_buffers[1] = resources->geometryInstanceBuffer;
            bindings.index_buffer = resources->indexBuffer;
            const bool diffractionDraw = resources->materialProgram != nullptr
                && resources->materialProgram->kind == NativeFixtureMaterialKind::DiffractionReflective;
            std::size_t submittedDiffractionDrawCount = 0;
            if (diffractionDraw)
            {
                const auto& drawObject = scene->objects[0];
                SceneTransform3D runtimeTransform {};
                runtimeTransform.translation = { runtimeInputs.objectTranslationOffset[0],
                    runtimeInputs.objectTranslationOffset[1],
                    runtimeInputs.objectTranslationOffset[2] };
                runtimeTransform = applyRuntimeObjectTransform (
                    runtimeTransform, runtimeInputs.objectRotationDegrees,
                    runtimeInputs.objectScale);
                const auto objectMatrix = multiplyMetalFixtureMatrices (
                    metalFixtureTransformMatrix (runtimeTransform),
                    metalFixtureWorldMatrix (*scene, drawObject));
                std::copy (objectMatrix.begin(), objectMatrix.end(), uniforms.objectMatrix);
                const auto* drawMaterial = visual3d_detail::findById (
                    scene->materials, scene->materialCount, drawObject.material);
                const auto winding = drawMaterial != nullptr && drawMaterial->doubleSided ? 0u
                    : (metalFixtureDeterminant3x3 (objectMatrix) < 0.0f ? 2u : 1u);
                sg_apply_pipeline (resources->pipelines[winding]);
                bindings.vertex_buffer_offsets[0] = static_cast<int> (
                    drawObject.firstVertex * sizeof (SceneVertex));
                bindings.index_buffer_offset = static_cast<int> (
                    drawObject.firstIndex * sizeof (std::uint32_t));
                std::size_t baseColorSlot = 0;
                if (drawMaterial != nullptr && drawMaterial->baseColorTexture.isValid())
                    for (std::size_t index = 0; index < scene->textureCount; ++index)
                        if (scene->textures[index].id == drawMaterial->baseColorTexture)
                        {
                            baseColorSlot = 2 + index * 2;
                            break;
                        }
                bindings.views[0] = resources->textureViews[baseColorSlot];
                bindings.samplers[0] = resources->samplers[baseColorSlot];
                for (std::size_t slot = 1; slot < 5; ++slot)
                {
                    bindings.views[slot] = resources->textureViews[0];
                    bindings.samplers[slot] = resources->samplers[0];
                }
                sg_apply_bindings (&bindings);
                const auto pathCount = resources->materialProgram != nullptr
                        && resources->materialProgram->kind
                            == NativeFixtureMaterialKind::DiffractionReflective
                    ? resources->materialProgram->diffractionPathCount : 1u;
                for (std::uint8_t pathIndex = 0; pathIndex < pathCount; ++pathIndex)
                {
                    if (pathCount > 1)
                    {
                        const auto& path = resources->materialProgram->diffractionPaths[pathIndex];
                        uniforms.diffraction = path.material;
                        std::memcpy(uniforms.diffractionIncidentDirectionAndIntensity,
                                    &path.incidentDirectionAndIntensity, 4 * sizeof(float));
                        std::copy(path.kindBounceAndReserved.begin(),
                                  path.kindBounceAndReserved.end(),
                                  uniforms.diffractionPathKindAndBounce);
                        std::memcpy(uniforms.diffractionFoilField,
                                    resources->materialProgram->diffractionFoilField.data(),
                                    sizeof(uniforms.diffractionFoilField));
                        std::memcpy(uniforms.diffractionOccupancyRectangles,
                                    resources->materialProgram->diffractionOccupancyRectangles.data(),
                                    sizeof(uniforms.diffractionOccupancyRectangles));
                        uniforms.diffractionSpatialCounts[0]
                            = resources->materialProgram->diffractionFoilMaximumEvaluations > 0 ? 4u : 0u;
                        uniforms.diffractionSpatialCounts[1]
                            = resources->materialProgram->diffractionOccupancyRectangleCount;
                        uniforms.diffractionEvaluationSchedule[0] = spatialSchedule.width;
                        uniforms.diffractionEvaluationSchedule[1] = spatialSchedule.height;
                        uniforms.diffractionEvaluationSchedule[2]
                            = spatialSchedule.maximumEvaluations;
                    }
                    uniforms.diffractionEvaluationSchedule[3] = pathIndex;
                    const sg_range uniformRange = { &uniforms, sizeof (uniforms) };
                    sg_apply_uniforms (0, &uniformRange);
                    sg_apply_uniforms (1, &uniformRange);
                    sg_draw (0, static_cast<int> (drawObject.indexCount),
                             static_cast<int>(noteInstances.count));
                    ++submittedDiffractionDrawCount;
                }
            }
            else
            {
                const auto textureSlot = [&] (SceneTextureId id, bool srgb)
                {
                    if (id.isValid())
                        for (std::size_t index = 0; index < scene->textureCount; ++index)
                            if (scene->textures[index].id == id)
                                return 1 + index * 2 + (srgb ? 1 : 0);
                    return std::size_t { 0 };
                };
                std::vector<std::size_t> drawOrder;
                std::vector<std::pair<float, std::size_t>> blendedDraws;
                drawOrder.reserve (scene->objectCount);
                for (std::size_t objectIndex = 0; objectIndex < scene->objectCount; ++objectIndex)
                {
                    const auto* queuedMaterial = visual3d_detail::findById (
                        scene->materials, scene->materialCount, scene->objects[objectIndex].material);
                    if (queuedMaterial == nullptr || queuedMaterial->alphaMode != SceneAlphaMode::Blend)
                        drawOrder.push_back (objectIndex);
                    else
                    {
                        const auto matrix = metalFixtureWorldMatrix (*scene, scene->objects[objectIndex]);
                        const auto dx = matrix[12] - uniforms.cameraTranslation[0];
                        const auto dy = matrix[13] - uniforms.cameraTranslation[1];
                        const auto dz = matrix[14] - uniforms.cameraTranslation[2];
                        blendedDraws.emplace_back (dx * dx + dy * dy + dz * dz, objectIndex);
                    }
                }
                std::stable_sort (blendedDraws.begin(), blendedDraws.end(),
                                  [] (const auto& left, const auto& right)
                                  { return left.first > right.first; });
                for (const auto& blend : blendedDraws) drawOrder.push_back (blend.second);
                for (const auto objectIndex : drawOrder)
                {
                    const auto& drawObject = scene->objects[objectIndex];
                    if (drawObject.indexCount == 0)
                        continue;
                    const auto* drawMaterial = visual3d_detail::findById (
                        scene->materials, scene->materialCount, drawObject.material);
                    SceneTransform3D runtimeTransform {};
                    runtimeTransform.translation = { runtimeInputs.objectTranslationOffset[0],
                        runtimeInputs.objectTranslationOffset[1],
                        runtimeInputs.objectTranslationOffset[2] };
                    runtimeTransform = applyRuntimeObjectTransform (
                        runtimeTransform, runtimeInputs.objectRotationDegrees,
                        runtimeInputs.objectScale);
                    const auto objectMatrix = multiplyMetalFixtureMatrices (
                        metalFixtureTransformMatrix (runtimeTransform),
                        metalFixtureWorldMatrix (*scene, drawObject));
                    std::copy (objectMatrix.begin(), objectMatrix.end(), uniforms.objectMatrix);
                    if (resources->materialProgram == nullptr && drawMaterial != nullptr)
                    {
                        storeVec3 (uniforms.baseColor, drawMaterial->baseColor, drawMaterial->opacity);
                        uniforms.materialParams[0] = drawMaterial->metallic;
                        uniforms.materialParams[1] = drawMaterial->roughness;
                        uniforms.materialParams[2] = drawMaterial->normalScale;
                        uniforms.materialParams[3] = 0.0f;
                        storeVec3 (uniforms.emissive, drawMaterial->emissive);
                        for (std::size_t channel = 0; channel < 3; ++channel)
                            uniforms.emissive[channel] *= runtimeInputs.emissionGain;
                    }
                    const std::array<SceneTextureId, 5> textureIds {
                        drawMaterial != nullptr ? drawMaterial->baseColorTexture : SceneTextureId {},
                        drawMaterial != nullptr ? drawMaterial->metallicRoughnessTexture : SceneTextureId {},
                        drawMaterial != nullptr ? drawMaterial->normalTexture : SceneTextureId {},
                        drawMaterial != nullptr ? drawMaterial->occlusionTexture : SceneTextureId {},
                        drawMaterial != nullptr ? drawMaterial->emissiveTexture : SceneTextureId {}
                    };
                    for (std::size_t unit = 0; unit < textureIds.size(); ++unit)
                    {
                        const auto slot = textureSlot (textureIds[unit], unit == 0 || unit == 4);
                        bindings.views[unit] = resources->textureViews[slot];
                        bindings.samplers[unit] = resources->samplers[slot];
                    }
                    uniforms.materialKind[1] = (textureIds[1].isValid() ? 1u : 0u)
                        | (textureIds[2].isValid() ? 2u : 0u)
                        | (textureIds[3].isValid() ? 4u : 0u)
                        | (textureIds[4].isValid() ? 8u : 0u);
                    const auto alphaMode = drawMaterial != nullptr
                        ? drawMaterial->alphaMode : SceneAlphaMode::Opaque;
                    uniforms.materialKind[2] = alphaMode == SceneAlphaMode::Mask ? 1u : 0u;
                    const auto alphaCutoff = drawMaterial != nullptr ? drawMaterial->alphaCutoff : 0.5f;
                    std::memcpy (&uniforms.materialKind[3], &alphaCutoff, sizeof (alphaCutoff));
                    const auto blend = alphaMode == SceneAlphaMode::Blend ? 1u : 0u;
                    const auto winding = drawMaterial != nullptr && drawMaterial->doubleSided ? 0u
                        : (metalFixtureDeterminant3x3 (objectMatrix) < 0.0f ? 2u : 1u);
                    sg_apply_pipeline (resources->pipelines[blend * 3u + winding]);
                    bindings.vertex_buffer_offsets[0] = static_cast<int> (
                        drawObject.firstVertex * sizeof (SceneVertex));
                    bindings.index_buffer_offset = static_cast<int> (
                        drawObject.firstIndex * sizeof (std::uint32_t));
                    sg_apply_bindings (&bindings);
                    uniforms.geometryInstanceControl[0]
                        = resources->instancedSharedGeometry ? 1u : 0u;
                    if (resources->instancedSharedGeometry)
                    {
                        for (std::size_t noteIndex = 0;
                             noteIndex < noteInstances.count; ++noteIndex)
                        {
                            uniforms.geometryInstanceControl[1]
                                = static_cast<std::uint32_t> (noteIndex);
                            const sg_range uniformRange = { &uniforms, sizeof (uniforms) };
                            sg_apply_uniforms (0, &uniformRange);
                            sg_apply_uniforms (1, &uniformRange);
                            sg_draw (0, static_cast<int> (drawObject.indexCount),
                                     static_cast<int> (scene->objectCount));
                            ++instancedDrawCount;
                            submittedInstanceCount += static_cast<std::uint32_t> (
                                scene->objectCount);
                            if (noteInstances.admitted)
                            {
                                ++noteInstanceDrawCount;
                                submittedNoteInstanceCount += static_cast<std::uint32_t> (
                                    scene->objectCount);
                            }
                        }
                        break;
                    }
                    const sg_range uniformRange = { &uniforms, sizeof (uniforms) };
                    sg_apply_uniforms (0, &uniformRange);
                    sg_apply_uniforms (1, &uniformRange);
                    sg_draw (0, static_cast<int> (drawObject.indexCount),
                             static_cast<int>(noteInstances.count));
                    if (noteInstances.admitted)
                    {
                        ++noteInstanceDrawCount;
                        submittedNoteInstanceCount += static_cast<std::uint32_t> (
                            noteInstances.count);
                    }
            }
            }
            sg_end_pass();
            sg_commit();

            result.rendered = true;
            result.frame = std::move (frame);
            result.stats.drawCount = resources->instancedSharedGeometry ? instancedDrawCount
                : diffractionDraw ? submittedDiffractionDrawCount
                : static_cast<std::size_t>(std::count_if(
                    scene->objects.begin(), scene->objects.begin() + scene->objectCount,
                    [] (const auto& drawObject) { return drawObject.indexCount != 0; }));
            result.stats.ordinaryDrawCount = resources->instancedSharedGeometry
                ? 0u : result.stats.drawCount;
            result.stats.instancedDrawCount = instancedDrawCount;
            result.stats.instanceBufferUploadCount = instanceBufferUploadCount;
            result.stats.submittedInstanceCount = submittedInstanceCount;
            result.stats.noteInstanceDrawCount = diffractionDraw && noteInstances.admitted
                ? result.stats.drawCount : noteInstanceDrawCount;
            result.stats.noteInstanceTransformUploadCount = noteInstances.admitted ? 1u : 0u;
            result.stats.submittedNoteInstanceCount = diffractionDraw && noteInstances.admitted
                ? result.stats.drawCount * static_cast<std::uint32_t>(noteInstances.count)
                : submittedNoteInstanceCount;
            result.stats.materialBytes = sizeof (uniforms) * result.stats.drawCount;
            result.stats.reusedStaticResources = true;
            result.stats.reusedMaterialProgram = resources->materialProgram != nullptr;
            result.stats.diffractionEvaluationBudget = diffractionDraw ? spatialSchedule.maximumEvaluations : 0;
            result.stats.diffractionEvaluationCount = diffractionDraw ? spatialSchedule.requiredEvaluations : 0;
            return result;
        }
    }
};

class MetalDeformationBackend final : public NativeDeformationBackend
{
public:
    BackendInfo info() const override
    {
        return queryNativeBackend();
    }

    NativeDeformationPreparation prepare (
        const std::shared_ptr<const NativeDeformationScene>& source,
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& materialProgram) override
    {
        using namespace HarmonicMIDI::grid;

        NativeDeformationPreparation result;
        const auto available = info();
        if (! available.available || ! available.compute)
        {
            result.error = available.error.empty()
                ? "native Metal deformation requires a compute-capable device"
                : available.error;
            return result;
        }
        if (source == nullptr || source->sourceStableId == 0
            || source->deformationStableId == 0 || source->structuralRevision == 0
            || ! source->clip.isValid() || ! source->mesh.isValid()
            || ! source->object.isValid() || ! source->scene || ! source->deformation
            || ! validateVisual3DScene (*source->scene).valid()
            || source->scene->objectCount != 1 || source->scene->materialCount != 1
            || source->scene->lightCount != 1 || source->scene->cameraCount != 1
            || source->scene->objects[0].id != source->object
            || source->scene->objects[0].firstVertex != 0
            || source->scene->objects[0].vertexCount != source->scene->vertexCount)
        {
            result.error = "native Metal deformation requires one exact bounded scene owner";
            return result;
        }
        const auto* mesh = source->deformation->findMesh (source->mesh);
        if (mesh == nullptr || mesh->vertexCount() != source->scene->vertexCount
            || mesh->vertexCount() == 0
            || mesh->vertexCount() > kNativeDeformationMaxVertices
            || mesh->jointWeightSets().size() > 1
            || mesh->morphTargets().size() > kNativeDeformationMaxMorphTargets)
        {
            result.error = "native Metal deformation mesh exceeds the bounded GPU subset";
            return result;
        }
        for (const auto& target : mesh->morphTargets())
            if (target.hasTangentDeltas())
            {
                result.error = "native Metal deformation does not admit tangent morph deltas";
                return result;
            }

        MetalFixtureSceneBackend fixtureBackend;
        auto fixturePreparation = fixtureBackend.prepare (source->scene, materialProgram);
        if (! fixturePreparation.prepared)
        {
            result.error = fixturePreparation.error;
            return result;
        }
        auto fixture = std::dynamic_pointer_cast<const MetalFixtureSceneResources> (
            fixturePreparation.resources);
        if (fixture == nullptr)
        {
            result.error = "native Metal deformation did not receive fixture GPU resources";
            return result;
        }

        std::lock_guard<std::mutex> lock (sokolMutex());
        @autoreleasepool
        {
            if (! ensureSokolMetal() || sg_query_backend() != SG_BACKEND_METAL_MACOS)
            {
                result.error = gSokolError.empty()
                    ? "native Metal deformation requires the Metal sokol backend"
                    : gSokolError;
                return result;
            }
            auto resources = std::make_shared<MetalDeformationResources>();
            resources->source = source;
            resources->fixture = fixture;
            resources->vertexCount = static_cast<std::uint32_t> (mesh->vertexCount());
            resources->indexCount = source->scene->objects[0].indexCount;
            resources->morphTargetCount
                = static_cast<std::uint32_t> (mesh->morphTargets().size());
            if (mesh->skin().isValid())
            {
                const auto* skin = source->deformation->findSkin (mesh->skin());
                if (skin == nullptr || skin->joints().size() > kNativeDeformationMaxJoints)
                {
                    result.error = "native Metal deformation skin exceeds the bounded GPU subset";
                    return result;
                }
                resources->jointCount = static_cast<std::uint32_t> (skin->joints().size());
            }
            auto fail = [&] (std::string diagnostic)
            {
                resources->destroyUnlocked();
                result.error = std::move (diagnostic);
                return result;
            };

            const auto vertexCount = mesh->vertexCount();
            const auto vertexBytes = vertexCount * sizeof (SceneVertex);
            std::vector<std::uint32_t> jointIndices (vertexCount * 4u, 0u);
            std::vector<float> jointWeights (vertexCount * 4u, 0.0f);
            if (! mesh->jointWeightSets().empty())
            {
                jointIndices = mesh->jointWeightSets()[0].jointIndices();
                jointWeights = mesh->jointWeightSets()[0].weights();
            }
            const auto morphValueCount = std::max<std::size_t> (
                1u, mesh->morphTargets().size() * vertexCount * 3u);
            std::vector<float> morphPositions (morphValueCount, 0.0f);
            std::vector<float> morphNormals (morphValueCount, 0.0f);
            for (std::size_t target = 0; target < mesh->morphTargets().size(); ++target)
            {
                const auto first = target * vertexCount * 3u;
                const auto& positions = mesh->morphTargets()[target].positionDeltas();
                std::copy (positions.begin(), positions.end(), morphPositions.begin() + first);
                const auto& normals = mesh->morphTargets()[target].normalDeltas();
                if (! normals.empty())
                    std::copy (normals.begin(), normals.end(), morphNormals.begin() + first);
            }

            const auto makeStorage = [&] (
                sg_buffer& buffer, sg_view& view, const void* data, std::size_t size,
                bool dynamic, const char* label)
            {
                sg_buffer_desc desc = {};
                desc.usage.storage_buffer = true;
                desc.usage.dynamic_update = dynamic;
                if (data != nullptr && ! dynamic)
                    desc.data = { data, size };
                else
                    desc.size = size;
                desc.label = label;
                buffer = sg_make_buffer (&desc);
                sg_view_desc viewDesc = {};
                viewDesc.storage_buffer.buffer = buffer;
                view = sg_make_view (&viewDesc);
            };
            makeStorage (resources->baseVertices, resources->baseVerticesView,
                source->scene->vertices.data(), vertexBytes, false,
                "arbit-metal-deformation-base-vertices");
            sg_view_desc outputViewDesc = {};
            outputViewDesc.storage_buffer.buffer = fixture->vertexBuffer;
            resources->deformedVerticesView = sg_make_view (&outputViewDesc);
            makeStorage (resources->jointIndices, resources->jointIndicesView,
                jointIndices.data(), jointIndices.size() * sizeof (std::uint32_t), false,
                "arbit-metal-deformation-joint-indices");
            makeStorage (resources->jointWeights, resources->jointWeightsView,
                jointWeights.data(), jointWeights.size() * sizeof (float), false,
                "arbit-metal-deformation-joint-weights");
            makeStorage (resources->morphPositions, resources->morphPositionsView,
                morphPositions.data(), morphPositions.size() * sizeof (float), false,
                "arbit-metal-deformation-morph-positions");
            makeStorage (resources->morphNormals, resources->morphNormalsView,
                morphNormals.data(), morphNormals.size() * sizeof (float), false,
                "arbit-metal-deformation-morph-normals");
            makeStorage (resources->jointPalette, resources->jointPaletteView,
                nullptr, kNativeDeformationMaxJoints * 16u * sizeof (float), true,
                "arbit-metal-deformation-joint-palette");
            makeStorage (resources->morphWeights, resources->morphWeightsView,
                nullptr, kNativeDeformationMaxMorphTargets * sizeof (float), true,
                "arbit-metal-deformation-morph-weights");

            NSError* deformationShaderError = nil;
            id<MTLLibrary> deformationShaderLibrary = [gMetalDevice
                newLibraryWithSource:[NSString stringWithUTF8String:kDeformationComputeShader]
                options:nil
                error:&deformationShaderError];
            if (deformationShaderLibrary == nil)
            {
                const auto* diagnostic = deformationShaderError.localizedDescription.UTF8String;
                return fail ("native Metal deformation compute source rejected: "
                    + std::string (diagnostic != nullptr ? diagnostic : "unknown Metal error"));
            }
#if ! __has_feature(objc_arc)
            [deformationShaderLibrary release];
#endif

            sg_shader_desc shaderDesc = {};
            shaderDesc.compute_func.source = kDeformationComputeShader;
            shaderDesc.mtl_threads_per_threadgroup = { 64, 1, 1 };
            shaderDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_COMPUTE;
            shaderDesc.uniform_blocks[0].size = sizeof (DeformationComputeUniforms);
            shaderDesc.uniform_blocks[0].msl_buffer_n = 0;
            for (int view = 0; view < 8; ++view)
            {
                shaderDesc.views[view].storage_buffer.stage = SG_SHADERSTAGE_COMPUTE;
                shaderDesc.views[view].storage_buffer.readonly = view != 1;
                shaderDesc.views[view].storage_buffer.msl_buffer_n = 8 + view;
            }
            shaderDesc.label = "arbit-metal-deformation-compute-shader";
            resources->computeShader = sg_make_shader (&shaderDesc);
            sg_pipeline_desc pipelineDesc = {};
            pipelineDesc.compute = true;
            pipelineDesc.shader = resources->computeShader;
            pipelineDesc.label = "arbit-metal-deformation-compute-pipeline";
            resources->computePipeline = sg_make_pipeline (&pipelineDesc);

            if (! resourceValid (sg_query_buffer_state (resources->baseVertices)))
                return fail ("native Metal deformation base vertex buffer creation failed");
            if (! resourceValid (sg_query_view_state (resources->baseVerticesView)))
                return fail ("native Metal deformation base vertex view creation failed");
            if (! resourceValid (sg_query_view_state (resources->deformedVerticesView)))
                return fail ("native Metal deformation output vertex view creation failed");
            if (! resourceValid (sg_query_buffer_state (resources->jointIndices))
                || ! resourceValid (sg_query_view_state (resources->jointIndicesView)))
                return fail ("native Metal deformation joint index resource creation failed");
            if (! resourceValid (sg_query_buffer_state (resources->jointWeights))
                || ! resourceValid (sg_query_view_state (resources->jointWeightsView)))
                return fail ("native Metal deformation joint weight resource creation failed");
            if (! resourceValid (sg_query_buffer_state (resources->morphPositions))
                || ! resourceValid (sg_query_view_state (resources->morphPositionsView)))
                return fail ("native Metal deformation morph position resource creation failed");
            if (! resourceValid (sg_query_buffer_state (resources->morphNormals))
                || ! resourceValid (sg_query_view_state (resources->morphNormalsView)))
                return fail ("native Metal deformation morph normal resource creation failed");
            if (! resourceValid (sg_query_buffer_state (resources->jointPalette))
                || ! resourceValid (sg_query_view_state (resources->jointPaletteView)))
                return fail ("native Metal deformation joint palette resource creation failed");
            if (! resourceValid (sg_query_buffer_state (resources->morphWeights))
                || ! resourceValid (sg_query_view_state (resources->morphWeightsView)))
                return fail ("native Metal deformation morph weight resource creation failed");
            if (! resourceValid (sg_query_shader_state (resources->computeShader)))
                return fail ("native Metal deformation compute shader creation failed: "
                    + lastSokolLog());
            if (! resourceValid (sg_query_pipeline_state (resources->computePipeline)))
                return fail ("native Metal deformation compute pipeline creation failed");

            result.prepared = true;
            result.resources = std::move (resources);
            result.stats.staticUploadCount = 1;
            result.stats.staticVertexBytes = vertexBytes * 2u;
            result.stats.staticDeformationBytes
                = jointIndices.size() * sizeof (std::uint32_t)
                + jointWeights.size() * sizeof (float)
                + morphPositions.size() * sizeof (float)
                + morphNormals.size() * sizeof (float);
            result.stats.sourceStableId = source->sourceStableId;
            result.stats.deformationStableId = source->deformationStableId;
            result.stats.clipId = source->clip.value;
            result.stats.meshId = source->mesh.value;
            result.stats.skinId = mesh->skin().value;
            result.stats.revision = source->structuralRevision;
            return result;
        }
    }

    NativeDeformationSubmission render (
        const std::shared_ptr<const NativeDeformationScene>& source,
        const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>& snapshot,
        const std::shared_ptr<const NativeDeformationResources>& nativeResources,
        std::uint32_t width,
        std::uint32_t height,
        NativeDeformationRuntimeInputs runtimeInputs) override
    {
        NativeDeformationSubmission result;
        const auto resources = std::dynamic_pointer_cast<const MetalDeformationResources> (
            nativeResources);
        if (source == nullptr || snapshot == nullptr || resources == nullptr
            || resources->source != source || resources->fixture == nullptr)
        {
            result.error = "native Metal deformation requires exact prepared owner resources";
            return result;
        }
        std::lock_guard<std::mutex> submissionLock (resources->submissionMutex);
        NativeDeformationFrameData frameData;
        if (! prepareNativeDeformationFrame (*source, *snapshot, frameData, result.error))
            return result;
        for (std::size_t index = 0; index < frameData.morphTargetCount; ++index)
        {
            const auto base = source->morphBaseWeights.empty()
                ? 0.0f : source->morphBaseWeights[index];
            frameData.morphWeights[index]
                = base + (frameData.morphWeights[index] - base) * runtimeInputs.morphWeight;
        }

        {
            std::lock_guard<std::mutex> lock (sokolMutex());
            @autoreleasepool
            {
                if (! ensureSokolMetal() || sg_query_backend() != SG_BACKEND_METAL_MACOS)
                {
                    result.error = gSokolError.empty()
                        ? "native Metal deformation requires the Metal sokol backend"
                        : gSokolError;
                    return result;
                }
                const sg_range paletteRange = {
                    frameData.jointPalette.data(), frameData.jointPalette.size() * sizeof (float) };
                const sg_range weightRange = {
                    frameData.morphWeights.data(), frameData.morphWeights.size() * sizeof (float) };
                sg_update_buffer (resources->jointPalette, &paletteRange);
                sg_update_buffer (resources->morphWeights, &weightRange);
                DeformationComputeUniforms uniforms;
                uniforms.vertexCount = frameData.vertexCount;
                uniforms.jointCount = frameData.jointCount;
                uniforms.morphTargetCount = frameData.morphTargetCount;
                sg_pass pass = {};
                pass.compute = true;
                pass.label = "arbit-metal-deformation-compute-pass";
                sg_begin_pass (&pass);
                sg_apply_pipeline (resources->computePipeline);
                sg_bindings bindings = {};
                bindings.views[0] = resources->baseVerticesView;
                bindings.views[1] = resources->deformedVerticesView;
                bindings.views[2] = resources->jointIndicesView;
                bindings.views[3] = resources->jointWeightsView;
                bindings.views[4] = resources->morphPositionsView;
                bindings.views[5] = resources->morphNormalsView;
                bindings.views[6] = resources->jointPaletteView;
                bindings.views[7] = resources->morphWeightsView;
                sg_apply_bindings (&bindings);
                const sg_range uniformRange = { &uniforms, sizeof (uniforms) };
                sg_apply_uniforms (0, &uniformRange);
                sg_dispatch (static_cast<int> ((frameData.vertexCount + 63u) / 64u), 1, 1);
                sg_end_pass();
                sg_commit();
            }
        }

        auto fixtureSubmission = nativeFixtureSceneBackend().render (
            source->scene, resources->fixture, width, height, runtimeInputs);
        if (! fixtureSubmission.rendered)
        {
            result.error = fixtureSubmission.error;
            return result;
        }
        const auto* mesh = source->deformation->findMesh (source->mesh);
        result.rendered = true;
        result.frame = std::move (fixtureSubmission.frame);
        result.stats.drawCount = fixtureSubmission.stats.drawCount;
        result.stats.dispatchCount = 1;
        result.stats.reusedStaticResources = true;
        result.stats.dynamicUniformBytes
            = static_cast<std::uint64_t> (frameData.jointCount) * 16u * sizeof (float)
            + static_cast<std::uint64_t> (frameData.morphTargetCount) * sizeof (float);
        result.stats.sourceStableId = source->sourceStableId;
        result.stats.deformationStableId = source->deformationStableId;
        result.stats.clipId = source->clip.value;
        result.stats.meshId = source->mesh.value;
        result.stats.skinId = mesh != nullptr ? mesh->skin().value : 0;
        result.stats.revision = source->structuralRevision;
        result.stats.time = snapshot->time();
        return result;
    }
};
} // namespace

BackendInfo queryNativeBackend()
{
    BackendInfo result;
    @autoreleasepool
    {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil)
        {
            result.error = "Metal returned no default device";
            return result;
        }

        result.available = true;
        result.compute = true;
        result.backend = "metal";
        result.device = deviceName (device);
    }
    return result;
}

BackendSelfTest runNativeBackendSelfTest()
{
    std::lock_guard<std::mutex> lock (sokolMutex());
    BackendSelfTest result;

    @autoreleasepool
    {
        if (! ensureSokolMetal())
        {
            result.error = gSokolError;
            return result;
        }

        result.available = true;
        result.backend = "metal";
        result.device = deviceName (gMetalDevice);

        result.compute = sg_query_features().compute;
        if (! result.compute || sg_query_backend() != SG_BACKEND_METAL_MACOS)
        {
            result.error = "sokol_gfx did not select a compute-capable macOS Metal backend";
            return result;
        }

        const std::array<uint32_t, 4> initial = { 1u, 2u, 3u, 4u };
        sg_buffer_desc bufferDesc = {};
        bufferDesc.usage.storage_buffer = true;
        bufferDesc.data.ptr = initial.data();
        bufferDesc.data.size = sizeof (initial);
        bufferDesc.label = "arbit-metal-selftest-buffer";
        const sg_buffer buffer = sg_make_buffer (&bufferDesc);

        sg_view_desc storageViewDesc = {};
        storageViewDesc.storage_buffer.buffer = buffer;
        const sg_view storageView = sg_make_view (&storageViewDesc);

        static const char* computeSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Values { uint value[4]; };
kernel void _main(device Values& values [[buffer(8)]],
                  uint index [[thread_position_in_grid]])
{
    if (index < 4) values.value[index] = values.value[index] * 3u + 7u;
}
)metal";

        sg_shader_desc computeShaderDesc = {};
        computeShaderDesc.compute_func.source = computeSource;
        computeShaderDesc.mtl_threads_per_threadgroup.x = 4;
        computeShaderDesc.mtl_threads_per_threadgroup.y = 1;
        computeShaderDesc.mtl_threads_per_threadgroup.z = 1;
        computeShaderDesc.views[0].storage_buffer.stage = SG_SHADERSTAGE_COMPUTE;
        computeShaderDesc.views[0].storage_buffer.readonly = false;
        computeShaderDesc.views[0].storage_buffer.msl_buffer_n = 8;
        computeShaderDesc.label = "arbit-metal-selftest-compute-shader";
        const sg_shader computeShader = sg_make_shader (&computeShaderDesc);

        sg_pipeline_desc computePipelineDesc = {};
        computePipelineDesc.compute = true;
        computePipelineDesc.shader = computeShader;
        computePipelineDesc.label = "arbit-metal-selftest-compute-pipeline";
        const sg_pipeline computePipeline = sg_make_pipeline (&computePipelineDesc);

        sg_image_desc imageDesc = {};
        imageDesc.usage.color_attachment = true;
        imageDesc.width = 4;
        imageDesc.height = 4;
        imageDesc.pixel_format = SG_PIXELFORMAT_RGBA8;
        imageDesc.sample_count = 1;
        imageDesc.label = "arbit-metal-selftest-image";
        const sg_image image = sg_make_image (&imageDesc);

        sg_view_desc colorViewDesc = {};
        colorViewDesc.color_attachment.image = image;
        const sg_view colorView = sg_make_view (&colorViewDesc);

        static const char* vertexSource = R"metal(
#include <metal_stdlib>
using namespace metal;
struct VertexOut { float4 position [[position]]; };
vertex VertexOut _main(uint vertexId [[vertex_id]])
{
    const float2 positions[3] = { float2(-1.0, -1.0),
                                  float2( 3.0, -1.0),
                                  float2(-1.0,  3.0) };
    VertexOut out;
    out.position = float4(positions[vertexId], 0.0, 1.0);
    return out;
}
)metal";
        static const char* fragmentSource = R"metal(
#include <metal_stdlib>
using namespace metal;
fragment float4 _main() { return float4(0.25, 0.5, 0.75, 1.0); }
)metal";

        sg_shader_desc renderShaderDesc = {};
        renderShaderDesc.vertex_func.source = vertexSource;
        renderShaderDesc.fragment_func.source = fragmentSource;
        renderShaderDesc.label = "arbit-metal-selftest-render-shader";
        const sg_shader renderShader = sg_make_shader (&renderShaderDesc);

        sg_pipeline_desc renderPipelineDesc = {};
        renderPipelineDesc.shader = renderShader;
        renderPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA8;
        renderPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        renderPipelineDesc.label = "arbit-metal-selftest-render-pipeline";
        const sg_pipeline renderPipeline = sg_make_pipeline (&renderPipelineDesc);

        const bool resourcesOk = resourceValid (sg_query_buffer_state (buffer))
            && resourceValid (sg_query_view_state (storageView))
            && resourceValid (sg_query_shader_state (computeShader))
            && resourceValid (sg_query_pipeline_state (computePipeline))
            && resourceValid (sg_query_image_state (image))
            && resourceValid (sg_query_view_state (colorView))
            && resourceValid (sg_query_shader_state (renderShader))
            && resourceValid (sg_query_pipeline_state (renderPipeline));
        if (! resourcesOk)
        {
            result.error = "Metal self-test resource creation failed";
            sg_destroy_pipeline (renderPipeline);
            sg_destroy_shader (renderShader);
            sg_destroy_view (colorView);
            sg_destroy_image (image);
            sg_destroy_pipeline (computePipeline);
            sg_destroy_shader (computeShader);
            sg_destroy_view (storageView);
            sg_destroy_buffer (buffer);
            return result;
        }

        sg_pass computePass = {};
        computePass.compute = true;
        computePass.label = "arbit-metal-selftest-compute-pass";
        sg_begin_pass (&computePass);
        sg_apply_pipeline (computePipeline);
        sg_bindings computeBindings = {};
        computeBindings.views[0] = storageView;
        sg_apply_bindings (&computeBindings);
        sg_dispatch (1, 1, 1);
        sg_end_pass();

        sg_pass renderPass = {};
        renderPass.attachments.colors[0] = colorView;
        renderPass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
        renderPass.action.colors[0].store_action = SG_STOREACTION_STORE;
        renderPass.action.colors[0].clear_value = { 0.0f, 0.0f, 0.0f, 1.0f };
        renderPass.label = "arbit-metal-selftest-render-pass";
        sg_begin_pass (&renderPass);
        sg_apply_pipeline (renderPipeline);
        sg_draw (0, 3, 1);
        sg_end_pass();
        sg_commit();

        const sg_mtl_buffer_info nativeBuffer = sg_mtl_query_buffer_info (buffer);
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>) nativeBuffer.buf[nativeBuffer.active_slot];
        std::array<uint32_t, 4> computed = {};
        const sg_mtl_image_info nativeImage = sg_mtl_query_image_info (image);
        id<MTLTexture> mtlImage = (__bridge id<MTLTexture>) nativeImage.tex[nativeImage.active_slot];
        std::array<uint8_t, 4 * 4 * 4> pixels = {};
        readBackSubmittedWork (gMetalDevice, mtlBuffer, mtlImage, computed, pixels);

        const std::array<uint32_t, 4> expected = { 10u, 13u, 16u, 19u };
        result.computeChecksum = fnv1a (computed.data(), sizeof (computed));
        result.computePassed = computed == expected;
        result.renderChecksum = fnv1a (pixels.data(), pixels.size());
        result.renderPassed = pixels[0] >= 63 && pixels[0] <= 64
            && pixels[1] >= 127 && pixels[1] <= 128
            && pixels[2] >= 191 && pixels[2] <= 192
            && pixels[3] == 255;

        sg_destroy_pipeline (renderPipeline);
        sg_destroy_shader (renderShader);
        sg_destroy_view (colorView);
        sg_destroy_image (image);
        sg_destroy_pipeline (computePipeline);
        sg_destroy_shader (computeShader);
        sg_destroy_view (storageView);
        sg_destroy_buffer (buffer);

        if (! result.computePassed)
            result.error = "Metal compute result mismatch";
        else if (! result.renderPassed)
            result.error = "Metal offscreen render result mismatch";
    }

    return result;
}

NativeSdfExecutionCapabilities queryNativeSdfExecution()
{
    return nativeSdfExecutionBackend().capabilities();
}

NativeSdfExecutionBackend& nativeSdfExecutionBackend()
{
    static MetalSdfExecutionBackend backend;
    return backend;
}

void invalidateNativeSdfExecutionContext (std::uintptr_t) noexcept
{
}

RenderPassOutputBackend& nativeRenderPassOutputBackend()
{
    static MetalRenderPassOutputBackend backend;
    return backend;
}

NativeOpticalFlowExecutionBackend& nativeOpticalFlowExecutionBackend()
{
    static auto* backend = new MetalOpticalFlowExecutionBackend();
    return *backend;
}

NativeFixtureSceneBackend& nativeFixtureSceneBackend()
{
    static MetalFixtureSceneBackend backend;
    return backend;
}

NativeDeformationBackend& nativeDeformationBackend()
{
    static MetalDeformationBackend backend;
    return backend;
}

} // namespace arbitgpu

#if ARBIT_HAVE_VIEWPORT
namespace
{

void iosurfaceSetInt (CFMutableDictionaryRef dict, CFStringRef key, int32_t value)
{
    CFNumberRef number = CFNumberCreate (kCFAllocatorDefault, kCFNumberSInt32Type, &value);
    CFDictionarySetValue (dict, key, number);
    CFRelease (number);
}

struct MetalComputeParams
{
    int32_t count;
    int32_t spawnTrack;
    float gravity;
    float force;
    float dt;
    int32_t frame;
    int32_t noteCount;
    float aspect;
    float lifetime;
    float padding[3];
};
static_assert (sizeof (MetalComputeParams) == 48, "MSL compute uniform layout changed");

struct MetalDrawParams
{
    float pointSize;
    float padding[3];
    float color[4];
};
static_assert (sizeof (MetalDrawParams) == 32, "MSL draw uniform layout changed");

struct MetalParticle
{
    float pos[2];
    float vel[2];
    float life;
    float maxLife;
    float hue;
    float padding;
};
static_assert (sizeof (MetalParticle) == 32, "MSL particle layout changed");

const char* kMetalParticleCompute = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Params {
    int count; int spawnTrack; float gravity; float force;
    float dt; int frame; int noteCount; float aspect;
    // Scalar padding keeps this MSL block byte-identical to the 48-byte C++ wire
    // block. float3 would align to 16 and silently make Params 64 bytes.
    float lifetime; float padding0; float padding1; float padding2;
};
struct Particle {
    float2 pos; float2 vel; float life; float maxLife; float hue; float pad;
};
float hash11(uint n) {
    n = (n << 13u) ^ n;
    n = n * (n * n * 15731u + 789221u) + 1376312589u;
    return float(n & 0x7fffffffu) / float(0x7fffffffu);
}
kernel void _main(constant Params& u [[buffer(0)]],
                  device Particle* particles [[buffer(8)]],
                  const device float4* notes [[buffer(9)]],
                  uint i [[thread_position_in_grid]])
{
    if (i >= uint(u.count)) return;
    Particle p = particles[i];
    p.life -= u.dt / max(p.maxLife, 1.0e-3f);
    if (p.life <= 0.0f) {
        int matches = 0;
        const int rows = min(u.noteCount, 128);
        for (int row = 0; row < rows; ++row) {
            const float4 t0 = notes[row * 4];
            const float4 t1 = notes[row * 4 + 1];
            if (int(t1.z + 0.5f) == u.spawnTrack && t0.y > 0.001f) ++matches;
        }
        if (matches > 0) {
            int pick = min(int(hash11(i * 747u + uint(u.frame) * 13u) * float(matches)),
                           matches - 1);
            int seen = 0;
            int chosen = -1;
            for (int row = 0; row < rows; ++row) {
                const float4 t0 = notes[row * 4];
                const float4 t1 = notes[row * 4 + 1];
                if (int(t1.z + 0.5f) == u.spawnTrack && t0.y > 0.001f) {
                    if (seen == pick) { chosen = row; break; }
                    ++seen;
                }
            }
            if (chosen >= 0) {
                const float4 t0 = notes[chosen * 4];
                const float midi = t0.x;
                const float velocity = t0.y;
                const float px = clamp((midi - 36.0f) / 60.0f, 0.0f, 1.0f) * 0.8f + 0.1f;
                p.pos = float2(px, 0.12f);
                const float angle = (hash11(i * 31u + uint(u.frame)) - 0.5f) * 2.2f;
                const float speed = (0.25f + velocity * 0.75f) * u.force;
                p.vel = float2(sin(angle) * speed / max(u.aspect, 1.0e-3f), cos(angle) * speed);
                p.maxLife = u.lifetime > 0.0f ? u.lifetime : 0.6f + hash11(i * 97u) * 1.2f;
                p.life = 1.0f;
                p.hue = fract(midi / 12.0f);
            } else {
                p.life = 0.0f; p.pos = float2(-10.0f);
            }
        } else {
            p.life = 0.0f; p.pos = float2(-10.0f);
        }
    } else {
        p.vel.y -= u.gravity * u.dt;
        p.pos += p.vel * u.dt;
    }
    particles[i] = p;
}
)metal";

const char* kMetalParticleVertex = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Particle {
    float2 pos; float2 vel; float life; float maxLife; float hue; float pad;
};
struct DrawParams {
    float pointSize; float padding0; float padding1; float padding2; float4 color;
};
struct VertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
    float life [[user(locn0)]];
    float hue [[user(locn1)]];
    float4 color [[user(locn2)]];
};
vertex VertexOut _main(const device Particle* particles [[buffer(8)]],
                       constant DrawParams& u [[buffer(0)]],
                       uint index [[vertex_id]])
{
    const Particle p = particles[index];
    VertexOut out;
    out.life = clamp(p.life, 0.0f, 1.0f);
    out.hue = p.hue;
    out.color = u.color;
    if (p.life <= 0.0f) {
        out.position = float4(-2.0f, -2.0f, 0.0f, 1.0f);
        out.pointSize = 1.0f;
    } else {
        out.position = float4(p.pos * 2.0f - 1.0f, 0.0f, 1.0f);
        out.pointSize = max(1.0f, u.pointSize * (0.5f + out.life * 0.8f));
    }
    return out;
}
)metal";

const char* kMetalParticleFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct VertexOut {
    float4 position [[position]];
    float life [[user(locn0)]];
    float hue [[user(locn1)]];
    float4 color [[user(locn2)]];
};
float3 hsv2rgb(float h) {
    const float3 c = abs(fract(h + float3(0.0f, 0.6667f, 0.3333f)) * 6.0f - 3.0f) - 1.0f;
    return clamp(c, 0.0f, 1.0f);
}
fragment float4 _main(VertexOut in [[stage_in]], float2 pointCoord [[point_coord]])
{
    const float radius = length(pointCoord - float2(0.5f));
    if (radius > 0.5f) discard_fragment();
    const float alpha = (1.0f - smoothstep(0.0f, 0.5f, radius)) * in.life;
    const float3 rgb = in.color.r >= 0.0f ? in.color.rgb : hsv2rgb(in.hue);
    return float4(rgb, alpha * in.color.a);
}
)metal";

} // namespace

namespace videorender
{

struct MetalParticleEngine::Impl
{
    std::string error;
    bool programsReady = false;
    int poolCount = 0;
    int outWidth = 0;
    int outHeight = 0;
    bool simSeeded = false;
    int lastSimFrame = 0;

    sg_buffer particleBuffer = {};
    sg_view particleView = {};
    sg_buffer notesBuffer = {};
    sg_view notesView = {};
    sg_shader computeShader = {};
    sg_pipeline computePipeline = {};
    sg_shader drawShader = {};
    sg_pipeline drawPipeline = {};
    sg_image outputImage = {};
    sg_view outputView = {};
    sg_view outputTextureView = {};
    bool nativeTarget = false;
    ParticleDiagnostics diagnostics;

    IOSurfaceRef surface = nullptr;
    id<MTLTexture> metalTexture = nil;
    unsigned rectangleTexture = 0;
    unsigned rectangleFbo = 0;
    unsigned outputTexture = 0;
    unsigned outputFbo = 0;

    bool ensurePrograms()
    {
        if (programsReady)
            return true;

        sg_shader_desc computeDesc = {};
        computeDesc.compute_func.source = kMetalParticleCompute;
        computeDesc.mtl_threads_per_threadgroup = { 256, 1, 1 };
        computeDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_COMPUTE;
        computeDesc.uniform_blocks[0].size = sizeof (MetalComputeParams);
        computeDesc.uniform_blocks[0].msl_buffer_n = 0;
        computeDesc.views[0].storage_buffer.stage = SG_SHADERSTAGE_COMPUTE;
        computeDesc.views[0].storage_buffer.readonly = false;
        computeDesc.views[0].storage_buffer.msl_buffer_n = 8;
        computeDesc.views[1].storage_buffer.stage = SG_SHADERSTAGE_COMPUTE;
        computeDesc.views[1].storage_buffer.readonly = true;
        computeDesc.views[1].storage_buffer.msl_buffer_n = 9;
        computeDesc.label = "arbit-metal-particle-compute";
        computeShader = sg_make_shader (&computeDesc);

        sg_pipeline_desc computePipelineDesc = {};
        computePipelineDesc.compute = true;
        computePipelineDesc.shader = computeShader;
        computePipelineDesc.label = "arbit-metal-particle-compute-pipeline";
        computePipeline = sg_make_pipeline (&computePipelineDesc);

        sg_shader_desc drawDesc = {};
        drawDesc.vertex_func.source = kMetalParticleVertex;
        drawDesc.fragment_func.source = kMetalParticleFragment;
        drawDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
        drawDesc.uniform_blocks[0].size = sizeof (MetalDrawParams);
        drawDesc.uniform_blocks[0].msl_buffer_n = 0;
        drawDesc.views[0].storage_buffer.stage = SG_SHADERSTAGE_VERTEX;
        drawDesc.views[0].storage_buffer.readonly = true;
        drawDesc.views[0].storage_buffer.msl_buffer_n = 8;
        drawDesc.label = "arbit-metal-particle-draw";
        drawShader = sg_make_shader (&drawDesc);

        sg_pipeline_desc drawPipelineDesc = {};
        drawPipelineDesc.shader = drawShader;
        drawPipelineDesc.primitive_type = SG_PRIMITIVETYPE_POINTS;
        drawPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_BGRA8;
        drawPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        drawPipelineDesc.colors[0].blend.enabled = true;
        drawPipelineDesc.colors[0].blend.src_factor_rgb = SG_BLENDFACTOR_SRC_ALPHA;
        drawPipelineDesc.colors[0].blend.dst_factor_rgb = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
        drawPipelineDesc.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_SRC_ALPHA;
        drawPipelineDesc.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
        drawPipelineDesc.label = "arbit-metal-particle-draw-pipeline";
        drawPipeline = sg_make_pipeline (&drawPipelineDesc);

        sg_buffer_desc notesDesc = {};
        notesDesc.size = 4u * 128u * 4u * sizeof (float);
        notesDesc.usage.storage_buffer = true;
        notesDesc.usage.dynamic_update = true;
        notesDesc.label = "arbit-metal-particle-notes";
        notesBuffer = sg_make_buffer (&notesDesc);
        sg_view_desc notesViewDesc = {};
        notesViewDesc.storage_buffer.buffer = notesBuffer;
        notesView = sg_make_view (&notesViewDesc);

        programsReady = resourceValid (sg_query_shader_state (computeShader))
            && resourceValid (sg_query_pipeline_state (computePipeline))
            && resourceValid (sg_query_shader_state (drawShader))
            && resourceValid (sg_query_pipeline_state (drawPipeline))
            && resourceValid (sg_query_buffer_state (notesBuffer))
            && resourceValid (sg_query_view_state (notesView));
        if (! programsReady)
            error = "Metal particle shader/pipeline creation failed";
        return programsReady;
    }

    bool ensurePool (int count)
    {
        if (particleBuffer.id != 0 && poolCount == count)
            return true;
        if (particleView.id != 0) sg_destroy_view (particleView);
        if (particleBuffer.id != 0) sg_destroy_buffer (particleBuffer);
        particleView = {};
        particleBuffer = {};
        poolCount = count;
        simSeeded = false;
        lastSimFrame = 0;

        const std::vector<MetalParticle> initial (static_cast<size_t> (count));
        sg_buffer_desc desc = {};
        desc.usage.storage_buffer = true;
        desc.data.ptr = initial.data();
        desc.data.size = initial.size() * sizeof (MetalParticle);
        desc.label = "arbit-metal-particle-pool";
        particleBuffer = sg_make_buffer (&desc);
        sg_view_desc viewDesc = {};
        viewDesc.storage_buffer.buffer = particleBuffer;
        particleView = sg_make_view (&viewDesc);
        if (! resourceValid (sg_query_buffer_state (particleBuffer))
            || ! resourceValid (sg_query_view_state (particleView)))
        {
            error = "Metal particle storage-buffer creation failed";
            return false;
        }
        return true;
    }

    void destroyTarget (const arbitgl::GlFuncs* gl)
    {
        if (outputTextureView.id != 0) sg_destroy_view (outputTextureView);
        if (outputView.id != 0) sg_destroy_view (outputView);
        if (outputImage.id != 0) sg_destroy_image (outputImage);
        outputView = {};
        outputTextureView = {};
        outputImage = {};
        if (rectangleFbo != 0 && gl != nullptr) gl->DeleteFramebuffers (1, &rectangleFbo);
        if (outputFbo != 0 && gl != nullptr) gl->DeleteFramebuffers (1, &outputFbo);
        if (rectangleTexture != 0) glDeleteTextures (1, &rectangleTexture);
        if (outputTexture != 0) glDeleteTextures (1, &outputTexture);
        rectangleFbo = outputFbo = rectangleTexture = outputTexture = 0;
#if ! __has_feature(objc_arc)
        [metalTexture release];
#endif
        metalTexture = nil;
        if (surface != nullptr) CFRelease (surface);
        surface = nullptr;
        nativeTarget = false;
        outWidth = outHeight = 0;
    }

    bool ensureTarget (const arbitgl::GlFuncs* gl, int width, int height,
                       bool nativeOnly)
    {
        if (outputImage.id != 0 && outWidth == width && outHeight == height
            && nativeTarget == nativeOnly)
            return true;
        destroyTarget (gl);

        if (nativeOnly)
        {
            sg_image_desc imageDesc = {};
            imageDesc.usage.color_attachment = true;
            imageDesc.width = width;
            imageDesc.height = height;
            imageDesc.pixel_format = SG_PIXELFORMAT_BGRA8;
            imageDesc.sample_count = 1;
            imageDesc.label = "arbit-metal-particle-native-output";
            outputImage = sg_make_image (&imageDesc);
            sg_view_desc attachmentDesc = {};
            attachmentDesc.color_attachment.image = outputImage;
            outputView = sg_make_view (&attachmentDesc);
            sg_view_desc textureDesc = {};
            textureDesc.texture.image = outputImage;
            outputTextureView = sg_make_view (&textureDesc);
            if (! resourceValid (sg_query_image_state (outputImage))
                || ! resourceValid (sg_query_view_state (outputView))
                || ! resourceValid (sg_query_view_state (outputTextureView)))
            {
                error = "Metal particle native target creation failed";
                destroyTarget (gl);
                return false;
            }
            nativeTarget = true;
            outWidth = width;
            outHeight = height;
            return true;
        }

        CFMutableDictionaryRef props = CFDictionaryCreateMutable (
            kCFAllocatorDefault, 0,
            &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        iosurfaceSetInt (props, kIOSurfaceWidth, width);
        iosurfaceSetInt (props, kIOSurfaceHeight, height);
        iosurfaceSetInt (props, kIOSurfaceBytesPerElement, 4);
        iosurfaceSetInt (props, kIOSurfacePixelFormat, static_cast<int32_t> ('BGRA'));
        surface = IOSurfaceCreate (props);
        CFRelease (props);
        if (surface == nullptr)
        {
            error = "Metal particle IOSurface creation failed";
            return false;
        }

        MTLTextureDescriptor* descriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                               width:width
                                                              height:height
                                                           mipmapped:NO];
        descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        metalTexture = [gMetalDevice newTextureWithDescriptor:descriptor iosurface:surface plane:0];
        if (metalTexture == nil)
        {
            error = "Metal particle IOSurface texture creation failed";
            destroyTarget (gl);
            return false;
        }

        sg_image_desc imageDesc = {};
        imageDesc.usage.color_attachment = true;
        imageDesc.width = width;
        imageDesc.height = height;
        imageDesc.pixel_format = SG_PIXELFORMAT_BGRA8;
        imageDesc.sample_count = 1;
#if ! __has_feature(objc_arc)
        // sokol's Metal resource-pool insertion balances one retain after it
        // stores an injected texture. Preserve this owner's +1 from
        // newTextureWithDescriptor so destroyTarget can release it safely.
        [metalTexture retain];
#endif
        imageDesc.mtl_textures[0] = (__bridge const void*) metalTexture;
        imageDesc.label = "arbit-metal-particle-output";
        outputImage = sg_make_image (&imageDesc);
        sg_view_desc viewDesc = {};
        viewDesc.color_attachment.image = outputImage;
        outputView = sg_make_view (&viewDesc);
        sg_view_desc textureViewDesc = {};
        textureViewDesc.texture.image = outputImage;
        outputTextureView = sg_make_view (&textureViewDesc);
        if (! resourceValid (sg_query_image_state (outputImage))
            || ! resourceValid (sg_query_view_state (outputView))
            || ! resourceValid (sg_query_view_state (outputTextureView)))
        {
            error = "Metal particle output attachment creation failed";
            destroyTarget (gl);
            return false;
        }

        CGLContextObj cgl = CGLGetCurrentContext();
        if (cgl == nullptr)
        {
            error = "Metal particle bridge has no current CGL context";
            destroyTarget (gl);
            return false;
        }
        glGenTextures (1, &rectangleTexture);
        glBindTexture (GL_TEXTURE_RECTANGLE, rectangleTexture);
        const CGLError cglError = CGLTexImageIOSurface2D (
            cgl, GL_TEXTURE_RECTANGLE, GL_RGBA8, width, height,
            GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, surface, 0);
        if (cglError != kCGLNoError)
        {
            error = std::string ("Metal particle CGL IOSurface import failed: ")
                  + CGLErrorString (cglError);
            glBindTexture (GL_TEXTURE_RECTANGLE, 0);
            destroyTarget (gl);
            return false;
        }
        glTexParameteri (GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri (GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture (GL_TEXTURE_RECTANGLE, 0);

        gl->GenFramebuffers (1, &rectangleFbo);
        gl->BindFramebuffer (GL_FRAMEBUFFER, rectangleFbo);
        gl->FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_TEXTURE_RECTANGLE, rectangleTexture, 0);
        if (gl->CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            error = "Metal particle IOSurface GL framebuffer incomplete";
            gl->BindFramebuffer (GL_FRAMEBUFFER, 0);
            destroyTarget (gl);
            return false;
        }

        glGenTextures (1, &outputTexture);
        glBindTexture (GL_TEXTURE_2D, outputTexture);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                      GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        gl->GenFramebuffers (1, &outputFbo);
        gl->BindFramebuffer (GL_FRAMEBUFFER, outputFbo);
        gl->FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_TEXTURE_2D, outputTexture, 0);
        if (gl->CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            error = "Metal particle compositor framebuffer incomplete";
            gl->BindFramebuffer (GL_FRAMEBUFFER, 0);
            destroyTarget (gl);
            return false;
        }
        gl->BindFramebuffer (GL_FRAMEBUFFER, 0);
        outWidth = width;
        outHeight = height;
        nativeTarget = false;
        return true;
    }

    int planSteps (const ShaderClock& clock, int& firstFrame)
    {
        if (! clock.playing) { firstFrame = clock.frame; return 0; }
        int steps = 0;
        if (! simSeeded)
        {
            if (clock.frame < 0 || clock.frame >= 10000) return -1;
            steps = clock.frame + 1;
            firstFrame = 0;
        }
        else if (clock.frame == lastSimFrame) { steps = 0; firstFrame = clock.frame; }
        else if (clock.frame < lastSimFrame) return -1;
        else
        {
            steps = clock.frame - lastSimFrame;
            if (steps > 10000) return -1;
            firstFrame = lastSimFrame + 1;
        }
        simSeeded = true;
        lastSimFrame = clock.frame;
        return steps;
    }
};

MetalParticleEngine::MetalParticleEngine() : impl_ (std::make_unique<Impl>()) {}
MetalParticleEngine::~MetalParticleEngine() = default;

bool MetalParticleEngine::enabled()
{
    const char* value = std::getenv ("ARBIT_VIDEO_METAL");
    // Physical Apple-silicon validation covers native compute, offscreen
    // rendering, and the IOSurface bridge. Keep an explicit recovery switch,
    // but use the native path by default on Apple builds.
    return value == nullptr || std::strcmp (value, "0") != 0;
}

uint32_t MetalParticleEngine::renderViewUnlocked (
    const arbitgl::GlFuncs* gl, const ShaderClock& clock,
    int width, int height, const ParticleParams& params,
    const ::canonicalblockc::CanonicalBlockCFrame* notes, bool nativeOnly)
{
    if (width <= 0 || height <= 0)
        return 0;
    if (! ensureSokolMetal())
    {
        impl_->error = gSokolError;
        return 0;
    }
    if (! sg_query_features().compute || ! impl_->ensurePrograms())
        return 0;

    int count = params.count;
    if (count < 1) count = 1;
    if (count > ParticleEngine::kMaxParticles) count = ParticleEngine::kMaxParticles;
    if (impl_->simSeeded && clock.frame < impl_->lastSimFrame)
    {
        if (impl_->particleView.id != 0) sg_destroy_view (impl_->particleView);
        if (impl_->particleBuffer.id != 0) sg_destroy_buffer (impl_->particleBuffer);
        impl_->particleView = {};
        impl_->particleBuffer = {};
        impl_->poolCount = 0;
        impl_->simSeeded = false;
        impl_->lastSimFrame = 0;
    }
    if (! impl_->ensurePool (count)
        || ! impl_->ensureTarget (gl, width, height, nativeOnly))
        return 0;

    std::array<float, 4 * 128 * 4> noteData = {};
    int noteCount = 0;
    if (notes != nullptr && notes->noteTexture().size() >= noteData.size())
    {
        std::copy_n (notes->noteTexture().data(), noteData.size(), noteData.data());
        noteCount = notes->noteRows();
    }
    impl_->diagnostics = {};
    impl_->diagnostics.noteRows = noteCount;
    const sg_range noteRange = { noteData.data(), sizeof (noteData) };
    sg_update_buffer (impl_->notesBuffer, &noteRange);

    int firstFrame = clock.frame;
    const int steps = impl_->planSteps (clock, firstFrame);
    if (steps < 0)
    {
        impl_->error = "Metal particle replay exceeds deterministic bound";
        return 0;
    }
    sg_pass computePass = {};
    computePass.compute = true;
    computePass.label = "arbit-metal-particle-compute-pass";
    sg_begin_pass (&computePass);
    sg_apply_pipeline (impl_->computePipeline);
    sg_bindings computeBindings = {};
    computeBindings.views[0] = impl_->particleView;
    computeBindings.views[1] = impl_->notesView;
    sg_apply_bindings (&computeBindings);
    for (int step = 0; step < steps; ++step)
    {
        const MetalComputeParams uniforms = {
            count,
            params.spawnTrack,
            params.gravity,
            params.force > 0.0f ? params.force : 0.0f,
            clock.playing ? static_cast<float> (clock.timeDelta) : 0.0f,
            firstFrame + step + params.seed,
            noteCount,
            height > 0 ? static_cast<float> (width) / static_cast<float> (height) : 1.0f,
            params.lifetime,
            { 0.0f, 0.0f, 0.0f },
        };
        const sg_range range = { &uniforms, sizeof (uniforms) };
        sg_apply_uniforms (0, &range);
        sg_dispatch ((count + 255) / 256, 1, 1);
    }
    sg_end_pass();

    sg_pass drawPass = {};
    drawPass.attachments.colors[0] = impl_->outputView;
    drawPass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
    drawPass.action.colors[0].store_action = SG_STOREACTION_STORE;
    drawPass.action.colors[0].clear_value = { 0.0f, 0.0f, 0.0f, 0.0f };
    drawPass.label = "arbit-metal-particle-draw-pass";
    sg_begin_pass (&drawPass);
    sg_apply_pipeline (impl_->drawPipeline);
    sg_bindings drawBindings = {};
    drawBindings.views[0] = impl_->particleView;
    sg_apply_bindings (&drawBindings);
    const MetalDrawParams drawParams = {
        params.size > 0.0f ? params.size : 1.0f,
        { 0.0f, 0.0f, 0.0f },
        { params.red, params.green, params.blue, params.alpha },
    };
    const sg_range drawRange = { &drawParams, sizeof (drawParams) };
    sg_apply_uniforms (0, &drawRange);
    sg_draw (0, count, 1);
    sg_end_pass();
    impl_->error.clear();
    return impl_->outputTextureView.id;
}

uint32_t MetalParticleEngine::renderMetalViewUnlocked (
    const ShaderClock& clock, int width, int height,
    const ParticleParams& params, const ::canonicalblockc::CanonicalBlockCFrame* notes)
{
    return renderViewUnlocked (nullptr, clock, width, height, params, notes, true);
}

unsigned MetalParticleEngine::render (const arbitgl::GlFuncs* gl,
                                      const ShaderClock& clock,
                                      int width, int height,
                                      const ParticleParams& params,
                                      const ::canonicalblockc::CanonicalBlockCFrame* notes)
{
    if (gl == nullptr)
        return 0;
    std::lock_guard<std::mutex> lock (sokolMutex());
    if (renderViewUnlocked (gl, clock, width, height, params, notes, false) == 0)
        return 0;
    sg_commit();

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
    id<MTLCommandBuffer> fence = [queue commandBuffer];
    [fence commit];
    [fence waitUntilCompleted];

    const char* diagnostic = std::getenv ("ARBIT_PARTICLE_DIAGNOSTICS");
    if (diagnostic != nullptr && std::strcmp (diagnostic, "1") == 0)
    {
        const sg_mtl_buffer_info info = sg_mtl_query_buffer_info (impl_->particleBuffer);
        id<MTLBuffer> source = (__bridge id<MTLBuffer>) info.buf[info.active_slot];
        const NSUInteger byteCount = static_cast<NSUInteger> (impl_->poolCount)
                                   * sizeof (MetalParticle);
        id<MTLBuffer> copy = [gMetalDevice newBufferWithLength:byteCount
                                                       options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        [encoder copyFromBuffer:source sourceOffset:0 toBuffer:copy destinationOffset:0
                           size:byteCount];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        const auto* particles = static_cast<const MetalParticle*> ([copy contents]);
        impl_->diagnostics.liveParticles = 0;
        impl_->diagnostics.visibleCandidates = 0;
        for (int i = 0; i < impl_->poolCount; ++i)
        {
            if (particles[i].life > 0.0f)
            {
                ++impl_->diagnostics.liveParticles;
                if (std::isfinite (particles[i].pos[0]) && std::isfinite (particles[i].pos[1])
                    && particles[i].pos[0] >= 0.0f && particles[i].pos[0] <= 1.0f
                    && particles[i].pos[1] >= 0.0f && particles[i].pos[1] <= 1.0f)
                    ++impl_->diagnostics.visibleCandidates;
            }
        }
#if ! __has_feature(objc_arc)
        [copy release];
#endif
        std::vector<unsigned char> pixels ((size_t) width * (size_t) height * 4);
        [impl_->metalTexture getBytes:pixels.data() bytesPerRow:(NSUInteger) width * 4
                           fromRegion:MTLRegionMake2D (0, 0, width, height) mipmapLevel:0];
        impl_->diagnostics.drawAlphaPixels = 0;
        for (size_t i = 3; i < pixels.size(); i += 4)
            if (pixels[i] != 0) ++impl_->diagnostics.drawAlphaPixels;
    }

    gl->BindFramebuffer (GL_READ_FRAMEBUFFER, impl_->rectangleFbo);
    gl->BindFramebuffer (GL_DRAW_FRAMEBUFFER, impl_->outputFbo);
    gl->BlitFramebuffer (0, 0, width, height, 0, 0, width, height,
                         GL_COLOR_BUFFER_BIT, GL_NEAREST);
    if (diagnostic != nullptr && std::strcmp (diagnostic, "1") == 0)
    {
        std::vector<unsigned char> pixels ((size_t) width * (size_t) height * 4);
        gl->BindFramebuffer (GL_READ_FRAMEBUFFER, impl_->outputFbo);
        glReadBuffer (GL_COLOR_ATTACHMENT0);
        glReadPixels (0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        impl_->diagnostics.readbackAlphaPixels = 0;
        for (size_t i = 3; i < pixels.size(); i += 4)
            if (pixels[i] != 0) ++impl_->diagnostics.readbackAlphaPixels;
    }
    gl->BindFramebuffer (GL_FRAMEBUFFER, 0);
    return impl_->outputTexture;
}

void MetalParticleEngine::shutdown (const arbitgl::GlFuncs* gl)
{
    if (impl_ == nullptr)
        return;
    std::lock_guard<std::mutex> lock (sokolMutex());
    shutdownUnlocked (gl);
}

void MetalParticleEngine::shutdownUnlocked (const arbitgl::GlFuncs* gl)
{
    impl_->destroyTarget (gl);
    if (impl_->particleView.id != 0) sg_destroy_view (impl_->particleView);
    if (impl_->particleBuffer.id != 0) sg_destroy_buffer (impl_->particleBuffer);
    if (impl_->notesView.id != 0) sg_destroy_view (impl_->notesView);
    if (impl_->notesBuffer.id != 0) sg_destroy_buffer (impl_->notesBuffer);
    if (impl_->drawPipeline.id != 0) sg_destroy_pipeline (impl_->drawPipeline);
    if (impl_->drawShader.id != 0) sg_destroy_shader (impl_->drawShader);
    if (impl_->computePipeline.id != 0) sg_destroy_pipeline (impl_->computePipeline);
    if (impl_->computeShader.id != 0) sg_destroy_shader (impl_->computeShader);
    impl_->particleView = {};
    impl_->particleBuffer = {};
    impl_->notesView = {};
    impl_->notesBuffer = {};
    impl_->drawPipeline = {};
    impl_->drawShader = {};
    impl_->computePipeline = {};
    impl_->computeShader = {};
    impl_->programsReady = false;
}

const std::string& MetalParticleEngine::log() const
{
    return impl_->error;
}

const ParticleDiagnostics& MetalParticleEngine::diagnostics() const
{
    return impl_->diagnostics;
}

} // namespace videorender
#endif

#if ARBIT_HAVE_VIEWPORT
namespace videorender
{

namespace
{

std::atomic<uint64_t> metalRendererConstructions { 0 };
std::atomic<uint64_t> metalPipelineAllocations { 0 };
std::atomic<uint64_t> metalIoSurfaceAllocations { 0 };
std::atomic<uint64_t> metalTextureAllocations { 0 };
std::atomic<uint64_t> metalTargetAllocations { 0 };

struct MetalGeometryParams
{
    float transform[16];
    float crop[4];
};

struct MetalMaskParams
{
    float maskRect[4];
    float opacity;
    int32_t maskType;
    float maskFeather;
    int32_t maskInvert;
    float matteRefine[4];
    float matteTexel[2];
    float matteChoke;
    int32_t matteFlags;
    float pathRect[4];
    float pathRectB[4];
    int32_t pathOperation;
    int32_t pathInvert;
    float pathPadding[2];
};

constexpr MetalMaskParams fullFrameMaskParams() noexcept
{
    return {
        { 0.5f, 0.5f, 1.0f, 1.0f }, 1.0f, 0, 0.0f, 0,
        { 0, 0, 0, 0 }, { 0, 0 }, 0.0f, 0,
        { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, -1, 0, { 0, 0 }
    };
}

struct MetalBlendParams
{
    float opacity;
    int32_t blendMode;
    float padding[2];
};

struct MetalFrameMixParams
{
    float mix;
    float padding[3];
};

struct MetalDepthFogParams
{
    float rangeDensity[4];
    float color[4];
};

struct MetalTransitionParams
{
    float progress;
    int32_t transitionType;
    int32_t blendMode;
    float padding;
};

struct MetalFilterParams
{
    float texelX;
    float texelY;
    float amount;
    float padding;
};

struct MetalColorTransformParams
{
    float inputToWorking[16];
    float workingToOutput[16];
    float workingLuma[4];
    float luminance[4];
    int32_t options[4];
    int32_t rendering[4];
};

struct MetalProductionFilterParams
{
    float resolutionX;
    float resolutionY;
    int32_t mode;
    float padding;
    float values[4];
};

struct MetalUvEffectParams
{
    float resolutionX;
    float resolutionY;
    float time;
    int32_t mode;
    float values[4];
};

struct MetalFeedbackParams
{
    float decay;
    float zoom;
    float swirl;
    float padding;
};

struct MetalPostParams
{
    float threshold;
    float intensity;
    float exposure;
    int32_t tonemap;
};

struct MetalCanvasParams
{
    float rect[4];
    float texel[2];
    float padding[2];
};

struct MetalPreviewParams
{
    float zoom;
    float alignmentPadding;
    float pan[2];
    float split;
    int32_t layout;
    int32_t background;
    float trailingPadding;
    float padding[2];
};

struct MetalDrawShapeParams
{
    float rect[4];
    float rectB[4];
    float color[4];
    int32_t operation;
    float padding[3];
};

enum MetalFxValue
{
    MfxBrightness = 0, MfxContrast, MfxSaturation, MfxHueShift, MfxExposure,
    MfxGamma, MfxVignetteAmount, MfxVignetteSoftness, MfxWarmth, MfxCoolness,
    MfxVintage, MfxSepia, MfxBw, MfxInvert, MfxPosterize, MfxNoise,
    MfxKeyR, MfxKeyG, MfxKeyB, MfxKeyTolerance, MfxKeySoftness, MfxKeySpill,
    MfxLumaLow, MfxLumaHigh, MfxLumaSoftness, MfxLumaInvert,
    MfxLiftR, MfxLiftG, MfxLiftB, MfxGammaR, MfxGammaG, MfxGammaB,
    MfxGainR, MfxGainG, MfxGainB,
    MfxKeyChoke, MfxKeyFeather, MfxKeyEdgeR, MfxKeyEdgeG, MfxKeyEdgeB,
    MfxKeyEdgeAmount, MfxKeyMatteView, MfxKeyTexelX, MfxKeyTexelY,
    MfxCount
};

inline constexpr int MfxStorageCount = 48;

struct MetalEffectParams
{
    int32_t mask = 0;
    float time = 0.0f;
    float lutEnabled = 0.0f;
    float lutSize = 0.0f;
    float values[MfxStorageCount] = {};
};

static_assert (sizeof (MetalGeometryParams) == 80, "Metal geometry uniforms changed");
static_assert (sizeof (MetalMaskParams) == 112, "Metal mask uniforms changed");
static_assert (fullFrameMaskParams().pathOperation == -1,
               "full-frame preprocessing must disable path matte coverage");
static_assert (sizeof (MetalBlendParams) == 16, "Metal blend uniforms changed");
static_assert (sizeof (MetalFrameMixParams) == 16, "Metal frame-mix uniforms changed");
static_assert (sizeof (MetalDepthFogParams) == 32, "Metal depth-fog uniforms changed");
static_assert (sizeof (MetalTransitionParams) == 16, "Metal transition uniforms changed");
static_assert (sizeof (MetalFilterParams) == 16, "Metal filter uniforms changed");
static_assert (sizeof (MetalColorTransformParams) == 192,
               "Metal color-transform uniforms changed");
static_assert (sizeof (MetalProductionFilterParams) == 32,
               "Metal production-filter uniforms changed");
static_assert (sizeof (MetalUvEffectParams) == 32, "Metal UV effect uniforms changed");
static_assert (sizeof (MetalFeedbackParams) == 16, "Metal feedback uniforms changed");
static_assert (sizeof (MetalPostParams) == 16, "Metal post uniforms changed");
static_assert (sizeof (MetalCanvasParams) == 32, "Metal canvas uniforms changed");
static_assert (sizeof (MetalDrawShapeParams) == 64, "Metal draw-shape uniforms changed");
static_assert (sizeof (MetalEffectParams) == 208, "Metal effect uniforms changed");

const char* kMetalLayerVertex = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Geometry { float4x4 transform; float4 crop; };
struct Out {
    float4 position [[position]];
    float2 uv [[user(locn0)]];
    float2 rawUV [[user(locn1)]];
};
vertex Out _main(uint vertexId [[vertex_id]], constant Geometry& g [[buffer(0)]])
{
    const float2 pos[6] = { float2(-1,-1), float2(1,-1), float2(1,1),
                            float2(-1,-1), float2(1,1), float2(-1,1) };
    const float2 uv[6] = { float2(0,0), float2(1,0), float2(1,1),
                           float2(0,0), float2(1,1), float2(0,1) };
    Out out;
    out.position = g.transform * float4(pos[vertexId], 0, 1);
    out.rawUV = uv[vertexId];
    out.uv = mix(float2(g.crop.x, g.crop.z),
                 float2(1.0f - g.crop.y, 1.0f - g.crop.w), uv[vertexId]);
    return out;
}
)metal";

const char* kMetalLayerFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In {
    float4 position [[position]];
    float2 uv [[user(locn0)]];
    float2 rawUV [[user(locn1)]];
};
struct Mask { float4 rect; float opacity; int type; float feather; int invert;
              float4 matteRefine; float2 matteTexel; float matteChoke; int matteFlags;
              float4 pathRect; float4 pathRectB; int pathOperation; int pathInvert;
              float2 pathPadding; };
struct Effects { int mask; float time; float lutEnabled; float lutSize; float values[48]; };
struct DepthFog { float4 rangeDensity; float4 color; };
float luminance(float3 c) { return dot(c, float3(0.2126f, 0.7152f, 0.0722f)); }
float randomValue(float2 st) {
    return fract(sin(dot(st, float2(12.9898f, 78.233f))) * 43758.5453f);
}
float hue2rgb(float p, float q, float t) {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
    if (t < 0.5f) return q;
    if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
    return p;
}
float3 rgb2hsl(float3 c) {
    const float maxC = max(max(c.r, c.g), c.b);
    const float minC = min(min(c.r, c.g), c.b);
    const float l = (maxC + minC) * 0.5f;
    float h = 0.0f, s = 0.0f;
    if (maxC != minC) {
        const float d = maxC - minC;
        s = l > 0.5f ? d / (2.0f - maxC - minC) : d / (maxC + minC);
        if (maxC == c.r) h = (c.g - c.b) / d + (c.g < c.b ? 6.0f : 0.0f);
        else if (maxC == c.g) h = (c.b - c.r) / d + 2.0f;
        else h = (c.r - c.g) / d + 4.0f;
        h /= 6.0f;
    }
    return float3(h, s, l);
}
float3 hsl2rgb(float3 hsl) {
    if (hsl.y == 0.0f) return float3(hsl.z);
    const float q = hsl.z < 0.5f ? hsl.z * (1.0f + hsl.y)
                                  : hsl.z + hsl.y - hsl.z * hsl.y;
    const float p = 2.0f * hsl.z - q;
    return float3(hue2rgb(p, q, hsl.x + 1.0f / 3.0f),
                  hue2rgb(p, q, hsl.x),
                  hue2rgb(p, q, hsl.x - 1.0f / 3.0f));
}
float chromaCoverage(float2 uv, float3 key, constant Effects& e,
                     texture2d<float> image, sampler imageSampler) {
    const float4 sampleColor = image.sample(imageSampler, clamp(uv, 0.0f, 1.0f));
    const float2 pixCbCr = float2(sampleColor.b - luminance(sampleColor.rgb),
                                  sampleColor.r - luminance(sampleColor.rgb));
    const float2 keyCbCr = float2(key.b - luminance(key), key.r - luminance(key));
    const float dist = length(pixCbCr - keyCbCr);
    return sampleColor.a * smoothstep(e.values[19],
        e.values[19] + max(e.values[20], 1.0e-5f), dist);
}
float4 applyEffects(float4 color, float2 uv, constant Effects& e,
                    texture2d<float> image, sampler imageSampler) {
    float3 rgb = color.rgb;
    float alpha = color.a;
    if ((e.mask & 131072) != 0) {
        const float3 key = float3(e.values[16], e.values[17], e.values[18]);
        float keyAlpha = chromaCoverage(uv, key, e, image, imageSampler);
        if (e.values[36] > 0.0f) {
            const int radius = int(ceil(min(e.values[36], 4.0f)));
            const float sigma = max(e.values[36] * 0.5f, 0.5f);
            float sum = 0.0f, weightSum = 0.0f;
            for (int y = -4; y <= 4; ++y)
                for (int x = -4; x <= 4; ++x) {
                    if (abs(x) > radius || abs(y) > radius) continue;
                    const float weight = exp(-float(x*x + y*y)
                        / (2.0f * sigma * sigma));
                    const float2 offset = float2(float(x) * e.values[42],
                                                 float(y) * e.values[43]);
                    sum += chromaCoverage(uv + offset, key, e, image, imageSampler) * weight;
                    weightSum += weight;
                }
            keyAlpha = sum / max(weightSum, 1.0e-5f);
        }
        alpha = clamp(keyAlpha + e.values[35], 0.0f, 1.0f);
        const float spill = e.values[21];
        if (spill > 0.0f) {
            if (key.g >= key.r && key.g >= key.b)
                rgb.g = mix(rgb.g, min(rgb.g, max(rgb.r, rgb.b)), spill);
            else if (key.b >= key.r)
                rgb.b = mix(rgb.b, min(rgb.b, max(rgb.r, rgb.g)), spill);
            else
                rgb.r = mix(rgb.r, min(rgb.r, max(rgb.g, rgb.b)), spill);
        }
        const float edgeWeight = e.values[40] * 4.0f * alpha * (1.0f - alpha);
        rgb = mix(rgb, float3(e.values[37], e.values[38], e.values[39]), edgeWeight);
        if (e.values[41] >= 0.5f) return float4(float3(alpha), 1.0f);
    }
    if ((e.mask & 262144) != 0) {
        const float luma = luminance(rgb);
        const float softness = max(e.values[24], 1.0e-5f);
        float keep = smoothstep(e.values[22] - softness, e.values[22], luma)
                   * (1.0f - smoothstep(e.values[23], e.values[23] + softness, luma));
        if (e.values[25] >= 0.5f) keep = 1.0f - keep;
        alpha *= keep;
    }
    if ((e.mask & 16) != 0 && e.values[4] != 0.0f) rgb *= pow(2.0f, e.values[4]);
    if ((e.mask & 1) != 0 && e.values[0] != 0.0f) rgb += float3(e.values[0]);
    if ((e.mask & 2) != 0 && e.values[1] != 1.0f) rgb = (rgb - 0.5f) * e.values[1] + 0.5f;
    if ((e.mask & 32) != 0 && e.values[5] != 1.0f)
        rgb = pow(max(rgb, float3(0.0f)), float3(1.0f / e.values[5]));
    if ((e.mask & 4) != 0 && e.values[2] != 1.0f)
        rgb = mix(float3(luminance(rgb)), rgb, e.values[2]);
    if ((e.mask & 8) != 0 && e.values[3] != 0.0f) {
        float3 hsl = rgb2hsl(rgb);
        const float shifted = hsl.x + e.values[3] / 360.0f;
        hsl.x = shifted - floor(shifted);
        rgb = hsl2rgb(hsl);
    }
    if ((e.mask & 524288) != 0) {
        const float3 lift = float3(e.values[26], e.values[27], e.values[28]);
        const float3 gamma = max(float3(e.values[29], e.values[30], e.values[31]),
                                 float3(1.0e-3f));
        const float3 gain = float3(e.values[32], e.values[33], e.values[34]);
        rgb = clamp(rgb * gain + lift * (1.0f - rgb), 0.0f, 1.0f);
        rgb = pow(rgb, 1.0f / gamma);
    }
    if ((e.mask & 512) != 0 && e.values[8] > 0.0f)
        rgb = mix(rgb, rgb * float3(1.1f, 0.9f, 0.7f), e.values[8] * 0.5f);
    if ((e.mask & 1024) != 0 && e.values[9] > 0.0f)
        rgb = mix(rgb, rgb * float3(0.8f, 0.9f, 1.2f), e.values[9] * 0.5f);
    if ((e.mask & 2048) != 0 && e.values[10] > 0.0f) {
        const float luma = luminance(rgb);
        float3 vintage = mix(float3(luma), rgb, 0.7f) * float3(1.1f, 1.0f, 0.9f);
        vintage = max(vintage, float3(0.03f));
        rgb = mix(rgb, vintage, e.values[10]);
    }
    if ((e.mask & 4096) != 0 && e.values[11] > 0.0f) {
        const float3 sepia = float3(dot(rgb, float3(0.393f, 0.769f, 0.189f)),
                                    dot(rgb, float3(0.349f, 0.686f, 0.168f)),
                                    dot(rgb, float3(0.272f, 0.534f, 0.131f)));
        rgb = mix(rgb, sepia, e.values[11]);
    }
    if ((e.mask & 8192) != 0 && e.values[12] > 0.0f)
        rgb = mix(rgb, float3(luminance(rgb)), e.values[12]);
    if ((e.mask & 16384) != 0 && e.values[13] > 0.0f)
        rgb = mix(rgb, 1.0f - rgb, e.values[13]);
    if ((e.mask & 32768) != 0 && e.values[14] > 0.0f)
        rgb = floor(rgb * e.values[14]) / e.values[14];
    if ((e.mask & 65536) != 0 && e.values[15] > 0.0f) {
        const float noise = randomValue(uv + float2(e.time)) * 2.0f - 1.0f;
        rgb += float3(noise * e.values[15] * 0.2f);
    }
    if ((e.mask & 256) != 0 && e.values[6] > 0.0f) {
        const float dist = length(uv - 0.5f) * 1.4142f;
        const float vignette = 1.0f - smoothstep(1.0f - e.values[6] - e.values[7],
                                                 1.0f - e.values[6] + 0.01f, dist);
        rgb *= vignette;
    }
    return float4(clamp(rgb, 0.0f, 1.0f), alpha);
}
float coverage(float2 uv, constant Mask& m)
{
    if (m.type == 0) return 1.0f;
    const float f = max(m.feather, 1.0e-5f);
    float cov = 1.0f;
    if (m.type == 1) {
        const float2 d = abs(uv - m.rect.xy) - 0.5f * m.rect.zw;
        cov = 1.0f - smoothstep(-f, 0.0f, max(d.x, d.y));
    } else {
        const float2 r = (uv - m.rect.xy) / max(0.5f * m.rect.zw, float2(1.0e-5f));
        const float fr = f / max(0.25f * (m.rect.z + m.rect.w), 1.0e-5f);
        cov = 1.0f - smoothstep(-fr, 0.0f, length(r) - 1.0f);
    }
    return m.invert != 0 ? 1.0f - cov : cov;
}
float pathShapeCoverage(float2 uv, float4 shape) {
    if (shape.z < 0.0f) {
        const float2 r = (uv - shape.xy)
            / max(0.5f * float2(-shape.z, shape.w), float2(1.0e-5f));
        return step(length(r), 1.0f);
    }
    const float2 d = abs(uv - shape.xy) - 0.5f * shape.zw;
    return step(max(d.x, d.y), 0.0f);
}
float pathCoverage(float2 uv, constant Mask& m) {
    if (m.pathOperation < 0) return 1.0f;
    const float a = pathShapeCoverage(uv, m.pathRect);
    float value = a;
    if (m.pathOperation > 0) {
        const float b = pathShapeCoverage(uv, m.pathRectB);
        if (m.pathOperation == 1) value = max(a, b);
        else if (m.pathOperation == 2) value = min(a, b);
        else value = max(a - b, 0.0f);
    }
    return m.pathInvert != 0 ? 1.0f - value : value;
}
float rawMatteCoverage(float2 uv, constant Mask& m, texture2d<float> matte,
                       texture2d<float> matteB, sampler imageSampler) {
    float value = matte.sample(imageSampler, uv).r;
    const int combineMode = (m.matteFlags >> 2) - 1;
    if (combineMode >= 0) {
        const float b = matteB.sample(imageSampler, uv).r;
        if (combineMode == 0) value = max(value, b);
        else if (combineMode == 1) value = min(value, b);
        else if (combineMode == 2) value = max(value - b, 0.0f);
        else value = abs(value - b);
    }
    return value;
}
float matteCoverage(float2 uv, constant Mask& m, texture2d<float> matte,
                    texture2d<float> matteB,
                    sampler imageSampler) {
    if ((m.matteFlags & 1) == 0) return 1.0f;
    const int radius = int(clamp(abs(m.matteRefine.z), 0.0f, 4.0f));
    float value = rawMatteCoverage(uv, m, matte, matteB, imageSampler);
    if (radius > 0) {
        float aggregate = m.matteRefine.z >= 0.0f ? 0.0f : 1.0f;
        for (int y = -4; y <= 4; ++y) for (int x = -4; x <= 4; ++x)
            if (abs(x) <= radius && abs(y) <= radius) {
                const float sampleValue = rawMatteCoverage(
                    uv + float2(x, y) * m.matteTexel, m, matte, matteB, imageSampler);
                aggregate = m.matteRefine.z >= 0.0f ? max(aggregate, sampleValue)
                                                    : min(aggregate, sampleValue);
            }
        value = aggregate;
    }
    const int feather = int(clamp(m.matteRefine.w, 0.0f, 4.0f));
    if (feather > 0) {
        float sum = 0.0f, count = 0.0f;
        for (int y = -4; y <= 4; ++y) for (int x = -4; x <= 4; ++x)
            if (abs(x) <= feather && abs(y) <= feather) {
                sum += rawMatteCoverage(uv + float2(x, y) * m.matteTexel,
                                        m, matte, matteB, imageSampler);
                count += 1.0f;
            }
        value = sum / max(count, 1.0f);
    }
    value = clamp((value - m.matteRefine.x)
                  / max(m.matteRefine.y - m.matteRefine.x, 1.0e-5f), 0.0f, 1.0f);
    value = clamp(value + m.matteChoke, 0.0f, 1.0f);
    return (m.matteFlags & 2) != 0 ? 1.0f - value : value;
}
fragment float4 _main(In in [[stage_in]], constant Mask& m [[buffer(0)]],
                      constant Effects& e [[buffer(1)]],
                      constant DepthFog& fog [[buffer(2)]],
                      texture2d<float> image [[texture(0)]],
                      texture3d<float> lut [[texture(1)]],
                      texture2d<float> matte [[texture(2)]],
                      texture2d<float> matteB [[texture(3)]],
                      texture2d<float> depth [[texture(4)]],
                      sampler imageSampler [[sampler(0)]])
{
    const int depthMode = int(fog.rangeDensity.w + 0.5f);
    const float depthValue = depth.sample(imageSampler, in.rawUV).r;
    float2 sampleUV = in.uv;
    if (depthMode == 3)
        sampleUV = clamp(in.uv + fog.rangeDensity.xy * (depthValue - fog.rangeDensity.z), 0.0f, 1.0f);
    float4 sampled = image.sample(imageSampler, sampleUV);
    if (depthMode == 2) {
        const float radius = clamp(abs(depthValue - fog.rangeDensity.y)
            / max(fog.rangeDensity.z, 1.0e-5f), 0.0f, 1.0f) * fog.rangeDensity.x;
        const float2 stepUV = radius / float2(image.get_width(), image.get_height());
        sampled = (sampled * 4.0f + image.sample(imageSampler, sampleUV + float2(stepUV.x, 0))
            + image.sample(imageSampler, sampleUV - float2(stepUV.x, 0))
            + image.sample(imageSampler, sampleUV + float2(0, stepUV.y))
            + image.sample(imageSampler, sampleUV - float2(0, stepUV.y))) / 8.0f;
    }
    float4 c = applyEffects(sampled, sampleUV, e, image, imageSampler);
    if (e.lutEnabled > 0.5f) {
        const float size = max(e.lutSize, 2.0f);
        const float3 uvw = clamp(c.rgb, 0.0f, 1.0f) * ((size - 1.0f) / size)
                         + 0.5f / size;
        c = float4(lut.sample(imageSampler, uvw).rgb, c.a);
    }
    if (depthMode == 1) {
        const float range = clamp((depthValue - fog.rangeDensity.x)
            / max(fog.rangeDensity.y - fog.rangeDensity.x, 1.0e-5f), 0.0f, 1.0f);
        const float amount = fog.color.a * (1.0f - exp(-fog.rangeDensity.z * range));
        c.rgb = mix(c.rgb, fog.color.rgb, amount);
    } else if (depthMode == 4) {
        c.rgb *= fog.color.rgb * max(0.0f, fog.rangeDensity.y
            + fog.rangeDensity.x * (1.0f - depthValue));
    }
    return float4(c.rgb, c.a * m.opacity * coverage(in.rawUV, m)
                             * pathCoverage(in.rawUV, m)
                             * matteCoverage(in.rawUV, m, matte, matteB, imageSampler));
}
)metal";

const char* kMetalFullscreenVertex = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Out { float4 position [[position]]; float2 uv [[user(locn0)]]; };
vertex Out _main(uint vertexId [[vertex_id]])
{
    const float2 pos[3] = { float2(-1,-1), float2(3,-1), float2(-1,3) };
    const float2 uv[3] = { float2(0,0), float2(2,0), float2(0,2) };
    Out out; out.position = float4(pos[vertexId], 0, 1); out.uv = uv[vertexId];
    return out;
}
)metal";

const char* kMetalColorTransformFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params {
    float4x4 inputToWorking;
    float4x4 workingToOutput;
    float4 workingLuma;
    float4 luminance;
    int4 options;
    int4 rendering;
};
float3 decodeTransfer(float3 value, int transfer, float sourceReferenceWhite)
{
    value = max(value, float3(0.0f));
    if (transfer == 1) return value;
    if (transfer == 2) {
        const bool3 low = value <= float3(0.04045f);
        return select(pow((value + 0.055f) / 1.055f, float3(2.4f)),
                      value / 12.92f, low);
    }
    if (transfer == 3) return pow(value, float3(2.2f));
    if (transfer == 4) {
        const float m1 = 2610.0f / 16384.0f;
        const float m2 = 2523.0f / 32.0f;
        const float c1 = 3424.0f / 4096.0f;
        const float c2 = 2413.0f / 128.0f;
        const float c3 = 2392.0f / 128.0f;
        const float3 p = pow(value, float3(1.0f / m2));
        const float3 absoluteNits = 10000.0f
            * pow(max(p - c1, float3(0.0f))
                / max(c2 - c3 * p, float3(1.0e-6f)), float3(1.0f / m1));
        return absoluteNits / sourceReferenceWhite;
    }
    const float a = 0.17883277f;
    const float b = 0.28466892f;
    const float c = 0.55991073f;
    const bool3 low = value <= float3(0.5f);
    return select((exp((value - c) / a) + b) / 12.0f,
                  value * value / 3.0f, low);
}
float3 encodeTransfer(float3 value, int transfer, float outputReferenceWhite)
{
    value = max(value, float3(0.0f));
    if (transfer == 1) return value;
    if (transfer == 2) {
        const bool3 low = value <= float3(0.0031308f);
        return select(1.055f * pow(value, float3(1.0f / 2.4f)) - 0.055f,
                      12.92f * value, low);
    }
    if (transfer == 3) return pow(value, float3(1.0f / 2.2f));
    if (transfer == 4) {
        const float m1 = 2610.0f / 16384.0f;
        const float m2 = 2523.0f / 32.0f;
        const float c1 = 3424.0f / 4096.0f;
        const float c2 = 2413.0f / 128.0f;
        const float c3 = 2392.0f / 128.0f;
        const float3 normalizedNits = value * outputReferenceWhite / 10000.0f;
        const float3 p = pow(normalizedNits, float3(m1));
        return pow((c1 + c2 * p) / (1.0f + c3 * p), float3(m2));
    }
    const float a = 0.17883277f;
    const float b = 0.28466892f;
    const float c = 0.55991073f;
    const bool3 low = value <= float3(1.0f / 12.0f);
    return select(a * log(12.0f * value - b) + c,
                  sqrt(3.0f * value), low);
}
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> image [[texture(0)]],
                      sampler imageSampler [[sampler(0)]])
{
    const float4 sampled = image.sample(imageSampler, in.uv);
    float alpha = p.options.z == 1 ? 1.0f : sampled.a;
    float3 encoded = sampled.rgb;
    if (p.options.z == 3)
        encoded = alpha > 0.0f ? encoded / alpha : float3(0.0f);
    float3 working = (p.inputToWorking
        * float4(decodeTransfer(encoded, p.options.x, p.luminance.x), 1.0f)).rgb;
    float3 absoluteNits = working * p.luminance.x;
    if (p.rendering.x == 1) {
        const float y = max(dot(max(absoluteNits, float3(0.0f)), p.workingLuma.rgb), 0.0f);
        const float shoulder = max(0.0f, 1.0f / p.luminance.w - 1.0f / p.luminance.y);
        const float mappedY = y / (1.0f + shoulder * y);
        absoluteNits *= y > 0.0f ? mappedY / y : 0.0f;
    }
    working = absoluteNits / p.luminance.z;
    const float3 outputLinear = (p.workingToOutput * float4(working, 1.0f)).rgb;
    float3 outputEncoded = encodeTransfer(outputLinear, p.options.y, p.luminance.z);
    if (p.options.w == 1) alpha = 1.0f;
    if (p.options.w == 3) outputEncoded *= alpha;
    if (alpha == 0.0f && p.options.w == 3) outputEncoded = float3(0.0f);
    if (p.rendering.y != 0) outputEncoded = clamp(outputEncoded, 0.0f, 1.0f);
    return float4(outputEncoded, alpha);
}
)metal";

const char* kMetalBlendFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float opacity; int mode; float2 padding; };
float3 overlay(float3 base, float3 blend) {
    return select(2.0f * base * blend,
                  1.0f - 2.0f * (1.0f - base) * (1.0f - blend),
                  base >= 0.5f);
}
float3 colorDodge(float3 base, float3 blend) {
    float3 result;
    for (int i = 0; i < 3; ++i) {
        if (base[i] <= 0.0f) result[i] = 0.0f;
        else if (blend[i] >= 1.0f) result[i] = 1.0f;
        else result[i] = min(1.0f, base[i] / (1.0f - blend[i]));
    }
    return result;
}
float3 colorBurn(float3 base, float3 blend) {
    float3 result;
    for (int i = 0; i < 3; ++i) {
        if (base[i] >= 1.0f) result[i] = 1.0f;
        else if (blend[i] <= 0.0f) result[i] = 0.0f;
        else result[i] = 1.0f - min(1.0f, (1.0f - base[i]) / blend[i]);
    }
    return result;
}
float3 softLight(float3 base, float3 blend) {
    float3 result;
    for (int i = 0; i < 3; ++i) {
        if (blend[i] <= 0.5f) {
            result[i] = base[i] - (1.0f - 2.0f * blend[i]) * base[i] * (1.0f - base[i]);
        } else {
            const float d = base[i] <= 0.25f
                ? ((16.0f * base[i] - 12.0f) * base[i] + 4.0f) * base[i]
                : sqrt(base[i]);
            result[i] = base[i] + (2.0f * blend[i] - 1.0f) * (d - base[i]);
        }
    }
    return result;
}
float3 hardLight(float3 base, float3 blend) {
    return select(2.0f * base * blend,
                  1.0f - 2.0f * (1.0f - base) * (1.0f - blend),
                  blend >= 0.5f);
}
float3 applyBlendMode(float3 base, float3 blend, int mode) {
    if (mode == 1) return min(base + blend, float3(1.0f));
    if (mode == 2) return base * blend;
    if (mode == 3) return 1.0f - (1.0f - base) * (1.0f - blend);
    if (mode == 4) return overlay(base, blend);
    if (mode == 5) return abs(base - blend);
    if (mode == 6) return base + blend - 2.0f * base * blend;
    if (mode == 7) return min(base, blend);
    if (mode == 8) return max(base, blend);
    if (mode == 9) return colorDodge(base, blend);
    if (mode == 10) return colorBurn(base, blend);
    if (mode == 11) return softLight(base, blend);
    if (mode == 12) return hardLight(base, blend);
    return blend;
}
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> frontTex [[texture(0)]],
                      texture2d<float> backTex [[texture(1)]],
                      sampler imageSampler [[sampler(0)]])
{
    const float4 front = frontTex.sample(imageSampler, in.uv);
    const float4 back = backTex.sample(imageSampler, in.uv);
    const float alpha = front.a * p.opacity;
    const float3 blended = applyBlendMode(back.rgb, front.rgb, p.mode);
    return float4(mix(back.rgb, blended, alpha), max(back.a, alpha));
}
)metal";

const char* kMetalFrameMixFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float mixValue; float3 padding; };
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> earlier [[texture(0)]],
                      texture2d<float> later [[texture(1)]],
                      sampler imageSampler [[sampler(0)]])
{
    return mix(earlier.sample(imageSampler, in.uv),
               later.sample(imageSampler, in.uv),
               clamp(p.mixValue, 0.0f, 1.0f));
}
)metal";

const char* kMetalTransitionFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float progress; int transitionType; int blendMode; float padding; };
float3 overlay(float3 base, float3 blend) {
    return select(2.0f * base * blend,
                  1.0f - 2.0f * (1.0f - base) * (1.0f - blend),
                  base >= 0.5f);
}
float3 colorDodge(float3 base, float3 blend) {
    float3 result;
    for (int i = 0; i < 3; ++i) {
        if (base[i] <= 0.0f) result[i] = 0.0f;
        else if (blend[i] >= 1.0f) result[i] = 1.0f;
        else result[i] = min(1.0f, base[i] / (1.0f - blend[i]));
    }
    return result;
}
float3 colorBurn(float3 base, float3 blend) {
    float3 result;
    for (int i = 0; i < 3; ++i) {
        if (base[i] >= 1.0f) result[i] = 1.0f;
        else if (blend[i] <= 0.0f) result[i] = 0.0f;
        else result[i] = 1.0f - min(1.0f, (1.0f - base[i]) / blend[i]);
    }
    return result;
}
float3 softLight(float3 base, float3 blend) {
    float3 result;
    for (int i = 0; i < 3; ++i) {
        if (blend[i] <= 0.5f) {
            result[i] = base[i] - (1.0f - 2.0f * blend[i]) * base[i] * (1.0f - base[i]);
        } else {
            const float d = base[i] <= 0.25f
                ? ((16.0f * base[i] - 12.0f) * base[i] + 4.0f) * base[i]
                : sqrt(base[i]);
            result[i] = base[i] + (2.0f * blend[i] - 1.0f) * (d - base[i]);
        }
    }
    return result;
}
float3 hardLight(float3 base, float3 blend) {
    return select(2.0f * base * blend,
                  1.0f - 2.0f * (1.0f - base) * (1.0f - blend),
                  blend >= 0.5f);
}
float3 applyBlendMode(float3 base, float3 blend, int mode) {
    if (mode == 1) return min(base + blend, float3(1.0f));
    if (mode == 2) return base * blend;
    if (mode == 3) return 1.0f - (1.0f - base) * (1.0f - blend);
    if (mode == 4) return overlay(base, blend);
    if (mode == 5) return abs(base - blend);
    if (mode == 6) return base + blend - 2.0f * base * blend;
    if (mode == 7) return min(base, blend);
    if (mode == 8) return max(base, blend);
    if (mode == 9) return colorDodge(base, blend);
    if (mode == 10) return colorBurn(base, blend);
    if (mode == 11) return softLight(base, blend);
    if (mode == 12) return hardLight(base, blend);
    return blend;
}
float easeInOutCubic(float t) {
    if (t < 0.5f) return 4.0f * t * t * t;
    const float p = 2.0f * t - 2.0f;
    return 0.5f * p * p * p + 1.0f;
}
float4 transitionColor(float4 from, float4 to, float2 uv, constant Params& p) {
    const float progress = clamp(p.progress, 0.0f, 1.0f);
    if (p.transitionType == 0) return mix(from, to, easeInOutCubic(progress));
    if (p.transitionType == 1) {
        if (progress < 0.5f) return float4(from.rgb * (1.0f - progress * 2.0f), from.a);
        return float4(to.rgb * ((progress - 0.5f) * 2.0f), to.a);
    }
    if (p.transitionType == 2) return uv.x > 1.0f - progress ? to : from;
    if (p.transitionType == 3) return uv.x < progress ? to : from;
    if (p.transitionType == 4) return uv.y > 1.0f - progress ? to : from;
    if (p.transitionType == 5) return uv.y < progress ? to : from;
    return progress >= 0.5f ? to : from;
}
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> fromTex [[texture(0)]],
                      texture2d<float> toTex [[texture(1)]],
                      texture2d<float> backTex [[texture(2)]],
                      sampler imageSampler [[sampler(0)]])
{
    const float4 front = transitionColor(fromTex.sample(imageSampler, in.uv),
                                         toTex.sample(imageSampler, in.uv), in.uv, p);
    const float4 back = backTex.sample(imageSampler, in.uv);
    const float3 blended = applyBlendMode(back.rgb, front.rgb, p.blendMode);
    return float4(mix(back.rgb, blended, front.a), max(back.a, front.a));
}
)metal";

const char* kMetalBlitFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
fragment float4 _main(In in [[stage_in]],
                      texture2d<float> image [[texture(0)]],
                      sampler imageSampler [[sampler(0)]])
{ return image.sample(imageSampler, in.uv); }
)metal";

const char* kMetalPreviewFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float zoom; float2 pan; float split; int layout; int background; float2 padding; };
float3 previewBackground(float2 uv, int mode) {
    if (mode == 1) return float3(0.0f);
    if (mode == 2) return float3(1.0f);
    float c = fmod(floor(uv.x * 32.0f) + floor(uv.y * 32.0f), 2.0f);
    return mix(float3(0.18f), float3(0.32f), c);
}
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> finalImage [[texture(0)]],
                      texture2d<float> previewImage [[texture(1)]],
                      sampler imageSampler [[sampler(0)]]) {
    if (p.layout == 1 && in.uv.x < p.split)
        return finalImage.sample(imageSampler, float2(in.uv.x / p.split, in.uv.y));
    const float4 overlayRect = float4(0.58f, 0.04f, 0.98f, 0.44f);
    if (p.layout == 2 && (in.uv.x < overlayRect.x || in.uv.x > overlayRect.z
                      || in.uv.y < overlayRect.y || in.uv.y > overlayRect.w))
        return finalImage.sample(imageSampler, in.uv);
    float2 region = p.layout == 1
        ? float2((in.uv.x - p.split) / (1.0f - p.split), in.uv.y)
        : (in.uv - overlayRect.xy) / (overlayRect.zw - overlayRect.xy);
    float2 uv = (region - 0.5f) / p.zoom + 0.5f - p.pan;
    float4 sample = all(uv >= 0.0f) && all(uv <= 1.0f)
        ? previewImage.sample(imageSampler, uv) : float4(0.0f);
    return float4(mix(previewBackground(region, p.background), sample.rgb, sample.a), 1.0f);
}
)metal";

const char* kMetalBlurFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float2 texelStep; float radius; float padding; };
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> image [[texture(0)]],
                      sampler imageSampler [[sampler(0)]])
{
    const int radius = int(min(p.radius, 20.0f) + 0.5f);
    if (radius <= 0) return image.sample(imageSampler, in.uv);
    const float sigma = max(p.radius * 0.5f, 0.5f);
    float4 sum = float4(0.0f);
    float weightSum = 0.0f;
    for (int i = -20; i <= 20; ++i) {
        if (i < -radius || i > radius) continue;
        const float weight = exp(-float(i * i) / (2.0f * sigma * sigma));
        sum += image.sample(imageSampler, in.uv + p.texelStep * float(i)) * weight;
        weightSum += weight;
    }
    return sum / weightSum;
}
)metal";

const char* kMetalSharpenFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float2 texelSize; float amount; float padding; };
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> image [[texture(0)]],
                      sampler imageSampler [[sampler(0)]])
{
    const float4 center = image.sample(imageSampler, in.uv);
    float3 blurred = float3(0.0f);
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
            blurred += image.sample(imageSampler,
                in.uv + float2(float(x), float(y)) * p.texelSize).rgb;
    blurred /= 9.0f;
    return float4(clamp(center.rgb + p.amount * (center.rgb - blurred),
                        0.0f, 1.0f), center.a);
}
)metal";

// Exact Phase 5 common production filters. This native Metal program mirrors
// the OpenGL production-filter pass and never leaves GPU memory.
const char* kMetalProductionFilterFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float2 resolution; int mode; float padding; float4 values; };
float4 sampleImage(texture2d<float> image, sampler imageSampler, float2 uv)
{
    return image.sample(imageSampler, clamp(uv, 0.0f, 1.0f));
}
float3 discBlur(texture2d<float> image, sampler imageSampler, float2 uv,
                float radius, float threshold, float2 resolution)
{
    const float2 texel = 1.0f / max(resolution, float2(1.0f));
    float3 sum = float3(0.0f);
    float weightSum = 0.0f;
    for (int i = 0; i < 24; ++i)
    {
        const float fi = float(i) + 0.5f;
        const float angle = fi * 2.399963229728653f;
        const float distancePx = radius * sqrt(fi / 24.0f);
        const float3 value = sampleImage(image, imageSampler,
            uv + float2(cos(angle), sin(angle)) * texel * distancePx).rgb;
        const float luma = dot(value, float3(0.2126f, 0.7152f, 0.0722f));
        const float bright = threshold <= 0.0f ? 1.0f
            : smoothstep(threshold, threshold + max(threshold * 0.5f, 0.05f), luma);
        const float weight = exp(-2.0f * distancePx * distancePx
            / max(radius * radius, 0.25f));
        sum += value * bright * weight;
        weightSum += weight;
    }
    return sum / max(weightSum, 1e-5f);
}
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> image [[texture(0)]],
                      sampler imageSampler [[sampler(0)]])
{
    const float4 center = sampleImage(image, imageSampler, in.uv);
    if (p.mode == 0)
    {
        const float3 glow = discBlur(image, imageSampler, in.uv,
                                     p.values.y, 0.0f, p.resolution);
        const float3 rgb = 1.0f - (1.0f - center.rgb)
            * (1.0f - clamp(glow * p.values.x, 0.0f, 1.0f));
        return float4(clamp(rgb, 0.0f, 1.0f), center.a);
    }
    if (p.mode == 1)
    {
        const float3 glow = discBlur(image, imageSampler, in.uv,
                                     p.values.z, p.values.x, p.resolution);
        return float4(clamp(center.rgb + glow * p.values.y, 0.0f, 1.0f), center.a);
    }
    if (p.mode == 2)
    {
        const float3 glow = discBlur(image, imageSampler, in.uv,
                                     p.values.y, p.values.x, p.resolution);
        const float3 filmRed = float3(glow.r + glow.g * 0.45f,
                                      glow.g * 0.22f, glow.b * 0.04f);
        return float4(clamp(center.rgb + filmRed * p.values.z, 0.0f, 1.0f), center.a);
    }
    if (p.mode == 3)
    {
        const float2 aspect = float2(p.resolution.x / max(p.resolution.y, 1.0f), 1.0f);
        const float2 point = (in.uv - 0.5f) * aspect;
        const float radiusSquared = dot(point, point);
        const float2 warped = point * (1.0f + p.values.x * radiusSquared) / aspect + 0.5f;
        const float2 shift = normalize(point + float2(1e-6f))
            * p.values.y * radiusSquared / aspect;
        const float red = sampleImage(image, imageSampler, warped + shift).r;
        const float4 green = sampleImage(image, imageSampler, warped);
        const float blue = sampleImage(image, imageSampler, warped - shift).b;
        return float4(red, green.g, blue, green.a);
    }
    if (p.mode == 4)
    {
        if (p.values.x <= 0.0f) return center;
        const int radius = int(clamp(floor(p.values.y + 0.5f), 1.0f, 4.0f));
        const float2 texel = 1.0f / max(p.resolution, float2(1.0f));
        const float colorSigma = mix(0.02f, 0.30f, p.values.x);
        float4 sum = float4(0.0f);
        float weightSum = 0.0f;
        for (int y = -4; y <= 4; ++y)
            for (int x = -4; x <= 4; ++x)
            {
                if (abs(x) > radius || abs(y) > radius) continue;
                const float4 value = sampleImage(image, imageSampler,
                    in.uv + float2(float(x), float(y)) * texel);
                const float3 delta = value.rgb - center.rgb;
                const float spatial = exp(-float(x * x + y * y)
                    / max(float(radius * radius), 1.0f));
                const float range = exp(-dot(delta, delta)
                    / max(colorSigma * colorSigma, 1e-5f));
                const float weight = spatial * range;
                sum += value * weight;
                weightSum += weight;
            }
        return mix(center, sum / max(weightSum, 1e-5f), p.values.x);
    }
    if (p.mode == 5)
    {
        constexpr float pi = 3.14159265358979323846f;
        const float angle = p.values.y * pi / 180.0f;
        const float2 stepUv = float2(cos(angle), sin(angle))
            / max(p.resolution, float2(1.0f));
        const int radius = int(min(p.values.x, 20.0f) + 0.5f);
        if (radius == 0) return center;
        float4 sum = float4(0.0f);
        float weightSum = 0.0f;
        const float sigma = max(p.values.x * 0.5f, 0.5f);
        for (int i = -20; i <= 20; ++i)
        {
            if (abs(i) > radius) continue;
            const float weight = exp(-float(i * i) / (2.0f * sigma * sigma));
            sum += sampleImage(image, imageSampler, in.uv + stepUv * float(i)) * weight;
            weightSum += weight;
        }
        return sum / max(weightSum, 1e-5f);
    }
    const float3 blurred = discBlur(image, imageSampler, in.uv,
                                    p.values.y, 0.0f, p.resolution);
    const float3 high = center.rgb - blurred;
    const float gate = p.mode == 6
        ? step(p.values.z, dot(abs(high), float3(0.2126f, 0.7152f, 0.0722f))) : 1.0f;
    return float4(clamp(center.rgb + high * p.values.x * gate, 0.0f, 1.0f), center.a);
}
)metal";

const char* kMetalLutFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float2 unused; float size; float padding; };
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> image [[texture(0)]],
                      texture3d<float> lut [[texture(1)]],
                      sampler imageSampler [[sampler(0)]])
{
    const float4 color = image.sample(imageSampler, in.uv);
    const float size = max(p.size, 2.0f);
    const float3 uvw = clamp(color.rgb, 0.0f, 1.0f) * ((size - 1.0f) / size)
                     + 0.5f / size;
    return float4(lut.sample(imageSampler, uvw).rgb, color.a);
}
)metal";

const char* kMetalUvEffectFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float2 resolution; float time; int mode; float4 values; };
float4 sampleClamped(texture2d<float> image, sampler imageSampler, float2 uv) {
    return image.sample(imageSampler, clamp(uv, 0.0f, 1.0f));
}
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> image [[texture(0)]],
                      sampler imageSampler [[sampler(0)]])
{
    float2 uv = in.uv;
    if (p.mode == 0) {
        const float aspect = p.resolution.x / max(p.resolution.y, 1.0f);
        const float2 point = (uv - 0.5f) * float2(aspect, 1.0f);
        const float radius = length(point);
        float angle = atan2(point.y, point.x);
        const float wedge = 6.28318530718f / max(p.values.x, 2.0f);
        angle -= (p.values.y + p.values.z * p.time) * 6.28318530718f;
        angle = fmod(angle, wedge);
        if (angle < 0.0f) angle += wedge;
        angle = abs(angle - 0.5f * wedge);
        const float2 folded = float2(cos(angle), sin(angle))
                            * (radius / max(p.values.w, 0.01f));
        uv = folded / float2(aspect, 1.0f) + 0.5f;
    } else if (p.mode == 1) {
        const int mode = int(p.values.x + 0.5f);
        if (mode == 0 && uv.x > 0.5f) uv.x = 1.0f - uv.x;
        else if (mode == 1 && uv.x < 0.5f) uv.x = 1.0f - uv.x;
        else if (mode == 2 && uv.y > 0.5f) uv.y = 1.0f - uv.y;
        else if (mode >= 3) {
            if (uv.x > 0.5f) uv.x = 1.0f - uv.x;
            if (uv.y > 0.5f) uv.y = 1.0f - uv.y;
        }
    } else if (p.mode == 2) {
        const float count = max(p.values.x, 1.0f);
        float2 tiled = uv * count;
        const float2 cell = floor(tiled);
        uv = fract(tiled);
        if (p.values.y > 0.5f) {
            if (fmod(cell.x, 2.0f) >= 1.0f) uv.x = 1.0f - uv.x;
            if (fmod(cell.y, 2.0f) >= 1.0f) uv.y = 1.0f - uv.y;
        }
    } else if (p.mode == 3) {
        const float phase = p.time * p.values.z;
        uv.x += p.values.x * sin(uv.y * p.values.y + phase);
        uv.y += p.values.x * cos(uv.x * p.values.y + phase * 1.3f);
    } else if (p.mode == 4) {
        const float2 texel = 1.0f / max(p.resolution, float2(1.0f));
        const float3 weights = float3(0.299f, 0.587f, 0.114f);
        const float lx = dot(sampleClamped(image, imageSampler,
            uv + float2(texel.x, 0.0f)).rgb, weights)
            - dot(sampleClamped(image, imageSampler,
            uv - float2(texel.x, 0.0f)).rgb, weights);
        const float ly = dot(sampleClamped(image, imageSampler,
            uv + float2(0.0f, texel.y)).rgb, weights)
            - dot(sampleClamped(image, imageSampler,
            uv - float2(0.0f, texel.y)).rgb, weights);
        uv += float2(lx, ly) * p.values.x;
    } else if (p.mode == 5) {
        const float aspect = p.resolution.x / max(p.resolution.y, 1.0f);
        const float2 point = (uv - 0.5f) * float2(aspect, 1.0f);
        const float radius = length(point);
        const float falloff = 1.0f - smoothstep(0.0f, max(p.values.y, 1.0e-3f), radius);
        const float angle = p.values.x * 6.28318530718f * falloff;
        const float c = cos(angle), s = sin(angle);
        const float2 rotated = float2(c * point.x - s * point.y,
                                      s * point.x + c * point.y);
        uv = rotated / float2(aspect, 1.0f) + 0.5f;
    } else if (p.mode == 6) {
        const float angle = p.values.y * 6.28318530718f;
        const float2 direction = float2(cos(angle), sin(angle)) * p.values.x;
        const float red = sampleClamped(image, imageSampler, uv + direction).r;
        const float4 center = image.sample(imageSampler, uv);
        const float blue = sampleClamped(image, imageSampler, uv - direction).b;
        return float4(red, center.g, blue, center.a);
    } else if (p.mode == 7) {
        const float block = max(p.values.x, 1.0f);
        const float2 resolution = max(p.resolution, float2(1.0f));
        uv = (floor(uv * resolution / block) + 0.5f) * block / resolution;
    }
    return sampleClamped(image, imageSampler, uv);
}
)metal";

const char* kMetalFeedbackFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float decay; float zoom; float swirl; float padding; };
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> image [[texture(0)]],
                      texture2d<float> history [[texture(1)]],
                      sampler imageSampler [[sampler(0)]])
{
    const float2 point = in.uv - 0.5f;
    const float c = cos(p.swirl), s = sin(p.swirl);
    const float2 trailUv = float2(c * point.x - s * point.y,
                                  s * point.x + c * point.y) * p.zoom + 0.5f;
    const float4 previous = history.sample(imageSampler,
        clamp(trailUv, 0.0f, 1.0f)) * clamp(p.decay, 0.0f, 0.999f);
    const float4 current = image.sample(imageSampler, in.uv);
    float4 result = 1.0f - (1.0f - current) * (1.0f - previous);
    result.a = max(current.a, previous.a);
    return result;
}
)metal";

const char* kMetalBloomThresholdFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float threshold; float intensity; float exposure; int tonemap; };
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> image [[texture(0)]],
                      sampler imageSampler [[sampler(0)]])
{
    const float3 color = image.sample(imageSampler, in.uv).rgb;
    const float luminance = dot(color, float3(0.2126f, 0.7152f, 0.0722f));
    const float knee = max(p.threshold * 0.5f, 1.0e-4f);
    const float value = clamp((luminance - p.threshold + knee) / (2.0f * knee),
                              0.0f, 1.0f);
    return float4(color * (value * value), 1.0f);
}
)metal";

const char* kMetalPostCombineFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float threshold; float intensity; float exposure; int tonemap; };
float3 aces(float3 value) {
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return clamp((value * (a * value + b)) /
                 (value * (c * value + d) + e), 0.0f, 1.0f);
}
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> image [[texture(0)]],
                      texture2d<float> bloom [[texture(1)]],
                      sampler imageSampler [[sampler(0)]])
{
    const float3 base = image.sample(imageSampler, in.uv).rgb;
    const float3 glow = bloom.sample(imageSampler, in.uv).rgb;
    const float3 hdr = (base + glow * p.intensity) * p.exposure;
    float3 ldr = clamp(hdr, 0.0f, 1.0f);
    if (p.tonemap == 1) ldr = hdr / (hdr + 1.0f);
    else if (p.tonemap == 2) ldr = aces(hdr);
    return float4(ldr, 1.0f);
}
)metal";

const char* kMetalDrawShapeFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct ShapeParams { float4 rect; float4 rectB; float4 color; int4 operation; };
bool shapeContains(float2 uv, float4 rect)
{
    float2 halfSize = max(abs(rect.zw) * 0.5f, float2(0.0f));
    float2 local = abs(uv - rect.xy);
    if (rect.z < 0.0f)
    {
        if (any(halfSize <= float2(0.0f))) return false;
        float2 normalized = local / halfSize;
        return dot(normalized, normalized) <= 1.0f;
    }
    float2 d = local - halfSize;
    return max(d.x, d.y) <= 0.0f;
}
fragment float4 _main(In in [[stage_in]], constant ShapeParams& p [[buffer(0)]])
{
    bool insideA = shapeContains(in.uv, p.rect);
    bool insideB = p.operation.x == 0 ? false : shapeContains(in.uv, p.rectB);
    bool inside = p.operation.x == 1 ? (insideA || insideB)
                : p.operation.x == 2 ? (insideA && insideB)
                : p.operation.x == 3 ? (insideA && !insideB)
                : insideA;
    return inside ? p.color : float4(0.0f);
}
)metal";

const char* kMetalCanvasFragment = R"metal(
#include <metal_stdlib>
using namespace metal;
struct In { float4 position [[position]]; float2 uv [[user(locn0)]]; };
struct Params { float4 rect; float2 texel; float2 padding; };
fragment float4 _main(In in [[stage_in]], constant Params& p [[buffer(0)]],
                      texture2d<float> image [[texture(0)]],
                      sampler imageSampler [[sampler(0)]])
{
    float4 color = image.sample(imageSampler, in.uv);
    const bool inside = in.uv.x >= p.rect.x && in.uv.x <= p.rect.z
                     && in.uv.y >= p.rect.y && in.uv.y <= p.rect.w;
    if (!inside) color.rgb *= 0.45f;
    const float border = 1.5f;
    const float dx = min(abs(in.uv.x - p.rect.x), abs(in.uv.x - p.rect.z)) / p.texel.x;
    const float dy = min(abs(in.uv.y - p.rect.y), abs(in.uv.y - p.rect.w)) / p.texel.y;
    const bool spanX = in.uv.x >= p.rect.x - border * p.texel.x
                    && in.uv.x <= p.rect.z + border * p.texel.x;
    const bool spanY = in.uv.y >= p.rect.y - border * p.texel.y
                    && in.uv.y <= p.rect.w + border * p.texel.y;
    if ((dx <= border && spanY) || (dy <= border && spanX))
        color = float4(0.365f, 0.663f, 1.0f, 1.0f);
    return color;
}
)metal";

void makeTransform (float tx, float ty, float rotationDeg, float sx, float sy,
                    float out[16])
{
    const float radians = rotationDeg * 0.01745329251994329577f;
    const float c = std::cos (radians), s = std::sin (radians);
    const float values[16] = {
        c * sx, s * sx, 0.0f, 0.0f,
       -s * sy, c * sy, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        tx, ty, 0.0f, 1.0f,
    };
    std::copy (std::begin (values), std::end (values), out);
}

int metalTransitionType (int wireType)
{
    switch (wireType)
    {
        case 1: return 1;
        case 2: return 0;
        case 3: return 2;
        case 4: return 3;
        case 5: return 4;
        case 6: return 5;
        default: return 0;
    }
}

int metalUvEffectMode (int type)
{
    switch (static_cast<videofx::EffectType> (type))
    {
        case videofx::EffectType::Kaleidoscope: return 0;
        case videofx::EffectType::Mirror: return 1;
        case videofx::EffectType::Tile: return 2;
        case videofx::EffectType::Warp: return 3;
        case videofx::EffectType::Displace: return 4;
        case videofx::EffectType::PolarSwirl: return 5;
        case videofx::EffectType::DisplaceRgb: return 6;
        case videofx::EffectType::Pixelate: return 7;
        default: return -1;
    }
}

int metalProductionFilterMode (int type)
{
    switch (static_cast<videofx::EffectType> (type))
    {
        case videofx::EffectType::Glow: return 0;
        case videofx::EffectType::Bloom: return 1;
        case videofx::EffectType::Halation: return 2;
        case videofx::EffectType::LensDistortion: return 3;
        case videofx::EffectType::Denoise: return 4;
        case videofx::EffectType::DirectionalBlur: return 5;
        case videofx::EffectType::UnsharpMask: return 6;
        case videofx::EffectType::HighPassSharpen: return 7;
        default: return -1;
    }
}

MetalEffectParams makeEffectParams (const LayerDesc& layer)
{
    static constexpr float neutral[MfxCount] = {
        0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.5f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 8.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.18f, 0.10f, 0.0f,
        0.0f, 1.0f, 0.1f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    };
    MetalEffectParams result;
    result.time = static_cast<float> (layer.timeSec);
    result.lutEnabled = layer.lutTexture != 0 && layer.lutSize >= 2 ? 1.0f : 0.0f;
    result.lutSize = static_cast<float> (layer.lutSize);
    std::copy (std::begin (neutral), std::end (neutral), result.values);
    for (int i = 0; layer.effects != nullptr && i < layer.effectCount; ++i)
    {
        const auto& effect = layer.effects[i];
        if (! effect.enabled || effect.type < 0
            || effect.type >= videofx::kEffectTypeCount)
            continue;
        result.mask |= videofx::kEffectBits[effect.type];
        switch (static_cast<videofx::EffectType> (effect.type))
        {
            case videofx::EffectType::Brightness: result.values[MfxBrightness] = effect.params[0]; break;
            case videofx::EffectType::Contrast: result.values[MfxContrast] = effect.params[0]; break;
            case videofx::EffectType::Saturation: result.values[MfxSaturation] = effect.params[0]; break;
            case videofx::EffectType::Hue: result.values[MfxHueShift] = effect.params[0]; break;
            case videofx::EffectType::Exposure: result.values[MfxExposure] = effect.params[0]; break;
            case videofx::EffectType::Gamma: result.values[MfxGamma] = effect.params[0]; break;
            case videofx::EffectType::Vignette:
                result.values[MfxVignetteAmount] = effect.params[0];
                result.values[MfxVignetteSoftness] = effect.params[1]; break;
            case videofx::EffectType::Warm: result.values[MfxWarmth] = effect.params[0]; break;
            case videofx::EffectType::Cool: result.values[MfxCoolness] = effect.params[0]; break;
            case videofx::EffectType::Vintage: result.values[MfxVintage] = effect.params[0]; break;
            case videofx::EffectType::Sepia: result.values[MfxSepia] = effect.params[0]; break;
            case videofx::EffectType::BlackAndWhite: result.values[MfxBw] = effect.params[0]; break;
            case videofx::EffectType::Invert: result.values[MfxInvert] = effect.params[0]; break;
            case videofx::EffectType::Posterize: result.values[MfxPosterize] = effect.params[0]; break;
            case videofx::EffectType::Noise: result.values[MfxNoise] = effect.params[0]; break;
            case videofx::EffectType::ChromaKey:
                for (int p = 0; p < 6; ++p) result.values[MfxKeyR + p] = effect.params[p];
                break;
            case videofx::EffectType::LumaKey:
                for (int p = 0; p < 4; ++p) result.values[MfxLumaLow + p] = effect.params[p];
                break;
            case videofx::EffectType::ColorWheels:
                for (int p = 0; p < 9; ++p) result.values[MfxLiftR + p] = effect.params[p];
                break;
            default: break;
        }
    }
    return result;
}

MetalEffectParams neutralEffectParams()
{
    const LayerDesc neutral;
    return makeEffectParams (neutral);
}

} // namespace

struct MetalFrameRenderer::Impl
{
    struct Target
    {
        sg_image image = {};
        sg_view attachment = {};
        sg_view texture = {};
    };

    struct Source
    {
        int width = 0, height = 0;
        sg_image image = {};
        sg_view view = {};
        std::vector<uint8_t> pixels;
        bool dirty = false;
        bool frameBlend = false;
        unsigned textureA = 0, textureB = 0;
        float blendMix = 0.0f;
        Target blendTarget;
    };

    struct Lut
    {
        int size = 0;
        sg_image image = {};
        sg_view view = {};
    };

    struct Depth
    {
        int width = 0, height = 0;
        sg_image image = {};
        sg_view view = {};
    };

    struct FeedbackHistory
    {
        Target target[2];
        int current = 0;
        bool ready = false;
    };

    struct DirectOutput
    {
        int width = 0, height = 0;
        id<MTLTexture> metalTexture = nil;
        sg_image image = {};
        sg_view attachment = {};
    };

    std::string error;
    videowire::VisualPlanTelemetry* visualTelemetry = nullptr;
    int width = 0, height = 0;
    int canvasWidth = 0, canvasHeight = 0;
    int presentWidth = 0, presentHeight = 0;
    float zoom = 1.0f, panX = 0.0f, panY = 0.0f;
    float bg[4] = { 0.04f, 0.04f, 0.05f, 1.0f };
    float bloomIntensity = 0.0f, bloomThreshold = 1.0f, bloomRadius = 0.0f;
    float exposure = 1.0f;
    int tonemap = 0;
    bool programsReady = false;
    bool directOnly = false;
    std::unordered_map<unsigned, Source> sources;
    std::unordered_map<unsigned, Lut> luts;
    std::unordered_map<unsigned, Depth> depths;
    std::unordered_map<int, std::unique_ptr<MetalParticleEngine>> particles;
#if ARBIT_HAVE_METAL_GENERATORS
    std::unordered_map<int, std::unique_ptr<MetalShaderGenerator>> generators;
    std::map<std::string, std::unique_ptr<MetalShaderGenerator>> operationGenerators;
#endif
    std::string particleBackend = "none";
    std::unordered_map<int, FeedbackHistory> feedback;
    std::unordered_map<void*, DirectOutput> directOutputs;
    void* requestedDirectSurface = nullptr;
    int requestedDirectWidth = 0, requestedDirectHeight = 0;
    uint64_t frameParity = 0;
    Lut identityLut;

    sg_sampler sampler = {};
    sg_shader layerShader = {}, blendShader = {}, frameMixShader = {}, transitionShader = {}, blitShader = {};
    sg_shader blurShader = {}, sharpenShader = {}, productionFilterShader = {};
    sg_shader colorTransformShader = {};
    sg_shader lutShader = {}, uvEffectShader = {};
    sg_shader feedbackShader = {};
    sg_shader bloomThresholdShader = {}, postCombineShader = {};
    sg_shader canvasShader = {}, previewShader = {}, drawShapeShader = {};
    sg_pipeline layerPipeline = {}, blendPipeline = {}, frameMixPipeline = {}, transitionPipeline = {}, blitPipeline = {};
    sg_pipeline blurPipeline = {}, sharpenPipeline = {}, productionFilterPipeline = {};
    sg_pipeline colorTransformPipeline = {};
    sg_pipeline lutPipeline = {}, uvEffectPipeline = {};
    sg_pipeline feedbackPipeline = {};
    sg_pipeline bloomThresholdPipeline = {}, postCombinePipeline = {};
    sg_pipeline canvasPipeline = {}, previewPipeline = {}, drawShapePipeline = {};
    Target layerTarget, transitionFrom, effect[2], accum[2], inspectionTarget;
    int inspectionClipId = -1;
    unsigned inspectionRequestedHandle = 0;
    unsigned inspectionRetainedHandle = 0;
    bool inspectionAdmitted = false;
    videopreview::State inspectionPresentation;

    IOSurfaceRef surface = nullptr;
    id<MTLTexture> metalTexture = nil;
    sg_image outputImage = {};
    sg_view outputAttachment = {};
    unsigned rectangleTexture = 0, rectangleFbo = 0;
    unsigned outputTexture = 0, outputFbo = 0;

    void destroySource (Source& source)
    {
        destroyTarget (source.blendTarget);
        if (source.view.id != 0) sg_destroy_view (source.view);
        if (source.image.id != 0) sg_destroy_image (source.image);
        source = {};
    }

    void destroyTarget (Target& target)
    {
        if (target.texture.id != 0) sg_destroy_view (target.texture);
        if (target.attachment.id != 0) sg_destroy_view (target.attachment);
        if (target.image.id != 0) sg_destroy_image (target.image);
        target = {};
    }

    void destroyLut (Lut& lut)
    {
        if (lut.view.id != 0) sg_destroy_view (lut.view);
        if (lut.image.id != 0) sg_destroy_image (lut.image);
        lut = {};
    }

    void destroyDepth (Depth& depth)
    {
        if (depth.view.id != 0) sg_destroy_view (depth.view);
        if (depth.image.id != 0) sg_destroy_image (depth.image);
        depth = {};
    }

    void destroyDirectOutput (DirectOutput& output)
    {
        if (output.attachment.id != 0) sg_destroy_view (output.attachment);
        if (output.image.id != 0) sg_destroy_image (output.image);
#if ! __has_feature(objc_arc)
        [output.metalTexture release];
#endif
        output = {};
    }

    void clearDirectOutputs()
    {
        for (auto& item : directOutputs) destroyDirectOutput (item.second);
        directOutputs.clear();
    }

    DirectOutput* directOutput (void* surfacePtr, int w, int h)
    {
        if (surfacePtr == nullptr || w <= 0 || h <= 0) return nullptr;
        auto& output = directOutputs[surfacePtr];
        if (output.image.id != 0 && output.width == w && output.height == h)
            return &output;
        destroyDirectOutput (output);
        IOSurfaceRef surface = static_cast<IOSurfaceRef> (surfacePtr);
        MTLTextureDescriptor* descriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                               width:w height:h mipmapped:NO];
        // The IOSurface is sampled by a Metal device in another process.
        // Managed storage would give each process a private GPU backing store,
        // so the consumer could legally observe undefined pixels even after
        // this queue completes. Shared storage keeps the IOSurface itself as
        // the single producer/consumer allocation.
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        output.metalTexture = [gMetalDevice newTextureWithDescriptor:descriptor
                                                            iosurface:surface plane:0];
        if (output.metalTexture == nil) return nullptr;
        sg_image_desc imageDesc = {};
        imageDesc.usage.color_attachment = true;
        imageDesc.width = w;
        imageDesc.height = h;
        imageDesc.pixel_format = SG_PIXELFORMAT_BGRA8;
        imageDesc.sample_count = 1;
#if ! __has_feature(objc_arc)
        [output.metalTexture retain];
#endif
        imageDesc.mtl_textures[0] = (__bridge const void*) output.metalTexture;
        imageDesc.label = "arbit-metal-direct-iosurface";
        output.image = sg_make_image (&imageDesc);
        sg_view_desc viewDesc = {};
        viewDesc.color_attachment.image = output.image;
        output.attachment = sg_make_view (&viewDesc);
        output.width = w;
        output.height = h;
        if (! resourceValid (sg_query_image_state (output.image))
            || ! resourceValid (sg_query_view_state (output.attachment)))
        {
            destroyDirectOutput (output);
            return nullptr;
        }
        return &output;
    }

    bool makeLut (Lut& lut, const float* rgbTriples, int size, const char* label)
    {
        destroyLut (lut);
        if (rgbTriples == nullptr || size < 2) return false;
        const size_t voxels = static_cast<size_t> (size) * size * size;
        std::vector<float> rgba (voxels * 4);
        for (size_t i = 0; i < voxels; ++i)
        {
            rgba[i * 4] = rgbTriples[i * 3];
            rgba[i * 4 + 1] = rgbTriples[i * 3 + 1];
            rgba[i * 4 + 2] = rgbTriples[i * 3 + 2];
            rgba[i * 4 + 3] = 1.0f;
        }
        sg_image_desc desc = {};
        desc.type = SG_IMAGETYPE_3D;
        desc.width = size;
        desc.height = size;
        desc.num_slices = size;
        desc.pixel_format = SG_PIXELFORMAT_RGBA32F;
        desc.data.mip_levels[0] = { rgba.data(), rgba.size() * sizeof (float) };
        desc.label = label;
        lut.image = sg_make_image (&desc);
        sg_view_desc viewDesc = {};
        viewDesc.texture.image = lut.image;
        lut.view = sg_make_view (&viewDesc);
        lut.size = size;
        return resourceValid (sg_query_image_state (lut.image))
            && resourceValid (sg_query_view_state (lut.view));
    }

    void destroyOutputs (const arbitgl::GlFuncs* gl)
    {
        for (auto& item : feedback)
        {
            destroyTarget (item.second.target[0]);
            destroyTarget (item.second.target[1]);
        }
        feedback.clear();
        destroyTarget (layerTarget);
        destroyTarget (transitionFrom);
        destroyTarget (effect[0]);
        destroyTarget (effect[1]);
        destroyTarget (accum[0]);
        destroyTarget (accum[1]);
        destroyTarget (inspectionTarget);
        inspectionRetainedHandle = 0;
        inspectionAdmitted = false;
        if (outputAttachment.id != 0) sg_destroy_view (outputAttachment);
        if (outputImage.id != 0) sg_destroy_image (outputImage);
        outputAttachment = {};
        outputImage = {};
        if (rectangleFbo != 0 && gl != nullptr) gl->DeleteFramebuffers (1, &rectangleFbo);
        if (outputFbo != 0 && gl != nullptr) gl->DeleteFramebuffers (1, &outputFbo);
        if (rectangleTexture != 0) glDeleteTextures (1, &rectangleTexture);
        if (outputTexture != 0) glDeleteTextures (1, &outputTexture);
        rectangleTexture = rectangleFbo = outputTexture = outputFbo = 0;
#if ! __has_feature(objc_arc)
        [metalTexture release];
#endif
        metalTexture = nil;
        if (surface != nullptr) CFRelease (surface);
        surface = nullptr;
        width = height = 0;
    }

    bool makeTarget (Target& target, int w, int h, const char* label)
    {
        sg_image_desc imageDesc = {};
        imageDesc.usage.color_attachment = true;
        imageDesc.width = w;
        imageDesc.height = h;
        imageDesc.pixel_format = SG_PIXELFORMAT_RGBA16F;
        imageDesc.sample_count = 1;
        imageDesc.label = label;
        target.image = sg_make_image (&imageDesc);
        sg_view_desc attachmentDesc = {};
        attachmentDesc.color_attachment.image = target.image;
        target.attachment = sg_make_view (&attachmentDesc);
        sg_view_desc textureDesc = {};
        textureDesc.texture.image = target.image;
        target.texture = sg_make_view (&textureDesc);
        return resourceValid (sg_query_image_state (target.image))
            && resourceValid (sg_query_view_state (target.attachment))
            && resourceValid (sg_query_view_state (target.texture));
    }

    bool ensurePrograms()
    {
        if (programsReady) return true;

        sg_sampler_desc samplerDesc = {};
        samplerDesc.min_filter = SG_FILTER_LINEAR;
        samplerDesc.mag_filter = SG_FILTER_LINEAR;
        samplerDesc.wrap_u = SG_WRAP_CLAMP_TO_EDGE;
        samplerDesc.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
        samplerDesc.wrap_w = SG_WRAP_CLAMP_TO_EDGE;
        sampler = sg_make_sampler (&samplerDesc);

        static constexpr float identity[24] = {
            0,0,0, 1,0,0, 0,1,0, 1,1,0,
            0,0,1, 1,0,1, 0,1,1, 1,1,1,
        };
        if (! makeLut (identityLut, identity, 2, "arbit-metal-identity-lut"))
        {
            error = "Metal identity LUT creation failed";
            return false;
        }

        sg_shader_desc layerDesc = {};
        layerDesc.vertex_func.source = kMetalLayerVertex;
        layerDesc.fragment_func.source = kMetalLayerFragment;
        layerDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
        layerDesc.uniform_blocks[0].size = sizeof (MetalGeometryParams);
        layerDesc.uniform_blocks[0].msl_buffer_n = 0;
        layerDesc.uniform_blocks[1].stage = SG_SHADERSTAGE_FRAGMENT;
        layerDesc.uniform_blocks[1].size = sizeof (MetalMaskParams);
        layerDesc.uniform_blocks[1].msl_buffer_n = 0;
        layerDesc.uniform_blocks[2].stage = SG_SHADERSTAGE_FRAGMENT;
        layerDesc.uniform_blocks[2].size = sizeof (MetalEffectParams);
        layerDesc.uniform_blocks[2].msl_buffer_n = 1;
        layerDesc.uniform_blocks[3].stage = SG_SHADERSTAGE_FRAGMENT;
        layerDesc.uniform_blocks[3].size = sizeof (MetalDepthFogParams);
        layerDesc.uniform_blocks[3].msl_buffer_n = 2;
        layerDesc.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        layerDesc.views[0].texture.image_type = SG_IMAGETYPE_2D;
        layerDesc.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        layerDesc.views[0].texture.msl_texture_n = 0;
        layerDesc.views[1].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        layerDesc.views[1].texture.image_type = SG_IMAGETYPE_3D;
        layerDesc.views[1].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        layerDesc.views[1].texture.msl_texture_n = 1;
        layerDesc.views[2].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        layerDesc.views[2].texture.image_type = SG_IMAGETYPE_2D;
        layerDesc.views[2].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        layerDesc.views[2].texture.msl_texture_n = 2;
        layerDesc.views[3].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        layerDesc.views[3].texture.image_type = SG_IMAGETYPE_2D;
        layerDesc.views[3].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        layerDesc.views[3].texture.msl_texture_n = 3;
        layerDesc.views[4].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        layerDesc.views[4].texture.image_type = SG_IMAGETYPE_2D;
        layerDesc.views[4].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        layerDesc.views[4].texture.msl_texture_n = 4;
        layerDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        layerDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        layerDesc.samplers[0].msl_sampler_n = 0;
        layerDesc.texture_sampler_pairs[0] = { SG_SHADERSTAGE_FRAGMENT, 0, 0, "image" };
        layerDesc.texture_sampler_pairs[1] = { SG_SHADERSTAGE_FRAGMENT, 1, 0, "lut" };
        layerDesc.texture_sampler_pairs[2] = { SG_SHADERSTAGE_FRAGMENT, 2, 0, "matte" };
        layerDesc.texture_sampler_pairs[3] = { SG_SHADERSTAGE_FRAGMENT, 3, 0, "matteB" };
        layerDesc.texture_sampler_pairs[4] = { SG_SHADERSTAGE_FRAGMENT, 4, 0, "depth" };
        layerDesc.label = "arbit-metal-frame-layer";
        layerShader = sg_make_shader (&layerDesc);

        sg_pipeline_desc layerPipelineDesc = {};
        layerPipelineDesc.shader = layerShader;
        layerPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
        layerPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        layerPipelineDesc.label = "arbit-metal-frame-layer-pipeline";
        layerPipeline = sg_make_pipeline (&layerPipelineDesc);

        sg_shader_desc blendDesc = {};
        blendDesc.vertex_func.source = kMetalFullscreenVertex;
        blendDesc.fragment_func.source = kMetalBlendFragment;
        blendDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        blendDesc.uniform_blocks[0].size = sizeof (MetalBlendParams);
        blendDesc.uniform_blocks[0].msl_buffer_n = 0;
        for (int i = 0; i < 2; ++i)
        {
            blendDesc.views[i].texture.stage = SG_SHADERSTAGE_FRAGMENT;
            blendDesc.views[i].texture.image_type = SG_IMAGETYPE_2D;
            blendDesc.views[i].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
            blendDesc.views[i].texture.msl_texture_n = static_cast<uint8_t> (i);
        }
        blendDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        blendDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        blendDesc.samplers[0].msl_sampler_n = 0;
        blendDesc.texture_sampler_pairs[0] = { SG_SHADERSTAGE_FRAGMENT, 0, 0, "frontTex" };
        blendDesc.texture_sampler_pairs[1] = { SG_SHADERSTAGE_FRAGMENT, 1, 0, "backTex" };
        blendDesc.label = "arbit-metal-frame-blend";
        blendShader = sg_make_shader (&blendDesc);

        sg_pipeline_desc blendPipelineDesc = {};
        blendPipelineDesc.shader = blendShader;
        blendPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
        blendPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        blendPipelineDesc.label = "arbit-metal-frame-blend-pipeline";
        blendPipeline = sg_make_pipeline (&blendPipelineDesc);

        sg_shader_desc frameMixDesc = {};
        frameMixDesc.vertex_func.source = kMetalFullscreenVertex;
        frameMixDesc.fragment_func.source = kMetalFrameMixFragment;
        frameMixDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        frameMixDesc.uniform_blocks[0].size = sizeof (MetalFrameMixParams);
        frameMixDesc.uniform_blocks[0].msl_buffer_n = 0;
        for (int i = 0; i < 2; ++i)
        {
            frameMixDesc.views[i].texture.stage = SG_SHADERSTAGE_FRAGMENT;
            frameMixDesc.views[i].texture.image_type = SG_IMAGETYPE_2D;
            frameMixDesc.views[i].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
            frameMixDesc.views[i].texture.msl_texture_n = static_cast<uint8_t> (i);
        }
        frameMixDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        frameMixDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        frameMixDesc.samplers[0].msl_sampler_n = 0;
        frameMixDesc.texture_sampler_pairs[0] = {
            SG_SHADERSTAGE_FRAGMENT, 0, 0, "earlier" };
        frameMixDesc.texture_sampler_pairs[1] = {
            SG_SHADERSTAGE_FRAGMENT, 1, 0, "later" };
        frameMixDesc.label = "arbit-metal-frame-mix";
        frameMixShader = sg_make_shader (&frameMixDesc);
        sg_pipeline_desc frameMixPipelineDesc = {};
        frameMixPipelineDesc.shader = frameMixShader;
        frameMixPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
        frameMixPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        frameMixPipelineDesc.label = "arbit-metal-frame-mix-pipeline";
        frameMixPipeline = sg_make_pipeline (&frameMixPipelineDesc);

        sg_shader_desc transitionDesc = {};
        transitionDesc.vertex_func.source = kMetalFullscreenVertex;
        transitionDesc.fragment_func.source = kMetalTransitionFragment;
        transitionDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        transitionDesc.uniform_blocks[0].size = sizeof (MetalTransitionParams);
        transitionDesc.uniform_blocks[0].msl_buffer_n = 0;
        for (int i = 0; i < 3; ++i)
        {
            transitionDesc.views[i].texture.stage = SG_SHADERSTAGE_FRAGMENT;
            transitionDesc.views[i].texture.image_type = SG_IMAGETYPE_2D;
            transitionDesc.views[i].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
            transitionDesc.views[i].texture.msl_texture_n = static_cast<uint8_t> (i);
        }
        transitionDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        transitionDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        transitionDesc.samplers[0].msl_sampler_n = 0;
        transitionDesc.texture_sampler_pairs[0] = { SG_SHADERSTAGE_FRAGMENT, 0, 0, "fromTex" };
        transitionDesc.texture_sampler_pairs[1] = { SG_SHADERSTAGE_FRAGMENT, 1, 0, "toTex" };
        transitionDesc.texture_sampler_pairs[2] = { SG_SHADERSTAGE_FRAGMENT, 2, 0, "backTex" };
        transitionDesc.label = "arbit-metal-frame-transition";
        transitionShader = sg_make_shader (&transitionDesc);
        sg_pipeline_desc transitionPipelineDesc = {};
        transitionPipelineDesc.shader = transitionShader;
        transitionPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
        transitionPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        transitionPipelineDesc.label = "arbit-metal-frame-transition-pipeline";
        transitionPipeline = sg_make_pipeline (&transitionPipelineDesc);

        sg_shader_desc blitDesc = {};
        blitDesc.vertex_func.source = kMetalFullscreenVertex;
        blitDesc.fragment_func.source = kMetalBlitFragment;
        blitDesc.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        blitDesc.views[0].texture.image_type = SG_IMAGETYPE_2D;
        blitDesc.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        blitDesc.views[0].texture.msl_texture_n = 0;
        blitDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        blitDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        blitDesc.samplers[0].msl_sampler_n = 0;
        blitDesc.texture_sampler_pairs[0] = { SG_SHADERSTAGE_FRAGMENT, 0, 0, "image" };
        blitDesc.label = "arbit-metal-frame-blit";
        blitShader = sg_make_shader (&blitDesc);
        sg_pipeline_desc blitPipelineDesc = {};
        blitPipelineDesc.shader = blitShader;
        blitPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_BGRA8;
        blitPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        blitPipelineDesc.label = "arbit-metal-frame-blit-pipeline";
        blitPipeline = sg_make_pipeline (&blitPipelineDesc);

        sg_shader_desc previewDesc = {};
        previewDesc.vertex_func.source = kMetalFullscreenVertex;
        previewDesc.fragment_func.source = kMetalPreviewFragment;
        previewDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        previewDesc.uniform_blocks[0].size = sizeof (MetalPreviewParams);
        previewDesc.uniform_blocks[0].msl_buffer_n = 0;
        for (int i = 0; i < 2; ++i)
        {
            previewDesc.views[i].texture.stage = SG_SHADERSTAGE_FRAGMENT;
            previewDesc.views[i].texture.image_type = SG_IMAGETYPE_2D;
            previewDesc.views[i].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
            previewDesc.views[i].texture.msl_texture_n = static_cast<uint8_t> (i);
        }
        previewDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        previewDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        previewDesc.samplers[0].msl_sampler_n = 0;
        previewDesc.texture_sampler_pairs[0] = { SG_SHADERSTAGE_FRAGMENT, 0, 0, "finalImage" };
        previewDesc.texture_sampler_pairs[1] = { SG_SHADERSTAGE_FRAGMENT, 1, 0, "previewImage" };
        previewDesc.label = "arbit-metal-node-preview";
        previewShader = sg_make_shader (&previewDesc);
        sg_pipeline_desc previewPipelineDesc = {};
        previewPipelineDesc.shader = previewShader;
        previewPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_BGRA8;
        previewPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        previewPipelineDesc.label = "arbit-metal-node-preview-pipeline";
        previewPipeline = sg_make_pipeline (&previewPipelineDesc);

        auto makeFilter = [&] (const char* fragment, const char* label,
                               sg_shader& shader, sg_pipeline& pipeline,
                               bool hasLut)
        {
            sg_shader_desc desc = {};
            desc.vertex_func.source = kMetalFullscreenVertex;
            desc.fragment_func.source = fragment;
            desc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
            desc.uniform_blocks[0].size = sizeof (MetalFilterParams);
            desc.uniform_blocks[0].msl_buffer_n = 0;
            desc.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
            desc.views[0].texture.image_type = SG_IMAGETYPE_2D;
            desc.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
            desc.views[0].texture.msl_texture_n = 0;
            desc.texture_sampler_pairs[0] = {
                SG_SHADERSTAGE_FRAGMENT, 0, 0, "image" };
            if (hasLut)
            {
                desc.views[1].texture.stage = SG_SHADERSTAGE_FRAGMENT;
                desc.views[1].texture.image_type = SG_IMAGETYPE_3D;
                desc.views[1].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
                desc.views[1].texture.msl_texture_n = 1;
                desc.texture_sampler_pairs[1] = {
                    SG_SHADERSTAGE_FRAGMENT, 1, 0, "lut" };
            }
            desc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
            desc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
            desc.samplers[0].msl_sampler_n = 0;
            desc.label = label;
            shader = sg_make_shader (&desc);
            sg_pipeline_desc pipelineDesc = {};
            pipelineDesc.shader = shader;
            pipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
            pipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
            pipelineDesc.label = label;
            pipeline = sg_make_pipeline (&pipelineDesc);
        };
        makeFilter (kMetalBlurFragment, "arbit-metal-frame-blur",
                    blurShader, blurPipeline, false);
        makeFilter (kMetalSharpenFragment, "arbit-metal-frame-sharpen",
                    sharpenShader, sharpenPipeline, false);
        makeFilter (kMetalLutFragment, "arbit-metal-frame-lut",
                    lutShader, lutPipeline, true);

        sg_shader_desc colorTransformDesc = {};
        colorTransformDesc.vertex_func.source = kMetalFullscreenVertex;
        colorTransformDesc.fragment_func.source = kMetalColorTransformFragment;
        colorTransformDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        colorTransformDesc.uniform_blocks[0].size = sizeof (MetalColorTransformParams);
        colorTransformDesc.uniform_blocks[0].msl_buffer_n = 0;
        colorTransformDesc.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        colorTransformDesc.views[0].texture.image_type = SG_IMAGETYPE_2D;
        colorTransformDesc.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        colorTransformDesc.views[0].texture.msl_texture_n = 0;
        colorTransformDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        colorTransformDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        colorTransformDesc.samplers[0].msl_sampler_n = 0;
        colorTransformDesc.texture_sampler_pairs[0] = {
            SG_SHADERSTAGE_FRAGMENT, 0, 0, "image" };
        colorTransformDesc.label = "arbit-metal-frame-color-transform";
        colorTransformShader = sg_make_shader (&colorTransformDesc);
        sg_pipeline_desc colorTransformPipelineDesc = {};
        colorTransformPipelineDesc.shader = colorTransformShader;
        colorTransformPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
        colorTransformPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        colorTransformPipelineDesc.label = "arbit-metal-frame-color-transform-pipeline";
        colorTransformPipeline = sg_make_pipeline (&colorTransformPipelineDesc);

        sg_shader_desc productionDesc = {};
        productionDesc.vertex_func.source = kMetalFullscreenVertex;
        productionDesc.fragment_func.source = kMetalProductionFilterFragment;
        productionDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        productionDesc.uniform_blocks[0].size = sizeof (MetalProductionFilterParams);
        productionDesc.uniform_blocks[0].msl_buffer_n = 0;
        productionDesc.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        productionDesc.views[0].texture.image_type = SG_IMAGETYPE_2D;
        productionDesc.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        productionDesc.views[0].texture.msl_texture_n = 0;
        productionDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        productionDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        productionDesc.samplers[0].msl_sampler_n = 0;
        productionDesc.texture_sampler_pairs[0] = {
            SG_SHADERSTAGE_FRAGMENT, 0, 0, "image" };
        productionDesc.label = "arbit-metal-frame-production-filter";
        productionFilterShader = sg_make_shader (&productionDesc);
        sg_pipeline_desc productionPipelineDesc = {};
        productionPipelineDesc.shader = productionFilterShader;
        productionPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
        productionPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        productionPipelineDesc.label = "arbit-metal-frame-production-filter-pipeline";
        productionFilterPipeline = sg_make_pipeline (&productionPipelineDesc);

        sg_shader_desc uvDesc = {};
        uvDesc.vertex_func.source = kMetalFullscreenVertex;
        uvDesc.fragment_func.source = kMetalUvEffectFragment;
        uvDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        uvDesc.uniform_blocks[0].size = sizeof (MetalUvEffectParams);
        uvDesc.uniform_blocks[0].msl_buffer_n = 0;
        uvDesc.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        uvDesc.views[0].texture.image_type = SG_IMAGETYPE_2D;
        uvDesc.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        uvDesc.views[0].texture.msl_texture_n = 0;
        uvDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        uvDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        uvDesc.samplers[0].msl_sampler_n = 0;
        uvDesc.texture_sampler_pairs[0] = {
            SG_SHADERSTAGE_FRAGMENT, 0, 0, "image" };
        uvDesc.label = "arbit-metal-frame-uv-effect";
        uvEffectShader = sg_make_shader (&uvDesc);
        sg_pipeline_desc uvPipelineDesc = {};
        uvPipelineDesc.shader = uvEffectShader;
        uvPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
        uvPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        uvPipelineDesc.label = "arbit-metal-frame-uv-effect-pipeline";
        uvEffectPipeline = sg_make_pipeline (&uvPipelineDesc);

        sg_shader_desc feedbackDesc = {};
        feedbackDesc.vertex_func.source = kMetalFullscreenVertex;
        feedbackDesc.fragment_func.source = kMetalFeedbackFragment;
        feedbackDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        feedbackDesc.uniform_blocks[0].size = sizeof (MetalFeedbackParams);
        feedbackDesc.uniform_blocks[0].msl_buffer_n = 0;
        for (int i = 0; i < 2; ++i)
        {
            feedbackDesc.views[i].texture.stage = SG_SHADERSTAGE_FRAGMENT;
            feedbackDesc.views[i].texture.image_type = SG_IMAGETYPE_2D;
            feedbackDesc.views[i].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
            feedbackDesc.views[i].texture.msl_texture_n = static_cast<uint8_t> (i);
        }
        feedbackDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        feedbackDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        feedbackDesc.samplers[0].msl_sampler_n = 0;
        feedbackDesc.texture_sampler_pairs[0] = {
            SG_SHADERSTAGE_FRAGMENT, 0, 0, "image" };
        feedbackDesc.texture_sampler_pairs[1] = {
            SG_SHADERSTAGE_FRAGMENT, 1, 0, "history" };
        feedbackDesc.label = "arbit-metal-frame-feedback";
        feedbackShader = sg_make_shader (&feedbackDesc);
        sg_pipeline_desc feedbackPipelineDesc = {};
        feedbackPipelineDesc.shader = feedbackShader;
        feedbackPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
        feedbackPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        feedbackPipelineDesc.label = "arbit-metal-frame-feedback-pipeline";
        feedbackPipeline = sg_make_pipeline (&feedbackPipelineDesc);

        auto makePost = [&] (const char* fragment, const char* label,
                             int textureCount, sg_shader& shader,
                             sg_pipeline& pipeline)
        {
            sg_shader_desc desc = {};
            desc.vertex_func.source = kMetalFullscreenVertex;
            desc.fragment_func.source = fragment;
            desc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
            desc.uniform_blocks[0].size = sizeof (MetalPostParams);
            desc.uniform_blocks[0].msl_buffer_n = 0;
            for (int i = 0; i < textureCount; ++i)
            {
                desc.views[i].texture.stage = SG_SHADERSTAGE_FRAGMENT;
                desc.views[i].texture.image_type = SG_IMAGETYPE_2D;
                desc.views[i].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
                desc.views[i].texture.msl_texture_n = static_cast<uint8_t> (i);
            }
            desc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
            desc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
            desc.samplers[0].msl_sampler_n = 0;
            desc.texture_sampler_pairs[0] = {
                SG_SHADERSTAGE_FRAGMENT, 0, 0, "image" };
            if (textureCount > 1)
                desc.texture_sampler_pairs[1] = {
                    SG_SHADERSTAGE_FRAGMENT, 1, 0, "bloom" };
            desc.label = label;
            shader = sg_make_shader (&desc);
            sg_pipeline_desc pipelineDesc = {};
            pipelineDesc.shader = shader;
            pipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
            pipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
            pipelineDesc.label = label;
            pipeline = sg_make_pipeline (&pipelineDesc);
        };
        makePost (kMetalBloomThresholdFragment, "arbit-metal-frame-bloom-threshold",
                  1, bloomThresholdShader, bloomThresholdPipeline);
        makePost (kMetalPostCombineFragment, "arbit-metal-frame-post-combine",
                  2, postCombineShader, postCombinePipeline);

        sg_shader_desc canvasDesc = {};
        canvasDesc.vertex_func.source = kMetalFullscreenVertex;
        canvasDesc.fragment_func.source = kMetalCanvasFragment;
        canvasDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        canvasDesc.uniform_blocks[0].size = sizeof (MetalCanvasParams);
        canvasDesc.uniform_blocks[0].msl_buffer_n = 0;
        canvasDesc.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        canvasDesc.views[0].texture.image_type = SG_IMAGETYPE_2D;
        canvasDesc.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        canvasDesc.views[0].texture.msl_texture_n = 0;
        canvasDesc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        canvasDesc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        canvasDesc.samplers[0].msl_sampler_n = 0;
        canvasDesc.texture_sampler_pairs[0] = {
            SG_SHADERSTAGE_FRAGMENT, 0, 0, "image" };
        canvasDesc.label = "arbit-metal-frame-canvas";
        canvasShader = sg_make_shader (&canvasDesc);
        sg_pipeline_desc canvasPipelineDesc = {};
        canvasPipelineDesc.shader = canvasShader;
        canvasPipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_BGRA8;
        canvasPipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        canvasPipelineDesc.label = "arbit-metal-frame-canvas-pipeline";
        canvasPipeline = sg_make_pipeline (&canvasPipelineDesc);

        sg_shader_desc drawShapeDesc = {};
        drawShapeDesc.vertex_func.source = kMetalFullscreenVertex;
        drawShapeDesc.fragment_func.source = kMetalDrawShapeFragment;
        drawShapeDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        drawShapeDesc.uniform_blocks[0].size = sizeof (MetalDrawShapeParams);
        drawShapeDesc.uniform_blocks[0].msl_buffer_n = 0;
        drawShapeDesc.label = "arbit-metal-draw-shape";
        drawShapeShader = sg_make_shader (&drawShapeDesc);
        sg_pipeline_desc drawShapePipelineDesc = {};
        drawShapePipelineDesc.shader = drawShapeShader;
        drawShapePipelineDesc.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
        drawShapePipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        drawShapePipelineDesc.label = "arbit-metal-draw-shape-pipeline";
        drawShapePipeline = sg_make_pipeline (&drawShapePipelineDesc);

        programsReady = resourceValid (sg_query_sampler_state (sampler))
            && resourceValid (sg_query_shader_state (layerShader))
            && resourceValid (sg_query_pipeline_state (layerPipeline))
            && resourceValid (sg_query_shader_state (blendShader))
            && resourceValid (sg_query_pipeline_state (blendPipeline))
            && resourceValid (sg_query_shader_state (frameMixShader))
            && resourceValid (sg_query_pipeline_state (frameMixPipeline))
            && resourceValid (sg_query_shader_state (transitionShader))
            && resourceValid (sg_query_pipeline_state (transitionPipeline))
            && resourceValid (sg_query_shader_state (blitShader))
            && resourceValid (sg_query_pipeline_state (blitPipeline))
            && resourceValid (sg_query_shader_state (previewShader))
            && resourceValid (sg_query_pipeline_state (previewPipeline))
            && resourceValid (sg_query_shader_state (blurShader))
            && resourceValid (sg_query_pipeline_state (blurPipeline))
            && resourceValid (sg_query_shader_state (sharpenShader))
            && resourceValid (sg_query_pipeline_state (sharpenPipeline))
            && resourceValid (sg_query_shader_state (productionFilterShader))
            && resourceValid (sg_query_pipeline_state (productionFilterPipeline))
            && resourceValid (sg_query_shader_state (lutShader))
            && resourceValid (sg_query_pipeline_state (lutPipeline))
            && resourceValid (sg_query_shader_state (colorTransformShader))
            && resourceValid (sg_query_pipeline_state (colorTransformPipeline))
            && resourceValid (sg_query_shader_state (uvEffectShader))
            && resourceValid (sg_query_pipeline_state (uvEffectPipeline))
            && resourceValid (sg_query_shader_state (feedbackShader))
            && resourceValid (sg_query_pipeline_state (feedbackPipeline))
            && resourceValid (sg_query_shader_state (bloomThresholdShader))
            && resourceValid (sg_query_pipeline_state (bloomThresholdPipeline))
            && resourceValid (sg_query_shader_state (postCombineShader))
            && resourceValid (sg_query_pipeline_state (postCombinePipeline))
            && resourceValid (sg_query_shader_state (canvasShader))
            && resourceValid (sg_query_pipeline_state (canvasPipeline))
            && resourceValid (sg_query_shader_state (drawShapeShader))
            && resourceValid (sg_query_pipeline_state (drawShapePipeline));
        if (programsReady)
            ++metalPipelineAllocations;
        else
            error = "Metal production compositor pipeline creation failed";
        return programsReady;
    }

    bool ensureOutputs (const arbitgl::GlFuncs* gl, int w, int h)
    {
        if (accum[0].image.id != 0 && width == w && height == h
            && (directOnly || outputTexture != 0))
            return true;
        destroyOutputs (gl);
        if (! makeTarget (layerTarget, w, h, "arbit-metal-frame-layer-target")
            || ! makeTarget (transitionFrom, w, h, "arbit-metal-frame-transition-from")
            || ! makeTarget (effect[0], w, h, "arbit-metal-frame-effect-a")
            || ! makeTarget (effect[1], w, h, "arbit-metal-frame-effect-b")
            || ! makeTarget (accum[0], w, h, "arbit-metal-frame-accum-a")
            || ! makeTarget (accum[1], w, h, "arbit-metal-frame-accum-b")
            || ! makeTarget (inspectionTarget, w, h, "arbit-metal-inspection-retained"))
        {
            error = "Metal production compositor target creation failed";
            destroyOutputs (gl);
            return false;
        }

        width = w;
        height = h;
        if (directOnly)
            return true;

        CFMutableDictionaryRef props = CFDictionaryCreateMutable (
            kCFAllocatorDefault, 0,
            &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        iosurfaceSetInt (props, kIOSurfaceWidth, w);
        iosurfaceSetInt (props, kIOSurfaceHeight, h);
        iosurfaceSetInt (props, kIOSurfaceBytesPerElement, 4);
        iosurfaceSetInt (props, kIOSurfacePixelFormat, static_cast<int32_t> ('BGRA'));
        surface = IOSurfaceCreate (props);
        CFRelease (props);
        if (surface == nullptr) { error = "Metal compositor IOSurface creation failed"; return false; }
        ++metalIoSurfaceAllocations;

        MTLTextureDescriptor* descriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                               width:w height:h mipmapped:NO];
        descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        metalTexture = [gMetalDevice newTextureWithDescriptor:descriptor iosurface:surface plane:0];
        if (metalTexture == nil) { error = "Metal compositor IOSurface texture creation failed"; return false; }

        sg_image_desc outputDesc = {};
        outputDesc.usage.color_attachment = true;
        outputDesc.width = w;
        outputDesc.height = h;
        outputDesc.pixel_format = SG_PIXELFORMAT_BGRA8;
        outputDesc.sample_count = 1;
#if ! __has_feature(objc_arc)
        // sg_make_image transfers the injected pointer through a temporary
        // which sokol releases after its resource pool has retained it. Keep
        // the compositor's original ownership independent of that transfer.
        [metalTexture retain];
#endif
        outputDesc.mtl_textures[0] = (__bridge const void*) metalTexture;
        outputDesc.label = "arbit-metal-frame-iosurface";
        outputImage = sg_make_image (&outputDesc);
        if (outputImage.id != 0)
            ++metalTextureAllocations;
        sg_view_desc outputViewDesc = {};
        outputViewDesc.color_attachment.image = outputImage;
        outputAttachment = sg_make_view (&outputViewDesc);
        if (outputAttachment.id != 0)
            ++metalTargetAllocations;
        if (! resourceValid (sg_query_image_state (outputImage))
            || ! resourceValid (sg_query_view_state (outputAttachment)))
        { error = "Metal compositor IOSurface attachment creation failed"; return false; }

        CGLContextObj cgl = CGLGetCurrentContext();
        if (cgl == nullptr) { error = "Metal compositor has no current CGL bridge context"; return false; }
        glGenTextures (1, &rectangleTexture);
        glBindTexture (GL_TEXTURE_RECTANGLE, rectangleTexture);
        const CGLError cglError = CGLTexImageIOSurface2D (
            cgl, GL_TEXTURE_RECTANGLE, GL_RGBA8, w, h,
            GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, surface, 0);
        if (cglError != kCGLNoError)
        { error = std::string ("Metal compositor CGL IOSurface import failed: ") + CGLErrorString (cglError); return false; }
        glTexParameteri (GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri (GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture (GL_TEXTURE_RECTANGLE, 0);
        gl->GenFramebuffers (1, &rectangleFbo);
        gl->BindFramebuffer (GL_FRAMEBUFFER, rectangleFbo);
        gl->FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_TEXTURE_RECTANGLE, rectangleTexture, 0);
        if (gl->CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        { error = "Metal compositor IOSurface GL framebuffer incomplete"; return false; }

        glGenTextures (1, &outputTexture);
        glBindTexture (GL_TEXTURE_2D, outputTexture);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        gl->GenFramebuffers (1, &outputFbo);
        gl->BindFramebuffer (GL_FRAMEBUFFER, outputFbo);
        gl->FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_TEXTURE_2D, outputTexture, 0);
        if (gl->CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        { error = "Metal compositor bridge framebuffer incomplete"; return false; }
        gl->BindFramebuffer (GL_FRAMEBUFFER, 0);
        return true;
    }

    bool supports (const LayerDesc* layers, int numLayers,
                   const ImageLayerDesc* overlays, int numOverlays) const
    {
        auto supportsLayer = [this] (const LayerDesc& layer)
        {
            if (layer.drawShape && (layer.isAdjustment || layer.transitionType != 0))
                return false;
            if (layer.pathMatte)
            {
                const auto bounded = [] (float cx, float cy, float width, float height)
                {
                    return std::isfinite(cx) && std::isfinite(cy)
                        && std::isfinite(width) && std::isfinite(height)
                        && cx >= -2.0f && cx <= 2.0f && cy >= -2.0f && cy <= 2.0f
                        && width >= 0.0f && width <= 4.0f
                        && height >= 0.0f && height <= 4.0f;
                };
                if (layer.pathMatteOperation < 0 || layer.pathMatteOperation > 3
                    || ! bounded(layer.pathMatteCx, layer.pathMatteCy,
                                 layer.pathMatteW, layer.pathMatteH)
                    || (layer.pathMatteHasSecondary
                        != (layer.pathMatteOperation != 0))
                    || (layer.pathMatteHasSecondary
                        && ! bounded(layer.pathMatte2Cx, layer.pathMatte2Cy,
                                    layer.pathMatte2W, layer.pathMatte2H)))
                    return false;
            }
            if (layer.shaderSource)
            {
#if ARBIT_HAVE_METAL_GENERATORS
                const auto generator = generators.find (layer.clipId);
                if (generator == generators.end() || generator->second == nullptr
                    || ! generator->second->hasProgram())
                    return false;
#else
                return false;
#endif
            }
            const bool hasNativeTexture = ! layer.nativeTextureBackend.empty()
                || layer.nativeTextureView != 0;
            if (hasNativeTexture)
            {
                const auto& descriptor = layer.nativeTextureDescriptor;
                if (layer.nativeTextureBackend != "metal"
                    || layer.nativeTextureView == 0
                    || layer.nativeTextureView > std::numeric_limits<std::uint32_t>::max()
                    || ! descriptor.complete() || descriptor.backend != "metal"
                    || descriptor.viewKind != arbitgpu::NativeTextureViewKind::Texture2D
                    || descriptor.format != arbitgpu::NativeTexturePixelFormat::Bgra8Unorm
                    || descriptor.textureViewHandle != layer.nativeTextureView
                    || descriptor.width != static_cast<std::uint32_t>(layer.texWidth)
                    || descriptor.height != static_cast<std::uint32_t>(layer.texHeight)
                    || descriptor.deviceOrContextIdentity
                        != reinterpret_cast<std::uintptr_t>((__bridge void*)gMetalDevice))
                    return false;
                const sg_view nativeView {
                    static_cast<std::uint32_t> (layer.nativeTextureView) };
                if (! resourceValid (sg_query_view_state (nativeView)))
                    return false;
            }
            if (! layer.isAdjustment && ! hasNativeTexture && layer.texture != 0
                && sources.find (layer.texture) == sources.end())
                return false;
            if (layer.lutTexture != 0 && luts.find (layer.lutTexture) == luts.end())
                return false;
            const bool hasNativeDepth = ! layer.nativeDepthTextureBackend.empty()
                || layer.nativeDepthTextureView != 0;
            if (hasNativeDepth)
            {
                const auto& descriptor = layer.nativeDepthTextureDescriptor;
                if (layer.nativeDepthTextureBackend != "metal"
                    || layer.nativeDepthTextureView == 0
                    || layer.nativeDepthTextureView > std::numeric_limits<std::uint32_t>::max()
                    || ! descriptor.complete() || descriptor.backend != "metal"
                    || descriptor.viewKind != arbitgpu::NativeTextureViewKind::Texture2D
                    || descriptor.format != arbitgpu::NativeTexturePixelFormat::R32Float
                    || descriptor.textureViewHandle != layer.nativeDepthTextureView
                    || descriptor.width != static_cast<std::uint32_t>(layer.depthWidth)
                    || descriptor.height != static_cast<std::uint32_t>(layer.depthHeight)
                    || descriptor.deviceOrContextIdentity
                        != reinterpret_cast<std::uintptr_t>((__bridge void*)gMetalDevice))
                    return false;
                const sg_view nativeDepthView {
                    static_cast<std::uint32_t> (layer.nativeDepthTextureView) };
                if (! resourceValid (sg_query_view_state (nativeDepthView)))
                    return false;
            }
            if ((layer.depthFog || layer.depthEffect != 0)
                && (layer.depthWidth <= 0 || layer.depthHeight <= 0
                    || (! hasNativeDepth
                        && (layer.depthTexture == 0
                            || depths.find (layer.depthTexture) == depths.end()))))
                return false;
            return true;
        };
        for (int i = 0; i < numLayers; ++i)
        {
            const auto& layer = layers[i];
            if (! supportsLayer (layer)) return false;
            if (layer.transitionType != 0 && layer.fromLayer != nullptr
                && ! supportsLayer (*layer.fromLayer))
                return false;
        }
        for (int i = 0; i < numOverlays; ++i)
            if (overlays[i].texture != 0
                && sources.find (overlays[i].texture) == sources.end())
                return false;
        return true;
    }
};

MetalFrameRenderer::MetalFrameRenderer() : impl_ (std::make_unique<Impl>())
{
    ++metalRendererConstructions;
}
MetalFrameRenderer::~MetalFrameRenderer() = default;

void MetalFrameRenderer::resetAllocationCountersForTesting() noexcept
{
    metalRendererConstructions.store(0);
    metalPipelineAllocations.store(0);
    metalIoSurfaceAllocations.store(0);
    metalTextureAllocations.store(0);
    metalTargetAllocations.store(0);
}

MetalFrameRenderer::AllocationCounters MetalFrameRenderer::allocationCountersForTesting() noexcept
{
    return { metalRendererConstructions.load(), metalPipelineAllocations.load(),
             metalIoSurfaceAllocations.load(), metalTextureAllocations.load(),
             metalTargetAllocations.load() };
}

bool MetalFrameRenderer::queryResourceCapabilities (
    int& maximumImageDimension, uint64_t& maximumBufferLengthBytes,
    uint64_t& recommendedWorkingSetBytes,
    std::string& deviceIdentity) const
{
    std::lock_guard<std::mutex> lock (sokolMutex());
    @autoreleasepool
    {
        maximumImageDimension = 0;
        maximumBufferLengthBytes = 0;
        recommendedWorkingSetBytes = 0;
        deviceIdentity.clear();
        id<MTLDevice> device = (__bridge id<MTLDevice>) arbitgpu::sokolmetal::device();
        if (! impl_->programsReady || device == nil)
            return false;
        maximumImageDimension = sg_query_limits().max_image_size_2d;
        if (@available(macOS 10.14, *))
            maximumBufferLengthBytes = static_cast<uint64_t>(device.maxBufferLength);
        if (@available(macOS 10.12, *))
            recommendedWorkingSetBytes = static_cast<uint64_t>(device.recommendedMaxWorkingSetSize);
        const std::string name = deviceName(device);
        if (@available(macOS 10.13, *))
        {
            char registry[32] = {};
            std::snprintf(registry, sizeof(registry), "0x%llx",
                          static_cast<unsigned long long>(device.registryID));
            deviceIdentity = name + "@" + registry;
        }
        else
        {
            deviceIdentity = name;
        }
        return maximumImageDimension > 0 && ! deviceIdentity.empty();
    }
}

void* MetalFrameRenderer::retainedDevice() const
{
    std::lock_guard<std::mutex> lock (sokolMutex());
    return impl_->programsReady ? arbitgpu::sokolmetal::device() : nullptr;
}

bool MetalFrameRenderer::initialize (const arbitgl::GlFuncs* gl, int width, int height,
                                     std::string& errorOut, bool directOnly)
{
    std::lock_guard<std::mutex> lock (sokolMutex());
    @autoreleasepool
    {
        impl_->directOnly = directOnly;
        if (! ensureSokolMetal() || ! impl_->ensurePrograms()
            || ! impl_->ensureOutputs (gl, width, height))
        {
            errorOut = ! impl_->error.empty() ? impl_->error : gSokolError;
            return false;
        }
    }
    errorOut.clear();
    return true;
}

void MetalFrameRenderer::shutdown (const arbitgl::GlFuncs* gl)
{
    std::lock_guard<std::mutex> lock (sokolMutex());
#if ARBIT_HAVE_METAL_GENERATORS
    for (auto& item : impl_->generators)
        if (item.second != nullptr) item.second->shutdownUnlocked();
    impl_->generators.clear();
    for (auto& item : impl_->operationGenerators)
        if (item.second != nullptr) item.second->shutdownUnlocked();
    impl_->operationGenerators.clear();
#endif
    for (auto& item : impl_->particles)
        if (item.second != nullptr) item.second->shutdownUnlocked (gl);
    impl_->particles.clear();
    for (auto& item : impl_->sources) impl_->destroySource (item.second);
    impl_->sources.clear();
    for (auto& item : impl_->luts) impl_->destroyLut (item.second);
    impl_->luts.clear();
    for (auto& item : impl_->depths) impl_->destroyDepth (item.second);
    impl_->depths.clear();
    impl_->destroyLut (impl_->identityLut);
    impl_->clearDirectOutputs();
    impl_->destroyOutputs (gl);
    if (impl_->layerPipeline.id != 0) sg_destroy_pipeline (impl_->layerPipeline);
    if (impl_->blendPipeline.id != 0) sg_destroy_pipeline (impl_->blendPipeline);
    if (impl_->frameMixPipeline.id != 0) sg_destroy_pipeline (impl_->frameMixPipeline);
    if (impl_->transitionPipeline.id != 0) sg_destroy_pipeline (impl_->transitionPipeline);
    if (impl_->blitPipeline.id != 0) sg_destroy_pipeline (impl_->blitPipeline);
    if (impl_->blurPipeline.id != 0) sg_destroy_pipeline (impl_->blurPipeline);
    if (impl_->sharpenPipeline.id != 0) sg_destroy_pipeline (impl_->sharpenPipeline);
    if (impl_->productionFilterPipeline.id != 0)
        sg_destroy_pipeline (impl_->productionFilterPipeline);
    if (impl_->colorTransformPipeline.id != 0)
        sg_destroy_pipeline (impl_->colorTransformPipeline);
    if (impl_->lutPipeline.id != 0) sg_destroy_pipeline (impl_->lutPipeline);
    if (impl_->uvEffectPipeline.id != 0) sg_destroy_pipeline (impl_->uvEffectPipeline);
    if (impl_->feedbackPipeline.id != 0) sg_destroy_pipeline (impl_->feedbackPipeline);
    if (impl_->bloomThresholdPipeline.id != 0) sg_destroy_pipeline (impl_->bloomThresholdPipeline);
    if (impl_->postCombinePipeline.id != 0) sg_destroy_pipeline (impl_->postCombinePipeline);
    if (impl_->canvasPipeline.id != 0) sg_destroy_pipeline (impl_->canvasPipeline);
    if (impl_->previewPipeline.id != 0) sg_destroy_pipeline (impl_->previewPipeline);
    if (impl_->drawShapePipeline.id != 0) sg_destroy_pipeline (impl_->drawShapePipeline);
    if (impl_->layerShader.id != 0) sg_destroy_shader (impl_->layerShader);
    if (impl_->blendShader.id != 0) sg_destroy_shader (impl_->blendShader);
    if (impl_->frameMixShader.id != 0) sg_destroy_shader (impl_->frameMixShader);
    if (impl_->transitionShader.id != 0) sg_destroy_shader (impl_->transitionShader);
    if (impl_->blitShader.id != 0) sg_destroy_shader (impl_->blitShader);
    if (impl_->blurShader.id != 0) sg_destroy_shader (impl_->blurShader);
    if (impl_->sharpenShader.id != 0) sg_destroy_shader (impl_->sharpenShader);
    if (impl_->productionFilterShader.id != 0)
        sg_destroy_shader (impl_->productionFilterShader);
    if (impl_->colorTransformShader.id != 0)
        sg_destroy_shader (impl_->colorTransformShader);
    if (impl_->lutShader.id != 0) sg_destroy_shader (impl_->lutShader);
    if (impl_->uvEffectShader.id != 0) sg_destroy_shader (impl_->uvEffectShader);
    if (impl_->feedbackShader.id != 0) sg_destroy_shader (impl_->feedbackShader);
    if (impl_->bloomThresholdShader.id != 0) sg_destroy_shader (impl_->bloomThresholdShader);
    if (impl_->postCombineShader.id != 0) sg_destroy_shader (impl_->postCombineShader);
    if (impl_->canvasShader.id != 0) sg_destroy_shader (impl_->canvasShader);
    if (impl_->previewShader.id != 0) sg_destroy_shader (impl_->previewShader);
    if (impl_->drawShapeShader.id != 0) sg_destroy_shader (impl_->drawShapeShader);
    if (impl_->sampler.id != 0) sg_destroy_sampler (impl_->sampler);
    impl_->layerPipeline = impl_->blendPipeline = impl_->frameMixPipeline = {};
    impl_->transitionPipeline = impl_->blitPipeline = {};
    impl_->layerShader = impl_->blendShader = impl_->frameMixShader = {};
    impl_->transitionShader = impl_->blitShader = {};
    impl_->blurPipeline = impl_->sharpenPipeline = impl_->lutPipeline = {};
    impl_->blurShader = impl_->sharpenShader = impl_->lutShader = {};
    impl_->productionFilterPipeline = {};
    impl_->productionFilterShader = {};
    impl_->colorTransformPipeline = {};
    impl_->colorTransformShader = {};
    impl_->uvEffectPipeline = {};
    impl_->uvEffectShader = {};
    impl_->feedbackPipeline = {};
    impl_->feedbackShader = {};
    impl_->bloomThresholdPipeline = impl_->postCombinePipeline = {};
    impl_->bloomThresholdShader = impl_->postCombineShader = {};
    impl_->canvasPipeline = {};
    impl_->previewPipeline = {};
    impl_->drawShapePipeline = {};
    impl_->canvasShader = {};
    impl_->previewShader = {};
    impl_->drawShapeShader = {};
    impl_->sampler = {};
    impl_->programsReady = false;
}

void MetalFrameRenderer::setOutputSize (const arbitgl::GlFuncs* gl, int width, int height)
{
    if (width == impl_->width && height == impl_->height) return;
    std::lock_guard<std::mutex> lock (sokolMutex());
    @autoreleasepool { impl_->ensureOutputs (gl, width, height); }
}

void MetalFrameRenderer::setCanvas (int width, int height)
{ impl_->canvasWidth = width; impl_->canvasHeight = height; }

void MetalFrameRenderer::setPresentSize (int width, int height)
{ impl_->presentWidth = width; impl_->presentHeight = height; }

void MetalFrameRenderer::setView (float zoom, float panX, float panY)
{ impl_->zoom = zoom; impl_->panX = panX; impl_->panY = panY; }

void MetalFrameRenderer::setBackgroundColor (float r, float g, float b, float a)
{ impl_->bg[0] = r; impl_->bg[1] = g; impl_->bg[2] = b; impl_->bg[3] = a; }

void MetalFrameRenderer::setPostFx (float bloomIntensity, float bloomThreshold,
                                    float bloomRadius, int tonemap, float exposure)
{
    impl_->bloomIntensity = std::max (bloomIntensity, 0.0f);
    impl_->bloomThreshold = bloomThreshold;
    impl_->bloomRadius = std::max (bloomRadius, 0.0f);
    impl_->tonemap = tonemap >= 0 && tonemap <= 2 ? tonemap : 0;
    impl_->exposure = exposure > 0.0f ? exposure : 1.0f;
}

void MetalFrameRenderer::setInspection (
    int transformClipId, unsigned requestedHandle,
    const videopreview::State& presentation)
{
    impl_->inspectionClipId = transformClipId;
    impl_->inspectionRequestedHandle = requestedHandle;
    impl_->inspectionPresentation = presentation;
    impl_->inspectionAdmitted = false;
    impl_->inspectionRetainedHandle = 0;
}

unsigned MetalFrameRenderer::inspectionHandle() const
{ return impl_->inspectionAdmitted ? impl_->inspectionRetainedHandle : 0; }

bool MetalFrameRenderer::inspectionResourceAdmitted() const
{ return impl_->inspectionAdmitted; }

void MetalFrameRenderer::uploadRgba (unsigned handle, const uint8_t* rgba,
                                     int width, int height, int strideBytes)
{
    if (handle == 0 || rgba == nullptr || width <= 0 || height <= 0) return;
    std::lock_guard<std::mutex> lock (sokolMutex());
    auto& source = impl_->sources[handle];
    if (source.frameBlend || source.image.id == 0
        || source.width != width || source.height != height)
    {
        impl_->destroySource (source);
        sg_image_desc imageDesc = {};
        imageDesc.usage.stream_update = true;
        imageDesc.width = width;
        imageDesc.height = height;
        imageDesc.pixel_format = SG_PIXELFORMAT_RGBA8;
        imageDesc.label = "arbit-metal-frame-source";
        source.image = sg_make_image (&imageDesc);
        sg_view_desc viewDesc = {};
        viewDesc.texture.image = source.image;
        source.view = sg_make_view (&viewDesc);
        source.width = width;
        source.height = height;
    }
    if (! resourceValid (sg_query_image_state (source.image))) return;
    const size_t packedBytes = static_cast<size_t> (width) * static_cast<size_t> (height) * 4;
    source.pixels.resize (packedBytes);
    for (int y = 0; y < height; ++y)
        std::memcpy (source.pixels.data() + static_cast<size_t> (y) * width * 4,
                     rgba + static_cast<size_t> (y) * strideBytes,
                     static_cast<size_t> (width) * 4);
    source.dirty = true;
}

bool MetalFrameRenderer::uploadR16 (unsigned handle, const uint16_t* pixels,
                                    int width, int height)
{
    if (handle == 0 || pixels == nullptr || width <= 0 || height <= 0
        || width > 16384 || height > 16384) return false;
    std::lock_guard<std::mutex> lock (sokolMutex());
    auto& depth = impl_->depths[handle];
    impl_->destroyDepth (depth);
    const size_t pixelCount = static_cast<size_t> (width) * static_cast<size_t> (height);
    std::vector<float> expanded (pixelCount);
    std::transform (pixels, pixels + pixelCount, expanded.begin(), [] (uint16_t value)
    {
        return static_cast<float> (value) / 65535.0f;
    });
    sg_image_desc desc = {};
    desc.width = width; desc.height = height; desc.pixel_format = SG_PIXELFORMAT_R32F;
    desc.data.mip_levels[0] = { expanded.data(), pixelCount * sizeof (float) };
    desc.label = "arbit-metal-depth-r32f";
    depth.image = sg_make_image (&desc);
    sg_view_desc viewDesc = {}; viewDesc.texture.image = depth.image;
    depth.view = sg_make_view (&viewDesc); depth.width = width; depth.height = height;
    if (! resourceValid (sg_query_image_state (depth.image))
        || ! resourceValid (sg_query_view_state (depth.view)))
    {
        impl_->destroyDepth (depth); impl_->depths.erase (handle);
        impl_->error = "Metal R16 depth texture creation failed"; return false;
    }
    return true;
}

void MetalFrameRenderer::setFrameBlend (unsigned handle, unsigned textureA,
                                        unsigned textureB, int width, int height,
                                        float mix)
{
    if (handle == 0 || textureA == 0 || textureB == 0
        || width <= 0 || height <= 0)
        return;
    std::lock_guard<std::mutex> lock (sokolMutex());
    auto& source = impl_->sources[handle];
    if (! source.frameBlend || source.width != width || source.height != height)
    {
        impl_->destroySource (source);
        source.width = width;
        source.height = height;
        source.frameBlend = true;
    }
    source.textureA = textureA;
    source.textureB = textureB;
    source.blendMix = std::clamp (mix, 0.0f, 1.0f);
}

void MetalFrameRenderer::uploadLut3D (unsigned handle, const float* rgbTriples, int size)
{
    if (handle == 0 || rgbTriples == nullptr || size < 2) return;
    std::lock_guard<std::mutex> lock (sokolMutex());
    auto& lut = impl_->luts[handle];
    if (! impl_->makeLut (lut, rgbTriples, size, "arbit-metal-frame-lut"))
        impl_->error = "Metal 3D LUT upload failed";
}

void MetalFrameRenderer::deleteTexture (unsigned handle)
{
    std::lock_guard<std::mutex> lock (sokolMutex());
    auto it = impl_->sources.find (handle);
    if (it != impl_->sources.end())
    {
        impl_->destroySource (it->second);
        impl_->sources.erase (it);
    }
    auto lut = impl_->luts.find (handle);
    if (lut != impl_->luts.end())
    {
        impl_->destroyLut (lut->second);
        impl_->luts.erase (lut);
    }
    auto depth = impl_->depths.find (handle);
    if (depth != impl_->depths.end())
    {
        impl_->destroyDepth (depth->second);
        impl_->depths.erase (depth);
    }
}

bool MetalFrameRenderer::setClipShader (int clipId, const std::string& source,
                                        std::string& logOut,
                                        std::vector<GenParam>& paramsOut)
{
#if ARBIT_HAVE_METAL_GENERATORS
    std::lock_guard<std::mutex> lock (sokolMutex());
    auto& generator = impl_->generators[clipId];
    if (generator == nullptr) generator = std::make_unique<MetalShaderGenerator>();
    const bool ok = generator->setSource (source);
    logOut = generator->log();
    paramsOut = generator->params();
    return ok;
#else
    (void) clipId; (void) source; (void) paramsOut;
    logOut = "native Metal shader compiler was not linked";
    return false;
#endif
}

bool MetalFrameRenderer::prepareShaderOperationPlan (const LayerDesc& layer,
                                                      std::string& errorOut)
{
#if ARBIT_HAVE_METAL_GENERATORS
    std::lock_guard<std::mutex> lock (sokolMutex());
    if (layer.shaderOperationPlan == nullptr || layer.shaderOperationPlan->operations.empty())
    { errorOut = "Metal shader operation plan is missing"; return false; }
    if (layer.shaderOperationPlan->operations.size()
            > videowire::ShaderOperationPlan::maximumOperations
        || layer.shaderOperationPlan->passTargetCount
            > videowire::ShaderOperationPlan::maximumPassTargets)
    { errorOut = "Metal shader operation plan exceeds its resource budget"; return false; }
    if (layer.shaderOperationPlan->revision != layer.visualPlanStructuralRevision
        || layer.shaderOperationPlan->digest.empty()
        || layer.shaderOperationPlan->digest != videowire::shaderOperationPlanDigest(
            *layer.shaderOperationPlan, layer.shaderOperationPlan->revision))
    { errorOut = "Metal shader operation plan identity differs from its immutable content"; return false; }
    std::size_t countedPasses = 0;
    for (const auto& operation : layer.shaderOperationPlan->operations)
    {
        countedPasses += operation.payload.passResources.passes.size();
        videowire::CuratedIsfPassResources sourcePasses;
        if (operation.payload.language == videowire::FlatShaderLanguage::isf
            && (!videowire::admitCuratedIsfPassResources(
                    operation.payload.source, sourcePasses, errorOut)
                || !(sourcePasses == operation.payload.passResources)))
        {
            if (errorOut.empty())
                errorOut = "Metal shader operation multipass schedule differs from its source";
            return false;
        }
        if (operation.nodeId == 0 || operation.outputNodeId == 0
            || videohelper::sha256Text(operation.payload.source) != operation.payload.sourceSha256
            || operation.generatedParameters != operation.payload.parameterValues
            || countedPasses > videowire::ShaderOperationPlan::maximumPassTargets
            || !videowire::admitCuratedIsfPassExtent(
                operation.payload.passResources, impl_->width, impl_->height, errorOut))
        { if (errorOut.empty()) errorOut = "Metal shader operation plan lost its exact payload"; return false; }
        if (operation.customGrant)
        {
            const auto kind = operation.payload.language == videowire::FlatShaderLanguage::isf
                ? programmableruntime::PayloadKind::isf : programmableruntime::PayloadKind::shader;
            if (!programmableruntime::admits(*operation.customGrant, kind,
                    operation.payload.source, errorOut)
                || operation.customGrant->verifiedBundledCurated
                || !operation.customGrant->catalogPackId.empty()
                || !operation.customGrant->catalogProgramId.empty()
                || !operation.payload.catalogPackId.empty()
                || !operation.payload.catalogProgramId.empty())
            { if (errorOut.empty()) errorOut = "Metal custom grant no longer owns the exact payload"; return false; }
        }
    }
    if (countedPasses != layer.shaderOperationPlan->passTargetCount)
    { errorOut = "Metal shader operation plan resource budget is stale"; return false; }
    const std::string ownerPrefix = std::to_string(layer.clipId) + ":";
    const std::string currentPrefix = ownerPrefix
        + std::to_string(layer.visualPlanStructuralRevision) + ":"
        + layer.shaderOperationPlan->digest + ":";
    std::map<std::string, std::unique_ptr<MetalShaderGenerator>> staged;
    struct StagedMetalCleanup
    {
        std::map<std::string, std::unique_ptr<MetalShaderGenerator>>& shaders;
        bool committed = false;
        ~StagedMetalCleanup()
        {
            if (committed) return;
            for (auto& shader : shaders)
                if (shader.second != nullptr) shader.second->shutdownUnlocked();
        }
    } stagedCleanup { staged };
    for (const auto& operation : layer.shaderOperationPlan->operations)
    {
        const std::string key = std::to_string(layer.clipId) + ":"
            + std::to_string(layer.visualPlanStructuralRevision) + ":"
            + layer.shaderOperationPlan->digest + ":"
            + std::to_string(operation.nodeId) + ":" + operation.payload.sourceSha256;
        if (impl_->operationGenerators.count(key) != 0) continue;
        auto generator = std::make_unique<MetalShaderGenerator>();
        if (! generator->setSource(operation.payload.source))
        { errorOut = generator->log(); return false; }
        const auto* catalog = shadercatalog::find(operation.payload.catalogPackId,
                                                   operation.payload.catalogProgramId);
        std::vector<videowire::FlatShaderRuntimeParameter> runtimeParameters;
        for (const auto& parameter : generator->params())
        {
            const auto components = genParamComponentCount(parameter.type);
            if (components == 0) continue;
            const char* type = parameter.type == 0 ? "bool" : parameter.type == 1 ? "long"
                : parameter.type == 2 ? "float" : parameter.type == 3 ? "point2D"
                : parameter.type == 4 ? "color" : "";
            runtimeParameters.push_back({parameter.name, type,
                                         static_cast<std::size_t>(components)});
        }
        const bool valid = operation.customGrant.has_value()
            ? runtimeParameters.empty() && operation.generatedParameters.empty()
            : catalog != nullptr && videowire::validateFlatShaderRuntimeParameters(
                *catalog, runtimeParameters, operation.generatedParameters, errorOut);
        if (! valid)
        { if (errorOut.empty()) errorOut = "Metal shader parameters differ from the admitted manifest";
          return false; }
        staged.emplace(key, std::move(generator));
    }
    for (auto entry = impl_->operationGenerators.begin();
         entry != impl_->operationGenerators.end();)
    {
        if (entry->first.rfind(ownerPrefix, 0) == 0
            && entry->first.rfind(currentPrefix, 0) != 0)
        {
            if (entry->second != nullptr) entry->second->shutdownUnlocked();
            entry = impl_->operationGenerators.erase(entry);
        }
        else ++entry;
    }
    for (auto& candidate : staged)
        impl_->operationGenerators.emplace(candidate.first, std::move(candidate.second));
    stagedCleanup.committed = true;
    errorOut.clear();
    return true;
#else
    (void) layer;
    errorOut = "native Metal shader compiler was not linked";
    return false;
#endif
}

void MetalFrameRenderer::clearClipShader (int clipId)
{
#if ARBIT_HAVE_METAL_GENERATORS
    std::lock_guard<std::mutex> lock (sokolMutex());
    const auto generator = impl_->generators.find (clipId);
    if (generator != impl_->generators.end())
    {
        if (generator->second != nullptr) generator->second->shutdownUnlocked();
        impl_->generators.erase (generator);
    }
    const std::string ownerPrefix = std::to_string(clipId) + ":";
    for (auto entry = impl_->operationGenerators.begin();
         entry != impl_->operationGenerators.end();)
    {
        if (entry->first.rfind(ownerPrefix, 0) == 0)
        {
            if (entry->second != nullptr) entry->second->shutdownUnlocked();
            entry = impl_->operationGenerators.erase(entry);
        }
        else ++entry;
    }
#else
    (void) clipId;
#endif
}

bool MetalFrameRenderer::hasClipShader (int clipId) const
{
#if ARBIT_HAVE_METAL_GENERATORS
    const auto generator = impl_->generators.find (clipId);
    return generator != impl_->generators.end() && generator->second != nullptr
        && generator->second->hasProgram();
#else
    (void) clipId;
    return false;
#endif
}

void MetalFrameRenderer::setClipImage (int clipId, const std::string& name,
                                       const uint8_t* rgba, int width, int height,
                                       int strideBytes)
{
#if ARBIT_HAVE_METAL_GENERATORS
    std::lock_guard<std::mutex> lock (sokolMutex());
    const auto generator = impl_->generators.find (clipId);
    if (generator != impl_->generators.end() && generator->second != nullptr)
        generator->second->setImage (name, rgba, width, height, strideBytes);
#else
    (void) clipId; (void) name; (void) rgba; (void) width; (void) height;
    (void) strideBytes;
#endif
}

unsigned MetalFrameRenderer::renderComposite (const arbitgl::GlFuncs* gl,
                                              const LayerDesc* layers, int numLayers,
                                              const ImageLayerDesc* overlays, int numOverlays)
{
    std::lock_guard<std::mutex> lock (sokolMutex());
    @autoreleasepool
    {
        const bool direct = impl_->requestedDirectSurface != nullptr;
        if (! impl_->programsReady || (! direct && impl_->outputTexture == 0)
            || ! impl_->supports (layers, numLayers, overlays, numOverlays))
            return 0;
        ++impl_->frameParity;
        impl_->particleBackend = "none";
        impl_->inspectionAdmitted = false;
        impl_->inspectionRetainedHandle = 0;

#if ARBIT_HAVE_METAL_GENERATORS
        // Submit dynamic generators before the first Sokol draw/update opens
        // this frame's compositor command buffer. Both use Sokol's Metal queue,
        // so commit order provides GPU-side synchronization without a CPU wait.
        std::unordered_map<int, sg_view> generatedViews;
        auto renderGenerator = [&] (const LayerDesc& layer) -> bool
        {
            if (! layer.shaderSource || generatedViews.count (layer.clipId) != 0)
                return true;
            const auto generator = impl_->generators.find (layer.clipId);
            if (generator == impl_->generators.end() || generator->second == nullptr)
                return false;
            sg_view view = {};
            const auto started = std::chrono::steady_clock::now();
            view.id = generator->second->renderViewUnlocked (
                layer.shaderClock, impl_->width, impl_->height,
                layer.audioPresent ? &layer.audioFeatures : nullptr,
                canonicalblockc::valid(layer.canonicalBlockCFrame) ? layer.canonicalBlockCFrame.get() : nullptr,
                &layer.genParams);
            if (impl_->visualTelemetry != nullptr)
                impl_->visualTelemetry->recordExecutionObservation(
                    videowire::VisualExecutionKind::generator,
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - started).count()));
            if (view.id == 0)
            {
                impl_->error = generator->second->log();
                return false;
            }
            generatedViews[layer.clipId] = view;
            return true;
        };
        for (int layerIndex = 0; layerIndex < numLayers; ++layerIndex)
        {
            if (! renderGenerator (layers[layerIndex])) return 0;
            if (layers[layerIndex].fromLayer != nullptr
                && ! renderGenerator (*layers[layerIndex].fromLayer)) return 0;
        }

#endif

        // Apply the latest CPU decode for each source once in this sokol frame.
        // Upload calls may outnumber composites while the GL recovery path is
        // active; deferring here avoids sokol's one-update-per-image-per-frame
        // rule without adding extra commits or stalling the decode thread.
        for (auto& item : impl_->sources)
        {
            auto& source = item.second;
            if (! source.dirty) continue;
            sg_image_data data = {};
            data.mip_levels[0] = { source.pixels.data(), source.pixels.size() };
            sg_update_image (source.image, &data);
            source.dirty = false;
        }

        // Tier-1 retiming remains GPU-resident in Metal-only mode. Decoded
        // bracket frames are sampled directly into a reusable RGBA16F target;
        // no GL texture, framebuffer, or CPU blend is created.
        for (auto& item : impl_->sources)
        {
            auto& source = item.second;
            if (! source.frameBlend) continue;
            const auto earlier = impl_->sources.find (source.textureA);
            const auto later = impl_->sources.find (source.textureB);
            if (earlier == impl_->sources.end() || later == impl_->sources.end()
                || earlier->second.frameBlend || later->second.frameBlend)
            {
                impl_->error = "Metal frame blend has invalid source textures";
                return 0;
            }
            if (source.blendTarget.image.id == 0
                && ! impl_->makeTarget (source.blendTarget, source.width, source.height,
                                        "arbit-metal-frame-mix-target"))
            {
                impl_->error = "Metal frame blend target creation failed";
                return 0;
            }
            sg_pass pass = {};
            pass.attachments.colors[0] = source.blendTarget.attachment;
            pass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
            pass.action.colors[0].store_action = SG_STOREACTION_STORE;
            sg_begin_pass (&pass);
            sg_apply_pipeline (impl_->frameMixPipeline);
            sg_bindings bindings = {};
            bindings.views[0] = earlier->second.view;
            bindings.views[1] = later->second.view;
            bindings.samplers[0] = impl_->sampler;
            sg_apply_bindings (&bindings);
            const MetalFrameMixParams params = {
                source.blendMix, { 0.0f, 0.0f, 0.0f } };
            const sg_range range = { &params, sizeof (params) };
            sg_apply_uniforms (0, &range);
            sg_draw (0, 3, 1);
            sg_end_pass();
        }

        // Ordered operations use explicit Metal command buffers. Submit all
        // current-frame Sokol uploads and blends first. Both paths use the same
        // Metal queue, so queue order makes those sources visible to every
        // operation and makes the later compositor wait for operation output.
        sg_commit();

        for (int layerIndex = 0; layerIndex < numLayers; ++layerIndex)
        {
            const auto& layer = layers[layerIndex];
            if (layer.shaderOperationPlan == nullptr) continue;
            struct Resource { sg_view view {}; std::uintptr_t texture = 0; };
            std::map<int, Resource> resources;
            auto sourceResource = [&] (const LayerDesc& owner) -> Resource
            {
                const auto found = impl_->sources.find(owner.texture);
                if (found == impl_->sources.end()) return {};
                const auto& source = found->second;
                const sg_view view = source.frameBlend ? source.blendTarget.texture : source.view;
                const sg_image image = source.frameBlend ? source.blendTarget.image : source.image;
                const auto native = sg_mtl_query_image_info(image);
                return { view, reinterpret_cast<std::uintptr_t>(native.tex[native.active_slot]) };
            };
            for (const auto& operation : layer.shaderOperationPlan->operations)
                for (std::size_t input = 0; input < operation.inputCount; ++input)
                    if (resources.count(operation.inputNodeIds[input]) == 0)
                    {
                        const LayerDesc* owner = input == 0
                            && operation.kind == videowire::ShaderOperationKind::transition
                            ? layer.fromLayer : &layer;
                        if (owner != nullptr)
                            resources[operation.inputNodeIds[input]] = sourceResource(*owner);
                    }
            Resource last;
            for (const auto& operation : layer.shaderOperationPlan->operations)
            {
                const std::string key = std::to_string(layer.clipId) + ":"
                    + std::to_string(layer.visualPlanStructuralRevision) + ":"
                    + layer.shaderOperationPlan->digest + ":"
                    + std::to_string(operation.nodeId) + ":" + operation.payload.sourceSha256;
                const auto found = impl_->operationGenerators.find(key);
                if (found == impl_->operationGenerators.end() || found->second == nullptr)
                { impl_->error = "Metal shader operation was not admitted"; return 0; }
                std::map<std::string, std::uintptr_t> images;
                if (operation.kind == videowire::ShaderOperationKind::filter)
                    images["inputImage"] = resources[operation.inputNodeIds[0]].texture;
                else if (operation.kind == videowire::ShaderOperationKind::transition)
                {
                    images["startImage"] = resources[operation.inputNodeIds[0]].texture;
                    images["endImage"] = resources[operation.inputNodeIds[1]].texture;
                }
                if (std::any_of(images.begin(), images.end(),
                    [] (const auto& image) { return image.second == 0; }))
                { impl_->error = "Metal shader operation references an unknown resource"; return 0; }
                const auto values = layer.shaderOperationParameters.find(operation.nodeId);
                last.view.id = found->second->renderViewUnlocked(layer.shaderClock,
                    impl_->width, impl_->height,
                    layer.audioPresent ? &layer.audioFeatures : nullptr,
                    canonicalblockc::valid(layer.canonicalBlockCFrame) ? layer.canonicalBlockCFrame.get() : nullptr,
                    values == layer.shaderOperationParameters.end() ? nullptr : &values->second,
                    &images);
                last.texture = found->second->outputTextureHandle();
                if (last.view.id == 0 || last.texture == 0)
                { impl_->error = found->second->log(); return 0; }
                resources[operation.outputNodeId] = last;
                resources[operation.nodeId] = last;
            }
            generatedViews[layer.clipId] = last.view;
        }
        auto sourceViewFor = [&] (unsigned handle) -> sg_view
        {
            const auto source = impl_->sources.find (handle);
            if (source == impl_->sources.end()) return {};
            return source->second.frameBlend
                ? source->second.blendTarget.texture : source->second.view;
        };

        int read = 0;
        sg_pass clearPass = {};
        clearPass.attachments.colors[0] = impl_->accum[read].attachment;
        clearPass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
        clearPass.action.colors[0].store_action = SG_STOREACTION_STORE;
        clearPass.action.colors[0].clear_value = {
            impl_->bg[0], impl_->bg[1], impl_->bg[2], impl_->bg[3] };
        sg_begin_pass (&clearPass);
        sg_end_pass();

        auto drawLayerTo = [&] (const LayerDesc& layer, Impl::Target& target,
                                float opacity, sg_view sourceOverride) -> bool
        {
            if ((layer.texture == 0 && layer.nativeTextureView == 0
                 && sourceOverride.id == 0
                 && ! layer.particleSource && ! layer.shaderSource)
                || opacity <= 0.0f)
            {
                sg_pass clear = {};
                clear.attachments.colors[0] = target.attachment;
                clear.action.colors[0].load_action = SG_LOADACTION_CLEAR;
                clear.action.colors[0].store_action = SG_STOREACTION_STORE;
                clear.action.colors[0].clear_value = { 0, 0, 0, 0 };
                sg_begin_pass (&clear);
                sg_end_pass();
                return true;
            }
            sg_view sourceView = sourceOverride;
            if (sourceView.id == 0)
            {
                if (layer.shaderSource || layer.shaderOperationPlan != nullptr)
                {
#if ARBIT_HAVE_METAL_GENERATORS
                    const auto generated = generatedViews.find (layer.clipId);
                    if (generated == generatedViews.end()) return false;
                    sourceView = generated->second;
#else
                    return false;
#endif
                }
                else if (layer.particleSource)
                {
                    ParticleParams params;
                    auto value = [&layer] (const char* key, double fallback)
                    {
                        const auto item = layer.genParams.find (key);
                        return item != layer.genParams.end() ? item->second : fallback;
                    };
                    params.count = static_cast<int> (value ("count", 512.0) + 0.5);
                    params.spawnTrack = static_cast<int> (value ("spawnTrack", 0.0) + 0.5);
                    params.size = static_cast<float> (value ("size", 2.0));
                    params.gravity = static_cast<float> (value ("gravity", 0.0));
                    params.force = static_cast<float> (value ("force", 1.0));
                    auto& engine = impl_->particles[layer.clipId];
                    if (engine == nullptr)
                        engine = std::make_unique<MetalParticleEngine>();
                    const auto started = std::chrono::steady_clock::now();
                    sourceView.id = engine->renderMetalViewUnlocked (
                        layer.shaderClock, impl_->width, impl_->height, params,
                        canonicalblockc::valid(layer.canonicalBlockCFrame) ? layer.canonicalBlockCFrame.get() : nullptr);
                    if (impl_->visualTelemetry != nullptr)
                    {
                        const auto elapsed = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - started).count());
                        impl_->visualTelemetry->recordExecutionObservation(
                            videowire::VisualExecutionKind::particle, elapsed);
                        if (layer.particleNodeId != 0)
                            impl_->visualTelemetry->recordNodeEvaluation(layer.clipId,
                                layer.visualPlanStructuralRevision, layer.particleNodeId, elapsed,
                                layer.visualPlanTelemetryHold);
                    }
                    if (sourceView.id == 0)
                    {
                        impl_->particleBackend = "metal-rejected";
                        impl_->error = engine->log();
                        return false;
                    }
                    impl_->particleBackend = "metal-compute";
                }
                else if (layer.nativeTextureBackend == "metal")
                    sourceView.id = static_cast<std::uint32_t> (layer.nativeTextureView);
                else
                    sourceView = sourceViewFor (layer.texture);
                if (sourceView.id == 0) return false;
            }

            MetalEffectParams effects = makeEffectParams (layer);
            float blurRadius = 0.0f;
            float sharpenAmount = 0.0f;
            bool blurOn = false, sharpenOn = false;
            bool orderedEffectPassOn = layer.lutTexture != 0;
            int orderedEffectBits = 0;
            for (int fx = 0; layer.effects != nullptr && fx < layer.effectCount; ++fx)
            {
                const auto& effect = layer.effects[fx];
                if (! effect.enabled) continue;
                if (effect.type == static_cast<int> (videofx::EffectType::Blur))
                {
                    blurRadius = effect.params[0];
                    blurOn = true;
                }
                else if (effect.type == static_cast<int> (videofx::EffectType::Sharpen))
                {
                    sharpenAmount = effect.params[0];
                    sharpenOn = true;
                }
                else if (metalUvEffectMode (effect.type) >= 0
                         || metalProductionFilterMode (effect.type) >= 0
                         || effect.type == static_cast<int> (videofx::EffectType::FeedbackTrail))
                {
                    orderedEffectPassOn = true;
                    orderedEffectBits |= videofx::kEffectBits[effect.type];
                }
            }

            sg_view processed = sourceView;
            int processedTarget = -1;
            auto nextEffectTarget = [&]() -> int
            {
                processedTarget = processedTarget < 0 ? 0 : processedTarget ^ 1;
                return processedTarget;
            };
            auto filterPass = [&] (sg_pipeline pipeline, const MetalFilterParams& params,
                                  sg_view input, sg_view lut, int targetIndex)
            {
                sg_pass pass = {};
                pass.attachments.colors[0] = impl_->effect[targetIndex].attachment;
                pass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                pass.action.colors[0].store_action = SG_STOREACTION_STORE;
                sg_begin_pass (&pass);
                sg_apply_pipeline (pipeline);
                sg_bindings bindings = {};
                bindings.views[0] = input;
                bindings.views[1] = lut;
                bindings.samplers[0] = impl_->sampler;
                sg_apply_bindings (&bindings);
                const sg_range range = { &params, sizeof (params) };
                sg_apply_uniforms (0, &range);
                sg_draw (0, 3, 1);
                sg_end_pass();
            };

            if (layer.graphColorTransformActive)
            {
                auto description = layer.graphColorTransform;
                const colortransform::Extent dispatchExtent {
                    static_cast<std::uint32_t> (impl_->width),
                    static_cast<std::uint32_t> (impl_->height) };
                description.input.extent = dispatchExtent;
                description.output.extent = dispatchExtent;
                colortransform::AdmissionFailure failure =
                    colortransform::AdmissionFailure::None;
                const auto admitted = colortransform::admit (
                    description, colortransform::BackendCapability::NativeGpu, failure);
                if (! admitted)
                {
                    impl_->error = std::string ("Metal color transform admission failed: ")
                        + std::string (colortransform::token (failure));
                    return false;
                }

                const auto& transform = admitted->description();
                MetalColorTransformParams params = {};
                const auto inputToWorking =
                    colortransform::gpumath::inputToWorking (transform);
                const auto workingToOutput =
                    colortransform::gpumath::workingToOutput (transform);
                const auto workingLuma =
                    colortransform::gpumath::workingLuma (transform);
                std::copy (inputToWorking.begin(), inputToWorking.end(),
                           params.inputToWorking);
                std::copy (workingToOutput.begin(), workingToOutput.end(),
                           params.workingToOutput);
                std::copy (workingLuma.begin(), workingLuma.end(),
                           params.workingLuma);
                params.luminance[0] = static_cast<float> (
                    transform.luminance.sourceReferenceWhiteNits);
                params.luminance[1] = static_cast<float> (
                    transform.luminance.sourcePeakNits);
                params.luminance[2] = static_cast<float> (
                    transform.luminance.outputReferenceWhiteNits);
                params.luminance[3] = static_cast<float> (
                    transform.luminance.outputPeakNits);
                params.options[0] = static_cast<std::int32_t> (transform.input.transfer);
                params.options[1] = static_cast<std::int32_t> (transform.output.transfer);
                params.options[2] = static_cast<std::int32_t> (transform.input.alpha);
                params.options[3] = static_cast<std::int32_t> (transform.output.alpha);
                params.rendering[0] = static_cast<std::int32_t> (transform.toneMap);
                params.rendering[1] = transform.outputIntent
                        == colortransform::OutputIntent::SdrDisplay
                    || transform.output.format == colortransform::PixelFormat::R16
                    || transform.output.format == colortransform::PixelFormat::RGBA8;

                const int targetIndex = nextEffectTarget();
                sg_pass pass = {};
                pass.attachments.colors[0] = impl_->effect[targetIndex].attachment;
                pass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                pass.action.colors[0].store_action = SG_STOREACTION_STORE;
                sg_begin_pass (&pass);
                sg_apply_pipeline (impl_->colorTransformPipeline);
                sg_bindings bindings = {};
                bindings.views[0] = processed;
                bindings.samplers[0] = impl_->sampler;
                sg_apply_bindings (&bindings);
                const sg_range range = { &params, sizeof (params) };
                sg_apply_uniforms (0, &range);
                sg_draw (0, 3, 1);
                sg_end_pass();
                processed = impl_->effect[targetIndex].texture;
            }

            const int blurBits = videofx::kEffectBits[
                static_cast<int> (videofx::EffectType::Blur)]
                | videofx::kEffectBits[
                    static_cast<int> (videofx::EffectType::Sharpen)];
            const bool colorEffectsPreprocessed =
                (effects.mask & ~(blurBits | orderedEffectBits)) != 0
                && (blurOn || sharpenOn || orderedEffectPassOn);
            if (colorEffectsPreprocessed)
            {
                const int targetIndex = nextEffectTarget();
                const MetalGeometryParams identityGeometry = {
                    { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
                    { 0, 0, 0, 0 } };
                const auto fullMask = fullFrameMaskParams();
                effects.mask &= ~blurBits;
                effects.lutEnabled = 0.0f;
                effects.lutSize = 0.0f;
                sg_pass pass = {};
                pass.attachments.colors[0] = impl_->effect[targetIndex].attachment;
                pass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                pass.action.colors[0].store_action = SG_STOREACTION_STORE;
                sg_begin_pass (&pass);
                sg_apply_pipeline (impl_->layerPipeline);
                sg_bindings bindings = {};
                bindings.views[0] = processed;
                bindings.views[1] = impl_->identityLut.view;
                bindings.views[2] = processed;
                bindings.views[3] = processed;
                // The shared fragment samples depth before checking its mode.
                bindings.views[4] = processed;
                bindings.samplers[0] = impl_->sampler;
                sg_apply_bindings (&bindings);
                const sg_range geometryRange = { &identityGeometry, sizeof (identityGeometry) };
                const sg_range maskRange = { &fullMask, sizeof (fullMask) };
                const sg_range effectsRange = { &effects, sizeof (effects) };
                sg_apply_uniforms (0, &geometryRange);
                sg_apply_uniforms (1, &maskRange);
                sg_apply_uniforms (2, &effectsRange);
                const MetalDepthFogParams noFog = {};
                const sg_range noFogRange = { &noFog, sizeof (noFog) };
                sg_apply_uniforms (3, &noFogRange);
                sg_draw (0, 6, 1);
                sg_end_pass();
                processed = impl_->effect[targetIndex].texture;
            }
            if (blurOn && blurRadius > 0.0f)
            {
                for (int pass = 0; pass < 2; ++pass)
                {
                    const int targetIndex = nextEffectTarget();
                    const MetalFilterParams params = {
                        pass == 0 ? 1.0f / impl_->width : 0.0f,
                        pass == 0 ? 0.0f : 1.0f / impl_->height,
                        blurRadius, 0.0f };
                    filterPass (impl_->blurPipeline, params, processed, {}, targetIndex);
                    processed = impl_->effect[targetIndex].texture;
                }
            }
            if (sharpenOn && sharpenAmount > 0.0f)
            {
                const int targetIndex = nextEffectTarget();
                const MetalFilterParams params = {
                    1.0f / impl_->width, 1.0f / impl_->height,
                    sharpenAmount, 0.0f };
                filterPass (impl_->sharpenPipeline, params, processed, {}, targetIndex);
                processed = impl_->effect[targetIndex].texture;
            }
            // Production filters are exact ordered native passes. Invalid
            // contracts fail closed instead of invoking a CPU approximation.
            for (int fx = 0; layer.effects != nullptr && fx < layer.effectCount; ++fx)
            {
                const auto& effect = layer.effects[fx];
                if (! effect.enabled) continue;
                const int mode = metalProductionFilterMode (effect.type);
                if (mode < 0) continue;
                const auto* definition = videofx::effectDefFor (effect.type);
                if (definition == nullptr || definition->paramCount < 1
                    || definition->paramCount > 3)
                {
                    impl_->error = "Metal production filter has an invalid effect contract";
                    return false;
                }
                const int targetIndex = nextEffectTarget();
                MetalProductionFilterParams params = {
                    static_cast<float> (impl_->width),
                    static_cast<float> (impl_->height), mode, 0.0f, {} };
                for (int value = 0; value < definition->paramCount; ++value)
                    params.values[value] = effect.params[value];

                sg_pass pass = {};
                pass.attachments.colors[0] = impl_->effect[targetIndex].attachment;
                pass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                pass.action.colors[0].store_action = SG_STOREACTION_STORE;
                sg_begin_pass (&pass);
                sg_apply_pipeline (impl_->productionFilterPipeline);
                sg_bindings bindings = {};
                bindings.views[0] = processed;
                bindings.samplers[0] = impl_->sampler;
                sg_apply_bindings (&bindings);
                const sg_range range = { &params, sizeof (params) };
                sg_apply_uniforms (0, &range);
                sg_draw (0, 3, 1);
                sg_end_pass();
                processed = impl_->effect[targetIndex].texture;
            }
            if (layer.lutTexture != 0 && layer.lutSize >= 2)
            {
                const auto lut = impl_->luts.find (layer.lutTexture);
                if (lut == impl_->luts.end()) return false;
                const int targetIndex = nextEffectTarget();
                const MetalFilterParams params = {
                    0.0f, 0.0f, static_cast<float> (layer.lutSize), 0.0f };
                filterPass (impl_->lutPipeline, params, processed,
                            lut->second.view, targetIndex);
                processed = impl_->effect[targetIndex].texture;
            }

            // Stateless UV effects are intentionally separate, ordered passes.
            // This preserves rack slot order after color/blur/sharpen/LUT.
            for (int fx = 0; layer.effects != nullptr && fx < layer.effectCount; ++fx)
            {
                const auto& effect = layer.effects[fx];
                if (! effect.enabled) continue;
                const int mode = metalUvEffectMode (effect.type);
                if (mode < 0) continue;
                const int targetIndex = nextEffectTarget();
                MetalUvEffectParams params = {
                    static_cast<float> (impl_->width),
                    static_cast<float> (impl_->height),
                    static_cast<float> (layer.timeSec), mode, {} };
                const auto* definition = videofx::effectDefFor (effect.type);
                const int count = definition != nullptr
                    ? std::min (definition->paramCount, 4) : 0;
                for (int value = 0; value < count; ++value)
                    params.values[value] = effect.params[value];

                sg_pass pass = {};
                pass.attachments.colors[0] = impl_->effect[targetIndex].attachment;
                pass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                pass.action.colors[0].store_action = SG_STOREACTION_STORE;
                sg_begin_pass (&pass);
                sg_apply_pipeline (impl_->uvEffectPipeline);
                sg_bindings bindings = {};
                bindings.views[0] = processed;
                bindings.samplers[0] = impl_->sampler;
                sg_apply_bindings (&bindings);
                const sg_range range = { &params, sizeof (params) };
                sg_apply_uniforms (0, &range);
                sg_draw (0, 3, 1);
                sg_end_pass();
                processed = impl_->effect[targetIndex].texture;
            }

            int feedbackSlot = -1;
            for (int fx = 0; layer.effects != nullptr && fx < layer.effectCount; ++fx)
                if (layer.effects[fx].enabled
                    && layer.effects[fx].type
                        == static_cast<int> (videofx::EffectType::FeedbackTrail))
                {
                    feedbackSlot = fx;
                    break;
                }
            if (feedbackSlot >= 0)
            {
                const int key = (layer.clipId << 8) | (feedbackSlot & 0xff);
                auto [historyIt, inserted] = impl_->feedback.try_emplace (key);
                auto& history = historyIt->second;
                if (inserted || layer.feedbackHistoryReset)
                {
                    if (! inserted)
                    {
                        impl_->destroyTarget(history.target[0]);
                        impl_->destroyTarget(history.target[1]);
                    }
                    history.current = 0;
                    history.ready = false;
                    if (! impl_->makeTarget (history.target[0], impl_->width, impl_->height,
                                             "arbit-metal-frame-feedback-a")
                        || ! impl_->makeTarget (history.target[1], impl_->width, impl_->height,
                                                "arbit-metal-frame-feedback-b"))
                    {
                        impl_->destroyTarget (history.target[0]);
                        impl_->destroyTarget (history.target[1]);
                        impl_->feedback.erase (historyIt);
                        impl_->error = "Metal feedback history creation failed";
                        return false;
                    }
                    for (int buffer = 0; buffer < 2; ++buffer)
                    {
                        sg_pass clear = {};
                        clear.attachments.colors[0] = history.target[buffer].attachment;
                        clear.action.colors[0].load_action = SG_LOADACTION_CLEAR;
                        clear.action.colors[0].store_action = SG_STOREACTION_STORE;
                        clear.action.colors[0].clear_value = { 0, 0, 0, 0 };
                        sg_begin_pass (&clear);
                        sg_end_pass();
                    }
                }
                if (layer.feedbackHistoryHold && history.ready)
                {
                    processed = history.target[history.current].texture;
                }
                else
                {
                const int readHistory = history.ready ? history.current : 0;
                const int writeHistory = history.ready ? (history.current ^ 1) : 1;
                const auto& effect = layer.effects[feedbackSlot];
                const MetalFeedbackParams params = {
                    effect.params[0], effect.params[1], effect.params[2], 0.0f };
                sg_pass pass = {};
                pass.attachments.colors[0] = history.target[writeHistory].attachment;
                pass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                pass.action.colors[0].store_action = SG_STOREACTION_STORE;
                sg_begin_pass (&pass);
                sg_apply_pipeline (impl_->feedbackPipeline);
                sg_bindings bindings = {};
                bindings.views[0] = processed;
                bindings.views[1] = history.target[readHistory].texture;
                bindings.samplers[0] = impl_->sampler;
                sg_apply_bindings (&bindings);
                const sg_range range = { &params, sizeof (params) };
                sg_apply_uniforms (0, &range);
                sg_draw (0, 3, 1);
                sg_end_pass();
                processed = history.target[writeHistory].texture;
                history.current = writeHistory;
                history.ready = true;
                }
            }

            const int displayWidth = impl_->presentWidth > 0 ? impl_->presentWidth : impl_->width;
            const int displayHeight = impl_->presentHeight > 0 ? impl_->presentHeight : impl_->height;
            const float outAspect = displayHeight > 0
                ? static_cast<float> (displayWidth) / displayHeight : 1.0f;
            const float refAspect = impl_->canvasWidth > 0 && impl_->canvasHeight > 0
                ? static_cast<float> (impl_->canvasWidth) / impl_->canvasHeight : outAspect;
            // Generated sources are rasterised into the compositor target and do
            // not carry decoded-media dimensions in LayerDesc. Match the OpenGL
            // path by treating them as full-canvas instead of falling back to a
            // square source aspect.
            const bool compositorSizedSource = layer.shaderSource || layer.particleSource;
            const int sourceWidth = compositorSizedSource ? impl_->width : layer.texWidth;
            const int sourceHeight = compositorSizedSource ? impl_->height : layer.texHeight;
            const float videoAspect = sourceHeight > 0
                ? static_cast<float> (sourceWidth) / sourceHeight : 1.0f;
            float lbx = 1.0f, lby = 1.0f;
            if (videoAspect > refAspect) lby = refAspect / videoAspect;
            else                         lbx = videoAspect / refAspect;

            MetalGeometryParams geometry = {};
            makeTransform (layer.translateX, layer.translateY, layer.rotationDeg,
                           layer.scale * lbx, layer.scale * lby, geometry.transform);
            float zx = impl_->zoom, zy = impl_->zoom;
            if (impl_->canvasWidth > 0 && impl_->canvasHeight > 0)
            {
                const float canvasAspect = static_cast<float> (impl_->canvasWidth)
                                         / impl_->canvasHeight;
                if (canvasAspect > outAspect) zy *= outAspect / canvasAspect;
                else                          zx *= canvasAspect / outAspect;
            }
            geometry.transform[0] *= zx; geometry.transform[1] *= zy;
            geometry.transform[4] *= zx; geometry.transform[5] *= zy;
            geometry.transform[12] = geometry.transform[12] * zx + impl_->panX;
            geometry.transform[13] = geometry.transform[13] * zy + impl_->panY;
            geometry.crop[0] = layer.cropLeft; geometry.crop[1] = layer.cropRight;
            geometry.crop[2] = layer.cropTop; geometry.crop[3] = layer.cropBottom;
            const MetalMaskParams mask = {
                { layer.maskCx, layer.maskCy, layer.maskW, layer.maskH },
                opacity, layer.maskType, layer.maskFeather, layer.maskInvert ? 1 : 0,
                { layer.matteBlack, layer.matteWhite, layer.matteErodeDilate,
                  layer.matteFeather },
                { layer.matteWidth > 0 ? 1.0f / layer.matteWidth : 0.0f,
                  layer.matteHeight > 0 ? 1.0f / layer.matteHeight : 0.0f },
                layer.matteChoke,
                (layer.matteApply ? 1 : 0) | (layer.matteInvert ? 2 : 0)
                    | ((layer.matteCombineMode + 1) << 2),
                { layer.pathMatteCx, layer.pathMatteCy,
                  layer.pathMatteEllipse ? -layer.pathMatteW : layer.pathMatteW,
                  layer.pathMatteH },
                { layer.pathMatte2Cx, layer.pathMatte2Cy,
                  layer.pathMatteSecondaryEllipse ? -layer.pathMatte2W : layer.pathMatte2W,
                  layer.pathMatte2H },
                layer.pathMatte ? std::clamp(layer.pathMatteOperation, 0, 3) : -1,
                layer.pathMatteInvert ? 1 : 0, { 0, 0 } };
            const MetalEffectParams neutralEffects = neutralEffectParams();
            const MetalEffectParams& layerEffects = colorEffectsPreprocessed
                ? neutralEffects : effects;
            const int depthMode = layer.depthEffect != 0 ? layer.depthEffect : (layer.depthFog ? 1 : 0);
            const MetalDepthFogParams fog = depthMode <= 1 ? MetalDepthFogParams {
                { layer.fogNear, layer.fogFar, layer.fogDensity, static_cast<float>(depthMode) },
                { layer.fogRed, layer.fogGreen, layer.fogBlue, layer.fogAlpha } }
                : MetalDepthFogParams {
                    { layer.depthParam0, layer.depthParam1, layer.depthParam2, static_cast<float>(depthMode) },
                    { layer.depthColorRed, layer.depthColorGreen, layer.depthColorBlue, 1.0f } };

            sg_pass layerPass = {};
            layerPass.attachments.colors[0] = target.attachment;
            layerPass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
            layerPass.action.colors[0].store_action = SG_STOREACTION_STORE;
            layerPass.action.colors[0].clear_value = { 0, 0, 0, 0 };
            sg_begin_pass (&layerPass);
            sg_apply_pipeline (impl_->layerPipeline);
            sg_bindings bindings = {};
            bindings.views[0] = processed;
            bindings.views[1] = impl_->identityLut.view;
            bindings.views[2] = layer.matteApply
                ? sourceViewFor(layer.matteTexture) : processed;
            if (bindings.views[2].id == 0) return false;
            bindings.views[3] = layer.matteCombineMode >= 0
                ? sourceViewFor(layer.matteTextureB) : processed;
            if (bindings.views[3].id == 0) return false;
            if (depthMode != 0)
            {
                if (layer.nativeDepthTextureBackend == "metal"
                    && layer.nativeDepthTextureView != 0)
                {
                    if (layer.nativeDepthTextureView > std::numeric_limits<std::uint32_t>::max())
                        return false;
                    bindings.views[4].id = static_cast<std::uint32_t> (
                        layer.nativeDepthTextureView);
                }
                else
                {
                    const auto depth = impl_->depths.find (layer.depthTexture);
                    if (depth == impl_->depths.end()
                        || depth->second.width != layer.depthWidth
                        || depth->second.height != layer.depthHeight)
                        return false;
                    bindings.views[4] = depth->second.view;
                }
            }
            else bindings.views[4] = processed;
            bindings.samplers[0] = impl_->sampler;
            sg_apply_bindings (&bindings);
            const sg_range geometryRange = { &geometry, sizeof (geometry) };
            const sg_range maskRange = { &mask, sizeof (mask) };
            const sg_range effectsRange = { &layerEffects, sizeof (layerEffects) };
            sg_apply_uniforms (0, &geometryRange);
            sg_apply_uniforms (1, &maskRange);
            sg_apply_uniforms (2, &effectsRange);
            const sg_range fogRange = { &fog, sizeof (fog) };
            sg_apply_uniforms (3, &fogRange);
            sg_draw (0, 6, 1);
            sg_end_pass();
            if (&target == &impl_->layerTarget
                && layer.clipId == impl_->inspectionClipId
                && ! layer.inspectionDrawShapeOutput)
            {
                sg_pass retainPass = {};
                retainPass.attachments.colors[0] = impl_->inspectionTarget.attachment;
                retainPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                retainPass.action.colors[0].store_action = SG_STOREACTION_STORE;
                sg_begin_pass (&retainPass);
                sg_apply_pipeline (impl_->blurPipeline);
                sg_bindings retainBindings = {};
                retainBindings.views[0] = impl_->layerTarget.texture;
                retainBindings.samplers[0] = impl_->sampler;
                sg_apply_bindings (&retainBindings);
                const MetalFilterParams retainParams = { 0, 0, 0, 0 };
                const sg_range retainRange = { &retainParams, sizeof (retainParams) };
                sg_apply_uniforms (0, &retainRange);
                sg_draw (0, 3, 1);
                sg_end_pass();
                impl_->inspectionRetainedHandle = impl_->inspectionTarget.texture.id;
                impl_->inspectionAdmitted = true;
            }
            return true;
        };

        std::vector<bool> overlayComposited (
            static_cast<size_t> (std::max (numOverlays, 0)), false);
        auto compositeOverlay = [&] (int index) -> bool
        {
            const auto& overlay = overlays[index];
            if (overlay.texture == 0 || overlay.opacity <= 0.0f) return true;
            const sg_view overlaySource = sourceViewFor (overlay.texture);
            if (overlaySource.id == 0) return false;

            const float refWidth = impl_->canvasWidth > 0
                ? static_cast<float> (impl_->canvasWidth)
                : static_cast<float> (impl_->width);
            const float refHeight = impl_->canvasHeight > 0
                ? static_cast<float> (impl_->canvasHeight)
                : static_cast<float> (impl_->height);
            MetalGeometryParams geometry = {};
            makeTransform (overlay.posX, overlay.posY, 0.0f,
                           overlay.width / std::max (refWidth, 1.0f),
                           overlay.height / std::max (refHeight, 1.0f),
                           geometry.transform);
            const int displayWidth = impl_->presentWidth > 0
                ? impl_->presentWidth : impl_->width;
            const int displayHeight = impl_->presentHeight > 0
                ? impl_->presentHeight : impl_->height;
            const float outAspect = displayHeight > 0
                ? static_cast<float> (displayWidth) / displayHeight : 1.0f;
            float zx = impl_->zoom, zy = impl_->zoom;
            if (impl_->canvasWidth > 0 && impl_->canvasHeight > 0)
            {
                const float canvasAspect = static_cast<float> (impl_->canvasWidth)
                                         / impl_->canvasHeight;
                if (canvasAspect > outAspect) zy *= outAspect / canvasAspect;
                else                          zx *= canvasAspect / outAspect;
            }
            geometry.transform[0] *= zx; geometry.transform[1] *= zy;
            geometry.transform[4] *= zx; geometry.transform[5] *= zy;
            geometry.transform[12] = geometry.transform[12] * zx + impl_->panX;
            geometry.transform[13] = geometry.transform[13] * zy + impl_->panY;
            const MetalMaskParams mask = {
                { 0.5f, 0.5f, 1.0f, 1.0f }, 1.0f, 0, 0.0f, 0,
                { 0, 0, 0, 0 }, { 0, 0 }, 0.0f, 0,
                { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, -1, 0, { 0, 0 } };
            const MetalEffectParams effects = neutralEffectParams();

            sg_pass overlayPass = {};
            overlayPass.attachments.colors[0] = impl_->layerTarget.attachment;
            overlayPass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
            overlayPass.action.colors[0].store_action = SG_STOREACTION_STORE;
            overlayPass.action.colors[0].clear_value = { 0, 0, 0, 0 };
            sg_begin_pass (&overlayPass);
            sg_apply_pipeline (impl_->layerPipeline);
            sg_bindings overlayBindings = {};
            overlayBindings.views[0] = overlaySource;
            overlayBindings.views[1] = impl_->identityLut.view;
            overlayBindings.views[2] = overlaySource;
            overlayBindings.views[3] = overlaySource;
            // The shared fragment samples depth before checking its mode.
            overlayBindings.views[4] = overlaySource;
            overlayBindings.samplers[0] = impl_->sampler;
            sg_apply_bindings (&overlayBindings);
            const sg_range geometryRange = { &geometry, sizeof (geometry) };
            const sg_range maskRange = { &mask, sizeof (mask) };
            const sg_range effectsRange = { &effects, sizeof (effects) };
            sg_apply_uniforms (0, &geometryRange);
            sg_apply_uniforms (1, &maskRange);
            sg_apply_uniforms (2, &effectsRange);
            const MetalDepthFogParams noFog = {};
            const sg_range noFogRange = { &noFog, sizeof (noFog) };
            sg_apply_uniforms (3, &noFogRange);
            sg_draw (0, 6, 1);
            sg_end_pass();

            sg_pass blendPass = {};
            blendPass.attachments.colors[0] = impl_->accum[read ^ 1].attachment;
            blendPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
            blendPass.action.colors[0].store_action = SG_STOREACTION_STORE;
            sg_begin_pass (&blendPass);
            sg_apply_pipeline (impl_->blendPipeline);
            sg_bindings blendBindings = {};
            blendBindings.views[0] = impl_->layerTarget.texture;
            blendBindings.views[1] = impl_->accum[read].texture;
            blendBindings.samplers[0] = impl_->sampler;
            sg_apply_bindings (&blendBindings);
            const MetalBlendParams blend = { overlay.opacity, 0, { 0, 0 } };
            const sg_range blendRange = { &blend, sizeof (blend) };
            sg_apply_uniforms (0, &blendRange);
            sg_draw (0, 3, 1);
            sg_end_pass();
            read ^= 1;
            overlayComposited[static_cast<size_t> (index)] = true;
            return true;
        };

        for (int i = 0; i < numLayers; ++i)
        {
            const auto& layer = layers[i];
            if (layer.isAdjustment)
            {
                if (layer.opacity <= 0.0f) continue;
                for (int overlay = 0; overlay < numOverlays; ++overlay)
                    if (! overlayComposited[static_cast<size_t> (overlay)]
                        && overlays[overlay].ownerClipId == layer.clipId
                        && ! compositeOverlay (overlay))
                        return 0;
                LayerDesc adjustment = layer;
                adjustment.texWidth = impl_->width;
                adjustment.texHeight = impl_->height;
                adjustment.transitionType = 0;
                if (! drawLayerTo (adjustment, impl_->layerTarget, 1.0f,
                                   impl_->accum[read].texture))
                    return 0;

                sg_pass blendPass = {};
                blendPass.attachments.colors[0] = impl_->accum[read ^ 1].attachment;
                blendPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                blendPass.action.colors[0].store_action = SG_STOREACTION_STORE;
                sg_begin_pass (&blendPass);
                sg_apply_pipeline (impl_->blendPipeline);
                sg_bindings bindings = {};
                bindings.views[0] = impl_->layerTarget.texture;
                bindings.views[1] = impl_->accum[read].texture;
                bindings.samplers[0] = impl_->sampler;
                sg_apply_bindings (&bindings);
                const MetalBlendParams blend = {
                    layer.opacity, 0, { 0.0f, 0.0f } };
                const sg_range range = { &blend, sizeof (blend) };
                sg_apply_uniforms (0, &range);
                sg_draw (0, 3, 1);
                sg_end_pass();
                read ^= 1;
                continue;
            }
            if (layer.transitionType != 0)
            {
                const LayerDesc empty;
                const LayerDesc& from = layer.fromLayer != nullptr ? *layer.fromLayer : empty;
                if (! drawLayerTo (from, impl_->transitionFrom, from.opacity, {})
                    || ! drawLayerTo (layer, impl_->layerTarget, layer.opacity, {}))
                    return 0;
                sg_pass transitionPass = {};
                transitionPass.attachments.colors[0] = impl_->accum[read ^ 1].attachment;
                transitionPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                transitionPass.action.colors[0].store_action = SG_STOREACTION_STORE;
                sg_begin_pass (&transitionPass);
                sg_apply_pipeline (impl_->transitionPipeline);
                sg_bindings bindings = {};
                bindings.views[0] = impl_->transitionFrom.texture;
                bindings.views[1] = impl_->layerTarget.texture;
                bindings.views[2] = impl_->accum[read].texture;
                bindings.samplers[0] = impl_->sampler;
                sg_apply_bindings (&bindings);
                const MetalTransitionParams params = {
                    std::clamp (layer.transitionProgress, 0.0f, 1.0f),
                    metalTransitionType (layer.transitionType), layer.blendMode, 0.0f };
                const sg_range range = { &params, sizeof (params) };
                sg_apply_uniforms (0, &range);
                sg_draw (0, 3, 1);
                sg_end_pass();
                read ^= 1;
                continue;
            }
            if ((layer.texture == 0 && ! layer.particleSource && ! layer.shaderSource)
                || layer.opacity <= 0.0f)
                continue;
            if (! drawLayerTo (layer, impl_->layerTarget, 1.0f, {})) return 0;

            sg_pass blendPass = {};
            blendPass.attachments.colors[0] = impl_->accum[read ^ 1].attachment;
            blendPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
            blendPass.action.colors[0].store_action = SG_STOREACTION_STORE;
            sg_begin_pass (&blendPass);
            sg_apply_pipeline (impl_->blendPipeline);
            sg_bindings blendBindings = {};
            blendBindings.views[0] = impl_->layerTarget.texture;
            blendBindings.views[1] = impl_->accum[read].texture;
            blendBindings.samplers[0] = impl_->sampler;
            sg_apply_bindings (&blendBindings);
            const MetalBlendParams blend = { layer.opacity, layer.blendMode, { 0, 0 } };
            sg_range blendRange = { &blend, sizeof (blend) };
            sg_apply_uniforms (0, &blendRange);
            sg_draw (0, 3, 1);
            sg_end_pass();
            read ^= 1;

            if (layer.drawShape)
            {
                const auto drawShapeStarted = std::chrono::steady_clock::now();
                sg_pass shapePass = {};
                shapePass.attachments.colors[0] = impl_->layerTarget.attachment;
                shapePass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
                shapePass.action.colors[0].store_action = SG_STOREACTION_STORE;
                shapePass.action.colors[0].clear_value = { 0, 0, 0, 0 };
                sg_begin_pass (&shapePass);
                sg_apply_pipeline (impl_->drawShapePipeline);
                const MetalDrawShapeParams shape = {
                    { layer.drawShapeCx, layer.drawShapeCy,
                      layer.drawShapeEllipse ? -std::max(layer.drawShapeW, 0.0f)
                                               : std::max(layer.drawShapeW, 0.0f),
                      std::max(layer.drawShapeH, 0.0f) },
                    { layer.drawShape2Cx, layer.drawShape2Cy,
                      layer.drawShapeSecondaryEllipse ? -std::max(layer.drawShape2W, 0.0f)
                                                       : std::max(layer.drawShape2W, 0.0f),
                      std::max(layer.drawShape2H, 0.0f) },
                    { std::clamp(layer.drawShapeR, 0.0f, 1.0f),
                      std::clamp(layer.drawShapeG, 0.0f, 1.0f),
                      std::clamp(layer.drawShapeB, 0.0f, 1.0f),
                      std::clamp(layer.drawShapeA, 0.0f, 1.0f) },
                    layer.drawShapeHasSecondary
                        ? std::clamp(layer.drawShapeOperation, 1, 3) : 0,
                    { 0.0f, 0.0f, 0.0f } };
                const sg_range shapeRange = { &shape, sizeof (shape) };
                sg_apply_uniforms (0, &shapeRange);
                sg_draw (0, 3, 1);
                sg_end_pass();
                if (impl_->visualTelemetry != nullptr && layer.drawShapeNodeId != 0)
                    impl_->visualTelemetry->recordNodeEvaluation(layer.clipId,
                        layer.visualPlanStructuralRevision, layer.drawShapeNodeId,
                        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - drawShapeStarted).count()),
                        layer.visualPlanTelemetryHold);
                if (layer.inspectionDrawShapeOutput
                    && layer.clipId == impl_->inspectionClipId)
                {
                    sg_pass retainPass = {};
                    retainPass.attachments.colors[0] = impl_->inspectionTarget.attachment;
                    retainPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                    retainPass.action.colors[0].store_action = SG_STOREACTION_STORE;
                    sg_begin_pass (&retainPass);
                    sg_apply_pipeline (impl_->blurPipeline);
                    sg_bindings retainBindings = {};
                    retainBindings.views[0] = impl_->layerTarget.texture;
                    retainBindings.samplers[0] = impl_->sampler;
                    sg_apply_bindings (&retainBindings);
                    const MetalFilterParams retainParams = { 0, 0, 0, 0 };
                    const sg_range retainRange = { &retainParams, sizeof (retainParams) };
                    sg_apply_uniforms (0, &retainRange);
                    sg_draw (0, 3, 1);
                    sg_end_pass();
                    impl_->inspectionRetainedHandle = impl_->inspectionTarget.texture.id;
                    impl_->inspectionAdmitted = true;
                }

                sg_pass shapeBlendPass = {};
                shapeBlendPass.attachments.colors[0] = impl_->accum[read ^ 1].attachment;
                shapeBlendPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                shapeBlendPass.action.colors[0].store_action = SG_STOREACTION_STORE;
                sg_begin_pass (&shapeBlendPass);
                sg_apply_pipeline (impl_->blendPipeline);
                sg_bindings shapeBindings = {};
                shapeBindings.views[0] = impl_->layerTarget.texture;
                shapeBindings.views[1] = impl_->accum[read].texture;
                shapeBindings.samplers[0] = impl_->sampler;
                sg_apply_bindings (&shapeBindings);
                const MetalBlendParams normal = { 1.0f, 0, { 0, 0 } };
                const sg_range normalRange = { &normal, sizeof (normal) };
                sg_apply_uniforms (0, &normalRange);
                sg_draw (0, 3, 1);
                sg_end_pass();
                read ^= 1;
            }
        }

        // Remaining project-level and normal-clip overlays composite on top.
        // Adjustment-owned overlays were inserted immediately before their
        // owner so the adjustment transforms/effects them with the video.
        for (int i = 0; i < numOverlays; ++i)
        {
            if (overlayComposited[static_cast<size_t> (i)]) continue;
            const auto& overlay = overlays[i];
            if (overlay.texture == 0 || overlay.opacity <= 0.0f) continue;
            const sg_view overlaySource = sourceViewFor (overlay.texture);
            if (overlaySource.id == 0) return 0;

            const float refWidth = impl_->canvasWidth > 0
                ? static_cast<float> (impl_->canvasWidth)
                : static_cast<float> (impl_->width);
            const float refHeight = impl_->canvasHeight > 0
                ? static_cast<float> (impl_->canvasHeight)
                : static_cast<float> (impl_->height);
            MetalGeometryParams geometry = {};
            makeTransform (overlay.posX, overlay.posY, 0.0f,
                           overlay.width / std::max (refWidth, 1.0f),
                           overlay.height / std::max (refHeight, 1.0f),
                           geometry.transform);
            const int displayWidth = impl_->presentWidth > 0
                ? impl_->presentWidth : impl_->width;
            const int displayHeight = impl_->presentHeight > 0
                ? impl_->presentHeight : impl_->height;
            const float outAspect = displayHeight > 0
                ? static_cast<float> (displayWidth) / displayHeight : 1.0f;
            float zx = impl_->zoom, zy = impl_->zoom;
            if (impl_->canvasWidth > 0 && impl_->canvasHeight > 0)
            {
                const float canvasAspect = static_cast<float> (impl_->canvasWidth)
                                         / impl_->canvasHeight;
                if (canvasAspect > outAspect) zy *= outAspect / canvasAspect;
                else                          zx *= canvasAspect / outAspect;
            }
            geometry.transform[0] *= zx; geometry.transform[1] *= zy;
            geometry.transform[4] *= zx; geometry.transform[5] *= zy;
            geometry.transform[12] = geometry.transform[12] * zx + impl_->panX;
            geometry.transform[13] = geometry.transform[13] * zy + impl_->panY;
            const MetalMaskParams mask = {
                { 0.5f, 0.5f, 1.0f, 1.0f }, 1.0f, 0, 0.0f, 0,
                { 0, 0, 0, 0 }, { 0, 0 }, 0.0f, 0,
                { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, -1, 0, { 0, 0 } };
            const MetalEffectParams effects = neutralEffectParams();

            sg_pass overlayPass = {};
            overlayPass.attachments.colors[0] = impl_->layerTarget.attachment;
            overlayPass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
            overlayPass.action.colors[0].store_action = SG_STOREACTION_STORE;
            overlayPass.action.colors[0].clear_value = { 0, 0, 0, 0 };
            sg_begin_pass (&overlayPass);
            sg_apply_pipeline (impl_->layerPipeline);
            sg_bindings overlayBindings = {};
            overlayBindings.views[0] = overlaySource;
            overlayBindings.views[1] = impl_->identityLut.view;
            overlayBindings.views[2] = overlaySource;
            overlayBindings.views[3] = overlaySource;
            // The shared fragment samples depth before checking its mode.
            overlayBindings.views[4] = overlaySource;
            overlayBindings.samplers[0] = impl_->sampler;
            sg_apply_bindings (&overlayBindings);
            const sg_range geometryRange = { &geometry, sizeof (geometry) };
            const sg_range maskRange = { &mask, sizeof (mask) };
            const sg_range effectsRange = { &effects, sizeof (effects) };
            sg_apply_uniforms (0, &geometryRange);
            sg_apply_uniforms (1, &maskRange);
            sg_apply_uniforms (2, &effectsRange);
            const MetalDepthFogParams noFog = {};
            const sg_range noFogRange = { &noFog, sizeof (noFog) };
            sg_apply_uniforms (3, &noFogRange);
            sg_draw (0, 6, 1);
            sg_end_pass();

            sg_pass blendPass = {};
            blendPass.attachments.colors[0] = impl_->accum[read ^ 1].attachment;
            blendPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
            blendPass.action.colors[0].store_action = SG_STOREACTION_STORE;
            sg_begin_pass (&blendPass);
            sg_apply_pipeline (impl_->blendPipeline);
            sg_bindings blendBindings = {};
            blendBindings.views[0] = impl_->layerTarget.texture;
            blendBindings.views[1] = impl_->accum[read].texture;
            blendBindings.samplers[0] = impl_->sampler;
            sg_apply_bindings (&blendBindings);
            const MetalBlendParams blend = { overlay.opacity, 0, { 0, 0 } };
            const sg_range blendRange = { &blend, sizeof (blend) };
            sg_apply_uniforms (0, &blendRange);
            sg_draw (0, 3, 1);
            sg_end_pass();
            read ^= 1;
        }

        sg_view finalView = impl_->accum[read].texture;
        const bool bloomOn = impl_->bloomIntensity > 0.0f
            && impl_->bloomRadius > 0.0f;
        const bool tonemapOn = impl_->tonemap != 0 || impl_->exposure != 1.0f;
        if (bloomOn)
        {
            const MetalPostParams threshold = {
                impl_->bloomThreshold, 0.0f, 1.0f, 0 };
            sg_pass pass = {};
            pass.attachments.colors[0] = impl_->effect[0].attachment;
            pass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
            pass.action.colors[0].store_action = SG_STOREACTION_STORE;
            sg_begin_pass (&pass);
            sg_apply_pipeline (impl_->bloomThresholdPipeline);
            sg_bindings bindings = {};
            bindings.views[0] = finalView;
            bindings.samplers[0] = impl_->sampler;
            sg_apply_bindings (&bindings);
            const sg_range range = { &threshold, sizeof (threshold) };
            sg_apply_uniforms (0, &range);
            sg_draw (0, 3, 1);
            sg_end_pass();

            const float radius = std::min (impl_->bloomRadius, 20.0f);
            const int passes = std::max (1, std::min (5,
                static_cast<int> (std::ceil (impl_->bloomRadius / 10.0f))));
            for (int iteration = 0; iteration < passes; ++iteration)
            {
                for (int direction = 0; direction < 2; ++direction)
                {
                    const int source = direction == 0 ? 0 : 1;
                    const int target = direction == 0 ? 1 : 0;
                    const MetalFilterParams params = {
                        direction == 0 ? 1.0f / impl_->width : 0.0f,
                        direction == 0 ? 0.0f : 1.0f / impl_->height,
                        radius, 0.0f };
                    sg_pass blurPass = {};
                    blurPass.attachments.colors[0] = impl_->effect[target].attachment;
                    blurPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
                    blurPass.action.colors[0].store_action = SG_STOREACTION_STORE;
                    sg_begin_pass (&blurPass);
                    sg_apply_pipeline (impl_->blurPipeline);
                    sg_bindings blurBindings = {};
                    blurBindings.views[0] = impl_->effect[source].texture;
                    blurBindings.samplers[0] = impl_->sampler;
                    sg_apply_bindings (&blurBindings);
                    const sg_range blurRange = { &params, sizeof (params) };
                    sg_apply_uniforms (0, &blurRange);
                    sg_draw (0, 3, 1);
                    sg_end_pass();
                }
            }
        }
        if (bloomOn || tonemapOn)
        {
            const MetalPostParams params = {
                impl_->bloomThreshold, bloomOn ? impl_->bloomIntensity : 0.0f,
                impl_->exposure, impl_->tonemap };
            sg_pass pass = {};
            pass.attachments.colors[0] = impl_->layerTarget.attachment;
            pass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
            pass.action.colors[0].store_action = SG_STOREACTION_STORE;
            sg_begin_pass (&pass);
            sg_apply_pipeline (impl_->postCombinePipeline);
            sg_bindings bindings = {};
            bindings.views[0] = finalView;
            bindings.views[1] = bloomOn ? impl_->effect[0].texture : finalView;
            bindings.samplers[0] = impl_->sampler;
            sg_apply_bindings (&bindings);
            const sg_range range = { &params, sizeof (params) };
            sg_apply_uniforms (0, &range);
            sg_draw (0, 3, 1);
            sg_end_pass();
            finalView = impl_->layerTarget.texture;
        }

        Impl::DirectOutput* directOutput = direct
            ? impl_->directOutput (impl_->requestedDirectSurface,
                                   impl_->requestedDirectWidth,
                                   impl_->requestedDirectHeight)
            : nullptr;
        if (direct && directOutput == nullptr)
        {
            impl_->error = "Metal direct IOSurface target creation failed";
            return 0;
        }
        sg_view previewView = {};
        if (impl_->inspectionClipId >= 0 && impl_->inspectionAdmitted)
            previewView = impl_->inspectionTarget.texture;
        else if (impl_->inspectionRequestedHandle == 0xffffffffu)
            previewView = finalView;
        else if (impl_->inspectionRequestedHandle != 0)
            previewView = sourceViewFor (impl_->inspectionRequestedHandle);
        if (previewView.id != 0)
        {
            impl_->inspectionAdmitted = true;
            impl_->inspectionRetainedHandle = previewView.id;
        }

        sg_pass outputPass = {};
        outputPass.attachments.colors[0] = direct
            ? directOutput->attachment : impl_->outputAttachment;
        outputPass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
        outputPass.action.colors[0].store_action = SG_STOREACTION_STORE;
        sg_begin_pass (&outputPass);
        const bool showPreview = direct && impl_->inspectionAdmitted
            && impl_->inspectionPresentation.layout != videopreview::Layout::hidden;
        const bool frameCanvas = direct && ! showPreview
            && impl_->canvasWidth > 0 && impl_->canvasHeight > 0;
        sg_apply_pipeline (showPreview ? impl_->previewPipeline
                                      : (frameCanvas ? impl_->canvasPipeline : impl_->blitPipeline));
        sg_bindings outputBindings = {};
        outputBindings.views[0] = finalView;
        if (showPreview) outputBindings.views[1] = previewView;
        outputBindings.samplers[0] = impl_->sampler;
        sg_apply_bindings (&outputBindings);
        if (showPreview)
        {
            const auto& state = impl_->inspectionPresentation;
            const MetalPreviewParams params = {
                state.zoom, 0.0f, { state.panX, state.panY }, state.split,
                static_cast<int32_t> (state.layout),
                static_cast<int32_t> (state.background), 0.0f, { 0, 0 } };
            const sg_range range = { &params, sizeof (params) };
            sg_apply_uniforms (0, &range);
        }
        else if (frameCanvas)
        {
            const float outAspect = static_cast<float> (impl_->requestedDirectWidth)
                                  / std::max (impl_->requestedDirectHeight, 1);
            float zx = impl_->zoom, zy = impl_->zoom;
            const float canvasAspect = static_cast<float> (impl_->canvasWidth)
                                     / impl_->canvasHeight;
            if (canvasAspect > outAspect) zy *= outAspect / canvasAspect;
            else                          zx *= canvasAspect / outAspect;
            const MetalCanvasParams params = {
                { (impl_->panX - zx + 1.0f) * 0.5f,
                  (impl_->panY - zy + 1.0f) * 0.5f,
                  (impl_->panX + zx + 1.0f) * 0.5f,
                  (impl_->panY + zy + 1.0f) * 0.5f },
                { 1.0f / impl_->requestedDirectWidth,
                  1.0f / impl_->requestedDirectHeight }, { 0.0f, 0.0f } };
            const sg_range range = { &params, sizeof (params) };
            sg_apply_uniforms (0, &range);
        }
        sg_draw (0, 3, 1);
        sg_end_pass();
        sg_commit();

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
        id<MTLCommandBuffer> fence = [queue commandBuffer];
        [fence commit];
        [fence waitUntilCompleted];
        if (direct)
        {
            // The persistent consumer imports this IOSurface once and samples
            // it from another Metal device/command queue. A completed producer
            // command buffer orders execution, while IOSurfaceLock/Unlock is
            // the public cross-process cache-coherence barrier for the shared
            // allocation. No pixels are read or copied here.
            IOSurfaceRef sharedSurface = static_cast<IOSurfaceRef> (
                impl_->requestedDirectSurface);
            if (sharedSurface == nullptr
                || IOSurfaceLock (sharedSurface, kIOSurfaceLockReadOnly, nullptr) != 0)
            {
                impl_->error = "Metal IOSurface coherence lock failed";
                return 0;
            }
            IOSurfaceUnlock (sharedSurface, kIOSurfaceLockReadOnly, nullptr);
            impl_->error.clear();
            return 1;
        }
        gl->BindFramebuffer (GL_READ_FRAMEBUFFER, impl_->rectangleFbo);
        gl->BindFramebuffer (GL_DRAW_FRAMEBUFFER, impl_->outputFbo);
        gl->BlitFramebuffer (0, 0, impl_->width, impl_->height,
                             0, 0, impl_->width, impl_->height,
                             GL_COLOR_BUFFER_BIT, GL_NEAREST);
        gl->BindFramebuffer (GL_FRAMEBUFFER, 0);
        impl_->error.clear();
        return impl_->outputTexture;
    }
}

bool MetalFrameRenderer::renderCompositeToIOSurface (
    const arbitgl::GlFuncs* gl, void* ioSurface, int width, int height,
    const LayerDesc* layers, int numLayers,
    const ImageLayerDesc* overlays, int numOverlays)
{
    if (ioSurface == nullptr || width <= 0 || height <= 0) return false;
    impl_->requestedDirectSurface = ioSurface;
    impl_->requestedDirectWidth = width;
    impl_->requestedDirectHeight = height;
    const bool rendered = renderComposite (
        gl, layers, numLayers, overlays, numOverlays) != 0;
    impl_->requestedDirectSurface = nullptr;
    impl_->requestedDirectWidth = impl_->requestedDirectHeight = 0;
    return rendered;
}

void MetalFrameRenderer::clearDirectOutputs()
{
    std::lock_guard<std::mutex> lock (sokolMutex());
    impl_->clearDirectOutputs();
}

bool MetalFrameRenderer::ready() const
{
    return impl_ != nullptr && impl_->programsReady
        && impl_->accum[0].image.id != 0
        && (impl_->directOnly || impl_->outputTexture != 0);
}

const std::string& MetalFrameRenderer::lastError() const
{ return impl_->error; }

const std::string& MetalFrameRenderer::particleBackend() const
{ return impl_->particleBackend; }

void MetalFrameRenderer::setVisualTelemetryOwner (videowire::VisualPlanTelemetry* owner)
{ impl_->visualTelemetry = owner; }

} // namespace videorender
#endif
