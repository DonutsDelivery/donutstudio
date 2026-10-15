#include "diffraction_material_metal.h"
#include "gpu_backend/sokol_metal_context.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <mutex>

#define SOKOL_METAL
#include "sokol_gfx.h"

namespace diffractionmaterial
{
namespace
{
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

constant int profileBinaryRectangular = 1;
constant int profileSinusoidal = 2;
constant int profileBlazedSawtooth = 3;
constant int coatingUncoated = 1;
constant int coatingIncoherentDielectric = 2;
constant float physicalPi = 3.14159265358979323846f;
constant float degreesToRadians = 0.01745329251994329577f;

struct Params {
    float4 geometry;
    float4 secondaryGeometry;
    float4 microstructure;
    float4 control;
    float4 incident;
    float4 coating;
    float4 roughness;
    float4 grooveField;
    float4 grooveVariation;
    float4 spectral[8];
    float4 spectralZ[8];
};

struct FragmentOut {
    float4 linearSrgb [[color(0)]];
    float4 energy [[color(1)]];
    float4 orders [[color(2)]];
    float4 directionMoment [[color(3)]];
};

float interfaceReflectance(float2 first, float2 second)
{
    const float2 difference = first - second;
    const float2 sum = first + second;
    return clamp(dot(difference, difference) / dot(sum, sum), 0.0f, 1.0f);
}

float spectralReflectance(constant Params& u, float wavelength,
                          int coatingModel, float substrateN, float substrateK)
{
    if (coatingModel == coatingUncoated)
        return interfaceReflectance(float2(1.0f, 0.0f),
                                    float2(substrateN, substrateK));
    if (coatingModel != coatingIncoherentDielectric)
        return -1.0f;

    const float2 layer = float2(u.coating.y, u.coating.z);
    const float r01 = interfaceReflectance(float2(1.0f, 0.0f), layer);
    const float r12 = interfaceReflectance(layer, float2(substrateN, substrateK));
    const float roundTrip = exp(-8.0f * physicalPi * u.coating.z
                                * u.coating.x / wavelength);
    const float denominator = 1.0f - r01 * r12 * roundTrip;
    return clamp(r01 + (1.0f - r01) * (1.0f - r01)
                    * r12 * roundTrip / denominator, 0.0f, 1.0f);
}

float coherentRoughnessFraction(constant Params& u, float wavelength,
                                float outgoingCosine)
{
    const float argument = 2.0f * physicalPi * u.roughness.x
                         * (u.incident.z + outgoingCosine) / wavelength;
    return exp(-(argument * argument));
}

float2 complexMultiply(float2 a, float2 b)
{
    return float2(a.x * b.x - a.y * b.y,
                  a.x * b.y + a.y * b.x);
}

float2 complexExponentialIntegral(float frequency, float begin, float end)
{
    if (abs(frequency) < 1.0e-6f)
        return float2(end - begin, 0.0f);
    const float beginPhase = frequency * begin;
    const float endPhase = frequency * end;
    return float2(
        (sin(endPhase) - sin(beginPhase)) / frequency,
        -(cos(endPhase) - cos(beginPhase)) / frequency);
}

// Integer-order Bessel J_n power series. Native admission bounds its argument
// to four, where 24 terms keep the float result inside the readback tolerance.
float besselJ(int order, float argument)
{
    const int magnitude = order < 0 ? -order : order;
    const float halfArgument = 0.5f * argument;
    float term = 1.0f;
    for (int factor = 1; factor <= 8; ++factor)
    {
        if (factor <= magnitude)
            term *= halfArgument / float(factor);
    }
    float sum = term;
    for (int seriesIndex = 1; seriesIndex <= 24; ++seriesIndex)
    {
        term *= -(halfArgument * halfArgument)
              / (float(seriesIndex) * float(magnitude + seriesIndex));
        sum += term;
    }
    return sum;
}

float profileEfficiency(int profile, int signedOrder, float duty,
                        float phase, float2 terracePhase)
{
    constexpr float pi = 3.14159265358979323846f;
    if (profile == profileBinaryRectangular)
    {
        if (signedOrder == 0)
        {
            const float2 amplitude = float2(1.0f - duty, 0.0f)
                                   + duty * terracePhase;
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
    if (profile == profileSinusoidal)
    {
        const float amplitude = besselJ(signedOrder, 0.5f * phase);
        return amplitude * amplitude;
    }
    if (profile == profileBlazedSawtooth)
    {
        const float orderFrequency = -2.0f * pi * float(signedOrder);
        const float2 ramp = complexExponentialIntegral(
            phase / duty + orderFrequency, 0.0f, duty);
        const float2 land = complexExponentialIntegral(
            orderFrequency, duty, 1.0f);
        const float2 amplitude = ramp + land;
        return dot(amplitude, amplitude);
    }
    return -1.0f;
}

fragment FragmentOut _main(constant Params& u [[buffer(0)]])
{
    constexpr float pi = 3.14159265358979323846f;
    const float duty = u.microstructure.x;
    const float n = u.microstructure.y;
    const float k = u.microstructure.z;
    const int firstOrder = int(u.microstructure.w + 0.5f);
    const int lastOrder = int(u.control.x + 0.5f);
    const int profile = int(u.control.y + 0.5f);
    const int coatingModel = int(u.control.w + 0.5f);
    const bool crossedTwoDimensional = int(u.secondaryGeometry.w + 0.5f) == 2;
    float4 localGeometry = u.geometry;
    float4 localSecondaryGeometry = u.secondaryGeometry;
    const int fieldMode = int(u.grooveField.x + 0.5f);
    if (fieldMode != 1)
    {
        const float2 offset = float2(0.5f) - u.grooveField.yz;
        const float coordinate = fieldMode == 2
            ? dot(offset, u.grooveVariation.xy) : length(offset);
        localGeometry.z += coordinate * u.grooveVariation.z;
        localSecondaryGeometry.z += coordinate * u.grooveVariation.w;
        const float angle = coordinate * u.grooveField.w * degreesToRadians;
        const float2x2 rotation = float2x2(
            float2(cos(angle), sin(angle)), float2(-sin(angle), cos(angle)));
        localGeometry.xy = rotation * localGeometry.xy;
        localSecondaryGeometry.xy = rotation * localSecondaryGeometry.xy;
    }

    float3 xyz = float3(0.0f);
    float3 directionMoment = float3(0.0f);
    float referenceWhiteY = 0.0f;
    float incidentTotal = 0.0f;
    float reflectedTotal = 0.0f;
    float zeroTotal = 0.0f;
    float higherTotal = 0.0f;
    float unresolvedTotal = 0.0f;
    float absorbedTotal = 0.0f;
    float propagatingCount = 0.0f;
    float rejectedCount = 0.0f;
    // A Gaussian angular deviation with sigma=2*rmsSlope has a first
    // directional moment of exp(-sigma^2/2). Energy remains in w unchanged.
    const float broadeningMoment = exp(-2.0f * u.roughness.y * u.roughness.y);

    for (int wavelengthIndex = 0; wavelengthIndex < 8; ++wavelengthIndex)
    {
        const float4 spectral = u.spectral[wavelengthIndex];
        const float4 spectralZ = u.spectralZ[wavelengthIndex];
        const float wavelength = spectral.x;
        const float quadrature = spectralZ.y;
        const float reflectance = spectralReflectance(
            u, wavelength, coatingModel, n, k);
        const float zeroCoherent = coherentRoughnessFraction(
            u, wavelength, u.incident.z);
        const float zeroPhase = 4.0f * pi * u.geometry.w * u.incident.z / wavelength;
        const float2 zeroTerracePhase = float2(cos(zeroPhase), sin(zeroPhase));

        const float zeroProfileEfficiency = profileEfficiency(
            profile, 0, duty, zeroPhase, zeroTerracePhase);
        const float zeroEfficiency = zeroCoherent * zeroProfileEfficiency
            * (crossedTwoDimensional ? zeroProfileEfficiency : 1.0f);
        float higherEfficiency = 0.0f;
        float3 wavelengthDirectionMoment = zeroEfficiency
            * float3(-u.incident.xy, u.incident.z);

        for (int primaryOrder = -8; primaryOrder <= 8; ++primaryOrder)
        {
            const int primaryMagnitude = primaryOrder < 0
                ? -primaryOrder : primaryOrder;
            const bool primaryAdmitted = primaryOrder == 0
                || (primaryMagnitude >= firstOrder && primaryMagnitude <= lastOrder);
            if (!primaryAdmitted)
                continue;
            for (int secondaryOrder = -8; secondaryOrder <= 8; ++secondaryOrder)
            {
                const int secondaryMagnitude = secondaryOrder < 0
                    ? -secondaryOrder : secondaryOrder;
                const bool secondaryAdmitted = secondaryOrder == 0
                    || (secondaryMagnitude >= firstOrder && secondaryMagnitude <= lastOrder);
                if (!secondaryAdmitted || (!crossedTwoDimensional && secondaryOrder != 0)
                    || (primaryOrder == 0 && secondaryOrder == 0))
                    continue;

                float2 tangent = -u.incident.xy
                    + float(primaryOrder) * wavelength
                        / localGeometry.z * localGeometry.xy;
                if (crossedTwoDimensional)
                    tangent += float(secondaryOrder) * wavelength
                        / localSecondaryGeometry.z * localSecondaryGeometry.xy;
                const float tangentSquared = dot(tangent, tangent);
                if (tangentSquared >= 1.0f - 1.0e-12f)
                {
                    rejectedCount += 1.0f;
                    continue;
                }

                const float3 outgoing = float3(
                    tangent, sqrt(max(0.0f, 1.0f - tangentSquared)));
                const float phase = 2.0f * pi * u.geometry.w
                    * (u.incident.z + outgoing.z) / wavelength;
                const float2 terracePhase = float2(cos(phase), sin(phase));
                float efficiency = coherentRoughnessFraction(
                    u, wavelength, outgoing.z) * profileEfficiency(
                        profile, primaryOrder, duty, phase, terracePhase);
                if (crossedTwoDimensional)
                    efficiency *= profileEfficiency(
                        profile, secondaryOrder, duty, phase, terracePhase);
                efficiency *= outgoing.z / u.incident.z;
                higherEfficiency += efficiency;
                wavelengthDirectionMoment += efficiency * outgoing;
                propagatingCount += 1.0f;
            }
        }

        const float resolvedEfficiency = zeroEfficiency + higherEfficiency;
        if (!(reflectance >= 0.0f) || reflectance > 1.0f
            || !(zeroCoherent >= 0.0f) || zeroCoherent > 1.0f
            || !(resolvedEfficiency >= 0.0f) || resolvedEfficiency > 1.0001f)
        {
            FragmentOut rejected;
            rejected.linearSrgb = float4(0.0f, 0.0f, 0.0f, -1.0f);
            rejected.energy = float4(-1.0f);
            rejected.orders = float4(-1.0f);
            rejected.directionMoment = float4(-1.0f);
            return rejected;
        }

        const float incidentEnergy = spectral.y * quadrature;
        const float reflectedEnergy = incidentEnergy * reflectance;
        const float zeroEnergy = reflectedEnergy * zeroEfficiency;
        const float higherEnergy = reflectedEnergy * higherEfficiency;
        const float unresolvedEnergy = reflectedEnergy
                                     * max(0.0f, 1.0f - resolvedEfficiency);
        const float absorbedEnergy = incidentEnergy - reflectedEnergy;
        const float resolvedEnergy = zeroEnergy + higherEnergy;

        incidentTotal += incidentEnergy;
        reflectedTotal += reflectedEnergy;
        zeroTotal += zeroEnergy;
        higherTotal += higherEnergy;
        unresolvedTotal += unresolvedEnergy;
        absorbedTotal += absorbedEnergy;
        directionMoment += broadeningMoment
            * reflectedEnergy * wavelengthDirectionMoment;
        xyz += resolvedEnergy * float3(spectral.z, spectral.w, spectralZ.x);
        referenceWhiteY += quadrature * spectral.w;
    }

    xyz /= referenceWhiteY;
    // One output conversion follows the complete wavelength/order accumulation.
    const float3 linearSrgb = max(float3(0.0f), float3(
        3.2406f * xyz.x - 1.5372f * xyz.y - 0.4986f * xyz.z,
       -0.9689f * xyz.x + 1.8758f * xyz.y + 0.0415f * xyz.z,
        0.0557f * xyz.x - 0.2040f * xyz.y + 1.0570f * xyz.z));

    FragmentOut out;
    out.linearSrgb = float4(linearSrgb, 1.0f);
    out.energy = float4(incidentTotal, reflectedTotal, zeroTotal, higherTotal);
    out.orders = float4(unresolvedTotal, absorbedTotal,
                        propagatingCount, rejectedCount);
    out.directionMoment = float4(directionMoment, zeroTotal + higherTotal);
    return out;
}
)metal";

bool resourceValid(sg_resource_state state) noexcept
{
    return state == SG_RESOURCESTATE_VALID;
}

std::string metalDeviceName(id<MTLDevice> device)
{
    return device != nil && device.name != nil
        ? std::string([device.name UTF8String]) : std::string();
}

} // namespace

struct MetalPhysicalDiffractionExecutor::Impl final
{
    std::array<sg_image, 4> images {};
    std::array<sg_view, 4> attachmentViews {};
    std::array<sg_view, 4> textureViews {};
    sg_shader shader {};
    sg_pipeline pipeline {};
    std::uint64_t generation = 0;

    void destroyUnlocked() noexcept
    {
        if (pipeline.id != 0) sg_destroy_pipeline(pipeline);
        if (shader.id != 0) sg_destroy_shader(shader);
        for (auto& view : textureViews)
            if (view.id != 0) sg_destroy_view(view);
        for (auto& view : attachmentViews)
            if (view.id != 0) sg_destroy_view(view);
        for (auto& image : images)
            if (image.id != 0) sg_destroy_image(image);
        images = {};
        attachmentViews = {};
        textureViews = {};
        shader = {};
        pipeline = {};
        generation = 0;
    }
};

MetalPhysicalDiffractionCapabilities queryMetalPhysicalDiffractionCapabilities()
{
    MetalPhysicalDiffractionCapabilities result;
    std::lock_guard<std::mutex> lock(arbitgpu::sokolmetal::mutex());
    @autoreleasepool
    {
        if (!arbitgpu::sokolmetal::ensure())
        {
            result.error = arbitgpu::sokolmetal::error();
            return result;
        }
        id<MTLDevice> device = (__bridge id<MTLDevice>) arbitgpu::sokolmetal::device();
        result.backend = "metal";
        result.device = metalDeviceName(device);
        if (sg_query_backend() != SG_BACKEND_METAL_MACOS)
        {
            result.error = "sokol_gfx did not select the macOS Metal backend";
            return result;
        }
        const auto limits = sg_query_limits();
        const auto format = sg_query_pixelformat(SG_PIXELFORMAT_RGBA32F);
        if (limits.max_color_attachments < 4 || !format.render)
        {
            result.error = "Metal does not support four RGBA32F diffraction targets";
            return result;
        }
        result.available = true;
        result.spectralSamples = kMaximumSpectralSamples;
        result.maximumOrder = kMaximumDiffractionOrder;
        result.canonicalWavelengthLayout = true;
        result.binaryRectangularProfile = true;
        result.sinusoidalProfile = true;
        result.blazedSawtoothProfile = true;
        result.crossedTwoDimensionalLattice = true;
        result.incoherentCoating = true;
        result.rmsHeightAttenuation = true;
        result.roughnessBroadening = true;
    }
    return result;
}

MetalPhysicalDiffractionExecutor::MetalPhysicalDiffractionExecutor()
    : impl_(std::make_unique<Impl>())
{
}

MetalPhysicalDiffractionExecutor::~MetalPhysicalDiffractionExecutor()
{
    shutdown();
}

bool MetalPhysicalDiffractionExecutor::initialize(std::string& error)
{
    shutdown();
    error.clear();
    std::lock_guard<std::mutex> lock(arbitgpu::sokolmetal::mutex());
    @autoreleasepool
    {
        if (!arbitgpu::sokolmetal::ensure())
        {
            error = arbitgpu::sokolmetal::error();
            return false;
        }
        if (sg_query_backend() != SG_BACKEND_METAL_MACOS)
        {
            error = "sokol_gfx did not select the macOS Metal backend";
            return false;
        }
        const auto limits = sg_query_limits();
        const auto format = sg_query_pixelformat(SG_PIXELFORMAT_RGBA32F);
        if (limits.max_color_attachments < 4 || !format.render)
        {
            error = "Metal does not support four RGBA32F diffraction targets";
            return false;
        }

        const auto compileShaderStage = [&error](const char* source,
                                                  NSString* stageName) -> bool
        {
            id<MTLDevice> device = (__bridge id<MTLDevice>) arbitgpu::sokolmetal::device();
            NSError* shaderError = nil;
            NSString* shaderSource = [NSString stringWithUTF8String:source];
            id<MTLLibrary> library = [device newLibraryWithSource:shaderSource
                                                          options:nil
                                                            error:&shaderError];
            id<MTLFunction> function = [library newFunctionWithName:@"_main"];
            const bool compiled = library != nil && function != nil;
#if !__has_feature(objc_arc)
            [function release];
            [library release];
#endif
            if (compiled)
                return true;

            error = "physical diffraction Metal "
                + std::string([stageName UTF8String]) + " shader compilation failed";
            if (shaderError != nil && shaderError.localizedDescription != nil)
                error += ": " + std::string([shaderError.localizedDescription UTF8String]);
            return false;
        };
        if (!compileShaderStage(kVertexShader, @"vertex")
            || !compileShaderStage(kFragmentShader, @"fragment"))
        {
            return false;
        }

        for (std::size_t index = 0; index < impl_->images.size(); ++index)
        {
            sg_image_desc imageDesc = {};
            imageDesc.usage.color_attachment = true;
            imageDesc.width = 1;
            imageDesc.height = 1;
            imageDesc.pixel_format = SG_PIXELFORMAT_RGBA32F;
            imageDesc.sample_count = 1;
            imageDesc.label = "arbit-metal-physical-diffraction-image";
            impl_->images[index] = sg_make_image(&imageDesc);

            sg_view_desc attachmentDesc = {};
            attachmentDesc.color_attachment.image = impl_->images[index];
            impl_->attachmentViews[index] = sg_make_view(&attachmentDesc);
            sg_view_desc textureDesc = {};
            textureDesc.texture.image = impl_->images[index];
            impl_->textureViews[index] = sg_make_view(&textureDesc);
        }

        sg_shader_desc shaderDesc = {};
        shaderDesc.vertex_func.source = kVertexShader;
        shaderDesc.fragment_func.source = kFragmentShader;
        shaderDesc.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        shaderDesc.uniform_blocks[0].size = sizeof(PhysicalDiffractionGpuParameters);
        shaderDesc.uniform_blocks[0].msl_buffer_n = 0;
        shaderDesc.label = "arbit-metal-physical-diffraction-shader";
        impl_->shader = sg_make_shader(&shaderDesc);

        sg_pipeline_desc pipelineDesc = {};
        pipelineDesc.shader = impl_->shader;
        pipelineDesc.color_count = static_cast<int>(impl_->images.size());
        for (auto& color : pipelineDesc.colors)
            color.pixel_format = SG_PIXELFORMAT_NONE;
        for (std::size_t index = 0; index < impl_->images.size(); ++index)
            pipelineDesc.colors[index].pixel_format = SG_PIXELFORMAT_RGBA32F;
        pipelineDesc.depth.pixel_format = SG_PIXELFORMAT_NONE;
        pipelineDesc.primitive_type = SG_PRIMITIVETYPE_TRIANGLES;
        pipelineDesc.sample_count = 1;
        pipelineDesc.label = "arbit-metal-physical-diffraction-pipeline";
        impl_->pipeline = sg_make_pipeline(&pipelineDesc);

        const auto shaderState = sg_query_shader_state(impl_->shader);
        const auto pipelineState = sg_query_pipeline_state(impl_->pipeline);
        std::array<sg_resource_state, 4> imageStates{};
        std::array<sg_resource_state, 4> attachmentStates{};
        std::array<sg_resource_state, 4> textureStates{};
        bool resourcesOk = resourceValid(shaderState) && resourceValid(pipelineState);
        for (std::size_t index = 0; index < impl_->images.size(); ++index)
        {
            imageStates[index] = sg_query_image_state(impl_->images[index]);
            attachmentStates[index] = sg_query_view_state(impl_->attachmentViews[index]);
            textureStates[index] = sg_query_view_state(impl_->textureViews[index]);
            resourcesOk = resourcesOk
                && resourceValid(imageStates[index])
                && resourceValid(attachmentStates[index])
                && resourceValid(textureStates[index]);
        }
        if (!resourcesOk)
        {
            const auto contextError = arbitgpu::sokolmetal::error();
            const auto contextLog = arbitgpu::sokolmetal::log();
            std::string resourceStates = " (shader="
                + std::to_string(static_cast<int>(shaderState)) + ", pipeline="
                + std::to_string(static_cast<int>(pipelineState));
            for (std::size_t index = 0; index < impl_->images.size(); ++index)
            {
                resourceStates += ", image" + std::to_string(index) + "="
                    + std::to_string(static_cast<int>(imageStates[index])) + ", attachment"
                    + std::to_string(index) + "="
                    + std::to_string(static_cast<int>(attachmentStates[index])) + ", texture"
                    + std::to_string(index) + "="
                    + std::to_string(static_cast<int>(textureStates[index]));
            }
            resourceStates += ")";
            impl_->destroyUnlocked();
            error = "physical diffraction Metal GPU resource creation failed" + resourceStates;
            if (!contextError.empty())
                error += ": " + contextError;
            else if (!contextLog.empty())
                error += ": " + contextLog;
            return false;
        }
    }
    return true;
}

bool MetalPhysicalDiffractionExecutor::execute(
    const AdmittedDiffractionMaterialIR& material,
    const SpectralIncidentLight& light,
    MetalPhysicalDiffractionFrame& output,
    std::string& error)
{
    return executeAtMaterialUv(material, light, { 0.5f, 0.5f }, output, error);
}

bool MetalPhysicalDiffractionExecutor::executeAtMaterialUv(
    const AdmittedDiffractionMaterialIR& material,
    const SpectralIncidentLight& light,
    const std::array<float, 2>& materialUv,
    MetalPhysicalDiffractionFrame& output,
    std::string& error)
{
    output = {};
    error.clear();
    std::lock_guard<std::mutex> lock(arbitgpu::sokolmetal::mutex());
    @autoreleasepool
    {
        if (!ready())
        {
            error = "physical diffraction native Metal backend is unavailable";
            return false;
        }
        if (!physicalcheckpoint::validate(material, light, "Metal", error))
            return false;

        auto parameters = physicalcheckpoint::makeGpuParameters(material, light);
        if (!std::isfinite(materialUv[0]) || !std::isfinite(materialUv[1])
            || materialUv[0] < 0.0f || materialUv[0] > 1.0f
            || materialUv[1] < 0.0f || materialUv[1] > 1.0f)
        {
            error = "physical diffraction Metal material UV is invalid";
            return false;
        }
        parameters.grooveField.y += 0.5f - materialUv[0];
        parameters.grooveField.z += 0.5f - materialUv[1];
        sg_pass pass = {};
        for (std::size_t index = 0; index < impl_->attachmentViews.size(); ++index)
        {
            pass.attachments.colors[index] = impl_->attachmentViews[index];
            pass.action.colors[index].load_action = SG_LOADACTION_CLEAR;
            pass.action.colors[index].store_action = SG_STOREACTION_STORE;
            pass.action.colors[index].clear_value = { 0.0f, 0.0f, 0.0f, 0.0f };
        }
        pass.label = "arbit-metal-physical-diffraction-pass";
        sg_begin_pass(&pass);
        sg_apply_pipeline(impl_->pipeline);
        const sg_range uniformRange = { &parameters, sizeof(parameters) };
        sg_apply_uniforms(0, &uniformRange);
        sg_draw(0, 3, 1);
        sg_end_pass();
        sg_commit();

        id<MTLDevice> device = (__bridge id<MTLDevice>) arbitgpu::sokolmetal::device();
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
        id<MTLBuffer> readback = [device newBufferWithLength:64
                                                    options:MTLResourceStorageModeShared];
        if (device == nil || queue == nil || readback == nil)
        {
#if !__has_feature(objc_arc)
            [readback release];
#endif
            error = "physical diffraction Metal validation readback allocation failed";
            return false;
        }

        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        bool nativeResourcesOk = command != nil && blit != nil;
        for (std::size_t index = 0; index < impl_->images.size(); ++index)
        {
            const auto native = sg_mtl_query_image_info(impl_->images[index]);
            id<MTLTexture> texture = (__bridge id<MTLTexture>)
                native.tex[native.active_slot];
            nativeResourcesOk = nativeResourcesOk && texture != nil;
            output.imageHandles[index] = impl_->images[index].id;
            output.textureViewHandles[index] = impl_->textureViews[index].id;
            output.metalTextureHandles[index] = reinterpret_cast<std::uintptr_t>(
                (__bridge void*) texture);
            if (texture != nil && blit != nil)
            {
                [blit copyFromTexture:texture
                          sourceSlice:0
                          sourceLevel:0
                         sourceOrigin:MTLOriginMake(0, 0, 0)
                           sourceSize:MTLSizeMake(1, 1, 1)
                             toBuffer:readback
                    destinationOffset:index * 16
               destinationBytesPerRow:16
             destinationBytesPerImage:16];
            }
        }
        if (blit != nil)
            [blit endEncoding];
        if (!nativeResourcesOk)
        {
#if !__has_feature(objc_arc)
            [readback release];
#endif
            output = {};
            error = "physical diffraction Metal native resource lookup failed";
            return false;
        }

        [command commit];
        [command waitUntilCompleted];
        const bool completed = command.status == MTLCommandBufferStatusCompleted;
        if (completed && readback.contents != nullptr)
        {
            auto* bytes = static_cast<const unsigned char*>(readback.contents);
            std::memcpy(output.linearSrgb.data(), bytes, 16);
            std::memcpy(output.energyLedger.data(), bytes + 16, 16);
            std::memcpy(output.orderLedger.data(), bytes + 32, 16);
            std::memcpy(output.directionMoment.data(), bytes + 48, 16);
        }
#if !__has_feature(objc_arc)
        [readback release];
#endif
        if (!completed)
        {
            output = {};
            error = "physical diffraction Metal command submission failed";
            return false;
        }
        if (!physicalcheckpoint::validateGpuReadback(
                output.linearSrgb, output.energyLedger, output.orderLedger,
                output.directionMoment,
                physicalcheckpoint::expectedSignedOrderEvaluations(material)))
        {
            output = {};
            error = "physical diffraction Metal validation readback rejected shader output";
            return false;
        }

        output.generation = ++impl_->generation;
        return true;
    }
}

bool MetalPhysicalDiffractionExecutor::executeTransport(
    const AdmittedDiffractionMaterialIR& material,
    const AdmittedLightingPlan& lighting,
    MetalPhysicalDiffractionFrame& output,
    TransportEnergyReceipt& receipt,
    std::string& error)
{
    output = {};
    receipt = {};
    const auto& description = lighting.description();
    for (std::size_t pathIndex = 0; pathIndex < description.pathCount; ++pathIndex)
    {
        MetalPhysicalDiffractionFrame pathFrame;
        if (!execute(material, description.paths[pathIndex].incident, pathFrame, error))
            return false;
        output = pathFrame;
        for (std::size_t component = 0; component < 4; ++component)
        {
            receipt.energyLedger[component] += pathFrame.energyLedger[component];
            receipt.orderLedger[component] += pathFrame.orderLedger[component];
            receipt.directionMoment[component] += pathFrame.directionMoment[component];
        }
        receipt.maximumBounceDepth = std::max(
            receipt.maximumBounceDepth, description.paths[pathIndex].bounceDepth);

        ++receipt.pathCount;
    }
    return true;
}

void MetalPhysicalDiffractionExecutor::shutdown() noexcept
{
    if (impl_ == nullptr)
        return;
    std::lock_guard<std::mutex> lock(arbitgpu::sokolmetal::mutex());
    if (impl_->pipeline.id != 0 || impl_->shader.id != 0)
        impl_->destroyUnlocked();
}

bool MetalPhysicalDiffractionExecutor::ready() const noexcept
{
    return impl_ != nullptr && impl_->pipeline.id != 0 && impl_->shader.id != 0;
}
} // namespace diffractionmaterial
