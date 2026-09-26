#include "sdf_native_backend_parity.h"
#include "sdf_reference_evaluator.h"
#include "sdf_output_test_scenes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <utility>

namespace videohelper::sdf::test
{
namespace
{
using videowire::SdfIr;
using videowire::SdfOperation;
using videowire::SdfRecord;
using videowire::SdfStableId;

SdfRecord record (SdfStableId id, SdfOperation operation,
                  std::initializer_list<SdfStableId> inputs,
                  std::initializer_list<double> parameters)
{
    SdfRecord result;
    result.stableId = id;
    result.operation = operation;
    result.inputCount = static_cast<std::uint8_t> (inputs.size());
    result.parameterCount = static_cast<std::uint8_t> (parameters.size());
    std::copy (inputs.begin(), inputs.end(), result.inputs.begin());
    std::copy (parameters.begin(), parameters.end(), result.parameters.begin());
    return result;
}

std::vector<double> parametersFor (SdfOperation operation,
                                   double polarOffset = 10000.0)
{
    switch (operation)
    {
        case SdfOperation::sphere: return { 0.72 };
        case SdfOperation::box: return { 0.7, 0.55, 0.65 };
        case SdfOperation::roundedBox: return { 0.75, 0.6, 0.7, 0.12 };
        case SdfOperation::plane: return { 0.0, 0.0, 1.0, 0.0 };
        case SdfOperation::torus: return { 0.45, 0.24 };
        case SdfOperation::capsule: return { 0.0, -0.65, 0.0, 0.0, 0.65, 0.0, 0.24 };
        case SdfOperation::cylinder: return { 0.62, 0.8 };
        case SdfOperation::cone: return { 0.7, 0.9 };
        case SdfOperation::gyroid: return { 2.0, 0.14 };
        case SdfOperation::smoothUnion:
        case SdfOperation::smoothIntersection:
        case SdfOperation::smoothSubtraction: return { 0.22 };
        case SdfOperation::translate: return { 0.18, -0.12, 0.2 };
        case SdfOperation::rotate: return { 1.0, 2.0, -0.5, 0.7 };
        case SdfOperation::scale: return { 1.25, 0.8, 1.4 };
        case SdfOperation::repeat: return { 1.7, 2.1, 2.5 };
        case SdfOperation::polarRepeat: return { 2.0, 7.0, polarOffset };
        case SdfOperation::mirror: return { 1.0, 0.0, 1.0 };
        case SdfOperation::twist: return { 2.0, 0.65 };
        case SdfOperation::bend: return { 0.0, -0.45 };
        case SdfOperation::taper: return { 1.0, 0.3 };
        case SdfOperation::displacement: return { 0.12, 2.3 };
        case SdfOperation::domainWarp: return { 0.16, -0.12, 0.1, 1.2, 1.7, 2.1 };
        case SdfOperation::unionOp:
        case SdfOperation::intersection:
        case SdfOperation::subtraction: return {};
    }
    return {};
}

SdfRecord operationRecord (SdfStableId id, SdfOperation operation,
                           std::initializer_list<SdfStableId> inputs,
                           double polarOffset = 10000.0)
{
    const auto parameters = parametersFor (operation, polarOffset);
    SdfRecord result;
    result.stableId = id;
    result.operation = operation;
    result.inputCount = static_cast<std::uint8_t> (inputs.size());
    result.parameterCount = static_cast<std::uint8_t> (parameters.size());
    std::copy (inputs.begin(), inputs.end(), result.inputs.begin());
    std::copy (parameters.begin(), parameters.end(), result.parameters.begin());
    return result;
}

std::shared_ptr<const AdmittedSdfIr> fixtureFor (SdfOperation operation,
                                                 std::string& error,
                                                 double polarOffset = 10000.0)
{
    const auto schema = videowire::sdfOperationSchema (operation);
    SdfIr source;
    if (schema.inputCount == 0)
    {
        source.rootId = 1;
        source.records = { operationRecord (1, operation, {}) };
    }
    else if (schema.inputCount == 2)
    {
        source.rootId = 3;
        source.records = {
            record (1, SdfOperation::sphere, {}, { 0.72 }),
            record (2, SdfOperation::box, {}, { 0.48, 0.62, 0.54 }),
            operationRecord (3, operation, { 1, 2 })
        };
    }
    else
    {
        source.rootId = operation == SdfOperation::translate ? 2 : 3;
        const auto baseRadius = operation == SdfOperation::translate ? 0.72
            : (operation == SdfOperation::polarRepeat ? 0.45 : 0.3);
        source.records = {
            record (1, SdfOperation::sphere, {}, { baseRadius })
        };
        if (operation == SdfOperation::translate)
            source.records.push_back (operationRecord (2, operation, { 1 }));
        else
        {
            source.records.push_back (
                record (2, SdfOperation::translate, { 1 },
                        { 0.55, operation == SdfOperation::polarRepeat ? 0.0 : 0.08, 0.0 }));
            source.records.push_back (operationRecord (
                3, operation, { 2 }, polarOffset));
        }
    }

    auto admitted = admitSdfIr (source, {}, error);
    if (! admitted) return {};
    return std::make_shared<const AdmittedSdfIr> (std::move (*admitted));
}

double qualityScale (arbitgpu::NativeSdfQuality quality)
{
    switch (quality)
    {
        case arbitgpu::NativeSdfQuality::low: return 4.0;
        case arbitgpu::NativeSdfQuality::medium: return 2.0;
        case arbitgpu::NativeSdfQuality::high: return 1.0;
        case arbitgpu::NativeSdfQuality::ultra: return 0.5;
        case arbitgpu::NativeSdfQuality::count: break;
    }
    return 1.0;
}

std::uint8_t expectedDepth (const AdmittedSdfIr& geometry,
                            const NativeSdfRenderControls& controls,
                            std::uint32_t width, std::uint32_t height,
                            std::uint32_t x, std::uint32_t y)
{
    const auto u = (2.0 * (static_cast<double> (x) + 0.5) - width) / height;
    const auto v = (2.0 * (static_cast<double> (y) + 0.5) - height) / height;
    const auto inverseLength = 1.0 / std::sqrt (u * u + v * v + 1.8 * 1.8);
    const reference::Point3 direction { u * inverseLength, v * inverseLength,
                                        -1.8 * inverseLength };
    double travel = 0.0;
    bool hit = false;
    for (std::uint32_t step = 0; step < controls.maximumSteps && step < 512; ++step)
    {
        if (travel > controls.maximumDistance) break;
        const reference::Point3 point { direction.x * travel,
                                        direction.y * travel,
                                        3.0 + direction.z * travel };
        const auto field = reference::evaluatePoint (geometry, point);
        const auto threshold = std::max (
            controls.epsilon * qualityScale (controls.adaptiveQuality)
                * std::max (1.0, travel * 0.05),
            0.000001);
        if (field <= threshold)
        {
            hit = true;
            break;
        }
        travel += field;
    }
    if (! hit) return 255;
    const auto normalized = std::clamp (travel / controls.maximumDistance, 0.0, 1.0);
    return static_cast<std::uint8_t> (std::lround (normalized * 255.0));
}

bool verifyPositiveYSphere (const NativeSdfFloatReader& readPixels,
                            const std::string& backend, std::string& error)
{
    const auto reject = [&] (const char* message) { error = message; return false; };
    SdfIr source;
    source.rootId = 72;
    source.records = { record(71, SdfOperation::sphere, {}, {0.45}),
                       record(72, SdfOperation::translate, {71}, {0.0, 0.6, 0.0}) };
    auto admitted = admitSdfIr(source, {}, error);
    if (!admitted) return false;
    const auto geometry = std::make_shared<const AdmittedSdfIr>(std::move(*admitted));
    NativeSdfRenderControls controls;
    controls.maximumSteps = 128;
    controls.maximumDistance = 8.0;
    controls.output = arbitgpu::NativeSdfOutput::depth;
    constexpr std::uint32_t extent = 33;
    const auto rowOrder = backend == "opengl" ? arbitgpu::NativeTextureRowOrder::BottomFirst
                                               : arbitgpu::NativeTextureRowOrder::TopFirst;
    // Closed-form intersections for camera O=(0,0,3), sphere C=(0,0.6,0),
    // radius 0.45, and focal length 1.8. In canonical top-first coordinates,
    // pixel (16,10) hits at t=2.609445154144592; its mirror (16,22) misses.
    // These values do not use the scalar IR evaluator or native ray marcher.
    constexpr std::array<std::pair<std::uint32_t, double>, 2> probes {{
        {10, 0.326180644268074}, {22, 1.0}
    }};
    std::vector<float> previewPixels;
    auto& renderer = nativeSdfRenderer();
    for (const auto use : {NativeSdfRenderUse::Preview, NativeSdfRenderUse::Export})
    {
        NativeSdfRenderedFrame rendered;
        const bool ok = use == NativeSdfRenderUse::Preview
            ? renderer.renderPreview(geometry, {extent, extent}, controls, kNativeGpuCapability, rendered, error)
            : renderer.renderExport(geometry, {extent, extent}, controls, kNativeGpuCapability, rendered, error);
        if (!ok || !rendered.nativeFrame) return false;
        const auto descriptor = rendered.nativeFrame->colorTextureDescriptor();
        if (!arbitgpu::isLinearSceneColor(descriptor) || descriptor.backend != backend
            || descriptor.rowOrder != rowOrder || descriptor.width != extent || descriptor.height != extent)
            return reject("positive-Y SDF sphere lost its native row descriptor");
        const auto pixels = readPixels(*rendered.nativeFrame);
        if (pixels.size() != extent * extent * 4u)
            return reject("positive-Y SDF sphere readback has the wrong extent");
        for (const auto& [topRow, expectedDepth] : probes)
        {
            const auto nativeRow = rowOrder == arbitgpu::NativeTextureRowOrder::BottomFirst
                ? extent - 1u - topRow : topRow;
            const auto offset = (nativeRow * extent + 16u) * 4u;
            for (std::size_t channel = 0; channel < 4; ++channel)
            {
                const auto expected = channel == 3 ? 1.0 : expectedDepth;
                if (!std::isfinite(pixels[offset + channel])
                    || std::abs(pixels[offset + channel] - expected) > 0.002)
                    return reject("positive scene Y must place the SDF sphere above centre in canonical top-first pixels");
            }
        }
        if (use == NativeSdfRenderUse::Preview) previewPixels = pixels;
        else if (pixels != previewPixels)
            return reject("positive-Y SDF sphere differs between native preview and export");
    }
    return true;
}
} // namespace

std::vector<NativeOperationFixture> nativeOperationFixtures (std::string& error)
{
    std::vector<NativeOperationFixture> result;
    result.reserve (26);
    for (std::uint32_t raw = static_cast<std::uint32_t> (SdfOperation::sphere);
         raw <= static_cast<std::uint32_t> (SdfOperation::domainWarp); ++raw)
    {
        const auto operation = static_cast<SdfOperation> (raw);
        const auto polarOffset = operation == SdfOperation::polarRepeat ? 0.0 : 10000.0;
        auto geometry = fixtureFor (operation, error, polarOffset);
        if (! geometry)
        {
            error = std::string (videowire::sdfOperationWireToken (operation))
                + " parity fixture failed admission: " + error;
            return {};
        }
        result.push_back ({ operation, std::move (geometry) });
    }
    if (result.size() != 26)
    {
        error = "native SDF parity fixture count is not 26";
        return {};
    }
    error.clear();
    return result;
}

std::vector<NativeOperationFixture> largeOffsetPolarRepeatFixtures (std::string& error)
{
    std::vector<NativeOperationFixture> result;
    for (const auto offset : { -10000.0, 10000.0 })
    {
        auto geometry = fixtureFor (SdfOperation::polarRepeat, error, offset);
        if (geometry == nullptr) return {};
        result.push_back ({ SdfOperation::polarRepeat, std::move (geometry) });
    }
    error.clear();
    return result;
}

bool verifyNativeDepthParity (const NativeOperationFixture& fixture,
                              const NativeSdfRenderControls& controls,
                              const std::vector<std::uint8_t>& rgba,
                              std::uint32_t width,
                              std::uint32_t height,
                              std::string& error)
{
    if (rgba.size() != static_cast<std::size_t> (width) * height * 4u)
    {
        error = "native SDF parity readback has the wrong byte count";
        return false;
    }

    bool expectedForeground = false;
    // Depth is normalized into one byte by the production output path. Two
    // byte levels cover half-float/readback rounding while still catching a
    // different raymarch result. Foreground membership must match exactly.
    constexpr int polarRepeatDepthTolerance = 2;
    const auto inset = fixture.operation == SdfOperation::repeat ? width / 3 : 0u;
    const bool topFirst = arbitgpu::nativeSdfExecutionBackend().capabilities().backend == "metal";
    for (std::uint32_t y = inset; y < height - inset; ++y)
    {
        for (std::uint32_t x = inset; x < width - inset; ++x)
        {
            const auto expected = expectedDepth (*fixture.geometry, controls, width, height, x, y);
            expectedForeground = expectedForeground || expected != 255;
            // Keep every scalar probe in its original world-space ray direction.
            const auto nativeRow = topFirst ? height - 1u - y : y;
            const auto offset = (static_cast<std::size_t> (nativeRow) * width + x) * 4u;
            const auto actual = rgba[offset];
            const auto delta = std::abs (static_cast<int> (actual) - static_cast<int> (expected));
            const auto repeatedDomain = fixture.operation == SdfOperation::polarRepeat;
            const auto maskMismatch = (expected == 255) != (actual == 255);
            const auto distanceMismatch = repeatedDomain
                ? (maskMismatch || (expected != 255 && delta > polarRepeatDepthTolerance))
                : delta > 2;
            if (distanceMismatch || rgba[offset + 1] != actual || rgba[offset + 2] != actual
                || rgba[offset + 3] != 255)
            {
                std::ostringstream message;
                message << videowire::sdfOperationWireToken (fixture.operation)
                        << " native depth differs from the CPU oracle at " << x << ',' << y
                        << ": expected " << static_cast<int> (expected)
                        << ", actual " << static_cast<int> (actual);
                error = message.str();
                return false;
            }
        }
    }
    if (! expectedForeground)
    {
        error = std::string (videowire::sdfOperationWireToken (fixture.operation))
            + " parity fixture has no oracle-visible samples";
        return false;
    }

    error.clear();
    return true;
}

bool verifyNativeUtilityOutputs (const NativeSdfFloatReader& readPixels, std::string& error)
{
    using Output = arbitgpu::NativeSdfOutput;
    auto& backend = arbitgpu::nativeSdfExecutionBackend();
    const auto capabilities = backend.capabilities();
    const auto reject = [&] (std::string message) { error = std::move (message); return false; };
    for (unsigned raw = 0; raw < static_cast<unsigned> (Output::count); ++raw)
        if (! capabilities.supports (static_cast<Output> (raw)))
            return reject ("native SDF capability omitted an implemented output");
    const std::array<Output,3> invalidOutputs {
        static_cast<Output> (-1), Output::count, static_cast<Output> (9) };
    for (const auto output : invalidOutputs)
        if (capabilities.supports (output))
            return reject ("native SDF capability admits an invalid output");
    if (capabilities.backend != "opengl" && capabilities.backend != "metal")
        return reject ("native SDF parity requires a declared GL or Metal row origin");
    if (!verifyPositiveYSphere(readPixels, capabilities.backend, error)) return false;

    NativeSdfRenderControls controls;
    controls.maximumSteps = 128;
    controls.maximumDistance = 12.0;
    controls.shadowQuality = arbitgpu::NativeSdfQuality::ultra;
    struct Case
    {
        const char* name;
        OutputScene scene;
        Output output;
        std::vector<std::array<unsigned,2>> probes;
        double tolerance;
    };
    const std::vector<Case> cases {
        {"plane color",OutputScene::Plane,Output::color,{{16,16},{8,20},{28,4}},0.002},
        {"sphere color",OutputScene::Sphere,Output::color,{{16,16},{19,16},{16,20},{0,0}},0.004},
        {"sphere depth",OutputScene::Sphere,Output::depth,{{16,16},{19,16},{16,20},{0,0}},0.002},
        {"sphere normal",OutputScene::Sphere,Output::normal,{{16,16},{19,16},{16,20},{0,0}},0.003},
        {"flat curvature",OutputScene::Plane,Output::curvature,{{16,16},{8,20},{28,4}},0.01},
        {"convex curvature",OutputScene::Sphere,Output::curvature,{{16,16},{19,16},{16,20},{0,0}},0.02},
        {"concave cut curvature",OutputScene::CarvedBox,Output::curvature,{{16,16},{17,16},{16,17},{0,0}},0.025},
        {"open AO",OutputScene::Plane,Output::ambientOcclusion,{{16,16},{8,20},{28,4}},0.01},
        {"corner AO",OutputScene::Corner,Output::ambientOcclusion,{{16,16},{8,16},{16,8}},0.01},
        {"open shadow",OutputScene::Plane,Output::softShadow,{{16,16},{8,20},{28,4}},0.01},
        {"occluder shadow",OutputScene::OccludedPlane,Output::softShadow,{{16,16},{26,16},{16,8}},0.015},
        {"sphere silhouette",OutputScene::Sphere,Output::edgeDistance,{{16,16},{26,16},{16,26},{0,0}},0.064},
        {"flat face silhouette",OutputScene::Box,Output::edgeDistance,{{16,16},{25,16},{16,25},{0,0}},0.064},
        {"unbounded plane silhouette",OutputScene::Plane,Output::edgeDistance,{{16,16},{0,0},{32,32}},0.001},
        {"64 bit contributors",OutputScene::Pair,Output::materialId,{{9,16},{23,16},{16,16},{0,0}},0.001},
        {"union contributors",OutputScene::Union,Output::materialId,{{14,16},{16,16},{18,16},{0,0}},0.001},
        {"intersection contributors",OutputScene::Intersection,Output::materialId,{{14,16},{16,16},{18,16},{0,0}},0.001},
        {"smooth union contributors",OutputScene::SmoothUnion,Output::materialId,{{14,16},{16,16},{18,16},{0,0}},0.001},
        {"smooth intersection contributors",OutputScene::SmoothIntersection,Output::materialId,{{14,16},{16,16},{18,16},{0,0}},0.001},
        {"cut contributor",OutputScene::CarvedBox,Output::materialId,{{16,16},{24,16},{0,0}},0.001},
        {"smooth cut contributor",OutputScene::SmoothCarvedBox,Output::materialId,{{16,16},{24,16},{0,0}},0.001}
    };
    constexpr unsigned extent = 33;
    const auto offsetAt = [topFirst = capabilities.backend == "metal"] (unsigned x, unsigned y)
    {
        // Existing scalar oracle probes use bottom-origin camera coordinates.
        const auto nativeRow = topFirst ? extent - 1u - y : y;
        return (nativeRow * extent + x) * 4u;
    };
    const auto channelAt = [&] (const std::vector<float>& pixels, unsigned x, unsigned y)
    { return pixels[offsetAt(x, y)]; };
    auto& renderer = nativeSdfRenderer();
    std::vector<float> pairPixels;
    std::array<bool,static_cast<std::size_t> (Output::count)> sampledOutputs {};
    for (const auto& test : cases)
    {
        const auto geometry = outputTestGeometry (test.scene,error);
        if (!geometry) return false;
        controls.output = test.output;
        sampledOutputs[static_cast<std::size_t> (test.output)] = true;
        NativeSdfRenderedFrame preview, exported;
        if (! renderer.renderPreview (geometry,{extent,extent},controls,kNativeGpuCapability,preview,error)
            || ! renderer.renderExport (geometry,{extent,extent},controls,kNativeGpuCapability,exported,error)
            || !preview.nativeFrame || !exported.nativeFrame)
            return reject (std::string (test.name) + " native draw failed: " + error);
        const auto previewDescriptor = preview.nativeFrame->colorTextureDescriptor();
        const auto exportDescriptor = exported.nativeFrame->colorTextureDescriptor();
        const auto rowOrder = capabilities.backend == "opengl"
            ? arbitgpu::NativeTextureRowOrder::BottomFirst : arbitgpu::NativeTextureRowOrder::TopFirst;
        if (!arbitgpu::isLinearSceneColor(previewDescriptor)
            || !arbitgpu::isLinearSceneColor(exportDescriptor)
            || previewDescriptor.rowOrder != rowOrder || exportDescriptor.rowOrder != rowOrder
            || previewDescriptor.deviceOrContextIdentity != exportDescriptor.deviceOrContextIdentity
            || previewDescriptor.rendererGeneration == exportDescriptor.rendererGeneration)
            return reject (std::string (test.name) + " native attachment descriptors lost format, orientation or allocation identity");
        const auto pixels = readPixels (*preview.nativeFrame);
        const auto exportPixels = readPixels (*exported.nativeFrame);
        if (pixels.size() != extent*extent*4u || pixels != exportPixels)
            return reject (std::string (test.name) + " preview/export float pixels or extent disagree");
        for (std::size_t i = 0; i < pixels.size(); ++i)
            if (!std::isfinite (pixels[i]) || pixels[i] < 0.0f
                || (test.output != Output::color && pixels[i] > 1.0f)
                || (i%4u == 3u && pixels[i] != 1.0f))
                return reject (std::string (test.name) + " contains a nonfinite, unbounded or invalid-alpha native pixel");
        for (const auto& probe : test.probes)
        {
            const auto expected = reference::evaluateOutputPixel (
                *geometry,controls,extent,extent,probe[0],probe[1]);
            const auto offset = offsetAt(probe[0], probe[1]);
            for (unsigned channel = 0; channel < 4; ++channel)
                if (!std::isfinite (expected[channel])
                    || std::abs (pixels[offset+channel]-expected[channel]) > test.tolerance)
                {
                    std::ostringstream message;
                    message << test.name << " differs from scalar IR oracle at " << probe[0] << ','
                        << probe[1] << " channel " << channel << ": expected " << expected[channel]
                        << ", actual " << pixels[offset+channel];
                    return reject (message.str());
                }
        }
        const auto center = channelAt (pixels,16,16);
        if (test.output == Output::curvature)
        {
            if (test.scene == OutputScene::Plane && std::abs (center-0.5f) > 0.01f)
                return reject ("native plane curvature is not neutral");
            if (test.scene == OutputScene::Sphere && (center < 0.73f || center > 0.77f))
                return reject ("native convex sphere does not show positive 1/R curvature");
            if (test.scene == OutputScene::CarvedBox && center > 0.2f)
                return reject ("native spherical cut lost its negative curvature");
        }
        if (test.scene == OutputScene::Corner && test.output == Output::ambientOcclusion
            && (center > 0.85f || channelAt (pixels,8,16) < 0.99f))
            return reject ("native AO does not distinguish the corner from an open region");
        if (test.scene == OutputScene::OccludedPlane && test.output == Output::softShadow
            && (center > 0.01f || channelAt (pixels,26,16) < 0.99f))
            return reject ("native shadow does not follow the light/occluder geometry");
        if (test.output == Output::edgeDistance && test.scene != OutputScene::Plane)
        {
            const auto edgeX = test.scene == OutputScene::Sphere ? 26u : 25u;
            if (center < 0.99f || channelAt (pixels,edgeX,16) >= 0.2f || channelAt (pixels,0,0) != 0.0f)
                return reject ("native silhouette map does not measure foreground edge distance");
        }
        if (test.scene == OutputScene::Union || test.scene == OutputScene::Intersection
            || test.scene == OutputScene::SmoothUnion || test.scene == OutputScene::SmoothIntersection)
        {
            auto reversed = outputTestSource (test.scene);
            std::swap (reversed.records.back().inputs[0],reversed.records.back().inputs[1]);
            auto admitted = admitSdfIr (reversed, outputTestAdmissionLimits(), error);
            if (!admitted) return false;
            const auto reverseGeometry = std::make_shared<const AdmittedSdfIr> (std::move (*admitted));
            NativeSdfRenderedFrame reverseFrame;
            if (!renderer.renderPreview (reverseGeometry,{extent,extent},controls,
                                         kNativeGpuCapability,reverseFrame,error) || !reverseFrame.nativeFrame)
                return reject (std::string (test.name) + " reversed-input draw failed: " + error);
            const auto reversePixels = readPixels (*reverseFrame.nativeFrame);
            if (reversePixels.size() != pixels.size())
                return reject (std::string (test.name) + " reversed-input readback failed");
            // The center ray is exactly equidistant to both spheres. Geometry
            // stays the same when inputs swap; ordered A must own that seam.
            constexpr std::array<double,3> leftColor {98.0/255.0,124.0/255.0,48.0/255.0};
            constexpr std::array<double,3> rightColor {116.0/255.0,162.0/255.0,48.0/255.0};
            const auto centerOffset = (16*extent+16)*4u;
            for (unsigned channel = 0; channel < 3; ++channel)
                if (!std::isfinite (reversePixels[centerOffset+channel])
                    || std::abs (pixels[centerOffset+channel]-leftColor[channel]) > 0.001
                    || std::abs (reversePixels[centerOffset+channel]-rightColor[channel]) > 0.001)
                    return reject (std::string (test.name) + " does not preserve ordered A ownership on an equal-weight seam");
        }
        if (test.scene == OutputScene::Pair) pairPixels = pixels;
    }
    if (!std::all_of (sampledOutputs.begin(),sampledOutputs.end(),[] (bool sampled) { return sampled; }))
        return reject ("native SDF semantic pixels did not cover all eight outputs");
    // Record order is not material identity. The two leaves also share their
    // low 32 bits, so truncating the full stable ID makes this test fail.
    if (pairPixels.empty()
        || std::equal (pairPixels.begin()+(16*extent+9)*4,pairPixels.begin()+(16*extent+9)*4+3,
                       pairPixels.begin()+(16*extent+23)*4))
        return reject ("distinct 64-bit primitive contributors share an invented constant material color");
    auto reordered = outputTestSource (OutputScene::Pair);
    std::reverse (reordered.records.begin(),reordered.records.end());
    auto admittedReordered = admitSdfIr (reordered, outputTestAdmissionLimits(), error);
    if (!admittedReordered) return false;
    const auto geometry = std::make_shared<const AdmittedSdfIr> (std::move (*admittedReordered));
    controls.output = Output::materialId;
    NativeSdfRenderedFrame reorderedFrame;
    if (!renderer.renderPreview (geometry,{extent,extent},controls,kNativeGpuCapability,reorderedFrame,error)
        || !reorderedFrame.nativeFrame || readPixels (*reorderedFrame.nativeFrame) != pairPixels)
        return reject ("SDF material colors changed when only record order changed");

    arbitgpu::NativeSdfDrawRequest invalid;
    invalid.geometry = outputTestSource (OutputScene::Sphere);
    invalid.width = invalid.height = extent;
    invalid.maximumSteps = controls.maximumSteps;
    invalid.epsilon = controls.epsilon;
    invalid.maximumDistance = controls.maximumDistance;
    for (const auto output : invalidOutputs)
    {
        invalid.output = output;
        const auto result = backend.render (invalid);
        if (result.rendered || result.frame || result.error.empty())
            return reject ("invalid native SDF output allocated a frame");
    }
    invalid.output = Output::curvature;
    invalid.epsilon = std::numeric_limits<double>::quiet_NaN();
    const auto nonfinite = backend.render (invalid);
    invalid.epsilon = 0.001;
    invalid.maximumSteps = 0;
    const auto zeroSteps = backend.render (invalid);
    if (nonfinite.rendered || nonfinite.frame || nonfinite.error.empty()
        || zeroSteps.rendered || zeroSteps.frame || zeroSteps.error.empty())
        return reject ("invalid SDF utility controls did not reject before native allocation");
    error.clear();
    return true;
}
} // namespace videohelper::sdf::test
