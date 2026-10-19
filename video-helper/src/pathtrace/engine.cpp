#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace videohelper::pathtrace
{

using HarmonicMIDI::grid::SceneCameraRecord;
using HarmonicMIDI::grid::SceneLightKind;
using HarmonicMIDI::grid::SceneLightRecord;
using HarmonicMIDI::grid::SceneTransform3D;

namespace
{
bool sameBits (float a, float b) noexcept
{
    return floatBits (a) == floatBits (b);
}

bool sameTransform (const SceneTransform3D& a, const SceneTransform3D& b) noexcept
{
    return sameBits (a.translation.x, b.translation.x) && sameBits (a.translation.y, b.translation.y)
        && sameBits (a.translation.z, b.translation.z) && sameBits (a.rotation.x, b.rotation.x)
        && sameBits (a.rotation.y, b.rotation.y) && sameBits (a.rotation.z, b.rotation.z)
        && sameBits (a.rotation.w, b.rotation.w) && sameBits (a.scale.x, b.scale.x)
        && sameBits (a.scale.y, b.scale.y) && sameBits (a.scale.z, b.scale.z);
}

bool sameVec3 (Vec3 a, Vec3 b) noexcept
{
    return sameBits (a.x, b.x) && sameBits (a.y, b.y) && sameBits (a.z, b.z);
}

std::uint8_t unorm8 (float value) noexcept
{
    if (! (value > 0.0f)) return 0;
    if (value >= 1.0f) return 255;
    return static_cast<std::uint8_t> (std::lround (value * 255.0f));
}
} // namespace

bool sameLightRecord (const SceneLightRecord& a, const SceneLightRecord& b) noexcept
{
    return a.id.value == b.id.value && a.kind == b.kind && sameTransform (a.transform, b.transform)
        && sameBits (a.color.x, b.color.x) && sameBits (a.color.y, b.color.y) && sameBits (a.color.z, b.color.z)
        && sameBits (a.intensity, b.intensity) && sameBits (a.range, b.range)
        && sameBits (a.innerConeAngle, b.innerConeAngle) && sameBits (a.outerConeAngle, b.outerConeAngle)
        && HarmonicMIDI::grid::sameAnalyticSky (a.sky, b.sky)
        && HarmonicMIDI::grid::sameEnvironmentImage (a.environmentImage, b.environmentImage)
        && a.spectralSource.version == b.spectralSource.version
        && std::memcmp (a.spectralSource.wavelengthsNanometres.data(), b.spectralSource.wavelengthsNanometres.data(),
                        sizeof (float) * a.spectralSource.wavelengthsNanometres.size()) == 0
        && std::memcmp (a.spectralSource.radiance.data(), b.spectralSource.radiance.data(),
                        sizeof (float) * a.spectralSource.radiance.size()) == 0
        && sameBits (a.spectralSource.traceMaximumDistance, b.spectralSource.traceMaximumDistance)
        && HarmonicMIDI::grid::sameDirectionalIrradiance (a.spectralSource.directionalIrradiance,
                                                         b.spectralSource.directionalIrradiance);
}

bool sameMaterial (const Material& a, const Material& b) noexcept
{
    return sameVec3 (a.baseColor, b.baseColor) && sameBits (a.metallic, b.metallic)
        && sameBits (a.roughness, b.roughness) && sameVec3 (a.emission, b.emission)
        && sameBits (a.opacity, b.opacity) && a.alphaMode == b.alphaMode
        && sameBits (a.alphaCutoff, b.alphaCutoff) && sameBits (a.transmission, b.transmission)
        && sameBits (a.ior, b.ior) && sameBits (a.dispersion, b.dispersion) && sameBits (a.abbe, b.abbe)
        && a.transport == b.transport && a.mediumPriority == b.mediumPriority
        && sameBits (a.absorptionDistance, b.absorptionDistance);
}

bool sameLightShape (const LightShape& a, const LightShape& b) noexcept
{
    return sameBits (a.radius, b.radius) && sameBits (a.angularRadius, b.angularRadius);
}

bool sameAreaLight (const AreaLightRecord& a, const AreaLightRecord& b) noexcept
{
    return a.shape == b.shape && sameVec3 (a.position, b.position) && sameBits (a.rotation.x, b.rotation.x)
        && sameBits (a.rotation.y, b.rotation.y) && sameBits (a.rotation.z, b.rotation.z)
        && sameBits (a.rotation.w, b.rotation.w) && sameBits (a.sizeX, b.sizeX) && sameBits (a.sizeY, b.sizeY)
        && sameVec3 (a.color, b.color) && sameBits (a.intensity, b.intensity) && a.twoSided == b.twoSided;
}

bool sameCameraRecord (const SceneCameraRecord& a, const SceneCameraRecord& b) noexcept
{
    return a.id.value == b.id.value && sameTransform (a.transform, b.transform)
        && sameBits (a.verticalFovRadians, b.verticalFovRadians)
        && sameBits (a.nearPlane, b.nearPlane) && sameBits (a.farPlane, b.farPlane);
}

bool sameSceneSetup (const EngineSceneSetup& a, const EngineSceneSetup& b) noexcept
{
    if (a.scene != b.scene || ! sameBits (a.emissionGain, b.emissionGain)
        || a.objectMaterials.size() != b.objectMaterials.size() || a.lights.size() != b.lights.size()
        || a.objectWorlds.size() != b.objectWorlds.size() || a.lightShapes.size() != b.lightShapes.size()
        || a.areaLights.size() != b.areaLights.size())
        return false;
    for (std::size_t i = 0; i < a.lightShapes.size(); ++i)
        if (! sameLightShape (a.lightShapes[i], b.lightShapes[i]))
            return false;
    for (std::size_t i = 0; i < a.areaLights.size(); ++i)
        if (! sameAreaLight (a.areaLights[i], b.areaLights[i]))
            return false;
    for (std::size_t i = 0; i < a.objectMaterials.size(); ++i)
        if (a.objectMaterials[i].objectId != b.objectMaterials[i].objectId
            || ! sameMaterial (a.objectMaterials[i].material, b.objectMaterials[i].material))
            return false;
    for (std::size_t i = 0; i < a.lights.size(); ++i)
        if (! sameLightRecord (a.lights[i], b.lights[i]))
            return false;
    for (std::size_t i = 0; i < a.objectWorlds.size(); ++i)
        if (std::memcmp (a.objectWorlds[i].m.data(), b.objectWorlds[i].m.data(), sizeof (float) * 16) != 0)
            return false;
    return true;
}

bool rasterDrawsSkyBackground (const std::vector<SceneLightRecord>& lights) noexcept
{
    // The raster draws its sky pass when the environment it binds is an
    // analytic sky or an image (backend_opengl.cpp, uSkyBackground).
    return std::any_of (lights.begin(), lights.end(), [] (const SceneLightRecord& light) {
        return light.kind == SceneLightKind::Environment
            && (light.sky.version != 0 || light.environmentImage.version != 0);
    });
}

std::uint16_t halfBits (float value) noexcept
{
    if (std::isnan (value)) return 0;
    const std::uint32_t bits = floatBits (value);
    const auto sign = static_cast<std::uint16_t> ((bits >> 16) & 0x8000u);
    const std::uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x477fe000u)  // 65504, the largest finite half, and above
        return static_cast<std::uint16_t> (sign | 0x7bffu);
    if (magnitude < 0x38800000u)   // below 2^-14: half subnormal or zero
    {
        if (magnitude <= 0x33000000u)  // at most 2^-25 rounds to zero
            return sign;
        const std::uint32_t exponent = magnitude >> 23;
        const std::uint32_t mantissa = (magnitude & 0x7fffffu) | 0x800000u;
        const std::uint32_t shift = 126u - exponent;
        std::uint32_t half = mantissa >> shift;
        const std::uint32_t rest = mantissa & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1u);
        if (rest > halfway || (rest == halfway && (half & 1u) != 0u)) ++half;
        return static_cast<std::uint16_t> (sign | half);
    }
    std::uint32_t half = (magnitude - 0x38000000u) >> 13;
    const std::uint32_t rest = magnitude & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (half & 1u) != 0u)) ++half;
    return static_cast<std::uint16_t> (sign | half);
}

SceneEngine::SceneEngine (unsigned threadCount)
    : tracer_ (threadCount)
{
}

bool SceneEngine::prepare (const EngineSceneSetup& setup, const EngineView& view, PinholeCamera& camera,
                           bool& rebuilt, std::string& error)
{
    rebuilt = false;
    if (setup.scene == nullptr)
    {
        error = "Path Trace requires a scene snapshot";
        return false;
    }
    if (view.width == 0 || view.height == 0)
    {
        error = "Path Trace requires a non-empty image";
        return false;
    }
    if (! setup.objectWorlds.empty() && setup.objectWorlds.size() != setup.scene->objectCount)
    {
        error = "Path Trace object worlds must list every scene object";
        return false;
    }
    if (! makePinholeCamera (view.camera, view.width, view.height, camera))
    {
        error = "Path Trace requires a finite camera with a near plane in front of the far plane";
        return false;
    }
    if (! sceneValid_ || ! sameSceneSetup (setup, cachedSetup_))
    {
        PrepareOptions options;
        options.ingest.objectWorlds = setup.objectWorlds.empty() ? nullptr : setup.objectWorlds.data();
        options.ingest.objectWorldCount = setup.objectWorlds.size();
        options.objectMaterials = setup.objectMaterials;
        options.lightRecords = &setup.lights;
        options.emissionGain = setup.emissionGain;
        options.lightShapes = setup.lightShapes;
        options.areaLights = setup.areaLights;
        options.environmentImageSource = environmentImageSource;
        sceneValid_ = false;
        prepareRenderScene (*setup.scene, scene_, options);
        cachedSetup_ = setup;
        skyVisible_ = rasterDrawsSkyBackground (setup.lights);
        sceneValid_ = true;
        ++sceneBuilds_;
        rebuilt = true;
    }
    tracer_.settings.environmentVisibleToCamera = skyVisible_;
    return true;
}

void SceneEngine::traceDepth (const PinholeCamera& camera)
{
    const std::size_t pixels = static_cast<std::size_t> (camera.width) * camera.height;
    depth_.assign (pixels, 1.0f);
    covered_.assign (pixels, 0);
    const Vec3 forward = rotate (camera.rotation, Vec3 { 0.0f, 0.0f, -1.0f });
    const float range = camera.farPlane - camera.nearPlane;
    const std::uint32_t width = camera.width;
    tracer_.forEachTile (camera.width, camera.height, cancel_, [&] (const Tile& tile, unsigned) {
        for (std::uint32_t y = tile.y0; y < tile.y1; ++y)
            for (std::uint32_t x = tile.x0; x < tile.x1; ++x)
            {
                const Ray ray = camera.generateRay (static_cast<float> (x) + 0.5f, static_cast<float> (y) + 0.5f);
                Hit hit;
                if (! scene_.bvh.intersect (ray, hit))
                    continue;
                const std::size_t index = static_cast<std::size_t> (y) * width + x;
                const float viewDepth = hit.t * dot (ray.direction, forward);
                depth_[index] = std::min (1.0f, std::max (0.0f, (viewDepth - camera.nearPlane) / range));
                covered_[index] = 1;
            }
    });
}

void SceneEngine::resolve (EngineColor color, EngineImage& image) const
{
    const Accumulator& accumulator = tracer_.accumulator();
    const std::uint32_t width = accumulator.width(), height = accumulator.height();
    const std::size_t pixels = static_cast<std::size_t> (width) * height;
    const std::size_t stride = color == EngineColor::LinearRgba16f ? 8 : 4;
    image.width = width;
    image.height = height;
    image.color = color;
    image.colorBytes.assign (pixels * stride, 0);
    image.depth.assign (pixels, 1.0f);
    const bool depthValid = depth_.size() == pixels && covered_.size() == pixels;
    const std::uint16_t halfOne = halfBits (1.0f);
    for (std::uint32_t y = 0; y < height; ++y)
    {
        // The tracer's rows run top first; GL scene attachments bottom first.
        const std::size_t target = static_cast<std::size_t> (height - 1u - y) * width;
        for (std::uint32_t x = 0; x < width; ++x)
        {
            const std::size_t source = static_cast<std::size_t> (y) * width + x;
            const Rgb mean = accumulator.mean (x, y);
            float rgb[3] { mean.r, mean.g, mean.b };
            const bool covered = depthValid && covered_[source] != 0;
            if (! skyVisible_ && ! covered)
                for (int channel = 0; channel < 3; ++channel)
                    rgb[channel] += kRasterBackground[channel];
            if (depthValid) image.depth[target + x] = depth_[source];
            std::uint8_t* out = image.colorBytes.data() + (target + x) * stride;
            if (color == EngineColor::LinearRgba16f)
            {
                const std::uint16_t halves[4] { halfBits (rgb[0]), halfBits (rgb[1]), halfBits (rgb[2]), halfOne };
                std::memcpy (out, halves, sizeof (halves));
            }
            else
            {
                out[0] = unorm8 (rgb[0]);
                out[1] = unorm8 (rgb[1]);
                out[2] = unorm8 (rgb[2]);
                out[3] = 255;
            }
        }
    }
}

bool SceneEngine::renderPreview (const EngineSceneSetup& setup, const EngineView& view, EngineColor color,
                                 EngineImage& image, std::string& error)
{
    // A cancel since the last call (or during it) invalidates the partial
    // pass; clear the token so this call can trace.
    const bool cancelled = cancel_.isCancelled();
    cancel_.reset();
    image = {};
    PinholeCamera camera;
    bool rebuilt = false;
    if (! prepare (setup, view, camera, rebuilt, error))
        return false;
    const bool restart = ! previewValid_ || rebuilt || cancelled
        || view.width != previewView_.width || view.height != previewView_.height
        || ! sameCameraRecord (view.camera, previewView_.camera)
        || ! sameBits (view.timeSeconds, previewView_.timeSeconds);
    if (restart)
    {
        if (++previewGeneration_ == kExportGeneration) ++previewGeneration_;
        previewSamples_ = 0;
        previewView_ = view;
        previewValid_ = true;
        traceDepth (camera);
    }
    image.restarted = restart;
    image.sceneRebuilt = rebuilt;
    const std::uint32_t pass = std::max (1u, samplesPerPass);
    if (previewSamples_ < maximumPreviewSamples)
    {
        const std::uint32_t samples = std::min (pass, maximumPreviewSamples - previewSamples_);
        image.stats = tracer_.renderProgressive (scene_, camera, view.width, view.height, samples,
                                                 previewGeneration_, cancel_);
        if (image.stats.invalidRequest)
        {
            previewValid_ = false;
            error = "Path Trace preview request is invalid";
            return false;
        }
        previewSamples_ += samples;
        image.traced = true;
        // Some pixels may lack this pass; the next call starts over.
        if (image.stats.cancelled) previewValid_ = false;
    }
    image.generation = previewGeneration_;
    image.samplesPerPixel = previewSamples_;
    // Capped and unchanged: the caller's previous image is still exact.
    if (! image.traced && ! restart && color == previewColor_)
    {
        image.unchanged = true;
        error.clear();
        return true;
    }
    previewColor_ = color;
    resolve (color, image);
    error.clear();
    return true;
}

bool SceneEngine::renderExport (const EngineSceneSetup& setup, const EngineView& view, EngineColor color,
                                std::uint32_t samplesPerPixel, EngineImage& image, std::string& error)
{
    cancel_.reset();
    image = {};
    if (samplesPerPixel == 0)
    {
        error = "Path Trace export requires at least one sample per pixel";
        return false;
    }
    PinholeCamera camera;
    bool rebuilt = false;
    if (! prepare (setup, view, camera, rebuilt, error))
        return false;
    // The export reuses the accumulator, so preview starts over afterwards.
    previewValid_ = false;
    traceDepth (camera);
    image.stats = tracer_.renderFixed (scene_, camera, view.width, view.height, samplesPerPixel,
                                       kExportGeneration, cancel_);
    if (image.stats.invalidRequest)
    {
        error = "Path Trace export request is invalid";
        return false;
    }
    if (image.stats.cancelled)
    {
        error = "Path Trace export was cancelled";
        return false;
    }
    resolve (color, image);
    image.generation = kExportGeneration;
    image.samplesPerPixel = samplesPerPixel;
    image.restarted = true;
    image.sceneRebuilt = rebuilt;
    image.traced = true;
    error.clear();
    return true;
}

} // namespace videohelper::pathtrace
