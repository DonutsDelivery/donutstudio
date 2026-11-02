#include "engine.h"
#include "../diffraction_material_admission.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <type_traits>

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

// Test hook: the accumulator's mean as a little-endian PFM, before the clamp, denoiser and
// view transform. Light held in frame layers isn't in it. PFM rows run bottom first.
bool writeExportFilm (const Accumulator& accumulator, const char* path, std::string& error)
{
    std::FILE* file = std::fopen (path, "wb");
    bool written = file != nullptr
        && std::fprintf (file, "PF\n%u %u\n-1.0\n", accumulator.width(), accumulator.height()) > 0;
    for (std::uint32_t y = accumulator.height(); written && y-- > 0;)
        for (std::uint32_t x = 0; written && x < accumulator.width(); ++x)
        {
            const Rgb mean = accumulator.mean (x, y);
            const float rgb[3] { mean.r, mean.g, mean.b };
            written = std::fwrite (rgb, sizeof (float), 3, file) == 3;
        }
    if (file != nullptr && std::fclose (file) != 0)
        written = false;
    if (! written)
        error = std::string ("Path Trace export could not write its film to ") + path;
    return written;
}
} // namespace

// sameLightRecord names every light field, and the spectral source's fields
// inline. A field added to either struct changes these sizes; update the
// comparison with it. The literal 4s are the padding after kind and before the
// eight-byte-aligned environment image.
namespace scene_record_size
{
namespace grid = HarmonicMIDI::grid;
static_assert (sizeof (grid::SceneSpectralSource)
               == sizeof (std::uint32_t) + 2 * sizeof (grid::SceneSpectralSource::radiance) + sizeof (float)
                      + sizeof (grid::SceneSpectralSource::directionalIrradiance));
static_assert (sizeof (grid::SceneLightRecord)
               == sizeof (grid::SceneLightId) + 4 + sizeof (grid::SceneTransform3D) + sizeof (grid::SceneVec3)
                      + 4 * sizeof (float) + sizeof (grid::SceneAnalyticSky)
                      + sizeof (bool) + 3 // environmentCameraVisible, then padding
                      + sizeof (grid::SceneEnvironmentImage) + sizeof (grid::SceneSpectralSource));
} // namespace scene_record_size

bool sameLightRecord (const SceneLightRecord& a, const SceneLightRecord& b) noexcept
{
    return a.id.value == b.id.value && a.kind == b.kind && sameTransform (a.transform, b.transform)
        && sameBits (a.color.x, b.color.x) && sameBits (a.color.y, b.color.y) && sameBits (a.color.z, b.color.z)
        && sameBits (a.intensity, b.intensity) && sameBits (a.range, b.range)
        && sameBits (a.innerConeAngle, b.innerConeAngle) && sameBits (a.outerConeAngle, b.outerConeAngle)
        && HarmonicMIDI::grid::sameAnalyticSky (a.sky, b.sky)
        && a.environmentCameraVisible == b.environmentCameraVisible
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
    return sameBits (a.radius, b.radius) && sameBits (a.angularRadius, b.angularRadius)
        && a.spectralBands == b.spectralBands
        && std::memcmp (a.bands.data(), b.bands.data(), sizeof (float) * a.bands.size()) == 0;
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

const char* sceneContentDifference (const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& a,
                                    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& b) noexcept
{
    if (a == b) return nullptr;
    if (a == nullptr || b == nullptr) return "scene snapshot";
    // The tracer ingests these records and arrays. The Motion history the scene
    // also carries is not one of them: Path Trace refuses Motion outputs, so a
    // prior-frame transform that moved cannot change what it traces.
    if (a->id != b->id) return "scene identity";
    if (a->activeCamera != b->activeCamera) return "active camera";
    if (a->ambientColor.x != b->ambientColor.x || a->ambientColor.y != b->ambientColor.y
        || a->ambientColor.z != b->ambientColor.z)
        return "ambient colour";
    if (a->vertexCount != b->vertexCount) return "vertex count";
    if (a->indexCount != b->indexCount) return "index count";
    if (a->objectCount != b->objectCount) return "object count";
    if (a->materialCount != b->materialCount) return "material count";
    if (a->textureCount != b->textureCount) return "texture count";
    if (a->textureTexelCount != b->textureTexelCount) return "texture texel count";
    if (a->lightCount != b->lightCount) return "light count";
    if (a->cameraCount != b->cameraCount) return "camera count";
    if (a->instancedObject != b->instancedObject) return "instanced object";
    if (a->objectInstanceCount != b->objectInstanceCount) return "instance count";
    // Raw bytes need trivially copyable records. A light holds the environment
    // image's asset strings, whose bytes are pointers into each snapshot, so
    // lights compare by field. Material records still carry padding, which the
    // producers value-initialize so equal materials have equal bytes.
    const auto same = [] (const auto& x, const auto& y, std::size_t count) {
        static_assert (std::is_trivially_copyable_v<std::decay_t<decltype (x[0])>>);
        return std::memcmp (x.data(), y.data(), count * sizeof (std::decay_t<decltype (x[0])>)) == 0;
    };
    if (! same (a->vertices, b->vertices, a->vertexCount)) return "vertices";
    if (! same (a->indices, b->indices, a->indexCount)) return "indices";
    if (! same (a->objects, b->objects, a->objectCount)) return "objects";
    if (! same (a->materials, b->materials, a->materialCount)) return "materials";
    if (! same (a->textures, b->textures, a->textureCount)) return "textures";
    for (std::size_t i = 0; i < a->lightCount; ++i)
        if (! sameLightRecord (a->lights[i], b->lights[i])) return "scene lights";
    if (! same (a->cameras, b->cameras, a->cameraCount)) return "cameras";
    if (! same (a->objectInstances, b->objectInstances, a->objectInstanceCount)) return "object instances";
    if (a->textureTexels.size() != b->textureTexels.size()
        || (! a->textureTexels.empty() && ! same (a->textureTexels, b->textureTexels, a->textureTexels.size())))
        return "texture texels";
    return nullptr;
}

// The first field two scene setups disagree on, or null when they are the same
// scene. A Preview accumulation restarts whenever the setup changes, so a setup
// that rebuilds itself identically every frame would hold the preview at one
// sample per pixel for ever; this names the part that actually moved.
const char* sceneSetupDifference (const EngineSceneSetup& a, const EngineSceneSetup& b) noexcept
{
    if (const char* sceneWhy = sceneContentDifference (a.scene, b.scene))
        return sceneWhy;
    if (! sameBits (a.emissionGain, b.emissionGain))
        return "emission gain";
    if (a.objectMaterials.size() != b.objectMaterials.size())
        return "object material count";
    if (a.objectGratings.size() != b.objectGratings.size())
        return "object grating count";
    if (a.objectFrames.size() != b.objectFrames.size())
        return "object frame count";
    for (std::size_t i = 0; i < a.objectFrames.size(); ++i)
        if (a.objectFrames[i].objectId != b.objectFrames[i].objectId
            || a.objectFrames[i].layer != b.objectFrames[i].layer)
            return "object frame";
    if (a.lights.size() != b.lights.size())
        return "light count";
    if (a.objectWorlds.size() != b.objectWorlds.size())
        return "object world count";
    if (a.lightShapes.size() != b.lightShapes.size())
        return "light shape count";
    if (a.areaLights.size() != b.areaLights.size())
        return "area light count";
    for (std::size_t i = 0; i < a.lightShapes.size(); ++i)
        if (! sameLightShape (a.lightShapes[i], b.lightShapes[i]))
            return "light shape";
    for (std::size_t i = 0; i < a.areaLights.size(); ++i)
        if (! sameAreaLight (a.areaLights[i], b.areaLights[i]))
            return "area light";
    for (std::size_t i = 0; i < a.objectMaterials.size(); ++i)
        if (a.objectMaterials[i].objectId != b.objectMaterials[i].objectId
            || ! sameMaterial (a.objectMaterials[i].material, b.objectMaterials[i].material))
            return "object material";
    for (std::size_t i = 0; i < a.objectGratings.size(); ++i)
    {
        const auto& left = a.objectGratings[i];
        const auto& right = b.objectGratings[i];
        if (left.objectId != right.objectId || (left.material == nullptr) != (right.material == nullptr)
            || left.field != right.field)
            return "object grating";
        if (left.material != right.material && left.material != nullptr
            && left.material->structuralDigest() != right.material->structuralDigest())
            return "object grating";
    }
    for (std::size_t i = 0; i < a.lights.size(); ++i)
        if (! sameLightRecord (a.lights[i], b.lights[i]))
            return "light record";
    for (std::size_t i = 0; i < a.objectWorlds.size(); ++i)
        if (std::memcmp (a.objectWorlds[i].m.data(), b.objectWorlds[i].m.data(), sizeof (float) * 16) != 0)
            return "object world";
    return nullptr;
}

bool sameSceneSetup (const EngineSceneSetup& a, const EngineSceneSetup& b) noexcept
{
    return sceneSetupDifference (a, b) == nullptr;
}

bool rasterDrawsSkyBackground (const std::vector<SceneLightRecord>& lights) noexcept
{
    // The raster draws its sky pass when the environment it binds is an
    // analytic sky or an image (backend_opengl.cpp, uSkyBackground), unless
    // the Environment Light hides itself from the camera.
    if (! HarmonicMIDI::grid::sceneEnvironmentVisibleToCamera (lights.data(), lights.size()))
        return false;
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

const char* engineRenderModeName (EngineRenderMode mode) noexcept
{
    switch (mode)
    {
        case EngineRenderMode::PlaybackFixed: return "playback-fixed";
        case EngineRenderMode::Export: return "export";
        case EngineRenderMode::Converging: break;
    }
    return "converging";
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
        options.objectGratings = setup.objectGratings;
        options.objectFrames = setup.objectFrames;
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
    const float* background = color == EngineColor::LinearRgba16f ? kLinearRasterBackground : kRasterBackground;
    for (std::uint32_t y = 0; y < height; ++y)
    {
        // The tracer's rows run top first; GL scene attachments bottom first.
        const std::size_t target = static_cast<std::size_t> (height - 1u - y) * width;
        for (std::uint32_t x = 0; x < width; ++x)
        {
            const std::size_t source = static_cast<std::size_t> (y) * width + x;
            const Rgb mean = historyShown_ ? history_[source] : accumulator.mean (x, y);
            float rgb[3] { mean.r, mean.g, mean.b };
            const bool covered = depthValid && covered_[source] != 0;
            if (! skyVisible_ && ! covered)
                for (int channel = 0; channel < 3; ++channel)
                    rgb[channel] += background[channel];
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
    image.frameLayers.resize (tracer_.frameLayers().size());
    for (std::size_t index = 0; index < image.frameLayers.size(); ++index)
    {
        const FrameLayerFilm& film = tracer_.frameLayers()[index];
        EngineFrameLayer& layer = image.frameLayers[index];
        layer.lightBytes.assign (pixels * 8, 0);
        layer.uvCoverage.assign (pixels * 3, 0.0f);
        for (std::uint32_t y = 0; y < height; ++y)
        {
            const std::size_t target = static_cast<std::size_t> (height - 1u - y) * width;
            for (std::uint32_t x = 0; x < width; ++x)
            {
                const FrameLayerFilm::Pixel& pixel = film.pixel (x, y);
                const std::uint32_t samples = accumulator.sampleCount (x, y);
                const float share = samples != 0 ? 1.0f / static_cast<float> (samples) : 0.0f;
                const std::uint16_t halves[4] { halfBits (pixel.r * share), halfBits (pixel.g * share),
                                                halfBits (pixel.b * share), halfOne };
                std::memcpy (layer.lightBytes.data() + (target + x) * 8, halves, sizeof (halves));
                if (pixel.hits == 0)
                    continue;
                const float perHit = 1.0f / static_cast<float> (pixel.hits);
                float* uv = layer.uvCoverage.data() + (target + x) * 3;
                uv[0] = pixel.u * perHit;
                uv[1] = pixel.v * perHit;
                uv[2] = static_cast<float> (pixel.hits) * share;
            }
        }
    }
}

void SceneEngine::updatePlaybackHistory (bool blend)
{
    const Accumulator& accumulator = tracer_.accumulator();
    const std::uint32_t width = accumulator.width(), height = accumulator.height();
    const std::size_t pixels = static_cast<std::size_t> (width) * height;
    historyCurrent_.resize (pixels);
    history_.resize (pixels);
    const CancellationToken neverCancelled;
    tracer_.forEachTile (width, height, neverCancelled, [&] (const Tile& tile, unsigned) {
        for (std::uint32_t y = tile.y0; y < tile.y1; ++y)
            for (std::uint32_t x = tile.x0; x < tile.x1; ++x)
                historyCurrent_[static_cast<std::size_t> (y) * width + x] = accumulator.mean (x, y);
    });
    if (blend)
    {
        const float weight = playbackCurrentWeight;
        // Each pixel reads only its own history, so the blend writes in place.
        tracer_.forEachTile (width, height, neverCancelled, [&] (const Tile& tile, unsigned) {
            for (std::uint32_t y = tile.y0; y < tile.y1; ++y)
                for (std::uint32_t x = tile.x0; x < tile.x1; ++x)
                {
                    const std::size_t index = static_cast<std::size_t> (y) * width + x;
                    const Rgb current = historyCurrent_[index];
                    const float depth = depth_[index], previousDepth = historyDepth_[index];
                    if (covered_[index] != historyCovered_[index]
                        || std::abs (depth - previousDepth)
                               > kPlaybackHistoryDepthTolerance * std::max (depth, previousDepth))
                    {
                        history_[index] = current;
                        continue;
                    }
                    float sum[3] {}, squares[3] {};
                    float count = 0.0f;
                    for (std::uint32_t ny = y > 0 ? y - 1 : y; ny <= y + 1 && ny < height; ++ny)
                        for (std::uint32_t nx = x > 0 ? x - 1 : x; nx <= x + 1 && nx < width; ++nx)
                        {
                            const Rgb neighbour = historyCurrent_[static_cast<std::size_t> (ny) * width + nx];
                            const float channels[3] { neighbour.r, neighbour.g, neighbour.b };
                            for (int channel = 0; channel < 3; ++channel)
                            {
                                sum[channel] += channels[channel];
                                squares[channel] += channels[channel] * channels[channel];
                            }
                            count += 1.0f;
                        }
                    const auto blended = [&] (float previous, float now, int channel) {
                        const float mean = sum[channel] / count;
                        const float spread = kPlaybackHistoryClipSigma
                            * std::sqrt (std::max (0.0f, squares[channel] / count - mean * mean));
                        const float clipped = std::min (std::max (previous, mean - spread), mean + spread);
                        return clipped + (now - clipped) * weight;
                    };
                    Rgb& pixel = history_[index];
                    pixel = { blended (pixel.r, current.r, 0), blended (pixel.g, current.g, 1),
                              blended (pixel.b, current.b, 2) };
                }
        });
    }
    else
    {
        history_ = historyCurrent_;
    }
    historyDepth_ = depth_;
    historyCovered_ = covered_;
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
    // Captured before prepare, which stores the incoming setup as the cached one.
    const char* setupDifference = sceneSetupDifference (setup, cachedSetup_);
    if (! prepare (setup, view, camera, rebuilt, error))
        return false;
    // A scene with frame layers never reaches the device, so it has nothing to wait for.
    // What prepare just found (a rebuild, a cancel) still applies when the device is ready.
    if (deviceReady_ && scene_.frameBindings.empty() && ! deviceReady_())
    {
        previewValid_ = false;
        image.pending = true;
        error.clear();
        return true;
    }
    const bool timeMoved = previewValid_ && ! sameBits (view.timeSeconds, previewView_.timeSeconds);
    const bool playbackFrame = playbackActive || timeMoved;
    const bool restart = ! previewValid_ || rebuilt || cancelled
        || view.width != previewView_.width || view.height != previewView_.height
        || ! sameCameraRecord (view.camera, previewView_.camera)
        || timeMoved
        || previewSeededLikeExport != previewSeeded_
        || tracer_.settings.maxDepth != previewMaxDepth_
        // A lowered target (Export Samples or Preview Samples edited while the preview is
        // live) restarts, so the settled image has exactly the target's samples, as the
        // export of the same count does.
        || previewSamples_ > maximumPreviewSamples;
    // History has no reprojection and does not outlive playback. A scene change keeps
    // it, because each pixel rejects what no longer holds.
    if (! playbackFrame || ! previewValid_
        || view.width != previewView_.width || view.height != previewView_.height
        || ! sameCameraRecord (view.camera, previewView_.camera)
        || previewSeededLikeExport != previewSeeded_)
        historyValid_ = false;
    const bool historyWasShown = historyShown_;
    if (restart || ! playbackFrame)
        historyShown_ = false;
    if (restart)
    {
        if (++previewGeneration_ == kExportGeneration) ++previewGeneration_;
        previewSamples_ = 0;
        previewView_ = view;
        previewValid_ = true;
        previewSeeded_ = previewSeededLikeExport;
        previewMaxDepth_ = tracer_.settings.maxDepth;
        traceDepth (camera);
    }
    image.restarted = restart;
    image.sceneRebuilt = rebuilt;
    renderMode_ = playbackFrame ? EngineRenderMode::PlaybackFixed : EngineRenderMode::Converging;
    image.mode = renderMode_;
    const std::uint32_t goal = playbackFrame ? std::min (std::max (1u, playbackSamples), maximumPreviewSamples)
                                             : maximumPreviewSamples;
    if (previewSamples_ < goal)
    {
        const std::uint32_t samples = playbackFrame ? goal - previewSamples_
                                                    : std::min (std::max (1u, samplesPerPass), goal - previewSamples_);
        image.stats = tracer_.renderProgressive (scene_, camera, view.width, view.height, samples,
                                                 previewGeneration_, cancel_,
                                                 previewSeeded_ ? kExportGeneration : 0);
    if (image.stats.deviceFailed)
    {
        error = "Path Trace device failed: " + tracer_.deviceFailure();
        return false;
    }
        if (image.stats.invalidRequest)
        {
            previewValid_ = false;
            error = "Path Trace preview request is invalid";
            return false;
        }
        previewSamples_ += samples;
        image.traced = true;
        // Some pixels may lack this pass; the next call starts over.
        if (image.stats.cancelled)
        {
            previewValid_ = false;
            historyValid_ = false;
            historyShown_ = false;
        }
        else if (playbackFrame && playbackCurrentWeight < 1.0f)
        {
            // Only a new frame blends. More samples on the same frame replace it, as
            // the history would count that frame twice.
            const bool blend = historyValid_ && restart;
            try
            {
                updatePlaybackHistory (blend);
            }
            catch (const std::bad_alloc& failure)
            {
                previewValid_ = false;
                historyValid_ = false;
                const std::size_t pixels = static_cast<std::size_t> (view.width) * view.height;
                error = "Path Trace temporal history could not allocate "
                    + std::to_string (pixels * (2 * sizeof (Rgb) + sizeof (float) + sizeof (std::uint8_t)))
                    + " bytes for a " + std::to_string (view.width) + "x" + std::to_string (view.height)
                    + " film: " + failure.what();
                return false;
            }
            historyValid_ = true;
            historyShown_ = blend;
        }
    }
    image.temporalDenoise = historyShown_;
    image.generation = previewGeneration_;
    image.samplesPerPixel = previewSamples_;
    // Which condition restarts a Preview accumulation, at most once a second: a
    // viewport that keeps asking for passes while the accumulation restarts each
    // time would spin at full frame rate and never converge, which no other
    // telemetry can name.
    {
        using Clock = std::chrono::steady_clock;
        static Clock::time_point last {};
        const auto now = Clock::now();
        if (now - last >= std::chrono::seconds (1))
        {
            last = now;
            std::fprintf (stderr,
                "[pt] preview mode=%s samples=%u/%u gen=%llu restart=%d rebuilt=%d cancelled=%d traced=%d"
                " unchanged=%d setup-diff=%s temporal=%d\n",
                engineRenderModeName (renderMode_), previewSamples_, goal,
                static_cast<unsigned long long> (previewGeneration_), restart ? 1 : 0,
                rebuilt ? 1 : 0, cancelled ? 1 : 0, image.traced ? 1 : 0,
                (! image.traced && ! restart) ? 1 : 0,
                setupDifference != nullptr ? setupDifference : "none", historyShown_ ? 1 : 0);
        }
    }
    // Capped and unchanged: the caller's previous image is still exact, unless it
    // was a blend and playback has stopped.
    if (! image.traced && ! restart && color == previewColor_ && historyShown_ == historyWasShown)
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
    // The export reuses the accumulator, so preview starts over afterwards. History
    // stops here too, and the export resolves the accumulator alone.
    previewValid_ = false;
    historyValid_ = false;
    historyShown_ = false;
    renderMode_ = EngineRenderMode::Export;
    image.mode = renderMode_;
    traceDepth (camera);
    image.stats = tracer_.renderFixed (scene_, camera, view.width, view.height, samplesPerPixel,
                                       kExportGeneration, cancel_);
    if (image.stats.deviceFailed)
    {
        error = "Path Trace device failed: " + tracer_.deviceFailure();
        return false;
    }
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
    if (const char* filmPath = std::getenv ("ARBIT_PATHTRACE_EXPORT_FILM"); filmPath != nullptr && *filmPath != '\0')
        if (! writeExportFilm (tracer_.accumulator(), filmPath, error))
            return false;
    resolve (color, image);
    image.generation = kExportGeneration;
    image.samplesPerPixel = samplesPerPixel;
    image.restarted = true;
    image.sceneRebuilt = rebuilt;
    image.traced = true;
    // At most once a second, like the preview line: an export calls this per frame.
    {
        using Clock = std::chrono::steady_clock;
        static Clock::time_point last {};
        const auto now = Clock::now();
        if (now - last >= std::chrono::seconds (1))
        {
            last = now;
            std::fprintf (stderr, "[pt] export mode=%s samples=%u\n", engineRenderModeName (renderMode_),
                          samplesPerPixel);
        }
    }
    error.clear();
    return true;
}

} // namespace videohelper::pathtrace
