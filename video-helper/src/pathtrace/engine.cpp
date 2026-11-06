#include "engine.h"
#include "../diffraction_material_admission.h"
#include "devices/device_cpu.h"

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
                      + sizeof (grid::SceneCameraBackground)
                      + sizeof (grid::SceneEnvironmentImage) + sizeof (grid::SceneSpectralSource)
                      + 5 * sizeof (float) + 2 + 2); // size and area fields, shape, visibility, padding
} // namespace scene_record_size

bool sameLightRecord (const SceneLightRecord& a, const SceneLightRecord& b) noexcept
{
    return a.id.value == b.id.value && a.kind == b.kind && sameTransform (a.transform, b.transform)
        && sameBits (a.color.x, b.color.x) && sameBits (a.color.y, b.color.y) && sameBits (a.color.z, b.color.z)
        && sameBits (a.intensity, b.intensity) && sameBits (a.range, b.range)
        && sameBits (a.innerConeAngle, b.innerConeAngle) && sameBits (a.outerConeAngle, b.outerConeAngle)
        && HarmonicMIDI::grid::sameAnalyticSky (a.sky, b.sky)
        && a.environmentCameraVisible == b.environmentCameraVisible
        && HarmonicMIDI::grid::sameCameraBackground (a.cameraBackground, b.cameraBackground)
        && HarmonicMIDI::grid::sameEnvironmentImage (a.environmentImage, b.environmentImage)
        && a.spectralSource.version == b.spectralSource.version
        && std::memcmp (a.spectralSource.wavelengthsNanometres.data(), b.spectralSource.wavelengthsNanometres.data(),
                        sizeof (float) * a.spectralSource.wavelengthsNanometres.size()) == 0
        && std::memcmp (a.spectralSource.radiance.data(), b.spectralSource.radiance.data(),
                        sizeof (float) * a.spectralSource.radiance.size()) == 0
        && sameBits (a.spectralSource.traceMaximumDistance, b.spectralSource.traceMaximumDistance)
        && HarmonicMIDI::grid::sameDirectionalIrradiance (a.spectralSource.directionalIrradiance,
                                                         b.spectralSource.directionalIrradiance)
        && HarmonicMIDI::grid::sameLightShapeAndVisibility (a, b);
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
        && sameBits (a.nearPlane, b.nearPlane) && sameBits (a.farPlane, b.farPlane)
        && sameBits (a.orthographicHeight, b.orthographicHeight)
        && HarmonicMIDI::grid::sameCameraLens (a, b);
}

// The cameras of two views agree, including the one a motion blur shutter opens from.
bool sameViewCamera (const EngineView& a, const EngineView& b) noexcept
{
    if (! sameCameraRecord (a.camera, b.camera) || a.previousCamera.has_value() != b.previousCamera.has_value())
        return false;
    return ! a.previousCamera || sameCameraRecord (*a.previousCamera, *b.previousCamera);
}

const char* sceneContentDifference (const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& a,
                                    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& b) noexcept
{
    if (a == b) return nullptr;
    if (a == nullptr || b == nullptr) return "scene snapshot";
    // The tracer ingests these records and arrays. The previous transforms Motion
    // reads are not among them: a moved time restarts the preview, and the first-hit
    // pass then reads them from the call's own snapshot (traceDepth).
    if (a->id != b->id) return "scene identity";
    if (a->activeCamera != b->activeCamera) return "active camera";
    if (a->ambientColor.x != b->ambientColor.x || a->ambientColor.y != b->ambientColor.y
        || a->ambientColor.z != b->ambientColor.z)
        return "ambient colour";
    if (! HarmonicMIDI::grid::sameSceneVolume (a->worldVolume, b->worldVolume)) return "world volume";
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
    if (! sameBits (a.shutterOpen, b.shutterOpen) || ! sameBits (a.shutterClose, b.shutterClose))
        return "motion blur shutter";
    if (a.previousObjectWorlds.size() != b.previousObjectWorlds.size()
        || a.previousInstances.size() != b.previousInstances.size())
        return "previous frame count";
    for (std::size_t i = 0; i < a.previousObjectWorlds.size(); ++i)
        if (std::memcmp (a.previousObjectWorlds[i].m.data(), b.previousObjectWorlds[i].m.data(), sizeof (float) * 16) != 0)
            return "previous object world";
    for (std::size_t i = 0; i < a.previousInstances.size(); ++i)
        if (std::memcmp (&a.previousInstances[i], &b.previousInstances[i], sizeof (a.previousInstances[i])) != 0)
            return "previous instance";
    if (a.previousPositions != b.previousPositions
        && (a.previousPositions == nullptr || b.previousPositions == nullptr
            || a.previousPositions->size() != b.previousPositions->size()
            || std::memcmp (a.previousPositions->data(), b.previousPositions->data(),
                            sizeof (HarmonicMIDI::grid::SceneVec3) * a.previousPositions->size()) != 0))
        return "previous positions";
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
    if (view.previousCamera)
        setCameraShutter (camera, *view.previousCamera, setup.shutterOpen, setup.shutterClose);
    if (! sceneValid_ || ! sameSceneSetup (setup, cachedSetup_))
    {
        PrepareOptions options;
        options.ingest.objectWorlds = setup.objectWorlds.empty() ? nullptr : setup.objectWorlds.data();
        options.ingest.objectWorldCount = setup.objectWorlds.size();
        options.ingest.shutterOpen = setup.shutterOpen;
        options.ingest.shutterClose = setup.shutterClose;
        options.ingest.previousObjectWorlds = setup.previousObjectWorlds.empty() ? nullptr : setup.previousObjectWorlds.data();
        options.ingest.previousObjectWorldCount = setup.previousObjectWorlds.size();
        options.ingest.previousInstances = setup.previousInstances.empty() ? nullptr : setup.previousInstances.data();
        options.ingest.previousInstanceCount = setup.previousInstances.size();
        if (setup.previousPositions != nullptr)
        {
            options.ingest.previousPositions = setup.previousPositions->data();
            options.ingest.previousPositionCount = setup.previousPositions->size();
        }
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
        cameraBackground_ = HarmonicMIDI::grid::sceneCameraBackground (setup.lights.data(), setup.lights.size());
        sceneValid_ = true;
        ++sceneBuilds_;
        rebuilt = true;
    }
    tracer_.settings.environmentVisibleToCamera = skyVisible_;
    return true;
}

bool SceneEngine::timedPrepare (const EngineSceneSetup& setup, const EngineView& view, PinholeCamera& camera,
                                bool& rebuilt, FramePhaseTimes& phases, std::string& error)
{
    const auto start = PhaseClock::now();
    if (! prepare (setup, view, camera, rebuilt, error))
        return false;
    const double bvh = rebuilt ? scene_.bvhBuildMs : 0.0;
    phases.set (FramePhase::SceneLower, std::max (0.0, millisecondsSince (start) - bvh));
    phases.set (FramePhase::CpuBvhBuild, bvh);
    return true;
}

void SceneEngine::traceDepth (const PinholeCamera& camera, const EngineView& engineView, const EngineSceneSetup& setup)
{
    const std::size_t pixels = static_cast<std::size_t> (camera.width) * camera.height;
    depth_.assign (pixels, 1.0f);
    covered_.assign (pixels, 0);
    outputs_.assign (pixels, FirstHitOutput {});
    const Vec3 forward = rotate (camera.rotation, Vec3 { 0.0f, 0.0f, -1.0f });
    const float range = camera.farPlane - camera.nearPlane;
    const std::uint32_t width = camera.width;
    // The render outputs (render_outputs.h) read the hit the depth ray found: the
    // integrator's own first-hit normal, the hit material's emission and the
    // triangle's ids, and the hit point carried back to its previous transform.
    const device::cpu::SceneView view = device::cpu::makeSceneView (scene_);
    const auto* scene = setup.scene.get();
    PinholeCamera previousCamera = camera;
    if (! engineView.previousCamera.has_value()
        || ! makePinholeCamera (*engineView.previousCamera, camera.width, camera.height, previousCamera))
        previousCamera = camera;
    const bool cameraMoved = ! sameView (camera, previousCamera);
    const bool motion = scene != nullptr && (cameraMoved || sceneHasMotionHistory (*scene));
    std::vector<Mat4> currentWorlds, previousWorlds;
    if (motion)
    {
        const std::size_t objectCount = scene->objectCount;
        const bool overrides = setup.objectWorlds.size() == objectCount;
        currentWorlds.resize (objectCount);
        previousWorlds.resize (objectCount);
        for (std::size_t object = 0; object < objectCount; ++object)
        {
            currentWorlds[object] = overrides ? setup.objectWorlds[object] : objectWorldMatrix (*scene, object);
            previousWorlds[object] = previousObjectWorld (*scene, object, currentWorlds[object]);
        }
    }
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
                FirstHitOutput& output = outputs_[index];
                output.depth = depth_[index];
                output.mask = 255;
                const Vec3 normal = device::cpu::firstHitWorldNormal (view, ray);
                output.normal = { normal.x, normal.y, normal.z };
                const Material& material = scene_.materialOf (hit.triangle);
                output.emission = { material.emission.x, material.emission.y, material.emission.z };
                if (hit.triangle >= scene_.geometry.triangles.size())
                    continue;
                const TriangleAttributes& attributes = scene_.geometry.triangles[hit.triangle];
                output.materialId = attributes.materialId;
                output.objectId = attributes.objectId;
                if (! motion || attributes.objectIndex >= currentWorlds.size())
                    continue;
                Mat4 current = currentWorlds[attributes.objectIndex];
                Mat4 previous = previousWorlds[attributes.objectIndex];
                if (attributes.instanceIndex != kNoInstance && attributes.instanceIndex < scene->objectInstanceCount)
                {
                    current = current * transformMatrix (scene->objectInstances[attributes.instanceIndex]);
                    previous = previous * previousInstanceMatrix (*scene, attributes.instanceIndex);
                }
                output.motion = pixelMotion (camera, previousCamera, ray.origin + ray.direction * hit.t, current, previous);
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
    image.albedo.assign (pixels * 3, 0.0f);
    image.normal.assign (pixels * 3, 0.0f);
    image.normalPass.assign (pixels * 4, 0);
    image.emissionPass.assign (pixels * 4, 0);
    image.maskPass.assign (pixels, 0);
    image.materialIdPass.assign (pixels, 0);
    image.objectIdPass.assign (pixels, 0);
    image.motionPass.assign (pixels * 2, 0);
    const bool depthValid = depth_.size() == pixels && covered_.size() == pixels;
    const bool outputsValid = outputs_.size() == pixels;
    const std::uint16_t halfOne = halfBits (1.0f);
    const float* background = color == EngineColor::LinearRgba16f ? kLinearRasterBackground : kRasterBackground;
    // A camera background replaces the stage colour: a colour (sRGB encoded for the
    // 8-bit image, as the stage colour is) or zero alpha.
    using BackgroundMode = HarmonicMIDI::grid::SceneCameraBackgroundMode;
    const auto encode = [&] (float linear) {
        if (color == EngineColor::LinearRgba16f) return linear;
        const float c = std::max (0.0f, linear);
        return c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow (c, 1.0f / 2.4f) - 0.055f;
    };
    const float colourBackground[3] { encode (cameraBackground_.color.x), encode (cameraBackground_.color.y),
                                      encode (cameraBackground_.color.z) };
    if (cameraBackground_.mode == BackgroundMode::Colour)
        background = colourBackground;
    const bool transparentBackground = cameraBackground_.mode == BackgroundMode::Transparent;
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
            const bool backdrop = ! skyVisible_ && ! covered;
            if (backdrop && ! transparentBackground)
                for (int channel = 0; channel < 3; ++channel)
                    rgb[channel] += background[channel];
            const bool clear = backdrop && transparentBackground;
            if (depthValid) image.depth[target + x] = depth_[source];
            if (outputsValid && outputs_[source].mask != 0)
            {
                const FirstHitOutput& output = outputs_[source];
                std::uint16_t* const normalOut = image.normalPass.data() + (target + x) * 4;
                std::uint16_t* const emissionOut = image.emissionPass.data() + (target + x) * 4;
                for (int channel = 0; channel < 3; ++channel)
                {
                    normalOut[channel] = halfBits (output.normal[static_cast<std::size_t> (channel)]);
                    emissionOut[channel] = halfBits (output.emission[static_cast<std::size_t> (channel)]);
                }
                normalOut[3] = emissionOut[3] = halfOne;
                image.maskPass[target + x] = output.mask;
                image.materialIdPass[target + x] = output.materialId;
                image.objectIdPass[target + x] = output.objectId;
                image.motionPass[(target + x) * 2] = halfBits (output.motion[0]);
                image.motionPass[(target + x) * 2 + 1] = halfBits (output.motion[1]);
            }
            Rgb albedo, normal;
            accumulator.guideMean (x, y, albedo, normal);
            float* const guideAlbedo = image.albedo.data() + (target + x) * 3;
            float* const guideNormal = image.normal.data() + (target + x) * 3;
            guideAlbedo[0] = albedo.r; guideAlbedo[1] = albedo.g; guideAlbedo[2] = albedo.b;
            guideNormal[0] = normal.r; guideNormal[1] = normal.g; guideNormal[2] = normal.b;
            std::uint8_t* out = image.colorBytes.data() + (target + x) * stride;
            if (color == EngineColor::LinearRgba16f)
            {
                const std::uint16_t halves[4] { halfBits (rgb[0]), halfBits (rgb[1]), halfBits (rgb[2]),
                                                clear ? halfBits (0.0f) : halfOne };
                std::memcpy (out, halves, sizeof (halves));
            }
            else
            {
                out[0] = unorm8 (rgb[0]);
                out[1] = unorm8 (rgb[1]);
                out[2] = unorm8 (rgb[2]);
                out[3] = clear ? 0 : 255;
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

namespace
{
// The pass's device phases plus the engine's own. A device that traces the CPU BVH
// (the CPU, and the compute devices) leaves AccelBuild to this: that BVH's build,
// when this call made a new scene.
void notePassPhases (FramePhaseTimes& passPhases, const FramePhaseTimes& enginePhases, bool rebuilt, double bvhBuildMs)
{
    FramePhaseTimes merged = enginePhases;
    merged.merge (passPhases);
    if (! merged.has (FramePhase::AccelBuild) && merged.absentReason (FramePhase::AccelBuild) == nullptr)
    {
        merged.set (FramePhase::AccelBuild, rebuilt ? bvhBuildMs : 0.0);
        merged.accelBuild = rebuilt ? "build" : "reused";
    }
    passPhases = merged;
}
} // namespace

bool SceneEngine::renderPreview (const EngineSceneSetup& setup, const EngineView& view, EngineColor color,
                                 EngineImage& image, std::string& error)
{
    // A cancel since the last call (or during it) invalidates the partial
    // pass; clear the token so this call can trace.
    const bool cancelled = cancel_.isCancelled();
    cancel_.reset();
    lastPassObsolete_ = false;
    image = {};
    PinholeCamera camera;
    bool rebuilt = false;
    // Captured before prepare, which stores the incoming setup as the cached one.
    const char* setupDifference = sceneSetupDifference (setup, cachedSetup_);
    FramePhaseTimes phases;
    if (! timedPrepare (setup, view, camera, rebuilt, phases, error))
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
        || ! sameViewCamera (view, previewView_)
        || timeMoved
        || previewSeededLikeExport != previewSeeded_
        || ! sameLightPathSettings (tracer_.settings, previewLightPaths_)
        // A lowered target (Export Samples or Preview Samples edited while the preview is
        // live) restarts, so the settled image has exactly the target's samples, as the
        // export of the same count does.
        || previewSamples_ > maximumPreviewSamples;
    // History has no reprojection and does not outlive playback. A scene change keeps
    // it, because each pixel rejects what no longer holds.
    if (! playbackFrame || ! previewValid_
        || view.width != previewView_.width || view.height != previewView_.height
        || ! sameViewCamera (view, previewView_)
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
        previewLightPaths_ = tracer_.settings;
        const auto depthStart = PhaseClock::now();
        traceDepth (camera, view, setup);
        phases.set (FramePhase::DepthPass, millisecondsSince (depthStart));
    }
    else
        phases.set (FramePhase::DepthPass, 0.0);
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
        notePassPhases (image.stats.phases, phases, rebuilt, scene_.bvhBuildMs);
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
            lastPassObsolete_ = cancel_.obsolete();
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
    const auto resolveStart = PhaseClock::now();
    resolve (color, image);
    if (! image.traced)
        image.stats.phases = phases;
    image.stats.phases.set (FramePhase::Resolve, millisecondsSince (resolveStart));
    error.clear();
    return true;
}

bool SceneEngine::renderExport (const EngineSceneSetup& setup, const EngineView& view, EngineColor color,
                                std::uint32_t samplesPerPixel, EngineImage& image, std::string& error)
{
    cancel_.reset();
    cancel_.watch (nullptr, 0);  // an edit during an export does not stop it
    image = {};
    if (samplesPerPixel == 0)
    {
        error = "Path Trace export requires at least one sample per pixel";
        return false;
    }
    PinholeCamera camera;
    bool rebuilt = false;
    FramePhaseTimes phases;
    if (! timedPrepare (setup, view, camera, rebuilt, phases, error))
        return false;
    // The export reuses the accumulator, so preview starts over afterwards. History
    // stops here too, and the export resolves the accumulator alone.
    previewValid_ = false;
    historyValid_ = false;
    historyShown_ = false;
    renderMode_ = EngineRenderMode::Export;
    image.mode = renderMode_;
    const auto depthStart = PhaseClock::now();
    traceDepth (camera, view, setup);
    phases.set (FramePhase::DepthPass, millisecondsSince (depthStart));
    image.stats = tracer_.renderFixed (scene_, camera, view.width, view.height, samplesPerPixel,
                                       kExportGeneration, cancel_);
    notePassPhases (image.stats.phases, phases, rebuilt, scene_.bvhBuildMs);
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
    const auto resolveStart = PhaseClock::now();
    resolve (color, image);
    image.stats.phases.set (FramePhase::Resolve, millisecondsSince (resolveStart));
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
