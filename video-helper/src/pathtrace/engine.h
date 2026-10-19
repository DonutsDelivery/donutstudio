#pragma once

// The CPU path tracer as a render engine for the 3D Render node (Plan 3,
// PT-C1). It turns one evaluated scene, camera and image size into the image
// the raster engine would hand the compositor, and owns the caches that make
// a progressive viewport cheap:
//
// - The RenderScene and its BVH are rebuilt only when the scene setup changes:
//   a new snapshot, other object materials, lights, object world matrices or
//   emission gain. A camera, size or time change keeps them.
// - Preview accumulates. Every preview call renders one pass of
//   samplesPerPass samples per pixel on the tracer's pool and returns the mean
//   so far. A change of scene, camera, size or time starts a new generation,
//   which clears the accumulator. After maximumPreviewSamples the image stops
//   changing and preview calls return it without tracing.
// - Export clears and renders exactly the requested samples per pixel with a
//   fixed generation, so the same request always gives the same bytes.
//
// Output matches the raster's compositor contract (backend_opengl.cpp, the
// fixture scene draw): Color is clamped linear RGBA8 with no transfer for SDR
// preview, or linear sRGB RGBA16F for the HDR working representation. Alpha
// is 1. Depth is R32F, (view depth - near) / (far - near) clamped to [0, 1],
// 1 where the centre ray of a pixel hits nothing. Rows are bottom first, like
// the GL scene attachments. Where the raster draws no sky, a pixel whose centre
// ray misses shows the raster's clear colour (7, 10, 18) / 255.
//
// Nothing here touches a GPU. The helper uploads the bytes (cpu_scene_frame.h).

#include "integrator.h"
#include "thread_pool.h"
#include "../../../shared/Visual3DScene.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace videohelper::pathtrace
{

// Everything that, when it changes, rebuilds the RenderScene and the BVH.
struct EngineSceneSetup
{
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
    // Replace the record material of these objects (PrepareOptions::objectMaterials).
    std::vector<ObjectMaterialOverride> objectMaterials;
    // The light records the raster reads after runtime overrides. The scene's
    // own records are not used.
    std::vector<HarmonicMIDI::grid::SceneLightRecord> lights;
    // Empty, or one column-major world matrix per scene object (runtime object
    // transforms already applied).
    std::vector<Mat4> objectWorlds;
    float emissionGain = 1.0f;
    // Sizes of `lights` by index and extra rect and disk lights
    // (PrepareOptions). Empty keeps the delta lights.
    std::vector<LightShape> lightShapes;
    std::vector<AreaLightRecord> areaLights;
};

// Everything that, when it changes, restarts accumulation.
struct EngineView
{
    HarmonicMIDI::grid::SceneCameraRecord camera {};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    float timeSeconds = 0.0f;
};

enum class EngineColor : std::uint8_t
{
    DisplayRgba8 = 0,   // clamped linear values, no transfer, 4 bytes per pixel
    LinearRgba16f = 1   // IEEE half floats, linear sRGB, 8 bytes per pixel
};

struct EngineImage
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    EngineColor color = EngineColor::DisplayRgba8;
    std::vector<std::uint8_t> colorBytes;  // tight rows, bottom row first
    std::vector<float> depth;              // width * height, bottom row first
    std::uint64_t generation = 0;          // accumulation this image belongs to
    std::uint32_t samplesPerPixel = 0;     // accumulated in this generation
    bool restarted = false;                // this call started a new generation
    bool sceneRebuilt = false;             // this call rebuilt the RenderScene and BVH
    bool traced = false;                   // this call added samples
    // Preview only: no samples were added and nothing restarted, so the
    // previous image is still exact. colorBytes and depth are left empty.
    bool unchanged = false;
    RenderStats stats {};                  // of the pass this call ran, if any
};

// Raster clear colour behind a scene without a visible sky (backend_opengl.cpp).
constexpr float kRasterBackground[3] { 7.0f / 255.0f, 10.0f / 255.0f, 18.0f / 255.0f };
// Accumulation generation used by every export, so exports are reproducible.
constexpr std::uint64_t kExportGeneration = 1;

class SceneEngine
{
public:
    // threadCount as for PathTracer: 0 picks the hardware concurrency.
    explicit SceneEngine (unsigned threadCount = 0);

    // Integrator settings. environmentVisibleToCamera is set per scene from
    // its lights (the raster draws a sky background only for an analytic sky
    // or environment image), so a value written here is overwritten.
    IntegratorSettings& settings() noexcept { return tracer_.settings; }

    std::uint32_t samplesPerPass = 1;
    std::uint32_t maximumPreviewSamples = 4096;
    // Decoded pixels for environment image records; unset lights none.
    std::function<EnvironmentPixels (const HarmonicMIDI::grid::SceneEnvironmentImage&)> environmentImageSource;

    // One progressive step. Returns false with error for an unusable request
    // (no scene, invalid camera, zero size). A cancelled pass is not an error:
    // the image keeps the samples that finished and the next call restarts.
    bool renderPreview (const EngineSceneSetup& setup, const EngineView& view, EngineColor color,
                        EngineImage& image, std::string& error);
    // Fixed samples per pixel. Fails when cancelled.
    bool renderExport (const EngineSceneSetup& setup, const EngineView& view, EngineColor color,
                       std::uint32_t samplesPerPixel, EngineImage& image, std::string& error);

    // Thread safe. Stops the pass in flight; the next preview restarts.
    void cancel() noexcept { cancel_.cancel(); }

    std::uint64_t sceneBuilds() const noexcept { return sceneBuilds_; }
    std::uint64_t previewGeneration() const noexcept { return previewGeneration_; }
    const RenderScene& renderScene() const noexcept { return scene_; }

private:
    bool prepare (const EngineSceneSetup& setup, const EngineView& view, PinholeCamera& camera,
                  bool& rebuilt, std::string& error);
    void traceDepth (const PinholeCamera& camera);
    void resolve (EngineColor color, EngineImage& image) const;

    PathTracer tracer_;
    CancellationToken cancel_;
    RenderScene scene_;
    bool sceneValid_ = false;
    bool skyVisible_ = false;
    EngineSceneSetup cachedSetup_;
    std::uint64_t sceneBuilds_ = 0;

    bool previewValid_ = false;
    std::uint64_t previewGeneration_ = kExportGeneration;
    EngineView previewView_ {};
    std::uint32_t previewSamples_ = 0;
    EngineColor previewColor_ = EngineColor::DisplayRgba8;
    std::vector<float> depth_;        // top row first, as traced
    std::vector<std::uint8_t> covered_;  // centre ray hit, top row first
};

// Exact comparisons the cache uses. Floats compare by bit pattern.
bool sameLightRecord (const HarmonicMIDI::grid::SceneLightRecord& a,
                      const HarmonicMIDI::grid::SceneLightRecord& b) noexcept;
bool sameMaterial (const Material& a, const Material& b) noexcept;
bool sameLightShape (const LightShape& a, const LightShape& b) noexcept;
bool sameAreaLight (const AreaLightRecord& a, const AreaLightRecord& b) noexcept;
bool sameCameraRecord (const HarmonicMIDI::grid::SceneCameraRecord& a,
                       const HarmonicMIDI::grid::SceneCameraRecord& b) noexcept;
bool sameSceneSetup (const EngineSceneSetup& a, const EngineSceneSetup& b) noexcept;

// True when the raster draws an analytic sky or environment image behind the
// scene for these lights, so camera rays that miss show the environment.
bool rasterDrawsSkyBackground (const std::vector<HarmonicMIDI::grid::SceneLightRecord>& lights) noexcept;

// IEEE 754 binary16 bits of value, round to nearest even, saturating to the
// largest finite half. NaN becomes 0.
std::uint16_t halfBits (float value) noexcept;

} // namespace videohelper::pathtrace
