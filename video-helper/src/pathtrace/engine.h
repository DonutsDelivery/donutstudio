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
// - Playback renders each frame at a fixed budget. While playbackActive is set,
//   or on a call whose time moved, the frame traces up to playbackSamples in one
//   call instead of one pass. Time still starts a new generation, so every
//   frame shows the same sample count. The next call at the same time without
//   playbackActive continues that accumulation toward maximumPreviewSamples.
// - Playback frames share a temporal history. Each new frame is an exponential
//   moving average of the frames before it, in linear radiance, without
//   reprojection: the camera is fixed while history lives, because a camera or
//   size change drops it. A scene change keeps it, since a music-driven scene
//   changes every frame. Each pixel guards itself instead. A pixel whose centre
//   ray hit something else takes the current frame, and elsewhere the history is
//   clipped to the current frame's 3x3 neighbourhood, so a lighting or material
//   step cannot smear. The accumulator never sees the blend, so leaving playback
//   resumes plain convergence, and export never reads the history.
// - Export clears and renders exactly the requested samples per pixel with a
//   fixed generation, so the same request always gives the same bytes.
//
// Output matches the raster's compositor contract (backend_opengl.cpp, the
// fixture scene draw): Color is clamped linear RGBA8 with no transfer for SDR
// preview, or linear sRGB RGBA16F for the HDR working representation. Alpha
// is 1. Depth is R32F, (view depth - near) / (far - near) clamped to [0, 1],
// 1 where the centre ray of a pixel hits nothing. Rows are bottom first, like
// the GL scene attachments. Where the raster draws no sky, a pixel whose centre
// ray misses shows the clear colour (7, 10, 18) / 255 on the display: those
// values directly on RGBA8, and their sRGB decode on RGBA16F, which the
// compositor encodes for display.
//
// A surface tinted by a frame texture leaves its samples out of colorBytes and in
// an EngineFrameLayer, untinted. The GPU pass in cpu_scene_frame.h samples the frame
// and adds them back, so the frame never comes to the CPU.
//
// Nothing here touches a GPU. The helper uploads the bytes (cpu_scene_frame.h).

#include "integrator.h"
#include "render_outputs.h"
#include "thread_pool.h"
#include "../../../shared/Visual3DScene.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
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
    // Objects that scatter by diffraction orders (PrepareOptions::objectGratings).
    std::vector<ObjectGratingBinding> objectGratings;
    // Objects tinted by a frame texture (PrepareOptions::objectFrames).
    std::vector<ObjectFrameBinding> objectFrames;
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
    // Motion blur (IngestOptions): the shutter in frames from the current one,
    // closed when shutterClose <= shutterOpen, and the scene one frame earlier.
    // previousObjectWorlds is empty or lists every object like objectWorlds;
    // previousInstances is empty or lists every instance; previousPositions
    // is empty or a prefix of the scene vertices.
    float shutterOpen = 0.0f;
    float shutterClose = 0.0f;
    std::vector<Mat4> previousObjectWorlds;
    std::vector<HarmonicMIDI::grid::SceneTransform3D> previousInstances;
    std::shared_ptr<const std::vector<HarmonicMIDI::grid::SceneVec3>> previousPositions;
};

// Everything that, when it changes, restarts accumulation.
struct EngineView
{
    HarmonicMIDI::grid::SceneCameraRecord camera {};
    // The camera one frame earlier, with its runtime offsets applied
    // (arbitgpu::priorRuntimeCamera). The Motion output and motion blur over
    // the setup's shutter both read it. Empty: the camera holds still.
    std::optional<HarmonicMIDI::grid::SceneCameraRecord> previousCamera {};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    float timeSeconds = 0.0f;
};

// How a call chose its samples. The [pt] log and EngineImage::mode report it.
enum class EngineRenderMode : std::uint8_t
{
    Converging = 0,     // one pass per call, up to maximumPreviewSamples
    PlaybackFixed = 1,  // the frame reaches playbackSamples within the call
    Export = 2          // the requested samples per pixel, from a clear accumulator
};

const char* engineRenderModeName (EngineRenderMode mode) noexcept;

enum class EngineColor : std::uint8_t
{
    // Clamped linear values, no transfer, 4 bytes per pixel. Bytes the compositor
    // shows as they are, so the product never draws a Path Trace image this way.
    DisplayRgba8 = 0,
    LinearRgba16f = 1   // IEEE half floats, linear sRGB, 8 bytes per pixel
};

// One frame's samples of an image, in the image's rows.
struct EngineFrameLayer
{
    // Mean untinted radiance over all the pixel's samples, RGBA16F whatever the image's format.
    std::vector<std::uint8_t> lightBytes;
    // Three floats per pixel: mean u and v of the samples that hit the surface, and
    // the share of the pixel's samples that did.
    std::vector<float> uvCoverage;
};

struct EngineImage
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    EngineColor color = EngineColor::DisplayRgba8;
    std::vector<std::uint8_t> colorBytes;  // tight rows, bottom row first
    std::vector<float> depth;              // width * height, bottom row first
    // The denoiser's guides (Plan 3, Denoising), three floats per pixel, bottom row
    // first: the mean first-hit albedo and the normalized mean world-space normal.
    // Zeros where the centre ray and every sample missed.
    std::vector<float> albedo;
    std::vector<float> normal;
    std::vector<EngineFrameLayer> frameLayers;  // by ObjectFrameBinding::layer
    // Render outputs (render_outputs.h), bottom row first, in the raster's raw
    // pass formats: RGBA16F half bits for normal and emission, R8 mask, R32UI
    // ids and RG16F motion. They come from the centre ray, so the denoiser never
    // touches them.
    std::vector<std::uint16_t> normalPass;
    std::vector<std::uint16_t> emissionPass;
    std::vector<std::uint8_t> maskPass;
    std::vector<std::uint32_t> materialIdPass;
    std::vector<std::uint32_t> objectIdPass;
    std::vector<std::uint16_t> motionPass;
    std::uint64_t generation = 0;          // accumulation this image belongs to
    std::uint32_t samplesPerPixel = 0;     // accumulated in this generation
    bool restarted = false;                // this call started a new generation
    bool sceneRebuilt = false;             // this call rebuilt the RenderScene and BVH
    bool traced = false;                   // this call added samples
    // The image blends earlier playback frames. A playback frame without history
    // yet, a converging frame and an export are the accumulation alone.
    bool temporalDenoise = false;
    // Preview only: no samples were added and nothing restarted, so the
    // previous image is still exact. colorBytes and depth are left empty.
    bool unchanged = false;
    // Preview only: the device isn't ready, so nothing changed and no image exists.
    bool pending = false;
    EngineRenderMode mode = EngineRenderMode::Converging;
    RenderStats stats {};                  // of the pass this call ran, if any
};

// Raster clear colour behind a scene without a visible sky (backend_opengl.cpp).
constexpr float kRasterBackground[3] { 7.0f / 255.0f, 10.0f / 255.0f, 18.0f / 255.0f };
// The same colour as linear sRGB. Writing the values above into a linear image
// encodes them twice and shows slate (46, 56, 75).
constexpr float kLinearRasterBackground[3] { 0.00212469f, 0.00303527f, 0.00604883f };
// Accumulation generation used by every export, so exports are reproducible.
constexpr std::uint64_t kExportGeneration = 1;
// Samples per pixel of one playback frame, four times the cost of the one-sample
// frames playback used to show. No measurement chose this value.
constexpr std::uint32_t kDefaultPlaybackSamples = 4;
// The current frame's share of a blended playback frame. 0.2 is the temporal weight
// of SVGF (Schied et al., HPG 2017). No measurement here chose this value.
constexpr float kDefaultPlaybackCurrentWeight = 0.2f;
// Standard deviations of the current frame's 3x3 neighbourhood that history may stray
// from its mean (variance clipping, Salvi, GDC 2016).
constexpr float kPlaybackHistoryClipSigma = 1.25f;
// A pixel keeps history while its centre depth moved by at most this share of the
// larger depth, which lets a surface deform slowly without resetting every frame.
constexpr float kPlaybackHistoryDepthTolerance = 0.01f;

// The Preview sample target. A Preview seeded like the export (Accurate) stops at the
// export's sample count, so its settled image is the export image. Otherwise (Fast) it
// stops at previewSamples. Zero becomes one, as an engine needs one sample to resolve.
inline std::uint32_t previewSampleTarget (bool seededLikeExport, std::uint32_t previewSamples,
                                          std::uint32_t exportSamples) noexcept
{
    const auto target = seededLikeExport ? exportSamples : previewSamples;
    return target > 0 ? target : 1u;
}

class SceneEngine
{
public:
    // threadCount as for PathTracer: 0 renders on sharedThreadPool().
    explicit SceneEngine (unsigned threadCount = 0);

    // Integrator settings. environmentVisibleToCamera is set per scene from
    // its lights (the raster draws a sky background only for an analytic sky
    // or environment image that is visible to the camera), so a value written
    // here is overwritten.
    IntegratorSettings& settings() noexcept { return tracer_.settings; }

    std::uint32_t samplesPerPass = 1;
    std::uint32_t maximumPreviewSamples = 4096;
    // The fixed budget of a playback frame, capped by maximumPreviewSamples. Zero
    // becomes one. Export ignores it.
    std::uint32_t playbackSamples = kDefaultPlaybackSamples;
    // The current frame's share of a blended playback frame. One or more turns the
    // temporal history off, so playback frames are the accumulation alone.
    float playbackCurrentWeight = kDefaultPlaybackCurrentWeight;
    // The transport is running. A frame already at the budget stays there and
    // traces nothing. A time change selects the budget without this flag; set it
    // when frames can repeat a time during playback. Clearing it resumes converging.
    bool playbackActive = false;
    // Accurate render mode: Preview passes draw the export sample streams
    // (kExportGeneration), so the preview after N samples equals renderExport
    // at N samples bit for bit. The accumulation still restarts on its own
    // generation. False keeps the preview streams independent of the export.
    bool previewSeededLikeExport = false;
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
    // The next preview passes stop once generation moves past captured, the value
    // read with the inputs they render (render_input_generation.h). Null watches
    // nothing. Exports never watch: they render a snapshot to the end.
    void watchInputs (const std::atomic<std::uint64_t>* generation, std::uint64_t captured) noexcept
    {
        cancel_.watch (generation, captured);
    }
    // The last preview pass stopped because its inputs were edited.
    bool lastPassObsolete() const noexcept { return lastPassObsolete_; }
    // Every pass renders on this device from now on (PathTracer::setDevicePass).
    // While ready says false, a preview of a scene the device can draw returns pending
    // without touching its film or sample stream. An export doesn't ask: its pass waits
    // for the device.
    void setDevicePass (PathTracer::DevicePassFunction pass, std::function<bool()> ready = {})
    {
        tracer_.setDevicePass (std::move (pass));
        deviceReady_ = std::move (ready);
    }
    void setDeviceSceneFallback (bool allowed) noexcept { tracer_.setDeviceSceneFallback (allowed); }
    // The next preview starts a new generation on an empty film at sample 0.
    void restartPreview() noexcept
    {
        previewValid_ = false;
        historyValid_ = false;
        historyShown_ = false;
    }

    std::uint64_t sceneBuilds() const noexcept { return sceneBuilds_; }
    std::uint64_t previewGeneration() const noexcept { return previewGeneration_; }
    // True while a Preview accumulation is live and not at maximumPreviewSamples:
    // below it the image keeps converging, and above it (the target was lowered) the
    // next call restarts. The viewport holds its idle gate open while this holds, so
    // the image never settles on a noisy frame or on more samples than the target.
    bool previewProgressing() const noexcept
    {
        return previewValid_ && previewSamples_ != maximumPreviewSamples;
    }
    unsigned previewTarget() const noexcept { return maximumPreviewSamples; }
    unsigned previewSampleCount() const noexcept { return previewSamples_; }
    // The mode of the last render call.
    EngineRenderMode renderMode() const noexcept { return renderMode_; }
    const RenderScene& renderScene() const noexcept { return scene_; }

private:
    std::function<bool()> deviceReady_;
    bool prepare (const EngineSceneSetup& setup, const EngineView& view, PinholeCamera& camera,
                  bool& rebuilt, std::string& error);
    // prepare, timed as SceneLower and CpuBvhBuild.
    bool timedPrepare (const EngineSceneSetup& setup, const EngineView& view, PinholeCamera& camera,
                       bool& rebuilt, FramePhaseTimes& phases, std::string& error);
    // The centre ray's depth and render outputs. setup is this call's, not the
    // cached one: a snapshot equal in what the tracer ingests can still carry
    // other previous transforms, which only Motion reads.
    void traceDepth (const PinholeCamera& camera, const EngineView& view, const EngineSceneSetup& setup);
    void resolve (EngineColor color, EngineImage& image) const;
    // Folds the accumulator's frame into the history, or seeds the history from it.
    void updatePlaybackHistory (bool blend);

    PathTracer tracer_;
    CancellationToken cancel_;
    RenderScene scene_;
    bool sceneValid_ = false;
    bool skyVisible_ = false;
    // What camera misses show when the world is hidden from the camera.
    HarmonicMIDI::grid::SceneCameraBackground cameraBackground_ {};
    EngineSceneSetup cachedSetup_;
    std::uint64_t sceneBuilds_ = 0;

    bool previewValid_ = false;
    // The light-path settings the live accumulation was drawn with.
    IntegratorSettings previewLightPaths_ {};
    std::uint64_t previewGeneration_ = kExportGeneration;
    EngineView previewView_ {};
    std::uint32_t previewSamples_ = 0;
    // Whether the live accumulation drew the export streams.
    bool previewSeeded_ = false;
    EngineColor previewColor_ = EngineColor::DisplayRgba8;
    EngineRenderMode renderMode_ = EngineRenderMode::Converging;
    std::vector<float> depth_;        // top row first, as traced
    std::vector<std::uint8_t> covered_;  // centre ray hit, top row first
    std::vector<FirstHitOutput> outputs_;  // top row first, as traced

    // The temporal history, all film-sized and top row first. It survives only
    // between playback frames with one camera and one film size.
    std::vector<Rgb> history_;
    std::vector<Rgb> historyCurrent_;  // the accumulator's frame, read by its neighbours
    std::vector<float> historyDepth_;
    std::vector<std::uint8_t> historyCovered_;
    bool historyValid_ = false;
    // The image on screen is history_, not the accumulator's mean.
    bool historyShown_ = false;
    bool lastPassObsolete_ = false;
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
