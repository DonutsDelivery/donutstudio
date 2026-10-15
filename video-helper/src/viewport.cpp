#include "viewport.h"
#include "viewport_publish_gate.h" // idle production gate for the docked (shm/shared) loop
#include "loop_step_probe.h" // times the calls inside the render loop that run long
#include "viewport_project_generation.h" // one project generation for the context and every owner
#include "block_c_frame_owner.h"
#include "particle_history.h"
#include "media.h"
#include "lut_loader.h"
#include "block_c_packer.h" // A2 Block C voice allocator (M5 live score packing)
#include "video_param_grammar.h" // Stage 6: shared render-graph param resolver (single source of truth)
#include "block_b_analyzer.h" // A3 Block B analyzer (M4 live audio features)
#include "block_b_audio_features.h" // FeatureFrame -> visual AudioFeatures (shared with export)
#include "mix_analyze.h"      // baked-mix offline analysis (audio parity when stopped)
#include "lua_hook.h"         // P2 Scripts-tab live preview (M8 per-frame Lua hook)
#include "js_hook.h"          // P2 Scripts-tab live preview (JS hook; shares FrameCtx)
#include "visual_plan_executor.h"
#include "optical_flow_preview_decode.h"
#include "optical_flow_product_receipt_json.h"
#include "visual_parameter_history.h"
#include "imported_animation_visual_plan_execution.h"
#include "imported_scene_visual_plan_execution.h"
#include "fixture_scene_renderer.h"
#include "VisualAnimationPoseControls.h"
#include "imported_geometry_execution.h"
#include "sdf_visual_plan_execution.h"
#include "volume_visual_plan_execution.h"
#include "depth_texture_runtime.h"
#include "tracking_runtime.h"
#include "visual_plan_publication.h"
#include "viewport_telemetry_owner.h"
#include "helper_glfw.h"
#include "VideoFrameSharedMemory.h"
#if ARBIT_HAVE_ONNX || ARBIT_HAVE_NCNN
#include "interp.h" // retime tier 2 (async RIFE "Speed Warp") worker
#endif

#include "gpu_export_common.h" // ARBIT_HAVE_GPU_SHARED + gpuexp::Exporter alias
#if ARBIT_HAVE_GPU_SHARED
#include "fd_socket.h"
#include "SharedGpuSurfaceProtocol.h"
#include "SharedGpuSurfaceCloseReason.h"
#endif

#include <nlohmann/json.hpp>

#include <GLFW/glfw3.h>
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
#include <IOSurface/IOSurface.h>
#include "mac_metal_viewport_surface.h"
#endif

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <optional>
#include <set>
#include <string_view>
#include <tuple>

#if ARBIT_HAVE_GPU_SHARED
// Fence fds only exist on the dmabuf transport; give the shared code one
// close call that compiles everywhere.
#if defined (_WIN32)
static void closeFdIfValid (int) {}
#else
#include <unistd.h>
static void closeFdIfValid (int fd) { if (fd >= 0) ::close (fd); }
#endif
#endif

using json = nlohmann::json;

namespace
{
struct SceneTraceColorStats
{
    size_t differentFromCorner = 0;
    size_t nonblack = 0;
    size_t bright = 0;
    size_t alpha = 0;
    uint64_t hash = 1469598103934665603ull;
    int minX = 0, minY = 0, maxX = -1, maxY = -1;
    uint8_t corner[3] {};
};

SceneTraceColorStats sceneTraceColorStats(const uint8_t* pixels, int width, int height,
                                          size_t rowBytes, bool bgra)
{
    SceneTraceColorStats stats;
    const int red = bgra ? 2 : 0;
    const int blue = bgra ? 0 : 2;
    stats.corner[0] = pixels[red];
    stats.corner[1] = pixels[1];
    stats.corner[2] = pixels[blue];
    stats.minX = width;
    stats.minY = height;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const auto* pixel = pixels + static_cast<size_t>(y) * rowBytes + x * 4;
            const uint8_t r = pixel[red], g = pixel[1], b = pixel[blue];
            stats.differentFromCorner += r != stats.corner[0] || g != stats.corner[1]
                || b != stats.corner[2];
            stats.nonblack += r != 0 || g != 0 || b != 0;
            stats.alpha += pixel[3] != 0;
            for (const uint8_t value : { r, g, b, pixel[3] })
            {
                stats.hash ^= value;
                stats.hash *= 1099511628211ull;
            }
            if (r > 32 || g > 32 || b > 32)
            {
                ++stats.bright;
                stats.minX = std::min(stats.minX, x);
                stats.minY = std::min(stats.minY, y);
                stats.maxX = std::max(stats.maxX, x);
                stats.maxY = std::max(stats.maxY, y);
            }
        }
    return stats;
}

void printSceneTraceColor(const char* stage, int clipId, uint64_t seqHint,
                          uint64_t resourceId, const uint8_t* pixels,
                          int width, int height, size_t rowBytes, bool bgra)
{
    const auto stats = sceneTraceColorStats(pixels, width, height, rowBytes, bgra);
    std::fprintf(stderr,
        "[viewport-scene-trace] %s clip=%d seqHint=%llu resource=%llu extent=%dx%d hash=%016llx corner=%u,%u,%u different=%zu nonblack=%zu bright=%zu bounds=%d,%d,%d,%d alpha=%zu total=%zu\n",
        stage, clipId, static_cast<unsigned long long>(seqHint),
        static_cast<unsigned long long>(resourceId), width, height,
        static_cast<unsigned long long>(stats.hash), static_cast<unsigned>(stats.corner[0]),
        static_cast<unsigned>(stats.corner[1]), static_cast<unsigned>(stats.corner[2]),
        stats.differentFromCorner, stats.nonblack, stats.bright,
        stats.bright ? stats.minX : 0, stats.bright ? stats.minY : 0,
        stats.bright ? stats.maxX - stats.minX + 1 : 0,
        stats.bright ? stats.maxY - stats.minY + 1 : 0, stats.alpha,
        static_cast<size_t>(width) * height);
}

inline bool hasNativeVisualSource(const std::vector<videowire::CompiledVisualLayerPlan>& plans, int clipId)
{
    if (videohelper::volume::hasVisualVolume(plans, clipId)
        || videohelper::sdf::hasVisualSdf(plans, clipId))
        return true;
    const auto* plan = videowire::findVisualLayerPlan(plans, clipId);
    if (plan == nullptr || videowire::isTypedParticlePlan(plans, clipId)
        || videowire::isGeometryCoreRenderPlan(plans, clipId))
        return false;
    const auto hasOperation = [&](std::string_view kind)
    {
        return std::any_of(plan->operations.begin(), plan->operations.end(),
            [&](const auto& operation) { return operation.kind == kind; });
    };
    // Every Render 3D scene (a direct imported scene, or a composed Scene 3D
    // graph with a retained scene and no imported source) needs its native
    // preparation. Without it the generic path composites the input layer and
    // reports success, so the scene silently drops out. Particle overlays and
    // Geometry Core keep their dedicated dispatch above.
    return hasOperation(visualimportedscenerender::kRenderNodeKind)
        || (hasOperation(typedscenepass::kSceneKind) && hasOperation(typedscenepass::kNodeKind));
}

int64_t nowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds> (
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Render-loop phase probe (diagnostic only). The render thread stamps the
// phase it enters; info() reads the stamp for loopPhase/loopPhaseAgeMs.
enum class LoopPhase : int
{
    Idle = 0, TickTop, SocketPump, LinearWait, Resize, TransportRead, EventGate,
    TransportStore, AudioDrain, ShouldProduce, Advance, Composite, Render, Present,
    Publish, FramePacing, Count
};
constexpr const char* kLoopPhaseNames[] = {
    "idle", "tick-top", "socket-pump", "linear-wait", "resize", "transport-read",
    "event-gate", "transport-store", "audio-drain", "should-produce", "advance",
    "composite", "render", "present", "publish", "frame-pacing"
};
static_assert (sizeof (kLoopPhaseNames) / sizeof (kLoopPhaseNames[0])
               == static_cast<size_t> (LoopPhase::Count), "loop phase names out of step");

void applyAlphaMatte (DecodedFrame& frame, MediaContext& matte, double sourceSec)
{
    DecodedFrame matteFrame;
    if (! matte.getFrame (sourceSec, frame.width, frame.height, matteFrame).empty()
        || matteFrame.width != frame.width || matteFrame.height != frame.height
        || matteFrame.rgba.empty())
        return;

    for (int y = 0; y < frame.height; ++y)
    {
        auto* frameRow = frame.rgba.data() + static_cast<size_t> (y) * frame.strideBytes;
        const auto* matteRow = matteFrame.rgba.data()
                             + static_cast<size_t> (y) * matteFrame.strideBytes;
        for (int x = 0; x < frame.width; ++x)
            frameRow[x * 4 + 3] = static_cast<uint8_t> (
                (frameRow[x * 4 + 3] * matteRow[x * 4]) / 255);
    }
}

bool shouldUseStrictMetalBackend (bool shared)
{
#if defined (__APPLE__) && ARBIT_HAVE_IOSURFACE
#if defined (__arm64__) || defined (__aarch64__)
    (void) shared;
    return true;
#else
    return shared || MacMetalViewportSurface::available();
#endif
#else
    (void) shared;
    return false;
#endif
}

void configureGlfwWindowHints (bool offscreen, bool shared, bool alwaysOnTop,
                               bool metalOnly = false)
{
    // OpenGL 4.3 core (4.1 on macOS, which caps desktop GL there).
#if defined (__APPLE__)
    if (metalOnly)
        glfwWindowHint (GLFW_CLIENT_API, GLFW_NO_API);
    else
    {
        glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 1);
        glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    }
#else
    (void) metalOnly;
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
    glfwWindowHint (GLFW_FLOATING, alwaysOnTop ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint (GLFW_RESIZABLE, GLFW_TRUE);
    if (offscreen)
    {
        glfwWindowHint (GLFW_VISIBLE, GLFW_FALSE);
#if ARBIT_HAVE_DMABUF && ! ARBIT_HAVE_VKFD
        if (shared)
            glfwWindowHint (GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
#else
        (void) shared;
#endif
    }
}

// Pending timestamped parameter change (graph_set_param with atBeat >= 0).
struct PendingParam
{
    std::string paramId;
    double value = 0.0;
    double atBeat = 0.0;
};

// Initialize an effect slot's params to the canonical table defaults.
void resetSlotParams (videorender::EffectSlotState& e)
{
    std::memset (e.params, 0, sizeof (e.params));
    if (const auto* def = videofx::effectDefFor (e.type))
        for (int i = 0; i < def->paramCount; ++i)
            e.params[i] = def->params[i].defaultValue;
}

// ---- Video scopes (PROTOCOL.md §Scopes) -----------------------------------

// Wire sizes. The sample frame is the renderer's fixed downscale target;
// 480x135 = 64,800 samples keeps every histogram bin within uint16_t even
// for a solid-colour frame.
constexpr int kScopeWaveW = 480, kScopeWaveH = 64;   // waveform: column x luma bin
constexpr int kScopeVecN = 128;                      // vectorscope: Cb x Cr density
static_assert (kScopeWaveW == videorender::FrameRenderer::kScopeW,
               "waveform width must match the scope readback width");
static_assert (videorender::FrameRenderer::kScopeW
                   * videorender::FrameRenderer::kScopeH <= 65535,
               "histogram bins are uint16_t — sample count must fit");

// Standard base64 ('+'/'/', '=' padding) — counterpart of the decoder in
// main.cpp; juce::Base64 on the Arbit side reads this directly.
std::string base64Encode (const uint8_t* data, size_t n)
{
    static const char* tab =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve ((n + 2) / 3 * 4);
    for (size_t i = 0; i < n; i += 3)
    {
        const uint32_t b0 = data[i];
        const uint32_t b1 = i + 1 < n ? data[i + 1] : 0;
        const uint32_t b2 = i + 2 < n ? data[i + 2] : 0;
        const uint32_t v = (b0 << 16) | (b1 << 8) | b2;
        out.push_back (tab[(v >> 18) & 63]);
        out.push_back (tab[(v >> 12) & 63]);
        out.push_back (i + 1 < n ? tab[(v >> 6) & 63] : '=');
        out.push_back (i + 2 < n ? tab[v & 63] : '=');
    }
    return out;
}
} // namespace

struct Viewport::Impl
{
    Impl() { visualPlanPublication->state.setTelemetryOwner (visualTelemetry); }
    mutable std::mutex mutex;
    std::string depthCacheRoot;

#if defined (__APPLE__)
    // Cocoa requires GLFW initialization and window creation/destruction on
    // the process main thread. The RPC thread is main, so macOS offscreen
    // modes create their hidden context there, detach it, then let the render
    // thread own the GL context for the lifetime of the viewport.
    GLFWwindow* macPrecreatedWindow = nullptr;
#endif

    // Control-plane state (written by RPC thread, read by render thread).
    std::vector<ViewportSegment> segments;
    std::vector<videowire::VisualEventScheduleBinding> visualEventSchedules;
    std::shared_ptr<const videorender::VisualParameterTimeline> parameterTimeline;
    std::shared_ptr<const videowire::ResolvedVisualSnapshot> opticalFlowSnapshot;
    nlohmann::json opticalFlowInspection = nullptr;
    videowire::VisualPlanTelemetry visualTelemetry;
    videowire::VisualPlanViewportPublicationOwner visualPlanPublication;
    uint64_t failedPlanLowerings = 0;
    std::string lastPlanLoweringError;
    int inspectionClipId = -1;
    uint64_t inspectionStructuralRevision = 0;
    int inspectionNodeId = 0;
    int inspectionOutputPort = -1;
    bool inspectionResourceReady = false;
    unsigned inspectionResourceHandle = 0;
    uint64_t inspectionResourceGeneration = 0;
    int inspectionResourceWidth = 0;
    int inspectionResourceHeight = 0;
    std::string inspectionResourceBackend;
    bool inspectionMetalAdmitted = false;
    videopreview::State inspectionPresentation;
    std::map<int, ClipGraphParams> clipParams; // keyed by clipId
    std::vector<PendingParam> pendingParams;
    videoshm::TransportRegion transport;
    std::string transportName;
    // M4 Block B live: the master-mix audio ring Arbit's audio thread writes.
    // The render loop drains it under `mutex` (guards against attachAudio's
    // open()/remap) and runs the BlockBAnalyzer; see the render loop.
    videoshm::AudioRingRegion audioRing;
    std::string audioName;
    uint64_t timelineGeneration = 0; // bumped by setTimeline (stream pruning)
    videotime::BeatTimeline beatTimeline;
    uint64_t beatTimelineGeneration = 0;

    // Frame-perfect audio parity (sweep): the offline-analyzed master mix
    // (setAudioMix). The live ring only flows WHILE PLAYING, so while STOPPED the
    // render loop reads THESE features (same analysis the exporter runs) at the
    // current frame's time instead of zero-feeding audio-reactive shaders/mods.
    // previewMixGeneration bumps on every push so the render thread re-copies its
    // local snapshot only when the mix actually changes (the vector is large —
    // one FeatureFrame per hop for the whole song). Empty ⇒ stopped = zero-feed.
    std::vector<arbitblockb::FeatureFrame> previewMixFeats;
    std::shared_ptr<const videohelper::AnalysisSources> previewSourceFeats;
    double previewMixSr = 0.0;
    uint64_t previewMixGeneration = 0;

    // M5 Block C live score (setScore). Read by the render loop each frame and
    // packed via the A2 BlockCPacker. scoreGeneration is bumped on every push so
    // the render thread re-copies + resets its packer only when the note set
    // actually changes (not every frame). Empty score ⇒ shaders see the
    // zero-feed (uNoteCount 0), identical to never calling setScore.
    std::shared_ptr<const arbitmod::Score> scoreSnapshot = std::make_shared<const arbitmod::Score>();
    uint64_t scoreGeneration = 0;
    uint64_t presentedScoreGeneration = 0;
    double publishedDisplaySec = 0.0;
    uint64_t publishedScoreRevision = 0;
    uint64_t publishedCanonicalNoteRows = 0;
    uint64_t publishedImportedSceneCount = 0;
    uint64_t publishedImportedSceneSubmittedInstances = 0;
    uint64_t publishedImportedSceneSubmittedNoteInstances = 0;
    canonicalblockc::OwnerIdentity scoreIdentity;

    // M6 cross-domain mod matrix (setModMatrix). Routings are evaluated each
    // frame against the live clock + score + audio (the live mix from the Block
    // B audio ring) and laid over the per-frame clip-param copy. modGeneration is
    // bumped on every push so the render thread re-copies + resets its routing
    // states only when the routing set changes. Empty ⇒ no overlay (identical
    // to never calling setModMatrix).
    std::vector<arbitmod::Routing> routings;
    uint64_t modGeneration = 0;
    videocontrol::Plan controlPlan;
    uint64_t controlPlanGeneration = 0;
    struct VisualEvent
    {
        std::string kind;
        int clipId = -1;
        uint64_t sequence = 0;
        uint64_t epoch = 0;
    };
    std::vector<VisualEvent> visualEvents;
    std::map<std::pair<std::string, int>, uint64_t> visualEventLastSequences;
    uint64_t visualEventEpoch = 0;

    // P2 Scripts-tab live preview (setScript). The project-global per-frame hook
    // source + engine ("js" | "lua"). scriptGeneration is bumped on every push so
    // the render thread re-copies + recompiles only when the script actually
    // changes (not every frame). Empty source ⇒ no hook (identical to never
    // calling setScript).
    std::string scriptSource;
    std::string scriptLang;
    uint32_t scriptCpuMs = 0;
    uint32_t scriptMemoryMiB = 0;
    uint64_t scriptGeneration = 0;

    // Text overlay store (text_set_image / text_remove) — independent of the
    // timeline so texts survive segment resends. Pixel uploads and texture
    // deletes run on the render thread (GL context); the RPC thread only
    // stages pixels / flags under the mutex.
    struct TextState
    {
        // control-plane (mutex-guarded)
        double startSec = 0.0, durationSec = 0.0;
        int ownerClipId = -1;                  // -1 = project-level text
        float posX = 0.0f, posY = 0.0f;       // base NDC from text_set_image
        float opacity = 1.0f;
        float translateX = 0.0f, translateY = 0.0f; // graph_set_param offsets
        float visible = 1.0f;
        float zOrder = 0.0f;
        std::vector<uint8_t> pendingPixels;   // straight RGBA, stride w*4
        int pendingW = 0, pendingH = 0;
        bool pixelsDirty = false;
        bool removed = false;                 // render thread frees + erases
        // render-thread owned
        unsigned tex = 0;
        int texW = 0, texH = 0;
    };
    std::map<int, TextState> texts;           // keyed by textId

    // Per-clip 3D LUT staging (graph_set_lut). The RPC thread parses the
    // .cube file and stages the float grid here; the render thread (GL
    // context) uploads/deletes textures and clears the stage flags.
    struct LutState
    {
        // control-plane (mutex-guarded)
        std::string path;                 // "" = clear
        std::vector<float> pendingData;   // size^3 RGB triples
        int pendingSize = 0;
        bool dirty = false;               // upload (or clear, if path empty)
        // render-thread owned
        unsigned tex = 0;
        int size = 0;
    };
    std::map<int, LutState> luts;             // keyed by clipId

    // Video scope results (PROTOCOL.md §Scopes) — published by the render
    // thread under the mutex, read by scopeDataJson on the RPC thread.
    // seq == 0 means "no data yet". Disabled scopes keep their last arrays
    // (cheap, and scope_data only reports enabled types anyway).
    uint64_t scopeSeq = 0;
    std::vector<uint8_t> scopeWaveform;       // kScopeWaveW * kScopeWaveH u8
    std::vector<uint8_t> scopeVector;         // kScopeVecN * kScopeVecN u8
    uint16_t scopeHistogram[3][256] = {};     // R/G/B value counts

    // CPU scope accumulation over one downscaled composite frame (~65k
    // pixels at ~10 Hz — negligible). rgba is top-row-flipped by the blit,
    // which no scope cares about: pixels are samples, columns preserved.
    void computeScopes (uint32_t mask, const uint8_t* rgba, int w, int h)
    {
        std::vector<uint16_t> wave;
        std::vector<uint16_t> vec;
        uint32_t hist[3][256] = {};
        if (mask & Viewport::kScopeWaveform)
            wave.assign ((size_t) kScopeWaveW * kScopeWaveH, 0);
        if (mask & Viewport::kScopeVectorscope)
            vec.assign ((size_t) kScopeVecN * kScopeVecN, 0);

        for (int y = 0; y < h; ++y)
        {
            const uint8_t* row = rgba + (size_t) y * (size_t) w * 4;
            for (int x = 0; x < w; ++x)
            {
                const int r = row[x * 4 + 0];
                const int g = row[x * 4 + 1];
                const int b = row[x * 4 + 2];

                if (mask & Viewport::kScopeHistogram)
                {
                    ++hist[0][r];
                    ++hist[1][g];
                    ++hist[2][b];
                }
                if (mask & Viewport::kScopeWaveform)
                {
                    // Rec.709 luma, integer weights summing to 256.
                    const int luma = (54 * r + 183 * g + 19 * b) >> 8;
                    const int bin = kScopeWaveH - 1
                                  - std::clamp (luma * kScopeWaveH / 256, 0, kScopeWaveH - 1);
                    ++wave[(size_t) bin * kScopeWaveW + (size_t) x];
                }
                if (mask & Viewport::kScopeVectorscope)
                {
                    // BT.709 Cb/Cr, integer weights x256. Cb right, Cr up
                    // (row 0 = top), neutral at the centre cell (64, 64).
                    const int cb = (-29 * r - 99 * g + 128 * b) >> 8;  // -128..127
                    const int cr = (128 * r - 116 * g - 12 * b) >> 8;
                    const int vx = std::clamp (kScopeVecN / 2 + cb / 2, 0, kScopeVecN - 1);
                    const int vy = std::clamp (kScopeVecN / 2 - cr / 2, 0, kScopeVecN - 1);
                    ++vec[(size_t) vy * kScopeVecN + (size_t) vx];
                }
            }
        }

        // Density -> saturating 8-bit intensity (display gamma is the
        // panel's job; any monotone mapping works for plausibility checks).
        std::vector<uint8_t> wave8, vec8;
        if (mask & Viewport::kScopeWaveform)
        {
            wave8.resize (wave.size());
            for (size_t i = 0; i < wave.size(); ++i)
                wave8[i] = (uint8_t) std::min<uint32_t> (255u, (uint32_t) wave[i] * 40u);
        }
        if (mask & Viewport::kScopeVectorscope)
        {
            vec8.resize (vec.size());
            for (size_t i = 0; i < vec.size(); ++i)
                vec8[i] = (uint8_t) std::min<uint32_t> (255u, (uint32_t) vec[i] * 32u);
        }

        std::lock_guard<std::mutex> lock (mutex);
        ++scopeSeq;
        if (mask & Viewport::kScopeWaveform)
            scopeWaveform = std::move (wave8);
        if (mask & Viewport::kScopeVectorscope)
            scopeVector = std::move (vec8);
        if (mask & Viewport::kScopeHistogram)
            for (int c = 0; c < 3; ++c)
                for (int i = 0; i < 256; ++i)
                    scopeHistogram[c][i] = (uint16_t) std::min<uint32_t> (hist[c][i], 65535u);
    }

    // Window control requests (GLFW calls must run on the render thread).
    std::atomic<bool> wantClose { false };
    std::atomic<bool> boundsDirty { false };
    int reqX = 0, reqY = 0, reqW = 0, reqH = 0;
    std::atomic<int> fullscreenRequest { -1 }; // -1 none, 0 windowed, 1 fullscreen

    // Stats (written by render thread).
    std::atomic<uint64_t> framesPresented { 0 };
    std::atomic<uint64_t> lastFrameHash { 0 };
    // Retime interpolation tier actually used on the most recent frame:
    // 0=nearest, 1=frame-blend, 2=rife-cuda, 3=rife-coreml, 4=rife-vulkan (reported by info()).
    std::atomic<int> interpBackend { 0 };
#if ARBIT_HAVE_ONNX || ARBIT_HAVE_NCNN
    // Async RIFE worker for retime tier 2 — lazily started on first tier-2 use
    // (render-thread-owned; the worker has no GL context).
    std::unique_ptr<arbitinterp::InterpEngine> interp;
#endif
    std::atomic<uint64_t> sharedFramesSent { 0 };
    std::atomic<uint64_t> sharedFramesDroppedNoBuffer { 0 };
    // Render-loop probe: current LoopPhase, its entry time (steady-clock ns),
    // and loop iterations. Written by the render thread, read by info().
    std::atomic<int> loopPhase { 0 };
    std::atomic<int64_t> loopPhaseEnterNs { 0 };
    // Advance sub-step label: a string literal naming the call entered last,
    // and its entry time. Written only by the render thread.
    std::atomic<const char*> loopStep { "idle" };
    std::atomic<int64_t> loopStepEnterNs { 0 };
    std::atomic<uint64_t> loopTicks { 0 };
    std::atomic<uint64_t> sharedReleasesReceived { 0 };
    std::atomic<uint64_t> sharedReleasesInvalid { 0 };
    std::atomic<int> sharedFreeBuffers { 0 };
    std::atomic<int> sharedBusyBuffers { 0 };
    double measuredFps = 0.0;
    // Project (canonical) frame rate that drives the VALUE grid — the rate the
    // exporter steps at (job.fps). Set from the viewport_open* RPC (clkFps) and
    // updated live by viewport_set_clkfps so the live preview snaps modulation /
    // shader-clock / score / lua / source selection to the SAME frame grid the
    // export uses (frame-perfect parity). 0 ⇒ fall back to the display targetFps.
    // Distinct from targetFps, which only paces the DISPLAY refresh — the window
    // can still repaint at 60Hz, holding the current project-fps frame. Atomic
    // because the RPC thread writes it while the render thread reads it per frame.
    std::atomic<double> valueFps { 0.0 };
    double avOffsetSec = 0.0;
    double displaySec = 0.0;
    double sourcePtsSec = 0.0;
    double sourceIdealSec = 0.0;
    bool transportOpen = false;
    bool transportPlaying = false;
    uint32_t transportGeneration = 0;
    double transportAgeSec = 0.0;
    double transportPlayheadBeats = 0.0;
    std::string particleBackend = "none";
    std::string compositorBackend = "opengl";
    std::string gpuPath = "opengl";
    std::string backendPolicy = "legacy-opengl";
    std::string rendererError;
    // Written by the render loop once per frame and read by info().
    std::atomic<bool> importedScenePending { false };
    uint64_t fallbackCount = 0;
    int curW = 0, curH = 0;
    arbitgl::GpuCaps gpuCaps;          // snapshot once after context creation (P1)
    std::atomic<bool> windowOpen { false };
    std::string openError;
    std::atomic<bool> openDone { false };

    // Shared (zero-copy docked) mode state — render-thread owned except
    // where noted. sharedMode is set before the thread starts and never
    // changes. The exporter is the platform backend behind gpuexp::Exporter
    // (dmabuf / IOSurface / D3D11 interop — gpu_export_common.h).
    bool sharedMode = false;
    std::atomic<bool> dmabufCapable { false }; // "shared-GPU capable" (name is wire-compat)

    // shm-docked mode state. shmMode is set and the ring (Arbit-owned,
    // opened by name) attached before the thread starts; cursor/seq are
    // render-thread only.
    bool shmMode = false;
    videoshm::Region shmRing;
    std::atomic<bool> shmResizeDirty { false };
    std::atomic<int> shmReqW { 0 }, shmReqH { 0 };
    uint32_t shmSlotCursor = 0;
    uint64_t shmFrameSeq = 0;
    // Idle production gate inputs (viewport_publish_gate.h). contentRevision
    // moves after each state RPC has been applied, and on visibility and frame
    // requests. visible is set by the plugin; hidden viewports produce nothing.
    std::atomic<uint64_t> contentRevision { 1 };
    std::atomic<bool> visible { true };
#if ARBIT_HAVE_GPU_SHARED
    int sharedBufferCount = 3;
    std::atomic<bool> sharedResizeDirty { false };
    std::atomic<int> sharedReqW { 0 }, sharedReqH { 0 };
    gpuexp::Exporter exporter;
    FdSocketServer fdSocket;
    std::vector<gpuexp::ExportedBuffer> buffers; // render thread only
    uint64_t frameSeq = 0;                       // render thread only
    Viewport::SharedOpenResult sharedResult;     // filled before openDone
#endif

    ClipGraphParams* paramsFor (int clipId)
    {
        return &clipParams[clipId]; // default-constructs on first use
    }

    // Parses "clip<id>/<node>/<param>". Returns false on malformed IDs.
    static bool parseParamId (const std::string& id, int& clipId,
                              std::string& node, std::string& param)
    {
        if (id.rfind ("clip", 0) != 0) return false;
        const auto slash1 = id.find ('/');
        if (slash1 == std::string::npos) return false;
        const auto slash2 = id.find ('/', slash1 + 1);
        if (slash2 == std::string::npos) return false;
        const auto clipToken = std::string_view(id).substr(4, slash1 - 4);
        if (clipToken.empty()) return false;
        const auto parsed = std::from_chars(clipToken.data(),
            clipToken.data() + clipToken.size(), clipId);
        if (parsed.ec != std::errc {} || parsed.ptr != clipToken.data() + clipToken.size()
            || clipId < 0)
            return false;
        node = id.substr (slash1 + 1, slash2 - slash1 - 1);
        param = id.substr (slash2 + 1);
        return ! node.empty() && ! param.empty() && param.find('/') == std::string::npos;
    }

    // Parses "text<id>/<param>". Returns false on malformed IDs.
    static bool parseTextParamId (const std::string& id, int& textId,
                                  std::string& param)
    {
        if (id.rfind ("text", 0) != 0) return false;
        const auto slash = id.find ('/');
        if (slash == std::string::npos) return false;
        try { textId = std::stoi (id.substr (4, slash - 4)); }
        catch (...) { return false; }
        param = id.substr (slash + 1);
        return ! param.empty();
    }

    // Applies one text overlay param. Caller holds the mutex. Creates the
    // entry on first use (mirrors paramsFor) so automation values streamed
    // before text_set_image are not lost.
    bool applyText (int textId, const std::string& param, double value)
    {
        auto& t = texts[textId];
        if (param == "opacity")    { t.opacity = (float) std::clamp (value, 0.0, 1.0); return true; }
        if (param == "translateX") { t.translateX = (float) value; return true; }
        if (param == "translateY") { t.translateY = (float) value; return true; }
        if (param == "posX")       { t.posX = (float) value; return true; }
        if (param == "posY")       { t.posY = (float) value; return true; }
        if (param == "visible")    { t.visible = value >= 0.5 ? 1.0f : 0.0f; return true; }
        if (param == "zOrder")     { t.zOrder = (float) value; return true; }
        return false;
    }

    // Applies one parsed param. Caller holds the mutex.
    bool apply (int clipId, const std::string& node, const std::string& param,
                double value)
    {
        return applyTo (*paramsFor (clipId), node, param, value);
    }

    // Writes one parsed param onto a ClipGraphParams. Used by apply() (on the stored
    // params) and by the M6 mod-matrix overlay (on a per-frame copy, so the stored base
    // is never mutated). Delegates to the shared videoparam resolver — the SAME grammar
    // exporter.cpp uses, so the live viewport and the export can no longer diverge.
    static bool applyTo (ClipGraphParams& p, const std::string& node,
                         const std::string& param, double value)
    {
        if (videoparam::applyGraphParam (p, node, param, value))
            return true;
        // ISF/generator INPUT ("clip<id>/gen/<name>") — exporter-only in the shared
        // grammar, so handled here as the viewport's caller-side glue (mirrors
        // exporter.cpp::applyGraphParam). Stored unclamped; the shader clamps.
        if (node == "gen") { p.genParams[param] = value; return true; }
        int visualNodeId = 0;
        if (shadercatalog::parseRuntimeNodeAlias(node, visualNodeId))
        {
            p.visualParams[node + "/" + param] = value;
            return true;
        }
        return false;
    }

    // Reads one parsed param from a ClipGraphParams (the `base` a mod routing modulates:
    // combine(mode, base, value)). The inverse of applyTo, via the shared resolver.
    static bool readFrom (const ClipGraphParams& p, const std::string& node,
                          const std::string& param, double& out)
    {
        if (videoparam::getGraphParam (p, node, param, out))
            return true;
        // ISF/generator INPUT base for a mod-matrix routing (mirrors
        // exporter.cpp::getGraphParam). An unseeded name returns false so the
        // routing is skipped rather than applied to a junk base.
        if (node == "gen")
        {
            const auto it = p.genParams.find (param);
            if (it == p.genParams.end()) return false;
            out = it->second;
            return true;
        }
        int visualNodeId = 0;
        if (shadercatalog::parseRuntimeNodeAlias(node, visualNodeId))
        {
            const auto it = p.visualParams.find(node + "/" + param);
            if (it != p.visualParams.end())
            {
                out = it->second;
                return true;
            }
            // An unset per-key morph delta is zero. Only a deformation root, which seeds
            // selectedMorphWeight, takes that base, so other nodes stay skipped.
            std::size_t deltaKey = 0;
            if (!visualanimation::parseMorphWeightDeltaParameter(param, deltaKey)
                || p.visualParams.find(node + "/selectedMorphWeight") == p.visualParams.end())
                return false;
            out = 0.0;
            return true;
        }
        return false;
    }
};

Viewport::Viewport(videohelper::modelpayload::Store* modelPayloadStore)
    : modelPayloadStore_(modelPayloadStore)
{
}
Viewport::~Viewport() { close(); }
void Viewport::setDepthCacheRoot(std::string root)
{
    depthCacheRoot_ = std::move(root);
    if (impl_ != nullptr)
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->depthCacheRoot = depthCacheRoot_;
    }
}

const char* Viewport::sharedGpuPathTag()
{
#if ARBIT_HAVE_GPU_SHARED
    return gpuexp::kSharedGpuTag;
#else
    return "";
#endif
}

std::string Viewport::open (int width, int height, int x, int y,
                            bool alwaysOnTop, double targetFps, double clkFps)
{
    if (running_.load())
    {
        if (impl_ != nullptr && impl_->sharedMode)
            return "viewport is in shared (dmabuf) mode — close it first";
        if (impl_ != nullptr && impl_->shmMode)
            return "viewport is in shm (docked) mode — close it first";
        return {}; // already open
    }

    impl_ = std::make_unique<Impl>();
    impl_->depthCacheRoot = depthCacheRoot_;
    impl_->valueFps = clkFps;   // project value-grid fps (frame-perfect parity)
    impl_->openDone = false;
    impl_->reqX = x >= 0 ? x : 100;
    impl_->reqY = y >= 0 ? y : 100;
    impl_->reqW = width;
    impl_->reqH = height;
#if defined (__APPLE__)
    // Cocoa requires NSWindow creation and destruction on the process main
    // thread. The render worker may own Metal encoding, but it must reuse the
    // GLFW window created here while the RPC handler is still on main.
    if (! initializeHelperGlfw())
    {
        impl_.reset();
        return "glfwInit failed on macOS main thread";
    }
    const bool metalOnly = shouldUseStrictMetalBackend (false);
    configureGlfwWindowHints (false, false, alwaysOnTop, metalOnly);
    impl_->macPrecreatedWindow = glfwCreateWindow (width, height, "Arbit Video", nullptr, nullptr);
    if (impl_->macPrecreatedWindow == nullptr)
    {
        glfwTerminate();
        impl_.reset();
        return metalOnly
            ? "glfwCreateWindow failed for the macOS Metal viewport"
            : "glfwCreateWindow failed for the macOS OpenGL viewport";
    }
    if (x >= 0 && y >= 0)
        glfwSetWindowPos (impl_->macPrecreatedWindow, x, y);
    glfwMakeContextCurrent (nullptr);
#endif
    running_ = true;
    thread_ = std::thread ([this, width, height, x, y, alwaysOnTop, targetFps]
                           { renderLoop (width, height, x, y, alwaysOnTop, targetFps); });

    // Wait for the window to come up (or fail) so the RPC reply is accurate.
    while (! impl_->openDone.load())
        std::this_thread::sleep_for (std::chrono::milliseconds (5));
    if (! impl_->windowOpen.load())
    {
        const std::string err = impl_->openError.empty() ? "viewport window failed to open"
                                                         : impl_->openError;
        close();
        return err;
    }
    return {};
}

std::string Viewport::openShared (int width, int height, double targetFps,
                                  int bufferCount, SharedOpenResult& out,
                                  double clkFps)
{
#if ARBIT_HAVE_GPU_SHARED
    if (running_.load())
        return impl_ != nullptr && impl_->sharedMode
                   ? "shared viewport already open"
                   : "viewport window already open — close it first";

    impl_ = std::make_unique<Impl>();
    impl_->depthCacheRoot = depthCacheRoot_;
    impl_->valueFps = clkFps;   // project value-grid fps (frame-perfect parity)
    impl_->openDone = false;
    impl_->sharedMode = true;
    impl_->sharedBufferCount = std::clamp (bufferCount, 2, 4);
#if defined (__APPLE__)
    if (! initializeHelperGlfw())
    {
        impl_.reset();
        return "glfwInit failed on macOS main thread";
    }
    configureGlfwWindowHints (true, true, false, true);
    impl_->macPrecreatedWindow = glfwCreateWindow (width, height, "Arbit Video", nullptr, nullptr);
    if (impl_->macPrecreatedWindow == nullptr)
    {
        glfwTerminate();
        impl_.reset();
        return "glfwCreateWindow failed (macOS OpenGL 4.1 offscreen context unavailable?)";
    }
    glfwMakeContextCurrent (nullptr);
#endif
    running_ = true;
    thread_ = std::thread ([this, width, height, targetFps]
                           { renderLoop (width, height, -1, -1, false, targetFps); });

    while (! impl_->openDone.load())
        std::this_thread::sleep_for (std::chrono::milliseconds (5));
    if (! impl_->windowOpen.load())
    {
        const std::string err = impl_->openError.empty() ? "shared viewport failed to open"
                                                         : impl_->openError;
        close();
        return err;
    }
    out = impl_->sharedResult;
    return {};
#else
    (void) width; (void) height; (void) targetFps; (void) bufferCount; (void) out; (void) clkFps;
    return "helper built without shared-GPU export support";
#endif
}

std::string Viewport::resizeShared (int width, int height)
{
#if ARBIT_HAVE_GPU_SHARED
    // One receipt per call: requested size and the RPC result ("ok" or the error).
    const auto receipt = [width, height] (std::string result)
    {
        std::fprintf (stderr, "[viewport] shared resize request %dx%d -> %s\n",
                      width, height, result.empty() ? "ok" : result.c_str());
        return result;
    };
    if (! running_.load() || impl_ == nullptr || ! impl_->sharedMode)
        return receipt ("shared viewport not open");
    if (width < 16 || height < 16 || width > 8192 || height > 8192)
        return receipt ("bad shared viewport size");
    impl_->sharedReqW = width;
    impl_->sharedReqH = height;
    impl_->sharedResizeDirty = true;
    return receipt ({});
#else
    (void) width; (void) height;
    return "helper built without shared-GPU export support";
#endif
}

std::string Viewport::openShm (int width, int height, double targetFps,
                               const std::string& shmName, double clkFps)
{
    if (running_.load())
        return impl_ != nullptr && impl_->shmMode
                   ? "shm viewport already open"
                   : "another viewport mode is open — close it first";
    if (shmName.empty())
        return "missing shmName";
    if (width < 16 || height < 16 || width > 8192 || height > 8192)
        return "bad shm viewport size";

    impl_ = std::make_unique<Impl>();
    impl_->depthCacheRoot = depthCacheRoot_;
    impl_->valueFps = clkFps;   // project value-grid fps (frame-perfect parity)
    impl_->openDone = false;
    impl_->shmMode = true;
    if (! impl_->shmRing.open (shmName))
    {
        impl_.reset();
        return "failed to open frame shm region: " + shmName;
    }
    const auto* hdr = impl_->shmRing.header();
    if (hdr == nullptr || hdr->slotCount == 0 || hdr->slotBytes == 0)
    {
        impl_.reset();
        return "bad frame shm region: " + shmName;
    }

#if defined (__APPLE__)
    if (! initializeHelperGlfw())
    {
        impl_.reset();
        return "glfwInit failed on macOS main thread";
    }
    const bool metalOnly = shouldUseStrictMetalBackend (false);
    configureGlfwWindowHints (true, false, false, metalOnly);
    impl_->macPrecreatedWindow = glfwCreateWindow (width, height, "Arbit Video", nullptr, nullptr);
    if (impl_->macPrecreatedWindow == nullptr)
    {
        glfwTerminate();
        impl_.reset();
        return metalOnly
            ? "glfwCreateWindow failed for the macOS Metal offscreen viewport"
            : "glfwCreateWindow failed (macOS OpenGL 4.1 offscreen context unavailable?)";
    }
    glfwMakeContextCurrent (nullptr);
#endif

    running_ = true;
    thread_ = std::thread ([this, width, height, targetFps]
                           { renderLoop (width, height, -1, -1, false, targetFps); });

    while (! impl_->openDone.load())
        std::this_thread::sleep_for (std::chrono::milliseconds (5));
    if (! impl_->windowOpen.load())
    {
        const std::string err = impl_->openError.empty() ? "shm viewport failed to open"
                                                         : impl_->openError;
        close();
        return err;
    }
    return {};
}

std::string Viewport::resizeShm (int width, int height)
{
    if (! running_.load() || impl_ == nullptr || ! impl_->shmMode)
        return "shm viewport not open";
    if (width < 16 || height < 16 || width > 8192 || height > 8192)
        return "bad shm viewport size";
    impl_->shmReqW = width;
    impl_->shmReqH = height;
    impl_->shmResizeDirty = true;
    return {};
}

void Viewport::close()
{
    if (impl_ != nullptr)
        impl_->wantClose = true;
    running_ = false;
    if (thread_.joinable())
        thread_.join();
#if defined (__APPLE__)
    if (impl_ != nullptr && impl_->macPrecreatedWindow != nullptr)
    {
        // The render thread detached this context before returning. Cocoa
        // requires destruction and glfwTerminate on this main RPC thread.
        glfwMakeContextCurrent (impl_->macPrecreatedWindow);
        arbitgpu::invalidateNativeSdfExecutionContext (
            reinterpret_cast<std::uintptr_t> (impl_->macPrecreatedWindow));
        videoopticalflow::invalidateNativeSourceUploads (
            reinterpret_cast<std::uintptr_t> (impl_->macPrecreatedWindow));
        arbitgpu::invalidateNativeOpticalFlowExecutionContext (
            reinterpret_cast<std::uintptr_t> (impl_->macPrecreatedWindow));
        glfwMakeContextCurrent (nullptr);
        glfwDestroyWindow (impl_->macPrecreatedWindow);
        impl_->macPrecreatedWindow = nullptr;
        glfwTerminate();
    }
#endif
    impl_.reset();
}

void Viewport::processWindowEvents()
{
#if defined (__APPLE__)
    if (! running_.load() || impl_ == nullptr || impl_->sharedMode || impl_->shmMode
        || impl_->macPrecreatedWindow == nullptr)
        return;

    auto* win = impl_->macPrecreatedWindow;
    glfwPollEvents();
    if (glfwWindowShouldClose (win))
    {
        impl_->wantClose = true;
        running_ = false;
        return;
    }
    if (impl_->boundsDirty.exchange (false))
    {
        std::lock_guard<std::mutex> lock (impl_->mutex);
        glfwSetWindowPos (win, impl_->reqX, impl_->reqY);
        glfwSetWindowSize (win, impl_->reqW, impl_->reqH);
    }
    const int fsReq = impl_->fullscreenRequest.exchange (-1);
    if (fsReq == 1)
    {
        auto* mon = glfwGetPrimaryMonitor();
        const auto* mode = glfwGetVideoMode (mon);
        glfwSetWindowMonitor (win, mon, 0, 0, mode->width, mode->height, mode->refreshRate);
    }
    else if (fsReq == 0)
    {
        std::lock_guard<std::mutex> lock (impl_->mutex);
        glfwSetWindowMonitor (win, nullptr, impl_->reqX, impl_->reqY,
                              impl_->reqW, impl_->reqH, 0);
    }
#endif
}

void Viewport::setBounds (int x, int y, int width, int height)
{
    if (impl_ == nullptr) return;
    std::lock_guard<std::mutex> lock (impl_->mutex);
    impl_->reqX = x; impl_->reqY = y; impl_->reqW = width; impl_->reqH = height;
    impl_->boundsDirty = true;
}

void Viewport::setFullscreen (bool fullscreen)
{
    if (impl_ == nullptr) return;
    impl_->fullscreenRequest = fullscreen ? 1 : 0;
}

std::string Viewport::setTimeline (videowire::ResolvedVisualSnapshot snapshot)
{
    if (impl_ == nullptr) return "viewport is not open";
    const bool hasOpticalFlow = std::any_of (snapshot.visualLayerPlans.begin(),
        snapshot.visualLayerPlans.end(), [&] (const auto& plan)
        { return videowire::isOpticalFlowPlan (snapshot.visualLayerPlans, plan.clipId); });
    std::string diagnostic;
    if (hasOpticalFlow && ! videowire::validateOpticalFlowSources (
            snapshot.segments, snapshot.visualLayerPlans, snapshot.opticalFlowSources,
            snapshot.snapshotGeneration, diagnostic))
        return diagnostic;
    std::sort (snapshot.segments.begin(), snapshot.segments.end(),
               [] (const ViewportSegment& a, const ViewportSegment& b)
               { return a.displayStartSec < b.displayStartSec; });

    std::shared_ptr<videowire::VisualPlanExecutionSnapshot> publication;
    const std::unique_lock<std::mutex> canvasLock (canvasMutex_);
    const CanvasExtent canvas { canvasW_, canvasH_, canvasGeneration_ };
    arbitgl::GpuCaps gpuCaps;
    {
        const std::lock_guard<std::mutex> lock (impl_->mutex);
        gpuCaps = impl_->gpuCaps;
    }
    videowire::VisualBackendResourceLimits::Capabilities resourceCapabilities;
    resourceCapabilities.backendProfile = gpuCaps.glMajor > 0
        ? "opengl-" + std::to_string(gpuCaps.glMajor) + "."
            + std::to_string(gpuCaps.glMinor)
        : (gpuCaps.maximumBufferLengthBytes > 0
               || gpuCaps.recommendedWorkingSetBytes > 0
            ? "metal-renderer-device" : "metal-unqueried");
    resourceCapabilities.backendDeviceIdentity = gpuCaps.deviceIdentity;
    resourceCapabilities.maximumImageDimension = gpuCaps.maxTextureSize;
    resourceCapabilities.combinedTextureImageUnits = gpuCaps.maxCombinedTextureImageUnits;
    resourceCapabilities.maximumBufferLengthBytes = gpuCaps.maximumBufferLengthBytes;
    resourceCapabilities.recommendedWorkingSetBytes = gpuCaps.recommendedWorkingSetBytes;
    resourceCapabilities.maximumImageDimensionQuery = gpuCaps.maxTextureSize > 0
        ? (gpuCaps.glMajor > 0 ? "OpenGL.GL_MAX_TEXTURE_SIZE"
                              : "Sokol.sg_query_limits.max_image_size_2d") : "";
    resourceCapabilities.recommendedWorkingSetQuery = gpuCaps.recommendedWorkingSetBytes > 0
        ? "Metal.MTLDevice.recommendedMaxWorkingSetSize" : "";
    const auto backendResourceLimits = videowire::VisualBackendResourceLimits::fromCapabilities(
        canvas.width, canvas.height, resourceCapabilities);
    if (! videowire::makeVisualPlanExecutionSnapshot(snapshot.visualLayerPlans, publication, diagnostic,
                                                     nullptr,
                                                     canvas.width, canvas.height, &backendResourceLimits))
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        ++impl_->failedPlanLowerings;
        impl_->lastPlanLoweringError = diagnostic;
        impl_->visualTelemetry.recordFailedLowering(diagnostic);
        return diagnostic;
    }

    // The prior published plan belongs to a different project snapshot. Drop
    // its process-owned SDF geometry before publication replaces that identity.
    videohelper::sdf::nativeSdfRenderer().invalidateCompiledGeometry();
    std::lock_guard<std::mutex> lock (impl_->mutex);
    impl_->segments = std::move (snapshot.segments);
    impl_->visualEventSchedules = std::move (snapshot.visualEventSchedules);
    impl_->parameterTimeline = std::move (snapshot.parameterTimeline);
    if (hasOpticalFlow)
    {
        snapshot.segments = impl_->segments;
        impl_->opticalFlowSnapshot = std::make_shared<const videowire::ResolvedVisualSnapshot> (
            std::move (snapshot));
    }
    else
        impl_->opticalFlowSnapshot.reset();
    impl_->opticalFlowInspection = nullptr;
    publication->state.setTelemetryOwner(impl_->visualTelemetry);
    videowire::publishVisualPlanExecutionSnapshotTelemetry(
        *publication, impl_->visualTelemetry);
    impl_->visualPlanPublication.replace(std::move(publication));
    impl_->lastPlanLoweringError.clear();
    ++impl_->timelineGeneration;
    return {};
}

std::string Viewport::setInspectionTarget (int clipId, uint64_t structuralRevision,
                                           int nodeId, int outputPort)
{
    if (impl_ == nullptr) return "viewport is not open";
    std::lock_guard<std::mutex> lock (impl_->mutex);
    std::string error;
    if (! videowire::validateVisualInspectionTarget(impl_->visualPlanPublication->plans,
            { clipId, structuralRevision, nodeId, outputPort }, error))
        return error;
    impl_->inspectionClipId = clipId;
    impl_->inspectionStructuralRevision = structuralRevision;
    impl_->inspectionNodeId = nodeId;
    impl_->inspectionOutputPort = outputPort;
    impl_->inspectionResourceReady = false;
    impl_->inspectionResourceHandle = 0;
    return {};
}

std::string Viewport::describeInspection() const
{
    if (impl_ == nullptr) return R"({"ready":false})";
    std::lock_guard<std::mutex> lock (impl_->mutex);
    nlohmann::json value {
        { "clipId", impl_->inspectionClipId },
        { "structuralRevision", impl_->inspectionStructuralRevision },
        { "nodeId", impl_->inspectionNodeId },
        { "outputPort", impl_->inspectionOutputPort },
        { "ready", impl_->inspectionResourceReady },
        { "resourceHandle", impl_->inspectionResourceHandle },
        { "generation", impl_->inspectionResourceGeneration },
        { "width", impl_->inspectionResourceWidth },
        { "height", impl_->inspectionResourceHeight },
        { "backend", impl_->inspectionResourceBackend },
        { "presentationBackend", videopreview::backendAvailability(
              shouldUseStrictMetalBackend(false), impl_->inspectionMetalAdmitted) },
        { "layout", static_cast<int>(impl_->inspectionPresentation.layout) },
        { "background", static_cast<int>(impl_->inspectionPresentation.background) },
        { "zoom", impl_->inspectionPresentation.zoom },
        { "panX", impl_->inspectionPresentation.panX },
        { "panY", impl_->inspectionPresentation.panY },
        { "split", impl_->inspectionPresentation.split }
    };
    value["opticalFlow"] = impl_->opticalFlowInspection;
    return value.dump();
}

std::string Viewport::setInspectionPresentation (videopreview::State state)
{
    if (impl_ == nullptr) return "viewport is not open";
    std::string error;
    if (! videopreview::normalize(state, error)) return error;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (! videopreview::presentationAdmitted(shouldUseStrictMetalBackend(false),
            impl_->inspectionMetalAdmitted, state, error))
        return error;
    impl_->inspectionPresentation = state;
    return {};
}

bool Viewport::installSnapshot (videowire::ResolvedVisualSnapshot snapshot,
                                std::string* errorOut)
{
    if (errorOut != nullptr) errorOut->clear();
    if (snapshot.authoringRevision == 0)
    {
        std::lock_guard<std::mutex> revisionLock (revisionMutex_);
        snapshot.authoringRevision = revisionLedger_.state().authoring + 1;
        if (! revisionLedger_.requestAuthoring (snapshot.authoringRevision))
        {
            if (errorOut != nullptr) *errorOut = "stale authoring revision";
            return false;
        }
        const auto authoringRevision = snapshot.authoringRevision;
        if (impl_ == nullptr)
        {
            // Nothing was installed, so this is not a rejection of the revision.
            if (errorOut != nullptr) *errorOut = "viewport is not open";
            return false;
        }
        const auto timelineError = setTimeline(std::move(snapshot));
        if (! timelineError.empty())
        {
            revisionLedger_.compileFailed (authoringRevision);
            if (errorOut != nullptr) *errorOut = timelineError;
            return false;
        }
        revisionLedger_.accept (authoringRevision);
        revisionLedger_.compileSucceeded (authoringRevision, true);
        acceptStructuralRevision (authoringRevision);
        return true;
    }
    if (! beginSnapshot (snapshot.authoringRevision))
    {
        if (errorOut != nullptr) *errorOut = "stale authoring revision";
        return false;
    }
    return completeSnapshot (std::move (snapshot), errorOut);
}

bool Viewport::beginSnapshot (uint64_t authoringRevision)
{
    std::lock_guard<std::mutex> revisionLock (revisionMutex_);
    if (authoringRevision == 0)
        authoringRevision = revisionLedger_.state().authoring + 1;
    return revisionLedger_.requestAuthoring (authoringRevision);
}

bool Viewport::completeSnapshot (videowire::ResolvedVisualSnapshot snapshot,
                                 std::string* errorOut)
{
    if (errorOut != nullptr) errorOut->clear();
    std::lock_guard<std::mutex> revisionLock (revisionMutex_);
    if (snapshot.authoringRevision != revisionLedger_.state().authoring)
    {
        if (errorOut != nullptr) *errorOut = "snapshot authoring revision is no longer current";
        return false;
    }
    const auto authoringRevision = snapshot.authoringRevision;
    if (impl_ == nullptr)
    {
        // Nothing was installed, so this is not a rejection of the revision.
        if (errorOut != nullptr) *errorOut = "viewport is not open";
        return false;
    }
    const auto timelineError = setTimeline(std::move(snapshot));
    if (! timelineError.empty())
    {
        revisionLedger_.compileFailed (authoringRevision);
        if (errorOut != nullptr) *errorOut = timelineError;
        return false;
    }
    revisionLedger_.accept (authoringRevision);
    revisionLedger_.compileSucceeded (authoringRevision, true);
    acceptStructuralRevision (authoringRevision);
    return true;
}

bool Viewport::rejectSnapshot (uint64_t authoringRevision)
{
    std::lock_guard<std::mutex> lock (revisionMutex_);
    if (authoringRevision == 0)
        authoringRevision = revisionLedger_.state().authoring + 1;
    if (! revisionLedger_.requestAuthoring (authoringRevision))
        return false;
    return revisionLedger_.reject (authoringRevision);
}

videowire::RevisionTuple Viewport::revisionState() const
{
    std::lock_guard<std::mutex> lock (revisionMutex_);
    return revisionLedger_.state();
}

bool Viewport::advanceEvaluation (uint64_t sequence)
{
    std::lock_guard<std::mutex> lock (revisionMutex_);
    return revisionLedger_.evaluate (revisionLedger_.state().compiled, sequence);
}

void Viewport::acceptStructuralRevision (uint64_t revision) noexcept
{
    auto seen = structuralRevision_.load (std::memory_order_relaxed);
    while (seen < revision
           && ! structuralRevision_.compare_exchange_weak (
               seen, revision, std::memory_order_release, std::memory_order_relaxed)) {}
}

void Viewport::acceptRuntimeRevision (uint64_t revision) noexcept
{
    auto seen = runtimeRevision_.load (std::memory_order_relaxed);
    while (seen < revision
           && ! runtimeRevision_.compare_exchange_weak (
               seen, revision, std::memory_order_release, std::memory_order_relaxed)) {}
}

void Viewport::setScore (arbitmod::Score score, canonicalblockc::OwnerIdentity identity)
{
    if (impl_ == nullptr) return;
    std::lock_guard<std::mutex> lock (impl_->mutex);
    impl_->scoreSnapshot = std::make_shared<const arbitmod::Score>(std::move(score));
    impl_->scoreIdentity = identity;
    ++impl_->scoreGeneration;
}

void Viewport::setBlockCLifecycle (canonicalblockc::OwnerIdentity identity)
{
    if (impl_ == nullptr) return;
    std::lock_guard<std::mutex> lock (impl_->mutex);
    impl_->scoreIdentity = identity;
    ++impl_->scoreGeneration;
}

void Viewport::setModMatrix (std::vector<arbitmod::Routing> routings)
{
    if (impl_ == nullptr) return;
    std::lock_guard<std::mutex> lock (impl_->mutex);
    impl_->routings = std::move (routings);
    ++impl_->modGeneration;
}

void Viewport::setControlPlan (videocontrol::Plan plan)
{
    if (impl_ == nullptr) return;
    std::lock_guard<std::mutex> lock (impl_->mutex);
    impl_->controlPlan = std::move(plan);
    ++impl_->controlPlanGeneration;
}

bool Viewport::enqueueVisualEvent(std::string kind, int clipId,
                                  uint64_t sequence, uint64_t epoch)
{
    if (impl_ == nullptr || clipId < 0
        || (kind != "particleBurst" && kind != "generatorRetrigger"))
        return false;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (epoch < impl_->visualEventEpoch) return false;
    if (epoch > impl_->visualEventEpoch)
    {
        impl_->visualEvents.clear();
        impl_->visualEventLastSequences.clear();
        impl_->visualEventEpoch = epoch;
    }
    const auto key = std::make_pair(kind, clipId);
    if (const auto found = impl_->visualEventLastSequences.find(key);
        found != impl_->visualEventLastSequences.end() && sequence <= found->second)
        return true;
    if (impl_->visualEvents.size() >= 64) return false;
    impl_->visualEventLastSequences[key] = sequence;
    impl_->visualEvents.push_back({ std::move(kind), clipId, sequence, epoch });
    return true;
}

void Viewport::setBeatTimeline (videotime::BeatTimeline timeline)
{
    if (impl_ == nullptr) return;
    std::lock_guard<std::mutex> lock (impl_->mutex);
    impl_->beatTimeline = std::move(timeline);
    ++impl_->beatTimelineGeneration;
}

void Viewport::setScript (std::string source, std::string lang,
                          uint32_t cpuMs, uint32_t memoryMiB)
{
    if (impl_ == nullptr) return;
    std::lock_guard<std::mutex> lock (impl_->mutex);
    impl_->scriptSource = std::move (source);
    impl_->scriptLang   = (lang == "lua") ? "lua" : "js";
    impl_->scriptCpuMs = cpuMs;
    impl_->scriptMemoryMiB = memoryMiB;
    ++impl_->scriptGeneration;
}

void Viewport::attachTransport (const std::string& shmName)
{
    if (impl_ == nullptr) return;
    std::lock_guard<std::mutex> lock (impl_->mutex);
    impl_->transportName = shmName;
    impl_->transport.open (shmName);
}

void Viewport::attachAudio (const std::string& shmName)
{
    if (impl_ == nullptr) return;
    std::lock_guard<std::mutex> lock (impl_->mutex);
    impl_->audioName = shmName;
    impl_->audioRing.open (shmName);
}

std::string Viewport::setAudioMix (const std::string& wavPath,
                                  const std::vector<videohelper::AnalysisSourcePath>& sources)
{
    if (impl_ == nullptr) return "Viewport is not open";
    // Decode + offline-analyze OUTSIDE the lock (runs on the RPC thread, can take
    // a moment for a long mix); publish the result under the lock with a bumped
    // generation so the render thread picks it up. Empty path / failed decode ⇒
    // clear (stopped preview falls back to zero-feed). The exporter and this path
    // share videohelper::analyzeMixWavOffline ⇒ identical frames ⇒ stopped-preview
    // audio == export audio at the same timeline time.
    std::vector<arbitblockb::FeatureFrame> feats;
    double sr = 0.0;
    if (! wavPath.empty())
        feats = videohelper::analyzeMixWavOffline (wavPath, sr);
    std::string error;
    auto sourceFeats=videohelper::analyzeSourceMixes(sources,error);

    std::lock_guard<std::mutex> lock (impl_->mutex);
    impl_->previewMixFeats = std::move (feats);
    impl_->previewMixSr    = sr;
    impl_->previewSourceFeats=std::move(sourceFeats);
    ++impl_->previewMixGeneration;
    return error;
}

std::string Viewport::setCanvas (int width, int height)
{
    if (width <= 0 || height <= 0 || width > 32768 || height > 32768)
        return "bad canvas size";
    const std::unique_lock<std::mutex> canvasLock (canvasMutex_);
    if (canvasW_ == width && canvasH_ == height) return {};
    if (impl_ == nullptr)
    {
        canvasW_ = width;
        canvasH_ = height;
        ++canvasGeneration_;
        return {};
    }

    for (;;)
    {
        std::vector<videowire::CompiledVisualLayerPlan> plans;
        arbitgl::GpuCaps gpuCaps;
        uint64_t timelineGeneration = 0;
        {
            const std::lock_guard<std::mutex> lock (impl_->mutex);
            plans = impl_->visualPlanPublication->plans;
            gpuCaps = impl_->gpuCaps;
            timelineGeneration = impl_->timelineGeneration;
        }

        videowire::VisualBackendResourceLimits::Capabilities capabilities;
        capabilities.backendProfile = gpuCaps.glMajor > 0
            ? "opengl-" + std::to_string(gpuCaps.glMajor) + "."
                + std::to_string(gpuCaps.glMinor)
            : (gpuCaps.maximumBufferLengthBytes > 0
                   || gpuCaps.recommendedWorkingSetBytes > 0
                ? "metal-renderer-device" : "metal-unqueried");
        capabilities.backendDeviceIdentity = gpuCaps.deviceIdentity;
        capabilities.maximumImageDimension = gpuCaps.maxTextureSize;
        capabilities.combinedTextureImageUnits = gpuCaps.maxCombinedTextureImageUnits;
        capabilities.maximumBufferLengthBytes = gpuCaps.maximumBufferLengthBytes;
        capabilities.recommendedWorkingSetBytes = gpuCaps.recommendedWorkingSetBytes;
        capabilities.maximumImageDimensionQuery = gpuCaps.maxTextureSize > 0
            ? (gpuCaps.glMajor > 0 ? "OpenGL.GL_MAX_TEXTURE_SIZE"
                                  : "Sokol.sg_query_limits.max_image_size_2d") : "";
        capabilities.recommendedWorkingSetQuery = gpuCaps.recommendedWorkingSetBytes > 0
            ? "Metal.MTLDevice.recommendedMaxWorkingSetSize" : "";
        const auto limits = videowire::VisualBackendResourceLimits::fromCapabilities(
            width, height, capabilities);
        std::shared_ptr<videowire::VisualPlanExecutionSnapshot> publication;
        std::string diagnostic;
        const bool admitted = videowire::makeVisualPlanExecutionSnapshot(
            std::move(plans), publication, diagnostic, nullptr, width, height, &limits);

        const std::lock_guard<std::mutex> lock (impl_->mutex);
        const auto& currentCaps = impl_->gpuCaps;
        if (impl_->timelineGeneration != timelineGeneration
            || currentCaps.glMajor != gpuCaps.glMajor
            || currentCaps.glMinor != gpuCaps.glMinor
            || currentCaps.maxTextureSize != gpuCaps.maxTextureSize
            || currentCaps.maxCombinedTextureImageUnits
                != gpuCaps.maxCombinedTextureImageUnits
            || currentCaps.maximumBufferLengthBytes != gpuCaps.maximumBufferLengthBytes
            || currentCaps.recommendedWorkingSetBytes != gpuCaps.recommendedWorkingSetBytes
            || currentCaps.deviceIdentity != gpuCaps.deviceIdentity)
            continue;
        if (! admitted)
        {
            ++impl_->failedPlanLowerings;
            impl_->lastPlanLoweringError = diagnostic;
            impl_->visualTelemetry.recordFailedLowering(diagnostic);
            return diagnostic;
        }
        canvasW_ = width;
        canvasH_ = height;
        ++canvasGeneration_;
        publication->state.setTelemetryOwner(impl_->visualTelemetry);
        videowire::publishVisualPlanExecutionSnapshotTelemetry(
            *publication, impl_->visualTelemetry);
        impl_->visualPlanPublication.replace(std::move(publication));
        impl_->lastPlanLoweringError.clear();
        impl_->contentRevision.fetch_add (1); // the idle gate wakes for the new canvas
        return {};
    }
}

Viewport::CanvasExtent Viewport::canvasExtent() const
{
    const std::lock_guard<std::mutex> lock (canvasMutex_);
    return { canvasW_, canvasH_, canvasGeneration_ };
}

std::string Viewport::setCanvasBackground (double r, double g, double b, double a)
{
    if (bgR_ == r && bgG_ == g && bgB_ == b && bgA_ == a)
        return {};
    bgR_ = r; bgG_ = g; bgB_ = b; bgA_ = a;
    if (impl_ != nullptr)
        impl_->contentRevision.fetch_add (1); // the idle gate wakes for the new background
    return {};
}

std::string Viewport::setView (double zoom, double panX, double panY)
{
    if (! std::isfinite (zoom) || ! std::isfinite (panX) || ! std::isfinite (panY))
        return "bad view transform";
    viewZoom_ = std::clamp (zoom, 0.02, 32.0);
    viewPanX_ = panX;
    viewPanY_ = panY;
    return {};
}

std::string Viewport::setPostFx (double bloomIntensity, double bloomThreshold,
                                 double bloomRadius, int tonemap, double exposure)
{
    bloomIntensity_ = bloomIntensity > 0.0 ? bloomIntensity : 0.0;
    bloomThreshold_ = bloomThreshold;
    bloomRadius_    = bloomRadius > 0.0 ? bloomRadius : 0.0;
    tonemap_        = (tonemap >= 0 && tonemap <= 2) ? tonemap : 0;
    exposure_       = exposure > 0.0 ? exposure : 1.0;
    return {};
}

std::string Viewport::setParam (const std::string& paramId, double value, double atBeat)
{
    if (!std::isfinite(value) || !std::isfinite(atBeat))
        return "nonfinite graph parameter value or timestamp: " + paramId;
    // Viewport-only view transform (PROTOCOL.md §Project canvas & view
    // transform). Stored on the Viewport so values survive close/reopen and
    // can be set before the viewport opens. Never timestamped.
    if (paramId.rfind ("view/", 0) == 0)
    {
        const std::string p = paramId.substr (5);
        if (p == "zoom") return setView (value, viewPanX_.load(), viewPanY_.load());
        if (p == "panX") { viewPanX_ = value; return {}; }
        if (p == "panY") { viewPanY_ = value; return {}; }
        return "unknown view param: " + paramId;
    }

    if (impl_ == nullptr) return "viewport not open";
    int clipId = 0, textId = 0;
    std::string node, param, textParam;
    const bool isClip = Impl::parseParamId (paramId, clipId, node, param);
    const bool isText = ! isClip && Impl::parseTextParamId (paramId, textId, textParam);
    if (! isClip && ! isText)
        return "malformed paramId (expected clip<id>/<node>/<param>, text<id>/<param> or view/<param>): " + paramId;

    std::lock_guard<std::mutex> lock (impl_->mutex);
    if (isClip && node.rfind("visual", 0) == 0)
    {
        int visualNodeId = 0;
        if (!shadercatalog::parseRuntimeNodeAlias(node, visualNodeId))
            return "malformed visual runtime node: " + paramId;
        bool known = false;
        if (const auto* plan = videowire::findVisualLayerPlan(
                impl_->visualPlanPublication->plans, clipId))
            if (const auto* execution = impl_->visualPlanPublication->state.compiled(
                    clipId, plan->structuralRevision))
                if (const auto shaders = videowire::scheduledShaderPlan(execution))
                    for (const auto& operation : shaders->operations)
                        if (operation.nodeId == visualNodeId)
                            if (const auto* catalog = shadercatalog::find(
                                    operation.payload.catalogPackId,
                                    operation.payload.catalogProgramId))
                                known = shadercatalog::hasParameterComponent(*catalog, param);
        if (!known)
        {
            std::map<std::string, double> importedParameters;
            videohelper::importedscene::seedImportedSceneRuntimeParameters(
                impl_->visualPlanPublication->plans, clipId, importedParameters);
            known = importedParameters.find(node + "/" + param) != importedParameters.end();
            // A per-key morph delta has no seed, since the mesh's key count is unknown here. A
            // deformation root seeds selectedMorphWeight, so it admits its deltas. The native
            // evaluator refuses a key the mesh lacks.
            std::size_t deltaKey = 0;
            if (!known && visualanimation::parseMorphWeightDeltaParameter(param, deltaKey))
                known = importedParameters.find(node + "/selectedMorphWeight")
                    != importedParameters.end();
        }
        if (!known) return "unknown param: " + paramId;
    }
    if (atBeat >= 0.0)
    {
        impl_->pendingParams.push_back ({ paramId, value, atBeat });
        return {};
    }
    const bool ok = isClip ? impl_->apply (clipId, node, param, value)
                           : impl_->applyText (textId, textParam, value);
    if (! ok)
        return "unknown param: " + paramId;
    return {};
}

std::string Viewport::setTextImage (int textId, std::vector<uint8_t> rgba,
                                    int width, int height,
                                    double startSec, double durationSec,
                                    double posX, double posY,
                                    double opacity, double zOrder,
                                    int ownerClipId)
{
    if (impl_ == nullptr) return "viewport not open";
    if (width <= 0 || height <= 0 || width > 8192 || height > 8192)
        return "bad text image size";
    if (rgba.size() < (size_t) width * (size_t) height * 4)
        return "text image pixel buffer too small";

    std::lock_guard<std::mutex> lock (impl_->mutex);
    auto& t = impl_->texts[textId];
    t.removed = false;
    t.pendingPixels = std::move (rgba);
    t.pendingW = width;
    t.pendingH = height;
    t.pixelsDirty = true;
    t.startSec = startSec;
    t.durationSec = durationSec;
    t.ownerClipId = ownerClipId;
    t.posX = (float) posX;
    t.posY = (float) posY;
    t.opacity = (float) std::clamp (opacity, 0.0, 1.0);
    t.zOrder = (float) zOrder;
    // translateX/translateY/visible are param-only and deliberately kept.
    return {};
}

std::string Viewport::removeText (int textId)
{
    if (impl_ == nullptr) return "viewport not open";
    std::lock_guard<std::mutex> lock (impl_->mutex);
    auto it = impl_->texts.find (textId);
    if (it == impl_->texts.end())
        return {}; // idempotent
    it->second.removed = true;     // render thread frees the texture + erases
    it->second.pixelsDirty = false;
    it->second.pendingPixels.clear();
    return {};
}

std::string Viewport::setEffects (int clipId, const std::vector<EffectSlotSpec>& slots)
{
    if (impl_ == nullptr) return "viewport not open";

    std::lock_guard<std::mutex> lock (impl_->mutex);
    auto* p = impl_->paramsFor (clipId);

    // Bulk replace: the call describes the whole rack.
    for (auto& e : p->effects)
        e = videorender::EffectSlotState {};

    for (const auto& s : slots)
    {
        if (s.slot < 0 || s.slot >= videofx::kMaxEffectSlots)
            return "bad slot index: " + std::to_string (s.slot);
        if (s.type < -1 || s.type >= videofx::kEffectTypeCount)
            return "bad effect type: " + std::to_string (s.type);
        auto& e = p->effects[s.slot];
        e.type = s.type;
        e.enabled = s.enabled && s.type >= 0;
        if (s.type < 0)
            continue;
        resetSlotParams (e);
        for (const auto& [name, value] : s.params)
        {
            const int idx = videofx::effectParamIndex (s.type, name.c_str());
            if (idx < 0)
                return "unknown param '" + name + "' for effect type "
                     + std::to_string (s.type);
            const auto& pd = videofx::kEffectDefs[s.type].params[idx];
            e.params[idx] = std::clamp ((float) value, pd.minValue, pd.maxValue);
        }
    }
    return {};
}

std::string Viewport::setLut (int clipId, const std::string& path)
{
    if (impl_ == nullptr) return "viewport not open";

    std::vector<float> data;
    int size = 0;
    if (! path.empty())
    {
        const std::string err = loadCubeLut (path, data, size);
        if (! err.empty())
            return err;
    }

    std::lock_guard<std::mutex> lock (impl_->mutex);
    auto& l = impl_->luts[clipId];
    l.path = path;
    l.pendingData = std::move (data);
    l.pendingSize = size;
    l.dirty = true;
    return {};
}

std::string Viewport::describeGraph() const
{
    json nodes = json::array();
    json texts = json::array();
    json segments = json::array();
    json visualLayerPlans = json::array();
    arbitgl::GpuCaps caps;
    const auto revisions = revisionState();
    const auto graphCanvas = canvasExtent();
    if (impl_ != nullptr)
    {
        std::lock_guard<std::mutex> lock (impl_->mutex);
        caps = impl_->gpuCaps;
        for (const auto& segment : impl_->segments)
            segments.push_back ({
                { "clipId", segment.clipId },
                { "sourceKind", videowire::sourceKindToWire (segment.sourceKind) },
                { "sourcePath", segment.sourcePath },
                { "trackLayer", segment.trackLayer },
                { "inSec", segment.inSec },
                { "outSec", segment.outSec },
                { "rate", segment.rate },
                { "displayStartSec", segment.displayStartSec },
                { "transitionType", segment.transitionType },
                { "transitionDurationSec", segment.transitionDurationSec },
                { "transitionFromClipId", segment.transitionFromClipId },
                { "transitionToClipId", segment.transitionToClipId },
                { "matteCacheKey", segment.matteCacheKey },
                { "matteFps", segment.matteFps },
                { "matteFrames", segment.matteFrames },
            });
        for (const auto& plan : impl_->visualPlanPublication->plans)
        {
            videowire::VisualPlanResourceUsage resourceUsage;
            std::string resourceError;
            const bool resourceUsageAvailable = videowire::accountVisualPlanResources(
                plan, graphCanvas.width, graphCanvas.height, resourceUsage, resourceError);
            json edges = json::array();
            for (const auto& edge : plan.edges)
                edges.push_back({ { "fromNodeId", edge.fromNodeId }, { "fromPort", edge.fromPort },
                                  { "toNodeId", edge.toNodeId }, { "toPort", edge.toPort } });
            json ports = json::array();
            for (const auto& port : plan.ports)
                ports.push_back({ { "nodeId", port.nodeId }, { "port", port.port },
                                  { "channels", port.channels }, { "direction", port.direction },
                                  { "carrier", port.carrier }, { "dataType", port.dataType },
                                  { "pixelFormat", port.pixelFormat }, { "colorSpace", port.colorSpace },
                                  { "transfer", port.transfer }, { "alpha", port.alpha } });
            json operations = json::array();
            for (const auto& operation : plan.operations)
            {
                json encodedOperation = {
                    { "nodeId", operation.nodeId }, { "kind", operation.kind },
                    { "backendCapability", operation.backendCapability },
                    { "payloadXml", operation.payloadXml }
                };
                operations.push_back(std::move(encodedOperation));
            }
            visualLayerPlans.push_back ({
                { "clipId", plan.clipId },
                { "structuralRevision", plan.structuralRevision },
                { "descriptorCount", plan.descriptorCount },
                { "operationCount", plan.operationCount },
                { "sceneRecordCount", plan.sceneRecordCount },
                { "frameOutputCount", plan.frameOutputCount },
                { "peakLiveFrameCount", plan.peakLiveFrameCount },
                { "allocatedFrameSlotCount", plan.allocatedFrameSlotCount },
                { "allocatedFrameBytes", resourceUsageAvailable
                    ? resourceUsage.allocatedFrameBytes : uint64_t { 0 } },
                { "compileDurationMicros", plan.compileDurationMicros },
                { "resourceDiagnostic", resourceError },
                { "nodeKinds", plan.nodeKinds },
                { "nodeIds", plan.nodeIds },
                { "edges", std::move(edges) },
                { "ports", std::move(ports) },
                { "operations", std::move(operations) },
            });
        }
        for (const auto& [textId, t] : impl_->texts)
        {
            if (t.removed)
                continue;
            const std::string prefix = "text" + std::to_string (textId);
            texts.push_back ({
                { "textId", textId },
                { "width", t.pixelsDirty ? t.pendingW : t.texW },
                { "height", t.pixelsDirty ? t.pendingH : t.texH },
                { "startSec", t.startSec },
                { "durationSec", t.durationSec },
                { "ownerClipId", t.ownerClipId },
                { "posX", t.posX },
                { "posY", t.posY },
                { "params", {
                    { prefix + "/opacity", t.opacity },
                    { prefix + "/translateX", t.translateX },
                    { prefix + "/translateY", t.translateY },
                    { prefix + "/visible", t.visible },
                    { prefix + "/zOrder", t.zOrder },
                } },
            });
        }
        for (const auto& [clipId, p] : impl_->clipParams)
        {
            const std::string prefix = "clip" + std::to_string (clipId);
            json effects = json::array();
            for (int slot = 0; slot < videofx::kMaxEffectSlots; ++slot)
            {
                const auto& e = p.effects[slot];
                if (e.type < 0)
                    continue;
                json params = json::object();
                if (const auto* def = videofx::effectDefFor (e.type))
                    for (int i = 0; i < def->paramCount; ++i)
                        params[def->params[i].name] = e.params[i];
                effects.push_back ({
                    { "slot", slot },
                    { "type", e.type },
                    { "enabled", e.enabled },
                    { "params", params },
                });
            }
            // retimeQuality is a per-segment field (uniform across a clip's
            // slices); surface the first slice's tier under the retime stage.
            int retimeQuality = 0;
            for (const auto& s : impl_->segments)
                if (s.clipId == clipId) { retimeQuality = s.retimeQuality; break; }
            json node = {
                { "clipId", clipId },
                { "topology", { "source", "retime", "effects", "mask", "transform2d", "display" } },
                { "params", {
                    { prefix + "/source/opacity", p.opacity },
                    { prefix + "/source/visible", p.visible },
                    { prefix + "/source/zOrder", p.zOrder },
                    { prefix + "/source/blendMode", p.blendMode },
                    { prefix + "/retime/quality", retimeQuality },
                    { prefix + "/transform2d/scale", p.scale },
                    { prefix + "/transform2d/translateX", p.translateX },
                    { prefix + "/transform2d/translateY", p.translateY },
                    { prefix + "/transform2d/rotation", p.rotationDeg },
                    { prefix + "/transform2d/cropLeft", p.cropLeft },
                    { prefix + "/transform2d/cropRight", p.cropRight },
                    { prefix + "/transform2d/cropTop", p.cropTop },
                    { prefix + "/transform2d/cropBottom", p.cropBottom },
                    { prefix + "/mask/type", p.maskType },
                    { prefix + "/mask/cx", p.maskCx },
                    { prefix + "/mask/cy", p.maskCy },
                    { prefix + "/mask/w", p.maskW },
                    { prefix + "/mask/h", p.maskH },
                    { prefix + "/mask/feather", p.maskFeather },
                    { prefix + "/mask/invert", p.maskInvert },
                } },
                { "effects", effects },
            };
            if (const auto it = impl_->luts.find (clipId);
                it != impl_->luts.end() && ! it->second.path.empty())
            {
                node["lutPath"] = it->second.path;
                node["lutSize"] = it->second.dirty ? it->second.pendingSize
                                                   : it->second.size;
            }
            nodes.push_back (std::move (node));
        }
    }
    const json canvas = {
        { "width", graphCanvas.width },
        { "height", graphCanvas.height },
        { "viewZoom", viewZoom_.load() },
        { "viewPanX", viewPanX_.load() },
        { "viewPanY", viewPanY_.load() },
    };
    // P1: report the GPU capability snapshot so agents/UI can gate
    // compute-class features (and the export path can flag them unavailable).
    const json gpu = {
        { "glMajor", caps.glMajor },
        { "glMinor", caps.glMinor },
        { "computeShaders", caps.computeShaders },
        { "particles", caps.particles },
        { "ssbo", caps.ssbo },
        { "imageLoadStore", caps.imageLoadStore },
        { "maxTextureSize", caps.maxTextureSize },
        { "maxCombinedTextureImageUnits", caps.maxCombinedTextureImageUnits },
        { "maxComputeWorkGroupCount", { caps.maxComputeWorkGroupCount[0],
                                        caps.maxComputeWorkGroupCount[1],
                                        caps.maxComputeWorkGroupCount[2] } },
        { "maxComputeWorkGroupSize", { caps.maxComputeWorkGroupSize[0],
                                       caps.maxComputeWorkGroupSize[1],
                                       caps.maxComputeWorkGroupSize[2] } },
        { "maxComputeWorkGroupInvocations", caps.maxComputeWorkGroupInvocations },
    };
    return json { { "nodes", nodes }, { "texts", texts }, { "segments", segments },
                  { "visualLayerPlans", visualLayerPlans },
                  { "canvas", canvas }, { "gpu", gpu },
                  { "structuralRevision", structuralRevision_.load (std::memory_order_acquire) },
                  { "runtimeRevision", runtimeRevision_.load (std::memory_order_acquire) },
                  { "authoringRevision", revisions.authoring },
                  { "acceptedRevision", revisions.accepted },
                  { "rejectedRevision", revisions.rejected },
                  { "compiledRevision", revisions.compiled },
                  { "lastGoodRevision", revisions.lastGood },
                  { "exportableRevision", revisions.exportable },
                  { "evaluationSequence", revisions.evaluation } }.dump();
}

ViewportInfo Viewport::info() const
{
    ViewportInfo vi;
    const auto canvas = canvasExtent();
    vi.canvasWidth = canvas.width;
    vi.canvasHeight = canvas.height;
    vi.viewZoom = viewZoom_.load();
    vi.viewPanX = viewPanX_.load();
    vi.viewPanY = viewPanY_.load();
    if (impl_ == nullptr) return vi;
    std::lock_guard<std::mutex> lock (impl_->mutex);
    vi.open = impl_->windowOpen.load();
    vi.width = impl_->curW;
    vi.height = impl_->curH;
    vi.measuredFps = impl_->measuredFps;
    vi.avOffsetSec = impl_->avOffsetSec;
    const int ib = impl_->interpBackend.load();
    vi.interpolationActive = ib != 0;
    vi.interpolationBackend = ib == 1 ? "frame-blend"
                            : ib == 2 ? "rife-cuda"
                            : ib == 3 ? "rife-coreml"
                            : ib == 4 ? "rife-vulkan"
                            : "nearest";
#if ARBIT_HAVE_ONNX || ARBIT_HAVE_NCNN
    if (impl_->interp != nullptr)
    {
        vi.interpReady = impl_->interp->available();
        vi.interpWorkerBackend = impl_->interp->backend();
        vi.interpInferences = impl_->interp->inferenceCount();
        vi.interpCacheHits = impl_->interp->cacheHits();
        vi.interpCacheMisses = impl_->interp->cacheMisses();
        vi.interpError = impl_->interp->lastError();
    }
#endif
    vi.interpTargetFps = interpTargetFps_.load (std::memory_order_relaxed);
    vi.gpuPath = impl_->gpuPath;
    vi.backendPolicy = impl_->backendPolicy;
    vi.compositorBackend = impl_->compositorBackend;
    vi.particleBackend = impl_->particleBackend;
    vi.rendererError = impl_->rendererError;
    vi.importedScenePending = impl_->importedScenePending.load();
    vi.fallbackCount = impl_->fallbackCount;
    vi.framesPresented = impl_->framesPresented.load();
    vi.lastFrameHash = impl_->lastFrameHash.load();
    vi.acceptedScoreGeneration = impl_->scoreGeneration;
    vi.presentedScoreGeneration = impl_->presentedScoreGeneration;
    vi.publishedDisplaySec = impl_->publishedDisplaySec;
    vi.publishedScoreRevision = impl_->publishedScoreRevision;
    vi.publishedCanonicalNoteRows = impl_->publishedCanonicalNoteRows;
    vi.publishedImportedSceneCount = impl_->publishedImportedSceneCount;
    vi.publishedImportedSceneSubmittedInstances = impl_->publishedImportedSceneSubmittedInstances;
    vi.publishedImportedSceneSubmittedNoteInstances = impl_->publishedImportedSceneSubmittedNoteInstances;
    vi.displaySec = impl_->displaySec;
    vi.sourcePtsSec = impl_->sourcePtsSec;
    vi.sourceIdealSec = impl_->sourceIdealSec;
    vi.transportOpen = impl_->transportOpen;
    vi.transportPlaying = impl_->transportPlaying;
    vi.transportGeneration = impl_->transportGeneration;
    vi.transportAgeSec = impl_->transportAgeSec;
    vi.transportPlayheadBeats = impl_->transportPlayheadBeats;
#if ARBIT_HAVE_GPU_SHARED
    const char* sharedPath = gpuexp::kSharedPathName;
#else
    const char* sharedPath = "dmabuf-docked"; // unreachable: sharedMode never set
#endif
    vi.presentPath = ! vi.open ? "none"
                   : impl_->sharedMode ? sharedPath
                   : impl_->shmMode ? "shm-docked" : "glfw-window";
    {
        const int phase = std::clamp (impl_->loopPhase.load (std::memory_order_relaxed),
                                      0, static_cast<int> (LoopPhase::Count) - 1);
        const int64_t enterNs = impl_->loopPhaseEnterNs.load (std::memory_order_relaxed);
        vi.loopPhase = kLoopPhaseNames[phase];
        vi.loopPhaseAgeMs = enterNs > 0 ? (double) (nowNs() - enterNs) * 1e-6 : 0.0;
        const char* step = impl_->loopStep.load (std::memory_order_relaxed);
        const int64_t stepNs = impl_->loopStepEnterNs.load (std::memory_order_relaxed);
        vi.loopStep = step;
        vi.loopStepAgeMs = stepNs > 0 ? (double) (nowNs() - stepNs) * 1e-6 : 0.0;
        vi.loopTicks = impl_->loopTicks.load (std::memory_order_relaxed);
    }
    vi.dmabufCapable = impl_->dmabufCapable.load();
    vi.sharedFramesSent = impl_->sharedFramesSent.load();
    vi.sharedFramesDroppedNoBuffer = impl_->sharedFramesDroppedNoBuffer.load();
    vi.sharedReleasesReceived = impl_->sharedReleasesReceived.load();
    vi.sharedReleasesInvalid = impl_->sharedReleasesInvalid.load();
    vi.sharedFreeBuffers = impl_->sharedFreeBuffers.load();
    vi.sharedBusyBuffers = impl_->sharedBusyBuffers.load();
    vi.gpuCaps = impl_->gpuCaps;
    vi.graphTelemetry = impl_->visualTelemetry.snapshot();
    vi.graphTelemetry.failedLowerings = impl_->failedPlanLowerings;
    vi.graphTelemetry.lastLoweringError = impl_->lastPlanLoweringError;
    return vi;
}

uint64_t Viewport::frameHash() const
{
    return impl_ != nullptr ? impl_->lastFrameHash.load() : 0;
}

void Viewport::setInterpolationSuspended (bool suspended)
{
    // Viewport-level atomic (not impl_): safe to call from the export worker
    // thread even as close() resets impl_.
    interpSuspended_.store (suspended, std::memory_order_relaxed);
}

void Viewport::setInterpTargetFps (double fps)
{
    // Viewport-level atomic (not impl_): survives close/reopen. Clamp junk to 0
    // (= legacy slowed-only gate). The render loop reads it per frame.
    interpTargetFps_.store (fps > 0.0 ? fps : 0.0, std::memory_order_relaxed);
}

void Viewport::setInterpMaxLongSide (int px)
{
    interpMaxLongSide_.store (px > 0 ? px : 0, std::memory_order_relaxed);
}

void Viewport::setValueFps (double fps)
{
    // Live project value-grid fps update (frame-perfect parity). impl_->valueFps
    // is atomic; the render loop re-reads it each frame. No-op when closed.
    if (impl_ != nullptr)
        impl_->valueFps.store (fps, std::memory_order_relaxed);
}

void Viewport::setScopeMask (uint32_t mask)
{
    scopeMask_ = mask & (kScopeWaveform | kScopeVectorscope | kScopeHistogram);
}

void Viewport::setVisible (bool visible)
{
    if (impl_ == nullptr)
        return; // closed: the next open starts visible and the plugin re-sends this
    // Only a real flip moves the revision: a repeated announce changes nothing.
    if (impl_->visible.exchange (visible) != visible)
        impl_->contentRevision.fetch_add (1);
}

void Viewport::requestFrame()
{
    if (impl_ != nullptr)
        impl_->contentRevision.fetch_add (1);
}

void Viewport::markContentChanged()
{
    if (impl_ != nullptr)
        impl_->contentRevision.fetch_add (1);
}

uint32_t Viewport::scopeMask() const
{
    return scopeMask_.load();
}

std::string Viewport::scopeDataJson() const
{
    const uint32_t mask = scopeMask_.load();
    json enabled = json::array();
    if (mask & kScopeWaveform)    enabled.push_back ("waveform");
    if (mask & kScopeVectorscope) enabled.push_back ("vectorscope");
    if (mask & kScopeHistogram)   enabled.push_back ("histogram");

    json out = {
        { "open", impl_ != nullptr && impl_->windowOpen.load() },
        { "enabled", enabled },
        { "seq", 0 },
        { "sampleWidth", videorender::FrameRenderer::kScopeW },
        { "sampleHeight", videorender::FrameRenderer::kScopeH },
        { "samples", videorender::FrameRenderer::kScopeW
                         * videorender::FrameRenderer::kScopeH },
    };
    if (impl_ == nullptr)
        return out.dump();

    std::lock_guard<std::mutex> lock (impl_->mutex);
    out["seq"] = impl_->scopeSeq;
    if (impl_->scopeSeq == 0)
        return out.dump();

    if ((mask & kScopeWaveform)
        && impl_->scopeWaveform.size() == (size_t) kScopeWaveW * kScopeWaveH)
    {
        out["waveform"] = base64Encode (impl_->scopeWaveform.data(),
                                        impl_->scopeWaveform.size());
        out["waveformWidth"] = kScopeWaveW;
        out["waveformHeight"] = kScopeWaveH;
    }
    if ((mask & kScopeVectorscope)
        && impl_->scopeVector.size() == (size_t) kScopeVecN * kScopeVecN)
    {
        out["vectorscope"] = base64Encode (impl_->scopeVector.data(),
                                           impl_->scopeVector.size());
        out["vectorscopeSize"] = kScopeVecN;
    }
    if (mask & kScopeHistogram)
    {
        json r = json::array(), g = json::array(), b = json::array();
        for (int i = 0; i < 256; ++i)
        {
            r.push_back (impl_->scopeHistogram[0][i]);
            g.push_back (impl_->scopeHistogram[1][i]);
            b.push_back (impl_->scopeHistogram[2][i]);
        }
        out["histogram"] = { { "r", r }, { "g", g }, { "b", b } };
    }
    return out.dump();
}

void Viewport::renderLoop (int width, int height, int x, int y,
                           bool alwaysOnTop, double targetFps)
{
    auto& im = *impl_;
    // This thread owns the advance sub-step atomics, so loopstep::Scope names
    // the inner call there. The guard restores the previous sink on every return.
    loopstep::Sink loopSink { &im.loopStep, &im.loopStepEnterNs };
    loopstep::SinkInstall loopSinkInstall (&loopSink);
    const bool shared = im.sharedMode;
    const bool shm = im.shmMode;
    const bool offscreen = shared || shm; // hidden window, no swapchain pacing
    const bool metalOnly = shouldUseStrictMetalBackend (shared);

    GLFWwindow* win = nullptr;
    bool windowOwnedByRenderThread = true;
#if defined (__APPLE__)
    if (im.macPrecreatedWindow != nullptr)
    {
        win = im.macPrecreatedWindow;
        windowOwnedByRenderThread = false;
    }
#endif
    if (win == nullptr)
    {
        if (! initializeHelperGlfw())
        {
            im.openError = "glfwInit failed (no display?)";
            im.openDone = true;
            return;
        }
        configureGlfwWindowHints (offscreen, shared, alwaysOnTop, metalOnly);
        win = glfwCreateWindow (width, height, "Arbit Video", nullptr, nullptr);
    }
    if (win == nullptr)
    {
        if (metalOnly)
            im.openError = "glfwCreateWindow failed for the strict Metal viewport";
        else
            im.openError = shared
                ? "glfwCreateWindow failed (EGL GL 4.3 context unavailable?)"
                : "glfwCreateWindow failed (GL 4.3 core unavailable?)";
        glfwTerminate();
        im.openDone = true;
        return;
    }
    if (! offscreen && x >= 0 && y >= 0)
        glfwSetWindowPos (win, x, y);
    if (! metalOnly)
    {
        glfwMakeContextCurrent (win);
        glfwSwapInterval (offscreen ? 0 : 1); // window mode: vsync paces the loop
    }

    const auto releaseWindow = [&]
    {
        videohelper::geometry::releaseGeometryExecutionCache();
        if (! metalOnly)
            glfwMakeContextCurrent (nullptr);
        if (windowOwnedByRenderThread)
        {
            if (! metalOnly)
            {
                glfwMakeContextCurrent (win);
                arbitgpu::invalidateNativeSdfExecutionContext (
                    reinterpret_cast<std::uintptr_t> (win));
                videoopticalflow::invalidateNativeSourceUploads (
                    reinterpret_cast<std::uintptr_t> (win));
                arbitgpu::invalidateNativeOpticalFlowExecutionContext (
                    reinterpret_cast<std::uintptr_t> (win));
            }
            glfwDestroyWindow (win);
            glfwTerminate();
        }
    };

    arbitgl::GlFuncs gl;
    std::string missing;
    if (! metalOnly && ! arbitgl::loadGlFunctions (gl, missing))
    {
        im.openError = "GL core loader failed, missing: " + missing;
        releaseWindow();
        im.openDone = true;
        return;
    }
    // P1: soft-load the optional 4.3 entry points and snapshot the caps once.
    if (! metalOnly)
    {
        arbitgl::loadGl43Functions (gl);
        arbitgl::GpuCaps caps;
        arbitgl::queryGpuCaps (gl, caps);
        std::lock_guard<std::mutex> lock (im.mutex);
        im.gpuCaps = caps;
    }
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    else
    {
        // The wire struct predates the native backend and retains GL-shaped
        // field names. In a Metal-only viewport these booleans describe the
        // equivalent production capabilities used by feature gates.
        arbitgl::GpuCaps caps;
        caps.computeShaders = true;
        caps.ssbo = true;
        caps.imageLoadStore = true;
        caps.particles = true;
        std::lock_guard<std::mutex> lock (im.mutex);
        im.gpuCaps = caps;
    }
#endif

    // Shrink an shm render size to what one ring slot can hold (aspect kept).
    auto clampToSlot = [&im] (int& w, int& h)
    {
        const auto* hdr = im.shmRing.header();
        if (hdr == nullptr)
            return;
        const double need = (double) w * (double) h * 4.0;
        if (need <= (double) hdr->slotBytes)
            return;
        const double s = std::sqrt ((double) hdr->slotBytes / need);
        w = std::max (16, (int) ((double) w * s));
        h = std::max (16, (int) ((double) h * s));
    };

    videorender::FrameRenderer renderer;

    std::unique_ptr<videohelper::modelpayload::ImportedAnimatedScenePayloadExecution>
        importedAnimationExecution;
    std::unique_ptr<videohelper::modelpayload::ImportedScenePayloadExecution>
        importedSceneExecution;
    videohelper::importedscene::VisualImportedScenePlanCache importedScenePlanCache;
    videohelper::geometry::ImportedGeometryExecution importedGeometryExecution(modelPayloadStore_);
    if (modelPayloadStore_ != nullptr)
    {
        importedAnimationExecution = std::make_unique<
            videohelper::modelpayload::ImportedAnimatedScenePayloadExecution>(
                *modelPayloadStore_, arbitgpu::nativeDeformationBackend());
        importedSceneExecution = std::make_unique<
            videohelper::modelpayload::ImportedScenePayloadExecution>(
                *modelPayloadStore_, arbitgpu::nativeFixtureSceneBackend(),
                arbitgpu::nativeDeformationBackend());
    }
    int fbW = 0, fbH = 0;
    if (offscreen)
    {
        fbW = width;
        fbH = height;
        if (shm)
            clampToSlot (fbW, fbH);
    }
    else
        glfwGetFramebufferSize (win, &fbW, &fbH);

    std::string rendererError;
    std::vector<videowire::CompiledVisualLayerPlan> initialPlans;
    {
        std::lock_guard<std::mutex> lock (im.mutex);
        initialPlans = im.visualPlanPublication->plans;
    }
    if (! renderer.initializeForVisualPlans (
            metalOnly ? nullptr : &gl, std::max (fbW, 1), std::max (fbH, 1),
            initialPlans, rendererError, metalOnly))
    {
        im.openError = "renderer init failed: " + rendererError;
        releaseWindow();
        im.openDone = true;
        return;
    }
    videowire::VisualBackendResourceLimits::Capabilities rendererCapabilities;
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalOnly)
    {
        rendererCapabilities.backendProfile = "metal-renderer-device";
        rendererCapabilities.maximumImageDimensionQuery =
            "Sokol.sg_query_limits.max_image_size_2d";
        rendererCapabilities.recommendedWorkingSetQuery =
            "Metal.MTLDevice.recommendedMaxWorkingSetSize";
        if (! renderer.queryMetalResourceCapabilities(
                rendererCapabilities.maximumImageDimension,
                rendererCapabilities.maximumBufferLengthBytes,
                rendererCapabilities.recommendedWorkingSetBytes,
                rendererCapabilities.backendDeviceIdentity))
        {
            im.openError = "renderer-owned Metal device capability query failed";
            renderer.shutdown();
            releaseWindow();
            im.openDone = true;
            return;
        }
        std::lock_guard<std::mutex> lock (im.mutex);
        im.gpuCaps.maxTextureSize = rendererCapabilities.maximumImageDimension;
        im.gpuCaps.maximumBufferLengthBytes = rendererCapabilities.maximumBufferLengthBytes;
        im.gpuCaps.recommendedWorkingSetBytes = rendererCapabilities.recommendedWorkingSetBytes;
        im.gpuCaps.deviceIdentity = rendererCapabilities.backendDeviceIdentity;
    }
    else
#endif
    {
        arbitgl::GpuCaps contextCaps;
        {
            std::lock_guard<std::mutex> lock (im.mutex);
            contextCaps = im.gpuCaps;
        }
        if (contextCaps.deviceIdentity.empty())
        {
            im.openError = "renderer-owned OpenGL device identity query failed";
            renderer.shutdown();
            releaseWindow();
            im.openDone = true;
            return;
        }
        rendererCapabilities.backendProfile = "opengl-"
            + std::to_string(contextCaps.glMajor) + "."
            + std::to_string(contextCaps.glMinor);
        rendererCapabilities.backendDeviceIdentity = contextCaps.deviceIdentity;
        rendererCapabilities.maximumImageDimension = contextCaps.maxTextureSize;
        rendererCapabilities.combinedTextureImageUnits =
            contextCaps.maxCombinedTextureImageUnits;
        rendererCapabilities.maximumImageDimensionQuery = "OpenGL.GL_MAX_TEXTURE_SIZE";
    }

    for (;;)
    {
        std::vector<videowire::CompiledVisualLayerPlan> plansForRendererDevice;
        uint64_t publicationGeneration = 0;
        {
            std::lock_guard<std::mutex> lock (im.mutex);
            plansForRendererDevice = im.visualPlanPublication->plans;
            publicationGeneration = im.timelineGeneration;
        }
        const auto canvas = canvasExtent();
        const auto rendererLimits = videowire::VisualBackendResourceLimits::fromCapabilities(
            canvas.width, canvas.height, rendererCapabilities);
        std::shared_ptr<videowire::VisualPlanExecutionSnapshot> rendererSnapshot;
        std::string rendererAdmissionError;
        const bool admitted = videowire::makeVisualPlanExecutionSnapshot(
                std::move(plansForRendererDevice), rendererSnapshot, rendererAdmissionError,
                nullptr, canvas.width, canvas.height, &rendererLimits);
        {
            const std::scoped_lock lock (im.mutex, canvasMutex_);
            if (admitted)
            {
                if (! videowire::publishVisualPlanExecutionSnapshotIfCurrent(
                        publicationGeneration, canvas.generation,
                        im.timelineGeneration, canvasGeneration_,
                        std::move(rendererSnapshot), im.visualPlanPublication.publicationRef(),
                        im.visualTelemetry))
                    continue;
                break;
            }
            if (im.timelineGeneration != publicationGeneration
                || canvasGeneration_ != canvas.generation)
                continue;
            im.openError = "renderer-owned plan admission failed: "
                + rendererAdmissionError;
        }
        renderer.shutdown();
        releaseWindow();
        im.openDone = true;
        return;
    }
    videowire::ViewportTelemetryOwner<> viewportTelemetry (im.visualTelemetry, metalOnly);
    renderer.setVisualTelemetryOwner(&im.visualTelemetry);
    const auto admittedBackend = metalOnly ? videowire::VisualBackend::metal
                                           : videowire::VisualBackend::openGL;
    if (! viewportTelemetry.admitBackend (admittedBackend))
    {
        im.openError = "viewport telemetry rejected production backend admission";
        renderer.shutdown();
        releaseWindow();
        im.openDone = true;
        return;
    }
    {
        std::lock_guard<std::mutex> lock (im.mutex);
        im.gpuPath = metalOnly ? "metal" : "opengl";
        im.backendPolicy = metalOnly ? "strict-metal" : "legacy-opengl";
        im.fallbackCount = 0;
    }

#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
    std::unique_ptr<MacMetalViewportSurface> localMetalSurface;
    if (metalOnly && ! shared)
    {
        localMetalSurface = std::make_unique<MacMetalViewportSurface>();
        if (! localMetalSurface->initialize (win, renderer.retainedMetalDevice(),
                                             fbW, fbH, ! offscreen, rendererError))
        {
            im.openError = "Metal viewport init failed: " + rendererError;
            renderer.shutdown();
            releaseWindow();
            im.openDone = true;
            return;
        }
    }
#endif

    bool linearSharedActive = false;
#if ARBIT_HAVE_GPU_SHARED
    // Shared-mode bring-up: capability probe, exported buffer ring, fd socket.
    auto destroySharedBuffers = [&im, &renderer, &viewportTelemetry]
    {
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
        renderer.clearMetalDirectOutputs();
#endif
        for (auto& b : im.buffers)
        {
            if (b.width > 0 && b.height > 0)
            {
                uint64_t allocationBytes = 0;
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
                if (b.surface != nullptr)
                    allocationBytes = IOSurfaceGetAllocSize (static_cast<IOSurfaceRef> (b.surface));
#endif
                viewportTelemetry.exportedBufferReleased (allocationBytes);
            }
            im.exporter.destroy (b);
        }
        im.buffers.clear();
        viewportTelemetry.exportedBufferPoolClosed();
    };
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
    gpusurf::LinearIOSurfaceNegotiation linearNegotiation;
    uint64_t sharedBufferGeneration = 0;
    gpusurf::LinearIOSurfaceOfferPayload linearOffer;
    linearOffer.metalRegistryId = renderer.metalDeviceRegistryId();
#endif
    auto allocateSharedBuffers = [&im, &viewportTelemetry, &linearSharedActive] (int w, int h, std::string& err) -> bool
    {
        im.buffers.resize ((size_t) im.sharedBufferCount);
        for (auto& b : im.buffers)
            if (! im.exporter.allocate (b, w, h, err
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
                , linearSharedActive ? gpuexp::IOSurfaceFormat::LinearRgba16Float
                                     : gpuexp::IOSurfaceFormat::Bgra8
#endif
                ))
                return false;
            else
            {
                uint64_t allocationBytes = 0;
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
                if (b.surface != nullptr)
                    allocationBytes = IOSurfaceGetAllocSize (static_cast<IOSurfaceRef> (b.surface));
#endif
                if (! viewportTelemetry.exportedBufferAllocated (allocationBytes))
                {
                    err = "exported buffer telemetry resource accounting overflow";
                    return false;
                }
            }
        return true;
    };
    auto sendBuffers = [&im, &linearSharedActive
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
                        , &sharedBufferGeneration, &linearOffer
#endif
                       ]
    {
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
        ++sharedBufferGeneration;
        if (linearSharedActive)
        {
            gpusurf::LinearIOSurfaceBuffersPayload list;
            list.declaration = linearOffer;
            list.generation = sharedBufferGeneration;
            list.bufferCount = static_cast<uint32_t>(im.buffers.size());
            std::vector<uint8_t> bytes(sizeof(list) + im.buffers.size() * sizeof(gpusurf::LinearIOSurfaceBufferInfo));
            std::memcpy(bytes.data(), &list, sizeof(list));
            for (size_t i = 0; i < im.buffers.size(); ++i)
            {
                const auto& b = im.buffers[i];
                const gpusurf::LinearIOSurfaceBufferInfo info {static_cast<uint32_t>(i),
                    static_cast<uint32_t>(b.modifier), static_cast<uint32_t>(b.width),
                    static_cast<uint32_t>(b.height), b.rowBytes, b.allocationBytes};
                std::memcpy(bytes.data() + sizeof(list) + i * sizeof(info), &info, sizeof(info));
            }
            im.fdSocket.sendMsg(static_cast<uint32_t>(gpusurf::MsgType::LinearIOSurfaceBuffers), bytes.data(), bytes.size());
            return;
        }
#endif
        // BUFFERS = BuffersPayload + N BufferInfo records; fds attached in
        // buffer order, planes within a buffer in plane order.
        std::vector<uint8_t> payload (sizeof (gpusurf::BuffersPayload)
                                      + im.buffers.size() * sizeof (gpusurf::BufferInfo));
        auto* bp = (gpusurf::BuffersPayload*) payload.data();
        bp->bufferCount = (uint32_t) im.buffers.size();
        auto* infos = (gpusurf::BufferInfo*) (payload.data() + sizeof (gpusurf::BuffersPayload));
        int fds[gpusurf::kMaxBuffers * gpusurf::kMaxPlanes];
        int fdCount = 0;
        for (size_t i = 0; i < im.buffers.size(); ++i)
        {
            const auto& b = im.buffers[i];
            auto& bi = infos[i];
            bi = gpusurf::BufferInfo {};
            bi.index = (uint32_t) i;
            bi.fourcc = b.fourcc;
            bi.width = (uint32_t) b.width;
            bi.height = (uint32_t) b.height;
            bi.modifier = b.modifier;
            bi.planeCount = (uint32_t) b.planeCount;
            for (int p = 0; p < b.planeCount; ++p)
            {
                bi.planes[p].strideBytes = b.strides[p];
                bi.planes[p].offsetBytes = b.offsets[p];
                fds[fdCount++] = b.fds[p];
            }
        }
        im.fdSocket.sendMsg ((uint32_t) gpusurf::MsgType::Buffers,
                             payload.data(), payload.size(), fds, fdCount);
    };
    auto sendHello = [&im
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
                      , &linearOffer
#endif
                     ]
    {
        gpusurf::HelloPayload hello;
        if (im.exporter.fenceSyncAvailable())
            hello.flags |= gpusurf::kFlagFenceSync;
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
        if (gpusurf::validLinearOffer(linearOffer)) hello.flags |= gpusurf::kFlagLinearIOSurfaceOffer;
#endif
        im.fdSocket.sendMsg ((uint32_t) gpusurf::MsgType::Hello, &hello, sizeof (hello));
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
        if (gpusurf::validLinearOffer(linearOffer))
            im.fdSocket.sendMsg(static_cast<uint32_t>(gpusurf::MsgType::LinearIOSurfaceOffer), &linearOffer, sizeof(linearOffer));
#endif
    };

    if (shared)
    {
        std::string err;
        if (! im.exporter.initialize (metalOnly ? nullptr : &gl, err))
        {
            im.openError = "no shared-GPU export support: " + err;
            renderer.shutdown();
            releaseWindow();
            im.openDone = true;
            return;
        }
        im.dmabufCapable = true;

        if (! im.fdSocket.listen (im.sharedResult.socketPath, err)
            || ! allocateSharedBuffers (fbW, fbH, err))
        {
            im.openError = err;
            destroySharedBuffers();
            im.fdSocket.close();
            im.exporter.shutdown();
            renderer.shutdown();
            releaseWindow();
            im.openDone = true;
            return;
        }
        im.sharedResult.fourcc = im.buffers[0].fourcc;
        im.sharedResult.modifier = im.buffers[0].modifier;
        im.sharedResult.bufferCount = (int) im.buffers.size();
        im.sharedResult.device = im.exporter.devicePath();
    }
#endif // ARBIT_HAVE_GPU_SHARED

    im.windowOpen = true;
    im.openDone = true;

    const double frameBudgetSec = targetFps > 0.0 ? 1.0 / targetFps : 1.0 / 60.0;
    std::deque<int64_t> frameTimes;
    DecodedFrame df; // decode scratch, reused across frames
#if ARBIT_HAVE_ONNX || ARBIT_HAVE_NCNN
    std::vector<uint8_t> interpScratch; // tier-2 fetch scratch (worker result -> GL upload)
#endif
    // Retime tier actually drawn this frame (reported via interpBackend):
    // 0=nearest, 1=frame-blend, 2=rife-cuda, 3=rife-coreml, 4=rife-vulkan.
    int interpTierThisFrame = 0;
    // Set when a Speed Warp layer fell back to Frame Blend while its RIFE frame
    // was still warming. The publish gate treats such a frame as unsettled.
    bool interpWarmingThisFrame = false;

    // Per-segment decode cursor: persistent decoder + GL texture per clipId,
    // so several segments (even of the same file) decode independently when
    // composited together.
    struct ClipStream
    {
        std::string path;
        std::unique_ptr<MediaContext> media;
        std::string matteDir;
        std::unique_ptr<MediaContext> matteMedia;
        std::unique_ptr<MediaContext> matteMediaB;
        bool matteOpenFailed = false;
        bool matteOpenFailedB = false;
        unsigned matteTex = 0;
        unsigned matteTexB = 0;
        videohelper::DepthTextureState depth;
        int matteTexW = 0, matteTexH = 0;
        int matteTexWB = 0, matteTexHB = 0;
        double matteLastSrcSec = -1.0e9;
        double matteLastSrcSecB = -1.0e9;
        std::string matteReceipt, matteReceiptB;
        unsigned tex = 0;
        int texW = 0, texH = 0;
        double lastSrcSec = -1.0e9; // pts of the uploaded frame
        decodedframecolor::Declaration frameColor;
        DecodedHdrFrame hdrFrame; // exact material source only; never an SDR graph texture
        bool openFailed = false;
        double seqFps = 0.0;        // image-sequence hint the stream was opened with
        int seqStart = -1;
        // Frame Blend (retime tier 1): two bracket-frame textures straddling
        // srcSec (texA=earlier pts, texB=later) + the blend output size. tex
        // holds the alpha-mixed result the compositor samples.
        unsigned texA = 0, texB = 0;
        double blendF0Pts = -1.0e9, blendF1Pts = -1.0e9;
        bool blendPairValid = false;
        int blendW = 0, blendH = 0;
        // tex currently holds a SYNTHESIZED frame (frame-blend or RIFE) that
        // represents srcSec exactly rather than a single decoded pts. Lets the
        // nearest path know it must re-decode if it takes over, while keeping
        // lastSrcSec = srcSec so the A/V-offset diagnostic stays honest (~0).
        bool texSynthesized = false;
    };
    std::map<int, ClipStream> streams; // render-thread only, keyed by clipId
    uint64_t streamsGeneration = ~0ull;
    auto frameVisualPlans = im.visualPlanPublication.shared(); // render-thread frame lease
    std::vector<videowire::VisualEventScheduleBinding> frameVisualScheduleBindings;
    videowire::VisualEventTriggerCursor visualEventCursor (false);
    const std::string eventCoveragePending = "Preview is retaining its last frame until coherent Event coverage is available";

    // One gathered layer = an active segment + a params snapshot (+ optional
    // transition A side).
    struct ActiveLayer
    {
        ViewportSegment seg;
        ClipGraphParams params;
        bool transitionActive = false;
        double transitionProgress = 0.0;
        bool haveFrom = false;
        ViewportSegment fromSeg;
        ClipGraphParams fromParams;
    };

    // Decode the frame for a segment at srcSec into its clip stream;
    // returns nullptr when the source cannot be opened / has no frame yet.
    auto decodeLayer = [&] (const ViewportSegment& s, double srcSec,
                            bool requireCurrentFrame = false, double dependencyFps = 0,
                            bool materialHdr = false) -> ClipStream*
    {
        // Generator clips (gen:// sentinel) carry no decodable media — their
        // pixels come from the shader/effect path, not a MediaContext. Skip
        // decode (matches the exporter). Live shader rendering in the viewport
        // is a later slice; for now the layer is simply empty here.
        if (videowire::resolveSourceKind (s.sourceKind, s.isAdjustment, s.sourcePath)
            != videowire::SourceKind::Media)
            return nullptr;

        auto& cs = streams[s.clipId];
        if (cs.path != s.sourcePath || cs.seqFps != s.sourceFps || cs.seqStart != s.seqStart)
        {
            renderer.deleteTexture(cs.matteTex);
            renderer.deleteTexture(cs.matteTexB);
            cs.matteTex = cs.matteTexB = 0;
            cs.depth.clear(renderer);
            cs.media.reset();
            cs.matteMedia.reset();
            cs.matteMediaB.reset();
            cs.path = s.sourcePath;
            cs.seqFps = s.sourceFps;
            cs.seqStart = s.seqStart;
            cs.openFailed = false;
            cs.matteOpenFailed = false;
            cs.matteOpenFailedB = false;
            cs.matteTexW = cs.matteTexH = 0;
            cs.matteTexWB = cs.matteTexHB = 0;
            cs.matteLastSrcSec = -1.0e9;
            cs.matteLastSrcSecB = -1.0e9;
            cs.texW = 0;
            cs.lastSrcSec = -1.0e9;
            cs.blendPairValid = false;
            cs.blendW = cs.blendH = 0;
        }
        if (cs.media == nullptr && ! cs.openFailed)
        {
            cs.media = std::make_unique<MediaContext>();
            if (! cs.media->open (s.sourcePath, true, s.sourceFps, s.seqStart).empty())
            {
                cs.media.reset();
                cs.openFailed = true;
            }
        }
        if (cs.media == nullptr)
            return nullptr;

        const double fps = std::max (cs.media->info().fps, 1.0);

        if (cs.matteDir != s.matteDir)
        {
            cs.matteDir = s.matteDir;
            cs.matteMedia.reset();
            cs.matteOpenFailed = false;
        }
        if (! s.matteDir.empty() && cs.matteMedia == nullptr && ! cs.matteOpenFailed)
        {
            cs.matteMedia = std::make_unique<MediaContext>();
            std::string validatedDir, pattern, matteError;
            const double matteFps = s.matteFps > 0.0 ? s.matteFps : fps;
            if (! videowire::revalidateMatteForOpen(s, validatedDir, pattern, matteError)
                || ! cs.matteMedia->open(pattern, false, matteFps, s.matteFirstFrame).empty())
            {
                cs.matteMedia.reset();
                cs.matteOpenFailed = true;
            }
        }
        // Should this segment be interpolated this frame? Two regimes:
        //   * Project target FPS set (interpTargetFps_ > 0): frame-rate
        //     up-conversion — interpolate whenever the segment's effective
        //     cadence (sourceFps * rate) is below the target, so a 24 fps clip in
        //     a 60 fps project is smoothed at ANY speed (mirrors the exporter
        //     trigger sourceFps * rate < job.fps).
        //   * No target (== 0): legacy slow-motion-only gate (rate < 0.999) — no
        //     behavior change until the user opts into a target.
        // The chosen engine is still per-clip retimeQuality (tier 2 RIFE / tier 1
        // Frame Blend / 0 = leave alone), gated below.
        const double interpTargetFps = interpTargetFps_.load (std::memory_order_relaxed);
        const bool belowTarget = dependencyFps > 0 ? fps * s.rate < dependencyFps * 0.999 : (interpTargetFps > 0.0)
                                     ? (fps * s.rate < interpTargetFps * 0.999)
                                     : (s.rate < 0.999);

        if (materialHdr && requireCurrentFrame)
        {
            cs.hdrFrame = {};
            if (std::isfinite(srcSec)
                && cs.media->getHdrFrame(srcSec,1920,1080,cs.hdrFrame).empty())
            {
                if (s.retimeQuality >= 1 && belowTarget)
                    return nullptr; // Float interpolation has no declared transfer-aware path.
                if (cs.hdrFrame.ptsTicks == AV_NOPTS_VALUE
                    || cs.hdrFrame.timeBase.num <= 0 || cs.hdrFrame.timeBase.den <= 0
                    || !std::isfinite(cs.hdrFrame.ptsSec)
                    || cs.hdrFrame.ptsSec > srcSec + 1.0e-6
                    || srcSec >= cs.hdrFrame.ptsSec + 1.0 / fps + 1.0e-6)
                    return nullptr;
                return &cs;
            }
        }

#if ARBIT_HAVE_ONNX || ARBIT_HAVE_NCNN
        // Speed Warp (retime tier 2): async RIFE. The render thread only reads
        // finished frames + posts requests; all decode/inference run on the
        // worker (its own decoder + RifeEngine, no GL context). On a cache miss
        // it falls through to Frame Blend (tier 1) below, so the picture is never
        // worse than tier 1 while the worker warms up. Suspended only while an
        // export holds the GPU (a resource gate, not a quality decision).
        //
        // NO fps-based auto-demote: the user's chosen tier is authoritative. We
        // never silently drop RIFE because the frame rate dipped — that's a
        // terrible surprise. (A cache miss still falls back to Frame Blend for the
        // single frame, then catches up; the *selected* tier stays tier 2.)
        if (s.retimeQuality >= 2 && belowTarget
            && ! interpSuspended_.load (std::memory_order_relaxed))
        {
            if (im.interp == nullptr)
            {
                // Publish the pointer under im.mutex (info() reads it under that
                // lock on the RPC thread); construct/start outside so the lock
                // never covers thread spawn or model build.
                auto engine = std::make_unique<arbitinterp::InterpEngine>();
                {
                    std::lock_guard<std::mutex> lk (im.mutex);
                    im.interp = std::move (engine);
                }
                im.interp->start();   // worker thread does the model build, off this thread
            }
            if (im.interp->available())
            {
                // Interpolate at the render-surface size, capped to
                // interpMaxLongSide_ (0 ⇒ engine default 360p). The source frame
                // is aspect-fit inside an ipw×iph box; we never oversample beyond
                // the panel. Higher cap = crisper preview, slower inference —
                // realtime 4K can't hold 60 (the EXPORT path is uncapped/full-res).
                int ipw = 0, iph = 0;
                if (const int cap = interpMaxLongSide_.load (std::memory_order_relaxed); cap > 0)
                {
                    const int eff = std::min (cap, std::max (width, height));
                    ipw = iph = std::max (16, eff);
                }
                const arbitinterp::InterpEngine::Request req {
                    s.clipId, s.sourcePath, s.sourceFps, s.seqStart, srcSec, ipw, iph };
                int rw = 0, rh = 0;
                if (im.interp->fetch (s.clipId, srcSec, interpScratch, rw, rh) && rw > 0)
                {
                    // Worker pixels do not currently publish a decoded colour
                    // receipt. Keep their transfer undeclared, never infer HDR.
                    cs.frameColor = {};
                    if (requireCurrentFrame) {
                        DecodedFrame declaration;
                        if (!cs.media->getFrame(srcSec,16,16,declaration).empty()) return nullptr;
                        if (!(declaration.color == decodedframecolor::Declaration{})) return nullptr;
                    }
                    cs.tex = renderer.uploadRgba (interpScratch.data(), rw, rh,
                                                  rw * 4, cs.tex);
                    cs.texW = rw;
                    cs.texH = rh;
                    cs.lastSrcSec = srcSec;        // synthesized frame == srcSec exactly
                    cs.texSynthesized = true;      // re-decode if nearest takes over
                    cs.blendPairValid = false;     // hand cs.tex back cleanly to tier 1
                    cs.blendW = cs.blendH = 0;     // force a resize-correct realloc there
                    interpTierThisFrame = im.interp->tierCode();   // 2 cuda / 3 coreml / 4 vulkan
                    im.interp->request (req);      // keep the bracket(s) around srcSec warm
                    return &cs;
                }
                im.interp->request (req);          // miss: warm it, fall to Frame Blend
            }
            // Blend stands in until the RIFE frame lands, so this frame is not final.
            interpWarmingThisFrame = true;
        }
#endif

        if (requireCurrentFrame && s.retimeQuality >= 2 && belowTarget)
            return nullptr; // Exact Surface evaluation cannot silently demote optical flow.

        // Frame Blend (retime tier 1): when the segment is slowed below the
        // display rate, GL alpha-mix the two source frames bracketing srcSec
        // (ports the exporter rifeDecodeLayer bracket machine — two getFrames +
        // forward-advance + rebuild-on-jump) instead of holding the nearest
        // frame. Any decode/seek failure leaves blendPairValid=false and falls
        // through to the nearest-frame path below.
        if (s.retimeQuality >= 1 && belowTarget)
        {
            const double frameDur = 1.0 / fps;
            // Next distinct frame after afterPts (steps widen for pts jitter);
            // rejects a size change so the bracket pair stays equally sized.
            auto decodeNextAfter = [&] (double afterPts, DecodedFrame& out,
                                        int w, int h) -> bool
            {
                for (const double k : { 1.05, 1.6, 2.3 })
                    if (cs.media->getFrame (afterPts + k * frameDur, 1920, 1080, out).empty()
                        && out.width > 0 && out.ptsSec > afterPts + 1e-6
                        && (w <= 0 || (out.width == w && out.height == h)))
                        return true;
                return false;
            };
            if (! cs.blendPairValid || srcSec < cs.blendF0Pts - 1e-6
                || srcSec > cs.blendF1Pts + 8.0 * frameDur)
            {
                // (Re)build the bracket pair around srcSec.
                cs.blendPairValid = false;
                if (cs.media->getFrame (srcSec, 1920, 1080, df).empty() && df.width > 0)
                {
                    const int w = df.width, h = df.height;
                    cs.texA = renderer.uploadRgba (df.rgba.data(), w, h,
                                                   df.strideBytes, cs.texA);
                    cs.blendF0Pts = df.ptsSec;
                    DecodedFrame df1;
                    if (decodeNextAfter (df.ptsSec, df1, w, h) && df.color == df1.color)
                    {
                        cs.frameColor = df.color;
                        cs.texB = renderer.uploadRgba (df1.rgba.data(), df1.width,
                                                       df1.height, df1.strideBytes, cs.texB);
                        cs.blendF1Pts = df1.ptsSec;
                        if (cs.blendW != w || cs.blendH != h)
                        {
                            if (cs.tex != 0) renderer.deleteTexture (cs.tex);
                            cs.tex = 0;
                            cs.blendW = w;
                            cs.blendH = h;
                        }
                        cs.texW = w;
                        cs.texH = h;
                        cs.blendPairValid = true;
                    }
                }
            }
            else
            {
                // Forward-advance the pair so it keeps straddling srcSec as slow
                // playback walks through source frames.
                int guard = 0;
                while (cs.blendPairValid && srcSec >= cs.blendF1Pts && guard++ < 16)
                {
                    cs.blendF0Pts = cs.blendF1Pts;
                    std::swap (cs.texA, cs.texB);   // old f1 becomes the new f0
                    DecodedFrame df1;
                    if (! decodeNextAfter (cs.blendF0Pts, df1, cs.blendW, cs.blendH) || !(df1.color == cs.frameColor))
                    {
                        cs.blendPairValid = false;  // clip tail: nearest covers it
                        break;
                    }
                    cs.texB = renderer.uploadRgba (df1.rgba.data(), df1.width,
                                                   df1.height, df1.strideBytes, cs.texB);
                    cs.blendF1Pts = df1.ptsSec;
                }
            }

            if (cs.blendPairValid && srcSec < cs.blendF1Pts && cs.blendW > 0)
            {
                const double span = std::max (1e-9, cs.blendF1Pts - cs.blendF0Pts);
                const float mix = (float) std::clamp (
                    (srcSec - cs.blendF0Pts) / span, 0.0, 1.0);
                cs.tex = renderer.frameBlendInto (cs.texA, cs.texB, cs.tex,
                                                  cs.blendW, cs.blendH, mix);
                cs.lastSrcSec = srcSec;  // blended frame == srcSec exactly
                cs.texSynthesized = true; // re-decode if nearest takes over
                interpTierThisFrame = 1;
                return &cs;
            }
        }

        // Nearest-frame retime (v1): skip the decode when the wanted time is
        // within half a frame of the uploaded frame's pts. A synthesized frame
        // (blend/RIFE) currently in cs.tex always forces a re-decode here.
        if (requireCurrentFrame || cs.texW == 0 || cs.texSynthesized
            || std::abs (srcSec - cs.lastSrcSec) >= 0.5 / fps)
        {
            if (cs.media->getFrame (srcSec, 1920, 1080, df).empty() && df.width > 0)
            {
                if (cs.matteMedia != nullptr
                    && ! videowire::visualPlanUsesTypedMatte(frameVisualPlans->plans, s.clipId))
                    applyAlphaMatte (df, *cs.matteMedia, srcSec);
                cs.tex = renderer.uploadRgba (df.rgba.data(), df.width, df.height,
                                              df.strideBytes, cs.tex);
                cs.texW = df.width;
                cs.texH = df.height;
                cs.frameColor = df.color;
                cs.lastSrcSec = df.ptsSec;
                cs.texSynthesized = false;
            }
            else if (requireCurrentFrame)
                return nullptr;
        }
        return cs.texW > 0 ? &cs : nullptr;
    };

    auto uploadTypedMatte = [&] (const ViewportSegment& segment, ClipStream& stream,
                                 double sourceSec) -> bool
    {
        if (! videowire::visualPlanUsesTypedMatte(frameVisualPlans->plans, segment.clipId))
            return true;
        if (stream.matteReceipt != segment.matteContentReceipt)
        {
            renderer.deleteTexture(stream.matteTex); stream.matteTex = 0;
            stream.matteMedia.reset(); stream.matteOpenFailed = false;
            stream.matteTexW = stream.matteTexH = 0;
            stream.matteReceipt = segment.matteContentReceipt;
            std::string dir, pattern, error;
            stream.matteMedia = std::make_unique<MediaContext>();
            if (!videowire::revalidateMatteForOpen(segment, dir, pattern, error)
                || !stream.matteMedia->open(pattern, false, segment.matteFps,
                                            segment.matteFirstFrame).empty())
            { stream.matteMedia.reset(); stream.matteOpenFailed = true; }
        }
        if (stream.matteMedia == nullptr) return false;
        videowire::RenderSegment secondary;
        const bool combined = videowire::visualPlanSecondaryMatte(
            frameVisualPlans->plans, segment.clipId, segment, secondary);
        const bool reusePrimary = combined && videowire::visualPlanReusesPrimaryMatte(
            frameVisualPlans->plans, segment.clipId);
        if (reusePrimary)
        {
            renderer.deleteTexture(stream.matteTexB); stream.matteTexB = 0;
            stream.matteMediaB.reset(); stream.matteOpenFailedB = false;
            stream.matteTexWB = stream.matteTexHB = 0;
            stream.matteReceiptB.clear();
        }
        if (combined && !reusePrimary && stream.matteReceiptB != secondary.matteContentReceipt)
        {
            renderer.deleteTexture(stream.matteTexB); stream.matteTexB = 0;
            stream.matteMediaB.reset(); stream.matteOpenFailedB = false;
            stream.matteTexWB = stream.matteTexHB = 0;
            stream.matteReceiptB = secondary.matteContentReceipt;
        }
        if (combined && !reusePrimary && stream.matteMediaB == nullptr && !stream.matteOpenFailedB)
        {
            std::string dir, pattern, error;
            stream.matteMediaB = std::make_unique<MediaContext>();
            if (!videowire::revalidateMatteForOpen(secondary, dir, pattern, error)
                || !stream.matteMediaB->open(pattern, false, secondary.matteFps,
                                             secondary.matteFirstFrame).empty())
            { stream.matteMediaB.reset(); stream.matteOpenFailedB = true; }
        }
        if (combined && !reusePrimary && stream.matteMediaB == nullptr) return false;
        const double fps = segment.matteFps > 0.0 ? segment.matteFps : 1.0;
        if (stream.matteTexW == 0
            || std::abs(sourceSec - stream.matteLastSrcSec) >= 0.5 / fps)
        {
            DecodedFrame matteFrame;
            if (! stream.matteMedia->getFrame(sourceSec, stream.texW, stream.texH, matteFrame).empty()
                || matteFrame.width <= 0 || matteFrame.height <= 0 || matteFrame.rgba.empty())
                return false;
            stream.matteTex = renderer.uploadRgba(matteFrame.rgba.data(), matteFrame.width,
                                                  matteFrame.height, matteFrame.strideBytes,
                                                  stream.matteTex);
            stream.matteTexW = matteFrame.width;
            stream.matteTexH = matteFrame.height;
            stream.matteLastSrcSec = matteFrame.ptsSec;
        }
        if (combined && !reusePrimary && (stream.matteTexWB == 0
            || std::abs(sourceSec - stream.matteLastSrcSecB) >= 0.5 / secondary.matteFps))
        {
            DecodedFrame frame;
            if (!stream.matteMediaB->getFrame(sourceSec, stream.texW, stream.texH, frame).empty()
                || frame.width <= 0 || frame.height <= 0 || frame.rgba.empty()) return false;
            stream.matteTexB = renderer.uploadRgba(frame.rgba.data(), frame.width, frame.height,
                                                   frame.strideBytes, stream.matteTexB);
            stream.matteTexWB = frame.width; stream.matteTexHB = frame.height;
            stream.matteLastSrcSecB = frame.ptsSec;
        }
        return stream.matteTex != 0 && (!combined || reusePrimary || stream.matteTexB != 0);
    };

    // Common per-clip params (everything except the source texture) — shared by
    // decoded-media layers and shader-generator layers (which carry no texture;
    // their pixels come from the compiled ShaderGenerator in renderComposite).
    auto fillDescCommon = [&im] (videorender::LayerDesc& d, const ClipGraphParams& p,
                                 double displaySec, int clipId)
    {
        d.clipId = clipId;             // keys the feedback-trail history texture
        d.scale = p.scale;
        d.translateX = p.translateX;
        d.translateY = p.translateY;
        d.rotationDeg = p.rotationDeg;
        d.cropLeft = p.cropLeft;
        d.cropRight = p.cropRight;
        d.cropTop = p.cropTop;
        d.cropBottom = p.cropBottom;
        d.opacity = p.opacity;
        d.blendMode = p.blendMode;
        d.effects = p.effects;
        d.effectCount = videofx::kMaxEffectSlots;
        d.timeSec = displaySec;
        d.maskType = p.maskType;
        d.maskCx = p.maskCx;
        d.maskCy = p.maskCy;
        d.maskW = p.maskW;
        d.maskH = p.maskH;
        d.maskFeather = p.maskFeather;
        d.maskInvert = p.maskInvert != 0;
        d.genParams = p.genParams;   // ISF/generator INPUTS (M7) — values for this frame
        // tex/size are render-thread-owned (serviced below under the mutex),
        // safe to read here on the render thread.
        if (const auto it = im.luts.find (clipId); it != im.luts.end())
        {
            d.lutTexture = it->second.tex;
            d.lutSize = it->second.size;
        }
    };

    auto fillDesc = [&fillDescCommon] (videorender::LayerDesc& d, const ClipGraphParams& p,
                                       const ClipStream& cs, double displaySec, int clipId)
    {
        d.texture = cs.tex;
        d.texWidth = cs.texW;
        d.texHeight = cs.texH;
        fillDescCommon (d, p, displaySec, clipId);
    };

    // Shader-generator programs compiled on this (the render) thread, keyed by
    // clipId -> the source last compiled. Lazy-compiled when a gen://shader
    // segment's source first appears or changes; cleared when the clip leaves.
    struct ShaderCompilation
    {
        std::string source, log;
        bool succeeded = false;
    };
    std::map<int, ShaderCompilation> shaderCompiled;

    // ISF INPUT defaults discovered at compile (clipId -> name -> default),
    // populated from setClipShader's GenParam list. Seeded under any explicit
    // gen value before mod routings apply (mirrors the exporter's clipGenDefaults
    // in stateAt) so a routing targeting a never-set ISF input still finds a base.
    // Persists across frames; refreshed only when a shader (re)compiles.
    std::map<int, std::map<std::string, double>> clipGenDefaults;

    // M5 Block C live score producer. It publishes one immutable carrier shared
    // by shaders, particles, score notation, and imported note instancing.
    canonicalblockc::FrameOwner scoreFrameOwner;
    arbitmod::Score localScore;
    std::shared_ptr<const arbitmod::Score> canonicalScoreSource;
    uint64_t scoreGenSeen = ~0ull;
    uint64_t pendingReadbackScoreGeneration = 0;
    canonicalblockc::OwnerIdentity scoreIdentity;
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> cachedCanonicalBlockCFrame;
    std::string scoreFrameError;
    // Error from a plan or binding failure that holds the frame. Cleared once a
    // frame presents without that failure.
    std::string planFrameError;
    // Reason for a device or resource failure that closes the channel. The
    // cleanup sends it to the consumer in the final BuffersGone message.
    std::string deviceCloseReason;
    // A seek or lifecycle push that lands while a frame is in flight retires the
    // Imported Frame source that frame was built on. That is a stale frame, not a
    // device failure, so the commit path drops it and retries instead of closing.
    constexpr const char* kPreviewLifecycleRetired =
        "Imported Frame preview source or lifecycle was retired";
    // Publishes the reason in rendererError before the channel closes. Failures
    // that keep the channel open use holdFrameForPlanError instead.
    const auto closeForDeviceFailure = [&](const std::string& what, const std::string& detail)
    {
        deviceCloseReason = detail.empty() ? what : what + ": " + detail;
        std::lock_guard<std::mutex> lock(im.mutex);
        im.rendererError = deviceCloseReason;
        im.wantClose = true;
    };

    // M6 mod matrix (render-thread owned). localRoutings mirrors the control-
    // plane copy; routingStates carry the one-pole smoothing memory advanced
    // once per timeline frame. Not warmed from frame 0 (live has no canonical
    // t=0) — the export path stays the deterministic ground truth.
    std::vector<arbitmod::Routing> localRoutings;
    std::vector<arbitmod::RoutingState> routingStates;
    uint64_t modGenSeen = ~0ull;
    long modAdvancedFrame = -1;
    videocontrol::Plan localControlPlan;
    videocontrol::Executor controlExecutor;
    bool signalWindowMissing = false;
    std::vector<videocontrol::SinkValue> controlSinkValues;
    uint64_t controlPlanGenSeen = ~0ull;
    std::shared_ptr<videorender::VisualParameterHistory> parameterHistory;
    std::tuple<uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,double> parameterHistoryIdentity {};
    std::string parameterHistoryError;
    std::shared_ptr<const videorender::VisualParameterTimeline> localParameterTimeline;

    // P2 Scripts-tab per-frame hook (render-thread owned). The selected engine is
    // (re)compiled only when the script set changes; scriptOverrides holds the
    // CURRENT timeline frame's {paramId->value}, applied as the TOP layer in
    // overlayMods (after the mod matrix — mirrors exporter stateAt). The hook runs
    // once per unique timeline frame (scriptRanFrame dedups paused/repeated
    // frames); it is NOT warmed from frame 0 like the exporter (live has no
    // canonical t=0) so cross-frame globals are best-effort, export is ground
    // truth. Both engine objects exist; only the selected one is compiled/run.
    arbitlua::LuaHook scriptLua;
    arbitjs::JsHook   scriptJs;
    std::string localScriptSource;
    std::string localScriptLang;
    uint32_t localScriptCpuMs = 0;
    uint32_t localScriptMemoryMiB = 0;
    uint64_t scriptGenSeen = ~0ull;
    bool scriptActive = false;          // a hook compiled OK and has frame()
    bool scriptWantJs = true;           // which engine the current source selects
    bool scriptErrLogged = false;       // throttle frame() error spam to once
    long scriptRanFrame = -1;           // last timeline frame frame() was run for
    std::map<std::string, double> scriptOverrides;     // this frame's overrides
    std::vector<arbitlua::HookNote> scriptNotes;       // active-only compatibility view
    std::vector<arbitlua::HookNote> scriptTimelineNotes; // past/future piano-roll view
    std::vector<arbitlua::HookLink> scriptLinks;       // scratch (rebuilt per run)

    // M4 Block B live audio (render-thread owned). The analyzer is stepped each
    // frame with whatever new mix samples arrived in the audio ring; its latest
    // FeatureFrame is cached as audio features for both audio-reactive shaders
    // (uRMS/uPeak/uOnset/uAudioBands) and audio-sourced mod routings. Like the
    // packers above it is NOT frame-0 warmed — it tracks live playback, so the
    // export path (whole-WAV analysis) stays the deterministic ground truth. The
    // read cursor snaps forward while paused so resuming never replays silence.
    arbitblockb::BlockBAnalyzer audioAnalyzer;
    std::vector<arbitblockb::FeatureFrame> audioFrameScratch;
    std::vector<float> audioDrainBuf (videoshm::kAudioRingCapacity, 0.0f);
    uint64_t audioReadCursor = 0;
    uint32_t audioAnalyzerSr = 0;
    videorender::AudioFeatures cachedAudioFeatures;

    // Frame-perfect audio parity (sweep): render-thread snapshot of the baked
    // master mix (Impl::previewMixFeats). Re-copied only when previewMixGeneration
    // changes (rare — only on a re-bake), since the vector is large. While the
    // transport is STOPPED and this is non-empty, the loop reads features from
    // HERE at the frame's time (previewMixFeatureAt) instead of zero-feeding —
    // making a stopped/scrubbed preview frame match export at the same time.
    std::vector<arbitblockb::FeatureFrame> localPreviewFeats;
    std::shared_ptr<const videohelper::AnalysisSources> localPreviewSourceFeats;
    double localPreviewSr = 0.0;
    uint64_t previewMixGenSeen = ~0ull;
    bool spectrumBakePending = false;
    videotime::BeatTimeline localBeatTimeline;
    videotime::BeatTimeline legacyBeatTimeline;
    uint64_t beatTimelineGenSeen = ~0ull;

    int frameCounter = 0;
    // Opt-in live frames, selected on the snapped project frame grid.
    const auto sceneTraceFrame = []() -> std::optional<std::int64_t>
    {
        const char* value = std::getenv("ARBIT_VIEWPORT_SCENE_TRACE_FRAME");
        if (value == nullptr || *value == '\0') return std::nullopt;
        std::int64_t frame = -1;
        const char* end = value + std::strlen(value);
        const auto parsed = std::from_chars(value, end, frame);
        return parsed.ec == std::errc{} && parsed.ptr == end && frame >= 0
            ? std::optional<std::int64_t>(frame) : std::nullopt;
    }();
    // Repeated paused frames expose a source/composite change without changing
    // the selected project time. Keep diagnostic GPU readbacks bounded.
    const auto sceneTraceLimit = []()
    {
        const char* value = std::getenv("ARBIT_VIEWPORT_SCENE_TRACE_REPEAT");
        if (value == nullptr || *value == '\0') return 1;
        int count = 0;
        const char* end = value + std::strlen(value);
        const auto parsed = std::from_chars(value, end, count);
        return parsed.ec == std::errc{} && parsed.ptr == end && count > 0
            ? std::min(count, 120) : 1;
    }();
    enum class SceneTraceReadback { Both, Native, Composite, None };
    const auto sceneTraceReadback = []()
    {
        const char* value = std::getenv("ARBIT_VIEWPORT_SCENE_TRACE_READBACK");
        if (value != nullptr && std::strcmp(value, "native") == 0)
            return SceneTraceReadback::Native;
        if (value != nullptr && std::strcmp(value, "composite") == 0)
            return SceneTraceReadback::Composite;
        if (value != nullptr && std::strcmp(value, "none") == 0)
            return SceneTraceReadback::None;
        return SceneTraceReadback::Both;
    }();
    int sceneTraceCount = 0;
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
    int sceneTracePublishedCount = 0;
#endif
    int glErrorsLogged = 0;
    int64_t nextTickNs = nowNs(); // shared-mode pacing (no vsync without a swap)
    int64_t nextScopeNs = 0;      // ~10 Hz scope readback pacing
    std::vector<uint8_t> scopeRgba; // scope readback scratch, reused
    std::vector<Impl::VisualEvent> frameVisualEvents;
    frameVisualEvents.reserve(64);
    videoopticalflow::PreviewDecodeCoordinator opticalFlowDecoder (
        [] (const videowire::ResolvedVisualSnapshot& snapshot, int clipId,
            double displaySec, int maxW, int maxH,
            std::shared_ptr<const videoopticalflow::DecodedPairReceipt>& out)
        {
            videoopticalflow::AuthoredPairSelection selection;
            auto error = videoopticalflow::selectAuthoredPair (snapshot, clipId, selection);
            if (! error.empty()) return error;
            return videoopticalflow::resolveDecodedPair (
                *selection.plan, selection.pair, snapshot.opticalFlowSources,
                snapshot.segments, snapshot.snapshotGeneration,
                displaySec, maxW, maxH, out);
        });
    std::map<int, videoopticalflow::AuthoredPairResult> opticalFlowFields;
    std::map<int, videoopticalflow::PreviewDecodeCoordinator::Key> opticalFlowFieldKeys;
    std::map<int, videoopticalflow::PreviewDecodeOwner> opticalDecodeOwners;
    struct OpticalUploadOwner
    {
        std::uint64_t helperGeneration = 0;
        std::uint64_t structuralRevision = 0;
        std::unique_ptr<videoopticalflow::NativeSourceUploader> uploader;
    };
    std::map<int, OpticalUploadOwner> opticalUploadOwners;
    std::uint64_t opticalFlowTimelineSeen = 0;
    std::uint64_t opticalFlowSeekSeen = 0;
    std::uint64_t opticalFlowLoopSeen = 0;
    std::uint64_t opticalFlowHelperSeen = 0;
    bool opticalFlowActiveLastFrame = false;
    // Idle production gate for the docked (shm / shared) modes. Window mode
    // keeps its own vsync pacing and never consults it.
    arbitviewport::PublishGate publishGate;
    arbitviewport::PublishGate::Snapshot gateSnapshot;

    // Loop-phase probe: stamp the phase being entered. The phase that just
    // ended is logged when it ran longer than 2 s. A phase that never ends
    // logs nothing here; loopPhaseAgeMs over RPC is the signal for that case.
    const auto enterLoopPhase = [&im] (LoopPhase next)
    {
        const int64_t now = nowNs();
        const int64_t since = im.loopPhaseEnterNs.load (std::memory_order_relaxed);
        const int prev = im.loopPhase.load (std::memory_order_relaxed);
        const char* lastStep = im.loopStep.load (std::memory_order_relaxed);
        if (since > 0 && now - since > 2000000000ll)
            std::fprintf (stderr, "[viewport] loop phase %s took %lld ms (last step %s)\n",
                          kLoopPhaseNames[prev], (long long) ((now - since) / 1000000), lastStep);
        im.loopPhase.store ((int) next, std::memory_order_relaxed);
        im.loopPhaseEnterNs.store (now, std::memory_order_relaxed);
        // A new phase starts at its own step label.
        im.loopStep.store (kLoopPhaseNames[(int) next], std::memory_order_relaxed);
        im.loopStepEnterNs.store (now, std::memory_order_relaxed);
    };

    // Advance sub-step probe: stamp the call about to run inside the phase.
    const auto enterLoopStep = [&im] (const char* step)
    {
        im.loopStepEnterNs.store (nowNs(), std::memory_order_relaxed);
        im.loopStep.store (step, std::memory_order_relaxed);
    };
    // Diagnostics for deferred frames: the run length, and the source and time of the last log line.
    std::uint64_t deferredConsecutive = 0;
    std::string_view lastDeferredSource;
    std::chrono::steady_clock::time_point lastDeferredLog {};
    // Retired-lifecycle commit retries: the run count, and the time of the last log line.
    std::uint64_t retiredPublishRetries = 0;
    std::chrono::steady_clock::time_point lastRetiredLog {};

    while (running_.load() && ! im.wantClose.load()
           && (offscreen || ! glfwWindowShouldClose (win)))
    {
        enterLoopPhase (LoopPhase::TickTop);
        im.loopTicks.fetch_add (1, std::memory_order_relaxed);
        frameVisualEvents.clear();
        {
            std::lock_guard<std::mutex> lock(im.mutex);
            frameVisualEvents.swap(im.visualEvents);
        }
        // Hidden shared/shm contexts have no OS-window events to process. On
        // macOS glfwPollEvents is main-thread-only, while rendering legitimately
        // owns the detached GL context on this worker thread.
        if (! offscreen)
        {
#if ! defined (__APPLE__)
            glfwPollEvents();
#endif
        }

        // Window control requests from the RPC thread (window mode only).
#if ! defined (__APPLE__)
        if (! offscreen)
        {
            if (im.boundsDirty.exchange (false))
            {
                std::lock_guard<std::mutex> lock (im.mutex);
                glfwSetWindowPos (win, im.reqX, im.reqY);
                glfwSetWindowSize (win, im.reqW, im.reqH);
            }
            const int fsReq = im.fullscreenRequest.exchange (-1);
            if (fsReq == 1)
            {
                auto* mon = glfwGetPrimaryMonitor();
                const auto* mode = glfwGetVideoMode (mon);
                glfwSetWindowMonitor (win, mon, 0, 0, mode->width, mode->height, mode->refreshRate);
            }
            else if (fsReq == 0)
            {
                glfwSetWindowMonitor (win, nullptr, 100, 100, width, height, 0);
            }
        }
#endif

        // shm-mode resize: just a render-size change — frames self-describe
        // their dims per slot, so no ring renegotiation is needed as long as
        // a frame fits one slot (clamped; Arbit recreates the ring to grow).
        enterLoopPhase (LoopPhase::Resize);
        if (shm && im.shmResizeDirty.exchange (false))
        {
            int newW = im.shmReqW.load();
            int newH = im.shmReqH.load();
            if (newW >= 16 && newH >= 16)
            {
                clampToSlot (newW, newH);
                fbW = newW;
                fbH = newH;
            }
        }

#if ARBIT_HAVE_GPU_SHARED
        if (shared)
        {
            enterLoopPhase (LoopPhase::SocketPump);
            // Socket pump: greet new clients, process FRAME_RELEASE, detect
            // disconnect (consumer gone => every announced buffer is free
            // again; the dmabuf memory itself stays valid via our own fds).
            if (im.fdSocket.poll())
            {
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
                if (linearSharedActive)
                {
                    linearSharedActive = false;
                    renderer.setHdrImageCapture(false);
                    destroySharedBuffers();
                    std::string error;
                    if (!allocateSharedBuffers(fbW, fbH, error))
                    { closeForDeviceFailure("linear IOSurface buffers could not be reallocated for a new consumer", error); continue; }
                }
                linearNegotiation.begin(linearOffer, static_cast<uint64_t>(nowNs() / 1000000));
#endif
                sendHello();
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
                if (!linearNegotiation.waiting)
#endif
                    sendBuffers();
                for (auto& b : im.buffers)
                    b.busy = false;
                // A settled scene would otherwise never send the new consumer a
                // first frame, so ask for one produce.
                publishGate.requestProduce();
            }
            for (;;)
            {
                FdSocketServer::Received msg;
                const auto r = im.fdSocket.recvMsg (msg);
                if (r == FdSocketServer::RecvResult::NoData)
                    break;
                if (r == FdSocketServer::RecvResult::Disconnected)
                {
                    for (auto& b : im.buffers)
                        b.busy = false;
                    break;
                }
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
                if (msg.type == static_cast<uint32_t>(gpusurf::MsgType::LinearIOSurfaceSelect)
                    && linearNegotiation.waiting)
                {
                    linearSharedActive = linearNegotiation.select(msg.payload.data(), msg.payload.size(),
                        static_cast<uint32_t>(msg.fds.size()), static_cast<uint64_t>(nowNs() / 1000000));
                    if (linearSharedActive)
                    {
                        destroySharedBuffers();
                        std::string error;
                        if (!allocateSharedBuffers(fbW, fbH, error))
                        { closeForDeviceFailure("linear IOSurface buffers could not be allocated for the selected format", error); break; }
                        renderer.setHdrImageCapture(true);
                    }
                    sendBuffers();
                }
                if (msg.type == static_cast<uint32_t>(gpusurf::MsgType::LinearIOSurfaceFrameRelease))
                {
                    gpusurf::LinearIOSurfaceFrameReleasePayload release;
                    const bool exact = msg.payload.size() == sizeof(release) && msg.fds.empty();
                    if (exact) std::memcpy(&release, msg.payload.data(), sizeof(release));
                    if (linearSharedActive && exact && gpusurf::validLinearRelease(release, sharedBufferGeneration,
                            static_cast<uint32_t>(im.buffers.size())))
                    {
                        im.buffers[release.frame.bufferIndex].busy = false;
                        im.sharedReleasesReceived.fetch_add(1);
                    }
                    else im.sharedReleasesInvalid.fetch_add(1);
                }
#endif
                for (int fd : msg.fds) // v1 expects no fds from the consumer
                    closeFdIfValid (fd);
                if (!linearSharedActive && msg.type == (uint32_t) gpusurf::MsgType::FrameRelease
                    && msg.payload.size() >= sizeof (gpusurf::FrameReleasePayload))
                {
                    gpusurf::FrameReleasePayload rel;
                    std::memcpy (&rel, msg.payload.data(), sizeof (rel));
                    if (rel.bufferIndex < im.buffers.size())
                    {
                        im.buffers[rel.bufferIndex].busy = false;
                        im.sharedReleasesReceived.fetch_add (1);
                    }
                    else
                        im.sharedReleasesInvalid.fetch_add (1);
                }
            }

#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
            enterLoopPhase (LoopPhase::LinearWait);
            if (linearNegotiation.waiting && im.fdSocket.hasClient())
            {
                if (!linearNegotiation.expire(static_cast<uint64_t>(nowNs() / 1000000)))
                { std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }
                sendBuffers(); // new helper / old consumer: bounded v2 fallback
            }
#endif
            // Resize renegotiation: tear everything down and re-announce.
            enterLoopPhase (LoopPhase::Resize);
            if (im.sharedResizeDirty.exchange (false))
            {
                const int newW = im.sharedReqW.load();
                const int newH = im.sharedReqH.load();
                if (newW > 0 && newH > 0 && (newW != fbW || newH != fbH))
                {
                    if (im.fdSocket.hasClient())
                        im.fdSocket.sendMsg ((uint32_t) gpusurf::MsgType::BuffersGone, nullptr, 0);
                    destroySharedBuffers();
                    std::string err;
                    if (! allocateSharedBuffers (newW, newH, err))
                    {
                        std::fprintf (stderr, "[viewport] shared resize failed: %s\n", err.c_str());
                        closeForDeviceFailure("shared buffers could not be resized", err);
                        continue;
                    }
                    fbW = newW;
                    fbH = newH;
                    renderer.setOutputSize (fbW, fbH);
                    if (im.fdSocket.hasClient())
                        sendBuffers();
                }
            }

            // Nobody connected yet (or consumer died): don't burn the GPU.
            if (! im.fdSocket.hasClient())
            {
                enterLoopPhase (LoopPhase::FramePacing);
                std::this_thread::sleep_for (std::chrono::milliseconds (20));
                continue;
            }
        }
#endif // ARBIT_HAVE_GPU_SHARED

        const int64_t frameStartNs = nowNs();

        // --- Clock: read transport and extrapolate against the wall clock.
        enterLoopPhase (LoopPhase::TransportRead);
        videoshm::TransportBlock tb {};
        bool haveClock = false;
        {
            std::lock_guard<std::mutex> lock (im.mutex);
            if (im.beatTimelineGeneration != beatTimelineGenSeen)
            {
                localBeatTimeline = im.beatTimeline;
                beatTimelineGenSeen = im.beatTimelineGeneration;
                cachedCanonicalBlockCFrame.reset();

            }
            for (int attempt = 0; attempt < 3 && ! haveClock; ++attempt)
                haveClock = im.transport.isOpen() && im.transport.read (tb);
        }

        // Canonical VALUE-grid frame rate (the export job.fps), read live each
        // frame so a project-fps change re-snaps the preview immediately. Drives
        // every per-frame value feed (modulation / shader clock / score / lua /
        // source selection) so the live preview lands on the SAME frame grid the
        // exporter steps — preview == export by construction. Falls back to the
        // display targetFps (legacy 60) when no project fps was supplied; the
        // DISPLAY refresh still uses targetFps independently (frameBudgetSec).
        const double vfRaw = im.valueFps.load (std::memory_order_relaxed);
        const double valueFps = vfRaw > 0.0 ? vfRaw
                                            : (targetFps > 0.0 ? targetFps : 30.0);

        double displaySec = 0.0, bpm = 120.0, beatsPerBar = 4.0;
        bool playing = false;
        if (haveClock)
        {
            playing = tb.playing != 0;
            if (beatTimelineGenSeen != 0)
            {
                const double elapsed = playing
                    ? std::max(0.0, std::min(
                        (double) (frameStartNs - tb.hostTimeNs) * 1e-9, 0.5)) : 0.0;
                displaySec = localBeatTimeline.beatToSeconds(tb.playheadBeats) + elapsed;
                const auto frameClock = localBeatTimeline.clockAtSeconds(displaySec);
                bpm = frameClock.bpm;
                beatsPerBar = frameClock.beatsPerBar;
            }
            else
            {
                bpm = tb.bpm > 0.0 ? tb.bpm : 120.0;
                beatsPerBar = tb.beatsPerBar > 0.0 ? tb.beatsPerBar : 4.0;
                double beats = tb.playheadBeats;
                if (playing)
                {
                    const double elapsed = (double) (frameStartNs - tb.hostTimeNs) * 1e-9;
                    beats += std::max (0.0, std::min (elapsed, 0.5)) * bpm / 60.0;
                }
                displaySec = beats * 60.0 / bpm;
            }
        }
        // Frame-perfect (faithful preview): snap the value-path clock to the
        // project frame grid so the live frame is sampled at exactly the time
        // the exporter steps (t = n/valueFps). This removes the continuous
        // wall-clock vs n/fps divergence (audit findings #1/#14/#15) and makes
        // the whole frame internally coherent: srcSec, makeShaderClock,
        // d.timeSec, the routing/lua/score beats and transition progress all
        // derive from this one snapped displaySec. The DISPLAY still repaints at
        // targetFps — it simply holds the current project-fps frame between grid
        // steps. Snapping a zero/paused displaySec is harmless (rounds to a grid
        // point), and matches what the exporter would render for that region.
        const double requestedDisplaySec = displaySec;
        displaySec = (double) std::llround (displaySec * valueFps) / valueFps;
        if (beatTimelineGenSeen == 0)
            legacyBeatTimeline.reset(bpm, beatsPerBar);
        const auto& activeBeatTimeline = beatTimelineGenSeen != 0
            ? localBeatTimeline : legacyBeatTimeline;
        // Select one timestamp before any time-dependent scene work. The lease
        // and row copy below remain the same through cursor evaluation/publication.
        uint64_t eventFrameTimelineGeneration = 0;
        videowire::VisualEventPreviewFrame eventFrame;
        enterLoopPhase (LoopPhase::EventGate);
        {
            std::lock_guard<std::mutex> lock(im.mutex);
            frameVisualPlans = im.visualPlanPublication.shared();
            frameVisualScheduleBindings = im.visualEventSchedules;
            eventFrameTimelineGeneration = im.timelineGeneration;
            const auto relevantClip = [&] (int clipId, double at)
            {
                for (const auto& segment : im.segments)
                {
                    const auto duration = (segment.outSec - segment.inSec) / std::max(segment.rate,1e-9);
                    if (at < segment.displayStartSec || at >= segment.displayStartSec + duration) continue;
                    if (segment.clipId == clipId) return true;
                    if (segment.transitionType != 0 && segment.transitionDurationSec > 0
                        && at < segment.displayStartSec + segment.transitionDurationSec)
                    {
                        const auto* from = videowire::resolveTransitionFrom(im.segments,segment,1e-3);
                        if (from != nullptr && from->clipId == clipId) return true;
                    }
                }
                return false;
            };
            if (im.beatTimelineGeneration == beatTimelineGenSeen)
                eventFrame = videowire::selectVisualEventPreviewFrame(frameVisualScheduleBindings,visualEventCursor,
                    requestedDisplaySec,valueFps,activeBeatTimeline,relevantClip,im.publishedDisplaySec);
            if (!eventFrame.ready && (im.rendererError.empty() || im.rendererError == eventCoveragePending))
                im.rendererError = eventCoveragePending;
            else if (im.rendererError == eventCoveragePending) im.rendererError.clear();
        }
        if (!eventFrame.ready)
        {
            // Existing publication remains intact. Retry after normal pacing,
            // without advancing scene state/cursor or converting pending input to silence.
            enterLoopPhase (LoopPhase::FramePacing);
            std::this_thread::sleep_for(std::chrono::duration<double>(std::min(frameBudgetSec,0.05)));
            nextTickNs = nowNs();
            continue;
        }
        enterLoopPhase (LoopPhase::TransportStore);
        displaySec = eventFrame.seconds;
        const auto snappedClock = activeBeatTimeline.clockAtSeconds(displaySec);
        const double displayBeat = snappedClock.beat;
        bpm = snappedClock.bpm;
        beatsPerBar = snappedClock.beatsPerBar;
        {
            std::lock_guard<std::mutex> lock (im.mutex);
            im.displaySec = displaySec;
            im.transportOpen = haveClock;
            im.transportPlaying = playing;
            im.transportGeneration = haveClock ? tb.generation.load (std::memory_order_relaxed) : 0;
            im.transportAgeSec = haveClock ? std::max (0.0, (double) (frameStartNs - tb.hostTimeNs) * 1e-9) : 0.0;
            im.transportPlayheadBeats = haveClock ? tb.playheadBeats : 0.0;
        }

        // --- M4 Block B live: drain the master-mix ring into the analyzer and
        // cache this frame's audio features. While playing we step the analyzer
        // with the newly-arrived samples so its onset/flux state stays warm; the
        // latest FeatureFrame feeds audio-reactive shaders and audio-sourced mod
        // routings below. While paused no audio flows, so we snap the read cursor
        // forward (resume never replays the silence backlog), reset the analyzer,
        // and zero-feed — a paused preview frame stays stable. Samples are copied
        // out under `im.mutex` (guards against attachAudio remapping the region);
        // the FFT runs outside the lock.
        enterLoopPhase (LoopPhase::AudioDrain);
        bool haveAudio = false;
        {
            int drained = 0;
            uint32_t sr = 0;
            bool ringOpen = false;
            {
                std::lock_guard<std::mutex> lock (im.mutex);
                ringOpen = im.audioRing.isOpen();
                if (ringOpen && playing)
                {
                    int n;
                    while (drained < (int) audioDrainBuf.size()
                           && (n = im.audioRing.read (audioDrainBuf.data() + drained,
                                                      (int) audioDrainBuf.size() - drained,
                                                      &audioReadCursor, &sr)) > 0)
                        drained += n;
                }
                else if (ringOpen)
                {
                    audioReadCursor = im.audioRing.writeCursor();
                }
                // Frame-perfect audio parity (sweep): refresh the baked-mix
                // snapshot under the same lock. Only on a re-bake (rare), so the
                // large copy never happens per-frame.
                if (im.previewMixGeneration != previewMixGenSeen)
                {
                    localPreviewFeats = im.previewMixFeats;
                    localPreviewSr    = im.previewMixSr;
                    localPreviewSourceFeats=im.previewSourceFeats;
                    previewMixGenSeen = im.previewMixGeneration;
                    if (spectrumBakePending && !localPreviewFeats.empty())
                    {
                        im.rendererError.clear();
                        spectrumBakePending = false;
                    }
                }
            }
            if (ringOpen && playing)
            {
                if (sr > 0 && sr != audioAnalyzerSr)
                {
                    audioAnalyzer.setSampleRate ((float) sr);
                    audioAnalyzerSr = sr;
                }
                if (drained > 0)
                {
                    audioFrameScratch.clear();
                    audioAnalyzer.pushSamples (audioDrainBuf.data(), drained, audioFrameScratch);
                    if (! audioFrameScratch.empty())
                    {
                        const arbitblockb::FeatureFrame& f = audioFrameScratch.back();
                        videorender::fillAudioFeatures (cachedAudioFeatures, f);
                        // Determinism (audit #3): when several analysis hops land
                        // in one drain, keeping only .back() drops an onset pulse
                        // that fired on an earlier hop, so an onset-triggered mod
                        // silently misses under load. The onset is a decaying
                        // envelope that never reaches zero, so an "any frame has
                        // onset" test would pin it at 1.0 indefinitely.
                        // mergeDrainOnset takes the maximum envelope across the
                        // drain instead. The continuous features stay .back()
                        // (onsetAge already tracks the freshest age).
                        videorender::mergeDrainOnset (cachedAudioFeatures, audioFrameScratch);
                    }
                }
                haveAudio = ! cachedAudioFeatures.bands.empty();
            }
            else if (ringOpen)
            {
                if (audioAnalyzerSr != 0) { audioAnalyzer.reset(); audioAnalyzerSr = 0; }
                cachedAudioFeatures = videorender::AudioFeatures {};
            }
        }

        // Frame-perfect audio parity (sweep): the offline-analyzed baked mix,
        // sampled at a timeline time exactly as the exporter's audioFeatureAt does
        // (idx = floor(t·sr/kHop) − 1; sub-frame-0 clamps to frame 0 — the same
        // parity contract). Returns the zero-feed when no mix is set.
        auto previewMixFeatureAt = [&] (double sec) -> videorender::AudioFeatures
        {
            videorender::AudioFeatures af;
            if (localPreviewFeats.empty() || localPreviewSr <= 0.0)
                return af;
            // Same frame index as the exporter (videohelper::analysisFrameIndex).
            videorender::fillAudioFeatures (af, localPreviewFeats[
                videohelper::analysisFrameIndex(localPreviewFeats.size(), localPreviewSr, sec)]);
            return af;
        };
        // While STOPPED/scrubbing no live audio flows, so the drain above zero-fed
        // (or held) cachedAudioFeatures — diverging from export. If a baked mix is
        // present, feed THIS frame's offline feature instead so the stopped frame
        // matches export at the same time. While playing the live ring stays
        // authoritative (zero added monitoring latency — the chosen design).
        const bool useBakedAudio = (! playing && ! localPreviewFeats.empty());
        if (useBakedAudio)
        {
            cachedAudioFeatures = previewMixFeatureAt (displaySec);
            haveAudio = ! cachedAudioFeatures.bands.empty();
        }

        // Block B mod source for M6 routings (built once; fed at the current
        // frame in the advance loop — see below).
        arbitmod::Audio liveAud {};
        if (haveAudio)
        {
            liveAud.rms   = cachedAudioFeatures.rms;
            liveAud.peak  = cachedAudioFeatures.peak;
            liveAud.onset = cachedAudioFeatures.onset;
            for (size_t i = 0; i < cachedAudioFeatures.bands.size() && i < 64; ++i)
                liveAud.bands[i] = cachedAudioFeatures.bands[i];
        }
        if (haveClock)
        {
            liveAud.sourceCount = std::min<int>((int) tb.trackFeatureCount,
                                                 (int) liveAud.sources.size());
            for (int i = 0; i < liveAud.sourceCount; ++i)
            {
                const auto& source = tb.trackFeatures[i];
                liveAud.sources[(size_t) i] = { source.trackId, (int) source.tapPoint,
                                                source.rms, source.peak, source.onset };
            }
        }

        // --- Idle production gate (viewport_publish_gate.h). A skipped tick
        // returns before any scene evaluation, so a stopped, unchanged scene
        // costs only the transport and audio reads above.
        if (offscreen)
        {
            enterLoopPhase (LoopPhase::ShouldProduce);
            gateSnapshot = arbitviewport::PublishGate::Snapshot();
            gateSnapshot.visible = im.visible.load();
            gateSnapshot.playing = playing;
            gateSnapshot.revision = im.contentRevision.load();
            gateSnapshot.mixGeneration = previewMixGenSeen;
            gateSnapshot.displaySec = displaySec;
            gateSnapshot.displayBeat = displayBeat;
            gateSnapshot.bpm = bpm;
            gateSnapshot.beatsPerBar = beatsPerBar;
            gateSnapshot.width = fbW;
            gateSnapshot.height = fbH;
            gateSnapshot.interpSuspended = interpSuspended_.load (std::memory_order_relaxed);
            if (! publishGate.shouldProduce (gateSnapshot))
            {
                // The swap at the top of the tick moved the visual events out of
                // im.visualEvents. Hand them back for the next produce.
                if (! frameVisualEvents.empty())
                {
                    std::lock_guard<std::mutex> lock (im.mutex);
                    im.visualEvents.insert (im.visualEvents.begin(),
                                            frameVisualEvents.begin(), frameVisualEvents.end());
                    frameVisualEvents.clear();
                }
                enterLoopPhase (LoopPhase::FramePacing);
                std::this_thread::sleep_for (std::chrono::duration<double> (frameBudgetSec));
                nextTickNs = nowNs();
                continue;
            }
        }

        // --- Gather every segment active at displaySec + transition sources.
        enterLoopPhase (LoopPhase::Advance);
        std::vector<ActiveLayer> act;
        std::vector<videorender::ImageLayerDesc> overlays; // active text layers
        uint64_t timelineGen = 0;
        std::shared_ptr<const videowire::ResolvedVisualSnapshot> frameOpticalSnapshot;
        std::set<int> liveClipIds;
        {
            enterLoopStep ("advance:state-lock");
            std::unique_lock<std::mutex> lock (im.mutex);

            // A newer timeline/plan must be selected again before any stateful work.
            if (im.timelineGeneration != eventFrameTimelineGeneration
                || im.beatTimelineGeneration != beatTimelineGenSeen
                || im.visualPlanPublication.shared() != frameVisualPlans)
            {
                lock.unlock();
                enterLoopPhase (LoopPhase::FramePacing);
                std::this_thread::sleep_for(std::chrono::duration<double>(std::min(frameBudgetSec,0.05)));
                nextTickNs = nowNs();
                continue;
            }

            // Apply pending timestamped params whose beat has passed.
            const double playheadBeats = haveClock ? displayBeat : 0.0;
            auto& pend = im.pendingParams;
            pend.erase (std::remove_if (pend.begin(), pend.end(),
                [&] (const PendingParam& pp)
                {
                    if (pp.atBeat > playheadBeats) return false;
                    int cid; std::string node, par;
                    if (Impl::parseParamId (pp.paramId, cid, node, par))
                        im.apply (cid, node, par, pp.value);
                    else if (Impl::parseTextParamId (pp.paramId, cid, par))
                        im.applyText (cid, par, pp.value);
                    return true;
                }), pend.end());

            timelineGen = im.timelineGeneration;
            frameOpticalSnapshot = im.opticalFlowSnapshot;
            localParameterTimeline=im.parameterTimeline;
            if (timelineGen != streamsGeneration)
                for (const auto& s : im.segments)
                    liveClipIds.insert (s.clipId);

            // M5: re-copy the live score + reset the packer only when the owned
            // note set actually changed (cheap per-frame check vs a vector copy).
            if (im.scoreGeneration != scoreGenSeen)
            {
                canonicalScoreSource = im.scoreSnapshot;
                localScore = *canonicalScoreSource;
                scoreGenSeen = im.scoreGeneration;
                scoreIdentity = im.scoreIdentity;
                cachedCanonicalBlockCFrame.reset();
            }

            // M6: re-copy the routings + reset their smoothing state only when
            // the routing set changed. The per-clip overlay happens in the
            // segment gather loop below.
            if (im.modGeneration != modGenSeen)
            {
                localRoutings = im.routings;
                modGenSeen = im.modGeneration;
                routingStates.assign (localRoutings.size(), arbitmod::RoutingState {});
                modAdvancedFrame = -1;
            }
            if (im.controlPlanGeneration != controlPlanGenSeen)
            {
                localControlPlan = im.controlPlan;
                controlPlanGenSeen = im.controlPlanGeneration;
                std::string controlError;
                enterLoopStep ("advance:control-plan-bind");
                if (!controlExecutor.bind(localControlPlan, controlError))
                    std::fprintf(stderr, "[viewport] rejected video control plan: %s\n",
                                 controlError.c_str());
                controlSinkValues.clear();
                modAdvancedFrame = -1;
            }

            // P2: re-copy + recompile the per-frame script hook only when the
            // script set changed (compiling is the heavy step — never per frame).
            // The engine is Lua OR JS (scriptLang selects; empty source ⇒ no
            // hook). A compile error logs once and leaves the hook inert
            // (scriptActive false), so the preview just shows the un-scripted
            // graph — never a crash. Mirrors exporter.cpp's compile-once block.
            if (im.scriptGeneration != scriptGenSeen)
            {
                localScriptSource = im.scriptSource;
                localScriptLang   = im.scriptLang;
                localScriptCpuMs = im.scriptCpuMs;
                localScriptMemoryMiB = im.scriptMemoryMiB;
                scriptGenSeen     = im.scriptGeneration;
                scriptActive      = false;
                scriptWantJs      = (localScriptLang != "lua");
                scriptErrLogged   = false;
                scriptRanFrame    = -1;
                scriptOverrides.clear();
                scriptLua.close();
                scriptJs.close();
                if (! localScriptSource.empty())
                {
                    std::string serr;
                    enterLoopStep ("advance:script-compile");
                    scriptActive = scriptWantJs
                        ? scriptJs.compile (localScriptSource, serr, localScriptCpuMs, localScriptMemoryMiB)
                        : scriptLua.compile (localScriptSource, serr, localScriptCpuMs, localScriptMemoryMiB);
                    if (! scriptActive)
                        std::fprintf (stderr, "[%s] live hook disabled: %s\n",
                                      scriptWantJs ? "js" : "lua", serr.c_str());
                }
            }

            // Advance each routing's smoothing/LFO/envelope state to the current
            // timeline frame (once per frame, monotonic — same as the exporter).
            // The discarded combined value leaves RoutingState::smoothed == the
            // modulation value the overlay reads. Audio-sourced routings see the
            // live mix at the current frame (liveAud, built above) — warm-up
            // frames on a scrub still zero-feed since there is no audio history;
            // score-sourced routings see the live notes. The warm-up window is
            // sized to the slowest smoother's convergence (see below) so the
            // smoothed value matches the frame-0-warmed export to epsilon.
            if (! localRoutings.empty() || !controlExecutor.empty())
            {
                const double clkFps = valueFps;  // project value-grid fps (== export job.fps)
                const long g = (long) std::llround (displaySec * clkFps);

                // Playhead seeked backward (scrub-back, looping, or a viewport
                // reopen whose first frame renders at the previous playhead before
                // the user seeks earlier): modAdvancedFrame is monotonic, so
                // without this reset startF would exceed g and the advance loop
                // below would never run — freezing every routing's smoothed value
                // at its last position (the live mod matrix appears stuck). Reset
                // so the ~1 s warm-up window re-runs ending at the new frame.
                if (g < modAdvancedFrame)
                {
                    modAdvancedFrame = -1;
                    routingStates.assign(localRoutings.size(), arbitmod::RoutingState {});
                    controlExecutor.reset();
                    controlSinkValues.clear();
                }
                // Warm-up window (audit #13): a flat ~1s was too short for slow
                // one-pole smoothers, so the first preview frames after a play-
                // start / seek differed from the export (which warms from frame
                // 0). A one-pole's initial-state influence decays as exp(-T/tau),
                // so ~9.2·tau of warm-up leaves <1e-4 residual — i.e. the smoothed
                // value matches the frame-0-warmed export to epsilon. Size the
                // window to the slowest enabled smoother (0 ⇒ nothing to warm),
                // capped at 30s so a pathologically slow tau can't spin the loop.
                // (Audio/score-sourced routings still zero-feed during warm-up —
                // no live history — exactly as before; the export stays ground
                // truth for those, matching the Pass-3 audio limitation.)
                float maxTauBeats = 0.0f;
                for (const auto& r : localRoutings)
                    if (r.enabled) maxTauBeats = std::max (maxTauBeats, r.smoothingBeats);
                long warmFrames = 0;
                if (maxTauBeats > 0.0f)
                {
                    const double warmBeats = 9.2 * (double) maxTauBeats;
                    const double warmStartBeat = std::max(0.0, displayBeat - warmBeats);
                    const double warmSeconds = displaySec
                        - activeBeatTimeline.beatToSeconds(warmStartBeat);
                    warmFrames = (long) std::ceil(std::max(0.0, warmSeconds) * clkFps);
                    warmFrames = std::min (warmFrames,
                                           (long) std::llround (clkFps * 30.0));
                }
                // The bounded Signal plan also warms downstream Control state
                // from frame zero, matching export after a seek or reopen.
                const bool exactSignalWarmup = localControlPlan.signalWindow.present()
                    && controlExecutor.signalWindowAvailable(g, clkFps);
                const long startF = exactSignalWarmup ? modAdvancedFrame + 1
                    : std::max (modAdvancedFrame + 1, g - warmFrames);
                const arbitmod::Audio zeroAud {};
                enterLoopStep ("advance:routing-warmup");
                for (long f = startF; f <= g; ++f)
                {
                    const double frameSec = (double) f / clkFps;
                    const auto frameClock = activeBeatTimeline.clockAtSeconds(frameSec);
                    const double dtBeats = activeBeatTimeline.secondsToBeat(
                        frameSec + 1.0 / clkFps) - frameClock.beat;
                    arbitmod::Clock clk;
                    clk.beat        = (float) frameClock.beat;
                    clk.bpm         = (float) frameClock.bpm;
                    clk.beatsPerBar = (float) frameClock.beatsPerBar;
                    // Audio per frame. While PLAYING only the current frame has real
                    // (live-ring) audio; warm-up frames have no history → zero-fed.
                    // While STOPPED with a baked mix, feed EVERY frame the offline
                    // feature at its time — exactly as the exporter's
                    // advanceRoutingsToFrame does — so audio-sourced routings warm to
                    // the same smoothed state export produces (frame-perfect parity).
                    arbitmod::Audio bakedAud {};
                    const arbitmod::Audio* audPtr;
                    if (useBakedAudio)
                    {
                        enterLoopStep ("advance:audio-feature-lookup");
                        const videorender::AudioFeatures vaf =
                            previewMixFeatureAt ((double) f / clkFps);
                        bakedAud.rms = vaf.rms; bakedAud.peak = vaf.peak; bakedAud.onset = vaf.onset;
                        const int nb = std::min (64, (int) vaf.bands.size());
                        for (int b = 0; b < nb; ++b) bakedAud.bands[(size_t) b] = vaf.bands[(size_t) b];
                        audPtr = &bakedAud;
                    }
                    else
                        audPtr = (f == g) ? &liveAud : &zeroAud;
                    for (size_t ri = 0; ri < localRoutings.size(); ++ri)
                    {
                        if (! localRoutings[ri].enabled) continue;
                        arbitmod::evaluateRouting (localRoutings[ri], routingStates[ri],
                                                   0.0f, localScore, clk, *audPtr, (float) dtBeats);
                    }
                    const bool missingWindow = !controlExecutor.signalWindowAvailable(f, clkFps);
                    if (missingWindow && !signalWindowMissing)
                        std::fprintf(stderr, "[viewport] video Signal window unavailable for frame/FPS; dependent sinks are inactive\n");
                    signalWindowMissing = missingWindow;
                    enterLoopStep ("advance:control-evaluate");
                    const auto& sinks = controlExecutor.evaluate(
                        localScore, clk, *audPtr, static_cast<float>(dtBeats),
                        static_cast<float>(1.0 / clkFps), f, clkFps);
                    controlSinkValues.assign(sinks.begin(), sinks.end());
                }
                if (g > modAdvancedFrame) modAdvancedFrame = g;
            }

            // P2: run the per-frame script hook ONCE for this timeline frame and
            // capture its {paramId->value} overrides (applied per clip as the TOP
            // layer in overlayMods below — mirrors exporter.cpp::stateAt). Deduped
            // on the rounded timeline frame so a paused/repeated frame is stable
            // and the script's ctx.frame does not race the render rate. The ctx is
            // built field-for-field like the exporter's: live clock + the Block B
            // live mix (liveAud, zero-fed when no audio) + the same past/future
            // note window the live Block C shader feed receives.
            if (scriptActive)
            {
                const double clkFps = valueFps;  // project value-grid fps (== export job.fps)
                const long g = (long) std::llround (displaySec * clkFps);
                if (g != scriptRanFrame)
                {
                    scriptRanFrame = g;
                    arbitlua::FrameCtx ctx;
                    ctx.t           = displaySec;
                    ctx.beat        = displayBeat;
                    ctx.bpm         = bpm;
                    ctx.beatsPerBar = beatsPerBar;
                    ctx.frame       = g;
                    ctx.rootFreq    = localScore.rootFreq;
                    ctx.scoreHistoryBeats = localScore.historyBeats;
                    ctx.scoreLookaheadBeats = localScore.lookaheadBeats;
                    // Block B live audio (zero-fed when no ring / before playback).
                    ctx.rms = liveAud.rms; ctx.peak = liveAud.peak; ctx.onset = liveAud.onset;
                    ctx.onsetAge = haveAudio ? cachedAudioFeatures.onsetAge : 0.0f;
                    const int liveBands = haveAudio
                        ? (int) std::min<size_t> (64, cachedAudioFeatures.bands.size()) : 0;
                    ctx.bands = liveBands > 0 ? liveAud.bands : nullptr;
                    ctx.bandCount = liveBands;
                    // Block C score: full links plus active and timeline views.
                    scriptLinks.clear();
                    for (const auto& l : localScore.links)
                    {
                        arbitlua::HookLink hl;
                        hl.slaveNoteId = l.slaveNoteId; hl.masterNoteId = l.masterNoteId;
                        hl.num = l.slaveHarmonic; hl.den = l.masterHarmonic;
                        hl.ratio = l.masterHarmonic != 0
                                 ? (double) l.slaveHarmonic / l.masterHarmonic : 0.0;
                        scriptLinks.push_back (hl);
                    }
                    scriptNotes.clear();
                    scriptTimelineNotes.clear();
                    const float beatF = (float) ctx.beat;
                    for (const auto& n : localScore.notes)
                    {
                        const bool active = n.activeAt (beatF);
                        const bool inTimeline = arbitmod::timelineResidency (
                            n, beatF, localScore.historyBeats, localScore.lookaheadBeats) > 0;
                        if (active || inTimeline)
                        {
                            arbitlua::HookNote hn;
                            hn.id = n.id;
                            hn.midi = n.midiNote; hn.freq = n.freqHz; hn.velocity = n.velocity;
                            hn.cents = arbitmod::centsFromRoot (n.freqHz, localScore.rootFreq);
                            hn.age = ctx.beat - n.startBeat;
                            hn.remain = n.endBeat() - ctx.beat;
                            hn.startBeat = n.startBeat;
                            hn.lengthBeats = n.lengthBeats;
                            hn.active = active;
                            hn.trackId = n.trackId; hn.ratioNum = n.ratioNum; hn.ratioDen = n.ratioDen;
                            hn.isRoot = n.isRoot;
                            for (int p = 0; p < 6; ++p) hn.primes[p] = n.primes[p];
                            if (active) scriptNotes.push_back (hn);
                            if (inTimeline) scriptTimelineNotes.push_back (hn);
                        }
                    }
                    ctx.notes = scriptNotes.empty() ? nullptr : scriptNotes.data();
                    ctx.noteCount = (int) scriptNotes.size();
                    ctx.timelineNotes = scriptTimelineNotes.empty()
                        ? nullptr : scriptTimelineNotes.data();
                    ctx.timelineNoteCount = (int) scriptTimelineNotes.size();
                    ctx.links = scriptLinks.empty() ? nullptr : scriptLinks.data();
                    ctx.linkCount = (int) scriptLinks.size();
                    scriptOverrides.clear();
                    std::string serr;
                    enterLoopStep ("advance:script-frame");
                    const bool sok = scriptWantJs
                        ? scriptJs.runFrame (ctx, scriptOverrides, serr)
                        : scriptLua.runFrame (ctx, scriptOverrides, serr);
                    if (! sok && ! scriptErrLogged)
                    {
                        std::fprintf (stderr, "[%s] live frame() error: %s\n",
                                      scriptWantJs ? "js" : "lua", serr.c_str());
                        scriptErrLogged = true;
                    }
                }
            }

            // M6: lay each routing targeting this clip over the per-frame param
            // copy (combine(mode, base, smoothed) — the same overlay the exporter
            // applies in stateAt). routingStates is already advanced to this
            // frame. A routing to an unknown/unreadable param is skipped (readFrom
            // returns false) rather than written onto a junk base. The stored
            // base (im.clipParams) is never mutated, so disabling/removing a
            // routing restores the underlying value next frame.
            auto overlayMods = [&] (ClipGraphParams& p, int clipId)
            {
                // Seed ISF INPUT defaults under any explicit gen value (emplace never
                // overwrites) so a routing to a never-set input modulates around the
                // shader's default — exactly as exporter.cpp::stateAt does. Defaults
                // come from the prior compile of this clip's shader (clipGenDefaults).
                if (const auto dit = clipGenDefaults.find (clipId); dit != clipGenDefaults.end())
                    for (const auto& [k, v] : dit->second)
                        p.genParams.emplace (k, v);
                if (frameVisualPlans != nullptr)
                {
                    videohelper::importedscene::seedImportedSceneRuntimeParameters(
                        frameVisualPlans->plans, clipId, p.visualParams);
                    if (const auto* plan = videowire::findVisualLayerPlan(
                            frameVisualPlans->plans, clipId))
                        videowire::seedFlatShaderRuntimeParameters(
                            frameVisualPlans->state.compiled(clipId, plan->structuralRevision),
                            p.visualParams);
                }
                for (size_t ri = 0; ri < localRoutings.size(); ++ri)
                {
                    const arbitmod::Routing& r = localRoutings[ri];
                    if (! r.enabled) continue;
                    int cid; std::string node, par;
                    if (! Impl::parseParamId (r.destination, cid, node, par)) continue;
                    if (cid != clipId) continue;
                    double base = 0.0;
                    if (! Impl::readFrom (p, node, par, base)) continue;
                    Impl::applyTo (p, node, par,
                                   arbitmod::combine (r.mode, (float) base,
                                                      routingStates[ri].smoothed));
                }
                for (const auto& sink : controlSinkValues)
                {
                    if (!sink.enabled) continue;
                    int cid; std::string node, par;
                    if (!Impl::parseParamId(sink.destination, cid, node, par) || cid != clipId)
                        continue;
                    double base = 0.0;
                    if (!Impl::readFrom(p, node, par, base)) continue;
                    Impl::applyTo(p, node, par,
                                  arbitmod::combine(sink.mode, static_cast<float>(base),
                                                    sink.value));
                }
                // P2: the per-frame script hook is the TOP layer — apply this
                // frame's overrides for THIS clip after static/baked/mod-matrix
                // (mirrors exporter.cpp::stateAt). scriptOverrides was computed
                // once for this timeline frame above; a script always wins where
                // it sets a value.
                for (const auto& [pid, value] : scriptOverrides)
                {
                    int cid; std::string node, par;
                    if (! Impl::parseParamId (pid, cid, node, par)) continue;
                    if (cid != clipId) continue;
                    Impl::applyTo (p, node, par, value);
                }
            };

            constexpr double kAbutEps = 1e-3;
            for (const auto& s : im.segments)
            {
                const double segDur = (s.outSec - s.inSec) / std::max (s.rate, 1e-9);
                if (! (displaySec >= s.displayStartSec
                       && displaySec < s.displayStartSec + segDur))
                    continue;

                ActiveLayer al;
                al.seg = s;
                al.params = *im.paramsFor (s.clipId);
                overlayMods (al.params, s.clipId);

                // Leading-edge transition window of this segment.
                if (s.transitionType != 0 && s.transitionDurationSec > 0.0
                    && displaySec < s.displayStartSec + s.transitionDurationSec)
                {
                    al.transitionActive = true;
                    al.transitionProgress = std::clamp (
                        (displaySec - s.displayStartSec) / s.transitionDurationSec,
                        0.0, 1.0);

                    const ViewportSegment* best =
                        videowire::resolveTransitionFrom (im.segments, s, kAbutEps);
                    if (best != nullptr)
                    {
                        al.haveFrom = true;
                        al.fromSeg = *best;
                        al.fromParams = *im.paramsFor (best->clipId);
                        overlayMods (al.fromParams, best->clipId);
                    }
                }
                act.push_back (std::move (al));
            }

            // --- Per-clip LUTs: service staged uploads/clears (GL context is
            // current on this thread).
            for (auto it = im.luts.begin(); it != im.luts.end();)
            {
                auto& l = it->second;
                if (l.dirty)
                {
                    if (l.path.empty() || l.pendingSize < 2)
                    {
                        if (l.tex != 0)
                            renderer.deleteTexture (l.tex);
                        l.tex = 0;
                        l.size = 0;
                    }
                    else
                    {
                        enterLoopStep ("advance:lut-upload");
                        l.tex = renderer.uploadLut3D (l.pendingData.data(),
                                                      l.pendingSize, l.tex);
                        l.size = l.pendingSize;
                    }
                    l.pendingData.clear();
                    l.pendingData.shrink_to_fit();
                    l.dirty = false;
                }
                if (l.path.empty() && l.tex == 0)
                    it = im.luts.erase (it);
                else
                    ++it;
            }

            // --- Text overlays: service uploads/removals (GL context is
            // current on this thread) and gather the time-active ones.
            for (auto it = im.texts.begin(); it != im.texts.end();)
            {
                auto& t = it->second;
                if (t.removed)
                {
                    if (t.tex != 0)
                        renderer.deleteTexture (t.tex);
                    it = im.texts.erase (it);
                    continue;
                }
                if (t.pixelsDirty)
                {
                    enterLoopStep ("advance:texture-upload");
                    t.tex = renderer.uploadRgba (t.pendingPixels.data(),
                                                 t.pendingW, t.pendingH,
                                                 t.pendingW * 4, t.tex);
                    t.texW = t.pendingW;
                    t.texH = t.pendingH;
                    t.pendingPixels.clear();
                    t.pendingPixels.shrink_to_fit();
                    t.pixelsDirty = false;
                }
                if (t.tex != 0 && t.visible >= 0.5f && t.opacity > 0.001f
                    && displaySec >= t.startSec
                    && displaySec < t.startSec + t.durationSec)
                {
                    videorender::ImageLayerDesc ov;
                    ov.texture = t.tex;
                    ov.width = t.texW;
                    ov.height = t.texH;
                    ov.posX = t.posX + t.translateX;
                    ov.posY = t.posY + t.translateY;
                    ov.opacity = t.opacity;
                    ov.zOrder = (int) t.zOrder;
                    ov.ownerClipId = t.ownerClipId;
                    overlays.push_back (ov);
                }
                ++it;
            }
        }

        // Texts composite above all video layers, ordered by zOrder (map
        // iteration keeps equal-zOrder texts stable by textId).
        std::stable_sort (overlays.begin(), overlays.end(),
                          [] (const videorender::ImageLayerDesc& a,
                              const videorender::ImageLayerDesc& b)
                          { return a.zOrder < b.zOrder; });

        // A segment serving as the A side of an active cross transition is
        // represented inside that transition — suppress its standalone layer
        // (matches the reference: during a transition the two frames render
        // blended, not stacked).
        {
            std::set<int> fromIds;
            for (const auto& al : act)
                if (al.haveFrom)
                    fromIds.insert (al.fromSeg.clipId);
            if (! fromIds.empty())
                act.erase (std::remove_if (act.begin(), act.end(),
                    [&] (const ActiveLayer& al)
                    {
                        return ! al.transitionActive
                            && fromIds.count (al.seg.clipId) > 0;
                    }), act.end());
        }

        // Composite order: (trackLayer, zOrder), displayStartSec for stability.
        std::sort (act.begin(), act.end(),
                   [] (const ActiveLayer& a, const ActiveLayer& b)
                   {
                       if (a.seg.trackLayer != b.seg.trackLayer)
                           return a.seg.trackLayer < b.seg.trackLayer;
                       if (a.params.zOrder != b.params.zOrder)
                           return a.params.zOrder < b.params.zOrder;
                       return a.seg.displayStartSec < b.seg.displayStartSec;
                   });

        // Publication replacement is independent of timeline replacement. Retire
        // stale native shader owners from the frame snapshot on every frame lease.
        std::set<int> orderedPlanClips;
        for (const auto& plan : frameVisualPlans->plans)
            if (const auto* execution = frameVisualPlans->state.compiled(
                    plan.clipId, plan.structuralRevision);
                videowire::scheduledShaderPlan(execution) != nullptr)
                orderedPlanClips.insert(plan.clipId);
        renderer.retainShaderPlanClips(orderedPlanClips);

        // Drop decode streams for clips that left the timeline.
        if (timelineGen != streamsGeneration)
        {
            for (auto it = streams.begin(); it != streams.end();)
            {
                if (liveClipIds.count (it->first) == 0)
                {
                    renderer.deleteTexture (it->second.tex);
                    renderer.deleteTexture (it->second.texA);   // frame-blend brackets
                    renderer.deleteTexture (it->second.texB);
#if ARBIT_HAVE_ONNX || ARBIT_HAVE_NCNN
                    if (im.interp != nullptr) im.interp->forget (it->first); // tier-2 decoder + cache
#endif
                    it = streams.erase (it);
                }
                else
                    ++it;
            }
            // Free compiled shader programs for clips that left the timeline.
            for (auto it = shaderCompiled.begin(); it != shaderCompiled.end();)
            {
                if (liveClipIds.count (it->first) == 0)
                {
                    renderer.clearClipShader (it->first);
                    it = shaderCompiled.erase (it);
                }
                else
                    ++it;
            }
            streamsGeneration = timelineGen;
        }

        if (! offscreen)
            glfwGetFramebufferSize (win, &fbW, &fbH);
        if (fbW <= 0 || fbH <= 0)
        {
            enterLoopStep ("advance:sleep-no-framebuffer");
            std::this_thread::sleep_for (std::chrono::milliseconds (50));
            continue;
        }
        // Resolution parity (audit #16): composite at the CANVAS resolution —
        // exactly what the export does — so pixel-scale effects (blur/bloom
        // radius, feedback trails, particle size) match it, then frame the result
        // into the panel/window at fbW×fbH via the present target. With no canvas
        // set, fall back to compositing at the window size (legacy, no present
        // target). The composite is byte-identical to the export's because both
        // run renderComposite with output == canvas and a canvas-aspect ref.
        const auto canvas = canvasExtent();
        const int cw = canvas.width;
        const int ch = canvas.height;
        if (cw > 0 && ch > 0)
        {
            renderer.setOutputSize (cw, ch);
            renderer.setPresentSize (fbW, fbH);
        }
        else
        {
            renderer.setOutputSize (fbW, fbH);
            renderer.setPresentSize (0, 0);
        }
        // Project canvas + view transform (PROTOCOL.md §Project canvas &
        // view transform) — viewport-only; the exporter's own FrameRenderer
        // never sets these.
        renderer.setCanvas (cw, ch);
        renderer.setView ((float) viewZoom_.load(),
                          (float) viewPanX_.load(),
                          (float) viewPanY_.load());
        // Frame-global render settings mirrored from the export job so preview ==
        // export (master-plan principle #1): the canvas background colour and the
        // bloom/tonemap post stack. Neutral defaults skip the post pass and clear
        // to the editor charcoal — byte-identical to before these were wired.
        renderer.setBackgroundColor ((float) bgR_.load(), (float) bgG_.load(),
                                     (float) bgB_.load(), (float) bgA_.load());
        // Lay any mod routing targeting the global "post/<node>/<param>" namespace
        // over the static post values (combine(mode, base, smoothed) — the same
        // overlay the exporter applies per frame), so e.g. a kick->bloom routing
        // pulses the bloom in the live preview exactly as it will on export.
        // routingStates was advanced to this frame while `act` was built above.
        float pBloomI = (float) bloomIntensity_.load();
        float pBloomT = (float) bloomThreshold_.load();
        float pBloomR = (float) bloomRadius_.load();
        float pExp    = (float) exposure_.load();
        for (size_t ri = 0; ri < localRoutings.size(); ++ri)
        {
            const arbitmod::Routing& r = localRoutings[ri];
            if (! r.enabled) continue;
            if (r.destination.rfind ("post/", 0) != 0) continue;
            float* dst = r.destination == "post/bloom/intensity" ? &pBloomI
                       : r.destination == "post/bloom/threshold" ? &pBloomT
                       : r.destination == "post/bloom/radius"    ? &pBloomR
                       : r.destination == "post/grade/exposure"  ? &pExp
                                                                  : nullptr;
            if (dst == nullptr) continue;   // tonemap mode stays static (an int select)
            *dst = arbitmod::combine (r.mode, *dst, routingStates[ri].smoothed);
        }
        renderer.setPostFx (pBloomI, pBloomT, pBloomR, tonemap_.load(), pExp);

        // --- Decode all visible layers and build the renderer's layer list.
        interpTierThisFrame = 0; // decodeLayer raises this to the tier it drew
        interpWarmingThisFrame = false;
        std::vector<videorender::LayerDesc> descs;
        std::vector<videorender::LayerDesc> fromDescs;
        std::vector<videohelper::sdf::NativeSdfRenderedFrame> sdfFrameOwners;
        std::vector<videohelper::modelpayload::ImportedAnimatedSceneReceipt>
            importedAnimationFrameOwners;
        std::vector<videohelper::modelpayload::ImportedSceneExecutionReceipt>
            importedSceneFrameOwners;
        descs.reserve (act.size());
        fromDescs.reserve (act.size()); // stable: fromLayer pointers held by descs
        sdfFrameOwners.reserve (act.size());
        importedAnimationFrameOwners.reserve(act.size());
        importedSceneFrameOwners.reserve(act.size());
        const videowire::VisualInspectionTarget inspectionTarget {
            im.inspectionClipId, im.inspectionStructuralRevision,
            im.inspectionNodeId, im.inspectionOutputPort };
        videowire::VisualInspectionResource inspectionResource;
        renderer.setTransformInspectionClip (-1);
        if (const auto* inspectionPlan = videowire::findVisualLayerPlan(
                frameVisualPlans->plans, inspectionTarget.clipId);
            inspectionPlan != nullptr
            && inspectionPlan->structuralRevision == inspectionTarget.structuralRevision
            && (videowire::classifyVisualInspectionTarget(*inspectionPlan, inspectionTarget)
                    == videowire::VisualInspectionSlice::transformedLayer
                || videowire::classifyVisualInspectionTarget(*inspectionPlan, inspectionTarget)
                    == videowire::VisualInspectionSlice::retainedDrawShape))
            renderer.setTransformInspectionClip (inspectionTarget.clipId);
        bool haveMain = false;
        double mainPts = 0.0, mainIdeal = 0.0;

        // Prepare both visible generators and exact Layer Source references on
        // the render thread. A referenced clip may not have rendered on its own.
        std::set<int> shaderClips;
        bool frameDeferred = false;
        // Set when an imported scene Preview is still compiling. The last frame
        // holds, and no plan or renderer error is recorded.
        bool importedPendingThisFrame = false;
        // Source of the pending deferral in this frame. A fixture-backend pending result
        // on this thread since the frame started means the GL fixture program was pending.
        std::string_view deferredSource = "other";
        const auto fixturePendingAtFrameStart = videorender::fixture3d::previewPendingResultsOnThisThread();
        const auto pendingSource = [&] (std::string_view importedSource)
        {
            return videorender::fixture3d::previewPendingResultsOnThisThread() != fixturePendingAtFrameStart
                ? std::string_view ("gl-fixture") : importedSource;
        };
        // A Geometry Core Preview prepare can return pending on this thread. As with the
        // fixture case, that defers the frame without recording a plan error.
        const auto geometryPendingAtFrameStart = videohelper::geometry::geometryCorePreviewPendingResultsOnThisThread();
        const auto geometryPendingThisFrame = [&]
        {
            return videohelper::geometry::geometryCorePreviewPendingResultsOnThisThread() != geometryPendingAtFrameStart;
        };
        // Cleared when a layer or transition source could not be decoded this
        // tick. Such a frame can change later with no new input.
        bool frameSettled = true;
        // A plan or binding failure holds the last presented frame and keeps the
        // channel open. The error stays on rendererError until a frame presents.
        const auto holdFrameForPlanError = [&](const std::string& error)
        {
            if (geometryPendingThisFrame())
            {
                // Pending defers the frame. It records no plan error, and the retained
                // cache keeps its entries, so the next tick retries the same plan.
                deferredSource = "geometry-core";
                frameDeferred = true;
                return;
            }
            planFrameError = error.empty() ? std::string("visual plan execution failed") : error;
            std::lock_guard<std::mutex> lock(im.mutex);
            im.rendererError = planFrameError;
            // A held plan error retries every tick until a frame presents. It is not a GL wait.
            deferredSource = "plan-error";
            frameDeferred = true;
        };
        const auto prepareShaderClip = [&](const ViewportSegment& segment, std::string& error)
        {
            if (segment.shaderSource.empty())
            {
                error = "generator source is unavailable";
                return false;
            }
            auto& compiled = shaderCompiled[segment.clipId];
            if (compiled.source != segment.shaderSource
                || (compiled.succeeded && !renderer.hasClipShader(segment.clipId)))
            {
                std::vector<videorender::GenParam> shaderParams;
                enterLoopStep ("advance:shader-compile");
                compiled.succeeded = renderer.setClipShader(segment.clipId, segment.shaderSource,
                                                             compiled.log, shaderParams);
                compiled.source = segment.shaderSource;

                // Capture this shader's ISF INPUT defaults (per component key, as
                // exporter.cpp does) so mod routings to an un-set input find a base.
                auto& defs = clipGenDefaults[segment.clipId];
                defs.clear();
                for (const auto& gp : shaderParams)
                {
                    const int nc = videorender::genParamComponentCount (gp.type);
                    for (int k = 0; k < nc; ++k)
                        defs[gp.name + videorender::genParamComponentSuffix (gp.type, k)] =
                            (nc == 1) ? gp.defaultV : gp.defaultVec[k];
                    if (gp.type == 5)
                    {
                        std::string path = gp.importedPath;
                        if (const auto image = segment.genImages.find(gp.name);
                            image != segment.genImages.end() && ! image->second.empty())
                            path = image->second;
                        if (! path.empty())
                        {
                            MediaContext media;
                            DecodedFrame frame;
                            enterLoopStep ("advance:gen-image-decode");
                            if (media.open(path, false).empty()
                                && media.getFrame(0.0, cw > 0 ? cw : 1920,
                                                  ch > 0 ? ch : 1080, frame).empty()
                                && ! frame.rgba.empty())
                                renderer.setClipImage(segment.clipId, gp.name,
                                    frame.rgba.data(), frame.width, frame.height,
                                    frame.strideBytes);
                        }
                    }
                }
            }
            if (!compiled.succeeded || !renderer.hasClipShader(segment.clipId))
            {
                error = compiled.log.empty() ? "generator program is unavailable" : compiled.log;
                return false;
            }
            return true;
        };
        for (const auto& al : act)
        {
            if (hasNativeVisualSource(frameVisualPlans->plans, al.seg.clipId)) continue;
            if (videowire::resolveSourceKind(al.seg.sourceKind, al.seg.isAdjustment, al.seg.sourcePath)
                    != videowire::SourceKind::Shader || al.seg.shaderSource.empty())
                continue;
            std::string shaderError;
            prepareShaderClip(al.seg, shaderError);
            // Visible editor previews retain setClipShader's last good program.
            if (renderer.hasClipShader (al.seg.clipId))
                shaderClips.insert (al.seg.clipId);
        }

        // M5: pack the live Block C score once per timeline frame for every
        // consumer, including imported scenes. The packed beat
        // matches makeShaderClock's shared beat timeline; re-packing a held frame is
        // skipped via scorePackedFrame so a paused/repeated frame stays stable.
        // A published empty score still needs a zero-row frame for note-instanced scenes.
        const bool haveScore = ! localScore.notes.empty() || localScore.scoreRevision != 0;
        if (haveScore)
        {
            const double clkFps = valueFps;  // project value-grid fps (== export job.fps)
            const long g = (long) std::llround (displaySec * clkFps);
            enterLoopStep ("advance:block-c-frame");
            cachedCanonicalBlockCFrame = scoreFrameOwner.frameAt(
                canonicalblockc::PreviewFrameRequest { scoreIdentity, g, clkFps },
                canonicalScoreSource,
                [&] (long frame)
                {
                    return activeBeatTimeline.secondsToBeat(
                        static_cast<double>(frame) / clkFps);
                });
            if (cachedCanonicalBlockCFrame == nullptr)
            {
                auto key = canonicalblockc::makePreviewFrameKey(
                    canonicalblockc::PreviewFrameRequest { scoreIdentity, g, clkFps });
                key.frame = g;
                key.fps = clkFps;
                const auto rejectedBeat = static_cast<float>(activeBeatTimeline.secondsToBeat(
                    static_cast<double>(g) / clkFps));
                key.beat = static_cast<double>(rejectedBeat);
                std::string diagnostic;
                canonicalblockc::explainAdmissibility(
                    key, canonicalScoreSource, rejectedBeat, diagnostic);
                const std::string failure = diagnostic.empty() ? "invalid Block C frame"
                    : "invalid Block C frame: " + diagnostic;
                if (scoreFrameError != failure)
                {
                    scoreFrameError = failure;
                    std::fprintf(stderr, "[viewport] %s\n", scoreFrameError.c_str());
                }
                {
                    std::lock_guard<std::mutex> lock(im.mutex);
                    im.rendererError = scoreFrameError;
                }
                // Keep servicing score/lifecycle updates, without busy-spinning
                // past normal presentation pacing or holding the control mutex.
                enterLoopStep ("advance:block-c-retry-sleep");
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                nextTickNs = nowNs();
                continue;
            }
        }
        if (!scoreFrameError.empty())
        {
            std::lock_guard<std::mutex> lock(im.mutex);
            if (im.rendererError == scoreFrameError) im.rendererError.clear();
            scoreFrameError.clear();
        }

        videowire::VisualPlanEvaluationContext visualEvaluationContext {
            videowire::geometryCoreHelperGeneration(),
            ! playing, visualtemporalsampling::EvaluationMode::preview,
            videohelper::viewportgeneration::projectGeneration(timelineGen),
            videowire::geometryCoreDeviceGeneration(), scoreIdentity.loopGeneration, scoreIdentity.seekGeneration,
            [&](auto& geometry, double seconds, std::uint64_t revision, bool exporting, std::string& diagnostic) {
                visualdeformation::RationalFrameTime frame;
                return videohelper::importedanimation::importedAnimationFrameIdentity(seconds,valueFps,frame,diagnostic)
                    && importedGeometryExecution.evaluate(geometry,frame,revision,exporting,diagnostic,&activeBeatTimeline);
            },
            [&](double seconds, auto& frame, std::string& diagnostic) {
                return videohelper::importedanimation::importedAnimationFrameIdentity(seconds,valueFps,frame,diagnostic);
            }
        };
        // The viewport is the only Preview owner. It may defer a Geometry Core shader that is still compiling.
        visualEvaluationContext.geometryPrepareUse = arbitgpu::NativeFixturePrepareUse::Preview;
        const auto opticalHelperGeneration = scoreIdentity.helperGeneration != 0
            ? scoreIdentity.helperGeneration : visualEvaluationContext.helperGeneration;
        if (opticalFlowTimelineSeen != timelineGen
            || opticalFlowSeekSeen != scoreIdentity.seekGeneration
            || opticalFlowLoopSeen != scoreIdentity.loopGeneration
            || opticalFlowHelperSeen != opticalHelperGeneration)
        {
            enterLoopStep ("advance:optical-cancel");
            opticalFlowDecoder.cancel();
            opticalFlowFields.clear();
            opticalFlowFieldKeys.clear();
            opticalDecodeOwners.clear();
            opticalUploadOwners.clear();
            opticalFlowTimelineSeen = timelineGen;
            opticalFlowSeekSeen = scoreIdentity.seekGeneration;
            opticalFlowLoopSeen = scoreIdentity.loopGeneration;
            opticalFlowHelperSeen = opticalHelperGeneration;
        }
        videowire::TemporalSamplingCommitTransaction temporalSamplingTransaction;
        const auto rendererOwner = static_cast<std::uint64_t>(
            reinterpret_cast<std::uintptr_t>(&frameVisualPlans->state));
        std::string framePublicationError;
        enterLoopStep ("advance:frame-publication-begin");
        auto framePublication = renderer.beginFramePublicationCandidate(
            rendererOwner == 0 ? 1 : rendererOwner,
            videohelper::viewportgeneration::projectGeneration(timelineGen),
            {}, framePublicationError);
        if (! framePublication)
        {
            std::lock_guard<std::mutex> lock(im.mutex);
            im.rendererError = framePublicationError;
            continue;
        }
        auto framePublicationHolder = std::make_shared<
            videorender::FrameRenderer::FramePublicationCandidatePtr>(framePublication);
        temporalSamplingTransaction.addPublication(
            [&renderer, framePublicationHolder](std::string& error)
            { return renderer.validateFramePublicationCandidate(*framePublicationHolder, error); },
            [&renderer, framePublicationHolder]
            { renderer.promoteFramePublicationCandidate(*framePublicationHolder); },
            [&renderer, framePublicationHolder]
            { renderer.discardFramePublicationCandidate(*framePublicationHolder); });

        auto particleHistory = std::make_shared<videorender::ParticleHistorySource>();
        particleHistory->score = canonicalScoreSource ? canonicalScoreSource : std::make_shared<const arbitmod::Score>();
        particleHistory->timeline = activeBeatTimeline;
        particleHistory->analysisSources = localPreviewSourceFeats;
        particleHistory->parameterFrameSeconds=1.0/valueFps;
        particleHistory->key = canonicalblockc::makeFrameKey(scoreIdentity);
        particleHistory->importedGeometry = [&, beatTimeline = particleHistory->timeline](auto& geometry, const auto& frame, std::uint64_t revision, std::string& error)
        { return importedGeometryExecution.evaluate(geometry,frame,revision,false,error,&beatTimeline); };
        if (!localPreviewFeats.empty() && localPreviewSr > 0)
        { particleHistory->audioFrames = &localPreviewFeats; particleHistory->audioSampleRate = localPreviewSr; }
        const auto historyIdentity=std::make_tuple(timelineGen,scoreGenSeen,modGenSeen,
            controlPlanGenSeen,previewMixGenSeen,beatTimelineGenSeen,valueFps);
        if (!parameterHistory || parameterHistoryIdentity!=historyIdentity)
        {
            parameterHistory=std::make_shared<videorender::VisualParameterHistory>();
            parameterHistoryIdentity=historyIdentity;
            parameterHistoryError.clear();
            enterLoopStep ("advance:parameter-history-bind");
            parameterHistory->bind(localRoutings,localControlPlan,particleHistory->score,
                activeBeatTimeline,valueFps,
                videowire::simulationAutomationDestinations(frameVisualPlans->plans),parameterHistoryError);
        }
        const auto historyAudio=*particleHistory;
        particleHistory->parameterAt=[parameterHistory,historyAudio,parameterHistoryError,localParameterTimeline]
            (const std::string& destination,double seconds,double& value,std::string& error)
        {
            if (!parameterHistoryError.empty()) { error=parameterHistoryError; return false; }
            if (localParameterTimeline) localParameterTimeline->sample(destination,seconds,value);
            return parameterHistory->sample(destination,seconds,value,
                [&historyAudio](double time,const std::vector<int>& selected,bool master,arbitmod::Audio& audio,std::string& diagnostic)
                { return videorender::particleHistoryModAudio(historyAudio,time,selected,master,audio,diagnostic); },error);
        };
        const auto attachGeometryAudio = [&](videorender::LayerDesc& desc)
        {
            desc.particleHistory = particleHistory;
            desc.particleProjectSeconds = displaySec;
            desc.audioPresent = haveAudio;
            desc.audioFeatures = cachedAudioFeatures;
            // History reads project time while playing as well as scrubbing.
            // Other audio consumers retain the live frame above.
            if (!localPreviewFeats.empty())
            {
                desc.spectrumHistoryFeaturesAt = [&](double seconds)
                {
                    videowire::geometry::spectrum::Bands bands {};
                    const auto features = previewMixFeatureAt(seconds);
                    std::copy_n(features.bands.begin(), std::min(features.bands.size(), bands.size()),
                                bands.begin());
                    return bands;
                };
                // Scalars read the same frame as the band reader above.
                desc.spectrumHistoryScalarFeaturesAt = videohelper::analysisScalarsReader(localPreviewFeats, localPreviewSr);
            }
            if (useBakedAudio)
            {
                desc.spectrumFeaturesAt=desc.spectrumHistoryFeaturesAt;
                desc.spectrumScalarFeaturesAt=desc.spectrumHistoryScalarFeaturesAt;
            }
            desc.spectrumSourceFeaturesAt=videohelper::sourceSpectrumReader(localPreviewSourceFeats);
            desc.spectrumSourceScalarFeaturesAt=videohelper::sourceScalarsReader(localPreviewSourceFeats);
        };
        auto prepareDecodedDesc = [&] (const ViewportSegment& segment,
                                       const ClipGraphParams& params,
                                       double sourceSec,
                                       videorender::LayerDesc& desc,
                                       ClipStream*& stream) -> bool
        {
            std::string planError;
            const auto rejectPlan = [&]
            {
                if (!geometryPendingThisFrame())
                    std::fprintf(stderr, "[viewport] visual plan execution rejected: %s\n",
                                 planError.c_str());
                // A rejected SDF plan holds the last frame instead of closing the channel.
                holdFrameForPlanError(planError);
                return false;
            };
            fillDescCommon(desc, params, displaySec, segment.clipId);
            enterLoopStep ("advance:sdf-prepare");
            const auto sdfResult = videohelper::sdf::prepareVisualSdfLayer(
                frameVisualPlans->plans, segment.clipId,
                cw > 0 ? cw : fbW, ch > 0 ? ch : fbH,
                videohelper::sdf::NativeSdfRenderUse::Preview,
                videohelper::sdf::nativeSdfRenderer(), desc, sdfFrameOwners, planError,
                { videohelper::viewportgeneration::projectGeneration(timelineGen), 0,
                  visualEvaluationContext.helperGeneration, segment.clipId });
            if (sdfResult != videohelper::sdf::VisualSdfPreparation::notPresent)
            {
                stream = nullptr;
                if (sdfResult == videohelper::sdf::VisualSdfPreparation::rendered)
                    return true;
                return rejectPlan();
            }
            enterLoopStep ("advance:volume-prepare");
            const auto volumeResult = videohelper::volume::prepareVisualVolumeLayer(
                frameVisualPlans->plans, segment.clipId, cw > 0 ? cw : fbW, ch > 0 ? ch : fbH,
                videohelper::volume::NativeVolumeRenderUse::Preview,
                renderer.volumeRenderCache(), desc, planError);
            if (volumeResult != videohelper::volume::VisualVolumePreparation::notPresent)
            {
                stream = nullptr;
                if (volumeResult == videohelper::volume::VisualVolumePreparation::rendered
                    && videowire::executeVisualLayerPlanForRenderer(
                        renderer, frameVisualPlans->plans, segment.clipId, desc, planError,
                        videohelper::geometry::PlanUse::preview, &temporalSamplingTransaction,
                        &inspectionTarget, &inspectionResource,
                        &frameVisualPlans->state, displaySec,
                        &frameVisualScheduleBindings, &visualEventCursor,
                        &visualEvaluationContext, &params.visualParams))
                    return true;
                return rejectPlan();
            }
            if (videowire::isImportedParticleOverlayPlan(frameVisualPlans->plans, segment.clipId)
                || videowire::isGeometryCoreRenderPlan(frameVisualPlans->plans, segment.clipId))
            {
                const double clockStart = segment.clockStartSec >= 0.0
                    ? segment.clockStartSec : segment.displayStartSec;
                const double clockDuration = segment.clockDurationSec > 0.0
                    ? segment.clockDurationSec : (segment.outSec - segment.inSec) / std::max(segment.rate, 1e-9);
                desc.shaderClock = videorender::makeShaderClock(displaySec, clockStart, clockDuration,
                    activeBeatTimeline, valueFps, playing,
                    static_cast<int>(std::llround((displaySec - clockStart) * valueFps)));
            }
            attachGeometryAudio(desc);
            if (haveScore)
                desc.canonicalBlockCFrame = cachedCanonicalBlockCFrame;
            desc.materialFrameStartSeconds = segment.displayStartSec;
            desc.materialInputsAt = [&](double at, videorender::LayerDesc& sampled,
                std::map<std::string,double>& values, std::string& failure) {
                if (scriptActive || !localRoutings.empty() || !controlSinkValues.empty())
                { failure = "Historical Surface shaders require baked parameter history; live routing/script history is unavailable"; return false; }
                {
                    std::lock_guard<std::mutex> lock(im.mutex);
                    const auto authored = im.clipParams.find(segment.clipId);
                    if (authored != im.clipParams.end() && !authored->second.visualParams.empty())
                    { failure = "Historical Surface shader parameter overrides need an exact project automation history"; return false; }
                }
                if (haveAudio && localPreviewFeats.empty())
                { failure = "Historical Surface shaders require baked project audio"; return false; }
                const double begin = segment.clockStartSec >= 0 ? segment.clockStartSec : segment.displayStartSec;
                const double duration = segment.clockDurationSec > 0 ? segment.clockDurationSec
                    : (segment.outSec-segment.inSec)/std::max(segment.rate,1e-9);
                sampled.shaderClock = videorender::makeShaderClock(at,begin,duration,activeBeatTimeline,
                    valueFps,playing,static_cast<int>(std::llround((at-begin)*valueFps)));
                sampled.audioFeatures = previewMixFeatureAt(at);
                sampled.audioPresent = !sampled.audioFeatures.bands.empty();
                sampled.canonicalBlockCFrame.reset();
                if (haveScore) {
                    canonicalblockc::FrameProducer producer;
                    auto key = particleHistory->key; key.frame = std::llround(at*valueFps); key.fps = valueFps;
                    key.beat = activeBeatTimeline.secondsToBeat(at);
                    sampled.canonicalBlockCFrame = producer.evaluate(key,particleHistory->score,key.beat);
                    if (!sampled.canonicalBlockCFrame) { failure = "Historical Surface score sample is unavailable"; return false; }
                }
                values.clear(); return true; // Exact immutable graph-authored shader parameters.
            };
            const videohelper::importedscene::MaterialFrameResolver materialFrameResolver =
                [&] (const videohelper::importedscene::MaterialFrameEvaluation& evaluation,
                     videohelper::importedscene::MaterialFrameReceipt& receipt,
                     std::string& frameError)
                {
                    auto requested = segment;
                    double requestedSource = sourceSec;
                    videosourcecolor::Policy sourcePolicy;
                    const auto* sourcePlan = videowire::findVisualLayerPlan(frameVisualPlans->plans,segment.clipId);
                    if (!sourcePlan || !videowire::materialframe::sourcePolicy(*sourcePlan,evaluation.endpoint,sourcePolicy,frameError)) return false;
                    if (videowire::isGeometryCoreRenderPlan(frameVisualPlans->plans,segment.clipId)) {
                        const auto* plan = videowire::findVisualLayerPlan(frameVisualPlans->plans,segment.clipId);
                        int sourceClip = 0;
                        if (!plan || !videowire::materialframe::sourceClip(*plan,evaluation.endpoint,sourceClip,frameError)) return false;
                        if (sourceClip != segment.clipId) {
                            std::lock_guard<std::mutex> lock(im.mutex);
                            const auto found = std::find_if(im.segments.begin(),im.segments.end(),
                                [&](const auto& source) { return source.clipId == sourceClip; });
                            if (found == im.segments.end()) { frameError = "Surface Layer Source clip is unavailable"; return false; }
                            requested = *found;
                        }
                        const double at = double(evaluation.frame.frame)*evaluation.frame.rateDenominator/evaluation.frame.rateNumerator;
                        requestedSource = requested.inSec + (at-requested.displayStartSec)*requested.rate;
                        if (requestedSource < requested.inSec || requestedSource >= requested.outSec)
                        { frameError = "Surface Layer Source is outside its saved media interval"; return false; }
                    }
                    enterLoopStep ("advance:material-decode");
                    auto* decoded = decodeLayer(requested, requestedSource, true, valueFps, true);
                    if (decoded == nullptr)
                    { frameError = "Material Frame source could not decode the exact preview time or publish its requested optical-flow/colour receipt"; return false; }
                    auto lease = decoded->hdrFrame.width > 0
                        ? renderer.leaseDecodedHdrFrameForMaterial(decoded->hdrFrame,sourcePolicy,frameError)
                        : renderer.leaseDecodedFrame(decoded->tex, decoded->texW,
                            decoded->texH, decoded->frameColor, frameError);
                    if (!lease) return false;
                    receipt = { evaluation, std::move(lease) };
                    return true;
                };
            if (videowire::isGeometryCoreRenderPlan(frameVisualPlans->plans, segment.clipId))
            {
                const auto plan = std::find_if(frameVisualPlans->plans.begin(), frameVisualPlans->plans.end(),
                    [&](const auto& candidate) { return candidate.clipId == segment.clipId; });
                if (plan != frameVisualPlans->plans.end() && localPreviewFeats.empty()
                    && videowire::requiresBakedSpectralHistory(*plan))
                {
                    std::lock_guard<std::mutex> lock(im.mutex);
                    im.rendererError = "Spectral History Field is waiting for the project's baked audio analysis";
                    spectrumBakePending = true;
                    deferredSource = "spectrum-bake";
                    return false;
                }
                // A stopped spectrum field reads the baked mix. Until the mix arrives,
                // the frame waits instead of failing the binding and closing the channel.
                if (plan != frameVisualPlans->plans.end() && ! playing && localPreviewFeats.empty()
                    && videowire::requiresBakedSpectrumAnalysis(*plan))
                {
                    std::lock_guard<std::mutex> lock(im.mutex);
                    im.rendererError = "Audio Spectrum Field is waiting for the project's baked audio analysis";
                    spectrumBakePending = true;
                    deferredSource = "spectrum-bake";
                    frameDeferred = true;
                    return false;
                }
                stream = nullptr;
                auto frameContext = visualEvaluationContext;
                frameContext.materialFrameResolver = materialFrameResolver;
                enterLoopStep ("advance:plan-execute");
                const bool rendered = videowire::executeVisualLayerPlanForRenderer(
                    renderer, frameVisualPlans->plans, segment.clipId, desc, planError,
                    videohelper::geometry::PlanUse::preview, &temporalSamplingTransaction,
                    &inspectionTarget, &inspectionResource, &frameVisualPlans->state, displaySec,
                    &frameVisualScheduleBindings, &visualEventCursor,
                    &frameContext, &params.visualParams);
                desc.materialInputsAt = {};
                // A failed plan holds the last presented frame. Closing the channel
                // here ends the Metal surface stream for every later frame.
                if (!rendered) holdFrameForPlanError(planError);
                return rendered;
            }
            const auto* importedFramePlan = videowire::findVisualLayerPlan(frameVisualPlans->plans,segment.clipId);
            const auto* importedFrameCompiled = importedFramePlan
                ? frameVisualPlans->state.compiled(segment.clipId,importedFramePlan->structuralRevision) : nullptr;
            if (importedFrameCompiled && importedFrameCompiled->importedSceneRender && importedFrameCompiled->materialFrameSource)
            {
                const double begin = segment.clockStartSec >= 0 ? segment.clockStartSec : segment.displayStartSec;
                const double duration = segment.clockDurationSec > 0 ? segment.clockDurationSec
                    : (segment.outSec-segment.inSec)/std::max(segment.rate,1e-9);
                desc.shaderClock = videorender::makeShaderClock(displaySec,begin,duration,activeBeatTimeline,
                    valueFps,playing,static_cast<int>(std::llround((displaySec-begin)*valueFps)));
            }
            const auto importedFrameInputs = desc;
            std::optional<videohelper::importedscene::ImportedMaterialFrameResolver> importedFrameResolver;
            if (importedFrameCompiled && importedFrameCompiled->importedSceneRender && importedFrameCompiled->materialFrameSource)
            {
                std::vector<ViewportSegment> savedSegments;
                { std::lock_guard<std::mutex> lock(im.mutex); savedSegments = im.segments; }
                auto invocationContext = visualEvaluationContext;
                invocationContext.projectGeneration = videohelper::viewportgeneration::projectGeneration(timelineGen);
                const auto activity = [&im, &renderer, frameVisualPlans, timelineGen, scoreIdentity,
                    scoreGenSeen, beatTimelineGenSeen, helper = visualEvaluationContext.helperGeneration,
                    device = visualEvaluationContext.deviceGeneration, framePublication](std::string& failure) {
                    {
                        std::lock_guard<std::mutex> lock(im.mutex);
                        if (im.wantClose.load() || im.timelineGeneration != timelineGen
                            || im.visualPlanPublication.shared() != frameVisualPlans
                            || im.scoreGeneration != scoreGenSeen
                            || im.scoreIdentity.seekGeneration != scoreIdentity.seekGeneration
                            || im.scoreIdentity.loopGeneration != scoreIdentity.loopGeneration
                            || im.scoreIdentity.sourceGeneration != scoreIdentity.sourceGeneration
                            || im.scoreIdentity.projectGeneration != scoreIdentity.projectGeneration
                            || im.scoreIdentity.helperGeneration != scoreIdentity.helperGeneration
                            || im.scoreIdentity.deviceGeneration != scoreIdentity.deviceGeneration
                            || im.scoreIdentity.backendGeneration != scoreIdentity.backendGeneration
                            || im.scoreIdentity.scoreGeneration != scoreIdentity.scoreGeneration
                            || im.scoreIdentity.beatMapGeneration != scoreIdentity.beatMapGeneration
                            || im.scoreIdentity.fpsGeneration != scoreIdentity.fpsGeneration
                            || im.beatTimelineGeneration != beatTimelineGenSeen)
                        { failure = kPreviewLifecycleRetired; return false; }
                    }
                    if (videowire::geometryCoreHelperGeneration() != helper
                        || videowire::geometryCoreDeviceGeneration() != device)
                    { failure = "Imported Frame preview helper or device was retired"; return false; }
                    return renderer.validateFramePublicationCandidate(framePublication,failure);
                };
                importedFrameResolver.emplace(videohelper::importedscene::ImportedMaterialFrameResolver::make<
                    videohelper::importedscene::NativeImportedSceneRenderUse::Preview>(renderer,
                    *importedFramePlan,*importedFrameCompiled,segment,std::move(savedSegments),displaySec,sourceSec,valueFps,
                    importedFrameInputs,&params.visualParams,invocationContext,temporalSamplingTransaction,activity,
                    [&](const auto& selected,double at,const auto&,std::string&) { return decodeLayer(selected,at,true,valueFps,true); },
                    [&](const auto& decoded,const auto& policy,std::string& failure) {
                        return decoded.hdrFrame.width > 0
                            ? renderer.leaseDecodedHdrFrameForMaterial(decoded.hdrFrame,policy,failure)
                            : renderer.leaseDecodedFrame(decoded.tex,decoded.texW,decoded.texH,decoded.frameColor,failure);
                    }));
            }
            enterLoopStep ("advance:imported-scene-exec");
            const auto importedSceneResult =
                videohelper::importedscene::prepareVisualImportedSceneLayerAtTime(
                    frameVisualPlans->plans, segment.clipId,
                    cw > 0 ? cw : fbW, ch > 0 ? ch : fbH,
                    sourceSec, valueFps,
                    videohelper::importedscene::NativeImportedSceneRenderUse::Preview,
                    importedSceneExecution.get(), &importedScenePlanCache, desc,
                    importedSceneFrameOwners, planError, &params.visualParams,
                    { videohelper::viewportgeneration::projectGeneration(timelineGen), visualEvaluationContext.helperGeneration }, &materialFrameResolver,
                    linearSharedActive, &activeBeatTimeline, displaySec, nullptr,
                    importedFrameResolver ? &*importedFrameResolver : nullptr);
            desc.materialInputsAt = {};
            if (importedSceneResult != videohelper::importedscene::
                                           VisualImportedScenePreparation::notApplicable)
            {
                stream = nullptr;
                enterLoopStep ("advance:imported-scene-plan-exec");
                if (importedSceneResult == videohelper::importedscene::
                                              VisualImportedScenePreparation::pending)
                {
                    // Pending holds the last presented frame and retries next tick.
                    importedPendingThisFrame = true;
                    frameDeferred = true;
                    deferredSource = pendingSource ("imported-scene");
                    return false;
                }
                if (importedSceneResult == videohelper::importedscene::
                                              VisualImportedScenePreparation::rendered
                    && videowire::executeVisualLayerPlanForRenderer(
                        renderer, frameVisualPlans->plans, segment.clipId, desc, planError,
                        videohelper::geometry::PlanUse::preview, &temporalSamplingTransaction,
                        &inspectionTarget, &inspectionResource,
                        &frameVisualPlans->state, displaySec,
                        &frameVisualScheduleBindings, &visualEventCursor,
                        &visualEvaluationContext, &params.visualParams))
                    return true;
                return rejectPlan();
            }
            enterLoopStep ("advance:imported-animation-exec");
            const auto importedResult =
                videohelper::importedanimation::prepareVisualImportedAnimationLayer(
                    frameVisualPlans->plans, segment.clipId,
                    cw > 0 ? cw : fbW, ch > 0 ? ch : fbH,
                    sourceSec, valueFps,
                    videohelper::importedanimation::NativeImportedAnimationRenderUse::Preview,
                    importedAnimationExecution.get(), desc,
                    importedAnimationFrameOwners, planError, &activeBeatTimeline, displaySec);
            if (importedResult == videohelper::importedanimation::
                                      VisualImportedAnimationPreparation::rendered)
            {
                stream = nullptr;
                return true;
            }
            if (importedResult == videohelper::importedanimation::
                                      VisualImportedAnimationPreparation::pending)
            {
                // Pending holds the last presented frame and retries next tick.
                importedPendingThisFrame = true;
                frameDeferred = true;
                deferredSource = pendingSource ("imported-animation");
                return false;
            }
            if (importedResult == videohelper::importedanimation::
                                      VisualImportedAnimationPreparation::rejected)
                return false;
            enterLoopStep ("advance:decode-layer");
            stream = decodeLayer(segment, sourceSec);
            if (stream == nullptr) return false;
            fillDesc(desc, params, *stream, displaySec, segment.clipId);
            if (! uploadTypedMatte(segment, *stream, sourceSec))
                planError = "typed matte asset frame is unavailable";
            else
            {
                desc.matteTexture = stream->matteTex;
                desc.matteWidth = stream->matteTexW;
                desc.matteHeight = stream->matteTexH;
                const bool reusePrimaryMatte = videowire::visualPlanReusesPrimaryMatte(
                    frameVisualPlans->plans, segment.clipId);
                desc.matteTextureB = reusePrimaryMatte ? stream->matteTex : stream->matteTexB;
                desc.matteWidthB = reusePrimaryMatte ? stream->matteTexW : stream->matteTexWB;
                desc.matteHeightB = reusePrimaryMatte ? stream->matteTexH : stream->matteTexHB;
                enterLoopStep ("advance:depth-tracking-exec");
                if (!videohelper::prepareDepthTexture(im.depthCacheRoot, frameVisualPlans->plans,
                        segment.clipId, sourceSec, 1, renderer, stream->depth, desc, planError))
                {
                    // Executable depth is strict; the caller closes the viewport below.
                }
                else if (!videohelper::trackingruntime::prepareTracking(im.depthCacheRoot,
                        frameVisualPlans->plans, segment.clipId, sourceSec, desc, planError))
                {
                    // Tracking receipts are equally strict and never auto-regenerated.
                }
                else if (videowire::executeVisualLayerPlanForRenderer(
                        renderer, frameVisualPlans->plans, segment.clipId, desc, planError,
                        videohelper::geometry::PlanUse::preview, &temporalSamplingTransaction,
                        &inspectionTarget, &inspectionResource,
                        &frameVisualPlans->state, displaySec,
                        &frameVisualScheduleBindings, &visualEventCursor,
                        &visualEvaluationContext, &params.visualParams))
                    return true;
            }
            return rejectPlan();
        };

        auto opticalCompletion = opticalFlowDecoder.takeCompletion();
        std::set<int> activeOpticalClips;
        for (const auto& al : act)
        {
            if (al.params.visible < 0.5f || al.params.opacity <= 0.001f)
                continue;

            if (videowire::isOpticalFlowPlan (frameVisualPlans->plans, al.seg.clipId))
            {
                activeOpticalClips.insert (al.seg.clipId);
                const auto* plan = videowire::findVisualLayerPlan (
                    frameVisualPlans->plans, al.seg.clipId);
                const videoopticalflow::PreviewDecodeCoordinator::Key key {
                    timelineGen,
                    frameOpticalSnapshot ? frameOpticalSnapshot->snapshotGeneration : 0,
                    plan ? plan->structuralRevision : 0,
                    opticalHelperGeneration,
                    scoreIdentity.loopGeneration,
                    scoreIdentity.seekGeneration,
                    al.seg.clipId, displaySec,
                    cw > 0 ? cw : fbW, ch > 0 ? ch : fbH
                };
                nlohmann::json inspection {
                    { "clipId", al.seg.clipId }, { "displaySec", displaySec },
                    { "timelineGeneration", key.timelineGeneration },
                    { "snapshotGeneration", key.snapshotGeneration },
                    { "structuralRevision", key.structuralRevision },
                    { "helperGeneration", key.helperGeneration },
                    { "loopGeneration", key.loopGeneration },
                    { "seekGeneration", key.seekGeneration },
                    { "visible", false }, { "reason", plan != nullptr
                        && videoopticalflow::hasMotionBlur (*plan)
                            ? "Motion Blur Image is pending"
                            : plan != nullptr && videoopticalflow::hasMotionView (*plan)
                                ? "Motion View Image is pending"
                                : "Motion vectors are a typed field" },
                    { "computed", false }, { "state", "pending" }
                };
                if (frameOpticalSnapshot == nullptr || plan == nullptr
                    || key.snapshotGeneration == 0 || key.helperGeneration == 0)
                {
                    inspection["state"] = "rejected";
                    inspection["error"] = "Optical Flow preview has no saved source or helper owner";
                }
                else
                {
                    auto& decodeOwner = opticalDecodeOwners[al.seg.clipId];
                    decodeOwner.discardStale (key, playing);
                    if (opticalCompletion
                        && opticalCompletion->key.clipId == al.seg.clipId)
                    {
                        const auto completedKey = opticalCompletion->key;
                        if (! decodeOwner.retain (std::move (*opticalCompletion), key, playing))
                            opticalFlowDecoder.discardUnclaimed (completedKey);
                        opticalCompletion.reset();
                    }
                    if (const auto field = opticalFlowFieldKeys.find (al.seg.clipId);
                        field != opticalFlowFieldKeys.end()
                        && ! field->second.usableAt (key, playing))
                    {
                        opticalFlowFields.erase (al.seg.clipId);
                        opticalFlowFieldKeys.erase (field);
                    }
                    const auto currentField = opticalFlowFieldKeys.find (al.seg.clipId);
                    const bool ownsCurrentField = currentField != opticalFlowFieldKeys.end()
                        && currentField->second == key;
                    if (! decodeOwner.owns (key) && ! ownsCurrentField)
                        opticalFlowDecoder.request (key, frameOpticalSnapshot);
                    const auto retainedStage = decodeOwner.stateFor (key);
                    inspection["decodeStage"] = ownsCurrentField ? "nativeFieldRetained"
                        : retainedStage != nullptr ? retainedStage
                        : opticalFlowDecoder.stateFor (key);
                    const auto* completion = decodeOwner.completion();
                    if (completion != nullptr)
                    {
                        const auto& decodedKey = completion->key;
                        inspection["decodedDisplaySec"] = decodedKey.displaySec;
                        inspection["decodedCurrent"] = decodedKey == key;
                        if (! completion->error.empty() || ! completion->decoded)
                        {
                            inspection[decodedKey == key ? "error" : "lastDecodeError"] =
                                completion->error.empty()
                                ? "Optical Flow media worker returned no decoded pair"
                                : completion->error;
                            if (decodedKey == key) inspection["state"] = "rejected";
                            decodeOwner.release();
                        }
                        else if (decodedKey.usableAt (key, playing))
                        {
                            inspection["uploadStage"] = "uploading";
                            if (decodedKey == key) inspection["decodeStage"] = "uploading";
                            auto existing = opticalFlowFieldKeys.find (al.seg.clipId);
                            if (existing != opticalFlowFieldKeys.end()
                                && existing->second.usableAt (key, playing)
                                && existing->second.displaySec >= decodedKey.displaySec)
                                decodeOwner.release();
                            else
                            {
                                std::string flowError;
                                videoopticalflow::AuthoredPairResult artifact;
                                std::shared_ptr<const videoopticalflow::PublishedNativeResult> field;
                                videoopticalflow::AuthoredPairSelection selection;
                                flowError = videoopticalflow::selectAuthoredPair (
                                    *frameOpticalSnapshot, al.seg.clipId, selection);
                                if (flowError.empty() && ! videoopticalflow::sameAuthoredPairPlan (
                                        *plan, selection))
                                    flowError = "Optical Flow preview plan changed after decode";
                                auto& uploadOwner = opticalUploadOwners[al.seg.clipId];
                                if (flowError.empty()
                                    && (! uploadOwner.uploader
                                        || uploadOwner.helperGeneration != decodedKey.helperGeneration
                                        || uploadOwner.structuralRevision != decodedKey.structuralRevision))
                                {
                                    uploadOwner.uploader = videoopticalflow::makeNativeSourceUploader (
                                        decodedKey.helperGeneration, decodedKey.structuralRevision);
                                    uploadOwner.helperGeneration = decodedKey.helperGeneration;
                                    uploadOwner.structuralRevision = decodedKey.structuralRevision;
                                }
                                if (flowError.empty() && ! uploadOwner.uploader)
                                    flowError = "Optical Flow preview has no context-owned native uploader";
                                if (flowError.empty())
                                {
                                    enterLoopStep ("advance:optical-flow-exec");
                                    videoopticalflow::NativeExecutor executor (
                                        videoopticalflow::EvaluationMode::Preview);
                                    const videoopticalflow::EvaluationContext context {
                                        decodedKey.structuralRevision,
                                        decodedKey.loopGeneration + decodedKey.seekGeneration,
                                        decodedKey.helperGeneration,
                                        videoopticalflow::EvaluationMode::Preview,
                                        false, 0
                                    };
                                    flowError = videoopticalflow::executeAuthoredDecodedPair (
                                        *frameOpticalSnapshot, decodedKey.clipId,
                                        decodedKey.displaySec,
                                        *completion->decoded, context, *uploadOwner.uploader,
                                        arbitgpu::nativeOpticalFlowExecutionBackend(), executor,
                                        field, &artifact, true);
                                }
                                {
                                    std::lock_guard<std::mutex> lock (im.mutex);
                                    if (im.timelineGeneration != decodedKey.timelineGeneration
                                        || im.opticalFlowSnapshot != frameOpticalSnapshot
                                        || im.scoreIdentity.seekGeneration != decodedKey.seekGeneration
                                        || im.scoreIdentity.loopGeneration != decodedKey.loopGeneration
                                        || (im.scoreIdentity.helperGeneration != 0
                                            && im.scoreIdentity.helperGeneration
                                                != decodedKey.helperGeneration))
                                        flowError = "Optical Flow preview owner changed during native execution";
                                }
                                if (flowError.empty())
                                {
                                    opticalFlowFields[al.seg.clipId] = std::move (artifact);
                                    opticalFlowFieldKeys[al.seg.clipId] = decodedKey;
                                    decodeOwner.release();
                                }
                                else if (flowError != videoopticalflow::kPreviewUploadPending)
                                {
                                    const bool currentDecode = decodedKey == key;
                                    opticalFlowFields.erase (al.seg.clipId);
                                    opticalFlowFieldKeys.erase (al.seg.clipId);
                                    decodeOwner.release();
                                    inspection[currentDecode ? "error" : "lastDecodeError"] = flowError;
                                    if (currentDecode) inspection["state"] = "rejected";
                                }
                                else
                                {
                                    inspection["uploadStage"] = "nativeDigestPending";
                                    if (decodedKey == key)
                                        inspection["decodeStage"] = "nativeDigestPending";
                                }
                            }
                        }
                        else decodeOwner.release();
                    }
                    if (const auto found = opticalFlowFields.find (al.seg.clipId);
                        found != opticalFlowFields.end())
                    {
                        const auto& retainedKey = opticalFlowFieldKeys.at (al.seg.clipId);
                        auto receipt = videoopticalflow::authoredPairReceiptJson (found->second);
                        if (retainedKey == key)
                        {
                            inspection["decodeStage"] = "nativeFieldRetained";
                            inspection["state"] = "retained";
                            inspection["computed"] = true;
                            inspection["receipt"] = std::move (receipt);
                        }
                        else
                        {
                            inspection["lastComputed"] = {
                                { "displaySec", retainedKey.displaySec },
                                { "current", false }, { "receipt", std::move (receipt) }
                            };
                            if (retainedKey.usableAt (key, playing))
                                inspection["state"] = "lagged";
                        }
                    }
                }
                if (plan != nullptr && (videoopticalflow::hasMotionView (*plan)
                    || videoopticalflow::hasMotionBlur (*plan)))
                {
                    const auto found = opticalFlowFields.find (al.seg.clipId);
                    const auto usable = found != opticalFlowFields.end()
                        && opticalFlowFieldKeys.at (al.seg.clipId).usableAt (key, playing);
                    if (usable)
                    {
                        const auto& renderedKey = opticalFlowFieldKeys.at (al.seg.clipId);
                        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> image;
                        enterLoopStep ("advance:optical-flow-render");
                        const auto viewError = videoopticalflow::hasMotionBlur (*plan)
                            ? videoopticalflow::renderAuthoredMotionBlur (
                                *plan, found->second, key.snapshotGeneration,
                                renderedKey.displaySec, key.helperGeneration,
                                arbitgpu::nativeOpticalFlowExecutionBackend(), image)
                            : videoopticalflow::renderAuthoredMotionView (
                                *plan, found->second, key.snapshotGeneration,
                                renderedKey.displaySec, key.helperGeneration,
                                arbitgpu::nativeOpticalFlowExecutionBackend(), image);
                        if (viewError.empty())
                        {
                            videorender::LayerDesc layer;
                            fillDescCommon (layer, al.params, displaySec, al.seg.clipId);
                            layer.nativeTextureBackend = image->backend();
                            layer.nativeTextureView = image->colorTextureViewHandle();
                            layer.nativeTextureDescriptor = image->colorTextureDescriptor();
                            layer.nativeTextureOwner = image;
                            layer.texWidth = static_cast<int> (image->width());
                            layer.texHeight = static_cast<int> (image->height());
                            if (image->backend() == "opengl")
                                layer.texture = static_cast<unsigned> (layer.nativeTextureView);
                            descs.push_back (std::move (layer));
                            inspection["imageReady"] = true;
                            inspection["imageDisplaySec"] = renderedKey.displaySec;
                            inspection["imageCurrent"] = renderedKey == key;
                            inspection["reason"] = videoopticalflow::hasMotionBlur (*plan)
                                ? "Motion Blur Image queued for composite"
                                : "Motion View Image queued for composite";
                        }
                        else
                        {
                            inspection["state"] = "rejected";
                            inspection["error"] = viewError;
                            deferredSource = "motion-view-rejected";
                            frameDeferred = true;
                        }
                    }
                    else
                    {
                        // The optical field for this time is not usable yet. It waits on the decode.
                        deferredSource = "optical-field";
                        frameDeferred = true;
                    }
                }
                {
                    std::lock_guard<std::mutex> lock (im.mutex);
                    if (im.timelineGeneration == key.timelineGeneration
                        && im.opticalFlowSnapshot == frameOpticalSnapshot)
                        im.opticalFlowInspection = std::move (inspection);
                }
                continue;
            }

            videorender::LayerDesc d;
            bool planExecuted = false;
            const auto sourceKind = videowire::isTypedParticlePlan(frameVisualPlans->plans, al.seg.clipId)
                ? videowire::SourceKind::Particles : videowire::resolveSourceKind (
                    al.seg.sourceKind, al.seg.isAdjustment, al.seg.sourcePath);
            const bool isAdjustment = sourceKind == videowire::SourceKind::Adjustment;
            const bool isParticles = sourceKind == videowire::SourceKind::Particles;
            const bool isScore = sourceKind == videowire::SourceKind::Score;
            const bool isShader = sourceKind == videowire::SourceKind::Shader
                               && shaderClips.count (al.seg.clipId) > 0;
            ClipStream* cs = nullptr;
            double srcSec = 0.0;
            if (hasNativeVisualSource(frameVisualPlans->plans, al.seg.clipId))
            {
                srcSec = al.seg.inSec
                    + (displaySec - al.seg.displayStartSec) * al.seg.rate;
                if (!prepareDecodedDesc(al.seg, al.params, srcSec, d, cs))
                {
                    frameSettled = false;
                    break;
                }
                planExecuted = true;
            }
            else if (isAdjustment)
            {
                // M8 "effect the world" in the live preview: no source of its
                // own — the shared FrameRenderer runs this clip's effect rack
                // over the composite beneath it (renderComposite's isAdjustment
                // branch), mixed back at the clip's opacity. Matches the export
                // path; the (trackLayer, zOrder) composite sort above placed it
                // after the layers it grades — same ordering as exporter.cpp.
                d.isAdjustment = true;
                fillDescCommon (d, al.params, displaySec, al.seg.clipId);
            }
            else if (isShader)
            {
                // Shader layer: no media decode; the Block A clock is a pure
                // function of display time + tempo — the SAME formula the
                // exporter uses, so preview == export. The Block C score is now
                // live too (cachedCanonicalBlockCFrame, set below), and so is Block B
                // audio (cachedAudioFeatures, drained from the live mix ring).
                const double segDur = (al.seg.outSec - al.seg.inSec)
                                    / std::max (al.seg.rate, 1e-9);
                const double clkFps = valueFps;  // project value-grid fps (== export job.fps)
                const double clockStart = al.seg.clockStartSec >= 0.0
                    ? al.seg.clockStartSec : al.seg.displayStartSec;
                const double clockDuration = al.seg.clockDurationSec > 0.0
                    ? al.seg.clockDurationSec : segDur;
                const int frameIdx = (int) std::llround ((displaySec - clockStart) * clkFps);
                d.shaderSource = true;
                d.shaderClock = videorender::makeShaderClock (
                    displaySec, clockStart, clockDuration, activeBeatTimeline,
                    clkFps, playing, frameIdx);
                fillDescCommon (d, al.params, displaySec, al.seg.clipId);
                if (haveScore)
                    d.canonicalBlockCFrame = cachedCanonicalBlockCFrame;
                if (haveAudio)
                {
                    d.audioFeatures = cachedAudioFeatures;
                    d.audioPresent = true;   // uploads uRMS/uPeak/uOnset/uAudioBands
                }
            }
            else if (isParticles)
            {
                // P4 particle clip: a GPU compute pool, no media decode. Same
                // Block A clock formula as the exporter ⇒ preview == export; the
                // live Block C score (cachedCanonicalBlockCFrame) seeds from the chosen
                // spawn track. v1 has no audio reactivity (Block B master-mix only).
                const double segDur = (al.seg.outSec - al.seg.inSec)
                                    / std::max (al.seg.rate, 1e-9);
                const double clkFps = valueFps;  // project value-grid fps (== export job.fps)
                const double clockStart = al.seg.clockStartSec >= 0.0
                    ? al.seg.clockStartSec : al.seg.displayStartSec;
                const double clockDuration = al.seg.clockDurationSec > 0.0
                    ? al.seg.clockDurationSec : segDur;
                const int frameIdx = (int) std::llround ((displaySec - clockStart) * clkFps);
                d.particleSource = true;
                d.shaderClock = videorender::makeShaderClock (
                    displaySec, clockStart, clockDuration, activeBeatTimeline,
                    clkFps, playing, frameIdx);
                fillDescCommon (d, al.params, displaySec, al.seg.clipId);
                if (haveScore)
                    d.canonicalBlockCFrame = cachedCanonicalBlockCFrame;
                if (videowire::isImportedParticleOverlayPlan(frameVisualPlans->plans, al.seg.clipId))
                {
                    std::string sceneError;
                    const double sceneSeconds = al.seg.inSec
                        + (displaySec - al.seg.displayStartSec) * al.seg.rate;
                    enterLoopStep ("advance:imported-particle-scene-exec");
                    const auto prepared = videohelper::importedscene::prepareVisualImportedSceneLayerAtTime(
                        frameVisualPlans->plans, al.seg.clipId, cw > 0 ? cw : fbW, ch > 0 ? ch : fbH,
                        sceneSeconds, valueFps,
                        videohelper::importedscene::NativeImportedSceneRenderUse::Preview,
                        importedSceneExecution.get(), &importedScenePlanCache, d,
                        importedSceneFrameOwners, sceneError, &al.params.visualParams,
                        { videohelper::viewportgeneration::projectGeneration(timelineGen), visualEvaluationContext.helperGeneration }, nullptr,
                        linearSharedActive, &activeBeatTimeline, displaySec);
                    if (prepared == videohelper::importedscene::VisualImportedScenePreparation::pending)
                    {
                        importedPendingThisFrame = true;
                        frameDeferred = true;
                        deferredSource = pendingSource ("imported-scene");
                        break;
                    }
                    if (prepared != videohelper::importedscene::VisualImportedScenePreparation::rendered)
                    {
                        holdFrameForPlanError(sceneError);
                        break;
                    }
                }
            }
            else if (isScore)
            {
                const double segDur = (al.seg.outSec - al.seg.inSec)
                                    / std::max(al.seg.rate, 1e-9);
                const double clkFps = valueFps;
                const double clockStart = al.seg.clockStartSec >= 0.0
                    ? al.seg.clockStartSec : al.seg.displayStartSec;
                const double clockDuration = al.seg.clockDurationSec > 0.0
                    ? al.seg.clockDurationSec : segDur;
                const int frameIdx = static_cast<int>(std::llround((displaySec - clockStart) * clkFps));
                d.scoreSource = true;
                d.canonicalBlockCFrame = cachedCanonicalBlockCFrame;
                d.shaderClock = videorender::makeShaderClock(
                    displaySec, clockStart, clockDuration, activeBeatTimeline,
                    clkFps, playing, frameIdx);
                fillDescCommon(d, al.params, displaySec, al.seg.clipId);
            }
            else
            {
                srcSec = al.seg.inSec
                    + (displaySec - al.seg.displayStartSec) * al.seg.rate;
                if (! prepareDecodedDesc(al.seg, al.params, srcSec, d, cs))
                {
                    frameSettled = false;
                    if (im.wantClose.load()) break;
                    if (spectrumBakePending || frameDeferred)
                    {
                        frameDeferred = true;
                        break;
                    }
                    continue;
                }
                planExecuted = true;
            }

            attachGeometryAudio(d);
            std::string planError;
            enterLoopStep ("advance:plan-execute");
            if (! planExecuted
                && ! videowire::executeVisualLayerPlanForRenderer (
                        renderer, frameVisualPlans->plans, al.seg.clipId, d, planError,
                        videohelper::geometry::PlanUse::preview, &temporalSamplingTransaction,
                        &inspectionTarget, &inspectionResource,
                        &frameVisualPlans->state, displaySec,
                        &frameVisualScheduleBindings, &visualEventCursor,
                        &visualEvaluationContext, &al.params.visualParams))
            {
                if (!geometryPendingThisFrame())
                    std::fprintf (stderr, "[viewport] visual plan execution rejected: %s\n",
                                  planError.c_str());
                holdFrameForPlanError(planError);
                break;
            }
            if (d.shaderOperationPlan != nullptr && d.shaderTransitionFromClipId > 0
                && d.fromLayer == nullptr)
            {
                ViewportSegment referenced;
                bool found = false;
                {
                    std::lock_guard<std::mutex> lock(im.mutex);
                    const auto* source = videowire::resolveClipSegmentAtDisplayTime(
                        im.segments, d.shaderTransitionFromClipId, displaySec);
                    if (source != nullptr) { referenced = *source; found = true; }
                }
                if (!found) { planError = "shader transition Layer Source clip is unavailable"; }
                else
                {
                    const double sourceSec = referenced.inSec
                        + (displaySec - referenced.displayStartSec) * referenced.rate;
                    ClipGraphParams referencedParams;
                    {
                        std::lock_guard<std::mutex> lock(im.mutex);
                        const auto params = im.clipParams.find(referenced.clipId);
                        if (params != im.clipParams.end()) referencedParams = params->second;
                    }
                    ClipStream* referencedStream = nullptr;
                    videorender::LayerDesc from;
                    const auto referencedKind = hasNativeVisualSource(frameVisualPlans->plans, referenced.clipId)
                        ? videowire::SourceKind::Media : videowire::resolveSourceKind(
                            referenced.sourceKind, referenced.isAdjustment, referenced.sourcePath);
                    if (referencedKind == videowire::SourceKind::Shader)
                    {
                        if (!prepareShaderClip(referenced, planError))
                            planError = "shader transition Layer Source clip " + std::to_string(referenced.clipId)
                                + ": " + planError;
                        else
                        {
                            fillDescCommon(from, referencedParams, displaySec, referenced.clipId);
                            attachGeometryAudio(from);
                            if (haveScore)
                                from.canonicalBlockCFrame = cachedCanonicalBlockCFrame;
                            from.shaderSource = true;
                            const double begin=referenced.clockStartSec>=0?referenced.clockStartSec:referenced.displayStartSec;
                            const double duration=referenced.clockDurationSec>0?referenced.clockDurationSec
                                :(referenced.outSec-referenced.inSec)/std::max(referenced.rate,1e-9);
                            from.shaderClock=videorender::makeShaderClock(displaySec,begin,duration,
                                activeBeatTimeline,valueFps,playing,(int)std::llround((displaySec-begin)*valueFps));
                        }
                    }
                    else if (!prepareDecodedDesc(referenced, referencedParams, sourceSec, from, referencedStream))
                    {
                        // Pending holds the frame through frameDeferred and records no plan error.
                        if (importedPendingThisFrame) break;
                        planError = "shader transition Layer Source could not produce its exact preview frame";
                    }
                    if (planError.empty()) { fromDescs.push_back(std::move(from)); d.fromLayer = &fromDescs.back(); }
                }
                if (!planError.empty())
                {
                    holdFrameForPlanError(planError);
                    break;
                }
            }

            for (const auto& event : frameVisualEvents)
                if (event.clipId == al.seg.clipId)
                {
                    d.feedbackHistoryReset = true;
                    if (event.kind == "particleBurst") d.particleStateReset = true;
                }

                if (al.transitionActive && d.fromLayer == nullptr)
            {
                d.transitionType = al.seg.transitionType;
                d.transitionProgress = (float) al.transitionProgress;
                if (al.haveFrom && al.fromParams.visible >= 0.5f
                    && al.fromParams.opacity > 0.001f)
                {
                    // The outgoing clip ended where this one starts; clamp
                    // its source time into its own span (its last frame
                    // holds for abutting cuts).
                    double fromSrc = al.fromSeg.inSec
                        + (displaySec - al.fromSeg.displayStartSec) * al.fromSeg.rate;
                    fromSrc = std::clamp (fromSrc, al.fromSeg.inSec,
                                          std::max (al.fromSeg.inSec, al.fromSeg.outSec));
                    ClipStream* fs = nullptr;
                    videorender::LayerDesc fd;
                    if (prepareDecodedDesc(al.fromSeg, al.fromParams, fromSrc, fd, fs))
                    {
                        fromDescs.push_back (fd);
                        d.fromLayer = &fromDescs.back();
                    }
                    else
                    {
                        frameSettled = false;
                        // A held frame is not presented, so later layers need no evaluation.
                        if (im.wantClose.load() || frameDeferred) break;
                    }
                }
            }

            if (! haveMain && cs != nullptr)
            {
                haveMain = true;
                mainPts = cs->lastSrcSec;
                mainIdeal = srcSec;
            }
            descs.push_back (d);
        }
        if (opticalCompletion)
            opticalFlowDecoder.discardUnclaimed (opticalCompletion->key);
        for (auto it = opticalFlowFields.begin(); it != opticalFlowFields.end();)
            if (activeOpticalClips.count (it->first) == 0)
            {
                opticalFlowFieldKeys.erase (it->first);
                it = opticalFlowFields.erase (it);
            }
            else ++it;
        for (auto it = opticalDecodeOwners.begin(); it != opticalDecodeOwners.end();)
            if (activeOpticalClips.count (it->first) == 0)
                it = opticalDecodeOwners.erase (it);
            else ++it;
        for (auto it = opticalUploadOwners.begin(); it != opticalUploadOwners.end();)
            if (activeOpticalClips.count (it->first) == 0)
                it = opticalUploadOwners.erase (it);
            else ++it;
        if (activeOpticalClips.empty())
        {
            enterLoopStep ("advance:optical-cancel");
            if (opticalFlowActiveLastFrame) opticalFlowDecoder.cancel();
            std::lock_guard<std::mutex> lock (im.mutex);
            im.opticalFlowInspection = nullptr;
        }
        opticalFlowActiveLastFrame = ! activeOpticalClips.empty();
        if (im.wantClose.load())
            continue; // strict plan rejection never presents a partial frame
        // Stored once per frame, so the status clears on the first frame that is not pending.
        im.importedScenePending.store(importedPendingThisFrame);
        if (frameDeferred)
        {
            ++deferredConsecutive;
            const auto deferredNow = std::chrono::steady_clock::now();
            const bool deferredSourceChanged = deferredSource != lastDeferredSource;
            if (deferredSourceChanged
                || deferredNow - lastDeferredLog >= std::chrono::seconds (1))
            {
                lastDeferredLog = deferredNow;
                lastDeferredSource = deferredSource;
                std::fprintf (stderr, "[viewport] frame deferred: source=%.*s consecutive=%llu\n",
                    static_cast<int> (deferredSource.size()), deferredSource.data(),
                    static_cast<unsigned long long> (deferredConsecutive));
                // A bake wait can last until the project is played or the analysis arrives.
                // Its cause is logged once per episode, not on every retry.
                if (deferredSourceChanged && deferredSource == "spectrum-bake")
                {
                    std::lock_guard<std::mutex> lock (im.mutex);
                    std::fprintf (stderr, "[viewport] spectrum-bake cause: %s\n", im.rendererError.c_str());
                }
            }
            temporalSamplingTransaction.discard();
            // The swap at the top of the tick moved the visual events out of
            // im.visualEvents. Hand them back for the next produce.
            if (! frameVisualEvents.empty())
            {
                std::lock_guard<std::mutex> lock (im.mutex);
                im.visualEvents.insert (im.visualEvents.begin(),
                                        frameVisualEvents.begin(), frameVisualEvents.end());
                frameVisualEvents.clear();
            }
            enterLoopStep ("advance:frame-deferred-sleep");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        deferredConsecutive = 0;
        lastDeferredSource = {};

        if (!planFrameError.empty())
        {
            // This frame presents without a plan hold, so the earlier error has ended.
            std::lock_guard<std::mutex> lock(im.mutex);
            if (im.rendererError == planFrameError) im.rendererError.clear();
            planFrameError.clear();
        }

        // --- Composite + present. On macOS shared viewports the free ring
        // IOSurface is the Metal render target itself: no Metal->GL->IOSurface
        // bridge and no fallback copy are permitted on this path.
        enterLoopPhase (LoopPhase::Composite);
        bool directMetalFrame = false;
        bool framePublished = false;
        uint64_t publishedScoreGeneration = scoreGenSeen;
        if (videowire::consumeTemporalPublicationFailure(
                videowire::TemporalPublicationFailurePoint::viewportPresentation))
        {
            temporalSamplingTransaction.discard();
            continue;
        }
        const bool traceSceneFrame = sceneTraceFrame && sceneTraceCount < sceneTraceLimit
            && std::isfinite(displaySec) && std::isfinite(valueFps)
            && std::llround(displaySec * valueFps) == *sceneTraceFrame;
        // A dropped attempt can reuse this hint; FrameReady logs the sent seq.
        const uint64_t traceSeqHint = traceSceneFrame ? im.frameSeq + 1 : 0;
        if (traceSceneFrame)
        {
            ++sceneTraceCount;
            size_t nativeLayers = 0;
            for (const auto& layer : descs)
            {
                nativeLayers += !layer.nativeTextureBackend.empty() || layer.nativeTextureView != 0;
                if (layer.fromLayer)
                    nativeLayers += !layer.fromLayer->nativeTextureBackend.empty()
                        || layer.fromLayer->nativeTextureView != 0;
            }
            std::fprintf(stderr,
                "[viewport-scene-trace] attempt=%d seqHint=%llu frame=%lld seconds=%.6f timeline=%llu structural=%llu runtime=%llu layers=%zu nativeScenes=%zu importedOwners=%zu animatedOwners=%zu canvas=%dx%d viewport=%dx%d readback=%s\n",
                sceneTraceCount, static_cast<unsigned long long>(traceSeqHint),
                static_cast<long long>(*sceneTraceFrame), displaySec,
                static_cast<unsigned long long>(timelineGen),
                static_cast<unsigned long long>(structuralRevision_.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(runtimeRevision_.load(std::memory_order_relaxed)),
                descs.size(), nativeLayers,
                importedSceneFrameOwners.size(), importedAnimationFrameOwners.size(),
                cw, ch, fbW, fbH,
                sceneTraceReadback == SceneTraceReadback::Native ? "native"
                    : sceneTraceReadback == SceneTraceReadback::Composite ? "composite"
                    : sceneTraceReadback == SceneTraceReadback::None ? "none" : "both");
            const auto traceLayer = [&](const videorender::LayerDesc& layer)
            {
                if (layer.nativeTextureBackend.empty() && layer.nativeTextureView == 0) return;
                const auto& d = layer.nativeTextureDescriptor;
                std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame = layer.rawExportFrame;
                if (!frame) frame = layer.nativeLinearImage;
                if (!frame)
                    for (const auto& owner : importedSceneFrameOwners)
                    {
                        const auto& candidate =
                            videohelper::modelpayload::ImportedScenePayloadExecution::nativeFrame(owner);
                        if (candidate && candidate->backend() == layer.nativeTextureBackend
                            && candidate->colorTextureViewHandle() == layer.nativeTextureView)
                        { frame = candidate; break; }
                    }
                if (!frame)
                    for (const auto& owner : importedAnimationFrameOwners)
                    {
                        const auto& candidate =
                            videohelper::modelpayload::ImportedAnimatedScenePayloadExecution::nativeFrame(owner);
                        if (candidate && candidate->backend() == layer.nativeTextureBackend
                            && candidate->colorTextureViewHandle() == layer.nativeTextureView)
                        { frame = candidate; break; }
                    }
                const auto actual = frame ? frame->colorTextureDescriptor()
                    : arbitgpu::NativeTextureViewDescriptor{};
                const bool matchesReceipt = frame && actual.complete()
                    && actual.backend == d.backend && actual.imageHandle == d.imageHandle
                    && actual.textureViewHandle == d.textureViewHandle
                    && actual.deviceOrContextIdentity == d.deviceOrContextIdentity
                    && actual.rendererGeneration == d.rendererGeneration
                    && actual.width == d.width && actual.height == d.height
                    && actual.rowOrder == d.rowOrder;
                std::fprintf(stderr,
                    "[viewport-scene-trace] layer clip=%d seqHint=%llu revision=%llu backend=%s view=%llu legacyTexture=%u descriptorComplete=%d descriptorBackend=%s image=%llu descriptorView=%llu context=%llu generation=%llu extent=%ux%u layerExtent=%dx%d kind=%u format=%u samples=%u sampled=%d row=%s owner=%llx frameOwner=%llx receiptMatch=%d\n",
                    layer.clipId, static_cast<unsigned long long>(traceSeqHint),
                    static_cast<unsigned long long>(layer.visualPlanStructuralRevision),
                    layer.nativeTextureBackend.c_str(),
                    static_cast<unsigned long long>(layer.nativeTextureView), layer.texture,
                    d.complete(), d.backend.c_str(), static_cast<unsigned long long>(d.imageHandle),
                    static_cast<unsigned long long>(d.textureViewHandle),
                    static_cast<unsigned long long>(d.deviceOrContextIdentity),
                    static_cast<unsigned long long>(d.rendererGeneration), d.width, d.height,
                    layer.texWidth, layer.texHeight, static_cast<unsigned>(d.viewKind),
                    static_cast<unsigned>(d.format), d.sampleCount, d.sampledBinding,
                    d.rowOrder == arbitgpu::NativeTextureRowOrder::TopFirst ? "top" : "bottom",
                    static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(
                        layer.nativeTextureOwner.get())),
                    static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(frame.get())),
                    matchesReceipt);
                if (sceneTraceReadback == SceneTraceReadback::Composite
                    || sceneTraceReadback == SceneTraceReadback::None) return;
                if (!frame)
                {
                    std::fprintf(stderr, "[viewport-scene-trace] nativeColor clip=%d seqHint=%llu read=unavailable\n",
                                 layer.clipId, static_cast<unsigned long long>(traceSeqHint));
                    return;
                }
                arbitgpu::NativeRawPassPixels pixels;
                std::string error;
                if (!frame->readRawPass(renderpassoutput::Output::Color, pixels, error)
                    || (pixels.format != arbitgpu::NativeTexturePixelFormat::Rgba8Unorm
                        && pixels.format != arbitgpu::NativeTexturePixelFormat::Bgra8Unorm)
                    || pixels.width == 0 || pixels.height == 0
                    || pixels.bytes.size() != static_cast<size_t>(pixels.width) * pixels.height * 4)
                {
                    std::fprintf(stderr,
                        "[viewport-scene-trace] nativeColor clip=%d seqHint=%llu read=unavailable format=%u extent=%ux%u error=%s\n",
                        layer.clipId, static_cast<unsigned long long>(traceSeqHint),
                        static_cast<unsigned>(pixels.format), pixels.width,
                        pixels.height, error.c_str());
                    return;
                }
                printSceneTraceColor("nativeColor", layer.clipId, traceSeqHint,
                    static_cast<uint64_t>(d.imageHandle), pixels.bytes.data(),
                    static_cast<int>(pixels.width), static_cast<int>(pixels.height),
                    static_cast<size_t>(pixels.width) * 4,
                    pixels.format == arbitgpu::NativeTexturePixelFormat::Bgra8Unorm);
            };
            for (const auto& layer : descs)
            {
                traceLayer(layer);
                if (layer.fromLayer) traceLayer(*layer.fromLayer);
            }
        }
        const bool requireDirectMetal =
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
            metalOnly;
#else
            false;
#endif
#if ARBIT_HAVE_GPU_SHARED
        gpuexp::ExportedBuffer* directTarget = nullptr;
#endif
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
        void* directSurface = nullptr;
        int directWidth = fbW;
        int directHeight = fbH;
        if (metalOnly && ! shared && localMetalSurface != nullptr)
        {
            std::string surfaceError;
            if (! localMetalSurface->resize (fbW, fbH, surfaceError))
            {
                std::fprintf (stderr, "[viewport] strict Metal resize failed: %s\n",
                              surfaceError.c_str());
                {
                    std::lock_guard<std::mutex> lock (im.mutex);
                    im.compositorBackend = "metal-zero-copy-rejected";
                }
                closeForDeviceFailure("strict Metal surface could not be resized", surfaceError);
                continue;
            }
            directSurface = localMetalSurface->ioSurface();
            directWidth = localMetalSurface->width();
            directHeight = localMetalSurface->height();
        }
        if (shared)
        {
            int freeBuffers = 0, busyBuffers = 0;
            for (auto& buffer : im.buffers)
                if (buffer.busy) ++busyBuffers;
                else
                {
                    ++freeBuffers;
                    if (directTarget == nullptr) directTarget = &buffer;
                }
            im.sharedFreeBuffers = freeBuffers;
            im.sharedBusyBuffers = busyBuffers;
            if (directTarget != nullptr)
            {
                directSurface = directTarget->surface;
                directWidth = directTarget->width;
                directHeight = directTarget->height;
            }
        }
        if (directSurface != nullptr)
        {
            videopreview::State metalPreviewState;
            {
                std::lock_guard<std::mutex> lock (im.mutex);
                metalPreviewState = im.inspectionPresentation;
            }
            unsigned requestedInspectionHandle = inspectionResource.handle;
            if (const auto* plan = videowire::findVisualLayerPlan(
                    frameVisualPlans->plans, inspectionTarget.clipId);
                plan != nullptr
                && videowire::classifyVisualInspectionTarget(*plan, inspectionTarget)
                    == videowire::VisualInspectionSlice::terminalComposite)
                requestedInspectionHandle = 0xffffffffu;
            renderer.setMetalInspectionPresentation (requestedInspectionHandle,
                                                      metalPreviewState);
            enterLoopPhase (LoopPhase::Render);
            directMetalFrame = viewportTelemetry.renderComposite ([&]
            {
                return renderer.renderCompositeToIOSurface (
                    directSurface, directWidth, directHeight,
                    descs.data(), (int) descs.size(),
                    overlays.data(), (int) overlays.size(), traceSeqHint, linearSharedActive);
            }, [] (bool valid) { return valid; });
        }
        if (requireDirectMetal && directSurface != nullptr && ! directMetalFrame)
        {
            if (traceSceneFrame)
                std::fprintf(stderr,
                    "[viewport-scene-trace] composite read=unavailable backend=metal error=%s\n",
                    renderer.lastError().c_str());
            std::fprintf (stderr, "[viewport] strict Metal compositor rejected frame\n");
            {
                std::lock_guard<std::mutex> lock (im.mutex);
                im.compositorBackend = renderer.compositorBackend();
            }
            closeForDeviceFailure("strict Metal compositor rejected frame", renderer.lastError());
            continue;
        }
        if (traceSceneFrame && metalOnly
            && (sceneTraceReadback == SceneTraceReadback::Both
                || sceneTraceReadback == SceneTraceReadback::Composite))
        {
            IOSurfaceRef surface = static_cast<IOSurfaceRef>(directSurface);
            if (directMetalFrame && surface != nullptr
                && IOSurfaceLock(surface, kIOSurfaceLockReadOnly, nullptr) == 0)
            {
                const auto* bgra = static_cast<const uint8_t*>(IOSurfaceGetBaseAddress(surface));
                const int surfaceW = static_cast<int>(IOSurfaceGetWidth(surface));
                const int surfaceH = static_cast<int>(IOSurfaceGetHeight(surface));
                const size_t rowBytes = IOSurfaceGetBytesPerRow(surface);
                if (bgra && surfaceW > 0 && surfaceH > 0
                    && IOSurfaceGetBytesPerElement(surface) == 4
                    && rowBytes >= static_cast<size_t>(surfaceW) * 4)
                    printSceneTraceColor("composite", -1, traceSeqHint,
                                         IOSurfaceGetID(surface), bgra, surfaceW, surfaceH,
                                         rowBytes, true);
                else
                    std::fprintf(stderr,
                        "[viewport-scene-trace] composite read=unavailable backend=metal extent=%dx%d rowBytes=%zu\n",
                        surfaceW, surfaceH, rowBytes);
                IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, nullptr);
            }
            else
                std::fprintf(stderr,
                    "[viewport-scene-trace] composite read=unavailable backend=metal directFrame=%d surface=%d lockFailed=%d\n",
                    directMetalFrame, surface != nullptr,
                    directMetalFrame && surface != nullptr);
        }
#endif
        const bool hashThisFrame = (frameCounter++ & 3) == 0;
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
        // The strict Metal compositor has already completed its GPU work before
        // returning. Read auxiliary diagnostics directly from the IOSurface
        // that is about to be handed to the consumer: presentation remains
        // zero-copy and no OpenGL/CPU compositor is introduced.
        if (directMetalFrame && directSurface != nullptr
            && (hashThisFrame || scopeMask_.load() != 0))
        {
            const uint32_t scopes = scopeMask_.load();
            const int64_t scopeNow = nowNs();
            const bool readScopes = scopes != 0 && scopeNow >= nextScopeNs;
            IOSurfaceRef surface = static_cast<IOSurfaceRef> (directSurface);
            if (surface != nullptr
                && IOSurfaceLock (surface, kIOSurfaceLockReadOnly, nullptr) == 0)
            {
                const auto* bgra = static_cast<const uint8_t*> (IOSurfaceGetBaseAddress (surface));
                const int surfaceW = static_cast<int> (IOSurfaceGetWidth (surface));
                const int surfaceH = static_cast<int> (IOSurfaceGetHeight (surface));
                const size_t rowBytes = IOSurfaceGetBytesPerRow (surface);
                if (bgra != nullptr && surfaceW > 0 && surfaceH > 0)
                {
                    if (hashThisFrame)
                    {
                        uint64_t hash = 1469598103934665603ull;
                        const size_t maxPixels = std::min<size_t> (
                            static_cast<size_t> (surfaceW) * surfaceH, (1024u * 1024u) / 4u);
                        for (size_t pixel = 0; pixel < maxPixels; ++pixel)
                        {
                            const size_t y = pixel / static_cast<size_t> (surfaceW);
                            const size_t x = pixel % static_cast<size_t> (surfaceW);
                            const uint8_t* source = bgra + y * rowBytes + x * 4;
                            const uint8_t rgbaPixel[4] = {
                                source[2], source[1], source[0], source[3] };
                            for (const uint8_t value : rgbaPixel)
                            {
                                hash ^= value;
                                hash *= 1099511628211ull;
                            }
                        }
                        im.lastFrameHash = hash;
                    }
                    if (readScopes)
                    {
                        scopeRgba.resize (static_cast<size_t> (
                            videorender::FrameRenderer::kScopeW)
                            * videorender::FrameRenderer::kScopeH * 4);
                        for (int y = 0; y < videorender::FrameRenderer::kScopeH; ++y)
                        {
                            const int sy = std::min (surfaceH - 1,
                                y * surfaceH / videorender::FrameRenderer::kScopeH);
                            for (int x = 0; x < videorender::FrameRenderer::kScopeW; ++x)
                            {
                                const int sx = std::min (surfaceW - 1,
                                    x * surfaceW / videorender::FrameRenderer::kScopeW);
                                const uint8_t* source = bgra + static_cast<size_t> (sy) * rowBytes
                                                      + static_cast<size_t> (sx) * 4;
                                uint8_t* dest = scopeRgba.data()
                                    + (static_cast<size_t> (y) * videorender::FrameRenderer::kScopeW
                                       + x) * 4;
                                dest[0] = source[2]; dest[1] = source[1];
                                dest[2] = source[0]; dest[3] = source[3];
                            }
                        }
                        im.computeScopes (scopes, scopeRgba.data(),
                                          videorender::FrameRenderer::kScopeW,
                                          videorender::FrameRenderer::kScopeH);
                        nextScopeNs = scopeNow + 100'000'000ll;
                    }
                }
                IOSurfaceUnlock (surface, kIOSurfaceLockReadOnly, nullptr);
            }
        }
#endif
        enterLoopPhase (LoopPhase::Render);
        const unsigned compositeTex =
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
            metalOnly ? 0u :
#endif
            viewportTelemetry.renderComposite ([&]
            {
                return renderer.renderComposite (descs.data(), (int) descs.size(),
                                                 overlays.data(), (int) overlays.size());
            }, [] (unsigned texture) { return texture != 0; });
        if (! metalOnly && compositeTex == 0)
        {
            if (traceSceneFrame)
                std::fprintf(stderr,
                    "[viewport-scene-trace] composite read=unavailable backend=opengl error=%s\n",
                    renderer.lastError().c_str());
            std::lock_guard<std::mutex> lock (im.mutex);
            im.rendererError = renderer.lastError().empty()
                ? "OpenGL compositor rejected frame" : renderer.lastError();
            continue;
        }
        if (traceSceneFrame && !metalOnly
            && (sceneTraceReadback == SceneTraceReadback::Both
                || sceneTraceReadback == SceneTraceReadback::Composite))
        {
            const int compositeW = cw > 0 ? cw : fbW;
            const int compositeH = ch > 0 ? ch : fbH;
            std::vector<uint8_t> rgba(static_cast<size_t>(compositeW) * compositeH * 4);
            GLint previousTexture = 0;
            GLint previousPackBuffer = 0;
            GLint packAlignment = 0, packRowLength = 0, packSkipRows = 0;
            GLint packSkipPixels = 0, packSwapBytes = 0;
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &previousTexture);
            glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &previousPackBuffer);
            glGetIntegerv(GL_PACK_ALIGNMENT, &packAlignment);
            glGetIntegerv(GL_PACK_ROW_LENGTH, &packRowLength);
            glGetIntegerv(GL_PACK_SKIP_ROWS, &packSkipRows);
            glGetIntegerv(GL_PACK_SKIP_PIXELS, &packSkipPixels);
            glGetIntegerv(GL_PACK_SWAP_BYTES, &packSwapBytes);
            gl.BindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glPixelStorei(GL_PACK_ROW_LENGTH, 0);
            glPixelStorei(GL_PACK_SKIP_ROWS, 0);
            glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
            glPixelStorei(GL_PACK_SWAP_BYTES, GL_FALSE);
            glBindTexture(GL_TEXTURE_2D, compositeTex);
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
            const GLenum readError = glGetError();
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(previousTexture));
            gl.BindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<GLuint>(previousPackBuffer));
            glPixelStorei(GL_PACK_ALIGNMENT, packAlignment);
            glPixelStorei(GL_PACK_ROW_LENGTH, packRowLength);
            glPixelStorei(GL_PACK_SKIP_ROWS, packSkipRows);
            glPixelStorei(GL_PACK_SKIP_PIXELS, packSkipPixels);
            glPixelStorei(GL_PACK_SWAP_BYTES, packSwapBytes);
            if (readError == GL_NO_ERROR)
                printSceneTraceColor("composite", -1, traceSeqHint, compositeTex,
                                     rgba.data(), compositeW,
                                     compositeH, static_cast<size_t>(compositeW) * 4, false);
            else
                std::fprintf(stderr,
                    "[viewport-scene-trace] composite read=unavailable backend=opengl glError=0x%04x\n",
                    readError);
        }
        {
            std::lock_guard<std::mutex> lock (im.mutex);
            im.particleBackend = renderer.particleBackend();
            im.compositorBackend = renderer.compositorBackend();
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
            im.inspectionMetalAdmitted = renderer.metalInspectionResourceAdmitted();
#else
            im.inspectionMetalAdmitted = false;
#endif
            const auto* inspectionPlan = videowire::findVisualLayerPlan(
                frameVisualPlans->plans, im.inspectionClipId);
            bool terminalResource = false;
            if (inspectionPlan != nullptr
                && inspectionPlan->structuralRevision == im.inspectionStructuralRevision)
            {
                const auto output = std::find_if(inspectionPlan->operations.begin(),
                    inspectionPlan->operations.end(), [](const auto& operation)
                    { return operation.kind == "video.out"; });
                if (output != inspectionPlan->operations.end())
                    terminalResource = std::any_of(inspectionPlan->edges.begin(),
                        inspectionPlan->edges.end(), [&](const auto& edge)
                        {
                            return edge.fromNodeId == im.inspectionNodeId
                                && edge.toNodeId == output->nodeId;
                        });
            }
            const bool opticalInspection = videowire::isOpticalFlowPlan (
                frameVisualPlans->plans, im.inspectionClipId);
            const bool terminalReady = terminalResource
                && ! opticalInspection
                && (compositeTex != 0 || im.compositorBackend.find("metal") != std::string::npos);
            const auto inspectionSlice = inspectionPlan != nullptr
                    && inspectionPlan->structuralRevision == inspectionTarget.structuralRevision
                ? videowire::classifyVisualInspectionTarget(*inspectionPlan, inspectionTarget)
                : videowire::VisualInspectionSlice::unsupported;
            const bool transformedReady = (inspectionSlice
                    == videowire::VisualInspectionSlice::transformedLayer
                || inspectionSlice == videowire::VisualInspectionSlice::retainedDrawShape)
                && renderer.transformInspectionTexture() != 0;
            if (transformedReady)
            {
                inspectionResource.handle = renderer.transformInspectionTexture();
                inspectionResource.width = cw > 0 ? cw : fbW;
                inspectionResource.height = ch > 0 ? ch : fbH;
                inspectionResource.nodeId = im.inspectionNodeId;
                inspectionResource.outputPort = im.inspectionOutputPort;
            }
            const bool nonTerminalReady = ! opticalInspection
                && inspectionResource.handle != 0
                && inspectionResource.nodeId == im.inspectionNodeId
                && inspectionResource.outputPort == im.inspectionOutputPort;
            im.inspectionResourceReady = terminalReady || nonTerminalReady;
            im.inspectionResourceHandle = opticalInspection ? 0
                : (nonTerminalReady ? inspectionResource.handle : compositeTex);
            if (opticalInspection)
            {
                im.inspectionResourceWidth = 0;
                im.inspectionResourceHeight = 0;
                im.inspectionResourceBackend.clear();
            }
            if (im.inspectionResourceReady)
            {
                ++im.inspectionResourceGeneration;
                im.inspectionResourceWidth = nonTerminalReady
                    ? inspectionResource.width : (cw > 0 ? cw : fbW);
                im.inspectionResourceHeight = nonTerminalReady
                    ? inspectionResource.height : (ch > 0 ? ch : fbH);
                im.inspectionResourceBackend = transformedReady
                    ? im.compositorBackend
                        + (inspectionSlice == videowire::VisualInspectionSlice::retainedDrawShape
                            ? "-draw-shape-target" : "-transform-target")
                    : (nonTerminalReady ? im.compositorBackend + "-decoded-source"
                                        : im.compositorBackend);
            }
        }
        // Report which retime tier actually drew this frame (1=frame-blend).
        im.interpBackend.store (interpTierThisFrame, std::memory_order_relaxed);
        // Export-frame border + overscan dim (viewport-only; no-op when no
        // canvas is set). presentTex is what the user sees; scopes keep
        // measuring the undecorated composite.
        videopreview::State previewState;
        unsigned previewTexture = 0;
        {
            std::lock_guard<std::mutex> lock(im.mutex);
            previewState = im.inspectionPresentation;
            if (im.inspectionResourceReady) previewTexture = im.inspectionResourceHandle;
        }
        const unsigned presentTex = requireDirectMetal ? 0u
            : (previewState.layout != videopreview::Layout::hidden && previewTexture != 0
                ? renderer.applyNodePreview(compositeTex, previewTexture, previewState)
                : (canvasFrameEnabled_.load() ? renderer.applyCanvasFrame(compositeTex)
                                              : compositeTex));
        enterLoopPhase (LoopPhase::Present);
        if (! offscreen)
        {
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
            if (metalOnly)
            {
                std::string presentError;
                if (! viewportTelemetry.present ([&]
                    { return localMetalSurface != nullptr
                          && localMetalSurface->present (presentError); }))
                {
                    std::fprintf (stderr, "[viewport] strict Metal present failed: %s\n",
                                  presentError.c_str());
                    {
                        std::lock_guard<std::mutex> lock (im.mutex);
                        im.compositorBackend = "metal-present-rejected";
                    }
                    closeForDeviceFailure("strict Metal present failed", presentError);
                    continue;
                }
            }
            else
#endif
            {
                while (glGetError() != GL_NO_ERROR) {}
                glfwGetError (nullptr);
                const bool presented = viewportTelemetry.present ([&]
                {
                    if (! renderer.presentToWindow (presentTex, fbW, fbH)) return false;
                    glfwSwapBuffers (win);
                    const char* glfwDescription = nullptr;
                    return glfwGetError (&glfwDescription) == GLFW_NO_ERROR;
                });
                if (! presented)
                {
                    std::lock_guard<std::mutex> lock (im.mutex);
                    im.rendererError = "OpenGL window presentation failed";
                    continue;
                }
            }
            framePublished = true;
        }

        // frameHash = async PBO hash of the undecorated composite (canvas-res,
        // == the export's composite for A/B parity; the hash readback region is
        // sized to outW_, which is now the canvas, so this must be the composite,
        // not the window-res presentTex — audit #16).
        if (! requireDirectMetal && hashThisFrame)
        {
            uint64_t h = 0;
            if (renderer.hashComposite (compositeTex, h))
                im.lastFrameHash = h;
        }

        // Video scopes (PROTOCOL.md §Scopes): downscaled async readback of
        // the same composited frame, ~10 Hz, ONLY while a scope is enabled.
        if (const uint32_t scopes = scopeMask_.load(); ! requireDirectMetal && scopes != 0)
        {
            const int64_t tScope = nowNs();
            if (tScope >= nextScopeNs)
            {
                nextScopeNs = tScope + 100'000'000ll;
                if (renderer.readbackScope (compositeTex, scopeRgba))
                    im.computeScopes (scopes, scopeRgba.data(),
                                      videorender::FrameRenderer::kScopeW,
                                      videorender::FrameRenderer::kScopeH);
            }
        }

        if (! metalOnly)
            for (GLenum err = glGetError(); err != GL_NO_ERROR; err = glGetError())
                if (glErrorsLogged < 16)
                {
                    std::fprintf (stderr, "[viewport] GL error 0x%04x\n", err);
                    ++glErrorsLogged;
                }

        // A/V offset: presented frame pts vs the ideal source time.
        if (haveMain)
        {
            std::lock_guard<std::mutex> lock (im.mutex);
            im.sourcePtsSec = mainPts;
            im.sourceIdealSec = mainIdeal;
            if (playing)
                im.avOffsetSec = mainPts - mainIdeal;
        }

        if (offscreen && shm)
        {
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
            if (metalOnly)
            {
                IOSurfaceRef surface = static_cast<IOSurfaceRef> (directSurface);
                if (directMetalFrame && surface != nullptr
                    && IOSurfaceLock (surface, kIOSurfaceLockReadOnly, nullptr) == 0)
                {
                    auto* hdr = im.shmRing.header();
                    const int pw = static_cast<int> (IOSurfaceGetWidth (surface));
                    const int ph = static_cast<int> (IOSurfaceGetHeight (surface));
                    const size_t sourceStride = IOSurfaceGetBytesPerRow (surface);
                    const auto* pixels = static_cast<const uint8_t*> (
                        IOSurfaceGetBaseAddress (surface));
                    const size_t rowBytes = static_cast<size_t> (pw) * 4;
                    uint64_t checkedBytes = 0;
                    const bool payloadSizeValid = viewportTelemetry.checkedPayloadBytes (
                        rowBytes, ph, checkedBytes)
                        && checkedBytes <= std::numeric_limits<size_t>::max();
                    if (hdr != nullptr && hdr->slotCount > 0 && pixels != nullptr
                        && sourceStride >= rowBytes && payloadSizeValid
                        && checkedBytes <= hdr->slotBytes)
                    {
                        const uint32_t idx = im.shmSlotCursor++ % hdr->slotCount;
                        auto* slot = im.shmRing.slot (idx);
                        slot->generation.fetch_add (1, std::memory_order_acq_rel);
                        slot->width = static_cast<uint32_t> (pw);
                        slot->height = static_cast<uint32_t> (ph);
                        slot->strideBytes = static_cast<uint32_t> (rowBytes);
                        slot->ptsSec = displaySec;
                        slot->mediaId = 0;
                        const uint64_t seq = ++im.shmFrameSeq;
                        slot->reserved[0] = static_cast<uint32_t> (seq);
                        slot->reserved[1] = static_cast<uint32_t> (seq >> 32);
                        slot->reserved[2] = 1;
                        auto* destination = im.shmRing.slotPayload (idx);
                        for (int row = 0; row < ph; ++row)
                            std::memcpy (destination + static_cast<size_t> (row) * rowBytes,
                                         pixels + static_cast<size_t> (row) * sourceStride,
                                         rowBytes);
                        slot->generation.fetch_add (1, std::memory_order_acq_rel);
                        im.sharedFramesSent.fetch_add (1);
                        framePublished = true;
                        viewportTelemetry.readbackCopied (true, rowBytes, ph,
                            fbW, fbH, pw, ph);
                    }
                    else
                        viewportTelemetry.readbackCopied (false, rowBytes, ph,
                            fbW, fbH, pw, ph);
                    IOSurfaceUnlock (surface, kIOSurfaceLockReadOnly, nullptr);
                }
            }
            else
#endif
            {
                // Pipelined PBO readback (one frame of latency): map the
                // previous frame's pixels and seqlock-write them into the next
                // ring slot. Round-robin slots; the reader scans for the highest
                // per-slot frameSeq (SlotHeader.reserved[0/1]) with an even
                // generation, so a torn slot is simply skipped.
                int pw = 0, ph = 0;
                const uint64_t readbackScoreGeneration = pendingReadbackScoreGeneration;
                bool readbackQueued = false;
                const uint8_t* px = renderer.beginMappedReadback (
                    presentTex, pw, ph, &readbackQueued);
                if (readbackQueued) pendingReadbackScoreGeneration = scoreGenSeen;
                if (px != nullptr)
                {
                    auto* hdr = im.shmRing.header();
                    const size_t stride = pw > 0 ? static_cast<size_t> (pw) * 4u : 0;
                    uint64_t checkedBytes = 0;
                    const bool payloadSizeValid = viewportTelemetry.checkedPayloadBytes (
                        stride, ph, checkedBytes)
                        && checkedBytes <= std::numeric_limits<size_t>::max();
                    if (hdr != nullptr && hdr->slotCount > 0 && payloadSizeValid
                        && checkedBytes <= hdr->slotBytes)
                    {
                        const uint32_t idx = im.shmSlotCursor++ % hdr->slotCount;
                        auto* slot = im.shmRing.slot (idx);
                        slot->generation.fetch_add (1, std::memory_order_acq_rel); // odd
                        slot->width = (uint32_t) pw;
                        slot->height = (uint32_t) ph;
                        slot->strideBytes = (uint32_t) pw * 4;
                        slot->ptsSec = displaySec;
                        slot->mediaId = 0;
                        const uint64_t seq = ++im.shmFrameSeq;
                        slot->reserved[0] = (uint32_t) (seq & 0xffffffffu);
                        slot->reserved[1] = (uint32_t) (seq >> 32);
                        slot->reserved[2] = 1; // payload format: BGRA8, top row first
                        std::memcpy (im.shmRing.slotPayload (idx), px,
                                     static_cast<size_t> (checkedBytes));
                        slot->generation.fetch_add (1, std::memory_order_acq_rel); // even
                        im.sharedFramesSent.fetch_add (1);
                        framePublished = true;
                        publishedScoreGeneration = readbackScoreGeneration;
                        viewportTelemetry.readbackCopied (true,
                            static_cast<size_t> (slot->strideBytes), ph,
                            fbW, fbH, pw, ph);
                    }
                    else
                        viewportTelemetry.readbackCopied (false,
                            static_cast<size_t> (pw > 0 ? pw : 0) * 4u, ph,
                            fbW, fbH, pw, ph);
                    renderer.endMappedReadback();
                }
            }
        }
#if ARBIT_HAVE_GPU_SHARED
        else
        {
            enterLoopPhase (LoopPhase::Publish);
            // FREE -> RENDERING -> QUEUED handoff: copy the composite into
            // the oldest free exported buffer, order the GPU work (v2: attach
            // an acquire fence fd to FRAME_READY so the consumer's GPU waits
            // without stalling our pipeline; fallback: glFinish as in v1),
            // announce. A buffer that has been announced and not yet released
            // is never written — when none are free the frame is dropped.
            gpuexp::ExportedBuffer* target =
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
                directMetalFrame ? directTarget : nullptr;
#else
                nullptr;
            int freeBuffers = 0;
            int busyBuffers = 0;
            for (auto& b : im.buffers)
            {
                if (b.busy)
                    ++busyBuffers;
                else
                {
                    ++freeBuffers;
                    if (target == nullptr)
                        target = &b;
                }
            }
            im.sharedFreeBuffers = freeBuffers;
            im.sharedBusyBuffers = busyBuffers;
#endif
            if (target != nullptr)
            {
                bool blitSucceeded = true;
#if ! (defined(__APPLE__) && ARBIT_HAVE_IOSURFACE)
                blitSucceeded = im.exporter.blit (presentTex, *target);
#endif

                const bool fenceRequired = im.exporter.fenceSyncAvailable();
                int fenceFd = blitSucceeded ? im.exporter.createNativeFenceFd() : -1;
                const bool fenceSucceeded = ! fenceRequired || fenceFd >= 0;
                if (fenceFd < 0 && ! directMetalFrame && ! fenceRequired && blitSucceeded)
                    glFinish();

                gpusurf::FrameReadyPayload ready;
                ready.bufferIndex = (uint32_t) (target - im.buffers.data());
                ready.flags = fenceFd >= 0 ? gpusurf::kFlagFenceSync : 0;
                ready.frameSeq = ++im.frameSeq;
                ready.displaySec = displaySec;
                ready.frameHash = im.lastFrameHash.load();
                bool sent = false;
                if (blitSucceeded && fenceSucceeded)
                {
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
                    if (linearSharedActive)
                    {
                        const gpusurf::LinearIOSurfaceFrameReadyPayload linearReady {sharedBufferGeneration, ready};
                        sent = im.fdSocket.sendMsg(static_cast<uint32_t>(gpusurf::MsgType::LinearIOSurfaceFrameReady),
                            &linearReady, sizeof(linearReady));
                    }
                    else
#endif
                        sent = im.fdSocket.sendMsg((uint32_t) gpusurf::MsgType::FrameReady, &ready, sizeof(ready),
                            fenceFd >= 0 ? &fenceFd : nullptr, fenceFd >= 0 ? 1 : 0);
                }
                if (fenceFd >= 0)
                    closeFdIfValid (fenceFd); // sendmsg dup'ed it (or send failed)
                const bool handedOff = viewportTelemetry.zeroCopyHandoff (
                    blitSucceeded, fenceSucceeded, sent);
                if (handedOff)
                {
                    target->busy = true;
                    im.sharedFramesSent.fetch_add (1);
                    framePublished = true;
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
                    // Count published frames separately from traced render attempts.
                    // A dropped attempt must not consume the next FrameReady sample.
                    const bool tracePublishedSceneFrame = sceneTraceFrame.has_value()
                        && std::isfinite (ready.displaySec) && std::isfinite (valueFps)
                        && std::llround (ready.displaySec * valueFps) == *sceneTraceFrame;
                    if (directMetalFrame && tracePublishedSceneFrame
                        && (sceneTraceReadback == SceneTraceReadback::Both
                            || sceneTraceReadback == SceneTraceReadback::Composite)
                        && sceneTracePublishedCount < sceneTraceLimit)
                    {
                        ++sceneTracePublishedCount;
                        IOSurfaceRef surface = static_cast<IOSurfaceRef> (target->surface);
                        const uint32_t id = surface != nullptr ? IOSurfaceGetID (surface) : 0;
                        if (surface != nullptr
                            && IOSurfaceLock (surface, kIOSurfaceLockReadOnly, nullptr) == 0)
                        {
                            const auto* pixels = static_cast<const uint8_t*> (IOSurfaceGetBaseAddress (surface));
                            const int width = static_cast<int> (IOSurfaceGetWidth (surface));
                            const int height = static_cast<int> (IOSurfaceGetHeight (surface));
                            const size_t stride = IOSurfaceGetBytesPerRow (surface);
                            const bool readable = pixels != nullptr && width > 0 && height > 0
                                && IOSurfaceGetBytesPerElement (surface) == 4
                                && stride >= static_cast<size_t> (width) * 4;
                            if (readable)
                            {
                                const auto stats = sceneTraceColorStats (pixels, width, height, stride, true);
                                const auto* center = pixels + static_cast<size_t> (height / 2) * stride
                                    + static_cast<size_t> (width / 2) * 4;
                                const auto* leftCenter = pixels + static_cast<size_t> (height / 2) * stride
                                    + static_cast<size_t> (width / 4) * 4;
                                const auto* rightCenter = pixels + static_cast<size_t> (height / 2) * stride
                                    + static_cast<size_t> (width * 3 / 4) * 4;
                                std::fprintf (stderr,
                                    "[metal-frame-boundary] stage=producer seq=%llu slot=%u id=%u seconds=%.9f repeat=%d extent=%dx%d bright=%zu bounds=%d,%d,%d,%d center=%u,%u,%u,%u leftCenter=%u,%u,%u,%u rightCenter=%u,%u,%u,%u readable=1\n",
                                    static_cast<unsigned long long> (ready.frameSeq), ready.bufferIndex,
                                    id, ready.displaySec, sceneTracePublishedCount, width, height,
                                    stats.bright, stats.bright ? stats.minX : 0,
                                    stats.bright ? stats.minY : 0,
                                    stats.bright ? stats.maxX - stats.minX + 1 : 0,
                                    stats.bright ? stats.maxY - stats.minY + 1 : 0,
                                    static_cast<unsigned> (center[0]), static_cast<unsigned> (center[1]),
                                    static_cast<unsigned> (center[2]), static_cast<unsigned> (center[3]),
                                    static_cast<unsigned> (leftCenter[0]), static_cast<unsigned> (leftCenter[1]),
                                    static_cast<unsigned> (leftCenter[2]), static_cast<unsigned> (leftCenter[3]),
                                    static_cast<unsigned> (rightCenter[0]), static_cast<unsigned> (rightCenter[1]),
                                    static_cast<unsigned> (rightCenter[2]), static_cast<unsigned> (rightCenter[3]));
                            }
                            else
                                std::fprintf (stderr,
                                    "[metal-frame-boundary] stage=producer seq=%llu slot=%u id=%u seconds=%.9f repeat=%d read=unavailable extent=%dx%d stride=%zu\n",
                                    static_cast<unsigned long long> (ready.frameSeq), ready.bufferIndex,
                                    id, ready.displaySec, sceneTracePublishedCount, width, height, stride);
                            IOSurfaceUnlock (surface, kIOSurfaceLockReadOnly, nullptr);
                        }
                        else
                            std::fprintf (stderr,
                                "[metal-frame-boundary] stage=producer seq=%llu slot=%u id=%u seconds=%.9f repeat=%d read=unavailable lockFailed=%d\n",
                                static_cast<unsigned long long> (ready.frameSeq), ready.bufferIndex,
                                id, ready.displaySec, sceneTracePublishedCount, surface != nullptr);
                    }
#endif
                }
                else
                {
                    if (blitSucceeded && fenceSucceeded)
                        im.fdSocket.dropClient(); // socket send failed: treat as gone
                }
            }
            else
            {
                im.sharedFramesDroppedNoBuffer.fetch_add (1);
                viewportTelemetry.noFreeExportedBuffer();
            }
        }
#endif // ARBIT_HAVE_GPU_SHARED
        if (framePublished)
        {
            std::string commitError;
            if (! temporalSamplingTransaction.commit(commitError))
            {
                // The stale frame is dropped, not presented. The gate is told that no
                // frame reached the consumer, so the next tick produces again and
                // re-resolves the Imported Frame source against the new lifecycle.
                if (commitError == kPreviewLifecycleRetired && ! im.wantClose.load())
                {
                    ++retiredPublishRetries;
                    const auto retiredNow = std::chrono::steady_clock::now();
                    if (retiredNow - lastRetiredLog >= std::chrono::seconds (1))
                    {
                        lastRetiredLog = retiredNow;
                        std::fprintf (stderr, "[viewport] publish retry: reason=%s count=%llu\n",
                            commitError.c_str(), static_cast<unsigned long long> (retiredPublishRetries));
                    }
                    temporalSamplingTransaction.discard();
                    if (offscreen) publishGate.finish (gateSnapshot, false, false, shm);
                    // The swap at the top of the tick moved the visual events out of
                    // im.visualEvents. Hand them back for the next produce.
                    if (! frameVisualEvents.empty())
                    {
                        std::lock_guard<std::mutex> lock (im.mutex);
                        im.visualEvents.insert (im.visualEvents.begin(),
                                                frameVisualEvents.begin(), frameVisualEvents.end());
                        frameVisualEvents.clear();
                    }
                    enterLoopPhase (LoopPhase::FramePacing);
                    std::this_thread::sleep_for (std::chrono::milliseconds (10));
                    nextTickNs = nowNs();
                    continue;
                }
                closeForDeviceFailure("temporal sampling could not be committed", commitError);
                continue;
            }
            std::lock_guard<std::mutex> lock(im.mutex);
            im.presentedScoreGeneration = publishedScoreGeneration;
            im.publishedDisplaySec = displaySec;
            im.publishedScoreRevision = localScore.scoreRevision;
            im.publishedCanonicalNoteRows = cachedCanonicalBlockCFrame
                ? cachedCanonicalBlockCFrame->noteRows() : 0;
            im.publishedImportedSceneCount = importedSceneFrameOwners.size();
            im.publishedImportedSceneSubmittedInstances = 0;
            im.publishedImportedSceneSubmittedNoteInstances = 0;
            for (const auto& receipt : importedSceneFrameOwners)
            {
                im.publishedImportedSceneSubmittedInstances
                    += receipt.frame.rendered.stats.submittedInstanceCount;
                im.publishedImportedSceneSubmittedNoteInstances
                    += receipt.frame.rendered.stats.submittedNoteInstanceCount;
            }
        }
        else
            temporalSamplingTransaction.discard();
        enterLoopPhase (LoopPhase::Publish);
        if (offscreen)
        {
            // Idle gate. Decides whether the next tick must produce again. A frame
            // is final when nothing in it is still warming or animating.
            const bool settled = frameSettled && ! interpWarmingThisFrame
                && ! spectrumBakePending && ! playing && ! scriptActive
                && activeOpticalClips.empty() && scopeMask_.load() == 0;
            publishGate.finish (gateSnapshot, framePublished, settled, shm);
        }
        im.framesPresented.fetch_add (1);

        // Measured fps over a 1 s sliding window.
        const int64_t t = nowNs();
        frameTimes.push_back (t);
        while (! frameTimes.empty() && t - frameTimes.front() > 1'000'000'000ll)
            frameTimes.pop_front();
        {
            std::lock_guard<std::mutex> lock (im.mutex);
            im.measuredFps = (double) frameTimes.size();
            im.curW = fbW; im.curH = fbH;
        }

        // Pacing: vsync paces window mode; offscreen modes (no swap) sleep
        // out the remainder of the frame budget against an absolute tick.
        enterLoopPhase (LoopPhase::FramePacing);
        if (offscreen)
        {
            const auto frameDurNs = (int64_t) (frameBudgetSec * 1.0e9);
            nextTickNs = std::max (nextTickNs + frameDurNs, t - frameDurNs);
            const int64_t sleepNs = nextTickNs - nowNs();
            if (sleepNs > 0)
                std::this_thread::sleep_for (std::chrono::nanoseconds (sleepNs));
        }
        else if (frameTimes.size() >= 2
                 && (double) (t - frameTimes[frameTimes.size() - 2]) * 1e-9 < frameBudgetSec * 0.25)
            std::this_thread::sleep_for (
                std::chrono::duration<double> (frameBudgetSec * 0.5));
    }

    im.windowOpen = false;
    opticalFlowDecoder.stop();
    opticalFlowFields.clear();
    opticalFlowFieldKeys.clear();
    opticalDecodeOwners.clear();
    opticalUploadOwners.clear();
    for (auto& [clipId, cs] : streams)
    {
        renderer.deleteTexture (cs.tex);
        renderer.deleteTexture (cs.texA);   // frame-blend brackets
        renderer.deleteTexture (cs.texB);
        renderer.deleteTexture (cs.matteTex);
        renderer.deleteTexture (cs.matteTexB);
        cs.depth.clear(renderer);
    }
    streams.clear();
#if ARBIT_HAVE_ONNX || ARBIT_HAVE_NCNN
    {
        // Clear the pointer under im.mutex (so a concurrent info() RPC sees
        // null), then let it destruct OUTSIDE the lock — ~InterpEngine joins the
        // worker (can block ~one inference), which must not be held under im.mutex.
        std::unique_ptr<arbitinterp::InterpEngine> dying;
        {
            std::lock_guard<std::mutex> lk (im.mutex);
            dying = std::move (im.interp);
        }
    }   // dying joins the tier-2 worker + frees its RIFE/CUDA session here
#endif
    {
        // Texts and LUTs are cleared on viewport close (Arbit re-pushes on
        // open via pushClipGraphState / text re-registration).
        std::lock_guard<std::mutex> lock (im.mutex);
        for (auto& [textId, t] : im.texts)
            if (t.tex != 0)
                renderer.deleteTexture (t.tex);
        im.texts.clear();
        for (auto& [clipId, l] : im.luts)
            if (l.tex != 0)
                renderer.deleteTexture (l.tex);
        im.luts.clear();
    }
#if ARBIT_HAVE_GPU_SHARED
    if (shared)
    {
        // A device failure names its reason in this final BuffersGone, which goes out
        // before the socket closes. Consumers that predate the reason ignore the payload.
        if (im.fdSocket.hasClient())
        {
            const size_t reasonBytes = gpusurf::closeReasonPrefixBytes (
                reinterpret_cast<const uint8_t*> (deviceCloseReason.data()), deviceCloseReason.size());
            im.fdSocket.sendMsg ((uint32_t) gpusurf::MsgType::BuffersGone,
                                 reasonBytes > 0 ? deviceCloseReason.data() : nullptr, reasonBytes);
        }
        destroySharedBuffers();
        im.fdSocket.close();
        im.exporter.shutdown();
    }
#endif
    renderer.shutdown();
    videohelper::geometry::releaseGeometryExecutionCache();
    importedGeometryExecution.releaseNativeResources();
    importedSceneExecution.reset();
    importedAnimationExecution.reset();
#if defined(__APPLE__) && ARBIT_HAVE_IOSURFACE
    localMetalSurface.reset();
#endif
    releaseWindow();
}
