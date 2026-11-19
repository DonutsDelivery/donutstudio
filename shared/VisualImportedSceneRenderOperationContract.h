#pragma once

#include "DiffractionProductPlans.h"
#include "ImportedLightControl.h"
#include "VisualImportedObjectLocalTransform.h"

#include "DiffractionMaterialBindingContract.h"
#include "SurfaceMaterialBindingContract.h"
#include "VisualImportedAnimationDeformationRequest.h"
#include "VisualImportedAnimationOperationContract.h"
#include "VisualImportedSceneIdentity.h"
#include "VisualModelAssetContract.h"
#include "VisualModelAssetPayload.h"
#include "Visual3DScene.h"
#include "Visual3DSceneComposition.h"
#include "SceneAovOperationContract.h"
#include "Render3DImageOutput.h"
#include "MaterialFieldBinding.h"
#include "SurfaceMaterialProgramBinding.h"
#include "GeometryTimelineSceneBinding.h"
#include "HarmonicLinkSceneBinding.h"
#include "PathTraceLightPaths.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <set>
#include <istream>
#include <sstream>
#include <streambuf>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace visualimportedscenerender
{

inline constexpr std::string_view kCompiledOperationHeader =
    "visual.imported-scene.render.v1";
inline constexpr std::string_view kCompiledOperationHeaderV2 =
    "visual.imported-scene.render.v2";
inline constexpr std::string_view kCompiledOperationHeaderV3 =
    "visual.imported-scene.render.v3";
inline constexpr std::string_view kCompiledOperationHeaderV4 =
    "visual.imported-scene.render.v4";
inline constexpr std::string_view kCompiledOperationHeaderV5 =
    "visual.imported-scene.render.v5";
inline constexpr std::string_view kCompiledOperationHeaderV6 =
    "visual.imported-scene.render.v6";
inline constexpr std::string_view kCompiledOperationHeaderV7 =
    "visual.imported-scene.render.v7";
inline constexpr std::string_view kCompiledOperationHeaderV8 =
    "visual.imported-scene.render.v8";
inline constexpr std::string_view kCompiledOperationHeaderV9 =
    "visual.imported-scene.render.v9";
inline constexpr std::string_view kCompiledOperationHeaderV10 =
    "visual.imported-scene.render.v10";
inline constexpr std::string_view kCompiledOperationHeaderV11 =
    "visual.imported-scene.render.v11";
inline constexpr std::string_view kCompiledOperationHeaderV12 =
    "visual.imported-scene.render.v12";
inline constexpr std::string_view kCompiledOperationHeaderV13 =
    "visual.imported-scene.render.v13";
inline constexpr std::string_view kCompiledOperationHeaderV14 =
    "visual.imported-scene.render.v14";
inline constexpr std::string_view kCompiledOperationHeaderV15 =
    "visual.imported-scene.render.v15";
inline constexpr std::string_view kCompiledOperationHeaderV16 =
    "visual.imported-scene.render.v16";
inline constexpr std::string_view kCompiledOperationHeaderV17 =
    "visual.imported-scene.render.v17";
inline constexpr std::string_view kCompiledOperationHeaderV18 =
    "visual.imported-scene.render.v18";
inline constexpr std::string_view kCompiledOperationHeaderV19 =
    "visual.imported-scene.render.v19";
inline constexpr std::string_view kCompiledOperationHeaderV20 =
    "visual.imported-scene.render.v20";
inline constexpr std::string_view kCompiledOperationHeaderV21 =
    "visual.imported-scene.render.v21";
inline constexpr std::string_view kCompiledOperationHeaderV22 =
    "visual.imported-scene.render.v22";
inline constexpr std::string_view kCompiledOperationHeaderV23 =
    "visual.imported-scene.render.v23";
inline constexpr std::string_view kCompiledOperationHeaderV24 =
    "visual.imported-scene.render.v24";
inline constexpr std::string_view kCompiledOperationHeaderV25 =
    "visual.imported-scene.render.v25";
// Several exact imported leaves joined by one Scene Compose. Emitted only when
// a composition has more than one leaf; one leaf keeps its V20-V25 bytes.
inline constexpr std::string_view kCompiledOperationHeaderV26 =
    "visual.imported-scene.render.v26";
// Render engine selection, the outermost envelope. Emitted only for a
// non-raster engine, so every raster graph keeps its V1-V26 bytes.
inline constexpr std::string_view kCompiledOperationHeaderV27 =
    "visual.imported-scene.render.v27";
// V28: the preview sample target beside the engine selection. Emitted only
// when that target differs from the default, so a graph on the default keeps
// its exact V27 bytes and V27 stays readable.
inline constexpr std::string_view kCompiledOperationHeaderV28 =
    "visual.imported-scene.render.v28";
// V29: the Render node's quality mode, the outermost envelope. Emitted only
// for the Fast mode, so every Accurate graph keeps its V1-V28 bytes.
inline constexpr std::string_view kCompiledOperationHeaderV29 =
    "visual.imported-scene.render.v29";
// V30: the Render node's Path Trace denoise switch, the outermost envelope.
// Emitted only when the switch is on, so every other graph keeps its V1-V29
// bytes.
inline constexpr std::string_view kCompiledOperationHeaderV30 =
    "visual.imported-scene.render.v30";
// V31: the Environment light's HDR image and camera visibility, the outermost
// envelope. Emitted only when either differs from its default, so every other
// graph keeps its V1-V30 bytes.
inline constexpr std::string_view kCompiledOperationHeaderV31 =
    "visual.imported-scene.render.v31";
// V32: the Render node's view transform, on any engine, the outermost envelope.
// Emitted only for AgX, so every other graph keeps its V1-V31 bytes.
inline constexpr std::string_view kCompiledOperationHeaderV32 =
    "visual.imported-scene.render.v32";
// V33: the scene's Harmonic Link Geometry objects, the outermost envelope.
// Emitted only when a scene has one, so every other graph keeps its V1-V32 bytes.
inline constexpr std::string_view kCompiledOperationHeaderV33 =
    "visual.imported-scene.render.v33";
// V34: the Path Trace bounce limit, the outermost envelope. Emitted only when it
// differs from the default, so every other graph keeps its V1-V33 bytes.
inline constexpr std::string_view kCompiledOperationHeaderV34 =
    "visual.imported-scene.render.v34";
// V35: Path Trace participating-media settings, the outermost envelope. Emitted
// only when they differ from the default, so every other graph keeps its V1-V34 bytes.
inline constexpr std::string_view kCompiledOperationHeaderV35 =
    "visual.imported-scene.render.v35";
// V36: the Render node's Path Trace light paths (PathTraceLightPaths.h), the
// outermost envelope. Emitted only when they differ from the default, so every
// other graph keeps its V1-V35 bytes.
inline constexpr std::string_view kCompiledOperationHeaderV36 =
    "visual.imported-scene.render.v36";
// V37: the scene fog an Environment Light on the Render input brings to an
// imported scene, the outermost envelope. Emitted only when the fog is on, so
// every other graph keeps its V1-V36 bytes.
inline constexpr std::string_view kCompiledOperationHeaderV37 =
    "visual.imported-scene.render.v37";
// V40: the imported scene's other moved objects, beside the primary object-local
// transform. Only V41 wraps it. V35-V39 are reserved for envelopes nested
// inside it. Emitted only when a request moves more than one object, so every
// other graph keeps its older bytes.
inline constexpr std::string_view kCompiledOperationHeaderV40 =
    "visual.imported-scene.render.v40";
// V38: motion blur shutter and position, outside V35-V37 and inside V40.
// Emitted only when motion blur is on, so every other graph keeps its older bytes.
inline constexpr std::string_view kCompiledOperationHeaderV38 =
    "visual.imported-scene.render.v38";
// V39: the Render node's Path Trace denoiser by name, the envelope inside V40 and outside V35-V38.
// Emitted only for OptiX or OIDN, so Auto keeps its V30 bytes and every graph
// saved before the choice existed keeps its exact bytes.
inline constexpr std::string_view kCompiledOperationHeaderV39 =
    "visual.imported-scene.render.v39";
// V41: the Wide material ID scheme of a multi-leaf Scene Compose, the
// outermost envelope. Emitted only for Wide, so every saved composition keeps
// its V1-V40 bytes and its Narrow Material ID literals.
inline constexpr std::string_view kCompiledOperationHeaderV41 =
    "visual.imported-scene.render.v41";
// Raster is the native GPU renderer and the default. PathTrace selects the
// helper's CPU path tracer. Realtime is the Eevee-class GPU engine: the same scene
// and materials through real-time approximations, with no ray walks and no
// caustics. Value 2 is reserved for a Workbench engine and is refused until that
// engine exists.
enum class RenderEngine : std::uint8_t
{
    Raster = 0,
    PathTrace = 1,
    Realtime = 3
};
inline constexpr std::uint8_t kReservedWorkbenchRenderEngine = 2;
// What the frame reports when a Realtime scene selects a glass caustic. The frame
// still renders, without the caustic. The HUD's caustic row drops the first word.
inline constexpr std::string_view kRealtimeCausticsNeedPathTrace = "caustic off: caustics need Path Trace";
inline constexpr std::string_view kRealtimeReliefNeedsPathTrace = "relief scatter off: Relief Height needs Path Trace";
// The Render node's display step for scene-linear colour. Standard encodes radiance
// to sRGB and clips what is over range. AgX is shared/AgxViewTransform.h.
enum class ViewTransform : std::uint8_t
{
    Standard = 0,
    AgX = 1
};
// The Render node's quality mode, one switch for the viewport and for output.
// Accurate is the default and runs the engine's full-quality path, so its
// preview matches export. Fast is a cheaper preview: fewer bounces and samples,
// a sparser denoise and no Realtime reflections (video-helper
// render_mode_policy.h). Export draws the full settings in either mode.
enum class RenderMode : std::uint8_t
{
    Accurate = 0,
    Fast = 1
};
// The Render node's Path Trace denoiser (Plan 3, Denoising). Auto takes the
// OptiX AI denoiser on the OptiX device and OIDN elsewhere; OptiX and OIDN name
// one. The helper reports a choice it could not honour.
enum class PathTraceDenoiser : std::uint8_t
{
    Off = 0,
    Auto = 1,
    OptiX = 2,
    Oidn = 3
};
// The denoiser in the one byte the older on/off switch held. It still reads as
// that switch: true is any denoiser, and true written to it is Auto.
struct PathTraceDenoise final
{
    PathTraceDenoiser denoiser = PathTraceDenoiser::Off;

    constexpr PathTraceDenoise() noexcept = default;
    constexpr PathTraceDenoise(bool on) noexcept
        : denoiser(on ? PathTraceDenoiser::Auto : PathTraceDenoiser::Off) {}
    constexpr PathTraceDenoise(PathTraceDenoiser value) noexcept : denoiser(value) {}
    constexpr operator bool() const noexcept { return denoiser != PathTraceDenoiser::Off; }
    friend constexpr bool operator==(PathTraceDenoise left, PathTraceDenoise right) noexcept
    {
        return left.denoiser == right.denoiser;
    }
    friend constexpr bool operator!=(PathTraceDenoise left, PathTraceDenoise right) noexcept
    {
        return left.denoiser != right.denoiser;
    }
};
static_assert(sizeof(PathTraceDenoise) == 1 && alignof(PathTraceDenoise) == 1,
    "the denoiser keeps the switch's byte");
// Fixed samples per pixel for a path traced export. Raster requests keep the
// default so the field never changes raster identity. The maximum is the field's
// 16-bit format limit and nothing else.
inline constexpr std::uint16_t kDefaultPathTraceExportSamples = 64;
inline constexpr std::uint16_t kMaximumPathTraceExportSamples = 65535;
// Preview accumulation target: the viewport's own count beside an export's.
// Raster requests keep the default so the field never changes raster identity.
inline constexpr std::uint16_t kDefaultPathTracePreviewSamples = 4096;
inline constexpr std::uint16_t kMaximumPathTracePreviewSamples = 65535;
// Bounces after the camera hit; 0 is what the camera sees directly. Raster and Realtime requests keep the default so
// the field never changes their identity.
inline constexpr std::uint16_t kDefaultPathTraceMaxBounces = 128;
inline constexpr std::uint16_t kMaximumPathTraceMaxBounces = 65535;
// Motion blur: how long the shutter stays open, in thousandths of a frame, and
// where that interval sits around the frame time. 0 is off. Every engine reads it.
enum class MotionBlurPosition : std::uint8_t
{
    Start = 0,   // opens at the frame time
    Centre = 1,  // centred on the frame time
    End = 2      // closes at the frame time
};
inline constexpr std::uint16_t kMaximumMotionBlurShutter = 1000;
// The shutter as an interval in frames around the frame time: Start [0, s],
// Centre [-s/2, s/2], End [-s, 0] for a shutter of s frames. open == close is off.
struct ShutterInterval
{
    float open = 0.0f;
    float close = 0.0f;

    bool active() const noexcept { return close > open; }
};
inline ShutterInterval shutterInterval(std::uint16_t thousandths, MotionBlurPosition position) noexcept
{
    if (thousandths == 0) return {};
    const float length = static_cast<float>(thousandths) / static_cast<float>(kMaximumMotionBlurShutter);
    if (position == MotionBlurPosition::Start) return { 0.0f, length };
    if (position == MotionBlurPosition::End) return { -length, 0.0f };
    return { -0.5f * length, 0.5f * length };
}
// Path Trace media settings in one byte. Bits 0-6 hold the volume bounce limit,
// where the top step lifts it, and bit 7 hides media from camera rays. The
// default lifts the limit and shows media, so the overall bounce limit governs.
inline constexpr std::uint8_t kPathTraceVolumeBounceMask = 0x7f;
inline constexpr std::uint8_t kPathTraceVolumeUnlimitedBounces = kPathTraceVolumeBounceMask;
inline constexpr std::uint8_t kPathTraceVolumeHiddenFromCamera = 0x80;
inline constexpr std::uint8_t kDefaultPathTraceVolume = kPathTraceVolumeUnlimitedBounces;
inline constexpr std::uint8_t pathTraceVolume(std::uint32_t bounces, bool visibleToCamera) noexcept
{
    const auto limit = bounces < kPathTraceVolumeUnlimitedBounces ? bounces : kPathTraceVolumeUnlimitedBounces;
    return static_cast<std::uint8_t>(limit | (visibleToCamera ? 0u : kPathTraceVolumeHiddenFromCamera));
}
// The volume bounce limit, or nothing when the top step lifts it.
inline constexpr std::optional<std::uint32_t> pathTraceVolumeBounces(std::uint8_t volume) noexcept
{
    const auto limit = static_cast<std::uint32_t>(volume & kPathTraceVolumeBounceMask);
    return limit == kPathTraceVolumeUnlimitedBounces ? std::nullopt : std::optional<std::uint32_t>(limit);
}
inline constexpr bool pathTraceVolumesVisibleToCamera(std::uint8_t volume) noexcept
{
    return (volume & kPathTraceVolumeHiddenFromCamera) == 0;
}
inline constexpr std::size_t kMaximumImportedLeaves =
    HarmonicMIDI::grid::scenecomposition::kMaximumImportedLeaves;
// Refusal for a route that still draws one imported leaf only.
inline constexpr std::string_view kMultipleImportedLeavesUnavailable =
    "Scene Compose refused: multiple imported leaves are not rendered yet";
// Render Passes still draw leaf 0 only; the Render node draws every leaf.
inline constexpr std::string_view kRenderPassesMultipleImportedLeavesUnavailable =
    "Scene Compose refused: Render Passes with multiple imported leaves are not rendered yet";
// Path Trace draws through the fixture scene renderer. Routes that the
// deformation renderer draws refuse it by name instead of drawing Raster.
inline constexpr std::string_view kPathTraceAnimatedMeshUnavailable =
    "Path Trace does not render animated meshes yet; choose Raster";
inline constexpr std::string_view kRealtimeAnimatedMeshUnavailable =
    "Realtime does not render this animated mesh yet; choose Raster";
// The deformation renderer draws Raster only. Every other engine draws an
// animated mesh through the fixture scene renderer, with this frame's skinned
// and morphed vertices baked into the scene.
inline constexpr bool animatedMeshDrawsBaked(RenderEngine engine) noexcept
{
    return engine != RenderEngine::Raster;
}
inline std::string_view animatedMeshUnavailable(RenderEngine engine) noexcept
{
    return engine == RenderEngine::Realtime ? kRealtimeAnimatedMeshUnavailable : kPathTraceAnimatedMeshUnavailable;
}
inline constexpr std::string_view kSourceNodeKind =
    "visual.3d.imported-animation";
inline constexpr std::string_view kRenderNodeKind = "visual.3d.render";
inline constexpr std::string_view kSourceBackendCapability = "source-decode";
inline constexpr std::string_view kRenderBackendCapability = "native-gpu";
// Every envelope holds what its size_t length fields can carry. A count read
// from the wire is checked against the bytes present before it allocates, so
// these bounds only keep the length arithmetic below from overflowing.
inline constexpr std::size_t kMaximumV7EncodedBytes = sceneaov::detail::kMaximumSceneBytes * 2;
inline constexpr std::size_t kMaximumEncodedBytes = kMaximumV7EncodedBytes;
inline constexpr std::size_t kMaximumV2EncodedBytes = kMaximumV7EncodedBytes;
inline constexpr std::size_t kMaximumV3EncodedBytes = kMaximumV7EncodedBytes;
inline constexpr std::size_t kMaximumV4EncodedBytes = kMaximumV7EncodedBytes;
inline constexpr std::size_t kMaximumV8EncodedBytes
    = sceneaov::detail::kMaximumSceneBytes * 2 + kMaximumV7EncodedBytes;
inline constexpr std::size_t kMaximumDeformationBytes = kMaximumV7EncodedBytes;

struct NoteLight final
{
    std::uint32_t track = 0;
    float gain = 2.0f;
    float pitchDegrees = 30.0f;
};

// Authored limits are retained with the request; native admission remains separate.
struct GlassCausticBranchPolicy final
{
    std::uint32_t version = 0;
    float maximumUnresolvedFraction = 0, minimumBranchFraction = 0;
    std::uint32_t maximumInterfaces = 0, maximumVisits = 0, maximumPending = 0;

    bool valid() const noexcept
    {
        return version == 1 && std::isfinite(maximumUnresolvedFraction)
            && std::isfinite(minimumBranchFraction) && maximumUnresolvedFraction > 0
            && maximumUnresolvedFraction <= 1.0e-4f && minimumBranchFraction > 0
            && minimumBranchFraction <= maximumUnresolvedFraction
            // The deposit receipt holds the interface and visit counts as exact floats.
            // The pending stack lives in device storage sized from maximumPending and
            // a walk never holds more than visits+1, so it shares the visit bound.
            && maximumInterfaces > 0 && maximumInterfaces <= 16777215u
            && maximumVisits > 0 && maximumVisits <= 16777215u
            && maximumPending > 0 && maximumPending <= 16777215u;
    }

    std::string identity() const
    {
        std::string out = "glass-caustic-branch-policy-v1:";
        const auto add = [&out](std::uint64_t value)
        {
            static constexpr char hex[] = "0123456789abcdef";
            for (unsigned digit = 16; digit > 0; --digit)
                out.push_back(hex[(value >> ((digit - 1) * 4)) & 15]);
        };
        const auto floating = [&add](float value)
        {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            add(bits);
        };
        add(version); floating(maximumUnresolvedFraction); floating(minimumBranchFraction);
        add(maximumInterfaces); add(maximumVisits); add(maximumPending);
        return out;
    }
};

struct GlassCausticSelection final
{
    std::uint32_t version = 0;
    HarmonicMIDI::grid::SceneLightId sourceLight;
    HarmonicMIDI::grid::SceneObjectId glassObject;
    HarmonicMIDI::grid::SceneMaterialId glassMaterial;
    GlassCausticBranchPolicy policy;
};

inline bool sameGlassCaustic(const GlassCausticSelection& a, const GlassCausticSelection& b)
{
    return a.version == b.version && a.sourceLight == b.sourceLight
        && a.glassObject == b.glassObject && a.glassMaterial == b.glassMaterial
        && a.policy.identity() == b.policy.identity();
}

inline bool validGlassCausticScene(const GlassCausticSelection& selection,
                                  const HarmonicMIDI::grid::Visual3DScene& scene) noexcept
{
    using namespace HarmonicMIDI::grid;
    if (selection.version != 1 || !selection.policy.valid() || !selection.sourceLight.isValid()
        || !selection.glassObject.isValid() || !selection.glassMaterial.isValid()
        || !validateVisual3DScene(scene).valid()) return false;
    const auto* source = visual3d_detail::findById(scene.lights, scene.lightCount, selection.sourceLight);
    const auto* object = visual3d_detail::findById(scene.objects, scene.objectCount, selection.glassObject);
    return source && source->kind == SceneLightKind::Directional
        && source->spectralSource.version == 1 && validSceneSpectralSource(*source)
        && (source->spectralSource.directionalIrradiance.version==1
            ? std::any_of(source->spectralSource.directionalIrradiance.coefficients.begin(),
                          source->spectralSource.directionalIrradiance.coefficients.end(), [](float value) { return value>0; })
            : std::any_of(source->spectralSource.radiance.begin(), source->spectralSource.radiance.end(),
                          [](float value) { return value>0; }))
        && object && object->indexCount != 0 && object->material == selection.glassMaterial;
}

// IDs remain authored procedural IDs here. The helper resolves actual imported
// records and maps procedural IDs only after admitting the selected source.
struct ImportedParentIntentV1 final
{
    struct RootMembership final
    {
        std::uint32_t objectId = 0;
        std::uint32_t inputIndex = 0;
    };
    std::uint32_t version = 1;
    std::uint32_t direction = 0;
    std::uint64_t composeStableId = 0;
    std::uint64_t importedSourceStableId = 0;
    std::uint32_t connectedInputMask = 0;
    std::uint32_t requestedParentMask = 0;
    std::uint32_t importedInputIndex = 0;
    std::uint32_t reserved = 0;
    std::uint32_t proceduralAnchorId = 0;
    std::uint32_t rootMembershipCount = 0;
    std::array<RootMembership, 64> rootMembership {};
};

inline constexpr std::size_t kImportedParentIntentBytes = 48 + 64 * 8;
inline constexpr std::size_t kImportedParentIntentHexBytes = 2 * kImportedParentIntentBytes;
inline constexpr std::size_t kImportedParentWrapperBytes = 1154;
static_assert(kCompiledOperationHeaderV23.size() + 2 + kImportedParentIntentHexBytes
    == kImportedParentWrapperBytes);

inline bool validImportedParentIntent(const ImportedParentIntentV1& value) noexcept
{
    using HarmonicMIDI::grid::SceneObjectId;
    if (value.version != 1 || (value.direction != 1 && value.direction != 2)
        || value.composeStableId == 0 || value.importedSourceStableId == 0
        || value.composeStableId == value.importedSourceStableId || value.reserved != 0
        || value.connectedInputMask > 15 || !(value.connectedInputMask & 1)
        || value.requestedParentMask == 0 || (value.requestedParentMask & ~14u)
        || (value.requestedParentMask & value.connectedInputMask) != value.requestedParentMask
        || value.importedInputIndex > 3
        || !(value.connectedInputMask & (1u << value.importedInputIndex))
        || value.rootMembershipCount > value.rootMembership.size()) return false;
    if (value.direction == 1)
    {
        if (value.importedInputIndex == 0
            || !(value.requestedParentMask & (1u << value.importedInputIndex))
            || !HarmonicMIDI::grid::scenecomposition::proceduralObjectId({value.proceduralAnchorId})
            || value.rootMembershipCount != 0) return false;
    }
    else if (value.importedInputIndex != 0 || value.proceduralAnchorId != 0
        || value.rootMembershipCount == 0) return false;
    std::uint32_t coveredInputs = 0;
    for (std::size_t i = 0; i < value.rootMembership.size(); ++i)
    {
        const auto& root = value.rootMembership[i];
        if (i >= value.rootMembershipCount)
        {
            if (root.objectId != 0 || root.inputIndex != 0) return false;
            continue;
        }
        if (root.inputIndex == 0 || root.inputIndex > 3
            || !(value.connectedInputMask & (1u << root.inputIndex))
            || !HarmonicMIDI::grid::scenecomposition::proceduralObjectId(SceneObjectId{root.objectId})) return false;
        if (i && (root.inputIndex < value.rootMembership[i - 1].inputIndex
            || (root.inputIndex == value.rootMembership[i - 1].inputIndex
                && root.objectId <= value.rootMembership[i - 1].objectId))) return false;
        for (std::size_t j = 0; j < i; ++j)
            if (value.rootMembership[j].objectId == root.objectId) return false;
        coveredInputs |= 1u << root.inputIndex;
    }
    return value.direction == 1 || coveredInputs == (value.connectedInputMask & ~1u);
}

inline bool sameImportedParentIntent(const ImportedParentIntentV1& a,
                                    const ImportedParentIntentV1& b) noexcept
{
    if (a.version != b.version || a.direction != b.direction
        || a.composeStableId != b.composeStableId || a.importedSourceStableId != b.importedSourceStableId
        || a.connectedInputMask != b.connectedInputMask || a.requestedParentMask != b.requestedParentMask
        || a.importedInputIndex != b.importedInputIndex || a.reserved != b.reserved
        || a.proceduralAnchorId != b.proceduralAnchorId || a.rootMembershipCount != b.rootMembershipCount) return false;
    for (std::size_t i = 0; i < a.rootMembership.size(); ++i)
        if (a.rootMembership[i].objectId != b.rootMembership[i].objectId
            || a.rootMembership[i].inputIndex != b.rootMembership[i].inputIndex) return false;
    return true;
}

struct Request final
{
    // One exact imported leaf joins a GLB-free procedural snapshot. Its
    // original selection and material target identities remain unchanged.
    std::shared_ptr<const Request> importedScene;
    std::optional<ImportedParentIntentV1> importedParentIntent {};
    visualanimationimport::StableId sourceStableId = 0;
    visualanimationimport::StableId renderStableId = 0;
    visualanimationimport::ExactContentAssetKey asset;
    std::optional<std::size_t> sceneIndex;
    // Genuine one-based source GLB camera node, never a synthesized fallback.
    std::optional<std::uint32_t> sourceCameraNodeId {};
    // Exact source node and mesh for the existing world-space modulation overlay.
    // Zero preserves legacy whole-scene targeting.
    std::uint32_t objectNodeStableId = 0;
    std::uint32_t objectMeshStableId = 0;
    visualimportedobjectlocal::ObjectLocalTransformV1 objectLocalTransform;
    // Other moved objects of the same scene, each on its own node. Non-empty only
    // beside a version 1 primary transform.
    std::vector<visualimportedobjectlocal::ObjectLocalTransformV1> additionalObjectLocalTransforms;
    std::optional<visualanimationimport::Request> deformation;
    std::optional<surfacematerialbinding::ImportedSceneMaterialRequest> material;
    std::optional<diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest>
        diffractionMaterial;
    std::optional<materialfield::Binding> materialField;
    geometrysurfacematerial::ObjectPrograms surfacePrograms;
    std::optional<HarmonicMIDI::grid::SceneCameraRecord> camera;
    std::optional<HarmonicMIDI::grid::SceneLightRecord> light;
    std::optional<NoteLight> noteLight;
    std::optional<GlassCausticSelection> glassCaustic;
    importedlightcontrol::Controls importedLights;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> sceneSnapshot;
    std::optional<videowire::geometry::TimelineSceneBinding> timelineGeometry;
    std::vector<videowire::geometry::TimelineSceneBinding> timelineGeometryFamily;
    std::vector<videowire::geometry::HarmonicLinkSceneBinding> harmonicLinks;
    std::uint64_t structuralRevision = 0;
    std::uint64_t evaluationRevision = 0;
    renderpassoutput::Output imageOutput = renderpassoutput::Output::Color;
    // The engine fields fill the padding between the one-byte imageOutput and
    // rawExportMask. The preview target has no padding left, so it extends the
    // struct and kRequestMemberBytes follows it.
    RenderEngine renderEngine = RenderEngine::Raster;
    std::uint16_t pathTraceExportSamples = kDefaultPathTraceExportSamples;
    std::uint16_t pathTracePreviewSamples = kDefaultPathTracePreviewSamples;
    // One byte, placed in the padding before rawExportMask.
    RenderMode renderMode = RenderMode::Accurate;
    // One byte, the last of that padding. Only Path Trace may set it.
    PathTraceDenoise pathTraceDenoise {};
    std::uint32_t rawExportMask = 0;
    // One byte, in the padding before the leaf vector, so kRequestMemberBytes keeps
    // its figure. Only Path Trace may set it.
    ViewTransform viewTransform = ViewTransform::Standard;
    // One byte, the last of that padding, after the view transform. Only Path Trace may set it.
    std::uint8_t pathTraceVolume = kDefaultPathTraceVolume;
    // Two bytes, in the rest of that padding. Only Path Trace may set it.
    std::uint16_t pathTraceMaxBounces = kDefaultPathTraceMaxBounces;
    // No padding is left, so the light paths extend the struct and
    // kRequestMemberBytes follows them. Only Path Trace may set them.
    PathTraceLightPaths pathTraceLightPaths {};
    // Scene fog from the Environment Light on the Render input. It replaces the
    // scene's own fog for every engine; absent leaves the scene's fog as it is.
    std::optional<HarmonicMIDI::grid::SceneVolume> worldVolume {};
    // No padding is left, so these extend the struct and kRequestMemberBytes
    // follows them. Off (0, Centre) on every request that does not blur.
    std::uint16_t motionBlurShutter = 0;
    MotionBlurPosition motionBlurPosition = MotionBlurPosition::Centre;
    // The material and texture ID scheme of the composed leaves; Wide needs
    // at least one additional leaf. pathTraceVolume took the last padding
    // byte, so this one extends the struct.
    HarmonicMIDI::grid::scenecomposition::ImportedLeafIdScheme leafIdScheme =
        HarmonicMIDI::grid::scenecomposition::ImportedLeafIdScheme::Narrow;
    // Leaves 1..N-1 of a multi-leaf Scene Compose, in ascending source node
    // order. importedScene stays leaf 0. Node, camera and material selectors
    // of leaf k use scenecomposition::encodeImportedLeafId.
    std::vector<std::shared_ptr<const Request>> additionalImportedScenes {};
};

// Request bytes without the multi-leaf vector. Single-leaf copy accounting uses
// this so its figures match the bound in force before multi-leaf compositions.
inline constexpr std::size_t kRequestMemberBytes =
    sizeof(Request) - sizeof(std::vector<std::shared_ptr<const Request>>);

// renderMode sits in the padding between the preview samples and rawExportMask.
// These mirrors of that tail keep the insertion from growing the request, so
// kRequestMemberBytes and the copy budgets built on it keep their old figure.
namespace detail
{
struct RequestTailBeforeMode final
{
    renderpassoutput::Output imageOutput; RenderEngine engine;
    std::uint16_t exportSamples; std::uint16_t previewSamples; std::uint32_t rawExportMask;
};
struct RequestTailWithMode final
{
    renderpassoutput::Output imageOutput; RenderEngine engine;
    std::uint16_t exportSamples; std::uint16_t previewSamples; RenderMode mode; std::uint32_t rawExportMask;
};
static_assert(sizeof(RequestTailWithMode) == sizeof(RequestTailBeforeMode)
        && alignof(RequestTailWithMode) == alignof(RequestTailBeforeMode),
    "renderMode must stay inside the padding before rawExportMask");
struct RequestTailWithDenoise final
{
    renderpassoutput::Output imageOutput; RenderEngine engine;
    std::uint16_t exportSamples; std::uint16_t previewSamples; RenderMode mode; bool denoise;
    std::uint32_t rawExportMask;
};
static_assert(sizeof(RequestTailWithDenoise) == sizeof(RequestTailBeforeMode)
        && alignof(RequestTailWithDenoise) == alignof(RequestTailBeforeMode),
    "pathTraceDenoise must stay inside the padding before rawExportMask");
// The tail starts on an eight-byte boundary after evaluationRevision and the leaf
// vector ends it on another, which leaves four bytes for the view transform.
struct RequestTailBeforeTransform final
{
    std::uint64_t revision; renderpassoutput::Output imageOutput; RenderEngine engine;
    std::uint16_t exportSamples; std::uint16_t previewSamples; RenderMode mode; bool denoise;
    std::uint32_t rawExportMask; std::vector<std::shared_ptr<const Request>> leaves;
};
struct RequestTailWithTransform final
{
    std::uint64_t revision; renderpassoutput::Output imageOutput; RenderEngine engine;
    std::uint16_t exportSamples; std::uint16_t previewSamples; RenderMode mode; bool denoise;
    std::uint32_t rawExportMask; ViewTransform viewTransform;
    std::vector<std::shared_ptr<const Request>> leaves;
};
static_assert(sizeof(RequestTailWithTransform) == sizeof(RequestTailBeforeTransform),
    "viewTransform must stay inside the padding before the leaf vector");
struct RequestTailWithBounces final
{
    std::uint64_t revision; renderpassoutput::Output imageOutput; RenderEngine engine;
    std::uint16_t exportSamples; std::uint16_t previewSamples; RenderMode mode; bool denoise;
    std::uint32_t rawExportMask; ViewTransform viewTransform; std::uint16_t maxBounces;
    std::vector<std::shared_ptr<const Request>> leaves;
};
static_assert(sizeof(RequestTailWithBounces) == sizeof(RequestTailBeforeTransform),
    "pathTraceMaxBounces must stay inside the padding before the leaf vector");
struct RequestTailWithVolume final
{
    std::uint64_t revision; renderpassoutput::Output imageOutput; RenderEngine engine;
    std::uint16_t exportSamples; std::uint16_t previewSamples; RenderMode mode; bool denoise;
    std::uint32_t rawExportMask; ViewTransform viewTransform; std::uint8_t volume; std::uint16_t maxBounces;
    std::vector<std::shared_ptr<const Request>> leaves;
};
static_assert(sizeof(RequestTailWithVolume) == sizeof(RequestTailBeforeTransform),
    "pathTraceVolume must stay inside the padding before the leaf vector");
} // namespace detail

namespace detail
{
// The packed wire is little endian and independent of host padding. Every
// fixed slot participates, including the canonical unused-zero suffix.
inline std::array<char, kImportedParentIntentHexBytes>
encodeImportedParentIntent(const ImportedParentIntentV1& value) noexcept
{
    std::array<char, kImportedParentIntentHexBytes> text {};
    std::size_t cursor = 0;
    const auto integer = [&](std::uint64_t number, std::size_t bytes)
    {
        constexpr char hex[] = "0123456789abcdef";
        for (std::size_t i = 0; i < bytes; ++i)
        {
            const auto byte = static_cast<unsigned>((number >> (8 * i)) & 255);
            text[cursor++] = hex[byte >> 4]; text[cursor++] = hex[byte & 15];
        }
    };
    integer(value.version, 4); integer(value.direction, 4);
    integer(value.composeStableId, 8); integer(value.importedSourceStableId, 8);
    integer(value.connectedInputMask, 4); integer(value.requestedParentMask, 4);
    integer(value.importedInputIndex, 4); integer(value.reserved, 4);
    integer(value.proceduralAnchorId, 4); integer(value.rootMembershipCount, 4);
    for (const auto& root : value.rootMembership)
    { integer(root.objectId, 4); integer(root.inputIndex, 4); }
    return text;
}

inline bool decodeImportedParentIntent(std::string_view text, ImportedParentIntentV1& output) noexcept
{
    if (text.size() != kImportedParentIntentHexBytes) return false;
    const auto nibble = [](char c) -> int
    { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1; };
    std::size_t cursor = 0;
    const auto integer = [&](auto& number) -> bool
    {
        number = 0;
        for (std::size_t i = 0; i < sizeof(number); ++i)
        {
            const auto high = nibble(text[cursor++]), low = nibble(text[cursor++]);
            if (high < 0 || low < 0) return false;
            number |= static_cast<std::decay_t<decltype(number)>>(high * 16 + low) << (8 * i);
        }
        return true;
    };
    ImportedParentIntentV1 value;
    if (!integer(value.version) || !integer(value.direction)
        || !integer(value.composeStableId) || !integer(value.importedSourceStableId)
        || !integer(value.connectedInputMask) || !integer(value.requestedParentMask)
        || !integer(value.importedInputIndex) || !integer(value.reserved)
        || !integer(value.proceduralAnchorId) || !integer(value.rootMembershipCount)) return false;
    for (auto& root : value.rootMembership)
        if (!integer(root.objectId) || !integer(root.inputIndex)) return false;
    if (!validImportedParentIntent(value)) return false;
    output = value; return true;
}

inline constexpr std::size_t kObjectLocalTransformBytes = 3 * 4 + 9 * 8;
inline constexpr std::size_t kObjectLocalTransformHexBytes = 2 * kObjectLocalTransformBytes;

inline std::array<char, kObjectLocalTransformHexBytes>
encodeObjectLocalTransform(const visualimportedobjectlocal::ObjectLocalTransformV1& value) noexcept
{
    std::array<char, kObjectLocalTransformHexBytes> text {};
    std::size_t cursor = 0;
    const auto integer = [&](std::uint64_t number, std::size_t bytes)
    {
        constexpr char hex[] = "0123456789abcdef";
        for (std::size_t i = 0; i < bytes; ++i)
        {
            const auto byte = static_cast<unsigned>((number >> (8 * i)) & 255);
            text[cursor++] = hex[byte >> 4]; text[cursor++] = hex[byte & 15];
        }
    };
    integer(value.version, 4); integer(value.nodeStableId, 4); integer(value.meshStableId, 4);
    for (const auto* triple : { &value.translation, &value.rotationDegrees, &value.scale })
        for (const auto number : *triple) integer(visualimportedobjectlocal::bits(number), 8);
    return text;
}

inline bool decodeObjectLocalTransform(std::string_view text,
    visualimportedobjectlocal::ObjectLocalTransformV1& output) noexcept
{
    if (text.size() != kObjectLocalTransformHexBytes) return false;
    const auto nibble = [](char c) -> int
    { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1; };
    std::size_t cursor = 0;
    const auto integer = [&](auto& number) -> bool
    {
        number = 0;
        for (std::size_t i = 0; i < sizeof(number); ++i)
        {
            const auto high = nibble(text[cursor++]), low = nibble(text[cursor++]);
            if (high < 0 || low < 0) return false;
            number |= static_cast<std::decay_t<decltype(number)>>(high * 16 + low) << (8 * i);
        }
        return true;
    };
    visualimportedobjectlocal::ObjectLocalTransformV1 value;
    if (!integer(value.version) || !integer(value.nodeStableId) || !integer(value.meshStableId)) return false;
    for (auto* triple : { &value.translation, &value.rotationDegrees, &value.scale })
        for (auto& number : *triple)
        {
            std::uint64_t bits = 0;
            if (!integer(bits)) return false;
            std::memcpy(&number, &bits, sizeof(number));
        }
    if (!visualimportedobjectlocal::valid(value)) return false;
    output = value; return true;
}

inline bool validImportedParentRequest(const Request& request) noexcept
{
    using namespace HarmonicMIDI::grid;
    if (!request.importedParentIntent) return true;
    if (!request.importedScene || !request.sceneSnapshot
        || request.importedScene->importedScene || request.importedScene->sceneSnapshot
        || request.importedScene->importedParentIntent
        || request.timelineGeometry || !request.timelineGeometryFamily.empty()
        || !request.harmonicLinks.empty()
        || request.glassCaustic || request.noteLight) return false;
    const auto& value = *request.importedParentIntent;
    const auto& scene = *request.sceneSnapshot;
    const auto& leaf = *request.importedScene;
    // Full scene validation remains on the unchanged procedural request below.
    // Establish the fixed object bound before any parent membership lookup.
    if (!validImportedParentIntent(value) || scene.objectCount > scene.objects.size()
        || value.composeStableId != request.sourceStableId || scene.id.value != value.composeStableId
        || value.importedSourceStableId != leaf.sourceStableId
        || leaf.structuralRevision != request.structuralRevision
        || leaf.evaluationRevision != request.evaluationRevision
        || leaf.renderStableId != request.renderStableId
        || leaf.timelineGeometry || !leaf.timelineGeometryFamily.empty()
        || (leaf.material && leaf.material->vertexModifier)) return false;
    for (const auto& object : request.surfacePrograms)
        if (object.program.material.vertexModifier) return false;
    if (value.direction == 1)
        return scene.objectCount > 0 && scene.objects[0].id.value == value.proceduralAnchorId;
    std::vector<bool> seen(scene.objectCount, false);
    std::size_t roots = 0;
    for (std::size_t i = 0; i < scene.objectCount; ++i)
        if (!scene.objects[i].parent.isValid()) ++roots;
    if (roots != value.rootMembershipCount) return false;
    for (std::size_t i = 0; i < value.rootMembershipCount; ++i)
    {
        const auto* object = visual3d_detail::findById(scene.objects, scene.objectCount,
            SceneObjectId{value.rootMembership[i].objectId});
        if (!object || object->parent.isValid()) return false;
        const auto index = static_cast<std::size_t>(object - scene.objects.data());
        if (seen[index]) return false;
        seen[index] = true;
    }
    for (std::size_t i = 0; i < scene.objectCount; ++i)
        if (!scene.objects[i].parent.isValid() && !seen[i]) return false;
    return true;
}

// Counts only the additional family codec, not native admission, allocation
// internals or driver work. The public codecs derive this before family decode.
struct TimelineFamilyCodecEnvelope final
{
    std::size_t textBytes = 0, decodedBytes = 0, operations = 0;
    std::size_t familyValidationPasses = 0, planDecodeCalls = 0;
    std::size_t sceneValidationCalls = 0, canonicalByteVisits = 0;
    std::size_t planByteVisits = 0, requestCopyBytes = 0;
};

// Read the released GCLP/GCRV header directly from hexadecimal, with no decoder
// staging. Attribute descriptors have an 8-byte ID, u16 name and three enums.
// Full contract and operation validation still belongs to the released decoder.
inline bool timelinePlanOperationCount(std::string_view text,std::size_t& operations)
{
    using namespace videowire::geometry;
    if (text.empty() || text.size()%2) return false;
    const auto nibble=[](char c)->int {
        return c>='0' && c<='9' ? c-'0' : c>='a' && c<='f' ? c-'a'+10 : -1;
    };
    for (const auto c : text) if (nibble(c)<0) return false;
    std::size_t cursor=0;
    const auto skip=[&](std::size_t bytes) {
        if (bytes>text.size()/2-cursor) return false;
        cursor+=bytes; return true;
    };
    const auto integer=[&](std::size_t bytes,std::uint64_t& result) {
        if (bytes>8 || bytes>text.size()/2-cursor) return false;
        result=0;
        for (std::size_t i=0;i<bytes;++i) {
            const auto offset=2*(cursor+i);
            result|=std::uint64_t((nibble(text[offset])<<4)|nibble(text[offset+1]))<<(8*i);
        }
        cursor+=bytes; return true;
    };
    std::uint64_t magic=0,version=0,flags=0,count=0;
    if (!integer(4,magic) || magic!=0x504c4347 || !integer(2,version) || version!=1
        || !integer(2,flags) || flags!=0 || !skip(4+5+8*8)
        || !integer(8,count) || count>kMaximumAttributes) return false;
    for (std::uint64_t i=0;i<count;++i) {
        std::uint64_t length=0;
        if (!skip(8) || !integer(2,length) || !length || length>kMaximumNameBytes
            || !skip(static_cast<std::size_t>(length)+3)) return false;
    }
    std::uint64_t runtimeBytes=0;
    if (!integer(8,runtimeBytes) || runtimeBytes!=text.size()/2-cursor || runtimeBytes<73
        || !integer(4,magic) || magic!=0x56524347 || !integer(2,version)
        || version<1 || version>kTransportVersion || !integer(2,flags) || flags!=0
        || !skip(4+1+3*8+4+2*8) || !integer(8,count) || count>kMaximumAttributes
        || !integer(8,count) || count==0 || count>kMaximumOperations) return false;
    operations=static_cast<std::size_t>(count); return true;
}

inline bool deriveTimelineFamilyCodecEnvelope(std::size_t bindingCount,std::size_t textBytes,
    std::size_t operations,std::size_t outerWrappers,std::size_t wholeBytes,TimelineFamilyCodecEnvelope& output)
{
    using namespace videowire::geometry;
    if (bindingCount<2 || textBytes==0 || textBytes%2
        || operations<bindingCount || outerWrappers>5 || wholeBytes>kMaximumV8EncodedBytes) return false;
    TimelineFamilyCodecEnvelope result;
    result.textBytes=textBytes; result.operations=operations;
    result.decodedBytes=result.textBytes/2;
    // V22 composition override, V20, V19, V17 and V15 precede V21.
    // At each decode ancestor: valid plus canonical encode, whose remaining
    // ancestors each validate once. V20 validates the procedural request twice
    // and encodes it recursively. (n+1)*(n+4) covers these repeated passes,
    // including the V21 check itself and malformed noncanonical outer branches.
    if (!checkedMul(outerWrappers+1,outerWrappers+4,result.familyValidationPasses)
        || !checkedMul(result.familyValidationPasses,bindingCount,result.planDecodeCalls)
        // Each binding check validates its scene and each request validates once.
        || !checkedAdd(result.planDecodeCalls,result.familyValidationPasses,result.sceneValidationCalls)
        // Header census, hexadecimal decode, lowered-runtime copy and runtime
        // read each traverse at most the complete lowered bytes.
        || !checkedMul(result.familyValidationPasses,4,result.planByteVisits)
        || !checkedMul(result.planByteVisits,result.textBytes,result.planByteVisits)
        // Each outer decode owns one stream copy. Each canonical encode owns
        // the nested/result/stream copies and final equality visits every byte.
        || !checkedMul(result.familyValidationPasses,4,result.canonicalByteVisits)
        || !checkedMul(result.canonicalByteVisits,wholeBytes,result.canonicalByteVisits)
        // Conservatively retain all ancestor Request copies concurrently.
        || !checkedMul(outerWrappers+2,result.textBytes,result.requestCopyBytes)
        || !addArrayBytes(outerWrappers+2,kRequestMemberBytes,result.requestCopyBytes)
        || !addArrayBytes(outerWrappers+2,bindingCount*sizeof(TimelineSceneBinding),result.requestCopyBytes)) return false;
    output=result; return true;
}

inline bool timelineFamilyEnvelope(
    const std::vector<videowire::geometry::TimelineSceneBinding>& bindings,
    std::size_t outerWrappers,std::size_t wholeBytes,TimelineFamilyCodecEnvelope& output)
{
    using namespace videowire::geometry;
    if (bindings.size()<2 || outerWrappers>4
        || wholeBytes>kMaximumV8EncodedBytes) return false;
    TimelineFamilyCodecEnvelope result;
    std::uint32_t previous=0;
    for (const auto& binding : bindings) {
        if (binding.object.value<=previous || binding.loweredPlan.empty() || binding.loweredPlan.size()%2
            || !checkedAdd(result.textBytes,binding.loweredPlan.size(),result.textBytes)) return false;
        previous=binding.object.value;
    }
    for (const auto& binding : bindings) {
        std::size_t operations=0;
        if (!timelinePlanOperationCount(binding.loweredPlan,operations)
            || !checkedAdd(result.operations,operations,result.operations)) return false;
    }
    return deriveTimelineFamilyCodecEnvelope(bindings.size(),result.textBytes,result.operations,outerWrappers,wholeBytes,output);
}

// One read of a request runs valid at every envelope and again in the canonical encode, and each of
// those holds the same plans over the same scene. Inside a scope a plan's value and a scene binding's
// result are kept, so the checks that replay the mesh run once. The cheap checks still run each time.
struct TimelineValidationMemo final
{
    using Binding = videowire::geometry::TimelineSceneBinding;
    struct Checked final
    {
        bool family = false;
        std::vector<Binding> bindings;
        // Held so a freed scene's address can't name a different one.
        std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
    };
    struct Value final
    {
        Binding binding;
        std::shared_ptr<const videowire::geometry::AdmittedValue> value;
    };
    std::vector<Checked> checked;
    std::vector<Value> values;
};
inline thread_local TimelineValidationMemo* timelineValidationMemo = nullptr;
// How many mesh replays validation ran on this thread.
inline thread_local std::size_t timelineValidationRuns = 0;

// The timeline binding whose object's mesh the scene wire leaves out (scene
// extension 21). It is set for the nested request while a V14 wrapper is encoded
// or read. Only a Geometry3D plan rebuilds a mesh, so other bindings set none.
inline thread_local const videowire::geometry::TimelineSceneBinding* replayedTimelineMesh = nullptr;
class ScopedReplayedTimelineMesh final
{
public:
    explicit ScopedReplayedTimelineMesh(const videowire::geometry::TimelineSceneBinding& binding)
        : previous(replayedTimelineMesh)
    {
        replayedTimelineMesh = videowire::geometry::rebuildsTimelineSceneMesh(binding) ? &binding : nullptr;
    }
    ~ScopedReplayedTimelineMesh() { replayedTimelineMesh = previous; }
    ScopedReplayedTimelineMesh(const ScopedReplayedTimelineMesh&) = delete;
    ScopedReplayedTimelineMesh& operator=(const ScopedReplayedTimelineMesh&) = delete;

private:
    const videowire::geometry::TimelineSceneBinding* previous;
};

inline bool sameTimelineBinding(const videowire::geometry::TimelineSceneBinding& left,
                                const videowire::geometry::TimelineSceneBinding& right) noexcept
{
    return left.object == right.object && left.loweredPlan == right.loweredPlan;
}

template <typename Check>
inline bool timelineCheckedOnce(bool family, const std::vector<videowire::geometry::TimelineSceneBinding>& bindings,
                                const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene, Check&& check)
{
    auto* memo = timelineValidationMemo;
    if (memo != nullptr)
        for (const auto& kept : memo->checked)
            if (kept.family == family && kept.scene == scene
                && std::equal(kept.bindings.begin(), kept.bindings.end(), bindings.begin(), bindings.end(), sameTimelineBinding))
                return true;
    ++timelineValidationRuns;
    if (!check()) return false;
    if (memo != nullptr) memo->checked.push_back({family, bindings, scene});
    return true;
}

inline void keepTimelineValue(const videowire::geometry::TimelineSceneBinding& binding,
                              videowire::geometry::AdmittedValue&& value)
{
    auto* memo = timelineValidationMemo;
    if (memo == nullptr) return;
    for (const auto& kept : memo->values)
        if (sameTimelineBinding(kept.binding, binding)) return;
    memo->values.push_back({binding, std::make_shared<const videowire::geometry::AdmittedValue>(std::move(value))});
}

inline std::shared_ptr<const videowire::geometry::AdmittedValue> keptTimelineValue(
    const videowire::geometry::TimelineSceneBinding& binding)
{
    if (const auto* memo = timelineValidationMemo)
        for (const auto& kept : memo->values)
            if (sameTimelineBinding(kept.binding, binding)) return kept.value;
    return nullptr;
}

// A scene whose wire left a timeline mesh out is rebuilt by replaying the binding's plan, which
// takes seconds for a large generated mesh, and every reader of one plan payload decodes it again:
// resource admission, lowering, the viewport's resolve and the first render. The rebuild is a pure
// function of the scene text and the binding, so the filled scene and its value are kept against
// those exact bytes, under the same seeding rule as sceneTokenMemo. A kept scene is const.
struct FilledTimelineScene final
{
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
    std::shared_ptr<const videowire::geometry::AdmittedValue> value;
};
// A project with more left-out meshes than this rebuilds the older ones again. Nothing is refused.
inline constexpr std::size_t kFilledTimelineSceneEntries = 2;

inline ExactTokenMemo<FilledTimelineScene>& filledTimelineSceneMemo()
{
    // Never destroyed: helper threads may still read while the process exits.
    static auto* memo = new ExactTokenMemo<FilledTimelineScene> (kFilledTimelineSceneEntries);
    return *memo;
}

inline std::string filledTimelineSceneKey(std::string_view sceneHex,
                                          const videowire::geometry::TimelineSceneBinding& binding)
{
    std::string key;
    key.reserve(sceneHex.size() + binding.loweredPlan.size() + 16);
    key.append(sceneHex).append(" ").append(std::to_string(binding.object.value)).append(" ");
    key.append(binding.loweredPlan);
    return key;
}

inline void keepTimelineValue(const videowire::geometry::TimelineSceneBinding& binding,
                              std::shared_ptr<const videowire::geometry::AdmittedValue> value)
{
    auto* memo = timelineValidationMemo;
    if (memo == nullptr) return;
    for (const auto& kept : memo->values)
        if (sameTimelineBinding(kept.binding, binding)) return;
    memo->values.push_back({binding, std::move(value)});
}

// The decoder rebuilt this scene's left-out mesh from the binding's value, or found the scene
// kept under these exact scene and binding bytes, so the mesh is the plan's by construction and
// the scope's check of the pair appends nothing. The object must still own its ranges.
inline void keepFilledTimelineCheck(const videowire::geometry::TimelineSceneBinding& binding,
                                    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
                                    const videowire::geometry::AdmittedValue& value)
{
    auto* memo = timelineValidationMemo;
    if (memo == nullptr || !scene || !videowire::geometry::validFilledTimelineSceneBinding(binding, *scene, value))
        return;
    memo->checked.push_back({ false, { binding }, scene });
}

inline bool validTimelineFamily(const std::vector<videowire::geometry::TimelineSceneBinding>& bindings,
                               const HarmonicMIDI::grid::Visual3DScene& scene)
{
    using namespace videowire::geometry;
    TimelineFamilyCodecEnvelope envelope;
    // Derive the largest supported outer chain before entering any decoder or
    // binding validator; the admission family's resource recipe is separate.
    // Nothing is encoded yet, so no whole-wire bytes count. The wire codecs
    // derive canonicalByteVisits from the size they read.
    if (!timelineFamilyEnvelope(bindings,4,0,envelope)) return false;
    ResourceLimits remaining;
    std::string error;
    try {
        for (const auto& binding : bindings) {
            std::size_t fields=0;
            auto value=decodeTimelineScenePlan(binding,remaining,error,&fields);
            if (!value || !validTimelineSceneBinding(binding,scene,*value,error)) return false;
            const auto& mesh=std::get<GeometryData>(value->descriptor().data);
            if (mesh.positions.size()>remaining.maxVertices || mesh.indices.size()>remaining.maxIndices
                || fields>remaining.maxFieldElements || value->descriptor().operations.size()>remaining.maxOperations) return false;
            remaining.maxVertices-=mesh.positions.size(); remaining.maxIndices-=mesh.indices.size();
            remaining.maxFieldElements-=fields; remaining.maxOperations-=value->descriptor().operations.size();
            keepTimelineValue(binding,std::move(*value));
        }
        return true;
    } catch (const std::bad_alloc&) { return false; }
}

// The added V21 Request copy also copies existing material vectors and their
// strings. Count those value-owned bytes separately; scene/imported pointers
// retain their owners and do not deep-copy the scene or leaf here.
inline bool timelineFamilyPayloadCopyBytes(const Request& request,std::size_t& output)
{
    using namespace videowire::geometry;
    // A multi-leaf composition never carries a Timeline Geometry3D family; its
    // V26 wrapper is not one of the family's counted outer wrappers.
    if (!request.additionalImportedScenes.empty()) return false;
    if (request.deformation || (request.camera && !(request.importedScene && request.importedScene->sourceCameraNodeId))
        || request.light || request.noteLight
        || request.materialField || !request.importedLights.empty()) return false;
    std::size_t bytes=kRequestMemberBytes,strings=0;
    const auto string=[&](const std::string& value) {
        return checkedAdd(strings,value.size(),strings) && strings<=kMaximumV8EncodedBytes
            && checkedAdd(bytes,value.size()+1,bytes);
    };
    const auto material=[&](const surfacematerialbinding::ImportedSceneMaterialRequest& value) {
        if (value.program.operations.size()>surfacematerial::kMaximumOperations
            || value.binding.textures.size()>surfacematerialbinding::kMaximumTextureBindings
            || !string(value.binding.surfaceMaterialDigest)
            || !addArrayBytes(value.program.operations.size(),sizeof(surfacematerial::Operation),bytes)
            || !addArrayBytes(value.binding.textures.size(),sizeof(surfacematerialbinding::TextureSlotBinding),bytes)) return false;
        return !value.vertexModifier || addArrayBytes(value.vertexModifier->records.size(),sizeof(videowire::VertexModifierRecord),bytes);
    };
    if (!string(request.asset.id) || !string(request.asset.contentSha256) || !string(request.asset.sourceMediaType)
        || (request.material && !material(*request.material))
        || (request.diffractionMaterial && !string(request.diffractionMaterial->structuralDigest))
        || !addArrayBytes(request.surfacePrograms.size(),sizeof(geometrysurfacematerial::ObjectProgram),bytes)) return false;
    for (const auto& object : request.surfacePrograms) {
        if (object.program.fields.size()>surfacematerialfield::kMaximumBindings || !material(object.program.material)
            || !addArrayBytes(object.program.fields.size(),sizeof(surfacematerialfield::Binding),bytes)) return false;
        for (const auto& field : object.program.fields)
            if (!string(field.sample.fieldPlan)) return false;
    }
    output=bytes; return true;
}

// V21 wire preflight uses bounded string views, before stream or plan copies.
// It returns all record views only after count, aggregate lengths and operation
// census succeed. No failed parse modifies a Request.
struct TimelineFamilyWire final {
    struct Child final { HarmonicMIDI::grid::SceneObjectId object; std::string_view plan; };
    std::vector<Child> children;
    std::size_t count=0;
    std::string_view nested;
    TimelineFamilyCodecEnvelope envelope;
};
inline bool readFamilyUnsigned(std::string_view& wire,std::uint64_t& value,char terminator)
{
    value=0; std::size_t digits=0;
    while (digits<wire.size() && wire[digits]>='0' && wire[digits]<='9') {
        const auto digit=static_cast<unsigned>(wire[digits]-'0');
        if (value>(std::numeric_limits<std::uint64_t>::max()-digit)/10) return false;
        value=value*10+digit; ++digits;
    }
    if (!digits || (digits>1 && wire[0]=='0') || digits>=wire.size() || wire[digits]!=terminator) return false;
    wire.remove_prefix(digits+1); return true;
}
inline bool containsRequestHeader(std::string_view wire,std::string_view header)
{
    for (std::size_t offset=wire.find(header);offset!=std::string_view::npos;offset=wire.find(header,offset+1))
        if ((offset==0 || wire[offset-1]=='\n') && wire.size()-offset>header.size()
            && wire[offset+header.size()]=='\n') return true;
    return false;
}
inline bool preflightTimelineFamilyWire(std::string_view wire,TimelineFamilyWire& output)
{
    using namespace videowire::geometry;
    const auto wholeBytes=wire.size();
    if (wire.size()<=kCompiledOperationHeaderV21.size()
        || wire.substr(0,kCompiledOperationHeaderV21.size())!=kCompiledOperationHeaderV21
        || wire[kCompiledOperationHeaderV21.size()]!='\n') return false;
    wire.remove_prefix(kCompiledOperationHeaderV21.size()+1);
    std::uint64_t count=0,nestedBytes=0;
    if (!readFamilyUnsigned(wire,count,' ') || count<2
        || !readFamilyUnsigned(wire,nestedBytes,'\n') || !nestedBytes || nestedBytes>kMaximumV8EncodedBytes) return false;
    TimelineFamilyWire result; result.count=static_cast<std::size_t>(count);
    std::size_t text=0,operations=0; std::uint64_t previous=0;
    for (std::size_t i=0;i<result.count;++i) {
        std::uint64_t object=0,bytes=0;
        if (!readFamilyUnsigned(wire,object,' ') || object<=previous || object>std::numeric_limits<std::uint32_t>::max()
            || !readFamilyUnsigned(wire,bytes,'\n') || !bytes
            || bytes>=wire.size() || wire[static_cast<std::size_t>(bytes)]!='\n'
            || !checkedAdd(text,static_cast<std::size_t>(bytes),text)) return false;
        const auto plan=wire.substr(0,static_cast<std::size_t>(bytes));
        result.children.push_back({{static_cast<std::uint32_t>(object)},plan}); previous=object;
        wire.remove_prefix(static_cast<std::size_t>(bytes)+1);
    }
    if (nestedBytes!=wire.size() || containsRequestHeader(wire,kCompiledOperationHeaderV21)
        || containsRequestHeader(wire,kCompiledOperationHeaderV14)
        || containsRequestHeader(wire,kCompiledOperationHeaderV20)) return false;
    for (std::size_t i=0;i<result.count;++i) {
        std::size_t childOperations=0;
        if (!timelinePlanOperationCount(result.children[i].plan,childOperations)
            || !checkedAdd(operations,childOperations,operations)) return false;
    }
    if (!deriveTimelineFamilyCodecEnvelope(result.count,text,operations,0,
            wholeBytes,result.envelope)) return false;
    result.nested=wire; output=result; return true;
}

// Family-bearing input must use the canonical outer ordering before the old
// recursive codec runs. Legacy input has exactly its original parsing behavior.
inline bool preflightFamilyOuterWire(std::string_view encoded)
{
    if (!containsRequestHeader(encoded,kCompiledOperationHeaderV21)) return true;
    auto wire=encoded;
    std::size_t wrappers=0;
    if (wire.substr(0,kCompiledOperationHeaderV22.size())==kCompiledOperationHeaderV22) {
        wire.remove_prefix(kCompiledOperationHeaderV22.size());
        constexpr std::string_view prefix="\noverride ";
        if (wire.substr(0,prefix.size())!=prefix) return false;
        wire.remove_prefix(prefix.size());
        std::uint64_t bytes=0;
        if (!readFamilyUnsigned(wire,bytes,'\n')) return false;
        const auto recordEnd=wire.find('\n');
        if (recordEnd==std::string_view::npos) return false;
        wire.remove_prefix(recordEnd+1);
        if (bytes!=wire.size()) return false;
        ++wrappers;
    }
    for (const auto header : {kCompiledOperationHeaderV20,kCompiledOperationHeaderV19,
                              kCompiledOperationHeaderV17,kCompiledOperationHeaderV15}) {
        if (wire.substr(0,header.size())!=header) continue;
        if (wire.size()<=header.size() || wire[header.size()]!='\n') return false;
        wire.remove_prefix(header.size()+1);
        ++wrappers;
        if (header==kCompiledOperationHeaderV20) {
            std::uint64_t scene=0,leaf=0;
            if (!readFamilyUnsigned(wire,scene,' ') || !readFamilyUnsigned(wire,leaf,'\n')
                || !scene || !leaf || scene>wire.size() || leaf!=wire.size()-scene
                || containsRequestHeader(wire.substr(static_cast<std::size_t>(scene)),kCompiledOperationHeaderV21)) return false;
            wire=wire.substr(0,static_cast<std::size_t>(scene));
        } else {
            const auto end=wire.find('\n');
            if (end==std::string_view::npos) return false;
            wire.remove_prefix(end+1);
        }
    }
    TimelineFamilyWire family;
    return preflightTimelineFamilyWire(wire,family)
        && deriveTimelineFamilyCodecEnvelope(family.count,family.envelope.textBytes,family.envelope.operations,
            wrappers,encoded.size(),family.envelope);
}

static_assert(sizeof(float) == sizeof(std::uint32_t)
                  && std::numeric_limits<float>::is_iec559,
              "Imported scene material transport requires IEEE-754 binary32 floats");
static_assert(sizeof(double) == sizeof(std::uint64_t)
                  && std::numeric_limits<double>::is_iec559,
              "Imported scene vertex transport requires IEEE-754 binary64 values");

template <typename Value>
inline bool readUnsigned(std::istream& input, Value& output)
{
    static_assert(std::is_unsigned_v<Value>);
    std::uint64_t value = 0;
    if (!(input >> value)
        || value > static_cast<std::uint64_t>(std::numeric_limits<Value>::max()))
        return false;
    output = static_cast<Value>(value);
    return true;
}

inline std::uint32_t floatBits(float value) noexcept
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline float floatFromBits(std::uint32_t bits) noexcept
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// The V37 scene fog as wire numbers: density, scatter colour, absorption colour,
// anisotropy, emission, field scale and offset and radius as float bits, then
// the field and its detail. volumeOnly is never on the wire: fog has a surface
// to cross nowhere.
inline constexpr std::size_t kWorldVolumeWireFields = 18;
inline std::array<std::uint64_t, kWorldVolumeWireFields> worldVolumeWireFields(
    const HarmonicMIDI::grid::SceneVolume& volume) noexcept
{
    return { floatBits(volume.density),
             floatBits(volume.color.x), floatBits(volume.color.y), floatBits(volume.color.z),
             floatBits(volume.absorptionColor.x), floatBits(volume.absorptionColor.y),
             floatBits(volume.absorptionColor.z), floatBits(volume.anisotropy),
             floatBits(volume.emission.x), floatBits(volume.emission.y), floatBits(volume.emission.z),
             floatBits(volume.fieldScale),
             floatBits(volume.fieldOffset.x), floatBits(volume.fieldOffset.y), floatBits(volume.fieldOffset.z),
             floatBits(volume.radius),
             static_cast<std::uint64_t>(volume.field), volume.fieldDetail };
}

inline bool worldVolumeFromWireFields(const std::array<std::uint64_t, kWorldVolumeWireFields>& fields,
                                      HarmonicMIDI::grid::SceneVolume& volume) noexcept
{
    for (std::size_t index = 0; index < 16; ++index)
        if (fields[index] > 0xffffffffu) return false;
    if (fields[16] > static_cast<std::uint64_t>(HarmonicMIDI::grid::SceneVolumeField::Noise)
        || fields[17] > 0xffffffffu) return false;
    const auto f = [&](std::size_t index) { return floatFromBits(static_cast<std::uint32_t>(fields[index])); };
    volume = {};
    volume.density = f(0);
    volume.color = { f(1), f(2), f(3) };
    volume.absorptionColor = { f(4), f(5), f(6) };
    volume.anisotropy = f(7);
    volume.emission = { f(8), f(9), f(10) };
    volume.fieldScale = f(11);
    volume.fieldOffset = { f(12), f(13), f(14) };
    volume.radius = f(15);
    volume.field = static_cast<HarmonicMIDI::grid::SceneVolumeField>(fields[16]);
    volume.fieldDetail = static_cast<std::uint32_t>(fields[17]);
    return true;
}

inline std::uint64_t doubleBits(double value) noexcept
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline double doubleFromBits(std::uint64_t bits) noexcept
{
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline bool sameVertexModifier(const videowire::VertexModifierIr& left,
                               const videowire::VertexModifierIr& right) noexcept
{
    if (left.schemaVersion != right.schemaVersion || left.rootId != right.rootId
        || left.records.size() != right.records.size())
        return false;
    for (std::size_t index = 0; index < left.records.size(); ++index)
    {
        const auto& a = left.records[index];
        const auto& b = right.records[index];
        if (a.stableId != b.stableId || a.operation != b.operation
            || a.resultType != b.resultType || a.inputs != b.inputs
            || a.inputCount != b.inputCount || a.parameterCount != b.parameterCount)
            return false;
        for (std::size_t parameter = 0; parameter < a.parameters.size(); ++parameter)
            if (doubleBits(a.parameters[parameter]) != doubleBits(b.parameters[parameter]))
                return false;
    }
    return true;
}

inline bool validVertexModifier(const videowire::VertexModifierIr& vertex) noexcept
{
    if (vertex.schemaVersion != videowire::kVertexModifierIrSchemaVersion
        || vertex.rootId == 0 || vertex.records.empty())
        return false;
    std::map<videowire::VertexModifierStableId, std::size_t> indices;
    for (std::size_t index = 0; index < vertex.records.size(); ++index)
        if (!indices.emplace(vertex.records[index].stableId, index).second) return false;
    for (std::size_t index = 0; index < vertex.records.size(); ++index)
    {
        const auto& record = vertex.records[index];
        const auto schema = videowire::vertexModifierOperationSchema(record.operation);
        if (record.stableId == 0 || !schema.valid || record.resultType != schema.resultType
            || record.inputCount != schema.inputCount
            || !videowire::vertexModifierParameterCountAdmitted(record.operation, record.parameterCount))
            return false;
        if (schema.terminal != (record.stableId == vertex.rootId)) return false;
        for (std::size_t input = 0; input < record.inputs.size(); ++input)
        {
            if (input < record.inputCount)
            {
                if (record.inputs[input] == 0 || record.inputs[input] == record.stableId
                    || indices.count(record.inputs[input]) == 0)
                    return false;
            }
            else if (record.inputs[input] != 0)
                return false;
        }
        for (std::size_t parameter = 0; parameter < record.parameters.size(); ++parameter)
        {
            if (parameter < record.parameterCount)
            {
                if (!std::isfinite(record.parameters[parameter])) return false;
            }
            else if (record.parameters[parameter] != 0.0)
                return false;
        }
    }
    const auto root = indices.find(vertex.rootId);
    if (root == indices.end()) return false;
    std::vector<std::uint8_t> colors(vertex.records.size(), 0);
    std::vector<std::pair<std::size_t, std::size_t>> pending;
    std::size_t visited = 0;
    colors[root->second] = 1;
    pending.emplace_back(root->second, 0);
    while (!pending.empty())
    {
        const auto& record = vertex.records[pending.back().first];
        if (pending.back().second == record.inputCount)
        {
            colors[pending.back().first] = 2;
            ++visited;
            pending.pop_back();
            continue;
        }
        const auto child = indices.at(record.inputs[pending.back().second++]);
        if (colors[child] == 1) return false;
        if (colors[child] == 0)
        {
            colors[child] = 1;
            pending.emplace_back(child, 0);
        }
    }
    return visited == vertex.records.size();
}

inline bool knownValueType(surfacematerial::ValueType type) noexcept
{
    const auto value = static_cast<std::uint8_t>(type);
    return value >= static_cast<std::uint8_t>(surfacematerial::ValueType::Scalar)
        && value <= static_cast<std::uint8_t>(surfacematerial::ValueType::UInt);
}

inline bool knownOperationKind(surfacematerial::OperationKind kind) noexcept
{
    const auto value = static_cast<std::uint8_t>(kind);
    return value >= static_cast<std::uint8_t>(
               surfacematerial::OperationKind::FloatConstant)
        && value <= static_cast<std::uint8_t>(
               surfacematerial::OperationKind::ComposeVec4);
}

inline bool knownInputSemantic(surfacematerial::InputSemantic semantic) noexcept
{
    return static_cast<std::uint8_t>(semantic)
        <= static_cast<std::uint8_t>(surfacematerial::InputSemantic::Control);
}

inline bool knownBindingTarget(
    surfacematerialbinding::BindingTargetKind target) noexcept
{
    return target == surfacematerialbinding::BindingTargetKind::MaterialSlot
        || target == surfacematerialbinding::BindingTargetKind::ObjectOverride;
}

inline bool knownTextureSource(
    surfacematerialbinding::TextureSourceKind source) noexcept
{
    return source == surfacematerialbinding::TextureSourceKind::ImportedBaseColor
        || source == surfacematerialbinding::TextureSourceKind::VideoResource
        || source == surfacematerialbinding::TextureSourceKind::GraphFrame;
}

inline bool sameProgram(const surfacematerial::ProgramDescription& left,
                        const surfacematerial::ProgramDescription& right) noexcept
{
    if (left.version != right.version
        || left.textureSlotCount != right.textureSlotCount
        || left.outputs != right.outputs
        || left.transport != right.transport
        || left.spectralReflectance.has_value() != right.spectralReflectance.has_value()
        || left.diffractionLayer.has_value() != right.diffractionLayer.has_value()
        || left.operations.size() != right.operations.size())
        return false;
    if (left.diffractionLayer && !surfacematerial::sameDiffractionLayer(*left.diffractionLayer, *right.diffractionLayer))
        return false;
    if (left.spectralReflectance)
    {
        if (left.spectralReflectance->version != right.spectralReflectance->version) return false;
        for (std::size_t band = 0; band < left.spectralReflectance->coefficients.size(); ++band)
            if (floatBits(left.spectralReflectance->coefficients[band])
                != floatBits(right.spectralReflectance->coefficients[band])) return false;
    }
    for (std::size_t index = 0; index < left.operations.size(); ++index)
    {
        const auto& a = left.operations[index];
        const auto& b = right.operations[index];
        if (a.id != b.id || a.kind != b.kind || a.resultType != b.resultType
            || a.inputCount != b.inputCount || a.inputs != b.inputs
            || a.semantic != b.semantic || a.parameter != b.parameter
            || a.unsignedLiteral != b.unsignedLiteral)
            return false;
        for (std::size_t literal = 0; literal < a.literal.size(); ++literal)
            if (floatBits(a.literal[literal]) != floatBits(b.literal[literal]))
                return false;
    }
    return true;
}

inline bool sameBinding(const surfacematerialbinding::MaterialBinding& left,
                        const surfacematerialbinding::MaterialBinding& right) noexcept
{
    if (left.targetKind != right.targetKind
        || left.materialSlot != right.materialSlot || left.object != right.object
        || left.surfaceMaterialDigest != right.surfaceMaterialDigest
        || left.surfaceMaterialRevision != right.surfaceMaterialRevision
        || left.textures.size() != right.textures.size())
        return false;
    for (std::size_t index = 0; index < left.textures.size(); ++index)
    {
        const auto& a = left.textures[index];
        const auto& b = right.textures[index];
        if (a.slot != b.slot || a.source != b.source
            || a.videoResource != b.videoResource
            || a.graphFrame.has_value() != b.graphFrame.has_value()
            || (a.graphFrame && (a.graphFrame->node != b.graphFrame->node
                                || a.graphFrame->port != b.graphFrame->port)))
            return false;
    }
    return true;
}

inline bool sameMaterial(
    const surfacematerialbinding::ImportedSceneMaterialRequest& left,
    const surfacematerialbinding::ImportedSceneMaterialRequest& right) noexcept
{
    return left.version == right.version && left.roughGlassScatteringVersion==right.roughGlassScatteringVersion && left.scene == right.scene
        && left.sceneRevision == right.sceneRevision
        && left.structuralRevision == right.structuralRevision
        && left.evaluationRevision == right.evaluationRevision
        && left.programRevision == right.programRevision
        && left.vertexModifier.has_value() == right.vertexModifier.has_value()
        && sameProgram(left.program, right.program)
        && (!left.vertexModifier
            || sameVertexModifier(*left.vertexModifier, *right.vertexModifier))
        && sameBinding(left.binding, right.binding);
}

inline bool validMaterial(
    const surfacematerialbinding::ImportedSceneMaterialRequest& material) noexcept
{
    const bool hasFrame = surfacematerialbinding::hasGraphFrameInput(material.binding);
    if (material.version != surfacematerialbinding::requestVersion(
            hasFrame, material.program.spectralReflectance.has_value(),material.roughGlassScatteringVersion,
            material.program.diffractionLayer.has_value())
        || !surfacematerialbinding::validRoughGlassVersion(material)
        || (material.program.spectralReflectance
            && !surfacematerial::validSpectralReflectance(*material.program.spectralReflectance))
        || (material.program.diffractionLayer
            && !surfacematerial::validDiffractionLayer(*material.program.diffractionLayer))
        || !material.scene.isValid() || material.sceneRevision == 0
        || material.structuralRevision == 0 || material.evaluationRevision == 0
        || material.programRevision == 0
        || surfacematerial::outputCount(material.program.version) == 0
        || (material.program.transport != surfacematerial::TransportMode::ThinSurface
            && material.program.transport != surfacematerial::TransportMode::ClosedVolume
            && material.program.transport != surfacematerial::TransportMode::ThinInterface)
        || (material.program.version == surfacematerial::kWireVersion
            && (material.program.outputs[10] != 0
                || material.program.transport != surfacematerial::TransportMode::ThinSurface))
        || material.program.textureSlotCount > surfacematerial::kMaximumTextureSlots
        || material.program.operations.size() > surfacematerial::kMaximumOperations
        || !knownBindingTarget(material.binding.targetKind)
        || !visualmodelasset::validSha256(
            material.binding.surfaceMaterialDigest)
        || material.binding.surfaceMaterialRevision == 0
        || (material.vertexModifier && !validVertexModifier(*material.vertexModifier))
        || material.binding.textures.size()
            > surfacematerialbinding::kMaximumTextureBindings)
        return false;
    for (const auto& operation : material.program.operations)
        if (!knownOperationKind(operation.kind)
            || !knownValueType(operation.resultType)
            || operation.inputCount > surfacematerial::kMaximumOperationInputs
            || !knownInputSemantic(operation.semantic))
            return false;
    for (const auto& texture : material.binding.textures)
        if (!knownTextureSource(texture.source)
            || (texture.source == surfacematerialbinding::TextureSourceKind::GraphFrame
                ? (!texture.graphFrame || texture.graphFrame->node < 0 || texture.graphFrame->port < 0
                    || surfacematerialbinding::hasIdentity(texture.videoResource)
                    || texture.slot != 0 || material.binding.textures.size() != 1
                    || material.program.textureSlotCount != 1)
                : texture.graphFrame.has_value()))
            return false;
    return true;
}

template <std::size_t Size>
inline bool sameFloatArray(const std::array<float, Size>& left,
                           const std::array<float, Size>& right) noexcept
{
    for (std::size_t index = 0; index < Size; ++index)
        if (floatBits(left[index]) != floatBits(right[index])) return false;
    return true;
}

inline bool sameDiffractionDescription(
    const diffractionmaterial::Description& left,
    const diffractionmaterial::Description& right) noexcept
{
    return left.version == right.version
        && sameFloatArray(left.geometry.directionUv, right.geometry.directionUv)
        && floatBits(left.geometry.grooveSpacingNanometres)
            == floatBits(right.geometry.grooveSpacingNanometres)
        && left.geometry.lattice == right.geometry.lattice
        && left.geometry.thirdGrooveFamily == right.geometry.thirdGrooveFamily
        && sameFloatArray(left.geometry.secondaryDirectionUv,
                          right.geometry.secondaryDirectionUv)
        && floatBits(left.geometry.secondaryGrooveSpacingNanometres)
            == floatBits(right.geometry.secondaryGrooveSpacingNanometres)
        && left.grooveField.mode == right.grooveField.mode
        && sameFloatArray(left.grooveField.originUv, right.grooveField.originUv)
        && sameFloatArray(left.grooveField.axisUv, right.grooveField.axisUv)
        && floatBits(left.grooveField.grooveSpacingDeltaNanometresPerUnit)
            == floatBits(right.grooveField.grooveSpacingDeltaNanometresPerUnit)
        && floatBits(left.grooveField.secondarySpacingDeltaNanometresPerUnit)
            == floatBits(right.grooveField.secondarySpacingDeltaNanometresPerUnit)
        && floatBits(left.grooveField.orientationDegreesPerUnit)
            == floatBits(right.grooveField.orientationDegreesPerUnit)
        && left.microstructure.profile == right.microstructure.profile
        && floatBits(left.microstructure.grooveDepthNanometres)
            == floatBits(right.microstructure.grooveDepthNanometres)
        && floatBits(left.microstructure.dutyCycle)
            == floatBits(right.microstructure.dutyCycle)
        && floatBits(left.microstructure.blazeAngleDegrees)
            == floatBits(right.microstructure.blazeAngleDegrees)
        && floatBits(left.substrate.refractiveIndex)
            == floatBits(right.substrate.refractiveIndex)
        && floatBits(left.substrate.extinctionCoefficient)
            == floatBits(right.substrate.extinctionCoefficient)
        && left.substrateData == right.substrateData
        && left.orderMedium == right.orderMedium
        && left.pitStatistics == right.pitStatistics
        && left.coating.model == right.coating.model
        && floatBits(left.coating.thicknessNanometres)
            == floatBits(right.coating.thicknessNanometres)
        && floatBits(left.coating.opticalConstants.refractiveIndex)
            == floatBits(right.coating.opticalConstants.refractiveIndex)
        && floatBits(left.coating.opticalConstants.extinctionCoefficient)
            == floatBits(right.coating.opticalConstants.extinctionCoefficient)
        && floatBits(left.roughness.rmsHeightNanometres)
            == floatBits(right.roughness.rmsHeightNanometres)
        && floatBits(left.roughness.rmsSlope) == floatBits(right.roughness.rmsSlope)
        && floatBits(left.roughness.acrossGrooveSlope) == floatBits(right.roughness.acrossGrooveSlope)
        && floatBits(left.roughness.streakSlope) == floatBits(right.roughness.streakSlope)
        && floatBits(left.roughness.streakSpacingUv) == floatBits(right.roughness.streakSpacingUv)
        && floatBits(left.roughness.reliefRmsHeightNanometres) == floatBits(right.roughness.reliefRmsHeightNanometres)
        && floatBits(left.roughness.reliefMinimumSpacingNanometres) == floatBits(right.roughness.reliefMinimumSpacingNanometres)
        && left.spectrum.integrationModel == right.spectrum.integrationModel
        && left.spectrum.wavelengthCount == right.spectrum.wavelengthCount
        && sameFloatArray(left.spectrum.wavelengthsNanometres,
                          right.spectrum.wavelengthsNanometres)
        && left.spectrum.firstOrder == right.spectrum.firstOrder
        && left.spectrum.lastOrder == right.spectrum.lastOrder;
}

inline constexpr std::uint32_t kLegacyV6MaterialWireVersion = 4;
inline constexpr std::uint32_t kLegacyV6BindingWireVersion = 1;

// Frozen validation for the fieldless material schema carried by V6 operations.
// Keep this independent from V7 so later canonical rules cannot reinterpret old bytes.
inline bool validLegacyV6DiffractionDescription(
    const diffractionmaterial::Description& material) noexcept
{
    using namespace diffractionmaterial;
    const auto canonical = [](float value)
    {
        return std::isfinite(value) && (value != 0.0f || !std::signbit(value));
    };
    const auto bounded = [&canonical](float value, float minimum, float maximum)
    {
        return canonical(value) && value >= minimum && value <= maximum;
    };
    const auto validOptical = [&bounded](const OpticalConstants& value)
    {
        return bounded(value.refractiveIndex, kMinimumSubstrateRefractiveIndex,
                       kMaximumSubstrateRefractiveIndex)
            && bounded(value.extinctionCoefficient, 0.0f,
                       kMaximumExtinctionCoefficient);
    };
    if (material.version != kLegacyV6MaterialWireVersion
        || !canonical(material.geometry.directionUv[0])
        || !canonical(material.geometry.directionUv[1])
        || std::abs(material.geometry.directionUv[0] * material.geometry.directionUv[0]
                    + material.geometry.directionUv[1] * material.geometry.directionUv[1]
                    - 1.0f) > 0.0001f
        || !bounded(material.geometry.grooveSpacingNanometres,
                    kMinimumGrooveSpacingNanometres, kMaximumGrooveSpacingNanometres))
        return false;
    if (material.geometry.thirdGrooveFamily) return false;
    if (material.geometry.lattice == GratingLattice::OneDimensional)
    {
        if (!canonical(material.geometry.secondaryDirectionUv[0])
            || !canonical(material.geometry.secondaryDirectionUv[1])
            || !canonical(material.geometry.secondaryGrooveSpacingNanometres)
            || material.geometry.secondaryDirectionUv != std::array<float, 2> {}
            || material.geometry.secondaryGrooveSpacingNanometres != 0.0f)
            return false;
    }
    else if (material.geometry.lattice == GratingLattice::CrossedTwoDimensional)
    {
        const auto& primary = material.geometry.directionUv;
        const auto& secondary = material.geometry.secondaryDirectionUv;
        if (!canonical(secondary[0]) || !canonical(secondary[1])
            || std::abs(secondary[0] * secondary[0]
                        + secondary[1] * secondary[1] - 1.0f) > 0.0001f
            || std::abs(primary[0] * secondary[0]
                        + primary[1] * secondary[1]) > 0.9999f
            || !bounded(material.geometry.secondaryGrooveSpacingNanometres,
                        kMinimumGrooveSpacingNanometres,
                        kMaximumGrooveSpacingNanometres))
            return false;
    }
    else
        return false;

    const auto& micro = material.microstructure;
    if ((micro.profile != GrooveProfile::BinaryRectangular
         && micro.profile != GrooveProfile::Sinusoidal
         && micro.profile != GrooveProfile::BlazedSawtooth)
        || !bounded(micro.grooveDepthNanometres,
                    kMinimumGrooveDepthNanometres, kMaximumGrooveDepthNanometres)
        || !bounded(micro.dutyCycle, 0.01f, 1.0f)
        || !canonical(micro.blazeAngleDegrees))
        return false;
    if (micro.profile == GrooveProfile::Sinusoidal
        && std::abs(micro.dutyCycle - 0.5f) > 1.0e-6f)
        return false;
    if (micro.profile != GrooveProfile::BlazedSawtooth)
    {
        if (micro.blazeAngleDegrees != 0.0f) return false;
    }
    else
    {
        constexpr double radiansToDegrees = 57.2957795130823208768;
        const auto expected = std::atan(
            static_cast<double>(micro.grooveDepthNanometres)
            / (static_cast<double>(micro.dutyCycle)
               * material.geometry.grooveSpacingNanometres)) * radiansToDegrees;
        if (!(micro.blazeAngleDegrees > 0.0f)
            || micro.blazeAngleDegrees >= 89.0f
            || std::abs(static_cast<double>(micro.blazeAngleDegrees) - expected) > 1.0e-3)
            return false;
    }
    if (!validOptical(material.substrate)) return false;
    if (material.substrateData != diffractionmaterial::SubstrateData::AuthoredConstants) return false;
    if (material.orderMedium != diffractionmaterial::OrderMedium::Air) return false;
    if (material.pitStatistics != diffractionmaterial::PitStatistics::None) return false;
    if (material.coating.model == CoatingModel::Uncoated)
    {
        if (!canonical(material.coating.thicknessNanometres)
            || !canonical(material.coating.opticalConstants.refractiveIndex)
            || !canonical(material.coating.opticalConstants.extinctionCoefficient)
            || material.coating.thicknessNanometres != 0.0f
            || material.coating.opticalConstants.refractiveIndex != 0.0f
            || material.coating.opticalConstants.extinctionCoefficient != 0.0f)
            return false;
    }
    else if (material.coating.model != CoatingModel::IncoherentDielectric
             || !bounded(material.coating.thicknessNanometres, 0.1f,
                         kMaximumCoatingThicknessNanometres)
             || !validOptical(material.coating.opticalConstants))
        return false;
    if (!bounded(material.roughness.rmsHeightNanometres, 0.0f,
                 kMaximumRmsHeightNanometres)
        || !bounded(material.roughness.rmsSlope, 0.0f, kMaximumRmsSlope)
        || !bounded(material.roughness.acrossGrooveSlope, 0.0f, kMaximumRmsSlope)
        || !bounded(material.roughness.streakSlope, 0.0f, kMaximumStreakSlope)
        || !bounded(material.roughness.streakSpacingUv, kMinimumStreakSpacingUv, kMaximumStreakSpacingUv)
        || !bounded(material.roughness.reliefRmsHeightNanometres, 0.0f, kMaximumReliefRmsHeightNanometres)
        || !bounded(material.roughness.reliefMinimumSpacingNanometres, kMinimumReliefSpacingNanometres, kMaximumReliefSpacingNanometres)
        || material.spectrum.integrationModel != SpectralIntegrationModel::Cie1931Xyz
        || material.spectrum.wavelengthCount < kMinimumSpectralSamples
        || material.spectrum.wavelengthCount > kMaximumSpectralSamples)
        return false;
    float previous = 0.0f;
    for (std::size_t index = 0;
         index < material.spectrum.wavelengthsNanometres.size(); ++index)
    {
        const auto wavelength = material.spectrum.wavelengthsNanometres[index];
        if (index >= material.spectrum.wavelengthCount)
        {
            if (wavelength != 0.0f || std::signbit(wavelength)) return false;
        }
        else if (!bounded(wavelength, kMinimumWavelengthNanometres,
                          kMaximumWavelengthNanometres)
                 || (index != 0 && wavelength <= previous))
            return false;
        previous = wavelength;
    }
    return material.spectrum.firstOrder != 0
        && material.spectrum.firstOrder <= material.spectrum.lastOrder
        && material.spectrum.lastOrder <= kMaximumDiffractionOrder;
}

inline bool validLegacyV6DiffractionMaterial(
    const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest& binding) noexcept
{
    return binding.version == kLegacyV6BindingWireVersion
        && binding.scene.isValid() && binding.object.isValid()
        && binding.sceneRevision != 0 && binding.structuralRevision != 0
        && binding.evaluationRevision != 0
        && binding.materialRevision == binding.structuralRevision
        && visualmodelasset::validSha256(binding.structuralDigest)
        && validLegacyV6DiffractionDescription(binding.material);
}

inline bool validDiffractionDescription(
    const diffractionmaterial::Description& material) noexcept
{
    using namespace diffractionmaterial;
    const auto canonical = [](float value)
    {
        return std::isfinite(value) && (value != 0.0f || !std::signbit(value));
    };
    const auto bounded = [&canonical](float value, float minimum, float maximum)
    {
        return canonical(value) && value >= minimum && value <= maximum;
    };
    const auto validOptical = [&bounded](const OpticalConstants& value)
    {
        return bounded(value.refractiveIndex, kMinimumSubstrateRefractiveIndex,
                       kMaximumSubstrateRefractiveIndex)
            && bounded(value.extinctionCoefficient, 0.0f,
                       kMaximumExtinctionCoefficient);
    };
    if (material.version != kWireVersion
        || !canonical(material.geometry.directionUv[0])
        || !canonical(material.geometry.directionUv[1])
        || std::abs(material.geometry.directionUv[0]
                        * material.geometry.directionUv[0]
                    + material.geometry.directionUv[1]
                        * material.geometry.directionUv[1]
                    - 1.0f) > 0.0001f
        || !bounded(material.geometry.grooveSpacingNanometres,
                    kMinimumGrooveSpacingNanometres,
                    kMaximumGrooveSpacingNanometres))
        return false;
    if (material.geometry.thirdGrooveFamily
        && material.geometry.lattice != GratingLattice::CrossedTwoDimensional)
        return false;
    if (material.geometry.lattice == GratingLattice::OneDimensional)
    {
        if (!canonical(material.geometry.secondaryDirectionUv[0])
            || !canonical(material.geometry.secondaryDirectionUv[1])
            || !canonical(material.geometry.secondaryGrooveSpacingNanometres)
            || material.geometry.secondaryDirectionUv != std::array<float, 2> {}
            || material.geometry.secondaryGrooveSpacingNanometres != 0.0f)
            return false;
    }
    else if (material.geometry.lattice == GratingLattice::CrossedTwoDimensional)
    {
        const auto& primary = material.geometry.directionUv;
        const auto& secondary = material.geometry.secondaryDirectionUv;
        if (!canonical(secondary[0]) || !canonical(secondary[1])
            || std::abs(secondary[0] * secondary[0]
                        + secondary[1] * secondary[1] - 1.0f) > 0.0001f
            || std::abs(primary[0] * secondary[0]
                        + primary[1] * secondary[1]) > 0.9999f
            || !bounded(material.geometry.secondaryGrooveSpacingNanometres,
                        kMinimumGrooveSpacingNanometres,
                        kMaximumGrooveSpacingNanometres))
            return false;
    }
    else
        return false;

    const auto& field = material.grooveField;
    if (field.mode == GrooveFieldMode::Constant)
    {
        if (field.originUv != std::array<float, 2> {}
            || field.axisUv != std::array<float, 2> {}
            || field.grooveSpacingDeltaNanometresPerUnit != 0.0f
            || field.secondarySpacingDeltaNanometresPerUnit != 0.0f
            || field.orientationDegreesPerUnit != 0.0f)
            return false;
    }
    else
    {
        if ((field.mode != GrooveFieldMode::Linear
             && field.mode != GrooveFieldMode::Radial)
            || !bounded(field.originUv[0], 0.0f, 1.0f)
            || !bounded(field.originUv[1], 0.0f, 1.0f)
            || !bounded(field.grooveSpacingDeltaNanometresPerUnit,
                        -kMaximumGrooveSpacingNanometres,
                        kMaximumGrooveSpacingNanometres)
            || !bounded(field.secondarySpacingDeltaNanometresPerUnit,
                        -kMaximumGrooveSpacingNanometres,
                        kMaximumGrooveSpacingNanometres)
            || !bounded(field.orientationDegreesPerUnit,
                        -kMaximumFieldOrientationDegreesPerUnit,
                        kMaximumFieldOrientationDegreesPerUnit)
            || (material.geometry.lattice == GratingLattice::OneDimensional
                && field.secondarySpacingDeltaNanometresPerUnit != 0.0f))
            return false;
        if (field.mode == GrooveFieldMode::Linear)
        {
            if (!canonical(field.axisUv[0]) || !canonical(field.axisUv[1])
                || std::abs(field.axisUv[0] * field.axisUv[0]
                            + field.axisUv[1] * field.axisUv[1] - 1.0f) > 0.0001f)
                return false;
        }
        else if (field.axisUv != std::array<float, 2> {})
            return false;

        float minimumCoordinate = 0.0f;
        float maximumCoordinate = 0.0f;
        bool first = true;
        for (const auto& corner : std::array<std::array<float, 2>, 4> {{
                 {{ 0.0f, 0.0f }}, {{ 1.0f, 0.0f }},
                 {{ 0.0f, 1.0f }}, {{ 1.0f, 1.0f }} }})
        {
            const auto dx = corner[0] - field.originUv[0];
            const auto dy = corner[1] - field.originUv[1];
            const auto coordinate = field.mode == GrooveFieldMode::Linear
                ? dx * field.axisUv[0] + dy * field.axisUv[1]
                : std::sqrt(dx * dx + dy * dy);
            if (first || coordinate < minimumCoordinate) minimumCoordinate = coordinate;
            if (first || coordinate > maximumCoordinate) maximumCoordinate = coordinate;
            first = false;
        }
        const auto periodValid = [&](float base, float delta)
        {
            return bounded(base + delta * minimumCoordinate,
                           kMinimumGrooveSpacingNanometres,
                           kMaximumGrooveSpacingNanometres)
                && bounded(base + delta * maximumCoordinate,
                           kMinimumGrooveSpacingNanometres,
                           kMaximumGrooveSpacingNanometres);
        };
        if (!periodValid(material.geometry.grooveSpacingNanometres,
                         field.grooveSpacingDeltaNanometresPerUnit)
            || (material.geometry.lattice == GratingLattice::CrossedTwoDimensional
                && !periodValid(material.geometry.secondaryGrooveSpacingNanometres,
                                field.secondarySpacingDeltaNanometresPerUnit)))
            return false;
    }

    const auto& micro = material.microstructure;
    if ((micro.profile != GrooveProfile::BinaryRectangular
         && micro.profile != GrooveProfile::Sinusoidal
         && micro.profile != GrooveProfile::BlazedSawtooth)
        || !bounded(micro.grooveDepthNanometres,
                    kMinimumGrooveDepthNanometres,
                    kMaximumGrooveDepthNanometres)
        || !bounded(micro.dutyCycle, 0.01f, 1.0f)
        || !canonical(micro.blazeAngleDegrees))
        return false;
    if (micro.profile == GrooveProfile::Sinusoidal
        && std::abs(micro.dutyCycle - 0.5f) > 1.0e-6f)
        return false;
    if (micro.profile != GrooveProfile::BlazedSawtooth)
    {
        if (micro.blazeAngleDegrees != 0.0f) return false;
    }
    else
    {
        constexpr double radiansToDegrees = 57.2957795130823208768;
        const auto expected = std::atan(
            static_cast<double>(micro.grooveDepthNanometres)
            / (static_cast<double>(micro.dutyCycle)
               * material.geometry.grooveSpacingNanometres)) * radiansToDegrees;
        if (!(micro.blazeAngleDegrees > 0.0f)
            || micro.blazeAngleDegrees >= 89.0f
            || std::abs(static_cast<double>(micro.blazeAngleDegrees) - expected)
                > 1.0e-3)
            return false;
        if (field.grooveSpacingDeltaNanometresPerUnit != 0.0f
            || field.secondarySpacingDeltaNanometresPerUnit != 0.0f)
            return false;
        if (material.geometry.lattice == GratingLattice::CrossedTwoDimensional
            && material.geometry.secondaryGrooveSpacingNanometres
                != material.geometry.grooveSpacingNanometres)
            return false;
    }
    if (!validOptical(material.substrate)) return false;
    if (material.substrateData != diffractionmaterial::SubstrateData::AuthoredConstants
        && material.substrateData != diffractionmaterial::SubstrateData::AluminiumMcPeak2015) return false;
    if (material.orderMedium != diffractionmaterial::OrderMedium::Air
        && material.orderMedium != diffractionmaterial::OrderMedium::SurroundingMedium) return false;
    if (material.pitStatistics != diffractionmaterial::PitStatistics::None
        && material.pitStatistics != diffractionmaterial::PitStatistics::PressedDisc) return false;
    if (material.coating.model == CoatingModel::Uncoated)
    {
        if (!canonical(material.coating.thicknessNanometres)
            || !canonical(material.coating.opticalConstants.refractiveIndex)
            || !canonical(material.coating.opticalConstants.extinctionCoefficient)
            || material.coating.thicknessNanometres != 0.0f
            || material.coating.opticalConstants.refractiveIndex != 0.0f
            || material.coating.opticalConstants.extinctionCoefficient != 0.0f)
            return false;
    }
    else if (material.coating.model != CoatingModel::IncoherentDielectric
             || !bounded(material.coating.thicknessNanometres, 0.1f,
                         kMaximumCoatingThicknessNanometres)
             || !validOptical(material.coating.opticalConstants))
        return false;
    if (!bounded(material.roughness.rmsHeightNanometres, 0.0f,
                 kMaximumRmsHeightNanometres)
        || !bounded(material.roughness.rmsSlope, 0.0f, kMaximumRmsSlope)
        || !bounded(material.roughness.acrossGrooveSlope, 0.0f, kMaximumRmsSlope)
        || !bounded(material.roughness.streakSlope, 0.0f, kMaximumStreakSlope)
        || !bounded(material.roughness.streakSpacingUv, kMinimumStreakSpacingUv, kMaximumStreakSpacingUv)
        || !bounded(material.roughness.reliefRmsHeightNanometres, 0.0f, kMaximumReliefRmsHeightNanometres)
        || !bounded(material.roughness.reliefMinimumSpacingNanometres, kMinimumReliefSpacingNanometres, kMaximumReliefSpacingNanometres)
        || material.spectrum.integrationModel
            != SpectralIntegrationModel::Cie1931Xyz
        || material.spectrum.wavelengthCount < kMinimumSpectralSamples
        || material.spectrum.wavelengthCount > kMaximumSpectralSamples)
        return false;
    float previous = 0.0f;
    for (std::size_t index = 0;
         index < material.spectrum.wavelengthsNanometres.size(); ++index)
    {
        const auto wavelength = material.spectrum.wavelengthsNanometres[index];
        if (index >= material.spectrum.wavelengthCount)
        {
            if (wavelength != 0.0f || std::signbit(wavelength)) return false;
        }
        else if (!bounded(wavelength, kMinimumWavelengthNanometres,
                          kMaximumWavelengthNanometres)
                 || (index != 0 && wavelength <= previous))
            return false;
        previous = wavelength;
    }
    return material.spectrum.firstOrder != 0
        && material.spectrum.firstOrder <= material.spectrum.lastOrder
        && material.spectrum.lastOrder <= kMaximumDiffractionOrder;
}

inline bool sameDiffractionMaterial(
    const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest& left,
    const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest& right) noexcept
{
    return left.version == right.version && left.scene == right.scene
        && left.sceneRevision == right.sceneRevision
        && left.structuralRevision == right.structuralRevision
        && left.evaluationRevision == right.evaluationRevision
        && left.materialRevision == right.materialRevision
        && left.object == right.object
        && left.structuralDigest == right.structuralDigest
        && left.graphFrame.has_value() == right.graphFrame.has_value()
        && (!left.graphFrame || (left.graphFrame->node == right.graphFrame->node
            && left.graphFrame->port == right.graphFrame->port))
        && sameDiffractionDescription(left.material, right.material)
        && diffractionmaterial::sameLightingDescription(left.lighting, right.lighting)
        && left.spatialFoil.has_value() == right.spatialFoil.has_value()
        && (!left.spatialFoil || [&] {
            const auto& a = *left.spatialFoil;
            const auto& b = *right.spatialFoil;
            if (a.version != b.version || a.workBudget.maximumEvaluations
                    != b.workBudget.maximumEvaluations)
                return false;
            for (std::size_t i = 0; i < a.grooveField.size(); ++i)
                if (a.grooveField[i].reciprocalDirectionUv
                        != b.grooveField[i].reciprocalDirectionUv
                    || a.grooveField[i].grooveSpacingNanometres
                        != b.grooveField[i].grooveSpacingNanometres
                    || a.grooveField[i].diffractionCoverage
                        != b.grooveField[i].diffractionCoverage)
                    return false;
            return true;
        }());
}

inline bool validDiffractionMaterial(
    const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest& material) noexcept
{
    std::string lightingError;
    const auto lighting = diffractionmaterial::AdmittedLightingPlan::admit(
        material.lighting, lightingError);
    return ((material.version == diffractionmaterialbinding::kWireVersion
             && diffractionmaterial::sameLightingDescription(
                 material.lighting, diffractionmaterial::makeReferenceLighting()))
            || (material.version == diffractionmaterialbinding::kEnvironmentWireVersion
                && material.lighting.version == 2 && lighting)
            || (material.version == diffractionmaterialbinding::kGraphFrameWireVersion
                && lighting && (material.lighting.version == 2
                    || diffractionmaterial::sameLightingDescription(
                        material.lighting, diffractionmaterial::makeReferenceLighting())))
            || (material.version == diffractionmaterialbinding::kSceneReceiverWireVersion
                && lighting && material.lighting.version == 3))
        && (material.version == diffractionmaterialbinding::kSceneReceiverWireVersion
            || (material.graphFrame.has_value()
                == (material.version == diffractionmaterialbinding::kGraphFrameWireVersion)))
        && (!material.graphFrame || (material.graphFrame->node >= 0 && material.graphFrame->port == 0))
        && material.scene.isValid() && material.object.isValid()
        && material.sceneRevision != 0 && material.structuralRevision != 0
        && material.evaluationRevision != 0
        && material.materialRevision == material.structuralRevision
        && visualmodelasset::validSha256(material.structuralDigest)
        && validDiffractionDescription(material.material)
        && (!material.spatialFoil
            || (material.spatialFoil->version == diffractivefoil::kWireVersion
                && sameDiffractionDescription(
                    material.spatialFoil->physicalBsdf, material.material)
                && material.spatialFoil->workBudget.maximumEvaluations > 0
                && material.spatialFoil->workBudget.maximumEvaluations
                    <= diffractivefoil::kMaximumEvaluations));
}

inline void encodeLegacyV6DiffractionMaterial(
    std::ostringstream& encoded,
    const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest& binding)
{
    const auto& material = binding.material;
    encoded << binding.version << ' ' << binding.scene.value << ' '
            << binding.sceneRevision << ' ' << binding.structuralRevision << ' '
            << binding.evaluationRevision << ' ' << binding.materialRevision << ' '
            << binding.object.value << ' ' << std::quoted(binding.structuralDigest) << '\n'
            << material.version << ' '
            << floatBits(material.geometry.directionUv[0]) << ' '
            << floatBits(material.geometry.directionUv[1]) << ' '
            << floatBits(material.geometry.grooveSpacingNanometres) << ' '
            << static_cast<unsigned>(material.geometry.lattice) << ' '
            << floatBits(material.geometry.secondaryDirectionUv[0]) << ' '
            << floatBits(material.geometry.secondaryDirectionUv[1]) << ' '
            << floatBits(material.geometry.secondaryGrooveSpacingNanometres) << '\n'
            << static_cast<unsigned>(material.microstructure.profile) << ' '
            << floatBits(material.microstructure.grooveDepthNanometres) << ' '
            << floatBits(material.microstructure.dutyCycle) << ' '
            << floatBits(material.microstructure.blazeAngleDegrees) << '\n'
            << floatBits(material.substrate.refractiveIndex) << ' '
            << floatBits(material.substrate.extinctionCoefficient) << '\n'
            << static_cast<unsigned>(material.coating.model) << ' '
            << floatBits(material.coating.thicknessNanometres) << ' '
            << floatBits(material.coating.opticalConstants.refractiveIndex) << ' '
            << floatBits(material.coating.opticalConstants.extinctionCoefficient) << '\n'
            << floatBits(material.roughness.rmsHeightNanometres) << ' '
            << floatBits(material.roughness.rmsSlope) << '\n'
            << static_cast<unsigned>(material.spectrum.integrationModel) << ' '
            << static_cast<unsigned>(material.spectrum.wavelengthCount);
    for (const auto wavelength : material.spectrum.wavelengthsNanometres)
        encoded << ' ' << floatBits(wavelength);
    encoded << ' ' << static_cast<unsigned>(material.spectrum.firstOrder)
            << ' ' << static_cast<unsigned>(material.spectrum.lastOrder) << '\n';
}

inline bool decodeLegacyV6DiffractionMaterial(
    std::istream& input,
    diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest& binding)
{
    std::uint8_t lattice = 0, profile = 0, coating = 0, integration = 0;
    if (!readUnsigned(input, binding.version)
        || !readUnsigned(input, binding.scene.value)
        || !readUnsigned(input, binding.sceneRevision)
        || !readUnsigned(input, binding.structuralRevision)
        || !readUnsigned(input, binding.evaluationRevision)
        || !readUnsigned(input, binding.materialRevision)
        || !readUnsigned(input, binding.object.value)
        || !(input >> std::quoted(binding.structuralDigest))
        || !readUnsigned(input, binding.material.version))
        return false;
    const auto readFloat = [&input](float& value)
    {
        std::uint32_t bits = 0;
        if (!readUnsigned(input, bits)) return false;
        value = floatFromBits(bits);
        return true;
    };
    auto& material = binding.material;
    if (!readFloat(material.geometry.directionUv[0])
        || !readFloat(material.geometry.directionUv[1])
        || !readFloat(material.geometry.grooveSpacingNanometres)
        || !readUnsigned(input, lattice)
        || !readFloat(material.geometry.secondaryDirectionUv[0])
        || !readFloat(material.geometry.secondaryDirectionUv[1])
        || !readFloat(material.geometry.secondaryGrooveSpacingNanometres)
        || !readUnsigned(input, profile)
        || !readFloat(material.microstructure.grooveDepthNanometres)
        || !readFloat(material.microstructure.dutyCycle)
        || !readFloat(material.microstructure.blazeAngleDegrees)
        || !readFloat(material.substrate.refractiveIndex)
        || !readFloat(material.substrate.extinctionCoefficient)
        || !readUnsigned(input, coating)
        || !readFloat(material.coating.thicknessNanometres)
        || !readFloat(material.coating.opticalConstants.refractiveIndex)
        || !readFloat(material.coating.opticalConstants.extinctionCoefficient)
        || !readFloat(material.roughness.rmsHeightNanometres)
        || !readFloat(material.roughness.rmsSlope)
        || !readUnsigned(input, integration)
        || !readUnsigned(input, material.spectrum.wavelengthCount))
        return false;
    material.geometry.lattice
        = static_cast<diffractionmaterial::GratingLattice>(lattice);
    material.microstructure.profile
        = static_cast<diffractionmaterial::GrooveProfile>(profile);
    material.coating.model = static_cast<diffractionmaterial::CoatingModel>(coating);
    material.spectrum.integrationModel
        = static_cast<diffractionmaterial::SpectralIntegrationModel>(integration);
    for (auto& wavelength : material.spectrum.wavelengthsNanometres)
        if (!readFloat(wavelength)) return false;
    return readUnsigned(input, material.spectrum.firstOrder)
        && readUnsigned(input, material.spectrum.lastOrder)
        && validLegacyV6DiffractionMaterial(binding);
}

inline void encodeDiffractionMaterial(
    std::ostringstream& encoded,
    const diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest& binding)
{
    const auto& material = binding.material;
    encoded << binding.version << ' ' << binding.scene.value << ' '
            << binding.sceneRevision << ' ' << binding.structuralRevision << ' '
            << binding.evaluationRevision << ' ' << binding.materialRevision << ' '
            << binding.object.value << ' ' << std::quoted(binding.structuralDigest) << '\n'
            << material.version << ' '
            << floatBits(material.geometry.directionUv[0]) << ' '
            << floatBits(material.geometry.directionUv[1]) << ' '
            << floatBits(material.geometry.grooveSpacingNanometres) << ' '
            << (material.geometry.thirdGrooveFamily ? 3u : static_cast<unsigned>(material.geometry.lattice)) << ' '
            << floatBits(material.geometry.secondaryDirectionUv[0]) << ' '
            << floatBits(material.geometry.secondaryDirectionUv[1]) << ' '
            << floatBits(material.geometry.secondaryGrooveSpacingNanometres) << '\n'
            << static_cast<unsigned>(material.grooveField.mode) << ' '
            << floatBits(material.grooveField.originUv[0]) << ' '
            << floatBits(material.grooveField.originUv[1]) << ' '
            << floatBits(material.grooveField.axisUv[0]) << ' '
            << floatBits(material.grooveField.axisUv[1]) << ' '
            << floatBits(material.grooveField.grooveSpacingDeltaNanometresPerUnit) << ' '
            << floatBits(material.grooveField.secondarySpacingDeltaNanometresPerUnit) << ' '
            << floatBits(material.grooveField.orientationDegreesPerUnit) << '\n'
            << static_cast<unsigned>(material.microstructure.profile) << ' '
            << floatBits(material.microstructure.grooveDepthNanometres) << ' '
            << floatBits(material.microstructure.dutyCycle) << ' '
            << floatBits(material.microstructure.blazeAngleDegrees) << '\n'
            << floatBits(material.substrate.refractiveIndex) << ' '
            << floatBits(material.substrate.extinctionCoefficient) << '\n'
            // Optional and only when set: wires already written have no such token.
            << (material.substrateData == diffractionmaterial::SubstrateData::AluminiumMcPeak2015 ? "a1\n" : "")
            << (material.orderMedium == diffractionmaterial::OrderMedium::SurroundingMedium ? "m1\n" : "")
            << (material.pitStatistics == diffractionmaterial::PitStatistics::PressedDisc ? "p1\n" : "")
            << static_cast<unsigned>(material.coating.model) << ' '
            << floatBits(material.coating.thicknessNanometres) << ' '
            << floatBits(material.coating.opticalConstants.refractiveIndex) << ' '
            << floatBits(material.coating.opticalConstants.extinctionCoefficient) << '\n'
            << floatBits(material.roughness.rmsHeightNanometres) << ' '
            << floatBits(material.roughness.rmsSlope);
    // Only when authored, so requests saved before this slope keep their exact bytes.
    if (floatBits(material.roughness.acrossGrooveSlope) != floatBits(diffractionmaterial::SurfaceRoughness{}.acrossGrooveSlope))
        encoded << " g " << floatBits(material.roughness.acrossGrooveSlope);
    // Only when authored, so requests saved before brushed relief keep their exact bytes.
    if (floatBits(material.roughness.streakSlope) != floatBits(diffractionmaterial::SurfaceRoughness{}.streakSlope)
        || floatBits(material.roughness.streakSpacingUv) != floatBits(diffractionmaterial::SurfaceRoughness{}.streakSpacingUv))
        encoded << " s " << floatBits(material.roughness.streakSlope) << ' '
                << floatBits(material.roughness.streakSpacingUv);
    // Only when authored, so requests saved before the relief lobe keep their exact bytes.
    if (floatBits(material.roughness.reliefRmsHeightNanometres) != floatBits(diffractionmaterial::SurfaceRoughness{}.reliefRmsHeightNanometres)
        || floatBits(material.roughness.reliefMinimumSpacingNanometres) != floatBits(diffractionmaterial::SurfaceRoughness{}.reliefMinimumSpacingNanometres))
        encoded << " r " << floatBits(material.roughness.reliefRmsHeightNanometres) << ' '
                << floatBits(material.roughness.reliefMinimumSpacingNanometres);
    encoded << '\n'
            << static_cast<unsigned>(material.spectrum.integrationModel) << ' '
            << static_cast<unsigned>(material.spectrum.wavelengthCount);
    for (const auto wavelength : material.spectrum.wavelengthsNanometres)
        encoded << ' ' << floatBits(wavelength);
    encoded << ' ' << static_cast<unsigned>(material.spectrum.firstOrder)
            << ' ' << static_cast<unsigned>(material.spectrum.lastOrder) << '\n'
            << static_cast<unsigned>(binding.spatialFoil.has_value());
    if (binding.spatialFoil)
    {
        encoded << ' ' << binding.spatialFoil->version << ' '
                << binding.spatialFoil->workBudget.maximumEvaluations;
        for (const auto& sample : binding.spatialFoil->grooveField)
            encoded << ' ' << floatBits(sample.reciprocalDirectionUv[0])
                    << ' ' << floatBits(sample.reciprocalDirectionUv[1])
                    << ' ' << floatBits(sample.grooveSpacingNanometres)
                    << ' ' << floatBits(sample.diffractionCoverage);
    }
    encoded << '\n';
    if (binding.version == diffractionmaterialbinding::kEnvironmentWireVersion
        || binding.version == diffractionmaterialbinding::kGraphFrameWireVersion)
    {
        encoded << binding.lighting.version << ' '
                << floatBits(binding.lighting.bouncePlaneHeight) << ' '
                << floatBits(binding.lighting.bounceMaximumDistance) << '\n';
        for (std::size_t i = 0; i < 3; ++i)
        {
            for (const auto value : binding.lighting.paths[i].incident.direction)
                encoded << floatBits(value) << ' ';
            for (const auto value : binding.lighting.paths[i].incident.radiance)
                encoded << floatBits(value) << ' ';
            encoded << '\n';
        }
    }
    if (binding.version == diffractionmaterialbinding::kSceneReceiverWireVersion)
    {
        encoded << binding.lighting.version << ' '
                << static_cast<unsigned>(binding.lighting.sceneReceiverTransport) << ' '
                << floatBits(binding.lighting.traceMaximumDistance) << '\n';
        for (const auto value : binding.lighting.paths[0].incident.direction)
            encoded << floatBits(value) << ' ';
        for (const auto value : binding.lighting.paths[0].incident.radiance)
            encoded << floatBits(value) << ' ';
        encoded << '\n' << (binding.graphFrame ? 1 : 0) << '\n';
        if (binding.graphFrame)
            encoded << binding.graphFrame->node << ' ' << binding.graphFrame->port << '\n';
    }
    if (binding.version == diffractionmaterialbinding::kGraphFrameWireVersion)
        encoded << binding.graphFrame->node << ' ' << binding.graphFrame->port << '\n';
}

inline bool decodeDiffractionMaterial(
    std::istream& input,
    diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest& binding)
{
    std::uint8_t lattice = 0, fieldMode = 0, profile = 0, coating = 0, integration = 0;
    if (!readUnsigned(input, binding.version)
        || !readUnsigned(input, binding.scene.value)
        || !readUnsigned(input, binding.sceneRevision)
        || !readUnsigned(input, binding.structuralRevision)
        || !readUnsigned(input, binding.evaluationRevision)
        || !readUnsigned(input, binding.materialRevision)
        || !readUnsigned(input, binding.object.value)
        || !(input >> std::quoted(binding.structuralDigest))
        || !readUnsigned(input, binding.material.version))
        return false;
    const auto readFloat = [&input](float& value)
    {
        std::uint32_t bits = 0;
        if (!readUnsigned(input, bits)) return false;
        value = floatFromBits(bits);
        return true;
    };
    auto& material = binding.material;
    const auto readAcrossGrooveSlope = [&]
    {
        input >> std::ws;
        if (input.peek() != 'g')
            return true;
        input.get();
        return readFloat(material.roughness.acrossGrooveSlope);
    };
    const auto readSubstrateData = [&]
    {
        input >> std::ws;
        if (input.peek() != 'a')
            return true;
        input.get();
        std::uint8_t data = 0;
        if (!readUnsigned(input, data) || data != 1)
            return false;
        material.substrateData = diffractionmaterial::SubstrateData::AluminiumMcPeak2015;
        return true;
    };
    const auto readOrderMedium = [&]
    {
        input >> std::ws;
        if (input.peek() != 'm')
            return true;
        input.get();
        std::uint8_t medium = 0;
        if (!readUnsigned(input, medium) || medium != 1)
            return false;
        material.orderMedium = diffractionmaterial::OrderMedium::SurroundingMedium;
        return true;
    };
    const auto readPitStatistics = [&]
    {
        input >> std::ws;
        if (input.peek() != 'p')
            return true;
        input.get();
        std::uint8_t statistics = 0;
        if (!readUnsigned(input, statistics) || statistics != 1)
            return false;
        material.pitStatistics = diffractionmaterial::PitStatistics::PressedDisc;
        return true;
    };
    const auto readReliefScatter = [&]
    {
        input >> std::ws;
        if (input.peek() != 'r')
            return true;
        input.get();
        return readFloat(material.roughness.reliefRmsHeightNanometres)
            && readFloat(material.roughness.reliefMinimumSpacingNanometres);
    };
    const auto readBrushedRelief = [&]
    {
        input >> std::ws;
        if (input.peek() != 's')
            return true;
        input.get();
        return readFloat(material.roughness.streakSlope)
            && readFloat(material.roughness.streakSpacingUv);
    };
    if (!readFloat(material.geometry.directionUv[0])
        || !readFloat(material.geometry.directionUv[1])
        || !readFloat(material.geometry.grooveSpacingNanometres)
        || !readUnsigned(input, lattice)
        || !readFloat(material.geometry.secondaryDirectionUv[0])
        || !readFloat(material.geometry.secondaryDirectionUv[1])
        || !readFloat(material.geometry.secondaryGrooveSpacingNanometres)
        || !readUnsigned(input, fieldMode)
        || !readFloat(material.grooveField.originUv[0])
        || !readFloat(material.grooveField.originUv[1])
        || !readFloat(material.grooveField.axisUv[0])
        || !readFloat(material.grooveField.axisUv[1])
        || !readFloat(material.grooveField.grooveSpacingDeltaNanometresPerUnit)
        || !readFloat(material.grooveField.secondarySpacingDeltaNanometresPerUnit)
        || !readFloat(material.grooveField.orientationDegreesPerUnit)
        || !readUnsigned(input, profile)
        || !readFloat(material.microstructure.grooveDepthNanometres)
        || !readFloat(material.microstructure.dutyCycle)
        || !readFloat(material.microstructure.blazeAngleDegrees)
        || !readFloat(material.substrate.refractiveIndex)
        || !readFloat(material.substrate.extinctionCoefficient)
        || !readSubstrateData()
        || !readOrderMedium()
        || !readPitStatistics()
        || !readUnsigned(input, coating)
        || !readFloat(material.coating.thicknessNanometres)
        || !readFloat(material.coating.opticalConstants.refractiveIndex)
        || !readFloat(material.coating.opticalConstants.extinctionCoefficient)
        || !readFloat(material.roughness.rmsHeightNanometres)
        || !readFloat(material.roughness.rmsSlope)
        || !readAcrossGrooveSlope()
        || !readBrushedRelief()
        || !readReliefScatter()
        || !readUnsigned(input, integration)
        || !readUnsigned(input, material.spectrum.wavelengthCount))
        return false;
    // Wire token 3 is a crossed lattice with its third groove family.
    material.geometry.thirdGrooveFamily = lattice == 3u;
    if (material.geometry.thirdGrooveFamily)
        lattice = static_cast<std::uint8_t>(diffractionmaterial::GratingLattice::CrossedTwoDimensional);
    material.geometry.lattice
        = static_cast<diffractionmaterial::GratingLattice>(lattice);
    material.grooveField.mode
        = static_cast<diffractionmaterial::GrooveFieldMode>(fieldMode);
    material.microstructure.profile
        = static_cast<diffractionmaterial::GrooveProfile>(profile);
    material.coating.model
        = static_cast<diffractionmaterial::CoatingModel>(coating);
    material.spectrum.integrationModel
        = static_cast<diffractionmaterial::SpectralIntegrationModel>(integration);
    for (auto& wavelength : material.spectrum.wavelengthsNanometres)
        if (!readFloat(wavelength)) return false;
    if (!readUnsigned(input, material.spectrum.firstOrder)
        || !readUnsigned(input, material.spectrum.lastOrder))
        return false;
    std::uint8_t hasFoil = 0;
    if (!readUnsigned(input, hasFoil) || hasFoil > 1)
        return false;
    if (hasFoil != 0)
    {
        binding.spatialFoil.emplace();
        auto& foil = *binding.spatialFoil;
        foil.physicalBsdf = material;
        if (!readUnsigned(input, foil.version)
            || !readUnsigned(input, foil.workBudget.maximumEvaluations))
            return false;
        for (auto& sample : foil.grooveField)
            if (!readFloat(sample.reciprocalDirectionUv[0])
                || !readFloat(sample.reciprocalDirectionUv[1])
                || !readFloat(sample.grooveSpacingNanometres)
                || !readFloat(sample.diffractionCoverage))
                return false;
    }
    binding.lighting = diffractionmaterial::makeReferenceLighting();
    if (binding.version == diffractionmaterialbinding::kEnvironmentWireVersion
        || binding.version == diffractionmaterialbinding::kGraphFrameWireVersion)
    {
        if (!readUnsigned(input, binding.lighting.version)
            || (binding.lighting.version != 2
                && binding.version != diffractionmaterialbinding::kGraphFrameWireVersion)
            || !readFloat(binding.lighting.bouncePlaneHeight)
            || !readFloat(binding.lighting.bounceMaximumDistance))
            return false;
        for (std::size_t i = 0; i < 3; ++i)
        {
            for (auto& value : binding.lighting.paths[i].incident.direction)
                if (!readFloat(value)) return false;
            for (auto& value : binding.lighting.paths[i].incident.radiance)
                if (!readFloat(value)) return false;
        }
    }
    if (binding.version == diffractionmaterialbinding::kSceneReceiverWireVersion)
    {
        binding.lighting = {};
        unsigned receiverTransport = 0;
        unsigned hasFrame = 0;
        if (!readUnsigned(input, binding.lighting.version) || binding.lighting.version != 3
            || !readUnsigned(input, receiverTransport) || receiverTransport != 1
            || !readFloat(binding.lighting.traceMaximumDistance)) return false;
        binding.lighting.sceneReceiverTransport
            = diffractionmaterial::SceneReceiverTransport::AuthoredSpectralLambertian;
        binding.lighting.pathCount = 1;
        binding.lighting.paths[0].kind = diffractionmaterial::LightingPathKind::Direct;
        for (auto& value : binding.lighting.paths[0].incident.direction)
            if (!readFloat(value)) return false;
        for (auto& value : binding.lighting.paths[0].incident.radiance)
            if (!readFloat(value)) return false;
        if (!readUnsigned(input, hasFrame) || hasFrame > 1) return false;
        binding.graphFrame.reset();
        if (hasFrame)
        {
            binding.graphFrame.emplace();
            if (!(input >> binding.graphFrame->node >> binding.graphFrame->port)) return false;
        }
    }
    if (binding.version == diffractionmaterialbinding::kGraphFrameWireVersion)
    {
        binding.graphFrame.emplace();
        if (!(input >> binding.graphFrame->node >> binding.graphFrame->port)) return false;
    }
    return validDiffractionMaterial(binding);
}

inline void encodeBase(std::ostringstream& encoded, std::string_view header,
                       const Request& request)
{
    encoded << header << '\n'
            << request.sourceStableId << ' ' << request.renderStableId << '\n'
            << std::quoted(request.asset.id) << '\n'
            << request.asset.version << '\n'
            << std::quoted(request.asset.contentSha256) << '\n'
            << std::quoted(request.asset.sourceMediaType) << '\n'
            << request.asset.sourceByteSize << '\n';
    if (request.sceneIndex)
        encoded << 1 << ' ' << *request.sceneIndex << '\n';
    else
        encoded << 0 << ' ' << 0 << '\n';
}

inline void encodeMaterial(
    std::ostringstream& encoded,
    const surfacematerialbinding::ImportedSceneMaterialRequest& material,
    bool encodeVertex)
{
    encoded << material.version << ' ' << material.scene.value << ' '
            << material.sceneRevision << ' ' << material.structuralRevision << ' '
            << material.evaluationRevision << ' ' << material.programRevision << '\n';
    if(material.version==surfacematerialbinding::kRoughGlassWireVersion
        ||material.version==surfacematerialbinding::kDiffractionLayerWireVersion)
        encoded << material.roughGlassScatteringVersion << ' ' << material.program.spectralReflectance.has_value() << '\n';
    encoded << material.program.version << ' '
            << material.program.textureSlotCount << ' '
            << material.program.operations.size() << '\n';
    for (const auto& operation : material.program.operations)
    {
        encoded << operation.id << ' '
                << static_cast<unsigned>(operation.kind) << ' '
                << static_cast<unsigned>(operation.resultType) << ' '
                << static_cast<unsigned>(operation.inputCount);
        for (const auto input : operation.inputs)
            encoded << ' ' << input;
        for (const auto literal : operation.literal)
            encoded << ' ' << floatBits(literal);
        encoded << ' ' << static_cast<unsigned>(operation.semantic)
                << ' ' << operation.parameter << ' '
                << operation.unsignedLiteral << '\n';
    }
    for (std::size_t index = 0; index < surfacematerial::outputCount(material.program.version); ++index)
    {
        if (index != 0) encoded << ' ';
        encoded << material.program.outputs[index];
    }
    if (material.program.version == surfacematerial::kSpectralWireVersion)
        encoded << ' ' << static_cast<unsigned>(material.program.transport);
    encoded << '\n';
    if (material.program.spectralReflectance)
    {
        const auto& reflectance = *material.program.spectralReflectance;
        encoded << reflectance.version;
        for (const auto coefficient : reflectance.coefficients)
            encoded << ' ' << floatBits(coefficient);
        encoded << '\n';
    }
    if (material.version == surfacematerialbinding::kDiffractionLayerWireVersion && material.program.diffractionLayer)
    {
        const auto& layer = *material.program.diffractionLayer;
        encoded << layer.version << ' ' << layer.grooveMode << ' ' << layer.baseMode << ' ' << layer.maxOrder;
        surfacematerial::forEachDiffractionLayerFloat(layer, [&encoded](const char*, float value) { encoded << ' ' << floatBits(value); });
        encoded << '\n';
    }
    if (encodeVertex)
    {
        const auto& vertex = *material.vertexModifier;
        encoded << vertex.schemaVersion << ' ' << vertex.rootId << ' '
                << vertex.records.size() << '\n';
        for (const auto& record : vertex.records)
        {
            encoded << record.stableId << ' '
                    << static_cast<unsigned>(record.operation) << ' '
                    << static_cast<unsigned>(record.resultType) << ' '
                    << static_cast<unsigned>(record.inputCount) << ' '
                    << static_cast<unsigned>(record.parameterCount);
            for (const auto input : record.inputs) encoded << ' ' << input;
            for (const auto parameter : record.parameters)
                encoded << ' ' << doubleBits(parameter);
            encoded << '\n';
        }
    }
    encoded
            << static_cast<unsigned>(material.binding.targetKind) << ' '
            << material.binding.materialSlot.value << ' '
            << material.binding.object.value << ' '
            << std::quoted(material.binding.surfaceMaterialDigest) << ' '
            << material.binding.surfaceMaterialRevision << '\n'
            << material.binding.textures.size() << '\n';
    for (const auto& texture : material.binding.textures)
    {
        encoded << texture.slot << ' ' << static_cast<unsigned>(texture.source);
        for (const auto byte : texture.videoResource)
            encoded << ' ' << static_cast<unsigned>(byte);
        if (material.version == surfacematerialbinding::kGraphFrameWireVersion
            || material.version == surfacematerialbinding::kReflectanceWireVersion
            || material.version == surfacematerialbinding::kRoughGlassWireVersion
            || material.version == surfacematerialbinding::kDiffractionLayerWireVersion)
            encoded << ' ' << (texture.graphFrame ? static_cast<std::uint32_t>(texture.graphFrame->node) : UINT32_MAX)
                    << ' ' << (texture.graphFrame ? static_cast<std::uint32_t>(texture.graphFrame->port) : UINT32_MAX);
        encoded << '\n';
    }
}

inline bool decodeBase(std::istream& input, Request& request)
{
    int hasScene = -1;
    std::uint64_t scene = 0;
    if (!(input >> request.sourceStableId >> request.renderStableId)
        || !(input >> std::quoted(request.asset.id))
        || !(input >> request.asset.version)
        || !(input >> std::quoted(request.asset.contentSha256))
        || !(input >> std::quoted(request.asset.sourceMediaType))
        || !(input >> request.asset.sourceByteSize)
        || !(input >> hasScene >> scene)
        || (hasScene != 0 && hasScene != 1))
        return false;
    if (hasScene != 0)
    {
        if (scene > static_cast<std::uint64_t>(
                        std::numeric_limits<std::size_t>::max()))
            return false;
        request.sceneIndex = static_cast<std::size_t>(scene);
    }
    else if (scene != 0)
        return false;
    return true;
}

inline bool decodeMaterial(
    std::istream& input,
    surfacematerialbinding::ImportedSceneMaterialRequest& material,
    bool decodeVertex)
{
    std::uint32_t scene = 0;
    std::uint64_t operationCount = 0;
    if (!readUnsigned(input, material.version)
        || !readUnsigned(input, scene)
        || !readUnsigned(input, material.sceneRevision)
        || !readUnsigned(input, material.structuralRevision)
        || !readUnsigned(input, material.evaluationRevision)
        || !readUnsigned(input, material.programRevision)) return false;
    bool hasReflectance=material.version==surfacematerialbinding::kReflectanceWireVersion;
    material.roughGlassScatteringVersion=1;
    if(material.version==surfacematerialbinding::kRoughGlassWireVersion
        ||material.version==surfacematerialbinding::kDiffractionLayerWireVersion) {
        unsigned flag=0;
        if(!readUnsigned(input,material.roughGlassScatteringVersion)||!readUnsigned(input,flag)||flag>1)return false;
        hasReflectance=flag!=0;
    }
    if (!readUnsigned(input, material.program.version)
        || !readUnsigned(input, material.program.textureSlotCount)
        || !readUnsigned(input, operationCount)
        || operationCount > surfacematerial::kMaximumOperations)
        return false;
    material.scene.value = scene;
    for (std::uint64_t index = 0; index < operationCount; ++index)
    {
        surfacematerial::Operation operation;
        std::uint8_t kind = 0;
        std::uint8_t resultType = 0;
        std::uint8_t inputCount = 0;
        std::uint8_t semantic = 0;
        if (!readUnsigned(input, operation.id)
            || !readUnsigned(input, kind)
            || !readUnsigned(input, resultType)
            || !readUnsigned(input, inputCount))
            return false;
        operation.kind = static_cast<surfacematerial::OperationKind>(kind);
        operation.resultType = static_cast<surfacematerial::ValueType>(resultType);
        operation.inputCount = inputCount;
        for (auto& source : operation.inputs)
            if (!readUnsigned(input, source)) return false;
        for (auto& literal : operation.literal)
        {
            std::uint32_t bits = 0;
            if (!readUnsigned(input, bits)) return false;
            literal = floatFromBits(bits);
        }
        if (!readUnsigned(input, semantic)
            || !readUnsigned(input, operation.parameter)
            || !readUnsigned(input, operation.unsignedLiteral))
            return false;
        operation.semantic = static_cast<surfacematerial::InputSemantic>(semantic);
        material.program.operations.push_back(operation);
    }
    const auto outputCount = surfacematerial::outputCount(material.program.version);
    if (outputCount == 0) return false;
    for (std::size_t index = 0; index < outputCount; ++index)
        if (!readUnsigned(input, material.program.outputs[index])) return false;
    if (material.program.version == surfacematerial::kSpectralWireVersion)
    {
        std::uint8_t transport = 0;
        if (!readUnsigned(input, transport) || transport > static_cast<std::uint8_t>(surfacematerial::TransportMode::ThinInterface)) return false;
        material.program.transport = static_cast<surfacematerial::TransportMode>(transport);
    }

    if (hasReflectance)
    {
        surfacematerial::SpectralReflectanceDescription reflectance;
        if (!readUnsigned(input, reflectance.version)) return false;
        for (auto& coefficient : reflectance.coefficients)
        {
            std::uint32_t bits = 0;
            if (!readUnsigned(input, bits)) return false;
            coefficient = floatFromBits(bits);
        }
        if (!surfacematerial::validSpectralReflectance(reflectance)) return false;
        material.program.spectralReflectance = reflectance;
    }
    if (material.version == surfacematerialbinding::kDiffractionLayerWireVersion)
    {
        surfacematerial::DiffractionLayerDescription layer;
        if (!readUnsigned(input, layer.version) || !readUnsigned(input, layer.grooveMode)
            || !readUnsigned(input, layer.baseMode) || !readUnsigned(input, layer.maxOrder)) return false;
        bool complete = true;
        surfacematerial::forEachDiffractionLayerFloat(layer, [&](const char*, float& value) {
            std::uint32_t bits = 0;
            if (complete && readUnsigned(input, bits)) value = floatFromBits(bits);
            else complete = false;
        });
        if (!complete || !surfacematerial::validDiffractionLayer(layer)) return false;
        material.program.diffractionLayer = layer;
    }
    if (decodeVertex)
    {
        videowire::VertexModifierIr vertex;
        std::uint64_t recordCount = 0;
        if (!readUnsigned(input, vertex.schemaVersion)
            || !readUnsigned(input, vertex.rootId)
            || !readUnsigned(input, recordCount))
            return false;
        // The count is untrusted, so records grow as they parse and end with the input.
        for (std::uint64_t index = 0; index < recordCount; ++index)
        {
            videowire::VertexModifierRecord record;
            std::uint8_t operation = 0;
            std::uint8_t resultType = 0;
            if (!readUnsigned(input, record.stableId)
                || !readUnsigned(input, operation)
                || !readUnsigned(input, resultType)
                || !readUnsigned(input, record.inputCount)
                || !readUnsigned(input, record.parameterCount))
                return false;
            record.operation = static_cast<videowire::VertexModifierOperation>(operation);
            record.resultType = static_cast<videowire::VertexModifierValueType>(resultType);
            for (auto& source : record.inputs)
                if (!readUnsigned(input, source)) return false;
            for (auto& parameter : record.parameters)
            {
                std::uint64_t bits = 0;
                if (!readUnsigned(input, bits)) return false;
                parameter = doubleFromBits(bits);
            }
            vertex.records.push_back(record);
        }
        if (!validVertexModifier(vertex)) return false;
        material.vertexModifier = std::move(vertex);
    }

    std::uint8_t targetKind = 0;
    if (!readUnsigned(input, targetKind)
        || !readUnsigned(input, material.binding.materialSlot.value)
        || !readUnsigned(input, material.binding.object.value)
        || !(input >> std::quoted(material.binding.surfaceMaterialDigest))
        || !readUnsigned(input, material.binding.surfaceMaterialRevision))
        return false;
    material.binding.targetKind
        = static_cast<surfacematerialbinding::BindingTargetKind>(targetKind);

    std::uint64_t textureCount = 0;
    // Textures are read one by one, so a forged count fails on the missing bytes.
    if (!readUnsigned(input, textureCount))
        return false;
    for (std::uint64_t index = 0; index < textureCount; ++index)
    {
        surfacematerialbinding::TextureSlotBinding texture;
        std::uint8_t source = 0;
        if (!readUnsigned(input, texture.slot) || !readUnsigned(input, source))
            return false;
        texture.source
            = static_cast<surfacematerialbinding::TextureSourceKind>(source);
        for (auto& byte : texture.videoResource)
            if (!readUnsigned(input, byte)) return false;
        if (material.version == surfacematerialbinding::kGraphFrameWireVersion
            || material.version == surfacematerialbinding::kReflectanceWireVersion
            || material.version == surfacematerialbinding::kRoughGlassWireVersion
            || material.version == surfacematerialbinding::kDiffractionLayerWireVersion)
        {
            std::uint32_t node = 0, port = 0;
            if (!readUnsigned(input, node) || !readUnsigned(input, port)) return false;
            if (texture.source == surfacematerialbinding::TextureSourceKind::GraphFrame)
            {
                if (node > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())
                    || port > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) return false;
                texture.graphFrame = surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint {
                    static_cast<std::int32_t>(node), static_cast<std::int32_t>(port) };
            }
            else if (node != UINT32_MAX || port != UINT32_MAX) return false;
        }
        material.binding.textures.push_back(texture);
    }
    return validMaterial(material);
}

inline bool validCamera(const HarmonicMIDI::grid::SceneCameraRecord& camera) noexcept
{
    using namespace HarmonicMIDI::grid;
    return camera.id.isValid()
        && visual3d_detail::validTransform(camera.transform)
        && camera.transform.scale.x == 1.0f
        && camera.transform.scale.y == 1.0f
        && camera.transform.scale.z == 1.0f
        && visual3d_detail::finite(camera.verticalFovRadians)
        && visual3d_detail::finite(camera.nearPlane)
        && visual3d_detail::finite(camera.farPlane)
        && camera.verticalFovRadians > 0.0f
        && camera.verticalFovRadians < 3.1415926535f
        && camera.nearPlane > 0.0f && camera.farPlane > camera.nearPlane
        && visual3d_detail::finite(camera.orthographicHeight) && camera.orthographicHeight >= 0.0f
        && validSceneCameraLens(camera);
}

inline bool sameCamera(const HarmonicMIDI::grid::SceneCameraRecord& left,
                       const HarmonicMIDI::grid::SceneCameraRecord& right) noexcept
{
    return left.id == right.id
        && std::memcmp(&left.transform, &right.transform, sizeof(left.transform)) == 0
        && floatBits(left.verticalFovRadians) == floatBits(right.verticalFovRadians)
        && floatBits(left.nearPlane) == floatBits(right.nearPlane)
        && floatBits(left.farPlane) == floatBits(right.farPlane)
        && floatBits(left.orthographicHeight) == floatBits(right.orthographicHeight)
        && HarmonicMIDI::grid::sameCameraLens(left, right);
}

inline void encodeCamera(std::ostringstream& encoded,
                         const HarmonicMIDI::grid::SceneCameraRecord& camera)
{
    encoded << camera.id.value;
    const std::array<float, 14> values {
        camera.transform.translation.x, camera.transform.translation.y,
        camera.transform.translation.z, camera.transform.rotation.x,
        camera.transform.rotation.y, camera.transform.rotation.z,
        camera.transform.rotation.w, camera.transform.scale.x,
        camera.transform.scale.y, camera.transform.scale.z,
        camera.verticalFovRadians, camera.nearPlane, camera.farPlane,
        camera.orthographicHeight
    };
    for (const auto value : values) encoded << ' ' << floatBits(value);
    // A camera with lens fields continues the line; a pinhole ends it as before.
    if (HarmonicMIDI::grid::sceneCameraLensAuthored(camera))
    {
        for (const auto value : { camera.fStop, camera.focusDistance, camera.bladeRotationRadians, camera.apertureRatio })
            encoded << ' ' << floatBits(value);
        encoded << ' ' << static_cast<unsigned int>(camera.apertureBlades);
        // A camera focused on an object continues once more.
        if (camera.focusObject != 0) encoded << ' ' << camera.focusObject;
    }
    encoded << '\n';
}

inline bool decodeCamera(std::istream& input,
                         HarmonicMIDI::grid::SceneCameraRecord& camera)
{
    if (!readUnsigned(input, camera.id.value)) return false;
    std::array<float*, 14> values {
        &camera.transform.translation.x, &camera.transform.translation.y,
        &camera.transform.translation.z, &camera.transform.rotation.x,
        &camera.transform.rotation.y, &camera.transform.rotation.z,
        &camera.transform.rotation.w, &camera.transform.scale.x,
        &camera.transform.scale.y, &camera.transform.scale.z,
        &camera.verticalFovRadians, &camera.nearPlane, &camera.farPlane,
        &camera.orthographicHeight
    };
    for (auto* value : values)
    {
        std::uint32_t bits = 0;
        if (!readUnsigned(input, bits)) return false;
        *value = floatFromBits(bits);
    }
    if (input.peek() == ' ')
    {
        for (float* value : { &camera.fStop, &camera.focusDistance, &camera.bladeRotationRadians, &camera.apertureRatio })
        {
            std::uint32_t bits = 0;
            if (!readUnsigned(input, bits)) return false;
            *value = floatFromBits(bits);
        }
        if (!readUnsigned(input, camera.apertureBlades)) return false;
        if (input.peek() == ' ' && (!readUnsigned(input, camera.focusObject) || camera.focusObject == 0))
            return false;
        if (!HarmonicMIDI::grid::sceneCameraLensAuthored(camera)) return false;
    }
    return validCamera(camera);
}

inline bool validLight(const HarmonicMIDI::grid::SceneLightRecord& light) noexcept
{
    using namespace HarmonicMIDI::grid;
    // This codec has no spectral trailer. Active spectra require sceneSnapshot.
    return light.spectralSource.version == 0 && validSceneSpectralSource(light)
        && light.id.isValid()
        && (light.kind == SceneLightKind::Directional
            || light.kind == SceneLightKind::Point
            || light.kind == SceneLightKind::Spot
            || light.kind == SceneLightKind::Area
            || light.kind == SceneLightKind::Environment)
        && (light.kind != SceneLightKind::Spot
            || (std::isfinite(light.innerConeAngle) && std::isfinite(light.outerConeAngle)
                && light.innerConeAngle >= 0.0f && light.outerConeAngle > light.innerConeAngle
                && light.outerConeAngle <= 1.5707964f))
        && HarmonicMIDI::grid::validSceneLightShapeFields(light)
        && visual3d_detail::validTransform(light.transform)
        && visual3d_detail::finite(light.color)
        && visual3d_detail::nonNegative(light.color)
        && visual3d_detail::finite(light.intensity)
        && visual3d_detail::finite(light.range)
        && light.intensity >= 0.0f && light.range >= 0.0f
        && validSceneAnalyticSky(light) && validSceneEnvironmentImage(light)
        && HarmonicMIDI::grid::validSceneCameraBackground(light)
        && (light.environmentCameraVisible || light.kind == SceneLightKind::Environment);
}

inline bool sameLight(const HarmonicMIDI::grid::SceneLightRecord& left,
                      const HarmonicMIDI::grid::SceneLightRecord& right) noexcept
{
    return left.id == right.id && left.kind == right.kind
        && std::memcmp(&left.transform, &right.transform, sizeof(left.transform)) == 0
        && floatBits(left.color.x) == floatBits(right.color.x)
        && floatBits(left.color.y) == floatBits(right.color.y)
        && floatBits(left.color.z) == floatBits(right.color.z)
        && floatBits(left.intensity) == floatBits(right.intensity)
        && floatBits(left.range) == floatBits(right.range)
        && HarmonicMIDI::grid::sameAnalyticSky(left.sky,right.sky)
        && HarmonicMIDI::grid::sameEnvironmentImage(left.environmentImage,right.environmentImage)
        && left.environmentCameraVisible == right.environmentCameraVisible
        && HarmonicMIDI::grid::sameCameraBackground(left.cameraBackground,right.cameraBackground)
        && HarmonicMIDI::grid::sameLightShapeAndVisibility(left,right);
}

// A light with cones, a size, an area or ray visibility sets bit 7 of the kind
// and carries those fields after the common ones, so other lights encode as before.
inline constexpr unsigned int kLightShapeKindFlag = 0x80u;

inline bool lightNeedsShapeFields(const HarmonicMIDI::grid::SceneLightRecord& light) noexcept
{
    return light.kind == HarmonicMIDI::grid::SceneLightKind::Spot
        || HarmonicMIDI::grid::sceneLightHasShapeOrVisibility(light);
}

inline void encodeLight(std::ostringstream& encoded,
                        const HarmonicMIDI::grid::SceneLightRecord& light)
{
    const bool shaped = lightNeedsShapeFields(light);
    encoded << light.id.value << ' ' << (static_cast<unsigned int>(light.kind) | (shaped ? kLightShapeKindFlag : 0u));
    const std::array<float, 15> values {
        light.transform.translation.x, light.transform.translation.y,
        light.transform.translation.z, light.transform.rotation.x,
        light.transform.rotation.y, light.transform.rotation.z,
        light.transform.rotation.w, light.transform.scale.x,
        light.transform.scale.y, light.transform.scale.z,
        light.color.x, light.color.y, light.color.z, light.intensity, light.range
    };
    for (const auto value : values) encoded << ' ' << floatBits(value);
    if (shaped)
    {
        for (const auto value : { light.innerConeAngle, light.outerConeAngle, light.radius, light.angularRadius,
                                  light.areaSizeX, light.areaSizeY, light.spread })
            encoded << ' ' << floatBits(value);
        encoded << ' ' << static_cast<unsigned int>(light.areaShape) << ' ' << static_cast<unsigned int>(light.visibility);
    }
    encoded << '\n';
}

inline bool decodeLight(std::istream& input,
                        HarmonicMIDI::grid::SceneLightRecord& light)
{
    std::uint8_t kind = 0;
    if (!readUnsigned(input, light.id.value) || !readUnsigned(input, kind)) return false;
    const bool shaped = (kind & kLightShapeKindFlag) != 0;
    light.kind = static_cast<HarmonicMIDI::grid::SceneLightKind>(kind & ~kLightShapeKindFlag);
    std::array<float*, 15> values {
        &light.transform.translation.x, &light.transform.translation.y,
        &light.transform.translation.z, &light.transform.rotation.x,
        &light.transform.rotation.y, &light.transform.rotation.z,
        &light.transform.rotation.w, &light.transform.scale.x,
        &light.transform.scale.y, &light.transform.scale.z,
        &light.color.x, &light.color.y, &light.color.z, &light.intensity, &light.range
    };
    for (auto* value : values)
    {
        std::uint32_t bits = 0;
        if (!readUnsigned(input, bits)) return false;
        *value = floatFromBits(bits);
    }
    if (shaped)
    {
        for (float* value : { &light.innerConeAngle, &light.outerConeAngle, &light.radius, &light.angularRadius,
                              &light.areaSizeX, &light.areaSizeY, &light.spread })
        {
            std::uint32_t bits = 0;
            if (!readUnsigned(input, bits)) return false;
            *value = floatFromBits(bits);
        }
        std::uint8_t shape = 0, visibility = 0;
        if (!readUnsigned(input, shape) || !readUnsigned(input, visibility)
            || shape > static_cast<std::uint8_t>(HarmonicMIDI::grid::SceneAreaLightShape::Ellipse)) return false;
        light.areaShape = static_cast<HarmonicMIDI::grid::SceneAreaLightShape>(shape);
        light.visibility = visibility;
        // The flag is written only when needed, so a redundant one is a forged record.
        if (!lightNeedsShapeFields(light)) return false;
    }
    return validLight(light);
}
inline std::string encodeSurfacePrograms(const geometrysurfacematerial::ObjectPrograms& programs)
{
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << programs.size() << '\n';
    for (const auto& object : programs)
    {
        const auto& program = object.program;
        output << program.sourceMaterial << ' ' << object.instanceIndex << ' ' << program.fields.size();
        for (const auto value : object.appearance.color) output << ' ' << floatBits(value);
        for (const auto value : object.appearance.emission) output << ' ' << floatBits(value);
        output << ' ' << floatBits(object.appearance.metallic) << ' ' << floatBits(object.appearance.roughness) << '\n';
        encodeMaterial(output, program.material, false);
        for (const auto& field : program.fields)
            output << field.sample.materialNode << ' ' << field.sample.target << ' '
                << static_cast<unsigned>(field.sample.reduction) << ' ' << field.sample.index << ' '
                << doubleBits(field.sample.gain) << ' ' << field.objectIndex << ' ' << field.sample.fieldPlan << '\n';
    }
    return output.str();
}

inline bool decodeSurfacePrograms(std::istream& input, geometrysurfacematerial::ObjectPrograms& programs)
{
    std::size_t count = 0;
    if (!readUnsigned(input, count) || count == 0) return false;
    // Grows record by record, so a forged count stops at the first failed read.
    programs.clear();
    for (std::size_t record = 0; record < count; ++record)
    {
        auto& object = programs.emplace_back();
        auto& program = object.program;
        std::size_t fields = 0;
        if (!readUnsigned(input, program.sourceMaterial) || !readUnsigned(input, object.instanceIndex)
            || !readUnsigned(input, fields)) return false;
        const auto readFloat = [&input](float& value) {
            std::uint32_t bits = 0;
            if (!readUnsigned(input, bits)) return false;
            value = floatFromBits(bits); return true;
        };
        for (auto& value : object.appearance.color) if (!readFloat(value)) return false;
        for (auto& value : object.appearance.emission) if (!readFloat(value)) return false;
        if (!readFloat(object.appearance.metallic) || !readFloat(object.appearance.roughness)
            || !decodeMaterial(input, program.material, false)) return false;
        program.fields.resize(fields);
        for (auto& field : program.fields)
        {
            unsigned reduction = 0, objectIndex = 0;
            std::uint64_t gain = 0;
            if (!(input >> field.sample.materialNode >> field.sample.target)
                || !readUnsigned(input, reduction) || reduction > 3 || !readUnsigned(input, field.sample.index)
                || !readUnsigned(input, gain) || !readUnsigned(input, objectIndex) || objectIndex > 1
                || !(input >> field.sample.fieldPlan)) return false;
            field.sample.reduction = static_cast<materialfield::Reduction>(reduction);
            field.sample.gain = doubleFromBits(gain); field.objectIndex = objectIndex != 0;
        }
    }
    return input.get() == '\n';
}
} // namespace detail

inline bool sameMaterialRequest(
    const surfacematerialbinding::ImportedSceneMaterialRequest& left,
    const surfacematerialbinding::ImportedSceneMaterialRequest& right) noexcept
{
    return detail::sameMaterial(left, right);
}

inline bool valid(const Request& request);

// Everything validated while one of these lives on this thread reuses the mesh replays above.
// The outermost scope owns the memo. Keep it short: it holds the plans' values alive.
class ScopedTimelineValidation final
{
public:
    ScopedTimelineValidation()
    {
        if (detail::timelineValidationMemo == nullptr)
        {
            owned.emplace();
            detail::timelineValidationMemo = &*owned;
        }
    }
    ~ScopedTimelineValidation()
    {
        if (owned) detail::timelineValidationMemo = nullptr;
    }
    ScopedTimelineValidation(const ScopedTimelineValidation&) = delete;
    ScopedTimelineValidation& operator=(const ScopedTimelineValidation&) = delete;
private:
    std::optional<detail::TimelineValidationMemo> owned;
};

// A timeline plan's value, from the scope's validation when it already decoded this plan.
inline std::shared_ptr<const videowire::geometry::AdmittedValue> timelineValue(
    const videowire::geometry::TimelineSceneBinding& binding, std::string& error)
{
    if (auto kept = detail::keptTimelineValue(binding)) return kept;
    auto value = videowire::geometry::decodeTimelineScenePlan(binding, error);
    if (!value) return nullptr;
    auto shared = std::make_shared<const videowire::geometry::AdmittedValue>(std::move(*value));
    if (auto* memo = detail::timelineValidationMemo) memo->values.push_back({binding, shared});
    return shared;
}

namespace detail
{
// Only the outer Render request names an engine; composed leaves stay raster.
// Every engine writes every render output, so the Image view and raw export
// selections hold on Path Trace as on Raster and Realtime.
inline bool validRenderEngine(const Request& request) noexcept
{
    const auto rasterLeaf = [](const std::shared_ptr<const Request>& leaf) noexcept
    {
        return !leaf || (leaf->renderEngine == RenderEngine::Raster
            && leaf->pathTraceExportSamples == kDefaultPathTraceExportSamples
            && leaf->pathTracePreviewSamples == kDefaultPathTracePreviewSamples
            && leaf->renderMode == RenderMode::Accurate
            && !leaf->pathTraceDenoise
            && leaf->viewTransform == ViewTransform::Standard
            && leaf->leafIdScheme == HarmonicMIDI::grid::scenecomposition::ImportedLeafIdScheme::Narrow
            && leaf->pathTraceMaxBounces == kDefaultPathTraceMaxBounces
            && leaf->pathTraceVolume == kDefaultPathTraceVolume
            && defaultLightPaths(leaf->pathTraceLightPaths)
            && !leaf->worldVolume
            && leaf->motionBlurShutter == 0
            && leaf->motionBlurPosition == MotionBlurPosition::Centre);
    };
    // The leaf scheme rides on the outer request of a multi-leaf composition.
    if (!HarmonicMIDI::grid::scenecomposition::validImportedLeafIdScheme(
            static_cast<std::uint64_t>(request.leafIdScheme))
        || (request.leafIdScheme != HarmonicMIDI::grid::scenecomposition::ImportedLeafIdScheme::Narrow
            && request.additionalImportedScenes.empty()))
        return false;
    if (request.motionBlurShutter > kMaximumMotionBlurShutter
        || request.motionBlurPosition > MotionBlurPosition::End
        || (request.motionBlurShutter == 0 && request.motionBlurPosition != MotionBlurPosition::Centre))
        return false;
    if (request.renderMode != RenderMode::Accurate && request.renderMode != RenderMode::Fast)
        return false;
    if (static_cast<std::uint8_t>(request.pathTraceDenoise.denoiser)
        > static_cast<std::uint8_t>(PathTraceDenoiser::Oidn))
        return false;
    if (request.viewTransform != ViewTransform::Standard && request.viewTransform != ViewTransform::AgX)
        return false;
    if (!rasterLeaf(request.importedScene)) return false;
    for (const auto& leaf : request.additionalImportedScenes)
        if (!rasterLeaf(leaf)) return false;
    if (!valid(request.pathTraceLightPaths)) return false;
    if (request.worldVolume
        && (!HarmonicMIDI::grid::sceneVolumeActive(*request.worldVolume)
            || !HarmonicMIDI::grid::visual3d_detail::validVolume(*request.worldVolume)
            || request.worldVolume->volumeOnly)) return false;
    if (request.renderEngine != RenderEngine::PathTrace
        && (request.pathTraceMaxBounces != kDefaultPathTraceMaxBounces
            || request.pathTraceVolume != kDefaultPathTraceVolume
            || !defaultLightPaths(request.pathTraceLightPaths)))
        return false;
    switch (request.renderEngine)
    {
        case RenderEngine::Raster:
            return request.pathTraceExportSamples == kDefaultPathTraceExportSamples
                && request.pathTracePreviewSamples == kDefaultPathTracePreviewSamples
                && !request.pathTraceDenoise;
        case RenderEngine::PathTrace:
            return request.pathTraceExportSamples >= 1
                && request.pathTraceExportSamples <= kMaximumPathTraceExportSamples
                && request.pathTracePreviewSamples >= 1
                && request.pathTracePreviewSamples <= kMaximumPathTracePreviewSamples;
        case RenderEngine::Realtime:
            return request.pathTraceExportSamples == kDefaultPathTraceExportSamples
                && request.pathTracePreviewSamples == kDefaultPathTracePreviewSamples
                && !request.pathTraceDenoise;
    }
    return false;
}
} // namespace detail

inline bool sameRequest(const Request& left, const Request& right) noexcept
{
    if (left.renderEngine != right.renderEngine
        || left.renderMode != right.renderMode
        || left.pathTraceExportSamples != right.pathTraceExportSamples
        || left.pathTracePreviewSamples != right.pathTracePreviewSamples
        || left.pathTraceDenoise != right.pathTraceDenoise
        || left.viewTransform != right.viewTransform
        || left.leafIdScheme != right.leafIdScheme
        || left.pathTraceMaxBounces != right.pathTraceMaxBounces
        || left.pathTraceVolume != right.pathTraceVolume
        || !sameLightPaths(left.pathTraceLightPaths, right.pathTraceLightPaths)
        || left.worldVolume.has_value() != right.worldVolume.has_value()
        || (left.worldVolume && !HarmonicMIDI::grid::sameSceneVolume(*left.worldVolume, *right.worldVolume))
        || left.motionBlurShutter != right.motionBlurShutter
        || left.motionBlurPosition != right.motionBlurPosition) return false;
    if ((left.importedParentIntent && !valid(left))
        || (right.importedParentIntent && !valid(right))) return false;
    if (left.importedParentIntent.has_value() != right.importedParentIntent.has_value()
        || (left.importedParentIntent && !sameImportedParentIntent(
            *left.importedParentIntent, *right.importedParentIntent))) return false;
    if ((left.importedScene && (!left.sceneSnapshot || left.importedScene->importedScene
            || left.importedScene->sceneSnapshot))
        || (right.importedScene && (!right.sceneSnapshot || right.importedScene->importedScene
            || right.importedScene->sceneSnapshot))) return false;
    if (left.additionalImportedScenes.size() != right.additionalImportedScenes.size()) return false;
    for (std::size_t i = 0; i < left.additionalImportedScenes.size(); ++i)
    {
        const auto& a = left.additionalImportedScenes[i];
        const auto& b = right.additionalImportedScenes[i];
        if (!a || !b || a->importedScene || b->importedScene || a->sceneSnapshot || b->sceneSnapshot
            || !a->additionalImportedScenes.empty() || !b->additionalImportedScenes.empty()
            || !sameRequest(*a, *b)) return false;
    }
    if (left.glassCaustic.has_value() != right.glassCaustic.has_value()
        || (left.glassCaustic && !sameGlassCaustic(*left.glassCaustic, *right.glassCaustic))) return false;
    if (!visualimportedobjectlocal::same(left.objectLocalTransform, right.objectLocalTransform)
        || !visualimportedobjectlocal::sameList(left.additionalObjectLocalTransforms,
            right.additionalObjectLocalTransforms)) return false;
    if (!importedlightcontrol::same(left.importedLights, right.importedLights)) return false;
    if (left.timelineGeometry.has_value() != right.timelineGeometry.has_value()
        || (left.timelineGeometry && (left.timelineGeometry->object != right.timelineGeometry->object
            || left.timelineGeometry->loweredPlan != right.timelineGeometry->loweredPlan))) return false;
    if (left.harmonicLinks.size() != right.harmonicLinks.size()) return false;
    for (std::size_t i = 0; i < left.harmonicLinks.size(); ++i)
        if (left.harmonicLinks[i].object != right.harmonicLinks[i].object
            || left.harmonicLinks[i].fitToView != right.harmonicLinks[i].fitToView
            || visualharmonicgeometry::encode(left.harmonicLinks[i].mapping)
                != visualharmonicgeometry::encode(right.harmonicLinks[i].mapping)) return false;
    if (left.timelineGeometryFamily.size()!=right.timelineGeometryFamily.size()) return false;
    for (std::size_t i=0;i<left.timelineGeometryFamily.size();++i)
        if (left.timelineGeometryFamily[i].object!=right.timelineGeometryFamily[i].object
            || left.timelineGeometryFamily[i].loweredPlan!=right.timelineGeometryFamily[i].loweredPlan) return false;
    if (detail::encodeSurfacePrograms(left.surfacePrograms) != detail::encodeSurfacePrograms(right.surfacePrograms)) return false;
    if (left.materialField.has_value() != right.materialField.has_value()
        || (left.materialField && !materialfield::same(*left.materialField, *right.materialField))) return false;
    if (left.imageOutput != right.imageOutput
        || left.rawExportMask != right.rawExportMask
        || left.sourceStableId != right.sourceStableId
        || left.renderStableId != right.renderStableId
        || !visualanimationimport::sameAsset(left.asset, right.asset)
        || left.sceneIndex != right.sceneIndex
        || left.objectNodeStableId != right.objectNodeStableId
        || left.objectMeshStableId != right.objectMeshStableId
        || left.sourceCameraNodeId != right.sourceCameraNodeId
        || left.deformation.has_value() != right.deformation.has_value()
        || left.material.has_value() != right.material.has_value()
        || left.diffractionMaterial.has_value()
            != right.diffractionMaterial.has_value()
        || left.camera.has_value() != right.camera.has_value()
        || left.light.has_value() != right.light.has_value()
        || left.noteLight.has_value() != right.noteLight.has_value()
        || left.structuralRevision != right.structuralRevision
        || left.evaluationRevision != right.evaluationRevision
        || static_cast<bool>(left.importedScene) != static_cast<bool>(right.importedScene)
        || (left.sceneSnapshot == nullptr) != (right.sceneSnapshot == nullptr))
        return false;
    return (!left.importedScene || sameRequest(*left.importedScene, *right.importedScene))
        && (!left.deformation
            || (visualanimationoperation::valid(*left.deformation)
                && visualanimationoperation::valid(*right.deformation)
                && visualanimationoperation::encode(*left.deformation)
                    == visualanimationoperation::encode(*right.deformation)))
        && (!left.material
            || detail::sameMaterial(*left.material, *right.material))
        && (!left.diffractionMaterial
            || detail::sameDiffractionMaterial(
                *left.diffractionMaterial, *right.diffractionMaterial))
        && (!left.camera || detail::sameCamera(*left.camera, *right.camera))
        && (!left.light || detail::sameLight(*left.light, *right.light))
        && (!left.noteLight || (left.noteLight->track == right.noteLight->track
            && detail::floatBits(left.noteLight->gain) == detail::floatBits(right.noteLight->gain)
            && detail::floatBits(left.noteLight->pitchDegrees) == detail::floatBits(right.noteLight->pitchDegrees)))
        && (!left.sceneSnapshot
            || left.sceneSnapshot == right.sceneSnapshot
            || sceneaov::detail::encodeScene(*left.sceneSnapshot)
                == sceneaov::detail::encodeScene(*right.sceneSnapshot));
}

// Not noexcept: the multi-leaf branch copies a Request and may throw bad_alloc.
inline bool valid(const Request& request)
{
    if (!detail::validRenderEngine(request)) return false;
    if (!request.additionalImportedScenes.empty())
    {
        // Several leaves join one procedural snapshot. Each leaf must pass the
        // one-leaf composition rules on its own, own its selectors in the leaf
        // ID scheme, and have a distinct source. Parenting, Timeline Geometry3D
        // and more than one selected source camera stay one-leaf features.
        namespace grid = HarmonicMIDI::grid;
        const auto leafCount = request.additionalImportedScenes.size() + 1;
        if (!request.importedScene || !request.sceneSnapshot || leafCount > kMaximumImportedLeaves
            || request.importedParentIntent || request.timelineGeometry
            || !request.timelineGeometryFamily.empty() || request.objectLocalTransform.version != 0
            || request.objectNodeStableId != 0) return false;
        std::array<const Request*, kMaximumImportedLeaves> leaves {};
        leaves[0] = request.importedScene.get();
        for (std::size_t i = 1; i < leafCount; ++i)
        {
            leaves[i] = request.additionalImportedScenes[i - 1].get();
            if (!leaves[i]) return false;
        }
        std::size_t cameraLeaves = 0;
        // materialFrameEndpoint schedules one Frame dependency. Every leaf
        // material texture and every procedural program must name that same
        // endpoint, or a leaf's dependency would be dropped.
        std::optional<surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint> frame;
        const auto sameFrame = [&frame](const auto& endpoint)
        {
            if (!endpoint) return true;
            if (frame && !(*frame == *endpoint)) return false;
            frame = endpoint;
            return true;
        };
        // A diffraction foil stays a one-leaf feature, so only procedural
        // programs and leaf Surface materials remain.
        if (!sameFrame(geometrysurfacematerial::objectFrameEndpoint(request.surfacePrograms))) return false;
        for (std::size_t i = 0; i < leafCount; ++i)
        {
            const auto& leaf = *leaves[i];
            if (leaf.material)
                for (const auto& texture : leaf.material->binding.textures)
                    if (!sameFrame(texture.graphFrame)) return false;
            if (!leaf.additionalImportedScenes.empty() || leaf.diffractionMaterial) return false;
            for (std::size_t j = 0; j < i; ++j)
                if (leaves[j]->sourceStableId == leaf.sourceStableId) return false;
            const auto owns = [&](std::uint64_t id)
            {
                return id == 0 || (id <= std::numeric_limits<std::uint32_t>::max()
                    && grid::scenecomposition::importedLeafOwnsId(static_cast<std::uint32_t>(id), i, leafCount));
            };
            const auto ownsMaterial = [&](std::uint64_t id)
            {
                return id == 0 || (id <= std::numeric_limits<std::uint32_t>::max()
                    && grid::scenecomposition::importedLeafOwnsId(static_cast<std::uint32_t>(id), i, leafCount,
                                                                  request.leafIdScheme));
            };
            // A leaf material names an object, whose node part follows the
            // scheme, and writes a MaterialId literal in the leaf's material
            // range, or zero for an object without a material.
            if (!owns(leaf.objectNodeStableId) || !owns(leaf.sourceCameraNodeId.value_or(0u))
                || (leaf.deformation && !owns(leaf.deformation->pose.nodeStableId))
                || (leaf.material && (!grid::scenecomposition::importedLeafOwnsObjectId(
                        leaf.material->binding.object.value, i, leafCount)
                    || !geometrysurfacematerial::materialIdIsConstant(*leaf.material)
                    || !ownsMaterial(geometrysurfacematerial::materialId(*leaf.material))))) return false;
            if (leaf.sourceCameraNodeId) ++cameraLeaves;
        }
        if (cameraLeaves > 1)
            return false;
        if (request.camera)
        {
            const auto& scene = *request.sceneSnapshot;
            if (cameraLeaves != 1 || scene.cameraCount > scene.cameras.size()) return false;
            const auto* active = grid::visual3d_detail::findById(scene.cameras, scene.cameraCount, scene.activeCamera);
            if (!active || !detail::sameCamera(*active, *request.camera)) return false;
        }
        auto single = request;
        single.additionalImportedScenes.clear();
        single.camera.reset();
        for (std::size_t i = 0; i < leafCount; ++i)
        {
            single.importedScene = i == 0 ? request.importedScene : request.additionalImportedScenes[i - 1];
            if (!valid(single)) return false;
        }
        return true;
    }
    if (!visualimportedobjectlocal::valid(request.objectLocalTransform)
        || !visualimportedobjectlocal::validList(request.objectLocalTransform,
            request.additionalObjectLocalTransforms)
        || (request.objectLocalTransform.version != 0
            && (request.sceneSnapshot || request.importedScene || !request.sceneIndex))) return false;
    if (!detail::validImportedParentRequest(request)) return false;
    if ((request.objectNodeStableId == 0) != (request.objectMeshStableId == 0)
        || (request.objectNodeStableId != 0 && (request.sceneSnapshot || request.importedScene
            || !request.sceneIndex || (request.deformation
                && (request.deformation->pose.nodeStableId != request.objectNodeStableId
                    || request.deformation->pose.meshStableId != request.objectMeshStableId))))) return false;
    if (!request.timelineGeometryFamily.empty()) {
        detail::TimelineFamilyCodecEnvelope envelope;
        std::size_t payloadBytes=0;
        // A request has no wire yet, so its envelope counts no whole-wire bytes.
        if (request.timelineGeometry || !request.sceneSnapshot
            || !detail::timelineFamilyEnvelope(request.timelineGeometryFamily,request.importedScene && request.camera ? 5 : 4,0,envelope)
            || !detail::timelineFamilyPayloadCopyBytes(request,payloadBytes)
            || !videowire::geometry::addArrayBytes(6,payloadBytes,envelope.requestCopyBytes)) return false;
    }
    if (request.importedScene)
    {
        const auto& leaf = *request.importedScene;
        if (!request.sceneSnapshot || leaf.importedScene || leaf.sceneSnapshot
            || !request.asset.id.empty() || request.asset.version != 0 || !request.asset.contentSha256.empty()
            || !request.asset.sourceMediaType.empty() || request.asset.sourceByteSize != 0
            || request.sceneIndex || request.sourceCameraNodeId || request.deformation
            || request.material || request.diffractionMaterial || request.glassCaustic
            || request.light || request.noteLight || !request.importedLights.empty()
            || leaf.renderStableId != request.renderStableId
            || leaf.sourceStableId == request.sourceStableId
            || leaf.structuralRevision != request.structuralRevision
            || leaf.evaluationRevision != request.evaluationRevision
            || leaf.camera || leaf.light || leaf.noteLight
            || !leaf.importedLights.intensityMultipliers.empty()
            || leaf.imageOutput != renderpassoutput::Output::Color || leaf.rawExportMask != 0
            || !valid(leaf)) return false;
        if (request.camera)
        {
            if (request.sceneSnapshot->cameraCount > request.sceneSnapshot->cameras.size()) return false;
            const auto* active = HarmonicMIDI::grid::visual3d_detail::findById(
                request.sceneSnapshot->cameras, request.sceneSnapshot->cameraCount, request.sceneSnapshot->activeCamera);
            if (!leaf.sourceCameraNodeId || !active || !detail::sameCamera(*active, *request.camera)) return false;
        }
        if (leaf.material)
        {
            const auto scene = leaf.sceneIndex ? HarmonicMIDI::grid::visualimportedsceneidentity::sceneId(*leaf.sceneIndex) : std::nullopt;
            if (!scene || leaf.material->scene != *scene || leaf.material->sceneSnapshot
                || leaf.material->sceneRevision != leaf.asset.version
                || leaf.material->structuralRevision != request.structuralRevision
                || leaf.material->evaluationRevision != request.evaluationRevision
                || leaf.material->programRevision != request.structuralRevision
                || leaf.material->binding.surfaceMaterialRevision != request.structuralRevision) return false;
        }
        // A leaf diffraction foil keeps its own scene identity, as a leaf
        // material does; the helper rebinds it to the composed scene. Its
        // imagery Frame is the one Frame the composition schedules.
        if (leaf.diffractionMaterial && leaf.diffractionMaterial->graphFrame)
            if (const auto procedural = geometrysurfacematerial::objectFrameEndpoint(request.surfacePrograms);
                procedural && !(*procedural == *leaf.diffractionMaterial->graphFrame)) return false;
        if (!HarmonicMIDI::grid::scenecomposition::distinctComposedProceduralObjects(*request.sceneSnapshot)) return false;
        auto procedural = request;
        procedural.importedScene.reset();
        procedural.importedParentIntent.reset();
        procedural.camera.reset();
        return valid(procedural);
    }
    if (!importedlightcontrol::valid(request.importedLights)
        || (!request.importedLights.empty() && request.sceneSnapshot)
        || (!request.importedLights.intensityMultipliers.empty() && request.light)) return false;
    if (request.glassCaustic)
    {
        const auto& selection = *request.glassCaustic;
        if (!request.sceneSnapshot || !validGlassCausticScene(selection, *request.sceneSnapshot)) return false;
        const surfacematerialbinding::ImportedSceneMaterialRequest* owner = nullptr;
        if (request.material && request.material->binding.object == selection.glassObject)
            owner = &*request.material;
        for (const auto& object : request.surfacePrograms)
            if (object.program.material.binding.object == selection.glassObject)
            {
                if (owner) return false;
                owner = &object.program.material;
            }
        if (!owner || owner->program.transport != surfacematerial::TransportMode::ClosedVolume
            || geometrysurfacematerial::materialId(*owner) != selection.glassMaterial.value) return false;
    }
    if (!request.timelineGeometryFamily.empty()
        && !detail::timelineCheckedOnce(true,request.timelineGeometryFamily,request.sceneSnapshot,[&]
            { return detail::validTimelineFamily(request.timelineGeometryFamily,*request.sceneSnapshot); })) return false;
    if (!request.harmonicLinks.empty()) {
        if (!request.sceneSnapshot) return false;
        std::uint32_t previous = 0;
        for (const auto& binding : request.harmonicLinks) {
            // Ascending object IDs, so one scene has one canonical encoding.
            if (binding.object.value <= previous || !visualharmonicgeometry::valid(binding.mapping)
                || !HarmonicMIDI::grid::visual3d_detail::findById(request.sceneSnapshot->objects,
                    request.sceneSnapshot->objectCount, binding.object)) return false;
            previous = binding.object.value;
        }
    }
    if (request.timelineGeometry) {
        if (!request.sceneSnapshot) return false;
        if (!detail::timelineCheckedOnce(false,{*request.timelineGeometry},request.sceneSnapshot,[&]
            {
                std::string error;
                // A mesh rebuilt from the wire already decoded this plan.
                if (const auto kept = detail::keptTimelineValue(*request.timelineGeometry))
                    return videowire::geometry::validTimelineSceneBinding(
                        *request.timelineGeometry, *request.sceneSnapshot, *kept, error);
                auto value = videowire::geometry::decodeTimelineScenePlan(*request.timelineGeometry, error);
                if (!value || !videowire::geometry::validTimelineSceneBinding(
                        *request.timelineGeometry, *request.sceneSnapshot, *value, error)) return false;
                detail::keepTimelineValue(*request.timelineGeometry, std::move(*value));
                return true;
            })) return false;
    }
    if (!request.surfacePrograms.empty())
    {
        if (!request.sceneSnapshot || request.material || request.materialField) return false;
        const auto& scene = *request.sceneSnapshot;
        if (!HarmonicMIDI::grid::validateVisual3DScene(scene).valid()) return false;
        std::uint32_t previous = 0;
        std::set<videowire::geometry::StableId> sources;
        std::map<videowire::geometry::StableId, std::string> authoredPrograms;
        std::optional<surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint> frame;
        for (const auto& object : request.surfacePrograms)
        {
            const auto& program = object.program;
            const auto& material = program.material;
            const auto found = std::find_if(scene.objects.begin(), scene.objects.begin() + scene.objectCount,
                [&](const auto& candidate) { return candidate.id == material.binding.object; });
            if ((request.diffractionMaterial && material.binding.object == request.diffractionMaterial->object)
                || material.binding.object.value <= previous || !detail::validMaterial(material)
                || material.vertexModifier || material.scene != scene.id || material.sceneSnapshot != request.sceneSnapshot
                || material.sceneRevision != request.structuralRevision || material.structuralRevision != request.structuralRevision
                || material.programRevision != request.structuralRevision || material.evaluationRevision != request.evaluationRevision
                || material.binding.surfaceMaterialRevision != material.programRevision
                || material.binding.targetKind != surfacematerialbinding::BindingTargetKind::ObjectOverride
                || material.binding.materialSlot.isValid() || found == scene.objects.begin() + scene.objectCount
                || found->indexCount == 0 || !geometrysurfacematerial::materialIdIsConstant(material)
                || found->material.value != geometrysurfacematerial::materialId(material)
                || program.sourceMaterial > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) + 1
                || program.fields.size() > surfacematerialfield::kMaximumBindings) return false;
            previous = material.binding.object.value;
            sources.insert(program.sourceMaterial);
            if (sources.size() > surfacematerialbinding::kMaximumPrograms) return false;
            // One authored source lowers to one program. Source zero is the
            // Render override, which a multi-leaf Scene Compose carries once
            // per imported leaf, each with its own leaf's MaterialId literal
            // and digest, so overrides are not one authored program.
            if (program.sourceMaterial != 0)
            {
                auto authored = object;
                authored.program.material.binding.object = {1}; authored.instanceIndex = 0; authored.appearance = {};
                const auto identity = detail::encodeSurfacePrograms({authored});
                const auto inserted = authoredPrograms.emplace(program.sourceMaterial, identity);
                if (!inserted.second && inserted.first->second != identity) return false;
            }
            for (const auto& field : program.fields) { std::string error; if (!surfacematerialfield::valid(field, error)) return false; }
            for (std::size_t i = 0; i < 4; ++i)
                if (!std::isfinite(object.appearance.color[i]) || !std::isfinite(object.appearance.emission[i])
                    || object.appearance.color[i] < 0 || object.appearance.emission[i] < 0
                    || object.appearance.color[i] > (i == 3 ? 1 : 1000000)
                    || (i == 3 && object.appearance.emission[i] > 1)) return false;
            for (const auto value : {object.appearance.metallic, object.appearance.roughness})
                if (!std::isfinite(value) || (value != -1 && (value < 0 || value > 1))) return false;
            for (const auto& texture : material.binding.textures)
            {
                if (texture.source == surfacematerialbinding::TextureSourceKind::ImportedBaseColor)
                {
                    const auto* imported = HarmonicMIDI::grid::visual3d_detail::findById(scene.materials, scene.materialCount, found->material);
                    if (!imported || !imported->baseColorTexture.isValid()) return false;
                }
                if (!texture.graphFrame) continue;
                if (frame && (frame->node != texture.graphFrame->node || frame->port != texture.graphFrame->port)) return false;
                frame = texture.graphFrame;
            }
        }
        if (request.diffractionMaterial && request.diffractionMaterial->graphFrame && frame
            && !(*request.diffractionMaterial->graphFrame == *frame)) return false;
    }
    if (request.materialField && (!request.diffractionMaterial || request.sceneSnapshot
        || !materialfield::valid(*request.materialField, *request.diffractionMaterial))) return false;
    if (request.noteLight && (!request.light || request.sceneSnapshot || request.diffractionMaterial
        || request.light->kind != HarmonicMIDI::grid::SceneLightKind::Directional
        || request.noteLight->track > 65535 || !std::isfinite(request.noteLight->gain)
        || request.noteLight->gain < 0 || request.noteLight->gain > 8
        || !std::isfinite(request.noteLight->pitchDegrees)
        || request.noteLight->pitchDegrees < 0 || request.noteLight->pitchDegrees > 180))
        return false;
    if (request.rawExportMask > render3dimage::kRawExportMask
        || !render3dimage::supported(request.imageOutput))
        return false;
    const auto fitsSigned = [](std::uint64_t value)
    {
        return value <= static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max());
    };
    if (request.sceneSnapshot != nullptr)
    {
        const auto& scene = *request.sceneSnapshot;
        const auto objectExists = [&scene](HarmonicMIDI::grid::SceneObjectId id)
        {
            for (std::size_t index = 0; index < scene.objectCount; ++index)
                if (scene.objects[index].id == id) return true;
            return false;
        };
        const bool validSurfaceIdentity = !request.material
            || (detail::validMaterial(*request.material)
                && request.material->scene == scene.id
                && request.material->sceneRevision == request.structuralRevision
                && request.material->structuralRevision == request.structuralRevision
                && request.material->evaluationRevision == request.evaluationRevision
                && request.material->programRevision == request.structuralRevision
                && request.material->binding.surfaceMaterialRevision
                    == request.material->programRevision
                && request.material->binding.targetKind
                    == surfacematerialbinding::BindingTargetKind::ObjectOverride
                && !request.material->binding.materialSlot.isValid()
                && objectExists(request.material->binding.object)
                && request.material->sceneSnapshot == request.sceneSnapshot);
        const bool validDiffractionIdentity = !request.diffractionMaterial
            || (detail::validDiffractionMaterial(*request.diffractionMaterial)
                && request.diffractionMaterial->scene == scene.id
                && request.diffractionMaterial->sceneRevision == request.structuralRevision
                && request.diffractionMaterial->structuralRevision
                    == request.structuralRevision
                && request.diffractionMaterial->evaluationRevision
                    == request.evaluationRevision
                && request.diffractionMaterial->materialRevision
                    == request.structuralRevision
                && objectExists(request.diffractionMaterial->object));
        return request.sourceStableId != 0 && request.renderStableId != 0
            && request.sourceStableId != request.renderStableId
            && fitsSigned(request.sourceStableId) && fitsSigned(request.renderStableId)
            && request.structuralRevision != 0 && request.evaluationRevision != 0
            && request.evaluationRevision >= request.structuralRevision
            && HarmonicMIDI::grid::validateVisual3DScene(scene).valid()
            && request.sceneSnapshot->id.value == request.sourceStableId
            && request.asset.id.empty() && !request.sceneIndex && !request.sourceCameraNodeId && !request.deformation
            && !(request.material && request.diffractionMaterial)
            && validSurfaceIdentity && validDiffractionIdentity
            && !request.camera && !request.light;
    }
    const bool validDeformation = !request.deformation
        || (visualanimationoperation::valid(*request.deformation)
            && request.deformation->sourceStableId == request.sourceStableId
            && request.deformation->deformationStableId != request.renderStableId
            && visualanimationimport::sameAsset(
                request.deformation->asset, request.asset)
            && request.deformation->sceneIndex == request.sceneIndex);
    const auto expectedDiffractionScene = request.sceneIndex
        ? HarmonicMIDI::grid::visualimportedsceneidentity::sceneId(
            *request.sceneIndex)
        : std::nullopt;
    const bool validDiffractionIdentity = !request.diffractionMaterial
        || (expectedDiffractionScene
            && request.diffractionMaterial->scene == *expectedDiffractionScene
            && request.diffractionMaterial->sceneRevision == request.asset.version);
    return request.sourceStableId != 0 && request.renderStableId != 0
        && request.sourceStableId != request.renderStableId
        && fitsSigned(request.sourceStableId) && fitsSigned(request.renderStableId)
        && visualmodelasset::validStableAssetId(request.asset.id)
        && request.asset.version != 0 && fitsSigned(request.asset.version)
        && visualmodelasset::validSha256(request.asset.contentSha256)
        && request.asset.sourceMediaType == "model/gltf-binary"
        && request.asset.sourceByteSize != 0
        && request.asset.sourceByteSize <= visualmodelassetpayload::kMaxPayloadBytes
        && (!request.sceneIndex || *request.sceneIndex <= static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max()))
        && (!request.sourceCameraNodeId || (*request.sourceCameraNodeId > 0
            && request.sceneIndex.has_value()))
        && validDeformation
        && !(request.material && request.diffractionMaterial)
        && (!request.material || detail::validMaterial(*request.material))
        && (!request.diffractionMaterial
            || (detail::validDiffractionMaterial(*request.diffractionMaterial)
                && validDiffractionIdentity))
        && (!request.camera || detail::validCamera(*request.camera))
        && (!request.light || detail::validLight(*request.light));
}

inline std::optional<surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint>
materialFrameEndpoint(const Request& request)
{
    const auto* imported = request.importedScene ? request.importedScene.get() : &request;
    if (imported->material)
        for (const auto& texture : imported->material->binding.textures)
            if (texture.graphFrame) return texture.graphFrame;
    if (request.importedScene && imported->diffractionMaterial && imported->diffractionMaterial->graphFrame)
        return imported->diffractionMaterial->graphFrame;
    for (const auto& leaf : request.additionalImportedScenes)
        if (leaf && leaf->material)
            for (const auto& texture : leaf->material->binding.textures)
                if (texture.graphFrame) return texture.graphFrame;
    if (request.diffractionMaterial && request.diffractionMaterial->graphFrame)
        return request.diffractionMaterial->graphFrame;
    return geometrysurfacematerial::objectFrameEndpoint(request.surfacePrograms);
}

// The foil of a one-leaf Scene Compose, bound to the composed scene the
// helper admits. The leaf object keeps its imported ID there, while procedural
// objects move to their composed IDs, so the foil names only its own object.
inline diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest composedLeafDiffractionMaterial(
    diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest foil,
    HarmonicMIDI::grid::Scene3DId composedScene, std::uint64_t structuralRevision)
{
    foil.scene = composedScene;
    foil.sceneRevision = structuralRevision;
    return foil;
}

// Call only after validation. Invalid or conflicting composition identities
// produce no transfer list; a pure generated snapshot needs no model payload.
inline std::vector<visualanimationimport::ExactContentAssetKey> modelAssets(const Request& request)
{
    if (!valid(request)) return {};
    if (request.importedScene)
    {
        std::vector<visualanimationimport::ExactContentAssetKey> assets {request.importedScene->asset};
        for (const auto& leaf : request.additionalImportedScenes)
            if (std::none_of(assets.begin(), assets.end(), [&](const auto& asset)
                    { return visualanimationimport::sameAsset(asset, leaf->asset); }))
                assets.push_back(leaf->asset);
        return assets;
    }
    if (!request.sceneSnapshot) return {request.asset};
    return {};
}

// Moves one composed leaf's object, camera and pose selectors into the
// imported-leaf ID scheme. Leaf 0 and one-leaf compositions keep their raw
// IDs. Refuses, leaving the leaf unchanged, when a selector cannot be mapped.
// A leaf material is not moved here: it is lowered in the leaf scheme, object
// and MaterialId literal, so its structural digest covers the leaf's program.
inline bool remapComposedLeafSelectors(Request& leaf, std::size_t index, std::size_t leafCount)
{
    using HarmonicMIDI::grid::scenecomposition::encodeImportedLeafId;
    const auto map = [&](std::uint64_t raw, std::uint64_t& mapped)
    {
        mapped = 0;
        if (raw == 0) return true;
        if (raw > std::numeric_limits<std::uint32_t>::max()) return false;
        const auto value = encodeImportedLeafId(index, leafCount, static_cast<std::uint32_t>(raw));
        if (!value) return false;
        mapped = *value;
        return true;
    };
    std::uint64_t object = 0, camera = 0, pose = 0;
    if (!map(leaf.objectNodeStableId, object) || !map(leaf.sourceCameraNodeId.value_or(0u), camera)
        || (leaf.deformation && !map(leaf.deformation->pose.nodeStableId, pose))) return false;
    leaf.objectNodeStableId = static_cast<std::uint32_t>(object);
    if (leaf.sourceCameraNodeId) leaf.sourceCameraNodeId = static_cast<std::uint32_t>(camera);
    if (leaf.deformation) leaf.deformation->pose.nodeStableId = pose;
    return true;
}

namespace detail
{
// The stream's doubling buffer and its str() copy would each hold the whole nested
// payload next to the nested string. This holds the nested string and one result.
inline std::string joinEncoded(const std::ostringstream& prefix, const std::string& nested)
{
    const auto head = prefix.str();
    std::string result;
    result.reserve(head.size() + nested.size());
    result.append(head);
    result.append(nested);
    return result;
}
} // namespace detail

inline std::string encode(const Request& request)
{
    const ScopedTimelineValidation oneValidation;
    if (!valid(request)) return {};
    if (request.leafIdScheme != HarmonicMIDI::grid::scenecomposition::ImportedLeafIdScheme::Narrow)
    {
        // V41: scheme and nested length, then the Narrow request bytes.
        auto base = request;
        base.leafIdScheme = HarmonicMIDI::grid::scenecomposition::ImportedLeafIdScheme::Narrow;
        const auto nested = encode(base);
        if (nested.empty()) return {};
        std::string result;
        result.append(kCompiledOperationHeaderV41); result.push_back('\n');
        result.append(std::to_string(static_cast<unsigned>(request.leafIdScheme))); result.push_back(' ');
        result.append(std::to_string(nested.size())); result.push_back('\n');
        result.append(nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (!request.additionalObjectLocalTransforms.empty())
    {
        // V40: the entry count and nested length, then each entry in the V25
        // hex form, then the request bytes with only its primary transform.
        auto base = request;
        base.additionalObjectLocalTransforms.clear();
        const auto nested = encode(base);
        if (nested.empty()) return {};
        std::string result;
        result.append(kCompiledOperationHeaderV40); result.push_back('\n');
        result.append(std::to_string(request.additionalObjectLocalTransforms.size())); result.push_back(' ');
        result.append(std::to_string(nested.size())); result.push_back('\n');
        for (const auto& local : request.additionalObjectLocalTransforms)
        {
            const auto text = detail::encodeObjectLocalTransform(local);
            result.append(text.data(), text.size());
        }
        result.append(nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.pathTraceDenoise.denoiser == PathTraceDenoiser::OptiX
        || request.pathTraceDenoise.denoiser == PathTraceDenoiser::Oidn)
    {
        // V39: the denoiser and nested length, then the request bytes with Auto.
        auto base = request;
        base.pathTraceDenoise = PathTraceDenoiser::Auto;
        const auto nested = encode(base);
        if (nested.empty()) return {};
        std::string result;
        result.append(kCompiledOperationHeaderV39); result.push_back('\n');
        result.append(std::to_string(static_cast<unsigned>(request.pathTraceDenoise.denoiser))); result.push_back(' ');
        result.append(std::to_string(nested.size())); result.push_back('\n');
        result.append(nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.motionBlurShutter != 0)
    {
        // V38: shutter, position and nested length, then the request bytes with
        // motion blur off.
        auto base = request;
        base.motionBlurShutter = 0;
        base.motionBlurPosition = MotionBlurPosition::Centre;
        const auto nested = encode(base);
        if (nested.empty()) return {};
        std::string result;
        result.append(kCompiledOperationHeaderV38); result.push_back('\n');
        result.append(std::to_string(request.motionBlurShutter)); result.push_back(' ');
        result.append(std::to_string(static_cast<unsigned>(request.motionBlurPosition))); result.push_back(' ');
        result.append(std::to_string(nested.size())); result.push_back('\n');
        result.append(nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.worldVolume)
    {
        // V37: the fog's floats as bits, its field and detail, and nested
        // length, then the request bytes without the fog.
        const auto& volume = *request.worldVolume;
        auto base = request;
        base.worldVolume.reset();
        const auto nested = encode(base);
        if (nested.empty()) return {};
        std::string result;
        result.append(kCompiledOperationHeaderV37); result.push_back('\n');
        for (const std::uint64_t value : detail::worldVolumeWireFields(volume))
        {
            result.append(std::to_string(value)); result.push_back(' ');
        }
        result.append(std::to_string(nested.size())); result.push_back('\n');
        result.append(nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (!defaultLightPaths(request.pathTraceLightPaths))
    {
        // V36: the four kind limits, the clamp and filter bits, the caustic
        // flags and nested length, then the request bytes on the default.
        const auto& paths = request.pathTraceLightPaths;
        auto base = request;
        base.pathTraceLightPaths = PathTraceLightPaths {};
        const auto nested = encode(base);
        if (nested.empty()) return {};
        std::string result;
        result.append(kCompiledOperationHeaderV36); result.push_back('\n');
        for (const std::uint64_t value : { std::uint64_t { paths.diffuseBounces },
                 std::uint64_t { paths.glossyBounces }, std::uint64_t { paths.transmissionBounces },
                 std::uint64_t { paths.transparentBounces },
                 std::uint64_t { detail::floatBits(paths.directClamp) },
                 std::uint64_t { detail::floatBits(paths.indirectClamp) },
                 std::uint64_t { detail::floatBits(paths.filterGlossy) },
                 std::uint64_t { lightPathCausticFlags(paths) } })
        {
            result.append(std::to_string(value)); result.push_back(' ');
        }
        result.append(std::to_string(nested.size())); result.push_back('\n');
        result.append(nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.pathTraceVolume != kDefaultPathTraceVolume)
    {
        // V35: the media byte and nested length, then the request bytes on the default.
        auto base = request;
        base.pathTraceVolume = kDefaultPathTraceVolume;
        const auto nested = encode(base);
        if (nested.empty()) return {};
        std::string result;
        result.append(kCompiledOperationHeaderV35); result.push_back('\n');
        result.append(std::to_string(request.pathTraceVolume)); result.push_back(' ');
        result.append(std::to_string(nested.size())); result.push_back('\n');
        result.append(nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.pathTraceMaxBounces != kDefaultPathTraceMaxBounces)
    {
        // V34: the limit and nested length, then the request bytes on the default.
        auto base = request;
        base.pathTraceMaxBounces = kDefaultPathTraceMaxBounces;
        const auto nested = encode(base);
        if (nested.empty()) return {};
        std::string result;
        result.append(kCompiledOperationHeaderV34); result.push_back('\n');
        result.append(std::to_string(request.pathTraceMaxBounces)); result.push_back(' ');
        result.append(std::to_string(nested.size())); result.push_back('\n');
        result.append(nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (!request.harmonicLinks.empty())
    {
        // V33: count and nested length, then each object, fit flag and mapping,
        // then the request bytes without them.
        auto base = request;
        base.harmonicLinks.clear();
        const auto nested = encode(base);
        if (nested.empty()) return {};
        std::string result;
        result.append(kCompiledOperationHeaderV33); result.push_back('\n');
        result.append(std::to_string(request.harmonicLinks.size())); result.push_back(' ');
        result.append(std::to_string(nested.size())); result.push_back('\n');
        for (const auto& binding : request.harmonicLinks)
        {
            const auto mapping = visualharmonicgeometry::encode(binding.mapping);
            if (mapping.empty()) return {};
            result.append(std::to_string(binding.object.value)); result.push_back(' ');
            result.push_back(binding.fitToView ? '1' : '0'); result.push_back(' ');
            result.append(std::to_string(mapping.size())); result.push_back('\n');
            result.append(mapping); result.push_back('\n');
        }
        result.append(nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.viewTransform != ViewTransform::Standard)
    {
        // V32: transform and nested length, then the Standard request bytes.
        // Outermost so every older wrapper keeps its layout, and absent for
        // Standard so every existing graph keeps its exact bytes.
        auto base = request;
        base.viewTransform = ViewTransform::Standard;
        const auto nested = encode(base);
        if (nested.empty()) return {};
        std::string result;
        result.append(kCompiledOperationHeaderV32); result.push_back('\n');
        result.append(std::to_string(static_cast<unsigned>(request.viewTransform))); result.push_back(' ');
        result.append(std::to_string(nested.size())); result.push_back('\n');
        result.append(nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.light && (request.light->environmentImage.version != 0
                          || !request.light->environmentCameraVisible))
    {
        // V31: camera visibility, image version and nested length, then the
        // image identity when one is bound, then the request bytes without
        // either field. Outermost so every older wrapper keeps its layout.
        // Visibility 0 hides the world, 1 shows it, 2 shows a colour background,
        // whose bits take a line of their own, and 3 a transparent one.
        auto base = request;
        base.light->environmentImage = {};
        base.light->environmentCameraVisible = true;
        base.light->cameraBackground = {};
        const auto nested = encode(base);
        if (nested.empty()) return {};
        const auto& image = request.light->environmentImage;
        const auto& background = request.light->cameraBackground;
        using BackgroundMode = HarmonicMIDI::grid::SceneCameraBackgroundMode;
        std::string result;
        result.append(kCompiledOperationHeaderV31); result.push_back('\n');
        result.append(request.light->environmentCameraVisible ? "1 "
            : background.mode == BackgroundMode::Colour ? "2 "
            : background.mode == BackgroundMode::Transparent ? "3 " : "0 ");
        result.append(std::to_string(image.version)); result.push_back(' ');
        result.append(std::to_string(nested.size())); result.push_back('\n');
        if (background.mode == BackgroundMode::Colour)
        {
            result.append(std::to_string(detail::floatBits(background.color.x))); result.push_back(' ');
            result.append(std::to_string(detail::floatBits(background.color.y))); result.push_back(' ');
            result.append(std::to_string(detail::floatBits(background.color.z))); result.push_back('\n');
        }
        if (image.version != 0)
        {
            result.append(std::to_string(image.asset.version)); result.push_back(' ');
            result.append(std::to_string(detail::floatBits(image.orientationDegrees))); result.push_back(' ');
            result.append(std::to_string(detail::floatBits(image.exposureStops))); result.push_back(' ');
            result.append(image.asset.contentSha256); result.push_back(' ');
            result.append(image.asset.id); result.push_back('\n');
        }
        result.append(nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.pathTraceDenoise)
    {
        // V30: nested length, then the same request's bytes with the switch
        // off. Outermost so every older wrapper keeps its layout, and absent
        // when off so every existing graph keeps its exact bytes.
        auto base = request;
        base.pathTraceDenoise = false;
        const auto nested = encode(base);
        if (nested.empty()) return {};
        std::string result;
        result.append(kCompiledOperationHeaderV30); result.push_back('\n');
        result.append(std::to_string(nested.size())); result.push_back('\n');
        result.append(nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.renderMode != RenderMode::Accurate)
    {
        // V29: mode and nested length, then the Accurate request bytes.
        // Outermost so every older wrapper keeps its layout, and absent for
        // Accurate so every existing graph keeps its exact bytes.
        auto base = request;
        base.renderMode = RenderMode::Accurate;
        const auto nested = encode(base);
        if (nested.empty()) return {};
        std::string result;
        result.append(kCompiledOperationHeaderV29); result.push_back('\n');
        result.append(std::to_string(static_cast<unsigned>(request.renderMode))); result.push_back(' ');
        result.append(std::to_string(nested.size())); result.push_back('\n');
        result.append(nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.renderEngine != RenderEngine::Raster)
    {
        // V27: engine, export samples and nested length, then the raster
        // request bytes. Outermost so every older wrapper keeps its layout.
        auto base = request;
        base.renderEngine = RenderEngine::Raster;
        base.pathTraceExportSamples = kDefaultPathTraceExportSamples;
        base.pathTracePreviewSamples = kDefaultPathTracePreviewSamples;
        const auto nested = encode(base);
        if (nested.empty()) return {};
        // V28 adds the preview target and appears only when it differs from
        // the default, so a graph on the default keeps its exact V27 bytes.
        const bool defaultPreview =
            request.pathTracePreviewSamples == kDefaultPathTracePreviewSamples;
        std::string result;
        result.append(defaultPreview ? kCompiledOperationHeaderV27 : kCompiledOperationHeaderV28);
        result.push_back('\n');
        result.append(std::to_string(static_cast<unsigned>(request.renderEngine))); result.push_back(' ');
        result.append(std::to_string(static_cast<unsigned>(request.pathTraceExportSamples))); result.push_back(' ');
        if (!defaultPreview)
        {
            result.append(std::to_string(static_cast<unsigned>(request.pathTracePreviewSamples)));
            result.push_back(' ');
        }
        result.append(std::to_string(nested.size())); result.push_back('\n');
        result.append(nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (!request.additionalImportedScenes.empty())
    {
        // V26: leaf count, camera mode and chunk sizes, an optional Render
        // camera override, then the one-leaf V20 composition of leaf 0 and
        // each further leaf. Leaf chunks omit the outer revisions, as in V20.
        auto base = request;
        base.additionalImportedScenes.clear();
        base.camera.reset();
        const auto nested = encode(base);
        if (nested.empty()) return {};
        std::vector<std::string> leaves;
        leaves.reserve(request.additionalImportedScenes.size());
        for (const auto& leaf : request.additionalImportedScenes)
        {
            leaves.push_back(encode(*leaf));
            if (leaves.back().empty()) return {};
        }
        std::ostringstream encoded;
        encoded << kCompiledOperationHeaderV26 << '\n' << (leaves.size() + 1)
                << (request.camera ? " override " : " plain ") << nested.size();
        for (const auto& leaf : leaves) encoded << ' ' << leaf.size();
        encoded << '\n';
        if (request.camera) detail::encodeCamera(encoded, *request.camera);
        encoded << nested;
        for (const auto& leaf : leaves) encoded << leaf;
        auto result = encoded.str();
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.objectLocalTransform.version == 1)
    {
        auto base = request;
        base.objectLocalTransform = {};
        const auto nested = encode(base);
        if (nested.empty()) return {};
        const auto local = detail::encodeObjectLocalTransform(request.objectLocalTransform);
        const auto nestedLength = std::to_string(nested.size());
        const auto wrapperBytes = kCompiledOperationHeaderV25.size() + local.size() + nestedLength.size() + 3;
        if (nested.size() > kMaximumV8EncodedBytes - wrapperBytes) return {};
        std::string result;
        result.reserve(wrapperBytes + nested.size());
        result.append(kCompiledOperationHeaderV25); result.push_back('\n');
        result.append(local.data(), local.size()); result.push_back(' ');
        result.append(nestedLength); result.push_back('\n'); result.append(nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.importedParentIntent)
    {
        auto base = request;
        base.importedParentIntent.reset();
        const auto nested = encode(base);
        if (nested.empty() || nested.size() > kMaximumV8EncodedBytes - kImportedParentWrapperBytes) return {};
        const auto intent = detail::encodeImportedParentIntent(*request.importedParentIntent);
        std::string result;
        result.reserve(nested.size() + kImportedParentWrapperBytes);
        result.append(kCompiledOperationHeaderV23); result.push_back('\n');
        result.append(intent.data(), intent.size()); result.push_back('\n'); result.append(nested);
        return result;
    }
    std::ostringstream encoded;
    if (request.objectNodeStableId != 0)
    {
        auto base = request;
        base.objectNodeStableId = base.objectMeshStableId = 0;
        const auto nested = encode(base);
        if (nested.empty()) return {};
        encoded << kCompiledOperationHeaderV24 << '\n' << request.objectNodeStableId
                << ' ' << request.objectMeshStableId << ' ' << nested.size() << '\n';
        auto result = detail::joinEncoded(encoded, nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.importedScene && request.camera)
    {
        auto base = request;
        base.camera.reset();
        const auto nested = encode(base);
        if (nested.empty()) return {};
        encoded << kCompiledOperationHeaderV22 << "\noverride " << nested.size() << '\n';
        detail::encodeCamera(encoded, *request.camera);
        auto result = detail::joinEncoded(encoded, nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.importedScene)
    {
        auto procedural = request;
        procedural.importedScene.reset();
        procedural.camera.reset();
        const auto scene = encode(procedural);
        const auto leaf = encode(*request.importedScene);
        if (scene.empty() || leaf.empty()) return {};
        encoded << kCompiledOperationHeaderV20 << '\n' << scene.size() << ' ' << leaf.size() << '\n'
                << scene << leaf;
        auto result = encoded.str();
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.sourceCameraNodeId)
    {
        auto base = request;
        base.sourceCameraNodeId.reset();
        const auto nested = encode(base);
        if (nested.empty()) return {};
        encoded << kCompiledOperationHeaderV22 << "\nsource " << *request.sourceCameraNodeId
                << ' ' << nested.size() << '\n';
        auto result = detail::joinEncoded(encoded, nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.glassCaustic)
    {
        auto base = request;
        base.glassCaustic.reset();
        const auto nested = encode(base);
        if (nested.empty()) return {};
        const auto& selection = *request.glassCaustic;
        const auto& policy = selection.policy;
        encoded << kCompiledOperationHeaderV19 << '\n' << selection.version << ' '
            << selection.sourceLight.value << ' ' << selection.glassObject.value << ' '
            << selection.glassMaterial.value << ' ' << policy.version << ' '
            << detail::floatBits(policy.maximumUnresolvedFraction) << ' '
            << detail::floatBits(policy.minimumBranchFraction) << ' '
            << policy.maximumInterfaces << ' ' << policy.maximumVisits << ' ' << policy.maximumPending
            << '\n';
        auto result = detail::joinEncoded(encoded, nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (!request.importedLights.empty())
    {
        auto base = request;
        base.importedLights = {};
        const auto nested = encode(base);
        if (nested.empty()) return {};
        encoded << kCompiledOperationHeaderV17 << '\n'
            << request.importedLights.selectedDefinition << ' '
            << request.importedLights.intensityMultipliers.size() << '\n';
        for (const auto& [definition, multiplier] : request.importedLights.intensityMultipliers)
            encoded << definition << ' ' << detail::floatBits(multiplier) << '\n';
        auto result = detail::joinEncoded(encoded, nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.light && request.light->sky.version!=0)
    {
        auto base=request;
        base.light->sky={};
        const auto nested=encode(base);
        if (nested.empty()) return {};
        const auto& sky=request.light->sky;
        encoded << kCompiledOperationHeaderV15 << '\n' << sky.version;
        const float values[] {sky.zenith.x,sky.zenith.y,sky.zenith.z,
            sky.horizon.x,sky.horizon.y,sky.horizon.z,sky.ground.x,sky.ground.y,sky.ground.z,
            sky.sunColor.x,sky.sunColor.y,sky.sunColor.z,sky.sunIntensity,
            sky.sunAzimuthDegrees,sky.sunElevationDegrees,sky.sunRadiusDegrees,sky.orientationDegrees};
        for (float value:values) encoded << ' ' << detail::floatBits(value);
        // Clouds trail the line only when authored, so clear skies keep their bytes.
        // The sun halo and cloud thickness follow the clouds, which they write even
        // when the clouds are clear.
        const bool extras=HarmonicMIDI::grid::analyticSkyExtrasAuthored(sky);
        if (extras || HarmonicMIDI::grid::analyticSkyCloudsAuthored(sky))
            for (float value:{sky.cloudCoverage,sky.cloudType,sky.cloudDrift,sky.cloudHeadingDegrees})
                encoded << ' ' << detail::floatBits(value);
        if (extras)
            for (float value:{sky.sunHalo,sky.cloudThickness})
                encoded << ' ' << detail::floatBits(value);
        encoded << '\n';
        auto result = detail::joinEncoded(encoded, nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (!request.timelineGeometryFamily.empty())
    {
        auto base=request;
        base.timelineGeometryFamily.clear();
        const auto nested=encode(base);
        if (nested.empty()) return {};
        const auto digits=[](std::size_t value) {
            std::size_t count=1;
            while (value>=10) { value/=10; ++count; }
            return count;
        };
        std::size_t bytes=nested.size();
        const auto prefix=kCompiledOperationHeaderV21.size()+3
            +digits(request.timelineGeometryFamily.size())+digits(nested.size());
        if (!videowire::geometry::checkedAdd(bytes,prefix,bytes)) return {};
        for (const auto& binding : request.timelineGeometryFamily)
            if (!videowire::geometry::checkedAdd(bytes,binding.loweredPlan.size()
                    +digits(binding.object.value)+digits(binding.loweredPlan.size())+3,bytes)
                || bytes>kMaximumV8EncodedBytes) return {};
        encoded << kCompiledOperationHeaderV21 << '\n' << request.timelineGeometryFamily.size()
            << ' ' << nested.size() << '\n';
        for (const auto& binding : request.timelineGeometryFamily)
            encoded << binding.object.value << ' ' << binding.loweredPlan.size() << '\n' << binding.loweredPlan << '\n';
        auto result = detail::joinEncoded(encoded, nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.timelineGeometry)
    {
        auto base = request;
        base.timelineGeometry.reset();
        const detail::ScopedReplayedTimelineMesh replayed(*request.timelineGeometry);
        const auto nested = encode(base);
        if (nested.empty()) return {};
        encoded << kCompiledOperationHeaderV14 << '\n' << request.timelineGeometry->object.value << ' '
            << request.timelineGeometry->loweredPlan.size() << '\n'
            << request.timelineGeometry->loweredPlan << '\n';
        auto result = detail::joinEncoded(encoded, nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (!request.surfacePrograms.empty())
    {
        auto base = request;
        base.surfacePrograms.clear();
        const auto nested = encode(base);
        if (nested.empty()) return {};
        encoded << (request.diffractionMaterial ? kCompiledOperationHeaderV18 : kCompiledOperationHeaderV13)
                << '\n' << detail::encodeSurfacePrograms(request.surfacePrograms);
        auto result = detail::joinEncoded(encoded, nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.materialField)
    {
        auto base = request;
        base.materialField.reset();
        const auto nested = encode(base);
        if (nested.empty()) return {};
        const auto& field = *request.materialField;
        std::uint64_t gainBits = 0;
        std::memcpy(&gainBits, &field.gain, sizeof(gainBits));
        encoded << kCompiledOperationHeaderV12 << '\n' << field.materialNode << ' ' << field.target << ' '
                << static_cast<unsigned>(field.reduction) << ' ' << field.index << ' '
                << gainBits << '\n'
                << field.fieldPlan << '\n';
        auto result = detail::joinEncoded(encoded, nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.rawExportMask != 0)
    {
        auto withoutExports = request;
        withoutExports.rawExportMask = 0;
        const auto nested = encode(withoutExports);
        if (nested.empty()) return {};
        encoded << kCompiledOperationHeaderV11 << '\n' << request.rawExportMask << '\n';
        auto result = detail::joinEncoded(encoded, nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.noteLight)
    {
        auto constantLightRequest = request;
        constantLightRequest.noteLight.reset();
        const auto nested = encode(constantLightRequest);
        if (nested.empty()) return {};
        encoded << kCompiledOperationHeaderV10 << '\n' << request.noteLight->track << ' '
                << detail::floatBits(request.noteLight->gain) << ' '
                << detail::floatBits(request.noteLight->pitchDegrees) << '\n';
        auto result = detail::joinEncoded(encoded, nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.imageOutput != renderpassoutput::Output::Color)
    {
        auto colorRequest = request;
        colorRequest.imageOutput = renderpassoutput::Output::Color;
        const auto nested = encode(colorRequest);
        if (nested.empty()) return {};
        encoded << kCompiledOperationHeaderV9 << '\n'
                << static_cast<unsigned>(request.imageOutput) << '\n';
        auto result = detail::joinEncoded(encoded, nested);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    if (request.sceneSnapshot != nullptr)
    {
        const auto sceneToken = sceneaov::detail::sceneTokenFor(*request.sceneSnapshot,
            detail::replayedTimelineMesh ? detail::replayedTimelineMesh->object : HarmonicMIDI::grid::SceneObjectId {});
        const auto& scene = *sceneToken;
        encoded << (request.material && request.material->program.version == surfacematerial::kSpectralWireVersion
                    ? kCompiledOperationHeaderV16 : kCompiledOperationHeaderV8) << '\n'
                << "scene " << request.sourceStableId << ' ' << request.renderStableId << ' '
                << request.structuralRevision << ' ' << request.evaluationRevision << ' ';
        const auto head = encoded.str();
        encoded.str(std::string{});
        encoded << '\n';
        const bool hasVertex = request.material && request.material->vertexModifier;
        encoded << (request.material ? 1 : 0) << ' ' << (hasVertex ? 1 : 0) << ' '
                << (request.diffractionMaterial ? 1 : 0) << '\n';
        if (request.material) detail::encodeMaterial(encoded, *request.material, hasVertex);
        if (request.diffractionMaterial)
            detail::encodeDiffractionMaterial(encoded, *request.diffractionMaterial);
        const auto tail = encoded.str();
        std::string result;
        result.reserve(head.size() + scene.size() + tail.size());
        result.append(head); result.append(scene); result.append(tail);
        if (result.size() > kMaximumV8EncodedBytes) return {};
        return result;
    }
    const bool hasVertex = request.material && request.material->vertexModifier;
    if (request.light || (request.material && request.material->program.version == surfacematerial::kSpectralWireVersion))
    {
        detail::encodeBase(encoded, request.material && request.material->program.version == surfacematerial::kSpectralWireVersion
            ? kCompiledOperationHeaderV16 : kCompiledOperationHeaderV8, request);
        encoded << (request.deformation ? 1 : 0) << ' '
                << (request.material ? 1 : 0) << ' ' << (hasVertex ? 1 : 0) << ' '
                << (request.diffractionMaterial ? 1 : 0) << ' '
                << (request.camera ? 1 : 0) << ' ' << (request.light ? 1 : 0) << '\n';
        if (request.camera) detail::encodeCamera(encoded, *request.camera);
        if (request.light) detail::encodeLight(encoded, *request.light);
        if (request.deformation)
        {
            const auto deformation = visualanimationoperation::encode(*request.deformation);
            if (deformation.empty() || deformation.size() > kMaximumDeformationBytes)
                return {};
            encoded << deformation.size() << '\n';
            encoded.write(deformation.data(), static_cast<std::streamsize>(deformation.size()));
        }
        if (request.material) detail::encodeMaterial(encoded, *request.material, hasVertex);
        if (request.diffractionMaterial)
            detail::encodeDiffractionMaterial(encoded, *request.diffractionMaterial);
        auto result = encoded.str();
        if (result.size() > (hasVertex ? kMaximumV8EncodedBytes : kMaximumV4EncodedBytes)) return {};
        return result;
    }
    if (request.diffractionMaterial)
    {
        detail::encodeBase(encoded, kCompiledOperationHeaderV7, request);
        encoded << (request.deformation ? 1 : 0) << ' '
                << 1 << ' ' << (request.camera ? 1 : 0) << '\n';
        if (request.camera) detail::encodeCamera(encoded, *request.camera);
        if (request.deformation)
        {
            const auto deformation = visualanimationoperation::encode(*request.deformation);
            if (deformation.empty() || deformation.size() > kMaximumDeformationBytes)
                return {};
            encoded << deformation.size() << '\n';
            encoded.write(deformation.data(),
                          static_cast<std::streamsize>(deformation.size()));
        }
        detail::encodeDiffractionMaterial(encoded, *request.diffractionMaterial);
        auto result = encoded.str();
        if (result.size() > kMaximumV4EncodedBytes) return {};
        return result;
    }
    if (request.camera)
    {
        detail::encodeBase(encoded, kCompiledOperationHeaderV5, request);
        encoded << (request.deformation ? 1 : 0) << ' '
                << (request.material ? 1 : 0) << ' ' << (hasVertex ? 1 : 0) << '\n';
        detail::encodeCamera(encoded, *request.camera);
        if (request.deformation)
        {
            const auto deformation = visualanimationoperation::encode(*request.deformation);
            if (deformation.empty() || deformation.size() > kMaximumDeformationBytes)
                return {};
            encoded << deformation.size() << '\n';
            encoded.write(deformation.data(), static_cast<std::streamsize>(deformation.size()));
        }
        if (request.material) detail::encodeMaterial(encoded, *request.material, hasVertex);
        auto result = encoded.str();
        if (result.size() > (hasVertex ? kMaximumV8EncodedBytes : kMaximumV4EncodedBytes)) return {};
        return result;
    }
    detail::encodeBase(encoded,
                       hasVertex ? kCompiledOperationHeaderV4
                       : request.deformation ? kCompiledOperationHeaderV3
                       : request.material ? kCompiledOperationHeaderV2
                                          : kCompiledOperationHeader,
                       request);
    if (hasVertex)
    {
        std::string deformation;
        if (request.deformation)
            deformation = visualanimationoperation::encode(*request.deformation);
        if (request.deformation
            && (deformation.empty() || deformation.size() > kMaximumDeformationBytes))
            return {};
        encoded << (request.deformation ? 1 : 0) << ' '
                << deformation.size() << '\n';
        if (!deformation.empty())
            encoded.write(deformation.data(), static_cast<std::streamsize>(deformation.size()));
    }
    else if (request.deformation)
    {
        const auto deformation = visualanimationoperation::encode(
            *request.deformation);
        if (deformation.empty()
            || deformation.size() > kMaximumDeformationBytes)
            return {};
        encoded << (request.material ? 1 : 0) << ' '
                << deformation.size() << '\n';
        encoded.write(deformation.data(),
                      static_cast<std::streamsize>(deformation.size()));
    }
    if (request.material)
        detail::encodeMaterial(encoded, *request.material, hasVertex);
    auto result = encoded.str();
    const auto maximum = hasVertex ? kMaximumV8EncodedBytes
                       : request.deformation ? kMaximumV3EncodedBytes
                       : request.material ? kMaximumV2EncodedBytes
                                          : kMaximumEncodedBytes;
    if (result.size() > maximum) return {};
    return result;
}

// Re-encodes only a validated legacy value. Production authors always emit V7 or newer.
inline std::string encodeLegacyV6(const Request& request)
{
    if (!request.diffractionMaterial
        || !detail::validLegacyV6DiffractionMaterial(*request.diffractionMaterial)
        || !diffractionmaterial::sameLightingDescription(
            request.diffractionMaterial->lighting,
            diffractionmaterial::makeReferenceLighting())
        || request.material
        || request.renderMode != RenderMode::Accurate
        || request.pathTraceDenoise
        || request.viewTransform != ViewTransform::Standard
        || request.leafIdScheme != HarmonicMIDI::grid::scenecomposition::ImportedLeafIdScheme::Narrow
        || request.pathTraceMaxBounces != kDefaultPathTraceMaxBounces
        || request.pathTraceVolume != kDefaultPathTraceVolume
        || !defaultLightPaths(request.pathTraceLightPaths)
        || request.worldVolume
        || request.motionBlurShutter != 0)
        return {};
    std::ostringstream encoded;
    detail::encodeBase(encoded, kCompiledOperationHeaderV6, request);
    encoded << (request.deformation ? 1 : 0) << ' '
            << 1 << ' ' << (request.camera ? 1 : 0) << '\n';
    if (request.camera) detail::encodeCamera(encoded, *request.camera);
    if (request.deformation)
    {
        const auto deformation = visualanimationoperation::encode(*request.deformation);
        if (deformation.empty() || deformation.size() > kMaximumDeformationBytes)
            return {};
        encoded << deformation.size() << '\n';
        encoded.write(deformation.data(), static_cast<std::streamsize>(deformation.size()));
    }
    detail::encodeLegacyV6DiffractionMaterial(encoded, *request.diffractionMaterial);
    auto result = encoded.str();
    if (result.size() > kMaximumV4EncodedBytes) return {};
    return result;
}

using LegacyV6DiffractionAdmission = bool (*)(
    diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest&);

namespace detail
{
// Reads the caller's bytes in place. Each stream envelope stays open while its
// nested envelope decodes, so a copy here holds one whole payload per level.
class ViewBuffer final : public std::streambuf
{
public:
    explicit ViewBuffer(std::string_view text)
    {
        auto* begin = const_cast<char*>(text.data());
        setg(begin, begin, begin + text.size());
    }

protected:
    pos_type seekoff(off_type offset, std::ios_base::seekdir direction,
                     std::ios_base::openmode which) override
    {
        if ((which & std::ios_base::in) == 0) return pos_type(off_type(-1));
        const off_type base = direction == std::ios_base::beg ? 0
            : direction == std::ios_base::cur ? gptr() - eback() : egptr() - eback();
        const off_type target = base + offset;
        if (target < 0 || target > egptr() - eback()) return pos_type(off_type(-1));
        setg(eback(), eback() + target, egptr());
        return pos_type(target);
    }
    pos_type seekpos(pos_type position, std::ios_base::openmode which) override
    {
        return seekoff(off_type(position), std::ios_base::beg, which);
    }
};

// The outermost envelope re-encodes all of its bytes, which contain every nested
// envelope's bytes, so a noncanonical nested envelope fails there. Checking at each
// level encoded the whole scene once per level.
inline thread_local int decodeDepth = 0;
struct DecodeDepth final
{
    DecodeDepth() noexcept { ++decodeDepth; }
    ~DecodeDepth() { --decodeDepth; }
    DecodeDepth(const DecodeDepth&) = delete;
    DecodeDepth& operator=(const DecodeDepth&) = delete;
};
inline bool canonicalBytes(const Request& request, std::string_view encoded)
{
    return decodeDepth > 1 || encode(request) == encoded;
}

inline bool decodeRequest(std::string_view encoded, Request& output,
                          LegacyV6DiffractionAdmission migrateLegacyV6,
                          bool allowComposition)
{
    const DecodeDepth depth;
    if (encoded.empty() || encoded.size() > kMaximumV8EncodedBytes
        || encoded.find('\r') != std::string_view::npos)
        return false;
    // V41 wraps every other envelope, V40 included, and appears only with Wide
    // leaves. An older envelope that nests it fails the canonical re-encode.
    if (encoded.size() > kCompiledOperationHeaderV41.size()
        && encoded.substr(0, kCompiledOperationHeaderV41.size()) == kCompiledOperationHeaderV41
        && encoded[kCompiledOperationHeaderV41.size()] == '\n')
    {
        auto wire = encoded.substr(kCompiledOperationHeaderV41.size() + 1);
        std::uint64_t scheme = 0, nestedBytes = 0;
        if (!readFamilyUnsigned(wire, scheme, ' ') || !readFamilyUnsigned(wire, nestedBytes, '\n')
            || !HarmonicMIDI::grid::scenecomposition::validImportedLeafIdScheme(scheme)
            || scheme == static_cast<std::uint64_t>(HarmonicMIDI::grid::scenecomposition::ImportedLeafIdScheme::Narrow)
            || nestedBytes != wire.size()
            || containsRequestHeader(wire, kCompiledOperationHeaderV41)) return false;
        Request request;
        if (!decodeRequest(wire, request, migrateLegacyV6, allowComposition)) return false;
        if (request.leafIdScheme != HarmonicMIDI::grid::scenecomposition::ImportedLeafIdScheme::Narrow) return false;
        request.leafIdScheme = static_cast<HarmonicMIDI::grid::scenecomposition::ImportedLeafIdScheme>(scheme);
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    // V40 wraps every other envelope and appears only when several imported
    // objects move. An older envelope that nests it fails the canonical re-encode.
    if (encoded.size() > kCompiledOperationHeaderV40.size()
        && encoded.substr(0, kCompiledOperationHeaderV40.size()) == kCompiledOperationHeaderV40
        && encoded[kCompiledOperationHeaderV40.size()] == '\n')
    {
        auto wire = encoded.substr(kCompiledOperationHeaderV40.size() + 1);
        std::uint64_t count = 0, nestedBytes = 0;
        if (!readFamilyUnsigned(wire, count, ' ') || !readFamilyUnsigned(wire, nestedBytes, '\n')
            || count == 0 || count > wire.size() / kObjectLocalTransformHexBytes) return false;
        std::vector<visualimportedobjectlocal::ObjectLocalTransformV1> others;
        others.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i)
        {
            visualimportedobjectlocal::ObjectLocalTransformV1 local;
            if (!decodeObjectLocalTransform(wire.substr(0, kObjectLocalTransformHexBytes), local)
                || local.version != 1) return false;
            others.push_back(local);
            wire.remove_prefix(kObjectLocalTransformHexBytes);
        }
        if (nestedBytes != wire.size() || containsRequestHeader(wire, kCompiledOperationHeaderV40)) return false;
        Request request;
        if (!decodeRequest(wire, request, migrateLegacyV6, allowComposition)
            || !request.additionalObjectLocalTransforms.empty()) return false;
        request.additionalObjectLocalTransforms = std::move(others);
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    // V39 wraps V35-V38 and appears only with Path Trace and an OptiX or OIDN
    // denoiser, since its nested request carries Auto through V30.
    if (encoded.size() > kCompiledOperationHeaderV39.size()
        && encoded.substr(0, kCompiledOperationHeaderV39.size()) == kCompiledOperationHeaderV39
        && encoded[kCompiledOperationHeaderV39.size()] == '\n')
    {
        auto wire = encoded.substr(kCompiledOperationHeaderV39.size() + 1);
        std::uint64_t denoiser = 0, nestedBytes = 0;
        if (!readFamilyUnsigned(wire, denoiser, ' ') || !readFamilyUnsigned(wire, nestedBytes, '\n')
            || (denoiser != static_cast<std::uint64_t>(PathTraceDenoiser::OptiX)
                && denoiser != static_cast<std::uint64_t>(PathTraceDenoiser::Oidn))
            || nestedBytes != wire.size()
            || containsRequestHeader(wire, kCompiledOperationHeaderV39)) return false;
        Request request;
        if (!decodeRequest(wire, request, migrateLegacyV6, allowComposition)
            || request.pathTraceDenoise.denoiser != PathTraceDenoiser::Auto) return false;
        request.pathTraceDenoise = static_cast<PathTraceDenoiser>(denoiser);
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    // V38 wraps V35-V37 and appears only with motion blur on. An older envelope
    // that nests it fails the canonical re-encode, which puts V38 first.
    if (encoded.size() > kCompiledOperationHeaderV38.size()
        && encoded.substr(0, kCompiledOperationHeaderV38.size()) == kCompiledOperationHeaderV38
        && encoded[kCompiledOperationHeaderV38.size()] == '\n')
    {
        auto wire = encoded.substr(kCompiledOperationHeaderV38.size() + 1);
        std::uint64_t shutter = 0, position = 0, nestedBytes = 0;
        if (!readFamilyUnsigned(wire, shutter, ' ') || !readFamilyUnsigned(wire, position, ' ')
            || !readFamilyUnsigned(wire, nestedBytes, '\n')
            || shutter == 0 || shutter > kMaximumMotionBlurShutter
            || position > static_cast<std::uint64_t>(MotionBlurPosition::End)
            || nestedBytes != wire.size()
            || containsRequestHeader(wire, kCompiledOperationHeaderV38)) return false;
        Request request;
        if (!decodeRequest(wire, request, migrateLegacyV6, allowComposition)
            || request.motionBlurShutter != 0) return false;
        request.motionBlurShutter = static_cast<std::uint16_t>(shutter);
        request.motionBlurPosition = static_cast<MotionBlurPosition>(position);
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    // V37 wraps every other envelope and appears only with scene fog.
    if (encoded.size() > kCompiledOperationHeaderV37.size()
        && encoded.substr(0, kCompiledOperationHeaderV37.size()) == kCompiledOperationHeaderV37
        && encoded[kCompiledOperationHeaderV37.size()] == '\n')
    {
        auto wire = encoded.substr(kCompiledOperationHeaderV37.size() + 1);
        std::array<std::uint64_t, detail::kWorldVolumeWireFields> fields {};
        std::uint64_t nestedBytes = 0;
        for (auto& field : fields)
            if (!readFamilyUnsigned(wire, field, ' ')) return false;
        if (!readFamilyUnsigned(wire, nestedBytes, '\n') || nestedBytes != wire.size()
            || containsRequestHeader(wire, kCompiledOperationHeaderV37)) return false;
        HarmonicMIDI::grid::SceneVolume volume;
        if (!detail::worldVolumeFromWireFields(fields, volume)) return false;
        Request request;
        if (!decodeRequest(wire, request, migrateLegacyV6, allowComposition) || request.worldVolume) return false;
        request.worldVolume = volume;
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    // V36 wraps every other envelope and appears only with Path Trace. An older
    // envelope that nests it fails the canonical re-encode, which puts V36 first.
    if (encoded.size() > kCompiledOperationHeaderV36.size()
        && encoded.substr(0, kCompiledOperationHeaderV36.size()) == kCompiledOperationHeaderV36
        && encoded[kCompiledOperationHeaderV36.size()] == '\n')
    {
        auto wire = encoded.substr(kCompiledOperationHeaderV36.size() + 1);
        std::array<std::uint64_t, 8> fields {};
        std::uint64_t nestedBytes = 0;
        for (auto& field : fields)
            if (!readFamilyUnsigned(wire, field, ' ')) return false;
        if (!readFamilyUnsigned(wire, nestedBytes, '\n')
            || fields[0] > kMaximumPathTraceKindBounces || fields[1] > kMaximumPathTraceKindBounces
            || fields[2] > kMaximumPathTraceKindBounces || fields[3] > kMaximumPathTraceKindBounces
            || fields[4] > 0xffffffffu || fields[5] > 0xffffffffu || fields[6] > 0xffffffffu
            || fields[7] > (kLightPathReflectiveCaustics | kLightPathRefractiveCaustics)
            || nestedBytes != wire.size()
            || containsRequestHeader(wire, kCompiledOperationHeaderV36)
            || containsRequestHeader(wire, kCompiledOperationHeaderV37)) return false;
        Request request;
        if (!decodeRequest(wire, request, migrateLegacyV6, allowComposition)
            || !defaultLightPaths(request.pathTraceLightPaths)) return false;
        auto& paths = request.pathTraceLightPaths;
        paths.diffuseBounces = static_cast<std::uint16_t>(fields[0]);
        paths.glossyBounces = static_cast<std::uint16_t>(fields[1]);
        paths.transmissionBounces = static_cast<std::uint16_t>(fields[2]);
        paths.transparentBounces = static_cast<std::uint16_t>(fields[3]);
        paths.directClamp = detail::floatFromBits(static_cast<std::uint32_t>(fields[4]));
        paths.indirectClamp = detail::floatFromBits(static_cast<std::uint32_t>(fields[5]));
        paths.filterGlossy = detail::floatFromBits(static_cast<std::uint32_t>(fields[6]));
        paths.reflectiveCaustics = (fields[7] & kLightPathReflectiveCaustics) != 0;
        paths.refractiveCaustics = (fields[7] & kLightPathRefractiveCaustics) != 0;
        if (defaultLightPaths(paths) || !valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    // V35 wraps every other envelope and appears only with Path Trace. An older
    // envelope that nests it fails the canonical re-encode, which puts V35 first.
    if (encoded.size() > kCompiledOperationHeaderV35.size()
        && encoded.substr(0, kCompiledOperationHeaderV35.size()) == kCompiledOperationHeaderV35
        && encoded[kCompiledOperationHeaderV35.size()] == '\n')
    {
        auto wire = encoded.substr(kCompiledOperationHeaderV35.size() + 1);
        std::uint64_t volume = 0, nestedBytes = 0;
        if (!readFamilyUnsigned(wire, volume, ' ') || !readFamilyUnsigned(wire, nestedBytes, '\n')
            || volume > 0xff || volume == kDefaultPathTraceVolume
            || nestedBytes != wire.size()
            || containsRequestHeader(wire, kCompiledOperationHeaderV35)
            || containsRequestHeader(wire, kCompiledOperationHeaderV36)
            || containsRequestHeader(wire, kCompiledOperationHeaderV37)) return false;
        Request request;
        if (!decodeRequest(wire, request, migrateLegacyV6, allowComposition)) return false;
        request.pathTraceVolume = static_cast<std::uint8_t>(volume);
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    // V34 wraps every other envelope and appears only with Path Trace. An older
    // envelope that nests it fails the canonical re-encode, which puts V34 first.
    if (encoded.size() > kCompiledOperationHeaderV34.size()
        && encoded.substr(0, kCompiledOperationHeaderV34.size()) == kCompiledOperationHeaderV34
        && encoded[kCompiledOperationHeaderV34.size()] == '\n')
    {
        auto wire = encoded.substr(kCompiledOperationHeaderV34.size() + 1);
        std::uint64_t bounces = 0, nestedBytes = 0;
        if (!readFamilyUnsigned(wire, bounces, ' ') || !readFamilyUnsigned(wire, nestedBytes, '\n')
            || bounces > kMaximumPathTraceMaxBounces
            || nestedBytes != wire.size()
            || containsRequestHeader(wire, kCompiledOperationHeaderV34)
            || containsRequestHeader(wire, kCompiledOperationHeaderV35)
            || containsRequestHeader(wire, kCompiledOperationHeaderV36)
            || containsRequestHeader(wire, kCompiledOperationHeaderV37)) return false;
        Request request;
        if (!decodeRequest(wire, request, migrateLegacyV6, allowComposition)) return false;
        request.pathTraceMaxBounces = static_cast<std::uint16_t>(bounces);
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    // V33 wraps every older envelope. An older envelope that nests it fails the
    // canonical re-encode, which always puts V33 first.
    if (encoded.size() > kCompiledOperationHeaderV33.size()
        && encoded.substr(0, kCompiledOperationHeaderV33.size()) == kCompiledOperationHeaderV33
        && encoded[kCompiledOperationHeaderV33.size()] == '\n')
    {
        auto wire = encoded.substr(kCompiledOperationHeaderV33.size() + 1);
        std::uint64_t count = 0, nestedBytes = 0;
        if (!readFamilyUnsigned(wire, count, ' ') || !readFamilyUnsigned(wire, nestedBytes, '\n')
            || count == 0 || nestedBytes > wire.size()) return false;
        std::vector<videowire::geometry::HarmonicLinkSceneBinding> bindings;
        for (std::uint64_t index = 0; index < count; ++index)
        {
            std::uint64_t object = 0, fit = 0, size = 0;
            if (!readFamilyUnsigned(wire, object, ' ') || !readFamilyUnsigned(wire, fit, ' ')
                || !readFamilyUnsigned(wire, size, '\n') || fit > 1
                || object > std::numeric_limits<std::uint32_t>::max()
                || size >= wire.size() || wire[size] != '\n') return false;
            videowire::geometry::HarmonicLinkSceneBinding binding;
            binding.object.value = static_cast<std::uint32_t>(object);
            binding.fitToView = fit == 1;
            if (!visualharmonicgeometry::decode(wire.substr(0, size), binding.mapping)) return false;
            wire.remove_prefix(size + 1);
            bindings.push_back(std::move(binding));
        }
        if (nestedBytes != wire.size() || containsRequestHeader(wire, kCompiledOperationHeaderV33)) return false;
        Request request;
        if (!decodeRequest(wire, request, migrateLegacyV6, allowComposition)) return false;
        request.harmonicLinks = std::move(bindings);
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    // V32 wraps every other envelope, on any engine. An older envelope that nests it
    // fails the canonical re-encode.
    if (encoded.size() > kCompiledOperationHeaderV32.size()
        && encoded.substr(0, kCompiledOperationHeaderV32.size()) == kCompiledOperationHeaderV32
        && encoded[kCompiledOperationHeaderV32.size()] == '\n')
    {
        auto wire = encoded.substr(kCompiledOperationHeaderV32.size() + 1);
        std::uint64_t transform = 0, nestedBytes = 0;
        if (!readFamilyUnsigned(wire, transform, ' ') || !readFamilyUnsigned(wire, nestedBytes, '\n')
            || transform != static_cast<std::uint64_t>(ViewTransform::AgX)
            || nestedBytes != wire.size()
            || containsRequestHeader(wire, kCompiledOperationHeaderV32)) return false;
        Request request;
        if (!decodeRequest(wire, request, migrateLegacyV6, allowComposition)) return false;
        request.viewTransform = ViewTransform::AgX;
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    // V31 wraps every older envelope. An older envelope that nests it fails the
    // canonical re-encode, which always puts V31 first.
    if (encoded.size() > kCompiledOperationHeaderV31.size()
        && encoded.substr(0, kCompiledOperationHeaderV31.size()) == kCompiledOperationHeaderV31
        && encoded[kCompiledOperationHeaderV31.size()] == '\n')
    {
        auto wire = encoded.substr(kCompiledOperationHeaderV31.size() + 1);
        std::uint64_t cameraVisible = 0, imageVersion = 0, nestedBytes = 0;
        if (!readFamilyUnsigned(wire, cameraVisible, ' ') || !readFamilyUnsigned(wire, imageVersion, ' ')
            || !readFamilyUnsigned(wire, nestedBytes, '\n')
            || cameraVisible > 3 || imageVersion > 1) return false;
        HarmonicMIDI::grid::SceneCameraBackground background;
        using BackgroundMode = HarmonicMIDI::grid::SceneCameraBackgroundMode;
        if (cameraVisible == 2)
        {
            std::uint64_t bits[3] {};
            if (!readFamilyUnsigned(wire, bits[0], ' ') || !readFamilyUnsigned(wire, bits[1], ' ')
                || !readFamilyUnsigned(wire, bits[2], '\n')) return false;
            for (const auto b : bits) if (b > std::numeric_limits<std::uint32_t>::max()) return false;
            background.mode = BackgroundMode::Colour;
            background.color = { detail::floatFromBits(static_cast<std::uint32_t>(bits[0])),
                                 detail::floatFromBits(static_cast<std::uint32_t>(bits[1])),
                                 detail::floatFromBits(static_cast<std::uint32_t>(bits[2])) };
        }
        else if (cameraVisible == 3) background.mode = BackgroundMode::Transparent;
        HarmonicMIDI::grid::SceneEnvironmentImage image;
        if (imageVersion != 0)
        {
            std::uint64_t orientation = 0, exposure = 0;
            image.version = static_cast<std::uint32_t>(imageVersion);
            if (!readFamilyUnsigned(wire, image.asset.version, ' ')
                || !readFamilyUnsigned(wire, orientation, ' ') || !readFamilyUnsigned(wire, exposure, ' ')
                || orientation > std::numeric_limits<std::uint32_t>::max()
                || exposure > std::numeric_limits<std::uint32_t>::max()) return false;
            image.orientationDegrees = detail::floatFromBits(static_cast<std::uint32_t>(orientation));
            image.exposureStops = detail::floatFromBits(static_cast<std::uint32_t>(exposure));
            const auto digestEnd = wire.find(' ');
            const auto lineEnd = wire.find('\n');
            if (digestEnd == std::string_view::npos || lineEnd == std::string_view::npos
                || digestEnd > lineEnd) return false;
            image.asset.contentSha256 = std::string(wire.substr(0, digestEnd));
            image.asset.id = std::string(wire.substr(digestEnd + 1, lineEnd - digestEnd - 1));
            wire.remove_prefix(lineEnd + 1);
        }
        if (nestedBytes != wire.size() || containsRequestHeader(wire, kCompiledOperationHeaderV31)) return false;
        Request request;
        if (!decodeRequest(wire, request, migrateLegacyV6, allowComposition) || !request.light
            || request.light->kind != HarmonicMIDI::grid::SceneLightKind::Environment
            || request.light->environmentImage.version != 0
            || !request.light->environmentCameraVisible
            || request.light->cameraBackground.mode != BackgroundMode::World) return false;
        request.light->environmentImage = std::move(image);
        request.light->environmentCameraVisible = cameraVisible == 1;
        request.light->cameraBackground = background;
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    // V30 wraps every older envelope and appears only with Path Trace.
    if (encoded.size() > kCompiledOperationHeaderV30.size()
        && encoded.substr(0, kCompiledOperationHeaderV30.size()) == kCompiledOperationHeaderV30
        && encoded[kCompiledOperationHeaderV30.size()] == '\n')
    {
        auto wire = encoded.substr(kCompiledOperationHeaderV30.size() + 1);
        std::uint64_t nestedBytes = 0;
        if (!readFamilyUnsigned(wire, nestedBytes, '\n')
            || nestedBytes != wire.size()
            || containsRequestHeader(wire, kCompiledOperationHeaderV30)) return false;
        Request request;
        if (!decodeRequest(wire, request, migrateLegacyV6, allowComposition)
            || request.renderEngine != RenderEngine::PathTrace) return false;
        request.pathTraceDenoise = true;
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    // V29 wraps V28 and V27, which wrap every other envelope, so they are
    // removed before the family preflight, which then checks the nested wire
    // on its own.
    if (encoded.size() > kCompiledOperationHeaderV29.size()
        && encoded.substr(0, kCompiledOperationHeaderV29.size()) == kCompiledOperationHeaderV29
        && encoded[kCompiledOperationHeaderV29.size()] == '\n')
    {
        auto wire = encoded.substr(kCompiledOperationHeaderV29.size() + 1);
        std::uint64_t mode = 0, nestedBytes = 0;
        if (!readFamilyUnsigned(wire, mode, ' ') || !readFamilyUnsigned(wire, nestedBytes, '\n')
            || mode != static_cast<std::uint64_t>(RenderMode::Fast)
            || nestedBytes != wire.size()
            || containsRequestHeader(wire, kCompiledOperationHeaderV29)
            || containsRequestHeader(wire, kCompiledOperationHeaderV30)) return false;
        Request request;
        if (!decodeRequest(wire, request, migrateLegacyV6, allowComposition)
            || request.renderMode != RenderMode::Accurate) return false;
        request.renderMode = RenderMode::Fast;
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    if (encoded.size() > kCompiledOperationHeaderV28.size()
        && encoded.substr(0, kCompiledOperationHeaderV28.size()) == kCompiledOperationHeaderV28
        && encoded[kCompiledOperationHeaderV28.size()] == '\n')
    {
        auto wire = encoded.substr(kCompiledOperationHeaderV28.size() + 1);
        std::uint64_t engine = 0, samples = 0, previewSamples = 0, nestedBytes = 0;
        if (!readFamilyUnsigned(wire, engine, ' ') || !readFamilyUnsigned(wire, samples, ' ')
            || !readFamilyUnsigned(wire, previewSamples, ' ')
            || !readFamilyUnsigned(wire, nestedBytes, '\n')
            || engine != static_cast<std::uint64_t>(RenderEngine::PathTrace)
            || samples < 1 || samples > kMaximumPathTraceExportSamples
            || previewSamples < 1 || previewSamples > kMaximumPathTracePreviewSamples
            || nestedBytes != wire.size()
            || containsRequestHeader(wire, kCompiledOperationHeaderV27)
            || containsRequestHeader(wire, kCompiledOperationHeaderV28)
            || containsRequestHeader(wire, kCompiledOperationHeaderV29)
            || containsRequestHeader(wire, kCompiledOperationHeaderV30)) return false;
        Request request;
        if (!decodeRequest(wire, request, migrateLegacyV6, allowComposition)
            || request.renderEngine != RenderEngine::Raster) return false;
        request.renderEngine = RenderEngine::PathTrace;
        request.pathTraceExportSamples = static_cast<std::uint16_t>(samples);
        request.pathTracePreviewSamples = static_cast<std::uint16_t>(previewSamples);
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    if (encoded.size() > kCompiledOperationHeaderV27.size()
        && encoded.substr(0, kCompiledOperationHeaderV27.size()) == kCompiledOperationHeaderV27
        && encoded[kCompiledOperationHeaderV27.size()] == '\n')
    {
        auto wire = encoded.substr(kCompiledOperationHeaderV27.size() + 1);
        std::uint64_t engine = 0, samples = 0, nestedBytes = 0;
        if (!readFamilyUnsigned(wire, engine, ' ') || !readFamilyUnsigned(wire, samples, ' ')
            || !readFamilyUnsigned(wire, nestedBytes, '\n')
            || (engine != static_cast<std::uint64_t>(RenderEngine::PathTrace)
                && engine != static_cast<std::uint64_t>(RenderEngine::Realtime))
            || samples < 1 || samples > kMaximumPathTraceExportSamples
            || nestedBytes != wire.size()
            || containsRequestHeader(wire, kCompiledOperationHeaderV27)
            || containsRequestHeader(wire, kCompiledOperationHeaderV29)
            || containsRequestHeader(wire, kCompiledOperationHeaderV30)) return false;
        Request request;
        if (!decodeRequest(wire, request, migrateLegacyV6, allowComposition)
            || request.renderEngine != RenderEngine::Raster) return false;
        request.renderEngine = static_cast<RenderEngine>(engine);
        request.pathTraceExportSamples = static_cast<std::uint16_t>(samples);
        request.pathTracePreviewSamples = kDefaultPathTracePreviewSamples;
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    if (!preflightFamilyOuterWire(encoded)) return false;
    if (encoded.substr(0, kCompiledOperationHeaderV23.size()) == kCompiledOperationHeaderV23)
    {
        if (!allowComposition || encoded.size() <= kImportedParentWrapperBytes
            || encoded[kCompiledOperationHeaderV23.size()] != '\n'
            || encoded[kImportedParentWrapperBytes - 1] != '\n') return false;
        ImportedParentIntentV1 intent;
        if (!decodeImportedParentIntent(encoded.substr(kCompiledOperationHeaderV23.size() + 1,
                kImportedParentIntentHexBytes), intent)) return false;
        const auto nested = encoded.substr(kImportedParentWrapperBytes);
        if (containsRequestHeader(nested, kCompiledOperationHeaderV23)
            || (nested.substr(0, kCompiledOperationHeaderV20.size()) != kCompiledOperationHeaderV20
                && nested.substr(0, kCompiledOperationHeaderV22.size()) != kCompiledOperationHeaderV22)) return false;
        Request request;
        if (!decodeRequest(nested, request, migrateLegacyV6, true)
            || !request.importedScene || request.importedParentIntent) return false;
        request.importedParentIntent = intent;
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    if (encoded.size() > kCompiledOperationHeaderV26.size()
        && encoded.substr(0, kCompiledOperationHeaderV26.size()) == kCompiledOperationHeaderV26
        && encoded[kCompiledOperationHeaderV26.size()] == '\n')
    {
        if (!allowComposition) return false;
        auto wire = encoded.substr(kCompiledOperationHeaderV26.size() + 1);
        std::uint64_t count = 0;
        if (!readFamilyUnsigned(wire, count, ' ') || count < 2 || count > kMaximumImportedLeaves) return false;
        constexpr std::string_view overrideMode = "override ", plainMode = "plain ";
        const bool cameraOverride = wire.substr(0, overrideMode.size()) == overrideMode;
        if (cameraOverride) wire.remove_prefix(overrideMode.size());
        else if (wire.substr(0, plainMode.size()) == plainMode) wire.remove_prefix(plainMode.size());
        else return false;
        std::array<std::uint64_t, kMaximumImportedLeaves> bytes {};
        std::uint64_t total = 0;
        for (std::size_t i = 0; i < count; ++i)
        {
            if (!readFamilyUnsigned(wire, bytes[i], i + 1 == count ? '\n' : ' ')
                || bytes[i] == 0 || bytes[i] > kMaximumV8EncodedBytes) return false;
            total += bytes[i];
        }
        std::optional<HarmonicMIDI::grid::SceneCameraRecord> camera;
        if (cameraOverride)
        {
            const auto end = wire.find('\n');
            if (end == std::string_view::npos) return false;
            std::istringstream record { std::string(wire.substr(0, end)) };
            HarmonicMIDI::grid::SceneCameraRecord value;
            if (!decodeCamera(record, value)) return false;
            record >> std::ws;
            if (!record.eof()) return false;
            camera = value;
            wire.remove_prefix(end + 1);
        }
        if (total != wire.size()) return false;
        const auto base = wire.substr(0, static_cast<std::size_t>(bytes[0]));
        if (base.substr(0, kCompiledOperationHeaderV20.size()) != kCompiledOperationHeaderV20
            || containsRequestHeader(base, kCompiledOperationHeaderV26)) return false;
        Request request;
        if (!decodeRequest(base, request, migrateLegacyV6, true) || !request.importedScene
            || request.camera || request.importedParentIntent
            || !request.additionalImportedScenes.empty()) return false;
        auto offset = static_cast<std::size_t>(bytes[0]);
        for (std::size_t i = 1; i < count; ++i)
        {
            const auto text = wire.substr(offset, static_cast<std::size_t>(bytes[i]));
            offset += static_cast<std::size_t>(bytes[i]);
            Request leaf;
            if (containsRequestHeader(text, kCompiledOperationHeaderV26)
                || !decodeRequest(text, leaf, nullptr, false) || leaf.importedScene
                || !leaf.additionalImportedScenes.empty()) return false;
            leaf.structuralRevision = request.structuralRevision;
            leaf.evaluationRevision = request.evaluationRevision;
            request.additionalImportedScenes.push_back(std::make_shared<const Request>(std::move(leaf)));
        }
        request.camera = camera;
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    ViewBuffer inputBuffer(encoded);
    std::istream input(&inputBuffer);
    std::string header;
    if (!std::getline(input, header)) return false;
    if (header == kCompiledOperationHeaderV25)
    {
        const auto prefix = kCompiledOperationHeaderV25.size() + 1;
        if (encoded.size() <= prefix + kObjectLocalTransformHexBytes
            || encoded[prefix + kObjectLocalTransformHexBytes] != ' ') return false;
        visualimportedobjectlocal::ObjectLocalTransformV1 local;
        if (!decodeObjectLocalTransform(encoded.substr(prefix, kObjectLocalTransformHexBytes), local)
            || local.version != 1) return false;
        auto nested = encoded.substr(prefix + kObjectLocalTransformHexBytes + 1);
        std::uint64_t nestedBytes = 0;
        if (!readFamilyUnsigned(nested, nestedBytes, '\n') || nestedBytes != nested.size()
            || containsRequestHeader(nested, kCompiledOperationHeaderV25)) return false;
        Request request;
        if (!decodeRequest(nested, request, migrateLegacyV6, false)
            || request.objectLocalTransform.version != 0) return false;
        request.objectLocalTransform = local;
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    if (header == kCompiledOperationHeaderV24)
    {
        std::uint32_t node = 0, mesh = 0;
        std::size_t nestedBytes = 0;
        if (!readUnsigned(input, node) || !readUnsigned(input, mesh)
            || !readUnsigned(input, nestedBytes) || input.get() != '\n'
            || node == 0 || mesh == 0) return false;
        const auto offset = input.tellg();
        if (offset < 0 || static_cast<std::size_t>(offset) > encoded.size()
            || nestedBytes != encoded.size() - static_cast<std::size_t>(offset)) return false;
        const auto nested = encoded.substr(static_cast<std::size_t>(offset));
        if (containsRequestHeader(nested, kCompiledOperationHeaderV24)) return false;
        Request request;
        if (!decodeRequest(nested, request, migrateLegacyV6, false)
            || request.objectNodeStableId != 0 || request.objectMeshStableId != 0) return false;
        request.objectNodeStableId = node;
        request.objectMeshStableId = mesh;
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    if (header == kCompiledOperationHeaderV22)
    {
        std::string mode;
        std::uint32_t node = 0;
        std::size_t nestedBytes = 0;
        std::optional<HarmonicMIDI::grid::SceneCameraRecord> override;
        if (!(input >> mode)) return false;
        if (mode == "source")
        {
            if (!readUnsigned(input, node) || node == 0) return false;
        }
        else if (mode != "override" || !allowComposition) return false;
        if (!readUnsigned(input, nestedBytes) || input.get() != '\n') return false;
        if (mode == "override")
        {
            HarmonicMIDI::grid::SceneCameraRecord camera;
            if (!decodeCamera(input, camera) || input.get() != '\n') return false;
            override = camera;
        }
        const auto offset = input.tellg();
        if (offset < 0 || static_cast<std::size_t>(offset) > encoded.size()
            || nestedBytes != encoded.size() - static_cast<std::size_t>(offset)) return false;
        const auto nested = encoded.substr(static_cast<std::size_t>(offset));
        // A composition override may contain exactly its separately selected leaf.
        if (mode == "source" && containsRequestHeader(nested, kCompiledOperationHeaderV22)) return false;
        if (mode == "override" && nested.substr(0, kCompiledOperationHeaderV20.size()) != kCompiledOperationHeaderV20) return false;
        Request request;
        if (!decodeRequest(nested, request, migrateLegacyV6, mode == "override")) return false;
        if (mode == "source")
        {
            if (request.sceneSnapshot || request.importedScene || request.sourceCameraNodeId) return false;
            request.sourceCameraNodeId = node;
        }
        else
        {
            if (!request.importedScene || request.camera || !request.importedScene->sourceCameraNodeId) return false;
            request.camera = override;
        }
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    if (header == kCompiledOperationHeaderV21)
    {
        TimelineFamilyWire family;
        if (!preflightTimelineFamilyWire(encoded,family)) return false;
        Request request;
        if (!decodeRequest(family.nested,request,migrateLegacyV6,false)
            || !request.sceneSnapshot || request.timelineGeometry || !request.timelineGeometryFamily.empty()) return false;
        request.timelineGeometryFamily.reserve(family.count);
        for (std::size_t i=0;i<family.count;++i)
            request.timelineGeometryFamily.push_back({family.children[i].object,std::string(family.children[i].plan)});
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output=std::move(request); return true;
    }
    if (header == kCompiledOperationHeaderV20)
    {
        if (!allowComposition) return false;
        std::size_t sceneBytes = 0, leafBytes = 0;
        if (!detail::readUnsigned(input, sceneBytes) || !detail::readUnsigned(input, leafBytes)
            || input.get() != '\n' || sceneBytes == 0 || leafBytes == 0
            || sceneBytes > kMaximumV8EncodedBytes || leafBytes > kMaximumV8EncodedBytes) return false;
        const auto offset = input.tellg();
        if (offset < 0 || static_cast<std::size_t>(offset) > encoded.size()
            || sceneBytes > encoded.size() - static_cast<std::size_t>(offset)
            || leafBytes != encoded.size() - static_cast<std::size_t>(offset) - sceneBytes) return false;
        const auto sceneEncoded = encoded.substr(static_cast<std::size_t>(offset), sceneBytes);
        const auto leafEncoded = encoded.substr(static_cast<std::size_t>(offset) + sceneBytes, leafBytes);
        Request request, leaf;
        if (!decodeRequest(sceneEncoded, request, nullptr, false)
            || !decodeRequest(leafEncoded, leaf, nullptr, false)
            || request.importedScene || leaf.importedScene) return false;
        // Legacy leaf bytes omit these outer revisions. Only V20 supplies
        // their context; embedded material revisions still validate below.
        leaf.structuralRevision = request.structuralRevision;
        leaf.evaluationRevision = request.evaluationRevision;
        request.importedScene = std::make_shared<const Request>(std::move(leaf));
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request);
        return true;
    }
    if (header == kCompiledOperationHeaderV19)
    {
        GlassCausticSelection selection;
        auto& policy = selection.policy;
        std::uint32_t unresolved = 0, minimum = 0;
        if (!detail::readUnsigned(input, selection.version)
            || !detail::readUnsigned(input, selection.sourceLight.value)
            || !detail::readUnsigned(input, selection.glassObject.value)
            || !detail::readUnsigned(input, selection.glassMaterial.value)
            || !detail::readUnsigned(input, policy.version)
            || !detail::readUnsigned(input, unresolved) || !detail::readUnsigned(input, minimum)
            || !detail::readUnsigned(input, policy.maximumInterfaces)
            || !detail::readUnsigned(input, policy.maximumVisits)
            || !detail::readUnsigned(input, policy.maximumPending) || input.get() != '\n') return false;
        policy.maximumUnresolvedFraction = detail::floatFromBits(unresolved);
        policy.minimumBranchFraction = detail::floatFromBits(minimum);
        const auto offset = input.tellg();
        if (offset < 0) return false;
        const auto nested = encoded.substr(static_cast<std::size_t>(offset));
        if (nested.find(kCompiledOperationHeaderV19) != std::string_view::npos) return false;
        Request request;
        if (!decodeRequest(nested, request, migrateLegacyV6, allowComposition) || request.glassCaustic) return false;
        request.glassCaustic = selection;
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request);
        return true;
    }
    if (header == kCompiledOperationHeaderV17)
    {
        importedlightcontrol::Controls controls;
        std::size_t count = 0;
        if (!detail::readUnsigned(input, controls.selectedDefinition)
            || !detail::readUnsigned(input, count)
            || count > importedlightcontrol::kMaximumDefinitions || count > encoded.size()
            || input.get() != '\n') return false;
        std::uint32_t previous = 0;
        for (std::size_t index = 0; index < count; ++index)
        {
            std::uint32_t definition = 0, bits = 0;
            if (!detail::readUnsigned(input, definition) || definition <= previous
                || !detail::readUnsigned(input, bits) || input.get() != '\n') return false;
            controls.intensityMultipliers.emplace(definition, detail::floatFromBits(bits));
            previous = definition;
        }
        if (controls.empty() || !importedlightcontrol::valid(controls)) return false;
        const auto offset = input.tellg();
        if (offset < 0) return false;
        const auto nested = encoded.substr(static_cast<std::size_t>(offset));
        if (nested.find(kCompiledOperationHeaderV17) != std::string_view::npos) return false;
        Request request;
        if (!decodeRequest(nested, request, migrateLegacyV6, allowComposition) || !request.importedLights.empty()) return false;
        request.importedLights = std::move(controls);
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request);
        return true;
    }
    if (header == kCompiledOperationHeaderV15)
    {
        HarmonicMIDI::grid::SceneAnalyticSky sky;
        if (!detail::readUnsigned(input,sky.version) || sky.version!=1) return false;
        float* values[] {&sky.zenith.x,&sky.zenith.y,&sky.zenith.z,
            &sky.horizon.x,&sky.horizon.y,&sky.horizon.z,&sky.ground.x,&sky.ground.y,&sky.ground.z,
            &sky.sunColor.x,&sky.sunColor.y,&sky.sunColor.z,&sky.sunIntensity,
            &sky.sunAzimuthDegrees,&sky.sunElevationDegrees,&sky.sunRadiusDegrees,&sky.orientationDegrees};
        for (auto* value:values)
        {
            std::uint32_t bits=0;
            if (!detail::readUnsigned(input,bits)) return false;
            *value=detail::floatFromBits(bits);
        }
        if (input.peek()==' ')
        {
            for (auto* value:{&sky.cloudCoverage,&sky.cloudType,&sky.cloudDrift,&sky.cloudHeadingDegrees})
            {
                std::uint32_t bits=0;
                if (!detail::readUnsigned(input,bits)) return false;
                *value=detail::floatFromBits(bits);
            }
            if (input.peek()==' ')
            {
                for (auto* value:{&sky.sunHalo,&sky.cloudThickness})
                {
                    std::uint32_t bits=0;
                    if (!detail::readUnsigned(input,bits)) return false;
                    *value=detail::floatFromBits(bits);
                }
                if (!HarmonicMIDI::grid::analyticSkyExtrasAuthored(sky)) return false;
            }
            else if (!HarmonicMIDI::grid::analyticSkyCloudsAuthored(sky)) return false;
        }
        if (input.get()!='\n') return false;
        const auto offset=input.tellg();
        if (offset<0) return false;
        const auto nested=encoded.substr(static_cast<std::size_t>(offset));
        if (nested.find(kCompiledOperationHeaderV15)!=std::string_view::npos) return false;
        Request request;
        if (!decodeRequest(nested,request,migrateLegacyV6,allowComposition) || !request.light
            || request.light->sky.version!=0) return false;
        request.light->sky=sky;
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output=std::move(request);
        return true;
    }
    if (header == kCompiledOperationHeaderV14)
    {
        videowire::geometry::TimelineSceneBinding binding;
        std::size_t size = 0;
        if (!detail::readUnsigned(input, binding.object.value) || !detail::readUnsigned(input, size)
            || size == 0 || size > encoded.size()
            || input.get() != '\n') return false;
        binding.loweredPlan.resize(size);
        if (!input.read(binding.loweredPlan.data(), static_cast<std::streamsize>(size))
            || input.get() != '\n') return false;
        const auto offset = input.tellg();
        if (offset < 0) return false;
        const auto nested = encoded.substr(static_cast<std::size_t>(offset));
        if (nested.find(kCompiledOperationHeaderV14) != std::string_view::npos) return false;
        Request request;
        {
            const detail::ScopedReplayedTimelineMesh replayed(binding);
            if (!decodeRequest(nested, request, migrateLegacyV6, allowComposition) || !request.sceneSnapshot) return false;
        }
        request.timelineGeometry = std::move(binding);
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request);
        return true;
    }
    if (header == kCompiledOperationHeaderV13 || header == kCompiledOperationHeaderV18)
    {
        geometrysurfacematerial::ObjectPrograms programs;
        if (!detail::decodeSurfacePrograms(input, programs)) return false;
        const auto offset = input.tellg();
        if (offset < 0) return false;
        const auto nested = encoded.substr(static_cast<std::size_t>(offset));
        if (nested.find(kCompiledOperationHeaderV13) != std::string_view::npos
            || nested.find(kCompiledOperationHeaderV18) != std::string_view::npos
            || nested.find(kCompiledOperationHeaderV14) != std::string_view::npos) return false;
        Request request;
        if (!decodeRequest(nested, request, migrateLegacyV6, allowComposition) || !request.sceneSnapshot
            || request.diffractionMaterial.has_value() != (header == kCompiledOperationHeaderV18)) return false;
        request.surfacePrograms = std::move(programs);
        for (auto& object : request.surfacePrograms) object.program.material.sceneSnapshot = request.sceneSnapshot;
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request); return true;
    }
    if (header == kCompiledOperationHeaderV12)
    {
        materialfield::Binding field;
        unsigned reduction = 0;
        std::uint64_t gainBits = 0;
        if (!(input >> field.materialNode >> field.target >> reduction >> field.index)
            || !detail::readUnsigned(input, gainBits)
            || reduction > 3 || input.get() != '\n' || !std::getline(input, field.fieldPlan)) return false;
        std::memcpy(&field.gain, &gainBits, sizeof(gainBits));
        field.reduction = static_cast<materialfield::Reduction>(reduction);
        const auto offset = input.tellg();
        if (offset < 0) return false;
        const auto nested = encoded.substr(static_cast<std::size_t>(offset));
        if (nested.find(kCompiledOperationHeaderV12) != std::string_view::npos) return false;
        Request request;
        if (!decodeRequest(nested, request, migrateLegacyV6, allowComposition)) return false;
        request.materialField = std::move(field);
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request);
        return true;
    }
    if (header == kCompiledOperationHeaderV11)
    {
        std::uint32_t mask = 0;
        if (!detail::readUnsigned(input, mask) || input.get() != '\n'
            || mask == 0 || mask > render3dimage::kRawExportMask) return false;
        const auto offset = input.tellg();
        if (offset < 0) return false;
        const auto nested = encoded.substr(static_cast<std::size_t>(offset));
        if (nested.substr(0, kCompiledOperationHeaderV11.size()) == kCompiledOperationHeaderV11)
            return false;
        Request request;
        if (!decodeRequest(nested, request, migrateLegacyV6, allowComposition)) return false;
        request.rawExportMask = mask;
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request);
        return true;
    }
    if (header == kCompiledOperationHeaderV10)
    {
        NoteLight mapping;
        std::uint32_t gain = 0, pitch = 0;
        if (!detail::readUnsigned(input, mapping.track)
            || !detail::readUnsigned(input, gain) || !detail::readUnsigned(input, pitch)
            || input.get() != '\n') return false;
        mapping.gain = detail::floatFromBits(gain);
        mapping.pitchDegrees = detail::floatFromBits(pitch);
        const auto offset = input.tellg();
        if (offset < 0) return false;
        const auto nested = encoded.substr(static_cast<std::size_t>(offset));
        // Envelopes nest only in descending version order.
        if (nested.substr(0, kCompiledOperationHeaderV10.size()) == kCompiledOperationHeaderV10
            || nested.substr(0, kCompiledOperationHeaderV11.size()) == kCompiledOperationHeaderV11)
            return false;
        Request request;
        if (!decodeRequest(nested, request, migrateLegacyV6, allowComposition)) return false;
        request.noteLight = mapping;
        if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request);
        return true;
    }
    if (header == kCompiledOperationHeaderV9)
    {
        unsigned outputKind = 0;
        if (!(input >> outputKind) || input.get() != '\n'
            || outputKind >= static_cast<unsigned>(renderpassoutput::Output::Count))
            return false;
        const auto offset = input.tellg();
        if (offset < 0) return false;
        const auto nested = encoded.substr(static_cast<std::size_t>(offset));
        // Only one envelope is admitted, so malformed input cannot recurse.
        if (nested.substr(0, kCompiledOperationHeaderV9.size()) == kCompiledOperationHeaderV9
            || nested.substr(0, kCompiledOperationHeaderV10.size()) == kCompiledOperationHeaderV10
            || nested.substr(0, kCompiledOperationHeaderV11.size()) == kCompiledOperationHeaderV11)
            return false;
        Request request;
        if (!decodeRequest(nested, request, migrateLegacyV6, allowComposition)) return false;
        request.imageOutput = static_cast<renderpassoutput::Output>(outputKind);
        if (request.imageOutput == renderpassoutput::Output::Color
            || !valid(request) || !canonicalBytes(request, encoded))
            return false;
        output = std::move(request);
        return true;
    }
    const bool hasComposition = header == kCompiledOperationHeaderV8 || header == kCompiledOperationHeaderV16;
    const bool hasDiffraction = header == kCompiledOperationHeaderV7;
    const bool hasLegacyV6Diffraction = header == kCompiledOperationHeaderV6;
    const bool hasCamera = header == kCompiledOperationHeaderV5;
    const bool hasVertex = header == kCompiledOperationHeaderV4;
    bool hasMaterial = header == kCompiledOperationHeaderV2 || hasVertex;
    bool hasDeformation = header == kCompiledOperationHeaderV3;
    if ((!hasMaterial && !hasDeformation && !hasVertex && !hasCamera && !hasComposition
         && !hasDiffraction && !hasLegacyV6Diffraction
         && header != kCompiledOperationHeader)
        || (!hasMaterial && !hasDeformation && !hasDiffraction && !hasComposition
            && !hasLegacyV6Diffraction
            && encoded.size() > kMaximumEncodedBytes)
        || (hasMaterial && !hasVertex && !hasComposition && encoded.size() > kMaximumV2EncodedBytes)
        || ((hasDiffraction || hasLegacyV6Diffraction)
            && encoded.size() > kMaximumV4EncodedBytes))
        return false;

    Request request;
    // A vertex modifier can be any size, and whether one is present is read below. The
    // canonical re-encode applies the bound for what was actually decoded.
    if (hasComposition && input.peek() == 's')
    {
        std::string mode;
        if (!(input >> mode >> request.sourceStableId >> request.renderStableId
                    >> request.structuralRevision >> request.evaluationRevision >> std::ws)
            || mode != "scene")
            return false;
        const auto sceneOffset = input.tellg();
        if (sceneOffset < 0) return false;
        const auto sceneText = encoded.substr(static_cast<std::size_t>(sceneOffset));
        const auto sceneHex = sceneText.substr(0, sceneText.find_first_of(" \t\n\v\f\r"));
        if (sceneHex.empty()) return false;
        input.seekg(static_cast<std::streamoff>(sceneHex.size()), std::ios_base::cur);
        std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
        std::string filledKey;
        if (const auto* binding = detail::replayedTimelineMesh)
        {
            // An earlier read of these exact bytes rebuilt the left-out mesh already.
            filledKey = detail::filledTimelineSceneKey(sceneHex, *binding);
            if (const auto kept = detail::filledTimelineSceneMemo().find(filledKey))
            {
                scene = kept->scene;
                detail::keepTimelineValue(*binding, kept->value);
                detail::keepFilledTimelineCheck(*binding, scene, *kept->value);
            }
        }
        if (!scene)
        {
            std::optional<HarmonicMIDI::grid::SceneObjectId> replayedMesh;
            scene = sceneaov::detail::decodeSceneToken(sceneHex, &replayedMesh);
            if (!scene) return false;
            if (replayedMesh)
            {
                // The wire left this object's mesh out, and its V14 binding rebuilds it.
                const auto* binding = detail::replayedTimelineMesh;
                if (binding == nullptr || binding->object != *replayedMesh) return false;
                // A scene with a left-out mesh is never kept by sceneTokenMemo, so this decode owns it alone.
                auto rebuilt = std::const_pointer_cast<HarmonicMIDI::grid::Visual3DScene>(scene);
                std::string error;
                auto value = videowire::geometry::fillTimelineSceneMesh(*binding, *rebuilt, error);
                if (!value) return false;
                auto kept = std::make_shared<const videowire::geometry::AdmittedValue>(std::move(*value));
                if (sceneaov::detail::sceneTokenMemoSeeding)
                    detail::filledTimelineSceneMemo().store(filledKey,
                        std::make_shared<const detail::FilledTimelineScene>(detail::FilledTimelineScene { scene, kept }));
                detail::keepFilledTimelineCheck(*binding, scene, *kept);
                detail::keepTimelineValue(*binding, std::move(kept));
            }
        }
        request.sceneSnapshot = std::move(scene);
        int materialFlag = -1;
        int vertexFlag = -1;
        int diffractionFlag = -1;
        if (!(input >> materialFlag >> vertexFlag >> diffractionFlag)
            || (materialFlag != 0 && materialFlag != 1)
            || (vertexFlag != 0 && vertexFlag != 1)
            || (diffractionFlag != 0 && diffractionFlag != 1)
            || (vertexFlag != 0 && materialFlag == 0)
            || (materialFlag != 0 && diffractionFlag != 0))
            return false;
        if (materialFlag != 0)
        {
            surfacematerialbinding::ImportedSceneMaterialRequest material;
            if (!detail::decodeMaterial(input, material, vertexFlag != 0)) return false;
            material.sceneSnapshot = request.sceneSnapshot;
            request.material = std::move(material);
        }
        if (diffractionFlag != 0)
        {
            diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest material;
            if (!detail::decodeDiffractionMaterial(input, material)) return false;
            request.diffractionMaterial = std::move(material);
        }
        input >> std::ws;
        if (!input.eof() || !valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request);
        return true;
    }
    if (!detail::decodeBase(input, request)) return false;
    if (hasComposition)
    {
        int deformationFlag = -1;
        int materialFlag = -1;
        int vertexFlag = -1;
        int diffractionFlag = -1;
        int cameraFlag = -1;
        int lightFlag = -1;
        if (!(input >> deformationFlag >> materialFlag >> vertexFlag
                    >> diffractionFlag >> cameraFlag >> lightFlag)
            || (deformationFlag != 0 && deformationFlag != 1)
            || (materialFlag != 0 && materialFlag != 1)
            || (vertexFlag != 0 && vertexFlag != 1)
            || (diffractionFlag != 0 && diffractionFlag != 1)
            || (cameraFlag != 0 && cameraFlag != 1)
            || (lightFlag != 1 && (header != kCompiledOperationHeaderV16 || lightFlag != 0))
            || (vertexFlag != 0 && materialFlag == 0)
            || (materialFlag != 0 && diffractionFlag != 0))
            return false;
        if (cameraFlag != 0)
        {
            HarmonicMIDI::grid::SceneCameraRecord camera;
            if (!detail::decodeCamera(input, camera)) return false;
            request.camera = camera;
        }
        if (lightFlag != 0)
        {
            HarmonicMIDI::grid::SceneLightRecord light;
            if (!detail::decodeLight(input, light)) return false;
            request.light = light;
        }
        if (deformationFlag != 0)
        {
            std::size_t deformationBytes = 0;
            if (!(input >> deformationBytes) || deformationBytes == 0
                || deformationBytes > encoded.size())
                return false;
            char newline = '\0';
            input.get(newline);
            if (newline != '\n') return false;
            std::string payload(deformationBytes, '\0');
            input.read(payload.data(), static_cast<std::streamsize>(deformationBytes));
            visualanimationimport::Request deformation;
            std::string deformationError;
            if (input.gcount() != static_cast<std::streamsize>(deformationBytes)
                || !visualanimationoperation::decode(payload, deformation, deformationError))
                return false;
            request.deformation = std::move(deformation);
        }
        if (materialFlag != 0)
        {
            surfacematerialbinding::ImportedSceneMaterialRequest material;
            if (!detail::decodeMaterial(input, material, vertexFlag != 0)) return false;
            request.material = std::move(material);
        }
        if (diffractionFlag != 0)
        {
            diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest material;
            if (!detail::decodeDiffractionMaterial(input, material)) return false;
            request.diffractionMaterial = std::move(material);
        }
        input >> std::ws;
        if (!input.eof() || !valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request);
        return true;
    }
    if (hasDiffraction || hasLegacyV6Diffraction)
    {
        int deformationFlag = -1;
        int diffractionFlag = -1;
        int cameraFlag = -1;
        if (!(input >> deformationFlag >> diffractionFlag >> cameraFlag)
            || (deformationFlag != 0 && deformationFlag != 1)
            || diffractionFlag != 1
            || (cameraFlag != 0 && cameraFlag != 1))
            return false;
        if (cameraFlag != 0)
        {
            HarmonicMIDI::grid::SceneCameraRecord camera;
            if (!detail::decodeCamera(input, camera)) return false;
            request.camera = camera;
        }
        if (deformationFlag != 0)
        {
            std::size_t deformationBytes = 0;
            if (!(input >> deformationBytes) || deformationBytes == 0
                || deformationBytes > encoded.size())
                return false;
            char newline = '\0';
            input.get(newline);
            if (newline != '\n') return false;
            std::string payload(deformationBytes, '\0');
            input.read(payload.data(), static_cast<std::streamsize>(deformationBytes));
            visualanimationimport::Request deformation;
            std::string deformationError;
            if (input.gcount() != static_cast<std::streamsize>(deformationBytes)
                || !visualanimationoperation::decode(
                    payload, deformation, deformationError))
                return false;
            request.deformation = std::move(deformation);
        }
        diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest material;
        const bool decodedMaterial = hasLegacyV6Diffraction
            ? detail::decodeLegacyV6DiffractionMaterial(input, material)
            : detail::decodeDiffractionMaterial(input, material);
        if (!decodedMaterial) return false;
        request.diffractionMaterial = std::move(material);
        input >> std::ws;
        if (!input.eof()) return false;
        if (!hasLegacyV6Diffraction)
        {
            if (!valid(request) || !canonicalBytes(request, encoded)) return false;
        }
        else
        {
            request.diffractionMaterial->lighting
                = diffractionmaterial::makeReferenceLighting();
            if (encodeLegacyV6(request) != encoded || migrateLegacyV6 == nullptr)
                return false;
            auto& migrated = *request.diffractionMaterial;
            migrated.version = diffractionmaterialbinding::kWireVersion;
            migrated.material.version = diffractionmaterial::kWireVersion;
            migrated.material.grooveField = {};
            if (!migrateLegacyV6(migrated) || !valid(request)) return false;
        }
        output = std::move(request);
        return true;
    }
    if (hasCamera)
    {
        int deformationFlag = -1;
        int materialFlag = -1;
        int vertexFlag = -1;
        if (!(input >> deformationFlag >> materialFlag >> vertexFlag)
            || (deformationFlag != 0 && deformationFlag != 1)
            || (materialFlag != 0 && materialFlag != 1)
            || (vertexFlag != 0 && vertexFlag != 1)
            || (vertexFlag != 0 && materialFlag == 0))
            return false;
        HarmonicMIDI::grid::SceneCameraRecord camera;
        if (!detail::decodeCamera(input, camera)) return false;
        request.camera = camera;
        if (deformationFlag != 0)
        {
            std::size_t deformationBytes = 0;
            if (!(input >> deformationBytes) || deformationBytes == 0
                || deformationBytes > encoded.size())
                return false;
            char newline = '\0';
            input.get(newline);
            if (newline != '\n') return false;
            std::string payload(deformationBytes, '\0');
            input.read(payload.data(), static_cast<std::streamsize>(deformationBytes));
            visualanimationimport::Request deformation;
            std::string deformationError;
            if (input.gcount() != static_cast<std::streamsize>(deformationBytes)
                || !visualanimationoperation::decode(payload, deformation, deformationError))
                return false;
            request.deformation = std::move(deformation);
        }
        if (materialFlag != 0)
        {
            surfacematerialbinding::ImportedSceneMaterialRequest material;
            if (!detail::decodeMaterial(input, material, vertexFlag != 0)) return false;
            request.material = std::move(material);
        }
        input >> std::ws;
        if (!input.eof() || !valid(request) || !canonicalBytes(request, encoded)) return false;
        output = std::move(request);
        return true;
    }
    if (hasDeformation || hasVertex)
    {
        int materialFlag = -1;
        std::size_t deformationBytes = 0;
        if (!(input >> materialFlag >> deformationBytes)
            || (materialFlag != 0 && materialFlag != 1)
            || (hasDeformation && deformationBytes == 0)
            || (hasVertex && ((materialFlag == 0) != (deformationBytes == 0)))
            || deformationBytes > encoded.size())
            return false;
        char newline = '\0';
        input.get(newline);
        if (newline != '\n') return false;
        if (deformationBytes != 0)
        {
            std::string deformationPayload(deformationBytes, '\0');
            input.read(deformationPayload.data(),
                       static_cast<std::streamsize>(deformationBytes));
            if (input.gcount() != static_cast<std::streamsize>(deformationBytes))
                return false;
            visualanimationimport::Request deformation;
            std::string deformationError;
            if (!visualanimationoperation::decode(
                    deformationPayload, deformation, deformationError))
                return false;
            request.deformation = std::move(deformation);
        }
        if (hasVertex)
            hasDeformation = deformationBytes != 0;
        else
            hasMaterial = materialFlag != 0;
    }
    if (hasMaterial)
    {
        surfacematerialbinding::ImportedSceneMaterialRequest material;
        if (!detail::decodeMaterial(input, material, hasVertex)) return false;
        request.material = std::move(material);
    }
    input >> std::ws;
    if (!input.eof() || !valid(request) || !canonicalBytes(request, encoded))
        return false;
    output = std::move(request);
    return true;
}
} // namespace detail

inline bool decode(std::string_view encoded, Request& output,
                   LegacyV6DiffractionAdmission migrateLegacyV6 = nullptr)
{
    const ScopedTimelineValidation oneValidation;
    return detail::decodeRequest(encoded, output, migrateLegacyV6, true);
}

inline bool decodeCanonical(std::string_view encoded, Request& output,
                            std::string& error)
{
    if (!decode(encoded, output))
    {
        error = "invalid or noncanonical imported scene render operation";
        return false;
    }
    error.clear();
    return true;
}

} // namespace visualimportedscenerender
