#pragma once

#include "imported_animation_playback_clock.h"
#include "note_reactive_light.h"
#include "geometry_score_field.h"
#include "composed_timeline_geometry.h"

#include "baked_target_modulation.h"
#include "imported_animation_visual_plan_execution.h"
#include "imported_scene_payload_execution.h"
#include "material_frame_evaluation.h"
#include "imported_scene_probe_diagnostic.h"
#include "visual_plan_executor.h"
#include "imported_material_frame_resolver.h"
#include "loop_step_probe.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace videohelper::importedscene
{
template <typename Layer, typename = void>
struct HasRawPassExport : std::false_type {};
template <typename Layer>
struct HasRawPassExport<Layer, std::void_t<
    decltype(std::declval<Layer>().rawExportFrame),
    decltype(std::declval<Layer>().rawExportMask),
    decltype(std::declval<Layer>().rawExportClipId),
    decltype(std::declval<Layer>().rawExportRenderId),
    decltype(std::declval<Layer>().rawExportImageOutput)>> : std::true_type {};

template <typename Layer, typename = void>
struct HasImportedTextureOwner : std::false_type {};
template <typename Layer>
struct HasImportedTextureOwner<Layer, std::void_t<
    decltype(std::declval<Layer>().nativeTextureOwner)>> : std::true_type {};

template <typename Layer, typename = void>
struct HasImportedParentPresentation : std::false_type {};
template <typename Layer>
struct HasImportedParentPresentation<Layer, std::void_t<
    decltype(std::declval<Layer>().importedParentPresentation)>> : std::true_type {};

template <typename Layer, typename = void>
struct HasImportedLinearImage : std::false_type {};
template <typename Layer>
struct HasImportedLinearImage<Layer, std::void_t<
    decltype(std::declval<Layer>().nativeLinearImage)>> : std::true_type {};

template <typename Layer, typename = void>
struct HasImportedVertexSpectrum : std::false_type {};
template <typename Layer>
struct HasImportedVertexSpectrum<Layer, std::void_t<
    decltype(std::declval<Layer>().audioPresent),
    decltype(std::declval<Layer>().audioFeatures.bands)>> : std::true_type {};

template <typename Layer, typename = void>
struct HasImportedSolidInput : std::false_type {};
template <typename Layer>
struct HasImportedSolidInput<Layer, std::void_t<decltype(std::declval<Layer>().particleParameters)>>
    : videowire::HasGeometryParticleInput<Layer> {};

template <typename Execution,typename = void>
struct HasImportedParentInspector : std::false_type {};
template <typename Execution>
struct HasImportedParentInspector<Execution,std::void_t<decltype(std::declval<const Execution&>().inspectParentReceipt(
    std::declval<const typename Execution::Receipt&>(),std::declval<const modelpayload::ImportedSceneRequest&>(),
    modelpayload::ImportedSceneUse::Preview))>> : std::true_type {};

// Test receipts without a pending member keep the template's original contract.
template <typename Receipt,typename = void>
struct HasImportedPendingReceipt : std::false_type {};
template <typename Receipt>
struct HasImportedPendingReceipt<Receipt,std::void_t<decltype(std::declval<const Receipt&>().pending)>>
    : std::true_type {};

struct AnalyticSkyRuntimeParameter
{
    const char* name;
    float* value;
    double minimum;
    double maximum;
};
inline auto analyticSkyRuntimeParameters(HarmonicMIDI::grid::SceneLightRecord& light)
{
    auto& s=light.sky;
    return std::array<AnalyticSkyRuntimeParameter,21> {{
        {"red",&light.color.x,0,64},{"green",&light.color.y,0,64},{"blue",&light.color.z,0,64},
        {"intensity",&light.intensity,0,1000000},
        {"skyZenithRed",&s.zenith.x,0,64},{"skyZenithGreen",&s.zenith.y,0,64},{"skyZenithBlue",&s.zenith.z,0,64},
        {"skyHorizonRed",&s.horizon.x,0,64},{"skyHorizonGreen",&s.horizon.y,0,64},{"skyHorizonBlue",&s.horizon.z,0,64},
        {"skyGroundRed",&s.ground.x,0,64},{"skyGroundGreen",&s.ground.y,0,64},{"skyGroundBlue",&s.ground.z,0,64},
        {"sunRed",&s.sunColor.x,0,64},{"sunGreen",&s.sunColor.y,0,64},{"sunBlue",&s.sunColor.z,0,64},
        {"sunIntensity",&s.sunIntensity,0,64},{"sunAzimuthDegrees",&s.sunAzimuthDegrees,-360,360},
        {"sunElevationDegrees",&s.sunElevationDegrees,-90,90},{"sunRadiusDegrees",&s.sunRadiusDegrees,.25,30},
        {"skyOrientationDegrees",&s.orientationDegrees,-360,360}
    }};
}
inline bool authoredAnalyticSkyLight(const videowire::CompiledVisualLayerPlan& plan,
                                    const HarmonicMIDI::grid::SceneLightRecord& light)
{
    if (light.kind!=HarmonicMIDI::grid::SceneLightKind::Environment || light.sky.version!=1) return false;
    return std::any_of(plan.operations.begin(),plan.operations.end(),[&](const auto& operation) {
        return operation.kind=="visual.3d.light.environment"
            && static_cast<std::uint64_t>(operation.nodeId)+1u==light.id.value;
    });
}
inline auto environmentImageRuntimeParameters(HarmonicMIDI::grid::SceneLightRecord& light)
{
    return std::array<AnalyticSkyRuntimeParameter,6>{{
        {"red",&light.color.x,0,64},{"green",&light.color.y,0,64},{"blue",&light.color.z,0,64},
        {"intensity",&light.intensity,0,1000000},
        {"environmentOrientationDegrees",&light.environmentImage.orientationDegrees,-360,360},
        {"environmentExposureStops",&light.environmentImage.exposureStops,-16,16}}};
}
inline bool evaluateEnvironmentImageRuntime(const std::map<std::string,double>* parameters,
                                            HarmonicMIDI::grid::SceneLightRecord& light)
{
    if (!light.environmentImage.version || !parameters) return true;
    const auto prefix="visual"+std::to_string(light.id.value)+"/";
    for (auto field:environmentImageRuntimeParameters(light))
    {
        const auto found=parameters->find(prefix+field.name);
        if (found==parameters->end()) continue;
        if (!std::isfinite(found->second) || found->second<field.minimum || found->second>field.maximum) return false;
        *field.value=static_cast<float>(found->second);
    }
    return HarmonicMIDI::grid::validSceneEnvironmentImage(light);
}
inline bool evaluateAnalyticSkyRuntime(const videowire::CompiledVisualLayerPlan& plan,
    const std::map<std::string,double>* parameters,
    HarmonicMIDI::grid::SceneLightRecord& light)
{
    if (!evaluateEnvironmentImageRuntime(parameters,light)) return false;
    if (!parameters || !authoredAnalyticSkyLight(plan,light)) return true;
    const auto prefix="visual"+std::to_string(light.id.value)+"/";
    for (auto field:analyticSkyRuntimeParameters(light))
    {
        const auto found=parameters->find(prefix+field.name);
        if (found==parameters->end()) continue;
        if (!std::isfinite(found->second)) return false;
        *field.value=static_cast<float>(std::clamp(found->second,field.minimum,field.maximum));
    }
    return HarmonicMIDI::grid::validSceneAnalyticSky(light);
}

// Directional and point lights. Colour and intensity are absolute values clamped to the
// node ranges. A point light takes position offsets and a directional light takes rotation
// offsets in degrees, composed in world space over the authored transform. The renderer
// reads these as per-frame light records, so no scene is derived for a changed value.
inline bool directionalOrPointLight(const HarmonicMIDI::grid::SceneLightRecord& light) noexcept
{
    return light.kind == HarmonicMIDI::grid::SceneLightKind::Directional
        || light.kind == HarmonicMIDI::grid::SceneLightKind::Point;
}

inline void seedDirectionalOrPointLight(const HarmonicMIDI::grid::SceneLightRecord& light,
                                        std::map<std::string, double>& parameters)
{
    if (!directionalOrPointLight(light)) return;
    const auto prefix = "visual" + std::to_string(light.id.value) + "/";
    parameters.emplace(prefix + "red", light.color.x);
    parameters.emplace(prefix + "green", light.color.y);
    parameters.emplace(prefix + "blue", light.color.z);
    parameters.emplace(prefix + "intensity", light.intensity);
    const std::array<const char*, 3> offsets = light.kind == HarmonicMIDI::grid::SceneLightKind::Point
        ? std::array<const char*, 3> { "positionX", "positionY", "positionZ" }
        : std::array<const char*, 3> { "rotationX", "rotationY", "rotationZ" };
    for (const char* name : offsets) parameters.emplace(prefix + name, 0.0);
}

inline bool sameLightControls(const HarmonicMIDI::grid::SceneLightRecord& a,
                              const HarmonicMIDI::grid::SceneLightRecord& b) noexcept
{
    return a.color.x == b.color.x && a.color.y == b.color.y && a.color.z == b.color.z
        && a.intensity == b.intensity
        && a.transform.translation.x == b.transform.translation.x
        && a.transform.translation.y == b.transform.translation.y
        && a.transform.translation.z == b.transform.translation.z
        && a.transform.rotation.x == b.transform.rotation.x
        && a.transform.rotation.y == b.transform.rotation.y
        && a.transform.rotation.z == b.transform.rotation.z
        && a.transform.rotation.w == b.transform.rotation.w;
}

// Returns false for a non-finite runtime value, or a composed rotation that cannot be normalised.
inline bool evaluateDirectionalOrPointLight(const std::map<std::string, double>& parameters,
                                            HarmonicMIDI::grid::SceneLightRecord& light)
{
    if (!directionalOrPointLight(light)) return true;
    const auto prefix = "visual" + std::to_string(light.id.value) + "/";
    const auto apply = [&](const char* name, float& field, double minimum, double maximum, bool offset) {
        const auto found = parameters.find(prefix + name);
        if (found == parameters.end()) return true;
        if (!std::isfinite(found->second)) return false;
        const auto clamped = static_cast<float>(std::clamp(found->second, minimum, maximum));
        field = offset ? field + clamped : clamped;
        return true;
    };
    if (!apply("red", light.color.x, 0.0, 64.0, false)
        || !apply("green", light.color.y, 0.0, 64.0, false)
        || !apply("blue", light.color.z, 0.0, 64.0, false)
        || !apply("intensity", light.intensity, 0.0, 1000000.0, false))
        return false;
    if (light.kind == HarmonicMIDI::grid::SceneLightKind::Point)
        return apply("positionX", light.transform.translation.x, -1000000.0, 1000000.0, true)
            && apply("positionY", light.transform.translation.y, -1000000.0, 1000000.0, true)
            && apply("positionZ", light.transform.translation.z, -1000000.0, 1000000.0, true);
    // Euler XYZ degrees to a quaternion, with the same half-angle convention as the plugin's
    // Transform 3D lowering.
    const std::array<const char*, 3> names { "rotationX", "rotationY", "rotationZ" };
    std::array<double, 3> degrees {};
    bool present = false;
    for (std::size_t axis = 0; axis < names.size(); ++axis)
    {
        const auto found = parameters.find(prefix + names[axis]);
        if (found == parameters.end()) continue;
        if (!std::isfinite(found->second)) return false;
        degrees[axis] = std::clamp(found->second, -3600.0, 3600.0);
        present = present || degrees[axis] != 0.0;
    }
    if (!present) return true;
    constexpr double degreesToHalfRadians = 0.0087266462599716478846;
    const auto sx = std::sin(degrees[0] * degreesToHalfRadians), cx = std::cos(degrees[0] * degreesToHalfRadians);
    const auto sy = std::sin(degrees[1] * degreesToHalfRadians), cy = std::cos(degrees[1] * degreesToHalfRadians);
    const auto sz = std::sin(degrees[2] * degreesToHalfRadians), cz = std::cos(degrees[2] * degreesToHalfRadians);
    const double o[4] { sx * cy * cz - cx * sy * sz, cx * sy * cz + sx * cy * sz,
                        cx * cy * sz - sx * sy * cz, cx * cy * cz + sx * sy * sz };
    // Hamilton product offset * authored: the offset rotates the light in world space.
    const auto& b = light.transform.rotation;
    const double q[4] {
        o[3] * b.x + o[0] * b.w + o[1] * b.z - o[2] * b.y,
        o[3] * b.y - o[0] * b.z + o[1] * b.w + o[2] * b.x,
        o[3] * b.z + o[0] * b.y - o[1] * b.x + o[2] * b.w,
        o[3] * b.w - o[0] * b.x - o[1] * b.y - o[2] * b.z };
    const double length = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (!std::isfinite(length) || length <= 0.0) return false;
    auto& r = light.transform.rotation;
    r.x = static_cast<float>(q[0] / length);
    r.y = static_cast<float>(q[1] / length);
    r.z = static_cast<float>(q[2] / length);
    r.w = static_cast<float>(q[3] / length);
    return true;
}

// Builds the override for one authored sampled-list record. Only fields that differ from
// the authored record are flagged, so an unmodulated light yields no override. Returns false
// for a non-finite control. Clamps match evaluateDirectionalOrPointLight.
inline bool directionalOrPointOverride(const std::map<std::string, double>& parameters,
                                       const HarmonicMIDI::grid::SceneLightRecord& base,
                                       std::size_t index,
                                       arbitgpu::NativeImportedLightOverride& out)
{
    using namespace HarmonicMIDI::grid;
    out = {};
    out.index = index;
    if (!directionalOrPointLight(base)) return true;
    const auto prefix = "visual" + std::to_string(base.id.value) + "/";
    const auto control = [&](const char* name) -> const double* {
        const auto found = parameters.find(prefix + name);
        return found == parameters.end() ? nullptr : &found->second;
    };
    // Absolute colour and intensity fall back to the authored value when a control is absent.
    const auto absolute = [&](const char* name, double minimum, double maximum, float fallback, float& field) {
        const double* found = control(name);
        if (!found) { field = fallback; return true; }
        if (!std::isfinite(*found)) return false;
        field = static_cast<float>(std::clamp(*found, minimum, maximum));
        return true;
    };
    // Offsets default to zero when a control is absent.
    const auto offset = [&](const char* name, float& field) {
        const double* found = control(name);
        if (!found) { field = 0.0f; return true; }
        if (!std::isfinite(*found)) return false;
        field = static_cast<float>(std::clamp(*found, -1000000.0, 1000000.0));
        return true;
    };
    SceneVec3 color {};
    float intensity = base.intensity;
    if (!absolute("red", 0.0, 64.0, base.color.x, color.x)
        || !absolute("green", 0.0, 64.0, base.color.y, color.y)
        || !absolute("blue", 0.0, 64.0, base.color.z, color.z)
        || !absolute("intensity", 0.0, 1000000.0, base.intensity, intensity))
        return false;
    out.hasColor = color.x != base.color.x || color.y != base.color.y || color.z != base.color.z;
    out.color = color;
    out.hasIntensity = intensity != base.intensity;
    out.intensity = intensity;
    if (base.kind == SceneLightKind::Point)
    {
        SceneVec3 position {};
        if (!offset("positionX", position.x) || !offset("positionY", position.y) || !offset("positionZ", position.z))
            return false;
        out.hasPositionOffset = position.x != 0.0f || position.y != 0.0f || position.z != 0.0f;
        out.positionOffset = position;
        return true;
    }
    // Euler XYZ degrees to the offset quaternion, with the same half-angle convention as the
    // plugin's Transform 3D lowering. The executor composes it as offset * sampled rotation.
    const std::array<const char*, 3> names { "rotationX", "rotationY", "rotationZ" };
    std::array<double, 3> degrees {};
    for (std::size_t axis = 0; axis < names.size(); ++axis)
    {
        const double* found = control(names[axis]);
        if (!found) continue;
        if (!std::isfinite(*found)) return false;
        degrees[axis] = std::clamp(*found, -3600.0, 3600.0);
    }
    if (degrees[0] == 0.0 && degrees[1] == 0.0 && degrees[2] == 0.0) return true;
    constexpr double degreesToHalfRadians = 0.0087266462599716478846;
    const auto sx = std::sin(degrees[0] * degreesToHalfRadians), cx = std::cos(degrees[0] * degreesToHalfRadians);
    const auto sy = std::sin(degrees[1] * degreesToHalfRadians), cy = std::cos(degrees[1] * degreesToHalfRadians);
    const auto sz = std::sin(degrees[2] * degreesToHalfRadians), cz = std::cos(degrees[2] * degreesToHalfRadians);
    out.hasRotationOffset = true;
    out.rotationOffset = { static_cast<float>(sx * cy * cz - cx * sy * sz),
                           static_cast<float>(cx * sy * cz + sx * cy * sz),
                           static_cast<float>(cx * cy * sz - sx * sy * cz),
                           static_cast<float>(cx * cy * cz + sx * sy * sz) };
    return true;
}

inline void seedImportedSceneRuntimeParameters(
    const std::vector<videowire::CompiledVisualLayerPlan>& plans,
    int clipId,
    std::map<std::string, double>& parameters)
{
    const auto* plan = videowire::findVisualLayerPlan(plans, clipId);
    if (plan == nullptr) return;
    videowire::VisualLayerExecution compiled;
    std::string error;
    if (!videowire::compileVisualLayerExecution(*plan, compiled, error)
        || !compiled.importedSceneRender)
        return;
    const auto seedSky=[&](HarmonicMIDI::grid::SceneLightRecord light) {
        if (light.environmentImage.version)
        {
            const auto prefix="visual"+std::to_string(light.id.value)+"/";
            for (auto field:environmentImageRuntimeParameters(light)) parameters.emplace(prefix+field.name,*field.value);
        }
        seedDirectionalOrPointLight(light,parameters);
        if (!authoredAnalyticSkyLight(*plan,light)) return;
        const auto prefix="visual"+std::to_string(light.id.value)+"/";
        for (auto field:analyticSkyRuntimeParameters(light)) parameters.emplace(prefix+field.name,*field.value);
    };
    if (compiled.importedSceneRender->light) seedSky(*compiled.importedSceneRender->light);
    if (compiled.importedSceneRender->sceneSnapshot)
        for (std::size_t i=0;i<compiled.importedSceneRender->sceneSnapshot->lightCount;++i)
            seedSky(compiled.importedSceneRender->sceneSnapshot->lights[i]);
    if (compiled.importedSceneRender->importedLights.selectedDefinition != 0
        && !compiled.importedSceneRender->light)
        parameters.emplace("visual" + std::to_string(compiled.importedSceneRender->sourceStableId)
            + "/" + importedlightcontrol::kParameter, compiled.importedSceneRender->importedLights.selectedMultiplier());
    if (compiled.importedSceneRender->deformation)
    {
        const auto& deformation = *compiled.importedSceneRender->deformation;
        const auto prefix = "visual" + std::to_string(deformation.deformationStableId) + "/";
        parameters.emplace(prefix + "timelineSeconds", deformation.playback.timelineSeconds);
        parameters.emplace(prefix + "speed", deformation.playback.speed);
        parameters.emplace(prefix + "clipStartTimelineSeconds", deformation.playback.clipStartTimelineSeconds);
        parameters.emplace(prefix + "timelineBeat", deformation.playback.timelineBeat);
        parameters.emplace(prefix + "clipStartBeat", deformation.playback.clipStartBeat);
        parameters.emplace(prefix + "beatsPerLoop", deformation.playback.beatsPerLoop);
        parameters.emplace(prefix + "trimStartSeconds",
                           deformation.playback.trimStartSeconds);
        parameters.emplace(prefix + "trimEndSeconds",
                           deformation.playback.trimEndSeconds);
        parameters.emplace(prefix + "weight", deformation.playback.weight);
        parameters.emplace(prefix + "selectedMorphWeight", deformation.pose.morphWeight);
        const char* translationIds[] { "boneTranslateX", "boneTranslateY", "boneTranslateZ" };
        const char* rotationIds[] { "boneRotateX", "boneRotateY", "boneRotateZ" };
        const char* scaleIds[] { "boneScaleX", "boneScaleY", "boneScaleZ" };
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            parameters.emplace(prefix + translationIds[axis], deformation.pose.translation[axis]);
            parameters.emplace(prefix + rotationIds[axis], deformation.pose.rotationDegrees[axis]);
            parameters.emplace(prefix + scaleIds[axis], deformation.pose.scale[axis]);
        }
    }
    const auto renderPrefix = "visual"
        + std::to_string(compiled.importedSceneRender->renderStableId) + "/";
    for (const auto* parameter : { "objectX", "objectY", "objectZ",
                                  "objectRotationX", "objectRotationY", "objectRotationZ",
                                  "cameraX", "cameraY", "cameraZ",
                                  "cameraRotateX", "cameraRotateY", "cameraRotateZ" })
        parameters.emplace(renderPrefix + parameter, 0.0);
    // Zero is the authored vertical field of view.
    parameters.emplace(renderPrefix + "cameraFovDegrees", 0.0);
    parameters.emplace(renderPrefix + "objectScale", 1.0);
    parameters.emplace(renderPrefix + "emissionGain", 1.0);
    if (compiled.importedSceneRender->diffractionMaterial)
        for (const auto& operation : plan->operations)
            if (operation.kind == "visual.material.diffraction-grating"
                || operation.kind == "visual.material.diffractive-foil")
            {
                const auto prefix = "visual" + std::to_string(static_cast<std::uint64_t>(operation.nodeId) + 1u) + "/";
                for (const auto& [name, value] : diffractionmaterialbinding::runtimeParameterValues(
                        *compiled.importedSceneRender->diffractionMaterial))
                    parameters.emplace(prefix + name, value);
            }
    bakedtarget::seed(*plan, compiled, parameters);
}

struct ImportedSceneStaticResourceIdentity final
{
    std::uint64_t projectGeneration = 0;
    std::uint64_t helperGeneration = 0;
};

inline std::string exactImportedScenePayloadIdentity(
    const videowire::CompiledVisualLayerPlan& plan)
{
    for (const auto& operation : plan.operations)
        if (operation.kind == visualimportedscenerender::kRenderNodeKind
            || (operation.kind == typedscenepass::kNodeKind && videowire::hasTypedScenePass(plan)))
            return operation.payloadXml;
    return {};
}

enum class VisualImportedScenePreparation : std::uint8_t
{
    notApplicable = 0,
    rendered = 1,
    rejected = 2,
    // Preview is still compiling. This is a retry state, not a refusal, and it sets no error.
    pending = 3
};

inline bool sameCompiledVisualPlan(
    const videowire::CompiledVisualLayerPlan& left,
    const videowire::CompiledVisualLayerPlan& right) noexcept
{
    if (left.clipId != right.clipId
        || left.structuralRevision != right.structuralRevision
        || left.identityMode != right.identityMode
        || left.producerValidated != right.producerValidated
        || left.error != right.error
        || left.descriptorCount != right.descriptorCount
        || left.operationCount != right.operationCount
        || left.sceneRecordCount != right.sceneRecordCount
        || left.frameOutputCount != right.frameOutputCount
        || left.peakLiveFrameCount != right.peakLiveFrameCount
        || left.allocatedFrameSlotCount != right.allocatedFrameSlotCount
        || left.nodeKinds != right.nodeKinds
        || left.nodeIds != right.nodeIds
        || left.edges.size() != right.edges.size()
        || left.ports.size() != right.ports.size()
        || left.operations.size() != right.operations.size())
        return false;

    for (std::size_t index = 0; index < left.edges.size(); ++index)
    {
        const auto& a = left.edges[index];
        const auto& b = right.edges[index];
        if (a.fromNodeId != b.fromNodeId || a.fromPort != b.fromPort
            || a.toNodeId != b.toNodeId || a.toPort != b.toPort)
            return false;
    }
    for (std::size_t index = 0; index < left.ports.size(); ++index)
    {
        const auto& a = left.ports[index];
        const auto& b = right.ports[index];
        if (a.nodeId != b.nodeId || a.port != b.port || a.channels != b.channels
            || a.direction != b.direction || a.carrier != b.carrier
            || a.dataType != b.dataType || a.pixelFormat != b.pixelFormat
            || a.colorSpace != b.colorSpace)
            return false;
    }
    for (std::size_t index = 0; index < left.operations.size(); ++index)
    {
        const auto& a = left.operations[index];
        const auto& b = right.operations[index];
        if (a.nodeId != b.nodeId || a.kind != b.kind
            || a.backendCapability != b.backendCapability
            || a.payloadXml != b.payloadXml
            || a.runtimeGrantJson != b.runtimeGrantJson)
            return false;
    }
    return true;
}

class VisualImportedScenePlanCache final
{
public:
    void publish(const videowire::CompiledVisualLayerPlan& plan,
                 ImportedSceneStaticResourceIdentity identity) noexcept
    {
        if (!entries_.empty()
            && (identity.projectGeneration != projectGeneration_
                || identity.helperGeneration != helperGeneration_))
            entries_.clear();
        projectGeneration_ = identity.projectGeneration;
        helperGeneration_ = identity.helperGeneration;
        const auto found = entries_.find(plan.clipId);
        if (found != entries_.end() && !sameCompiledVisualPlan(found->second.plan, plan))
            entries_.erase(found);
    }

    const videowire::VisualLayerExecution* resolve(
        const videowire::CompiledVisualLayerPlan& plan,
        std::string& error)
    {
        auto found = entries_.find(plan.clipId);
        if (found != entries_.end() && sameCompiledVisualPlan(found->second.plan, plan))
        {
            found->second.lastUse = ++useSerial_;
            return &found->second.execution;
        }

        videowire::VisualLayerExecution compiled;
        if (!videowire::compileVisualLayerExecution(plan, compiled, error))
            return nullptr;

        if (entries_.size() >= kMaximumEntries)
        {
            auto victim = entries_.begin();
            for (auto candidate = std::next(victim); candidate != entries_.end(); ++candidate)
                if (candidate->second.lastUse < victim->second.lastUse
                    || (candidate->second.lastUse == victim->second.lastUse
                        && candidate->first < victim->first))
                    victim = candidate;
            entries_.erase(victim);
            found = entries_.end();
        }
        Entry replacement { plan, std::move(compiled), ++useSerial_, {} };
        if (found == entries_.end())
            found = entries_.emplace(plan.clipId, std::move(replacement)).first;
        else
            found->second = std::move(replacement);
        return &found->second.execution;
    }

    // Baked-value derivations for the clip's current compiled plan. A replaced
    // or evicted plan drops its slot with the entry.
    bakedtarget::Slot* bakedTargetSlot(int clipId) noexcept
    {
        const auto found = entries_.find(clipId);
        return found == entries_.end() ? nullptr : &found->second.baked;
    }

    bakedtarget::RevisionFloor& bakedRevisionFloor(int clipId)
    {
        return revisionFloors_[clipId];
    }

    void reset() noexcept
    {
        entries_.clear();
        revisionFloors_.clear();
        projectGeneration_ = 0;
        helperGeneration_ = 0;
        useSerial_ = 0;
    }

private:
    struct Entry final
    {
        videowire::CompiledVisualLayerPlan plan;
        videowire::VisualLayerExecution execution;
        std::uint64_t lastUse = 0;
        bakedtarget::Slot baked {};
    };
    static constexpr std::size_t kMaximumEntries = 16;
    std::map<int, Entry> entries_;
    std::map<int, bakedtarget::RevisionFloor> revisionFloors_;
    std::uint64_t projectGeneration_ = 0;
    std::uint64_t helperGeneration_ = 0;
    std::uint64_t useSerial_ = 0;
};

inline double importedTimelineEvaluationSeconds(double sourceSeconds,
    const visualdeformation::RationalFrameTime& frame, bool solidOverlay)
{
    return solidOverlay ? sourceSeconds
        : static_cast<double>(frame.frame)*frame.rateDenominator/frame.rateNumerator;
}

template <typename Execution, typename Layer, typename FrameOwnerContainer>
VisualImportedScenePreparation prepareVisualImportedSceneLayerAtTime(
    const std::vector<videowire::CompiledVisualLayerPlan>& plans,
    int clipId,
    int width,
    int height,
    double sourceSeconds,
    double framesPerSecond,
    NativeImportedSceneRenderUse use,
    Execution* execution,
    VisualImportedScenePlanCache* planCache,
    Layer& layer,
    FrameOwnerContainer& frameOwners,
    std::string& error,
    const std::map<std::string, double>* runtimeParameters = nullptr,
    ImportedSceneStaticResourceIdentity staticIdentity = {},
    const MaterialFrameResolver* materialFrameResolver = nullptr,
    bool linearColor = false,
    const videotime::BeatTimeline* beatTimeline = nullptr,
    double projectSeconds = 0.0,
    geometry::GeometryExecutionCache* ownedGeometryCache = nullptr,
    const ImportedMaterialFrameResolver* composedMaterialFrameResolver = nullptr)
{
    const auto* plan = videowire::findVisualLayerPlan(plans, clipId);
    // Refuse before resetting the layer or publishing source/cache retirement.
    // Canonical V26 carries several imported leaves. The Render node draws
    // them all; Render Passes would draw leaf 0 alone, so refuse them whole.
    if (plan && std::any_of(plan->operations.begin(), plan->operations.end(), [](const auto& operation) {
            return operation.kind == typedscenepass::kNodeKind
                && visualimportedscenerender::detail::containsRequestHeader(operation.payloadXml,
                    visualimportedscenerender::kCompiledOperationHeaderV26);
        }))
    {
        error = std::string(visualimportedscenerender::kRenderPassesMultipleImportedLeavesUnavailable);
        return VisualImportedScenePreparation::rejected;
    }
    // Canonical V25 is active, including when nested inside a composition.
    if (plan && std::any_of(plan->operations.begin(), plan->operations.end(), [](const auto& operation) {
            return (operation.kind == visualimportedscenerender::kRenderNodeKind
                    || operation.kind == typedscenepass::kNodeKind)
                && visualimportedscenerender::detail::containsRequestHeader(operation.payloadXml,
                    visualimportedscenerender::kCompiledOperationHeaderV25);
        }))
    {
        error = "Imported object local transforms require a native evaluation owner";
        return VisualImportedScenePreparation::rejected;
    }
    // The header selects preflight work; only the canonical compiler authenticates
    // the mapping and its exact imported source-to-render schedule.
    if (plan && std::any_of(plan->operations.begin(), plan->operations.end(), [](const auto& operation) {
            const std::string_view payload = operation.payloadXml;
            const auto header = visualnoteinstancing::kVelocityScaleMappingTransportHeader;
            return operation.kind == "visual.3d.note-instanced-mesh"
                && payload.size() > header.size()
                && payload.substr(0, header.size()) == header
                && payload[header.size()] == '\n';
        }))
    {
        videowire::VisualLayerExecution preflight;
        if (!videowire::compileVisualLayerExecution(*plan, preflight, error))
            return VisualImportedScenePreparation::rejected;
        if (preflight.importedSceneRender && preflight.noteInstanceMapping
            && preflight.noteInstanceMapping->schemaVersion
                == visualnoteinstancing::kVelocityScaleMappingSchemaVersion)
        {
            error = "Native per-note velocity scale transport is unavailable";
            return VisualImportedScenePreparation::rejected;
        }
    }
    const auto inputLayer = layer;
    struct FrameDependencyRollback {
        const ImportedMaterialFrameResolver* resolver;
        bool complete = false;
        ~FrameDependencyRollback() { if (resolver && !complete) resolver->discard(); }
    } frameDependencyRollback {composedMaterialFrameResolver};

    const bool solidCandidate=plan && videowire::isImportedParticleOverlayPlan(plans,clipId)
        && std::any_of(plan->operations.begin(),plan->operations.end(),[](const auto& operation) {
            return operation.kind=="visual.particles"
                && videowire::visualPayloadFloat(operation.payloadXml,"simulationSpace",0)==1;
        });
    modelpayload::ImportedSceneExecutionReceipt::ParentPresentationGuard parentPresentation;
    if constexpr (std::is_same_v<Execution,modelpayload::ImportedScenePayloadExecution>)
        if (execution)
        {
            if constexpr (HasImportedParentPresentation<Layer>::value)
                parentPresentation=execution->parentPresentationGuard(inputLayer.importedParentPresentation,clipId);
            else
                parentPresentation=execution->parentPresentationGuard(clipId);
        }
    struct SolidLayerRollback {
        Layer& layer; const Layer& previous;
        const modelpayload::ImportedSceneExecutionReceipt::ParentPresentationGuard& parentPresentation;
        bool active=false;
        ~SolidLayerRollback() { if (active && !parentPresentation.retired()) layer=previous; }
    } rollback{layer,inputLayer,parentPresentation,solidCandidate || (plan && parentPresentation.present()) || composedMaterialFrameResolver};
    const auto canonicalBlockCFrame = layer.canonicalBlockCFrame;
    const auto audioFeatures = [&]
    {
        if constexpr (HasImportedVertexSpectrum<Layer>::value) return layer.audioFeatures;
        else return std::array<float, 0> {};
    }();
    const bool audioPresent = [&]
    {
        if constexpr (HasImportedVertexSpectrum<Layer>::value) return layer.audioPresent;
        else return false;
    }();
    layer = Layer {};
    if constexpr (HasImportedVertexSpectrum<Layer>::value)
    {
        layer.audioFeatures = audioFeatures;
        layer.audioPresent = audioPresent;
    }
    const auto reject = [&error](const char* message)
    {
        error = message;
        return VisualImportedScenePreparation::rejected;
    };
    if (plan == nullptr) return VisualImportedScenePreparation::notApplicable;

    const auto exactPayloadIdentity = exactImportedScenePayloadIdentity(*plan);
    if (execution != nullptr)
    {
        loopstep::Scope publishStaticScope ("advance:imported-scene:static-publish");
        execution->publishStaticPayload(clipId, staticIdentity.projectGeneration,
                                        staticIdentity.helperGeneration,
                                        exactPayloadIdentity);
    }
    if constexpr (std::is_same_v<Execution,modelpayload::ImportedScenePayloadExecution>)
        if (execution && !execution->parentPublicationReady())
            return reject("imported parent publication identity exceeds permanent work or logical CPU capacity");

    if (planCache == nullptr)
        return reject("imported scene render requires a production-owned compiled plan cache");
    {
        loopstep::Scope planPublishScope ("advance:imported-scene:plan-publish");
        planCache->publish(*plan, staticIdentity);
    }
    // A cache miss compiles the plan here. Cache hits return at once.
    const videowire::VisualLayerExecution* compiled = nullptr;
    {
        loopstep::Scope resolveScope ("advance:imported-scene:resolve-compile");
        compiled = planCache->resolve(*plan, error);
    }
    if (compiled == nullptr)
        return VisualImportedScenePreparation::rejected;
    if (!compiled->importedSceneRender.has_value())
        return VisualImportedScenePreparation::notApplicable;
    const bool geometryFamily = !compiled->importedSceneRender->timelineGeometryFamily.empty();
    const bool solidOverlay=compiled->importedParticleOverlay && compiled->particleParameters.simulationSpace==1;
    rollback.active=composedMaterialFrameResolver || solidOverlay || geometryFamily
        || compiled->importedSceneRender->importedParentIntent.has_value();
    std::optional<videorender::ParticleParams> preparedSolidParameters;
    std::shared_ptr<videorender::ParticleImportedSolidFrame> solidReceipt;
    std::shared_ptr<const videorender::WaterCoupledFrame> preparedCoupledWater;
    if (execution == nullptr)
        return reject("imported scene render requires the processor-owned exact payload store");
    if (width <= 0 || height <= 0 || width > 8192 || height > 8192
        || static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height)
            > 67108864u)
        return reject("imported scene render dimensions exceed the product bound");

    if (compiled->typedScenePassExtent)
    {
        if constexpr (!HasImportedTextureOwner<Layer>::value || !HasImportedLinearImage<Layer>::value)
            return reject("Render Passes requires a compositor layer with an owned native texture");
        linearColor = true;
    }

    const auto renderExtent = compiled->typedScenePassExtent.value_or(renderpassoutput::Extent {
        static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)});
    if (renderExtent.width == 0 || renderExtent.height == 0 || renderExtent.width > 8192
        || renderExtent.height > 8192
        || static_cast<std::uint64_t>(renderExtent.width) * renderExtent.height > 67108864u)
        return reject("imported scene render dimensions exceed the product bound");

    if (compiled->importedSceneRender->glassCaustic)
    {
        if constexpr (!HasImportedTextureOwner<Layer>::value || !HasImportedLinearImage<Layer>::value)
            return reject("Glass caustics require a compositor layer with an owned linear native texture");
        linearColor = true;
    }

    if constexpr (!HasImportedParentInspector<Execution>::value)
        if (compiled->importedSceneRender->importedParentIntent)
            return reject("Imported Scene Compose parenting requires its native parent binding owner and receipt");

    modelpayload::ImportedSceneRequest request;
    request.parentIntent = compiled->importedSceneRender->importedParentIntent
        ? &*compiled->importedSceneRender->importedParentIntent : nullptr;
    request.importedScene = compiled->importedSceneRender->importedScene;
    const auto& importedSource = request.importedScene ? *request.importedScene : *compiled->importedSceneRender;
    request.asset = importedSource.asset;
    request.sourceStableId = importedSource.sourceStableId;
    request.renderStableId = importedSource.renderStableId;
    request.sceneIndex = importedSource.sceneIndex;
    request.sourceCameraNodeId = importedSource.sourceCameraNodeId;
    request.objectNodeStableId = importedSource.objectNodeStableId;
    request.objectMeshStableId = importedSource.objectMeshStableId;
    request.objectLocalTransform = importedSource.objectLocalTransform;
    request.dimensions = {renderExtent.width, renderExtent.height};
    request.deformation = importedSource.deformation;
    request.runtimeInputs.imageOutput = compiled->importedSceneRender->imageOutput;
    request.runtimeInputs.linearColor = linearColor;
    if (linearColor && request.runtimeInputs.imageOutput != renderpassoutput::Output::Color
        && request.runtimeInputs.imageOutput != renderpassoutput::Output::Emission)
        return reject("HDR Scene3D requires Image/Color or Emission, not a data-pass visualization");
    request.runtimeInputs.rawExportMask = use == NativeImportedSceneRenderUse::Export
        ? compiled->importedSceneRender->rawExportMask : 0;
    // Solid 3D Motion is the Motion of the whole composed draw, scene and
    // bodies together. Export Motion Data applies to export renders only, as
    // for a standalone Solid 3D draw.
    const bool solidMotion=solidOverlay && videorender::particleSolidMotionRequested(
        compiled->particleParameters,use==NativeImportedSceneRenderUse::Export);
    if (solidMotion)
    {
        if (linearColor && compiled->particleParameters.solidMotionView)
            return reject("HDR Solid 3D requires the Color image, not the Motion View");
        if (compiled->particleParameters.solidMotionView)
            request.runtimeInputs.imageOutput=renderpassoutput::Output::Motion;
        if (use==NativeImportedSceneRenderUse::Export && compiled->particleParameters.solidExportMotion)
            request.runtimeInputs.rawExportMask|=1u<<static_cast<unsigned>(renderpassoutput::Output::Motion);
    }
    request.runtimeInputs.passComposite = compiled->passComposite;
    request.runtimeInputs.passProgram = compiled->passProgram;
    // Any Motion from the composed draw needs each body's prior transform.
    const bool solidPrior=solidOverlay && arbitgpu::sceneUsesMotionPass(request.runtimeInputs);
    request.projectGeneration = staticIdentity.projectGeneration;
    request.helperGeneration = staticIdentity.helperGeneration;
    request.clipId = clipId;
    request.staticPayloadIdentity = exactPayloadIdentity;
    if (compiled->noteInstanceMapping)
    {
        if (!canonicalBlockCFrame)
            return reject("note-instanced imported scene requires one canonical Block C frame");
        request.runtimeInputs.canonicalBlockCFrame = canonicalBlockCFrame;
        request.runtimeInputs.noteInstanceMapping = compiled->noteInstanceMapping;
    }
    // Each imported leaf takes its own runtime controls, named by its own
    // deformation stable ID. Leaf 0 keeps the released one-leaf path.
    const auto rejectBool = [&](const char* message) { error = message; return false; };
    const auto applyDeformationParameters = [&](visualanimationimport::Request& deformation) -> bool
    {
        const auto prefix = "visual"
            + std::to_string(deformation.deformationStableId) + "/";
        const auto value = [&] (const char* parameter, double fallback)
        {
            const auto found = runtimeParameters->find(prefix + parameter);
            return found != runtimeParameters->end() && std::isfinite(found->second)
                ? found->second : fallback;
        };
        deformation.playback.timelineSeconds = std::clamp(
            value("timelineSeconds", deformation.playback.timelineSeconds),
            -365.0 * 24.0 * 60.0 * 60.0, 365.0 * 24.0 * 60.0 * 60.0);
        for (const auto* name : { "timelineBeat", "clipStartBeat", "beatsPerLoop", "clipStartTimelineSeconds" })
        {
            const auto found = runtimeParameters->find(prefix + name);
            if (found != runtimeParameters->end() && !std::isfinite(found->second))
                return rejectBool("Imported animation timing parameter is nonfinite");
        }
        deformation.playback.timelineBeat = value("timelineBeat", deformation.playback.timelineBeat);
        deformation.playback.clipStartBeat = value("clipStartBeat", deformation.playback.clipStartBeat);
        deformation.playback.beatsPerLoop = value("beatsPerLoop", deformation.playback.beatsPerLoop);
        deformation.playback.clipStartTimelineSeconds = value(
            "clipStartTimelineSeconds", deformation.playback.clipStartTimelineSeconds);
        if (!visualanimationoperation::valid(deformation))
            return rejectBool("Imported animation timing parameters exceed the playback bound");
        deformation.playback.speed = std::clamp(
            value("speed", deformation.playback.speed), -64.0, 64.0);
        deformation.playback.trimStartSeconds = std::clamp(
            value("trimStartSeconds", deformation.playback.trimStartSeconds),
            0.0, 86400.0);
        deformation.playback.trimEndSeconds = std::clamp(
            value("trimEndSeconds", deformation.playback.trimEndSeconds),
            0.0, 86400.0);
        deformation.playback.weight = std::clamp(
            value("weight", deformation.playback.weight), 0.0, 1.0);
        auto& pose = deformation.pose;
        pose.morphWeight = std::clamp(value("selectedMorphWeight", pose.morphWeight), -16.0, 16.0);
        // Per-key deltas come from the runtime morph<i>Weight parameters. An absent or zero
        // key adds nothing, and a nonfinite value is refused rather than dropped.
        pose.morphWeightDeltas.clear();
        for (std::size_t key = 0; key < visualanimation::kMaximumMorphWeightDeltas; ++key)
        {
            const auto found = runtimeParameters->find(
                prefix + visualanimation::morphWeightDeltaParameter(key));
            if (found == runtimeParameters->end()) continue;
            if (!std::isfinite(found->second))
                return rejectBool("Imported animation morph weight delta is nonfinite");
            const auto delta = std::clamp(static_cast<double>(found->second), -16.0, 16.0);
            if (delta != 0.0)
                pose.morphWeightDeltas.emplace_back(static_cast<std::uint32_t>(key), delta);
        }
        const char* translationIds[] { "boneTranslateX", "boneTranslateY", "boneTranslateZ" };
        const char* rotationIds[] { "boneRotateX", "boneRotateY", "boneRotateZ" };
        const char* scaleIds[] { "boneScaleX", "boneScaleY", "boneScaleZ" };
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            pose.translation[axis] = std::clamp(value(translationIds[axis], pose.translation[axis]), -10000.0, 10000.0);
            pose.rotationDegrees[axis] = std::clamp(value(rotationIds[axis], pose.rotationDegrees[axis]), -3600.0, 3600.0);
            pose.scale[axis] = std::clamp(value(scaleIds[axis], pose.scale[axis]), static_cast<double>(0.0001f), 1000.0);
        }
        return true;
    };
    if (request.deformation && runtimeParameters != nullptr
        && !applyDeformationParameters(*request.deformation))
        return VisualImportedScenePreparation::rejected;
    for (const auto& leaf : compiled->importedSceneRender->additionalImportedScenes)
    {
        if (!leaf) return reject("Scene Compose imported leaf is missing");
        modelpayload::ImportedSceneLeafRequest entry;
        entry.scene = leaf;
        entry.deformation = leaf->deformation;
        if (entry.deformation && runtimeParameters != nullptr
            && !applyDeformationParameters(*entry.deformation))
            return VisualImportedScenePreparation::rejected;
        request.additionalImportedScenes.push_back(std::move(entry));
    }
    if (runtimeParameters != nullptr)
    {
        const auto prefix = "visual"
            + std::to_string(compiled->importedSceneRender->renderStableId) + "/";
        const auto value = [&] (const char* parameter, double fallback,
                                double minimum, double maximum)
        {
            const auto found = runtimeParameters->find(prefix + parameter);
            const auto candidate = found != runtimeParameters->end()
                && std::isfinite(found->second) ? found->second : fallback;
            return static_cast<float>(std::clamp(candidate, minimum, maximum));
        };
        request.runtimeInputs.objectTranslationOffset = std::array<float, 3> {
            value("objectX", 0.0, -1000000.0, 1000000.0),
            value("objectY", 0.0, -1000000.0, 1000000.0),
            value("objectZ", 0.0, -1000000.0, 1000000.0)
        };
        request.runtimeInputs.objectRotationDegrees = std::array<float, 3> {
            value("objectRotationX", 0.0, -360.0, 360.0),
            value("objectRotationY", 0.0, -360.0, 360.0),
            value("objectRotationZ", 0.0, -360.0, 360.0)
        };
        request.runtimeInputs.objectScale = value("objectScale", 1.0, 0.01, 100.0);
        request.runtimeInputs.cameraTranslationOffset = std::array<float, 3> {
            value("cameraX", 0.0, -1000000.0, 1000000.0),
            value("cameraY", 0.0, -1000000.0, 1000000.0),
            value("cameraZ", 0.0, -1000000.0, 1000000.0)
        };
        request.runtimeInputs.cameraRotationDegrees = std::array<float, 3> {
            value("cameraRotateX", 0.0, -360.0, 360.0),
            value("cameraRotateY", 0.0, -360.0, 360.0),
            value("cameraRotateZ", 0.0, -360.0, 360.0)
        };
        // Values in (0, 1) round up to the lowest lowered field of view. Zero or
        // negative values keep the authored field of view.
        const auto fieldOfView = value("cameraFovDegrees", 0.0, 0.0, 179.0);
        request.runtimeInputs.cameraVerticalFovDegrees = fieldOfView > 0.0f
            ? std::max(fieldOfView, 1.0f) : 0.0f;
        request.runtimeInputs.emissionGain = value("emissionGain", 1.0, 0.0, 64.0);
    }
    if (!importedanimation::importedAnimationFrameIdentity(
            sourceSeconds, framesPerSecond, request.frame, error))
        return VisualImportedScenePreparation::rejected;
    if (request.deformation)
    {
        if (!importedanimation::applyImportedAnimationPlaybackClock(
                request.deformation->playback, request.frame, beatTimeline, projectSeconds, error))
            return VisualImportedScenePreparation::rejected;
    }
    for (auto& leaf : request.additionalImportedScenes)
        if (leaf.deformation && !importedanimation::applyImportedAnimationPlaybackClock(
                leaf.deformation->playback, request.frame, beatTimeline, projectSeconds, error))
            return VisualImportedScenePreparation::rejected;
    request.structuralRevision = plan->structuralRevision;
    request.evaluationRevision = compiled->importedSceneRender->evaluationRevision;
    request.material = importedSource.material;
    request.surfacePrograms = compiled->importedSceneRender->surfacePrograms;
    request.glassCaustic = compiled->importedSceneRender->glassCaustic;
    if (!request.surfacePrograms.empty()) request.runtimeInputs.canonicalBlockCFrame = canonicalBlockCFrame;
    request.diffractionMaterial = compiled->importedSceneRender->diffractionMaterial;
    if (request.diffractionMaterial && runtimeParameters)
        for (const auto& operation : plan->operations)
            if (operation.kind == "visual.material.diffraction-grating"
                || operation.kind == "visual.material.diffractive-foil")
            {
                const auto prefix = "visual" + std::to_string(static_cast<std::uint64_t>(operation.nodeId) + 1u) + "/";
                const auto baseline = diffractionmaterialbinding::runtimeParameterValues(*request.diffractionMaterial);
                for (const auto& [key, value] : *runtimeParameters)
                    if (key.compare(0, prefix.size(), prefix) == 0)
                    {
                        const auto name = key.substr(prefix.size());
                        const auto original = baseline.find(name);
                        if (original == baseline.end() || !std::isfinite(value))
                            return reject("Diffraction modulation targets an unavailable or nonfinite parameter");
                        if (value != original->second) request.runtimeInputs.diffractionParameters.emplace(name, value);
                    }
            }
    request.camera = compiled->importedSceneRender->camera;
    if (compiled->importedSceneRender->materialField)
    {
        if (!request.diffractionMaterial) return reject("Material Field requires an authored diffraction material");
        videowire::geometry::RuntimeFieldEvaluation evaluation;
        evaluation.timelineSeconds = static_cast<double>(request.frame.frame)
            * request.frame.rateDenominator / request.frame.rateNumerator;
        evaluation.scoreAt = [&](const auto& operation, const auto& positions, auto& result, auto& diagnostic)
        {
            return scorefield::evaluateSampleField(operation, positions, canonicalBlockCFrame, result, diagnostic);
        };
        if (!materialfield::evaluate(*compiled->importedSceneRender->materialField, evaluation,
                *request.diffractionMaterial, request.runtimeInputs.diffractionParameters, error))
            return VisualImportedScenePreparation::rejected;
        request.runtimeInputs.canonicalBlockCFrame = canonicalBlockCFrame;
    }
    request.importedLights = importedSource.importedLights;
    if (runtimeParameters && request.importedLights.selectedDefinition != 0)
    {
        const auto parameter = runtimeParameters->find("visual"
            + std::to_string(importedSource.sourceStableId) + "/" + importedlightcontrol::kParameter);
        if (parameter != runtimeParameters->end())
        {
            if (!std::isfinite(parameter->second)) return reject("Imported light modulation must be finite");
            const auto multiplier = static_cast<float>(std::clamp(parameter->second, 0.0,
                static_cast<double>(importedlightcontrol::kMaximumMultiplier)));
            request.importedLights.intensityMultipliers[request.importedLights.selectedDefinition]
                = multiplier == 0 ? 0.0f : multiplier;
        }
    }
    request.light = compiled->importedSceneRender->light;
    if (compiled->importedSceneRender->noteLight)
    {
        if (!request.light || !applyNoteReactiveLight(*request.light,
                *compiled->importedSceneRender->noteLight, canonicalBlockCFrame))
            return reject("note-reactive light requires the canonical score frame for this preview or export time");
    }
    request.sceneSnapshot = compiled->importedSceneRender->sceneSnapshot;
    // PBR numbers and the bounded displacement limit are baked into the plan.
    // A modulated value derives a scene or material once per quantized change.
    auto* baked = planCache->bakedTargetSlot(plan->clipId);
    if (baked == nullptr)
        return reject("imported scene render lost its compiled plan cache entry");
    if (!bakedtarget::evaluate(*plan, *compiled, runtimeParameters, *baked,
            planCache->bakedRevisionFloor(plan->clipId), error))
        return VisualImportedScenePreparation::rejected;
    if (baked->material) request.material = baked->material;
    if (baked->leaf) request.importedScene = baked->leaf;
    if (baked->surfacePrograms) request.surfacePrograms = *baked->surfacePrograms;
    if (baked->scene)
    {
        request.sceneSnapshot = baked->scene;
        if (request.material && !request.importedScene) request.material->sceneSnapshot = request.sceneSnapshot;
    }
    // Per-object programs bind the frame's scene, whether it was derived or compiled.
    for (auto& program : request.surfacePrograms)
        program.program.material.sceneSnapshot = request.sceneSnapshot;
    // Resolve authored controls once for this exact request frame. Preview and
    // export share this path; static scene/resource identity is never rewritten.
    if (request.light && !evaluateAnalyticSkyRuntime(*plan,runtimeParameters,*request.light))
        return reject("Analytic sky runtime override exceeds its finite radiance bound");
    if (request.sceneSnapshot && runtimeParameters)
    {
        bool hasAuthoredSky=false;
        for (std::size_t i=0;i<request.sceneSnapshot->lightCount;++i)
            hasAuthoredSky |= authoredAnalyticSkyLight(*plan,request.sceneSnapshot->lights[i])
                || request.sceneSnapshot->lights[i].environmentImage.version!=0;
        if (hasAuthoredSky)
        {
            request.runtimeInputs.animatedLights.assign(request.sceneSnapshot->lights.begin(),
                request.sceneSnapshot->lights.begin()+request.sceneSnapshot->lightCount);
            for (auto& light:request.runtimeInputs.animatedLights)
                if (!evaluateAnalyticSkyRuntime(*plan,runtimeParameters,light))
                    return reject("Analytic sky runtime controls exceed their finite radiance bound");
        }
    }
    // Directional and point light controls ride the same per-frame light records as the
    // Environment light. Timeline Geometry3D and Solid 3D rebuild their scene from their own
    // owner, so the PBR route gate also withholds these controls there.
    if (runtimeParameters && bakedtarget::pbrRoute(*compiled))
    {
        if (request.light && !evaluateDirectionalOrPointLight(*runtimeParameters, *request.light))
            return reject("Directional or point light modulation must be finite and in range");
        if (request.sceneSnapshot)
        {
            // Modulated fields ride lightOverrides. The backend reader applies them to the
            // record each index reads, so the authored scene is never copied or rewritten here.
            // Caller animatedLights stay untouched; a mismatched layout is refused, never guessed.
            request.runtimeInputs.lightOverrides.clear();
            for (std::size_t i = 0; i < request.sceneSnapshot->lightCount; ++i)
            {
                arbitgpu::NativeImportedLightOverride modulated;
                if (!directionalOrPointOverride(*runtimeParameters, request.sceneSnapshot->lights[i], i, modulated))
                    return reject("Directional or point light modulation must be finite and in range");
                if (modulated.any()) request.runtimeInputs.lightOverrides.push_back(modulated);
            }
            if (!request.runtimeInputs.lightOverrides.empty() && !request.runtimeInputs.animatedLights.empty()
                && request.runtimeInputs.animatedLights.size() != request.sceneSnapshot->lightCount)
                return reject("Directional or point light overrides need the authored light layout");
        }
    }
    if (solidOverlay)
    {
        if (renderExtent.width!=static_cast<std::uint32_t>(width)
            || renderExtent.height!=static_cast<std::uint32_t>(height))
            return reject("Imported Solid 3D requires the same native scene and compositor output extent");
        if (request.material && request.material->vertexModifier)
            return reject("Imported solid composition cannot extend the one-object graph vertex-modifier path");
        if constexpr (!HasImportedSolidInput<Layer>::value)
            return reject("Imported Solid 3D requires the owned project history and particle clock");
        else
        {
            if (std::any_of(plan->edges.begin(),plan->edges.end(),[&](const auto& edge) {
                return edge.toNodeId==compiled->particleNodeId && edge.toPort==0;
            })) return reject("Imported Solid 3D composition does not support Event-triggered input");
            auto parameters=compiled->particleParameters;
            parameters.history=inputLayer.particleHistory;
            parameters.historyProjectSeconds=inputLayer.particleProjectSeconds;
            parameters.historyClipId=clipId; parameters.historyNodeId=compiled->particleNodeId;
            parameters.geometryRevision=plan->structuralRevision;
            const auto seconds=inputLayer.shaderClock.timeSec;
            if (parameters.historyProjectSeconds!=projectSeconds
                || !videorender::particleHistoryReady(parameters,error,seconds))
                return reject("Imported solid replay requires the exact current project history clock");
            // Refuse what the prior replay cannot represent before either replay runs.
            if (solidPrior && !videorender::admitParticleSolidMotion(parameters,error))
                return VisualImportedScenePreparation::rejected;
            videorender::ParticlePointReplayTimes solidTimes;
            auto solids=std::make_shared<videorender::ParticleSolidState>();
            videorender::ParticleBodyState bodies;
            if (videorender::hasBodyCoupledWater(parameters))
            {
                if (sourceSeconds!=projectSeconds || seconds!=projectSeconds)
                    return reject("Imported water coupling requires the same project-zero source, body and scene clock");
                videorender::WaterCoupledFrame water;
                if (!videorender::replayWaterCoupledBodies(parameters,seconds,float(width)/height,
                    canonicalBlockCFrame.get(),bodies,*solids,nullptr,error,nullptr,&water))
                    return VisualImportedScenePreparation::rejected;
                preparedCoupledWater=std::make_shared<const videorender::WaterCoupledFrame>(std::move(water));
            }
            else
            {
                bodies=videorender::replayParticleBodies(parameters,seconds,float(width)/height,
                    canonicalBlockCFrame.get(),&error,solids.get(),solidPrior ? &solidTimes:nullptr);
                if (!error.empty()) return VisualImportedScenePreparation::rejected;
            }
            std::optional<videorender::ParticleParams> replayParameters;
            if (solidPrior) replayParameters=parameters;
            auto currentParameters=parameters;
            if (!videorender::sampleParticleParameters(parameters,currentParameters,parameters.historyProjectSeconds,error))
                return VisualImportedScenePreparation::rejected;
            const auto window=videorender::particleReplayWindow(currentParameters,seconds);
            if (!videorender::sampleParticleParameters(parameters,parameters,
                double(window.start+window.age)/videorender::kParticleReplayTicksPerSecond,error))
                return VisualImportedScenePreparation::rejected;
            parameters.preparedBodies=std::make_shared<const videorender::ParticleBodyState>(std::move(bodies));
            parameters.preparedSolids=std::move(solids);
            if (solidPrior)
            {
                // The replay one presentation frame earlier gives each body its
                // own prior transform.
                parameters.preparedSolidMotion=videorender::replayParticleSolidMotionPrior(*replayParameters,
                    seconds,double(request.frame.rateDenominator)/request.frame.rateNumerator,
                    float(width)/height,canonicalBlockCFrame.get(),*parameters.preparedSolids,solidTimes,
                    parameters.bodyRadius,error);
                if (!parameters.preparedSolidMotion) return VisualImportedScenePreparation::rejected;
            }
            preparedSolidParameters=std::move(parameters);
            solidReceipt=std::make_shared<videorender::ParticleImportedSolidFrame>();
            solidReceipt->owner={staticIdentity.projectGeneration,staticIdentity.helperGeneration,
                videowire::runtimeDeviceGeneration(geometry::geometryExecutionCache().source.geometryCoreCapabilities()),
                static_cast<std::uint64_t>(clipId),plan->structuralRevision,
                use==NativeImportedSceneRenderUse::Preview ? geometry::PlanUse::preview:geometry::PlanUse::exportRender};
            if (!solidReceipt->owner.valid()) return reject("Imported solid replay requires exact native scene ownership");
            solidReceipt->nodeId=compiled->particleNodeId;
            solidReceipt->projectSeconds=projectSeconds; solidReceipt->clipSeconds=seconds;
            solidReceipt->bodies=preparedSolidParameters->preparedBodies;
            solidReceipt->state=preparedSolidParameters->preparedSolids;
            solidReceipt->geometry=preparedSolidParameters->geometryBinding;
            solidReceipt->water=preparedCoupledWater;
        }
    }
    if (geometryFamily)
    {
        auto& cache = ownedGeometryCache ? *ownedGeometryCache : geometry::geometryExecutionCache();
        const geometry::PlanOwnerIdentity owner {
            staticIdentity.projectGeneration, staticIdentity.helperGeneration,
            videowire::runtimeDeviceGeneration(cache.source.geometryCoreCapabilities()),
            static_cast<std::uint64_t>(clipId), plan->structuralRevision,
            use == NativeImportedSceneRenderUse::Preview ? geometry::PlanUse::preview : geometry::PlanUse::exportRender };
        if (!request.sceneSnapshot || !std::isfinite(projectSeconds)
            || !cache.runtime.reconcileGeneration(owner.projectGeneration, owner.helperGeneration,
                owner.deviceGeneration, error))
        {
            if (error.empty()) error = "Timeline Geometry3D family requires its full authored scene and finite project clock";
            return VisualImportedScenePreparation::rejected;
        }
        const auto& bindings = compiled->importedSceneRender->timelineGeometryFamily;
        const auto family = use == NativeImportedSceneRenderUse::Preview
            ? cache.runtime.admitTimelineScenePreview(owner, *request.sceneSnapshot, bindings, {}, error)
            : cache.runtime.admitTimelineSceneExport(owner, *request.sceneSnapshot, bindings, {}, error);
        if (!family) return VisualImportedScenePreparation::rejected;
        // Geometry and Block C keep separate generation authorities. The
        // evaluator authenticates borrowed history to the current immutable
        // canonical frame synchronously; no borrowed reference escapes.
        const videorender::ParticleHistorySource* waterHistory = nullptr;
        if constexpr (videowire::HasGeometryParticleInput<Layer>::value)
            waterHistory = inputLayer.particleHistory.get();
        // Solid replay sampled each child at sourceSeconds, so the scene must too.
        const double frameSeconds = importedTimelineEvaluationSeconds(sourceSeconds, request.frame, solidOverlay);
        request.sceneSnapshot = cache.familyEvaluator.evaluate(owner, family, request.sceneSnapshot,
            frameSeconds, cache.familyEvaluationLimits, error, nullptr, projectSeconds,
            nullptr, waterHistory, canonicalBlockCFrame);
        if (!request.sceneSnapshot) return VisualImportedScenePreparation::rejected;
        if (request.material && !request.importedScene) request.material->sceneSnapshot = request.sceneSnapshot;
        for (auto& program : request.surfacePrograms)
            program.program.material.sceneSnapshot = request.sceneSnapshot;
        request.runtimeInputs.canonicalBlockCFrame = canonicalBlockCFrame;
        // The family evaluator replaces child vertices in place and keeps vertex
        // order, so the retained prior positions line up with this frame.
        request.deformedSceneMotion = true;
    }
    if (compiled->importedSceneRender->timelineGeometry)
    {
        auto& cache = geometry::geometryExecutionCache();
        const geometry::PlanOwnerIdentity owner {
            staticIdentity.projectGeneration, staticIdentity.helperGeneration,
            videowire::runtimeDeviceGeneration(cache.source.geometryCoreCapabilities()),
            static_cast<std::uint64_t>(clipId), plan->structuralRevision,
            use == NativeImportedSceneRenderUse::Preview ? geometry::PlanUse::preview : geometry::PlanUse::exportRender };
        const double frameSeconds = importedTimelineEvaluationSeconds(sourceSeconds,request.frame,solidOverlay);
        const videorender::ParticleHistorySource* waterHistory = nullptr;
        if constexpr (videowire::HasGeometryParticleInput<Layer>::value)
            waterHistory = inputLayer.particleHistory.get();
        request.sceneSnapshot = geometry::evaluateComposedTimelineGeometry(cache.runtime, owner,
            *compiled->importedSceneRender->timelineGeometry, request.sceneSnapshot, frameSeconds, error,
            waterHistory, projectSeconds, canonicalBlockCFrame, preparedCoupledWater.get());
        if (!request.sceneSnapshot) return VisualImportedScenePreparation::rejected;
        if (request.material && !request.importedScene) request.material->sceneSnapshot = request.sceneSnapshot;
        for (auto& program : request.surfacePrograms)
            program.program.material.sceneSnapshot = request.sceneSnapshot;
        // Timeline mesh replacement keeps vertex ids and order. The deformation
        // depends on water history and coupled bodies, so the payload keeps the
        // evaluated positions instead of evaluating the prior frame again.
        request.deformedSceneMotion = true;
    }
    if (solidOverlay)
    {
        if (!preparedSolidParameters || !solidReceipt || !compiled->importedSceneRender->sceneSnapshot
            || (!compiled->importedSceneRender->timelineGeometry && !geometryFamily))
            return reject("Imported Solid 3D requires the prepared TimelineScene3D composition owner");
        solidReceipt->sourceScene=request.sceneSnapshot;
        // The scene part keeps its retained positions. Each appended body is
        // its own object with its prior transform from the prior replay.
        request.retainedMotionVertexCount=request.sceneSnapshot ? request.sceneSnapshot->vertexCount : 0;
        std::optional<videorender::ParticleSolidMotionPrior> bodyPrior;
        if (const auto& motion=preparedSolidParameters->preparedSolidMotion)
            bodyPrior.emplace(videorender::ParticleSolidMotionPrior{motion->previous,motion->previousBirths,
                motion->currentBirths,motion->previousBodyRadius});
        const auto* prior=bodyPrior ? &*bodyPrior : nullptr;
        request.sceneSnapshot=geometryFamily
            ? videorender::appendParticleSolidsToFamilyScene(request.sceneSnapshot,
                *compiled->importedSceneRender->sceneSnapshot,compiled->importedSceneRender->timelineGeometryFamily,
                *preparedSolidParameters,*preparedSolidParameters->preparedSolids,compiled->particleNodeId,error,prior)
            : videorender::appendParticleSolidsToScene(request.sceneSnapshot,
                *compiled->importedSceneRender->sceneSnapshot,*compiled->importedSceneRender->timelineGeometry,
                *preparedSolidParameters,*preparedSolidParameters->preparedSolids,compiled->particleNodeId,error,prior);
        if (!request.sceneSnapshot) return VisualImportedScenePreparation::rejected;
        solidReceipt->scene=request.sceneSnapshot;
        if (request.material && !request.importedScene) request.material->sceneSnapshot=request.sceneSnapshot;
        for (auto& program:request.surfacePrograms) program.program.material.sceneSnapshot=request.sceneSnapshot;
    }
    const bool motionUsed = request.runtimeInputs.imageOutput == renderpassoutput::Output::Motion
        || arbitgpu::sceneUsesMotionPass(request.runtimeInputs);
    if (request.deformedSceneMotion && motionUsed)
    {
        // An imported composition merges a second vertex layout into a new
        // owner each frame, and no prior positions are kept for it.
        if (request.importedScene)
            return reject("Motion output is unavailable for timeline Geometry3D composed with an imported scene");
    }
    request.runtimeInputs.timeSeconds = static_cast<float>(sourceSeconds);
    if (request.material && request.material->vertexModifier)
    {
        const bool usesSpectrum = std::any_of(request.material->vertexModifier->records.begin(),
            request.material->vertexModifier->records.end(), [](const auto& record)
            { return record.operation == videowire::VertexModifierOperation::audioParameter; });
        if (usesSpectrum)
        {
            if constexpr (HasImportedVertexSpectrum<Layer>::value)
            {
                if (layer.audioPresent)
                {
                    const auto& bands = layer.audioFeatures.bands;
                    if (bands.size() != request.runtimeInputs.vertexSpectrum.size())
                        return reject("Imported vertex deformation requires the shared 64-band audio frame");
                    std::copy(bands.begin(), bands.end(), request.runtimeInputs.vertexSpectrum.begin());
                }
            }
            else return reject("Imported vertex deformation requires the shared audio frame");
        }
        // The native draw runs the program again at the prior frame's time,
        // spectrum and prior skinned or timeline position. Inputs other than
        // position come from the current vertex, so refuse programs needing them.
        const auto prior = !motionUsed ? videowire::VertexModifierPriorGeometry::none
            : request.deformedSceneMotion ? videowire::VertexModifierPriorGeometry::retainedPositions
            : request.deformation ? videowire::VertexModifierPriorGeometry::skinnedPose
            : videowire::VertexModifierPriorGeometry::none;
        if (const auto refusal = videowire::vertexModifierPriorGeometryRefusal(
                request.material->vertexModifier->records, prior))
            return reject(refusal);
    }

    const auto materialFrameEndpoint = request.diffractionMaterial && request.diffractionMaterial->graphFrame
        ? request.diffractionMaterial->graphFrame
        : request.material && surfacematerialbinding::hasGraphFrameInput(request.material->binding)
            ? request.material->binding.textures.front().graphFrame : geometrysurfacematerial::objectFrameEndpoint(request.surfacePrograms);
    if (materialFrameEndpoint)
    {
        if (!compiled->materialFrameSource || (!composedMaterialFrameResolver
            && (materialFrameResolver == nullptr || !*materialFrameResolver)))
            return reject("Graph Frame texture rendering requires an owned native Frame resolver");
        const auto source = std::find_if(plan->operations.begin(), plan->operations.end(),
            [&](const auto& operation) { return operation.nodeId == materialFrameEndpoint->node; });
        if (source != plan->operations.end() && source->kind == "video.layer.source"
            && !videowire::isGeometryCoreRenderPlan(plans, clipId) && !composedMaterialFrameResolver)
        {
            int sourceClip = 0;
            if (!videowire::materialframe::sourceClip(*plan, *materialFrameEndpoint, sourceClip, error))
                return VisualImportedScenePreparation::rejected;
            // A bare decoder callback has no saved clip/time provenance.
            // Only the composed factory replaces this safe legacy refusal.
            if (sourceClip != clipId)
                return reject("Imported Surface Layer Source requires its saved clip and frame-time resolver");
        }
        const MaterialFrameEvaluation evaluation {
            *materialFrameEndpoint, request.frame, clipId,
            request.structuralRevision, request.evaluationRevision,
            staticIdentity.projectGeneration, staticIdentity.helperGeneration, use };
        MaterialFrameReceipt resolved;
        const bool resolvedFrame = composedMaterialFrameResolver
            ? composedMaterialFrameResolver->resolvePrepared(evaluation,request.frame,resolved,error)
            : resolveMaterialFrame(evaluation,*materialFrameResolver,resolved,error);
        if (!resolvedFrame)
            return VisualImportedScenePreparation::rejected;
        request.runtimeInputs.materialFrameTexture = std::move(resolved.nativeFrame);
        if (arbitgpu::materialFrameIsLinear(request.runtimeInputs.materialFrameTexture->colorTextureDescriptor()))
            request.runtimeInputs.linearColor = linearColor = true;
    }

    if (composedMaterialFrameResolver && !composedMaterialFrameResolver->active(error))
        return VisualImportedScenePreparation::rejected;
    // A derived material carries a raised evaluation revision. Apply it after
    // the material Frame resolved against the compiled revision.
    if (baked->evaluationRevision != 0)
        bakedtarget::applyEvaluationRevision(request, baked->evaluationRevision);
    typename Execution::Receipt receipt;
    const bool executed = use == NativeImportedSceneRenderUse::Preview
        ? execution->executePreview(request, receipt, error)
        : execution->executeExport(request, receipt, error);
    if (!executed) return VisualImportedScenePreparation::rejected;
    if constexpr (HasImportedPendingReceipt<typename Execution::Receipt>::value)
        if (receipt.pending) return VisualImportedScenePreparation::pending;
    if (!Execution::validReceipt(receipt))
        return reject("imported scene renderer returned an incomplete native frame receipt");

    if constexpr (HasImportedParentInspector<Execution>::value)
        if (request.parentIntent && !execution->inspectParentReceipt(receipt,request,
                use==NativeImportedSceneRenderUse::Preview?modelpayload::ImportedSceneUse::Preview:modelpayload::ImportedSceneUse::Export))
            return reject("imported parent native receipt does not match the expected compiled request");

    const auto& frame = Execution::nativeFrame(receipt);
    if (!frame)
        return reject("imported scene renderer returned no native compositor frame");
    if (frame->width() != renderExtent.width || frame->height() != renderExtent.height)
        return reject("imported scene native frame dimensions do not match the declared render extent");
    const auto backend = frame->backend();
    if (backend != "opengl" && backend != "metal")
        return reject("imported scene native frame backend is unsupported by the compositor");
    const auto nativeView = frame->colorTextureViewHandle();
    const auto colorDescriptor = frame->colorTextureDescriptor();
    const auto depthDescriptor = frame->depthTextureDescriptor();
    if (request.runtimeInputs.materialFrameTexture)
    {
        const auto inputDescriptor = request.runtimeInputs.materialFrameTexture->colorTextureDescriptor();
        if (inputDescriptor.backend != colorDescriptor.backend
            || inputDescriptor.deviceOrContextIdentity != colorDescriptor.deviceOrContextIdentity)
            return reject("Graph Frame texture and scene output belong to different native devices or contexts");
    }
    const auto exactDescriptor = [&] (const arbitgpu::NativeTextureViewDescriptor& descriptor,
                                      arbitgpu::NativeTexturePixelFormat format,
                                      std::uintptr_t image, std::uintptr_t view)
    {
        return descriptor.complete() && descriptor.backend == backend
            && descriptor.format == format && descriptor.imageHandle == image
            && descriptor.textureViewHandle == view
            && descriptor.width == renderExtent.width
            && descriptor.height == renderExtent.height;
    };
    if (nativeView == 0)
        return reject("imported scene native frame has no texture view");
    if (!exactDescriptor(colorDescriptor,
                         linearColor ? arbitgpu::NativeTexturePixelFormat::Rgba16Float
                         : backend == "metal" ? arbitgpu::NativeTexturePixelFormat::Bgra8Unorm
                                              : arbitgpu::NativeTexturePixelFormat::Rgba8Unorm,
                         frame->colorImageHandle(), nativeView)
        || (linearColor && !arbitgpu::isLinearSceneColor(colorDescriptor)))
        return reject("imported scene native color descriptor violates the compositor contract");
    if (frame->depthImageHandle() == 0 || frame->depthTextureViewHandle() == 0)
        return reject("imported scene native frame has no published depth resource");
    const auto nativeDepthView = frame->depthTextureViewHandle();
    if (!exactDescriptor(depthDescriptor, arbitgpu::NativeTexturePixelFormat::R32Float,
                         frame->depthImageHandle(), nativeDepthView))
        return reject("imported scene native depth descriptor must be exact sampleable R32F");
    if (compiled->passComposite)
    {
        const std::array<arbitgpu::NativeTexturePixelFormat, 6> formats {
            arbitgpu::NativeTexturePixelFormat::Rgba16Float, arbitgpu::NativeTexturePixelFormat::Rgba16Float,
            arbitgpu::NativeTexturePixelFormat::R8Unorm, arbitgpu::NativeTexturePixelFormat::R32Uint,
            arbitgpu::NativeTexturePixelFormat::R32Uint, arbitgpu::NativeTexturePixelFormat::Rg16Float };
        for (std::size_t i = 0; i < formats.size(); ++i)
        {
            if (i == 5 && !arbitgpu::sceneUsesMotionPass(request.runtimeInputs)) continue;
            const auto raw = frame->passTextureDescriptor(renderpasscomposite::kInputs[i + 2]);
            if (!exactDescriptor(raw, formats[i], raw.imageHandle, raw.textureViewHandle)
                || raw.deviceOrContextIdentity != colorDescriptor.deviceOrContextIdentity
                || raw.rendererGeneration != colorDescriptor.rendererGeneration)
                return reject("Render Pass Composite requires retained raw attachments from the same native scene draw");
        }
    }

    unsigned openGlTexture = 0;
    if (backend == "opengl")
    {
        if (nativeView > std::numeric_limits<unsigned>::max())
            return reject("imported scene compositor OpenGL texture view exceeds compositor width");
        if (nativeDepthView > std::numeric_limits<unsigned>::max())
            return reject("imported scene compositor OpenGL depth view exceeds compositor width");
        openGlTexture = static_cast<unsigned>(nativeView);
    }

    if (compiled->importedParticleOverlay) layer = inputLayer;
    layer.canonicalBlockCFrame = compiled->importedParticleOverlay
        ? canonicalBlockCFrame : receipt.canonicalBlockCFrame;
    if constexpr (HasRawPassExport<Layer>::value)
    {
        layer.rawExportFrame = frame;
        layer.rawExportMask = request.runtimeInputs.rawExportMask;
        layer.rawExportClipId = clipId;
        // Export Motion Data alone keys its frames by the particle node, as a
        // standalone Solid 3D draw does.
        layer.rawExportRenderId = solidMotion && compiled->importedSceneRender->rawExportMask == 0
            ? static_cast<std::uint32_t>(render3dimage::renderStableIdForNodeValue(
                static_cast<std::uint64_t>(compiled->particleNodeId)))
            : compiled->importedSceneRender->renderStableId;
        layer.rawExportImageOutput = request.runtimeInputs.imageOutput;
    }
    else if (request.runtimeInputs.rawExportMask != 0)
        return reject("Raw pass export requires a layer that retains the native frame owner");
    if constexpr (std::is_same_v<typename Execution::Receipt,
                                 modelpayload::ImportedSceneExecutionReceipt>)
    {
        if (backend == "metal" && receipt.deformationFrame
            && receipt.deformationFrame->source && receipt.deformationFrame->source->scene)
        {
            const auto& scene = *receipt.deformationFrame->source->scene;
            arbitgpu::NativeFixtureSceneStats probeStats;
            probeStats.drawCount = receipt.deformationFrame->frame.stats.drawCount;
            videohelper::importedsceneprobe::record(*frame, probeStats,
                scene.objectCount, scene.vertexCount, scene.indexCount, scene.cameraCount,
                request.runtimeInputs.materialFrameTexture);
        }
        else if (backend == "metal" && !receipt.deformationFrame && receipt.admission.scene)
        {
            const auto& scene = *receipt.admission.scene;
            videohelper::importedsceneprobe::record(*frame, receipt.frame.rendered.stats,
                scene.objectCount, scene.vertexCount, scene.indexCount, scene.cameraCount,
                request.runtimeInputs.materialFrameTexture);
        }
    }
    if (solidOverlay)
    {
        if constexpr (HasImportedSolidInput<Layer>::value)
        {
            solidReceipt->frame=frame;
            preparedSolidParameters->preparedImportedSolids=std::move(solidReceipt);
            layer.particleParameters=std::move(*preparedSolidParameters);
        }
    }
    if constexpr (std::is_same_v<Execution,modelpayload::ImportedScenePayloadExecution>
        && HasImportedParentPresentation<Layer>::value)
        if (request.parentIntent) layer.importedParentPresentation=execution->parentPresentationToken(receipt);
    if (composedMaterialFrameResolver && !composedMaterialFrameResolver->active(error))
        return VisualImportedScenePreparation::rejected;
    frameOwners.push_back(std::move(receipt));
    layer.texture = openGlTexture;
    layer.nativeTextureBackend = backend;
    layer.nativeTextureView = nativeView;
    layer.nativeTextureDescriptor = colorDescriptor;
    if constexpr (HasImportedTextureOwner<Layer>::value && HasImportedLinearImage<Layer>::value)
        if (compiled->typedScenePassExtent || linearColor)
        {
            layer.nativeTextureOwner = frame;
            layer.nativeLinearImage = frame;
        }
    layer.depthTexture = backend == "opengl" ? static_cast<unsigned>(nativeDepthView) : 0;
    layer.nativeDepthTextureBackend = backend;
    layer.nativeDepthTextureView = nativeDepthView;
    layer.nativeDepthTextureDescriptor = depthDescriptor;
    layer.depthWidth = static_cast<int>(renderExtent.width);
    layer.depthHeight = static_cast<int>(renderExtent.height);
    layer.texWidth = static_cast<int>(renderExtent.width);
    layer.texHeight = static_cast<int>(renderExtent.height);
    if (composedMaterialFrameResolver) composedMaterialFrameResolver->submitted();
    frameDependencyRollback.complete = true;
    rollback.active=false;
    error.clear();
    return VisualImportedScenePreparation::rendered;
}

template <typename Execution, typename Layer, typename FrameOwnerContainer>
VisualImportedScenePreparation prepareVisualImportedSceneLayerAtTime(
    const std::vector<videowire::CompiledVisualLayerPlan>& plans,
    int clipId,
    int width,
    int height,
    double sourceSeconds,
    double framesPerSecond,
    NativeImportedSceneRenderUse use,
    Execution* execution,
    Layer& layer,
    FrameOwnerContainer& frameOwners,
    std::string& error,
    const std::map<std::string, double>* runtimeParameters = nullptr)
{
    VisualImportedScenePlanCache oneShotCache;
    return prepareVisualImportedSceneLayerAtTime(
        plans, clipId, width, height, sourceSeconds, framesPerSecond, use,
        execution, &oneShotCache, layer, frameOwners, error, runtimeParameters);
}

template <typename Execution, typename Layer, typename FrameOwnerContainer>
VisualImportedScenePreparation prepareVisualImportedSceneLayer(
    const std::vector<videowire::CompiledVisualLayerPlan>& plans,
    int clipId,
    int width,
    int height,
    NativeImportedSceneRenderUse use,
    Execution* execution,
    VisualImportedScenePlanCache* planCache,
    Layer& layer,
    FrameOwnerContainer& frameOwners,
    std::string& error)
{
    return prepareVisualImportedSceneLayerAtTime(
        plans, clipId, width, height, 0.0, 60.0, use,
        execution, planCache, layer, frameOwners, error);
}

template <typename Execution, typename Layer, typename FrameOwnerContainer>
VisualImportedScenePreparation prepareVisualImportedSceneLayer(
    const std::vector<videowire::CompiledVisualLayerPlan>& plans,
    int clipId,
    int width,
    int height,
    NativeImportedSceneRenderUse use,
    Execution* execution,
    Layer& layer,
    FrameOwnerContainer& frameOwners,
    std::string& error)
{
    VisualImportedScenePlanCache oneShotCache;
    return prepareVisualImportedSceneLayer(
        plans, clipId, width, height, use, execution, &oneShotCache,
        layer, frameOwners, error);
}

} // namespace videohelper::importedscene
