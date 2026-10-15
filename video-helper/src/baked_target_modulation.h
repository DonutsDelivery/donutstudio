#pragma once

#include "visual_plan_executor.h"
#include "surface_material_admission.h"
#include "vertex_modifier_ir_admission.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

// Modulation of values the compiled plan bakes into its payload.
//
// PBR material numbers live in the immutable scene snapshot, and the bounded
// displacement limit lives in the vertex IR digest and its shader constants.
// Neither can be overlaid per frame, so each quantized change derives a new
// scene or material once and reuses it while the value holds. An unchanged
// value reuses the cached derivation and never replans.
namespace videohelper::bakedtarget
{
struct PbrField final
{
    const char* name;
    double maximum;
};

// PBR scene records admit only opacity one, so this table omits opacity. The limit
// applies to PBR scene records only. Surface constants offer opacity on thin-surface
// programs.
inline constexpr std::array<PbrField, 8> kPbrFields {{
    { "baseRed", 64.0 }, { "baseGreen", 64.0 }, { "baseBlue", 64.0 },
    { "metallic", 1.0 }, { "roughness", 1.0 },
    { "emissiveRed", 64.0 }, { "emissiveGreen", 64.0 }, { "emissiveBlue", 64.0 }
}};
inline constexpr const char* kMaximumDisplacement = "maximumDisplacement";
inline constexpr double kMinimumDisplacementBound = 0.000001;
inline constexpr double kMaximumDisplacementBound = 1000.0;

inline float& pbrValue(HarmonicMIDI::grid::SceneMaterialRecord& material, std::size_t field) noexcept
{
    switch (field)
    {
        case 0: return material.baseColor.x;
        case 1: return material.baseColor.y;
        case 2: return material.baseColor.z;
        case 3: return material.metallic;
        case 4: return material.roughness;
        case 5: return material.emissive.x;
        case 6: return material.emissive.y;
        default: return material.emissive.z;
    }
}

inline float pbrValue(const HarmonicMIDI::grid::SceneMaterialRecord& material, std::size_t field) noexcept
{
    return pbrValue(const_cast<HarmonicMIDI::grid::SceneMaterialRecord&>(material), field);
}

// Round to 10 mantissa bits, about 0.1 percent of the value. A slow ramp then
// derives a new scene or program only when the value moves by a visible step.
inline double quantize(double value) noexcept
{
    if (!std::isfinite(value) || value == 0.0) return value;
    int exponent = 0;
    const double mantissa = std::frexp(value, &exponent);
    return std::ldexp(std::round(mantissa * 1024.0) / 1024.0, exponent);
}

// Routes whose scene comes straight from the compiled snapshot. Timeline
// Geometry3D and Solid 3D rebuild the scene every frame from their own
// admitted owner, so their material records are not offered here.
inline bool pbrRoute(const videowire::VisualLayerExecution& compiled) noexcept
{
    if (!compiled.importedSceneRender || !compiled.importedSceneRender->sceneSnapshot) return false;
    const auto& render = *compiled.importedSceneRender;
    const bool solid = compiled.importedParticleOverlay && compiled.particleParameters.simulationSpace == 1;
    return !render.timelineGeometry && render.timelineGeometryFamily.empty() && !solid;
}

// A primitive or compose plan names each PBR Material node. A retained Geometry
// Scene keeps the PBR node identity on its material record instead. Records a
// Surface program replaces are not PBR targets.
inline bool pbrMaterial(const videowire::CompiledVisualLayerPlan& plan,
                        const visualimportedscenerender::Request& render,
                        HarmonicMIDI::grid::SceneMaterialId id) noexcept
{
    if (!id.isValid()) return false;
    for (const auto& object : render.surfacePrograms)
        if (geometrysurfacematerial::materialId(object.program.material) == id.value) return false;
    bool named = false, matched = false, retained = false;
    for (const auto& operation : plan.operations)
    {
        if (operation.kind == "visual.3d.material.pbr")
        {
            named = true;
            matched |= operation.nodeId >= 0
                && static_cast<std::uint64_t>(operation.nodeId) + 1u == id.value;
        }
        else if (operation.kind == "visual.3d.scene.retained")
            retained = true;
    }
    return named ? matched : retained;
}

inline const std::optional<surfacematerialbinding::ImportedSceneMaterialRequest>& sourceMaterial(
    const visualimportedscenerender::Request& render) noexcept
{
    return render.importedScene ? render.importedScene->material : render.material;
}

// The root record of a graph vertex program that ends in Bounded Displacement.
inline const videowire::VertexModifierRecord* displacementRoot(
    const videowire::CompiledVisualLayerPlan& plan,
    const visualimportedscenerender::Request& render) noexcept
{
    const auto& material = sourceMaterial(render);
    if (!material || !material->vertexModifier) return nullptr;
    const auto& vertex = *material->vertexModifier;
    const auto root = std::find_if(vertex.records.begin(), vertex.records.end(),
        [&](const auto& record) { return record.stableId == vertex.rootId; });
    if (root == vertex.records.end()
        || root->operation != videowire::VertexModifierOperation::boundedDisplacementOutput
        || root->parameterCount < 1)
        return nullptr;
    const auto* kind = videowire::vertexModifierGraphKindToken(root->operation);
    const bool planned = std::any_of(plan.operations.begin(), plan.operations.end(), [&](const auto& operation) {
        return operation.kind == kind && operation.nodeId >= 0
            && static_cast<std::uint64_t>(operation.nodeId) + 1u == root->stableId;
    });
    return planned ? &*root : nullptr;
}

inline std::string prefixFor(std::uint64_t id)
{
    return "visual" + std::to_string(id) + "/";
}

// Publish the authored values so the modulation sinks know these keys.
inline void seed(const videowire::CompiledVisualLayerPlan& plan,
                 const videowire::VisualLayerExecution& compiled,
                 std::map<std::string, double>& parameters)
{
    if (!compiled.importedSceneRender) return;
    const auto& render = *compiled.importedSceneRender;
    if (pbrRoute(compiled))
    {
        const auto& scene = *render.sceneSnapshot;
        const auto count = std::min<std::size_t>(scene.materialCount, scene.materials.size());
        for (std::size_t index = 0; index < count; ++index)
        {
            const auto& material = scene.materials[index];
            if (!pbrMaterial(plan, render, material.id)) continue;
            const auto prefix = prefixFor(material.id.value);
            for (std::size_t field = 0; field < kPbrFields.size(); ++field)
                parameters.emplace(prefix + kPbrFields[field].name, pbrValue(material, field));
        }
    }
    if (pbrRoute(compiled))
    {
        // A per-object program's constants publish under its source Surface
        // Material, the same key scheme the plugin offers. Metallic, roughness and
        // the RGB lanes publish the literal the program was lowered with.
        for (const auto& object : render.surfacePrograms)
        {
            const auto source = object.program.sourceMaterial;
            if (source == 0) continue;
            const auto& program = object.program.material.program;
            for (const auto& field : geometrysurfacematerial::kConstantFields)
            {
                if (!geometrysurfacematerial::constantOffered(program, field)) continue;
                const auto operation = geometrysurfacematerial::constantOperation(program, field);
                if (operation)
                    parameters.emplace(prefixFor(source) + field.name,
                        static_cast<double>(program.operations[*operation].literal[field.lane]));
            }
        }
    }
    if (const auto* root = displacementRoot(plan, render))
        parameters.emplace(prefixFor(root->stableId) + kMaximumDisplacement, root->parameters[0]);
}

// The last evaluation revision a clip published for a derived material. The
// native material owner admits a changed program under one plan revision only
// when its evaluation revision grows, so this outlives a plan cache entry.
struct RevisionFloor final
{
    std::uint64_t structuralRevision = 0;
    std::uint64_t evaluationRevision = 0;
};

// Derivations for one compiled plan. A replaced plan starts a fresh slot.
struct Slot final
{
    using SceneKey = std::vector<std::tuple<std::size_t, std::size_t, float>>;
    SceneKey sceneKey;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
    bool materialEvaluated = false;
    std::optional<double> materialKey;
    std::optional<surfacematerialbinding::ImportedSceneMaterialRequest> material;
    std::shared_ptr<const visualimportedscenerender::Request> leaf;
    // Per-object program constants as (program index, field index, literal). An
    // empty key keeps the compiled programs.
    using SurfaceKey = std::vector<std::tuple<std::size_t, std::size_t, float>>;
    SurfaceKey surfaceKey;
    std::optional<geometrysurfacematerial::ObjectPrograms> surfacePrograms;
    // Zero keeps the compiled evaluation revision.
    std::uint64_t evaluationRevision = 0;
    std::uint64_t derivations = 0;
};

// The per-object program constants that the modulation map moves, in program and
// field order. Each constant is keyed by its source Surface Material's stable id,
// so one destination moves every program that node lowers to. Only a finite value
// is admitted. A non-finite value refuses the whole set, and the caller keeps the
// last good programs.
inline bool surfaceKeyFor(const visualimportedscenerender::Request& render,
                          const std::map<std::string, double>& parameters,
                          Slot::SurfaceKey& key)
{
    key.clear();
    for (std::size_t index = 0; index < render.surfacePrograms.size(); ++index)
    {
        const auto& object = render.surfacePrograms[index];
        const auto source = object.program.sourceMaterial;
        if (source == 0) continue;
        const auto& program = object.program.material.program;
        for (std::size_t field = 0; field < geometrysurfacematerial::kConstantFields.size(); ++field)
        {
            const auto& constant = geometrysurfacematerial::kConstantFields[field];
            const auto found = parameters.find(prefixFor(source) + constant.name);
            if (found == parameters.end()) continue;
            if (!geometrysurfacematerial::constantOffered(program, constant)) continue;
            const auto operation = geometrysurfacematerial::constantOperation(program, constant);
            if (!operation) continue;
            if (!std::isfinite(found->second)) return false;
            const auto baseline = program.operations[*operation].literal[constant.lane];
            if (static_cast<float>(found->second) == baseline) continue;
            const auto value = static_cast<float>(
                std::clamp(quantize(found->second), constant.minimum, constant.maximum));
            if (value != baseline) key.emplace_back(index, field, value);
        }
    }
    return true;
}

// The compiled programs with the key's literals written. Each program is admitted
// again, so its structural digest follows the new constants. The derived programs
// keep the compiled identity, revisions and scene. The caller lifts the revision,
// and the executor binds the frame's scene to the programs.
inline bool deriveSurfacePrograms(const visualimportedscenerender::Request& render,
                                  const Slot::SurfaceKey& key,
                                  geometrysurfacematerial::ObjectPrograms& programs,
                                  std::string& error)
{
    programs = render.surfacePrograms;
    for (const auto& [index, field, value] : key)
    {
        auto& program = programs[index].program.material.program;
        const auto& constant = geometrysurfacematerial::kConstantFields[field];
        const auto operation = geometrysurfacematerial::constantOperation(program, constant);
        if (!operation)
        {
            error = "Surface constant modulation lost its constant";
            return false;
        }
        program.operations[*operation].literal[constant.lane] = value;
    }
    for (auto& object : programs)
    {
        auto& material = object.program.material;
        const auto admitted = surfacematerial::admit(material.program, error);
        if (!admitted)
        {
            error = "Surface constant modulation was refused: " + error;
            return false;
        }
        material.binding.surfaceMaterialDigest = admitted->structuralDigest();
    }
    return true;
}

// Resolve this frame's overrides into the slot. Reuses the slot's scene and
// material while the quantized values hold.
inline bool evaluate(const videowire::CompiledVisualLayerPlan& plan,
                     const videowire::VisualLayerExecution& compiled,
                     const std::map<std::string, double>* parameters,
                     Slot& slot, RevisionFloor& floor, std::string& error)
{
    if (!compiled.importedSceneRender) return true;
    const auto& render = *compiled.importedSceneRender;

    Slot::SceneKey sceneKey;
    if (parameters && pbrRoute(compiled))
    {
        const auto& scene = *render.sceneSnapshot;
        const auto count = std::min<std::size_t>(scene.materialCount, scene.materials.size());
        for (std::size_t index = 0; index < count; ++index)
        {
            const auto& material = scene.materials[index];
            if (!pbrMaterial(plan, render, material.id)) continue;
            const auto prefix = prefixFor(material.id.value);
            for (std::size_t field = 0; field < kPbrFields.size(); ++field)
            {
                const auto found = parameters->find(prefix + kPbrFields[field].name);
                if (found == parameters->end()) continue;
                if (!std::isfinite(found->second))
                {
                    error = "PBR material modulation must be finite";
                    return false;
                }
                const auto baseline = pbrValue(material, field);
                if (static_cast<float>(found->second) == baseline) continue;
                const auto value = static_cast<float>(std::clamp(quantize(found->second), 0.0, kPbrFields[field].maximum));
                if (value != baseline) sceneKey.emplace_back(index, field, value);
            }
        }
    }
    if (sceneKey.empty())
    {
        slot.sceneKey.clear();
        slot.scene.reset();
    }
    else if (sceneKey != slot.sceneKey || !slot.scene)
    {
        auto scene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*render.sceneSnapshot);
        for (const auto& [index, field, value] : sceneKey)
            pbrValue(scene->materials[index], field) = value;
        if (!HarmonicMIDI::grid::validateVisual3DScene(*scene).valid())
        {
            error = "PBR material modulation produced an invalid scene";
            return false;
        }
        slot.scene = std::move(scene);
        slot.sceneKey = std::move(sceneKey);
        ++slot.derivations;
    }

    // Per-object program constants. A non-finite value keeps the last good programs.
    // An empty key returns the slot to the compiled programs, as the scene does.
    Slot::SurfaceKey surfaceKey;
    if (parameters && pbrRoute(compiled) && !render.surfacePrograms.empty()
        && !surfaceKeyFor(render, *parameters, surfaceKey))
        surfaceKey = slot.surfaceKey;
    const bool surfaceChanged = surfaceKey != slot.surfaceKey;
    if (surfaceChanged)
    {
        if (surfaceKey.empty())
            slot.surfacePrograms.reset();
        else
        {
            geometrysurfacematerial::ObjectPrograms programs;
            if (!deriveSurfacePrograms(render, surfaceKey, programs, error)) return false;
            slot.surfacePrograms = std::move(programs);
        }
        slot.surfaceKey = std::move(surfaceKey);
    }

    std::optional<double> displacement;
    const auto* root = displacementRoot(plan, render);
    if (parameters && root)
    {
        const auto found = parameters->find(prefixFor(root->stableId) + kMaximumDisplacement);
        if (found != parameters->end())
        {
            if (!std::isfinite(found->second))
            {
                error = "Bounded displacement modulation must be finite";
                return false;
            }
            if (found->second != root->parameters[0])
            {
                const auto value = std::clamp(quantize(found->second),
                    kMinimumDisplacementBound, kMaximumDisplacementBound);
                if (value != root->parameters[0]) displacement = value;
            }
        }
    }
    const auto& source = sourceMaterial(render);
    // A clip that already published a derived material or program under this plan
    // revision keeps lifting, or the native owner would see its revision go backwards.
    const bool carriedLift = (source || !render.surfacePrograms.empty())
        && floor.structuralRevision == plan.structuralRevision
        && floor.evaluationRevision >= render.evaluationRevision;
    const bool changed = slot.materialEvaluated
        ? displacement != slot.materialKey || surfaceChanged
        : displacement.has_value() || carriedLift || surfaceChanged;
    slot.materialEvaluated = true;
    if (!changed) return true;

    const auto previous = floor.structuralRevision == plan.structuralRevision ? floor.evaluationRevision : 0;
    const auto lifted = std::max(render.evaluationRevision, previous) + 1u;
    if (source)
    {
        auto material = *source;
        if (displacement)
        {
            auto& vertex = *material.vertexModifier;
            for (auto& record : vertex.records)
                if (record.stableId == vertex.rootId) record.parameters[0] = *displacement;
            if (!vertexmodifier::admitVertexModifierIr(vertex, {}, error))
            {
                error = "Bounded displacement modulation was refused: " + error;
                return false;
            }
        }
        material.evaluationRevision = lifted;
        slot.material = std::move(material);
        slot.materialKey = displacement;
    }
    if (render.importedScene)
    {
        // The leaf follows the lifted revision, as the scene material does.
        auto leaf = *render.importedScene;
        if (slot.material) leaf.material = *slot.material;
        leaf.evaluationRevision = lifted;
        slot.leaf = std::make_shared<const visualimportedscenerender::Request>(std::move(leaf));
    }
    if (slot.surfacePrograms)
        for (auto& object : *slot.surfacePrograms) object.program.material.evaluationRevision = lifted;
    slot.evaluationRevision = lifted;
    floor = { plan.structuralRevision, lifted };
    ++slot.derivations;
    return true;
}

// Every material identity in one request shares the request's evaluation revision.
template <typename Request>
inline void applyEvaluationRevision(Request& request, std::uint64_t revision)
{
    request.evaluationRevision = revision;
    if (request.material) request.material->evaluationRevision = revision;
    if (request.diffractionMaterial) request.diffractionMaterial->evaluationRevision = revision;
    for (auto& object : request.surfacePrograms)
        object.program.material.evaluationRevision = revision;
}
} // namespace videohelper::bakedtarget
