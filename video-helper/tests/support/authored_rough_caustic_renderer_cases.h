#pragma once

#include "glass_caustic_oracle.h"

namespace videohelper::tests::authoredroughcaustic
{
inline surfacematerialbinding::ImportedSceneMaterialRequest request(
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    float roughness, std::uint64_t revision = 1)
{
    auto program = smoothglassfixture::authoredProgram(scene->objects[0].material.value);
    const auto roughnessId = program.outputs[static_cast<std::size_t>(surfacematerial::OutputSemantic::Roughness)];
    for (auto& operation : program.operations)
        if (operation.id == roughnessId) operation.literal[0] = roughness;
    std::string error;
    const auto admitted = surfacematerial::admit(program, error);
    surfacematerialbinding::ImportedSceneMaterialRequest result;
    result.sceneSnapshot = scene;
    result.scene = scene->id;
    result.sceneRevision = 1;
    result.structuralRevision = result.evaluationRevision = result.programRevision = revision;
    result.program = std::move(program);
    result.binding.targetKind = surfacematerialbinding::BindingTargetKind::ObjectOverride;
    result.binding.object = scene->objects[0].id;
    result.binding.surfaceMaterialRevision = revision;
    result.binding.surfaceMaterialDigest = admitted ? admitted->structuralDigest() : std::string{};
    return result;
}

inline std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene()
{
    // Reuse authored triangles and Li8 only. The fixture's synthetic native
    // programs are discarded; every draw below uses production material admission.
    auto value = glasscaustictests::roughFixture(.35f).scene;
    value->materials[1].baseColor = {.5f, .5f, .5f};
    return value;
}

inline arbitgpu::NativeFixtureSceneRuntimeInputs inputs(const HarmonicMIDI::grid::Visual3DScene& scene)
{
    arbitgpu::NativeFixtureSceneRuntimeInputs result;
    result.linearColor = true;
    visualimportedscenerender::GlassCausticSelection selection;
    selection.version = 1;
    selection.sourceLight = scene.lights[0].id;
    selection.glassObject = scene.objects[0].id;
    selection.glassMaterial = scene.objects[0].material;
    selection.policy = {1, 1.0e-4f, 1.0e-5f, 32, 64, 16};
    result.glassCaustic = selection;
    return result;
}

// Reader adapts the backend's existing receipt API to an owned NativeReadback.
// Native fixture owners invoke this with their real backend and context.
template<class Reader>
bool nativeCases(arbitgpu::NativeFixtureSceneBackend& backend,
    videohelper::materialprogram::BackendTarget target, Reader read, std::string& error)
{
    namespace renderer = videorender::fixture3d;
    namespace gc = arbitgpu::glasscaustics;
    const auto fail = [&](const char* message) { error = message; return false; };
    const auto snapshot = scene();
    const auto authored = request(snapshot, .35f);
    const auto material = renderer::admitSurfaceMaterialBinding(snapshot, authored, target, error);
    if (!material) return false;
    if (material->sceneSnapshot() != snapshot || material->programRevision() != 1
        || material->nativeProgram()->parameters.emissionRoughness[3] != .35f)
        return fail("rough caustic renderer lost its exact authored material owner");
    auto runtime = inputs(*snapshot);
    renderer::FixtureSceneRenderer preview(backend), exporter(backend);
    renderer::RenderedFrame a, b;
    if (!preview.renderPreview(snapshot, material, {}, runtime, {32,32}, renderer::kNativeGpuCapability, a, error)
        || !exporter.renderExport(snapshot, material, {}, runtime, {32,32}, renderer::kNativeGpuCapability, b, error))
        return false;
    gc::NativeReadback pa, pb;
    gc::EnergyLedger ea, eb;
    if (!read(a.nativeFrame, pa, ea, error) || !read(b.nativeFrame, pb, eb, error)) return false;
    const auto positive = [](const gc::NativeReadback& receipt) {
        return !receipt.paths.empty() && !receipt.deposits.empty()
            && std::all_of(receipt.paths.begin(), receipt.paths.end(), [](const auto& path) {
                return path.complete && path.multipleScatterVersion == gc::kPrivateMultipleScatterVersion
                    && path.roughTransportVersion == gc::kRoughTransportVersion
                    && path.microDraws == 4 * path.microEvents && path.maskedPower == 0;
            }) && std::any_of(receipt.paths.begin(), receipt.paths.end(),
                [](const auto& path) { return path.microEvents > 0; })
            && std::any_of(receipt.spectralRadiance.begin(), receipt.spectralRadiance.end(),
                [](float value) { return value > 0; });
    };
    if (!positive(pa) || !positive(pb) || pa.spectralRadiance != pb.spectralRadiance
        || ea.launched != eb.launched || ea.deposited != eb.deposited
        || a.use != renderer::RenderUse::Preview || b.use != renderer::RenderUse::Export)
        return fail("authored rough preview/export did not retain one completed physical native route");
    renderer::RenderedFrame seek;
    if (!preview.renderPreview(snapshot, material, {}, runtime, {32,32}, renderer::kNativeGpuCapability, seek, error))
        return false;
    gc::NativeReadback repeated; gc::EnergyLedger repeatedEnergy;
    if (!read(seek.nativeFrame, repeated, repeatedEnergy, error)) return false;
    if (repeated.spectralRadiance != pa.spectralRadiance || !seek.stats.reusedMaterialProgram)
        return fail("repeated authored rough draw changed its native output or material owner");

    const auto edited = renderer::admitSurfaceMaterialBinding(snapshot, request(snapshot, .75f, 2), target, error);
    if (!edited || edited->bindingDigest() == material->bindingDigest()
        || edited->nativeProgram()->programIdentity == material->nativeProgram()->programIdentity)
        return fail("roughness edit failed to revise the actual authored material binding");
    renderer::RenderedFrame changed;
    if (!preview.renderPreview(snapshot, edited, {}, runtime, {32,32}, renderer::kNativeGpuCapability, changed, error)
        || !exporter.renderExport(snapshot, edited, {}, runtime, {32,32}, renderer::kNativeGpuCapability, b, error))
        return false;
    gc::NativeReadback pc, changedExport; gc::EnergyLedger ec, changedExportEnergy;
    if (!read(changed.nativeFrame, pc, ec, error) || !read(b.nativeFrame, changedExport, changedExportEnergy, error))
        return false;
    if (!positive(pc) || !positive(changedExport) || pc.spectralRadiance == pa.spectralRadiance
        || pc.spectralRadiance != changedExport.spectralRadiance || changed.stats.reusedMaterialProgram)
        return fail("authored roughness edit did not change completed native preview/export caustic radiance");

    auto dimmedValue = std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*snapshot);
    for (auto& band : dimmedValue->lights[0].spectralSource.radiance) band *= .5f;
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> dimmed = dimmedValue;
    const auto dimmedMaterial = renderer::admitSurfaceMaterialBinding(dimmed, request(dimmed, .35f), target, error);
    if (!dimmedMaterial || !exporter.renderExport(dimmed, dimmedMaterial, {}, inputs(*dimmed),
            {32,32}, renderer::kNativeGpuCapability, changed, error)) return false;
    gc::NativeReadback pd; gc::EnergyLedger ed;
    if (!read(changed.nativeFrame, pd, ed, error)) return false;
    for (unsigned band = 0; band < 8; ++band)
        if (std::abs(ed.launched[band] - .5 * ea.launched[band]) > 1e-5 * ea.launched[band]
            || std::abs(ed.deposited[band] - .5 * ea.deposited[band]) > 1e-5 * ea.launched[band])
            return fail("authored Li8 edit did not scale native rough caustic power");

    // The edited scene has a different owner and must clear earlier Color.
    changed = a;
    if (preview.renderPreview(dimmed, material, {}, runtime, {32,32}, renderer::kNativeGpuCapability, changed, error)
        || changed.nativeFrame)
        return fail("stale rough material owner published an earlier whole frame");
    runtime.glassCaustic->policy.maximumVisits = 65;
    changed = a;
    if (exporter.renderExport(snapshot, material, {}, runtime, {32,32}, renderer::kNativeGpuCapability, changed, error)
        || changed.nativeFrame)
        return fail("unsupported rough branch policy published partial or previous Color");

    // A positive float whose squared GGX alpha underflows cannot complete the
    // physical interface walk. Refusal is separate from the positive cases.
    const auto underflow = renderer::admitSurfaceMaterialBinding(snapshot, request(snapshot, 1e-30f, 4), target, error);
    if (!underflow) return false;
    changed = a;
    if (preview.renderPreview(snapshot, underflow, {}, inputs(*snapshot),
            {32,32}, renderer::kNativeGpuCapability, changed, error)
        || changed.nativeFrame || error.empty())
        return fail("numerically unsupported authored roughness published prior or partial Color");

    const auto smooth = renderer::admitSurfaceMaterialBinding(snapshot, request(snapshot, 0, 3), target, error);
    if (!smooth || !preview.renderPreview(snapshot, smooth, {}, inputs(*snapshot),
            {32,32}, renderer::kNativeGpuCapability, changed, error)) return false;
    gc::NativeReadback legacy; gc::EnergyLedger legacyEnergy;
    if (!read(changed.nativeFrame, legacy, legacyEnergy, error)) return false;
    if (legacy.paths.empty() || !std::all_of(legacy.paths.begin(), legacy.paths.end(), [](const auto& path) {
            return path.complete && path.multipleScatterVersion == 0 && path.roughTransportVersion == 0
                && path.microEvents == 0 && path.microDraws == 0 && path.microResidualPower == 0;
        })) return fail("zero authored roughness changed the legacy native caustic path");
    error.clear();
    return true;
}
} // namespace videohelper::tests::authoredroughcaustic
