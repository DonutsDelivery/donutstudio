// The CPU path tracer as the 3D Render node's Path Trace engine (Plan 3,
// PT-C1). FixtureSceneRenderer hands it a draw that passed the raster's
// admission; this file turns that draw into a SceneEngine request with the
// raster's runtime transforms, camera, lights and surface materials, then
// uploads the result as a native frame the compositor takes like a raster one.
//
// Only arbit-video-helper and arbit-pathtrace-route-tests link this file. It
// registers the engine factory at static initialisation
// (setPathTraceSceneEngineFactory).
//
// Material mapping from the admitted Surface PBR block (raster references:
// fixture_scene_renderer.cpp, the native program fill; surface_material_admission.h,
// the block layout and clamps; gpu_backend/backend_opengl.cpp, the per-draw
// uniforms and the thin coverage term; gpu_backend/fixture_smooth_glass.h, the
// closed glass owner):
//   baseColor    = baseColorMetallic.xyz, mixed toward timeMixEndColor by
//                  clamp(time, 0, 1) for a time mix
//   metallic     = baseColorMetallic.w
//   roughness    = emissionRoughness.w
//   emission     = emissionRoughness.xyz (times the runtime emission gain)
//   opacity      = normalOpacity.w
//   ior          = transmissionIorClearcoat.y, the index at the d line
//                  (587.56 nm, SurfaceMaterialIR.h)
//   transmission = transmissionIorClearcoat.x for a closed volume or thin
//                  interface
//   dispersion   = transmissionIorClearcoat.w, the Cauchy B in um^2. It
//                  acts on refracting materials only, and a dispersive glass
//                  switches the tracer to spectral transport
//                  (pathtrace/spectrum.h).
//   transport    : closed volume -> DielectricTransport::ClosedVolume; thin
//                  interface (a water surface, the raster's
//                  thinInterfaceTransport) -> Interface, one refracting
//                  surface whose back side is the medium.
//   thin surface : no refraction and no dielectric lobe. Like the raster,
//                  transmission lowers the coverage to 1 - t (1 - F0),
//                  F0 = ((ior - 1) / (ior + 1))^2, so the background shows
//                  through untinted by t (1 - F0), and the surface blends
//                  when the raster's rule says so. The tracer's ThinSheet
//                  transport would tint the background by the base colour and
//                  use the angle-dependent slab reflectance 2R / (1 + R)
//                  instead of F0, which the raster does not draw.
//   absorption   : the IR has no absorption distance. For closed volumes and
//                  interfaces the base colour is the light left after 1 scene
//                  unit inside the medium (Beer-Lambert), not a tint at each
//                  crossing. The raster refuses tinted glass, so white glass,
//                  the only kind both engines draw, is unchanged by this.
//   priority     : 0 for every medium; nested media resolve by entry order.
//   frame texture: the tracer cannot read a graph Frame, which stays on the GPU.
//                  Camera samples that first hit a frame-tinted object go to a
//                  frame layer, and the upload's GPU pass multiplies them by the
//                  sampled frame (gpu_backend/cpu_scene_frame.h). The factor in
//                  baseColorMetallic still applies. The tint covers the whole
//                  sample, so it also tints a dielectric's specular and an
//                  emission, and light leaving the surface for other objects
//                  stays untinted.
// Clearcoat is not traced yet. The dielectrics this maps onto
// (pathtrace/bsdf.h) follow Blender Cycles
// intern/cycles/kernel/closure/bsdf_microfacet.h and bsdf_util.h at
// 3e451b5580bb54df6029ecfa5d0f3958e32d1000 as a behavioural reference only;
// no code is taken from it.

#include "pathtrace_fixture_engine.h"
#include "gpu_backend/cpu_scene_frame.h"
#include "pathtrace/devices/auto_device_latch.h"
#include "pathtrace/devices/device_registry.h"
#include "pathtrace/devices/device_compute_host.h"
#if defined(ARBIT_HAVE_OPTIX)
#include "pathtrace/devices/device_optix_host.h"
#endif
#if defined(ARBIT_HAVE_METAL_PATHTRACE)
#include "pathtrace/devices/device_metal_host.h"
#endif
#include "pathtrace/engine.h"
#include "pathtrace/light_paths.h"
#include "pathtrace/environment_resident.h"
#include "pathtrace/thread_pool.h"
#include "pathtrace_denoiser.h"
#include "loop_step_probe.h"
#include "pathtrace_oidn.h"
#include "pathtrace_preview_denoise.h"
#include "render_input_generation.h"
#include "render_mode_policy.h"
#include "vertex_program_motion.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace videorender::fixture3d
{
namespace
{
namespace pt = videohelper::pathtrace;
static_assert (kUnlimitedVolumeBounces == pt::kUnlimitedBounces);
using HarmonicMIDI::grid::SceneCameraRecord;
using HarmonicMIDI::grid::SceneMaterialRecord;
using HarmonicMIDI::grid::SceneTransform3D;
using Program = arbitgpu::NativeFixtureSurfaceMaterialProgram;

// Scene units of medium after which a refracting material's base colour is
// the light left (Beer-Lambert), since the material IR has no distance.
constexpr float kTintDistance = 1.0f;

// The machine's device setting (Preferences > System, passed to the helper at launch
// and on each change), and the selection it makes. ARBIT_PATHTRACE_DEVICE overrides
// the setting for tests. Each change makes a new generation, and engines drop their
// device pass and fallback when they see one. The log names every device passed over.
struct DeviceChoice
{
    pt::device::DeviceSelection selection;
    std::uint64_t generation = 0;
};

struct DeviceSetting
{
    std::mutex lock;
    std::string preference = "auto";
    std::shared_ptr<const DeviceChoice> choice;
    std::uint64_t generations = 0;
};

DeviceSetting& deviceSetting()
{
    static auto* setting = new DeviceSetting();
    return *setting;
}

std::shared_ptr<const DeviceChoice> chooseDevice (const std::string& preference, std::uint64_t generation)
{
    auto choice = std::make_shared<DeviceChoice>();
    choice->generation = generation;
    choice->selection = pt::device::selectDevice (
        pt::device::effectiveDeviceSetting (std::getenv ("ARBIT_PATHTRACE_DEVICE"), preference.c_str()),
        pt::device::productDeviceEnvironment());
    const auto& chosen = choice->selection;
    if (chosen.ok)
        std::fprintf (stderr, "[pt] device=%s requested=%s\n", pt::device::deviceName (chosen.device), chosen.requested);
    else
        std::fprintf (stderr, "[pt] %s\n", chosen.error.c_str());
    for (const auto& status : chosen.devices)
        if (! status.loadable)
            std::fprintf (stderr, "[pt] %s unavailable: %s\n", pt::device::deviceName (status.kind), status.problem.c_str());
    return choice;
}

std::shared_ptr<const DeviceChoice> pathTraceDeviceSelection()
{
    auto& setting = deviceSetting();
    const std::lock_guard<std::mutex> lock (setting.lock);
    if (setting.choice == nullptr)
        setting.choice = chooseDevice (setting.preference, ++setting.generations);
    return setting.choice;
}

bool fail (std::string& error, std::string message)
{
    error = std::move (message);
    return false;
}
} // namespace

namespace pathtracefixture
{
// Refuses inputs whose raster passes the tracer does not have, instead of
// rendering a different picture. Every Image view, data export and pass
// composite reads the engine's render outputs (pathtrace/render_outputs.h).
bool admitInputs (const arbitgpu::NativeFixtureSceneRuntimeInputs& in, std::string& error)
{
    if (! arbitgpu::validFixtureRuntimeInputs (in))
        return fail (error, "Path Trace runtime inputs are invalid");
    return true;
}

bool readsRenderOutputs (const arbitgpu::NativeFixtureSceneRuntimeInputs& in) noexcept
{
    return in.imageOutput != renderpassoutput::Output::Color || in.rawExportMask != 0 || in.passComposite
        || in.passProgram;
}

bool bindNoteInstances (const HarmonicMIDI::grid::Visual3DScene& scene,
                        const arbitgpu::NativeFixtureSceneRuntimeInputs& in, bool motionBlur,
                        pt::EngineSceneSetup& setup, std::string& error)
{
    setup.noteInstanced = false;
    setup.noteInstances.clear();
    setup.previousNoteInstances.clear();
    setup.noteAppearance.clear();
    if (! in.noteInstanceMapping)
        return true;
    // The raster's own refusal for an instanced object under the mapping.
    if (! arbitgpu::nativeObjectInstanceRuntimeSupported (scene, in, error))
        return false;
    const auto batch = arbitgpu::prepareNativeNoteInstances (in);
    if (! batch.admitted)
        return fail (error, "Path Trace note instancing has no valid score frame or mapping");
    // As the raster: objectMatrix * (position * meshScale + note position).
    const auto placement = [] (const std::array<float, 4>& note) {
        SceneTransform3D transform;
        transform.translation = { note[0], note[1], note[2] };
        transform.scale = { note[3], note[3], note[3] };
        return transform;
    };
    setup.noteInstanced = true;
    setup.noteInstances.reserve (batch.count);
    for (std::size_t index = 0; index < batch.count; ++index)
        setup.noteInstances.push_back (placement (batch.transforms[index]));
    // The raster's vNoteAppearance: mix (low, high, value), added to the shaded colour.
    const std::array<float, 4> plain { 1.0f, 1.0f, 1.0f, 0.0f };
    if (batch.appearanceLow != plain || batch.appearanceHigh != plain)
    {
        setup.noteAppearance.reserve (batch.count);
        for (std::size_t index = 0; index < batch.count; ++index)
        {
            const float t = batch.appearanceValues[index];
            std::array<float, 4> mixed;
            for (std::size_t channel = 0; channel < 4; ++channel)
                mixed[channel] = batch.appearanceLow[channel] + (batch.appearanceHigh[channel] - batch.appearanceLow[channel]) * t;
            setup.noteAppearance.push_back (mixed);
        }
    }
    // The shutter opens on each note's prior placement; a note that just started holds still.
    const auto& prior = in.previousMotion ? in.previousMotion->noteTransforms : nullptr;
    if (motionBlur && prior != nullptr)
    {
        std::unordered_map<std::int64_t, std::size_t> priorRow;
        priorRow.reserve (prior->count);
        for (std::size_t row = 0; row < prior->count; ++row)
            priorRow.emplace (prior->identities[row], row);
        setup.previousNoteInstances.reserve (batch.count);
        for (std::size_t index = 0; index < batch.count; ++index)
        {
            const auto found = priorRow.find (batch.identities[index]);
            setup.previousNoteInstances.push_back (found == priorRow.end() ? setup.noteInstances[index]
                                                                             : placement (prior->transforms[found->second]));
        }
    }
    return true;
}

// The raster's material for one object drawn with a Surface program.
bool mapSurfaceProgram (const Program& program, const SceneMaterialRecord* record, float timeSeconds,
                        pt::Material& material, std::string& error)
{
    // A diffraction program scatters by its grating (buildSetup). Where its mask
    // leaves the foil out, the raster shows litBase: the object's scene material.
    if (program.kind == arbitgpu::NativeFixtureMaterialKind::DiffractionReflective)
    {
        material = record != nullptr ? pt::materialFromRecord (*record) : pt::Material {};
        return true;
    }
    if (program.kind != arbitgpu::NativeFixtureMaterialKind::SurfacePbr)
        return fail (error, "Path Trace surface material kind is invalid");
    if (! arbitgpu::validNativeFixtureSurfaceParameters (program.parameters))
        return fail (error, "Path Trace surface material parameters are invalid");
    const auto& pbr = program.parameters;
    material = record != nullptr ? pt::materialFromRecord (*record) : pt::Material {};
    pt::Vec3 base { pbr.baseColorMetallic[0], pbr.baseColorMetallic[1], pbr.baseColorMetallic[2] };
    if (program.baseColorSource == Program::BaseColorSource::TimeLinearMix)
    {
        const float t = std::isfinite (timeSeconds) ? std::min (1.0f, std::max (0.0f, timeSeconds)) : 0.0f;
        base = { base.x + (program.timeMixEndColor[0] - base.x) * t,
                 base.y + (program.timeMixEndColor[1] - base.y) * t,
                 base.z + (program.timeMixEndColor[2] - base.z) * t };
    }
    // An imported base colour texture is not sampled yet; its constant factor is.
    material.baseColor = base;
    material.metallic = pbr.baseColorMetallic[3];
    material.roughness = pbr.emissionRoughness[3];
    material.emission = { pbr.emissionRoughness[0], pbr.emissionRoughness[1], pbr.emissionRoughness[2] };
    material.opacity = pbr.normalOpacity[3];
    const float transmission = pbr.transmissionIorClearcoat[0];
    material.ior = pbr.transmissionIorClearcoat[1];
    material.dispersion = pbr.transmissionIorClearcoat[3];
    material.mediumPriority = 0;
    const auto transport = pbr.identifiers[1];
    if (transport == static_cast<std::uint32_t> (surfacematerial::TransportMode::ThinSurface))
    {
        // The raster's coverage term (backend_opengl.cpp): the background
        // shows through untinted by t (1 - F0) and the surface shades on top.
        const float ratio = (material.ior - 1.0f) / (material.ior + 1.0f);
        const float f0 = ratio * ratio;
        material.transmission = 0.0f;
        material.opacity *= 1.0f - transmission * (1.0f - f0);
        material.absorptionDistance = 0.0f;
        if (arbitgpu::nativeSurfaceRequiresBlend (&program) && material.alphaMode != pt::MaterialAlphaMode::Mask)
            material.alphaMode = pt::MaterialAlphaMode::Blend;
    }
    else
    {
        // Closed volume and thin interface keep full coverage, as the raster does.
        material.transmission = transmission;
        material.transport = transport == static_cast<std::uint32_t> (surfacematerial::TransportMode::ThinInterface)
                               ? pt::DielectricTransport::Interface
                               : pt::DielectricTransport::ClosedVolume;
        material.absorptionDistance = kTintDistance;
    }
    return true;
}
} // namespace pathtracefixture

namespace
{
bool hasVertexProgram (const Program& root) noexcept
{
    if (root.vertexProgram) return true;
    return std::any_of (root.objectPrograms.begin(), root.objectPrograms.end(),
                        [] (const auto& program) { return program && program->vertexProgram; });
}

// The pre-BVH displacement hook: each object's vertex program moves a copy of the
// scene on the host, so the CPU and OptiX devices trace the same triangles.
bool displacedScene (const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& source, const Program& root,
                     const arbitgpu::NativeFixtureSceneRuntimeInputs& in,
                     std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& out, std::string& error)
{
    auto displaced = std::make_shared<HarmonicMIDI::grid::Visual3DScene> (*source);
    if (! videohelper::vertexprogramcpu::displaceScene (*displaced, root, videohelper::vertexprogramcpu::currentInputs (in),
                                                        error))
        return false;
    out = std::move (displaced);
    return true;
}

// One layer per distinct frame. A frame the draw carries that no object samples,
// or a frame-tinted program without a frame, would lose the imagery, so both stop the draw.
bool bindFrameLayers (const SceneEngineDraw& draw, pt::EngineSceneSetup& setup,
                      std::vector<std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>>& frames, std::string& error)
{
    const auto& scene = *draw.snapshot;
    const auto& in = *draw.inputs;
    setup.objectFrames.clear();
    frames.clear();
    for (std::size_t index = 0; index < scene.objectCount; ++index)
    {
        const auto& object = scene.objects[index];
        const auto* program = arbitgpu::nativeSurfaceForObject (draw.material.get(), object.id);
        if (program == nullptr || program->baseColorSource != Program::BaseColorSource::GraphFrameSrgbTexture)
            continue;
        const auto frame = arbitgpu::materialFrameForProgram (program, in);
        if (frame == nullptr)
            return fail (error, "Path Trace material names a frame texture the draw does not carry");
        const auto found = std::find (frames.begin(), frames.end(), frame);
        const auto layer = static_cast<std::uint32_t> (found - frames.begin());
        if (found == frames.end())
            frames.push_back (frame);
        setup.objectFrames.push_back ({ object.id.value, layer });
    }
    if ((in.materialFrameTexture || ! in.materialFrameTextures.empty()) && frames.empty())
        return fail (error, "Path Trace received a frame texture that no scene object samples");
    return true;
}

// The parent chain of object index without the object's own local transform.
pt::Mat4 parentWorldMatrix (const HarmonicMIDI::grid::Visual3DScene& scene, std::size_t index)
{
    const auto parent = scene.objects[index].parent;
    for (std::size_t slot = 0; parent.isValid() && slot < scene.objectCount; ++slot)
        if (scene.objects[slot].id == parent)
            return pt::objectWorldMatrix (scene, slot);
    return {};
}

// Motion blur: the shutter, and the scene one frame earlier, as the raster's
// Motion pass reads it (sceneMotionUniforms). The camera one frame earlier is
// view.previousCamera, which buildSetup set for the Motion output too. Without
// an earlier frame everything holds still and the shutter traces the frame as it is.
void bindShutter (const SceneEngineDraw& draw, pt::EngineSceneSetup& setup)
{
    setup.shutterOpen = 0.0f;
    setup.shutterClose = 0.0f;
    setup.previousObjectWorlds.clear();
    setup.previousInstances.clear();
    setup.previousPositions = nullptr;
    if (! draw.motionBlur.active())
        return;
    setup.shutterOpen = draw.motionBlur.open;
    setup.shutterClose = draw.motionBlur.close;
    const auto& scene = *draw.snapshot;
    const auto& in = *draw.inputs;
    const auto previous = arbitgpu::priorMotionSample (in);

    SceneTransform3D runtime {};
    runtime.translation = { previous.objectTranslationOffset[0], previous.objectTranslationOffset[1],
                            previous.objectTranslationOffset[2] };
    runtime = arbitgpu::applyRuntimeObjectTransform (runtime, previous.objectRotationDegrees, previous.objectScale);
    const pt::Mat4 runtimeMatrix = pt::transformMatrix (runtime);
    setup.previousObjectWorlds.reserve (scene.objectCount);
    for (std::size_t index = 0; index < scene.objectCount; ++index)
    {
        const auto& object = scene.objects[index];
        pt::Mat4 world;
        if (arbitgpu::sceneObjectHasOwnPrior (scene, index))
            world = parentWorldMatrix (scene, index) * pt::transformMatrix (scene.previousObjectTransforms[index]);
        else
            world.m = arbitgpu::nativeObjectWorldMatrix (previous.objectWorlds ? previous.objectWorlds.get()
                                                                               : in.objectWorlds.get(),
                                                         object, pt::objectWorldMatrix (scene, index).m);
        if (arbitgpu::runtimeTargetsObject (in, object))
            world = runtimeMatrix * world;
        setup.previousObjectWorlds.push_back (world);
    }
    if (scene.previousObjectInstanceCount == scene.objectInstanceCount && scene.objectInstanceCount != 0)
        setup.previousInstances.assign (scene.previousObjectInstances.begin(),
                                        scene.previousObjectInstances.begin() + static_cast<std::ptrdiff_t> (scene.objectInstanceCount));
    if (previous.scenePositions != nullptr && previous.scenePositions->size() <= scene.vertexCount
        && arbitgpu::validSceneMotionPositions (previous.scenePositions))
        setup.previousPositions = previous.scenePositions;
}

bool buildSetup (const SceneEngineDraw& draw, pt::EngineSceneSetup& setup, pt::EngineView& view,
                 std::vector<std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>>& frames,
                 std::string& error)
{
    const auto& scene = *draw.snapshot;
    const auto& in = *draw.inputs;

    // Camera: override, else the animated camera, else the active scene camera,
    // then the runtime offsets (backend_opengl.cpp, runtimeCameraRecord).
    const auto* active = HarmonicMIDI::grid::visual3d_detail::findById (scene.cameras, scene.cameraCount,
                                                                        scene.activeCamera);
    if (! in.cameraOverride && ! in.animatedCamera && active == nullptr)
        return fail (error, "Path Trace requires an active camera");
    const SceneCameraRecord authored = in.cameraOverride ? *in.cameraOverride
        : in.animatedCamera ? *in.animatedCamera : *active;
    view.camera = arbitgpu::runtimeCameraRecord (authored, in.cameraTranslationOffset, in.cameraRotationDegrees,
                                                 in.cameraVerticalFovDegrees, in.cameraLens);
    // Focus on object: this frame's depth of the object sets the focus distance.
    HarmonicMIDI::grid::resolveSceneCameraFocus (scene, view.camera);
    if (! arbitgpu::validFixtureCameraOverride (view.camera))
        return fail (error, "Path Trace camera is invalid");
    // The prior camera for the Motion output and motion blur, as the raster's
    // sceneMotionUniforms builds it.
    view.previousCamera = arbitgpu::priorRuntimeCamera (in, authored, scene.previousCamera);
    if (view.previousCamera && ! arbitgpu::validFixtureCameraOverride (*view.previousCamera))
        view.previousCamera.reset();
    view.width = draw.dimensions.width;
    view.height = draw.dimensions.height;
    view.timeSeconds = in.timeSeconds;

    // Object worlds: runtime transform (targeted objects only) times the checked
    // static world or the authored chain, as the raster's objectMatrix.
    SceneTransform3D runtime {};
    runtime.translation = { in.objectTranslationOffset[0], in.objectTranslationOffset[1],
                            in.objectTranslationOffset[2] };
    runtime = arbitgpu::applyRuntimeObjectTransform (runtime, in.objectRotationDegrees, in.objectScale);
    const pt::Mat4 runtimeMatrix = pt::transformMatrix (runtime);
    setup.scene = draw.snapshot;
    setup.objectWorlds.clear();
    setup.objectWorlds.reserve (scene.objectCount);
    for (std::size_t index = 0; index < scene.objectCount; ++index)
    {
        const auto& object = scene.objects[index];
        pt::Mat4 world;
        world.m = arbitgpu::nativeObjectWorldMatrix (in.objectWorlds.get(), object,
                                                     pt::objectWorldMatrix (scene, index).m);
        if (arbitgpu::runtimeTargetsObject (in, object))
            world = runtimeMatrix * world;
        setup.objectWorlds.push_back (world);
    }
    bindShutter (draw, setup);
    if (! pathtracefixture::bindNoteInstances (scene, in, draw.motionBlur.active(), setup, error))
        return false;

    // Lights: the records the raster reads, overrides applied.
    if (! arbitgpu::validNativeImportedLightOverrideTargets (scene, in, error))
        return false;
    const std::size_t lightCount = in.lightOverride ? 1u : arbitgpu::effectiveNativeLightCount (scene, in);
    setup.lights.clear();
    setup.lights.reserve (lightCount);
    for (std::size_t index = 0; index < lightCount; ++index)
        setup.lights.push_back (arbitgpu::effectiveNativeLightRecord (scene, in, index));
    // The source of an authored spectral lighting plan is a light of the scene.
    // Nothing else carries it: it is not a scene record.
    setup.lightShapes.clear();
    if (draw.diffractionLighting.has_value())
    {
        HarmonicMIDI::grid::SceneLightRecord source;
        pt::LightShape shape;
        if (pt::authoredSpectralSourceLight (*draw.diffractionLighting, source, shape))
        {
            setup.lightShapes.resize (setup.lights.size());
            setup.lights.push_back (source);
            setup.lightShapes.push_back (shape);
        }
    }
    setup.emissionGain = in.emissionGain;

    // Diffraction: the admitted material scatters its object by diffraction
    // orders. Runtime diffraction programs without that material are refused.
    setup.objectGratings.clear();
    const bool diffractionInputs = ! in.diffractionParameters.empty() || in.diffractionSourceProgram
                                   || in.diffractionEvaluatedProgram;
    if (diffractionInputs && draw.diffractionMaterial == nullptr)
        return fail (error, "Path Trace needs the admitted diffraction material of these diffraction inputs");
    // Every object the raster draws with a diffraction program scatters by the
    // draw's admitted material, with that program's product occupancy.
    const auto bindGrating = [&] (HarmonicMIDI::grid::SceneObjectId object) {
        pt::ObjectGratingBinding binding { object.value, draw.diffractionMaterial, draw.diffractionField };
        const auto* program = arbitgpu::nativeSurfaceForObject (draw.material.get(), object);
        if (program != nullptr && program->kind == arbitgpu::NativeFixtureMaterialKind::DiffractionReflective
            && program->diffractionOccupancyRectangleCount != 0)
        {
            binding.occupancyCount = std::min<std::uint32_t> (program->diffractionOccupancyRectangleCount, 5u);
            for (std::size_t index = 0; index < binding.occupancyCount; ++index)
            {
                const auto& rectangle = program->diffractionOccupancyRectangles[index];
                binding.occupancy[index] = { rectangle.x, rectangle.y, rectangle.z, rectangle.w };
            }
            binding.maskCoverage = program->diffractionMaskCoverage;
        }
        const bool partial = binding.occupancyCount != 0;
        setup.objectGratings.push_back (std::move (binding));
        return partial;
    };
    std::vector<std::uint32_t> partialFoils;
    if (draw.diffractionMaterial != nullptr)
    {
        if (HarmonicMIDI::grid::visual3d_detail::findById (scene.objects, scene.objectCount,
                                                           draw.diffractionObject) == nullptr)
            return fail (error, "Path Trace diffraction material names no scene object");
        if (bindGrating (draw.diffractionObject))
            partialFoils.push_back (draw.diffractionObject.value);
    }

    if (! bindFrameLayers (draw, setup, frames, error))
        return false;

    // Materials: objects bound to a Surface program use its block. A grating
    // object scatters by its grating instead.
    setup.objectMaterials.clear();
    if (draw.material != nullptr)
    {
        // A mask shows the object's own material through the foil.
        bool maskedField = false;
        if (draw.diffractionField.has_value())
            for (const auto& corner : *draw.diffractionField)
                maskedField = maskedField || corner[3] < 1.0f;
        for (std::size_t index = 0; index < scene.objectCount; ++index)
        {
            const auto& object = scene.objects[index];
            const auto* program = arbitgpu::nativeSurfaceForObject (draw.material.get(), object.id);
            if (program == nullptr) continue;
            const bool diffractionProgram = program->kind == arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;
            bool gratingObject = draw.diffractionMaterial != nullptr && object.id == draw.diffractionObject;
            if (diffractionProgram && ! gratingObject)
            {
                if (draw.diffractionMaterial == nullptr)
                    return fail (error, "Path Trace needs the admitted diffraction material of a diffraction program");
                if (bindGrating (object.id))
                    partialFoils.push_back (object.id.value);
                gratingObject = true;
            }
            const bool partial = maskedField
                || std::find (partialFoils.begin(), partialFoils.end(), object.id.value) != partialFoils.end();
            if (gratingObject && ! partial) continue;
            // The foil's own diffraction program is the grating above. Under its
            // mask the object keeps its scene material, as the raster's litBase.
            if (gratingObject && diffractionProgram) continue;
            const auto* record = HarmonicMIDI::grid::visual3d_detail::findById (scene.materials, scene.materialCount,
                                                                                object.material);
            pt::ObjectMaterialOverride entry;
            entry.objectId = object.id.value;
            if (! pathtracefixture::mapSurfaceProgram (*program, record, in.timeSeconds, entry.material, error))
                return false;
            setup.objectMaterials.push_back (entry);
        }
    }
    return true;
}

// The selected device's own readiness call, the one the engine polls once a draw has been
// lowered. Asking first lets the renderer draw the Raster picture meanwhile. An Auto
// selection held on the CPU by an earlier device failure has no kernel to wait for.
bool deviceKernelBuilding() noexcept
{
    const auto choice = pathTraceDeviceSelection();
    const auto& selection = choice->selection;
    if (! selection.ok)
        return false;
    if (std::strcmp (selection.requested, "Auto") == 0 && pt::device::autoDeviceLatch().failure() != nullptr)
        return false;
    switch (selection.device)
    {
#if defined(ARBIT_HAVE_OPTIX_KERNEL)
        case pt::device::DeviceKind::OptiX: return ! pt::device::optixPipelineReady();
#endif
        case pt::device::DeviceKind::Cuda: return ! pt::device::cudaModuleReady();
        case pt::device::DeviceKind::Hip: return ! pt::device::hipModuleReady();
        case pt::device::DeviceKind::OneApi: return ! pt::device::oneApiModuleReady();
#if defined(ARBIT_HAVE_METAL_PATHTRACE)
        case pt::device::DeviceKind::Metal: return ! pt::device::metalPipelineReady();
#endif
        default: return false;
    }
}

class PathTraceFixtureEngine final : public SceneRenderEngine
{
public:
    ~PathTraceFixtureEngine() override { joinPrimer(); }

    explicit PathTraceFixtureEngine (pathtracefixture::CpuFrameUpload upload) : upload_ (upload)
    {
        engine_.samplesPerPass = 1;
        engine_.environmentImageSource = [] (const HarmonicMIDI::grid::SceneEnvironmentImage& image) {
            return pt::residentEnvironmentPixels (image);
        };
    }

    // The selected device's own readiness call, the one the engine polls once a draw has been
    // lowered. Asking first lets the renderer draw the Raster picture meanwhile.
    bool kernelBuilding() noexcept override
    {
        const bool building = ! fellBackToCpu_ && deviceKernelBuilding();
        if (! building)
            primedThisWait_ = false;
        return building;
    }

    // Lowers a draw to the engine's scene setup (setup_) and view: the fog copy, the vertex
    // programs' displaced scene and the object, material and light tables. Render thread only.
    bool lower (const SceneEngineDraw& draw, pt::EngineView& view, std::string& error)
    {
        if (draw.snapshot == nullptr || draw.inputs == nullptr)
            return fail (error, "Path Trace draw is incomplete");
        if (! pathtracefixture::admitInputs (*draw.inputs, error))
            return false;
        auto drawn = draw;
        // Fog from an imported scene's Environment Light replaces the scene's own.
        // The copy is kept while the source and fog stay put, so accumulation holds.
        if (draw.worldVolume && ! HarmonicMIDI::grid::sameSceneVolume (*draw.worldVolume, drawn.snapshot->worldVolume))
        {
            if (fogScene_ == nullptr || fogSceneSource_ != drawn.snapshot
                || ! HarmonicMIDI::grid::sameSceneVolume (fogScene_->worldVolume, *draw.worldVolume))
            {
                auto fogged = std::make_shared<HarmonicMIDI::grid::Visual3DScene> (*drawn.snapshot);
                fogged->worldVolume = *draw.worldVolume;
                fogScene_ = std::move (fogged);
                fogSceneSource_ = drawn.snapshot;
            }
            drawn.snapshot = fogScene_;
        }
        // Vertex programs displace the scene before the BVH. A program whose
        // inputs did not change keeps the moved copy, and with it the accumulation.
        std::shared_ptr<const std::vector<HarmonicMIDI::grid::SceneVec3>> displacedOpen;
        if (draw.material != nullptr && hasVertexProgram (*draw.material))
        {
            // The shutter opens on the prior frame's displaced positions, so a
            // program that moves its vertices over time blurs along the move.
            if (draw.motionBlur.active())
            {
                const auto prior = videohelper::vertexprogramcpu::priorInputs (*draw.inputs);
                const auto& positions = draw.inputs->previousMotion ? draw.inputs->previousMotion->scenePositions : nullptr;
                if (displacedPrior_ == nullptr || displacedPriorSource_ != drawn.snapshot
                    || displacedPriorProgram_ != draw.material || displacedPriorPositions_ != positions
                    || displacedPriorTime_ != prior.timeSeconds || displacedPriorSpectrum_ != prior.spectrum
                    || displacedPriorControls_ != prior.controls)
                {
                    if (! videohelper::vertexprogramcpu::priorDisplacedPositions (*drawn.snapshot, *draw.material, *draw.inputs,
                                                                                  displacedPrior_, error))
                        return false;
                    displacedPriorSource_ = drawn.snapshot;
                    displacedPriorProgram_ = draw.material;
                    displacedPriorPositions_ = positions;
                    displacedPriorTime_ = prior.timeSeconds;
                    displacedPriorSpectrum_ = prior.spectrum;
                    displacedPriorControls_ = prior.controls;
                }
                displacedOpen = displacedPrior_;
            }
            if (displacedScene_ == nullptr || displacedSource_ != drawn.snapshot || displacedProgram_ != draw.material
                || displacedTime_ != draw.inputs->timeSeconds || displacedSpectrum_ != draw.inputs->vertexSpectrum
                || displacedControls_ != draw.inputs->vertexControls)
            {
                if (! displacedScene (drawn.snapshot, *draw.material, *draw.inputs, displacedScene_, error))
                    return false;
                displacedSource_ = drawn.snapshot;
                displacedProgram_ = draw.material;
                displacedTime_ = draw.inputs->timeSeconds;
                displacedSpectrum_ = draw.inputs->vertexSpectrum;
                displacedControls_ = draw.inputs->vertexControls;
            }
            drawn.snapshot = displacedScene_;
        }
        if (! buildSetup (drawn, setup_, view, frames_, error))
            return false;
        // bindShutter read the undisplaced prior positions; a vertex program's
        // shutter opens on its own prior evaluation instead.
        if (displacedOpen != nullptr)
            setup_.previousPositions = std::move (displacedOpen);
        return true;
    }

    // While the device's kernel builds, the CPU scene for the first Path Trace frame is built on
    // a worker, so the frame after the kernel is ready doesn't start with that wait. The draw is
    // lowered here and only the engine's own prepare runs on the worker. It runs once per wait,
    // as lowering every frame of an animated scene would cost the stand-in its frame rate.
    // render() joins it first.
    void primeWhileKernelBuilds (const SceneEngineDraw& draw) noexcept override try
    {
        if (primerRunning_.load())
            return;
        if (primer_.joinable())
            primer_.join();
        if (primedThisWait_ || fellBackToCpu_ || draw.snapshot == nullptr || draw.inputs == nullptr)
            return;
        pt::EngineView view;
        std::string ignored;
        if (! lower (draw, view, ignored))
            return;
        primedThisWait_ = true;
        primerRunning_ = true;
        primer_ = std::thread ([this, setup = setup_, view]
        {
            std::string error;
            try { engine_.primeScene (setup, view, error); }
            catch (...) {}
            primerRunning_ = false;
        });
    }
    catch (...)
    {
        primerRunning_ = false;
    }

    void joinPrimer() noexcept
    {
        if (primer_.joinable())
            primer_.join();
    }

    bool render (const SceneEngineDraw& draw, RenderedFrame& output, std::string& error) override try
    {
        joinPrimer();
        const auto callStart = pt::PhaseClock::now();
        // A named device that does not load stops the frame by name. CPU, OptiX, CUDA
        // and HIP are the devices with a render path.
        const auto deviceChoice = pathTraceDeviceSelection();
        const auto& selection = deviceChoice->selection;
        // A changed setting starts this engine over on the new device.
        if (choice_ == nullptr || choice_->generation != deviceChoice->generation)
        {
            if (devicePassSet_ || fellBackToCpu_)
            {
                engine_.setDevicePass ({});
                engine_.restartPreview();
            }
            devicePassSet_ = false;
            fellBackToCpu_ = false;
            deviceReason_ = nullptr;
            choice_ = deviceChoice;
        }
        if (! selection.ok)
            return fail (error, selection.error);
        // A failure another engine hit holds here too, so this engine compiles nothing.
        // When the latch lets go, this engine goes back to the device on a new film.
        if (std::strcmp (selection.requested, "Auto") == 0)
        {
            auto held = pt::device::autoDeviceLatch().failure();
            if (held != nullptr && ! fellBackToCpu_)
            {
                engine_.setDevicePass ({});
                fellBackToCpu_ = true;
                deviceReason_ = std::move (held);
                engine_.restartPreview();
            }
            else if (held == nullptr && fellBackToCpu_)
            {
                fellBackToCpu_ = false;
                devicePassSet_ = false;
                deviceReason_ = nullptr;
                engine_.restartPreview();
            }
        }
        if (! devicePassSet_ && ! fellBackToCpu_)
        {
#if defined(ARBIT_HAVE_OPTIX_KERNEL)
            if (selection.device == pt::device::DeviceKind::OptiX)
            {
                engine_.setDevicePass (pt::device::makeOptixDevicePass ([this] { return engine_.sceneBuilds(); }),
                                       [] { return pt::device::optixPipelineReady(); });
                devicePassSet_ = true;
            }
#endif
            if (selection.device == pt::device::DeviceKind::Cuda)
            {
                engine_.setDevicePass (pt::device::makeCudaDevicePass ([this] { return engine_.sceneBuilds(); }),
                                       [] { return pt::device::cudaModuleReady(); });
                devicePassSet_ = true;
            }
            else if (selection.device == pt::device::DeviceKind::Hip)
            {
                engine_.setDevicePass (pt::device::makeHipDevicePass ([this] { return engine_.sceneBuilds(); }),
                                       [] { return pt::device::hipModuleReady(); });
                devicePassSet_ = true;
            }
#if defined(ARBIT_HAVE_METAL_PATHTRACE)
            else if (selection.device == pt::device::DeviceKind::Metal)
            {
                engine_.setDevicePass (pt::device::makeMetalDevicePass ([this] { return engine_.sceneBuilds(); }),
                                       [] { return pt::device::metalPipelineReady(); });
                devicePassSet_ = true;
            }
#endif
            else if (selection.device == pt::device::DeviceKind::OneApi)
            {
                engine_.setDevicePass (pt::device::makeOneApiDevicePass ([this] { return engine_.sceneBuilds(); }),
                                       [] { return pt::device::oneApiModuleReady(); });
                devicePassSet_ = true;
            }
            if (selection.device != pt::device::DeviceKind::Cpu && ! devicePassSet_)
                return fail (error, std::string ("Path Trace cannot launch ") + pt::device::deviceName (selection.device) + " in this build");
        }
        if (! fellBackToCpu_ && deviceReason_ == nullptr)
            if (auto reason = pt::device::autoPassedOverReason (selection); ! reason.empty())
                deviceReason_ = std::make_shared<const std::string> (std::move (reason));
        pt::EngineView view;
        if (! lower (draw, view, error))
            return false;
        const auto color = draw.inputs->linearColor ? pt::EngineColor::LinearRgba16f : pt::EngineColor::DisplayRgba8;
        pt::EngineImage image;
        const bool preview = draw.use == RenderUse::Preview;
        // The preview target travels with the draw and is written on preview draws
        // alone, so an export draw never moves the cap. Its value comes from the
        // mode below: an Accurate preview targets the draw's exportSamples, so the
        // settled preview is the export frame, and a Fast one targets previewSamples.
        if (preview)
        {
            // Accurate previews draw the export sample streams and settle at the export
            // sample count, so the settled preview is the export frame. Fast keeps
            // independent streams and converges to previewSamples.
            const bool accurate = draw.mode == SceneEngineMode::Accurate;
            engine_.previewSeededLikeExport = accurate;
            engine_.maximumPreviewSamples = pt::previewSampleTarget (accurate, draw.previewSamples, draw.exportSamples);
            engine_.playbackActive = draw.transportPlaying;
            engine_.watchInputs (draw.inputGeneration.has_value() ? &videohelper::renderinput::counter (draw.inputClip) : nullptr,
                                 draw.inputGeneration.value_or (0));
        }
        // Render Mode Fast traces cheaper previews and exports (render_mode_policy.h);
        // Accurate keeps the node's settings.
        const bool fast = draw.mode == SceneEngineMode::Fast;
        const bool fastPreview = preview && fast;
        const std::uint32_t renderedExportSamples = exportSamples (fast, draw.exportSamples, draw.previewSamples);
        const auto cost = pathTracePreviewCost (fast, draw.maxBounces, draw.maxVolumeBounces,
                                                pt::kDefaultPlaybackSamples);
        engine_.settings().maxDepth = cost.maxBounces;
        engine_.settings().maxVolumeBounces = cost.maxVolumeBounces;
        if (preview)
            engine_.playbackSamples = cost.playbackSamples;
        engine_.settings().volumesVisibleToCamera = draw.volumesVisibleToCamera;
        pt::applyLightPaths (draw.lightPaths, engine_.settings());
        engine_.setDeviceSceneFallback (std::strcmp (selection.requested, "Auto") == 0);
        const auto renderOnDevice = [&] {
            return preview ? engine_.renderPreview (setup_, view, color, image, error)
                           : engine_.renderExport (setup_, view, color, renderedExportSamples, image, error);
        };
        // Everything above lowered the draw to the tracer's setup; the engine adds its own lowering.
        const double fixtureLowerMs = pt::millisecondsSince (callStart);
        bool rendered = renderOnDevice();
        if (rendered && image.pending)
        {
            RenderedFrame waiting;
            waiting.use = draw.use;
            waiting.sceneId = draw.snapshot->id;
            waiting.dimensions = draw.dimensions;
            waiting.pending = true;
            output = std::move (waiting);
            error.clear();
            return true;
        }
        // Auto never loses a frame to a device that started and then failed. The failed
        // pass added nothing, but earlier passes may have, so the CPU starts a new
        // generation on an empty film, by name in the log. A device asked for by name
        // keeps its error. The latch keeps this failure past engine rebuilds. Moving the
        // Render node off Path Trace tries the device again, or a helper restart once the
        // context is lost, and the reason says which.
        if (! rendered && image.stats.deviceFailed && std::strcmp (selection.requested, "Auto") == 0)
        {
            std::fprintf (stderr, "[pt] device=CPU requested=Auto: %s\n", error.c_str());
            engine_.setDevicePass ({});
            fellBackToCpu_ = true;
            deviceReason_ = std::make_shared<const std::string> (pt::device::deviceFailureReason (error, pt::device::deviceLost(), selection.device));
            pt::device::autoDeviceLatch().fail (deviceReason_);
            engine_.restartPreview();
            image = {};
            error.clear();
            rendered = renderOnDevice();
        }
        if (! rendered)
            return false;
        // A capped preview traces nothing and leaves the film as the last pass drew it.
        if (image.traced)
        {
            sceneOnCpu_ = image.stats.deviceSkipped != nullptr;
            if (sceneOnCpu_ && sceneReason_ == nullptr)
                sceneReason_ = std::make_shared<const std::string> (image.stats.deviceSkipped);
        }
        noteStats (draw, image);
        // A capped preview that traced nothing keeps the phases of the call that last traced.
        const bool timedThisCall = image.traced || image.restarted;
        const bool restartedThisCall = image.restarted;
        const bool tracedThisCall = image.traced;
        if (timedThisCall)
            stats_.phases.add (pt::FramePhase::SceneLower, fixtureLowerMs);
        // Capped preview: nothing new to resolve, so upload the last image again.
        // A fresh frame per call keeps the raster's one-frame-per-render contract.
        const bool resolvedNewImage = ! image.unchanged;
        if (resolvedNewImage)
            last_ = std::move (image);
        else if (last_.width != view.width || last_.height != view.height || last_.color != color)
            return fail (error, "Path Trace lost its resolved preview image");
        // The denoiser reads last_ and writes denoised_, so accumulation never
        // sees a denoised image and a switched-off frame uploads last_ untouched.
        const auto choice = draw.denoise ? draw.denoiser : pathtracedenoise::DenoiserChoice::Off;
        if (choice == pathtracedenoise::DenoiserChoice::Off)
        {
            denoised_.clear();
            denoisedLayers_.clear();
            reportDenoiser (pathtracedenoise::Denoiser::None, "none", {});
        }
        else
        {
            const auto denoiseStart = pt::PhaseClock::now();
            // A Fast Preview still converging denoises at 1, 2, 4, 8... samples
            // per pixel (render_mode_policy.h); a new denoiser choice always reruns.
            const bool converging = preview && last_.mode == pt::EngineRenderMode::Converging
                                 && engine_.previewProgressing();
            const bool due = resolvedNewImage && denoiseThisImage (fastPreview, converging, last_.samplesPerPixel);
            if ((due || denoised_.empty() || denoisedLayers_.size() != last_.frameLayers.size()
                 || choice != denoisedChoice_)
                && ! denoiseLastImage (draw, view, choice, error))
                return false;
            if (timedThisCall)
                stats_.phases.set (pt::FramePhase::Denoise, pt::millisecondsSince (denoiseStart));
        }
        if (timedThisCall && choice == pathtracedenoise::DenoiserChoice::Off)
            stats_.phases.markAbsent (pt::FramePhase::Denoise, "the denoiser is off");
        denoisedChoice_ = choice;
        stats_.denoiser = denoiserLabel_;
        stats_.denoiserRequested = pathtracedenoise::choiceName (choice);
        stats_.denoiserReason = denoiserReason_;
        arbitgpu::CpuSceneImage upload;
        upload.width = last_.width;
        upload.height = last_.height;
        upload.linearColor = last_.color == pt::EngineColor::LinearRgba16f;
        upload.color = choice != pathtracedenoise::DenoiserChoice::Off ? &denoised_ : &last_.colorBytes;
        upload.depth = &last_.depth;
        if (last_.frameLayers.size() != frames_.size())
            return fail (error, "Path Trace resolved another number of frame layers than the draw has frames");
        for (std::size_t index = 0; index < frames_.size(); ++index)
            upload.frameLayers.push_back ({ frames_[index],
                choice != pathtracedenoise::DenoiserChoice::Off ? &denoisedLayers_[index] : &last_.frameLayers[index].lightBytes,
                &last_.frameLayers[index].uvCoverage });
        std::copy (pt::kRasterBackground, pt::kRasterBackground + 3, upload.background);
        // The outputs travel only when something reads them. Denoising touched the
        // Image alone, so they are the centre ray's values either way.
        if (pathtracefixture::readsRenderOutputs (*draw.inputs))
        {
            upload.normal = &last_.normalPass;
            upload.emission = &last_.emissionPass;
            upload.mask = &last_.maskPass;
            upload.materialId = &last_.materialIdPass;
            upload.objectId = &last_.objectIdPass;
            upload.motion = &last_.motionPass;
            upload.imageOutput = draw.inputs->imageOutput;
            upload.passProgram = arbitgpu::scenePassProgram (*draw.inputs);
            if (draw.inputs->passComposite && draw.inputs->passComposite->inspectionImage)
                upload.resolveClear = renderpasscomposite::inspectionClear (*draw.inputs->passComposite);
        }
        const auto imageUploadStart = pt::PhaseClock::now();
        auto frame = upload_ (draw.backend, upload, error);
        if (frame == nullptr)
            return false;
        const auto uploaded = pt::PhaseClock::now();
        if (timedThisCall)
        {
            stats_.phases.set (pt::FramePhase::ImageUpload, pt::millisecondsBetween (imageUploadStart, uploaded));
            stats_.phases.set (pt::FramePhase::Total, pt::millisecondsBetween (callStart, uploaded));
            // The viewport's first-frame stage totals take the same phases, in enum order.
            static constexpr const char* kStageLabels[pt::kFramePhaseCount] = {
                "pt:sceneLower", "pt:cpuBvhBuild", "pt:accelBuild", "pt:pipelineWait", "pt:upload",
                "pt:depthPass", "pt:trace", "pt:readback", "pt:resolve", "pt:denoise", "pt:imageUpload",
                "pt:total" };
            for (std::size_t i = 0; i < pt::kFramePhaseCount; ++i)
                if (stats_.phases.has (static_cast<pt::FramePhase> (i)))
                    loopstep::addStage (kStageLabels[i], static_cast<std::int64_t> (
                        stats_.phases.get (static_cast<pt::FramePhase> (i)) * 1e6));
        }
        noteConvergence (draw, restartedThisCall, tracedThisCall, callStart, uploaded);
        RenderedFrame result;
        result.use = draw.use;
        result.sceneId = draw.snapshot->id;
        result.dimensions = draw.dimensions;
        result.nativeFrame = std::move (frame);
        result.stats.textureUploadCount = 2 + 2 * static_cast<std::uint32_t> (frames_.size())
            + (upload.normal != nullptr ? 6u : 0u);
        result.stats.textureBytes = static_cast<std::uint64_t> (last_.colorBytes.size())
            + static_cast<std::uint64_t> (last_.depth.size()) * sizeof (float);
        if (upload.normal != nullptr)
            result.stats.textureBytes += (last_.normalPass.size() + last_.emissionPass.size() + last_.motionPass.size()) * 2u
                + last_.maskPass.size() + (last_.materialIdPass.size() + last_.objectIdPass.size()) * 4u;
        for (const auto& layer : last_.frameLayers)
            result.stats.textureBytes += static_cast<std::uint64_t> (layer.lightBytes.size())
                + static_cast<std::uint64_t> (layer.uvCoverage.size()) * sizeof (float);
        output = std::move (result);
        error.clear();
        return true;
    }
    catch (const std::bad_alloc&)
    {
        return fail (error, "Path Trace could not allocate this frame's scene");
    }
    catch (const std::length_error&)
    {
        return fail (error, "Path Trace could not allocate this frame's scene");
    }

    void cancel() noexcept override { engine_.cancel(); }
    bool previewProgressing() const noexcept override { return engine_.previewProgressing(); }
    unsigned previewTarget() const noexcept override { return engine_.previewTarget(); }
    unsigned previewSampleCount() const noexcept override { return engine_.previewSampleCount(); }
    const char* denoiserRan() const noexcept override { return denoiserLabel_; }

    bool pathTraceStats (PathTraceFrameStats& stats) const noexcept override
    {
        if (! statsValid_)
            return false;
        stats = stats_;
        return true;
    }

private:
    // RenderStats covers one pass, and a Preview runs one pass per call. Sum the
    // passes of the generation so the counters match the accumulated image.
    void noteStats (const SceneEngineDraw& draw, const pt::EngineImage& image) noexcept
    {
        if (draw.use != RenderUse::Preview || image.restarted)
            stats_ = {};
        stats_.use = draw.use;
        stats_.mode = draw.mode;
        stats_.maxBounces = engine_.settings().maxDepth;
        const auto choice = choice_ != nullptr ? choice_ : pathTraceDeviceSelection();
        stats_.device = pt::device::deviceName (fellBackToCpu_ || sceneOnCpu_ ? pt::device::DeviceKind::Cpu : choice->selection.device);
        stats_.deviceRequested = choice->selection.requested;
        stats_.deviceReason = deviceReason_ != nullptr ? deviceReason_ : sceneOnCpu_ ? sceneReason_ : nullptr;
        stats_.fallbackReason = fellBackToCpu_ ? deviceReason_ : nullptr;
        stats_.temporalDenoise = image.temporalDenoise;
        const bool preview = draw.use == RenderUse::Preview;
        if (preview && engine_.lastPassObsolete())
            ++obsoletePasses_;
        stats_.obsoletePasses = obsoletePasses_;
        const bool playback = preview && image.mode == pt::EngineRenderMode::PlaybackFixed;
        stats_.renderActivity = ! preview ? "export"
            : playback ? "playback-fixed"
            : engine_.previewProgressing() ? "converging" : "settled";
        stats_.sampleTarget = ! preview ? exportSamples (draw.mode == SceneEngineMode::Fast, draw.exportSamples, draw.previewSamples)
            : playback ? std::min (std::max (1u, engine_.playbackSamples), engine_.maximumPreviewSamples)
            : engine_.previewTarget();
        if (image.traced || image.restarted)
        {
            stats_.phases = image.stats.phases;
            stats_.phaseSamples = image.traced ? image.stats.samplesAdded : 0;
        }
        if (image.traced)
        {
            stats_.samplesAdded += image.stats.samplesAdded;
            stats_.samplesPerPixel = image.samplesPerPixel;
            stats_.nonFiniteSamples += image.stats.nonFiniteSamples;
            stats_.bounceLimitPaths += image.stats.bounceLimitPaths;
            stats_.transparentLimitPaths += image.stats.transparentLimitPaths;
            stats_.limitEndedPaths += image.stats.limitEndedPaths;
            stats_.lightPaths += image.stats.lightPaths;
            stats_.causticSplats += image.stats.causticSplats;
            if (image.stats.deviceMemoryTotalBytes != 0)
            {
                stats_.deviceMemoryFreeBytes = image.stats.deviceMemoryFreeBytes;
                stats_.deviceMemoryTotalBytes = image.stats.deviceMemoryTotalBytes;
            }
            statsValid_ = true;
        }
    }

    // The viewport's clocks from an accumulation's restart, read after the image
    // reached the compositor's texture. They stop there, before the compositor and
    // the window's swap, and they start at the render call that restarted, so the
    // time an edit takes to reach the tracer is not in them.
    void noteConvergence (const SceneEngineDraw& draw, bool restarted, bool traced,
                          pt::PhaseClock::time_point callStart, pt::PhaseClock::time_point uploaded) noexcept
    {
        if (draw.use != RenderUse::Preview)
        {
            stats_.firstSampleMs.reset();
            stats_.targetMs.reset();
            stats_.convergenceAbsent = "an export traces its samples in one call";
            return;
        }
        if (restarted)
        {
            accumulationStart_ = callStart;
            firstSampleMs_.reset();
            targetMs_.reset();
        }
        if (accumulationStart_.has_value())
        {
            if (traced && ! firstSampleMs_.has_value())
                firstSampleMs_ = pt::millisecondsBetween (*accumulationStart_, uploaded);
            const bool playback = last_.mode == pt::EngineRenderMode::PlaybackFixed;
            if (! playback && ! targetMs_.has_value() && last_.samplesPerPixel >= engine_.previewTarget())
                targetMs_ = pt::millisecondsBetween (*accumulationStart_, uploaded);
        }
        stats_.firstSampleMs = firstSampleMs_;
        stats_.targetMs = targetMs_;
        stats_.convergenceAbsent = ! accumulationStart_.has_value() ? "no accumulation has restarted yet"
            : last_.mode == pt::EngineRenderMode::PlaybackFixed ? "playback frames stop at the playback budget"
            : ! targetMs_.has_value() ? "the accumulation has not reached its target yet" : nullptr;
    }

    std::optional<pt::PhaseClock::time_point> accumulationStart_;
    std::optional<double> firstSampleMs_;
    std::optional<double> targetMs_;
    PathTraceFrameStats stats_;
    bool statsValid_ = false;
    std::uint64_t obsoletePasses_ = 0;
    // Plans the denoiser for last_ from the choice and what can run, runs it, and
    // falls back by name when it fails: OptiX AI to OIDN, OIDN to the native filter.
    bool denoiseLastImage (const SceneEngineDraw& draw, const pt::EngineView& view,
                           pathtracedenoise::DenoiserChoice choice, std::string& error)
    {
        namespace dn = pathtracedenoise;
        dn::DenoiserAvailability available;
        available.tracedOnOptix = std::strcmp (stats_.device, pt::device::deviceName (pt::device::DeviceKind::OptiX)) == 0;
#if defined(ARBIT_HAVE_OPTIX)
        available.optixProblem = pt::device::optixDenoiserProblem();
#else
        available.optixProblem = "this build of DonutStudio has no OptiX device";
#endif
        available.oidnProblem = dn::oidnLoadProblem();
        auto plan = dn::planDenoiser (choice, available);
        std::string reason = plan.reason;
        const auto addReason = [&reason] (const std::string& why) {
            reason += reason.empty() ? why : "; " + why;
        };
        dn::OidnDevice oidnUsed = dn::OidnDevice::Cpu;
        // The temporal model follows playback frames. Converging, settled and export
        // frames each stand alone, so they take the single-frame model.
        const bool temporal = plan.denoiser == dn::Denoiser::OptixAi && draw.use == RenderUse::Preview
            && last_.mode == pt::EngineRenderMode::PlaybackFixed;
        bool ran = false;
        if (plan.denoiser == dn::Denoiser::OptixAi)
        {
            std::string failure, note;
            ran = runAiDenoiser (true, temporal, view, dn::OidnDevice::Cpu, oidnUsed, note, failure);
            if (! ran)
            {
                addReason ("OptiX AI failed: " + failure);
                plan.denoiser = available.oidnProblem.empty() ? dn::Denoiser::Oidn : dn::Denoiser::NativeFilter;
                if (! available.oidnProblem.empty())
                    addReason ("OIDN did not run: " + available.oidnProblem);
            }
        }
        if (! temporal)
            optixSequenceLive_ = false;
        if (! ran && plan.denoiser == dn::Denoiser::Oidn)
        {
            std::string failure, note;
            ran = runAiDenoiser (false, false, view, dn::oidnDeviceFor (stats_.device), oidnUsed, note, failure);
            if (ran && ! note.empty())
                addReason (note);
            if (! ran)
            {
                addReason ("OIDN failed: " + failure);
                plan.denoiser = dn::Denoiser::NativeFilter;
            }
        }
        if (! ran && plan.denoiser == dn::Denoiser::NativeFilter && ! nativeFilterLastImage (error))
            return false;
        restoreBackdrop();
        reportDenoiser (plan.denoiser, dn::denoiserLabel (plan.denoiser, oidnUsed, temporal), reason);
        return true;
    }

    // The backdrop is the raster's clear, not traced light, and its albedo guide is
    // zero, so a denoiser may darken it (round 31 showed (7, 10, 18) as (1, 3, 8)
    // after denoising). Put the clear back where nothing but the backdrop is.
    void restoreBackdrop() noexcept
    {
        const std::size_t pixels = static_cast<std::size_t> (last_.width) * last_.height;
        if (pixels == 0 || last_.backdropPass.size() != pixels || denoised_.size() != last_.colorBytes.size())
            return;
        const std::size_t stride = last_.colorBytes.size() / pixels;
        for (std::size_t pixel = 0; pixel < pixels; ++pixel)
            if (last_.backdropPass[pixel] != 0)
                std::memcpy (denoised_.data() + pixel * stride, last_.colorBytes.data() + pixel * stride, stride);
    }

    // OptiX AI (optix) or OIDN on last_'s colour and each frame layer, with its albedo
    // and normal guides. Fills failure and leaves denoised_ unusable when it fails.
    bool runAiDenoiser (bool optix, bool temporal, const pt::EngineView& view, pathtracedenoise::OidnDevice want,
                        pathtracedenoise::OidnDevice& used, std::string& note, std::string& failure)
    {
        namespace dn = pathtracedenoise;
        const std::size_t pixels = static_cast<std::size_t> (last_.width) * last_.height;
        if (last_.albedo.size() != pixels * dn::kGuideChannels || last_.normal.size() != pixels * dn::kGuideChannels)
        {
            failure = "the frame has no albedo and normal guides";
            return false;
        }
        pt::PinholeCamera camera;
        const bool haveCamera = pt::makePinholeCamera (view.camera, last_.width, last_.height, camera);
        const float* normals = last_.normal.data();
        if (optix)
        {
            // The temporal model reads camera-space normals; the single-frame model takes them too.
            if (! haveCamera)
            {
                failure = "the frame's camera is invalid";
                return false;
            }
            denoiseNormals_.resize (last_.normal.size());
            const pt::Quat toCamera = pt::conjugate (camera.rotation);
            for (std::size_t index = 0; index < pixels; ++index)
            {
                const float* n = last_.normal.data() + index * dn::kGuideChannels;
                const pt::Vec3 turned = pt::rotate (toCamera, pt::Vec3 { n[0], n[1], n[2] });
                denoiseNormals_[index * dn::kGuideChannels] = turned.x;
                denoiseNormals_[index * dn::kGuideChannels + 1] = turned.y;
                denoiseNormals_[index * dn::kGuideChannels + 2] = turned.z;
            }
            normals = denoiseNormals_.data();
        }
        bool usePrevious = false;
        if (temporal)
        {
            usePrevious = optixSequenceLive_
                && dn::reprojectionFlow (camera, optixSequenceCamera_, last_.depth, denoiseFlow_);
            if (! usePrevious)
                denoiseFlow_.assign (pixels * dn::kFlowChannels, 0.0f);
        }
        const auto denoise = [&] (const std::vector<std::uint8_t>& bytes, bool linear, bool sequence, std::vector<std::uint8_t>& output) {
            if (! dn::colorToFloat3 (bytes, linear, pixels, denoiseColor_))
            {
                failure = "the frame's colour does not match its size";
                return false;
            }
            denoiseOutput_.assign (pixels * dn::kGuideChannels, 0.0f);
            if (optix)
            {
#if defined(ARBIT_HAVE_OPTIX)
                pt::device::OptixDenoiseImage image;
                image.width = last_.width;
                image.height = last_.height;
                image.color = denoiseColor_.data();
                image.albedo = last_.albedo.data();
                image.normal = normals;
                image.output = denoiseOutput_.data();
                image.temporal = sequence;
                image.usePrevious = sequence && usePrevious;
                image.flow = sequence ? denoiseFlow_.data() : nullptr;
                if (! pt::device::optixDenoise (image, failure))
                    return false;
#else
                (void) sequence;
                failure = "this build of DonutStudio has no OptiX device";
                return false;
#endif
            }
            else
            {
                dn::DenoiseFrame frame;
                frame.width = last_.width;
                frame.height = last_.height;
                frame.color = denoiseColor_.data();
                frame.albedo = last_.albedo.data();
                frame.normal = normals;
                frame.output = denoiseOutput_.data();
                frame.hdr = linear;
                if (! dn::oidnDenoise (want, frame, used, note, failure))
                    return false;
            }
            dn::float3ToColor (denoiseOutput_, linear, bytes, output);
            return true;
        };
        // Each layer is denoised apart from the frame, so the denoiser never blurs the
        // imagery. A layer stands alone; only the frame's colour is a temporal sequence.
        if (! denoise (last_.colorBytes, last_.color == pt::EngineColor::LinearRgba16f, temporal, denoised_))
        {
            if (temporal)
                optixSequenceLive_ = false;
            return false;
        }
        if (temporal)
        {
            optixSequenceLive_ = true;
            optixSequenceCamera_ = camera;
        }
        denoisedLayers_.resize (last_.frameLayers.size());
        for (std::size_t index = 0; index < last_.frameLayers.size(); ++index)
            if (! denoise (last_.frameLayers[index].lightBytes, true, false, denoisedLayers_[index]))
                return false;
        return true;
    }

    bool nativeFilterLastImage (std::string& error)
    {
        try
        {
            const pt::CancellationToken neverCancelled;
            const auto filter = [&] (const std::vector<std::uint8_t>& input, bool linear, std::vector<std::uint8_t>& output) {
                if (! pathtracedenoise::loadPixels (input, linear, last_.width, last_.height, denoiseInput_))
                    return false;
                output.resize (input.size());
                pt::sharedThreadPool().runTiles (pt::TileGrid (last_.width, last_.height), neverCancelled,
                    [&] (const pt::Tile& tile, unsigned) {
                        pathtracedenoise::filterRegion (denoiseInput_.data(), last_.width, last_.height, linear,
                                                        tile.x0, tile.x1, tile.y0, tile.y1, output.data());
                    });
                if (! linear)
                    return true;
                if (! pathtracedenoise::loadPixels (output, true, last_.width, last_.height, denoiseFiltered_))
                    return false;
                pathtracedenoise::restoreEnergy (denoiseInput_.data(), denoiseFiltered_.data(), last_.width, last_.height,
                                                 denoiseScales_, denoiseTotals_, output.data(), [&] (auto&& region) {
                    pt::sharedThreadPool().runTiles (pt::TileGrid (last_.width, last_.height), neverCancelled,
                        [&] (const pt::Tile& tile, unsigned) { region (tile.x0, tile.x1, tile.y0, tile.y1); });
                });
                return true;
            };
            // Each layer filters apart from the frame, so the filter never blurs the imagery.
            bool complete = filter (last_.colorBytes, last_.color == pt::EngineColor::LinearRgba16f, denoised_);
            denoisedLayers_.resize (last_.frameLayers.size());
            for (std::size_t index = 0; complete && index < last_.frameLayers.size(); ++index)
                complete = filter (last_.frameLayers[index].lightBytes, true, denoisedLayers_[index]);
            if (! complete)
                return fail (error, "Path Trace denoise received an incomplete image");
        }
        catch (const std::exception&)
        {
            denoised_.clear();
            denoisedLayers_.clear();
            return fail (error, "Path Trace denoise ran out of resources");
        }
        return true;
    }

    // One log line per change, so the run's log names the denoiser and any
    // fallback without a line per frame.
    void reportDenoiser (pathtracedenoise::Denoiser denoiser, const char* label, const std::string& reason)
    {
        denoiserRan_ = denoiser;
        const bool sameReason = denoiserReason_ != nullptr ? *denoiserReason_ == reason : reason.empty();
        if (std::strcmp (label, denoiserLabel_) == 0 && sameReason)
            return;
        denoiserLabel_ = label;
        denoiserReason_ = reason.empty() ? nullptr : std::make_shared<const std::string> (reason);
        if (reason.empty())
            std::fprintf (stderr, "[pt] denoiser=%s\n", label);
        else
            std::fprintf (stderr, "[pt] denoiser=%s: %s\n", label, reason.c_str());
    }

    pathtracefixture::CpuFrameUpload upload_;
    pt::SceneEngine engine_;
    std::thread primer_;
    std::atomic<bool> primerRunning_ { false };
    bool primedThisWait_ = false; // the scene for this kernel wait has been handed to the worker
    // The device selection this engine renders with; a newer generation resets it.
    std::shared_ptr<const DeviceChoice> choice_;
    bool devicePassSet_ = false;
    bool fellBackToCpu_ = false;
    std::shared_ptr<const std::string> deviceReason_;
    bool sceneOnCpu_ = false;
    std::shared_ptr<const std::string> sceneReason_;
    pt::EngineSceneSetup setup_;  // reused so per-frame vectors keep their storage
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> fogScene_, fogSceneSource_;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> displacedScene_, displacedSource_;
    std::shared_ptr<const Program> displacedProgram_;
    float displacedTime_ = 0.0f;
    std::array<float, 64> displacedSpectrum_ {};
    std::vector<float> displacedControls_;
    // The prior frame's displaced positions, the shutter-open key of a vertex program.
    std::shared_ptr<const std::vector<HarmonicMIDI::grid::SceneVec3>> displacedPrior_, displacedPriorPositions_;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> displacedPriorSource_;
    std::shared_ptr<const Program> displacedPriorProgram_;
    double displacedPriorTime_ = 0.0;
    std::array<float, 64> displacedPriorSpectrum_ {};
    std::vector<float> displacedPriorControls_;
    pt::EngineImage last_;
    std::vector<float> denoiseInput_;
    std::vector<float> denoiseFiltered_;
    std::vector<float> denoiseScales_;
    std::vector<float> denoiseTotals_;
    std::vector<std::uint8_t> denoised_;
    std::vector<std::vector<std::uint8_t>> denoisedLayers_;
    // The frame of each layer of last_. Reassigned per draw, so a settled preview
    // samples the current frame without tracing again.
    std::vector<std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>> frames_;
    pathtracedenoise::Denoiser denoiserRan_ = pathtracedenoise::Denoiser::None;
    const char* denoiserLabel_ = "none";
    std::shared_ptr<const std::string> denoiserReason_;
    pathtracedenoise::DenoiserChoice denoisedChoice_ = pathtracedenoise::DenoiserChoice::Off;
    // The AI denoisers' float images, kept for their storage.
    std::vector<float> denoiseColor_;
    std::vector<float> denoiseOutput_;
    std::vector<float> denoiseNormals_;
    std::vector<float> denoiseFlow_;
    // The OptiX temporal sequence: live while playback frames follow each other,
    // with the camera of the last one for the flow.
    bool optixSequenceLive_ = false;
    pt::PinholeCamera optixSequenceCamera_ {};
};

} // namespace

std::shared_ptr<SceneRenderEngine> pathtracefixture::makeEngineWithUpload (CpuFrameUpload upload)
{
    try
    {
        return std::make_shared<PathTraceFixtureEngine> (upload);
    }
    catch (const std::exception&)
    {
        return nullptr;
    }
}

std::shared_ptr<SceneRenderEngine> pathtracefixture::makeEngine()
{
    return makeEngineWithUpload (&arbitgpu::uploadCpuSceneFrame);
}

namespace
{
struct PathTraceEngineRegistration final
{
    PathTraceEngineRegistration() noexcept
    {
        setPathTraceSceneEngineFactory (&pathtracefixture::makeEngine);
        setPathTraceKernelBuildingProbe (&deviceKernelBuilding);
    }
};

const PathTraceEngineRegistration registration;
} // namespace
namespace pathtracefixture
{
void prepareDeviceAtStartup()
{
    const auto choice = pathTraceDeviceSelection();
    const auto& selection = choice->selection;
    if (! selection.ok)
        return;
#if defined(ARBIT_HAVE_OPTIX_KERNEL)
    if (selection.device == pt::device::DeviceKind::OptiX)
    {
        // A kernel the driver's cache holds builds in 0.1 s, so it starts as it always did. A
        // miss is a compile of over a minute: it waits for the host's go (armKernelWarmup),
        // which comes once the window has painted and the app is idle, and runs at low
        // priority. The look at the cache loads the driver, so it stays off the caller's thread.
        std::thread ([]
        {
            if (pt::device::optixModuleHashPresent())
                pt::device::startOptixPipelineBuild();
        }).detach();
    }
#endif
    if (pt::device::isComputeDevice (selection.device))
        pt::device::startComputeModuleBuild (selection.device);
#if defined(ARBIT_HAVE_METAL_PATHTRACE)
    if (selection.device == pt::device::DeviceKind::Metal)
        pt::device::startMetalPipelineBuild();
#endif
}

void armKernelWarmup()
{
#if defined(ARBIT_HAVE_OPTIX_KERNEL)
    const auto choice = pathTraceDeviceSelection();
    if (choice->selection.ok && choice->selection.device == pt::device::DeviceKind::OptiX)
        pt::device::armOptixWarmup();
#endif
}

KernelWarmupInfo kernelWarmupInfo()
{
    KernelWarmupInfo info;
    const auto choice = pathTraceDeviceSelection();
#if defined(ARBIT_HAVE_OPTIX_KERNEL)
    if (choice->selection.ok && choice->selection.device == pt::device::DeviceKind::OptiX)
    {
        const auto reading = pt::device::readOptixWarmup();
        info.applies = true;
        info.state = pt::kernelcompile::warmupStateName (reading.state);
        info.reason = reading.reason;
        info.armed = reading.armed;
        info.moduleHashPresent = reading.moduleHashPresent;
        info.joined = reading.joined;
        info.contextReleased = reading.contextReleased;
        info.busy = reading.busy;
        info.lowPriority = reading.lowPriority;
        info.elapsedSeconds = reading.elapsedSeconds;
        info.error = reading.error;
        return info;
    }
#endif
    info.state = "idle";
    info.reason = choice->selection.ok && choice->selection.device == pt::device::DeviceKind::OptiX
        ? "this build has no OptiX kernel" : "the Path Trace device is not OptiX";
    return info;
}

std::string setDevicePreference (const std::string& value)
{
    {
        auto& setting = deviceSetting();
        const std::lock_guard<std::mutex> lock (setting.lock);
        if (setting.choice != nullptr && setting.preference == value)
            return setting.choice->selection.ok ? std::string() : setting.choice->selection.error;
        setting.preference = value;
        setting.choice = chooseDevice (setting.preference, ++setting.generations);
    }
    prepareDeviceAtStartup();
    const auto choice = pathTraceDeviceSelection();
    return choice->selection.ok ? std::string() : choice->selection.error;
}

std::string devicePreference()
{
    auto& setting = deviceSetting();
    const std::lock_guard<std::mutex> lock (setting.lock);
    return setting.preference;
}

bool cancelDeviceBuild()
{
    const auto device = pathTraceDeviceSelection()->selection.device;
    if (pt::device::isComputeDevice (device))
        return pt::device::stopComputeModuleBuildByUser (device);
#if defined(ARBIT_HAVE_METAL_PATHTRACE)
    if (device == pt::device::DeviceKind::Metal)
        return pt::device::stopMetalPipelineBuildByUser();
#endif
#if defined(ARBIT_HAVE_OPTIX_KERNEL)
    return pt::device::stopOptixPipelineBuildByUser();
#else
    return false;
#endif
}

void startDeviceBuildAgain()
{
    const auto device = pathTraceDeviceSelection()->selection.device;
    if (pt::device::isComputeDevice (device))
    {
        pt::device::startComputeModuleBuildByUser (device);
        return;
    }
#if defined(ARBIT_HAVE_METAL_PATHTRACE)
    if (device == pt::device::DeviceKind::Metal)
    {
        pt::device::startMetalPipelineBuildByUser();
        return;
    }
#endif
#if defined(ARBIT_HAVE_OPTIX_KERNEL)
    pt::device::startOptixPipelineBuildByUser();
#endif
}

void stopDeviceBuildAtExit()
{
#if defined(ARBIT_HAVE_OPTIX_KERNEL)
    pt::device::cancelOptixPipelineBuild();
#endif
    pt::device::cancelComputeModuleBuild (pt::device::DeviceKind::Cuda);
    pt::device::cancelComputeModuleBuild (pt::device::DeviceKind::Hip);
    pt::device::cancelComputeModuleBuild (pt::device::DeviceKind::OneApi);
#if defined(ARBIT_HAVE_METAL_PATHTRACE)
    pt::device::cancelMetalPipelineBuild();
#endif
}
} // namespace pathtracefixture
} // namespace videorender::fixture3d
