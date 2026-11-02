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
#if defined(ARBIT_HAVE_OPTIX_KERNEL)
#include "pathtrace/devices/device_optix_host.h"
#endif
#include "pathtrace/engine.h"
#include "pathtrace/environment_resident.h"
#include "pathtrace/thread_pool.h"
#include "pathtrace_preview_denoise.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace videorender::fixture3d
{
namespace
{
namespace pt = videohelper::pathtrace;
using HarmonicMIDI::grid::SceneCameraRecord;
using HarmonicMIDI::grid::SceneMaterialRecord;
using HarmonicMIDI::grid::SceneTransform3D;
using Program = arbitgpu::NativeFixtureSurfaceMaterialProgram;

// Scene units of medium after which a refracting material's base colour is
// the light left (Beer-Lambert), since the material IR has no distance.
constexpr float kTintDistance = 1.0f;

// The machine's device setting, read once. ARBIT_PATHTRACE_DEVICE carries it
// until Preferences has the row. The log names every device that was passed over.
const pt::device::DeviceSelection& pathTraceDeviceSelection()
{
    static const pt::device::DeviceSelection selection = []
    {
        auto chosen = pt::device::selectDevice (std::getenv ("ARBIT_PATHTRACE_DEVICE"), pt::device::productDeviceEnvironment());
        if (chosen.ok)
            std::fprintf (stderr, "[pt] device=%s requested=%s\n", pt::device::deviceName (chosen.device), chosen.requested);
        else
            std::fprintf (stderr, "[pt] %s\n", chosen.error.c_str());
        for (const auto& status : chosen.devices)
            if (! status.loadable)
                std::fprintf (stderr, "[pt] %s unavailable: %s\n", pt::device::deviceName (status.kind), status.problem.c_str());
        return chosen;
    }();
    return selection;
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
// rendering a different picture.
bool admitInputs (const arbitgpu::NativeFixtureSceneRuntimeInputs& in, std::string& error)
{
    if (! arbitgpu::validFixtureRuntimeInputs (in))
        return fail (error, "Path Trace runtime inputs are invalid");
    if (in.imageOutput != renderpassoutput::Output::Color || in.rawExportMask != 0)
        return fail (error, "Path Trace renders the Color image only; choose Raster for other outputs and data exports");
    if (in.passComposite || in.passProgram)
        return fail (error, "Path Trace does not run render pass composites yet; choose Raster");
    if (in.compositionFrame || in.localSceneFrame || (in.canonicalBlockCFrame && ! in.noteInstanceMapping))
        return fail (error, "Path Trace does not render deformed or composed animation frames yet; choose Raster");
    return true;
}

bool noteInstancedScene (const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
                         const arbitgpu::NativeFixtureSceneRuntimeInputs& in,
                         std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& out, std::string& error)
{
    const auto batch = arbitgpu::prepareNativeNoteInstances (in);
    if (! batch.admitted)
        return fail (error, "Path Trace note instancing has no valid score frame or mapping");
    const std::array<float, 4> plain { 1.0f, 1.0f, 1.0f, 0.0f };
    if (batch.appearanceLow != plain || batch.appearanceHigh != plain)
        return fail (error, "Path Trace does not render note appearance yet; choose Raster");
    if (scene->objectCount != 1 || scene->instancedObject.isValid())
        return fail (error, "Path Trace renders note instancing on one mesh object only; choose Raster");
    auto instanced = std::make_shared<HarmonicMIDI::grid::Visual3DScene> (*scene);
    instanced->instancedObject = instanced->objects[0].id;
    instanced->objectInstances.resize (batch.count);
    instanced->objectInstanceCount = batch.count;
    instanced->previousObjectInstanceCount = 0;
    // As the raster: objectMatrix * (position * meshScale + note position).
    for (std::size_t index = 0; index < batch.count; ++index)
    {
        const auto& note = batch.transforms[index];
        SceneTransform3D transform;
        transform.translation = { note[0], note[1], note[2] };
        transform.scale = { note[3], note[3], note[3] };
        instanced->objectInstances[index] = transform;
    }
    out = std::move (instanced);
    return true;
}

// The raster's material for one object drawn with a Surface program.
bool mapSurfaceProgram (const Program& program, const SceneMaterialRecord* record, float timeSeconds,
                        pt::Material& material, std::string& error)
{
    if (program.kind != arbitgpu::NativeFixtureMaterialKind::SurfacePbr)
        return fail (error, "Path Trace renders Surface materials only; choose Raster for diffraction materials");
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
                                                 in.cameraVerticalFovDegrees);
    if (! arbitgpu::validFixtureCameraOverride (view.camera))
        return fail (error, "Path Trace camera is invalid");
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
    if (draw.diffractionMaterial != nullptr)
    {
        if (HarmonicMIDI::grid::visual3d_detail::findById (scene.objects, scene.objectCount,
                                                           draw.diffractionObject) == nullptr)
            return fail (error, "Path Trace diffraction material names no scene object");
        setup.objectGratings.push_back ({ draw.diffractionObject.value, draw.diffractionMaterial, draw.diffractionField });
    }

    if (! bindFrameLayers (draw, setup, frames, error))
        return false;

    // Materials: objects bound to a Surface program use its block. The
    // diffraction object scatters by its grating instead.
    setup.objectMaterials.clear();
    if (draw.material != nullptr)
    {
        if (hasVertexProgram (*draw.material))
            return fail (error, "Path Trace does not run vertex programs yet; choose Raster");
        for (std::size_t index = 0; index < scene.objectCount; ++index)
        {
            const auto& object = scene.objects[index];
            // A mask shows the object's own material through the foil.
            bool maskedFoil = false;
            if (draw.diffractionField.has_value())
                for (const auto& corner : *draw.diffractionField)
                    maskedFoil = maskedFoil || corner[3] < 1.0f;
            if (draw.diffractionMaterial != nullptr && object.id == draw.diffractionObject && ! maskedFoil) continue;
            const auto* program = arbitgpu::nativeSurfaceForObject (draw.material.get(), object.id);
            if (program == nullptr) continue;
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

class PathTraceFixtureEngine final : public SceneRenderEngine
{
public:
    explicit PathTraceFixtureEngine (pathtracefixture::CpuFrameUpload upload) : upload_ (upload)
    {
        engine_.samplesPerPass = 1;
        engine_.environmentImageSource = [] (const HarmonicMIDI::grid::SceneEnvironmentImage& image) {
            return pt::residentEnvironmentPixels (image);
        };
    }

    bool render (const SceneEngineDraw& draw, RenderedFrame& output, std::string& error) override try
    {
        // A named device that does not load stops the frame by name. CPU and OptiX
        // are the devices with a render path.
        const auto& selection = pathTraceDeviceSelection();
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
#if defined(ARBIT_HAVE_OPTIX_KERNEL)
        if (selection.device == pt::device::DeviceKind::OptiX && ! devicePassSet_ && ! fellBackToCpu_)
        {
            engine_.setDevicePass (pt::device::makeOptixDevicePass ([this] { return engine_.sceneBuilds(); }),
                                   [] { return pt::device::optixPipelineReady(); });
            devicePassSet_ = true;
        }
        else
#endif
        if (selection.device != pt::device::DeviceKind::Cpu && selection.device != pt::device::DeviceKind::OptiX)
            return fail (error, std::string ("Path Trace cannot launch ") + pt::device::deviceName (selection.device) + " yet");
        if (! fellBackToCpu_ && deviceReason_ == nullptr)
            if (auto reason = pt::device::autoPassedOverReason (selection); ! reason.empty())
                deviceReason_ = std::make_shared<const std::string> (std::move (reason));
        if (draw.snapshot == nullptr || draw.inputs == nullptr)
            return fail (error, "Path Trace draw is incomplete");
        if (! pathtracefixture::admitInputs (*draw.inputs, error))
            return false;
        // The engine reuses its scene while the setup holds the same snapshot, so a
        // frame with the same notes keeps the instanced copy and its accumulation.
        auto drawn = draw;
        if (draw.inputs->noteInstanceMapping)
        {
            const auto batch = arbitgpu::prepareNativeNoteInstances (*draw.inputs);
            if (noteScene_ == nullptr || noteSceneSource_ != draw.snapshot || noteSceneCount_ != batch.count
                || ! std::equal (batch.transforms.begin(), batch.transforms.begin() + batch.count, noteSceneTransforms_.begin()))
            {
                if (! pathtracefixture::noteInstancedScene (draw.snapshot, *draw.inputs, noteScene_, error))
                    return false;
                noteSceneSource_ = draw.snapshot;
                noteSceneCount_ = batch.count;
                noteSceneTransforms_ = batch.transforms;
            }
            drawn.snapshot = noteScene_;
        }
        pt::EngineView view;
        if (! buildSetup (drawn, setup_, view, frames_, error))
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
        }
        engine_.settings().maxDepth = draw.maxBounces;
        engine_.setDeviceSceneFallback (std::strcmp (selection.requested, "Auto") == 0);
        const auto renderOnDevice = [&] {
            return preview ? engine_.renderPreview (setup_, view, color, image, error)
                           : engine_.renderExport (setup_, view, color, draw.exportSamples, image, error);
        };
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
            deviceReason_ = std::make_shared<const std::string> (pt::device::deviceFailureReason (error, pt::device::deviceLost()));
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
        // Capped preview: nothing new to resolve, so upload the last image again.
        // A fresh frame per call keeps the raster's one-frame-per-render contract.
        const bool resolvedNewImage = ! image.unchanged;
        if (resolvedNewImage)
            last_ = std::move (image);
        else if (last_.width != view.width || last_.height != view.height || last_.color != color)
            return fail (error, "Path Trace lost its resolved preview image");
        // The denoiser reads last_ and writes denoised_, so accumulation never
        // sees a filtered image and a switched-off frame uploads last_ untouched.
        if (! draw.denoise)
        {
            denoised_.clear();
            denoisedLayers_.clear();
        }
        else if ((resolvedNewImage || denoised_.empty()) && ! denoiseLastImage (error))
            return false;
        reportDenoiser (draw.denoise ? pathtracedenoise::kBuildDenoiser : pathtracedenoise::Denoiser::None);
        stats_.denoiser = pathtracedenoise::denoiserName (denoiserRan_);
        arbitgpu::CpuSceneImage upload;
        upload.width = last_.width;
        upload.height = last_.height;
        upload.linearColor = last_.color == pt::EngineColor::LinearRgba16f;
        upload.color = draw.denoise ? &denoised_ : &last_.colorBytes;
        upload.depth = &last_.depth;
        if (last_.frameLayers.size() != frames_.size())
            return fail (error, "Path Trace resolved another number of frame layers than the draw has frames");
        for (std::size_t index = 0; index < frames_.size(); ++index)
            upload.frameLayers.push_back ({ frames_[index],
                draw.denoise ? &denoisedLayers_[index] : &last_.frameLayers[index].lightBytes,
                &last_.frameLayers[index].uvCoverage });
        const float* background = upload.linearColor ? pt::kLinearRasterBackground : pt::kRasterBackground;
        std::copy (background, background + 3, upload.background);
        auto frame = upload_ (draw.backend, upload, error);
        if (frame == nullptr)
            return false;
        RenderedFrame result;
        result.use = draw.use;
        result.sceneId = draw.snapshot->id;
        result.dimensions = draw.dimensions;
        result.nativeFrame = std::move (frame);
        result.stats.textureUploadCount = 2 + 2 * static_cast<std::uint32_t> (frames_.size());
        result.stats.textureBytes = static_cast<std::uint64_t> (last_.colorBytes.size())
            + static_cast<std::uint64_t> (last_.depth.size()) * sizeof (float);
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
    const char* denoiserRan() const noexcept override { return pathtracedenoise::denoiserName (denoiserRan_); }

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
        stats_.device = pt::device::deviceName (fellBackToCpu_ || sceneOnCpu_ ? pt::device::DeviceKind::Cpu : pathTraceDeviceSelection().device);
        stats_.deviceRequested = pathTraceDeviceSelection().requested;
        stats_.deviceReason = deviceReason_ != nullptr ? deviceReason_ : sceneOnCpu_ ? sceneReason_ : nullptr;
        stats_.temporalDenoise = image.temporalDenoise;
        const bool preview = draw.use == RenderUse::Preview;
        const bool playback = preview && image.mode == pt::EngineRenderMode::PlaybackFixed;
        stats_.renderActivity = ! preview ? "export"
            : playback ? "playback-fixed"
            : engine_.previewProgressing() ? "converging" : "settled";
        stats_.sampleTarget = ! preview ? draw.exportSamples
            : playback ? std::min (std::max (1u, engine_.playbackSamples), engine_.maximumPreviewSamples)
            : engine_.previewTarget();
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
            statsValid_ = true;
        }
    }

    PathTraceFrameStats stats_;
    bool statsValid_ = false;
    bool denoiseLastImage (std::string& error)
    {
        if (pathtracedenoise::kBuildDenoiser != pathtracedenoise::Denoiser::NativeFilter)
            return fail (error, std::string ("Path Trace denoiser ")
                                    + pathtracedenoise::denoiserName (pathtracedenoise::kBuildDenoiser)
                                    + " is not implemented in this build");
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

    // One log line per change, so the run's log names the denoiser without a
    // line per frame.
    void reportDenoiser (pathtracedenoise::Denoiser denoiser)
    {
        if (denoiser == denoiserRan_) return;
        denoiserRan_ = denoiser;
        std::fprintf (stderr, "[pt] denoiser=%s\n", pathtracedenoise::denoiserName (denoiser));
    }

    pathtracefixture::CpuFrameUpload upload_;
    pt::SceneEngine engine_;
    bool devicePassSet_ = false;
    bool fellBackToCpu_ = false;
    std::shared_ptr<const std::string> deviceReason_;
    bool sceneOnCpu_ = false;
    std::shared_ptr<const std::string> sceneReason_;
    pt::EngineSceneSetup setup_;  // reused so per-frame vectors keep their storage
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> noteScene_, noteSceneSource_;
    std::size_t noteSceneCount_ = 0;
    std::array<std::array<float, 4>, visualnoteinstancing::kMaximumInstances> noteSceneTransforms_ {};
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
    PathTraceEngineRegistration() noexcept { setPathTraceSceneEngineFactory (&pathtracefixture::makeEngine); }
};

const PathTraceEngineRegistration registration;
} // namespace
namespace pathtracefixture
{
void prepareDeviceAtStartup()
{
#if defined(ARBIT_HAVE_OPTIX_KERNEL)
    const auto& selection = pathTraceDeviceSelection();
    if (selection.ok && selection.device == pt::device::DeviceKind::OptiX)
        pt::device::startOptixPipelineBuild();
#endif
}

bool cancelDeviceBuild()
{
#if defined(ARBIT_HAVE_OPTIX_KERNEL)
    return pt::device::stopOptixPipelineBuildByUser();
#else
    return false;
#endif
}

void startDeviceBuildAgain()
{
#if defined(ARBIT_HAVE_OPTIX_KERNEL)
    pt::device::startOptixPipelineBuildByUser();
#endif
}

void stopDeviceBuildAtExit()
{
#if defined(ARBIT_HAVE_OPTIX_KERNEL)
    pt::device::cancelOptixPipelineBuild();
#endif
}
} // namespace pathtracefixture
} // namespace videorender::fixture3d
