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
// Clearcoat is not traced yet. The dielectrics this maps onto
// (pathtrace/bsdf.h) follow Blender Cycles
// intern/cycles/kernel/closure/bsdf_microfacet.h and bsdf_util.h at
// 3e451b5580bb54df6029ecfa5d0f3958e32d1000 as a behavioural reference only;
// no code is taken from it.

#include "pathtrace_fixture_engine.h"
#include "gpu_backend/cpu_scene_frame.h"
#include "pathtrace/engine.h"
#include "pathtrace/environment_resident.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <memory>
#include <new>
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
    if (in.glassCaustic || ! in.diffractionParameters.empty() || in.diffractionSourceProgram
        || in.diffractionEvaluatedProgram)
        return fail (error, "Path Trace does not render diffraction or glass caustic passes yet; choose Raster");
    if (in.noteInstanceMapping)
        return fail (error, "Path Trace does not render note instancing yet; choose Raster");
    if (in.compositionFrame || in.localSceneFrame || in.canonicalBlockCFrame)
        return fail (error, "Path Trace does not render deformed or composed animation frames yet; choose Raster");
    if (in.materialFrameTexture || ! in.materialFrameTextures.empty())
        return fail (error, "Path Trace does not sample frame textures in materials yet; choose Raster");
    return true;
}

// The raster's material for one object drawn with a Surface program.
bool mapSurfaceProgram (const Program& program, const SceneMaterialRecord* record, float timeSeconds,
                        pt::Material& material, std::string& error)
{
    if (program.kind != arbitgpu::NativeFixtureMaterialKind::SurfacePbr)
        return fail (error, "Path Trace renders Surface materials only; choose Raster for diffraction materials");
    if (program.baseColorSource == Program::BaseColorSource::GraphFrameSrgbTexture)
        return fail (error, "Path Trace does not sample frame textures in materials yet; choose Raster");
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

bool buildSetup (const SceneEngineDraw& draw, pt::EngineSceneSetup& setup, pt::EngineView& view,
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
    setup.emissionGain = in.emissionGain;

    // Materials: objects bound to a Surface program use its block.
    setup.objectMaterials.clear();
    if (draw.material != nullptr)
    {
        if (hasVertexProgram (*draw.material))
            return fail (error, "Path Trace does not run vertex programs yet; choose Raster");
        for (std::size_t index = 0; index < scene.objectCount; ++index)
        {
            const auto& object = scene.objects[index];
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
    PathTraceFixtureEngine()
    {
        engine_.samplesPerPass = 1;
        engine_.environmentImageSource = [] (const HarmonicMIDI::grid::SceneEnvironmentImage& image) {
            return pt::residentEnvironmentPixels (image);
        };
    }

    bool render (const SceneEngineDraw& draw, RenderedFrame& output, std::string& error) override
    {
        if (draw.snapshot == nullptr || draw.inputs == nullptr)
            return fail (error, "Path Trace draw is incomplete");
        if (! pathtracefixture::admitInputs (*draw.inputs, error))
            return false;
        pt::EngineView view;
        if (! buildSetup (draw, setup_, view, error))
            return false;
        const auto color = draw.inputs->linearColor ? pt::EngineColor::LinearRgba16f : pt::EngineColor::DisplayRgba8;
        pt::EngineImage image;
        const bool rendered = draw.use == RenderUse::Preview
            ? engine_.renderPreview (setup_, view, color, image, error)
            : engine_.renderExport (setup_, view, color, draw.exportSamples, image, error);
        if (! rendered)
            return false;
        // Capped preview: nothing new to resolve, so upload the last image again.
        // A fresh frame per call keeps the raster's one-frame-per-render contract.
        if (! image.unchanged)
            last_ = std::move (image);
        else if (last_.width != view.width || last_.height != view.height || last_.color != color)
            return fail (error, "Path Trace lost its resolved preview image");
        arbitgpu::CpuSceneImage upload;
        upload.width = last_.width;
        upload.height = last_.height;
        upload.linearColor = last_.color == pt::EngineColor::LinearRgba16f;
        upload.color = &last_.colorBytes;
        upload.depth = &last_.depth;
        auto frame = arbitgpu::uploadCpuSceneFrame (draw.backend, upload, error);
        if (frame == nullptr)
            return false;
        RenderedFrame result;
        result.use = draw.use;
        result.sceneId = draw.snapshot->id;
        result.dimensions = draw.dimensions;
        result.nativeFrame = std::move (frame);
        result.stats.textureUploadCount = 2;
        result.stats.textureBytes = static_cast<std::uint64_t> (last_.colorBytes.size())
            + static_cast<std::uint64_t> (last_.depth.size()) * sizeof (float);
        output = std::move (result);
        error.clear();
        return true;
    }

    void cancel() noexcept override { engine_.cancel(); }

private:
    pt::SceneEngine engine_;
    pt::EngineSceneSetup setup_;  // reused so per-frame vectors keep their storage
    pt::EngineImage last_;
};

} // namespace

std::shared_ptr<SceneRenderEngine> pathtracefixture::makeEngine()
{
    try
    {
        return std::make_shared<PathTraceFixtureEngine>();
    }
    catch (const std::exception&)
    {
        return nullptr;
    }
}

namespace
{
struct PathTraceEngineRegistration final
{
    PathTraceEngineRegistration() noexcept { setPathTraceSceneEngineFactory (&pathtracefixture::makeEngine); }
};

const PathTraceEngineRegistration registration;
} // namespace
} // namespace videorender::fixture3d
