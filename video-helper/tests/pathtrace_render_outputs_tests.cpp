// Render outputs of Path Trace (Plan 3, Compositing): the motion math, the
// first-hit outputs the engine writes next to the Image, and the wire contract
// that now keeps an Image view and data exports on Path Trace. The engine runs
// one sample per pixel on a 16 x 16 image of the glass starter scene.

#include "pathtrace/engine.h"
#include "pathtrace/render_outputs.h"
#include "support/pathtrace_glass_scene.h"
#include "../../shared/SceneAovOperationContract.h"
#include "../../shared/VisualImportedSceneRenderOperationContract.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <memory>
#include <optional>
#include <string>

namespace
{

namespace pt = videohelper::pathtrace;
namespace grid = HarmonicMIDI::grid;
namespace render = visualimportedscenerender;
namespace glass = pathtraceglassscene;

int checks = 0;
int failures = 0;

void check (bool condition, const std::string& message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

bool near (float a, float b, float tolerance = 1.0e-4f)
{
    return std::abs (a - b) <= tolerance;
}

float halfValue (std::uint16_t bits)
{
    const int exponent = (bits >> 10) & 31;
    const float mantissa = static_cast<float> (bits & 1023);
    const float magnitude = exponent == 0 ? std::ldexp (mantissa, -24)
                                          : std::ldexp (1.0f + mantissa / 1024.0f, exponent - 15);
    return (bits & 0x8000u) != 0 ? -magnitude : magnitude;
}

pt::Mat4 trs (grid::SceneVec3 translation, grid::SceneQuaternion rotation, grid::SceneVec3 scale)
{
    grid::SceneTransform3D transform;
    transform.translation = translation;
    transform.rotation = rotation;
    transform.scale = scale;
    return pt::transformMatrix (transform);
}

void inverseTests()
{
    const pt::Mat4 m = trs ({ 1.0f, -2.0f, 3.0f }, glass::axisAngle (0.3f, 1.0f, 0.2f, 0.7f), { 2.0f, 0.5f, 1.5f });
    const pt::Mat4 product = m * pt::affineInverse (m);
    bool identity = true;
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row)
            identity = identity && near (product.m[static_cast<std::size_t> (column * 4 + row)], row == column ? 1.0f : 0.0f);
    check (identity, "an affine matrix times its inverse is the identity");
    const pt::Vec3 p { 0.4f, -1.2f, 2.5f };
    const pt::Vec3 back = pt::transformPoint (pt::affineInverse (m), pt::transformPoint (m, p));
    check (near (back.x, p.x) && near (back.y, p.y) && near (back.z, p.z), "the inverse carries a point back");

    pt::Mat4 flat = trs ({ 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 1.0f });
    const pt::Mat4 none = pt::affineInverse (flat);
    check (none.m == pt::Mat4 {}.m, "a collapsed matrix inverts to the identity, so it reads as not moving");
}

void motionMathTests()
{
    pt::PinholeCamera camera;
    camera.width = 100;
    camera.height = 50;
    camera.aspect = 2.0f;
    camera.tanHalfFov = 1.0f;
    const pt::Vec3 point { 0.0f, 0.0f, -5.0f };
    const pt::Mat4 current = trs ({ 0.0f, 0.0f, -5.0f }, { 0.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f });
    const auto still = pt::pixelMotion (camera, point, current, current);
    check (still[0] == 0.0f && still[1] == 0.0f, "an object at its previous transform has zero motion");

    // A frame ago the object sat 1 unit left and 1 unit lower: it moved right and up.
    const pt::Mat4 previous = trs ({ -1.0f, -1.0f, -5.0f }, { 0.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f });
    const auto moved = pt::pixelMotion (camera, point, current, previous);
    // At depth 5 one unit spans 100 / (2 * 1 * 2 * 5) = 5 pixels across and 50 / (2 * 5) = 5 pixels up.
    check (near (moved[0], 5.0f, 1.0e-3f) && near (moved[1], 5.0f, 1.0e-3f),
           "motion is current minus previous pixel position with Y up, as the raster's RG16F");

    const pt::Mat4 behind = trs ({ 0.0f, 0.0f, 10.0f }, { 0.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f });
    const auto hidden = pt::pixelMotion (camera, point, current, behind);
    check (hidden[0] == 0.0f && hidden[1] == 0.0f, "a previous position behind the eye gives zero motion");

    // A frame ago the camera sat 1 unit left: the still point slid left on screen.
    pt::PinholeCamera previousCamera = camera;
    previousCamera.position.x = -1.0f;
    const auto panned = pt::pixelMotion (camera, previousCamera, point, current, current);
    check (near (panned[0], -5.0f, 1.0e-3f) && near (panned[1], 0.0f, 1.0e-3f),
           "a camera move gives a still object motion through the previous camera");
    const auto both = pt::pixelMotion (camera, previousCamera, point, current, previous);
    check (near (both[0], 0.0f, 1.0e-3f) && near (both[1], 5.0f, 1.0e-3f),
           "camera and object moves combine: following the object cancels its sideways motion");
}

void previousWorldTests()
{
    auto owned = std::make_shared<grid::Visual3DScene>();
    grid::Visual3DScene& scene = *owned;
    scene.objects.resize (2);
    scene.objectCount = 2;
    scene.objects[0].id.value = 1;
    scene.objects[0].transform.translation = { 2.0f, 0.0f, 0.0f };
    scene.objects[1].id.value = 2;
    scene.objects[1].parent.value = 1;
    scene.objects[1].transform.translation = { 0.0f, 1.0f, 0.0f };
    const pt::Mat4 world = pt::objectWorldMatrix (scene, 1);
    check (pt::previousObjectWorld (scene, 1, world).m == world.m, "a scene without history keeps the current world");
    check (! pt::sceneHasMotionHistory (scene), "a scene without previous transforms has no motion history");

    // The parent was 1 unit further left a frame ago; the child kept its local offset.
    scene.previousObjectTransforms.resize (2);
    scene.previousObjectTransformCount = 2;
    scene.previousObjectTransforms[0].translation = { 1.0f, 0.0f, 0.0f };
    scene.previousObjectTransforms[1] = scene.objects[1].transform;
    const pt::Vec3 origin = pt::transformPoint (pt::previousObjectWorld (scene, 1, world), { 0.0f, 0.0f, 0.0f });
    check (pt::sceneHasMotionHistory (scene) && near (origin.x, 1.0f) && near (origin.y, 1.0f),
           "the previous world walks the parent chain through the previous transforms");

    // A runtime override shifts the drawn world; the shift carries to the previous one.
    pt::Mat4 shifted = world;
    shifted.m[14] += 3.0f;
    const pt::Vec3 shiftedOrigin = pt::transformPoint (pt::previousObjectWorld (scene, 1, shifted), { 0.0f, 0.0f, 0.0f });
    check (near (shiftedOrigin.x, 1.0f) && near (shiftedOrigin.z, 3.0f),
           "a runtime offset of the drawn world applies to the previous world too");
}

struct Rendered
{
    pt::EngineImage image;
    bool ok = false;
};

Rendered renderGlass (const std::shared_ptr<const grid::Visual3DScene>& scene,
                       std::optional<grid::SceneCameraRecord> previousCamera = std::nullopt)
{
    Rendered result;
    pt::SceneEngine engine (1);
    std::string error;
    auto view = glass::makeView (*scene, 16, 16);
    view.previousCamera = previousCamera;
    result.ok = engine.renderExport (glass::makeSetup (scene), view,
                                     pt::EngineColor::LinearRgba16f, 1, result.image, error);
    if (! result.ok)
        std::cerr << "render failed: " << error << '\n';
    return result;
}

void engineOutputTests()
{
    const auto scene = glass::makeScene();
    const Rendered still = renderGlass (scene);
    check (still.ok, "the glass scene renders");
    const std::size_t pixels = 16u * 16u;
    const auto& image = still.image;
    check (image.normalPass.size() == pixels * 4 && image.emissionPass.size() == pixels * 4
               && image.maskPass.size() == pixels && image.materialIdPass.size() == pixels
               && image.objectIdPass.size() == pixels && image.motionPass.size() == pixels * 2,
           "every render output has one value per pixel in its raw pass format");

    bool sawGlass = false, sawFloor = false, idsMatch = true, normalsUnit = true, stillMotion = true, maskMatchesDepth = true;
    for (std::size_t i = 0; i < pixels; ++i)
    {
        const bool covered = image.maskPass[i] == 255;
        maskMatchesDepth = maskMatchesDepth && (covered ? image.depth[i] < 1.0f : image.maskPass[i] == 0);
        if (! covered)
            continue;
        const std::uint32_t object = image.objectIdPass[i], material = image.materialIdPass[i];
        sawGlass = sawGlass || object == glass::kGlassObject;
        sawFloor = sawFloor || object == glass::kFloorObject;
        if (object == glass::kGlassObject) idsMatch = idsMatch && material == glass::kGlassMaterialId;
        if (object == glass::kFloorObject) idsMatch = idsMatch && material == 1u;
        const float nx = halfValue (image.normalPass[i * 4]), ny = halfValue (image.normalPass[i * 4 + 1]),
                    nz = halfValue (image.normalPass[i * 4 + 2]);
        normalsUnit = normalsUnit && near (nx * nx + ny * ny + nz * nz, 1.0f, 1.0e-2f);
        stillMotion = stillMotion && image.motionPass[i * 2] == 0 && image.motionPass[i * 2 + 1] == 0;
    }
    check (sawGlass && sawFloor, "the image shows the glass cube and the floor");
    check (maskMatchesDepth, "the mask covers exactly the pixels with a depth");
    check (idsMatch, "each covered pixel carries its object's id and material id");
    check (normalsUnit, "each covered pixel carries a unit normal");
    check (stillMotion, "a scene without previous transforms has zero motion everywhere");

    // The cube was 0.2 units left a frame ago; every other object stayed.
    auto moving = std::make_shared<grid::Visual3DScene> (*scene);
    moving->previousObjectTransforms.resize (moving->objectCount);
    moving->previousObjectTransformCount = moving->objectCount;
    for (std::size_t i = 0; i < moving->objectCount; ++i)
        moving->previousObjectTransforms[i] = moving->objects[i].transform;
    moving->previousObjectTransforms[3].translation.x -= 0.2f;
    const Rendered moved = renderGlass (moving);
    bool glassMovedRight = true, floorStill = true, anyGlass = false;
    for (std::size_t i = 0; i < pixels && moved.ok; ++i)
    {
        if (moved.image.maskPass[i] != 255)
            continue;
        const float dx = halfValue (moved.image.motionPass[i * 2]);
        if (moved.image.objectIdPass[i] == glass::kGlassObject)
        {
            anyGlass = true;
            glassMovedRight = glassMovedRight && dx > 0.0f;
        }
        else
            floorStill = floorStill && moved.image.motionPass[i * 2] == 0 && moved.image.motionPass[i * 2 + 1] == 0;
    }
    check (moved.ok && anyGlass && glassMovedRight, "the moved cube's pixels move right");
    check (floorStill, "objects at their previous transforms keep zero motion");

    // The camera was 0.2 units to its own left a frame ago: every covered pixel
    // slides left, and the sky stays at zero.
    grid::SceneCameraRecord previousCamera = scene->cameras[0];
    const pt::Mat4 cameraWorld = pt::transformMatrix (previousCamera.transform);
    previousCamera.transform.translation.x -= 0.2f * cameraWorld.m[0];
    previousCamera.transform.translation.y -= 0.2f * cameraWorld.m[1];
    previousCamera.transform.translation.z -= 0.2f * cameraWorld.m[2];
    const Rendered panned = renderGlass (scene, previousCamera);
    bool pannedLeft = true, skyStill = true, anyCovered = false;
    for (std::size_t i = 0; i < pixels && panned.ok; ++i)
    {
        if (panned.image.maskPass[i] != 255)
        {
            skyStill = skyStill && panned.image.motionPass[i * 2] == 0 && panned.image.motionPass[i * 2 + 1] == 0;
            continue;
        }
        anyCovered = true;
        pannedLeft = pannedLeft && halfValue (panned.image.motionPass[i * 2]) < 0.0f;
    }
    check (panned.ok && anyCovered && pannedLeft, "a camera move to the right slides every covered pixel left");
    check (skyStill, "pixels with no hit keep zero motion when the camera moves");

    // From under the floor, looking up at its back face: the Normal output keeps
    // the floor's world normal (+Y) as the raster does, where the denoiser's
    // guide would turn it to face the camera (-Y).
    auto below = std::make_shared<grid::Visual3DScene> (*scene);
    below->cameras[0].transform.translation = { 0.0f, -1.0f, -1.0f };
    below->cameras[0].transform.rotation = glass::axisAngle (1.0f, 0.0f, 0.0f, 1.5707963f);
    const Rendered underneath = renderGlass (below);
    bool floorSeen = false, floorUp = true;
    for (std::size_t i = 0; i < pixels && underneath.ok; ++i)
        if (underneath.image.maskPass[i] == 255 && underneath.image.objectIdPass[i] == glass::kFloorObject)
        {
            floorSeen = true;
            floorUp = floorUp && halfValue (underneath.image.normalPass[i * 4 + 1]) > 0.99f;
        }
    check (underneath.ok && floorSeen && floorUp, "the Normal output is the unflipped world normal, not the guide's");
}

void previousCameraWireTests()
{
    const auto scene = glass::makeScene();
    const auto plain = sceneaov::detail::encodeScene (*scene);
    grid::Visual3DScene decoded;
    check (! plain.empty() && sceneaov::detail::decodeScene (plain, decoded) && ! decoded.previousCamera.has_value(),
           "a scene without a previous camera reads back without one");

    grid::Visual3DScene moving = *scene;
    moving.previousCamera = scene->cameras[0];
    moving.previousCamera->transform.translation.x += 0.5f;
    moving.previousCamera->verticalFovRadians = 0.9f;
    const auto withCamera = sceneaov::detail::encodeScene (moving);
    check (withCamera.size() == plain.size() + 8 + sceneaov::detail::kPreviousCameraRecordBytes
               && std::equal (plain.begin(), plain.end(), withCamera.begin()),
           "the previous camera is a trailer, so scenes without one keep their exact bytes");
    grid::Visual3DScene round;
    check (sceneaov::detail::decodeScene (withCamera, round) && round.previousCamera.has_value()
               && round.previousCamera->transform.translation.x == moving.previousCamera->transform.translation.x
               && round.previousCamera->verticalFovRadians == 0.9f
               && round.previousCamera->farPlane == moving.previousCamera->farPlane,
           "the previous camera round-trips");
    auto truncated = withCamera;
    truncated.pop_back();
    grid::Visual3DScene refused;
    check (! sceneaov::detail::decodeScene (truncated, refused), "a truncated previous camera is refused");

    moving.previousCamera->verticalFovRadians = 0.0f;
    check (! grid::validateVisual3DScene (moving).valid(), "a previous camera with no field of view is invalid");
}

void contractTests()
{
    render::Request traced;
    traced.sourceStableId = 71;
    traced.renderStableId = 72;
    traced.asset.id = "model-asset-1";
    traced.asset.version = 3;
    traced.asset.contentSha256 = std::string (64, 'a');
    traced.asset.sourceMediaType = "model/gltf-binary";
    traced.asset.sourceByteSize = 4096;
    traced.renderEngine = render::RenderEngine::PathTrace;
    traced.pathTraceExportSamples = 16;
    traced.pathTracePreviewSamples = 16;
    for (const auto output : { renderpassoutput::Output::Depth, renderpassoutput::Output::Normal,
                               renderpassoutput::Output::Motion, renderpassoutput::Output::Emission,
                               renderpassoutput::Output::Mask, renderpassoutput::Output::MaterialId,
                               renderpassoutput::Output::ObjectId })
    {
        auto request = traced;
        request.imageOutput = output;
        request.rawExportMask = render3dimage::kRawExportMask;
        render::Request decoded;
        check (render::decode (render::encode (request), decoded) && decoded.imageOutput == output
                   && decoded.rawExportMask == render3dimage::kRawExportMask
                   && decoded.renderEngine == render::RenderEngine::PathTrace,
               "Path Trace keeps the " + std::string (renderpassoutput::token (output)) + " view and every data export");
    }
}

} // namespace

int main()
{
    inverseTests();
    motionMathTests();
    previousWorldTests();
    engineOutputTests();
    previousCameraWireTests();
    contractTests();
    std::cout << (checks - failures) << "/" << checks << " Path Trace render output checks passed\n";
    return failures == 0 ? 0 : 1;
}
