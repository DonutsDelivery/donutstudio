// Focus on object: a camera that names an object takes that object's view
// depth as its focus distance each frame, through the object's parents and as
// the camera turns. The scene wire and the imported-scene codec carry the
// reference, and cameras without one keep their bytes.

#include "pathtrace/scene_ingest.h"
#include "SceneAovOperationContract.h"
#include "VisualImportedSceneRenderOperationContract.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

namespace
{

namespace pt = videohelper::pathtrace;
namespace grid = HarmonicMIDI::grid;

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

bool close (float a, float b) { return std::fabs (a - b) <= 1.0e-4f; }

// Two objects: a parent at z = -2 turned half a circle about Y and scaled by
// two, and its child one unit along the parent's -Z, so at z = 0 in the world.
grid::Visual3DScene focusScene()
{
    grid::Visual3DScene scene;
    scene.id = grid::Scene3DId { 1 };
    scene.vertices[0] = { { -0.5f, -0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f } };
    scene.vertices[1] = { { 0.5f, -0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f } };
    scene.vertices[2] = { { 0.0f, 0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.5f, 0.0f } };
    scene.indices[0] = 0;
    scene.indices[1] = 1;
    scene.indices[2] = 2;
    scene.vertexCount = scene.indexCount = 3;
    scene.materials[0].id = grid::SceneMaterialId { 2 };
    scene.materialCount = 1;
    for (std::size_t i = 0; i < 2; ++i)
    {
        scene.objects[i].id = grid::SceneObjectId { static_cast<std::uint32_t> (10 + i) };
        scene.objects[i].material = scene.materials[0].id;
        scene.objects[i].vertexCount = scene.objects[i].indexCount = 3;
    }
    scene.objects[0].transform.translation = { 0.0f, 0.0f, -2.0f };
    scene.objects[0].transform.rotation = { 0.0f, 1.0f, 0.0f, 0.0f };
    scene.objects[0].transform.scale = { 2.0f, 2.0f, 2.0f };
    scene.objects[1].parent = scene.objects[0].id;
    scene.objects[1].transform.translation = { 0.0f, 0.0f, -1.0f };
    scene.objectCount = 2;
    scene.cameras[0].id = grid::SceneCameraId { 4 };
    scene.cameras[0].transform.translation = { 0.0f, 0.0f, 5.0f };
    scene.cameras[0].fStop = 2.0f;
    scene.cameras[0].focusDistance = 1.0f;
    scene.cameraCount = 1;
    scene.activeCamera = scene.cameras[0].id;
    return scene;
}

void resolveTests()
{
    auto scene = focusScene();
    const auto parentOrigin = grid::sceneObjectWorldOrigin (scene, 0);
    const auto childOrigin = grid::sceneObjectWorldOrigin (scene, 1);
    check (close (parentOrigin.z, -2.0f), "a root object's origin is its translation");
    check (close (childOrigin.x, 0.0f) && close (childOrigin.z, 0.0f), "a child's origin goes through its parent's turn and scale");

    auto camera = scene.cameras[0];
    grid::resolveSceneCameraFocus (scene, camera);
    check (close (camera.focusDistance, 1.0f), "a camera without a focus object keeps its distance");
    camera.focusObject = 1;
    grid::resolveSceneCameraFocus (scene, camera);
    check (close (camera.focusDistance, 7.0f), "focus on the parent is its view depth");
    camera.focusObject = 2;
    grid::resolveSceneCameraFocus (scene, camera);
    check (close (camera.focusDistance, 5.0f), "focus on the child follows its world position");

    // The object moves: the next frame's scene gives the next distance.
    scene.objects[0].transform.translation.z = -4.0f;
    grid::resolveSceneCameraFocus (scene, camera);
    check (close (camera.focusDistance, 7.0f), "a moving object moves the focus");

    // A quarter turn to the left leaves the object to the camera's right, level with the lens.
    camera.transform.rotation = { 0.0f, std::sqrt (0.5f), 0.0f, std::sqrt (0.5f) };
    grid::resolveSceneCameraFocus (scene, camera);
    check (close (camera.focusDistance, camera.nearPlane), "an object level with the lens focuses at the near plane");

    // Path Trace's active camera resolves the same way.
    scene.cameras[0].focusObject = 1;
    pt::PinholeCamera pinhole;
    check (pt::resolveActiveCamera (scene, 8, 8, pinhole) && close (pinhole.focusDistance, 9.0f),
           "the Path Trace camera focuses on the object");
}

void wireTests()
{
    const auto scene = focusScene();
    check (grid::validateVisual3DScene (scene).valid(), "the fixture scene validates");
    const auto lensBytes = sceneaov::detail::encodeScene (scene);
    auto focused = scene;
    focused.cameras[0].focusObject = 2;
    check (grid::validateVisual3DScene (focused).valid(), "a focus object validates");
    const auto focusBytes = sceneaov::detail::encodeScene (focused);
    check (focusBytes.size() == lensBytes.size() + 8 + sceneaov::detail::kCameraFocusObjectRecordBytes,
           "the focus object adds one trailer");
    grid::Visual3DScene decoded;
    check (sceneaov::detail::decodeScene (focusBytes, decoded) && decoded.cameras[0].focusObject == 2
               && sceneaov::detail::encodeScene (decoded) == focusBytes,
           "the focus object round-trips on the scene wire");
    grid::Visual3DScene decodedLens;
    check (sceneaov::detail::decodeScene (lensBytes, decodedLens) && decodedLens.cameras[0].focusObject == 0
               && sceneaov::detail::encodeScene (decodedLens) == lensBytes,
           "a camera without a focus object keeps its bytes");
    auto zero = focusBytes;
    std::fill (zero.end() - 4, zero.end(), std::uint8_t { 0 });
    check (! sceneaov::detail::decodeScene (zero, decoded), "an all-zero focus trailer is refused");

    const auto encodeCamera = [] (const grid::SceneCameraRecord& camera) {
        std::ostringstream stream;
        visualimportedscenerender::detail::encodeCamera (stream, camera);
        return stream.str();
    };
    const auto lensText = encodeCamera (scene.cameras[0]);
    const auto focusText = encodeCamera (focused.cameras[0]);
    check (std::count (lensText.begin(), lensText.end(), ' ') == 18, "a lens camera line is unchanged");
    check (focusText == lensText.substr (0, lensText.size() - 1) + " 2\n", "a focus object continues the line");
    std::istringstream stream (focusText + lensText);
    grid::SceneCameraRecord first, second;
    check (visualimportedscenerender::detail::decodeCamera (stream, first) && first.focusObject == 2
               && visualimportedscenerender::detail::decodeCamera (stream, second) && second.focusObject == 0,
           "imported-scene camera lines round-trip with and without a focus object");
}

} // namespace

int main()
{
    resolveTests();
    wireTests();
    std::cout << checks - failures << '/' << checks << " camera focus object checks passed\n";
    return failures == 0 ? 0 : 1;
}
