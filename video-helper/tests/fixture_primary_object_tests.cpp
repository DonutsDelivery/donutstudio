// Primary-object selection for non-diffraction fixture renders. A GLB import
// places a transform-only root before the meshes it parents. The root has no
// vertices or indices and references material 0, so objects[0] is not the
// object that draws.
#include "gpu_backend/fixture_primary_object.h"

#include <cstdio>
#include <memory>

namespace
{
using HarmonicMIDI::grid::Scene3DId;
using HarmonicMIDI::grid::SceneCameraId;
using HarmonicMIDI::grid::SceneMaterialId;
using HarmonicMIDI::grid::SceneObjectId;
using HarmonicMIDI::grid::Visual3DScene;
using HarmonicMIDI::grid::Visual3DSceneValidationCode;
using HarmonicMIDI::grid::validateVisual3DScene;

int failures = 0;

void expect (bool condition, const char* message)
{
    if (condition)
        return;
    ++failures;
    std::fprintf (stderr, "FAIL: %s\n", message);
}

// objects[0] is the transform-only root with id 1. objects[1] is one
// triangle with id 2 and material 1, parented to the root.
std::unique_ptr<Visual3DScene> retainedRootScene ()
{
    auto scene = std::make_unique<Visual3DScene> ();
    scene->id = Scene3DId { 1 };
    scene->cameras[0].id = SceneCameraId { 1 };
    scene->cameraCount = 1;
    scene->activeCamera = SceneCameraId { 1 };
    scene->materials[0].id = SceneMaterialId { 1 };
    scene->materialCount = 1;
    scene->vertices[0].normal = { 0.0f, 0.0f, 1.0f };
    scene->vertices[1].normal = { 0.0f, 0.0f, 1.0f };
    scene->vertices[1].position = { 1.0f, 0.0f, 0.0f };
    scene->vertices[2].normal = { 0.0f, 0.0f, 1.0f };
    scene->vertices[2].position = { 0.0f, 1.0f, 0.0f };
    scene->vertexCount = 3;
    scene->indices[0] = 0;
    scene->indices[1] = 1;
    scene->indices[2] = 2;
    scene->indexCount = 3;
    scene->objects[0].id = SceneObjectId { 1 };
    scene->objects[1].id = SceneObjectId { 2 };
    scene->objects[1].parent = SceneObjectId { 1 };
    scene->objects[1].material = SceneMaterialId { 1 };
    scene->objects[1].vertexCount = 3;
    scene->objects[1].indexCount = 3;
    scene->objectCount = 2;
    return scene;
}

void validatesRetainedRoot ()
{
    const auto scene = retainedRootScene ();
    expect (validateVisual3DScene (*scene).valid (),
            "a transform-only root ahead of its drawn child passes validation");
}

void selectsFirstDrawableObject ()
{
    const auto scene = retainedRootScene ();
    const auto* primaryObject = arbitgpu::fixtureprimary::firstDrawableObject (*scene);
    expect (primaryObject == &scene->objects[1],
            "the first drawable object is the triangle, not the transform-only root");

    const auto* primaryMaterial = arbitgpu::fixtureprimary::materialOf (*scene, primaryObject);
    expect (primaryMaterial == &scene->materials[0],
            "the triangle's material 1 resolves to the scene material record");
    expect (arbitgpu::fixtureprimary::materialOf (*scene, &scene->objects[0]) == nullptr,
            "the transform-only root has no resolvable material");
    expect (arbitgpu::fixtureprimary::materialOf (*scene, nullptr) == nullptr,
            "a null object has no material");
}

void refusesMissingPrimaryData ()
{
    auto scene = retainedRootScene ();
    scene->objects[1].material = SceneMaterialId { 99 };
    expect (validateVisual3DScene (*scene).code
                == Visual3DSceneValidationCode::MissingObjectMaterial,
            "the validator refuses a drawn object whose material does not resolve");
    const auto* primaryObject = arbitgpu::fixtureprimary::firstDrawableObject (*scene);
    expect (primaryObject == &scene->objects[1],
            "an unresolved material does not move the primary to another object");
    expect (arbitgpu::fixtureprimary::materialOf (*scene, primaryObject) == nullptr,
            "an unresolved material yields null, which the render refuses");

    scene = retainedRootScene ();
    scene->objectCount = 1;
    expect (arbitgpu::fixtureprimary::firstDrawableObject (*scene) == nullptr,
            "a scene whose only object is a transform-only root has no primary");
    scene->objectCount = 0;
    expect (arbitgpu::fixtureprimary::firstDrawableObject (*scene) == nullptr,
            "an empty scene has no primary");
}
} // namespace

int main ()
{
    validatesRetainedRoot ();
    selectsFirstDrawableObject ();
    refusesMissingPrimaryData ();
    if (failures != 0)
    {
        std::fprintf (stderr, "%d fixture primary-object check(s) failed\n", failures);
        return 1;
    }
    std::puts ("fixture primary-object checks passed");
    return 0;
}
