#include "VisualImportedSceneRenderOperationContract.h"

#include <cstdio>
#include <cstdlib>
#include <memory>

namespace render = visualimportedscenerender;
namespace geometry = videowire::geometry;
using namespace HarmonicMIDI::grid;

namespace
{
int failures = 0;

void check (bool condition, const char* what)
{
    if (condition) return;
    std::fprintf (stderr, "FAIL: %s\n", what);
    ++failures;
}

// One other object first, so the links' ranges start past the buffers' front.
std::shared_ptr<Visual3DScene> sceneWithPlaceholder()
{
    auto scene = std::make_shared<Visual3DScene>();
    scene->id = { 9 };
    scene->materials[0].id = { 5 };
    scene->materialCount = 1;
    for (std::uint32_t object = 0; object < 2; ++object)
    {
        for (std::uint32_t corner = 0; corner < 3; ++corner)
        {
            auto& vertex = scene->vertices[scene->vertexCount++];
            vertex.position = { static_cast<float> (corner), static_cast<float> (corner % 2), 0.0f };
            vertex.normal = { 0.0f, 0.0f, 1.0f };
            scene->indices[scene->indexCount++] = corner;
        }
        auto& record = scene->objects[scene->objectCount++];
        record.id = { 10 + object };
        record.material = { 5 };
        record.firstVertex = record.firstIndex = object * 3;
        record.vertexCount = record.indexCount = 3;
    }
    scene->cameras[0].id = { 3 };
    scene->cameraCount = 1;
    scene->activeCamera = { 3 };
    return scene;
}

geometry::HarmonicLinkSceneBinding binding()
{
    geometry::HarmonicLinkSceneBinding value;
    value.object = { 11 };
    value.mapping.stableId = 11;
    return value;
}

render::Request composedRequest()
{
    render::Request request;
    request.sourceStableId = 9;
    request.renderStableId = 20;
    request.structuralRevision = request.evaluationRevision = 1;
    request.sceneSnapshot = sceneWithPlaceholder();
    return request;
}

void wire()
{
    auto request = composedRequest();
    const auto without = render::encode (request);
    check (! without.empty(), "a composed scene without links encodes");
    check (without.find (render::kCompiledOperationHeaderV33) == std::string::npos,
           "a scene without links keeps its older bytes");

    request.harmonicLinks.push_back (binding());
    request.harmonicLinks[0].fitToView = true;
    request.harmonicLinks[0].mapping.shape = visualharmonicgeometry::LinkShape::curve;
    const auto with = render::encode (request);
    check (with.rfind (render::kCompiledOperationHeaderV33, 0) == 0, "links wrap the request in V33");
    render::Request decoded;
    check (render::decode (with, decoded), "the V33 wire decodes");
    check (decoded.harmonicLinks.size() == 1 && decoded.harmonicLinks[0].object.value == 11
               && decoded.harmonicLinks[0].fitToView
               && decoded.harmonicLinks[0].mapping.shape == visualharmonicgeometry::LinkShape::curve,
           "the binding survives the wire");
    check (render::encode (decoded) == with, "the V33 wire re-encodes to the same bytes");

    auto missing = composedRequest();
    missing.harmonicLinks.push_back (binding());
    missing.harmonicLinks[0].object = { 77 };
    check (! render::valid (missing), "a binding must name an object in the scene");

    auto unordered = composedRequest();
    unordered.harmonicLinks = { binding(), binding() };
    check (! render::valid (unordered), "one object takes one binding");

    auto noScene = composedRequest();
    noScene.sceneSnapshot.reset();
    noScene.harmonicLinks.push_back (binding());
    check (! render::valid (noScene), "links need a retained scene");
}

// The link count follows the score, so the mesh may be larger or smaller than
// the last frame's, and far larger than any contract budget.
void meshReplacement()
{
    auto scene = sceneWithPlaceholder();
    geometry::GeometryData mesh;
    const std::size_t quads = 5000;
    for (std::size_t quad = 0; quad < quads; ++quad)
    {
        const auto base = static_cast<std::uint32_t> (mesh.positions.size());
        const auto x = static_cast<float> (quad);
        mesh.positions.push_back ({ x, 0.0f, 0.0f });
        mesh.positions.push_back ({ x + 1.0f, 0.0f, 0.0f });
        mesh.positions.push_back ({ x + 1.0f, 1.0f, 0.0f });
        mesh.positions.push_back ({ x, 1.0f, 0.0f });
        for (const auto corner : { 0u, 1u, 2u, 0u, 2u, 3u }) mesh.indices.push_back (base + corner);
    }
    check (mesh.positions.size() > visualharmonicgeometry::kVertexBudget,
           "the fixture exceeds the raster route's vertex budget");
    std::string error;
    check (geometry::replaceHarmonicLinkSceneMesh (binding(), mesh, *scene, error),
           "a mesh past the raster budget enters the scene");
    const auto* object = visual3d_detail::findById (scene->objects, scene->objectCount, SceneObjectId { 11 });
    check (object != nullptr && object->vertexCount == mesh.positions.size()
               && object->indexCount == mesh.indices.size() && object->firstVertex == 6 && object->firstIndex == 6,
           "the object points at the appended mesh");
    check (validateVisual3DScene (*scene).valid(), "the scene with the appended mesh is valid");
    const auto* other = visual3d_detail::findById (scene->objects, scene->objectCount, SceneObjectId { 10 });
    check (other != nullptr && other->firstVertex == 0 && other->vertexCount == 3, "other objects keep their ranges");
    check (object != nullptr && scene->vertices[object->firstVertex].normal.z == 1.0f,
           "the appended mesh has normals");

    auto smaller = sceneWithPlaceholder();
    geometry::GeometryData triangle;
    triangle.positions = { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 } };
    triangle.indices = { 0, 1, 2 };
    check (geometry::replaceHarmonicLinkSceneMesh (binding(), triangle, *smaller, error)
               && validateVisual3DScene (*smaller).valid(), "a smaller mesh replaces the placeholder");

    triangle.indices = { 0, 1, 3 };
    check (! geometry::replaceHarmonicLinkSceneMesh (binding(), triangle, *smaller, error),
           "an out-of-range mesh index is refused");
}

// Sources with different ambient compose, whatever their order.
void composedAmbient()
{
    const auto composed = [] (SceneVec3 a, SceneVec3 b, Visual3DScene& destination)
    {
        auto first = sceneWithPlaceholder(), second = sceneWithPlaceholder();
        first->ambientColor = a;
        second->ambientColor = b;
        for (std::size_t object = 0; object < second->objectCount; ++object)
            second->objects[object].id.value += 100;
        std::string error;
        const auto composedAll = scenecomposition::appendScene (*first, destination, true, error)
            && scenecomposition::appendScene (*second, destination, false, error);
        if (! composedAll) std::fprintf (stderr, "compose: %s\n", error.c_str());
        return composedAll;
    };
    auto lit = std::make_shared<Visual3DScene>(), reversed = std::make_shared<Visual3DScene>();
    check (composed ({ 0.3f, 0.0f, 0.1f }, { 0.0f, 0.2f, 0.1f }, *lit)
               && composed ({ 0.0f, 0.2f, 0.1f }, { 0.3f, 0.0f, 0.1f }, *reversed),
           "sources with different ambient compose");
    check (lit->ambientColor.x == 0.3f && lit->ambientColor.y == 0.2f && lit->ambientColor.z == 0.1f,
           "each channel takes its brightest source");
    check (reversed->ambientColor.x == lit->ambientColor.x && reversed->ambientColor.y == lit->ambientColor.y
               && reversed->ambientColor.z == lit->ambientColor.z,
           "source order doesn't change the ambient");
}

void fit()
{
    std::vector<geometry::Vec3> positions { { 100, 0, 0 }, { 110, 4, 2 } };
    geometry::fitHarmonicLinksToView (positions, 1920, 1080);
    check (std::abs (positions[0].x + positions[1].x) < 0.001f, "the fit centres the links");
    const auto halfHeight = 10.0f * std::tan (0.7853981634f * 0.5f);
    check (std::abs (positions[1].y) <= halfHeight, "the fit keeps the links in the camera's view");
}
}

int main()
{
    wire();
    meshReplacement();
    composedAmbient();
    fit();
    if (failures == 0) std::puts ("harmonic link scene tests passed");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
