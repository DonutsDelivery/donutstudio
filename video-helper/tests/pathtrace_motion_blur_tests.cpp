// Path Trace motion blur: the shutter interval each position gives, vertex and
// camera motion over the shutter, a motion BVH whose bounds cover the sweep and
// whose hits follow ray time, ingest of open and close vertices from the
// previous frame, and the V38 wire that carries the Render node's shutter while
// requests without motion blur keep their bytes, and a vertex program's mesh
// whose shutter opens on the program's prior-frame evaluation.

#include "gpu_backend/fixture_vertex_modifier_shader.h"
#include "pathtrace/scene_ingest.h"
#include "vertex_program_motion.h"
#include "VisualImportedSceneRenderOperationContract.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{

namespace pt = videohelper::pathtrace;
namespace grid = HarmonicMIDI::grid;
namespace render = visualimportedscenerender;
using pt::Vec3;

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

bool near (float a, float b, float tolerance = 1.0e-5f) { return std::fabs (a - b) <= tolerance; }
bool near (Vec3 a, Vec3 b, float tolerance = 1.0e-5f)
{
    return near (a.x, b.x, tolerance) && near (a.y, b.y, tolerance) && near (a.z, b.z, tolerance);
}

void shutterIntervalTests()
{
    const auto centre = render::shutterInterval (500, render::MotionBlurPosition::Centre);
    check (near (centre.open, -0.25f) && near (centre.close, 0.25f), "a half-frame centred shutter spans -1/4 to 1/4");
    const auto start = render::shutterInterval (250, render::MotionBlurPosition::Start);
    check (near (start.open, 0.0f) && near (start.close, 0.25f), "a start shutter opens at the frame time");
    const auto end = render::shutterInterval (1000, render::MotionBlurPosition::End);
    check (near (end.open, -1.0f) && near (end.close, 0.0f), "an end shutter closes at the frame time");
    check (! render::shutterInterval (0, render::MotionBlurPosition::Centre).active(), "shutter 0 is off");
    check (centre.active() && start.active() && end.active(), "an open shutter is active");
}

void pointAndRotationTests()
{
    const Vec3 before { 0.0f, 0.0f, 0.0f };
    const Vec3 now { 2.0f, 0.0f, 0.0f };
    check (near (pt::pointAtFrame (before, now, 0.0f), now), "frame 0 is the current position");
    check (near (pt::pointAtFrame (before, now, -1.0f), before), "frame -1 is the previous position");
    check (near (pt::pointAtFrame (before, now, 0.25f), { 2.5f, 0.0f, 0.0f }), "past the frame the motion carries on");
    check (near (pt::motionLerp ({ 1.0f, 0.0f, 0.0f }, { 3.0f, 2.0f, 0.0f }, 0.5f), { 2.0f, 1.0f, 0.0f }),
           "a vertex at mid-shutter is halfway between open and close");

    // A quarter turn about +Y and its half.
    const float s = std::sqrt (0.5f);
    const pt::Quat identity {};
    const pt::Quat quarter { 0.0f, s, 0.0f, s };
    const pt::Quat half = pt::nlerpRotation (identity, quarter, 0.5f);
    const Vec3 turned = pt::rotate (half, { 0.0f, 0.0f, -1.0f });
    check (near (turned, { -std::sin (0.3926991f), 0.0f, -std::cos (0.3926991f) }, 1.0e-4f),
           "halfway rotation turns an eighth of a circle");
    const pt::Quat flipped { -quarter.x, -quarter.y, -quarter.z, -quarter.w };
    check (near (pt::rotate (pt::nlerpRotation (identity, flipped, 0.5f), { 0.0f, 0.0f, -1.0f }), turned, 1.0e-4f),
           "the rotation takes the shorter arc for either sign of the quaternion");
}

void cameraTests()
{
    grid::SceneCameraRecord record;
    record.transform.translation = { 2.0f, 0.0f, 5.0f };
    pt::PinholeCamera camera;
    check (pt::makePinholeCamera (record, 64, 64, camera), "the current camera builds");
    grid::SceneCameraRecord previous = record;
    previous.transform.translation = { 0.0f, 0.0f, 5.0f };
    pt::setCameraShutter (camera, previous, -0.25f, 0.25f);
    check (camera.moving, "a camera that moved since the last frame moves over the shutter");
    check (near (camera.position, { 1.5f, 0.0f, 5.0f }) && near (camera.closePosition, { 2.5f, 0.0f, 5.0f }),
           "the camera opens a quarter frame back and closes a quarter frame on");
    check (near (camera.at (0.5f).position, { 2.0f, 0.0f, 5.0f }), "mid-shutter is the frame's own camera");
    check (! camera.at (0.5f).moving, "a camera at one time is still");
    const pt::Ray ray = camera.at (1.0f).generateRay (32.0f, 32.0f);
    check (near (ray.origin, { 2.5f, 0.0f, 5.0f }), "a ray at shutter close starts at the close position");

    pt::PinholeCamera still;
    check (pt::makePinholeCamera (record, 64, 64, still), "the still camera builds");
    pt::setCameraShutter (still, record, -0.25f, 0.25f);
    check (! still.moving, "an unmoved camera stays still");
    pt::setCameraShutter (still, previous, 0.0f, 0.0f);
    check (! still.moving, "a closed shutter keeps the camera still");
}

// One triangle facing +Z at z = 0 that moves from x = 0 at shutter open to
// x = 4 at shutter close.
void motionBvhTests()
{
    const std::vector<Vec3> open { { -0.5f, -0.5f, 0.0f }, { 0.5f, -0.5f, 0.0f }, { 0.0f, 0.5f, 0.0f } };
    std::vector<Vec3> close = open;
    for (auto& p : close)
        p.x += 4.0f;
    pt::Bvh bvh;
    bvh.buildMotion (open.data(), close.data(), 1);
    check (bvh.hasMotion(), "a motion build keeps its close vertices");
    const pt::Bounds bounds = bvh.bounds();
    check (bounds.lo.x <= -0.5f && bounds.hi.x >= 4.5f, "node bounds cover the whole sweep");

    pt::Ray ray;
    ray.origin = { 0.0f, 0.0f, 1.0f };
    ray.direction = { 0.0f, 0.0f, -1.0f };
    pt::Hit hit;
    ray.time = 0.0f;
    check (bvh.intersect (ray, hit) && near (hit.t, 1.0f), "at shutter open the ray at x = 0 hits");
    check (near (hit.time, 0.0f), "the hit keeps the ray's time");
    ray.time = 1.0f;
    hit = {};
    check (! bvh.intersect (ray, hit), "at shutter close the triangle has left x = 0");
    check (! bvh.occluded (ray), "and it no longer blocks a shadow ray there");
    ray.origin = { 4.0f, 0.0f, 1.0f };
    check (bvh.intersect (ray, hit) && near (hit.time, 1.0f), "at shutter close the ray at x = 4 hits");
    ray.origin = { 2.0f, 0.0f, 1.0f };
    ray.time = 0.5f;
    check (bvh.intersect (ray, hit), "at mid-shutter the triangle is at x = 2");
    ray.time = 0.0f;
    check (! bvh.occluded (ray), "at x = 2 nothing blocks at shutter open");

    pt::Bvh still;
    still.build (open);
    check (! still.hasMotion(), "a still build has no motion");
    ray.origin = { 0.0f, 0.0f, 1.0f };
    ray.time = 1.0f;
    check (still.intersect (ray, hit), "a still triangle ignores ray time");
}

std::unique_ptr<grid::Visual3DScene> triangleScene()
{
    auto scene = std::make_unique<grid::Visual3DScene>();
    scene->objectCount = 1;
    auto& object = scene->objects[0];
    object.id.value = 1;
    object.firstVertex = 0;
    object.vertexCount = 3;
    object.firstIndex = 0;
    object.indexCount = 3;
    const Vec3 corners[3] { { -0.5f, -0.5f, 0.0f }, { 0.5f, -0.5f, 0.0f }, { 0.0f, 0.5f, 0.0f } };
    for (std::uint32_t i = 0; i < 3; ++i)
    {
        auto& vertex = scene->vertices[scene->vertexCount++];
        vertex.position = { corners[i].x, corners[i].y, corners[i].z };
        vertex.normal = { 0.0f, 0.0f, 1.0f };
        scene->indices[scene->indexCount++] = i;
    }
    return scene;
}

void ingestTests()
{
    const auto scene = triangleScene();
    grid::SceneTransform3D moved;
    moved.translation = { -1.0f, 0.0f, 0.0f };
    const pt::Mat4 previous = pt::transformMatrix (moved);

    pt::IngestOptions options;
    options.previousObjectWorlds = &previous;
    options.previousObjectWorldCount = 1;
    options.shutterOpen = -0.5f;
    options.shutterClose = 0.5f;
    pt::IngestedScene out;
    pt::ingestScene (*scene, out, options);
    check (out.motionOpen.size() == out.positions.size() && out.motionClose.size() == out.positions.size(),
           "a moving object ingests open and close vertices for every triangle");
    if (out.motionOpen.size() == 3)
    {
        check (near (out.motionOpen[0], { -1.0f, -0.5f, 0.0f }), "the open vertex is half a frame back");
        check (near (out.motionClose[0], { 0.0f, -0.5f, 0.0f }), "the close vertex is half a frame on");
        check (near (out.positions[0], { -0.5f, -0.5f, 0.0f }), "the still positions stay the current frame");
    }
    check (out.bounds.lo.x <= -1.0f && out.bounds.hi.x >= 1.0f, "scene bounds cover the sweep");

    pt::IngestOptions closed = options;
    closed.shutterClose = closed.shutterOpen;
    pt::ingestScene (*scene, out, closed);
    check (out.motionOpen.empty() && out.motionClose.empty(), "a closed shutter ingests no motion");

    const pt::Mat4 current = pt::objectWorldMatrix (*scene, 0);
    pt::IngestOptions unmoved = options;
    unmoved.previousObjectWorlds = &current;
    pt::ingestScene (*scene, out, unmoved);
    check (out.motionOpen.empty(), "an object that did not move ingests no motion");

    // A deforming mesh: the first vertex was one unit lower a frame ago.
    std::vector<grid::SceneVec3> before { { -0.5f, -1.5f, 0.0f } };
    pt::IngestOptions deforming;
    deforming.previousPositions = before.data();
    deforming.previousPositionCount = before.size();
    deforming.shutterOpen = 0.0f;
    deforming.shutterClose = 1.0f;
    pt::ingestScene (*scene, out, deforming);
    check (out.motionOpen.size() == 3, "a deforming mesh ingests motion");
    if (out.motionOpen.size() == 3)
    {
        check (near (out.motionOpen[0], { -0.5f, -0.5f, 0.0f }) && near (out.motionClose[0], { -0.5f, 0.5f, 0.0f }),
               "a deformed vertex moves on from its previous position");
        check (near (out.motionOpen[1], out.motionClose[1]), "vertices past the previous prefix hold still");
    }
}

// Position plus (c, c, c), where c reads control slot 0.
std::shared_ptr<const videohelper::materialprogram::MaterialProgramRecord> controlProgram (std::string& error)
{
    using videowire::VertexModifierOperation;
    const auto record = [] (videowire::VertexModifierStableId id, VertexModifierOperation operation,
                            std::initializer_list<videowire::VertexModifierStableId> inputs,
                            std::initializer_list<double> parameters) {
        videowire::VertexModifierRecord result;
        result.stableId = id;
        result.operation = operation;
        result.resultType = videowire::vertexModifierOperationSchema (operation).resultType;
        for (const auto input : inputs) result.inputs[result.inputCount++] = input;
        for (const auto parameter : parameters) result.parameters[result.parameterCount++] = parameter;
        return result;
    };
    videowire::VertexModifierIr ir;
    ir.rootId = 3;
    ir.records = { record (1, VertexModifierOperation::importedPosition, {}, {}),
                   record (2, VertexModifierOperation::controlParameter, {}, { 0, 0.0 }),
                   record (4, VertexModifierOperation::vec3Compose, { 2, 2, 2 }, {}),
                   record (3, VertexModifierOperation::boundedDisplacementOutput, { 1, 4 }, { 10 }) };
    const auto admitted = videohelper::vertexmodifier::admitVertexModifierIr (ir, {}, error);
    if (! admitted) return nullptr;
    auto program = videohelper::materialprogram::compileMaterialProgram (
        nullptr, &*admitted, videohelper::materialprogram::BackendTarget::OpenGl, error);
    if (! program) return nullptr;
    return std::make_shared<const videohelper::materialprogram::MaterialProgramRecord> (std::move (*program));
}

// Path Trace opens the shutter of a vertex program's mesh on the program run
// with the prior frame's inputs, as the engine does before the BVH, so a
// displacement that changes over time blurs from where it was to where it is.
void vertexProgramTests()
{
    std::string error;
    const auto program = controlProgram (error);
    check (program != nullptr, "the control program compiles: " + error);
    if (! program) return;
    arbitgpu::NativeFixtureSurfaceMaterialProgram root;
    root.object = { 1 };
    root.vertexProgram = program;
    const auto source = triangleScene();
    arbitgpu::NativeFixtureSceneRuntimeInputs in;
    in.vertexControls = { 1.0f };
    in.previousMotion = arbitgpu::sceneMotionSample (in);
    in.previousMotion->vertexControls = { 0.0f };

    auto displaced = std::make_unique<grid::Visual3DScene> (*source);
    check (videohelper::vertexprogramcpu::displaceScene (*displaced, root, videohelper::vertexprogramcpu::currentInputs (in),
                                                         error),
           "this frame's program displaces the mesh");
    std::shared_ptr<const std::vector<grid::SceneVec3>> prior;
    check (videohelper::vertexprogramcpu::priorDisplacedPositions (*source, root, in, prior, error) && prior
               && prior->size() == source->vertexCount,
           "the prior frame's displaced positions cover every scene vertex");
    if (! prior) return;

    pt::IngestOptions options;
    options.previousPositions = prior->data();
    options.previousPositionCount = prior->size();
    options.shutterOpen = -1.0f;
    options.shutterClose = 0.0f;
    pt::IngestedScene out;
    pt::ingestScene (*displaced, out, options);
    check (out.motionOpen.size() == 3, "a program whose displacement changed ingests motion");
    if (out.motionOpen.size() == 3)
    {
        check (near (out.motionOpen[0], { -0.5f, -0.5f, 0.0f }) && near (out.motionClose[0], { 0.5f, 0.5f, 1.0f }),
               "the shutter opens on the prior displacement and closes on this frame's");
        check (near (out.motionOpen[2], { 0.0f, 0.5f, 0.0f }) && near (out.motionClose[2], { 1.0f, 1.5f, 1.0f }),
               "every displaced vertex sweeps the program's change");
    }

    // A program that held still keeps the mesh still, where the undisplaced
    // prior positions would sweep the whole displacement.
    in.previousMotion->vertexControls = in.vertexControls;
    check (videohelper::vertexprogramcpu::priorDisplacedPositions (*source, root, in, prior, error),
           "a program that held still has prior positions");
    options.previousPositions = prior->data();
    pt::ingestScene (*displaced, out, options);
    check (out.motionOpen.empty(), "a displacement that held still ingests no motion");
}

void wireTests()
{
    // The request the engine wire tests use, as Path Trace with 128 samples. Its
    // V27 bytes are pinned there; here they must not move without motion blur.
    render::Request request;
    request.sourceStableId = 71;
    request.renderStableId = 72;
    request.asset.id = "model-asset-1";
    request.asset.version = 3;
    request.asset.contentSha256 = std::string (64, 'a');
    request.asset.sourceMediaType = "model/gltf-binary";
    request.asset.sourceByteSize = 4096;
    request.sceneIndex = 0;
    request.renderEngine = render::RenderEngine::PathTrace;
    request.pathTraceExportSamples = 128;
    const std::string plainWire =
        "visual.imported-scene.render.v27\n1 128 152\n"
        "visual.imported-scene.render.v1\n71 72\n\"model-asset-1\"\n3\n"
        "\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"\n"
        "\"model/gltf-binary\"\n4096\n1 0\n";
    const auto plain = render::encode (request);
    check (plain == plainWire, "a request without motion blur keeps its bytes");

    auto blurred = request;
    blurred.motionBlurShutter = 500;
    blurred.motionBlurPosition = render::MotionBlurPosition::End;
    const auto blurredWire = render::encode (blurred);
    check (blurredWire == std::string (render::kCompiledOperationHeaderV38) + "\n500 2 "
                              + std::to_string (plain.size()) + "\n" + plain,
           "V38 carries shutter, position and the exact bytes without motion blur");
    render::Request decoded;
    check (render::decode (blurredWire, decoded) && decoded.motionBlurShutter == 500
               && decoded.motionBlurPosition == render::MotionBlurPosition::End
               && render::sameRequest (blurred, decoded) && render::encode (decoded) == blurredWire,
           "the V38 wire round-trips shutter and position");
    check (! render::sameRequest (request, decoded), "motion blur is part of request identity");

    const auto header = std::string (render::kCompiledOperationHeaderV38) + "\n";
    const auto size = std::to_string (plain.size());
    render::Request refused;
    check (! render::decode (header + "0 1 " + size + "\n" + plain, refused), "a V38 shutter of 0 is refused");
    check (! render::decode (header + "1001 1 " + size + "\n" + plain, refused), "a shutter past one frame is refused");
    check (! render::decode (header + "500 3 " + size + "\n" + plain, refused), "an unknown position is refused");
    check (! render::decode (header + "500 1 " + std::to_string (blurredWire.size()) + "\n" + blurredWire, refused),
           "V38 inside V38 is refused");

    auto raster = blurred;
    raster.renderEngine = render::RenderEngine::Raster;
    raster.pathTraceExportSamples = render::kDefaultPathTraceExportSamples;
    check (render::valid (raster), "Raster carries motion blur too");
    auto offAtEnd = request;
    offAtEnd.motionBlurPosition = render::MotionBlurPosition::End;
    check (! render::valid (offAtEnd), "motion blur off keeps the canonical Centre position");
    auto tooLong = blurred;
    tooLong.motionBlurShutter = render::kMaximumMotionBlurShutter + 1;
    check (! render::valid (tooLong), "a shutter past one frame is invalid");
}

} // namespace

int main()
{
    shutterIntervalTests();
    pointAndRotationTests();
    cameraTests();
    motionBvhTests();
    ingestTests();
    vertexProgramTests();
    wireTests();
    std::cout << checks - failures << " of " << checks << " motion blur checks passed\n";
    return failures == 0 ? 0 : 1;
}
