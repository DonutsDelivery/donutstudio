// Path Trace camera lens: aperture samples cover the disc or blade polygon
// uniformly, the focal length and aperture radius follow the f-stop, every lens
// ray of a pixel meets on the focus plane, the light tracer's lens projection
// inverts generateRay, a pinhole stays a pinhole, and the scene wire and the
// imported-scene codec carry the lens fields while pinhole cameras encode as
// before.

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
using pt::Vec2;
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

bool close (double a, double b, double tolerance) { return std::fabs (a - b) <= tolerance; }

Vec2 grid2 (int i, int j, int n)
{
    return { (static_cast<float> (i) + 0.5f) / static_cast<float> (n), (static_cast<float> (j) + 0.5f) / static_cast<float> (n) };
}

void apertureTests()
{
    // Round: inside the unit disc, centred, E[r^2] = 1/2 for a uniform disc.
    constexpr int n = 64;
    double meanX = 0.0, meanY = 0.0, meanR2 = 0.0, worst = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const Vec2 p = pt::sampleAperture (grid2 (i, j, n), 0, 0.0f);
            meanX += p.x;
            meanY += p.y;
            meanR2 += p.x * p.x + p.y * p.y;
            worst = std::max (worst, static_cast<double> (std::sqrt (p.x * p.x + p.y * p.y)));
        }
    meanX /= n * n;
    meanY /= n * n;
    meanR2 /= n * n;
    check (worst <= 1.0 + 1e-6, "round aperture samples stay in the unit disc");
    check (close (meanX, 0.0, 1e-3) && close (meanY, 0.0, 1e-3), "round aperture samples are centred");
    check (close (meanR2, 0.5, 2e-3), "round aperture samples are uniform over the disc");

    // Hexagon with a corner at 30 degrees: inside the polygon, uniform over it.
    // For a regular n-gon of circumradius 1, E[r^2] = (1 + 2 cos^2(pi/n)) / 6.
    const std::uint32_t blades = 6;
    const float rotation = 0.5235988f;
    const double apothem = std::cos (3.14159265358979 / blades);
    double hexR2 = 0.0;
    bool inside = true;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const Vec2 p = pt::sampleAperture (grid2 (i, j, n), blades, rotation);
            hexR2 += p.x * p.x + p.y * p.y;
            const double angle = std::atan2 (p.y, p.x) - rotation;
            const double wedge = 2.0 * 3.14159265358979 / blades;
            double local = std::fmod (angle, wedge);
            if (local < 0.0) local += wedge;
            inside &= std::sqrt (p.x * p.x + p.y * p.y) * std::cos (local - 0.5 * wedge) <= apothem + 1e-5;
        }
    hexR2 /= n * n;
    check (inside, "blade aperture samples stay inside the polygon");
    check (close (hexR2, (1.0 + 2.0 * apothem * apothem) / 6.0, 3e-3), "blade aperture samples are uniform over the polygon");
}

grid::SceneCameraRecord lensRecord()
{
    grid::SceneCameraRecord camera;
    camera.id = grid::SceneCameraId { 4 };
    // A 50 mm lens on the 24 mm sensor: tan(fov / 2) = 12 / 50.
    camera.verticalFovRadians = 2.0f * std::atan (0.24f);
    camera.fStop = 2.0f;
    camera.focusDistance = 3.0f;
    camera.apertureBlades = 5;
    camera.bladeRotationRadians = 0.3f;
    camera.apertureRatio = 1.5f;
    return camera;
}

void focusTests()
{
    const auto record = lensRecord();
    check (close (grid::sceneCameraFocalLength (record), 0.05, 1e-6), "a 24 mm tall sensor at this fov has a 50 mm lens");
    check (close (grid::sceneCameraApertureRadius (record), 0.0125, 1e-6), "f/2 at 50 mm is a 12.5 mm radius");
    check (close (grid::sceneCameraCircleOfConfusion (record, 3.0f), 0.0, 1e-9), "the focus plane is sharp");
    check (close (grid::sceneCameraCircleOfConfusion (record, 6.0f), 0.025 * 0.5, 1e-6),
           "twice the focus distance blurs by half the aperture diameter");
    auto pinhole = record;
    pinhole.fStop = 0.0f;
    check (grid::sceneCameraApertureRadius (pinhole) == 0.0f && ! grid::sceneCameraHasLens (pinhole), "f-stop 0 is a pinhole");

    pt::PinholeCamera camera;
    check (pt::makePinholeCamera (record, 64, 48, camera) && camera.hasLens()
               && close (camera.apertureRadius, 0.0125, 1e-6) && camera.apertureBlades == 5,
           "the path tracer camera takes the lens");

    // Every lens ray through one pixel reaches the same point at the focus depth.
    const Vec3 forward = pt::rotate (camera.rotation, Vec3 { 0.0f, 0.0f, -1.0f });
    const pt::Ray centre = camera.generateRay (20.3f, 11.7f);
    const float centreT = camera.focusDistance / pt::dot (centre.direction, forward);
    const Vec3 target = centre.origin + centre.direction * centreT;
    double spread = 0.0, lensWidth = 0.0;
    bool inverse = true;
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
        {
            const pt::Ray ray = camera.generateRay (20.3f, 11.7f, grid2 (i, j, 8));
            const float t = camera.focusDistance / pt::dot (ray.direction, forward);
            const Vec3 at = ray.origin + ray.direction * t;
            spread = std::max (spread, static_cast<double> (pt::length (at - target)));
            lensWidth = std::max (lensWidth, static_cast<double> (pt::length (ray.origin - camera.position)));
            // A point further along the same ray projects back to the same pixel.
            const Vec3 far = ray.origin + ray.direction * (3.0f * t);
            const Vec3 offset = pt::rotate (pt::conjugate (camera.rotation), ray.origin - camera.position);
            float x = 0.0f, y = 0.0f, depth = 0.0f;
            inverse &= camera.projectThroughLens (far, offset, x, y, depth) && close (x, 20.3, 1e-3) && close (y, 11.7, 1e-3)
                       && close (depth, 3.0 * camera.focusDistance, 1e-3);
            // The near plane still clips at view depth near.
            inverse &= close (ray.tMin * pt::dot (ray.direction, forward), camera.nearPlane, 1e-5);
        }
    check (spread < 1e-5, "lens rays of one pixel converge on the focus plane");
    check (lensWidth > 0.005, "lens rays leave from across the aperture");
    check (inverse, "projectThroughLens inverts generateRay and keeps the near plane");

    pt::PinholeCamera pin;
    check (pt::makePinholeCamera (pinhole, 64, 48, pin) && ! pin.hasLens(), "a pinhole record makes a pinhole camera");
    const pt::Ray a = pin.generateRay (5.5f, 6.5f);
    const pt::Ray b = pin.generateRay (5.5f, 6.5f, { 0.9f, 0.1f });
    check (a.origin.x == b.origin.x && a.direction.x == b.direction.x && a.direction.z == b.direction.z,
           "a pinhole ignores the lens sample");
}

void wireTests()
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
    scene.objects[0].id = grid::SceneObjectId { 3 };
    scene.objects[0].material = scene.materials[0].id;
    scene.objects[0].vertexCount = scene.objects[0].indexCount = 3;
    scene.objectCount = 1;
    scene.cameras[0].id = grid::SceneCameraId { 4 };
    scene.cameraCount = 1;
    scene.activeCamera = scene.cameras[0].id;

    const auto plainBytes = sceneaov::detail::encodeScene (scene);
    auto lens = scene;
    lens.cameras[0] = lensRecord();
    check (grid::validateVisual3DScene (lens).valid(), "a lens camera validates");
    const auto lensBytes = sceneaov::detail::encodeScene (lens);
    check (lensBytes.size() == plainBytes.size() + 8 + sceneaov::detail::kCameraLensRecordBytes, "the lens adds one trailer");
    grid::Visual3DScene decoded;
    check (sceneaov::detail::decodeScene (lensBytes, decoded) && grid::sameCameraLens (decoded.cameras[0], lens.cameras[0]),
           "the lens fields round-trip on the scene wire");
    grid::Visual3DScene decodedPlain;
    check (sceneaov::detail::decodeScene (plainBytes, decodedPlain) && ! grid::sceneCameraLensAuthored (decodedPlain.cameras[0])
               && sceneaov::detail::encodeScene (decodedPlain) == plainBytes,
           "a pinhole scene keeps its bytes");
    auto badLens = lens;
    badLens.cameras[0].apertureBlades = 2;
    check (! grid::validateVisual3DScene (badLens).valid(), "two blades are refused");

    // The imported-scene codec continues a lens camera's line and ends a pinhole's as before.
    const auto encodeCamera = [] (const grid::SceneCameraRecord& camera) {
        std::ostringstream stream;
        visualimportedscenerender::detail::encodeCamera (stream, camera);
        return stream.str();
    };
    const auto pinholeText = encodeCamera (scene.cameras[0]);
    const auto lensText = encodeCamera (lens.cameras[0]);
    check (std::count (pinholeText.begin(), pinholeText.end(), ' ') == 13, "a pinhole camera line is unchanged");
    std::istringstream stream (lensText + pinholeText);
    grid::SceneCameraRecord first, second;
    check (visualimportedscenerender::detail::decodeCamera (stream, first) && grid::sameCameraLens (first, lens.cameras[0])
               && visualimportedscenerender::detail::decodeCamera (stream, second) && ! grid::sceneCameraLensAuthored (second),
           "imported-scene camera lines round-trip with and without a lens");
}

} // namespace

int main()
{
    apertureTests();
    focusTests();
    wireTests();
    std::cout << checks - failures << '/' << checks << " camera lens checks passed\n";
    return failures == 0 ? 0 : 1;
}
