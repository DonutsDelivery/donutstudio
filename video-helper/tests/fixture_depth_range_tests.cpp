// The fixture depth test keeps the Grooved Disc's grating film in front of the
// body cap and label beneath it. The starter frames the disc from about 80 units
// with near 0.1 and far 400, so the authored planes give one 24-bit step of
// about 4 mm against a 1.5 mm gap, and the film tied the cap in stripes.
#include "gpu_backend/fixture_depth_range.h"

#include <cmath>
#include <cstdio>

namespace
{
namespace fd = arbitgpu::fixturedepth;

int failures = 0;

void expect(bool condition, const char* message)
{
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message);
}

const fd::Matrix identity {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

// The starter's Disc Camera: at (0, 57.5, 55), pitched down to the disc's centre.
struct Camera
{
    std::array<float, 4> rotation {};
    std::array<float, 3> translation {0.0f, 57.5f, 55.0f};
    Camera()
    {
        const float pitch = -std::atan2(57.5f, 55.0f);
        rotation = {std::sin(0.5f*pitch), 0.0f, 0.0f, std::cos(0.5f*pitch)};
    }
};

// How many points across the disc's grating annulus the film strictly wins at.
int filmWins(const Camera& camera, float nearPlane, float farPlane, float filmY, float underY, int& points)
{
    int wins = 0;
    points = 0;
    for (int ring = 0; ring < 32; ++ring)
        for (int step = 0; step < 64; ++step) {
            const float radius = 0.29f + 0.352f*float(ring)/31.0f;
            const float angle = 6.2831853f*float(step)/64.0f;
            const float x = radius*std::cos(angle), z = radius*std::sin(angle);
            const auto depth = [&](float y) {
                return fd::quantizedDepth(fd::perspectiveWindowDepth(
                    fd::cameraDistance({x, y, z}, camera.rotation, camera.translation), nearPlane, farPlane), 24);
            };
            ++points;
            wins += depth(filmY) < depth(underY) ? 1 : 0;
        }
    return wins;
}

void starterDiscFilmWinsTheDepthTest()
{
    const Camera camera;
    expect(std::fabs(fd::cameraDistance({0, 0, 0}, camera.rotation, camera.translation) - 79.57f) < 0.01f,
           "camera distance matches the starter's");
    fd::Range scene;
    // Disc body and its film and label: radius 0.654, from 11 mm under the cap to the film's top.
    fd::includeBox(scene, identity, {-0.654f, -0.011f, -0.654f}, {0.654f, 0.0035f, 0.654f},
                   camera.rotation, camera.translation);
    const auto range = fd::fitted(scene, 0.1f, 400.0f);
    expect(range[0] > 78.0f && range[1] < 81.0f, "fitted range frames the disc");
    expect(range[0] < scene.nearest && range[1] > scene.farthest, "fitted range pads outward");

    int points = 0;
    const int authoredCap = filmWins(camera, 0.1f, 400.0f, 0.0035f, 0.002f, points);
    expect(authoredCap < points, "authored planes tie the film with the body cap somewhere (the stripes)");
    expect(filmWins(camera, range[0], range[1], 0.0035f, 0.002f, points) == points,
           "fitted range: the film wins over the body cap everywhere");
    expect(filmWins(camera, range[0], range[1], 0.0035f, 0.00195f, points) == points,
           "fitted range: the film wins over the label everywhere");
    // The film's own grating faces 0.25 mm apart still separate.
    expect(filmWins(camera, range[0], range[1], 0.0035f, 0.00325f, points) == points,
           "fitted range separates quarter-millimetre layers");
}

void rangeStaysInsideTheAuthoredPlanes()
{
    const Camera camera;
    fd::Range scene;
    fd::includeBox(scene, identity, {-500, -500, -500}, {500, 500, 500}, camera.rotation, camera.translation);
    const auto range = fd::fitted(scene, 0.1f, 400.0f);
    expect(range[0] == 0.1f && range[1] == 400.0f, "a scene past both planes keeps both planes");
    expect(fd::fitted(fd::Range {}, 0.1f, 400.0f) == std::array<float, 2> {0.1f, 400.0f},
           "an empty scene keeps the authored planes");
    fd::Range behind;
    fd::includeBox(behind, identity, {-1, 200, 200}, {1, 201, 201}, camera.rotation, camera.translation);
    expect(fd::fitted(behind, 0.1f, 400.0f) == std::array<float, 2> {0.1f, 400.0f},
           "a scene wholly behind the camera keeps the authored planes");
    // A translated world matrix moves the box.
    fd::Matrix moved = identity;
    moved[13] = 10.0f;
    fd::Range lifted;
    fd::includeBox(lifted, moved, {-0.1f, -0.1f, -0.1f}, {0.1f, 0.1f, 0.1f}, camera.rotation, camera.translation);
    expect(lifted.farthest < 79.57f - 6.0f, "the world matrix's translation moves the box toward the camera");
}
} // namespace

int main()
{
    starterDiscFilmWinsTheDepthTest();
    rangeStaysInsideTheAuthoredPlanes();
    if (failures != 0) return 1;
    std::puts("fixture depth range: ok");
    return 0;
}
