// Rough closed glass runs the V2 walk on OpenGL where the walk exists.
//
// V1 refuses a rough closed volume at ordinary roughness, because its masked power
// stays unresolved, and V1 is what a saved material authors by default. This runs
// the production route and admission on the CPU and needs no GPU.

#include "../src/gpu_backend/fixture_smooth_glass.h"
#include <array>
#include <cstdio>
#include <memory>
#include <string>

namespace
{
namespace sg = arbitgpu::smoothglass;
using namespace HarmonicMIDI::grid;
using Backend = arbitgpu::NativeFixtureMaterialBackend;
using Program = arbitgpu::NativeFixtureSurfaceMaterialProgram;
int failures = 0;
void check(bool condition, const std::string& label)
{
    std::printf("%s: %s\n", condition ? "ok" : "FAILED", label.c_str());
    if (!condition) ++failures;
}
constexpr std::uint32_t kGlass = 0x3003u, kMaterial = 7;

std::shared_ptr<const Visual3DScene> cube()
{
    auto scene = std::make_shared<Visual3DScene>();
    scene->id = {7};
    auto& object = scene->objects[scene->objectCount++];
    object = {};
    object.id = {kGlass};
    object.material = {kMaterial};
    auto& material = scene->materials[scene->materialCount++];
    material = {};
    material.id = {kMaterial};
    material.opacity = 1;
    for (int c = 0; c < 8; ++c)
    {
        auto& v = scene->vertices[scene->vertexCount++];
        v = {};
        v.position = {c & 1 ? 1.f : -1.f, c & 2 ? 1.f : -1.f, c & 4 ? 1.f : -1.f};
        v.normal = {0, 1, 0};
    }
    constexpr std::array<std::array<std::uint32_t, 4>, 6> faces {{{{0, 2, 3, 1}}, {{4, 5, 7, 6}}, {{0, 1, 5, 4}},
                                                                  {{2, 6, 7, 3}}, {{0, 4, 6, 2}}, {{1, 3, 7, 5}}}};
    for (const auto& f : faces)
        for (const auto i : {f[0], f[1], f[2], f[0], f[2], f[3]})
            scene->indices[scene->indexCount++] = i;
    object.vertexCount = std::uint32_t(scene->vertexCount);
    object.indexCount = std::uint32_t(scene->indexCount);
    return scene;
}
std::shared_ptr<Program> glass(const Visual3DScene& scene, float roughness, std::uint32_t version)
{
    auto p = std::make_shared<Program>();
    p->backend = Backend::OpenGl;
    p->object = {kGlass};
    p->bindingDigest = "rough-version-binding";
    p->programIdentity = "rough-version-program";
    p->authoredScene = scene.id;
    p->authoredMaterial = {kMaterial};
    p->sceneRevision = p->structuralRevision = p->evaluationRevision = p->materialRevision = 1;
    p->roughGlassScatteringVersion = version;
    p->parameters.identifiers[0] = kMaterial;
    p->parameters.identifiers[1] = std::uint32_t(surfacematerial::TransportMode::ClosedVolume);
    p->parameters.normalOpacity = {0, 0, 1, 1};
    p->parameters.baseColorMetallic = {1, 1, 1, 0};
    p->parameters.transmissionIorClearcoat = {1, 1.5f, 0, 0};
    p->parameters.emissionRoughness = {0, 0, 0, roughness};
    return p;
}
}

int main()
{
    const sg::RoughNativeConsumer withWalk {Backend::OpenGl, true, true, true, sg::kMultipleScatterVersion, true};
    const sg::RoughNativeConsumer noWalk {Backend::OpenGl, true, true, true, sg::kMultipleScatterVersion, false};

    check(sg::roughCameraVersion(0, true, .28f, withWalk) == 2, "a new closed volume at .28 takes V2");
    check(sg::roughCameraVersion(0, true, 0, withWalk) == 1, "zero roughness takes V1");
    check(sg::roughCameraVersion(0, false, .28f, withWalk) == 1, "thin water takes V1");
    check(sg::roughCameraVersion(0, true, .28f, noWalk) == 1, "a backend without the walk takes V1");
    auto metal = withWalk;
    metal.backend = Backend::Metal;
    check(sg::roughCameraVersion(0, true, .28f, metal) == 1, "Metal takes V1 for a new material");
    check(sg::roughCameraVersion(1, true, .28f, withWalk) == 1, "a saved or chosen V1 stays V1");
    check(sg::roughCameraVersion(2, true, .28f, withWalk) == 2, "a chosen V2 stays V2");
    check(sg::roughCameraVersion(2, true, .28f, metal) == 2, "and on Metal too");

    const auto scene = cube();
    std::string error;
    sg::Prepared routed;
    check(sg::prepare(scene, glass(*scene, .28f, 0), {}, routed, error, withWalk), "a new material admits: " + error);
    check(sg::multipleScatterCamera(routed), "and prepares the V2 camera");
    sg::Prepared saved;
    error.clear();
    check(sg::prepare(scene, glass(*scene, .28f, 1), {}, saved, error, withWalk), "a saved V1 admits: " + error);
    check(!sg::multipleScatterCamera(saved), "and prepares the V1 camera");
    sg::Prepared kept;
    error.clear();
    check(sg::prepare(scene, glass(*scene, .28f, 0), {}, kept, error, noWalk), "a new material without the walk admits: " + error);
    check(!sg::multipleScatterCamera(kept), "and prepares the V1 camera");
    sg::Prepared refused;
    check(!sg::prepare(scene, glass(*scene, .28f, 2), {}, refused, error, noWalk), "a chosen V2 without the walk still refuses");

    // A stored V1 that can't finish gets V2 for its frame, and the stored value isn't rewritten.
    const auto stored = glass(*scene, .28f, 1);
    const auto best = sg::bestRoughProgram(stored);
    check(best && best->roughGlassScatteringVersion == 0, "a stored V1 closed volume has a best-walk program");
    check(stored->roughGlassScatteringVersion == 1, "and the stored program stays V1");
    check(best && best->bindingDigest != stored->bindingDigest, "the second program has its own identity");
    sg::Prepared second;
    error.clear();
    check(best && sg::prepare(scene, best, {}, second, error, withWalk) && sg::multipleScatterCamera(second), "it prepares the V2 camera: " + error);
    check(!sg::bestRoughProgram(glass(*scene, .28f, 2)), "a chosen V2 has no second program");
    auto thin = std::make_shared<Program>(*stored);
    thin->parameters.identifiers[1] = std::uint32_t(surfacematerial::TransportMode::ThinInterface);
    check(!sg::bestRoughProgram(thin), "nor does thin water");
    auto mixed = std::make_shared<Program>(*glass(*scene, .28f, 2));
    mixed->objectPrograms.push_back(stored);
    const auto mixedBest = sg::bestRoughProgram(mixed);
    check(mixedBest && mixedBest->objectPrograms[0]->roughGlassScatteringVersion == 0 && mixedBest->roughGlassScatteringVersion == 2,
          "a mixed material changes only its V1 volume");
    check(mixed->objectPrograms[0]->roughGlassScatteringVersion == 1, "and leaves the stored one alone");

    sg::RoughV1Fallback fallback;
    check(!fallback.shouldTry(""), "a frame V1 finishes gets no second try");
    check(!fallback.shouldTry("Rough glass energy receipt is incomplete"), "nor does another failure");
    check(fallback.shouldTry(sg::kRoughUnresolvedError), "a frame V1 can't finish gets one");
    fallback.secondTryFailed("OpenGL lost its context");
    check(fallback.shouldTry(sg::kRoughUnresolvedError), "an unrelated second failure doesn't end the fallback");
    fallback.secondTryFailed(sg::kRoughUnresolvedError);
    check(!fallback.shouldTry(sg::kRoughUnresolvedError), "a renderer with no V2 stops trying");
    std::string unresolved;
    sg::RoughEnergyReceipt receipt;
    receipt.terminal = {.9, .9, .9};receipt.masked = {.1, .1, .1};
    check(!sg::completeRoughEnergy(receipt, unresolved) && unresolved == sg::kRoughUnresolvedError,
          "the receipt check reports the same words the fallback reads");
    return failures == 0 ? 0 : 1;
}
