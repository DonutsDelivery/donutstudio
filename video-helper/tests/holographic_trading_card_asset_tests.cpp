#include "support/fixture_scene.h"
#include "../src/gltf_glb.h"
#include "../../shared/HolographicTradingCardAsset.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace
{
bool expect(bool condition, const char* message)
{
    if (!condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}
}

int main()
{
    bool ok = true;
    std::string error;
    const auto card = videohelper::fixture3d::loadHolographicTradingCardScene(error);
    ok &= expect(card.has_value(), "exact card bytes must decode and adapt");
    if (card)
    {
        const auto& scene = *card->scene;
        ok &= expect(scene.id.value == 1u && scene.objects[0].id.value == 1u
                         && scene.materials[0].id.value == 1u
                         && scene.cameras[0].id.value == 2u
                         && scene.lights[0].id.value == 3u
                         && scene.activeCamera.value == 2u,
                     "stable scene, object, material, camera, and light identities must survive");
        ok &= expect(scene.vertexCount == 24u && scene.indexCount == 36u
                         && scene.objects[0].vertexCount == 24u
                         && scene.objects[0].indexCount == 36u,
                     "adapter must retain the authored card geometry");
    }

    videohelper::gltf::GlbAdmissionOptions options;
    options.supportedRequiredExtensions = { "KHR_lights_punctual" };

    auto malformed = std::vector<std::uint8_t>(
        holographictradingcard::kGlb.begin(), holographictradingcard::kGlb.end());
    malformed[0] = 0;
    error.clear();
    ok &= expect(!videohelper::gltf::decodeStaticGlb(malformed, options, error),
                 "malformed exact card bytes must fail closed");

    auto wrongDeclaredLength = std::vector<std::uint8_t>(
        holographictradingcard::kGlb.begin(), holographictradingcard::kGlb.end());
    ++wrongDeclaredLength[8];
    error.clear();
    ok &= expect(!videohelper::gltf::decodeStaticGlb(wrongDeclaredLength, options, error),
                 "card GLB declared-length mismatch must fail closed");

    auto oversizedOptions = options;
    oversizedOptions.limits.maxContainerBytes = holographictradingcard::kGlb.size() - 1u;
    error.clear();
    ok &= expect(!videohelper::gltf::decodeStaticGlb(
                     holographictradingcard::kGlb.data(),
                     holographictradingcard::kGlb.size(), oversizedOptions, error),
                 "oversized card GLB must reject before decode allocation");

    return ok ? 0 : 1;
}
