#pragma once

#include "../../src/renderer.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace videohelper::tests
{
template <typename ReadFrame>
bool frameTextureLeaseCases(videorender::FrameRenderer& renderer, int width, int height,
                           ReadFrame readFrame)
{
    bool ok = true;
    const auto check = [&](bool value, const char* message)
    {
        if (!value) { std::cerr << "FAIL: " << message << '\n'; ok = false; }
        return value;
    };
    std::vector<std::uint8_t> red(static_cast<std::size_t>(width) * height * 4);
    auto blue = red;
    for (std::size_t i = 0; i < red.size(); i += 4)
    {
        red[i] = 128; red[i + 3] = 255;
        blue[i + 2] = 64; blue[i + 3] = 255;
    }
    std::string error;
    const auto source = renderer.uploadRgba(red.data(), width, height, width * 4, 0);
    check(!renderer.leaseRgbaTexture(source, width + 1, height, error),
          "lease factory rejects mismatched source dimensions");
    auto redLease = renderer.leaseRgbaTexture(source, width, height, error);
    renderer.uploadRgba(blue.data(), width, height, width * 4, source);
    auto blueLease = renderer.leaseRgbaTexture(source, width, height, error);
    const auto other = renderer.uploadRgba(red.data(), width, height, width * 4, 0);
    const auto mixed = renderer.frameBlendInto(source, other, 0, width, height, 0.25f);
    auto mixedLease = renderer.leaseRgbaTexture(mixed, width, height, error);
    renderer.deleteTexture(mixed); renderer.deleteTexture(other); renderer.deleteTexture(source);
    check(!renderer.leaseRgbaTexture(source, width, height, error),
          "lease factory rejects a deleted source");
    if (!check(arbitgpu::validMaterialFrameTexture(redLease)
                && arbitgpu::validMaterialFrameTexture(blueLease)
                && arbitgpu::validMaterialFrameTexture(mixedLease),
               "decode and GPU frame-blend leases expose complete owned descriptors")) return false;
    check(redLease->colorTextureDescriptor().rendererGeneration
              != blueLease->colorTextureDescriptor().rendererGeneration,
          "successive immutable leases have different lifetime generations");
    const auto redPixels = readFrame(redLease);
    const auto bluePixels = readFrame(blueLease);
    const auto mixedPixels = readFrame(mixedLease);
    check(redPixels == red && bluePixels == blue,
          "leased pixels survive source replacement and deletion without CPU snapshotting");
    bool blendOk = mixedPixels.size() == red.size();
    for (std::size_t i = 0; blendOk && i < mixedPixels.size(); i += 4)
        blendOk = std::abs(static_cast<int>(mixedPixels[i]) - 32) <= 1
            && mixedPixels[i + 1] == 0
            && std::abs(static_cast<int>(mixedPixels[i + 2]) - 48) <= 1
            && mixedPixels[i + 3] == 255;
    check(blendOk, "the pre-3D lease resolves GPU retiming before its bracket sources are released");
    return ok;
}
} // namespace videohelper::tests
