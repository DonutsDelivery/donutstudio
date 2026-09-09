#pragma once

#include "../../src/sdf_native_renderer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace videohelper::sdf::test
{
struct NativeOperationFixture final
{
    videowire::SdfOperation operation;
    std::shared_ptr<const AdmittedSdfIr> geometry;
};

std::vector<NativeOperationFixture> nativeOperationFixtures (std::string& error);
std::vector<NativeOperationFixture> largeOffsetPolarRepeatFixtures (std::string& error);

bool verifyNativeDepthParity (const NativeOperationFixture& fixture,
                              const NativeSdfRenderControls& controls,
                              const std::vector<std::uint8_t>& rgba,
                              std::uint32_t width,
                              std::uint32_t height,
                              std::string& error);
} // namespace videohelper::sdf::test
