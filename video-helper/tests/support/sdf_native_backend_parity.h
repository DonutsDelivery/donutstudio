#pragma once

#include "../../src/sdf_native_renderer.h"

#include <cstdint>
#include <functional>
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

// Readbacks retain native rows: GL bottom first, Metal top first. The shared
// checks map those rows to the scalar oracle's bottom-origin probe coordinates.
bool verifyNativeDepthParity (const NativeOperationFixture& fixture,
                              const NativeSdfRenderControls& controls,
                              const std::vector<std::uint8_t>& rgba,
                              std::uint32_t width,
                              std::uint32_t height,
                              std::string& error);

// Returns RGBA floats in the frame descriptor's native row order.
using NativeSdfFloatReader = std::function<std::vector<float> (const arbitgpu::NativeSdfSceneFrame&)>;
bool verifyNativeUtilityOutputs (const NativeSdfFloatReader& readPixels, std::string& error);
} // namespace videohelper::sdf::test
