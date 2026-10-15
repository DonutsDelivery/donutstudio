#pragma once

#include "backend.h"

namespace arbitgpu
{
// Private float32 scratch readback for native acceptance. These are owned
// transport diagnostics, never additional public render-pass outputs.
bool readMetalRoughGlassReceipt(
    const std::shared_ptr<const NativeFixtureSceneFrame>& owner,
    std::array<NativeRawPassPixels, 3>& receipts,
    NativeRawPassPixels& objectIds, NativeRawPassPixels& materialIds, std::string& error);
}
