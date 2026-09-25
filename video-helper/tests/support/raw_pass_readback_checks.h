#pragma once
#include "../../src/gpu_backend/backend.h"

namespace rawpasschecks
{
template <typename Check>
void verify(const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame, Check&& check)
{
    check(frame != nullptr, "typed pass readback requires a rendered owner");
    if (!frame) return;
    for (const auto output : renderpassoutput::kOutputs)
    {
        arbitgpu::NativeRawPassPixels first, repeated;
        std::string error;
        const bool ok = frame->readRawPass(output, first, error);
        check(ok && error.empty(), "all selected native attachments support typed readback");
        if (!ok) continue;
        check(arbitgpu::rawPassFormatMatches(output, first.format)
            && first.format != arbitgpu::NativeTexturePixelFormat::Bgra8Unorm
            && first.width == frame->width() && first.height == frame->height()
            && first.bytes.size() == static_cast<std::uint64_t>(first.width) * first.height
                * arbitgpu::rawPassChannels(first.format) * arbitgpu::rawPassScalarBytes(first.format),
            "OpenGL and Metal publish identical typed shapes with RGBA channel order");
        check(frame->readRawPass(output, repeated, error) && repeated.bytes == first.bytes,
            "repeated raw readback does not mutate or display-encode the attachment");
        if (output == renderpassoutput::Output::Motion)
            check(std::all_of(first.bytes.begin(), first.bytes.end(), [](auto byte) { return byte == 0; }),
                "stationary first-frame motion exports exact half-float zero");
    }
    arbitgpu::NativeRawPassPixels invalid;
    std::string error;
    check(!frame->readRawPass(renderpassoutput::Output::Count, invalid, error)
        && invalid.bytes.empty() && !error.empty(), "unknown raw attachments fail closed");
}
} // namespace rawpasschecks
