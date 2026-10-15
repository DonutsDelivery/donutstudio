#pragma once

#include "gpu_backend/backend.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

namespace videohelper::importedsceneprobe
{
// Populated only by a composite frame probe, after the exact native draw.
struct Receipt
{
    std::size_t sceneCount = 0;
    std::size_t objectCount = 0;
    std::size_t vertexCount = 0;
    std::size_t indexCount = 0;
    std::size_t cameraCount = 0;
    std::size_t drawCount = 0;
    std::size_t submittedInstanceCount = 0;
    bool materialFramePresent = false;
    bool materialFrameRead = false;
    std::string materialFrameFormat;
    std::uint32_t materialFrameWidth = 0;
    std::uint32_t materialFrameHeight = 0;
    std::size_t materialFrameAlphaPixels = 0;
    std::size_t materialFrameOpaquePixels = 0;
    std::size_t materialFrameNonblackPixels = 0;
    bool scenePassesRead = false;
    std::size_t sceneColorPixelsDifferentFromCorner = 0;
    std::uint8_t sceneCornerRed = 0;
    std::uint8_t sceneCornerGreen = 0;
    std::uint8_t sceneCornerBlue = 0;
    std::size_t maskPixels = 0;
    std::size_t objectIdPixels = 0;
    std::size_t depthPixels = 0;
    std::string error;
};

inline thread_local Receipt* activeReceipt = nullptr;

struct ScopedCapture
{
    explicit ScopedCapture(Receipt* receipt) : previous(activeReceipt)
    {
        activeReceipt = receipt;
    }
    ~ScopedCapture() { activeReceipt = previous; }
    ScopedCapture(const ScopedCapture&) = delete;
    ScopedCapture& operator=(const ScopedCapture&) = delete;
    Receipt* previous;
};

inline void record(const arbitgpu::NativeFixtureSceneFrame& sceneFrame,
                   const arbitgpu::NativeFixtureSceneStats& stats,
                   std::size_t objects, std::size_t vertices, std::size_t indices,
                   std::size_t cameras,
                   const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& materialFrame)
{
    auto* receipt = activeReceipt;
    if (receipt == nullptr || sceneFrame.backend() != "metal") return;
    ++receipt->sceneCount;
    receipt->objectCount += objects;
    receipt->vertexCount += vertices;
    receipt->indexCount += indices;
    receipt->cameraCount += cameras;
    receipt->drawCount += stats.drawCount;
    receipt->submittedInstanceCount += stats.submittedInstanceCount;
    receipt->materialFramePresent |= static_cast<bool>(materialFrame);
    std::string error;
    if (materialFrame)
    {
        const auto descriptor = materialFrame->colorTextureDescriptor();
        receipt->materialFrameFormat = descriptor.format
            == arbitgpu::NativeTexturePixelFormat::Bgra8Unorm ? "bgra8unorm" : "unexpected";
        receipt->materialFrameWidth = descriptor.width;
        receipt->materialFrameHeight = descriptor.height;
        arbitgpu::NativeRawPassPixels pixels;
        if (!materialFrame->readRawPass(renderpassoutput::Output::Color, pixels, error))
            receipt->error = "material Frame readback: " + error;
        else if (pixels.format != arbitgpu::NativeTexturePixelFormat::Rgba8Unorm
            || pixels.bytes.size() != static_cast<std::size_t>(descriptor.width) * descriptor.height * 4)
            receipt->error = "material Frame readback has an unexpected format or extent";
        else
        {
            receipt->materialFrameRead = true;
            for (std::size_t i = 0; i < pixels.bytes.size(); i += 4)
            {
                receipt->materialFrameOpaquePixels += pixels.bytes[i + 3] == 255;
                receipt->materialFrameAlphaPixels += pixels.bytes[i + 3] != 0;
                receipt->materialFrameNonblackPixels += pixels.bytes[i] != 0
                    || pixels.bytes[i + 1] != 0 || pixels.bytes[i + 2] != 0;
            }
        }
    }

    arbitgpu::NativeRawPassPixels color, mask, ids, depth;
    if (!sceneFrame.readRawPass(renderpassoutput::Output::Color, color, error)
        || !sceneFrame.readRawPass(renderpassoutput::Output::Mask, mask, error)
        || !sceneFrame.readRawPass(renderpassoutput::Output::ObjectId, ids, error)
        || !sceneFrame.readRawPass(renderpassoutput::Output::Depth, depth, error))
    { receipt->error = "scene pass readback: " + error; return; }
    const auto extent = static_cast<std::size_t>(sceneFrame.width()) * sceneFrame.height();
    if (color.format != arbitgpu::NativeTexturePixelFormat::Rgba8Unorm || color.bytes.size() != extent * 4
        || mask.format != arbitgpu::NativeTexturePixelFormat::R8Unorm || mask.bytes.size() != extent
        || ids.format != arbitgpu::NativeTexturePixelFormat::R32Uint || ids.bytes.size() != extent * 4
        || depth.format != arbitgpu::NativeTexturePixelFormat::R32Float || depth.bytes.size() != extent * 4)
    { receipt->error = "scene pass readback has an unexpected format or extent"; return; }
    receipt->scenePassesRead = true;
    receipt->sceneCornerRed = color.bytes[0];
    receipt->sceneCornerGreen = color.bytes[1];
    receipt->sceneCornerBlue = color.bytes[2];
    for (std::size_t i = 0; i < extent; ++i)
    {
        const auto* rgba = color.bytes.data() + 4 * i;
        receipt->sceneColorPixelsDifferentFromCorner += rgba[0] != receipt->sceneCornerRed
            || rgba[1] != receipt->sceneCornerGreen || rgba[2] != receipt->sceneCornerBlue;
        receipt->maskPixels += mask.bytes[i] != 0;
        std::uint32_t objectId = 0;
        float distance = 1.0f;
        std::memcpy(&objectId, ids.bytes.data() + 4 * i, sizeof(objectId));
        std::memcpy(&distance, depth.bytes.data() + 4 * i, sizeof(distance));
        receipt->objectIdPixels += objectId != 0;
        receipt->depthPixels += distance < 1.0f;
    }
}
} // namespace videohelper::importedsceneprobe
