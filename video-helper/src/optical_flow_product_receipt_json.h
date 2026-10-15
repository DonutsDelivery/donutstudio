#pragma once

#include "optical_flow_product_execution.h"

#include <nlohmann/json.hpp>

namespace videoopticalflow
{
template <typename Bytes>
inline std::string receiptHex (const Bytes& bytes)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve (bytes.size() * 2);
    for (const auto byte : bytes)
    {
        result.push_back (digits[byte >> 4]);
        result.push_back (digits[byte & 15]);
    }
    return result;
}

inline nlohmann::json authoredPairReceiptJson (const AuthoredPairResult& artifact,
                                               bool resourceRetained = true)
{
    if (artifact.field == nullptr) return nullptr;
    const auto sourceJson = [] (const SourceFrameV2& source)
    {
        return nlohmann::json {
            { "clipId", source.clipId },
            { "sourceGeneration", source.sourceGeneration },
            { "sourceSha256", receiptHex (source.sourceContent) },
            { "frameIndex", source.frame.identity.frameIndex },
            { "timestampTicks", source.frame.timestamp.ticks },
            { "timestampTimescale", source.frame.timestamp.timescale },
            { "frameContentSha256", receiptHex (source.frame.identity.content) }
        };
    };
    const auto& field = artifact.field;
    const auto& output = field->receipt().motionVectors();
    nlohmann::json result {
        { "kind", "opticalFlowMotionVectors" },
        { "computed", true },
        { "resourceRetained", resourceRetained },
        { "nodeId", artifact.nodeId },
        { "graphClipId", artifact.sources.graphClipId },
        { "structuralRevision", artifact.sources.structuralRevision },
        { "snapshotGeneration", artifact.snapshotGeneration },
        { "displaySec", artifact.displaySec },
        { "sources", { sourceJson (artifact.sources.first),
                         sourceJson (artifact.sources.second) } },
        { "width", output.descriptor.extent.width },
        { "height", output.descriptor.extent.height },
        { "pixelFormat", "rg16f" },
        { "units", "pixelDisplacement" },
        { "direction", "firstToSecond" },
        { "coordinateOrigin", "imageTopLeft" },
        { "byteCount", output.byteCount },
        { "resourceIdentity", receiptHex (output.identity) },
        { "backendIdentity", receiptHex (field->receipt().backendImplementation()) },
        { "backendRevision", field->receipt().backendImplementationRevision() },
        { "submissionSerial", field->submissionSerial() }
    };
    if (resourceRetained)
        result["imageHandle"] = static_cast<std::uint64_t> (field->imageHandle());
    return result;
}
} // namespace videoopticalflow
