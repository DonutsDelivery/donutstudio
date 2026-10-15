#pragma once

#include "media.h"
#include "render_snapshot.h"

#include <array>
#include <memory>
#include <string>

namespace videoopticalflow
{
// A decoder-owned RGBA8 receipt. pixelSha256 covers these exact owned bytes;
// it is not the V2 linear RGBA16F frame-content identity.
struct DecodedSourceReceipt
{
    std::uint32_t clipId = 0;
    std::uint64_t sourceGeneration = 0;
    double sourceTimeSec = 0.0;
    std::string sourceSha256;
    std::string pixelSha256;
    std::uint64_t presentationOrdinal = 0;
    DecodedFrame frame;
};

struct DecodedPairReceipt
{
    std::uint64_t snapshotGeneration = 0;
    std::array<DecodedSourceReceipt, 2> sources;
};

// The plan and ordered pair must already have passed snapshot normalization.
// This rechecks their exact binding against the current catalog and segments.
std::string resolveDecodedPair (
    const videowire::CompiledVisualLayerPlan& plan,
    const opticalflowoperation::SourcePair& orderedPair,
    const std::vector<videowire::OpticalFlowSourceBinding>& catalog,
    const std::vector<videowire::RenderSegment>& segments,
    std::uint64_t snapshotGeneration, double displaySec,
    int maxW, int maxH, std::shared_ptr<const DecodedPairReceipt>& out);
} // namespace videoopticalflow
