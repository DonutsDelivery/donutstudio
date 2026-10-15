#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace videohelper::stabilization
{
// Independently authored CPU estimator. This format is private cache data,
// separate from project serialization and from vid.stab transform files.
constexpr uint32_t trackSchema = 2;
constexpr uint32_t estimatorRevision = 1;
constexpr size_t maximumFrames = 250000;
constexpr size_t maximumTrackBytes = 32 * 1024 * 1024;
using CancelCheck = std::function<bool()>;

struct SourceIdentity
{
    std::array<uint8_t, 32> sha256 {};
    int32_t streamIndex = -1;
    uint32_t width = 0, height = 0; // Visible extent, excluding coded padding.
    int32_t sarNumerator = 1, sarDenominator = 1;
    int32_t timeBaseNumerator = 0, timeBaseDenominator = 0;
    double inSeconds = 0, outSeconds = 0; // Exact requested [in, out) range.
};

struct FrameKey
{
    uint64_t ordinal = 0; // Contiguous presentation ordinal within the range.
    int64_t pts = 0;      // Source ticks, before timeline rebasing.
};

struct RigidTransform
{
    // q = R(angle) * p + (x,y), centered on the visible extent. X is in
    // display pixels, so sample aspect ratio participates in fitting/warping.
    double x = 0, y = 0, angle = 0;
};

enum class FrameStatus : uint32_t { anchor = 0, tracked = 1, uncertain = 2, cut = 3 };

struct TrackFrame
{
    FrameKey key;
    std::array<uint8_t, 32> graySha256 {}; // Visible Gray8 rows, no stride padding.
    RigidTransform camera; // Accumulated within this segment, angle unwrapped.
    double confidence = 0;
    uint32_t segment = 0;
    FrameStatus status = FrameStatus::anchor;
};

struct Track
{
    SourceIdentity source;
    uint32_t schema = trackSchema, estimator = estimatorRevision;
    std::vector<TrackFrame> frames;
};

struct GrayFrame
{
    const uint8_t* pixels = nullptr;
    uint32_t width = 0, height = 0;
    size_t strideBytes = 0, byteCount = 0;
};

// Owns only the previous image pyramid and bounded transform records. A failed
// or cancelled append does not admit a partial record. The caller owns decode,
// content hashing, file identity rechecks and atomic cache publication.
class Analyzer
{
public:
    explicit Analyzer (const SourceIdentity& source);
    ~Analyzer();
    Analyzer (Analyzer&&) noexcept;
    Analyzer& operator= (Analyzer&&) noexcept;
    Analyzer (const Analyzer&) = delete;
    Analyzer& operator= (const Analyzer&) = delete;
    bool append (const GrayFrame&, FrameKey, std::string& error,
                 const CancelCheck& cancelled = {});
    bool finish (Track& result, std::string& error) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

bool grayFrameDigest (const GrayFrame&, std::array<uint8_t, 32>&, std::string& error);
bool validateTrack (const Track&, std::string& error);
bool sameSource (const SourceIdentity&, const SourceIdentity&);
bool matchesFrame (const Track&, size_t index, FrameKey, std::string& error);
// Canonical little-endian bytes with SHA-256 over the entire record payload.
// Decode rejects invalid metadata, oversized counts, trailing bytes and any
// malformed record. Neither function reads or writes a filesystem path.
bool encodeTrack (const Track&, std::vector<uint8_t>&, std::string& error);
bool decodeTrack (const uint8_t*, size_t, Track&, std::string& error);

struct RenderPlan
{
    SourceIdentity source;
    std::vector<FrameKey> keys;
    std::vector<RigidTransform> outputToSource;
    unsigned smoothingRadius = 0;
    double zoom = 1;
};

// The released strength maps to round(strength * 60), clamped to 1..60.
// One zoom applies to every frame; excessive crop fails rather than exposing
// borders or silently weakening the requested smoothing.
bool makeRenderPlan (const Track&, double strength, RenderPlan&, std::string& error,
                     double maximumZoom = 1.35, const CancelCheck& cancelled = {});

enum class SampleType { u8, u16, f32 };
struct ConstPlane
{
    const void* pixels = nullptr;
    uint32_t width = 0, height = 0, channels = 1;
    size_t strideBytes = 0, byteCount = 0;
    SampleType sampleType = SampleType::u8;
};
struct MutablePlane
{
    void* pixels = nullptr;
    uint32_t width = 0, height = 0, channels = 1;
    size_t strideBytes = 0, byteCount = 0;
    SampleType sampleType = SampleType::u8;
};

// Preserves the input plane's extent/sample representation. Supports planar
// chroma through center-aligned scale from the visible full-frame extent.
// Float samples remain linear and unclipped. Input/output must not overlap.
// Output is unpublished scratch data until every plane/frame succeeds.
bool warpPlane (const RenderPlan&, size_t index, FrameKey, const ConstPlane&,
                const MutablePlane&, std::string& error,
                const CancelCheck& cancelled = {});
} // namespace videohelper::stabilization
