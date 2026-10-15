// audio_feature_parity_tests.cpp: live and baked Block B feature parity (P3 49).
//
// The live viewport drains the ring in blocks and keeps the newest frame. The
// baked paths (exporter audioFeatureAt, viewport previewMixFeatureAt) analyze the
// whole mix once and read frame videohelper::analysisFrameIndex(...). These
// checks feed one deterministic buffer both ways from the same reset point and
// require the same features at every block boundary. They also check that the
// shared mapping videorender::fillAudioFeatures gives the same result for a
// fresh and a reused destination, which is what the live and export sites use.
// The live drain's onset is merged across all hops of a tick by mergeDrainOnset,
// so multi-hop ticks are checked against the max of their baked frames.
//
// Tolerance: none. Both paths run the same pushSamples arithmetic over the same
// sample stream in the same frame order, so the results are bit-identical. This
// is the partition-independence guarantee in block_b_analyzer.h, which
// block_b_analyzer_tests.cpp checks with ==. A tolerance would hide a real
// divergence.
//
// Time mapping: after `fed` samples the analyzer has emitted floor(fed/kHop)
// frames, so the newest live frame is baked frame floor(fed/kHop) - 1. The
// query uses (fed + 0.5) / rate rather than fed / rate. The half sample keeps
// the floor away from float rounding at multiples of kHop and selects the same
// frame.
//
// Signal: silence, a unit impulse at sample 4096 (onset at frame 8, as in
// block_b_analyzer_tests.cpp), then a 750 Hz sine of amplitude 0.5 (period 64
// samples) from sample 8192 to the end.

#include "block_b_audio_features.h"
#include "mix_analyze.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace
{
using arbitblockb::BlockBAnalyzer;
using arbitblockb::FeatureFrame;
using videorender::AudioFeatures;

constexpr float kRate = 48000.0f;
constexpr std::size_t kTotal = 16384;
constexpr std::size_t kHopSamples = static_cast<std::size_t> (arbitblockb::kHop);
constexpr std::size_t kBandSamples = static_cast<std::size_t> (arbitblockb::kBandCount);
constexpr std::size_t kLiveBlock = 512;   // hop-sized drain
constexpr std::size_t kOddBlock = 333;    // not a divisor of kHop
constexpr std::size_t kDrainBlock = 4 * kHopSamples;  // four hops per drain, so ticks have several frames
constexpr std::size_t kNone = static_cast<std::size_t> (-1);

int failures = 0;

void check (bool condition, const char* message)
{
    if (! condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::vector<float> makeSignal()
{
    std::vector<float> x (kTotal, 0.0f);
    x[4096] = 1.0f;
    for (std::size_t i = 8192; i < kTotal; ++i)
    {
        const double phase = 2.0 * arbitblockb::kPi * static_cast<double> (i % 64) / 64.0;
        x[i] = static_cast<float> (0.5 * std::sin (phase));
    }
    return x;
}

// Result of feeding a buffer in fixed-size blocks, as a drain would.
struct LiveRun
{
    std::vector<FeatureFrame> frames;  // every emitted frame, in emission order
    std::vector<std::size_t> fed;      // samples fed after each block
    std::vector<std::size_t> newest;   // index into frames of the newest frame after each block, or kNone
};

LiveRun runLive (const std::vector<float>& x, std::size_t block)
{
    LiveRun run;
    BlockBAnalyzer analyzer (kRate);
    std::size_t consumed = 0;
    while (consumed < x.size())
    {
        const std::size_t n = std::min (block, x.size() - consumed);
        analyzer.pushSamples (x.data() + consumed, static_cast<int> (n), run.frames);
        consumed += n;
        run.fed.push_back (consumed);
        run.newest.push_back (run.frames.empty() ? kNone : run.frames.size() - 1);
    }
    return run;
}

// Exact equality of every frame field. Frames never hold NaN here, so == is safe.
bool sameFrame (const FeatureFrame& a, const FeatureFrame& b)
{
    return a.rms == b.rms && a.peak == b.peak && a.onset == b.onset
        && a.onsetAge == b.onsetAge && a.bands == b.bands
        && a.flux == b.flux && a.frameIndex == b.frameIndex
        && a.sampleIndex == b.sampleIndex;
}

// Exact equality of the visual feature fields, including onsetAge.
bool sameFeatures (const AudioFeatures& a, const AudioFeatures& b)
{
    return a.rms == b.rms && a.peak == b.peak && a.onset == b.onset
        && a.onsetAge == b.onsetAge && a.bands == b.bands;
}

// fillAudioFeatures must copy every field of each frame, and a reused
// destination must come out the same as a fresh one. The reused destination
// starts with 100 non-zero bands and a non-zero onset, so any field the
// function forgets to overwrite shows up.
void checkMapping (const std::vector<FeatureFrame>& frames)
{
    bool fieldsMatch = true;
    bool reuseMatches = true;
    AudioFeatures reused;
    reused.bands.assign (std::size_t (100), 0.25f);
    reused.onset = 0.5f;
    for (const FeatureFrame& f : frames)
    {
        AudioFeatures fresh;
        videorender::fillAudioFeatures (fresh, f);
        fieldsMatch = fieldsMatch && fresh.rms == f.rms && fresh.peak == f.peak
            && fresh.onset == f.onset && fresh.onsetAge == f.onsetAge
            && fresh.bands.size() == kBandSamples
            && std::equal (fresh.bands.begin(), fresh.bands.end(), f.bands.begin());
        videorender::fillAudioFeatures (reused, f);
        reuseMatches = reuseMatches && sameFeatures (reused, fresh);
    }
    check (fieldsMatch, "fillAudioFeatures copies rms, peak, onset, onsetAge and all 64 bands from the frame");
    check (reuseMatches, "fillAudioFeatures into a reused destination matches a fresh destination");
}

// A trigger on an earlier hop of the drain keeps the 1.0 pulse even though the
// newest frame has decayed. The envelope is merged by maximum, not by "any frame
// has onset", so a decayed tail never reads as a fresh pulse.
void checkDrainOnset()
{
    FeatureFrame trigger;
    trigger.onset = 1.0f;
    FeatureFrame decayed;
    decayed.onset = 0.9f;

    AudioFeatures kept;
    videorender::mergeDrainOnset (kept, { trigger, decayed });
    check (kept.onset == 1.0f, "a drain with a trigger on an earlier hop keeps onset 1.0");

    // Only decayed values, well after the onset. The drain's onset is the max of
    // the tick, and the previous value in the destination must not survive.
    FeatureFrame tail1;
    tail1.onset = 0.01f;
    FeatureFrame tail2;
    tail2.onset = 0.008f;

    AudioFeatures tailOnly;
    tailOnly.onset = 1.0f;
    videorender::mergeDrainOnset (tailOnly, { tail1, tail2 });
    check (tailOnly.onset == 0.01f, "a drain of decayed frames gets their max envelope, not 1.0");

    AudioFeatures empty;
    empty.onset = 1.0f;
    videorender::mergeDrainOnset (empty, {});
    check (empty.onset == 0.0f, "an empty drain sets onset to 0");
}

// The live per-tick merged onset against the baked onset. A one-frame tick must
// equal the baked frame at that project time. A multi-frame tick must equal the
// max of the baked frames it produced, which is what the viewport publishes.
void checkTickOnset (const std::vector<float>& x, const std::vector<FeatureFrame>& baked, std::size_t block, bool multiHop)
{
    const LiveRun live = runLive (x, block);
    const double rate = static_cast<double> (kRate);
    bool singleMatches = true;
    bool multiMatches = true;
    bool sawMulti = false;
    bool sawPulse = false;
    bool pulseKept = true;

    std::size_t start = 0;
    for (std::size_t b = 0; b < live.fed.size(); ++b)
    {
        const std::size_t end = live.newest[b] == kNone ? 0 : live.newest[b] + 1;
        if (end == start)
            continue;

        const std::vector<FeatureFrame> tick (
            live.frames.begin() + static_cast<std::ptrdiff_t> (start),
            live.frames.begin() + static_cast<std::ptrdiff_t> (end));
        AudioFeatures merged;
        merged.onset = 0.5f;  // a stale value must be overwritten
        videorender::mergeDrainOnset (merged, tick);

        float tickMax = 0.0f;
        for (std::size_t k = start; k < end; ++k)
            tickMax = std::max (tickMax, baked[k].onset);

        if (end - start == 1)
        {
            const double t = (static_cast<double> (live.fed[b]) + 0.5) / rate;
            const std::size_t idx = videohelper::analysisFrameIndex (baked.size(), rate, t);
            if (merged.onset != baked[idx].onset)
                singleMatches = false;
        }
        else
        {
            sawMulti = true;
            if (merged.onset != tickMax)
                multiMatches = false;
            // The trigger fired on an earlier hop of this tick and the newest frame has decayed.
            if (tickMax == 1.0f && baked[end - 1].onset < 1.0f)
            {
                sawPulse = true;
                if (merged.onset != 1.0f)
                    pulseKept = false;
            }
        }
        start = end;
    }

    check (singleMatches, "a one-frame drain tick merges to the baked onset at that project time");
    if (multiHop)
    {
        check (multiMatches, "a multi-frame drain tick merges to the max baked onset of the frames it produced");
        check (sawMulti, "the multi-hop block size produces drain ticks with several frames");
        check (sawPulse, "the scenario has a multi-frame tick where the trigger is followed by a decayed newest frame");
        check (pulseKept, "a multi-frame tick that holds the trigger keeps onset 1.0 while its newest frame has decayed");
    }
    else
    {
        check (! sawMulti, "a block of one hop never produces a multi-frame tick");
    }
}
} // namespace

int main()
{
    const std::vector<float> x = makeSignal();
    const double rate = static_cast<double> (kRate);

    // Baked path: one whole-mix pass, as the exporter does.
    BlockBAnalyzer bakedAnalyzer (kRate);
    const std::vector<FeatureFrame> baked =
        bakedAnalyzer.analyzeOffline (x.data(), static_cast<int> (x.size()));
    check (baked.size() == kTotal / kHopSamples, "the baked pass emits one frame per hop");

    // The scenario must exercise onset and a non-clamped onsetAge, or the
    // parity checks below would pass without covering them.
    bool sawOnset = false;
    bool sawAge = false;
    for (const FeatureFrame& f : baked)
    {
        sawOnset = sawOnset || f.onset == 1.0f;
        sawAge = sawAge || f.onsetAge < arbitblockb::kOnsetAgeMax;
    }
    check (sawOnset && sawAge, "the scenario contains an onset and a non-clamped onsetAge");

    // Live block order against baked order, at two partitions.
    const std::size_t blocks[] = { kLiveBlock, kOddBlock };
    for (std::size_t block : blocks)
    {
        const LiveRun live = runLive (x, block);
        bool framesMatch = live.frames.size() == baked.size();
        for (std::size_t k = 0; framesMatch && k < baked.size(); ++k)
            framesMatch = sameFrame (live.frames[k], baked[k]);
        check (framesMatch, "live block-order frames are bit-identical to the baked frames");

        bool atTime = true;
        bool sawNonClamped = false;
        for (std::size_t b = 0; b < live.fed.size(); ++b)
        {
            if (live.newest[b] == kNone)
                continue;
            const FeatureFrame& liveNewest = live.frames[live.newest[b]];
            const double t = (static_cast<double> (live.fed[b]) + 0.5) / rate;
            const std::size_t idx = videohelper::analysisFrameIndex (baked.size(), rate, t);

            if (! sameFrame (liveNewest, baked[idx]))
                atTime = false;

            // Visual features after the shared mapping, onsetAge included.
            AudioFeatures liveFeatures;
            AudioFeatures bakedFeatures;
            videorender::fillAudioFeatures (liveFeatures, liveNewest);
            videorender::fillAudioFeatures (bakedFeatures, baked[idx]);
            if (! sameFeatures (liveFeatures, bakedFeatures))
                atTime = false;
            if (liveFeatures.onsetAge < arbitblockb::kOnsetAgeMax)
                sawNonClamped = true;
        }
        check (atTime, "at every block boundary the newest live frame equals the baked frame at the same project time, onsetAge included");
        check (sawNonClamped, "the parity sweep covers a boundary with a non-clamped onsetAge");
    }

    checkMapping (baked);
    checkDrainOnset();
    checkTickOnset (x, baked, kLiveBlock, false);
    checkTickOnset (x, baked, kDrainBlock, true);

    if (failures == 0)
        std::cout << "audio feature parity: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
