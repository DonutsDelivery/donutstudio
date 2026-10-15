#pragma once

#include "particle_history.h"
#include "../../shared/GeometryCore.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace videorender
{
namespace waterreplay
{
inline constexpr double kMaximumSeconds = 10;

struct Impulse
{
    std::int64_t tick = 0, identity = 0;
    int kind = 0;
    double x = 0, y = 0, velocity = 0;
};

inline double identityY(int identity)
{
    // SplitMix64 preserves the signed canonical note identity without row state.
    auto hash = static_cast<std::uint64_t>(static_cast<std::int64_t>(identity));
    hash += 0x9e3779b97f4a7c15ull;
    hash = (hash ^ (hash >> 30)) * 0xbf58476d1ce4e5b9ull;
    hash = (hash ^ (hash >> 27)) * 0x94d049bb133111ebull;
    hash ^= hash >> 31;
    return double(hash >> 11) / 9007199254740991.0;
}

inline bool validAudioFrame(const arbitblockb::FeatureFrame& frame)
{
    const auto unit = [](float value) { return std::isfinite(value) && value >= 0 && value <= 1; };
    return unit(frame.rms) && unit(frame.peak) && unit(frame.onset)
        && std::isfinite(frame.onsetAge) && frame.onsetAge >= 0
        && frame.onsetAge <= arbitblockb::kOnsetAgeMax
        && std::isfinite(frame.flux) && frame.flux >= 0
        && std::all_of(frame.bands.begin(), frame.bands.end(), unit);
}
}

// Prepared state belongs to one project-zero replay request. It owns the
// admitted impulses and wave buffers, and retains no borrowed history reader.
class WaterSurfaceReplayState
{
public:
    bool prepare(const ParticleHistorySource& source,
        const videowire::geometry::OperationRecord& operation,
        const videowire::geometry::GeometryData& restMesh,
        const videowire::geometry::WaterSurfaceGrid& grid, double projectSeconds,
        std::string& error)
    {
        using namespace videowire::geometry;
        const auto refuse = [&](const char* reason) { error = reason; return false; };
        if (!std::isfinite(projectSeconds) || projectSeconds < 0 || projectSeconds > waterreplay::kMaximumSeconds)
            return refuse("Water replay requires project time from zero through 10 seconds; longer history is unavailable");
        WaterSurfaceGrid verifiedGrid;
        if (!validateWaterSurfaceGrid(restMesh, verifiedGrid, error)) return false;
        if (grid.columns != verifiedGrid.columns || grid.rows != verifiedGrid.rows
            || grid.spacingX != verifiedGrid.spacingX || grid.spacingY != verifiedGrid.spacingY)
            return refuse("Water replay grid does not match the retained rest mesh");
        if (!validateWaterSurfaceSettings(operation, grid, error)) return false;
        auto key = source.key;
        key.frame = 0; key.fps = kParticleReplayTicksPerSecond; key.beat = 0;
        if (!canonicalblockc::FrameProducer::admissible(key, source.score, 0, &error)) return false;

        const auto projectTick = std::llround(projectSeconds * kParticleReplayTicksPerSecond);
        const auto steps = projectTick / kParticleReplayStepTicks;
        const auto& controls = operation.field.values;
        std::vector<waterreplay::Impulse> impulses;
        const auto add = [&](double seconds, int kind, std::int64_t identity,
                             double x, double y, double velocity)
        {
            if (!std::isfinite(seconds) || !std::isfinite(velocity))
                return refuse("Water history produced a nonfinite impulse");
            // Check seconds before tick conversion. This also excludes a future
            // event when query rounding crosses a step by less than half a tick.
            if (seconds < 0 || seconds > projectSeconds) return true;
            const auto tick = std::llround(seconds * kParticleReplayTicksPerSecond);
            if (impulses.size() >= static_cast<std::size_t>(kParticleReplayMaxEvents))
                return refuse("Water replay exceeds 512 score/audio impulses; history cannot be truncated");
            impulses.push_back({tick, identity, kind, x, y, velocity});
            return true;
        };
        if (controls[2] > 0)
            for (const auto& note : source.score->notes)
            {
                if (note.muted || note.startBeat < 0) continue;
                const auto seconds = source.timeline.beatToSeconds(note.startBeat);
                // Note.velocity is MIDI 0..127 in mod_defs.h, as in Block C packing.
                if (!add(seconds, 0, note.id, std::clamp(double(note.midiNote) / 127, 0.0, 1.0),
                         waterreplay::identityY(note.id), controls[2] * std::clamp(double(note.velocity) / 127, 0.0, 1.0)))
                    return false;
            }
        if (controls[3] > 0)
        {
            if (!source.audioFrames || source.audioFrames->empty()
                || !std::isfinite(source.audioSampleRate) || source.audioSampleRate <= 0)
                return refuse("Water audio impulses require contiguous project-zero master Block B history");
            const auto& frames = *source.audioFrames;
            const auto hops = std::floor(projectSeconds * source.audioSampleRate / arbitblockb::kHop);
            if (!std::isfinite(hops) || hops > double(frames.size()))
                return refuse("Water master Block B history does not cover the requested project time");
            auto count = static_cast<std::size_t>(hops);
            // Compare endpoint seconds directly. ceil(seconds*rate/hop) can ask
            // for an extra frame at an exact hop because of multiplication rounding.
            if (double(count) * arbitblockb::kHop / source.audioSampleRate < projectSeconds) ++count;
            if (count > frames.size())
                return refuse("Water master Block B history does not cover the requested project time");
            // Validate the whole requested coverage before any simulation update.
            for (std::size_t i = 0; i < count; ++i)
            {
                const auto& frame = frames[i];
                if (i > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max() / arbitblockb::kHop - 1)
                    || frame.sampleIndex != static_cast<std::int64_t>(i + 1) * arbitblockb::kHop
                    || frame.frameIndex != static_cast<std::int64_t>(i) || !waterreplay::validAudioFrame(frame))
                    return refuse("Water master Block B history is malformed or not contiguous from project zero");
            }
            for (std::size_t i = 0; i < count; ++i)
            {
                const auto& frame = frames[i];
                if (frame.onset > 0 && frame.onsetAge == 0)
                {
                    const auto onsetSample = frame.sampleIndex - std::llround(
                        frame.onsetAge * source.audioSampleRate / arbitblockb::kHop) * arbitblockb::kHop;
                    if (!add(double(onsetSample) / source.audioSampleRate, 1, onsetSample, 0.5, 0.5,
                             controls[3] * std::clamp(double(frame.onset), 0.0, 1.0))) return false;
                }
            }
        }
        std::sort(impulses.begin(), impulses.end(), [](const auto& a, const auto& b)
        {
            if (a.tick != b.tick) return a.tick < b.tick;
            if (a.kind != b.kind) return a.kind < b.kind;
            return a.identity < b.identity;
        });

        WaterSurfaceReplayState next;
        const auto count = restMesh.positions.size();
        next.height_.assign(count, 0); next.velocity_.assign(count, 0);
        next.nextHeight_.resize(count); next.nextVelocity_.resize(count);
        next.dt_ = double(kParticleReplayStepTicks) / kParticleReplayTicksPerSecond;
        next.retention_ = 1 - controls[1] * next.dt_;
        next.acceleration_ = next.dt_ * controls[0] * controls[0];
        next.invX_ = 1 / (grid.spacingX * grid.spacingX);
        next.invY_ = 1 / (grid.spacingY * grid.spacingY);
        next.grid_ = grid; next.projectTick_ = projectTick; next.steps_ = steps;
        next.impulses_ = std::move(impulses); next.prepared_ = true;
        *this = std::move(next);
        error.clear(); return true;
    }

    bool prepared() const { return prepared_; }
    bool complete() const { return prepared_ && step_ == steps_; }
    std::int64_t requestedTicks() const { return projectTick_; }
    std::int64_t completedTicks() const { return step_ * kParticleReplayStepTicks; }
    std::int64_t stepCount() const { return steps_; }
    const videowire::geometry::WaterSurfaceGrid& grid() const { return grid_; }
    const std::vector<double>& height() const { return height_; }
    const std::vector<double>& velocity() const { return velocity_; }

    // An impulse belongs to [step, step+1). Only completed intervals run,
    // so neither a boundary impulse nor an unfinished interval affects rest.
    bool advance(std::string& error)
    {
        const auto refuse = [&](const char* reason) { error = reason; return false; };
        if (!prepared_) return refuse("Water step requires a prepared replay request");
        if (step_ >= steps_) return refuse("Water step exceeds the prepared completed-interval boundary");
        const auto& grid = grid_;
        const auto& impulses = impulses_;
        const auto step = step_;
        auto event = event_;
        const auto& height = height_;
        auto& nextHeight = nextHeight_;
        // Stage impulses in the existing scratch velocity buffer. Each stencil
        // reads only its own velocity, so the original arithmetic is unchanged.
        // Visible height, velocity and impulse cursor commit after all checks.
        nextVelocity_ = velocity_;
        auto& velocity = nextVelocity_;
        auto& nextVelocity = nextVelocity_;
        const double dt = dt_, retention = retention_, acceleration = acceleration_;
        const double invX = invX_, invY = invY_;
        while (event < impulses.size() && impulses[event].tick / kParticleReplayStepTicks == step)
        {
            const auto& impulse = impulses[event++];
            const double x = impulse.x * (grid.columns - 1), y = impulse.y * (grid.rows - 1);
            const auto x0 = static_cast<std::uint32_t>(std::floor(x)), y0 = static_cast<std::uint32_t>(std::floor(y));
            const auto x1 = std::min(x0 + 1, grid.columns - 1), y1 = std::min(y0 + 1, grid.rows - 1);
            const double fx = x - x0, fy = y - y0;
            velocity[y0 * grid.columns + x0] += impulse.velocity * (1 - fx) * (1 - fy);
            velocity[y0 * grid.columns + x1] += impulse.velocity * fx * (1 - fy);
            velocity[y1 * grid.columns + x0] += impulse.velocity * (1 - fx) * fy;
            velocity[y1 * grid.columns + x1] += impulse.velocity * fx * fy;
        }
        for (std::uint32_t y = 0; y < grid.rows; ++y)
            for (std::uint32_t x = 0; x < grid.columns; ++x)
            {
                const auto i = y * grid.columns + x;
                // Clamped neighbors give zero normal gradient at each edge.
                const auto left = x == 0 ? i : i - 1, right = x + 1 == grid.columns ? i : i + 1;
                const auto below = y == 0 ? i : i - grid.columns, above = y + 1 == grid.rows ? i : i + grid.columns;
                const auto laplacian = (height[left] - 2 * height[i] + height[right]) * invX
                    + (height[below] - 2 * height[i] + height[above]) * invY;
                nextVelocity[i] = retention * (velocity[i] + acceleration * laplacian);
                nextHeight[i] = height[i] + dt * nextVelocity[i];
                if (!std::isfinite(nextVelocity[i]) || !std::isfinite(nextHeight[i]))
                    return refuse("Water replay overflowed; no partial surface was published");
            }
        height_.swap(nextHeight_); velocity_.swap(nextVelocity_);
        event_ = event; ++step_;
        error.clear(); return true;
    }

    bool applyBodyVelocity(const std::vector<double>& impulseVelocity, std::string& error)
    {
        if (!prepared_ || impulseVelocity.size()!=velocity_.size())
        { error="Water contact velocity requires the exact prepared vertex count"; return false; }
        nextVelocity_=velocity_;
        for (std::size_t i=0;i<velocity_.size();++i) {
            nextVelocity_[i]+=impulseVelocity[i];
            if (!std::isfinite(impulseVelocity[i]) || !std::isfinite(nextVelocity_[i]))
            { error="Water contact velocity overflowed; no partial impulse was committed"; return false; }
        }
        velocity_.swap(nextVelocity_); error.clear(); return true;
    }

    // Publication consumes a complete request. An unfinished or refused
    // request cannot replace the caller's previously published surface.
    bool publishHeight(std::vector<double>& localZOffsets, std::string& error)
    {
        if (!complete())
        { error = "Water height publication requires a complete prepared replay"; return false; }
        localZOffsets = std::move(height_);
        *this = WaterSurfaceReplayState{};
        error.clear(); return true;
    }

private:
    bool prepared_ = false;
    std::int64_t projectTick_ = 0, step_ = 0, steps_ = 0;
    std::size_t event_ = 0;
    videowire::geometry::WaterSurfaceGrid grid_;
    std::vector<waterreplay::Impulse> impulses_;
    std::vector<double> height_, velocity_, nextHeight_, nextVelocity_;
    double dt_ = 0, retention_ = 0, acceleration_ = 0, invX_ = 0, invY_ = 0;
};

// One completed body/water request, borrowed by geometry consumers at the
// exact current project time. Historical queries require another full request.
struct WaterCoupledFrame final
{
    videowire::geometry::OperationRecord operation;
    videowire::geometry::GeometryData restMesh;
    videowire::geometry::WaterSurfaceGrid grid;
    double projectSeconds = 0;
    std::vector<double> heights;

    bool sample(const videowire::geometry::OperationRecord& requested,
        const videowire::geometry::GeometryData& mesh,
        const videowire::geometry::WaterSurfaceGrid& requestedGrid, double seconds,
        std::vector<double>& output, std::string& error) const
    {
        if (seconds!=projectSeconds || requested.code!=operation.code
            || requested.stableId!=operation.stableId || requested.inputStableId!=operation.inputStableId
            || requested.outputStableId!=operation.outputStableId
            || requested.secondaryInputStableId!=operation.secondaryInputStableId
            || requested.parameters!=operation.parameters || !(requested.field==operation.field)
            || !videowire::geometry::equal(requested.transform,operation.transform)
            || requested.retainedMesh!=operation.retainedMesh || requested.sdfSurface!=operation.sdfSurface
            || requestedGrid.columns!=grid.columns || requestedGrid.rows!=grid.rows
            || requestedGrid.spacingX!=grid.spacingX || requestedGrid.spacingY!=grid.spacingY
            || !videowire::geometry::equal(mesh,restMesh) || heights.size()!=mesh.positions.size()
            || !std::all_of(heights.begin(),heights.end(),[](double value) { return std::isfinite(value); }))
        { error="Completed water frame does not match the exact coupled mesh and project time"; return false; }
        output=heights; error.clear(); return true;
    }
};

// The caller supplies PROJECT-ZERO seconds, not clip-local mesh time. Every
// request starts at authored rest at zero; no frame-order state is retained.
inline bool replayWaterSurface(const ParticleHistorySource& source,
    const videowire::geometry::OperationRecord& operation,
    const videowire::geometry::GeometryData& restMesh,
    const videowire::geometry::WaterSurfaceGrid& grid, double projectSeconds,
    std::vector<double>& localZOffsets, std::string& error)
{
    if (operation.code==videowire::geometry::OperationCode::fieldWaterHeight && operation.field.values[4]>0) {
        error="Water body coupling requires the coupled frame coordinator"; return false;
    }
    WaterSurfaceReplayState replay;
    if (!replay.prepare(source, operation, restMesh, grid, projectSeconds, error)) return false;
    while (!replay.complete()) if (!replay.advance(error)) return false;
    return replay.publishHeight(localZOffsets, error);
}

// Borrowed history is consumed synchronously during the owner's frame replay.
// The scene anchor is the exact seconds passed to the mesh evaluator.
inline videowire::geometry::RuntimeFieldEvaluation waterSurfaceFrameEvaluation(
    const ParticleHistorySource* history, double projectSeconds, double sceneSeconds,
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> frame,
    const WaterCoupledFrame* coupled = nullptr)
{
    videowire::geometry::RuntimeFieldEvaluation evaluation;
    evaluation.timelineSeconds = sceneSeconds;
    evaluation.waterHeightAt = [history, projectSeconds, sceneSeconds, coupled, frame=std::move(frame)](const auto& operation,
        const auto& restMesh, const auto& grid, double requestedSeconds,
        auto& heights, std::string& error)
    {
        if (!history)
        {
            error = "Water surface frame requires its owned project-zero history";
            return false;
        }
        if (!std::isfinite(projectSeconds) || !std::isfinite(sceneSeconds)
            || !std::isfinite(requestedSeconds))
        {
            error = "Water surface frame requires finite project and scene time anchors";
            return false;
        }
        // Geometry admission and Block C have separate generation authorities.
        // The current immutable Block C frame owns history identity and time.
        if (!canonicalblockc::valid(frame))
        {
            error = "Water replay requires its current canonical Block C frame";
            return false;
        }
        auto expected = history->key;
        expected.fps = frame->key.fps;
        const double frameAtProjectTime = projectSeconds * expected.fps;
        if (!std::isfinite(frameAtProjectTime) || frameAtProjectTime < 0
            || frameAtProjectTime >= static_cast<double>(std::numeric_limits<std::int64_t>::max()))
        {
            error = "Water project time exceeds the canonical frame clock";
            return false;
        }
        expected.frame = std::llround(frameAtProjectTime);
        // FrameOwner packs the beat at the rounded value-grid frame. Water
        // still replays at the exact project/scene anchors below.
        expected.beat = static_cast<double>(static_cast<float>(history->timeline.secondsToBeat(
            static_cast<double>(expected.frame) / expected.fps)));
        if (!(expected == frame->key))
        {
            error = "Water history and project time do not match the current canonical Block C frame";
            return false;
        }
        const double projectQuery = projectSeconds + (requestedSeconds - sceneSeconds);
        if (operation.code==videowire::geometry::OperationCode::fieldWaterHeight && operation.field.values[4]>0) {
            if (!coupled) { error="Water body coupling requires the coupled frame coordinator"; return false; }
            return coupled->sample(operation,restMesh,grid,projectQuery,heights,error);
        }
        return replayWaterSurface(*history, operation, restMesh, grid, projectQuery, heights, error);
    };
    return evaluation;
}

}
