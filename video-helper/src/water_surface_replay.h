#pragma once

#include "particle_history.h"
#include "../../shared/GeometryCore.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <map>
#include <mutex>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace videorender
{
namespace waterreplay
{
// The tick clock stays exact in a double up to here, about ten years of project
// time. This is the clock's range and not a length limit.
inline constexpr double kMaximumClockSeconds = 9007199254740991.0 / kParticleReplayTicksPerSecond;

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
        if (!std::isfinite(projectSeconds) || projectSeconds < 0 || projectSeconds > waterreplay::kMaximumClockSeconds)
            return refuse("Water replay requires a project time from zero that its tick clock can represent");
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
        try {
        const auto add = [&](double seconds, int kind, std::int64_t identity,
                             double x, double y, double velocity)
        {
            if (!std::isfinite(seconds) || !std::isfinite(velocity))
                return refuse("Water history produced a nonfinite impulse");
            // Check seconds before tick conversion. This also excludes a future
            // event when query rounding crosses a step by less than half a tick.
            if (seconds < 0 || seconds > projectSeconds) return true;
            const auto tick = std::llround(seconds * kParticleReplayTicksPerSecond);
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
        } catch (const std::bad_alloc&) {
            error = "Water replay could not allocate " + std::to_string(impulses.size() + 1) + " score/audio impulses";
            return false;
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

    // A completed-step snapshot. The steps before it consumed only the impulses
    // earlier than its boundary, and every request reaching past that boundary
    // lists those impulses first, in the same order.
    struct Checkpoint final
    {
        std::int64_t step = 0;
        std::size_t event = 0;
        std::vector<double> height, velocity;
        std::uint64_t consumed = 0;   // hash of the impulses before event
    };
    Checkpoint checkpoint() const
    {
        prefixHash(event_);
        return {step_, event_, height_, velocity_, hash_};
    }

    // Continues a freshly prepared request from a snapshot of the same simulation.
    bool resume(const Checkpoint& saved, std::string& error)
    {
        const auto stepOf = [](const waterreplay::Impulse& impulse) { return impulse.tick / kParticleReplayStepTicks; };
        if (!prepared_ || step_ != 0 || saved.step < 0 || saved.step > steps_
            || saved.height.size() != height_.size() || saved.velocity.size() != velocity_.size()
            || saved.event > impulses_.size()
            || (saved.event > 0 && stepOf(impulses_[saved.event - 1]) >= saved.step)
            || (saved.event < impulses_.size() && stepOf(impulses_[saved.event]) < saved.step)
            || (prefixHash(saved.event), hash_ != saved.consumed))
        { error = "Water checkpoint does not match this replay request"; return false; }
        step_ = saved.step; event_ = saved.event; height_ = saved.height; velocity_ = saved.velocity;
        error.clear(); return true;
    }

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
    // FNV-1a over every impulse field, extended as the prefix grows, so a snapshot
    // resumes only a request whose score and audio impulses agree up to it.
    void prefixHash(std::size_t events) const
    {
        if (events < hashed_) { hashed_ = 0; hash_ = 14695981039346656037ull; }
        const auto mix = [&](const void* data, std::size_t size)
        {
            const auto* bytes = static_cast<const unsigned char*>(data);
            for (std::size_t i = 0; i < size; ++i) { hash_ ^= bytes[i]; hash_ *= 1099511628211ull; }
        };
        for (; hashed_ < events; ++hashed_)
        {
            const auto& impulse = impulses_[hashed_];
            mix(&impulse.tick, sizeof impulse.tick); mix(&impulse.identity, sizeof impulse.identity);
            mix(&impulse.kind, sizeof impulse.kind); mix(&impulse.x, sizeof impulse.x);
            mix(&impulse.y, sizeof impulse.y); mix(&impulse.velocity, sizeof impulse.velocity);
        }
    }
    mutable std::size_t hashed_ = 0;
    mutable std::uint64_t hash_ = 14695981039346656037ull;
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
// Completed-step snapshots of uncoupled Water, one per simulated second, so a
// frame anywhere in the project resumes from the nearest earlier second instead
// of replaying from zero. Memory grows with the project length the user has
// visited and the grid's vertex count; a changed plan, score or source replaces
// the operation's entry.
class WaterSurfaceCheckpointStore final
{
public:
    static constexpr std::int64_t kIntervalSteps = kParticleReplayTicksPerSecond / kParticleReplayStepTicks;

    bool find(const ParticleHistorySource& source, const videowire::geometry::OperationRecord& operation,
              const videowire::geometry::GeometryData& restMesh, const videowire::geometry::WaterSurfaceGrid& grid,
              std::int64_t step, WaterSurfaceReplayState::Checkpoint& checkpoint)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto at = entries_.find(operation.stableId);
        if (at == entries_.end() || !at->second.same(source, operation, restMesh, grid)) return false;
        auto latest = at->second.saved.upper_bound(step);
        if (latest == at->second.saved.begin()) return false;
        checkpoint = std::prev(latest)->second; ++resumes_;
        return true;
    }

    void record(const ParticleHistorySource& source, const videowire::geometry::OperationRecord& operation,
                const videowire::geometry::GeometryData& restMesh, const videowire::geometry::WaterSurfaceGrid& grid,
                const WaterSurfaceReplayState& state)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& entry = entries_[operation.stableId];
        if (!entry.same(source, operation, restMesh, grid))
            entry = Entry::from(source, operation, restMesh, grid);
        entry.saved[state.checkpoint().step] = state.checkpoint(); ++recorded_;
    }

    std::size_t resumes() const { std::lock_guard<std::mutex> lock(mutex_); return resumes_; }
    std::size_t recorded() const { std::lock_guard<std::mutex> lock(mutex_); return recorded_; }

private:
    struct Entry final
    {
        const void* score = nullptr; const void* audio = nullptr; double sampleRate = 0;
        canonicalblockc::FrameKey key {};
        std::array<double, 8> values {};
        std::vector<videowire::geometry::Vec3> positions;
        videowire::geometry::WaterSurfaceGrid grid;
        std::map<std::int64_t, WaterSurfaceReplayState::Checkpoint> saved;

        static Entry from(const ParticleHistorySource& source, const videowire::geometry::OperationRecord& operation,
                          const videowire::geometry::GeometryData& restMesh, const videowire::geometry::WaterSurfaceGrid& grid)
        {
            Entry entry;
            entry.score = source.score.get(); entry.audio = source.audioFrames; entry.sampleRate = source.audioSampleRate;
            entry.key = source.key; entry.values = operation.field.values;
            entry.positions = restMesh.positions; entry.grid = grid;
            return entry;
        }

        // The frame, beat, seek, loop and fps generations move with playback and
        // do not change the simulation.
        bool same(const ParticleHistorySource& source, const videowire::geometry::OperationRecord& operation,
                  const videowire::geometry::GeometryData& restMesh, const videowire::geometry::WaterSurfaceGrid& other) const
        {
            const auto& k = source.key;
            if (score != source.score.get() || audio != source.audioFrames || sampleRate != source.audioSampleRate
                || key.carrierVersion != k.carrierVersion || key.projectGeneration != k.projectGeneration
                || key.sourceGeneration != k.sourceGeneration || key.helperGeneration != k.helperGeneration
                || key.backendGeneration != k.backendGeneration || key.deviceGeneration != k.deviceGeneration
                || key.scoreGeneration != k.scoreGeneration || key.beatMapGeneration != k.beatMapGeneration
                || values != operation.field.values || grid.columns != other.columns || grid.rows != other.rows
                || grid.spacingX != other.spacingX || grid.spacingY != other.spacingY
                || positions.size() != restMesh.positions.size()) return false;
            for (std::size_t i = 0; i < positions.size(); ++i)
                if (positions[i].x != restMesh.positions[i].x || positions[i].y != restMesh.positions[i].y
                    || positions[i].z != restMesh.positions[i].z) return false;
            return true;
        }
    };

    mutable std::mutex mutex_;
    std::map<videowire::geometry::StableId, Entry> entries_;
    std::size_t resumes_ = 0, recorded_ = 0;
};

inline WaterSurfaceCheckpointStore& waterSurfaceCheckpoints()
{
    static WaterSurfaceCheckpointStore store;
    return store;
}

inline bool replayWaterSurface(const ParticleHistorySource& source,
    const videowire::geometry::OperationRecord& operation,
    const videowire::geometry::GeometryData& restMesh,
    const videowire::geometry::WaterSurfaceGrid& grid, double projectSeconds,
    std::vector<double>& localZOffsets, std::string& error,
    WaterSurfaceCheckpointStore* checkpoints = &waterSurfaceCheckpoints())
{
    if (operation.code==videowire::geometry::OperationCode::fieldWaterHeight && operation.field.values[4]>0) {
        error="Water body coupling requires the coupled frame coordinator"; return false;
    }
    WaterSurfaceReplayState replay;
    if (!replay.prepare(source, operation, restMesh, grid, projectSeconds, error)) return false;
    WaterSurfaceReplayState::Checkpoint saved;
    if (checkpoints && checkpoints->find(source, operation, restMesh, grid, replay.stepCount(), saved)
        && !replay.resume(saved, error)) return false;
    while (!replay.complete())
    {
        if (!replay.advance(error)) return false;
        if (checkpoints && replay.checkpoint().step % WaterSurfaceCheckpointStore::kIntervalSteps == 0)
            checkpoints->record(source, operation, restMesh, grid, replay);
    }
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
