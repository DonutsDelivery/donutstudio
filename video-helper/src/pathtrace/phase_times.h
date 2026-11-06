#pragma once

// Wall-clock phase times of one Path Trace frame, the speed fields of
// pathTraceStats (path_trace_stats_reply.h). Every time is host steady_clock
// milliseconds. A GPU phase is timed between host calls that already wait for
// the device, so no phase adds a synchronisation of its own; where queued work
// from one phase finishes inside the next, the device code says so.
//
// Header only and free of the tracer's types, so the reply tests link nothing.

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace videohelper::pathtrace
{

enum class FramePhase : std::uint8_t
{
    // The draw lowered to the tracer's RenderScene: the fixture engine's setup
    // and scene copies plus ingest, materials and lights. The CPU BVH is not in it.
    SceneLower = 0,
    // The CPU BVH, built with a new RenderScene. Every device has one: the CPU
    // traces with it, and the depth pass reads it on any device.
    CpuBvhBuild,
    // The acceleration structure the tracing device traces: the CPU BVH on the
    // CPU, the GAS on OptiX, the device BVH on compute and Metal.
    AccelBuild,
    // Waiting for the device's kernel build and its lock before the first upload.
    PipelineWait,
    // Scene buffers from host to device. Zero when the device kept its scene.
    Upload,
    // The centre-ray depth and render outputs, on the CPU on every device.
    DepthPass,
    // Every sample pass of the frame.
    Trace,
    // The device film back to host memory.
    Readback,
    // The film resolved to the image's bytes.
    Resolve,
    Denoise,
    // The resolved image to the compositor's textures.
    ImageUpload,
    // The fixture engine's render call, end to end.
    Total,
    Count
};

constexpr std::size_t kFramePhaseCount = static_cast<std::size_t> (FramePhase::Count);

// The reply key of each phase.
constexpr const char* framePhaseKey (FramePhase phase) noexcept
{
    switch (phase)
    {
        case FramePhase::SceneLower: return "sceneLowerMs";
        case FramePhase::CpuBvhBuild: return "cpuBvhBuildMs";
        case FramePhase::AccelBuild: return "accelBuildMs";
        case FramePhase::PipelineWait: return "pipelineWaitMs";
        case FramePhase::Upload: return "uploadMs";
        case FramePhase::DepthPass: return "depthPassMs";
        case FramePhase::Trace: return "traceMs";
        case FramePhase::Readback: return "readbackMs";
        case FramePhase::Resolve: return "resolveMs";
        case FramePhase::Denoise: return "denoiseMs";
        case FramePhase::ImageUpload: return "imageUploadMs";
        case FramePhase::Total: return "totalMs";
        case FramePhase::Count: break;
    }
    return "unknownMs";
}

using PhaseClock = std::chrono::steady_clock;

inline double millisecondsBetween (PhaseClock::time_point start, PhaseClock::time_point end) noexcept
{
    return std::chrono::duration<double, std::milli> (end - start).count();
}

inline double millisecondsSince (PhaseClock::time_point start) noexcept
{
    return millisecondsBetween (start, PhaseClock::now());
}

struct FramePhaseTimes
{
    // NaN until measured. A phase left unmeasured names why in absent.
    std::array<double, kFramePhaseCount> ms = filled();
    std::array<const char*, kFramePhaseCount> absent {};
    // What the tracing device did with its acceleration structure this frame:
    // "build", "refit" or "reused". Null when no pass traced.
    const char* accelBuild = nullptr;

    bool has (FramePhase phase) const noexcept { return ! std::isnan (ms[index (phase)]); }
    double get (FramePhase phase) const noexcept { return ms[index (phase)]; }
    const char* absentReason (FramePhase phase) const noexcept { return absent[index (phase)]; }

    void set (FramePhase phase, double value) noexcept
    {
        ms[index (phase)] = value;
        absent[index (phase)] = nullptr;
    }

    // Adds to a measured phase, or measures it.
    void add (FramePhase phase, double value) noexcept
    {
        set (phase, has (phase) ? get (phase) + value : value);
    }

    // Names why phase was not measured. A measured phase keeps its time.
    void markAbsent (FramePhase phase, const char* reason) noexcept
    {
        if (! has (phase))
            absent[index (phase)] = reason;
    }

    // Folds in another frame part, such as a second Path Trace layer of one
    // export frame: measured times add, and a phase neither measured keeps a reason.
    void merge (const FramePhaseTimes& other) noexcept
    {
        for (std::size_t i = 0; i < kFramePhaseCount; ++i)
        {
            const auto phase = static_cast<FramePhase> (i);
            if (other.has (phase))
                add (phase, other.get (phase));
            else if (other.absent[i] != nullptr)
                markAbsent (phase, other.absent[i]);
        }
        if (other.accelBuild != nullptr)
            accelBuild = other.accelBuild;
    }

private:
    static constexpr std::size_t index (FramePhase phase) noexcept { return static_cast<std::size_t> (phase); }
    static std::array<double, kFramePhaseCount> filled() noexcept
    {
        std::array<double, kFramePhaseCount> values {};
        values.fill (std::numeric_limits<double>::quiet_NaN());
        return values;
    }
};

// Samples per second of a trace phase, or NaN when the phase is unmeasured or empty.
inline double samplesPerSecond (std::uint64_t samples, const FramePhaseTimes& phases) noexcept
{
    if (! phases.has (FramePhase::Trace) || ! (phases.get (FramePhase::Trace) > 0.0))
        return std::numeric_limits<double>::quiet_NaN();
    return static_cast<double> (samples) * 1000.0 / phases.get (FramePhase::Trace);
}

} // namespace videohelper::pathtrace
