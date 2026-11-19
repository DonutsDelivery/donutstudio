#pragma once

// Which path the open sea took on the last Preview frame: moved by the GPU wave
// pass or replayed on the CPU, and why. Kept apart from the geometry headers so
// the viewport_info reply can read it without including them.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <optional>

namespace videohelper::geometry {
// The sea path each Preview frame took, kept for video_viewport_info, and one log
// line whenever a clip's path or reason changes, so a run shows which one ran.
struct SurfaceWavePathNote final {
  bool gpu = false;
  const char* reason = nullptr;
  std::uint64_t gpuFrames = 0, cpuFrames = 0;
};
inline std::mutex& surfaceWavePathMutex() { static std::mutex mutex; return mutex; }
inline std::optional<SurfaceWavePathNote>& lastSurfaceWavePath() {
  static std::optional<SurfaceWavePathNote> note;
  return note;
}
inline void noteSurfaceWavePath(int clipId, bool gpu, const char* reason) {
  const char* shown = gpu ? "" : reason ? reason : "unknown";
  bool changed = false;
  {
    std::lock_guard<std::mutex> lock(surfaceWavePathMutex());
    auto& note = lastSurfaceWavePath();
    changed = !note || note->gpu != gpu || std::strcmp(note->reason ? note->reason : "", shown) != 0;
    if (!note) note.emplace();
    note->gpu = gpu;
    note->reason = shown;
    ++(gpu ? note->gpuFrames : note->cpuFrames);
  }
  if (changed)
    std::fprintf(stderr, "[gl] surface-wave path clip=%d path=%s%s%s\n", clipId, gpu ? "gpu" : "cpu-replay",
        gpu ? "" : " reason=", shown);
}
inline std::optional<SurfaceWavePathNote> surfaceWavePath() {
  std::lock_guard<std::mutex> lock(surfaceWavePathMutex());
  return lastSurfaceWavePath();
}
} // namespace videohelper::geometry
