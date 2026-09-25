#pragma once

#include "geometry_core_backend.h"

#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace videohelper::geometry {

// A preview render thread and an export/probe thread own different graphics
// contexts, even when both contexts use the same physical GPU. Native geometry
// resources must remain with that context until its owner releases them.
struct GeometryExecutionCache final {
  struct Retained final {
    struct Entry final {
      std::uint64_t lastUse = 0;
      std::shared_ptr<const AdmittedPlanValue> admittedPlan;
      std::shared_ptr<const NativeGeometryExecution> execution;
      bool reserved = false;
      std::vector<videowire::geometry::spectrum::Follower> spectrumFollowers;
    };
    std::mutex mutex;
    std::map<PlanOwnerIdentity, Entry> owners;
    std::uint64_t clock = 0;
    std::uint64_t projectGeneration = 0;
    std::uint64_t helperGeneration = 0;
    std::uint64_t deviceGeneration = 0;
  };

  GeometryExecutionCache()
      : source(arbitgpu::nativeFixtureSceneBackend()), runtime(source) {}

  // Member order retires retained frames before their admission/resource owners.
  NativeGeometryCoreCapabilitySource source;
  GeometryCorePlanRuntime runtime;
  Retained retained;
};

inline std::unique_ptr<GeometryExecutionCache>& geometryExecutionCacheStorage() {
  static thread_local std::unique_ptr<GeometryExecutionCache> cache;
  return cache;
}

inline GeometryExecutionCache& geometryExecutionCache() {
  auto& cache = geometryExecutionCacheStorage();
  if (!cache) cache = std::make_unique<GeometryExecutionCache>();
  return *cache;
}

// Call after the last compositor lease is released and before destroying the
// current graphics context. Empty/early-failure paths do not allocate a cache.
inline void releaseGeometryExecutionCache() noexcept {
  geometryExecutionCacheStorage().reset();
  arbitgpu::nativeFixtureSceneBackend().releaseCachedProgramsForCurrentContext();
}

} // namespace videohelper::geometry
