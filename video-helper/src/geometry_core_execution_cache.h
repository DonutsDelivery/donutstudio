#pragma once

#include "geometry_core_backend.h"
#include "composed_timeline_geometry.h"

#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace videohelper::geometry {

// Named attribute lifetime (Store Attribute, Read Attribute). Pinned by
// video-helper/tests/named_attribute_lifetime_tests.cpp.
// 1. Transient per evaluation. A store's attribute lives on its output value and
//    is visible only to operations downstream in the same plan. validateOperationPlan
//    builds retainedAttributes from empty on every call (shared/GeometryCore.h) and
//    never writes back to the source. Nothing persists on the geometry between
//    frames. The retained cache below holds frames, not attributes.
// 2. Cache hit. A hit reuses the stored frame for the same owner, admitted plan
//    pointer, extent and color space, and only when the plan is not dynamic
//    (dynamicGeometry in visual_plan_executor.h). Runtime fields, spectrum fields,
//    score fields and imported animation are dynamic, so a hit never stands in for
//    an attribute that a fresh replay would compute differently.
// 3. Missing name. Read Attribute refuses with an error (lowerNamedAttribute and
//    the replay). It never defaults to a value or zero.
// 4. Same name. A store replaces the attribute with that name in place, including
//    its domain and type (materializeStoredAttribute). That is not a conflict
//    error. A read needs the exact domain and value type, or it refuses.
// 5. Restart and seek. A static attribute chain depends only on its admitted plan,
//    so restart, seek and preview versus export produce the same attributes.
//    Time enters only through runtime fields. Live spectrum history resets on seek
//    and loop generation (evaluateHistory in GeometrySpectrumField.h).
//
// This block lives here rather than at lowerNamedAttribute because
// GeometryFieldLowering.h is reached through GeometryCore.h, which about ten
// shared and video-helper files include directly. A comment edit there would
// rebuild all of them.

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
      : GeometryExecutionCache(arbitgpu::nativeFixtureSceneBackend()) {}
  explicit GeometryExecutionCache(arbitgpu::NativeFixtureSceneBackend& backend)
      : source(backend), runtime(source), familyEvaluator(runtime) {}

  static timelineevaluation::Limits productionFamilyEvaluationLimits() noexcept {
    timelineevaluation::Limits limits;
    // Match the released source-visit policy for these additional logical
    // passes. Neither allowance measures elapsed CPU or allocator/driver work.
    limits.maxWaterIdentityVisits = 10000000;
    limits.maxWaterRestPositionVisits = 10000000;
    limits.maxWaterMemoCopyElements = 10000000;
    limits.maxWaterColumnDiscoveryVisits = 10000000;
    return limits;
  }

  // Member order retires retained frames before their admission/resource owners.
  NativeGeometryCoreCapabilitySource source;
  GeometryCorePlanRuntime runtime;
  // Escaped scene aliases remain charged to this context's evaluator ledger.
  // This thread-local cache does not impose a process-wide memory/work cap.
  TimelineSceneFamilyEvaluator familyEvaluator;
  timelineevaluation::Limits familyEvaluationLimits = productionFamilyEvaluationLimits();
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
