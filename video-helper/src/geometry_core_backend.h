#pragma once

#include "geometry_core_admission.h"
#include "gpu_backend/backend.h"
#include <functional>
#include <limits>

namespace videorender { struct ParticleHistorySource; struct WaterCoupledFrame; class ParticlePointsBodyRequest; class ParticlePointsBodyFrame; }

namespace videohelper::geometry {
using PointsEvaluator = std::function<bool(videowire::geometry::StableId,
    const videowire::geometry::DeferredPointsSource&, const PlanOwnerIdentity&,
    videowire::geometry::ValueDescriptor&, std::string&)>;
using PointsBodyEvaluator = std::function<std::shared_ptr<const videorender::ParticlePointsBodyFrame>(
    const std::shared_ptr<const videorender::ParticlePointsBodyRequest>&,std::string&)>;
using ImportedGeometryEvaluator = std::function<bool(videowire::geometry::RetainedMeshData&, std::string&)>;
class NativeGeometryCoreCapabilitySource final
    : public BackendCapabilitySource {
public:
  explicit NativeGeometryCoreCapabilitySource(
      arbitgpu::NativeFixtureSceneBackend &backend) noexcept
      : backend_(backend) {}
  BackendCapabilities geometryCoreCapabilities() const override;

private:
  arbitgpu::NativeFixtureSceneBackend &backend_;
};

struct NativeGeometryExecution final {
  std::shared_ptr<const arbitgpu::NativeFixtureSceneResources> resources;
  std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame;
  std::shared_ptr<const std::uint8_t> ownerLease;
  std::vector<videowire::geometry::StableId> drawnInstanceIds;
  arbitgpu::NativeFixtureSceneStats stats {};
  std::uint64_t receipt = 0;
};

struct SpectrumEvaluation final {
  videowire::geometry::spectrum::Bands bands {};
  double timeSeconds = 0.0;
  bool liveFrameAvailable = true;
  std::uint64_t seekGeneration = 0, loopGeneration = 0;
  videowire::geometry::spectrum::FeaturesAt featuresAt;
  videowire::geometry::spectrum::FeaturesAt historyFeaturesAt;
  videowire::geometry::spectrum::SourceFeaturesAt sourceFeaturesAt;
  // Analyzer rms, peak and onset for the current frame, and the scalar readers
  // that match featuresAt, historyFeaturesAt and sourceFeaturesAt by frame time.
  videowire::geometry::spectrum::Scalars scalars {};
  videowire::geometry::spectrum::ScalarsAt scalarFeaturesAt;
  videowire::geometry::spectrum::ScalarsAt historyScalarFeaturesAt;
  videowire::geometry::spectrum::SourceScalarsAt sourceScalarFeaturesAt;
  std::vector<videowire::geometry::spectrum::Follower> followers;
};

std::optional<NativeGeometryExecution>
executeNativeGeometry(arbitgpu::NativeFixtureSceneBackend &backend,
                      const RuntimeAdmission &admission, std::uint32_t width,
                      std::uint32_t height, std::string &error,
                      bool diagnosticInstanceIdentityColors = false,
                      arbitgpu::NativeFixtureSceneRuntimeInputs runtimeInputs = {},
                      SpectrumEvaluation* spectrumEvaluation = nullptr,
                      const ImportedGeometryEvaluator& importedGeometry = {},
                      double timelineTimeSeconds = 0.0, const PointsEvaluator& pointsEvaluator = {},
                      const videorender::ParticleHistorySource* waterHistory = nullptr,
                      double waterProjectSeconds = std::numeric_limits<double>::quiet_NaN(),
                      const videorender::WaterCoupledFrame* coupledWater = nullptr,
                      const std::shared_ptr<const videorender::ParticlePointsBodyRequest>& pointsBodyRequest = {},
                      const PointsBodyEvaluator& pointsBodyEvaluator = {},
                      arbitgpu::NativeFixturePrepareUse prepareUse = arbitgpu::NativeFixturePrepareUse::Blocking);
// Counts the Preview prepares that returned pending on this thread. A pending
// result carries no value and no error, so a caller compares two readings of
// this count to tell a deferral from a failure.
std::uint64_t geometryCorePreviewPendingResultsOnThisThread() noexcept;
// For a Preview prepare on another fixture path (Solid 3D) that returned pending.
// It counts the same deferral, so the viewport defers the frame and records no error.
void noteGeometryCorePreviewPendingOnThisThread() noexcept;
} // namespace videohelper::geometry
