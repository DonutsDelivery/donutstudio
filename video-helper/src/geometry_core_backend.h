#pragma once

#include "geometry_core_admission.h"
#include "gpu_backend/backend.h"
#include <functional>

namespace videohelper::geometry {
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
                      double timelineTimeSeconds = 0.0);
} // namespace videohelper::geometry
