#pragma once

#include "geometry_core_admission.h"
#include "gpu_backend/backend.h"

namespace videohelper::geometry {
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

std::optional<NativeGeometryExecution>
executeNativeGeometry(arbitgpu::NativeFixtureSceneBackend &backend,
                      const RuntimeAdmission &admission, std::uint32_t width,
                      std::uint32_t height, std::string &error,
                      bool diagnosticInstanceIdentityColors = false,
                      arbitgpu::NativeFixtureSceneRuntimeInputs runtimeInputs = {});
} // namespace videohelper::geometry
