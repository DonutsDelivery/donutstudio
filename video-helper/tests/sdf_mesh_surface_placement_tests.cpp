#include "../../shared/GeometryCore.h"
#include "../../shared/GeometryCoreTransport.h"

#include <cmath>
#include <cstdio>
#include <optional>
#include <string>
#include <variant>
#include <vector>

// SDF Surface into a Spectrum Instancer's Mesh Surface placement, evaluated at
// runtime: the admitted plan crosses the transport, the helper's operation replay
// materializes the SDF surface and the instances on it, and frame bands move
// each instance by its own band.
namespace {
using namespace videowire::geometry;

int failures = 0;

void check(bool condition, const std::string& message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++failures;
  }
}

constexpr double kRadius = 1.013;
constexpr StableId kSdf = 9701, kInstancer = 9891, kPrototype = 9850;

PortContract instanceContract() {
  PortContract contract;
  contract.carrier = CarrierKind::instances3D; contract.overflow = OverflowPolicy::reject;
  contract.maxInstances = 64;
  return contract;
}

std::optional<ValueDescriptor> sphere(std::string& error) {
  videowire::sdfsurface::Request request;
  request.cells = {10, 10, 10};
  request.maximumResidual = 0.06;
  videowire::SdfRecord record;
  record.stableId = 9007199254740001ull;
  record.operation = videowire::SdfOperation::sphere;
  record.parameterCount = 1;
  record.parameters[0] = kRadius;
  request.geometry.rootId = record.stableId;
  request.geometry.records = {record};
  PortContract contract;
  contract.carrier = CarrierKind::geometry3D; contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = 100000; contract.maxIndices = 600000; contract.maxAttributes = 3;
  contract.attributes = sdfSurfaceAttributes(9700);
  return lowerSdfSurface(9700, kSdf, 1, request, contract, error);
}

double length(const Vec3& v) {
  return std::sqrt(static_cast<double>(v.x) * v.x + static_cast<double>(v.y) * v.y + static_cast<double>(v.z) * v.z);
}

void sdfPlacementEvaluatesAtRuntime() {
  std::string error;
  PortContract meshContract;
  meshContract.carrier = CarrierKind::geometry3D; meshContract.overflow = OverflowPolicy::reject;
  meshContract.maxVertices = 64; meshContract.maxIndices = 256;
  const auto prototype = lowerMeshGenerator(kPrototype, kPrototype, 1, OperationCode::plane,
      {0.1f, 0.1f, 0.0f, 0.0f}, meshContract, error);
  const auto surface = prototype ? sphere(error) : std::nullopt;
  spectrum::Binding binding;
  binding.firstBand = 8; binding.lastBand = 11; binding.direction = {0.0f, 1.0f, 0.0f};
  // Arrangement 5 is Mesh Surface.
  const auto lowered = surface ? lowerSpectrumInstancer(*prototype, kInstancer, binding,
      {5.0f, 2.0f, 2.0f, 2.0f}, instanceContract(), error, &*surface) : std::nullopt;
  check(lowered.has_value(), "an SDF sphere lowers as a Mesh Surface Spectrum Instancer placement: " + error);
  if (!lowered) return;

  const auto admitted = admitValue(*lowered, instanceContract(), error);
  check(admitted.has_value(), "the SDF-placed instancer admits: " + error);
  if (!admitted) return;
  const auto bytes = encodeRuntimeValue(*admitted);
  const auto decoded = decodeRuntimeValue(bytes, instanceContract(), {}, {}, error);
  check(decoded.has_value() && encodeRuntimeValue(*decoded) == bytes,
      "the SDF-placed instancer crosses the runtime transport unchanged: " + error);
  if (!decoded) return;

  // The helper's replay of the decoded plan, as executeNativeGeometry runs it.
  MaterializedInstancePlan replay;
  check(validateOperationPlan(decoded->descriptor(), {}, &replay, error),
      "the runtime replays the SDF surface and its instancer: " + error);
  const auto& authored = std::get<InstancesData>(lowered->data).instances;
  const auto& instances = replay.instances.instances;
  check(instances.size() == 4 && authored.size() == 4, "one runtime instance per selected band");
  if (instances.size() != 4 || authored.size() != 4) return;
  const auto sdfMesh = replay.geometries.find(kSdf);
  const auto& loweredPositions = std::get<GeometryData>(surface->data).positions;
  bool remeshed = sdfMesh != replay.geometries.end() && sdfMesh->second.positions.size() == loweredPositions.size();
  for (std::size_t i = 0; remeshed && i < loweredPositions.size(); ++i)
    remeshed = equal(sdfMesh->second.positions[i], loweredPositions[i]);
  check(remeshed,
      "the runtime remeshes the SDF surface to the lowered triangles");
  check(replay.geometries.count(kPrototype) == 1, "the runtime materializes the instanced prototype");
  bool placed = true;
  for (std::size_t i = 0; i < 4; ++i) {
    const auto& t = instances[i].transform.translation;
    const auto& a = authored[i].transform.translation;
    placed = placed && instances[i].stableId == static_cast<StableId>(9 + i)
        && instances[i].sourceStableId == kPrototype
        && t.x == a.x && t.y == a.y && t.z == a.z && std::abs(length(t) - kRadius) < 0.1;
  }
  check(placed, "runtime instances keep band identities and sit on the SDF sphere where lowering put them");

  // Frame bands drive each instance along Y by its own band, as the native
  // backend applies a Spectrum Instancer binding.
  const auto& runtimeBinding = decoded->descriptor().spectrumFields.back();
  check(runtimeBinding.target == spectrum::Target::instanceTransform && runtimeBinding.coordinates.size() == 4,
      "the decoded binding targets the four instances");
  if (runtimeBinding.coordinates.size() != 4) return;
  spectrum::Bands bands {};
  for (std::size_t band = 0; band < bands.size(); ++band) bands[band] = 0.05f * static_cast<float>(band);
  spectrum::Follower follower;
  spectrum::Bands followed {};
  check(spectrum::evaluate(runtimeBinding, bands, 1.0, {}, follower, followed, error),
      "the instancer binding evaluates a frame: " + error);
  bool driven = true;
  for (std::size_t i = 0; i < 4; ++i) {
    const auto amount = spectrum::sample(runtimeBinding, followed, runtimeBinding.coordinates[i]);
    const auto& rest = instances[i].transform.translation;
    const Vec3 moved {rest.x + amount * runtimeBinding.direction[0], rest.y + amount * runtimeBinding.direction[1],
                      rest.z + amount * runtimeBinding.direction[2]};
    driven = driven && std::abs(amount - bands[8 + i]) < 1e-6f && moved.x == rest.x && moved.z == rest.z
        && std::abs(moved.y - rest.y - bands[8 + i]) < 1e-6f;
  }
  check(driven, "each SDF-placed instance rises by its own band, bands 8 to 11 in order");
}
}  // namespace

int main() {
  sdfPlacementEvaluatesAtRuntime();
  if (failures != 0) return 1;
  std::puts("SDF mesh surface placement runtime tests passed");
  return 0;
}
