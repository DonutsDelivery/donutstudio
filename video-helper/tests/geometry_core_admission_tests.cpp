#include "../src/geometry_core_admission.h"
#include "../src/glb_scene_adapter.h"
#include "../src/mix_analyze.h"
#include "../../shared/GeometryAudioDeformer.h"
#include "point_distribution_fixture.h"
#include "particle_points_fixture.h"
#include "spectral_history_fixture.h"
#include "constructed_field_fixture.h"
#include "instance_appearance_fixture.h"
#include "instance_material_fields_fixture.h"
#include "reactive_surface_fixture.h"
#include "reactive_frame_fixture.h"
#include "material_table_fixture.h"
#include "timeline_curve_fixture.h"
#include "water_surface_contract_fixture.h"
#include "water_surface_replay_fixture.h"
#include "water_replay_source_budget_fixture.h"
#include "ordered_audio_geometry_fixture.h"
#include "imported_geometry_fixture.h"
#include "spectrum_field_domains_fixture.h"
#include "spectrum_instancer_layouts_fixture.h"
#include "score_instance_field_fixture.h"
#include "../../shared/GeometryCoreScene.h"
#include "../../shared/ReactiveCharacterAsset.h"

#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>
#if !defined(_WIN32)
#include <pthread.h>
#endif


#if defined(ARBIT_ALLOCATION_FAILURE_TEST_HOOKS)
#include "../../shared/tests/AllocationFailureTestSupport.h"
#include <atomic>
#include <cstdlib>
#include <new>
#include <thread>
#if defined(_WIN32)
#include <malloc.h>
#endif

namespace {
namespace allocationtest = arbit::allocationfailuretest;

void* geometryAllocateOrdinary(std::size_t size, allocationtest::Family family) {
  const auto requestedSize = size;
  if (size == 0) size = 1;
  allocationtest::AllocationObservation observation(
      family, requestedSize, size, 0, alignof(std::max_align_t));
  if (observation.selected()) throw std::bad_alloc();
  for (;;) {
    if (void* memory = std::malloc(size)) {
      observation.succeeded(memory);
      return memory;
    }
    const auto handler = std::get_new_handler();
    if (handler == nullptr) throw std::bad_alloc();
    handler();
  }
}

void* geometryAllocateAligned(std::size_t size, std::size_t alignment,
                              allocationtest::Family family) {
  const auto requestedSize = size;
  const auto requestedAlignment = alignment;
  if (alignment == 0 || (alignment & (alignment - 1)) != 0)
    throw std::bad_alloc();
#if !defined(_WIN32)
  if (alignment < sizeof(void*)) alignment = sizeof(void*);
#endif
  if (size == 0) size = 1;
  allocationtest::AllocationObservation observation(
      family, requestedSize, size, requestedAlignment, alignment);
  if (observation.selected()) throw std::bad_alloc();
  for (;;) {
    void* memory = nullptr;
#if defined(_WIN32)
    memory = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&memory, alignment, size) != 0) memory = nullptr;
#endif
    if (memory != nullptr) {
      observation.succeeded(memory);
      return memory;
    }
    const auto handler = std::get_new_handler();
    if (handler == nullptr) throw std::bad_alloc();
    handler();
  }
}

void geometryFreeAligned(void* memory) noexcept {
#if defined(_WIN32)
  _aligned_free(memory);
#else
  std::free(memory);
#endif
}
} // namespace

void* operator new(std::size_t size) {
  return geometryAllocateOrdinary(size, allocationtest::Family::scalar);
}
void* operator new[](std::size_t size) {
  return geometryAllocateOrdinary(size, allocationtest::Family::array);
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  try { return geometryAllocateOrdinary(size, allocationtest::Family::scalarNothrow); }
  catch (...) { return nullptr; }
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
  try { return geometryAllocateOrdinary(size, allocationtest::Family::arrayNothrow); }
  catch (...) { return nullptr; }
}
void* operator new(std::size_t size, std::align_val_t alignment) {
  return geometryAllocateAligned(size, static_cast<std::size_t>(alignment),
                                  allocationtest::Family::alignedScalar);
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
  return geometryAllocateAligned(size, static_cast<std::size_t>(alignment),
                                  allocationtest::Family::alignedArray);
}
void* operator new(std::size_t size, std::align_val_t alignment,
                   const std::nothrow_t&) noexcept {
  try { return geometryAllocateAligned(size, static_cast<std::size_t>(alignment),
                                        allocationtest::Family::alignedScalarNothrow); }
  catch (...) { return nullptr; }
}
void* operator new[](std::size_t size, std::align_val_t alignment,
                     const std::nothrow_t&) noexcept {
  try { return geometryAllocateAligned(size, static_cast<std::size_t>(alignment),
                                        allocationtest::Family::alignedArrayNothrow); }
  catch (...) { return nullptr; }
}

void operator delete(void* memory) noexcept {
  allocationtest::observeDelete(memory, allocationtest::Family::scalar,
                                allocationtest::DeleteForm::unsized);
  std::free(memory);
}
void operator delete[](void* memory) noexcept {
  allocationtest::observeDelete(memory, allocationtest::Family::array,
                                allocationtest::DeleteForm::unsized);
  std::free(memory);
}
void operator delete(void* memory, std::size_t size) noexcept {
  allocationtest::observeDelete(memory, allocationtest::Family::scalar,
                                allocationtest::DeleteForm::sized, size);
  std::free(memory);
}
void operator delete[](void* memory, std::size_t size) noexcept {
  allocationtest::observeDelete(memory, allocationtest::Family::array,
                                allocationtest::DeleteForm::sized, size);
  std::free(memory);
}
void operator delete(void* memory, const std::nothrow_t&) noexcept {
  allocationtest::observeDelete(memory, allocationtest::Family::scalarNothrow,
                                allocationtest::DeleteForm::nothrowCleanup);
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
  std::free(memory);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}
void operator delete[](void* memory, const std::nothrow_t&) noexcept {
  allocationtest::observeDelete(memory, allocationtest::Family::arrayNothrow,
                                allocationtest::DeleteForm::nothrowCleanup);
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
  std::free(memory);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}
void operator delete(void* memory, std::align_val_t alignment) noexcept {
  allocationtest::observeDelete(memory, allocationtest::Family::alignedScalar,
                                allocationtest::DeleteForm::unsized, 0,
                                static_cast<std::size_t>(alignment));
  geometryFreeAligned(memory);
}
void operator delete[](void* memory, std::align_val_t alignment) noexcept {
  allocationtest::observeDelete(memory, allocationtest::Family::alignedArray,
                                allocationtest::DeleteForm::unsized, 0,
                                static_cast<std::size_t>(alignment));
  geometryFreeAligned(memory);
}
void operator delete(void* memory, std::size_t size, std::align_val_t alignment) noexcept {
  allocationtest::observeDelete(memory, allocationtest::Family::alignedScalar,
                                allocationtest::DeleteForm::sized, size,
                                static_cast<std::size_t>(alignment));
  geometryFreeAligned(memory);
}
void operator delete[](void* memory, std::size_t size, std::align_val_t alignment) noexcept {
  allocationtest::observeDelete(memory, allocationtest::Family::alignedArray,
                                allocationtest::DeleteForm::sized, size,
                                static_cast<std::size_t>(alignment));
  geometryFreeAligned(memory);
}
void operator delete(void* memory, std::align_val_t alignment,
                      const std::nothrow_t&) noexcept {
  allocationtest::observeDelete(memory, allocationtest::Family::alignedScalarNothrow,
                                allocationtest::DeleteForm::nothrowCleanup, 0,
                                static_cast<std::size_t>(alignment));
  geometryFreeAligned(memory);
}
void operator delete[](void* memory, std::align_val_t alignment,
                        const std::nothrow_t&) noexcept {
  allocationtest::observeDelete(memory, allocationtest::Family::alignedArrayNothrow,
                                allocationtest::DeleteForm::nothrowCleanup, 0,
                                static_cast<std::size_t>(alignment));
  geometryFreeAligned(memory);
}
#endif

namespace {
using namespace videowire::geometry;
int failures = 0;
void check(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

PortContract contractFor(CarrierKind carrier,
                         ValueType fieldType = ValueType::unspecified) {
  PortContract c;
  c.carrier = carrier;
  c.overflow = OverflowPolicy::reject;
  switch (carrier) {
  case CarrierKind::geometry3D:
    c.maxVertices = 8;
    c.maxIndices = 12;
    break;
  case CarrierKind::points3D:
    c.maxPoints = 8;
    break;
  case CarrierKind::curves3D:
    c.maxCurvePoints = 8;
    c.maxSplines = 4;
    break;
  case CarrierKind::instances3D:
    c.maxInstances = 8;
    break;
  case CarrierKind::field:
    c.fieldDomain = Domain::point;
    c.fieldValueType = fieldType;
    c.fieldInterpolation = Interpolation::linear;
    c.maxFieldElements = 3;
    break;
  default:
    break;
  }
  return c;
}
ValueDescriptor base(CarrierKind carrier) {
  ValueDescriptor v;
  v.carrier = carrier;
  v.stableId = 101 + static_cast<unsigned>(carrier);
  v.sourceStableId = 77;
  v.sourceRevision = 9;
  return v;
}
ValueDescriptor geometryValue() {
  auto v = base(CarrierKind::geometry3D);
  GeometryData d;
  d.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  d.vertexIds = {11, 12, 13};
  d.indices = {0, 1, 2};
  v.data = std::move(d);
  v.operations = {{v.stableId, OperationCode::copy, 0, v.stableId, {0, 0, 0, 0}}};
  v.dispatchCount = 1;
  return v;
}
ValueDescriptor pointsValue() {
  auto v = base(CarrierKind::points3D);
  PointsData d;
  d.points = {{{0, 0, 0}, 21, 1, {0, 0, 0, 1}},
              {{1, 2, 3}, 22, 0.5f, {0, 0, 0, 1}}};
  v.data = std::move(d);
  v.operations = {{v.stableId, OperationCode::copy, 0, v.stableId,
                   {0, 0, 0, 0}}};
  v.dispatchCount = 1;
  return v;
}
ValueDescriptor curvesValue() {
  auto v = base(CarrierKind::curves3D);
  CurvesData d;
  d.points = {{{0, 0, 0}, 31, 1, 0},
              {{1, 0, 0}, 32, 0.8f, 0.2f},
              {{2, 0, 0}, 33, 0.5f, 0}};
  d.splines = {{41, 0, 3, false}};
  v.data = std::move(d);
  v.operations = {{v.stableId, OperationCode::copy, 0, v.stableId,
                   {0, 0, 0, 0}}};
  v.dispatchCount = 1;
  return v;
}
ValueDescriptor instancesValue() {
  std::string error;
  auto pointGrid = lowerGrid(600, 77, 9, 2, 2, 1.0f,
                             contractFor(CarrierKind::geometry3D), error);
  auto points = lowerPointsFromVertices(*pointGrid, 601,
                                        contractFor(CarrierKind::points3D), error);
  auto sourceGrid = lowerGrid(700, 77, 9, 2, 2, 0.25f,
                              contractFor(CarrierKind::geometry3D), error);
  auto instances = lowerInstanceOnPoints(*points, sourceGrid->stableId, 105,
                                         contractFor(CarrierKind::instances3D), error);
  instances->operations.insert(instances->operations.begin(),
                               sourceGrid->operations.begin(),
                               sourceGrid->operations.end());
  instances->dispatchCount += sourceGrid->dispatchCount;
  return *instances;
}
ValueDescriptor fieldValue(ValueType type = ValueType::vector) {
  auto v = base(CarrierKind::field);
  v.fieldDomain = Domain::point;
  v.fieldValueType = type;
  v.fieldInterpolation = Interpolation::linear;
  v.fieldDomainCardinality = 3;
  FieldData d;
  d.elements.resize(3);
  for (std::size_t i = 0; i < d.elements.size(); ++i) {
    d.elements[i].components[0] = static_cast<double>(i);
    if (type == ValueType::vector) {
      d.elements[i].components[1] = i + 1.0;
      d.elements[i].components[2] = i + 2.0;
    } else if (type == ValueType::color || type == ValueType::rotation) {
      d.elements[i].components[1] = 0.25;
      d.elements[i].components[2] = 0.5;
      d.elements[i].components[3] = type == ValueType::rotation ? 1.0 : 0.75;
    } else if (type == ValueType::boolean)
      d.elements[i].components[0] = i % 2;
  }
  v.data = std::move(d);
  v.operations = {{v.stableId, OperationCode::evaluateField, v.stableId,
                   v.stableId, {0, 0, 0, 0}}};
  v.dispatchCount = 1;
  return v;
}
AdmissionContext context() {
  AdmissionContext c;
  c.admittedGeometryOrSceneSources.insert(700);
  return c;
}

class TestBackend final
    : public videohelper::geometry::BackendCapabilitySource {
public:
  videohelper::geometry::BackendCapabilities
  geometryCoreCapabilities() const override {
    return capabilities;
  }
  videohelper::geometry::BackendCapabilities capabilities = []() {
    videohelper::geometry::BackendCapabilities c;
    c.backendIdentity = "OpenGL";
    c.nativeGpuAvailable = true;
    c.geometryCoreExecution = true;
    c.immutableSourceBuffers = true;
    c.stableElementIds = true;
    c.typedFieldEvaluation = true;
    c.gpuInstancingWithoutMeshExpansion = true;
    c.supportedCarriers =
        videohelper::geometry::carrierBit(CarrierKind::geometry3D) |
        videohelper::geometry::carrierBit(CarrierKind::points3D) |
        videohelper::geometry::carrierBit(CarrierKind::curves3D) |
        videohelper::geometry::carrierBit(CarrierKind::instances3D) |
        videohelper::geometry::carrierBit(CarrierKind::field);
    return c;
  }();
};

void roundtrip(ValueDescriptor value, PortContract contract,
               const AdmissionContext &ctx, const char *name) {
  std::string error;
  auto admitted =
      admitValue(std::move(value), contract, ResourceLimits{}, ctx, error);
  check(admitted.has_value(), name);
  if (!admitted)
    return;
  const auto bytes = encodeRuntimeValue(*admitted);
  auto decoded =
      decodeRuntimeValue(bytes, contract, ResourceLimits{}, ctx, error);
  check(decoded.has_value() && error.empty(),
        "valid runtime transport decodes");
  if (decoded) {
    check(decoded->receipt().deterministicDigest ==
              admitted->receipt().deterministicDigest,
          "roundtrip keeps the deterministic budget receipt");
    check(encodeRuntimeValue(*decoded) == bytes,
          "runtime transport is frozen and byte-exact");
  }
}

void validRoundtrips() {
  roundtrip(geometryValue(), contractFor(CarrierKind::geometry3D), {},
            "Geometry3D with positions and indices admits");
  roundtrip(pointsValue(), contractFor(CarrierKind::points3D), {},
            "Points3D records admit");
  roundtrip(curvesValue(), contractFor(CarrierKind::curves3D), {},
            "Curves3D spline records admit");
  roundtrip(instancesValue(), contractFor(CarrierKind::instances3D), context(),
            "Instances3D references and transforms admit");
  roundtrip(fieldValue(), contractFor(CarrierKind::field, ValueType::vector),
            {}, "typed Field data admits");
  static_assert(!std::is_assignable_v<AdmittedValue &, AdmittedValue>);
  static_assert(!RenderContract::kAllowsCpuMeshExpansion);
  static_assert(!RenderContract::kAllowsCpuImageFallback);
}

void hostileTransportFailsClosed() {
  std::string error;
  auto admitted =
      admitValue(pointsValue(), contractFor(CarrierKind::points3D), error);
  check(admitted.has_value(), "transport fixture admits");
  if (!admitted)
    return;
  auto bytes = encodeRuntimeValue(*admitted);
  for (std::size_t cut = 0; cut < bytes.size(); ++cut)
    check(!decodeRuntimeValue(bytes.data(), cut,
                              contractFor(CarrierKind::points3D),
                              ResourceLimits{}, {}, error),
          "every truncated transport is rejected");
  auto malformed = bytes;
  malformed[0] ^= 0xff;
  check(!decodeRuntimeValue(malformed, contractFor(CarrierKind::points3D),
                            ResourceLimits{}, {}, error),
        "malformed magic is rejected");
  malformed = bytes;
  malformed[4] = 255;
  check(!decodeRuntimeValue(malformed, contractFor(CarrierKind::points3D),
                            ResourceLimits{}, {}, error),
        "unknown transport version is rejected");
  malformed = bytes;
  malformed[12] = 255;
  check(!decodeRuntimeValue(malformed, contractFor(CarrierKind::points3D),
                            ResourceLimits{}, {}, error),
        "unknown carrier enum is rejected");
  malformed = bytes;
  malformed.push_back(0);
  check(!decodeRuntimeValue(malformed, contractFor(CarrierKind::points3D),
                            ResourceLimits{}, {}, error),
        "trailing transport data is rejected");
  auto lowered =
      lowerRuntimePlan(contractFor(CarrierKind::points3D), *admitted);
  lowered[4] = 2;
  check(!decodeLoweredRuntimePlan(lowered, error),
        "unknown lowered-plan version is rejected");
  check(!decodeLoweredPlanText("ABC0", error),
        "non-canonical lowered-plan text is rejected");
}

void invalidNumbersAndShapesFail() {
  std::string error;
  auto v = pointsValue();
  std::get<PointsData>(v.data).points[0].position.x =
      std::numeric_limits<float>::quiet_NaN();
  check(!admitValue(v, contractFor(CarrierKind::points3D), error),
        "NaN position is rejected");
  v = pointsValue();
  std::get<PointsData>(v.data).points[0].radius =
      std::numeric_limits<float>::infinity();
  check(!admitValue(v, contractFor(CarrierKind::points3D), error),
        "infinite radius is rejected");
  auto g = geometryValue();
  std::get<GeometryData>(g.data).indices[2] = 3;
  check(!admitValue(g, contractFor(CarrierKind::geometry3D), error),
        "out-of-range geometry index is rejected");
  auto c = curvesValue();
  std::get<CurvesData>(c.data).splines[0].pointCount = 4;
  check(!admitValue(c, contractFor(CarrierKind::curves3D), error),
        "bad spline range is rejected");
  auto i = instancesValue();
  auto sourceOperation = std::find_if(i.operations.begin(), i.operations.end(),
      [](const auto &operation) { return operation.outputStableId == 700; });
  check(sourceOperation != i.operations.end(), "source operation fixture exists");
  sourceOperation->code = OperationCode::pointsFromVertices;
  sourceOperation->inputStableId = 600;
  sourceOperation->parameters = {};
  auto movedSourceOperation = *sourceOperation;
  i.operations.erase(sourceOperation);
  auto instanceOperation = std::find_if(i.operations.begin(), i.operations.end(),
      [](const auto &operation) { return operation.code == OperationCode::instanceOnPoints; });
  i.operations.insert(instanceOperation, movedSourceOperation);
  instanceOperation = std::find_if(i.operations.begin(), i.operations.end(),
      [](const auto &operation) { return operation.code == OperationCode::instanceOnPoints; });
  instanceOperation->secondaryInputStableId = 700;
  check(!admitValue(i, contractFor(CarrierKind::instances3D), error),
        "a connected Points3D producer cannot authorize an instance geometry source");
  check(error == "Instance on Points has no explicit Geometry3D provenance",
        ("the connected wrong-opcode source reaches the exact provenance rejection: " + error).c_str());
  i = instancesValue();
  auto &reorderedInstances = std::get<InstancesData>(i.data).instances;
  std::swap(reorderedInstances[0], reorderedInstances[1]);
  check(!admitValue(i, contractFor(CarrierKind::instances3D), ResourceLimits{},
                    context(), error),
        "retained Instances3D order must match its deterministic operation result");
  check(error == "retained Instances3D disagrees with its operation result",
        "retained Instances3D mismatch keeps its exact fail-closed diagnostic");
  i = instancesValue();
  for (auto &record : std::get<InstancesData>(i.data).instances)
    record.sourceStableId = 899999;
  i.operations.push_back({899999, OperationCode::grid, i.sourceStableId, 899999,
                           {2.0f, 2.0f, 1.0f, 0.0f}});
  instanceOperation = std::find_if(i.operations.begin(), i.operations.end(),
      [](const auto &operation) { return operation.code == OperationCode::instanceOnPoints; });
  instanceOperation->secondaryInputStableId = 899999;
  ++i.dispatchCount;
  check(!admitValue(i, contractFor(CarrierKind::instances3D), error),
        "a geometry producer ordered after Instance on Points cannot forge provenance");
  i = instancesValue();
  std::get<InstancesData>(i.data).instances[0].sourceStableId = 701;
  check(!admitValue(i, contractFor(CarrierKind::instances3D), ResourceLimits{},
                    context(), error),
        "mixed instance source provenance is rejected");
  auto f = fieldValue();
  std::get<FieldData>(f.data).elements.pop_back();
  check(
      !admitValue(f, contractFor(CarrierKind::field, ValueType::vector), error),
      "field cardinality must match its declared domain capacity contract "
      "input");
  auto foreign = pointsValue();
  foreign.data = GeometryData{};
  check(!admitValue(foreign, contractFor(CarrierKind::points3D), error),
        "foreign carrier payload is rejected");
  auto contract = contractFor(CarrierKind::points3D);
  contract.attributes = {{88, "faces", ValueType::floatValue, Domain::face,
                          Interpolation::linear}};
  contract.maxAttributes = 1;
  check(!validatePortContract(contract, error),
        "foreign attribute domain is rejected");
  std::size_t result = 0;
  check(!checkedMul(std::numeric_limits<std::size_t>::max(), 2, result),
        "checked byte multiplication rejects arithmetic overflow");
  check(!checkedAdd(std::numeric_limits<std::size_t>::max(), 1, result),
        "checked byte addition rejects arithmetic overflow");
}

void attributeCardinalityAndBudgets() {
  auto v = pointsValue();
  auto c = contractFor(CarrierKind::points3D);
  c.maxAttributes = 1;
  c.attributes = {{80, "weight", ValueType::floatValue, Domain::point,
                   Interpolation::linear}};
  AttributeData a;
  a.descriptor = c.attributes[0];
  a.elements.resize(1);
  v.attributes.push_back(a);
  std::string error;
  check(!admitValue(v, c, error), "attribute cardinality mismatch is rejected");
  v = pointsValue();
  ResourceLimits limits;
  limits.maxBufferBytes = 1;
  check(!admitValue(v, contractFor(CarrierKind::points3D), limits, error),
        "derived bytes, not caller bytes, enforce the budget");
  auto a1 =
      admitValue(pointsValue(), contractFor(CarrierKind::points3D), error);
  auto a2 =
      admitValue(pointsValue(), contractFor(CarrierKind::points3D), error);
  check(a1 && a2 &&
            a1->receipt().deterministicDigest ==
                a2->receipt().deterministicDigest &&
            a1->receipt().bufferBytes > 0,
        "budget receipts are immutable and deterministic");
}

void typedFieldKindsAdmit() {
  for (auto type :
       {ValueType::floatValue, ValueType::vector, ValueType::color,
        ValueType::boolean, ValueType::integer, ValueType::rotation}) {
    std::string error;
    auto admitted = admitValue(fieldValue(type),
                               contractFor(CarrierKind::field, type), error);
    check(admitted.has_value(), "each typed Field value kind admits");
  }
}

void productionLoweringAndTypedEvaluation() {
  std::string error;
  auto grid = lowerGrid(900, 700, 1, 3, 2, 0.5f,
                        contractFor(CarrierKind::geometry3D), error);
  check(grid && std::get<GeometryData>(grid->data).positions.size() == 6 &&
            std::get<GeometryData>(grid->data).indices.size() == 12 &&
            std::get<GeometryData>(grid->data).indices[0] == 0 &&
            std::get<GeometryData>(grid->data).indices[1] == 1 &&
            std::get<GeometryData>(grid->data).indices[2] == 3,
        "Grid lowering produces exact front-facing indexed Geometry3D before admission");
  if (!grid)
    return;
  FieldData offsets;
  offsets.elements.resize(6);
  for (auto &element : offsets.elements)
    element.components = {0.0, 0.0, 0.25, 0.0};
  check(applyTypedVectorField(*grid, offsets, Domain::vertex, error) &&
            std::get<GeometryData>(grid->data).positions[0].z == 0.25f,
        "a typed vector field chain evaluates only over its matching domain");
  const auto positionBeforeRejectedField =
      std::get<GeometryData>(grid->data).positions.front();
  offsets.elements.back().components[2] =
      std::numeric_limits<double>::quiet_NaN();
  check(!applyTypedVectorField(*grid, offsets, Domain::vertex, error) &&
            std::get<GeometryData>(grid->data).positions.front().z ==
                positionBeforeRejectedField.z,
        "typed field validation rejects every element before mutation");
  offsets.elements.back().components[2] = 0.25;
  offsets.elements.pop_back();
  check(!applyTypedVectorField(*grid, offsets, Domain::vertex, error),
        "field evaluation fails closed on cardinality mismatch");
  check(!lowerGrid(900, 700, 1, 1, 2, 0.5f,
                  contractFor(CarrierKind::geometry3D), error),
        "Grid lowering rejects invalid dimensions before allocation");
  auto circle = lowerCurveCircle(901, 700, 1, 8, 1.0f,
                                 contractFor(CarrierKind::curves3D), error);
  check(circle && std::get<CurvesData>(circle->data).splines[0].cyclic,
        "Curve Circle lowering produces one closed typed spline");
  auto meshContract = contractFor(CarrierKind::geometry3D);
  meshContract.maxVertices = 16;
  meshContract.maxIndices = 48;
  auto mesh = circle ? lowerCurveToMesh(*circle, 902, meshContract, error)
                     : std::nullopt;
  check(mesh && std::get<GeometryData>(mesh->data).positions.size() == 16,
        "Curve to Mesh lowering retains paired curve-point identities");
  auto instances = lowerInstanceOnPoints(pointsValue(), 700, 903,
                                         contractFor(CarrierKind::instances3D), error);
  check(instances && std::get<InstancesData>(instances->data).instances.size() == 2,
        "Instance on Points lowers without expanding source mesh data");
  Transform transform; transform.translation = {1, 2, 3};
  auto transformed = lowerTransformGeometry(*grid, 904, transform,
                                            contractFor(CarrierKind::geometry3D), error);
  check(transformed && std::get<GeometryData>(transformed->data).positions[0].x == 0.5f,
        "Transform Geometry lowers a bounded immutable position transform");
  check(transformed && lowerSetMaterial(*transformed, 905, error),
        "Set Material lowers as an immutable typed operation");
}

void productionAdmissionSharesPreviewAndExportData() {
  std::string error;
  auto admitted =
      admitValue(instancesValue(), contractFor(CarrierKind::instances3D),
                 ResourceLimits{}, context(), error);
  check(admitted.has_value(), "shared-plan fixture admits");
  MaterializedInstancePlan materialized;
  check(admitted && validateOperationPlan(admitted->descriptor(), {}, &materialized, error)
            && materialized.geometries.count(700) == 1
            && materialized.points.count(601) == 1
            && materialized.instances.instances.size() == 4,
        "production admission materializes the Grid to Points to Instances chain");
  auto mismatched = instancesValue();
  std::get<InstancesData>(mismatched.data).instances[0].transform.translation.x += 0.25f;
  check(!admitValue(mismatched, contractFor(CarrierKind::instances3D), error),
        "retained Instances3D must equal the materialized operation result");
  if (!admitted)
    return;
  auto bytes =
      lowerRuntimePlan(contractFor(CarrierKind::instances3D), *admitted);
  auto payload = encodeLoweredPlanText(bytes);
  TestBackend backend;
  videohelper::geometry::GeometryCorePlanRuntime runtime(backend);
  auto preview =
      runtime.admitPreviewText(payload, ResourceLimits{}, context(), error);
  auto exportUse =
      runtime.admitExportText(payload, ResourceLimits{}, context(), error);
  check(preview && exportUse && preview->plan.get() != exportUse->plan.get(),
        "isolated preview and export owners receive distinct immutable authority objects");
  check(preview && preview->use == videohelper::geometry::PlanUse::preview &&
            exportUse &&
            exportUse->use == videohelper::geometry::PlanUse::exportRender,
        "preview and export uses remain explicit");
  check(preview && exportUse &&
            preview->plan->receipt().backendBinding ==
                exportUse->plan->receipt().backendBinding &&
            preview->plan->receipt().backendBinding != 0,
        "execution receipt deterministically binds value, backend identity, "
        "and effective limits");
  backend.capabilities.geometryCoreExecution = false;
  videohelper::geometry::GeometryCorePlanRuntime unsupported(backend);
  check(!unsupported.admitPreview(bytes, ResourceLimits{}, context(), error),
        "backend capability comes from the production capability source and "
        "fails closed");

  TestBackend noFields;
  noFields.capabilities.typedFieldEvaluation=false;
  auto fieldConsumer=geometryValue();
  fieldConsumer.operations.push_back({910,OperationCode::evaluateField,910,910,{}});
  ++fieldConsumer.dispatchCount;
  check(admitValue(fieldConsumer,contractFor(CarrierKind::geometry3D),error).has_value(),
        "legacy evaluated field fixture retains a valid mesh plan");
  check(!videohelper::geometry::admitPlanValue(fieldConsumer,
            contractFor(CarrierKind::geometry3D),{}, {},noFields,error)
            && error.find("required typed geometry operation")!=std::string::npos,
        "evaluated fields require typed field capability even with a mesh output");
}

void deferredParticleSpectrumFieldsKeepFrameOwnership() {
  using namespace videohelper::geometry;
  std::string error;
  const auto context=particlePointsFixtureContext();
  const auto contract=particlePointsInstanceContract();
  for (const unsigned mode : {0u,1u,2u,3u,4u,7u}) {
    const auto value=deferredParticleSpectrumFixture(error,mode);
    check(value.has_value(),"direct deferred particle spectrum lowers for each non-UV coordinate mode");
    if (!value) { std::cerr<<error<<'\n'; continue; }
    const auto admitted=admitValue(*value,contract,{},context,error);
    check(admitted.has_value(),"deferred spectrum admits only with compiled owner declaration");
    if (!admitted) continue;
    check(value->spectrumFields.size()==1 && value->spectrumFields[0].coordinates.empty()
        && std::get<InstancesData>(value->data).instances.empty(),
        "immutable spectrum plan contains neither invented coordinates nor frame cardinality");
    const auto wire=encodeRuntimeValue(*admitted);
    const auto decoded=decodeRuntimeValue(wire,contract,{},context,error);
    check(wire[4]==13 && decoded && encodeRuntimeValue(*decoded)==wire,
        "deferred spectrum uses unchanged non-text v13 transport and roundtrips exact authored bytes");
    check(!decodeRuntimeValue(wire,contract,{}, {},error),"bare deferred spectrum transport grants no external owner");
    check(!spectrum::validate(value->spectrumFields[0],0,error),"ordinary spectrum binding still rejects empty coordinates");
    RuntimeFieldEvaluation runtime;
    runtime.spectrumAt=[](const auto& binding,auto& bands,std::string& diagnostic) {
      if (binding.source!=spectrum::Source::group || binding.sourceTrackId!=17 || !binding.coordinates.empty()) {
        diagnostic="Selected group binding changed during frame evaluation"; return false;
      }
      bands=spectrumDomainBands(); return true;
    };
    ValueDescriptor sample,first,changed,again;
    auto frame=context;
    check(sampleParticlePointsFixture(particlePointsFixture(),.5,sample,error),"owned particle trajectory provides spectral sample frame");
    frame.evaluatedPoints.emplace(801,sample);
    const auto replayed=validateOperationPlan(*value,frame,nullptr,error,&first,&runtime);
    check(replayed,"typed spectral displacement replays after exact particle frame admission");
    if (!replayed) { std::cerr<<error<<'\n'; continue; }
    const auto& points=std::get<PointsData>(sample.data).points;
    const auto& instances=std::get<InstancesData>(first.data).instances;
    check(instances.size()==2 && first.attributes.size()==3,
        "particle spectrum retains evaluated cardinality and promotes existing point attributes");
    for (std::size_t i=0;i<instances.size();++i) {
      const auto& p=points[i]; const auto& instance=instances[i];
      check(instance.stableId==p.stableId && instance.sourceStableId==802
          && instance.transform.translation.y==p.position.y && instance.transform.translation.z==p.position.z,
          "displaced instances preserve exact stable/source IDs and undisplaced axes");
      if (mode==0 || mode==7)
        check(std::abs(instance.transform.translation.x-(p.position.x+(i==0 ? .035:.315)))<2e-6,
            "independent band-endpoint oracle reaches actual particle-instance translations");
    }
    if (mode==7) {
      auto reordered=frame;
      auto& records=std::get<PointsData>(reordered.evaluatedPoints[801].data).points;
      std::swap(records[0],records[1]);
      for (auto& attribute : reordered.evaluatedPoints[801].attributes) std::swap(attribute.elements[0],attribute.elements[1]);
      check(validateOperationPlan(*value,reordered,nullptr,error,&again,&runtime),"particle stable-index sampling admits an exact record reorder");
      const auto& ranked=std::get<InstancesData>(again.data).instances;
      check(ranked.size()==2 && ranked[0].stableId==22 && ranked[1].stableId==13
          && std::abs(ranked[0].transform.translation.x-(records[0].position.x+.315))<2e-6
          && std::abs(ranked[1].transform.translation.x-(records[1].position.x+.035))<2e-6,
          "spectral coordinates follow stable identity ranks rather than reordered particle slots");
    }
    runtime.spectrumAt=[](const auto&,auto& bands,std::string&) { bands=spectrumDomainBands(true); return true; };
    check(validateOperationPlan(*value,frame,nullptr,error,&changed,&runtime)
        && !equal(std::get<InstancesData>(first.data),std::get<InstancesData>(changed.data)),
        "two selected-source spectra cause distinct transforms on the same point frame");
    runtime.spectrumAt=[](const auto&,auto& bands,std::string&) { bands=spectrumDomainBands(); return true; };
    for (double seconds : {1.1,1.3,.5}) {
      check(sampleParticlePointsFixture(particlePointsFixture(),seconds,frame.evaluatedPoints[801],error)
          && validateOperationPlan(*value,frame,nullptr,error,&again,&runtime),
          "particle spectrum replays changing cardinality, expiry and backward seek");
      const auto& data=std::get<InstancesData>(again.data).instances;
      check(data.size()==(seconds==.5 ? 2u : seconds==1.1 ? 1u : 0u),
          "spectral field cardinality follows the current particle frame exactly");
      if (seconds==.5) check(equal(std::get<InstancesData>(first.data),std::get<InstancesData>(again.data))
          && equalAttributes(first.attributes,again.attributes),"backward seek reproduces particle spectrum state");
      if (seconds==1.1) check(data[0].stableId==22,"surviving particle keeps stable identity after cardinality change");
    }
    auto reset=particlePointsFixture(); reset.resetTime=.375f;
    check(sampleParticlePointsFixture(reset,.5,frame.evaluatedPoints[801],error)
        && validateOperationPlan(*value,frame,nullptr,error,&again,&runtime)
        && !equal(std::get<InstancesData>(first.data),std::get<InstancesData>(again.data)),
        "authored particle reset causally changes spectral instance trajectory");
    check(sampleParticlePointsFixture(particlePointsFixture(),.5,frame.evaluatedPoints[801],error)
        && validateOperationPlan(*value,frame,nullptr,error,&again,&runtime)
        && equal(std::get<InstancesData>(first.data),std::get<InstancesData>(again.data)),
        "restored particle replay repeats the exact spectral transforms");
    auto denied=runtime;
    denied.spectrumAt=[](const auto&,auto&,std::string& diagnostic) { diagnostic="Selected group is unavailable"; return false; };
    auto untouched=first;
    check(!validateOperationPlan(*value,frame,nullptr,error,&untouched,&denied)
        && equal(std::get<InstancesData>(first.data),std::get<InstancesData>(untouched.data)),
        "missing selected analysis refuses transactionally without publishing partial points");
    check(sampleParticlePointsFixture(particlePointsFixture(),1.3,frame.evaluatedPoints[801],error)
        && !validateOperationPlan(*value,frame,nullptr,error,&untouched,&denied),
        "an expired empty frame still refuses unavailable selected-source analysis");
    frame.evaluatedPoints[801]=sample;
    check(!validateOperationPlan(*value,context,nullptr,error,&untouched,&runtime),
        "runtime spectrum cannot substitute an authoring placeholder for its point provider");
    auto hostile=frame; hostile.evaluatedPoints[801].sourceRevision++;
    check(!validateOperationPlan(*value,hostile,nullptr,error,&untouched,&runtime),"stale evaluated point revision refuses");
    hostile=frame; hostile.evaluatedPoints[801].sourceStableId=804;
    check(!validateOperationPlan(*value,hostile,nullptr,error,&untouched,&runtime),"foreign evaluated point owner refuses");
    hostile=frame; std::get<PointsData>(hostile.evaluatedPoints[801].data).points[1].stableId=13;
    check(!validateOperationPlan(*value,hostile,nullptr,error,&untouched,&runtime),"duplicate particle IDs refuse before sampling");
    hostile=frame; std::get<PointsData>(hostile.evaluatedPoints[801].data).points[0].position.x=std::numeric_limits<float>::infinity();
    check(!validateOperationPlan(*value,hostile,nullptr,error,&untouched,&runtime),"nonfinite particle positions refuse before sampling");
    hostile=frame; hostile.deferredPoints[801].capacity=1;
    check(!validateOperationPlan(*value,hostile,nullptr,error,&untouched,&runtime),"over-capacity particle frame refuses");
    hostile=context; hostile.deferredPoints[801].sourceRevision++;
    check(!admitValue(*value,contract,{},hostile,error),"stale compiled particle declaration refuses");
    hostile=context; hostile.deferredPoints[801].capacity=0;
    check(!admitValue(*value,contract,{},hostile,error),"missing compiled particle capacity refuses");
    auto substituted=*value;
    substituted.operations[2].field.mode=5;
    check(!admitValue(substituted,contract,{},context,error),"deferred particles cannot claim authored UV coordinates");
    substituted=*value; substituted.operations[2].field.domain=Domain::instance;
    check(!admitValue(substituted,contract,{},context,error),"wrong spectral element domain refuses");
    substituted=*value; substituted.operations[2].field.valueType=ValueType::rotation;
    check(!admitValue(substituted,contract,{},context,error),"wrong spectral value type refuses");
    substituted=*value; substituted.spectrumFields[0].coordinates={0};
    check(!admitValue(substituted,contract,{},context,error),"invented static coordinates cannot describe deferred particles");
    auto limits=ResourceLimits{}; limits.maxFieldElements=1;
    check(!admitValue(*value,contract,limits,context,error),"declared spectral capacity obeys field limits before frame allocation");
    auto smallInstances=contract; smallInstances.maxInstances=1;
    check(!admitValue(*value,smallInstances,{},context,error),"unresolved particle spectrum obeys the exact terminal instance contract");
    limits={}; limits.maxBufferBytes=admitted->receipt().bufferBytes-1;
    check(!admitValue(*value,contract,limits,context,error),"deferred spectral coordinates and working fields obey retained byte budget");
    TestBackend backend; backend.capabilities.limits.maxFieldElements=1;
    check(!admitPlanValue(*value,contract,{},context,backend,error),"native backend field limit bounds unresolved particle spectrum");
  }
  ValueDescriptor empty;
  empty.carrier=CarrierKind::points3D; empty.stableId=empty.sourceStableId=801;
  empty.sourceRevision=3; empty.data=PointsData{};
  empty.operations={{801,OperationCode::copy,0,801,{}}}; empty.dispatchCount=1;
  const auto fieldContract=constructedFieldContract(Domain::point,ValueType::vector);
  const auto settings=spectrumDomainSettings(Domain::point,ValueType::vector,0);
  check(!lowerAudioSpectrumField(empty,9630,spectrumDomainBinding(),settings,fieldContract,error),
      "uncontextualized empty Points3D never becomes a generally valid spectrum field");
  auto stale=empty; stale.sourceRevision=4;
  check(!lowerAudioSpectrumField(stale,9630,spectrumDomainBinding(),settings,fieldContract,error,context),
      "deferred lowering refuses a stale authored revision before producing a field");
  auto smallField=fieldContract; smallField.maxFieldElements=1;
  check(!lowerAudioSpectrumField(empty,9630,spectrumDomainBinding(),settings,smallField,error,context),
      "deferred lowering respects the original typed field contract capacity");
  auto field=lowerAudioSpectrumField(empty,9630,spectrumDomainBinding(),settings,fieldContract,error,context);
  check(field && !admitValue(*field,fieldContract,error),"unresolved typed fields remain unavailable without their exact declaration");
  if (field) {
    check(!admitValue(*field,smallField,{},context,error),"serialized deferred field cannot borrow capacity beyond its original port");
    auto frame=context; ValueDescriptor sample,replayed;
    check(sampleParticlePointsFixture(particlePointsFixture(),.5,sample,error),"direct typed field gets its owned particle frame");
    frame.evaluatedPoints.emplace(801,sample);
    RuntimeFieldEvaluation runtime;
    runtime.spectrumAt=[](const auto&,auto& bands,std::string&) { bands=spectrumDomainBands(); return true; };
    check(validateOperationPlan(*field,frame,nullptr,error,&replayed,&runtime)
        && replayed.fieldDomainCardinality==2 && std::get<FieldData>(replayed.data).elements.size()==2,
        "direct typed field replay publishes exactly the evaluated domain cardinality");
  }
}

void deferredParticleSpectrumColorsPromoteExactFrame() {
  std::string error;
  const auto context=particlePointsFixtureContext();
  const auto value=deferredParticleSpectrumFixture(error,7,true);
  check(value.has_value(),"particle Color spectrum stores through the exact deferred point context");
  if (!value) { std::cerr<<error<<'\n'; return; }
  const auto contract=withAttributeContract(particlePointsInstanceContract(),*value);
  const auto admitted=admitValue(*value,contract,{},context,error);
  check(admitted.has_value(),"stored particle Color descriptor admits as the existing instance appearance attribute");
  if (!admitted) { std::cerr<<error<<'\n'; return; }
  const auto* authoredColor=findNamedAttribute(value->attributes,"color");
  check(authoredColor && authoredColor->descriptor.domain==Domain::instance
      && authoredColor->descriptor.valueType==ValueType::color && authoredColor->elements.empty(),
      "authored instance color has a typed descriptor without persisted frame values");
  if (!authoredColor) return;
  ValueDescriptor authored;
  check(validateOperationPlan(*value,context,nullptr,error,&authored)
      && equalAttributes(authored.attributes,value->attributes),
      "immutable terminal attributes equal ordered replay before any frame is evaluated");
  const auto bytes=encodeRuntimeValue(*admitted);
  const auto decoded=decodeRuntimeValue(bytes,contract,{},context,error);
  check(bytes[4]==12 && decoded && encodeRuntimeValue(*decoded)==bytes,
      "particle Color retains released transport and exact descriptor roundtrip");
  check(!decodeRuntimeValue(bytes,contract,{}, {},error),"stored Color transport cannot grant its own deferred owner");
  auto frame=context;
  ValueDescriptor sample,first,changed,again;
  check(sampleParticlePointsFixture(particlePointsFixture(),.5,sample,error),"Color sampling gets the replay-owned point frame");
  frame.evaluatedPoints.emplace(801,sample);
  RuntimeFieldEvaluation runtime;
  runtime.spectrumAt=[](const auto& binding,auto& bands,std::string&) {
    if (binding.source!=spectrum::Source::group || binding.sourceTrackId!=17 || !binding.coordinates.empty()) return false;
    bands=spectrumDomainBands(); return true;
  };
  const auto replayed=validateOperationPlan(*value,frame,nullptr,error,&first,&runtime);
  check(replayed,"point Color stores and promotes through the existing instance replay");
  if (!replayed) { std::cerr<<error<<'\n'; return; }
  const auto& instances=std::get<InstancesData>(first.data).instances;
  const auto& points=std::get<PointsData>(sample.data).points;
  const auto* color=findNamedAttribute(first.attributes,"color");
  check(color && color->descriptor==authoredColor->descriptor && color->elements.size()==2
      && first.attributes.size()==4 && instances.size()==2,"current point attributes promote exactly once to the instance domain");
  if (!color || color->elements.size()!=2 || instances.size()!=2) return;
  for (std::size_t i=0;i<instances.size();++i) {
    const auto appearance=instanceAppearance(first.attributes,i);
    const auto amount=i==0 ? .1:.9;
    check(instances[i].stableId==points[i].stableId && instances[i].sourceStableId==802
        && equal(instances[i].transform.translation,points[i].position)
        && std::abs(appearance.color[0]-amount*.2)<1e-6
        && std::abs(appearance.color[1]-amount)<1e-6
        && std::abs(appearance.color[2]-amount*.1)<1e-6 && appearance.color[3]==1,
        "independent selected-band oracle reaches existing appearance without changing particle transforms or IDs");
  }
  runtime.spectrumAt=[](const auto&,auto& bands,std::string&) { bands=spectrumDomainBands(true); return true; };
  check(validateOperationPlan(*value,frame,nullptr,error,&changed,&runtime)
      && equal(std::get<InstancesData>(first.data),std::get<InstancesData>(changed.data))
      && !equalAttributes(first.attributes,changed.attributes),"two selected spectra causally change colors on the same particle frame");
  runtime.spectrumAt=[](const auto&,auto& bands,std::string&) { bands=spectrumDomainBands(); return true; };
  auto reordered=frame;
  std::swap(std::get<PointsData>(reordered.evaluatedPoints[801].data).points[0],
      std::get<PointsData>(reordered.evaluatedPoints[801].data).points[1]);
  for (auto& attribute : reordered.evaluatedPoints[801].attributes) std::swap(attribute.elements[0],attribute.elements[1]);
  check(validateOperationPlan(*value,reordered,nullptr,error,&again,&runtime)
      && std::get<InstancesData>(again.data).instances.size()==2
      && std::get<InstancesData>(again.data).instances[0].stableId==22
      && std::abs(instanceAppearance(again.attributes,0).color[1]-.9)<1e-6
      && std::abs(instanceAppearance(again.attributes,1).color[1]-.1)<1e-6,
      "record reorder keeps each spectral color attached to its stable particle identity");
  for (const auto seconds : {1.1,1.3,.5}) {
    check(sampleParticlePointsFixture(particlePointsFixture(),seconds,frame.evaluatedPoints[801],error)
        && validateOperationPlan(*value,frame,nullptr,error,&again,&runtime),"Color replay admits expiry and backward seek");
    const auto* current=findNamedAttribute(again.attributes,"color");
    check(current && current->elements.size()==(seconds==.5 ? 2u : seconds==1.1 ? 1u : 0u)
        && current->elements.size()==std::get<InstancesData>(again.data).instances.size(),
        "promoted colors use the current particle cardinality including an empty frame");
    if (seconds==.5) check(equalAttributes(first.attributes,again.attributes)
        && equal(std::get<InstancesData>(first.data),std::get<InstancesData>(again.data)),"backward seek restores exact particle colors");
  }
  const auto refuses=[&](const ValueDescriptor& candidate,const AdmissionContext& owner,const RuntimeFieldEvaluation& evaluation) {
    auto untouched=first;
    return !validateOperationPlan(candidate,owner,nullptr,error,&untouched,&evaluation)
        && equalAttributes(first.attributes,untouched.attributes)
        && equal(std::get<InstancesData>(first.data),std::get<InstancesData>(untouched.data));
  };
  auto denied=runtime; denied.spectrumAt=[](const auto&,auto&,std::string&) { return false; };
  check(refuses(*value,frame,denied),"missing selected Color analysis refuses without partial publication");
  check(sampleParticlePointsFixture(particlePointsFixture(),1.3,frame.evaluatedPoints[801],error)
      && refuses(*value,frame,denied),"expired Color frames still require selected analysis");
  frame.evaluatedPoints[801]=sample;
  check(refuses(*value,context,runtime),"Color replay refuses missing evaluated points atomically");
  for (unsigned fault=0;fault<5;++fault) {
    auto hostile=frame;
    if (fault==0) hostile.evaluatedPoints[801].sourceRevision++;
    if (fault==1) hostile.evaluatedPoints[801].sourceStableId=804;
    if (fault==2) std::get<PointsData>(hostile.evaluatedPoints[801].data).points[1].stableId=13;
    if (fault==3) std::get<PointsData>(hostile.evaluatedPoints[801].data).points[0].position.x=std::numeric_limits<float>::infinity();
    if (fault==4) hostile.deferredPoints[801].capacity=1;
    check(refuses(*value,hostile,runtime),"invalid Color owner, identity, finite position or capacity refuses atomically");
  }
  for (unsigned fault=0;fault<4;++fault) {
    auto invalid=*value;
    if (fault==0) invalid.operations[2].field.mode=5;
    if (fault==1) invalid.operations[2].field.domain=Domain::instance;
    if (fault==2) invalid.operations[2].field.valueType=ValueType::vector;
    if (fault==3) invalid.spectrumFields[0].direction[0]=std::numeric_limits<float>::infinity();
    check(refuses(invalid,frame,runtime),"UV, mistyped or nonfinite Color field refuses without publishing an attribute");
  }
  ValueDescriptor empty;
  empty.carrier=CarrierKind::points3D; empty.stableId=empty.sourceStableId=801; empty.sourceRevision=3;
  empty.data=PointsData{}; empty.operations={{801,OperationCode::copy,0,801,{}}}; empty.dispatchCount=1;
  const auto settings=spectrumDomainSettings(Domain::point,ValueType::color,7);
  const auto field=lowerAudioSpectrumField(empty,9630,spectrumDomainBinding(),settings,
      constructedFieldContract(Domain::point,ValueType::color),error,context);
  check(field.has_value(),"direct Color field keeps exact unresolved authority");
  if (field) {
    auto pointsContract=contractFor(CarrierKind::points3D);
    check(!lowerNamedAttribute(empty,&*field,9631,OperationCode::storeNamedAttribute,"color",ValueType::color,pointsContract,error),
        "Store Attribute cannot generally admit empty fields without context");
    const auto stored=lowerNamedAttribute(empty,&*field,9631,OperationCode::storeNamedAttribute,"color",ValueType::color,pointsContract,error,context);
    check(stored && stored->attributes.size()==1 && stored->attributes[0].descriptor.domain==Domain::point,
        "Store Attribute retains the typed point Color before promotion");
    auto stale=context; stale.deferredPoints[801].sourceRevision++;
    check(!lowerNamedAttribute(empty,&*field,9631,OperationCode::storeNamedAttribute,"color",ValueType::color,pointsContract,error,stale),
        "Store Attribute refuses stale compiled authority");
    auto foreign=context; foreign.deferredPoints[801].sourceStableId=804;
    check(!lowerNamedAttribute(empty,&*field,9631,OperationCode::storeNamedAttribute,"color",ValueType::color,pointsContract,error,foreign),
        "Store Attribute refuses a declaration for another source owner");
    pointsContract.maxPoints=1;
    check(!lowerNamedAttribute(empty,&*field,9631,OperationCode::storeNamedAttribute,"color",ValueType::color,pointsContract,error,context),
        "Store Attribute retains original point port capacity");
  }
  const auto grid=lowerGrid(9645,9645,1,2,2,1,constructedFieldMeshContract(),error);
  const auto staticPoints=grid ? lowerPointsFromVertices(*grid,9646,contractFor(CarrierKind::points3D),error) : std::nullopt;
  const auto staticField=staticPoints ? lowerAudioSpectrumField(*staticPoints,9647,spectrumDomainBinding(),settings,
      constructedFieldContract(Domain::point,ValueType::color),error) : std::nullopt;
  const auto staticStored=staticField ? lowerNamedAttribute(*staticPoints,&*staticField,9648,OperationCode::storeNamedAttribute,
      "color",ValueType::color,contractFor(CarrierKind::points3D),error) : std::nullopt;
  check(staticStored.has_value(),"ordinary static Color Store Attribute retains its existing admission");
  if (staticStored) {
    const auto staticWithContext=lowerNamedAttribute(*staticPoints,&*staticField,9648,OperationCode::storeNamedAttribute,
        "color",ValueType::color,contractFor(CarrierKind::points3D),error,{});
    const auto staticContract=withAttributeContract(contractFor(CarrierKind::points3D),*staticStored);
    const auto a=admitValue(*staticStored,staticContract,error);
    const auto b=staticWithContext ? admitValue(*staticWithContext,staticContract,error) : std::nullopt;
    check(a && b && encodeRuntimeValue(*a)==encodeRuntimeValue(*b),
        "empty optional context preserves exact static Store Attribute transport bytes");
    const auto defaultInstances=lowerInstanceOnPoints(*staticStored,802,9649,particlePointsInstanceContract(),error);
    const auto emptyContextInstances=lowerInstanceOnPoints(*staticStored,802,9649,particlePointsInstanceContract(),error,{});
    check(defaultInstances && emptyContextInstances && defaultInstances->attributes.empty()
        && emptyContextInstances->attributes.empty(),"new optional context preserves existing static instance descriptor behavior");
  }
}

void particlePointsKeepImmutableOwnerAndCapacity() {
  using namespace videohelper::geometry;
  std::string error;
  const auto value=particlePointsInstanceFixture(error);
  const auto context=particlePointsFixtureContext();
  const auto contract=particlePointsInstanceContract();
  const auto admitted=value ? admitValue(*value,contract,{},context,error) : std::nullopt;
  check(admitted.has_value(),"particle deferred source admits without serialized frame samples");
  if (!admitted) return;
  const auto bytes=lowerRuntimePlan(contract,*admitted);
  TestBackend backend;
  GeometryCorePlanRuntime runtime(backend);
  const PlanOwnerIdentity owner{1,1,1,4,3,PlanUse::preview};
  const auto first=runtime.admitPreview(owner,bytes,{},context,error);
  const auto same=runtime.admitPreview(owner,bytes,{},context,error);
  check(first && same && first->plan==same->plan && first->sourceContext.evaluatedPoints.empty(),
        "repeat admission retains immutable point authority without a cached frame");
  auto changed=context; changed.deferredPoints[801].capacity=3;
  check(!runtime.admitPreview(owner,bytes,{},changed,error),
        "an existing owner cannot replace its point capacity");
  auto staleOwner=owner; staleOwner.planGeneration=4;
  check(!runtime.admitPreview(staleOwner,bytes,{},context,error),
        "point source revision must match the exact native plan owner");
  ValueDescriptor sampled;
  check(sampleParticlePointsFixture(particlePointsFixture(),.5,sampled,error),
        "known point frame is available for admission refusal fixtures");
  auto frameContext=context; frameContext.evaluatedPoints.emplace(801,sampled);
  check(!runtime.admitPreview(owner,bytes,{},frameContext,error),
        "immutable admission refuses retained dynamic point samples");
  auto smallContract=contract; smallContract.maxInstances=1;
  check(!admitPlanValue(*value,smallContract,{},context,backend,error),
        "declared point capacity obeys the original terminal contract even at zero current cardinality");
  TestBackend limited=backend; limited.capabilities.limits.maxPoints=1;
  check(!admitPlanValue(*value,contract,{},context,limited,error),
        "deferred point capacity obeys the actual backend point limit");
  limited=backend; limited.capabilities.limits.maxInstances=1;
  check(!admitPlanValue(*value,contract,{},context,limited,error),
        "deferred instance capacity obeys the actual backend instance limit");
  ValueDescriptor frameValue;
  check(validateOperationPlan(*value,frameContext,nullptr,error,&frameValue),
        "dynamic points replay an exact terminal instance value before resource admission");
  limited=backend; limited.capabilities.limits.maxAttributes=2;
  check(!admitPlanValue(frameValue,withAttributeContract(contract,frameValue),{},frameContext,limited,error),
        "evaluated point attributes obey the effective backend attribute limit");
  BudgetReceipt terminalBytes;
  check(inspectValue(frameValue,frameContext,terminalBytes,error),
        "dynamic frame receipt includes retained point and attribute storage");
  limited=backend; limited.capabilities.limits.maxBufferBytes=terminalBytes.bufferBytes-1;
  check(!admitPlanValue(frameValue,withAttributeContract(contract,frameValue),{},frameContext,limited,error),
        "retained point samples obey the combined actual frame byte budget");
  check(first && first->plan->contract().maxInstances==contract.maxInstances,
        "dynamic replay retains the authored terminal contract");
}

void ownerIdentityAndBoundedLeases() {
  std::string error;
  auto admitted = admitValue(pointsValue(), contractFor(CarrierKind::points3D), error);
  check(admitted.has_value(), "owner fixture admits");
  if (!admitted)
    return;
  const auto bytes = lowerRuntimePlan(contractFor(CarrierKind::points3D), *admitted);
  TestBackend backend;
  videohelper::geometry::GeometryCorePlanRuntime runtime(backend, 1);
  const videohelper::geometry::PlanOwnerIdentity preview{10, 20, 30, 100, 40,
      videohelper::geometry::PlanUse::preview};
  const videohelper::geometry::PlanOwnerIdentity exportOwner{10, 20, 30, 101, 41,
      videohelper::geometry::PlanUse::exportRender};
  auto first = runtime.admitPreview(preview, bytes, ResourceLimits{}, {}, error);
  check(first && first->owner.projectGeneration == 10 && first->owner == preview,
        "admission retains the exact preview owner generations");
  auto repeat = runtime.admitPreview(preview, bytes, ResourceLimits{}, {}, error);
  check(repeat && first && repeat->plan.get() == first->plan.get(),
        "the same owner reuses its immutable plan");
  auto changed = bytes;
  changed.back() ^= 1;
  check(!runtime.admitPreview(preview, changed, ResourceLimits{}, {}, error),
        "an owner cannot replace its immutable admitted plan");
  check(!runtime.admitExport(exportOwner, bytes, ResourceLimits{}, {}, error),
        "a bounded cache rejects a new live owner before allocating its plan");
  first.reset();
  repeat.reset();
  auto evicted = runtime.admitExport(exportOwner, bytes, ResourceLimits{}, {}, error);
  check(evicted && evicted->owner == exportOwner,
        "the least-recent unused owner is deterministically evicted");

  videohelper::geometry::GeometryCorePlanRuntime clipOwners(backend, 2);
  const videohelper::geometry::PlanOwnerIdentity firstClip{10, 20, 30, 200, 40,
      videohelper::geometry::PlanUse::preview};
  const videohelper::geometry::PlanOwnerIdentity secondClip{10, 20, 30, 201, 40,
      videohelper::geometry::PlanUse::preview};
  auto firstClipLease = clipOwners.admitPreview(
      firstClip, bytes, ResourceLimits{}, {}, error);
  auto secondClipLease = clipOwners.admitPreview(
      secondClip, bytes, ResourceLimits{}, {}, error);
  check(firstClipLease && secondClipLease
            && firstClipLease->owner.clipIdentity == 200
            && secondClipLease->owner.clipIdentity == 201
            && firstClipLease->plan.get() != secondClipLease->plan.get()
            && firstClipLease->ownerLease != secondClipLease->ownerLease
            && firstClipLease->nativeResources != secondClipLease->nativeResources,
        "clips with identical payloads retain separate plan authority, owner leases, and resources");

  videohelper::geometry::GeometryCorePlanRuntime generations(backend, 2);
  const videohelper::geometry::PlanOwnerIdentity original{50, 60, 70, 90, 80,
      videohelper::geometry::PlanUse::preview};
  const videohelper::geometry::PlanOwnerIdentity helperRestart{50, 61, 70, 90, 80,
      videohelper::geometry::PlanUse::preview};
  const videohelper::geometry::PlanOwnerIdentity deviceRestart{50, 61, 71, 90, 80,
      videohelper::geometry::PlanUse::preview};
  auto originalLease = generations.admitPreview(
      original, bytes, ResourceLimits{}, {}, error);
  auto restartedLease = generations.admitPreview(
      helperRestart, bytes, ResourceLimits{}, {}, error);
  check(originalLease && restartedLease
            && !(originalLease->owner == restartedLease->owner),
        "a helper restart creates an exact new owner without reusing the live lease");
  check(!generations.admitPreview(
            deviceRestart, bytes, ResourceLimits{}, {}, error),
        "a device restart cannot evict either live generation lease");
  restartedLease.reset();
  auto deviceLease = generations.admitPreview(
      deviceRestart, bytes, ResourceLimits{}, {}, error);
  check(deviceLease && deviceLease->owner == deviceRestart
            && originalLease && originalLease->owner == original,
        "device invalidation evicts only the least-recent released generation");
  const videohelper::geometry::PlanOwnerIdentity invalid{};
  check(!runtime.admitPreview(invalid, bytes, ResourceLimits{}, {}, error),
        "incomplete owner identity fails closed");
}
void meshGeneratorsRoundtripAndRejectInvalidPlans() {
  auto contract = contractFor(CarrierKind::geometry3D);
  contract.maxVertices = 4096;
  contract.maxIndices = 12288;
  struct Fixture {
    OperationCode code;
    std::array<float, 4> parameters;
    std::size_t vertices, indices;
  };
  const std::array<Fixture, 8> fixtures {{
      {OperationCode::plane, {2, 3, 0, 0}, 4, 6},
      {OperationCode::cube, {2, 3, 4, 0}, 24, 36},
      {OperationCode::sphere, {1, 8, 4, 0}, 26, 144},
      {OperationCode::sphere, {2, 3, 2, 0}, 5, 18},
      {OperationCode::icosphere, {1, 0, 0, 0}, 12, 60},
      {OperationCode::icosphere, {2, 3, 0, 0}, 642, 3840},
      {OperationCode::cylinder, {1, 2, 8, 0}, 34, 96},
      {OperationCode::cylinder, {2, 3, 3, 0}, 14, 36}
  }};
  TestBackend backend;
  std::string error;
  for (const auto &fixture : fixtures) {
    auto value = lowerMeshGenerator(501, 501, 1, fixture.code,
                                    fixture.parameters, contract, error);
    check(value.has_value(), "mesh generator lowers within its contract");
    if (!value) continue;
    const auto &mesh = std::get<GeometryData>(value->data);
    check(mesh.positions.size() == fixture.vertices && mesh.indices.size() == fixture.indices,
          "mesh generator has the declared exact topology");
    for (std::size_t i = 0; i < mesh.positions.size(); ++i) {
      check(mesh.vertexIds[i] == i + 1, "mesh vertex identity follows deterministic construction order");
      const auto &p = mesh.positions[i];
      if (fixture.code == OperationCode::sphere || fixture.code == OperationCode::icosphere)
        check(std::abs(std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z) - fixture.parameters[0]) < 0.00001f,
              "spherical generator vertices lie on the requested radius");
    }
    for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
      const auto &a = mesh.positions.at(mesh.indices[i]);
      const auto &b = mesh.positions.at(mesh.indices[i + 1]);
      const auto &c = mesh.positions.at(mesh.indices[i + 2]);
      const Vec3 u{b.x-a.x,b.y-a.y,b.z-a.z}, v{c.x-a.x,c.y-a.y,c.z-a.z};
      const Vec3 n{u.y*v.z-u.z*v.y,u.z*v.x-u.x*v.z,u.x*v.y-u.y*v.x};
      const auto outward = fixture.code == OperationCode::plane ? n.z
          : n.x*(a.x+b.x+c.x) + n.y*(a.y+b.y+c.y) + n.z*(a.z+b.z+c.z);
      check(outward > 0, "generated triangles are nondegenerate with outward winding");
    }
    auto admitted = admitValue(*value, contract, error);
    check(admitted.has_value(), "generated immutable mesh admits");
    if (!admitted) continue;
    const auto bytes = lowerRuntimePlan(contract, *admitted);
    auto repeated = lowerMeshGenerator(501, 501, 1, fixture.code,
                                       fixture.parameters, contract, error);
    auto repeatedAdmission = admitValue(*repeated, contract, error);
    check(repeatedAdmission && lowerRuntimePlan(contract, *repeatedAdmission) == bytes,
          "repeated generation preserves byte-exact transport");
    roundtrip(*value, contract, {}, "generated mesh roundtrips through runtime transport");
    videohelper::geometry::GeometryCorePlanRuntime runtime(backend);
    const videohelper::geometry::PlanOwnerIdentity previewOwner{1,1,1,1,1,
        videohelper::geometry::PlanUse::preview};
    const videohelper::geometry::PlanOwnerIdentity exportOwner{2,1,1,1,1,
        videohelper::geometry::PlanUse::exportRender};
    auto preview = runtime.admitPreview(previewOwner, bytes, {}, {}, error);
    auto exported = runtime.admitExport(exportOwner, bytes, {}, {}, error);
    check(preview && exported
              && preview->plan->value().receipt().deterministicDigest
                  == exported->plan->value().receipt().deterministicDigest,
          "preview and export admit the same generated mesh");
    auto hostile = *value;
    std::get<GeometryData>(hostile.data).positions[0].x += 0.125f;
    check(!admitValue(hostile, contract, error),
          "retained mesh cannot disagree with its generator operation");
    hostile = *value;
    hostile.operations[0].parameters[3] = 1;
    check(!admitValue(hostile, contract, error), "unknown generator parameters fail closed");
    auto small = contract;
    small.maxVertices = fixture.vertices - 1;
    check(!lowerMeshGenerator(501,501,1,fixture.code,fixture.parameters,small,error),
          "vertex capacity is checked before geometry allocation");
    small = contract;
    small.maxIndices = fixture.indices - 1;
    check(!lowerMeshGenerator(501,501,1,fixture.code,fixture.parameters,small,error),
          "index capacity is checked before geometry allocation");

    auto pointGrid = lowerGrid(601,601,1,2,2,1,contract,error);
    auto points = lowerPointsFromVertices(*pointGrid,602,contractFor(CarrierKind::points3D),error);
    auto instances = lowerInstanceOnPoints(*points,value->stableId,603,
                                           contractFor(CarrierKind::instances3D),error);
    instances->operations.insert(instances->operations.begin(),value->operations.begin(),value->operations.end());
    instances->dispatchCount += value->dispatchCount;
    auto instanceAdmission = admitValue(*instances,contractFor(CarrierKind::instances3D),error);
    check(instanceAdmission.has_value(), "generated meshes retain independent instance-source provenance");
    MaterializedInstancePlan materialized;
    check(validateOperationPlan(*instances,{},&materialized,error)
              && materialized.geometries.count(501) == 1
              && equal(materialized.geometries.at(501),mesh),
          "native instancing reconstructs the exact generated source mesh");
  }
  for (const auto &fixture : std::array<Fixture, 5>{{
      {OperationCode::sphere,{1,3.5f,4,0},0,0},
      {OperationCode::sphere,{1,1000000,1000000,0},0,0},
      {OperationCode::icosphere,{1,9,0,0},0,0},
      {OperationCode::cylinder,{1,2,2,0},0,0},
      {OperationCode::cube,{1,1,std::numeric_limits<float>::infinity(),0},0,0}}})
    check(!lowerMeshGenerator(501,501,1,fixture.code,fixture.parameters,contract,error),
          "invalid and excessive mesh generator requests reject before allocation");
}

void gridLoweringExactTopologyBoundsAndRefusals() {
  auto contract = contractFor(CarrierKind::geometry3D);
  contract.maxVertices = 64;
  contract.maxIndices = 96;
  struct Fixture {
    std::uint32_t columns, rows;
    float spacing;
    std::size_t vertices, indices;
    float halfWidth, halfHeight;
  };
  const std::array<Fixture, 2> fixtures {{
      {2, 2, 1.0f, 4, 6, 0.5f, 0.5f},
      {4, 3, 0.25f, 12, 36, 0.375f, 0.25f},
  }};
  std::string error;
  for (const auto &fixture : fixtures) {
    auto value = lowerGrid(801, 801, 1, fixture.columns, fixture.rows,
                           fixture.spacing, contract, error);
    check(value.has_value(), "grid lowers within its contract");
    if (!value) continue;
    const auto &mesh = std::get<GeometryData>(value->data);
    check(mesh.positions.size() == fixture.vertices &&
              mesh.vertexIds.size() == fixture.vertices &&
              mesh.indices.size() == fixture.indices,
          "grid has the exact vertex and index counts");
    check(value->operations.size() == 1 && value->dispatchCount == 1,
          "grid retains exactly one generator operation");
    float minX = mesh.positions[0].x, maxX = minX;
    float minY = mesh.positions[0].y, maxY = minY;
    for (std::size_t i = 0; i < mesh.positions.size(); ++i) {
      const auto &p = mesh.positions[i];
      minX = std::min(minX, p.x);
      maxX = std::max(maxX, p.x);
      minY = std::min(minY, p.y);
      maxY = std::max(maxY, p.y);
      check(mesh.vertexIds[i] == i + 1,
            "grid vertex identities follow row-major construction order");
      check(p.z == 0.0f, "grid lies on the z=0 plane");
    }
    check(minX == -fixture.halfWidth && maxX == fixture.halfWidth &&
              minY == -fixture.halfHeight && maxY == fixture.halfHeight,
          "grid bounds are centered on the origin at the requested spacing");
    for (const auto index : mesh.indices)
      check(index < mesh.positions.size(), "grid indices stay inside the vertex table");
    for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
      const auto &a = mesh.positions.at(mesh.indices[i]);
      const auto &b = mesh.positions.at(mesh.indices[i + 1]);
      const auto &c = mesh.positions.at(mesh.indices[i + 2]);
      const float cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
      check(cross > 0.0f, "grid triangles are nondegenerate and front-facing (+Z)");
    }
    auto admitted = admitValue(*value, contract, error);
    check(admitted.has_value(), "grid admits as immutable geometry");
    roundtrip(*value, contract, {}, "grid roundtrips through runtime transport");
    auto hostile = *value;
    std::get<GeometryData>(hostile.data).positions[0].x += 0.125f;
    check(!admitValue(hostile, contract, error),
          "retained grid cannot disagree with its generator operation");
    hostile = *value;
    hostile.operations[0].parameters[3] = 1;
    check(!admitValue(hostile, contract, error),
          "grid operation rejects a nonzero reserved parameter");
  }

  const auto refusalText = [&](std::uint32_t columns, std::uint32_t rows,
                               float spacing, const PortContract &c) {
    std::string e;
    return lowerGrid(802, 802, 1, columns, rows, spacing, c, e) ? std::string() : e;
  };
  const std::string dimensionError =
      "Grid lowering has an invalid typed contract or dimensions";
  const std::string capacityError =
      "Grid lowering exceeds its admitted geometry capacity";
  check(refusalText(1, 2, 1.0f, contract) == dimensionError,
        "a single-column grid is refused as invalid dimensions");
  check(refusalText(2, 1, 1.0f, contract) == dimensionError,
        "a single-row grid is refused as invalid dimensions");
  check(refusalText(0, 2, 1.0f, contract) == dimensionError,
        "a zero-column grid is refused as invalid dimensions");
  check(refusalText(2, 2, 0.0f, contract) == dimensionError,
        "a zero-spacing grid is refused");
  check(refusalText(2, 2, -1.0f, contract) == dimensionError,
        "a negative-spacing grid is refused");
  check(refusalText(2, 2, std::numeric_limits<float>::quiet_NaN(), contract) == dimensionError,
        "a NaN-spacing grid is refused");
  check(refusalText(2, 2, std::numeric_limits<float>::infinity(), contract) == dimensionError,
        "an infinite-spacing grid is refused");
  check(refusalText(2, 2, 1.0f, contractFor(CarrierKind::points3D)) == dimensionError,
        "a Grid lowered against a non-Geometry3D contract is refused");
  auto vertexCapped = contract;
  vertexCapped.maxVertices = 5;
  check(refusalText(2, 3, 1.0f, vertexCapped) == capacityError,
        "a 6-vertex grid is refused by a 5-vertex capacity");
  auto indexCapped = contract;
  indexCapped.maxIndices = 23;
  check(refusalText(3, 3, 1.0f, indexCapped) == capacityError,
        "a 24-index grid is refused by a 23-index capacity");
  indexCapped.maxIndices = 24;
  check(refusalText(3, 3, 1.0f, indexCapped).empty(),
        "a 24-index grid is admitted at exactly its 24-index capacity");
  check(refusalText(std::numeric_limits<std::uint32_t>::max(),
                    std::numeric_limits<std::uint32_t>::max(), 1.0f, contract) == capacityError,
        "a maximal grid is refused before any vertex allocation");
}

void curveRibbonsRetainSourceAndTopology() {
  auto curveContract = contractFor(CarrierKind::curves3D);
  auto meshContract = contractFor(CarrierKind::geometry3D);
  meshContract.maxVertices = 16;
  meshContract.maxIndices = 48;
  std::string error;
  TestBackend backend;
  for (const bool circle : {false, true}) {
    auto curve = circle ? lowerCurveCircle(801,801,3,8,1,curveContract,error)
                        : lowerCurveLine(801,801,3,8,2,curveContract,error);
    check(curve.has_value(), "bounded curve generator lowers");
    if (!curve) continue;
    roundtrip(*curve, curveContract, {}, "generated curve survives transport");
    const auto &source = std::get<CurvesData>(curve->data);
    check(source.splines.size() == 1 && source.splines[0].cyclic == circle,
          "line remains open and circle remains closed");
    auto value = lowerCurveToMesh(*curve,802,meshContract,error,0.2f);
    check(value.has_value(), "curve lowers to an indexed ribbon");
    if (!value) continue;
    const auto &mesh = std::get<GeometryData>(value->data);
    check(mesh.positions.size() == 16 && mesh.indices.size() == (circle ? 48u : 42u),
          "ribbon has two vertices per point and two triangles per segment");
    check(value->sourceStableId == 801 && value->sourceRevision == 3
              && value->operations.size() == 2 && value->dispatchCount == 2
              && value->operations[0].code == (circle ? OperationCode::curveCircle : OperationCode::curveLine)
              && value->operations[1].inputStableId == 801,
          "curve conversion retains the complete source operation chain");
    for (std::size_t i = 0; i < source.points.size(); ++i)
      check(mesh.vertexIds[i * 2] == source.points[i].stableId * 2 - 1
                && mesh.vertexIds[i * 2 + 1] == source.points[i].stableId * 2,
            "paired vertex identity derives from stable curve-point identity");
    for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
      const auto &a = mesh.positions[mesh.indices[i]], &b = mesh.positions[mesh.indices[i + 1]],
                 &c = mesh.positions[mesh.indices[i + 2]];
      const auto area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
      check(area > 0, "planar curve ribbon triangles have positive area and consistent winding");
    }
    if (circle)
      check(mesh.indices[mesh.indices.size() - 5] == 0,
            "closed ribbon joins its last point to the first point");
    else
      check(mesh.positions.front().x == -1 && mesh.positions.back().x == 1,
            "line endpoints span its authored length");
    auto admitted = admitValue(*value,meshContract,error);
    check(admitted.has_value(), "generated ribbon passes native value admission");
    if (!admitted) continue;
    const auto bytes = lowerRuntimePlan(meshContract,*admitted);
    roundtrip(*value,meshContract,{},"curve ribbon survives binary runtime transport");
    videohelper::geometry::GeometryCorePlanRuntime runtime(backend);
    auto preview = runtime.admitPreviewText(encodeLoweredPlanText(bytes),{}, {},error);
    auto exported = runtime.admitExportText(encodeLoweredPlanText(bytes),{}, {},error);
    check(preview && exported && preview->plan->value().receipt().deterministicDigest
                                  == exported->plan->value().receipt().deterministicDigest,
          "preview and export retain the same curve ribbon");
    auto hostile = *value;
    hostile.operations.erase(hostile.operations.begin());
    hostile.dispatchCount = 1;
    check(!admitValue(hostile,meshContract,error), "missing curve generator fails admission");
    hostile = *value;
    std::get<GeometryData>(hostile.data).indices[1] = 0;
    check(!admitValue(hostile,meshContract,error), "degenerate retained topology fails operation validation");
    hostile = *value;
    std::get<GeometryData>(hostile.data).vertexIds[0] = 900;
    check(!admitValue(hostile,meshContract,error), "forged paired vertex identity fails operation validation");
    auto small = meshContract;
    small.maxVertices = 15;
    check(!lowerCurveToMesh(*curve,802,small,error), "ribbon vertex budget rejects before allocation");
    small = meshContract;
    small.maxIndices = mesh.indices.size() - 1;
    check(!lowerCurveToMesh(*curve,802,small,error), "ribbon index budget rejects before allocation");
    check(!lowerCurveToMesh(*curve,802,meshContract,error,0), "zero width cannot create degenerate ribbons");
    auto malformed = *curve;
    std::get<CurvesData>(malformed.data).splines[0].firstPoint = 1;
    check(!lowerCurveToMesh(malformed,802,meshContract,error), "malformed spline ranges fail before traversal");
  }
  check(!lowerCurveLine(801,801,1,1,2,curveContract,error), "line requires at least two points");
  check(!lowerCurveLine(801,801,1,8,0,curveContract,error), "line rejects zero length");
  check(!lowerCurveLine(801,801,1,9,2,curveContract,error), "line rejects excess point count");
  auto legacy = lowerCurveCircle(801,801,1,8,1,curveContract,error);
  legacy->operations[0].parameters[1] = 0;
  check(admitValue(*legacy,curveContract,error).has_value(),
        "legacy circle records without a retained point count still admit");
}

void curveAuthoringRetainsEditsAndFields() {
  std::string error;
  auto curvesContract=contractFor(CarrierKind::curves3D);
  curvesContract.maxCurvePoints=64;
  auto meshContract=contractFor(CarrierKind::geometry3D);
  meshContract.maxVertices=512; meshContract.maxIndices=3072;
  auto pointsContract=contractFor(CarrierKind::points3D);
  pointsContract.maxPoints=64;
  auto fieldContract=contractFor(CarrierKind::field,ValueType::floatValue);
  fieldContract.fieldDomain=Domain::curvePoint; fieldContract.maxFieldElements=64;
  auto source=lowerCurveLine(4001,4001,7,3,2,curvesContract,error);
  check(source.has_value(),"curve authoring has a retained source");
  if (!source) return;
  auto resampled=lowerCurveOperation(*source,4002,OperationCode::curveResample,{16,0,0,0},curvesContract,error);
  check(resampled.has_value(),"arc-length resampling lowers");
  if (!resampled) return;
  auto trimmed=lowerCurveOperation(*resampled,4003,OperationCode::curveTrim,{0.1f,0.9f,0,0},curvesContract,error);
  check(trimmed.has_value(),"trim lowers with a fixed point count");
  if (!trimmed) return;
  auto radius=lowerCurveOperation(*trimmed,4004,OperationCode::curveSetRadius,{0.5f,1.5f,0,0},curvesContract,error);
  check(radius.has_value(),"varying per-point radius lowers");
  if (!radius) return;
  const auto &radii=std::get<CurvesData>(radius->data).points;
  check(radii.front().radius==0.5f && radii.back().radius==1.5f,"radius ramp follows each spline's arc length");
  for (const auto code : {OperationCode::positionField,OperationCode::indexField,OperationCode::stableIdentityField,
                         OperationCode::curveTangentField,OperationCode::curveParameterField,OperationCode::curveRadiusField,
                         OperationCode::curveTiltField,OperationCode::curveEndpointField}) {
    auto contract=fieldContract; contract.fieldValueType=elementFieldType(code);
    auto field=lowerElementField(*radius,4010+static_cast<StableId>(code),code,contract,error);
    check(field.has_value(),"typed curve field lowers");
    if (!field) continue;
    roundtrip(*field,contract,{},"typed curve field retains exact values through transport");
    const auto &elements=std::get<FieldData>(field->data).elements;
    if (code==OperationCode::curveParameterField)
      check(elements.front().components[0]==0 && elements.back().components[0]==1,"open curve parameters include both endpoints");
    if (code==OperationCode::curveEndpointField)
      check(elements.front().components[0]==1 && elements.back().components[0]==1 && elements[1].components[0]==0,
            "endpoint field distinguishes open ends from interior points");
    if (code==OperationCode::curveTangentField) {
      auto displaced=lowerVectorDisplacement(*radius,*field,4110,0.2f,curvesContract,error);
      check(displaced.has_value(),"tangent field drives Curve Vector Displacement");
      if (displaced) roundtrip(*displaced,curvesContract,{},"curve field displacement replays its shared source once");
    }
  }
  auto parameter=lowerElementField(*radius,4200,OperationCode::curveParameterField,fieldContract,error);
  if (!parameter) { check(false,"parameter field lowers for tilt"); return; }
  auto tilted=lowerCurveOperation(*radius,4201,OperationCode::curveSetTilt,{0,0,0,0},curvesContract,error,&*parameter);
  check(tilted.has_value(),"curve parameter field drives per-point tilt");
  if (!tilted) return;
  auto waved=lowerCurveOperation(*tilted,4202,OperationCode::curveWave,{0.2f,2,0.25f,1},curvesContract,error);
  check(waved.has_value(),"phase-controlled wave lowers");
  if (!waved) return;
  roundtrip(*waved,curvesContract,{},"curve authoring chain retains all edits through transport");
  const auto &wavePoints=std::get<CurvesData>(waved->data).points;
  check(wavePoints.front().position.y>0.19f && wavePoints.front().stableId==radii.front().stableId,
        "wave changes positions and preserves curve point identities");
  auto loopStart=lowerCurveOperation(*tilted,4250,OperationCode::curveWave,{0.2f,2,0,1},curvesContract,error);
  auto loopEnd=lowerCurveOperation(*tilted,4250,OperationCode::curveWave,{0.2f,2,1,1},curvesContract,error);
  check(loopStart && loopEnd,"both ends of the authored loop phase lower");
  if (loopStart && loopEnd) {
    const auto &a=std::get<CurvesData>(loopStart->data).points, &b=std::get<CurvesData>(loopEnd->data).points;
    for (std::size_t i=0;i<a.size();++i)
      check(a[i].stableId==b[i].stableId && std::abs(a[i].position.y-b[i].position.y)<0.000001f,
            "phase zero and one close the wave loop without changing point identity");
  }
  auto tube=lowerCurveOperation(*waved,4203,OperationCode::curveToTube,{0.1f,8,0,0},meshContract,error);
  check(tube.has_value(),"edited curves lower to a native indexed tube");
  if (!tube) return;
  roundtrip(*tube,meshContract,{},"tube retains curve field dependencies through binary transport");
  const auto &mesh=std::get<GeometryData>(tube->data);
  check(mesh.positions.size()==128 && mesh.indices.size()==720,"tube has bounded rings and profile topology");
  TestBackend backend;
  auto admitted=admitValue(*tube,meshContract,error);
  check(admitted.has_value(),"complete curve chain admits");
  if (admitted) {
    videohelper::geometry::GeometryCorePlanRuntime runtime(backend);
    const auto text=encodeLoweredPlanText(lowerRuntimePlan(meshContract,*admitted));
    auto preview=runtime.admitPreviewText(text,{}, {},error), exported=runtime.admitExportText(text,{}, {},error);
    check(preview && exported && preview->plan->value().receipt().deterministicDigest==
          exported->plan->value().receipt().deterministicDigest,"preview and export admit identical edited tube data");
  }
  auto points=lowerCurveOperation(*waved,4204,OperationCode::pointsFromCurves,{0,0.1f,0,0},pointsContract,error);
  check(points.has_value(),"curve positions and tangent orientations lower for instancing");
  if (points) {
    roundtrip(*points,pointsContract,{},"curve points retain radius and tangent orientation");
    auto instanceContract=contractFor(CarrierKind::instances3D); instanceContract.maxInstances=64;
    auto geometry=lowerMeshGenerator(4205,4205,1,OperationCode::cube,{1,1,1,0},meshContract,error);
    auto instances=lowerInstanceOnPoints(*points,4205,4206,instanceContract,error);
    if (geometry && instances) {
      auto operations=geometry->operations;
      check(appendUniqueOperations(operations,instances->operations,error),"curve instance source recipes combine");
      instances->operations=operations; instances->dispatchCount=operations.size();
      roundtrip(*instances,instanceContract,{},"curve-authored native instances retain exact source geometry");
    } else check(false,"curve instance source lowers");
  }
  auto endpoints=lowerCurveOperation(*waved,4207,OperationCode::pointsFromCurves,{1,0.1f,0,0},pointsContract,error);
  check(endpoints && std::get<PointsData>(endpoints->data).points.size()==2,"endpoint point output selects exactly two open ends");
  auto hostile=*tube;
  std::get<GeometryData>(hostile.data).positions[0].x+=0.1f;
  check(!admitValue(hostile,meshContract,error),"forged curve tube positions fail immutable replay");
  hostile=*tube; hostile.operations[2].parameters[1]=0.5f;
  check(!admitValue(hostile,meshContract,error),"changing retained trim without changing tube data fails");
  auto small=curvesContract; small.maxCurvePoints=15;
  check(!lowerCurveOperation(*source,4300,OperationCode::curveResample,{16,0,0,0},small,error),"resampling rejects excess capacity");
  check(!lowerCurveOperation(*source,4300,OperationCode::curveTrim,{0.5f,0.5f,0,0},curvesContract,error),"empty trim rejects before mesh conversion");
  check(!lowerCurveOperation(*source,4300,OperationCode::curveSetRadius,{0,1,0,0},curvesContract,error),"zero radius cannot create a collapsed tube");
  auto circle=lowerCurveCircle(4301,4301,1,8,1,curvesContract,error);
  if (circle) {
    auto cut=lowerCurveOperation(*circle,4302,OperationCode::curveTrim,{0.25f,0.75f,0,0},curvesContract,error);
    check(cut && !std::get<CurvesData>(cut->data).splines[0].cyclic,"partial circle trim opens the spline");
    auto two=lowerCurveOperation(*circle,4303,OperationCode::curveResample,{2,0,0,0},curvesContract,error);
    check(!two,"cyclic resampling rejects fewer than three points");
  }
  auto irregular=std::get<CurvesData>(source->data);
  irregular.points[0].position.x=0; irregular.points[1].position.x=1; irregular.points[2].position.x=4;
  CurvesData uniform;
  OperationRecord resample{4401,OperationCode::curveResample,4001,4401,{5,0,0,0}};
  check(materializeCurveEdit(irregular,resample,nullptr,uniform,64,error) && uniform.points.size()==5 &&
        uniform.points[2].position.x==2 && uniform.points[3].position.x==3,"resampling measures distance rather than source index");
}

void elementFieldsAndDisplacementRetainProvenance() {
  std::string error;
  const auto meshContract = contractFor(CarrierKind::geometry3D);
  auto mesh = lowerGrid(901,901,7,2,2,1,meshContract,error);
  auto points = lowerPointsFromVertices(*mesh,902,contractFor(CarrierKind::points3D),error);
  auto instances = lowerInstanceOnPoints(*points,mesh->stableId,903,
                                         contractFor(CarrierKind::instances3D),error);
  TestBackend backend;
  for (const auto &source : {*mesh,*points,*instances}) {
    for (const auto code : {OperationCode::positionField,OperationCode::normalField,
                            OperationCode::indexField,OperationCode::stableIdentityField}) {
      auto fieldContract = contractFor(CarrierKind::field,elementFieldType(code));
      fieldContract.fieldDomain = elementDomain(source.carrier);
      fieldContract.fieldInterpolation = Interpolation::constant;
      fieldContract.maxFieldElements = 4;
      auto field = lowerElementField(source,910,code,fieldContract,error);
      check(field.has_value(), "element query lowers on geometry, points, and instances");
      if (!field) continue;
      const auto &values = std::get<FieldData>(field->data).elements;
      check(field->sourceRevision == 7 && field->sourceStableId == 901 && values.size() == 4,
            "element field keeps the original source revision and exact domain cardinality");
      for (std::size_t i = 0; i < values.size(); ++i) {
        const auto expected = code == OperationCode::indexField ? double(i)
            : code == OperationCode::stableIdentityField ? double(i + 1)
            : code == OperationCode::normalField ? 0.0 : (i % 2 == 0 ? -0.5 : 0.5);
        check(values[i].components[0] == expected, "element query returns source values in stable storage order");
        if (code == OperationCode::normalField)
          check(values[i].components[2] == 1, "normal query uses mesh winding and oriented local Z");
      }
      roundtrip(*field,fieldContract,{},"element query survives immutable binary transport");
      auto forged = *field;
      std::get<FieldData>(forged.data).elements.back().components[0] += 1;
      check(!admitValue(forged,fieldContract,error), "forged element query values fail native admission");
      auto small = fieldContract;
      small.maxFieldElements = 3;
      check(!lowerElementField(source,910,code,small,error), "query capacity rejects before allocating fields");
      if (elementFieldType(code) != ValueType::vector) continue;
      const auto outputContract = contractFor(source.carrier);
      auto displaced = lowerVectorDisplacement(source,*field,920,0.25f,outputContract,error);
      check(displaced.has_value(), "vector field displaces its typed source without duplicating shared operations");
      if (!displaced) continue;
      check(displaced->operations.size() == source.operations.size() + 2
                && displaced->operations.back().secondaryInputStableId == field->stableId,
            "displacement records both source and field provenance exactly once");
      FieldData identitiesBefore, identitiesAfter;
      check(materializeElementField(source.data,source.carrier,OperationCode::stableIdentityField,
                                     identitiesBefore,4,error)
                && materializeElementField(displaced->data,source.carrier,OperationCode::stableIdentityField,
                                            identitiesAfter,4,error), "displaced identities remain queryable");
      for (std::size_t i = 0; i < identitiesBefore.elements.size(); ++i)
        check(identitiesBefore.elements[i].components == identitiesAfter.elements[i].components,
              "displacement never changes element identity or order");
      FieldData positions;
      check(materializeElementField(displaced->data,source.carrier,OperationCode::positionField,
                                     positions,4,error), "displaced positions remain queryable");
      check(positions.elements[0].components[code == OperationCode::normalField ? 2 : 0]
                == (code == OperationCode::normalField ? 0.25 : -0.625),
            "normal and position fields produce the expected displacement");
      roundtrip(*displaced,outputContract,{},"displacement survives immutable binary transport");
      auto admitted = admitValue(*displaced,outputContract,error);
      check(admitted.has_value(), "displacement passes operation replay admission");
      if (!admitted) continue;
      videohelper::geometry::GeometryCorePlanRuntime runtime(backend);
      const auto payload = encodeLoweredPlanText(lowerRuntimePlan(outputContract,*admitted));
      const auto preview = runtime.admitPreviewText(payload,{}, {},error);
      const auto exported = runtime.admitExportText(payload,{}, {},error);
      check(preview && exported && preview->plan->value().receipt().deterministicDigest
                                    == exported->plan->value().receipt().deterministicDigest,
            "preview and export use the same displaced value");
      auto wrongField = *displaced;
      wrongField.operations.back().secondaryInputStableId = source.stableId;
      check(!admitValue(wrongField,outputContract,error), "displacement cannot substitute a non-field dependency");
      auto changed = *displaced;
      if (auto *geometry = std::get_if<GeometryData>(&changed.data)) geometry->positions[0].x += 1;
      if (auto *pointData = std::get_if<PointsData>(&changed.data)) pointData->points[0].position.x += 1;
      if (auto *instanceData = std::get_if<InstancesData>(&changed.data)) instanceData->instances[0].transform.translation.x += 1;
      check(!admitValue(changed,outputContract,error), "displaced retained positions must match operation replay");
    }
  }
  auto arbitrary = pointsValue();
  auto &records = std::get<PointsData>(arbitrary.data).points;
  records[0].stableId = kMaximumStableId;
  records[1].stableId = 17;
  FieldData ids;
  check(materializeElementField(arbitrary.data,CarrierKind::points3D,OperationCode::stableIdentityField,ids,2,error)
            && ids.elements[0].components[0] == static_cast<double>(kMaximumStableId)
            && ids.elements[1].components[0] == 17,
        "stable identity fields retain all 53 exact integer bits without renumbering");
  std::swap(records[0],records[1]);
  check(materializeElementField(arbitrary.data,CarrierKind::points3D,OperationCode::stableIdentityField,ids,2,error)
            && ids.elements[1].components[0] == static_cast<double>(kMaximumStableId),
        "stable identity follows the element after reordering");
  records[0].rotation = {0,1,0,0};
  FieldData normals;
  check(materializeElementField(arbitrary.data,CarrierKind::points3D,OperationCode::normalField,normals,2,error)
            && normals.elements[0].components[2] == -1,
        "point normal respects orientation");
  records[0].rotation = {0,0,0,0};
  check(!materializeElementField(arbitrary.data,CarrierKind::points3D,OperationCode::normalField,normals,2,error),
        "normal query rejects a zero quaternion");
  auto invalid = mesh->data;
  FieldData offsets;
  offsets.elements.resize(4);
  offsets.elements.back().components[2] = std::numeric_limits<double>::infinity();
  check(!displaceElements(invalid,CarrierKind::geometry3D,offsets,1,error)
            && equal(std::get<GeometryData>(invalid),std::get<GeometryData>(mesh->data)),
        "a bad late displacement value leaves every source vertex unchanged");
}

void spectrumFieldsRoundtripAndFollow() {
  std::string error;
  auto value = geometryValue();
  const auto source = value.stableId;
  value.operations.push_back({901,OperationCode::evaluateField,901,901,{}});
  value.operations.push_back({902,OperationCode::transform,source,902,{}});
  value.stableId = 902; value.dispatchCount = 3;
  spectrum::Binding binding;
  binding.fieldStableId = 901; binding.sourceGeometryStableId = source;
  binding.consumerGeometryStableId = 902;
  binding.firstBand = 10; binding.lastBand = 12;
  binding.coordinates = {0.0f,0.5f,1.0f};
  binding.gain = 2.0f; binding.bias = -0.25f;
  value.spectrumFields.push_back(binding);
  const auto contract = contractFor(CarrierKind::geometry3D);
  auto admitted = admitValue(value,contract,error);
  check(admitted.has_value(),"spectrum displacement retains typed field and consumer provenance");
  if (!admitted) { std::cerr << error << '\n'; return; }
  const auto bytes = encodeRuntimeValue(*admitted);
  auto decoded = decodeRuntimeValue(bytes,contract,{}, {},error);
  check(decoded && encodeRuntimeValue(*decoded)==bytes,"spectrum controls and coordinates roundtrip exactly");
  auto versionEleven=bytes;
  versionEleven[4]=11; versionEleven.resize(versionEleven.size()-1-sizeof(std::int32_t));
  const auto oldMasterEleven=decodeRuntimeValue(versionEleven,contract,{}, {},error);
  check(oldMasterEleven && oldMasterEleven->descriptor().spectrumFields[0].source==spectrum::Source::master,
        "v11 spectrum bindings retain master analysis by default");
  auto versionTen=versionEleven; versionTen[4]=10;
  check(decodeRuntimeValue(versionTen,contract,{}, {},error).has_value(),"v10 static geometry remains readable");
  auto versionNine=versionEleven; versionNine[4]=9;
  const auto oldMaster=decodeRuntimeValue(versionNine,contract,{}, {},error);
  check(oldMaster && oldMaster->descriptor().spectrumFields[0].source==spectrum::Source::master,
        "v9 spectrum bindings retain master analysis by default");
  auto selected=value;
  selected.spectrumFields[0].source=spectrum::Source::group;
  selected.spectrumFields[0].sourceTrackId=2147483647;
  const auto group=admitValue(selected,contract,error);
  check(group.has_value(),"selected group source accepts the exact int32 stable ID");
  if (group) {
    const auto reopened=decodeRuntimeValue(encodeRuntimeValue(*group),contract,{}, {},error);
    check(reopened && reopened->descriptor().spectrumFields[0].source==spectrum::Source::group
          && reopened->descriptor().spectrumFields[0].sourceTrackId==2147483647,
          "selected spectrum source and full-width stable track ID survive transport");
  }
  selected.spectrumFields[0].sourceTrackId=-1;
  check(!admitValue(selected,contract,error),"selected source without an exact track ID fails closed");
  auto versionSix=versionNine;
  versionSix[4]=6; versionSix.resize(versionSix.size()-sizeof(std::uint32_t));
  check(decodeRuntimeValue(versionSix,contract,{}, {},error).has_value(),"v6 field-math transport remains readable");
  auto versionFive=versionSix; versionFive[4]=5;
  check(decodeRuntimeValue(versionFive,contract,{}, {},error).has_value(),"v5 score transport remains readable");
  auto versionFour=versionFive;
  versionFour[4]=4; versionFour.resize(versionFour.size()-sizeof(std::uint32_t));
  check(decodeRuntimeValue(versionFour,contract,{}, {},error).has_value(),"v4 audio bindings remain readable");
  auto versionThree=versionFour;
  versionThree[4]=3; versionThree.resize(versionThree.size()-1-14*sizeof(float));
  check(decodeRuntimeValue(versionThree,contract,{}, {},error).has_value(),"v3 spectrum bindings remain readable");
  auto versionTwo=versionThree;
  versionTwo[4]=2; versionTwo.resize(versionTwo.size()-1-6*sizeof(float));
  const auto oldBinding=decodeRuntimeValue(versionTwo,contract,{}, {},error);
  check(oldBinding && oldBinding->descriptor().spectrumFields.front().target==spectrum::Target::vertexDisplacement,
        "v2 vertex spectrum bindings remain readable with their original target");
  for (std::size_t cut=bytes.size()-20;cut<bytes.size();++cut)
    check(!decodeRuntimeValue(bytes.data(),cut,contract,{}, {},error),"truncated spectrum coordinates fail before allocation");
  auto invalid=value;
  invalid.spectrumFields[0].consumerGeometryStableId=999;
  check(!admitValue(invalid,contract,error),"spectrum rejects a forged consumer");
  invalid=value; invalid.spectrumFields[0].coordinates.pop_back();
  check(!admitValue(invalid,contract,error),"spectrum rejects mismatched vertex coordinates");
  invalid=value; invalid.spectrumFields[0].lastBand=64;
  check(!admitValue(invalid,contract,error),"spectrum rejects unavailable bands");
  TestBackend backend;
  videohelper::geometry::GeometryCorePlanRuntime runtime(backend);
  const auto lowered=lowerRuntimePlan(contract,*admitted);
  auto preview=runtime.admitPreview(lowered,{}, {},error);
  auto exported=runtime.admitExport(lowered,{}, {},error);
  check(preview && exported && encodeRuntimeValue(preview->plan->value())
      ==encodeRuntimeValue(exported->plan->value()),"preview and export admit the same immutable spectrum binding");
  spectrum::Bands bands {}; bands[10]=0.2f; bands[11]=0.4f; bands[12]=0.8f;
  check(std::abs(spectrum::sample(binding,bands,0.25f)-0.35f)<1.0e-6f,
        "band interpolation applies authored gain and bias");
  binding.interpolate=false;
  check(std::abs(spectrum::sample(binding,bands,0.25f)-0.55f)<1.0e-6f,
        "nearest-band sampling follows the selected range");
  binding.normalize=true;
  auto normalized=spectrum::normalizeBands(bands,binding);
  check(normalized[12]==1.0f && normalized[11]==0.5f,"normalization uses the selected band peak");
  binding.attackSeconds=0.1f; binding.releaseSeconds=0.4f;
  const spectrum::FeaturesAt stream=[](double seconds) {
    spectrum::Bands result {}; result.fill(seconds>=0.5 && seconds<1.0 ? 0.8f : 0.0f); return result;
  };
  spectrum::Follower state, otherState;
  spectrum::Bands attack {}, release {}, repeated {}, exportedBands {};
  check(spectrum::evaluate(binding,stream(0.7),0.7,stream,state,attack,error)
      && attack[10]>0.8f && attack[10]<1.0f,"attack follows a finite shared feature stream");
  check(spectrum::evaluate(binding,stream(1.2),1.2,stream,state,release,error)
      && release[10]>0.0f && release[10]<attack[10],"release follows falling energy");
  check(spectrum::evaluate(binding,stream(0.7),0.7,stream,state,repeated,error)
      && spectrum::evaluate(binding,stream(0.7),0.7,stream,otherState,exportedBands,error)
      && repeated==attack && repeated==exportedBands,"seek, repeated frame and export use identical baked following");
  check(!spectrum::evaluate(binding,bands,-1,{},state,repeated,error),"negative frame times are rejected");
  auto plain=admitValue(geometryValue(),contract,error);
  auto legacy=encodeRuntimeValue(*plain);
  legacy[4]=1; legacy.resize(legacy.size()-3*sizeof(std::uint32_t));
  check(decodeRuntimeValue(legacy,contract,{}, {},error).has_value(),"v1 static geometry remains readable");
}

void spectralHistoryRetainsTimelineAndBounds() {
  std::string error;
  PortContract contract;
  const auto value=spectralHistoryTerrain(contract,error);
  const auto admitted=value ? admitValue(*value,contract,error) : std::nullopt;
  check(admitted.has_value(),"history displacement retains typed field and consumer identity");
  if (!admitted) { std::cerr << error << '\n'; return; }
  const auto bytes=encodeRuntimeValue(*admitted);
  const auto reopened=decodeRuntimeValue(bytes,contract,{}, {},error);
  check(reopened && encodeRuntimeValue(*reopened)==bytes,"history controls and row coordinates roundtrip");
  for (std::size_t cut=bytes.size()-80;cut<bytes.size();++cut)
    check(!decodeRuntimeValue(bytes.data(),cut,contract,{}, {},error),"truncated history coordinates fail before publication");
  auto binding=value->spectrumFields.front();
  spectrum::Follower state, exportState;
  std::vector<spectrum::Bands> rows, repeated, exported;
  check(spectrum::evaluateHistory(binding,{},1,spectralHistoryPulse,false,false,state,rows,error)
      && rows[0][8]==0 && rows[2][8]==0.8f,
      "history rows sample past project time while the current spectrum is silent");
  check(spectrum::sampleHistory(binding,rows,8)==0.8f && spectrum::sampleHistory(binding,rows,0)==0,
      "independent frequency and history coordinates reach the requested vertex");
  binding.attackSeconds=0.1f; binding.releaseSeconds=0.25f; binding.normalize=true;
  check(spectrum::evaluateHistory(binding,{},1,spectralHistoryPulse,false,false,state,rows,error)
      && rows[2][8]>0 && rows[2][8]<1,"history following and normalization use the shared spectrum controls");
  check(spectrum::evaluateHistory(binding,{},0.2,spectralHistoryPulse,false,false,state,repeated,error)
      && spectrum::evaluateHistory(binding,{},1,spectralHistoryPulse,false,false,state,repeated,error)
      && spectrum::evaluateHistory(binding,{},1,spectralHistoryPulse,false,true,exportState,exported,error)
      && rows==repeated && rows==exported,"seek, loop and fresh export reproduce history following exactly");
  binding.history.useTimeline=false; binding.history.currentTimeSeconds=1;
  check(spectrum::evaluateHistory(binding,{},200,spectralHistoryPulse,false,false,state,repeated,error)
      && repeated==rows,"authored current time overrides the timeline deterministically");
  binding.history.useTimeline=true;
  check(!spectrum::evaluateHistory(binding,{},1,{},false,false,state,rows,error)
      && error.find("baked audio analysis")!=std::string::npos,"missing project analysis diagnoses");
  std::size_t calls=0;
  binding.history.depth=128; binding.history.spacingSeconds=0.15f;
  const spectrum::FeaturesAt counted=[&](double time) { ++calls; return spectralHistoryPulse(time); };
  check(spectrum::evaluateHistory(binding,{},100000,counted,false,true,state,rows,error)
      && calls<=2600 && rows.size()==128,"large timeline times retain a bounded history and follower work window");
  auto invalid=*value;
  invalid.spectrumFields.front().history.depth=129;
  check(!admitValue(invalid,contract,error),"history row capacity is enforced before native execution");
  invalid=*value; invalid.spectrumFields.front().history.spacingSeconds=10;
  check(!admitValue(invalid,contract,error),"history time span is capped at 20 seconds");
  invalid=*value; invalid.spectrumFields.front().history.coordinates.pop_back();
  check(!admitValue(invalid,contract,error),"history coordinates must match source vertex cardinality");
  binding=value->spectrumFields.front(); binding.history.live=true;
  binding.history.depth=3; binding.history.spacingSeconds=0.125f;
  state={};
  spectrum::Bands pulse {}; pulse[8]=0.7f;
  check(spectrum::evaluateHistory(binding,pulse,0.125,{},true,false,state,rows,error)
      && rows[0][8]==0.7f && rows[1][8]==0,"live history starts empty and records only received input");
  check(spectrum::evaluateHistory(binding,{},0.25,{},true,false,state,rows,error)
      && rows[1][8]==0.7f,"live history retains a received older sample");
  check(spectrum::evaluateHistory(binding,{},0.125,{},true,false,state,rows,error)
      && rows[1][8]==0,"live history resets when the timeline loops backward");
  check(spectrum::evaluateHistory(binding,pulse,0.25,{},true,false,state,rows,error)
      && spectrum::evaluateHistory(binding,{},1,{},true,false,state,rows,error)
      && rows[1][8]==0,"a live input gap longer than 250 ms clears old rows");
  check(!spectrum::evaluateHistory(binding,pulse,1,{},true,true,state,rows,error)
      && error.find("cannot be exported")!=std::string::npos,"live-only history export is explicit");
  check(!spectrum::evaluateHistory(binding,pulse,1,{},false,false,state,rows,error),"missing live master input diagnoses");
  state={};
  check(spectrum::evaluateHistory(binding,pulse,2,{},true,false,state,rows,error,1,1)
      && spectrum::evaluateHistory(binding,{},2.125,{},true,false,state,rows,error,2,1)
      && rows[1][8]==0,"canonical seek generation resets even a small forward seek");
  for (int frame=0;frame<1800;++frame)
    spectrum::evaluateHistory(binding,pulse,frame/60.0,{},true,false,state,rows,error);
  check(state.history.size()<=spectrum::kMaximumLiveHistoryFrames,"live presentation history has bounded storage");
}

void spectralHistoryRetainedFieldsReplayTypedValues() {
  std::string error;
  for (unsigned domain=1;domain<4;++domain) for (const auto type : {ValueType::floatValue,ValueType::vector,ValueType::color}) {
    std::optional<ValueDescriptor> source;
    if (domain==3) source=instanceAppearanceFixture(0,error);
    else if (domain==2) {
      PortContract curves; curves.carrier=CarrierKind::curves3D; curves.maxCurvePoints=8; curves.maxSplines=1;
      source=lowerCurveCircle(9701,9701,1,8,0.7f,curves,error);
    } else {
      auto grid=lowerGrid(9701,9701,1,2,2,1.2f,constructedFieldMeshContract(),error);
      PortContract points; points.carrier=CarrierKind::points3D; points.maxPoints=4;
      source=grid ? lowerPointsFromVertices(*grid,9702,points,error) : std::nullopt;
    }
    check(bool(source),"retained history has a real typed carrier"); if (!source) continue;
    auto binding=spectrumDomainBinding(); binding.source=spectrum::Source::track; binding.sourceTrackId=71;
    binding.history.enabled=true; binding.history.depth=4; binding.history.spacingSeconds=0.25f;
    auto settings=spectrumDomainSettings(elementDomain(source->carrier),type,0);
    settings.values[4]=0; settings.values[5]=-1; settings.values[6]=1;
    const auto contract=constructedFieldContract(settings.domain,type);
    const auto field=lowerAudioSpectrumField(*source,9710,binding,settings,contract,error);
    check(bool(field),"all retained domains lower Float, Vector and Color history"); if (!field) continue;
    const auto& admittedBinding=field->spectrumFields.back();
    const auto count=admittedBinding.coordinates.size();
    check(count>=3 && admittedBinding.history.coordinates.front()==0 && admittedBinding.history.coordinates.back()==1,
        "history row index covers current through oldest retained elements");
    const auto admitted=admitValue(*field,contract,error);
    const auto bytes=admitted ? encodeRuntimeValue(*admitted) : std::vector<std::uint8_t>{};
    const auto reopened=decodeRuntimeValue(bytes,contract,{}, {},error);
    check(reopened && encodeRuntimeValue(*reopened)==bytes
        && reopened->descriptor().spectrumFields.back().sourceTrackId==71,
        "typed history retains exact source and unchanged transport encoding");
    if (domain==2) {
      ResourceLimits domainLimit; domainLimit.maxVertices=1;
      check(bool(decodeRuntimeValue(bytes,contract,domainLimit,{},error)),
          "curve history transport uses its existing curve-point cap instead of a mesh cap");
      domainLimit.maxCurvePoints=count-1;
      check(!decodeRuntimeValue(bytes,contract,domainLimit,{},error),
          "curve history transport refuses its actual domain capacity overflow");
    }
    RuntimeFieldEvaluation runtime;
    unsigned evaluations=0, samples=0;
    double seconds=1;
    spectrum::Follower follower;
    runtime.spectrumAt=[](const auto&,auto&,std::string& diagnostic) { diagnostic="Current bands cannot replace history"; return false; };
    runtime.spectrumHistoryAt=[&](const auto& selected,auto& rows,std::string& diagnostic) {
      ++evaluations;
      if (!spectrum::sameBinding(selected,admittedBinding)) { diagnostic="Replaced field/source identity"; return false; }
      return spectrum::evaluateHistory(selected,{},seconds,[&](double at) {
        ++samples; spectrum::Bands bands{}; if (at==0.5) bands.fill(0.8f); return bands;
      },false,false,follower,rows,diagnostic);
    };
    ValueDescriptor evaluated;
    const auto replayed=validateOperationPlan(*field,{},nullptr,error,&evaluated,&runtime);
    check(replayed,"ordered replay materializes authentic history"); if (!replayed) continue;
    check(evaluations==1 && samples==4,"one admitted field evaluates one exact-time history per frame");
    const auto first=std::get<FieldData>(evaluated.data).elements;
    for (std::size_t i=0;i<first.size();++i) {
      const auto row=static_cast<unsigned>(std::round(static_cast<double>(i)/(count-1)*3));
      const double amount=row==2 ? 0.8f : 0;
      const float expected=type==ValueType::floatValue ? static_cast<float>(amount) : static_cast<float>(amount)*admittedBinding.direction[0];
      check(first[i].components[0]==expected
          && first[i].components[1]==0 && first[i].components[2]==0
          && first[i].components[3]==(type==ValueType::color ? 1 : 0),
          "literal past pulse becomes the exact typed field element");
    }
    seconds=0.1;
    check(validateOperationPlan(*field,{},nullptr,error,&evaluated,&runtime)
        && std::all_of(std::get<FieldData>(evaluated.data).elements.begin(),std::get<FieldData>(evaluated.data).elements.end(),
          [](const auto& e){return e.components[0]==0;}),"backward replay has no cached future history");
    seconds=1;
    check(validateOperationPlan(*field,{},nullptr,error,&evaluated,&runtime)
        && std::equal(first.begin(),first.end(),std::get<FieldData>(evaluated.data).elements.begin(),
          [](const auto& a,const auto& b){return a.components==b.components;}),"loop replay restores exact typed history values");
    auto carrierContract=contractFor(source->carrier); carrierContract.maxAttributes=kMaximumAttributes;
    const auto stored=lowerNamedAttribute(*source,&*field,9711,OperationCode::storeNamedAttribute,
        "past-audio",type,carrierContract,error);
    check(bool(stored),"generic ordered Store Attribute consumes every retained history type");
    if (stored) {
      ValueDescriptor consumed; evaluations=0;
      const auto consumedFrame=validateOperationPlan(*stored,{},nullptr,error,&consumed,&runtime);
      const auto* attribute=consumedFrame ? findNamedAttribute(consumed.attributes,"past-audio") : nullptr;
      check(attribute && evaluations==1 && attribute->elements.size()==first.size()
          && std::equal(first.begin(),first.end(),attribute->elements.begin(),
              [](const auto& a,const auto& b){return a.components==b.components;}),
          "ordered carrier consumer stores authentic typed history once per field/frame");
    }
    runtime.spectrumHistoryAt={};
    check(!validateOperationPlan(*field,{},nullptr,error,&evaluated,&runtime),"missing selected history never reuses a prior field");
    auto invalid=*field; invalid.spectrumFields.back().history.coordinates[0]=0.25f;
    check(!admitValue(invalid,contract,error),"forged row coordinates refuse before runtime publication");
    invalid=*field; invalid.spectrumFields.back().history.coordinates.pop_back();
    check(!admitValue(invalid,contract,error),"row cardinality must match the exact retained carrier");
    invalid=*field; invalid.spectrumFields.back().history.coordinates[0]=std::numeric_limits<float>::quiet_NaN();
    check(!admitValue(invalid,contract,error),"nonfinite history row coordinate refuses");
    invalid=*field; invalid.spectrumFields.back().history.spacingSeconds=10;
    check(!admitValue(invalid,contract,error),"typed history keeps the 20-second span cap");
    invalid=*field; invalid.operations.back().field.values[4]=4;
    check(!admitValue(invalid,contract,error),"unknown authored row mode refuses");
    invalid=*field; invalid.operations.back().field.mode=8;
    check(!admitValue(invalid,contract,error),"unknown authored band mode refuses");
    invalid=*field; invalid.spectrumFields.back().history.depth=129;
    check(!admitValue(invalid,contract,error),"typed history keeps the 128-row cap");
    invalid=*field; invalid.spectrumFields.back().history.live=true;
    check(!admitValue(invalid,contract,error),"selected track history does not claim live master authority");
    auto small=contract; small.maxFieldElements=count-1;
    check(!admitValue(*field,small,error),"retained history obeys the existing field port capacity");
    settings.mode=1; settings.values[4]=1;
    const auto positioned=lowerAudioSpectrumField(*source,9710,binding,settings,contract,error);
    check(bool(positioned),"authored band and history position modes use the same retained carrier");
    if (positioned) {
      FieldData positions;
      materializeElementField(source->data,source->carrier,OperationCode::positionField,positions,kMaximumFieldElements,error);
      const auto& sampled=positioned->spectrumFields.back();
      for (std::size_t i=0;i<positions.elements.size();++i)
        check(sampled.coordinates[i]==static_cast<float>(std::clamp(positions.elements[i].components[0],0.0,1.0))
            && sampled.history.coordinates[i]==static_cast<float>(std::clamp((positions.elements[i].components[0]+1)/2,0.0,1.0)),
            "independent authored ranges produce literal band and row coordinates");
    }
    BudgetReceipt receipt;
    check(inspectValue(*field,{},receipt,error),"ordered history owns its bounded workspace receipt");
    ResourceLimits limit; limit.maxBufferBytes=receipt.bufferBytes-1;
    check(!admitValue(*field,contract,limit,{},error),"history workspace obeys checked byte admission");
  }
  PortContract mesh;
  const auto terrain=spectralHistoryTerrain(mesh,error);
  if (terrain) {
    auto live=terrain->spectrumFields.back(); live.history.live=true;
    live.history.depth=3; live.history.spacingSeconds=0.125f;
    spectrum::Follower state; std::vector<spectrum::Bands> rows;
    spectrum::Bands pulse{}; pulse[8]=0.7f;
    check(spectrum::evaluateHistory(live,pulse,2,{},true,false,state,rows,error,1,1),"live retained field receives its authentic frame");
    ++live.fieldStableId;
    check(spectrum::evaluateHistory(live,{},2.125,{},true,false,state,rows,error,1,1) && rows[1][8]==0,
        "field/source replacement clears previously received live rows");
    check(spectrum::evaluateHistory(live,pulse,2.25,{},true,false,state,rows,error,1,1)
        && spectrum::evaluateHistory(live,{},2.375,{},true,false,state,rows,error,1,2) && rows[1][8]==0,
        "canonical loop generation clears live history without a backward time jump");
    check(!spectrum::evaluateHistory(live,pulse,2.4,{},false,false,state,rows,error)
        && state.history.empty(),"missing live input drops cached prior history");
  }
  auto deferred=deferredParticleSpectrumFixture(error);
  if (deferred) {
    deferred->spectrumFields.back().history.enabled=true;
    check(!admitValue(*deferred,particlePointsInstanceContract(),{},particlePointsFixtureContext(),error),
        "history explicitly refuses unresolved deferred topology");
  }
}

void spectrumInstancesRetainBandIdentity() {
  std::string error;
  const auto meshContract = contractFor(CarrierKind::geometry3D);
  auto contract = contractFor(CarrierKind::instances3D);
  contract.maxInstances = 64;
  const auto mesh = lowerGrid(910,910,1,2,2,1,meshContract,error);
  check(mesh.has_value(),"Spectrum Instancer source is replayable Geometry3D");
  if (!mesh) return;
  spectrum::Binding binding;
  binding.firstBand=10; binding.lastBand=13;
  binding.direction={0,0.5f,0}; binding.scaleResponse={1,2,3};
  for (int arrangement=0;arrangement<3;++arrangement) {
    auto value=lowerSpectrumInstancer(*mesh,911,binding,{static_cast<float>(arrangement),2,2,2},contract,error);
    auto admitted=value ? admitValue(*value,contract,error) : std::nullopt;
    check(admitted.has_value(),"line, ring and grid instances retain source geometry and controls");
    if (!admitted) { std::cerr << error << '\n'; continue; }
    const auto bytes=encodeRuntimeValue(*admitted);
    const auto decoded=decodeRuntimeValue(bytes,contract,{}, {},error);
    check(decoded && encodeRuntimeValue(*decoded)==bytes,"instance spectrum v3 transport roundtrips exactly");
    MaterializedInstancePlan replay;
    check(validateOperationPlan(*value,{},&replay,error) && replay.geometries.size()==1
        && replay.instances.instances.size()==4,"instancer replays one source mesh and four instance transforms");
    const auto& instances=std::get<InstancesData>(value->data).instances;
    check(instances.front().stableId==11 && instances.back().stableId==14,
          "instance identity is the absolute band plus one");
    if (arrangement==0)
      check(instances.front().transform.translation.x==-1 && instances.back().transform.translation.x==1,
            "line layout includes both authored endpoints");
    if (arrangement==1)
      check(std::abs(instances[1].transform.translation.y-1)<1.0e-6f,
            "ring layout distributes bins around the authored ellipse");
    if (arrangement==2)
      check(instances.front().transform.translation.y==-1 && instances.back().transform.translation.y==1,
            "grid layout covers the authored rows");
    auto forged=*value;
    std::get<InstancesData>(forged.data).instances[1].stableId=60;
    check(!admitValue(forged,contract,error),"forged per-band identity is rejected against replay");
    forged=*value; forged.spectrumFields[0].coordinates[1]=0;
    check(!admitValue(forged,contract,error),"relabeling a bin's spectrum sample is rejected");
    forged=*value; forged.spectrumFields[0].sourceGeometryStableId=999;
    check(!admitValue(forged,contract,error),"a forged spectrum source is rejected");
    forged=*value; forged.spectrumFields[0].baseScale[0]=std::numeric_limits<float>::infinity();
    check(!admitValue(forged,contract,error),"non-finite instance scale is rejected");
  }
  binding.firstBand=12;
  auto narrower=lowerSpectrumInstancer(*mesh,911,binding,{0,2,2,2},contract,error);
  check(narrower && std::get<InstancesData>(narrower->data).instances.front().stableId==13,
        "changing the selected range preserves the surviving absolute-band IDs");
  contract.maxInstances=1;
  check(!lowerSpectrumInstancer(*mesh,911,binding,{0,2,2,2},contract,error),"instance capacity rejects before allocation");
  binding.firstBand=binding.lastBand;
  auto single=lowerSpectrumInstancer(*mesh,911,binding,{0,2,2,2},contract,error);
  check(single && admitValue(*single,contract,error)
      && std::get<InstancesData>(single->data).instances.front().transform.translation.x==0,
        "one selected band gives one centered instance");
  binding.lastBand=64;
  check(!lowerSpectrumInstancer(*mesh,911,binding,{0,2,2,2},contract,error),"a nonexistent band is rejected");
}

void spectrumInstancerLayoutsRetainPlacementAuthority() {
  std::string error;
  const auto contract=appearanceInstanceContract();
  for (unsigned mode=3;mode<=5;++mode) {
    auto value=spectrumInstancerLayoutFixture(mode,error);
    auto admitted=value ? admitValue(*value,contract,error) : std::nullopt;
    check(admitted.has_value(),"curve, point-set and mesh spectrum layouts admit exact retained sources");
    if (!admitted) { std::cerr<<error<<'\n'; continue; }
    const auto bytes=encodeRuntimeValue(*admitted);
    const auto decoded=decodeRuntimeValue(bytes,contract,{}, {},error);
    check(decoded && encodeRuntimeValue(*decoded)==bytes,"new spectrum layouts roundtrip existing v12 operation bytes exactly");
    MaterializedInstancePlan replay;
    check(validateOperationPlan(*value,{},&replay,error) && replay.instances.instances.size()==4,
        "layout replay retains four absolute selected-band identities");
    check(replay.materialTables[value->stableId].count(0)==1
        && replay.materialTables[value->stableId][0].secondaryInputStableId==9803,
        "all layouts preserve the prototype's authored material table");
    const auto& instances=std::get<InstancesData>(value->data).instances;
    check(instances.front().stableId==3 && instances.back().stableId==6
        && instances.front().sourceStableId==9802,"placement never replaces prototype or band identity");
    if (mode==3) check(!equal(instances.front().transform.translation,instances.back().transform.translation),
        "cyclic curve placement does not duplicate its closing endpoint");
    auto forged=*value;
    forged.operations.back().secondaryInputStableId=99999;
    check(!admitValue(forged,contract,error),"missing retained placement source refuses");
    forged=*value; forged.operations.back().parameters[0]=6;
    check(!admitValue(forged,contract,error),"unsupported SDF layout refuses instead of inventing surface points");
    forged=*value; std::get<InstancesData>(forged.data).instances[1].transform.translation.x+=0.25f;
    check(!admitValue(forged,contract,error),"forged placement transform refuses against replay");
    forged=*value; forged.operations.insert(forged.operations.begin(),forged.operations.front()); ++forged.dispatchCount;
    check(!admitValue(forged,contract,error),"duplicate retained operation identity refuses");
    ResourceLimits limits; limits.maxBufferBytes=admitted->receipt().bufferBytes-1;
    check(!admitValue(*value,contract,limits,{},error),"retained placement and workspace bytes obey the effective budget");
    limits={}; limits.maxVertices=1;
    check(!admitValue(*value,contract,limits,{},error),"prototype and mesh placement obey the effective vertex cap");
    limits={};
    if (mode==3) limits.maxCurvePoints=1;
    else if (mode==4) limits.maxPoints=1;
    else limits.maxIndices=1;
    check(!admitValue(*value,contract,limits,{},error),"each placement carrier obeys its effective cardinality cap");
  }
  const auto meshContract=constructedFieldMeshContract();
  auto prototype=lowerMeshGenerator(9830,9830,1,OperationCode::plane,{0.5f,0.5f,0,0},meshContract,error);
  PortContract curvesContract; curvesContract.carrier=CarrierKind::curves3D;
  curvesContract.maxCurvePoints=8; curvesContract.maxSplines=1;
  auto curve=lowerCurveLine(9831,9831,1,3,2,curvesContract,error);
  spectrum::Binding binding; binding.firstBand=10; binding.lastBand=12;
  auto curveInstances=prototype && curve ? lowerSpectrumInstancer(*prototype,9832,binding,{3,2,2,2},contract,error,&*curve) : std::nullopt;
  check(curveInstances.has_value(),"curve placement lowers independently authored line geometry");
  if (curveInstances) {
    const auto& instances=std::get<InstancesData>(curveInstances->data).instances;
    check(instances[0].transform.translation.x==-1 && instances[1].transform.translation.x==0
        && instances[2].transform.translation.x==1,"independent arc-length oracle covers exact line endpoints and midpoint");
  }
  auto grid=lowerGrid(9833,9833,1,2,2,2,meshContract,error);
  PortContract pointsContract; pointsContract.carrier=CarrierKind::points3D; pointsContract.maxPoints=4;
  auto points=grid ? lowerPointsFromVertices(*grid,9834,pointsContract,error) : std::nullopt;
  auto pointInstances=prototype && points ? lowerSpectrumInstancer(*prototype,9835,binding,{4,2,2,2},contract,error,&*points) : std::nullopt;
  check(pointInstances.has_value(),"point placement lowers the released Points3D authority");
  if (pointInstances && points) {
    const auto& data=std::get<PointsData>(points->data).points;
    const auto& instances=std::get<InstancesData>(pointInstances->data).instances;
    check(equal(instances[0].transform.translation,data[0].position)
        && equal(instances[1].transform.translation,data[2].position)
        && equal(instances[2].transform.translation,data[3].position),
        "independent stable-rank oracle selects point zero, two and three");
    auto operation=pointInstances->operations.back(); auto reordered=points->data;
    std::reverse(std::get<PointsData>(reordered).points.begin(),std::get<PointsData>(reordered).points.end());
    InstancesData replay;
    check(materializeSpectrumInstances(operation,pointInstances->spectrumFields[0],replay,error,&reordered)
        && equal(replay,std::get<InstancesData>(pointInstances->data)),"point record reorder preserves stable placement associations");
    auto bad=*points; std::get<PointsData>(bad.data).points[1].stableId=std::get<PointsData>(bad.data).points[0].stableId;
    check(!lowerSpectrumInstancer(*prototype,9835,binding,{4,2,2,2},contract,error,&bad),"duplicate placement point identities refuse");
    bad=*points; std::get<PointsData>(bad.data).points[0].position.x+=1;
    check(!lowerSpectrumInstancer(*prototype,9835,binding,{4,2,2,2},contract,error,&bad),"stale placement data refuses against its source operation");
    bad=*points; std::get<PointsData>(bad.data).points.clear();
    check(!lowerSpectrumInstancer(*prototype,9835,binding,{4,2,2,2},contract,error,&bad),"empty placement cannot invent point samples");
    check(!lowerSpectrumInstancer(*prototype,9835,binding,{5,2,2,2},contract,error,&*points),"wrong placement carrier refuses");
    check(!lowerSpectrumInstancer(*prototype,9835,binding,{4,2,2,2},contract,error),"unconnected placement source refuses");
    bad=*points; bad.spectrumFields.push_back(pointInstances->spectrumFields[0]);
    check(!lowerSpectrumInstancer(*prototype,9835,binding,{4,2,2,2},contract,error,&bad),"legacy post-draw spectral composition stays refused");
  }
  if (prototype && grid) {
    auto meshInstances=lowerSpectrumInstancer(*prototype,9836,binding,{5,2,2,2},contract,error,&*grid);
    check(meshInstances.has_value(),"mesh placement uses retained triangle surfaces");
    if (meshInstances) for (const auto& instance : std::get<InstancesData>(meshInstances->data).instances) {
      const auto& position=instance.transform.translation;
      check(position.x>=-1 && position.x<=1 && position.y>=-1 && position.y<=1 && position.z==0,
          "independent plane oracle bounds every distributed surface point");
    }
    auto bad=*grid; std::get<GeometryData>(bad.data).indices[0]=9999;
    check(!lowerSpectrumInstancer(*prototype,9836,binding,{5,2,2,2},contract,error,&bad),"invalid mesh placement topology refuses");
    bad=*grid;
    for (auto& position : std::get<GeometryData>(bad.data).positions) position={};
    check(!lowerSpectrumInstancer(*prototype,9836,binding,{5,2,2,2},contract,error,&bad),"zero-area mesh placement refuses");
  }
}

void retainedGeometryOperations() {
  auto contract = contractFor(CarrierKind::geometry3D);
  contract.maxVertices=64; contract.maxIndices=256;
  std::string error;
  const auto cube = lowerMeshGenerator(2001,2001,1,OperationCode::cube,{1,2,3,0},contract,error);
  check(cube.has_value(),"geometry operations fixture creates a cube");
  if (!cube) return;
  Transform transform;
  transform.translation={2,3,4}; transform.scale={2,-3,0.5f};
  transform.rotation={0,0,0.7071067812f,0.7071067812f};
  auto transformed=lowerTransformGeometry(*cube,2002,transform,contract,error);
  check(transformed.has_value(),"TRS transform lowers");
  if (!transformed) return;
  auto material=lowerAssignMaterial(*transformed,2003,2001,contract,error);
  check(material.has_value(),"material identity may equal an existing node identity without colliding");
  if (!material) return;
  auto secondMaterial=lowerAssignMaterial(*material,2004,2001,contract,error);
  check(secondMaterial.has_value(),"repeated material assignments retain separate operation identities");
  if (!secondMaterial) return;
  roundtrip(*secondMaterial,contract,{},"TRS and material assignment survive runtime transport");
  MaterializedInstancePlan replay;
  check(validateOperationPlan(*secondMaterial,{},&replay,error) &&
        equal(replay.geometries.at(2004),std::get<GeometryData>(secondMaterial->data)) &&
        replay.materialStableId==2001 && retainedMaterialIdentity(*secondMaterial)==2001,
        "material assignment retains exact transformed vertices and material");
  auto fieldContract=contractFor(CarrierKind::field,ValueType::vector);
  fieldContract.fieldDomain=Domain::vertex;
  fieldContract.fieldInterpolation=Interpolation::constant;
  fieldContract.maxFieldElements=64;
  auto positions=lowerElementField(*secondMaterial,2005,OperationCode::positionField,fieldContract,error);
  check(positions && admitValue(*positions,fieldContract,error),"downstream position query replays transform and material");
  auto center=lowerElementField(*secondMaterial,2006,OperationCode::boundsCenterField,fieldContract,error);
  check(center && std::get<FieldData>(center->data).elements.size()==1 &&
        std::get<FieldData>(center->data).elements[0].components==std::array<double,4>{2,3,4,0},
        "bounding center reports translated center after nonuniform scale and rotation");
  auto size=lowerElementField(*secondMaterial,2007,OperationCode::boundsSizeField,fieldContract,error);
  check(size && std::get<FieldData>(size->data).elements[0].components==std::array<double,4>{6,2,1.5,0},
        "bounding size reflects rotated nonuniform and negative scales");
  if (!positions || !center || !size) return;
  auto box=lowerBoundingBox(*secondMaterial,2008,contract,error);
  check(box && std::get<GeometryData>(box->data).positions.size()==8 &&
        std::get<GeometryData>(box->data).indices.size()==36,"bounding box is renderable indexed Geometry3D");
  if (box) roundtrip(*box,contract,{},"bounding box retains its source chain through transport");
  auto joined=lowerJoinGeometry(*secondMaterial,*secondMaterial,2009,contract,error);
  check(joined.has_value(),"joining one geometry twice deduplicates shared operations");
  if (!joined) return;
  const auto &mesh=std::get<GeometryData>(joined->data);
  check(mesh.positions.size()==48 && mesh.indices.size()==72 &&
        std::set<StableId>(mesh.vertexIds.begin(),mesh.vertexIds.end()).size()==48 &&
        std::all_of(mesh.vertexIds.begin(),mesh.vertexIds.end(),[](auto id) {
          return id<=std::numeric_limits<std::uint32_t>::max();
        }),"join offsets triangle indices and generates distinct stable native-compatible IDs");
  auto repeated=lowerJoinGeometry(*secondMaterial,*secondMaterial,2009,contract,error);
  check(repeated && equal(mesh,std::get<GeometryData>(repeated->data)),"join IDs and topology are deterministic");
  roundtrip(*joined,contract,{},"joined geometry survives transport");
  auto offset=lowerVectorDisplacement(*joined,*center,2010,0.25f,contract,error);
  check(offset.has_value(),"single bounding vector broadcasts into a geometry field consumer");
  auto independentlyShaded=lowerVectorDisplacement(*cube,*center,2013,0.25f,contract,error);
  check(independentlyShaded && retainedMaterialIdentity(*independentlyShaded)==1,
        "a query branch cannot replace the rendered geometry's material");
  auto small=contract; small.maxVertices=47;
  check(!lowerJoinGeometry(*secondMaterial,*secondMaterial,2009,small,error),"join rejects insufficient capacity");
  auto other=lowerAssignMaterial(*cube,2011,99,contract,error);
  check(other && !lowerJoinGeometry(*other,*secondMaterial,2012,contract,error),
        "join rejects differing materials until multi-material draw support exists");
  auto forged=*secondMaterial;
  forged.operations[1].transform.scale.x=10;
  check(!admitValue(forged,contract,error),"forged retained scale fails admission");
  auto invalidTransform=transform; invalidTransform.rotation={0,0,0,0};
  check(!lowerTransformGeometry(*cube,2012,invalidTransform,contract,error),"zero quaternion fails before publication");
  auto reactiveSource=*cube;
  reactiveSource.operations.push_back({2020,OperationCode::evaluateField,2020,2020,{}});
  ++reactiveSource.dispatchCount;
  spectrum::Binding binding;
  binding.fieldStableId=2020; binding.sourceGeometryStableId=cube->stableId;
  binding.consumerGeometryStableId=2021; binding.direction={1,0,0};
  binding.coordinates.resize(std::get<GeometryData>(cube->data).positions.size(),0);
  reactiveSource.spectrumFields={binding};
  auto consumer=lowerTransformGeometry(reactiveSource,2021,{},contract,error);
  auto reactive=consumer ? lowerTransformGeometry(*consumer,2022,transform,contract,error) : std::nullopt;
  auto shaded=reactive ? lowerAssignMaterial(*reactive,2023,99,contract,error) : std::nullopt;
  check(shaded && shaded->spectrumFields[0].direction==std::array<float,3>{0,2,0},
        "TRS rotates and scales spectrum offsets and material assignment retains the binding");
  if (shaded) {
    roundtrip(*shaded,contract,{},"reactive TRS and material binding survive transport");
    check(!lowerBoundingBox(*shaded,2024,contract,error) &&
          !lowerJoinGeometry(*shaded,*cube,2025,contract,error),
          "topology-changing operations reject unmapped reactive bindings");
  }
  auto plain=admitValue(*cube,contract,error);
  auto legacy=encodeRuntimeValue(*plain); legacy[4]=2;
  legacy.resize(legacy.size()-2*sizeof(std::uint32_t));
  check(decodeRuntimeValue(legacy,contract,{}, {},error).has_value(),"v2 operation records remain readable");
  TestBackend backend;
  videohelper::geometry::GeometryCorePlanRuntime runtime(backend);
  auto admitted=admitValue(*joined,contract,error);
  const auto bytes=lowerRuntimePlan(contract,*admitted);
  auto preview=runtime.admitPreview(bytes,{}, {},error);
  auto exported=runtime.admitExport(bytes,{}, {},error);
  check(preview && exported && encodeRuntimeValue(preview->plan->value())==encodeRuntimeValue(exported->plan->value()),
        "preview and export admit the same joined TRS geometry");
}

void audioDeformationAndEffector() {
  std::string error;
  auto contract=contractFor(CarrierKind::geometry3D);
  contract.maxVertices=100; contract.maxIndices=600;
  auto source=lowerGrid(3010,3010,1,9,9,0.25f,contract,error);
  check(source.has_value(),"Audio Deformer has a retained grid source");
  if (!source) return;
  spectrum::Binding binding;
  binding.firstBand=4; binding.lastBand=4;
  binding.deformer.threshold=0.5f; binding.deformer.hysteresis=0.2f;
  auto value=lowerAudioDeformer(*source,3011,binding,contract,error);
  check(value.has_value(),"Audio Deformer lowering retains geometry");
  if (!value) return;
  roundtrip(*value,contract,{},"Audio Deformer controls roundtrip through native transport");
  auto forged=*value; forged.spectrumFields[0].consumerGeometryStableId=3010;
  check(!admitValue(forged,contract,error),"Audio Deformer rejects a forged consumer");
  forged=*value; forged.spectrumFields[0].deformer.envelopePower=std::numeric_limits<float>::infinity();
  check(!admitValue(forged,contract,error),"Audio Effector rejects non-finite controls");
  forged=*value; forged.spectrumFields[0].deformer.hysteresis=0.7f;
  check(!admitValue(forged,contract,error),"Audio Effector rejects hysteresis above its threshold");
  check(!lowerTransformGeometry(*value,3012,{},contract,error),"deformation local coordinates cannot be changed after evaluation");
  const spectrum::FeaturesAt gateStream=[](double t) {
    spectrum::Bands bands {};
    bands[4]=t<0.5 ? 0.8f : t<0.75 ? 0.4f : t<1.0 ? 0.2f : 0.4f;
    return bands;
  };
  auto activeBinding=value->spectrumFields.front();
  spectrum::Follower state, exportState;
  spectrum::Bands result {}, earlier {}, exported {};
  check(spectrum::evaluate(activeBinding,gateStream(0.6),0.6,gateStream,state,result,error)
      && std::abs(result[0]-0.4f)<1.0e-6f,"hysteresis keeps the gate open within its lower boundary");
  check(spectrum::evaluate(activeBinding,gateStream(1.1),1.1,gateStream,state,result,error)
      && result[0]==0,"a closed hysteresis gate stays closed below its upper boundary");
  activeBinding.deformer.peakHoldSeconds=0.2f;
  activeBinding.deformer.envelopePower=2; activeBinding.deformer.quantizeSteps=4;
  check(spectrum::evaluate(activeBinding,gateStream(0.6),0.6,gateStream,state,result,error)
      && result[0]==0.75f,"peak hold precedes envelope shaping and quantization");
  activeBinding.attackSeconds=0.1f; activeBinding.releaseSeconds=0.25f;
  activeBinding.deformer.delaySeconds=0.1f;
  check(spectrum::evaluate(activeBinding,gateStream(0.7),0.7,gateStream,state,earlier,error)
      && spectrum::evaluate(activeBinding,gateStream(1.1),1.1,gateStream,state,result,error)
      && spectrum::evaluate(activeBinding,gateStream(0.7),0.7,gateStream,state,result,error)
      && spectrum::evaluate(activeBinding,gateStream(0.7),0.7,gateStream,exportState,exported,error)
      && earlier==result && result==exported,"delayed envelope seek and export are independent of evaluation order");
  check(!spectrum::evaluate(activeBinding,gateStream(0.7),0.7,{},state,result,error),
        "live-only analysis diagnoses an unavailable delayed history");
  spectrum::Bands bands {}; bands.fill(0.6f);
  for (int mode=0;mode<=10;++mode) {
    auto controls=value->spectrumFields.front(); controls.deformer.mode=static_cast<spectrum::Deformation>(mode);
    controls.deformer.falloffRadius=3; controls.deformer.phase=0.3f;
    auto mesh=std::get<GeometryData>(source->data), original=mesh;
    check(applyAudioDeformer(mesh,controls,bands,1.1,error) && !equal(mesh,original)
        && mesh.indices==original.indices && mesh.vertexIds==original.vertexIds,
        "every Audio Deformer mode changes positions while preserving topology and identities");
    auto repeated=original;
    check(applyAudioDeformer(repeated,controls,bands,1.1,error) && equal(mesh,repeated),
          "Audio Deformer evaluates each timestamp deterministically");
    controls.deformer.falloffRadius=0.01f;
    mesh=original;
    check(applyAudioDeformer(mesh,controls,bands,1.1,error) && equal(mesh.positions.front(),original.positions.front()),
          "spatial falloff leaves distant vertices unchanged");
  }
  auto chained=*value;
  for (StableId id=3012;id<=3014;++id) {
    auto next=lowerAudioDeformer(chained,id,binding,contract,error);
    check(next.has_value(),"bass, mids and treble can chain bounded deformation stages");
    if (!next) return;
    chained=*next;
  }
  roundtrip(chained,contract,{},"chained Audio Deformers retain all operation identities");
  check(!lowerAudioDeformer(chained,3015,binding,contract,error),"a fifth audio stage exceeds the explicit runtime capacity");
}

void audioEffectorFeatureInputs() {
  std::string error;
  auto contract=contractFor(CarrierKind::geometry3D);
  contract.maxVertices=100; contract.maxIndices=600;
  auto source=lowerGrid(3020,3020,1,9,9,0.25f,contract,error);
  check(source.has_value(),"Audio Effector input test has a retained grid source");
  if (!source) return;
  spectrum::Binding binding;
  binding.firstBand=4; binding.lastBand=4;
  binding.deformer.threshold=0.5f;
  auto value=lowerAudioDeformer(*source,3021,binding,contract,error);
  check(value.has_value(),"Audio Effector input test lowers its deformer stage");
  if (!value) return;
  const auto gate=value->spectrumFields.front();
  check(gate.input==spectrum::EffectorInput::bandRange,"a new spectrum binding defaults to band range input");

  // Band 4 sits below the 0.5 gate. Scalar inputs must open the gate on their own value.
  spectrum::Bands quiet {}; quiet[4]=0.1f;
  spectrum::Bands loud {}; loud.fill(1.0f);
  const spectrum::FeaturesAt noBands;
  spectrum::Scalars features; features.rms=0.8f; features.peak=0.6f; features.onset=0.9f;
  const auto evaluateWith=[&](spectrum::EffectorInput input, const spectrum::Bands& bands,
                              const spectrum::Scalars& scalars, spectrum::Bands& out) {
    auto controls=gate; controls.input=input;
    spectrum::Follower state;
    return spectrum::evaluate(controls,bands,1.0,noBands,state,out,error,spectrum::ScalarInput{&scalars,{}});
  };
  spectrum::Bands out {};
  check(evaluateWith(spectrum::EffectorInput::bandRange,quiet,features,out) && out[0]==0.0f,
        "band range input keeps the gate closed below its threshold");
  check(evaluateWith(spectrum::EffectorInput::rms,quiet,features,out) && std::abs(out[0]-0.8f)<1.0e-6f,
        "RMS input opens the gate from the analyzer feature");
  check(evaluateWith(spectrum::EffectorInput::peak,quiet,features,out) && std::abs(out[0]-0.6f)<1.0e-6f,
        "Peak input passes its feature through the gate");
  check(evaluateWith(spectrum::EffectorInput::onset,quiet,features,out) && std::abs(out[0]-0.9f)<1.0e-6f,
        "Onset input passes its feature through the gate");
  spectrum::Scalars weak; weak.onset=0.3f;
  check(evaluateWith(spectrum::EffectorInput::onset,loud,weak,out) && out[0]==0.0f,
        "a scalar below the threshold stays gated even when every band is loud");
  check(evaluateWith(spectrum::EffectorInput::rms,loud,spectrum::Scalars{},out) && out[0]==0.0f,
        "a scalar input ignores the band values entirely");
  {
    auto controls=gate; controls.input=spectrum::EffectorInput::rms;
    spectrum::Follower refused;
    check(!spectrum::evaluate(controls,quiet,1.0,noBands,refused,out,error)
          && error.find("RMS")!=std::string::npos,
          "a scalar input without an analyzer frame fails closed instead of reading zero");
  }
  {
    // Baked analysis supplies its own scalar timeline. Onset holds for 0.2 seconds after a pulse.
    auto controls=gate; controls.input=spectrum::EffectorInput::onset; controls.deformer.peakHoldSeconds=0.2f;
    const spectrum::FeaturesAt silentBands=[](double) { return spectrum::Bands {}; };
    const spectrum::ScalarsAt pulse=[](double time, spectrum::Scalars& scalars) {
      scalars={};
      if (time>=0.5 && time<0.55) scalars.onset=1.0f;
      return true;
    };
    spectrum::Follower heldState, releasedState;
    check(spectrum::evaluate(controls,quiet,0.6,silentBands,heldState,out,error,spectrum::ScalarInput{nullptr,pulse})
          && out[0]==1.0f,"baked onset input holds its peak through the replay");
    check(spectrum::evaluate(controls,quiet,1.0,silentBands,releasedState,out,error,spectrum::ScalarInput{nullptr,pulse})
          && out[0]==0.0f,"baked onset input releases after the hold expires");
    check(!spectrum::evaluate(controls,quiet,0.6,silentBands,heldState,out,error,spectrum::ScalarInput{})
          && error.find("RMS")!=std::string::npos,
          "baked onset input without a scalar timeline fails closed");
  }

  // Transport: band range keeps the pre-input layout. Scalar choices write version 16.
  const auto bandAdmitted=admitValue(*value,contract,error);
  check(bandAdmitted.has_value(),"band-range Audio Effector binding admits");
  if (!bandAdmitted) return;
  const auto bandBytes=encodeRuntimeValue(*bandAdmitted);
  check(bandBytes[4]<16,"band-range transport keeps a pre-input version");
  const auto bandDecoded=decodeRuntimeValue(bandBytes,contract,{},{},error);
  check(bandDecoded && bandDecoded->descriptor().spectrumFields.front().input==spectrum::EffectorInput::bandRange
        && encodeRuntimeValue(*bandDecoded)==bandBytes,
        "old-version band-range transport decodes to band range and re-encodes byte-exact");
  auto versionFifteen=bandBytes; versionFifteen[4]=15;
  const auto oldFifteen=decodeRuntimeValue(versionFifteen,contract,{},{},error);
  check(oldFifteen && oldFifteen->descriptor().spectrumFields.front().input==spectrum::EffectorInput::bandRange,
        "a version 15 spectrum binding defaults to band range input");

  // The input byte is located by diffing encodings, not by a fixed tail offset.
  // Two scalar choices of the same binding differ only in that one byte.
  std::vector<std::uint8_t> rmsBytes,peakBytes;
  for (const auto choice : {spectrum::EffectorInput::rms,spectrum::EffectorInput::peak,spectrum::EffectorInput::onset}) {
    auto chosen=*value; chosen.spectrumFields.front().input=choice;
    const auto admitted=admitValue(chosen,contract,error);
    check(admitted.has_value(),"each scalar Audio Effector input admits");
    if (!admitted) return;
    const auto bytes=encodeRuntimeValue(*admitted);
    check(bytes[4]==16,"a scalar Audio Effector input writes transport version 16");
    const auto decoded=decodeRuntimeValue(bytes,contract,{},{},error);
    check(decoded && decoded->descriptor().spectrumFields.front().input==choice
          && encodeRuntimeValue(*decoded)==bytes,
          "each scalar Audio Effector input roundtrips through transport exactly");
    if (choice==spectrum::EffectorInput::rms) rmsBytes=bytes;
    if (choice==spectrum::EffectorInput::peak) peakBytes=bytes;
  }
  std::size_t inputOffset=rmsBytes.size(),differing=0;
  if (rmsBytes.size()==peakBytes.size())
    for (std::size_t index=0;index<rmsBytes.size();++index)
      if (rmsBytes[index]!=peakBytes[index]) { ++differing; inputOffset=index; }
  check(differing==1 && inputOffset<rmsBytes.size()
        && rmsBytes[inputOffset]==static_cast<std::uint8_t>(spectrum::EffectorInput::rms)
        && peakBytes[inputOffset]==static_cast<std::uint8_t>(spectrum::EffectorInput::peak),
        "the Audio Effector input is the only byte that changes between scalar choices");
  auto outOfRange=rmsBytes;
  if (inputOffset<outOfRange.size()) outOfRange[inputOffset]=7;
  error.clear();
  check(!decodeRuntimeValue(outOfRange,contract,{},{},error)
        && error.find("Invalid Audio Effector input")!=std::string::npos,
        "transport refuses an out-of-range Audio Effector input with its range error");
  auto unknownInput=*value; unknownInput.spectrumFields.front().input=static_cast<spectrum::EffectorInput>(7);
  error.clear();
  check(!admitValue(unknownInput,contract,error) && error.find("Audio Effector input is unknown")!=std::string::npos,
        "admission refuses an unknown Audio Effector input with its reason");
  auto wrongTarget=*value; wrongTarget.spectrumFields.front().target=spectrum::Target::vertexDisplacement;
  wrongTarget.spectrumFields.front().input=spectrum::EffectorInput::rms;
  error.clear();
  check(!admitValue(wrongTarget,contract,error)
        && error.find("Only Audio Effector stages choose an analyzer scalar input")!=std::string::npos,
        "only Audio Effector stages accept a scalar input, refused with its reason");
}

void scoreFieldsRetainBoundedProvenance() {
  const auto contract=contractFor(CarrierKind::geometry3D);
  std::string error;
  auto mesh=lowerGrid(4011,4011,1,2,2,1,contract,error);
  check(mesh.has_value(),"Score Field generated source lowers");
  if (!mesh) return;
  scorefield::Binding binding;
  binding.fieldStableId=4012; binding.sourceGeometryStableId=4011;
  binding.consumerGeometryStableId=4013; binding.scoreSourceStableId=4014;
  binding.elementIds=std::get<GeometryData>(mesh->data).vertexIds;
  for (const auto& p : std::get<GeometryData>(mesh->data).positions)
    binding.positions.push_back({p.x,p.y,p.z});
  mesh->scoreFields.push_back(binding);
  mesh->operations.push_back({4012,OperationCode::evaluateField,4012,4012,{}});
  mesh->operations.push_back({4013,OperationCode::transformTRS,4011,4013,{}});
  mesh->stableId=4013; mesh->dispatchCount=3;
  for (unsigned mode=0;mode<12;++mode) {
    mesh->scoreFields[0].mode=static_cast<scorefield::Mode>(mode);
    roundtrip(*mesh,contract,{},"all Score Field modes retain exact controls and stable geometry IDs");
  }
  const auto admitted=admitValue(*mesh,contract,error);
  if (!admitted) { check(false,error.c_str()); return; }
  const auto bytes=encodeRuntimeValue(*admitted);
  for (std::size_t cut=bytes.size()-24;cut<bytes.size();++cut)
    check(!decodeRuntimeValue(bytes.data(),cut,contract,{}, {},error),
          "truncated Score Field samples fail before native execution");
  auto invalid=*mesh;
  invalid.scoreFields[0].consumerGeometryStableId=999;
  check(!admitValue(invalid,contract,error),"Score Field rejects a forged consumer");
  invalid=*mesh; invalid.scoreFields[0].scoreSourceStableId=0;
  check(!admitValue(invalid,contract,error),"Score Field rejects absent score-source identity");
  invalid=*mesh; invalid.scoreFields[0].elementIds[0]=999;
  check(!admitValue(invalid,contract,error),"Score Field rejects mismatched stable geometry elements");
  invalid=*mesh; invalid.scoreFields[0].positions.pop_back();
  check(!admitValue(invalid,contract,error),"Score Field rejects mismatched sample cardinality");
  invalid=*mesh; invalid.scoreFields[0].mode=static_cast<scorefield::Mode>(12);
  check(!admitValue(invalid,contract,error),"Score Field rejects unknown measurements");
  invalid=*mesh; invalid.scoreFields[0].primes[0]=6;
  check(!admitValue(invalid,contract,error),"Score Field rejects an absent prime basis column");
  Transform transform; transform.translation.x=1;
  check(!lowerTransformGeometry(*mesh,4015,transform,contract,error),
        "transforms after score sampling reject instead of silently changing field coordinates");
  TestBackend backend;
  videohelper::geometry::GeometryCorePlanRuntime runtime(backend);
  const auto lowered=lowerRuntimePlan(contract,*admitted);
  const auto preview=runtime.admitPreview(lowered,{}, {},error);
  const auto exported=runtime.admitExport(lowered,{}, {},error);
  check(preview && exported && encodeRuntimeValue(preview->plan->value())
      ==encodeRuntimeValue(exported->plan->value()),"preview and export share exact Score Field controls");
}

void distributedPointsAndInstanceSelection() {
  std::string error;
  const auto contract=distributedInstanceContract();
  for (const auto code : {OperationCode::pointsOnFaces,OperationCode::pointsInBoxVolume}) {
    const auto original=distributedInstanceFixture(code,29,false,error);
    const auto repeated=distributedInstanceFixture(code,29,false,error);
    const auto reseeded=distributedInstanceFixture(code,197,false,error);
    const auto edited=distributedInstanceFixture(code,29,true,error);
    check(original && repeated && reseeded && edited,"point distribution and instance edit fixtures lower");
    if (!original || !repeated || !reseeded || !edited) continue;
    const auto admitted=admitValue(*original,contract,error);
    const auto editedAdmission=admitValue(*edited,contract,error);
    check(admitted && editedAdmission,"point distributions and instance edits replay exactly");
    if (!admitted || !editedAdmission) continue;
    const auto bytes=encodeRuntimeValue(*editedAdmission);
    const auto reopened=decodeRuntimeValue(bytes,contract,{}, {},error);
    check(reopened && encodeRuntimeValue(*reopened)==bytes,"instance selection and TRS roundtrip through transport v5");
    const auto &before=std::get<InstancesData>(original->data), &after=std::get<InstancesData>(edited->data);
    check(equal(before,std::get<InstancesData>(repeated->data)),"repeated seeded distribution is deterministic");
    check(!equal(before,std::get<InstancesData>(reseeded->data)),"seed changes point positions");
    check(after.instances.size()==11,"culling removes the selected indices");
    for (std::size_t i=0;i<before.instances.size();++i) {
      const auto &a=before.instances[i];
      const auto &b=std::get<InstancesData>(reseeded->data).instances[i];
      check(a.stableId==i+1 && a.stableId==b.stableId,"sample identities survive seed edits");
      const auto &p=a.transform.translation;
      check(std::abs(p.x)<=1 && std::abs(p.y)<=1 && std::abs(p.z)<=0.5f,"distributed points stay in source bounds");
      if (code==OperationCode::pointsOnFaces)
        check(std::abs(p.x)==1 || std::abs(p.y)==1 || std::abs(p.z)==0.5f,"face samples lie on the cube surface");
    }
    for (const auto &instance:after.instances) {
      const auto index=instance.stableId-1;
      check(!(index>=1 && (index-1)%3==0),"culling preserves surviving stable identities");
      if (index%2==0) {
        check(instance.transform.scale.x==1.6f && instance.transform.scale.y==0.6f,
              "selected instances receive local scale");
        check(instance.transform.rotation.z!=0,"selected instances receive rotation");
      } else check(equal(instance.transform,before.instances[index].transform),"unselected transforms stay unchanged");
    }
    auto forged=*edited;
    std::get<InstancesData>(forged.data).instances.front().transform.translation.x+=0.1f;
    check(!admitValue(forged,contract,error),"forged retained instance transforms are rejected");
    forged=*edited; forged.operations.back().parameters[2]=0;
    check(!admitValue(forged,contract,error),"zero selection stride is rejected");
    forged=*original;
    for (auto &operation:forged.operations) if (operation.code==code) operation.parameters[0]=65;
    check(!admitValue(forged,contract,error),"distribution over the 64-instance native bound is rejected");
    check(!lowerInstanceEdit(*original,7999,OperationCode::cullInstances,{0,63,1,0},{},contract,error)
          && error.find("no renderable")!=std::string::npos,"cull-all reports the current empty-scene limitation");
    auto tiny=contract; tiny.maxInstances=1;
    check(!lowerInstanceEdit(*original,7999,OperationCode::transformInstances,{0,63,1,0},{},tiny,error),
          "instance edits respect the destination capacity");
  }
  GeometryData degenerate;
  degenerate.positions={{0,0,0},{1,0,0},{2,0,0}}; degenerate.indices={0,1,2};
  PointsData points;
  check(!materializeDistributedPoints(degenerate,{1,OperationCode::pointsOnFaces,1,1,{16,0,0,0}},points,error),
        "degenerate faces cannot supply a distribution");
  check(!materializeDistributedPoints(degenerate,{1,OperationCode::pointsInBoxVolume,1,1,{16,0,0,0}},points,error),
        "zero-volume bounds cannot supply a volume distribution");
  GeometryData weighted;
  weighted.positions={{0,0,0},{1,0,0},{0,1,0},{10,0,0},{12,0,0},{10,2,0}};
  weighted.indices={0,1,2,3,4,5};
  check(materializeDistributedPoints(weighted,{1,OperationCode::pointsOnFaces,1,1,{64,29,0,0}},points,error),
        "unequal-area face fixture distributes");
  const auto largeFace=std::count_if(points.points.begin(),points.points.end(),[](const auto &point) {
    return point.position.x>=10;
  });
  check(largeFace>40 && largeFace<62,"face distribution weights area rather than triangle count");
}

// Spectral History on Audio Deformer vertices and Spectrum Instancer instances.
// The replay below mirrors the backend deform path: history rows first, then
// one response per retained vertex, each replay with a fresh follower.
void spectralHistoryDrivesDeformerAndInstancerRows() {
  std::string error;
  PortContract meshContract;
  meshContract.carrier=CarrierKind::geometry3D; meshContract.overflow=OverflowPolicy::reject;
  meshContract.maxVertices=16; meshContract.maxIndices=54;
  const auto grid=lowerGrid(9801,9801,1,4,4,0.5f,meshContract,error);
  check(grid.has_value(),"history deformer source is replayable Geometry3D");
  if (!grid) { std::cerr << error << '\n'; return; }
  const auto unchanged=std::get<GeometryData>(grid->data).positions;
  check(unchanged.size()==16,"history deformer source has the expected vertex count");

  spectrum::Binding binding;
  binding.firstBand=8; binding.lastBand=8; binding.direction={0,0,1};
  binding.history.enabled=true; binding.history.depth=4; binding.history.spacingSeconds=0.25f;
  const spectrum::HistoryAxis byIndex{0,-1,1};
  check(!lowerAudioDeformer(*grid,9803,binding,meshContract,error,nullptr).has_value(),
      "history Audio Deformer refuses lowering without a row axis");
  const auto deformed=lowerAudioDeformer(*grid,9803,binding,meshContract,error,&byIndex);
  check(deformed.has_value(),"history Audio Deformer lowers onto retained vertices");
  if (!deformed) { std::cerr << error << '\n'; return; }
  const auto& lowered=deformed->spectrumFields.back();
  check(lowered.target==spectrum::Target::geometryDeformation && lowered.history.enabled
      && lowered.history.coordinates.size()==16 && lowered.history.coordinates.front()==0
      && lowered.history.coordinates.back()==1,
      "deformer history rows follow the vertex index axis");

  // Vertex i has age i/15, so rows 8 through 12 select the 0.5 s row (0.8).
  const auto replay=[](const ValueDescriptor& value, double seconds, std::vector<Vec3>& positions) {
    spectrum::Follower follower;
    const auto frameDeformer=[&](GeometryData& geometry, const spectrum::Binding& selected, std::string& failure) {
      std::vector<spectrum::Bands> rows;
      if (!spectrum::evaluateHistory(selected,{},seconds,spectralHistoryPulse,false,false,follower,rows,failure)) return false;
      std::vector<double> responses(geometry.positions.size());
      for (std::size_t vertex=0;vertex<responses.size();++vertex)
        responses[vertex]=spectrum::deformerHistoryResponse(selected,rows,vertex);
      return applyAudioDeformerResponses(geometry,selected,responses,seconds,failure);
    };
    ValueDescriptor evaluated;
    std::string failure;
    if (!validateOperationPlan(value,{},nullptr,failure,&evaluated,nullptr,frameDeformer)) return false;
    positions=std::get<GeometryData>(evaluated.data).positions;
    return true;
  };
  const auto samePositions=[](const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
    if (a.size()!=b.size()) return false;
    for (std::size_t i=0;i<a.size();++i)
      if (a[i].x!=b[i].x || a[i].y!=b[i].y || a[i].z!=b[i].z) return false;
    return true;
  };
  std::vector<Vec3> sounding, repeated, quiet;
  check(replay(*deformed,1,sounding) && replay(*deformed,1,repeated) && samePositions(sounding,repeated),
      "history deformer replay is deterministic across fresh evaluations");
  check(replay(*deformed,0.1,quiet) && samePositions(quiet,unchanged),
      "history deformer is silent before its sounding row exists");
  bool exact=sounding.size()==16;
  for (std::size_t v=0;exact && v<sounding.size();++v) {
    const double dx=sounding[v].x-unchanged[v].x, dy=sounding[v].y-unchanged[v].y, dz=sounding[v].z-unchanged[v].z;
    const double moved=std::sqrt(dx*dx+dy*dy+dz*dz);
    const bool selectedRow=v>=8 && v<=12;
    exact=selectedRow ? std::abs(moved-0.8)<1.0e-4 : moved<1.0e-6;
  }
  check(exact,"only vertices whose history age selects the sounding row move by its response");

  const auto admitted=admitValue(*deformed,meshContract,error);
  check(admitted.has_value(),"history Audio Deformer admits its retained vertex rows");
  if (admitted) {
    const auto bytes=encodeRuntimeValue(*admitted);
    const auto reopened=decodeRuntimeValue(bytes,meshContract,{}, {},error);
    std::vector<Vec3> fromTransport;
    check(reopened && encodeRuntimeValue(*reopened)==bytes && replay(reopened->descriptor(),1,fromTransport)
        && samePositions(fromTransport,sounding),
        "deformer history rows roundtrip and replay exactly from transport");
  }
  auto invalid=*deformed;
  invalid.spectrumFields.back().history.coordinates.pop_back();
  check(!admitValue(invalid,meshContract,error),"deformer history rows must match retained vertices");

  PortContract instanceContract=contractFor(CarrierKind::instances3D);
  instanceContract.maxInstances=64;
  spectrum::Binding lineBinding;
  lineBinding.firstBand=8; lineBinding.lastBand=11; lineBinding.direction={0,0.5f,0};
  lineBinding.history.enabled=true; lineBinding.history.depth=4; lineBinding.history.spacingSeconds=0.25f;
  check(!lowerSpectrumInstancer(*grid,9806,lineBinding,{0,2,2,2},instanceContract,error,nullptr,nullptr).has_value(),
      "instancer history refuses lowering without a row axis");
  const auto line=lowerSpectrumInstancer(*grid,9805,lineBinding,{0,2,2,2},instanceContract,error,nullptr,&byIndex);
  check(line.has_value(),"history Spectrum Instancer lowers one row per retained instance");
  if (!line) { std::cerr << error << '\n'; return; }
  const auto& instancer=line->spectrumFields.back();
  check(instancer.target==spectrum::Target::instanceTransform && instancer.history.enabled
      && instancer.history.coordinates.size()==4 && instancer.history.coordinates.front()==0
      && instancer.history.coordinates.back()==1,
      "instancer history rows follow the instance index axis");
  const auto admittedInstances=admitValue(*line,instanceContract,error);
  check(admittedInstances.has_value(),"history Spectrum Instancer admits its instance rows");
  if (admittedInstances) {
    const auto bytes=encodeRuntimeValue(*admittedInstances);
    const auto reopened=decodeRuntimeValue(bytes,instanceContract,{}, {},error);
    check(reopened && encodeRuntimeValue(*reopened)==bytes
        && reopened->descriptor().spectrumFields.back().history.depth==4,
        "instancer history rows roundtrip through transport");
  }
  spectrum::HistoryAxis byX{1,-1,1};
  const auto lineByX=lowerSpectrumInstancer(*grid,9807,lineBinding,{0,2,2,2},instanceContract,error,nullptr,&byX);
  check(lineByX.has_value() && lineByX->spectrumFields.back().history.coordinates.size()==4
      && lineByX->spectrumFields.back().history.coordinates.front()==0
      && lineByX->spectrumFields.back().history.coordinates.back()==1,
      "instancer history can select position X as its row axis");
}

// Audio Deformer history rows hold the deformer envelope. Each row must equal the
// current-frame deformer at its row time, so attack, release, peak hold and delay
// mean the same with and without Spectral History. Rows are read at t = 1 s unless
// noted: row r is at 1 - 0.05 r seconds.
void spectralHistoryDeformerEnvelopeRows() {
  std::string error;
  spectrum::Binding base;
  base.fieldStableId=9901; base.sourceGeometryStableId=9900; base.consumerGeometryStableId=9902;
  base.target=spectrum::Target::geometryDeformation;
  base.firstBand=8; base.lastBand=8;
  base.coordinates={0.0f};
  base.history.enabled=true; base.history.depth=32; base.history.spacingSeconds=0.05f;
  base.history.coordinates={0.0f};
  const auto historyRows=[&](const spectrum::Binding& selected, double seconds, std::vector<spectrum::Bands>& rows) {
    spectrum::Follower follower;
    return spectrum::evaluateHistory(selected,{},seconds,spectralHistoryPulse,false,false,follower,rows,error);
  };
  const auto currentEnvelope=[&](const spectrum::Binding& selected, double seconds, float& envelope) {
    spectrum::Follower follower; spectrum::Bands result{};
    if (!spectrum::evaluate(selected,{},seconds,spectralHistoryPulse,follower,result,error)) return false;
    envelope=result[selected.firstBand];
    return true;
  };
  const auto matchesCurrentFrame=[&](const spectrum::Binding& selected, double seconds) {
    std::vector<spectrum::Bands> rows;
    if (!historyRows(selected,seconds,rows)) return false;
    bool sounding=false;
    for (std::size_t row=0;row<rows.size();++row) {
      const double time=seconds-row*static_cast<double>(selected.history.spacingSeconds);
      if (time<0) continue;
      float expected=0;
      if (!currentEnvelope(selected,time,expected)) return false;
      if (std::abs(rows[row][selected.firstBand]-expected)>1.0e-6f) return false;
      sounding = sounding || expected>0;
    }
    return sounding;
  };

  auto attack=base; attack.attackSeconds=0.2f;
  auto release=base; release.releaseSeconds=0.5f;
  auto hold=base; hold.deformer.peakHoldSeconds=0.5f;
  auto delay=base; delay.deformer.delaySeconds=0.25f;
  auto combined=base;
  combined.attackSeconds=0.05f; combined.releaseSeconds=0.3f;
  combined.deformer.peakHoldSeconds=0.2f; combined.deformer.delaySeconds=0.1f;
  combined.deformer.threshold=0.3f; combined.deformer.hysteresis=0.1f;

  check(matchesCurrentFrame(attack,1.0),"attack rows equal the current-frame envelope at each row time");
  check(matchesCurrentFrame(release,1.0),"release rows equal the current-frame envelope at each row time");
  check(matchesCurrentFrame(hold,1.0),"peak hold rows equal the current-frame envelope at each row time");
  check(matchesCurrentFrame(delay,1.0),"delay rows equal the current-frame envelope at each row time");
  check(matchesCurrentFrame(combined,1.0),"combined controls equal the current-frame envelope at each row time");

  std::vector<spectrum::Bands> rows;
  check(historyRows(attack,1.0,rows) && rows[9][8]>0 && rows[9][8]<0.8f && rows[10][8]<rows[9][8],
      "attack ramps up inside the pulse, so earlier rows are smaller");
  check(historyRows(release,1.0,rows) && rows[7][8]<0.8f && rows[7][8]>rows[5][8] && rows[5][8]>rows[3][8]
      && rows[3][8]>rows[1][8] && rows[1][8]>rows[0][8] && rows[0][8]>0,
      "release decays monotonically after the pulse ends");
  check(historyRows(release,1.0,rows) && rows[14][8]==0.0f,"release stays silent before the pulse starts");
  check(historyRows(hold,1.0,rows) && rows[0][8]==0.8f,"peak hold keeps the pulse level inside the hold time");
  check(historyRows(hold,1.2,rows) && rows[0][8]==0.0f,"peak hold releases once the hold time expires");
  check(historyRows(delay,1.0,rows) && rows[10][8]==0.0f && rows[6][8]==0.8f,
      "delay reads the input at row time minus delay");

  auto shaped=base; shaped.deformer.envelopePower=2.0f; shaped.deformer.quantizeSteps=4.0f;
  std::vector<spectrum::Bands> one(2);
  one[0].fill(0.8f); one[1].fill(0.0f);
  check(std::abs(spectrum::deformerHistoryResponse(shaped,one,0)-0.75f)<1.0e-6f,
      "history shaping applies envelope power, then quantization");

  std::vector<spectrum::Bands> first, away, again;
  check(historyRows(combined,1.0,first) && historyRows(combined,0.35,away) && historyRows(combined,1.0,again)
      && first==again,"seek-stable: rendering t1, t0 and t1 again gives identical rows");

  // Sustained input over [0, 20) with a 2 second release. The 20 second warm-up
  // starts the replay with zero state, so the oldest row can differ from an
  // unbounded replay from time zero. The bound is about exp(-10) times the
  // envelope at the warm-up start, far below 1e-4.
  auto longRelease=base;
  longRelease.releaseSeconds=2.0f;
  longRelease.history.depth=128; longRelease.history.spacingSeconds=0.15f;
  const spectrum::FeaturesAt sustained=[](double time) {
    spectrum::Bands bands{}; bands[8]=time<20.0 ? 0.8f : 0.0f; return bands;
  };
  std::vector<float> truth(3601);
  spectrum::Follower exact;
  for (int frame=0;frame<=3600;++frame) {
    const double time=frame/60.0;
    spectrum::advanceEffector(longRelease,sustained(time),time,1.0/60.0,exact);
    truth[frame]=exact.envelope;
  }
  spectrum::Follower windowed;
  std::vector<spectrum::Bands> longRows;
  const bool bounded=spectrum::evaluateHistory(longRelease,{},60.0,sustained,false,false,windowed,longRows,error)
      && longRows.size()==128;
  float worst=0;
  for (std::size_t row=0;bounded && row<128;++row)
    worst=std::max(worst,std::abs(longRows[row][8]-truth[3600-9*row]));
  check(bounded && worst<1.0e-4f,"a 20 second warm-up keeps a 2 second release within 1e-4 of an unbounded replay");

  auto overlong=base; overlong.releaseSeconds=2.5f;
  auto overhold=base; overhold.deformer.peakHoldSeconds=2.5f;
  std::string refusal;
  check(!spectrum::validate(overlong,1,refusal) && !spectrum::validate(overhold,1,refusal),
      "release and peak hold beyond their 2 second bound are refused");

  // Live attack and release keep the previous spectrum smoothing. The reference
  // runs the same live follower on a vertex-displacement binding, whose rows are the
  // smoothed bands, then applies the previous response formula to those rows.
  const auto previousResponse=[](const spectrum::Binding& selected, const std::vector<spectrum::Bands>& rows,
                                 std::size_t vertex) {
    const auto row=static_cast<std::size_t>(std::round(selected.history.coordinates[vertex]*(rows.size()-1)));
    const auto bands=spectrum::normalizeBands(rows[row],selected);
    float value=0.0f;
    for (auto i=selected.firstBand;i<=selected.lastBand;++i) value+=bands[i];
    value/=static_cast<float>(selected.lastBand-selected.firstBand+1);
    const auto& d=selected.deformer;
    if (value<d.threshold) value=0.0f;
    auto shaped=std::pow(std::clamp(value,0.0f,1.0f),d.envelopePower);
    if (d.quantizeSteps>0) shaped=std::round(shaped*d.quantizeSteps)/d.quantizeSteps;
    return shaped*selected.gain+selected.bias;
  };
  auto liveDeformer=base;
  liveDeformer.history.live=true; liveDeformer.attackSeconds=0.1f; liveDeformer.releaseSeconds=0.25f;
  liveDeformer.coordinates={0.0f,1.0f}; liveDeformer.history.coordinates={0.0f,1.0f};
  liveDeformer.deformer.threshold=0.2f; liveDeformer.gain=1.5f; liveDeformer.bias=0.1f;
  auto liveSpectrum=liveDeformer; liveSpectrum.target=spectrum::Target::vertexDisplacement;
  spectrum::Follower deformerState, spectrumState;
  std::vector<spectrum::Bands> deformerRows, spectrumRows;
  bool liveEqual=true, liveSounded=false;
  for (int frame=0;frame<240 && liveEqual;++frame) {
    const double time=frame/60.0;
    spectrum::Bands current{};
    current[8]=(frame/20)%2==0 ? 0.7f : 0.1f;
    liveEqual=spectrum::evaluateHistory(liveDeformer,current,time,{},true,false,deformerState,deformerRows,error)
        && spectrum::evaluateHistory(liveSpectrum,current,time,{},true,false,spectrumState,spectrumRows,error);
    for (std::size_t vertex=0;liveEqual && vertex<2;++vertex) {
      const float actual=spectrum::deformerHistoryResponse(liveDeformer,deformerRows,vertex);
      liveEqual=std::abs(actual-previousResponse(liveDeformer,spectrumRows,vertex))<1.0e-6f;
      liveSounded = liveSounded || actual>0;
    }
  }
  check(liveEqual && liveSounded,"live attack and release are admitted and keep the previous spectrum smoothing output");

  // Live peak hold, hysteresis and delay were ignored before, so they stay refused.
  auto liveHold=base; liveHold.history.live=true; liveHold.deformer.peakHoldSeconds=0.2f;
  check(!spectrum::validate(liveHold,1,refusal) && refusal.find("peak hold, hysteresis or delay")!=std::string::npos
      && refusal.find("Project Baked Analysis")!=std::string::npos,
      "live history refuses Audio Deformer peak hold with a clear diagnostic");
  auto liveHysteresis=base; liveHysteresis.history.live=true;
  liveHysteresis.deformer.threshold=0.3f; liveHysteresis.deformer.hysteresis=0.1f;
  check(!spectrum::validate(liveHysteresis,1,refusal),"live history refuses Audio Deformer hysteresis");
  auto liveDelay=base; liveDelay.history.live=true; liveDelay.deformer.delaySeconds=0.1f;
  check(!spectrum::validate(liveDelay,1,refusal),"live history refuses Audio Deformer delay");
  check(spectrum::validate(liveDeformer,2,refusal),"live attack and release pass admission");
  auto liveStateless=base; liveStateless.history.live=true;
  check(spectrum::validate(liveStateless,1,refusal),"live history keeps stateless threshold rows");
}

void constructedFieldsAndNamedAttributes() {
  std::string error;
  for (int variant=0;variant<3;++variant) {
    auto value=constructedFieldFixture(variant,error);
    check(value.has_value(),error.c_str());
    if (!value) continue;
    const auto contract=withAttributeContract(constructedFieldMeshContract(),*value);
    roundtrip(*value,contract,{},"constructed fields retain exact immutable preview and export data");
    if (variant>0) {
      auto forged=*value; forged.attributes.front().elements.front().components[0]+=1;
      check(!admitValue(forged,contract,error),"forged named attribute values fail replay");
      forged=*value; forged.operations[3].field.domain=Domain::point;
      check(!admitValue(forged,contract,error),"forged field domain fails replay");
      Transform transform; transform.rotation={0,0.70710677f,0,0.70710677f}; transform.scale={2,1,0.5f};
      auto transformed=lowerTransformGeometry(*value,8200,transform,contract,error);
      check(transformed.has_value(),"authored normals survive nonsingular geometry transforms");
      if (transformed) roundtrip(*transformed,contract,{},"inverse-transpose authored normals replay");
    }
  }
  auto mesh=lowerGrid(8300,8300,1,2,2,1,constructedFieldMeshContract(),error);
  if (!mesh) { check(false,"field test mesh lowers"); return; }
  for (const auto type : {ValueType::floatValue,ValueType::vector,ValueType::color,ValueType::integer}) {
    FieldOperationSettings settings; settings.domain=Domain::vertex; settings.valueType=type;
    settings.values={type==ValueType::integer ? 7.0 : 0.5,0,0,0,1,0,0,0};
    if (type==ValueType::vector || type==ValueType::color) { settings.values[1]=0.25; settings.values[2]=0.75; }
    if (type==ValueType::color) settings.values[3]=1;
    const auto fieldContract=constructedFieldContract(Domain::vertex,type);
    auto field=lowerConstructedField(8301,OperationCode::fieldConstant,settings,nullptr,nullptr,nullptr,fieldContract,error);
    auto stored=field ? lowerNamedAttribute(*mesh,&*field,8302,OperationCode::storeNamedAttribute,"value",type,constructedFieldMeshContract(),error) : std::nullopt;
    check(stored.has_value(),"float vector color and material-index values store on vertex attributes");
    if (!stored) continue;
    auto read=lowerNamedAttribute(*stored,nullptr,8303,OperationCode::readNamedAttribute,"value",type,fieldContract,error);
    check(read && std::get<FieldData>(read->data).elements.size()==4,"named attribute read expands constant values over its domain");
    if (read) roundtrip(*read,fieldContract,{},"named attribute reads replay and serialize");
    settings.values={}; settings.attributeName="value";
    auto sampled=lowerNearestField(*mesh,*stored,8304,OperationCode::fieldSampleNearest,settings,fieldContract,error);
    check(sampled && std::get<FieldData>(sampled->data).elements[0].components==std::get<FieldData>(field->data).elements[0].components,
          "nearest target attribute sampling preserves the exact typed value");
    settings.valueType=type==ValueType::vector ? ValueType::color : ValueType::vector;
    check(!lowerNearestField(*mesh,*stored,8304,OperationCode::fieldSampleNearest,settings,
                             constructedFieldContract(Domain::vertex,settings.valueType),error),"nearest sampling rejects a different attribute type");
  }
  for (const auto domain : {Domain::vertex,Domain::point,Domain::curvePoint,Domain::instance}) {
    const auto contract=constructedFieldContract(domain,ValueType::floatValue);
    FieldOperationSettings settings; settings.domain=domain; settings.valueType=ValueType::floatValue;
    settings.values={0.25,0,0,0,1,0,0,0};
    auto a=lowerConstructedField(8401,OperationCode::fieldConstant,settings,nullptr,nullptr,nullptr,contract,error);
    settings.values[0]=2;
    auto b=lowerConstructedField(8402,OperationCode::fieldConstant,settings,nullptr,nullptr,nullptr,contract,error);
    check(a && b,"typed constants lower in every supported element domain");
    if (!a || !b) continue;
    settings.values={}; settings.mode=2;
    auto multiplied=lowerConstructedField(8403,OperationCode::fieldMath,settings,&*a,&*b,nullptr,contract,error);
    check(multiplied && std::get<FieldData>(multiplied->data).elements[0].components[0]==0.5,"field multiplication is retained in every domain");
    if (multiplied) roundtrip(*multiplied,contract,{},"field math roundtrip retains its typed domain");
    auto wrong=*b; wrong.fieldDomain=domain==Domain::point ? Domain::vertex : Domain::point;
    check(!lowerConstructedField(8403,OperationCode::fieldMath,settings,&*a,&wrong,nullptr,contract,error),"cross-domain fields cannot silently combine");
    auto longer=*b; std::get<FieldData>(longer.data).elements.resize(3); longer.fieldDomainCardinality=3;
    auto two=*a; std::get<FieldData>(two.data).elements.resize(2); two.fieldDomainCardinality=2;
    check(!lowerConstructedField(8403,OperationCode::fieldMath,settings,&two,&longer,nullptr,contract,error),"incompatible nonconstant field cardinality is rejected");
  }
  FieldData a,b,out;
  a.elements={{{0.25,0,0,0}}}; b.elements={{{0.75,0,0,0}}};
  const auto evaluate=[&](OperationCode code, ValueType type, std::uint8_t mode,std::array<double,8> params,
                           const FieldData& input,const FieldData* other=nullptr) {
    OperationRecord op{8501,code,1,8501}; op.field.domain=Domain::vertex; op.field.valueType=type;
    op.field.mode=mode; op.field.values=params;
    return materializeConstructedField(op,&input,other,nullptr,out,16,error);
  };
  check(evaluate(OperationCode::fieldCompare,ValueType::boolean,0,{},a,&b) && out.elements[0].components[0]==1,"comparison creates a selection mask");
  FieldData mask=out;
  check(evaluate(OperationCode::fieldBoolean,ValueType::boolean,3,{},mask) && out.elements[0].components[0]==0,"Boolean Not inverts a mask");
  check(evaluate(OperationCode::fieldMix,ValueType::floatValue,0,{0,0,0,0,0.5,0,0,0},a,&b) && out.elements[0].components[0]==0.5,"field mix uses a bounded factor");
  check(evaluate(OperationCode::fieldSelect,ValueType::floatValue,0,{0,0,0,0,1,0,0,0},a,&b) && out.elements[0].components[0]==0.75,"mask selection chooses the second field");
  check(evaluate(OperationCode::fieldMapRange,ValueType::floatValue,1,{0,1,-1,1,0,0,0,0},a) && out.elements[0].components[0]==-0.5,"range remapping uses both ranges");
  check(evaluate(OperationCode::fieldClamp,ValueType::floatValue,0,{0.4,1,0,0,0,0,0,0},a) && out.elements[0].components[0]==0.4,"clamp respects authored bounds");
  FieldData vector; vector.elements={{{0.25,0.5,0.75,0}}};
  check(evaluate(OperationCode::fieldGradient,ValueType::floatValue,1,{0,0,0,0,1,0,0,0},vector) && out.elements[0].components[0]==0.5,"gradient samples the chosen position axis");
  check(evaluate(OperationCode::fieldFalloff,ValueType::floatValue,1,{0,0,0,1,2,0,0,0},vector) && out.elements[0].components[0]==0.0625,"box falloff applies the exponent");
  check(evaluate(OperationCode::fieldNoise,ValueType::floatValue,0,{2,19,0,0,0,0,0,0},vector),"deterministic trilinear noise evaluates");
  const auto noise=out;
  check(evaluate(OperationCode::fieldNoise,ValueType::floatValue,0,{2,19,0,0,0,0,0,0},vector) && out.elements[0].components==noise.elements[0].components,"identical noise coordinates and seed repeat exactly");
  check(evaluate(OperationCode::fieldNoise,ValueType::floatValue,0,{2,20,0,0,0,0,0,0},vector) && out.elements[0].components!=noise.elements[0].components,"noise seed affects the result");
  FieldData ids; ids.elements={{{17,0,0,0}},{{94,0,0,0}}};
  check(evaluate(OperationCode::fieldRandom,ValueType::floatValue,0,{27,0,1,0,0,0,0,0},ids),"stable identity random field evaluates");
  const auto random=out; std::swap(ids.elements[0],ids.elements[1]);
  check(evaluate(OperationCode::fieldRandom,ValueType::floatValue,0,{27,0,1,0,0,0,0,0},ids)
      && out.elements[0].components==random.elements[1].components && out.elements[1].components==random.elements[0].components,
      "random field follows stable identities through reordering");
  check(evaluate(OperationCode::fieldComposeVector,ValueType::vector,1,{0,0,2,0,0,0,0,0},a) && out.elements[0].components[2]==0.5,"scalar influence constructs a displacement vector");
  check(evaluate(OperationCode::fieldVectorMeasure,ValueType::floatValue,4,{},vector) && out.elements[0].components[0]==0.5,"vector components become scalar fields");
  FieldOperationSettings nearest; nearest.domain=Domain::vertex; nearest.valueType=ValueType::integer;
  GeometryData target; target.positions={{-1,0,0},{1,0,0}}; target.vertexIds={20,10};
  GeometryData source; source.positions={{0,0,0}}; source.vertexIds={1};
  OperationRecord query{8601,OperationCode::fieldNearestIndex,1,8601,{},2,{},nearest};
  check(materializeNearestField(source,CarrierKind::geometry3D,target,{},query,out,1,error)
      && out.elements[0].components[0]==1,"equidistant nearest vertices tie-break by stable identity");
}

void sampleIndexSamplesExactAttributes() {
  std::string error;
  AttributeData attribute;
  attribute.descriptor={namedAttributeIdentity("value",Domain::vertex,ValueType::floatValue),"value",
      ValueType::floatValue,Domain::vertex,Interpolation::nearest};
  attribute.elements={{{0.5,0,0,0}},{{1.5,0,0,0}},{{2.5,0,0,0}}};
  FieldOperationSettings settings; settings.domain=Domain::vertex; settings.valueType=ValueType::floatValue;
  settings.attributeName="value";
  OperationRecord op{8701,OperationCode::fieldSampleIndex,1,8701,{},2,{},settings};
  FieldData index; index.elements={{{2,0,0,0}},{{0,0,0,0}},{{1,0,0,0}}};
  FieldData sampled;
  check(materializeSampleIndexField(&attribute,index,op,sampled,3,error)
      && sampled.elements.size()==3 && sampled.elements[0].components[0]==2.5
      && sampled.elements[1].components[0]==0.5 && sampled.elements[2].components[0]==1.5,
      "sample index reads exact source elements in index order");
  FieldData outside; outside.elements={{{3,0,0,0}}};
  check(!materializeSampleIndexField(&attribute,outside,op,sampled,3,error),"sample index past the attribute end is refused");
  outside.elements={{{-1,0,0,0}}};
  check(!materializeSampleIndexField(&attribute,outside,op,sampled,3,error),"negative sample index is refused");
  outside.elements={{{0.5,0,0,0}}};
  check(!materializeSampleIndexField(&attribute,outside,op,sampled,3,error),"fractional sample index is refused");
  check(!materializeSampleIndexField(&attribute,index,op,sampled,2,error),"sample index respects its element capacity");
  check(!materializeSampleIndexField(nullptr,index,op,sampled,3,error),"sample index without a source attribute is refused");
  auto wrongDomain=op; wrongDomain.field.domain=Domain::point;
  check(!materializeSampleIndexField(&attribute,index,wrongDomain,sampled,3,error),"sample index refuses a different element domain");

  auto mesh=lowerGrid(8700,8700,1,2,2,1,constructedFieldMeshContract(),error);
  if (!mesh) { check(false,"sample index mesh lowers"); return; }
  const auto vertexCount=std::get<GeometryData>(mesh->data).positions.size();
  const auto vertexContract=constructedFieldContract(Domain::vertex,ValueType::integer);
  FieldOperationSettings nearest; nearest.domain=Domain::vertex; nearest.valueType=ValueType::integer;
  auto identity=lowerNearestField(*mesh,*mesh,8701,OperationCode::fieldNearestIndex,nearest,vertexContract,error);
  check(identity.has_value(),"nearest vertex indices of a retained mesh form an exact identity field");
  if (!identity) return;
  auto stored=lowerNamedAttribute(*mesh,&*identity,8702,OperationCode::storeNamedAttribute,"value",
      ValueType::integer,constructedFieldMeshContract(),error);
  check(stored.has_value(),"identity indices store as a retained integer attribute");
  if (!stored) return;
  FieldOperationSettings constant; constant.domain=Domain::vertex; constant.valueType=ValueType::integer;
  constant.values={2,0,0,0,static_cast<double>(vertexCount),0,0,0};
  auto index2=lowerConstructedField(8703,OperationCode::fieldConstant,constant,nullptr,nullptr,nullptr,vertexContract,error);
  check(index2.has_value(),"constant integer index field lowers");
  if (!index2) return;
  FieldOperationSettings sample; sample.domain=Domain::vertex; sample.valueType=ValueType::integer;
  sample.attributeName="value";
  const auto sampleContract=constructedFieldContract(Domain::vertex,ValueType::integer);
  auto constantSample=lowerSampleIndexField(*stored,*index2,8704,sample,sampleContract,error);
  check(constantSample && std::get<FieldData>(constantSample->data).elements.size()==vertexCount
      && std::all_of(std::get<FieldData>(constantSample->data).elements.begin(),
             std::get<FieldData>(constantSample->data).elements.end(),
             [](const FieldElement& e) { return e.components[0]==2; }),
      "constant sample index returns the same source element for every vertex");
  auto identitySample=lowerSampleIndexField(*stored,*identity,8705,sample,sampleContract,error);
  check(identitySample && [&] {
      const auto& elements=std::get<FieldData>(identitySample->data).elements;
      for (std::size_t i=0;i<elements.size();++i) if (elements[i].components[0]!=static_cast<double>(i)) return false;
      return elements.size()==vertexCount; }(),
      "identity sample index reproduces each exact source element");
  if (identitySample) roundtrip(*identitySample,sampleContract,{},"sample index fields replay and serialize");

  constant.values[0]=static_cast<double>(vertexCount);
  auto outOfRange=lowerConstructedField(8706,OperationCode::fieldConstant,constant,nullptr,nullptr,nullptr,vertexContract,error);
  check(outOfRange.has_value() && !lowerSampleIndexField(*stored,*outOfRange,8707,sample,sampleContract,error),
        "sample index past the source attribute is refused at lowering");
  FieldOperationSettings missing=sample; missing.attributeName="offset";
  check(!lowerSampleIndexField(*stored,*identity,8708,missing,sampleContract,error),
        "sample index refuses a missing source attribute");
  FieldOperationSettings wrongType=sample; wrongType.valueType=ValueType::floatValue;
  check(!lowerSampleIndexField(*stored,*identity,8709,wrongType,constructedFieldContract(Domain::vertex,ValueType::floatValue),error),
        "sample index refuses a different value type");

  std::string admissionError;
  auto admittedSample=admitValue(*identitySample,sampleContract,ResourceLimits{},{},admissionError);
  if (!admittedSample) { check(false,"identity sample index admits for transport tampering"); return; }
  const auto bytes=encodeRuntimeValue(*admittedSample);
  for (std::size_t cut=0;cut<bytes.size();++cut) {
    std::vector<std::uint8_t> truncated(bytes.begin(),bytes.begin()+static_cast<std::ptrdiff_t>(cut));
    check(!decodeRuntimeValue(truncated,sampleContract,ResourceLimits{},{},admissionError),
          "every truncated sample index transport is rejected");
  }
  auto malformed=bytes; malformed[0]^=0xff;
  check(!decodeRuntimeValue(malformed,sampleContract,ResourceLimits{},{},admissionError),"sample index bad magic is rejected");
  malformed=bytes; malformed[4]=255;
  check(!decodeRuntimeValue(malformed,sampleContract,ResourceLimits{},{},admissionError),"sample index unknown version is rejected");
  malformed=bytes; malformed.push_back(0);
  check(!decodeRuntimeValue(malformed,sampleContract,ResourceLimits{},{},admissionError),"sample index trailing data is rejected");
  auto forged=*identitySample; forged.operations.back().field.domain=Domain::point;
  check(!admitValue(forged,sampleContract,admissionError),"forged sample index domain fails replay");
  forged=*identitySample; forged.operations.back().field.attributeName.clear();
  check(!admitValue(forged,sampleContract,admissionError),"sample index without an attribute name fails replay");
}

void sampleIndexTransportVersionAndReplayRefusals() {
  std::string error;
  auto mesh=lowerGrid(8720,8720,1,2,2,1,constructedFieldMeshContract(),error);
  if (!mesh) { check(false,"sample index forgery mesh lowers"); return; }
  const auto vertexCount=std::get<GeometryData>(mesh->data).positions.size();
  const auto vertexContract=constructedFieldContract(Domain::vertex,ValueType::integer);
  FieldOperationSettings nearest; nearest.domain=Domain::vertex; nearest.valueType=ValueType::integer;
  auto identity=lowerNearestField(*mesh,*mesh,8721,OperationCode::fieldNearestIndex,nearest,vertexContract,error);
  if (!identity) { check(false,"sample index forgery identity indices lower"); return; }
  auto stored=lowerNamedAttribute(*mesh,&*identity,8722,OperationCode::storeNamedAttribute,"value",
      ValueType::integer,constructedFieldMeshContract(),error);
  if (!stored) { check(false,"sample index forgery source attribute stores"); return; }
  FieldOperationSettings constant; constant.domain=Domain::vertex; constant.valueType=ValueType::integer;
  constant.values={2,0,0,0,static_cast<double>(vertexCount),0,0,0};
  auto index=lowerConstructedField(8723,OperationCode::fieldConstant,constant,nullptr,nullptr,nullptr,vertexContract,error);
  if (!index) { check(false,"sample index forgery constant index lowers"); return; }
  FieldOperationSettings sample; sample.domain=Domain::vertex; sample.valueType=ValueType::integer;
  sample.attributeName="value";
  const auto sampleContract=constructedFieldContract(Domain::vertex,ValueType::integer);
  auto sampled=lowerSampleIndexField(*stored,*index,8724,sample,sampleContract,error);
  if (!sampled) { check(false,"sample index forgery sample value lowers"); return; }

  std::string admissionError;
  auto admitted=admitValue(*sampled,sampleContract,ResourceLimits{},{},admissionError);
  if (!admitted) { check(false,"sample index forgery base value admits"); return; }
  const auto bytes=encodeRuntimeValue(*admitted);
  auto decoded=decodeRuntimeValue(bytes,sampleContract,ResourceLimits{},{},admissionError);
  check(decoded.has_value() && admissionError.empty(),
        "sample index is written at its introducing transport version and decodes");

  // Version 14 predates operation 137, so the same bytes must not decode.
  auto belowGate=bytes; belowGate[4]=14; belowGate[5]=0;
  admissionError.clear();
  const bool refusedBelowGate=!decodeRuntimeValue(belowGate,sampleContract,ResourceLimits{},{},admissionError);
  check(refusedBelowGate && admissionError=="unknown Geometry Core operation",
        "sample index operation below transport version 15 is refused");

  // The index field's domain is vertex; a replay that moves it to point must not match the source attribute.
  auto domainForged=*sampled;
  for (auto& op : domainForged.operations) if (op.code==OperationCode::fieldConstant) op.field.domain=Domain::point;
  admissionError.clear();
  const bool refusedDomain=!admitValue(domainForged,sampleContract,admissionError);
  check(refusedDomain && admissionError=="Sample index requires a retained source attribute and an Integer index field",
        "sample index replay refuses an index field whose domain differs from its source attribute");

  // A forged index value one past the last source element must be refused, not read.
  auto indexForged=*sampled;
  for (auto& op : indexForged.operations) if (op.code==OperationCode::fieldConstant) op.field.values[0]=static_cast<double>(vertexCount);
  admissionError.clear();
  const bool refusedIndex=!admitValue(indexForged,sampleContract,admissionError);
  check(refusedIndex && admissionError=="Sample index is outside the source attribute",
        "sample index replay refuses an index past the source element count");
}

void legacyContractsMigrate() {
  std::string error;
  for (const auto carrier : {CarrierKind::geometry3D, CarrierKind::points3D,
                             CarrierKind::curves3D, CarrierKind::instances3D,
                             CarrierKind::field}) {
    auto legacy = contractFor(carrier, carrier == CarrierKind::field
        ? ValueType::floatValue : ValueType::unspecified);
    legacy.schemaVersion = 1;
    legacy.maxVertices = 8;
    legacy.maxIndices = 12;
    legacy.maxPoints = 8;
    legacy.maxCurvePoints = 8;
    legacy.maxSplines = 4;
    legacy.maxInstances = 8;
    legacy.maxFieldElements = carrier == CarrierKind::field ? 3 : 8;
    check(migratePortContractV1ToV2(legacy, error),
          "schema-v1 all-capacity contracts migrate to schema v2");
    check(legacy.schemaVersion == kSchemaVersion
              && validatePortContract(legacy, error),
          "migrated contracts use canonical carrier-specific capacities");
  }
}
} // namespace

void instanceAppearanceRetainsAttributes() {
  using namespace videowire::geometry;
  std::string error;
  for (unsigned variant=0;variant<5;++variant) {
    const auto value=instanceAppearanceFixture(variant,error);
    if (!value) std::cerr << error << '\n';
    check(value.has_value(),"instance appearance fixture lowers");
    if (!value) continue;
    const auto contract=withAttributeContract(appearanceInstanceContract(),*value);
    roundtrip(*value,contract,{},"instance appearance persists with exact operation replay");
    if (variant==0) continue;
    const auto color=findNamedAttribute(value->attributes,"color");
    check(color && color->elements[0].components[1]==1 && color->elements[2].components[0]==1,
          "selection assigns green to two instances while retaining the existing red values");
    auto forged=*value; forged.attributes[0].elements[0].components[1]=0.25;
    check(!admitValue(forged,contract,error),"appearance bytes cannot override retained field operations");
    forged=*value; forged.attributes[0].elements.pop_back();
    check(!admitValue(forged,contract,error),"appearance requires exact instance cardinality");
    Transform transform; transform.translation={0.2f,0,0};
    auto moved=lowerInstanceEdit(*value,9150,OperationCode::transformInstances,{0,3,1,0},transform,contract,error);
    auto culled=moved ? lowerInstanceEdit(*moved,9151,OperationCode::cullInstances,{0,3,2,0},{},contract,error) : std::nullopt;
    check(culled.has_value(),"appearance survives instance transforms and culling");
    if (culled) {
      check(std::get<InstancesData>(culled->data).instances.size()==2,"culling retains two instances");
      const auto* kept=findNamedAttribute(culled->attributes,"color");
      check(kept && kept->elements.size()==2 && kept->elements[0].components==color->elements[1].components
          && kept->elements[1].components==color->elements[3].components,"culled attributes stay attached to stable instance identities");
      roundtrip(*culled,withAttributeContract(appearanceInstanceContract(),*culled),{},"culled appearance replays");
    }
    const auto admitted=admitValue(*value,contract,error);
    auto scene=std::make_unique<HarmonicMIDI::grid::Visual3DScene>();
    HarmonicMIDI::grid::SceneObjectRecord object; object.id={50};
    HarmonicMIDI::grid::SceneMaterialRecord material; material.id={10}; material.baseColor={1,1,1};
    check(admitted && appendGeometryToScene(*admitted,object,material,*scene,error),
          "composed scenes carry appearance as bounded material variants");
    check(scene->objectCount==5 && scene->materialCount==3
          && scene->objects[1].material==scene->objects[2].material
          && scene->objects[1].material!=scene->objects[3].material,
          "composed scene deduplicates equal appearance without expanding the mesh");
  }
  auto source=instanceAppearanceFixture(0,error);
  if (!source) return;
  FieldOperationSettings settings; settings.domain=Domain::instance; settings.valueType=ValueType::floatValue;
  settings.values={1.01,0,0,0,1,0,0,0};
  auto opacity=lowerConstructedField(9170,OperationCode::fieldConstant,settings,nullptr,nullptr,nullptr,
      constructedFieldContract(Domain::instance,ValueType::floatValue),error);
  check(opacity && !lowerInstanceAppearance(*source,*opacity,nullptr,9171,"opacity",appearanceInstanceContract(),error),
        "opacity outside zero to one is rejected");
  settings.values={0.5,0,0,0,3,0,0,0};
  opacity=lowerConstructedField(9172,OperationCode::fieldConstant,settings,nullptr,nullptr,nullptr,
      constructedFieldContract(Domain::instance,ValueType::floatValue),error);
  check(opacity && !lowerInstanceAppearance(*source,*opacity,nullptr,9173,"opacity",appearanceInstanceContract(),error),
        "nonbroadcast mismatched appearance cardinality is rejected");
  auto mesh=lowerGrid(9180,9180,1,2,2,1,constructedFieldMeshContract(),error);
  spectrum::Binding binding; binding.firstBand=8; binding.lastBand=11; binding.scaleResponse={0,1,0};
  auto reactive=mesh ? lowerSpectrumInstancer(*mesh,9181,binding,{0,2,2,2},appearanceInstanceContract(),error) : std::nullopt;
  settings.values={0.7,0,0,0,1,0,0,0};
  opacity=lowerConstructedField(9182,OperationCode::fieldConstant,settings,nullptr,nullptr,nullptr,
      constructedFieldContract(Domain::instance,ValueType::floatValue),error);
  auto appeared=reactive && opacity ? lowerInstanceAppearance(*reactive,*opacity,nullptr,9183,"opacity",appearanceInstanceContract(),error) : std::nullopt;
  check(appeared && appeared->spectrumFields.size()==1,"appearance preserves the spectrum binding and canonical bin identities");
  if (appeared) roundtrip(*appeared,withAttributeContract(appearanceInstanceContract(),*appeared),{},"reactive instance appearance replays");
}

void triangleSurfaceFields() {
  std::string error;
  GeometryData source; source.positions={{0.25f,0.25f,2}}; source.vertexIds={1};
  GeometryData target; target.positions={{0,0,0},{1,0,0},{0,1,0}};
  target.vertexIds={2,3,4}; target.indices={0,1,2,0,1,2};
  FieldOperationSettings settings; settings.domain=Domain::vertex; settings.valueType=ValueType::floatValue;
  OperationRecord query{9300,OperationCode::fieldSurfaceDistance,1,9300,{},2,{},settings};
  FieldData output;
  check(materializeNearestField(source,CarrierKind::geometry3D,target,{},query,output,1,error)
        && output.elements[0].components[0]==2,"surface proximity reaches a triangle interior rather than its vertices");
  query.code=OperationCode::fieldSurfaceOffset; query.field.valueType=ValueType::vector;
  check(materializeNearestField(source,CarrierKind::geometry3D,target,{},query,output,1,error)
        && output.elements[0].components==std::array<double,4>{0,0,-2,0},"surface offset projects onto the face");
  query.field.values={0,0,-4,10,0,0,0,0};
  for (const auto code : {OperationCode::fieldRaycastDistance,OperationCode::fieldRaycastPosition,
                         OperationCode::fieldRaycastHit,OperationCode::fieldRaycastFace}) {
    query.code=code; query.field.valueType=code==OperationCode::fieldRaycastPosition ? ValueType::vector
        : code==OperationCode::fieldRaycastHit ? ValueType::boolean
        : code==OperationCode::fieldRaycastFace ? ValueType::integer : ValueType::floatValue;
    check(materializeNearestField(source,CarrierKind::geometry3D,target,{},query,output,1,error),"typed raycast succeeds");
    if (output.elements.empty()) continue;
    const auto expected=code==OperationCode::fieldRaycastPosition ? std::array<double,4>{0.25,0.25,0,0}
        : std::array<double,4>{code==OperationCode::fieldRaycastDistance ? 2.0 : code==OperationCode::fieldRaycastHit ? 1.0 : 0.0,0,0,0};
    check(output.elements[0].components==expected,"ray direction is normalized and duplicate faces choose the first index");
  }
  query.field.values[2]=1;
  check(materializeNearestField(source,CarrierKind::geometry3D,target,{},query,output,1,error)
        && output.elements[0].components[0]==-1,"raycast misses carry the explicit minus-one face index");
  query.field.values[2]=0;
  check(!materializeNearestField(source,CarrierKind::geometry3D,target,{},query,output,1,error),"zero ray direction is rejected");
  query.field.values[2]=-1; target.indices[0]=99;
  check(!materializeNearestField(source,CarrierKind::geometry3D,target,{},query,output,1,error),"untrusted triangle references fail closed");
  auto mesh=lowerGrid(9310,9310,1,2,2,1,constructedFieldMeshContract(),error);
  Transform transform; transform.translation={0,0,2};
  auto raised=mesh ? lowerTransformGeometry(*mesh,9311,transform,constructedFieldMeshContract(),error) : std::nullopt;
  settings.valueType=ValueType::vector;
  auto field=mesh && raised ? lowerNearestField(*raised,*mesh,9312,OperationCode::fieldSurfaceOffset,settings,
      constructedFieldContract(Domain::vertex,ValueType::vector),error) : std::nullopt;
  check(field.has_value(),"surface field lowers with retained source and target operations");
  if (field) roundtrip(*field,constructedFieldContract(Domain::vertex,ValueType::vector),{},"surface fields replay identically after transport");
}

void retainedMaterialTables() {
  std::string error;
  for (const bool instances : {false,true}) {
    auto value=materialTableFixture(instances,false,error);
    check(value.has_value(),error.c_str()); if (!value) continue;
    const auto contract=withAttributeContract(instances ? appearanceInstanceContract() : constructedFieldMeshContract(),*value);
    roundtrip(*value,contract,{},"material slots and index fields preserve exact preview/export replay");
    auto admitted=admitValue(*value,contract,error);
    auto scene=std::make_unique<HarmonicMIDI::grid::Visual3DScene>();
    HarmonicMIDI::grid::SceneObjectRecord object; object.id={100};
    HarmonicMIDI::grid::SceneMaterialRecord fallback; fallback.id={100};
    check(admitted && appendGeometryToScene(*admitted,object,fallback,*scene,error),error.c_str());
    check(scene->materialCount==2,"equal PBR table entries deduplicate independently of their slot and graph identities");
    check(scene->vertexCount==4 && scene->indexCount==6,"material batches keep one shared mesh allocation");
    check(scene->objectCount==(instances ? 5 : 3),"face batches preserve a parent identity and instances preserve their object count");
    auto forged=*value;
    for (auto& op : forged.operations) if (op.code==OperationCode::materialTableSlot) { op.field.mode=32; break; }
    check(!admitValue(forged,contract,error),"slot overflow fails immutable replay");
    // A field may be authored before its table, but the final native scene must
    // reject a missing slot instead of substituting the previous material.
    FieldOperationSettings settings; settings.domain=instances ? Domain::instance : Domain::face;
    settings.valueType=ValueType::integer; settings.values={31,0,0,0,1,0,0,0};
    auto index=lowerConstructedField(9490,OperationCode::fieldConstant,settings,nullptr,nullptr,nullptr,
        constructedFieldContract(settings.domain,ValueType::integer),error);
    auto invalid=index ? lowerMaterialIndices(*value,*index,9491,contract,error) : std::nullopt;
    auto invalidAdmitted=invalid ? admitValue(*invalid,withAttributeContract(contract,*invalid),error) : std::nullopt;
    scene=std::make_unique<HarmonicMIDI::grid::Visual3DScene>();
    check(invalidAdmitted && !appendGeometryToScene(*invalidAdmitted,object,fallback,*scene,error)
        && error.find("authored slot")!=std::string::npos,"missing slot diagnoses the invalid material index");
  }
}

void timelineFieldsReplayExactFrames() {
  std::string error;
  RuntimeFieldEvaluation clock;
  clock.timelineSeconds=-1;
  FieldOperationSettings settings; settings.domain=Domain::curvePoint; settings.valueType=ValueType::floatValue;
  settings.mode=1; settings.values={4,0,1,0,1,0,0,0};
  OperationRecord time{9699,OperationCode::fieldTimeline,9698,9699,{},0,{},settings};
  FieldData coordinates; coordinates.elements={FieldElement{{0,0,0,0}},FieldElement{{0.5,0,0,0}}};
  FieldData phases;
  check(materializeConstructedField(time,&coordinates,nullptr,nullptr,phases,2,error,&clock)
      && phases.elements[0].components[0]==0.75 && phases.elements[1].components[0]==0.25,
      "negative timeline time wraps phase per coordinate without prior playback");
  time.field.mode=0;
  check(materializeConstructedField(time,&coordinates,nullptr,nullptr,phases,2,error,&clock)
      && phases.elements[0].components[0]==-1,"seconds mode exposes the same signed evaluation time");
  auto grid=timelineInstanceFixture(error);
  check(grid.has_value(),error.c_str());
  if (grid) {
    roundtrip(*grid,timelineContract(CarrierKind::instances3D),{},"grid loop retains per-instance timeline fields");
    const auto admitted=admitValue(*grid,timelineContract(CarrierKind::instances3D),error);
    check(admitted.has_value(),error.c_str());
    if (admitted) {
      auto versionEleven=encodeRuntimeValue(*admitted); versionEleven[4]=11;
      check(decodeRuntimeValue(versionEleven,timelineContract(CarrierKind::instances3D),{}, {},error).has_value(),
          "v11 timeline fields remain readable without a selected-spectrum footer");
      auto legacy=encodeRuntimeValue(*admitted); legacy[4]=10;
      check(!decodeRuntimeValue(legacy,timelineContract(CarrierKind::instances3D),{}, {},error),
          "timeline operation codes require v11 instead of reinterpreting v10");
    }
    MaterializedInstancePlan a,b; clock.timelineSeconds=0;
    check(validateOperationPlan(*grid,{},&a,error,nullptr,&clock),error.c_str()); clock.timelineSeconds=1;
    check(validateOperationPlan(*grid,{},&b,error,nullptr,&clock),error.c_str());
    check(a.instances.instances.size()==4 && b.instances.instances.size()==4
        && !equal(a.instances,b.instances),"instance fields update transforms without expanding shared geometry");
  }
  auto value=timelineCurveFixture(error);
  check(value.has_value(),error.c_str()); if (!value) return;
  roundtrip(*value,timelineContract(CarrierKind::geometry3D),{},"timeline and score fields retain curve dependencies in transport");
  RuntimeFieldEvaluation evaluation;
  auto frame=scoreFieldFrame();
  evaluation.scoreAt=[&](const auto& op,const auto& positions,auto& output,auto& diagnostic) {
    return videohelper::scorefield::evaluateSampleField(op,positions,frame,output,diagnostic);
  };
  const auto replay=[&](double time,MaterializedInstancePlan& out) {
    evaluation.timelineSeconds=time;
    return validateOperationPlan(*value,{},&out,error,nullptr,&evaluation);
  };
  MaterializedInstancePlan zero,quarter,loop,seek,exported,silent;
  check(replay(0,zero),error.c_str()); check(replay(1,quarter),error.c_str()); check(replay(4,loop),error.c_str());
  check(replay(0,seek),error.c_str()); frame=scoreFieldFrame(true); check(replay(1,exported),error.c_str());
  if (zero.curves.count(9715) && zero.curves.count(9708)) {
    const auto& trimmed=zero.curves.at(9715);
    const auto& radii=zero.curves.at(9708).points;
    check(trimmed.points.size()==16 && !trimmed.splines.front().cyclic
        && std::abs(trimmed.points.front().position.x+1.2f)<0.000001f
        && std::abs(trimmed.points.back().position.x-1.2f)<0.000001f,
        "resampling and trim retain 16 ordered slots over the requested arc interval");
    check(radii.front().radius==0.5f && radii.back().radius==0.5f && radii[1].radius>1,
        "endpoint masks narrow the ends while interior radii follow canonical notes");
  } else check(false,"complete curve runtime retains trim, radius and endpoint stages");
  if (zero.geometries.count(9711) && quarter.geometries.count(9711) && loop.geometries.count(9711)
      && seek.geometries.count(9711) && exported.geometries.count(9711)) {
    check(!equal(zero.geometries.at(9711),quarter.geometries.at(9711)),"time changes the retained tube vertices");
    check(equal(zero.geometries.at(9711),loop.geometries.at(9711)),"authored period closes exactly");
    check(equal(zero.geometries.at(9711),seek.geometries.at(9711)),"backward seek has no accumulated state");
    check(equal(quarter.geometries.at(9711),exported.geometries.at(9711)),"reordered canonical notes give the same export geometry");
    check(zero.geometries.at(9711).indices==quarter.geometries.at(9711).indices
        && zero.geometries.at(9711).vertexIds==quarter.geometries.at(9711).vertexIds,"runtime fields preserve topology and element identities");
  }
  frame=scoreFieldFrame(false,true); check(replay(0,silent),error.c_str());
  if (silent.curves.count(9708))
    check(silent.curves.at(9708).points[1].radius==1
        && silent.curves.at(9708).points.front().radius==0.5f,
        "empty canonical score keeps the neutral interior radius and authored endpoint mask");
  frame.reset(); check(!replay(0,silent),"missing canonical frame fails instead of freezing a prior score value");
  auto forged=*value; std::get<GeometryData>(forged.data).positions[0].x+=1;
  check(!admitValue(forged,timelineContract(CarrierKind::geometry3D),error),"dynamic plans still reject forged admission geometry");
  forged=*value;
  for (auto& op : forged.operations) if (op.code==OperationCode::fieldTimeline) op.field.values[0]=0;
  check(!admitValue(forged,timelineContract(CarrierKind::geometry3D),error),"zero loop duration fails admission");
  frame=scoreFieldFrame(); evaluation.timelineSeconds=std::numeric_limits<double>::quiet_NaN();
  check(!validateOperationPlan(*value,{},&silent,error,nullptr,&evaluation),"nonfinite runtime time fails before native drawing");
}

void orderedAudioGeometryReplay() {
  std::string error;
  const auto value=orderedAudioGeometryFixture(error);
  check(value.has_value(),error.c_str()); if (!value) return;
  roundtrip(*value,withAttributeContract(appearanceInstanceContract(),*value),{},
      "ordered audio geometry retains its bindings through the point and instance transport");
  MaterializedInstancePlan silent,active,repeated;
  spectrum::Bands bands{};
  const auto deform=[&](GeometryData& mesh,const spectrum::Binding& binding,std::string& failure) {
    return applyAudioDeformer(mesh,binding,bands,0.5,failure);
  };
  check(validateOperationPlan(*value,{},&silent,error,nullptr,nullptr,deform),error.c_str());
  bands[0]=1;
  check(validateOperationPlan(*value,{},&active,error,nullptr,nullptr,deform),error.c_str());
  check(active.instances.instances.size()==12,"surface point count is fixed during reactive replay");
  if (silent.instances.instances.size()!=12 || active.instances.instances.size()!=12) return;
  for (std::size_t i=0;i<12;++i) {
    check(active.instances.instances[i].stableId==silent.instances.instances[i].stableId,
        "audio replay preserves distributed point and instance identities");
    check(std::abs(active.instances.instances[i].transform.translation.z-silent.instances.instances[i].transform.translation.z-0.6f)<0.00001f,
        "point distribution samples the displaced surface before instancing");
  }
  check(instanceAppearance(active.terminalAttributes,0).color!=instanceAppearance(silent.terminalAttributes,0).color,
      "appearance evaluates the displaced instance height");
  ValueDescriptor replayed;
  check(validateOperationPlan(*value,{},nullptr,error,&replayed,nullptr,deform)
      && equal(std::get<InstancesData>(replayed.data),active.instances)
      && equalAttributes(replayed.attributes,active.terminalAttributes),
      "animated-source replay output remains compatible with ordered audio evaluation");
  bands[0]=0;
  check(validateOperationPlan(*value,{},&repeated,error,nullptr,nullptr,deform)
      && equal(repeated.instances,silent.instances) && equalAttributes(repeated.terminalAttributes,silent.terminalAttributes),
      "seeking to silence resets both geometry and appearance without accumulated state");
  auto forged=*value; forged.spectrumFields.front().coordinates.pop_back();
  check(!admitValue(forged,withAttributeContract(appearanceInstanceContract(),forged),error),
      "deformer cardinality is checked against its source mesh rather than terminal instances");
  const auto curve=timelineCurveFixture(error);
  spectrum::Binding binding; binding.firstBand=binding.lastBand=0; binding.gain=0.2f;
  binding.deformer.mode=spectrum::Deformation::vector;
  const auto combined=curve ? lowerAudioDeformer(*curve,9722,binding,timelineContract(CarrierKind::geometry3D),error)
                            : std::nullopt;
  check(combined.has_value(),error.c_str()); if (!combined) return;
  RuntimeFieldEvaluation clock; clock.timelineSeconds=1;
  const auto frame=scoreFieldFrame();
  clock.scoreAt=[&](const auto& operation,const auto& positions,auto& output,auto& diagnostic) {
    return videohelper::scorefield::evaluateSampleField(operation,positions,frame,output,diagnostic);
  };
  MaterializedInstancePlan timed,audioTimed;
  bands[0]=1;
  check(validateOperationPlan(*curve,{},&timed,error,nullptr,&clock)
      && validateOperationPlan(*combined,{},&audioTimed,error,nullptr,&clock,deform),error.c_str());
  if (timed.geometries.count(curve->stableId) && audioTimed.geometries.count(combined->stableId)) {
    auto expected=timed.geometries.at(curve->stableId);
    check(applyAudioDeformer(expected,combined->spectrumFields.back(),bands,0.5,error)
        && equal(expected,audioTimed.geometries.at(combined->stableId)),
        "one ordered replay retains both timeline/score deformation and audio deformation");
  }
}

void selectedSourceSpectrumParity() {
  spectrum::Binding branch;
  branch.fieldStableId=9715; branch.target=spectrum::Target::geometryDeformation;
  branch.source=spectrum::Source::track; branch.sourceTrackId=71;
  std::vector<spectrum::Binding> bindings{branch};
  std::string error;
  check(spectrum::appendOrderedBindings(bindings,{branch},error) && bindings.size()==1,
        "ordered branches share an exact selected-source binding");
  branch.sourceTrackId=72;
  check(!spectrum::appendOrderedBindings(bindings,{branch},error),
        "ordered branches reject a shared deformer ID with different selected tracks");
  branch.sourceTrackId=71; branch.source=spectrum::Source::group;
  check(!spectrum::appendOrderedBindings(bindings,{branch},error),
        "ordered branches reject a shared deformer ID with different source kinds");
  auto sources=std::make_shared<videohelper::AnalysisSources>();
  for (int index=0;index<2;++index) {
    videohelper::AnalysisSourceStream source;
    source.trackId=71+index; source.group=index==1; source.sampleRate=48000;
    std::vector<float> samples(48000);
    for (size_t i=0;i<samples.size();++i)
      samples[i]=0.5f*std::sin(2.0*arbitblockb::kPi*(index==0 ? 110.0 : 3000.0)*i/48000.0);
    arbitblockb::BlockBAnalyzer analyzer(48000);
    source.frames=analyzer.analyzeOffline(samples.data(),static_cast<int>(samples.size()));
    sources->push_back(std::move(source));
  }
  const auto sample=videohelper::sourceSpectrumReader(sources);
  spectrum::Bands track{},group{},reopened{};
  check(sample(spectrum::Source::track,71,0.75,track)
        && sample(spectrum::Source::group,72,0.75,group) && track!=group,
        "selected sources retain independent canonical 64-band spectra");
  const auto rangeReader=videohelper::sourceSpectrumReader(sources);
  check(rangeReader(spectrum::Source::track,71,0.75,reopened) && reopened==track,
        "range export and scrub preview sample the same project-zero analysis hop");
  check(!sample(spectrum::Source::group,71,0.75,reopened)
        && !sample(spectrum::Source::track,999,0.75,reopened),
        "missing or wrong-kind selected sources diagnose instead of reading another track");
  check(sample(spectrum::Source::track,71,3.0,reopened) && reopened==spectrum::Bands{},
        "ended analysis is silence rather than a latched final spectrum");
}

// Live and export share the analyzer frame index, and every scalar reader
// resolves the same frame as the band reader it pairs with. Each frame stores
// rms in bands[0], so a scalar that comes from a different frame fails here.
void audioScalarParity() {
  using namespace videowire::geometry;
  const auto frameSet=[](int count,float offset) {
    std::vector<arbitblockb::FeatureFrame> frames(static_cast<std::size_t>(count));
    for (int k=0;k<count;++k) {
      auto& frame=frames[static_cast<std::size_t>(k)];
      frame.rms=offset+static_cast<float>(k+1)/1000.0f;
      frame.bands[0]=frame.rms;
      frame.peak=offset+static_cast<float>(k+1)/500.0f;
      frame.onset=static_cast<float>(k%3)/3.0f;
      frame.sampleIndex=static_cast<std::int64_t>(k+1)*arbitblockb::kHop;
    }
    return frames;
  };
  const double sampleRate=48000.0;
  const double times[]={0.0,0.02,0.5,1.0,5.0};
  check(videohelper::analysisFrameIndex(100,sampleRate,-1.0)==0
        && videohelper::analysisFrameIndex(100,sampleRate,0.0)==0
        && videohelper::analysisFrameIndex(100,sampleRate,5.0)==99,
        "analysis frame index clamps before the first window and after the last");

  const auto frames=frameSet(100,0.0f);
  const auto projectBands=videohelper::analysisBandsReader(frames,sampleRate);
  const auto projectScalars=videohelper::analysisScalarsReader(frames,sampleRate);
  bool projectParity=true;
  for (const double t : times) {
    spectrum::Scalars scalars{};
    const auto bands=projectBands(t);
    const auto& expected=frames[videohelper::analysisFrameIndex(frames.size(),sampleRate,t)];
    projectParity=projectParity && projectScalars(t,scalars)
        && bands==expected.bands && scalars.rms==expected.rms
        && scalars.peak==expected.peak && scalars.onset==expected.onset
        && scalars.rms==bands[0];
  }
  check(projectParity, "project scalar readers return rms, peak and onset from the frame that supplies the bands");

  auto sources=std::make_shared<videohelper::AnalysisSources>();
  for (int index=0;index<2;++index) {
    videohelper::AnalysisSourceStream stream;
    stream.trackId=71+index; stream.group=index==1; stream.sampleRate=sampleRate;
    stream.frames=frameSet(100,index==0 ? 0.0f : 0.5f);
    sources->push_back(std::move(stream));
  }
  const auto sourceBands=videohelper::sourceSpectrumReader(sources);
  const auto sourceScalars=videohelper::sourceScalarsReader(sources);
  bool sourceParity=true;
  for (const double t : times) {
    spectrum::Bands trackBands{},groupBands{};
    spectrum::Scalars track{},group{};
    sourceParity=sourceParity
        && sourceBands(spectrum::Source::track,71,t,trackBands) && sourceScalars(spectrum::Source::track,71,t,track)
        && track.rms==trackBands[0]
        && sourceBands(spectrum::Source::group,72,t,groupBands) && sourceScalars(spectrum::Source::group,72,t,group)
        && group.rms==groupBands[0];
  }
  check(sourceParity, "track and group scalar readers match their band readers frame for frame");
  spectrum::Scalars track{},group{},ignored{};
  check(sourceScalars(spectrum::Source::track,71,0.5,track) && sourceScalars(spectrum::Source::group,72,0.5,group)
        && group.rms!=track.rms,
        "track and group scalars come from their own streams");
  check(!sourceScalars(spectrum::Source::group,71,0.5,ignored)
        && !sourceScalars(spectrum::Source::track,999,0.5,ignored)
        && !sourceScalars(spectrum::Source::master,71,0.5,ignored),
        "missing, wrong-kind and master source scalars fail instead of reading another stream");
  check(sourceScalars(spectrum::Source::track,71,5.0,ignored) && ignored.rms==0.0f
        && ignored.peak==0.0f && ignored.onset==0.0f,
        "ended source analysis is silence for scalars as for bands");
}

void instanceMaterialFieldsRetainRuntimeAndSelection() {
  using namespace videowire::geometry;
  std::string error;
  for (const auto* property : {"color","emission","opacity","metallic","roughness","materialIndex"}) {
    auto value=instanceMaterialFieldsFixture(property,true,false,error);
    check(value.has_value(),error.c_str()); if (!value) continue;
    const auto contract=withAttributeContract(appearanceInstanceContract(),*value);
    roundtrip(*value,contract,{},"instance material fields retain transport and operation replay");
    RuntimeFieldEvaluation clock; MaterializedInstancePlan first,later,sought;
    check(validateOperationPlan(*value,{},&first,error,nullptr,&clock),error.c_str());
    clock.timelineSeconds=3;
    check(validateOperationPlan(*value,{},&later,error,nullptr,&clock),error.c_str());
    clock.timelineSeconds=0;
    check(validateOperationPlan(*value,{},&sought,error,nullptr,&clock),error.c_str());
    check(!equalAttributes(first.terminalAttributes,later.terminalAttributes)
        && equalAttributes(first.terminalAttributes,sought.terminalAttributes),
        "instance material attributes change with time and reproduce a backward seek");
    if (isInstanceSurfaceScalar(property)) {
      const auto selected=instanceAppearance(later.terminalAttributes,0);
      const auto unselected=instanceAppearance(later.terminalAttributes,2);
      check((std::string(property)=="metallic" ? selected.metallic : selected.roughness)>0.6f
          && unselected.metallic<0 && unselected.roughness<0,
          "unselected instances inherit their material instead of receiving a property default");
    }
    auto forged=*value;
    forged.attributes.front().elements.front().components[0]=0.37;
    check(!admitValue(forged,contract,error),"runtime replay never trusts forged authored appearance values");
  }
  const auto source=instanceAppearanceFixture(0,error);
  if (!source) { check(false,error.c_str()); return; }
  const auto scalar=constructedFieldContract(Domain::instance,ValueType::floatValue);
  FieldOperationSettings settings; settings.domain=Domain::instance; settings.valueType=ValueType::floatValue;
  settings.values={0.8,0,0,0,1,0,0,0};
  auto field=lowerConstructedField(9540,OperationCode::fieldConstant,settings,nullptr,nullptr,nullptr,scalar,error);
  auto value=field ? lowerInstanceAppearance(*source,*field,nullptr,9541,"metallic",appearanceInstanceContract(),error) : std::nullopt;
  check(value.has_value(),error.c_str()); if (!value) return;
  const auto contract=withAttributeContract(appearanceInstanceContract(),*value);
  roundtrip(*value,contract,{},"static metallic appearance roundtrips with its selection attribute");
  const auto admitted=admitValue(*value,contract,error);
  auto scene=std::make_unique<HarmonicMIDI::grid::Visual3DScene>();
  HarmonicMIDI::grid::SceneObjectRecord object; object.id={50};
  HarmonicMIDI::grid::SceneMaterialRecord material; material.id={10}; material.metallic=0.25f;
  check(admitted && appendGeometryToScene(*admitted,object,material,*scene,error),error.c_str());
  const auto* assigned=HarmonicMIDI::grid::visual3d_detail::findById(scene->materials,scene->materialCount,scene->objects[1].material);
  check(assigned && assigned->metallic==0.8f && scene->vertexCount==4,
        "composed scenes carry metallic material variants without duplicating vertices");
  settings.values[0]=1.01;
  field=lowerConstructedField(9542,OperationCode::fieldConstant,settings,nullptr,nullptr,nullptr,scalar,error);
  for (const auto* property : {"metallic","roughness"})
    check(field && !lowerInstanceAppearance(*source,*field,nullptr,9543,property,appearanceInstanceContract(),error),
          "material properties outside zero to one fail admission");
  auto forged=*value;
  for (auto& attribute : forged.attributes)
    if (attribute.descriptor.name=="metallicSelected") attribute.elements.pop_back();
  check(!admitValue(forged,contract,error),"material selections require the exact instance cardinality");
}

void reactiveSurfaceTransportAndCache() {
  using namespace videohelper::geometry;
  std::string error;
  const auto contract = contractFor(CarrierKind::geometry3D);
  const auto value = admitValue(geometryValue(), contract, error);
  check(value.has_value(), "reactive Surface fixture geometry admits");
  if (!value) return;
  const auto material = reactiveSurfaceRequest();
  const auto wire = geometrysurfacematerial::encode(material);
  const auto timedWire = geometrysurfacematerial::encode(reactiveSurfaceRequest(true));
  check(!wire.empty() && !timedWire.empty() && wire != timedWire, "Surface fixtures retain exact canonical programs");
  const auto plain = lowerRuntimePlan(contract,*value);
  const auto bytes = lowerRuntimePlan(contract,*value,wire);
  const auto decoded = decodeLoweredRuntimePlan(bytes,error);
  check(plain[4] == 1 && bytes[4] == 2 && decoded && decoded->surfaceMaterial == wire
      && decoded->runtimeValue == encodeRuntimeValue(*value), "version two appends Surface IR without changing legacy geometry runtime bytes");
  auto truncated = bytes; truncated.pop_back();
  check(!decodeLoweredRuntimePlan(truncated,error), "truncated Surface payload fails closed");
  auto trailing = bytes; trailing.push_back(0);
  check(!decodeLoweredRuntimePlan(trailing,error), "Surface payload rejects trailing bytes");
  check(!geometrysurfacematerial::decode(wire + "\n",error), "noncanonical Surface whitespace cannot alias a cache identity");
  auto stale = material; stale.binding.surfaceMaterialDigest = "stale";
  check(!geometrysurfacematerial::valid(stale,error), "Surface IR mutations require the exact binding digest");
  auto texture = material; texture.program.textureSlotCount = 1;
  check(!geometrysurfacematerial::valid(texture,error), "reactive Surface cannot acquire textures without a Frame scheduler owner");
  const auto frameWire = geometrysurfacematerial::encode(reactiveFrameRequest());
  const auto frame = geometrysurfacematerial::decode(frameWire,error);
  check(frame && frame->binding.textures.front().graphFrame->node == 91,
      "reactive Surface retains its canonical Frame endpoint across transport");
  check(frameWire != geometrysurfacematerial::encode(reactiveFrameRequest(92)),
      "Frame endpoint changes participate in Geometry Surface cache identity");
  auto foreign = reactiveFrameRequest(); foreign.binding.textures.front().videoResource[0] = 1;
  check(!geometrysurfacematerial::valid(foreign,error), "graph Frame cannot impersonate a video resource identity");
  geometrysurfacematerial::ProgramSet framePrograms{{11,reactiveSurfaceRequest(),{}},
      {12,reactiveFrameRequest(),{}},{13,reactiveFrameRequest(),{}}};
  const auto frameSetWire = geometrysurfacematerial::encodeSet(framePrograms);
  const auto frameSet = geometrysurfacematerial::decodeSet(frameSetWire,error);
  check(frameSet && frameSet->size() == 3 && geometrysurfacematerial::graphFrameEndpoint(*frameSet)->node == 91,
      "distinct material-table programs share an exact Frame endpoint without changing untextured programs");
  framePrograms.back().material = reactiveFrameRequest(92);
  check(geometrysurfacematerial::valid(framePrograms,error)
      && geometrysurfacematerial::graphFrameEndpoints(framePrograms).size() == 2,
      "independent material-table Frame endpoints persist without aliasing another program's image");
  TestBackend backend;
  GeometryCorePlanRuntime runtime(backend);
  const PlanOwnerIdentity owner {1,1,1,9601,1,PlanUse::preview};
  const auto admitted = runtime.admitPreview(owner,bytes,{}, {},error);
  const auto repeated = runtime.admitPreview(owner,bytes,{}, {},error);
  check(admitted && repeated && admitted->plan == repeated->plan && admitted->surfaceMaterial == wire,
      "same owner and material bytes reuse the exact immutable geometry admission");
  check(!runtime.admitPreview(owner,lowerRuntimePlan(contract,*value,timedWire),{}, {},error)
      && error.find("different immutable plan") != std::string::npos,
      "material-only mutations cannot reuse a live geometry owner");
  auto wrongRevision = owner; wrongRevision.planGeneration = 2;
  check(!runtime.admitPreview(wrongRevision,bytes,{}, {},error)
      && error.find("revision") != std::string::npos, "Surface revision must match the plan owner");
  auto exportOwner = owner; exportOwner.use = PlanUse::exportRender;
  check(runtime.admitExport(exportOwner,bytes,{}, {},error).has_value(),
      "preview and export admit identical material bytes under separate owners");
}

void surfaceTableAndFieldTransport() {
  using namespace videohelper::geometry;
  std::string error;
  const auto field=reactiveSurfaceField(8,error);
  check(field.has_value(),"Surface Field fixture admits canonical timeline sampling");
  if (!field) return;
  geometrysurfacematerial::ProgramSet programs{{9521,tableSurfaceRequest(17,false),{*field}},
      {9522,tableSurfaceRequest(29,true),{}}};
  const auto wire=geometrysurfacematerial::encodeSet(programs);
  const auto decoded=geometrysurfacematerial::decodeSet(wire,error);
  check(decoded && decoded->size()==2 && geometrysurfacematerial::encodeSet(*decoded)==wire
      && geometrysurfacematerial::timeDependent(*decoded),"Surface table/Field wire round-trips exact ordered programs and timeline dependence");
  auto invalid=programs; invalid[1].sourceMaterial=invalid[0].sourceMaterial;
  check(geometrysurfacematerial::encodeSet(invalid).empty(),"duplicate Surface table source IDs fail closed");
  invalid=programs; std::reverse(invalid.begin(),invalid.end());
  check(geometrysurfacematerial::encodeSet(invalid).empty(),"noncanonical Surface program order fails closed");
  invalid=programs; invalid[0].fields.resize(surfacematerialfield::kMaximumBindings+1,*field);
  check(geometrysurfacematerial::encodeSet(invalid).empty(),"Surface Field count is bounded before serialization");
  check(!geometrysurfacematerial::decodeSet(wire+" ",error),"trailing Surface table bytes fail closed");
  const auto value=instanceMaterialFieldsFixture("materialIndex",true,false,error);
  check(value.has_value(),"Surface table fixture retains exact native material indices");
  if (!value) return;
  const auto contract=withAttributeContract(appearanceInstanceContract(),*value);
  const auto admitted=admitValue(*value,contract,error);
  if (!admitted) { check(false,"Surface table geometry admits"); return; }
  TestBackend backend;
  GeometryCorePlanRuntime runtime(backend);
  const PlanOwnerIdentity owner{1,1,1,9700,1,PlanUse::preview};
  const auto bytes=lowerRuntimePlan(contract,*admitted,wire);
  const auto first=runtime.admitPreview(owner,bytes,{}, {},error);
  const auto again=runtime.admitPreview(owner,bytes,{}, {},error);
  check(first && again && first->plan==again->plan,"Surface table programs reuse only their exact immutable geometry owner");
  invalid=programs; invalid[0].fields[0].sample.gain=-0.5;
  check(!runtime.admitPreview(owner,lowerRuntimePlan(contract,*admitted,geometrysurfacematerial::encodeSet(invalid)),{}, {},error),
      "a Field-only change cannot reuse an active owner/cache identity");
  invalid=programs; invalid[1].sourceMaterial=9523;
  GeometryCorePlanRuntime foreign(backend);
  check(!foreign.admitPreview(owner,lowerRuntimePlan(contract,*admitted,geometrysurfacematerial::encodeSet(invalid)),{}, {},error),
      "Surface programs must reference an exact authored table material identity");

  MaterializedInstancePlan retained;
  RuntimeFieldEvaluation evaluation; evaluation.timelineSeconds=1.5;
  ValueDescriptor replayed;
  check(validateOperationPlan(*value,{},&retained,error,&replayed,&evaluation),"Surface table replays the mixed frame");
  auto scene=std::make_shared<HarmonicMIDI::grid::Visual3DScene>();
  if (!retained.instances.instances.empty()) {
    const auto sourceId=retained.instances.instances.front().sourceStableId;
    const auto sourceMesh=retained.geometries.find(sourceId);
    const auto attrs=retained.geometryAttributes.find(sourceId);
    if (sourceMesh==retained.geometries.end()) { check(false,"table source mesh exists"); return; }
    scene->vertexCount=sourceMesh->second.positions.size(); scene->indexCount=sourceMesh->second.indices.size();
    std::copy(sourceMesh->second.indices.begin(),sourceMesh->second.indices.end(),scene->indices.begin());
    scene->materialCount=1; scene->materials[0].id={1};
    scene->objectCount=retained.instances.instances.size();
    for (std::size_t index=0;index<scene->objectCount;++index) {
      auto& object=scene->objects[index]; object.id={static_cast<std::uint32_t>(retained.instances.instances[index].stableId)};
      object.material={1}; object.vertexCount=scene->vertexCount; object.indexCount=scene->indexCount;
    }
    std::map<StableId,HarmonicMIDI::grid::SceneMaterialId> ids{{9521,{17}},{9522,{29}}};
    std::vector<MaterialTableDrawBinding> receipts;
    check(sourceMesh!=retained.geometries.end() && applyGeometryMaterialTable(replayed,retained,
        attrs==retained.geometryAttributes.end() ? nullptr : &attrs->second,*scene,0,scene->objectCount,error,&ids,&receipts),
        "material-table batches accept canonical Surface IDs");
    check(receipts.size()==4,"each native table draw has one source and instance receipt");
    for (std::size_t index=0;index<receipts.size();++index) {
      check(receipts[index].instanceIndex==index && scene->objects[index].id.value==receipts[index].object
          && scene->objects[index].material==ids.at(receipts[index].sourceMaterial),
          "Surface table receipt selects exact object, instance index and authored MaterialId");
    }
  } else check(false,"Surface table scene fixture contains instances");

  const auto request=reactiveSurfaceRequest();
  const auto surface=surfacematerial::admit(request.program,error);
  const auto base=surface ? surfacematerial::evaluateReference(*surface,{},error) : std::nullopt;
  check(base.has_value(),"Surface property reference fixture admits");
  if (!base) return;
  for (std::uint32_t target=0;target<std::size(surfacematerialfield::kTargets);++target) {
    const auto binding=reactiveSurfaceField(target,error);
    auto firstBlock=*base,laterBlock=*base,reopenedBlock=*base;
    std::array<float,3> end{};
    evaluation.timelineSeconds=0;
    check(binding && surfacematerialfield::evaluate(*binding,evaluation,0,firstBlock,end,error),"Surface property evaluates first sample");
    evaluation.timelineSeconds=3;
    check(binding && surfacematerialfield::evaluate(*binding,evaluation,0,laterBlock,end,error),"Surface property evaluates later sample");
    const auto property=[](const auto& block,std::uint32_t index) {
      return index<4 ? block.baseColorMetallic[index] : index==4 ? block.emissionRoughness[3]
          : index<8 ? block.emissionRoughness[index-5] : index==8 ? block.normalOpacity[3] : block.transmissionIorClearcoat[index-9];
    };
    for (std::uint32_t index=0;index<std::size(surfacematerialfield::kTargets);++index)
      check(index==target ? property(firstBlock,index)!=property(laterBlock,index)
          : property(firstBlock,index)==property(laterBlock,index),"Surface Field modifies only the requested property");
    const geometrysurfacematerial::ProgramSet entry{{0,request,{*binding}}};
    const auto reopened=geometrysurfacematerial::decodeSet(geometrysurfacematerial::encodeSet(entry),error);
    check(reopened && surfacematerialfield::evaluate(reopened->front().fields[0],evaluation,0,reopenedBlock,end,error)
        && property(reopenedBlock,target)==property(laterBlock,target),"Surface Field replay is unchanged by transport/reopen");
  }
  auto indexed=*field; indexed.objectIndex=true; indexed.sample.reduction=materialfield::Reduction::index;
  auto block=*base; std::array<float,3> end{};
  check(surfacematerialfield::evaluate(indexed,evaluation,3,block,end,error)
      && !surfacematerialfield::evaluate(indexed,evaluation,4,block,end,error),
      "per-object Field samples reject unavailable indices instead of wrapping or broadcasting");
}

void surfaceGeometryDomainsAndRetainedTransport() {
  std::string error;
  const auto surface=surfacematerial::admit(reactiveSurfaceRequest().program,error);
  const auto base=surface ? surfacematerial::evaluateReference(*surface,{},error) : std::nullopt;
  if (!base) { check(false,"Surface domain baseline admits"); return; }
  for (const auto domain : {Domain::vertex,Domain::face,Domain::point,Domain::curvePoint,Domain::spline,Domain::instance}) {
    const auto binding=surfaceDomainFieldFixture(domain,error);
    check(binding.has_value(),error.c_str()); if (!binding) continue;
    const auto field=materialfield::admit(binding->sample,error,true);
    check(field && field->descriptor().fieldDomain==domain,
        "Surface Material Fields preserve the canonical Geometry domain through admission");
    check(!materialfield::admit(binding->sample,error),
        "broader Surface sampling does not widen the physical diffraction adapter implicitly");
    auto parameters=*base;
    std::array<float,3> end{};
    RuntimeFieldEvaluation evaluation;
    const auto expected=domain==Domain::face || domain==Domain::spline ? 0.125f : 0.375f;
    check(surfacematerialfield::evaluate(*binding,evaluation,0,parameters,end,error)
        && std::abs(parameters.baseColorMetallic[0]-base->baseColorMetallic[0]-expected)<0.000001f,
        "geometry-backed field means reach the requested Surface property deterministically");
    const geometrysurfacematerial::ProgramSet programs{{0,reactiveSurfaceRequest(),{*binding}}};
    const auto wire=geometrysurfacematerial::encodeSet(programs);
    const auto reopened=geometrysurfacematerial::decodeSet(wire,error);
    check(reopened && geometrysurfacematerial::encodeSet(*reopened)==wire,
        "every Material Field domain survives immutable program transport byte-for-byte");
    if (domain==Domain::instance) {
      auto indexed=*binding; indexed.objectIndex=true; indexed.sample.reduction=materialfield::Reduction::index;
      auto first=*base,last=*base;
      check(surfacematerialfield::evaluate(indexed,evaluation,0,first,end,error)
          && surfacematerialfield::evaluate(indexed,evaluation,3,last,end,error)
          && last.baseColorMetallic[0]-first.baseColorMetallic[0]==0.75f
          && !surfacematerialfield::evaluate(indexed,evaluation,4,last,end,error),
          "exact instance indices receive distinct samples and never wrap missing data");
    }
  }
  const auto imported=videohelper::gltf::adaptGlbMeshToGeometryCore(importedGeometryFixture(),1,9905,9904,1,error);
  const auto scalar=constructedFieldContract(Domain::vertex,ValueType::floatValue);
  const auto importedIndex=imported ? lowerElementField(imported->descriptor(),9906,OperationCode::indexField,
      constructedFieldContract(Domain::vertex,ValueType::integer),error) : std::nullopt;
  FieldOperationSettings settings; settings.domain=Domain::vertex; settings.valueType=ValueType::floatValue;
  const auto importedFloat=importedIndex ? lowerConstructedField(9907,OperationCode::fieldIntegerToFloat,settings,
      &*importedIndex,nullptr,nullptr,scalar,error) : std::nullopt;
  const auto importedAdmitted=importedFloat ? admitValue(*importedFloat,scalar,error) : std::nullopt;
  if (importedAdmitted) {
    materialfield::Binding sample; sample.materialNode=9999;
    sample.fieldPlan=encodeLoweredPlanText(lowerRuntimePlan(scalar,*importedAdmitted));
    double amount=0;
    check(materialfield::sample(sample,{},amount,error,true) && amount==1,
        "Surface Material Field samples exact retained imported mesh provenance");
  } else check(false,error.c_str());
  const auto fixture=retainedSurfaceFixture(error);
  check(fixture.has_value(),error.c_str()); if (!fixture) return;
  const auto wire=visualimportedscenerender::encode(*fixture);
  visualimportedscenerender::Request reopened;
  check(wire.rfind("visual.imported-scene.render.v13\n",0)==0
      && visualimportedscenerender::decode(wire,reopened) && visualimportedscenerender::sameRequest(*fixture,reopened)
      && reopened.surfacePrograms.size()==5,"retained V13 scene transports exact programs, fields and object identities");
  for (const auto& object : reopened.surfacePrograms)
    check(object.program.material.sceneSnapshot==reopened.sceneSnapshot,
        "all retained object programs share the one decoded immutable scene owner");
  auto invalid=*fixture;
  invalid.surfacePrograms[1].program.material.binding.object=invalid.surfacePrograms[0].program.material.binding.object;
  check(visualimportedscenerender::encode(invalid).empty(),"duplicate object Surface identities fail closed");
  invalid=*fixture; invalid.surfacePrograms[0].program.material.sceneSnapshot=std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(*fixture->sceneSnapshot);
  check(visualimportedscenerender::encode(invalid).empty(),"equal scene bytes cannot replace exact Surface snapshot ownership");
  invalid=*fixture; invalid.surfacePrograms.back().program.fields[0].sample.gain=0.7;
  check(visualimportedscenerender::encode(invalid).empty(),"one authored Surface source cannot name inconsistent per-object programs");
  const auto previous=reopened;
  check(!visualimportedscenerender::decode(wire+"trailing",reopened)
      && visualimportedscenerender::sameRequest(previous,reopened),"malformed collection decode preserves the previous complete request");
  invalid=*fixture; invalid.surfacePrograms.clear();
  check(visualimportedscenerender::encode(invalid).rfind("visual.imported-scene.render.v8\n",0)==0,
      "ordinary retained scenes retain their legacy immutable transport");
}

#if !defined(_WIN32)
void animatedGlbAdmissionFitsViewportThreadStack() {
  std::string error;
  videohelper::gltf::GlbAdmissionOptions options;
  options.admitAnimations=true;
  options.admitSkins=true;
  options.sceneIndex=0;
  const auto asset=videohelper::gltf::decodeAnimatedGlbBaseScene(
      std::vector<std::uint8_t>(reactivecharacter::kGlb.begin(),reactivecharacter::kGlb.end()),
      options,error);
  check(asset.has_value(),error.c_str());
  if (!asset) return;

  struct AdmissionCase {
    const videohelper::gltf::GlbStaticMeshDocument* asset;
    bool admitted=false;
    std::string error;
  } test{&*asset,false,{}};
  pthread_attr_t attributes{};
  if (pthread_attr_init(&attributes)!=0) { check(false,"animated GLB thread attributes initialize"); return; }
  const bool stackSet=pthread_attr_setstacksize(&attributes,544u*1024u)==0;
  check(stackSet,"animated GLB viewport-sized POSIX stack is available");
  pthread_t thread{};
  const int created=stackSet ? pthread_create(&thread,&attributes,[](void* value)->void* {
    auto* test=static_cast<AdmissionCase*>(value);
    auto scene=videohelper::gltf::adaptAnimatedGlbMeshToVisual3DSceneHeap(
        *test->asset,test->error,0,true);
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> admitted(std::move(scene));
    test->admitted=admitted && admitted->objectCount>0 && admitted->vertexCount>3
        && admitted->cameraCount==1;
    return nullptr;
  },&test) : -1;
  pthread_attr_destroy(&attributes);
  check(created==0,"animated GLB viewport-sized POSIX admission thread starts");
  if (created!=0) return;
  if (pthread_join(thread,nullptr)!=0) { check(false,"animated GLB viewport-sized POSIX admission thread joins"); return; }
  check(test.admitted,test.error.empty()
      ? "bundled animated GLB admits on the viewport-sized stack" : test.error.c_str());
}

void v8SceneDecodeFitsViewportThreadStack() {
  std::string error;
  auto fixture=retainedSurfaceFixture(error);
  check(fixture.has_value(),error.c_str()); if (!fixture) return;
  fixture->surfacePrograms.clear();
  const auto wire=visualimportedscenerender::encode(*fixture);
  check(wire.rfind("visual.imported-scene.render.v8\n",0)==0,
      "small-stack fixture uses the canonical V8 scene branch");
  if (wire.empty()) return;

  struct DecodeCase {
    const std::string* wire;
    visualimportedscenerender::Request decoded;
    bool admitted=false;
    bool rejectedMalformed=false;
  } test{&wire,{},false,false};
  pthread_attr_t attributes{};
  if (pthread_attr_init(&attributes)!=0) { check(false,"small-stack thread attributes initialize"); return; }
  const bool stackSet=pthread_attr_setstacksize(&attributes,544u*1024u)==0;
  check(stackSet,"viewport-sized POSIX stack is available");
  pthread_t thread{};
  const int created=stackSet ? pthread_create(&thread,&attributes,[](void* value)->void* {
    auto* test=static_cast<DecodeCase*>(value);
    test->admitted=visualimportedscenerender::decode(*test->wire,test->decoded);
    if (test->admitted)
      test->rejectedMalformed=!visualimportedscenerender::decode(*test->wire+"trailing",test->decoded);
    return nullptr;
  },&test) : -1;
  pthread_attr_destroy(&attributes);
  check(created==0,"viewport-sized POSIX decode thread starts");
  if (created!=0) return;
  if (pthread_join(thread,nullptr)!=0) { check(false,"viewport-sized POSIX decode thread joins"); return; }
  check(test.admitted && test.rejectedMalformed
      && visualimportedscenerender::sameRequest(*fixture,test.decoded),
      "V8 scene decodes on the viewport stack and malformed input preserves the output");
}
#endif

void retainedSdfSurfaceReplay() {
  using namespace videowire;
  const auto request=[](unsigned cells) {
    sdfsurface::Request r;
    r.cells={cells,cells,cells}; r.maximumResidual=0.06;
    SdfRecord sphere; sphere.stableId=9007199254740001ull;
    sphere.operation=SdfOperation::sphere; sphere.parameterCount=1; sphere.parameters[0]=1.013;
    r.geometry.rootId=sphere.stableId; r.geometry.records={sphere}; return r;
  };
  const auto contract=[](StableId id) {
    auto c=contractFor(CarrierKind::geometry3D);
    c.maxVertices=100000; c.maxIndices=600000; c.maxAttributes=3; c.attributes=sdfSurfaceAttributes(id); return c;
  };
  std::string error;
  const auto coarseRequest=request(10),fineRequest=request(20);
  const auto c=contract(9700);
  const auto coarse=lowerSdfSurface(9700,9701,1,coarseRequest,c,error);
  const auto fine=lowerSdfSurface(9700,9701,1,fineRequest,c,error);
  check(coarse && fine,"SDF zero set resolves two independent sphere grids");
  if(!coarse || !fine) return;
  const auto sphereError=[&](const ValueDescriptor& v) {
    const auto& mesh=std::get<GeometryData>(v.data);
    double maximumError=0;
    std::map<std::pair<std::uint32_t,std::uint32_t>,unsigned> edgeUse;
    std::vector<std::vector<std::uint32_t>> neighbors(mesh.positions.size());
    std::set<StableId> identities(mesh.vertexIds.begin(),mesh.vertexIds.end());
    check(identities.size()==mesh.positions.size(),"SDF grid crossings have unique exact stable identities");
    for(std::size_t i=0;i<mesh.positions.size();++i) {
      const auto& p=mesh.positions[i];
      // Direct length-radius oracle does not call the production SDF evaluator.
      const double length=std::sqrt(double(p.x)*p.x+double(p.y)*p.y+double(p.z)*p.z);
      maximumError=std::max(maximumError,std::abs(length-1.013));
      const auto& n=v.attributes[0].elements[i].components;
      const double norm=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
      check(std::abs(norm-1)<1e-6 && (n[0]*p.x+n[1]*p.y+n[2]*p.z)/length>0.99999,
          "SDF normals agree with analytic outward sphere gradient");
      const auto& uv=v.attributes[1].elements[i].components;
      check(std::abs(uv[0]-(p.x+1.5)/3)<1e-6 && std::abs(uv[1]-(p.z+1.5)/3)<1e-6
          && uv[2]==0 && uv[3]==0,"SDF UV retains normalized explicit domain coordinates");
    }
    for(std::size_t face=0;face<mesh.indices.size()/3;++face) {
      const auto a=mesh.indices[face*3],b=mesh.indices[face*3+1],d=mesh.indices[face*3+2];
      const auto& p=mesh.positions[a]; const auto& q=mesh.positions[b]; const auto& r=mesh.positions[d];
      const double ux=q.x-p.x,uy=q.y-p.y,uz=q.z-p.z,vx=r.x-p.x,vy=r.y-p.y,vz=r.z-p.z;
      const double nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
      const double cx=(double(p.x)+q.x+r.x)/3,cy=(double(p.y)+q.y+r.y)/3,cz=(double(p.z)+q.z+r.z)/3;
      maximumError=std::max(maximumError,std::abs(std::sqrt(cx*cx+cy*cy+cz*cz)-1.013));
      check(nx*(p.x+q.x+r.x)+ny*(p.y+q.y+r.y)+nz*(p.z+q.z+r.z)>0,
          "SDF sphere triangles are nondegenerate and wind outward");
      check(v.attributes[2].elements[face].components[0]==static_cast<double>(coarseRequest.geometry.rootId),
          "SDF face provenance preserves the full primitive integer without float or RGB conversion");
      const std::array<std::uint32_t,3> corners{a,b,d};
      for(unsigned j=0;j<3;++j) {
        auto x=corners[j],y=corners[(j+1)%3];
        neighbors[x].push_back(y); neighbors[y].push_back(x);
        if(x>y) std::swap(x,y);
        ++edgeUse[{x,y}];
      }
    }
    check(std::all_of(edgeUse.begin(),edgeUse.end(),[](const auto& edge){return edge.second==2;}),
        "Analytic sphere sampled mesh has no boundary or nonmanifold edges");
    std::set<std::uint32_t> connected; std::vector<std::uint32_t> pending{0};
    while(!pending.empty()) {
      const auto i=pending.back(); pending.pop_back();
      if(connected.insert(i).second) pending.insert(pending.end(),neighbors[i].begin(),neighbors[i].end());
    }
    check(connected.size()==mesh.positions.size(),"Analytic sphere sampled mesh is connected");
    return maximumError;
  };
  const auto coarseError=sphereError(*coarse),fineError=sphereError(*fine);
  check(coarseError<0.06 && fineError<coarseError*0.65,
      "Halving SDF grid spacing reduces independent sphere radial error");
  const auto again=lowerSdfSurface(9700,9701,1,fineRequest,c,error);
  check(again && equal(std::get<GeometryData>(again->data),std::get<GeometryData>(fine->data))
      && equalAttributes(again->attributes,fine->attributes),"SDF surface replay is deterministic");
  auto admitted=admitValue(*fine,c,error);
  check(admitted.has_value(),"SDF surface immutable plan admits");
  if(!admitted) return;
  sdfsurface::Budget sampledBudget;
  check(sdfsurface::inspect(fineRequest,sampledBudget,error)
      && admitted->receipt().sdfSamples==sampledBudget.samples
      && admitted->receipt().sdfSampleWork==sampledBudget.sampleWork
      && admitted->receipt().sdfTemporaryBytes==sampledBudget.temporaryBytes,
      "SDF receipt retains checked sample work and conservative temporary memory");
  const auto wire=encodeRuntimeValue(*admitted);
  check(wire.size()>8 && wire[4]==13 && wire[5]==0,"SDF surface appends GCRV version thirteen");
  auto decoded=decodeRuntimeValue(wire.data(),wire.size(),c,ResourceLimits{},AdmissionContext{},error);
  check(decoded && encodeRuntimeValue(*decoded)==wire,"SDF exact IR sampling settings and geometry survive canonical transport");
  // Frozen algorithm-one/v1 scalar payload: no resource field or table count.
  const std::string legacyPayloadHex=
      "01000000000000000000f8bf000000000000f8bf000000000000f8bf000000000000f83f00000000"
      "0000f83f000000000000f83f1400000014000000140000002d431cebe2361a3fb81e85eb51b8ae3f"
      "0100000021fcffffffff1f000100000021fcffffffff1f0001000100000000000000000000000000"
      "0000006891ed7c3f35f03f0000000000000000000000000000000000000000000000000000000000"
      "000000000000000000000000000000000000000000000000000000";
  std::vector<std::uint8_t> legacyPayload;
  const auto nibble=[](char x) { return x>='a' ? x-'a'+10 : x-'0'; };
  for(std::size_t i=0;i<legacyPayloadHex.size();i+=2)
    legacyPayload.push_back(static_cast<std::uint8_t>(16*nibble(legacyPayloadHex[i])+nibble(legacyPayloadHex[i+1])));
  const auto payloadAt=std::search(wire.begin(),wire.end(),legacyPayload.begin(),legacyPayload.end());
  check(payloadAt!=wire.end(),"Nontext SDF surface retains the exact released v13 scalar payload bytes");
  if(payloadAt!=wire.end()) {
    const auto end=static_cast<std::size_t>(payloadAt-wire.begin())+legacyPayload.size();
    auto additive=wire; additive[4]=14;
    additive.insert(additive.begin()+end,12,0); // zero record resource ID, zero outline count
    const auto decodedAdditive=decodeRuntimeValue(additive,c,{}, {},error);
    check(decodedAdditive && encodeRuntimeValue(*decodedAdditive)==wire,
        "Additive v14 reads legacy SDF surfaces and canonical nontext re-emission remains v13");
    additive[end]=1;
    check(!decodeRuntimeValue(additive,c,{}, {},error),"Legacy IR inside v14 cannot claim a text resource");
  }
  auto legacy=wire; legacy[4]=12;
  check(!decodeRuntimeValue(legacy.data(),legacy.size(),c,ResourceLimits{},AdmissionContext{},error),
      "Older GCRV cannot claim the SDF surface operation");
  auto truncated=wire; truncated.pop_back();
  check(!decodeRuntimeValue(truncated.data(),truncated.size(),c,ResourceLimits{},AdmissionContext{},error),
      "Truncated SDF retained geometry refuses");
  auto trailing=wire; trailing.push_back(0);
  check(!decodeRuntimeValue(trailing.data(),trailing.size(),c,ResourceLimits{},AdmissionContext{},error),
      "Trailing SDF retained geometry bytes refuse");
  auto hostile=*fine;
  auto changed=std::make_shared<sdfsurface::Request>(*hostile.operations[0].sdfSurface);
  changed->geometry.records[0].parameters[0]=0.95; hostile.operations[0].sdfSurface=changed;
  check(!admitValue(hostile,c,error),"Stale SDF mesh refuses changed immutable radius");
  hostile=*fine; changed=std::make_shared<sdfsurface::Request>(*hostile.operations[0].sdfSurface);
  changed->minimum[0]=-1.4; hostile.operations[0].sdfSurface=changed;
  check(!admitValue(hostile,c,error),"Stale SDF mesh refuses changed finite sample domain");
  hostile=*fine; hostile.attributes[2].elements[0].components[0]=1;
  check(!admitValue(hostile,c,error),"Forged SDF primitive provenance fails replay equality");
  auto operations=fine->operations;
  hostile=*fine; changed=std::make_shared<sdfsurface::Request>(*hostile.operations[0].sdfSurface);
  changed->maximumResidual*=0.9; hostile.operations[0].sdfSurface=changed;
  check(!appendUniqueOperations(operations,hostile.operations,error),"Branch merging requires exact SDF request identity");
  const auto modified=lowerSdfSurface(9700,9701,1,*changed,c,error);
  const auto modifiedAdmitted=modified ? admitValue(*modified,c,error) : std::nullopt;
  check(modifiedAdmitted && modifiedAdmitted->receipt().deterministicDigest!=admitted->receipt().deterministicDigest,
      "SDF settings participate in the deterministic receipt even when mesh positions coincide");
  auto alias=std::make_shared<sdfsurface::Request>(fineRequest); hostile=*fine; hostile.operations[0].sdfSurface=alias;
  const auto owned=admitValue(hostile,c,error);
  alias->geometry.records[0].parameters[0]=0.8;
  check(owned && owned->descriptor().operations[0].sdfSurface->geometry.records[0].parameters[0]==1.013,
      "Admitted SDF packet owns its immutable IR despite a producer alias");
  auto small=c; small.maxVertices=3;
  check(!lowerSdfSurface(9700,9701,1,fineRequest,small,error),"SDF exact vertex count rejects before output allocation");
  small=c; small.maxIndices=3;
  check(!lowerSdfSurface(9700,9701,1,fineRequest,small,error),"SDF exact index count rejects before output allocation");
  auto limits=ResourceLimits{}; limits.maxBufferBytes=1024;
  check(!admitValue(*fine,c,limits,error),"SDF sample memory rejects before replay allocation");
  limits=ResourceLimits{}; limits.maxVertices=3;
  check(!admitValue(*fine,c,limits,error),"SDF replay applies reduced vertex resource limits");
  auto exhausted=fineRequest; exhausted.cells={64,64,64};
  check(!lowerSdfSurface(9700,9701,1,exhausted,c,error),"SDF sampling memory exhaustion refuses before large allocation");
  exhausted=fineRequest; exhausted.maximumResidual=1e-9;
  check(!lowerSdfSurface(9700,9701,1,exhausted,c,error),"Unresolved SDF edge error explicitly requests a finer grid");
  exhausted=fineRequest; exhausted.algorithmVersion=2;
  check(!lowerSdfSurface(9700,9701,1,exhausted,c,error),"Unsupported SDF sampling algorithm refuses");
  exhausted=fineRequest; exhausted.geometry.records[0].operation=static_cast<SdfOperation>(255);
  check(!lowerSdfSurface(9700,9701,1,exhausted,c,error),"Unsupported SDF operations refuse before sampling");
  exhausted=fineRequest; exhausted.geometry.rootId=1234;
  check(!lowerSdfSurface(9700,9701,1,exhausted,c,error),"Forged SDF root refuses before sampling");
  exhausted=fineRequest; exhausted.geometry.records[0].parameters[7]=1;
  check(!lowerSdfSurface(9700,9701,1,exhausted,c,error),"SDF source refuses hidden data beyond the declared parameters");
  exhausted=fineRequest; exhausted.maximum[0]=std::numeric_limits<double>::infinity();
  check(!lowerSdfSurface(9700,9701,1,exhausted,c,error),"Unbounded SDF sample domains refuse");
  auto hidden=request(2); hidden.geometry.records[0].parameters[0]=0.1;
  SdfRecord hiddenTranslation; hiddenTranslation.stableId=9703; hiddenTranslation.operation=SdfOperation::translate;
  hiddenTranslation.inputCount=1; hiddenTranslation.inputs[0]=hidden.geometry.rootId;
  hiddenTranslation.parameterCount=3; hiddenTranslation.parameters={-0.75,-0.75,-0.75,0,0,0,0,0};
  hidden.geometry.records.push_back(hiddenTranslation); hidden.geometry.rootId=hiddenTranslation.stableId;
  check(!lowerSdfSurface(9700,9701,1,hidden,c,error),"A sign-changing cell center with same-sign corners refuses under-resolution");
  // A plane is intentionally clipped, with an open boundary and no box caps.
  auto plane=fineRequest; plane.geometry.records[0].operation=SdfOperation::plane;
  plane.geometry.records[0].parameterCount=4; plane.geometry.records[0].parameters={0,1,0,0.173,0,0,0,0};
  const auto clipped=lowerSdfSurface(9700,9701,1,plane,c,error);
  check(clipped.has_value(),"An unbounded SDF plane admits only the authored finite clipping domain");
  if(clipped) for(const auto& p:std::get<GeometryData>(clipped->data).positions)
    check(std::abs(p.y+0.173)<1e-6 && p.x>=-1.5 && p.x<=1.5 && p.z>=-1.5 && p.z<=1.5,
        "Clipped SDF plane stays on the analytic zero set and finite domain");
  plane.geometry.records[0].parameters[3]=0;
  check(!lowerSdfSurface(9700,9701,1,plane,c,error),"Zero-valued lattice edges explicitly refuse an ambiguous plane tie");
  auto cut=fineRequest;
  SdfRecord inner=cut.geometry.records[0]; inner.stableId-=1; inner.parameters[0]=0.487;
  SdfRecord subtraction; subtraction.stableId=9702; subtraction.operation=SdfOperation::subtraction;
  subtraction.inputCount=2; subtraction.inputs={cut.geometry.rootId,inner.stableId};
  cut.geometry.records.push_back(inner); cut.geometry.records.push_back(subtraction); cut.geometry.rootId=subtraction.stableId;
  const auto hollow=lowerSdfSurface(9700,9701,1,cut,c,error);
  check(hollow.has_value(),"CSG subtraction resolves outer and cut surfaces");
  if(hollow) {
    bool outer=false,cutter=false;
    const auto& mesh=std::get<GeometryData>(hollow->data);
    for(std::size_t face=0;face<mesh.indices.size()/3;++face) {
      double x=0,y=0,z=0;
      for(unsigned i=0;i<3;++i) { const auto& p=mesh.positions[mesh.indices[face*3+i]]; x+=p.x/3; y+=p.y/3; z+=p.z/3; }
      const double radial=std::sqrt(x*x+y*y+z*z),a=radial-1.013,b=0.487-radial;
      const auto winner=a>=b ? fineRequest.geometry.rootId : inner.stableId;
      const auto provenance=hollow->attributes[2].elements[face].components[0];
      check(provenance==static_cast<double>(winner),"CSG face contributor follows independent max(A,-B) ownership");
      outer|=winner==fineRequest.geometry.rootId; cutter|=winner==inner.stableId;
    }
    check(outer && cutter,"CSG retains both exact outer and subtraction primitive IDs");
  }
  // A shared DAG can have exponential evaluator work despite a small record count.
  auto expensive=fineRequest;
  for(unsigned i=0;i<28;++i) {
    SdfRecord v; v.stableId=9800+i; v.operation=SdfOperation::unionOp; v.inputCount=2;
    v.inputs={expensive.geometry.rootId,expensive.geometry.rootId}; expensive.geometry.records.push_back(v); expensive.geometry.rootId=v.stableId;
  }
  check(!lowerSdfSurface(9700,9701,1,expensive,c,error),"Sample times expanded IR work refuses an exponential shared DAG");
  auto oldValue=geometryValue(); const auto oldContract=contractFor(CarrierKind::geometry3D);
  const auto old=admitValue(oldValue,oldContract,error);
  if(old) {
    // Frozen released v13 copy/triangle fixture, including every tail count.
    const std::string goldenHex=
      "474352560d000000020000000166000000000000004d000000000000000900000000000000010000"
      "00000000000000000001000000000000000000000000000000010000000000000066000000000000"
      "00010000000000000000660000000000000000000000000000000000000000000000000000000000"
      "0000030000000000000003000000000000000000000000000000000000000b000000000000000000"
      "803f00000000000000000c00000000000000000000000000803f000000000d000000000000000000"
      "00000100000002000000000000000000000000000000";
    std::vector<std::uint8_t> golden;
    const auto nibble=[](char c) { return c>='a' ? c-'a'+10 : c-'0'; };
    for(std::size_t i=0;i<goldenHex.size();i+=2)
      golden.push_back(static_cast<std::uint8_t>(16*nibble(goldenHex[i])+nibble(goldenHex[i+1])));
    check(encodeRuntimeValue(*old)==golden,"Ordinary nontext emission remains byte-identical to released GCRV thirteen");
    const auto goldenDecoded=decodeRuntimeValue(golden,oldContract,{}, {},error);
    check(goldenDecoded && encodeRuntimeValue(*goldenDecoded)==golden,"Frozen GCRV thirteen fixture retains legacy decoder identity");
    auto oldWire=encodeRuntimeValue(*old); oldWire[4]=12;
    check(decodeRuntimeValue(oldWire.data(),oldWire.size(),oldContract,ResourceLimits{},AdmissionContext{},error).has_value(),
        "Released GCRV twelve ordinary geometry retains its exact input path");
  }
}

// Analytic vector fixtures exercise retained consumers, not embedded-font extraction.
videowire::sdfsurface::Request retainedTextSurfaceRequest() {
  using namespace videowire;
  SdfTextOutline outline;
  outline.sourceText="O"; outline.sourceTextSha256=sdfTextSourceSha256(outline.sourceText);
  outline.embeddedFontToken="InterRegular_ttf";
  outline.embeddedFontSha256=sdfTextEmbeddedFontSha256(outline.embeddedFontToken);
  outline.glyphs={{1,0,0,0,2.026,0}};
  outline.contours={{0,0,4,1},{0,4,4,-1}};
  outline.segments={{-1.013,-1.013,1.013,-1.013},{1.013,-1.013,1.013,1.013},
      {1.013,1.013,-1.013,1.013},{-1.013,1.013,-1.013,-1.013},
      {-.487,-.487,-.487,.487},{-.487,.487,.487,.487},
      {.487,.487,.487,-.487},{.487,-.487,-.487,-.487}};
  outline.stableId=sdfTextOutlineStableId(outline);
  sdfsurface::Request request;
  request.maximumResidual=.06;
  request.geometry.schemaVersion=kSdfIrTextSchemaVersion;
  request.geometry.rootId=9007199254740001ull;
  SdfRecord text; text.stableId=request.geometry.rootId; text.operation=SdfOperation::textExtrusion;
  text.parameterCount=1; text.parameters[0]=.613; text.resourceStableId=outline.stableId;
  request.geometry.records={text}; request.geometry.textOutlines={outline};
  return request;
}

void retainedSdfTextSurfaceReplay() {
  using namespace videowire;
  auto request=retainedTextSurfaceRequest();
  auto contract=contractFor(CarrierKind::geometry3D);
  contract.maxVertices=100000; contract.maxIndices=600000;
  contract.maxAttributes=3; contract.attributes=sdfSurfaceAttributes(9900);
  std::string error;
  sdfsurface::Budget budget;
  const std::uint64_t samples=24*24*24,cells=23*23*23;
  const auto queries=samples+2*cells+7*(7*samples)+12*cells;
  check(sdfsurface::inspect(request,budget,error) && budget.sampleWork==queries
      && budget.textRecordVisits==queries && budget.textSegmentVisits==8*queries,
      "Retained text charges every lattice, cell, residual, gradient and face query against exact segment visits");
  auto equivalentLegacy=request; equivalentLegacy.geometry.schemaVersion=kSdfIrSchemaVersion;
  equivalentLegacy.geometry.textOutlines.clear(); equivalentLegacy.geometry.records[0].resourceStableId=0;
  equivalentLegacy.geometry.records[0].operation=SdfOperation::sphere;
  sdfsurface::Budget legacyBudget;
  check(sdfsurface::inspect(equivalentLegacy,legacyBudget,error) && budget.temporaryBytes>legacyBudget.temporaryBytes,
      "Text resources add immutable vector, string, lookup and validation-buffer memory");
  const auto value=lowerSdfSurface(9900,9901,1,request,contract,error);
  check(value.has_value(),"Canonical SDF v2 text resolves a retained counter and extrusion");
  if(!value) { std::cerr<<error<<'\n'; return; }
  const auto distance=[](double x,double y,double z) {
    const auto rectangle=[](double x,double y,double radius) {
      const double a=std::abs(x)-radius,b=std::abs(y)-radius;
      return std::hypot(std::max(a,0.0),std::max(b,0.0))+std::min(std::max(a,b),0.0);
    };
    const double xy=std::max(rectangle(x,y,1.013),-rectangle(x,y,.487)),cap=std::abs(z)-.613/2;
    return std::hypot(std::max(xy,0.0),std::max(cap,0.0))+std::min(std::max(xy,cap),0.0);
  };
  sdfsurface::detail::Evaluator evaluator(request.geometry);
  const std::array<std::array<double,3>,6> queriesAt{{{0,0,0},{.75,0,0},{.75,0,.9},
      {.75,0,-.9},{1.4,1.4,.9},{.1,.2,.1}}};
  for(const auto& point:queriesAt) {
    const auto sample=evaluator.at(point);
    check(std::abs(sample.distance-distance(point[0],point[1],point[2]))<1e-12
        && sample.primitiveId==request.geometry.rootId,
        "Text evaluator agrees with independent counter, solid, front/back and Euclidean corner distances");
  }
  const auto& mesh=std::get<GeometryData>(value->data);
  bool front=false,back=false,counter=false,side=false;
  for(std::size_t i=0;i<mesh.positions.size();++i) {
    const auto& p=mesh.positions[i]; const auto& n=value->attributes[0].elements[i].components;
    check(std::abs(distance(p.x,p.y,p.z))<=request.maximumResidual,
        "Retained text crossings agree with independent rectangular-ring extrusion distance");
    const auto h=request.gradientStep;
    const std::array<double,3> gradient{distance(p.x+h,p.y,p.z)-distance(p.x-h,p.y,p.z),
      distance(p.x,p.y+h,p.z)-distance(p.x,p.y-h,p.z),distance(p.x,p.y,p.z+h)-distance(p.x,p.y,p.z-h)};
    const auto length=std::sqrt(gradient[0]*gradient[0]+gradient[1]*gradient[1]+gradient[2]*gradient[2]);
    const double norm=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
    check(std::abs(norm-1)<1e-6 && length>0
        && (n[0]*gradient[0]+n[1]*gradient[1]+n[2]*gradient[2])/length>.99999,
        "Retained text normals match independent outward cap, side and counter gradients");
    const auto& uv=value->attributes[1].elements[i].components;
    check(std::abs(uv[0]-(p.x+1.5)/3)<1e-6 && std::abs(uv[1]-(p.z+1.5)/3)<1e-6
        && uv[2]==0 && uv[3]==0,"Text surface UV keeps authored normalized box X/Z coordinates");
    front|=n[2]>.99; back|=n[2]<-.99;
    if(std::abs(p.z)<.2) {
      counter|=std::max(std::abs(p.x),std::abs(p.y))<.6 && n[0]*p.x+n[1]*p.y<0;
      side|=std::max(std::abs(p.x),std::abs(p.y))>.9 && n[0]*p.x+n[1]*p.y>0;
    }
  }
  check(front && back && counter && side,"Retained text contains both caps, outward sides and inward counter walls");
  for(std::size_t face=0;face<mesh.indices.size()/3;++face) {
    const auto& a=mesh.positions[mesh.indices[3*face]];
    const auto& b=mesh.positions[mesh.indices[3*face+1]];
    const auto& c=mesh.positions[mesh.indices[3*face+2]];
    const double x=(double(a.x)+b.x+c.x)/3,y=(double(a.y)+b.y+c.y)/3,z=(double(a.z)+b.z+c.z)/3;
    check(std::abs(distance(x,y,z))<=request.maximumResidual
        && !(std::abs(x)<.4 && std::abs(y)<.4),"Text triangles retain the open glyph counter on front and back");
    const std::array<double,3> normal{(b.y-a.y)*(c.z-a.z)-(b.z-a.z)*(c.y-a.y),
      (b.z-a.z)*(c.x-a.x)-(b.x-a.x)*(c.z-a.z),(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x)};
    double orientation=0;
    for(unsigned i=0;i<3;++i) for(unsigned axis=0;axis<3;++axis)
      orientation+=normal[axis]*value->attributes[0].elements[mesh.indices[3*face+i]].components[axis];
    check(orientation>0,"Retained text triangles wind with the outward sampled normals");
    check(value->attributes[2].elements[face].components[0]==static_cast<double>(request.geometry.rootId),
        "Text provenance keeps the full operation ID rather than its outline resource ID");
  }
  const auto again=lowerSdfSurface(9900,9901,1,request,contract,error);
  check(again && equal(std::get<GeometryData>(again->data),mesh) && equalAttributes(again->attributes,value->attributes),
      "Canonical retained text sampling is deterministic");
  const auto admitted=admitValue(*value,contract,error);
  check(admitted && admitted->receipt().sdfTextRecordVisits==budget.textRecordVisits
      && admitted->receipt().sdfTextSegmentVisits==budget.textSegmentVisits
      && admitted->receipt().sdfTemporaryBytes==budget.temporaryBytes,"Geometry Core retains checked text work and memory receipts");
  if(!admitted) return;
  const auto wire=encodeRuntimeValue(*admitted);
  check(wire[4]==14 && wire[5]==0,"Retained text appends GCRV fourteen without changing ordinary geometry emission");
  // Pin the released v14 envelope independently of the current GCRV writer.
  // Reuse canonical sampled arrays and the unchanged outline codec, so this
  // oracle measures format compatibility without freezing the SDF algorithm.
  std::vector<std::uint8_t> releasedText{'G','C','R','V'};
  const auto integer=[&](std::uint64_t bits,std::size_t width) {
    for (std::size_t i=0;i<width;++i) releasedText.push_back(static_cast<std::uint8_t>(bits>>(8*i)));
  };
  const auto doubleValue=[&](double value) {
    std::uint64_t bits=0; std::memcpy(&bits,&value,sizeof(bits)); integer(bits,8);
  };
  const auto floatValue=[&](float value) {
    std::uint32_t bits=0; std::memcpy(&bits,&value,sizeof(bits)); integer(bits,4);
  };
  const auto stringValue=[&](const std::string& value) {
    integer(value.size(),2); releasedText.insert(releasedText.end(),value.begin(),value.end());
  };
  integer(14,2); integer(0,2); integer(2,4); integer(1,1);
  integer(9900,8); integer(9901,8); integer(1,8);
  integer(1,1); integer(0,1); integer(0,1); integer(0,1);
  integer(0,8); integer(1,8); integer(3,8); integer(1,8);
  const std::array<const char*,3> names{"normal","uv","sdfPrimitiveId"};
  const std::array<std::uint8_t,3> types{2,2,5},domains{2,2,4},interpolations{4,3,2};
  for (std::size_t i=0;i<3;++i) {
    integer(9900+i,8); stringValue(names[i]);
    integer(types[i],1); integer(domains[i],1); integer(interpolations[i],1);
    integer(value->attributes[i].elements.size(),8);
    for (const auto& element:value->attributes[i].elements)
      for (const auto component:element.components) doubleValue(component);
  }
  integer(9900,8); integer(134,1); integer(9901,8); integer(9900,8); integer(0,8);
  for (unsigned i=0;i<4;++i) floatValue(0);
  integer(1,4);
  for (unsigned i=0;i<3;++i) doubleValue(-1.5);
  for (unsigned i=0;i<3;++i) doubleValue(1.5);
  for (unsigned i=0;i<3;++i) integer(23,4);
  doubleValue(.0001); doubleValue(.06);
  integer(2,4); integer(9007199254740001ull,8); integer(1,4);
  integer(9007199254740001ull,8); integer(27,1); integer(0,1); integer(1,1);
  integer(0,8); integer(0,8); doubleValue(.613);
  for (unsigned i=1;i<8;++i) doubleValue(0);
  integer(request.geometry.textOutlines[0].stableId,8); integer(1,4);
  const auto outlineBytes=encodeSdfTextOutline(request.geometry.textOutlines[0]);
  integer(outlineBytes.size(),4); releasedText.insert(releasedText.end(),outlineBytes.begin(),outlineBytes.end());
  integer(mesh.positions.size(),8); integer(mesh.indices.size(),8);
  for (std::size_t i=0;i<mesh.positions.size();++i) {
    floatValue(mesh.positions[i].x); floatValue(mesh.positions[i].y); floatValue(mesh.positions[i].z);
    integer(mesh.vertexIds[i],8);
  }
  for (const auto index:mesh.indices) integer(index,4);
  integer(0,4); integer(0,4); integer(0,4);
  check(wire==releasedText,"Zero-coupling retained text preserves every released GCRV fourteen format byte");
  const auto decoded=decodeRuntimeValue(wire,contract,{}, {},error);
  check(decoded && encodeRuntimeValue(*decoded)==wire
      && sdfsurface::equal(*decoded->descriptor().operations[0].sdfSurface,request)
      && decoded->receipt().deterministicDigest==admitted->receipt().deterministicDigest,
      "GCRV fourteen roundtrips exact vector resources, references, mesh and receipt identity");
  auto hostileWire=wire; hostileWire[4]=13;
  check(!decodeRuntimeValue(hostileWire,contract,{}, {},error),"GCRV thirteen refuses a text schema instead of dropping resources");
  hostileWire=wire; hostileWire.pop_back();
  check(!decodeRuntimeValue(hostileWire,contract,{}, {},error),"Truncated retained text refuses");
  hostileWire=wire; hostileWire.push_back(0);
  check(!decodeRuntimeValue(hostileWire,contract,{}, {},error),"Trailing retained text bytes refuse");
  const auto payload=encodeSdfTextOutline(request.geometry.textOutlines[0]);
  const auto payloadAt=std::search(wire.begin(),wire.end(),payload.begin(),payload.end(),
      [](std::uint8_t a,char b){return a==static_cast<std::uint8_t>(b);});
  check(payloadAt!=wire.end(),"GCRV fourteen contains the existing exact outline payload");
  if(payloadAt!=wire.end()) {
    const auto offset=static_cast<std::size_t>(payloadAt-wire.begin());
    hostileWire=wire; hostileWire[offset]^=1;
    check(!decodeRuntimeValue(hostileWire,contract,{}, {},error),"Forged wire resource identity refuses");
    hostileWire=wire; hostileWire[offset-16]^=1;
    check(!decodeRuntimeValue(hostileWire,contract,{}, {},error),"Forged wire record-to-resource reference refuses");
    hostileWire=wire; for(unsigned i=0;i<4;++i) hostileWire[offset-4+i]=255;
    check(!decodeRuntimeValue(hostileWire,contract,{}, {},error),"Over-capacity wire outline payload refuses before allocation");
    hostileWire=wire; hostileWire.resize(offset+payload.size()-1);
    check(!decodeRuntimeValue(hostileWire,contract,{}, {},error),"Truncated vector payload refuses before geometry replay");
  }
  auto changed=request; changed.geometry.records[0].resourceStableId+=1;
  check(!sdfsurface::equal(changed,request),"Surface equality includes text resource references");
  changed=request; changed.geometry.textOutlines[0].glyphs[0].glyphId=2;
  auto& changedOutline=changed.geometry.textOutlines[0]; changedOutline.stableId=sdfTextOutlineStableId(changedOutline);
  changed.geometry.records[0].resourceStableId=changedOutline.stableId;
  check(!sdfsurface::equal(changed,request),"Surface equality includes complete shaping identity even with unchanged contours");
  auto merged=value->operations,other=value->operations;
  other[0].sdfSurface=std::make_shared<const sdfsurface::Request>(changed);
  check(!appendUniqueOperations(merged,other,error),"Branches cannot alias distinct retained text resources");
  const auto changedValue=lowerSdfSurface(9900,9901,1,changed,contract,error);
  const auto changedAdmitted=changedValue ? admitValue(*changedValue,contract,error) : std::nullopt;
  check(changedAdmitted && equal(std::get<GeometryData>(changedValue->data),mesh)
      && changedAdmitted->receipt().deterministicDigest!=admitted->receipt().deterministicDigest,
      "Exact glyph identity changes the receipt even when sampled geometry is identical");
  auto alias=std::make_shared<sdfsurface::Request>(request); auto aliased=*value; aliased.operations[0].sdfSurface=alias;
  const auto owned=admitValue(aliased,contract,error); alias->geometry.textOutlines[0].segments[0].x0=-9;
  check(owned && sdfsurface::equal(*owned->descriptor().operations[0].sdfSurface,request),
      "Admitted text owns its immutable outline vectors despite a mutable producer alias");
  const auto refusal=[&](const sdfsurface::Request& invalid,const char* message) {
    sdfsurface::Budget refused; check(!sdfsurface::inspect(invalid,refused,error),message);
  };
  changed=request; changed.geometry.schemaVersion=kSdfIrSchemaVersion;
  refusal(changed,"Legacy SDF surface schema refuses retained text resources");
  changed=request; changed.geometry.schemaVersion=3; refusal(changed,"Unknown SDF surface schema refuses");
  changed=request; changed.geometry.records[0].resourceStableId=1; refusal(changed,"Missing text resource reference refuses");
  changed=request; changed.geometry.records[0].parameters[0]=0; refusal(changed,"Zero extrusion depth refuses before sampling");
  changed=request; changed.geometry.textOutlines.push_back(changed.geometry.textOutlines[0]);
  refusal(changed,"Duplicate text resources refuse canonical ordering");
  changed=request; changed.geometry.textOutlines[0].segments[0].x0-=.01;
  changed.geometry.textOutlines[0].stableId=sdfTextOutlineStableId(changed.geometry.textOutlines[0]);
  changed.geometry.records[0].resourceStableId=changed.geometry.textOutlines[0].stableId;
  refusal(changed,"Open text contours refuse even with a refreshed resource identity");
  changed=request; changed.geometry.textOutlines[0].sourceTextSha256[0]='x';
  refusal(changed,"Malformed source hash refuses");
  changed=request; changed.geometry.records[0].operation=SdfOperation::sphere;
  refusal(changed,"Nontext records cannot own vector text resources");
  changed=request; changed.geometry.textOutlines.clear(); refusal(changed,"Canonical SDF v2 cannot omit its text resource table");
  changed=request; changed.maximumResidual=1e-9;
  check(!lowerSdfSurface(9900,9901,1,changed,contract,error),"Under-resolved text residual refuses rather than losing quality");
  auto limits=ResourceLimits{}; limits.maxBufferBytes=budget.temporaryBytes-1;
  check(!admitValue(*value,contract,limits,error),"Text sample and resource memory respects reduced caller capacity");
  // Expanded DAG edges charge repeated segment evaluation, not unique resources.
  changed=request;
  for(unsigned i=0;i<5;++i) {
    SdfRecord unite; unite.stableId=9910+i; unite.operation=SdfOperation::unionOp; unite.inputCount=2;
    unite.inputs={changed.geometry.rootId,changed.geometry.rootId};
    changed.geometry.records.push_back(unite); changed.geometry.rootId=unite.stableId;
  }
  refusal(changed,"Expanded text segment work refuses the unchanged hundred-million visit cap");
  check(error=="SDF text evaluation work capacity exceeded","Segment work refuses before the looser two-hundred-million sample cap");
  changed.geometry.records.resize(4); changed.geometry.rootId=9912;
  sdfsurface::Budget sharedBudget;
  check(sdfsurface::inspect(changed,sharedBudget,error) && sharedBudget.textSegmentVisits<100000000
      && sharedBudget.textSegmentVisits*2>100000000,"Two individually admitted text surfaces can exceed the aggregate segment cap");
  auto aggregate=*value; aggregate.operations[0].sdfSurface=std::make_shared<const sdfsurface::Request>(changed);
  aggregate.operations.push_back(aggregate.operations[0]); aggregate.operations.back().stableId=9920;
  BudgetReceipt receipt;
  check(!inspectValue(aggregate,AdmissionContext{},receipt,error)
      && error=="SDF surface plan exceeds aggregate sample work or temporary memory",
      "Geometry Core rejects aggregate retained text segment work before replay");
}

void typedSpectrumDomainsRetainSourcesAndSamples() {
  std::string error;
  const auto admittedSource=videohelper::gltf::adaptGlbMeshToGeometryCore(importedGeometryFixture(),1,9601,9600,1,error);
  check(admittedSource.has_value(),"typed spectrum retains the released imported mesh authority");
  if (!admittedSource) return;
  const auto source=std::make_optional(admittedSource->descriptor());
  auto binding=spectrumDomainBinding(); binding.direction={1,0,0};
  auto settings=spectrumDomainSettings(Domain::vertex,ValueType::floatValue,5);
  const auto scalar=constructedFieldContract(Domain::vertex,ValueType::floatValue);
  auto field=lowerAudioSpectrumField(*source,9610,binding,settings,scalar,error);
  check(field.has_value(),"typed spectrum samples the exact authored UV domain");
  if (!field) { std::cerr<<error<<'\n'; return; }
  check(field->spectrumFields[0].coordinates==std::vector<float>{0,1,0.5f},"UV samples retain authored U, not XYZ positions");
  RuntimeFieldEvaluation runtime;
  runtime.spectrumAt=[](const auto&,auto& bands,std::string&) { bands=spectrumDomainBands(); return true; };
  ValueDescriptor replayed;
  const auto replayedUv=validateOperationPlan(*field,{},nullptr,error,&replayed,&runtime);
  check(replayedUv,"UV scalar field evaluates in ordered replay");
  if (!replayedUv) { std::cerr<<error<<'\n'; return; }
  const auto& values=std::get<FieldData>(replayed.data).elements;
  check(values.size()==3 && std::abs(values[0].components[0]-0.1)<1e-6
      && std::abs(values[1].components[0]-0.9)<1e-6 && std::abs(values[2].components[0]-0.5)<1e-6,
      "independent UV oracle selects bins two/five and interpolates bins three/four equally");
  binding.interpolate=false;
  auto nearest=lowerAudioSpectrumField(*source,9610,binding,settings,scalar,error);
  check(nearest && validateOperationPlan(*nearest,{},nullptr,error,&replayed,&runtime)
      && std::abs(std::get<FieldData>(replayed.data).elements[2].components[0]-0.7)<1e-6,
      "nearest-bin oracle rounds the halfway sample to band four");
  binding.interpolate=true; binding.gain=2; binding.bias=-0.25f;
  auto shaped=lowerAudioSpectrumField(*source,9610,binding,settings,scalar,error);
  check(shaped && validateOperationPlan(*shaped,{},nullptr,error,&replayed,&runtime)
      && std::abs(std::get<FieldData>(replayed.data).elements[2].components[0]-0.75)<1e-6,
      "field gain and bias operate on the independently computed interpolation");
  binding.gain=1; binding.bias=0;
  auto vSettings=settings; vSettings.mode=6;
  auto vField=lowerAudioSpectrumField(*source,9610,binding,vSettings,scalar,error);
  check(vField && vField->spectrumFields[0].coordinates==std::vector<float>{0,0,1},
      "UV V oracle preserves the imported texture second coordinate");
  auto frequencySettings=settings; frequencySettings.mode=4;
  const auto frequency=arbitblockb::kBandLoHz*std::pow(arbitblockb::kBandHiHz/arbitblockb::kBandLoHz,3.5/arbitblockb::kBandCount);
  frequencySettings.values[2]=frequency; frequencySettings.values[3]=frequency;
  auto frequencyField=lowerAudioSpectrumField(*source,9610,binding,frequencySettings,scalar,error);
  check(frequencyField && validateOperationPlan(*frequencyField,{},nullptr,error,&replayed,&runtime)
      && std::all_of(std::get<FieldData>(replayed.data).elements.begin(),std::get<FieldData>(replayed.data).elements.end(),
          [](const auto& e){return std::abs(e.components[0]-0.3)<1e-6;}),
      "independent logarithmic band-center frequency oracle selects band three");
  binding.normalize=true;
  auto normalized=lowerAudioSpectrumField(*source,9610,binding,settings,scalar,error);
  RuntimeFieldEvaluation normalizedRuntime;
  normalizedRuntime.spectrumAt=[](const auto& b,auto& bands,std::string& failure) {
    spectrum::Follower follower;
    return spectrum::evaluate(b,spectrumDomainBands(),3.25,{},follower,bands,failure);
  };
  check(normalized && validateOperationPlan(*normalized,{},nullptr,error,&replayed,&normalizedRuntime)
      && std::abs(std::get<FieldData>(replayed.data).elements[2].components[0]-5.0/9.0)<1e-6,
      "selected peak normalization agrees with the independent half-bin oracle");
  auto op=field->operations.back();
  std::vector<float> coordinates;
  auto moved=source->data;
  for (auto& position : std::get<GeometryData>(moved).positions) position.x+=100;
  check(spectrumCoordinates(moved,source->carrier,source->attributes,op,binding,coordinates,error)
      && coordinates==std::vector<float>{0,1,0.5f},"UV sampling is invariant to a position-only source perturbation");
  check(!spectrumCoordinates(moved,source->carrier,{},op,binding,coordinates,error),"missing authored UV fails closed");
  auto wrongUv=source->attributes;
  for (auto& a : wrongUv) if (a.descriptor.name=="uv") a.descriptor.domain=Domain::point;
  check(!spectrumCoordinates(moved,source->carrier,wrongUv,op,binding,coordinates,error),"wrong UV domain fails closed");
  auto instances=instanceAppearanceFixture(0,error);
  check(instances.has_value(),"stable instance coordinate fixture is retained");
  if (instances) {
    op.field=spectrumDomainSettings(Domain::instance,ValueType::floatValue,7);
    auto data=instances->data;
    check(spectrumCoordinates(data,CarrierKind::instances3D,{},op,binding,coordinates,error)
        && coordinates==std::vector<float>{0,1.0f/3,2.0f/3,1},"instance-index oracle follows stable identity ranks");
    auto& rows=std::get<InstancesData>(data).instances;
    std::swap(rows[0],rows[3]);
    check(spectrumCoordinates(data,CarrierKind::instances3D,{},op,binding,coordinates,error)
        && coordinates==std::vector<float>{1,1.0f/3,2.0f/3,0},"instance-index samples follow identity after record reorder");
    rows[1].stableId=rows[0].stableId;
    check(!spectrumCoordinates(data,CarrierKind::instances3D,{},op,binding,coordinates,error),"duplicate instance identities fail closed");
  }
  for (unsigned domain=0;domain<4;++domain) {
    auto value=spectrumDomainFixture(domain,error);
    check(value.has_value(),"all retained spectrum carrier domains lower through existing consumers");
    if (!value) { std::cerr<<error<<'\n'; continue; }
    const auto contract=spectrumDomainTerminalContract(*value);
    auto admitted=admitValue(*value,contract,error);
    check(admitted.has_value(),"typed spectrum terminal retains identity and resource admission");
    if (!admitted) continue;
    const auto bytes=encodeRuntimeValue(*admitted);
    const auto decoded=decodeRuntimeValue(bytes,contract,{}, {},error);
    check(decoded && encodeRuntimeValue(*decoded)==bytes,"v12 typed spectrum bindings and source program roundtrip exactly");
    auto old=bytes; old[4]=11;
    check(!decodeRuntimeValue(old,contract,{}, {},error),"older schema rejects the appended spectrum operation");
    const auto replayedDomain=validateOperationPlan(*value,{},nullptr,error,&replayed,&runtime);
    check(replayedDomain,"all spectrum domains replay at the requested frame");
    if (!replayedDomain) { std::cerr<<error<<'\n'; continue; }
    RuntimeFieldEvaluation missing;
    check(!validateOperationPlan(*value,{},nullptr,error,nullptr,&missing),"runtime spectrum evaluation refuses missing shared analysis");
    auto hostile=*value; hostile.spectrumFields[0].coordinates[0]=0.25f;
    check(!admitValue(hostile,contract,error),"persisted sample coordinates cannot contradict their source");
    hostile=*value; ++hostile.spectrumFields[0].sourceGeometryStableId;
    check(!admitValue(hostile,contract,error),"spectrum fields reject a different source identity");
    hostile=*value; hostile.spectrumFields.push_back(hostile.spectrumFields[0]);
    check(!admitValue(hostile,contract,error),"duplicate spectrum field identities are refused");
    auto limits=ResourceLimits{}; limits.maxBufferBytes=32;
    check(!admitValue(*value,contract,limits,error),"typed spectrum retains hard buffer budget refusal");
    RuntimeFieldEvaluation changed;
    changed.spectrumAt=[](const auto&,auto& bands,std::string&) { bands=spectrumDomainBands(true); return true; };
    ValueDescriptor perturbed;
    const auto replayedChanged=validateOperationPlan(*value,{},nullptr,error,&perturbed,&changed);
    check(replayedChanged,"source-spectrum perturbation replays the same operation program");
    if (!replayedChanged) { std::cerr<<error<<'\n'; continue; }
    if (domain==3) {
      const auto* emission=findNamedAttribute(replayed.attributes,"emission");
      const auto* later=findNamedAttribute(perturbed.attributes,"emission");
      check(emission && later && emission->elements[0].components!=later->elements[0].components,
          "existing instance emission consumes sampled typed Color values");
    } else if (domain==1) {
      const auto& a=std::get<InstancesData>(replayed.data).instances;
      const auto& b=std::get<InstancesData>(perturbed.data).instances;
      check(a.size()==b.size() && a[0].stableId==b[0].stableId && a[0].transform.translation.x!=b[0].transform.translation.x,
          "point field displacement reaches retained instances without changing identity");
    } else {
      const auto& a=std::get<GeometryData>(replayed.data); const auto& b=std::get<GeometryData>(perturbed.data);
      check(a.vertexIds==b.vertexIds && a.indices==b.indices && a.positions[0].x!=b.positions[0].x,
          "vertex and curve fields reach retained mesh displacement with unchanged topology");
    }
  }
}

void staticPointSpectrumStableIdentitySampling() {
  std::string error;
  const auto mesh=lowerGrid(9640,9640,1,2,2,1,constructedFieldMeshContract(),error);
  const auto points=mesh ? lowerPointsFromVertices(*mesh,9641,contractFor(CarrierKind::points3D),error) : std::nullopt;
  check(points.has_value(),"static spectrum point ranks retain the existing mesh-to-points authority");
  if (!points) return;
  const auto settings=spectrumDomainSettings(Domain::point,ValueType::floatValue,7);
  const auto contract=constructedFieldContract(Domain::point,ValueType::floatValue);
  const auto field=lowerAudioSpectrumField(*points,9642,spectrumDomainBinding(),settings,contract,error);
  check(field && field->fieldDomainCardinality==4
      && field->spectrumFields[0].coordinates==std::vector<float>{0,1.0f/3,2.0f/3,1},
      "static point spectrum fields persist exact nonempty stable identity rank coordinates");
  if (!field) return;
  RuntimeFieldEvaluation runtime;
  runtime.spectrumAt=[](const auto&,auto& bands,std::string&) { bands=spectrumDomainBands(); return true; };
  ValueDescriptor replayed;
  check(validateOperationPlan(*field,{},nullptr,error,&replayed,&runtime)
      && std::get<FieldData>(replayed.data).elements.size()==4
      && std::abs(std::get<FieldData>(replayed.data).elements.front().components[0]-0.1)<1e-6
      && std::abs(std::get<FieldData>(replayed.data).elements.back().components[0]-0.9)<1e-6,
      "static point rank endpoints replay the independently selected bands");
  const auto admitted=admitValue(*field,contract,error);
  check(admitted.has_value(),"static point rank field retains strict ordinary admission");
  if (admitted) {
    const auto bytes=encodeRuntimeValue(*admitted);
    const auto decoded=decodeRuntimeValue(bytes,contract,{}, {},error);
    check(decoded && encodeRuntimeValue(*decoded)==bytes,"static point rank field transport roundtrips exactly");
  }
  auto invalid=*field;
  invalid.spectrumFields[0].coordinates.clear();
  check(!admitValue(invalid,contract,error),"static point rank fields reject empty saved coordinates");
  invalid=*field; invalid.spectrumFields[0].coordinates[0]=1;
  check(!admitValue(invalid,contract,error),"static point rank fields reject coordinates that disagree with their retained source");
  auto reordered=points->data;
  auto& rows=std::get<PointsData>(reordered).points;
  std::swap(rows.front(),rows.back());
  std::vector<float> coordinates;
  check(spectrumCoordinates(reordered,CarrierKind::points3D,{},field->operations.back(),
      field->spectrumFields[0],coordinates,error)
      && coordinates==std::vector<float>{1,1.0f/3,2.0f/3,0},
      "static point rank samples follow stable identities after record reorder");
  for (const auto identity : {StableId{0},kMaximumStableId+1,rows.front().stableId}) {
    auto bad=*points;
    std::get<PointsData>(bad.data).points[1].stableId=identity;
    check(!lowerAudioSpectrumField(bad,9642,spectrumDomainBinding(),settings,contract,error),
        "static point ranks reject zero, out-of-range and duplicate stable identities");
  }
}

void spectrumInstancerAppearanceComposition() {
  std::string error;
  const auto source=spectrumInstancerLayoutFixture(5,error);
  check(source.has_value(),"appearance composition retains spectrum mesh placement");
  if (!source) return;
  const auto scalarContract=constructedFieldContract(Domain::instance,ValueType::floatValue);
  const auto colorContract=constructedFieldContract(Domain::instance,ValueType::color);
  const auto booleanContract=constructedFieldContract(Domain::instance,ValueType::boolean);
  const auto settings=spectrumDomainSettings(Domain::instance,ValueType::floatValue,7);
  auto binding=spectrumDomainBinding(); binding.direction={0.2f,1,0.1f};
  const auto scalar=lowerAudioSpectrumField(*source,9830,binding,settings,scalarContract,error);
  auto colorSettings=settings; colorSettings.valueType=ValueType::color;
  const auto color=lowerAudioSpectrumField(*source,9831,binding,colorSettings,colorContract,error);
  check(scalar && color,"Spectrum Instancer supplies exact typed instance fields");
  if (!scalar || !color) return;
  FieldOperationSettings math; math.domain=Domain::instance; math.valueType=ValueType::floatValue;
  math.mode=2; math.values[0]=0.5;
  const auto scaled=lowerConstructedField(9832,OperationCode::fieldMath,math,&*scalar,nullptr,nullptr,scalarContract,error);
  math.valueType=ValueType::boolean; math.mode=2; math.values[0]=0.25;
  const auto selection=lowerConstructedField(9833,OperationCode::fieldCompare,math,&*scalar,nullptr,nullptr,booleanContract,error);
  FieldOperationSettings mix; mix.domain=Domain::instance; mix.valueType=ValueType::color;
  mix.values={0.1,0.2,0.3,1,0.25,0,0,0};
  const auto mixed=lowerConstructedField(9834,OperationCode::fieldMix,mix,&*color,nullptr,nullptr,colorContract,error);
  check(scaled && selection && mixed,"sampled fields retain deterministic math, color mix and selection");
  if (!scaled || !selection || !mixed) return;
  auto terminal=*source;
  StableId id=9840;
  for (const auto* property : {"color","emission","opacity","metallic","roughness"}) {
    const auto& field=std::string(property)=="color" || std::string(property)=="emission" ? *mixed : *scaled;
    const auto next=lowerInstanceAppearance(terminal,field,&*selection,id++,property,appearanceInstanceContract(),error);
    check(next.has_value(),"all existing appearance endpoints accept composed instancer fields");
    if (!next) return;
    terminal=*next;
  }
  const auto contract=withAttributeContract(appearanceInstanceContract(),terminal);
  const auto admitted=admitValue(terminal,contract,error);
  check(admitted.has_value(),"combined appearance closure admits through existing authority");
  if (!admitted) return;
  check(terminal.spectrumFields.size()==3 && terminal.spectrumFields[0].target==spectrum::Target::instanceTransform
      && terminal.spectrumFields[1].sourceGeometryStableId==source->stableId
      && terminal.spectrumFields[2].sourceGeometryStableId==source->stableId,
      "transform and both sampled fields retain exact independent bindings");
  const auto bytes=encodeRuntimeValue(*admitted);
  const auto decoded=decodeRuntimeValue(bytes,contract,{}, {},error);
  check(decoded && encodeRuntimeValue(*decoded)==bytes,"v13 composed instance appearance roundtrips byte exactly");
  RuntimeFieldEvaluation runtime;
  runtime.spectrumAt=[](const auto&,auto& bands,std::string&) { bands=spectrumDomainBands(); return true; };
  ValueDescriptor first,changed,sought;
  MaterializedInstancePlan materialized;
  const bool firstReplayed=validateOperationPlan(terminal,{},&materialized,error,&first,&runtime);
  check(firstReplayed,"appearance replays the exact requested bands");
  if (!firstReplayed) return;
  check(materialized.materialTables.count(terminal.stableId)==1
      && materialized.materialTables.at(terminal.stableId).count(0)==1
      && materialized.materialTables.at(terminal.stableId).at(0).secondaryInputStableId==9803,
      "appearance retains the authored material table and its original material identity");
  runtime.spectrumAt=[](const auto&,auto& bands,std::string&) { bands=spectrumDomainBands(true); return true; };
  const bool changedReplayed=validateOperationPlan(terminal,{},nullptr,error,&changed,&runtime);
  check(changedReplayed,"changed bands replay appearance");
  if (!changedReplayed) return;
  runtime.timelineSeconds=1;
  runtime.spectrumAt=[](const auto&,auto& bands,std::string&) { bands=spectrumDomainBands(); return true; };
  check(validateOperationPlan(terminal,{},nullptr,error,&sought,&runtime)
      && equalAttributes(first.attributes,sought.attributes),"seeking with identical bands preserves appearance");
  check(!equalAttributes(first.attributes,changed.attributes),"selected-band change causally changes appearance");
  const auto* emission=findNamedAttribute(first.attributes,"emission");
  check(emission && emission->elements.size()==4 && emission->elements[0].components[0]==0
      && std::abs(emission->elements[1].components[1]-0.275)<1e-6,
      "independent math/mix/selection oracle preserves silent mask and exact second-band emission");
  const auto& instances=std::get<InstancesData>(first.data).instances;
  check(instances.size()==4 && instances[0].stableId==3 && instances[3].stableId==6
      && instances[0].sourceStableId==9802,
      "appearance preserves absolute band IDs and the retained material-table prototype");
  RuntimeFieldEvaluation missing;
  check(!validateOperationPlan(terminal,{},nullptr,error,nullptr,&missing),"missing frame analysis refuses appearance replay");
  auto hostile=terminal; hostile.spectrumFields[1].sourceGeometryStableId=9810;
  check(!admitValue(hostile,contract,error),"forged sampled instance source fails admission");
  hostile=terminal; hostile.spectrumFields[1].coordinates.pop_back();
  check(!admitValue(hostile,contract,error),"noncanonical sampled cardinality fails admission");
  hostile=terminal; std::get<InstancesData>(hostile.data).instances[0].stableId=30;
  check(!admitValue(hostile,contract,error),"forged absolute band identity fails admission");
  const auto foreignPrototype=lowerMeshGenerator(9850,9850,1,OperationCode::plane,{0.5f,0.5f,0,0},
      constructedFieldMeshContract(),error);
  const auto foreign=foreignPrototype ? lowerSpectrumInstancer(*foreignPrototype,9851,binding,{0,2,2,2},
      appearanceInstanceContract(),error) : std::nullopt;
  const auto foreignField=foreign ? lowerAudioSpectrumField(*foreign,9852,binding,settings,scalarContract,error) : std::nullopt;
  check(foreignField && !lowerInstanceAppearance(*source,*foreignField,nullptr,9853,"opacity",appearanceInstanceContract(),error),
      "equal-count foreign Spectrum Instancer cannot supply instance appearance");
  ResourceLimits limits; limits.maxBufferBytes=1;
  check(!admitValue(terminal,contract,limits,error),"composed appearance retains hard memory admission");
  hostile=terminal; hostile.spectrumFields[1].history.enabled=true;
  check(!admitValue(hostile,contract,error),"unsupported sampled history refuses admission");
  auto positionSettings=settings; positionSettings.mode=1;
  check(!lowerAudioSpectrumField(*source,9854,binding,positionSettings,scalarContract,error),
      "post-draw instance positions cannot masquerade as ordered spectrum sampling");
  FieldOperationSettings timeline; timeline.domain=Domain::instance; timeline.valueType=ValueType::floatValue;
  timeline.values={1,0,0,0,1,0,0,0};
  check(!lowerConstructedField(9855,OperationCode::fieldTimeline,timeline,&*scalar,nullptr,nullptr,scalarContract,error),
      "sampled instancer composition does not admit unrelated runtime routes");
}

void scoreSampleFieldMeasures() {
  std::string error;
  const auto score=scoreFieldMeasureScore();
  canonicalblockc::FrameProducer producer;
  const auto frame=producer.evaluate(scoreFieldMeasureKey(),score,2);
  check(canonicalblockc::valid(frame),"Score Sample Field measurement fixture has a canonical frame");
  if (!canonicalblockc::valid(frame)) return;
  const auto positions=scoreSampleMeasurePositions();
  // Expected measures are written out here, not read back from the helper table.
  const std::array<scorefield::Mode,14> expected {
    scorefield::Mode::closestNoteDistance, scorefield::Mode::velocityInfluence, scorefield::Mode::pitchSpacing,
    scorefield::Mode::harmonicLinkDistance, scorefield::Mode::pitchBendSemitones, scorefield::Mode::noteAgeBeats,
    scorefield::Mode::remainingDurationBeats, scorefield::Mode::rootFlag, scorefield::Mode::midiNoteNumber,
    scorefield::Mode::resolvedFrequencyHz, scorefield::Mode::harmonicLinkRatio,
    scorefield::Mode::primeLattice, scorefield::Mode::primeLattice, scorefield::Mode::primeLattice
  };
  static_assert(std::tuple_size<decltype(expected)>::value==scorefield::kSampleFieldModeCount,
                "expected measures must list every Score Sample Field choice");
  for (std::uint8_t choice=0;choice<expected.size();++choice) {
    scorefield::Mode measure {};
    check(scorefield::sampleFieldMode(choice,measure) && measure==expected[choice],
          "Score Sample Field choice maps to its released or appended measure");
    const auto op=scoreSampleMeasureOperation(choice);
    // The sampled values must equal the canonical Score Field evaluator for the same measure.
    // Prime Lattice choices compare against the vector component on their own axis.
    const std::size_t component=choice>=scorefield::kFirstSamplePrimeLatticeChoice
        ? choice-scorefield::kFirstSamplePrimeLatticeChoice : 0;
    scorefield::Binding binding;
    binding.fieldStableId=op.stableId; binding.sourceGeometryStableId=op.inputStableId;
    binding.scoreSourceStableId=9903; binding.mode=measure;
    binding.radius=4; binding.gain=1; binding.bias=0; binding.direction={0,0,0};
    binding.direction[component]=1;
    for (std::size_t index=0;index<positions.elements.size();++index) {
      const auto& c=positions.elements[index].components;
      binding.positions.push_back({static_cast<float>(c[0]),static_cast<float>(c[1]),static_cast<float>(c[2])});
      binding.elementIds.push_back(index+1);
    }
    FieldData sampled;
    check(videohelper::scorefield::evaluateSampleField(op,positions,frame,sampled,error),
          "Score Sample Field evaluates every scalar measurement");
    videohelper::scorefield::Evaluation direct;
    check(videohelper::scorefield::evaluate(binding,frame,direct,error),
          "canonical Score Field evaluates the same scalar measurement");
    check(direct.offsets.size()==sampled.elements.size(),
          "Score Sample Field keeps one value per bounded position");
    for (std::size_t index=0;index<sampled.elements.size() && index<direct.offsets.size();++index)
      check(sampled.elements[index].components[0]==direct.offsets[index][component],
            "Score Sample Field returns the canonical Score Field measurement");
  }
  FieldData refused;
  check(!videohelper::scorefield::evaluateSampleField(scoreSampleMeasureOperation(scorefield::kSampleFieldModeCount),
                                                      positions,frame,refused,error),
        "Score Sample Field refuses choices beyond its scalar measurements");

  // Transport keeps each choice, and replay refuses an out-of-range choice.
  auto value=timelineCurveFixture(error);
  check(value.has_value(),error.c_str()); if (!value) return;
  for (std::uint8_t choice=0;choice<expected.size();++choice) {
    auto variant=*value;
    std::size_t changed=0;
    for (auto& operation : variant.operations)
      if (operation.code==OperationCode::fieldScoreSample && operation.stableId==9707) {
        operation.field.mode=choice; ++changed;
      }
    check(changed==1,"timeline score sample fixture exposes one measured sample operation");
    roundtrip(variant,timelineContract(CarrierKind::geometry3D),{},
        "every Score Sample Field choice retains its measurement through transport");
  }
  auto invalid=*value;
  std::size_t changed=0;
  for (auto& operation : invalid.operations)
    if (operation.code==OperationCode::fieldScoreSample && operation.stableId==9707) {
      operation.field.mode=scorefield::kSampleFieldModeCount; ++changed;
    }
  check(changed==1,"timeline score sample fixture exposes one measured sample operation");
  auto liveFrame=scoreFieldFrame();
  RuntimeFieldEvaluation evaluation;
  evaluation.timelineSeconds=1;
  evaluation.scoreAt=[&](const auto& operation,const auto& samples,auto& output,auto& diagnostic) {
    return videohelper::scorefield::evaluateSampleField(operation,samples,liveFrame,output,diagnostic);
  };
  MaterializedInstancePlan refusedPlan;
  check(!validateOperationPlan(invalid,{},&refusedPlan,error,nullptr,&evaluation),
        "replay refuses a Score Sample Field choice outside its measurements");
}

void legacyScoreFieldWireGolden() {
  std::string error;
  ValueDescriptor value; value.carrier=CarrierKind::field; value.stableId=value.sourceStableId=9960; value.sourceRevision=1;
  value.fieldDomain=Domain::vertex; value.fieldValueType=ValueType::vector; value.fieldInterpolation=Interpolation::constant;
  value.fieldDomainCardinality=1; value.data=FieldData{{FieldElement{}}};
  value.operations={{9960,OperationCode::evaluateField,9960,9960,{}}}; value.dispatchCount=1;
  scorefield::Binding binding; binding.fieldStableId=9960; binding.sourceGeometryStableId=9961; binding.scoreSourceStableId=9962;
  binding.positions={{{0,0,0}}}; binding.elementIds={9}; value.scoreFields={binding};
  const auto contract=constructedFieldContract(Domain::vertex,ValueType::vector);
  const auto admitted=admitValue(value,contract,error);
  check(admitted.has_value(),"released vertex Score Field remains admitted");
  if (!admitted) return;
  // Fixed v13 header, legacy evaluateField, Vector payload and score binding.
  const std::string goldenHex=
      "474352560d0000000200000005e826000000000000e826000000000000010000000000000001020201010000"
      "0000000000010000000000000000000000000000000100000000000000e82600000000000003e82600000000"
      "0000e82600000000000000000000000000000000000000000000000000000000000001000000000000000000"
      "0000000000000000000000000000000000000000000000000000000000000000000001000000e82600000000"
      "0000e9260000000000000000000000000000ea26000000000000010001040000803f0000803f0000803f0000"
      "803f0000803f0000000000000000000000000000803f00010201000000000000000000000000000000000000"
      "00090000000000000000000000";
  std::vector<std::uint8_t> golden;
  const auto nibble=[](char c) { return c>='a' ? c-'a'+10 : c-'0'; };
  for (std::size_t i=0;i<goldenHex.size();i+=2)
    golden.push_back(static_cast<std::uint8_t>(16*nibble(goldenHex[i])+nibble(goldenHex[i+1])));
  check(encodeRuntimeValue(*admitted)==golden,"legacy vertex Score Field keeps exact released v13 bytes");
  const auto reopened=decodeRuntimeValue(golden,contract,{}, {},error);
  check(reopened && encodeRuntimeValue(*reopened)==golden,"frozen vertex Score Field bytes retain decoder identity");
  ResourceLimits limits; limits.maxVertices=0; limits.maxInstances=64;
  // The legacy samples belong to vertices even when the instance allowance is larger.
  check(!decodeRuntimeValue(golden,contract,limits,{},error),"legacy vertex samples cannot borrow the instance allowance");
}

void instanceScoreFieldsRetainCanonicalOwnership() {
  std::string error;
  const auto source=instanceAppearanceFixture(0,error);
  check(source.has_value(),"instance Score Field retains an existing static instancer");
  if (!source) return;
  for (unsigned mode=0;mode<12;++mode) {
    const auto terminal=instanceScoreFieldFixture(mode,"emission",error);
    check(terminal.has_value(),"all canonical Score Field measurements lower on instance records");
    if (!terminal) continue;
    const auto contract=withAttributeContract(appearanceInstanceContract(),*terminal);
    const auto admitted=admitValue(*terminal,contract,error);
    check(admitted.has_value(),"score instance composition admits through the existing geometry authority");
    if (!admitted) continue;
    const auto bytes=encodeRuntimeValue(*admitted);
    const auto reopened=decodeRuntimeValue(bytes,contract,{}, {},error);
    check(reopened && encodeRuntimeValue(*reopened)==bytes,"v13 instance Score Field binding bytes roundtrip exactly");
    const auto& binding=terminal->scoreFields.front();
    check(binding.sourceGeometryStableId==source->stableId && binding.consumerGeometryStableId==0
        && binding.elementIds==std::vector<StableId>{1,2,3,4},"Score Field binds exact ordered instance IDs");
    const auto& instances=std::get<InstancesData>(source->data).instances;
    for (std::size_t i=0;i<instances.size();++i) {
      const auto& p=instances[i].transform.translation;
      check(binding.positions[i]==std::array<float,3>{p.x,p.y,p.z},"score samples are the exact InstanceRecord translations");
    }
    auto frame=scoreFieldFrame();
    videohelper::scorefield::Evaluation values;
    check(videohelper::scorefield::evaluate(binding,frame,values,error),"existing full canonical score evaluator accepts instance samples");
    check(values.closestNoteIds==std::vector<std::int64_t>{-1,-1,-1,-1},"canonical signed note IDs never become voice slots");
    if (mode==4 || mode==11) check(values.closestLinkIds==std::vector<std::int64_t>{-301,-301,-301,-301} && !values.linkRatios.empty()
        && values.linkRatios.front()==std::array<float,2>{2,1},"harmonic link mode preserves canonical IDs and ratios");
    if (mode==1 && !values.offsets.empty()) {
      const double distance=std::sqrt(0.6*0.6+0.6*0.6);
      const double expected=(1-distance/2)*0.4;
      check(std::abs(values.offsets[0][0]-expected)<1e-6
          && std::abs(values.offsets[0][1]-expected*0.5)<1e-6,
          "independent note-distance and velocity oracle predicts the instance Vector result");
    }
    RuntimeFieldEvaluation runtime;
    runtime.scoreAt=[&](const auto& op,const auto& positions,auto& out,auto& diagnostic) {
      const auto b=std::find_if(terminal->scoreFields.begin(),terminal->scoreFields.end(),[&](const auto& x) { return x.fieldStableId==op.stableId; });
      return b!=terminal->scoreFields.end() && videohelper::scorefield::evaluateInstanceField(*b,positions,frame,out,diagnostic);
    };
    ValueDescriptor first,exported,sought;
    const bool firstOk=validateOperationPlan(*terminal,{},nullptr,error,&first,&runtime);
    check(firstOk,"ordered instance Score Field replays before appearance composition");
    if (!firstOk) continue;
    frame=scoreFieldFrame(true); runtime.timelineSeconds=8;
    check(validateOperationPlan(*terminal,{},nullptr,error,&exported,&runtime)
        && equalAttributes(first.attributes,exported.attributes),"canonical row reordering preserves same-frame appearance");
    frame=scoreFieldFrame(); runtime.timelineSeconds=0;
    check(validateOperationPlan(*terminal,{},nullptr,error,&sought,&runtime)
        && equalAttributes(first.attributes,sought.attributes)
        && equal(std::get<InstancesData>(first.data),std::get<InstancesData>(sought.data)),
        "seek/reset reevaluates immutable score samples without prior-frame state");
    if (mode==1) {
      frame=scoreFieldFrame(false,false,true,32); ValueDescriptor changed;
      check(validateOperationPlan(*terminal,{},nullptr,error,&changed,&runtime)
          && !equalAttributes(first.attributes,changed.attributes),"canonical note velocity causally changes composed instance emission");
    }
    frame={};
    check(!validateOperationPlan(*terminal,{},nullptr,error,nullptr,&runtime),"missing canonical frame refuses instance Score Field replay");
    frame=scoreFieldFrame(false,true);
    check(!validateOperationPlan(*terminal,{},nullptr,error,nullptr,&runtime),"empty score windows refuse instead of retaining previous appearance");
    auto hostile=*terminal; hostile.scoreFields[0].elementIds[0]=123;
    check(!admitValue(hostile,contract,error),"forged ordered instance IDs refuse");
    hostile=*terminal; hostile.scoreFields[0].positions[0][0]+=0.1f;
    check(!admitValue(hostile,contract,error),"stale instance position samples refuse");
    hostile=*terminal; hostile.scoreFields[0].sourceGeometryStableId=9101;
    check(!admitValue(hostile,contract,error),"wrong source carrier and identity refuse");
    hostile=*terminal; hostile.scoreFields[0].consumerGeometryStableId=9920;
    check(!admitValue(hostile,contract,error),"ordered score samples cannot forge a legacy deformation consumer");
    ResourceLimits limits; limits.maxInstances=3;
    check(!admitValue(*terminal,contract,limits,error),"instance score samples obey a reduced instance budget");
    limits=ResourceLimits{}; limits.maxVertices=3;
    check(!decodeRuntimeValue(bytes,contract,limits,{},error),"larger instance budget cannot hide an oversized retained prototype");
  }
  scorefield::Binding binding; binding.scoreSourceStableId=9950;
  const auto vectorContract=constructedFieldContract(Domain::instance,ValueType::vector);
  const auto field=lowerInstanceScoreField(*source,9951,binding,vectorContract,error);
  check(field.has_value(),"standalone typed instance Score Field is admitted");
  if (!field) return;
  auto forgedSource=*source;
  std::get<InstancesData>(forgedSource.data).instances[0].sourceStableId=9101;
  check(!lowerInstanceScoreField(forgedSource,9952,binding,vectorContract,error),"forged instance prototype ownership refuses before sampling");
  auto foreign=*source; foreign.stableId=9953;
  foreign.operations.push_back({9953,OperationCode::transformInstances,source->stableId,9953,{0,3,1,0}});
  foreign.dispatchCount=foreign.operations.size();
  const auto foreignField=lowerInstanceScoreField(foreign,9954,binding,vectorContract,error);
  FieldOperationSettings measure; measure.domain=Domain::instance; measure.valueType=ValueType::floatValue; measure.mode=3;
  const auto scalar=foreignField ? lowerConstructedField(9955,OperationCode::fieldVectorMeasure,measure,&*foreignField,nullptr,nullptr,
      constructedFieldContract(Domain::instance,ValueType::floatValue),error) : std::nullopt;
  check(scalar && !lowerInstanceAppearance(*source,*scalar,nullptr,9956,"opacity",appearanceInstanceContract(),error),
      "equal-count foreign instance descriptors cannot supply score appearance");
  ResourceLimits limits; limits.maxVertices=4; limits.maxInstances=3;
  const auto admitted=admitValue(*field,vectorContract,error);
  check(admitted && !decodeRuntimeValue(encodeRuntimeValue(*admitted),vectorContract,limits,{},error),
      "instance field transport cannot borrow a larger vertex budget");
  scorefield::Binding oversized=binding;
  oversized.fieldStableId=9957; oversized.sourceGeometryStableId=source->stableId;
  oversized.positions.resize(2049); oversized.elementIds.resize(2049);
  for (std::size_t i=0;i<oversized.elementIds.size();++i) oversized.elementIds[i]=i+1;
  check(!scorefield::validate(oversized,2049,error),"instance Score Field retains the 2048-sample hard cap");
  const auto dynamic=spectrumInstancerLayoutFixture(5,error);
  check(dynamic && lowerInstanceScoreField(*dynamic,9958,binding,vectorContract,error),
      "fixed Spectrum placements retain authored Score Field anchors for ordered frame replay");
}

void fixedDynamicInstanceScoreFields() {
  std::string error;
  for (unsigned layout : {3u,4u,5u}) {
    const auto source=spectrumInstancerLayoutFixture(layout,error);
    const auto terminal=source ? fixedPlacementScoreFixture(*source,9980,error) : std::nullopt;
    check(terminal.has_value(),"fixed Spectrum placement layouts lower before instance Score Fields");
    if (!terminal) continue;
    const auto contract=withAttributeContract(appearanceInstanceContract(),*terminal);
    const auto admitted=admitValue(*terminal,contract,error);
    check(admitted.has_value(),"dynamic instance Score Field retains exact authored admission");
    if (!admitted) continue;
    const auto bytes=encodeRuntimeValue(*admitted);
    const auto reopened=decodeRuntimeValue(bytes,contract,{}, {},error);
    check(reopened && encodeRuntimeValue(*reopened)==bytes,"dynamic placements preserve exact existing v13 wire roundtrip");
    auto bands=spectrumDomainBands(); auto frame=instanceScoreFrameAt(195);
    unsigned spectrumCalls=0; FieldData sampled;
    RuntimeFieldEvaluation runtime;
    runtime.timelineSeconds=3.25;
    runtime.spectrumAt=[&](const auto& binding,auto& out,std::string& diagnostic) {
      if (!spectrum::sameBinding(binding,terminal->spectrumFields.front())) {
        diagnostic="Wrong Spectrum placement owner"; return false;
      }
      ++spectrumCalls; out=bands; return true;
    };
    runtime.scoreAt=[&](const auto& op,const auto& positions,auto& out,auto& diagnostic) {
      if (op.stableId!=terminal->scoreFields.front().fieldStableId) return false;
      sampled=positions;
      return videohelper::scorefield::evaluateInstanceField(terminal->scoreFields.front(),positions,frame,out,diagnostic,true);
    };
    ValueDescriptor first,exported,changed,sought;
    const bool ok=validateOperationPlan(*terminal,{},nullptr,error,&first,&runtime);
    check(ok,"current followed Spectrum placements reach the canonical Score Field callback");
    if (!ok) continue;
    check(spectrumCalls==1,"ordered placement applies and samples its exact Spectrum binding once");
    const auto& base=std::get<InstancesData>(source->data).instances;
    const auto& current=std::get<InstancesData>(first.data).instances;
    const auto& transformBinding=terminal->spectrumFields.front();
    const auto* opacity=findNamedAttribute(first.attributes,"opacity");
    check(opacity && opacity->elements.size()==base.size(),"current score influence reaches existing per-instance opacity");
    for (std::size_t i=0;i<base.size();++i) {
      // With default gain/bias and no interpolation/following, the selected
      // absolute bin is the amount. This oracle does not call spectrum::sample.
      const float amount=bands[transformBinding.firstBand+i];
      const auto p=base[i].transform.translation;
      const std::array<float,3> expected{p.x+amount*transformBinding.direction[0],
          p.y+amount*transformBinding.direction[1],p.z+amount*transformBinding.direction[2]};
      for (std::size_t axis=0;axis<3;++axis)
        check(std::abs(sampled.elements[i].components[axis]-expected[axis])<1e-7,
            "Score Field sees the independently predicted single Spectrum translation");
      check(current[i].stableId==base[i].stableId && current[i].sourceStableId==base[i].sourceStableId
          && current[i].transform.translation.x==expected[0] && current[i].transform.translation.y==expected[1],
          "dynamic appearance retains exact ordered element/prototype identity and one transform application");
      const double distance=std::sqrt(double(expected[0])*expected[0]+double(expected[1])*expected[1]+double(expected[2])*expected[2]);
      if (opacity) check(std::abs(opacity->elements[i].components[0]-(distance*0.2+0.1))<1e-6,
          "independent current-position distance predicts composed canonical score opacity");
    }
    check(first.scoreFields.front().positions==terminal->scoreFields.front().positions
        && encodeRuntimeValue(*admitted)==bytes,"frame evaluation leaves immutable authored ownership anchors unchanged");
    frame=instanceScoreFrameAt(195,true);
    const bool exportedOk=reopened && validateOperationPlan(reopened->descriptor(),{},nullptr,error,&exported,&runtime);
    check(exportedOk,"reopened export descriptor replays current placements");
    if (exportedOk) check(equal(std::get<InstancesData>(first.data),std::get<InstancesData>(exported.data))
        && equalAttributes(first.attributes,exported.attributes),"equal-time reordered canonical score preserves preview/export placements and appearance");
    bands=spectrumDomainBands(true); frame=instanceScoreFrameAt(195);
    check(validateOperationPlan(*terminal,{},nullptr,error,&changed,&runtime)
        && !equalAttributes(first.attributes,changed.attributes),"changed Spectrum placement causally changes score appearance");
    bands=spectrumDomainBands(); runtime.timelineSeconds=0; frame=instanceScoreFrameAt(0);
    check(validateOperationPlan(*terminal,{},nullptr,error,&sought,&runtime)
        && equal(std::get<InstancesData>(first.data),std::get<InstancesData>(sought.data))
        && equalAttributes(first.attributes,sought.attributes),"direct replay and rewind/reset retain the same exact samples");
    auto hostile=*terminal; hostile.scoreFields[0].positions[0][0]+=0.1f;
    check(!admitValue(hostile,contract,error),"dynamic admission still refuses stale authored anchor positions");
    hostile=*terminal; hostile.spectrumFields[0].sourceGeometryStableId=9810;
    check(!admitValue(hostile,contract,error),"wrong retained Spectrum prototype owner refuses");
    hostile=*terminal; hostile.operations.push_back(hostile.operations.front()); ++hostile.dispatchCount;
    check(!admitValue(hostile,contract,error),"repeated operation identity cannot apply a placement twice");
    hostile=*terminal; hostile.spectrumFields.push_back(hostile.spectrumFields.front());
    check(!admitValue(hostile,contract,error),"ambiguous repeated Spectrum binding refuses admission");
    RuntimeFieldEvaluation missing=runtime; missing.spectrumAt={};
    check(!validateOperationPlan(*terminal,{},nullptr,error,nullptr,&missing),"missing current Spectrum placement samples refuse");
    missing.spectrumAt=[](const auto&,auto&,std::string& diagnostic) {
      diagnostic="Stale selected-source frame provider"; return false;
    };
    check(!validateOperationPlan(*terminal,{},nullptr,error,nullptr,&missing),"a stale selected-source provider cannot reuse previous placement samples");
    frame={}; check(!validateOperationPlan(*terminal,{},nullptr,error,nullptr,&runtime),"missing canonical score refuses after exact placement replay");
    auto foreign=*source; foreign.stableId=9990;
    foreign.spectrumFields[0].fieldStableId=foreign.spectrumFields[0].consumerGeometryStableId=9990;
    for (auto& op : foreign.operations) if (op.code==OperationCode::spectrumInstancer) op.stableId=op.outputStableId=9990;
    scorefield::Binding binding; binding.scoreSourceStableId=9910;
    const auto field=lowerInstanceScoreField(foreign,9991,binding,constructedFieldContract(Domain::instance,ValueType::vector),error);
    FieldOperationSettings measure; measure.domain=Domain::instance; measure.valueType=ValueType::floatValue; measure.mode=3;
    const auto scalar=field ? lowerConstructedField(9992,OperationCode::fieldVectorMeasure,measure,&*field,nullptr,nullptr,
        constructedFieldContract(Domain::instance,ValueType::floatValue),error) : std::nullopt;
    check(scalar && !lowerInstanceAppearance(*source,*scalar,nullptr,9993,"opacity",appearanceInstanceContract(),error),
        "equal-content Spectrum placements from a different descriptor owner cannot supply appearance");
  }
  const auto animated=animatedScorePlacementFixture(error);
  check(animated.has_value(),"retained animation placement lowers with fixed ordered score anchors");
  if (!animated) return;
  auto current=*animated;
  for (auto& op : current.operations) if (op.retainedMesh) {
    auto frame=std::make_shared<RetainedMeshData>(*op.retainedMesh);
    for (auto& p : frame->geometry.positions) p.x+=0.3f;
    op.retainedMesh=std::move(frame);
  }
  const auto canonical=scoreFieldFrame(); FieldData sampled;
  RuntimeFieldEvaluation runtime;
  runtime.scoreAt=[&](const auto&,const auto& positions,auto& out,auto& diagnostic) {
    sampled=positions;
    return videohelper::scorefield::evaluateInstanceField(animated->scoreFields.front(),positions,canonical,out,diagnostic,true);
  };
  ValueDescriptor base,moved,repeated;
  check(validateOperationPlan(*animated,{},nullptr,error,&base,&runtime)
      && validateOperationPlan(current,{},nullptr,error,&moved,&runtime)
      && !equalAttributes(base.attributes,moved.attributes),"same-topology current animation positions causally change Score Field appearance");
  check(sampled.elements.size()==animated->scoreFields.front().positions.size(),"animation preserves exact sampled cardinality");
  for (std::size_t i=0;i<sampled.elements.size();++i)
    check(std::abs(sampled.elements[i].components[0]-(animated->scoreFields.front().positions[i][0]+0.3f))<1e-7,
        "animation score sampling uses the exact current retained translation");
  FieldData strict;
  check(!videohelper::scorefield::evaluateInstanceField(animated->scoreFields.front(),sampled,canonical,strict,error),
      "the original static evaluator still refuses positions that differ from saved samples");
  check(validateOperationPlan(*animated,{},nullptr,error,&repeated,&runtime)
      && equalAttributes(base.attributes,repeated.attributes),"animation rewind replays authored anchors without latching current samples");
  auto hostile=current; hostile.scoreFields[0].elementIds[0]=123;
  check(!validateOperationPlan(hostile,{},nullptr,error,nullptr,&runtime),"changed animation element IDs refuse before score evaluation");
  hostile=current; hostile.scoreFields[0].positions.pop_back(); hostile.scoreFields[0].elementIds.pop_back();
  check(!validateOperationPlan(hostile,{},nullptr,error,nullptr,&runtime),"changed dynamic sampled cardinality refuses");
  const auto particle=particlePointsInstanceFixture(error); scorefield::Binding binding; binding.scoreSourceStableId=9910;
  check(particle && !lowerInstanceScoreField(*particle,9995,binding,constructedFieldContract(Domain::instance,ValueType::vector),error),
      "deferred Particle Points retain refusal until source-owned field admission exists");
}

void scoreDeformedPrototypeInstancesRetainCanonicalFrame() {
  std::string error;
  auto source=lowerGrid(12001,12001,1,2,2,0.4f,contractFor(CarrierKind::geometry3D),error);
  const auto placement=lowerGrid(12010,12010,1,2,2,3,contractFor(CarrierKind::geometry3D),error);
  check(source && placement,"Score prototype and placement grids lower");
  if (!source || !placement) return;
  Transform authored; authored.translation={0.25f,-0.15f,0}; authored.scale={1.5f,0.75f,1};
  authored.rotation={0,0,0.7071067811865475f,0.7071067811865475f};
  source=lowerTransformGeometry(*source,12002,authored,contractFor(CarrierKind::geometry3D),error);
  check(source.has_value(),"authored prototype transform lowers before Score sampling");
  if (!source) return;
  scorefield::Binding binding;
  binding.fieldStableId=12003; binding.sourceGeometryStableId=12002;
  binding.consumerGeometryStableId=12004; binding.scoreSourceStableId=12005;
  binding.mapping.xScale=binding.mapping.yScale=binding.mapping.zScale=0;
  binding.radius=4; binding.gain=0.3f; binding.direction={0.5f,0.25f,1};
  const auto mesh=std::get<GeometryData>(source->data);
  binding.elementIds=mesh.vertexIds;
  for (const auto& p : mesh.positions) binding.positions.push_back({p.x,p.y,p.z});
  source->scoreFields={binding};
  source->operations.push_back({12003,OperationCode::evaluateField,12003,12003,{}});
  ++source->dispatchCount;
  source=lowerTransformGeometry(*source,12004,{},contractFor(CarrierKind::geometry3D),error);
  source=source ? lowerAssignMaterial(*source,12006,19,contractFor(CarrierKind::geometry3D),error) : std::nullopt;
  const auto points=lowerPointsFromVertices(*placement,12011,contractFor(CarrierKind::points3D),error);
  const auto value=source && points ? lowerInstanceOnPoints(*points,*source,12012,contractFor(CarrierKind::instances3D),error) : std::nullopt;
  check(value.has_value(),"one Score-deformed prototype composes through released Instance on Points");
  if (!value) { check(false,error.c_str()); return; }
  const auto contract=contractFor(CarrierKind::instances3D);
  const auto admitted=admitValue(*value,contract,error);
  check(admitted.has_value(),"Score prototype retains bounded instance admission");
  if (!admitted) return;
  const auto bytes=encodeRuntimeValue(*admitted);
  const auto reopened=decodeRuntimeValue(bytes,contract,{}, {},error);
  check(reopened && encodeRuntimeValue(*reopened)==bytes,"Score prototype instance bytes reopen exactly without frame baking");
  TestBackend backend; videohelper::geometry::GeometryCorePlanRuntime plans(backend);
  const auto planBytes=lowerRuntimePlan(contract,*admitted);
  const auto preview=plans.admitPreview({1,1,1,12012,1,videohelper::geometry::PlanUse::preview},planBytes,{}, {},error);
  const auto exported=plans.admitExport({1,1,1,12012,1,videohelper::geometry::PlanUse::exportRender},planBytes,{}, {},error);
  check(preview && exported && preview->owner.use!=exported->owner.use
      && encodeRuntimeValue(preview->plan->value())==encodeRuntimeValue(exported->plan->value()),
      "Preview and Export keep distinct owners with the same immutable Score prototype bytes");
  auto frame=scoreFieldFrame(); const canonicalblockc::CanonicalBlockCFrame* consumed=nullptr;
  RuntimeFieldEvaluation runtime;
  runtime.scoreAt=[&](const auto& op,const auto& positions,auto& out,auto& diagnostic) {
    consumed=frame.get();
    if (op.stableId!=binding.fieldStableId) return false;
    return videohelper::scorefield::evaluateInstanceField(binding,positions,frame,out,diagnostic);
  };
  MaterializedInstancePlan first,soft,repeated;
  check(validateOperationPlan(*value,{},&first,error,nullptr,&runtime),"canonical vertex Score replay precedes retained instancing");
  if (!first.geometries.count(12006)) return;
  check(consumed==frame.get(),"Score prototype callback consumes the exact supplied canonical frame owner");
  const auto& drawn=first.geometries.at(12006);
  const auto& original=std::get<GeometryData>(placement->data);
  check(first.instances.instances.size()==original.positions.size(),"Score prototype retains every ordered placement");
  if (first.instances.instances.size()!=original.positions.size()) return;
  check(drawn.indices==mesh.indices && drawn.vertexIds==mesh.vertexIds && first.materialStableId==19,
      "Score deformation keeps prototype topology, ordered vertex IDs and material identity");
  for (std::size_t i=0;i<drawn.positions.size();++i) {
    const float x=(i%2==0 ? -0.2f : 0.2f), y=(i/2==0 ? -0.2f : 0.2f);
    const float px=0.25f-0.75f*y, py=-0.15f+1.5f*x;
    const double amount=0.6*std::max(0.0,1.0-std::sqrt(double(px)*px+double(py)*py)/4.0);
    const auto& p=drawn.positions[i];
    check(std::abs(p.x-(px+amount*0.5))<1e-6 && std::abs(p.y-(py+amount*0.25))<1e-6
        && std::abs(p.z-amount)<1e-6,"independent transformed-vertex and two-note velocity oracle matches the prototype");
    const auto& instance=first.instances.instances[i];
    check(instance.stableId==original.vertexIds[i] && instance.sourceStableId==12006
        && equal(instance.transform.translation,original.positions[i])
        && equal(instance.transform.scale,Vec3{1,1,1}) && equal(instance.transform.rotation,Quaternion{}),
        "point order, full prototype identity and placement transforms survive Score-only variation");
  }
  frame=scoreFieldFrame(false,false,true,32);
  check(validateOperationPlan(*value,{},&soft,error,nullptr,&runtime)
      && !equal(soft.geometries.at(12006),drawn) && equal(soft.instances,first.instances),
      "canonical Score-only velocity variation changes vertices without changing instances");
  frame=scoreFieldFrame(true);
  check(validateOperationPlan(*value,{},&repeated,error,nullptr,&runtime)
      && equal(repeated.geometries.at(12006),drawn),"rewind and canonical row reordering reproduce the same prototype");
  frame.reset(); MaterializedInstancePlan untouched=first;
  check(!validateOperationPlan(*value,{},&untouched,error,nullptr,&runtime)
      && equal(untouched.geometries.at(12006),drawn),"missing canonical score refuses before replacing published geometry");
  frame=scoreFieldFrame(false,true);
  check(!validateOperationPlan(*value,{},nullptr,error,nullptr,&runtime),"empty canonical Score refuses the prototype frame");
  auto hostile=*value; hostile.scoreFields[0].positions.pop_back();
  check(!admitValue(hostile,contract,error),"missing prototype samples fail before publication");
  hostile=*value;
  hostile.scoreFields[0].positions.resize(scorefield::kMaximumElements+1);
  hostile.scoreFields[0].elementIds.resize(scorefield::kMaximumElements+1);
  check(!admitValue(hostile,contract,error),"over-cap prototype Score samples refuse before publication");
  hostile=*value; hostile.scoreFields[0].elementIds[0]=99;
  check(!admitValue(hostile,contract,error),"forged ordered prototype vertex identity refuses admission");
  hostile=*value; hostile.scoreFields[0].consumerGeometryStableId=12010;
  check(!admitValue(hostile,contract,error),"a foreign prototype consumer cannot claim the retained instance source");
  ResourceLimits limits; limits.maxVertices=3;
  check(!admitValue(*value,contract,limits,error),"prototype samples cannot borrow the instance vertex allowance");
  limits={}; limits.maxIndices=5;
  check(!admitValue(*value,contract,limits,error),"prototype topology obeys the original index budget");
  limits={}; limits.maxPoints=3;
  check(!admitValue(*value,contract,limits,error),"retained placements obey the original point budget");
  limits={}; limits.maxInstances=3;
  check(!admitValue(*value,contract,limits,error),"retained placements obey the original instance budget");
  limits={}; limits.maxBufferBytes=1;
  check(!admitValue(*value,contract,limits,error),"Score prototype work and retained buffers obey memory admission");
  RuntimeFieldEvaluation truncated;
  truncated.scoreAt=[](const auto&,const auto&,auto& out,auto&) { out.elements.resize(1); return true; };
  check(!validateOperationPlan(*value,{},&untouched,error,nullptr,&truncated)
      && equal(untouched.geometries.at(12006),drawn),"wrong callback cardinality fails before publication");
  AdmissionContext deferred;
  deferred.deferredPoints.emplace(12011,DeferredPointsSource{});
  check(!validateOperationPlan(*value,deferred,nullptr,error),"deferred Particle Points remain outside Score prototype admission");
}

void timelineSceneFamiliesRetainExactChildrenAndBudgets() {
  using namespace videohelper::geometry;
  using namespace HarmonicMIDI::grid;
  static_assert(!std::is_default_constructible_v<TimelineSceneChildAdmission>);
  static_assert(!std::is_default_constructible_v<AdmittedTimelineScenePlans>);
  std::string error;
  const auto contract=contractFor(CarrierKind::geometry3D);
  const auto makeValue=[&](StableId id,double amplitude) -> std::optional<AdmittedValue> {
    const auto grid=lowerGrid(id,id,1,2,2,0.5f,contract,error);
    const auto water=grid ? lowerWaterSurfaceSimulation(*grid,id+1,id+2,1,2,amplitude,0.4,contract,error) : std::nullopt;
    return water ? admitValue(*water,contract,error) : std::nullopt;
  };
  const auto a=makeValue(15000,0.2),b=makeValue(15100,0.4);
  check(a && b,"family fixtures lower two independently retained timeline plans");
  if (!a || !b) return;
  auto scene=std::make_shared<Visual3DScene>();
  scene->id={7}; scene->activeCamera={50}; scene->cameraCount=1;
  scene->cameras[0].id={50}; scene->cameras[0].transform.translation.z=4;
  SceneObjectRecord first; first.id={10}; SceneMaterialRecord material; material.id={20};
  SceneObjectRecord second; second.id={11}; second.transform.translation.x=2;
  auto secondMaterial=material; secondMaterial.id={21};
  check(appendGeometryToScene(*a,first,material,*scene,error)
      && appendGeometryToScene(*b,second,secondMaterial,*scene,error),"family fixture retains exclusive authored mesh ranges");
  if (!validateVisual3DScene(*scene).valid()) { check(false,"family fixture scene validates"); return; }
  std::vector<TimelineSceneBinding> bindings{{first.id,encodeLoweredPlanText(lowerRuntimePlan(contract,*a))},
      {second.id,encodeLoweredPlanText(lowerRuntimePlan(contract,*b))}};
  const PlanOwnerIdentity owner{10,20,30,15000,1,PlanUse::preview};
  auto next=owner; ++next.planGeneration;
  TestBackend backend;
  GeometryCorePlanRuntime runtime(backend,1);
  auto family=runtime.admitTimelineScenePreview(owner,*scene,bindings,{},error);
  check(bool(family),"two distinct timeline plans share one bounded parent cache slot");
  if (!family) { check(false,error.c_str()); return; }
  check(family->owner()==owner && family->children().size()==2
      && family->children()[0].object().id==first.id && family->children()[1].object().id==second.id
      && family->children()[0].object().firstVertex!=family->children()[1].object().firstVertex,
      "private child identities retain exact object, material and exclusive ranges");
  check(&family->children()[0].plan()!=&family->children()[1].plan()
      && &family->children()[0].nativeResources()!=&family->children()[1].nativeResources()
      && &family->children()[0].scene()==&family->children()[1].scene()
      && &family->children()[0].scene()!=scene.get(),
      "children own independent immutable plans/resources and one frozen scene anchor");
  check(family->aggregateReceipt().operations==a->receipt().operations+b->receipt().operations
      && family->envelope().replayIterations<=384 && family->envelope().attemptBytes>family->envelope().retainedBytes
      && family->envelope().decodeByteVisits>0 && family->envelope().waterSampleElements>0,
      "aggregate receipts and prospective logical storage/work envelopes remain bounded");
  auto repeat=runtime.admitTimelineScenePreview(owner,*scene,bindings,{},error);
  check(repeat==family && runtime.admissionsInGeneration()==1,"exact full scene/binding reuse keeps one family admission");
  auto changed=bindings;
  changed[1].loweredPlan=bindings[0].loweredPlan;
  check(!runtime.admitTimelineScenePreview(owner,*scene,changed,{},error),"changed exact child bytes require a new parent generation");
  auto foreign=std::make_shared<Visual3DScene>(*scene);
  foreign->cameras[0].transform.translation.x=0.25f;
  check(!runtime.admitTimelineScenePreview(owner,*foreign,bindings,{},error),"foreign camera state refuses despite identical mesh ranges");
  foreign=std::make_shared<Visual3DScene>(*scene); foreign->objects[1].transform.translation.y=0.25f;
  check(!runtime.admitTimelineScenePreview(owner,*foreign,bindings,{},error),"foreign sibling transform refuses exact immutable scene reuse");
  foreign=std::make_shared<Visual3DScene>(*scene); foreign->materials[0].roughness=0.5f;
  check(!runtime.admitTimelineScenePreview(owner,*foreign,bindings,{},error),"foreign material refuses exact immutable scene reuse");
  changed=bindings; changed[1].object=first.id;
  check(!runtime.admitTimelineScenePreview(next,*scene,changed,{},error),"duplicate child objects refuse before publication");
  changed=bindings; std::swap(changed[0],changed[1]);
  check(!runtime.admitTimelineScenePreview(next,*scene,changed,{},error),"noncanonical child order refuses");
  changed=bindings; changed[1].object={999};
  check(!runtime.admitTimelineScenePreview(next,*scene,changed,{},error),"missing exact object refuses");
  foreign=std::make_shared<Visual3DScene>(*scene);
  foreign->objects[1].firstVertex=foreign->objects[0].firstVertex;
  check(!runtime.admitTimelineScenePreview(next,*foreign,bindings,{},error),"overlapping sibling ranges cannot borrow child ownership");
  check(!runtime.admitPreview(owner,lowerRuntimePlan(contract,*a),{}, {},error),"legacy plan cannot replace a family under its existing key");
  check(!runtime.admitTimelineSceneExport(owner,*scene,bindings,{},error),"export entry point refuses a preview parent identity");
  auto exported=owner; exported.use=PlanUse::exportRender;
  check(!runtime.admitTimelineSceneExport(exported,*scene,bindings,{},error),"escaped family blocks pressure from a new use");
  std::optional<TimelineSceneChildAdmission> escapedChild(family->children()[0]);
  repeat.reset(); family.reset();
  check(!runtime.admitTimelineScenePreview(next,*scene,bindings,{},error),"one escaped child independently holds the complete family cache slot");
  escapedChild.reset();
  family=runtime.admitTimelineScenePreview(next,*scene,bindings,{},error);
  check(bool(family),"released child permits deterministic family eviction");
  if (!family) return;
  auto heldChild=std::optional<TimelineSceneChildAdmission>(family->children()[1]);
  check(runtime.reconcileGeneration(10,21,30,error),"family generation reconciliation succeeds");
  check(!runtime.admitTimelineScenePreview(next,*scene,bindings,{},error),"stale helper generation refuses family reuse");
  auto restarted=next; restarted.helperGeneration=21;
  family.reset();
  check(!runtime.admitTimelineScenePreview(restarted,*scene,bindings,{},error),"escaped old child remains charged across helper restart");
  heldChild.reset();
  check(bool(runtime.admitTimelineScenePreview(restarted,*scene,bindings,{},error)),"old unused generation can retire only after child release");

  // Identical exact bytes still create separate children. The authored rest
  // geometry is identical, so both exclusive object ranges remain authentic.
  GeometryCorePlanRuntime identical(backend,1);
  changed=bindings; changed[1].loweredPlan=bindings[0].loweredPlan;
  const auto twins=identical.admitTimelineScenePreview(owner,*scene,changed,{},error);
  check(twins && &twins->children()[0].plan()!=&twins->children()[1].plan()
      && &twins->children()[0].nativeResources()!=&twins->children()[1].nativeResources(),
      "identical plan bytes on distinct objects retain independent authority and resources");

  GeometryCorePlanRuntime budgets(backend,2);
  ResourceLimits shortOperations; shortOperations.maxOperations=a->receipt().operations;
  check(!budgets.admitTimelineScenePreview(owner,*scene,bindings,shortOperations,error)
      && budgets.admissionsInGeneration()==0,"remaining aggregate operations refuse before decoding a sibling");
  std::size_t retainedFields=0;
  check(bool(decodeTimelineScenePlan(bindings[0],ResourceLimits{},error,&retainedFields)) && retainedFields>0,
      "limits-aware decoder derives retained intermediate field cardinality before replay");
  ResourceLimits shortFields; shortFields.maxFieldElements=retainedFields;
  check(!budgets.admitTimelineScenePreview(owner,*scene,bindings,shortFields,error),
      "intermediate field storage shares one remaining allowance across siblings");
  ResourceLimits shortDispatch; shortDispatch.maxDispatches=a->receipt().dispatches;
  check(!budgets.admitTimelineScenePreview(owner,*scene,bindings,shortDispatch,error),"aggregate dispatches do not multiply per child");
  ResourceLimits shortVertices; shortVertices.maxVertices=scene->vertexCount-1;
  check(!budgets.admitTimelineScenePreview(owner,*scene,bindings,shortVertices,error),"complete scene shares the backend vertex cap");
  ResourceLimits shortIndices; shortIndices.maxIndices=scene->indexCount-1;
  check(!budgets.admitTimelineScenePreview(owner,*scene,bindings,shortIndices,error),"complete scene shares the backend index cap");
  ResourceLimits shortBytes; shortBytes.maxBufferBytes=1;
  check(!budgets.admitTimelineScenePreview(owner,*scene,bindings,shortBytes,error),"logical storage refuses before parser/scene allocation");
  changed=bindings; changed[1].loweredPlan.assign(kMaximumTimelineScenePlanBytes,'0');
  check(!budgets.admitTimelineScenePreview(owner,*scene,changed,{},error),"aggregate text/binary capacity refuses oversized later children");
  changed.assign(65,bindings[0]);
  check(!budgets.admitTimelineScenePreview(owner,*scene,changed,{},error),"family binding capacity remains 64");
  ResourceLimits exhausted; exhausted.maxOperations=0;
  check(!decodeTimelineScenePlan(bindings[0],exhausted,error),"limits-aware decoder refuses exhausted remaining budget");
  check(bool(decodeTimelineScenePlan(bindings[0],error)),"old timeline decoder overload remains available");

  GeometryCorePlanRuntime rollback(backend,1);
  auto incumbent=rollback.admitTimelineScenePreview(owner,*scene,bindings,{},error);
  const auto* originalFamily=incumbent.get();
  const auto* originalPlan=incumbent ? &incumbent->children()[0].plan() : nullptr;
  incumbent.reset();
  changed=bindings; changed[1].loweredPlan.back()='z';
  check(!rollback.admitTimelineScenePreview(next,*scene,changed,{},error)
      && rollback.admissionsInGeneration()==1,"later-child decoder refusal does not publish a partial family");
  incumbent=rollback.admitTimelineScenePreview(owner,*scene,bindings,{},error);
  check(incumbent && incumbent.get()==originalFamily && &incumbent->children()[0].plan()==originalPlan,
      "failed staged family conserves incumbent cache entry and child resources");
  if (incumbent) {
    shortBytes.maxBufferBytes=incumbent->envelope().attemptBytes+incumbent->envelope().retainedBytes-1;
    check(!rollback.admitTimelineScenePreview(owner,*scene,bindings,shortBytes,error),
        "cached reuse charges escaped retained families as well as prospective attempts");
  }
  GeometryCorePlanRuntime uses(backend,2);
  const auto previewFamily=uses.admitTimelineScenePreview(owner,*scene,bindings,{},error);
  const auto exportFamily=uses.admitTimelineSceneExport(exported,*scene,bindings,{},error);
  check(previewFamily && exportFamily && previewFamily->owner().use!=exportFamily->owner().use
      && &previewFamily->children()[0].plan()!=&exportFamily->children()[0].plan()
      && timelinefamily::sameScene(previewFamily->scene(),exportFamily->scene())
      && encodeRuntimeValue(previewFamily->children()[0].plan().value())
          ==encodeRuntimeValue(exportFamily->children()[0].plan().value()),
      "preview/export families keep distinct leases/resources over identical immutable authored bytes");
  GeometryCorePlanRuntime legacy(backend,1);
  const auto old=legacy.admitPreview(owner,lowerRuntimePlan(contract,*a),{}, {},error);
  check(old && !legacy.admitTimelineScenePreview(owner,*scene,bindings,{},error),
      "family cannot replace a legacy immutable owner key");
  TestBackend limited=backend; limited.capabilities.limits.maxOperations=a->receipt().operations;
  GeometryCorePlanRuntime backendLimits(limited,2);
  check(!backendLimits.admitTimelineScenePreview(owner,*scene,bindings,{},error),
      "effective backend operation cap is shared across all children");

  // Authentic transform-only ancestors and one exclusive retained-mesh leaf.
  // Identity transforms keep the longest valid hierarchy non-singular.
  auto deep=std::make_shared<Visual3DScene>();
  deep->id={8}; deep->activeCamera={50}; deep->cameraCount=1;
  deep->cameras[0]=scene->cameras[0];
  for (std::size_t i=0;i<Visual3DScene::kMaxObjects-1;++i) {
    deep->objects[i].id={static_cast<std::uint32_t>(i+1)};
    if (i) deep->objects[i].parent={static_cast<std::uint32_t>(i)};
    ++deep->objectCount;
  }
  SceneObjectRecord leaf; leaf.id={64}; leaf.parent={63};
  check(appendGeometryToScene(*a,leaf,material,*deep,error)
      && validateVisual3DScene(*deep).valid(),"64-object identity parent chain has an authentic bound leaf");
  std::vector<TimelineSceneBinding> deepBinding{{leaf.id,bindings[0].loweredPlan}};
  std::size_t parentSearchLowerBound=0,ancestryEdges=0;
  // In this ordering, ancestor at index j takes j+1 findById record visits.
  // This independent census counts actual ancestry, not envelope constants.
  for (std::size_t child=0;child<deep->objectCount;++child)
    for (std::size_t ancestor=0;ancestor<child;++ancestor) {
      parentSearchLowerBound+=ancestor+1;
      ++ancestryEdges;
    }
  check(parentSearchLowerBound==43680 && ancestryEdges==2016,
      "deep chain independently requires 43680 searches in each hierarchy traversal");
  timelinefamily::Envelope deepEnvelope;
  check(timelinefamily::deriveEnvelope(*deep,deepBinding,ResourceLimits{},0,deepEnvelope,error)
      && deepEnvelope.entryCountChecks==13 && deepEnvelope.entryRecordVisits==17
      && deepEnvelope.sceneValidationCalls==4
      && deepEnvelope.hierarchyLookupVisits>=deepEnvelope.sceneValidationCalls*2*parentSearchLowerBound
      && deepEnvelope.sceneRecordVisits>=deepEnvelope.hierarchyLookupVisits
      && deepEnvelope.hierarchyDepthIterations>=deepEnvelope.sceneValidationCalls*2*ancestryEdges
      && deepEnvelope.matrixConstructions>=deepEnvelope.sceneValidationCalls*(deep->objectCount+ancestryEdges)
      && deepEnvelope.matrixMultiplyTerms>=deepEnvelope.sceneValidationCalls*ancestryEdges*64
      && deepEnvelope.matrixElementVisits>=deepEnvelope.matrixMultiplyTerms*3
      && deepEnvelope.matrixPostcheckIterations>=deepEnvelope.sceneValidationCalls*deep->objectCount*6
      && deepEnvelope.sceneCopyElements>=5*(deep->vertices.size()+deep->indices.size()+deep->objects.size()),
      "prospective schedule covers repeated hierarchy searches, matrix loops and complete scene copies");
  GeometryCorePlanRuntime deepRuntime(backend,1);
  auto deepOwner=owner; ++deepOwner.clipIdentity;
  const auto deepFamily=deepRuntime.admitTimelineScenePreview(deepOwner,*deep,deepBinding,{},error);
  check(deepFamily && deepFamily->children().size()==1
      && deepFamily->children()[0].object().id==leaf.id
      && timelinefamily::sameScene(deepFamily->scene(),*deep),
      "deep valid chain is admitted with unchanged exact scene and child identity");
  if (!deepFamily) return;
  const auto* deepPlan=&deepFamily->children()[0].plan();
  const auto* deepResources=&deepFamily->children()[0].nativeResources();
  auto deepNext=deepOwner; ++deepNext.planGeneration;
  auto invalid=std::make_shared<Visual3DScene>(*deep);
  invalid->objects[0].parent=leaf.id;
  timelinefamily::Envelope invalidEnvelope;
  check(timelinefamily::deriveEnvelope(*invalid,deepBinding,ResourceLimits{},0,invalidEnvelope,error),
      "prospective derivation inspects capacities without executing invalid hierarchy validation");
  ResourceLimits beforeWalk;
  beforeWalk.maxBufferBytes=deepFamily->envelope().attemptBytes+deepFamily->envelope().retainedBytes-1;
  check(!deepRuntime.admitTimelineScenePreview(deepOwner,*invalid,deepBinding,beforeWalk,error)
      && error=="Timeline scene family and retained owners exceed the attempt budget"
      && deepRuntime.admissionsInGeneration()==1,
      "short retained/attempt storage refuses before the invalid hierarchy is traversed");
  GeometryCorePlanRuntime invalidRuntime(backend,1);
  check(!invalidRuntime.admitTimelineScenePreview(deepNext,*invalid,deepBinding,{},error)
      && error=="Timeline scene family requires a valid scene and 1..64 exact bindings"
      && invalidRuntime.admissionsInGeneration()==0,
      "admitted work schedule still refuses a cyclic hierarchy before publication");
  invalid->objects[0].parent={999};
  check(!invalidRuntime.admitTimelineScenePreview(deepNext,*invalid,deepBinding,{},error)
      && invalidRuntime.admissionsInGeneration()==0,"missing deep-chain parent remains refused");
  invalid->objects[0].parent={}; invalid->objects[1].transform.scale.x=0;
  check(!invalidRuntime.admitTimelineScenePreview(deepNext,*invalid,deepBinding,{},error),
      "singular ancestor transform remains refused after schedule admission");
  invalid->objectCount=Visual3DScene::kMaxObjects+1;
  check(!timelinefamily::deriveEnvelope(*invalid,deepBinding,ResourceLimits{},0,invalidEnvelope,error)
      && !invalidRuntime.admitTimelineScenePreview(deepNext,*invalid,deepBinding,{},error)
      && error=="Timeline scene family exceeds scene counts or 1..64 binding capacity",
      "invalid object count refuses in the cheap entry schedule without indexing records");
  const auto conserved=deepRuntime.admitTimelineScenePreview(deepOwner,*deep,deepBinding,{},error);
  check(conserved==deepFamily && &conserved->children()[0].plan()==deepPlan
      && &conserved->children()[0].nativeResources()==deepResources
      && deepRuntime.admissionsInGeneration()==1,
      "pre-traversal shortage conserves exact incumbent scene, child resources and admission count");

}


#if defined(ARBIT_ALLOCATION_FAILURE_TEST_HOOKS)
namespace {
using allocationtest::Attempt;
using allocationtest::Family;
using allocationtest::Mode;
using allocationtest::Phase;
using allocationtest::Receipt;
using allocationtest::Record;
using allocationtest::Site;
using allocationtest::SiteScope;

void* infrastructureNew(Family family, std::size_t bytes,
                         std::size_t alignment = 32) {
  const auto aligned = static_cast<std::align_val_t>(alignment);
  switch (family) {
  case Family::scalar: return ::operator new(bytes);
  case Family::array: return ::operator new[](bytes);
  case Family::scalarNothrow: return ::operator new(bytes, std::nothrow);
  case Family::arrayNothrow: return ::operator new[](bytes, std::nothrow);
  case Family::alignedScalar: return ::operator new(bytes, aligned);
  case Family::alignedArray: return ::operator new[](bytes, aligned);
  case Family::alignedScalarNothrow: return ::operator new(bytes, aligned, std::nothrow);
  case Family::alignedArrayNothrow: return ::operator new[](bytes, aligned, std::nothrow);
  }
  return nullptr;
}

void infrastructureDelete(Family family, void* memory, std::size_t bytes = 0,
                           bool sized = false, std::size_t alignment = 32) noexcept {
  const auto aligned = static_cast<std::align_val_t>(alignment);
  switch (family) {
  case Family::scalar:
    if (sized) ::operator delete(memory, bytes); else ::operator delete(memory);
    break;
  case Family::array:
    if (sized) ::operator delete[](memory, bytes); else ::operator delete[](memory);
    break;
  case Family::scalarNothrow: ::operator delete(memory, std::nothrow); break;
  case Family::arrayNothrow: ::operator delete[](memory, std::nothrow); break;
  case Family::alignedScalar:
    if (sized) ::operator delete(memory, bytes, aligned);
    else ::operator delete(memory, aligned);
    break;
  case Family::alignedArray:
    if (sized) ::operator delete[](memory, bytes, aligned);
    else ::operator delete[](memory, aligned);
    break;
  case Family::alignedScalarNothrow: ::operator delete(memory, aligned, std::nothrow); break;
  case Family::alignedArrayNothrow: ::operator delete[](memory, aligned, std::nothrow); break;
  }
}

unsigned infrastructureHandlerCalls = 0;
void infrastructureNewHandler() {
  ++infrastructureHandlerCalls;
  throw std::bad_alloc();
}

// These calls test the selector and replaceable families, not product refusals.
void allocationFailureInfrastructure() {
  const Family families[] = {
      Family::scalar, Family::array, Family::scalarNothrow, Family::arrayNothrow,
      Family::alignedScalar, Family::alignedArray,
      Family::alignedScalarNothrow, Family::alignedArrayNothrow};
  Receipt count;
  {
    Attempt attempt(count, Mode::countOnly);
    SiteScope site(Site::infrastructurePrimary, Phase::control);
    for (const auto family : families)
      infrastructureDelete(family, infrastructureNew(family, 0));
  }
  check(count.errors == 0 && count.allocations == 8 && count.deallocations == 8
      && count.events == 16 && count.recorded == 0 && allocationtest::isCurrent(count),
      "count-only control observes actual backend entries and deletes without trace storage");

  Record records[32];
  Receipt normal;
  {
    Attempt attempt(normal, Mode::observe, records, 32);
    SiteScope site(Site::infrastructurePrimary, Phase::control);
    for (const auto family : families)
      infrastructureDelete(family, infrastructureNew(family, 0));
  }
  check(normal.errors == 0 && normal.recorded == 16 && normal.events == count.events
      && !allocationtest::isCurrent(count) && allocationtest::isCurrent(normal),
      "new generations reject stale receipts and match the exact count-only control");
  auto staleGeneration = normal;
  ++staleGeneration.generation;
  check(!allocationtest::isCurrent(staleGeneration) && allocationtest::isCurrent(normal),
      "receipt validation rejects a foreign generation without changing the valid receipt");
  for (std::size_t i = 0; i < 8; ++i) {
    const auto& allocation = records[2*i];
    const auto& deletion = records[2*i+1];
    const bool aligned = i >= 4;
    check(allocation.event == allocationtest::Event::allocation
        && allocation.result == allocationtest::Result::succeeded
        && allocation.family == families[i] && allocation.requestedBytes == 0
        && allocation.effectiveBytes == 1 && allocation.allocationBase != nullptr
        && allocation.site == Site::infrastructurePrimary && allocation.phase == Phase::control
        && allocation.ordinal == i+1 && allocation.sequence == 2*i+1
        && allocation.generation == normal.generation
        && allocation.requestedAlignment == (aligned ? 32 : 0)
        && allocation.effectiveAlignment == (aligned ? 32 : alignof(std::max_align_t))
        && reinterpret_cast<std::uintptr_t>(allocation.allocationBase)
             % allocation.effectiveAlignment == 0
        && deletion.event == allocationtest::Event::deallocation
        && deletion.result == allocationtest::Result::deallocated
        && deletion.family == families[i] && deletion.sequence == 2*i+2
        && deletion.ordinal == i+1
        && deletion.allocationBase == allocation.allocationBase,
        "ordinary/aligned scalar/array throwing/nothrow records retain actual bases and normalization");
  }
  const Family sizedFamilies[] = {
      Family::scalar, Family::array, Family::alignedScalar, Family::alignedArray};
  Receipt sized;
  {
    Attempt attempt(sized, Mode::observe, records, 32);
    for (const auto family : sizedFamilies)
      infrastructureDelete(family, infrastructureNew(family, 37), 37, true);
  }
  for (std::size_t i = 0; i < 4; ++i)
    check(sized.errors == 0 && sized.recorded == 8
        && records[2*i+1].deleteForm == allocationtest::DeleteForm::sized
        && records[2*i+1].requestedBytes == 37
        && records[2*i+1].allocationBase == records[2*i].allocationBase,
        "each ordinary/aligned scalar/array sized delete observes the correct allocation base");

  struct HandlerRestore {
    std::new_handler previous;
    ~HandlerRestore() noexcept { std::set_new_handler(previous); }
  };
  const auto previousHandler = std::set_new_handler(infrastructureNewHandler);
  HandlerRestore restoreHandler {previousHandler};
  infrastructureHandlerCalls = 0;
  for (const auto family : families) {
    Receipt failure;
    auto outcome = allocationtest::Outcome::notCompleted;
    bool continuation = false;
    {
      Attempt attempt(failure, Mode::inject, records, 32,
                      {Site::infrastructurePrimary, Phase::p2Final, 1});
      SiteScope site(Site::infrastructurePrimary, Phase::p2Final);
      try {
        void* memory = infrastructureNew(family, 41);
        outcome = memory == nullptr ? allocationtest::Outcome::returnedEmpty
                                    : allocationtest::Outcome::returnedValue;
        infrastructureDelete(family, memory);
      } catch (const std::bad_alloc&) {
        outcome = allocationtest::Outcome::threwBadAlloc;
      } catch (...) {
        outcome = allocationtest::Outcome::threwOther;
      }
      void* memory = infrastructureNew(family, 41);
      continuation = memory != nullptr;
      infrastructureDelete(family, memory);
    }
    const bool nothrow = family == Family::scalarNothrow || family == Family::arrayNothrow
        || family == Family::alignedScalarNothrow || family == Family::alignedArrayNothrow;
    check(failure.errors == 0 && failure.hits == 1 && failure.allocations == 2
        && failure.deallocations == 1 && failure.recorded == 3 && continuation
        && outcome == (nothrow ? allocationtest::Outcome::returnedEmpty
                              : allocationtest::Outcome::threwBadAlloc)
        && records[0].family == family && records[0].result == allocationtest::Result::injectedFailure
        && records[0].allocationBase == nullptr && records[0].requestedBytes == 41
        && records[0].ordinal == 1 && records[1].ordinal == 2
        && records[1].result == allocationtest::Result::succeeded
        && records[2].allocationBase == records[1].allocationBase,
        "every replaceable new family fails once with the real throwing/nothrow outcome and continuation");
  }
  std::set_new_handler(previousHandler);
  check(infrastructureHandlerCalls == 0, "selected failures bypass the installed new_handler");

  Receipt separation;
  bool phaseSucceeded = false, siteSucceeded = false, selected = false;
  {
    Attempt attempt(separation, Mode::inject, records, 32,
                    {Site::infrastructurePrimary, Phase::p2Final, 1});
    {
      SiteScope otherPhase(Site::infrastructurePrimary, Phase::p2Temporary);
      void* memory = ::operator new(7); phaseSucceeded = memory != nullptr;
      ::operator delete(memory);
    }
    {
      SiteScope otherSite(Site::infrastructureOther, Phase::p2Final);
      void* memory = ::operator new(9); siteSucceeded = memory != nullptr;
      ::operator delete(memory);
    }
    {
      SiteScope site(Site::infrastructurePrimary, Phase::p2Final);
      try { void* memory = ::operator new(11); ::operator delete(memory); }
      catch (const std::bad_alloc&) { selected = true; }
    }
  }
  check(separation.errors == 0 && separation.hits == 1
      && separation.allocations == 3 && selected && phaseSucceeded && siteSucceeded
      && records[4].site == Site::infrastructurePrimary && records[4].phase == Phase::p2Final
      && records[4].ordinal == 1,
      "site and phase counts separate real entries and scopes restore their predecessors");

  Receipt nestedSites;
  bool restoredSelection = false;
  {
    Attempt attempt(nestedSites, Mode::inject, records, 32,
                    {Site::infrastructurePrimary, Phase::control, 2});
    SiteScope outerSite(Site::infrastructurePrimary, Phase::control);
    void* first = ::operator new(5); ::operator delete(first);
    {
      SiteScope innerSite(Site::infrastructureOther, Phase::control);
      void* inner = ::operator new(6); ::operator delete(inner);
    }
    try { void* second = ::operator new(7); ::operator delete(second); }
    catch (const std::bad_alloc&) { restoredSelection = true; }
  }
  check(nestedSites.errors == 0 && nestedSites.hits == 1 && restoredSelection
      && records[4].site == Site::infrastructurePrimary && records[4].ordinal == 2,
      "nested scopes restore the outer site's accumulated ordinal before selection");

  std::atomic<bool> workerReady {false}, startWorker {false}, workerDone {false};
  bool workerSucceeded = false, workerInactive = false;
  std::thread worker([&] {
    workerReady.store(true, std::memory_order_release);
    while (!startWorker.load(std::memory_order_acquire)) {}
    workerInactive = !allocationtest::detail::state.active;
    void* memory = ::operator new(13);
    workerSucceeded = memory != nullptr;
    ::operator delete(memory);
    workerDone.store(true, std::memory_order_release);
  });
  while (!workerReady.load(std::memory_order_acquire)) {}
  Receipt threaded;
  bool threadSelected = false;
  {
    Attempt attempt(threaded, Mode::inject, records, 32,
                    {Site::infrastructurePrimary, Phase::control, 1});
    startWorker.store(true, std::memory_order_release);
    while (!workerDone.load(std::memory_order_acquire)) {}
    SiteScope site(Site::infrastructurePrimary, Phase::control);
    try { void* memory = ::operator new(15); ::operator delete(memory); }
    catch (const std::bad_alloc&) { threadSelected = true; }
  }
  worker.join();
  check(threaded.errors == 0 && threaded.hits == 1 && threaded.allocations == 1
      && threaded.deallocations == 0 && workerInactive && workerSucceeded && threadSelected,
      "a worker prepared disarmed cannot consume another thread's selected ordinal");

  std::atomic<bool> finishWorkerReady {false}, startWrongFinish {false}, wrongFinishDone {false};
  Attempt* wrongFinishAttempt = nullptr;
  bool finishWorkerInactive = false, finishWorkerUnchanged = true;
  std::thread finishWorker([&] {
    const auto& workerState = allocationtest::detail::state;
    const auto before = workerState;
    finishWorkerInactive = !workerState.active;
    finishWorkerReady.store(true, std::memory_order_release);
    while (!startWrongFinish.load(std::memory_order_acquire)) {}
    wrongFinishAttempt->finish();
    finishWorkerUnchanged = workerState.active == before.active
        && workerState.selecting == before.selecting && workerState.observing == before.observing
        && workerState.mode == before.mode && workerState.selection.site == before.selection.site
        && workerState.selection.phase == before.selection.phase
        && workerState.selection.ordinal == before.selection.ordinal
        && workerState.records == before.records && workerState.capacity == before.capacity
        && workerState.site == before.site && workerState.phase == before.phase
        && workerState.scope == before.scope && workerState.nextScope == before.nextScope
        && workerState.depth == before.depth && workerState.nextGeneration == before.nextGeneration
        && workerState.completedGeneration == before.completedGeneration
        && workerState.receipt.threadState == before.receipt.threadState
        && workerState.receipt.generation == before.receipt.generation
        && workerState.receipt.events == before.receipt.events
        && workerState.receipt.allocations == before.receipt.allocations
        && workerState.receipt.deallocations == before.receipt.deallocations
        && workerState.receipt.hits == before.receipt.hits
        && workerState.receipt.recorded == before.receipt.recorded
        && workerState.receipt.errors == before.receipt.errors;
    for (std::size_t site = 0; site < allocationtest::siteCount; ++site)
      for (std::size_t phase = 0; phase < allocationtest::phaseCount; ++phase)
        finishWorkerUnchanged = finishWorkerUnchanged
            && workerState.receipt.siteAllocations[site][phase] == before.receipt.siteAllocations[site][phase]
            && workerState.receipt.siteDeallocations[site][phase] == before.receipt.siteDeallocations[site][phase];
    wrongFinishDone.store(true, std::memory_order_release);
  });
  while (!finishWorkerReady.load(std::memory_order_acquire)) {}
  Receipt wrongFinish;
  bool rejectionRecorded = false, ownerStillArmed = false, ownerCleared = false;
  std::uint64_t wrongFinishGeneration = 0;
  {
    Attempt attempt(wrongFinish, Mode::observe, records, 32);
    wrongFinishGeneration = attempt.generation();
    wrongFinishAttempt = &attempt;
    // The worker finishes before any owner access resumes; concurrent finish is unsupported.
    startWrongFinish.store(true, std::memory_order_release);
    while (!wrongFinishDone.load(std::memory_order_acquire)) {}
    rejectionRecorded = allocationtest::hasError(wrongFinish, allocationtest::Error::staleGeneration);
    ownerStillArmed = attempt.armed() && allocationtest::detail::state.active;
    {
      SiteScope site(Site::infrastructurePrimary, Phase::control);
      void* memory = ::operator new(14); ::operator delete(memory);
    }
    attempt.finish();
    const auto& state = allocationtest::detail::state;
    ownerCleared = !attempt.armed() && !state.active && !state.selecting && !state.observing
        && state.records == nullptr && state.capacity == 0
        && state.site == Site::none && state.phase == Phase::none
        && state.scope == 0 && state.depth == 0;
  }
  finishWorker.join();
  check(rejectionRecorded && ownerStillArmed && ownerCleared
      && finishWorkerInactive && finishWorkerUnchanged
      && allocationtest::hasError(wrongFinish, allocationtest::Error::staleGeneration)
      && wrongFinish.generation == wrongFinishGeneration
      && wrongFinish.allocations == 1 && wrongFinish.deallocations == 1
      && wrongFinish.recorded == 2 && !allocationtest::isCurrent(wrongFinish),
      "serialized wrong-thread finish stays rejected after owner disarm without changing worker TLS");
  Receipt finishRecovery;
  { Attempt attempt(finishRecovery, Mode::observe, records, 32);
    SiteScope site(Site::infrastructureOther, Phase::p1);
    void* memory = ::operator new(16); ::operator delete(memory); }
  check(finishRecovery.errors == 0 && finishRecovery.hits == 0
      && finishRecovery.generation > wrongFinishGeneration
      && finishRecovery.allocations == 1 && finishRecovery.deallocations == 1
      && finishRecovery.recorded == 2 && allocationtest::isCurrent(finishRecovery)
      && records[0].site == Site::infrastructureOther && records[0].phase == Phase::p1
      && records[0].ordinal == 1 && records[1].allocationBase == records[0].allocationBase,
      "a fresh owner attempt succeeds with reset site counts after wrong-thread finish rejection");

  bool cleanupSucceeded = false;
  struct UnwindOwner {
    void* memory;
    bool& succeeded;
    ~UnwindOwner() noexcept {
      void* continuation = ::operator new(17, std::nothrow);
      succeeded = continuation != nullptr;
      ::operator delete(continuation);
      ::operator delete(memory);
    }
  };
  void* unwoundMemory = ::operator new(19);
  Receipt unwinding;
  auto unwindOutcome = allocationtest::Outcome::notCompleted;
  {
    Attempt attempt(unwinding, Mode::inject, records, 32,
                    {Site::infrastructurePrimary, Phase::control, 1});
    try {
      SiteScope site(Site::infrastructurePrimary, Phase::control);
      UnwindOwner owner {unwoundMemory, cleanupSucceeded};
      void* memory = ::operator new(21);
      ::operator delete(memory);
      unwindOutcome = allocationtest::Outcome::returnedValue;
    } catch (const std::bad_alloc&) {
      unwindOutcome = allocationtest::Outcome::threwBadAlloc;
    } catch (...) {
      unwindOutcome = allocationtest::Outcome::threwOther;
    }
  }
  check(unwinding.errors == 0 && unwinding.hits == 1 && unwinding.allocations == 2
      && unwinding.deallocations == 2 && cleanupSucceeded
      && unwindOutcome == allocationtest::Outcome::threwBadAlloc
      && records[1].site == Site::infrastructurePrimary
      && records[1].result == allocationtest::Result::succeeded
      && records[3].allocationBase == unwoundMemory
      && records[3].sequence > records[0].sequence,
      "observation survives real exception cleanup and records bases before scope restoration");

  Receipt zero;
  { Attempt attempt(zero, Mode::inject, records, 32,
                    {Site::infrastructurePrimary, Phase::control, 0}); }
  check(allocationtest::hasError(zero, allocationtest::Error::zeroOrdinal),
      "ordinal zero rejects arming");
  Receipt invalidTrace;
  { Attempt attempt(invalidTrace, Mode::observe); }
  check(allocationtest::hasError(invalidTrace, allocationtest::Error::invalidTrace)
      && !allocationtest::detail::state.active,
      "observed attempts reject absent prepared trace storage before arming");
  Receipt unhit;
  { Attempt attempt(unhit, Mode::inject, records, 32,
                    {Site::infrastructurePrimary, Phase::control, 2});
    SiteScope site(Site::infrastructurePrimary, Phase::control);
    void* memory = ::operator new(23); ::operator delete(memory); }
  check(allocationtest::hasError(unhit, allocationtest::Error::unhitOrdinal)
      && unhit.hits == 0, "an unhit requested ordinal rejects the attempt instead of passing");
  Receipt outer, nested;
  { Attempt attempt(outer, Mode::observe, records, 32);
    { Attempt inner(nested, Mode::observe, records, 32); }
    void* memory = ::operator new(25); ::operator delete(memory); }
  check(allocationtest::hasError(outer, allocationtest::Error::nestedAttempt)
      && allocationtest::hasError(nested, allocationtest::Error::nestedAttempt)
      && outer.allocations == 1, "nested owners invalidate evidence without disarming their owner");
  Receipt overflow;
  { Attempt attempt(overflow, Mode::observe, records, 1);
    void* memory = ::operator new(27); ::operator delete(memory); }
  check(allocationtest::hasError(overflow, allocationtest::Error::traceOverflow)
      && overflow.events == 2 && overflow.recorded == 1 && !allocationtest::isCurrent(overflow),
      "delete trace overflow remains explicit and cannot produce accepted truncated evidence");
  Receipt failingOverflow;
  { Attempt attempt(failingOverflow, Mode::inject, records, 1,
                    {Site::infrastructurePrimary, Phase::control, 1});
    SiteScope site(Site::infrastructurePrimary, Phase::control);
    void* selectedMemory = ::operator new(28, std::nothrow);
    ::operator delete(selectedMemory);
    void* continuation = ::operator new(29); ::operator delete(continuation); }
  check(failingOverflow.hits == 1 && failingOverflow.events == 3
      && allocationtest::hasError(failingOverflow, allocationtest::Error::traceOverflow)
      && !allocationtest::isCurrent(failingOverflow),
      "trace overflow during one-shot failure continuation invalidates the complete attempt");
  Receipt reentrant;
  { Attempt attempt(reentrant, Mode::observe, records, 32);
    allocationtest::detail::ObservationGuard observation;
    void* memory = ::operator new(29); ::operator delete(memory); }
  check(allocationtest::hasError(reentrant, allocationtest::Error::reentrantObservation),
      "reentrant fixed observation rejects evidence without a measured callback");
  Receipt imbalance;
  alignas(SiteScope) unsigned char outerStorage[sizeof(SiteScope)];
  alignas(SiteScope) unsigned char innerStorage[sizeof(SiteScope)];
  { Attempt attempt(imbalance, Mode::observe, records, 32);
    auto* first = new (outerStorage) SiteScope(Site::infrastructurePrimary);
    auto* second = new (innerStorage) SiteScope(Site::infrastructureOther);
    first->~SiteScope(); second->~SiteScope(); }
  check(allocationtest::hasError(imbalance, allocationtest::Error::siteImbalance),
      "out-of-order site restoration invalidates the receipt");
  Receipt earlyFinish, staleScope;
  alignas(SiteScope) unsigned char staleStorage[sizeof(SiteScope)];
  SiteScope* oldSite = nullptr;
  { Attempt attempt(earlyFinish, Mode::observe, records, 32);
    oldSite = new (staleStorage) SiteScope(Site::infrastructurePrimary);
    attempt.finish(); }
  { Attempt attempt(staleScope, Mode::observe, records, 32);
    oldSite->~SiteScope(); }
  check(allocationtest::hasError(earlyFinish, allocationtest::Error::siteImbalance)
      && allocationtest::hasError(staleScope, allocationtest::Error::staleGeneration),
      "early disarm and a surviving old-generation site each reject their evidence");
  Receipt invalidAlignment;
  bool invalidThrew = false;
  { Attempt attempt(invalidAlignment, Mode::inject, records, 32,
                    {Site::infrastructurePrimary, Phase::control, 1});
    SiteScope site(Site::infrastructurePrimary, Phase::control);
    try { void* memory = infrastructureNew(Family::alignedScalar, 31, 3);
          infrastructureDelete(Family::alignedScalar, memory, 0, false, 3); }
    catch (const std::bad_alloc&) { invalidThrew = true; } }
  check(invalidThrew && invalidAlignment.allocations == 0 && invalidAlignment.hits == 0
      && allocationtest::hasError(invalidAlignment, allocationtest::Error::unhitOrdinal),
      "invalid alignment refuses before selection or backend observation");
#if !defined(_WIN32)
  Receipt normalizedAlignment;
  { Attempt attempt(normalizedAlignment, Mode::observe, records, 32);
    void* memory = infrastructureNew(Family::alignedScalar, 0, 1);
    infrastructureDelete(Family::alignedScalar, memory, 0, false, 1); }
  check(normalizedAlignment.errors == 0 && normalizedAlignment.allocations == 1
      && records[0].requestedAlignment == 1
      && records[0].effectiveAlignment == sizeof(void*) && records[0].effectiveBytes == 1,
      "POSIX pointer alignment normalization precedes the single observed backend entry");
#endif
  Receipt recovery;
  { Attempt attempt(recovery, Mode::observe, records, 32);
    void* memory = ::operator new(33); ::operator delete(memory); }
  check(recovery.errors == 0 && recovery.hits == 0 && recovery.recorded == 2
      && records[0].site == Site::none && records[0].phase == Phase::none
      && records[0].ordinal == 1 && allocationtest::isCurrent(recovery)
      && !allocationtest::detail::state.active,
      "normal recovery after rejected attempts restores inactive TLS and fresh site counts");
  void* disarmed = ::operator new(35);
  ::operator delete(disarmed);
  check(allocationtest::isCurrent(recovery) && recovery.events == 2,
      "disarmed allocation leaves the completed receipt unchanged");
}
} // namespace
#endif

int main() {
  { std::string error; const bool passed=verifyParticlePointsBodyAuthority(error); check(passed,error.c_str()); }
  { std::string error; const bool passed=verifyParticleBodyMinimumNativeProofOracles(error); check(passed,error.c_str()); }
#if !defined(_WIN32)
  animatedGlbAdmissionFitsViewportThreadStack();
  v8SceneDecodeFitsViewportThreadStack();
#endif
  surfaceGeometryDomainsAndRetainedTransport();
  surfaceTableAndFieldTransport();
  reactiveSurfaceTransportAndCache();
  instanceMaterialFieldsRetainRuntimeAndSelection();
  timelineFieldsReplayExactFrames();
  waterSurfaceTypedContract(check);
  waterSurfaceHistoryReplay(check);
  waterSurfaceFrameCallback(check);
  waterReplaySourceBudget(check);
  orderedAudioGeometryReplay();
  selectedSourceSpectrumParity();
  audioScalarParity();
  retainedMaterialTables();
  triangleSurfaceFields();
  instanceAppearanceRetainsAttributes();
  spectralHistoryRetainsTimelineAndBounds();
  spectralHistoryRetainedFieldsReplayTypedValues();
  spectralHistoryDrivesDeformerAndInstancerRows();
  spectralHistoryDeformerEnvelopeRows();
  constructedFieldsAndNamedAttributes();
  sampleIndexSamplesExactAttributes();
  sampleIndexTransportVersionAndReplayRefusals();
  distributedPointsAndInstanceSelection();
  scoreFieldsRetainBoundedProvenance();
  instanceScoreFieldsRetainCanonicalOwnership();
  fixedDynamicInstanceScoreFields();
  scoreSampleFieldMeasures();
  legacyScoreFieldWireGolden();
  scoreDeformedPrototypeInstancesRetainCanonicalFrame();
  curveAuthoringRetainsEditsAndFields();
  audioDeformationAndEffector();
  audioEffectorFeatureInputs();
  retainedGeometryOperations();
  elementFieldsAndDisplacementRetainProvenance();
  spectrumFieldsRoundtripAndFollow();
  typedSpectrumDomainsRetainSourcesAndSamples();
  staticPointSpectrumStableIdentitySampling();
  spectrumInstancerAppearanceComposition();
  retainedSdfSurfaceReplay();
  retainedSdfTextSurfaceReplay();
  spectrumInstancesRetainBandIdentity();
  spectrumInstancerLayoutsRetainPlacementAuthority();
  curveRibbonsRetainSourceAndTopology();
  meshGeneratorsRoundtripAndRejectInvalidPlans();
  gridLoweringExactTopologyBoundsAndRefusals();
  legacyContractsMigrate();
  validRoundtrips();
  hostileTransportFailsClosed();
  invalidNumbersAndShapesFail();
  attributeCardinalityAndBudgets();
  typedFieldKindsAdmit();
  productionLoweringAndTypedEvaluation();
  productionAdmissionSharesPreviewAndExportData();
  particlePointsKeepImmutableOwnerAndCapacity();
  deferredParticleSpectrumFieldsKeepFrameOwnership();
  deferredParticleSpectrumColorsPromoteExactFrame();
  ownerIdentityAndBoundedLeases();
  timelineSceneFamiliesRetainExactChildrenAndBudgets();
#if defined(ARBIT_ALLOCATION_FAILURE_TEST_HOOKS)
  allocationFailureInfrastructure();
#endif
  if (failures)
    std::cerr << failures << " geometry core checks failed\n";
  return failures ? 1 : 0;
}
