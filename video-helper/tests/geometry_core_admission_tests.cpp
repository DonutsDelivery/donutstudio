#include "../src/geometry_core_admission.h"

#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

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
  malformed[4] = 2;
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
  meshContract.maxIndices = 24;
  auto mesh = circle ? lowerCurveToMesh(*circle, 902, meshContract, error)
                     : std::nullopt;
  check(mesh && std::get<GeometryData>(mesh->data).positions.size() == 8,
        "Curve to Mesh lowering retains bounded curve-point identity");
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

int main() {
  legacyContractsMigrate();
  validRoundtrips();
  hostileTransportFailsClosed();
  invalidNumbersAndShapesFail();
  attributeCardinalityAndBudgets();
  typedFieldKindsAdmit();
  productionLoweringAndTypedEvaluation();
  productionAdmissionSharesPreviewAndExportData();
  ownerIdentityAndBoundedLeases();
  if (failures)
    std::cerr << failures << " geometry core checks failed\n";
  return failures ? 1 : 0;
}
