#include "../src/geometry_core_admission.h"
#include "../src/mix_analyze.h"
#include "../../shared/GeometryAudioDeformer.h"
#include "point_distribution_fixture.h"
#include "spectral_history_fixture.h"
#include "constructed_field_fixture.h"
#include "instance_appearance_fixture.h"
#include "instance_material_fields_fixture.h"
#include "reactive_surface_fixture.h"
#include "reactive_frame_fixture.h"
#include "material_table_fixture.h"
#include "timeline_curve_fixture.h"
#include "ordered_audio_geometry_fixture.h"
#include "imported_geometry_fixture.h"
#include "../../shared/GeometryCoreScene.h"

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
  for (unsigned mode=0;mode<5;++mode) {
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
  invalid=*mesh; invalid.scoreFields[0].mode=static_cast<scorefield::Mode>(5);
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
  const auto combined=curve ? lowerAudioDeformer(*curve,9720,binding,timelineContract(CarrierKind::geometry3D),error)
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

int main() {
  surfaceGeometryDomainsAndRetainedTransport();
  surfaceTableAndFieldTransport();
  reactiveSurfaceTransportAndCache();
  instanceMaterialFieldsRetainRuntimeAndSelection();
  timelineFieldsReplayExactFrames();
  orderedAudioGeometryReplay();
  selectedSourceSpectrumParity();
  retainedMaterialTables();
  triangleSurfaceFields();
  instanceAppearanceRetainsAttributes();
  spectralHistoryRetainsTimelineAndBounds();
  constructedFieldsAndNamedAttributes();
  distributedPointsAndInstanceSelection();
  scoreFieldsRetainBoundedProvenance();
  curveAuthoringRetainsEditsAndFields();
  audioDeformationAndEffector();
  retainedGeometryOperations();
  elementFieldsAndDisplacementRetainProvenance();
  spectrumFieldsRoundtripAndFollow();
  spectrumInstancesRetainBandIdentity();
  curveRibbonsRetainSourceAndTopology();
  meshGeneratorsRoundtripAndRejectInvalidPlans();
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
