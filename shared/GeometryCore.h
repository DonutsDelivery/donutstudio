#pragma once

#include "GeometrySpectrumField.h"
#include "SdfSurfaceOperationContract.h"
#include "GeometryScoreField.h"
#include "VisualImportedAnimationOperationContract.h"
#include "GeometryPortContract.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include <new>
#include <stdexcept>
#include <tuple>

namespace videowire::geometry {

struct GeometryData final {
  std::vector<Vec3> positions;
  std::vector<StableId> vertexIds;
  std::vector<std::uint32_t> indices;
  // Generator-carried per-vertex uv (plane corners, sphere longitude and
  // latitude, cylinder side and polar caps), so uv-driven materials such as
  // the diffraction grating's groove field have real coordinates on graph
  // geometry. Retained meshes carry authored uv
  // through the attribute channel instead; empty means no uv.
  std::vector<Vec3> uv;
};
struct PointRecord final {
  Vec3 position{};
  StableId stableId = 0;
  float radius = 1;
  Quaternion rotation{};
};
struct PointsData final {
  std::vector<PointRecord> points;
};
struct CurvePointRecord final {
  Vec3 position{};
  StableId stableId = 0;
  float radius = 1;
  float tilt = 0;
};
struct SplineRecord final {
  StableId stableId = 0;
  std::uint32_t firstPoint = 0;
  std::uint32_t pointCount = 0;
  bool cyclic = false;
};
struct CurvesData final {
  std::vector<CurvePointRecord> points;
  std::vector<SplineRecord> splines;
};
struct InstanceRecord final {
  StableId stableId = 0;
  StableId sourceStableId = 0;
  Transform transform{};
};
struct InstancesData final {
  std::vector<InstanceRecord> instances;
};
struct FieldElement final {
  std::array<double, 4> components{};
};
struct FieldData final {
  std::vector<FieldElement> elements;
};
using CarrierData = std::variant<GeometryData, PointsData, CurvesData,
                                 InstancesData, FieldData>;
struct AttributeData final {
  AttributeDescriptor descriptor;
  std::vector<FieldElement> elements;
};
struct RetainedMeshData final {
  GeometryData geometry;
  std::vector<AttributeData> attributes;
  std::string importedAnimation {};
};
inline std::vector<AttributeDescriptor> sdfSurfaceAttributes(StableId valueId) {
  return {{valueId,"normal",ValueType::vector,Domain::vertex,Interpolation::normalizedLinear},
      {valueId==kMaximumStableId ? valueId-1 : valueId+1,"uv",ValueType::vector,Domain::vertex,Interpolation::linear},
      {valueId>=kMaximumStableId-1 ? valueId-2 : valueId+2,"sdfPrimitiveId",ValueType::integer,Domain::face,Interpolation::nearest}};
}
inline bool materializeSdfSurface(const sdfsurface::Request& request,StableId valueId,
    std::size_t maxVertices,std::size_t maxIndices,GeometryData& geometry,
    std::vector<AttributeData>& attributes,std::string& error) {
  sdfsurface::Mesh mesh;
  if(!sdfsurface::materialize(request,valueId,maxVertices,maxIndices,mesh,error)) return false;
  geometry={}; geometry.positions.reserve(mesh.positions.size());
  for(const auto& p:mesh.positions) geometry.positions.push_back({p[0],p[1],p[2]});
  geometry.vertexIds=std::move(mesh.vertexIds); geometry.indices=std::move(mesh.indices);
  const auto descriptors=sdfSurfaceAttributes(valueId);
  attributes={{descriptors[0],{}},{descriptors[1],{}},{descriptors[2],{}}};
  attributes[0].elements.reserve(mesh.normals.size()); attributes[1].elements.reserve(mesh.uv.size());
  attributes[2].elements.reserve(mesh.primitiveIds.size());
  for(const auto& n:mesh.normals) attributes[0].elements.push_back({{n[0],n[1],n[2],0}});
  for(const auto& uv:mesh.uv) attributes[1].elements.push_back({{uv[0],uv[1],uv[2],0}});
  for(const auto id:mesh.primitiveIds) attributes[2].elements.push_back({{static_cast<double>(id),0,0,0}});
  return true;
}
struct FieldOperationSettings final {
  Domain domain = Domain::unspecified;
  ValueType valueType = ValueType::unspecified;
  std::uint8_t mode = 0;
  StableId auxiliaryInputStableId = 0;
  std::array<double, 8> values{};
  std::string attributeName;
};
inline bool operator==(const FieldOperationSettings &a, const FieldOperationSettings &b) noexcept {
  return a.domain == b.domain && a.valueType == b.valueType && a.mode == b.mode &&
         a.auxiliaryInputStableId == b.auxiliaryInputStableId && a.values == b.values &&
         a.attributeName == b.attributeName;
}
// One edge's crease, keyed by the stable ids of its vertices with vertexA < vertexB.
// weight 1 keeps the edge untouched at every level, as in Blender.
struct SubdivisionCrease final {
  StableId vertexA = 0, vertexB = 0;
  float weight = 0;
};
struct SubdivisionSurfaceRecord final {
  std::vector<SubdivisionCrease> creases;
};
struct OperationRecord final {
  StableId stableId = 0;
  OperationCode code = OperationCode::copy;
  StableId inputStableId = 0;
  StableId outputStableId = 0;
  std::array<float, 4> parameters{};
  StableId secondaryInputStableId = 0;
  Transform transform{};
  FieldOperationSettings field{};
  std::shared_ptr<const RetainedMeshData> retainedMesh {};
  std::shared_ptr<const sdfsurface::Request> sdfSurface {};
  // Shade Mesh only: edges marked sharp, as pairs of input vertex indices.
  std::vector<std::array<std::uint32_t, 2>> sharpEdges {};
  std::shared_ptr<const SubdivisionSurfaceRecord> subdivision {};};

struct ValueDescriptor final {
  std::uint32_t schemaVersion = kSchemaVersion;
  CarrierKind carrier = CarrierKind::unspecified;
  StableId stableId = 0, sourceStableId = 0;
  std::uint64_t sourceRevision = 0;
  OverflowPolicy overflow = OverflowPolicy::reject;
  Domain fieldDomain = Domain::unspecified;
  ValueType fieldValueType = ValueType::unspecified;
  Interpolation fieldInterpolation = Interpolation::unspecified;
  std::size_t fieldDomainCardinality = 0;
  std::size_t dispatchCount = 0;
  CarrierData data = GeometryData{};
  std::vector<AttributeData> attributes;
  std::vector<OperationRecord> operations;
  std::vector<spectrum::Binding> spectrumFields;
  std::vector<scorefield::Binding> scoreFields;
};

struct ResourceLimits final {
  std::size_t maxVertices = kMaximumVertices, maxIndices = kMaximumIndices,
              maxPoints = kMaximumPoints, maxCurvePoints = kMaximumCurvePoints,
              maxSplines = kMaximumSplines, maxInstances = kMaximumInstances,
              maxFieldElements = kMaximumFieldElements,
              maxAttributes = kMaximumAttributes,
              maxOperations = kMaximumOperations,
              maxDispatches = kMaximumDispatches,
              maxBufferBytes = kMaximumBufferBytes;
};
// Compiled graphs own these declarations. GCRV alone never grants an external
// source, and frame samples are not part of immutable admission or its cache.
struct DeferredPointsSource final {
  StableId sourceStableId = 0;
  std::uint64_t sourceRevision = 0;
  std::size_t capacity = 0;
  bool operator==(const DeferredPointsSource& other) const noexcept {
    return sourceStableId == other.sourceStableId && sourceRevision == other.sourceRevision
        && capacity == other.capacity;
  }
};
class AdmittedValue;
// These records are evaluator data. Helper-private replay owners authorize their
// installation for native execution; a public declaration alone grants no draw.
struct DeferredBodyGeometrySource final {
  StableId outputStableId=0, pointsOutputStableId=0;
  DeferredPointsSource points;
  StableId terminalStableId=0, sourceStableId=0;
  std::uint64_t sourceRevision=0, structuralRevision=0;
  std::shared_ptr<const AdmittedValue> source;
  std::shared_ptr<const GeometryData> reference;
  std::size_t maxVertices=0, maxIndices=0;
};
struct BodyGeometryFrameData final {
  std::shared_ptr<const GeometryData> geometry;
  std::shared_ptr<const AdmittedValue> source;
  std::uint64_t structuralRevision=0;
  std::int64_t absoluteTick=-1;
};
inline bool validDeferredBodyGeometry(const DeferredBodyGeometrySource&, std::string&);
struct AdmissionContext final {
  std::set<StableId> admittedGeometryOrSceneSources;
  std::map<StableId, DeferredPointsSource> deferredPoints;
  std::map<StableId, ValueDescriptor> evaluatedPoints;
  std::map<StableId, DeferredBodyGeometrySource> deferredBodyGeometry;
  std::map<StableId, BodyGeometryFrameData> evaluatedBodyGeometry;
};
struct BudgetReceipt final {
  StableId valueStableId = 0, sourceStableId = 0;
  std::uint64_t sourceRevision = 0;
  CarrierKind carrier = CarrierKind::unspecified;
  std::size_t vertices = 0, indices = 0, points = 0, curvePoints = 0,
              splines = 0, instances = 0, fieldElements = 0, attributes = 0,
              operations = 0, dispatches = 0, bufferBytes = 0;
  std::size_t sdfSamples = 0, sdfSampleWork = 0, sdfTemporaryBytes = 0;
  std::uint64_t sdfTextRecordVisits = 0, sdfTextSegmentVisits = 0;
  std::uint64_t deterministicDigest = 0;
};

inline bool finite(float v) noexcept { return std::isfinite(v); }
inline bool finite(double v) noexcept { return std::isfinite(v); }
inline bool finite(const Vec3 &v) noexcept {
  return finite(v.x) && finite(v.y) && finite(v.z);
}
inline bool finite(const Quaternion &v) noexcept {
  return finite(v.x) && finite(v.y) && finite(v.z) && finite(v.w);
}
inline bool finite(const Transform &v) noexcept {
  return finite(v.translation) && finite(v.rotation) && finite(v.scale);
}
inline bool checkedAdd(std::size_t a, std::size_t b,
                       std::size_t &out) noexcept {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    return false;
  out = a + b;
  return true;
}
inline bool checkedMul(std::size_t a, std::size_t b,
                       std::size_t &out) noexcept {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    return false;
  out = a * b;
  return true;
}
inline bool addArrayBytes(std::size_t count, std::size_t width,
                          std::size_t &total) noexcept {
  std::size_t n = 0;
  return checkedMul(count, width, n) && checkedAdd(total, n, total);
}

inline bool validateLimits(const ResourceLimits &l, std::string &e) {
  if (l.maxVertices == 0 || l.maxVertices > kMaximumVertices ||
      l.maxIndices == 0 || l.maxIndices > kMaximumIndices || l.maxPoints == 0 ||
      l.maxPoints > kMaximumPoints || l.maxCurvePoints == 0 ||
      l.maxCurvePoints > kMaximumCurvePoints || l.maxSplines == 0 ||
      l.maxSplines > kMaximumSplines || l.maxInstances == 0 ||
      l.maxInstances > kMaximumInstances || l.maxFieldElements == 0 ||
      l.maxFieldElements > kMaximumFieldElements || l.maxAttributes == 0 ||
      l.maxAttributes > kMaximumAttributes || l.maxOperations == 0 ||
      l.maxOperations > kMaximumOperations || l.maxDispatches == 0 ||
      l.maxDispatches > kMaximumDispatches || l.maxBufferBytes == 0 ||
      l.maxBufferBytes > kMaximumBufferBytes) {
    e = "geometry resource limits are invalid";
    return false;
  }
  return true;
}
inline std::size_t valueTypeComponents(ValueType t) noexcept {
  switch (t) {
  case ValueType::floatValue:
  case ValueType::boolean:
  case ValueType::integer:
    return 1;
  case ValueType::vector:
    return 3;
  case ValueType::color:
  case ValueType::rotation:
    return 4;
  default:
    return 0;
  }
}
inline bool validFieldElement(const FieldElement &v, ValueType t) noexcept {
  const auto n = valueTypeComponents(t);
  if (n == 0)
    return false;
  for (std::size_t i = 0; i < n; ++i)
    if (!finite(v.components[i]))
      return false;
  for (std::size_t i = n; i < 4; ++i)
    if (v.components[i] != 0.0)
      return false;
  if (t == ValueType::boolean && v.components[0] != 0.0 &&
      v.components[0] != 1.0)
    return false;
  if (t == ValueType::integer &&
      (std::trunc(v.components[0]) != v.components[0] ||
       std::abs(v.components[0]) > 9007199254740991.0))
    return false;
  return true;
}
inline std::size_t domainCardinality(const BudgetReceipt &r,
                                     Domain d) noexcept {
  switch (d) {
  case Domain::point:
    return r.points;
  case Domain::vertex:
    return r.vertices;
  case Domain::faceCorner:
    return r.indices;
  case Domain::face:
    return r.indices / 3;
  case Domain::curvePoint:
    return r.curvePoints;
  case Domain::spline:
    return r.splines;
  case Domain::instance:
    return r.instances;
  default:
    return 0;
  }
}
namespace owner_detail {
struct CoreZeroParametersPredicate final {
  bool operator()(float value) const { return value == 0.0f; }
};
struct CoreParametersFinitePredicate final {
  bool operator()(float x) const { return finite(x); }
};
struct CoreDisplacementSpectrumPredicate final {
  const OperationRecord& operation;
  template <typename T>
  bool operator()(const T& op) const {
    return op.code==OperationCode::fieldAudioSpectrum && op.outputStableId==operation.secondaryInputStableId;
  }
};
}
inline bool zeroParameters(const OperationRecord &operation) noexcept {
  return std::all_of(operation.parameters.begin(), operation.parameters.end(),
                     owner_detail::CoreZeroParametersPredicate{});
}

inline bool exactIntegerParameter(float value, std::uint32_t minimum,
                                  std::uint32_t &result) noexcept {
  if (!finite(value) || value < static_cast<float>(minimum) ||
      value >= 4294967296.0f ||
      std::trunc(value) != value)
    return false;
  result = static_cast<std::uint32_t>(value);
  return static_cast<float>(result) == value;
}

struct MaterializedInstancePlan final {
  std::map<StableId, GeometryData> geometries;
  std::map<StableId, std::vector<AttributeData>> geometryAttributes;
  std::map<StableId, PointsData> points;
  std::map<StableId, CurvesData> curves;
  InstancesData instances;
  StableId terminalStableId = 0;
  StableId materialStableId = 1;
  std::map<StableId, std::map<std::uint8_t, OperationRecord>> materialTables;
  std::vector<AttributeData> terminalAttributes;
};

inline bool equal(const Vec3 &a, const Vec3 &b) noexcept {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
inline bool equal(const Quaternion &a, const Quaternion &b) noexcept {
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
inline bool equal(const Transform &a, const Transform &b) noexcept {
  return equal(a.translation, b.translation) && equal(a.rotation, b.rotation) &&
         equal(a.scale, b.scale);
}
inline bool equal(const GeometryData &a, const GeometryData &b) noexcept {
  if (a.vertexIds != b.vertexIds || a.indices != b.indices ||
      a.positions.size() != b.positions.size())
    return false;
  for (std::size_t i = 0; i < a.positions.size(); ++i)
    if (!equal(a.positions[i], b.positions[i]))
      return false;
  return true;
}
inline bool equal(const PointsData &a, const PointsData &b) noexcept {
  if (a.points.size() != b.points.size())
    return false;
  for (std::size_t i = 0; i < a.points.size(); ++i) {
    const auto &x = a.points[i], &y = b.points[i];
    if (x.stableId != y.stableId || x.radius != y.radius ||
        !equal(x.position, y.position) || !equal(x.rotation, y.rotation))
      return false;
  }
  return true;
}
inline bool equal(const CurvesData &a, const CurvesData &b) noexcept {
  if (a.points.size() != b.points.size() || a.splines.size() != b.splines.size())
    return false;
  for (std::size_t i = 0; i < a.points.size(); ++i) {
    const auto &x = a.points[i], &y = b.points[i];
    if (x.stableId != y.stableId || x.radius != y.radius || x.tilt != y.tilt ||
        !equal(x.position, y.position)) return false;
  }
  for (std::size_t i = 0; i < a.splines.size(); ++i) {
    const auto &x = a.splines[i], &y = b.splines[i];
    if (x.stableId != y.stableId || x.firstPoint != y.firstPoint ||
        x.pointCount != y.pointCount || x.cyclic != y.cyclic) return false;
  }
  return true;
}
inline bool equal(const InstancesData &a, const InstancesData &b) noexcept {
  if (a.instances.size() != b.instances.size())
    return false;
  for (std::size_t i = 0; i < a.instances.size(); ++i) {
    const auto &x = a.instances[i], &y = b.instances[i];
    if (x.stableId != y.stableId || x.sourceStableId != y.sourceStableId ||
        !equal(x.transform, y.transform))
      return false;
  }
  return true;
}

inline bool materializeCurveGenerator(const OperationRecord &operation,
                                      CurvesData &result, std::size_t maxPoints,
                                      std::string &error) try {
  const bool circle = operation.code == OperationCode::curveCircle;
  std::uint32_t count = 0;
  const auto size = operation.parameters[0];
  if ((!circle && operation.code != OperationCode::curveLine) ||
      !finite(size) || size <= 0 ||
      !exactIntegerParameter(operation.parameters[1], circle ? 3 : 2, count) ||
      count > std::min(maxPoints, kMaximumCurvePoints) ||
      operation.parameters[2] != 0 || operation.parameters[3] != 0) {
    error = "Curve generator dimensions or capacity are invalid";
    return false;
  }
  result.points.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    Vec3 position;
    if (circle) {
      const auto angle = static_cast<float>(i) * 6.2831853071795864769f / count;
      position = {std::cos(angle) * size, std::sin(angle) * size, 0};
    } else {
      position = {(static_cast<float>(i) / (count - 1) - 0.5f) * size, 0, 0};
    }
    result.points.push_back({position, static_cast<StableId>(i + 1), 1, 0});
  }
  result.splines.push_back({1, 0, count, circle});
  return true;
} catch (const std::bad_alloc &) {
  error = "Curve generator could not allocate its points";
  return false;
} catch (const std::length_error &) {
  error = "Curve generator could not allocate its points";
  return false;
}

inline bool materializeCurveRibbon(const CurvesData &curves, float width,
                                   GeometryData &result, std::size_t maxVertices,
                                   std::size_t maxIndices, std::string &error) {
  std::size_t vertices = 0, indices = 0, previousEnd = 0;
  if (!finite(width) || width <= 0 || curves.points.empty() ||
      curves.points.size() > kMaximumCurvePoints || curves.splines.size() > kMaximumSplines ||
      !checkedMul(curves.points.size(), 2, vertices) ||
      vertices > std::min(maxVertices, kMaximumVertices)) {
    error = "Curve to Mesh width or vertex capacity is invalid";
    return false;
  }
  std::set<StableId> pointIds, splineIds;
  for (const auto &point : curves.points)
    if (!finite(point.position) || !finite(point.radius) || point.radius <= 0 ||
        !finite(point.tilt) || !validStableId(point.stableId) ||
        point.stableId > kMaximumStableId / 2 || !pointIds.insert(point.stableId).second) {
      error = "Curve to Mesh point or paired vertex identity is invalid";
      return false;
    }
  for (const auto &spline : curves.splines) {
    std::size_t end = 0, splineIndices = 0;
    if (!validStableId(spline.stableId) || !splineIds.insert(spline.stableId).second ||
        spline.pointCount < (spline.cyclic ? 3u : 2u) || spline.firstPoint != previousEnd ||
        !checkedAdd(spline.firstPoint, spline.pointCount, end) || end > curves.points.size() ||
        !checkedMul(spline.pointCount - (spline.cyclic ? 0u : 1u), 6, splineIndices) ||
        !checkedAdd(indices, splineIndices, indices) || indices > std::min(maxIndices, kMaximumIndices)) {
      error = "Curve to Mesh spline range or index capacity is invalid";
      return false;
    }
    previousEnd = end;
  }
  if (previousEnd != curves.points.size()) {
    error = "Curve to Mesh splines do not cover their points";
    return false;
  }
  GeometryData mesh;
  mesh.positions.reserve(vertices); mesh.vertexIds.reserve(vertices); mesh.indices.reserve(indices);
  for (const auto &spline : curves.splines) {
    for (std::uint32_t i = 0; i < spline.pointCount; ++i) {
      const auto first = spline.firstPoint;
      const auto &point = curves.points[first + i];
      const auto &before = curves.points[first + (i == 0 ? (spline.cyclic ? spline.pointCount - 1 : 0) : i - 1)].position;
      const auto &after = curves.points[first + (i + 1 == spline.pointCount ? (spline.cyclic ? 0 : i) : i + 1)].position;
      double tx = static_cast<double>(after.x) - before.x;
      double ty = static_cast<double>(after.y) - before.y;
      double tz = static_cast<double>(after.z) - before.z;
      const auto length = std::sqrt(tx * tx + ty * ty + tz * tz);
      if (!finite(length) || length == 0) {
        error = "Curve to Mesh requires a non-zero tangent at each point";
        return false;
      }
      tx /= length; ty /= length; tz /= length;
      // Use the XY plane for planar curves, with a second axis for vertical tangents.
      double sx = -ty, sy = tx, sz = 0;
      if (std::abs(tz) > 0.999) { sx = tz; sy = 0; sz = -tx; }
      const auto sideLength = std::sqrt(sx * sx + sy * sy + sz * sz);
      sx /= sideLength; sy /= sideLength; sz /= sideLength;
      const auto cosine = std::cos(static_cast<double>(point.tilt));
      const auto sine = std::sin(static_cast<double>(point.tilt));
      const auto halfWidth = static_cast<double>(width) * point.radius * 0.5;
      const auto dx = (sx * cosine + (ty * sz - tz * sy) * sine) * halfWidth;
      const auto dy = (sy * cosine + (tz * sx - tx * sz) * sine) * halfWidth;
      const auto dz = (sz * cosine + (tx * sy - ty * sx) * sine) * halfWidth;
      for (const auto sign : {-1.0, 1.0}) {
        Vec3 position{static_cast<float>(point.position.x + sign * dx),
                      static_cast<float>(point.position.y + sign * dy),
                      static_cast<float>(point.position.z + sign * dz)};
        if (!finite(position)) { error = "Curve to Mesh position is not finite"; return false; }
        mesh.positions.push_back(position);
      }
      mesh.vertexIds.push_back(point.stableId * 2 - 1);
      mesh.vertexIds.push_back(point.stableId * 2);
    }
    const auto segments = spline.pointCount - (spline.cyclic ? 0u : 1u);
    for (std::uint32_t i = 0; i < segments; ++i) {
      const auto a = (spline.firstPoint + i) * 2;
      const auto b = (spline.firstPoint + (i + 1) % spline.pointCount) * 2;
      mesh.indices.insert(mesh.indices.end(), {a, b, a + 1, b, b + 1, a + 1});
    }
  }
  for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
    const auto &a = mesh.positions[mesh.indices[i]], &b = mesh.positions[mesh.indices[i + 1]],
               &c = mesh.positions[mesh.indices[i + 2]];
    const double ux = static_cast<double>(b.x) - a.x, uy = static_cast<double>(b.y) - a.y,
                 uz = static_cast<double>(b.z) - a.z;
    const double vx = static_cast<double>(c.x) - a.x, vy = static_cast<double>(c.y) - a.y,
                 vz = static_cast<double>(c.z) - a.z;
    const auto nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    if (nx * nx + ny * ny + nz * nz == 0) {
      error = "Curve to Mesh would produce a degenerate triangle";
      return false;
    }
  }
  result = std::move(mesh);
  return true;
}

inline bool validateCurves(const CurvesData &curves, std::string &error) {
  if (curves.points.empty() || curves.points.size() > kMaximumCurvePoints ||
      curves.splines.empty() || curves.splines.size() > kMaximumSplines) {
    error = "Curve operation requires bounded points and splines"; return false;
  }
  std::set<StableId> pointIds, splineIds;
  for (const auto &point : curves.points)
    if (!validStableId(point.stableId) || !pointIds.insert(point.stableId).second ||
        !finite(point.position) || !finite(point.radius) || point.radius <= 0 || !finite(point.tilt)) {
      error = "Curve operation requires finite points, positive radii and unique identities"; return false;
    }
  std::size_t end = 0;
  for (const auto &spline : curves.splines) {
    if (!validStableId(spline.stableId) || !splineIds.insert(spline.stableId).second ||
        spline.firstPoint != end || spline.pointCount < (spline.cyclic ? 3u : 2u) ||
        !checkedAdd(end, spline.pointCount, end) || end > curves.points.size()) {
      error = "Curve operation has an invalid spline range"; return false;
    }
  }
  if (end != curves.points.size()) { error = "Curve splines do not cover their points"; return false; }
  return true;
}

inline bool curveLengths(const CurvesData &curves, const SplineRecord &spline,
                          std::vector<double> &lengths, std::string &error) {
  const auto segments = spline.pointCount - (spline.cyclic ? 0u : 1u);
  lengths.assign(segments + 1, 0);
  for (std::uint32_t i = 0; i < segments; ++i) {
    const auto &a = curves.points[spline.firstPoint + i].position;
    const auto &b = curves.points[spline.firstPoint + (i + 1) % spline.pointCount].position;
    const double x = double(b.x)-a.x, y = double(b.y)-a.y, z = double(b.z)-a.z;
    const auto length = std::sqrt(x*x+y*y+z*z);
    if (!finite(length) || length <= 0) { error = "Curve contains a zero-length segment"; return false; }
    lengths[i+1] = lengths[i] + length;
  }
  return true;
}

inline CurvePointRecord sampleCurve(const CurvesData &curves, const SplineRecord &spline,
                                     const std::vector<double> &lengths, double parameter) {
  const auto distance = std::clamp(parameter, 0.0, 1.0) * lengths.back();
  const auto upper = std::upper_bound(lengths.begin(), lengths.end(), distance);
  const auto segment = std::min<std::size_t>(std::max<std::size_t>(1, upper-lengths.begin())-1,
                                             lengths.size()-2);
  const auto t = (distance-lengths[segment])/(lengths[segment+1]-lengths[segment]);
  const auto &a = curves.points[spline.firstPoint + segment];
  const auto &b = curves.points[spline.firstPoint + (segment+1)%spline.pointCount];
  const auto mix = [t](float x, float y) { return static_cast<float>(double(x)+(double(y)-x)*t); };
  return {{mix(a.position.x,b.position.x),mix(a.position.y,b.position.y),mix(a.position.z,b.position.z)},
          a.stableId,mix(a.radius,b.radius),mix(a.tilt,b.tilt)};
}

inline bool curveTangent(const CurvesData &curves, const SplineRecord &spline,
                          std::uint32_t index, Vec3 &tangent, std::string &error) {
  const auto before = index == 0 ? (spline.cyclic ? spline.pointCount-1 : 0) : index-1;
  const auto after = index+1 == spline.pointCount ? (spline.cyclic ? 0 : index) : index+1;
  const auto &a = curves.points[spline.firstPoint+before].position;
  const auto &b = curves.points[spline.firstPoint+after].position;
  const double x=double(b.x)-a.x, y=double(b.y)-a.y, z=double(b.z)-a.z;
  const auto length=std::sqrt(x*x+y*y+z*z);
  if (!finite(length) || length<=0) { error="Curve requires a nonzero tangent"; return false; }
  tangent={static_cast<float>(x/length),static_cast<float>(y/length),static_cast<float>(z/length)};
  return true;
}

inline constexpr std::uint32_t kCurveAnimateGrow = 0, kCurveAnimateTravel = 1, kCurveAnimateNoise = 2;
// The shortest visible part of a growing or travelling curve, as a fraction of its length.
inline constexpr double kCurveAnimateMinimumWindow = 0.01;

// Loop phase in [0,1) of a curveAnimate operation at the given timeline seconds.
inline double curveAnimatePhase(const OperationRecord &operation, double seconds) noexcept {
  const auto cycles = seconds / double(operation.parameters[1])
      + (operation.parameters[0] == kCurveAnimateNoise ? 0.0 : double(operation.parameters[3]));
  return cycles - std::floor(cycles);
}

// Visible window [start, end] of a grow or travel animation at a loop phase. A cyclic
// spline travels around its loop, so its window may run past 1 and wraps.
inline std::pair<double,double> curveAnimateWindow(std::uint32_t mode, double amount, double phase,
                                                   bool cyclic) noexcept {
  if (mode == kCurveAnimateGrow) return {0.0, std::max(phase, kCurveAnimateMinimumWindow)};
  const auto length = std::clamp(amount, kCurveAnimateMinimumWindow, 1.0);
  if (cyclic) return {phase, phase + length};
  // An open spline's window enters at the start and leaves past the end each loop.
  const auto head = phase * (1.0 + length);
  auto start = std::clamp(head - length, 0.0, 1.0), end = std::clamp(head, 0.0, 1.0);
  if (end - start < kCurveAnimateMinimumWindow) {
    if (end < 1.0) end = std::min(1.0, start + kCurveAnimateMinimumWindow);
    else start = end - kCurveAnimateMinimumWindow;
  }
  return {start, end};
}

// Smooth looping displacement in [-1,1] for one axis: two sines along the spline whose
// phases advance with the loop, offset per spline and axis so splines move independently.
inline double curveAnimateNoise(StableId spline, std::uint32_t axis, double coordinate,
                                double cycles, double phase) noexcept {
  std::uint32_t state = static_cast<std::uint32_t>(spline * 2654435761ull) ^ (axis * 0x68e31da4u);
  const auto offset = [&]() {
    state += 0x9e3779b9u;
    auto value = state;
    value = (value ^ (value >> 16u)) * 0x85ebca6bu;
    value = (value ^ (value >> 13u)) * 0xc2b2ae35u;
    return double((value ^ (value >> 16u)) >> 8u) / 16777216.0;
  };
  const auto a = offset(), b = offset();
  constexpr double tau = 6.283185307179586;
  return 0.6 * std::sin(tau * (cycles * coordinate + phase + a))
       + 0.4 * std::sin(tau * (2.0 * cycles * coordinate - phase + b));
}

inline bool materializeCurveEdit(const CurvesData &source, const OperationRecord &operation,
                                 const FieldData *field, CurvesData &result,
                                 std::size_t capacity, std::string &error, double seconds = 0) {
  if (!validateCurves(source,error)) return false;
  const auto &p=operation.parameters;
  std::uint32_t resampleCount=0, axis=0;
  const bool resample=operation.code==OperationCode::curveResample;
  const bool trim=operation.code==OperationCode::curveTrim;
  const bool radius=operation.code==OperationCode::curveSetRadius;
  const bool tilt=operation.code==OperationCode::curveSetTilt;
  const bool wave=operation.code==OperationCode::curveWave;
  const bool animate=operation.code==OperationCode::curveAnimate;
  std::uint32_t animateMode=0;
  if (!std::all_of(p.begin(),p.end(),[](float value){return finite(value);}) ||
      (animate && (!exactIntegerParameter(p[0],0,animateMode) || animateMode>kCurveAnimateNoise
                   || !(p[1]>0) || p[1]>86400 || !std::isfinite(seconds)
                   || (animateMode==kCurveAnimateGrow && p[2]!=0)
                   || (animateMode==kCurveAnimateTravel && (p[2]<=0 || p[2]>1))
                   || (animateMode==kCurveAnimateNoise && std::abs(p[2])>1000000)
                   || (animateMode!=kCurveAnimateNoise && (p[3]<0 || p[3]>1))
                   || (animateMode==kCurveAnimateNoise && (p[3]<0 || p[3]>64)))) ||
      (resample && (!exactIntegerParameter(p[0],2,resampleCount) || p[1]!=0 || p[2]!=0 || p[3]!=0)) ||
      (trim && (p[0]<0 || p[1]>1 || p[0]>=p[1] || p[2]!=0 || p[3]!=0)) ||
      ((radius || tilt) && (p[2]!=0 || p[3]!=0 || (radius && (p[0]<=0 || p[1]<=0)))) ||
      (wave && (p[1]<0 || p[1]>64 || p[2]<0 || p[2]>1 ||
                !exactIntegerParameter(p[3],0,axis) || axis>2)) ||
      (!resample && !trim && !radius && !tilt && !wave && !animate) ||
      (field && (!(radius || tilt) || (field->elements.size()!=1 && field->elements.size()!=source.points.size())))) {
    error="Curve operation parameters or field cardinality are invalid"; return false;
  }
  std::size_t pointCount=source.points.size();
  if ((resample && !checkedMul(source.splines.size(),resampleCount,pointCount)) ||
      pointCount>std::min(capacity,kMaximumCurvePoints)) {
    error="Curve operation exceeds its point capacity"; return false;
  }
  CurvesData edited;
  edited.points.reserve(pointCount); edited.splines.reserve(source.splines.size());
  for (const auto &spline : source.splines) {
    std::vector<double> lengths;
    if (!curveLengths(source,spline,lengths,error)) return false;
    const auto count=resample ? resampleCount : spline.pointCount;
    const auto phase=animate ? curveAnimatePhase(operation,seconds) : 0.0;
    // Grow and travel keep each point slot and move it inside the visible window.
    const bool window=animate && animateMode!=kCurveAnimateNoise;
    const auto visible=window ? curveAnimateWindow(animateMode,p[2],phase,spline.cyclic)
                              : std::pair<double,double>{0.0,1.0};
    const bool cyclic=spline.cyclic && (!trim || (p[0]==0 && p[1]==1)) && !window;
    if (cyclic && count<3) { error="Cyclic curves require at least three points"; return false; }
    edited.splines.push_back({spline.stableId,static_cast<std::uint32_t>(edited.points.size()),count,cyclic});
    for (std::uint32_t i=0;i<count;++i) {
      const double parameter=double(i)/(cyclic ? count : count-1);
      auto windowed=visible.first+(visible.second-visible.first)*parameter;
      if (windowed>1) windowed-=1;
      auto point=(resample || trim || window)
          ? sampleCurve(source,spline,lengths,trim ? p[0]+(p[1]-p[0])*parameter : window ? windowed : parameter)
          : source.points[spline.firstPoint+i];
      // Resampling creates sample slots; trim and deformations retain those slots across edits.
      point.stableId=resample ? edited.points.size()+1 : source.points[spline.firstPoint+i].stableId;
      if (radius || tilt) {
        const auto coordinate=lengths[i]/lengths.back();
        double amount=p[0]+(p[1]-p[0])*coordinate;
        if (field) {
          const auto &element=field->elements[field->elements.size()==1 ? 0 : spline.firstPoint+i];
          if (!validFieldElement(element,ValueType::floatValue)) {
            error="Curve attribute field contains an invalid value"; return false;
          }
          amount=element.components[0];
        }
        if (!finite(amount) || (radius && amount<=0) || !finite(static_cast<float>(amount))) {
          error="Curve radius must be positive and curve attributes must be finite"; return false;
        }
        if (radius) point.radius=static_cast<float>(amount); else point.tilt=static_cast<float>(amount);
      }
      if (wave) {
        const auto displacement=static_cast<float>(p[0]*std::sin(6.283185307179586*
            (p[1]*lengths[i]/lengths.back()+p[2])));
        if (axis==0) point.position.x+=displacement;
        else if (axis==1) point.position.y+=displacement;
        else point.position.z+=displacement;
      }
      if (animate && animateMode==kCurveAnimateNoise) {
        const auto coordinate=lengths[std::min<std::size_t>(i,lengths.size()-1)]/lengths.back();
        point.position.x+=static_cast<float>(p[2]*curveAnimateNoise(spline.stableId,0,coordinate,p[3],phase));
        point.position.y+=static_cast<float>(p[2]*curveAnimateNoise(spline.stableId,1,coordinate,p[3],phase));
        point.position.z+=static_cast<float>(p[2]*curveAnimateNoise(spline.stableId,2,coordinate,p[3],phase));
      }
      if (!finite(point.position)) { error="Curve position exceeds finite bounds"; return false; }
      edited.points.push_back(point);
    }
  }
  result=std::move(edited); error.clear(); return true;
}

inline bool materializeCurvePoints(const CurvesData &source, bool endpointsOnly, float scale,
                                    PointsData &result, std::size_t capacity, std::string &error) {
  if (!validateCurves(source,error)) return false;
  if (!finite(scale) || scale<=0) { error="Curve to Points requires positive finite scale"; return false; }
  PointsData points;
  for (const auto &spline : source.splines)
    for (std::uint32_t i=0;i<spline.pointCount;++i) {
      if (endpointsOnly && (spline.cyclic || (i!=0 && i+1!=spline.pointCount))) continue;
      if (points.points.size()>=std::min(capacity,kMaximumPoints)) {
        error="Curve to Points exceeds its capacity"; return false;
      }
      const auto &point=source.points[spline.firstPoint+i];
      Vec3 tangent;
      if (!curveTangent(source,spline,i,tangent,error)) return false;
      // Orient local +Z along the tangent, then apply the curve's local roll.
      Quaternion rotation= tangent.z < -0.999999f ? Quaternion{1,0,0,0}
          : Quaternion{-tangent.y,tangent.x,0,1+tangent.z};
      const auto norm=std::sqrt(rotation.x*rotation.x+rotation.y*rotation.y+rotation.z*rotation.z+rotation.w*rotation.w);
      rotation={rotation.x/norm,rotation.y/norm,rotation.z/norm,rotation.w/norm};
      const auto sine=std::sin(point.tilt*0.5f), cosine=std::cos(point.tilt*0.5f);
      rotation={rotation.x*cosine+rotation.y*sine,rotation.y*cosine-rotation.x*sine,
                rotation.w*sine,rotation.w*cosine};
      if (!finite(point.radius*scale)) { error="Curve point scale exceeds finite bounds"; return false; }
      points.points.push_back({point.position,point.stableId,point.radius*scale,rotation});
    }
  result=std::move(points); error.clear(); return true;
}

inline bool materializeCurveTube(const CurvesData &curves, float radius, std::uint32_t sides,
                                  GeometryData &result, std::size_t vertexCapacity,
                                  std::size_t indexCapacity, std::string &error) {
  if (!validateCurves(curves,error)) return false;
  std::size_t vertices=0, indices=0;
  if (!finite(radius) || radius<=0 || sides<3 || sides>64 ||
      !checkedMul(curves.points.size(),sides,vertices) || vertices>std::min(vertexCapacity,kMaximumVertices)) {
    error="Curve tube radius, profile sides or vertex capacity is invalid"; return false;
  }
  for (const auto &spline : curves.splines) {
    std::size_t count=0;
    if (!checkedMul(spline.pointCount-(spline.cyclic ? 0u : 1u),sides*6,count) ||
        !checkedAdd(indices,count,indices) || indices>std::min(indexCapacity,kMaximumIndices)) {
      error="Curve tube exceeds its index capacity"; return false;
    }
  }
  GeometryData mesh;
  mesh.positions.reserve(vertices); mesh.vertexIds.reserve(vertices); mesh.indices.reserve(indices);
  for (const auto &spline : curves.splines) {
    for (std::uint32_t i=0;i<spline.pointCount;++i) {
      const auto &point=curves.points[spline.firstPoint+i];
      if (point.stableId>kMaximumStableId/64) { error="Curve tube point identity exceeds its profile range"; return false; }
      Vec3 tangent;
      if (!curveTangent(curves,spline,i,tangent,error)) return false;
      Vec3 side=std::abs(tangent.z)>0.999f ? Vec3{tangent.z,0,-tangent.x} : Vec3{-tangent.y,tangent.x,0};
      const auto length=std::sqrt(side.x*side.x+side.y*side.y+side.z*side.z);
      side={side.x/length,side.y/length,side.z/length};
      const Vec3 normal{tangent.y*side.z-tangent.z*side.y,tangent.z*side.x-tangent.x*side.z,
                        tangent.x*side.y-tangent.y*side.x};
      for (std::uint32_t j=0;j<sides;++j) {
        const auto angle=6.283185307179586*double(j)/sides+point.tilt;
        const auto c=std::cos(angle)*radius*point.radius, s=std::sin(angle)*radius*point.radius;
        Vec3 position{static_cast<float>(point.position.x+c*side.x+s*normal.x),
                      static_cast<float>(point.position.y+c*side.y+s*normal.y),
                      static_cast<float>(point.position.z+c*side.z+s*normal.z)};
        if (!finite(position)) { error="Curve tube position exceeds finite bounds"; return false; }
        mesh.positions.push_back(position); mesh.vertexIds.push_back((point.stableId-1)*64+j+1);
      }
    }
    for (std::uint32_t i=0;i<spline.pointCount-(spline.cyclic ? 0u : 1u);++i)
      for (std::uint32_t j=0;j<sides;++j) {
        const auto a=(spline.firstPoint+i)*sides+j;
        const auto b=(spline.firstPoint+(i+1)%spline.pointCount)*sides+j;
        const auto c=(spline.firstPoint+i)*sides+(j+1)%sides;
        const auto d=(spline.firstPoint+(i+1)%spline.pointCount)*sides+(j+1)%sides;
        mesh.indices.insert(mesh.indices.end(),{a,c,b,b,c,d});
      }
  }
  for (std::size_t i=0;i<mesh.indices.size();i+=3) {
    const auto &a=mesh.positions[mesh.indices[i]], &b=mesh.positions[mesh.indices[i+1]],
               &c=mesh.positions[mesh.indices[i+2]];
    const double ux=double(b.x)-a.x, uy=double(b.y)-a.y, uz=double(b.z)-a.z;
    const double vx=double(c.x)-a.x, vy=double(c.y)-a.y, vz=double(c.z)-a.z;
    const auto nx=uy*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy*vx;
    if (nx*nx+ny*ny+nz*nz==0) { error="Curve tube would create a degenerate triangle"; return false; }
  }
  result=std::move(mesh); error.clear(); return true;
}

// A camera grade widens grid cells with distance from the centre: cell k out
// from the middle is spacing * (1 + grade * (2k + 1)) wide, so cell size grows
// linearly with distance and screen density stays even for a camera above the
// centre. Grade 0 is the regular grid.
inline constexpr float kMaximumGridGrade = 1.0f;
inline float gridCoordinate(std::uint32_t index, std::uint32_t count, float spacing, float grade) noexcept {
  const double cells = static_cast<double>(index) - (count - 1) * 0.5;
  return static_cast<float>(cells * spacing * (1.0 + grade * std::abs(cells)));
}

inline bool materializeGridOperation(const OperationRecord &operation,
                                     GeometryData &result,
                                     std::string &error) try {
  std::uint32_t columns = 0, rows = 0;
  const auto spacing = operation.parameters[2], grade = operation.parameters[3];
  if (!exactIntegerParameter(operation.parameters[0], 2, columns) ||
      !exactIntegerParameter(operation.parameters[1], 2, rows) ||
      !finite(spacing) || spacing <= 0.0f || !(grade >= 0.0f && grade <= kMaximumGridGrade)) {
    error = "Grid operation dimensions or spacing are invalid";
    return false;
  }
  std::size_t vertices = 0, cells = 0, indices = 0;
  if (!checkedMul(columns, rows, vertices) ||
      !checkedMul(columns - 1, rows - 1, cells) || !checkedMul(cells, 6, indices) ||
      vertices > kMaximumVertices || indices > kMaximumIndices) {
    error = "Grid operation exceeds immutable geometry bounds";
    return false;
  }
  result.positions.reserve(vertices);
  result.vertexIds.reserve(vertices);
  result.indices.reserve(indices);
  for (std::uint32_t y = 0; y < rows; ++y)
    for (std::uint32_t x = 0; x < columns; ++x) {
      const auto id = static_cast<StableId>(y) * columns + x + 1;
      result.positions.push_back({gridCoordinate(x, columns, spacing, grade),
                                  gridCoordinate(y, rows, spacing, grade), 0.0f});
      result.vertexIds.push_back(id);
    }
  for (std::uint32_t y = 0; y + 1 < rows; ++y)
    for (std::uint32_t x = 0; x + 1 < columns; ++x) {
      const auto a = y * columns + x, b = a + 1, c = a + columns,
                 d = c + 1;
      result.indices.insert(result.indices.end(), {a, b, c, b, d, c});
    }
  return true;
} catch (const std::bad_alloc &) {
  error = "Grid operation could not allocate its geometry";
  return false;
} catch (const std::length_error &) {
  error = "Grid operation could not allocate its geometry";
  return false;
}

#if defined(__GNUC__)
#pragma GCC diagnostic push
// These switches name the operation codes they accept and reject the rest in default.
#pragma GCC diagnostic ignored "-Wswitch-enum"
#endif
// 10 * 4^14 + 2 vertices fit a 32-bit index and level 15 doesn't.
inline constexpr std::uint32_t kMaximumIcosphereSubdivisions = 14;
inline bool materializeMeshGenerator(const OperationRecord &operation,
                                     GeometryData &result,
                                     std::size_t maxVertices,
                                     std::size_t maxIndices,
                                     std::string &error) try {
  const auto &p = operation.parameters;
  std::uint32_t segments = 0, rings = 0, subdivisions = 0;
  std::size_t vertices = 0, indices = 0;
  const auto positive = [](float x) { return finite(x) && x > 0.0f; };
  bool valid = std::all_of(p.begin(), p.end(), [](float x) { return finite(x); });
  switch (operation.code) {
  case OperationCode::plane:
    valid = valid && positive(p[0]) && positive(p[1]) && p[2] == 0 && p[3] == 0;
    vertices = 4; indices = 6;
    break;
  case OperationCode::cube:
    valid = valid && positive(p[0]) && positive(p[1]) && positive(p[2]) && p[3] == 0;
    vertices = 24; indices = 36;
    break;
  case OperationCode::sphere:
    valid = valid && positive(p[0]) && p[3] == 0
        && exactIntegerParameter(p[1], 3, segments)
        && exactIntegerParameter(p[2], 2, rings)
        && checkedMul(segments, rings - 1, vertices)
        && checkedMul(vertices, 6, indices);
    vertices += 2;
    break;
  case OperationCode::icosphere:
    valid = valid && positive(p[0]) && p[2] == 0 && p[3] == 0
        && exactIntegerParameter(p[1], 0, subdivisions) && subdivisions <= kMaximumIcosphereSubdivisions;
    if (valid) {
      const auto faces = std::size_t{20} << (2 * subdivisions);
      vertices = faces / 2 + 2; indices = faces * 3;
    }
    break;
  case OperationCode::cylinder:
    // p[3] is the radius of a hole down the axis. 0 is the solid cylinder.
    valid = valid && positive(p[0]) && positive(p[1]) && p[3] >= 0 && p[3] < p[0]
        && exactIntegerParameter(p[2], 3, segments)
        && checkedMul(segments, p[3] > 0 ? 8 : 4, vertices)
        && checkedMul(segments, p[3] > 0 ? 24 : 12, indices);
    if (p[3] == 0)
      vertices += 2;
    break;
  default:
    valid = false;
    break;
  }
  if (!valid) {
    error = "Mesh generator parameters are invalid";
    return false;
  }
  if (vertices > std::min(maxVertices, kMaximumVertices)
      || indices > std::min(maxIndices, kMaximumIndices)) {
    error = "Mesh generator exceeds its admitted geometry capacity";
    return false;
  }
  GeometryData mesh;
  mesh.positions.reserve(vertices);
  mesh.vertexIds.reserve(vertices);
  mesh.indices.reserve(indices);
  const auto vertex = [&](Vec3 position) {
    const auto index = static_cast<std::uint32_t>(mesh.positions.size());
    mesh.positions.push_back(position);
    mesh.vertexIds.push_back(static_cast<StableId>(index) + 1);
    return index;
  };
  const auto triangle = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c) {
    mesh.indices.insert(mesh.indices.end(), {a, b, c});
  };
  constexpr double pi = 3.14159265358979323846;
  if (operation.code == OperationCode::plane) {
    vertex({-p[0] * 0.5f, -p[1] * 0.5f, 0});
    vertex({ p[0] * 0.5f, -p[1] * 0.5f, 0});
    vertex({-p[0] * 0.5f,  p[1] * 0.5f, 0});
    vertex({ p[0] * 0.5f,  p[1] * 0.5f, 0});
    triangle(0, 1, 2); triangle(1, 3, 2);
    // Corner uv, so the plate's radial groove field measures from its centre.
    mesh.uv.push_back({0, 0, 0});
    mesh.uv.push_back({1, 0, 0});
    mesh.uv.push_back({0, 1, 0});
    mesh.uv.push_back({1, 1, 0});
  } else if (operation.code == OperationCode::cube) {
    // Separate face vertices retain hard edges in the native normal pass.
    constexpr std::array<std::array<int, 4>, 6> faces {{{{0, 2, 3, 1}},
        {{4, 5, 7, 6}}, {{0, 1, 5, 4}}, {{2, 6, 7, 3}},
        {{0, 4, 6, 2}}, {{1, 3, 7, 5}}}};
    for (const auto &face : faces) {
      const auto first = static_cast<std::uint32_t>(mesh.positions.size());
      for (const auto corner : face)
        vertex({(corner & 1 ? 0.5f : -0.5f) * p[0],
                (corner & 2 ? 0.5f : -0.5f) * p[1],
                (corner & 4 ? 0.5f : -0.5f) * p[2]});
      triangle(first, first + 1, first + 2);
      triangle(first, first + 2, first + 3);
    }
  } else if (operation.code == OperationCode::sphere) {
    // Longitude and latitude uv: u runs once around the axis, v runs from the
    // south pole (0) to the north pole (1). dP/du is the latitude direction on
    // every triangle, so a grating whose reciprocal axis is v has its grooves
    // on latitude rings. The topology has no duplicated seam column, so the
    // last column's u steps back from (segments - 1) / segments to 0; that
    // column keeps a non-degenerate uv frame with its tangent reversed along
    // the same latitude edge. Each pole is one vertex at u 0.5.
    vertex({0, p[0], 0});
    mesh.uv.push_back({0.5f, 1, 0});
    for (std::uint32_t row = 1; row < rings; ++row) {
      const auto latitude = pi * row / rings;
      for (std::uint32_t column = 0; column < segments; ++column) {
        const auto longitude = 2 * pi * column / segments;
        vertex({static_cast<float>(p[0] * std::sin(latitude) * std::cos(longitude)),
                static_cast<float>(p[0] * std::cos(latitude)),
                static_cast<float>(p[0] * std::sin(latitude) * std::sin(longitude))});
        mesh.uv.push_back({static_cast<float>(column) / static_cast<float>(segments),
                           1.0f - static_cast<float>(row) / static_cast<float>(rings), 0});
      }
    }
    const auto bottom = vertex({0, -p[0], 0});
    mesh.uv.push_back({0.5f, 0, 0});
    for (std::uint32_t column = 0; column < segments; ++column) {
      const auto next = (column + 1) % segments;
      triangle(0, 1 + next, 1 + column);
      for (std::uint32_t row = 0; row + 2 < rings; ++row) {
        const auto a = 1 + row * segments + column, b = 1 + row * segments + next;
        triangle(a, b, a + segments);
        triangle(b, b + segments, a + segments);
      }
      triangle(bottom, 1 + (rings - 2) * segments + column,
               1 + (rings - 2) * segments + next);
    }
  } else if (operation.code == OperationCode::icosphere) {
    const float t = static_cast<float>((1.0 + std::sqrt(5.0)) * 0.5);
    const auto project = [&](Vec3 v) {
      const auto length = std::sqrt(static_cast<double>(v.x) * v.x
          + static_cast<double>(v.y) * v.y + static_cast<double>(v.z) * v.z);
      return Vec3{static_cast<float>(v.x / length * p[0]),
                  static_cast<float>(v.y / length * p[0]),
                  static_cast<float>(v.z / length * p[0])};
    };
    for (const auto &v : std::array<Vec3, 12>{{{-1,t,0},{1,t,0},{-1,-t,0},{1,-t,0},
        {0,-1,t},{0,1,t},{0,-1,-t},{0,1,-t},{t,0,-1},{t,0,1},{-t,0,-1},{-t,0,1}}})
      vertex(project(v));
    mesh.indices = {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11,
                    1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
                    3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9,
                    4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1};
    for (std::uint32_t level = 0; level < subdivisions; ++level) {
      std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> midpoints;
      const auto midpoint = [&](std::uint32_t a, std::uint32_t b) {
        const auto key = std::make_pair(std::min(a, b), std::max(a, b));
        const auto found = midpoints.find(key);
        if (found != midpoints.end()) return found->second;
        const auto &x = mesh.positions[a], &y = mesh.positions[b];
        const auto index = vertex(project({x.x * 0.5f + y.x * 0.5f,
            x.y * 0.5f + y.y * 0.5f, x.z * 0.5f + y.z * 0.5f}));
        midpoints.emplace(key, index);
        return index;
      };
      auto previous = std::move(mesh.indices);
      mesh.indices.clear();
      mesh.indices.reserve(previous.size() * 4);
      for (std::size_t i = 0; i < previous.size(); i += 3) {
        const auto a = previous[i], b = previous[i + 1], c = previous[i + 2];
        const auto ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
        triangle(a, ab, ca); triangle(b, bc, ab);
        triangle(c, ca, bc); triangle(ab, bc, ca);
      }
    }
  } else if (operation.code == OperationCode::cylinder) {
    // Cap rings are separate so the cap normals stay flat.
    // Rings 0 and 1 are the side: u runs once around the axis, v from the
    // bottom (0) to the top (1). Rings 2 and 3 are the caps, in polar uv: u is
    // the same angle and v is the radius, 0 at the hub and 1 on the rim. dP/du
    // is the circumferential direction on the side and on both caps, so a
    // grating whose reciprocal axis is v has concentric grooves on the caps
    // and rings around the side. The seam column's u steps back to 0, as on
    // the sphere, and each hub is one vertex at u 0.5. With a hole (p[3] > 0)
    // there are no hubs. Rings 4 and 5 are the caps' inner edges, with v at the
    // hole's share of the radius so the polar uv stays the same, and rings 6
    // and 7 are the hole's wall.
    const bool hole = p[3] > 0;
    for (std::uint32_t ring = 0; ring < (hole ? 8u : 4u); ++ring)
      for (std::uint32_t column = 0; column < segments; ++column) {
        const auto angle = 2 * pi * column / segments;
        const auto radius = ring < 4 ? p[0] : p[3];
        vertex({static_cast<float>(radius * std::cos(angle)),
                (ring % 2 == 0 ? -0.5f : 0.5f) * p[1],
                static_cast<float>(radius * std::sin(angle))});
        const auto v = ring == 4 || ring == 5 ? p[3] / p[0]
                     : ring == 0 || ring == 6 ? 0.0f : 1.0f;
        mesh.uv.push_back({static_cast<float>(column) / static_cast<float>(segments),
                           static_cast<float>(v), 0});
      }
    if (hole) {
      for (std::uint32_t column = 0; column < segments; ++column) {
        const auto next = (column + 1) % segments;
        const auto at = [&](std::uint32_t ring, std::uint32_t c) { return ring * segments + c; };
        triangle(at(0, column), at(1, column), at(0, next));
        triangle(at(0, next), at(1, column), at(1, next));
        triangle(at(4, column), at(2, column), at(2, next));
        triangle(at(4, column), at(2, next), at(4, next));
        triangle(at(5, column), at(3, next), at(3, column));
        triangle(at(5, column), at(5, next), at(3, next));
        triangle(at(6, column), at(6, next), at(7, column));
        triangle(at(6, next), at(7, next), at(7, column));
      }
    } else {
      const auto bottom = vertex({0, -0.5f * p[1], 0});
      const auto top = vertex({0, 0.5f * p[1], 0});
      mesh.uv.push_back({0.5f, 0, 0});
      mesh.uv.push_back({0.5f, 0, 0});
      for (std::uint32_t column = 0; column < segments; ++column) {
        const auto next = (column + 1) % segments;
        triangle(column, column + segments, next);
        triangle(next, column + segments, next + segments);
        triangle(bottom, 2 * segments + column, 2 * segments + next);
        triangle(top, 3 * segments + next, 3 * segments + column);
      }
    }
  }
  if (mesh.positions.size() != vertices || mesh.indices.size() != indices
      || (!mesh.uv.empty() && mesh.uv.size() != mesh.positions.size())
      || !std::all_of(mesh.positions.begin(), mesh.positions.end(),
                      [](const Vec3 &v) { return finite(v); })) {
    error = "Mesh generator result is outside its declared bounds";
    return false;
  }
  result = std::move(mesh);
  error.clear();
  return true;
} catch (const std::bad_alloc &) {
  error = "Mesh generator could not allocate its geometry";
  return false;
} catch (const std::length_error &) {
  error = "Mesh generator could not allocate its geometry";
  return false;
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

inline Vec3 transformVector(Vec3 p, const Transform &transform) noexcept {
  const auto &q = transform.rotation;
  const double norm = double(q.x)*q.x + double(q.y)*q.y + double(q.z)*q.z + double(q.w)*q.w;
  const double x = double(p.x)*transform.scale.x, y = double(p.y)*transform.scale.y,
               z = double(p.z)*transform.scale.z;
  return {static_cast<float>((1-2*(double(q.y)*q.y+double(q.z)*q.z)/norm)*x
              + 2*(double(q.x)*q.y-double(q.w)*q.z)/norm*y
              + 2*(double(q.x)*q.z+double(q.w)*q.y)/norm*z),
          static_cast<float>(2*(double(q.x)*q.y+double(q.w)*q.z)/norm*x
              + (1-2*(double(q.x)*q.x+double(q.z)*q.z)/norm)*y
              + 2*(double(q.y)*q.z-double(q.w)*q.x)/norm*z),
          static_cast<float>(2*(double(q.x)*q.z-double(q.w)*q.y)/norm*x
              + 2*(double(q.y)*q.z+double(q.w)*q.x)/norm*y
              + (1-2*(double(q.x)*q.x+double(q.y)*q.y)/norm)*z)};
}

inline bool materializeTransform(GeometryData &mesh, const Transform &transform,
                                  std::string &error) {
  const auto &q = transform.rotation;
  if (!finite(transform) || double(q.x)*q.x + double(q.y)*q.y + double(q.z)*q.z + double(q.w)*q.w <= 0) {
    error = "Transform Geometry requires a finite transform and nonzero orientation";
    return false;
  }
  for (auto &position : mesh.positions) {
    position = transformVector(position, transform);
    position.x += transform.translation.x;
    position.y += transform.translation.y;
    position.z += transform.translation.z;
    if (!finite(position)) {
      error = "Transform Geometry exceeds finite position bounds";
      return false;
    }
  }
  error.clear();
  return true;
}

inline bool geometryBounds(const GeometryData &mesh, Vec3 &minimum, Vec3 &maximum,
                            std::string &error) {
  if (mesh.positions.empty() || mesh.positions.size() > kMaximumVertices) {
    error = "Bounding Box requires nonempty bounded Geometry3D";
    return false;
  }
  minimum = maximum = mesh.positions.front();
  for (const auto &p : mesh.positions) {
    if (!finite(p)) { error = "Bounding Box contains a non-finite position"; return false; }
    minimum = {std::min(minimum.x,p.x),std::min(minimum.y,p.y),std::min(minimum.z,p.z)};
    maximum = {std::max(maximum.x,p.x),std::max(maximum.y,p.y),std::max(maximum.z,p.z)};
  }
  error.clear();
  return true;
}

inline bool materializeBoundingBox(const GeometryData &source, GeometryData &result,
                                    std::string &error) {
  Vec3 a, b;
  if (!geometryBounds(source,a,b,error)) return false;
  result.positions = {{a.x,a.y,a.z},{b.x,a.y,a.z},{b.x,b.y,a.z},{a.x,b.y,a.z},
                      {a.x,a.y,b.z},{b.x,a.y,b.z},{b.x,b.y,b.z},{a.x,b.y,b.z}};
  result.vertexIds = {1,2,3,4,5,6,7,8};
  result.indices = {0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,
                    3,7,6,3,6,2,0,4,7,0,7,3,1,2,6,1,6,5};
  return true;
}

inline std::uint32_t distributionRandom(std::uint32_t &state) noexcept {
  state += 0x9e3779b9u;
  auto value = state;
  value = (value ^ (value >> 16u)) * 0x85ebca6bu;
  value = (value ^ (value >> 13u)) * 0xc2b2ae35u;
  return value ^ (value >> 16u);
}

// Points in Volume shapes, stored in parameters[2] of pointsInBoxVolume. Box (0) is the saved default.
enum class PointVolumeShape : std::uint32_t { box = 0, sphere = 1, cylinder = 2, meshInterior = 3 };

// Ray-parity inside test for closed triangle meshes. The ray direction is skewed so it does not run
// along axis-aligned edges of primitive meshes.
inline bool pointInsideTriangleMesh(const GeometryData &mesh, double px, double py, double pz) noexcept {
  constexpr double dx = 0.5773502691896258, dy = 0.5345224838248488, dz = 0.6172133998483676;
  std::uint32_t crossings = 0;
  for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
    const auto &a = mesh.positions[mesh.indices[i]], &b = mesh.positions[mesh.indices[i + 1]],
               &c = mesh.positions[mesh.indices[i + 2]];
    const double e1x = double(b.x) - a.x, e1y = double(b.y) - a.y, e1z = double(b.z) - a.z;
    const double e2x = double(c.x) - a.x, e2y = double(c.y) - a.y, e2z = double(c.z) - a.z;
    const double hx = dy * e2z - dz * e2y, hy = dz * e2x - dx * e2z, hz = dx * e2y - dy * e2x;
    const double det = e1x * hx + e1y * hy + e1z * hz;
    if (std::abs(det) < 1e-12) continue;
    const double inv = 1.0 / det, sx = px - a.x, sy = py - a.y, sz = pz - a.z;
    const double u = (sx * hx + sy * hy + sz * hz) * inv;
    if (u < 0 || u > 1) continue;
    const double qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
    const double v = (dx * qx + dy * qy + dz * qz) * inv;
    if (v < 0 || u + v > 1) continue;
    if ((e2x * qx + e2y * qy + e2z * qz) * inv > 0) ++crossings;
  }
  return (crossings & 1u) != 0;
}

inline bool materializeDistributedPoints(const GeometryData &mesh,
                                         const OperationRecord &operation,
                                         PointsData &result, std::string &error,
                                         bool clampToMaximum = false) try {
  // Density distribution (pointsOnFacesDensity): parameters[0] is the maximum count,
  // parameters[2] the points per unit of surface area. The realized count is
  // floor(area * density) and must lie in [1, maximum]; larger counts are rejected.
  // Volume distribution (pointsInBoxVolume): parameters[2] is the PointVolumeShape. Sphere and
  // cylinder (Y axis) are inscribed in the input bounds; mesh interior keeps points inside the mesh.
  const bool density = operation.code == OperationCode::pointsOnFacesDensity;
  const bool volume = operation.code == OperationCode::pointsInBoxVolume;
  std::uint32_t count = 0, seed = 0, shapeIndex = 0;
  if (!exactIntegerParameter(operation.parameters[0],1,count)
      || !exactIntegerParameter(operation.parameters[1],0,seed)
      || (volume && (!exactIntegerParameter(operation.parameters[2],0,shapeIndex)
                     || shapeIndex > static_cast<std::uint32_t>(PointVolumeShape::meshInterior)))
      || (!density && !volume && operation.parameters[2] != 0) || operation.parameters[3] != 0
      || (density && !(operation.parameters[2] > 0 && std::isfinite(operation.parameters[2])))) {
    error = "Point distribution requires at least 1 point and a whole seed";
    return false;
  }
  const std::uint32_t maximumCount = count;
  Vec3 minimum, maximum;
  if (!geometryBounds(mesh,minimum,maximum,error)) return false;
  std::vector<double> cumulative;
  double total = 0;
  if (operation.code == OperationCode::pointsOnFaces || density) {
    if (mesh.indices.empty() || mesh.indices.size() % 3 != 0 || mesh.indices.size() > kMaximumIndices) {
      error = "Face distribution requires indexed triangles"; return false;
    }
    for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
      if (mesh.indices[i] >= mesh.positions.size() || mesh.indices[i+1] >= mesh.positions.size()
          || mesh.indices[i+2] >= mesh.positions.size()) {
        error = "Face distribution has an invalid triangle index"; return false;
      }
      const auto &a=mesh.positions[mesh.indices[i]], &b=mesh.positions[mesh.indices[i+1]],
                 &c=mesh.positions[mesh.indices[i+2]];
      const double ux=double(b.x)-a.x, uy=double(b.y)-a.y, uz=double(b.z)-a.z;
      const double vx=double(c.x)-a.x, vy=double(c.y)-a.y, vz=double(c.z)-a.z;
      const double x=uy*vz-uz*vy, y=uz*vx-ux*vz, z=ux*vy-uy*vx;
      total += std::sqrt(x*x+y*y+z*z);
      cumulative.push_back(total);
    }
    if (!std::isfinite(total) || total <= 0) {
      error = "Face distribution requires a nonzero finite surface area"; return false;
    }
    if (density) {
      // total sums cross-product magnitudes, which are twice each triangle's area.
      // A frame re-sample on a moving surface keeps the node's maximum as its only bound.
      double expected = std::floor(0.5 * total * double(operation.parameters[2]));
      if (clampToMaximum && std::isfinite(expected))
        expected = std::clamp(expected, 1.0, double(maximumCount));
      if (!std::isfinite(expected) || expected < 1 || expected > double(maximumCount)) {
        error = "Density distribution must yield 1 to the maximum count without overflow"; return false;
      }
      count = static_cast<std::uint32_t>(expected);
    }
  } else if (operation.code != OperationCode::pointsInBoxVolume
             || maximum.x <= minimum.x || maximum.y <= minimum.y || maximum.z <= minimum.z) {
    error = "Volume distribution requires nonzero extent on every axis"; return false;
  }
  const auto shape = static_cast<PointVolumeShape>(shapeIndex);
  if (shape == PointVolumeShape::meshInterior) {
    if (mesh.indices.empty() || mesh.indices.size() % 3 != 0) {
      error = "Mesh-interior distribution requires indexed triangles"; return false;
    }
    for (const auto index : mesh.indices)
      if (index >= mesh.positions.size()) {
        error = "Mesh-interior distribution has an invalid triangle index"; return false;
      }
  }
  const auto random = [&]() { return double(distributionRandom(seed) >> 8u) / 16777216.0; };
  const double cx = 0.5 * (double(minimum.x) + maximum.x), cy = 0.5 * (double(minimum.y) + maximum.y),
               cz = 0.5 * (double(minimum.z) + maximum.z);
  const double rx = 0.5 * (double(maximum.x) - minimum.x), ry = 0.5 * (double(maximum.y) - minimum.y),
               rz = 0.5 * (double(maximum.z) - minimum.z);
  // A mesh with no enclosed volume would reject forever; after this many misses per point the
  // distribution reports the open or flat mesh instead of hanging.
  constexpr std::uint64_t kInteriorAttemptsPerPoint = 4096;
  std::uint64_t interiorMisses = 0;
  result.points.clear(); result.points.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    Vec3 position;
    if (!cumulative.empty()) {
      const auto face=static_cast<std::size_t>(std::upper_bound(cumulative.begin(),cumulative.end(),
                                                               random()*total)-cumulative.begin());
      const auto &a=mesh.positions[mesh.indices[face*3]], &b=mesh.positions[mesh.indices[face*3+1]],
                 &c=mesh.positions[mesh.indices[face*3+2]];
      const double root=std::sqrt(random()), v=random(), wa=1-root, wb=root*(1-v), wc=root*v;
      position={static_cast<float>(wa*a.x+wb*b.x+wc*c.x),static_cast<float>(wa*a.y+wb*b.y+wc*c.y),
                static_cast<float>(wa*a.z+wb*b.z+wc*c.z)};
    } else if (shape == PointVolumeShape::sphere) {
      double x = 0, y = 0, z = 0;
      do { x = 2 * random() - 1; y = 2 * random() - 1; z = 2 * random() - 1; }
      while (x * x + y * y + z * z > 1);
      position={static_cast<float>(cx+x*rx),static_cast<float>(cy+y*ry),static_cast<float>(cz+z*rz)};
    } else if (shape == PointVolumeShape::cylinder) {
      const double radius = std::sqrt(random()), angle = 6.283185307179586 * random(), height = random();
      position={static_cast<float>(cx+radius*std::cos(angle)*rx),
                static_cast<float>(double(minimum.y)+height*(double(maximum.y)-minimum.y)),
                static_cast<float>(cz+radius*std::sin(angle)*rz)};
    } else if (shape == PointVolumeShape::meshInterior) {
      for (;;) {
        const double x = minimum.x + random() * (double(maximum.x) - minimum.x),
                     y = minimum.y + random() * (double(maximum.y) - minimum.y),
                     z = minimum.z + random() * (double(maximum.z) - minimum.z);
        if (pointInsideTriangleMesh(mesh, x, y, z)) {
          position={static_cast<float>(x),static_cast<float>(y),static_cast<float>(z)};
          break;
        }
        if (++interiorMisses > kInteriorAttemptsPerPoint * (std::uint64_t(i) + 1)) {
          error = "Mesh-interior distribution found no enclosed volume; close the mesh"; return false;
        }
      }
    } else {
      position={static_cast<float>(minimum.x+random()*(double(maximum.x)-minimum.x)),
                static_cast<float>(minimum.y+random()*(double(maximum.y)-minimum.y)),
                static_cast<float>(minimum.z+random()*(double(maximum.z)-minimum.z))};
    }
    result.points.push_back({position,i+1,1,{}});
  }
  error.clear(); return true;
} catch (const std::bad_alloc &) {
  error = "Point distribution could not allocate its points"; return false;
} catch (const std::length_error &) {
  error = "Point distribution could not allocate its points"; return false;
}

inline bool materializeInstanceEdit(const InstancesData &input,
                                     const OperationRecord &operation,
                                     InstancesData &result, std::string &error) {
  std::uint32_t first=0, last=0, stride=0;
  if (!exactIntegerParameter(operation.parameters[0],0,first)
      || !exactIntegerParameter(operation.parameters[1],0,last) || last < first
      || !exactIntegerParameter(operation.parameters[2],1,stride)
      || (operation.parameters[3] != 0 && operation.parameters[3] != 1)
      || input.instances.size() > kMaximumInstances) {
    error = "Instance selection requires ordered whole indices and a positive stride"; return false;
  }
  const auto &delta=operation.transform;
  const auto &q=delta.rotation;
  if (operation.code == OperationCode::transformInstances
      && (!finite(delta) || q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w <= 0 || operation.parameters[3] != 0)) {
    error = "Transform Instances requires a finite transform and nonzero orientation"; return false;
  }
  if (operation.code == OperationCode::cullInstances && !equal(delta,Transform{})) {
    error="Instance culling cannot carry a transform"; return false;
  }
  result.instances.clear(); result.instances.reserve(input.instances.size());
  for (std::size_t i=0; i<input.instances.size(); ++i) {
    bool selected=i>=first && i<=last && (i-first)%stride==0;
    if (operation.code == OperationCode::cullInstances) {
      if (operation.parameters[3] != 0) selected=!selected;
      if (selected) continue;
    }
    auto instance=input.instances[i];
    if (selected && operation.code == OperationCode::transformInstances) {
      auto &transform=instance.transform;
      transform.translation={transform.translation.x+delta.translation.x,
                             transform.translation.y+delta.translation.y,
                             transform.translation.z+delta.translation.z};
      transform.scale={transform.scale.x*delta.scale.x,transform.scale.y*delta.scale.y,
                       transform.scale.z*delta.scale.z};
      const auto r=transform.rotation;
      transform.rotation={r.w*q.x+r.x*q.w+r.y*q.z-r.z*q.y,
                          r.w*q.y-r.x*q.z+r.y*q.w+r.z*q.x,
                          r.w*q.z+r.x*q.y-r.y*q.x+r.z*q.w,
                          r.w*q.w-r.x*q.x-r.y*q.y-r.z*q.z};
      if (!finite(transform)) { error="Instance transform exceeds finite bounds"; return false; }
    }
    result.instances.push_back(instance);
  }
  if (result.instances.empty()) {
    error="Instance culling leaves no renderable instances"; return false;
  }
  error.clear(); return true;
}

// Cull Geometry and Cull Points take the same first, last, stride and keep-selected
// parameters as Cull Instances. Mesh faces are triangles. A vertex that no kept
// face references is removed and the remaining indices are renumbered. Vertex
// and face attributes are compacted with the same selection. Attributes in any
// other domain are refused, because the cull cannot renumber them.
struct CullRange final {
  std::size_t first = 0, last = 0, stride = 1;
  bool keepSelected = false;
};

inline bool parseCullRange(const OperationRecord &operation, CullRange &range,
                           std::string &error) {
  std::uint32_t first = 0, last = 0, stride = 0;
  if (!exactIntegerParameter(operation.parameters[0], 0, first)
      || !exactIntegerParameter(operation.parameters[1], 0, last) || last < first
      || !exactIntegerParameter(operation.parameters[2], 1, stride)
      || (operation.parameters[3] != 0 && operation.parameters[3] != 1)) {
    error = "Cull requires an ordered whole index range, a positive stride and a keep flag of 0 or 1";
    return false;
  }
  range.first = first;
  range.last = last;
  range.stride = stride;
  range.keepSelected = operation.parameters[3] == 1;
  return true;
}

// True when the element at index is removed by the cull.
inline bool culledByRange(const CullRange &range, std::size_t index) noexcept {
  const bool selected = index >= range.first && index <= range.last
      && (index - range.first) % range.stride == 0;
  return range.keepSelected ? !selected : selected;
}

struct GeometryCullSelection final {
  std::vector<std::size_t> faces;
  std::vector<std::size_t> vertices;
};

struct CullKeep final {
  Domain domain = Domain::unspecified;
  const std::vector<std::size_t> *kept = nullptr;
  std::size_t count = 0;
};

// Keeps the listed entries of each attribute. An attribute whose domain has no
// entry in keep, or whose cardinality differs from its entry, is refused.
inline bool compactCulledAttributes(std::vector<AttributeData> &attributes,
                                    const std::vector<CullKeep> &keep, std::string &error) {
  for (auto &attribute : attributes) {
    const auto found = std::find_if(keep.begin(), keep.end(), [&](const CullKeep &entry) {
      return entry.domain == attribute.descriptor.domain;
    });
    if (found == keep.end() || found->kept == nullptr) {
      error = "Cull cannot renumber attribute '" + attribute.descriptor.name + "' in the "
          + toWire(attribute.descriptor.domain) + " domain";
      return false;
    }
    if (attribute.elements.size() != found->count) {
      error = "Cull attribute '" + attribute.descriptor.name + "' does not match the cardinality of its "
          + toWire(attribute.descriptor.domain) + " domain";
      return false;
    }
    std::vector<FieldElement> compacted;
    compacted.reserve(found->kept->size());
    for (const auto index : *found->kept) compacted.push_back(attribute.elements[index]);
    attribute.elements = std::move(compacted);
  }
  error.clear();
  return true;
}

inline bool materializeGeometryCull(const GeometryData &input, const OperationRecord &operation,
                                    GeometryData &result, GeometryCullSelection &selection,
                                    std::string &error) {
  CullRange range;
  if (!parseCullRange(operation, range, error)) return false;
  if (input.indices.empty()) {
    error = "Cull Geometry needs a non-empty mesh"; return false;
  }
  if (input.indices.size() % 3 != 0
      || input.indices.size() > kMaximumIndices
      || (!input.vertexIds.empty() && input.vertexIds.size() != input.positions.size())) {
    error = "Mesh culling requires indexed triangles with one identity per vertex";
    return false;
  }
  selection = {};
  const std::size_t faceCount = input.indices.size() / 3;
  for (std::size_t face = 0; face < faceCount; ++face)
    if (!culledByRange(range, face)) selection.faces.push_back(face);
  if (selection.faces.empty()) {
    error = "Mesh culling leaves no renderable faces"; return false;
  }
  std::vector<bool> used(input.positions.size(), false);
  for (const auto face : selection.faces)
    for (std::size_t corner = 0; corner < 3; ++corner) {
      const auto vertex = input.indices[face * 3 + corner];
      if (vertex >= input.positions.size()) {
        error = "Mesh culling has an invalid triangle index"; return false;
      }
      used[vertex] = true;
    }
  // A cull that removes no face keeps every vertex, including any no face references.
  if (selection.faces.size() == faceCount) std::fill(used.begin(), used.end(), true);
  std::vector<std::uint32_t> renumbered(input.positions.size(), 0);
  result = {};
  for (std::size_t vertex = 0; vertex < input.positions.size(); ++vertex) {
    if (!used[vertex]) continue;
    renumbered[vertex] = static_cast<std::uint32_t>(selection.vertices.size());
    selection.vertices.push_back(vertex);
    result.positions.push_back(input.positions[vertex]);
    if (!input.vertexIds.empty()) result.vertexIds.push_back(input.vertexIds[vertex]);
  }
  result.indices.reserve(selection.faces.size() * 3);
  for (const auto face : selection.faces)
    for (std::size_t corner = 0; corner < 3; ++corner)
      result.indices.push_back(renumbered[input.indices[face * 3 + corner]]);
  error.clear();
  return true;
}

inline bool materializePointsCull(const PointsData &input, const OperationRecord &operation,
                                  PointsData &result, std::vector<std::size_t> &kept,
                                  std::string &error) {
  CullRange range;
  if (!parseCullRange(operation, range, error)) return false;
  if (input.points.empty()) {
    error = "Cull Points needs at least one input point"; return false;
  }
  kept.clear();
  for (std::size_t i = 0; i < input.points.size(); ++i)
    if (!culledByRange(range, i)) kept.push_back(i);
  if (kept.empty()) {
    error = "Point culling leaves no points"; return false;
  }
  result = {};
  result.points.reserve(kept.size());
  for (const auto i : kept) result.points.push_back(input.points[i]);
  error.clear();
  return true;
}

inline bool materializeJoin(const GeometryData &a, const GeometryData &b,
                            StableId aId, StableId bId, GeometryData &result,
                            std::size_t vertexCapacity, std::size_t indexCapacity,
                            std::string &error) {
  std::size_t vertices = 0, indices = 0;
  if (!checkedAdd(a.positions.size(),b.positions.size(),vertices) ||
      !checkedAdd(a.indices.size(),b.indices.size(),indices) ||
      vertices > std::min(vertexCapacity,kMaximumVertices) ||
      indices > std::min(indexCapacity,kMaximumIndices) ||
      a.positions.size() != a.vertexIds.size() || b.positions.size() != b.vertexIds.size()) {
    error = "Join Geometry exceeds its capacity or has incomplete identities";
    return false;
  }
  GeometryData joined;
  joined.positions.reserve(vertices); joined.vertexIds.reserve(vertices); joined.indices.reserve(indices);
  std::set<StableId> ids;
  const auto append = [&](const GeometryData &mesh, StableId sourceId, StableId branch) {
    const auto offset = static_cast<std::uint32_t>(joined.positions.size());
    if (mesh.indices.size()%3 != 0) return false;
    for (std::size_t i=0;i<mesh.positions.size();++i) {
      if (!finite(mesh.positions[i]) || !validStableId(mesh.vertexIds[i])) return false;
      // Include the input slot so joining one source twice produces distinct elements.
      // Keep IDs in the native instance range for Points from Vertices consumers.
      std::uint64_t hash = 1469598103934665603ull;
      for (const auto component : {sourceId,mesh.vertexIds[i],branch})
        for (unsigned shift=0;shift<64;shift+=8) {
          hash ^= (component >> shift)&0xffu;
          hash *= 1099511628211ull;
        }
      StableId id = hash % std::numeric_limits<std::uint32_t>::max() + 1;
      while (!ids.insert(id).second)
        id = id == std::numeric_limits<std::uint32_t>::max() ? 1 : id+1;
      joined.positions.push_back(mesh.positions[i]); joined.vertexIds.push_back(id);
    }
    for (const auto index : mesh.indices) {
      if (index >= mesh.positions.size()) return false;
      joined.indices.push_back(offset+index);
    }
    return true;
  };
  if (!append(a,aId,0) || !append(b,bId,1)) {
    error = "Join Geometry requires finite vertices and valid triangle topology";
    return false;
  }
  result = std::move(joined);
  error.clear();
  return true;
}

inline bool isBoundsFieldQuery(OperationCode code) noexcept {
  return code == OperationCode::boundsMinimumField || code == OperationCode::boundsMaximumField ||
         code == OperationCode::boundsSizeField || code == OperationCode::boundsCenterField;
}

inline bool isElementFieldQuery(OperationCode code) noexcept {
  return code == OperationCode::positionField || code == OperationCode::normalField ||
         code == OperationCode::indexField || code == OperationCode::stableIdentityField ||
         code == OperationCode::curveTangentField || code == OperationCode::curveParameterField ||
         code == OperationCode::curveRadiusField || code == OperationCode::curveTiltField ||
         code == OperationCode::curveEndpointField ||
         isBoundsFieldQuery(code);
}

inline Domain elementDomain(CarrierKind carrier) noexcept {
  if (carrier == CarrierKind::geometry3D) return Domain::vertex;
  if (carrier == CarrierKind::points3D) return Domain::point;
  if (carrier == CarrierKind::instances3D) return Domain::instance;
  if (carrier == CarrierKind::curves3D) return Domain::curvePoint;
  return Domain::unspecified;
}

inline ValueType elementFieldType(OperationCode code) noexcept {
  if (code == OperationCode::curveEndpointField) return ValueType::boolean;
  if (code == OperationCode::curveParameterField || code == OperationCode::curveRadiusField ||
      code == OperationCode::curveTiltField) return ValueType::floatValue;
  return code == OperationCode::indexField || code == OperationCode::stableIdentityField
      ? ValueType::integer : ValueType::vector;
}

// Point and instance normals use their oriented local +Z axis. Mesh normals
// are area-weighted vertex normals from the current triangle winding.
inline bool materializeElementField(const CarrierData &source, CarrierKind carrier,
                                    OperationCode code, FieldData &result,
                                    std::size_t capacity, std::string &error) {
  if (!isElementFieldQuery(code) || elementDomain(carrier) == Domain::unspecified) {
    error = "Element field requires geometry, points, or instances";
    return false;
  }
  const auto *mesh = std::get_if<GeometryData>(&source);
  const auto *points = std::get_if<PointsData>(&source);
  const auto *instances = std::get_if<InstancesData>(&source);
  if (carrier == CarrierKind::curves3D) {
    const auto *curves=std::get_if<CurvesData>(&source);
    if (!curves || !validateCurves(*curves,error) || curves->points.size()>capacity) {
      if (error.empty()) error="Curve field exceeds its point capacity";
      return false;
    }
    FieldData field;
    field.elements.resize(curves->points.size());
    for (const auto &spline : curves->splines) {
      std::vector<double> lengths;
      if (!curveLengths(*curves,spline,lengths,error)) return false;
      for (std::uint32_t i=0;i<spline.pointCount;++i) {
        const auto &point=curves->points[spline.firstPoint+i];
        auto &element=field.elements[spline.firstPoint+i].components;
        if (code==OperationCode::curveParameterField) element[0]=lengths[i]/lengths.back();
        else if (code==OperationCode::curveRadiusField) element[0]=point.radius;
        else if (code==OperationCode::curveTiltField) element[0]=point.tilt;
        else if (code==OperationCode::curveEndpointField) element[0]=!spline.cyclic && (i==0 || i+1==spline.pointCount);
        else if (code==OperationCode::positionField) element={point.position.x,point.position.y,point.position.z,0};
        else if (code==OperationCode::indexField) element[0]=spline.firstPoint+i;
        else if (code==OperationCode::stableIdentityField) element[0]=static_cast<double>(point.stableId);
        else if (code==OperationCode::curveTangentField) {
          Vec3 tangent;
          if (!curveTangent(*curves,spline,i,tangent,error)) return false;
          element={tangent.x,tangent.y,tangent.z,0};
        } else { error="Field operation is not available on curves"; return false; }
      }
    }
    result=std::move(field); error.clear(); return true;
  }
  if (code==OperationCode::curveTangentField || code==OperationCode::curveParameterField ||
      code==OperationCode::curveRadiusField || code==OperationCode::curveTiltField ||
      code==OperationCode::curveEndpointField) {
    error="Curve field requires Curves3D"; return false;
  }
  if (isBoundsFieldQuery(code)) {
    Vec3 a, b;
    if (carrier != CarrierKind::geometry3D || !mesh || capacity < 1 ||
        !geometryBounds(*mesh,a,b,error)) {
      if (error.empty()) error = "Bounding Box field requires Geometry3D and one field element";
      return false;
    }
    FieldElement element;
    if (code == OperationCode::boundsMinimumField) element.components = {a.x,a.y,a.z,0};
    else if (code == OperationCode::boundsMaximumField) element.components = {b.x,b.y,b.z,0};
    else if (code == OperationCode::boundsSizeField)
      element.components = {double(b.x)-a.x,double(b.y)-a.y,double(b.z)-a.z,0};
    else element.components = {(double(a.x)+b.x)*0.5,(double(a.y)+b.y)*0.5,(double(a.z)+b.z)*0.5,0};
    result.elements = {element};
    error.clear();
    return true;
  }
  if ((carrier == CarrierKind::geometry3D && mesh == nullptr) ||
      (carrier == CarrierKind::points3D && points == nullptr) ||
      (carrier == CarrierKind::instances3D && instances == nullptr)) {
    error = "Element field source does not match its typed carrier";
    return false;
  }
  const auto count = mesh ? mesh->positions.size()
      : points ? points->points.size() : instances->instances.size();
  if (count > capacity || count > kMaximumFieldElements ||
      count > (mesh ? kMaximumVertices : points ? kMaximumPoints : kMaximumInstances) ||
      (mesh && mesh->vertexIds.size() != count)) {
    error = "Element field exceeds its capacity or has incomplete vertex identities";
    return false;
  }
  FieldData field;
  field.elements.resize(count);
  if (code == OperationCode::normalField && mesh) {
    if (mesh->indices.size() % 3 != 0 || mesh->indices.size() > kMaximumIndices) {
      error = "Normal field requires triangle topology";
      return false;
    }
    for (std::size_t i = 0; i < mesh->indices.size(); i += 3) {
      const auto a = mesh->indices[i], b = mesh->indices[i + 1], c = mesh->indices[i + 2];
      if (a >= count || b >= count || c >= count) {
        error = "Normal field contains an invalid triangle index";
        return false;
      }
      const auto &pa = mesh->positions[a], &pb = mesh->positions[b], &pc = mesh->positions[c];
      const double ux = double(pb.x) - pa.x, uy = double(pb.y) - pa.y, uz = double(pb.z) - pa.z;
      const double vx = double(pc.x) - pa.x, vy = double(pc.y) - pa.y, vz = double(pc.z) - pa.z;
      for (const auto vertex : {a, b, c}) {
        auto &normal = field.elements[vertex].components;
        normal[0] += uy * vz - uz * vy;
        normal[1] += uz * vx - ux * vz;
        normal[2] += ux * vy - uy * vx;
      }
    }
  }
  const auto identityAt = [&](std::size_t i) {
    return mesh ? mesh->vertexIds[i] : points ? points->points[i].stableId : instances->instances[i].stableId;
  };
  // A million-vertex sea spent a tenth of each frame inserting its ids into a set.
  // Ids that ascend are unique, any other order is checked on a sorted copy, and
  // only a mesh with a repeat takes the ordered set scan, so its error is unchanged.
  bool uniqueIdentities = true;
  for (std::size_t i = 1; i < count && uniqueIdentities; ++i)
    uniqueIdentities = identityAt(i - 1) < identityAt(i);
  if (!uniqueIdentities) {
    std::vector<StableId> sorted(count);
    for (std::size_t i = 0; i < count; ++i) sorted[i] = identityAt(i);
    std::sort(sorted.begin(), sorted.end());
    uniqueIdentities = std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
  }
  std::set<StableId> identities;
  for (std::size_t i = 0; i < count; ++i) {
    const auto id = identityAt(i);
    const auto position = mesh ? mesh->positions[i]
        : points ? points->points[i].position : instances->instances[i].transform.translation;
    if (!validStableId(id) || (!uniqueIdentities && !identities.insert(id).second) || !finite(position)) {
      error = "Element field requires finite positions and unique stable identities";
      return false;
    }
    auto &element = field.elements[i].components;
    if (code == OperationCode::positionField) element = {position.x, position.y, position.z, 0};
    else if (code == OperationCode::indexField) element[0] = static_cast<double>(i);
    else if (code == OperationCode::stableIdentityField) element[0] = static_cast<double>(id);
    else {
      if (!mesh) {
        const auto q = points ? points->points[i].rotation : instances->instances[i].transform.rotation;
        const double norm = double(q.x)*q.x + double(q.y)*q.y + double(q.z)*q.z + double(q.w)*q.w;
        if (!finite(q) || norm <= 0 ||
            (instances && (!finite(instances->instances[i].transform.scale) ||
                           instances->instances[i].transform.scale.z == 0))) {
          error = "Normal field requires a finite orientation and nonzero normal scale";
          return false;
        }
        const double sign = instances && instances->instances[i].transform.scale.z < 0 ? -1 : 1;
        element = {sign * 2 * (double(q.x)*q.z + double(q.w)*q.y) / norm,
                   sign * 2 * (double(q.y)*q.z - double(q.w)*q.x) / norm,
                   sign * (1 - 2 * (double(q.x)*q.x + double(q.y)*q.y) / norm), 0};
      }
      const auto length = std::sqrt(element[0]*element[0] + element[1]*element[1] + element[2]*element[2]);
      if (length > 0) for (std::size_t axis = 0; axis < 3; ++axis) element[axis] /= length;
      else element = {0, 0, 1, 0};
    }
    if (!validFieldElement(field.elements[i], elementFieldType(code))) {
      error = "Element field contains a non-finite value";
      return false;
    }
  }
  result = std::move(field);
  error.clear();
  return true;
}

inline bool displaceElements(CarrierData &data, CarrierKind carrier,
                             const FieldData &field, float strength,
                             std::string &error) {
  auto *mesh = std::get_if<GeometryData>(&data);
  auto *points = std::get_if<PointsData>(&data);
  auto *instances = std::get_if<InstancesData>(&data);
  auto *curves = std::get_if<CurvesData>(&data);
  if (!finite(strength) || elementDomain(carrier) == Domain::unspecified ||
      (carrier == CarrierKind::geometry3D && !mesh) ||
      (carrier == CarrierKind::points3D && !points) ||
      (carrier == CarrierKind::instances3D && !instances) ||
      (carrier == CarrierKind::curves3D && !curves)) {
    error = "Vector displacement requires a finite strength and typed elements";
    return false;
  }
  const auto count = mesh ? mesh->positions.size()
      : points ? points->points.size() : curves ? curves->points.size() : instances->instances.size();
  if (count > (mesh ? kMaximumVertices : points ? kMaximumPoints : curves ? kMaximumCurvePoints : kMaximumInstances)) {
    error = "Vector displacement exceeds the source element capacity";
    return false;
  }
  if (field.elements.size() != count && field.elements.size() != 1) {
    error = "Vector displacement field does not match the element count";
    return false;
  }
  std::vector<Vec3> positions;
  positions.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto &element = field.elements[field.elements.size() == 1 ? 0 : i];
    const auto position = mesh ? mesh->positions[i]
        : points ? points->points[i].position : curves ? curves->points[i].position : instances->instances[i].transform.translation;
    if (!validFieldElement(element, ValueType::vector) || !finite(position)) {
      error = "Vector displacement contains a non-finite input";
      return false;
    }
    Vec3 displaced{static_cast<float>(position.x + strength * element.components[0]),
                   static_cast<float>(position.y + strength * element.components[1]),
                   static_cast<float>(position.z + strength * element.components[2])};
    if (!finite(displaced)) {
      error = "Vector displacement exceeds finite position bounds";
      return false;
    }
    positions.push_back(displaced);
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (mesh) mesh->positions[i] = positions[i];
    else if (points) points->points[i].position = positions[i];
    else if (curves) curves->points[i].position = positions[i];
    else instances->instances[i].transform.translation = positions[i];
  }
  error.clear();
  return true;
}

inline bool materializeSpectrumInstances(const OperationRecord& operation,
                                         const spectrum::Binding& binding,
                                         InstancesData& instances, std::string& error,
                                         const CarrierData* placement = nullptr) {
  const auto arrangement = operation.parameters[0];
  const auto width = operation.parameters[1], height = operation.parameters[2];
  const auto columns = operation.parameters[3];
  const auto count = binding.lastBand >= binding.firstBand
      ? static_cast<std::size_t>(binding.lastBand - binding.firstBand + 1) : 0;
  if (!spectrum::validate(binding, count, error)) return false;
  if (binding.target != spectrum::Target::instanceTransform
      || binding.fieldStableId != operation.stableId
      || binding.consumerGeometryStableId != operation.outputStableId
      || binding.sourceGeometryStableId != operation.inputStableId
      || !finite(arrangement) || arrangement < 0 || arrangement > 5
      || std::floor(arrangement) != arrangement || !finite(width) || width <= 0 || width > 1000
      || !finite(height) || height <= 0 || height > 1000
      || !finite(columns) || columns < 1 || columns > spectrum::kBands
      || std::floor(columns) != columns) {
    error = "Spectrum Instancer has invalid arrangement or source identity";
    return false;
  }
  if ((arrangement <= 2 && (operation.secondaryInputStableId != 0 || placement))
      || (arrangement >= 3 && (!validStableId(operation.secondaryInputStableId) || !placement))) {
    error = "Spectrum Instancer requires the exact retained placement source"; return false;
  }
  PointsData placements;
  if (arrangement == 3) {
    const auto* curves=std::get_if<CurvesData>(placement);
    if (!curves || !validateCurves(*curves,error)) {
      if (error.empty()) error="Spectrum Instancer curve layout requires Curves3D";
      return false;
    }
    std::vector<const SplineRecord*> splines;
    for (const auto& spline : curves->splines) splines.push_back(&spline);
    std::sort(splines.begin(),splines.end(),[](const auto* a,const auto* b){return a->stableId<b->stableId;});
    std::vector<std::vector<double>> lengths(splines.size());
    std::vector<double> ends;
    double total=0;
    for (std::size_t i=0;i<splines.size();++i) {
      if (!curveLengths(*curves,*splines[i],lengths[i],error)) return false;
      total+=lengths[i].back(); ends.push_back(total);
    }
    if (!finite(total) || total<=0) { error="Spectrum Instancer curve length is invalid"; return false; }
    const bool closed=std::all_of(splines.begin(),splines.end(),[](const auto* spline){return spline->cyclic;});
    for (std::size_t i=0;i<count;++i) {
      const auto distance=count>1 ? total*double(i)/double(closed ? count : count-1) : total*0.5;
      const auto index=std::min<std::size_t>(std::upper_bound(ends.begin(),ends.end(),distance)-ends.begin(),splines.size()-1);
      const auto begin=index==0 ? 0 : ends[index-1];
      const auto point=sampleCurve(*curves,*splines[index],lengths[index],(distance-begin)/lengths[index].back());
      placements.points.push_back({point.position,i+1,point.radius,
          {0,0,static_cast<float>(std::sin(point.tilt*0.5)),static_cast<float>(std::cos(point.tilt*0.5))}});
    }
  } else if (arrangement == 4) {
    const auto* points=std::get_if<PointsData>(placement);
    if (!points || points->points.empty() || points->points.size()>kMaximumPoints) {
      error="Spectrum Instancer point layout requires nonempty bounded Points3D"; return false;
    }
    std::vector<const PointRecord*> ordered;
    for (const auto& point : points->points) {
      if (!validStableId(point.stableId) || !finite(point.position) || !finite(point.rotation) || !finite(point.radius) || point.radius<=0
          || point.rotation.x*point.rotation.x+point.rotation.y*point.rotation.y
              +point.rotation.z*point.rotation.z+point.rotation.w*point.rotation.w<=0) {
        error="Spectrum Instancer point placement has invalid identity or transform"; return false;
      }
      ordered.push_back(&point);
    }
    std::sort(ordered.begin(),ordered.end(),[](const auto* a,const auto* b){return a->stableId<b->stableId;});
    if (std::adjacent_find(ordered.begin(),ordered.end(),[](const auto* a,const auto* b) {
        return a->stableId==b->stableId;
      })!=ordered.end()) {
      error="Spectrum Instancer point placement has duplicate identities"; return false;
    }
    for (std::size_t i=0;i<count;++i) {
      const auto coordinate=count>1 ? double(i)/double(count-1) : 0.5;
      placements.points.push_back(*ordered[static_cast<std::size_t>(std::round(coordinate*(ordered.size()-1)))]);
    }
  } else if (arrangement == 5) {
    const auto* mesh=std::get_if<GeometryData>(placement);
    // Reuse the retained triangle area distribution and its deterministic seed.
    const OperationRecord distribution{operation.stableId,OperationCode::pointsOnFaces,
        operation.secondaryInputStableId,operation.outputStableId,{static_cast<float>(count),0,0,0}};
    if (!mesh || !materializeDistributedPoints(*mesh,distribution,placements,error)) {
      if (error.empty()) error="Spectrum Instancer mesh layout requires Geometry3D triangles";
      return false;
    }
  }
  instances.instances.clear();
  instances.instances.reserve(count);
  const auto gridColumns = std::min(count, static_cast<std::size_t>(columns));
  const auto rows = (count + gridColumns - 1) / gridColumns;
  for (std::size_t i = 0; i < count; ++i) {
    const float coordinate = count > 1 ? static_cast<float>(i) / static_cast<float>(count - 1) : 0;
    if (binding.coordinates[i] != coordinate) {
      error = "Spectrum Instancer sample coordinates must retain exact selected-band identity";
      return false;
    }
    Vec3 position {};
    Quaternion rotation {};
    float radius=1;
    if (arrangement >= 3) {
      const auto& point=placements.points[i];
      position=point.position; rotation=point.rotation; radius=point.radius;
    } else if (arrangement == 0) position.x = count > 1 ? (coordinate - 0.5f) * width : 0;
    else if (arrangement == 1) {
      const auto angle = 6.283185307179586 * static_cast<double>(i) / static_cast<double>(count);
      position = {width * 0.5f * static_cast<float>(std::cos(angle)),
                  height * 0.5f * static_cast<float>(std::sin(angle)), 0};
    } else {
      position.x = gridColumns > 1
          ? (static_cast<float>(i % gridColumns) / static_cast<float>(gridColumns - 1) - 0.5f) * width : 0;
      position.y = rows > 1
          ? (static_cast<float>(i / gridColumns) / static_cast<float>(rows - 1) - 0.5f) * height : 0;
    }
    instances.instances.push_back({binding.firstBand + i + 1, operation.inputStableId,
        {position, rotation, {binding.baseScale[0]*radius, binding.baseScale[1]*radius, binding.baseScale[2]*radius}}});
    if (!finite(instances.instances.back().transform)) {
      error="Spectrum Instancer placement exceeds finite transform bounds"; return false;
    }
  }
  error.clear();
  return true;
}

} // namespace videowire::geometry
#include "GeometryFieldOperations.h"

namespace videowire::geometry {
// Shade Smooth, Shade Flat and Smooth by Angle. parameters[0] is the mode and
// parameters[1] the Smooth by Angle threshold in degrees. Corners sharing a
// welded edge blend unless that edge is sharp, so a vertex splits once per
// smooth group. The first group keeps its index and stable id, which leaves
// indices stable for sharp edge marks, and later groups append. Welding by
// exact position lets generator seams and split cube faces blend. A marked edge
// outside the mesh marks nothing, so a topology change upstream never fails.
inline constexpr float kShadeSmooth = 1, kShadeFlat = 2, kShadeByAngle = 3;
// parameters[2] on By Angle over a moving mesh: every corner gets its own vertex, so
// the crease can follow the motion while the vertices and indices stay the same.
inline constexpr float kShadeEveryCorner = 1;
inline bool materializeShadedMesh(const GeometryData &input, const OperationRecord &operation,
                                  GeometryData &output, std::vector<AttributeData> &attributes,
                                  std::string &error) {
  using Sum = std::array<double, 3>;
  const auto mode = operation.parameters[0];
  const auto angle = static_cast<double>(operation.parameters[1]);
  const auto vertexCount = input.positions.size();
  const bool everyCorner = mode == kShadeByAngle && operation.parameters[2] == kShadeEveryCorner;
  if ((mode != kShadeSmooth && mode != kShadeFlat && mode != kShadeByAngle)
      || !(angle >= 0 && angle <= 180) || (operation.parameters[2] != 0 && !everyCorner) || operation.parameters[3] != 0
      || input.indices.size() % 3 != 0
      || std::any_of(input.indices.begin(), input.indices.end(),
             [&](std::uint32_t index) { return index >= vertexCount; })) {
    error = "Shade Mesh has invalid settings";
    return false;
  }
  const auto bitsOf = [](float value) {
    value += 0.0f; // Negative zero welds with zero.
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    return bits;
  };
  // Welded vertices are numbered in first-seen order through an open-addressed table.
  // Only the partition reaches the output: edge ends, groups and sums follow corners,
  // whatever the numbering. Sorting every position key cost more on a moving sea.
  std::vector<std::array<std::uint32_t, 3>> keys(vertexCount);
  for (std::size_t v = 0; v < vertexCount; ++v)
    keys[v] = {bitsOf(input.positions[v].x), bitsOf(input.positions[v].y), bitsOf(input.positions[v].z)};
  std::vector<std::uint32_t> weld(vertexCount);
  std::uint32_t weldCount = 0;
  {
    std::size_t capacity = 16;
    while (capacity < vertexCount * 2) capacity <<= 1;
    const auto mask = capacity - 1;
    constexpr auto kEmpty = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> slots(capacity, kEmpty);
    for (std::size_t v = 0; v < vertexCount; ++v) {
      const auto &key = keys[v];
      std::uint64_t hash = (std::uint64_t{key[0]} << 32 | key[1]) * 0x9E3779B97F4A7C15ull;
      hash ^= (hash >> 29) ^ (std::uint64_t{key[2]} * 0xBF58476D1CE4E5B9ull);
      hash ^= hash >> 31;
      auto slot = static_cast<std::size_t>(hash) & mask;
      while (slots[slot] != kEmpty && keys[slots[slot]] != key) slot = (slot + 1) & mask;
      if (slots[slot] == kEmpty) {
        slots[slot] = static_cast<std::uint32_t>(v);
        weld[v] = weldCount++;
      } else weld[v] = weld[slots[slot]];
    }
  }

  const auto cornerCount = input.indices.size();
  const auto faceCount = cornerCount / 3;
  std::vector<Sum> faceNormals(faceCount);
  struct WeldedEdge final {
    std::uint32_t low, high, face, cornerLow, cornerHigh;
  };
  std::vector<WeldedEdge> edges;
  edges.reserve(cornerCount);
  for (std::size_t face = 0; face < faceCount; ++face) {
    const auto &a = input.positions[input.indices[face * 3]];
    const auto &b = input.positions[input.indices[face * 3 + 1]];
    const auto &c = input.positions[input.indices[face * 3 + 2]];
    const Sum e1{double(b.x) - a.x, double(b.y) - a.y, double(b.z) - a.z};
    const Sum e2{double(c.x) - a.x, double(c.y) - a.y, double(c.z) - a.z};
    faceNormals[face] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    for (std::size_t k = 0; k < 3; ++k) {
      auto first = static_cast<std::uint32_t>(face * 3 + k);
      auto second = static_cast<std::uint32_t>(face * 3 + (k + 1) % 3);
      if (weld[input.indices[first]] == weld[input.indices[second]]) continue;
      if (weld[input.indices[first]] > weld[input.indices[second]]) std::swap(first, second);
      edges.push_back({weld[input.indices[first]], weld[input.indices[second]],
                       static_cast<std::uint32_t>(face), first, second});
    }
  }
  // Edges sort by (low, high, face), and no two share all three. On a million-vertex
  // sea one comparison sort over six million edges was a third of each frame. Edges
  // arrive in face order, so a counting pass by low keeps face order within each
  // bucket, and each small bucket then sorts alone into the same total order.
  {
    const auto before = [](const WeldedEdge &l, const WeldedEdge &r) {
      return std::tie(l.low, l.high, l.face) < std::tie(r.low, r.high, r.face);
    };
    std::vector<std::size_t> starts(static_cast<std::size_t>(weldCount) + 1, 0u);
    for (const auto &edge : edges) ++starts[edge.low + 1];
    for (std::size_t i = 1; i < starts.size(); ++i) starts[i] += starts[i - 1];
    std::vector<WeldedEdge> ordered(edges.size());
    {
      auto next = starts;
      for (const auto &edge : edges) ordered[next[edge.low]++] = edge;
    }
    for (std::size_t low = 0; low + 1 < starts.size(); ++low)
      if (starts[low + 1] - starts[low] > 1)
        std::sort(ordered.begin() + starts[low], ordered.begin() + starts[low + 1], before);
    edges = std::move(ordered);
  }
  std::vector<std::pair<std::uint32_t, std::uint32_t>> marked;
  marked.reserve(operation.sharpEdges.size());
  for (const auto &edge : operation.sharpEdges) {
    if (edge[0] >= vertexCount || edge[1] >= vertexCount || weld[edge[0]] == weld[edge[1]]) continue;
    marked.push_back(std::minmax(weld[edge[0]], weld[edge[1]]));
  }
  std::sort(marked.begin(), marked.end());

  std::vector<std::uint32_t> group(cornerCount);
  std::iota(group.begin(), group.end(), 0u);
  const auto find = [&group](std::uint32_t corner) {
    while (group[corner] != corner) corner = group[corner] = group[group[corner]];
    return corner;
  };
  const auto overAngle = [&](std::uint32_t first, std::uint32_t second) {
    const auto &n = faceNormals[first];
    const auto &m = faceNormals[second];
    const Sum cross{n[1] * m[2] - n[2] * m[1], n[2] * m[0] - n[0] * m[2], n[0] * m[1] - n[1] * m[0]};
    const auto sine = std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
    const auto cosine = n[0] * m[0] + n[1] * m[1] + n[2] * m[2];
    if (sine == 0 && cosine == 0) return false; // A degenerate face has no angle.
    // The tolerance keeps a face pair exactly at the threshold smooth.
    return std::atan2(sine, cosine) * 180.0 / 3.14159265358979323846 > angle + 1.0e-4;
  };
  if (mode != kShadeFlat)
    for (std::size_t begin = 0; begin < edges.size();) {
      auto end = begin + 1;
      while (end < edges.size() && edges[end].low == edges[begin].low && edges[end].high == edges[begin].high) ++end;
      const bool sharp = std::binary_search(marked.begin(), marked.end(), std::make_pair(edges[begin].low, edges[begin].high));
      if (!sharp)
        for (auto i = begin; i < end; ++i)
          for (auto j = i + 1; j < end; ++j) {
            if (mode == kShadeByAngle && overAngle(edges[i].face, edges[j].face)) continue;
            group[find(edges[i].cornerLow)] = find(edges[j].cornerLow);
            group[find(edges[i].cornerHigh)] = find(edges[j].cornerHigh);
          }
      begin = end;
    }

  // Each group sums into its own dense slot, numbered by its root corner, rather than
  // a slot per corner: six million corners on the sea made that a 150 MB array. A root
  // keeps its own slot in the second pass, so every corner reads its root's.
  std::vector<std::uint32_t> groupSlot(cornerCount);
  std::uint32_t groupCount = 0;
  for (std::uint32_t corner = 0; corner < cornerCount; ++corner)
    if (find(corner) == corner) groupSlot[corner] = groupCount++;
  for (std::uint32_t corner = 0; corner < cornerCount; ++corner) groupSlot[corner] = groupSlot[find(corner)];
  std::vector<Sum> groupSums(groupCount, Sum{}), weldSums(weldCount, Sum{});
  // Each corner adds its face's direction weighted by the angle it opens, so how a
  // surface is split into triangles does not lean the normal. A corner within rounding
  // of 0 or 180 degrees adds nothing: its direction is noise, but its angle can be large.
  constexpr double kMinimumCornerSine = 1.0e-6;
  for (std::uint32_t corner = 0; corner < cornerCount; ++corner) {
    const auto &n = faceNormals[corner / 3];
    const auto sine = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    const auto first = corner - corner % 3;
    const auto &p = input.positions[input.indices[corner]];
    const auto &q = input.positions[input.indices[first + (corner + 1) % 3]];
    const auto &r = input.positions[input.indices[first + (corner + 2) % 3]];
    const Sum e1{double(q.x) - p.x, double(q.y) - p.y, double(q.z) - p.z};
    const Sum e2{double(r.x) - p.x, double(r.y) - p.y, double(r.z) - p.z};
    const auto lengths = std::sqrt((e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2])
                                   * (e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]));
    if (!(sine > kMinimumCornerSine * lengths) || !std::isfinite(sine)) continue;
    const auto weight = std::atan2(sine, e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]) / sine;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      groupSums[groupSlot[corner]][axis] += n[axis] * weight;
      weldSums[weld[input.indices[corner]]][axis] += n[axis] * weight;
    }
  }
  const auto unit = [](const Sum &value, Sum &result) {
    const auto length = std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
    if (!(length > 0) || !std::isfinite(length)) return false;
    result = {value[0] / length, value[1] / length, value[2] / length};
    return true;
  };
  const auto normalFor = [&](const Sum *sum, std::uint32_t vertex) {
    Sum result{0, 0, 1};
    if ((sum == nullptr || !unit(*sum, result)) && weldCount != 0) unit(weldSums[weld[vertex]], result);
    return result;
  };

  GeometryData shaded;
  shaded.positions = input.positions;
  shaded.vertexIds = input.vertexIds;
  shaded.uv = input.uv;
  shaded.indices = input.indices;
  std::vector<std::uint32_t> sources(vertexCount);
  std::iota(sources.begin(), sources.end(), 0u);
  std::vector<Sum> normals(vertexCount);
  for (std::uint32_t v = 0; v < vertexCount; ++v) normals[v] = normalFor(nullptr, v);
  constexpr auto kUnassigned = std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint32_t> firstGroup(vertexCount, kUnassigned);
  std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> splits;
  auto nextVertexId = input.vertexIds.empty() ? StableId{0}
      : *std::max_element(input.vertexIds.begin(), input.vertexIds.end());
  for (std::uint32_t corner = 0; corner < cornerCount; ++corner) {
    const auto vertex = input.indices[corner];
    const auto root = find(corner);
    if (firstGroup[vertex] == kUnassigned) {
      firstGroup[vertex] = root;
      normals[vertex] = normalFor(&groupSums[groupSlot[corner]], vertex);
      continue;
    }
    if (firstGroup[vertex] == root && !everyCorner) continue;
    const auto [split, added] = splits.try_emplace({vertex, everyCorner ? corner : root}, static_cast<std::uint32_t>(shaded.positions.size()));
    shaded.indices[corner] = split->second;
    if (!added) continue;
    shaded.positions.push_back(input.positions[vertex]);
    if (!input.uv.empty()) shaded.uv.push_back(input.uv[vertex]);
    if (!input.vertexIds.empty()) {
      if (nextVertexId >= kMaximumStableId) {
        error = "Shade Mesh ran out of vertex identities";
        return false;
      }
      shaded.vertexIds.push_back(++nextVertexId);
    }
    sources.push_back(vertex);
    normals.push_back(normalFor(&groupSums[groupSlot[corner]], vertex));
  }

  attributes.erase(std::remove_if(attributes.begin(), attributes.end(),
      [](const AttributeData &attribute) { return attribute.descriptor.name == "normal"; }), attributes.end());
  for (auto &attribute : attributes) {
    if (attribute.descriptor.domain != Domain::vertex || attribute.elements.size() != vertexCount) continue;
    std::vector<FieldElement> elements(sources.size());
    for (std::size_t v = 0; v < sources.size(); ++v) elements[v] = attribute.elements[sources[v]];
    attribute.elements = std::move(elements);
  }
  AttributeData normal;
  normal.descriptor = {namedAttributeIdentity("normal", Domain::vertex, ValueType::vector), "normal",
                       ValueType::vector, Domain::vertex, Interpolation::nearest};
  normal.elements.resize(normals.size());
  for (std::size_t v = 0; v < normals.size(); ++v)
    normal.elements[v].components = {normals[v][0], normals[v][1], normals[v][2], 0};
  attributes.push_back(std::move(normal));
  output = std::move(shaded);
  error.clear();
  return true;
}
} // namespace videowire::geometry

namespace videowire::geometry::owner_detail {
using IdSet = std::set<StableId>;
using ParentsMap = std::map<StableId, IdSet>;
using ProducedMap = std::map<StableId, CarrierKind>;
using MaterialMap = std::map<StableId, StableId>;
using FieldsMap = std::map<StableId, FieldData>;
using InstancesMap = std::map<StableId, InstancesData>;
using FieldTypesMap = std::map<StableId, std::pair<Domain, ValueType>>;
using RetainedAttributesMap = std::map<StableId, std::vector<AttributeData>>;

struct CoreAddParent final {
  ParentsMap& parents;
  const OperationRecord& op;
  void operator()(StableId input) const {
    if (input!=0 && input!=op.outputStableId) parents[op.outputStableId].insert(input);
  }
};
struct CoreRetainedDataLookup final {
  FieldsMap& fields;
  MaterializedInstancePlan& local;
  InstancesMap& instanceValues;
  std::optional<CarrierData> operator()(StableId id) const {
    if (const auto found = fields.find(id); found != fields.end()) return found->second;
    if (const auto found = local.geometries.find(id); found != local.geometries.end()) return found->second;
    if (const auto found = local.points.find(id); found != local.points.end()) return found->second;
    if (const auto found = local.curves.find(id); found != local.curves.end()) return found->second;
    if (const auto found = instanceValues.find(id); found != instanceValues.end()) return found->second;
    return std::nullopt;
  }
};
struct CoreRequireInput final {
  ProducedMap& produced;
  const OperationRecord& operation;
  bool operator()(CarrierKind carrier) const {
    const auto found = produced.find(operation.inputStableId);
    return found != produced.end() && found->second == carrier;
  }
};
struct CoreGenerateOutput final {
  const OperationRecord& operation;
  IdSet& generatedOutputs;
  ProducedMap& produced;
  MaterialMap& materials;
  MaterializedInstancePlan& local;
  ParentsMap& dependencies;
  RetainedAttributesMap& retainedAttributes;
  std::size_t& fieldBytes;
  bool operator()(CarrierKind carrier) const {
    if (operation.stableId != operation.outputStableId ||
        !generatedOutputs.insert(operation.outputStableId).second)
      return false;
    produced[operation.outputStableId] = carrier;
    materials[operation.outputStableId] = materials.count(operation.inputStableId)
        ? materials.at(operation.inputStableId) : 1;
    local.materialTables[operation.outputStableId]=local.materialTables[operation.inputStableId];
    dependencies[operation.outputStableId].insert(operation.inputStableId);
    if (carrier != CarrierKind::field && (operation.code==OperationCode::vectorDisplacement
        || operation.code==OperationCode::transformTRS || operation.code==OperationCode::assignMaterial
        || operation.code==OperationCode::joinGeometry || operation.code==OperationCode::transformInstances
        || operation.code==OperationCode::cullInstances || operation.code==OperationCode::storeNamedAttribute
        || operation.code==OperationCode::setNormal || operation.code==OperationCode::transform
        || operation.code==OperationCode::audioDeformer || operation.code==OperationCode::setInstanceAppearance
        || operation.code==OperationCode::materialTableSlot || operation.code==OperationCode::setMaterialIndex
        || operation.code==OperationCode::cullGeometry || operation.code==OperationCode::cullPoints
        || operation.code==OperationCode::shadeMesh)) {
      std::size_t bytes=0;
      for (const auto& attribute : retainedAttributes[operation.inputStableId])
        if (!addArrayBytes(attribute.elements.size(),sizeof(FieldElement),bytes)) return false;
      if (bytes>kMaximumBufferBytes-fieldBytes) return false;
      fieldBytes+=bytes;
      retainedAttributes[operation.outputStableId]=retainedAttributes[operation.inputStableId];
    }
    return true;
  }
};
struct CoreGetField final {
  FieldsMap& fields;
  FieldTypesMap& fieldTypes;
  const FieldOperationSettings& settings;
  const FieldData* operator()(StableId id, ValueType type) const {
    const auto field=fields.find(id); const auto fieldType=fieldTypes.find(id);
    if (field==fields.end() || fieldType==fieldTypes.end()
        || fieldType->second!=std::make_pair(settings.domain,type)) return nullptr;
    return &field->second;
  }
};
struct CoreExecutingLocalFunctorSizes final {
  std::size_t addParent;
  std::size_t retainedDataLookup;
  std::size_t requireInput;
  std::size_t generateOutput;
  std::size_t getField;
};
inline constexpr CoreExecutingLocalFunctorSizes coreExecutingLocalFunctorSizes() noexcept {
  return {sizeof(CoreAddParent),sizeof(CoreRetainedDataLookup),sizeof(CoreRequireInput),
          sizeof(CoreGenerateOutput),sizeof(CoreGetField)};
}
} // namespace videowire::geometry::owner_detail

namespace videowire::geometry {

// Subdivision Surface: Catmull-Clark on a welded cage. The Geometry Core stores
// triangles, so a pair of consecutive triangles that share their longest edge
// and lie near one plane becomes a quad again. Any other triangle stays a
// three-sided face, which Catmull-Clark handles directly. Topology welds by exact
// position, so a cube's 24 unwelded vertices subdivide as its 8 corners. uv and
// vertex or face-corner attributes interpolate linearly per face corner and split
// an output vertex only where their values differ. Normals are not carried: the
// backend derives them from the subdivided triangles.
//
// Edges with one face, or more than two, stay sharp. A crease weight w gives an
// edge sharpness 10*w*w, as Blender does, and w = 1 never smooths at any level.
// Each level subtracts 1 from a finite sharpness. A vertex with fewer than two
// sharp edges smooths, with two follows the crease, and with three or more stays
// fixed, blended by its mean sharpness below 1.
//
// The only failures are integer overflow and allocation failure, both reported
// with counts and bytes. Level and mesh size have no other bound.
namespace subdivision_detail {
struct Point final { double x = 0, y = 0, z = 0; };
inline Point operator+(Point a, Point b) noexcept { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
inline Point operator-(Point a, Point b) noexcept { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline Point operator*(Point a, double s) noexcept { return {a.x*s,a.y*s,a.z*s}; }
inline Point lerp(Point a, Point b, double t) noexcept { return a*(1.0-t)+b*t; }
inline double dot(Point a, Point b) noexcept { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline Point cross(Point a, Point b) noexcept {
  return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
inline Point toPoint(const Vec3 &v) noexcept {
  return {static_cast<double>(v.x),static_cast<double>(v.y),static_cast<double>(v.z)};
}
inline std::uint64_t edgeKey(std::uint32_t a, std::uint32_t b) noexcept {
  const auto low = std::min(a,b), high = std::max(a,b);
  return (static_cast<std::uint64_t>(low) << 32u) | high;
}
struct SharpEdge final { std::uint64_t key = 0; double sharpness = 0; };
struct Cage final {
  std::vector<Point> positions;
  std::vector<StableId> ids;
  std::vector<std::size_t> faceStarts;
  std::vector<std::uint32_t> corners;
  std::vector<std::size_t> faceOrigins;
  std::vector<double> cornerData;
  std::vector<std::uint8_t> discreteChannels;
  std::vector<SharpEdge> sharpEdges;
};
struct Channels final {
  bool uv = false;
  std::size_t stride = 0;
  std::vector<std::size_t> offsets;
};
inline std::string countsAndBytes(const char *what, std::size_t count, std::size_t width) {
  std::size_t bytes = 0;
  const bool fits = checkedMul(count,width,bytes);
  return std::to_string(count)+" "+what+" ("+(fits ? std::to_string(bytes)+" bytes" : std::string("more bytes than size_t holds"))+")";
}
inline bool carriesNormal(const AttributeData &attribute) noexcept {
  return attribute.descriptor.name == "normal"
      && (attribute.descriptor.domain == Domain::vertex || attribute.descriptor.domain == Domain::faceCorner);
}
inline bool interpolates(const AttributeData &attribute) noexcept {
  return attribute.descriptor.interpolation == Interpolation::linear
      || attribute.descriptor.interpolation == Interpolation::normalizedLinear;
}
inline void readCorner(const GeometryData &source, const std::vector<AttributeData> &attributes,
                       const Channels &channels, std::size_t indexPosition, double *out) noexcept {
  const auto vertex = source.indices[indexPosition];
  if (channels.uv) {
    out[0] = source.uv[vertex].x; out[1] = source.uv[vertex].y; out[2] = source.uv[vertex].z;
  }
  for (std::size_t a = 0; a < attributes.size(); ++a) {
    const auto &attribute = attributes[a];
    if (carriesNormal(attribute)) continue;
    const auto domain = attribute.descriptor.domain;
    if (domain != Domain::vertex && domain != Domain::faceCorner) continue;
    const auto &element = attribute.elements[domain == Domain::vertex ? vertex : indexPosition];
    std::copy(element.components.begin(),element.components.end(),out+channels.offsets[a]);
  }
}
// Welds by exact position and returns each index's cage vertex.
inline void weldCage(const GeometryData &source, std::vector<std::uint32_t> &cageOfVertex, Cage &cage) {
  const auto bits = [](float value) noexcept {
    std::uint32_t result = 0;
    if (value != 0.0f) std::memcpy(&result,&value,sizeof(result));
    return result;
  };
  const auto key = [&](std::uint32_t vertex) noexcept {
    const auto &p = source.positions[vertex];
    return std::array<std::uint32_t,3>{bits(p.x),bits(p.y),bits(p.z)};
  };
  std::vector<std::uint32_t> order(source.positions.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<std::uint32_t>(i);
  std::sort(order.begin(),order.end(),[&](std::uint32_t a, std::uint32_t b) noexcept {
    const auto left = key(a), right = key(b);
    return left != right ? left < right : a < b;
  });
  std::vector<std::uint32_t> leader(source.positions.size());
  for (std::size_t i = 0; i < order.size(); ++i)
    leader[order[i]] = i > 0 && key(order[i-1]) == key(order[i]) ? leader[order[i-1]] : order[i];
  std::vector<std::uint8_t> used(source.positions.size(),0);
  for (const auto index : source.indices) used[leader[index]] = 1;
  constexpr auto unassigned = std::numeric_limits<std::uint32_t>::max();
  cageOfVertex.assign(source.positions.size(),unassigned);
  for (std::size_t vertex = 0; vertex < source.positions.size(); ++vertex)
    if (leader[vertex] == vertex && used[vertex]) {
      cageOfVertex[vertex] = static_cast<std::uint32_t>(cage.positions.size());
      cage.positions.push_back(toPoint(source.positions[vertex]));
      cage.ids.push_back(source.vertexIds[vertex]);
    }
  for (std::size_t vertex = 0; vertex < source.positions.size(); ++vertex)
    cageOfVertex[vertex] = cageOfVertex[leader[vertex]];
}
inline void loadCreases(const GeometryData &source, const std::vector<std::uint32_t> &cageOfVertex,
                        const std::vector<SubdivisionCrease> &creases, Cage &cage) {
  if (creases.empty()) return;
  constexpr auto unassigned = std::numeric_limits<std::uint32_t>::max();
  std::vector<std::pair<StableId,std::uint32_t>> byId;
  byId.reserve(source.vertexIds.size());
  for (std::size_t vertex = 0; vertex < source.vertexIds.size(); ++vertex)
    if (cageOfVertex[vertex] != unassigned) byId.emplace_back(source.vertexIds[vertex],cageOfVertex[vertex]);
  std::sort(byId.begin(),byId.end());
  const auto find = [&](StableId id) noexcept {
    const auto at = std::lower_bound(byId.begin(),byId.end(),std::make_pair(id,std::uint32_t{0}));
    return at != byId.end() && at->first == id ? at->second : unassigned;
  };
  // A crease on a vertex the mesh no longer has marks nothing, as a stale edge does in Blender.
  for (const auto &crease : creases) {
    const auto a = find(crease.vertexA), b = find(crease.vertexB);
    if (a == unassigned || b == unassigned || a == b || crease.weight <= 0.0f) continue;
    const double weight = crease.weight;
    cage.sharpEdges.push_back({edgeKey(a,b),crease.weight >= 1.0f
        ? std::numeric_limits<double>::infinity() : 10.0*weight*weight});
  }
  std::sort(cage.sharpEdges.begin(),cage.sharpEdges.end(),[](const SharpEdge &a, const SharpEdge &b) noexcept {
    return a.key != b.key ? a.key < b.key : a.sharpness > b.sharpness;
  });
  cage.sharpEdges.erase(std::unique(cage.sharpEdges.begin(),cage.sharpEdges.end(),
      [](const SharpEdge &a, const SharpEdge &b) noexcept { return a.key == b.key; }),cage.sharpEdges.end());
}
// Rebuilds quads from consecutive triangles. The diagonal must be the longest
// edge of both, so a pair that really is two triangles stays two faces.
inline void buildCageFaces(const GeometryData &source, const std::vector<AttributeData> &attributes,
                           const Channels &channels, const std::vector<std::uint32_t> &cageOfVertex, Cage &cage) {
  const auto triangles = source.indices.size()/3;
  std::vector<double> first(channels.stride), second(channels.stride);
  const auto sameCornerData = [&](std::size_t a, std::size_t b) {
    if (channels.stride == 0) return true;
    readCorner(source,attributes,channels,a,first.data());
    readCorner(source,attributes,channels,b,second.data());
    return first == second;
  };
  const auto sameFaceData = [&](std::size_t a, std::size_t b) {
    for (const auto &attribute : attributes)
      if (attribute.descriptor.domain == Domain::face && attribute.elements[a].components != attribute.elements[b].components)
        return false;
    return true;
  };
  const auto pushFace = [&](std::initializer_list<std::size_t> positions, std::size_t origin) {
    cage.faceStarts.push_back(cage.corners.size());
    cage.faceOrigins.push_back(origin);
    for (const auto position : positions) {
      cage.corners.push_back(cageOfVertex[source.indices[position]]);
      const auto at = cage.cornerData.size();
      cage.cornerData.resize(at+channels.stride);
      if (channels.stride != 0) readCorner(source,attributes,channels,position,cage.cornerData.data()+at);
    }
  };
  const auto corner = [&](std::size_t triangle, std::size_t k) noexcept {
    return cageOfVertex[source.indices[triangle*3+k%3]];
  };
  const auto point = [&](std::size_t triangle, std::size_t k) noexcept {
    return cage.positions[corner(triangle,k)];
  };
  const auto length2 = [](Point a, Point b) noexcept { return dot(a-b,a-b); };
  const auto mergeable = [&](std::size_t t, std::size_t k, std::size_t m) {
    const auto shared = length2(point(t,k),point(t,k+1))*(1.0+1.0e-6);
    if (shared < length2(point(t,k+1),point(t,k+2)) || shared < length2(point(t,k+2),point(t,k))
        || shared < length2(point(t+1,m+1),point(t+1,m+2)) || shared < length2(point(t+1,m+2),point(t+1,m)))
      return false;
    const auto n1 = cross(point(t,k+1)-point(t,k),point(t,k+2)-point(t,k));
    const auto n2 = cross(point(t+1,m+1)-point(t+1,m),point(t+1,m+2)-point(t+1,m));
    const auto scale = dot(n1,n1)*dot(n2,n2);
    if (!(scale > 0.0)) return false;
    const auto alignment = dot(n1,n2);
    if (alignment <= 0.0 || alignment*alignment < 0.25*scale) return false;
    return sameFaceData(t,t+1)
        && sameCornerData(t*3+k,(t+1)*3+(m+1)%3) && sameCornerData(t*3+(k+1)%3,(t+1)*3+m);
  };
  std::size_t t = 0;
  while (t < triangles) {
    bool merged = false;
    for (std::size_t k = 0; k < 3 && !merged && t+1 < triangles; ++k)
      for (std::size_t m = 0; m < 3 && !merged; ++m)
        if (corner(t,k) == corner(t+1,m+1) && corner(t,k+1) == corner(t+1,m) && mergeable(t,k,m)) {
          pushFace({t*3+(k+2)%3,t*3+k,(t+1)*3+(m+2)%3,t*3+(k+1)%3},t);
          merged = true;
        }
    if (merged) { t += 2; continue; }
    pushFace({t*3,t*3+1,t*3+2},t);
    ++t;
  }
  cage.faceStarts.push_back(cage.corners.size());
}
inline bool subdivideCage(Cage &cage, StableId &nextId, std::string &error) {
  const auto cornerCount = cage.corners.size();
  const auto faceCount = cage.faceStarts.size()-1;
  const auto vertexCount = cage.positions.size();
  const auto stride = cage.discreteChannels.size();
  struct HalfEdge final { std::uint64_t key = 0; std::size_t corner = 0, face = 0; };
  std::vector<HalfEdge> halfEdges(cornerCount);
  std::vector<Point> facePoints(faceCount);
  for (std::size_t face = 0; face < faceCount; ++face) {
    const auto begin = cage.faceStarts[face], end = cage.faceStarts[face+1];
    Point sum;
    for (auto corner = begin; corner < end; ++corner) {
      const auto next = corner+1 == end ? begin : corner+1;
      sum = sum+cage.positions[cage.corners[corner]];
      halfEdges[corner] = {edgeKey(cage.corners[corner],cage.corners[next]),corner,face};
    }
    facePoints[face] = sum*(1.0/static_cast<double>(end-begin));
  }
  std::sort(halfEdges.begin(),halfEdges.end(),[](const HalfEdge &a, const HalfEdge &b) noexcept {
    return a.key != b.key ? a.key < b.key : a.corner < b.corner;
  });
  std::size_t edgeCount = 0;
  for (std::size_t i = 0; i < halfEdges.size(); ++i)
    if (i == 0 || halfEdges[i].key != halfEdges[i-1].key) ++edgeCount;
  std::size_t outputVertices = 0;
  if (!checkedAdd(vertexCount,edgeCount,outputVertices) || !checkedAdd(outputVertices,faceCount,outputVertices)
      || outputVertices > std::numeric_limits<std::uint32_t>::max()
      || outputVertices > static_cast<std::size_t>(kMaximumStableId)
      || nextId > kMaximumStableId-static_cast<StableId>(outputVertices)) {
    error = "Subdivision Surface overflows its 32-bit vertex index or stable identity: "
        +std::to_string(vertexCount)+" vertices, "+std::to_string(edgeCount)+" edges and "+std::to_string(faceCount)
        +" faces need "+countsAndBytes("vertices",outputVertices,sizeof(Vec3));
    return false;
  }
  struct Accumulator final {
    Point faceSum, midpointSum, sharpNeighbors;
    double sharpnessSum = 0;
    std::uint32_t faces = 0, edges = 0, sharp = 0;
  };
  std::vector<Accumulator> accumulators(vertexCount);
  for (std::size_t face = 0; face < faceCount; ++face)
    for (auto corner = cage.faceStarts[face]; corner < cage.faceStarts[face+1]; ++corner) {
      auto &entry = accumulators[cage.corners[corner]];
      entry.faceSum = entry.faceSum+facePoints[face];
      ++entry.faces;
    }
  Cage next;
  next.positions.resize(outputVertices);
  next.ids.resize(outputVertices);
  std::copy(cage.ids.begin(),cage.ids.end(),next.ids.begin());
  std::vector<std::size_t> edgeOfCorner(cornerCount);
  std::size_t edge = 0;
  for (std::size_t i = 0; i < halfEdges.size(); ++edge) {
    auto j = i;
    while (j < halfEdges.size() && halfEdges[j].key == halfEdges[i].key) edgeOfCorner[halfEdges[j++].corner] = edge;
    const auto a = static_cast<std::uint32_t>(halfEdges[i].key >> 32u);
    const auto b = static_cast<std::uint32_t>(halfEdges[i].key & 0xffffffffull);
    const auto pa = cage.positions[a], pb = cage.positions[b];
    const auto midpoint = (pa+pb)*0.5;
    double sharpness = 0;
    if (!cage.sharpEdges.empty()) {
      const auto at = std::lower_bound(cage.sharpEdges.begin(),cage.sharpEdges.end(),halfEdges[i].key,
          [](const SharpEdge &entry, std::uint64_t key) noexcept { return entry.key < key; });
      if (at != cage.sharpEdges.end() && at->key == halfEdges[i].key) sharpness = at->sharpness;
    }
    const auto edgeVertex = vertexCount+edge;
    const bool manifold = j-i == 2;
    Point position = midpoint;
    if (manifold && sharpness < 1.0) {
      const auto smooth = (pa+pb+facePoints[halfEdges[i].face]+facePoints[halfEdges[i+1].face])*0.25;
      position = lerp(smooth,midpoint,sharpness);
    }
    next.positions[edgeVertex] = position;
    next.ids[edgeVertex] = nextId++;
    const auto vertexSharpness = manifold ? std::min(sharpness,1.0) : 1.0;
    for (const auto& pair : {std::make_pair(a,pb),std::make_pair(b,pa)}) {
      auto &entry = accumulators[pair.first];
      entry.midpointSum = entry.midpointSum+midpoint;
      ++entry.edges;
      if (vertexSharpness > 0.0) {
        ++entry.sharp;
        entry.sharpNeighbors = entry.sharpNeighbors+pair.second;
        entry.sharpnessSum += vertexSharpness;
      }
    }
    if (manifold && sharpness > 1.0) {
      const auto remaining = sharpness-1.0;
      next.sharpEdges.push_back({edgeKey(a,static_cast<std::uint32_t>(edgeVertex)),remaining});
      next.sharpEdges.push_back({edgeKey(b,static_cast<std::uint32_t>(edgeVertex)),remaining});
    }
    i = j;
  }
  std::sort(next.sharpEdges.begin(),next.sharpEdges.end(),[](const SharpEdge &a, const SharpEdge &b) noexcept {
    return a.key < b.key;
  });
  for (std::size_t vertex = 0; vertex < vertexCount; ++vertex) {
    const auto &entry = accumulators[vertex];
    const auto original = cage.positions[vertex];
    if (entry.faces == 0 || entry.edges == 0) { next.positions[vertex] = original; continue; }
    const double valence = entry.edges;
    const auto smooth = (entry.faceSum*(1.0/static_cast<double>(entry.faces))
        +entry.midpointSum*(2.0/valence)+original*(valence-3.0))*(1.0/valence);
    if (entry.sharp < 2) { next.positions[vertex] = smooth; continue; }
    const auto sharp = entry.sharp == 2 ? (entry.sharpNeighbors+original*6.0)*0.125 : original;
    next.positions[vertex] = lerp(smooth,sharp,std::min(entry.sharpnessSum/static_cast<double>(entry.sharp),1.0));
  }
  for (std::size_t face = 0; face < faceCount; ++face) {
    next.positions[vertexCount+edgeCount+face] = facePoints[face];
    next.ids[vertexCount+edgeCount+face] = nextId++;
  }
  std::vector<Accumulator>().swap(accumulators);
  std::vector<HalfEdge>().swap(halfEdges);
  std::size_t nextCorners = 0;
  if (!checkedMul(cornerCount,4,nextCorners)) {
    error = "Subdivision Surface overflows: "+countsAndBytes("corners",cornerCount,sizeof(std::uint32_t))+" quadruple at the next level";
    return false;
  }
  next.corners.resize(nextCorners);
  next.faceStarts.resize(cornerCount+1);
  next.faceOrigins.resize(cornerCount);
  next.discreteChannels = cage.discreteChannels;
  if (stride != 0) next.cornerData.resize(nextCorners*stride);
  std::vector<double> center(stride);
  std::size_t child = 0;
  for (std::size_t face = 0; face < faceCount; ++face) {
    const auto begin = cage.faceStarts[face], end = cage.faceStarts[face+1];
    if (stride != 0) {
      std::fill(center.begin(),center.end(),0.0);
      for (auto corner = begin; corner < end; ++corner)
        for (std::size_t c = 0; c < stride; ++c) center[c] += cage.cornerData[corner*stride+c];
      for (auto &value : center) value /= static_cast<double>(end-begin);
    }
    const auto centerVertex = static_cast<std::uint32_t>(vertexCount+edgeCount+face);
    for (auto corner = begin; corner < end; ++corner, ++child) {
      const auto after = corner+1 == end ? begin : corner+1;
      const auto before = corner == begin ? end-1 : corner-1;
      next.faceStarts[child] = child*4;
      next.faceOrigins[child] = cage.faceOrigins[face];
      next.corners[child*4] = cage.corners[corner];
      next.corners[child*4+1] = static_cast<std::uint32_t>(vertexCount+edgeOfCorner[corner]);
      next.corners[child*4+2] = centerVertex;
      next.corners[child*4+3] = static_cast<std::uint32_t>(vertexCount+edgeOfCorner[before]);
      for (std::size_t c = 0; c < stride; ++c) {
        const auto here = cage.cornerData[corner*stride+c];
        const bool discrete = cage.discreteChannels[c] != 0;
        auto *out = next.cornerData.data()+child*4*stride+c;
        out[0] = here;
        out[stride] = discrete ? here : (here+cage.cornerData[after*stride+c])*0.5;
        out[2*stride] = discrete ? here : center[c];
        out[3*stride] = discrete ? here : (here+cage.cornerData[before*stride+c])*0.5;
      }
    }
  }
  next.faceStarts[cornerCount] = nextCorners;
  cage = std::move(next);
  return true;
}
// Each level follows V' = V+E+F, E' = 2E+C, F' = C, C' = 4C, so the exact final
// size is known before any level allocates. Only integer overflow fails here.
inline bool predictSubdivisionCounts(const Cage &cage, std::uint32_t level, std::size_t &finalIndices, std::string &error) {
  std::vector<std::uint64_t> keys(cage.corners.size());
  for (std::size_t face = 0; face+1 < cage.faceStarts.size(); ++face) {
    const auto begin = cage.faceStarts[face], end = cage.faceStarts[face+1];
    for (auto corner = begin; corner < end; ++corner)
      keys[corner] = edgeKey(cage.corners[corner],cage.corners[corner+1 == end ? begin : corner+1]);
  }
  std::sort(keys.begin(),keys.end());
  const auto describe = [&](std::size_t vertices, std::size_t edges, std::size_t faces, std::size_t corners, std::uint32_t step) {
    return "Subdivision Surface level "+std::to_string(level)+" overflows at step "+std::to_string(step)+" from "
        +std::to_string(vertices)+" vertices, "+std::to_string(edges)+" edges, "+std::to_string(faces)+" faces and "
        +countsAndBytes("corners",corners,sizeof(std::uint32_t));
  };
  std::size_t vertices = cage.positions.size();
  std::size_t edges = static_cast<std::size_t>(std::unique(keys.begin(),keys.end())-keys.begin());
  std::size_t faces = cage.faceStarts.size()-1;
  std::size_t corners = cage.corners.size();
  for (std::uint32_t step = 1; step <= level; ++step) {
    std::size_t nextVertices = 0, nextEdges = 0, nextCorners = 0;
    if (!checkedAdd(vertices,edges,nextVertices) || !checkedAdd(nextVertices,faces,nextVertices)
        || nextVertices > std::numeric_limits<std::uint32_t>::max()
        || !checkedMul(edges,2,nextEdges) || !checkedAdd(nextEdges,corners,nextEdges)
        || !checkedMul(corners,4,nextCorners)) {
      error = describe(vertices,edges,faces,corners,step);
      return false;
    }
    faces = corners;
    vertices = nextVertices; edges = nextEdges; corners = nextCorners;
  }
  if (!checkedMul(corners/4,6,finalIndices)) {
    error = describe(vertices,edges,faces,corners,level);
    return false;
  }
  return true;
}
} // namespace subdivision_detail

inline bool materializeSubdivisionSurface(const GeometryData &source, const std::vector<AttributeData> &attributes,
                                          const std::vector<SubdivisionCrease> &creases, std::uint32_t level,
                                          GeometryData &result, std::vector<AttributeData> &resultAttributes,
                                          std::string &error) {
  using namespace subdivision_detail;
  if (level == 0 || source.indices.empty()) {
    result = source; resultAttributes = attributes; error.clear();
    return true;
  }
  const auto triangles = source.indices.size()/3;
  if (source.indices.size()%3 != 0 || source.vertexIds.size() != source.positions.size()
      || (!source.uv.empty() && source.uv.size() != source.positions.size())
      || std::any_of(source.indices.begin(),source.indices.end(),[&](std::uint32_t index) { return index >= source.positions.size(); })) {
    error = "Subdivision Surface requires complete triangle topology";
    return false;
  }
  Channels channels;
  channels.uv = !source.uv.empty();
  channels.stride = channels.uv ? 3 : 0;
  channels.offsets.assign(attributes.size(),0);
  std::vector<std::uint8_t> discrete(channels.stride,0);
  for (std::size_t a = 0; a < attributes.size(); ++a) {
    const auto &attribute = attributes[a];
    const auto domain = attribute.descriptor.domain;
    const auto expected = domain == Domain::vertex ? source.positions.size()
        : domain == Domain::faceCorner ? source.indices.size() : domain == Domain::face ? triangles : attribute.elements.size();
    if (attribute.elements.size() != expected) {
      error = "Subdivision Surface attribute '"+attribute.descriptor.name+"' has "+std::to_string(attribute.elements.size())
          +" elements for a "+std::to_string(expected)+" element domain";
      return false;
    }
    if (carriesNormal(attribute) || (domain != Domain::vertex && domain != Domain::faceCorner)) continue;
    channels.offsets[a] = channels.stride;
    channels.stride += std::tuple_size<decltype(FieldElement::components)>::value;
    discrete.resize(channels.stride,interpolates(attribute) ? 0 : 1);
  }
  std::size_t finalIndices = 0;
  try {
    Cage cage;
    cage.discreteChannels = std::move(discrete);
    std::vector<std::uint32_t> cageOfVertex;
    weldCage(source,cageOfVertex,cage);
    loadCreases(source,cageOfVertex,creases,cage);
    buildCageFaces(source,attributes,channels,cageOfVertex,cage);
    if (!predictSubdivisionCounts(cage,level,finalIndices,error)) return false;
    StableId nextId = 0;
    for (const auto id : source.vertexIds) nextId = std::max(nextId,id);
    if (nextId == kMaximumStableId) {
      error = "Subdivision Surface has no stable identity above vertex "+std::to_string(nextId);
      return false;
    }
    ++nextId;
    for (std::uint32_t step = 0; step < level; ++step)
      if (!subdivideCage(cage,nextId,error)) return false;
    const auto stride = channels.stride;
    const auto cornerCount = cage.corners.size();
    std::vector<std::uint32_t> outputVertex;
    std::vector<std::size_t> representative;
    GeometryData mesh;
    if (stride == 0) {
      outputVertex = std::move(cage.corners);
      mesh.vertexIds = std::move(cage.ids);
      mesh.positions.reserve(cage.positions.size());
      for (const auto &point : cage.positions)
        mesh.positions.push_back({static_cast<float>(point.x),static_cast<float>(point.y),static_cast<float>(point.z)});
    } else {
      std::vector<std::size_t> order(cornerCount);
      for (std::size_t i = 0; i < cornerCount; ++i) order[i] = i;
      const auto data = [&](std::size_t corner) noexcept { return cage.cornerData.data()+corner*stride; };
      std::sort(order.begin(),order.end(),[&](std::size_t a, std::size_t b) noexcept {
        if (cage.corners[a] != cage.corners[b]) return cage.corners[a] < cage.corners[b];
        const auto *left = data(a), *right = data(b);
        for (std::size_t c = 0; c < stride; ++c) if (left[c] != right[c]) return left[c] < right[c];
        return a < b;
      });
      outputVertex.resize(cornerCount);
      for (std::size_t i = 0; i < cornerCount; ++i) {
        const auto corner = order[i];
        const bool sameVertex = i > 0 && cage.corners[order[i-1]] == cage.corners[corner];
        const bool same = sameVertex && std::equal(data(corner),data(corner)+stride,data(order[i-1]));
        if (!same) {
          if (representative.size() >= std::numeric_limits<std::uint32_t>::max() || (sameVertex && nextId >= kMaximumStableId)) {
            error = "Subdivision Surface overflows its 32-bit vertex index or stable identity with "
                +countsAndBytes("vertices",representative.size()+1,sizeof(Vec3));
            return false;
          }
          const auto &point = cage.positions[cage.corners[corner]];
          mesh.positions.push_back({static_cast<float>(point.x),static_cast<float>(point.y),static_cast<float>(point.z)});
          mesh.vertexIds.push_back(sameVertex ? nextId++ : cage.ids[cage.corners[corner]]);
          representative.push_back(corner);
        }
        outputVertex[corner] = static_cast<std::uint32_t>(representative.size()-1);
      }
    }
    mesh.indices.resize(finalIndices);
    for (std::size_t quad = 0; quad < cornerCount/4; ++quad) {
      const auto *corners = outputVertex.data()+quad*4;
      auto *out = mesh.indices.data()+quad*6;
      out[0] = corners[0]; out[1] = corners[1]; out[2] = corners[2];
      out[3] = corners[0]; out[4] = corners[2]; out[5] = corners[3];
    }
    if (channels.uv) {
      mesh.uv.reserve(representative.size());
      for (const auto corner : representative) {
        const auto *values = cage.cornerData.data()+corner*stride;
        mesh.uv.push_back({static_cast<float>(values[0]),static_cast<float>(values[1]),static_cast<float>(values[2])});
      }
    }
    std::vector<AttributeData> outputAttributes;
    outputAttributes.reserve(attributes.size());
    const auto fromCorner = [&](const AttributeData &attribute, std::size_t offset, std::size_t corner) {
      FieldElement element;
      std::copy_n(cage.cornerData.data()+corner*stride+offset,element.components.size(),element.components.begin());
      if (attribute.descriptor.interpolation == Interpolation::normalizedLinear) {
        const auto &c = element.components;
        const auto length = std::sqrt(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]);
        if (length > 0.0) for (std::size_t k = 0; k < 3; ++k) element.components[k] /= length;
      }
      return element;
    };
    for (std::size_t a = 0; a < attributes.size(); ++a) {
      const auto &attribute = attributes[a];
      if (carriesNormal(attribute)) continue;
      AttributeData output;
      output.descriptor = attribute.descriptor;
      if (attribute.descriptor.domain == Domain::vertex) {
        output.elements.reserve(representative.size());
        for (const auto corner : representative) output.elements.push_back(fromCorner(attribute,channels.offsets[a],corner));
      } else if (attribute.descriptor.domain == Domain::faceCorner) {
        output.elements.reserve(finalIndices);
        for (std::size_t quad = 0; quad < cornerCount/4; ++quad)
          for (const std::size_t k : {0u,1u,2u,0u,2u,3u})
            output.elements.push_back(fromCorner(attribute,channels.offsets[a],quad*4+k));
      } else if (attribute.descriptor.domain == Domain::face) {
        output.elements.reserve(cornerCount/2);
        for (std::size_t quad = 0; quad < cornerCount/4; ++quad) {
          output.elements.push_back(attribute.elements[cage.faceOrigins[quad]]);
          output.elements.push_back(attribute.elements[cage.faceOrigins[quad]]);
        }
      } else {
        output.elements = attribute.elements;
      }
      outputAttributes.push_back(std::move(output));
    }
    result = std::move(mesh);
    resultAttributes = std::move(outputAttributes);
    error.clear();
    return true;
  } catch (const std::bad_alloc &) {
  } catch (const std::length_error &) {
  }
  error = "Subdivision Surface could not allocate level "+std::to_string(level)+" of "
      +std::to_string(source.positions.size())+" vertices and "+std::to_string(triangles)+" triangles: "
      +countsAndBytes("indices",finalIndices,sizeof(std::uint32_t))+" plus their vertices";
  return false;
}

inline bool equalSubdivision(const SubdivisionSurfaceRecord &a, const SubdivisionSurfaceRecord &b) noexcept {
  return a.creases.size() == b.creases.size()
      && std::equal(a.creases.begin(),a.creases.end(),b.creases.begin(),
          [](const SubdivisionCrease &x, const SubdivisionCrease &y) noexcept {
            return x.vertexA == y.vertexA && x.vertexB == y.vertexB && x.weight == y.weight;
          });
}
inline bool validSubdivisionCreases(const std::vector<SubdivisionCrease> &creases) noexcept {
  for (std::size_t i = 0; i < creases.size(); ++i) {
    const auto &crease = creases[i];
    if (!validStableId(crease.vertexA) || !validStableId(crease.vertexB) || crease.vertexA >= crease.vertexB
        || !finite(crease.weight) || crease.weight < 0.0f || crease.weight > 1.0f) return false;
    if (i > 0 && !(std::make_pair(creases[i-1].vertexA,creases[i-1].vertexB)
                   < std::make_pair(crease.vertexA,crease.vertexB))) return false;
  }
  return true;
}

// The level a cage edge needs so its subdivided pieces span at most targetEdgePixels
// on screen. Each level halves an edge. Frame replay calls it through
// adaptiveCageLevel when a Subdivision Surface has a dicing rate and the frame
// owner supplies a view.
inline std::uint32_t adaptiveSubdivisionLevel(double cageEdgeLength, double cameraDistance,
                                              double tanHalfVerticalFov, double viewportHeightPixels,
                                              double targetEdgePixels, std::uint32_t minimumLevel) noexcept {
  const auto positive = [](double value) noexcept { return std::isfinite(value) && value > 0.0; };
  if (!positive(cageEdgeLength) || !positive(cameraDistance) || !positive(tanHalfVerticalFov)
      || !positive(viewportHeightPixels) || !positive(targetEdgePixels)) return minimumLevel;
  auto pixels = cageEdgeLength*viewportHeightPixels/(2.0*cameraDistance*tanHalfVerticalFov);
  if (!std::isfinite(pixels)) return minimumLevel;
  std::uint32_t level = 0;
  while (pixels > targetEdgePixels) { pixels *= 0.5; ++level; }
  return std::max(level,minimumLevel);
}
// A dicing rate above this many pixels is refused; 0 turns dicing off.
inline constexpr float kMaximumSubdivisionDicingRate = 1000.0f;
// The level for a whole cage seen from a camera: its positions go to camera space
// through localToCamera (row-major 3x4), so object scale and rotation count. The
// longest camera-space edge is measured at the nearest point of the camera-space
// bounds, never below the authored level. A camera inside the bounds sits at the
// near distance 0.1, the native near plane.
inline std::uint32_t adaptiveCageLevel(const GeometryData &cage, double dicingRatePixels,
                                       const std::array<double,12> &localToCamera, double tanHalfVerticalFov,
                                       double viewportHeightPixels, std::uint32_t authoredLevel) {
  if (cage.positions.empty()) return authoredLevel;
  const auto &m = localToCamera;
  std::vector<std::array<double,3>> view(cage.positions.size());
  for (std::size_t i = 0; i < cage.positions.size(); ++i) {
    const double x = cage.positions[i].x, y = cage.positions[i].y, z = cage.positions[i].z;
    view[i] = {m[0]*x + m[1]*y + m[2]*z + m[3], m[4]*x + m[5]*y + m[6]*z + m[7], m[8]*x + m[9]*y + m[10]*z + m[11]};
  }
  double longest = 0;
  for (std::size_t t = 0; t + 2 < cage.indices.size(); t += 3)
    for (std::size_t corner = 0; corner < 3; ++corner) {
      const auto a = cage.indices[t + corner], b = cage.indices[t + (corner + 1) % 3];
      if (a >= view.size() || b >= view.size()) continue;
      const double dx = view[a][0] - view[b][0], dy = view[a][1] - view[b][1], dz = view[a][2] - view[b][2];
      longest = std::max(longest, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
  auto low = view[0], high = low;
  for (const auto &v : view)
    for (std::size_t axis = 0; axis < 3; ++axis) {
      low[axis] = std::min(low[axis], v[axis]); high[axis] = std::max(high[axis], v[axis]);
    }
  double squared = 0;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const auto gap = std::max({low[axis], 0.0, -high[axis]});
    squared += gap * gap;
  }
  return adaptiveSubdivisionLevel(longest, std::max(std::sqrt(squared), 0.1), tanHalfVerticalFov,
                                  viewportHeightPixels, dicingRatePixels, authoredLevel);
}
// Whether a frame owner must replay the plan with its view so dicing can choose levels.
inline bool hasAdaptiveSubdivision(const ValueDescriptor &value) noexcept {
  return std::any_of(value.operations.begin(), value.operations.end(), [](const OperationRecord &op) {
    return op.code == OperationCode::subdivisionSurface && op.parameters[3] > 0.0f;
  });
}
// These existing typed operations preserve the sampled instance domain.
inline bool isSampledInstanceFieldComposition(OperationCode code) noexcept {
  return (code>=OperationCode::fieldMath && code<=OperationCode::fieldClamp)
      || code==OperationCode::fieldComposeVector || code==OperationCode::fieldIntegerToFloat
      || code==OperationCode::fieldSelect || code==OperationCode::fieldFloatToInteger;
}

inline std::set<StableId> orderedInstanceScorePlacementAncestors(const ValueDescriptor& value) {
  std::map<StableId,std::set<StableId>> parents;
  for (const auto& op : value.operations) {
    if (op.code==OperationCode::copy || op.code==OperationCode::grid
        || (op.code>=OperationCode::plane && op.code<=OperationCode::cylinder)
        || op.code==OperationCode::curveLine || op.code==OperationCode::curveCircle
        || op.code==OperationCode::retainedMesh || op.code==OperationCode::sdfSurface
        || op.code==OperationCode::fieldConstant) continue;
    const owner_detail::CoreAddParent add{parents,op};
    add(op.inputStableId);
    const bool fieldInputs=isConstructedFieldOperation(op.code) && op.code!=OperationCode::materialTableSlot;
    if (fieldInputs || op.code==OperationCode::instanceOnPoints || op.code==OperationCode::spectrumInstancer
        || op.code==OperationCode::vectorDisplacement || op.code==OperationCode::joinGeometry
        || op.code==OperationCode::curveSetRadius || op.code==OperationCode::curveSetTilt) add(op.secondaryInputStableId);
    if (fieldInputs) add(op.field.auxiliaryInputStableId);
  }
  std::vector<StableId> pending;
  for (const auto& binding : value.scoreFields)
    for (const auto& op : value.operations)
      if (op.code==OperationCode::evaluateField && op.stableId==binding.fieldStableId
          && op.outputStableId==binding.fieldStableId && op.inputStableId==binding.sourceGeometryStableId
          && op.inputStableId!=op.outputStableId) pending.push_back(op.inputStableId);
  std::set<StableId> ancestors;
  while (!pending.empty()) {
    const auto id=pending.back(); pending.pop_back();
    if (!ancestors.insert(id).second) continue;
    const auto found=parents.find(id);
    if (found!=parents.end()) pending.insert(pending.end(),found->second.begin(),found->second.end());
  }
  return ancestors;
}

// A Set Position displacement by a spectrum or score binding is added at the scene adapter
// after replay while only transform, material, field and deformer operations follow its
// consumer, so those plans render as they always did. When an operation that changes the
// element layout follows, the frame owner adds it in replay order at its consumer
// instead, and that operation reads the displaced mesh.
inline bool keepsDisplacementLayout(OperationCode code) noexcept {
  return code==OperationCode::transform || code==OperationCode::transformTRS
      || code==OperationCode::setMaterial || code==OperationCode::assignMaterial
      || code==OperationCode::audioDeformer || code==OperationCode::vectorDisplacement
      || code==OperationCode::constantVectorField || code==OperationCode::evaluateField
      || isConstructedFieldOperation(code) || isElementFieldQuery(code);
}
inline bool displacementInReplay(const ValueDescriptor& value, StableId consumer) {
  if (value.carrier!=CarrierKind::geometry3D || consumer==0 || consumer==value.stableId) return false;
  const auto at=std::find_if(value.operations.begin(),value.operations.end(),[&](const auto& op) {
    return op.outputStableId==consumer && op.code==OperationCode::transformTRS;
  });
  return at!=value.operations.end() && std::any_of(std::next(at),value.operations.end(),[](const auto& op) {
    return !keepsDisplacementLayout(op.code);
  });
}
// Bindings the scene adapter adds by vertex: a spectrum displacement without its own
// Audio Spectrum Field operation, and a vertex Score Field.
inline bool adapterSpectrumDisplacement(const ValueDescriptor& value, const spectrum::Binding& binding) {
  return binding.target==spectrum::Target::vertexDisplacement
      && std::none_of(value.operations.begin(),value.operations.end(),[&](const auto& op) {
        return op.code==OperationCode::fieldAudioSpectrum && op.stableId==binding.fieldStableId;
      });
}
inline bool adapterScoreDisplacement(const ValueDescriptor& value, const scorefield::Binding& binding) {
  return binding.consumerGeometryStableId!=0
      && std::none_of(value.operations.begin(),value.operations.end(),[&](const auto& op) {
        return op.code==OperationCode::evaluateField && op.stableId==binding.fieldStableId
            && op.inputStableId!=op.outputStableId;
      });
}
inline bool displacedInReplay(const ValueDescriptor& value, StableId id) {
  return displacementInReplay(value,id)
      && (std::any_of(value.spectrumFields.begin(),value.spectrumFields.end(),[&](const auto& binding) {
            return binding.consumerGeometryStableId==id && adapterSpectrumDisplacement(value,binding);
          })
          || std::any_of(value.scoreFields.begin(),value.scoreFields.end(),[&](const auto& binding) {
            return binding.consumerGeometryStableId==id && adapterScoreDisplacement(value,binding);
          }));
}
// True when every binding on this Geometry3D either deforms in replay order or is a
// Set Position displacement that a following layout change moves into replay order.
inline bool layoutChangeFollowsBindings(const ValueDescriptor& value) {
  return value.carrier==CarrierKind::geometry3D
      && std::all_of(value.spectrumFields.begin(),value.spectrumFields.end(),[&](const auto& binding) {
           return binding.target==spectrum::Target::geometryDeformation || adapterSpectrumDisplacement(value,binding);
         })
      && std::all_of(value.scoreFields.begin(),value.scoreFields.end(),[&](const auto& binding) {
           return adapterScoreDisplacement(value,binding);
         });
}
// Density points whose surface Audio Deformer bindings move each frame.
inline bool densityFollowsBindings(const ValueDescriptor& value) {
  return !value.spectrumFields.empty()
      && std::any_of(value.operations.begin(),value.operations.end(),[](const auto& operation) {
           return operation.code==OperationCode::pointsOnFacesDensity;
         });
}
// Applies the frame owner's Set Position displacement for consumer to the retained mesh.
inline bool displaceReplayMesh(std::map<StableId,GeometryData>& geometries, StableId mesh, StableId consumer,
                               const RuntimeFieldEvaluation& runtime, std::string& error) {
  const auto found=geometries.find(mesh);
  if (!runtime.displaceConsumer || found==geometries.end()) {
    error="Set Position displacement requires its frame owner and retained mesh"; return false;
  }
  const auto vertices=found->second.positions.size();
  if (!runtime.displaceConsumer(found->second,consumer,error)) return false;
  if (found->second.positions.size()!=vertices
      || !std::all_of(found->second.positions.begin(),found->second.positions.end(),[](const auto& p) { return finite(p); })) {
    error="Set Position displacement must keep finite positions and the vertex count"; return false;
  }
  return true;
}
inline bool hasReplayDisplacement(const ValueDescriptor& value) {
  return std::any_of(value.operations.begin(),value.operations.end(),[&](const auto& op) {
    return displacedInReplay(value,op.outputStableId);
  });
}

#if defined(__GNUC__)
#pragma GCC diagnostic push
// These switches name the operation codes they accept and reject the rest in default.
#pragma GCC diagnostic ignored "-Wswitch-enum"
#endif
inline bool validateOperationPlan(const ValueDescriptor &value,
                                  const AdmissionContext &context,
                                  MaterializedInstancePlan *materialized,
                                  std::string &error,
                                  ValueDescriptor* replayed = nullptr,
                                  const RuntimeFieldEvaluation* runtime = nullptr,
                                  const std::function<bool(GeometryData&, const spectrum::Binding&,
                                                           std::string&)>& frameDeformer = {},
                                  const ResourceLimits* replayLimits = nullptr) {
  std::map<StableId, CarrierKind> produced;
  std::map<StableId, StableId> materials;
  std::map<StableId, std::set<StableId>> dependencies;
  std::set<StableId> operationIds;
  std::set<StableId> generatedOutputs;
  std::set<StableId> auxiliaryOutputs;
  std::set<StableId> deferredOutputs, bodyOutputs;
  MaterializedInstancePlan local;
  std::map<StableId, InstancesData> instanceValues;
  std::map<StableId, FieldData> fields;
  std::map<StableId, StableId> sampledInstanceOwners;
  std::map<StableId, std::set<StableId>> sampledInstanceFieldSources;
  std::map<StableId, std::pair<Domain, ValueType>> fieldTypes;
  std::map<StableId, std::vector<AttributeData>> retainedAttributes;
  const auto scorePlacementAncestors=orderedInstanceScorePlacementAncestors(value);
  bool hasStoredAttributes = false;
  std::size_t fieldBytes = 0;
  const owner_detail::CoreRetainedDataLookup retainedData{fields,local,instanceValues};
  bool sawInstanceOperation = false;
  const bool scorePrototype = value.carrier==CarrierKind::instances3D
      && std::any_of(value.scoreFields.begin(),value.scoreFields.end(),[](const auto& binding) {
        return binding.consumerGeometryStableId!=0;
      });
  if (scorePrototype && (value.scoreFields.size()!=1 || !value.spectrumFields.empty()
      || hasRuntimeFields(value) || !context.deferredPoints.empty()
      || std::any_of(value.operations.begin(),value.operations.end(),[](const auto& op) {
        return op.retainedMesh && !op.retainedMesh->importedAnimation.empty();
      }))) {
    error="Score-deformed instancing requires one static retained prototype and Points3D source";
    return false;
  }

  for (const auto &operation : value.operations) {
    if (!validStableId(operation.stableId) ||
        !operationIds.insert(operation.stableId).second ||
        !validStableId(operation.outputStableId) ||
        (operation.code != OperationCode::instanceOnPoints
         && operation.code != OperationCode::spectrumInstancer
         && operation.code != OperationCode::vectorDisplacement
         && operation.code != OperationCode::joinGeometry
         && operation.code != OperationCode::assignMaterial
         && operation.code != OperationCode::curveSetRadius
         && operation.code != OperationCode::curveSetTilt
         && !isConstructedFieldOperation(operation.code)
         && operation.secondaryInputStableId != 0) ||
        (operation.code != OperationCode::transformTRS && operation.code != OperationCode::transformInstances
         && !equal(operation.transform,Transform{})) ||
        (!isConstructedFieldOperation(operation.code) && !(operation.field == FieldOperationSettings{})) ||
        (operation.code != OperationCode::retainedMesh && operation.retainedMesh) ||
        (operation.code != OperationCode::sdfSurface && operation.sdfSurface) ||
        (operation.code != OperationCode::shadeMesh && !operation.sharpEdges.empty()) ||
        (operation.code != OperationCode::subdivisionSurface && operation.subdivision) ||        !std::all_of(operation.parameters.begin(), operation.parameters.end(),
                     owner_detail::CoreParametersFinitePredicate{})) {
      error = "geometry operation plan is invalid";
      return false;
    }
    const owner_detail::CoreRequireInput requireInput{produced,operation};
    const owner_detail::CoreGenerateOutput generate{operation,generatedOutputs,produced,
        materials,local,dependencies,retainedAttributes,fieldBytes};

    const auto instanceOwner=sampledInstanceOwners.find(operation.inputStableId);
    const bool spectrumInstanceOwner=instanceOwner!=sampledInstanceOwners.end()
        && std::any_of(value.operations.begin(),value.operations.end(),[&](const auto& op) {
          return op.code==OperationCode::spectrumInstancer && op.outputStableId==instanceOwner->second;
        });
    if (instanceOwner!=sampledInstanceOwners.end()
        && operation.code!=OperationCode::fieldAudioSpectrum
        && operation.code!=OperationCode::evaluateField
        && !(!spectrumInstanceOwner && isElementFieldQuery(operation.code))
        && operation.code!=OperationCode::setInstanceAppearance
        && operation.code!=OperationCode::setMaterial
        && operation.code!=OperationCode::materialTableSlot) {
      error="Sampled instances support exact ordered field sampling and appearance only"; return false;
    }
    std::set<StableId> sampledInstanceSources;
    for (const auto input : {operation.inputStableId,operation.secondaryInputStableId,operation.field.auxiliaryInputStableId}) {
      const auto found=sampledInstanceFieldSources.find(input);
      if (found!=sampledInstanceFieldSources.end()) sampledInstanceSources.insert(found->second.begin(),found->second.end());
    }
    if (!sampledInstanceSources.empty() && operation.code!=OperationCode::setInstanceAppearance
        && !isSampledInstanceFieldComposition(operation.code)) {
      error="Sampled instance fields require deterministic typed composition and appearance"; return false;
    }
    if (instanceOwner!=sampledInstanceOwners.end() && operation.code!=OperationCode::fieldAudioSpectrum
        && operation.code!=OperationCode::evaluateField && !isElementFieldQuery(operation.code))
      sampledInstanceOwners[operation.outputStableId]=instanceOwner->second;

    if (isConstructedFieldOperation(operation.code)) {
      if (!validFieldOperationSettings(operation,error)) return false;
      const auto& settings=operation.field;
      const owner_detail::CoreGetField getField{fields,fieldTypes,settings};
      FieldData field;
      if (operation.code==OperationCode::materialTableSlot) {
        const auto source=produced.find(operation.inputStableId);
        const auto data=retainedData(operation.inputStableId);
        const auto& p=settings.values;
        if (source==produced.end() || !data
            || (source->second!=CarrierKind::geometry3D && source->second!=CarrierKind::instances3D)
            || settings.domain!=elementDomain(source->second) || settings.valueType!=ValueType::color
            || settings.mode>=32 || !validStableId(operation.secondaryInputStableId)
            || operation.parameters[0]<0 || operation.parameters[0]>1
            || operation.parameters[1]!=0 || operation.parameters[2]!=0 || operation.parameters[3]!=0
            || std::any_of(p.begin(),p.begin()+3,[](double v){return v<0 || v>64;})
            || std::any_of(p.begin()+3,p.begin()+6,[](double v){return v<0 || v>std::numeric_limits<float>::max();})
            || p[6]<0 || p[6]>1 || p[7]<0 || p[7]>1 || !generate(source->second)) {
          error="Material table requires a PBR material and a slot between zero and 31"; return false;
        }
        local.materialTables[operation.outputStableId][settings.mode]=operation;
        if (source->second==CarrierKind::geometry3D) local.geometries.emplace(operation.outputStableId,std::get<GeometryData>(*data));
        else instanceValues.emplace(operation.outputStableId,std::get<InstancesData>(*data));
        continue;
      }
      if (operation.code==OperationCode::storeNamedAttribute || operation.code==OperationCode::setNormal
          || operation.code==OperationCode::setInstanceAppearance || operation.code==OperationCode::setMaterialIndex) {
        const auto source=produced.find(operation.inputStableId);
        auto data=retainedData(operation.inputStableId);
        const auto* inputField=getField(operation.secondaryInputStableId,settings.valueType);
        FieldData positions;
        const bool faceIndex=operation.code==OperationCode::setMaterialIndex && source!=produced.end()
            && source->second==CarrierKind::geometry3D && settings.domain==Domain::face;
        const bool splineAttribute=operation.code==OperationCode::storeNamedAttribute && source!=produced.end()
            && source->second==CarrierKind::curves3D && settings.domain==Domain::spline;
        if (source==produced.end() || !data || !inputField
            || (!faceIndex && !splineAttribute && settings.domain!=elementDomain(source->second))
            || (operation.code==OperationCode::setNormal && source->second!=CarrierKind::geometry3D)
            || !generate(source->second)
            || !materializeElementField(*data,source->second,OperationCode::positionField,positions,kMaximumFieldElements,error)) {
          if (error.empty()) error="Named attribute requires a retained source and a matching typed field";
          return false;
        }
        if (faceIndex) positions.elements.resize(std::get<GeometryData>(*data).indices.size()/3);
        if (splineAttribute) positions.elements.resize(std::get<CurvesData>(*data).splines.size());
        if (operation.code==OperationCode::setMaterialIndex
            && (settings.attributeName!="materialIndex" || settings.valueType!=ValueType::integer
                || (!faceIndex && source->second!=CarrierKind::instances3D))) {
          error="Material indices require a face or instance Integer field"; return false;
        }
        const auto attributeCopies=operation.code==OperationCode::setInstanceAppearance
            && isInstanceSurfaceScalar(settings.attributeName) ? std::size_t{2} : std::size_t{1};
        if (positions.elements.size()>(kMaximumBufferBytes-fieldBytes)/(sizeof(FieldElement)*attributeCopies)) {
          error="Named attributes exceed the retained field memory budget"; return false;
        }
        fieldBytes+=positions.elements.size()*sizeof(FieldElement)*attributeCopies;
        if (operation.code==OperationCode::setInstanceAppearance) {
          for (const auto sampledSource : sampledInstanceSources) {
            const auto sampled=instanceValues.find(sampledSource);
            const auto sampledOwner=sampledInstanceOwners.find(sampledSource);
            const auto& instances=std::get<InstancesData>(*data).instances;
            if (instanceOwner==sampledInstanceOwners.end() || sampledOwner==sampledInstanceOwners.end()
                || sampledOwner->second!=instanceOwner->second || sampled==instanceValues.end()
                || sampled->second.instances.size()!=instances.size()
                || inputField->elements.size()!=instances.size()
                || !std::equal(instances.begin(),instances.end(),sampled->second.instances.begin(),[](const auto& a,const auto& b) {
                  return a.stableId==b.stableId && a.sourceStableId==b.sourceStableId && equal(a.transform,b.transform);
                })) {
              error="Sampled instance appearance field has a different source or stable element identity"; return false;
            }
          }
          const auto* mask=settings.auxiliaryInputStableId==0 ? nullptr
              : getField(settings.auxiliaryInputStableId,ValueType::boolean);
          if (source->second!=CarrierKind::instances3D
              || (settings.auxiliaryInputStableId!=0 && !mask)
              || !materializeInstanceAppearance(retainedAttributes[operation.outputStableId],
                  positions.elements.size(),*inputField,mask,operation,error)) return false;
          if (mask) dependencies[operation.outputStableId].insert(settings.auxiliaryInputStableId);
        } else if (!materializeStoredAttribute(retainedAttributes[operation.outputStableId],positions.elements.size(),
                                                *inputField,operation,error)) return false;
        if (source->second==CarrierKind::geometry3D) local.geometries.emplace(operation.outputStableId,std::get<GeometryData>(*data));
        else if (source->second==CarrierKind::points3D) local.points.emplace(operation.outputStableId,std::get<PointsData>(*data));
        else if (source->second==CarrierKind::curves3D) local.curves.emplace(operation.outputStableId,std::get<CurvesData>(*data));
        else instanceValues.emplace(operation.outputStableId,std::get<InstancesData>(*data));
        dependencies[operation.outputStableId].insert(operation.secondaryInputStableId);
        hasStoredAttributes=true;
        continue;
      }
      if (operation.code==OperationCode::fieldAudioSpectrum) {
        const auto source=produced.find(operation.inputStableId);
        const auto data=retainedData(operation.inputStableId);
        const auto binding=std::find_if(value.spectrumFields.begin(),value.spectrumFields.end(),[&](const auto& b) {
          return b.fieldStableId==operation.stableId;
        });
        const auto* deferred=binding==value.spectrumFields.end() ? nullptr
            : deferredSpectrumSource(value,operation,*binding,context);
        if (binding!=value.spectrumFields.end() && binding->history.enabled
            && context.deferredPoints.count(operation.inputStableId)!=0) {
          error="Spectral History Field does not support deferred topology"; return false;
        }
        if (deferred) {
          const auto maximumBytes=replayLimits ? replayLimits->maxBufferBytes : kMaximumBufferBytes;
          if ((runtime && context.evaluatedPoints.count(operation.inputStableId)==0)
              || deferred->capacity>(replayLimits ? replayLimits->maxFieldElements : kMaximumFieldElements)
              || deferred->capacity>(replayLimits ? replayLimits->maxPoints : kMaximumPoints)
              || (value.carrier==CarrierKind::instances3D
                  && deferred->capacity>(replayLimits ? replayLimits->maxInstances : kMaximumInstances))
              || fieldBytes>maximumBytes
              || deferred->capacity>(maximumBytes-fieldBytes)/(3*sizeof(FieldElement)+sizeof(double)+sizeof(float))) {
            error="Deferred Audio Spectrum Field requires its exact frame and admitted field capacity"; return false;
          }
        }
        std::vector<float> coordinates;
        if (instanceOwner!=sampledInstanceOwners.end()) {
          if (operation.field.domain!=Domain::instance
              || (spectrumInstanceOwner && operation.field.mode!=0 && operation.field.mode!=7)) {
            error="Spectrum Instancer fields must sample exact stable instance indices"; return false;
          }
          sampledInstanceSources.insert(operation.inputStableId);
        }
        if (source==produced.end() || !data || binding==value.spectrumFields.end()
            || binding->sourceGeometryStableId!=operation.inputStableId || binding->consumerGeometryStableId!=0
            || !spectrumCoordinates(*data,source->second,retainedAttributes[operation.inputStableId],operation,*binding,coordinates,error,deferred!=nullptr)) {
          if (error.empty()) error="Audio Spectrum Field lost its retained typed source";
          return false;
        }
        if (!deferred && ((!runtime && coordinates!=binding->coordinates) || coordinates.size()!=binding->coordinates.size())) {
          error="Audio Spectrum Field coordinates disagree with the retained source topology"; return false;
        }
        std::vector<float> historyCoordinates;
        if (binding->history.enabled) {
          if (deferred || !spectrumHistoryCoordinates(*data,source->second,operation,historyCoordinates,error)) {
            if (error.empty()) error="Spectral History Field does not support deferred topology";
            return false;
          }
          if ((!runtime && historyCoordinates!=binding->history.coordinates)
              || historyCoordinates.size()!=binding->history.coordinates.size()) {
            error="Spectral History Field rows disagree with the retained source topology"; return false;
          }
        }
        if (!materializeSpectrumField(*binding,operation,coordinates,field,runtime,error,deferred!=nullptr,
            binding->history.enabled ? &historyCoordinates : nullptr)) return false;
      } else if (operation.code==OperationCode::fieldWaterHeight) {
        const auto source=local.geometries.find(operation.inputStableId);
        if (findNamedAttribute(retainedAttributes[operation.inputStableId],"normal")) {
          error="Water surface requires scene-derived normals; retained normal attributes are unavailable"; return false;
        }
        if (source==local.geometries.end()
            || !materializeWaterHeightField(operation,source->second,field,
                (kMaximumBufferBytes-fieldBytes)/sizeof(FieldElement),error,runtime)) {
          if (error.empty()) error="Water height field requires its retained rest grid";
          return false;
        }
      } else if (operation.code==OperationCode::fieldFaceIndex) {
        const auto source=local.geometries.find(operation.inputStableId);
        if (source==local.geometries.end() || settings.domain!=Domain::face || settings.valueType!=ValueType::integer
            || settings.mode!=0 || operation.secondaryInputStableId!=0) {
          error="Face index field requires retained triangle geometry"; return false;
        }
        field.elements.resize(source->second.indices.size()/3);
        for (std::size_t i=0;i<field.elements.size();++i) field.elements[i].components[0]=static_cast<double>(i);
      } else if (operation.code==OperationCode::readNamedAttribute) {
        const auto source=produced.find(operation.inputStableId);
        const auto* attribute=findNamedAttribute(retainedAttributes[operation.inputStableId],settings.attributeName);
        if (source==produced.end() || (settings.domain!=elementDomain(source->second)
            && !(source->second==CarrierKind::geometry3D && settings.domain==Domain::face)
            && !(source->second==CarrierKind::curves3D && settings.domain==Domain::spline)) || !attribute
            || attribute->descriptor.domain!=settings.domain || attribute->descriptor.valueType!=settings.valueType
            || operation.secondaryInputStableId!=0 || settings.mode!=0) {
          error="Named attribute is missing or has the wrong domain or value type"; return false;
        }
        field.elements=attribute->elements;
      } else if (operation.code==OperationCode::fieldSampleIndex) {
        const auto source=produced.find(operation.inputStableId);
        const auto* attribute=findNamedAttribute(retainedAttributes[operation.inputStableId],settings.attributeName);
        const auto* index=operation.secondaryInputStableId==0 ? nullptr : getField(operation.secondaryInputStableId,ValueType::integer);
        if (source==produced.end() || settings.domain!=elementDomain(source->second) || !attribute || !index
            || settings.mode!=0 || settings.auxiliaryInputStableId!=0
            || !materializeSampleIndexField(attribute,*index,operation,field,
                (kMaximumBufferBytes-fieldBytes)/sizeof(FieldElement),error)) {
          if (error.empty()) error="Sample index requires a retained source attribute and an Integer index field";
          return false;
        }
      } else if (isNearestFieldOperation(operation.code)) {
        const auto source=produced.find(operation.inputStableId);
        const auto data=retainedData(operation.inputStableId);
        const auto target=local.geometries.find(operation.secondaryInputStableId);
        if (source==produced.end() || !data || target==local.geometries.end()
            || !materializeNearestField(*data,source->second,target->second,
                retainedAttributes[operation.secondaryInputStableId],operation,field,
                (kMaximumBufferBytes-fieldBytes)/sizeof(FieldElement),error)) {
          if (error.empty()) error="Nearest vertex field requires retained source and target geometry";
          return false;
        }
      } else {
        const auto types=fieldInputTypes(operation.code,settings.valueType,settings.mode);
        const auto* a=operation.code==OperationCode::fieldConstant ? nullptr : getField(operation.inputStableId,types.first);
        const auto* b=operation.secondaryInputStableId==0 ? nullptr : getField(operation.secondaryInputStableId,types.second);
        const auto auxiliaryType=operation.code==OperationCode::fieldSelect ? ValueType::boolean : ValueType::floatValue;
        const auto* c=settings.auxiliaryInputStableId==0 ? nullptr : getField(settings.auxiliaryInputStableId,auxiliaryType);
        if ((operation.code==OperationCode::fieldConstant && (operation.inputStableId!=operation.outputStableId
              || operation.secondaryInputStableId!=0 || settings.auxiliaryInputStableId!=0))
            || (operation.code!=OperationCode::fieldConstant && !a)
            || (operation.secondaryInputStableId!=0 && !b) || (settings.auxiliaryInputStableId!=0 && !c)
            || !materializeConstructedField(operation,a,b,c,field,
                (kMaximumBufferBytes-fieldBytes)/sizeof(FieldElement),error,runtime)) {
          if (error.empty()) error="Field operation inputs have incompatible types or domains";
          return false;
        }
      }
      if (field.elements.size()>(kMaximumBufferBytes-fieldBytes)/sizeof(FieldElement)
          || !generate(CarrierKind::field)) { error="Constructed field exceeds its replay budget"; return false; }
      fieldBytes+=field.elements.size()*sizeof(FieldElement);
      fields.emplace(operation.outputStableId,std::move(field));
      if (!sampledInstanceSources.empty())
        sampledInstanceFieldSources.emplace(operation.outputStableId,std::move(sampledInstanceSources));
      fieldTypes.emplace(operation.outputStableId,std::make_pair(settings.domain,settings.valueType));
      if (operation.code==OperationCode::fieldConstant) dependencies[operation.outputStableId].clear();
      if (operation.secondaryInputStableId!=0) dependencies[operation.outputStableId].insert(operation.secondaryInputStableId);
      if (settings.auxiliaryInputStableId!=0) dependencies[operation.outputStableId].insert(settings.auxiliaryInputStableId);
      continue;
    }

    switch (operation.code) {
    case OperationCode::sdfSurface: {
      if(!operation.sdfSurface || !validStableId(operation.inputStableId)
          || !zeroParameters(operation) || !generate(CarrierKind::geometry3D)) {
        error="SDF surface has no immutable expression source"; return false;
      }
      GeometryData mesh; std::vector<AttributeData> attributes;
      if(!materializeSdfSurface(*operation.sdfSurface,operation.outputStableId,
          replayLimits ? replayLimits->maxVertices : kMaximumVertices,
          replayLimits ? replayLimits->maxIndices : kMaximumIndices,mesh,attributes,error)) return false;
      for(const auto& attribute:attributes)
        if(!addArrayBytes(attribute.elements.size(),sizeof(FieldElement),fieldBytes)
            || fieldBytes>kMaximumBufferBytes) { error="SDF surface attributes exceed replay memory"; return false; }
      dependencies[operation.outputStableId].clear();
      local.geometries.emplace(operation.outputStableId,std::move(mesh));
      retainedAttributes[operation.outputStableId]=std::move(attributes); hasStoredAttributes=true;
      break;
    }
    case OperationCode::retainedMesh: {
      if (!operation.retainedMesh || !validStableId(operation.inputStableId)
          || !zeroParameters(operation) || !generate(CarrierKind::geometry3D)) {
        error="Imported mesh has no immutable geometry source"; return false;
      }
      const GeometryData* selectedMesh=&operation.retainedMesh->geometry;
      const auto body=context.deferredBodyGeometry.find(operation.outputStableId);
      if (body!=context.deferredBodyGeometry.end()) {
        const auto& source=body->second;
        const auto points=context.deferredPoints.find(source.pointsOutputStableId);
        if (source.outputStableId!=body->first || operation.inputStableId!=source.terminalStableId
            || !validDeferredBodyGeometry(source,error) || points==context.deferredPoints.end()
            || !(points->second==source.points) || !operation.retainedMesh->attributes.empty()
            || !operation.retainedMesh->importedAnimation.empty()
            || !equal(operation.retainedMesh->geometry,*source.reference)) {
          if (error.empty()) error="Deferred body Geometry3D has no exact retained source and Points declaration";
          return false;
        }
        bodyOutputs.insert(operation.outputStableId);
        const auto frame=context.evaluatedBodyGeometry.find(operation.outputStableId);
        if (frame!=context.evaluatedBodyGeometry.end()) {
          const auto& sampled=frame->second;
          if (!sampled.geometry || sampled.source!=source.source
              || sampled.structuralRevision!=source.structuralRevision || sampled.absoluteTick<0
              || sampled.absoluteTick>9007199254740991LL
              || sampled.geometry->vertexIds!=source.reference->vertexIds
              || sampled.geometry->indices!=source.reference->indices
              || sampled.geometry->positions.size()!=source.reference->positions.size()) {
            error="Evaluated body Geometry3D differs from its declared source or topology"; return false;
          }
          selectedMesh=sampled.geometry.get();
        }
      }
      const auto& mesh=*selectedMesh;
      const auto& attributes=operation.retainedMesh->attributes;
      if (!operation.retainedMesh->importedAnimation.empty()) {
        visualanimationimport::Request request;
        if (!visualanimationoperation::decode(operation.retainedMesh->importedAnimation,request,error))
          return false;
        const auto hasIdentityAttribute=[&](const char* name,Domain domain) {
          return std::any_of(attributes.begin(),attributes.end(),[&](const auto& attribute) {
            return attribute.descriptor.name==name && attribute.descriptor.domain==domain
                && attribute.descriptor.valueType==ValueType::integer
                && attribute.descriptor.interpolation==Interpolation::nearest;
          });
        };
        if (request.sourceStableId!=operation.inputStableId
            || (request.meshStableId==0 && (!hasIdentityAttribute("importedObjectId",Domain::face)
                || !hasIdentityAttribute("importedNodeId",Domain::vertex)))) {
          error="Imported mesh animation request does not match its source identity"; return false;
        }
      }
      if (mesh.positions.empty() || mesh.positions.size()>kMaximumVertices || mesh.indices.empty()
          || mesh.indices.size()%3!=0
          || mesh.vertexIds.size()!=mesh.positions.size() || attributes.size()>kMaximumAttributes) {
        error="Imported mesh has no triangles, more vertices than 32-bit indices address, or incomplete topology";
        return false;
      }
      std::set<StableId> vertexIds, attributeIds;
      std::set<std::string> attributeNames;
      for (std::size_t i=0;i<mesh.positions.size();++i)
        if (!finite(mesh.positions[i]) || !validStableId(mesh.vertexIds[i])
            || !vertexIds.insert(mesh.vertexIds[i]).second) {
          error="Imported mesh has invalid positions or vertex identities"; return false;
        }
      for (const auto index : mesh.indices) if (index>=mesh.positions.size()) {
        error="Imported mesh index is out of range"; return false;
      }
      for (const auto& attribute : attributes) {
        const auto& descriptor=attribute.descriptor;
        const auto count=descriptor.domain==Domain::vertex ? mesh.positions.size()
            : descriptor.domain==Domain::face ? mesh.indices.size()/3 : 0;
        if (count==0 || attribute.elements.size()!=count || !validStableId(descriptor.stableId)
            || !attributeIds.insert(descriptor.stableId).second || descriptor.name.empty()
            || descriptor.name.size()>kMaximumNameBytes || !attributeNames.insert(descriptor.name).second
            || descriptor.interpolation==Interpolation::unspecified
            || descriptor.interpolation>Interpolation::normalizedLinear
            || !std::all_of(attribute.elements.begin(),attribute.elements.end(),[&](const auto& element) {
                return validFieldElement(element,descriptor.valueType);
              }) || !addArrayBytes(attribute.elements.size(),sizeof(FieldElement),fieldBytes)
            || fieldBytes>kMaximumBufferBytes) {
          error="Imported mesh attribute is invalid or exceeds its budget"; return false;
        }
      }
      dependencies[operation.outputStableId].clear();
      local.geometries.emplace(operation.outputStableId,mesh);
      retainedAttributes[operation.outputStableId]=attributes;
      hasStoredAttributes=true;
      break;
    }
    case OperationCode::copy: {
      const auto deferred = context.deferredPoints.find(operation.outputStableId);
      if (operation.inputStableId != 0 || !zeroParameters(operation) ||
          !generate(deferred == context.deferredPoints.end() ? value.carrier : CarrierKind::points3D)) {
        error = "Copy operation identity or parameters are invalid";
        return false;
      }
      dependencies[operation.outputStableId].clear();
      if (deferred != context.deferredPoints.end()) {
        deferredOutputs.insert(operation.outputStableId);
        const auto& binding = deferred->second;
        if (!validStableId(binding.sourceStableId) || binding.sourceRevision == 0
            || binding.sourceStableId != value.sourceStableId || binding.sourceRevision != value.sourceRevision
            || binding.capacity == 0 || binding.capacity > 64) {
          error = "Deferred Points3D source identity or capacity is invalid"; return false;
        }
        const auto sample = context.evaluatedPoints.find(operation.outputStableId);
        if (sample == context.evaluatedPoints.end()) {
          local.points.emplace(operation.outputStableId, PointsData{});
        } else {
          const auto& source = sample->second;
          const auto* points = std::get_if<PointsData>(&source.data);
          if (source.schemaVersion != kSchemaVersion || source.carrier != CarrierKind::points3D
              || source.stableId != operation.outputStableId || source.sourceStableId != binding.sourceStableId
              || source.sourceRevision != binding.sourceRevision || !points || points->points.size() > binding.capacity
              || source.overflow != OverflowPolicy::reject || source.dispatchCount != 0
              || source.fieldDomain != Domain::unspecified || source.fieldValueType != ValueType::unspecified
              || source.fieldInterpolation != Interpolation::unspecified || source.fieldDomainCardinality != 0
              || !source.operations.empty() || !source.spectrumFields.empty() || !source.scoreFields.empty()) {
            error = "Evaluated Points3D does not match its compiled source"; return false;
          }
          PortContract pointContract;
          pointContract.carrier = CarrierKind::points3D; pointContract.maxPoints = binding.capacity;
          pointContract.maxAttributes = kMaximumAttributes;
          for (const auto& attribute : source.attributes) pointContract.attributes.push_back(attribute.descriptor);
          if (!validatePortContract(pointContract, error)) return false;
          std::set<StableId> pointIds;
          for (const auto& point : points->points)
            if (!finite(point.position) || !finite(point.rotation) || !finite(point.radius) || point.radius < 0
                || !validStableId(point.stableId) || !pointIds.insert(point.stableId).second) {
              error = "Evaluated Points3D contains invalid or duplicate points"; return false;
            }
          for (const auto& attribute : source.attributes) {
            if (attribute.elements.size() > (kMaximumBufferBytes-fieldBytes)/sizeof(FieldElement)) {
              error="Evaluated Points3D attributes exceed the retained memory budget"; return false;
            }
            fieldBytes += attribute.elements.size()*sizeof(FieldElement);
            if (attribute.elements.size() != points->points.size()
                || !std::all_of(attribute.elements.begin(),attribute.elements.end(),[&](const auto& element) {
                  return validFieldElement(element,attribute.descriptor.valueType);
                })) { error = "Evaluated Points3D attributes are invalid"; return false; }
          }
          local.points.emplace(operation.outputStableId, *points);
          retainedAttributes[operation.outputStableId] = source.attributes;
        }
      }
      break;
    }
    case OperationCode::grid: {
      GeometryData grid;
      if ((operation.inputStableId != value.sourceStableId &&
           operation.inputStableId != operation.outputStableId) ||
          !generate(CarrierKind::geometry3D) ||
          !materializeGridOperation(operation, grid, error))
        return false;
      dependencies[operation.outputStableId].clear();
      local.geometries.emplace(operation.outputStableId, std::move(grid));
      break;
    }
    case OperationCode::plane:
    case OperationCode::cube:
    case OperationCode::sphere:
    case OperationCode::icosphere:
    case OperationCode::cylinder: {
      GeometryData mesh;
      if ((operation.inputStableId != value.sourceStableId
           && operation.inputStableId != operation.outputStableId)
          || !generate(CarrierKind::geometry3D)) {
        error = "Mesh generator identity is invalid";
        return false;
      }
      if (!materializeMeshGenerator(operation, mesh, kMaximumVertices,
                                    kMaximumIndices, error)) return false;
      dependencies[operation.outputStableId].clear();
      local.geometries.emplace(operation.outputStableId, std::move(mesh));
      break;
    }
    case OperationCode::curveCircle:
    case OperationCode::curveLine: {
      if ((operation.inputStableId != value.sourceStableId &&
           operation.inputStableId != operation.outputStableId) ||
          operation.parameters[0] <= 0.0f ||
          operation.parameters[2] != 0.0f ||
          operation.parameters[3] != 0.0f || !generate(CarrierKind::curves3D)) {
        error = "Curve generator operation semantics are invalid";
        return false;
      }
      // Older circle records omit point count. Their saved graphs acquire it on recompilation.
      if (operation.code != OperationCode::curveCircle || operation.parameters[1] != 0) {
        CurvesData curves;
        if (!materializeCurveGenerator(operation, curves, kMaximumCurvePoints, error)) return false;
        local.curves.emplace(operation.outputStableId, std::move(curves));
      }
      dependencies[operation.outputStableId].clear();
      break;
    }
    case OperationCode::curveToMesh: {
      if (!requireInput(CarrierKind::curves3D) || operation.parameters[0] < 0 ||
          operation.parameters[1] != 0 || operation.parameters[2] != 0 ||
          operation.parameters[3] != 0 ||
          !generate(CarrierKind::geometry3D)) {
        error = "Curve to Mesh operation transition is invalid";
        return false;
      }
      const auto curves = local.curves.find(operation.inputStableId);
      if (curves != local.curves.end()) {
        GeometryData mesh;
        const auto width = operation.parameters[0] == 0 ? 0.1f : operation.parameters[0];
        if (!materializeCurveRibbon(curves->second, width, mesh, kMaximumVertices,
                                    kMaximumIndices, error)) return false;
        local.geometries.emplace(operation.outputStableId, std::move(mesh));
      } else if (operation.parameters[0] != 0) {
        error = "Curve to Mesh has no retained curve generator";
        return false;
      }
      break;
    }
    case OperationCode::curveResample:
    case OperationCode::curveTrim:
    case OperationCode::curveSetRadius:
    case OperationCode::curveSetTilt:
    case OperationCode::curveWave:
    case OperationCode::curveAnimate: {
      const auto source=local.curves.find(operation.inputStableId);
      const FieldData *field=nullptr;
      if (operation.secondaryInputStableId!=0) {
        const auto found=fields.find(operation.secondaryInputStableId);
        const auto type=fieldTypes.find(operation.secondaryInputStableId);
        if (found==fields.end() || type==fieldTypes.end() ||
            type->second!=std::make_pair(Domain::curvePoint,ValueType::floatValue)) {
          error="Curve attribute requires a retained Float field on curve points"; return false;
        }
        field=&found->second;
      }
      CurvesData curves;
      if (!requireInput(CarrierKind::curves3D) || source==local.curves.end() ||
          !generate(CarrierKind::curves3D) ||
          !materializeCurveEdit(source->second,operation,field,curves,kMaximumCurvePoints,error,
                                runtime ? runtime->timelineSeconds : 0.0)) {
        if (error.empty()) error="Curve edit requires retained Curves3D";
        return false;
      }
      if (field) dependencies[operation.outputStableId].insert(operation.secondaryInputStableId);
      local.curves.emplace(operation.outputStableId,std::move(curves));
      break;
    }
    case OperationCode::curveToTube: {
      const auto source=local.curves.find(operation.inputStableId);
      std::uint32_t sides=0;
      GeometryData mesh;
      if (!requireInput(CarrierKind::curves3D) || source==local.curves.end() ||
          !exactIntegerParameter(operation.parameters[1],3,sides) ||
          operation.parameters[2]!=0 || operation.parameters[3]!=0 ||
          !generate(CarrierKind::geometry3D) ||
          !materializeCurveTube(source->second,operation.parameters[0],sides,mesh,
                               kMaximumVertices,kMaximumIndices,error)) {
        if (error.empty()) error="Curve tube requires retained curves and a bounded profile";
        return false;
      }
      local.geometries.emplace(operation.outputStableId,std::move(mesh));
      break;
    }
    case OperationCode::pointsFromCurves: {
      const auto source=local.curves.find(operation.inputStableId);
      PointsData points;
      if (!requireInput(CarrierKind::curves3D) || source==local.curves.end() ||
          (operation.parameters[0]!=0 && operation.parameters[0]!=1) ||
          operation.parameters[2]!=0 || operation.parameters[3]!=0 ||
          !generate(CarrierKind::points3D) ||
          !materializeCurvePoints(source->second,operation.parameters[0]!=0,operation.parameters[1],
                                 points,kMaximumPoints,error)) {
        if (error.empty()) error="Curve to Points requires retained curves and positive scale";
        return false;
      }
      local.points.emplace(operation.outputStableId,std::move(points));
      break;
    }
    case OperationCode::pointsFromVertices: {
      if (!requireInput(CarrierKind::geometry3D) || !zeroParameters(operation) ||
          !generate(CarrierKind::points3D)) {
        error = "Points from Vertices operation transition is invalid";
        return false;
      }
      const auto geometry = local.geometries.find(operation.inputStableId);
      if (geometry != local.geometries.end()) {
        PointsData points;
        points.points.reserve(geometry->second.positions.size());
        for (std::size_t i = 0; i < geometry->second.positions.size(); ++i)
          points.points.push_back({geometry->second.positions[i],
                                   geometry->second.vertexIds[i], 1.0f, {}});
        local.points.emplace(operation.outputStableId, std::move(points));
      }
      break;
    }
    case OperationCode::pointsOnFaces:
    case OperationCode::pointsOnFacesDensity:
    case OperationCode::pointsInBoxVolume: {
      // A density count follows each frame's deformed area. The node's maximum is its only
      // bound, and the frame's point and instance storage grows to fit it.
      const auto source=local.geometries.find(operation.inputStableId);
      PointsData points;
      if (!requireInput(CarrierKind::geometry3D) || source==local.geometries.end()
          || !generate(CarrierKind::points3D)
          || !materializeDistributedPoints(source->second,operation,points,error,
                 (runtime || frameDeformer) && operation.code==OperationCode::pointsOnFacesDensity)) {
        if (error.empty()) error="Point distribution requires retained Geometry3D";
        return false;
      }
      local.points.emplace(operation.outputStableId,std::move(points));
      break;
    }
    case OperationCode::transformInstances:
    case OperationCode::cullInstances: {
      const auto source=instanceValues.find(operation.inputStableId);
      InstancesData instances;
      if (!requireInput(CarrierKind::instances3D) || source==instanceValues.end()
          || !generate(CarrierKind::instances3D)
          || !materializeInstanceEdit(source->second,operation,instances,error)) {
        if (error.empty()) error="Instance editing requires retained Instances3D";
        return false;
      }
      if (operation.code==OperationCode::cullInstances)
        for (auto& attribute : retainedAttributes[operation.outputStableId]) {
          const auto original=std::move(attribute.elements);
          attribute.elements.clear(); std::size_t retained=0;
          for (std::size_t i=0;i<source->second.instances.size() && retained<instances.instances.size();++i)
            if (source->second.instances[i].stableId==instances.instances[retained].stableId) {
              if (i>=original.size()) { error="Culled attribute cardinality is invalid"; return false; }
              attribute.elements.push_back(original[i]); ++retained;
            }
        }
      instanceValues.emplace(operation.outputStableId,std::move(instances));
      break;
    }
    case OperationCode::cullGeometry: {
      const auto source = local.geometries.find(operation.inputStableId);
      if (!requireInput(CarrierKind::geometry3D) || source == local.geometries.end()
          || operation.secondaryInputStableId != 0 || !generate(CarrierKind::geometry3D)) {
        error = "Cull Geometry requires retained Geometry3D";
        return false;
      }
      GeometryData mesh;
      GeometryCullSelection selection;
      const auto vertexCount = source->second.positions.size();
      const auto faceCount = source->second.indices.size() / 3;
      if (!materializeGeometryCull(source->second, operation, mesh, selection, error)
          || !compactCulledAttributes(retainedAttributes[operation.outputStableId],
              {{Domain::vertex, &selection.vertices, vertexCount},
               {Domain::face, &selection.faces, faceCount}}, error)) return false;
      local.geometries.emplace(operation.outputStableId, std::move(mesh));
      break;
    }
    case OperationCode::shadeMesh: {
      const auto source = local.geometries.find(operation.inputStableId);
      if (!requireInput(CarrierKind::geometry3D) || source == local.geometries.end()
          || operation.secondaryInputStableId != 0 || !generate(CarrierKind::geometry3D)) {
        error = "Shade Mesh requires retained Geometry3D";
        return false;
      }
      GeometryData mesh;
      if (!materializeShadedMesh(source->second, operation, mesh,
              retainedAttributes[operation.outputStableId], error)) return false;
      local.geometries.emplace(operation.outputStableId, std::move(mesh));
      break;
    }
    case OperationCode::cullPoints: {
      const auto source = local.points.find(operation.inputStableId);
      if (!requireInput(CarrierKind::points3D) || source == local.points.end()
          || operation.secondaryInputStableId != 0 || !generate(CarrierKind::points3D)) {
        error = "Cull Points requires retained Points3D";
        return false;
      }
      PointsData points;
      std::vector<std::size_t> kept;
      const auto pointCount = source->second.points.size();
      if (!materializePointsCull(source->second, operation, points, kept, error)
          || !compactCulledAttributes(retainedAttributes[operation.outputStableId],
              {{Domain::point, &kept, pointCount}}, error)) return false;
      local.points.emplace(operation.outputStableId, std::move(points));
      break;
    }
    case OperationCode::instanceOnPoints: {
      if (!requireInput(CarrierKind::points3D) || !zeroParameters(operation) ||
          !validStableId(operation.secondaryInputStableId)
          || !generate(CarrierKind::instances3D)) {
        error = "Instance on Points operation transition is invalid";
        return false;
      }
      const auto sourceCarrier = produced.find(operation.secondaryInputStableId);
      if ((sourceCarrier == produced.end()
           || sourceCarrier->second != CarrierKind::geometry3D)
          && context.admittedGeometryOrSceneSources.count(
                 operation.secondaryInputStableId) == 0) {
        error = "Instance on Points has no explicit Geometry3D provenance";
        return false;
      }
      const auto points = local.points.find(operation.inputStableId);
      if (points == local.points.end()) {
        error = "Instance on Points has no retained Points3D provenance";
        return false;
      }
      if (scorePrototype) {
        const auto& binding=value.scoreFields.front();
        std::set<StableId> pointAncestors;
        std::vector<StableId> pendingPoints{operation.inputStableId};
        while (!pendingPoints.empty()) {
          const auto id=pendingPoints.back(); pendingPoints.pop_back();
          if (!pointAncestors.insert(id).second) continue;
          const auto prior=dependencies.find(id);
          if (prior!=dependencies.end()) pendingPoints.insert(pendingPoints.end(),prior->second.begin(),prior->second.end());
        }
        if (pointAncestors.count(binding.consumerGeometryStableId)!=0) {
          error="Score-deformed instancing requires independent static placements"; return false;
        }
        auto prototype=operation.secondaryInputStableId;
        const auto priorEnd=value.operations.begin()+(&operation-value.operations.data());
        while (prototype!=binding.consumerGeometryStableId) {
          const auto prior=std::find_if(value.operations.begin(),priorEnd,[&](const auto& op) {
            return op.outputStableId==prototype && op.code==OperationCode::assignMaterial;
          });
          if (prior==priorEnd) {
            error="Score-deformed instancing lost its exact Set Position prototype"; return false;
          }
          prototype=prior->inputStableId;
        }
        const auto mesh=local.geometries.find(operation.secondaryInputStableId);
        if (mesh==local.geometries.end() || (replayLimits
            && (mesh->second.positions.size()>replayLimits->maxVertices
                || mesh->second.indices.size()>replayLimits->maxIndices
                || points->second.points.size()>replayLimits->maxPoints
                || points->second.points.size()>replayLimits->maxInstances))) {
          error="Score-deformed instancing exceeds its retained prototype or placement limits"; return false;
        }
      }
      if (!context.deferredPoints.empty()) for (auto attribute : retainedAttributes[operation.inputStableId]) {
        attribute.descriptor.domain=Domain::instance;
        retainedAttributes[operation.outputStableId].push_back(std::move(attribute));
      }
      InstancesData generated;
      generated.instances.reserve(points->second.points.size());
      for (std::size_t i = 0; i < points->second.points.size(); ++i) {
        const auto &point = points->second.points[i];
        generated.instances.push_back(
            {point.stableId, operation.secondaryInputStableId,
             {point.position, point.rotation,
              {point.radius, point.radius, point.radius}}});
      }
      dependencies[operation.outputStableId].insert(
          operation.secondaryInputStableId);
      materials[operation.outputStableId] = materials.count(operation.secondaryInputStableId)
          ? materials.at(operation.secondaryInputStableId) : 1;
      local.materialTables[operation.outputStableId]=local.materialTables[operation.secondaryInputStableId];
      instanceValues.emplace(operation.outputStableId, std::move(generated));
      sawInstanceOperation = true;
      break;
    }
    case OperationCode::spectrumInstancer: {
      const auto binding = std::find_if(value.spectrumFields.begin(), value.spectrumFields.end(),
          [&](const auto& candidate) { return candidate.fieldStableId == operation.stableId; });
      InstancesData generated;
      const auto placement=operation.secondaryInputStableId==0 ? std::optional<CarrierData>{}
          : retainedData(operation.secondaryInputStableId);
      if (!requireInput(CarrierKind::geometry3D) || !generate(CarrierKind::instances3D)
          || local.geometries.count(operation.inputStableId) == 0
          || binding == value.spectrumFields.end()
          || !materializeSpectrumInstances(operation, *binding, generated, error,placement ? &*placement : nullptr)) {
        if (error.empty()) error = "Spectrum Instancer requires retained source Geometry3D";
        return false;
      }
      if (placement) dependencies[operation.outputStableId].insert(operation.secondaryInputStableId);
      if (runtime && scorePlacementAncestors.count(operation.outputStableId)!=0) {
        // Score placement samples the current frame only, so it cannot honor rows.
        if (binding->history.enabled) {
          error="Spectral History Spectrum Instancer cannot place an Instance Score Field"; return false;
        }
        spectrum::Bands followed{};
        if (!runtime->spectrumAt || !runtime->spectrumAt(*binding,followed,error)) {
          if (error.empty()) error="Instance Score Field requires its exact frame spectrum placements";
          return false;
        }
        for (std::size_t i=0;i<generated.instances.size();++i) {
          const auto amount=spectrum::sample(*binding,followed,binding->coordinates[i]);
          auto& transform=generated.instances[i].transform;
          transform.translation.x+=amount*binding->direction[0];
          transform.translation.y+=amount*binding->direction[1];
          transform.translation.z+=amount*binding->direction[2];
          const auto scale=[&](std::size_t axis,float base) {
            return std::clamp(base*(1.0f+amount*binding->scaleResponse[axis]),0.000001f,1000.0f);
          };
          transform.scale={scale(0,transform.scale.x),scale(1,transform.scale.y),scale(2,transform.scale.z)};
          if (!finite(transform)) { error="Instance Score Field spectrum placement is non-finite"; return false; }
        }
      }
      instanceValues.emplace(operation.outputStableId, std::move(generated));
      sampledInstanceOwners.emplace(operation.outputStableId,operation.outputStableId);
      sawInstanceOperation = true;
      break;
    }
    case OperationCode::audioDeformer: {
      const auto binding = std::find_if(value.spectrumFields.begin(), value.spectrumFields.end(),
          [&](const auto& candidate) { return candidate.fieldStableId == operation.stableId; });
      if (!requireInput(CarrierKind::geometry3D) || !generate(CarrierKind::geometry3D)
          || !zeroParameters(operation) || local.geometries.count(operation.inputStableId) == 0
          || binding == value.spectrumFields.end()
          || binding->target != spectrum::Target::geometryDeformation) {
        error = "Audio Deformer requires retained Geometry3D and its spectrum binding";
        return false;
      }
      local.geometries.emplace(operation.outputStableId, local.geometries.at(operation.inputStableId));
      if (frameDeformer) {
        if (!frameDeformer(local.geometries.at(operation.outputStableId), *binding, error)) return false;
        const auto& original=local.geometries.at(operation.inputStableId);
        const auto& evaluated=local.geometries.at(operation.outputStableId);
        if (original.indices!=evaluated.indices || original.vertexIds!=evaluated.vertexIds
            || original.positions.size()!=evaluated.positions.size()
            || !std::all_of(evaluated.positions.begin(),evaluated.positions.end(),[](const auto& p) { return finite(p); })) {
          error="Audio geometry replay must preserve finite positions and the exact source topology"; return false;
        }
        auto& attributes=retainedAttributes[operation.outputStableId];
        attributes.erase(std::remove_if(attributes.begin(),attributes.end(),[](const auto& attribute) {
          return attribute.descriptor.name=="normal";
        }),attributes.end());
      }
      break;
    }
    case OperationCode::positionField:
    case OperationCode::normalField:
    case OperationCode::indexField:
    case OperationCode::stableIdentityField:
    case OperationCode::boundsMinimumField:
    case OperationCode::boundsMaximumField:
    case OperationCode::boundsSizeField:
    case OperationCode::curveTangentField:
    case OperationCode::curveParameterField:
    case OperationCode::curveRadiusField:
    case OperationCode::curveTiltField:
    case OperationCode::curveEndpointField:
    case OperationCode::boundsCenterField: {
      const auto source = produced.find(operation.inputStableId);
      const auto data = retainedData(operation.inputStableId);
      FieldData field;
      if (source == produced.end() || !data || !zeroParameters(operation) ||
          !generate(CarrierKind::field) ||
          !materializeElementField(*data, source->second, operation.code, field,
                                   (kMaximumBufferBytes - fieldBytes) / sizeof(FieldElement), error)) {
        if (error.empty()) error = "Element field requires a retained source operation";
        return false;
      }
      fieldBytes += field.elements.size() * sizeof(FieldElement);
      if (operation.code==OperationCode::normalField)
        if (const auto* normal=findNamedAttribute(retainedAttributes[operation.inputStableId],"normal"))
          field.elements=normal->elements;
      fields.emplace(operation.outputStableId, std::move(field));
      fieldTypes.emplace(operation.outputStableId,
                         std::make_pair(elementDomain(source->second), elementFieldType(operation.code)));
      break;
    }
    case OperationCode::constantVectorField: {
      const auto count = operation.parameters[3];
      if (operation.inputStableId != operation.outputStableId ||
          count < 1 || count > kMaximumFieldElements || std::floor(count) != count ||
          count > (kMaximumBufferBytes - fieldBytes) / sizeof(FieldElement) ||
          !generate(CarrierKind::field)) {
        error = "Constant vector field has invalid identity or capacity";
        return false;
      }
      FieldData field;
      field.elements.resize(static_cast<std::size_t>(count),
          {{operation.parameters[0], operation.parameters[1], operation.parameters[2], 0}});
      fieldBytes += field.elements.size() * sizeof(FieldElement);
      fields.emplace(operation.outputStableId, std::move(field));
      fieldTypes.emplace(operation.outputStableId, std::make_pair(Domain::vertex, ValueType::vector));
      dependencies[operation.outputStableId].clear();
      break;
    }
    case OperationCode::vectorDisplacement: {
      const auto spectral=std::find_if(value.operations.begin(),value.operations.end(),
          owner_detail::CoreDisplacementSpectrumPredicate{operation});
      if (spectral!=value.operations.end()) {
        const auto binding=std::find_if(value.spectrumFields.begin(),value.spectrumFields.end(),[&](const auto& b) {
          return b.fieldStableId==spectral->stableId;
        });
        if (binding!=value.spectrumFields.end() && deferredSpectrumSource(value,*spectral,*binding,context)
            && operation.inputStableId!=spectral->inputStableId) {
          error="Deferred spectral displacement requires the exact sampled Points3D source"; return false;
        }
      }
      const auto source = produced.find(operation.inputStableId);
      auto data = retainedData(operation.inputStableId);
      const auto field = fields.find(operation.secondaryInputStableId);
      const auto type = fieldTypes.find(operation.secondaryInputStableId);
      if (source == produced.end() || !data || field == fields.end() || type == fieldTypes.end() ||
          type->second != std::make_pair(elementDomain(source->second), ValueType::vector) ||
          operation.parameters[1] != 0 || operation.parameters[2] != 0 || operation.parameters[3] != 0 ||
          !generate(source->second) ||
          !displaceElements(*data, source->second, field->second, operation.parameters[0], error)) {
        if (error.empty()) error = "Vector displacement requires a retained source and matching Vector field";
        return false;
      }
      if (source->second == CarrierKind::geometry3D)
        local.geometries.emplace(operation.outputStableId, std::get<GeometryData>(std::move(*data)));
      else if (source->second == CarrierKind::points3D)
        local.points.emplace(operation.outputStableId, std::get<PointsData>(std::move(*data)));
      else if (source->second == CarrierKind::curves3D)
        local.curves.emplace(operation.outputStableId, std::get<CurvesData>(std::move(*data)));
      else if (source->second == CarrierKind::instances3D)
        instanceValues.emplace(operation.outputStableId, std::get<InstancesData>(std::move(*data)));
      dependencies[operation.outputStableId].insert(operation.secondaryInputStableId);
      break;
    }
    case OperationCode::transformTRS: {
      const auto source = local.geometries.find(operation.inputStableId);
      if (!requireInput(CarrierKind::geometry3D) || source == local.geometries.end() ||
          !zeroParameters(operation) || !generate(CarrierKind::geometry3D)) {
        error = "Transform Geometry requires retained Geometry3D";
        return false;
      }
      auto mesh = source->second;
      if (scorePrototype && operation.outputStableId==value.scoreFields.front().consumerGeometryStableId) {
        const auto& binding=value.scoreFields.front();
        const auto field=fields.find(binding.fieldStableId);
        if (operation.inputStableId!=binding.sourceGeometryStableId || !equal(operation.transform,Transform{})
            || field==fields.end()) {
          error="Score-deformed prototype lost its identity Set Position consumer"; return false;
        }
        CarrierData data=std::move(mesh);
        if (!displaceElements(data,CarrierKind::geometry3D,field->second,1,error)) return false;
        mesh=std::get<GeometryData>(std::move(data));
      }
      if (!materializeTransform(mesh,operation.transform,error)
          || !transformAuthoredNormals(retainedAttributes[operation.outputStableId],operation.transform,error)) return false;
      local.geometries.emplace(operation.outputStableId,std::move(mesh));
      break;
    }
    case OperationCode::joinGeometry: {
      const auto a = local.geometries.find(operation.inputStableId);
      const auto b = local.geometries.find(operation.secondaryInputStableId);
      if (!requireInput(CarrierKind::geometry3D) || a == local.geometries.end() ||
          b == local.geometries.end() || !zeroParameters(operation) ||
          materials.at(operation.inputStableId) != materials.at(operation.secondaryInputStableId) ||
          !generate(CarrierKind::geometry3D)) {
        error = "Join Geometry requires retained meshes with the same material";
        return false;
      }
      GeometryData mesh;
      if (!materializeJoin(a->second,b->second,operation.inputStableId,operation.secondaryInputStableId,
                            mesh,kMaximumVertices,kMaximumIndices,error)) return false;
      dependencies[operation.outputStableId].insert(operation.secondaryInputStableId);
      auto& joinedTable=local.materialTables[operation.outputStableId];
      for (const auto& slot : local.materialTables[operation.secondaryInputStableId]) {
        const auto existing=joinedTable.find(slot.first);
        if (existing!=joinedTable.end() && (existing->second.parameters!=slot.second.parameters
            || existing->second.field.values!=slot.second.field.values)) {
          error="Join Geometry has conflicting material table slots"; return false;
        }
        joinedTable[slot.first]=slot.second;
      }
      auto& joinedAttributes=retainedAttributes[operation.outputStableId];
      const auto& otherAttributes=retainedAttributes[operation.secondaryInputStableId];
      if (joinedAttributes.size()!=otherAttributes.size()) { error="Join requires matching named attributes"; return false; }
      for (auto& attribute : joinedAttributes) {
        const auto* other=findNamedAttribute(otherAttributes,attribute.descriptor.name);
        if (!other || !(other->descriptor==attribute.descriptor)
            || other->elements.size()>kMaximumFieldElements-attribute.elements.size()
            || other->elements.size()>(kMaximumBufferBytes-fieldBytes)/sizeof(FieldElement)) {
          error="Join requires matching named attribute types and domains"; return false;
        }
        fieldBytes+=other->elements.size()*sizeof(FieldElement);
        attribute.elements.insert(attribute.elements.end(),other->elements.begin(),other->elements.end());
      }
      local.geometries.emplace(operation.outputStableId,std::move(mesh));
      break;
    }
    case OperationCode::boundingBox: {
      const auto source = local.geometries.find(operation.inputStableId);
      if (!requireInput(CarrierKind::geometry3D) || source == local.geometries.end() ||
          !zeroParameters(operation) || !generate(CarrierKind::geometry3D)) {
        error = "Bounding Box requires retained Geometry3D";
        return false;
      }
      GeometryData mesh;
      if (!materializeBoundingBox(source->second,mesh,error)) return false;
      local.geometries.emplace(operation.outputStableId,std::move(mesh));
      break;
    }
    case OperationCode::subdivisionSurface: {
      // parameters: viewport level, render level, 0 uses the viewport level and 1 the render level,
      // and the dicing rate in pixels (0 is off). With a rate and a frame view, the level rises
      // until each cage edge's pieces span at most that many pixels.
      const auto source = local.geometries.find(operation.inputStableId);
      std::uint32_t viewportLevel = 0, renderLevel = 0;
      const auto &p = operation.parameters;
      if (!requireInput(CarrierKind::geometry3D) || source == local.geometries.end()
          || !exactIntegerParameter(p[0],0,viewportLevel) || !exactIntegerParameter(p[1],0,renderLevel)
          || (p[2] != 0.0f && p[2] != 1.0f) || !(p[3] >= 0.0f && p[3] <= kMaximumSubdivisionDicingRate)
          || !operation.subdivision
          || !validSubdivisionCreases(operation.subdivision->creases) || !generate(CarrierKind::geometry3D)) {
        error = "Subdivision Surface requires retained Geometry3D, whole levels and sorted crease edges";
        return false;
      }
      GeometryData mesh;
      std::vector<AttributeData> attributes;
      auto level = p[2] != 0.0f ? renderLevel : viewportLevel;
      if (p[3] > 0.0f && runtime && runtime->subdivisionViewportHeight > 0)
        level = adaptiveCageLevel(source->second,p[3],runtime->subdivisionLocalToCamera,runtime->subdivisionTanHalfFov,
                                  runtime->subdivisionViewportHeight,level);
      if (!materializeSubdivisionSurface(source->second,retainedAttributes[operation.inputStableId],
              operation.subdivision->creases,level,mesh,attributes,error)) return false;
      retainedAttributes[operation.outputStableId] = std::move(attributes);
      local.geometries.emplace(operation.outputStableId,std::move(mesh));
      break;
    }
    case OperationCode::assignMaterial: {
      const auto source = local.geometries.find(operation.inputStableId);
      if (!requireInput(CarrierKind::geometry3D) || source == local.geometries.end() ||
          !validStableId(operation.secondaryInputStableId) || !zeroParameters(operation) ||
          !generate(CarrierKind::geometry3D)) {
        error = "Set Material requires retained Geometry3D and a stable material identity";
        return false;
      }
      local.geometries.emplace(operation.outputStableId,source->second);
      materials[operation.outputStableId] = operation.secondaryInputStableId;
      break;
    }
    case OperationCode::transform:
      if (!requireInput(CarrierKind::geometry3D) || operation.parameters[3] != 0.0f ||
          !generate(CarrierKind::geometry3D)) {
        error = "Transform operation transition is invalid";
        return false;
      }
      break;
    case OperationCode::evaluateField:
      if (operation.inputStableId!=operation.outputStableId) {
        const auto source=instanceValues.find(operation.inputStableId);
        const auto binding=std::find_if(value.scoreFields.begin(),value.scoreFields.end(),[&](const auto& b) {
          return b.fieldStableId==operation.stableId;
        });
        if (!requireInput(CarrierKind::instances3D) || source==instanceValues.end()
            || binding==value.scoreFields.end() || binding->sourceGeometryStableId!=operation.inputStableId
            || binding->consumerGeometryStableId!=0 || !zeroParameters(operation)
            || !scorefield::validate(*binding,source->second.instances.size(),error)
            || (replayLimits && source->second.instances.size()>replayLimits->maxInstances)) {
          if (error.empty()) error="Instance Score Field lost its exact retained source";
          return false;
        }
        std::set<StableId> ancestors;
        std::vector<StableId> pending{operation.inputStableId};
        while (!pending.empty()) {
          const auto id=pending.back(); pending.pop_back();
          if (!ancestors.insert(id).second) continue;
          const auto prior=dependencies.find(id);
          if (prior!=dependencies.end()) pending.insert(pending.end(),prior->second.begin(),prior->second.end());
        }
        const bool dynamicPlacements=std::any_of(value.operations.begin(),value.operations.end(),[&](const auto& op) {
          return ancestors.count(op.outputStableId)!=0 && (op.code==OperationCode::spectrumInstancer
              || (op.retainedMesh && !op.retainedMesh->importedAnimation.empty()));
        });
        if (std::any_of(value.operations.begin(),value.operations.end(),[&](const auto& op) {
              return ancestors.count(op.outputStableId)!=0
                  && (op.code==OperationCode::fieldTimeline || op.code==OperationCode::fieldScoreSample
                      || op.code==OperationCode::fieldWaterHeight || op.code==OperationCode::fieldAudioSpectrum || op.code==OperationCode::audioDeformer
                      || (op.code==OperationCode::evaluateField && op.inputStableId!=op.outputStableId));
            }) || std::any_of(context.deferredPoints.begin(),context.deferredPoints.end(),[&](const auto& item) {
              return ancestors.count(item.first)!=0;
            })) {
          error="Instance Score Field requires fixed ordered retained placements"; return false;
        }
        FieldData positions,field;
        const auto count=source->second.instances.size();
        const auto maximumBytes=replayLimits ? replayLimits->maxBufferBytes : kMaximumBufferBytes;
        if (fieldBytes>maximumBytes || count>(maximumBytes-fieldBytes)/(2*sizeof(FieldElement))) {
          error="Instance Score Field exceeds its retained field memory budget"; return false;
        }
        positions.elements.resize(count); field.elements.resize(count);
        for (std::size_t i=0;i<count;++i) {
          const auto& instance=source->second.instances[i]; const auto& p=instance.transform.translation;
          if (binding->elementIds[i]!=instance.stableId
              || ((!runtime || !dynamicPlacements) && binding->positions[i]!=std::array<float,3>{p.x,p.y,p.z})
              || !validStableId(instance.sourceStableId) || local.geometries.count(instance.sourceStableId)==0) {
            error="Instance Score Field samples disagree with source positions, ordered identities or prototypes"; return false;
          }
          const auto& prototype=local.geometries.at(instance.sourceStableId);
          if (replayLimits && (prototype.positions.size()>replayLimits->maxVertices
              || prototype.indices.size()>replayLimits->maxIndices)) {
            error="Instance Score Field prototype exceeds its admitted geometry budget"; return false;
          }
          positions.elements[i].components={p.x,p.y,p.z,0};
        }
        if (runtime && (!runtime->scoreAt || !runtime->scoreAt(operation,positions,field,error))) {
          if (error.empty()) error="Instance Score Field requires the canonical score frame";
          return false;
        }
        if (field.elements.size()!=count || !std::all_of(field.elements.begin(),field.elements.end(),[](const auto& e) {
              return validFieldElement(e,ValueType::vector);
            }) || !generate(CarrierKind::field)) {
          error="Instance Score Field returned a different typed cardinality"; return false;
        }
        fieldBytes+=2*count*sizeof(FieldElement);
        fields.emplace(operation.outputStableId,std::move(field));
        fieldTypes.emplace(operation.outputStableId,std::make_pair(Domain::instance,ValueType::vector));
        sampledInstanceOwners.emplace(operation.inputStableId,operation.inputStableId);
        sampledInstanceFieldSources[operation.outputStableId].insert(operation.inputStableId);
        break;
      }
      if (operation.inputStableId != operation.outputStableId ||
          operation.stableId != operation.outputStableId || !zeroParameters(operation) ||
          !generatedOutputs.insert(operation.outputStableId).second) {
        error = "Field evaluation operation semantics are invalid";
        return false;
      }
      if (scorePrototype) {
        const auto& binding=value.scoreFields.front();
        const auto source=local.geometries.find(binding.sourceGeometryStableId);
        if (operation.stableId!=binding.fieldStableId || source==local.geometries.end()
            || !scorefield::validate(binding,source->second.positions.size(),error)
            || binding.elementIds!=source->second.vertexIds
            || (replayLimits && source->second.positions.size()>replayLimits->maxVertices)) {
          if (error.empty()) error="Score-deformed prototype lost its exact retained vertex samples";
          return false;
        }
        const auto count=source->second.positions.size();
        const auto maximumBytes=replayLimits ? replayLimits->maxBufferBytes : kMaximumBufferBytes;
        if (fieldBytes>maximumBytes || count>(maximumBytes-fieldBytes)/(2*sizeof(FieldElement))) {
          error="Score-deformed prototype exceeds its retained field memory budget"; return false;
        }
        FieldData positions,field;
        positions.elements.resize(count); field.elements.resize(count);
        for (std::size_t i=0;i<count;++i) {
          const auto& p=source->second.positions[i];
          if (binding.positions[i]!=std::array<float,3>{p.x,p.y,p.z}) {
            error="Score-deformed prototype samples disagree with its authored source"; return false;
          }
          positions.elements[i].components={p.x,p.y,p.z,0};
        }
        if (runtime && (!runtime->scoreAt || !runtime->scoreAt(operation,positions,field,error))) {
          if (error.empty()) error="Score-deformed prototype requires its canonical score frame";
          return false;
        }
        if (field.elements.size()!=count || !std::all_of(field.elements.begin(),field.elements.end(),[](const auto& e) {
          return validFieldElement(e,ValueType::vector);
        })) {
          error="Score-deformed prototype returned a different typed cardinality"; return false;
        }
        fieldBytes+=2*count*sizeof(FieldElement);
        fields.emplace(operation.outputStableId,std::move(field));
      }
      produced[operation.outputStableId] = CarrierKind::field;
      auxiliaryOutputs.insert(operation.outputStableId);
      dependencies[operation.outputStableId].clear();
      break;
    case OperationCode::prepareInstances:
      if (!requireInput(CarrierKind::geometry3D) || !zeroParameters(operation) ||
          !generate(CarrierKind::instances3D)) {
        error = "Prepare Instances operation transition is invalid";
        return false;
      }
      break;
    case OperationCode::setMaterial: {
      const auto found = produced.find(operation.inputStableId);
      if (found == produced.end() || operation.outputStableId != operation.inputStableId ||
          (found->second != CarrierKind::geometry3D &&
           found->second != CarrierKind::instances3D) ||
          operation.parameters[0] != static_cast<float>(operation.stableId) ||
          operation.parameters[1] != 0.0f || operation.parameters[2] != 0.0f ||
          operation.parameters[3] != 0.0f) {
        error = "Set Material operation semantics are invalid";
        return false;
      }
      materials[operation.outputStableId] = operation.stableId;
      break;
    }
    default:
      error = "geometry operation code is unsupported";
      return false;
    }
    if (runtime && displacedInReplay(value,operation.outputStableId)
        && !displaceReplayMesh(local.geometries,operation.outputStableId,operation.outputStableId,*runtime,error))
      return false;
  }
  // A composed scene has no adapter after the replay. Set Position displacements that no
  // layout change follows go on the terminal mesh, after every later operation, as the
  // image path's adapter adds them.
  if (runtime && runtime->displaceAdapters) {
    std::set<StableId> consumers;
    for (const auto& binding : value.spectrumFields)
      if (adapterSpectrumDisplacement(value,binding) && !displacementInReplay(value,binding.consumerGeometryStableId)
          && consumers.insert(binding.consumerGeometryStableId).second
          && !displaceReplayMesh(local.geometries,value.stableId,binding.consumerGeometryStableId,*runtime,error))
        return false;
  }

  for (const auto& source : context.deferredPoints) {
    if (deferredOutputs.count(source.first)==0) {
      error="Deferred Points3D declaration has no exact operation source"; return false;
    }
  }
  for (const auto& sample : context.evaluatedPoints)
    if (context.deferredPoints.count(sample.first)==0) {
      error="Evaluated Points3D has no compiled source declaration"; return false;
    }
  for (const auto& source : context.deferredBodyGeometry)
    if (bodyOutputs.count(source.first)==0) {
      error="Deferred body Geometry3D declaration has no exact retained operation"; return false;
    }
  for (const auto& sample : context.evaluatedBodyGeometry)
    if (context.deferredBodyGeometry.count(sample.first)==0) {
      error="Evaluated body Geometry3D has no compiled source declaration"; return false;
    }
  // Density points re-sampled on a moving surface change their count each frame. Their
  // ids stay 1..count, so the comparisons below check the shared prefix only.
  const bool movingCount=(runtime || frameDeformer) && densityFollowsBindings(value);
  const auto sameIds=[&](const auto& retained,const auto& admitted,const auto& same) {
    if (!movingCount && retained.size()!=admitted.size()) return false;
    const auto count=std::min(retained.size(),admitted.size());
    return std::equal(retained.begin(),retained.begin()+count,admitted.begin(),same);
  };
  const auto terminal = produced.find(value.stableId);
  if (terminal == produced.end() || terminal->second != value.carrier) {
    error = "geometry operation plan does not produce the exact typed value";
    return false;
  }
  if (!replayed && !runtime && !frameDeformer && hasStoredAttributes && value.carrier!=CarrierKind::field
      && !equalAttributes(value.attributes,retainedAttributes[value.stableId])) {
    error="Retained named attributes disagree with their operation result"; return false;
  }
  std::set<StableId> closure;
  std::vector<StableId> pending{value.stableId};
  while (!pending.empty()) {
    const auto id = pending.back();
    pending.pop_back();
    if (!closure.insert(id).second)
      continue;
    const auto found = dependencies.find(id);
    if (found != dependencies.end())
      pending.insert(pending.end(), found->second.begin(), found->second.end());
  }
  for (const auto &entry : generatedOutputs)
    if (closure.count(entry) == 0 && auxiliaryOutputs.count(entry) == 0) {
      error = "geometry operation plan contains a disconnected operation";
      return false;
    }

  if (value.carrier == CarrierKind::geometry3D) {
    const auto retained = local.geometries.find(value.stableId);
    if (retained != local.geometries.end() &&
        !replayed && !((runtime || frameDeformer) ? retained->second.vertexIds==std::get<GeometryData>(value.data).vertexIds
                     && retained->second.indices==std::get<GeometryData>(value.data).indices
                  : equal(retained->second, std::get<GeometryData>(value.data)))) {
      error = "retained Geometry3D disagrees with its operation result";
      return false;
    }
  } else if (value.carrier == CarrierKind::curves3D) {
    const auto retained = local.curves.find(value.stableId);
    if (retained != local.curves.end() &&
        !replayed && !((runtime || frameDeformer) ? retained->second.points.size()==std::get<CurvesData>(value.data).points.size()
                     && std::equal(retained->second.points.begin(),retained->second.points.end(),
                         std::get<CurvesData>(value.data).points.begin(),[](const auto& a,const auto& b){return a.stableId==b.stableId;})
                  : equal(retained->second, std::get<CurvesData>(value.data)))) {
      error = "retained Curves3D disagrees with its operation result";
      return false;
    }
  } else if (value.carrier == CarrierKind::points3D) {
    const auto retained = local.points.find(value.stableId);
    if (retained != local.points.end() &&
        !replayed && !((runtime || frameDeformer) ? sameIds(retained->second.points,std::get<PointsData>(value.data).points,
                         [](const auto& a,const auto& b){return a.stableId==b.stableId;})
                  : equal(retained->second, std::get<PointsData>(value.data)))) {
      error = "retained Points3D disagrees with its operation result";
      return false;
    }
  } else if (value.carrier == CarrierKind::instances3D) {
    const auto retained = instanceValues.find(value.stableId);
    if (!sawInstanceOperation || retained == instanceValues.end() ||
        (!replayed && !((runtime || frameDeformer) ? sameIds(retained->second.instances,std::get<InstancesData>(value.data).instances,
                         [](const auto& a,const auto& b){
                           return a.stableId==b.stableId && a.sourceStableId==b.sourceStableId;})
                  : equal(retained->second, std::get<InstancesData>(value.data))))) {
      error = "retained Instances3D disagrees with its operation result";
      return false;
    }
    local.terminalStableId = value.stableId;
    local.instances = retained->second;
  } else if (value.carrier == CarrierKind::field) {
    const auto retained = fields.find(value.stableId);
    if (retained != fields.end()) {
      const auto spectral=std::find_if(value.operations.begin(),value.operations.end(),[&](const auto& op) {
        return op.code==OperationCode::fieldAudioSpectrum && op.outputStableId==value.stableId;
      });
      const auto binding=std::find_if(value.spectrumFields.begin(),value.spectrumFields.end(),[&](const auto& b) {
        return b.fieldStableId==value.stableId;
      });
      const bool deferredReplay=replayed && spectral!=value.operations.end() && binding!=value.spectrumFields.end()
          && deferredSpectrumSource(value,*spectral,*binding,context);
      const auto &claimed = std::get<FieldData>(value.data).elements;
      const auto type = fieldTypes.at(value.stableId);
      if (type != std::make_pair(value.fieldDomain, value.fieldValueType) ||
          (!deferredReplay && claimed.size() != retained->second.elements.size()) ||
          (!deferredReplay && !runtime && !frameDeformer && !std::equal(claimed.begin(), claimed.end(), retained->second.elements.begin(),
                      [](const auto &a, const auto &b) { return a.components == b.components; }))) {
        error = "retained Field disagrees with its operation result";
        return false;
      }
    }
  }
  local.materialStableId = materials.count(value.stableId) ? materials.at(value.stableId) : 1;
  if (replayed) {
    const auto data=retainedData(value.stableId);
    if (!data) { error="Animated geometry replay has no retained terminal value"; return false; }
    *replayed=value;
    replayed->data=*data;
    replayed->attributes=retainedAttributes[value.stableId];
    if (value.carrier==CarrierKind::field)
      replayed->fieldDomainCardinality=std::get<FieldData>(*data).elements.size();
  }
  if (materialized != nullptr) {
    local.terminalAttributes=retainedAttributes[value.stableId];
    for (auto& entry : retainedAttributes)
      if (local.geometries.count(entry.first)!=0)
        local.geometryAttributes.emplace(entry.first,std::move(entry.second));
    *materialized = std::move(local);
  }
  return true;
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
// Points every Subdivision Surface at the level for this use and replays the plan,
// so operations after it see the chosen mesh. The bytes stay light: a preview plan
// carries the viewport mesh and an export rebuilds the render level here.
inline bool selectSubdivisionSurfaceUse(ValueDescriptor &value, bool finalRender,
                                        const AdmissionContext &context, std::string &error) {
  const float active = finalRender ? 1.0f : 0.0f;
  bool changed = false;
  for (auto &operation : value.operations)
    if (operation.code == OperationCode::subdivisionSurface && operation.parameters[2] != active) {
      changed = changed || operation.parameters[0] != operation.parameters[1];
      operation.parameters[2] = active;
    }
  if (!changed) { error.clear(); return true; }
  ValueDescriptor replayed;
  if (!validateOperationPlan(value,context,nullptr,error,&replayed)) return false;
  value = std::move(replayed);
  return true;
}
inline StableId retainedMaterialIdentity(const ValueDescriptor &value) {
  // Admission already validated the operation graph. Resolve the terminal
  // material without regenerating static geometry on every rendered frame.
  std::map<StableId,StableId> materials;
  for (const auto &operation : value.operations) {
    if (operation.code == OperationCode::setMaterial)
      materials[operation.outputStableId] = operation.stableId;
    else if (operation.code == OperationCode::assignMaterial)
      materials[operation.outputStableId] = operation.secondaryInputStableId;
    else {
      const auto input = operation.code == OperationCode::instanceOnPoints
          ? operation.secondaryInputStableId : operation.inputStableId;
      materials[operation.outputStableId] = materials.count(input) ? materials.at(input) : 1;
    }
  }
  return materials.count(value.stableId) ? materials.at(value.stableId) : 1;
}

inline void digestBytes(std::uint64_t &h, const void *p,
                        std::size_t n) noexcept {
  const auto *b = static_cast<const unsigned char *>(p);
  for (std::size_t i = 0; i < n; ++i) {
    h ^= b[i];
    h *= 1099511628211ull;
  }
}
template <class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
inline void digestValue(std::uint64_t &h, T value) noexcept {
  const auto canonical = static_cast<std::uint64_t>(value);
  for (unsigned shift = 0; shift < 64; shift += 8) {
    const auto byte = static_cast<unsigned char>((canonical >> shift) & 0xffu);
    digestBytes(h, &byte, 1);
  }
}
template <class T, std::enable_if_t<std::is_enum_v<T>, int> = 0>
inline void digestValue(std::uint64_t &h, T value) noexcept {
  digestValue(h, static_cast<std::underlying_type_t<T>>(value));
}

// The request's canonical bytes, each integer widened to 64 bits as digestValue does, so
// digesting them gives the receipt the digest it has always had.
inline void appendCanonicalInteger(std::vector<std::uint8_t> &out, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffu));
}
inline void appendSdfSurfaceRequest(std::vector<std::uint8_t> &out, const sdfsurface::Request &q) {
  const auto integer=[&](auto x) { appendCanonicalInteger(out,static_cast<std::uint64_t>(x)); };
  const auto real=[&](double x) { std::uint64_t bits=0; std::memcpy(&bits,&x,sizeof(bits)); integer(bits); };
  integer(q.algorithmVersion);
  for(auto x:q.minimum) real(x);
  for(auto x:q.maximum) real(x);
  for(auto x:q.cells) integer(x);
  real(q.gradientStep);
  real(q.maximumResidual);
  integer(q.geometry.schemaVersion); integer(q.geometry.rootId);
  for(const auto& record:q.geometry.records) {
    integer(record.stableId); integer(record.operation);
    integer(record.inputCount); integer(record.parameterCount);
    for(auto x:record.inputs) integer(x);
    for(auto x:record.parameters) real(x);
    if(q.geometry.schemaVersion==kSdfIrTextSchemaVersion) integer(record.resourceStableId);
  }
  if(q.geometry.schemaVersion==kSdfIrTextSchemaVersion) {
    integer(q.geometry.textOutlines.size());
    for(const auto& outline:q.geometry.textOutlines) {
      const auto bytes=encodeSdfTextOutline(outline);
      integer(bytes.size()); for(const auto byte:bytes) integer(static_cast<std::uint8_t>(byte));
    }
  }
}
inline void digestSdfSurfaceRequest(std::uint64_t &h, const sdfsurface::Request &q) {
  std::vector<std::uint8_t> bytes;
  appendSdfSurfaceRequest(bytes,q);
  digestBytes(h,bytes.data(),bytes.size());
}

inline bool inspectValue(const ValueDescriptor &v,
                         const AdmissionContext &context, BudgetReceipt &r,
                         std::string &e, const ResourceLimits* replayLimits = nullptr) {
  r = {};
  r.valueStableId = v.stableId;
  r.sourceStableId = v.sourceStableId;
  r.sourceRevision = v.sourceRevision;
  r.carrier = v.carrier;
  r.attributes = v.attributes.size();
  r.operations = v.operations.size();
  r.dispatches = v.dispatchCount;
  for(const auto& operation:v.operations) if(operation.sdfSurface) {
    sdfsurface::Budget budget;
    if(!sdfsurface::inspect(*operation.sdfSurface,budget,e)
        || !checkedAdd(r.sdfSamples,budget.samples,r.sdfSamples)
        || !checkedAdd(r.sdfTemporaryBytes,budget.temporaryBytes,r.sdfTemporaryBytes)
        || !addArrayBytes(1,budget.temporaryBytes,r.bufferBytes)
        || r.bufferBytes>(replayLimits ? replayLimits->maxBufferBytes : kMaximumBufferBytes)) {
      if(e.empty()) e="SDF surface plan exceeds aggregate samples or temporary memory";
      return false;
    }
    r.sdfSampleWork=sdfSaturatingAdd(r.sdfSampleWork,budget.sampleWork);
    r.sdfTextRecordVisits=sdfSaturatingAdd(r.sdfTextRecordVisits,budget.textRecordVisits);
    r.sdfTextSegmentVisits=sdfSaturatingAdd(r.sdfTextSegmentVisits,budget.textSegmentVisits);
  }
  bool payload = false;
  if (const auto *p = std::get_if<GeometryData>(&v.data)) {
    payload = v.carrier == CarrierKind::geometry3D;
    r.vertices = p->positions.size();
    r.indices = p->indices.size();
    if (p->vertexIds.size() != p->positions.size() ||
        p->indices.size() % 3 != 0) {
      e = "geometry topology cardinality is invalid";
      return false;
    }
    std::set<StableId> ids;
    for (std::size_t i = 0; i < p->positions.size(); ++i)
      if (!finite(p->positions[i]) || !validStableId(p->vertexIds[i]) ||
          !ids.insert(p->vertexIds[i]).second) {
        e = "geometry positions or vertex identities are invalid";
        return false;
      }
    for (auto index : p->indices)
      if (index >= p->positions.size()) {
        e = "geometry index references a missing vertex";
        return false;
      }
    if (!addArrayBytes(r.vertices, sizeof(Vec3) + sizeof(StableId),
                       r.bufferBytes) ||
        !addArrayBytes(r.indices, sizeof(std::uint32_t), r.bufferBytes)) {
      e = "geometry byte derivation overflow";
      return false;
    }
  } else if (const auto *p = std::get_if<PointsData>(&v.data)) {
    payload = v.carrier == CarrierKind::points3D;
    r.points = p->points.size();
    std::set<StableId> ids;
    for (const auto &x : p->points)
      if (!finite(x.position) || !finite(x.radius) || x.radius < 0 ||
          !finite(x.rotation) || !validStableId(x.stableId) ||
          !ids.insert(x.stableId).second) {
        e = "point record is invalid";
        return false;
      }
    if (!addArrayBytes(r.points, sizeof(PointRecord), r.bufferBytes)) {
      e = "point byte derivation overflow";
      return false;
    }
  } else if (const auto *p = std::get_if<CurvesData>(&v.data)) {
    payload = v.carrier == CarrierKind::curves3D;
    r.curvePoints = p->points.size();
    r.splines = p->splines.size();
    std::set<StableId> ids;
    for (const auto &x : p->points)
      if (!finite(x.position) || !finite(x.radius) || x.radius < 0 ||
          !finite(x.tilt) || !validStableId(x.stableId) ||
          !ids.insert(x.stableId).second) {
        e = "curve point record is invalid";
        return false;
      }
    ids.clear();
    std::size_t previousEnd = 0;
    for (const auto &s : p->splines) {
      std::size_t end = 0;
      if (!validStableId(s.stableId) || !ids.insert(s.stableId).second ||
          s.pointCount == 0 || s.firstPoint != previousEnd ||
          !checkedAdd(s.firstPoint, s.pointCount, end) ||
          end > p->points.size()) {
        e = "curve spline range or identity is invalid";
        return false;
      }
      previousEnd = end;
    }
    if (previousEnd != p->points.size()) {
      e = "curve splines do not cover their point records";
      return false;
    }
    if (!addArrayBytes(r.curvePoints, sizeof(CurvePointRecord),
                       r.bufferBytes) ||
        !addArrayBytes(r.splines, sizeof(SplineRecord), r.bufferBytes)) {
      e = "curve byte derivation overflow";
      return false;
    }
  } else if (const auto *p = std::get_if<InstancesData>(&v.data)) {
    payload = v.carrier == CarrierKind::instances3D;
    r.instances = p->instances.size();
    std::set<StableId> ids;
    for (const auto &x : p->instances)
      if (!validStableId(x.stableId) || !ids.insert(x.stableId).second ||
          !validStableId(x.sourceStableId) || !finite(x.transform)) {
        e = "instance identity, source reference, or transform is invalid";
        return false;
      }
    if (!addArrayBytes(r.instances, sizeof(InstanceRecord), r.bufferBytes)) {
      e = "instance byte derivation overflow";
      return false;
    }
  } else if (const auto *p = std::get_if<FieldData>(&v.data)) {
    payload = v.carrier == CarrierKind::field;
    r.fieldElements = p->elements.size();
    const auto spectral=std::find_if(v.operations.begin(),v.operations.end(),[&](const auto& op) {
      return op.code==OperationCode::fieldAudioSpectrum && op.outputStableId==v.stableId;
    });
    const auto binding=std::find_if(v.spectrumFields.begin(),v.spectrumFields.end(),[&](const auto& b) {
      return b.fieldStableId==v.stableId;
    });
    const bool deferred=spectral!=v.operations.end() && binding!=v.spectrumFields.end()
        && deferredSpectrumSource(v,*spectral,*binding,context);
    if ((!deferred && v.fieldDomainCardinality == 0) ||
        r.fieldElements != v.fieldDomainCardinality) {
      e = "typed field cardinality does not match its source domain";
      return false;
    }
    for (const auto &x : p->elements)
      if (!validFieldElement(x, v.fieldValueType)) {
        e = "typed field data is non-finite or has invalid components";
        return false;
      }
    std::size_t width = 0;
    if (!checkedMul(valueTypeComponents(v.fieldValueType), sizeof(double),
                    width) ||
        !addArrayBytes(r.fieldElements, width, r.bufferBytes)) {
      e = "field byte derivation overflow";
      return false;
    }
  }
  if (!payload) {
    e = "geometry carrier and payload shape disagree";
    return false;
  }
  for (const auto &a : v.attributes) {
    if (!domainAllowed(v.carrier, a.descriptor.domain) ||
        a.elements.size() != domainCardinality(r, a.descriptor.domain)) {
      e = "geometry attribute domain or cardinality is invalid";
      return false;
    }
    for (const auto &x : a.elements)
      if (!validFieldElement(x, a.descriptor.valueType)) {
        e = "geometry attribute data is invalid";
        return false;
      }
    std::size_t width = 0;
    if (!checkedMul(valueTypeComponents(a.descriptor.valueType), sizeof(double),
                    width) ||
        !addArrayBytes(a.elements.size(), width, r.bufferBytes)) {
      e = "attribute byte derivation overflow";
      return false;
    }
  }
  if (v.carrier==CarrierKind::instances3D && !validInstanceAppearanceAttributes(
      v.attributes,std::get<InstancesData>(v.data).instances.size(),e)) return false;
  MaterializedInstancePlan retainedSpectrumSources;
  if (!validateOperationPlan(v, context, v.spectrumFields.empty() && v.scoreFields.empty() ? nullptr : &retainedSpectrumSources, e, nullptr, nullptr, {}, replayLimits))
    return false;
  // New retained layouts own both prototype and placement buffers, plus the
  // bounded sorting/length/triangle-area workspace used during replay.
  for (const auto& operation : v.operations) if (operation.code==OperationCode::spectrumInstancer
      && operation.parameters[0]>=3) {
    std::set<StableId> sources{operation.inputStableId,operation.secondaryInputStableId};
    for (const auto id : sources) {
      if (const auto mesh=retainedSpectrumSources.geometries.find(id); mesh!=retainedSpectrumSources.geometries.end()) {
        if (!checkedAdd(r.vertices,mesh->second.positions.size(),r.vertices)
            || !checkedAdd(r.indices,mesh->second.indices.size(),r.indices)
            || !addArrayBytes(mesh->second.positions.size(),sizeof(Vec3)+sizeof(StableId),r.bufferBytes)
            || !addArrayBytes(mesh->second.indices.size(),sizeof(std::uint32_t),r.bufferBytes)
            || (id==operation.secondaryInputStableId
                && !addArrayBytes(mesh->second.indices.size()/3,sizeof(double),r.bufferBytes))) {
          e="Spectrum Instancer retained mesh budget overflow"; return false;
        }
      } else if (const auto points=retainedSpectrumSources.points.find(id); points!=retainedSpectrumSources.points.end()) {
        if (!checkedAdd(r.points,points->second.points.size(),r.points)
            || !addArrayBytes(points->second.points.size(),sizeof(PointRecord)+sizeof(const PointRecord*),r.bufferBytes)) {
          e="Spectrum Instancer retained point budget overflow"; return false;
        }
      } else if (const auto curves=retainedSpectrumSources.curves.find(id); curves!=retainedSpectrumSources.curves.end()) {
        if (!checkedAdd(r.curvePoints,curves->second.points.size(),r.curvePoints)
            || !checkedAdd(r.splines,curves->second.splines.size(),r.splines)
            || !addArrayBytes(curves->second.points.size(),sizeof(CurvePointRecord)+sizeof(double),r.bufferBytes)
            || !addArrayBytes(curves->second.splines.size(),sizeof(SplineRecord)+sizeof(std::vector<double>)+2*sizeof(double)+sizeof(const SplineRecord*),r.bufferBytes)) {
          e="Spectrum Instancer retained curve budget overflow"; return false;
        }
      }
    }
    if (!addArrayBytes(spectrum::kBands,sizeof(PointRecord),r.bufferBytes)) {
      e="Spectrum Instancer placement workspace budget overflow"; return false;
    }
  }
  if ((!v.spectrumFields.empty() && v.carrier != CarrierKind::geometry3D
          && v.carrier != CarrierKind::field && v.carrier != CarrierKind::instances3D
          && v.carrier != CarrierKind::points3D && v.carrier != CarrierKind::curves3D)) {
    e = "Spectrum binding requires a field, Geometry3D or Instances3D consumer";
    return false;
  }
  std::set<StableId> spectrumIds;
  for (const auto& binding : v.spectrumFields) {
    const auto fieldOperation=std::find_if(v.operations.begin(),v.operations.end(),[&](const auto& op) {
      return op.stableId==binding.fieldStableId && op.code==OperationCode::fieldAudioSpectrum;
    });
    if (fieldOperation!=v.operations.end()) {
      // Ordered replay above derives and verifies coordinates against the exact
      // retained source. The terminal carrier may belong to a later consumer.
      const auto* deferred=deferredSpectrumSource(v,*fieldOperation,binding,context);
      const auto capacity=deferred ? deferred->capacity : binding.coordinates.size();
      if (!validStableId(binding.fieldStableId) || !validStableId(binding.sourceGeometryStableId)
          || !spectrumIds.insert(binding.fieldStableId).second
          || (deferred && replayLimits && capacity>replayLimits->maxFieldElements)
          || !validateSpectrumFieldBinding(binding,binding.coordinates,deferred!=nullptr,e)
          || !addArrayBytes(capacity,sizeof(float),r.bufferBytes)
          || !addArrayBytes(binding.history.coordinates.size(),sizeof(float),r.bufferBytes)
          || (binding.history.enabled && !addArrayBytes(binding.history.depth,sizeof(spectrum::Bands),r.bufferBytes))
          || (binding.history.enabled && binding.history.live
              && !addArrayBytes(spectrum::liveHistoryFrames(binding.history),sizeof(spectrum::Follower::HistoryFrame),r.bufferBytes))
          || (deferred && !addArrayBytes(capacity,3*sizeof(FieldElement)+sizeof(double),r.bufferBytes))
          || !addArrayBytes(1,sizeof(spectrum::Binding),r.bufferBytes)) {
        if (e.empty()) e="Audio Spectrum Field lost its replay identity or budget";
        return false;
      }
      continue;
    }
    auto cardinality = binding.target==spectrum::Target::instanceTransform
        ? static_cast<std::size_t>(binding.lastBand-binding.firstBand+1)
        : v.carrier == CarrierKind::field ? r.fieldElements
        : v.carrier == CarrierKind::instances3D ? r.instances : r.vertices;
    // A displacement added in replay order counts its consumer's vertices.
    if (adapterSpectrumDisplacement(v,binding) && displacementInReplay(v,binding.consumerGeometryStableId)) {
      const auto consumer=retainedSpectrumSources.geometries.find(binding.consumerGeometryStableId);
      if (consumer==retainedSpectrumSources.geometries.end()) {
        e="Audio Spectrum Field consumer is not retained in its operation program"; return false;
      }
      cardinality=consumer->second.positions.size();
    }
    if (binding.target==spectrum::Target::geometryDeformation) {
      const auto source=retainedSpectrumSources.geometries.find(binding.sourceGeometryStableId);
      if (source==retainedSpectrumSources.geometries.end()) {
        e="Audio Deformer source is not retained in its operation program"; return false;
      }
      cardinality=source->second.positions.size();
    }
    if (!spectrum::validate(binding, cardinality, e)) return false;
    const auto operation = std::find_if(v.operations.begin(), v.operations.end(),
        [&](const auto& op) { return op.stableId == binding.fieldStableId; });
    if (!validStableId(binding.fieldStableId) || !validStableId(binding.sourceGeometryStableId)
        || !spectrumIds.insert(binding.fieldStableId).second
        || operation == v.operations.end()
        || (v.carrier == CarrierKind::field && binding.target!=spectrum::Target::geometryDeformation
            && binding.target!=spectrum::Target::instanceTransform
            && (v.stableId != binding.fieldStableId || v.fieldDomain != Domain::vertex
                || v.fieldValueType != ValueType::vector))) {
      e = "Audio Spectrum Field lost its typed operation identity";
      return false;
    }
    if (binding.target == spectrum::Target::instanceTransform) {
      if ((v.carrier != CarrierKind::instances3D && v.carrier != CarrierKind::field)
          || operation->code != OperationCode::spectrumInstancer
          || operation->outputStableId != binding.consumerGeometryStableId
          || operation->inputStableId != binding.sourceGeometryStableId) {
        e = "Spectrum Instancer lost its typed instance consumer";
        return false;
      }
    } else if (binding.target == spectrum::Target::geometryDeformation) {
      if (operation->code != OperationCode::audioDeformer
          || operation->inputStableId != binding.sourceGeometryStableId
          || operation->outputStableId != binding.consumerGeometryStableId) {
        e = "Audio Deformer lost its typed geometry consumer";
        return false;
      }
    } else if (operation->code != OperationCode::evaluateField) {
      e = "Audio Spectrum Field lost its field operation";
      return false;
    } else if (v.carrier == CarrierKind::geometry3D) {
      const auto consumer = std::find_if(v.operations.begin(), v.operations.end(),
          [&](const auto& op) { return op.outputStableId == binding.consumerGeometryStableId
              && op.inputStableId == binding.sourceGeometryStableId
              && (op.code == OperationCode::transform || op.code == OperationCode::transformTRS); });
      if (!validStableId(binding.consumerGeometryStableId) || consumer == v.operations.end()) {
        e = "Audio Spectrum Field lost its Set Position consumer";
        return false;
      }
    } else if (v.carrier != CarrierKind::field || binding.consumerGeometryStableId != 0) {
      e = "Audio Spectrum Field cannot claim a consumer before it is connected";
      return false;
    }
    if (!addArrayBytes(binding.coordinates.size(), sizeof(float), r.bufferBytes)
        || !addArrayBytes(binding.history.coordinates.size(), sizeof(float), r.bufferBytes)
        || (binding.history.enabled && !addArrayBytes(binding.history.depth, sizeof(spectrum::Bands), r.bufferBytes))
        || (binding.history.enabled && binding.history.live
            && !addArrayBytes(spectrum::liveHistoryFrames(binding.history),
                             sizeof(spectrum::Follower::HistoryFrame), r.bufferBytes))
        || !addArrayBytes(1, sizeof(spectrum::Binding), r.bufferBytes)) {
      e = "Audio Spectrum Field byte derivation overflow";
      return false;
    }
  }
  if ((!v.scoreFields.empty() && v.carrier != CarrierKind::field
          && v.carrier != CarrierKind::geometry3D && v.carrier != CarrierKind::instances3D)) {
    e = "Score Field requires a typed field or geometry consumer";
    return false;
  }
  std::set<StableId> scoreFieldIds;
  for (const auto& binding : v.scoreFields) {
    const auto operation = std::find_if(v.operations.begin(), v.operations.end(),
        [&](const auto& op) { return op.stableId == binding.fieldStableId; });
    const bool orderedInstance=operation!=v.operations.end() && operation->code==OperationCode::evaluateField
        && operation->inputStableId!=operation->outputStableId;
    const bool prototypeScore=v.carrier==CarrierKind::instances3D && !orderedInstance;
    const auto prototype=retainedSpectrumSources.geometries.find(binding.sourceGeometryStableId);
    const auto consumer=displacementInReplay(v,binding.consumerGeometryStableId)
        ? retainedSpectrumSources.geometries.find(binding.consumerGeometryStableId)
        : retainedSpectrumSources.geometries.end();
    const auto cardinality = orderedInstance ? binding.positions.size()
        : prototypeScore && prototype!=retainedSpectrumSources.geometries.end() ? prototype->second.positions.size()
        : consumer!=retainedSpectrumSources.geometries.end() ? consumer->second.positions.size()
        : v.carrier == CarrierKind::field ? r.fieldElements : r.vertices;
    if (!scorefield::validate(binding, cardinality, e)
        || (!orderedInstance && replayLimits && cardinality>replayLimits->maxVertices)) {
      if (e.empty()) e="Vertex Score Field exceeds its admitted vertex sample budget";
      return false;
    }
    if (!validStableId(binding.fieldStableId) || !validStableId(binding.sourceGeometryStableId)
        || !validStableId(binding.scoreSourceStableId)
        || !scoreFieldIds.insert(binding.fieldStableId).second
        || spectrumIds.count(binding.fieldStableId) != 0
        || operation == v.operations.end() || operation->code != OperationCode::evaluateField) {
      e = "Score Field lost its typed operation or score source identity";
      return false;
    }
    if (orderedInstance) {
      if (v.carrier==CarrierKind::field && !addArrayBytes(cardinality,sizeof(InstanceRecord),r.bufferBytes)) {
        e="Instance Score Field retained instance byte derivation overflow"; return false;
      }
      if (operation->inputStableId!=binding.sourceGeometryStableId || binding.consumerGeometryStableId!=0
          || (v.carrier!=CarrierKind::field && v.carrier!=CarrierKind::instances3D)) {
        e="Instance Score Field lost its ordered instance source"; return false;
      }
    } else if (v.carrier == CarrierKind::field) {
      if (v.stableId != binding.fieldStableId || v.fieldDomain != Domain::vertex
          || v.fieldValueType != ValueType::vector || binding.consumerGeometryStableId != 0) {
        e = "Score Field requires an unconsumed vertex Vector field";
        return false;
      }
    } else if (v.carrier == CarrierKind::geometry3D) {
      const auto consumer = std::find_if(v.operations.begin(), v.operations.end(),
          [&](const auto& op) { return op.outputStableId == binding.consumerGeometryStableId
              && op.inputStableId == binding.sourceGeometryStableId
              && (op.code == OperationCode::transform || op.code == OperationCode::transformTRS); });
      if (!validStableId(binding.consumerGeometryStableId) || consumer == v.operations.end()
          || binding.elementIds != std::get<GeometryData>(v.data).vertexIds) {
        e = "Score Field lost its Set Position consumer or stable vertex identities";
        return false;
      }
    } else if (prototypeScore && prototype!=retainedSpectrumSources.geometries.end()) {
      std::set<StableId> pointSources;
      for (const auto& op : v.operations) if (op.code==OperationCode::instanceOnPoints)
        pointSources.insert(op.inputStableId);
      for (const auto id : pointSources) {
        const auto points=retainedSpectrumSources.points.find(id);
        if (points==retainedSpectrumSources.points.end()
            || !checkedAdd(r.points,points->second.points.size(),r.points)
            || !addArrayBytes(points->second.points.size(),sizeof(PointRecord),r.bufferBytes)) {
          e="Score-deformed instancing exceeds its retained point buffer budget"; return false;
        }
      }
      if (!validStableId(binding.consumerGeometryStableId) || binding.elementIds!=prototype->second.vertexIds
          || !checkedAdd(r.vertices,cardinality,r.vertices)
          || !checkedAdd(r.indices,prototype->second.indices.size(),r.indices)
          || !addArrayBytes(cardinality,sizeof(Vec3)+sizeof(StableId)+2*sizeof(FieldElement),r.bufferBytes)
          || !addArrayBytes(prototype->second.indices.size(),sizeof(std::uint32_t),r.bufferBytes)) {
        e="Score-deformed instancing lost its prototype or retained memory budget"; return false;
      }
    } else {
      e = "Instances3D requires an ordered instance Score Field binding";
      return false;
    }
    if (!addArrayBytes(cardinality, sizeof(std::array<float,3>) + sizeof(StableId), r.bufferBytes)
        || !addArrayBytes(1, sizeof(scorefield::Binding), r.bufferBytes)) {
      e = "Score Field byte derivation overflow";
      return false;
    }
  }
  if (!addArrayBytes(v.operations.size(), sizeof(OperationRecord),
                     r.bufferBytes)) {
    e = "operation byte derivation overflow";
    return false;
  }
  for (const auto& operation : v.operations) {
    if (!addArrayBytes(operation.field.attributeName.size(),sizeof(char),r.bufferBytes)) {
      e="Field operation name byte derivation overflow"; return false;
    }
    if (operation.retainedMesh) {
      const auto& retained=*operation.retainedMesh;
      if (!addArrayBytes(retained.geometry.positions.size(),sizeof(Vec3)+sizeof(StableId),r.bufferBytes)
          || !addArrayBytes(retained.importedAnimation.size(),sizeof(char),r.bufferBytes)
          || !addArrayBytes(retained.geometry.indices.size(),sizeof(std::uint32_t),r.bufferBytes)) {
        e="Imported mesh source byte derivation overflow"; return false;
      }
      for (const auto& attribute : retained.attributes)
        if (!addArrayBytes(attribute.elements.size(),sizeof(FieldElement),r.bufferBytes)) {
          e="Imported mesh attribute byte derivation overflow"; return false;
        }
    }
  }
  for (const auto& sample : context.evaluatedPoints) {
    const auto* points=std::get_if<PointsData>(&sample.second.data);
    if (!points || !addArrayBytes(points->points.size(),sizeof(PointRecord),r.bufferBytes)) {
      e="Evaluated Points3D source byte derivation overflow"; return false;
    }
    for (const auto& attribute : sample.second.attributes)
      if (!addArrayBytes(attribute.elements.size(),sizeof(FieldElement),r.bufferBytes)) {
        e="Evaluated Points3D attribute byte derivation overflow"; return false;
      }
  }
  for (const auto& source : context.deferredBodyGeometry) {
    if (!source.second.reference
        || !checkedAdd(r.bufferBytes,sizeof(DeferredBodyGeometrySource),r.bufferBytes)
        || !addArrayBytes(source.second.reference->positions.capacity(),sizeof(Vec3),r.bufferBytes)
        || !addArrayBytes(source.second.reference->vertexIds.capacity(),sizeof(StableId),r.bufferBytes)
        || !addArrayBytes(source.second.reference->indices.capacity(),sizeof(std::uint32_t),r.bufferBytes)) {
      e="Deferred body Geometry3D declaration storage overflow"; return false;
    }
  }
  for (const auto& sample : context.evaluatedBodyGeometry) {
    if (!sample.second.geometry || !checkedAdd(r.bufferBytes,sizeof(BodyGeometryFrameData),r.bufferBytes)
        || !addArrayBytes(sample.second.geometry->positions.capacity(),sizeof(Vec3),r.bufferBytes)
        || !addArrayBytes(sample.second.geometry->vertexIds.capacity(),sizeof(StableId),r.bufferBytes)
        || !addArrayBytes(sample.second.geometry->indices.capacity(),sizeof(std::uint32_t),r.bufferBytes)) {
      e="Evaluated body Geometry3D storage overflow"; return false;
    }
  }
  std::uint64_t h = 1469598103934665603ull;
  digestValue(h, r.valueStableId);
  digestValue(h, r.sourceStableId);
  digestValue(h, r.sourceRevision);
  digestValue(h, r.carrier);
  digestValue(h, r.vertices);
  digestValue(h, r.indices);
  digestValue(h, r.points);
  digestValue(h, r.curvePoints);
  digestValue(h, r.splines);
  digestValue(h, r.instances);
  digestValue(h, r.fieldElements);
  digestValue(h, r.attributes);
  digestValue(h, r.operations);
  digestValue(h, r.dispatches);
  digestValue(h, r.bufferBytes);
  // Retain the exact implicit expression and sampling settings in the receipt.
  for(const auto& operation:v.operations) if(operation.sdfSurface) {
    digestValue(h,operation.stableId);
    digestSdfSurfaceRequest(h,*operation.sdfSurface);
  }
  r.deterministicDigest = h;
  return true;
}

class AdmittedValue final {
public:
  AdmittedValue(const AdmittedValue &) = default;
  AdmittedValue(AdmittedValue &&) = default;
  AdmittedValue &operator=(const AdmittedValue &) = delete;
  AdmittedValue &operator=(AdmittedValue &&) = delete;
  const ValueDescriptor &descriptor() const noexcept { return descriptor_; }
  const BudgetReceipt &receipt() const noexcept { return receipt_; }

private:
  friend std::optional<AdmittedValue>
  admitValue(ValueDescriptor, const PortContract &, const ResourceLimits &,
             const AdmissionContext &, std::string &);
  friend std::optional<AdmittedValue>
  admitValueWithRetainedResources(ValueDescriptor, const PortContract &, const ResourceLimits &,
             const AdmissionContext &, std::string &, const RetainedResourceOwner &);
  AdmittedValue(ValueDescriptor v, BudgetReceipt r)
      : descriptor_(std::move(v)), receipt_(r) {}
  const ValueDescriptor descriptor_;
  const BudgetReceipt receipt_;
};
inline bool validDeferredBodyGeometry(const DeferredBodyGeometrySource& source, std::string& error) {
  if (!source.source || !source.reference || !validStableId(source.outputStableId)
      || !validStableId(source.pointsOutputStableId) || source.structuralRevision==0
      || source.points.sourceRevision!=source.structuralRevision || source.points.capacity==0
      || source.maxVertices==0 || source.maxIndices==0) {
    error="Deferred body Geometry3D declaration is incomplete"; return false;
  }
  const auto& value=source.source->descriptor();
  const auto* mesh=std::get_if<GeometryData>(&value.data);
  if (value.carrier!=CarrierKind::geometry3D || !mesh || value.stableId!=source.terminalStableId
      || value.sourceStableId!=source.sourceStableId || value.sourceRevision!=source.sourceRevision
      || source.sourceRevision==0 || mesh->positions.empty() || mesh->indices.empty()
      || mesh->positions.size()>source.maxVertices || mesh->indices.size()>source.maxIndices
      || mesh->vertexIds.size()!=mesh->positions.size() || !equal(*mesh,*source.reference)) {
    error="Deferred body Geometry3D declaration differs from its exact admitted reference"; return false;
  }
  return true;
}
inline std::optional<AdmittedValue>
admitValueWithRetainedResources(ValueDescriptor v, const PortContract &c, const ResourceLimits &l,
           const AdmissionContext &context, std::string &e, const RetainedResourceOwner &resources) {
  e.clear();
  if (!validatePortContract(c, e) || !validateLimits(l, e))
    return std::nullopt;
  if (v.schemaVersion != kSchemaVersion || v.carrier != c.carrier ||
      !validStableId(v.stableId) || !validStableId(v.sourceStableId) ||
      v.sourceRevision == 0 || v.overflow != c.overflow) {
    e = "geometry value identity or carrier contract is invalid";
    return std::nullopt;
  }
  if (v.carrier == CarrierKind::field) {
    if (v.fieldDomain != c.fieldDomain ||
        v.fieldValueType != c.fieldValueType ||
        v.fieldInterpolation != c.fieldInterpolation) {
      e = "geometry field value does not match its typed port contract";
      return std::nullopt;
    }
  } else if (v.fieldDomain != Domain::unspecified ||
             v.fieldValueType != ValueType::unspecified ||
             v.fieldInterpolation != Interpolation::unspecified ||
             v.fieldDomainCardinality != 0) {
    e = "non-field geometry value carries field semantics";
    return std::nullopt;
  }
  if (v.attributes.size() != c.attributes.size()) {
    e = "geometry value attribute schema does not match its port contract";
    return std::nullopt;
  }
  for (std::size_t i = 0; i < v.attributes.size(); ++i)
    if (!(v.attributes[i].descriptor == c.attributes[i])) {
      e = "geometry value attribute schema does not match its port contract";
      return std::nullopt;
    }
  BudgetReceipt r;
  if (!inspectValue(v, context, r, e, &l))
    return std::nullopt;
  for (const auto& operation : v.operations) if (operation.code==OperationCode::fieldAudioSpectrum) {
    const auto binding=std::find_if(v.spectrumFields.begin(),v.spectrumFields.end(),[&](const auto& b) {
      return b.fieldStableId==operation.stableId;
    });
    const auto* deferred=binding==v.spectrumFields.end() ? nullptr : deferredSpectrumSource(v,operation,*binding,context);
    if (deferred && ((v.carrier==CarrierKind::field && deferred->capacity>c.maxFieldElements)
        || (v.carrier==CarrierKind::points3D && deferred->capacity>c.maxPoints)
        || (v.carrier==CarrierKind::instances3D && deferred->capacity>c.maxInstances))) {
      e="Deferred Audio Spectrum Field exceeds its typed port capacity"; return std::nullopt;
    }
  }
  const auto exceeds = [](std::size_t n, std::size_t contractLimit,
                          std::size_t resourceLimit) {
    return n > resourceLimit || n > contractLimit;
  };
  bool bad = false;
  switch (v.carrier) {
  case CarrierKind::geometry3D:
    bad = exceeds(r.vertices, c.maxVertices, l.maxVertices) ||
          exceeds(r.indices, c.maxIndices, l.maxIndices);
    break;
  case CarrierKind::points3D:
    bad = exceeds(r.points, c.maxPoints, l.maxPoints);
    break;
  case CarrierKind::curves3D:
    bad = exceeds(r.curvePoints, c.maxCurvePoints, l.maxCurvePoints) ||
          exceeds(r.splines, c.maxSplines, l.maxSplines);
    break;
  case CarrierKind::instances3D:
    bad = exceeds(r.instances, c.maxInstances, l.maxInstances)
        || r.vertices>l.maxVertices || r.indices>l.maxIndices || r.points>l.maxPoints
        || r.curvePoints>l.maxCurvePoints || r.splines>l.maxSplines;
    break;
  case CarrierKind::field:
    bad = exceeds(r.fieldElements, c.maxFieldElements, l.maxFieldElements);
    break;
  default:
    bad = true;
    break;
  }
  if (bad || r.attributes > c.maxAttributes || r.attributes > l.maxAttributes ||
      r.operations > l.maxOperations || r.dispatches > l.maxDispatches ||
      r.bufferBytes > l.maxBufferBytes) {
    e = "geometry value exceeds an admitted resource budget";
    return std::nullopt;
  }
  // Admission owns the retained bytes even if the producer kept a mutable alias.
  for (auto& operation : v.operations) {
    if (operation.retainedMesh) {
      if (resources)
        operation.retainedMesh=makeRetainedResourcePayload<const RetainedMeshData>(resources,*operation.retainedMesh);
      else
        operation.retainedMesh=std::make_shared<const RetainedMeshData>(*operation.retainedMesh);
    }
    if (operation.sdfSurface)
      operation.sdfSurface=std::make_shared<const sdfsurface::Request>(*operation.sdfSurface);
  }
  return AdmittedValue(std::move(v), r);
}
inline std::optional<AdmittedValue>
admitValue(ValueDescriptor v, const PortContract &c, const ResourceLimits &l,
           const AdmissionContext &context, std::string &e) {
  return admitValueWithRetainedResources(std::move(v),c,l,context,e,{});
}
inline std::optional<AdmittedValue> admitValue(ValueDescriptor v,
                                               const PortContract &c,
                                               const ResourceLimits &l,
                                               std::string &e) {
  return admitValue(std::move(v), c, l, AdmissionContext{}, e);
}
inline std::optional<AdmittedValue>
admitValue(ValueDescriptor v, const PortContract &c, std::string &e) {
  return admitValue(std::move(v), c, ResourceLimits{}, AdmissionContext{}, e);
}

inline std::optional<std::vector<StableId>>
applyOverflowPolicy(const std::vector<StableId> &s, std::size_t capacity,
                    OverflowPolicy p, std::string &e) {
  e.clear();
  std::set<StableId> u;
  for (auto id : s)
    if (!validStableId(id) || !u.insert(id).second) {
      e = "geometry element identities are invalid or duplicated";
      return std::nullopt;
    }
  if (s.size() <= capacity)
    return s;
  if (p == OverflowPolicy::reject) {
    e = "geometry element capacity exceeded";
    return std::nullopt;
  }
  if (p != OverflowPolicy::keepLowestStableIds) {
    e = "geometry overflow policy is unsupported";
    return std::nullopt;
  }
  std::vector<StableId> out(u.begin(), u.end());
  out.resize(capacity);
  return out;
}

// Bounded production lowering primitives.  They produce ordinary immutable
// descriptors, so the same admission, transport, and GPU path is used by
// preview and export.  They never resize after their exact cardinality check.
inline std::optional<ValueDescriptor>
lowerGrid(StableId valueId, StableId sourceId, std::uint64_t revision,
          std::uint32_t columns, std::uint32_t rows, float spacing,
          const PortContract &contract, std::string &error, float grade = 0.0f) try {
  if (contract.carrier != CarrierKind::geometry3D || columns < 2 || rows < 2 ||
      !finite(spacing) || spacing <= 0.0f || !(grade >= 0.0f && grade <= kMaximumGridGrade)) {
    error = "Grid lowering has an invalid typed contract or dimensions";
    return std::nullopt;
  }
  std::size_t vertices = 0, cells = 0, indices = 0;
  if (!checkedMul(columns, rows, vertices) || !checkedMul(columns - 1, rows - 1, cells) ||
      !checkedMul(cells, 6, indices) || vertices > contract.maxVertices ||
      indices > contract.maxIndices) {
    error = "Grid lowering exceeds its admitted geometry capacity";
    return std::nullopt;
  }
  ValueDescriptor result;
  result.carrier = CarrierKind::geometry3D; result.stableId = valueId;
  result.sourceStableId = sourceId; result.sourceRevision = revision;
  result.overflow = contract.overflow; result.data = GeometryData{};
  auto &data = std::get<GeometryData>(result.data);
  data.positions.reserve(vertices); data.vertexIds.reserve(vertices); data.indices.reserve(indices);
  for (std::uint32_t y = 0; y < rows; ++y)
    for (std::uint32_t x = 0; x < columns; ++x) {
      const auto index = static_cast<StableId>(y) * columns + x + 1;
      data.positions.push_back({gridCoordinate(x, columns, spacing, grade),
                                gridCoordinate(y, rows, spacing, grade), 0.0f});
      data.vertexIds.push_back(index);
    }
  for (std::uint32_t y = 0; y + 1 < rows; ++y)
    for (std::uint32_t x = 0; x + 1 < columns; ++x) {
      const auto a = y * columns + x, b = a + 1, c = a + columns, d = c + 1;
      data.indices.insert(data.indices.end(), {a, b, c, b, d, c});
    }
  result.operations = {{valueId, OperationCode::grid, sourceId, valueId,
                        {static_cast<float>(columns), static_cast<float>(rows),
                         spacing, grade}}};
  result.dispatchCount = 1; error.clear(); return result;
} catch (const std::bad_alloc &) {
  error = "Grid lowering could not allocate its geometry";
  return std::nullopt;
} catch (const std::length_error &) {
  error = "Grid lowering could not allocate its geometry";
  return std::nullopt;
}

inline std::optional<ValueDescriptor>
lowerSdfSurface(StableId valueId,StableId sourceId,std::uint64_t revision,
    const sdfsurface::Request& request,const PortContract& contract,std::string& error) {
  if(contract.carrier!=CarrierKind::geometry3D || contract.attributes!=sdfSurfaceAttributes(valueId)
      || contract.maxAttributes<3 || !validStableId(valueId) || !validStableId(sourceId) || revision==0) {
    error="SDF surface requires its exact Geometry3D attributes and source identity"; return std::nullopt;
  }
  ValueDescriptor result;
  result.carrier=CarrierKind::geometry3D; result.stableId=valueId; result.sourceStableId=sourceId;
  result.sourceRevision=revision; result.overflow=contract.overflow;
  GeometryData mesh;
  if(!materializeSdfSurface(request,valueId,std::min(contract.maxVertices,kMaximumVertices),
      std::min(contract.maxIndices,kMaximumIndices),mesh,result.attributes,error)) return std::nullopt;
  result.data=std::move(mesh);
  OperationRecord operation{valueId,OperationCode::sdfSurface,sourceId,valueId,{}};
  operation.sdfSurface=std::make_shared<const sdfsurface::Request>(request);
  result.operations={std::move(operation)}; result.dispatchCount=1;
  error.clear(); return result;
}

inline std::optional<ValueDescriptor>
lowerMeshGenerator(StableId valueId, StableId sourceId, std::uint64_t revision,
                   OperationCode code, std::array<float, 4> parameters,
                   const PortContract &contract, std::string &error) {
  if (contract.carrier != CarrierKind::geometry3D) {
    error = "Mesh generator requires a Geometry3D contract";
    return std::nullopt;
  }
  OperationRecord operation{valueId, code, sourceId, valueId, parameters};
  GeometryData mesh;
  if (!materializeMeshGenerator(operation, mesh, contract.maxVertices,
                                contract.maxIndices, error)) return std::nullopt;
  ValueDescriptor result;
  result.carrier = CarrierKind::geometry3D;
  result.stableId = valueId; result.sourceStableId = sourceId;
  result.sourceRevision = revision; result.overflow = contract.overflow;
  result.data = std::move(mesh); result.operations = {operation};
  result.dispatchCount = 1;
  error.clear();
  return result;
}

inline std::optional<ValueDescriptor>
lowerElementField(const ValueDescriptor &source, StableId valueId, OperationCode code,
                  const PortContract &contract, std::string &error) {
  if (contract.carrier != CarrierKind::field ||
      contract.fieldDomain != elementDomain(source.carrier) ||
      contract.fieldValueType != elementFieldType(code) ||
      source.operations.size() >= kMaximumOperations || source.dispatchCount >= kMaximumDispatches) {
    error = "Element field requires a matching domain, value type, and operation budget";
    return std::nullopt;
  }
  FieldData field;
  if (!materializeElementField(source.data, source.carrier, code, field,
                               contract.maxFieldElements, error)) return std::nullopt;
  if (code==OperationCode::normalField)
    if (const auto* normal=findNamedAttribute(source.attributes,"normal")) field.elements=normal->elements;
  ValueDescriptor result;
  result.carrier = CarrierKind::field;
  result.stableId = valueId; result.sourceStableId = source.sourceStableId;
  result.sourceRevision = source.sourceRevision; result.overflow = contract.overflow;
  result.fieldDomain = contract.fieldDomain; result.fieldValueType = contract.fieldValueType;
  result.fieldInterpolation = contract.fieldInterpolation;
  result.fieldDomainCardinality = field.elements.size();
  result.data = std::move(field);
  result.operations = source.operations;
  result.spectrumFields = source.spectrumFields;
  result.operations.push_back({valueId, code, source.stableId, valueId, {0, 0, 0, 0}});
  result.dispatchCount = source.dispatchCount + 1;
  error.clear();
  return result;
}

inline bool appendUniqueOperations(std::vector<OperationRecord> &operations,
                                    const std::vector<OperationRecord> &additional,
                                    std::string &error) {
  for (const auto &operation : additional) {
    const auto existing = std::find_if(operations.begin(), operations.end(),
        [&](const auto &candidate) { return candidate.stableId == operation.stableId; });
    if (existing != operations.end()) {
      if (existing->code != operation.code || existing->inputStableId != operation.inputStableId ||
          existing->outputStableId != operation.outputStableId || existing->parameters != operation.parameters ||
          existing->secondaryInputStableId != operation.secondaryInputStableId ||
          !equal(existing->transform,operation.transform) || !(existing->field==operation.field)
          || existing->sharpEdges!=operation.sharpEdges
          || static_cast<bool>(existing->retainedMesh)!=static_cast<bool>(operation.retainedMesh)
          || (existing->retainedMesh && (!equal(existing->retainedMesh->geometry,operation.retainedMesh->geometry)
              || !equalAttributes(existing->retainedMesh->attributes,operation.retainedMesh->attributes)
              || existing->retainedMesh->importedAnimation!=operation.retainedMesh->importedAnimation))
          || static_cast<bool>(existing->sdfSurface)!=static_cast<bool>(operation.sdfSurface)
          || (existing->sdfSurface && !sdfsurface::equal(*existing->sdfSurface,*operation.sdfSurface))
          || static_cast<bool>(existing->subdivision)!=static_cast<bool>(operation.subdivision)
          || (existing->subdivision && !equalSubdivision(*existing->subdivision,*operation.subdivision))) {
        error = "Geometry branches disagree on a shared operation identity";
        return false;
      }
    } else {
      if (operations.size() >= kMaximumOperations) {
        error = "Geometry branch combination exceeds the operation budget";
        return false;
      }
      operations.push_back(operation);
    }
  }
  return true;
}

inline bool hasOnlyOrderedSpectrumBindings(const ValueDescriptor& value) {
  return std::all_of(value.spectrumFields.begin(),value.spectrumFields.end(),[&](const auto& binding) {
    return binding.target==spectrum::Target::geometryDeformation
        || (binding.target==spectrum::Target::instanceTransform
            && std::any_of(value.operations.begin(),value.operations.end(),[&](const auto& op) {
              return op.code==OperationCode::spectrumInstancer && op.stableId==binding.fieldStableId
                  && op.outputStableId==binding.consumerGeometryStableId && op.inputStableId==binding.sourceGeometryStableId;
            }))
        || std::any_of(value.operations.begin(),value.operations.end(),[&](const auto& op) {
          return op.stableId==binding.fieldStableId && op.code==OperationCode::fieldAudioSpectrum;
        });
  });
}
inline bool appendReplaySpectrumBindings(std::vector<spectrum::Binding>& bindings,
    const ValueDescriptor& value, std::string& error) {
  if (!hasOnlyOrderedSpectrumBindings(value)) {
    error="Geometry replay requires ordered audio field bindings"; return false;
  }
  for (const auto& binding : value.spectrumFields) {
    const auto prior=std::find_if(bindings.begin(),bindings.end(),[&](const auto& b){return b.fieldStableId==binding.fieldStableId;});
    if (prior!=bindings.end()) {
      if (!spectrum::sameBinding(*prior,binding)) { error="Geometry branches disagree on an audio field binding"; return false; }
    } else {
      bindings.push_back(binding);
    }
  }
  return true;
}

inline bool hasOnlyOrderedScoreBindings(const ValueDescriptor& value) {
  return std::all_of(value.scoreFields.begin(),value.scoreFields.end(),[&](const auto& binding) {
    return binding.consumerGeometryStableId==0 && std::any_of(value.operations.begin(),value.operations.end(),[&](const auto& op) {
      return op.code==OperationCode::evaluateField && op.stableId==binding.fieldStableId
          && op.outputStableId==binding.fieldStableId && op.inputStableId==binding.sourceGeometryStableId
          && op.inputStableId!=op.outputStableId;
    });
  });
}
inline bool appendReplayScoreBindings(std::vector<scorefield::Binding>& bindings,
    const ValueDescriptor& value, std::string& error) {
  if (!hasOnlyOrderedScoreBindings(value)) {
    error="Field composition requires ordered instance Score Field bindings"; return false;
  }
  for (const auto& binding : value.scoreFields) {
    const auto prior=std::find_if(bindings.begin(),bindings.end(),[&](const auto& b){return b.fieldStableId==binding.fieldStableId;});
    if (prior!=bindings.end()) {
      if (!scorefield::sameBinding(*prior,binding)) { error="Geometry branches disagree on a Score Field binding"; return false; }
    } else {
      bindings.push_back(binding);
    }
  }
  return true;
}
inline std::optional<ValueDescriptor> lowerInstanceScoreField(const ValueDescriptor& source,
    StableId valueId, scorefield::Binding binding, const PortContract& contract, std::string& error) {
  if (source.carrier!=CarrierKind::instances3D || contract.carrier!=CarrierKind::field
      || contract.fieldDomain!=Domain::instance || contract.fieldValueType!=ValueType::vector
      || !hasOnlyOrderedSpectrumBindings(source) || !source.scoreFields.empty() || hasRuntimeFields(source)
      || std::any_of(source.spectrumFields.begin(),source.spectrumFields.end(),[](const auto& b) {
        return b.target!=spectrum::Target::instanceTransform || b.history.enabled;
      })
      || source.operations.size()>=kMaximumOperations || valueId==source.stableId) {
    error="Instance Score Field requires fixed retained Instances3D and a Vector output"; return std::nullopt;
  }
  BudgetReceipt sourceReceipt;
  if (!inspectValue(source,{},sourceReceipt,error)) return std::nullopt;
  binding.fieldStableId=valueId; binding.sourceGeometryStableId=source.stableId; binding.consumerGeometryStableId=0;
  binding.positions.clear(); binding.elementIds.clear();
  const auto& instances=std::get<InstancesData>(source.data).instances;
  if (instances.size()>contract.maxFieldElements) {
    error="Score Field instance samples exceed the output contract's field elements"; return std::nullopt;
  }
  if (!scorefield::reserveSamples(binding,instances.size(),"Score Field",error)) return std::nullopt;
  for (const auto& instance : instances) {
    const auto& p=instance.transform.translation;
    binding.positions.push_back({p.x,p.y,p.z}); binding.elementIds.push_back(instance.stableId);
  }
  if (!scorefield::validate(binding,instances.size(),error)) return std::nullopt;
  ValueDescriptor result;
  result.carrier=CarrierKind::field; result.stableId=valueId; result.sourceStableId=source.sourceStableId;
  result.sourceRevision=source.sourceRevision; result.fieldDomain=Domain::instance; result.fieldValueType=ValueType::vector;
  result.fieldInterpolation=Interpolation::constant; result.fieldDomainCardinality=instances.size();
  FieldData field; field.elements.resize(instances.size()); result.data=std::move(field);
  result.operations=source.operations; result.operations.push_back({valueId,OperationCode::evaluateField,source.stableId,valueId,{}});
  result.spectrumFields=source.spectrumFields;
  result.scoreFields.push_back(std::move(binding)); result.dispatchCount=result.operations.size();
  if (!admitValue(result,contract,error)) return std::nullopt;
  return result;
}

inline std::optional<ValueDescriptor> lowerAudioSpectrumField(const ValueDescriptor& source,
    StableId valueId, spectrum::Binding binding, FieldOperationSettings settings,
    const PortContract& contract, std::string& error, const AdmissionContext& context = {}) {
  if (contract.carrier!=CarrierKind::field || contract.fieldDomain!=elementDomain(source.carrier)
      || contract.fieldDomain!=settings.domain || contract.fieldValueType!=settings.valueType
      || source.operations.size()>=kMaximumOperations || !hasOnlyOrderedSpectrumBindings(source)
      || !source.scoreFields.empty()) {
    error="Audio Spectrum Field requires a bounded typed source and matching output"; return std::nullopt;
  }
  OperationRecord operation{valueId,OperationCode::fieldAudioSpectrum,source.stableId,valueId,{},0,{},settings};
  binding.fieldStableId=valueId; binding.sourceGeometryStableId=source.stableId; binding.consumerGeometryStableId=0;
  binding.target=spectrum::Target::vertexDisplacement;
  if (binding.history.enabled && context.deferredPoints.count(source.stableId)!=0) {
    error="Spectral History Field does not support deferred topology"; return std::nullopt;
  }
  const auto* deferred=deferredSpectrumSource(source,operation,binding,context);
  if (deferred && (source.carrier!=CarrierKind::points3D || !std::holds_alternative<PointsData>(source.data)
      || !std::get<PointsData>(source.data).points.empty() || !source.attributes.empty()
      || !context.evaluatedPoints.empty() || deferred->capacity>contract.maxFieldElements)) {
    error="Deferred Audio Spectrum Field requires an unresolved source and bounded field capacity"; return std::nullopt;
  }
  if (!spectrumCoordinates(source.data,source.carrier,source.attributes,operation,binding,binding.coordinates,error,deferred!=nullptr)) return std::nullopt;
  if (binding.history.enabled && (deferred
      || !spectrumHistoryCoordinates(source.data,source.carrier,operation,binding.history.coordinates,error))) {
    if (error.empty()) error="Spectral History Field does not support deferred topology";
    return std::nullopt;
  }
  FieldData field;
  if (binding.coordinates.size()>contract.maxFieldElements
      || !materializeSpectrumField(binding,operation,binding.coordinates,field,nullptr,error,deferred!=nullptr)) return std::nullopt;
  ValueDescriptor result;
  result.carrier=CarrierKind::field; result.stableId=valueId; result.sourceStableId=source.sourceStableId;
  result.sourceRevision=source.sourceRevision; result.fieldDomain=settings.domain; result.fieldValueType=settings.valueType;
  result.fieldInterpolation=Interpolation::constant; result.fieldDomainCardinality=field.elements.size(); result.data=std::move(field);
  result.operations=source.operations; result.operations.push_back(operation); result.spectrumFields=source.spectrumFields;
  result.spectrumFields.push_back(std::move(binding)); result.dispatchCount=result.operations.size();
  if (!admitValue(result,contract,{},context,error)) return std::nullopt;
  return result;
}

inline std::optional<ValueDescriptor>
lowerVectorDisplacement(const ValueDescriptor &source, const ValueDescriptor &field,
                         StableId valueId, float strength, const PortContract &contract,
                         std::string &error, const AdmissionContext& context = {}) {
  if (source.carrier != contract.carrier || field.carrier != CarrierKind::field ||
      field.fieldDomain != elementDomain(source.carrier) || field.fieldValueType != ValueType::vector ||
      !std::holds_alternative<FieldData>(field.data)) {
    error = "Vector displacement requires a Vector field on the source element domain";
    return std::nullopt;
  }
  ValueDescriptor result = source;
  if (!appendUniqueOperations(result.operations, field.operations, error)
      || !appendReplaySpectrumBindings(result.spectrumFields,field,error)) return std::nullopt;
  if (result.operations.size() >= kMaximumOperations) {
    error = "Vector displacement exceeds the operation budget";
    return std::nullopt;
  }
  if (!displaceElements(result.data, source.carrier, std::get<FieldData>(field.data), strength, error))
    return std::nullopt;
  result.stableId = valueId; result.overflow = contract.overflow;
  result.operations.push_back({valueId, OperationCode::vectorDisplacement, source.stableId, valueId,
                               {strength, 0, 0, 0}, field.stableId});
  result.dispatchCount = result.operations.size();
  if (!admitValue(result, contract, {}, context, error)) return std::nullopt;
  error.clear();
  return result;
}

inline bool applyTypedVectorField(ValueDescriptor &geometry, const FieldData &field,
                                  Domain domain, std::string &error) {
  if (geometry.carrier != CarrierKind::geometry3D || domain != Domain::vertex ||
      field.elements.size() != std::get<GeometryData>(geometry.data).positions.size()) {
    error = "typed vector field does not match the Geometry3D vertex domain";
    return false;
  }
  auto &positions = std::get<GeometryData>(geometry.data).positions;
  // Validate the complete field before changing the first position. Admission
  // failures are transactional, including a bad element late in the field.
  for (const auto &element : field.elements)
    if (!validFieldElement(element, ValueType::vector)) {
      error = "typed vector field contains an invalid element"; return false;
    }
  for (std::size_t i = 0; i < positions.size(); ++i) {
    positions[i].x += static_cast<float>(field.elements[i].components[0]);
    positions[i].y += static_cast<float>(field.elements[i].components[1]);
    positions[i].z += static_cast<float>(field.elements[i].components[2]);
  }
  error.clear(); return true;
}

inline std::optional<ValueDescriptor>
lowerCurveCircle(StableId valueId, StableId sourceId, std::uint64_t revision,
                 std::uint32_t count, float radius, const PortContract &contract,
                 std::string &error) {
  if (contract.carrier != CarrierKind::curves3D || count < 3 || !finite(radius) ||
      radius <= 0.0f || count > contract.maxCurvePoints || contract.maxSplines < 1) {
    error = "Curve Circle lowering exceeds its typed capacity"; return std::nullopt;
  }
  ValueDescriptor result;
  result.carrier = CarrierKind::curves3D; result.stableId = valueId;
  result.sourceStableId = sourceId; result.sourceRevision = revision;
  result.overflow = contract.overflow; result.data = CurvesData{};
  result.operations = {{valueId, OperationCode::curveCircle, sourceId, valueId,
                        {radius, static_cast<float>(count), 0, 0}}};
  if (!materializeCurveGenerator(result.operations.front(), std::get<CurvesData>(result.data),
                                 contract.maxCurvePoints, error)) return std::nullopt;
  result.dispatchCount = 1; error.clear(); return result;
}

inline std::optional<ValueDescriptor>
lowerCurveLine(StableId valueId, StableId sourceId, std::uint64_t revision,
               std::uint32_t count, float length, const PortContract &contract,
               std::string &error) {
  if (contract.carrier != CarrierKind::curves3D || contract.maxSplines < 1) {
    error = "Curve Line requires a bounded Curves3D contract"; return std::nullopt;
  }
  ValueDescriptor result;
  result.carrier = CarrierKind::curves3D; result.stableId = valueId;
  result.sourceStableId = sourceId; result.sourceRevision = revision;
  result.overflow = contract.overflow; result.data = CurvesData{};
  result.operations = {{valueId, OperationCode::curveLine, sourceId, valueId,
                        {length, static_cast<float>(count), 0, 0}}};
  if (!materializeCurveGenerator(result.operations.front(), std::get<CurvesData>(result.data),
                                 contract.maxCurvePoints, error)) return std::nullopt;
  result.dispatchCount = 1; error.clear(); return result;
}

inline std::optional<ValueDescriptor>
lowerCurveToMesh(const ValueDescriptor &curves, StableId valueId,
                 const PortContract &contract, std::string &error, float width = 0.1f) {
  if (curves.carrier != CarrierKind::curves3D || !std::holds_alternative<CurvesData>(curves.data) ||
      contract.carrier != CarrierKind::geometry3D) {
    error = "Curve to Mesh requires Curves3D to Geometry3D contracts"; return std::nullopt;
  }
  if (curves.operations.size() >= kMaximumOperations || curves.dispatchCount >= kMaximumDispatches) {
    error = "Curve to Mesh exceeds the operation budget"; return std::nullopt;
  }
  GeometryData mesh;
  if (!materializeCurveRibbon(std::get<CurvesData>(curves.data), width, mesh,
                              contract.maxVertices, contract.maxIndices, error)) return std::nullopt;
  ValueDescriptor result;
  result.carrier = CarrierKind::geometry3D; result.stableId = valueId;
  result.sourceStableId = curves.sourceStableId; result.sourceRevision = curves.sourceRevision;
  result.overflow = contract.overflow; result.data = std::move(mesh);
  result.operations = curves.operations; result.spectrumFields=curves.spectrumFields;
  result.operations.push_back({valueId, OperationCode::curveToMesh, curves.stableId, valueId,
                               {width, 0, 0, 0}});
  result.dispatchCount = curves.dispatchCount + 1; error.clear(); return result;
}

inline std::optional<ValueDescriptor>
lowerCurveOperation(const ValueDescriptor &curves, StableId valueId, OperationCode code,
                     std::array<float,4> parameters, const PortContract &contract,
                     std::string &error, const ValueDescriptor *field=nullptr) {
  if (curves.carrier!=CarrierKind::curves3D || !std::holds_alternative<CurvesData>(curves.data) ||
      !hasOnlyOrderedSpectrumBindings(curves) || !curves.scoreFields.empty() || curves.operations.size()>=kMaximumOperations) {
    error="Curve operation requires retained Curves3D within its operation budget"; return std::nullopt;
  }
  ValueDescriptor result;
  result.carrier=contract.carrier; result.stableId=valueId;
  result.sourceStableId=curves.sourceStableId; result.sourceRevision=curves.sourceRevision;
  result.overflow=contract.overflow; result.operations=curves.operations; result.spectrumFields=curves.spectrumFields;
  OperationRecord operation{valueId,code,curves.stableId,valueId,parameters};
  const FieldData *fieldData=nullptr;
  if (field) {
    if (field->carrier!=CarrierKind::field || field->fieldDomain!=Domain::curvePoint ||
        field->fieldValueType!=ValueType::floatValue || !std::holds_alternative<FieldData>(field->data) ||
        !hasOnlyOrderedSpectrumBindings(*field) || !field->scoreFields.empty() ||
        (!appendUniqueOperations(result.operations,field->operations,error)
          || !appendReplaySpectrumBindings(result.spectrumFields,*field,error))) {
      if (error.empty()) error="Curve attribute input requires a Float field on curve points";
      return std::nullopt;
    }
    operation.secondaryInputStableId=field->stableId;
    fieldData=&std::get<FieldData>(field->data);
  }
  const auto &source=std::get<CurvesData>(curves.data);
  if (code==OperationCode::curveToTube && contract.carrier==CarrierKind::geometry3D && !field) {
    GeometryData mesh;
    std::uint32_t sides=0;
    if (!exactIntegerParameter(parameters[1],3,sides) || parameters[2]!=0 || parameters[3]!=0 ||
        !materializeCurveTube(source,parameters[0],sides,mesh,contract.maxVertices,contract.maxIndices,error))
      return std::nullopt;
    result.data=std::move(mesh);
  } else if (code==OperationCode::pointsFromCurves && contract.carrier==CarrierKind::points3D && !field) {
    PointsData points;
    if ((parameters[0]!=0 && parameters[0]!=1) || parameters[2]!=0 || parameters[3]!=0 ||
        !materializeCurvePoints(source,parameters[0]!=0,parameters[1],points,contract.maxPoints,error))
      return std::nullopt;
    result.data=std::move(points);
  } else if (contract.carrier==CarrierKind::curves3D) {
    CurvesData edited;
    if (source.splines.size()>contract.maxSplines ||
        !materializeCurveEdit(source,operation,fieldData,edited,contract.maxCurvePoints,error)) return std::nullopt;
    result.data=std::move(edited);
  } else { error="Curve operation output carrier is incompatible"; return std::nullopt; }
  if (result.operations.size()>=kMaximumOperations) { error="Curve operation exceeds its replay budget"; return std::nullopt; }
  result.operations.push_back(operation); result.dispatchCount=result.operations.size();
  error.clear(); return result;
}

inline std::optional<ValueDescriptor>
lowerPointsFromVertices(const ValueDescriptor &geometry, StableId valueId,
                        const PortContract &contract, std::string &error) {
  if (geometry.carrier != CarrierKind::geometry3D ||
      contract.carrier != CarrierKind::points3D) {
    error = "Points from Vertices requires Geometry3D to Points3D contracts";
    return std::nullopt;
  }
  const auto &input = std::get<GeometryData>(geometry.data);
  if (input.positions.size() != input.vertexIds.size() ||
      input.positions.size() > contract.maxPoints ||
      geometry.operations.size() >= kMaximumOperations ||
      geometry.dispatchCount >= kMaximumDispatches) {
    error = "Points from Vertices exceeds typed capacity";
    return std::nullopt;
  }
  std::set<StableId> identities;
  for (std::size_t index = 0; index < input.positions.size(); ++index)
    if (!finite(input.positions[index]) || !validStableId(input.vertexIds[index]) ||
        !identities.insert(input.vertexIds[index]).second) {
      error = "Points from Vertices requires finite positions and unique stable vertex identities";
      return std::nullopt;
    }
  ValueDescriptor result;
  result.carrier = CarrierKind::points3D;
  result.stableId = valueId;
  result.sourceStableId = geometry.sourceStableId;
  result.sourceRevision = geometry.sourceRevision;
  result.overflow = contract.overflow;
  result.data = PointsData{};
  auto &points = std::get<PointsData>(result.data).points;
  points.reserve(input.positions.size());
  for (std::size_t index = 0; index < input.positions.size(); ++index)
    points.push_back({input.positions[index], input.vertexIds[index], 1.0f, {}});
  result.operations = geometry.operations;
  result.spectrumFields = geometry.spectrumFields;
  result.operations.push_back({valueId, OperationCode::pointsFromVertices,
                               geometry.stableId, valueId, {0, 0, 0, 0}});
  result.dispatchCount = geometry.dispatchCount + 1;
  error.clear();
  return result;
}

// Static or animated Geometry3D input. The operation replays on the mesh it
// receives, so the same settings shade whatever the plan produces.
inline std::optional<ValueDescriptor>
lowerShadeEdit(const ValueDescriptor &input, StableId valueId, float mode, float angleDegrees,
               std::vector<std::array<std::uint32_t, 2>> sharpEdges, std::string &error) {
  const auto *geometry = std::get_if<GeometryData>(&input.data);
  if (input.carrier != CarrierKind::geometry3D || geometry == nullptr) {
    error = "Shade Mesh requires Geometry3D";
    return std::nullopt;
  }
  // A mesh that moves each frame splits every corner under By Angle. A Set Position
  // displacement moves it too once the shade puts it into replay order.
  const bool moving = hasRuntimeFields(input) || !input.scoreFields.empty()
      || std::any_of(input.spectrumFields.begin(), input.spectrumFields.end(), [&](const auto& binding) {
           return adapterSpectrumDisplacement(input, binding);
         });
  const float corners = mode == kShadeByAngle && moving ? kShadeEveryCorner : 0;
  OperationRecord operation{valueId, OperationCode::shadeMesh, input.stableId, valueId, {{mode, angleDegrees, corners, 0}}, 0, {}};
  operation.sharpEdges = std::move(sharpEdges);
  auto result = input;
  result.stableId = valueId;
  GeometryData shaded;
  if (!materializeShadedMesh(*geometry, operation, shaded, result.attributes, error)) return std::nullopt;
  result.data = std::move(shaded);
  result.operations.push_back(std::move(operation));
  ++result.dispatchCount;
  error.clear();
  return result;
}

// Geometry3D or Points3D input. Cull Geometry and Cull Points record their selection on
// the lowered value, and admission replays it. An input moved by Audio Deformer bindings
// keeps them: the selection is an index range, so the helper re-runs the cull on each
// frame's deformed mesh and the layout matches the one admitted here. Cull Geometry also
// follows a Set Position spectrum or score displacement, which then moves into replay order.
inline std::optional<ValueDescriptor>
lowerCullEdit(const ValueDescriptor &input, StableId valueId, OperationCode code,
              const std::array<float, 4> &selection, const PortContract &contract,
              std::string &error) {
  const bool mesh = code == OperationCode::cullGeometry;
  const auto *geometry = mesh ? std::get_if<GeometryData>(&input.data) : nullptr;
  const auto *points = mesh ? nullptr : std::get_if<PointsData>(&input.data);
  if ((code != OperationCode::cullGeometry && code != OperationCode::cullPoints)
      || input.carrier != (mesh ? CarrierKind::geometry3D : CarrierKind::points3D)
      || contract.carrier != input.carrier || (mesh ? geometry == nullptr : points == nullptr)
      || (mesh ? !layoutChangeFollowsBindings(input)
               : !input.scoreFields.empty()
                 || std::any_of(input.spectrumFields.begin(), input.spectrumFields.end(), [](const auto& binding) {
                      return binding.target != spectrum::Target::geometryDeformation;
                    }))
      || input.operations.size() >= kMaximumOperations || input.dispatchCount >= kMaximumDispatches) {
    error = "Cull requires retained Geometry3D or Points3D, bindings it can follow, and an available operation budget";
    return std::nullopt;
  }
  OperationRecord operation{valueId, code, input.stableId, valueId, selection, 0, {}};
  auto result = input;
  result.stableId = valueId;
  if (mesh) {
    GeometryData culled;
    GeometryCullSelection kept;
    if (!materializeGeometryCull(*geometry, operation, culled, kept, error)) return std::nullopt;
    if (culled.positions.size() > contract.maxVertices || culled.indices.size() > contract.maxIndices) {
      error = "Mesh culling exceeds its geometry capacity"; return std::nullopt;
    }
    if (!compactCulledAttributes(result.attributes,
            {{Domain::vertex, &kept.vertices, geometry->positions.size()},
             {Domain::face, &kept.faces, geometry->indices.size() / 3}}, error)) return std::nullopt;
    result.data = std::move(culled);
  } else {
    PointsData culled;
    std::vector<std::size_t> kept;
    if (!materializePointsCull(*points, operation, culled, kept, error)) return std::nullopt;
    if (culled.points.size() > contract.maxPoints) {
      error = "Point culling exceeds its point capacity"; return std::nullopt;
    }
    if (!compactCulledAttributes(result.attributes, {{Domain::point, &kept, points->points.size()}}, error))
      return std::nullopt;
    result.data = std::move(culled);
  }
  result.operations.push_back(operation);
  ++result.dispatchCount;
  error.clear();
  return result;
}

inline std::optional<ValueDescriptor>
lowerPointDistributionOperation(const ValueDescriptor &geometry, const OperationRecord &operation,
                                const PortContract &contract, std::string &error) {
  const auto valueId=operation.outputStableId;
  // A density count follows the Audio Deformer's per-frame area. The helper re-samples it on
  // each frame's deformed mesh, up to the maximum count admitted here.
  if (geometry.carrier!=CarrierKind::geometry3D || contract.carrier!=CarrierKind::points3D
      || !geometry.scoreFields.empty()
      || std::any_of(geometry.spectrumFields.begin(),geometry.spectrumFields.end(),[](const auto& binding) {
           return binding.target!=spectrum::Target::geometryDeformation;
         })
      || operation.parameters[0]>static_cast<float>(contract.maxPoints) || geometry.operations.size()>=kMaximumOperations
      || geometry.dispatchCount>=kMaximumDispatches) {
    error="Point distribution requires static or Audio Deformer geometry and an available point budget"; return std::nullopt;
  }
  PointsData points;
  if (!materializeDistributedPoints(std::get<GeometryData>(geometry.data),operation,points,error))
    return std::nullopt;
  if (points.points.size()>contract.maxPoints) {
    error="Point distribution exceeds its point capacity"; return std::nullopt;
  }
  ValueDescriptor result;
  result.carrier=CarrierKind::points3D; result.stableId=valueId;
  result.sourceStableId=geometry.sourceStableId; result.sourceRevision=geometry.sourceRevision;
  result.overflow=contract.overflow; result.data=std::move(points);
  result.operations=geometry.operations; result.operations.push_back(operation);
  result.spectrumFields=geometry.spectrumFields;
  result.dispatchCount=geometry.dispatchCount+1;
  error.clear(); return result;
}

inline std::optional<ValueDescriptor>
lowerPointDistribution(const ValueDescriptor &geometry, StableId valueId,
                        OperationCode code, std::uint32_t count, std::uint32_t seed,
                        const PortContract &contract, std::string &error,
                        PointVolumeShape shape = PointVolumeShape::box) {
  OperationRecord operation{valueId,code,geometry.stableId,valueId,
                             {static_cast<float>(count),static_cast<float>(seed),
                              static_cast<float>(static_cast<std::uint32_t>(shape)),0}};
  return lowerPointDistributionOperation(geometry,operation,contract,error);
}

// Distribute Points on Faces by density: at most maximumCount points, floor(area * density) of them, seeded.
inline std::optional<ValueDescriptor>
lowerPointDensityDistribution(const ValueDescriptor &geometry, StableId valueId,
                              std::uint32_t maximumCount, std::uint32_t seed, float density,
                              const PortContract &contract, std::string &error) {
  OperationRecord operation{valueId,OperationCode::pointsOnFacesDensity,geometry.stableId,valueId,
                             {static_cast<float>(maximumCount),static_cast<float>(seed),density,0}};
  return lowerPointDistributionOperation(geometry,operation,contract,error);
}

inline std::optional<ValueDescriptor>
lowerInstanceEdit(const ValueDescriptor &input, StableId valueId, OperationCode code,
                   const std::array<float,4> &selection, const Transform &transform,
                   const PortContract &contract, std::string &error) {
  if (input.carrier!=CarrierKind::instances3D || contract.carrier!=CarrierKind::instances3D
      || !input.scoreFields.empty()
      || std::any_of(input.spectrumFields.begin(),input.spectrumFields.end(),[](const auto& binding) {
           return binding.target!=spectrum::Target::geometryDeformation;
         })
      || input.operations.size()>=kMaximumOperations || input.dispatchCount>=kMaximumDispatches
      || (code!=OperationCode::transformInstances && code!=OperationCode::cullInstances)) {
    error="Instance editing requires static or Audio Deformer instances and an available operation budget"; return std::nullopt;
  }
  OperationRecord operation{valueId,code,input.stableId,valueId,selection,0,transform};
  InstancesData instances;
  if (!materializeInstanceEdit(std::get<InstancesData>(input.data),operation,instances,error)) return std::nullopt;
  if (instances.instances.size()>contract.maxInstances) {
    error="Instance editing exceeds its instance capacity"; return std::nullopt;
  }
  auto result=input;
  result.stableId=valueId; result.data=std::move(instances);
  if (code==OperationCode::cullInstances) {
    for (auto &attribute:result.attributes) {
      const auto &source=std::get<InstancesData>(input.data).instances;
      const auto &kept=std::get<InstancesData>(result.data).instances;
      if (attribute.descriptor.domain!=Domain::instance || attribute.elements.size()!=source.size()) {
        error="Instance culling requires matching instance attribute cardinality"; return std::nullopt;
      }
      const auto original=std::move(attribute.elements);
      attribute.elements.clear();
      std::size_t retained=0;
      for (std::size_t i=0;i<source.size() && retained<kept.size();++i)
        if (source[i].stableId==kept[retained].stableId) {
          attribute.elements.push_back(original[i]); ++retained;
        }
    }
  }
  result.operations.push_back(operation); ++result.dispatchCount;
  error.clear(); return result;
}

inline std::optional<ValueDescriptor>
lowerInstanceOnPoints(const ValueDescriptor &points, StableId sourceGeometry,
                      StableId valueId, const PortContract &contract,
                      std::string &error, const AdmissionContext& context = {}) {
  if (points.carrier != CarrierKind::points3D || contract.carrier != CarrierKind::instances3D ||
      !validStableId(sourceGeometry)) { error = "Instance on Points requires typed point and source identities"; return std::nullopt; }
  const auto &input = std::get<PointsData>(points.data);
  if (input.points.size() > contract.maxInstances ||
      points.operations.size() >= kMaximumOperations ||
      points.dispatchCount >= kMaximumDispatches) {
    error = "Instance on Points exceeds typed capacity";
    return std::nullopt;
  }
  ValueDescriptor result;
  result.carrier = CarrierKind::instances3D; result.stableId = valueId;
  result.sourceStableId = points.sourceStableId; result.sourceRevision = points.sourceRevision;
  result.overflow = contract.overflow; result.data = InstancesData{};
  auto &instances = std::get<InstancesData>(result.data);
  for (const auto &point : input.points)
    instances.instances.push_back({point.stableId, sourceGeometry,
        {point.position, point.rotation, {point.radius, point.radius, point.radius}}});
  result.operations = points.operations;
  result.spectrumFields = points.spectrumFields;
  const auto stored=std::find_if(points.operations.begin(),points.operations.end(),[&](const auto& op) {
    return op.outputStableId==points.stableId && op.code==OperationCode::storeNamedAttribute
        && op.field.attributeName=="color" && op.field.domain==Domain::point && op.field.valueType==ValueType::color;
  });
  if (stored!=points.operations.end()) for (const auto& binding : points.spectrumFields) {
    const auto sampled=std::find_if(points.operations.begin(),points.operations.end(),[&](const auto& op) {
      return op.code==OperationCode::fieldAudioSpectrum && op.stableId==stored->secondaryInputStableId
          && op.field.valueType==ValueType::color && op.inputStableId==stored->inputStableId;
    });
    const auto* color=findNamedAttribute(points.attributes,"color");
    if (sampled!=points.operations.end() && deferredSpectrumSource(points,*sampled,binding,context)
        && color && color->descriptor.domain==Domain::point && color->descriptor.valueType==ValueType::color
        && color->elements.empty() && input.points.empty()) {
      auto promoted=*color; promoted.descriptor.domain=Domain::instance;
      result.attributes.push_back(std::move(promoted));
      break;
    }
  }
  result.operations.push_back({valueId, OperationCode::instanceOnPoints,
                               points.stableId, valueId, {0, 0, 0, 0},
                               sourceGeometry});
  result.dispatchCount = points.dispatchCount + 1;
  error.clear(); return result;
}

inline std::optional<ValueDescriptor>
lowerInstanceOnPoints(const ValueDescriptor& points, const ValueDescriptor& geometry,
                      StableId valueId, const PortContract& contract, std::string& error,
                      const AdmissionContext& context = {}) {
  if (geometry.carrier!=CarrierKind::geometry3D) {
    error="Instance on Points requires source Geometry3D"; return std::nullopt;
  }
  auto result=lowerInstanceOnPoints(points,geometry.stableId,valueId,contract,error,context);
  if (!result) return std::nullopt;
  auto operations=geometry.operations;
  if (!appendUniqueOperations(operations,result->operations,error)) return std::nullopt;
  result->operations=std::move(operations);
  if (!spectrum::appendOrderedBindings(result->spectrumFields,geometry.spectrumFields,error)) return std::nullopt;
  result->scoreFields=geometry.scoreFields;
  result->dispatchCount=result->operations.size();
  if (!geometry.scoreFields.empty() && !admitValue(*result,contract,error)) return std::nullopt;
  error.clear(); return result;
}

inline std::optional<ValueDescriptor>
lowerSpectrumInstancer(const ValueDescriptor& geometry, StableId valueId,
                        spectrum::Binding binding, const std::array<float, 4>& arrangement,
                        const PortContract& contract, std::string& error,
                        const ValueDescriptor* placement = nullptr,
                        const spectrum::HistoryAxis* historyAxis = nullptr) {
  if (geometry.carrier != CarrierKind::geometry3D || !geometry.spectrumFields.empty() || !geometry.scoreFields.empty()
      || contract.carrier != CarrierKind::instances3D || !validStableId(valueId)
      || geometry.operations.size() >= kMaximumOperations
      || geometry.dispatchCount >= kMaximumDispatches) {
    error = "Spectrum Instancer requires static Geometry3D and an Instances3D contract";
    return std::nullopt;
  }
  binding.target = spectrum::Target::instanceTransform;
  binding.interpolate = false;
  binding.fieldStableId = binding.consumerGeometryStableId = valueId;
  binding.sourceGeometryStableId = geometry.stableId;
  if (binding.firstBand > binding.lastBand || binding.lastBand >= spectrum::kBands) {
    error = "Spectrum Instancer band range is invalid";
    return std::nullopt;
  }
  const auto count = binding.lastBand - binding.firstBand + 1;
  if (count > contract.maxInstances) {
    error = "Spectrum Instancer exceeds its instance capacity";
    return std::nullopt;
  }
  binding.coordinates.resize(count);
  for (std::size_t i = 0; i < count; ++i)
    binding.coordinates[i] = count > 1 ? static_cast<float>(i) / static_cast<float>(count - 1) : 0;
  const auto mode=arrangement[0];
  const auto expected=mode==3 ? CarrierKind::curves3D : mode==4 ? CarrierKind::points3D : CarrierKind::geometry3D;
  if ((mode<=2 && placement) || (mode>=3 && (!placement || placement->carrier!=expected
      || !placement->spectrumFields.empty() || !placement->scoreFields.empty()
      || !validStableId(placement->stableId) || placement->sourceRevision==0))) {
    error="Spectrum Instancer requires a static placement source of the exact layout type"; return std::nullopt;
  }
  if (placement) {
    BudgetReceipt receipt;
    if (!inspectValue(*placement,{},receipt,error)) return std::nullopt;
  }
  OperationRecord operation {valueId, OperationCode::spectrumInstancer,
                                    geometry.stableId, valueId, arrangement};
  operation.secondaryInputStableId=placement ? placement->stableId : 0;
  if (binding.history.enabled && !historyAxis) {
    error="Spectral History Spectrum Instancer requires a row axis"; return std::nullopt;
  }
  // Placement is static, so row ages come from the layout before any audio
  // displacement. The layout pass therefore runs without history controls.
  InstancesData instances;
  auto layout=binding; layout.history={};
  if (!materializeSpectrumInstances(operation, layout, instances, error,placement ? &placement->data : nullptr)) return std::nullopt;
  if (binding.history.enabled) {
    std::vector<Vec3> positions;
    positions.reserve(instances.instances.size());
    for (const auto& instance : instances.instances) positions.push_back(instance.transform.translation);
    if (!spectrumHistoryAges(positions,*historyAxis,binding.history.coordinates,error)
        || !spectrum::validate(binding,count,error)) {
      if (error.empty()) error="Spectral History Spectrum Instancer rows disagree with its instances";
      return std::nullopt;
    }
  }
  ValueDescriptor result;
  result.carrier = CarrierKind::instances3D; result.stableId = valueId;
  result.sourceStableId = geometry.sourceStableId; result.sourceRevision = geometry.sourceRevision;
  result.overflow = contract.overflow; result.data = std::move(instances);
  result.operations = geometry.operations;
  if (placement && !appendUniqueOperations(result.operations,placement->operations,error)) return std::nullopt;
  if (result.operations.size()>=kMaximumOperations || result.operations.size()>=kMaximumDispatches) {
    error="Spectrum Instancer placement exceeds its operation budget"; return std::nullopt;
  }
  result.operations.push_back(operation);
  result.dispatchCount = placement ? result.operations.size() : geometry.dispatchCount + 1;
  result.spectrumFields.push_back(std::move(binding));
  error.clear();
  return result;
}

inline std::optional<ValueDescriptor>
lowerAudioDeformer(const ValueDescriptor& input, StableId valueId, spectrum::Binding binding,
                   const PortContract& contract, std::string& error,
                   const spectrum::HistoryAxis* historyAxis = nullptr) {
  if (input.carrier != CarrierKind::geometry3D || contract.carrier != CarrierKind::geometry3D
      || !input.scoreFields.empty()
      || !validStableId(valueId) || input.operations.size() >= kMaximumOperations
      || input.dispatchCount >= kMaximumDispatches) {
    error = "Audio Deformer requires bounded Geometry3D";
    return std::nullopt;
  }
  binding.target = spectrum::Target::geometryDeformation;
  binding.fieldStableId = binding.consumerGeometryStableId = valueId;
  binding.sourceGeometryStableId = input.stableId;
  const auto& source = std::get<GeometryData>(input.data);
  const auto count = source.positions.size();
  binding.coordinates.assign(count, 0.0f);
  if (binding.history.enabled) {
    if (!historyAxis || !spectrumHistoryAges(source.positions,*historyAxis,binding.history.coordinates,error)) {
      if (error.empty()) error = "Spectral History Audio Deformer requires a row axis and retained vertices";
      return std::nullopt;
    }
  }
  if (!spectrum::validate(binding, count, error)) return std::nullopt;
  auto result = input;
  result.stableId = valueId;
  result.operations.push_back({valueId,OperationCode::audioDeformer,input.stableId,valueId,{}});
  result.spectrumFields.push_back(std::move(binding));
  ++result.dispatchCount;
  error.clear(); return result;
}

inline std::optional<ValueDescriptor>
lowerTransformGeometry(const ValueDescriptor &input, StableId valueId,
                       const Transform &transform, const PortContract &contract,
                       std::string &error) {
  if (input.carrier != CarrierKind::geometry3D || contract.carrier != CarrierKind::geometry3D ||
      !finite(transform)) { error = "Transform Geometry requires finite Geometry3D inputs"; return std::nullopt; }
  if (!input.scoreFields.empty() && !equal(transform,Transform{})) {
    error = "Place Transform Geometry before Score Field sampling"; return std::nullopt;
  }
  if (std::any_of(input.spectrumFields.begin(), input.spectrumFields.end(), [](const auto& binding) {
        return binding.target == spectrum::Target::geometryDeformation;
      })) {
    error = "Place Transform Geometry before Audio Deformer to retain the deformation's local coordinates";
    return std::nullopt;
  }
  const auto &source = std::get<GeometryData>(input.data);
  if (source.positions.size() > contract.maxVertices || source.indices.size() > contract.maxIndices
      || input.operations.size() >= kMaximumOperations
      || input.dispatchCount >= kMaximumDispatches) {
    error = "Transform Geometry exceeds its typed capacity"; return std::nullopt;
  }
  for (const auto &position : source.positions)
    if (!finite(position)) {
      error = "Transform Geometry contains a non-finite source position";
      return std::nullopt;
    }
  ValueDescriptor result = input; result.stableId = valueId; result.overflow = contract.overflow;
  if (!materializeTransform(std::get<GeometryData>(result.data),transform,error)
      || !transformAuthoredNormals(result.attributes,transform,error)) return std::nullopt;
  for (auto& binding : result.spectrumFields) {
    const auto direction = transformVector({binding.direction[0],binding.direction[1],binding.direction[2]},transform);
    binding.direction = {direction.x,direction.y,direction.z};
  }
  result.operations.push_back({valueId, OperationCode::transformTRS, input.stableId, valueId,
                               {},0,transform});
  ++result.dispatchCount; error.clear(); return result;
}

inline std::optional<ValueDescriptor>
lowerAssignMaterial(const ValueDescriptor &input, StableId valueId, StableId materialId,
                     const PortContract &contract, std::string &error) {
  if (input.carrier != CarrierKind::geometry3D || contract.carrier != input.carrier ||
      !validStableId(materialId) || input.operations.size() >= kMaximumOperations) {
    error = "Set Material requires bounded Geometry3D and a stable material identity";
    return std::nullopt;
  }
  auto result = input;
  result.stableId = valueId; result.overflow = contract.overflow;
  result.operations.push_back({valueId,OperationCode::assignMaterial,input.stableId,valueId,{},materialId});
  result.dispatchCount = result.operations.size();
  if (!admitValue(result,contract,error)) return std::nullopt;
  return result;
}

// Bindings on either input are kept: the joined value carries both spectrum and score
// lists, a binding shared by the two branches once, and the helper's per-frame replay
// joins each frame's deformed and displaced meshes.
inline std::optional<ValueDescriptor>
lowerJoinGeometry(const ValueDescriptor &a, const ValueDescriptor &b, StableId valueId,
                   const PortContract &contract, std::string &error) {
  if (a.carrier != CarrierKind::geometry3D || b.carrier != a.carrier || contract.carrier != a.carrier ||
      !layoutChangeFollowsBindings(a) || !layoutChangeFollowsBindings(b)) {
    error = "Join Geometry requires Geometry3D inputs with bindings it can follow";
    return std::nullopt;
  }
  auto result = a;
  if (!appendUniqueOperations(result.operations,b.operations,error)) return std::nullopt;
  for (const auto& binding : b.spectrumFields) {
    const auto shared=std::find_if(result.spectrumFields.begin(),result.spectrumFields.end(),[&](const auto& candidate) {
      return candidate.fieldStableId==binding.fieldStableId;
    });
    if (shared==result.spectrumFields.end()) result.spectrumFields.push_back(binding);
    else if (!spectrum::sameBinding(*shared,binding)) {
      error = "Join Geometry branches disagree on a shared Audio Deformer"; return std::nullopt;
    }
  }
  const auto scoreControls=[](const scorefield::Binding& v) {
    return std::tie(v.fieldStableId,v.sourceGeometryStableId,v.consumerGeometryStableId,v.scoreSourceStableId,
        v.mode,v.mapping.x,v.mapping.y,v.mapping.z,v.mapping.xScale,v.mapping.yScale,v.mapping.zScale,
        v.radius,v.gain,v.bias,v.direction,v.primes,v.positions,v.elementIds);
  };
  for (const auto& binding : b.scoreFields) {
    const auto shared=std::find_if(result.scoreFields.begin(),result.scoreFields.end(),[&](const auto& candidate) {
      return candidate.fieldStableId==binding.fieldStableId;
    });
    if (shared==result.scoreFields.end()) result.scoreFields.push_back(binding);
    else if (scoreControls(*shared)!=scoreControls(binding)) {
      error = "Join Geometry branches disagree on a shared Score Field"; return std::nullopt;
    }
  }
  if (result.operations.size() >= kMaximumOperations || a.attributes.size() != b.attributes.size()) {
    error = "Join Geometry requires matching attributes and an available operation budget";
    return std::nullopt;
  }
  for (auto &attribute : result.attributes) {
    const auto other = std::find_if(b.attributes.begin(),b.attributes.end(),[&](const auto &candidate) {
      return candidate.descriptor == attribute.descriptor;
    });
    if (other == b.attributes.end() ||
        (attribute.descriptor.domain != Domain::vertex && attribute.descriptor.domain != Domain::face &&
         attribute.descriptor.domain != Domain::faceCorner) ||
        other->elements.size() > kMaximumFieldElements - attribute.elements.size()) {
      error = "Join Geometry requires matching vertex, face or corner attributes";
      return std::nullopt;
    }
    attribute.elements.insert(attribute.elements.end(),other->elements.begin(),other->elements.end());
  }
  GeometryData mesh;
  if (!materializeJoin(std::get<GeometryData>(a.data),std::get<GeometryData>(b.data),a.stableId,b.stableId,
                        mesh,contract.maxVertices,contract.maxIndices,error)) return std::nullopt;
  result.data = std::move(mesh); result.stableId = valueId; result.overflow = contract.overflow;
  result.operations.push_back({valueId,OperationCode::joinGeometry,a.stableId,valueId,{},b.stableId});
  result.dispatchCount = result.operations.size();
  if (!admitValue(result,contract,error)) return std::nullopt;
  return result;
}

// An input moved by Audio Deformer bindings keeps them, and the helper's per-frame replay
// boxes each frame's deformed mesh. The box is always eight vertices.
inline std::optional<ValueDescriptor>
lowerBoundingBox(const ValueDescriptor &input, StableId valueId,
                  const PortContract &contract, std::string &error) {
  if (input.carrier != CarrierKind::geometry3D || contract.carrier != input.carrier ||
      !layoutChangeFollowsBindings(input) || contract.maxVertices < 8 || contract.maxIndices < 36 ||
      input.operations.size() >= kMaximumOperations) {
    error = "Bounding Box requires Geometry3D, bindings it can follow, and capacity for eight vertices";
    return std::nullopt;
  }
  auto result = input;
  GeometryData mesh;
  if (!materializeBoundingBox(std::get<GeometryData>(input.data),mesh,error)) return std::nullopt;
  result.data = std::move(mesh); result.stableId = valueId; result.overflow = contract.overflow;
  result.attributes.clear();
  result.operations.push_back({valueId,OperationCode::boundingBox,input.stableId,valueId,{}});
  result.dispatchCount = result.operations.size();
  if (!admitValue(result,contract,error)) return std::nullopt;
  return result;
}

// Absent levels leave the input value untouched, so its bytes do not change. A value
// lowered for preview holds the viewport mesh and for final render the render mesh.
inline std::optional<ValueDescriptor>
lowerSubdivisionSurface(const ValueDescriptor &input, StableId valueId, std::uint32_t viewportLevel,
                        std::uint32_t renderLevel, std::vector<SubdivisionCrease> creases, bool finalRender,
                        const PortContract &contract, std::string &error, float dicingRatePixels = 0.0f) {
  if (input.carrier != CarrierKind::geometry3D || contract.carrier != input.carrier
      || input.operations.size() >= kMaximumOperations) {
    error = "Subdivision Surface requires Geometry3D";
    return std::nullopt;
  }
  if (!(dicingRatePixels >= 0.0f && dicingRatePixels <= kMaximumSubdivisionDicingRate)) {
    error = "Subdivision Surface dicing rate " + std::to_string(dicingRatePixels) + " px is outside 0 to "
        + std::to_string(kMaximumSubdivisionDicingRate);
    return std::nullopt;
  }
  if (viewportLevel == 0 && renderLevel == 0 && dicingRatePixels == 0.0f) { error.clear(); return input; }
  std::sort(creases.begin(),creases.end(),[](const SubdivisionCrease &a, const SubdivisionCrease &b) noexcept {
    return std::make_pair(a.vertexA,a.vertexB) < std::make_pair(b.vertexA,b.vertexB);
  });
  if (!validSubdivisionCreases(creases)) {
    error = "Subdivision Surface creases need two different vertex ids, one entry per edge and a weight from 0 to 1";
    return std::nullopt;
  }
  auto record = std::make_shared<SubdivisionSurfaceRecord>();
  record->creases = std::move(creases);
  auto result = input;
  GeometryData mesh;
  std::vector<AttributeData> attributes;
  if (!materializeSubdivisionSurface(std::get<GeometryData>(input.data),input.attributes,record->creases,
          finalRender ? renderLevel : viewportLevel,mesh,attributes,error)) return std::nullopt;
  result.data = std::move(mesh); result.attributes = std::move(attributes);
  result.stableId = valueId; result.overflow = contract.overflow;
  OperationRecord operation{valueId,OperationCode::subdivisionSurface,input.stableId,valueId,
      {static_cast<float>(viewportLevel),static_cast<float>(renderLevel),finalRender ? 1.0f : 0.0f,dicingRatePixels}};
  operation.subdivision = std::move(record);
  result.operations.push_back(std::move(operation));
  result.dispatchCount = result.operations.size();
  if (!admitValue(result,contract,error)) return std::nullopt;
  return result;
}

inline bool lowerSetMaterial(ValueDescriptor &geometry, StableId materialId,
                             std::string &error) {
  if ((geometry.carrier != CarrierKind::geometry3D && geometry.carrier != CarrierKind::instances3D) ||
      !validStableId(materialId)) { error = "Set Material requires Geometry3D or Instances3D and a stable material identity"; return false; }
  if (geometry.operations.size() >= kMaximumOperations || geometry.dispatchCount >= kMaximumDispatches) {
    error = "Set Material exceeds the immutable operation budget"; return false;
  }
  geometry.operations.push_back({materialId, OperationCode::setMaterial, geometry.stableId,
                                 geometry.stableId, {static_cast<float>(materialId), 0, 0, 0}});
  ++geometry.dispatchCount; error.clear(); return true;
}

template <CarrierKind Kind> class TypedCarrier final {
public:
  static constexpr CarrierKind kind = Kind;
  static std::optional<TypedCarrier> from(AdmittedValue v, std::string &e) {
    if (v.descriptor().carrier != Kind) {
      e = "admitted geometry value has the wrong typed carrier";
      return std::nullopt;
    }
    e.clear();
    return TypedCarrier(std::move(v));
  }
  const AdmittedValue &value() const noexcept { return value_; }

private:
  explicit TypedCarrier(AdmittedValue v) : value_(std::move(v)) {}
  const AdmittedValue value_;
};
using Geometry3D = TypedCarrier<CarrierKind::geometry3D>;
using Points3D = TypedCarrier<CarrierKind::points3D>;
using Curves3D = TypedCarrier<CarrierKind::curves3D>;
using Instances3D = TypedCarrier<CarrierKind::instances3D>;
template <ValueType Type> class Field final {
public:
  static constexpr ValueType valueType = Type;
  static std::optional<Field> from(AdmittedValue v, std::string &e) {
    if (v.descriptor().carrier != CarrierKind::field ||
        v.descriptor().fieldValueType != Type) {
      e = "admitted field has the wrong typed value contract";
      return std::nullopt;
    }
    e.clear();
    return Field(std::move(v));
  }
  const AdmittedValue &value() const noexcept { return value_; }

private:
  explicit Field(AdmittedValue v) : value_(std::move(v)) {}
  const AdmittedValue value_;
};
struct RenderContract final {
  static constexpr bool kAllowsCpuMeshExpansion = false;
  static constexpr bool kAllowsCpuImageFallback = false;
  static constexpr bool kRequiresNativeGpuExecution = true;
};
} // namespace videowire::geometry
#include "GeometryFieldLowering.h"
