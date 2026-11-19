#pragma once

// Geometry Core port contracts: schema limits, carrier and attribute enums, wire names,
// and contract validation. Graph headers include this instead of GeometryCore.h.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace videowire::geometry {
inline constexpr std::uint32_t kSchemaVersion = 2;
inline constexpr std::uint64_t kMaximumStableId = 9007199254740991ull;
// These are format bounds, not scene policy: elements address each other through
// 32-bit indices and byte sizes must survive checked arithmetic. A scene below
// them allocates what it needs, and only a real allocation failure stops it.
inline constexpr std::size_t kMaximumVertices = 0xffffffffu;
inline constexpr std::size_t kMaximumIndices = std::size_t{1} << 62u;
inline constexpr std::size_t kMaximumPoints = 0xffffffffu;
inline constexpr std::size_t kMaximumCurvePoints = 0xffffffffu;
inline constexpr std::size_t kMaximumSplines = 0xffffffffu;
inline constexpr std::size_t kMaximumInstances = 0xffffffffu;
inline constexpr std::size_t kMaximumFieldElements = std::size_t{1} << 62u;
inline constexpr std::size_t kMaximumAttributes = 0xffffffffu;
inline constexpr std::size_t kMaximumOperations = 0xffffffffu;
inline constexpr std::size_t kMaximumDispatches = 0xffffffffu;
inline constexpr std::size_t kMaximumBufferBytes = std::size_t{1} << 62u;
inline constexpr std::size_t kMaximumNameBytes = 255;

using StableId = std::uint64_t;

using RetainedResourceOwner = std::shared_ptr<const void>;

// Payload disposal precedes the last dependency release, including aliases
// extracted from a descriptor whose other owners have already retired.
template <typename T> struct RetainedResourcePayload final {
  const RetainedResourceOwner resources;
  T value;
  template <typename... Args>
  RetainedResourcePayload(RetainedResourceOwner owner, Args&&... args)
      : resources(std::move(owner)), value(std::forward<Args>(args)...) {}
};
template <typename T, typename... Args>
inline std::shared_ptr<T> makeRetainedResourcePayload(
    const RetainedResourceOwner& resources, Args&&... args) {
  if (!resources) return std::make_shared<T>(std::forward<Args>(args)...);
  auto holder=std::make_shared<RetainedResourcePayload<T>>(resources,std::forward<Args>(args)...);
  return std::shared_ptr<T>(holder,&holder->value);
}

enum class CarrierKind : std::uint8_t {
  unspecified = 0,
  geometry3D,
  points3D,
  curves3D,
  instances3D,
  field
};
enum class Domain : std::uint8_t {
  unspecified = 0,
  point,
  vertex,
  faceCorner,
  face,
  curvePoint,
  spline,
  instance
};
enum class ValueType : std::uint8_t {
  unspecified = 0,
  floatValue,
  vector,
  color,
  boolean,
  integer,
  rotation
};
enum class Interpolation : std::uint8_t {
  unspecified = 0,
  constant,
  nearest,
  linear,
  normalizedLinear
};
enum class OverflowPolicy : std::uint8_t {
  unspecified = 0,
  reject,
  keepLowestStableIds
};
enum class OperationCode : std::uint8_t {
  copy = 1,
  transform,
  evaluateField,
  prepareInstances,
  grid,
  curveCircle,
  curveToMesh,
  instanceOnPoints,
  setMaterial,
  pointsFromVertices,
  plane,
  cube,
  sphere,
  icosphere,
  cylinder,
  curveLine,
  positionField,
  normalField,
  indexField,
  stableIdentityField,
  constantVectorField,
  vectorDisplacement,
  spectrumInstancer,
  transformTRS,
  assignMaterial,
  joinGeometry,
  boundingBox,
  boundsMinimumField,
  boundsMaximumField,
  boundsSizeField,
  boundsCenterField,
  audioDeformer,
  curveResample,
  curveTrim,
  curveSetRadius,
  curveSetTilt,
  curveWave,
  curveToTube,
  pointsFromCurves,
  curveTangentField,
  curveParameterField,
  curveRadiusField,
  curveTiltField,
  curveEndpointField,
  pointsOnFaces = 64,
  pointsInBoxVolume,
  transformInstances,
  cullInstances,
  fieldConstant = 80,
  fieldMath,
  fieldVectorMath,
  fieldVectorMeasure,
  fieldCompare,
  fieldBoolean,
  fieldMix,
  fieldMapRange,
  fieldClamp,
  fieldNoise,
  fieldGradient,
  fieldRandom,
  fieldFalloff,
  fieldComposeVector,
  fieldIntegerToFloat,
  fieldSelect,
  fieldProximity,
  fieldNearestDirection,
  fieldNearestIndex,
  fieldSampleNearest,
  storeNamedAttribute,
  readNamedAttribute,
  setNormal,
  retainedMesh = 112,
  setInstanceAppearance = 120,
  fieldSurfaceDistance,
  fieldSurfaceOffset,
  fieldRaycastDistance,
  fieldRaycastPosition,
  fieldRaycastHit,
  fieldRaycastFace,
  materialTableSlot,
  setMaterialIndex,
  fieldFaceIndex,
  fieldFloatToInteger,
  fieldTimeline,
  fieldScoreSample,
  fieldAudioSpectrum,
  sdfSurface,
  fieldWaterHeight = 136,
  fieldSampleIndex,
  cullGeometry,
  cullPoints,
  pointsOnFacesDensity,
  shadeMesh,
  subdivisionSurface,
  // Time-driven curve edit: parameters are mode (0 grow, 1 travel, 2 noise),
  // loop period in seconds, amount (travel window or noise amplitude) and
  // phase (grow and travel) or noise cycles per spline.
  curveAnimate
};

inline const char *toWire(CarrierKind v) noexcept {
  switch (v) {
  case CarrierKind::geometry3D:
    return "geometry3D";
  case CarrierKind::points3D:
    return "points3D";
  case CarrierKind::curves3D:
    return "curves3D";
  case CarrierKind::instances3D:
    return "instances3D";
  case CarrierKind::field:
    return "field";
  default:
    return "unspecified";
  }
}
inline const char *toWire(Domain v) noexcept {
  switch (v) {
  case Domain::point:
    return "point";
  case Domain::vertex:
    return "vertex";
  case Domain::faceCorner:
    return "faceCorner";
  case Domain::face:
    return "face";
  case Domain::curvePoint:
    return "curvePoint";
  case Domain::spline:
    return "spline";
  case Domain::instance:
    return "instance";
  default:
    return "unspecified";
  }
}
inline const char *toWire(ValueType v) noexcept {
  switch (v) {
  case ValueType::floatValue:
    return "float";
  case ValueType::vector:
    return "vector";
  case ValueType::color:
    return "color";
  case ValueType::boolean:
    return "bool";
  case ValueType::integer:
    return "integer";
  case ValueType::rotation:
    return "rotation";
  default:
    return "unspecified";
  }
}
inline const char *toWire(Interpolation v) noexcept {
  switch (v) {
  case Interpolation::constant:
    return "constant";
  case Interpolation::nearest:
    return "nearest";
  case Interpolation::linear:
    return "linear";
  case Interpolation::normalizedLinear:
    return "normalizedLinear";
  default:
    return "unspecified";
  }
}
inline const char *toWire(OverflowPolicy v) noexcept {
  switch (v) {
  case OverflowPolicy::reject:
    return "reject";
  case OverflowPolicy::keepLowestStableIds:
    return "keepLowestStableIds";
  default:
    return "unspecified";
  }
}
inline CarrierKind carrierKindFromWire(const std::string &v) noexcept {
  if (v == "geometry3D")
    return CarrierKind::geometry3D;
  if (v == "points3D")
    return CarrierKind::points3D;
  if (v == "curves3D")
    return CarrierKind::curves3D;
  if (v == "instances3D")
    return CarrierKind::instances3D;
  if (v == "field")
    return CarrierKind::field;
  return CarrierKind::unspecified;
}
inline Domain domainFromWire(const std::string &v) noexcept {
  if (v == "point")
    return Domain::point;
  if (v == "vertex")
    return Domain::vertex;
  if (v == "faceCorner")
    return Domain::faceCorner;
  if (v == "face")
    return Domain::face;
  if (v == "curvePoint")
    return Domain::curvePoint;
  if (v == "spline")
    return Domain::spline;
  if (v == "instance")
    return Domain::instance;
  return Domain::unspecified;
}
inline ValueType valueTypeFromWire(const std::string &v) noexcept {
  if (v == "float")
    return ValueType::floatValue;
  if (v == "vector")
    return ValueType::vector;
  if (v == "color")
    return ValueType::color;
  if (v == "bool")
    return ValueType::boolean;
  if (v == "integer")
    return ValueType::integer;
  if (v == "rotation")
    return ValueType::rotation;
  return ValueType::unspecified;
}
inline Interpolation interpolationFromWire(const std::string &v) noexcept {
  if (v == "constant")
    return Interpolation::constant;
  if (v == "nearest")
    return Interpolation::nearest;
  if (v == "linear")
    return Interpolation::linear;
  if (v == "normalizedLinear")
    return Interpolation::normalizedLinear;
  return Interpolation::unspecified;
}
inline OverflowPolicy overflowPolicyFromWire(const std::string &v) noexcept {
  if (v == "reject")
    return OverflowPolicy::reject;
  if (v == "keepLowestStableIds")
    return OverflowPolicy::keepLowestStableIds;
  return OverflowPolicy::unspecified;
}

struct Vec3 final {
  float x = 0, y = 0, z = 0;
};
struct Quaternion final {
  float x = 0, y = 0, z = 0, w = 1;
};
struct Transform final {
  Vec3 translation{};
  Quaternion rotation{};
  Vec3 scale{1, 1, 1};
};

struct AttributeDescriptor final {
  StableId stableId = 0;
  std::string name;
  ValueType valueType = ValueType::unspecified;
  Domain domain = Domain::unspecified;
  Interpolation interpolation = Interpolation::unspecified;
};
inline bool operator==(const AttributeDescriptor &a,
                       const AttributeDescriptor &b) noexcept {
  return a.stableId == b.stableId && a.name == b.name &&
         a.valueType == b.valueType && a.domain == b.domain &&
         a.interpolation == b.interpolation;
}

struct PortContract final {
  std::uint32_t schemaVersion = kSchemaVersion;
  CarrierKind carrier = CarrierKind::unspecified;
  Domain fieldDomain = Domain::unspecified;
  ValueType fieldValueType = ValueType::unspecified;
  Interpolation fieldInterpolation = Interpolation::unspecified;
  OverflowPolicy overflow = OverflowPolicy::reject;
  std::size_t maxVertices = 0, maxIndices = 0, maxPoints = 0,
              maxCurvePoints = 0, maxSplines = 0, maxInstances = 0,
              maxFieldElements = 0, maxAttributes = 0;
  std::vector<AttributeDescriptor> attributes;
};
inline bool operator==(const PortContract &a, const PortContract &b) noexcept {
  return a.schemaVersion == b.schemaVersion && a.carrier == b.carrier &&
         a.fieldDomain == b.fieldDomain &&
         a.fieldValueType == b.fieldValueType &&
         a.fieldInterpolation == b.fieldInterpolation &&
         a.overflow == b.overflow && a.maxVertices == b.maxVertices &&
         a.maxIndices == b.maxIndices && a.maxPoints == b.maxPoints &&
         a.maxCurvePoints == b.maxCurvePoints && a.maxSplines == b.maxSplines &&
         a.maxInstances == b.maxInstances &&
         a.maxFieldElements == b.maxFieldElements &&
         a.maxAttributes == b.maxAttributes && a.attributes == b.attributes;
}

inline bool validStableId(StableId v) noexcept {
  return v != 0 && v <= kMaximumStableId;
}
inline bool domainAllowed(CarrierKind c, Domain d) noexcept {
  switch (c) {
  case CarrierKind::geometry3D:
    return d == Domain::vertex || d == Domain::faceCorner || d == Domain::face;
  case CarrierKind::points3D:
    return d == Domain::point;
  case CarrierKind::curves3D:
    return d == Domain::curvePoint || d == Domain::spline;
  case CarrierKind::instances3D:
    return d == Domain::instance;
  case CarrierKind::field:
    return d != Domain::unspecified;
  default:
    return false;
  }
}
inline bool validatePortContract(const PortContract &c, std::string &e);
inline bool migratePortContractV1ToV2(PortContract &c, std::string &e) {
  e.clear();
  if (c.schemaVersion != 1) {
    e = "Geometry Core migration requires a schema-v1 port contract";
    return false;
  }
  if (c.carrier == CarrierKind::unspecified) {
    e = "Geometry Core schema-v1 migration has no carrier identity";
    return false;
  }
  // Schema v1 wrote every graph capacity into every carrier. Schema v2 keeps
  // only the capacities that the carrier can consume. Preserve those values
  // exactly and clear the legacy all-capacity fields.
  switch (c.carrier) {
  case CarrierKind::geometry3D:
    c.maxPoints = c.maxCurvePoints = c.maxSplines = c.maxInstances = 0;
    c.maxFieldElements = 0;
    break;
  case CarrierKind::points3D:
    c.maxVertices = c.maxIndices = c.maxCurvePoints = c.maxSplines = 0;
    c.maxInstances = c.maxFieldElements = 0;
    break;
  case CarrierKind::curves3D:
    c.maxVertices = c.maxIndices = c.maxPoints = c.maxInstances = 0;
    c.maxFieldElements = 0;
    break;
  case CarrierKind::instances3D:
    c.maxVertices = c.maxIndices = c.maxPoints = c.maxCurvePoints = 0;
    c.maxSplines = c.maxFieldElements = 0;
    break;
  case CarrierKind::field:
    c.maxVertices = c.maxIndices = c.maxPoints = c.maxCurvePoints = 0;
    c.maxSplines = c.maxInstances = 0;
    break;
  default:
    e = "Geometry Core schema-v1 migration has an unsupported carrier";
    return false;
  }
  c.schemaVersion = kSchemaVersion;
  if (!validatePortContract(c, e)) {
    e = "Geometry Core schema-v1 migration produced an invalid schema-v2 contract: " + e;
    return false;
  }
  return true;
}
inline bool validatePortContract(const PortContract &c, std::string &e) {
  e.clear();
  if (c.schemaVersion != kSchemaVersion ||
      c.carrier == CarrierKind::unspecified ||
      c.overflow == OverflowPolicy::unspecified) {
    e = "geometry port contract header is invalid";
    return false;
  }
  const bool field = c.carrier == CarrierKind::field;
  if (field != (c.fieldDomain != Domain::unspecified &&
                c.fieldValueType != ValueType::unspecified &&
                c.fieldInterpolation != Interpolation::unspecified &&
                c.maxFieldElements != 0) ||
      (!field && c.maxFieldElements != 0)) {
    e = "geometry field contract is incomplete or attached to a non-field "
        "carrier";
    return false;
  }
  bool shape = false;
  switch (c.carrier) {
  case CarrierKind::geometry3D:
    shape = c.maxVertices && c.maxIndices && !c.maxPoints &&
            !c.maxCurvePoints && !c.maxSplines && !c.maxInstances;
    break;
  case CarrierKind::points3D:
    shape = c.maxPoints && !c.maxVertices && !c.maxIndices &&
            !c.maxCurvePoints && !c.maxSplines && !c.maxInstances;
    break;
  case CarrierKind::curves3D:
    shape = c.maxCurvePoints && c.maxSplines && !c.maxVertices &&
            !c.maxIndices && !c.maxPoints && !c.maxInstances;
    break;
  case CarrierKind::instances3D:
    shape = c.maxInstances && !c.maxVertices && !c.maxIndices && !c.maxPoints &&
            !c.maxCurvePoints && !c.maxSplines;
    break;
  case CarrierKind::field:
    shape = !c.maxVertices && !c.maxIndices && !c.maxPoints &&
            !c.maxCurvePoints && !c.maxSplines && !c.maxInstances;
    break;
  default:
    break;
  }
  if (!shape) {
    e = "geometry port contract has capacities for the wrong carrier";
    return false;
  }
  if (c.maxVertices > kMaximumVertices || c.maxIndices > kMaximumIndices ||
      c.maxPoints > kMaximumPoints || c.maxCurvePoints > kMaximumCurvePoints ||
      c.maxSplines > kMaximumSplines || c.maxInstances > kMaximumInstances ||
      c.maxFieldElements > kMaximumFieldElements ||
      c.maxAttributes > kMaximumAttributes ||
      c.attributes.size() > c.maxAttributes) {
    e = "geometry port capacity exceeds hard bounds";
    return false;
  }
  std::set<StableId> ids;
  std::set<std::string> names;
  for (const auto &a : c.attributes)
    if (!validStableId(a.stableId) || a.name.empty() ||
        a.name.size() > kMaximumNameBytes ||
        a.valueType == ValueType::unspecified ||
        !domainAllowed(c.carrier, a.domain) ||
        a.interpolation == Interpolation::unspecified ||
        !ids.insert(a.stableId).second || !names.insert(a.name).second) {
      e = "geometry port attribute schema is malformed, foreign, or ambiguous";
      return false;
    }
  return true;
}
} // namespace videowire::geometry
