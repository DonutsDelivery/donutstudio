#pragma once

#include "GeometryCore.h"
#include "ExactTokenMemo.h"

#include <cstring>
#include <type_traits>

namespace videowire::geometry {
inline constexpr std::array<std::uint8_t, 4> kTransportMagic{
    {'G', 'C', 'R', 'V'}};
inline constexpr std::uint16_t kTransportVersion = 21;
// Version 21 leaves out the carrier data and attributes of a value whose
// operations rebuild them exactly. The receiver replays the operations instead.
inline constexpr std::uint16_t kReplayedValueTransportVersion = 21;
// The Audio Effector delay in beats was introduced with version 20.
inline constexpr std::uint16_t kEffectorBeatDelayTransportVersion = 20;
// Shade Mesh (operation 141) was introduced with version 19.
inline constexpr std::uint16_t kShadeTransportVersion = 19;
inline constexpr std::uint16_t kSubdivisionTransportVersion = 19;
// Curve Animate (operation 143) was introduced with version 19 and no bump.
inline constexpr std::uint16_t kCurveAnimateTransportVersion = 19;// Distribute Points on Faces by density (operation 140) was introduced with version 18.
inline constexpr std::uint16_t kPointDensityTransportVersion = 18;
// Cull Geometry and Cull Points (operations 138 and 139) were introduced with version 17.
inline constexpr std::uint16_t kCullTransportVersion = 17;
// The Set Normal normalize flag (parameters[0] set to 1, preserve length) was introduced with version 17.
inline constexpr std::uint16_t kSetNormalFlagTransportVersion = 17;
// The Audio Effector input byte was introduced with version 16.
inline constexpr std::uint16_t kEffectorInputTransportVersion = 16;
inline constexpr std::uint16_t kBodyCouplingTransportVersion = 15;
// fieldSampleIndex (operation 137) was introduced with version 15 and no bump.
inline constexpr std::uint16_t kSampleIndexTransportVersion = 15;
inline constexpr std::uint16_t kTextTransportVersion = 14;
inline constexpr std::uint16_t kNonTextTransportVersion = 13;

namespace transport_detail {
class Writer final {
public:
  template <class T> void integer(T value) {
    using U = std::make_unsigned_t<T>;
    auto bits = static_cast<U>(value);
    for (std::size_t i = 0; i < sizeof(T); ++i)
      bytes.push_back(static_cast<std::uint8_t>((bits >> (i * 8)) & 0xffu));
  }
  void floating(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    integer(bits);
  }
  void floating(double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    integer(bits);
  }
  void string(const std::string &value) {
    integer<std::uint16_t>(static_cast<std::uint16_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
  }
  std::vector<std::uint8_t> bytes;
};

class Reader final {
public:
  Reader(const std::uint8_t *source, std::size_t size)
      : cursor(source), remaining(size) {}
  template <class T> bool integer(T &value) {
    if (remaining < sizeof(T))
      return fail("truncated Geometry Core transport");
    using U = std::make_unsigned_t<T>;
    U bits = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i)
      bits |= static_cast<U>(cursor[i]) << (i * 8);
    cursor += sizeof(T);
    remaining -= sizeof(T);
    value = static_cast<T>(bits);
    return true;
  }
  bool floating(float &value) {
    std::uint32_t bits = 0;
    if (!integer(bits))
      return false;
    std::memcpy(&value, &bits, sizeof(value));
    return true;
  }
  bool floating(double &value) {
    std::uint64_t bits = 0;
    if (!integer(bits))
      return false;
    std::memcpy(&value, &bits, sizeof(value));
    return true;
  }
  bool string(std::string &value, bool allowEmpty = false, std::size_t maximum = kMaximumNameBytes) {
    std::uint16_t count = 0;
    if (!integer(count))
      return false;
    if ((!allowEmpty && count == 0) || count > maximum || remaining < count)
      return fail("invalid or truncated Geometry Core string");
    value.assign(reinterpret_cast<const char *>(cursor), count);
    cursor += count;
    remaining -= count;
    return true;
  }
  bool count(std::size_t &value, std::size_t maximum) {
    std::uint64_t wire = 0;
    if (!integer(wire))
      return false;
    if (wire > maximum || wire > std::numeric_limits<std::size_t>::max())
      return fail("Geometry Core count exceeds its exact bound");
    value = static_cast<std::size_t>(wire);
    return true;
  }
  bool arrayBytes(std::size_t count, std::size_t elementBytes) {
    if (elementBytes == 0 || count > remaining / elementBytes)
      return fail("truncated Geometry Core array payload");
    return true;
  }
  bool fail(const char *message) {
    if (error.empty())
      error = message;
    return false;
  }
  const std::uint8_t *cursor;
  std::size_t remaining;
  std::string error;
};

inline void writeVec3(Writer &w, const Vec3 &v) {
  w.floating(v.x);
  w.floating(v.y);
  w.floating(v.z);
}
inline bool readVec3(Reader &r, Vec3 &v) {
  return r.floating(v.x) && r.floating(v.y) && r.floating(v.z);
}
inline void writeQuaternion(Writer &w, const Quaternion &v) {
  w.floating(v.x);
  w.floating(v.y);
  w.floating(v.z);
  w.floating(v.w);
}
inline bool readQuaternion(Reader &r, Quaternion &v) {
  return r.floating(v.x) && r.floating(v.y) && r.floating(v.z) &&
         r.floating(v.w);
}
inline void writeFieldElement(Writer &w, const FieldElement &v) {
  for (auto x : v.components)
    w.floating(x);
}
inline bool readFieldElement(Reader &r, FieldElement &v) {
  for (auto &x : v.components)
    if (!r.floating(x))
      return false;
  return true;
}
inline void writeAttributeDescriptor(Writer &w, const AttributeDescriptor &a) {
  w.integer(a.stableId);
  w.string(a.name);
  w.integer(static_cast<std::uint8_t>(a.valueType));
  w.integer(static_cast<std::uint8_t>(a.domain));
  w.integer(static_cast<std::uint8_t>(a.interpolation));
}
inline bool readAttributeDescriptor(Reader &r, AttributeDescriptor &a) {
  std::uint8_t t = 0, d = 0, i = 0;
  if (!r.integer(a.stableId) || !r.string(a.name) || !r.integer(t) ||
      !r.integer(d) || !r.integer(i))
    return false;
  if (t < 1 || t > static_cast<std::uint8_t>(ValueType::rotation) || d < 1 ||
      d > static_cast<std::uint8_t>(Domain::instance) || i < 1 ||
      i > static_cast<std::uint8_t>(Interpolation::normalizedLinear))
    return r.fail("unknown Geometry Core attribute enum");
  a.valueType = static_cast<ValueType>(t);
  a.domain = static_cast<Domain>(d);
  a.interpolation = static_cast<Interpolation>(i);
  return true;
}
inline void writeCarrierData(Writer &w, CarrierKind carrier, const CarrierData &data) {
  switch (carrier) {
  case CarrierKind::geometry3D: {
    const auto &p = std::get<GeometryData>(data);
    w.integer<std::uint64_t>(p.positions.size());
    w.integer<std::uint64_t>(p.indices.size());
    for (std::size_t i = 0; i < p.positions.size(); ++i) {
      writeVec3(w, p.positions[i]);
      w.integer(p.vertexIds[i]);
    }
    for (auto x : p.indices)
      w.integer(x);
    break;
  }
  case CarrierKind::points3D: {
    const auto &p = std::get<PointsData>(data);
    w.integer<std::uint64_t>(p.points.size());
    for (const auto &x : p.points) {
      writeVec3(w, x.position);
      w.integer(x.stableId);
      w.floating(x.radius);
      writeQuaternion(w, x.rotation);
    }
    break;
  }
  case CarrierKind::curves3D: {
    const auto &p = std::get<CurvesData>(data);
    w.integer<std::uint64_t>(p.points.size());
    w.integer<std::uint64_t>(p.splines.size());
    for (const auto &x : p.points) {
      writeVec3(w, x.position);
      w.integer(x.stableId);
      w.floating(x.radius);
      w.floating(x.tilt);
    }
    for (const auto &s : p.splines) {
      w.integer(s.stableId);
      w.integer(s.firstPoint);
      w.integer(s.pointCount);
      w.integer<std::uint8_t>(s.cyclic ? 1 : 0);
    }
    break;
  }
  case CarrierKind::instances3D: {
    const auto &p = std::get<InstancesData>(data);
    w.integer<std::uint64_t>(p.instances.size());
    for (const auto &x : p.instances) {
      w.integer(x.stableId);
      w.integer(x.sourceStableId);
      writeVec3(w, x.transform.translation);
      writeQuaternion(w, x.transform.rotation);
      writeVec3(w, x.transform.scale);
    }
    break;
  }
  case CarrierKind::field: {
    const auto &p = std::get<FieldData>(data);
    w.integer<std::uint64_t>(p.elements.size());
    for (const auto &x : p.elements)
      writeFieldElement(w, x);
    break;
  }
  default:
    break;
  }
}
inline void writeAttributes(Writer &w, const std::vector<AttributeData> &attributes) {
  for (const auto &a : attributes) {
    writeAttributeDescriptor(w, a.descriptor);
    w.integer<std::uint64_t>(a.elements.size());
    for (const auto &x : a.elements)
      writeFieldElement(w, x);
  }
}
// True when replaying the operations with no admission context rebuilds the
// stored data and attributes bit for bit. Every source must be procedural: a
// retained mesh carries imported bytes, and a value that reads deferred or
// external inputs does not replay without its context. Field carriers keep
// their elements on the wire because their plan check compares stored elements.
inline bool replaysExactly(const ValueDescriptor &v) {
  if (v.carrier == CarrierKind::field || v.operations.empty()) return false;
  if (std::any_of(v.operations.begin(), v.operations.end(), [](const OperationRecord &op) {
        return op.code == OperationCode::retainedMesh || op.retainedMesh;
      }))
    return false;
  ValueDescriptor replayed;
  std::string error;
  if (!validateOperationPlan(v, AdmissionContext{}, nullptr, error, &replayed))
    return false;
  Writer stored, rebuilt;
  writeAttributes(stored, v.attributes);
  writeCarrierData(stored, v.carrier, v.data);
  writeAttributes(rebuilt, replayed.attributes);
  writeCarrierData(rebuilt, replayed.carrier, replayed.data);
  return stored.bytes == rebuilt.bytes;
}
} // namespace transport_detail

namespace transport_detail {
inline std::vector<std::uint8_t> writeRuntimeValue(const ValueDescriptor &v, bool replayed) {
  const bool hasText=std::any_of(v.operations.begin(),v.operations.end(),[](const OperationRecord& op) {
    return op.sdfSurface && op.sdfSurface->geometry.schemaVersion==kSdfIrTextSchemaVersion;
  });
  const bool hasBodyCoupling=std::any_of(v.operations.begin(),v.operations.end(),[](const OperationRecord& op) {
    return op.code==OperationCode::fieldWaterHeight && op.field.values[4]!=0;
  });
  const bool hasSampleIndex=std::any_of(v.operations.begin(),v.operations.end(),[](const OperationRecord& op) {
    return op.code==OperationCode::fieldSampleIndex;
  });
  // Only graphs that choose an Audio Effector analyzer scalar need version 16.
  // Every other graph keeps the byte layout it had before.
  const bool hasEffectorInput=std::any_of(v.spectrumFields.begin(),v.spectrumFields.end(),[](const auto& binding) {
    return binding.input!=spectrum::EffectorInput::bandRange;
  });
  const bool hasCull=std::any_of(v.operations.begin(),v.operations.end(),[](const OperationRecord& op) {
    return op.code==OperationCode::cullGeometry || op.code==OperationCode::cullPoints;
  });
  // A Set Normal with its preserve-length flag set needs version 17. Flag 0 keeps the old bytes.
  const bool hasPreservedNormal=std::any_of(v.operations.begin(),v.operations.end(),[](const OperationRecord& op) {
    return op.code==OperationCode::setNormal && op.parameters[0]!=0;
  });
  const bool hasPointDensity=std::any_of(v.operations.begin(),v.operations.end(),[](const OperationRecord& op) {
    return op.code==OperationCode::pointsOnFacesDensity;
  });
  const bool hasShade=std::any_of(v.operations.begin(),v.operations.end(),[](const OperationRecord& op) {
    return op.code==OperationCode::shadeMesh;
  });
  const bool hasSubdivision=std::any_of(v.operations.begin(),v.operations.end(),[](const OperationRecord& op) {
    return op.code==OperationCode::subdivisionSurface;
  });
  // Only an Audio Effector delayed in beats needs version 20.
  const bool hasBeatDelay=std::any_of(v.spectrumFields.begin(),v.spectrumFields.end(),[](const auto& binding) {
    return binding.deformer.delayBeats!=0;
  });
  const bool hasCurveAnimate=std::any_of(v.operations.begin(),v.operations.end(),[](const OperationRecord& op) {
    return op.code==OperationCode::curveAnimate;
  });
  const auto version=replayed ? kReplayedValueTransportVersion
      : hasBeatDelay ? kEffectorBeatDelayTransportVersion
      : hasCurveAnimate ? kCurveAnimateTransportVersion
      : (hasShade||hasSubdivision) ? std::max(kShadeTransportVersion,kSubdivisionTransportVersion)      : hasPointDensity ? kPointDensityTransportVersion
      : hasCull ? kCullTransportVersion
      : hasPreservedNormal ? kSetNormalFlagTransportVersion
      : hasEffectorInput ? kEffectorInputTransportVersion
      : (hasBodyCoupling || hasSampleIndex) ? kSampleIndexTransportVersion
      : hasText ? kTextTransportVersion : kNonTextTransportVersion;
  Writer w;
  for (auto b : kTransportMagic)
    w.integer(b);
  w.integer(version);
  w.integer<std::uint16_t>(0);
  w.integer(v.schemaVersion);
  w.integer(static_cast<std::uint8_t>(v.carrier));
  w.integer(v.stableId);
  w.integer(v.sourceStableId);
  w.integer(v.sourceRevision);
  w.integer(static_cast<std::uint8_t>(v.overflow));
  w.integer(static_cast<std::uint8_t>(v.fieldDomain));
  w.integer(static_cast<std::uint8_t>(v.fieldValueType));
  w.integer(static_cast<std::uint8_t>(v.fieldInterpolation));
  w.integer<std::uint64_t>(v.fieldDomainCardinality);
  w.integer<std::uint64_t>(v.dispatchCount);
  w.integer<std::uint64_t>(replayed ? 0u : v.attributes.size());
  w.integer<std::uint64_t>(v.operations.size());
  if (!replayed)
    writeAttributes(w, v.attributes);
  for (const auto &op : v.operations) {
    w.integer(op.stableId);
    w.integer(static_cast<std::uint8_t>(op.code));
    w.integer(op.inputStableId);
    w.integer(op.outputStableId);
    w.integer(op.secondaryInputStableId);
    for (auto x : op.parameters)
      w.floating(x);
    if (op.code == OperationCode::transformTRS || op.code == OperationCode::transformInstances) {
      writeVec3(w,op.transform.translation);
      writeQuaternion(w,op.transform.rotation);
      writeVec3(w,op.transform.scale);
    }
    if (isConstructedFieldOperation(op.code)) {
      w.integer(static_cast<std::uint8_t>(op.field.domain));
      w.integer(static_cast<std::uint8_t>(op.field.valueType));
      w.integer(op.field.mode);
      w.integer(op.field.auxiliaryInputStableId);
      for (const auto value : op.field.values) w.floating(value);
      w.string(op.field.attributeName);
    }
    if (op.code==OperationCode::shadeMesh) {
      w.integer(static_cast<std::uint64_t>(op.sharpEdges.size()));
      for (const auto& edge : op.sharpEdges) { w.integer(edge[0]); w.integer(edge[1]); }
    }
    if(op.code==OperationCode::sdfSurface) {
      const auto& q=*op.sdfSurface;
      w.integer(q.algorithmVersion);
      for(auto x:q.minimum) w.floating(x);
      for(auto x:q.maximum) w.floating(x);
      for(auto x:q.cells) w.integer(x);
      w.floating(q.gradientStep); w.floating(q.maximumResidual);
      w.integer(q.geometry.schemaVersion); w.integer(q.geometry.rootId);
      w.integer<std::uint32_t>(static_cast<std::uint32_t>(q.geometry.records.size()));
      for(const auto& record:q.geometry.records) {
        w.integer(record.stableId); w.integer(static_cast<std::uint8_t>(record.operation));
        w.integer(record.inputCount); w.integer(record.parameterCount);
        for(auto input:record.inputs) w.integer(input);
        for(auto parameter:record.parameters) w.floating(parameter);
        if(version>=14) w.integer(record.resourceStableId);
      }
      if(version>=14) {
        w.integer<std::uint32_t>(static_cast<std::uint32_t>(q.geometry.textOutlines.size()));
        for(const auto& outline:q.geometry.textOutlines) {
          const auto payload=encodeSdfTextOutline(outline);
          w.integer<std::uint32_t>(static_cast<std::uint32_t>(payload.size()));
          w.bytes.insert(w.bytes.end(),payload.begin(),payload.end());
        }
      }
    }
    if (op.code==OperationCode::subdivisionSurface) {
      w.integer<std::uint64_t>(op.subdivision ? op.subdivision->creases.size() : 0u);
      if (op.subdivision) for (const auto& crease : op.subdivision->creases) {
        w.integer(crease.vertexA); w.integer(crease.vertexB); w.floating(crease.weight);
      }
    }
    if (op.code==OperationCode::retainedMesh) {
      const auto& source=*op.retainedMesh;
      w.integer<std::uint64_t>(source.geometry.positions.size());
      w.integer<std::uint64_t>(source.geometry.indices.size());
      w.integer<std::uint64_t>(source.attributes.size());
      for (std::size_t i=0;i<source.geometry.positions.size();++i) {
        writeVec3(w,source.geometry.positions[i]); w.integer(source.geometry.vertexIds[i]);
      }
      for (const auto index : source.geometry.indices) w.integer(index);
      for (const auto& attribute : source.attributes) {
        writeAttributeDescriptor(w,attribute.descriptor);
        w.integer<std::uint64_t>(attribute.elements.size());
        for (const auto& element : attribute.elements) writeFieldElement(w,element);
      }
      w.string(source.importedAnimation);
    }
  }
  if (!replayed)
    writeCarrierData(w, v.carrier, v.data);
  w.integer<std::uint32_t>(static_cast<std::uint32_t>(v.spectrumFields.size()));
  for (const auto& binding : v.spectrumFields) {
    w.integer(binding.fieldStableId);
    w.integer(binding.sourceGeometryStableId);
    w.integer(binding.consumerGeometryStableId);
    w.integer(binding.firstBand); w.integer(binding.lastBand);
    w.integer<std::uint8_t>(binding.interpolate ? 1 : 0);
    w.integer<std::uint8_t>(binding.normalize ? 1 : 0);
    w.floating(binding.gain); w.floating(binding.bias);
    w.floating(binding.attackSeconds); w.floating(binding.releaseSeconds);
    for (const auto value : binding.direction) w.floating(value);
    w.integer<std::uint64_t>(binding.coordinates.size());
    for (const auto value : binding.coordinates) w.floating(value);
    w.integer<std::uint8_t>(static_cast<std::uint8_t>(binding.target));
    for (const auto value : binding.baseScale) w.floating(value);
    for (const auto value : binding.scaleResponse) w.floating(value);
    const auto& d = binding.deformer;
    w.integer<std::uint8_t>(static_cast<std::uint8_t>(d.mode));
    for (const auto value : d.center) w.floating(value);
    for (const auto value : {d.frequency,d.speed,d.phase,d.falloffRadius,d.falloffPower,
         d.delaySeconds,d.threshold,d.hysteresis,d.peakHoldSeconds,d.quantizeSteps,d.envelopePower}) w.floating(value);
    if (version >= kEffectorInputTransportVersion) w.integer<std::uint8_t>(static_cast<std::uint8_t>(binding.input));
    if (version >= kEffectorBeatDelayTransportVersion) w.floating(d.delayBeats);
  }
  w.integer<std::uint32_t>(static_cast<std::uint32_t>(v.scoreFields.size()));
  for (const auto& binding : v.scoreFields) {
    w.integer(binding.fieldStableId); w.integer(binding.sourceGeometryStableId);
    w.integer(binding.consumerGeometryStableId); w.integer(binding.scoreSourceStableId);
    w.integer(static_cast<std::uint8_t>(binding.mode));
    w.integer(static_cast<std::uint8_t>(binding.mapping.x));
    w.integer(static_cast<std::uint8_t>(binding.mapping.y));
    w.integer(static_cast<std::uint8_t>(binding.mapping.z));
    w.floating(binding.mapping.xScale); w.floating(binding.mapping.yScale); w.floating(binding.mapping.zScale);
    w.floating(binding.radius); w.floating(binding.gain); w.floating(binding.bias);
    for (auto value : binding.direction) w.floating(value);
    for (auto prime : binding.primes) w.integer(prime);
    w.integer<std::uint64_t>(binding.positions.size());
    for (std::size_t index = 0; index < binding.positions.size(); ++index) {
      for (auto value : binding.positions[index]) w.floating(value);
      w.integer(binding.elementIds[index]);
    }
  }
  w.integer<std::uint32_t>(static_cast<std::uint32_t>(std::count_if(v.spectrumFields.begin(), v.spectrumFields.end(),
      [](const auto& binding) { return binding.history.enabled; })));
  for (const auto& binding : v.spectrumFields) {
    const auto& history = binding.history;
    if (!history.enabled) continue;
    w.integer(binding.fieldStableId);
    w.integer<std::uint8_t>(history.live ? 1 : 0);
    w.integer<std::uint8_t>(history.useTimeline ? 1 : 0);
    w.floating(history.currentTimeSeconds); w.integer(history.depth); w.floating(history.spacingSeconds);
    w.integer<std::uint64_t>(history.coordinates.size());
    for (auto coordinate : history.coordinates) w.floating(coordinate);
  }
  for (const auto& binding : v.spectrumFields) {
    w.integer<std::uint8_t>(static_cast<std::uint8_t>(binding.source));
    w.integer(binding.sourceTrackId);
  }
  return std::move(w.bytes);
}

// replaysExactly's answer for each value, against the exact bytes the value encodes to with
// its data. Those bytes carry the whole value, so equal bytes are an equal value and the same
// answer; any changed byte misses and replays. A graph edit that leaves a generated mesh alone
// lowers it again, and its replay is seconds on a million-vertex sea.
inline ExactTokenMemo<bool>& replaysExactlyMemo() {
  // Never destroyed: worker threads may still encode while the process exits.
  static auto* memo = new ExactTokenMemo<bool>(2);
  return *memo;
}
} // namespace transport_detail

// replayBody asks for version 21 when replaysExactly admits the value. Timeline
// bindings ask for it, because the helper replays their plans anyway. Every other
// value keeps the bytes it had before.
inline std::vector<std::uint8_t>
encodeRuntimeValue(const AdmittedValue &admitted, bool replayBody = false) {
  using namespace transport_detail;
  const auto &v = admitted.descriptor();
  auto full = writeRuntimeValue(v, false);
  if (!replayBody) return full;
  const std::string_view token(reinterpret_cast<const char*>(full.data()), full.size());
  bool exact = false;
  if (const auto kept = replaysExactlyMemo().find(token)) exact = *kept;
  else {
    exact = replaysExactly(v);
    replaysExactlyMemo().store(token, std::make_shared<const bool>(exact));
  }
  return exact ? writeRuntimeValue(v, true) : full;
}

inline std::optional<AdmittedValue>
decodeRuntimeValueWithRetainedResources(const std::uint8_t *bytes, std::size_t size,
                   const PortContract &contract, const ResourceLimits &limits,
                   const AdmissionContext &context, std::string &error,
                   const RetainedResourceOwner &resources,
                   const std::function<bool(const OperationRecord&, std::string&)>& admitOperation = {}) {
  using namespace transport_detail;
  Reader r(bytes, size);
  for (auto expected : kTransportMagic) {
    std::uint8_t actual = 0;
    if (!r.integer(actual) || actual != expected) {
      error =
          r.error.empty() ? "invalid Geometry Core transport magic" : r.error;
      return std::nullopt;
    }
  }
  std::uint16_t version = 0, reserved = 0;
  if (!r.integer(version) || !r.integer(reserved)) {
    error = r.error;
    return std::nullopt;
  }
  if ((version < 1 || version > kTransportVersion) || reserved != 0) {
    error = "unknown Geometry Core transport version or flags";
    return std::nullopt;
  }
  ValueDescriptor v;
  std::uint8_t carrier = 0, overflow = 0, domain = 0, type = 0,
               interpolation = 0;
  std::uint64_t domainCardinality = 0, dispatches = 0;
  std::size_t attributeCount = 0, operationCount = 0;
  if (!r.integer(v.schemaVersion) || !r.integer(carrier) ||
      !r.integer(v.stableId) || !r.integer(v.sourceStableId) ||
      !r.integer(v.sourceRevision) || !r.integer(overflow) ||
      !r.integer(domain) || !r.integer(type) || !r.integer(interpolation) ||
      !r.integer(domainCardinality) || !r.integer(dispatches) ||
      !r.count(attributeCount, std::min(kMaximumAttributes, limits.maxAttributes)) ||
      !r.count(operationCount, std::min(kMaximumOperations, limits.maxOperations))) {
    error = r.error;
    return std::nullopt;
  }
  if (carrier < 1 || carrier > static_cast<std::uint8_t>(CarrierKind::field) ||
      overflow < 1 ||
      overflow >
          static_cast<std::uint8_t>(OverflowPolicy::keepLowestStableIds) ||
      domain > static_cast<std::uint8_t>(Domain::instance) ||
      type > static_cast<std::uint8_t>(ValueType::rotation) ||
      interpolation >
          static_cast<std::uint8_t>(Interpolation::normalizedLinear) ||
      domainCardinality > kMaximumFieldElements ||
      dispatches > kMaximumDispatches) {
    error = "unknown or out-of-range Geometry Core header value";
    return std::nullopt;
  }
  v.carrier = static_cast<CarrierKind>(carrier);
  v.overflow = static_cast<OverflowPolicy>(overflow);
  v.fieldDomain = static_cast<Domain>(domain);
  v.fieldValueType = static_cast<ValueType>(type);
  v.fieldInterpolation = static_cast<Interpolation>(interpolation);
  v.fieldDomainCardinality = static_cast<std::size_t>(domainCardinality);
  v.dispatchCount = static_cast<std::size_t>(dispatches);
  if (!r.arrayBytes(attributeCount, 22) || !r.arrayBytes(operationCount, 41)) {
    error = r.error;
    return std::nullopt;
  }
  v.attributes.resize(attributeCount);
  for (auto &a : v.attributes) {
    std::size_t count = 0;
    if (!readAttributeDescriptor(r, a.descriptor) ||
        !r.count(count, std::min(kMaximumFieldElements, limits.maxFieldElements)) ||
        !r.arrayBytes(count, 32)) {
      error = r.error;
      return std::nullopt;
    }
    a.elements.resize(count);
    for (auto &x : a.elements)
      if (!readFieldElement(r, x)) {
        error = r.error;
        return std::nullopt;
      }
  }
  v.operations.resize(operationCount);
  for (auto &op : v.operations) {
    std::uint8_t code = 0;
    if (!r.integer(op.stableId) || !r.integer(code) ||
        !r.integer(op.inputStableId) || !r.integer(op.outputStableId)
        || !r.integer(op.secondaryInputStableId)) {
      error = r.error;
      return std::nullopt;
    }
    if (code < 1 || (code!=static_cast<std::uint8_t>(OperationCode::sdfSurface)
        && code!=static_cast<std::uint8_t>(OperationCode::retainedMesh)
        && code!=static_cast<std::uint8_t>(OperationCode::cullGeometry)
        && code!=static_cast<std::uint8_t>(OperationCode::cullPoints)
        && code!=static_cast<std::uint8_t>(OperationCode::pointsOnFacesDensity)
        && code!=static_cast<std::uint8_t>(OperationCode::shadeMesh)
        && code!=static_cast<std::uint8_t>(OperationCode::subdivisionSurface)
        && code!=static_cast<std::uint8_t>(OperationCode::curveAnimate)        && !isConstructedFieldOperation(static_cast<OperationCode>(code)) &&
        (code > static_cast<std::uint8_t>(OperationCode::curveEndpointField)
         && (code < static_cast<std::uint8_t>(OperationCode::pointsOnFaces)
             || code > static_cast<std::uint8_t>(OperationCode::cullInstances)))) ||
        (version < kSubdivisionTransportVersion && code==static_cast<std::uint8_t>(OperationCode::subdivisionSurface)) ||
        (version < kCurveAnimateTransportVersion && code==static_cast<std::uint8_t>(OperationCode::curveAnimate)) ||
        (version < kCullTransportVersion && (code==static_cast<std::uint8_t>(OperationCode::cullGeometry)
            || code==static_cast<std::uint8_t>(OperationCode::cullPoints))) ||
        (version < kPointDensityTransportVersion && code==static_cast<std::uint8_t>(OperationCode::pointsOnFacesDensity)) ||
        (version < kShadeTransportVersion && code==static_cast<std::uint8_t>(OperationCode::shadeMesh)) ||
        (version < kSampleIndexTransportVersion && code==static_cast<std::uint8_t>(OperationCode::fieldSampleIndex)) ||
        (version < 13 && code==static_cast<std::uint8_t>(OperationCode::fieldWaterHeight)) ||
        (version < 13 && code==static_cast<std::uint8_t>(OperationCode::sdfSurface)) ||
        (version < 12 && code==static_cast<std::uint8_t>(OperationCode::fieldAudioSpectrum)) ||
        (version < 11 && code>=static_cast<std::uint8_t>(OperationCode::fieldTimeline)) ||
        (version < 9 && code>=static_cast<std::uint8_t>(OperationCode::fieldSurfaceDistance)) ||
        (version < 8 && code==static_cast<std::uint8_t>(OperationCode::retainedMesh)) ||
        (version < 6 && isConstructedFieldOperation(static_cast<OperationCode>(code))) ||
        (version < 5 && code >= static_cast<std::uint8_t>(OperationCode::pointsOnFaces)) ||
        (version < 3 && code > static_cast<std::uint8_t>(OperationCode::vectorDisplacement))) {
      error = "unknown Geometry Core operation";
      return std::nullopt;
    }
    op.code = static_cast<OperationCode>(code);
    for (auto &x : op.parameters)
      if (!r.floating(x)) {
        error = r.error;
        return std::nullopt;
      }
    // Older headers cannot carry the preserve-length flag, so a nonzero flag below version 17 is refused.
    if (op.code == OperationCode::setNormal && version < kSetNormalFlagTransportVersion &&
        op.parameters[0] != 0) {
      error = "Set Normal preserve-length flag requires transport version 17";
      return std::nullopt;
    }
    if ((op.code == OperationCode::transformTRS || op.code == OperationCode::transformInstances) &&
        (!readVec3(r,op.transform.translation) || !readQuaternion(r,op.transform.rotation) ||
         !readVec3(r,op.transform.scale))) {
      error = r.error;
      return std::nullopt;
    }
    if (isConstructedFieldOperation(op.code)) {
      std::uint8_t fieldDomain=0,fieldType=0;
      if (!r.integer(fieldDomain) || !r.integer(fieldType) || !r.integer(op.field.mode)
          || !r.integer(op.field.auxiliaryInputStableId)) { error=r.error; return std::nullopt; }
      op.field.domain=static_cast<Domain>(fieldDomain);
      op.field.valueType=static_cast<ValueType>(fieldType);
      for (auto& value : op.field.values)
        if (!r.floating(value)) { error=r.error; return std::nullopt; }
      if (version<15 && op.code==OperationCode::fieldWaterHeight && op.field.values[4]!=0) {
        error="Water body coupling mass requires GCRV version 15"; return std::nullopt;
      }
      if (!r.string(op.field.attributeName,true) || !validFieldOperationSettings(op,error)) {
        if (error.empty()) error=r.error;
        return std::nullopt;
      }
    }
    if (op.code==OperationCode::shadeMesh) {
      // Grows as bytes arrive, so a forged count fails at the end of the buffer.
      std::uint64_t edges=0;
      if (!r.integer(edges)) { error=r.error; return std::nullopt; }
      for (std::uint64_t i=0;i<edges;++i) {
        std::array<std::uint32_t,2> edge{};
        if (!r.integer(edge[0]) || !r.integer(edge[1])) { error=r.error; return std::nullopt; }
        op.sharpEdges.push_back(edge);
      }
    }
    if(op.code==OperationCode::sdfSurface) {
      auto q=std::make_shared<sdfsurface::Request>();
      if(!r.integer(q->algorithmVersion)) { error=r.error; return std::nullopt; }
      for(auto& x:q->minimum) if(!r.floating(x)) { error=r.error; return std::nullopt; }
      for(auto& x:q->maximum) if(!r.floating(x)) { error=r.error; return std::nullopt; }
      for(auto& x:q->cells) if(!r.integer(x)) { error=r.error; return std::nullopt; }
      std::uint32_t recordCount=0;
      if(!r.floating(q->gradientStep) || !r.floating(q->maximumResidual)
          || !r.integer(q->geometry.schemaVersion) || !r.integer(q->geometry.rootId)
          || !r.integer(recordCount) || recordCount==0 || recordCount>sdfsurface::kMaximumRecords
          || (version<14 && q->geometry.schemaVersion!=kSdfIrSchemaVersion)
          || !r.arrayBytes(recordCount,version>=14 ? 99 : 91)) {
        error=r.error.empty() ? "Invalid SDF surface IR record count" : r.error; return std::nullopt;
      }
      q->geometry.records.resize(recordCount);
      for(auto& record:q->geometry.records) {
        std::uint8_t operation=0;
        if(!r.integer(record.stableId) || !r.integer(operation) || !r.integer(record.inputCount)
            || !r.integer(record.parameterCount)) { error=r.error; return std::nullopt; }
        record.operation=static_cast<SdfOperation>(operation);
        for(auto& input:record.inputs) if(!r.integer(input)) { error=r.error; return std::nullopt; }
        for(auto& parameter:record.parameters) if(!r.floating(parameter)) { error=r.error; return std::nullopt; }
        if(version>=14 && !r.integer(record.resourceStableId)) { error=r.error; return std::nullopt; }
      }
      if(version>=14) {
        std::uint32_t resourceCount=0;
        if(!r.integer(resourceCount) || resourceCount>sdfsurface::kMaximumRecords) {
          error=r.error.empty() ? "Invalid SDF surface text resource count" : r.error; return std::nullopt;
        }
        std::size_t payloadBytes=0;
        for(std::uint32_t i=0;i<resourceCount;++i) {
          std::uint32_t count=0;
          if(!r.integer(count) || count==0 || count>r.remaining
              || count>kSdfTextMaximumPayloadBytes-payloadBytes || count>limits.maxBufferBytes) {
            error=r.error.empty() ? "Invalid SDF surface text payload capacity" : r.error; return std::nullopt;
          }
          SdfTextOutline outline;
          if(!decodeSdfTextOutline(std::string_view(reinterpret_cast<const char*>(r.cursor),count),outline)) {
            error="Invalid SDF surface text outline payload"; return std::nullopt;
          }
          q->geometry.textOutlines.push_back(std::move(outline));
          r.cursor+=count; r.remaining-=count; payloadBytes+=count;
        }
      }
      sdfsurface::Budget budget;
      if(!sdfsurface::inspect(*q,budget,error) || budget.temporaryBytes>limits.maxBufferBytes) {
        if(error.empty()) error="SDF surface transport exceeds temporary memory capacity";
        return std::nullopt;
      }
      op.sdfSurface=std::move(q);
    }
    if (op.code==OperationCode::subdivisionSurface) {
      auto record=std::make_shared<SubdivisionSurfaceRecord>();
      std::size_t creases=0;
      if (!r.count(creases,std::numeric_limits<std::size_t>::max()) || !r.arrayBytes(creases,20)) {
        error=r.error; return std::nullopt;
      }
      record->creases.resize(creases);
      for (auto& crease : record->creases)
        if (!r.integer(crease.vertexA) || !r.integer(crease.vertexB) || !r.floating(crease.weight)) {
          error=r.error; return std::nullopt;
        }
      op.subdivision=std::move(record);
    }
    if (op.code==OperationCode::retainedMesh) {
      auto source=resources ? makeRetainedResourcePayload<RetainedMeshData>(resources)
                            : std::make_shared<RetainedMeshData>();
      std::size_t vertices=0,indices=0,attributes=0;
      if (!r.count(vertices,limits.maxVertices)
          || !r.count(indices,limits.maxIndices)
          || !r.count(attributes,limits.maxAttributes)
          || !r.arrayBytes(vertices,20) || !r.arrayBytes(indices,4)) {
        error=r.error; return std::nullopt;
      }
      source->geometry.positions.resize(vertices); source->geometry.vertexIds.resize(vertices);
      source->geometry.indices.resize(indices); source->attributes.resize(attributes);
      for (std::size_t i=0;i<vertices;++i)
        if (!readVec3(r,source->geometry.positions[i]) || !r.integer(source->geometry.vertexIds[i])) {
          error=r.error; return std::nullopt;
        }
      for (auto& index : source->geometry.indices) if (!r.integer(index)) { error=r.error; return std::nullopt; }
      for (auto& attribute : source->attributes) {
        std::size_t count=0;
        if (!readAttributeDescriptor(r,attribute.descriptor)
            || !r.count(count,limits.maxFieldElements) || !r.arrayBytes(count,32)) {
          error=r.error; return std::nullopt;
        }
        attribute.elements.resize(count);
        for (auto& element : attribute.elements) if (!readFieldElement(r,element)) { error=r.error; return std::nullopt; }
      }
      if (version >= 10 && !r.string(source->importedAnimation,true,kMaximumBufferBytes)) {
        error=r.error; return std::nullopt;
      }
      op.retainedMesh=std::move(source);
    }
  }
  const bool replayedValue = version >= kReplayedValueTransportVersion && attributeCount == 0;
  if (version >= kReplayedValueTransportVersion && !replayedValue) {
    error = "Geometry Core transport version 21 carries attributes it should rebuild";
    return std::nullopt;
  }
  if (replayedValue) {
    // The encoder left out the data because the operations rebuild it exactly.
    if (v.carrier == CarrierKind::field) {
      error = "Geometry Core transport version 21 cannot rebuild a Field value";
      return std::nullopt;
    }
  } else if (v.carrier == CarrierKind::geometry3D) {
    GeometryData p;
    std::size_t vertices = 0, indices = 0;
    if (!r.count(vertices, std::min(kMaximumVertices, limits.maxVertices)) ||
        !r.count(indices, std::min(kMaximumIndices, limits.maxIndices)) ||
        !r.arrayBytes(vertices, 20) || !r.arrayBytes(indices, 4)) {
      error = r.error;
      return std::nullopt;
    }
    p.positions.resize(vertices);
    p.vertexIds.resize(vertices);
    p.indices.resize(indices);
    for (std::size_t i = 0; i < vertices; ++i)
      if (!readVec3(r, p.positions[i]) || !r.integer(p.vertexIds[i])) {
        error = r.error;
        return std::nullopt;
      }
    for (auto &x : p.indices)
      if (!r.integer(x)) {
        error = r.error;
        return std::nullopt;
      }
    v.data = std::move(p);
  } else if (v.carrier == CarrierKind::points3D) {
    PointsData p;
    std::size_t count = 0;
    if (!r.count(count, std::min(kMaximumPoints, limits.maxPoints)) ||
        !r.arrayBytes(count, 40)) {
      error = r.error;
      return std::nullopt;
    }
    p.points.resize(count);
    for (auto &x : p.points)
      if (!readVec3(r, x.position) || !r.integer(x.stableId) ||
          !r.floating(x.radius) || !readQuaternion(r, x.rotation)) {
        error = r.error;
        return std::nullopt;
      }
    v.data = std::move(p);
  } else if (v.carrier == CarrierKind::curves3D) {
    CurvesData p;
    std::size_t points = 0, splines = 0;
    if (!r.count(points, std::min(kMaximumCurvePoints, limits.maxCurvePoints)) ||
        !r.count(splines, std::min(kMaximumSplines, limits.maxSplines)) ||
        !r.arrayBytes(points, 28) || !r.arrayBytes(splines, 17)) {
      error = r.error;
      return std::nullopt;
    }
    p.points.resize(points);
    p.splines.resize(splines);
    for (auto &x : p.points)
      if (!readVec3(r, x.position) || !r.integer(x.stableId) ||
          !r.floating(x.radius) || !r.floating(x.tilt)) {
        error = r.error;
        return std::nullopt;
      }
    for (auto &s : p.splines) {
      std::uint8_t cyclic = 0;
      if (!r.integer(s.stableId) || !r.integer(s.firstPoint) ||
          !r.integer(s.pointCount) || !r.integer(cyclic) || cyclic > 1) {
        error = r.error.empty() ? "invalid Geometry Core spline flag" : r.error;
        return std::nullopt;
      }
      s.cyclic = cyclic != 0;
    }
    v.data = std::move(p);
  } else if (v.carrier == CarrierKind::instances3D) {
    InstancesData p;
    std::size_t count = 0;
    if (!r.count(count, std::min(kMaximumInstances, limits.maxInstances)) ||
        !r.arrayBytes(count, 56)) {
      error = r.error;
      return std::nullopt;
    }
    p.instances.resize(count);
    for (auto &x : p.instances)
      if (!r.integer(x.stableId) || !r.integer(x.sourceStableId) ||
          !readVec3(r, x.transform.translation) ||
          !readQuaternion(r, x.transform.rotation) ||
          !readVec3(r, x.transform.scale)) {
        error = r.error;
        return std::nullopt;
      }
    v.data = std::move(p);
  } else {
    FieldData p;
    std::size_t count = 0;
    if (!r.count(count, std::min(kMaximumFieldElements, limits.maxFieldElements)) ||
        !r.arrayBytes(count, 32)) {
      error = r.error;
      return std::nullopt;
    }
    p.elements.resize(count);
    for (auto &x : p.elements)
      if (!readFieldElement(r, x)) {
        error = r.error;
        return std::nullopt;
      }
    v.data = std::move(p);
  }
  if (version >= 2) {
    std::uint32_t bindings = 0;
    // Each encoded binding takes at least eight bytes, so the count is bounded by the bytes present.
    if (!r.integer(bindings) || bindings > r.remaining / 8) {
      error = "Audio Spectrum Field binding count exceeds the encoded bytes";
      return std::nullopt;
    }
    const auto spectrumCoordinateLimit=[&](StableId id) {
      const auto operation=std::find_if(v.operations.begin(),v.operations.end(),[&](const auto& op) {
        return op.code==OperationCode::fieldAudioSpectrum && op.stableId==id;
      });
      if (operation!=v.operations.end()) {
        const auto fieldLimit=std::min(kMaximumFieldElements,limits.maxFieldElements);
        switch (operation->field.domain) {
          case Domain::point: return std::min(fieldLimit,std::min(kMaximumPoints,limits.maxPoints));
          case Domain::curvePoint: return std::min(fieldLimit,std::min(kMaximumCurvePoints,limits.maxCurvePoints));
          case Domain::instance: return std::min(fieldLimit,std::min(kMaximumInstances,limits.maxInstances));
          case Domain::unspecified: case Domain::vertex: case Domain::faceCorner:
          case Domain::face: case Domain::spline:
          default: break;
        }
      }
      return std::min(kMaximumVertices,limits.maxVertices);
    };
    v.spectrumFields.resize(bindings);
    for (auto& binding : v.spectrumFields) {
      std::uint8_t interpolate = 0, normalize = 0;
      if (!r.integer(binding.fieldStableId) || !r.integer(binding.sourceGeometryStableId)
          || !r.integer(binding.consumerGeometryStableId)
          || !r.integer(binding.firstBand) || !r.integer(binding.lastBand)
          || !r.integer(interpolate) || !r.integer(normalize)
          || interpolate > 1 || normalize > 1 || !r.floating(binding.gain)
          || !r.floating(binding.bias) || !r.floating(binding.attackSeconds)
          || !r.floating(binding.releaseSeconds)) {
        error = r.error.empty() ? "Invalid Audio Spectrum Field binding" : r.error;
        return std::nullopt;
      }
      binding.interpolate = interpolate != 0; binding.normalize = normalize != 0;
      for (auto& value : binding.direction)
        if (!r.floating(value)) { error = r.error; return std::nullopt; }
      std::size_t count = 0;
      if (!r.count(count, spectrumCoordinateLimit(binding.fieldStableId))
          || !r.arrayBytes(count, sizeof(float))) {
        error = r.error; return std::nullopt;
      }
      binding.coordinates.resize(count);
      for (auto& value : binding.coordinates)
        if (!r.floating(value)) { error = r.error; return std::nullopt; }
      if (version >= 3) {
        std::uint8_t target = 0;
        if (!r.integer(target) || target > static_cast<std::uint8_t>(version >= 4
            ? spectrum::Target::geometryDeformation : spectrum::Target::instanceTransform)) {
          error = "Invalid spectrum binding target"; return std::nullopt;
        }
        binding.target = static_cast<spectrum::Target>(target);
        for (auto& value : binding.baseScale)
          if (!r.floating(value)) { error = r.error; return std::nullopt; }
        for (auto& value : binding.scaleResponse)
          if (!r.floating(value)) { error = r.error; return std::nullopt; }
      }
      if (version >= 4) {
        auto& d = binding.deformer;
        std::uint8_t mode = 0;
        if (!r.integer(mode) || mode > static_cast<std::uint8_t>(spectrum::Deformation::attraction)) {
          error = "Invalid Audio Deformer mode"; return std::nullopt;
        }
        d.mode = static_cast<spectrum::Deformation>(mode);
        for (auto& value : d.center)
          if (!r.floating(value)) { error = r.error; return std::nullopt; }
        for (auto* value : {&d.frequency,&d.speed,&d.phase,&d.falloffRadius,&d.falloffPower,
             &d.delaySeconds,&d.threshold,&d.hysteresis,&d.peakHoldSeconds,&d.quantizeSteps,&d.envelopePower})
          if (!r.floating(*value)) { error = r.error; return std::nullopt; }
      }
      // Versions before 16 have no input choice, so they keep band range.
      if (version >= kEffectorInputTransportVersion) {
        std::uint8_t input = 0;
        if (!r.integer(input) || input > static_cast<std::uint8_t>(spectrum::EffectorInput::onset)) {
          error = "Invalid Audio Effector input"; return std::nullopt;
        }
        // Same refusal as spectrum::validate: only geometry deformation stages choose a scalar.
        if (input != static_cast<std::uint8_t>(spectrum::EffectorInput::bandRange)
            && binding.target != spectrum::Target::geometryDeformation) {
          error = "Only Audio Effector stages choose an analyzer scalar input"; return std::nullopt;
        }
        binding.input = static_cast<spectrum::EffectorInput>(input);
      }
      // Versions before 20 delay only in seconds.
      if (version >= kEffectorBeatDelayTransportVersion && !r.floating(binding.deformer.delayBeats)) {
        error = r.error; return std::nullopt;
      }
    }
  }
  if (version >= 5) {
    std::uint32_t bindings = 0;
    if (!r.integer(bindings) || bindings > r.remaining / 8) {
      error = "Score Field binding count exceeds the encoded bytes"; return std::nullopt;
    }
    v.scoreFields.resize(bindings);
    for (auto& binding : v.scoreFields) {
      std::uint8_t mode = 0, x = 0, y = 0, z = 0;
      if (!r.integer(binding.fieldStableId) || !r.integer(binding.sourceGeometryStableId)
          || !r.integer(binding.consumerGeometryStableId) || !r.integer(binding.scoreSourceStableId)
          || !r.integer(mode) || !r.integer(x) || !r.integer(y) || !r.integer(z)
          || !r.floating(binding.mapping.xScale) || !r.floating(binding.mapping.yScale)
          || !r.floating(binding.mapping.zScale) || !r.floating(binding.radius)
          || !r.floating(binding.gain) || !r.floating(binding.bias)) {
        error = r.error; return std::nullopt;
      }
      binding.mode = static_cast<scorefield::Mode>(mode);
      binding.mapping.x = static_cast<visualnoteinstancing::MappingAxis>(x);
      binding.mapping.y = static_cast<visualnoteinstancing::MappingAxis>(y);
      binding.mapping.z = static_cast<visualnoteinstancing::MappingAxis>(z);
      for (auto& value : binding.direction)
        if (!r.floating(value)) { error = r.error; return std::nullopt; }
      for (auto& prime : binding.primes)
        if (!r.integer(prime)) { error = r.error; return std::nullopt; }
      std::size_t count = 0;
      if (!r.count(count, std::max(limits.maxVertices,limits.maxInstances))
          || !r.arrayBytes(count, 3 * sizeof(float) + sizeof(StableId))) {
        error = r.error; return std::nullopt;
      }
      binding.positions.resize(count); binding.elementIds.resize(count);
      for (std::size_t index = 0; index < count; ++index) {
        for (auto& value : binding.positions[index])
          if (!r.floating(value)) { error = r.error; return std::nullopt; }
        if (!r.integer(binding.elementIds[index])) { error = r.error; return std::nullopt; }
      }
    }
  }
  if (version >= 7) {
    std::uint32_t histories = 0;
    if (!r.integer(histories) || histories > v.spectrumFields.size()) {
      error = "Spectral History binding count exceeds the retained spectrum bindings"; return std::nullopt;
    }
    for (std::uint32_t index = 0; index < histories; ++index) {
      StableId identity = 0;
      if (!r.integer(identity)) { error = r.error; return std::nullopt; }
      const auto binding = std::find_if(v.spectrumFields.begin(), v.spectrumFields.end(),
          [identity](const auto& value) { return value.fieldStableId == identity; });
      if (binding == v.spectrumFields.end() || binding->history.enabled) {
        error = "Spectral History has an unknown or duplicate spectrum identity"; return std::nullopt;
      }
      auto coordinateLimit=std::min(kMaximumVertices,limits.maxVertices);
      const auto operation=std::find_if(v.operations.begin(),v.operations.end(),[&](const auto& op) {
        return op.code==OperationCode::fieldAudioSpectrum && op.stableId==identity;
      });
      if (operation!=v.operations.end()) {
        if (operation->field.domain==Domain::point) coordinateLimit=std::min(kMaximumPoints,limits.maxPoints);
        else if (operation->field.domain==Domain::curvePoint) coordinateLimit=std::min(kMaximumCurvePoints,limits.maxCurvePoints);
        else if (operation->field.domain==Domain::instance) coordinateLimit=std::min(kMaximumInstances,limits.maxInstances);
        coordinateLimit=std::min(coordinateLimit,std::min(kMaximumFieldElements,limits.maxFieldElements));
      }
      auto& history = binding->history;
      std::uint8_t live = 0, timeline = 0;
      std::size_t count = 0;
      if (!r.integer(live) || live > 1 || !r.integer(timeline) || timeline > 1
          || !r.floating(history.currentTimeSeconds) || !r.integer(history.depth)
          || !r.floating(history.spacingSeconds)
          || !r.count(count, coordinateLimit)
          || count != binding->coordinates.size() || !r.arrayBytes(count, sizeof(float))) {
        error = r.error.empty() ? "Invalid Spectral History controls or coordinates" : r.error; return std::nullopt;
      }
      // Same refusal as spectrum::validate: live history cannot read a scalar input.
      if (live != 0 && binding->input != spectrum::EffectorInput::bandRange) {
        error = "Live Spectral History cannot use Audio Effector RMS, Peak or Onset input. Select Project Baked Analysis";
        return std::nullopt;
      }
      history.enabled = true; history.live = live != 0; history.useTimeline = timeline != 0;
      history.coordinates.resize(count);
      for (auto& coordinate : history.coordinates)
        if (!r.floating(coordinate)) { error = r.error; return std::nullopt; }
    }
  }
  if (version >= 12) {
    for (auto& binding : v.spectrumFields) {
      std::uint8_t source = 0;
      if (!r.integer(source) || source > static_cast<std::uint8_t>(spectrum::Source::group)
          || !r.integer(binding.sourceTrackId)) {
        error = "Invalid spectrum analysis source";
        return std::nullopt;
      }
      binding.source = static_cast<spectrum::Source>(source);
    }
  }
  if (r.remaining != 0) {
    error = "trailing data after Geometry Core runtime value";
    return std::nullopt;
  }
  // Consumer-specific subsets must reject unsupported generators before
  // admission materializes their operation results.
  if (admitOperation)
    for (const auto& operation : v.operations)
      if (!admitOperation(operation, error)) return std::nullopt;
  if (replayedValue) {
    // The plan check reads the carrier's type, so it starts from an empty one.
    if (v.carrier == CarrierKind::geometry3D) v.data = GeometryData{};
    else if (v.carrier == CarrierKind::points3D) v.data = PointsData{};
    else if (v.carrier == CarrierKind::curves3D) v.data = CurvesData{};
    else v.data = InstancesData{};
    // Same replay the encoder checked, with no admission context. Admission below
    // then checks the rebuilt value against its contract and context as usual.
    ValueDescriptor rebuilt;
    if (!validateOperationPlan(v, AdmissionContext{}, nullptr, error, &rebuilt, nullptr, {}, &limits))
      return std::nullopt;
    v.data = std::move(rebuilt.data);
    v.attributes = std::move(rebuilt.attributes);
  }
  return resources ? admitValueWithRetainedResources(std::move(v), contract, limits, context, error, resources)
                   : admitValue(std::move(v), contract, limits, context, error);
}
inline std::optional<AdmittedValue>
decodeRuntimeValue(const std::uint8_t *bytes, std::size_t size,
                   const PortContract &contract, const ResourceLimits &limits,
                   const AdmissionContext &context, std::string &error,
                   const std::function<bool(const OperationRecord&, std::string&)>& admitOperation = {}) {
  return decodeRuntimeValueWithRetainedResources(bytes,size,contract,limits,context,error,{},admitOperation);
}
inline std::optional<AdmittedValue>
decodeRuntimeValueWithRetainedResources(const std::vector<std::uint8_t> &bytes,
                   const PortContract &contract, const ResourceLimits &limits,
                   const AdmissionContext &context, std::string &error,
                   const RetainedResourceOwner &resources,
                   const std::function<bool(const OperationRecord&, std::string&)>& admitOperation = {}) {
  return decodeRuntimeValueWithRetainedResources(bytes.data(),bytes.size(),contract,limits,context,error,resources,admitOperation);
}
inline std::optional<AdmittedValue>
decodeRuntimeValue(const std::vector<std::uint8_t> &bytes,
                   const PortContract &contract, const ResourceLimits &limits,
                   const AdmissionContext &context, std::string &error,
                   const std::function<bool(const OperationRecord&, std::string&)>& admitOperation = {}) {
  return decodeRuntimeValue(bytes.data(), bytes.size(), contract, limits,
                            context, error, admitOperation);
}

struct LoweredRuntimePlan final {
  PortContract contract;
  std::vector<std::uint8_t> runtimeValue;
  // Canonical Surface binding wire; admitted by the native scene consumer.
  // Part of the immutable plan bytes (and therefore the existing owner cache key).
  std::string surfaceMaterial;
  // Encoded slot media (GeometryCoreSlotMedia.h), version 3 and 4.
  std::string slotMedia;
  // Render 3D settings of a direct route (GeometryCoreRenderSettings.h), version 4 only.
  std::string renderSettings;
};
inline std::string
encodeLoweredPlanText(const std::vector<std::uint8_t> &bytes) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string text;
  text.resize(bytes.size() * 2);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    text[i * 2] = digits[bytes[i] >> 4];
    text[i * 2 + 1] = digits[bytes[i] & 15];
  }
  return text;
}
inline std::optional<std::vector<std::uint8_t>>
decodeLoweredPlanText(const std::string &text, std::string &error) {
  if (text.empty() || text.size() % 2 != 0 ||
      text.size() > kMaximumBufferBytes * 2) {
    error = "lowered Geometry Core text length is invalid";
    return std::nullopt;
  }
  const auto nibble = [](char c) -> int {
    if (c >= '0' && c <= '9')
      return c - '0';
    if (c >= 'a' && c <= 'f')
      return c - 'a' + 10;
    return -1;
  };
  std::vector<std::uint8_t> bytes(text.size() / 2);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    const int high = nibble(text[i * 2]), low = nibble(text[i * 2 + 1]);
    if (high < 0 || low < 0) {
      error = "lowered Geometry Core text is not canonical hexadecimal";
      return std::nullopt;
    }
    bytes[i] = static_cast<std::uint8_t>((high << 4) | low);
  }
  error.clear();
  return bytes;
}
inline std::vector<std::uint8_t> lowerRuntimePlan(const PortContract &contract,
                                                  const AdmittedValue &value,
                                                  const std::string& surfaceMaterial = {},
                                                  const std::string& slotMedia = {},
                                                  bool replayBody = false,
                                                  const std::string& renderSettings = {}) {
  using namespace transport_detail;
  Writer w;
  for (auto b : std::array<std::uint8_t, 4>{{'G', 'C', 'L', 'P'}})
    w.integer(b);
  // Version 3 adds slot media after a Surface binding that may be empty.
  // Version 4 adds render settings after both, either of which may be empty.
  // Plans without them keep their version 1, 2 or 3 bytes.
  w.integer<std::uint16_t>(!renderSettings.empty() ? 4 : !slotMedia.empty() ? 3 : surfaceMaterial.empty() ? 1 : 2);
  w.integer<std::uint16_t>(0);
  w.integer(contract.schemaVersion);
  w.integer(static_cast<std::uint8_t>(contract.carrier));
  w.integer(static_cast<std::uint8_t>(contract.fieldDomain));
  w.integer(static_cast<std::uint8_t>(contract.fieldValueType));
  w.integer(static_cast<std::uint8_t>(contract.fieldInterpolation));
  w.integer(static_cast<std::uint8_t>(contract.overflow));
  for (auto count :
       {contract.maxVertices, contract.maxIndices, contract.maxPoints,
        contract.maxCurvePoints, contract.maxSplines, contract.maxInstances,
        contract.maxFieldElements, contract.maxAttributes})
    w.integer<std::uint64_t>(count);
  w.integer<std::uint64_t>(contract.attributes.size());
  for (const auto &a : contract.attributes)
    writeAttributeDescriptor(w, a);
  const auto runtime = encodeRuntimeValue(value, replayBody);
  w.integer<std::uint64_t>(runtime.size());
  w.bytes.insert(w.bytes.end(), runtime.begin(), runtime.end());
  if (!surfaceMaterial.empty() || !slotMedia.empty() || !renderSettings.empty()) {
    w.integer<std::uint64_t>(surfaceMaterial.size());
    w.bytes.insert(w.bytes.end(), surfaceMaterial.begin(), surfaceMaterial.end());
  }
  if (!slotMedia.empty() || !renderSettings.empty()) {
    w.integer<std::uint64_t>(slotMedia.size());
    w.bytes.insert(w.bytes.end(), slotMedia.begin(), slotMedia.end());
  }
  if (!renderSettings.empty()) {
    w.integer<std::uint64_t>(renderSettings.size());
    w.bytes.insert(w.bytes.end(), renderSettings.begin(), renderSettings.end());
  }
  return std::move(w.bytes);
}
inline std::optional<LoweredRuntimePlan>
decodeLoweredRuntimePlan(const std::vector<std::uint8_t> &bytes,
                         std::string &error) {
  using namespace transport_detail;
  Reader r(bytes.data(), bytes.size());
  for (auto expected : std::array<std::uint8_t, 4>{{'G', 'C', 'L', 'P'}}) {
    std::uint8_t actual = 0;
    if (!r.integer(actual) || actual != expected) {
      error = r.error.empty() ? "invalid lowered Geometry Core plan magic"
                              : r.error;
      return std::nullopt;
    }
  }
  std::uint16_t version = 0, flags = 0;
  if (!r.integer(version) || !r.integer(flags)) {
    error = r.error;
    return std::nullopt;
  }
  if (version < 1 || version > 4 || flags != 0) {
    error = "unknown lowered Geometry Core plan version or flags";
    return std::nullopt;
  }
  LoweredRuntimePlan plan;
  std::uint8_t carrier = 0, domain = 0, type = 0, interpolation = 0,
               overflow = 0;
  if (!r.integer(plan.contract.schemaVersion) || !r.integer(carrier) ||
      !r.integer(domain) || !r.integer(type) || !r.integer(interpolation) ||
      !r.integer(overflow)) {
    error = r.error;
    return std::nullopt;
  }
  if (carrier < 1 || carrier > static_cast<std::uint8_t>(CarrierKind::field) ||
      domain > static_cast<std::uint8_t>(Domain::instance) ||
      type > static_cast<std::uint8_t>(ValueType::rotation) ||
      interpolation >
          static_cast<std::uint8_t>(Interpolation::normalizedLinear) ||
      overflow < 1 ||
      overflow >
          static_cast<std::uint8_t>(OverflowPolicy::keepLowestStableIds)) {
    error = "unknown lowered Geometry Core contract enum";
    return std::nullopt;
  }
  plan.contract.carrier = static_cast<CarrierKind>(carrier);
  plan.contract.fieldDomain = static_cast<Domain>(domain);
  plan.contract.fieldValueType = static_cast<ValueType>(type);
  plan.contract.fieldInterpolation = static_cast<Interpolation>(interpolation);
  plan.contract.overflow = static_cast<OverflowPolicy>(overflow);
  std::size_t *counts[] = {
      &plan.contract.maxVertices,      &plan.contract.maxIndices,
      &plan.contract.maxPoints,        &plan.contract.maxCurvePoints,
      &plan.contract.maxSplines,       &plan.contract.maxInstances,
      &plan.contract.maxFieldElements, &plan.contract.maxAttributes};
  const std::size_t maxima[] = {kMaximumVertices,      kMaximumIndices,
                                kMaximumPoints,        kMaximumCurvePoints,
                                kMaximumSplines,       kMaximumInstances,
                                kMaximumFieldElements, kMaximumAttributes};
  for (std::size_t i = 0; i < 8; ++i)
    if (!r.count(*counts[i], maxima[i])) {
      error = r.error;
      return std::nullopt;
    }
  std::size_t attributes = 0;
  if (!r.count(attributes, kMaximumAttributes)) {
    error = r.error;
    return std::nullopt;
  }
  plan.contract.attributes.resize(attributes);
  for (auto &a : plan.contract.attributes)
    if (!readAttributeDescriptor(r, a)) {
      error = r.error;
      return std::nullopt;
    }
  std::string validation;
  if (!validatePortContract(plan.contract, validation)) {
    error = validation;
    return std::nullopt;
  }
  std::size_t runtimeSize = 0;
  if (!r.count(runtimeSize, kMaximumBufferBytes) ||
      runtimeSize > r.remaining || (version == 1 && runtimeSize != r.remaining)) {
    error = r.error.empty() ? "lowered Geometry Core runtime length is invalid"
                            : r.error;
    return std::nullopt;
  }
  plan.runtimeValue.assign(r.cursor, r.cursor + runtimeSize);
  r.cursor += runtimeSize;
  r.remaining -= runtimeSize;
  if (version == 2) {
    std::size_t materialSize = 0;
    if (!r.count(materialSize, 512 * 1024) || materialSize == 0 || materialSize != r.remaining) {
      error = "lowered Geometry Core Surface binding length is invalid";
      return std::nullopt;
    }
    plan.surfaceMaterial.assign(reinterpret_cast<const char*>(r.cursor), materialSize);
  }
  if (version == 3) {
    std::size_t materialSize = 0, mediaSize = 0;
    if (!r.count(materialSize, 512 * 1024) || materialSize > r.remaining) {
      error = "lowered Geometry Core Surface binding length is invalid";
      return std::nullopt;
    }
    plan.surfaceMaterial.assign(reinterpret_cast<const char*>(r.cursor), materialSize);
    r.cursor += materialSize;
    r.remaining -= materialSize;
    if (!r.count(mediaSize, 512 * 1024) || mediaSize == 0 || mediaSize != r.remaining) {
      error = "lowered Geometry Core slot media length is invalid";
      return std::nullopt;
    }
    plan.slotMedia.assign(reinterpret_cast<const char*>(r.cursor), mediaSize);
  }
  if (version == 4) {
    std::size_t materialSize = 0, mediaSize = 0, settingsSize = 0;
    if (!r.count(materialSize, 512 * 1024) || materialSize > r.remaining) {
      error = "lowered Geometry Core Surface binding length is invalid";
      return std::nullopt;
    }
    plan.surfaceMaterial.assign(reinterpret_cast<const char*>(r.cursor), materialSize);
    r.cursor += materialSize;
    r.remaining -= materialSize;
    if (!r.count(mediaSize, 512 * 1024) || mediaSize > r.remaining) {
      error = "lowered Geometry Core slot media length is invalid";
      return std::nullopt;
    }
    plan.slotMedia.assign(reinterpret_cast<const char*>(r.cursor), mediaSize);
    r.cursor += mediaSize;
    r.remaining -= mediaSize;
    // Settings may carry Render 3D's lights and camera, so they fill the rest of the plan.
    if (!r.count(settingsSize, r.remaining) || settingsSize == 0 || settingsSize != r.remaining) {
      error = "lowered Geometry Core render settings length is invalid";
      return std::nullopt;
    }
    plan.renderSettings.assign(reinterpret_cast<const char*>(r.cursor), settingsSize);
  }
  error.clear();
  return plan;
}
} // namespace videowire::geometry
