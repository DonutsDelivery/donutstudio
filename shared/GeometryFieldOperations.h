#pragma once

// Included by GeometryCore.h after the carrier and element-query definitions.
namespace videowire::geometry {
// Surface grids allocate what they need. Water replay owns rest positions and
// identities, four wave arrays, the height and offset fields, and the retained
// triangles. A wave holds its ten fields with the rest vertices.
inline constexpr std::uint64_t kWaterSurfaceBytesPerVertex =
    sizeof(Vec3) + sizeof(StableId) + 4 * sizeof(double) + 2 * sizeof(FieldElement);
inline constexpr std::uint64_t kDirectionalSurfaceWaveBytesPerVertex =
    sizeof(Vec3) + sizeof(StableId) + 10 * sizeof(FieldElement);
inline constexpr double kWaterSurfaceStepSeconds = 1.0 / 120.0;

// False when the grid's size does not fit in 64 bits.
inline bool waterSurfaceRequiredBytes(std::uint64_t columns, std::uint64_t rows, std::uint64_t& bytes) noexcept {
  if (columns == 0 || rows == 0) { bytes = 0; return true; }
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  if (columns > maximum / rows) return false;
  const auto vertices = columns * rows;
  if (vertices > maximum / kWaterSurfaceBytesPerVertex) return false;
  const auto cells = (columns - 1) * (rows - 1);
  const auto triangleBytes = 6 * sizeof(std::uint32_t);
  if (cells > maximum / triangleBytes) return false;
  bytes = vertices * kWaterSurfaceBytesPerVertex;
  if (cells * triangleBytes > maximum - bytes) return false;
  bytes += cells * triangleBytes;
  return true;
}

inline std::string surfaceGridByteText(std::uint64_t bytes) {
  return std::to_string(bytes) + " bytes (" + std::to_string(bytes >> 20) + " MiB)";
}

struct WaterSurfaceGrid final {
  std::uint32_t columns = 0, rows = 0;
  double spacingX = 0, spacingY = 0;
};

struct RuntimeFieldEvaluation final {
  double timelineSeconds = 0;
  std::function<bool(const OperationRecord&, const FieldData&, FieldData&, std::string&)> scoreAt;
  // The frame owner supplies the exact selected and followed Block B bands.
  std::function<bool(const spectrum::Binding&, spectrum::Bands&, std::string&)> spectrumAt;
  std::function<bool(const spectrum::Binding&, std::vector<spectrum::Bands>&, std::string&)> spectrumHistoryAt;
  // The frame owner maps the requested mesh timeline seconds to its existing
  // project history and reconstructs one local-Z offset per retained vertex.
  std::function<bool(const OperationRecord&, const GeometryData&, const WaterSurfaceGrid&,
                     double, std::vector<double>&, std::string&)> waterHeightAt;
  // The render view that Subdivision Surface dicing reads: a row-major 3x4
  // transform from the mesh's local space to camera space (camera at the origin),
  // tan of half the vertical field of view and the viewport height in pixels.
  // A height of 0 is no view, and every level stays as authored.
  std::array<double,12> subdivisionLocalToCamera{{1,0,0,0, 0,1,0,0, 0,0,1,0}};
  double subdivisionTanHalfFov=0, subdivisionViewportHeight=0;
  // The frame owner adds the spectrum and score displacements of the Set Position whose
  // output this mesh is. Replay calls it only for a consumer that a layout change follows.
  std::function<bool(GeometryData&, StableId, std::string&)> displaceConsumer;
  // Composed scenes apply each Audio Deformer through this frame hook when the replay
  // has no direct frame deformer.
  std::function<bool(GeometryData&, const spectrum::Binding&, std::string&)> audioDeformAt;
  // Composed scenes have no scene adapter, so every Set Position spectrum displacement
  // runs at its own operation through displaceConsumer.
  bool displaceAdapters = false;
};

namespace owner_detail {
struct FieldRuntimePredicate final {
  template <typename T>
  bool operator()(const T& op) const {
    return op.code==OperationCode::fieldWaterHeight || op.code==OperationCode::fieldTimeline || op.code==OperationCode::fieldScoreSample || op.code==OperationCode::fieldAudioSpectrum
        || op.code==OperationCode::curveAnimate
        || (op.code==OperationCode::evaluateField && op.inputStableId!=op.outputStableId);
  }
};
}
inline bool hasRuntimeFields(const ValueDescriptor& value) noexcept {
  return std::any_of(value.operations.begin(),value.operations.end(),owner_detail::FieldRuntimePredicate{});
}
inline bool hasRuntimeScoreFields(const ValueDescriptor& value) noexcept {
  return std::any_of(value.operations.begin(),value.operations.end(),[](const auto& op) {
    return op.code==OperationCode::fieldScoreSample
        || (op.code==OperationCode::evaluateField && op.inputStableId!=op.outputStableId);
  });
}
inline bool isSurfaceFieldOperation(OperationCode code) noexcept {
  return code>=OperationCode::fieldSurfaceDistance && code<=OperationCode::fieldRaycastFace;
}

inline bool isConstructedFieldOperation(OperationCode code) noexcept {
  return (code >= OperationCode::fieldConstant && code <= OperationCode::setNormal)
      || code==OperationCode::fieldWaterHeight || code==OperationCode::setInstanceAppearance || isSurfaceFieldOperation(code)
      || code==OperationCode::fieldSampleIndex
      || (code>=OperationCode::materialTableSlot && code<=OperationCode::fieldAudioSpectrum);
}

inline bool isNearestFieldOperation(OperationCode code) noexcept {
  return (code >= OperationCode::fieldProximity && code <= OperationCode::fieldSampleNearest)
      || isSurfaceFieldOperation(code);
}

inline bool equalAttributes(const std::vector<AttributeData>& a,
                            const std::vector<AttributeData>& b) noexcept {
  if (a.size() != b.size()) return false;
  for (std::size_t i=0;i<a.size();++i) {
    if (!(a[i].descriptor == b[i].descriptor) || a[i].elements.size()!=b[i].elements.size()) return false;
    for (std::size_t j=0;j<a[i].elements.size();++j)
      if (a[i].elements[j].components!=b[i].elements[j].components) return false;
  }
  return true;
}

inline StableId namedAttributeIdentity(const std::string& name, Domain domain, ValueType type) noexcept {
  std::uint64_t hash=1469598103934665603ull;
  for (const unsigned char ch : name) { hash^=ch; hash*=1099511628211ull; }
  hash^=static_cast<unsigned>(domain); hash*=1099511628211ull;
  hash^=static_cast<unsigned>(type); hash*=1099511628211ull;
  return (hash % kMaximumStableId)+1;
}

inline const AttributeData* findNamedAttribute(const std::vector<AttributeData>& attributes,
                                               const std::string& name) noexcept {
  const auto found=std::find_if(attributes.begin(),attributes.end(),[&](const auto& a) {
    return a.descriptor.name==name;
  });
  return found==attributes.end() ? nullptr : &*found;
}

inline bool transformAuthoredNormals(std::vector<AttributeData>& attributes,
                                      const Transform& transform, std::string& error) {
  for (auto& attribute : attributes) if (attribute.descriptor.name=="normal") {
    if (attribute.descriptor.domain!=Domain::vertex || attribute.descriptor.valueType!=ValueType::vector
        || transform.scale.x==0 || transform.scale.y==0 || transform.scale.z==0) {
      error="Authored normals require a nonsingular transform"; return false;
    }
    auto normalTransform=transform;
    normalTransform.translation={};
    normalTransform.scale={1/transform.scale.x,1/transform.scale.y,1/transform.scale.z};
    for (auto& element : attribute.elements) {
      const auto n=transformVector({static_cast<float>(element.components[0]),static_cast<float>(element.components[1]),
          static_cast<float>(element.components[2])},normalTransform);
      const auto length=std::sqrt(double(n.x)*n.x+double(n.y)*n.y+double(n.z)*n.z);
      if (!finite(length) || length==0) { error="Transformed normal is invalid"; return false; }
      element.components={n.x/length,n.y/length,n.z/length,0};
    }
  }
  return true;
}

namespace owner_detail {
struct FieldSettingsFinitePredicate final {
  bool operator()(double value) const { return finite(value); }
};
}
inline bool validFieldOperationSettings(const OperationRecord& op, std::string& error) {
  const auto& s=op.field;
  const bool named=op.code==OperationCode::storeNamedAttribute || op.code==OperationCode::readNamedAttribute
      || op.code==OperationCode::fieldSampleNearest || op.code==OperationCode::fieldSampleIndex
      || op.code==OperationCode::setNormal || op.code==OperationCode::setInstanceAppearance
      || op.code==OperationCode::setMaterialIndex;
  // Set Normal's parameters[0] is its preserve-length flag; materializeStoredAttribute names a bad value.
  auto checkedParameters=op.parameters;
  if (op.code==OperationCode::setNormal) checkedParameters[0]=0;
  const bool parametersZero=std::all_of(checkedParameters.begin(),checkedParameters.end(),[](float p) { return p==0; });
  if (s.domain<Domain::point || s.domain>Domain::instance || s.valueType<ValueType::floatValue
      || s.valueType>ValueType::rotation || (op.code!=OperationCode::materialTableSlot && !parametersZero)
      || !std::all_of(s.values.begin(),s.values.end(),owner_detail::FieldSettingsFinitePredicate{})
      || (named && (s.attributeName.empty() || s.attributeName.size()>kMaximumNameBytes
          || s.attributeName.find('\0')!=std::string::npos))
      || (!named && !s.attributeName.empty())
      || (op.code!=OperationCode::fieldMix && op.code!=OperationCode::fieldSelect
          && op.code!=OperationCode::fieldComposeVector && op.code!=OperationCode::setInstanceAppearance
          && !(op.code==OperationCode::fieldNoise && s.mode==2 /* ocean footprint */)
          && s.auxiliaryInputStableId!=0)) {
    error="Field operation has invalid type, domain, parameters or attribute name"; return false;
  }
  return true;
}

// Water retains the generated grid ordering, including triangle winding and IDs.
// A graded grid (the Ocean) may widen its cells with distance: rows and columns
// must still line up and increase, and its spacing is the narrowest cell.
inline bool validateWaterSurfaceGrid(const GeometryData& mesh, WaterSurfaceGrid& grid,
                                     std::string& error, bool graded = false) {
  grid = {};
  const auto count = mesh.positions.size();
  if (count < 4 || mesh.vertexIds.size() != count) {
    error = "Water surface requires a regular XY grid of at least 2 by 2 vertices"; return false;
  }
  const auto& origin = mesh.positions.front();
  std::size_t columns = 1;
  while (columns < count && mesh.positions[columns].y == origin.y) ++columns;
  if (columns < 2 || count % columns != 0 || count / columns < 2) {
    error = "Water surface requires row-major regular XY grid ordering"; return false;
  }
  std::uint64_t requiredBytes = 0;
  if (columns > std::numeric_limits<std::uint32_t>::max() || count / columns > std::numeric_limits<std::uint32_t>::max()
      || !waterSurfaceRequiredBytes(columns, count / columns, requiredBytes)) {
    error = "Water surface size overflows: its " + std::to_string(columns) + " by " + std::to_string(count / columns)
        + " grid has " + std::to_string(count) + " vertices.";
    return false;
  }
  grid.columns = static_cast<std::uint32_t>(columns);
  grid.rows = static_cast<std::uint32_t>(count / columns);
  grid.spacingX = (double(mesh.positions[columns - 1].x) - origin.x) / (columns - 1);
  grid.spacingY = (double(mesh.positions[(grid.rows - 1) * columns].y) - origin.y) / (grid.rows - 1);
  if (!finite(grid.spacingX) || !finite(grid.spacingY) || grid.spacingX <= 0 || grid.spacingY <= 0) {
    error = "Water surface grid spacing must be finite and positive"; return false;
  }
  // Float rounding grows with the coordinate, so a wide grid at a fine spacing
  // needs a tolerance from its extent. A quarter cell still bounds that growth.
  const auto regular = [](double actual, double expected, double spacing) {
    const auto epsilon = 16 * double(std::numeric_limits<float>::epsilon());
    return std::abs(actual - expected)
        <= std::max(epsilon * spacing, std::min(epsilon * std::abs(expected), 0.25 * spacing));
  };
  if (graded) {
    double narrowestX = grid.spacingX, narrowestY = grid.spacingY;
    bool even = true;
    std::set<StableId> ids;
    for (std::size_t y = 0; y < grid.rows; ++y)
      for (std::size_t x = 0; x < columns; ++x) {
        const auto i = y * columns + x;
        const auto& p = mesh.positions[i];
        if (!finite(p) || !validStableId(mesh.vertexIds[i]) || !ids.insert(mesh.vertexIds[i]).second
            || p.z != origin.z || p.x != mesh.positions[x].x || p.y != mesh.positions[y * columns].y
            || (x > 0 && !(p.x > mesh.positions[i - 1].x)) || (y > 0 && !(p.y > mesh.positions[i - columns].y))) {
          error = "Ocean rest mesh needs an XY grid whose rows and columns line up and increase"; return false;
        }
        if (y == 0 && x > 0) narrowestX = std::min(narrowestX, double(p.x) - mesh.positions[i - 1].x);
        if (x == 0 && y > 0) narrowestY = std::min(narrowestY, double(p.y) - mesh.positions[i - columns].y);
        even = even && regular(p.x, double(origin.x) + x * grid.spacingX, grid.spacingX)
            && regular(p.y, double(origin.y) + y * grid.spacingY, grid.spacingY);
      }
    // A regular grid keeps its span spacing, so ungraded projects replay unchanged.
    if (!even) { grid.spacingX = narrowestX; grid.spacingY = narrowestY; }
  } else {
  std::set<StableId> ids;
  for (std::size_t y = 0; y < grid.rows; ++y)
    for (std::size_t x = 0; x < columns; ++x) {
      const auto i = y * columns + x;
      const auto& p = mesh.positions[i];
      if (!finite(p) || !validStableId(mesh.vertexIds[i]) || !ids.insert(mesh.vertexIds[i]).second
          || p.z != origin.z || p.x != mesh.positions[x].x || p.y != mesh.positions[y * columns].y
          || !regular(p.x, double(origin.x) + x * grid.spacingX, grid.spacingX)
          || !regular(p.y, double(origin.y) + y * grid.spacingY, grid.spacingY)) {
        error = "Water surface rest mesh has nonfinite positions, irregular spacing or invalid identities"; return false;
      }
    }
  }
  if (mesh.indices.size() != (columns - 1) * (grid.rows - 1) * 6) {
    error = "Water surface requires exact retained grid triangles"; return false;
  }
  std::size_t cursor = 0;
  for (std::uint32_t y = 0; y + 1 < grid.rows; ++y)
    for (std::uint32_t x = 0; x + 1 < grid.columns; ++x) {
      const auto a = y * grid.columns + x, b = a + 1, c = a + grid.columns, d = c + 1;
      for (const auto index : {a, b, c, b, d, c})
        if (mesh.indices[cursor++] != index) {
          error = "Water surface changed retained grid triangle order"; return false;
        }
    }
  error.clear(); return true;
}

// Replay uses vNext=(1-damping*dt)*(v+dt*c*c*Laplacian(h)+impulse)
// and hNext=h+dt*vNext. These bounds cover that combined damped update.
inline bool validateWaterSurfaceSettings(const OperationRecord& op, const WaterSurfaceGrid& grid,
                                         std::string& error) {
  const auto& s = op.field;
  const auto& p = s.values;
  if (!validFieldOperationSettings(op, error) || op.code != OperationCode::fieldWaterHeight
      || s.domain != Domain::vertex || s.valueType != ValueType::floatValue || s.mode != 0
      || op.secondaryInputStableId != 0 || s.auxiliaryInputStableId != 0
      || p[0] < 0 || p[0] > 1000 || p[1] < 0 || p[1] * kWaterSurfaceStepSeconds > 1
      || p[2] < 0 || p[2] > 1000 || p[3] < 0 || p[3] > 1000
      || (p[4] != 0 && (p[4] < 0.01 || p[4] > 100)) || p[5] != 0 || p[6] != 0 || p[7] != 0
      || grid.columns < 2 || grid.rows < 2
      || !finite(grid.spacingX) || !finite(grid.spacingY) || grid.spacingX <= 0 || grid.spacingY <= 0) {
    if (error.empty()) error = "Water surface requires finite bounded authored controls";
    return false;
  }
  const auto courant = p[0] * p[0] * kWaterSurfaceStepSeconds * kWaterSurfaceStepSeconds
      * (1 / (grid.spacingX * grid.spacingX) + 1 / (grid.spacingY * grid.spacingY));
  if (!finite(courant) || courant > 1) {
    error = "Water propagation speed exceeds the fixed 120 Hz stability bound"; return false;
  }
  error.clear(); return true;
}

inline bool materializeWaterHeightField(const OperationRecord& op, const GeometryData& restMesh,
    FieldData& result, std::size_t capacity, std::string& error,
    const RuntimeFieldEvaluation* runtime = nullptr) {
  WaterSurfaceGrid grid;
  if (!validateWaterSurfaceGrid(restMesh, grid, error)
      || !validateWaterSurfaceSettings(op, grid, error)) return false;
  if (restMesh.positions.size() > capacity) {
    error = "Water surface port declares " + std::to_string(capacity) + " vertices but its "
        + std::to_string(grid.columns) + " by " + std::to_string(grid.rows) + " grid has "
        + std::to_string(restMesh.positions.size()) + ".";
    return false;
  }
  std::vector<double> heights(restMesh.positions.size(), 0);
  if (runtime) {
    if (!finite(runtime->timelineSeconds) || !runtime->waterHeightAt) {
      error = "Water surface runtime requires the frame owner's historical replay callback"; return false;
    }
    heights.clear();
    if (!runtime->waterHeightAt(op, restMesh, grid, runtime->timelineSeconds, heights, error)) {
      if (error.empty()) error = "Water surface historical replay is unavailable";
      return false;
    }
  }
  if (heights.size() != restMesh.positions.size()
      || !std::all_of(heights.begin(), heights.end(), [](double h) { return finite(h); })) {
    error = "Water surface replay changed vertex count or returned nonfinite heights"; return false;
  }
  FieldData field; field.elements.resize(heights.size());
  for (std::size_t i = 0; i < heights.size(); ++i) field.elements[i].components[0] = heights[i];
  result = std::move(field); error.clear(); return true;
}

// Spectrum coordinates are derived from the retained carrier, including its
// authored UV attribute. Instance index follows stable identity order.
// Only an exact compiled deferred copy can leave its spectral coordinates
// unresolved. The declaration remains outside GCRV and owns every frame sample.
inline const DeferredPointsSource* deferredSpectrumSource(const ValueDescriptor& value,
    const OperationRecord& op, const spectrum::Binding& binding, const AdmissionContext& context) {
  const auto source=context.deferredPoints.find(op.inputStableId);
  if (source==context.deferredPoints.end() || binding.history.enabled || !binding.coordinates.empty()
      || op.code!=OperationCode::fieldAudioSpectrum || op.field.domain!=Domain::point
      || binding.fieldStableId!=op.stableId || op.outputStableId!=op.stableId
      || binding.sourceGeometryStableId!=op.inputStableId || binding.consumerGeometryStableId!=0
      || source->second.sourceStableId!=op.inputStableId
      || source->second.sourceStableId!=value.sourceStableId
      || source->second.sourceRevision==0 || source->second.sourceRevision!=value.sourceRevision
      || source->second.capacity==0 || source->second.capacity>64) return nullptr;
  const auto copy=std::find_if(value.operations.begin(),value.operations.end(),[&](const auto& prior) {
    return prior.code==OperationCode::copy && prior.stableId==op.inputStableId
        && prior.outputStableId==op.inputStableId && prior.inputStableId==0;
  });
  return copy==value.operations.end() ? nullptr : &source->second;
}

inline bool validateSpectrumFieldBinding(const spectrum::Binding& binding,
    const std::vector<float>& coordinates, bool deferred, std::string& error) {
  if (!deferred) return spectrum::validate(binding,coordinates.size(),error);
  // Validate unchanged authored controls using a local coordinate probe. Neither
  // the serialized binding nor the selected-source callback gets frame state.
  auto frame=binding;
  frame.coordinates=coordinates.empty() ? std::vector<float>{0} : coordinates;
  return spectrum::validate(frame,frame.coordinates.size(),error);
}

inline bool spectrumCoordinates(const CarrierData& data, CarrierKind carrier,
    const std::vector<AttributeData>& attributes, const OperationRecord& op,
    const spectrum::Binding& binding, std::vector<float>& coordinates, std::string& error,
    bool deferred = false) {
  const auto& s=op.field; const auto& p=s.values;
  if (!validFieldOperationSettings(op,error) || s.domain!=elementDomain(carrier)
      || (s.valueType!=ValueType::floatValue && s.valueType!=ValueType::vector && s.valueType!=ValueType::color)
      || s.mode>7 || (s.mode==7 && carrier!=CarrierKind::instances3D && carrier!=CarrierKind::points3D)
      || (deferred && (carrier!=CarrierKind::points3D || s.mode==5 || s.mode==6))
      || binding.firstBand>binding.lastBand || binding.lastBand>=spectrum::kBands
      || p[0]>=p[1] || std::abs(p[0])>1000000 || std::abs(p[1])>1000000 || p[2]<arbitblockb::kBandLoHz || p[3]>arbitblockb::kBandHiHz || p[2]>p[3]
      || (binding.history.enabled
          ? (deferred || p[4]<0 || p[4]>3 || std::floor(p[4])!=p[4] || p[5]>=p[6]
              || std::abs(p[5])>1000000 || std::abs(p[6])>1000000 || p[7]!=0)
          : std::any_of(p.begin()+4,p.end(),[](double v){return v!=0;}))
      || op.secondaryInputStableId!=0) {
    error="Audio Spectrum Field requires a matching element domain and bounded sampling controls"; return false;
  }
  FieldData positions, identities;
  if (!materializeElementField(data,carrier,OperationCode::positionField,positions,kMaximumFieldElements,error)) return false;
  if (positions.elements.empty()) {
    if (!deferred) { error="Audio Spectrum Field requires a nonempty retained source"; return false; }
    coordinates.clear(); return true;
  }
  const auto count=positions.elements.size();
  const AttributeData* uv=nullptr;
  if (s.mode==5 || s.mode==6) {
    uv=findNamedAttribute(attributes,"uv");
    if (!uv || uv->descriptor.domain!=s.domain || uv->descriptor.valueType!=ValueType::vector
        || uv->elements.size()!=count || std::any_of(uv->elements.begin(),uv->elements.end(),[](const auto& e){
          return !validFieldElement(e,ValueType::vector);
        })) { error="Audio Spectrum Field UV sampling requires the exact typed uv attribute"; return false; }
  }
  std::vector<double> sorted;
  if (s.mode==7) {
    if (!materializeElementField(data,carrier,OperationCode::stableIdentityField,identities,kMaximumFieldElements,error)) return false;
    for (const auto& identity : identities.elements) sorted.push_back(identity.components[0]);
    std::sort(sorted.begin(),sorted.end());
    if (std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end()) {
      error="Audio Spectrum Field indices require unique stable identities"; return false;
    }
  }
  coordinates.resize(count);
  for (std::size_t i=0;i<count;++i) {
    double coordinate=count>1 ? static_cast<double>(i)/(count-1) : 0;
    if (s.mode>=1 && s.mode<=3) coordinate=(positions.elements[i].components[s.mode-1]-p[0])/(p[1]-p[0]);
    else if (uv) coordinate=uv->elements[i].components[s.mode-5];
    else if (s.mode==7 && count>1)
      coordinate=static_cast<double>(std::lower_bound(sorted.begin(),sorted.end(),identities.elements[i].components[0])-sorted.begin())/(count-1);
    if (s.mode==4) {
      const auto frequency=p[2]+(p[3]-p[2])*std::clamp(coordinate,0.0,1.0);
      const auto bin=std::log(frequency/arbitblockb::kBandLoHz)/std::log(arbitblockb::kBandHiHz/arbitblockb::kBandLoHz)*arbitblockb::kBandCount-0.5;
      coordinate=binding.lastBand>binding.firstBand ? (bin-binding.firstBand)/(binding.lastBand-binding.firstBand) : 0;
    }
    if (!finite(coordinate)) { error="Audio Spectrum Field has a non-finite sample coordinate"; return false; }
    coordinates[i]=static_cast<float>(std::clamp(coordinate,0.0,1.0));
  }
  return true;
}

inline bool spectrumHistoryCoordinates(const CarrierData& data, CarrierKind carrier,
    const OperationRecord& op, std::vector<float>& coordinates, std::string& error) {
  const auto& p=op.field.values;
  if (op.field.domain!=elementDomain(carrier) || p[4]<0 || p[4]>3 || std::floor(p[4])!=p[4]
      || !finite(p[5]) || !finite(p[6]) || p[5]>=p[6]
      || std::abs(p[5])>1000000 || std::abs(p[6])>1000000 || p[7]!=0) {
    error="Spectral History Field requires a retained element domain and bounded row sampling"; return false;
  }
  FieldData positions;
  if (!materializeElementField(data,carrier,OperationCode::positionField,positions,kMaximumFieldElements,error)
      || positions.elements.empty()) {
    if (error.empty()) error="Spectral History Field requires nonempty retained topology";
    return false;
  }
  coordinates.resize(positions.elements.size());
  for (std::size_t i=0;i<coordinates.size();++i) {
    const auto coordinate=p[4]==0 ? (coordinates.size()>1 ? static_cast<double>(i)/(coordinates.size()-1) : 0)
        : (positions.elements[i].components[static_cast<std::size_t>(p[4])-1]-p[5])/(p[6]-p[5]);
    if (!finite(coordinate)) { error="Spectral History Field has a non-finite row coordinate"; return false; }
    coordinates[i]=static_cast<float>(std::clamp(coordinate,0.0,1.0));
  }
  return true;
}

// Row ages for retained elements that carry no field of their own. Instances and
// Audio Deformer vertices use the same axis rule as Spectral History Fields:
// index spreads evenly, and a position axis maps its bounded range onto [0, 1].
inline bool spectrumHistoryAges(const std::vector<Vec3>& positions, const spectrum::HistoryAxis& axis,
    std::vector<float>& ages, std::string& error) {
  if (positions.empty() || axis.by>3 || !finite(axis.minimum) || !finite(axis.maximum)
      || std::abs(axis.minimum)>1000000 || std::abs(axis.maximum)>1000000 || axis.minimum>=axis.maximum) {
    error="Spectral History requires nonempty retained elements and a bounded row axis"; return false;
  }
  ages.resize(positions.size());
  for (std::size_t i=0;i<positions.size();++i) {
    const auto& p=positions[i];
    const double coordinate=axis.by==0
        ? (positions.size()>1 ? static_cast<double>(i)/(positions.size()-1) : 0)
        : ((axis.by==1 ? p.x : axis.by==2 ? p.y : p.z)-axis.minimum)/(axis.maximum-axis.minimum);
    if (!finite(coordinate)) { error="Spectral History row coordinate is non-finite"; return false; }
    ages[i]=static_cast<float>(std::clamp(coordinate,0.0,1.0));
  }
  return true;
}

inline bool materializeSpectrumField(const spectrum::Binding& binding, const OperationRecord& op,
    const std::vector<float>& coordinates, FieldData& field, const RuntimeFieldEvaluation* runtime,
    std::string& error, bool deferred = false, const std::vector<float>* historyCoordinates = nullptr) {
  if (!validateSpectrumFieldBinding(binding,coordinates,deferred,error)) return false;
  // Instancer history samples per-instance rows. Deformer history reshapes
  // vertices through the deformer callback, so it cannot produce a field here.
  if (binding.target==spectrum::Target::geometryDeformation
      || (binding.target==spectrum::Target::instanceTransform && !binding.history.enabled)) {
    error="Ordered Audio Spectrum Field cannot claim a deformer target, or an instancer target without history"; return false;
  }
  spectrum::Bands bands{};
  std::vector<spectrum::Bands> rows;
  const auto& ages=historyCoordinates ? *historyCoordinates : binding.history.coordinates;
  if (binding.history.enabled && (deferred || ages.size()!=coordinates.size()
      || std::any_of(ages.begin(),ages.end(),[](float age){return !std::isfinite(age) || age<0 || age>1;}))) {
    error="Spectral History Field row coordinates disagree with retained topology"; return false;
  }
  if (runtime && binding.history.enabled) {
    if (!runtime->spectrumHistoryAt || !runtime->spectrumHistoryAt(binding,rows,error)) {
      if (error.empty()) error="Spectral History Field requires the shared project-time history";
      return false;
    }
    if (rows.size()!=binding.history.depth) { error="Spectral History Field received an invalid row count"; return false; }
  } else if (runtime && (!runtime->spectrumAt || !runtime->spectrumAt(binding,bands,error))) {
    if (error.empty()) error="Audio Spectrum Field requires the shared frame analysis";
    return false;
  }
  field.elements.resize(coordinates.size());
  for (std::size_t i=0;i<coordinates.size();++i) {
    const auto amount=!runtime ? 0 : binding.history.enabled
        ? spectrum::sample(binding,rows[static_cast<std::size_t>(std::round(ages[i]*(rows.size()-1)))],coordinates[i])
        : spectrum::sample(binding,bands,coordinates[i]);
    auto& value=field.elements[i];
    if (op.field.valueType==ValueType::floatValue) value.components={amount,0,0,0};
    else value.components={amount*binding.direction[0],amount*binding.direction[1],amount*binding.direction[2],
        op.field.valueType==ValueType::color ? 1.0 : 0.0};
    if (!validFieldElement(value,op.field.valueType)) { error="Audio Spectrum Field produced an invalid typed value"; return false; }
  }
  return true;
}

// Noise mode 1 is the wind spectrum: position input, timeline seconds input.
inline constexpr std::uint8_t kNoiseFieldWindSpectrumMode = 1;
// Noise mode 2 is the ocean (GeometryOceanSpectrum.h): position input, timeline
// input, vector displacement output. Mode 3 is its foam: position and
// displacement inputs, foam output from 0 to 1.
inline constexpr std::uint8_t kNoiseFieldOceanMode = 2;
inline constexpr std::uint8_t kNoiseFieldOceanFoamMode = 3;

inline std::pair<ValueType,ValueType> fieldInputTypes(OperationCode code,
                                                     ValueType outputType,
                                                     unsigned mode = 0) noexcept {
  if (code==OperationCode::fieldVectorMath || code==OperationCode::fieldVectorMeasure)
    return {ValueType::vector,ValueType::vector};
  if (code==OperationCode::fieldNoise && (mode==kNoiseFieldWindSpectrumMode || mode==kNoiseFieldOceanMode))
    return {ValueType::vector,ValueType::floatValue};
  if (code==OperationCode::fieldNoise && mode==kNoiseFieldOceanFoamMode) return {ValueType::vector,ValueType::vector};
  if (code==OperationCode::fieldNoise || code==OperationCode::fieldGradient || code==OperationCode::fieldFalloff
      || code==OperationCode::fieldScoreSample)
    return {ValueType::vector,ValueType::unspecified};
  if (code==OperationCode::fieldRandom || code==OperationCode::fieldIntegerToFloat)
    return {ValueType::integer,ValueType::unspecified};
  if (code==OperationCode::fieldBoolean) return {ValueType::boolean,ValueType::boolean};
  if (code==OperationCode::fieldMix || code==OperationCode::fieldSelect) return {outputType,outputType};
  return {ValueType::floatValue,ValueType::floatValue};
}

inline double stableRandom(std::uint64_t identity, std::uint64_t seed) noexcept {
  auto value=identity ^ (seed+0x9e3779b97f4a7c15ull);
  value=(value^(value>>30))*0xbf58476d1ce4e5b9ull;
  value=(value^(value>>27))*0x94d049bb133111ebull;
  value^=value>>31;
  return static_cast<double>(value>>11)*0x1.0p-53;
}

inline double latticeNoise(double x, double y, double z, std::uint64_t seed) noexcept {
  const auto ix=static_cast<std::int64_t>(std::floor(x));
  const auto iy=static_cast<std::int64_t>(std::floor(y));
  const auto iz=static_cast<std::int64_t>(std::floor(z));
  const auto smooth=[](double t){return t*t*(3-2*t);};
  const auto u=smooth(x-ix),v=smooth(y-iy),w=smooth(z-iz);
  double result=0;
  for (int dz=0;dz<2;++dz) for (int dy=0;dy<2;++dy) for (int dx=0;dx<2;++dx) {
    const auto hash=static_cast<std::uint64_t>(ix+dx)*0x9e3779b185ebca87ull
        ^ static_cast<std::uint64_t>(iy+dy)*0xc2b2ae3d27d4eb4full
        ^ static_cast<std::uint64_t>(iz+dz)*0x165667b19e3779f9ull;
    result+=(dx ? u : 1-u)*(dy ? v : 1-v)*(dz ? w : 1-w)*stableRandom(hash,seed);
  }
  return result;
}

// Wind spectrum heights, evaluated on the CPU at each frame replay like every
// other Geometry Core field. Scene append then derives normals from the triangles.
//   Phillips: P(k) = exp(-1/(k L)^2) / k^4 * cos^2(angle from wind), L = V^2 / g.
//   Log-spaced rings give dkx dky proportional to k^2, so amplitude(k) is
//   exp(-1/(2 (k L)^2)) * cos(angle from wind) / k before normalization.
//   height(x,y,t) = sum amplitude * sin(kx x + ky y - sqrt(g k) t + phase).
// Waves run within 90 degrees of the heading, and the set is scaled to a root
// mean square height of rmsHeight. Phases are fixed, so replay at a time is exact.
inline constexpr double kWindSpectrumGravity = 9.81;
inline constexpr double kWindSpectrumMinimumSpeed = 0.1;
inline constexpr double kWindSpectrumMaximumSpeed = 60;
inline constexpr double kWindSpectrumMaximumHeadingDegrees = 360;
inline constexpr double kWindSpectrumReferenceSpeed = 10;
inline constexpr std::size_t kWindSpectrumRings = 12;
inline constexpr std::size_t kWindSpectrumDirections = 8;

struct WindSpectrumWave final { double kx=0, ky=0, amplitude=0, omega=0, phase=0; };

// Open-sea height grows with the square of wind speed, so the authored
// amplitude is the root mean square height at the 10 m/s reference wind.
inline double windSpectrumRmsHeight(double amplitude, double windSpeed) noexcept {
  const auto ratio=windSpeed/kWindSpectrumReferenceSpeed;
  return amplitude*ratio*ratio;
}

inline bool validWindSpectrumSettings(double windSpeed, double headingDegrees, std::string& error) {
  if (!(windSpeed>=kWindSpectrumMinimumSpeed && windSpeed<=kWindSpectrumMaximumSpeed)) {
    error="Wind Spectrum refused: wind speed "+std::to_string(windSpeed)+" m/s is outside "
        +std::to_string(kWindSpectrumMinimumSpeed)+" to "+std::to_string(kWindSpectrumMaximumSpeed)
        +" m/s. Use 10 m/s for a moderate sea.";
    return false;
  }
  if (!(std::abs(headingDegrees)<=kWindSpectrumMaximumHeadingDegrees)) {
    error="Wind Spectrum refused: wind heading "+std::to_string(headingDegrees)+" degrees is outside -"
        +std::to_string(kWindSpectrumMaximumHeadingDegrees)+" to "+std::to_string(kWindSpectrumMaximumHeadingDegrees)
        +" degrees. Use 0 degrees for wind along +X.";
    return false;
  }
  return true;
}

// Sea shape controls. Zero in every field keeps the original open deep-water sea.
// alignment adds 2*alignment to the cosine power of the directional spread, so
// waves crossing or opposing the wind fade. depth is the water depth in metres;
// shallower water slows long waves by the finite-depth dispersion relation.
// loopPeriod rounds each wave's frequency to whole cycles of that period, so the
// sea repeats seamlessly; a wave that would round to zero keeps one cycle.
struct WindSpectrumShape final { double alignment=0, depth=0, loopPeriod=0; };

inline constexpr double kWindSpectrumMaximumAlignment = 10;
inline constexpr double kWindSpectrumMinimumDepth = 0.01;
inline constexpr double kWindSpectrumMaximumDepth = 1000000;
inline constexpr double kWindSpectrumMaximumLoopPeriod = 86400;

inline bool validWindSpectrumShape(const WindSpectrumShape& shape, std::string& error) {
  if (!(shape.alignment>=0 && shape.alignment<=kWindSpectrumMaximumAlignment)) {
    error="Wind Spectrum refused: wave alignment "+std::to_string(shape.alignment)+" is outside 0 to "
        +std::to_string(kWindSpectrumMaximumAlignment)+". Use 0 for the open-sea spread.";
    return false;
  }
  if (!(shape.depth==0 || (shape.depth>=kWindSpectrumMinimumDepth && shape.depth<=kWindSpectrumMaximumDepth))) {
    error="Wind Spectrum refused: water depth "+std::to_string(shape.depth)+" m is outside "
        +std::to_string(kWindSpectrumMinimumDepth)+" to "+std::to_string(kWindSpectrumMaximumDepth)
        +" m. Use 0 for deep water.";
    return false;
  }
  if (!(shape.loopPeriod>=0 && shape.loopPeriod<=kWindSpectrumMaximumLoopPeriod)) {
    error="Wind Spectrum refused: loop period "+std::to_string(shape.loopPeriod)+" s is outside 0 to "
        +std::to_string(kWindSpectrumMaximumLoopPeriod)+" s. Use 0 for a sea that does not repeat.";
    return false;
  }
  return true;
}

inline bool windSpectrumWaves(double windSpeed, double headingDegrees, double rmsHeight,
                              double minimumWavelength, double maximumWavelength,
                              std::vector<WindSpectrumWave>& waves, std::string& error,
                              const WindSpectrumShape& shape = {}) {
  constexpr double tau=6.283185307179586476925286766559;
  if (!validWindSpectrumSettings(windSpeed,headingDegrees,error) || !validWindSpectrumShape(shape,error)) return false;
  if (!finite(rmsHeight) || rmsHeight<0 || !finite(minimumWavelength) || !finite(maximumWavelength)
      || minimumWavelength<=0 || maximumWavelength<minimumWavelength) {
    error="Wind Spectrum requires a finite height and an ordered positive wavelength band"; return false;
  }
  const auto scale=windSpeed*windSpeed/kWindSpectrumGravity;
  const auto heading=headingDegrees*tau/360;
  waves.clear(); waves.reserve(kWindSpectrumRings*kWindSpectrumDirections);
  double energy=0;
  for (std::size_t ring=0;ring<kWindSpectrumRings;++ring) {
    const auto wavelength=minimumWavelength*std::pow(maximumWavelength/minimumWavelength,(ring+0.5)/kWindSpectrumRings);
    const auto k=tau/wavelength;
    for (std::size_t direction=0;direction<kWindSpectrumDirections;++direction) {
      const auto id=ring*kWindSpectrumDirections+direction;
      const auto offset=((direction+0.5+0.8*(stableRandom(id,1)-0.5))/kWindSpectrumDirections-0.5)*tau/2;
      const auto angle=heading+offset;
      const auto spread=shape.alignment==0 ? std::cos(offset) : std::pow(std::cos(offset),1+2*shape.alignment);
      const auto amplitude=std::exp(-0.5/((k*scale)*(k*scale)))*spread/k;
      auto omega=std::sqrt(kWindSpectrumGravity*k*(shape.depth>0 ? std::tanh(k*shape.depth) : 1.0));
      if (shape.loopPeriod>0) {
        const auto cycles=std::max(1.0,std::round(omega*shape.loopPeriod/tau));
        omega=cycles*tau/shape.loopPeriod;
      }
      waves.push_back({k*std::cos(angle),k*std::sin(angle),amplitude,omega,tau*stableRandom(id,2)});
      energy+=amplitude*amplitude;
    }
  }
  // A band far below the Phillips cutoff has no energy, which leaves the surface flat.
  const auto gain=energy>0 ? rmsHeight/std::sqrt(energy/2) : 0.0;
  for (auto& wave : waves) wave.amplitude*=gain;
  error.clear(); return true;
}

inline double windSpectrumHeight(const std::vector<WindSpectrumWave>& waves,
                                 double x, double y, double seconds) noexcept {
  double height=0;
  for (const auto& wave : waves)
    height+=wave.amplitude*std::sin(wave.kx*x+wave.ky*y-wave.omega*seconds+wave.phase);
  return height;
}

// A 1024 by 1024 sea costs about 100 million sines per replay at the formula above,
// and admission replays it on the message thread. Over a row-major XY grid at one
// time, sin(A+B)=sin(A)cos(B)+cos(A)sin(B) lets each wave reuse one table per column
// and one sine pair per row. Heights match windSpectrumHeight up to rounding. Any
// other layout returns false and leaves heights empty, so the caller uses the formula.
inline bool windSpectrumGridHeights(const std::vector<WindSpectrumWave>& waves,
                                    const std::vector<FieldElement>& positions,
                                    const std::vector<FieldElement>& times,
                                    std::vector<double>& heights) {
  heights.clear();
  const auto count=positions.size();
  if (count<4 || (times.size()!=1 && times.size()!=count)) return false;
  const auto seconds=times[0].components[0];
  if (!std::all_of(times.begin(),times.end(),[&](const auto& t){return t.components[0]==seconds;})) return false;
  std::size_t columns=1;
  while (columns<count && positions[columns].components[1]==positions[0].components[1]) ++columns;
  const auto rows=count/columns;
  if (columns<2 || rows<2 || columns*rows!=count) return false;
  for (std::size_t i=0;i<count;++i) {
    const auto column=i%columns;
    if (positions[i].components[0]!=positions[column].components[0]
        || positions[i].components[1]!=positions[i-column].components[1]) return false;
  }
  heights.assign(count,0.0);
  std::vector<double> columnSin(columns),columnCos(columns);
  for (const auto& wave : waves) {
    for (std::size_t column=0;column<columns;++column) {
      const auto angle=wave.kx*positions[column].components[0];
      columnSin[column]=std::sin(angle); columnCos[column]=std::cos(angle);
    }
    for (std::size_t row=0;row<rows;++row) {
      const auto rowAngle=wave.ky*positions[row*columns].components[1]-wave.omega*seconds+wave.phase;
      const auto rowSin=wave.amplitude*std::sin(rowAngle),rowCos=wave.amplitude*std::cos(rowAngle);
      auto* out=heights.data()+row*columns;
      for (std::size_t column=0;column<columns;++column) out[column]+=columnSin[column]*rowCos+columnCos[column]*rowSin;
    }
  }
  return true;
}

} // namespace videowire::geometry
#include "GeometryOceanSpectrum.h"
namespace videowire::geometry {

inline bool materializeConstructedField(const OperationRecord& op, const FieldData* a,
                                        const FieldData* b, const FieldData* c,
                                        FieldData& result, std::size_t capacity,
                                        std::string& error,
                                        const RuntimeFieldEvaluation* runtime = nullptr) {
  if (!validFieldOperationSettings(op,error)) return false;
  const auto& s=op.field;
  const auto& p=s.values;
  const auto code=op.code;
  if (code==OperationCode::fieldConstant) {
    if (a || b || c || s.mode!=0 || p[4]<1 || p[4]>capacity || p[4]>kMaximumFieldElements
        || std::floor(p[4])!=p[4] || p[5]!=0 || p[6]!=0 || p[7]!=0) {
      error="Constant field count exceeds its capacity"; return false;
    }
    FieldElement value{{p[0],p[1],p[2],p[3]}};
    if (!validFieldElement(value,s.valueType)) { error="Constant field value does not match its type"; return false; }
    result.elements.assign(static_cast<std::size_t>(p[4]),value); return true;
  }
  if (!a || a->elements.empty()) { error="Field operation requires a nonempty input"; return false; }
  if (code==OperationCode::fieldTimeline
      && (s.mode>3 || p[0]<=0 || p[0]>86400 || std::abs(p[1])>1000000
          || std::abs(p[2])>1000 || std::abs(p[3])>1000 || std::abs(p[4])>1000
          || std::abs(p[5])>1000 || p[6]!=0 || p[7]!=0 || b || c
          || (runtime && !finite(runtime->timelineSeconds)))) {
    error="Timeline Field requires a finite time and bounded duration, phase, gain and bias"; return false;
  }
  if (code==OperationCode::fieldScoreSample
      && (s.mode>=scorefield::kSampleFieldModeCount || p[0]<=0|| p[0]>10000 || std::abs(p[1])>1000 || std::abs(p[2])>1000
          || std::abs(p[3])>10000 || p[4]<1 || p[4]>kMaximumStableId || std::floor(p[4])!=p[4]
          || p[5]!=0 || p[6]!=0 || p[7]!=0 || b || c)) {
    error="Score Sample Field requires a score source and bounded position samples"; return false;
  }
  const auto count=std::max({a->elements.size(),b ? b->elements.size() : 1,c ? c->elements.size() : 1});
  if (count>capacity || count>kMaximumFieldElements
      || (a->elements.size()!=1 && a->elements.size()!=count)
      || (b && b->elements.size()!=1 && b->elements.size()!=count)
      || (c && c->elements.size()!=1 && c->elements.size()!=count)) {
    error="Field operation inputs must have equal cardinality or one broadcast element"; return false;
  }
  const bool windSpectrum=code==OperationCode::fieldNoise && s.mode==kNoiseFieldWindSpectrumMode;
  std::vector<WindSpectrumWave> windWaves;
  if (windSpectrum) {
    if (!b || c) {
      error="Wind Spectrum requires positions and one timeline field"; return false;
    }
    if (!windSpectrumWaves(p[0],p[1],p[2],p[3],p[4],windWaves,error,{p[5],p[6],p[7]})) return false;
  }
  const bool ocean=code==OperationCode::fieldNoise && s.mode==kNoiseFieldOceanMode;
  const bool oceanFoam=code==OperationCode::fieldNoise && s.mode==kNoiseFieldOceanFoamMode;
  OceanSettings oceanSettings;
  std::vector<WindSpectrumWave> oceanWaveSet;
  if (ocean && (!b || !oceanSettingsFromValues(p,oceanSettings,error) || !oceanWaves(oceanSettings,oceanWaveSet,error))) {
    if (error.empty()) error="Ocean requires positions and one timeline field";
    return false;
  }
  // An ocean footprint input turns on wave detail filtering (GeometryOceanSpectrum.h).
  std::vector<double> oceanFootprint;
  if (ocean && c && !oceanFootprints(a->elements,c->elements,oceanFootprint,error)) return false;
  if (oceanFoam && (!b || c || std::any_of(p.begin()+1,p.end(),[](double v){return v!=0;}))) {
    error="Ocean foam requires positions, one displacement field and a coverage"; return false;
  }
  const auto types=fieldInputTypes(code,s.valueType,s.mode);
  const auto outputType=code==OperationCode::fieldVectorMath || code==OperationCode::fieldComposeVector
      ? ValueType::vector : code==OperationCode::fieldCompare || code==OperationCode::fieldBoolean
      ? ValueType::boolean : ocean ? ValueType::vector : code==OperationCode::fieldMix || code==OperationCode::fieldSelect
      ? s.valueType : code==OperationCode::fieldFloatToInteger ? ValueType::integer : ValueType::floatValue;
  if (s.valueType!=outputType || ((code==OperationCode::fieldMix || code==OperationCode::fieldSelect)
      && s.valueType!=ValueType::floatValue && s.valueType!=ValueType::vector && s.valueType!=ValueType::color)) {
    error="Field operation output type is unsupported"; return false;
  }
  result.elements.assign(count,FieldElement{});
  std::vector<double> windGrid;
  if (windSpectrum) windSpectrumGridHeights(windWaves,a->elements,b->elements,windGrid);
  std::vector<std::array<double,3>> oceanGrid;
  std::vector<double> foam;
  if (ocean) oceanGridDisplacements(oceanWaveSet,oceanSettings,a->elements,b->elements,oceanGrid,
      oceanFootprint.empty() ? nullptr : &oceanFootprint);
  if (oceanFoam && !oceanFoamValues(a->elements,b->elements,p[0],foam,error)) return false;
  for (std::size_t i=0;i<count;++i) {
    const auto& x=a->elements[a->elements.size()==1 ? 0 : i];
    const FieldElement fallback{{p[0],p[1],p[2],p[3]}};
    const auto& y=b ? b->elements[b->elements.size()==1 ? 0 : i] : fallback;
    const double factor=c ? c->elements[c->elements.size()==1 ? 0 : i].components[0] : p[4];
    auto& out=result.elements[i].components;
    if (!validFieldElement(x,types.first) || (b && !validFieldElement(y,types.second))) {
      error="Field operation input contains an invalid typed value"; return false;
    }
    const auto xv=x.components[0],yv=y.components[0];
    if (code==OperationCode::fieldTimeline) {
      const auto seconds=(runtime ? runtime->timelineSeconds : 0.0)-p[1];
      const auto cycles=seconds/p[0];
      const auto offset=cycles-std::floor(cycles)+xv*p[2]+p[3];
      const auto phase=offset-std::floor(offset);
      const auto value=s.mode==0 ? seconds : s.mode==1 ? phase
          : s.mode==2 ? std::sin(phase*6.283185307179586) : std::cos(phase*6.283185307179586);
      out[0]=value*p[4]+p[5];
    } else if (code==OperationCode::fieldScoreSample) {
      out[0]=p[2];
    } else if (code==OperationCode::fieldMath) {
      switch (s.mode) {
        case 0: out[0]=xv+yv; break; case 1: out[0]=xv-yv; break;
        case 2: out[0]=xv*yv; break; case 3: out[0]=yv==0 ? 0 : xv/yv; break;
        case 4: out[0]=std::min(xv,yv); break; case 5: out[0]=std::max(xv,yv); break;
        case 6: out[0]=std::pow(xv,yv); break; case 7: out[0]=std::sin(xv); break;
        case 8: out[0]=std::cos(xv); break; case 9: out[0]=std::abs(xv); break;
        case 10: out[0]=std::sqrt(std::max(0.0,xv)); break;
        case 11: out[0]=std::floor(xv); break; case 12: out[0]=xv-std::floor(xv); break;
        default: error="Unknown field math operation"; return false;
      }
    } else if (code==OperationCode::fieldVectorMath) {
      const auto& xx=x.components; const auto& yy=y.components;
      const auto dot=xx[0]*yy[0]+xx[1]*yy[1]+xx[2]*yy[2];
      const auto length=std::sqrt(xx[0]*xx[0]+xx[1]*xx[1]+xx[2]*xx[2]);
      for (std::size_t j=0;j<3;++j) {
        if (s.mode==0) out[j]=xx[j]+yy[j]; else if (s.mode==1) out[j]=xx[j]-yy[j];
        else if (s.mode==2) out[j]=xx[j]*yy[j]; else if (s.mode==3) out[j]=yy[j]==0 ? 0 : xx[j]/yy[j];
        else if (s.mode==4) out[j]=xx[(j+1)%3]*yy[(j+2)%3]-xx[(j+2)%3]*yy[(j+1)%3];
        else if (s.mode==5) out[j]=length==0 ? 0 : xx[j]/length;
        else if (s.mode==6) {
          const auto nn=yy[0]*yy[0]+yy[1]*yy[1]+yy[2]*yy[2];
          out[j]=nn==0 ? xx[j] : xx[j]-2*dot*yy[j]/nn;
        } else { error="Unknown vector math operation"; return false; }
      }
    } else if (code==OperationCode::fieldVectorMeasure) {
      for (std::size_t j=0;j<3;++j) {
        if (s.mode==0) out[0]+=x.components[j]*y.components[j];
        else if (s.mode==1) out[0]+=x.components[j]*x.components[j];
        else if (s.mode==2) out[0]+=std::pow(x.components[j]-y.components[j],2);
      }
      if (s.mode==1 || s.mode==2) out[0]=std::sqrt(out[0]);
      else if (s.mode>=3 && s.mode<=5) out[0]=x.components[s.mode-3];
      else if (s.mode!=0) { error="Unknown vector measure operation"; return false; }
    } else if (code==OperationCode::fieldCompare) {
      if (s.mode==0) out[0]=xv<yv; else if (s.mode==1) out[0]=xv<=yv;
      else if (s.mode==2) out[0]=xv>yv; else if (s.mode==3) out[0]=xv>=yv;
      else if (s.mode==4) out[0]=std::abs(xv-yv)<=std::abs(p[4]);
      else if (s.mode==5) out[0]=std::abs(xv-yv)>std::abs(p[4]);
      else { error="Unknown field comparison"; return false; }
    } else if (code==OperationCode::fieldBoolean) {
      if (s.mode==0) out[0]=xv!=0 && yv!=0; else if (s.mode==1) out[0]=xv!=0 || yv!=0;
      else if (s.mode==2) out[0]=(xv!=0)!=(yv!=0); else if (s.mode==3) out[0]=xv==0;
      else { error="Unknown boolean field operation"; return false; }
    } else if (code==OperationCode::fieldMix || code==OperationCode::fieldSelect) {
      if (s.mode!=0 || (code==OperationCode::fieldSelect && factor!=0 && factor!=1)) {
        error="Field selection requires a Boolean mask"; return false;
      }
      const auto t=code==OperationCode::fieldSelect ? factor : std::clamp(factor,0.0,1.0);
      for (std::size_t j=0;j<valueTypeComponents(s.valueType);++j) out[j]=x.components[j]*(1-t)+y.components[j]*t;
    } else if (code==OperationCode::fieldMapRange) {
      if (s.mode>1 || p[1]==p[0]) { error="Range mapping requires distinct input endpoints"; return false; }
      auto t=(xv-p[0])/(p[1]-p[0]); if (s.mode==1) t=std::clamp(t,0.0,1.0);
      out[0]=p[2]+t*(p[3]-p[2]);
    } else if (code==OperationCode::fieldClamp) {
      if (s.mode!=0 || p[0]>p[1]) { error="Clamp minimum exceeds maximum"; return false; }
      out[0]=std::clamp(xv,p[0],p[1]);
    } else if (ocean) {
      const auto d=oceanGrid.empty()
          ? oceanDisplacement(oceanWaveSet,x.components[0],x.components[1],oceanSeconds(oceanSettings,yv),
              oceanFootprint.empty() ? 0.0 : oceanFootprint[i]) : oceanGrid[i];
      out={d[0],d[1],d[2],0};
    } else if (oceanFoam) {
      out[0]=foam[i];
    } else if (windSpectrum) {
      out[0]=windGrid.empty() ? windSpectrumHeight(windWaves,x.components[0],x.components[1],yv) : windGrid[i];
    } else if (code==OperationCode::fieldNoise) {
      const auto px=x.components[0]*p[0]+p[2],py=x.components[1]*p[0]+p[3],pz=x.components[2]*p[0]+p[4];
      if (s.mode!=0 || p[1]<0 || p[1]>4294967295.0 || std::floor(p[1])!=p[1]
          || !finite(px) || !finite(py) || !finite(pz)
          || std::max({std::abs(px),std::abs(py),std::abs(pz)})>1e12) {
        error="Noise coordinates or seed exceed their bounds"; return false;
      }
      out[0]=latticeNoise(px,py,pz,static_cast<std::uint64_t>(p[1]));
    } else if (code==OperationCode::fieldGradient || code==OperationCode::fieldFalloff) {
      const auto dx=x.components[0]-p[0],dy=x.components[1]-p[1],dz=x.components[2]-p[2];
      if (code==OperationCode::fieldGradient) {
        if (s.mode>2 || p[4]==0) { error="Gradient axis or extent is invalid"; return false; }
        out[0]=((s.mode==0 ? dx : s.mode==1 ? dy : dz)-p[3])/p[4];
      } else {
        if (s.mode>1 || p[3]<=0 || p[4]<=0 || p[4]>32) { error="Falloff radius or power is invalid"; return false; }
        const auto distance=s.mode==0 ? std::sqrt(dx*dx+dy*dy+dz*dz) : std::max({std::abs(dx),std::abs(dy),std::abs(dz)});
        out[0]=std::pow(std::clamp(1-distance/p[3],0.0,1.0),p[4]);
      }
    } else if (code==OperationCode::fieldRandom) {
      if (s.mode!=0 || xv<1 || xv>static_cast<double>(kMaximumStableId)
          || p[0]<0 || p[0]>4294967295.0 || std::floor(p[0])!=p[0]) {
        error="Random field requires stable identities and a bounded integer seed"; return false;
      }
      out[0]=p[1]+(p[2]-p[1])*stableRandom(static_cast<std::uint64_t>(xv),static_cast<std::uint64_t>(p[0]));
    } else if (code==OperationCode::fieldComposeVector) {
      if (s.mode>1) { error="Unknown vector construction operation"; return false; }
      if (s.mode==0) out={xv,yv,factor,0};
      else out={xv*p[0],xv*p[1],xv*p[2],0};
    } else if (code==OperationCode::fieldIntegerToFloat && s.mode==0) out[0]=xv;
    else if (code==OperationCode::fieldFloatToInteger && s.mode<=2)
      out[0]=s.mode==0 ? std::floor(xv) : s.mode==1 ? std::ceil(xv) : std::round(xv);
    else { error="Unsupported field construction operation"; return false; }
    if (!validFieldElement(result.elements[i],s.valueType)) {
      error="Field evaluation produced a non-finite or mistyped value"; return false;
    }
  }
  if (code==OperationCode::fieldScoreSample && runtime) {
    if (!runtime->scoreAt || !runtime->scoreAt(op,*a,result,error)) {
      if (error.empty()) error="Score Sample Field requires the canonical score frame";
      return false;
    }
    if (result.elements.size()!=count || std::any_of(result.elements.begin(),result.elements.end(),
        [&](const auto& value){return !validFieldElement(value,s.valueType);})) {
      error="Score Sample Field returned invalid values or cardinality"; return false;
    }
  }
  error.clear(); return true;
}

// Triangle order is retained in the immutable mesh. Equal-distance hits choose
// the first triangle, and degenerate triangles do not participate.
inline bool materializeSurfaceField(const CarrierData& source, CarrierKind sourceCarrier,
                                     const GeometryData& target, const OperationRecord& op,
                                     FieldData& result, std::size_t capacity, std::string& error) {
  using D3=std::array<double,3>;
  const auto sub=[](D3 a,D3 b)->D3 { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; };
  const auto add=[](D3 a,D3 b,double t)->D3 { return {a[0]+t*b[0],a[1]+t*b[1],a[2]+t*b[2]}; };
  const auto dot=[](D3 a,D3 b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; };
  const auto cross=[](D3 a,D3 b)->D3 { return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; };
  const auto point=[&](std::uint32_t i)->D3 { const auto& p=target.positions[i]; return {p.x,p.y,p.z}; };
  const bool ray=op.code>=OperationCode::fieldRaycastDistance;
  const auto type=op.code==OperationCode::fieldSurfaceOffset || op.code==OperationCode::fieldRaycastPosition
      ? ValueType::vector : op.code==OperationCode::fieldRaycastHit ? ValueType::boolean
      : op.code==OperationCode::fieldRaycastFace ? ValueType::integer : ValueType::floatValue;
  const auto& p=op.field.values;
  D3 direction{p[0],p[1],p[2]}; const auto length=std::sqrt(dot(direction,direction));
  if (!validFieldOperationSettings(op,error) || !isSurfaceFieldOperation(op.code)
      || op.field.domain!=elementDomain(sourceCarrier) || op.field.mode!=0 || op.field.valueType!=type
      || target.positions.empty() || target.positions.size()>kMaximumVertices
      || target.indices.empty() || target.indices.size()>kMaximumIndices || target.indices.size()%3!=0
      || (ray && (length==0 || !finite(length) || p[3]<=0 || p[3]>1000000))
      || (!ray && std::any_of(p.begin(),p.end(),[](double v){return v!=0;}))
      || (ray && std::any_of(p.begin()+4,p.end(),[](double v){return v!=0;}))) {
    error="Surface query requires triangle geometry and a bounded nonzero ray direction and distance"; return false;
  }
  for (const auto& vertex : target.positions)
    if (!finite(vertex)) { error="Surface query target has a non-finite position"; return false; }
  for (const auto index : target.indices)
    if (index>=target.positions.size()) { error="Surface query target has an out-of-range triangle index"; return false; }
  FieldData positions;
  if (!materializeElementField(source,sourceCarrier,OperationCode::positionField,positions,capacity,error)) return false;
  constexpr std::size_t maximumTests=16u*1024u*1024u;
  const auto triangles=target.indices.size()/3;
  if (positions.elements.size()>maximumTests/triangles) {
    error="Surface query exceeds 16777216 triangle tests; reduce source or target geometry"; return false;
  }
  if (ray) for (auto& axis : direction) axis/=length;
  const auto closest=[&](D3 q,D3 a,D3 b,D3 c)->D3 {
    const auto ab=sub(b,a),ac=sub(c,a),ap=sub(q,a);
    const auto d1=dot(ab,ap),d2=dot(ac,ap);
    if (d1<=0 && d2<=0) return a;
    const auto bp=sub(q,b); const auto d3=dot(ab,bp),d4=dot(ac,bp);
    if (d3>=0 && d4<=d3) return b;
    const auto vc=d1*d4-d3*d2;
    if (vc<=0 && d1>=0 && d3<=0) return add(a,ab,d1/(d1-d3));
    const auto cp=sub(q,c); const auto d5=dot(ab,cp),d6=dot(ac,cp);
    if (d6>=0 && d5<=d6) return c;
    const auto vb=d5*d2-d1*d6;
    if (vb<=0 && d2>=0 && d6<=0) return add(a,ac,d2/(d2-d6));
    const auto va=d3*d6-d5*d4;
    if (va<=0 && d4>=d3 && d5>=d6) return add(b,sub(c,b),(d4-d3)/((d4-d3)+(d5-d6)));
    return add(add(a,ab,vb/(va+vb+vc)),ac,vc/(va+vb+vc));
  };
  result.elements.assign(positions.elements.size(),FieldElement{});
  for (std::size_t i=0;i<positions.elements.size();++i) {
    const auto& origin=positions.elements[i].components; D3 q{origin[0],origin[1],origin[2]};
    double best=ray ? p[3] : std::numeric_limits<double>::infinity();
    D3 hit=ray ? add(q,direction,p[3]) : q; std::int64_t face=-1;
    for (std::size_t j=0;j<triangles;++j) {
      const auto a=point(target.indices[j*3]),b=point(target.indices[j*3+1]),c=point(target.indices[j*3+2]);
      const auto ab=sub(b,a),ac=sub(c,a),normal=cross(ab,ac);
      if (dot(normal,normal)==0) continue;
      double distance=0; D3 candidate{};
      if (ray) {
        const auto h=cross(direction,ac); const auto determinant=dot(ab,h);
        if (std::abs(determinant)<=1e-12*std::sqrt(dot(ab,ab)*dot(ac,ac))) continue;
        const auto s=sub(q,a); const auto u=dot(s,h)/determinant;
        if (u<0 || u>1) continue;
        const auto v=dot(direction,cross(s,ab))/determinant;
        if (v<0 || u+v>1) continue;
        distance=dot(ac,cross(s,ab))/determinant;
        if (distance<0 || distance>p[3]) continue;
        candidate=add(q,direction,distance);
      } else { candidate=closest(q,a,b,c); const auto delta=sub(candidate,q); distance=dot(delta,delta); }
      if (distance<best || (distance==best && face<0)) { best=distance; hit=candidate; face=static_cast<std::int64_t>(j); }
    }
    if (!ray && face<0) { error="Nearest surface requires at least one nondegenerate triangle"; return false; }
    auto& out=result.elements[i].components;
    if (op.code==OperationCode::fieldSurfaceDistance) out[0]=std::sqrt(best);
    else if (op.code==OperationCode::fieldSurfaceOffset) { const auto delta=sub(hit,q); out={delta[0],delta[1],delta[2],0}; }
    else if (op.code==OperationCode::fieldRaycastPosition) out={hit[0],hit[1],hit[2],0};
    else if (op.code==OperationCode::fieldRaycastHit) out[0]=face>=0;
    else if (op.code==OperationCode::fieldRaycastFace) out[0]=static_cast<double>(face);
    else out[0]=best;
    if (!validFieldElement(result.elements[i],type)) { error="Surface query produced an invalid result"; return false; }
  }
  error.clear(); return true;
}

inline bool materializeNearestField(const CarrierData& source, CarrierKind sourceCarrier,
                                     const GeometryData& target,
                                     const std::vector<AttributeData>& targetAttributes,
                                     const OperationRecord& op, FieldData& result,
                                     std::size_t capacity, std::string& error) {
  if (isSurfaceFieldOperation(op.code))
    return materializeSurfaceField(source,sourceCarrier,target,op,result,capacity,error);
  if (!validFieldOperationSettings(op,error) || !isNearestFieldOperation(op.code)
      || op.field.domain!=elementDomain(sourceCarrier) || op.field.mode!=0
      || target.positions.empty() || target.positions.size()>kMaximumVertices
      || target.vertexIds.size()!=target.positions.size()) {
    error="Nearest vertex field requires a nonempty retained target mesh"; return false;
  }
  FieldData positions;
  if (!materializeElementField(source,sourceCarrier,OperationCode::positionField,positions,capacity,error)) return false;
  const auto* attribute=op.code==OperationCode::fieldSampleNearest
      ? findNamedAttribute(targetAttributes,op.field.attributeName) : nullptr;
  const auto type=op.code==OperationCode::fieldNearestDirection ? ValueType::vector
      : op.code==OperationCode::fieldNearestIndex ? ValueType::integer
      : op.code==OperationCode::fieldSampleNearest ? op.field.valueType : ValueType::floatValue;
  if (op.field.valueType!=type || (op.code==OperationCode::fieldSampleNearest
      && (!attribute || attribute->descriptor.domain!=Domain::vertex || attribute->descriptor.valueType!=type
          || attribute->elements.size()!=target.positions.size()))) {
    error="Nearest vertex attribute is missing or has the wrong domain or type"; return false;
  }
  struct TreeNode { std::size_t index; int left=-1,right=-1; unsigned axis=0; };
  std::vector<std::size_t> order(target.positions.size());
  std::iota(order.begin(),order.end(),0);
  std::vector<TreeNode> tree; tree.reserve(order.size());
  const auto coordinate=[&](std::size_t index,unsigned axis) {
    const auto& p=target.positions[index]; return axis==0 ? p.x : axis==1 ? p.y : p.z;
  };
  std::set<StableId> identities;
  for (std::size_t i=0;i<target.positions.size();++i)
    if (!finite(target.positions[i]) || !validStableId(target.vertexIds[i]) || !identities.insert(target.vertexIds[i]).second) {
      error="Nearest vertex target requires finite positions and unique stable identities"; return false;
    }
  const auto build=[&](auto&& self,std::size_t begin,std::size_t end,unsigned axis)->int {
    if (begin==end) return -1;
    const auto mid=begin+(end-begin)/2;
    std::nth_element(order.begin()+begin,order.begin()+mid,order.begin()+end,[&](auto a,auto b) {
      if (coordinate(a,axis)!=coordinate(b,axis)) return coordinate(a,axis)<coordinate(b,axis);
      return target.vertexIds[a]<target.vertexIds[b];
    });
    const auto node=static_cast<int>(tree.size()); tree.push_back({order[mid],-1,-1,axis});
    const auto left=self(self,begin,mid,(axis+1)%3),right=self(self,mid+1,end,(axis+1)%3);
    tree[node].left=left; tree[node].right=right; return node;
  };
  const auto root=build(build,0,order.size(),0);
  constexpr std::size_t maximumVisits=16u*1024u*1024u;
  std::size_t visits=0;
  result.elements.assign(positions.elements.size(),FieldElement{});
  for (std::size_t i=0;i<positions.elements.size();++i) {
    const auto& p=positions.elements[i].components;
    double best=std::numeric_limits<double>::infinity(); std::size_t nearest=0;
    const auto search=[&](auto&& self,int index)->bool {
      if (index<0) return true;
      if (++visits>maximumVisits) return false;
      const auto& node=tree[static_cast<std::size_t>(index)]; const auto& q=target.positions[node.index];
      const auto dx=double(q.x)-p[0],dy=double(q.y)-p[1],dz=double(q.z)-p[2];
      const auto distance=dx*dx+dy*dy+dz*dz;
      if (distance<best || (distance==best && target.vertexIds[node.index]<target.vertexIds[nearest])) {
        best=distance; nearest=node.index;
      }
      const auto split=p[node.axis]-coordinate(node.index,node.axis);
      if (!self(self,split<0 ? node.left : node.right)) return false;
      return split*split>best || self(self,split<0 ? node.right : node.left);
    };
    if (!search(search,root)) { error="Nearest vertex query exceeds the bounded search budget"; return false; }
    auto& out=result.elements[i].components;
    if (op.code==OperationCode::fieldProximity) out[0]=std::sqrt(best);
    else if (op.code==OperationCode::fieldNearestIndex) out[0]=static_cast<double>(nearest);
    else if (op.code==OperationCode::fieldNearestDirection) {
      const auto& q=target.positions[nearest]; out={double(q.x)-p[0],double(q.y)-p[1],double(q.z)-p[2],0};
    } else result.elements[i]=attribute->elements[nearest];
    if (!validFieldElement(result.elements[i],type)) { error="Nearest vertex result is invalid"; return false; }
  }
  error.clear(); return true;
}

// Sample Index takes one source attribute element per whole-number index.
// Fractional, negative, non-finite or out-of-range indices are refused.
inline bool materializeSampleIndexField(const AttributeData* attribute, const FieldData& index,
                                        const OperationRecord& op, FieldData& result,
                                        std::size_t capacity, std::string& error) {
  if (!validFieldOperationSettings(op,error) || op.code!=OperationCode::fieldSampleIndex || op.field.mode!=0
      || !attribute || attribute->descriptor.domain!=op.field.domain
      || attribute->descriptor.valueType!=op.field.valueType || attribute->elements.empty()
      || index.elements.empty() || index.elements.size()>capacity || index.elements.size()>kMaximumFieldElements) {
    error="Sample index requires a same-domain source attribute and a bounded Integer index field"; return false;
  }
  FieldData sampled;
  sampled.elements.resize(index.elements.size());
  for (std::size_t i=0;i<index.elements.size();++i) {
    const auto position=index.elements[i].components[0];
    if (!finite(position) || std::floor(position)!=position || position<0
        || position>=static_cast<double>(attribute->elements.size())) {
      error="Sample index is outside the source attribute"; return false;
    }
    sampled.elements[i]=attribute->elements[static_cast<std::size_t>(position)];
    if (!validFieldElement(sampled.elements[i],op.field.valueType)) {
      error="Sample index produced an invalid value"; return false;
    }
  }
  result=std::move(sampled);
  error.clear(); return true;
}

inline bool materializeStoredAttribute(std::vector<AttributeData>& attributes,
                                        std::size_t count, const FieldData& field,
                                        const OperationRecord& op, std::string& error) {
  if (op.code==OperationCode::setNormal && op.parameters[0]!=0 && op.parameters[0]!=1) {
    error="Set Normal normalize flag must be 0 or 1"; return false;
  }
  if (!validFieldOperationSettings(op,error)) return false;
  if (op.field.mode!=0 || count>kMaximumFieldElements
      || (field.elements.size()!=count && field.elements.size()!=1)
      || (op.code==OperationCode::setNormal && (op.field.attributeName!="normal"
          || op.field.domain!=Domain::vertex || op.field.valueType!=ValueType::vector))) {
    error="Stored attribute field does not match its element domain"; return false;
  }
  AttributeData attribute;
  attribute.descriptor={namedAttributeIdentity(op.field.attributeName,op.field.domain,op.field.valueType),
      op.field.attributeName,op.field.valueType,op.field.domain,Interpolation::nearest};
  attribute.elements.resize(count);
  for (std::size_t i=0;i<count;++i) {
    auto element=field.elements[field.elements.size()==1 ? 0 : i];
    if (!validFieldElement(element,op.field.valueType)) { error="Stored attribute contains an invalid value"; return false; }
    if (op.code==OperationCode::setNormal && op.parameters[0]==0) {
      auto& n=element.components; const auto length=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
      if (!finite(length) || length==0) { error="Set Normal requires nonzero vectors"; return false; }
      for (int axis=0;axis<3;++axis) n[axis]/=length;
    }
    attribute.elements[i]=element;
  }
  auto found=std::find_if(attributes.begin(),attributes.end(),[&](const auto& a) {
    return a.descriptor.name==op.field.attributeName;
  });
  if (found==attributes.end()) {
    if (attributes.size()>=kMaximumAttributes) { error="Named attribute budget is full"; return false; }
    attributes.push_back(std::move(attribute));
  } else *found=std::move(attribute);
  error.clear(); return true;
}
struct InstanceAppearance final {
  std::array<float,4> color{1,1,1,1};
  std::array<float,4> emission{};
  // Negative values inherit the object's material. Wire attributes stay in [0, 1].
  float metallic=-1, roughness=-1;
};

inline bool isInstanceSurfaceScalar(const std::string& name) noexcept {
  return name=="metallic" || name=="roughness";
}

inline bool validInstanceAppearanceAttributes(const std::vector<AttributeData>& attributes,
                                               std::size_t count, std::string& error) {
  for (const auto& attribute : attributes) {
    const auto& name=attribute.descriptor.name;
    const bool surfaceScalar=isInstanceSurfaceScalar(name);
    const bool selection=name=="metallicSelected" || name=="roughnessSelected";
    if (name!="color" && name!="emission" && name!="opacity" && !surfaceScalar && !selection) continue;
    const auto type=selection ? ValueType::boolean
        : name=="opacity" || surfaceScalar ? ValueType::floatValue : ValueType::color;
    if (attribute.descriptor.domain!=Domain::instance || attribute.descriptor.valueType!=type
        || attribute.elements.size()!=count) {
      error="Instance appearance requires matching instance-domain Color or Float attributes"; return false;
    }
    for (const auto& element : attribute.elements) {
      if (!validFieldElement(element,type)) { error="Instance appearance is not finite"; return false; }
      for (std::size_t channel=0;channel<valueTypeComponents(type);++channel) {
        const auto maximum=name=="opacity" || surfaceScalar || selection || channel==3 ? 1.0 : 1000000.0;
        if (element.components[channel]<0 || element.components[channel]>maximum) {
          error="Instance colors and emission must be nonnegative, with opacity and alpha between zero and one";
          return false;
        }
      }
    }
    if (selection && !findNamedAttribute(attributes,name.substr(0,name.size()-8))) {
      error="Instance material selection requires its matching material attribute"; return false;
    }
  }
  return true;
}

// Admission validates domain, type and cardinality before native draws use these values.
inline InstanceAppearance instanceAppearance(const std::vector<AttributeData>& attributes,
                                               std::size_t index) noexcept {
  InstanceAppearance result;
  if (const auto* color=findNamedAttribute(attributes,"color"))
    for (std::size_t i=0;i<4;++i) result.color[i]=static_cast<float>(color->elements[index].components[i]);
  if (const auto* emission=findNamedAttribute(attributes,"emission"))
    for (std::size_t i=0;i<3;++i) result.emission[i]=static_cast<float>(std::min(static_cast<double>(emission->elements[index].components[i]),static_cast<double>(std::numeric_limits<float>::max())));
  if (const auto* opacity=findNamedAttribute(attributes,"opacity"))
    result.color[3]*=static_cast<float>(opacity->elements[index].components[0]);
  for (const auto* name : {"metallic","roughness"})
    if (const auto* attribute=findNamedAttribute(attributes,name)) {
      const auto* selected=findNamedAttribute(attributes,std::string(name)+"Selected");
      if (!selected || selected->elements[index].components[0]!=0)
        (std::string(name)=="metallic" ? result.metallic : result.roughness)
            =static_cast<float>(attribute->elements[index].components[0]);
    }
  return result;
}

inline bool materializeInstanceAppearance(std::vector<AttributeData>& attributes,
                                          std::size_t count, const FieldData& field,
                                          const FieldData* mask, const OperationRecord& operation,
                                          std::string& error) {
  const auto& settings=operation.field;
  const auto& name=settings.attributeName;
  const bool surfaceScalar=isInstanceSurfaceScalar(name);
  if (operation.code!=OperationCode::setInstanceAppearance || settings.domain!=Domain::instance
      || (name!="color" && name!="emission" && name!="opacity" && !surfaceScalar)
      || settings.valueType!=(name=="opacity" || surfaceScalar ? ValueType::floatValue : ValueType::color)
      || count==0 || count>kMaximumInstances || (field.elements.size()!=1 && field.elements.size()!=count)
      || (mask && mask->elements.size()!=1 && mask->elements.size()!=count)
      || !validInstanceAppearanceAttributes(attributes,count,error)) {
    if (error.empty()) error="Set Instance Appearance requires a matching field and optional Boolean selection";
    return false;
  }
  const auto* previous=findNamedAttribute(attributes,name);
  const auto* previousSelection=surfaceScalar ? findNamedAttribute(attributes,name+"Selected") : nullptr;
  FieldData selected;
  selected.elements.resize(count);
  FieldData materialSelection;
  if (surfaceScalar) materialSelection.elements.resize(count);
  const FieldElement fallback{name=="color" ? std::array<double,4>{1,1,1,1}
      : name=="opacity" || name=="roughness" ? std::array<double,4>{1,0,0,0}
      : name=="metallic" ? std::array<double,4>{0,0,0,0} : std::array<double,4>{0,0,0,1}};
  for (std::size_t i=0;i<count;++i) {
    bool enabled=true;
    if (mask) {
      const auto& value=mask->elements[mask->elements.size()==1 ? 0 : i];
      if (!validFieldElement(value,ValueType::boolean)) { error="Instance appearance selection must be Boolean"; return false; }
      enabled=value.components[0]!=0;
    }
    selected.elements[i]=enabled ? field.elements[field.elements.size()==1 ? 0 : i]
        : previous ? previous->elements[i] : fallback;
    if (surfaceScalar)
      materialSelection.elements[i].components[0]=enabled || (previous
          && (!previousSelection || previousSelection->elements[i].components[0]!=0)) ? 1 : 0;
  }
  if (!materializeStoredAttribute(attributes,count,selected,operation,error)) return false;
  if (surfaceScalar) {
    auto selectionOperation=operation;
    selectionOperation.field.attributeName=name+"Selected";
    selectionOperation.field.valueType=ValueType::boolean;
    if (!materializeStoredAttribute(attributes,count,materialSelection,selectionOperation,error)) return false;
  }
  return validInstanceAppearanceAttributes(attributes,count,error);
}
} // namespace videowire::geometry
