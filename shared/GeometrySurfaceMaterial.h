#pragma once

#include "VisualImportedSceneRenderOperationContract.h"
#include "../video-helper/src/surface_material_admission.h"
#include "SurfaceMaterialProgramBinding.h"

namespace geometrysurfacematerial {
// A Geometry Surface may carry the Surface's vertex stage. Geometry Core draws
// run it on the raster with the frame's spectrum, and Path Trace runs it before
// its BVH. Control reads take the authored values of the shared control bank.
inline const char* geometryVertexStageRefusal(const videowire::VertexModifierIr& vertex) noexcept {
  if (!visualimportedscenerender::detail::validVertexModifier(vertex))
    return "Geometry Surface vertex stage is malformed";
  return nullptr;
}
inline constexpr const char* kVertexWirePrefix = "geometry-surface-vertex-v1\n";
// A generated-scene binding template. Scene/object 1 identify the Geometry Core
// renderer, not a foreign snapshot. Each replay binds this same admitted IR to
// the exact generated object before native program compilation.
inline bool valid(const surfacematerialbinding::ImportedSceneMaterialRequest& request,
                  std::string& error) {
  using namespace surfacematerialbinding;
  const auto program = surfacematerial::admit(request.program, error);
  if (!program) return false;
  const bool frame = hasGraphFrameInput(request.binding);
  const bool validFrame = request.binding.textures.size() == 1
      && request.binding.textures[0].slot == 0
      && request.binding.textures[0].source == TextureSourceKind::GraphFrame
      && !hasIdentity(request.binding.textures[0].videoResource)
      && request.binding.textures[0].graphFrame
      && request.binding.textures[0].graphFrame->node >= 0
      && request.binding.textures[0].graphFrame->port >= 0;
  if (request.scene.value != 1 || request.binding.object.value != 1 || request.sceneSnapshot
      || request.binding.targetKind != BindingTargetKind::ObjectOverride
      || request.binding.materialSlot.isValid()
      || (request.vertexModifier && geometryVertexStageRefusal(*request.vertexModifier))
      || request.version != requestVersion(frame,request.program.spectralReflectance.has_value(),request.roughGlassScatteringVersion,
          request.program.diffractionLayer.has_value())
      // Version and rough-glass rules match visualimportedscenerender::detail::validMaterial:
      // reflectance and the diffraction layer each compose with either scattering version.
      || !validRoughGlassVersion(request)
      || (frame ? !validFrame : !request.binding.textures.empty())
      || request.program.textureSlotCount != (frame ? 1 : 0) || request.programRevision == 0
      || request.sceneRevision != request.programRevision
      || request.structuralRevision != request.programRevision
      || request.evaluationRevision != request.programRevision
      || request.binding.surfaceMaterialRevision != request.programRevision
      || request.binding.surfaceMaterialDigest != program->structuralDigest()) {
    error = request.vertexModifier && geometryVertexStageRefusal(*request.vertexModifier)
        ? geometryVertexStageRefusal(*request.vertexModifier)
        : "Reactive Geometry Surface requires an exact Surface program with an optional graph Frame endpoint";
    return false;
  }
  error.clear(); return true;
}
inline std::string encode(const surfacematerialbinding::ImportedSceneMaterialRequest& request) {
  std::string error;
  if (!valid(request, error)) return {};
  std::ostringstream output;
  output.imbue(std::locale::classic());
  if (request.vertexModifier) output << kVertexWirePrefix;
  visualimportedscenerender::detail::encodeMaterial(output, request, request.vertexModifier.has_value());
  return output.str();
}
inline std::optional<surfacematerialbinding::ImportedSceneMaterialRequest>
decode(const std::string& wire, std::string& error) {
  if (wire.empty()) {
    error = "Reactive Geometry Surface binding is empty"; return std::nullopt;
  }
  const bool vertex = wire.rfind(kVertexWirePrefix, 0) == 0;
  std::istringstream input(vertex ? wire.substr(std::char_traits<char>::length(kVertexWirePrefix)) : wire);
  input.imbue(std::locale::classic());
  surfacematerialbinding::ImportedSceneMaterialRequest request;
  if (!visualimportedscenerender::detail::decodeMaterial(input, request, vertex)
      || !valid(request, error) || encode(request) != wire) {
    if (error.empty()) error = "Reactive Geometry Surface binding is not canonical";
    return std::nullopt;
  }
  return request;
}
inline bool timeDependent(const surfacematerialbinding::ImportedSceneMaterialRequest& request) {
  return std::any_of(request.program.operations.begin(), request.program.operations.end(),
      [](const auto& operation) { return operation.kind == surfacematerial::OperationKind::Input
          && operation.semantic == surfacematerial::InputSemantic::Time; })
      || (request.vertexModifier && std::any_of(request.vertexModifier->records.begin(), request.vertexModifier->records.end(),
          [](const auto& record) { return record.operation == videowire::VertexModifierOperation::timeSeconds; }));
}

inline std::vector<surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint>
graphFrameEndpoints(const ProgramSet& programs) {
  std::vector<surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint> result;
  for (const auto& program : programs)
    for (const auto& texture : program.material.binding.textures)
      if (texture.source == surfacematerialbinding::TextureSourceKind::GraphFrame && texture.graphFrame)
        result.push_back(*texture.graphFrame);
  std::sort(result.begin(),result.end());
  result.erase(std::unique(result.begin(),result.end()),result.end());
  return result;
}
inline std::optional<surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint>
graphFrameEndpoint(const ProgramSet& programs) {
  for (const auto& program : programs)
    if (surfacematerialbinding::hasGraphFrameInput(program.material.binding))
      return program.material.binding.textures.front().graphFrame;
  return std::nullopt;
}
inline bool valid(const ProgramSet& programs, std::string& error) {
  if (programs.empty() || programs.size() > surfacematerialbinding::kMaximumPrograms
      || (programs.size() > 1 && programs.front().sourceMaterial == 0)) {
    error = "Geometry Surface requires one override or at most 32 material-table programs"; return false;
  }
  std::uint64_t previous = 0, revision = programs.front().material.structuralRevision;
  for (std::size_t index = 0; index < programs.size(); ++index) {
    const auto& program = programs[index];
    if ((index && program.sourceMaterial <= previous)
        || program.sourceMaterial > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) + 1
        || program.material.structuralRevision != revision
        || program.fields.size() > surfacematerialfield::kMaximumBindings || !valid(program.material,error)
        || materialId(program.material)==0) {
      if (error.empty()) error = "Geometry Surface programs require unique ordered source identities and one revision";
      return false;
    }
    previous = program.sourceMaterial;
    for (const auto& field : program.fields) if (!surfacematerialfield::valid(field,error)) return false;
  }
  error.clear(); return true;
}
inline std::string encodeSet(const ProgramSet& programs) {
  std::string error;
  if (!valid(programs,error)) return {};
  if (programs.size() == 1 && programs[0].sourceMaterial == 0 && programs[0].fields.empty()) return encode(programs[0].material);
  std::ostringstream output;
  output.imbue(std::locale::classic());
  // Version 2 adds a vertex flag per entry. Sets without a vertex stage keep version 1.
  const bool vertices = std::any_of(programs.begin(), programs.end(),
      [](const auto& program) { return program.material.vertexModifier.has_value(); });
  output << (vertices ? "geometry-surface-set-v2\n" : "geometry-surface-set-v1\n") << programs.size() << '\n';
  for (const auto& program : programs) {
    output << program.sourceMaterial << ' ' << program.fields.size();
    if (vertices) output << ' ' << (program.material.vertexModifier ? 1 : 0);
    output << '\n';
    visualimportedscenerender::detail::encodeMaterial(output,program.material,program.material.vertexModifier.has_value());
    for (const auto& field : program.fields) {
      std::uint64_t gain = 0; std::memcpy(&gain,&field.sample.gain,sizeof(gain));
      output << field.sample.materialNode << ' ' << field.sample.target << ' '
          << static_cast<unsigned>(field.sample.reduction) << ' ' << field.sample.index << ' '
          << gain << ' ' << field.objectIndex << ' ' << field.sample.fieldPlan << '\n';
    }
  }
  return output.str();
}
inline std::optional<ProgramSet> decodeSet(const std::string& wire, std::string& error) {
  const bool version2 = wire.rfind("geometry-surface-set-v2\n",0) == 0;
  if (!version2 && wire.rfind("geometry-surface-set-v1\n",0) != 0) {
    const auto request = decode(wire,error);
    if (!request) return std::nullopt;
    ProgramSet programs{{0,*request,{}}};
    if (!valid(programs,error)) return std::nullopt;
    return programs;
  }
  std::istringstream input(wire);
  input.imbue(std::locale::classic());
  std::string header; std::size_t count = 0;
  input >> header >> count;
  // Every entry takes more than one character, so the wire length bounds the counts.
  if (!input || count == 0 || count > wire.size()) {
    error = "Geometry Surface program count is invalid"; return std::nullopt;
  }
  ProgramSet programs(count);
  for (auto& program : programs) {
    std::size_t fieldCount = 0;
    unsigned vertex = 0;
    input >> program.sourceMaterial >> fieldCount;
    if (version2) input >> vertex;
    if (!input || fieldCount > wire.size() || vertex > 1
        || !visualimportedscenerender::detail::decodeMaterial(input,program.material,vertex != 0)) {
      error = "Geometry Surface program entry is malformed"; return std::nullopt;
    }
    program.fields.resize(fieldCount);
    for (auto& field : program.fields) {
      unsigned reduction = 0, objectIndex = 0; std::uint64_t gain = 0;
      input >> field.sample.materialNode >> field.sample.target >> reduction >> field.sample.index
          >> gain >> objectIndex >> field.sample.fieldPlan;
      if (!input || reduction > 3 || objectIndex > 1) { error = "Geometry Surface Field entry is malformed"; return std::nullopt; }
      field.sample.reduction = static_cast<materialfield::Reduction>(reduction);
      std::memcpy(&field.sample.gain,&gain,sizeof(gain)); field.objectIndex = objectIndex != 0;
    }
  }
  if (!valid(programs,error) || encodeSet(programs) != wire) {
    if (error.empty()) error = "Geometry Surface program set is not canonical";
    return std::nullopt;
  }
  return programs;
}
inline bool usesScore(const ProgramSet& programs) {
  std::string error;
  for (const auto& program : programs) for (const auto& field : program.fields) {
    const auto value = materialfield::admit(field.sample,error,true);
    if (value && videowire::geometry::hasRuntimeScoreFields(value->descriptor())) return true;
  }
  return false;
}
inline bool timeDependent(const ProgramSet& programs) {
  std::string error;
  for (const auto& program : programs) {
    if (timeDependent(program.material)) return true;
    for (const auto& field : program.fields) {
      const auto value = materialfield::admit(field.sample,error,true);
      if (value && videowire::geometry::hasRuntimeFields(value->descriptor())) return true;
    }
  }
  return false;
}
// An imported Surface override names one exact imported object. Beside the
// procedural programs it joins the object collection as the Render override
// (source zero), bound to the composed scene and the request's exact revisions.
// The composed list must pass the same collection validity as procedural
// programs, and the object limit counts the override.
inline bool appendImportedOverride(ObjectPrograms& programs,
    const surfacematerialbinding::ImportedSceneMaterialRequest& source,
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& composed,
    std::uint64_t structuralRevision, std::uint64_t evaluationRevision,
    std::string& error) {
  if (!composed) {
    error = "Scene Compose imported Surface override has no composed scene";
    return false;
  }
  ObjectProgram object;
  object.program.sourceMaterial = 0;
  object.program.material = source;
  auto& material = object.program.material;
  material.scene = composed->id;
  material.sceneSnapshot = composed;
  material.sceneRevision = structuralRevision;
  material.structuralRevision = structuralRevision;
  material.evaluationRevision = evaluationRevision;
  material.programRevision = structuralRevision;
  material.binding.surfaceMaterialRevision = structuralRevision;
  programs.push_back(std::move(object));
  std::sort(programs.begin(), programs.end(), [](const auto& left, const auto& right) {
    return left.program.material.binding.object.value < right.program.material.binding.object.value; });
  visualimportedscenerender::Request collection;
  collection.sourceStableId = composed->id.value;
  collection.renderStableId = collection.sourceStableId == 1 ? 2 : 1;
  collection.sceneSnapshot = composed;
  collection.structuralRevision = structuralRevision;
  collection.evaluationRevision = evaluationRevision;
  collection.surfacePrograms = programs;
  if (!visualimportedscenerender::valid(collection)) {
    error = "Scene Compose imported Surface override cannot share one admitted collection with its procedural Surface programs";
    return false;
  }
  error.clear();
  return true;
}

} // namespace geometrysurfacematerial
