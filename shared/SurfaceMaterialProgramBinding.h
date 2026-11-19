#pragma once

#include "SurfaceMaterialBindingContract.h"
#include "SurfaceMaterialField.h"
#include "Visual3DSceneComposition.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <optional>

namespace geometrysurfacematerial {
struct Program final {
  // Zero is the Render override. Other IDs retain the authored material node.
  videowire::geometry::StableId sourceMaterial = 0;
  surfacematerialbinding::ImportedSceneMaterialRequest material;
  std::vector<surfacematerialfield::Binding> fields;
};
using ProgramSet = std::vector<Program>;
inline std::uint32_t materialId(const surfacematerialbinding::ImportedSceneMaterialRequest& request) {
  const auto output=request.program.outputs[static_cast<std::size_t>(surfacematerial::OutputSemantic::MaterialId)];
  const auto found=std::find_if(request.program.operations.begin(),request.program.operations.end(),
      [output](const auto& operation){return operation.id==output;});
  return found!=request.program.operations.end() && found->kind==surfacematerial::OperationKind::UIntConstant
      ? found->unsignedLiteral : 0;
}
// A MaterialId output must be one unsigned constant. materialId() reads zero
// for any other output, which would pass as an object without a material.
inline bool materialIdIsConstant(const surfacematerialbinding::ImportedSceneMaterialRequest& request) {
  const auto output=request.program.outputs[static_cast<std::size_t>(surfacematerial::OutputSemantic::MaterialId)];
  const auto found=std::find_if(request.program.operations.begin(),request.program.operations.end(),
      [output](const auto& operation){return operation.id==output;});
  return found!=request.program.operations.end() && found->kind==surfacematerial::OperationKind::UIntConstant;
}
// A multi-leaf Scene Compose moves leaf k's material IDs into its leaf range,
// as it moves the leaf's objects. A Surface program bound to an object of that
// leaf writes its MaterialId literal in the same range, so the literal is
// moved before admission computes the program's structural digest. Zero stays
// zero, like an object without a material. A one-leaf scene keeps any
// literal; leaf 0 of several keeps its literal, which must be below the stride.
// The scheme picks the material stride: 2048 for saved Scene Compose nodes,
// 2^27 for Scene Compose nodes placed since the Wide scheme.
// Refuses, leaving the program unchanged, when the output is not one unsigned
// constant or its literal is outside the leaf range.
inline bool remapImportedLeafMaterialId(surfacematerial::ProgramDescription& program,
                                        std::size_t leaf, std::size_t leafCount,
                                        HarmonicMIDI::grid::scenecomposition::ImportedLeafIdScheme scheme
                                            = HarmonicMIDI::grid::scenecomposition::ImportedLeafIdScheme::Narrow) {
  using HarmonicMIDI::grid::scenecomposition::encodeImportedLeafId;
  using HarmonicMIDI::grid::scenecomposition::validImportedLeafCount;
  if (!validImportedLeafCount(leafCount) || leaf >= leafCount) return false;
  if (leafCount == 1) return true;
  const auto output=program.outputs[static_cast<std::size_t>(surfacematerial::OutputSemantic::MaterialId)];
  const auto found=std::find_if(program.operations.begin(),program.operations.end(),
      [output](const auto& operation){return operation.id==output;});
  if (found==program.operations.end() || found->kind!=surfacematerial::OperationKind::UIntConstant) return false;
  if (found->unsignedLiteral==0) return true;
  const auto encoded=encodeImportedLeafId(leaf,leafCount,found->unsignedLiteral,scheme);
  if (!encoded) return false;
  found->unsignedLiteral=*encoded;
  return true;
}
// A per-object Surface Material constant that a modulation target writes. The
// destination is visual<N>/<name>, where N is the source Surface Material's stable
// id (node id plus one), which is the sourceMaterial of every program the node
// lowers to. A write changes the literal in each of those programs, so every
// object the node draws moves together. Metallic, roughness and opacity are scalars.
// Base colour and emission are RGB, one lane per channel. Base colour is bounded to
// one, where the render clamps it. Opacity is bounded to the unit interval, and a
// value below one selects the blended pass. Emission keeps the bound of the PBR
// Material targets.
struct ConstantField final {
  const char* name;
  surfacematerial::OutputSemantic output;
  std::size_t lane;
  double minimum;
  double maximum;
};
inline constexpr std::array<ConstantField, 9> kConstantFields {{
  { "metallic", surfacematerial::OutputSemantic::Metallic, 0, 0.0, 1.0 },
  { "roughness", surfacematerial::OutputSemantic::Roughness, 0, 0.0, 1.0 },
  { "opacity", surfacematerial::OutputSemantic::Opacity, 0, 0.0, 1.0 },
  { "baseRed", surfacematerial::OutputSemantic::BaseColor, 0, 0.0, 1.0 },
  { "baseGreen", surfacematerial::OutputSemantic::BaseColor, 1, 0.0, 1.0 },
  { "baseBlue", surfacematerial::OutputSemantic::BaseColor, 2, 0.0, 1.0 },
  { "emissiveRed", surfacematerial::OutputSemantic::Emission, 0, 0.0, std::numeric_limits<float>::max() },
  { "emissiveGreen", surfacematerial::OutputSemantic::Emission, 1, 0.0, std::numeric_limits<float>::max() },
  { "emissiveBlue", surfacematerial::OutputSemantic::Emission, 2, 0.0, std::numeric_limits<float>::max() },
}};
// The operation whose literal the field writes, or nothing. The output's operation
// must be a FloatConstant of the output's type with the field's lane. A constant that
// another output or operation also reads would move with the write, so it is not
// offered. The output itself is the one permitted reader.
inline std::optional<std::size_t> constantOperation(const surfacematerial::ProgramDescription& program,
                                                    const ConstantField& field) {
  const auto output = program.output(field.output);
  const auto found = std::find_if(program.operations.begin(), program.operations.end(),
      [output](const auto& operation) { return operation.id == output; });
  if (found == program.operations.end() || found->kind != surfacematerial::OperationKind::FloatConstant
      || found->resultType != surfacematerial::outputType(field.output)) return std::nullopt;
  const std::size_t width = found->resultType == surfacematerial::ValueType::Vec3 ? 3 : 1;
  if (field.lane >= width) return std::nullopt;
  std::size_t readers = 0;
  for (const auto& operation : program.operations)
    for (std::size_t input = 0; input < operation.inputCount && input < operation.inputs.size(); ++input)
      if (operation.inputs[input] == output) ++readers;
  for (const auto candidate : program.outputs)
    if (candidate == output) ++readers;
  if (readers != 1) return std::nullopt;
  return static_cast<std::size_t>(found - program.operations.begin());
}
// Whether a modulation target is offered for this program. Opacity is offered only on a
// thin-surface program, the one transport whose native pass blends. A closed volume or
// thin interface program would render an opacity below one opaque.
inline bool constantOffered(const surfacematerial::ProgramDescription& program,
                            const ConstantField& field) {
  if (field.output == surfacematerial::OutputSemantic::Opacity
      && program.transport != surfacematerial::TransportMode::ThinSurface) return false;
  return constantOperation(program, field).has_value();
}

struct ObjectProgram final {
  Program program;
  std::uint32_t instanceIndex = 0;
  videowire::geometry::InstanceAppearance appearance;
};
using ObjectPrograms = std::vector<ObjectProgram>;
inline std::optional<surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint>
objectFrameEndpoint(const ObjectPrograms& programs) {
  for (const auto& object : programs)
    for (const auto& texture : object.program.material.binding.textures)
      if (texture.graphFrame) return texture.graphFrame;
  return std::nullopt;
}
} // namespace geometrysurfacematerial
