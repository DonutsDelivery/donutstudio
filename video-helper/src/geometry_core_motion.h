#pragma once

#include "gpu_backend/backend.h"
#include "../../shared/GeometrySurfaceMaterial.h"
#include "../../shared/VertexControlBank.h"
#include "../../shared/VisualAnimationDeformationEvaluation.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Geometry Core motion history. Each draw keeps the sample it rendered: the
// vertex program's time, spectrum and control bank, the scene positions the
// CPU geometry produced before the program ran, and each instance's transform.
// The next presentation frame at the same rate and extent uses it as its
// predecessor, so OpenGL and Metal run the vertex program a second time on the
// prior positions with the prior inputs, move each instance from its prior
// transform, and write Motion from the difference.
namespace videohelper::geometry {
struct GeometryMotionRecord final {
  visualdeformation::RationalFrameTime frame;
  std::uint32_t width = 0, height = 0;
  arbitgpu::NativeSceneMotionSample sample;
  // Instanced draws: each instance's scene identity and local transform,
  // sorted by identity. Empty for a single mesh.
  std::vector<std::pair<std::uint32_t, HarmonicMIDI::grid::SceneTransform3D>> instanceTransforms;
};

// The prior record, only for the frame right after the recorded one at the
// same rate and extent. A seek, a rate change or a resize starts at zero
// motion, as a first frame does.
inline std::optional<GeometryMotionRecord> geometryMotionPredecessorRecord(
    const std::optional<GeometryMotionRecord>& previous, const visualdeformation::RationalFrameTime& frame,
    std::uint32_t width, std::uint32_t height) {
  if (!previous || frame.rateNumerator == 0 || frame.rateDenominator == 0
      || previous->frame.rateNumerator != frame.rateNumerator
      || previous->frame.rateDenominator != frame.rateDenominator
      || previous->frame.frame + 1 != frame.frame || previous->width != width || previous->height != height)
    return std::nullopt;
  return previous;
}

inline std::optional<arbitgpu::NativeSceneMotionSample> geometryMotionPredecessor(
    const std::optional<GeometryMotionRecord>& previous, const visualdeformation::RationalFrameTime& frame,
    std::uint32_t width, std::uint32_t height) {
  auto record = geometryMotionPredecessorRecord(previous, frame, width, height);
  if (!record) return std::nullopt;
  return std::move(record->sample);
}

// Gives every scene object of an instanced draw its prior local transform: the
// predecessor's transform of the same instance identity, or its current one
// for an instance born this frame or a draw without a predecessor. OpenGL and
// Metal keep the one shared draw and give each instance record its prior world
// matrix, so each moves by its own step under any scale.
inline void geometryPriorInstanceTransforms(HarmonicMIDI::grid::Visual3DScene& scene,
                                            const GeometryMotionRecord* predecessor) {
  for (std::size_t index = 0; index < scene.objectCount; ++index) {
    const auto& object = scene.objects[index];
    auto prior = object.transform;
    if (predecessor != nullptr) {
      const auto& priors = predecessor->instanceTransforms;
      const auto found = std::lower_bound(priors.begin(), priors.end(), object.id.value,
          [](const auto& entry, std::uint32_t id) { return entry.first < id; });
      if (found != priors.end() && found->first == object.id.value) prior = found->second;
    }
    scene.previousObjectTransforms[index] = prior;
  }
  scene.previousObjectTransformCount = scene.objectCount;
}

// The sample a draw records: its runtime inputs, its scene positions and, for
// an instanced draw, each instance's transform.
inline GeometryMotionRecord geometryMotionRecord(const arbitgpu::NativeFixtureSceneRuntimeInputs& inputs,
    const HarmonicMIDI::grid::Visual3DScene& scene, const visualdeformation::RationalFrameTime& frame,
    std::uint32_t width, std::uint32_t height, bool instanced = false) {
  GeometryMotionRecord record { frame, width, height, arbitgpu::sceneMotionSample(inputs), {} };
  auto positions = std::make_shared<std::vector<HarmonicMIDI::grid::SceneVec3>>();
  positions->reserve(scene.vertexCount);
  for (std::size_t i = 0; i < scene.vertexCount; ++i) positions->push_back(scene.vertices[i].position);
  record.sample.scenePositions = std::move(positions);
  if (instanced) {
    record.instanceTransforms.reserve(scene.objectCount);
    for (std::size_t i = 0; i < scene.objectCount; ++i)
      record.instanceTransforms.emplace_back(scene.objects[i].id.value, scene.objects[i].transform);
    std::sort(record.instanceTransforms.begin(), record.instanceTransforms.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
  }
  return record;
}

// True when a published runtime value reaches one of the Surfaces' Vertex
// Control Parameter reads, so the draw changes every frame.
inline bool geometryControlsModulated(const geometrysurfacematerial::ProgramSet& programs,
                                      const std::map<std::string, double>* runtimeParameters) {
  if (runtimeParameters == nullptr) return false;
  return std::any_of(programs.begin(), programs.end(), [&](const auto& program) {
    return program.material.vertexModifier && std::any_of(program.material.vertexModifier->records.begin(),
        program.material.vertexModifier->records.end(), [&](const auto& record) {
          return record.operation == videowire::VertexModifierOperation::controlParameter
              && runtimeParameters->count(videowire::vertexcontrol::runtimeKey(record.stableId)) != 0;
        });
  });
}
} // namespace videohelper::geometry
