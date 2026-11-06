#include "../../shared/GeometrySurfaceMaterial.h"
#include "../../shared/generated/SurfaceMaterialStarterPrograms.h"
#include "support/vertex_modifier_fixture.h"

#include <cstdio>
#include <string>

// A Geometry Surface carries the Surface's vertex stage: it validates, round
// trips through the single and set wires, makes the set wire version 2 only
// when present, keeps the old wires unchanged without it, admits spectrum and
// noise records, and admits control reads.
namespace {
int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

surfacematerialbinding::ImportedSceneMaterialRequest surface() {
  surfacematerialbinding::ImportedSceneMaterialRequest request;
  request.scene = {1}; request.binding.object = {1};
  request.sceneRevision = request.structuralRevision = request.evaluationRevision = request.programRevision = 1;
  request.binding.targetKind = surfacematerialbinding::BindingTargetKind::ObjectOverride;
  request.binding.surfaceMaterialRevision = 1;
  request.program = surfacematerialstarterfixture::kPrograms[0].program();
  std::string error;
  const auto admitted = surfacematerial::admit(request.program, error);
  if (admitted) request.binding.surfaceMaterialDigest = admitted->structuralDigest();
  return request;
}

void singleWire() {
  std::string error;
  const auto plain = surface();
  check(geometrysurfacematerial::valid(plain, error), "a plain Surface is valid");
  const auto plainWire = geometrysurfacematerial::encode(plain);
  check(plainWire.rfind(geometrysurfacematerial::kVertexWirePrefix, 0) != 0, "a plain Surface keeps the old wire");
  check(!geometrysurfacematerial::timeDependent(plain), "a plain starter Surface is not time dependent");

  auto moving = surface();
  moving.vertexModifier = videohelper::test::timeAxisDisplacement();
  check(geometrysurfacematerial::valid(moving, error), "a Surface with a vertex stage is valid");
  const auto wire = geometrysurfacematerial::encode(moving);
  check(wire.rfind(geometrysurfacematerial::kVertexWirePrefix, 0) == 0, "the vertex stage marks the wire");
  const auto decoded = geometrysurfacematerial::decode(wire, error);
  check(decoded && decoded->vertexModifier
            && decoded->vertexModifier->records.size() == moving.vertexModifier->records.size()
            && geometrysurfacematerial::encode(*decoded) == wire,
        "the vertex stage round trips");
  check(geometrysurfacematerial::timeDependent(moving), "a time vertex stage makes the Surface time dependent");
}

void setWire() {
  std::string error;
  geometrysurfacematerial::ProgramSet plain(2);
  plain[0].sourceMaterial = 1; plain[0].material = surface();
  plain[1].sourceMaterial = 2; plain[1].material = surface();
  const auto plainWire = geometrysurfacematerial::encodeSet(plain);
  check(plainWire.rfind("geometry-surface-set-v1\n", 0) == 0, "a set without vertex stages stays version 1");
  check(geometrysurfacematerial::decodeSet(plainWire, error).has_value(), "a version 1 set still decodes");

  auto moving = plain;
  moving[1].material.vertexModifier = videohelper::test::timeAxisDisplacement(0.3, 2.0);
  const auto wire = geometrysurfacematerial::encodeSet(moving);
  check(wire.rfind("geometry-surface-set-v2\n", 0) == 0, "a set with a vertex stage is version 2");
  const auto decoded = geometrysurfacematerial::decodeSet(wire, error);
  check(decoded && decoded->size() == 2 && !(*decoded)[0].material.vertexModifier
            && (*decoded)[1].material.vertexModifier && geometrysurfacematerial::encodeSet(*decoded) == wire,
        "the version 2 set round trips each entry's vertex stage");
  check(geometrysurfacematerial::timeDependent(moving), "the set is time dependent through its vertex stage");
}

void spectrumNoiseAndControl() {
  std::string error;
  auto audio = surface();
  audio.vertexModifier = videohelper::test::audioNormalDisplacement();
  check(geometrysurfacematerial::valid(audio, error), ("a spectrum vertex stage is valid: " + error).c_str());
  check(!geometrysurfacematerial::encode(audio).empty(), "a spectrum vertex stage encodes");

  using Operation = videowire::VertexModifierOperation;
  videowire::VertexModifierIr noise;
  videohelper::test::appendVertexModifierRecord(noise, Operation::importedPosition, {}, {});
  videohelper::test::appendVertexModifierRecord(noise, Operation::scalarNoise3d, {1}, {1.0, 0.0});
  videohelper::test::appendVertexModifierRecord(noise, Operation::importedNormal, {}, {});
  videohelper::test::appendVertexModifierRecord(noise, Operation::vec3Scale, {3, 2}, {});
  videohelper::test::appendVertexModifierRecord(noise, Operation::boundedDisplacementOutput, {1, 4}, {0.5});
  noise.rootId = 5;
  check(geometrysurfacematerial::geometryVertexStageRefusal(noise) == nullptr, "a noise vertex stage is admitted");

  auto control = noise;
  control.records[1].operation = Operation::controlParameter;
  control.records[1].inputCount = 0;
  control.records[1].inputs[0] = 0;
  control.records[1].parameterCount = 1;
  control.records[1].parameters[0] = 2.0;
  control.records[1].parameters[1] = 0.0;
  check(geometrysurfacematerial::geometryVertexStageRefusal(control) == nullptr,
        "a control read is admitted");
}
} // namespace

int main() {
  singleWire();
  setWire();
  spectrumNoiseAndControl();
  if (failures != 0) return 1;
  std::puts("geometry surface vertex stage tests passed");
  return 0;
}
