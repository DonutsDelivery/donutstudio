#include "../../shared/SdfSurfaceOperationContract.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

// SDF surface evaluation work follows the record count of a shared graph, and
// sharing never changes an answer.
namespace {
int failures = 0;

void check(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

// Union of a record with itself leaves the field unchanged, so forty doublings
// must give the plain sphere's mesh at a cost of one evaluation per record.
void sdfDoubledGraphWorkFollowsRecordCount() {
  namespace surface = videowire::sdfsurface;
  surface::Request plain;
  plain.cells = {12, 12, 12};
  plain.maximumResidual = 0.2;
  videowire::SdfRecord sphere;
  sphere.stableId = 1;
  sphere.operation = videowire::SdfOperation::sphere;
  sphere.parameterCount = 1;
  sphere.parameters[0] = 1.013;
  plain.geometry.rootId = sphere.stableId;
  plain.geometry.records = {sphere};
  auto doubled = plain;
  for (std::uint64_t i = 0; i < 40; ++i) {
    videowire::SdfRecord both;
    both.stableId = 2 + i;
    both.operation = videowire::SdfOperation::unionOp;
    both.inputCount = 2;
    both.inputs = {doubled.geometry.rootId, doubled.geometry.rootId};
    doubled.geometry.records.push_back(both);
    doubled.geometry.rootId = both.stableId;
  }
  surface::detail::Evaluator evaluator(doubled.geometry);
  const auto expected = surface::detail::Evaluator(plain.geometry).at({0.25, 0.5, 0.75});
  const auto sample = evaluator.at({0.25, 0.5, 0.75});
  check(sample.distance == expected.distance && sample.primitiveId == sphere.stableId,
        "a doubled graph keeps the sphere's distance and primitive");
  check(evaluator.recordEvaluations() == 41, "a depth 40 doubled graph computes each of its 41 records once");
  evaluator.at({0.25, 0.5, 0.75});
  evaluator.at({-0.5, 0.125, 0.0});
  check(evaluator.recordEvaluations() == 3 * 41, "each query costs the record count again");

  surface::Mesh plainMesh, doubledMesh;
  std::string error;
  const bool plainDone = surface::materialize(plain, 9700, 1000000, 6000000, plainMesh, error);
  check(plainDone, error.c_str());
  const bool doubledDone = surface::materialize(doubled, 9700, 1000000, 6000000, doubledMesh, error);
  check(doubledDone, error.c_str());
  check(!plainMesh.positions.empty() && doubledMesh.positions == plainMesh.positions
            && doubledMesh.normals == plainMesh.normals && doubledMesh.uv == plainMesh.uv
            && doubledMesh.vertexIds == plainMesh.vertexIds && doubledMesh.indices == plainMesh.indices
            && doubledMesh.primitiveIds == plainMesh.primitiveIds,
        "a depth 40 doubled graph meshes identically to its sphere");
}

// Depth 64 is the deepest legal graph, and its expanded size passes 64 bits once
// multiplied by the query count. The work count saturates and the surface meshes.
void sdfDeepestDoubledGraphSaturatesItsWorkCount() {
  namespace surface = videowire::sdfsurface;
  surface::Request request;
  request.cells = {12, 12, 12};
  request.maximumResidual = 0.2;
  videowire::SdfRecord sphere;
  sphere.stableId = 1;
  sphere.operation = videowire::SdfOperation::sphere;
  sphere.parameterCount = 1;
  sphere.parameters[0] = 1.013;
  request.geometry.rootId = sphere.stableId;
  request.geometry.records = {sphere};
  for (std::uint64_t i = 0; i < 63; ++i) {
    videowire::SdfRecord both;
    both.stableId = 2 + i;
    both.operation = videowire::SdfOperation::unionOp;
    both.inputCount = 2;
    both.inputs = {request.geometry.rootId, request.geometry.rootId};
    request.geometry.records.push_back(both);
    request.geometry.rootId = both.stableId;
  }
  surface::Budget budget;
  std::string error;
  const bool inspected = surface::inspect(request, budget, error);
  check(inspected, error.c_str());
  check(budget.sampleWork == std::numeric_limits<std::size_t>::max(), "depth 64 sample work saturates");
  surface::Mesh mesh;
  const bool meshed = surface::materialize(request, 9700, 1000000, 6000000, mesh, error);
  check(meshed, error.c_str());
  check(!mesh.positions.empty(), "the depth 64 doubled graph meshes");
}

// The memo must not change an answer: a DAG with moved copies of shared records
// agrees bit for bit with the same graph written out as a tree, which has nothing
// to memoize.
void sdfSharedRecordsMatchTheirExpandedTree() {
  namespace surface = videowire::sdfsurface;
  using videowire::SdfOperation;
  using videowire::SdfRecord;
  const auto record = [](std::uint64_t id, SdfOperation operation, std::array<std::uint64_t, 2> inputs,
                         unsigned inputCount, std::vector<double> parameters) {
    SdfRecord r;
    r.stableId = id;
    r.operation = operation;
    r.inputCount = static_cast<decltype(r.inputCount)>(inputCount);
    r.inputs = inputs;
    r.parameterCount = static_cast<decltype(r.parameterCount)>(parameters.size());
    for (std::size_t i = 0; i < parameters.size(); ++i) r.parameters[i] = parameters[i];
    return r;
  };
  videowire::SdfIr dag;
  dag.records = {
      record(1, SdfOperation::sphere, {0, 0}, 0, {0.6}),
      record(2, SdfOperation::translate, {1, 0}, 1, {0.3, -0.2, 0.1}),
      record(3, SdfOperation::smoothUnion, {1, 2}, 2, {0.2}),
      record(4, SdfOperation::rotate, {3, 0}, 1, {0, 1, 0, 0.7}),
      record(5, SdfOperation::mirror, {3, 0}, 1, {1, 0, 0}),
      record(6, SdfOperation::subtraction, {4, 5}, 2, {}),
      record(7, SdfOperation::scale, {3, 0}, 1, {1.5, 0.75, 1.25}),
      record(8, SdfOperation::unionOp, {6, 7}, 2, {}),
      record(9, SdfOperation::smoothIntersection, {8, 8}, 2, {0.1}),
  };
  dag.rootId = 9;

  videowire::SdfIr tree;
  std::vector<std::uint64_t> origin{0};
  std::function<std::uint64_t(std::uint64_t)> copy = [&](std::uint64_t id) {
    auto r = *std::find_if(dag.records.begin(), dag.records.end(),
                           [&](const SdfRecord& candidate) { return candidate.stableId == id; });
    for (unsigned i = 0; i < r.inputCount; ++i) r.inputs[i] = copy(r.inputs[i]);
    r.stableId = origin.size();
    origin.push_back(id);
    tree.records.push_back(r);
    return r.stableId;
  };
  tree.rootId = copy(dag.rootId);
  check(tree.records.size() > 2 * dag.records.size(), "the fixture shares records");

  surface::detail::Evaluator shared(dag), expanded(tree);
  bool same = true;
  const auto compare = [&](std::array<double, 3> point) {
    const auto a = shared.at(point), b = expanded.at(point);
    same = same && std::memcmp(&a.distance, &b.distance, sizeof(double)) == 0
           && a.primitiveId == origin.at(b.primitiveId);
  };
  for (int x = -8; x <= 8; ++x) for (int y = -8; y <= 8; ++y) for (int z = -8; z <= 8; ++z)
    compare({x * 0.17, y * 0.13, z * 0.11});
  compare({-0.0, 0.0, 0.0});
  compare({0.0, -0.0, 0.0});
  check(same, "a shared DAG agrees bit for bit with its expanded tree");
  check(shared.recordEvaluations() < expanded.recordEvaluations(), "the shared DAG computes fewer records");
}

} // namespace

int main() {
  sdfDoubledGraphWorkFollowsRecordCount();
  sdfDeepestDoubledGraphSaturatesItsWorkCount();
  sdfSharedRecordsMatchTheirExpandedTree();
  if (failures != 0) {
    std::fprintf(stderr, "%d SDF surface work check(s) failed\n", failures);
    return 1;
  }
  std::printf("SDF surface work tests passed\n");
  return 0;
}
