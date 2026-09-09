#include "../src/vertex_modifier_ir_admission.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using videowire::VertexModifierIr;
using videowire::VertexModifierOperation;
using videowire::VertexModifierRecord;
using videowire::VertexModifierStableId;
using videowire::VertexModifierValueType;
using videohelper::vertexmodifier::AdmissionLimits;
using videohelper::vertexmodifier::AdmittedVertexModifierIr;

int failures = 0;

void check(bool condition, const char* message)
{
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

VertexModifierRecord record(VertexModifierStableId id, VertexModifierOperation operation,
                            std::initializer_list<VertexModifierStableId> inputs = {},
                            std::initializer_list<double> parameters = {})
{
    const auto schema = videowire::vertexModifierOperationSchema(operation);
    VertexModifierRecord value;
    value.stableId = id;
    value.operation = operation;
    value.resultType = schema.resultType;
    value.inputCount = static_cast<std::uint8_t>(inputs.size());
    value.parameterCount = static_cast<std::uint8_t>(parameters.size());
    std::copy(inputs.begin(), inputs.end(), value.inputs.begin());
    std::copy(parameters.begin(), parameters.end(), value.parameters.begin());
    return value;
}

VertexModifierIr completeFixture()
{
    using Operation = VertexModifierOperation;
    VertexModifierIr ir;
    ir.rootId = 26;
    ir.records = {
        record(1, Operation::importedPosition),
        record(2, Operation::importedNormal),
        record(3, Operation::importedUv),
        record(4, Operation::importedVertexColor),
        record(5, Operation::timeSeconds),
        record(6, Operation::audioParameter, {}, { 3.0 }),
        record(7, Operation::controlParameter, {}, { 12.0 }),
        record(8, Operation::scalarConstant, {}, { 0.1 }),
        record(9, Operation::scalarAdd, { 5, 6 }),
        record(10, Operation::scalarSubtract, { 9, 7 }),
        record(11, Operation::scalarMultiply, { 10, 8 }),
        record(12, Operation::scalarRemap, { 11 }, { -2.0, 2.0, -0.5, 0.5, 1.0 }),
        record(13, Operation::scalarNoise3d, { 3 }, { 2.0, 7.0 }),
        record(14, Operation::scalarAdd, { 12, 13 }),
        record(15, Operation::componentX, { 4 }),
        record(16, Operation::scalarMultiply, { 14, 15 }),
        record(17, Operation::vec3Compose, { 16, 8, 7 }),
        record(18, Operation::vec3Noise3d, { 1 }, { 1.5, 11.0 }),
        record(19, Operation::vec3Add, { 17, 18 }),
        record(20, Operation::vec3Constant, {}, { 0.1, 0.2, 0.0 }),
        record(21, Operation::vec3Subtract, { 19, 20 }),
        record(22, Operation::vec3Multiply, { 21, 4 }),
        record(23, Operation::vec3Scale, { 2, 16 }),
        record(24, Operation::vec3Add, { 22, 23 }),
        record(25, Operation::vec3Remap, { 24 }, { -2.0, 2.0, -1.0, 1.0, 1.0 }),
        record(26, Operation::boundedDisplacementOutput, { 1, 25 }, { 0.25 })
    };
    return ir;
}

VertexModifierRecord* findRecord(VertexModifierIr& ir, VertexModifierStableId id)
{
    const auto found = std::find_if(ir.records.begin(), ir.records.end(),
        [id](const VertexModifierRecord& candidate) { return candidate.stableId == id; });
    return found == ir.records.end() ? nullptr : &*found;
}

bool rejected(const VertexModifierIr& ir, AdmissionLimits limits = {})
{
    std::string error;
    return ! videohelper::vertexmodifier::admitVertexModifierIr(ir, limits, error).has_value()
        && ! error.empty();
}
} // namespace

int main()
{
    static_assert(std::is_trivially_copyable<VertexModifierRecord>::value,
                  "the restricted wire record cannot own source text or runtime objects");
    static_assert(std::is_same<decltype(std::declval<const AdmittedVertexModifierIr&>().records()),
                               const std::vector<VertexModifierRecord>&>::value,
                  "admitted vertex modifier records must be exposed read-only");
    static_assert(std::is_same<decltype(std::declval<const AdmittedVertexModifierIr&>()
                                           .structuralDigest()),
                               const std::string&>::value,
                  "the structural digest must be exposed read-only");

    std::set<std::string> operationTokens;
    for (int raw = static_cast<int>(VertexModifierOperation::importedPosition);
         raw <= static_cast<int>(VertexModifierOperation::boundedDisplacementOutput); ++raw)
    {
        const auto operation = static_cast<VertexModifierOperation>(raw);
        const auto token = std::string(videowire::vertexModifierOperationWireToken(operation));
        const auto schema = videowire::vertexModifierOperationSchema(operation);
        check(schema.valid, "every declared vertex modifier operation has a typed schema");
        check(operationTokens.emplace(token).second,
              "every declared vertex modifier operation has a unique wire token");
        check(videowire::vertexModifierOperationFromWireToken(token) == operation,
              "every operation wire token round-trips");
    }
    check(operationTokens.size() == 25,
          "the contract contains only the bounded vertex modifier operation set");
    check(videowire::vertexModifierOperationFromWireToken("shader-source")
              == VertexModifierOperation::invalid,
          "arbitrary shader source has no operation token");

    std::set<std::string> valueTokens;
    for (const auto type : { VertexModifierValueType::scalar, VertexModifierValueType::vec3 })
    {
        const auto token = std::string(videowire::vertexModifierValueTypeWireToken(type));
        check(valueTokens.emplace(token).second, "value type wire tokens are unique");
        check(videowire::vertexModifierValueTypeFromWireToken(token) == type,
              "value type wire tokens round-trip");
    }
    check(valueTokens.size() == 2
              && videowire::vertexModifierValueTypeFromWireToken("source")
                   == VertexModifierValueType::invalid,
          "only scalar and Vec3 values are admitted");

    auto ir = completeFixture();
    std::reverse(ir.records.begin(), ir.records.end());
    findRecord(ir, 20)->parameters[2] = -0.0;
    std::string error;
    const auto admitted = videohelper::vertexmodifier::admitVertexModifierIr(ir, {}, error);
    check(admitted.has_value() && error.empty(),
          "the complete backend-neutral vertex modifier fixture is admitted");
    check(admitted && admitted->operationCount() == 26 && admitted->maximumDepth() > 5,
          "admission reports bounded operation count and dependency depth");
    check(admitted && admitted->maximumDisplacement() == 0.25,
          "the terminal output publishes its admitted displacement bound");
    check(admitted && admitted->records().front().stableId == 1
                   && admitted->records().back().stableId == 26,
          "admitted records use canonical stable-ID order");
    check(admitted && admitted->structuralDigest().size() == 64,
          "the admitted structure has a SHA-256 digest");

    auto reordered = completeFixture();
    const auto reorderedAdmission =
        videohelper::vertexmodifier::admitVertexModifierIr(reordered, {}, error);
    check(admitted && reorderedAdmission
              && admitted->structuralDigest() == reorderedAdmission->structuralDigest(),
          "record order and negative zero do not change the structural digest");
    check(reorderedAdmission
              && reorderedAdmission->structuralDigest()
                   == "aba7e84ca35da9ce5c67168fe001478137c9fec77ad74bb0d69151f3a79c8cc7",
          "the reference fixture has a stable cross-process digest");

    auto changed = completeFixture();
    std::swap(findRecord(changed, 24)->inputs[0], findRecord(changed, 24)->inputs[1]);
    const auto changedAdmission =
        videohelper::vertexmodifier::admitVertexModifierIr(changed, {}, error);
    check(reorderedAdmission && changedAdmission
              && reorderedAdmission->structuralDigest() != changedAdmission->structuralDigest(),
          "ordered arithmetic operands participate in the digest");

    auto sourceMutation = completeFixture();
    const auto immutable =
        videohelper::vertexmodifier::admitVertexModifierIr(sourceMutation, {}, error);
    findRecord(sourceMutation, 8)->parameters[0] = 99.0;
    check(immutable && immutable->records()[7].parameters[0] == 0.1,
          "admission owns an immutable copy rather than aliasing producer records");

    auto duplicate = completeFixture();
    duplicate.records.back().stableId = 1;
    check(rejected(duplicate), "duplicate stable IDs are rejected");

    auto missing = completeFixture();
    findRecord(missing, 24)->inputs[0] = 999;
    check(rejected(missing), "missing stable-ID references are rejected");

    auto cycle = completeFixture();
    findRecord(cycle, 19)->inputs[1] = 19;
    check(rejected(cycle), "cycles are rejected independently of record order");

    auto unreachable = completeFixture();
    unreachable.records.push_back(record(99, VertexModifierOperation::scalarConstant, {}, { 1.0 }));
    check(rejected(unreachable), "records outside the root dependency graph are rejected");

    auto badShape = completeFixture();
    findRecord(badShape, 9)->inputCount = 1;
    check(rejected(badShape), "operation arity is exact");

    auto badResult = completeFixture();
    findRecord(badResult, 9)->resultType = VertexModifierValueType::vec3;
    check(rejected(badResult), "declared result types must match operation schemas");

    auto badInputType = completeFixture();
    findRecord(badInputType, 9)->inputs[0] = 4;
    check(rejected(badInputType), "referenced input types must match typed arity");

    auto hiddenInput = completeFixture();
    findRecord(hiddenInput, 8)->inputs[2] = 1;
    check(rejected(hiddenInput), "undeclared input slots cannot carry hidden data");

    auto hiddenParameter = completeFixture();
    findRecord(hiddenParameter, 8)->parameters[5] = 1.0;
    check(rejected(hiddenParameter), "undeclared parameter slots cannot carry hidden data");

    auto nonFinite = completeFixture();
    findRecord(nonFinite, 8)->parameters[0] = std::numeric_limits<double>::infinity();
    check(rejected(nonFinite), "non-finite parameters are rejected");

    auto outOfRange = completeFixture();
    findRecord(outOfRange, 20)->parameters[0] = 1000001.0;
    check(rejected(outOfRange), "parameter magnitude bounds are enforced");

    auto badAudioIndex = completeFixture();
    findRecord(badAudioIndex, 6)->parameters[0] = 3.5;
    check(rejected(badAudioIndex), "audio parameter indices are bounded integers");

    auto badControlIndex = completeFixture();
    findRecord(badControlIndex, 7)->parameters[0] = 4096.0;
    check(rejected(badControlIndex), "control parameter indices are bounded integers");

    auto badRemap = completeFixture();
    findRecord(badRemap, 12)->parameters[1] = -2.0;
    check(rejected(badRemap), "remap input ranges must increase");

    auto badClamp = completeFixture();
    findRecord(badClamp, 12)->parameters[4] = 0.5;
    check(rejected(badClamp), "remap clamp flags are binary");

    auto badNoiseFrequency = completeFixture();
    findRecord(badNoiseFrequency, 13)->parameters[0] = 0.0;
    check(rejected(badNoiseFrequency), "noise frequency must be positive");

    auto badNoiseSeed = completeFixture();
    findRecord(badNoiseSeed, 18)->parameters[1] = 1.25;
    check(rejected(badNoiseSeed), "noise seeds are bounded integers");

    auto nonTerminalRoot = completeFixture();
    nonTerminalRoot.rootId = 25;
    check(rejected(nonTerminalRoot), "the root must be the terminal displacement output");

    VertexModifierIr nestedTerminal;
    nestedTerminal.rootId = 4;
    nestedTerminal.records = {
        record(1, VertexModifierOperation::importedPosition),
        record(2, VertexModifierOperation::vec3Constant, {}, { 0.0, 0.0, 0.0 }),
        record(3, VertexModifierOperation::boundedDisplacementOutput, { 1, 2 }, { 0.1 }),
        record(4, VertexModifierOperation::boundedDisplacementOutput, { 1, 3 }, { 0.1 })
    };
    check(rejected(nestedTerminal), "terminal displacement outputs cannot be nested");

    auto indirectBase = completeFixture();
    findRecord(indirectBase, 26)->inputs[0] = 18;
    check(rejected(indirectBase), "the output base must directly reference imported position");

    auto zeroBound = completeFixture();
    findRecord(zeroBound, 26)->parameters[0] = 0.0;
    check(rejected(zeroBound), "the displacement output requires a positive bound");

    auto excessiveBound = completeFixture();
    findRecord(excessiveBound, 26)->parameters[0] = 1001.0;
    check(rejected(excessiveBound), "the displacement output cannot exceed admission limits");

    AdmissionLimits operationLimit;
    operationLimit.maxOperations = 25;
    check(rejected(completeFixture(), operationLimit), "operation count is bounded");

    AdmissionLimits depthLimit;
    depthLimit.maxDepth = 5;
    check(rejected(completeFixture(), depthLimit), "dependency depth is bounded");

    AdmissionLimits parameterLimit;
    parameterLimit.maxParametersPerRecord = 4;
    check(rejected(completeFixture(), parameterLimit), "per-record parameter count is bounded");

    auto badStableId = completeFixture();
    findRecord(badStableId, 1)->stableId = 9007199254740992ull;
    check(rejected(badStableId), "stable IDs outside the exact transport range are rejected");

    auto badSchema = completeFixture();
    badSchema.schemaVersion = 2;
    check(rejected(badSchema), "unknown schema versions are rejected");

    AdmissionLimits invalidLimits;
    invalidLimits.maxDepth = 65;
    check(rejected(completeFixture(), invalidLimits), "admission limits have absolute ceilings");

    if (reorderedAdmission)
        std::cout << "fixture digest: " << reorderedAdmission->structuralDigest() << '\n';
    if (failures != 0) return EXIT_FAILURE;
    std::cout << "Vertex modifier IR admission checks passed\n";
    return EXIT_SUCCESS;
}
