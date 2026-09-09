#pragma once

#include "../../shared/VertexModifierIr.h"
#include "sha256.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace videohelper::vertexmodifier
{

struct AdmissionLimits final
{
    std::size_t maxOperations = 128;
    std::size_t maxDepth = 32;
    std::size_t maxParametersPerRecord = videowire::kVertexModifierMaximumParametersPerRecord;
    videowire::VertexModifierStableId maxStableId = 9007199254740991ull;
    double maxAbsoluteParameter = 1000000.0;
    double minPositiveParameter = 0.000001;
    std::uint32_t maxAudioParameterIndex = 63;
    std::uint32_t maxControlParameterIndex = 4095;
    std::uint32_t maxNoiseSeed = 16777215;
    double maxDeclaredDisplacement = 1000.0;
};

class AdmittedVertexModifierIr final
{
public:
    AdmittedVertexModifierIr(const AdmittedVertexModifierIr&) = default;
    AdmittedVertexModifierIr(AdmittedVertexModifierIr&&) noexcept = default;
    AdmittedVertexModifierIr& operator=(const AdmittedVertexModifierIr&) = default;
    AdmittedVertexModifierIr& operator=(AdmittedVertexModifierIr&&) noexcept = default;

    std::uint32_t schemaVersion() const noexcept
    {
        return videowire::kVertexModifierIrSchemaVersion;
    }
    videowire::VertexModifierStableId rootId() const noexcept { return rootId_; }
    std::size_t operationCount() const noexcept { return records_.size(); }
    std::size_t maximumDepth() const noexcept { return maximumDepth_; }
    double maximumDisplacement() const noexcept { return maximumDisplacement_; }
    const std::vector<videowire::VertexModifierRecord>& records() const noexcept
    {
        return records_;
    }
    const std::string& structuralDigest() const noexcept { return structuralDigest_; }

private:
    friend std::optional<AdmittedVertexModifierIr> admitVertexModifierIr(
        const videowire::VertexModifierIr&, const AdmissionLimits&, std::string&);

    AdmittedVertexModifierIr(videowire::VertexModifierStableId rootId,
                             std::vector<videowire::VertexModifierRecord> records,
                             std::size_t maximumDepth, double maximumDisplacement,
                             std::string structuralDigest)
        : rootId_(rootId), records_(std::move(records)), maximumDepth_(maximumDepth),
          maximumDisplacement_(maximumDisplacement),
          structuralDigest_(std::move(structuralDigest))
    {
    }

    videowire::VertexModifierStableId rootId_ = 0;
    std::vector<videowire::VertexModifierRecord> records_;
    std::size_t maximumDepth_ = 0;
    double maximumDisplacement_ = 0.0;
    std::string structuralDigest_;
};

namespace detail
{
inline void hashU8(videohelper::Sha256& hash, std::uint8_t value)
{
    hash.update(&value, sizeof(value));
}

inline void hashU32(videohelper::Sha256& hash, std::uint32_t value)
{
    const std::uint8_t bytes[] {
        static_cast<std::uint8_t>(value >> 24u),
        static_cast<std::uint8_t>(value >> 16u),
        static_cast<std::uint8_t>(value >> 8u),
        static_cast<std::uint8_t>(value)
    };
    hash.update(bytes, sizeof(bytes));
}

inline void hashU64(videohelper::Sha256& hash, std::uint64_t value)
{
    std::uint8_t bytes[8];
    for (int index = 0; index < 8; ++index)
        bytes[index] = static_cast<std::uint8_t>(value >> (56 - index * 8));
    hash.update(bytes, sizeof(bytes));
}

inline std::string structuralDigest(
    videowire::VertexModifierStableId rootId,
    const std::vector<videowire::VertexModifierRecord>& records)
{
    static_assert(sizeof(double) == sizeof(std::uint64_t),
                  "vertex modifier digest requires 64-bit doubles");
    static_assert(std::numeric_limits<double>::is_iec559,
                  "vertex modifier digest requires IEEE-754 doubles");
    videohelper::Sha256 hash;
    hashU32(hash, videowire::kVertexModifierIrSchemaVersion);
    hashU64(hash, rootId);
    hashU32(hash, static_cast<std::uint32_t>(records.size()));
    for (const auto& record : records)
    {
        hashU64(hash, record.stableId);
        hashU8(hash, static_cast<std::uint8_t>(record.operation));
        hashU8(hash, static_cast<std::uint8_t>(record.resultType));
        hashU8(hash, record.inputCount);
        for (std::size_t input = 0; input < record.inputCount; ++input)
            hashU64(hash, record.inputs[input]);
        hashU8(hash, record.parameterCount);
        for (std::size_t parameter = 0; parameter < record.parameterCount; ++parameter)
        {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &record.parameters[parameter], sizeof(bits));
            hashU64(hash, bits);
        }
    }
    return hash.finishHex();
}

inline bool isIntegralIndex(double value, std::uint32_t maximum) noexcept
{
    return value >= 0.0 && value <= maximum && std::floor(value) == value;
}

inline bool validateParameters(const videowire::VertexModifierRecord& record,
                               const AdmissionLimits& limits, std::string& error)
{
    using Operation = videowire::VertexModifierOperation;
    const auto fail = [&](const char* reason)
    {
        error = "vertex modifier record " + std::to_string(record.stableId) + " " + reason;
        return false;
    };
    const auto& p = record.parameters;
    switch (record.operation)
    {
        case Operation::audioParameter:
            if (! isIntegralIndex(p[0], limits.maxAudioParameterIndex))
                return fail("requires a bounded integral audio parameter index");
            break;
        case Operation::controlParameter:
            if (! isIntegralIndex(p[0], limits.maxControlParameterIndex))
                return fail("requires a bounded integral control parameter index");
            break;
        case Operation::scalarRemap:
        case Operation::vec3Remap:
            if (p[0] >= p[1] || (p[4] != 0.0 && p[4] != 1.0))
                return fail("requires an increasing input range and a binary clamp flag");
            break;
        case Operation::scalarNoise3d:
        case Operation::vec3Noise3d:
            if (p[0] < limits.minPositiveParameter
                || ! isIntegralIndex(p[1], limits.maxNoiseSeed))
                return fail("requires a positive frequency and bounded integral seed");
            break;
        case Operation::boundedDisplacementOutput:
            if (p[0] < limits.minPositiveParameter
                || p[0] > limits.maxDeclaredDisplacement)
                return fail("requires a positive admitted displacement bound");
            break;
        case Operation::invalid:
        case Operation::importedPosition:
        case Operation::importedNormal:
        case Operation::importedUv:
        case Operation::importedVertexColor:
        case Operation::scalarConstant:
        case Operation::vec3Constant:
        case Operation::timeSeconds:
        case Operation::scalarAdd:
        case Operation::scalarSubtract:
        case Operation::scalarMultiply:
        case Operation::vec3Add:
        case Operation::vec3Subtract:
        case Operation::vec3Multiply:
        case Operation::vec3Scale:
        case Operation::vec3Compose:
        case Operation::componentX:
        case Operation::componentY:
        case Operation::componentZ:
            break;
    }
    return true;
}
} // namespace detail

inline std::optional<AdmittedVertexModifierIr> admitVertexModifierIr(
    const videowire::VertexModifierIr& source, const AdmissionLimits& limits,
    std::string& error)
{
    using Record = videowire::VertexModifierRecord;
    using Operation = videowire::VertexModifierOperation;
    error.clear();

    constexpr std::size_t absoluteOperationLimit = 512;
    constexpr std::size_t absoluteDepthLimit = 64;
    if (limits.maxOperations == 0 || limits.maxOperations > absoluteOperationLimit
        || limits.maxDepth == 0 || limits.maxDepth > absoluteDepthLimit
        || limits.maxParametersPerRecord > videowire::kVertexModifierMaximumParametersPerRecord
        || limits.maxStableId == 0 || ! std::isfinite(limits.maxAbsoluteParameter)
        || ! std::isfinite(limits.minPositiveParameter)
        || ! std::isfinite(limits.maxDeclaredDisplacement)
        || limits.maxAbsoluteParameter <= 0.0 || limits.minPositiveParameter <= 0.0
        || limits.minPositiveParameter > limits.maxAbsoluteParameter
        || limits.maxDeclaredDisplacement < limits.minPositiveParameter
        || limits.maxDeclaredDisplacement > limits.maxAbsoluteParameter)
    {
        error = "vertex modifier admission limits are invalid";
        return std::nullopt;
    }
    if (source.schemaVersion != videowire::kVertexModifierIrSchemaVersion)
    {
        error = "vertex modifier IR schema version is unsupported";
        return std::nullopt;
    }
    if (source.records.empty() || source.records.size() > limits.maxOperations)
    {
        error = "vertex modifier operation capacity exceeded or empty";
        return std::nullopt;
    }
    if (source.rootId == 0 || source.rootId > limits.maxStableId)
    {
        error = "vertex modifier root stable ID is invalid";
        return std::nullopt;
    }

    std::vector<Record> records = source.records;
    std::sort(records.begin(), records.end(), [](const Record& left, const Record& right)
    {
        return left.stableId < right.stableId;
    });
    std::unordered_map<videowire::VertexModifierStableId, std::size_t> indices;
    indices.reserve(records.size());
    for (std::size_t index = 0; index < records.size(); ++index)
    {
        auto& record = records[index];
        if (record.stableId == 0 || record.stableId > limits.maxStableId
            || ! indices.emplace(record.stableId, index).second)
        {
            error = "vertex modifier record has a duplicate or invalid stable ID";
            return std::nullopt;
        }
        const auto schema = videowire::vertexModifierOperationSchema(record.operation);
        if (! schema.valid || record.resultType != schema.resultType
            || record.inputCount != schema.inputCount
            || record.parameterCount != schema.parameterCount)
        {
            error = "vertex modifier record " + std::to_string(record.stableId)
                + " has an invalid typed operation shape";
            return std::nullopt;
        }
        if (record.parameterCount > limits.maxParametersPerRecord)
        {
            error = "vertex modifier parameter capacity exceeded";
            return std::nullopt;
        }
        for (std::size_t input = record.inputCount; input < record.inputs.size(); ++input)
        {
            if (record.inputs[input] != 0)
            {
                error = "vertex modifier record has data outside its declared inputs";
                return std::nullopt;
            }
        }
        for (std::size_t parameter = 0; parameter < record.parameters.size(); ++parameter)
        {
            const auto value = record.parameters[parameter];
            if (parameter >= record.parameterCount)
            {
                if (value != 0.0)
                {
                    error = "vertex modifier record has data outside its declared parameters";
                    return std::nullopt;
                }
                continue;
            }
            if (! std::isfinite(value) || std::abs(value) > limits.maxAbsoluteParameter)
            {
                error = "vertex modifier record parameter is non-finite or out of bounds";
                return std::nullopt;
            }
            if (value == 0.0) record.parameters[parameter] = 0.0;
        }
        if (! detail::validateParameters(record, limits, error)) return std::nullopt;
    }

    const auto rootFound = indices.find(source.rootId);
    if (rootFound == indices.end())
    {
        error = "vertex modifier root stable ID does not name a record";
        return std::nullopt;
    }
    const auto rootIndex = rootFound->second;
    if (records[rootIndex].operation != Operation::boundedDisplacementOutput)
    {
        error = "vertex modifier root must be the bounded displacement output";
        return std::nullopt;
    }

    for (std::size_t index = 0; index < records.size(); ++index)
    {
        const auto& record = records[index];
        const auto schema = videowire::vertexModifierOperationSchema(record.operation);
        if (schema.terminal && index != rootIndex)
        {
            error = "vertex modifier contains a non-root terminal output";
            return std::nullopt;
        }
        for (std::size_t input = 0; input < record.inputCount; ++input)
        {
            const auto found = indices.find(record.inputs[input]);
            if (record.inputs[input] == 0 || found == indices.end())
            {
                error = "vertex modifier record references a missing stable ID";
                return std::nullopt;
            }
            if (records[found->second].resultType != schema.inputTypes[input])
            {
                error = "vertex modifier record has an incompatible input type";
                return std::nullopt;
            }
        }
    }
    const auto basePositionIndex = indices.at(records[rootIndex].inputs[0]);
    if (records[basePositionIndex].operation != Operation::importedPosition)
    {
        error = "vertex modifier output requires imported position as its direct base input";
        return std::nullopt;
    }

    std::vector<std::uint8_t> colors(records.size(), 0);
    std::vector<std::size_t> depths(records.size(), 0);
    std::function<bool(std::size_t)> visit = [&](std::size_t index)
    {
        if (colors[index] == 1)
        {
            error = "vertex modifier IR contains a cycle";
            return false;
        }
        if (colors[index] == 2) return true;
        colors[index] = 1;
        std::size_t depth = 1;
        for (std::size_t input = 0; input < records[index].inputCount; ++input)
        {
            const auto child = indices.at(records[index].inputs[input]);
            if (! visit(child)) return false;
            depth = std::max(depth, depths[child] + 1);
        }
        colors[index] = 2;
        depths[index] = depth;
        return true;
    };
    for (std::size_t index = 0; index < records.size(); ++index)
        if (! visit(index)) return std::nullopt;

    std::vector<bool> rootReachable(records.size(), false);
    std::function<void(std::size_t)> markReachable = [&](std::size_t index)
    {
        if (rootReachable[index]) return;
        rootReachable[index] = true;
        for (std::size_t input = 0; input < records[index].inputCount; ++input)
            markReachable(indices.at(records[index].inputs[input]));
    };
    markReachable(rootIndex);
    if (std::find(rootReachable.begin(), rootReachable.end(), false) != rootReachable.end())
    {
        error = "vertex modifier IR contains a record unreachable from its root";
        return std::nullopt;
    }
    if (depths[rootIndex] > limits.maxDepth)
    {
        error = "vertex modifier depth capacity exceeded: "
            + std::to_string(depths[rootIndex]) + " > " + std::to_string(limits.maxDepth);
        return std::nullopt;
    }

    const auto maximumDisplacement = records[rootIndex].parameters[0];
    auto digest = detail::structuralDigest(source.rootId, records);
    return AdmittedVertexModifierIr(source.rootId, std::move(records), depths[rootIndex],
                                    maximumDisplacement, std::move(digest));
}

} // namespace videohelper::vertexmodifier
