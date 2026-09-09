#pragma once

#include "../../shared/SdfIr.h"
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

namespace videohelper::sdf
{
struct SdfAdmissionLimits final
{
    std::size_t maxOperations = 256;
    std::size_t maxDepth = 64;
    std::size_t maxParametersPerRecord = videowire::kSdfMaximumParametersPerRecord;
    videowire::SdfStableId maxStableId = 9007199254740991ull; // exact in JSON numbers
    double maxAbsoluteParameter = 1000000.0;
    double minPositiveParameter = 0.000001;
    std::uint32_t maxPolarRepeatCount = 1024;
};

class AdmittedSdfIr final
{
public:
    AdmittedSdfIr (const AdmittedSdfIr&) = default;
    AdmittedSdfIr (AdmittedSdfIr&&) noexcept = default;
    AdmittedSdfIr& operator= (const AdmittedSdfIr&) = default;
    AdmittedSdfIr& operator= (AdmittedSdfIr&&) noexcept = default;

    std::uint32_t schemaVersion() const noexcept { return videowire::kSdfIrSchemaVersion; }
    videowire::SdfStableId rootId() const noexcept { return rootId_; }
    std::size_t operationCount() const noexcept { return records_.size(); }
    std::size_t maximumDepth() const noexcept { return maximumDepth_; }
    const std::vector<videowire::SdfRecord>& records() const noexcept { return records_; }
    const std::string& structuralDigest() const noexcept { return structuralDigest_; }

private:
    friend std::optional<AdmittedSdfIr> admitSdfIr (
        const videowire::SdfIr&, const SdfAdmissionLimits&, std::string&);

    AdmittedSdfIr (videowire::SdfStableId rootId,
                   std::vector<videowire::SdfRecord> records,
                   std::size_t maximumDepth,
                   std::string structuralDigest)
        : rootId_ (rootId), records_ (std::move (records)), maximumDepth_ (maximumDepth),
          structuralDigest_ (std::move (structuralDigest))
    {
    }

    videowire::SdfStableId rootId_ = 0;
    std::vector<videowire::SdfRecord> records_;
    std::size_t maximumDepth_ = 0;
    std::string structuralDigest_;
};

namespace detail
{
inline void hashU8 (Sha256& hash, std::uint8_t value)
{
    hash.update (&value, sizeof (value));
}

inline void hashU32 (Sha256& hash, std::uint32_t value)
{
    const std::uint8_t bytes[] {
        static_cast<std::uint8_t> (value >> 24u), static_cast<std::uint8_t> (value >> 16u),
        static_cast<std::uint8_t> (value >> 8u), static_cast<std::uint8_t> (value)
    };
    hash.update (bytes, sizeof (bytes));
}

inline void hashU64 (Sha256& hash, std::uint64_t value)
{
    std::uint8_t bytes[8];
    for (int index = 0; index < 8; ++index)
        bytes[index] = static_cast<std::uint8_t> (value >> (56 - index * 8));
    hash.update (bytes, sizeof (bytes));
}

inline std::string structuralDigest (videowire::SdfStableId rootId,
                                     const std::vector<videowire::SdfRecord>& records)
{
    static_assert (sizeof (double) == sizeof (std::uint64_t), "SDF digest requires 64-bit doubles");
    static_assert (std::numeric_limits<double>::is_iec559, "SDF digest requires IEEE-754 doubles");
    Sha256 hash;
    hashU32 (hash, videowire::kSdfIrSchemaVersion);
    hashU64 (hash, rootId);
    hashU32 (hash, static_cast<std::uint32_t> (records.size()));
    for (const auto& record : records)
    {
        hashU64 (hash, record.stableId);
        hashU8 (hash, static_cast<std::uint8_t> (record.operation));
        hashU8 (hash, record.inputCount);
        for (std::size_t input = 0; input < record.inputCount; ++input)
            hashU64 (hash, record.inputs[input]);
        hashU8 (hash, record.parameterCount);
        for (std::size_t parameter = 0; parameter < record.parameterCount; ++parameter)
        {
            std::uint64_t bits = 0;
            std::memcpy (&bits, &record.parameters[parameter], sizeof (bits));
            hashU64 (hash, bits);
        }
    }
    return hash.finishHex();
}

inline bool nonZeroVector (const videowire::SdfRecord& record, std::size_t first,
                           double minimum) noexcept
{
    return std::hypot (record.parameters[first], record.parameters[first + 1],
                       record.parameters[first + 2]) >= minimum;
}

inline bool positive (double value, double minimum) noexcept { return value >= minimum; }
inline bool axis (double value) noexcept { return value == 0.0 || value == 1.0 || value == 2.0; }
inline bool binaryFlag (double value) noexcept { return value == 0.0 || value == 1.0; }

inline bool validateParameters (const videowire::SdfRecord& record,
                                const SdfAdmissionLimits& limits, std::string& error)
{
    const auto fail = [&] (const char* reason)
    {
        error = "SDF record " + std::to_string (record.stableId) + " " + reason;
        return false;
    };
    const auto& p = record.parameters;
    const auto allPositive = [&] (std::size_t first, std::size_t count)
    {
        for (std::size_t index = first; index < first + count; ++index)
            if (! positive (p[index], limits.minPositiveParameter)) return false;
        return true;
    };

    switch (record.operation)
    {
        case videowire::SdfOperation::sphere:
        case videowire::SdfOperation::smoothUnion:
        case videowire::SdfOperation::smoothIntersection:
        case videowire::SdfOperation::smoothSubtraction:
            if (! allPositive (0, 1)) return fail ("requires a positive radius");
            break;
        case videowire::SdfOperation::box:
            if (! allPositive (0, 3)) return fail ("requires positive half extents");
            break;
        case videowire::SdfOperation::roundedBox:
            if (! allPositive (0, 4)) return fail ("requires positive half extents and radius");
            if (p[3] > std::min ({ p[0], p[1], p[2] }))
                return fail ("radius exceeds the minimum half extent");
            break;
        case videowire::SdfOperation::plane:
            if (! nonZeroVector (record, 0, limits.minPositiveParameter))
                return fail ("requires a non-zero normal");
            break;
        case videowire::SdfOperation::torus:
            if (! allPositive (0, 2))
                return fail ("requires positive major and minor radii");
            break;
        case videowire::SdfOperation::cylinder:
            if (! allPositive (0, 2))
                return fail ("requires a positive radius and half height");
            break;
        case videowire::SdfOperation::cone:
            if (! allPositive (0, 2)) return fail ("requires positive dimensions");
            break;
        case videowire::SdfOperation::capsule:
            if (! positive (p[6], limits.minPositiveParameter)
                || std::hypot (p[3] - p[0], p[4] - p[1], p[5] - p[2])
                       < limits.minPositiveParameter)
                return fail ("requires distinct endpoints and a positive radius");
            break;
        case videowire::SdfOperation::gyroid:
            if (! allPositive (0, 2))
                return fail ("requires positive reciprocal cell scale and thickness");
            break;
        case videowire::SdfOperation::rotate:
            if (! nonZeroVector (record, 0, limits.minPositiveParameter))
                return fail ("requires a non-zero rotation axis");
            break;
        case videowire::SdfOperation::scale:
            for (std::size_t index = 0; index < 3; ++index)
                if (std::abs (p[index]) < limits.minPositiveParameter)
                    return fail ("requires non-zero scale components");
            break;
        case videowire::SdfOperation::repeat:
            if (! allPositive (0, 3)) return fail ("requires positive repeat periods");
            break;
        case videowire::SdfOperation::polarRepeat:
            if (! axis (p[0]) || p[1] < 1.0 || p[1] > limits.maxPolarRepeatCount
                || std::floor (p[1]) != p[1])
                return fail ("requires an axis and bounded integral repeat count");
            break;
        case videowire::SdfOperation::mirror:
            if (! binaryFlag (p[0]) || ! binaryFlag (p[1]) || ! binaryFlag (p[2])
                || (p[0] == 0.0 && p[1] == 0.0 && p[2] == 0.0))
                return fail ("requires at least one mirrored axis");
            break;
        case videowire::SdfOperation::twist:
        case videowire::SdfOperation::bend:
        case videowire::SdfOperation::taper:
            if (! axis (p[0])) return fail ("requires an axis in the range 0..2");
            break;
        case videowire::SdfOperation::displacement:
            if (! positive (p[1], limits.minPositiveParameter))
                return fail ("requires a positive displacement frequency");
            break;
        case videowire::SdfOperation::domainWarp:
            if (! allPositive (3, 3)) return fail ("requires positive warp frequencies");
            break;
        case videowire::SdfOperation::unionOp:
        case videowire::SdfOperation::intersection:
        case videowire::SdfOperation::subtraction:
        case videowire::SdfOperation::translate:
            break;
    }
    return true;
}
} // namespace detail

inline std::optional<AdmittedSdfIr> admitSdfIr (const videowire::SdfIr& source,
                                                 const SdfAdmissionLimits& limits,
                                                 std::string& error)
{
    using videowire::SdfRecord;
    error.clear();
    constexpr std::size_t absoluteOperationLimit = 1024;
    constexpr std::size_t absoluteDepthLimit = 128;
    if (limits.maxOperations == 0 || limits.maxOperations > absoluteOperationLimit
        || limits.maxDepth == 0 || limits.maxDepth > absoluteDepthLimit
        || limits.maxParametersPerRecord > videowire::kSdfMaximumParametersPerRecord
        || limits.maxStableId == 0 || ! std::isfinite (limits.maxAbsoluteParameter)
        || ! std::isfinite (limits.minPositiveParameter)
        || limits.maxAbsoluteParameter <= 0.0 || limits.minPositiveParameter <= 0.0
        || limits.minPositiveParameter > limits.maxAbsoluteParameter
        || limits.maxPolarRepeatCount == 0)
    {
        error = "SDF admission limits are invalid";
        return std::nullopt;
    }
    if (source.schemaVersion != videowire::kSdfIrSchemaVersion)
    {
        error = "SDF IR schema version is unsupported";
        return std::nullopt;
    }
    if (source.records.empty() || source.records.size() > limits.maxOperations)
    {
        error = "SDF operation capacity exceeded or empty";
        return std::nullopt;
    }
    if (source.rootId == 0 || source.rootId > limits.maxStableId)
    {
        error = "SDF root stable ID is invalid";
        return std::nullopt;
    }

    std::vector<SdfRecord> records = source.records;
    std::sort (records.begin(), records.end(), [] (const SdfRecord& left, const SdfRecord& right)
        { return left.stableId < right.stableId; });
    std::unordered_map<videowire::SdfStableId, std::size_t> indices;
    indices.reserve (records.size());
    for (std::size_t index = 0; index < records.size(); ++index)
    {
        auto& record = records[index];
        if (record.stableId == 0 || record.stableId > limits.maxStableId
            || ! indices.emplace (record.stableId, index).second)
        {
            error = "SDF record has a duplicate or invalid stable ID";
            return std::nullopt;
        }
        const auto schema = videowire::sdfOperationSchema (record.operation);
        if (! schema.valid || record.inputCount != schema.inputCount
            || record.parameterCount != schema.parameterCount)
        {
            error = "SDF record " + std::to_string (record.stableId)
                + " has an invalid operation shape";
            return std::nullopt;
        }
        if (record.parameterCount > limits.maxParametersPerRecord)
        {
            error = "SDF parameter capacity exceeded";
            return std::nullopt;
        }
        for (std::size_t input = record.inputCount; input < record.inputs.size(); ++input)
            if (record.inputs[input] != 0)
            {
                error = "SDF record has data outside its declared inputs";
                return std::nullopt;
            }
        for (std::size_t parameter = 0; parameter < record.parameters.size(); ++parameter)
        {
            const auto value = record.parameters[parameter];
            if (parameter >= record.parameterCount)
            {
                if (value != 0.0)
                {
                    error = "SDF record has data outside its declared parameters";
                    return std::nullopt;
                }
                continue;
            }
            if (! std::isfinite (value) || std::abs (value) > limits.maxAbsoluteParameter)
            {
                error = "SDF record parameter is non-finite or out of bounds";
                return std::nullopt;
            }
            if (value == 0.0) record.parameters[parameter] = 0.0; // canonicalize negative zero
        }
        if (! detail::validateParameters (record, limits, error)) return std::nullopt;
    }

    if (indices.count (source.rootId) == 0)
    {
        error = "SDF root stable ID does not name a record";
        return std::nullopt;
    }
    for (const auto& record : records)
        for (std::size_t input = 0; input < record.inputCount; ++input)
            if (record.inputs[input] == 0 || indices.count (record.inputs[input]) == 0)
            {
                error = "SDF record references a missing stable ID";
                return std::nullopt;
            }

    std::vector<std::uint8_t> colors (records.size(), 0);
    std::vector<std::size_t> depths (records.size(), 0);
    std::vector<bool> rootReachable (records.size(), false);
    std::function<bool (std::size_t)> visit = [&] (std::size_t index)
    {
        if (colors[index] == 1)
        {
            error = "SDF IR contains a cycle";
            return false;
        }
        if (colors[index] == 2) return true;
        colors[index] = 1;
        std::size_t depth = 1;
        for (std::size_t input = 0; input < records[index].inputCount; ++input)
        {
            const auto child = indices.at (records[index].inputs[input]);
            if (! visit (child)) return false;
            depth = std::max (depth, depths[child] + 1);
        }
        colors[index] = 2;
        depths[index] = depth;
        return true;
    };
    for (std::size_t index = 0; index < records.size(); ++index)
        if (! visit (index)) return std::nullopt;

    std::function<void (std::size_t)> markReachable = [&] (std::size_t index)
    {
        if (rootReachable[index]) return;
        rootReachable[index] = true;
        for (std::size_t input = 0; input < records[index].inputCount; ++input)
            markReachable (indices.at (records[index].inputs[input]));
    };
    const auto rootIndex = indices.at (source.rootId);
    markReachable (rootIndex);
    if (std::find (rootReachable.begin(), rootReachable.end(), false) != rootReachable.end())
    {
        error = "SDF IR contains a record unreachable from its root";
        return std::nullopt;
    }
    if (depths[rootIndex] > limits.maxDepth)
    {
        error = "SDF depth capacity exceeded: " + std::to_string (depths[rootIndex])
            + " > " + std::to_string (limits.maxDepth);
        return std::nullopt;
    }

    auto digest = detail::structuralDigest (source.rootId, records);
    return AdmittedSdfIr (source.rootId, std::move (records), depths[rootIndex], std::move (digest));
}
} // namespace videohelper::sdf
