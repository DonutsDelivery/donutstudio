#pragma once

#include "SdfIr.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>

namespace videowire
{
inline constexpr std::uint32_t kSdfRaymarchOperationSchemaVersion = 1;
inline constexpr std::uint32_t kSdfRaymarchTextOperationSchemaVersion = 2;
inline constexpr std::size_t kSdfRaymarchMaximumRecords = 256;
inline constexpr std::string_view kSdfRaymarchOperationPrefix = "sdf-raymarch-v1:";
inline constexpr std::string_view kSdfRaymarchTextOperationPrefix = "sdf-raymarch-v2:";

struct SdfRaymarchOperation final
{
    std::uint64_t terminalStableId = 0;
    std::uint32_t maximumSteps = 128;
    double epsilon = 0.001;
    double maximumDistance = 100.0;
    std::uint8_t adaptiveQuality = 1;
    std::uint8_t normalQuality = 1;
    std::uint8_t shadowQuality = 1;
    std::uint8_t outputPass = 0;
    SdfIr geometry;
};

// Renderer control ceilings. Graph lowering, wire admission and both native
// backends (OpenGL and Metal capabilities) use these same values.
inline constexpr std::uint32_t kSdfRaymarchMaximumSteps = 512;
inline constexpr std::uint8_t kSdfRaymarchMaximumQuality = 3;
inline constexpr std::uint8_t kSdfRaymarchMaximumOutputPass = 7;
inline constexpr double kSdfRaymarchMinimumEpsilon = 0.000001;
inline constexpr double kSdfRaymarchMaximumEpsilon = 0.1;
inline constexpr double kSdfRaymarchMaximumDistance = 1000.0;

// Written as negated comparisons so NaN and infinities are refused.
inline bool sdfRaymarchControlsAdmissible (const SdfRaymarchOperation& operation)
{
    return operation.maximumSteps >= 1 && operation.maximumSteps <= kSdfRaymarchMaximumSteps
        && operation.epsilon >= kSdfRaymarchMinimumEpsilon
        && operation.epsilon <= kSdfRaymarchMaximumEpsilon
        && operation.maximumDistance >= operation.epsilon
        && operation.maximumDistance <= kSdfRaymarchMaximumDistance
        && operation.adaptiveQuality <= kSdfRaymarchMaximumQuality
        && operation.normalQuality <= kSdfRaymarchMaximumQuality
        && operation.shadowQuality <= kSdfRaymarchMaximumQuality
        && operation.outputPass <= kSdfRaymarchMaximumOutputPass;
}

namespace sdfoperationdetail
{
inline void appendU8 (std::string& bytes, std::uint8_t value)
{
    bytes.push_back (static_cast<char> (value));
}

inline void appendU32 (std::string& bytes, std::uint32_t value)
{
    for (unsigned shift = 0; shift < 32; shift += 8)
        appendU8 (bytes, static_cast<std::uint8_t> (value >> shift));
}

inline void appendU64 (std::string& bytes, std::uint64_t value)
{
    for (unsigned shift = 0; shift < 64; shift += 8)
        appendU8 (bytes, static_cast<std::uint8_t> (value >> shift));
}

inline void appendDouble (std::string& bytes, double value)
{
    static_assert (sizeof (double) == sizeof (std::uint64_t));
    std::uint64_t bits = 0;
    std::memcpy (&bits, &value, sizeof (bits));
    appendU64 (bytes, bits);
}

inline bool readU8 (std::string_view bytes, std::size_t& offset, std::uint8_t& value)
{
    if (offset >= bytes.size()) return false;
    value = static_cast<std::uint8_t> (bytes[offset++]);
    return true;
}

inline bool readU32 (std::string_view bytes, std::size_t& offset, std::uint32_t& value)
{
    value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8)
    {
        std::uint8_t byte = 0;
        if (! readU8 (bytes, offset, byte)) return false;
        value |= static_cast<std::uint32_t> (byte) << shift;
    }
    return true;
}

inline bool readU64 (std::string_view bytes, std::size_t& offset, std::uint64_t& value)
{
    value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
    {
        std::uint8_t byte = 0;
        if (! readU8 (bytes, offset, byte)) return false;
        value |= static_cast<std::uint64_t> (byte) << shift;
    }
    return true;
}

inline bool readDouble (std::string_view bytes, std::size_t& offset, double& value)
{
    std::uint64_t bits = 0;
    if (! readU64 (bytes, offset, bits)) return false;
    std::memcpy (&value, &bits, sizeof (value));
    return true;
}

inline char hexDigit (std::uint8_t value)
{
    return value < 10 ? static_cast<char> ('0' + value)
                      : static_cast<char> ('a' + value - 10);
}

inline int hexValue (char value)
{
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    return -1;
}
} // namespace sdfoperationdetail

inline std::string encodeSdfRaymarchOperation (const SdfRaymarchOperation& operation)
{
    if (operation.geometry.records.size() > kSdfRaymarchMaximumRecords
        || ! sdfRaymarchControlsAdmissible (operation))
        return {};

    const bool hasText = ! operation.geometry.textOutlines.empty();
    if (hasText)
    {
        std::string error;
        SdfTextWorkBudget work;
        if (! chargeSdfTextIrEvaluationWork (operation.geometry, 1, work, error)) return {};
    }
    else
    {
        if (operation.geometry.schemaVersion == kSdfIrTextSchemaVersion) return {};
        for (const auto& record : operation.geometry.records)
            if (record.resourceStableId != 0 || record.operation == SdfOperation::textExtrusion) return {};
    }
    std::string bytes;
    bytes.reserve (64 + operation.geometry.records.size() * 100);
    sdfoperationdetail::appendU32 (bytes, hasText ? kSdfRaymarchTextOperationSchemaVersion
                                               : kSdfRaymarchOperationSchemaVersion);
    sdfoperationdetail::appendU64 (bytes, operation.terminalStableId);
    sdfoperationdetail::appendU32 (bytes, operation.maximumSteps);
    sdfoperationdetail::appendDouble (bytes, operation.epsilon);
    sdfoperationdetail::appendDouble (bytes, operation.maximumDistance);
    sdfoperationdetail::appendU8 (bytes, operation.adaptiveQuality);
    sdfoperationdetail::appendU8 (bytes, operation.normalQuality);
    sdfoperationdetail::appendU8 (bytes, operation.shadowQuality);
    sdfoperationdetail::appendU8 (bytes, operation.outputPass);
    sdfoperationdetail::appendU32 (bytes, operation.geometry.schemaVersion);
    sdfoperationdetail::appendU64 (bytes, operation.geometry.rootId);
    sdfoperationdetail::appendU32 (bytes,
        static_cast<std::uint32_t> (operation.geometry.records.size()));
    for (const auto& record : operation.geometry.records)
    {
        sdfoperationdetail::appendU64 (bytes, record.stableId);
        sdfoperationdetail::appendU8 (bytes, static_cast<std::uint8_t> (record.operation));
        sdfoperationdetail::appendU8 (bytes, record.inputCount);
        sdfoperationdetail::appendU8 (bytes, record.parameterCount);
        sdfoperationdetail::appendU8 (bytes, 0);
        for (const auto input : record.inputs)
            sdfoperationdetail::appendU64 (bytes, input);
        for (const auto parameter : record.parameters)
            sdfoperationdetail::appendDouble (bytes, parameter);
        if (hasText) sdfoperationdetail::appendU64 (bytes, record.resourceStableId);
    }
    if (hasText)
    {
        sdfoperationdetail::appendU32 (bytes, static_cast<std::uint32_t> (operation.geometry.textOutlines.size()));
        for (const auto& resource : operation.geometry.textOutlines)
        {
            const auto payload = encodeSdfTextOutline (resource);
            sdfoperationdetail::appendU32 (bytes, static_cast<std::uint32_t> (payload.size()));
            bytes += payload;
        }
    }

    std::string encoded (hasText ? kSdfRaymarchTextOperationPrefix : kSdfRaymarchOperationPrefix);
    encoded.reserve (encoded.size() + bytes.size() * 2);
    for (const unsigned char byte : bytes)
    {
        encoded.push_back (sdfoperationdetail::hexDigit (byte >> 4));
        encoded.push_back (sdfoperationdetail::hexDigit (byte & 0x0f));
    }
    return encoded;
}

inline bool decodeSdfRaymarchOperation (std::string_view encoded,
                                        SdfRaymarchOperation& destination)
{
    const bool hasText = encoded.size() >= kSdfRaymarchTextOperationPrefix.size()
        && encoded.compare (0, kSdfRaymarchTextOperationPrefix.size(), kSdfRaymarchTextOperationPrefix) == 0;
    const auto prefix = hasText ? kSdfRaymarchTextOperationPrefix : kSdfRaymarchOperationPrefix;
    if (encoded.size() < prefix.size() || encoded.compare (0, prefix.size(), prefix) != 0) return false;
    const auto hex = encoded.substr (prefix.size());
    const std::size_t maximumHexBytes = hasText
        ? 112 + kSdfRaymarchMaximumRecords * 208 + kSdfTextMaximumPayloadBytes * 2
        : 104 + kSdfRaymarchMaximumRecords * 184;
    if ((hex.size() & 1u) != 0 || hex.size() > maximumHexBytes)
        return false;

    std::string bytes;
    bytes.reserve (hex.size() / 2);
    for (std::size_t index = 0; index < hex.size(); index += 2)
    {
        const int high = sdfoperationdetail::hexValue (hex[index]);
        const int low = sdfoperationdetail::hexValue (hex[index + 1]);
        if (high < 0 || low < 0) return false;
        bytes.push_back (static_cast<char> ((high << 4) | low));
    }

    SdfRaymarchOperation candidate;
    std::size_t offset = 0;
    std::uint32_t schema = 0;
    std::uint32_t recordCount = 0;
    if (! sdfoperationdetail::readU32 (bytes, offset, schema)
        || schema != (hasText ? kSdfRaymarchTextOperationSchemaVersion : kSdfRaymarchOperationSchemaVersion)
        || ! sdfoperationdetail::readU64 (bytes, offset, candidate.terminalStableId)
        || ! sdfoperationdetail::readU32 (bytes, offset, candidate.maximumSteps)
        || ! sdfoperationdetail::readDouble (bytes, offset, candidate.epsilon)
        || ! sdfoperationdetail::readDouble (bytes, offset, candidate.maximumDistance)
        || ! sdfoperationdetail::readU8 (bytes, offset, candidate.adaptiveQuality)
        || ! sdfoperationdetail::readU8 (bytes, offset, candidate.normalQuality)
        || ! sdfoperationdetail::readU8 (bytes, offset, candidate.shadowQuality)
        || ! sdfoperationdetail::readU8 (bytes, offset, candidate.outputPass)
        || ! sdfoperationdetail::readU32 (bytes, offset, candidate.geometry.schemaVersion)
        || ! sdfoperationdetail::readU64 (bytes, offset, candidate.geometry.rootId)
        || ! sdfoperationdetail::readU32 (bytes, offset, recordCount)
        || recordCount > kSdfRaymarchMaximumRecords)
        return false;

    const std::size_t recordBytes = 8 + 4
        + kSdfMaximumInputsPerRecord * 8 + kSdfMaximumParametersPerRecord * 8 + (hasText ? 8 : 0);
    if (recordCount > (bytes.size() - offset) / recordBytes
        || (! hasText && bytes.size() - offset != static_cast<std::size_t> (recordCount) * recordBytes))
        return false;

    candidate.geometry.records.resize (recordCount);
    for (auto& record : candidate.geometry.records)
    {
        std::uint8_t operation = 0;
        std::uint8_t reserved = 0;
        if (! sdfoperationdetail::readU64 (bytes, offset, record.stableId)
            || ! sdfoperationdetail::readU8 (bytes, offset, operation)
            || ! sdfoperationdetail::readU8 (bytes, offset, record.inputCount)
            || ! sdfoperationdetail::readU8 (bytes, offset, record.parameterCount)
            || ! sdfoperationdetail::readU8 (bytes, offset, reserved)
            || reserved != 0)
            return false;
        record.operation = static_cast<SdfOperation> (operation);
        for (auto& input : record.inputs)
            if (! sdfoperationdetail::readU64 (bytes, offset, input)) return false;
        for (auto& parameter : record.parameters)
            if (! sdfoperationdetail::readDouble (bytes, offset, parameter)) return false;
        if (hasText && ! sdfoperationdetail::readU64 (bytes, offset, record.resourceStableId)) return false;
    }
    if (hasText)
    {
        std::uint32_t resourceCount = 0;
        if (! sdfoperationdetail::readU32 (bytes, offset, resourceCount)
            || resourceCount == 0 || resourceCount > kSdfRaymarchMaximumRecords
            || resourceCount > (bytes.size() - offset) / 4) return false;
        std::size_t payloadBytes = 0;
        candidate.geometry.textOutlines.reserve (resourceCount);
        for (std::uint32_t index = 0; index < resourceCount; ++index)
        {
            std::uint32_t count = 0;
            if (! sdfoperationdetail::readU32 (bytes, offset, count)
                || count > bytes.size() - offset || count > kSdfTextMaximumPayloadBytes - payloadBytes) return false;
            SdfTextOutline resource;
            if (! decodeSdfTextOutline (std::string_view (bytes).substr (offset, count), resource)) return false;
            offset += count;
            payloadBytes += count;
            candidate.geometry.textOutlines.push_back (std::move (resource));
        }
        std::string error;
        SdfTextWorkBudget work;
        if (! chargeSdfTextIrEvaluationWork (candidate.geometry, 1, work, error)) return false;
    }
    if (offset != bytes.size() || ! sdfRaymarchControlsAdmissible (candidate)) return false;
    destination = std::move (candidate);
    return true;
}
} // namespace videowire
