#pragma once

#include "../../shared/LumaKeyContract.h"
#include "effect_defs.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <string>
#include <string_view>
#include <unordered_map>

namespace videowire
{
enum class LumaKeyPayloadFailure : std::uint8_t
{
    none = 0,
    malformedPayload,
    unsupportedSchema,
    invalidRange,
    invalidParameter
};

// Values for the native LumaKey effect slot: low, high, softness, invert.
struct ExecutableLumaKeyPayload
{
    float low = 0.0f;
    float high = 1.0f;
    float softness = 0.1f;
    float invert = 0.0f;
};

inline bool parseLumaKeyAttributes(
    std::string_view xml, std::unordered_map<std::string, std::string>& attributes)
{
    size_t cursor = 0;
    const auto skip = [&]
    {
        while (cursor < xml.size() && std::isspace(static_cast<unsigned char>(xml[cursor])))
            ++cursor;
    };
    const auto take = [&](std::string_view token)
    {
        if (xml.substr(cursor, token.size()) != token) return false;
        cursor += token.size();
        return true;
    };
    skip();
    if (!take("<LumaKey") || (cursor < xml.size()
        && !std::isspace(static_cast<unsigned char>(xml[cursor])))) return false;
    while (true)
    {
        skip();
        if (take("/>")) break;
        const size_t nameStart = cursor;
        while (cursor < xml.size() && (std::isalnum(static_cast<unsigned char>(xml[cursor]))
            || xml[cursor] == '_')) ++cursor;
        if (cursor == nameStart) return false;
        const std::string name(xml.substr(nameStart, cursor - nameStart));
        skip();
        if (!take("=")) return false;
        skip();
        if (cursor >= xml.size() || xml[cursor] != '"') return false;
        const size_t valueStart = ++cursor;
        while (cursor < xml.size() && xml[cursor] != '"')
            if (xml[cursor++] == '<' || xml[cursor - 1] == '&') return false;
        if (cursor >= xml.size()) return false;
        if (!attributes.emplace(name,
                std::string(xml.substr(valueStart, cursor - valueStart))).second) return false;
        ++cursor;
    }
    skip();
    return cursor == xml.size();
}

inline bool parseExecutableLumaKeyPayload(
    std::string_view xml, ExecutableLumaKeyPayload& out,
    LumaKeyPayloadFailure* payloadFailure = nullptr)
{
    const auto fail = [&](LumaKeyPayloadFailure value)
    {
        if (payloadFailure != nullptr) *payloadFailure = value;
        return false;
    };
    if (payloadFailure != nullptr) *payloadFailure = LumaKeyPayloadFailure::none;

    std::unordered_map<std::string, std::string> attributes;
    if (!parseLumaKeyAttributes(xml, attributes) || attributes.size() != 5)
        return fail(LumaKeyPayloadFailure::malformedPayload);
    const auto number = [&](const char* name, double& target)
    {
        const auto found = attributes.find(name);
        if (found == attributes.end() || found->second.empty()) return false;
        char* end = nullptr;
        errno = 0;
        const double value = std::strtod(found->second.c_str(), &end);
        if (errno != 0 || end != found->second.c_str() + found->second.size()
            || !std::isfinite(value))
            return false;
        target = value;
        return true;
    };

    lumakey::Description description;
    double schema = 0.0;
    if (!number("schemaVersion", schema))
        return fail(LumaKeyPayloadFailure::malformedPayload);
    if (schema != lumakey::kSchemaVersion)
        return fail(LumaKeyPayloadFailure::unsupportedSchema);
    if (!number("low", description.low)
        || !number("high", description.high)
        || !number("softness", description.softness)
        || !number("invert", description.invert))
        return fail(LumaKeyPayloadFailure::malformedPayload);

    lumakey::AdmissionFailure admissionFailure = lumakey::AdmissionFailure::none;
    if (!lumakey::admit(description, admissionFailure))
    {
        switch (admissionFailure)
        {
            case lumakey::AdmissionFailure::unsupportedSchema:
                return fail(LumaKeyPayloadFailure::unsupportedSchema);
            case lumakey::AdmissionFailure::invalidRange:
                return fail(LumaKeyPayloadFailure::invalidRange);
            default:
                return fail(LumaKeyPayloadFailure::invalidParameter);
        }
    }

    out.low = static_cast<float>(description.low);
    out.high = static_cast<float>(description.high);
    out.softness = static_cast<float>(description.softness);
    out.invert = static_cast<float>(description.invert);
    return true;
}
} // namespace videowire
