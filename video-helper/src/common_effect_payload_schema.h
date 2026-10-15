#pragma once

#include "../../shared/CommonEffectContract.h"

#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <string>
#include <string_view>
#include <unordered_map>

namespace videowire
{
enum class CommonEffectPayloadFailure : std::uint8_t
{
    none = 0,
    malformedPayload,
    unsupportedSchema,
    invalidParameter,
    productionBackendUnavailable
};

struct ExecutableCommonEffectPayload
{
    commoneffect::Kind kind = commoneffect::Kind::sharpen;
    int rendererEffectType = -1;
    std::size_t parameterCount = 0;
    std::array<float, commoneffect::kMaximumParameterCount> parameters {};
};

inline bool parseCommonEffectAttributes(
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
    if (!take("<CommonEffect") || (cursor < xml.size()
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

inline bool parseExecutableCommonEffectPayload(
    std::string_view xml, ExecutableCommonEffectPayload& out,
    CommonEffectPayloadFailure* payloadFailure = nullptr,
    commoneffect::BackendSupport backend = commoneffect::currentProductionBackendSupport())
{
    const auto fail = [&](CommonEffectPayloadFailure value)
    {
        if (payloadFailure != nullptr) *payloadFailure = value;
        return false;
    };
    if (payloadFailure != nullptr) *payloadFailure = CommonEffectPayloadFailure::none;

    std::unordered_map<std::string, std::string> attributes;
    if (!parseCommonEffectAttributes(xml, attributes))
        return fail(CommonEffectPayloadFailure::malformedPayload);
    const auto get = [&](const char* name) -> const std::string*
    {
        const auto found = attributes.find(name);
        return found == attributes.end() ? nullptr : &found->second;
    };
    const auto number = [&](const char* name, double& target)
    {
        const auto* text = get(name);
        if (text == nullptr || text->empty()) return false;
        char* end = nullptr;
        errno = 0;
        const double value = std::strtod(text->c_str(), &end);
        if (errno != 0 || end != text->c_str() + text->size() || !std::isfinite(value))
            return false;
        target = value;
        return true;
    };

    double schema = 0.0;
    const auto* effect = get("effect");
    if (!number("schemaVersion", schema) || effect == nullptr)
        return fail(CommonEffectPayloadFailure::malformedPayload);
    const auto kind = commoneffect::kindForWireName(*effect);
    if (!kind.has_value())
        return fail(CommonEffectPayloadFailure::malformedPayload);
    if (schema != commoneffect::kSchemaVersion)
        return fail(CommonEffectPayloadFailure::unsupportedSchema);
    const auto count = commoneffect::parameterCount(*kind);
    if (attributes.size() != count + 2)
        return fail(CommonEffectPayloadFailure::malformedPayload);

    commoneffect::Description description;
    description.schemaVersion = static_cast<std::uint32_t>(schema);
    description.kind = *kind;
    for (std::size_t index = 0; index < count; ++index)
        if (!number(commoneffect::parameterName(description.kind, index),
                    description.parameters[index]))
            return fail(CommonEffectPayloadFailure::malformedPayload);

    commoneffect::AdmissionFailure admissionFailure = commoneffect::AdmissionFailure::none;
    const auto admitted = commoneffect::admitCommonEffect(description, backend, admissionFailure);
    if (!admitted)
    {
        if (admissionFailure == commoneffect::AdmissionFailure::unsupportedSchema)
            return fail(CommonEffectPayloadFailure::unsupportedSchema);
        if (admissionFailure == commoneffect::AdmissionFailure::productionBackendUnavailable)
            return fail(CommonEffectPayloadFailure::productionBackendUnavailable);
        return fail(CommonEffectPayloadFailure::invalidParameter);
    }

    ExecutableCommonEffectPayload parsed;
    parsed.kind = admitted->description().kind;
    parsed.rendererEffectType = commoneffect::rendererEffectType(parsed.kind);
    parsed.parameterCount = count;
    for (std::size_t index = 0; index < count; ++index)
        parsed.parameters[index] = static_cast<float>(admitted->description().parameters[index]);
    out = parsed;
    return true;
}
} // namespace videowire
