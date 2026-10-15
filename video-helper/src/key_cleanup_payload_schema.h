#pragma once

#include "../../shared/KeyCleanupContract.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <string>
#include <string_view>
#include <unordered_map>

namespace videowire
{
enum class KeyCleanupPayloadFailure : std::uint8_t
{
    none = 0,
    malformedPayload,
    unsupportedSchema,
    invalidFrameContract,
    invalidParameter,
    productionBackendUnavailable
};

struct ExecutableKeyCleanupPayload
{
    float keyR = 0.0f, keyG = 1.0f, keyB = 0.0f;
    float tolerance = 0.18f, softness = 0.10f, despill = 0.5f;
    float choke = 0.0f, feather = 0.0f;
    float edgeRed = 1.0f, edgeGreen = 1.0f, edgeBlue = 1.0f, edgeAmount = 0.0f;
    bool matteView = false;
};

inline bool parseKeyCleanupAttributes(
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
    if (!take("<KeyCleanup") || (cursor < xml.size()
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

inline bool parseExecutableKeyCleanupPayload(
    std::string_view xml, ExecutableKeyCleanupPayload& out,
    KeyCleanupPayloadFailure* payloadFailure = nullptr,
    keycleanup::BackendSupport backend = keycleanup::currentProductionBackendSupport())
{
    const auto fail = [&](KeyCleanupPayloadFailure value)
    {
        if (payloadFailure != nullptr) *payloadFailure = value;
        return false;
    };
    if (payloadFailure != nullptr) *payloadFailure = KeyCleanupPayloadFailure::none;

    std::unordered_map<std::string, std::string> attributes;
    if (!parseKeyCleanupAttributes(xml, attributes) || attributes.size() != 14)
        return fail(KeyCleanupPayloadFailure::malformedPayload);
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

    keycleanup::Description description;
    double schema = 0.0;
    double view = 0.0;
    if (!number("schemaVersion", schema) || schema != keycleanup::kSchemaVersion
        || !number("keyR", description.keyRed)
        || !number("keyG", description.keyGreen)
        || !number("keyB", description.keyBlue)
        || !number("tolerance", description.tolerance)
        || !number("softness", description.softness)
        || !number("despill", description.despill)
        || !number("choke", description.choke)
        || !number("feather", description.featherPixels)
        || !number("edgeRed", description.edgeRed)
        || !number("edgeGreen", description.edgeGreen)
        || !number("edgeBlue", description.edgeBlue)
        || !number("edgeAmount", description.edgeAmount)
        || !number("view", view) || (view != 0.0 && view != 1.0))
        return fail(KeyCleanupPayloadFailure::malformedPayload);
    description.matteView = view == 1.0;

    keycleanup::AdmissionFailure admissionFailure = keycleanup::AdmissionFailure::none;
    const auto admitted = keycleanup::admit(description, backend, admissionFailure);
    if (!admitted)
    {
        if (admissionFailure == keycleanup::AdmissionFailure::unsupportedSchema)
            return fail(KeyCleanupPayloadFailure::unsupportedSchema);
        if (admissionFailure == keycleanup::AdmissionFailure::inexactFrameDescriptor)
            return fail(KeyCleanupPayloadFailure::invalidFrameContract);
        if (admissionFailure == keycleanup::AdmissionFailure::productionBackendUnavailable)
            return fail(KeyCleanupPayloadFailure::productionBackendUnavailable);
        return fail(KeyCleanupPayloadFailure::invalidParameter);
    }

    const auto& value = admitted->description();
    ExecutableKeyCleanupPayload parsed;
    parsed.keyR = static_cast<float>(value.keyRed);
    parsed.keyG = static_cast<float>(value.keyGreen);
    parsed.keyB = static_cast<float>(value.keyBlue);
    parsed.tolerance = static_cast<float>(value.tolerance);
    parsed.softness = static_cast<float>(value.softness);
    parsed.despill = static_cast<float>(value.despill);
    parsed.choke = static_cast<float>(value.choke);
    parsed.feather = static_cast<float>(value.featherPixels);
    parsed.edgeRed = static_cast<float>(value.edgeRed);
    parsed.edgeGreen = static_cast<float>(value.edgeGreen);
    parsed.edgeBlue = static_cast<float>(value.edgeBlue);
    parsed.edgeAmount = static_cast<float>(value.edgeAmount);
    parsed.matteView = value.matteView;
    out = parsed;
    return true;
}
} // namespace videowire
