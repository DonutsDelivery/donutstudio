#pragma once

#include "curated_isf_multipass.h"
#include "programmable_admission.h"
#include "render_snapshot.h"
#include "sha256.h"
#include "../../shared/ShaderCatalogParameters.h"
#include "../../shared/CuratedShaderTransitionContract.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace videowire
{

enum class FlatShaderRole { generator, filter };
enum class FlatShaderLanguage { glsl, isf };

struct FlatShaderBridgePayload
{
    int schemaVersion = 0;
    FlatShaderRole role = FlatShaderRole::generator;
    FlatShaderLanguage language = FlatShaderLanguage::glsl;
    std::string source;
    std::string sourceSha256;
    std::string catalogPackId;
    std::string catalogProgramId;
    std::string parameters;
    std::map<std::string, double> parameterValues;
    CuratedIsfPassResources passResources;
    std::string exactPayloadXml;
};

inline bool sameExactFlatShaderPayload (const FlatShaderBridgePayload& left,
                                        const FlatShaderBridgePayload& right) noexcept
{
    return left.exactPayloadXml == right.exactPayloadXml
        && left.schemaVersion == right.schemaVersion
        && left.role == right.role
        && left.language == right.language
        && left.source == right.source
        && left.sourceSha256 == right.sourceSha256
        && left.catalogPackId == right.catalogPackId
        && left.catalogProgramId == right.catalogProgramId
        && left.parameters == right.parameters
        && left.parameterValues == right.parameterValues
        && left.passResources == right.passResources;
}

enum class ShaderOperationKind { generator, filter, transition };

// One admitted GPU operation in graph execution order. Resource identities are
// graph node IDs, never renderer handles. A generator has no inputs, a filter
// has one input, and a transition has two inputs in start/end port order.
struct ShaderOperation
{
    ShaderOperationKind kind = ShaderOperationKind::generator;
    int nodeId = 0;
    std::array<int, 2> inputNodeIds {};
    std::size_t inputCount = 0;
    int outputNodeId = 0;
    FlatShaderBridgePayload payload;
    std::map<std::string, double> generatedParameters;
    std::optional<shadertransition::Payload> transitionPayload;
    std::optional<programmableruntime::Grant> customGrant;
};

struct ShaderOperationPlan
{
    static constexpr std::size_t maximumOperations = 4;
    static constexpr std::size_t maximumPassTargets = 16;
    std::vector<ShaderOperation> operations;
    std::size_t passTargetCount = 0;
    uint64_t revision = 0;
    std::string digest;
};

inline std::string shaderOperationPlanDigest(const ShaderOperationPlan& plan, uint64_t revision)
{
    std::string bytes;
    const auto add = [&bytes](std::string_view value)
    {
        bytes += std::to_string(value.size()); bytes += ':'; bytes.append(value.data(), value.size());
    };
    const auto number = [&add](uint64_t value) { add(std::to_string(value)); };
    const auto real = [&add](double value)
    {
        uint64_t bits = 0; static_assert(sizeof bits == sizeof value, "double width");
        std::memcpy(&bits, &value, sizeof bits); add(std::to_string(bits));
    };
    add("DonutStudio/ShaderOperationPlan/v1"); number(revision);
    number(plan.passTargetCount); number(plan.operations.size());
    for (const auto& operation : plan.operations)
    {
        number(static_cast<uint64_t>(operation.kind)); number(static_cast<uint32_t>(operation.nodeId));
        number(operation.inputCount);
        for (std::size_t i = 0; i < operation.inputCount; ++i) number(static_cast<uint32_t>(operation.inputNodeIds[i]));
        number(static_cast<uint32_t>(operation.outputNodeId));
        const auto& payload = operation.payload;
        number(payload.schemaVersion); number(static_cast<uint64_t>(payload.role));
        number(static_cast<uint64_t>(payload.language)); add(payload.source); add(payload.sourceSha256);
        add(payload.catalogPackId); add(payload.catalogProgramId); add(payload.parameters);
        add(payload.exactPayloadXml);
        number(payload.parameterValues.size());
        for (const auto& value : payload.parameterValues) { add(value.first); real(value.second); }
        number(payload.passResources.passes.size()); number(payload.passResources.namedTargets);
        number(payload.passResources.retainedImages);
        for (const auto& pass : payload.passResources.passes)
        { add(pass.target); number(pass.persistent); }
        number(operation.generatedParameters.size());
        for (const auto& value : operation.generatedParameters) { add(value.first); real(value.second); }
        number(operation.transitionPayload.has_value());
        if (operation.transitionPayload)
        {
            const auto& transition = *operation.transitionPayload;
            number(static_cast<uint64_t>(transition.direction));
            number(static_cast<uint64_t>(transition.easing)); real(transition.progress);
            add(transition.source); add(transition.sourceSha256); add(transition.catalogPackId);
            add(transition.catalogProgramId); add(transition.parameters);
        }
        number(operation.customGrant.has_value());
        if (operation.customGrant)
        {
            const auto& grant = *operation.customGrant;
            number(grant.version); number(static_cast<uint64_t>(grant.kind)); add(grant.fingerprint);
            add(grant.catalogPackId); add(grant.catalogProgramId); number(grant.sessionGeneration);
            number(grant.nonce); number(grant.issuedAtMs); number(grant.approved); number(grant.disk);
            number(grant.network); number(grant.verifiedBundledCurated); number(grant.cpuMs);
            number(grant.gpuMs); number(grant.memoryMiB); add(grant.mac);
        }
    }
    return videohelper::sha256Text(bytes);
}

using ImmutableShaderOperationPlan = std::shared_ptr<const ShaderOperationPlan>;

inline const std::map<std::string, double>& shaderOperationParameters(
    const ShaderOperation& operation,
    const std::map<int, std::map<std::string, double>>& evaluatedParameters) noexcept
{
    const auto found = evaluatedParameters.find(operation.nodeId);
    return found == evaluatedParameters.end() ? operation.generatedParameters : found->second;
}

template <typename Visitor>
bool visitOrderedShaderOperationsOnce(
    const ImmutableShaderOperationPlan& plan,
    const std::map<int, std::map<std::string, double>>& evaluatedParameters,
    Visitor&& visitor)
{
    if (plan == nullptr)
        return false;
    for (const auto& operation : plan->operations)
        if (!visitor(operation, shaderOperationParameters(operation, evaluatedParameters)))
            return false;
    return true;
}

struct FlatShaderRuntimeParameter
{
    std::string name;
    std::string type;
    std::size_t componentCount = 0;
};

inline bool validateFlatShaderRuntimeParameters(
    const shadercatalog::Entry& catalog,
    const std::vector<FlatShaderRuntimeParameter>& runtimeParameters,
    const std::map<std::string, double>& values,
    std::string& error)
{
    if (runtimeParameters.size() != catalog.parameterCount)
    {
        error = "curated FlatShaderBridge runtime parameter schema does not match catalog order";
        return false;
    }

    std::size_t expectedComponents = 0;
    for (std::size_t index = 0; index < catalog.parameterCount; ++index)
    {
        const auto& expected = shadercatalog::parameterAt(catalog, index);
        const auto& runtime = runtimeParameters[index];
        if (runtime.name != expected.name || runtime.type != expected.type
            || runtime.componentCount != expected.componentCount)
        {
            error = "curated FlatShaderBridge runtime parameter schema does not match catalog order";
            return false;
        }
        expectedComponents += expected.componentCount;
        for (std::size_t component = 0; component < expected.componentCount; ++component)
            if (values.find(shadercatalog::parameterComponentKey(expected, component)) == values.end())
            {
                error = "curated FlatShaderBridge runtime parameter values are incomplete";
                return false;
            }
    }
    if (values.size() != expectedComponents)
    {
        error = "curated FlatShaderBridge runtime parameter values contain an unknown component";
        return false;
    }
    error.clear();
    return true;
}

inline bool applyFlatShaderRuntimeParameters(
    const FlatShaderBridgePayload& payload,
    int nodeId,
    const std::map<std::string, double>& published,
    std::map<std::string, double>& values,
    std::string& error)
{
    values = payload.parameterValues;
    const auto* catalog = shadercatalog::find(payload.catalogPackId, payload.catalogProgramId);
    if (catalog == nullptr)
    {
        error = "curated FlatShaderBridge runtime catalog identity is unavailable";
        return false;
    }
    const auto alias = shadercatalog::runtimeNodeAlias(nodeId);
    if (alias.empty())
    {
        error = "curated FlatShaderBridge runtime node identity is invalid";
        return false;
    }
    const auto prefix = alias + "/";
    for (const auto& [id, value] : published)
    {
        if (id.rfind(prefix, 0) != 0) continue;
        const auto componentId = std::string_view(id).substr(prefix.size());
        if (!shadercatalog::hasParameterComponent(*catalog, componentId))
        {
            error = "curated FlatShaderBridge runtime parameter is unknown: " + id;
            return false;
        }
        if (!std::isfinite(value))
        {
            error = "curated FlatShaderBridge runtime parameter is not finite: " + id;
            return false;
        }
        values[std::string(componentId)] = value;
    }
    std::string validationError;
    const auto canonical = shadercatalog::parameterWire(*catalog, values, validationError);
    if (canonical.empty() && catalog->parameterCount != 0)
    {
        error = validationError;
        return false;
    }
    error.clear();
    return true;
}

inline bool applyFlatShaderBridgeRuntimeParameters(
    const FlatShaderBridgePayload& payload,
    int nodeId,
    bool custom,
    const std::map<std::string, double>& published,
    std::map<std::string, double>& values,
    std::string& error)
{
    if (!custom)
        return applyFlatShaderRuntimeParameters(payload, nodeId, published, values, error);

    values = payload.parameterValues;
    const auto alias = shadercatalog::runtimeNodeAlias(nodeId);
    if (alias.empty())
    {
        error = "custom FlatShaderBridge runtime node identity is invalid";
        return false;
    }
    const auto prefix = alias + "/";
    const auto targeted = std::find_if(published.begin(), published.end(),
        [&](const auto& entry) { return entry.first.rfind(prefix, 0) == 0; });
    if (targeted != published.end())
    {
        error = "custom FlatShaderBridge does not expose catalog runtime parameters: "
            + targeted->first;
        return false;
    }
    error.clear();
    return true;
}

namespace flatshaderbridge_detail
{
inline bool appendUtf8(uint32_t value, std::string& out)
{
    if (value == 0 || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) return false;
    if (value <= 0x7f) out.push_back(static_cast<char>(value));
    else if (value <= 0x7ff)
    {
        out.push_back(static_cast<char>(0xc0 | (value >> 6)));
        out.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    }
    else if (value <= 0xffff)
    {
        out.push_back(static_cast<char>(0xe0 | (value >> 12)));
        out.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    }
    else
    {
        out.push_back(static_cast<char>(0xf0 | (value >> 18)));
        out.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    }
    return true;
}

inline bool decodeEntities(std::string_view encoded, std::string& decoded)
{
    decoded.clear();
    for (size_t i = 0; i < encoded.size(); ++i)
    {
        if (encoded[i] != '&') { decoded.push_back(encoded[i]); continue; }
        const auto semi = encoded.find(';', i + 1);
        if (semi == std::string_view::npos) return false;
        const auto entity = encoded.substr(i + 1, semi - i - 1);
        if (entity == "amp") decoded.push_back('&');
        else if (entity == "lt") decoded.push_back('<');
        else if (entity == "gt") decoded.push_back('>');
        else if (entity == "quot") decoded.push_back('"');
        else if (entity == "apos") decoded.push_back('\'');
        else if (entity.size() >= 2 && entity.front() == '#')
        {
            const bool hex = entity.size() >= 3 && (entity[1] == 'x' || entity[1] == 'X');
            const size_t first = hex ? 2 : 1;
            if (first == entity.size()) return false;
            uint32_t value = 0;
            for (size_t n = first; n < entity.size(); ++n)
            {
                const char c = entity[n];
                int digit = c >= '0' && c <= '9' ? c - '0'
                    : hex && c >= 'a' && c <= 'f' ? c - 'a' + 10
                    : hex && c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
                if (digit < 0 || value > (0x10ffffu - static_cast<uint32_t>(digit)) / (hex ? 16u : 10u))
                    return false;
                value = value * (hex ? 16u : 10u) + static_cast<uint32_t>(digit);
            }
            if (!appendUtf8(value, decoded)) return false;
        }
        else return false;
        i = semi;
    }
    return true;
}

inline bool validUtf8(std::string_view text)
{
    for (size_t i = 0; i < text.size();)
    {
        const auto c = static_cast<uint8_t>(text[i]);
        if (c < 0x80) { if (c == 0) return false; ++i; continue; }
        int continuation = c >= 0xc2 && c <= 0xdf ? 1
            : c >= 0xe0 && c <= 0xef ? 2
            : c >= 0xf0 && c <= 0xf4 ? 3 : -1;
        if (continuation < 0 || i + static_cast<size_t>(continuation) >= text.size()) return false;
        const auto next = static_cast<uint8_t>(text[i + 1]);
        if ((c == 0xe0 && next < 0xa0) || (c == 0xed && next >= 0xa0)
            || (c == 0xf0 && next < 0x90) || (c == 0xf4 && next >= 0x90)) return false;
        for (int n = 1; n <= continuation; ++n)
            if ((static_cast<uint8_t>(text[i + static_cast<size_t>(n)]) & 0xc0) != 0x80) return false;
        i += static_cast<size_t>(continuation + 1);
    }
    return true;
}

inline bool isNameChar(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
        || (c >= '0' && c <= '9') || c == '_' || c == '-' || c == ':' || c == '.';
}
}

inline bool parseFlatShaderBridgePayload(std::string_view xml,
                                         FlatShaderBridgePayload& payload,
                                         std::string& error,
                                         bool allowCustom = false)
{
    payload = {};
    constexpr std::array<std::string_view, 9> allowed {{
        "schemaVersion", "role", "language", "source", "sourceSha256", "trust",
        "catalogPackId", "catalogProgramId", "parameters" }};
    std::map<std::string, std::string> attributes;
    size_t p = 0;
    const auto whitespace = [&]() { while (p < xml.size() && (xml[p] == ' ' || xml[p] == '\t' || xml[p] == '\r' || xml[p] == '\n')) ++p; };
    whitespace();
    constexpr std::string_view root = "<FlatShaderBridge";
    if (xml.substr(p, root.size()) != root) { error = "FlatShaderBridge payload requires the exact root element"; return false; }
    p += root.size();
    if (p >= xml.size() || !(xml[p] == ' ' || xml[p] == '\t' || xml[p] == '\r' || xml[p] == '\n'))
    { error = "FlatShaderBridge root name is malformed"; return false; }
    while (true)
    {
        whitespace();
        if (p + 2 <= xml.size() && xml.substr(p, 2) == "/>") { p += 2; break; }
        if (p >= xml.size() || xml[p] == '>') { error = "FlatShaderBridge must be one self-closing element with no content"; return false; }
        const size_t nameStart = p;
        while (p < xml.size() && flatshaderbridge_detail::isNameChar(xml[p])) ++p;
        if (p == nameStart) { error = "FlatShaderBridge attribute syntax is malformed"; return false; }
        const std::string name(xml.substr(nameStart, p - nameStart));
        if (std::find(allowed.begin(), allowed.end(), name) == allowed.end())
        { error = "FlatShaderBridge contains unknown attribute '" + name + "'"; return false; }
        if (attributes.find(name) != attributes.end())
        { error = "FlatShaderBridge contains duplicate attribute '" + name + "'"; return false; }
        whitespace();
        if (p >= xml.size() || xml[p++] != '=') { error = "FlatShaderBridge attribute syntax is malformed"; return false; }
        whitespace();
        if (p >= xml.size() || (xml[p] != '"' && xml[p] != '\''))
        { error = "FlatShaderBridge attributes must be quoted"; return false; }
        const char quote = xml[p++];
        const size_t valueStart = p;
        while (p < xml.size() && xml[p] != quote && xml[p] != '<') ++p;
        if (p >= xml.size() || xml[p] != quote) { error = "FlatShaderBridge attribute value is malformed"; return false; }
        std::string value;
        if (!flatshaderbridge_detail::decodeEntities(xml.substr(valueStart, p - valueStart), value)
            || !flatshaderbridge_detail::validUtf8(value))
        { error = "FlatShaderBridge attribute entity or UTF-8 encoding is malformed"; return false; }
        attributes.emplace(name, std::move(value));
        ++p;
    }
    whitespace();
    if (p != xml.size()) { error = "FlatShaderBridge must contain no nested or trailing content"; return false; }
    for (const char* required : { "schemaVersion", "role", "language", "source", "sourceSha256",
                                  "catalogPackId", "catalogProgramId" })
        if (attributes.find(required) == attributes.end())
        { error = std::string("FlatShaderBridge is missing required attribute '") + required + "'"; return false; }
    if (attributes["schemaVersion"] == "1") payload.schemaVersion = 1;
    else if (attributes["schemaVersion"] == "2") payload.schemaVersion = 2;
    else { error = "FlatShaderBridge schemaVersion must be exactly 1 or 2"; return false; }
    const bool trust = attributes.find("trust") != attributes.end();
    const bool parameters = attributes.find("parameters") != attributes.end();
    if (payload.schemaVersion == 1 && (!trust || parameters))
    { error = "FlatShaderBridge schemaVersion 1 requires trust and forbids parameters"; return false; }
    if (payload.schemaVersion == 2 && (trust || !parameters))
    { error = "FlatShaderBridge schemaVersion 2 requires parameters and forbids trust or grant attributes"; return false; }
    if (attributes["role"] == "generator") payload.role = FlatShaderRole::generator;
    else if (attributes["role"] == "filter") payload.role = FlatShaderRole::filter;
    else { error = "FlatShaderBridge role must be generator or filter"; return false; }
    if (attributes["language"] == "glsl") payload.language = FlatShaderLanguage::glsl;
    else if (attributes["language"] == "isf") payload.language = FlatShaderLanguage::isf;
    else { error = "FlatShaderBridge language must be glsl or isf"; return false; }
    payload.source = std::move(attributes["source"]);
    payload.sourceSha256 = std::move(attributes["sourceSha256"]);
    payload.catalogPackId = std::move(attributes["catalogPackId"]);
    payload.catalogProgramId = std::move(attributes["catalogProgramId"]);
    payload.parameters = parameters ? attributes["parameters"] : std::string {};
    if (payload.source.empty() || payload.source.size() > 1024u * 1024u)
    { error = "FlatShaderBridge source is empty or exceeds 1 MiB"; return false; }
    if (payload.sourceSha256.size() != 64
        || !std::all_of(payload.sourceSha256.begin(), payload.sourceSha256.end(), [](char c)
            { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }))
    { error = "FlatShaderBridge sourceSha256 must be 64 lowercase hexadecimal characters"; return false; }
    const auto freshHash = videohelper::sha256Text(payload.source);
    if (freshHash != payload.sourceSha256)
    { error = "FlatShaderBridge decoded source hash mismatch"; return false; }
    if (payload.schemaVersion == 1 && attributes["trust"] != "verified-bundled-curated")
    { error = "FlatShaderBridge schemaVersion 1 trust label is not the curated compatibility value"; return false; }
    const bool custom = payload.catalogPackId.empty() && payload.catalogProgramId.empty();
    if (custom)
    {
        if (!allowCustom || payload.schemaVersion != 2
            || payload.role != FlatShaderRole::generator || !payload.parameters.empty())
        { error = "curated FlatShaderBridge requires non-empty catalog IDs"; return false; }
        payload.exactPayloadXml.assign(xml.data(), xml.size());
        error.clear();
        return true;
    }
    if (payload.catalogPackId.empty() || payload.catalogProgramId.empty())
    { error = "FlatShaderBridge catalog identity must be complete"; return false; }
    const auto* entry = shadercatalog::find(payload.catalogPackId, payload.catalogProgramId);
    if (entry == nullptr) { error = "curated FlatShaderBridge catalog identity is unknown"; return false; }
    const std::string_view expectedKind = payload.language == FlatShaderLanguage::isf ? "isf" : "shader";
    if (entry->kind != expectedKind)
    { error = "FlatShaderBridge language and catalog kind mismatch"; return false; }
    if (entry->sourceSha256 != payload.sourceSha256 || entry->sourceSha256 != freshHash)
    { error = "curated FlatShaderBridge catalog source digest mismatch"; return false; }
    if (payload.schemaVersion == 2
        && !shadercatalog::validateParameterWire(*entry, payload.parameters,
                                                 payload.parameterValues, error))
        return false;
    if (payload.language == FlatShaderLanguage::isf
        && !admitCuratedIsfPassResources(payload.source, payload.passResources, error))
    {
        payload = {};
        return false;
    }
    payload.exactPayloadXml.assign(xml.data(), xml.size());
    error.clear();
    return true;
}

inline bool admitFlatShaderBridgeOperation(const CompiledVisualOperation& operation,
                                            FlatShaderBridgePayload& payload,
                                            std::string& error)
{
    const bool generator = operation.kind == "visual.shader.generator";
    const bool filter = operation.kind == "visual.shader.filter";
    if (operation.kind == "visual.shader.custom")
    { error = "custom shader operation requires exact-payload grant admission"; return false; }
    if (!generator && !filter) { error = "operation is not a FlatShaderBridge operation kind"; return false; }
    if (operation.backendCapability != "native-gpu")
    { error = "FlatShaderBridge operation requires backendCapability native-gpu"; return false; }
    if (!parseFlatShaderBridgePayload(operation.payloadXml, payload, error)) return false;
    const auto expected = generator ? FlatShaderRole::generator : FlatShaderRole::filter;
    if (payload.role != expected)
    { error = "FlatShaderBridge payload role does not match operation kind"; return false; }
    return true;
}

inline bool admitCustomFlatShaderBridgeOperation(const CompiledVisualOperation& operation,
                                                  FlatShaderBridgePayload& payload,
                                                  std::string& error)
{
    if (operation.kind != "visual.shader.custom"
        || operation.backendCapability != "native-gpu")
    { error = "custom FlatShaderBridge operation requires native-gpu"; return false; }
    if (!parseFlatShaderBridgePayload(operation.payloadXml, payload, error, true)) return false;
    if (!payload.catalogPackId.empty() || !payload.catalogProgramId.empty()
        || payload.role != FlatShaderRole::generator)
    { error = "custom FlatShaderBridge payload must be an uncurated generator"; return false; }
    if (operation.runtimeGrantJson.empty())
    { error = "custom shader runtime grant is missing"; return false; }
    try
    {
        const auto grantJson = nlohmann::json::parse(operation.runtimeGrantJson);
        const auto kind = payload.language == FlatShaderLanguage::isf
            ? programmableruntime::PayloadKind::isf
            : programmableruntime::PayloadKind::shader;
        programmableruntime::Grant admitted;
        if (!programmableadmission::admit(grantJson, kind, payload.source, admitted, error))
            return false;
        if (admitted.verifiedBundledCurated
            || !admitted.catalogPackId.empty() || !admitted.catalogProgramId.empty())
        { error = "custom shader runtime grant must not claim curated catalog authority"; return false; }
    }
    catch (const std::exception&)
    {
        error = "custom shader runtime grant is malformed";
        return false;
    }
    return true;
}

inline bool admitCatalogIsfFilterOperation(const CompiledVisualOperation& operation,
                                           FlatShaderBridgePayload& payload,
                                           std::string& error)
{
    if (!admitFlatShaderBridgeOperation(operation, payload, error)) return false;
    if (payload.role != FlatShaderRole::filter || payload.schemaVersion != 2
        || payload.language != FlatShaderLanguage::isf)
    {
        error = "catalog ISF filter requires the exact schemaVersion 2 filter/isf payload";
        payload = {};
        return false;
    }
    return true;
}

} // namespace videowire
