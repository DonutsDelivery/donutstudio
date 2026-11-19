#pragma once

#include "ShaderCatalogParameters.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <string>
#include <string_view>

namespace shadertransition
{
inline constexpr int schemaVersion = 2;
inline constexpr size_t maxSourceBytes = 1024 * 1024;
inline constexpr size_t maxIdentityBytes = 128;
inline constexpr const char* operationKind = "visual.shader.transition";

struct CuratedIdentity
{
    std::string_view packId;
    std::string_view programId;
    std::string_view sourceSha256;
};

// This is an execution allowlist, not imported display data. Catalog names and
// labels remain editor metadata and cannot expand this admitted program set.
inline constexpr CuratedIdentity curatedIdentities[] = {
    { "vidvox-isf", "bow-tie-horizontal", "612ca07fb9fb5cf2a0b4e590af9e1ef3d4cc2ccadd45e876ffda007b364a3cef" },
    { "vidvox-isf", "bow-tie-vertical", "599f9863d1a886e2e45086e50d899671e5337ccd9fbae36f1ec825608798c8da" },
    { "vidvox-isf", "crosszoom", "6619841325561a8e8e7783ec66a754866b31952727dee5f0f457e0acb6feefec" },
    { "vidvox-isf", "cube", "5b5267feeeb8a27a80808ea1c9fa509d036b6765115bc62421bd2548b81d143a" },
    { "vidvox-isf", "directional-wipe", "1fcdfc7ab93a4734cde49708dbacfb1594ef285923b5be105d281d7e1ee15a1d" },
    { "vidvox-isf", "doorway", "d5a111308cf39336208548d5f4f01f3daf285170e53704bbfab0b007b6384772" },
    { "vidvox-isf", "fade", "d9e68ff50d9551ee5af2385c9f013f7166103fef5c2dfacdc5e9b3e678431466" },
    { "vidvox-isf", "rotate-scale-fade", "4faf5d59fe818f81e1072b2719f8dca18971f6a82c3f16e339f7bae52e558725" },
    { "vidvox-isf", "wipe-down", "4bbec586200a90d3bb8dc1b721a66fce7e6d23fbc35df38fd7e4ddb807fbeef8" },
    { "vidvox-isf", "wipe-left", "81b5f04cb77af1605e1d28d44285e9bd66a6d26391e621e6e0b7c739da7c53bf" },
    { "vidvox-isf", "wipe-right", "701a7cd6d564a07ce75bf530218fe8f8517793cc9307bc0023d4b20ec4047bfd" },
    { "vidvox-isf", "wipe-up", "37925ebe4367760e424cb8388bfafb07c79c3ba030b49d2aafdb6cc9d4cef998" }
};

enum class Direction { invalid, forward, reverse };
enum class Easing { invalid, linear, easeIn, easeOut, easeInOut };

inline double evaluateProgress(double progress, Direction direction, Easing easing)
{
    double value = std::clamp(progress, 0.0, 1.0);
    if (direction == Direction::reverse) value = 1.0 - value;
    switch (easing)
    {
        case Easing::linear:    return value;
        case Easing::easeIn:    return value * value * value;
        case Easing::easeOut:
        {
            const double inverse = 1.0 - value;
            return 1.0 - inverse * inverse * inverse;
        }
        case Easing::easeInOut:
            if (value < 0.5) return 4.0 * value * value * value;
            return 1.0 - std::pow(-2.0 * value + 2.0, 3.0) / 2.0;
        case Easing::invalid:   return value;
    }
    return value;
}

inline std::string_view toWire(Direction value)
{
    if (value == Direction::forward) return "forward";
    if (value == Direction::reverse) return "reverse";
    return "invalid";
}

inline Direction directionFromWire(std::string_view value)
{
    if (value == "forward") return Direction::forward;
    if (value == "reverse") return Direction::reverse;
    return Direction::invalid;
}

inline std::string_view toWire(Easing value)
{
    if (value == Easing::linear) return "linear";
    if (value == Easing::easeIn) return "ease-in";
    if (value == Easing::easeOut) return "ease-out";
    if (value == Easing::easeInOut) return "ease-in-out";
    return "invalid";
}

inline Easing easingFromWire(std::string_view value)
{
    if (value == "linear") return Easing::linear;
    if (value == "ease-in") return Easing::easeIn;
    if (value == "ease-out") return Easing::easeOut;
    if (value == "ease-in-out") return Easing::easeInOut;
    return Easing::invalid;
}

struct Payload
{
    std::string catalogPackId;
    std::string catalogProgramId;
    std::string sourceSha256;
    std::string source;
    std::string parameters;
    double progress = 0.0;
    Direction direction = Direction::forward;
    Easing easing = Easing::linear;
};

inline const CuratedIdentity* findCuratedIdentity(std::string_view packId,
                                                   std::string_view programId)
{
    for (const auto& entry : curatedIdentities)
        if (entry.packId == packId && entry.programId == programId)
            return &entry;
    return nullptr;
}

inline bool isIdentityToken(std::string_view value)
{
    if (value.empty() || value.size() > maxIdentityBytes) return false;
    for (const unsigned char c : value)
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
              || c == '-' || c == '_' || c == '.' || c == '@'))
            return false;
    return true;
}

inline bool isLowerSha256(std::string_view value)
{
    if (value.size() != 64) return false;
    for (const char c : value)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}

inline bool validate(const Payload& payload, std::string& error)
{
    error.clear();
    if (!isIdentityToken(payload.catalogPackId) || !isIdentityToken(payload.catalogProgramId))
    {
        error = "shader transition catalog identities must be bounded lowercase tokens";
        return false;
    }
    if (!isLowerSha256(payload.sourceSha256))
    {
        error = "shader transition sourceSha256 must be 64 lowercase hexadecimal characters";
        return false;
    }
    if (payload.source.empty() || payload.source.size() > maxSourceBytes)
    {
        error = "shader transition source is empty or exceeds the 1 MiB limit";
        return false;
    }
    if (!std::isfinite(payload.progress) || payload.progress < 0.0 || payload.progress > 1.0)
    {
        error = "shader transition progress must be finite and in [0,1]";
        return false;
    }
    if (payload.direction == Direction::invalid)
    {
        error = "shader transition direction must be forward or reverse";
        return false;
    }
    if (payload.easing == Easing::invalid)
    {
        error = "shader transition easing must be linear, ease-in, ease-out, or ease-in-out";
        return false;
    }
    const auto* curated = findCuratedIdentity(payload.catalogPackId, payload.catalogProgramId);
    const auto* catalog = shadercatalog::find(payload.catalogPackId, payload.catalogProgramId);
    if (curated == nullptr || catalog == nullptr || catalog->kind != "isf"
        || curated->sourceSha256 != payload.sourceSha256
        || catalog->sourceSha256 != payload.sourceSha256)
    {
        error = "shader transition pack, program, and source digest are not an exact curated catalog entry";
        return false;
    }
    std::map<std::string, double> parameterValues;
    if (!shadercatalog::validateParameterWire(*catalog, payload.parameters,
                                               parameterValues, error))
        return false;
    const auto progress = parameterValues.find("progress");
    if (progress == parameterValues.end() || progress->second != payload.progress)
    {
        error = "shader transition progress must exactly match the curated parameter list";
        return false;
    }
    return true;
}

inline std::string base64Encode(std::string_view input)
{
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve((input.size() + 2) / 3 * 4);
    for (size_t i = 0; i < input.size(); i += 3)
    {
        const unsigned a = static_cast<unsigned char>(input[i]);
        const unsigned b = i + 1 < input.size() ? static_cast<unsigned char>(input[i + 1]) : 0;
        const unsigned c = i + 2 < input.size() ? static_cast<unsigned char>(input[i + 2]) : 0;
        const unsigned bits = (a << 16) | (b << 8) | c;
        result.push_back(alphabet[(bits >> 18) & 63]);
        result.push_back(alphabet[(bits >> 12) & 63]);
        result.push_back(i + 1 < input.size() ? alphabet[(bits >> 6) & 63] : '=');
        result.push_back(i + 2 < input.size() ? alphabet[bits & 63] : '=');
    }
    return result;
}

inline int base64Value(char c)
{
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

inline bool base64Decode(std::string_view input, std::string& output)
{
    output.clear();
    if (input.empty() || input.size() % 4 != 0
        || input.size() > ((maxSourceBytes + 2) / 3) * 4)
        return false;
    output.reserve(input.size() / 4 * 3);
    for (size_t i = 0; i < input.size(); i += 4)
    {
        const bool final = i + 4 == input.size();
        const int a = base64Value(input[i]);
        const int b = base64Value(input[i + 1]);
        const int c = input[i + 2] == '=' ? -2 : base64Value(input[i + 2]);
        const int d = input[i + 3] == '=' ? -2 : base64Value(input[i + 3]);
        if (a < 0 || b < 0 || c == -1 || d == -1 || (!final && (c == -2 || d == -2))
            || (c == -2 && d != -2))
            return false;
        const unsigned bits = (static_cast<unsigned>(a) << 18)
                            | (static_cast<unsigned>(b) << 12)
                            | (static_cast<unsigned>(c < 0 ? 0 : c) << 6)
                            | static_cast<unsigned>(d < 0 ? 0 : d);
        output.push_back(static_cast<char>((bits >> 16) & 0xff));
        if (c >= 0) output.push_back(static_cast<char>((bits >> 8) & 0xff));
        if (d >= 0) output.push_back(static_cast<char>(bits & 0xff));
    }
    return output.size() <= maxSourceBytes && base64Encode(output) == input;
}

inline std::string canonicalProgress(double value)
{
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(std::numeric_limits<double>::max_digits10)
           << (value == 0.0 ? 0.0 : value);
    return stream.str();
}

inline std::string parametersForProgress(const shadercatalog::Entry& entry, double progress,
                                         std::string& error)
{
    if (!std::isfinite(progress) || progress < 0.0 || progress > 1.0)
    {
        error = "shader transition progress must be finite and in [0,1]";
        return {};
    }
    if (entry.parameterCount == 0)
    {
        error = "curated shader transition requires a catalog progress parameter";
        return {};
    }
    const auto& parameter = shadercatalog::parameterAt(entry, 0);
    if (parameter.name != "progress" || parameter.type != "float"
        || parameter.componentCount != 1)
    {
        error = "curated shader transition catalog progress parameter is not exact";
        return {};
    }
    auto parameters = shadercatalog::defaultParameterWire(entry);
    const auto end = parameters.find(';');
    const auto equals = parameters.find('=');
    if (equals == std::string::npos || (end != std::string::npos && equals > end))
    {
        error = "curated shader transition catalog parameter wire is malformed";
        return {};
    }
    parameters.replace(equals + 1,
                       (end == std::string::npos ? parameters.size() : end) - equals - 1,
                       canonicalProgress(progress));
    error.clear();
    return parameters;
}

inline std::string serialize(const Payload& payload, std::string& error)
{
    if (!validate(payload, error)) return {};
    return "<CuratedShaderTransition schemaVersion=\"2\" catalogPackId=\""
        + payload.catalogPackId + "\" catalogProgramId=\"" + payload.catalogProgramId
        + "\" sourceSha256=\"" + payload.sourceSha256 + "\" parameters=\""
        + payload.parameters + "\" progress=\""
        + canonicalProgress(payload.progress) + "\" direction=\""
        + std::string(toWire(payload.direction)) + "\" easing=\""
        + std::string(toWire(payload.easing)) + "\" sourceBase64=\""
        + base64Encode(payload.source) + "\"/>";
}

inline bool consume(std::string_view& input, std::string_view literal)
{
    if (input.substr(0, literal.size()) != literal) return false;
    input.remove_prefix(literal.size());
    return true;
}

inline bool consumeAttribute(std::string_view& input, std::string_view name, std::string& value)
{
    const std::string prefix = " " + std::string(name) + "=\"";
    if (!consume(input, prefix)) return false;
    const auto end = input.find('"');
    if (end == std::string_view::npos) return false;
    value.assign(input.substr(0, end));
    input.remove_prefix(end + 1);
    return true;
}

inline bool parse(std::string_view encoded, Payload& payload, std::string& error)
{
    payload = {};
    if (encoded.size() > ((maxSourceBytes + 2) / 3) * 4 + 1024)
    {
        error = "shader transition payload exceeds the bounded schema size";
        return false;
    }
    auto input = encoded;
    std::string version, progress, direction, easing, sourceBase64;
    if (!consume(input, "<CuratedShaderTransition")
        || !consumeAttribute(input, "schemaVersion", version)
        || !consumeAttribute(input, "catalogPackId", payload.catalogPackId)
        || !consumeAttribute(input, "catalogProgramId", payload.catalogProgramId)
        || !consumeAttribute(input, "sourceSha256", payload.sourceSha256)
        || !consumeAttribute(input, "parameters", payload.parameters)
        || !consumeAttribute(input, "progress", progress)
        || !consumeAttribute(input, "direction", direction)
        || !consumeAttribute(input, "easing", easing)
        || !consumeAttribute(input, "sourceBase64", sourceBase64)
        || !consume(input, "/>" ) || !input.empty())
    {
        error = "shader transition payload is not the exact canonical schema";
        return false;
    }
    if (version != "2")
    {
        error = "shader transition schemaVersion must be 2";
        return false;
    }
    std::istringstream progressStream(progress);
    progressStream.imbue(std::locale::classic());
    progressStream >> payload.progress;
    if (!progressStream || progressStream.peek() != std::char_traits<char>::eof())
    {
        error = "shader transition progress is malformed";
        return false;
    }
    payload.direction = directionFromWire(direction);
    payload.easing = easingFromWire(easing);
    if (!base64Decode(sourceBase64, payload.source))
    {
        error = "shader transition sourceBase64 is malformed or exceeds the source limit";
        return false;
    }
    if (!validate(payload, error)) return false;
    std::string canonicalError;
    if (serialize(payload, canonicalError) != encoded)
    {
        error = "shader transition payload is not canonically serialized";
        return false;
    }
    return true;
}
} // namespace shadertransition
