#pragma once

#include <cmath>
#include <cstddef>
#include <string_view>

namespace builtingeneratorshader
{
inline constexpr const char* catalogPackId = "arbit-builtin-generators-v1";

namespace detail
{
inline bool take(std::string_view& remaining, std::string_view literal) noexcept
{
    if (remaining.substr(0, literal.size()) != literal) return false;
    remaining.remove_prefix(literal.size());
    return true;
}

// GeneratorSpec prints six decimal places. Accept only bounded numeric literals,
// never expressions, exponents, comments, non-finite values, or shader statements.
inline bool scalar(std::string_view& remaining, bool signedValue, double& value) noexcept
{
    const bool negative = signedValue && take(remaining, "-");
    if (remaining.size() < 8 || (remaining[0] != '0' && remaining[0] != '1')
        || remaining[1] != '.') return false;
    int millionths = (remaining[0] - '0') * 1000000;
    int fraction = 0;
    for (std::size_t i = 2; i < 8; ++i)
    {
        if (remaining[i] < '0' || remaining[i] > '9') return false;
        fraction = fraction * 10 + remaining[i] - '0';
    }
    millionths += fraction;
    if (millionths > 1000000) return false;
    value = (negative ? -millionths : millionths) / 1000000.0;
    remaining.remove_prefix(8);
    return true;
}

inline bool colour(std::string_view& remaining) noexcept
{
    for (int channel = 0; channel < 4; ++channel)
    {
        double value = 0.0;
        if ((channel != 0 && !take(remaining, ",")) || !scalar(remaining, false, value)
            || std::abs(value - std::round(value * 255.0) / 255.0) > 0.0000006)
            return false;
    }
    return true;
}
} // namespace detail

// These are the exact, bounded application templates emitted by GeneratorSpec.
// Both the producer and helper verify this contract before using the existing
// catalog grant path; the session MAC still binds all source bytes and metadata.
inline bool validSource(std::string_view programId, std::string_view source) noexcept
{
    if (source.empty() || source.size() > 1024) return false;
    if (programId == "image")
        return source == "/*{ \"ISFVSN\": \"2.0\", \"INPUTS\": [ { \"NAME\": \"uGenImage\", \"TYPE\": \"image\" } ] }*/\n"
                         "void main(){ gl_FragColor = IMG_NORM_PIXEL(uGenImage, isf_FragNormCoord); }";
    if (programId == "solid")
        return detail::take(source, "void mainImage(out vec4 o, in vec2 fc){ o = vec4(")
            && detail::colour(source) && detail::take(source, "); }") && source.empty();
    if (programId != "gradient") return false;
    double x = 0.0, y = 0.0;
    return detail::take(source, "void mainImage(out vec4 o, in vec2 fc){\n"
                               "  vec2 uv = fc / uResolution;\n"
                               "  float t = clamp(dot(uv - vec2(0.5), vec2(")
        && detail::scalar(source, true, x) && detail::take(source, ",")
        && detail::scalar(source, true, y) && std::abs(x * x + y * y - 1.0) <= 0.000002
        && detail::take(source, ")) + 0.5, 0.0, 1.0);\n  o = mix(vec4(")
        && detail::colour(source) && detail::take(source, "), vec4(")
        && detail::colour(source) && detail::take(source, "), t);\n}") && source.empty();
}
} // namespace builtingeneratorshader
