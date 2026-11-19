#pragma once

#include "PortableFloatingPointChars.h"
#include <cmath>
#include <optional>
#include <string>
#include <string_view>

namespace videosourcecolor
{
// Authored interpretation, independent of decoded transfer/matrix/range facts
// and export mastering/content-light metadata. Absence never implies 1000 nits.
struct Policy
{
    std::optional<double> hlgNominalDisplayPeakNits;
};
inline bool valid(const Policy& policy) noexcept
{
    return !policy.hlgNominalDisplayPeakNits
        || (std::isfinite(*policy.hlgNominalDisplayPeakNits)
            && *policy.hlgNominalDisplayPeakNits >= 400.0
            && *policy.hlgNominalDisplayPeakNits <= 10000.0);
}
inline std::string encode(const Policy& policy)
{
    if (!valid(policy)) return {};
    if (!policy.hlgNominalDisplayPeakNits) return {};
    char number[32] {};
    const auto result = portablefloat::toChars(number,number+sizeof(number),
        *policy.hlgNominalDisplayPeakNits,std::chars_format::general,17);
    return result.ec == std::errc{} ? "video-source-color-v1|hlgPeak="
        + std::string(number,result.ptr) : std::string{};
}
inline bool decode(std::string_view text, Policy& policy)
{
    policy = {};
    if (text.empty()) return true; // Older projects have no authored policy.
    constexpr std::string_view prefix = "video-source-color-v1|hlgPeak=";
    if (text.substr(0,prefix.size()) != prefix) return false;
    const auto original = text;
    text.remove_prefix(prefix.size());
    double peak = 0;
    const auto result = portablefloat::fromChars(text.data(),text.data()+text.size(),peak);
    if (result.ec != std::errc{} || result.ptr != text.data()+text.size()) return false;
    Policy candidate {peak};
    if (!valid(candidate) || encode(candidate) != original) return false;
    policy = candidate; return true;
}
} // namespace videosourcecolor
