#pragma once

#include <cmath>
#include <string>

namespace hdrvideo
{
struct Output
{
    std::string primaries;
    std::string transfer;
    std::string alpha;
    double peakNits = 0.0;
};

inline std::string validate(const Output& output, const std::string& hdrImageProfile)
{
    if (hdrImageProfile != "off")
        return "HDR movie output cannot use an HDR image sequence profile";
    if (output.primaries != "rec2020-d65")
        return "HDR movie output requires Rec.2020 D65 primaries";
    if (output.transfer != "pq" && output.transfer != "hlg")
        return "HDR movie output requires PQ or HLG transfer";
    if (output.alpha != "opaque")
        return "HDR movie output requires opaque alpha";
    if (!std::isfinite(output.peakNits) || output.peakNits < 400.0
        || output.peakNits > 10000.0)
        return "HDR movie output requires peakNits in [400, 10000]";
    return {};
}
} // namespace hdrvideo
