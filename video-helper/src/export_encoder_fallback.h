#pragma once

#include <string>
#include <utility>

namespace videohelper
{
template <typename Open>
auto openEncoderWithAutomaticSoftwareFallback (
    std::string& encoderName, bool automatic, const std::string& softwareName,
    Open&& open, std::string* fallbackReason = nullptr) -> decltype(open(encoderName))
{
    auto encoder = open(encoderName);
    if (! encoder && automatic && encoderName != softwareName)
    {
        // Auto falls back to software, by name: the export record says so.
        if (fallbackReason != nullptr)
            *fallbackReason = encoderName + " did not open; " + softwareName + " encodes instead";
        encoderName = softwareName;
        encoder = open(encoderName);
    }
    return encoder;
}

template <typename Open, typename AcquireGpuOwners>
auto openEncoderBeforeGpuOwners (
    std::string& encoderName, bool automatic, const std::string& softwareName,
    Open&& open, AcquireGpuOwners&& acquireGpuOwners, std::string& gpuOwnerError,
    std::string* fallbackReason = nullptr)
    -> decltype(open(encoderName))
{
    auto encoder = openEncoderWithAutomaticSoftwareFallback(
        encoderName, automatic, softwareName, std::forward<Open>(open), fallbackReason);
    gpuOwnerError.clear();
    if (encoder)
        gpuOwnerError = acquireGpuOwners();
    return encoder;
}
} // namespace videohelper
