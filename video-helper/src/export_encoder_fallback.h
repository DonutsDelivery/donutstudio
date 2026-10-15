#pragma once

#include <string>
#include <utility>

namespace videohelper
{
template <typename Open>
auto openEncoderWithAutomaticSoftwareFallback (
    std::string& encoderName, bool automatic, const std::string& softwareName,
    Open&& open) -> decltype(open(encoderName))
{
    auto encoder = open(encoderName);
    if (! encoder && automatic && encoderName != softwareName)
    {
        encoderName = softwareName;
        encoder = open(encoderName);
    }
    return encoder;
}

template <typename Open, typename AcquireGpuOwners>
auto openEncoderBeforeGpuOwners (
    std::string& encoderName, bool automatic, const std::string& softwareName,
    Open&& open, AcquireGpuOwners&& acquireGpuOwners, std::string& gpuOwnerError)
    -> decltype(open(encoderName))
{
    auto encoder = openEncoderWithAutomaticSoftwareFallback(
        encoderName, automatic, softwareName, std::forward<Open>(open));
    gpuOwnerError.clear();
    if (encoder)
        gpuOwnerError = acquireGpuOwners();
    return encoder;
}
} // namespace videohelper
