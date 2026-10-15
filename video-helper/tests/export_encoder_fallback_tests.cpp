#include "export_encoder_fallback.h"

#include <stdexcept>
#include <string>
#include <vector>

void require (bool condition)
{
    if (! condition)
        throw std::runtime_error("export encoder fallback invariant failed");
}

int main()
{
    std::vector<std::string> events;
    std::string selected = "h264_nvenc";
    std::string gpuOwnerError;
    int softwareEncoder = 1;
    auto* encoder = videohelper::openEncoderBeforeGpuOwners(
        selected, true, "libx264",
        [&] (const std::string& name) -> int*
        {
            events.push_back("open:" + name);
            return name == "libx264" ? &softwareEncoder : nullptr;
        },
        [&] () -> std::string
        {
            events.push_back("acquire:rife");
            events.push_back("acquire:compositor");
            return {};
        }, gpuOwnerError);
    require(encoder == &softwareEncoder);
    require(selected == "libx264");
    require(gpuOwnerError.empty());
    require((events == std::vector<std::string> {
        "open:h264_nvenc", "open:libx264", "acquire:rife", "acquire:compositor" }));

    events.clear();
    selected = "h264_nvenc";
    encoder = videohelper::openEncoderBeforeGpuOwners(
        selected, false, "libx264",
        [&] (const std::string& name) -> int*
        {
            events.push_back("open:" + name);
            return nullptr;
        },
        [&] () -> std::string
        {
            events.push_back("acquire:compositor");
            return {};
        }, gpuOwnerError);
    require(encoder == nullptr);
    require(selected == "h264_nvenc");
    require(gpuOwnerError.empty());
    require((events == std::vector<std::string> { "open:h264_nvenc" }));

    events.clear();
    selected = "libx264";
    encoder = videohelper::openEncoderBeforeGpuOwners(
        selected, false, "libx264",
        [&] (const std::string& name) -> int*
        {
            events.push_back("open:" + name);
            return &softwareEncoder;
        },
        [&] () -> std::string
        {
            events.push_back("acquire:compositor");
            return "GPU owner failed";
        }, gpuOwnerError);
    require(encoder == &softwareEncoder);
    require(gpuOwnerError == "GPU owner failed");
    require((events == std::vector<std::string> { "open:libx264", "acquire:compositor" }));
    return 0;
}
