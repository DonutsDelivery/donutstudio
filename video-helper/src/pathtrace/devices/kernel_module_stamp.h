#pragma once

// Tells whether the driver's disk cache holds this kernel without making a CUDA context. The
// OptiX cache keeps its keys to itself, so a build that succeeds leaves a stamp file named by
// our own key in the cache's directory. The key covers the kernel text, the compile options and
// the driver and GPU that compiled it. A cleared cache takes the stamp with it. A cache that
// evicted the entry on size keeps the stamp, and then the build runs as it always did.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace videohelper::pathtrace::kernelstamp
{

class KeyHasher
{
public:
    void add (const void* bytes, std::size_t size) noexcept
    {
        const auto* data = static_cast<const unsigned char*> (bytes);
        for (std::size_t i = 0; i < size; ++i)
        {
            low_ = (low_ ^ data[i]) * 0x100000001b3ULL;
            high_ = (high_ ^ data[i]) * 0x100000001b3ULL + (high_ >> 29);
        }
    }
    void add (const std::string& text) noexcept
    {
        add (text.data(), text.size());
        add ("\0", 1);
    }
    template <typename Number>
    void addNumber (const char* label, Number value) { add (std::string (label) + "=" + std::to_string (value)); }

    std::string hex() const
    {
        char text[33];
        std::snprintf (text, sizeof (text), "%016llx%016llx", static_cast<unsigned long long> (high_),
                       static_cast<unsigned long long> (low_));
        return text;
    }

private:
    std::uint64_t low_ = 0xcbf29ce484222325ULL;
    std::uint64_t high_ = 0x84222325cbf29ce4ULL;
};

inline std::string stampName (const std::string& key) { return "donutstudio-kernel-" + key + ".stamp"; }

inline bool stampPresent (const std::filesystem::path& directory, const std::string& key)
{
    std::error_code ignored;
    return ! directory.empty() && std::filesystem::is_regular_file (directory / stampName (key), ignored);
}

inline bool stampPresentIn (const std::vector<std::filesystem::path>& directories, const std::string& key)
{
    for (const auto& directory : directories)
        if (stampPresent (directory, key))
            return true;
    return false;
}

// Leaves the stamp in directory. False when the directory can't be written, which only means
// the next launch warms up again.
inline bool writeStamp (const std::filesystem::path& directory, const std::string& key)
{
    if (directory.empty())
        return false;
    std::error_code ignored;
    if (! std::filesystem::is_directory (directory, ignored))
        return false;
    std::FILE* file = std::fopen ((directory / stampName (key)).string().c_str(), "wb");
    if (file == nullptr)
        return false;
    std::fputs ("DonutStudio compiled this Path Trace kernel once; its driver cache entry sits beside this file.\n", file);
    return std::fclose (file) == 0;
}

// Where the driver keeps its cache when the environment names nothing, and the places a
// previous run could have left a stamp. ownerName is the account name on Linux.
inline std::vector<std::filesystem::path> cacheDirectories (const char* environmentPath, const char* home,
                                                            const char* localAppData, const std::string& ownerName)
{
    std::vector<std::filesystem::path> directories;
    if (environmentPath != nullptr && environmentPath[0] != '\0')
    {
        directories.emplace_back (environmentPath);
        return directories;
    }
    if (localAppData != nullptr && localAppData[0] != '\0')
        directories.push_back (std::filesystem::path (localAppData) / "NVIDIA" / "OptixCache");
    if (! ownerName.empty())
        directories.push_back (std::filesystem::path ("/var/tmp") / ("OptixCache_" + ownerName));
    if (home != nullptr && home[0] != '\0')
        directories.push_back (std::filesystem::path (home) / ".cache" / "nvidia" / "OptixCache");
    return directories;
}

} // namespace videohelper::pathtrace::kernelstamp
