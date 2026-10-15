#pragma once

#include "shader_dialect.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace videowire
{

inline constexpr std::size_t kMaximumCuratedIsfPasses = 8;
inline constexpr std::size_t kMaximumCuratedIsfRetainedImages = 8;
inline constexpr std::uint64_t kMaximumCuratedIsfRetainedBytes = 512ull * 1024ull * 1024ull;
inline constexpr std::uint64_t kCuratedIsfBytesPerPixel = 8;

struct CuratedIsfPass
{
    std::string target;
    bool persistent = false;

    bool operator== (const CuratedIsfPass& other) const noexcept
    {
        return target == other.target && persistent == other.persistent;
    }
};

struct CuratedIsfPassResources
{
    std::vector<CuratedIsfPass> passes;
    std::size_t namedTargets = 0;
    std::size_t retainedImages = 0;

    bool multipass() const noexcept
    {
        return passes.size() > 1 || retainedImages > namedTargets;
    }

    bool operator== (const CuratedIsfPassResources& other) const noexcept
    {
        return passes == other.passes && namedTargets == other.namedTargets
            && retainedImages == other.retainedImages;
    }
};

inline bool admitCuratedIsfPassResources (const std::string& exactSource,
                                           CuratedIsfPassResources& resources,
                                           std::string& error)
{
    resources = {};
    const auto header = arbitshader::parseIsfHeader (exactSource);
    if (! header.valid)
    {
        error = "curated ISF multipass admission requires a valid exact ISF header: "
            + header.parseError;
        return false;
    }
    if (header.passes.size() > kMaximumCuratedIsfPasses)
    {
        error = "curated ISF pass count exceeds the bounded graph execution limit";
        return false;
    }

    resources.passes.reserve (header.passes.size());
    std::map<std::string, bool> targets;
    for (const auto& pass : header.passes)
    {
        resources.passes.push_back ({ pass.target, pass.persistent });
        if (pass.target.empty())
        {
            if (pass.persistent)
            {
                error = "curated ISF output passes cannot retain an unnamed persistent resource";
                resources = {};
                return false;
            }
            continue;
        }
        auto [target, inserted] = targets.emplace (pass.target, pass.persistent);
        if (! inserted && target->second != pass.persistent)
        {
            error = "curated ISF target persistence must be identical on every write";
            resources = {};
            return false;
        }
    }

    resources.namedTargets = targets.size();
    for (const auto& target : targets)
        resources.retainedImages += target.second ? 2u : 1u;
    if (resources.retainedImages > kMaximumCuratedIsfRetainedImages)
    {
        error = "curated ISF retained pass images exceed the bounded graph execution limit";
        resources = {};
        return false;
    }
    error.clear();
    return true;
}

inline bool admitCuratedIsfPassExtent (const CuratedIsfPassResources& resources,
                                       int width, int height, std::string& error)
{
    if (width <= 0 || height <= 0)
    {
        error = "curated ISF pass extent must be positive";
        return false;
    }
    const auto pixels = static_cast<std::uint64_t> (width)
        * static_cast<std::uint64_t> (height);
    if (resources.retainedImages != 0
        && pixels > kMaximumCuratedIsfRetainedBytes
            / kCuratedIsfBytesPerPixel / resources.retainedImages)
    {
        error = "curated ISF retained pass bytes exceed the bounded graph execution limit";
        return false;
    }
    error.clear();
    return true;
}

} // namespace videowire
