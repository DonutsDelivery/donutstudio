#pragma once

#include <charconv>
#include <string>
#include <string_view>

namespace materialframesource
{
// References the existing project clip identity, never a path or decoder handle.
inline std::string encode(int clipId)
{ return clipId > 0 ? "material-frame-source-v1|clip=" + std::to_string(clipId) : std::string{}; }
inline bool decode(std::string_view text, int& clipId)
{
    constexpr std::string_view prefix = "material-frame-source-v1|clip=";
    if (text.substr(0,prefix.size()) != prefix) return false;
    text.remove_prefix(prefix.size());
    const auto result = std::from_chars(text.data(),text.data()+text.size(),clipId);
    return result.ec == std::errc{} && result.ptr == text.data()+text.size()
        && clipId > 0 && encode(clipId) == std::string(prefix) + std::string(text);
}
}
