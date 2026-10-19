#pragma once

#include <algorithm>
#include <cctype>
#include <iterator>
#include <string>
#include <string_view>

// Names the clip behind an export layer failure.
//
// The plugin isolates a clip when an export, probe or viewport poll error names it, so a
// clip whose plan or content cannot render stops blocking the rest of the project. A
// failure that comes from the device, the encoder or the helper process says nothing
// about the clip that happened to be under construction. Isolating on those would refuse
// a healthy clip for as long as the machine is busy. attributeLayerError() therefore
// prefixes "clip N: " only for errors about the clip's own plan, source or content.
//
// The test is a denylist of device and context phrases, not of single words. Words such
// as "OpenGL", "readback" or "encoder" also appear in errors about a clip's own shader or
// material ("physical diffraction OpenGL shader compile failed"), and those must name the
// clip. An unlisted message is attributed to its clip, and a device error the list misses
// only wrongly isolates a clip until the next retry.

namespace videoexport
{

inline bool isEnvironmentError (std::string_view error)
{
    std::string lower (error);
    std::transform (lower.begin(), lower.end(), lower.begin(),
                    [] (unsigned char c) { return static_cast<char> (std::tolower (c)); });
    static constexpr std::string_view markers[] = {
        "opengl context", "gl context", "context lost", "device lost", "out of memory",
        "fbo incomplete", "framebuffer incomplete", "is busy", "helper process",
        "helper exited", "native source worker", "export encoder", "video encoder",
    };
    return std::any_of (std::begin (markers), std::end (markers), [&] (std::string_view marker)
    {
        return lower.find (marker) != std::string::npos;
    });
}

// Returns the error with "clip N: " in front when it is about that clip. Cancellation,
// an error that already names a clip, and device or environment errors come back as is.
inline std::string attributeLayerError (int clipId, const std::string& error)
{
    if (error.empty() || clipId <= 0 || error == "cancelled"
        || error.rfind ("clip ", 0) == 0 || isEnvironmentError (error))
        return error;
    return "clip " + std::to_string (clipId) + ": " + error;
}

} // namespace videoexport
