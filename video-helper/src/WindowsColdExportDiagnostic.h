#pragma once

#include <string_view>

namespace windowscoldexport
{
inline const char* openCallerLabel (std::string_view category) noexcept
{
    if (category == "thumbnail-cache") return "thumbnail-cache";
    if (category == "reusable-media") return "reusable-media";
    if (category == "one-shot-probe") return "one-shot-probe";
    return "other";
}

// Return a fixed label even when the private pipe receives an unknown method.
inline const char* requestLabel (std::string_view method) noexcept
{
    constexpr const char* labels[] {
        "capture_list_sources", "open", "capture_preview_open", "capture_record_start",
        "record_open", "viewport_open", "viewport_open_shared", "viewport_open_shm",
        "viewport_info", "viewport_set_canvas", "viewport_set_canvas_background",
        "viewport_set_clkfps", "pathtrace_device", "pathtrace_warmup",
        "composite_frame_probe", "recipe_preview", "render_cache_build", "proxy_generate",
        "export"
    };
    for (const auto* label : labels)
        if (method == label)
            return label;
    return "other";
}
}
