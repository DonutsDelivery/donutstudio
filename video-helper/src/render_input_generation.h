#pragma once

// The render-input generation (Plan 3, export and encoders: an edit cancels
// obsolete render work). Every RPC that changes what a frame renders bumps it
// after the edit lands. A Path Trace preview pass captures it with the inputs it
// renders and stops, inside its sample loop, once it moves: the pass is
// rendering a frame nobody will see, and the next frame restarts with the
// edit. Exports render a snapshot and do not watch it.
//
// Each clip has its own generation, so an edit restarts only the previews whose
// inputs it changed: a parameter, effect or LUT edit on clip 7 leaves a Path
// Trace preview of clip 3 converging. Edits that reach every clip (the
// timeline, the canvas, score, modulation, scripts) bump them all. A draw made
// outside any clip scope watches the shared generation, which every edit moves.

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>

namespace videohelper::renderinput
{

namespace detail
{
struct Registry
{
    std::mutex lock;
    // Never erased: a pass may hold a counter's address while its clip goes away.
    std::map<int, std::unique_ptr<std::atomic<std::uint64_t>>> clips;
};

inline Registry& registry()
{
    static Registry instance;
    return instance;
}

inline int& activeClipSlot() noexcept
{
    thread_local int clip = 0;
    return clip;
}
} // namespace detail

// The shared generation: draws outside a clip scope watch it, and every edit
// moves it.
inline std::atomic<std::uint64_t>& counter() noexcept
{
    static std::atomic<std::uint64_t> generation { 0 };
    return generation;
}

// Clip ids are positive. Zero or less names the shared generation.
inline std::atomic<std::uint64_t>& counter (int clip)
{
    if (clip <= 0)
        return counter();
    auto& registry = detail::registry();
    const std::lock_guard<std::mutex> guard (registry.lock);
    auto& slot = registry.clips[clip];
    if (slot == nullptr)
        slot = std::make_unique<std::atomic<std::uint64_t>> (0);
    return *slot;
}

inline std::uint64_t current() noexcept
{
    return counter().load (std::memory_order_acquire);
}

inline std::uint64_t current (int clip)
{
    return counter (clip).load (std::memory_order_acquire);
}

// An edit that reaches every clip.
inline std::uint64_t bump() noexcept
{
    auto& registry = detail::registry();
    {
        const std::lock_guard<std::mutex> guard (registry.lock);
        for (auto& entry : registry.clips)
            entry.second->fetch_add (1, std::memory_order_acq_rel);
    }
    return counter().fetch_add (1, std::memory_order_acq_rel) + 1;
}

// An edit to one clip. The shared generation moves too: a draw outside any clip
// scope cannot tell whose inputs it renders.
inline std::uint64_t bump (int clip)
{
    if (clip <= 0)
        return bump();
    counter().fetch_add (1, std::memory_order_acq_rel);
    return counter (clip).fetch_add (1, std::memory_order_acq_rel) + 1;
}

// The clip whose layer this thread is rendering, set by the layer's preparation
// so the renderer below can watch that clip's generation.
inline int activeClip() noexcept
{
    return detail::activeClipSlot();
}

class ClipScope
{
public:
    explicit ClipScope (int clip) noexcept : previous_ (detail::activeClipSlot())
    {
        detail::activeClipSlot() = clip;
    }
    ~ClipScope() { detail::activeClipSlot() = previous_; }
    ClipScope (const ClipScope&) = delete;
    ClipScope& operator= (const ClipScope&) = delete;

private:
    int previous_;
};

// The RPCs whose edits change a rendered frame's inputs: the graph, its
// parameters, effects, LUTs, text, the timeline and its plans, the canvas,
// post, score, modulation, control plans and scripts. Transport, window and
// inspection calls leave the inputs alone.
inline bool changesFrameInputs (std::string_view method) noexcept
{
    constexpr std::string_view methods[] = {
        "graph_set_param", "graph_set_effects", "graph_set_lut", "text_set_image",
        "viewport_set_timeline", "viewport_set_full_canvas_scene", "viewport_set_canvas",
        "viewport_set_canvas_background", "viewport_set_post", "viewport_set_score",
        "viewport_set_beat_timeline", "viewport_set_mod_matrix", "viewport_set_control_plan",
        "viewport_set_script", "viewport_set_audio_mix",
    };
    for (const auto name : methods)
        if (method == name)
            return true;
    return false;
}

// Which generations an edit moves.
struct EditScope
{
    enum class Kind { None, Clip, All };
    Kind kind = Kind::None;
    int clip = 0;
};

// The clip a graph parameter id names: "clip<id>/<node>/<param>". View
// parameters change no frame. Text and anything else reach every clip.
inline std::optional<int> clipOfParamId (std::string_view paramId) noexcept
{
    if (paramId.substr (0, 4) != "clip")
        return std::nullopt;
    int clip = 0;
    std::size_t i = 4;
    for (; i < paramId.size() && paramId[i] >= '0' && paramId[i] <= '9'; ++i)
    {
        if (clip > 100000000)
            return std::nullopt;
        clip = clip * 10 + (paramId[i] - '0');
    }
    if (i == 4 || i >= paramId.size() || paramId[i] != '/' || clip <= 0)
        return std::nullopt;
    return clip;
}

// paramId is the graph_set_param id; clipId the clipId field of the effect and
// LUT calls. Either is empty when the call does not carry it.
inline EditScope editScope (std::string_view method, std::string_view paramId, std::optional<int> clipId) noexcept
{
    if (! changesFrameInputs (method))
        return {};
    if (method == "graph_set_param")
    {
        if (paramId.substr (0, 5) == "view/")
            return {};
        if (const auto clip = clipOfParamId (paramId))
            return { EditScope::Kind::Clip, *clip };
        return { EditScope::Kind::All, 0 };
    }
    if ((method == "graph_set_effects" || method == "graph_set_lut") && clipId.has_value() && *clipId > 0)
        return { EditScope::Kind::Clip, *clipId };
    return { EditScope::Kind::All, 0 };
}

inline void bump (const EditScope& scope)
{
    if (scope.kind == EditScope::Kind::Clip)
        bump (scope.clip);
    else if (scope.kind == EditScope::Kind::All)
        bump();
}

} // namespace videohelper::renderinput
