// viewport_rpc_policy.h: which handle() calls move the docked viewport's content
// revision, and what the repeat memo does with each call. The render loop produces
// only while the revision moves (viewport_publish_gate.h), so this table decides
// whether a state RPC reaches the screen. It has no JSON or GL dependency, so the
// helper tests drive it directly.
//
// Three classes:
//   exempt  Read-only polls and recorder sinks. They never move the revision and
//           never touch the memo.
//   setter  An idempotent state push over plain values: numbers, flags, and ids.
//           It moves the revision unless it repeats the previous setter exactly,
//           since a repeat changes nothing. A plugin that re-sends an unchanged
//           setter wakes nothing.
//   other   Any other content RPC. It always moves the revision and clears the
//           memo. An RPC that names a file or a shared-memory region belongs here,
//           because the data behind an identical call can change.
//
// Three setters move the revision inside Viewport, and only on a real change: the
// canvas pair and viewport_set_visible. The guard still records them in the memo,
// so a change breaks the repeat chain. Returning early would leave an older setter
// in the memo across the change. An identical repeat of that setter would then skip
// its bump, and the guard cannot know whether that setter's output depends on the
// canvas.
//
// A failed call always moves the revision and clears the memo. The guard cannot tell
// whether a failing handler applied part of its state.
//
// The memo and the guard that applies decide() are here too, as templates over the
// params type. handle() uses them with json, and the tests use int, so the tests
// drive the whole sequence, including a throwing handler.
#pragma once

#include <exception>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <utility>

namespace arbitviewport::rpc
{

inline bool isExempt (const std::string& method)
{
    static const std::set<std::string> exempt = {
        "ping", "version", "gpu_backend_info", "get_av_offset", "viewport_info",
        "viewport_describe_inspection", "graph_describe", "scope_data",
        "export_progress", "proxy_progress", "render_cache_progress",
        "stabilize_progress", "capture_list_sources", "capture_preview_latest",
        "record_push_frame", "thumbnails", "composite_frame_artifact_read" };
    return exempt.count (method) != 0;
}

inline bool isSetter (const std::string& method)
{
    static const std::set<std::string> setters = {
        "viewport_set_clkfps", "viewport_set_target_fps", "viewport_set_interp_max_dim",
        "viewport_set_bounds", "viewport_set_fullscreen", "viewport_set_canvas",
        "viewport_set_view", "viewport_set_canvas_frame_enabled",
        "viewport_set_canvas_background", "viewport_set_post",
        "viewport_set_inspection", "viewport_set_inspection_presentation",
        "viewport_set_score", "viewport_set_beat_timeline",
        "viewport_set_block_c_lifecycle", "viewport_set_mod_matrix",
        "viewport_set_control_plan", "viewport_set_visible", "graph_set_param",
        "graph_set_effects", "text_remove", "scope_enable" };
    return setters.count (method) != 0;
}

// The setters that move the revision inside Viewport, and only on a real change.
inline bool bumpsItself (const std::string& method)
{
    return method == "viewport_set_canvas" || method == "viewport_set_canvas_background"
        || method == "viewport_set_visible";
}

// What the guard does when a handler returns.
struct Action
{
    bool bump = false;      // move the content revision
    bool remember = false;  // record this call as the repeat memo
    bool forget = false;    // clear the repeat memo
};

// `failed` is true when the handler reported an error or threw. `repeats` is true
// when the memo holds this method with identical params.
inline Action decide (const std::string& method, bool failed, bool repeats)
{
    if (isExempt (method))
        return {};
    if (failed || ! isSetter (method))
        return { true, false, true };
    if (repeats)
        return {};
    return { ! bumpsItself (method), true, false };
}

// The last setter call and its params. An identical repeat of it changes nothing.
template <typename Params>
struct RepeatMemo
{
    std::mutex mutex;
    std::string method;
    Params params {};
    bool valid = false;
};

// Runs when a handle() call returns. `error` is the handler's output, and it is final
// by then. A throwing handler counts as failed, the same as an error string. The guard
// holds references to `params` and `error`, so both must outlive it.
template <typename Params>
class RevisionGuard
{
public:
    // `bump` moves the content revision when decide() says so.
    RevisionGuard (RepeatMemo<Params>& memo, std::string method, const Params& params,
                   const std::string& error, std::function<void()> bump)
        : memo_ (memo), method_ (std::move (method)), params_ (params), error_ (error),
          bump_ (std::move (bump)) {}

    ~RevisionGuard()
    {
        if (isExempt (method_))
            return;
        const bool failed = ! error_.empty() || std::uncaught_exceptions() > 0;
        std::lock_guard<std::mutex> lock (memo_.mutex);
        const bool repeats = memo_.valid && memo_.method == method_ && memo_.params == params_;
        const Action action = decide (method_, failed, repeats);
        // A canvas or visibility call records the memo too, even though Viewport
        // bumps for it. That breaks the repeat chain around the change.
        if (action.remember)
        {
            memo_.method = method_;
            memo_.params = params_;
            memo_.valid = true;
        }
        if (action.forget)
            memo_.valid = false;
        if (action.bump)
            bump_();
    }

    RevisionGuard (const RevisionGuard&) = delete;
    RevisionGuard& operator= (const RevisionGuard&) = delete;

private:
    RepeatMemo<Params>& memo_;
    std::string method_;
    const Params& params_;
    const std::string& error_;
    std::function<void()> bump_;
};

} // namespace arbitviewport::rpc
