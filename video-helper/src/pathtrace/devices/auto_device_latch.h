// A device setup that failed under Auto, held for the whole helper process.
// Path Trace engines die with their scene owners, and a timeline edit retires
// every owner, so a failure held by an engine would recompile the device's
// programs on each edit. The latch outlives them: the next try comes when the
// user moves a Render node off Path Trace.
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>

namespace videohelper::pathtrace::device
{
class AutoDeviceLatch
{
public:
    // The user-facing reason. Null while no failure is held.
    std::shared_ptr<const std::string> failure() const
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        return failure_;
    }

    void fail (std::shared_ptr<const std::string> reason)
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        failure_ = std::move (reason);
    }

    // Each draw says which engine its Render node chose. A node that was on Path
    // Trace and now is not releases the failure. A node that was never on Path
    // Trace does not, so a Raster node beside a Path Trace node starts no retry.
    void noteEngine (int clipId, std::uint64_t renderNode, bool pathTrace)
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        const std::pair<int, std::uint64_t> node { clipId, renderNode };
        if (pathTrace)
            onPathTrace_.insert (node);
        else if (onPathTrace_.erase (node) != 0)
            failure_.reset();
    }

private:
    mutable std::mutex mutex_;
    std::shared_ptr<const std::string> failure_;
    std::set<std::pair<int, std::uint64_t>> onPathTrace_;
};

inline AutoDeviceLatch& autoDeviceLatch()
{
    static AutoDeviceLatch latch;
    return latch;
}
} // namespace videohelper::pathtrace::device
