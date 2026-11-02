#pragma once

// Undoes a device setup that did not finish. Each step that creates something adds
// how to release it, and unless commit() is reached the releases run newest first,
// on failure or on an exception. A retry then starts from nothing.

#include <functional>
#include <utility>
#include <vector>

namespace videohelper::pathtrace::device
{

class SetupUnwind
{
public:
    SetupUnwind() = default;
    SetupUnwind (const SetupUnwind&) = delete;
    SetupUnwind& operator= (const SetupUnwind&) = delete;

    ~SetupUnwind()
    {
        for (auto release = releases_.rbegin(); release != releases_.rend(); ++release)
            (*release)();
    }

    void add (std::function<void()> release) { releases_.push_back (std::move (release)); }
    void commit() noexcept { releases_.clear(); }

private:
    std::vector<std::function<void()>> releases_;
};

} // namespace videohelper::pathtrace::device
