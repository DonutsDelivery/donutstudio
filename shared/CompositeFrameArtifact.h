#pragma once

#include "SecureEntropy.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace compositeartifact
{
constexpr int width = 3840, height = 2160;
constexpr size_t maxBytes = size_t(width) * size_t(height) * 4u;
constexpr size_t chunkBytes = 1024u * 1024u;
constexpr size_t maxHandles = 2;
constexpr int lifetimeMs = 60000;
static_assert(maxBytes / 4u / size_t(height) == size_t(width));

inline bool validHandle(const std::string& value)
{
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c)
        { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

// JSON has no integer type, and the command server decodes requests into
// doubles before it re-encodes them (shortest form, so 1048576 travels as
// 1.048576e+06). Accept any integral number inside the artifact; Var is
// juce::var or anything with the same isInt/isInt64/isDouble accessors.
template <typename Var>
inline bool offsetFromVar(const Var& value, size_t& offset)
{
    if (value.isInt() || value.isInt64())
    {
        const auto number = static_cast<long long>(value);
        if (number < 0 || static_cast<unsigned long long>(number) >= maxBytes) return false;
        offset = static_cast<size_t>(number);
        return true;
    }
    if (!value.isDouble()) return false;
    const auto number = static_cast<double>(value);
    if (!std::isfinite(number) || number < 0.0 || number >= static_cast<double>(maxBytes)
        || std::trunc(number) != number)
        return false;
    offset = static_cast<size_t>(number);
    return true;
}

// Handles are process-private, never paths. Every read consumes exactly one
// chunk. Unknown, released and completed handles all fail closed.
class Store
{
public:
    using Clock = std::function<std::chrono::steady_clock::time_point()>;
    using Validate = std::function<bool()>;
    explicit Store(Clock clock = [] { return std::chrono::steady_clock::now(); }) : clock_(std::move(clock)) {}
    std::string create(const std::string& owner, std::vector<uint8_t> bytes,
                       Validate validate, std::string& error)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        error.clear();
        pruneLocked();
        if (owner.empty() || bytes.empty() || bytes.size() > maxBytes || !validate || !validate())
        { error = "invalid or stale composite artifact"; return {}; }
        if (entries_.size() >= maxHandles || std::any_of(entries_.begin(), entries_.end(),
                [&owner](const auto& item) { return item.second.owner == owner; }))
        { error = "composite artifact quota exceeded; release the previous artifact"; return {}; }
        std::array<uint8_t, 32> token {};
        if (!programmableruntime::secureentropy::osFill(token.data(), token.size()))
        { error = "composite artifact entropy unavailable"; return {}; }
        static constexpr char hex[] = "0123456789abcdef";
        std::string handle;
        for (auto byte : token) { handle += hex[byte >> 4]; handle += hex[byte & 15]; }
        if (entries_.count(handle)) { error = "composite artifact handle collision"; return {}; }
        entries_.emplace(handle, Entry { owner, std::move(bytes), std::move(validate),
            clock_() + std::chrono::milliseconds(lifetimeMs), 0, nextGeneration_++ });
        return handle;
    }
    bool read(const std::string& owner, const std::string& handle, size_t offset,
              std::vector<uint8_t>& bytes, bool& done, std::string& error)
    {
        error.clear();
        Validate validate;
        uint64_t generation = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pruneLocked();
            auto it = entries_.find(handle);
            if (!validHandle(handle) || it == entries_.end() || it->second.owner != owner)
            { error = "unknown composite artifact"; return false; }
            if (offset != it->second.offset || offset >= it->second.bytes.size())
            { entries_.erase(it); error = "stale or out-of-sequence composite artifact"; return false; }
            validate = it->second.validate;
            generation = it->second.generation;
        }
        // Validation may synchronously visit the message thread, where project
        // invalidation and server shutdown clear this store.
        const bool valid = validate();
        std::lock_guard<std::mutex> lock(mutex_);
        pruneLocked();
        auto it = entries_.find(handle);
        if (it == entries_.end() || it->second.owner != owner || it->second.generation != generation)
        { error = "unknown composite artifact"; return false; }
        auto& entry = it->second;
        if (!valid || offset != entry.offset || offset >= entry.bytes.size())
        { entries_.erase(it); error = "stale or out-of-sequence composite artifact"; return false; }
        const auto count = std::min(chunkBytes, entry.bytes.size() - offset);
        bytes.assign(entry.bytes.begin() + offset, entry.bytes.begin() + offset + count);
        entry.offset += count;
        done = entry.offset == entry.bytes.size();
        if (done) entries_.erase(it);
        return true;
    }
    bool release(const std::string& owner, const std::string& handle, std::string& error)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        error.clear();
        pruneLocked();
        auto it = entries_.find(handle);
        if (!validHandle(handle) || it == entries_.end() || it->second.owner != owner)
        { error = "unknown composite artifact"; return false; }
        entries_.erase(it);
        return true;
    }
    void clear() { std::lock_guard<std::mutex> lock(mutex_); entries_.clear(); }
    void prune() { std::lock_guard<std::mutex> lock(mutex_); pruneLocked(); }
private:
    struct Entry
    {
        std::string owner;
        std::vector<uint8_t> bytes;
        Validate validate;
        std::chrono::steady_clock::time_point expires;
        size_t offset;
        uint64_t generation;
    };
    void pruneLocked()
    {
        const auto now = clock_();
        for (auto it = entries_.begin(); it != entries_.end();)
            if (now >= it->second.expires) it = entries_.erase(it); else ++it;
    }
    Clock clock_;
    std::mutex mutex_;
    std::map<std::string, Entry> entries_;
    uint64_t nextGeneration_ = 0;
};
}
