#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

// Keeps what an expensive read of a long text token produced, so the same bytes are not read again.
//
// A hit needs equal bytes. There is no hash, since a collision would hand back the wrong value, and
// comparing is far cheaper than the read: memcmp over 263 MiB takes tens of milliseconds where the
// read takes tens of seconds. Changed bytes can only miss. A token holds no reference to anything
// outside itself, so a helper restart or a new project cannot make an entry wrong.
//
// Eviction only forgets. An evicted token is read again the next time it appears, and nothing is
// refused at any size or count. A memo given bytesOf also forgets its oldest values once theirs
// add up past keptBytes, but always keeps the newest, however large.
template <typename Value>
class ExactTokenMemo
{
public:
    explicit ExactTokenMemo (std::size_t kept, std::size_t keptBytes = 0,
                             std::size_t (*bytesOf) (const Value&) = nullptr)
        : kept_ (kept), keptBytes_ (keptBytes), bytesOf_ (bytesOf) {}

    std::shared_ptr<const Value> find (std::string_view token)
    {
        // Compare outside the lock: another thread's read should not wait on a long memcmp.
        for (const auto& entry : snapshot())
            if (entry.token->size() == token.size()
                && std::memcmp (entry.token->data(), token.data(), token.size()) == 0)
            {
                touch (entry.value.get());
                return entry.value;
            }
        return nullptr;
    }

    // The exact token a kept value was read from, so writing it again needs no encode.
    std::shared_ptr<const std::string> tokenOf (const Value* value)
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        for (const auto& entry : entries_)
            if (entry.value.get() == value)
                return entry.token;
        return nullptr;
    }

    void store (std::string_view token, std::shared_ptr<const Value> value)
    {
        // The copy happens once per distinct token, outside the lock.
        auto kept = std::make_shared<const std::string> (token);
        const std::lock_guard<std::mutex> lock (mutex_);
        for (const auto& entry : entries_)
            if (*entry.token == *kept)
                return;
        entries_.insert (entries_.begin(), Entry { std::move (kept), std::move (value) });
        if (entries_.size() > kept_)
            entries_.pop_back();
        if (bytesOf_ == nullptr) return;
        std::size_t bytes = 0;
        for (std::size_t index = 0; index < entries_.size(); ++index)
        {
            bytes += bytesOf_ (*entries_[index].value);
            if (index > 0 && bytes > keptBytes_)
            {
                entries_.erase (entries_.begin() + static_cast<std::ptrdiff_t> (index), entries_.end());
                return;
            }
        }
    }

private:
    struct Entry
    {
        std::shared_ptr<const std::string> token;
        std::shared_ptr<const Value> value;
    };

    std::vector<Entry> snapshot()
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        return entries_;
    }

    void touch (const Value* value)
    {
        const std::lock_guard<std::mutex> lock (mutex_);
        for (std::size_t index = 0; index < entries_.size(); ++index)
            if (entries_[index].value.get() == value)
            {
                auto entry = std::move (entries_[index]);
                entries_.erase (entries_.begin() + static_cast<std::ptrdiff_t> (index));
                entries_.insert (entries_.begin(), std::move (entry));
                return;
            }
    }

    std::size_t kept_;
    std::size_t keptBytes_;
    std::size_t (*bytesOf_) (const Value&);
    std::mutex mutex_;
    std::vector<Entry> entries_;
};
