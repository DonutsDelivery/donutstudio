#pragma once

#include <cstddef>
#include <cstring>
#include <new>
#include <vector>

namespace videohelper
{
// Splits the request stream into newline-terminated lines of any length. A line
// that can't be stored is a real allocation failure: the rest of it is discarded
// so the next request still starts on a line boundary, and the helper keeps running.
template <typename Buffer = std::vector<char>>
class RequestLineAssembler
{
public:
    // onLine (const char*, size_t) gets each non-empty line. onAllocationFailure (size_t)
    // gets the full length of a line that couldn't be stored.
    template <typename OnLine, typename OnAllocationFailure>
    void feed (const char* data, size_t size, OnLine&& onLine, OnAllocationFailure&& onAllocationFailure)
    {
        while (size > 0)
        {
            const auto* newline = static_cast<const char*> (std::memchr (data, '\n', size));
            const size_t run = newline != nullptr ? static_cast<size_t> (newline - data) : size;
            store (data, run);
            if (newline == nullptr)
                return;
            finish (onLine, onAllocationFailure);
            data += run + 1;
            size -= run + 1;
        }
    }

    // A final line without a newline, when the stream closes.
    template <typename OnLine, typename OnAllocationFailure>
    void close (OnLine&& onLine, OnAllocationFailure&& onAllocationFailure)
    {
        finish (onLine, onAllocationFailure);
    }

private:
    void store (const char* data, size_t size)
    {
        if (! allocationFailed_)
        {
            try
            {
                pending_.insert (pending_.end(), data, data + size);
                return;
            }
            catch (const std::bad_alloc&)
            {
                allocationFailed_ = true;
                failedBytes_ = pending_.size();
                Buffer().swap (pending_);
            }
        }
        failedBytes_ += size;
    }

    template <typename OnLine, typename OnAllocationFailure>
    void finish (OnLine& onLine, OnAllocationFailure& onAllocationFailure)
    {
        if (allocationFailed_)
        {
            const auto bytes = failedBytes_;
            allocationFailed_ = false;
            failedBytes_ = 0;
            onAllocationFailure (bytes);
        }
        else if (! pending_.empty())
        {
            onLine (pending_.data(), pending_.size());
        }
        // A plan line can be hundreds of MiB. Don't keep that memory for the next short request.
        Buffer().swap (pending_);
    }

    Buffer pending_;
    size_t failedBytes_ = 0;
    bool allocationFailed_ = false;
};
} // namespace videohelper
