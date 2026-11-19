#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#include <windows.h>
#else
#include <cerrno>
#include <poll.h>
#include <unistd.h>
#endif

namespace programmableruntime::privatepayload
{
constexpr std::size_t packetSize = 40;

enum class Capability { available, unavailable };

constexpr Capability platformCapability()
{
    return Capability::available;
}

#if defined(_WIN32)
inline bool readOneShot(HANDLE handle, std::array<uint8_t, packetSize>& packet)
{
    std::size_t offset = 0;
    while (offset < packet.size())
    {
        DWORD count = 0;
        if (! ::ReadFile(handle, packet.data() + offset,
                         static_cast<DWORD>(packet.size() - offset), &count, nullptr)
            || count == 0)
            break;
        offset += static_cast<std::size_t>(count);
    }

    uint8_t extra = 0;
    DWORD extraCount = 0;
    const bool eof = offset == packet.size()
        && ! ::ReadFile(handle, &extra, 1, &extraCount, nullptr)
        && ::GetLastError() == ERROR_BROKEN_PIPE;
    ::CloseHandle(handle);
    if (! eof) packet.fill(0);
    return eof;
}
#else
inline bool readOneShot(int fd, std::array<uint8_t, packetSize>& packet,
                        int timeoutMs = 5000)
{
    const auto readReady = [fd, timeoutMs]()
    {
        pollfd item { fd, POLLIN | POLLHUP, 0 };
        int result;
        do result = ::poll(&item, 1, timeoutMs); while (result < 0 && errno == EINTR);
        return result > 0 && (item.revents & (POLLIN | POLLHUP)) != 0;
    };
    std::size_t offset = 0;
    while (offset < packet.size())
    {
        if (! readReady()) break;
        const auto count = ::read(fd, packet.data() + offset, packet.size() - offset);
        if (count > 0) { offset += static_cast<std::size_t>(count); continue; }
        if (count < 0 && errno == EINTR) continue;
        break;
    }
    bool exact = offset == packet.size();
    uint8_t extra = 0;
    if (exact && readReady())
    {
        ssize_t count;
        do count = ::read(fd, &extra, 1); while (count < 0 && errno == EINTR);
        exact = count == 0;
    }
    else exact = false;
    ::close(fd);
    if (! exact) packet.fill(0);
    return exact;
}
#endif
}