#pragma once

#include <cstddef>
#include <cstdint>

// The helper names why it closed in its final BuffersGone. Every surface
// consumer keeps that reason, and the helper and consumers share this cap and cut.
namespace gpusurf
{

// Fits the Linux 4096-byte receive buffer.
inline constexpr std::size_t kMaxCloseReasonBytes = 512;

// End-of-stream text for a producer that closed without a reason.
inline constexpr const char* kProducerClosedText = "producer closed the surface channel";

// How many leading bytes of a close reason to keep. The cut stops at maxBytes
// and backs off to a byte that starts a sequence, so it never splits UTF-8.
inline std::size_t closeReasonPrefixBytes (const std::uint8_t* text, std::size_t size,
                                           std::size_t maxBytes = kMaxCloseReasonBytes)
{
    std::size_t cut = size < maxBytes ? size : maxBytes;
    while (cut > 0 && cut < size && (text[cut] & 0xC0u) == 0x80u)
        --cut;
    return cut;
}

} // namespace gpusurf
