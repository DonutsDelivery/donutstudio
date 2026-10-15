#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <string>

namespace videohelper::harmonicprobe
{
// Counts from one composite probe request. The capture never owns score or GPU data.
struct Receipt
{
    bool scoreObjectPresent = false;
    std::uint64_t parsedScoreRevision = 0;
    std::size_t parsedNoteCount = 0;
    std::size_t parsedLinkCount = 0;
    std::size_t plannedHarmonicLayerCount = 0;
    bool canonicalFramePresent = false;
    int canonicalNoteRows = 0;
    int canonicalLinkRows = 0;
    std::size_t lowerCallCount = 0;
    std::size_t emittedVertexCount = 0;
    std::size_t emittedIndexCount = 0;
    std::size_t emptyLayerCount = 0;
    std::size_t nativeLayerCount = 0;
    bool nativeColorRead = false;
    std::string nativeColorReadError;
    int nativeMaximumContrast = 0;
    std::size_t nativeForegroundPixels = 0;
    std::array<int, 4> nativeForegroundBounds {};
};

inline thread_local Receipt* activeReceipt = nullptr;

struct ScopedCapture
{
    explicit ScopedCapture(Receipt* receipt) : previous(activeReceipt)
    {
        activeReceipt = receipt;
    }
    ~ScopedCapture() { activeReceipt = previous; }

    ScopedCapture(const ScopedCapture&) = delete;
    ScopedCapture& operator=(const ScopedCapture&) = delete;

    Receipt* previous;
};
} // namespace videohelper::harmonicprobe
