#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace arbitgpu
{
inline constexpr std::size_t kMaximumFrameMemoryAllocationClasses = 8;

enum class FrameMemoryAdmissionFailure : std::uint8_t
{
    none = 0,
    invalidRequest,
    byteCountOverflow,
    byteBudgetExceeded
};

struct FrameMemoryAllocationClass final
{
    std::uint64_t width = 0;
    std::uint64_t height = 0;
    std::uint64_t bytesPerPixel = 0;
    std::uint64_t allocatedSlots = 0;
};

struct FrameMemoryAdmissionRequest final
{
    std::array<FrameMemoryAllocationClass, kMaximumFrameMemoryAllocationClasses> classes {};
    std::size_t classCount = 0;
    std::uint64_t byteBudget = 0;
};

struct FrameMemoryAdmission final
{
    FrameMemoryAdmissionFailure failure = FrameMemoryAdmissionFailure::none;
    std::size_t failingClass = 0;
    std::uint64_t allocatedSlots = 0;
    std::uint64_t requestedBytes = 0;
    std::uint64_t byteBudget = 0;
};

inline bool admitFrameMemory (const FrameMemoryAdmissionRequest& request,
                              FrameMemoryAdmission& admission,
                              std::string& error)
{
    admission = {};
    admission.byteBudget = request.byteBudget;
    error.clear();
    if (request.classCount == 0
        || request.classCount > request.classes.size()
        || request.byteBudget == 0)
    {
        admission.failure = FrameMemoryAdmissionFailure::invalidRequest;
        error = "native frame-memory admission requires allocation classes and a positive byte budget";
        return false;
    }

    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t index = 0; index < request.classCount; ++index)
    {
        const auto& allocation = request.classes[index];
        admission.failingClass = index;
        if (allocation.width == 0 || allocation.height == 0
            || allocation.bytesPerPixel == 0 || allocation.allocatedSlots == 0)
        {
            admission.failure = FrameMemoryAdmissionFailure::invalidRequest;
            error = "native frame-memory allocation class " + std::to_string(index)
                + " has a zero dimension, pixel size, or slot count";
            return false;
        }
        if (allocation.width > maximum / allocation.height)
        {
            admission.failure = FrameMemoryAdmissionFailure::byteCountOverflow;
            error = "native frame-memory byte accounting overflow in allocation class "
                + std::to_string(index) + ": " + std::to_string(allocation.width)
                + " x " + std::to_string(allocation.height) + " pixels";
            return false;
        }
        const auto pixels = allocation.width * allocation.height;
        if (pixels > maximum / allocation.bytesPerPixel)
        {
            admission.failure = FrameMemoryAdmissionFailure::byteCountOverflow;
            error = "native frame-memory byte accounting overflow in allocation class "
                + std::to_string(index) + ": " + std::to_string(pixels)
                + " pixels x " + std::to_string(allocation.bytesPerPixel)
                + " bytes per pixel";
            return false;
        }
        const auto bytesPerSlot = pixels * allocation.bytesPerPixel;
        if (bytesPerSlot > maximum / allocation.allocatedSlots)
        {
            admission.failure = FrameMemoryAdmissionFailure::byteCountOverflow;
            error = "native frame-memory byte accounting overflow in allocation class "
                + std::to_string(index) + ": " + std::to_string(bytesPerSlot)
                + " bytes per slot x " + std::to_string(allocation.allocatedSlots)
                + " slots";
            return false;
        }
        const auto classBytes = bytesPerSlot * allocation.allocatedSlots;
        if (classBytes > maximum - admission.requestedBytes
            || allocation.allocatedSlots > maximum - admission.allocatedSlots)
        {
            admission.failure = FrameMemoryAdmissionFailure::byteCountOverflow;
            error = "native frame-memory aggregate accounting overflow at allocation class "
                + std::to_string(index);
            return false;
        }
        admission.requestedBytes += classBytes;
        admission.allocatedSlots += allocation.allocatedSlots;
    }

    if (admission.requestedBytes > admission.byteBudget)
    {
        admission.failure = FrameMemoryAdmissionFailure::byteBudgetExceeded;
        error = "native frame-memory budget exceeded: requested "
            + std::to_string(admission.requestedBytes) + " bytes for "
            + std::to_string(admission.allocatedSlots) + " slots, budget "
            + std::to_string(admission.byteBudget) + " bytes";
        return false;
    }
    return true;
}
} // namespace arbitgpu
