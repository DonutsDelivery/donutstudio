#include "../src/gpu_backend/frame_memory_budget.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

namespace
{
int failures = 0;

void check (bool condition, const char* message)
{
    if (condition) return;
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}
}

int main()
{
    arbitgpu::FrameMemoryAdmissionRequest request;
    request.classCount = 1;
    request.classes[0] = { 1920, 1080, 8, 4 };
    request.byteBudget = 1920ull * 1080ull * 8ull * 4ull;

    arbitgpu::FrameMemoryAdmission admission;
    std::string error;
    check (arbitgpu::admitFrameMemory (request, admission, error)
           && error.empty()
           && admission.failure == arbitgpu::FrameMemoryAdmissionFailure::none
           && admission.requestedBytes == request.byteBudget
           && admission.byteBudget == request.byteBudget
           && admission.allocatedSlots == 4,
           "an exact-fit RGBA16F four-slot allocation must be admitted");

    request.byteBudget -= 1;
    check (! arbitgpu::admitFrameMemory (request, admission, error)
           && admission.failure
                  == arbitgpu::FrameMemoryAdmissionFailure::byteBudgetExceeded
           && admission.requestedBytes == 66'355'200
           && admission.byteBudget == 66'355'199
           && admission.allocatedSlots == 4
           && error == "native frame-memory budget exceeded: requested 66355200 bytes for 4 slots, budget 66355199 bytes",
           "one byte over budget must reject with exact attempted telemetry");

    request = {};
    request.classCount = 1;
    request.byteBudget = std::numeric_limits<std::uint64_t>::max();
    request.classes[0] = {
        std::numeric_limits<std::uint64_t>::max(), 2, 1, 1
    };
    check (! arbitgpu::admitFrameMemory (request, admission, error)
           && admission.failure
                  == arbitgpu::FrameMemoryAdmissionFailure::byteCountOverflow
           && admission.failingClass == 0
           && admission.requestedBytes == 0
           && error.find ("18446744073709551615 x 2 pixels") != std::string::npos,
           "dimension multiplication overflow must reject without wrapping");

    request = {};
    request.classCount = 2;
    request.byteBudget = std::numeric_limits<std::uint64_t>::max();
    request.classes[0] = { 1, 1, 1, std::numeric_limits<std::uint64_t>::max() };
    request.classes[1] = { 1, 1, 1, 1 };
    check (! arbitgpu::admitFrameMemory (request, admission, error)
           && admission.failure
                  == arbitgpu::FrameMemoryAdmissionFailure::byteCountOverflow
           && admission.failingClass == 1
           && admission.requestedBytes == std::numeric_limits<std::uint64_t>::max()
           && admission.allocatedSlots == std::numeric_limits<std::uint64_t>::max()
           && error == "native frame-memory aggregate accounting overflow at allocation class 1",
           "aggregate byte and slot overflow must reject without disturbing prior class telemetry");

    request = {};
    request.classCount = 2;
    request.byteBudget = 1000;
    request.classes[0] = { 10, 10, 4, 2 };
    request.classes[1] = { 10, 10, 2, 1 };
    check (arbitgpu::admitFrameMemory (request, admission, error)
           && admission.requestedBytes == 1000
           && admission.allocatedSlots == 3,
           "heterogeneous format classes must add exact bytes and slot counts");

    std::cout << "frame memory budget admission: "
              << (failures == 0 ? "PASS" : "FAIL") << '\n';
    return failures == 0 ? 0 : 1;
}
