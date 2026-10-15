// Rotating-plane value anchors on every platform, including Apple, where the
// GL visual plan executor test is not built. Literal float and double bits
// assume no fused multiply-add contraction; CMake pins -ffp-contract=off.
#include "../src/particle_rotating_plane_response.h"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>

namespace
{
std::atomic<std::uint64_t> allocations { 0 };
int failures = 0;
void check (bool condition, const char* message)
{
    if (! condition) { ++failures; std::fprintf (stderr, "FAIL: %s\n", message); }
}
}

void* operator new (std::size_t size)
{
    allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* value = std::malloc(size)) return value;
    throw std::bad_alloc();
}
void operator delete (void* value) noexcept { std::free(value); }
void operator delete (void* value, std::size_t) noexcept { std::free(value); }

#include "rotating_plane_anchor_fixture.h"

int main()
{
    checkRotatingExactScalarPrerequisite();
    checkRotatingPlaneGeometricPrerequisite();
    checkRotatingPlaneResponsePrerequisite();
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
