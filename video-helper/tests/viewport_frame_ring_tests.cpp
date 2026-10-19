// Docked-viewport shm ring across writer sessions (PROTOCOL.md §shm-docked).
//
// Arbit keeps one frame ring across editor close/open and helper restarts, and
// each helper session numbers its frames from 1 again. These tests pin the
// session handover: once the new writer retires the old sequences, its first
// frame is the newest one the reader sees, and a slot the dead writer left
// mid-write becomes readable again.

#include "../../plugin/Source/VideoFrameSharedMemory.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace
{
int failures = 0;

void check (bool condition, const char* message)
{
    if (! condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

constexpr uint32_t kSlots = 3;
constexpr uint32_t kWidth = 4;
constexpr uint32_t kHeight = 2;
constexpr uint32_t kSlotBytes = kWidth * kHeight * 4u;

// One writer session, written the way Viewport's shm publish path writes:
// round-robin slots, seqlock generation, per-session frame sequence.
struct WriterSession
{
    explicit WriterSession (videoshm::Region& target) : ring (target) {}

    videoshm::Region& ring;
    uint32_t cursor = 0;
    uint64_t seq = 0;

    void publish (double ptsSec, uint8_t fill)
    {
        const uint32_t index = cursor++ % ring.header()->slotCount;
        auto* slot = ring.slot (index);
        slot->generation.fetch_add (1, std::memory_order_acq_rel);
        slot->width = kWidth;
        slot->height = kHeight;
        slot->strideBytes = kWidth * 4u;
        slot->ptsSec = ptsSec;
        const uint64_t frame = ++seq;
        slot->reserved[0] = (uint32_t) (frame & 0xffffffffu);
        slot->reserved[1] = (uint32_t) (frame >> 32);
        slot->reserved[2] = 1;
        std::vector<uint8_t> pixels (kSlotBytes, fill);
        std::memcpy (ring.slotPayload (index), pixels.data(), pixels.size());
        slot->generation.fetch_add (1, std::memory_order_acq_rel);
    }
};

std::string uniqueName (const char* prefix)
{
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    return prefix + std::to_string ((unsigned long long) ticks % 1000000000ull);
}

bool openRing (videoshm::Region& ring, const char* prefix)
{
    if (ring.create (uniqueName (prefix), kSlots, kSlotBytes))
        return true;
    std::cerr << "FAIL: could not create the test shm ring\n";
    ++failures;
    return false;
}

// The defect this guards: without retirement, the old session's frames outrank
// the new session's first frames.
void staleSlotsHideNewFramesWithoutRetirement()
{
    videoshm::Region ring;
    if (! openRing (ring, "arb-vrt-a")) return;
    WriterSession oldSession { ring };
    for (int i = 0; i < 21; ++i) oldSession.publish (1.0, 0x10);

    WriterSession newSession { ring };
    newSession.publish (0.5, 0x20);
    uint32_t index = 0;
    uint64_t seq = 0;
    check (ring.newestFrameSlot (index, seq), "a written ring has a newest frame");
    check (seq == 21, "an unretired ring keeps the old session's seq on top");
    check (ring.slot (index)->ptsSec == 1.0, "an unretired ring shows the old session's picture");
}

void retirementLetsTheNewSessionPresent()
{
    videoshm::Region ring;
    if (! openRing (ring, "arb-vrt-b")) return;
    WriterSession oldSession { ring };
    for (int i = 0; i < 39; ++i) oldSession.publish (1.0, 0x10);

    ring.retireFrameSequences();
    uint32_t index = 0;
    uint64_t seq = 0;
    check (! ring.newestFrameSlot (index, seq), "a retired ring has no frame to show");
    check (seq == 0, "a retired ring reports seq 0");
    for (uint32_t i = 0; i < kSlots; ++i)
    {
        check ((ring.slot (i)->generation.load() & 1u) == 0, "retirement leaves every slot stable");
        check (videoshm::slotFrameSeq (*ring.slot (i)) == 0, "retirement clears every slot seq");
    }

    WriterSession newSession { ring };
    newSession.publish (0.5, 0x20);
    check (ring.newestFrameSlot (index, seq), "the new session's first frame is visible");
    check (seq == 1, "the new session's first frame is the newest");
    check (ring.slot (index)->ptsSec == 0.5, "the newest frame carries the new session's PTS");

    // A seek in the same session must keep raising the seq the reader reports.
    newSession.publish (0.0, 0x30);
    check (ring.newestFrameSlot (index, seq) && seq == 2, "a seek frame raises the seq");
    check (ring.slot (index)->ptsSec == 0.0, "the seek frame carries the seek PTS");
}

void retirementRepairsASlotLeftMidWrite()
{
    videoshm::Region ring;
    if (! openRing (ring, "arb-vrt-c")) return;
    WriterSession oldSession { ring };
    for (int i = 0; i < 4; ++i) oldSession.publish (1.0, 0x10);
    // The old helper died inside a write: generation stays odd.
    ring.slot (2)->generation.fetch_add (1, std::memory_order_acq_rel);
    check ((ring.slot (2)->generation.load() & 1u) != 0, "fixture leaves slot 2 mid-write");

    ring.retireFrameSequences();
    check ((ring.slot (2)->generation.load() & 1u) == 0, "retirement makes the torn slot stable");

    WriterSession newSession { ring };
    for (int i = 0; i < 3; ++i) newSession.publish (0.25 * i, (uint8_t) (0x40 + i));
    uint32_t index = 0;
    uint64_t seq = 0;
    check (ring.newestFrameSlot (index, seq) && index == 2 && seq == 3,
           "the new session's write into the repaired slot is the newest frame");

    uint32_t width = 0, height = 0, stride = 0;
    double pts = -1.0;
    std::vector<uint8_t> pixels (kSlotBytes, 0);
    check (ring.readSlot (2, width, height, stride, pts, pixels.data(), pixels.size()),
           "the repaired slot reads back stable");
    check (pixels[0] == 0x42, "the repaired slot holds the new session's pixels");
}
}

int main()
{
    staleSlotsHideNewFramesWithoutRetirement();
    retirementLetsTheNewSessionPresent();
    retirementRepairsASlotLeftMidWrite();
    if (failures != 0)
    {
        std::cerr << failures << " viewport frame ring check(s) failed\n";
        return 1;
    }
    std::cout << "viewport frame ring tests passed\n";
    return 0;
}
