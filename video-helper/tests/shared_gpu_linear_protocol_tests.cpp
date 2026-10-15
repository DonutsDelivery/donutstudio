#include "SharedGpuSurfaceProtocol.h"
#include <cassert>
#include <limits>

int main()
{
    using namespace gpusurf;
    static_assert(static_cast<uint32_t>(MsgType::Hello) == 1 && static_cast<uint32_t>(MsgType::FrameRelease) == 5);
    static_assert(sizeof(MsgHeader) == 16 && sizeof(HelloPayload) == 8 && sizeof(BufferInfo) == 60);
    static_assert(sizeof(FrameReadyPayload) == 32 && sizeof(FrameReleasePayload) == 8);
    const LinearIOSurfaceOfferPayload offer {kLinearIOSurfaceVersion, kLinearSceneProfile, 0xabc123};
    LinearIOSurfaceNegotiation producer;
    assert(producer.begin(offer, 1000));
    assert(!producer.expire(1099));
    assert(producer.select(&offer, sizeof(offer), 0, 1099) && producer.selected);
    assert(!producer.select(&offer, sizeof(offer), 0, 1100)); // frozen per connection
    // Old helper offers no capability. A new consumer keeps byte-exact v2.
    auto absent = offer; absent.version = 0;
    assert(!producer.begin(absent, 1000) && !producer.selected);
    // Old consumer does not reply. The new helper falls back once without waiting on I/O.
    assert(producer.begin(offer, 1000));
    assert(producer.expire(1100) && !producer.expire(1101));
    assert(!producer.select(&offer, sizeof(offer), 0, 1101) && !producer.selected);
    for (int malformed = 0; malformed < 7; ++malformed)
    {
        auto selected = offer;
        if (malformed == 0) selected.version++;
        if (malformed == 1) selected.profile++;
        if (malformed == 2) selected.metalRegistryId++;
        if (malformed == 3) selected.metalRegistryId = 0;
        assert(producer.begin(offer, 1000));
        assert(!producer.select(&selected, malformed == 4 ? sizeof(selected) - 1 : sizeof(selected),
            malformed == 5 ? 1 : 0, malformed == 6 ? 1100 : 1001));
        assert(!producer.selected && !producer.waiting);
    }
    LinearIOSurfaceBuffersPayload list {offer, 1, 3};
    const size_t bytes = sizeof(list) + 3 * sizeof(LinearIOSurfaceBufferInfo);
    assert(validLinearBufferGeneration(list, offer, 0, bytes));
    assert(!validLinearBufferGeneration(list, absent, 0, bytes)); // unselected
    assert(!validLinearBufferGeneration(list, offer, 1, bytes)); // stale generation
    assert(!validLinearBufferGeneration(list, offer, 0, bytes - 1));
    auto badList = list; badList.declaration.profile++;
    assert(!validLinearBufferGeneration(badList, offer, 0, bytes));
    badList = list; badList.generation = 0;
    assert(!validLinearBufferGeneration(badList, offer, 0, bytes));
    badList = list; badList.bufferCount = 5;
    assert(!validLinearBufferGeneration(badList, offer, 0, bytes));
    LinearIOSurfaceBufferInfo buffer {0, 99, 13, 7, 128, 1024};
    assert(validLinearBuffer(buffer, 0)); // actual padded stride/allocation allowed
    for (int malformed = 0; malformed < 7; ++malformed)
    {
        auto bad = buffer;
        if (malformed == 0) bad.index++;
        if (malformed == 1) bad.surfaceId = 0;
        if (malformed == 2) bad.width = 0;
        if (malformed == 3) bad.rowBytes = 13 * 4; // BGRA bytes may not masquerade as float
        if (malformed == 4) bad.rowBytes++;
        if (malformed == 5) bad.allocationBytes = 128 * 7 - 1;
        if (malformed == 6) bad.rowBytes = UINT64_MAX - 7;
        assert(!validLinearBuffer(bad, 0));
    }
    LinearIOSurfaceFrameReadyPayload frame {1, {2, 0, 7, 0.5, 0}};
    assert(validLinearReady(frame, 1, 3, 6));
    assert(!validLinearReady(frame, 0, 3, 6)); // unselected or teardown
    assert(!validLinearReady(frame, 2, 3, 6)); // resize invalidates old generation
    assert(!validLinearReady(frame, 1, 2, 6));
    assert(!validLinearReady(frame, 1, 3, 7));
    frame.frame.flags = kFlagFenceSync;
    assert(!validLinearReady(frame, 1, 3, 6));
    frame.frame.flags = 0; frame.frame.displaySec = std::numeric_limits<double>::quiet_NaN();
    assert(!validLinearReady(frame, 1, 3, 6));
    const LinearIOSurfaceFrameReleasePayload release {1, {2, 0}};
    assert(validLinearRelease(release, 1, 3));
    assert(!validLinearRelease(release, 2, 3)); // a delayed GPU completion cannot free a resized ring
    assert(!validLinearRelease(release, 0, 3));
    assert(!validLinearRelease(release, 1, 2));
    list.generation = 2;
    assert(validLinearBufferGeneration(list, offer, 1, bytes));
    return 0;
}
