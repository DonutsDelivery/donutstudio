#include "../src/composite_probe_contract.h"
#include "../src/composite_render_diagnostic.h"
#include "../src/bounded_line.h"
#include "../src/compositor_ownership.h"

#include <cstdio>
#include <limits>
#include <new>
#include <sstream>
#include <string>

namespace
{
int checks = 0;
int failures = 0;
void check (bool condition, const char* message)
{
    ++checks;
    if (! condition) { ++failures; std::fprintf (stderr, "FAIL: %s\n", message); }
}

videowire::ResolvedVisualSnapshot validSnapshot()
{
    videowire::ResolvedVisualSnapshot snapshot;
    snapshot.segments.resize (1);
    snapshot.segments[0].clipId = 4;
    snapshot.visualLayerPlans.resize (1);
    snapshot.visualLayerPlans[0].clipId = 4;
    snapshot.visualLayerPlans[0].structuralRevision = 9;
    snapshot.visualLayerPlans[0].producerValidated = true;
    snapshot.authoringRevision = 5;
    return snapshot;
}

void diagnosticContract()
{
    using namespace videohelper::compositediagnostic;
    check(!enabledValue(nullptr) && !enabledValue("") && !enabledValue("0")
              && !enabledValue("01") && !enabledValue("1 ") && !enabledValue("true"),
          "diagnostic opt-in accepts only the exact environment value 1");
    check(enabledValue("1"), "exact diagnostic opt-in is enabled");
    char line[512];
    Record record;
    check(format(line, record) > 0
              && std::string(line) == "[composite-render-diagnostic] rpcId=0 phase=1 backend=0"
                  " revision=0 generation=0 width=0 height=0 elapsedMs=0 commandBufferStatus=0\n",
          "numeric diagnostic grammar and LF are exact at zero");
    record = {UINT64_MAX, Phase::fenceWaitEnd, Backend::metal, UINT64_MAX, UINT64_MAX,
              UINT32_MAX, UINT32_MAX, UINT64_MAX, 5};
    const int length = format(line, record);
    check(length > 0 && length < 512 && static_cast<std::size_t>(length) == std::string(line).size()
              && std::string(line) == "[composite-render-diagnostic] rpcId=18446744073709551615"
                  " phase=20 backend=2 revision=18446744073709551615 generation=18446744073709551615"
                  " width=4294967295 height=4294967295 elapsedMs=18446744073709551615"
                  " commandBufferStatus=5\n",
          "maximum numeric values remain canonical and below the collector line cap");
    record.commandBufferStatus = 255;
    format(line, record);
    check(std::string(line).find("commandBufferStatus=0\n") != std::string::npos,
          "unknown command-buffer status never escapes the numeric range");
    RecordBudget budget;
    unsigned admitted = 0;
    for (unsigned i = 0; i < 4096; ++i) admitted += budget.claim() ? 1 : 0;
    check(admitted == 2048 && !budget.claim(), "diagnostic record budget saturates at 2048");

    const char* oldValue = std::getenv("ARBIT_COMPOSITE_RENDER_DIAGNOSTICS");
    const bool wasSet = oldValue != nullptr;
    const std::string saved = wasSet ? oldValue : "";
    const auto setValue = [](const char* value) {
#if defined(_WIN32)
        return _putenv_s("ARBIT_COMPOSITE_RENDER_DIAGNOSTICS", value != nullptr ? value : "");
#else
        return value != nullptr ? setenv("ARBIT_COMPOSITE_RENDER_DIAGNOSTICS", value, 1)
                                : unsetenv("ARBIT_COMPOSITE_RENDER_DIAGNOSTICS");
#endif
    };
    check(setValue(nullptr) == 0, "unset diagnostic environment for default-off check");
    {
        ScopedRequest off(true, 99);
        check(current == nullptr, "unset environment creates no diagnostic context");
    }
    check(setValue("0") == 0, "explicitly disable diagnostic environment");
    {
        ScopedRequest off(true, 99);
        check(current == nullptr, "explicit zero creates no diagnostic context");
    }
    check(setValue("1") == 0, "enable diagnostic environment for scoped fixture");
    {
        ScopedRequest outer(true, UINT64_MAX);
        identity(17, 23, 1920, 1080);
        backend(Backend::metal);
        Context* owner = current;
        check(owner != nullptr && owner->record.rpcId == UINT64_MAX,
              "scoped diagnostic retains the full unsigned RPC identity");
        {
            ScopedRequest nested(true, 7);
            check(current != nullptr && current != owner && current->record.rpcId == 7
                      && current->record.generation == 0 && current->record.width == 0
                      && current->record.backend == Backend::unknown,
                  "nested request begins with its own empty numeric context");
            identity(31, 37, -1, 0);
            check(current != nullptr && current->record.width == 0 && current->record.height == 0,
                  "unavailable dimensions remain zero");
        }
        check(current == owner && current != nullptr && current->record.revision == 17
                  && current->record.generation == 23 && current->record.width == 1920
                  && current->record.height == 1080 && current->record.backend == Backend::metal,
              "nested request restores the outer identity and backend");
        {
            ScopedRequest unrelated(false, 8);
            check(current == nullptr, "unrelated request cannot emit under the outer context");
        }
        check(current == owner, "unrelated nested request restores the outer context");
        try { ScopedRequest failed(true, 9); throw 1; }
        catch (int) {}
        check(current == owner, "exception restores the preceding request context");
    }
    check(current == nullptr, "request completion clears diagnostic identity");
    check(setValue(wasSet ? saved.c_str() : nullptr) == 0, "restore diagnostic environment");
}
}

int main()
{
    diagnosticContract();
    std::string error;
    auto snapshot = validSnapshot();
    check (videohelper::validateCompositeProbeContract(snapshot, 9, 1.25, 1920, 1080, 30.0, error),
           "processor snapshot is admitted");
    snapshot.visualLayerPlans.push_back(snapshot.visualLayerPlans.front());
    snapshot.visualLayerPlans.back().clipId = 5;
    snapshot.visualLayerPlans.back().structuralRevision = 17;
    check (videohelper::validateCompositeProbeContract(snapshot, 10, 1.25, 1920, 1080, 30.0, error),
           "valid multi-clip snapshots may have unequal authored revisions");
    snapshot.visualLayerPlans[0].identityMode = "transientLegacyProjection";
    snapshot.visualLayerPlans[0].structuralRevision = 0;
    check (videohelper::validateCompositeProbeContract(snapshot, 10, 1.25, 1920, 1080, 30.0, error),
           "mixed legacy and authored plans with unequal revisions are admitted");
    std::vector<videohelper::SnapshotClipIdentity> identities {
        { 4, 0, "transientLegacyProjection", true }, { 5, 17, "authoredGraph", true } };
    check (videohelper::validateSnapshotClipIdentities(snapshot.visualLayerPlans, identities, error),
           "records match plans by clip id and identity mode");
    identities[0].mode = "authoredGraph";
    check (! videohelper::validateSnapshotClipIdentities(snapshot.visualLayerPlans, identities, error),
           "record identity-mode mismatch is rejected");
    identities[0] = { 4, 0, "transientLegacyProjection", true };
    identities[1].revision = 18;
    check (! videohelper::validateSnapshotClipIdentities(snapshot.visualLayerPlans, identities, error),
           "record revision mismatch is rejected");
    snapshot = validSnapshot();
    snapshot.visualLayerPlans[0].producerValidated = false;
    check (! videohelper::validateCompositeProbeContract(snapshot, 9, 1.25, 1920, 1080, 30.0, error),
           "invalid compiled plan is rejected");
    snapshot = validSnapshot();
    check (! videohelper::validateCompositeProbeContract(snapshot, 0, 1.25, 1920, 1080, 30.0, error),
           "unidentified processor snapshot is rejected");
    snapshot.authoringRevision = 0;
    check (! videohelper::validateCompositeProbeContract(snapshot, 9, 1.25, 1920, 1080, 30.0, error)
               && error == "composite probe requires a processor-owned authoring revision",
           "probe without an authoring revision is refused before any render");
    snapshot.authoringRevision = 5;
    check (! videohelper::validateCompositeProbeContract(snapshot, 9,
               std::numeric_limits<double>::quiet_NaN(), 1920, 1080, 30.0, error),
           "non-finite time is rejected");
    check (! videohelper::validateCompositeProbeContract(snapshot, 9, 0.0, 1921, 1080, 30.0, error),
           "oversized canvas is rejected");

    check (!videohelper::validateCompositeProbeContract(snapshot, 9, 0.0, 3840, 2160, 30.0, error),
           "4K cannot enter the inline helper response");
    check (videohelper::validateCompositeProbeContract(snapshot, 9, 0.0, 3840, 2160, 30.0, error, true),
           "exact 4K is admitted for artifact transport");
    check (!videohelper::validateCompositeProbeContract(snapshot, 9, 0.0, 3842, 2160, 30.0, error, true),
           "artifact dimensions are exact");
    {
        using namespace compositeartifact;
        auto now = std::chrono::steady_clock::now();
        Store store([&] { return now; });
        bool current = true;
        const auto create = [&](const std::string& owner)
        { return store.create(owner, std::vector<uint8_t>(maxBytes, 41), [&] { return current; }, error); };
        const auto first = create("first");
        check(validHandle(first), "artifact handles use secure opaque tokens");
        check(create("first").empty(), "one live handle per principal");
        const auto second = create("second");
        check(!second.empty() && create("third").empty(), "global handle and byte budget is bounded");
        std::vector<uint8_t> bytes;
        bool done = false;
        check(!store.read("wrong", first, 0, bytes, done, error), "wrong owner cannot read");
        check(!store.release("wrong", first, error), "wrong owner cannot release");
        check(!store.read("first", "../path", 0, bytes, done, error), "malformed handles fail");
        size_t offset = 0;
        int chunks = 0;
        while (offset < maxBytes)
        {
            if (!store.read("first", first, offset, bytes, done, error))
            { check(false, "sequential 4K transfer succeeds"); break; }
            check(bytes.size() == std::min(chunkBytes, maxBytes - offset), "chunk size is exact and bounded");
            check(std::all_of(bytes.begin(), bytes.end(), [](auto byte) { return byte == 41; }), "pixels are lossless");
            offset += bytes.size();
            check(done == (offset == maxBytes), "only last chunk finishes the transfer");
            ++chunks;
        }
        check(offset == 33177600 && chunks == 32, "full 4K is delivered in 32 bounded chunks");
        check(!store.read("first", first, 0, bytes, done, error), "completion destroys the handle");
        check(!store.release("first", first, error), "completed handles cannot be replayed");
        const auto replay = create("first");
        check(store.read("first", replay, 0, bytes, done, error), "first read succeeds");
        check(!store.read("first", replay, 0, bytes, done, error), "repeated offset fails and clears owned bytes");
        check(!store.read("first", replay, chunkBytes, bytes, done, error), "failed transfer cannot resume");
        const auto skipped = create("first");
        check(!store.read("first", skipped, chunkBytes, bytes, done, error), "skipped offset fails");
        const auto stale = create("first");
        current = false;
        check(!store.read("first", stale, 0, bytes, done, error), "stale identity fails and frees bytes");
        current = true;
        const auto expired = create("first");
        now += std::chrono::milliseconds(lifetimeMs);
        check(!store.read("first", expired, 0, bytes, done, error), "exact TTL boundary expires");
        check(!create("third").empty(), "expiry frees quota for every principal");
        store.clear();
        check(!store.read("second", second, 0, bytes, done, error), "shutdown/reopen clear invalidates handles");
        const auto released = create("first");
        check(store.release("first", released, error), "explicit release succeeds");
        check(!store.read("first", released, 0, bytes, done, error), "released handle is invalid");
        check(store.create("first", std::vector<uint8_t>(16), [] { return true; }, error).empty(),
              "incorrect raw byte count is rejected");
    }

    std::vector<char> line;
    std::istringstream exact (std::string (64, 'x') + "\n");
    check (videohelper::readBoundedLine (exact, line, 64) == videohelper::BoundedLineResult::line
               && line.size() == 64, "exact-limit helper request is accepted");
    std::istringstream oversized (std::string (65, 'x'));
    check (videohelper::readBoundedLine (oversized, line, 64) == videohelper::BoundedLineResult::oversized
               && line.empty(), "newline-free oversized helper request is rejected at the cap");
    std::istringstream partial ("partial");
    check (videohelper::readBoundedLine (partial, line, 64) == videohelper::BoundedLineResult::line
               && std::string (line.begin(), line.end()) == "partial",
           "bounded parser preserves a final partial line at EOF");
    std::istringstream empty;
    check (videohelper::readBoundedLine (empty, line, 64) == videohelper::BoundedLineResult::end,
           "clean EOF is distinguished from a partial line");

    videohelper::CompositorOwnershipGate gate;
    auto exportLease = gate.tryClaim(videohelper::CompositorOwnershipGate::Owner::exportJob);
    check (exportLease && ! gate.tryClaim(videohelper::CompositorOwnershipGate::Owner::frameProbe)
               && ! gate.tryClaim(videohelper::CompositorOwnershipGate::Owner::renderCache)
               && ! gate.tryClaim(videohelper::CompositorOwnershipGate::Owner::recipePreview),
           "one atomic gate deterministically excludes preview, probe, and cache during export");
    exportLease.reset();
    check (gate.tryClaim(videohelper::CompositorOwnershipGate::Owner::frameProbe) != nullptr,
           "RAII completion releases compositor ownership");

    videohelper::CompositorOwnershipGate previewGate;
    auto previewLease = previewGate.tryClaim(
        videohelper::CompositorOwnershipGate::Owner::recipePreview);
    check (previewLease
               && ! previewGate.tryClaim(videohelper::CompositorOwnershipGate::Owner::exportJob)
               && ! previewGate.tryClaim(videohelper::CompositorOwnershipGate::Owner::frameProbe),
           "recipe preview exclusively owns the shared production compositor");

    std::string previewError;
    check (videohelper::validateRecipePreviewPixels(
               320, 180, 320u * 180u * 4u, "opengl", previewError),
           "recipe preview admits complete production GPU pixels");
    check (! videohelper::validateRecipePreviewPixels(
               320, 180, 0, "opengl", previewError)
               && previewError == "recipe preview produced incomplete production GPU pixels",
           "recipe preview fails closed on absent rendered pixels");
    check (! videohelper::validateRecipePreviewPixels(
               320, 180, 320u * 180u * 4u, {}, previewError)
               && previewError == "recipe preview produced no production GPU backend receipt",
           "recipe preview fails closed without a production backend receipt");

    videohelper::CompositorOwnershipGate allocationFailureGate;
    bool sawBadAlloc = false;
    try
    {
        allocationFailureGate.tryClaimWithFactory(
            videohelper::CompositorOwnershipGate::Owner::exportJob,
            [] (auto&, auto) -> std::shared_ptr<videohelper::CompositorOwnershipGate::Lease>
            { throw std::bad_alloc(); });
    }
    catch (const std::bad_alloc&) { sawBadAlloc = true; }
    check (sawBadAlloc
               && allocationFailureGate.owner() == videohelper::CompositorOwnershipGate::Owner::none
               && allocationFailureGate.tryClaim(
                      videohelper::CompositorOwnershipGate::Owner::renderCache) != nullptr,
           "lease allocation failure restores the gate for the next compositor owner");

    std::printf ("composite probe contract: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
