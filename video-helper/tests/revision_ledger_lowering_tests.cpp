// Revision ledger transitions for a lowering or compile failure of the pending
// authoring revision.
//
// RevisionLedger is header-only in render_snapshot.h. These cases drive the
// ledger alone, so they need no viewport, GL, or Metal backend.

#include "../src/render_snapshot.h"

#include <cstdio>

namespace
{
int failures = 0;

void check (bool condition, const char* message)
{
    if (! condition) { ++failures; std::fprintf (stderr, "FAIL: %s\n", message); }
}

bool sameTuple (const videowire::RevisionTuple& a, const videowire::RevisionTuple& b)
{
    return a.authoring == b.authoring && a.accepted == b.accepted
        && a.rejected == b.rejected && a.compiled == b.compiled
        && a.lastGood == b.lastGood && a.exportable == b.exportable
        && a.evaluation == b.evaluation;
}
} // namespace

int main()
{
    using videowire::RevisionLedger;

    // Revision 1 compiles as last-good and exportable.
    RevisionLedger ledger;
    check (ledger.requestAuthoring (1) && ledger.accept (1)
           && ledger.compileSucceeded (1, true),
           "revision 1 compiles as last-good and exportable");

    // A lowering failure of pending revision 2 rejects it, clears exportability,
    // and leaves revision 1 as last-good.
    check (ledger.requestAuthoring (2) && ledger.compileFailed (2),
           "lowering failure of pending revision 2 is recorded");
    const auto rejected = ledger.state();
    check (rejected.authoring == 2 && rejected.accepted == 1 && rejected.rejected == 2
           && rejected.compiled == 1 && rejected.lastGood == 1 && rejected.exportable == 0,
           "revision 2 is rejected with exportability cleared and revision 1 last-good");
    check (! ledger.accept (2) && ! ledger.compileSucceeded (2, true),
           "rejected revision 2 cannot be accepted or compiled later");

    // Stale and foreign failures are refused and leave the ledger unchanged.
    check (ledger.requestAuthoring (3) && ledger.state().rejected == 0,
           "requesting revision 3 clears the rejection of revision 2");
    const auto pending = ledger.state();
    check (! ledger.compileFailed (2) && ! ledger.compileFailed (4)
           && sameTuple (ledger.state(), pending),
           "a stale or foreign lowering failure is refused without changing the ledger");

    // A lowering failure of revision 3 rejects it the same way.
    check (ledger.compileFailed (3) && ledger.state().rejected == 3
           && ledger.state().exportable == 0 && ledger.state().lastGood == 1,
           "a second lowering failure rejects revision 3 and keeps revision 1 last-good");

    // A later good compile clears the rejection and becomes exportable.
    check (ledger.requestAuthoring (4) && ledger.accept (4)
           && ledger.compileSucceeded (4, true)
           && ledger.state().rejected == 0 && ledger.state().lastGood == 4
           && ledger.state().exportable == 4,
           "a later good compile clears the lowering rejection");

    // A completed revision is never rejected by a late lowering failure.
    const auto completed = ledger.state();
    check (! ledger.compileFailed (4) && sameTuple (ledger.state(), completed),
           "a late lowering failure cannot reject the completed revision");

    // A lowering failure leaves exactly the state that reject() leaves.
    RevisionLedger viaReject;
    RevisionLedger viaCompile;
    check (viaReject.requestAuthoring (1) && viaReject.accept (1)
           && viaReject.compileSucceeded (1, true)
           && viaCompile.requestAuthoring (1) && viaCompile.accept (1)
           && viaCompile.compileSucceeded (1, true)
           && viaReject.requestAuthoring (2) && viaCompile.requestAuthoring (2)
           && viaReject.reject (2) && viaCompile.compileFailed (2)
           && sameTuple (viaReject.state(), viaCompile.state()),
           "a lowering failure leaves the same ledger state as a rejection");

    if (failures != 0) {
        std::fprintf (stderr, "%d revision ledger lowering check(s) failed\n", failures);
        return 1;
    }
    std::printf ("revision ledger lowering checks passed\n");
    return 0;
}
