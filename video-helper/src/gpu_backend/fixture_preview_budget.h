#pragma once

// Which shader budgets a Preview fixture draw takes (tier K, the Render node mode).
//
// A Preview draw (NativeImportedSceneRuntimeInputs::previewUse) keeps its pixels
// on the GPU: no host readback, no CPU frame validation and no CPU wait. That
// stays true in both modes. The budget is a separate choice:
//
// - Preview budgets (the 3D Render node's Fast mode) are the reductions the
//   preview has always drawn with: the GLASS_PREVIEW_VISITS camera walk, the
//   coarser caustic entry schedule and the caustic budget ladder.
// - Export budgets (Accurate, the default) draw the Preview frame with the
//   shader programs, caustic schedule and resolve export uses, so a settled
//   Preview frame equals the export frame. The caustic may still arrive a few
//   frames late through the fenced submission poll; it is never reduced or
//   dropped for time.
//
// FixtureSceneRenderer opens a scope around every fixture draw it submits: Export
// for an export draw and for an Accurate Preview, Preview for a Fast Preview. The
// backend reads it only on Preview draws, at every site where a Preview reduction
// changes pixels. It
// is per thread because a fixture draw runs synchronously on the thread that
// owns the GL context; it travels this way and not as a runtime-input field so
// the shared backend.h layout does not change.
#include <cstdint>

namespace arbitgpu::previewbudget
{
enum class Budget : std::uint8_t
{
    Preview = 0,
    Export = 1
};

inline Budget& currentBudget() noexcept
{
    thread_local Budget budget = Budget::Preview;
    return budget;
}

// True inside an export draw or an Accurate Preview draw. Only Preview draws
// consult it; an export draw takes the export budgets either way.
inline bool exportBudgets() noexcept { return currentBudget() == Budget::Export; }

class Scope final
{
public:
    explicit Scope (Budget budget) noexcept : previous_ (currentBudget()) { currentBudget() = budget; }
    ~Scope() { currentBudget() = previous_; }
    Scope (const Scope&) = delete;
    Scope& operator= (const Scope&) = delete;

private:
    Budget previous_;
};
} // namespace arbitgpu::previewbudget
