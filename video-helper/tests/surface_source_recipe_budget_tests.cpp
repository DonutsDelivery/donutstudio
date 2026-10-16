// Restricted SurfacePbr source budget and Metal pool arithmetic, on Linux.
//
// The Metal backend builds its null-vertex SurfacePbr shader through
// surfacesource::prepare, and the stage ceiling there is the hand-derived
// Plan::maximumStages. On 13 Oct the closed-glass status defines grew the glass
// transport recipe from 170 to 293 stages, past the old 312-stage total. Every
// Metal fixture preparation then refused with "Surface source setup exceeds
// admission", and the refusal, retried each frame, drained Sokol's Metal id
// pool until the video helper crashed. The OpenGL backend builds the same text
// as std::string and never runs this ledger, so no Linux test saw it.
//
// These cases run the header-only parts of restrictedSurfaceSourceRecipe through
// the same ledger and compare each with its StageBudget. The vertex, final
// fragment, environment and spectral parts live in backend_sokol.mm or need the
// environment image header, so they count at their documented budgets.

#include "../src/gpu_backend/fixture_smooth_glass.h"
#include "../src/gpu_backend/fixture_analytic_sky.h"
#include "../src/gpu_backend/metal_resource_pools.h"

#include <cstddef>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{
namespace ss = arbitgpu::surfacesource;
namespace pools = arbitgpu::metalpools;
using Budget = ss::StageBudget;

int checks = 0, failures = 0;

void check (bool condition, const std::string& text)
{
    ++checks;
    if (condition) return;
    ++failures;
    std::fprintf (stderr, "FAIL: %s\n", text.c_str());
}

// Stages one recipe part records in the census pass of a real prepare().
template<class Recipe> std::size_t stages (Recipe&& recipe, const char* name)
{
    try
    {
        const auto archive = ss::prepare ([&] { return std::make_pair (recipe(), ss::Source ("")); });
        return archive->plan.count;
    }
    catch (const std::exception& exception)
    {
        check (false, std::string (name) + " refused: " + exception.what());
        return 0;
    }
}

void recipeBudgets()
{
    using namespace arbitgpu;
    const auto opaque = stages ([] { return opaquevisibility::metalShaderRecipe<ss::Source>(); },
                                "opaque visibility");
    const auto sky = stages ([] { return analyticsky::shaderRecipe<ss::Source> (true); }, "analytic sky");
    const auto query = stages ([] { return smoothglass::queryShaderRecipe<ss::Source> (true); }, "glass query");
    const auto transport = stages ([] {
        return smoothglass::transportShaderRecipe<ss::Source> (true, true, false); }, "glass transport");
    const auto renamed = query >= opaque ? query - opaque : query;

    check (opaque > 0 && opaque <= Budget::opaque,
           "opaque visibility uses " + std::to_string (opaque) + " stages, budget "
               + std::to_string (Budget::opaque));
    check (sky > 0 && sky <= Budget::sky,
           "analytic sky uses " + std::to_string (sky) + " stages, budget " + std::to_string (Budget::sky));
    check (query >= opaque && renamed <= Budget::renamedQuery,
           "glass query renaming uses " + std::to_string (renamed) + " stages, budget "
               + std::to_string (Budget::renamedQuery));
    check (transport > 0 && transport <= Budget::transport,
           "glass transport uses " + std::to_string (transport) + " stages, budget "
               + std::to_string (Budget::transport));

    const auto total = Budget::nullVertex + Budget::finalFragment + 3 * opaque + sky + Budget::environment
        + 2 * renamed + Budget::spectral + transport;
    check (total <= ss::Plan::maximumStages,
           "restricted recipe needs " + std::to_string (total) + " stages, admission allows "
               + std::to_string (ss::Plan::maximumStages));

    // The header-only modules also fit one ledger together, as in the backend.
    const auto modules = stages ([] {
        return opaquevisibility::metalShaderRecipe<ss::Source>() + analyticsky::shaderRecipe<ss::Source> (true)
            + smoothglass::queryShaderRecipe<ss::Source> (true)
            + smoothglass::transportShaderRecipe<ss::Source> (true, true, false); }, "combined modules");
    check (modules == opaque + sky + query + transport + 3,
           "combined modules use " + std::to_string (modules) + " stages");

    std::printf ("surface source stages: opaque %zu, sky %zu, query %zu, transport %zu, total %zu of %zu\n",
                 opaque, sky, renamed, transport, total, ss::Plan::maximumStages);
}

// A recipe one stage past the ceiling refuses by name in the census pass,
// before the writer pass allocates its arena.
void overBudgetRefusal()
{
    bool refused = false, writerVisited = false;
    std::string message;
    try
    {
        ss::prepare ([&] {
            writerVisited = writerVisited || ss::Context::current->arena != nullptr;
            ss::Source text ("a");
            for (std::size_t stage = 0; stage <= ss::Plan::maximumStages; ++stage)
                text += ss::Source ("b");
            return std::make_pair (text, ss::Source (""));
        });
    }
    catch (const std::length_error& exception)
    {
        refused = true;
        message = exception.what();
    }
    check (refused && ! writerVisited && message == "Surface source setup exceeds admission",
           "a recipe past Plan::maximumStages must refuse in census, got '" + message + "'");
}

void metalPoolArithmetic()
{
    using Scene = HarmonicMIDI::grid::Visual3DScene;
    const auto textures = static_cast<int> (Scene::kMaxTextures);
    // Fallback slot plus both roles of every texture, and the environment image.
    const int images = 1 + 2 * textures + 1, samplers = 1 + 2 * textures;
    constexpr int concurrentPreparations = 4;
    check (concurrentPreparations * images <= pools::kImagePool,
           "image pool cannot hold four maximum-texture fixture preparations");
    check (concurrentPreparations * images <= pools::kViewPool,
           "view pool cannot hold four maximum-texture fixture preparations");
    check (concurrentPreparations * samplers <= pools::kSamplerPool,
           "sampler pool cannot hold four maximum-texture fixture preparations");
    check (pools::kIdPoolSlots == 2 * (2 * 512 + 2 * 512 + 256 + 6 * 128 + 3 * 128 + 1024 + 128),
           "Metal id pool size differs from sokol_gfx _sg_mtl_init_pool()");
    check (pools::fixturePreparationIds (Scene::kMaxTextures) <= pools::kMinimumFreeIds,
           "a maximum-texture fixture preparation needs more ids than the headroom guard keeps");
    check (2 * pools::kMinimumFreeIds < pools::kIdPoolSlots / 2,
           "the headroom guard reserves too much of the Metal id pool");
}
} // namespace

int main()
{
    recipeBudgets();
    overBudgetRefusal();
    metalPoolArithmetic();
    std::printf ("surface source recipe budget: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
