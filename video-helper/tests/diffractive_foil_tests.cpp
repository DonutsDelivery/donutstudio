#include "../src/diffractive_foil_admission.h"
#include "support/diffraction_reference_oracle.h"
#include "DiffractionMaterialPresets.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace
{
int failures = 0;
int checks = 0;

void check(bool condition, const char* message)
{
    ++checks;
    if (!condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

diffractivefoil::Description makeFoil()
{
    diffractivefoil::Description foil;
    foil.physicalBsdf = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    foil.physicalBsdf.roughness = {};
    foil.grooveField = {{
        { { 1.0f, 0.0f }, 800.0f, 1.0f },
        { { 0.0f, 1.0f }, 1600.0f, 0.5f },
        { { 1.0f, 0.0f }, 1200.0f, 0.75f },
        { { 0.70710677f, 0.70710677f }, 2000.0f, 0.25f }
    }};
    foil.physicalBsdf.geometry.directionUv
        = foil.grooveField.front().reciprocalDirectionUv;
    foil.physicalBsdf.geometry.grooveSpacingNanometres
        = foil.grooveField.front().grooveSpacingNanometres;
    foil.workBudget.maximumEvaluations = 4;
    return foil;
}
} // namespace

int main()
{
    using namespace diffractionmaterial;
    using namespace diffractivefoil;

    std::string error;
    auto source = makeFoil();
    auto admitted = admit(source, error);
    check(admitted && error.empty(),
          "a bounded 2x2 physical groove field is admitted");
    if (!admitted)
        return EXIT_FAILURE;

    const auto digest = admitted->structuralDigest();
    source.grooveField[0].grooveSpacingNanometres = 9000.0f;
    check(admitted->description().grooveField[0].grooveSpacingNanometres == 800.0f
              && admitted->structuralDigest() == digest,
          "admission owns an immutable groove field snapshot");
    check(admitted->physicalOrderEvaluationsPerPoint() == 72
              && admitted->maximumPhysicalOrderEvaluations() == 4u * 72u,
          "the admitted work cost counts and bounds physical order evaluations");

    const auto left = evaluateLocalPhysicalBsdf(*admitted, { 0.0f, 0.0f }, 0, error);
    const auto right = evaluateLocalPhysicalBsdf(*admitted, { 1.0f, 0.0f }, 1, error);
    const auto centre = evaluateLocalPhysicalBsdf(*admitted, { 0.5f, 0.5f }, 2, error);
    check(left && right && centre && error.empty(),
          "bounded UV samples lower to the exact admitted physical BSDF");
    if (!left || !right || !centre)
        return EXIT_FAILURE;

    check(left->physicalBsdf.description().geometry.directionUv[0] == 1.0f
              && left->physicalBsdf.description().geometry.directionUv[1] == 0.0f
              && left->physicalBsdf.description().geometry.grooveSpacingNanometres == 800.0f
              && right->physicalBsdf.description().geometry.directionUv[0] == 0.0f
              && right->physicalBsdf.description().geometry.directionUv[1] == 1.0f
              && right->physicalBsdf.description().geometry.grooveSpacingNanometres == 1600.0f,
          "spatial field samples change physical groove orientation and period");
    check(std::abs(centre->physicalBsdf.description().geometry.grooveSpacingNanometres
                   - 1400.0f) < 1.0e-4f
              && std::abs(centre->diffractionCoverage - 0.625f) < 1.0e-6f,
          "period and diffraction mask use fixed-cost bilinear interpolation");

    reference::EvaluationInput input;
    reference::EvaluationFailure failure = reference::EvaluationFailure::None;
    const auto leftResult = reference::evaluate(left->physicalBsdf, input, failure);
    const auto rightResult = reference::evaluate(right->physicalBsdf, input, failure);
    check(leftResult && rightResult && failure == reference::EvaluationFailure::None,
          "local foil samples reuse the physical scalar diffraction oracle");
    if (!leftResult || !rightResult)
        return EXIT_FAILURE;

    const auto findOrder = [](const reference::EvaluationResult& result, int order)
        -> const reference::OrderEvent*
    {
        for (std::size_t index = 0; index < result.eventCount; ++index)
            if (result.events[index].wavelengthIndex == 0
                && result.events[index].order == order)
                return &result.events[index];
        return nullptr;
    };
    const auto* leftOrder = findOrder(*leftResult, 1);
    const auto* rightOrder = findOrder(*rightResult, 1);
    check(leftOrder != nullptr && rightOrder != nullptr
              && std::abs(leftOrder->outgoingDirection.x)
                   > std::abs(leftOrder->outgoingDirection.y)
              && std::abs(rightOrder->outgoingDirection.y)
                   > std::abs(rightOrder->outgoingDirection.x)
              && std::abs(leftOrder->outgoingDirection.x)
                   > std::abs(rightOrder->outgoingDirection.y),
          "orientation rotates the physical order and shorter period increases its deflection");

    const auto coveredResolved = rightResult->resolvedReflectedEnergy
        * right->diffractionCoverage;
    const auto coveredIncident = rightResult->incidentEnergy
        * right->diffractionCoverage;
    check(coveredResolved >= 0.0
              && coveredResolved <= coveredIncident + 1.0e-10
              && rightResult->resolvedReflectedEnergy
                   <= rightResult->incidentEnergy + 1.0e-10
              && rightResult->substrateReflectedEnergy
                   <= rightResult->incidentEnergy + 1.0e-10,
          "the physical BSDF and bounded mask keep foil energy below incident energy");

    check(!evaluateLocalPhysicalBsdf(*admitted, { 0.5f, 0.5f }, 4, error)
              && error == "diffractive foil deterministic work budget is exhausted",
          "evaluation fails closed when the deterministic work budget is exhausted");
    check(!evaluateLocalPhysicalBsdf(*admitted, { -0.1f, 0.5f }, 0, error)
              && error == "diffractive foil material UV is outside the normalized domain",
          "UVs outside the bounded field fail closed");

    auto malformed = makeFoil();
    malformed.grooveField[1].diffractionCoverage = 1.1f;
    check(!admit(malformed, error)
              && error == "diffractive foil groove field sample is out of bounds",
          "mask values above one fail admission");
    malformed = makeFoil();
    malformed.grooveField[1].reciprocalDirectionUv = { -1.0f, 0.0f };
    check(!admit(malformed, error)
              && error == "diffractive foil groove field exceeds the bounded orientation span",
          "orientation interpolation that can cancel fails admission");

    check(requireProductRoute(ProductUse::CpuReferenceValidation,
                              Backend::CpuReference, error)
              && error.empty(),
          "CPU reference evaluation has an explicit validation-only route");
    check(!requireProductRoute(ProductUse::Preview, Backend::CpuReference, error)
              && error == "diffractive foil production route requires a native spatial groove-field executor",
          "CPU reference behavior cannot become a production fallback");
    check(requireProductRoute(ProductUse::Export, Backend::OpenGl, error)
              && error.empty(),
          "OpenGL admits the bounded spatial groove-field executor");
    check(requireProductRoute(ProductUse::Preview, Backend::Metal, error)
              && error.empty(),
          "Metal admits the same bounded spatial groove-field executor");

    std::cout << "diffractive foil: " << checks - failures << '/' << checks
              << " checks passed\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
