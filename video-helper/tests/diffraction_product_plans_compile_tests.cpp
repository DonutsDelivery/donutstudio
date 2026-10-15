#include "../../shared/DiffractionProductPlans.h"

#include <type_traits>
#include <utility>

static_assert(std::is_same_v<
    decltype(diffractionmaterial::makePreviewRequest(
        std::declval<const diffractionmaterial::ImmutableDiffractionScenePlan&>())),
    diffractionmaterial::DiffractionProductRenderRequest>);

int main()
{
    const auto plan = diffractionmaterial::makeDiffractionReferenceScene(
        diffractionmaterial::ReferenceSceneKind::LinearGrating);
    const auto preview = diffractionmaterial::makePreviewRequest(plan);
    const auto exportRequest = diffractionmaterial::makeExportRequest(plan);
    return preview.plan != nullptr
            && exportRequest.plan != nullptr
            && !preview.productDigest.empty()
            && preview.productDigest == exportRequest.productDigest
        ? 0 : 1;
}