#pragma once

#include "../../shared/CuratedShaderTransitionContract.h"
#include "../../shared/ShaderCatalogManifest.h"
#include "sha256.h"

#include <map>
#include <memory>
#include <string>
#include <utility>

namespace videowire
{
struct AdmittedCuratedShaderTransition
{
    shadertransition::Payload payload;
};

// One helper-owned, immutable execution operation. The exact admitted source
// bytes and catalog controls travel together; frame producers are stable graph
// identities, never paths, texture handles, grants, or inferred trust labels.
struct CuratedShaderTransitionOperation
{
    int nodeId = 0;
    int fromNodeId = 0;
    int toNodeId = 0;
    shadertransition::Payload payload;
    std::map<std::string, double> parameterValues;

    double evaluatedProgress() const
    {
        return shadertransition::evaluateProgress(
            payload.progress, payload.direction, payload.easing);
    }
};

using ImmutableCuratedShaderTransition =
    std::shared_ptr<const CuratedShaderTransitionOperation>;

inline bool admitCuratedShaderTransitionPayload(const std::string& encoded,
                                                AdmittedCuratedShaderTransition& admitted,
                                                std::string& error)
{
    admitted = {};
    shadertransition::Payload payload;
    if (!shadertransition::parse(encoded, payload, error)) return false;

    if (videohelper::sha256Text(payload.source) != payload.sourceSha256)
    {
        error = "shader transition source hash does not match its exact source bytes";
        return false;
    }
    const auto* catalog = shadercatalog::find(
        payload.catalogPackId, payload.catalogProgramId);
    if (catalog == nullptr || catalog->kind != "isf"
        || catalog->sourceSha256 != payload.sourceSha256)
    {
        error = "shader transition pack, program, and source digest are not an exact curated catalog entry";
        return false;
    }
    admitted.payload = std::move(payload);
    return true;
}
} // namespace videowire
