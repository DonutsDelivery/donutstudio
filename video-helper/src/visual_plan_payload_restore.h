#pragma once

#include "../../shared/VisualPlanPayloadDelta.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

// Puts the whole payload text back into a viewport_set_timeline request, so everything after this
// sees what the app compiled. A refusal leaves the store as it was, since nothing was installed.
inline bool restoreVisualPlanPayloads (nlohmann::json& params, visualplanpayloaddelta::Store& store, std::string& error)
{
    visualplanpayloaddelta::Store::Payloads kept;
    const auto plans = params.find ("visualLayerPlans");
    if (plans != params.end() && plans->is_array())
        for (auto& plan : *plans)
        {
            const auto operations = plan.find ("operations");
            if (operations == plan.end() || ! operations->is_array())
                continue;
            for (auto& operation : *operations)
            {
                if (operation.contains ("payloadId"))
                {
                    kept[operation["payloadId"].get<std::uint64_t>()] = operation.value ("payloadXml", std::string {});
                    operation.erase ("payloadId");
                    continue;
                }
                const auto delta = operation.find ("payloadDelta");
                if (delta == operation.end())
                    continue;
                const auto* base = store.find (delta->value ("baseId", std::uint64_t { 0 }));
                if (base == nullptr || ! delta->contains ("pieces") || ! (*delta)["pieces"].is_array())
                {
                    error = visualplanpayloaddelta::kBaseNotHeldError;
                    return false;
                }
                std::vector<visualplanpayloaddelta::Piece> pieces;
                for (const auto& wire : (*delta)["pieces"])
                {
                    visualplanpayloaddelta::Piece piece;
                    if (wire.is_string())
                    {
                        piece.literal = wire.get_ref<const std::string&>();
                        piece.length = piece.literal.size();
                    }
                    else if (wire.is_array() && wire.size() == 2)
                    {
                        piece.fromBase = true;
                        piece.offset = wire[0].get<std::size_t>();
                        piece.length = wire[1].get<std::size_t>();
                    }
                    else
                    {
                        error = visualplanpayloaddelta::kBaseNotHeldError;
                        return false;
                    }
                    pieces.push_back (piece);
                }
                std::string text;
                if (! visualplanpayloaddelta::applyPieces (*base, pieces, delta->value ("size", std::size_t { 0 }), text))
                {
                    error = visualplanpayloaddelta::kBaseNotHeldError;
                    return false;
                }
                kept[delta->value ("id", std::uint64_t { 0 })] = text;
                operation["payloadXml"] = std::move (text);
                operation.erase ("payloadDelta");
            }
        }
    store.replace (std::move (kept));
    return true;
}
