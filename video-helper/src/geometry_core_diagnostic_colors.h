#pragma once

#include "../../shared/GeometryCore.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace videohelper::geometry {
inline std::uint32_t diagnosticColorSlot(
    std::vector<videowire::geometry::StableId> admittedStableIdentities,
    videowire::geometry::StableId stableIdentity) {
  std::sort(admittedStableIdentities.begin(), admittedStableIdentities.end());
  admittedStableIdentities.erase(
      std::unique(admittedStableIdentities.begin(), admittedStableIdentities.end()),
      admittedStableIdentities.end());
  const auto found = std::lower_bound(admittedStableIdentities.begin(),
                                      admittedStableIdentities.end(), stableIdentity);
  if (found == admittedStableIdentities.end() || *found != stableIdentity)
    return 0;
  const auto rank = static_cast<std::uint32_t>(found - admittedStableIdentities.begin());
  // Admitted instance plans contain far fewer than 2^24 distinct identities.
  // Odd multiplication permutes the 24-bit slots, assigning each identity in
  // this exact bounded plan one unique color independent of input order.
  return ((rank + 1u) * 0x9e3779u) & 0xffffffu;
}
} // namespace videohelper::geometry
