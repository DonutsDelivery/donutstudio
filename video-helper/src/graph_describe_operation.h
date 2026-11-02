#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

namespace videowire {
// graph_describe is a diagnostic query. A retained scene and its lowered plan travel
// in one operation payload as hex, 264 bytes per vertex on the open sea, which no
// client can take. Above this size the reply says how big the payload is and which
// bytes it holds, and a before and after comparison still sees any change.
inline constexpr std::size_t kDescribedOperationPayloadBytes = 64 * 1024;

// Sixteen hex digits, because a 64-bit number loses low bits in JSON readers that
// hold numbers as doubles, and JUCE and JavaScript both sit on this reply's route.
inline std::string describedPayloadDigest(const std::string& payload) {
  std::uint64_t digest=14695981039346656037ull;
  for (const unsigned char byte : payload) {
    digest ^= byte;
    digest *= 1099511628211ull;
  }
  std::string text(16,'0');
  for (int at=15; at>=0; --at, digest>>=4) text[static_cast<std::size_t>(at)]="0123456789abcdef"[digest&15];
  return text;
}

inline void describeOperationPayload(nlohmann::json& operation, const std::string& payload) {
  if (payload.size()<=kDescribedOperationPayloadBytes) {
    operation["payloadXml"]=payload;
    return;
  }
  operation["payloadXml"]="";
  operation["payloadOmitted"]=true;
  operation["payloadBytes"]=payload.size();
  operation["payloadDigest"]=describedPayloadDigest(payload);
}
} // namespace videowire
