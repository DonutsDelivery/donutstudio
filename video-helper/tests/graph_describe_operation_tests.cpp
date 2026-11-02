// graph_describe keeps small operation payloads and reports large ones by size and digest.
#include "graph_describe_operation.h"

#include <cstdio>

namespace {
int failures=0;
void check(bool condition,const char* what) {
  if (condition) return;
  ++failures; std::fprintf(stderr,"FAILED: %s\n",what);
}
nlohmann::json described(const std::string& payload) {
  nlohmann::json operation={{"nodeId",7},{"kind","visual.3d.render"}};
  videowire::describeOperationPayload(operation,payload);
  return operation;
}
}

int main() {
  using videowire::kDescribedOperationPayloadBytes;
  const std::string limit(kDescribedOperationPayloadBytes,'a');
  const auto kept=described(limit);
  check(kept["payloadXml"]==limit,"a payload at the limit stays whole");
  check(!kept.contains("payloadOmitted"),"a kept payload is not marked omitted");
  check(described("<NodeParams seed=\"17\"/>")["payloadXml"]=="<NodeParams seed=\"17\"/>","a node's XML stays whole");

  auto large=limit+"b";
  const auto omitted=described(large);
  check(omitted["payloadXml"]=="","a payload past the limit is left out");
  check(omitted["payloadOmitted"]==true,"a left-out payload says so");
  check(omitted["payloadBytes"]==large.size(),"a left-out payload reports its size");
  check(omitted["kind"]=="visual.3d.render" && omitted["nodeId"]==7,"the operation keeps its identity");
  check(omitted.dump().size()<512,"the reply is small");

  const auto digest=omitted["payloadDigest"];
  check(digest.is_string() && digest.get<std::string>().size()==16,"the digest is a fixed-width string");
  check(digest.is_string() && digest.get<std::string>().find_first_not_of("0123456789abcdef")==std::string::npos,"the digest is hex");
  check(videowire::describedPayloadDigest("")=="cbf29ce484222325","the empty digest is the FNV-1a offset basis");
  check(videowire::describedPayloadDigest("a")=="af63dc4c8601ec8c","the digest matches FNV-1a");
  // These differ only below the 53 bits a double holds, and a number would merge them.
  const auto asDouble=[](std::uint64_t value) { return static_cast<double>(value); };
  check(asDouble(0xcbf29ce484222325ull)==asDouble(0xcbf29ce484222326ull),"a double loses the low bits");
  check(videowire::describedPayloadDigest("")!="cbf29ce484222326","the string keeps the low bits");

  auto changed=large; changed[10]='c';
  check(described(changed)["payloadDigest"]!=omitted["payloadDigest"],"a changed payload changes the digest");
  check(described(large)["payloadDigest"]==omitted["payloadDigest"],"the same payload keeps its digest");

  if (failures==0) std::puts("graph describe operation: ok");
  return failures==0 ? 0 : 1;
}
