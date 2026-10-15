#include "../src/model_payload_transport.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace
{
int failures = 0;

void check(bool condition, const char* message)
{
    if (condition) return;
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

visualanimationimport::ExactContentAssetKey key(std::string digest =
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
{
    return { "asset-one", 7, std::move(digest), "model/gltf-binary", 3 };
}

std::string base64Byte(std::uint8_t value)
{
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.push_back(alphabet[value >> 2u]);
    encoded.push_back(alphabet[(value & 0x03u) << 4u]);
    encoded += "==";
    return encoded;
}
}

int main()
{
    using namespace videohelper::modelpayload;
    Store store;

    auto oversized = key();
    oversized.sourceByteSize = visualmodelassetpayload::kMaxPayloadBytes + 1u;
    check(store.begin("oversized", oversized).failure == Failure::PayloadLimitExceeded
          && store.pendingTransferCount() == 0 && store.pendingByteCount() == 0,
          "payload bounds are checked before transfer allocation");

    check(static_cast<bool>(store.begin("transfer-one", key())),
          "exact transfer begins");
    check(store.begin("transfer-one", key()).failure == Failure::DuplicateTransfer,
          "duplicate transfer identity is rejected");
    check(store.appendBase64("transfer-one", 1, "YWJj").failure
              == Failure::NonSequentialChunk,
          "chunks must be sequential");
    check(store.appendBase64("transfer-one", 0, "YWJ!").failure
              == Failure::InvalidChunkEncoding,
          "non-canonical base64 is rejected");
    check(store.appendBase64("transfer-one", 0, "YWJj").receivedBytes == 3,
          "bounded chunk appends exact bytes");
    const auto first = store.commit("transfer-one");
    check(first && first.payload && first.payload->bytes().size() == 3
              && first.payload->bytes()[0] == 'a'
              && store.pendingTransferCount() == 0
              && store.residentPayloadCount() == 1
              && store.residentByteCount() == 3,
          "commit validates digest and publishes immutable bytes");
    check(store.resolvePreview(key()) == first.payload
              && store.resolveExport(key()) == first.payload,
          "preview and export resolve the same immutable payload owner");

    check(static_cast<bool>(store.begin("transfer-two", key())),
          "duplicate exact content transfer begins");
    check(static_cast<bool>(store.appendBase64("transfer-two", 0, "YWJj")),
          "duplicate exact content bytes append");
    const auto second = store.commit("transfer-two");
    check(second && second.payload == first.payload
              && store.residentPayloadCount() == 1
              && store.residentByteCount() == 3,
          "duplicate exact content reuses one resident owner");

    auto incorrectDigest = key(std::string(64, '0'));
    check(static_cast<bool>(store.begin("bad-digest", incorrectDigest)),
          "mismatched digest transfer begins");
    check(static_cast<bool>(store.appendBase64("bad-digest", 0, "YWJj")),
          "mismatched digest bytes append");
    check(store.commit("bad-digest").failure == Failure::DigestMismatch
              && store.pendingTransferCount() == 0
              && store.residentPayloadCount() == 1,
          "digest mismatch fails closed without resident publication");

    check(static_cast<bool>(store.begin("incomplete", key())),
          "incomplete transfer begins");
    check(store.commit("incomplete").failure == Failure::PayloadSizeMismatch
              && store.pendingTransferCount() == 1,
          "incomplete commit retains an abortable transfer");
    check(store.abort("incomplete") && store.pendingTransferCount() == 0,
          "abort releases pending ownership");

    for (std::size_t index = 0;
         index < visualmodelassetpayload::kMaxResidentPayloads * 2u;
         ++index)
    {
        const auto byte = static_cast<std::uint8_t>(index);
        videohelper::Sha256 digest;
        digest.update(&byte, 1);
        const visualanimationimport::ExactContentAssetKey candidate {
            "inspected-asset-" + std::to_string(index), 1, digest.finishHex(),
            "model/gltf-binary", 1
        };
        const auto transferId = "inspection-" + std::to_string(index);
        check(static_cast<bool>(store.begin(transferId, candidate))
                  && static_cast<bool>(store.appendBase64(
                      transferId, 0, base64Byte(byte)))
                  && static_cast<bool>(store.commit(transferId)),
              "unreferenced inspected payload admits under bounded eviction");
    }
    check(store.residentPayloadCount() == visualmodelassetpayload::kMaxResidentPayloads
              && store.resolvePreview(key()) == first.payload,
          "bounded eviction preserves an actively owned preview payload");

    store.reset();
    check(store.pendingTransferCount() == 0 && store.pendingByteCount() == 0
              && store.residentPayloadCount() == 0 && store.residentByteCount() == 0,
          "reset releases pending and resident payloads");

    if (failures != 0)
        return EXIT_FAILURE;
    std::cout << "model-payload-transport: PASS\n";
    return EXIT_SUCCESS;
}
