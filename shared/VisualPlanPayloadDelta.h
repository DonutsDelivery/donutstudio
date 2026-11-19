#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Sends a large plan payload as the blocks that changed since the payload the helper already holds.
//
// A scene payload writes the plan's revisions before the scene text, so no two edits send equal
// bytes even when the mesh is untouched. Blocks at the same offset still match, because the
// vertices and indices sit at fixed offsets. When the revision gains a digit every later block
// shifts, nothing matches, and that one request carries the whole text.
//
// This belongs to viewport_set_timeline. Export and every other request carry the whole text.
namespace visualplanpayloaddelta
{
inline constexpr std::size_t kMinimumPayloadBytes = std::size_t { 1 } << 20;
inline constexpr std::size_t kBlockBytes = std::size_t { 1 } << 20;
// The sender matches on this text and sends the whole payloads again.
inline constexpr const char* kBaseNotHeldError = "visual plan payload base is not held";

struct Piece
{
    bool fromBase = false;
    // Into the base, for a piece taken from the base.
    std::size_t offset = 0;
    std::size_t length = 0;
    // The bytes, for a piece carried on the wire.
    std::string_view literal;
};

// Empty when no block of the base can be reused, and the caller sends the whole payload.
inline std::vector<Piece> makePieces (std::string_view base, std::string_view payload)
{
    std::vector<Piece> pieces;
    bool reused = false;
    for (std::size_t position = 0; position < payload.size();)
    {
        auto end = std::min (position + kBlockBytes, payload.size());
        // JSON carries a literal as text, so a block must not end inside a UTF-8 sequence.
        while (end < payload.size() && (static_cast<unsigned char> (payload[end]) & 0xC0) == 0x80)
            --end;
        const auto length = end - position;
        const bool same = end <= base.size()
            && std::memcmp (base.data() + position, payload.data() + position, length) == 0;
        reused = reused || same;
        if (! pieces.empty() && pieces.back().fromBase == same)
        {
            pieces.back().length += length;
            if (! same)
                pieces.back().literal = payload.substr (pieces.back().offset, pieces.back().length);
        }
        else
        {
            Piece piece;
            piece.fromBase = same;
            piece.offset = position;
            piece.length = length;
            if (! same)
                piece.literal = payload.substr (position, length);
            pieces.push_back (piece);
        }
        position = end;
    }
    if (! reused)
        pieces.clear();
    return pieces;
}

inline bool applyPieces (std::string_view base, const std::vector<Piece>& pieces,
                         std::size_t size, std::string& output)
{
    std::size_t total = 0;
    for (const auto& piece : pieces)
    {
        if (piece.fromBase && (piece.offset > base.size() || piece.length > base.size() - piece.offset))
            return false;
        if (! piece.fromBase && piece.literal.size() != piece.length)
            return false;
        total += piece.length;
    }
    if (total != size)
        return false;
    output.clear();
    output.reserve (size);
    for (const auto& piece : pieces)
        if (piece.fromBase)
            output.append (base.data() + piece.offset, piece.length);
        else
            output.append (piece.literal.data(), piece.literal.size());
    return true;
}

// The helper's side: the payloads the app may name as a base. It holds what the last request named
// and nothing else, so its size follows the live plans. A helper restart starts it empty, and the
// app recovers from the refusal by sending whole payloads.
class Store
{
public:
    using Payloads = std::map<std::uint64_t, std::string>;

    const std::string* find (std::uint64_t id) const
    {
        const auto found = payloads_.find (id);
        return found != payloads_.end() ? &found->second : nullptr;
    }

    void replace (Payloads payloads) { payloads_ = std::move (payloads); }
    std::size_t size() const noexcept { return payloads_.size(); }

private:
    Payloads payloads_;
};
} // namespace visualplanpayloaddelta
