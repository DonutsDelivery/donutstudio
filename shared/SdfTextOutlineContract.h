#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace videowire
{
using SdfTextResourceId = std::uint64_t;
inline constexpr std::uint32_t kSdfTextOutlineVersion = 1;
inline constexpr std::uint32_t kSdfTextShapingRevision = 1;
inline constexpr std::uint32_t kSdfTextFlatteningRevision = 1;
inline constexpr double kSdfTextFlatteningToleranceEm = 1.0 / 4096.0;
inline constexpr std::size_t kSdfTextMaximumUtf8Bytes = 256;
inline constexpr std::size_t kSdfTextMaximumGlyphs = 128;
inline constexpr std::size_t kSdfTextMaximumSegments = 16384;
inline constexpr std::size_t kSdfTextMaximumPayloadBytes = 1024 * 1024;
inline constexpr SdfTextResourceId kSdfTextMaximumStableId = 9007199254740991ull;
inline constexpr double kSdfTextMaximumCoordinate = 1000000.0;
inline constexpr double kSdfTextMinimumDepth = 0.000001;
inline constexpr double kSdfTextMinimumSegmentLength = 1.0e-12;

struct SdfTextGlyph final
{
    std::uint32_t glyphId = 0; // zero is the missing-glyph refusal
    std::uint32_t utf8Cluster = 0; // byte offset into exact source text
    double x = 0.0, y = 0.0, advanceX = 0.0, advanceY = 0.0;
};

struct SdfTextSegment final
{
    double x0 = 0.0, y0 = 0.0, x1 = 0.0, y1 = 0.0;
};

struct SdfTextContour final
{
    std::uint32_t glyphIndex = 0;
    std::uint32_t firstSegment = 0;
    std::uint32_t segmentCount = 0;
    std::int8_t winding = 1; // +1 CCW, -1 CW, in the scene XY plane
};

// Producer data only. Admission copies this into immutable storage. Source
// identity names embedded BinaryData bytes, never an OS font or raster mask.
// Shaping revision 1 means embedded JUCE GlyphArrangement and glyph Path
// extraction, with no font fallback. Exact glyph/cluster/advance outputs are
// retained, so consumers never repeat shaping. Changed shaping needs a revision.
// XY segments include glyph placement and emSize, in right-handed scene units.
// The producer flattens at kSdfTextFlatteningToleranceEm before emSize scaling.
// Each contour starts at its lexicographically smallest XY vertex. Contours
// retain glyph/path order; this order is part of the saved shaping identity.
struct SdfTextOutline final
{
    SdfTextResourceId stableId = 0;
    std::uint32_t version = kSdfTextOutlineVersion;
    std::string sourceText;
    std::string sourceTextSha256;
    std::string embeddedFontToken;
    std::string embeddedFontSha256;
    std::uint32_t shapingRevision = kSdfTextShapingRevision;
    std::uint32_t flatteningRevision = kSdfTextFlatteningRevision;
    double emSize = 1.0;
    std::vector<SdfTextGlyph> glyphs;
    std::vector<SdfTextContour> contours;
    std::vector<SdfTextSegment> segments;
};

namespace sdftextdetail
{
// Same SHA-256 algorithm as the existing helper identity implementation. Kept
// private here so the shared wire contract has no helper/runtime dependency.
struct Sha256
{
    std::uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    std::uint8_t buf[64] {};
    std::uint64_t total = 0;
    std::size_t fill = 0;

    static std::uint32_t rotr (std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block (const std::uint8_t* p)
    {
        static const std::uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
            0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
            0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
            0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
            0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
            0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
            0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
            0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
            0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (std::uint32_t) p[i * 4] << 24 | (std::uint32_t) p[i * 4 + 1] << 16
                 | (std::uint32_t) p[i * 4 + 2] << 8 | (std::uint32_t) p[i * 4 + 3];
        for (int i = 16; i < 64; ++i)
        {
            const auto s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
            const auto s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        std::uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7];
        for (int i=0;i<64;++i) {
            const auto s1=rotr(e,6)^rotr(e,11)^rotr(e,25), ch=(e&f)^(~e&g);
            const auto t1=z+s1+ch+k[i]+w[i], s0=rotr(a,2)^rotr(a,13)^rotr(a,22);
            const auto t2=s0+((a&b)^(a&c)^(b&c)); z=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=z;
    }
    void update (const void* data, std::size_t n)
    {
        auto* p = static_cast<const std::uint8_t*>(data); total += n;
        while (n) { const auto take=std::min(n,sizeof(buf)-fill); std::memcpy(buf+fill,p,take);
            fill+=take;p+=take;n-=take;if(fill==sizeof(buf)){block(buf);fill=0;} }
    }
    std::string finishHex()
    {
        const std::uint64_t bits=total*8; const std::uint8_t pad=0x80, zero=0; update(&pad,1);
        while (fill != 56)
            update (&zero, 1);
        std::uint8_t len[8];
        for (int i = 0; i < 8; ++i)
            len[i] = (std::uint8_t) (bits >> (56 - i * 8));
        update (len, 8);
        constexpr char digits[] = "0123456789abcdef";
        std::string out;
        out.reserve (64);
        for (const auto word : h)
            for (int shift = 28; shift >= 0; shift -= 4)
                out.push_back (digits[(word >> shift) & 15]);
        return out;
    }
};

inline std::string sha256 (std::string_view bytes)
{
    Sha256 hash;
    hash.update (bytes.data(), bytes.size());
    return hash.finishHex();
}

inline void u8 (std::string& bytes, std::uint8_t value) { bytes.push_back (static_cast<char> (value)); }
inline void u32 (std::string& bytes, std::uint32_t value)
{
    for (unsigned shift = 0; shift < 32; shift += 8) u8 (bytes, static_cast<std::uint8_t> (value >> shift));
}
inline void u64 (std::string& bytes, std::uint64_t value)
{
    for (unsigned shift = 0; shift < 64; shift += 8) u8 (bytes, static_cast<std::uint8_t> (value >> shift));
}
inline void number (std::string& bytes, double value)
{
    static_assert (sizeof (double) == 8 && std::numeric_limits<double>::is_iec559);
    std::uint64_t bits = 0;
    std::memcpy (&bits, &value, 8);
    u64 (bytes, bits);
}
inline void string (std::string& bytes, std::string_view value)
{
    u32 (bytes, static_cast<std::uint32_t> (value.size()));
    bytes.append (value.data(), value.size());
}

struct Reader final
{
    std::string_view bytes;
    std::size_t offset = 0;
    bool integer (std::uint64_t& value, std::size_t count)
    {
        if (count > bytes.size() - offset) return false;
        value = 0;
        for (std::size_t i = 0; i < count; ++i)
            value |= static_cast<std::uint64_t> (static_cast<unsigned char> (bytes[offset++])) << (i * 8);
        return true;
    }
    bool u32 (std::uint32_t& value)
    {
        std::uint64_t wide = 0;
        if (! integer (wide, 4)) return false;
        value = static_cast<std::uint32_t> (wide);
        return true;
    }
    bool number (double& value)
    {
        std::uint64_t bits = 0;
        if (! integer (bits, 8)) return false;
        std::memcpy (&value, &bits, 8);
        return true;
    }
    bool string (std::string& value, std::size_t maximum)
    {
        std::uint32_t count = 0;
        if (! u32 (count) || count > maximum || count > bytes.size() - offset) return false;
        value.assign (bytes.data() + offset, count);
        offset += count;
        return true;
    }
};

inline bool finiteCanonical (double value)
{
    return std::isfinite (value) && std::abs (value) <= kSdfTextMaximumCoordinate
        && (value != 0.0 || ! std::signbit (value));
}

inline bool validUtf8 (std::string_view text)
{
    if (text.empty() || text.size() > kSdfTextMaximumUtf8Bytes) return false;
    for (std::size_t i = 0; i < text.size();)
    {
        const auto first = static_cast<unsigned char> (text[i++]);
        if (first == 0) return false;
        if (first < 0x80) continue;
        unsigned extra = 0;
        std::uint32_t code = 0, minimum = 0;
        if (first >= 0xc2 && first <= 0xdf) { extra = 1; code = first & 31; minimum = 0x80; }
        else if (first >= 0xe0 && first <= 0xef) { extra = 2; code = first & 15; minimum = 0x800; }
        else if (first >= 0xf0 && first <= 0xf4) { extra = 3; code = first & 7; minimum = 0x10000; }
        else return false;
        if (extra > text.size() - i) return false;
        while (extra--)
        {
            const auto next = static_cast<unsigned char> (text[i++]);
            if ((next & 0xc0) != 0x80) return false;
            code = (code << 6) | (next & 63);
        }
        if (code < minimum || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) return false;
    }
    return true;
}
} // namespace sdftextdetail

inline std::string sdfTextSourceSha256 (std::string_view sourceText)
{
    return sdftextdetail::validUtf8 (sourceText) ? sdftextdetail::sha256 (sourceText) : std::string {};
}

inline std::string_view sdfTextEmbeddedFontSha256 (std::string_view token)
{
    struct Font { std::string_view token, sha256; };
    // Exact checked-in OFL font bytes. Asset changes need an explicit revision.
    constexpr Font fonts[] {
        { "IBMPlexSansBold_ttf", "9e6c74a889a700d707613d24548fe4ffa6bc59559a0689d2cf9e133bdcdafb2f" },
        { "IBMPlexSansRegular_ttf", "975dcda37d80f038dcd143c22e33ca2d97a0cc5a929aace1c749153b0fe1afa5" },
        { "IBMPlexSansSemiBold_ttf", "a20caf8286023a6a7a85e40b1d2a4ae9fc3e3b1f9eda8f4c542dd4986af67bb1" },
        { "InterBold_ttf", "288316099b1e0a47a4716d159098005eef7c0066921f34e3200393dbdb01947f" },
        { "InterRegular_ttf", "40d692fce188e4471e2b3cba937be967878f631ad3ebbbdcd587687c7ebe0c82" },
        { "InterSemiBold_ttf", "78a843fade9d4612a5567302fb595b56976eb5fcebf4fea5a5912d638bafcde3" },
        { "ManropeBold_ttf", "2da33eb378d59e6314ed7afdfa837cdcb60e41ac8b1f5d3c4909471b95fcf7d9" },
        { "ManropeRegular_ttf", "2d9a9960fd191a7f1d9060768818074dd2b76ba84a64a35efd2c22bf39030903" },
        { "ManropeSemiBold_ttf", "e80a2c37e47ec09bff5a9960257bf9001e87d5fb7bf35719ef4958763a0c70ac" },
        { "NunitoBold_ttf", "9a580061c1993a96965a02550ba408092145b700ae8af4fa192e5c9ce6f28b76" },
        { "NunitoRegular_ttf", "75e7adfd04e8c4c2219a65bc224a9ce41928ee7d17efadd66c1fdefdc55292d5" },
        { "NunitoSemiBold_ttf", "cfb5a7cb2937b04fb546e861afa1d2ec4418e1ff98d264fe57f9611492714469" }
    };
    for (const auto& font : fonts) if (font.token == token) return font.sha256;
    return {};
}

inline std::size_t sdfTextOutlinePayloadBytes (const SdfTextOutline& outline) noexcept
{
    // Includes the resource ID and every count/string prefix on the wire.
    return 8 + 4 + 4 * 4 + outline.sourceText.size() + outline.sourceTextSha256.size()
        + outline.embeddedFontToken.size() + outline.embeddedFontSha256.size()
        + 4 + 4 + 8 + 4 * 3 + outline.glyphs.size() * 40
        + outline.contours.size() * 13 + outline.segments.size() * 32;
}

inline std::string sdfTextOutlineIdentityBytes (const SdfTextOutline& outline)
{
    using namespace sdftextdetail;
    // Bounds precede allocation, including calls made on untrusted descriptors.
    if (outline.sourceText.size() > kSdfTextMaximumUtf8Bytes
        || outline.sourceTextSha256.size() > 64 || outline.embeddedFontToken.size() > 64
        || outline.embeddedFontSha256.size() > 64 || outline.glyphs.size() > kSdfTextMaximumGlyphs
        || outline.contours.size() > kSdfTextMaximumSegments / 3
        || outline.segments.size() > kSdfTextMaximumSegments
        || sdfTextOutlinePayloadBytes (outline) > kSdfTextMaximumPayloadBytes) return {};
    std::string bytes;
    bytes.reserve (sdfTextOutlinePayloadBytes (outline) - 8);
    u32 (bytes, outline.version);
    string (bytes, outline.sourceText); string (bytes, outline.sourceTextSha256);
    string (bytes, outline.embeddedFontToken); string (bytes, outline.embeddedFontSha256);
    u32 (bytes, outline.shapingRevision); u32 (bytes, outline.flatteningRevision);
    number (bytes, outline.emSize);
    u32 (bytes, static_cast<std::uint32_t> (outline.glyphs.size()));
    for (const auto& glyph : outline.glyphs)
    {
        u32 (bytes, glyph.glyphId); u32 (bytes, glyph.utf8Cluster);
        number (bytes, glyph.x); number (bytes, glyph.y);
        number (bytes, glyph.advanceX); number (bytes, glyph.advanceY);
    }
    u32 (bytes, static_cast<std::uint32_t> (outline.contours.size()));
    for (const auto& contour : outline.contours)
    {
        u32 (bytes, contour.glyphIndex); u32 (bytes, contour.firstSegment);
        u32 (bytes, contour.segmentCount); u8 (bytes, static_cast<std::uint8_t> (contour.winding));
    }
    u32 (bytes, static_cast<std::uint32_t> (outline.segments.size()));
    for (const auto& segment : outline.segments)
    {
        number (bytes, segment.x0); number (bytes, segment.y0);
        number (bytes, segment.x1); number (bytes, segment.y1);
    }
    return bytes;
}

inline SdfTextResourceId sdfTextOutlineStableId (const SdfTextOutline& outline)
{
    const auto bytes = sdfTextOutlineIdentityBytes (outline);
    if (bytes.empty()) return 0;
    // Full SHA-256 remains available as identity; the typed reference is its
    // first 53 bits in big-endian order, exact in the existing JSON carrier.
    const auto digest = sdftextdetail::sha256 (bytes);
    std::uint64_t id = 0;
    for (std::size_t i = 0; i < 14; ++i)
        id = (id << 4) | static_cast<std::uint64_t> (digest[i] <= '9' ? digest[i] - '0' : digest[i] - 'a' + 10);
    id >>= 3;
    return id == 0 ? 1 : id;
}

inline bool validateSdfTextOutline (const SdfTextOutline& outline, std::string& error)
{
    const auto fail = [&] (const char* reason) { error = reason; return false; };
    error.clear();
    if (outline.version != kSdfTextOutlineVersion || outline.shapingRevision != kSdfTextShapingRevision
        || outline.flatteningRevision != kSdfTextFlatteningRevision)
        return fail ("SDF text outline algorithm revision is unsupported");
    if (! sdftextdetail::validUtf8 (outline.sourceText)
        || outline.sourceTextSha256 != sdftextdetail::sha256 (outline.sourceText))
        return fail ("SDF text source UTF-8 or SHA-256 is invalid");
    const auto fontHash = sdfTextEmbeddedFontSha256 (outline.embeddedFontToken);
    if (fontHash.empty() || outline.embeddedFontSha256 != fontHash)
        return fail ("SDF text embedded font identity is unsupported");
    if (! sdftextdetail::finiteCanonical (outline.emSize) || outline.emSize < kSdfTextMinimumDepth
        || outline.glyphs.empty() || outline.glyphs.size() > kSdfTextMaximumGlyphs
        || outline.contours.empty() || outline.contours.size() > kSdfTextMaximumSegments / 3
        || outline.segments.empty() || outline.segments.size() > kSdfTextMaximumSegments
        || sdfTextOutlinePayloadBytes (outline) > kSdfTextMaximumPayloadBytes)
        return fail ("SDF text outline capacity or em size is invalid");
    for (const auto& glyph : outline.glyphs)
        if (glyph.glyphId == 0 || glyph.utf8Cluster >= outline.sourceText.size()
            || (static_cast<unsigned char> (outline.sourceText[glyph.utf8Cluster]) & 0xc0) == 0x80
            || ! sdftextdetail::finiteCanonical (glyph.x) || ! sdftextdetail::finiteCanonical (glyph.y)
            || ! sdftextdetail::finiteCanonical (glyph.advanceX) || ! sdftextdetail::finiteCanonical (glyph.advanceY))
            return fail ("SDF text glyph identity or placement is invalid");
    std::size_t nextSegment = 0;
    std::uint32_t previousGlyph = 0;
    for (const auto& contour : outline.contours)
    {
        if (contour.glyphIndex >= outline.glyphs.size() || contour.glyphIndex < previousGlyph
            || contour.firstSegment != nextSegment || contour.segmentCount < 3
            || contour.segmentCount > outline.segments.size() - nextSegment
            || (contour.winding != 1 && contour.winding != -1))
            return fail ("SDF text contour range or winding is invalid");
        previousGlyph = contour.glyphIndex;
        const auto& first = outline.segments[nextSegment];
        double twiceArea = 0.0;
        for (std::size_t i = 0; i < contour.segmentCount; ++i)
        {
            const auto& segment = outline.segments[nextSegment + i];
            const auto& next = outline.segments[nextSegment + (i + 1) % contour.segmentCount];
            if (! sdftextdetail::finiteCanonical (segment.x0) || ! sdftextdetail::finiteCanonical (segment.y0)
                || ! sdftextdetail::finiteCanonical (segment.x1) || ! sdftextdetail::finiteCanonical (segment.y1)
                || std::hypot (segment.x1 - segment.x0, segment.y1 - segment.y0) < kSdfTextMinimumSegmentLength
                || segment.x1 != next.x0 || segment.y1 != next.y0
                || segment.x0 < first.x0 || (segment.x0 == first.x0 && segment.y0 < first.y0))
                return fail ("SDF text contour is open, degenerate or noncanonical");
            twiceArea += segment.x0 * segment.y1 - segment.x1 * segment.y0;
        }
        if (! std::isfinite (twiceArea) || std::abs (twiceArea) < 1.0e-12
            || (twiceArea > 0.0 ? 1 : -1) != contour.winding)
            return fail ("SDF text contour area disagrees with winding or is degenerate");
        nextSegment += contour.segmentCount;
    }
    if (nextSegment != outline.segments.size()) return fail ("SDF text outline has unowned segments");
    if (outline.stableId == 0 || outline.stableId > kSdfTextMaximumStableId
        || outline.stableId != sdfTextOutlineStableId (outline))
        return fail ("SDF text outline stable ID disagrees with its exact identity");
    return true;
}

// Structural/source identity admission does not prove a producer extracted
// these contours from the named font. Authoring must build and independently
// verify that association; native rendering capability is admitted separately.
class AdmittedSdfTextOutline final
{
public:
    AdmittedSdfTextOutline (const AdmittedSdfTextOutline&) = default;
    AdmittedSdfTextOutline (AdmittedSdfTextOutline&&) noexcept = default;
    AdmittedSdfTextOutline& operator= (const AdmittedSdfTextOutline&) = default;
    AdmittedSdfTextOutline& operator= (AdmittedSdfTextOutline&&) noexcept = default;
    const SdfTextOutline& outline() const noexcept { return outline_; }
    const std::string& identitySha256() const noexcept { return identitySha256_; }
private:
    friend std::optional<AdmittedSdfTextOutline> admitSdfTextOutline (const SdfTextOutline&, std::string&);
    explicit AdmittedSdfTextOutline (const SdfTextOutline& source)
        : outline_ (source), identitySha256_ (sdftextdetail::sha256 (sdfTextOutlineIdentityBytes (source))) {}
    SdfTextOutline outline_;
    std::string identitySha256_;
};

inline std::optional<AdmittedSdfTextOutline> admitSdfTextOutline (const SdfTextOutline& source, std::string& error)
{
    if (! validateSdfTextOutline (source, error)) return std::nullopt;
    return AdmittedSdfTextOutline (source);
}

inline std::string encodeSdfTextOutline (const SdfTextOutline& outline)
{
    std::string error;
    if (! validateSdfTextOutline (outline, error)) return {};
    std::string bytes;
    sdftextdetail::u64 (bytes, outline.stableId);
    bytes += sdfTextOutlineIdentityBytes (outline);
    return bytes;
}

inline bool decodeSdfTextOutline (std::string_view bytes, SdfTextOutline& destination)
{
    if (bytes.size() > kSdfTextMaximumPayloadBytes) return false;
    sdftextdetail::Reader reader { bytes };
    SdfTextOutline candidate;
    std::uint32_t count = 0;
    if (! reader.integer (candidate.stableId, 8) || ! reader.u32 (candidate.version)
        || ! reader.string (candidate.sourceText, kSdfTextMaximumUtf8Bytes)
        || ! reader.string (candidate.sourceTextSha256, 64)
        || ! reader.string (candidate.embeddedFontToken, 64)
        || ! reader.string (candidate.embeddedFontSha256, 64)
        || ! reader.u32 (candidate.shapingRevision) || ! reader.u32 (candidate.flatteningRevision)
        || ! reader.number (candidate.emSize) || ! reader.u32 (count)
        || count > kSdfTextMaximumGlyphs || count > (bytes.size() - reader.offset) / 40) return false;
    candidate.glyphs.resize (count);
    for (auto& glyph : candidate.glyphs)
        if (! reader.u32 (glyph.glyphId) || ! reader.u32 (glyph.utf8Cluster)
            || ! reader.number (glyph.x) || ! reader.number (glyph.y)
            || ! reader.number (glyph.advanceX) || ! reader.number (glyph.advanceY)) return false;
    if (! reader.u32 (count) || count > kSdfTextMaximumSegments / 3
        || count > (bytes.size() - reader.offset) / 13) return false;
    candidate.contours.resize (count);
    for (auto& contour : candidate.contours)
    {
        std::uint64_t winding = 0;
        if (! reader.u32 (contour.glyphIndex) || ! reader.u32 (contour.firstSegment)
            || ! reader.u32 (contour.segmentCount) || ! reader.integer (winding, 1)
            || (winding != 1 && winding != 255)) return false;
        contour.winding = winding == 1 ? 1 : -1;
    }
    if (! reader.u32 (count) || count > kSdfTextMaximumSegments
        || count > (bytes.size() - reader.offset) / 32) return false;
    candidate.segments.resize (count);
    for (auto& segment : candidate.segments)
        if (! reader.number (segment.x0) || ! reader.number (segment.y0)
            || ! reader.number (segment.x1) || ! reader.number (segment.y1)) return false;
    std::string error;
    if (reader.offset != bytes.size() || ! validateSdfTextOutline (candidate, error)) return false;
    destination = std::move (candidate);
    return true;
}

// Work counts are receipts, and a shared graph can expand past 64 bits, so they
// stop at the largest value instead of wrapping or refusing.
template <typename Count>
constexpr Count sdfSaturatingAdd (Count a, Count b) noexcept
{
    return b > std::numeric_limits<Count>::max() - a ? std::numeric_limits<Count>::max() : a + b;
}
template <typename Count>
constexpr Count sdfSaturatingMultiply (Count a, Count b) noexcept
{
    return b != 0 && a > std::numeric_limits<Count>::max() / b ? std::numeric_limits<Count>::max() : a * b;
}

struct SdfTextWorkBudget final
{
    // Counters for receipts. By default only 64-bit overflow stops them.
    std::uint64_t maximumRecordVisits = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t maximumSegmentVisits = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t recordVisits = 0;
    std::uint64_t segmentVisits = 0;
    bool charge (std::uint64_t records, std::uint64_t segments) noexcept
    {
        const auto nextRecords = sdfSaturatingAdd (recordVisits, records);
        const auto nextSegments = sdfSaturatingAdd (segmentVisits, segments);
        if (nextRecords > maximumRecordVisits || nextSegments > maximumSegmentVisits) return false;
        recordVisits = nextRecords;
        segmentVisits = nextSegments;
        return true;
    }
};

// Normative analytic oracle only. Native consumers must implement this field
// from the vector resource; this helper does not produce product pixels.
// Non-zero winding defines inside, including clockwise glyph counters. Half-
// open Y crossings avoid counting shared vertices twice. The extrusion field
// uses dz=abs(z)-depth/2 and the exact signed extrusion combination
// hypot(max(d2,0),max(dz,0))+min(max(d2,dz),0), including outside corners.
inline std::optional<double> evaluateSdfTextExtrusion (const AdmittedSdfTextOutline& admitted,
    double x, double y, double z, double depth, SdfTextWorkBudget& budget)
{
    const auto& outline = admitted.outline();
    if (! std::isfinite (x) || ! std::isfinite (y) || ! std::isfinite (z)
        || std::abs (x) > kSdfTextMaximumCoordinate || std::abs (y) > kSdfTextMaximumCoordinate
        || std::abs (z) > kSdfTextMaximumCoordinate || ! std::isfinite (depth)
        || depth < kSdfTextMinimumDepth || depth > kSdfTextMaximumCoordinate
        || ! budget.charge (1, outline.segments.size())) return std::nullopt;
    double minimumSquared = std::numeric_limits<double>::infinity();
    int winding = 0;
    for (const auto& segment : outline.segments)
    {
        const auto dx = segment.x1 - segment.x0, dy = segment.y1 - segment.y0;
        const auto px = x - segment.x0, py = y - segment.y0;
        const auto t = std::clamp ((px * dx + py * dy) / (dx * dx + dy * dy), 0.0, 1.0);
        const auto ex = px - t * dx, ey = py - t * dy;
        minimumSquared = std::min (minimumSquared, ex * ex + ey * ey);
        const auto side = dx * py - dy * px;
        if (segment.y0 <= y && segment.y1 > y && side > 0.0) ++winding;
        if (segment.y0 > y && segment.y1 <= y && side < 0.0) --winding;
    }
    const auto distance2D = std::sqrt (minimumSquared) * (winding == 0 ? 1.0 : -1.0);
    const auto distanceZ = std::abs (z) - depth / 2.0;
    const auto value = std::hypot (std::max (distance2D, 0.0), std::max (distanceZ, 0.0))
        + std::min (std::max (distance2D, distanceZ), 0.0);
    return std::isfinite (value) ? std::optional<double> (value) : std::nullopt;
}
} // namespace videowire
