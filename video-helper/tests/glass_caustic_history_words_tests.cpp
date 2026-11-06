// Smooth caustic reflection history is 128 bits on the CPU, OpenGL and Metal.
// The OpenGL and Metal mirrors copy each shader's bit-set and deposit packing,
// so equal words here mean both backends write the same deposit history for
// the same branch. The GPU image comparison runs later in live acceptance.
// Past 128 interfaces the CPU history grows word by word; the tail checks that.
// Grating incident tails reuse the same 16-bit halves for words 2 and up.

#include "../src/gpu_backend/glass_caustic_history.h"

#include <array>
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace arbitgpu::glasscaustics;

namespace
{
int failures = 0;

void check(bool ok, const char* what)
{
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

// A fixed, irregular reflect/transmit pattern over the interfaces.
bool reflectsAt(std::uint32_t i) { return ((i * 2654435761u) >> 7) % 3 != 0 || i == 119 || i == 64 || i == 96; }
}

int main()
{
    constexpr std::uint32_t interfaces = 120;
    static_assert(interfaces < kHistoryBits);

    CausticHistory cpu;
    OpenGlHistoryState gl;
    MetalHistoryState metal;
    for (std::uint32_t i = 0; i < interfaces; ++i)
    {
        cpu = cpu.with(i, reflectsAt(i));
        if (reflectsAt(i)) { gl = openGlHistoryReflect(gl, i); metal = metalHistoryReflect(metal, i); }
    }

    const auto texels = openGlHistoryTexels(gl);
    const auto metalWords = metalHistoryDepositWords(metal);
    const auto fromGl = openGlHistoryDecode(texels);
    const auto fromMetal = metalHistoryDecode(metalWords);
    check(fromGl == fromMetal, "OpenGL and Metal encoders write identical words for a 120-interface history");
    check(fromGl == cpu && fromMetal == cpu, "both GPU encoders match the CPU trace history");

    bool every = true;
    for (std::uint32_t i = 0; i < interfaces; ++i) every &= fromGl.test(i) == reflectsAt(i);
    check(every, "each of the 120 interfaces decodes to its own reflect bit");
    check(cpu.words[2] != 0 && cpu.words[3] != 0, "interfaces 64 to 119 reach the two wide words");
    check(cpu.test(119) && cpu.test(96) && cpu.test(64), "bits 64, 96 and 119 survive the round trip");

    bool exact = true;
    for (const auto& texel : texels)
        for (float half : texel) exact &= half >= 0 && half <= 65535 && float(std::uint32_t(half)) == half;
    check(exact, "OpenGL texels hold exact 16-bit integer halves");

    check(cpu.clearFrom(interfaces) && !cpu.clearFrom(119) && !cpu.clearFrom(64),
          "clearFrom refuses bits at or above the interface count");
    check(CausticHistory {}.clearFrom(0) && !CausticHistory {}.with(127).clearFrom(127), "clearFrom covers the top bit");

    const std::uint64_t narrow = 0x8000000100000005ull;
    const CausticHistory widened = narrow;
    check(widened.low64() == narrow && widened.high64() == 0 && widened == CausticHistory::fromWords(5, 0x80000001u, 0, 0),
          "64-interface histories widen in place");

    // A single high bit lands in the same word and texel on both backends.
    for (std::uint32_t bit : {0u, 31u, 32u, 63u, 64u, 95u, 96u, 127u})
    {
        const auto g = openGlHistoryDecode(openGlHistoryTexels(openGlHistoryReflect({}, bit)));
        const auto m = metalHistoryDecode(metalHistoryDepositWords(metalHistoryReflect({}, bit)));
        if (!(g == m && g == CausticHistory {}.with(bit))) { check(false, "single bit placement matches"); return 1; }
    }
    check(true, "single bits at every word edge land in the same word on both backends");

    // The CPU history grows past the four GPU words instead of dropping bits.
    constexpr std::uint32_t deep = 300;
    CausticHistory wide;
    for (std::uint32_t i = 0; i < deep; ++i) wide = wide.with(i, reflectsAt(i));
    bool deepEvery = true;
    for (std::uint32_t i = 0; i < deep; ++i) deepEvery &= wide.test(i) == reflectsAt(i);
    check(deepEvery && wide.words.size() == historyWordsFor(deep) && historyWordsFor(deep) == 10,
          "a 300-interface history keeps every bit in ten words");
    check(wide.clearFrom(deep) && !wide.clearFrom(deep - 1) == reflectsAt(deep - 1) && !wide.test(deep),
          "clearFrom and test read the grown words");
    check(historyWordsFor(0) == kHistoryWords && historyWordsFor(128) == 4 && historyWordsFor(129) == 5,
          "word count never drops below the four inline GPU words");

    // Trailing zero words never change equality or order.
    auto padded = CausticHistory {}.with(5);
    padded.words.resize(12, 0u);
    check(padded == CausticHistory {}.with(5) && !(padded < CausticHistory {}.with(5)) && !(CausticHistory {}.with(5) < padded),
          "trailing zero words compare equal");
    check(CausticHistory {}.with(127) < CausticHistory {}.with(200) && !(CausticHistory {}.with(200) < CausticHistory {}.with(127)),
          "a higher reflected interface orders after a lower one");

    // Extra words returned by a kernel append after the inline four.
    const std::uint32_t extra[] = {wide.words[4], wide.words[5], wide.words[6], wide.words[7], wide.words[8], wide.words[9]};
    const auto rebuilt = CausticHistory::fromWords(wide.words[0], wide.words[1], wide.words[2], wide.words[3]).withExtraWords(extra, 6);
    check(rebuilt == wide, "inline words plus extra words rebuild the full history");

    // Both kernels' tail code writes the same 300-interface history as the CPU.
    {
        const std::size_t extra = historyWordsFor(deep) - kHistoryWords;
        OpenGlHistoryState glInline;
        MetalHistoryState metalInline;
        std::vector<std::array<float, 4>> glTail((extra + 1) / 2, std::array<float, 4> {});
        std::vector<std::uint32_t> metalTail(extra, 0u);
        for (std::uint32_t i = 0; i < deep; ++i)
        {
            if (!reflectsAt(i)) continue;
            if (i < kHistoryBits) { glInline = openGlHistoryReflect(glInline, i); metalInline = metalHistoryReflect(metalInline, i); }
            openGlHistoryTailReflect(glTail, extra, i);
            metalHistoryTailReflect(metalTail, extra, i);
        }
        std::vector<std::uint32_t> glWords(extra);
        const bool decoded = openGlHistoryTailDecode(glTail.data(), extra, glWords.data());
        const auto fromGlDeep = openGlHistoryDecode(openGlHistoryTexels(glInline)).withExtraWords(glWords.data(), extra);
        const auto fromMetalDeep = metalHistoryDecode(metalHistoryDepositWords(metalInline)).withExtraWords(metalTail.data(), extra);
        check(extra == 6 && glTail.size() == 3 && decoded, "a 300-interface tail is six words in three OpenGL texels");
        check(fromGlDeep == wide && fromMetalDeep == wide, "OpenGL and Metal tails match the CPU history past 128 interfaces");

        // The kernels ignore bits past their tail instead of writing out of bounds.
        auto glShort = glTail;
        openGlHistoryTailReflect(glShort, extra, std::uint32_t(128 + 32 * extra));
        check(glShort == glTail, "a reflection past the tail leaves it unchanged");

        // An odd word count pads the last texel; the decoder refuses a nonzero pad.
        std::vector<std::array<float, 4>> odd(3, std::array<float, 4> {});
        std::uint32_t oddWords[5] {};
        check(openGlHistoryTailDecode(odd.data(), 5, oddWords), "a zero pad half decodes");
        odd[2][2] = 1;
        check(!openGlHistoryTailDecode(odd.data(), 5, oddWords), "a nonzero pad half is refused");
        odd[2][2] = 0;
        odd[0][1] = 0.5f;
        check(!openGlHistoryTailDecode(odd.data(), 5, oddWords), "a fractional half is refused");
    }

    // Multiple-scatter ancestry tokens for interfaces 32 and up share the block
    // after the history words, four tokens per word, on both kernels.
    {
        const std::size_t historyExtra = historyWordsFor(deep) - kHistoryWords, ancestryExtra = (deep - 32 + 3) / 4;
        const std::size_t total = historyExtra + ancestryExtra;
        const auto tokenAt = [](std::uint32_t i) { return std::uint32_t(((i * 40503u) >> 3) % 64) | (reflectsAt(i) ? 0u : 64u); };
        std::vector<std::array<float, 4>> glBlock((total + 1) / 2, std::array<float, 4> {});
        std::vector<std::uint32_t> metalBlock(total, 0u);
        for (std::uint32_t i = 0; i < deep; ++i)
        {
            if (reflectsAt(i)) { openGlHistoryTailReflect(glBlock, historyExtra, i); metalHistoryTailReflect(metalBlock, historyExtra, i); }
            openGlAncestryTailReflect(glBlock, historyExtra, ancestryExtra, i, tokenAt(i));
            metalAncestryTailReflect(metalBlock, historyExtra, ancestryExtra, i, tokenAt(i));
        }
        std::vector<std::uint32_t> glWords(total);
        check(openGlHistoryTailDecode(glBlock.data(), total, glWords.data()) && glWords == metalBlock,
              "OpenGL and Metal write the same history and ancestry tail words");
        const auto tokens = ancestryTailTokens(metalBlock.data(), historyExtra, ancestryExtra);
        bool everyToken = tokens.size() == 4 * ancestryExtra;
        for (std::uint32_t i = 32; i < deep && everyToken; ++i) everyToken = tokens[i - 32] == tokenAt(i);
        for (std::size_t b = deep - 32; b < tokens.size() && everyToken; ++b) everyToken = tokens[b] == 0;
        check(ancestryExtra == 67 && everyToken, "268 ancestry tokens for interfaces 32 to 299 round-trip");
        bool historyKept = true;
        for (std::size_t k = 0; k < historyExtra; ++k) historyKept &= metalBlock[k] == wide.words[kHistoryWords + k];
        check(historyKept, "ancestry words never touch the history words before them");
        auto beyond = metalBlock;
        metalAncestryTailReflect(beyond, historyExtra, ancestryExtra, std::uint32_t(32 + 4 * ancestryExtra), 65);
        metalAncestryTailReflect(beyond, historyExtra, ancestryExtra, 31, 65);
        check(beyond == metalBlock, "tokens below 32 or past the tail stay out of it");
    }

    {
        // Grating incidents: texel 1 holds words 0 and 1, the tail words 2 and up.
        // Both kernels' tails decode to the CPU history of a 300-interface path.
        const std::uint32_t deep = 300;
        const std::size_t extra = historyWordsFor(deep) - kHistoryWords, tailWords = historyWordsFor(deep) - 2;
        OpenGlHistoryState gl;
        MetalHistoryState metal;
        std::vector<std::array<float, 4>> glWalked((extra + 1) / 2, std::array<float, 4>{});
        std::vector<std::uint32_t> metalWalked(extra, 0u);
        CausticHistory cpu;
        for (std::uint32_t i = 0; i < deep; ++i)
        {
            if (!reflectsAt(i)) continue;
            cpu = cpu.with(i);
            gl = openGlHistoryReflect(gl, i);
            metal = metalHistoryReflect(metal, i);
            openGlHistoryTailReflect(glWalked, extra, i);
            metalHistoryTailReflect(metalWalked, extra, i);
        }
        const auto glTail = openGlGratingIncidentTail(gl, glWalked, tailWords);
        const auto metalTail = metalGratingIncidentTail(metal, metalWalked, tailWords);
        check(tailWords == 8 && glTail == metalTail, "OpenGL and Metal write the same grating incident tail");
        CausticHistory decoded;
        const auto low = std::uint64_t(gl.x) | (std::uint64_t(gl.y) << 32);
        check(gratingIncidentHistoryDecode(low, glTail.data(), tailWords, decoded) && decoded == cpu && decoded.clearFrom(deep),
              "a 300-interface grating incident history round-trips through its tail");
        CausticHistory shallow;
        check(gratingIncidentHistoryDecode(low, nullptr, 0, shallow) && shallow == CausticHistory(low),
              "incidents of 64 interfaces or fewer have no tail and keep texel 1 alone");
        auto corrupt = glTail;
        corrupt.back()[3] = 0.5f;
        check(!gratingIncidentHistoryDecode(low, corrupt.data(), tailWords, decoded), "a non-integer tail half is refused");
        check(!gratingIncidentHistoryDecode(low, glTail.data(), 1, decoded), "a one-word tail cannot hold words 2 and 3");

        // Cell transport replays every interface of the incident: inline bytes
        // for the first 32, then texel 1 and the tail, never a 32 cap.
        std::array<std::array<float, 4>, 24> incident{};
        incident[1] = {float(gl.x & 65535u), float(gl.x >> 16), float(gl.y & 65535u), float(gl.y >> 16)};
        for (std::uint32_t i = 0; i < 32; ++i) incident[2 + i / 4][i % 4] = reflectsAt(i) ? 1.0f : 65.0f;
        bool tokens = true;
        for (std::uint32_t i = 0; i < deep; ++i)
            tokens &= gratingCellTokenMirror(incident.data(), glTail.data(), i, deep) == (cpu.test(i) ? 1u : 65u);
        check(tokens && gratingCellTokenMirror(incident.data(), glTail.data(), deep, deep) == 0,
              "cell transport tokens follow all 300 interfaces and end with the endpoint");
        bool metalTokens = true;
        for (std::uint32_t i = 32; i < deep; ++i)
            metalTokens &= gratingCellTokenMirror(incident.data(), metalTail.data(), i, deep)
                == gratingCellTokenMirror(incident.data(), glTail.data(), i, deep);
        check(metalTokens, "OpenGL and Metal cell transport read the same tokens past 32");
    }

    {
        // Camera trace visits carry history words 2 and up in extra lanes,
        // each copied from the parent's record with this branch's reflection.
        // A 300-visit chain needs 10 words: 8 extra words in 2 lanes.
        const std::size_t extraLanes = 2;
        std::vector<std::array<std::uint32_t, 4>> lanes = cameraTraceHistoryLanesMirror(nullptr, 0, 0, extraLanes);
        CausticHistory cpu;
        bool rootZero = lanes == std::vector<std::array<std::uint32_t, 4>>(extraLanes, std::array<std::uint32_t, 4>{});
        for (std::uint32_t i = 0; i < 299; ++i)
        {
            const bool reflected = reflectsAt(i);
            cpu = cpu.with(i, reflected);
            lanes = cameraTraceHistoryLanesMirror(&lanes, i + 1, reflected ? 1u : 65u, extraLanes);
        }
        bool same = true;
        for (std::size_t k = 0; k < 4 * extraLanes; ++k) same &= lanes[k / 4][k % 4] == cpu.word(2 + k);
        check(rootZero && same, "camera trace visit lanes rebuild history words 2 to 9 of a 299-interface chain");
    }

    std::printf("%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
