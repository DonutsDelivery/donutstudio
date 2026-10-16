#pragma once

// Dimension-indexed sample values for the path tracer.
//
// The default pattern is a shuffled, Owen-scrambled Sobol sequence built as
// Burley describes in "Practical Hash-based Owen Scrambling", JCGT 9(4), 2020:
// the first four Sobol dimensions, nested uniform scrambling through a
// Laine-Karras style hash, and padding to any number of dimensions by hashing
// the dimension into the seed. Within one 2D request the two values are a
// stratified pair; different requests are decorrelated. Behavioural reference
// (studied, not copied): Blender Cycles intern/cycles/kernel/sample/sobol_burley.h
// and intern/cycles/kernel/sample/pattern.h.
//
// A value depends only on (pixel, sample index, generation, dimension), never
// on call order, thread or tile, so a render is reproducible bit for bit.

#include "accumulator.h"
#include "vec_math.h"

#include <cstdint>

namespace videohelper::pathtrace
{

enum class SamplePattern : std::uint8_t
{
    OwenSobol = 0,
    // Plain PCG streams from PixelRng, one per dimension. Kept as a fallback
    // and as an unstratified reference for tests.
    Independent = 1
};

namespace sampling
{

constexpr std::uint32_t reverseBits (std::uint32_t x) noexcept
{
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    return (x >> 16) | (x << 16);
}

// Generator matrices of the first four Sobol dimensions, one 32-bit column per
// index bit, most significant output bit first. Dimension 0 is the van der
// Corput sequence. Dimensions 1 to 3 use the primitive polynomials and initial
// direction numbers of Joe and Kuo, "Constructing Sobol sequences with better
// two-dimensional projections", SIAM J. Sci. Comput. 30(5), 2008.
struct SobolMatrices
{
    std::uint32_t columns[4][32];
};

constexpr SobolMatrices makeSobolMatrices() noexcept
{
    SobolMatrices out {};
    const std::uint32_t degree[4] { 0, 1, 2, 3 };
    const std::uint32_t polynomial[4] { 0, 0, 1, 1 };
    const std::uint32_t initial[4][3] { { 0, 0, 0 }, { 1, 0, 0 }, { 1, 3, 0 }, { 1, 3, 1 } };
    for (std::uint32_t k = 0; k < 32; ++k)
        out.columns[0][k] = 1u << (31 - k);
    for (std::uint32_t d = 1; d < 4; ++d)
    {
        const std::uint32_t s = degree[d];
        for (std::uint32_t k = 0; k < 32; ++k)
        {
            if (k < s)
            {
                out.columns[d][k] = initial[d][k] << (31 - k);
                continue;
            }
            std::uint32_t value = out.columns[d][k - s] ^ (out.columns[d][k - s] >> s);
            for (std::uint32_t j = 1; j < s; ++j)
                if (((polynomial[d] >> (s - 1 - j)) & 1u) != 0)
                    value ^= out.columns[d][k - j];
            out.columns[d][k] = value;
        }
    }
    return out;
}

inline constexpr SobolMatrices kSobolMatrices = makeSobolMatrices();

inline std::uint32_t sobol (std::uint32_t index, std::uint32_t dimension) noexcept
{
    std::uint32_t result = 0;
    for (std::uint32_t bit = 0; index != 0; ++bit, index >>= 1)
        if ((index & 1u) != 0)
            result ^= kSobolMatrices.columns[dimension & 3u][bit];
    return result;
}

// Burley's Laine-Karras style permutation: every step only lets lower bits
// change higher bits, so applied to bit-reversed input it is an Owen scramble.
inline std::uint32_t laineKarras (std::uint32_t x, std::uint32_t seed) noexcept
{
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return x;
}

inline std::uint32_t nestedUniformScramble (std::uint32_t x, std::uint32_t seed) noexcept
{
    return reverseBits (laineKarras (reverseBits (x), seed));
}

inline std::uint32_t hashCombine (std::uint32_t seed, std::uint32_t value) noexcept
{
    return pcgHash (seed ^ pcgHash (value + 0x9e3779b9u));
}

// [0, 1) with 24 bits of resolution.
inline float toUnitFloat (std::uint32_t bits) noexcept
{
    return static_cast<float> (bits >> 8) * (1.0f / 16777216.0f);
}

} // namespace sampling

// Sample values of one path: one (pixel, sample index, generation) triple.
// get1D and get2D take a dimension number the caller assigns; a 1D and a 2D
// request with the same number are independent of each other.
class PathSampler
{
public:
    PathSampler (std::uint32_t pixelIndex, std::uint32_t sampleIndex, std::uint64_t generation,
                 SamplePattern pattern = SamplePattern::OwenSobol) noexcept
        : pixel_ (pixelIndex), sample_ (sampleIndex), generation_ (generation), pattern_ (pattern)
    {
        std::uint32_t h = pcgHash (static_cast<std::uint32_t> (generation >> 32) ^ 0x2545f491u);
        h = pcgHash (h ^ static_cast<std::uint32_t> (generation));
        seed_ = pcgHash (h ^ pixelIndex);
    }

    std::uint32_t sampleIndex() const noexcept { return sample_; }
    SamplePattern pattern() const noexcept { return pattern_; }

    float get1D (std::uint32_t dimension) const noexcept
    {
        if (pattern_ == SamplePattern::Independent)
            return independent (dimension * 2u + 0x40000000u).nextFloat();
        const std::uint32_t seed = sampling::hashCombine (seed_ ^ 0x68bc21ebu, dimension);
        const std::uint32_t index = sampling::nestedUniformScramble (sample_, pcgHash (seed ^ 0x02e5be93u));
        const std::uint32_t x = sampling::nestedUniformScramble (sampling::sobol (index, 0), pcgHash (seed ^ 0x967a889bu));
        return sampling::toUnitFloat (x);
    }

    Vec2 get2D (std::uint32_t dimension) const noexcept
    {
        if (pattern_ == SamplePattern::Independent)
        {
            PixelRng rng = independent (dimension * 2u + 1u);
            const float x = rng.nextFloat();
            return { x, rng.nextFloat() };
        }
        const std::uint32_t seed = sampling::hashCombine (seed_ ^ 0x8f1bbcdcu, dimension);
        const std::uint32_t index = sampling::nestedUniformScramble (sample_, pcgHash (seed ^ 0x5851f42du));
        const std::uint32_t x = sampling::nestedUniformScramble (sampling::sobol (index, 0), pcgHash (seed ^ 0x14057b7eu));
        const std::uint32_t y = sampling::nestedUniformScramble (sampling::sobol (index, 1), pcgHash (seed ^ 0xb5297a4du));
        return { sampling::toUnitFloat (x), sampling::toUnitFloat (y) };
    }

private:
    PixelRng independent (std::uint32_t stream) const noexcept
    {
        return PixelRng (pixel_, sample_, generation_ ^ (static_cast<std::uint64_t> (stream) * 0x9e3779b97f4a7c15ull));
    }

    std::uint32_t pixel_ = 0;
    std::uint32_t sample_ = 0;
    std::uint64_t generation_ = 0;
    std::uint32_t seed_ = 0;
    SamplePattern pattern_ = SamplePattern::OwenSobol;
};

} // namespace videohelper::pathtrace
