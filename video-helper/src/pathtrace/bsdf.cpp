#include "bsdf.h"
#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace videohelper::pathtrace
{
namespace
{

constexpr float kTwoPi = 6.28318530717958647692f;
// Below this Fresnel reflectance a smooth dielectric never samples reflection.
constexpr float kMinSampledFresnel = 1.0e-12f;

PT_SHARED_ENTRY float clampUnit (float x) noexcept
{
    return std::isfinite (x) ? std::min (1.0f, std::max (0.0f, x)) : 0.0f;
}

PT_SHARED_ENTRY Vec3 clampUnit (Vec3 v) noexcept
{
    return { clampUnit (v.x), clampUnit (v.y), clampUnit (v.z) };
}

PT_SHARED_ENTRY float average (Vec3 v) noexcept
{
    return (v.x + v.y + v.z) * (1.0f / 3.0f);
}

PT_SHARED_ENTRY bool finitePositive (float x) noexcept
{
    return std::isfinite (x) && x > 0.0f;
}

PT_SHARED_ENTRY float pow5 (float x) noexcept
{
    const float x2 = x * x;
    return x2 * x2 * x;
}

// GGX normal distribution for a unit half vector in the local frame.
PT_SHARED_ENTRY float ggxD (Vec3 h, float alpha) noexcept
{
    if (h.z <= 0.0f)
        return 0.0f;
    const float a2 = alpha * alpha;
    // sin^2 + a2 cos^2, from h's tangent part. The usual cos^2 (a2 - 1) + 1 cancels
    // near the peak, and at alpha under 1.7e-4 a2 - 1 rounds to -1 and D to infinity.
    const float d = h.x * h.x + h.y * h.y + a2 * h.z * h.z;
    return a2 / (kPi * d * d);
}

// Smith Lambda of GGX. Grazing directions give infinity, so G becomes 0.
PT_SHARED_ENTRY float smithLambda (float cosTheta, float alpha) noexcept
{
    const float c2 = cosTheta * cosTheta;
    if (! (c2 > 0.0f))
        return std::numeric_limits<float>::infinity();
    const float t2 = std::max (0.0f, 1.0f - c2) / c2;
    return 0.5f * (std::sqrt (1.0f + alpha * alpha * t2) - 1.0f);
}

// Visible-normal sample around wo (wo.z > 0) by the projected-hemisphere
// construction of Heitz 2018. The result is a unit microfacet normal with z > 0.
PT_SHARED_ENTRY Vec3 sampleVisibleNormal (Vec3 wo, float alpha, Vec2 u) noexcept
{
    const Vec3 stretched = normalizeOrZero ({ alpha * wo.x, alpha * wo.y, wo.z });
    const float planar = stretched.x * stretched.x + stretched.y * stretched.y;
    const Vec3 axisA = planar > 1.0e-12f ? Vec3 { -stretched.y, stretched.x, 0.0f } * (1.0f / std::sqrt (planar))
                                         : Vec3 { 1.0f, 0.0f, 0.0f };
    const Vec3 axisB = cross (stretched, axisA);
    const float radius = std::sqrt (u.x);
    const float angle = kTwoPi * u.y;
    const float diskA = radius * std::cos (angle);
    float diskB = radius * std::sin (angle);
    const float blend = 0.5f * (1.0f + stretched.z);
    diskB = (1.0f - blend) * std::sqrt (std::max (0.0f, 1.0f - diskA * diskA)) + blend * diskB;
    const float lift = std::sqrt (std::max (0.0f, 1.0f - diskA * diskA - diskB * diskB));
    const Vec3 onHemisphere = axisA * diskA + axisB * diskB + stretched * lift;
    return normalizeOrZero ({ alpha * onHemisphere.x, alpha * onHemisphere.y, std::max (1.0e-6f, onHemisphere.z) });
}

PT_SHARED_ENTRY Vec3 reflectAbout (Vec3 wo, Vec3 h) noexcept
{
    return h * (2.0f * dot (wo, h)) - wo;
}

// Refracts wo (pointing away, dot(wo, n) > 0) through a surface with normal n.
// eta is the far index over the near index. False on total internal reflection.
PT_SHARED_ENTRY bool refractThrough (Vec3 wo, Vec3 n, float eta, Vec3& wt) noexcept
{
    const float cosI = dot (wo, n);
    const float sin2T = std::max (0.0f, 1.0f - cosI * cosI) / (eta * eta);
    if (sin2T >= 1.0f)
        return false;
    const float cosT = std::sqrt (1.0f - sin2T);
    wt = (-wo) * (1.0f / eta) + n * (cosI / eta - cosT);
    return true;
}

// Cosine-weighted hemisphere through the concentric disk mapping
// (Shirley and Chiu, "A Low Distortion Map Between Disk and Square", 1997).
PT_SHARED_ENTRY Vec3 cosineHemisphere (Vec2 u) noexcept
{
    const float ox = 2.0f * u.x - 1.0f;
    const float oy = 2.0f * u.y - 1.0f;
    if (ox == 0.0f && oy == 0.0f)
        return { 0.0f, 0.0f, 1.0f };
    float r = 0.0f;
    float theta = 0.0f;
    if (std::fabs (ox) > std::fabs (oy))
    {
        r = ox;
        theta = 0.25f * kPi * (oy / ox);
    }
    else
    {
        r = oy;
        theta = 0.5f * kPi - 0.25f * kPi * (ox / oy);
    }
    const float x = r * std::cos (theta);
    const float y = r * std::sin (theta);
    return { x, y, std::sqrt (std::max (0.0f, 1.0f - x * x - y * y)) };
}

constexpr int kGgxTableSize = 32;

// The bilinear lookup of one table at row r0 + tr, column c0 + tc.
PT_SHARED_ENTRY float ggxBilinear (const PT_TABLE_SPACE float* table, int r0, int c0, float tr, float tc) noexcept
{
    const auto at = static_cast<std::size_t> (r0 * kGgxTableSize + c0);
    const float top = table[at] * (1.0f - tc) + table[at + 1] * tc;
    const float bottom = table[at + kGgxTableSize] * (1.0f - tc) + table[at + kGgxTableSize + 1] * tc;
    return top * (1.0f - tr) + bottom * tr;
}

// Bilinear in (sqrt(alpha), cosO) over two kGgxTableSize^2 tables.
PT_SHARED_ENTRY void ggxAlbedoLookup (const PT_TABLE_SPACE float* scaleTable, const PT_TABLE_SPACE float* biasTable, float cosO, float alpha, float& scale,
                      float& bias) noexcept
{
    const float r = std::sqrt (clampUnit (alpha));
    const float fr = r * static_cast<float> (kGgxTableSize - 1);
    const float fc = clampUnit (cosO) * static_cast<float> (kGgxTableSize - 1);
    const int r0 = std::min (static_cast<int> (fr), kGgxTableSize - 2);
    const int c0 = std::min (static_cast<int> (fc), kGgxTableSize - 2);
    const float tr = fr - static_cast<float> (r0);
    const float tc = fc - static_cast<float> (c0);
    scale = ggxBilinear (scaleTable, r0, c0, tr, tc);
    bias = ggxBilinear (biasTable, r0, c0, tr, tc);
}

#if ! defined(__CUDACC__)
// Directional albedo table of the single-scatter GGX reflection lobe, split in
// the Fresnel-independent scale and bias terms. Rows are perceptual roughness
// in [0, 1] (alpha = max(r^2, the opaque floor)), columns cos(theta_o) in [0, 1].
class GgxAlbedoTable
{
public:
    static constexpr int kSize = kGgxTableSize;
    static constexpr int kSamples = 1024;

    GgxAlbedoTable() noexcept
    {
        for (int row = 0; row < kSize; ++row)
        {
            const float r = static_cast<float> (row) / static_cast<float> (kSize - 1);
            const float alpha = std::max (r * r, kMinOpaqueGgxAlpha);
            for (int column = 0; column < kSize; ++column)
            {
                const float c = std::max (static_cast<float> (column) / static_cast<float> (kSize - 1), 1.0e-3f);
                const Vec3 wo { std::sqrt (std::max (0.0f, 1.0f - c * c)), 0.0f, c };
                const float lambdaO = smithLambda (wo.z, alpha);
                double scale = 0.0;
                double bias = 0.0;
                // Hammersley points: stratified first coordinate, radical inverse second.
                for (int i = 0; i < kSamples; ++i)
                {
                    const Vec2 u { (static_cast<float> (i) + 0.5f) / static_cast<float> (kSamples),
                                   sampling::toUnitFloat (sampling::reverseBits (static_cast<std::uint32_t> (i))) };
                    const Vec3 h = sampleVisibleNormal (wo, alpha, u);
                    const float woh = dot (wo, h);
                    if (woh <= 0.0f)
                        continue;
                    const Vec3 wi = reflectAbout (wo, h);
                    if (wi.z <= 0.0f)
                        continue;
                    const float lambdaI = smithLambda (wi.z, alpha);
                    const float ratio = (1.0f + lambdaO) / (1.0f + lambdaO + lambdaI);  // G2 / G1(wo)
                    const float s = pow5 (1.0f - woh);
                    scale += static_cast<double> (ratio * (1.0f - s));
                    bias += static_cast<double> (ratio * s);
                }
                const std::size_t slot = static_cast<std::size_t> (row * kSize + column);
                scale_[slot] = static_cast<float> (scale / kSamples);
                bias_[slot] = static_cast<float> (bias / kSamples);
            }
        }
    }

    const float* scale() const noexcept { return scale_.data(); }
    const float* bias() const noexcept { return bias_.data(); }

private:
    std::array<float, kSize * kSize> scale_ {};
    std::array<float, kSize * kSize> bias_ {};
};

const GgxAlbedoTable& ggxAlbedoTable() noexcept
{
    static const GgxAlbedoTable table;
    return table;
}
#endif

template <class Color>
PT_SHARED_ENTRY constexpr int channelCount() noexcept;
template <>
PT_SHARED_ENTRY constexpr int channelCount<Vec3>() noexcept { return 3; }
template <>
PT_SHARED_ENTRY constexpr int channelCount<Spectrum>() noexcept { return kSpectrumChannels; }

PT_SHARED_ENTRY bool finiteColor (const Vec3& c) noexcept { return isFinite (c); }
PT_SHARED_ENTRY bool finiteColor (const Spectrum& c) noexcept { return c.finite(); }

template <class Color>
PT_SHARED_ENTRY Color schlickColor (const Color& f0, float cosine) noexcept
{
    const float s = pow5 (clampUnit (1.0f - cosine));
    Color out = f0;
    for (int c = 0; c < channelCount<Color>(); ++c)
        out[c] = f0[c] + (1.0f - f0[c]) * s;
    return out;
}

template <class Color>
PT_SHARED_ENTRY Color uniformColor (float x) noexcept;
template <>
PT_SHARED_ENTRY Vec3 uniformColor<Vec3> (float x) noexcept { return { x, x, x }; }
template <>
PT_SHARED_ENTRY Spectrum uniformColor<Spectrum> (float x) noexcept { return Spectrum::filled (x); }

// Keeps a ratio of indices finite, inside [1/3, 3], and away from exactly 1
// where the rough transmission half vector degenerates.
PT_SHARED_ENTRY float sanitizeEta (float eta, bool rough) noexcept
{
    if (! std::isfinite (eta) || ! (eta > 0.0f))
        return 1.5f;
    if (rough && std::fabs (eta - 1.0f) < 1.0e-4f)
        eta = eta >= 1.0f ? 1.0f + 1.0e-4f : 1.0f - 1.0e-4f;
    return eta;
}

PT_SHARED_ENTRY std::uint32_t lobeKindFlags (LobeKind kind) noexcept
{
    switch (kind)
    {
        case LobeKind::Lambert:
            return kBsdfReflection | kBsdfDiffuse;
        case LobeKind::GgxReflection:
            return kBsdfReflection | kBsdfGlossy;
        case LobeKind::SmoothDielectric:
        case LobeKind::ThinSheet:
            return kBsdfReflection | kBsdfTransmission | kBsdfDelta;
        case LobeKind::RoughDielectric:
            return kBsdfReflection | kBsdfTransmission | kBsdfGlossy;
    }
    return 0;
}

template <class Color>
PT_SHARED_ENTRY BasicBsdfEval<Color> evalLobeImpl (const BasicLobe<Color>& lobe, Vec3 wo, Vec3 wi) noexcept
{
    BasicBsdfEval<Color> out;
    if (! (wo.z > 0.0f))
        return out;
    switch (lobe.kind)
    {
        case LobeKind::Lambert:
        {
            if (! (wi.z > 0.0f))
                return out;
            out.pdf = wi.z * kInvPi;
            out.value = lobe.weight * out.pdf;
            return out;
        }
        case LobeKind::GgxReflection:
        {
            if (! (wi.z > 0.0f))
                return out;
            const Vec3 h = normalizeOrZero (wo + wi);
            const float woh = dot (wo, h);
            if (! (h.z > 0.0f) || ! (woh > 0.0f))
                return out;
            const float d = ggxD (h, lobe.alpha);
            const float lambdaO = smithLambda (wo.z, lobe.alpha);
            const float lambdaI = smithLambda (wi.z, lobe.alpha);
            const float g1 = 1.0f / (1.0f + lambdaO);
            const float g2 = 1.0f / (1.0f + lambdaO + lambdaI);
            out.value = lobe.weight * schlickColor (lobe.f0, woh) * (d * g2 / (4.0f * wo.z));
            out.pdf = g1 * d / (4.0f * wo.z);
            return out;
        }
        case LobeKind::SmoothDielectric:
        case LobeKind::ThinSheet:
            return out;
        case LobeKind::RoughDielectric:
        {
            const float alpha = lobe.alpha;
            const float eta = lobe.eta;
            const float lambdaO = smithLambda (wo.z, alpha);
            const float g1 = 1.0f / (1.0f + lambdaO);
            if (wi.z > 0.0f)
            {
                const Vec3 h = normalizeOrZero (wo + wi);
                const float woh = dot (wo, h);
                if (! (h.z > 0.0f) || ! (woh > 0.0f))
                    return out;
                const float f = fresnelDielectric (woh, eta);
                const float d = ggxD (h, alpha);
                const float g2 = 1.0f / (1.0f + lambdaO + smithLambda (wi.z, alpha));
                out.value = lobe.weight * (f * d * g2 / (4.0f * wo.z));
                out.pdf = f * g1 * d / (4.0f * wo.z);
                return out;
            }
            if (wi.z < 0.0f)
            {
                // Generalized half vector of refraction, turned to the wo side.
                Vec3 h = normalizeOrZero (wo + wi * eta);
                if (h.z < 0.0f)
                    h = -h;
                const float woh = dot (wo, h);
                const float wih = dot (wi, h);
                if (! (h.z > 0.0f) || ! (woh > 0.0f) || ! (wih < 0.0f))
                    return out;
                const float t = 1.0f - fresnelDielectric (woh, eta);
                const float root = wih + woh / eta;
                const float denominator = root * root;
                if (! (t > 0.0f) || ! (denominator > 1.0e-12f))
                    return out;
                const float d = ggxD (h, alpha);
                const float g2 = 1.0f / (1.0f + lambdaO + smithLambda (wi.z, alpha));
                const float jacobian = -wih / denominator;  // d(h) / d(wi)
                out.value = lobe.transmissionTint * (t * d * g2 * woh * jacobian / wo.z);
                out.pdf = t * g1 * woh * d / wo.z * jacobian;
                return out;
            }
            return out;
        }
    }
    return out;
}

template <class Color>
PT_SHARED_ENTRY bool sampleLobeImpl (const BasicLobe<Color>& lobe, Vec3 wo, float uChoice, Vec2 u, BasicBsdfSample<Color>& out) noexcept
{
    out = {};
    if (! (wo.z > 0.0f))
        return false;
    switch (lobe.kind)
    {
        case LobeKind::Lambert:
        {
            out.wi = cosineHemisphere (u);
            if (! (out.wi.z > 0.0f))
                return false;
            out.pdf = out.wi.z * kInvPi;
            out.weight = lobe.weight;
            out.flags = kBsdfReflection | kBsdfDiffuse;
            return true;
        }
        case LobeKind::GgxReflection:
        {
            const Vec3 h = sampleVisibleNormal (wo, lobe.alpha, u);
            const float woh = dot (wo, h);
            if (! (woh > 0.0f))
                return false;
            out.wi = reflectAbout (wo, h);
            if (! (out.wi.z > 0.0f))
                return false;
            const float lambdaO = smithLambda (wo.z, lobe.alpha);
            const float lambdaI = smithLambda (out.wi.z, lobe.alpha);
            const float g1 = 1.0f / (1.0f + lambdaO);
            const float ratio = (1.0f + lambdaO) / (1.0f + lambdaO + lambdaI);
            out.pdf = g1 * ggxD (h, lobe.alpha) / (4.0f * wo.z);
            out.weight = lobe.weight * schlickColor (lobe.f0, woh) * ratio;
            out.flags = kBsdfReflection | kBsdfGlossy;
            return out.pdf > 0.0f;
        }
        case LobeKind::SmoothDielectric:
        {
            const float f = fresnelDielectric (wo.z, lobe.eta);
            // A denormal reflectance (index ratio near 1) counts as none, so
            // the channels' Fresnel over f below stays finite.
            if (f >= kMinSampledFresnel && uChoice < f)
            {
                out.wi = { -wo.x, -wo.y, wo.z };
                out.weight = lobe.weight;
                out.pdf = f;
                out.flags = kBsdfReflection | kBsdfDelta;
                if (lobe.dispersive)
                    for (int c = 1; c < channelCount<Color>(); ++c)
                    {
                        // Each channel reflects its own Fresnel share; the
                        // hero's choice probability f is shared.
                        const float ratio = fresnelDielectric (wo.z, lobe.channelEta[static_cast<std::size_t> (c)]) / f;
                        out.weight[c] *= ratio;
                        out.channelPdfRatio[static_cast<std::size_t> (c)] = ratio;
                    }
                return true;
            }
            if (! refractThrough (wo, { 0.0f, 0.0f, 1.0f }, lobe.eta, out.wi) || ! (out.wi.z < 0.0f))
                return false;
            out.weight = lobe.transmissionTint;
            out.pdf = 1.0f - f;
            out.flags = kBsdfTransmission | kBsdfDelta;
            if (lobe.dispersive)
            {
                // Other wavelengths leave in other directions; only the hero
                // follows this one.
                for (int c = 1; c < channelCount<Color>(); ++c)
                {
                    out.weight[c] = 0.0f;
                    out.channelPdfRatio[static_cast<std::size_t> (c)] = 0.0f;
                }
                out.flags |= kBsdfHeroOnly;
            }
            return out.pdf > 0.0f;
        }
        case LobeKind::RoughDielectric:
        {
            const float alpha = lobe.alpha;
            const float eta = lobe.eta;
            const Vec3 h = sampleVisibleNormal (wo, alpha, u);
            const float woh = dot (wo, h);
            if (! (woh > 0.0f))
                return false;
            const float f = fresnelDielectric (woh, eta);
            const float d = ggxD (h, alpha);
            const float lambdaO = smithLambda (wo.z, alpha);
            const float g1 = 1.0f / (1.0f + lambdaO);
            if (uChoice < f)
            {
                out.wi = reflectAbout (wo, h);
                if (! (out.wi.z > 0.0f))
                    return false;
                const float ratio = (1.0f + lambdaO) / (1.0f + lambdaO + smithLambda (out.wi.z, alpha));
                out.pdf = f * g1 * d / (4.0f * wo.z);
                out.weight = lobe.weight * ratio;
                out.flags = kBsdfReflection | kBsdfGlossy;
                return out.pdf > 0.0f;
            }
            if (! refractThrough (wo, h, eta, out.wi) || ! (out.wi.z < 0.0f))
                return false;
            const float wih = dot (out.wi, h);
            const float root = wih + woh / eta;
            const float denominator = root * root;
            if (! (wih < 0.0f) || ! (denominator > 1.0e-12f))
                return false;
            const float ratio = (1.0f + lambdaO) / (1.0f + lambdaO + smithLambda (out.wi.z, alpha));
            out.pdf = (1.0f - f) * g1 * woh * d / wo.z * (-wih / denominator);
            out.weight = lobe.transmissionTint * ratio;
            out.flags = kBsdfTransmission | kBsdfGlossy;
            return out.pdf > 0.0f;
        }
        case LobeKind::ThinSheet:
        {
            const float r = thinSheetReflectance (wo.z, lobe.eta);
            if (uChoice < r)
            {
                out.wi = { -wo.x, -wo.y, wo.z };
                out.weight = lobe.weight;
                out.pdf = r;
                out.flags = kBsdfReflection | kBsdfDelta;
                return true;
            }
            // No thickness, no bending: the ray carries on.
            out.wi = -wo;
            out.weight = lobe.transmissionTint;
            out.pdf = 1.0f - r;
            out.flags = kBsdfTransmission | kBsdfDelta | kBsdfStraight;
            return out.pdf > 0.0f;
        }
    }
    return false;
}

template <class Color, class ToColor>
PT_SHARED_ENTRY void buildBsdf (const Material& material, Vec3 shadingNormal, Vec3 geometricNormal, Vec3 wo,
                const DielectricSetup& setup, const ToColor& toColor, BasicBsdf<Color>& out) noexcept
{
    out.reset (shadingNormal, geometricNormal);
    const Vec3 base = clampUnit (material.baseColor);
    const Color baseColor = toColor (base);
    const float metallic = clampUnit (material.metallic);
    const float roughness = clampUnit (material.roughness);
    const float ior = std::isfinite (material.ior) ? std::min (3.0f, std::max (1.01f, material.ior)) : 1.5f;
    const float cosO = clampUnit (dot (shadingNormal, wo));
    const float opaqueAlpha = std::max (roughness * roughness, float (kMinOpaqueGgxAlpha));
    const float glassWeight = dielectricWeight (material);
    const float transmission = clampUnit (material.transmission);

    if (metallic > 0.0f)
    {
        float scale = 0.0f;
        float bias = 0.0f;
        ggxAlbedoScaleBias (cosO, opaqueAlpha, scale, bias);
        BasicLobe<Color> metal;
        metal.kind = LobeKind::GgxReflection;
        metal.weight = uniformColor<Color> (metallic);
        metal.f0 = baseColor;
        metal.alpha = opaqueAlpha;
        metal.selectWeight = metallic * average (base * scale + Vec3 { bias, bias, bias });
        out.addLobe (metal);
    }

    if (glassWeight > 0.0f && material.transport == DielectricTransport::ThinSheet)
    {
        BasicLobe<Color> sheet;
        sheet.kind = LobeKind::ThinSheet;
        sheet.eta = sanitizeEta (setup.etaD, false);
        sheet.weight = uniformColor<Color> (glassWeight);
        sheet.transmissionTint = baseColor * glassWeight;
        const float r = thinSheetReflectance (cosO, sheet.eta);
        sheet.selectWeight = std::max (glassWeight * (r + (1.0f - r) * average (base)), glassWeight * 1.0e-3f);
        out.addLobe (sheet);
    }
    else if (glassWeight > 0.0f)
    {
        BasicLobe<Color> glass;
        const float alpha = roughness * roughness;
        const bool rough = alpha >= kSmoothDielectricAlpha;
        glass.kind = rough ? LobeKind::RoughDielectric : LobeKind::SmoothDielectric;
        glass.alpha = std::max (alpha, float (kSmoothDielectricAlpha));
        glass.eta = sanitizeEta (setup.eta[0], rough);
        if (setup.dispersive)
        {
            if (rough)
            {
                out.requireHeroWavelength();
            }
            else
            {
                glass.dispersive = true;
                for (std::size_t c = 0; c < glass.channelEta.size(); ++c)
                    glass.channelEta[c] = sanitizeEta (setup.eta[c], false);
            }
        }
        glass.weight = uniformColor<Color> (glassWeight);
        // Beer-Lambert media take their colour inside the volume instead.
        const bool absorbing = std::isfinite (material.absorptionDistance) && material.absorptionDistance > 0.0f;
        const Vec3 tintRgb = absorbing ? Vec3 { 1.0f, 1.0f, 1.0f } : base;
        glass.transmissionTint = (absorbing ? uniformColor<Color> (1.0f) : baseColor) * glassWeight;
        const float f = fresnelDielectric (cosO, sanitizeEta (setup.etaD, false));
        glass.selectWeight = glassWeight * (f + (1.0f - f) * average (tintRgb));
        // A black tint still reflects, so keep the lobe selectable.
        glass.selectWeight = std::max (glass.selectWeight, glassWeight * 1.0e-3f);
        out.addLobe (glass);
    }

    const float opaqueWeight = (1.0f - metallic) * (1.0f - transmission);
    if (opaqueWeight > 0.0f)
    {
        const float r = (ior - 1.0f) / (ior + 1.0f);
        const float f0 = r * r;
        float scale = 0.0f;
        float bias = 0.0f;
        ggxAlbedoScaleBias (cosO, opaqueAlpha, scale, bias);
        const float specularAlbedo = clampUnit (f0 * scale + bias);

        BasicLobe<Color> specular;
        specular.kind = LobeKind::GgxReflection;
        specular.weight = uniformColor<Color> (opaqueWeight);
        specular.f0 = uniformColor<Color> (f0);
        specular.alpha = opaqueAlpha;
        specular.selectWeight = opaqueWeight * specularAlbedo;
        out.addLobe (specular);

        BasicLobe<Color> diffuse;
        diffuse.kind = LobeKind::Lambert;
        const float diffuseScale = opaqueWeight * (1.0f - specularAlbedo);
        diffuse.weight = baseColor * diffuseScale;
        diffuse.selectWeight = average (base * diffuseScale);
        out.addLobe (diffuse);
    }
    out.finalize();
}

// Reflectance spectrum of an RGB colour clamped to [0, 1].
PT_SHARED_ENTRY Spectrum reflectanceSpectrum (Vec3 rgb, const Wavelengths& wavelengths) noexcept
{
    Spectrum s = upsampleRgb (clampUnit (rgb), wavelengths);
    for (float& x : s.v)
        x = std::min (x, 1.0f);
    return s;
}

} // namespace

PT_SHARED_ENTRY float materialCoverage (const Material& material) noexcept
{
    switch (material.alphaMode)
    {
        case MaterialAlphaMode::Opaque:
            return 1.0f;
        case MaterialAlphaMode::Mask:
            return material.opacity >= material.alphaCutoff ? 1.0f : 0.0f;
        case MaterialAlphaMode::Blend:
            return std::isfinite (material.opacity) ? clampUnit (material.opacity) : 1.0f;
    }
    return 1.0f;
}

PT_SHARED_ENTRY float dielectricWeight (const Material& material) noexcept
{
    return (1.0f - clampUnit (material.metallic)) * clampUnit (material.transmission);
}

PT_SHARED_ENTRY bool isMediumBoundary (const Material& material) noexcept
{
    return dielectricWeight (material) > 0.0f && material.transport != DielectricTransport::ThinSheet;
}

PT_SHARED_ENTRY bool isThinSheet (const Material& material) noexcept
{
    return dielectricWeight (material) > 0.0f && material.transport == DielectricTransport::ThinSheet;
}

PT_SHARED_ENTRY float materialCauchyB (const Material& material) noexcept
{
    // Any finite nonnegative B renders as authored, like the admission's.
    if (std::isfinite (material.dispersion) && material.dispersion != 0.0f)
        return std::max (0.0f, material.dispersion);
    const float ior = std::isfinite (material.ior) ? std::min (3.0f, std::max (1.01f, material.ior)) : 1.5f;
    return cauchyBFromAbbe (ior, material.abbe);
}

PT_SHARED_ENTRY bool materialDispersive (const Material& material) noexcept
{
    return isMediumBoundary (material) && materialCauchyB (material) > 0.0f;
}

PT_SHARED_ENTRY float materialIorAt (const Material& material, float nm, bool spectral) noexcept
{
    const float ior = std::isfinite (material.ior) ? std::min (3.0f, std::max (1.01f, material.ior)) : 1.5f;
    if (! spectral)
        return ior;
    const float n = cauchyIor (ior, materialCauchyB (material), nm);
    return std::isfinite (n) ? std::max (1.0f, n) : ior;
}

PT_SHARED_ENTRY Spectrum mediumAbsorption (const Material& material, const Wavelengths& wavelengths) noexcept
{
    Spectrum sigma;
    if (! isMediumBoundary (material) || ! std::isfinite (material.absorptionDistance)
        || ! (material.absorptionDistance > 0.0f))
        return sigma;
    const Spectrum tint = reflectanceSpectrum (material.baseColor, wavelengths);
    const int channels = wavelengths.spectral ? kSpectrumChannels : 3;
    for (int c = 0; c < channels; ++c)
        sigma[c] = -std::log (std::max (tint[c], 1.0e-4f)) / material.absorptionDistance;
    return sigma;
}

PT_SHARED_ENTRY bool materialHasVolume (const Material& material) noexcept
{
    return std::isfinite (material.volumeDensity) && material.volumeDensity > 0.0f;
}

PT_SHARED_ENTRY void volumeCoefficients (const Material& material, const Wavelengths& wavelengths, Spectrum& absorption,
                                         Spectrum& scattering, Spectrum& emission) noexcept
{
    absorption = {};
    scattering = {};
    emission = {};
    if (! materialHasVolume (material))
        return;
    const float density = material.volumeDensity;
    const Vec3 colour = clampUnit (material.volumeColor);
    const Vec3 absorptionColour = clampUnit (material.volumeAbsorptionColor);
    const Vec3 absorbed { (1.0f - colour.x) * (1.0f - absorptionColour.x), (1.0f - colour.y) * (1.0f - absorptionColour.y),
                          (1.0f - colour.z) * (1.0f - absorptionColour.z) };
    scattering = reflectanceSpectrum (colour, wavelengths) * density;
    absorption = reflectanceSpectrum (absorbed, wavelengths) * density;
    const Vec3 e = material.volumeEmission;
    if (std::isfinite (e.x) && std::isfinite (e.y) && std::isfinite (e.z))
        emission = upsampleRgb ({ std::max (0.0f, e.x), std::max (0.0f, e.y), std::max (0.0f, e.z) }, wavelengths);
    const int channels = wavelengths.spectral ? kSpectrumChannels : 3;
    for (int c = channels; c < kSpectrumChannels; ++c)
    {
        absorption[c] = 0.0f;
        scattering[c] = 0.0f;
        emission[c] = 0.0f;
    }
}

PT_SHARED_ENTRY Spectrum straightTransmittance (const Material& material, float cosine, float etaD,
                                const Wavelengths& wavelengths) noexcept
{
    if (! isThinSheet (material))
        return {};
    const float r = thinSheetReflectance (std::fabs (cosine), sanitizeEta (etaD, false));
    return reflectanceSpectrum (material.baseColor, wavelengths) * (dielectricWeight (material) * (1.0f - r));
}

PT_SHARED_ENTRY Frame Frame::fromNormal (Vec3 n) noexcept
{
    Frame frame;
    const float sign = n.z >= 0.0f ? 1.0f : -1.0f;
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    frame.tangent = { 1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x };
    frame.bitangent = { b, sign + n.y * n.y * a, -n.y };
    frame.normal = n;
    return frame;
}

PT_SHARED_ENTRY float fresnelDielectric (float cosI, float eta) noexcept
{
    cosI = clampUnit (cosI);
    const float sin2T = (1.0f - cosI * cosI) / (eta * eta);
    if (sin2T >= 1.0f)
        return 1.0f;
    const float cosT = std::sqrt (1.0f - sin2T);
    const float rs = (cosI - eta * cosT) / (cosI + eta * cosT);
    const float rp = (eta * cosI - cosT) / (eta * cosI + cosT);
    return clampUnit (0.5f * (rs * rs + rp * rp));
}

PT_SHARED_ENTRY float thinSheetReflectance (float cosI, float eta) noexcept
{
    const float r = fresnelDielectric (cosI, eta);
    return clampUnit (2.0f * r / (1.0f + r));
}

PT_SHARED_ENTRY void ggxAlbedoScaleBias (float cosO, float alpha, float& scale, float& bias) noexcept
{
#if defined(__CUDACC__)
    const device::DeviceLookupTables& tables = device::deviceLookupTables();
    ggxAlbedoLookup (tables.ggxScale, tables.ggxBias, cosO, alpha, scale, bias);
#else
    const GgxAlbedoTable& table = ggxAlbedoTable();
    ggxAlbedoLookup (table.scale(), table.bias(), cosO, alpha, scale, bias);
#endif
}

PT_SHARED_ENTRY std::uint32_t lobeFlags (const Lobe& lobe) noexcept { return lobeKindFlags (lobe.kind); }
PT_SHARED_ENTRY std::uint32_t lobeFlags (const SpectralLobe& lobe) noexcept { return lobeKindFlags (lobe.kind); }

PT_SHARED_ENTRY BsdfEval evalLobe (const Lobe& lobe, Vec3 wo, Vec3 wi) noexcept { return evalLobeImpl (lobe, wo, wi); }
PT_SHARED_ENTRY SpectralBsdfEval evalLobe (const SpectralLobe& lobe, Vec3 wo, Vec3 wi) noexcept { return evalLobeImpl (lobe, wo, wi); }

PT_SHARED_ENTRY bool sampleLobe (const Lobe& lobe, Vec3 wo, float uChoice, Vec2 u, BsdfSample& out) noexcept
{
    return sampleLobeImpl (lobe, wo, uChoice, u, out);
}

PT_SHARED_ENTRY bool sampleLobe (const SpectralLobe& lobe, Vec3 wo, float uChoice, Vec2 u, SpectralBsdfSample& out) noexcept
{
    return sampleLobeImpl (lobe, wo, uChoice, u, out);
}

template <class Color>
PT_SHARED_ENTRY void BasicBsdf<Color>::reset (Vec3 shadingNormal, Vec3 geometricNormal) noexcept
{
    frame_ = Frame::fromNormal (shadingNormal);
    geometricNormal_ = geometricNormal;
    count_ = 0;
    nonDelta_ = false;
    heroOnly_ = false;
    for (float& probability : probability_)
        probability = 0.0f;
}

template <class Color>
PT_SHARED_ENTRY bool BasicBsdf<Color>::addLobe (const BasicLobe<Color>& lobe) noexcept
{
    if (count_ >= kMaxLobes || ! finitePositive (lobe.selectWeight))
        return false;
    lobes_[static_cast<std::size_t> (count_)] = lobe;
    ++count_;
    return true;
}

template <class Color>
PT_SHARED_ENTRY void BasicBsdf<Color>::finalize() noexcept
{
    float total = 0.0f;
    for (int i = 0; i < count_; ++i)
        total += lobes_[static_cast<std::size_t> (i)].selectWeight;
    nonDelta_ = false;
    for (int i = 0; i < count_; ++i)
    {
        const auto slot = static_cast<std::size_t> (i);
        probability_[slot] = total > 0.0f ? lobes_[slot].selectWeight / total : 0.0f;
        if ((lobeKindFlags (lobes_[slot].kind) & kBsdfDelta) == 0)
            nonDelta_ = true;
    }
}

// The ray kind (sceneRayVisibility bit) of light reaching a lobe from wiLocal.
PT_SHARED_ENTRY std::uint8_t lobeRayKind (LobeKind kind, Vec3 wiLocal) noexcept
{
    constexpr std::uint8_t diffuse = 4u, glossy = 8u, transmission = 16u;
    if (kind == LobeKind::Lambert)
        return diffuse;
    if (kind == LobeKind::GgxReflection)
        return glossy;
    return wiLocal.z > 0.0f ? glossy : transmission;
}

template <class Color>
PT_SHARED_ENTRY BasicBsdfEval<Color> BasicBsdf<Color>::evalLocal (Vec3 woLocal, Vec3 wiLocal, std::uint8_t rayKinds) const noexcept
{
    BasicBsdfEval<Color> out;
    for (int i = 0; i < count_; ++i)
    {
        const auto slot = static_cast<std::size_t> (i);
        if ((lobeKindFlags (lobes_[slot].kind) & kBsdfDelta) != 0)
            continue;
        const BasicBsdfEval<Color> e = evalLobeImpl (lobes_[slot], woLocal, wiLocal);
        if ((lobeRayKind (lobes_[slot].kind, wiLocal) & rayKinds) != 0)
            out.value = out.value + e.value;
        out.pdf += probability_[slot] * e.pdf;
    }
    return out;
}

template <class Color>
PT_SHARED_ENTRY BasicBsdfEval<Color> BasicBsdf<Color>::eval (Vec3 wo, Vec3 wi) const noexcept
{
    const Vec3 woLocal = frame_.toLocal (wo);
    const Vec3 wiLocal = frame_.toLocal (wi);
    if (! (woLocal.z > 0.0f))
        return {};
    const float side = dot (geometricNormal_, wi);
    if (wiLocal.z > 0.0f ? ! (side > 0.0f) : ! (side < 0.0f))
        return {};
    return evalLocal (woLocal, wiLocal);
}

template <class Color>
PT_SHARED_ENTRY BasicBsdfEval<Color> BasicBsdf<Color>::eval (Vec3 wo, Vec3 wi, std::uint8_t rayKinds) const noexcept
{
    const Vec3 woLocal = frame_.toLocal (wo);
    const Vec3 wiLocal = frame_.toLocal (wi);
    if (! (woLocal.z > 0.0f))
        return {};
    const float side = dot (geometricNormal_, wi);
    if (wiLocal.z > 0.0f ? ! (side > 0.0f) : ! (side < 0.0f))
        return {};
    return evalLocal (woLocal, wiLocal, rayKinds);
}

template <class Color>
PT_SHARED_ENTRY bool BasicBsdf<Color>::sample (Vec3 wo, float uLobe, Vec2 uDirection, BasicBsdfSample<Color>& out) const noexcept
{
    out = {};
    if (count_ == 0)
        return false;
    const Vec3 woLocal = frame_.toLocal (wo);
    if (! (woLocal.z > 0.0f))
        return false;

    // Pick a lobe and stretch the remainder of uLobe back to [0, 1).
    int chosen = count_ - 1;
    float lower = 0.0f;
    for (int i = 0; i < count_; ++i)
    {
        const float p = probability_[static_cast<std::size_t> (i)];
        if (uLobe < lower + p || i == count_ - 1)
        {
            chosen = i;
            break;
        }
        lower += p;
    }
    const auto slot = static_cast<std::size_t> (chosen);
    const float probability = probability_[slot];
    if (! (probability > 0.0f))
        return false;
    const float uChoice = std::min (std::max ((uLobe - lower) / probability, 0.0f), 0.99999994f);

    BasicBsdfSample<Color> local;
    if (! sampleLobeImpl (lobes_[slot], woLocal, uChoice, uDirection, local))
        return false;
    // A straight sample is exactly the incoming ray, not a rotated copy.
    const Vec3 wi = (local.flags & kBsdfStraight) != 0 ? -wo : normalizeOrZero (frame_.toWorld (local.wi));
    const float side = dot (geometricNormal_, wi);
    if (local.wi.z > 0.0f ? ! (side > 0.0f) : ! (side < 0.0f))
        return false;

    out.wi = wi;
    out.flags = local.flags;
    out.channelPdfRatio = local.channelPdfRatio;
    if ((local.flags & kBsdfDelta) != 0)
    {
        out.weight = local.weight * (1.0f / probability);
        out.pdf = probability * local.pdf;
        return finiteColor (out.weight) && out.pdf > 0.0f;
    }
    // At the direction the path takes, as eval reads it. The lobe's own wi is not quite
    // unit, and a narrow lobe's D turns on that last bit.
    const BasicBsdfEval<Color> mixture = evalLocal (woLocal, frame_.toLocal (wi));
    if (! (mixture.pdf > 0.0f) || ! std::isfinite (mixture.pdf))
        return false;
    out.weight = mixture.value * (1.0f / mixture.pdf);
    out.pdf = mixture.pdf;
    return finiteColor (out.weight);
}

template class BasicBsdf<Vec3>;
template class BasicBsdf<Spectrum>;

// buildBsdf's colour conversions: RGB as it is, or the reflectance spectrum at
// the path's wavelengths. Function objects, as Metal has no lambdas.
struct RgbColor
{
    PT_SHARED_FN Vec3 operator() (Vec3 rgb) const noexcept { return rgb; }
};

struct SpectralColor
{
    const Wavelengths& wavelengths;
    PT_SHARED_FN Spectrum operator() (Vec3 rgb) const noexcept { return reflectanceSpectrum (rgb, wavelengths); }
};

PT_SHARED_ENTRY void buildMaterialBsdf (const Material& material, Vec3 shadingNormal, Vec3 geometricNormal,
                        Vec3 wo, bool entering, Bsdf& out) noexcept
{
    const float ior = std::isfinite (material.ior) ? std::min (3.0f, std::max (1.01f, material.ior)) : 1.5f;
    DielectricSetup setup;
    // A sheet has the same medium on both sides; a volume or interface
    // refracts into the material when entering and out of it otherwise.
    const float eta = material.transport == DielectricTransport::ThinSheet || entering ? ior : 1.0f / ior;
    for (float& channel : setup.eta)
        channel = eta;
    setup.etaD = eta;
    buildBsdf (material, shadingNormal, geometricNormal, wo, setup, RgbColor {}, out);
}

PT_SHARED_ENTRY void buildMaterialBsdf (const Material& material, Vec3 shadingNormal, Vec3 geometricNormal,
                        Vec3 wo, const Wavelengths& wavelengths, const DielectricSetup& setup,
                        SpectralBsdf& out) noexcept
{
    DielectricSetup used = setup;
    used.dispersive = setup.dispersive && wavelengths.spectral;
    buildBsdf (material, shadingNormal, geometricNormal, wo, used,
               SpectralColor { wavelengths }, out);
}

#if ! defined(__CUDACC__)
void exportGgxTables (device::DeviceLookupTables& out) noexcept
{
    const GgxAlbedoTable& table = ggxAlbedoTable();
    out.ggxScale = table.scale();
    out.ggxBias = table.bias();
    out.ggxEntries = kGgxTableSize * kGgxTableSize;
}
#endif

} // namespace videohelper::pathtrace
