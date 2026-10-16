#include "bsdf.h"
#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace videohelper::pathtrace
{
namespace
{

constexpr float kTwoPi = 6.28318530717958647692f;

float clampUnit (float x) noexcept
{
    return std::isfinite (x) ? std::min (1.0f, std::max (0.0f, x)) : 0.0f;
}

Vec3 clampUnit (Vec3 v) noexcept
{
    return { clampUnit (v.x), clampUnit (v.y), clampUnit (v.z) };
}

float average (Vec3 v) noexcept
{
    return (v.x + v.y + v.z) * (1.0f / 3.0f);
}

bool finitePositive (float x) noexcept
{
    return std::isfinite (x) && x > 0.0f;
}

float pow5 (float x) noexcept
{
    const float x2 = x * x;
    return x2 * x2 * x;
}

Vec3 schlick (Vec3 f0, float cosine) noexcept
{
    const float s = pow5 (clampUnit (1.0f - cosine));
    return { f0.x + (1.0f - f0.x) * s, f0.y + (1.0f - f0.y) * s, f0.z + (1.0f - f0.z) * s };
}

// GGX normal distribution for a unit half vector in the local frame.
float ggxD (Vec3 h, float alpha) noexcept
{
    if (h.z <= 0.0f)
        return 0.0f;
    const float a2 = alpha * alpha;
    const float c2 = h.z * h.z;
    const float d = c2 * (a2 - 1.0f) + 1.0f;
    return a2 / (kPi * d * d);
}

// Smith Lambda of GGX. Grazing directions give infinity, so G becomes 0.
float smithLambda (float cosTheta, float alpha) noexcept
{
    const float c2 = cosTheta * cosTheta;
    if (! (c2 > 0.0f))
        return std::numeric_limits<float>::infinity();
    const float t2 = std::max (0.0f, 1.0f - c2) / c2;
    return 0.5f * (std::sqrt (1.0f + alpha * alpha * t2) - 1.0f);
}

// Visible-normal sample around wo (wo.z > 0) by the projected-hemisphere
// construction of Heitz 2018. The result is a unit microfacet normal with z > 0.
Vec3 sampleVisibleNormal (Vec3 wo, float alpha, Vec2 u) noexcept
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

Vec3 reflectAbout (Vec3 wo, Vec3 h) noexcept
{
    return h * (2.0f * dot (wo, h)) - wo;
}

// Refracts wo (pointing away, dot(wo, n) > 0) through a surface with normal n.
// eta is the far index over the near index. False on total internal reflection.
bool refractThrough (Vec3 wo, Vec3 n, float eta, Vec3& wt) noexcept
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
Vec3 cosineHemisphere (Vec2 u) noexcept
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

// Directional albedo table of the single-scatter GGX reflection lobe, split in
// the Fresnel-independent scale and bias terms. Rows are perceptual roughness
// in [0, 1] (alpha = max(r^2, the opaque floor)), columns cos(theta_o) in [0, 1].
class GgxAlbedoTable
{
public:
    static constexpr int kSize = 32;
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

    void lookup (float cosO, float alpha, float& scale, float& bias) const noexcept
    {
        const float r = std::sqrt (clampUnit (alpha));
        const float fr = r * static_cast<float> (kSize - 1);
        const float fc = clampUnit (cosO) * static_cast<float> (kSize - 1);
        const int r0 = std::min (static_cast<int> (fr), kSize - 2);
        const int c0 = std::min (static_cast<int> (fc), kSize - 2);
        const float tr = fr - static_cast<float> (r0);
        const float tc = fc - static_cast<float> (c0);
        const auto at = [] (const std::array<float, kSize * kSize>& table, int row, int column) {
            return table[static_cast<std::size_t> (row * kSize + column)];
        };
        const auto bilinear = [&] (const std::array<float, kSize * kSize>& table) {
            const float top = at (table, r0, c0) * (1.0f - tc) + at (table, r0, c0 + 1) * tc;
            const float bottom = at (table, r0 + 1, c0) * (1.0f - tc) + at (table, r0 + 1, c0 + 1) * tc;
            return top * (1.0f - tr) + bottom * tr;
        };
        scale = bilinear (scale_);
        bias = bilinear (bias_);
    }

private:
    std::array<float, kSize * kSize> scale_ {};
    std::array<float, kSize * kSize> bias_ {};
};

const GgxAlbedoTable& ggxAlbedoTable() noexcept
{
    static const GgxAlbedoTable table;
    return table;
}

} // namespace

float materialCoverage (const Material& material) noexcept
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

Frame Frame::fromNormal (Vec3 n) noexcept
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

float fresnelDielectric (float cosI, float eta) noexcept
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

void ggxAlbedoScaleBias (float cosO, float alpha, float& scale, float& bias) noexcept
{
    ggxAlbedoTable().lookup (cosO, alpha, scale, bias);
}

std::uint32_t lobeFlags (const Lobe& lobe) noexcept
{
    switch (lobe.kind)
    {
        case LobeKind::Lambert:
            return kBsdfReflection | kBsdfDiffuse;
        case LobeKind::GgxReflection:
            return kBsdfReflection | kBsdfGlossy;
        case LobeKind::SmoothDielectric:
            return kBsdfReflection | kBsdfTransmission | kBsdfDelta;
        case LobeKind::RoughDielectric:
            return kBsdfReflection | kBsdfTransmission | kBsdfGlossy;
    }
    return 0;
}

BsdfEval evalLobe (const Lobe& lobe, Vec3 wo, Vec3 wi) noexcept
{
    BsdfEval out;
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
            out.value = lobe.weight * schlick (lobe.f0, woh) * (d * g2 / (4.0f * wo.z));
            out.pdf = g1 * d / (4.0f * wo.z);
            return out;
        }
        case LobeKind::SmoothDielectric:
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

bool sampleLobe (const Lobe& lobe, Vec3 wo, float uChoice, Vec2 u, BsdfSample& out) noexcept
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
            out.weight = lobe.weight * schlick (lobe.f0, woh) * ratio;
            out.flags = kBsdfReflection | kBsdfGlossy;
            return out.pdf > 0.0f;
        }
        case LobeKind::SmoothDielectric:
        {
            const float f = fresnelDielectric (wo.z, lobe.eta);
            if (uChoice < f)
            {
                out.wi = { -wo.x, -wo.y, wo.z };
                out.weight = lobe.weight;
                out.pdf = f;
                out.flags = kBsdfReflection | kBsdfDelta;
                return true;
            }
            if (! refractThrough (wo, { 0.0f, 0.0f, 1.0f }, lobe.eta, out.wi) || ! (out.wi.z < 0.0f))
                return false;
            out.weight = lobe.transmissionTint;
            out.pdf = 1.0f - f;
            out.flags = kBsdfTransmission | kBsdfDelta;
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
    }
    return false;
}

void Bsdf::reset (Vec3 shadingNormal, Vec3 geometricNormal) noexcept
{
    frame_ = Frame::fromNormal (shadingNormal);
    geometricNormal_ = geometricNormal;
    count_ = 0;
    nonDelta_ = false;
    probability_.fill (0.0f);
}

bool Bsdf::addLobe (const Lobe& lobe) noexcept
{
    if (count_ >= kMaxLobes || ! finitePositive (lobe.selectWeight))
        return false;
    lobes_[static_cast<std::size_t> (count_)] = lobe;
    ++count_;
    return true;
}

void Bsdf::finalize() noexcept
{
    float total = 0.0f;
    for (int i = 0; i < count_; ++i)
        total += lobes_[static_cast<std::size_t> (i)].selectWeight;
    nonDelta_ = false;
    for (int i = 0; i < count_; ++i)
    {
        const auto slot = static_cast<std::size_t> (i);
        probability_[slot] = total > 0.0f ? lobes_[slot].selectWeight / total : 0.0f;
        if ((lobeFlags (lobes_[slot]) & kBsdfDelta) == 0)
            nonDelta_ = true;
    }
}

BsdfEval Bsdf::evalLocal (Vec3 woLocal, Vec3 wiLocal) const noexcept
{
    BsdfEval out;
    for (int i = 0; i < count_; ++i)
    {
        const auto slot = static_cast<std::size_t> (i);
        if ((lobeFlags (lobes_[slot]) & kBsdfDelta) != 0)
            continue;
        const BsdfEval e = evalLobe (lobes_[slot], woLocal, wiLocal);
        out.value = out.value + e.value;
        out.pdf += probability_[slot] * e.pdf;
    }
    return out;
}

BsdfEval Bsdf::eval (Vec3 wo, Vec3 wi) const noexcept
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

bool Bsdf::sample (Vec3 wo, float uLobe, Vec2 uDirection, BsdfSample& out) const noexcept
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

    BsdfSample local;
    if (! sampleLobe (lobes_[slot], woLocal, uChoice, uDirection, local))
        return false;
    const Vec3 wi = normalizeOrZero (frame_.toWorld (local.wi));
    const float side = dot (geometricNormal_, wi);
    if (local.wi.z > 0.0f ? ! (side > 0.0f) : ! (side < 0.0f))
        return false;

    out.wi = wi;
    out.flags = local.flags;
    if ((local.flags & kBsdfDelta) != 0)
    {
        out.weight = local.weight * (1.0f / probability);
        out.pdf = probability * local.pdf;
        return isFinite (out.weight) && out.pdf > 0.0f;
    }
    const BsdfEval mixture = evalLocal (woLocal, local.wi);
    if (! (mixture.pdf > 0.0f) || ! std::isfinite (mixture.pdf))
        return false;
    out.weight = mixture.value * (1.0f / mixture.pdf);
    out.pdf = mixture.pdf;
    return isFinite (out.weight);
}

void buildMaterialBsdf (const Material& material, Vec3 shadingNormal, Vec3 geometricNormal,
                        Vec3 wo, bool entering, Bsdf& out) noexcept
{
    out.reset (shadingNormal, geometricNormal);
    const Vec3 base = clampUnit (material.baseColor);
    const float metallic = clampUnit (material.metallic);
    const float roughness = clampUnit (material.roughness);
    const float transmission = clampUnit (material.transmission);
    const float ior = std::isfinite (material.ior) ? std::min (3.0f, std::max (1.01f, material.ior)) : 1.5f;
    const float cosO = clampUnit (dot (shadingNormal, wo));
    const float opaqueAlpha = std::max (roughness * roughness, kMinOpaqueGgxAlpha);

    if (metallic > 0.0f)
    {
        float scale = 0.0f;
        float bias = 0.0f;
        ggxAlbedoScaleBias (cosO, opaqueAlpha, scale, bias);
        Lobe metal;
        metal.kind = LobeKind::GgxReflection;
        metal.weight = { metallic, metallic, metallic };
        metal.f0 = base;
        metal.alpha = opaqueAlpha;
        metal.selectWeight = metallic * average (base * scale + Vec3 { bias, bias, bias });
        out.addLobe (metal);
    }

    const float glassWeight = (1.0f - metallic) * transmission;
    if (glassWeight > 0.0f)
    {
        Lobe glass;
        const float alpha = roughness * roughness;
        glass.kind = alpha < kSmoothDielectricAlpha ? LobeKind::SmoothDielectric : LobeKind::RoughDielectric;
        glass.alpha = std::max (alpha, kSmoothDielectricAlpha);
        glass.eta = entering ? ior : 1.0f / ior;
        glass.weight = { glassWeight, glassWeight, glassWeight };
        glass.transmissionTint = base * glassWeight;
        const float f = fresnelDielectric (cosO, glass.eta);
        glass.selectWeight = glassWeight * (f + (1.0f - f) * average (base));
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

        Lobe specular;
        specular.kind = LobeKind::GgxReflection;
        specular.weight = { opaqueWeight, opaqueWeight, opaqueWeight };
        specular.f0 = { f0, f0, f0 };
        specular.alpha = opaqueAlpha;
        specular.selectWeight = opaqueWeight * specularAlbedo;
        out.addLobe (specular);

        Lobe diffuse;
        diffuse.kind = LobeKind::Lambert;
        diffuse.weight = base * (opaqueWeight * (1.0f - specularAlbedo));
        diffuse.selectWeight = average (diffuse.weight);
        out.addLobe (diffuse);
    }
    out.finalize();
}

} // namespace videohelper::pathtrace
