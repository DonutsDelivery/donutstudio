#pragma once

// Nested dielectrics for the CPU path tracer.
//
// A path keeps a small stack of the media (closed volumes and interfaces)
// it is inside. The medium that owns a point is the one with the largest
// Material::mediumPriority, and among equal priorities the one entered last.
// A surface hit is then one of three things (after Schmidt and Budge,
// "Simple Nested Dielectrics in Ray Traced Images", JGT 7(2), 2002):
// - a real interface between two media: it refracts with the ratio of their
//   indices at each wavelength;
// - a false interface: the boundary of a medium that is not the owner on the
//   side the ray comes from or goes to, such as a glass wall inside water that
//   has a higher priority. The ray passes straight through it and only the
//   stack changes;
// - an unbalanced exit through a medium the stack does not hold (a path that
//   started inside it, or an open interface seen from behind): it is treated
//   as leaving that medium into the current owner, so the index ratio stays
//   finite and the stack is left alone.
// Without priorities (all 0) this reduces to: entering a medium refracts from
// the current one into it, leaving refracts back into the one around it. With
// one medium at a time it is the PT-A2 behaviour of air on the far side.
//
// Behavioural reference (studied, not copied): Blender Cycles
// intern/cycles/kernel/integrator/volume_stack.h for stack entry and exit by
// shader, and the priority rule of the nested dielectric paper above.

#include "devices/device_annotations.h"
#include "bsdf.h"
#include "spectrum.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace videohelper::pathtrace
{

class MediumStack
{
public:
    static constexpr int kCapacity = 8;

    PT_SHARED_FN bool empty() const noexcept { return count_ == 0; }
    PT_SHARED_FN int size() const noexcept { return count_; }

    // The medium the path is in, or null outside every medium (index 1).
    PT_SHARED_FN const PT_GLOBAL Material* active() const noexcept
    {
        const PT_GLOBAL Material* best = nullptr;
        for (int i = 0; i < count_; ++i)
        {
            const PT_GLOBAL Material* m = entries_[static_cast<std::size_t> (i)];
            if (best == nullptr || m->mediumPriority >= best->mediumPriority)
                best = m;
        }
        return best;
    }

    PT_SHARED_FN bool contains (const PT_GLOBAL Material* material) const noexcept
    {
        for (int i = 0; i < count_; ++i)
            if (entries_[static_cast<std::size_t> (i)] == material)
                return true;
        return false;
    }

    // False (and no change) when the stack is full; a later exit through
    // that medium is then an unbalanced exit.
    PT_SHARED_FN bool push (const PT_GLOBAL Material* material) noexcept
    {
        if (material == nullptr || count_ >= kCapacity)
            return false;
        entries_[static_cast<std::size_t> (count_)] = material;
        ++count_;
        return true;
    }

    // Removes the latest entry of material. False when there is none.
    PT_SHARED_FN bool remove (const PT_GLOBAL Material* material) noexcept
    {
        for (int i = count_ - 1; i >= 0; --i)
        {
            if (entries_[static_cast<std::size_t> (i)] != material)
                continue;
            for (int j = i; j + 1 < count_; ++j)
                entries_[static_cast<std::size_t> (j)] = entries_[static_cast<std::size_t> (j + 1)];
            --count_;
            entries_[static_cast<std::size_t> (count_)] = nullptr;
            return true;
        }
        return false;
    }

private:
    std::array<const PT_GLOBAL Material*, kCapacity> entries_ {};
    int count_ = 0;
};

// What a hit on a medium boundary means for the path.
struct BoundaryCrossing
{
    // Pass straight through without scattering; only the stack changes.
    bool falseInterface = false;
    // Medium on the side the ray comes from and on the far side; null is
    // outside every medium.
    const PT_GLOBAL Material* nearSide = nullptr;
    const PT_GLOBAL Material* farSide = nullptr;
    // Crossing it enters the material (push) rather than leaving it (remove).
    bool entering = false;
};

// frontFace: the ray arrives on the geometric front face, so it enters the
// material when it crosses.
PT_SHARED_FN BoundaryCrossing classifyBoundary (const MediumStack& stack, const PT_GLOBAL Material* material, bool frontFace) noexcept
{
    BoundaryCrossing crossing;
    const PT_GLOBAL Material* owner = stack.active();
    if (frontFace)
    {
        crossing.entering = true;
        if (owner != nullptr && owner->mediumPriority > material->mediumPriority)
        {
            crossing.falseInterface = true;
            return crossing;
        }
        crossing.nearSide = owner;
        crossing.farSide = material;
        return crossing;
    }
    crossing.entering = false;
    if (! stack.contains (material))
    {
        crossing.nearSide = material;
        crossing.farSide = owner;
        return crossing;
    }
    if (owner != material)
    {
        crossing.falseInterface = true;
        return crossing;
    }
    MediumStack after = stack;
    after.remove (material);
    crossing.nearSide = material;
    crossing.farSide = after.active();
    return crossing;
}

// Updates the stack for a transmission through the boundary (real or false).
PT_SHARED_FN void applyCrossing (MediumStack& stack, const PT_GLOBAL Material* material, const BoundaryCrossing& crossing) noexcept
{
    if (crossing.entering)
        stack.push (material);
    else
        stack.remove (material);
}

// Index of a medium at a wavelength; 1 outside every medium.
PT_SHARED_FN float mediumIor (const PT_GLOBAL Material* medium, float nm, bool spectral) noexcept
{
    return medium != nullptr ? materialIorAt (*medium, nm, spectral) : 1.0f;
}

// Far index over near index per channel, for buildMaterialBsdf.
PT_SHARED_FN DielectricSetup dielectricSetupFor (const BoundaryCrossing& crossing, const Wavelengths& wavelengths) noexcept
{
    DielectricSetup setup;
    for (std::size_t c = 0; c < setup.eta.size(); ++c)
        setup.eta[c] = mediumIor (crossing.farSide, wavelengths.nm[c], wavelengths.spectral)
                     / mediumIor (crossing.nearSide, wavelengths.nm[c], wavelengths.spectral);
    setup.etaD = mediumIor (crossing.farSide, kDLineNm, false) / mediumIor (crossing.nearSide, kDLineNm, false);
    setup.dispersive = wavelengths.spectral
                    && ((crossing.farSide != nullptr && materialDispersive (*crossing.farSide))
                        || (crossing.nearSide != nullptr && materialDispersive (*crossing.nearSide)));
    return setup;
}

// Beer-Lambert transmittance over distance through absorption sigma. A
// channel without absorption passes everything, even over an infinite
// distance.
PT_SHARED_FN Spectrum beerLambert (const Spectrum& sigma, float distance) noexcept
{
    Spectrum t = Spectrum::filled (1.0f);
    for (int c = 0; c < kSpectrumChannels; ++c)
        if (sigma[c] > 0.0f)
            t[c] = std::exp (-sigma[c] * distance);
    return t;
}

// ---------------------------------------------------------------- participating media
//
// A medium is the material that owns the path's position (the stack's
// owner), or the scene's world volume outside every medium. Its coefficients
// per channel are a uniform Beer-Lambert absorption (Material::absorptionDistance)
// plus the volume's absorption, scattering and emission, which a density
// field may scale by f(x) in [0, 1]. The sum at f = 1 is the majorant.
//
// Homogeneous media sample a free-flight distance in closed form. The
// channel that draws the distance is chosen uniformly among the path's live
// channels, and the weight divides by the mixture density over all of them
// (spectral MIS, as Cycles' volume_sample_channel), so the four wavelengths
// share one distance density and the hero wavelength weights stay as they are.
// Heterogeneous media use spectral tracking (Kutz et al. 2017) for the
// distance and ratio tracking (Novak et al. 2014) for transmittance, both
// against the scalar majorant. Emission and lamps on the segment are added
// deterministically for the whole segment, so they never depend on whether a
// scatter happened before them.
//
// Behavioural reference (studied, not copied): Cycles
// intern/cycles/kernel/integrator/shade_volume.h and
// intern/cycles/kernel/closure/volume.h, and PBRT-v4's media.h.

struct MediumCoefficients
{
    Spectrum absorption {};        // uniform, never scaled by the field
    Spectrum volumeAbsorption {};  // the volume's terms at full density
    Spectrum scattering {};
    Spectrum emission {};          // radiance per unit length
    float anisotropy = 0.0f;
    // Heterogeneous: the material whose density field scales the volume terms.
    const PT_GLOBAL Material* field = nullptr;
    // The world volume fills a ball around the scene, not all of space.
    bool bounded = false;
    Vec3 centre {};
    float radius = 0.0f;

    PT_SHARED_FN Spectrum majorant() const noexcept { return absorption + volumeAbsorption + scattering; }
    PT_SHARED_FN bool scatters() const noexcept { return scattering.anyPositive(); }
    PT_SHARED_FN bool emits() const noexcept { return emission.anyPositive(); }
    PT_SHARED_FN bool any() const noexcept { return majorant().anyPositive() || emits(); }
};

// The coefficients of medium (null is vacuum). volumes false keeps only the
// uniform absorption, for rays that must not see the volume.
PT_SHARED_FN MediumCoefficients mediumCoefficientsOf (const PT_GLOBAL Material* medium, const Wavelengths& wavelengths, bool volumes) noexcept
{
    MediumCoefficients m;
    if (medium == nullptr)
        return m;
    m.absorption = mediumAbsorption (*medium, wavelengths);
    if (! volumes || ! materialHasVolume (*medium))
        return m;
    volumeCoefficients (*medium, wavelengths, m.volumeAbsorption, m.scattering, m.emission);
    const float g = std::isfinite (medium->volumeAnisotropy) ? medium->volumeAnisotropy : 0.0f;
    m.anisotropy = g < -0.99f ? -0.99f : (g > 0.99f ? 0.99f : g);
    if (medium->volumeField == VolumeDensityField::Noise)
        m.field = medium;
    return m;
}

// Small stream for tracking, seeded per segment.
struct VolumeRng
{
    std::uint32_t state = 0;

    PT_SHARED_FN static std::uint32_t hash (std::uint32_t input) noexcept
    {
        const std::uint32_t state = input * 747796405u + 2891336453u;
        const std::uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }
    // Uniform in [0, 1).
    PT_SHARED_FN float next() noexcept
    {
        state = hash (state);
        return static_cast<float> (state >> 8) * (1.0f / 16777216.0f);
    }
};

// A seed from a sample in [0, 1) and a position, for streams that have no
// sample dimension of their own (shadow rays).
// x wrapped into [0, period) as an integer, 0 for anything not finite.
PT_SHARED_FN std::uint32_t volumeWrapped (float x, float period) noexcept
{
    const float w = x - period * std::floor (x / period);
    return std::isfinite (w) && w >= 0.0f && w < period ? static_cast<std::uint32_t> (w) : 0u;
}

PT_SHARED_FN std::uint32_t volumeSeed (float u, Vec3 p) noexcept
{
    std::uint32_t h = VolumeRng::hash (volumeWrapped (u * 16777216.0f, 16777216.0f));
    h = VolumeRng::hash (h ^ volumeWrapped (p.x * 4096.0f, 16777216.0f));
    h = VolumeRng::hash (h ^ volumeWrapped (p.y * 4096.0f, 16777216.0f));
    return VolumeRng::hash (h ^ volumeWrapped (p.z * 4096.0f, 16777216.0f));
}

// Value at a lattice point, wrapped to a large period so any position hashes.
PT_SHARED_FN float volumeLattice (float x, float y, float z) noexcept
{
    constexpr float kPeriod = 1048576.0f;
    std::uint32_t h = VolumeRng::hash (volumeWrapped (x, kPeriod));
    h = VolumeRng::hash (h ^ (volumeWrapped (y, kPeriod) * 0x9e3779b9u));
    h = VolumeRng::hash (h ^ (volumeWrapped (z, kPeriod) * 0x85ebca6bu));
    return static_cast<float> (h >> 8) * (1.0f / 16777216.0f);
}

PT_SHARED_FN float volumeFade (float t) noexcept { return t * t * (3.0f - 2.0f * t); }
PT_SHARED_FN float volumeLerp (float a, float b, float t) noexcept { return a + (b - a) * t; }

// Trilinear value noise with a smoothstep fade, in [0, 1].
PT_SHARED_FN float volumeValueNoise (Vec3 p) noexcept
{
    const float fx = std::floor (p.x);
    const float fy = std::floor (p.y);
    const float fz = std::floor (p.z);
    const float tx = volumeFade (p.x - fx);
    const float ty = volumeFade (p.y - fy);
    const float tz = volumeFade (p.z - fz);
    const float c00 = volumeLerp (volumeLattice (fx, fy, fz), volumeLattice (fx + 1.0f, fy, fz), tx);
    const float c10 = volumeLerp (volumeLattice (fx, fy + 1.0f, fz), volumeLattice (fx + 1.0f, fy + 1.0f, fz), tx);
    const float c01 = volumeLerp (volumeLattice (fx, fy, fz + 1.0f), volumeLattice (fx + 1.0f, fy, fz + 1.0f), tx);
    const float c11 = volumeLerp (volumeLattice (fx, fy + 1.0f, fz + 1.0f), volumeLattice (fx + 1.0f, fy + 1.0f, fz + 1.0f), tx);
    return volumeLerp (volumeLerp (c00, c10, ty), volumeLerp (c01, c11, ty), tz);
}

// The density field of a material at world position p, in [0, 1]; 1 for a
// uniform volume.
PT_SHARED_FN float volumeFieldAt (const PT_GLOBAL Material& material, Vec3 p) noexcept
{
    if (material.volumeField != VolumeDensityField::Noise)
        return 1.0f;
    const float scale = std::isfinite (material.volumeFieldScale) && material.volumeFieldScale > 0.0f ? material.volumeFieldScale : 1.0f;
    Vec3 q = (p + material.volumeFieldOffset) * scale;
    const int octaves = material.volumeFieldDetail < 1 ? 1 : (material.volumeFieldDetail > 8 ? 8 : material.volumeFieldDetail);
    float sum = 0.0f;
    float amplitude = 1.0f;
    float total = 0.0f;
    for (int o = 0; o < octaves; ++o)
    {
        sum += amplitude * volumeValueNoise (q);
        total += amplitude;
        amplitude *= 0.5f;
        q = q * 2.0f + Vec3 { 17.1f, 31.7f, 5.3f };
    }
    const float f = sum / total;
    return f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
}

// Henyey-Greenstein phase function of the angle between the incoming
// propagation direction and the outgoing one; g > 0 scatters forward. It
// integrates to 1 over the sphere and is its own sampling density.
PT_SHARED_FN float henyeyGreenstein (float cosTheta, float g) noexcept
{
    const float g2 = g * g;
    const float denom = 1.0f + g2 - 2.0f * g * cosTheta;
    if (! (denom > 0.0f))
        return 0.0f;
    return (1.0f - g2) / (4.0f * kPi * denom * std::sqrt (denom));
}

// A direction around forward (unit) with density henyeyGreenstein.
PT_SHARED_FN Vec3 sampleHenyeyGreenstein (Vec3 forward, float g, Vec2 u, float& pdf) noexcept
{
    float cosTheta = 0.0f;
    if (std::fabs (g) < 1.0e-3f)
        cosTheta = 1.0f - 2.0f * u.x;
    else
    {
        const float k = (1.0f - g * g) / (1.0f - g + 2.0f * g * u.x);
        cosTheta = (1.0f + g * g - k * k) / (2.0f * g);
    }
    cosTheta = cosTheta < -1.0f ? -1.0f : (cosTheta > 1.0f ? 1.0f : cosTheta);
    const float sinSquared = 1.0f - cosTheta * cosTheta;
    const float sinTheta = sinSquared > 0.0f ? std::sqrt (sinSquared) : 0.0f;
    const float phi = 2.0f * kPi * u.y;
    pdf = henyeyGreenstein (cosTheta, g);
    return Frame::fromNormal (forward).toWorld ({ sinTheta * std::cos (phi), sinTheta * std::sin (phi), cosTheta });
}

// The part [a, b] of the segment from origin along unit dir over distance d
// that the medium fills. False when it fills none of it.
PT_SHARED_FN bool mediumSpan (const MediumCoefficients& m, Vec3 origin, Vec3 dir, float d, float& a, float& b) noexcept
{
    a = 0.0f;
    b = d;
    if (! m.bounded)
        return true;
    const Vec3 oc = origin - m.centre;
    const float half = dot (oc, dir);
    const float c = dot (oc, oc) - m.radius * m.radius;
    const float disc = half * half - c;
    if (! (disc > 0.0f))
        return false;
    const float root = std::sqrt (disc);
    const float enter = -half - root;
    const float leave = -half + root;
    a = enter > 0.0f ? enter : 0.0f;
    b = leave < d ? leave : d;
    return b > a;
}

// Ratio tracking transmittance over [a, b] of a heterogeneous medium.
PT_SHARED_FN Spectrum ratioTrackingTransmittance (const MediumCoefficients& m, Vec3 origin, Vec3 dir, float a, float b,
                                                  VolumeRng& rng) noexcept
{
    const Spectrum majorant = m.majorant();
    const float mu = majorant.maxValue();
    Spectrum t = Spectrum::filled (1.0f);
    if (! (mu > 0.0f))
        return t;
    if (! std::isfinite (b))
    {
        // A field with a positive mean absorbs everything over an infinite span.
        for (int c = 0; c < kSpectrumChannels; ++c)
            t[c] = majorant[c] > 0.0f ? 0.0f : 1.0f;
        return t;
    }
    float x = a;
    for (;;)
    {
        x += -std::log (1.0f - rng.next()) / mu;
        if (! (x < b))
            return t;
        const float f = volumeFieldAt (*m.field, origin + dir * x);
        for (int c = 0; c < kSpectrumChannels; ++c)
        {
            const float sigmaT = m.absorption[c] + f * (m.volumeAbsorption[c] + m.scattering[c]);
            const float keep = 1.0f - sigmaT / mu;
            t[c] *= keep > 0.0f ? keep : 0.0f;
        }
        // Russian roulette once the estimate is small, so the walk ends.
        const float peak = t.maxValue();
        if (peak < 0.1f)
        {
            if (! (peak > 0.0f) || rng.next() >= peak)
                return {};
            t = t * (1.0f / peak);
        }
    }
}

// Fraction of light that crosses distance d of the medium from origin along
// unit dir, per channel.
PT_SHARED_FN Spectrum mediumTransmittance (const MediumCoefficients& m, Vec3 origin, Vec3 dir, float d, VolumeRng& rng) noexcept
{
    float a = 0.0f;
    float b = 0.0f;
    if (! mediumSpan (m, origin, dir, d, a, b))
        return Spectrum::filled (1.0f);
    if (m.field != nullptr)
        return ratioTrackingTransmittance (m, origin, dir, a, b, rng);
    return beerLambert (m.majorant(), b - a);
}

// Outcome of crossing one segment of a medium.
struct MediumEvent
{
    bool scattered = false;   // at distance t; otherwise the path reaches the segment's end
    float t = 0.0f;
    // Multiplies the throughput: transmittance (and scattering) over the
    // density of the event. 0 when the path was absorbed.
    Spectrum weight = Spectrum::filled (1.0f);
    // Emitted radiance gathered along the segment, to add times the
    // throughput the path had before it.
    Spectrum emitted {};
};

// Samples where a path crossing distance d of the medium scatters, if it
// does. live marks the channels the path still carries (throughput > 0);
// they decide the distance density alike, so it does not depend on which
// one is the hero. u is a sample in [0, 1).
PT_SHARED_FN MediumEvent sampleMediumEvent (const MediumCoefficients& m, Vec3 origin, Vec3 dir, float d,
                                            const Spectrum& live, float u) noexcept
{
    MediumEvent event;
    float a = 0.0f;
    float b = 0.0f;
    if (! mediumSpan (m, origin, dir, d, a, b))
        return event;
    const Spectrum majorant = m.majorant();
    int liveCount = 0;
    for (int c = 0; c < kSpectrumChannels; ++c)
        liveCount += live[c] > 0.0f ? 1 : 0;
    if (liveCount == 0)
        return event;

    if (m.field == nullptr)
    {
        const float span = b - a;
        // Emission over the whole span, closed form.
        for (int c = 0; c < kSpectrumChannels; ++c)
        {
            const float e = m.emission[c];
            if (! (e > 0.0f))
                continue;
            const float sigma = majorant[c];
            event.emitted[c] = sigma > 0.0f ? e * (1.0f - std::exp (-sigma * span)) / sigma
                                            : (std::isfinite (span) ? e * span : 0.0f);
        }
        if (! m.scatters())
        {
            event.weight = beerLambert (majorant, span);
            return event;
        }
        // Channel by u, then reuse what is left of u for the distance.
        const float scaled = u * static_cast<float> (liveCount);
        int pick = static_cast<int> (scaled);
        pick = pick >= liveCount ? liveCount - 1 : pick;
        const float v = scaled - static_cast<float> (pick);
        int channel = 0;
        for (int c = 0, seen = 0; c < kSpectrumChannels; ++c)
        {
            if (! (live[c] > 0.0f))
                continue;
            if (seen++ == pick)
            {
                channel = c;
                break;
            }
        }
        const float sigmaPick = majorant[channel];
        const float distance = sigmaPick > 0.0f ? -std::log (1.0f - v) / sigmaPick : std::numeric_limits<float>::infinity();
        const float share = 1.0f / static_cast<float> (liveCount);
        if (distance < span)
        {
            float pdf = 0.0f;
            for (int c = 0; c < kSpectrumChannels; ++c)
                if (live[c] > 0.0f)
                    pdf += share * majorant[c] * std::exp (-majorant[c] * distance);
            const Spectrum transmittance = beerLambert (majorant, distance);
            event.scattered = true;
            event.t = a + distance;
            event.weight = pdf > 0.0f ? m.scattering * transmittance * (1.0f / pdf) : Spectrum {};
            return event;
        }
        float pass = 0.0f;
        const Spectrum transmittance = beerLambert (majorant, span);
        for (int c = 0; c < kSpectrumChannels; ++c)
            if (live[c] > 0.0f)
                pass += share * transmittance[c];
        event.weight = pass > 0.0f ? transmittance * (1.0f / pass) : Spectrum {};
        return event;
    }

    // Spectral tracking against the scalar majorant.
    const float mu = majorant.maxValue();
    if (! (mu > 0.0f))
        return event;
    if (! std::isfinite (b))
    {
        // An open mesh seen from inside: nothing passes an infinite span of a
        // field with a positive mean, except channels the medium leaves alone.
        for (int c = 0; c < kSpectrumChannels; ++c)
            if (majorant[c] > 0.0f)
                event.weight[c] = 0.0f;
        return event;
    }
    VolumeRng rng { VolumeRng::hash (static_cast<std::uint32_t> (u * 16777216.0f) ^ 0x6a09e667u) };
    float x = a;
    for (;;)
    {
        x += -std::log (1.0f - rng.next()) / mu;
        if (! (x < b))
            return event;
        const float f = volumeFieldAt (*m.field, origin + dir * x);
        Spectrum sa;
        Spectrum ss;
        Spectrum sn;
        float ps = 0.0f;
        float pn = 0.0f;
        float total = 0.0f;
        for (int c = 0; c < kSpectrumChannels; ++c)
        {
            sa[c] = m.absorption[c] + f * m.volumeAbsorption[c];
            ss[c] = f * m.scattering[c];
            const float nullPart = mu - sa[c] - ss[c];
            sn[c] = nullPart > 0.0f ? nullPart : 0.0f;
            event.emitted[c] += event.weight[c] * f * m.emission[c] / mu;
            if (! (live[c] > 0.0f))
                continue;
            const float w = event.weight[c];
            ps += ss[c] * w;
            pn += sn[c] * w;
            total += (sa[c] + ss[c] + sn[c]) * w;
        }
        if (! (total > 0.0f))
        {
            event.weight = {};
            return event;
        }
        const float xi = rng.next() * total;
        if (xi < ps)
        {
            event.scattered = true;
            event.t = x;
            event.weight = event.weight * ss * (total / (mu * ps));
            return event;
        }
        if (xi < ps + pn)
        {
            event.weight = event.weight * sn * (total / (mu * pn));
            continue;
        }
        event.weight = {};
        return event;
    }
}

} // namespace videohelper::pathtrace
