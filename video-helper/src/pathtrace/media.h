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

namespace videohelper::pathtrace
{

class MediumStack
{
public:
    static constexpr int kCapacity = 8;

    PT_SHARED_FN bool empty() const noexcept { return count_ == 0; }
    PT_SHARED_FN int size() const noexcept { return count_; }

    // The medium the path is in, or null outside every medium (index 1).
    PT_SHARED_FN const Material* active() const noexcept
    {
        const Material* best = nullptr;
        for (int i = 0; i < count_; ++i)
        {
            const Material* m = entries_[static_cast<std::size_t> (i)];
            if (best == nullptr || m->mediumPriority >= best->mediumPriority)
                best = m;
        }
        return best;
    }

    PT_SHARED_FN bool contains (const Material* material) const noexcept
    {
        for (int i = 0; i < count_; ++i)
            if (entries_[static_cast<std::size_t> (i)] == material)
                return true;
        return false;
    }

    // False (and no change) when the stack is full; a later exit through
    // that medium is then an unbalanced exit.
    PT_SHARED_FN bool push (const Material* material) noexcept
    {
        if (material == nullptr || count_ >= kCapacity)
            return false;
        entries_[static_cast<std::size_t> (count_)] = material;
        ++count_;
        return true;
    }

    // Removes the latest entry of material. False when there is none.
    PT_SHARED_FN bool remove (const Material* material) noexcept
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
    std::array<const Material*, kCapacity> entries_ {};
    int count_ = 0;
};

// What a hit on a medium boundary means for the path.
struct BoundaryCrossing
{
    // Pass straight through without scattering; only the stack changes.
    bool falseInterface = false;
    // Medium on the side the ray comes from and on the far side; null is
    // outside every medium.
    const Material* nearSide = nullptr;
    const Material* farSide = nullptr;
    // Crossing it enters the material (push) rather than leaving it (remove).
    bool entering = false;
};

// frontFace: the ray arrives on the geometric front face, so it enters the
// material when it crosses.
PT_SHARED_FN BoundaryCrossing classifyBoundary (const MediumStack& stack, const Material* material, bool frontFace) noexcept
{
    BoundaryCrossing crossing;
    const Material* owner = stack.active();
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
PT_SHARED_FN void applyCrossing (MediumStack& stack, const Material* material, const BoundaryCrossing& crossing) noexcept
{
    if (crossing.entering)
        stack.push (material);
    else
        stack.remove (material);
}

// Index of a medium at a wavelength; 1 outside every medium.
PT_SHARED_FN float mediumIor (const Material* medium, float nm, bool spectral) noexcept
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
    Spectrum t = Spectrum::constant (1.0f);
    for (int c = 0; c < kSpectrumChannels; ++c)
        if (sigma[c] > 0.0f)
            t[c] = std::exp (-sigma[c] * distance);
    return t;
}

} // namespace videohelper::pathtrace
