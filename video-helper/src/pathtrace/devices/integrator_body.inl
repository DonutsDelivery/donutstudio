// Single-source path integrator body (Plan 3, Tier V). Not a header: a
// device driver TU includes it once after defining
//
//   PT_DEVICE            the device struct (device_interface.h lists its members)
//   PT_DEVICE_NAMESPACE  the namespace under videohelper::pathtrace::device that
//                        receives tracePath, renderSample and usesSpectralTransport
//
// The transport rules are documented in integrator.h; this file is where they
// live, so every device type renders them identically.

#if ! defined(PT_DEVICE) || ! defined(PT_DEVICE_NAMESPACE)
#error "Define PT_DEVICE and PT_DEVICE_NAMESPACE before including integrator_body.inl"
#endif

#include "scene_view.h"
#include "../../diffraction_order_core.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace videohelper::pathtrace::device::PT_DEVICE_NAMESPACE
{

using Device = PT_DEVICE;
using SceneView = device::SceneView<Device::Accel>;

namespace
{

// Sample dimensions: the pixel filter takes dimension 0, then every path
// vertex owns a block of kDimensionsPerVertex numbers.
enum : std::uint32_t
{
    kSlotTransparency = 0,
    kSlotLightSelect = 1,
    kSlotLightPosition = 2,
    kSlotLobe = 3,
    kSlotDirection = 4,
    kSlotRoulette = 5,
    kSlotCoverage = 6,
    kSlotVolume = 7,  // scatter distance in a participating medium
    kDimensionsPerVertex = 8
};

PT_DEVICE_FN std::uint32_t dimensionOf (std::uint32_t vertex, std::uint32_t slot) noexcept
{
    return 1u + vertex * kDimensionsPerVertex + slot;
}

// A caustic light path draws from its own dimensions, far above any camera
// path's, so the two estimates of one pixel sample stay independent.
enum : std::uint32_t
{
    kLightPathBase = 0x00100000u,
    kLightPathSelect = kLightPathBase + 0,
    kLightPathPosition = kLightPathBase + 1,
    kLightPathDirection = kLightPathBase + 2,
    kLightPathWavelength = kLightPathBase + 3,
    kLightPathLens = kLightPathBase + 4,
    kLightPathTime = kLightPathBase + 5
};

PT_DEVICE_FN std::uint32_t lightPathDimensionOf (std::uint32_t vertex, std::uint32_t slot) noexcept
{
    return kLightPathBase + 8u + vertex * kDimensionsPerVertex + slot;
}

// Surfaces where a caustic light path ends in a camera connection, and where
// a camera path's first scatter makes its later delta chain the light
// tracer's. Refracting interfaces and thin sheets would need the medium and
// radiance scaling across them at the connection, so both methods leave
// them to the camera path.
PT_DEVICE_FN bool causticReceiver (const PT_GLOBAL Material& material, const PT_GLOBAL GratingBinding* grating) noexcept
{
    return grating == nullptr && ! isMediumBoundary (material) && ! isThinSheet (material);
}

constexpr std::uint32_t kMaxShadowCrossings = 16;

PT_DEVICE_FN bool anyPositive (Vec3 v) noexcept { return v.x > 0.0f || v.y > 0.0f || v.z > 0.0f; }

// Power heuristic with exponent 2, written as a ratio so huge pdfs cannot
// overflow into inf / inf.
PT_DEVICE_FN float powerHeuristic (float chosen, float other) noexcept
{
    if (! (chosen > 0.0f))
        return 0.0f;
    if (! (other > 0.0f))
        return 1.0f;
    const float ratio = other / chosen;
    return 1.0f / (1.0f + ratio * ratio);
}

// Weight of an emitter reached by a BSDF ray. lightPdf is the density light
// sampling would have given the same direction, 0 when it cannot reach it.
PT_DEVICE_FN float emitterHitWeight (LightStrategy strategy, float bsdfPdf, float lightPdf) noexcept
{
    if (! (lightPdf > 0.0f))
        return 1.0f;
    switch (strategy)
    {
        case LightStrategy::MultipleImportance:
            return powerHeuristic (bsdfPdf, lightPdf);
        case LightStrategy::LightOnly:
            return 0.0f;
        case LightStrategy::BsdfOnly:
            return 1.0f;
    }
    return 1.0f;
}

PT_DEVICE_FN const PT_GLOBAL Material& materialOf (const SceneView& view, std::uint32_t triangle) noexcept
{
    if (triangle >= view.triangles.count)
        return *view.fallbackMaterial;
    const std::uint32_t index = view.triangles[triangle].materialIndex;
    return index < view.materials.count ? view.materials[index] : *view.fallbackMaterial;
}

// One of a triangle's per-vertex values at barycentric (u, v), w = 1 - u - v.
PT_DEVICE_FN float barycentricMix (float a, float b, float c, float w, float u, float v) noexcept
{
    return a * w + b * u + c * v;
}

// The BSDF's copy of a hit's material with the vertex lanes applied, as Raster
// applies them: the interpolated vertex colour multiplies the base colour, and
// the vertex roughness replaces the material's by its weight. Media and
// coverage keep materialOf's shared material, whose address names the medium.
PT_DEVICE_FN Material vertexShaded (const SceneView& view, const Hit& hit, const PT_GLOBAL Material& material) noexcept
{
    Material shaded = material;
    if (hit.triangle >= view.triangles.count)
        return shaded;
    const PT_GLOBAL TriangleAttributes& t = view.triangles[hit.triangle];
    const float w = 1.0f - hit.u - hit.v;
    shaded.baseColor = { material.baseColor.x * barycentricMix (t.colors[0][0], t.colors[1][0], t.colors[2][0], w, hit.u, hit.v),
                         material.baseColor.y * barycentricMix (t.colors[0][1], t.colors[1][1], t.colors[2][1], w, hit.u, hit.v),
                         material.baseColor.z * barycentricMix (t.colors[0][2], t.colors[1][2], t.colors[2][2], w, hit.u, hit.v) };
    const float roughness = barycentricMix (t.roughness[0][0], t.roughness[1][0], t.roughness[2][0], w, hit.u, hit.v);
    float weight = barycentricMix (t.roughness[0][1], t.roughness[1][1], t.roughness[2][1], w, hit.u, hit.v);
    weight = weight < 0.0f ? 0.0f : weight > 1.0f ? 1.0f : weight;
    shaded.roughness = material.roughness + (roughness - material.roughness) * weight;
    return shaded;
}

PT_DEVICE_FN const PT_GLOBAL GratingBinding* gratingOf (const SceneView& view, std::uint32_t triangle) noexcept
{
    if (view.gratings.empty() || triangle >= view.triangles.count)
        return nullptr;
    const std::uint32_t index = view.triangles[triangle].materialIndex;
    for (std::uint32_t i = 0; i < view.gratings.count; ++i)
        if (view.gratings[i].materialIndex == index)
            return &view.gratings[i];
    return nullptr;
}

struct SurfacePoint
{
    Vec3 position {};
    Vec3 geometricNormal {};  // front face
    Vec3 shadingNormal {};    // same hemisphere as geometricNormal
};

// Something in the scene or the camera moves over the motion blur shutter.
PT_DEVICE_FN bool sceneMoves (const SceneView& view, const PinholeCamera& camera) noexcept
{
    return ! view.motionOpen.empty() || camera.moving;
}

// Stored vertex index of a triangle where it is at ray time t. Without motion
// the still positions.
PT_DEVICE_FN Vec3 vertexAt (const SceneView& view, std::uint32_t index, float t) noexcept
{
    if (view.motionOpen.empty())
        return view.positions[index];
    return motionLerp (view.motionOpen[index], view.motionClose[index], t);
}

PT_DEVICE_FN SurfacePoint surfaceAt (const SceneView& view, const Hit& hit) noexcept
{
    const std::uint32_t base = hit.triangle * 3u;
    const PT_GLOBAL TriangleAttributes& attributes = view.triangles[hit.triangle];
    const float w = 1.0f - hit.u - hit.v;
    SurfacePoint point;
    const Vec3 p0 = vertexAt (view, base, hit.time);
    const Vec3 p1 = vertexAt (view, base + 1, hit.time);
    const Vec3 p2 = vertexAt (view, base + 2, hit.time);
    point.position = p0 * w + p1 * hit.u + p2 * hit.v;
    point.geometricNormal = attributes.geometricNormal;
    // A moving triangle turns over the shutter: its normal at the hit's time.
    // Stored order is counter-clockwise from the front, as at ingest.
    if (! view.motionOpen.empty())
    {
        const Vec3 moved = normalizeOrZero (cross (p1 - p0, p2 - p0));
        if (dot (moved, moved) > 0.0f)
            point.geometricNormal = moved;
    }
    point.shadingNormal = point.geometricNormal;
    if (attributes.hasShadingNormals)
    {
        const Vec3 n = normalizeOrZero (attributes.shadingNormals[0] * w + attributes.shadingNormals[1] * hit.u
                                        + attributes.shadingNormals[2] * hit.v);
        if (dot (n, n) > 0.0f)
            point.shadingNormal = dot (n, point.geometricNormal) < 0.0f ? -n : n;
    }
    return point;
}

// Shadow-terminator origin for authored normals (Hanika 2021). Each triangle
// corner's tangent plane lifts the point onto the curved surface the normals
// describe, so a ray to a light that the normals face clears the neighbouring
// facets and a coarse mesh casts no faceted shadow edge on itself. Elsewhere
// the origin is the usual geometric offset.
PT_DEVICE_FN Vec3 spawnFromSurface (const SceneView& view, const Hit& hit, const SurfacePoint& point,
                                    Vec3 ng, Vec3 ns, Vec3 direction) noexcept
{
    const PT_GLOBAL TriangleAttributes& attributes = view.triangles[hit.triangle];
    if (! attributes.authoredShading)
        return spawnRayOrigin (point.position, ng, direction);
    const Vec3 away = dot (ns, direction) < 0.0f ? -ns : ns;
    const std::uint32_t base = hit.triangle * 3u;
    const float weights[3] { 1.0f - hit.u - hit.v, hit.u, hit.v };
    Vec3 origin = point.position;
    for (std::uint32_t k = 0; k < 3u; ++k)
    {
        const Vec3 stored = attributes.shadingNormals[k];
        const Vec3 corner = dot (stored, away) < 0.0f ? -stored : stored;
        const float below = dot (point.position - vertexAt (view, base + k, hit.time), corner);
        if (below < 0.0f)
            origin = origin - corner * (below * weights[k]);
    }
    // A direction the normals face but the facet does not dives through the facet
    // and leaves through a neighbour, which would pin the terminator to a facet
    // edge. Start past that chord, at most two triangle edges long, so the
    // terminator falls where the normals turn away from the light.
    if (dot (ng, direction) < 0.0f && dot (ns, direction) > 0.0f)
    {
        const Vec3 a = vertexAt (view, base + 1u, hit.time) - vertexAt (view, base, hit.time);
        const Vec3 b = vertexAt (view, base + 2u, hit.time) - vertexAt (view, base + 1u, hit.time);
        const Vec3 c = vertexAt (view, base, hit.time) - vertexAt (view, base + 2u, hit.time);
        const float longest = std::sqrt (std::max (dot (a, a), std::max (dot (b, b), dot (c, c))));
        origin = origin + direction * (2.0f * longest);
    }
    return offsetRayOrigin (origin, away);
}

// The shading normal to use when an interpolated authored normal turns away from
// the viewer at a silhouette: tilted just far enough to face it, which stays
// continuous where falling back to the facet normal would show the facet.
PT_DEVICE_FN Vec3 faceViewer (Vec3 ns, Vec3 ng, Vec3 wo) noexcept
{
    const Vec3 tilted = normalizeOrZero (ns + wo * (1.0e-3f - dot (ns, wo)));
    return dot (tilted, wo) > 1.0e-4f ? tilted : ng;
}

// The medium a path with this stack is in: the stack's owner, else the world
// volume. volumes false keeps only the owner's uniform absorption.
PT_DEVICE_FN MediumCoefficients mediumAt (const SceneView& view, const MediumStack& media, const Wavelengths& wavelengths,
                                          bool volumes = true) noexcept
{
    const PT_GLOBAL Material* owner = media.active();
    if (owner != nullptr)
        return mediumCoefficientsOf (owner, wavelengths, volumes);
    MediumCoefficients world;
    if (! volumes || view.worldMedium == nullptr)
        return world;
    world = mediumCoefficientsOf (view.worldMedium, wavelengths, true);
    world.field = nullptr;
    world.bounded = true;
    world.centre = view.boundsCentre;
    world.radius = view.worldVolumeRadius;
    return world;
}

// Transmittance of the straight line between two points in the medium of media.
PT_DEVICE_FN Spectrum lineTransmittance (const SceneView& view, const MediumStack& media, const Wavelengths& wavelengths,
                                         Vec3 from, Vec3 to) noexcept
{
    const MediumCoefficients medium = mediumAt (view, media, wavelengths);
    if (! medium.majorant().anyPositive())
        return Spectrum::filled (1.0f);
    const Vec3 delta = to - from;
    const float distance = length (delta);
    VolumeRng rng { volumeSeed (0.5f, from) };
    return mediumTransmittance (medium, from, normalizeOrZero (delta), distance, rng);
}

// A volume-only surface: rays cross it and only the medium stack changes.
PT_DEVICE_FN bool volumeOnlySurface (const PT_GLOBAL Material& material) noexcept
{
    return material.volumeOnly && materialHasVolume (material);
}

// Ray kinds, as HarmonicMIDI::grid::sceneRayVisibility numbers them.
constexpr std::uint8_t kRayCamera = 1u;
constexpr std::uint8_t kRayShadow = 2u;
constexpr std::uint8_t kRayDiffuse = 4u;
constexpr std::uint8_t kRayGlossy = 8u;
constexpr std::uint8_t kRayTransmission = 16u;
// A ray leaving a scatter in a medium (an object's volume or the world fog).
// Lights that clear this bit add no in-scatter: next event estimation at the
// scatter skips them, and the ray it sends on does not see them.
constexpr std::uint8_t kRayVolumeScatter = 32u;

// Closest hit among the triangles a ray of kind rayKind sees. A hidden object's
// triangles are stepped over; the next search starts just past the hidden hit,
// so every step moves forward and the walk ends.
PT_DEVICE_FN bool intersectVisible (const SceneView& view, Ray ray, Hit& hit, std::uint8_t rayKind) noexcept
{
    bool found = Device::intersect (view.accel, ray, hit);
    while (found && view.anyHiddenObjects && (view.triangles[hit.triangle].visibility & rayKind) == 0)
    {
        ray.tMin = std::nextafter (hit.t, std::numeric_limits<float>::infinity());
        ray.ignoreTriangle = hit.triangle;
        found = Device::intersect (view.accel, ray, hit);
    }
    return found;
}

// Fraction of light that crosses the segment of `ray`, per channel. media is
// the stack at the ray origin. Opaque surfaces and real refracting interfaces
// block it; alpha cut-outs and thin sheets attenuate it, false interfaces let
// it through, and the owning medium absorbs along the way. Objects hidden from
// rays of kind rayKind are passed over.
PT_DEVICE_FN Spectrum shadowTransmittance (const SceneView& view, Ray ray, MediumStack media,
                                           const Wavelengths& wavelengths, PathCounters& counters,
                                           std::uint8_t rayKind = kRayShadow, bool volumes = true) noexcept
{
    ++counters.shadowRays;
    if (media.empty() && ! view.anyPartialCoverage && ! view.anyThinSheet && ! view.anyVolume && ! view.anyHiddenObjects)
        return Device::occluded (view.accel, ray) ? Spectrum {} : Spectrum::filled (1.0f);
    const float directionLength = length (ray.direction);
    const Vec3 direction = normalizeOrZero (ray.direction);
    Spectrum transmittance = Spectrum::filled (1.0f);
    float segmentStart = ray.tMin;
    VolumeRng rng { volumeSeed (0.25f, ray.origin + direction) };
    for (std::uint32_t crossing = 0; crossing <= kMaxShadowCrossings; ++crossing)
    {
        Hit hit;
        // Objects hidden from shadow rays cast no shadow.
        const bool blocked = intersectVisible (view, ray, hit, rayKind);
        const float segmentEnd = blocked ? hit.t : ray.tMax;
        const MediumCoefficients medium = mediumAt (view, media, wavelengths, volumes);
        if (medium.majorant().anyPositive())
            transmittance = transmittance
                          * mediumTransmittance (medium, ray.origin + direction * (segmentStart * directionLength), direction,
                                                 (segmentEnd - segmentStart) * directionLength, rng);
        if (! blocked)
            return transmittance;
        const PT_GLOBAL Material& material = materialOf (view, hit.triangle);
        const float coverage = materialCoverage (material);
        const Vec3 geometricNormal = view.triangles[hit.triangle].geometricNormal;
        const bool front = dot (geometricNormal, direction) <= 0.0f;
        if (volumeOnlySurface (material))
            applyCrossing (media, &material, classifyBoundary (media, &material, front));
        else if (isMediumBoundary (material))
        {
            const BoundaryCrossing boundary = classifyBoundary (media, &material, front);
            if (boundary.falseInterface)
                applyCrossing (media, &material, boundary);
            else
                transmittance = transmittance * (1.0f - coverage);
        }
        else if (isThinSheet (material))
        {
            const float around = mediumIor (media.active(), kDLineNm, false);
            const Spectrum straight = straightTransmittance (material, dot (geometricNormal, direction),
                                                             materialIorAt (material, kDLineNm, false) / around, wavelengths);
            transmittance = transmittance * (Spectrum::filled (1.0f - coverage) + straight * coverage);
        }
        else
        {
            transmittance = transmittance * (1.0f - coverage);
        }
        if (! transmittance.anyPositive())
            return {};
        segmentStart = hit.t;
        ray.tMin = hit.t;
        ray.ignoreTriangle = hit.triangle;
    }
    return {};
}

// A lamp seen from inside a body through one flat, clear face. A shadow ray stops at that
// face, so the straight line cannot reach the lamp; this finds the refracted line instead.
// wi is the direction inside the body, and solidAnglePerArea the density of that direction
// per unit area of the lamp, which carries a light sample's area density through the face.
struct PlanarExit
{
    Vec3 wi {};
    Vec3 target {};
    float solidAnglePerArea = 0.0f;
    float transmission = 0.0f;   // 1 - Fresnel, also the chance a sampled path crosses here
    Spectrum tint {};            // the face's tint and the absorption up to it
    Hit hit {};
    SurfacePoint point {};
    Vec3 inward {};              // the face's normal on the body's side
    MediumStack outside {};
};

// Fermat's principle on the plane through the face. depth and height are the point's and the
// lamp's distances from it, span their distance along it, eta the near index over the far one.
// Returns the distance along the plane to the crossing. The function is increasing with one
// root in [0, span]: Newton, kept inside a bracket that halves when a step leaves it.
PT_DEVICE_FN double planarCrossing (double depth, double height, double span, double eta) noexcept
{
    double low = 0.0;
    double high = span;
    double x = span * depth / (depth + height);
    for (int i = 0; i < 48; ++i)
    {
        const double inside = std::sqrt (x * x + depth * depth);
        const double rest = span - x;
        const double outside = std::sqrt (rest * rest + height * height);
        const double value = eta * x / inside - rest / outside;
        if (value > 0.0)
            high = x;
        else
            low = x;
        const double slope = eta * depth * depth / (inside * inside * inside) + height * height / (outside * outside * outside);
        const double next = x - value / slope;
        const double stepped = next > low && next < high ? next : 0.5 * (low + high);
        const bool settled = std::fabs (stepped - x) <= 1.0e-12 * (span + depth);
        x = stepped;
        if (settled)
            break;
    }
    return x;
}

// Whether the face at `at` is one flat, fully covering, smooth dielectric boundary, with its
// BSDF built facing `side` into bsdf.
// The one lobe of a flat clear face, all its callers read. A whole BSDF per face kept
// five of them on a GPU thread's stack at once.
struct ClearFace
{
    float eta = 1.0f;
    Spectrum transmissionTint;
};

PT_DEVICE_FN bool flatClearFace (const SceneView& view, const Wavelengths& wavelengths, const Hit& at, const SurfacePoint& surface,
                                 Vec3 wo, Vec3 side, const BoundaryCrossing& crossing, ClearFace& face) noexcept
{
    const PT_GLOBAL Material& material = materialOf (view, at.triangle);
    if (materialCoverage (material) < 1.0f || crossing.falseInterface
        || std::fabs (dot (surface.shadingNormal, surface.geometricNormal)) < 0.999999f)
        return false;
    SpectralBsdf bsdf;
    buildMaterialBsdf (vertexShaded (view, at, material), side, side, wo, wavelengths, dielectricSetupFor (crossing, wavelengths), bsdf);
    if (bsdf.lobeCount() != 1 || bsdf.lobe (0).kind != LobeKind::SmoothDielectric)
        return false;
    face.eta = bsdf.lobe (0).eta;
    face.transmissionTint = bsdf.lobe (0).transmissionTint;
    return true;
}

// False means this lamp point has no such line: no boundary in the way, a boundary that is rough,
// mixed, curved or faceted, or a shading normal off the face's. Those paths reach the lamp by
// scattering alone, as before; nothing is refused. Whatever lies past the face is the shadow
// ray's to find, so a second interface blocks this line and leaves the light to scattering too.
// The answer depends only on the two points, so a light sample and a lamp hit agree on whether
// the other could have made the same path.
PT_DEVICE_FN bool planarExitTo (const SceneView& view, const MediumStack& media, const Wavelengths& wavelengths, const Hit& hit,
                                const SurfacePoint& point, Vec3 ng, Vec3 ns, Vec3 lampPoint, Vec3 lampNormal, PlanarExit& out) noexcept
{
    const Vec3 straight = lampPoint - point.position;
    if (! (dot (straight, ng) > 0.0f))
        return false;
    Ray probe;
    probe.time = hit.time;
    probe.origin = spawnFromSurface (view, hit, point, ng, ns, straight);
    probe.direction = straight;
    probe.tMin = 0.0f;
    probe.tMax = 1.0f;
    probe.ignoreTriangle = hit.triangle;
    Hit first;
    if (! Device::intersect (view.accel, probe, first) || ! isMediumBoundary (materialOf (view, first.triangle)))
        return false;
    const PT_GLOBAL Material* faceMaterial = &materialOf (view, first.triangle);
    const SurfacePoint firstPoint = surfaceAt (view, first);
    const bool front = dot (firstPoint.geometricNormal, straight) <= 0.0f;
    const Vec3 inward = normalizeOrZero (front ? firstPoint.geometricNormal : -firstPoint.geometricNormal);
    const BoundaryCrossing crossing = classifyBoundary (media, faceMaterial, front);
    ClearFace face;
    if (! flatClearFace (view, wavelengths, first, firstPoint, -normalizeOrZero (straight), inward, crossing, face))
        return false;

    const float depth = dot (point.position - firstPoint.position, inward);
    const float height = dot (firstPoint.position - lampPoint, inward);
    if (! (depth > 0.0f) || ! (height > 0.0f))
        return false;
    // The lobe's eta is the far index over the near one, at the hero wavelength.
    const double eta = 1.0 / static_cast<double> (face.eta);
    const Vec3 sideways = straight + inward * (depth + height);
    const double span = length (sideways);
    const double along = span > 0.0 ? planarCrossing (depth, height, span, eta) : 0.0;
    const Vec3 onward = span > 0.0 ? sideways * static_cast<float> (1.0 / span) : Vec3 {};
    const Vec3 toFace = onward * static_cast<float> (along) - inward * depth;
    if (! (dot (toFace, ns) > 0.0f) || ! (dot (toFace, ng) > 0.0f))
        return false;

    Ray line;
    line.time = hit.time;
    line.origin = spawnFromSurface (view, hit, point, ng, ns, toFace);
    line.direction = toFace;
    line.tMin = 0.0f;
    line.tMax = 1.001f;
    line.ignoreTriangle = hit.triangle;
    if (! Device::intersect (view.accel, line, out.hit) || &materialOf (view, out.hit.triangle) != faceMaterial)
        return false;
    out.point = surfaceAt (view, out.hit);
    if (std::fabs (dot (normalizeOrZero (out.point.geometricNormal), inward)) < 0.999999f
        || std::fabs (dot (out.point.position - firstPoint.position, inward)) > 1.0e-3f * depth)
        return false;
    out.wi = normalizeOrZero (toFace);
    ClearFace crossed;
    if (! flatClearFace (view, wavelengths, out.hit, out.point, -out.wi, inward, crossing, crossed))
        return false;

    const double inLength = std::sqrt (along * along + static_cast<double> (depth) * depth);
    const double rest = span - along;
    const double outLength = std::sqrt (rest * rest + static_cast<double> (height) * height);
    const double cosIn = depth / inLength;
    const double cosOut = height / outLength;
    const Vec3 leaving = normalizeOrZero (lampPoint - out.point.position);
    // The lamp lies at span(theta) along the plane. Rays over a ring of theta and of azimuth
    // reach a ring of span and of azimuth at the lamp's height, tilted to the lamp by the cosine
    // at the lamp over cosOut. Near span 0 the ratio sin(theta) / span has its limit.
    const double spanPerTheta = depth / (cosIn * cosIn) + height * eta * cosIn / (cosOut * cosOut * cosOut);
    const double sineOverSpan = span > 1.0e-6 * (depth + height) ? (along / inLength) / span : 1.0 / (depth + height * eta);
    out.solidAnglePerArea = static_cast<float> (sineOverSpan / spanPerTheta * std::fabs (dot (lampNormal, leaving)) / cosOut);
    out.transmission = 1.0f - fresnelDielectric (static_cast<float> (cosIn), crossed.eta);
    out.tint = crossed.transmissionTint * lineTransmittance (view, media, wavelengths, point.position, out.point.position);
    out.inward = inward;
    out.outside = media;
    applyCrossing (out.outside, faceMaterial, crossing);
    out.target = offsetRayOrigin (lampPoint, dot (lampNormal, leaving) < 0.0f ? lampNormal : -lampNormal);
    return out.solidAnglePerArea > 0.0f && out.transmission > 0.0f;
}

// A lamp seen from outside a clear body with two parallel flat faces, the straight line stopped
// by the near one. Enter and leave at one index and the direction in air comes back unchanged:
// theta in air, phi in the body, sin(theta) = n sin(phi). With a and h the point's and the lamp's
// distances from the faces and t the thickness, the lamp lies along the faces at
//     S(theta) = (a + h) tan(theta) + t tan(phi).
// Rays over a ring of theta and of azimuth reach a ring of S and of azimuth at the lamp's height,
// tilted to the lamp by the cosine at the lamp over cos(theta), so
//     d(omega) / dA = sin(theta) / S * d(theta) / dS * |cos lamp| / cos(theta)
//     dS / d(theta) = (a + h) / cos^2(theta) + t cos(theta) / (n cos^3(phi))
// and near S = 0 the ratio sin(theta) / S tends to 1 / (a + h + t / n). wi is theta's direction
// at the point, solidAnglePerArea the density above, transmission the (1 - Fresnel)^2 a scattered
// path also pays to cross both faces. The index squared of radiance cancels over the two.
struct PlanarSlab
{
    Vec3 wi {};
    Vec3 target {};
    float solidAnglePerArea = 0.0f;
    float transmission = 0.0f;
    Spectrum tint {};            // both faces' tint and the absorption up to the far one
    Hit hit {};                  // the far face
    SurfacePoint point {};
    Vec3 inward {};              // the faces' normal on the point's side
    Vec3 entry {};
    MediumStack outside {};
    // Set before anything that depends on the hero, so that whether the other heroes get a pass
    // does not depend on this one finding a line.
    bool dispersive = false;
};

// S in x = tan(theta) rises with one root in [0, span / (a + h)]: Newton, kept inside
// a bracket that halves when a step leaves it.
PT_DEVICE_FN double slabCrossing (double air, double thickness, double span, double index) noexcept
{
    double low = 0.0;
    double high = span / air;
    double x = span / (air + thickness / index);
    for (int i = 0; i < 48; ++i)
    {
        const double root = std::sqrt (index * index + (index * index - 1.0) * x * x);
        const double value = air * x + thickness * x / root - span;
        if (value > 0.0)
            high = x;
        else
            low = x;
        const double next = x - value / (air + thickness * index * index / (root * root * root));
        const double stepped = next > low && next < high ? next : 0.5 * (low + high);
        const bool settled = std::fabs (stepped - x) <= 1.0e-13 * (high + x);
        x = stepped;
        if (settled)
            break;
    }
    return x;
}

// False means this lamp point has no such line, and the lamp is left to scattering as before.
// The answer depends only on the two points and the hero, so a light sample and a lamp hit agree
// on whether the other could have made the same path.
PT_DEVICE_FN bool planarSlabTo (const SceneView& view, const MediumStack& media, const Wavelengths& wavelengths, const Hit& hit,
                                const SurfacePoint& point, Vec3 ng, Vec3 ns, Vec3 lampPoint, Vec3 lampNormal, PlanarSlab& out) noexcept
{
    const Vec3 straight = lampPoint - point.position;
    if (! (dot (straight, ng) > 0.0f))
        return false;
    Ray probe;
    probe.time = hit.time;
    probe.origin = spawnFromSurface (view, hit, point, ng, ns, straight);
    probe.direction = straight;
    probe.tMin = 0.0f;
    probe.tMax = 1.0f;
    probe.ignoreTriangle = hit.triangle;
    Hit first;
    if (! Device::intersect (view.accel, probe, first) || ! isMediumBoundary (materialOf (view, first.triangle)))
        return false;
    const PT_GLOBAL Material* body = &materialOf (view, first.triangle);
    const SurfacePoint firstPoint = surfaceAt (view, first);
    const bool front = dot (firstPoint.geometricNormal, straight) <= 0.0f;
    const Vec3 inward = normalizeOrZero (front ? firstPoint.geometricNormal : -firstPoint.geometricNormal);
    const BoundaryCrossing entering = classifyBoundary (media, body, front);
    ClearFace face;
    if (! entering.entering || ! flatClearFace (view, wavelengths, first, firstPoint, -normalizeOrZero (straight), inward, entering, face))
        return false;
    out.dispersive = dielectricSetupFor (entering, wavelengths).dispersive;

    // The far face, straight across from where the straight line went in.
    Ray across;
    across.time = first.time;
    across.origin = spawnFromSurface (view, first, firstPoint, inward, inward, -inward);
    across.direction = -inward;
    across.tMin = 0.0f;
    across.tMax = std::numeric_limits<float>::infinity();
    across.ignoreTriangle = first.triangle;
    Hit farHit;
    if (! Device::intersect (view.accel, across, farHit) || &materialOf (view, farHit.triangle) != body)
        return false;
    const SurfacePoint farPoint = surfaceAt (view, farHit);
    if (std::fabs (dot (normalizeOrZero (farPoint.geometricNormal), inward)) < 0.999999f)
        return false;
    MediumStack inside = media;
    applyCrossing (inside, body, entering);
    const BoundaryCrossing leaving = classifyBoundary (inside, body, dot (farPoint.geometricNormal, -inward) <= 0.0f);
    if (leaving.entering || leaving.farSide != entering.nearSide)
        return false;

    const float depth = dot (point.position - firstPoint.position, inward);
    const float thickness = dot (firstPoint.position - farPoint.position, inward);
    const float height = dot (farPoint.position - lampPoint, inward);
    if (! (depth > 0.0f) || ! (thickness > 0.0f) || ! (height > 0.0f))
        return false;
    const double index = static_cast<double> (face.eta);
    if (! (index > 1.0))
        return false;
    const double air = static_cast<double> (depth) + height;
    const Vec3 sideways = straight + inward * (depth + thickness + height);
    const double span = length (sideways);
    const double tanAir = span > 0.0 ? slabCrossing (air, thickness, span, index) : 0.0;
    const double root = std::sqrt (index * index + (index * index - 1.0) * tanAir * tanAir);
    const double tanBody = tanAir / root;
    const Vec3 onward = span > 0.0 ? sideways * static_cast<float> (1.0 / span) : Vec3 {};
    const Vec3 toFace = (onward * static_cast<float> (tanAir) - inward) * depth;
    if (! (dot (toFace, ns) > 0.0f) || ! (dot (toFace, ng) > 0.0f))
        return false;
    const float tolerance = 1.0e-3f * (depth + thickness);

    Ray line;
    line.time = hit.time;
    line.origin = spawnFromSurface (view, hit, point, ng, ns, toFace);
    line.direction = toFace;
    line.tMin = 0.0f;
    line.tMax = 1.001f;
    line.ignoreTriangle = hit.triangle;
    Hit in;
    if (! Device::intersect (view.accel, line, in) || &materialOf (view, in.triangle) != body)
        return false;
    const SurfacePoint inPoint = surfaceAt (view, in);
    if (std::fabs (dot (normalizeOrZero (inPoint.geometricNormal), inward)) < 0.999999f
        || std::fabs (dot (inPoint.position - firstPoint.position, inward)) > tolerance)
        return false;
    out.wi = normalizeOrZero (toFace);
    ClearFace crossedIn;
    if (! flatClearFace (view, wavelengths, in, inPoint, -out.wi, inward, entering, crossedIn))
        return false;

    const Vec3 through = (onward * static_cast<float> (tanBody) - inward) * thickness;
    Ray inner;
    inner.time = in.time;
    inner.origin = spawnFromSurface (view, in, inPoint, inward, inward, through);
    inner.direction = through;
    inner.tMin = 0.0f;
    inner.tMax = 1.001f;
    inner.ignoreTriangle = in.triangle;
    if (! Device::intersect (view.accel, inner, out.hit) || &materialOf (view, out.hit.triangle) != body)
        return false;
    out.point = surfaceAt (view, out.hit);
    if (std::fabs (dot (normalizeOrZero (out.point.geometricNormal), inward)) < 0.999999f
        || std::fabs (dot (out.point.position - farPoint.position, inward)) > tolerance)
        return false;
    ClearFace crossedOut;
    if (! flatClearFace (view, wavelengths, out.hit, out.point, normalizeOrZero (-through), inward, leaving, crossedOut))
        return false;
    if (std::fabs (static_cast<double> (crossedOut.eta) * index - 1.0) > 1.0e-5)
        return false;

    const double cosAir = 1.0 / std::sqrt (1.0 + tanAir * tanAir);
    const double cosBody = 1.0 / std::sqrt (1.0 + tanBody * tanBody);
    const Vec3 onwards = normalizeOrZero (lampPoint - out.point.position);
    const double spanPerTheta = air / (cosAir * cosAir) + thickness * cosAir / (index * cosBody * cosBody * cosBody);
    const double sineOverSpan = span > 1.0e-6 * (air + thickness) ? (tanAir * cosAir) / span : 1.0 / (air + thickness / index);
    out.solidAnglePerArea = static_cast<float> (sineOverSpan / spanPerTheta * std::fabs (dot (lampNormal, onwards)) / cosAir);
    const float crossing = 1.0f - fresnelDielectric (static_cast<float> (cosAir), crossedIn.eta);
    out.transmission = crossing * crossing;
    out.tint = crossedIn.transmissionTint * crossedOut.transmissionTint
             * lineTransmittance (view, media, wavelengths, point.position, inPoint.position)
             * lineTransmittance (view, inside, wavelengths, inPoint.position, out.point.position);
    out.inward = inward;
    out.entry = inPoint.position;
    out.outside = inside;
    applyCrossing (out.outside, body, leaving);
    out.target = offsetRayOrigin (lampPoint, dot (lampNormal, onwards) < 0.0f ? lampNormal : -lampNormal);
    return out.solidAnglePerArea > 0.0f && out.transmission > 0.0f;
}

PT_DEVICE_FN std::array<float, 3> toArray (Vec3 v) noexcept { return { v.x, v.y, v.z }; }

// A channel set under a light's authored spectrum, where it has one.
PT_DEVICE_FN Spectrum withSpectralBands (Spectrum channels, const LightSetView& lights, std::uint32_t light, const Wavelengths& wavelengths) noexcept
{
    if (wavelengths.spectral)
        for (int c = 0; c < kSpectrumChannels; ++c)
            channels[c] *= lights.spectralBandScale (light, wavelengths.nm[static_cast<std::size_t> (c)]);
    return channels;
}

// A spatial foil's corners at uv, bilinear as Raster reads them
// (backend_sokol.mm: lower = mix(f[0], f[1], u), upper = mix(f[2], f[3], u)).
PT_DEVICE_FN std::array<float, 4> grooveFieldAt (const GratingBinding& grating, float u, float v) noexcept
{
    const float s = std::clamp (u, 0.0f, 1.0f);
    const float t = std::clamp (v, 0.0f, 1.0f);
    std::array<float, 4> local {};
    for (std::size_t i = 0; i < 4; ++i)
        local[i] = (grating.field[0][i] * (1.0f - s) + grating.field[1][i] * s) * (1.0f - t)
                 + (grating.field[2][i] * (1.0f - s) + grating.field[3][i] * s) * t;
    return local;
}

// Whether this path meets the foil and not the object's own material. Raster blends
// the two by coverage for each pixel. One path takes one of them, so the mean over
// paths is that same blend.
PT_DEVICE_FN bool foilCoversHit (const PT_GLOBAL GratingBinding& bound, const SceneView& view, const Hit& hit, float choice) noexcept
{
    PT_GLOBAL_READ (GratingBinding) grating = bound;
    const PT_GLOBAL TriangleAttributes& attributes = view.triangles[hit.triangle];
    const float w = 1.0f - hit.u - hit.v;
    const float coverage = grooveFieldAt (grating, attributes.uvs[0].x * w + attributes.uvs[1].x * hit.u + attributes.uvs[2].x * hit.v,
                                          attributes.uvs[0].y * w + attributes.uvs[1].y * hit.u + attributes.uvs[2].y * hit.v)[3];
    return choice < coverage;
}

// A surface position's own tilt from brushed relief, in units of the slope, with
// unit RMS over uv: three octaves of streaks across v. Each streak has its own
// strength and fades in and out along u, so it reads as a scratch and not as a ruled
// line. The divisor is that sum's measured RMS at spacing 0.004 over 400000 samples.
PT_DEVICE_FN float streakCellHash (std::int64_t cell, std::uint32_t salt) noexcept
{
    std::uint32_t x = static_cast<std::uint32_t> (cell) * 2654435761u + salt * 40503u;
    x ^= x >> 15;
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    x ^= x >> 16;
    return static_cast<float> (x) / 4294967296.0f;
}

// One streak cell's signed relief at u.
PT_DEVICE_FN float streakCell (float u, std::uint32_t salt, std::int64_t k) noexcept
{
    const float rate = 3.0f + std::floor (4.0f * streakCellHash (k, salt + 1u));
    const float along = 0.55f + 0.45f * std::sin (2.0f * kPi * (u * rate + streakCellHash (k, salt + 2u)));
    return (2.0f * streakCellHash (k, salt) - 1.0f) * along;
}

PT_DEVICE_FN float streakOctave (float u, float v, float spacing, std::uint32_t salt) noexcept
{
    const float position = v / spacing;
    const float floorCell = std::floor (position);
    const float f = position - floorCell;
    const float s = f * f * (3.0f - 2.0f * f);
    const auto k = static_cast<std::int64_t> (floorCell);
    return streakCell (u, salt, k) * (1.0f - s) + streakCell (u, salt, k + 1) * s;
}

PT_DEVICE_FN float brushedRelief (float u, float v, float spacing) noexcept
{
    return (streakOctave (u, v, spacing, 1u) + 0.7f * streakOctave (u, v, spacing * 0.5f, 11u)
            + 0.5f * streakOctave (u, v, spacing * 0.25f, 23u)) / 0.435f;
}

// Samples one diffraction order of a grating (diffraction_order_bsdf.h) at the
// hero wavelength. Orders are delta lobes. The grating equation shifts the
// tangential direction cosines by a constant, which preserves projected solid
// angle (cos_i dw_i = cos_o dw_o), so the radiance throughput of an order is
// its reflected power over its selection probability. The authority is queried
// with wo as its incident direction (reciprocity); the order it returns is wi.
// False when the authority refuses the vertex (grazing wo, a wavelength
// outside its sampled range); the path then ends there.
//
// The authority's ledger also holds the substrate's reflected power that no
// admitted order resolves. That power leaves as the surface's own metallic
// reflection: a Gaussian lobe about the mirror direction with the authority's
// angular standard deviation. Order 0 leaves through the same lobe, since it is
// that same rough surface's reflection and has no dispersion to lose. As a delta
// it mirrored the environment's hard shapes onto the face. The other orders stay
// sharp because that deviation, applied to them, would spread sin(theta) over the
// whole visible band at the module's period and turn its colour bands grey. Both
// vertices carry a total of the substrate's power at most, and a lobe sample
// below the surface ends the path, so this adds no energy.
PT_DEVICE_FN bool gratingOrdersAt (const PT_GLOBAL GratingBinding& bound, const SceneView& view, const Hit& hit, Vec3& ns, Vec3 wo,
                      float wavelengthNm, const PT_GLOBAL Material* medium, bool spectral,
                      diffractionmaterial::DiffractionOrderDistribution& orders) noexcept
{
    PT_GLOBAL_READ (GratingBinding) grating = bound;
    const PT_GLOBAL TriangleAttributes& attributes = view.triangles[hit.triangle];
    const float w = 1.0f - hit.u - hit.v;
    // Unit before the tangent is made orthogonal to it, which the frame the authority
    // checks depends on. A refused frame ends the path black.
    std::array<float, 3> normal = toArray (ns);
    {
        const float size = std::sqrt (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        for (float& c : normal)
            c /= size;
    }
    // The authority reads groove directions in uv coordinates, so the frame is
    // dP/du and dP/dv. Vertex tangents cannot stand in for it: generated meshes
    // all carry the default +X tangent, which would turn a polar cap's concentric
    // grooves into one straight grating across the disc.
    const std::uint32_t base = hit.triangle * 3u;
    const std::array<float, 3> edge1 = toArray (vertexAt (view, base + 1, hit.time) - vertexAt (view, base, hit.time));
    const std::array<float, 3> edge2 = toArray (vertexAt (view, base + 2, hit.time) - vertexAt (view, base, hit.time));
    const float du1 = static_cast<float> (attributes.uvs[1].x) - attributes.uvs[0].x;
    const float dv1 = static_cast<float> (attributes.uvs[1].y) - attributes.uvs[0].y;
    const float du2 = static_cast<float> (attributes.uvs[2].x) - attributes.uvs[0].x;
    const float dv2 = static_cast<float> (attributes.uvs[2].y) - attributes.uvs[0].y;
    const float uvArea = du1 * dv2 - du2 * dv1;
    std::array<float, 3> tangent {};
    // The smooth dP/du from ingest. Each triangle's own dP/du is constant, which
    // makes a polar cap a fan of straight gratings, stepping by a whole wedge
    // from one triangle to the next. Reversing it on a back face keeps dP/dv's
    // axis, as the flip below does.
    const Vec3 smoothTangent = attributes.grooveTangents[0] * w + attributes.grooveTangents[1] * hit.u
                             + attributes.grooveTangents[2] * hit.v;
    tangent = toArray (dot (ns, attributes.geometricNormal) < 0.0f ? smoothTangent * -1.0f : smoothTangent);
    const bool smoothFrame = diffractionmaterial::ordercore::orthonormalTangent (normal, tangent);
    bool uvFrame = smoothFrame || std::abs (uvArea) > 1.0e-14f;
    if (uvFrame && ! smoothFrame)
    {
        std::array<float, 3> bitangent {};
        for (std::size_t i = 0; i < 3; ++i)
        {
            tangent[i] = (edge1[i] * dv2 - edge2[i] * dv1) / uvArea;
            bitangent[i] = (edge2[i] * du1 - edge1[i] * du2) / uvArea;
        }
        uvFrame = diffractionmaterial::ordercore::orthonormalTangent (normal, tangent);
        // The authority takes right-handed frames only. Where uv is mirrored,
        // such as a polar cap's seam wedge, reversing dP/du keeps dP/dv's axis.
        const float handedness = bitangent[0] * (normal[1] * tangent[2] - normal[2] * tangent[1])
                                + bitangent[1] * (normal[2] * tangent[0] - normal[0] * tangent[2])
                                + bitangent[2] * (normal[0] * tangent[1] - normal[1] * tangent[0]);
        if (handedness < 0.0f)
            for (float& c : tangent)
                c = -c;
    }
    if (! uvFrame)
    {
        // Flat uv: the authored tangent, else world +X on the surface, the generated meshes'
        // default. A triangle's own edge turned the grooves at a flat quad's diagonal.
        for (std::size_t i = 0; i < 3; ++i)
            tangent[i] = attributes.tangents[0][i] * w + attributes.tangents[1][i] * hit.u + attributes.tangents[2][i] * hit.v;
        if (! diffractionmaterial::ordercore::orthonormalTangent (normal, tangent))
        {
            tangent = std::abs (normal[0]) < 0.9f ? std::array<float, 3> { 1.0f, 0.0f, 0.0f } : std::array<float, 3> { 0.0f, 1.0f, 0.0f };
            if (! diffractionmaterial::ordercore::orthonormalTangent (normal, tangent))
                return false;
        }
    }
    const auto& relief = grating.description.roughness;
    if (relief.streakSlope > 0.0f)
    {
        // Tilts the surface towards dP/dv, so the caller's mirror direction, light
        // sampling and spawn all use the same normal as the orders below.
        const float tilt = relief.streakSlope
                         * brushedRelief (attributes.uvs[0].x * w + attributes.uvs[1].x * hit.u + attributes.uvs[2].x * hit.v,
                                          attributes.uvs[0].y * w + attributes.uvs[1].y * hit.u + attributes.uvs[2].y * hit.v,
                                          relief.streakSpacingUv);
        const std::array<float, 3> across { normal[1] * tangent[2] - normal[2] * tangent[1],
                                             normal[2] * tangent[0] - normal[0] * tangent[2],
                                             normal[0] * tangent[1] - normal[1] * tangent[0] };
        std::array<float, 3> tilted {};
        float size = 0.0f;
        for (std::size_t i = 0; i < 3; ++i)
        {
            tilted[i] = normal[i] + static_cast<float> (tilt) * across[i];
            size += tilted[i] * tilted[i];
        }
        size = std::sqrt (size);
        for (float& c : tilted)
            c /= size;
        const Vec3 tiltedFloat { static_cast<float> (tilted[0]), static_cast<float> (tilted[1]), static_cast<float> (tilted[2]) };
        // A streak on a surface seen at grazing keeps the viewer on its side.
        if (dot (tiltedFloat, wo) > 1.0e-4f)
        {
            normal = tilted;
            ns = tiltedFloat;
            if (! diffractionmaterial::ordercore::orthonormalTangent (normal, tangent))
                return false;
        }
    }
    diffractionmaterial::DiffractionLocalFrame frame;
    frame.normal = normal;
    frame.tangent = tangent;
    frame.bitangent = { frame.normal[1] * tangent[2] - frame.normal[2] * tangent[1],
                        frame.normal[2] * tangent[0] - frame.normal[0] * tangent[2],
                        frame.normal[0] * tangent[1] - frame.normal[1] * tangent[0] };
    const float uvX = attributes.uvs[0].x * w + attributes.uvs[1].x * hit.u + attributes.uvs[2].x * hit.v;
    const float uvY = attributes.uvs[0].y * w + attributes.uvs[1].y * hit.u + attributes.uvs[2].y * hit.v;
    // The authority's groove field is defined on [0, 1]^2.
    const std::array<float, 2> materialUv { std::clamp (static_cast<float> (uvX), 0.0f, 1.0f),
                                             std::clamp (static_cast<float> (uvY), 0.0f, 1.0f) };
    std::array<float, 3> incident = toArray (wo);
    {
        const float size = std::sqrt (incident[0] * incident[0] + incident[1] * incident[1] + incident[2] * incident[2]);
        for (float& c : incident)
            c /= size;
    }
    // A material that asks for it forms its orders in the medium the path is in.
    const float mediumIndex = grating.description.orderMedium == diffractionmaterial::OrderMedium::SurroundingMedium
                                 ? static_cast<float> (mediumIor (medium, wavelengthNm, spectral))
                                 : 1.0f;
    // Corners that face opposite ways cancel at a point with no direction of their own,
    // where the material's grating stands.
    diffractionmaterial::LocalGrating local;
    bool hasLocal = false;
    if (grating.varied)
    {
        const std::array<float, 4> field = grooveFieldAt (grating, uvX, uvY);
        const float size = std::hypot (static_cast<float> (field[0]), static_cast<float> (field[1]));
        if (size > 1.0e-6f)
        {
            local.directionUv = { field[0] / size, field[1] / size };
            local.periodNanometres = field[2];
            hasLocal = true;
        }
    }
    return diffractionmaterial::evaluateDiffractionOrdersCore (grating.description, wavelengthNm, frame, incident, materialUv,
                                                               orders, mediumIndex, hasLocal ? &local : nullptr)
           == diffractionmaterial::DiffractionOrderProblem::None;
}

// The power that leaves through the metal lobe: the unresolved remainder and order 0.
// A pressed disc's pit scatter and the relief's scatter leave by their own lobes.
PT_DEVICE_FN float gratingLobePower (const diffractionmaterial::DiffractionOrderDistribution& orders) noexcept
{
    float power = orders.unresolvedReflectedPower - orders.pitScatterPower - orders.reliefLeavingPower;
    for (std::size_t i = 0; i < orders.orderCount; ++i)
        if (orders.orders[i].primary == 0 && orders.orders[i].secondary == 0)
            power += orders.orders[i].reflectedPower;
    return static_cast<float> (power);
}

PT_DEVICE_FN float gratingLobeDeviation (const diffractionmaterial::DiffractionOrderDistribution& orders) noexcept
{
    return orders.orderCount > 0 ? static_cast<float> (orders.orders[0].angularStandardDeviationRadians) : 0.0f;
}

// A groove family's axis on the plane square to a lobe's centre. A centre along the axis
// has no such projection, and any tangent of the plane serves, the same one both ways.
PT_DEVICE_FN Vec3 gratingFamilyAxisAround (const diffractionmaterial::DiffractionOrderDistribution& orders, std::size_t family, Vec3 centre) noexcept
{
    const auto& axis = orders.grooveFamilyAxes[family];
    const Vec3 world { static_cast<float> (axis[0]), static_cast<float> (axis[1]), static_cast<float> (axis[2]) };
    const Vec3 projected = world - centre * dot (world, centre);
    return dot (projected, projected) > 1.0e-6f ? normalizeOrZero (projected) : Frame::fromNormal (centre).tangent;
}

// Solid-angle density of the lobe's sampling below: a Gaussian offset on the plane
// one unit along the mirror direction, carried to the sphere by 1 / cos^3.
PT_DEVICE_FN float gratingSpreadDensity (const diffractionmaterial::DiffractionOrderDistribution& orders, Vec3 centre, Vec3 wi) noexcept
{
    const float deviation = gratingLobeDeviation (orders);
    const float along = dot (wi, centre);
    if (! (along > 0.0f) || ! (deviation > 0.0f))
        return 0.0f;
    const float variance = deviation * deviation;
    if (orders.acrossGrooveDeviation > 0.0f && orders.grooveFamilyCount > 0)
    {
        // An even mix of one Gaussian per groove family on the same plane, long along
        // that family's axis. The two standard deviations add in quadrature.
        const float wide2 = variance + static_cast<float> (orders.acrossGrooveDeviation * orders.acrossGrooveDeviation);
        const Vec3 offset = wi * (1.0f / along) - centre;
        const float offset2 = dot (offset, offset);
        float density = 0.0f;
        for (std::size_t k = 0; k < orders.grooveFamilyCount; ++k)
        {
            const float x = dot (offset, gratingFamilyAxisAround (orders, k, centre));
            const float x2 = x * x;
            density += std::exp (-x2 / (2.0f * wide2) - std::max (offset2 - x2, 0.0f) / (2.0f * variance));
        }
        return density / (static_cast<float> (orders.grooveFamilyCount) * 2.0f * kPi * deviation * std::sqrt (wide2)
                          * along * along * along);
    }
    const float offset2 = std::max (1.0f / (along * along) - 1.0f, 0.0f);
    return std::exp (-offset2 / (2.0f * variance)) / (2.0f * kPi * variance * along * along * along);
}

// The BSDF times |cos wi| toward wi: each lobe's power times its density, the metal lobe
// about the mirror and each nonzero order about its own direction. Divided by the total
// power it is also the density sampleGratingOrder draws wi with, so a light sample's
// weight is this over the light's pdf. Pit scatter adds its own through pitScatterDensity.
PT_DEVICE_FN float gratingScatterDensity (const diffractionmaterial::DiffractionOrderDistribution& orders, Vec3 ns, Vec3 wo,
                                          Vec3 wi) noexcept
{
    const Vec3 mirror = normalizeOrZero (ns * (2.0f * dot (ns, wo) / dot (ns, ns)) - wo);
    float density = gratingLobePower (orders) * gratingSpreadDensity (orders, mirror, wi);
    for (std::size_t i = 0; i < orders.orderCount; ++i)
    {
        const auto& order = orders.orders[i];
        if (order.primary == 0 && order.secondary == 0)
            continue;
        const Vec3 centre = normalizeOrZero ({ static_cast<float> (order.direction[0]), static_cast<float> (order.direction[1]),
                                               static_cast<float> (order.direction[2]) });
        density += static_cast<float> (order.reflectedPower) * gratingSpreadDensity (orders, centre, wi);
    }
    return density;
}

// Pit scatter toward wi, per unit solid angle. A plane-of-tangents density carries to the
// sphere by cos wi. value is the BSDF times |cos wi|: sinc^2 across the tracks, a Lorentzian
// along them, times the scatter's power. sampled is sampleGratingOrder's proposal for the
// same direction, which uses a Cauchy across the tracks in place of the sinc^2. Relief
// scatter adds to both, exactly, since its proposal is its value.
struct PitScatterDensity
{
    float value = 0.0f;
    float sampled = 0.0f;
};

PT_DEVICE_FN PitScatterDensity pitScatterDensity (const diffractionmaterial::DiffractionOrderDistribution& orders, Vec3 ns, Vec3 wo,
                                                  Vec3 wi) noexcept
{
    PitScatterDensity density;
    const Vec3 normal = normalizeOrZero (ns);
    const float cosine = dot (wi, normal);
    if (orders.reliefScatterBound > 0.0f)
        density.value = density.sampled = diffractionmaterial::reliefScatterDensity (
            orders, { normal.x, normal.y, normal.z }, { wo.x, wo.y, wo.z }, { wi.x, wi.y, wi.z });
    if (! (orders.pitScatterPower > 0.0f) || ! (cosine > 0.0f))
        return density;
    // The offset from the mirror's tangent is the same in wi and in -wo.
    const Vec3 offset = wi - normal * cosine + wo - normal * dot (normal, wo);
    const float acrossWidth = static_cast<float> (orders.pitAcrossWidth);
    const float alongWidth = static_cast<float> (orders.pitAlongWidth);
    const float x = dot (offset, Vec3 { static_cast<float> (orders.pitAcrossAxis[0]), static_cast<float> (orders.pitAcrossAxis[1]),
                                        static_cast<float> (orders.pitAcrossAxis[2]) }) / acrossWidth;
    const float along = dot (offset, Vec3 { static_cast<float> (orders.pitAlongAxis[0]), static_cast<float> (orders.pitAlongAxis[1]),
                                            static_cast<float> (orders.pitAlongAxis[2]) });
    const float argument = kPi * x;
    const float sinc = std::fabs (argument) > 1.0e-4f ? std::sin (argument) / argument : 1.0f;
    const float cauchy = 0.5f / (kPi * (x * x + 0.25f));
    const float lorentzian = alongWidth / (kPi * (along * along + alongWidth * alongWidth));
    const float carried = static_cast<float> (orders.pitScatterPower) * lorentzian * cosine / acrossWidth;
    density.value += carried * sinc * sinc;
    density.sampled += carried * cauchy;
    return density;
}

// Roughness spreads every order by the same Gaussian, 2 x rmsSlope. A delta here
// mirrors each edge in the environment as a hard line across the face.
PT_DEVICE_FN Vec3 gratingSpread (const diffractionmaterial::DiffractionOrderDistribution& orders, Vec2 uDirection, Vec3 centre) noexcept
{
    const float deviation = gratingLobeDeviation (orders);
    const float radius = std::sqrt (-2.0f * std::log (std::max (1.0f - uDirection.x, 1.0e-6f)));
    if (orders.acrossGrooveDeviation > 0.0f && orders.grooveFamilyCount > 0 && deviation > 0.0f)
    {
        // The turn's random number also picks the family, evenly, and stays uniform.
        const float scaled = uDirection.y * static_cast<float> (orders.grooveFamilyCount);
        const std::size_t family = std::min (static_cast<std::size_t> (scaled), orders.grooveFamilyCount - 1);
        const float turn = 2.0f * kPi * (scaled - static_cast<float> (family));
        const Vec3 axis = gratingFamilyAxisAround (orders, family, centre);
        const float wide = std::sqrt (deviation * deviation + static_cast<float> (orders.acrossGrooveDeviation * orders.acrossGrooveDeviation));
        return normalizeOrZero (centre + axis * (wide * radius * std::cos (turn))
                                + cross (centre, axis) * (deviation * radius * std::sin (turn)));
    }
    const float offset = deviation * radius;
    const float turn = 2.0f * kPi * uDirection.y;
    const Frame around = Frame::fromNormal (centre);
    return normalizeOrZero (centre + around.tangent * (offset * std::cos (turn)) + around.bitangent * (offset * std::sin (turn)));
}

// The substrate's metallic reflection about the mirror direction, carrying power.
PT_DEVICE_FN bool gratingLeaveByMetalLobe (const diffractionmaterial::DiffractionOrderDistribution& orders, Vec3 ns, Vec3 wo,
                                           Vec2 uDirection, float power, Vec3& wi, float& throughput, bool& lightSampled) noexcept
{
    wi = gratingSpread (orders, uDirection, normalizeOrZero (ns * (2.0f * dot (ns, wo) / dot (ns, ns)) - wo));
    throughput = power;
    lightSampled = true;
    return dot (wi, ns) > 0.0f && std::isfinite (throughput);
}

PT_DEVICE_FN bool sampleGratingOrder (const diffractionmaterial::DiffractionOrderDistribution& orders, Vec3 ns, Vec3 wo, float u,
                         Vec2 uDirection, Vec3& wi, float& throughput, bool& lightSampled, bool& pitScattered) noexcept
{
    diffractionmaterial::DiffractionOrderSample sample;
    lightSampled = false;
    pitScattered = false;
    const float total = orders.resolvedReflectedPower + orders.unresolvedReflectedPower;
    if (! (total > 0.0f))
        return false;
    const float choice = std::min (static_cast<float> (u), std::nextafter (1.0f, 0.0f)) * total;
    if (orders.reliefScatterBound > 0.0f)
    {
        const float reached = choice - (orders.resolvedReflectedPower + orders.pitScatterPower);
        if (reached >= 0.0f && reached < orders.reliefScatterBound)
        {
            // The band's position is uniform, so it decides the acceptance, and the weight is
            // the total power for every draw.
            const float acceptance = reached / orders.reliefScatterBound;
            const Vec3 normal = normalizeOrZero (ns);
            std::array<float, 3> leaving {};
            if (diffractionmaterial::sampleReliefScatter (orders, { normal.x, normal.y, normal.z }, { wo.x, wo.y, wo.z },
                                                          acceptance, uDirection.x, uDirection.y, leaving))
            {
                wi = normalizeOrZero ({ leaving[0], leaving[1], leaving[2] });
                throughput = total;
                pitScattered = true;
                return dot (wi, wi) > 0.0f && std::isfinite (throughput);
            }
            // A refused draw's numbers lean to the band's far ends, and the mirror's spread
            // needs them uniform, so it takes a hash of them.
            std::uint32_t hash = static_cast<std::uint32_t> (uDirection.x * 16777216.0f)
                                 ^ (static_cast<std::uint32_t> (uDirection.y * 16777216.0f) * 2654435761u)
                                 ^ (static_cast<std::uint32_t> (acceptance * 16777216.0f) * 2246822519u);
            hash = (hash ^ (hash >> 15)) * 2654435761u;
            hash = (hash ^ (hash >> 13)) * 3266489917u;
            const float first = static_cast<float> ((hash ^ (hash >> 16)) & 0xffffffu) / 16777216.0f;
            hash = (hash ^ (hash >> 15)) * 2246822519u;
            uDirection = { first, static_cast<float> ((hash ^ (hash >> 16)) & 0xffffffu) / 16777216.0f };
            return gratingLeaveByMetalLobe (orders, ns, wo, uDirection, total, wi, throughput, lightSampled);
        }
    }
    if (choice >= orders.resolvedReflectedPower + orders.pitScatterPower)
        return gratingLeaveByMetalLobe (orders, ns, wo, uDirection, static_cast<float> (total), wi, throughput, lightSampled);
    if (choice >= orders.resolvedReflectedPower)
    {
        // Across the tracks a Cauchy of half width 1 / 2 proposes for sinc^2, so the weight
        // 2 pi (x^2 + 1/4) sinc^2(x) stays within 1.6. Along them the Lorentzian is exact.
        const float x = 0.5f * std::tan (kPi * (uDirection.x - 0.5f));
        const float argument = kPi * x;
        const float sinc = std::fabs (argument) > 1.0e-4f ? std::sin (argument) / argument : 1.0f;
        const float weight = 2.0f * kPi * (x * x + 0.25f) * sinc * sinc;
        const float along = static_cast<float> (orders.pitAlongWidth) * std::tan (kPi * (uDirection.y - 0.5f));
        const float across = static_cast<float> (orders.pitAcrossWidth) * x;
        const Vec3 normal = normalizeOrZero (ns);
        const Vec3 tangent = dot (normal, wo) * normal * 2.0f - wo - normal * dot (normal, wo)
                           + Vec3 { static_cast<float> (orders.pitAcrossAxis[0]), static_cast<float> (orders.pitAcrossAxis[1]),
                                    static_cast<float> (orders.pitAcrossAxis[2]) } * across
                           + Vec3 { static_cast<float> (orders.pitAlongAxis[0]), static_cast<float> (orders.pitAlongAxis[1]),
                                    static_cast<float> (orders.pitAlongAxis[2]) } * along;
        const float tangent2 = dot (tangent, tangent);
        // Past the horizon the wave is evanescent, and its power stays in the mirror direction.
        // No light sample covers that share, so its hits keep their full weight.
        if (! (tangent2 < 1.0f))
        {
            const bool left = gratingLeaveByMetalLobe (orders, ns, wo, uDirection, static_cast<float> (total) * weight, wi, throughput, lightSampled);
            lightSampled = false;
            return left;
        }
        wi = normalizeOrZero (tangent + normal * std::sqrt (1.0f - tangent2));
        throughput = static_cast<float> (total) * weight;
        pitScattered = true;
        return dot (wi, wi) > 0.0f && std::isfinite (throughput);
    }
    if (diffractionmaterial::sampleDiffractionOrderCore (orders, std::min (choice / orders.resolvedReflectedPower, std::nextafter (1.0f, 0.0f)),
                                                         sample) != diffractionmaterial::DiffractionOrderProblem::None)
        return false;
    if (sample.order.primary == 0 && sample.order.secondary == 0)
        return gratingLeaveByMetalLobe (orders, ns, wo, uDirection, static_cast<float> (sample.throughput * total / orders.resolvedReflectedPower), wi, throughput, lightSampled);
    const Vec3 centre = normalizeOrZero ({ static_cast<float> (sample.order.direction[0]), static_cast<float> (sample.order.direction[1]),
                                           static_cast<float> (sample.order.direction[2]) });
    wi = gratingSpread (orders, uDirection, centre);
    lightSampled = true;
    // sample.throughput is the resolved power; this branch runs its share of the time.
    throughput = static_cast<float> (sample.throughput * total / orders.resolvedReflectedPower);
    // A spread that crosses the surface leaves no light, as the metal lobe does.
    return dot (wi, wi) > 0.0f && dot (wi, ns) * dot (centre, ns) > 0.0f && std::isfinite (throughput);
}

} // namespace

PT_DEVICE_ENTRY bool usesSpectralTransport (const SceneView& view, const IntegratorSettings& settings) noexcept
{
    switch (settings.spectral)
    {
        case SpectralMode::Off:
            return false;
        case SpectralMode::Automatic:
            return view.anyDispersive || ! view.gratings.empty();
        case SpectralMode::On:
            return true;
    }
    return false;
}

PT_DEVICE_ENTRY bool causticLightTracingActive (const SceneView& view, const IntegratorSettings& settings) noexcept
{
    return settings.causticLightPaths && (settings.reflectiveCaustics || settings.refractiveCaustics)
        && view.lights != nullptr && view.lights->size() > 0
        && (view.anyMediumBoundary || ! view.gratings.empty()) && std::isfinite (view.boundsRadius)
        && view.boundsRadius > 0.0f;
}

namespace
{

// The next hero of a dispersive path: the same four wavelengths, moved one
// channel along, so channel 0 is another of them.
PT_DEVICE_FN Wavelengths rotatedHero (const Wavelengths& wavelengths) noexcept
{
    Wavelengths out = wavelengths;
    for (std::size_t c = 0; c < out.nm.size(); ++c)
        out.nm[c] = wavelengths.nm[(c + 1) % out.nm.size()];
    return out;
}

// Takes one bounce of a kind, or reports that its limit is reached.
PT_DEVICE_FN bool takeBounce (PathCounters& counters, std::uint32_t& taken, std::uint32_t limit) noexcept
{
    if (taken >= limit)
    {
        ++counters.bounceLimitPaths;
        return false;
    }
    ++taken;
    return true;
}

// Hero wavelength MIS weight, 1 / mean(pdfRatio); left at 1 in RGB mode.
PT_DEVICE_FN void updateWavelengthWeight (bool spectral, const std::array<float, kSpectrumChannels>& pdfRatio,
                                          float& wavelengthWeight) noexcept
{
    if (! spectral)
        return;
    float total = 0.0f;
    for (const float r : pdfRatio)
        total += r;
    wavelengthWeight = total > 0.0f ? static_cast<float> (kSpectrumChannels) / total : 0.0f;
}

// Drops every channel but the hero's, from here to the path's end.
PT_DEVICE_FN void keepHeroOnly (Spectrum& throughput, std::array<float, kSpectrumChannels>& pdfRatio, bool& heroOnly,
                                bool spectral, float& wavelengthWeight) noexcept
{
    for (int c = 1; c < kSpectrumChannels; ++c)
    {
        throughput[c] = 0.0f;
        pdfRatio[static_cast<std::size_t> (c)] = 0.0f;
    }
    heroOnly = true;
    updateWavelengthWeight (spectral, pdfRatio, wavelengthWeight);
}

// The light tracer's paths: a first scatter on a caustic receiver by a non-delta
// lobe, then only delta scatters, then an emitter. With excludeCausticPaths their
// emitter terms are left to the light tracer.
PT_DEVICE_FN bool causticPath (bool excludeCausticPaths, bool firstScatterReceiver, bool deltaSinceFirst,
                               std::uint32_t depth) noexcept
{
    return excludeCausticPaths && firstScatterReceiver && deltaSinceFirst && depth >= 2;
}

// One of the light tracer's paths a camera path also takes, by a light sample: camera,
// receiver, one grating lobe of some width, then a light at infinity with a size. The
// two estimators share it by the balance heuristic, the light tracer's weight R / (1 + R)
// and the camera path's 1 / (1 + R), with R the light path's density over the camera
// path's, both per sample of the frame:
//   R = [p_G cos_l / (cos_c pi r^2)] [cos_1 / (p_cam p_R)]
// p_G is the grating's sampling density toward the light seen from the receiver, cos_c
// and cos_l the grating's facet cosines toward the receiver and the light, r the
// bounding sphere's radius, whose disk the light tracer emits through. p_cam is the
// camera's density for the receiver point, per area of a frame's samples, p_R the
// receiver's density toward the grating seen from the camera, cos_1 the receiver's
// facet cosine toward the grating. The light sample's density and the emission's
// direction density are taken as equal. Both sides compute R the same way from the
// path, so it stays unbiased; the approximations change only noise. A mirror-like
// receiver has a large p_R and goes to the camera path, a diffuse one to the light
// tracer. gratingShare is the first bracket, -1 when the vertex is not such a lobe.
// The camera path also reaches an environment by its scatter at the grating, which
// under a broad sky carries most of the light, so the balance runs over three
// strategies (sharedPathWeights).
PT_DEVICE_FN float gratingLobeDensity (const diffractionmaterial::DiffractionOrderDistribution& orders, Vec3 ns, Vec3 wo, Vec3 wi) noexcept
{
    const float total = static_cast<float> (orders.resolvedReflectedPower + orders.unresolvedReflectedPower);
    if (! (total > 0.0f))
        return 0.0f;
    return (gratingScatterDensity (orders, ns, wo, wi) + pitScatterDensity (orders, ns, wo, wi).sampled) / total;
}

PT_DEVICE_FN float gratingShare (const diffractionmaterial::DiffractionOrderDistribution& towardReceiver, Vec3 ns, Vec3 ng,
                                 Vec3 toReceiver, Vec3 toLight, float boundsRadius) noexcept
{
    const float total = static_cast<float> (towardReceiver.resolvedReflectedPower + towardReceiver.unresolvedReflectedPower);
    const float cosReceiver = std::fabs (dot (ng, toReceiver));
    const float diskArea = kPi * boundsRadius * boundsRadius;
    if (! (gratingLobeDeviation (towardReceiver) > 0.0f) || ! (total > 0.0f) || ! (cosReceiver > 0.0f) || ! (diskArea > 0.0f)
        || ! std::isfinite (diskArea))
        return -1.0f;
    return gratingLobeDensity (towardReceiver, ns, toReceiver, toLight) * std::fabs (dot (ng, toLight)) / (cosReceiver * diskArea);
}

// The balance heuristic over a shared path's three strategies, as densities per solid
// angle at the grating toward the light: the camera path's light sample (lightPdf), its
// scatter (lobePdf) and the light tracer (ratio * lightPdf, R above). Every side passes
// the same densities for the same path, so the three weights sum to one.
struct SharedPathWeights
{
    float light = 0.0f;
    float lobe = 0.0f;
    float tracer = 1.0f;
};

PT_DEVICE_FN SharedPathWeights sharedPathWeights (float ratio, float lightPdf, float lobePdf) noexcept
{
    SharedPathWeights out;
    if (! (ratio >= 0.0f) || ! std::isfinite (ratio))
        return out;
    const float light = lightPdf > 0.0f && std::isfinite (lightPdf) ? lightPdf : 0.0f;
    const float lobe = lobePdf > 0.0f && std::isfinite (lobePdf) ? lobePdf : 0.0f;
    const float tracer = ratio * light;
    const float total = light + lobe + tracer;
    if (! (total > 0.0f) || ! std::isfinite (total))
        return out;
    out.light = light / total;
    out.lobe = lobe / total;
    out.tracer = tracer / total;
    return out;
}

// The densities sharedPathWeights takes for light `light` toward toLight. A sun with a
// disk is no environment a scatter can hit, so it has the light sample alone, at any
// density: the weights then depend on the ratio only. Without light sampling
// (lightSampling false) the camera path has its scatter alone.
PT_DEVICE_FN void sharedLightDensities (const LightSetView& lights, std::uint32_t light, Vec3 toLight, float lobePdf,
                                        bool lightSampling, float& lightPdf, float& scatterPdf) noexcept
{
    const bool environment = lights.isEnvironment (light);
    lightPdf = ! lightSampling ? 0.0f : environment ? lights.environmentPdf (toLight) : 1.0f;
    scatterPdf = environment ? lobePdf : 0.0f;
}

// A light path's only delta vertex so far as a grating the camera path shares
// (gratingShare), with the camera path's densities there (sharedLightDensities).
struct SharedGrating
{
    float share = -1.0f;
    float lightPdf = 0.0f;
    float lobePdf = 0.0f;
};

// The second bracket of R above: cos_1 / (p_cam p_R). A receiver whose BSDF cannot
// sample the direction is the light tracer's alone.
PT_DEVICE_FN float receiverShare (float cameraDensity, float cosTowardGrating, float receiverPdf) noexcept
{
    if (! (receiverPdf > 0.0f) || ! (cameraDensity > 0.0f))
        return std::numeric_limits<float>::infinity();
    return cosTowardGrating / (cameraDensity * receiverPdf);
}

// The camera's density for the point it sees at distance d, facet cosine cosAtPoint
// toward the camera and cosine cosView off the view axis, per area of a frame's
// samples: the frame's image area at distance 1 is imageArea (world units when
// orthographic), and a camera ray's density on the image plane is 1 / imageArea.
PT_DEVICE_FN float cameraPointDensity (bool orthographic, float imageArea, float distance2, float cosView, float cosAtPoint) noexcept
{
    if (! (imageArea > 0.0f))
        return 0.0f;
    return orthographic ? cosAtPoint / imageArea : cosAtPoint / (imageArea * distance2 * cosView * cosView * cosView);
}

PT_DEVICE_FN float cameraImageArea (const PinholeCamera& camera) noexcept
{
    const float halfExtent = camera.orthographicHalfHeight > 0.0f ? camera.orthographicHalfHeight : camera.tanHalfFov;
    return 4.0f * halfExtent * halfExtent * camera.aspect;
}

// Adds a camera path's contribution to sum. bounces is the number of scattering
// events it went through.
PT_DEVICE_FN void addContribution (Spectrum& sum, bool& finite, PathCounters& counters, const IntegratorSettings& settings,
                                   bool clampIndirect, bool clampDirect, float wavelengthWeight, Spectrum contribution,
                                   std::uint32_t bounces) noexcept
{
    contribution = (contribution * wavelengthWeight).saturated();
    if (! contribution.finite())
    {
        finite = false;
        return;
    }
    if ((bounces >= 2 && clampIndirect) || (bounces < 2 && clampDirect))
    {
        const float limit = bounces >= 2 ? settings.indirectClamp : settings.directClamp;
        const float peak = contribution.maxValue();
        if (peak > limit)
        {
            contribution = contribution * (limit / peak);
            ++counters.clampedContributions;
        }
    }
    sum = sum + contribution;
}

// Continues a camera ray through the surface just hit without scattering.
PT_DEVICE_FN void passThroughCamera (Ray& ray, std::uint32_t& vertex, const Hit& hit, Vec3 position, Vec3 geometricNormal) noexcept
{
    ++vertex;
    ray.origin = spawnRayOrigin (position, geometricNormal, ray.direction);
    if (std::isfinite (ray.tMax))
        ray.tMax = std::max (0.0f, ray.tMax - hit.t);
    ray.tMin = 0.0f;
    ray.ignoreTriangle = hit.triangle;
}

// A light's colour, with its authored spectrum where it has one.
PT_DEVICE_FN Spectrum lightChannels (const LightSetView& lights, const Wavelengths& wavelengths, Vec3 rgb, std::uint32_t light) noexcept
{
    return withSpectralBands (upsampleRgb (rgb, wavelengths), lights, light, wavelengths);
}

// A light with shadows off reaches through every blocker.
PT_DEVICE_FN Spectrum lightShadowTransmittance (const SceneView& view, const Wavelengths& wavelengths, PathCounters& counters,
                                                std::uint32_t light, const Ray& shadow, const MediumStack& from) noexcept
{
    return (view.lights->visibility (light) & kRayShadow) != 0 ? shadowTransmittance (view, shadow, from, wavelengths, counters)
                                                               : Spectrum::filled (1.0f);
}

// One camera path at the given wavelengths. reachedDispersion is set at the
// first vertex whose channels stop sharing a direction (see tracePath below).
// gratingOrders is the caller's, as is the light path's (traceCameraSample).
PT_DEVICE_ENTRY bool tracePathAt (const SceneView& view, const IntegratorSettings& settings, const Ray& cameraRay,
                  const Device::Sampler& sampler, const Wavelengths& wavelengths, PathCounters& counters,
                  Vec3& radiance, bool excludeCausticPaths, bool& reachedDispersion, const PinholeCamera* camera,
                  diffractionmaterial::DiffractionOrderDistribution& gratingOrders) noexcept
{
    const LightSetView& lights = *view.lights;
    const bool lightSamplingForAreas = settings.strategy != LightStrategy::BsdfOnly;
    const bool clampIndirect = settings.indirectClamp > 0.0f && std::isfinite (settings.indirectClamp);
    const bool clampDirect = settings.directClamp > 0.0f && std::isfinite (settings.directClamp);
    const bool spectral = wavelengths.spectral;

    Spectrum sum;
    Spectrum throughput = Spectrum::filled (1.0f);
    if (! spectral)
        throughput[3] = 0.0f;
    // Density of the path had each channel been the hero, over the hero's.
    std::array<float, kSpectrumChannels> pdfRatio { { 1.0f, 1.0f, 1.0f, 1.0f } };
    // Hero wavelength MIS weight, 1 / mean(pdfRatio); 1 in RGB mode.
    float wavelengthWeight = 1.0f;
    bool heroOnly = false;

    MediumStack media;
    // gratingOrders holds the current vertex's, on a grating. Every read follows this path's own write.
    bool previousDelta = true;  // the camera acts as a delta "lobe"
    float previousPdf = 0.0f;
    // The kind of the current ray for ray visibility: the scatter that sent it.
    std::uint8_t rayKind = kRayCamera;
    // A grating vertex whose scatter may leave its body by one flat face and hit a lamp:
    // 1 after the scatter, 2 after crossing a face. The lamp hit weighs against planarExitTo.
    int exitStage = 0;
    Hit exitHit {};
    SurfacePoint exitPoint {};
    Vec3 exitNg {};
    Vec3 exitNs {};
    MediumStack exitMedia {};
    float exitPdf = 0.0f;
    Vec3 exitCrossing {};
    // A vertex whose scatter may go through a clear body and hit a lamp beyond it: 1 after the
    // scatter, 2 and 3 after each face. It keeps its vertex in the exit fields above.
    int slabStage = 0;
    Vec3 slabEntry {};
    // Last scattering vertex: emitter MIS densities are measured from here.
    Vec3 misOrigin = cameraRay.origin;
    std::uint32_t depth = 0;            // bounces taken
    std::uint32_t vertex = 0;           // surface interactions, including pass-throughs
    std::uint32_t transparentHits = 0;
    std::uint32_t diffuseBounces = 0;
    std::uint32_t glossyBounces = 0;
    std::uint32_t transmissionBounces = 0;
    std::uint32_t volumeBounces = 0;
    // Smallest density of the non-delta bounces taken, for Filter Glossy.
    float minRayPdf = std::numeric_limits<float>::infinity();
    bool finite = true;
    Ray ray = cameraRay;
    // The light tracer's paths: a first scatter on a caustic receiver by a
    // non-delta lobe, then only delta scatters, then an emitter. With
    // excludeCausticPaths their emitter terms are left to the light tracer.
    bool firstScatterReceiver = false;
    bool deltaSinceFirst = true;
    // The first scatter's receiverShare, for a light sample the next vertex shares
    // with the light tracer (gratingShare); 0 when there is no camera to weigh it by.
    float firstReceiverShare = 0.0f;
    // After a shared grating scatter (gratingShare), its normals and incoming direction,
    // for the environment the scattered ray may hit.
    bool sharedScatter = false;
    Vec3 sharedNs {};
    Vec3 sharedNg {};
    Vec3 sharedWo {};



    // Termination contract, the same on every device type. A path ends when it
    // escapes, when its next bounce would exceed settings.maxDepth or the
    // limit of its kind (diffuse, glossy, transmission), when Russian roulette
    // or the transparent-hit limit stops it, or when no lobe can continue it;
    // in every case it keeps the radiance gathered so far. A path ended by a
    // limit is counted in bounceLimitPaths or transparentLimitPaths.
    // Nothing here fails a frame. A path whose estimate turns NaN or infinite
    // returns false, and renderSample drops that one sample and counts it;
    // other samples and pixels render as usual. Whether a frame with dropped
    // samples is acceptable is decided by tests, not by the integrator.
    while (finite)
    {
        Hit hit;
        ++counters.closestRays;
        const bool surfaceHit = intersectVisible (view, ray, hit, rayKind);
        const Vec3 direction = normalizeOrZero (ray.direction);
        const float directionLength = length (ray.direction);
        const float segmentLength = surfaceHit ? hit.t * directionLength : std::numeric_limits<float>::infinity();
        // The medium along this segment and where in it the path scatters, if it does.
        const MediumCoefficients medium = mediumAt (view, media, wavelengths, depth > 0 || settings.volumesVisibleToCamera);
        const bool inMedium = medium.any();
        MediumEvent event;
        VolumeRng segmentRng {};
        if (inMedium)
        {
            Spectrum live;
            for (int c = 0; c < kSpectrumChannels; ++c)
                live[c] = throughput[c] > 0.0f ? 1.0f : 0.0f;
            const float u = sampler.get1D (dimensionOf (vertex, kSlotVolume));
            event = sampleMediumEvent (medium, ray.origin, direction, segmentLength, live, u);
            segmentRng.state = volumeSeed (u, ray.origin);
            // Emission along the whole segment, whatever the sample did.
            if (event.emitted.anyPositive())
                addContribution (sum, finite, counters, settings, clampIndirect, clampDirect, wavelengthWeight, throughput * event.emitted, depth);
        }

        // Sized lights along the segment, for the ray kinds each one is visible to.
        if (lights.analyticCount() > 0)
        {
            for (std::size_t k = 0; k < lights.analyticCount(); ++k)
            {
                if (! lights.analyticSeenBy (k, rayKind))
                    continue;
                Vec3 emitted;
                float lightPdf = 0.0f;
                float distance = 0.0f;
                // Visibility: the light lies on this segment.
                if (! lights.analyticHit (k, ray.origin, direction, segmentLength, emitted, lightPdf, distance))
                    continue;
                // Density and radiance as seen from the last scattering
                // vertex itself, not the offset ray origin, as next event
                // estimation measures them.
                if (! lights.analyticHit (k, misOrigin, direction, std::numeric_limits<float>::infinity(), emitted,
                                          lightPdf, distance))
                    continue;
                const float weight = previousDelta ? 1.0f : emitterHitWeight (settings.strategy, previousPdf, lightPdf);
                if (! (weight > 0.0f) || ! anyPositive (emitted) || causticPath (excludeCausticPaths, firstScatterReceiver, deltaSinceFirst, depth))
                    continue;
                Spectrum contribution = throughput * lightChannels (lights, wavelengths, emitted, lights.analyticLight (k)) * weight;
                if (inMedium)
                {
                    // Transmittance up to the light, along this segment only.
                    float along = 0.0f;
                    Vec3 ignored;
                    float ignoredPdf = 0.0f;
                    if (lights.analyticHit (k, ray.origin, direction, segmentLength, ignored, ignoredPdf, along))
                        contribution = contribution * mediumTransmittance (medium, ray.origin, direction, along, segmentRng);
                }
                addContribution (sum, finite, counters, settings, clampIndirect, clampDirect, wavelengthWeight, contribution, depth);
            }
        }

        if (inMedium)
            throughput = throughput * event.weight;

        // A scatter in the medium: a vertex with the phase function as its BSDF.
        if (event.scattered)
        {
            if (! throughput.finite())
            {
                finite = false;
                break;
            }
            if (! throughput.anyPositive())
                break;
            if (depth >= settings.maxDepth)
            {
                ++counters.bounceLimitPaths;
                break;
            }
            const Vec3 position = ray.origin + direction * event.t;
            const float g = medium.anisotropy;
            // Next event estimation from any light type, through the medium.
            LightSample light;
            if (lights.size() > 0
                && lights.sample (position, sampler.get1D (dimensionOf (vertex, kSlotLightSelect)),
                                  sampler.get2D (dimensionOf (vertex, kSlotLightPosition)), light)
                && (light.delta || lightSamplingForAreas) && anyPositive (light.radiance)
                && (lights.visibility (light.light) & kRayVolumeScatter) != 0)
            {
                const float phase = henyeyGreenstein (dot (direction, light.wi), g);
                if (phase > 0.0f)
                {
                    Ray shadow;
                    shadow.time = ray.time;
                    shadow.origin = position;
                    shadow.tMin = 0.0f;
                    if (light.infinite)
                    {
                        shadow.direction = light.wi;
                        shadow.tMax = std::numeric_limits<float>::infinity();
                    }
                    else
                    {
                        Vec3 target = light.point;
                        if (light.area)
                        {
                            const Vec3 facing = dot (light.lightNormal, light.wi) < 0.0f ? light.lightNormal : -light.lightNormal;
                            target = offsetRayOrigin (light.point, facing);
                        }
                        shadow.direction = target - shadow.origin;
                        shadow.tMax = 1.0f;
                    }
                    const Spectrum transmittance = shadowTransmittance (view, shadow, media, wavelengths, counters);
                    if (transmittance.anyPositive())
                    {
                        const float weight = light.delta || settings.strategy != LightStrategy::MultipleImportance
                                               ? 1.0f
                                               : powerHeuristic (light.pdf, phase);
                        addContribution (sum, finite, counters, settings, clampIndirect, clampDirect, wavelengthWeight, throughput * lightChannels (lights, wavelengths, light.radiance, light.light) * transmittance * (phase * weight / light.pdf),
                             depth + 1);
                    }
                }
            }
            if (depth >= settings.rouletteStartDepth)
            {
                const float survive = std::min (1.0f, std::sqrt (std::max (0.0f, throughput.maxValue() * wavelengthWeight)));
                if (! (survive > 0.0f))
                    break;
                if (survive < 1.0f)
                {
                    if (sampler.get1D (dimensionOf (vertex, kSlotRoulette)) >= survive)
                        break;
                    throughput = throughput * (1.0f / survive);
                }
            }
            if (! takeBounce (counters, volumeBounces, settings.maxVolumeBounces))
                break;
            float phasePdf = 0.0f;
            const Vec3 wi = sampleHenyeyGreenstein (direction, g, sampler.get2D (dimensionOf (vertex, kSlotDirection)), phasePdf);
            if (! (phasePdf > 0.0f))
                break;
            // The phase function is its own density, so the throughput stays.
            if (depth == 0)
                firstScatterReceiver = false;
            deltaSinceFirst = false;
            previousDelta = false;
            previousPdf = phasePdf;
            exitStage = 0;
            slabStage = 0;
            misOrigin = position;
            rayKind = kRayVolumeScatter;
            const float time = ray.time;
            ray = Ray {};
            ray.time = time;
            ray.origin = position;
            ray.direction = wi;
            ++depth;
            ++vertex;
            continue;
        }

        if (! surfaceHit)
        {
            const bool shared = sharedScatter && depth == 2 && causticPath (excludeCausticPaths, firstScatterReceiver, deltaSinceFirst, depth);
            if (lights.hasEnvironment() && (depth > 0 || settings.environmentVisibleToCamera)
                && (shared || ! causticPath (excludeCausticPaths, firstScatterReceiver, deltaSinceFirst, depth)))
            {
                const Vec3 emitted = lights.environmentRadiance (direction);
                float weight = previousDelta
                                 ? 1.0f
                                 : emitterHitWeight (settings.strategy, previousPdf, lights.environmentPdf (direction));
                if (shared)
                {
                    const float share = gratingShare (gratingOrders, sharedNs, sharedNg, sharedWo, direction, view.boundsRadius);
                    weight = share >= 0.0f
                               ? sharedPathWeights (share * firstReceiverShare, lightSamplingForAreas ? lights.environmentPdf (direction) : 0.0f,
                                                    previousPdf).lobe
                               : 0.0f;
                }
                if (weight > 0.0f && anyPositive (emitted) && throughput.anyPositive())
                    addContribution (sum, finite, counters, settings, clampIndirect, clampDirect, wavelengthWeight, throughput * upsampleRgb (emitted, wavelengths) * weight, depth);
            }
            break;
        }
        if (! throughput.anyPositive())
            break;

        const SurfacePoint point = surfaceAt (view, hit);
        const PT_GLOBAL Material& material = materialOf (view, hit.triangle);
        const float coverage = materialCoverage (material);

        // A volume-only surface: the path enters or leaves its medium unchanged.
        if (volumeOnlySurface (material))
        {
            if (++transparentHits > settings.maxTransparentHits)
            {
                ++counters.transparentLimitPaths;
                break;
            }
            const bool towardFront = dot (point.geometricNormal, direction) <= 0.0f;
            applyCrossing (media, &material, classifyBoundary (media, &material, towardFront));
            passThroughCamera (ray, vertex, hit, point.position, point.geometricNormal);
            continue;
        }

        // Emission from either face. An emitter the light set holds is the
        // light tracer's at the end of a caustic path.
        if (coverage > 0.0f && (material.emission.x != 0.0f || material.emission.y != 0.0f || material.emission.z != 0.0f)
            && ! (causticPath (excludeCausticPaths, firstScatterReceiver, deltaSinceFirst, depth) && lights.trianglePdf (hit.triangle, 1.0f, 1.0f) > 0.0f))
        {
            float weight = 1.0f;
            if (! previousDelta)
            {
                const float distance = length (point.position - misOrigin);
                const float cosAtLight = std::fabs (dot (point.geometricNormal, direction));
                weight = emitterHitWeight (settings.strategy, previousPdf,
                                           lights.trianglePdf (hit.triangle, distance, cosAtLight));
            }
            else if (exitStage == 2 && lightSamplingForAreas)
            {
                // The light sample of the grating vertex reaches this point through the face.
                PlanarExit exit;
                const float areaPdf = lights.trianglePdf (hit.triangle, 1.0f, 1.0f);
                if (areaPdf > 0.0f
                    && planarExitTo (view, exitMedia, wavelengths, exitHit, exitPoint, exitNg, exitNs, point.position, point.geometricNormal, exit)
                    // A body with more faces has other lines to this point, which no light sample takes.
                    && length (exit.point.position - exitCrossing) < 0.01f * length (exit.point.position - exitPoint.position))
                    weight = emitterHitWeight (settings.strategy, exitPdf * exit.transmission, areaPdf / exit.solidAnglePerArea);
            }
            else if (slabStage == 3 && lightSamplingForAreas)
            {
                // The light sample of the stored vertex reaches this point through both faces.
                PlanarSlab slab;
                const float areaPdf = lights.trianglePdf (hit.triangle, 1.0f, 1.0f);
                const float reach = length (point.position - exitPoint.position);
                if (areaPdf > 0.0f
                    && planarSlabTo (view, exitMedia, wavelengths, exitHit, exitPoint, exitNg, exitNs, point.position, point.geometricNormal, slab)
                    // A body with more faces has other lines to this point, which no light sample takes.
                    && length (slab.entry - slabEntry) < 0.01f * reach && length (slab.point.position - exitCrossing) < 0.01f * reach)
                    weight = emitterHitWeight (settings.strategy, exitPdf * slab.transmission, areaPdf / slab.solidAnglePerArea);
            }
            if (weight > 0.0f)
                addContribution (sum, finite, counters, settings, clampIndirect, clampDirect, wavelengthWeight, throughput * upsampleRgb (material.emission, wavelengths) * (coverage * weight), depth);
        }

        const Vec3 wo = -direction;
        const bool front = dot (point.geometricNormal, wo) >= 0.0f;
        const Vec3 ng = front ? point.geometricNormal : -point.geometricNormal;

        // Alpha cut-outs: cross with probability 1 - coverage, keeping the
        // path state, so the expected result is the coverage blend.
        if (coverage < 1.0f)
        {
            const float u = sampler.get1D (dimensionOf (vertex, kSlotTransparency));
            if (u >= coverage)
            {
                if (++transparentHits > settings.maxTransparentHits)
                {
                    ++counters.transparentLimitPaths;
                    break;
                }
                passThroughCamera (ray, vertex, hit, point.position, point.geometricNormal);
                continue;
            }
        }

        // A false interface between nested media: only the stack changes.
        const bool boundary = isMediumBoundary (material);
        bool slabVertex = false;
        BoundaryCrossing crossing;
        if (boundary)
        {
            crossing = classifyBoundary (media, &material, front);
            if (crossing.falseInterface)
            {
                if (++transparentHits > settings.maxTransparentHits)
                {
                    ++counters.transparentLimitPaths;
                    break;
                }
                applyCrossing (media, &material, crossing);
                passThroughCamera (ray, vertex, hit, point.position, point.geometricNormal);
                continue;
            }
        }

        if (depth >= settings.maxDepth)
        {
            ++counters.bounceLimitPaths;
            break;
        }

        const bool authoredShading = view.triangles[hit.triangle].authoredShading;
        Vec3 ns = front ? point.shadingNormal : -point.shadingNormal;
        if (dot (ns, wo) <= 1.0e-4f)
            ns = authoredShading ? faceViewer (ns, ng, wo) : ng;
        // Authored normals decide the side of a direction, so the BSDF keeps the
        // terminator the facet normal would zero.
        const Vec3 sideNormal = authoredShading ? ns : ng;

        // A grating's metal lobe and diffraction orders take a light sample. Without one, a
        // small bright source reaches a lobe only when a sample happens to land on it, a rare
        // hit carrying the source's whole radiance: sparks around the mirror point, and arms
        // torn into specks, more so the longer the arm spreads the same light.
        // Orders depend on wavelength, so a spectral path keeps the hero alone
        // from here on.
        const PT_GLOBAL GratingBinding* grating = gratingOf (view, hit.triangle);
        if (grating != nullptr && grating->masked
            && ! foilCoversHit (*grating, view, hit, sampler.get1D (dimensionOf (vertex, kSlotCoverage))))
            grating = nullptr;
        SpectralBsdf bsdf;
        if (grating != nullptr)
        {
            reachedDispersion = reachedDispersion || spectral;
            if (spectral && ! heroOnly)
                keepHeroOnly (throughput, pdfRatio, heroOnly, spectral, wavelengthWeight);
            if (! gratingOrdersAt (*grating, view, hit, ns, wo, wavelengths.nm[0], media.active(), wavelengths.spectral, gratingOrders))
                break;
            const float deviation = gratingLobeDeviation (gratingOrders);
            // A receiver's light through this lobe belongs to the light tracer.
            const bool lightTraced = excludeCausticPaths && firstScatterReceiver && deltaSinceFirst;
            // Right after the receiver, a lobe of some width to a sized light at infinity
            // is shared with the light tracer (gratingShare).
            const bool sharedWithLight = lightTraced && depth == 1 && firstReceiverShare > 0.0f && deviation > 0.0f;
            LightSample light;
            if ((deviation > 0.0f || gratingOrders.pitScatterPower > 0.0f || gratingOrders.reliefScatterBound > 0.0f)
                && lights.size() > 0 && (! lightTraced || sharedWithLight)
                && lights.sample (point.position, sampler.get1D (dimensionOf (vertex, kSlotLightSelect)),
                                  sampler.get2D (dimensionOf (vertex, kSlotLightPosition)), light)
                && (! lightTraced || lights.distantSized (light.light))
                && (light.delta || lightSamplingForAreas) && anyPositive (light.radiance)
                && (lights.visibility (light.light) & kRayGlossy) != 0)
            {
                PlanarExit exit;
                // Only where a scattered path could also make this crossing and go on to the lamp.
                if (light.area && depth + 1 < settings.maxDepth && glossyBounces < settings.maxGlossyBounces
                    && transmissionBounces < settings.maxTransmissionBounces
                    && planarExitTo (view, media, wavelengths, hit, point, ng, ns, light.point, light.lightNormal, exit))
                {
                    const PitScatterDensity pit = pitScatterDensity (gratingOrders, ns, wo, exit.wi);
                    const float lobes = gratingScatterDensity (gratingOrders, ns, wo, exit.wi);
                    const float density = lobes + pit.value;
                    if (density > 0.0f)
                    {
                        Ray shadow;
                        shadow.time = exit.hit.time;
                        shadow.origin = spawnFromSurface (view, exit.hit, exit.point, exit.inward, exit.inward, exit.target - exit.point.position);
                        shadow.direction = exit.target - shadow.origin;
                        shadow.tMax = 1.0f;
                        shadow.tMin = 0.0f;
                        shadow.ignoreTriangle = exit.hit.triangle;
                        const Spectrum transmittance = lightShadowTransmittance (view, wavelengths, counters, light.light, shadow, exit.outside);
                        if (transmittance.anyPositive())
                        {
                            // The light's density per solid angle from here, carried to its area and
                            // then to directions inside the body, where scatter sampling draws.
                            const Vec3 toLamp = light.point - point.position;
                            const float areaPdf = light.pdf * std::fabs (dot (light.lightNormal, light.wi)) / dot (toLamp, toLamp);
                            const float total = static_cast<float> (gratingOrders.resolvedReflectedPower + gratingOrders.unresolvedReflectedPower);
                            const float weight = settings.strategy != LightStrategy::MultipleImportance
                                                   ? 1.0f
                                                   : powerHeuristic (areaPdf / exit.solidAnglePerArea, (lobes + pit.sampled) / total * exit.transmission);
                            addContribution (sum, finite, counters, settings, clampIndirect, clampDirect, wavelengthWeight, throughput * withSpectralBands (upsampleRgb (light.radiance, wavelengths), lights, light.light, wavelengths) * exit.tint * transmittance
                                     * (density * exit.transmission * exit.solidAnglePerArea * weight / areaPdf),
                                 depth + 2);
                        }
                    }
                }
                else if (dot (light.wi, ns) > 0.0f)
                {
                    const PitScatterDensity pit = pitScatterDensity (gratingOrders, ns, wo, light.wi);
                    const float lobes = gratingScatterDensity (gratingOrders, ns, wo, light.wi);
                    const float density = lobes + pit.value;
                    if (density > 0.0f)
                    {
                        Ray shadow;
                        shadow.time = hit.time;
                        shadow.origin = spawnFromSurface (view, hit, point, ng, ns, light.wi);
                        shadow.ignoreTriangle = hit.triangle;
                        shadow.tMin = 0.0f;
                        if (light.infinite)
                        {
                            shadow.direction = light.wi;
                            shadow.tMax = std::numeric_limits<float>::infinity();
                        }
                        else
                        {
                            Vec3 target = light.point;
                            if (light.area)
                            {
                                const Vec3 facing = dot (light.lightNormal, light.wi) < 0.0f ? light.lightNormal : -light.lightNormal;
                                target = offsetRayOrigin (light.point, facing);
                            }
                            shadow.direction = target - shadow.origin;
                            shadow.tMax = 1.0f;
                        }
                        const Spectrum transmittance = lightShadowTransmittance (view, wavelengths, counters, light.light, shadow, media);
                        if (transmittance.anyPositive())
                        {
                            // Scatter sampling reaches this direction by any lobe, with its selection share.
                            // On a path the light tracer shares, the scattered path is the light
                            // tracer's, so the light sample weighs against the light tracer alone.
                            const float total = static_cast<float> (gratingOrders.resolvedReflectedPower + gratingOrders.unresolvedReflectedPower);
                            float weight = light.delta || settings.strategy != LightStrategy::MultipleImportance
                                             ? 1.0f
                                             : powerHeuristic (light.pdf, (lobes + pit.sampled) / total);
                            if (lightTraced)
                            {
                                const float share = gratingShare (gratingOrders, ns, ng, wo, light.wi, view.boundsRadius);
                                float lightPdf = 0.0f;
                                float scatterPdf = 0.0f;
                                sharedLightDensities (lights, light.light, light.wi, (lobes + pit.sampled) / total, lightSamplingForAreas, lightPdf, scatterPdf);
                                weight = share >= 0.0f ? sharedPathWeights (share * firstReceiverShare, lightPdf, scatterPdf).light : 0.0f;
                            }
                            addContribution (sum, finite, counters, settings, clampIndirect, clampDirect, wavelengthWeight, throughput * withSpectralBands (upsampleRgb (light.radiance, wavelengths), lights, light.light, wavelengths) * transmittance
                                     * (density * weight / light.pdf),
                                 depth + 1);
                        }
                    }
                }
            }
        }
        else
        {
            DielectricSetup setup;
            if (boundary)
            {
                setup = dielectricSetupFor (crossing, wavelengths);
                reachedDispersion = reachedDispersion || (spectral && setup.dispersive);
            }
            else
            {
                // Sheets and opaque surfaces: the material against the medium around it.
                const float ratio = materialIorAt (material, kDLineNm, false) / mediumIor (media.active(), kDLineNm, false);
                for (float& channel : setup.eta)
                    channel = ratio;
                setup.etaD = ratio;
            }
            const Material shaded = vertexShaded (view, hit, material);
            const float blurred = filteredGlossRoughness (shaded.roughness, settings.filterGlossy, minRayPdf);
            if (blurred > shaded.roughness)
            {
                Material filtered = shaded;
                filtered.roughness = blurred;
                buildMaterialBsdf (filtered, ns, sideNormal, wo, wavelengths, setup, bsdf);
            }
            else
                buildMaterialBsdf (shaded, ns, sideNormal, wo, wavelengths, setup, bsdf);
            if (diffuseBounces > 0 && (! settings.reflectiveCaustics || ! settings.refractiveCaustics))
                bsdf.dropCausticLobes (! settings.reflectiveCaustics, ! settings.refractiveCaustics);
            if (bsdf.lobeCount() == 0)
                break;
            if (spectral && bsdf.requiresHeroWavelength() && ! heroOnly)
                keepHeroOnly (throughput, pdfRatio, heroOnly, spectral, wavelengthWeight);

            // Only where a scattered path could also cross both faces of a clear body and go on,
            // and where the lamp hit is not the light tracer's.
            slabVertex = ! boundary && bsdf.hasNonDelta() && lights.size() > 0 && depth + 2 < settings.maxDepth
                      && settings.refractiveCaustics
                      && diffuseBounces < settings.maxDiffuseBounces && glossyBounces < settings.maxGlossyBounces
                      && transmissionBounces + 1 < settings.maxTransmissionBounces
                      && ! (excludeCausticPaths && depth == 0 && causticReceiver (material, nullptr));

            // Next event estimation: one light sample.
            if (bsdf.hasNonDelta() && lights.size() > 0)
            {
                LightSample light;
                if (lights.sample (point.position, sampler.get1D (dimensionOf (vertex, kSlotLightSelect)),
                                   sampler.get2D (dimensionOf (vertex, kSlotLightPosition)), light)
                    && (light.delta || lightSamplingForAreas) && anyPositive (light.radiance))
                {
                    const std::uint8_t lightKinds = lights.visibility (light.light);
                    const SpectralBsdfEval eval = bsdf.eval (wo, light.wi, lightKinds);
                    bool reachedStraight = false;
                    if (eval.value.anyPositive())
                    {
                        Ray shadow;
                        shadow.time = hit.time;
                        shadow.origin = spawnFromSurface (view, hit, point, ng, ns, light.wi);
                        shadow.ignoreTriangle = hit.triangle;
                        shadow.tMin = 0.0f;
                        if (light.infinite)
                        {
                            shadow.direction = light.wi;
                            shadow.tMax = std::numeric_limits<float>::infinity();
                        }
                        else
                        {
                            Vec3 target = light.point;
                            if (light.area)
                            {
                                const Vec3 facing = dot (light.lightNormal, light.wi) < 0.0f ? light.lightNormal : -light.lightNormal;
                                target = offsetRayOrigin (light.point, facing);
                            }
                            shadow.direction = target - shadow.origin;
                            shadow.tMax = 1.0f;
                        }
                        // The shadow ray starts in the medium on its side of the surface.
                        MediumStack shadowMedia = media;
                        if (boundary && dot (light.wi, sideNormal) < 0.0f)
                            applyCrossing (shadowMedia, &material, crossing);
                        const Spectrum transmittance = lightShadowTransmittance (view, wavelengths, counters, light.light, shadow, shadowMedia);
                        if (transmittance.anyPositive())
                        {
                            reachedStraight = true;
                            const float weight = light.delta || settings.strategy != LightStrategy::MultipleImportance
                                                   ? 1.0f
                                                   : powerHeuristic (light.pdf, eval.pdf);
                            addContribution (sum, finite, counters, settings, clampIndirect, clampDirect, wavelengthWeight, throughput * eval.value * withSpectralBands (upsampleRgb (light.radiance, wavelengths), lights, light.light, wavelengths) * transmittance * (weight / light.pdf),
                                 depth + 1);
                        }
                    }
                    PlanarSlab slab;
                    const bool slabLine = ! reachedStraight && slabVertex && light.area
                                       && planarSlabTo (view, media, wavelengths, hit, point, ng, ns, light.point, light.lightNormal, slab);
                    reachedDispersion = reachedDispersion || (spectral && slab.dispersive);
                    if (slabLine)
                    {
                        const SpectralBsdfEval slabEval = bsdf.eval (wo, slab.wi, lightKinds);
                        Ray shadow;
                        shadow.time = slab.hit.time;
                        shadow.origin = spawnFromSurface (view, slab.hit, slab.point, slab.inward, slab.inward, slab.target - slab.point.position);
                        shadow.direction = slab.target - shadow.origin;
                        shadow.tMax = 1.0f;
                        shadow.tMin = 0.0f;
                        shadow.ignoreTriangle = slab.hit.triangle;
                        const Spectrum transmittance = slabEval.value.anyPositive()
                                                         ? lightShadowTransmittance (view, wavelengths, counters, light.light, shadow, slab.outside)
                                                         : Spectrum {};
                        if (transmittance.anyPositive())
                        {
                            // The light's density per solid angle from here, carried to its area and then
                            // through the faces to the direction a scatter would draw.
                            const Vec3 toLamp = light.point - point.position;
                            const float areaPdf = light.pdf * std::fabs (dot (light.lightNormal, light.wi)) / dot (toLamp, toLamp);
                            const float weight = settings.strategy != LightStrategy::MultipleImportance
                                                   ? 1.0f
                                                   : powerHeuristic (areaPdf / slab.solidAnglePerArea, slabEval.pdf * slab.transmission);
                            Spectrum contribution = throughput * slabEval.value * withSpectralBands (upsampleRgb (light.radiance, wavelengths), lights, light.light, wavelengths) * slab.tint * transmittance
                                                  * (slab.transmission * slab.solidAnglePerArea * weight / areaPdf);
                            // A dispersive body's line holds for the hero alone. A scattered path keeps only
                            // the hero at its first crossing, so this term weighs as that path would.
                            if (spectral && slab.dispersive && ! heroOnly)
                            {
                                for (int c = 1; c < kSpectrumChannels; ++c)
                                    contribution[c] = 0.0f;
                                const float heroWeight = pdfRatio[0] > 0.0f ? static_cast<float> (kSpectrumChannels) / pdfRatio[0] : 0.0f;
                                contribution = contribution * (wavelengthWeight > 0.0f ? heroWeight / wavelengthWeight : 0.0f);
                            }
                            addContribution (sum, finite, counters, settings, clampIndirect, clampDirect, wavelengthWeight, contribution, depth + 3);
                        }
                    }
                }
            }
        }

        // Russian roulette on the path throughput.
        if (depth >= settings.rouletteStartDepth)
        {
            const float survive = std::min (1.0f, std::sqrt (std::max (0.0f, throughput.maxValue() * wavelengthWeight)));
            if (! (survive > 0.0f))
                break;
            if (survive < 1.0f)
            {
                if (sampler.get1D (dimensionOf (vertex, kSlotRoulette)) >= survive)
                    break;
                throughput = throughput * (1.0f / survive);
            }
        }

        Vec3 wi;
        if (grating != nullptr)
        {
            float orderThroughput = 0.0f;
            bool lightSampled = false;
            bool pitScattered = false;
            if (! sampleGratingOrder (gratingOrders, ns, wo, sampler.get1D (dimensionOf (vertex, kSlotLobe)),
                                      sampler.get2D (dimensionOf (vertex, kSlotDirection)), wi, orderThroughput, lightSampled, pitScattered))
                break;
            // A diffraction order is glossy when reflected.
            if (dot (wi, ng) < 0.0f ? ! takeBounce (counters, transmissionBounces, settings.maxTransmissionBounces)
                                    : ! takeBounce (counters, glossyBounces, settings.maxGlossyBounces))
                break;
            rayKind = dot (wi, ng) < 0.0f ? kRayTransmission : kRayGlossy;
            throughput = throughput * orderThroughput;
            // A lobe of no width is a mirror, which no light sample can reach. Pit scatter has
            // its own width, whatever the slope.
            previousDelta = ! pitScattered && (! lightSampled || ! (gratingLobeDeviation (gratingOrders) > 0.0f));
            previousPdf = 0.0f;
            if (! previousDelta)
            {
                previousPdf = (gratingScatterDensity (gratingOrders, ns, wo, wi) + pitScatterDensity (gratingOrders, ns, wo, wi).sampled)
                            / static_cast<float> (gratingOrders.resolvedReflectedPower + gratingOrders.unresolvedReflectedPower);
                minRayPdf = std::min (minRayPdf, previousPdf);
            }
            // The scatter shares its environment hit when the light sample above could.
            sharedScatter = ! previousDelta && depth == 1 && excludeCausticPaths && firstScatterReceiver && deltaSinceFirst
                         && firstReceiverShare > 0.0f && gratingLobeDeviation (gratingOrders) > 0.0f && dot (wi, ng) > 0.0f;
            sharedNs = ns;
            sharedNg = ng;
            sharedWo = wo;
            exitStage = previousDelta ? 0 : 1;
            slabStage = 0;
            if (exitStage == 1)
            {
                exitHit = hit;
                exitPoint = point;
                exitNg = ng;
                exitNs = ns;
                exitMedia = media;
                exitPdf = previousPdf;
            }
            if (depth == 0)
                firstScatterReceiver = false;
        }
        else
        {
            SpectralBsdfSample scattered;
            if (! bsdf.sample (wo, sampler.get1D (dimensionOf (vertex, kSlotLobe)),
                               sampler.get2D (dimensionOf (vertex, kSlotDirection)), scattered))
                break;
            throughput = throughput * scattered.weight;
            if (spectral)
            {
                for (std::size_t c = 0; c < pdfRatio.size(); ++c)
                    pdfRatio[c] *= scattered.channelPdfRatio[c];
                if ((scattered.flags & kBsdfHeroOnly) != 0)
                    keepHeroOnly (throughput, pdfRatio, heroOnly, spectral, wavelengthWeight);
                else
                    updateWavelengthWeight (spectral, pdfRatio, wavelengthWeight);
            }
            if (! throughput.finite())
            {
                finite = false;
                break;
            }
            if (! throughput.anyPositive())
                break;

            const bool transmitted = (scattered.flags & kBsdfTransmission) != 0;
            exitStage = exitStage == 1 && transmitted && (scattered.flags & kBsdfDelta) != 0 ? 2 : 0;
            const bool slabFace = transmitted && boundary && (scattered.flags & (kBsdfDelta | kBsdfStraight)) == kBsdfDelta;
            slabStage = (slabStage == 1 || slabStage == 2) && slabFace ? slabStage + 1 : 0;
            if (slabStage == 2)
                slabEntry = point.position;
            exitCrossing = point.position;
            if (slabVertex && (scattered.flags & kBsdfDelta) == 0)
            {
                slabStage = 1;
                exitHit = hit;
                exitPoint = point;
                exitNg = ng;
                exitNs = ns;
                exitMedia = media;
                exitPdf = scattered.pdf;
            }
            if ((scattered.flags & kBsdfStraight) != 0)
            {
                // Straight through a thin sheet: for MIS the path continues the
                // direction sampled at misOrigin.
                if (++transparentHits > settings.maxTransparentHits)
                {
                    ++counters.transparentLimitPaths;
                    break;
                }
                passThroughCamera (ray, vertex, hit, point.position, point.geometricNormal);
                continue;
            }
            // Cycles' bounce kinds: any transmission, else diffuse, else glossy
            // (delta reflection included).
            const bool kindAllowed = transmitted ? takeBounce (counters, transmissionBounces, settings.maxTransmissionBounces)
                                   : (scattered.flags & kBsdfDiffuse) != 0
                                       ? takeBounce (counters, diffuseBounces, settings.maxDiffuseBounces)
                                       : takeBounce (counters, glossyBounces, settings.maxGlossyBounces);
            if (! kindAllowed)
                break;
            rayKind = transmitted ? kRayTransmission : (scattered.flags & kBsdfDiffuse) != 0 ? kRayDiffuse : kRayGlossy;
            if (transmitted && boundary)
                applyCrossing (media, &material, crossing);
            if (depth == 0)
            {
                firstScatterReceiver = (scattered.flags & kBsdfDelta) == 0 && causticReceiver (material, nullptr);
                if (firstScatterReceiver && excludeCausticPaths && camera != nullptr)
                {
                    const Vec3 forward = -rotate (camera->rotation, Vec3 { 0.0f, 0.0f, 1.0f });
                    const Vec3 fromCamera = point.position - cameraRay.origin;
                    const float cameraDensity = cameraPointDensity (camera->orthographicHalfHeight > 0.0f, cameraImageArea (*camera),
                                                                    dot (fromCamera, fromCamera), dot (direction, forward),
                                                                    std::fabs (dot (ng, wo)));
                    firstReceiverShare = receiverShare (cameraDensity, std::fabs (dot (ng, scattered.wi)), scattered.pdf);
                }
            }
            else if ((scattered.flags & kBsdfDelta) == 0)
                deltaSinceFirst = false;
            previousDelta = (scattered.flags & kBsdfDelta) != 0;
            previousPdf = scattered.pdf;
            if (! previousDelta)
                minRayPdf = std::min (minRayPdf, scattered.pdf);
            wi = scattered.wi;
        }
        if (! throughput.finite())
        {
            finite = false;
            break;
        }
        if (! throughput.anyPositive())
            break;
        misOrigin = point.position;

        ray = Ray {};
        ray.time = hit.time;
        ray.origin = spawnFromSurface (view, hit, point, ng, ns, wi);
        ray.direction = wi;
        ray.tMin = 0.0f;
        ray.tMax = std::numeric_limits<float>::infinity();
        ray.ignoreTriangle = hit.triangle;
        ++depth;
        ++vertex;
    }

    radiance = saturatedRadiance (channelsToRgb (sum.saturated(), wavelengths));
    return finite && isFinite (radiance);
}

} // namespace

// A path that meets dispersion is traced once per wavelength with the same
// random numbers, so each of the four keeps its own direction and the sample
// holds all four instead of one (see integrator.h). The path up to the first
// dispersive vertex uses every channel alike, so whether a path gets there
// does not depend on which channel is the hero. Given that, each pass is the
// same estimator at a uniform hero, and their mean is unbiased.
PT_DEVICE_FN bool tracePathWith (const SceneView& view, const IntegratorSettings& settings, const Ray& cameraRay,
                    const Device::Sampler& sampler, PathCounters& counters, Vec3& radiance, bool excludeCausticPaths,
                    const PinholeCamera* camera, diffractionmaterial::DiffractionOrderDistribution& gratingOrders) noexcept
{
    Wavelengths wavelengths = usesSpectralTransport (view, settings)
                                  ? sampleHeroWavelengths (sampler.get1D (kWavelengthDimension))
                                  : rgbChannels();
    // One call site, as OptiX inlines every call into the raygen program: nvcc's copies of
    // this path are the kernel's compile time, on the user's first run.
    bool finite = true;
    Vec3 total;
    PT_NO_UNROLL
    for (int pass = 0; pass < kSpectrumChannels && finite; ++pass)
    {
        Vec3 estimate;
        bool reachedDispersion = false;
        finite = tracePathAt (view, settings, cameraRay, sampler, wavelengths, counters, estimate, excludeCausticPaths,
                              reachedDispersion, camera, gratingOrders);
        if (pass == 0 && ! reachedDispersion)
        {
            radiance = estimate;
            return finite;
        }
        total = total + estimate;
        wavelengths = rotatedHero (wavelengths);
    }
    radiance = saturatedRadiance (saturatedRadiance (total) * (1.0f / static_cast<float> (kSpectrumChannels)));
    return finite && isFinite (radiance);
}

PT_DEVICE_ENTRY bool tracePath (const SceneView& view, const IntegratorSettings& settings, const Ray& cameraRay,
                const Device::Sampler& sampler, PathCounters& counters, Vec3& radiance, bool excludeCausticPaths,
                const PinholeCamera* camera) noexcept
{
    diffractionmaterial::DiffractionOrderDistribution gratingOrders;
    return tracePathWith (view, settings, cameraRay, sampler, counters, radiance, excludeCausticPaths, camera, gratingOrders);
}

namespace
{

// Continues a light ray through the surface just hit without scattering.
PT_DEVICE_FN void passThroughLight (Ray& ray, std::uint32_t& vertex, const Hit& hit, Vec3 position, Vec3 geometricNormal) noexcept
{
    ++vertex;
    ray.origin = spawnRayOrigin (position, geometricNormal, ray.direction);
    ray.tMin = 0.0f;
    ray.tMax = std::numeric_limits<float>::infinity();
    ray.ignoreTriangle = hit.triangle;
}

// What a light path's camera connections read and write, the same for the whole path.
struct LightPathCamera
{
    const SceneView& view;
    const IntegratorSettings& settings;
    const PinholeCamera& camera;
    const Wavelengths& wavelengths;
    const EmissionSample& emission;
    Device::Film& film;
    PathCounters& counters;
    float pixelArea;
    bool firstPass;
    bool orthographic;
    Vec3 lensOffset;    // the light path's lens point, in camera space
    Vec3 lensPosition;  // the same point in world space
};

// Connects receiver point x, lit from toLight, to the camera. The
// pinhole's importance for a pixel of area A on the plane at distance 1
// is 1 / (A cos^4) per unit of image-plane measure, and the solid-angle
// to area change is |toCamera.ng| / d^2, so a light path adds
// flux * f |toLight.ns| * |toCamera.ng| / (|toLight.ng| A d^2 cos^3) to the
// pixel it projects into.
// A thin lens connects to one lens point per light path. Its importance has
// the pinhole's form with d and cos measured from that point, as the lens
// area cancels against the lens point's density.
PT_DEVICE_FN void connectToCamera (const LightPathCamera& target, bool reachedDispersion, const MediumStack& media,
                                   const Spectrum& throughput, float wavelengthWeight, const Hit& hit,
                                   const SurfacePoint& point, const PT_GLOBAL Material& material, Vec3 toLight,
                                   const SharedGrating& sharedGrating) noexcept
{
    const SceneView& view = target.view;
    const IntegratorSettings& settings = target.settings;
    const PinholeCamera& camera = target.camera;
    const Wavelengths& wavelengths = target.wavelengths;
    const EmissionSample& emission = target.emission;
    Device::Film& film = target.film;
    PathCounters& counters = target.counters;
    const float pixelArea = target.pixelArea;
    const bool firstPass = target.firstPass;
    const bool orthographic = target.orthographic;
    const Vec3 lensOffset = target.lensOffset;
    const Vec3 lensPosition = target.lensPosition;
    if (settings.maxDiffuseBounces == 0 || (! firstPass && ! reachedDispersion))
        return;
    float pixelX = 0.0f;
    float pixelY = 0.0f;
    float viewDepth = 0.0f;
    if ((view.triangles[hit.triangle].visibility & kRayCamera) == 0
        || ! (camera.hasLens () ? camera.projectThroughLens (point.position, lensOffset, pixelX, pixelY, viewDepth)
                                : camera.project (point.position, pixelX, pixelY, viewDepth)))
        return;
    if (! (pixelX >= 0.0f) || ! (pixelY >= 0.0f) || ! (pixelX < static_cast<float> (camera.width))
        || ! (pixelY < static_cast<float> (camera.height)) || ! (viewDepth > camera.nearPlane)
        || ! (viewDepth < camera.farPlane))
        return;
    // An orthographic camera sees the point straight back along its view axis.
    // It has no lens, so lensPosition is the camera position otherwise.
    const Vec3 toCameraVector = orthographic
        ? rotate (camera.rotation, Vec3 { 0.0f, 0.0f, 1.0f }) * viewDepth
        : lensPosition - point.position;
    const float distance2 = dot (toCameraVector, toCameraVector);
    if (! (distance2 > 0.0f) || ! std::isfinite (distance2))
        return;
    const float distance = std::sqrt (distance2);
    const Vec3 toCamera = toCameraVector * (1.0f / distance);
    const bool front = dot (point.geometricNormal, toCamera) >= 0.0f;
    const Vec3 ng = front ? point.geometricNormal : -point.geometricNormal;
    const bool authoredShading = view.triangles[hit.triangle].authoredShading;
    Vec3 ns = front ? point.shadingNormal : -point.shadingNormal;
    if (dot (ns, toCamera) <= 1.0e-4f)
        ns = authoredShading ? faceViewer (ns, ng, toCamera) : ng;
    const Vec3 sideNormal = authoredShading ? ns : ng;
    // The camera path's BSDF at this point: wo toward the camera.
    DielectricSetup setup;
    const float ratio = materialIorAt (material, kDLineNm, false) / mediumIor (media.active(), kDLineNm, false);
    for (float& channel : setup.eta)
        channel = ratio;
    setup.etaD = ratio;
    SpectralBsdf bsdf;
    buildMaterialBsdf (vertexShaded (view, hit, material), ns, sideNormal, toCamera, wavelengths, setup, bsdf);
    if (bsdf.lobeCount() == 0 || ! bsdf.hasNonDelta())
        return;
    // eval.value is f |toLight.ns|, what the camera path takes here. Both
    // remaining cosines change measure, so they read the facet.
    const SpectralBsdfEval eval = bsdf.eval (toCamera, toLight);
    const float cosLight = std::fabs (dot (ng, toLight));
    const float cosCamera = std::fabs (dot (ng, toCamera));
    if (! eval.value.anyPositive() || ! (cosLight > 1.0e-6f))
        return;
    Ray shadow;
    shadow.time = hit.time;
    shadow.origin = spawnFromSurface (view, hit, point, ng, ns, toCamera);
    shadow.direction = orthographic ? toCameraVector : lensPosition - shadow.origin;
    shadow.tMin = 0.0f;
    // Geometry nearer than the near plane is clipped from camera rays.
    shadow.tMax = std::max (0.0f, 1.0f - camera.nearPlane / viewDepth);
    shadow.ignoreTriangle = hit.triangle;
    PathCounters visibility;
    const Spectrum transmittance = shadowTransmittance (view, shadow, media, wavelengths, visibility, kRayCamera,
                                                        settings.volumesVisibleToCamera);
    counters.lightPathRays += visibility.shadowRays;
    if (! transmittance.anyPositive())
        return;
    const float cosTheta = viewDepth / distance;
    const float importance = orthographic
        ? cosCamera / (cosLight * pixelArea)
        : cosCamera / (cosLight * pixelArea * distance2 * cosTheta * cosTheta * cosTheta);
    const float passWeight = reachedDispersion ? 1.0f / static_cast<float> (kSpectrumChannels) : 1.0f;
    // A path the camera path also takes by a light sample at its grating (gratingShare).
    float shareWeight = 1.0f;
    if (sharedGrating.share >= 0.0f)
    {
        const float cameraDensity = cameraPointDensity (orthographic, pixelArea * static_cast<float> (camera.width) * static_cast<float> (camera.height),
                                                        distance2, cosTheta, cosCamera);
        shareWeight = sharedPathWeights (sharedGrating.share * receiverShare (cameraDensity, cosLight, eval.pdf), sharedGrating.lightPdf,
                                         sharedGrating.lobePdf).tracer;
    }
    const Spectrum contribution = throughput * eval.value * transmittance * (importance * wavelengthWeight * passWeight * shareWeight);
    if (! contribution.finite())
        return;
    // Signed: the colour matching functions' negative lobes are what make
    // a narrow band of wavelengths come out saturated.
    const Vec3 rgb = channelsToRgb (contribution, wavelengths);
    if (! isFinite (rgb) || (rgb.x == 0.0f && rgb.y == 0.0f && rgb.z == 0.0f))
        return;
    const auto x = std::min (static_cast<std::uint32_t> (pixelX), camera.width - 1);
    const auto y = std::min (static_cast<std::uint32_t> (pixelY), camera.height - 1);
    Device::filmSplat (film, x, y, rgb);
    ++counters.causticSplats;
    const auto type = static_cast<std::size_t> (emission.type);
    if (type < counters.causticSplatsByLight.size())
        ++counters.causticSplatsByLight[type];
}

// One caustic light path (see integrator.h): emitted by
// LightSet::sampleEmission, carried through delta lobes only, connected to
// the camera at the first non-delta surface after at least one delta
// vertex. The weight is flux, so a delta vertex multiplies it by the lobe's
// sample weight (Fresnel split, reflected order power over its selection
// probability). Its lobes carry no eta^2 radiance scaling, so the only adjoint
// factor is the shading normal's (Veach 1997, 5.3): a camera path takes
// f |toLight.ns| at a vertex, and light arrives per area by |toLight.ng| and leaves
// by |toCamera.ng|, which ns does not replace. Splats go to the film unclamped. On a pass after the
// first (firstPass false) a splat before the first dispersive vertex is left
// out, as the first pass already added it at full weight; a splat past that
// vertex carries a quarter of the weight on every pass.
// Ray visibility, read backwards from the camera path this light path stands
// for: the receiver is a camera hit, its link to the lens a camera ray, and the
// ray that reaches the light leaves the last delta vertex, so the light needs
// the glossy bit for a reflection there and the transmission bit for a
// refraction. Segments between surfaces take the kind of the scatter at their
// far end, which is not known before the hit; an object blocks them unless it
// hides from diffuse, glossy and transmission rays alike.
PT_DEVICE_ENTRY void traceLightPathAt (const SceneView& view, const IntegratorSettings& settings, const PinholeCamera& camera,
                       const Device::Sampler& sampler, const EmissionSample& emission, const Wavelengths& wavelengths,
                       bool firstPass, Device::Film& film, PathCounters& counters, bool& reachedDispersion,
                       float time, diffractionmaterial::DiffractionOrderDistribution& gratingOrders) noexcept
{
    const LightSetView& lights = *view.lights;
    const bool spectral = wavelengths.spectral;
    Spectrum throughput = withSpectralBands (upsampleRgb (emission.weight, wavelengths), lights, emission.light, wavelengths);
    if (! spectral)
        throughput[3] = 0.0f;
    std::array<float, kSpectrumChannels> pdfRatio { { 1.0f, 1.0f, 1.0f, 1.0f } };
    float wavelengthWeight = 1.0f;
    bool heroOnly = false;

    // Pixel footprint on the image plane at distance 1, or in world units on the
    // view plane of an orthographic camera.
    const bool orthographic = camera.orthographicHalfHeight > 0.0f;
    const float halfExtent = orthographic ? camera.orthographicHalfHeight : camera.tanHalfFov;
    const float pixelArea = 4.0f * halfExtent * halfExtent * camera.aspect
                          / (static_cast<float> (camera.width) * static_cast<float> (camera.height));
    if (! (pixelArea > 0.0f) || ! std::isfinite (pixelArea))
        return;

    const Vec3 lensOffset = camera.lensPoint (sampler.get2D (kLightPathLens));
    const Vec3 lensPosition = camera.position + rotate (camera.rotation, lensOffset);
    const LightPathCamera target { view, settings, camera, wavelengths, emission, film, counters, pixelArea, firstPass,
                                   orthographic, lensOffset, lensPosition };

    MediumStack media;
    Ray ray;
    ray.origin = emission.origin;
    ray.direction = emission.direction;
    ray.tMin = 0.0f;
    ray.tMax = std::numeric_limits<float>::infinity();
    ray.ignoreTriangle = emission.ignoreTriangle;
    if ((emission.type == LightType::Directional || emission.type == LightType::Environment) && view.worldMedium != nullptr)
    {
        // Light from infinitely far crosses the world volume behind the disk it leaves.
        const MediumCoefficients world = mediumAt (view, media, wavelengths);
        VolumeRng rng { volumeSeed (sampler.get1D (kLightPathWavelength), emission.origin) };
        throughput = throughput * mediumTransmittance (world, emission.origin, -emission.direction,
                                                       std::numeric_limits<float>::infinity(), rng);
    }
    ray.time = time;
    bool firstHit = true;
    std::uint32_t deltaVertices = 0;
    // The path's only delta vertex so far, when shared; share -1 otherwise.
    SharedGrating sharedGrating;
    std::uint32_t vertex = 0;
    std::uint32_t transparentHits = 0;
    std::uint32_t glossyBounces = 0;
    std::uint32_t transmissionBounces = 0;
    // A camera path holds at most maxDepth - 1 delta vertices between its
    // first scatter and the emitter.
    const std::uint32_t maxDeltaVertices = settings.maxDepth > 0 ? settings.maxDepth - 1 : 0;



    for (;;)
    {
        Hit hit;
        ++counters.lightPathRays;
        if (! intersectVisible (view, ray, hit, kRayDiffuse | kRayGlossy | kRayTransmission))
            return;
        const Vec3 direction = normalizeOrZero (ray.direction);
        // Only the unscattered light: paths with a volume scatter are the camera path's.
        const MediumCoefficients medium = mediumAt (view, media, wavelengths);
        if (medium.majorant().anyPositive())
        {
            VolumeRng rng { volumeSeed (sampler.get1D (lightPathDimensionOf (vertex, kSlotVolume)), ray.origin) };
            throughput = throughput * mediumTransmittance (medium, ray.origin, direction, hit.t * length (ray.direction), rng);
        }
        const SurfacePoint point = surfaceAt (view, hit);
        if (! throughput.anyPositive() || ! throughput.finite())
            return;

        const PT_GLOBAL Material& material = materialOf (view, hit.triangle);
        const float coverage = materialCoverage (material);
        const Vec3 wo = -direction;  // toward where the light came from
        const bool front = dot (point.geometricNormal, wo) >= 0.0f;
        const Vec3 ng = front ? point.geometricNormal : -point.geometricNormal;

        if (volumeOnlySurface (material))
        {
            if (++transparentHits > settings.maxTransparentHits)
                return;
            applyCrossing (media, &material, classifyBoundary (media, &material, front));
            passThroughLight (ray, vertex, hit, point.position, point.geometricNormal);
            continue;
        }

        if (coverage < 1.0f && sampler.get1D (lightPathDimensionOf (vertex, kSlotTransparency)) >= coverage)
        {
            if (++transparentHits > settings.maxTransparentHits)
                return;
            passThroughLight (ray, vertex, hit, point.position, point.geometricNormal);
            continue;
        }

        const bool boundary = isMediumBoundary (material);
        BoundaryCrossing crossing;
        if (boundary)
        {
            crossing = classifyBoundary (media, &material, front);
            if (crossing.falseInterface)
            {
                if (++transparentHits > settings.maxTransparentHits)
                    return;
                applyCrossing (media, &material, crossing);
                passThroughLight (ray, vertex, hit, point.position, point.geometricNormal);
                continue;
            }
        }

        if (firstHit)
        {
            // Point and spot lights fall off as the raster's do, measured at
            // the first scattering vertex as the camera path measures them.
            throughput = throughput * lights.emissionFalloff (emission, point.position);
            firstHit = false;
            if (! throughput.anyPositive() || ! throughput.finite())
                return;
        }

        const bool authoredShading = view.triangles[hit.triangle].authoredShading;
        Vec3 ns = front ? point.shadingNormal : -point.shadingNormal;
        if (dot (ns, wo) <= 1.0e-4f)
            ns = authoredShading ? faceViewer (ns, ng, wo) : ng;
        // Authored normals decide the side of a direction, so the BSDF keeps the
        // terminator the facet normal would zero.
        const Vec3 sideNormal = authoredShading ? ns : ng;

        Vec3 wi;
        SharedGrating vertexShare;
        const PT_GLOBAL GratingBinding* grating = gratingOf (view, hit.triangle);
        if (grating != nullptr && grating->masked
            && ! foilCoversHit (*grating, view, hit, sampler.get1D (lightPathDimensionOf (vertex, kSlotCoverage))))
            grating = nullptr;
        if (grating != nullptr)
        {
            if (deltaVertices >= maxDeltaVertices)
                return;
            reachedDispersion = reachedDispersion || spectral;
            if (spectral && ! heroOnly)
                keepHeroOnly (throughput, pdfRatio, heroOnly, spectral, wavelengthWeight);
            float orderThroughput = 0.0f;
            // The authority's incident direction points away from the
            // surface, toward the source: wo here.
            bool lightSampled = false;
            bool pitScattered = false;
            if (! gratingOrdersAt (*grating, view, hit, ns, wo, wavelengths.nm[0], media.active(), wavelengths.spectral, gratingOrders)
                || ! sampleGratingOrder (gratingOrders, ns, wo, sampler.get1D (lightPathDimensionOf (vertex, kSlotLobe)),
                                         sampler.get2D (lightPathDimensionOf (vertex, kSlotDirection)), wi, orderThroughput, lightSampled, pitScattered))
                return;
            if (dot (wi, ng) < 0.0f ? ++transmissionBounces > settings.maxTransmissionBounces || ! settings.refractiveCaustics
                                    : ++glossyBounces > settings.maxGlossyBounces || ! settings.reflectiveCaustics)
                return;
            if (deltaVertices == 0 && (lights.visibility (emission.light) & (dot (wi, ng) < 0.0f ? kRayTransmission : kRayGlossy)) == 0)
                return;
            throughput = throughput * orderThroughput;
            // The orders as the camera path meets them, from the receiver, for gratingShare. A
            // delta order is no lobe the camera path samples or a light sample reaches. They
            // take the sampled orders' place, which nothing reads again.
            diffractionmaterial::DiffractionOrderDistribution& towardReceiver = gratingOrders;
            if (deltaVertices == 0 && (lightSampled || pitScattered) && dot (wi, ng) > 0.0f && lights.distantSized (emission.light)
                && gratingOrdersAt (*grating, view, hit, ns, wi, wavelengths.nm[0], media.active(), wavelengths.spectral, towardReceiver))
            {
                vertexShare.share = gratingShare (towardReceiver, ns, ng, wi, wo, view.boundsRadius);
                sharedLightDensities (lights, emission.light, wo, gratingLobeDensity (towardReceiver, ns, wi, wo),
                                      settings.strategy != LightStrategy::BsdfOnly, vertexShare.lightPdf, vertexShare.lobePdf);
            }
        }
        else
        {
            DielectricSetup setup;
            if (boundary)
            {
                setup = dielectricSetupFor (crossing, wavelengths);
                reachedDispersion = reachedDispersion || (spectral && setup.dispersive);
            }
            else
            {
                const float ratio = materialIorAt (material, kDLineNm, false) / mediumIor (media.active(), kDLineNm, false);
                for (float& channel : setup.eta)
                    channel = ratio;
                setup.etaD = ratio;
            }
            SpectralBsdf bsdf;
            buildMaterialBsdf (vertexShaded (view, hit, material), ns, sideNormal, wo, wavelengths, setup, bsdf);
            if (bsdf.lobeCount() == 0)
                return;
            if (spectral && bsdf.requiresHeroWavelength() && ! heroOnly)
                keepHeroOnly (throughput, pdfRatio, heroOnly, spectral, wavelengthWeight);
            if (deltaVertices > 0 && bsdf.hasNonDelta() && causticReceiver (material, nullptr))
                connectToCamera (target, reachedDispersion, media, throughput, wavelengthWeight, hit, point, material, wo,
                                 deltaVertices == 1 ? sharedGrating : SharedGrating {});

            SpectralBsdfSample scattered;
            if (! bsdf.sample (wo, sampler.get1D (lightPathDimensionOf (vertex, kSlotLobe)),
                               sampler.get2D (lightPathDimensionOf (vertex, kSlotDirection)), scattered))
                return;
            if ((scattered.flags & kBsdfStraight) != 0)
            {
                throughput = throughput * scattered.weight;
                if (++transparentHits > settings.maxTransparentHits || ! throughput.anyPositive() || ! throughput.finite())
                    return;
                passThroughLight (ray, vertex, hit, point.position, point.geometricNormal);
                continue;
            }
            // Past a non-delta scatter the path is the camera path's.
            if ((scattered.flags & kBsdfDelta) == 0 || deltaVertices >= maxDeltaVertices)
                return;
            const bool transmitted = (scattered.flags & kBsdfTransmission) != 0;
            if (transmitted ? ++transmissionBounces > settings.maxTransmissionBounces || ! settings.refractiveCaustics
                            : ++glossyBounces > settings.maxGlossyBounces || ! settings.reflectiveCaustics)
                return;
            if (deltaVertices == 0 && (lights.visibility (emission.light) & (transmitted ? kRayTransmission : kRayGlossy)) == 0)
                return;
            throughput = throughput * scattered.weight;
            if (spectral)
            {
                for (std::size_t c = 0; c < pdfRatio.size(); ++c)
                    pdfRatio[c] *= scattered.channelPdfRatio[c];
                if ((scattered.flags & kBsdfHeroOnly) != 0)
                    keepHeroOnly (throughput, pdfRatio, heroOnly, spectral, wavelengthWeight);
                else
                    updateWavelengthWeight (spectral, pdfRatio, wavelengthWeight);
            }
            if (transmitted && boundary)
                applyCrossing (media, &material, crossing);
            wi = scattered.wi;
        }
        // A straight sample never gets here, and its factor would be 1.
        const float adjointBelow = std::fabs (dot (wo, ng)) * std::fabs (dot (wi, ns));
        if (! (adjointBelow > 0.0f))
            return;
        throughput = throughput * (std::fabs (dot (wo, ns)) * std::fabs (dot (wi, ng)) / adjointBelow);
        if (! throughput.finite() || ! throughput.anyPositive())
            return;
        sharedGrating = deltaVertices == 0 ? vertexShare : SharedGrating {};
        ++deltaVertices;
        ray = Ray {};
        ray.time = hit.time;
        ray.origin = spawnFromSurface (view, hit, point, ng, ns, wi);
        ray.direction = wi;
        ray.tMin = 0.0f;
        ray.tMax = std::numeric_limits<float>::infinity();
        ray.ignoreTriangle = hit.triangle;
        ++vertex;
    }
}

// The light path's counterpart of tracePath: one pass, or four with the same
// random numbers once it meets dispersion. It counts as one light path.
PT_DEVICE_ENTRY void traceLightPath (const SceneView& view, const IntegratorSettings& settings, const PinholeCamera& camera,
                     const Device::Sampler& sampler, Device::Film& film, PathCounters& counters,
                     diffractionmaterial::DiffractionOrderDistribution& gratingOrders) noexcept
{
    ++counters.lightPaths;
    Device::filmCountLightPath (film);
    EmissionSample emission;
    if (! view.lights->sampleEmission (sampler.get1D (kLightPathSelect), sampler.get2D (kLightPathPosition),
                                       sampler.get2D (kLightPathDirection), view.boundsCentre, view.boundsRadius, emission))
        return;
    Wavelengths wavelengths = usesSpectralTransport (view, settings)
                                  ? sampleHeroWavelengths (sampler.get1D (kLightPathWavelength))
                                  : rgbChannels();
    // Under motion blur the whole light path, and the camera it connects to,
    // sit at one time in the shutter.
    const float time = sceneMoves (view, camera) ? sampler.get1D (kLightPathTime) : 0.0f;
    const PinholeCamera still = camera.at (time);
    // One call site, for the reason in tracePath.
    PT_NO_UNROLL
    for (int pass = 0; pass < kSpectrumChannels; ++pass)
    {
        bool reachedDispersion = false;
        traceLightPathAt (view, settings, still, sampler, emission, wavelengths, pass == 0, film, counters, reachedDispersion, time,
                          gratingOrders);
        if (pass == 0 && ! reachedDispersion)
            return;
        wavelengths = rotatedHero (wavelengths);
    }
}

} // namespace

PT_DEVICE_ENTRY SampleOutcome traceCameraSample (const SceneView& view, const IntegratorSettings& settings, const PinholeCamera& camera,
                                 std::uint32_t x, std::uint32_t y, std::uint32_t width, std::uint32_t sampleIndex,
                                 std::uint64_t generation, Device::Film& film, PathCounters& counters,
                                 Ray& ray, Vec3& radiance) noexcept
{
    const std::uint32_t pixel = y * width + x;
    const Device::Sampler sampler = Device::makeSampler (pixel, sampleIndex, generation, settings.pattern);
    const Vec2 jitter = sampler.get2D (kPixelFilterDimension);
    // Under motion blur each camera ray samples its own time in the shutter.
    const float time = sceneMoves (view, camera) ? sampler.get1D (kShutterTimeDimension) : 0.0f;
    const PinholeCamera now = camera.at (time);
    ray = now.generateRay (static_cast<float> (x) + jitter.x, static_cast<float> (y) + jitter.y, sampler.get2D (kLensDimension));
    ray.time = time;
    const bool lightTracing = causticLightTracingActive (view, settings);
    // A grating's orders are 8 KB. The light path is done with them before the camera
    // path starts, so the two share one, and a GPU thread's stack holds one.
    diffractionmaterial::DiffractionOrderDistribution gratingOrders;
    // Every pixel sample traces its light path, whatever its camera path
    // returns, so the splat estimator divides by the samples taken.
    if (lightTracing)
        traceLightPath (view, settings, camera, sampler, film, counters, gratingOrders);
    if (! tracePathWith (view, settings, ray, sampler, counters, radiance, lightTracing, &now, gratingOrders))
        return SampleOutcome::NonFinite;
    return SampleOutcome::Added;
}

// The denoiser's guides for one camera ray (Plan 3, Denoising): the albedo and the
// world-space shading normal, turned to face the camera, of the first surface the
// ray hits. A miss gives zeros. A clear dielectric reads as albedo 1, as OIDN and
// the OptiX denoiser expect of glass, since its look comes from what lies behind it.
PT_DEVICE_ENTRY void firstHitGuides (const SceneView& view, const Ray& cameraRay, Vec3& albedo, Vec3& normal) noexcept
{
    albedo = {};
    normal = {};
    Hit hit;
    if (! Device::intersect (view.accel, cameraRay, hit))
        return;
    const SurfacePoint point = surfaceAt (view, hit);
    const PT_GLOBAL Material& material = materialOf (view, hit.triangle);
    const float clear = dielectricWeight (material);
    albedo = Vec3 { std::min (1.0f, std::max (0.0f, material.baseColor.x + (1.0f - material.baseColor.x) * clear)),
                    std::min (1.0f, std::max (0.0f, material.baseColor.y + (1.0f - material.baseColor.y) * clear)),
                    std::min (1.0f, std::max (0.0f, material.baseColor.z + (1.0f - material.baseColor.z) * clear)) };
    normal = dot (point.shadingNormal, cameraRay.direction) > 0.0f ? -point.shadingNormal : point.shadingNormal;
}

// The Normal render output of the first surface the ray hits: the world-space
// shading normal on the front-face side, unflipped towards the camera, as the
// raster's normalize (vWorldNormal). A miss gives zeros. The denoiser's guide
// normal (firstHitGuides) faces the camera instead.
PT_DEVICE_ENTRY Vec3 firstHitWorldNormal (const SceneView& view, const Ray& cameraRay) noexcept
{
    Hit hit;
    if (! Device::intersect (view.accel, cameraRay, hit))
        return {};
    return surfaceAt (view, hit).shadingNormal;
}

PT_DEVICE_ENTRY SampleOutcome renderSample (const SceneView& view, const IntegratorSettings& settings, const PinholeCamera& camera,
                            std::uint32_t x, std::uint32_t y, std::uint32_t width, std::uint32_t sampleIndex,
                            std::uint64_t generation, Device::Film& film, PathCounters& counters) noexcept
{
    Ray ray;
    Vec3 radiance;
    const SampleOutcome outcome = traceCameraSample (view, settings, camera, x, y, width, sampleIndex, generation,
                                                     film, counters, ray, radiance);
    if (outcome == SampleOutcome::Added)
    {
        Device::filmAdd (film, x, y, radiance);
        Vec3 albedo;
        Vec3 normal;
        firstHitGuides (view, ray, albedo, normal);
        Device::filmGuide (film, x, y, albedo, normal);
    }
    return outcome;
}

} // namespace videohelper::pathtrace::device::PT_DEVICE_NAMESPACE
