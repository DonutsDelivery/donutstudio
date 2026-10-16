#pragma once

// Principled-subset surface scattering for the CPU path tracer.
//
// A material becomes a small set of lobes at each shading point: Lambert
// diffuse, GGX reflection with Schlick Fresnel (conductor and dielectric
// specular), and a dielectric interface that is either smooth (exact Fresnel,
// delta reflection and refraction) or rough (GGX transmission after Walter et
// al., "Microfacet Models for Refraction through Rough Surfaces", EGSR 2007).
// GGX directions come from visible-normal sampling (Heitz, "Sampling the GGX
// Distribution of Visible Normals", JCGT 7(4), 2018) and shadowing is the
// height-correlated Smith term. One lobe is picked per sample with probability
// proportional to its albedo estimate, and non-delta samples are weighted by
// the full mixture (value sum over pdf sum).
//
// Every lobe reflects and transmits at most the energy it receives. Diffuse
// sits under the dielectric specular lobe, scaled by one minus that lobe's
// directional albedo. Refraction carries no (eta)^2 radiance factor; it
// cancels for closed volumes with the camera and lights outside, which is the
// case the raster smooth-glass path renders as well.
//
// Behavioural reference (studied, not copied): Blender Cycles
// intern/cycles/kernel/closure/bsdf_microfacet.h, bsdf_diffuse.h, bsdf.h and
// bsdf_util.h for lobe structure, closure selection and Fresnel handling.

#include "vec_math.h"

#include <array>
#include <cstdint>

namespace videohelper::pathtrace
{

constexpr float kPi = 3.14159265358979323846f;
constexpr float kInvPi = 0.31830988618379067154f;

enum class MaterialAlphaMode : std::uint8_t
{
    Opaque = 0,
    Mask = 1,
    Blend = 2
};

// Path tracer view of a surface material. Defaults equal the scene's default
// SceneMaterialRecord. transmission and ior have no scene record field yet; a
// caller that knows them (for example from a surface material program) sets
// them on RenderScene::materials.
struct Material
{
    Vec3 baseColor { 1.0f, 1.0f, 1.0f };
    float metallic = 0.0f;
    float roughness = 1.0f;
    Vec3 emission {};  // emitted radiance, scene-linear, from both faces
    float opacity = 1.0f;
    MaterialAlphaMode alphaMode = MaterialAlphaMode::Opaque;
    float alphaCutoff = 0.5f;
    float transmission = 0.0f;  // 0 opaque dielectric, 1 clear dielectric
    float ior = 1.5f;
};

// Fraction of light the surface stops: 1 opaque, 0 fully cut out. Opaque mode
// ignores opacity, Mask compares it with the cutoff, Blend uses it directly.
float materialCoverage (const Material& material) noexcept;

enum BsdfFlags : std::uint32_t
{
    kBsdfReflection = 1u,
    kBsdfTransmission = 2u,
    kBsdfDiffuse = 4u,
    kBsdfGlossy = 8u,
    kBsdfDelta = 16u
};

// Orthonormal basis around a unit normal (Duff et al., "Building an
// Orthonormal Basis, Revisited", JCGT 6(1), 2017). Local z is the normal.
struct Frame
{
    Vec3 tangent { 1.0f, 0.0f, 0.0f };
    Vec3 bitangent { 0.0f, 1.0f, 0.0f };
    Vec3 normal { 0.0f, 0.0f, 1.0f };

    static Frame fromNormal (Vec3 n) noexcept;
    Vec3 toLocal (Vec3 v) const noexcept { return { dot (v, tangent), dot (v, bitangent), dot (v, normal) }; }
    Vec3 toWorld (Vec3 v) const noexcept { return tangent * v.x + bitangent * v.y + normal * v.z; }
};

enum class LobeKind : std::uint8_t
{
    Lambert = 0,
    GgxReflection = 1,
    SmoothDielectric = 2,
    RoughDielectric = 3
};

struct Lobe
{
    LobeKind kind = LobeKind::Lambert;
    // Lambert: albedo. GGX and dielectric: weight on reflected light.
    Vec3 weight {};
    // Dielectric lobes only: weight on refracted light.
    Vec3 transmissionTint {};
    // GGX reflection only: Schlick reflectance at normal incidence.
    Vec3 f0 {};
    // GGX alpha, the square of perceptual roughness.
    float alpha = 1.0f;
    // Dielectric lobes: index of refraction on the far side over the index on
    // the side wo is on, so 1.5 entering glass and 1 / 1.5 leaving it.
    float eta = 1.5f;
    // Unnormalized selection weight, the lobe's albedo estimate.
    float selectWeight = 0.0f;
};

// value is f * |cos(wi)| (already including the lobe weight), pdf is the
// solid-angle density of the sampling strategy for wi.
struct BsdfEval
{
    Vec3 value {};
    float pdf = 0.0f;
};

// weight is f * |cos(wi)| / pdf for a non-delta sample, and the discrete
// lobe weight divided by its selection probability for a delta sample. pdf is
// a solid-angle density for non-delta samples and a discrete probability for
// delta samples (flags has kBsdfDelta).
struct BsdfSample
{
    Vec3 wi {};
    Vec3 weight {};
    float pdf = 0.0f;
    std::uint32_t flags = 0;
};

// Lobe-level functions in the local frame: z is the shading normal and wo.z > 0.
std::uint32_t lobeFlags (const Lobe& lobe) noexcept;
BsdfEval evalLobe (const Lobe& lobe, Vec3 wo, Vec3 wi) noexcept;
// uChoice picks reflection or refraction for dielectric lobes.
bool sampleLobe (const Lobe& lobe, Vec3 wo, float uChoice, Vec2 u, BsdfSample& out) noexcept;

// Exact unpolarized Fresnel reflectance of a dielectric interface. cosI is the
// cosine on the incident side, eta the far index over the near index.
float fresnelDielectric (float cosI, float eta) noexcept;

// Directional albedo of the single-scatter GGX reflection lobe with Schlick
// Fresnel, written as albedo = f0 * scale + bias. Tabulated once on first use.
void ggxAlbedoScaleBias (float cosO, float alpha, float& scale, float& bias) noexcept;

// Smallest GGX alpha of the opaque reflection lobes, the raster BRDF's floor.
constexpr float kMinOpaqueGgxAlpha = 0.0025f;
// Below this alpha a dielectric interface is treated as perfectly smooth.
constexpr float kSmoothDielectricAlpha = 1.0e-4f;

class Bsdf
{
public:
    static constexpr int kMaxLobes = 4;

    // Both normals unit length and on the side wo leaves from.
    void reset (Vec3 shadingNormal, Vec3 geometricNormal) noexcept;
    // Ignores a lobe with a non-positive or non-finite selection weight.
    bool addLobe (const Lobe& lobe) noexcept;
    // Turns selection weights into probabilities. Call after the last addLobe.
    void finalize() noexcept;

    int lobeCount() const noexcept { return count_; }
    const Lobe& lobe (int index) const noexcept { return lobes_[static_cast<std::size_t> (index)]; }
    float selectionProbability (int index) const noexcept { return probability_[static_cast<std::size_t> (index)]; }
    bool hasNonDelta() const noexcept { return nonDelta_; }
    const Frame& frame() const noexcept { return frame_; }

    // World-space directions, both pointing away from the surface. Delta
    // lobes contribute nothing. A direction on the wrong side of the
    // geometric normal for its lobe type (reflection below, refraction above)
    // evaluates to zero.
    BsdfEval eval (Vec3 wo, Vec3 wi) const noexcept;
    bool sample (Vec3 wo, float uLobe, Vec2 uDirection, BsdfSample& out) const noexcept;

private:
    BsdfEval evalLocal (Vec3 woLocal, Vec3 wiLocal) const noexcept;

    Frame frame_ {};
    Vec3 geometricNormal_ { 0.0f, 0.0f, 1.0f };
    std::array<Lobe, kMaxLobes> lobes_ {};
    std::array<float, kMaxLobes> probability_ {};
    int count_ = 0;
    bool nonDelta_ = false;
};

// Builds the lobes of a material at one shading point. shadingNormal and
// geometricNormal face wo. entering is true when wo is on the geometric front
// face, which decides the direction of refraction for transmissive materials.
// Base colour, metallic, roughness and transmission are clamped to [0, 1] and
// ior to [1.01, 3] here, so any input keeps every lobe energy conserving.
void buildMaterialBsdf (const Material& material, Vec3 shadingNormal, Vec3 geometricNormal,
                        Vec3 wo, bool entering, Bsdf& out) noexcept;

} // namespace videohelper::pathtrace
