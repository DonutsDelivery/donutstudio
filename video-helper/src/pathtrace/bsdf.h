#pragma once

// Principled-subset surface scattering for the CPU path tracer.
//
// A material becomes a small set of lobes at each shading point: Lambert
// diffuse, GGX reflection with Schlick Fresnel (conductor and dielectric
// specular), a dielectric interface that is either smooth (exact Fresnel,
// delta reflection and refraction) or rough (GGX transmission after Walter et
// al., "Microfacet Models for Refraction through Rough Surfaces", EGSR 2007),
// and a thin sheet (see below). GGX directions come from visible-normal
// sampling (Heitz, "Sampling the GGX Distribution of Visible Normals", JCGT
// 7(4), 2018) and shadowing is the height-correlated Smith term. One lobe is
// picked per sample with probability proportional to its albedo estimate, and
// non-delta samples are weighted by the full mixture (value sum over pdf sum).
//
// Every lobe reflects and transmits at most the energy it receives. Diffuse
// sits under the dielectric specular lobe, scaled by one minus that lobe's
// directional albedo. Refraction carries no (eta)^2 radiance factor; it
// cancels for closed volumes with the camera and lights outside, which is the
// case the raster smooth-glass path renders as well.
//
// The lobes are templates over the colour they carry: Vec3 for the RGB API
// (Lobe, Bsdf, buildMaterialBsdf with an entering flag) and Spectrum for the
// integrator's four transport channels (SpectralLobe, SpectralBsdf). Colours
// only scale the lobes, except where a dielectric's index depends on the
// wavelength (dispersion):
// - a smooth dielectric samples reflection or refraction with the hero
//   channel's Fresnel. Reflection keeps every channel, weighted by its own
//   Fresnel over the hero's, and reports that ratio as channelPdfRatio for the
//   integrator's wavelength MIS. Refraction bends each wavelength differently,
//   so only the hero continues (kBsdfHeroOnly, the other channels are 0);
// - a rough dielectric samples and evaluates with the hero's index, and the
//   whole BSDF asks for a hero-only path (requiresHeroWavelength).
//
// Thin sheet (DielectricTransport::ThinSheet) is a dielectric layer with no
// thickness, such as a window pane modelled by one polygon. Light reflects
// with the slab reflectance of two parallel interfaces with incoherent
// inter-reflection, Rs = 2R / (1 + R), and the rest goes straight through
// without bending, Ts = (1 - R) / (1 + R), tinted by the base colour. The
// transmitted direction is the incoming one, so the integrator lets shadow
// rays through it as well (straightTransmittance).
//
// Behavioural reference (studied, not copied): Blender Cycles
// intern/cycles/kernel/closure/bsdf_microfacet.h, bsdf_diffuse.h, bsdf.h and
// bsdf_util.h for lobe structure, closure selection and Fresnel handling, and
// bsdf_microfacet.h bsdf_glass_ior for the Cauchy index of a dispersive glass.

#include "spectrum.h"
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

// How light crosses the transmissive part of a material.
enum class DielectricTransport : std::uint8_t
{
    // The surface bounds a closed volume of the material: light refracts in
    // and out, and the volume is a medium on the integrator's medium stack.
    ClosedVolume = 0,
    // One open optical interface, such as a water surface: the front side is
    // whatever medium the ray is in, the back side is this material.
    Interface = 1,
    // A sheet with no thickness: reflection and straight transmission, no
    // bending, no medium.
    ThinSheet = 2
};

// Path tracer view of a surface material. Defaults equal the scene's default
// SceneMaterialRecord. The transmission fields have no scene record field; a
// caller that knows them (for example from a surface material program) sets
// them on RenderScene::materials. The defaults give the PT-A2 behaviour: a
// closed, non-dispersive volume tinted at its surface, priority 0.
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
    float ior = 1.5f;           // at the d line, 587.56 nm
    // Cauchy B in square micrometres: n(lambda) = ior + B (1 / lambda^2 -
    // 1 / lambda_d^2), the surface material IR's transmissionIorClearcoat.w.
    // 0 is non-dispersive.
    float dispersion = 0.0f;
    // Abbe number V_d, used only when dispersion is 0. 0 or less is none.
    float abbe = 0.0f;
    DielectricTransport transport = DielectricTransport::ClosedVolume;
    // Nested dielectrics: where volumes overlap, the medium with the larger
    // priority owns the overlap and the other one's surfaces inside it are
    // skipped. Equal priorities resolve by entry order: the medium entered
    // last owns the overlap.
    std::int32_t mediumPriority = 0;
    // 0: the transmitted light is tinted by the base colour at every crossing
    // of the surface. Greater than 0 (closed volumes and interfaces only):
    // Beer-Lambert absorption inside the medium instead, with the base colour
    // as the fraction of light left after this distance in scene units.
    float absorptionDistance = 0.0f;
};

// Fraction of light the surface stops: 1 opaque, 0 fully cut out. Opaque mode
// ignores opacity, Mask compares it with the cutoff, Blend uses it directly.
float materialCoverage (const Material& material) noexcept;

// (1 - metallic) * transmission, clamped: the share of the dielectric lobe.
float dielectricWeight (const Material& material) noexcept;
// A refracting boundary (closed volume or interface) with some transmission.
bool isMediumBoundary (const Material& material) noexcept;
// A thin sheet with some transmission.
bool isThinSheet (const Material& material) noexcept;
// The material's Cauchy B: dispersion, else the one its Abbe number implies.
float materialCauchyB (const Material& material) noexcept;
// True when the material refracts and its index depends on the wavelength.
bool materialDispersive (const Material& material) noexcept;
// Index of refraction at a wavelength (the d-line index when spectral is
// false), clamped to [1, 3].
float materialIorAt (const Material& material, float nm, bool spectral) noexcept;
// Beer-Lambert absorption coefficient per scene unit for each channel, 0
// when the material does not absorb by distance.
Spectrum mediumAbsorption (const Material& material, const Wavelengths& wavelengths) noexcept;
// Fraction of light a thin sheet passes straight through at |cos| = cosine,
// excluding alpha coverage. etaD is the sheet's index over the surrounding
// index at the d line. 0 for anything that is not a thin sheet.
Spectrum straightTransmittance (const Material& material, float cosine, float etaD,
                                const Wavelengths& wavelengths) noexcept;

enum BsdfFlags : std::uint32_t
{
    kBsdfReflection = 1u,
    kBsdfTransmission = 2u,
    kBsdfDiffuse = 4u,
    kBsdfGlossy = 8u,
    kBsdfDelta = 16u,
    // The sample continues the incoming ray unchanged (thin sheet).
    kBsdfStraight = 32u,
    // Only the hero channel continues; the others are 0 (dispersive refraction).
    kBsdfHeroOnly = 64u
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
    RoughDielectric = 3,
    ThinSheet = 4
};

template <class Color>
struct BasicLobe
{
    LobeKind kind = LobeKind::Lambert;
    // Lambert: albedo. GGX, dielectric and sheet: weight on reflected light.
    Color weight {};
    // Dielectric and sheet lobes only: weight on transmitted light.
    Color transmissionTint {};
    // GGX reflection only: Schlick reflectance at normal incidence.
    Color f0 {};
    // GGX alpha, the square of perceptual roughness.
    float alpha = 1.0f;
    // Dielectric lobes: index of refraction on the far side over the index on
    // the side wo is on, so 1.5 entering glass and 1 / 1.5 leaving it. For a
    // dispersive interface, the hero channel's. Sheets: the sheet's index over
    // the surrounding one.
    float eta = 1.5f;
    // Smooth dielectric with dispersive set: eta of every channel.
    std::array<float, kSpectrumChannels> channelEta { { 1.5f, 1.5f, 1.5f, 1.5f } };
    bool dispersive = false;
    // Unnormalized selection weight, the lobe's albedo estimate.
    float selectWeight = 0.0f;
};

// value is f * |cos(wi)| (already including the lobe weight), pdf is the
// solid-angle density of the sampling strategy for wi.
template <class Color>
struct BasicBsdfEval
{
    Color value {};
    float pdf = 0.0f;
};

// weight is f * |cos(wi)| / pdf for a non-delta sample, and the discrete
// lobe weight divided by its selection probability for a delta sample. pdf is
// a solid-angle density for non-delta samples and a discrete probability for
// delta samples (flags has kBsdfDelta).
template <class Color>
struct BasicBsdfSample
{
    Vec3 wi {};
    Color weight {};
    float pdf = 0.0f;
    std::uint32_t flags = 0;
    // Probability of this sample had channel i been the hero, over the hero's.
    // All 1 unless a dispersive interface reflected.
    std::array<float, kSpectrumChannels> channelPdfRatio { { 1.0f, 1.0f, 1.0f, 1.0f } };
};

using Lobe = BasicLobe<Vec3>;
using BsdfEval = BasicBsdfEval<Vec3>;
using BsdfSample = BasicBsdfSample<Vec3>;
using SpectralLobe = BasicLobe<Spectrum>;
using SpectralBsdfEval = BasicBsdfEval<Spectrum>;
using SpectralBsdfSample = BasicBsdfSample<Spectrum>;

// Lobe-level functions in the local frame: z is the shading normal and wo.z > 0.
std::uint32_t lobeFlags (const Lobe& lobe) noexcept;
std::uint32_t lobeFlags (const SpectralLobe& lobe) noexcept;
BsdfEval evalLobe (const Lobe& lobe, Vec3 wo, Vec3 wi) noexcept;
SpectralBsdfEval evalLobe (const SpectralLobe& lobe, Vec3 wo, Vec3 wi) noexcept;
// uChoice picks reflection or transmission for dielectric and sheet lobes.
bool sampleLobe (const Lobe& lobe, Vec3 wo, float uChoice, Vec2 u, BsdfSample& out) noexcept;
bool sampleLobe (const SpectralLobe& lobe, Vec3 wo, float uChoice, Vec2 u, SpectralBsdfSample& out) noexcept;

// Exact unpolarized Fresnel reflectance of a dielectric interface. cosI is the
// cosine on the incident side, eta the far index over the near index.
float fresnelDielectric (float cosI, float eta) noexcept;
// Reflectance of a thin sheet: two parallel interfaces with incoherent
// inter-reflection and no absorption, 2R / (1 + R).
float thinSheetReflectance (float cosI, float eta) noexcept;

// Directional albedo of the single-scatter GGX reflection lobe with Schlick
// Fresnel, written as albedo = f0 * scale + bias. Tabulated once on first use.
void ggxAlbedoScaleBias (float cosO, float alpha, float& scale, float& bias) noexcept;

// Smallest GGX alpha of the opaque reflection lobes, the raster BRDF's floor.
constexpr float kMinOpaqueGgxAlpha = 0.0025f;
// Below this alpha a dielectric interface is treated as perfectly smooth.
constexpr float kSmoothDielectricAlpha = 1.0e-4f;

template <class Color>
class BasicBsdf
{
public:
    static constexpr int kMaxLobes = 4;

    // Both normals unit length and on the side wo leaves from.
    void reset (Vec3 shadingNormal, Vec3 geometricNormal) noexcept;
    // Ignores a lobe with a non-positive or non-finite selection weight.
    bool addLobe (const BasicLobe<Color>& lobe) noexcept;
    // Turns selection weights into probabilities. Call after the last addLobe.
    void finalize() noexcept;

    int lobeCount() const noexcept { return count_; }
    const BasicLobe<Color>& lobe (int index) const noexcept { return lobes_[static_cast<std::size_t> (index)]; }
    float selectionProbability (int index) const noexcept { return probability_[static_cast<std::size_t> (index)]; }
    bool hasNonDelta() const noexcept { return nonDelta_; }
    const Frame& frame() const noexcept { return frame_; }
    // A dispersive rough dielectric is present: the path must carry the hero
    // wavelength only before this BSDF is evaluated or sampled.
    bool requiresHeroWavelength() const noexcept { return heroOnly_; }
    void requireHeroWavelength() noexcept { heroOnly_ = true; }

    // World-space directions, both pointing away from the surface. Delta
    // lobes contribute nothing. A direction on the wrong side of the
    // geometric normal for its lobe type (reflection below, refraction above)
    // evaluates to zero.
    BasicBsdfEval<Color> eval (Vec3 wo, Vec3 wi) const noexcept;
    bool sample (Vec3 wo, float uLobe, Vec2 uDirection, BasicBsdfSample<Color>& out) const noexcept;

private:
    BasicBsdfEval<Color> evalLocal (Vec3 woLocal, Vec3 wiLocal) const noexcept;

    Frame frame_ {};
    Vec3 geometricNormal_ { 0.0f, 0.0f, 1.0f };
    std::array<BasicLobe<Color>, kMaxLobes> lobes_ {};
    std::array<float, kMaxLobes> probability_ {};
    int count_ = 0;
    bool nonDelta_ = false;
    bool heroOnly_ = false;
};

extern template class BasicBsdf<Vec3>;
extern template class BasicBsdf<Spectrum>;

using Bsdf = BasicBsdf<Vec3>;
using SpectralBsdf = BasicBsdf<Spectrum>;

// Indices across a dielectric boundary, worked out by the caller from the
// media on both sides.
struct DielectricSetup
{
    // Far index over near index for each channel.
    std::array<float, kSpectrumChannels> eta { { 1.5f, 1.5f, 1.5f, 1.5f } };
    // The same ratio at the d line: selection weights and thin sheets use it.
    float etaD = 1.5f;
    // The channels' ratios differ (spectral mode with a dispersive medium on
    // either side).
    bool dispersive = false;
};

// Builds the lobes of a material at one shading point. shadingNormal and
// geometricNormal face wo. entering is true when wo is on the geometric front
// face, which decides the direction of refraction for transmissive materials;
// the other side is taken to be air. Base colour, metallic, roughness and
// transmission are clamped to [0, 1] and ior to [1.01, 3] here, so any input
// keeps every lobe energy conserving.
void buildMaterialBsdf (const Material& material, Vec3 shadingNormal, Vec3 geometricNormal,
                        Vec3 wo, bool entering, Bsdf& out) noexcept;

// The same for the integrator's channels: colours are upsampled for the
// wavelengths, and dielectric lobes refract with setup's ratios.
void buildMaterialBsdf (const Material& material, Vec3 shadingNormal, Vec3 geometricNormal,
                        Vec3 wo, const Wavelengths& wavelengths, const DielectricSetup& setup,
                        SpectralBsdf& out) noexcept;

} // namespace videohelper::pathtrace
