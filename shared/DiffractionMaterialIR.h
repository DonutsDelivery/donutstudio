#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace diffractionmaterial
{
// Bounded data for one reflective scalar-wave far-field grating. This schema
// contains no authored color lookup, shader source, backend object, or execution
// grant. Native backends must separately admit the exact physical subset they run.
inline constexpr std::uint32_t kWireVersion = 5;
inline constexpr std::size_t kMaximumSpectralSamples = 8;
inline constexpr std::size_t kMinimumSpectralSamples = kMaximumSpectralSamples;
inline constexpr std::uint8_t kMaximumDiffractionOrder = 8;

enum class WavelengthQualityTier : std::uint8_t
{
    Invalid = 0,
    BoundedProductionV1 = 1
};

// The only production tier. Backends evaluate these wavelengths in this order,
// then integrate CIE 1931 XYZ. There are no production 4-7 sample tiers.
inline constexpr WavelengthQualityTier kProductionWavelengthQualityTier
    = WavelengthQualityTier::BoundedProductionV1;
// The Metal path tracer's kernel (video-helper device_metal.metal) has no string_view.
#if ! defined(PATHTRACE_DEVICE_METAL)
inline constexpr std::string_view kProductionWavelengthQualityTierToken
    = "bounded-production-v1";
#endif
inline constexpr std::array<float, kMaximumSpectralSamples>
    kProductionWavelengthsNanometres {
        380.0f, 420.0f, 460.0f, 500.0f, 540.0f, 580.0f, 640.0f, 700.0f
    };
// Where the source radiance slider stops, W/(m^2 sr nm), the same 1e6 as the RGB
// intensity slider on the light node. It is a control range only. Radiance has
// no upper bound, so authored values above it stay in the graph and render. The
// float32 lighting lane is the one real limit: a value past its largest finite
// number arrives as infinity and fails the finiteness check.
inline constexpr float kSpectralRadianceControlMaximum = 1000000.0f;
inline constexpr float kMinimumWavelengthNanometres = 380.0f;
inline constexpr float kMaximumWavelengthNanometres = 700.0f;
inline constexpr float kMinimumGrooveSpacingNanometres = 100.0f;
inline constexpr float kMaximumGrooveSpacingNanometres = 10000.0f;
inline constexpr float kMinimumGrooveDepthNanometres = 0.1f;
inline constexpr float kMaximumGrooveDepthNanometres = 2000.0f;
inline constexpr float kMinimumSubstrateRefractiveIndex = 0.01f;
inline constexpr float kMaximumSubstrateRefractiveIndex = 10.0f;
inline constexpr float kMaximumExtinctionCoefficient = 20.0f;
inline constexpr float kMaximumCoatingThicknessNanometres = 1000000.0f;
inline constexpr float kMaximumRmsHeightNanometres = 1000.0f;
inline constexpr float kMaximumRmsSlope = 0.5f;
inline constexpr float kMaximumReliefRmsHeightNanometres = 100.0f;
inline constexpr float kMinimumReliefSpacingNanometres = 100.0f;
inline constexpr float kMaximumReliefSpacingNanometres = 100000.0f;
inline constexpr float kDefaultReliefSpacingNanometres = 400.0f;
inline constexpr float kMaximumStreakSlope = 0.2f;
inline constexpr float kMinimumStreakSpacingUv = 0.0005f;
inline constexpr float kMaximumStreakSpacingUv = 0.1f;
inline constexpr float kMaximumFieldOrientationDegreesPerUnit = 360.0f;

// The first physical foundation is an unpolarized scalar phase-screen model. It
// does not claim vector electromagnetics, birefringence, or polarization fidelity.
enum class GrooveProfile : std::uint8_t
{
    Invalid = 0,
    // Two terraces. grooveDepthNanometres is their height difference and
    // dutyCycle is the raised-terrace fraction.
    BinaryRectangular = 1,
    // Sinusoidal height with grooveDepthNanometres peak-to-valley. dutyCycle is
    // fixed at 0.5 because it has no independent duty parameter.
    Sinusoidal = 2,
    // Linear ramp over dutyCycle*period followed by a reset/flat land.
    // blazeAngleDegrees must equal atan(depth/(dutyCycle*period)).
    BlazedSawtooth = 3
};

enum class CoatingModel : std::uint8_t
{
    Invalid = 0,
    Uncoated = 1,
    // Coherence-averaged interface/absorption model. Thickness has physical
    // meaning, but phase is deliberately discarded: this is not thin-film color.
    IncoherentDielectric = 2
};

enum class SpectralIntegrationModel : std::uint8_t
{
    Invalid = 0,
    // Scatter every wavelength independently, integrate CIE 1931 2-degree XYZ,
    // then convert to the selected output color space at the output boundary.
    Cie1931Xyz = 1
};

enum class GratingLattice : std::uint8_t
{
    Invalid = 0,
    OneDimensional = 1,
    // Two periodic groove families at any angle. The bounded native model treats
    // their scalar phase screens as separable and evaluates paired (m,n) orders.
    CrossedTwoDimensional = 2
};

enum class GrooveFieldMode : std::uint8_t
{
    Invalid = 0,
    Constant = 1,
    Linear = 2,
    Radial = 3
};

// The Metal path tracer's kernel (video-helper device_metal.metal) has no string_view.
#if ! defined(PATHTRACE_DEVICE_METAL)
inline constexpr std::string_view token(GrooveProfile profile) noexcept
{
    switch (profile)
    {
        case GrooveProfile::BinaryRectangular: return "binary-rectangular";
        case GrooveProfile::Sinusoidal: return "sinusoidal";
        case GrooveProfile::BlazedSawtooth: return "blazed-sawtooth";
        case GrooveProfile::Invalid: break;
    }
    return {};
}

inline constexpr std::string_view token(CoatingModel model) noexcept
{
    switch (model)
    {
        case CoatingModel::Uncoated: return "uncoated";
        case CoatingModel::IncoherentDielectric: return "incoherent-dielectric";
        case CoatingModel::Invalid: break;
    }
    return {};
}

inline constexpr std::string_view token(SpectralIntegrationModel model) noexcept
{
    switch (model)
    {
        case SpectralIntegrationModel::Cie1931Xyz: return "cie-1931-xyz";
        case SpectralIntegrationModel::Invalid: break;
    }
    return {};
}

inline constexpr std::string_view token(GratingLattice lattice) noexcept
{
    switch (lattice)
    {
        case GratingLattice::OneDimensional: return "one-dimensional";
        case GratingLattice::CrossedTwoDimensional: return "crossed-two-dimensional";
        case GratingLattice::Invalid: break;
    }
    return {};
}

inline constexpr std::string_view token(GrooveFieldMode mode) noexcept
{
    switch (mode)
    {
        case GrooveFieldMode::Constant: return "constant";
        case GrooveFieldMode::Linear: return "linear";
        case GrooveFieldMode::Radial: return "radial";
        case GrooveFieldMode::Invalid: break;
    }
    return {};
}

inline constexpr std::string_view token(WavelengthQualityTier tier) noexcept
{
    return tier == WavelengthQualityTier::BoundedProductionV1
        ? kProductionWavelengthQualityTierToken : std::string_view {};
}
#endif

// Bounded analytic field evaluated from the shading sample's material UV. The
// scalar coordinate is dot(uv-origin, axis) for Linear and length(uv-origin)
// for Radial. Period deltas are nanometres per coordinate unit. Blazed profiles
// require zero period deltas because their fixed depth and blaze angle define
// one physical period. Orientation rotates both reciprocal-lattice axes together,
// preserving crossed axial order.
struct GrooveField
{
    GrooveFieldMode mode = GrooveFieldMode::Constant;
    std::array<float, 2> originUv {};
    std::array<float, 2> axisUv {};
    float grooveSpacingDeltaNanometresPerUnit = 0.0f;
    float secondarySpacingDeltaNanometresPerUnit = 0.0f;
    float orientationDegreesPerUnit = 0.0f;
};
struct GratingGeometry
{
    // Unit reciprocal-lattice direction in the material tangent plane. The
    // grooves themselves run perpendicular to this vector.
    std::array<float, 2> directionUv {};
    // Physical grating period d in nanometres.
    float grooveSpacingNanometres = 0.0f;
    GratingLattice lattice = GratingLattice::OneDimensional;
    // The 2D model admits one second reciprocal-lattice direction, at any angle
    // that isn't parallel to the first, and period. One-dimensional gratings
    // carry canonical zeros.
    std::array<float, 2> secondaryDirectionUv {};
    float secondaryGrooveSpacingNanometres = 0.0f;
    // A crossed lattice may add the third groove family its two axes imply, with
    // reciprocal vector g1/d1 - g2/d2. Its orders land on the same (m, n) grid,
    // so it changes order power, not order directions. At 60 degrees with equal
    // periods the three families are equal and the first orders form six equal arms.
    bool thirdGrooveFamily = false;
};

// Orders of the third family summed per lattice order. With maximum order 4 the
// profile index stays within 8, which the shaders' Bessel series covers.
inline constexpr int kThirdFamilyConvolutionRadius = 4;

struct GratingMicrostructure
{
    GrooveProfile profile = GrooveProfile::Invalid;
    float grooveDepthNanometres = 0.0f;
    float dutyCycle = 0.0f;
    float blazeAngleDegrees = 0.0f;
};

struct OpticalConstants
{
    // Wavelength-invariant scalar n + i*k approximation for this bounded first
    // checkpoint. The values still participate spectrally through Fresnel and
    // absorption formulas at each evaluated wavelength.
    float refractiveIndex = 0.0f;
    float extinctionCoefficient = 0.0f;
};

// Where a grating's metal gets its index. AuthoredConstants reads the scalar substrate
// above at every wavelength. A table is read per wavelength by the order evaluator only;
// the RGB preview's shaders keep the scalar.
enum class SubstrateData : std::uint8_t
{
    AuthoredConstants = 0,
    AluminiumMcPeak2015 = 1
};

// The medium a grating's orders form in. Air evaluates in vacuum, as every material has.
// SurroundingMedium reads the index of the medium the path is in at the grating.
enum class OrderMedium : std::uint8_t
{
    Air = 0,
    SurroundingMedium = 1
};

// What fills the grating's tracks. None is a continuous groove, as every material has.
// PressedDisc is a pressed disc's track: dutyCycle is the pit's share of the track pitch,
// and along the track pits and lands alternate at random, half of each.
enum class PitStatistics : std::uint8_t
{
    None = 0,
    PressedDisc = 1
};

struct SurfaceCoating
{
    CoatingModel model = CoatingModel::Invalid;
    float thicknessNanometres = 0.0f;
    OpticalConstants opticalConstants;
};

struct SurfaceRoughness
{
    // RMS physical height drives a scalar Debye-Waller coherent-order attenuation.
    // Lost coherent energy remains in the unresolved reflected-energy ledger.
    float rmsHeightNanometres = 0.0f;
    // RMS tangent-plane slope gives each discrete order a documented small-slope
    // Gaussian angular standard deviation of 2*rmsSlope without duplicating energy.
    float rmsSlope = 0.0f;
    // RMS slope across the grooves: micro-scratches along each groove family stretch
    // every order and the mirror into a line along that family's reciprocal axis, by
    // a Gaussian of standard deviation 2*acrossGrooveSlope, one equal lobe per family.
    // It is geometric, so it does not depend on wavelength. It needs rmsSlope above 0,
    // which sets the line's width. Path Trace only; other engines refuse a nonzero slope.
    float acrossGrooveSlope = 0.0f;
    // Brushed relief: RMS tilt of fine streaks that run along u, turning each path's
    // normal towards dP/dv. Unlike rmsSlope this is a surface position's own tilt, so
    // neighbouring streaks send their orders to different directions. The spacing is
    // the widest streak's width in v. Path Trace tilts per hit, Raster per fragment.
    float streakSlope = 0.0f;
    float streakSpacingUv = 0.004f;
    // Shallow relief that isn't periodic along each groove family's axis, a material
    // parameter fitted to a look: no source measures it on a real bauble. Rayleigh-Rice
    // smooth-surface scatter (Tayabaly et al. 2016, arXiv 1610.00723, Eqs. 2 to 4): each
    // wavelength scatters by its own spatial frequency, f = g / lambda. The height's
    // spectrum is flat from 0 to 1 / reliefMinimumSpacing, and this is its RMS per family,
    // 0 for none. It needs rmsSlope above 0, which sets the arm's width. Path Trace only:
    // Realtime renders without it and shows a notice, and Raster refuses a nonzero height.
    float reliefRmsHeightNanometres = 0.0f;
    float reliefMinimumSpacingNanometres = kDefaultReliefSpacingNanometres;
};

struct SpectralSampling
{
    SpectralIntegrationModel integrationModel = SpectralIntegrationModel::Invalid;
    std::uint8_t wavelengthCount = 0;
    std::array<float, kMaximumSpectralSamples> wavelengthsNanometres {};
    // Zero order is always evaluated from the same profile. These fields bound
    // the signed nonzero orders evaluated and emitted by the scalar model.
    std::uint8_t firstOrder = 0;
    std::uint8_t lastOrder = 0;
};

// One bounded graph-authored direct light. Keeping this in the shared IR lets
// the immutable imported-surface request retain physical spectral radiance all
// the way to the native diffraction pass; no RGB reconstruction is admitted.
struct SpectralIncidentLight
{
    std::array<float, 3> direction { 0.0f, 0.0f, 1.0f };
    std::array<float, kMaximumSpectralSamples> radiance {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
    };
};

// Directional beam angular integral D = integral Li dOmega, W / (m^2 nm).
// It is authored independently of legacy radiance and excludes surface cosine.
// Version zero is absent. Version one explicitly identifies this physical unit
// and declares the owning scene world coordinates in metres. It confers no
// metric or irradiance interpretation on legacy unmarked scene/light inputs.
enum class DirectionalSpectralMeasure : std::uint32_t
{
    Unspecified = 0,
    IrradianceCoefficient = 1
};
struct DirectionalSpectralIrradiance
{
    std::uint32_t version = 0;
    DirectionalSpectralMeasure measure = DirectionalSpectralMeasure::Unspecified;
    std::array<float, kMaximumSpectralSamples> coefficients {};
};

// Transient camera integration input. This is not part of Description or its
// wire encoding. A radiance sample or transported flux cannot establish surface
// irradiance without its angular measure or surface-area transport witness.
enum class DiffractionIncidentMeasure : std::uint8_t
{
    Unspecified = 0,
    SurfaceSpectralIrradiance = 1, // W / (m^2 nm), already includes incident cosine
    SpectralRadiance = 2,        // W / (m^2 sr nm), refused by the delta pixel API
    SpectralFlux = 3             // W / nm, refused without a surface footprint
};
struct DiffractionCameraIncident
{
    DiffractionIncidentMeasure measure = DiffractionIncidentMeasure::Unspecified;
    double value = 0.0;
};

struct Description
{
    std::uint32_t version = kWireVersion;
    GratingGeometry geometry;
    GrooveField grooveField;
    GratingMicrostructure microstructure;
    OpticalConstants substrate;
    SurfaceCoating coating;
    SurfaceRoughness roughness;
    SpectralSampling spectrum;
    SubstrateData substrateData = SubstrateData::AuthoredConstants;
    OrderMedium orderMedium = OrderMedium::Air;
    PitStatistics pitStatistics = PitStatistics::None;
};

struct AdmissionLimits
{
    std::size_t spectralSamples = kMaximumSpectralSamples;
    std::uint8_t diffractionOrder = kMaximumDiffractionOrder;
};
} // namespace diffractionmaterial
