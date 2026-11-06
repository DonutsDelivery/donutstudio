// Pins the diffraction authority's CPU results bit for bit. The authority moved
// into a header device code can call (diffraction_order_core.h), and a Path Trace
// frame must not change because of where it runs from. Every field of every result,
// and every refusal's text, goes into one hash. kPinned came from the float core, whose
// differences from the double one were measured when it was pinned, so a change here is
// a change of rendered light: make it on purpose.
//
// A tolerance tighter than float's rounding refuses good inputs, and a refused path is
// black with no message. checkNoSilentRefusals feeds inputs built the way the integrator
// builds them, and fails on any refusal only rounding can have caused.
//
// Usage: arbit-diffraction-order-core-tests [case count], which prints the hash.

#include "diffraction_order_bsdf.h"
#include "gpu_backend/realtime_diffraction_environment.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace
{
namespace dm = diffractionmaterial;

constexpr std::uint64_t kPinnedCases = 200000;
constexpr std::uint64_t kPinned = 0xe640b0a3e56c7ef4ull;

struct Hash
{
    std::uint64_t value = 14695981039346656037ull;
    void bytes (const void* data, std::size_t size)
    {
        const auto* p = static_cast<const unsigned char*> (data);
        for (std::size_t i = 0; i < size; ++i)
            value = (value ^ p[i]) * 1099511628211ull;
    }
    void add (float v) { bytes (&v, sizeof (v)); }
    void add (std::uint64_t v) { bytes (&v, sizeof (v)); }
    void add (const std::array<float, 3>& v) { for (const float c : v) add (c); }
    void add (const std::string& v) { bytes (v.data(), v.size()); add (static_cast<std::uint64_t> (v.size())); }
};

struct Random
{
    std::uint64_t state;
    std::uint64_t next()
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }
    double unit() { return static_cast<double> (next() >> 11) * (1.0 / 9007199254740992.0); }
    float between (float low, float high) { return low + (high - low) * static_cast<float> (unit()); }
    bool chance (double p) { return unit() < p; }
};

// Admitted materials over every model the authority branches on.
dm::Description randomDescription (Random& r)
{
    dm::Description d;
    const float turn = r.between (0.0f, 6.2831853f);
    d.geometry.directionUv = { std::cos (turn), std::sin (turn) };
    d.geometry.grooveSpacingNanometres = r.between (300.0f, 4000.0f);
    const auto choice = r.next() % 3u;
    d.microstructure.profile = choice == 0 ? dm::GrooveProfile::BinaryRectangular
                             : choice == 1 ? dm::GrooveProfile::Sinusoidal : dm::GrooveProfile::BlazedSawtooth;
    d.microstructure.grooveDepthNanometres = r.between (5.0f, 200.0f);
    d.microstructure.dutyCycle = choice == 1 ? 0.5f : r.between (0.1f, 0.9f);
    if (choice == 2)
        d.microstructure.blazeAngleDegrees = static_cast<float> (
            std::atan (static_cast<double> (d.microstructure.grooveDepthNanometres)
                       / (static_cast<double> (d.microstructure.dutyCycle) * d.geometry.grooveSpacingNanometres))
            * 57.2957795130823208768);
    if (r.chance (0.4))
    {
        d.geometry.lattice = dm::GratingLattice::CrossedTwoDimensional;
        const float second = turn + r.between (0.5f, 2.5f);
        d.geometry.secondaryDirectionUv = { std::cos (second), std::sin (second) };
        d.geometry.secondaryGrooveSpacingNanometres = choice == 2 ? d.geometry.grooveSpacingNanometres : r.between (300.0f, 4000.0f);
        d.geometry.thirdGrooveFamily = r.chance (0.5);
    }
    if (choice != 2 && r.chance (0.4))
    {
        const bool linear = r.chance (0.5);
        d.grooveField.mode = linear ? dm::GrooveFieldMode::Linear : dm::GrooveFieldMode::Radial;
        d.grooveField.originUv = { r.between (0.0f, 1.0f), r.between (0.0f, 1.0f) };
        if (linear)
            d.grooveField.axisUv = { 1.0f, 0.0f };
        d.grooveField.grooveSpacingDeltaNanometresPerUnit = r.between (-100.0f, 100.0f);
        if (d.geometry.lattice == dm::GratingLattice::CrossedTwoDimensional)
            d.grooveField.secondarySpacingDeltaNanometresPerUnit = r.between (-100.0f, 100.0f);
        d.grooveField.orientationDegreesPerUnit = r.between (-90.0f, 90.0f);
    }
    d.substrate = { r.between (0.1f, 3.0f), r.between (0.0f, 8.0f) };
    if (r.chance (0.3))
        d.substrateData = dm::SubstrateData::AluminiumMcPeak2015;
    d.coating.model = dm::CoatingModel::Uncoated;
    if (r.chance (0.4))
    {
        d.coating.model = dm::CoatingModel::IncoherentDielectric;
        d.coating.thicknessNanometres = r.between (1.0f, 500.0f);
        d.coating.opticalConstants = { r.between (1.1f, 2.5f), r.between (0.0f, 0.2f) };
    }
    d.roughness.rmsHeightNanometres = r.chance (0.5) ? r.between (0.0f, 20.0f) : 0.0f;
    d.roughness.rmsSlope = r.chance (0.5) ? r.between (0.0f, 0.1f) : 0.0f;
    d.roughness.acrossGrooveSlope = r.chance (0.3) ? r.between (0.0f, 0.1f) : 0.0f;
    d.roughness.reliefRmsHeightNanometres = r.chance (0.3) && d.roughness.rmsSlope > 0.0f ? r.between (0.0f, 30.0f) : 0.0f;
    d.roughness.reliefMinimumSpacingNanometres = r.between (200.0f, 2000.0f);
    d.spectrum.integrationModel = dm::SpectralIntegrationModel::Cie1931Xyz;
    d.spectrum.wavelengthCount = dm::kMaximumSpectralSamples;
    d.spectrum.wavelengthsNanometres = dm::kProductionWavelengthsNanometres;
    d.spectrum.firstOrder = 1;
    d.spectrum.lastOrder = static_cast<std::uint8_t> (1u + r.next() % dm::kMaximumDiffractionOrder);
    if (d.geometry.lattice == dm::GratingLattice::OneDimensional && choice == 0 && r.chance (0.3))
        d.pitStatistics = dm::PitStatistics::PressedDisc;
    return d;
}

std::array<double, 3> normalized (std::array<double, 3> v)
{
    const double size = std::sqrt (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return { v[0] / size, v[1] / size, v[2] / size };
}

std::array<float, 3> narrow (const std::array<double, 3>& v)
{
    return { static_cast<float> (v[0]), static_cast<float> (v[1]), static_cast<float> (v[2]) };
}

void addDistribution (Hash& hash, const dm::DiffractionOrderDistribution& d)
{
    hash.add (static_cast<std::uint64_t> (d.orderCount));
    for (std::size_t i = 0; i < d.orderCount; ++i)
    {
        const auto& o = d.orders[i];
        hash.add (static_cast<std::uint64_t> (static_cast<std::int64_t> (o.primary)));
        hash.add (static_cast<std::uint64_t> (static_cast<std::int64_t> (o.secondary)));
        hash.add (o.direction);
        hash.add (o.reflectedPower);
        hash.add (o.selectionProbability);
        hash.add (o.angularStandardDeviationRadians);
    }
    for (const float v : { d.substrateReflectedPower, d.resolvedReflectedPower, d.unresolvedReflectedPower,
                            d.pitScatterPower, d.pitAcrossWidth, d.pitAlongWidth, d.acrossGrooveDeviation,
                            d.reliefScatterBound, d.reliefLeavingPower, d.reliefReach, d.reliefAcrossDeviation })
        hash.add (v);
    hash.add (d.pitAcrossAxis);
    hash.add (d.pitAlongAxis);
    hash.add (static_cast<std::uint64_t> (d.grooveFamilyCount));
    for (const auto& axis : d.grooveFamilyAxes)
        hash.add (axis);
}

std::uint64_t run (std::uint64_t cases)
{
    Hash hash;
    Random r { 0x9E3779B97F4A7C15ull };
    std::uint64_t evaluated = 0, sampled = 0, refused = 0;
    while (evaluated < cases)
    {
        std::string error;
        const auto admitted = dm::admit (randomDescription (r), error);
        hash.add (error);
        if (! admitted.has_value())
        {
            ++refused;
            ++evaluated;
            continue;
        }
        for (int i = 0; i < 64 && evaluated < cases; ++i, ++evaluated)
        {
            dm::DiffractionLocalFrame frame;
            if (r.chance (0.5))
            {
                const auto t = normalized ({ r.unit() - 0.5, r.unit() - 0.5, 0.1 });
                const auto bitangent = normalized ({ -t[1], t[0], 0.0 });
                frame.bitangent = narrow (bitangent);
                frame.tangent = narrow (normalized ({ bitangent[1], -bitangent[0], 0.0 }));
            }
            // Some invalid inputs on purpose, so each refusal keeps its text.
            const double wavelength = r.chance (0.02) ? r.unit() * 1000.0 : 380.0 + r.unit() * 400.0;
            auto incident = normalized ({ r.unit() - 0.5, r.unit() - 0.5, r.unit() - (r.chance (0.03) ? 0.9 : 0.0) });
            if (r.chance (0.01))
                incident[0] += 0.5;
            const std::array<double, 2> uv { r.chance (0.01) ? 1.5 : r.unit(), r.unit() };
            const double medium = r.chance (0.3) ? 1.0 + r.unit() : 1.0;
            dm::LocalGrating local;
            const float turn = r.between (0.0f, 6.2831853f);
            local.directionUv = { std::cos (turn), std::sin (turn) };
            local.periodNanometres = static_cast<float> (r.chance (0.02) ? 50.0 : 400.0 + r.unit() * 2000.0);
            dm::DiffractionOrderDistribution result;
            error = "stale";
            const bool ok = dm::evaluateDiffractionOrders (*admitted, static_cast<float> (wavelength), frame, narrow (incident),
                                                           { static_cast<float> (uv[0]), static_cast<float> (uv[1]) }, result, error,
                                                           static_cast<float> (medium),
                                                           r.chance (0.3) ? &local : nullptr);
            hash.add (static_cast<std::uint64_t> (ok));
            hash.add (error);
            addDistribution (hash, result);
            if (! ok)
            {
                ++refused;
                continue;
            }
            dm::DiffractionOrderDistribution input = result;
            // Damaged distributions reach each of the sampler's refusals.
            if (r.chance (0.05))
            {
                const std::size_t at = r.next() % input.orderCount;
                const auto how = r.next() % 4u;
                if (how == 0) input.orders[at].selectionProbability += 0.1f;
                if (how == 1) input.orders[at].reflectedPower += 0.1f;
                if (how == 2) input.orders[at].direction[0] += 0.1f;
                if (how == 3) input.resolvedReflectedPower *= 1.5f;
            }
            const float u = static_cast<float> (r.chance (0.01) ? 1.0 : r.unit());
            dm::DiffractionOrderSample sample;
            error = "stale";
            const bool drew = dm::sampleDiffractionOrder (input, u, sample, error);
            ++sampled;
            hash.add (static_cast<std::uint64_t> (drew));
            hash.add (error);
            hash.add (static_cast<std::uint64_t> (sample.orderIndex));
            hash.add (sample.pdf);
            hash.add (sample.throughput);
            hash.add (sample.order.direction);
            hash.add (sample.order.reflectedPower);
            hash.add (sample.order.selectionProbability);
        }
    }
    std::printf ("%" PRIu64 " evaluated, %" PRIu64 " refused, %" PRIu64 " sampled\n", evaluated, refused, sampled);
    return hash.value;
}

std::array<float, 3> unitFloat (std::array<float, 3> v)
{
    const float size = std::sqrt (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return { v[0] / size, v[1] / size, v[2] / size };
}

// Frames as the integrator builds them, down to a tangent almost along the normal.
int checkNoSilentRefusals (std::uint64_t cases)
{
    using Problem = dm::DiffractionOrderProblem;
    Random r { 0xC2B2AE3D27D4EB4Full };
    std::uint64_t evaluated = 0, ok = 0, physical = 0, noEnergy = 0, wrong = 0;
    const auto axis = [&r]
    {
        return std::array<float, 3> { static_cast<float> (r.unit()) - 0.5f, static_cast<float> (r.unit()) - 0.5f,
                                      static_cast<float> (r.unit()) - 0.5f };
    };
    while (evaluated < cases)
    {
        std::string error;
        const auto admitted = dm::admit (randomDescription (r), error);
        if (! admitted.has_value())
            continue;
        for (int i = 0; i < 64 && evaluated < cases; ++i, ++evaluated)
        {
            dm::DiffractionLocalFrame frame;
            frame.normal = unitFloat (axis());
            const float tilt = std::pow (10.0f, -4.0f * static_cast<float> (r.unit()));
            const auto across = axis();
            frame.tangent = { frame.normal[0] + tilt * across[0], frame.normal[1] + tilt * across[1], frame.normal[2] + tilt * across[2] };
            if (! dm::ordercore::orthonormalTangent (frame.normal, frame.tangent))
                continue;
            frame.bitangent = dm::ordercore::cross (frame.normal, frame.tangent);
            // Well above the horizon, where only the material can decline.
            const float x = static_cast<float> (r.unit()) * 1.8f - 0.9f, y = static_cast<float> (r.unit()) * 1.8f - 0.9f;
            const float z = 0.05f + static_cast<float> (r.unit());
            std::array<float, 3> incident;
            for (std::size_t a = 0; a < 3; ++a)
                incident[a] = x * frame.tangent[a] + y * frame.bitangent[a] + z * frame.normal[a];
            incident = unitFloat (incident);
            const float wavelength = 380.0f + static_cast<float> (r.unit()) * 400.0f;
            const std::array<float, 2> uv { static_cast<float> (r.unit()), static_cast<float> (r.unit()) };
            const float medium = r.chance (0.3) ? 1.0f + static_cast<float> (r.unit()) : 1.0f;
            dm::DiffractionOrderDistribution orders;
            const auto problem = dm::evaluateDiffractionOrdersCore (admitted->description(), wavelength, frame, incident, uv, orders, medium, nullptr);
            if (problem == Problem::SinusoidalPhase || problem == Problem::GrazingIncident || problem == Problem::GrooveField)
            {
                ++physical;
                continue;
            }
            if (problem == Problem::NoResolvedEnergy)
            {
                ++noEnergy;
                continue;
            }
            if (problem != Problem::None)
            {
                if (wrong++ < 5)
                    std::printf ("FAIL: refused valid input, evaluate problem %d\n", static_cast<int> (problem));
                continue;
            }
            ++ok;
            dm::DiffractionOrderSample sample;
            // The largest sample too: rounded bins must still reach it.
            for (const float u : { static_cast<float> (r.unit()), std::nextafter (1.0f, 0.0f) })
            {
                const auto drawn = dm::sampleDiffractionOrderCore (orders, u, sample);
                if (drawn != Problem::None && wrong++ < 5)
                    std::printf ("FAIL: refused valid distribution, sample problem %d\n", static_cast<int> (drawn));
            }
        }
    }
    std::printf ("silent refusals: %" PRIu64 " wrong, %" PRIu64 " ok, %" PRIu64 " declined by the material, %" PRIu64 " no resolved energy\n",
                 wrong, ok, physical, noEnergy);
    return wrong == 0 && ok > cases / 2 ? 0 : 1;
}

// The Realtime engine's environment through the orders (realtime_diffraction_environment.h).
// A grooved surface under one small bright patch of sky: each wavelength's first order
// looks in its own direction, so only bands whose order points into the patch light up.
// The spread across bands is the spectral colour Realtime shows. A grazing order and a
// degenerate grating give finite numbers, never NaN.
int checkRealtimeEnvironmentOrders()
{
    namespace rd = arbitgpu::realtimediffraction;
    int failures = 0;
    const auto fail = [&] (const char* what) { std::printf ("FAIL: realtime environment: %s\n", what); ++failures; };

    rd::Grating grating;
    grating.profile = dm::GrooveProfile::Sinusoidal;
    grating.depthNanometres = 120.0f;
    grating.periodNanometres = 1600.0f;
    grating.direction = { 1.0f, 0.0f };
    grating.firstOrder = 1;
    grating.lastOrder = 3;
    const float tilt = 0.3f;
    const std::array<float, 3> outgoing { tilt, 0.0f, std::sqrt (1.0f - tilt * tilt) };

    std::array<float, 3> patch {};
    if (! rd::orderIncident (grating, 550.0f, 1, outgoing, patch))
        fail ("the green first order does not propagate");
    const auto sky = [&] (const std::array<float, 3>& incident)
    {
        const float cosine = incident[0] * patch[0] + incident[1] * patch[1] + incident[2] * patch[2];
        return cosine > std::cos (0.02f) ? 10.0f : 0.0f;
    };
    constexpr std::array<float, 8> wavelengths { 400.0f, 450.0f, 500.0f, 550.0f, 600.0f, 650.0f, 700.0f, 750.0f };
    std::array<float, 8> bands {};
    float brightest = 0.0f, dimmest = 1.0e30f;
    for (std::size_t b = 0; b < bands.size(); ++b)
    {
        bands[b] = rd::bandRadiance (grating, wavelengths[b], outgoing, sky);
        if (! std::isfinite (bands[b]) || bands[b] < 0.0f)
            fail ("a band is negative or not finite");
        brightest = std::max (brightest, bands[b]);
        dimmest = std::min (dimmest, bands[b]);
    }
    if (! (bands[3] > 0.0f))
        fail ("the band whose order points at the patch stays dark");
    if (! (brightest > 0.0f && dimmest < 0.1f * brightest))
        fail ("the bands show no spectral spread");
    // Turning the grooves turns the direction each order looks in, so the patch goes dark.
    auto turned = grating;
    turned.direction = { 0.0f, 1.0f };
    if (rd::bandRadiance (turned, 550.0f, outgoing, sky) >= bands[3])
        fail ("the groove direction does not steer the orders");
    // A finer pitch moves the same order away from the patch.
    auto finer = grating;
    finer.periodNanometres = 1200.0f;
    if (rd::bandRadiance (finer, 550.0f, outgoing, sky) >= bands[3])
        fail ("the groove pitch does not steer the orders");

    // A white sky reads one in every band.
    if (std::abs (rd::bandShare ({ 0.3f, 0.5f, 0.2f }, { 1.0f, 1.0f, 1.0f }) - 1.0f) > 1.0e-5f)
        fail ("a white sky is not one in a band");
    // A blue sky leaves a red band darker than a blue band.
    if (! (rd::bandShare ({ 0.6f, 0.3f, 0.0f }, { 0.1f, 0.2f, 1.0f })
           < rd::bandShare ({ 0.3f, 0.1f, 1.6f }, { 0.1f, 0.2f, 1.0f })))
        fail ("a coloured sky does not tint the bands");

    // NaN guards: an order that leaves almost along the surface, a zero period and a
    // view below the horizon each give a finite answer.
    std::array<float, 3> grazing {};
    const float edge = std::nextafter (1.0f, 0.0f);
    const std::array<float, 3> nearlyFlat { edge, 0.0f, std::sqrt (std::max (0.0f, 1.0f - edge * edge)) };
    for (int order = -3; order <= 3; ++order)
        if (rd::orderIncident (grating, 550.0f, order, nearlyFlat, grazing)
            && ! std::isfinite (rd::orderWeight (grating, 550.0f, order, grazing, nearlyFlat)))
            fail ("a grazing order weight is not finite");
    auto flat = grating;
    flat.periodNanometres = 0.0f;
    if (! std::isfinite (rd::bandRadiance (flat, 550.0f, outgoing, sky)))
        fail ("a zero period is not finite");
    if (rd::bandRadiance (grating, 550.0f, { 0.0f, 0.0f, -1.0f }, sky) != 0.0f)
        fail ("a view below the surface reads light");
    std::printf ("realtime environment orders: %s\n", failures == 0 ? "ok" : "FAILED");
    return failures;
}
} // namespace

int main (int argc, char** argv)
{
    if (checkRealtimeEnvironmentOrders() != 0)
        return 1;
    const std::uint64_t cases = argc > 1 ? std::strtoull (argv[1], nullptr, 10) : kPinnedCases;
    if (checkNoSilentRefusals (cases) != 0)
        return 1;
    const std::uint64_t value = run (cases);
    std::printf ("hash %016" PRIx64 "\n", value);
#if defined(__linux__) && defined(__x86_64__)
    // The libm of another platform may round a sine differently, which is no change here.
    if (cases == kPinnedCases && value != kPinned)
    {
        std::printf ("FAIL: expected %016" PRIx64 "\n", kPinned);
        return 1;
    }
#endif
    return 0;
}
