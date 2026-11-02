// Numeric checks on the AgX curve and its admission. The scalar math is
// shared/AgxViewTransform.h; agx_view_transform_source_contract.py pins the GLSL and
// Metal constants to it, so these figures hold for what the GPU runs.
#include "../../shared/AgxViewTransform.h"
#include "../../shared/ColorTransformContract.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <iostream>

namespace
{
using agxviewtransform::Rgb;

void require(bool condition, const char* message)
{
    if (! condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

Rgb scaled(const Rgb& colour, double stops)
{
    const double gain = std::pow(2.0, stops);
    return { colour[0] * gain, colour[1] * gain, colour[2] * gain };
}

// What the Standard choice shows: over-range channels clip one at a time.
Rgb standard(const Rgb& radiance)
{
    return { std::clamp(radiance[0], 0.0, 1.0), std::clamp(radiance[1], 0.0, 1.0),
             std::clamp(radiance[2], 0.0, 1.0) };
}

double chroma(const Rgb& c)
{
    return std::max({ c[0], c[1], c[2] }) - std::min({ c[0], c[1], c[2] });
}

double hueDegrees(const Rgb& c)
{
    const double top = std::max({ c[0], c[1], c[2] });
    const double span = chroma(c);
    if (span <= 0.0) return 0.0;
    const double hue = top == c[0] ? std::fmod((c[1] - c[2]) / span + 6.0, 6.0)
        : top == c[1] ? (c[2] - c[0]) / span + 2.0 : (c[0] - c[1]) / span + 4.0;
    return hue * 60.0;
}

double hueDistance(double a, double b)
{
    const double d = std::abs(a - b);
    return std::min(d, 360.0 - d);
}

double luma(const Rgb& c)
{
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}

void testSigmoidAndNeutralAxis()
{
    double previous = agxviewtransform::sigmoid(0.0);
    for (int i = 1; i <= 100000; ++i)
    {
        const double next = agxviewtransform::sigmoid(i / 100000.0);
        require(next > previous, "the sigmoid rises across its whole domain");
        previous = next;
    }

    double previousGrey = -1.0;
    for (double stops = -16.0; stops <= 10.0; stops += 0.125)
    {
        const double value = std::pow(2.0, stops);
        const auto grey = agxviewtransform::apply({ value, value, value });
        require(chroma(grey) < 1.0e-6, "a neutral input stays neutral at every exposure");
        require(grey[1] >= previousGrey, "the neutral axis never falls as radiance rises");
        previousGrey = grey[1];
    }
    const auto middle = agxviewtransform::apply({ 0.18, 0.18, 0.18 });
    require(middle[1] > 0.20 && middle[1] < 0.23, "middle grey lands where AgX puts it");
    require(agxviewtransform::apply({ 0.0, 0.0, 0.0 })[1] == 0.0, "black stays black");
    require(agxviewtransform::apply({ -1.0, -1.0, -1.0 })[1] == 0.0, "negative radiance is black");
}

void testSaturatedHighlightsKeepColour()
{
    for (const Rgb colour : { Rgb { 1.0, 0.05, 0.02 }, Rgb { 0.05, 1.0, 0.05 }, Rgb { 0.03, 0.05, 1.0 },
                              Rgb { 1.0, 0.4, 0.05 }, Rgb { 0.5, 0.05, 1.0 } })
    {
        const auto strongest = std::max_element(colour.begin(), colour.end()) - colour.begin();
        const double reference = hueDegrees(agxviewtransform::apply(scaled(colour, -2.0)));
        double previousLuma = -1.0;
        Rgb previous { -1.0, -1.0, -1.0 };
        for (double stops = -4.0; stops <= 8.0; stops += 0.25)
        {
            const auto shown = agxviewtransform::apply(scaled(colour, stops));
            require(luma(shown) >= previousLuma - 0.004, "brightness never falls as radiance rises");
            // The top channel clips at 1 and settles on the sigmoid's 0.997 ceiling, a
            // drop smaller than one 8-bit step.
            for (int c = 0; c < 3; ++c)
                require(shown[c] >= previous[c] - 0.004, "no channel falls as radiance rises");
            previousLuma = luma(shown);
            previous = shown;
            if (stops >= -2.0 && stops <= 4.0)
            {
                const auto top = std::max_element(shown.begin(), shown.end()) - shown.begin();
                require(top == strongest, "the strongest channel stays strongest through +4 stops");
                // AgX bends hue a little as it desaturates. Standard's bend is the
                // failure this replaces: orange goes to yellow, then white.
                require(hueDistance(hueDegrees(shown), reference) <= 25.0, "the hue holds through +4 stops");
            }
        }
        require(chroma(agxviewtransform::apply(scaled(colour, 4.0))) >= 0.18,
                "a saturated colour is still coloured at +4 stops");
    }

    // The defect in the frame: at the same radiance Standard is flat white and AgX is not.
    const Rgb orange { 1.0, 0.4, 0.05 };
    require(chroma(standard(scaled(orange, 6.0))) == 0.0, "Standard clips this colour to flat white");
    require(chroma(agxviewtransform::apply(scaled(orange, 6.0))) >= 0.03, "AgX still has colour there");
    const double shift = hueDistance(hueDegrees(agxviewtransform::apply(scaled(orange, 2.0))), hueDegrees(orange));
    const double standardShift = hueDistance(hueDegrees(standard(scaled(orange, 2.0))), hueDegrees(orange));
    require(shift < standardShift, "AgX bends this hue less than Standard's clip");
    // White is reached at the top of the transform's own range, not before.
    require(chroma(agxviewtransform::apply(scaled({ 1.0, 1.0, 1.0 }, 12.0))) < 1.0e-6,
            "far over range ends at the transform's white");
}

colortransform::Description agxDescription()
{
    using namespace colortransform;
    Description description;
    description.toneMap = ToneMap::AgX;
    description.input = { { 64, 64 }, PixelFormat::RGBA16F, ColorSpace::LinearSRGB,
                          TransferFunction::Linear, AlphaMode::Straight };
    description.output = { { 64, 64 }, PixelFormat::RGBA8, ColorSpace::SRGB,
                           TransferFunction::SRGB, AlphaMode::Straight };
    return description;
}

void testAdmission()
{
    using namespace colortransform;
    AdmissionFailure failure {};
    const auto admitted = admit(agxDescription(), BackendCapability::NativeGpu, failure);
    require(admitted.has_value(), "AgX admits linear sRGB to SDR sRGB");
    require(std::any_of(admitted->stages().begin(), admitted->stages().end(),
                        [](StageKind stage) { return stage == StageKind::ViewTransform; }),
            "the admitted route names its view transform stage");

    auto encodedInput = agxDescription();
    encodedInput.input.transfer = TransferFunction::SRGB;
    encodedInput.input.colorSpace = ColorSpace::SRGB;
    require(! admit(encodedInput, BackendCapability::NativeGpu, failure).has_value(),
            "AgX refuses an already encoded input, the double-encode this route must not do");
    auto linearOutput = agxDescription();
    linearOutput.outputIntent = OutputIntent::SceneLinear;
    linearOutput.output.format = PixelFormat::RGBA16F;
    linearOutput.output.colorSpace = ColorSpace::LinearSRGB;
    linearOutput.output.transfer = TransferFunction::Linear;
    require(! admit(linearOutput, BackendCapability::NativeGpu, failure).has_value(),
            "AgX refuses a scene-linear output, which stays untransformed");
}
} // namespace

int main()
{
    testSigmoidAndNeutralAxis();
    testSaturatedHighlightsKeepColour();
    testAdmission();
    std::cout << "agx_view_transform_numeric: pass\n";
    return EXIT_SUCCESS;
}
