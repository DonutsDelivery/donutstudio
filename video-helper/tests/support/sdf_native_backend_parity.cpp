#include "sdf_native_backend_parity.h"
#include "sdf_reference_evaluator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>

namespace videohelper::sdf::test
{
namespace
{
using videowire::SdfIr;
using videowire::SdfOperation;
using videowire::SdfRecord;
using videowire::SdfStableId;

SdfRecord record (SdfStableId id, SdfOperation operation,
                  std::initializer_list<SdfStableId> inputs,
                  std::initializer_list<double> parameters)
{
    SdfRecord result;
    result.stableId = id;
    result.operation = operation;
    result.inputCount = static_cast<std::uint8_t> (inputs.size());
    result.parameterCount = static_cast<std::uint8_t> (parameters.size());
    std::copy (inputs.begin(), inputs.end(), result.inputs.begin());
    std::copy (parameters.begin(), parameters.end(), result.parameters.begin());
    return result;
}

std::vector<double> parametersFor (SdfOperation operation,
                                   double polarOffset = 10000.0)
{
    switch (operation)
    {
        case SdfOperation::sphere: return { 0.72 };
        case SdfOperation::box: return { 0.7, 0.55, 0.65 };
        case SdfOperation::roundedBox: return { 0.75, 0.6, 0.7, 0.12 };
        case SdfOperation::plane: return { 0.0, 0.0, 1.0, 0.0 };
        case SdfOperation::torus: return { 0.45, 0.24 };
        case SdfOperation::capsule: return { 0.0, -0.65, 0.0, 0.0, 0.65, 0.0, 0.24 };
        case SdfOperation::cylinder: return { 0.62, 0.8 };
        case SdfOperation::cone: return { 0.7, 0.9 };
        case SdfOperation::gyroid: return { 2.0, 0.14 };
        case SdfOperation::smoothUnion:
        case SdfOperation::smoothIntersection:
        case SdfOperation::smoothSubtraction: return { 0.22 };
        case SdfOperation::translate: return { 0.18, -0.12, 0.2 };
        case SdfOperation::rotate: return { 1.0, 2.0, -0.5, 0.7 };
        case SdfOperation::scale: return { 1.25, 0.8, 1.4 };
        case SdfOperation::repeat: return { 1.7, 2.1, 2.5 };
        case SdfOperation::polarRepeat: return { 2.0, 7.0, polarOffset };
        case SdfOperation::mirror: return { 1.0, 0.0, 1.0 };
        case SdfOperation::twist: return { 2.0, 0.65 };
        case SdfOperation::bend: return { 0.0, -0.45 };
        case SdfOperation::taper: return { 1.0, 0.3 };
        case SdfOperation::displacement: return { 0.12, 2.3 };
        case SdfOperation::domainWarp: return { 0.16, -0.12, 0.1, 1.2, 1.7, 2.1 };
        case SdfOperation::unionOp:
        case SdfOperation::intersection:
        case SdfOperation::subtraction: return {};
    }
    return {};
}

SdfRecord operationRecord (SdfStableId id, SdfOperation operation,
                           std::initializer_list<SdfStableId> inputs,
                           double polarOffset = 10000.0)
{
    const auto parameters = parametersFor (operation, polarOffset);
    SdfRecord result;
    result.stableId = id;
    result.operation = operation;
    result.inputCount = static_cast<std::uint8_t> (inputs.size());
    result.parameterCount = static_cast<std::uint8_t> (parameters.size());
    std::copy (inputs.begin(), inputs.end(), result.inputs.begin());
    std::copy (parameters.begin(), parameters.end(), result.parameters.begin());
    return result;
}

std::shared_ptr<const AdmittedSdfIr> fixtureFor (SdfOperation operation,
                                                 std::string& error,
                                                 double polarOffset = 10000.0)
{
    const auto schema = videowire::sdfOperationSchema (operation);
    SdfIr source;
    if (schema.inputCount == 0)
    {
        source.rootId = 1;
        source.records = { operationRecord (1, operation, {}) };
    }
    else if (schema.inputCount == 2)
    {
        source.rootId = 3;
        source.records = {
            record (1, SdfOperation::sphere, {}, { 0.72 }),
            record (2, SdfOperation::box, {}, { 0.48, 0.62, 0.54 }),
            operationRecord (3, operation, { 1, 2 })
        };
    }
    else
    {
        source.rootId = operation == SdfOperation::translate ? 2 : 3;
        const auto baseRadius = operation == SdfOperation::translate ? 0.72
            : (operation == SdfOperation::polarRepeat ? 0.45 : 0.3);
        source.records = {
            record (1, SdfOperation::sphere, {}, { baseRadius })
        };
        if (operation == SdfOperation::translate)
            source.records.push_back (operationRecord (2, operation, { 1 }));
        else
        {
            source.records.push_back (
                record (2, SdfOperation::translate, { 1 },
                        { 0.55, operation == SdfOperation::polarRepeat ? 0.0 : 0.08, 0.0 }));
            source.records.push_back (operationRecord (
                3, operation, { 2 }, polarOffset));
        }
    }

    auto admitted = admitSdfIr (source, {}, error);
    if (! admitted) return {};
    return std::make_shared<const AdmittedSdfIr> (std::move (*admitted));
}

double qualityScale (arbitgpu::NativeSdfQuality quality)
{
    switch (quality)
    {
        case arbitgpu::NativeSdfQuality::low: return 4.0;
        case arbitgpu::NativeSdfQuality::medium: return 2.0;
        case arbitgpu::NativeSdfQuality::high: return 1.0;
        case arbitgpu::NativeSdfQuality::ultra: return 0.5;
        case arbitgpu::NativeSdfQuality::count: break;
    }
    return 1.0;
}

std::uint8_t expectedDepth (const AdmittedSdfIr& geometry,
                            const NativeSdfRenderControls& controls,
                            std::uint32_t width, std::uint32_t height,
                            std::uint32_t x, std::uint32_t y)
{
    const auto u = (2.0 * (static_cast<double> (x) + 0.5) - width) / height;
    const auto v = (2.0 * (static_cast<double> (y) + 0.5) - height) / height;
    const auto inverseLength = 1.0 / std::sqrt (u * u + v * v + 1.8 * 1.8);
    const reference::Point3 direction { u * inverseLength, v * inverseLength,
                                        -1.8 * inverseLength };
    double travel = 0.0;
    bool hit = false;
    for (std::uint32_t step = 0; step < controls.maximumSteps && step < 512; ++step)
    {
        if (travel > controls.maximumDistance) break;
        const reference::Point3 point { direction.x * travel,
                                        direction.y * travel,
                                        3.0 + direction.z * travel };
        const auto field = reference::evaluatePoint (geometry, point);
        const auto threshold = std::max (
            controls.epsilon * qualityScale (controls.adaptiveQuality)
                * std::max (1.0, travel * 0.05),
            0.000001);
        if (field <= threshold)
        {
            hit = true;
            break;
        }
        travel += field;
    }
    if (! hit) return 255;
    const auto normalized = std::clamp (travel / controls.maximumDistance, 0.0, 1.0);
    return static_cast<std::uint8_t> (std::lround (normalized * 255.0));
}
} // namespace

std::vector<NativeOperationFixture> nativeOperationFixtures (std::string& error)
{
    std::vector<NativeOperationFixture> result;
    result.reserve (26);
    for (std::uint32_t raw = static_cast<std::uint32_t> (SdfOperation::sphere);
         raw <= static_cast<std::uint32_t> (SdfOperation::domainWarp); ++raw)
    {
        const auto operation = static_cast<SdfOperation> (raw);
        const auto polarOffset = operation == SdfOperation::polarRepeat ? 0.0 : 10000.0;
        auto geometry = fixtureFor (operation, error, polarOffset);
        if (! geometry)
        {
            error = std::string (videowire::sdfOperationWireToken (operation))
                + " parity fixture failed admission: " + error;
            return {};
        }
        result.push_back ({ operation, std::move (geometry) });
    }
    if (result.size() != 26)
    {
        error = "native SDF parity fixture count is not 26";
        return {};
    }
    error.clear();
    return result;
}

std::vector<NativeOperationFixture> largeOffsetPolarRepeatFixtures (std::string& error)
{
    std::vector<NativeOperationFixture> result;
    for (const auto offset : { -10000.0, 10000.0 })
    {
        auto geometry = fixtureFor (SdfOperation::polarRepeat, error, offset);
        if (geometry == nullptr) return {};
        result.push_back ({ SdfOperation::polarRepeat, std::move (geometry) });
    }
    error.clear();
    return result;
}

bool verifyNativeDepthParity (const NativeOperationFixture& fixture,
                              const NativeSdfRenderControls& controls,
                              const std::vector<std::uint8_t>& rgba,
                              std::uint32_t width,
                              std::uint32_t height,
                              std::string& error)
{
    if (rgba.size() != static_cast<std::size_t> (width) * height * 4u)
    {
        error = "native SDF parity readback has the wrong byte count";
        return false;
    }

    bool expectedForeground = false;
    // Depth is normalized into one byte by the production output path. Two
    // byte levels cover half-float/readback rounding while still catching a
    // different raymarch result. Foreground membership must match exactly.
    constexpr int polarRepeatDepthTolerance = 2;
    const auto inset = fixture.operation == SdfOperation::repeat ? width / 3 : 0u;
    for (std::uint32_t y = inset; y < height - inset; ++y)
    {
        for (std::uint32_t x = inset; x < width - inset; ++x)
        {
            const auto expected = expectedDepth (*fixture.geometry, controls, width, height, x, y);
            expectedForeground = expectedForeground || expected != 255;
            const auto offset = (static_cast<std::size_t> (y) * width + x) * 4u;
            const auto actual = rgba[offset];
            const auto delta = std::abs (static_cast<int> (actual) - static_cast<int> (expected));
            const auto repeatedDomain = fixture.operation == SdfOperation::polarRepeat;
            const auto maskMismatch = (expected == 255) != (actual == 255);
            const auto distanceMismatch = repeatedDomain
                ? (maskMismatch || (expected != 255 && delta > polarRepeatDepthTolerance))
                : delta > 2;
            if (distanceMismatch || rgba[offset + 1] != actual || rgba[offset + 2] != actual
                || rgba[offset + 3] != 255)
            {
                std::ostringstream message;
                message << videowire::sdfOperationWireToken (fixture.operation)
                        << " native depth differs from the CPU oracle at " << x << ',' << y
                        << ": expected " << static_cast<int> (expected)
                        << ", actual " << static_cast<int> (actual);
                error = message.str();
                return false;
            }
        }
    }
    if (! expectedForeground)
    {
        error = std::string (videowire::sdfOperationWireToken (fixture.operation))
            + " parity fixture has no oracle-visible samples";
        return false;
    }

    error.clear();
    return true;
}
} // namespace videohelper::sdf::test
