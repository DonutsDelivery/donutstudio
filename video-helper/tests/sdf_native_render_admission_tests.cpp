#include "../src/sdf_native_render_admission.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace
{
int failures = 0;
int checks = 0;

void check (bool condition, const char* message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::fprintf (stderr, "FAIL: %s\n", message);
    }
}

videohelper::sdf::AdmittedSdfIr sphereGeometry()
{
    videowire::SdfIr source;
    source.rootId = 41;
    videowire::SdfRecord sphere;
    sphere.stableId = source.rootId;
    sphere.operation = videowire::SdfOperation::sphere;
    sphere.parameterCount = 1;
    sphere.parameters[0] = 1.25;
    source.records.push_back (sphere);
    std::string error;
    auto admitted = videohelper::sdf::admitSdfIr (source, {}, error);
    if (! admitted)
    {
        std::fprintf (stderr, "test setup failed: %s\n", error.c_str());
        std::exit (2);
    }
    return std::move (*admitted);
}

videohelper::sdf::AdmittedSdfIr boxGeometry()
{
    videowire::SdfIr source;
    source.rootId = 42;
    videowire::SdfRecord box;
    box.stableId = source.rootId;
    box.operation = videowire::SdfOperation::box;
    box.parameterCount = 3;
    box.parameters[0] = 0.75;
    box.parameters[1] = 0.5;
    box.parameters[2] = 0.25;
    source.records.push_back (box);
    std::string error;
    auto admitted = videohelper::sdf::admitSdfIr (source, {}, error);
    if (! admitted)
    {
        std::fprintf (stderr, "test setup failed: %s\n", error.c_str());
        std::exit (2);
    }
    return std::move (*admitted);
}

videohelper::sdf::AdmittedSdfIr twoParameterGeometry (
    videowire::SdfOperation operation, std::uint64_t stableId,
    double first, double second)
{
    videowire::SdfIr source;
    source.rootId = stableId;
    videowire::SdfRecord primitive;
    primitive.stableId = source.rootId;
    primitive.operation = operation;
    primitive.parameterCount = 2;
    primitive.parameters[0] = first;
    primitive.parameters[1] = second;
    source.records.push_back (primitive);
    std::string error;
    auto admitted = videohelper::sdf::admitSdfIr (source, {}, error);
    if (! admitted)
    {
        std::fprintf (stderr, "test setup failed: %s\n", error.c_str());
        std::exit (2);
    }
    return std::move (*admitted);
}

arbitgpu::NativeSdfExecutionCapabilities primitiveBackend()
{
    arbitgpu::NativeSdfExecutionCapabilities result;
    result.available = true;
    result.backend = "test-native";
    result.device = "synthetic admission fixture";
    result.supportedOperations[static_cast<std::size_t> (videowire::SdfOperation::sphere)] = true;
    result.supportedOperations[static_cast<std::size_t> (videowire::SdfOperation::box)] = true;
    result.supportedOperations[static_cast<std::size_t> (videowire::SdfOperation::torus)] = true;
    result.supportedOperations[static_cast<std::size_t> (videowire::SdfOperation::cylinder)] = true;
    result.supportedOutputs[static_cast<std::size_t> (arbitgpu::NativeSdfOutput::color)] = true;
    result.supportedOutputs[static_cast<std::size_t> (arbitgpu::NativeSdfOutput::depth)] = true;
    result.supportedOutputs[static_cast<std::size_t> (arbitgpu::NativeSdfOutput::normal)] = true;
    result.maxOperations = 1;
    result.maxDepth = 1;
    result.maxExtent = 1024;
    result.maxPixels = 1024ull * 1024ull;
    result.maxSteps = 256;
    result.minEpsilon = 0.000001;
    result.maxEpsilon = 0.1;
    result.maxDistance = 1000.0;
    return result;
}
} // namespace

int main()
{
    using namespace videohelper::sdf;

    const auto geometry = sphereGeometry();
    const auto capabilities = primitiveBackend();
    NativeSdfRenderControls controls;
    std::string error;
    const auto admitted = admitNativeSdfRender (
        geometry, { 320, 180 }, controls, capabilities, error);
    check (admitted.has_value() && error.empty(),
           "one bounded sphere render reaches native execution admission");
    check (admitted && admitted->backend() == "test-native"
           && admitted->dimensions().width == 320
           && admitted->dimensions().height == 180
           && admitted->geometry().structuralDigest() == geometry.structuralDigest(),
           "admission retains immutable geometry, dimensions, and backend identity");

    const auto box = boxGeometry();
    const auto admittedBox = admitNativeSdfRender (
        box, { 320, 180 }, controls, capabilities, error);
    check (admittedBox.has_value() && error.empty()
           && admittedBox->geometry().records().size() == 1
           && admittedBox->geometry().records()[0].operation == videowire::SdfOperation::box
           && admittedBox->geometry().records()[0].parameters[0] == 0.75
           && admittedBox->geometry().records()[0].parameters[1] == 0.5
           && admittedBox->geometry().records()[0].parameters[2] == 0.25,
           "one bounded box reaches immutable native execution admission without parameter loss");

    const auto torus = twoParameterGeometry (
        videowire::SdfOperation::torus, 43, 1.75, 0.375);
    const auto admittedTorus = admitNativeSdfRender (
        torus, { 320, 180 }, controls, capabilities, error);
    check (admittedTorus && error.empty()
           && admittedTorus->geometry().records().size() == 1
           && admittedTorus->geometry().records()[0].operation == videowire::SdfOperation::torus
           && admittedTorus->geometry().records()[0].parameters[0] == 1.75
           && admittedTorus->geometry().records()[0].parameters[1] == 0.375,
           "torus major and minor radii reach native admission without parameter loss");

    const auto cylinder = twoParameterGeometry (
        videowire::SdfOperation::cylinder, 44, 0.625, 1.375);
    const auto admittedCylinder = admitNativeSdfRender (
        cylinder, { 320, 180 }, controls, capabilities, error);
    check (admittedCylinder && error.empty()
           && admittedCylinder->geometry().records().size() == 1
           && admittedCylinder->geometry().records()[0].operation
                == videowire::SdfOperation::cylinder
           && admittedCylinder->geometry().records()[0].parameters[0] == 0.625
           && admittedCylinder->geometry().records()[0].parameters[1] == 1.375,
           "cylinder radius and half height reach native admission without parameter loss");

    controls.output = arbitgpu::NativeSdfOutput::normal;
    check (admitNativeSdfRender (
               torus, { 320, 180 }, controls, capabilities, error).has_value()
           && error.empty(),
           "torus retains the native normal-output route");
    controls.output = arbitgpu::NativeSdfOutput::depth;
    check (admitNativeSdfRender (
               cylinder, { 320, 180 }, controls, capabilities, error).has_value()
           && error.empty(),
           "cylinder retains the native depth-output route");
    controls = {};

    auto noSphere = capabilities;
    noSphere.supportedOperations.fill (false);
    check (! admitNativeSdfRender (geometry, { 320, 180 }, controls, noSphere, error)
           && error == "native GPU SDF backend does not support operation sphere",
           "unsupported SDF operations fail before execution");

    auto noDepth = capabilities;
    noDepth.supportedOutputs[static_cast<std::size_t> (
        arbitgpu::NativeSdfOutput::depth)] = false;
    controls.output = arbitgpu::NativeSdfOutput::depth;
    check (! admitNativeSdfRender (geometry, { 320, 180 }, controls, noDepth, error)
           && error == "native GPU SDF backend does not support output depth",
           "unsupported output passes fail before execution");

    controls.output = arbitgpu::NativeSdfOutput::curvature;
    check (! admitNativeSdfRender (geometry, { 320, 180 }, controls, noDepth, error)
           && error == "native GPU SDF backend does not support output curvature",
           "reference utility outputs retain an exact native-backend rejection");

    controls = {};
    controls.maximumSteps = capabilities.maxSteps + 1;
    check (! admitNativeSdfRender (geometry, { 320, 180 }, controls, capabilities, error)
           && error == "native GPU SDF raymarch controls exceed backend limits",
           "raymarch controls are bounded before execution");

    controls = {};
    check (! admitNativeSdfRender (geometry, { capabilities.maxExtent + 1, 1 },
                                   controls, capabilities, error)
           && error == "native GPU SDF render dimensions exceed backend limits",
           "render extent is bounded before execution");

    auto unavailable = capabilities;
    unavailable.available = false;
    unavailable.error = "production SDF renderer unavailable";
    check (! admitNativeSdfRender (geometry, { 320, 180 }, controls, unavailable, error)
           && error == "production SDF renderer unavailable",
           "backend unavailability remains an exact fail-closed diagnostic");

    auto malformed = capabilities;
    malformed.backend.clear();
    check (! admitNativeSdfRender (geometry, { 320, 180 }, controls, malformed, error)
           && error == "native GPU SDF execution capabilities are invalid",
           "contradictory backend capabilities fail closed");

    const auto production = arbitgpu::queryNativeSdfExecution();
    check (! production.available
           && production.error == "native GPU SDF execution is not compiled in"
           && ! admitNativeSdfRender (geometry, { 320, 180 }, controls, production, error)
           && error == production.error,
           "the compiled backend does not claim SDF execution without a production draw");

    std::printf ("sdf-native-render-admission: %d/%d checks passed\n",
                 checks - failures, checks);
    return failures == 0 ? 0 : 1;
}