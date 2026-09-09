#include "visual_plan_executor.h"
#include "../../shared/ColorTransformOperationContract.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace
{
void require(bool condition, const char* message)
{
    if (! condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

colortransformoperation::Payload payload(double sourcePeak,
                                         colortransform::ToneMap toneMap)
{
    colortransformoperation::Payload value;
    value.inputFormat = colortransform::PixelFormat::RGBA8;
    value.inputColorSpace = colortransform::ColorSpace::SRGB;
    value.inputTransfer = colortransform::TransferFunction::SRGB;
    value.inputAlpha = colortransform::AlphaMode::Straight;
    value.outputFormat = colortransform::PixelFormat::RGBA8;
    value.outputColorSpace = colortransform::ColorSpace::SRGB;
    value.outputTransfer = colortransform::TransferFunction::SRGB;
    value.outputAlpha = colortransform::AlphaMode::Straight;
    value.workingColorSpace = colortransform::ColorSpace::LinearSRGB;
    value.workingFormat = colortransform::PixelFormat::RGBA16F;
    value.outputIntent = colortransform::OutputIntent::SdrDisplay;
    value.toneMap = toneMap;
    value.luminance = { 100.0, sourcePeak, 100.0, 100.0 };
    return value;
}

videowire::CompiledVisualLayerPlan planFor(const std::string& wire)
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 19;
    plan.structuralRevision = 7;
    plan.producerValidated = true;
    plan.nodeKinds = { "video.source", "visual.color.transform", "video.out" };
    plan.nodeIds = { 11, 12, 13 };
    plan.edges = { { 11, 0, 12, 0 }, { 12, 1, 13, 0 } };
    plan.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 12, 1, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 13, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    plan.operations = {
        { 11, "video.source", "source-decode", "" },
        { 12, "visual.color.transform", "native-gpu", wire },
        { 13, "video.out", "native-gpu", "" }
    };
    return plan;
}
} // namespace

int main()
{
    using namespace colortransform;
    using namespace colortransformoperation;

    const auto identityPayload = payload(100.0, ToneMap::None);
    const auto identityWire = serialize(identityPayload);
    Payload decoded;
    require(parse(identityWire, decoded), "canonical operation payload did not parse");
    require(serialize(decoded) == identityWire,
            "operation payload did not round-trip canonically");
    AdmissionFailure failure = AdmissionFailure::None;
    const auto identity = parseAndAdmit(identityWire, { 1920, 1080 }, failure);
    require(identity.has_value() && failure == AdmissionFailure::None,
            "identity transform did not admit at the concrete frame extent");
    require(identity->description().input.extent == Extent { 1920, 1080 },
            "concrete frame extent was not bound before admission");

    const auto mappedPayload = payload(1000.0, ToneMap::Reinhard);
    const auto mappedWire = serialize(mappedPayload);
    const auto mapped = parseAndAdmit(mappedWire, { 64, 32 }, failure);
    require(mapped.has_value() && mapped->stageCount() == 3,
            "HDR-range-to-SDR transform did not admit its exact stage plan");

    auto malformed = mappedWire;
    malformed.insert(malformed.size() - 2, " invented=\"1\"");
    require(! parse(malformed, decoded), "invented operation payload key was accepted");
    require(! parseAndAdmit(serialize(payload(1000.0, ToneMap::None)), { 64, 32 }, failure)
                && failure == AdmissionFailure::ToneMapRequired,
            "range reduction without tone mapping was accepted");

    videowire::VisualLayerExecution execution;
    std::string error;
    const auto exactPlan = planFor(mappedWire);
    require(videowire::compileVisualLayerExecution(exactPlan, execution, error)
                && execution.colorTransform
                && execution.colorTransformPayload.luminance.sourcePeakNits == 1000.0,
            "exact color transform plan did not compile");

    auto wrongTopology = exactPlan;
    wrongTopology.edges = { { 11, 0, 13, 0 }, { 12, 1, 13, 0 } };
    require(! videowire::compileVisualLayerExecution(wrongTopology, execution, error)
                && error == "visual.color.transform requires exactly Source -> Color Transform -> Output",
            "unsupported color transform topology was accepted");

    auto wrongDescriptor = exactPlan;
    wrongDescriptor.ports[1].colorSpace = "linearSRGB";
    require(! videowire::compileVisualLayerExecution(wrongDescriptor, execution, error)
                && error == "visual.color.transform requires exact RGBA8 sRGB Frame<Image> ports",
            "implicit color-space conversion was accepted at a typed edge");

    auto wrongPayload = exactPlan;
    auto unsupported = mappedPayload;
    unsupported.outputColorSpace = ColorSpace::DisplayP3;
    wrongPayload.operations[1].payloadXml = serialize(unsupported);
    require(! videowire::compileVisualLayerExecution(wrongPayload, execution, error)
                && error == "visual.color.transform has an unsupported immutable payload",
            "unsupported output semantics were accepted");

    std::cout << "color transform operation execution: all checks passed\n";
    return EXIT_SUCCESS;
}
