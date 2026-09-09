#include "../src/flat_shader_bridge.h"
#include "../src/visual_plan_executor.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <string>

namespace
{
int failures = 0;

void check(bool condition, const char* message)
{
    if (!condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

std::string readFile(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    return { std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
}

std::string xmlEscape(const std::string& value)
{
    std::string result;
    result.reserve(value.size());
    for (const char c : value)
    {
        switch (c)
        {
            case '&': result += "&amp;"; break;
            case '<': result += "&lt;"; break;
            case '>': result += "&gt;"; break;
            case '"': result += "&quot;"; break;
            case '\'': result += "&apos;"; break;
            default: result.push_back(c); break;
        }
    }
    return result;
}

std::string payloadV1(const std::string& role, const std::string& language,
                      const std::string& source, const std::string& pack,
                      const std::string& program,
                      const std::string& trust = "verified-bundled-curated")
{
    return "<FlatShaderBridge schemaVersion=\"1\" role=\"" + role
        + "\" language=\"" + language + "\" source=\"" + xmlEscape(source)
        + "\" sourceSha256=\"" + videohelper::sha256Text(source)
        + "\" trust=\"" + trust + "\" catalogPackId=\"" + pack
        + "\" catalogProgramId=\"" + program + "\"/>";
}

std::string payloadV2(const std::string& role, const std::string& language,
                      const std::string& source, const std::string& pack,
                      const std::string& program, const std::string& parameters)
{
    return "<FlatShaderBridge schemaVersion=\"2\" role=\"" + role
        + "\" language=\"" + language + "\" source=\"" + xmlEscape(source)
        + "\" sourceSha256=\"" + videohelper::sha256Text(source)
        + "\" catalogPackId=\"" + pack + "\" catalogProgramId=\"" + program
        + "\" parameters=\"" + xmlEscape(parameters) + "\"/>";
}

videowire::CompiledVisualLayerPlan generatorPlan(const std::string& xml)
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 31;
    plan.structuralRevision = 7;
    plan.producerValidated = true;
    plan.nodeKinds = { "visual.shader.generator", "video.out" };
    plan.nodeIds = { 41, 42 };
    plan.edges = { { 41, 0, 42, 0 } };
    plan.operations = {
        { 41, "visual.shader.generator", "native-gpu", xml },
        { 42, "video.out", "native-gpu", "" }
    };
    return plan;
}

videowire::CompiledVisualLayerPlan customPlan(
    const std::string& source, const programmableruntime::Grant& grant)
{
    auto plan = generatorPlan(payloadV2("generator", "glsl", source, "", "", ""));
    plan.nodeKinds[0] = "visual.shader.custom";
    plan.operations[0].kind = "visual.shader.custom";
    plan.operations[0].runtimeGrantJson = programmableadmission::toJson(grant).dump();
    return plan;
}

videowire::CompiledVisualLayerPlan filterPlan(const std::string& xml)
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 31;
    plan.structuralRevision = 8;
    plan.producerValidated = true;
    plan.nodeKinds = { "video.source", "visual.shader.filter", "video.out" };
    plan.nodeIds = { 40, 41, 42 };
    plan.edges = { { 40, 0, 41, 0 }, { 41, 1, 42, 0 } };
    plan.ports = {
        { 41, 0, 1, "in", "frame", "image", "rgba8", "srgb" },
        { 41, 1, 1, "out", "frame", "image", "rgba8", "srgb" }
    };
    plan.operations = {
        { 40, "video.source", "source-decode", "" },
        { 41, "visual.shader.filter", "native-gpu", xml },
        { 42, "video.out", "native-gpu", "" }
    };
    return plan;
}

videowire::CompiledVisualLayerPlan shaderChainPlan(const std::string& generatorXml,
                                                    const std::string& filterXml,
                                                    bool decodedSource)
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 33;
    plan.structuralRevision = decodedSource ? 11 : 10;
    plan.producerValidated = true;
    if (decodedSource)
    {
        plan.nodeKinds = { "video.source", "visual.shader.filter",
                           "visual.shader.filter", "video.out" };
        plan.nodeIds = { 60, 61, 62, 63 };
        plan.edges = { { 60, 0, 61, 0 }, { 61, 1, 62, 0 }, { 62, 1, 63, 0 } };
        plan.operations = { { 60, "video.source", "source-decode", "" },
            { 61, "visual.shader.filter", "native-gpu", filterXml },
            { 62, "visual.shader.filter", "native-gpu", filterXml },
            { 63, "video.out", "native-gpu", "" } };
    }
    else
    {
        plan.nodeKinds = { "visual.shader.generator", "visual.shader.filter", "video.out" };
        plan.nodeIds = { 60, 61, 63 };
        plan.edges = { { 60, 0, 61, 0 }, { 61, 1, 63, 0 } };
        plan.operations = { { 60, "visual.shader.generator", "native-gpu", generatorXml },
            { 61, "visual.shader.filter", "native-gpu", filterXml },
            { 63, "video.out", "native-gpu", "" } };
    }
    for (const int nodeId : { 61, 62 })
    {
        if (!decodedSource && nodeId == 62) continue;
        plan.ports.push_back({ nodeId, 0, 1, "in", "frame", "image", "rgba8", "srgb" });
        plan.ports.push_back({ nodeId, 1, 1, "out", "frame", "image", "rgba8", "srgb" });
    }
    return plan;
}

videowire::CompiledVisualLayerPlan filterCompositePlan(const std::string& xml)
{
    auto plan = filterPlan(xml);
    plan.nodeKinds = { "video.source", "visual.shader.filter", "visual.effect.glow",
                       "video.mask.shape", "visual.shape.rectangle", "visual.draw.shape",
                       "video.layer.source", "video.blend", "video.out" };
    plan.nodeIds = { 40, 41, 45, 46, 47, 48, 43, 44, 42 };
    plan.edges = { { 40, 0, 41, 0 }, { 41, 1, 45, 0 }, { 45, 1, 46, 0 },
                   { 46, 1, 44, 0 }, { 47, 0, 48, 0 }, { 48, 1, 44, 2 },
                   { 43, 0, 44, 3 }, { 44, 1, 42, 0 } };
    plan.ports = {
        { 40, 0, 1, "out", "frame", "image", "rgba8", "srgb" },
        { 41, 0, 1, "in", "frame", "image", "rgba8", "srgb" },
        { 41, 1, 1, "out", "frame", "image", "rgba8", "srgb" },
        { 45, 0, 1, "in", "frame", "image", "rgba8", "srgb" },
        { 45, 1, 1, "out", "frame", "image", "rgba8", "srgb" },
        { 46, 0, 1, "in", "frame", "image", "rgba8", "srgb" },
        { 46, 1, 1, "out", "frame", "image", "rgba8", "srgb" },
        { 47, 0, 1, "out", "control", "shape", "unspecified", "unspecified" },
        { 48, 0, 1, "in", "control", "shape", "unspecified", "unspecified" },
        { 48, 1, 1, "out", "frame", "image", "rgba8", "srgb" },
        { 43, 0, 1, "out", "frame", "image", "rgba8", "srgb" },
        { 44, 0, 1, "in", "frame", "image", "rgba8", "srgb" },
        { 44, 1, 1, "out", "frame", "image", "rgba8", "srgb" },
        { 44, 2, 1, "in", "frame", "image", "rgba8", "srgb" },
        { 44, 3, 1, "in", "frame", "image", "rgba8", "srgb" },
        { 42, 0, 1, "in", "frame", "image", "rgba8", "srgb" }
    };
    plan.operations = {
        { 40, "video.source", "source-decode", "" },
        { 41, "visual.shader.filter", "native-gpu", xml },
        { 45, "visual.effect.glow", "native-gpu",
          "<CommonEffect schemaVersion=\"1\" effect=\"glow\" intensity=\"0.5\" radius=\"6.0\"/>" },
        { 46, "video.mask.shape", "native-gpu", "" },
        { 47, "visual.shape.rectangle", "control-eval",
          "<NodeParams centerX=\"0.25\" centerY=\"0.75\" width=\"0.5\" height=\"0.2\"/>" },
        { 48, "visual.draw.shape", "native-gpu",
          "<NodeParams red=\"0.1\" green=\"0.2\" blue=\"0.3\" alpha=\"0.4\"/>" },
        { 43, "video.layer.source", "source-decode", "" },
        { 44, "video.blend", "native-gpu", "" },
        { 42, "video.out", "native-gpu", "" }
    };
    return plan;
}

videowire::CompiledVisualLayerPlan transitionPlan(const std::string& encoded)
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 32;
    plan.structuralRevision = 9;
    plan.producerValidated = true;
    plan.nodeKinds = { "video.source", "video.layer.source", shadertransition::operationKind, "video.out" };
    plan.nodeIds = { 50, 51, 52, 53 };
    plan.edges = { { 51, 0, 52, 0 }, { 50, 0, 52, 1 }, { 52, 2, 53, 0 } };
    plan.ports = {
        { 50, 0, 1, "out", "frame", "image", "rgba8", "srgb" },
        { 51, 0, 1, "out", "frame", "image", "rgba8", "srgb" },
        { 52, 0, 1, "in", "frame", "image", "rgba8", "srgb" },
        { 52, 1, 1, "in", "frame", "image", "rgba8", "srgb" },
        { 52, 2, 1, "out", "frame", "image", "rgba8", "srgb" },
        { 53, 0, 1, "in", "frame", "image", "rgba8", "srgb" }
    };
    plan.operations = {
        { 50, "video.source", "source-decode", "" },
        { 51, "video.layer.source", "source-decode", "" },
        { 52, shadertransition::operationKind, "native-gpu", encoded },
        { 53, "video.out", "native-gpu", "" }
    };
    return plan;
}

videowire::CompiledVisualLayerPlan chainedTransitionPlan(
    const std::string& generatorXml, const std::string& transitionXml,
    const std::string& filterXml)
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 34;
    plan.structuralRevision = 12;
    plan.producerValidated = true;
    plan.nodeKinds = { "visual.shader.generator", "video.source",
        shadertransition::operationKind, "visual.shader.filter", "video.out" };
    plan.nodeIds = { 70, 71, 72, 73, 74 };
    plan.edges = { { 70, 0, 72, 0 }, { 71, 0, 72, 1 },
                   { 72, 2, 73, 0 }, { 73, 1, 74, 0 } };
    plan.ports = {
        { 72, 0, 1, "in", "frame", "image", "rgba8", "srgb" },
        { 72, 1, 1, "in", "frame", "image", "rgba8", "srgb" },
        { 72, 2, 1, "out", "frame", "image", "rgba8", "srgb" },
        { 73, 0, 1, "in", "frame", "image", "rgba8", "srgb" },
        { 73, 1, 1, "out", "frame", "image", "rgba8", "srgb" }
    };
    plan.operations = {
        { 70, "visual.shader.generator", "native-gpu", generatorXml },
        { 71, "video.source", "source-decode", "" },
        { 72, shadertransition::operationKind, "native-gpu", transitionXml },
        { 73, "visual.shader.filter", "native-gpu", filterXml },
        { 74, "video.out", "native-gpu", "" }
    };
    return plan;
}

void expectRejected(std::string xml, const char* expected, const char* message)
{
    videowire::FlatShaderBridgePayload parsed;
    std::string error;
    check(!videowire::parseFlatShaderBridgePayload(xml, parsed, error), message);
    check(error.find(expected) != std::string::npos, "rejection diagnostic identifies the failed contract");
}
}

int main()
{
    std::string error;
    const std::string root = SHADER_CATALOG_SOURCE_ROOT;
    const auto generatorSource = readFile(root + "/arbit-isf-starters/shaders/plasma.fs");
    const auto filterSource = readFile(root + "/vidvox-isf/ISF/RGB Invert.fs");
    const auto vectorSource = readFile(root + "/arbit-isf-starters/shaders/orbit.fs");
    const auto multipassSource = readFile(root + "/arbit-isf-starters/shaders/light_trails.fs");
    const auto transitionSource = readFile(root + "/vidvox-isf/ISF/Fade.fs");
    check(!generatorSource.empty(), "generator fixture source loaded");
    check(!filterSource.empty(), "filter fixture source loaded");
    check(!vectorSource.empty(), "vector fixture source loaded");
    check(!multipassSource.empty(), "multipass fixture source loaded");
    check(!transitionSource.empty(), "transition fixture source loaded");

    const auto* generatorEntry = shadercatalog::find("arbit-isf-starters-v1", "plasma");
    const auto* filterEntry = shadercatalog::find("vidvox-isf", "rgb-invert");
    const auto* multipassEntry = shadercatalog::find("arbit-isf-starters-v1", "light_trails");
    const auto* transitionEntry = shadercatalog::find("vidvox-isf", "fade");
    check(generatorEntry != nullptr, "generator catalog metadata loaded");
    check(filterEntry != nullptr, "filter catalog metadata loaded");
    check(multipassEntry != nullptr, "multipass catalog metadata loaded");
    check(transitionEntry != nullptr, "transition catalog metadata loaded");
    const shadercatalog::Parameter ambiguousComponents[] {
        { "color", "color", 4, { 0.0, 0.0, 0.0, 1.0 },
          { 0.0, 0.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0, 1.0 }, true, true },
        { "color.r", "float", 1, { 0.0, 0.0, 0.0, 0.0 },
          { 0.0, 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0, 0.0 }, true, true }
    };
    check(!shadercatalog::validateParameterMetadata(
              ambiguousComponents, std::size(ambiguousComponents), error),
          "component-derived parameter IDs cannot collide with scalar IDs");
    auto reservedWireName = ambiguousComponents[1];
    reservedWireName.name = "gain;drop";
    check(!shadercatalog::validateParameterMetadata(&reservedWireName, 1, error),
          "parameter names reject wire-grammar delimiters");
    const auto generatorDefaults = shadercatalog::defaultParameterWire(*generatorEntry);
    const auto filterDefaults = shadercatalog::defaultParameterWire(*filterEntry);
    const auto generatorXml = payloadV2("generator", "isf", generatorSource,
                                       "arbit-isf-starters-v1", "plasma", generatorDefaults);
    const auto filterXml = payloadV2("filter", "isf", filterSource,
                                    "vidvox-isf", "rgb-invert", filterDefaults);
    const auto multipassXml = payloadV2(
        "generator", "isf", multipassSource, "arbit-isf-starters-v1", "light_trails",
        shadercatalog::defaultParameterWire(*multipassEntry));

    shadertransition::Payload transitionPayload;
    transitionPayload.catalogPackId = "vidvox-isf";
    transitionPayload.catalogProgramId = "fade";
    transitionPayload.sourceSha256 = videohelper::sha256Text(transitionSource);
    transitionPayload.source = transitionSource;
    transitionPayload.progress = 0.25;
    transitionPayload.parameters = shadertransition::parametersForProgress(
        *transitionEntry, transitionPayload.progress, error);
    transitionPayload.direction = shadertransition::Direction::reverse;
    transitionPayload.easing = shadertransition::Easing::easeInOut;
    const auto transitionXml = shadertransition::serialize(transitionPayload, error);
    check(!transitionXml.empty() && error.empty(),
          "valid curated transition payload serializes canonically");

    videowire::AdmittedCuratedShaderTransition admittedTransition;
    check(videowire::admitCuratedShaderTransitionPayload(
              transitionXml, admittedTransition, error),
          "exact curated transition payload is admitted");
    check(admittedTransition.payload.source == transitionSource
              && admittedTransition.payload.parameters == "progress:float=0.25"
              && admittedTransition.payload.progress == 0.25
              && admittedTransition.payload.direction == shadertransition::Direction::reverse
              && admittedTransition.payload.easing == shadertransition::Easing::easeInOut,
          "transition source, catalog parameters, and controls survive the canonical wire schema");
    for (const auto& identity : shadertransition::curatedIdentities)
    {
        const auto* catalog = shadercatalog::find(identity.packId, identity.programId);
        check(catalog != nullptr && catalog->kind == "isf"
                  && catalog->sourceSha256 == identity.sourceSha256
                  && shadercatalog::parameterAt(*catalog, 0).name == "progress",
              "transition execution allowlist matches exact generated catalog metadata");
    }

    videowire::FlatShaderBridgePayload parsed;
    check(videowire::parseFlatShaderBridgePayload(generatorXml, parsed, error),
          "valid curated generator payload admitted");
    check(parsed.source == generatorSource, "XML entities round-trip to exact source bytes");
    check(parsed.schemaVersion == 2, "schema v2 decoded");
    check(parsed.role == videowire::FlatShaderRole::generator, "generator role decoded");
    check(parsed.language == videowire::FlatShaderLanguage::isf, "ISF language decoded");
    check(parsed.parameterValues.at("scale") == 3.0
          && parsed.parameterValues.at("speed") == 1.0,
          "compiled catalog scalar defaults decoded");
    std::map<std::string, double> published {
        { "visual41/brightness", 0.25 },
        { "visual99/brightness", 0.75 }
    };
    std::map<std::string, double> runtimeValues;
    check(videowire::applyFlatShaderRuntimeParameters(
              parsed, 41, published, runtimeValues, error)
              && runtimeValues.at("brightness") == 0.25
              && runtimeValues.at("scale") == 3.0,
          "preview and export overlay the exact visual-node runtime identifier");
    published["visual41/unknown"] = 0.5;
    check(!videowire::applyFlatShaderRuntimeParameters(
              parsed, 41, published, runtimeValues, error)
              && error.find("unknown") != std::string::npos,
          "runtime publication rejects an unknown shader parameter");
    published.erase("visual41/unknown");
    published["visual41/brightness"] = std::numeric_limits<double>::quiet_NaN();
    check(!videowire::applyFlatShaderRuntimeParameters(
              parsed, 41, published, runtimeValues, error)
              && error.find("not finite") != std::string::npos,
          "runtime publication rejects nonfinite shader values");
    published["visual41/brightness"] = 2.1;
    check(!videowire::applyFlatShaderRuntimeParameters(
              parsed, 41, published, runtimeValues, error)
              && error.find("outside") != std::string::npos,
          "runtime publication rejects values outside the catalog range");
    const std::vector<videowire::FlatShaderRuntimeParameter> generatorRuntimeParameters {
        { "scale", "float", 1 }, { "speed", "float", 1 },
        { "hueShift", "float", 1 }, { "brightness", "float", 1 }
    };
    check(videowire::validateFlatShaderRuntimeParameters(
              *generatorEntry, generatorRuntimeParameters, parsed.parameterValues, error),
          "catalog-order runtime parameters reach the shader bridge");
    auto reorderedRuntimeParameters = generatorRuntimeParameters;
    std::swap(reorderedRuntimeParameters[0], reorderedRuntimeParameters[1]);
    check(!videowire::validateFlatShaderRuntimeParameters(
              *generatorEntry, reorderedRuntimeParameters, parsed.parameterValues, error)
              && error.find("catalog order") != std::string::npos,
          "shader bridge rejects runtime parameter reordering");
    auto incompleteRuntimeValues = parsed.parameterValues;
    incompleteRuntimeValues.erase("brightness");
    check(!videowire::validateFlatShaderRuntimeParameters(
              *generatorEntry, generatorRuntimeParameters, incompleteRuntimeValues, error)
              && error.find("incomplete") != std::string::npos,
          "shader bridge rejects an incomplete immutable parameter set");

    check(videowire::parseFlatShaderBridgePayload(filterXml, parsed, error),
          "valid curated ISF filter payload admitted");
    check(parsed.parameterValues.at("r") == 1.0 && parsed.parameterValues.at("a") == 0.0,
          "compiled catalog bool defaults decoded");
    check(filterSource.find("inputImage") != std::string::npos,
          "filter fixture declares exact inputImage sampler");

    check(videowire::parseFlatShaderBridgePayload(multipassXml, parsed, error),
          "exact curated multipass payload admitted");
    check(parsed.exactPayloadXml == multipassXml
              && parsed.passResources.passes.size() == 2
              && parsed.passResources.passes[0].target == "trail"
              && parsed.passResources.passes[0].persistent
              && parsed.passResources.passes[1].target.empty()
              && parsed.passResources.namedTargets == 1
              && parsed.passResources.retainedImages == 2,
          "multipass admission retains exact pass order and bounded persistent resources");
    check(videowire::admitCuratedIsfPassExtent(parsed.passResources, 3840, 2160, error),
          "bounded curated multipass resources admit a 4K render extent");
    check(!videowire::admitCuratedIsfPassExtent(parsed.passResources, 16384, 16384, error)
              && error.find("retained pass bytes") != std::string::npos,
          "curated multipass retained bytes reject an oversized render extent");

    videowire::CuratedIsfPassResources syntheticResources;
    const std::string tooManyPasses = "/*{\"ISFVSN\":\"2\",\"PASSES\":["
        "{\"TARGET\":\"a\"},{\"TARGET\":\"b\"},{\"TARGET\":\"c\"},"
        "{\"TARGET\":\"d\"},{\"TARGET\":\"e\"},{\"TARGET\":\"f\"},"
        "{\"TARGET\":\"g\"},{\"TARGET\":\"h\"},{}]}*/\nvoid main(){}";
    check(!videowire::admitCuratedIsfPassResources(tooManyPasses, syntheticResources, error)
              && error.find("pass count") != std::string::npos,
          "curated multipass admission caps the exact pass schedule");
    const std::string tooManyImages = "/*{\"ISFVSN\":\"2\",\"PASSES\":["
        "{\"TARGET\":\"a\",\"PERSISTENT\":true},"
        "{\"TARGET\":\"b\",\"PERSISTENT\":true},"
        "{\"TARGET\":\"c\",\"PERSISTENT\":true},"
        "{\"TARGET\":\"d\",\"PERSISTENT\":true},"
        "{\"TARGET\":\"e\",\"PERSISTENT\":true},{}]}*/\nvoid main(){}";
    check(!videowire::admitCuratedIsfPassResources(tooManyImages, syntheticResources, error)
              && error.find("retained pass images") != std::string::npos,
          "curated multipass admission caps persistent ping-pong images");

    videowire::VisualLayerExecution execution;
    check(videowire::compileVisualLayerExecution(generatorPlan(generatorXml), execution, error),
          "exact generator topology lowers");
    check(execution.flatShaderBridge && execution.shaderOperationPlan != nullptr
              && execution.shaderOperationPlan->operations.size() == 1
              && execution.shaderOperationPlan->operations.front().kind
                  == videowire::ShaderOperationKind::generator,
          "generator lowering retains bridge role");
    check(execution.shaderOperationPlan->operations.front().nodeId == 41
              && execution.shaderOperationPlan->operations.front().payload.sourceSha256
                  == videohelper::sha256Text(generatorSource),
          "generator lowering retains stable node and source identity");
    check(execution.shaderOperationPlan->operations.front().generatedParameters.at("brightness") == 1.0,
          "generator lowering retains validated parameter values");
    std::map<std::string, double> modulationBases;
    videowire::seedFlatShaderRuntimeParameters(&execution, modulationBases);
    check(modulationBases.at("visual41/brightness") == 1.0,
          "modulation reads the same visual-node parameter identifier as publication");

    check(videowire::compileVisualLayerExecution(
              generatorPlan(multipassXml), execution, error),
          "exact curated multipass generator topology lowers");
    check(execution.shaderOperationPlan != nullptr
              && execution.shaderOperationPlan->operations.front().nodeId == 41
              && execution.shaderOperationPlan->operations.front().inputCount == 0
              && execution.shaderOperationPlan->operations.front().payload.exactPayloadXml == multipassXml
              && execution.shaderOperationPlan->operations.front().payload.passResources.retainedImages == 2,
          "graph lowering retains one immutable exact multipass operation");
    auto changedRetainedPayload = execution.shaderOperationPlan->operations.front().payload;
    changedRetainedPayload.parameterValues["feedback"] = 0.0;
    check(!videowire::sameExactFlatShaderPayload(
              changedRetainedPayload, execution.shaderOperationPlan->operations.front().payload),
          "renderer exact-payload comparison rejects drift after admission");

    check(videowire::compileVisualLayerExecution(filterPlan(filterXml), execution, error),
          "exact filter topology lowers");
    check(execution.flatShaderBridge && execution.shaderOperationPlan != nullptr
              && execution.shaderOperationPlan->operations.front().kind
                  == videowire::ShaderOperationKind::filter,
          "filter lowering retains bridge role");
    check(execution.shaderOperationPlan->operations.front().nodeId == 41
              && execution.shaderOperationPlan->operations.front().inputNodeIds[0] == 40
              && execution.shaderOperationPlan->operations.front().payload.catalogPackId == "vidvox-isf"
              && execution.shaderOperationPlan->operations.front().payload.catalogProgramId == "rgb-invert"
              && execution.shaderOperationPlan->operations.front().generatedParameters.at("r") == 1.0,
          "filter lowering retains one immutable exact catalog operation");

    check(videowire::compileVisualLayerExecution(
              shaderChainPlan(generatorXml, filterXml, false), execution, error)
              && execution.shaderOperationPlan->operations.size() == 2
              && execution.shaderOperationPlan->operations[1].inputNodeIds[0] == 60,
          "generator to filter to Output lowers in exact execution order");
    check(videowire::compileVisualLayerExecution(
              shaderChainPlan(generatorXml, filterXml, true), execution, error)
              && execution.shaderOperationPlan->operations.size() == 2
              && execution.shaderOperationPlan->operations[0].inputNodeIds[0] == 60
              && execution.shaderOperationPlan->operations[1].inputNodeIds[0] == 61,
          "Source to filter A to filter B to Output retains each resource identity");

    const auto savedCompositePlan = filterCompositePlan(filterXml);
    check(!videowire::compileVisualLayerExecution(savedCompositePlan, execution, error),
          "legacy composite shader placement fails closed instead of selecting one singleton operation");

    auto changedComposite = filterCompositePlan(filterXml);
    changedComposite.operations[1].payloadXml.insert(
        changedComposite.operations[1].payloadXml.size() - 2,
        " runtimeGrant=\"forged\"");
    check(!videowire::compileVisualLayerExecution(changedComposite, execution, error),
          "ordinary DAG coexistence does not admit invented shader authority");

    auto legacyFilter = filterPlan(payloadV1("filter", "isf", filterSource,
                                             "vidvox-isf", "rgb-invert"));
    check(!videowire::compileVisualLayerExecution(legacyFilter, execution, error),
          "catalog filter rejects legacy trust-bearing payload schema");
    check(error.find("exact schemaVersion 2 filter/isf payload") != std::string::npos,
          "legacy filter rejection names the exact payload boundary");

    auto missingFilterPort = filterPlan(filterXml);
    missingFilterPort.ports.pop_back();
    check(!videowire::compileVisualLayerExecution(missingFilterPort, execution, error),
          "catalog filter rejects incomplete port descriptors");
    check(error.find("Frame<Image>") != std::string::npos
              || error.find("input resource") != std::string::npos,
          "filter port rejection names the exact descriptor boundary");

    auto filterFormatMismatch = filterPlan(filterXml);
    filterFormatMismatch.ports[1].pixelFormat = "rgba16f";
    check(!videowire::compileVisualLayerExecution(filterFormatMismatch, execution, error),
          "filter rejects an output pixel format different from its input");
    auto filterColorSpaceMismatch = filterPlan(filterXml);
    filterColorSpaceMismatch.ports[1].colorSpace = "linearSRGB";
    check(!videowire::compileVisualLayerExecution(filterColorSpaceMismatch, execution, error),
          "filter rejects an output color space different from its input");

    check(videowire::compileVisualLayerExecution(
              transitionPlan(transitionXml), execution, error),
          "admitted transition lowers to the immutable helper operation");
    check(execution.shaderOperationPlan != nullptr
              && execution.shaderOperationPlan->operations.size() == 1
              && execution.shaderOperationPlan->operations.front().kind
                  == videowire::ShaderOperationKind::transition
              && execution.shaderOperationPlan->operations.front().nodeId == 52
              && execution.shaderOperationPlan->operations.front().inputNodeIds[0] == 51
              && execution.shaderOperationPlan->operations.front().inputNodeIds[1] == 50
              && execution.shaderOperationPlan->operations.front().payload.source == transitionSource
              && execution.shaderOperationPlan->operations.front().transitionPayload.has_value()
              && execution.shaderOperationPlan->operations.front().transitionPayload->direction
                  == shadertransition::Direction::reverse
              && execution.shaderOperationPlan->operations.front().transitionPayload->easing
                  == shadertransition::Easing::easeInOut
              && shadertransition::evaluateProgress(
                     execution.shaderOperationPlan->operations.front().transitionPayload->progress,
                     execution.shaderOperationPlan->operations.front().transitionPayload->direction,
                     execution.shaderOperationPlan->operations.front().transitionPayload->easing)
                  == 0.9375,
          "transition operation retains exact source and ordered frame bindings");
    check(execution.shaderOperationPlan->digest.size() == 64
              && execution.shaderOperationPlan->digest == videowire::shaderOperationPlanDigest(
                  *execution.shaderOperationPlan, execution.shaderOperationPlan->revision),
          "ordered operation content has one deterministic immutable plan digest");
    auto changedIdentity = *execution.shaderOperationPlan;
    changedIdentity.operations.front().inputNodeIds[0] = 5001;
    check(videowire::shaderOperationPlanDigest(changedIdentity, changedIdentity.revision)
              != execution.shaderOperationPlan->digest,
          "resource identity changes cannot reuse an admitted plan identity");

    check(videowire::compileVisualLayerExecution(
              chainedTransitionPlan(generatorXml, transitionXml, filterXml),
              execution, error),
          "transition lowers inside a bounded generator and filter chain");
    check(execution.shaderOperationPlan != nullptr
              && execution.shaderOperationPlan->operations.size() == 3
              && execution.shaderOperationPlan->operations[0].kind
                  == videowire::ShaderOperationKind::generator
              && execution.shaderOperationPlan->operations[1].kind
                  == videowire::ShaderOperationKind::transition
              && execution.shaderOperationPlan->operations[1].inputNodeIds[0] == 70
              && execution.shaderOperationPlan->operations[1].inputNodeIds[1] == 71
              && execution.shaderOperationPlan->operations[2].kind
                  == videowire::ShaderOperationKind::filter
              && execution.shaderOperationPlan->operations[2].inputNodeIds[0] == 72,
          "ordered chain retains execution order and both transition resources");

    auto sharedProducer = transitionPlan(transitionXml);
    sharedProducer.edges[1].fromNodeId = sharedProducer.edges[0].fromNodeId;
    check(!videowire::compileVisualLayerExecution(sharedProducer, execution, error),
          "transition rejects one producer wired to both inputs");
    check(error.find("two distinct exact frame producers") != std::string::npos,
          "transition producer topology rejection is precise");

    auto transitionFormatMismatch = transitionPlan(transitionXml);
    transitionFormatMismatch.ports[4].pixelFormat = "rgba16f";
    check(!videowire::compileVisualLayerExecution(transitionFormatMismatch, execution, error),
          "transition rejects an output pixel format different from both inputs");
    auto transitionColorSpaceMismatch = transitionPlan(transitionXml);
    transitionColorSpaceMismatch.ports[3].colorSpace = "linearSRGB";
    check(!videowire::compileVisualLayerExecution(transitionColorSpaceMismatch, execution, error),
          "transition rejects a second-input color space different from its first input and output");

    auto extraTransitionPort = transitionPlan(transitionXml);
    extraTransitionPort.ports.push_back(
        { 52, 3, 1, "in", "frame", "image", "rgba8", "srgb" });
    check(!videowire::compileVisualLayerExecution(extraTransitionPort, execution, error),
          "transition rejects undeclared frame ports");
    check(error.find("exactly two Frame<Image> inputs") != std::string::npos,
          "transition port rejection is precise");

    auto nonTransitionPayload = transitionPayload;
    nonTransitionPayload.catalogProgramId = "rgb-invert";
    check(shadertransition::serialize(nonTransitionPayload, error).empty(),
          "general curated shader catalog entries do not gain transition authority");
    check(error.find("exact curated catalog entry") != std::string::npos,
          "transition authority rejection names the curated identity boundary");

    auto changedTransition = transitionPayload;
    changedTransition.source += "\n// changed";
    changedTransition.sourceSha256 = videohelper::sha256Text(changedTransition.source);
    check(shadertransition::serialize(changedTransition, error).empty(),
          "modified source cannot retain a curated transition identity");

    auto mismatchedProgress = transitionPayload;
    mismatchedProgress.progress = 0.5;
    check(shadertransition::serialize(mismatchedProgress, error).empty(),
          "transition progress cannot disagree with catalog parameters");
    check(error.find("exactly match") != std::string::npos,
          "transition progress mismatch rejection is precise");

    auto duplicateParameters = transitionPayload;
    duplicateParameters.parameters += ";progress:float=0.25";
    check(shadertransition::serialize(duplicateParameters, error).empty(),
          "duplicate transition parameters are rejected");

    auto inventedTransitionAttribute = transitionXml;
    inventedTransitionAttribute.insert(inventedTransitionAttribute.size() - 2,
                                       " runtimeGrant=\"forged\"");
    check(!videowire::admitCuratedShaderTransitionPayload(
              inventedTransitionAttribute, admittedTransition, error),
          "invented transition authority attribute is rejected");
    check(admittedTransition.payload.source.empty(),
          "failed transition admission clears any prior admitted payload");
    check(error.find("exact canonical schema") != std::string::npos,
          "invented transition authority rejection is precise");

    auto wrongTopology = filterPlan(filterXml);
    wrongTopology.edges.pop_back();
    check(!videowire::compileVisualLayerExecution(wrongTopology, execution, error),
          "filter missing output edge rejected");
    check(error.find("Output input") != std::string::npos
              || error.find("exact output resource") != std::string::npos,
          "filter topology rejection is precise");

    auto fanIn = filterPlan(filterXml);
    fanIn.edges.push_back({ 40, 0, 41, 0 });
    check(!videowire::compileVisualLayerExecution(fanIn, execution, error),
          "filter duplicate frame input rejected");

    const std::string customSource =
        "void mainImage(out vec4 c, in vec2 p) { c = vec4(0.25, 0.5, 1.0, 1.0); }";
    programmableruntime::SessionSecret customSecret {};
    for (size_t i = 0; i < customSecret.size(); ++i)
        customSecret[i] = static_cast<uint8_t>(i * 5 + 1);
    const uint64_t customGeneration = 7001;
    const auto makeCustomGrant = [&](const std::string& payload, uint64_t nonce)
    {
        programmableruntime::Grant grant;
        grant.kind = programmableruntime::PayloadKind::shader;
        grant.fingerprint = programmableruntime::fingerprint(grant.kind, payload);
        grant.sessionGeneration = customGeneration;
        grant.nonce = nonce;
        grant.issuedAtMs = programmableadmission::nowMs();
        grant.approved = true;
        grant.cpuMs = programmableruntime::defaultCpuMs;
        grant.gpuMs = programmableruntime::defaultGpuMs;
        grant.memoryMiB = programmableruntime::defaultMemoryMiB;
        grant.mac = programmableruntime::sign(customSecret, grant, grant.kind);
        return grant;
    };

    programmableadmission::verifier().reset(customSecret, customGeneration);
    auto custom = customPlan(customSource, makeCustomGrant(customSource, 1));
    check(videowire::compileVisualLayerExecution(custom, execution, error),
          "approved exact custom shader reaches production execution");
    check(execution.flatShaderBridge && execution.shaderOperationPlan != nullptr
              && execution.shaderOperationPlan->operations.front().customGrant.has_value()
              && execution.shaderOperationPlan->operations.front().payload.source == customSource,
          "custom shader execution preserves exact approved source");

    std::map<std::string, double> customRuntimeValues;
    const std::map<std::string, double> unrelatedRuntimeValues {
        { "visual999/brightness", 0.5 }
    };
    check(applyFlatShaderBridgeRuntimeParameters(
              execution.shaderOperationPlan->operations.front().payload,
              execution.shaderOperationPlan->operations.front().nodeId,
              execution.shaderOperationPlan->operations.front().customGrant.has_value(), unrelatedRuntimeValues,
              customRuntimeValues, error)
              && customRuntimeValues.empty(),
          "approved custom shader ignores unrelated runtime parameter publication");
    const std::map<std::string, double> targetedCustomRuntimeValue {
        { "visual41/brightness", 0.5 }
    };
    check(!applyFlatShaderBridgeRuntimeParameters(
              execution.shaderOperationPlan->operations.front().payload,
              execution.shaderOperationPlan->operations.front().nodeId,
              execution.shaderOperationPlan->operations.front().customGrant.has_value(), targetedCustomRuntimeValue,
              customRuntimeValues, error),
          "custom shader rejects invented catalog runtime parameters");
    check(error.find("does not expose catalog runtime parameters") != std::string::npos,
          "custom runtime parameter rejection is precise");

    auto missingGrant = custom;
    missingGrant.operations[0].runtimeGrantJson.clear();
    check(!videowire::compileVisualLayerExecution(missingGrant, execution, error),
          "custom shader without runtime grant rejected");
    check(error == "custom shader runtime grant is missing",
          "missing grant rejection names required authority");

    check(!videowire::compileVisualLayerExecution(custom, execution, error),
          "replayed custom shader grant rejected");
    check(error.find("replay") != std::string::npos,
          "custom replay rejection names consumed authority");

    auto alteredCustom = customPlan(customSource + "\n// altered",
                                    makeCustomGrant(customSource, 2));
    check(!videowire::compileVisualLayerExecution(alteredCustom, execution, error),
          "source altered after approval rejected");
    check(error.find("fingerprint") != std::string::npos,
          "altered source rejection names exact-payload mismatch");

    auto forgedCustom = customPlan(customSource, makeCustomGrant(customSource, 3));
    auto forgedWire = nlohmann::json::parse(forgedCustom.operations[0].runtimeGrantJson);
    forgedWire["mac"] = std::string(64, '0');
    forgedCustom.operations[0].runtimeGrantJson = forgedWire.dump();
    check(!videowire::compileVisualLayerExecution(forgedCustom, execution, error),
          "forged persisted custom shader grant rejected");
    check(error.find("authentication") != std::string::npos,
          "forged grant rejection names authentication failure");

    programmableruntime::SessionSecret reopenedSecret = customSecret;
    reopenedSecret[0] ^= 0xff;
    programmableadmission::verifier().reset(reopenedSecret, customGeneration + 1);
    auto reopenedCustom = customPlan(customSource, makeCustomGrant(customSource, 4));
    check(!videowire::compileVisualLayerExecution(reopenedCustom, execution, error),
          "custom shader grant from prior helper session rejected");
    check(error.find("session") != std::string::npos,
          "reopened-session rejection names session mismatch");

    auto wrongRole = generatorPlan(filterXml);
    check(!videowire::compileVisualLayerExecution(wrongRole, execution, error),
          "operation kind and payload role mismatch rejected");
    check(error.find("role does not match") != std::string::npos,
          "role mismatch rejection is precise");

    auto replaceOnce = [](std::string value, const std::string& from, const std::string& to)
    {
        const auto position = value.find(from);
        if (position != std::string::npos) value.replace(position, from.size(), to);
        return value;
    };

    expectRejected(replaceOnce(generatorXml, " schemaVersion=", " future=\"1\" schemaVersion="),
                   "unknown attribute", "unknown attribute rejected");
    expectRejected(replaceOnce(generatorXml, " role=", " role=\"generator\" role="),
                   "duplicate attribute", "duplicate attribute rejected");
    expectRejected(replaceOnce(generatorXml, " schemaVersion=\"2\"", ""),
                   "missing required", "missing schema rejected");
    expectRejected(replaceOnce(generatorXml, "schemaVersion=\"2\"", "schemaVersion=\"3\""),
                   "schemaVersion", "wrong schema rejected");
    expectRejected(replaceOnce(generatorXml, "language=\"isf\"", "language=\"spirv\""),
                   "language", "wrong language rejected");
    expectRejected(replaceOnce(generatorXml, "sourceSha256=\"", "sourceSha256=\"A"),
                   "64 lowercase", "malformed hash rejected");
    expectRejected(replaceOnce(generatorXml, "source=\"", "source=\"x"),
                   "source hash mismatch", "decoded source hash mismatch rejected");
    expectRejected(replaceOnce(generatorXml, "catalogProgramId=\"plasma\"", "catalogProgramId=\"warp_tunnel\""),
                   "catalog source digest mismatch", "catalog source mismatch rejected");
    expectRejected(replaceOnce(generatorXml, "language=\"isf\"", "language=\"glsl\""),
                   "language and catalog kind mismatch", "catalog language mismatch rejected");
    expectRejected(replaceOnce(generatorXml, " parameters=", " runtimeGrant=\"forged\" parameters="),
                   "unknown attribute", "invented runtime grant rejected");
    expectRejected(replaceOnce(generatorXml, " parameters=", " sourcePath=\"/tmp/forged\" parameters="),
                   "unknown attribute", "invented source path rejected");
    expectRejected(replaceOnce(generatorXml, " parameters=", " authority=\"curated\" parameters="),
                   "unknown attribute", "invented authority label rejected");
    expectRejected(replaceOnce(generatorXml, " parameters=", " paramGain=\"1\" parameters="),
                   "unknown attribute", "invented parameter encoding rejected");
    expectRejected(generatorXml.substr(0, generatorXml.size() - 2) + ">child</FlatShaderBridge>",
                   "self-closing", "nested payload rejected");
    expectRejected(payloadV1("generator", "isf", generatorSource, "arbit-isf-starters-v1",
                             "plasma", "review-metadata-only"),
                   "compatibility value", "review-only trust rejected");

    const auto legacyXml = payloadV1("generator", "isf", generatorSource,
                                     "arbit-isf-starters-v1", "plasma");
    check(videowire::parseFlatShaderBridgePayload(legacyXml, parsed, error)
          && parsed.schemaVersion == 1 && parsed.parameterValues.empty(),
          "schema v1 remains accepted without parameters");
    expectRejected(replaceOnce(legacyXml, " trust=", " parameters=\"\" trust="),
                   "forbids parameters", "schema v1 rejects parameter transport");
    expectRejected(replaceOnce(generatorXml, " parameters=", " trust=\"verified-bundled-curated\" parameters="),
                   "forbids trust", "schema v2 rejects trust labels");
    expectRejected(replaceOnce(generatorXml, " parameters=\"" + xmlEscape(generatorDefaults) + "\"", ""),
                   "requires parameters", "schema v2 requires explicit parameters");

    expectRejected(payloadV2("generator", "isf", generatorSource, "arbit-isf-starters-v1",
                             "plasma", generatorDefaults + ";scale:float=3"),
                   "duplicate or unknown", "duplicate parameter rejected");
    expectRejected(payloadV2("generator", "isf", generatorSource, "arbit-isf-starters-v1",
                             "plasma", generatorDefaults + ";"),
                   "trailing separator", "trailing parameter separator rejected");
    expectRejected(payloadV2("generator", "isf", generatorSource, "arbit-isf-starters-v1",
                             "plasma", replaceOnce(generatorDefaults, "scale:float", "unknown:float")),
                   "exact catalog order", "unknown parameter rejected");
    expectRejected(payloadV2("generator", "isf", generatorSource, "arbit-isf-starters-v1",
                             "plasma", replaceOnce(generatorDefaults, "scale:float", "scale:long")),
                   "type does not match", "parameter type mismatch rejected");
    expectRejected(payloadV2("generator", "isf", generatorSource, "arbit-isf-starters-v1",
                             "plasma", replaceOnce(generatorDefaults, "scale:float=3", "scale:float=13")),
                   "outside the catalog range", "out-of-range scalar rejected");
    expectRejected(payloadV2("generator", "isf", generatorSource, "arbit-isf-starters-v1",
                             "plasma", replaceOnce(generatorDefaults, "scale:float=3", "scale:float=nan")),
                   "nonfinite", "nonfinite scalar rejected");
    expectRejected(payloadV2("generator", "isf", generatorSource, "arbit-isf-starters-v1",
                             "plasma", replaceOnce(generatorDefaults, "scale:float=3", "scale=3")),
                   "syntax is malformed", "malformed parameter item rejected");

    const std::string vectorValues = "center:point2D=0.25,0.75;count:long=12;"
        "size:float=0.1;spread:float=0.4;color:color=0.1,0.2,0.3,0.4";
    const auto vectorXml = payloadV2("generator", "isf", vectorSource,
                                     "arbit-isf-starters-v1", "orbit", vectorValues);
    check(videowire::parseFlatShaderBridgePayload(vectorXml, parsed, error)
          && parsed.parameterValues.at("center.x") == 0.25
          && parsed.parameterValues.at("center.y") == 0.75
          && parsed.parameterValues.at("count") == 12.0
          && parsed.parameterValues.at("color.a") == 0.4,
          "integer and vector values use renderer component keys");
    expectRejected(payloadV2("generator", "isf", vectorSource, "arbit-isf-starters-v1",
                             "orbit", replaceOnce(vectorValues, "count:long=12", "count:long=8.5")),
                   "wrong type", "fractional integer rejected");
    expectRejected(payloadV2("generator", "isf", vectorSource, "arbit-isf-starters-v1",
                             "orbit", replaceOnce(vectorValues, "center:point2D=0.25,0.75",
                                                  "center:point2D=0.25")),
                   "component count", "short vector rejected");
    expectRejected(payloadV2("filter", "isf", filterSource, "vidvox-isf", "rgb-invert",
                             replaceOnce(filterDefaults, "r:bool=true", "r:bool=1")),
                   "wrong type", "numeric bool rejected");

    if (failures != 0)
    {
        std::cerr << failures << " FlatShaderBridge checks failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Flat shader and curated transition contract checks passed\n";
    return EXIT_SUCCESS;
}
