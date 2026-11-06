// Depth Blur Focus follows live modulation: the executor seeds "visual<node>/focus"
// from the saved focus, and a runtime value under that key replaces it in the layer.
#include "../src/visual_plan_executor.h"

#include <cstdio>
#include <cstdlib>
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
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

struct FakeLayer
{
    struct Clock { double time = 0.0, timeSec = 0.0, timeDelta = 0.0, beat = 0.0; int frame = 0; bool playing = false; };
    struct Effect { bool enabled = false; int type = -1; float params[9] {}; };
    Clock shaderClock;
    std::map<std::string, double> genParams;
    unsigned texture = 0;
    int texWidth = 0, texHeight = 0;
    std::string nativeTextureBackend;
    std::uintptr_t nativeTextureView = 0;
    arbitgpu::NativeTextureViewDescriptor nativeTextureDescriptor;
    std::shared_ptr<const void> nativeTextureOwner;
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> canonicalBlockCFrame;
    int shaderTransitionFromClipId = 0;
    bool shaderSource = false;
    bool scoreSource = false;
    bool isAdjustment = false;
    bool flatShaderBridge = false;
    videowire::ImmutableShaderOperationPlan shaderOperationPlan;
    std::map<int, std::map<std::string, double>> shaderOperationParameters;
    bool particleSource = false;
    bool importedParticleOverlay = false;
    bool particleStateReset = false;
    bool particleTriggerConnected = false;
    videorender::ParticleParams particleParameters;
    std::shared_ptr<const videorender::ParticleHistorySource> particleHistory;
    double particleProjectSeconds = 0.0;
    bool audioPresent = false;
    struct Audio { float rms = 0.0f, onset = 0.0f, onsetAge = 0.0f; } audioFeatures;
    int particleTriggerCount = 0;
    float particleTriggerStrength = 0.0f;
    std::uint64_t visualPlanStructuralRevision = 0;
    bool visualPlanTelemetryHold = false;
    bool graphTemporalActive = false;
    int graphTemporalNodeId = 0;
    visualtemporaloperation::Payload graphTemporalPayload;
    int particleNodeId = 0;
    int drawShapeNodeId = 0;
    bool drawShape = false;
    bool inspectionDrawShapeOutput = false;
    bool drawShapeEllipse = false;
    bool drawShapeHasSecondary = false;
    bool drawShapeSecondaryEllipse = false;
    int drawShapeOperation = 0;
    float drawShapeCx = 0.5f, drawShapeCy = 0.5f, drawShapeW = 1.0f, drawShapeH = 1.0f;
    float drawShape2Cx = 0.5f, drawShape2Cy = 0.5f, drawShape2W = 0.0f, drawShape2H = 0.0f;
    float drawShapeR = 1.0f, drawShapeG = 1.0f, drawShapeB = 1.0f, drawShapeA = 1.0f;
    bool drawShapeGradient = false;
    float drawGradientCos = 1.0f, drawGradientSin = 0.0f;
    float drawGradientR = 1.0f, drawGradientG = 1.0f, drawGradientB = 1.0f, drawGradientA = 1.0f;
    bool pathMatte = false;
    bool pathMatteEllipse = false, pathMatteHasSecondary = false;
    bool pathMatteSecondaryEllipse = false, pathMatteInvert = false;
    int pathMatteOperation = 0;
    float pathMatteCx = 0.5f, pathMatteCy = 0.5f, pathMatteW = 1.0f, pathMatteH = 1.0f;
    float pathMatte2Cx = 0.5f, pathMatte2Cy = 0.5f, pathMatte2W = 0.0f, pathMatte2H = 0.0f;
    float scale = 2.0f, translateX = 1.0f, translateY = 1.0f, rotationDeg = 4.0f;
    float cropLeft = 0.1f, cropRight = 0.1f, cropTop = 0.1f, cropBottom = 0.1f;
    const Effect* effects = reinterpret_cast<const Effect*> (1);
    int effectCount = 2;
    Effect graphFeedbackEffect;
    Effect graphKeyCleanupEffect;
    Effect graphLumaKeyEffect;
    Effect graphCommonEffect;
    bool graphKeyCleanupActive = false;
    colortransform::Description graphColorTransform;
    bool graphColorTransformActive = false;
    bool graphMotionBlurActive = false;
    unsigned graphMotionTexture = 0;
    visualtemporalsampling::Payload graphMotionBlur;
    float graphKeyCleanupChoke = 0.0f, graphKeyCleanupFeather = 0.0f;
    float graphKeyCleanupEdgeR = 1.0f, graphKeyCleanupEdgeG = 1.0f;
    float graphKeyCleanupEdgeB = 1.0f, graphKeyCleanupEdgeAmount = 0.0f;
    bool graphKeyCleanupMatteView = false;
    bool feedbackHistoryReset = false;
    bool feedbackHistoryHold = false;
    unsigned lutTexture = 4;
    int lutSize = 17;
    int maskType = 2;
    bool maskInvert = true;
    unsigned matteTexture = 0;
    unsigned matteTextureB = 0;
    int matteWidth = 0, matteHeight = 0;
    int matteWidthB = 0, matteHeightB = 0;
    bool matteApply = false, matteInvert = false;
    int matteCombineMode = -1;
    float matteBlack = 0.0f, matteWhite = 1.0f;
    float matteErodeDilate = 0.0f, matteFeather = 0.0f, matteChoke = 0.0f;
    unsigned depthTexture = 0;
    int depthWidth = 0, depthHeight = 0;
    bool depthFog = false;
    int depthEffect = 0;
    float fogNear = 0.0f, fogFar = 1.0f, fogDensity = 1.0f;
    float fogRed = 1.0f, fogGreen = 1.0f, fogBlue = 1.0f, fogAlpha = 1.0f;
    float depthParam0 = 0.0f, depthParam1 = 0.0f, depthParam2 = 0.0f;
    float depthColorRed = 1.0f, depthColorGreen = 1.0f, depthColorBlue = 1.0f;
    float opacity = 0.4f;
    int blendMode = 3;
};
// Source -> Depth Blur (node 62) <- depth asset (61) -> Out, the exact depth topology.
videowire::CompiledVisualLayerPlan depthBlurPlan()
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 7;
    plan.producerValidated = true;
    plan.nodeKinds = { "video.source", "visual.depth.asset", "visual.depth.blur", "video.out" };
    plan.nodeIds = { 11, 61, 62, 12 };
    plan.ports = {
        { 11, 0, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 61, 0, 0, "out", "none", "unspecified", "unspecified", "unspecified" },
        { 61, 1, 1, "out", "frame", "depth", "r16", "linearSRGB" },
        { 62, 0, 1, "in", "frame", "image", "rgba8", "sRGB" },
        { 62, 1, 1, "in", "frame", "depth", "r16", "linearSRGB" },
        { 62, 2, 1, "out", "frame", "image", "rgba8", "sRGB" },
        { 12, 0, 1, "in", "frame", "image", "rgba8", "sRGB" }
    };
    plan.edges = { { 11, 0, 62, 0 }, { 61, 1, 62, 1 }, { 62, 2, 12, 0 } };
    plan.operations = {
        { 11, "video.source", "source-decode", "" },
        { 61, "visual.depth.asset", "source-decode",
          "<DepthAssetBinding depthAssetId=\"depth-1\" depthAssetVersion=\"depth-sequence-v1\" state=\"available\" cacheKey=\"depth-job-1-abc/published\" contentReceipt=\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" analysisReceipt=\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\" framePrefix=\"depth_\" frameExtension=\".png\" firstFrame=\"1\" frameDigits=\"6\" width=\"2\" height=\"2\" fps=\"24\" frames=\"3\" format=\"r16-unorm\"/>" },
        { 62, "visual.depth.blur", "native-gpu", "<NodeParams radius=\"4\" focus=\"0.25\" falloff=\"0.5\"/>" },
        { 12, "video.out", "native-gpu", "" }
    };
    return plan;
}

bool run(const videowire::CompiledVisualLayerPlan& plan, const std::map<std::string, double>* runtime, FakeLayer& layer)
{
    layer.depthTexture = 9;
    layer.depthWidth = layer.depthHeight = 2;
    std::string error;
    const bool ok = videowire::executeVisualLayerPlan({ plan }, 7, layer, error,
        videohelper::geometry::PlanUse::preview, nullptr, nullptr, nullptr, 0.0, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, runtime);
    if (!ok) std::fprintf(stderr, "execution error: %s\n", error.c_str());
    return ok;
}
} // namespace

int main()
{
    const auto plan = depthBlurPlan();

    videowire::VisualLayerExecution compiled;
    std::string error;
    check(videowire::compileVisualLayerExecution(plan, compiled, error) && compiled.depthEffect == 2,
          "the Depth Blur plan compiles to the blur mode");
    std::map<std::string, double> seeded;
    videowire::seedDepthEffectRuntimeParameters(&compiled, seeded);
    check(seeded.size() == 1 && seeded.count("visual62/focus") == 1 && seeded["visual62/focus"] == 0.25,
          "the saved focus seeds the live parameter under the node's alias");

    FakeLayer saved;
    check(run(plan, nullptr, saved) && saved.depthParam1 == 0.25f, "without live values the saved focus applies");

    std::map<std::string, double> live { { "visual62/focus", 0.75 } };
    FakeLayer modulated;
    check(run(plan, &live, modulated) && modulated.depthParam1 == 0.75f, "a live focus replaces the saved one");
    check(modulated.depthParam0 == 4.0f && modulated.depthParam2 == 0.5f, "radius and falloff stay saved");

    live["visual62/focus"] = 3.0;
    FakeLayer clamped;
    check(run(plan, &live, clamped) && clamped.depthParam1 == 1.0f, "a live focus clamps to the node range");

    live["visual62/focus"] = std::numeric_limits<double>::quiet_NaN();
    FakeLayer ignored;
    check(run(plan, &live, ignored) && ignored.depthParam1 == 0.25f, "a non-finite live focus keeps the saved one");

    // Fog has no focus, so it seeds nothing.
    auto fog = plan;
    fog.nodeKinds[2] = "visual.depth.fog";
    fog.operations[2].kind = "visual.depth.fog";
    fog.operations[2].payloadXml = "<NodeParams near=\"0\" far=\"1\"/>";
    videowire::VisualLayerExecution fogCompiled;
    std::map<std::string, double> fogSeeded;
    check(videowire::compileVisualLayerExecution(fog, fogCompiled, error), "the fog plan compiles");
    videowire::seedDepthEffectRuntimeParameters(&fogCompiled, fogSeeded);
    check(fogSeeded.empty(), "Depth Fog seeds no focus");

    if (failures != 0) return EXIT_FAILURE;
    std::puts("depth blur live focus tests passed");
    return EXIT_SUCCESS;
}
