#include "../src/imported_animation_visual_plan_execution.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{
int failures = 0;

void check(bool condition, const char* message)
{
    if (condition) return;
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

struct FakeNativeFrame final
{
    std::string backendName = "opengl";
    std::uintptr_t view = 7001;
    std::uint32_t frameWidth = 640;
    std::uint32_t frameHeight = 360;

    const std::string& backend() const noexcept { return backendName; }
    std::uintptr_t colorTextureViewHandle() const noexcept { return view; }
    std::uint32_t width() const noexcept { return frameWidth; }
    std::uint32_t height() const noexcept { return frameHeight; }
};

struct FakeReceipt final
{
    std::shared_ptr<const FakeNativeFrame> frame;
    bool complete = true;
};

struct FakeExecution final
{
    using Receipt = FakeReceipt;

    bool reject = false;
    int previewCalls = 0;
    int exportCalls = 0;
    videohelper::modelpayload::ImportedAnimatedSceneRequest lastRequest;
    std::shared_ptr<const FakeNativeFrame> nextFrame = std::make_shared<FakeNativeFrame>();

    static bool validReceipt(const Receipt& receipt) noexcept
    {
        return receipt.complete && receipt.frame != nullptr;
    }

    static const std::shared_ptr<const FakeNativeFrame>& nativeFrame(
        const Receipt& receipt) noexcept
    {
        return receipt.frame;
    }

    bool executePreview(
        const videohelper::modelpayload::ImportedAnimatedSceneRequest& request,
        Receipt& output,
        std::string& error)
    {
        ++previewCalls;
        return execute(request, output, error);
    }

    bool executeExport(
        const videohelper::modelpayload::ImportedAnimatedSceneRequest& request,
        Receipt& output,
        std::string& error)
    {
        ++exportCalls;
        return execute(request, output, error);
    }

private:
    bool execute(
        const videohelper::modelpayload::ImportedAnimatedSceneRequest& request,
        Receipt& output,
        std::string& error)
    {
        lastRequest = request;
        if (reject)
        {
            error = "fake native imported-animation rejection";
            return false;
        }
        output.frame = nextFrame;
        error.clear();
        return true;
    }
};

struct FakeLayer final
{
    unsigned texture = 99;
    std::string nativeTextureBackend;
    std::uintptr_t nativeTextureView = 0;
    unsigned depthTexture = 0;
    std::string nativeDepthTextureBackend;
    std::uintptr_t nativeDepthTextureView = 0;
    int depthWidth = 0;
    int depthHeight = 0;
    int texWidth = 0;
    int texHeight = 0;
};

videowire::CompiledVisualLayerPlan importedAnimationPlan()
{
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId = 7;
    plan.structuralRevision = 9;
    plan.producerValidated = true;
    plan.nodeKinds = {
        std::string(visualanimationoperation::kSourceNodeKind),
        std::string(visualanimationoperation::kDeformationNodeKind)
    };
    plan.nodeIds = { 70, 71 };
    const std::array<const char*, 4> types {
        "mesh", "skeleton", "morphTargets", "animationClip"
    };
    for (int port = 0; port < 4; ++port)
    {
        plan.ports.push_back({ 70, port, 1, "out", "control",
                              types[static_cast<std::size_t>(port)],
                              "unspecified", "unspecified" });
        plan.ports.push_back({ 71, port, 1, "in", "control",
                              types[static_cast<std::size_t>(port)],
                              "unspecified", "unspecified" });
        plan.edges.push_back({ 70, port, 71, port });
    }
    plan.ports.push_back({ 70, 4, 1, "out", "control", "scene3D",
                           "unspecified", "unspecified" });
    plan.ports.push_back({ 71, 4, 1, "out", "control", "mesh",
                           "unspecified", "unspecified" });

    visualanimationimport::Request request;
    request.sourceStableId = 71;
    request.deformationStableId = 72;
    request.schedule = { 71, 72 };
    request.asset.id = "model-asset-1";
    request.asset.version = 3;
    request.asset.contentSha256 = std::string(64, 'a');
    request.asset.sourceMediaType = "model/gltf-binary";
    request.asset.sourceByteSize = 4096;
    request.sceneIndex = 0;
    request.clipName = "Walk";
    plan.operations = {
        { 70, std::string(visualanimationoperation::kSourceNodeKind),
          std::string(visualanimationoperation::kSourceBackendCapability), "" },
        { 71, std::string(visualanimationoperation::kDeformationNodeKind),
          std::string(visualanimationoperation::kDeformationBackendCapability),
          visualanimationoperation::encode(request) }
    };
    return plan;
}
} // namespace

int main()
{
    using videohelper::importedanimation::NativeImportedAnimationRenderUse;
    using videohelper::importedanimation::VisualImportedAnimationPreparation;
    using videohelper::importedanimation::prepareVisualImportedAnimationLayer;

    std::string error;
    FakeExecution execution;
    FakeLayer layer;
    std::vector<FakeReceipt> owners;

    check(prepareVisualImportedAnimationLayer(
              {}, 7, 640, 360, 1.0, 24.0,
              NativeImportedAnimationRenderUse::Preview,
              &execution, layer, owners, error)
              == VisualImportedAnimationPreparation::notPresent
          && execution.previewCalls == 0 && owners.empty(),
          "a clip without an imported-animation operation remains on its ordinary route");

    const auto plan = importedAnimationPlan();
    auto staleMalformedPlan = plan;
    --staleMalformedPlan.structuralRevision;
    staleMalformedPlan.operations[1].payloadXml.clear();
    check(prepareVisualImportedAnimationLayer(
              { staleMalformedPlan, plan }, 7, 640, 360, 1.0, 24.0,
              NativeImportedAnimationRenderUse::Preview,
              &execution, layer, owners, error)
              == VisualImportedAnimationPreparation::rendered
          && execution.previewCalls == 1
          && execution.lastRequest.structuralRevision == plan.structuralRevision,
          "imported animation ignores a stale duplicate in favor of the newest clip revision");
    execution.previewCalls = 0;
    owners.clear();
    check(prepareVisualImportedAnimationLayer(
              { plan }, 7, 640, 360, 1.0, 24.0,
              NativeImportedAnimationRenderUse::Preview,
              &execution, layer, owners, error)
              == VisualImportedAnimationPreparation::rendered
          && execution.previewCalls == 1 && execution.exportCalls == 0
          && execution.lastRequest.frame.frame == 24
          && execution.lastRequest.frame.rateNumerator == 24
          && execution.lastRequest.frame.rateDenominator == 1
          && execution.lastRequest.structuralRevision == 9
          && execution.lastRequest.width == 640
          && execution.lastRequest.height == 360
          && owners.size() == 1
          && layer.texture == 7001
          && layer.nativeTextureBackend == "opengl"
          && layer.nativeTextureView == 7001
          && layer.texWidth == 640 && layer.texHeight == 360,
          "preview retains and binds the exact native imported-animation frame");

    auto newerOrdinaryPlan = plan;
    ++newerOrdinaryPlan.structuralRevision;
    newerOrdinaryPlan.nodeKinds.clear();
    newerOrdinaryPlan.nodeIds.clear();
    newerOrdinaryPlan.ports.clear();
    newerOrdinaryPlan.edges.clear();
    newerOrdinaryPlan.operations.clear();
    FakeExecution replacedExecution;
    FakeLayer replacedLayer;
    std::vector<FakeReceipt> replacedOwners;
    check(prepareVisualImportedAnimationLayer(
              { plan, newerOrdinaryPlan }, 7, 640, 360, 1.0, 24.0,
              NativeImportedAnimationRenderUse::Preview,
              &replacedExecution, replacedLayer, replacedOwners, error)
              == VisualImportedAnimationPreparation::notPresent
          && replacedExecution.previewCalls == 0 && replacedOwners.empty(),
          "a newer ordinary plan removes stale imported-animation execution");

    auto legacyPlan = plan;
    legacyPlan.ports.erase(
        std::remove_if(legacyPlan.ports.begin(), legacyPlan.ports.end(), [](const auto& port)
        {
            return port.nodeId == 70 && port.port == 4;
        }),
        legacyPlan.ports.end());
    FakeExecution legacyExecution;
    FakeLayer legacyLayer;
    std::vector<FakeReceipt> legacyOwners;
    error.clear();
    check(prepareVisualImportedAnimationLayer(
              { legacyPlan }, 7, 640, 360, 1.0, 24.0,
              NativeImportedAnimationRenderUse::Preview,
              &legacyExecution, legacyLayer, legacyOwners, error)
              == VisualImportedAnimationPreparation::rendered
          && legacyOwners.size() == 1,
          "saved animation plans without the later Scene3D output remain admissible");

    FakeExecution exportExecution;
    auto metalFrame = std::make_shared<FakeNativeFrame>();
    metalFrame->backendName = "metal";
    metalFrame->view = 8002;
    exportExecution.nextFrame = metalFrame;
    FakeLayer exportLayer;
    std::vector<FakeReceipt> exportOwners;
    check(prepareVisualImportedAnimationLayer(
              { plan }, 7, 640, 360, 0.5, 30.0,
              NativeImportedAnimationRenderUse::Export,
              &exportExecution, exportLayer, exportOwners, error)
              == VisualImportedAnimationPreparation::rendered
          && exportExecution.previewCalls == 0 && exportExecution.exportCalls == 1
          && exportExecution.lastRequest.frame.frame == 15
          && exportExecution.lastRequest.frame.rateNumerator == 30
          && exportExecution.lastRequest.frame.rateDenominator == 1
          && exportOwners.size() == 1
          && exportLayer.texture == 0
          && exportLayer.nativeTextureBackend == "metal"
          && exportLayer.nativeTextureView == 8002,
          "export keeps an independent Metal frame owner without narrowing its view handle");

    FakeLayer missingContextLayer;
    std::vector<FakeReceipt> missingContextOwners;
    error.clear();
    check(prepareVisualImportedAnimationLayer<FakeExecution>(
              { plan }, 7, 640, 360, 0.0, 24.0,
              NativeImportedAnimationRenderUse::Preview,
              nullptr, missingContextLayer, missingContextOwners, error)
              == VisualImportedAnimationPreparation::rejected
          && error == "imported animation compositor product context is unavailable"
          && missingContextOwners.empty(),
          "an imported-animation plan fails closed without its product execution context");

    FakeExecution rejectedExecution;
    rejectedExecution.reject = true;
    FakeLayer rejectedLayer;
    std::vector<FakeReceipt> rejectedOwners;
    error.clear();
    check(prepareVisualImportedAnimationLayer(
              { plan }, 7, 640, 360, 0.0, 24.0,
              NativeImportedAnimationRenderUse::Preview,
              &rejectedExecution, rejectedLayer, rejectedOwners, error)
              == VisualImportedAnimationPreparation::rejected
          && error == "fake native imported-animation rejection"
          && rejectedOwners.empty(),
          "native imported-animation rejection does not publish a borrowed view");

    FakeExecution wrongExtentExecution;
    auto wrongExtentFrame = std::make_shared<FakeNativeFrame>();
    wrongExtentFrame->frameWidth = 320;
    wrongExtentExecution.nextFrame = wrongExtentFrame;
    FakeLayer wrongExtentLayer;
    std::vector<FakeReceipt> wrongExtentOwners;
    error.clear();
    check(prepareVisualImportedAnimationLayer(
              { plan }, 7, 640, 360, 0.0, 24.0,
              NativeImportedAnimationRenderUse::Preview,
              &wrongExtentExecution, wrongExtentLayer, wrongExtentOwners, error)
              == VisualImportedAnimationPreparation::rejected
          && error == "imported animation compositor native frame dimensions are incompatible"
          && wrongExtentOwners.empty(),
          "a native frame with the wrong extent fails before owner publication");

    FakeExecution oversizedOpenGlExecution;
    auto oversizedOpenGlFrame = std::make_shared<FakeNativeFrame>();
    oversizedOpenGlFrame->view =
        static_cast<std::uintptr_t>(std::numeric_limits<unsigned>::max()) + 1u;
    oversizedOpenGlExecution.nextFrame = oversizedOpenGlFrame;
    FakeLayer oversizedOpenGlLayer;
    std::vector<FakeReceipt> oversizedOpenGlOwners;
    error.clear();
    check(prepareVisualImportedAnimationLayer(
              { plan }, 7, 640, 360, 0.0, 24.0,
              NativeImportedAnimationRenderUse::Preview,
              &oversizedOpenGlExecution, oversizedOpenGlLayer,
              oversizedOpenGlOwners, error)
              == VisualImportedAnimationPreparation::rejected
          && error == "imported animation compositor OpenGL texture view exceeds compositor width"
          && oversizedOpenGlOwners.empty(),
          "an OpenGL view that cannot fit the compositor texture type fails closed");

    visualdeformation::RationalFrameTime invalidFrame;
    error.clear();
    check(!videohelper::importedanimation::importedAnimationFrameIdentity(
              0.0, 0.0, invalidFrame, error)
          && error == "imported animation compositor frame clock is invalid",
          "invalid animation frame clocks fail before native execution");

    if (failures != 0)
    {
        std::cerr << "imported animation visual plan execution: "
                  << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "imported animation visual plan execution: PASS\n";
    return 0;
}
