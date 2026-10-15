#pragma once

#include <cstring>
#include <iostream>

#include "../src/imported_animated_scene_payload_execution.h"
#include "../src/imported_scene_payload_execution.h"
#include "../src/imported_scene_visual_plan_execution.h"
#include "../src/renderer.h"
#include "support/native_deformation_composition_fixture.h"
#include "multiobject_animation_fixture.h"

namespace videohelper::fixture3d
{
// Mirrors the private NativeSceneAnimationCompositionFrame::kVisit work unit
// (native_scene_animation_composition.h). This fixture is deliberately not a
// friend of that class, so the value is restated here and must track it.
inline constexpr std::uint64_t kNativeFrameVisitWork = 64;
inline std::string productionCompositionBase64(const std::vector<std::uint8_t>& bytes)
{
    constexpr char alphabet[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve((bytes.size()+2)/3*4);
    for (std::size_t i=0;i<bytes.size();i+=3)
    {
        const auto remaining=bytes.size()-i;
        const auto value=std::uint32_t(bytes[i])<<16
            |(remaining>1?std::uint32_t(bytes[i+1])<<8:0)|(remaining>2?bytes[i+2]:0);
        encoded.push_back(alphabet[(value>>18)&63]);encoded.push_back(alphabet[(value>>12)&63]);
        encoded.push_back(remaining>1?alphabet[(value>>6)&63]:'=');encoded.push_back(remaining>2?alphabet[value&63]:'=');
    }
    return encoded;
}

inline std::vector<std::uint8_t> importedCameraChoiceBytes()
{
    const auto original = multiobjectanimationfixture::bytes(true, false, 1);
    const auto read = [](const auto* p) { return std::uint32_t(p[0]) | std::uint32_t(p[1]) << 8
        | std::uint32_t(p[2]) << 16 | std::uint32_t(p[3]) << 24; };
    const auto jsonBytes = read(original.data()+12);
    auto root = nlohmann::json::parse(original.begin()+20, original.begin()+20+jsonBytes);
    // Original source camera is node 2, its animated ancestor is node 5.
    root["cameras"][0]["name"] = "View";
    root["cameras"].push_back({{"name","View"},{"type","perspective"},
        {"perspective",{{"yfov",.6},{"znear",.05},{"zfar",30.0}}}});
    root["nodes"].push_back({{"name","Second View"},{"camera",1},{"translation",{0.0,3.75,0.0}},{"rotation",{-.70710678,0.0,0.0,.70710678}}});
    root["nodes"][5]["children"].push_back(6);
    root["cameras"].push_back({{"type","perspective"},
        {"perspective",{{"yfov",.9},{"znear",.1},{"zfar",40.0}}}});
    root["nodes"].push_back({{"camera",2},{"translation",{0.0,4.5,0.0}},{"rotation",{-.70710678,0.0,0.0,.70710678}}});
    root["scenes"][0]["nodes"].push_back(7);
    root["nodes"].push_back({{"name","Other Scene View"},{"camera",1},{"translation",{0.0,5.0,0.0}},{"rotation",{-.70710678,0.0,0.0,.70710678}}});
    root["scenes"].push_back({{"name","Other scene"},{"nodes",{0,1,8}}});
    auto json = root.dump(); while (json.size()%4) json += ' ';
    const auto binStart = 28u+jsonBytes;
    std::vector<std::uint8_t> result;
    const auto append = [&](std::uint32_t value) { for (unsigned shift=0;shift<32;shift+=8)
        result.push_back(static_cast<std::uint8_t>(value>>shift)); };
    const auto binBytes = read(original.data()+binStart-8);
    for (auto value : {0x46546c67u,2u,std::uint32_t(28+json.size()+binBytes),std::uint32_t(json.size()),0x4e4f534au}) append(value);
    result.insert(result.end(),json.begin(),json.end()); append(binBytes); append(0x004e4942u);
    result.insert(result.end(),original.begin()+binStart,original.begin()+binStart+binBytes);
    return result;
}

#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
inline std::vector<std::uint8_t> twoLargeTextureGlb()
{
    // Two distinct original 512x512 8-bit palette PNGs decode to separate RGBA
    // images. Their total 524288 texels is below the released 1048576 bound.
    const std::vector<std::uint8_t> black {
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
        0, 0, 2, 0, 0, 0, 2, 0, 8, 3, 0, 0, 0, 195, 166, 36,
        200, 0, 0, 0, 3, 80, 76, 84, 69, 0, 0, 0, 167, 122, 61, 218,
        0, 0, 1, 21, 73, 68, 65, 84, 120, 218, 237, 193, 49, 1, 0, 0,
        0, 194, 160, 245, 79, 237, 105, 9, 160, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 224, 6, 2, 60, 0, 1, 201, 149, 160,
        85, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130
    };
    const std::vector<std::uint8_t> white {
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
        0, 0, 2, 0, 0, 0, 2, 0, 8, 3, 0, 0, 0, 195, 166, 36,
        200, 0, 0, 0, 3, 80, 76, 84, 69, 255, 255, 255, 167, 196, 27, 200,
        0, 0, 1, 21, 73, 68, 65, 84, 120, 218, 237, 193, 49, 1, 0, 0,
        0, 194, 160, 245, 79, 237, 105, 9, 160, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 224, 6, 2, 60, 0, 1, 201, 149, 160,
        85, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130
    };
    const auto original = multiobjectanimationfixture::bytes(true, true);
    const auto read = [](const auto* p) { return std::uint32_t(p[0]) | std::uint32_t(p[1]) << 8
        | std::uint32_t(p[2]) << 16 | std::uint32_t(p[3]) << 24; };
    const auto jsonBytes = read(original.data()+12), binStart = 28u+jsonBytes;
    auto root = nlohmann::json::parse(original.begin()+20, original.begin()+20+jsonBytes);
    std::vector<std::uint8_t> bin(original.begin()+binStart, original.begin()+binStart+read(original.data()+binStart-8));
    root.erase("animations");
    root["nodes"] = nlohmann::json::array({{{"mesh",0}}, {{"camera",0}, {"translation",{0,0,4}}}});
    root["scenes"] = nlohmann::json::array({{{"nodes",{0,1}}}});root["scene"]=0;
    const auto append = [](auto& bytes, std::uint32_t value) {
        for (unsigned shift=0;shift<32;shift+=8) bytes.push_back(static_cast<std::uint8_t>(value>>shift));
    };
    const auto uvView = root["bufferViews"].size(), uvAccessor = root["accessors"].size();
    const auto uvOffset = bin.size();
    for (const float value : {0.0f,0.0f,1.0f,0.0f,.5f,1.0f})
    { std::uint32_t bits;std::memcpy(&bits,&value,sizeof(bits));append(bin,bits); }
    root["bufferViews"].push_back({{"buffer",0}, {"byteOffset",uvOffset}, {"byteLength",24}, {"target",34962}});
    root["accessors"].push_back({{"bufferView",uvView}, {"componentType",5126}, {"count",3}, {"type","VEC2"}});
    for (auto& mesh : root["meshes"]) for (auto& primitive : mesh["primitives"])
        primitive["attributes"]["TEXCOORD_0"] = uvAccessor;
    root["images"] = nlohmann::json::array();root["textures"] = nlohmann::json::array();
    root["samplers"] = nlohmann::json::array({{{"magFilter",9728}, {"minFilter",9728}, {"wrapS",10497}, {"wrapT",10497}}});
    for (const auto* image : {&black,&white})
    {
        while (bin.size()%4) bin.push_back(0);
        const auto view = root["bufferViews"].size(), index = root["images"].size();
        root["bufferViews"].push_back({{"buffer",0}, {"byteOffset",bin.size()}, {"byteLength",image->size()}});
        bin.insert(bin.end(),image->begin(),image->end());
        root["images"].push_back({{"bufferView",view}, {"mimeType","image/png"}});
        root["textures"].push_back({{"source",index}, {"sampler",0}});
    }
    root["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {{"index",0}};
    root["materials"][0]["emissiveTexture"] = {{"index",1}};
    root["materials"][0]["emissiveFactor"] = {1,1,1};
    root["materials"][0]["occlusionTexture"] = {{"index",0}};
    root["buffers"][0]["byteLength"] = bin.size();
    auto json = root.dump();while (json.size()%4) json+=' ';
    while (bin.size()%4) bin.push_back(0);
    std::vector<std::uint8_t> bytes;
    for (const auto value : {0x46546c67u,2u,std::uint32_t(28+json.size()+bin.size()),std::uint32_t(json.size()),0x4e4f534au}) append(bytes,value);
    bytes.insert(bytes.end(),json.begin(),json.end());append(bytes,static_cast<std::uint32_t>(bin.size()));append(bytes,0x004e4942u);
    bytes.insert(bytes.end(),bin.begin(),bin.end());return bytes;
}
#endif

#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
inline bool verifyStaticColdTextureGrowthBoundaries(std::string& error)
{
    using namespace videohelper;
    using namespace gltf;
    using namespace HarmonicMIDI::grid;
    using Budget = modelpayload::NativeSceneAnimationCompositionBudget;
    using ParentProbe = modelpayload::ImportedSceneParentStagingFailureProbe;
    const auto bytes = twoLargeTextureGlb();
    class Backend final : public arbitgpu::NativeFixtureSceneBackend
    {
    public:
        unsigned preparations = 0, submissions = 0;
        arbitgpu::BackendInfo info() const override { return {true,false,"staging","controlled",{}}; }
        arbitgpu::NativeFixtureScenePreparation prepare(const std::shared_ptr<const Visual3DScene>&,
            const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>&,
            arbitgpu::NativeFixturePrepareUse = arbitgpu::NativeFixturePrepareUse::Blocking) override
        { ++preparations; return {}; }
        arbitgpu::NativeFixtureSceneSubmission render(const std::shared_ptr<const Visual3DScene>&,
            const std::shared_ptr<const arbitgpu::NativeFixtureSceneResources>&,
            std::uint32_t, std::uint32_t, arbitgpu::NativeFixtureSceneRuntimeInputs) override
        { ++submissions; return {}; }
    } backend;
    GlbNativeRenderSeam seam(backend);
    bool passed = true;
    std::string failure;
    // Each failing check prints its label, so a later failure cannot hide an earlier one.
    const auto check = [&](bool value, const char* message) {
        if (value) return;
        passed=false;failure=message;
        std::cerr << "static cold check failed: " << message << '\n';
    };
    // A budget object starts with its own sizeof(Budget) logical bytes, so an
    // empty executor budget reads that baseline rather than zero.
    const auto emptyBudgetBytes = Budget().logicalBytes();
    std::cerr << "static-local empty budget object baseline: bytes=" << emptyBudgetBytes << '\n';
    auto budget = std::make_shared<Budget>();
    const auto baseline = budget->logicalBytes();
    GlbNativeSceneAdmission admission;
    bool allowanceCorrect = true;
    GlbTextureGrowthProbeScope growthScope;
    auto& growth = GlbTextureGrowthProbe::state();
    growth.context = &allowanceCorrect;
    growth.observer = [](const GlbTextureGrowthProbe& actual, void* context) {
        const auto& prospective = GlbColdAdmissionProbe::state();
        *static_cast<bool*>(context) &= !actual.overflow && actual.newCapacity <= prospective.textureTexels
            && actual.oldCapacity <= prospective.textureGrowthOverlapTexels
            && actual.maxOverlapTexels <= prospective.textureTexels + prospective.textureGrowthOverlapTexels
            && prospective.adaptations == 1 && prospective.reservedBytes <= Budget::kMaximumBytes;
    };
    GlbColdAdmissionProbe::state() = {};
    const auto admitted = seam.admit(bytes, 0, admission, error, 0, 0, budget);
    const auto coldBytes = budget->logicalBytes()-baseline;
    check(admitted && admission.document && admission.scene && admission.scene->textureCount == 2
        && admission.scene->textureTexelCount == 2*512u*512u
        && admission.scene->textureTexels.capacity() == 2*512u*512u
        && admission.scene->textureTexels.front().red == 0
        && admission.scene->textureTexels.back().red == 255
        && admission.document->images.size() == 2
        && admission.document->images[0].decodedRgba8.size() == 512u*512u*4
        && admission.document->images[1].decodedRgba8.size() == 512u*512u*4
        && allowanceCorrect && growth.growths == 2 && !growth.overflow && growth.oldCapacity == 512u*512u
        && growth.newCapacity == 2*512u*512u && growth.maxOverlapTexels == 3*512u*512u
        && GlbColdAdmissionProbe::state().textureGrowthOverlapTexels == 512u*512u,
        "two distinct used images witness the selected vector's 1 MiB old and 2 MiB new allocation overlap");
    admission = {};
    check(budget->logicalBytes() == baseline && coldBytes && coldBytes < Budget::kMaximumBytes-baseline,
        "cold texture scene/document aliases release the complete prospective closure");
    if (!admitted || !coldBytes || coldBytes >= Budget::kMaximumBytes-baseline) { error=failure;return false; }
    {
        auto exact = Budget::reserve(budget, Budget::kMaximumBytes-baseline-coldBytes, error);
        allowanceCorrect = true;
        growth.oldCapacity = growth.newCapacity = growth.maxOverlapTexels = growth.growths = 0;
        GlbColdAdmissionProbe::state() = {};
        check(exact && seam.admit(bytes, 0, admission, error, 0, 0, budget)
            && budget->logicalBytes() == Budget::kMaximumBytes
            && GlbColdAdmissionProbe::state().reservedBytes == Budget::kMaximumBytes
            && allowanceCorrect && growth.growths == 2 && growth.maxOverlapTexels == 3*512u*512u,
            "exact cold texture allowance stages after prospective old/new peak reservation");
        admission = {};
    }
    // Keep a genuine neutral caller scene while the budgeted candidate refuses.
    growth.observer = nullptr;growth.context = nullptr;
    check(seam.admit(multiobjectanimationfixture::bytes(true,true), 0, admission, error),
        "neutral caller admission remains on the released route");
    const auto callerScene = admission.scene;
    const auto callerSourceBytes = admission.sourceBytes;
    const auto prepared = backend.preparations, submitted = backend.submissions;
    {
        auto oneShort = Budget::reserve(budget, Budget::kMaximumBytes-baseline-coldBytes+1, error);
        const auto liveBytes = budget->logicalBytes(), attemptedWork = budget->logicalWork();
        GlbColdAdmissionProbe::state() = {}; growth.growths = 0;
        check(oneShort && !seam.admit(bytes, 0, admission, error, 0, 0, budget)
            && GlbColdAdmissionProbe::state().decodes == 1 && !GlbColdAdmissionProbe::state().adaptations
            && !GlbColdAdmissionProbe::state().sharedScenes && !GlbColdAdmissionProbe::state().metadataCopies
            && !GlbColdAdmissionProbe::state().lightMaps && !growth.growths
            && admission.scene == callerScene && admission.sourceBytes == callerSourceBytes
            && !admission.document && budget->logicalBytes() == liveBytes
            && budget->logicalWork() > attemptedWork && backend.preparations == prepared && backend.submissions == submitted,
            "one-short two-image peak refuses before adapter growth and conserves caller plus live debit");
    }
    // The public guarded executor also reaches the real two-image cold path.
    // A later full-candidate byte refusal remains valid here; this fixture
    // asserts cold admission and conservation, not an issued/native frame.
    modelpayload::Store store;
    visualanimationimport::ExactContentAssetKey key {"static-cold-two-textures",1,
        modelpayload::detail::digestHex(bytes),"model/gltf-binary",bytes.size()};
    const std::string transfer = "static-cold-two-textures-transfer";
    check(store.begin(transfer,key)
        && store.appendBase64(transfer,0,productionCompositionBase64(bytes)) && store.commit(transfer),
        "two-image exact source enters the payload store");
    modelpayload::ImportedScenePayloadExecution execution(store, backend);
    modelpayload::ImportedSceneRequest request;
    request.asset=key;request.sceneIndex=0;request.sourceStableId=701;request.renderStableId=702;
    request.structuralRevision=request.evaluationRevision=request.projectGeneration=request.helperGeneration=1;
    request.clipId=703;request.dimensions={32,32};request.frame={0,24,1};
    request.objectLocalTransform.version=1;request.objectLocalTransform.nodeStableId=1;
    request.objectLocalTransform.meshStableId=1;request.objectLocalTransform.translation[0]=1;
    modelpayload::ImportedSceneExecutionReceipt receipt;receipt.materialDiagnostic="two-image caller";
    growth.observer = nullptr; GlbColdAdmissionProbe::state() = {};
    const auto publicPreparations = backend.preparations, publicSubmissions = backend.submissions;
    const bool publicReturned = execution.executePreview(request, receipt, error);
    std::cerr << "two-image public preview: returned=" << publicReturned << " error=\"" << error
        << "\" preparation delta=" << (backend.preparations-publicPreparations)
        << " submission delta=" << (backend.submissions-publicSubmissions)
        << " decodes=" << GlbColdAdmissionProbe::state().decodes << '\n';
    check(!publicReturned && GlbColdAdmissionProbe::state().adaptations == 1
        && !error.empty()
        && receipt.materialDiagnostic.empty() && !receipt.admission.scene
        && backend.preparations-publicPreparations <= 1 && backend.submissions == publicSubmissions,
        "two-image optional-retention candidate may adapt, prepares at most once, submits nothing and clears the caller receipt on refusal");
    const auto executorBudget = ParentProbe::staticLocalBudget(execution).lock();
    check(bool(executorBudget), "two-image executor creates its static-local budget on the first call");
    if (!executorBudget) { error=failure;return false; }
    check(executorBudget->logicalBytes() == emptyBudgetBytes,
        "refused two-image preview rolls back its static owner, so only the empty budget object remains");
    std::cerr << "two-image executor budget after the public preview: bytes=" << executorBudget->logicalBytes() << '\n';
    // Each mode gets its own fresh execution. The static-local budget exists only
    // after a call, so the first call also yields the fresh baseline.
    for (const bool exported : {false,true})
    {
        const char* mode = exported ? "export" : "preview";
        auto fresh = std::make_unique<modelpayload::ImportedScenePayloadExecution>(store, backend);
        receipt = {};receipt.materialDiagnostic="one-short caller";
        growth.observer = nullptr; GlbColdAdmissionProbe::state() = {};
        const bool materialRefused = !(exported ? fresh->executeExport(request, receipt, error)
                                                : fresh->executePreview(request, receipt, error));
        const auto freshBudget = ParentProbe::staticLocalBudget(*fresh).lock();
        check(materialRefused && bool(freshBudget),
            "fresh execution creates its static-local budget on the first refused call");
        if (!freshBudget) { error=failure;return false; }
        const auto executorBaseline = freshBudget->logicalBytes();
        check(executorBaseline == emptyBudgetBytes,
            "a fresh refused call leaves the static-local budget at its empty object baseline");
        // What optional retention charges for this scene, measured by admitting
        // it on a fresh budget with the same optional flag the executor uses.
        auto measureBudget = std::make_shared<Budget>();
        GlbNativeSceneAdmission measured;
        const bool measuredAdmitted = seam.admit(bytes, 0, measured, error, 0, 0, measureBudget, true);
        const auto retentionCharge = measureBudget->logicalBytes();
        measured = {};
        measureBudget.reset();
        check(measuredAdmitted && retentionCharge > 0,
            "optional retention charge of the two-image scene is measurable on a fresh budget");
        // Leave one byte less than the retention charge free after the baseline.
        // When the baseline and that charge reach the limit, the room cannot
        // exist, and the unsigned subtraction would wrap. Reserve nothing then.
        const bool boundaryFormed = retentionCharge > 0 && executorBaseline+retentionCharge < Budget::kMaximumBytes;
        std::shared_ptr<const Budget::Charge> oneShort;
        if (boundaryFormed)
            oneShort = Budget::reserve(freshBudget, Budget::kMaximumBytes-executorBaseline-retentionCharge+1, error);
        const auto liveBytes = freshBudget->logicalBytes();
        const bool reserved = boundaryFormed && bool(oneShort);
        GlbColdAdmissionProbe::state() = {}; growth.growths = 0;
        receipt = {};receipt.materialDiagnostic="one-short caller";
        const auto preparedBefore = backend.preparations, submittedBefore = backend.submissions;
        // The call still runs without a boundary, so its probe values print.
        // Closed still requires reserved, so that case fails.
        const bool refused = !(exported ? fresh->executeExport(request, receipt, error)
                                        : fresh->executePreview(request, receipt, error));
        const auto& probe = GlbColdAdmissionProbe::state();
        const bool decodedOnce = probe.decodes == 1;
        const bool prepareAtMostOnce = backend.preparations-preparedBefore <= 1;
        const bool submissionsUnchanged = backend.submissions == submittedBefore;
        const bool budgetConserved = freshBudget->logicalBytes() == liveBytes;
        const bool callerCleared = receipt.materialDiagnostic.empty() && !receipt.admission.scene;
        const bool closed = reserved && refused && decodedOnce && prepareAtMostOnce
            && submissionsUnchanged && budgetConserved && callerCleared;
        std::cerr << "one-short optional-retention two-image closure (" << mode << "): boundaryFormed=" << boundaryFormed
            << " executorBaseline=" << executorBaseline << " measuredRetentionCharge=" << retentionCharge
            << " kMaximumBytes=" << Budget::kMaximumBytes << " reserved=" << reserved << " refused=" << refused
            << " error=\"" << error << "\" decodes=" << probe.decodes << " adaptations=" << probe.adaptations
            << " growths=" << growth.growths << '\n';
        std::cerr << "one-short optional-retention two-image closure (" << mode << "): budget=" << freshBudget->logicalBytes()
            << " live=" << liveBytes << " conserved=" << budgetConserved << " callerCleared=" << callerCleared
            << " preparations=" << (backend.preparations-preparedBefore) << " submissionsUnchanged=" << submissionsUnchanged << '\n';
        check(boundaryFormed, "one-short boundary forms below the budget limit (executorBaseline plus the measured optional-retention charge is below kMaximumBytes)");
        check(closed, "Preview and Export one-short optional-retention candidate may adapt, prepare at most once, submit nothing, conserve the executor budget to its pre-call live bytes and clear the caller receipt");
    }
    if (!passed) error=failure;else error.clear();
    return passed;
}
#endif

#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
// CPU staging checks run beside the production join. They produce no GPU frame
// or Motion evidence and retain no issuing frame after its refused candidate.
inline bool verifyStaticLocalCandidateStaging(std::string& error)
{
    using namespace modelpayload;
    using namespace HarmonicMIDI::grid;
    using Probe = ImportedSceneParentStagingFailureProbe;
    class Backend final : public arbitgpu::NativeFixtureSceneBackend
    {
    public:
        unsigned calls = 0;
        arbitgpu::BackendInfo info() const override { return {true,false,"staging","controlled",{}}; }
        arbitgpu::NativeFixtureScenePreparation prepare(const std::shared_ptr<const Visual3DScene>&,
            const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>&,
            arbitgpu::NativeFixturePrepareUse = arbitgpu::NativeFixturePrepareUse::Blocking) override
        { ++calls; return {}; }
        arbitgpu::NativeFixtureSceneSubmission render(const std::shared_ptr<const Visual3DScene>&,
            const std::shared_ptr<const arbitgpu::NativeFixtureSceneResources>&,
            std::uint32_t, std::uint32_t, arbitgpu::NativeFixtureSceneRuntimeInputs) override
        { ++calls; return {}; }
    } backend;
    const auto original = multiobjectanimationfixture::bytes(true, true);
    const auto read = [](const auto* p) { return std::uint32_t(p[0]) | std::uint32_t(p[1]) << 8
        | std::uint32_t(p[2]) << 16 | std::uint32_t(p[3]) << 24; };
    const auto jsonBytes = read(original.data()+12);
    auto root = nlohmann::json::parse(original.begin()+20, original.begin()+20+jsonBytes);
    root.erase("animations");
    root["nodes"] = nlohmann::json::array({
        {{"children", {1,3,4}}, {"rotation", {0.0,0.0,std::sqrt(.5),std::sqrt(.5)}}},
        {{"mesh",0}}, {{"mesh",0}}, {{"camera",0}, {"translation",{0.0,0.0,2.0}}},
        {{"extensions", {{"KHR_lights_punctual", {{"light",0}}}}}}, {{"mesh",0}}
    });
    root["scenes"] = nlohmann::json::array({{{"nodes", {0,2,5}}}});
    const auto binStart = 28u+jsonBytes;
    const auto binBytes = read(original.data()+binStart-8);
    const auto package = [&](const nlohmann::json& source) {
        auto json=source.dump();while (json.size()%4) json+=' ';
        std::vector<std::uint8_t> result;
        const auto append=[&](std::uint32_t value) { for (unsigned shift=0;shift<32;shift+=8)
            result.push_back(static_cast<std::uint8_t>(value>>shift)); };
        for (auto value : {0x46546c67u,2u,std::uint32_t(28+json.size()+binBytes),std::uint32_t(json.size()),0x4e4f534au}) append(value);
        result.insert(result.end(),json.begin(),json.end());append(binBytes);append(0x004e4942u);
        result.insert(result.end(),original.begin()+binStart,original.begin()+binStart+binBytes);
        return result;
    };
    const auto bytes=package(root);
    Store store;
    visualanimationimport::ExactContentAssetKey key {"local-static-staging",1,
        modelpayload::detail::digestHex(bytes),"model/gltf-binary",bytes.size()};
    const auto encoded = productionCompositionBase64(bytes);
    const std::string transfer = "local-static-staging-transfer";
    if (!store.begin(transfer,key)
        || !store.appendBase64(transfer,0,encoded) || !store.commit(transfer))
    { error = "static local staging exact source transfer refused"; return false; }
    ImportedScenePayloadExecution execution(store,backend);
    ImportedSceneRequest request;
    request.asset=key;request.sceneIndex=0;request.sourceStableId=91;request.renderStableId=92;
    request.structuralRevision=request.evaluationRevision=request.projectGeneration=request.helperGeneration=1;
    request.clipId=93;request.dimensions={32,32};request.frame={0,24,1};request.sourceCameraNodeId=4;
    request.objectLocalTransform.version=1;request.objectLocalTransform.nodeStableId=1;
    request.objectLocalTransform.translation[0]=1;
    request.objectNodeStableId=2;request.objectMeshStableId=1;
    request.runtimeInputs.objectNodeStableId=2;request.runtimeInputs.objectTranslationOffset[0]=2;
    struct Observation final
    {
        std::shared_ptr<NativeSceneAnimationCompositionBudget> budget;
        std::size_t peakBytes = 0;
        bool rigidRotation = false, saturateTransfers = false;
        const ImportedSceneRequest* request = nullptr;
        const ImportedScenePayloadExecution* execution = nullptr;
        ImportedAnimationEvaluationOwner use = ImportedAnimationEvaluationOwner::Preview;
        bool reached = false, correct = false;
        std::shared_ptr<const arbitgpu::NativeImportedObjectWorlds> worlds;
        static Observation*& live() { static thread_local Observation* value=nullptr; return value; }
        // A composed candidate lists the leaf objects under the moved node by raw
        // id, at the world the executor checks without runtime placement. Procedural
        // ids never appear, and the unmoved sibling and untouched objects are absent.
        static void inspectComposed(const std::shared_ptr<const NativeSceneAnimationCompositionFrame>& frame)
        {
            auto& value = *live(); value.reached = true;
            value.worlds = frame->objectWorlds();
            const auto& scene = frame->owner()->scene();
            const auto nodeObject = [](std::size_t node) { return *visualimportedsceneidentity::objectId(node,0); };
            const auto near = [](float a, float b) { return std::abs(a-b)<.0001f; };
            const auto listed = [&value](SceneObjectId object) -> const arbitgpu::NativeImportedObjectWorld* {
                if (value.worlds)
                    for (const auto& entry : *value.worlds)
                        if (entry.object == object) return &entry;
                return nullptr;
            };
            NativeSceneAnimationCompositionFrame::Matrix expected{};
            std::string error;
            const auto* moved = listed(nodeObject(1));
            value.correct = value.worlds && !value.worlds->empty() && moved
                && !listed(nodeObject(2)) && !listed(nodeObject(5))
                && frame->checkedStaticWorld(scene,nodeObject(1),expected,error,false)
                && moved->world==expected
                && near(moved->world[13],float(value.request->objectLocalTransform.translation[0]));
            if (value.worlds)
                for (const auto& entry : *value.worlds)
                    value.correct = value.correct && !(entry.object.value & 0x80000000u);
        }
        static void inspect(const std::shared_ptr<const NativeSceneAnimationCompositionFrame>& frame)
        {
            auto& value = *live(); value.reached = true;
            const auto& request = *value.request;
            const auto& scene = frame->owner()->scene();
            value.peakBytes = value.budget ? value.budget->logicalBytes() : 0;
            if (value.rigidRotation)
            {
                const auto camera = frame->camera();
                value.correct = camera && visualimportedscenerender::detail::validCamera(*camera)
                    && camera->transform.scale.x==1 && camera->transform.scale.y==1 && camera->transform.scale.z==1
                    && !frame->sample() && !frame->owner()->animation()
                    && std::abs(camera->transform.rotation.x)+std::abs(camera->transform.rotation.y) > .01f
                    && frame->lights().size()==1 && visualimportedscenerender::detail::validLight(frame->lights()[0]);
                return;
            }
            NativeSceneAnimationCompositionFrame::Matrix child{}, sibling{}, untouched{}, sentinel{};
            std::string error;
            const auto nodeObject = [](std::size_t node) { return *visualimportedsceneidentity::objectId(node,0); };
            const auto near = [](float a, float b) { return std::abs(a-b)<.0001f; };
            auto runtime = request.runtimeInputs;runtime.localSceneFrame=frame;
            runtime.animatedCamera=frame->camera();runtime.animatedLights=frame->lights();
            // The binding compares the worlds the candidate published, so the frame's own list is passed back.
            runtime.objectWorlds=frame->objectWorlds();
            const bool match = frame->matchesStaticCandidate(frame->owner()->payload(),scene,
                request.sourceStableId,request.renderStableId,request.structuralRevision,request.evaluationRevision,
                request.projectGeneration,request.helperGeneration,request.clipId,value.use,request.frame,
                request.dimensions.width,request.dimensions.height,request.objectLocalTransform,runtime,request.sourceCameraNodeId);
            // One moved world is a different list, so the same candidate must not bind to it.
            const bool worldsStaged=frame->objectWorlds() && !frame->objectWorlds()->empty();
            bool worldsMoved=false;
            if (worldsStaged)
            {
                auto movedWorlds=std::make_shared<arbitgpu::NativeImportedObjectWorlds>(*frame->objectWorlds());
                (*movedWorlds)[0].world[12]+=1.0f;
                runtime.objectWorlds=movedWorlds;
                worldsMoved=!frame->matchesStaticCandidate(frame->owner()->payload(),scene,
                    request.sourceStableId,request.renderStableId,request.structuralRevision,request.evaluationRevision,
                    request.projectGeneration,request.helperGeneration,request.clipId,value.use,request.frame,
                    request.dimensions.width,request.dimensions.height,request.objectLocalTransform,runtime,request.sourceCameraNodeId);
                runtime.objectWorlds=frame->objectWorlds();
            }
            ++runtime.objectNodeStableId;
            const bool forged = frame->matchesStaticCandidate(frame->owner()->payload(),scene,
                request.sourceStableId,request.renderStableId,request.structuralRevision,request.evaluationRevision,
                request.projectGeneration,request.helperGeneration,request.clipId,value.use,request.frame,
                request.dimensions.width,request.dimensions.height,request.objectLocalTransform,runtime,request.sourceCameraNodeId);
            value.correct = match && worldsStaged && worldsMoved && !forged && !frame->sample() && !frame->owner()->animation()
                && frame->owner()->mode()==NativeSceneAnimationCompositionOwner::Mode::StaticLocalHierarchy
                && frame->owner()->document()->nodes.size()==6
                && frame->owner()->document()->nodes[0].localTransform[12]==0
                && frame->owner()->document()->nodes[3].worldTransform[13]==0
                && frame->checkedStaticWorld(scene,nodeObject(1),child,error)
                && frame->checkedStaticWorld(scene,nodeObject(2),sibling,error)
                && frame->checkedStaticWorld(scene,nodeObject(5),untouched,error)
                && near(child[12],request.objectNodeStableId==2?2:0) && near(child[13],float(request.objectLocalTransform.translation[0]))
                && near(sibling[12],request.objectNodeStableId==3?2:0) && near(sibling[13],0)
                && near(untouched[12],0) && near(untouched[13],0)
                && frame->camera() && near(frame->camera()->transform.translation.y,float(request.objectLocalTransform.translation[0]))
                && frame->lights().size()==1 && near(frame->lights()[0].transform.translation.y,float(request.objectLocalTransform.translation[0]))
                && !frame->checkedStaticWorld({},nodeObject(1),sentinel,error)
                && !frame->checkedStaticWorld(scene,SceneObjectId{0},sentinel,error);
            // Use the actual issued candidate to prepare and copy the existing
            // public receipt value. These checks grant no native completion.
            auto callerPool=Probe::staticLocalCallerBudget(*value.execution).lock();
            if (!callerPool) { value.correct=false; return; }
            const auto baseline=callerPool->logicalBytes();
            {
                gltf::GlbNativeFrameReceipt prepared;
                if (!prepared.prepareLocal(frame,*scene,"opengl",error)) {value.correct=false;return;}
                gltf::GlbNativeDrawReceipt draw;draw.object=scene->objects[0].id;
                prepared.draws.push_back(draw);prepared.nativeBackend="first";
                const auto preparedBytes=callerPool->logicalBytes()-baseline;
                auto copy=prepared;copy.nativeBackend="second";copy.draws[0].indexCount=123;
                const auto copyBytes=callerPool->logicalBytes()-baseline-preparedBytes;
                value.correct=value.correct && prepared.localFrame()==frame && copy.localFrame()==frame
                    && copy.draws.size()==1 && copy.draws[0].object==draw.object && copy.draws[0].indexCount==123
                    && copy.draws.capacity()<=prepared.draws.capacity() && callerPool->logicalBytes()>baseline;
                auto full=NativeCompositionRetentionBudget::reserve(callerPool,
                    NativeCompositionRetentionBudget::kMaximumBytes-callerPool->logicalBytes(),error);
                gltf::GlbNativeFrameReceipt destination;
                destination.nativeBackend="preserved public destination";
                bool refused=false;
                try {destination=prepared;} catch (const std::bad_alloc&) {refused=true;}
                value.correct=value.correct && full && refused
                    && destination.nativeBackend=="preserved public destination" && !destination.localFrame();
                value.correct=value.correct && !destination.prepareLocal(frame,*scene,"opengl",error)
                    && destination.nativeBackend=="preserved public destination" && !destination.localFrame();
                full.reset();
                const auto workBudget=Probe::staticLocalBudget(*value.execution).lock();
                if (!workBudget) {value.correct=false;return;}
                auto attempted=workBudget->logicalWork();
                const auto liveBytes=callerPool->logicalBytes();
                gltf::GlbNativeFrameReceipt moved(std::move(copy));
                value.correct=value.correct && workBudget->logicalWork()>attempted
                    && moved.localFrame()==frame && moved.nativeBackend=="second" && moved.draws[0].indexCount==123
                    && !copy.localFrame() && copy.draws.empty() && copy.nativeBackend.empty()
                    && callerPool->logicalBytes()==liveBytes;
                attempted=workBudget->logicalWork();destination=std::move(moved);
                value.correct=value.correct && workBudget->logicalWork()>attempted
                    && destination.localFrame()==frame && destination.nativeBackend=="second"
                    && !moved.localFrame() && moved.draws.empty() && moved.nativeBackend.empty()
                    && callerPool->logicalBytes()==liveBytes;
                for (unsigned i=0;i<3;++i)
                {
                    attempted=workBudget->logicalWork();destination.swap(copy);
                    const auto& active=i%2?destination:copy;
                    const auto& inactive=i%2?copy:destination;
                    value.correct=value.correct && workBudget->logicalWork()>attempted
                        && active.localFrame()==frame && active.nativeBackend=="second" && active.draws[0].indexCount==123
                        && !inactive.localFrame() && inactive.draws.empty() && inactive.nativeBackend.empty()
                        && callerPool->logicalBytes()==liveBytes;
                }
                attempted=workBudget->logicalWork();prepared.swap(copy);
                value.correct=value.correct && workBudget->logicalWork()>attempted
                    && prepared.localFrame()==frame && copy.localFrame()==frame
                    && prepared.nativeBackend=="second" && prepared.draws[0].indexCount==123
                    && copy.nativeBackend=="first" && copy.draws[0].indexCount==0
                    && callerPool->logicalBytes()==liveBytes;
                attempted=workBudget->logicalWork();prepared=gltf::GlbNativeFrameReceipt{};
                value.correct=value.correct && workBudget->logicalWork()>attempted && !prepared.localFrame()
                    && callerPool->logicalBytes()==liveBytes-copyBytes && copy.localFrame()==frame;
                gltf::GlbNativeFrameReceipt assigned(copy);assigned.nativeBackend="third";assigned.draws[0].indexCount=789;
                const auto beforeAssignmentBytes=callerPool->logicalBytes();
                attempted=workBudget->logicalWork();copy=std::move(assigned);
                value.correct=value.correct && workBudget->logicalWork()>attempted
                    && copy.localFrame()==frame && copy.nativeBackend=="third" && copy.draws[0].indexCount==789
                    && !assigned.localFrame() && assigned.draws.empty() && assigned.nativeBackend.empty()
                    && callerPool->logicalBytes()==beforeAssignmentBytes-preparedBytes;
                attempted=workBudget->logicalWork();const auto selfBytes=callerPool->logicalBytes();
                auto& self=copy;copy=self;copy=std::move(self);copy.swap(self);
                value.correct=value.correct && workBudget->logicalWork()==attempted && callerPool->logicalBytes()==selfBytes
                    && copy.localFrame()==frame && copy.nativeBackend=="third" && copy.draws[0].indexCount==789;
                if (value.saturateTransfers)
                {
                    gltf::GlbNativeFrameReceipt second(copy);second.nativeBackend="fourth";
                    const auto secondBytes=callerPool->logicalBytes()-selfBytes;
                    const auto saturationBytes=callerPool->logicalBytes();
                    value.correct=value.correct && workBudget->consumeWork(
                        NativeSceneAnimationCompositionBudget::kMaximumWork-workBudget->logicalWork()-1,1);
                    attempted=workBudget->logicalWork();
                    gltf::GlbNativeFrameReceipt saturatedMove(std::move(copy));
                    value.correct=value.correct && workBudget->logicalWork()>attempted
                        && workBudget->logicalWork()==NativeSceneAnimationCompositionBudget::kMaximumWork
                        && saturatedMove.localFrame()==frame && saturatedMove.nativeBackend=="third" && !copy.localFrame();
                    destination=std::move(saturatedMove);destination.swap(copy);copy.swap(second);
                    value.correct=value.correct && workBudget->logicalWork()==NativeSceneAnimationCompositionBudget::kMaximumWork
                        && !destination.localFrame() && !saturatedMove.localFrame()
                        && copy.localFrame()==frame && copy.nativeBackend=="fourth"
                        && second.localFrame()==frame && second.nativeBackend=="third"
                        && callerPool->logicalBytes()==saturationBytes;
                    second=std::move(copy);
                    value.correct=value.correct && workBudget->logicalWork()==NativeSceneAnimationCompositionBudget::kMaximumWork
                        && !copy.localFrame() && second.localFrame()==frame && second.nativeBackend=="fourth"
                        && callerPool->logicalBytes()==baseline+secondBytes;
                    auto& saturatedSelf=second;second=saturatedSelf;second=std::move(saturatedSelf);second.swap(saturatedSelf);
                    value.correct=value.correct && workBudget->logicalWork()==NativeSceneAnimationCompositionBudget::kMaximumWork
                        && callerPool->logicalBytes()==baseline+secondBytes && second.localFrame()==frame
                        && second.nativeBackend=="fourth" && second.draws[0].indexCount==789;
                }
                auto guarded=runtime;guarded.objectNodeStableId=request.runtimeInputs.objectNodeStableId;
                guarded.glassCaustic=visualimportedscenerender::GlassCausticSelection{};
                value.correct=value.correct && !arbitgpu::validFixtureRuntimeInputs(guarded);
                guarded.previousMotion=arbitgpu::sceneMotionSample(guarded);
                guarded.localSceneFrame.reset();
                value.correct=value.correct && guarded.previousMotion->objectNodeStableId==request.runtimeInputs.objectNodeStableId
                    && !arbitgpu::validFixtureRuntimeInputs(guarded);
            }
            value.correct=value.correct && callerPool->logicalBytes()==baseline;
        }
    } observation;
    Observation::live()=&observation;observation.execution=&execution;
    ImportedSceneExecutionReceipt output;output.materialDiagnostic="preserved sentinel";
    {
        // Even a released v0 admission whose native preparation later refuses
        // must keep its first document; local activation cannot decode again.
        Backend legacyBackend;
        ImportedScenePayloadExecution legacyExecution(store,legacyBackend);
        auto neutral=request;neutral.objectLocalTransform={};
        ImportedSceneExecutionReceipt legacyOutput;legacyOutput.materialDiagnostic="legacy caller";
        gltf::GlbColdAdmissionProbe::state()={};
        if (legacyExecution.executePreview(neutral,legacyOutput,error) || !legacyBackend.calls
            || gltf::GlbColdAdmissionProbe::state().decodes!=1)
        {Observation::live()=nullptr;error="released static pending-owner setup failed";return false;}
        const auto calls=legacyBackend.calls;
        const auto decodes=gltf::GlbColdAdmissionProbe::state().decodes;
        observation.execution=&legacyExecution;observation.request=&request;
        observation.reached=observation.correct=false;
        Probe::Scope scope(Probe::Site::None);Probe::state().localObserver=&Observation::inspect;
        if (legacyExecution.executePreview(request,legacyOutput,error) || !observation.reached || !observation.correct
            || legacyBackend.calls!=calls+1 || gltf::GlbColdAdmissionProbe::state().decodes!=decodes
            || !Probe::state().candidateLocalFrame.expired() || legacyOutput.materialDiagnostic!="legacy caller")
        {Observation::live()=nullptr;error="static v0-to-local candidate decoded again or changed its pending owner/caller";return false;}
        observation.execution=&execution;
    }
    // A mesh-valid node outside scene zero must scan only the retained source
    // draws. Both uses borrow the v0 owner and never re-enter native work.
    {
        auto outside=root;outside["nodes"].push_back({{"mesh",0}});
        const auto outsideBytes=package(outside);
        visualanimationimport::ExactContentAssetKey outsideKey {"local-static-outside-overlay",1,
            modelpayload::detail::digestHex(outsideBytes),"model/gltf-binary",outsideBytes.size()};
        const std::string transfer="local-static-outside-overlay-transfer";
        if (!store.begin(transfer,outsideKey)
            || !store.appendBase64(transfer,0,productionCompositionBase64(outsideBytes)) || !store.commit(transfer))
        {Observation::live()=nullptr;error="outside-scene overlay source transfer failed";return false;}
        struct WarmSource final
        {
            // The document aliases original ColdRetention. Candidate scene
            // and payload aliases have their own charge owners, so keep only
            // their pointee identities and never dereference these pointers.
            std::weak_ptr<const gltf::GlbStaticMeshDocument> document;
            const Visual3DScene* scene=nullptr;
            const visualmodelassetpayload::ImmutablePayload* payload=nullptr;
            std::size_t draws=0;
            bool reached=false, same=true;
            static WarmSource*& live() {static thread_local WarmSource* value=nullptr;return value;}
            static void inspect(const std::shared_ptr<const NativeSceneAnimationCompositionFrame>& frame)
            {
                auto& value=*live();const auto& owner=frame->owner();
                if (value.reached)
                {
                    const auto document=value.document.lock();
                    value.same=value.same && document && document==owner->document()
                        && value.scene==owner->importedScene().get() && value.payload==owner->payload().get();
                }
                else
                {
                    value.document=owner->document();value.scene=owner->importedScene().get();value.payload=owner->payload().get();
                    value.draws=owner->importedScene()->objectCount;value.reached=true;
                }
                value.same=value.same && owner->document()->nodes.size()==7 && owner->document()->nodes[6].mesh==0;
            }
        } warm;
        WarmSource::live()=&warm;
        for (const bool exported : {false,true})
        {
            Backend warmBackend;ImportedScenePayloadExecution warmExecution(store,warmBackend);
            auto inside=request;inside.asset=outsideKey;
            auto neutral=inside;neutral.objectLocalTransform={};
            ImportedSceneExecutionReceipt caller;caller.materialDiagnostic="exact warm caller";caller.frame.nativeBackend="caller backend";
            gltf::GlbColdAdmissionProbe::state()={};
            if (warmExecution.executePreview(neutral,caller,error) || warmBackend.calls!=1
                || gltf::GlbColdAdmissionProbe::state().decodes!=1)
            {WarmSource::live()=nullptr;Observation::live()=nullptr;error="warm outside-overlay v0 setup failed";return false;}
            // A staged candidate reaches the backend exactly once and that backend refuses it.
            // Pre-staging refusals leave the count where the previous attempt left it.
            auto calls=warmBackend.calls;
            const auto budget=Probe::staticLocalBudget(warmExecution).lock();
            if (!budget) {WarmSource::live()=nullptr;Observation::live()=nullptr;error="warm v0 issuing budget is absent";return false;}
            const auto bytes=budget->logicalBytes();
            std::array<unsigned char,sizeof(caller)> callerBytes{};std::memcpy(callerBytes.data(),&caller,sizeof(caller));
            const auto conserved=[&] {
                const auto document=warm.document.lock();
                return document && !std::memcmp(callerBytes.data(),&caller,sizeof(caller))
                    && caller.materialDiagnostic=="exact warm caller" && caller.frame.nativeBackend=="caller backend"
                    && warmBackend.calls==calls && !gltf::GlbColdAdmissionProbe::state().decodes
                    && !gltf::GlbColdAdmissionProbe::state().adaptations
                    && Probe::state().candidateLocalFrame.expired() && budget->logicalBytes()==bytes;
            };
            warm={};gltf::GlbColdAdmissionProbe::state()={};
            {
                Probe::Scope scope(Probe::Site::None);Probe::state().localObserver=&WarmSource::inspect;
                const bool issued=exported?warmExecution.executeExport(inside,caller,error):warmExecution.executePreview(inside,caller,error);
                const bool oneBackendAttempt=warmBackend.calls==calls+1;
                calls=warmBackend.calls;
                if (issued || !oneBackendAttempt || !warm.reached || !warm.same
                    || !Probe::state().candidateLocalFrame.expired() || !conserved())
                {WarmSource::live()=nullptr;Observation::live()=nullptr;error="warm exact first document was not borrowed";return false;}
            }
            auto rejected=inside;rejected.objectNodeStableId=7;rejected.runtimeInputs.objectNodeStableId=7;
            auto wrongMesh=rejected;wrongMesh.objectMeshStableId=2;
            gltf::GlbColdAdmissionProbe::state()={};const auto beforeHeader=budget->logicalWork();
            if ((exported?warmExecution.executeExport(wrongMesh,caller,error):warmExecution.executePreview(wrongMesh,caller,error))
                || error.find("old overlay target")==std::string::npos || gltf::GlbColdAdmissionProbe::state().oldOverlaySourceDrawVisits
                || budget->logicalBytes()!=bytes || !conserved())
            {WarmSource::live()=nullptr;Observation::live()=nullptr;error="warm wrong-mesh refusal reached source draws";return false;}
            const auto headerWork=budget->logicalWork()-beforeHeader;
            gltf::GlbColdAdmissionProbe::state()={};const auto before=budget->logicalWork();
            if ((exported?warmExecution.executeExport(rejected,caller,error):warmExecution.executePreview(rejected,caller,error))
                || error.find("old overlay target")==std::string::npos
                || gltf::GlbColdAdmissionProbe::state().oldOverlaySourceDrawVisits!=warm.draws
                || budget->logicalBytes()!=bytes || !conserved())
            {WarmSource::live()=nullptr;Observation::live()=nullptr;error="warm mesh-valid outside-scene overlay refusal changed source/caller";return false;}
            const auto refusalWork=budget->logicalWork()-before;
            gltf::GlbColdAdmissionProbe::state()={};
            {
                Probe::Scope scope(Probe::Site::None);Probe::state().localObserver=&WarmSource::inspect;
                const bool issued=exported?warmExecution.executeExport(inside,caller,error):warmExecution.executePreview(inside,caller,error);
                const bool oneBackendAttempt=warmBackend.calls==calls+1;
                calls=warmBackend.calls;
                if (issued || !oneBackendAttempt || !warm.reached || !warm.same || !Probe::state().candidateLocalFrame.expired()
                    || !conserved() || budget->logicalBytes()!=bytes)
                {WarmSource::live()=nullptr;Observation::live()=nullptr;error="outside-scene refusal replaced the v0 source document/scene";return false;}
            }
            if (refusalWork!=headerWork+warm.draws*kNativeFrameVisitWork
                || !budget->consumeWork(NativeSceneAnimationCompositionBudget::kMaximumWork-budget->logicalWork()-refusalWork+1,1))
            {WarmSource::live()=nullptr;Observation::live()=nullptr;error="warm one-short source-draw work setup failed";return false;}
            gltf::GlbColdAdmissionProbe::state()={};
            if ((exported?warmExecution.executeExport(rejected,caller,error):warmExecution.executePreview(rejected,caller,error))
                || error.find("named work")==std::string::npos || gltf::GlbColdAdmissionProbe::state().oldOverlaySourceDrawVisits
                || budget->logicalWork()!=NativeSceneAnimationCompositionBudget::kMaximumWork
                || budget->logicalBytes()!=bytes || !conserved())
            {WarmSource::live()=nullptr;Observation::live()=nullptr;error="one-short warm overlay work reached the source draw scan";return false;}
        }
        WarmSource::live()=nullptr;
    }
    // A staged candidate reaches the backend exactly once, and the fake backend refuses it there.
    for (const bool exported : {false,true})
        for (const auto overlay : {2u,3u})
        for (const auto translation : {1.0,2.0})
        {
            request.objectLocalTransform.translation[0]=translation;
            request.objectNodeStableId=overlay;request.runtimeInputs.objectNodeStableId=overlay;
            observation.request=&request;observation.reached=observation.correct=false;
            observation.use=exported?ImportedAnimationEvaluationOwner::Export:ImportedAnimationEvaluationOwner::Preview;
            const auto callsBefore=backend.calls;
            Probe::Scope scope(Probe::Site::None);Probe::state().localObserver=&Observation::inspect;
            if ((exported?execution.executeExport(request,output,error):execution.executePreview(request,output,error))
                || !observation.reached || !observation.correct || backend.calls!=callsBefore+1
                || !Probe::state().candidateLocalFrame.expired() || output.materialDiagnostic!="preserved sentinel")
            { Observation::live()=nullptr;error="static local genuine staging or backend refusal conservation failed";return false; }
        }
    // Exact authored local scale refuses even inside the old 0.0001 near
    // tolerance. A genuine nontrivial rotation still reaches the observer.
    request.objectLocalTransform.scale={1.00005,1,1};
    observation.reached=false;
    {
        const auto callsBefore=backend.calls;
        Probe::Scope scope(Probe::Site::None);Probe::state().localObserver=&Observation::inspect;
        if (execution.executePreview(request,output,error) || observation.reached
            || error.find("not representable")==std::string::npos || backend.calls!=callsBefore
            || output.materialDiagnostic!="preserved sentinel")
        {Observation::live()=nullptr;error="small authored camera scale was issued or coerced";return false;}
    }
    request.objectLocalTransform.scale={1,1,1};request.objectLocalTransform.rotationDegrees={23,31,17};
    observation.rigidRotation=true;observation.reached=observation.correct=false;
    {
        const auto callsBefore=backend.calls;
        Probe::Scope scope(Probe::Site::None);Probe::state().localObserver=&Observation::inspect;
        if (execution.executePreview(request,output,error) || !observation.reached || !observation.correct
            || !Probe::state().candidateLocalFrame.expired() || backend.calls!=callsBefore+1
            || output.materialDiagnostic!="preserved sentinel")
        {Observation::live()=nullptr;error="valid nontrivial unit-scale camera rotation did not reach the backend once";return false;}
    }
    observation.rigidRotation=false;request.objectLocalTransform.rotationDegrees={0,0,0};
    const auto publishSource=[&](const nlohmann::json& source,const std::string& id) {
        const auto sourceBytes=package(source);
        visualanimationimport::ExactContentAssetKey sourceKey {id,1,modelpayload::detail::digestHex(sourceBytes),"model/gltf-binary",sourceBytes.size()};
        const auto transfer=id+"-transfer";
        if (!store.begin(transfer,sourceKey)
            || !store.appendBase64(transfer,0,productionCompositionBase64(sourceBytes))
            || !store.commit(transfer)) return false;
        request.asset=sourceKey;return true;
    };
    for (const bool camera : {true,false})
    {
        auto bounded=root;bounded["nodes"][camera?3:4]["translation"]={0,0,999999};
        if (!publishSource(bounded,camera?"local-static-observer-camera-bound":"local-static-observer-light-bound"))
        {Observation::live()=nullptr;error="component observer source transfer failed";return false;}
        request.objectLocalTransform.translation={0,0,10000};observation.reached=false;
        const auto callsBefore=backend.calls;
        Probe::Scope scope(Probe::Site::None);Probe::state().localObserver=&Observation::inspect;
        if (execution.executePreview(request,output,error) || observation.reached
            || !Probe::state().candidateLocalFrame.expired()
            || error.find(camera?"final camera record":"final light record")==std::string::npos
            || backend.calls!=callsBefore || output.materialDiagnostic!="preserved sentinel")
        {Observation::live()=nullptr;error="out-of-bound final camera/light reached the CPU frame observer";return false;}
    }
    auto scaleBound=root;scaleBound["nodes"][4]["scale"]={1001,1001,1001};
    scaleBound["nodes"][4]["children"]={5};scaleBound["scenes"][0]["nodes"]={0,2};
    if (!publishSource(scaleBound,"local-static-observer-light-scale-bound"))
    {Observation::live()=nullptr;error="light scale observer source transfer failed";return false;}
    request.objectLocalTransform.nodeStableId=5;request.objectLocalTransform.translation={0,0,0};
    request.objectLocalTransform.scale={1000,1000,1000};observation.reached=false;
    {
        const auto callsBefore=backend.calls;
        Probe::Scope scope(Probe::Site::None);Probe::state().localObserver=&Observation::inspect;
        if (execution.executePreview(request,output,error) || observation.reached
            || !Probe::state().candidateLocalFrame.expired() || error.find("final light record")==std::string::npos
            || backend.calls!=callsBefore || output.materialDiagnostic!="preserved sentinel")
        {Observation::live()=nullptr;error="out-of-bound final light scale reached the CPU frame observer";return false;}
    }
    request.asset=key;request.objectLocalTransform.nodeStableId=1;
    request.objectLocalTransform.translation={2,0,0};request.objectLocalTransform.scale={1,1,1};
    Observation::live()=nullptr;
    // Allocation probes sit inside stageStaticLocalFrame, so none reaches the backend.
    const auto allocationCalls=backend.calls;
    for (const auto site : {Probe::Site::StaticLocalOwner,Probe::Site::StaticLocalFrame,Probe::Site::StaticLocalSource})
    {
        Probe::Scope scope(site);
        if (execution.executePreview(request,output,error) || !Probe::state().hit || backend.calls!=allocationCalls
            || output.materialDiagnostic!="preserved sentinel")
        {error="static local allocation refusal changed a caller or reached the backend";return false;}
    }
    const auto budget=Probe::staticLocalBudget(execution).lock();
    if (!budget) {error="static local candidate lost its attempted-work ledger";return false;}
    // The fixed cold lower bound can fit while the decoded closure is one
    // byte short. Decoding remains the old bounded temporary; adaptation and
    // every newly retained scene/metadata/map allocation must stay untouched.
    {
        auto coldBudget=std::make_shared<NativeSceneAnimationCompositionBudget>();
        gltf::GlbNativeRenderSeam seam(backend);gltf::GlbNativeSceneAdmission admission;
        const auto baseline=coldBudget->logicalBytes();
        if (!seam.admit(bytes,0,admission,error,0,0,coldBudget))
        {error="cold closure boundary measurement refused";return false;}
        const auto coldBytes=coldBudget->logicalBytes()-baseline;admission={};
        auto oneShort=NativeSceneAnimationCompositionBudget::reserve(coldBudget,
            NativeSceneAnimationCompositionBudget::kMaximumBytes-baseline-coldBytes+1,error);
        gltf::GlbColdAdmissionProbe::state()={};
        const auto& witness=gltf::GlbColdAdmissionProbe::state();
        if (!oneShort || seam.admit(bytes,0,admission,error,0,0,coldBudget) || admission.valid()
            || witness.decodes!=1 || witness.adaptations || witness.sharedScenes || witness.metadataCopies || witness.lightMaps)
        {error="cold decoded closure refused after new scene/metadata/map allocation";return false;}
        if (!coldBudget->consumeWork(NativeSceneAnimationCompositionBudget::kMaximumWork-coldBudget->logicalWork()-1,1))
        {error="cold work boundary setup failed";return false;}
        gltf::GlbColdAdmissionProbe::state()={};
        if (seam.admit(bytes,0,admission,error,0,0,coldBudget)
            || coldBudget->logicalWork()!=NativeSceneAnimationCompositionBudget::kMaximumWork
            || error.find("named work")==std::string::npos || witness.decodes || witness.adaptations
            || witness.sharedScenes || witness.metadataCopies || witness.lightMaps)
        {error="cold named-work exhaustion reached decoder/adaptation/allocation";return false;}
    }
    observation.budget=budget;
    // A real composed procedural environment light owns long ID/SHA strings.
    // Compare short and long source capacities to prove all three allocations
    // are in the same reservation, including the frame's previously missing copy.
    auto procedural=std::make_shared<Visual3DScene>();
    procedural->id={101};procedural->cameraCount=1;procedural->cameras[0].id={101};procedural->activeCamera={101};
    procedural->lightCount=1;auto& environment=procedural->lights[0];
    environment.id={101};environment.kind=SceneLightKind::Environment;
    environment.environmentImage.version=1;environment.environmentImage.asset.version=1;
    environment.environmentImage.asset.id=std::string(visualmodelasset::kMaximumStableAssetIdBytes,'e');
    environment.environmentImage.asset.contentSha256=std::string(64,'a');
    auto shortProcedural=std::make_shared<Visual3DScene>(*procedural);
    std::string shortId="env";shortProcedural->lights[0].environmentImage.asset.id.swap(shortId);
    if (!validateVisual3DScene(*procedural).valid() || !validateVisual3DScene(*shortProcedural).valid())
    {error="composed long-identity source is invalid";return false;}
    auto noImportedLights=root;noImportedLights["nodes"][4].erase("extensions");
    if (!publishSource(noImportedLights,"local-static-composed-no-imported-lights"))
    {error="composed no-imported-light source transfer failed";return false;}
    auto leaf=std::make_shared<visualimportedscenerender::Request>();
    leaf->asset=request.asset;leaf->sourceStableId=request.sourceStableId;leaf->renderStableId=request.renderStableId;
    leaf->sceneIndex=request.sceneIndex;leaf->sourceCameraNodeId=request.sourceCameraNodeId;
    leaf->objectNodeStableId=request.objectNodeStableId;leaf->objectMeshStableId=request.objectMeshStableId;
    leaf->objectLocalTransform=request.objectLocalTransform;
    leaf->structuralRevision=request.structuralRevision;leaf->evaluationRevision=request.evaluationRevision;
    request.importedScene=leaf;
    observation.request=&request;Observation::live()=&observation;
    // Scene Compose stages one imported leaf beside its procedural scene, whatever
    // the identity length of the composed scene. A separate execution keeps the
    // owner state of the plain leaf checks above.
    ImportedScenePayloadExecution composedExecution(store,backend);
    const auto stagesComposed=[&](const auto& snapshot) {
        request.sceneSnapshot=snapshot;observation.reached=observation.correct=false;observation.worlds.reset();
        const auto callsBefore=backend.calls;
        ImportedSceneExecutionReceipt composedOutput;
        Probe::Scope scope(Probe::Site::None);Probe::state().localObserver=&Observation::inspectComposed;
        return composedExecution.executePreview(request,composedOutput,error) && observation.reached
            && observation.correct && backend.calls>callsBefore;
    };
    if (!stagesComposed(shortProcedural)) {Observation::live()=nullptr;error="short composed static leaf did not stage its checked worlds";return false;}
    const auto shortWorlds=observation.worlds;
    if (!stagesComposed(procedural)) {Observation::live()=nullptr;error="long composed static leaf did not stage its checked worlds";return false;}
    const auto longWorlds=observation.worlds;
    // A changed leaf translation stages new worlds, and the procedural side changes neither.
    leaf->objectLocalTransform.translation[0]=2;request.objectLocalTransform.translation[0]=2;
    const bool movedStaged=stagesComposed(procedural);
    const bool changed=movedStaged && !arbitgpu::sameNativeImportedObjectWorlds(longWorlds,observation.worlds);
    leaf->objectLocalTransform.translation[0]=1;request.objectLocalTransform.translation[0]=1;
    if (!changed || !arbitgpu::sameNativeImportedObjectWorlds(shortWorlds,longWorlds))
    {Observation::live()=nullptr;error="a changed leaf translation did not change the staged worlds";return false;}
    // A second Scene Compose leaf stays refused before any candidate exists.
    ImportedSceneLeafRequest extraLeaf;extraLeaf.scene=leaf;
    request.additionalImportedScenes.push_back(extraLeaf);
    {
        observation.reached=observation.correct=false;observation.worlds.reset();
        const auto callsBefore=backend.calls;
        ImportedSceneExecutionReceipt refusedOutput;refusedOutput.materialDiagnostic="preserved sentinel";
        Probe::Scope scope(Probe::Site::None);Probe::state().localObserver=&Observation::inspectComposed;
        const bool refused=!composedExecution.executePreview(request,refusedOutput,error) && !observation.reached
            && error.find("several Scene Compose leaves are not rendered yet")!=std::string::npos
            && backend.calls==callsBefore && refusedOutput.materialDiagnostic=="preserved sentinel";
        request.additionalImportedScenes.clear();
        if (!refused) {Observation::live()=nullptr;error="several composed static leaves were not refused before staging";return false;}
    }
    // A Scene Compose whose moved node draws no object is refused before a candidate exists.
    {
        const auto keptNode=request.objectLocalTransform.nodeStableId;
        request.objectLocalTransform.nodeStableId=4;leaf->objectLocalTransform.nodeStableId=4;
        observation.reached=observation.correct=false;observation.worlds.reset();
        ImportedSceneExecutionReceipt drawlessOutput;
        Probe::Scope scope(Probe::Site::None);Probe::state().localObserver=&Observation::inspectComposed;
        const bool refused=!composedExecution.executePreview(request,drawlessOutput,error) && !observation.reached;
        request.objectLocalTransform.nodeStableId=keptNode;leaf->objectLocalTransform.nodeStableId=keptNode;
        if (!refused) {Observation::live()=nullptr;error="a Scene Compose that moves no drawn object was not refused before staging";return false;}
    }
    request.importedScene.reset();request.sceneSnapshot.reset();
    {
        ImportedScenePayloadExecution saturatedExecution(store,backend);
        auto saturatedRequest=request;saturatedRequest.asset=key;
        saturatedRequest.objectLocalTransform.translation={1,0,0};saturatedRequest.objectNodeStableId=2;
        saturatedRequest.runtimeInputs.objectNodeStableId=2;
        observation.execution=&saturatedExecution;observation.request=&saturatedRequest;observation.saturateTransfers=true;
        observation.use=ImportedAnimationEvaluationOwner::Preview;observation.reached=observation.correct=false;
        const auto callsBefore=backend.calls;
        Probe::Scope scope(Probe::Site::None);Probe::state().localObserver=&Observation::inspect;
        if (saturatedExecution.executePreview(saturatedRequest,output,error) || !observation.reached || !observation.correct
            || !Probe::state().candidateLocalFrame.expired() || backend.calls!=callsBefore+1
            || output.materialDiagnostic!="preserved sentinel")
        {Observation::live()=nullptr;error="issued prepared receipt saturated transfer changed its value/charge association";return false;}
        observation.execution=&execution;observation.saturateTransfers=false;
    }
    Observation::live()=nullptr;
    const auto attempted=budget->logicalWork();
    const auto refusedCalls=backend.calls;
    {
        auto fill=NativeSceneAnimationCompositionBudget::reserve(budget,
            NativeSceneAnimationCompositionBudget::kMaximumBytes-budget->logicalBytes()-1,error);
        gltf::GlbColdAdmissionProbe::state()={};
        if (!fill || execution.executePreview(request,output,error) || budget->logicalWork()<=attempted
            || gltf::GlbColdAdmissionProbe::state().decodes || gltf::GlbColdAdmissionProbe::state().adaptations
            || output.materialDiagnostic!="preserved sentinel" || backend.calls!=refusedCalls)
        {error="one-short byte refusal reached an inadmissible cold decode/allocation";return false;}
    }
    // A separate genuine work boundary refuses before decoder/adaptation work.
    if (!budget->consumeWork(NativeSceneAnimationCompositionBudget::kMaximumWork-budget->logicalWork()-1,1))
    {error="work boundary setup saturated prematurely";return false;}
    gltf::GlbColdAdmissionProbe::state()={};
    if (execution.executePreview(request,output,error)
        || budget->logicalWork()!=NativeSceneAnimationCompositionBudget::kMaximumWork
        || gltf::GlbColdAdmissionProbe::state().decodes || gltf::GlbColdAdmissionProbe::state().adaptations
        || output.materialDiagnostic!="preserved sentinel" || backend.calls!=refusedCalls)
    {error="genuine exhausted-work refusal reached cold operations or changed the caller";return false;}
    error.clear();return true;
}
#endif

template <typename ReadColor>
bool verifyCameraLightCompositionProductionJoin(arbitgpu::NativeDeformationBackend& backend,
    ReadColor readColor, unsigned target, std::string& error)
{
    using namespace modelpayload;
    using namespace HarmonicMIDI::grid;
    const auto refuse=[&](const char* message){error=message;return false;};
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
    if (!verifyStaticColdTextureGrowthBoundaries(error) || !verifyStaticLocalCandidateStaging(error)) return false;
#endif
    Store store;
    visualanimationimport::ExactContentAssetKey key;
    if (!multiobjectanimationfixture::publish(store,key,true,false,target))
        return refuse("camera/light composition original GLB transfer refused");
    auto procedural=std::make_shared<Visual3DScene>(makeScene());
    procedural->materials[0].id={99};procedural->objects[0].material={99};
    procedural->objects[0].transform.translation.x=-.8f;
    ImportedAnimatedSceneRequest request;
    request.operation.asset=key;request.operation.sourceStableId=901;request.operation.deformationStableId=902;
    request.operation.schedule={901,902};request.operation.sceneIndex=0;
    request.operation.animationClipStableId=1;request.operation.clipName=std::string(visualstartermodel::kClipName);
    request.proceduralScene=procedural;request.structuralRevision=51;request.evaluationRevision=55;
    request.projectGeneration=21;request.helperGeneration=22;request.renderClipId=62;
    request.width=96;request.height=96;request.frame={0,24000,1001};
    const auto setFrame=[&](std::int64_t frame){request.frame.frame=frame;request.operation.playback.timelineSeconds=double(frame)*1001/24000;};
    ImportedAnimatedScenePayloadExecution execution(store,backend);
    ImportedAnimatedSceneReceipt start,moved,exported,output;
    if (!execution.executePreview(request,start,error)) return false;
    setFrame(12);
    if (!execution.executePreview(request,moved,error) || !execution.executeExport(request,exported,error)) return false;
    const auto typed=moved.sceneAnimationComposition;
    if (!typed || !moved.valid() || moved.evaluation.deformation || moved.source->deformation
        || moved.source->compositionOwner || !moved.source->draws.empty() || moved.runtimeInputs.compositionFrame
        || moved.frame.stats.dispatchCount!=0 || moved.frame.stats.drawCount!=4
        || start.source!=moved.source || !exported.sceneAnimationComposition
        || exported.sceneAnimationComposition->use()!=ImportedAnimationEvaluationOwner::Export
        || typed==exported.sceneAnimationComposition || typed->owner()!=exported.sceneAnimationComposition->owner()
        || typed->owner()->payload()!=store.resolvePreview(key) || typed->owner()->selection().sceneIndex!=0
        || typed->owner()->structuralRevision()!=51 || typed->owner()->evaluationRevision()!=55
        || typed->owner()->projectGeneration()!=21 || typed->owner()->helperGeneration()!=22
        || typed->owner()->renderClipId()!=62 || !(typed->time()==visualdeformation::RationalFrameTime {12,24000,1001})
        || typed->nativeFrame()!=moved.frame.nativeFrame || typed->owner()->scene()->objectCount!=4
        || typed->owner()->recordMappingCount()!=typed->owner()->scene()->cameraCount+typed->owner()->scene()->lightCount
        || moved.runtimeInputs.previousMotion || exported.runtimeInputs.previousMotion)
        return refuse("camera/light composition omitted authentic static owner, epoch, full scene or use provenance");
    if (typed->owner()->scene()->activeCamera.value!=(procedural->activeCamera.value|0x80000000u)
        || moved.runtimeInputs.animatedCamera)
        return refuse("camera/light composition replaced the authored procedural camera selection");
    const float displacement=.35f*float(12.0*1001/24000);
    if (target==1)
    {
        if (typed->sampledCameras().empty() || start.sceneAnimationComposition->sampledCameras().empty()
            || std::abs(typed->sampledCameras().front().transform.translation.x
                -start.sceneAnimationComposition->sampledCameras().front().transform.translation.x-displacement)>1.e-4f)
            return refuse("composed camera sample disagrees with literal original ancestor motion");
    }
    else if (typed->lights().empty() || std::abs(typed->lights().front().transform.translation.x
        -start.sceneAnimationComposition->lights().front().transform.translation.x-displacement)>1.e-4f)
        return refuse("composed light sample disagrees with literal original ancestor motion");
    for (std::size_t i=0;i<typed->owner()->recordMappingCount();++i)
    {
        const auto& map=typed->owner()->recordMappings()[i];
        if (!map.original || map.composed!=(map.imported?map.original:map.original|0x80000000u))
            return refuse("camera/light composition lost an exact original to composed record association");
    }
    const auto firstPixels=readColor(start.frame.nativeFrame),movedPixels=readColor(moved.frame.nativeFrame);
    if (firstPixels.empty() || movedPixels.empty() || movedPixels!=readColor(exported.frame.nativeFrame)
        || firstPixels!=readColor(start.frame.nativeFrame))
        return refuse("camera/light composition lost retained native images or rational Preview/Export parity");
    // Independent native reference uses the initial authored records plus the
    // literal translation above, never the newly sampled output records.
    auto referenceRuntime=start.runtimeInputs;
    referenceRuntime.timeSeconds=float(12.0*1001/24000);
    if (target==2) referenceRuntime.animatedLights.front().transform.translation.x+=displacement;
    auto& staticBackend=arbitgpu::nativeFixtureSceneBackend();
    const auto prepared=staticBackend.prepare(typed->owner()->scene(),{});
    if (!prepared.prepared) {error=prepared.error;return false;}
    const auto rendered=staticBackend.render(typed->owner()->scene(),prepared.resources,96,96,referenceRuntime);
    if (!rendered.rendered) {error=rendered.error;return false;}
    if (movedPixels!=readColor(rendered.frame))
        return refuse("camera/light composition disagrees with independent native authored-record reference");
    if (target==2 && firstPixels==movedPixels)
        return refuse("active composed imported light animation did not change native pixels");
    if (target==1) request.runtimeInputs.cameraOverride=typed->sampledCameras().front();
    else request.runtimeInputs.lightOverride=start.runtimeInputs.animatedLights.front();
    if (!execution.executePreview(request,output,error)) return false;
    if (target==1 && (!output.runtimeInputs.cameraOverride || output.runtimeInputs.animatedCamera))
        return refuse("graph camera override lost precedence over authored composed selection");
    auto overrideRuntime=start.runtimeInputs;
    overrideRuntime.timeSeconds=float(12.0*1001/24000);
    overrideRuntime.cameraOverride=request.runtimeInputs.cameraOverride;
    overrideRuntime.lightOverride=request.runtimeInputs.lightOverride;
    const auto overridden=staticBackend.render(typed->owner()->scene(),prepared.resources,96,96,overrideRuntime);
    if (!overridden.rendered || readColor(output.frame.nativeFrame)!=readColor(overridden.frame))
        return refuse("composed graph override pixels disagree with independent selected native records");
    request.runtimeInputs={};
    if (target==2)
    {
        request.importedLights.selectedDefinition=1;request.importedLights.intensityMultipliers[1]=2;
        if (!execution.executePreview(request,output,error)
            || output.runtimeInputs.animatedLights.front().intensity!=2*moved.runtimeInputs.animatedLights.front().intensity)
            return refuse("composed imported intensity control lost the exact definition association");
        request.importedLights={};
    }
    setFrame(0);
    if (!execution.executePreview(request,output,error) || readColor(output.frame.nativeFrame)!=firstPixels)
        return refuse("camera/light composition seek retained prior-frame state");
    request.operation.playback.playback=visualanimation::Playback::Loop;
    const auto duration=typed->owner()->animation()->clips.front().clip->durationSeconds();
    request.operation.playback.timelineSeconds=duration+12.0*1001/24000;
    request.frame.frame=12;
    if (!execution.executeExport(request,output,error) || readColor(output.frame.nativeFrame)!=movedPixels)
        return refuse("camera/light composition loop or Export sampling disagrees with source time");
    request.operation.playback.playback=visualanimation::Playback::Clamp;setFrame(12);
    auto conserved=output;
    request.evaluationRevision=54;
    if (execution.executePreview(request,output,error) || output.frame.nativeFrame!=conserved.frame.nativeFrame)
        return refuse("camera/light stale epoch changed incumbent output");
    request.evaluationRevision=55;request.operation.animationClipStableId=2;
    if (execution.executePreview(request,output,error) || output.source!=conserved.source)
        return refuse("camera/light stale selection changed incumbent source");
    request.operation.animationClipStableId=1;
    const auto project=request.projectGeneration;request.projectGeneration=0;
    if (execution.executePreview(request,output,error) || output.source!=conserved.source)
        return refuse("camera/light missing project owner changed incumbent source");
    request.projectGeneration=project;
    request.material=surfacematerialbinding::ImportedSceneMaterialRequest {};
    request.material->structuralRevision=51;request.material->evaluationRevision=54;
    if (execution.executeExport(request,output,error) || output.source!=conserved.source)
        return refuse("camera/light stale material epoch changed incumbent source");
    request.material.reset();
    request.runtimeInputs.imageOutput=renderpassoutput::Output::Motion;
    if (execution.executePreview(request,output,error) || output.frame.nativeFrame!=conserved.frame.nativeFrame)
        return refuse("camera/light composition accepted unsupported Motion or changed incumbent output");
    request.runtimeInputs={};++request.helperGeneration;request.evaluationRevision=56;
    if (!execution.executePreview(request,output,error) || output.sceneAnimationComposition->owner()==typed->owner()
        || typed->matches(output.payload,output.source,output.evaluation,output.frame.nativeFrame,output.owner,request.frame,output.runtimeInputs))
        return refuse("camera/light owner replacement reused foreign source, time or native frame authority");
    auto forged=output;forged.evaluation.sceneAnimation=start.evaluation.sceneAnimation;
    if (forged.valid()) return refuse("camera/light receipt accepted a substituted rational-time sample");
    forged=output;forged.owner=ImportedAnimationEvaluationOwner::Export;
    if (forged.valid()) return refuse("camera/light Preview authority accepted Export use");
    forged=output;++forged.frame.stats.time.frame;
    if (forged.valid()) return refuse("camera/light authority accepted a substituted rational project time");
    forged=output;forged.runtimeInputs.animatedLights.front().intensity+=1;
    if (forged.valid()) return refuse("camera/light authority accepted substituted sampled lighting");
    forged=output;forged.sceneAnimationComposition.reset();
    if (forged.valid()) return refuse("camera/light receipt degraded to the legacy untyped validity route");
    // Actual immutable sources survive reset, independently of native images.
    // Repeated source replacement must eventually refuse until escaped charges
    // retire; the failure leaves the prior receipt exactly intact.
    std::vector<std::shared_ptr<const NativeSceneAnimationCompositionOwner>> escaped;
    bool bounded=false;
    for (std::uint64_t epoch=57;epoch<89;++epoch)
    {
        escaped.push_back(output.sceneAnimationComposition->owner());
        execution.reset();request.evaluationRevision=epoch;
        const auto incumbent=output.frame.nativeFrame;
        if (!execution.executePreview(request,output,error))
        {
            if (error.find("logical CPU capacity")==std::string::npos || output.frame.nativeFrame!=incumbent)
                return refuse("camera/light escaped source shortage lost rollback or precise capacity diagnostic");
            bounded=true;break;
        }
    }
    if (!bounded) return refuse("camera/light escaped reset sources exceeded their finite logical allowance");
    escaped.clear();execution.reset();
    if (!execution.executePreview(request,output,error)) return false;
    // Inactive slots are valid authored input but are still copied and retained.
    // The oversized request must refuse at CPU admission, preserving the exact
    // previously submitted image and source rather than reaching native render.
    auto identityRequest=request;
    auto oversized=std::make_shared<Visual3DScene>(*procedural);
    oversized->lights.back().environmentImage.asset.id.assign(
        NativeSceneAnimationCompositionBudget::kMaximumBytes,'i');
    if (!validateVisual3DScene(*oversized).valid())
        return refuse("inactive oversized identity fixture is not a valid authored scene");
    identityRequest.proceduralScene=oversized;
    const auto conservedIdentityFrame=output.frame.nativeFrame;
    const auto conservedIdentitySource=output.source;
    if (execution.executePreview(identityRequest,output,error)
        || error.find("logical CPU capacity")==std::string::npos
        || output.frame.nativeFrame!=conservedIdentityFrame || output.source!=conservedIdentitySource)
        return refuse("inactive oversized light identity escaped prospective admission or changed incumbent");
    identityRequest=request;
    identityRequest.runtimeInputs.lightOverride=start.runtimeInputs.animatedLights.front();
    // Empty version-zero identities remain valid even with excess capacity.
    identityRequest.runtimeInputs.lightOverride->environmentImage.asset.contentSha256.reserve(
        NativeSceneAnimationCompositionBudget::kMaximumBytes);
    if (!validSceneEnvironmentImage(*identityRequest.runtimeInputs.lightOverride))
        return refuse("oversized runtime identity fixture is not a valid light override");
    if (execution.executePreview(identityRequest,output,error)
        || error.find("light identity")==std::string::npos
        || output.frame.nativeFrame!=conservedIdentityFrame || output.source!=conservedIdentitySource)
        return refuse("runtime light identity escaped pre-submission admission or changed incumbent");

    identityRequest=request;
    identityRequest.runtimeInputs.lightOverride=start.runtimeInputs.animatedLights.front();
    ImportedAnimatedSceneReceipt identityReference;
    if (!execution.executePreview(identityRequest,identityReference,error)) return false;
    auto identityScene=std::make_shared<Visual3DScene>(*procedural);
    identityScene->lights.back().environmentImage.asset.contentSha256.assign(48*1024,'h');
    identityScene->lights.front().environmentImage.asset.id.reserve(16*1024);
    if (!validateVisual3DScene(*identityScene).valid())
        return refuse("retained inactive and active identity fixture is not a valid authored scene");
    identityRequest=request;identityRequest.proceduralScene=identityScene;
    identityRequest.runtimeInputs.lightOverride=start.runtimeInputs.animatedLights.front();
    identityRequest.runtimeInputs.lightOverride->environmentImage.asset.id.reserve(16*1024);
    ImportedAnimatedScenePayloadExecution identityExecution(store,backend);
    ImportedAnimatedSceneReceipt identityOutput;
    if (!identityExecution.executePreview(identityRequest,identityOutput,error)) return false;
    if (readColor(identityOutput.frame.nativeFrame)!=readColor(identityReference.frame.nativeFrame)
        || identityOutput.sceneAnimationComposition->owner()->proceduralScene()->lights.back()
            .environmentImage.asset.contentSha256!=identityScene->lights.back().environmentImage.asset.contentSha256)
        return refuse("valid active or inactive light identities changed rendering or retained scene bytes");
    // Escape the scene alias itself. Holding only this input must keep its
    // identity debit, independently of the typed owner or native frame.
    std::vector<std::shared_ptr<const Visual3DScene>> escapedIdentityScenes;
    bool identityBounded=false;
    const auto identityEpochLimit=identityRequest.evaluationRevision+33;
    for (std::uint64_t epoch=identityRequest.evaluationRevision+1;epoch<identityEpochLimit;)
    {
        escapedIdentityScenes.push_back(identityOutput.sceneAnimationComposition->owner()->proceduralScene());
        identityExecution.reset();identityRequest.evaluationRevision=epoch++;
        const auto prior=identityOutput.frame.nativeFrame;
        if (!identityExecution.executePreview(identityRequest,identityOutput,error))
        {
            if (error.find("logical CPU capacity")==std::string::npos || identityOutput.frame.nativeFrame!=prior)
                return refuse("escaped nested identity debit lost reset conservation or rollback");
            identityBounded=true;break;
        }
    }
    if (!identityBounded) return refuse("escaped light identity aliases exceeded finite storage");
    escapedIdentityScenes.clear();identityExecution.reset();
    if (!identityExecution.executePreview(identityRequest,identityOutput,error)) return false;

    SceneLightRecord identityLight;
    identityLight.environmentImage.asset.id.reserve(127);
    identityLight.environmentImage.asset.contentSha256.reserve(63);
    const auto identityBytes=identityLight.environmentImage.asset.id.capacity()+1
        +identityLight.environmentImage.asset.contentSha256.capacity()+1;
    std::size_t exactIdentityBytes=NativeSceneAnimationCompositionBudget::kMaximumBytes-identityBytes;
    std::size_t shortIdentityBytes=exactIdentityBytes+1;
    if (!NativeSceneAnimationCompositionBudget::addLightIdentityBytes(identityLight,1,exactIdentityBytes,error)
        || exactIdentityBytes!=NativeSceneAnimationCompositionBudget::kMaximumBytes
        || NativeSceneAnimationCompositionBudget::addLightIdentityBytes(identityLight,1,shortIdentityBytes,error))
        return refuse("one-short nested light identity storage admitted a copy");
    auto identityWorkBudget=std::make_shared<NativeSceneAnimationCompositionBudget>();
    if (!identityWorkBudget->work(NativeSceneAnimationCompositionBudget::kMaximumWork-identityBytes+1,1,error)
        || identityWorkBudget->consumeLightIdentityWork(identityLight,1)
        || identityWorkBudget->consumeLightIdentityWork(identityLight,1)
        || identityWorkBudget->logicalWork()!=NativeSceneAnimationCompositionBudget::kMaximumWork)
        return refuse("one-short identity copy or comparison work was admitted or refunded");
    auto budget=std::make_shared<NativeSceneAnimationCompositionBudget>();
    const auto room=NativeSceneAnimationCompositionBudget::kMaximumBytes-budget->logicalBytes();
    auto charge=NativeSceneAnimationCompositionBudget::reserve(budget,room-1,error);
    if (!charge || NativeSceneAnimationCompositionBudget::reserve(budget,2,error))
        return refuse("camera/light one-short storage admission exceeded the typed allowance");
    charge.reset();
    if (!budget->work(NativeSceneAnimationCompositionBudget::kMaximumWork-1,1,error)
        || budget->work(2,1,error) || budget->work(1,1,error)
        || budget->logicalWork()!=NativeSceneAnimationCompositionBudget::kMaximumWork)
        return refuse("camera/light failed named work was refunded or reset");
    error.clear();return true;
}

template <typename ReadColor>
bool verifyImportedCameraSelectionProductionJoin(arbitgpu::NativeDeformationBackend& backend,
    ReadColor readColor, std::string& error)
{
    using namespace modelpayload;
    using namespace HarmonicMIDI::grid;
    const auto refuse = [&](const char* message) { error=message; return false; };
    Store store;
    const auto bytes = importedCameraChoiceBytes();
    visualanimationimport::ExactContentAssetKey key {"camera-choice",1,
        modelpayload::detail::digestHex(bytes),"model/gltf-binary",bytes.size()};
    if (!store.begin(key.id,key) || !store.appendBase64(key.id,0,productionCompositionBase64(bytes)) || !store.commit(key.id))
        return refuse("selected-camera exact source transfer failed");
    visualimportedscenerender::Request leaf;
    leaf.asset=key; leaf.sceneIndex=0; leaf.sourceStableId=901; leaf.renderStableId=902;
    leaf.structuralRevision=51; leaf.evaluationRevision=55; leaf.sourceCameraNodeId=7;
    leaf.deformation.emplace();
    auto& animation=*leaf.deformation;
    animation.asset=key; animation.sceneIndex=0; animation.sourceStableId=901; animation.deformationStableId=903;
    animation.schedule={901,903}; animation.animationClipStableId=1; animation.clipName=std::string(visualstartermodel::kClipName);
    auto procedural=std::make_shared<Visual3DScene>(makeScene());
    procedural->materials[0].id={99}; procedural->objects[0].material={99};
    visualimportedscenerender::Request authored;
    authored.sourceStableId=procedural->id.value; authored.renderStableId=902;
    authored.structuralRevision=51; authored.evaluationRevision=55; authored.sceneSnapshot=procedural;
    authored.importedScene=std::make_shared<const visualimportedscenerender::Request>(leaf);
    const auto wire=visualimportedscenerender::encode(authored);
    visualimportedscenerender::Request decoded;
    if (wire.empty() || !visualimportedscenerender::decodeCanonical(wire,decoded,error)
        || !visualimportedscenerender::sameRequest(authored,decoded)) return false;
    ImportedSceneRequest request;
    request.importedScene=decoded.importedScene; request.asset=decoded.importedScene->asset;
    request.sceneIndex=decoded.importedScene->sceneIndex; request.sourceCameraNodeId=decoded.importedScene->sourceCameraNodeId;
    request.deformation=decoded.importedScene->deformation; request.sceneSnapshot=decoded.sceneSnapshot;
    request.structuralRevision=51; request.evaluationRevision=55; request.projectGeneration=21; request.helperGeneration=22;
    request.clipId=62; request.dimensions={96,96}; request.frame={0,24000,1001}; request.staticPayloadIdentity=wire;
    ImportedScenePayloadExecution execution(store,arbitgpu::nativeFixtureSceneBackend(),backend);
    ImportedSceneExecutionReceipt first, moved, exported;
    // Exercise the ordinary canonical compiled-plan to per-frame request handoff.
    videowire::CompiledVisualLayerPlan plan;
    plan.clipId=62; plan.structuralRevision=51; plan.producerValidated=true;
    const auto sceneNode=static_cast<int>(authored.sourceStableId-1);
    plan.nodeIds={sceneNode,901}; plan.nodeKinds={"visual.3d.scene.retained","visual.3d.render"};
    plan.operations={{sceneNode,plan.nodeKinds[0],"control-eval",""},{901,plan.nodeKinds[1],"native-gpu",wire}};
    plan.edges={{sceneNode,0,901,0}};
    plan.ports={{sceneNode,0,1,"out","control","scene3D","unspecified","unspecified"},
        {901,0,1,"in","control","scene3D","unspecified","unspecified"},{901,1,1,"out","frame","image","rgba8","sRGB"}};
    importedscene::VisualImportedScenePlanCache planCache;
    videorender::LayerDesc layer;
    std::vector<ImportedSceneExecutionReceipt> owners;
    const auto prepare=[&](double seconds, importedscene::NativeImportedSceneRenderUse use, ImportedSceneExecutionReceipt& out) {
        if (importedscene::prepareVisualImportedSceneLayerAtTime({plan},62,96,96,seconds,24000.0/1001,use,
                &execution,&planCache,layer,owners,error,nullptr,{21,22}) != importedscene::VisualImportedScenePreparation::rendered
            || owners.empty()) return false;
        out=owners.back();return true;
    };
    if (!prepare(0,importedscene::NativeImportedSceneRenderUse::Preview,first)) return false;
    request.frame.frame=12; request.deformation->playback.timelineSeconds=12.0*1001/24000;
    if (!prepare(12.0*1001/24000,importedscene::NativeImportedSceneRenderUse::Preview,moved)
        || !prepare(12.0*1001/24000,importedscene::NativeImportedSceneRenderUse::Export,exported)) return false;
    if (!first.deformationFrame || !moved.deformationFrame || !exported.deformationFrame) return refuse("selected camera omitted actual animated owners");
    const auto& f=*first.deformationFrame; const auto& m=*moved.deformationFrame;
    const auto typed=m.sceneAnimationComposition;
    const float displacement=.35f*float(12.0*1001/24000);
    if (!typed || !m.valid() || typed->owner()->sourceCameraNodeId()!=7 || m.sourceCameraNodeId!=7
        || !f.runtimeInputs.animatedCamera || !m.runtimeInputs.animatedCamera
        || f.runtimeInputs.animatedCamera->id.value!=7 || m.runtimeInputs.animatedCamera->id.value!=7
        || std::abs(f.runtimeInputs.animatedCamera->verticalFovRadians-.6f)>1.e-6f
        || std::abs(f.runtimeInputs.animatedCamera->nearPlane-.05f)>1.e-6f
        || f.runtimeInputs.animatedCamera->farPlane!=30 || f.runtimeInputs.animatedCamera->transform.translation.y!=3.75f
        || std::abs(f.runtimeInputs.animatedCamera->transform.translation.x)>1.e-6f
        || std::abs(m.runtimeInputs.animatedCamera->transform.translation.x-displacement)>1.e-4f
        || m.runtimeInputs.cameraOverride || typed->owner()->scene()->activeCamera.value!=(procedural->activeCamera.value|0x80000000u)
        || typed->owner()->document()->nodes[6].camera!=1 || typed->owner()->document()->nodes[6].parent!=5
        || !typed->matches(m.payload,m.source,m.evaluation,m.frame.nativeFrame,m.owner,m.frame.stats.time,m.runtimeInputs,7)
        || typed->matches(m.payload,m.source,m.evaluation,m.frame.nativeFrame,m.owner,m.frame.stats.time,m.runtimeInputs,3))
        return refuse("selected source node/projection/ancestor/time/owner authority disagrees with literal camera 7");
    const auto pixels=readColor(m.frame.nativeFrame);
    if (pixels.empty() || !std::any_of(pixels.begin(),pixels.end(),[](auto channel){return channel!=0;})
        || pixels!=readColor(exported.deformationFrame->frame.nativeFrame))
        return refuse("selected source camera lost native Preview/Export parity");
    auto reference=f.runtimeInputs;
    reference.timeSeconds=float(12.0*1001/24000); reference.animatedCamera->transform.translation.x=displacement;
    auto& staticBackend=arbitgpu::nativeFixtureSceneBackend();
    const auto prepared=staticBackend.prepare(typed->owner()->scene(),{});
    if (!prepared.prepared) {error=prepared.error;return false;}
    const auto expected=staticBackend.render(typed->owner()->scene(),prepared.resources,96,96,reference);
    if (!expected.rendered || pixels!=readColor(expected.frame)) return refuse("selected-camera native pixels disagree with literal source-record oracle");
    auto selected=request;
    auto selectedLeaf=std::make_shared<visualimportedscenerender::Request>(*request.importedScene);
    selectedLeaf->sourceCameraNodeId=3; selected.importedScene=selectedLeaf; selected.sourceCameraNodeId=3;
    ImportedSceneExecutionReceipt changed;
    if (!execution.executePreview(selected,changed,error) || !changed.deformationFrame
        || changed.deformationFrame->source==m.source || changed.deformationFrame->runtimeInputs.animatedCamera->id.value!=3
        || std::abs(changed.deformationFrame->runtimeInputs.animatedCamera->verticalFovRadians-.39959652f)>1.e-6f
        || changed.deformationFrame->runtimeInputs.animatedCamera->farPlane!=20)
        return refuse("changed selector reused a stale source camera owner");
    for (const auto node : {9u,10u})
    {
        auto stale=request; auto staleLeaf=std::make_shared<visualimportedscenerender::Request>(*request.importedScene);
        staleLeaf->sourceCameraNodeId=node; stale.importedScene=staleLeaf; stale.sourceCameraNodeId=node;
        if (execution.executePreview(stale,changed,error)) return refuse("out-of-scene or synthesized camera identity was admitted");
    }
    // Distinct actual graph camera record remains highest priority after private mapping.
    auto overridden=request;
    overridden.camera=procedural->cameras[0];
    if (!execution.executePreview(overridden,changed,error) || !changed.deformationFrame
        || !changed.deformationFrame->runtimeInputs.cameraOverride
        || changed.deformationFrame->runtimeInputs.cameraOverride->id.value!=(procedural->activeCamera.value|0x80000000u)
        || changed.deformationFrame->runtimeInputs.animatedCamera->id.value!=7)
        return refuse("authored graph camera override lost precedence or exact private mapping");
    auto mismatch=overridden; mismatch.camera->transform.translation.x+=1;
    if (execution.executePreview(mismatch,changed,error)) return refuse("mismatched graph override bypassed authored record equality");
    // The same selected record reaches the static owner without animation or composition.
    ImportedSceneRequest staticRequest;
    staticRequest.asset=key; staticRequest.sceneIndex=0; staticRequest.sourceCameraNodeId=7;
    staticRequest.clipId=70; staticRequest.projectGeneration=21; staticRequest.helperGeneration=22;
    staticRequest.dimensions={96,96}; staticRequest.frame={0,24000,1001};
    ImportedSceneExecutionReceipt staticFrame;
    if (!execution.executePreview(staticRequest,staticFrame,error) || !staticFrame.valid()) return false;
    arbitgpu::NativeFixtureSceneRuntimeInputs staticInputs;
    const auto selectedCamera=std::find_if(staticFrame.admission.scene->cameras.begin(),
        staticFrame.admission.scene->cameras.begin()+staticFrame.admission.scene->cameraCount,
        [](const auto& camera){return camera.id.value==7;});
    if (selectedCamera==staticFrame.admission.scene->cameras.begin()+staticFrame.admission.scene->cameraCount
        || selectedCamera->verticalFovRadians!=.6f || selectedCamera->transform.translation.y!=3.75f)
        return refuse("static camera reference lost literal source node/projection");
    staticInputs.animatedCamera=*selectedCamera;
    const auto staticPrepared=staticBackend.prepare(staticFrame.admission.scene,{});
    if (!staticPrepared.prepared) {error=staticPrepared.error;return false;}
    const auto staticExpected=staticBackend.render(staticFrame.admission.scene,staticPrepared.resources,96,96,staticInputs);
    if (!staticExpected.rendered || readColor(staticExpected.frame)!=readColor(staticFrame.frame.rendered.nativeFrame))
        return refuse("static selected-camera native frame disagrees with literal source-camera oracle");
    for (const auto node : {9u,10u}) { staticRequest.sourceCameraNodeId=node;
        if (execution.executePreview(staticRequest,staticFrame,error)) return refuse("static camera admitted stale or synthetic source identity"); }
    error.clear();return true;
}

inline std::vector<std::uint8_t> selectedCameraSkinBytes()
{
    const auto original=compositionGlb(false,false,CompositionSkinCase::Node);
    const auto read=[](const auto* p) {return std::uint32_t(p[0])|std::uint32_t(p[1])<<8
        |std::uint32_t(p[2])<<16|std::uint32_t(p[3])<<24;};
    const auto jsonBytes=read(original.data()+12),binStart=28u+jsonBytes;
    auto root=nlohmann::json::parse(original.begin()+20,original.begin()+20+jsonBytes);
    root["cameras"]=nlohmann::json::array({
        {{"type","perspective"},{"perspective",{{"yfov",.6},{"znear",.05},{"zfar",30.0}}}},
        {{"type","perspective"},{"perspective",{{"yfov",.9},{"znear",.1},{"zfar",40.0}}}}});
    root["nodes"].push_back({{"name","Animated camera rig"},{"children",{2}}});
    root["nodes"].push_back({{"name","Selected skin view"},{"camera",0},{"translation",{0.0,0.0,6.0}}});
    root["nodes"].push_back({{"name","Alternate skin view"},{"camera",1},{"translation",{0.0,0.0,6.0}}});
    root["scenes"][0]["nodes"].push_back(1);root["scenes"][0]["nodes"].push_back(3);
    // The actual cube joint and camera ancestor share the original 0 -> 2
    // translation sampler. The original authored skin and morph remain intact.
    root["animations"][0]["channels"].push_back({{"sampler",0},{"target",{{"node",1},{"path","translation"}}}});
    auto json=root.dump();while (json.size()%4) json+=' ';
    const auto binBytes=read(original.data()+binStart-8);
    std::vector<std::uint8_t> bytes;
    const auto append=[&](std::uint32_t value){for (unsigned shift=0;shift<32;shift+=8)
        bytes.push_back(static_cast<std::uint8_t>(value>>shift));};
    for (auto value:{0x46546c67u,2u,std::uint32_t(28+json.size()+binBytes),std::uint32_t(json.size()),0x4e4f534au}) append(value);
    bytes.insert(bytes.end(),json.begin(),json.end());append(binBytes);append(0x004e4942u);
    bytes.insert(bytes.end(),original.begin()+binStart,original.begin()+binStart+binBytes);return bytes;
}

template <typename ReadColor>
bool verifySelectedDeformationCameraReceipt(arbitgpu::NativeDeformationBackend& backend,
    ReadColor readColor,std::string& error)
{
    using namespace modelpayload;
    using namespace HarmonicMIDI::grid;
    const auto refuse=[&](const char* message){error=message;return false;};
    const auto bytes=selectedCameraSkinBytes();
    const visualanimationimport::ExactContentAssetKey key {"selected-camera-real-skin",1,
        sha256Bytes(bytes.data(),bytes.size()),"model/gltf-binary",bytes.size()};
    Store store;
    if (!store.begin("selected-skin",key) || !store.appendBase64("selected-skin",0,productionCompositionBase64(bytes))
        || !store.commit("selected-skin")) return refuse("selected-camera skin original GLB transfer refused");
    for (const bool composed:{false,true})
    {
        ImportedAnimatedScenePayloadExecution execution(store,backend);
        ImportedAnimatedSceneRequest request;
        request.operation.asset=key;request.operation.sourceStableId=1001;request.operation.deformationStableId=1002;
        request.operation.schedule={1001,1002};request.operation.sceneIndex=0;request.operation.meshStableId=1;
        request.operation.animationClipStableId=1;request.operation.clipName="Original composition action";
        request.sourceCameraNodeId=3;request.structuralRevision=71;request.width=96;request.height=96;
        request.projectGeneration=21;request.helperGeneration=22;request.renderClipId=72;
        if (composed) {request.proceduralScene=std::make_shared<Visual3DScene>(makeScene());request.evaluationRevision=75;}
        request.frame={15,30,1};request.operation.playback.timelineSeconds=.5;
        ImportedAnimatedSceneReceipt selected,exported,foreign;
        if (!execution.executePreview(request,selected,error) || !execution.executeExport(request,exported,error)) return false;
        if (!selected.valid() || !exported.valid() || !selected.evaluation.deformation
            || !selected.source->deformation || selected.evaluation.deformation->jointTransforms().empty()
            || selected.evaluation.deformation->morphWeights().empty() || !selected.evaluation.sceneAnimation
            || selected.sceneAnimationComposition || !selected.runtimeInputs.animatedCamera
            || selected.runtimeInputs.animatedCamera->id.value!=3
            || std::abs(selected.runtimeInputs.animatedCamera->transform.translation.x-1)>1.e-5f
            || selected.runtimeInputs.animatedCamera->transform.translation.z!=6
            || selected.runtimeInputs.animatedCamera->verticalFovRadians!=.6f
            || selected.runtimeInputs.cameraOverride)
            return refuse("genuine selected-camera skin receipt omitted exact sampled camera or actual deformation");
        const auto pixels=readColor(selected.frame.nativeFrame);
        if (pixels.empty() || pixels!=readColor(exported.frame.nativeFrame))
            return refuse("selected-camera skin native Preview/Export parity failed");
        auto changed=selected;changed.runtimeInputs.animatedCamera->verticalFovRadians=.9f;
        if (changed.valid()) return refuse("selected skin receipt accepted a copied projection mutation");
        changed=selected;changed.runtimeInputs.animatedCamera->transform.translation.x+=1;
        if (changed.valid()) return refuse("selected skin receipt accepted a copied sampled-transform mutation");
        changed=selected;changed.sourceCameraNodeId=4;changed.runtimeInputs.animatedCamera->id.value=4;
        if (changed.valid()) return refuse("selected skin receipt accepted a copied selector and camera ID mutation");
        changed=selected;changed.sourceCameraNodeId.reset();
        if (changed.valid()) return refuse("selected skin receipt accepted removal of its public selector");
        changed.runtimeInputs.animatedCamera.reset();changed.sceneAnimationComposition.reset();
        if (changed.valid()) return refuse("selected skin receipt accepted removal of all public camera companions");
        changed=selected;changed.runtimeInputs.cameraOverride=selected.runtimeInputs.animatedCamera;
        if (changed.valid()) return refuse("selected skin receipt accepted a foreign override presence");
        changed=selected;changed.frame.nativeFrame=exported.frame.nativeFrame;
        if (changed.valid()) return refuse("selected skin receipt accepted a foreign completed native frame");
        changed=selected;changed.evaluation.sceneAnimation.reset();
        if (changed.valid()) return refuse("selected skin receipt accepted removal of its actual animation sample");
        changed=selected;changed.evaluation.deformation=exported.evaluation.deformation;
        changed.frame.deformation=exported.evaluation.deformation;
        if (changed.valid()) return refuse("selected skin receipt accepted a foreign deformation sample");
        changed=selected;changed.owner=ImportedAnimationEvaluationOwner::Export;
        if (changed.valid()) return refuse("selected skin receipt accepted a copied use mutation");
        changed=selected;++changed.frame.stats.time.frame;
        if (changed.valid()) return refuse("selected skin receipt accepted a copied rational-time mutation");
        request.runtimeInputs.cameraOverride=composed?request.proceduralScene->cameras[0]:*selected.runtimeInputs.animatedCamera;
        if (!execution.executePreview(request,foreign,error) || !foreign.valid() || !foreign.runtimeInputs.cameraOverride)
            return refuse("selected skin genuine graph-camera override failed");
        changed=foreign;changed.runtimeInputs.cameraOverride.reset();
        if (changed.valid()) return refuse("selected skin receipt accepted removal of its actual override");
        changed=foreign;changed.runtimeInputs.cameraOverride->nearPlane+=.01f;
        if (changed.valid()) return refuse("selected skin receipt accepted mutation of its actual override");
        request.runtimeInputs={};request.sourceCameraNodeId.reset();
        if (!execution.executePreview(request,foreign,error) || !foreign.valid() || foreign.sourceCameraNodeId)
            return refuse("selected camera association changed no-selection skin behavior");
        // A copied selected receipt remains authentic after the executor drops
        // histories. Actual native and animation owners carry the retained debit.
        execution.reset();
        if (!selected.valid()) return refuse("selected skin receipt lost its escaped private association after reset");
    }
    error.clear();return true;
}

// These capabilities come exclusively from the actual production executor.
// The existing named test friend is used by other fixtures, never by this one.
template <typename ReadColor>
bool verifyNativeCompositionProductionJoin(arbitgpu::NativeDeformationBackend& backend,
    ReadColor readColor,std::string& error)
{
    using namespace modelpayload;
    using namespace HarmonicMIDI::grid;
    error.clear();
    const auto refuse=[&](const char* message) {error=message;return false;};
    const auto bytes=compositionGlb(false,false,CompositionSkinCase::Node);
    const visualanimationimport::ExactContentAssetKey key {"native-production-composition",1,
        sha256Bytes(bytes.data(),bytes.size()),"model/gltf-binary",bytes.size()};
    Store store;
    if (!store.begin("production",key) || !store.appendBase64("production",0,productionCompositionBase64(bytes))
        || !store.commit("production")) return refuse("production composition original GLB transfer refused");
    auto procedural=std::make_shared<Visual3DScene>(makeScene());
    procedural->objects[0].id={65537};procedural->objects[0].transform.translation.x=-.8f;
    ImportedAnimatedScenePayloadExecution execution(store,backend);
    ImportedAnimatedSceneRequest request;
    request.operation.sourceStableId=801;request.operation.deformationStableId=802;
    request.operation.schedule={801,802};request.operation.asset=key;
    request.operation.sceneIndex=0;request.operation.meshStableId=1;request.operation.animationClipStableId=1;
    request.operation.clipName="Original composition action";
    request.proceduralScene=procedural;request.structuralRevision=31;request.evaluationRevision=35;
    request.projectGeneration=11;request.helperGeneration=12;request.renderClipId=42;
    request.width=96;request.height=96;request.frame={14,30,1};
    request.runtimeInputs.rawExportMask=render3dimage::kRawExportMask;
    const auto setFrame=[&](std::int64_t frame) {request.frame.frame=frame;request.operation.playback.timelineSeconds=double(frame)/30;};
    ImportedAnimatedSceneReceipt first,current,exported;
    setFrame(14);
    if (!execution.executePreview(request,first,error)) return false;
    setFrame(15);
    if (!execution.executePreview(request,current,error)) return false;
    const auto typed=current.runtimeInputs.compositionFrame;
    if (!typed || !current.source->compositionOwner || typed->owner()!=current.source->compositionOwner
        || current.source->draws.size()!=2 || typed->owner()->memberCount()!=2
        || current.source->structuralRevision!=31 || current.source->evaluationRevision!=35
        || current.evaluation.deformation->revision()!=31 || typed->evaluationRevision()!=35
        || typed->owner()->projectGeneration()!=11 || typed->owner()->helperGeneration()!=12
        || current.evaluation.sourceStableId!=801 || current.evaluation.deformationStableId!=802
        || !(typed->time()==visualdeformation::RationalFrameTime {15,30,1})
        || !current.runtimeInputs.previousMotion
        || current.runtimeInputs.previousMotion->compositionFrame!=first.runtimeInputs.compositionFrame
        || current.runtimeInputs.previousMotion->deformation!=first.evaluation.deformation
        || current.runtimeInputs.previousMotion->owner.evaluationRevision!=35)
        return refuse("production composition omitted exact owner, member, epoch, time or predecessor provenance");
    arbitgpu::NativeDeformationFrameData data;
    bool imported=false,generated=false;
    for (const auto& draw:current.source->draws)
    {
        if (!arbitgpu::prepareNativeDeformationFrame(*draw,*current.evaluation.deformation,typed,data,error))
        {
            if (error.empty()) error="production composition frame preparation refused a member without a diagnostic";
            return false;
        }
        if (draw->object.value==1)
        {
            imported=true;
            // Authored translation is 0 -> 2 metres over one second. At 15/30
            // the original (-1,-1,-1) corner becomes (0,-1,-1). Zero morph
            // deltas and identity post keep this independent literal oracle.
            if (data.jointCount!=1 || std::abs(data.jointPalette[12]-1)>1.0e-5f
                || std::abs(data.compositionPost.point()[12])>1.0e-5f)
                return refuse("production imported member disagrees with literal translated coordinates");
        }
        else if (draw->object.value==(0x80000000u|65537u))
        {
            generated=true;
            if (draw->animationNodeStableId!=0 || data.jointCount!=1
                || data.compositionPost.control()[2]!=2 || std::abs(data.jointPalette[12])>1.0e-5f
                || std::abs(data.compositionPost.point()[12])>1.0e-5f)
                return refuse("production procedural member was omitted or treated as an imported node");
        }
        else return refuse("production composition emitted a foreign member identity");
    }
    if (!imported || !generated) return refuse("production composition omitted a complete native member");
    std::string colorDetail;
    const auto verifyColor=[&](const ImportedAnimatedSceneReceipt& receipt,std::int64_t frame) {
        auto reference=std::make_shared<Visual3DScene>(*receipt.source->scene);
        const auto object=std::find_if(reference->objects.begin(),reference->objects.begin()+reference->objectCount,
            [](const auto& value){return value.id.value==1;});
        if (object==reference->objects.begin()+reference->objectCount) {colorDetail="imported object 1 not found";return false;}
        for (std::size_t i=object->firstVertex;i<object->firstVertex+object->vertexCount;++i)
            reference->vertices[i].position.x+=float(frame)/15;
        auto& staticBackend=arbitgpu::nativeFixtureSceneBackend();
        const auto prepared=staticBackend.prepare(reference,{});
        if (!prepared.prepared) {error=prepared.error;return false;}
        const auto expected=staticBackend.render(reference,prepared.resources,96,96,{});
        if (!expected.rendered) {error=expected.error;return false;}
        const auto actualPixels=readColor(receipt.frame.nativeFrame),expectedPixels=readColor(expected.frame);
        if (actualPixels.empty() || actualPixels.size()!=expectedPixels.size())
        {colorDetail="pixel bytes actual "+std::to_string(actualPixels.size())+" vs expected "+std::to_string(expectedPixels.size());return false;}
        std::size_t mismatches=0,firstByte=actualPixels.size();
        for (std::size_t i=0;i<actualPixels.size();++i)
            if (std::abs(int(actualPixels[i])-int(expectedPixels[i]))>6) {++mismatches; if (firstByte==actualPixels.size()) firstByte=i;}
        if (mismatches==0) return true;
        const std::size_t pixel=firstByte/4;
        const auto rgba=[](const auto& bytes,std::size_t p) {
            return "("+std::to_string(int(bytes[4*p]))+","+std::to_string(int(bytes[4*p+1]))+","
                +std::to_string(int(bytes[4*p+2]))+","+std::to_string(int(bytes[4*p+3]))+")";
        };
        colorDetail="pixels compared "+std::to_string(actualPixels.size()/4)+", mismatched channels "+std::to_string(mismatches)
            +", first mismatching pixel "+std::to_string(pixel)+" actual RGBA "+rgba(actualPixels,pixel)
            +" expected RGBA "+rgba(expectedPixels,pixel)+", tolerance 6";
        // The list covers every mismatched pixel, capped at 64 lines. A pixel is
        // mismatched when any RGBA channel differs by more than 6, the same threshold
        // as the count above. The side matches the 96 by 96 reference render.
        constexpr std::size_t side=96, listCap=64;
        const std::size_t pixelCount=actualPixels.size()/4;
        std::size_t mismatchedPixels=0,listed=0;
        for (std::size_t p=0;p<pixelCount;++p)
        {
            bool differs=false;
            for (std::size_t c=0;c<4;++c)
                if (std::abs(int(actualPixels[4*p+c])-int(expectedPixels[4*p+c]))>6) differs=true;
            if (!differs) continue;
            ++mismatchedPixels;
            if (listed==listCap) continue;
            ++listed;
            std::cerr << "production composition mismatched pixel (" << p%side << "," << p/side << ") actual "
                << rgba(actualPixels,p) << " expected " << rgba(expectedPixels,p) << '\n';
        }
        std::cerr << "production composition mismatched pixels " << mismatchedPixels << " of " << pixelCount
            << ", listed " << listed << ", corner (0,0) actual " << rgba(actualPixels,0)
            << " expected " << rgba(expectedPixels,0) << " match=" << (rgba(actualPixels,0)==rgba(expectedPixels,0)) << '\n';
        return false;
    };
    if (!verifyColor(current,15)) {
        std::cerr << "production composition pixel diagnostic: " << colorDetail << '\n';
        return refuse("production composition pixels disagree with independent translated geometry");
    }
    if (!execution.executeExport(request,exported,error) || exported.runtimeInputs.previousMotion)
        return refuse("production Export borrowed Preview history");
    setFrame(16);
    ImportedAnimatedSceneReceipt previewNext,exportNext;
    if (!execution.executeExport(request,exportNext,error) || !exportNext.runtimeInputs.previousMotion
        || exportNext.runtimeInputs.previousMotion->deformation!=exported.evaluation.deformation
        || !execution.executePreview(request,previewNext,error) || !previewNext.runtimeInputs.previousMotion
        || previewNext.runtimeInputs.previousMotion->deformation!=current.evaluation.deformation)
        return refuse("production Preview and Export lost independent adjacent predecessors");
    auto missing=std::make_shared<arbitgpu::NativeDeformationScene>(*current.source);missing->draws.pop_back();
    const auto incumbent=data;
    if (arbitgpu::prepareNativeDeformationFrame(*missing,*current.evaluation.deformation,typed,data,error)
        || data.jointPalette!=incumbent.jointPalette) return refuse("production typed owner accepted an omitted member");
    auto substituted=std::make_shared<arbitgpu::NativeDeformationScene>(*current.source);++substituted->sourceStableId;
    if (arbitgpu::prepareNativeDeformationFrame(*substituted,*current.evaluation.deformation,typed,data,error))
        return refuse("production typed owner accepted substituted source identity");
    auto changedDeformation=std::make_shared<arbitgpu::NativeDeformationScene>(*current.source);
    ++changedDeformation->deformationStableId;
    if (arbitgpu::prepareNativeDeformationFrame(*changedDeformation,*current.evaluation.deformation,typed,data,error)
        || arbitgpu::prepareNativeDeformationFrame(*current.source,*first.evaluation.deformation,typed,data,error))
        return refuse("production typed frame accepted foreign deformation or rational time");
    request.evaluationRevision=36;
    ImportedAnimatedSceneReceipt changed;
    if (!execution.executePreview(request,changed,error) || changed.runtimeInputs.previousMotion
        || changed.source==previewNext.source || changed.runtimeInputs.compositionFrame->owner()==typed->owner()
        || arbitgpu::prepareNativeDeformationFrame(*changed.source,*current.evaluation.deformation,typed,data,error))
        return refuse("production changed epoch reused a stale owner or Motion sample");
    request.evaluationRevision=35;
    const auto keptFrame=changed.frame.nativeFrame;
    const auto keptSource=changed.source;
    if (execution.executeExport(request,changed,error) || changed.frame.nativeFrame!=keptFrame || changed.source!=keptSource)
        return refuse("production stale sibling changed the incumbent output");
    request.evaluationRevision=36;setFrame(17);
    if (!execution.executePreview(request,changed,error) || !changed.runtimeInputs.previousMotion
        || !(changed.runtimeInputs.previousMotion->deformation->time()==visualdeformation::RationalFrameTime {16,30,1}))
        return refuse("production refused sibling destroyed the Preview predecessor");
    // Retain issued CPU authorities while releasing native images. Their debit
    // must outlive both history slots and any receipt. Native-driver allocation
    // remains a separate authority. Refusal conserves output; release admits.
    struct Escaped final
    {
        std::shared_ptr<const arbitgpu::NativeDeformationScene> source;
        std::shared_ptr<const arbitgpu::NativeDeformationCompositionFrame> frame;
        std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot> snapshot;
        std::shared_ptr<const visualanimation::Sample> animation;
    };
    std::vector<Escaped> escaped;
    bool bounded=false;
    for (std::uint64_t epoch=37;epoch<293;++epoch)
    {
        request.evaluationRevision=epoch;
        ImportedAnimatedSceneReceipt receipt=changed;
        if (!execution.executePreview(request,receipt,error))
        {
            if (error.find("logical CPU capacity")==std::string::npos || receipt.source!=changed.source)
                return refuse("production escaped retention refusal changed output or lacked a logical capacity diagnostic");
            bounded=true;break;
        }
        escaped.push_back({receipt.source,receipt.runtimeInputs.compositionFrame,receipt.evaluation.deformation,receipt.evaluation.sceneAnimation});
    }
    if (!bounded) return refuse("production escaped composition receipts exceeded their finite logical pool");
    escaped.clear();
    if (!execution.executePreview(request,changed,error)) return false;
    ++request.helperGeneration;
    ImportedAnimatedSceneReceipt generationChanged;
    if (!execution.executePreview(request,generationChanged,error) || generationChanged.runtimeInputs.previousMotion
        || generationChanged.runtimeInputs.compositionFrame->owner()->helperGeneration()!=13
        || arbitgpu::prepareNativeDeformationFrame(*generationChanged.source,*current.evaluation.deformation,typed,data,error))
        return refuse("production helper generation change retained a foreign owner or predecessor");
    request.proceduralScene.reset();request.evaluationRevision=0;
    setFrame(15);
    ImportedAnimatedSceneReceipt legacy;
    if (!execution.executePreview(request,legacy,error) || legacy.source->compositionOwner
        || legacy.runtimeInputs.compositionFrame || !verifyColor(legacy,15))
        return refuse("production disabled composition changed the legacy source or draw");
    error.clear();return true;
}

// All native owners/frames below are issued by the normal payload executor.
// This authors original GLB bytes, including two distinct skins and real morph
// channels. The first adapted transform is nonroot node 0, before root node 1.
inline std::vector<std::uint8_t> parentedProductionGlb()
{
    const auto original=compositionGlb(false,false,CompositionSkinCase::Node);
    const auto read=[](const auto* p){return std::uint32_t(p[0])|std::uint32_t(p[1])<<8
        |std::uint32_t(p[2])<<16|std::uint32_t(p[3])<<24;};
    const auto jsonBytes=read(original.data()+12),binStart=28u+jsonBytes;
    auto root=nlohmann::json::parse(original.begin()+20,original.begin()+20+jsonBytes);
    root["nodes"][0]["scale"]={.2,.2,.2};
    root["nodes"].push_back({{"name","Animated original ancestor"},{"children",{0}},{"camera",0}});
    root["cameras"]=nlohmann::json::array({{{"type","perspective"},
        {"perspective",{{"yfov",.9},{"znear",.1},{"zfar",20.0}}}}});
    root["meshes"].push_back(root["meshes"][0]);
    root["nodes"].push_back({{"name","Second genuine skin"},{"mesh",1},{"skin",1},{"scale",{.2,.2,.2}}});
    root["nodes"].push_back({{"name","Second original root"},{"children",{2}},{"translation",{0.0,.8,0.0}}});
    root["skins"].push_back({{"joints",{2}}});
    root["scenes"][0]["nodes"]={1,3};
    root["animations"][0]["channels"].push_back({{"sampler",0},{"target",{{"node",1},{"path","translation"}}}});
    root["animations"][0]["channels"].push_back({{"sampler",0},{"target",{{"node",2},{"path","translation"}}}});
    root["animations"][0]["channels"].push_back({{"sampler",1},{"target",{{"node",2},{"path","weights"}}}});
    auto json=root.dump();while(json.size()%4)json+=' ';
    const auto binBytes=read(original.data()+binStart-8);
    std::vector<std::uint8_t> bytes;
    const auto append=[&](std::uint32_t value){for(unsigned shift=0;shift<32;shift+=8)bytes.push_back(std::uint8_t(value>>shift));};
    for(auto value:{0x46546c67u,2u,std::uint32_t(28+json.size()+binBytes),std::uint32_t(json.size()),0x4e4f534au})append(value);
    bytes.insert(bytes.end(),json.begin(),json.end());append(binBytes);append(0x004e4942u);
    bytes.insert(bytes.end(),original.begin()+binStart,original.begin()+binStart+binBytes);return bytes;
}

// Literal current-minus-prior projection, using raw typed depth only to locate
// the current visible surface. Neither native palette nor issued motion enters
// this oracle. R32Uint Object ID selects the known authored displacement.
inline bool parentedProductionMotion(const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame,
    const HarmonicMIDI::grid::SceneCameraRecord& camera,unsigned direction,unsigned mask,bool neutral,std::string& error,
    std::string* diagnostic=nullptr)
{
    const auto fail=[&](std::string text) {
        if (diagnostic) *diagnostic=std::move(text);
        return false;
    };
    arbitgpu::NativeRawPassPixels motion,depth,coverage,ids;
    if (!frame || !frame->readRawPass(renderpassoutput::Output::Motion,motion,error)
        || !frame->readRawPass(renderpassoutput::Output::Depth,depth,error)
        || !frame->readRawPass(renderpassoutput::Output::Mask,coverage,error)
        || !frame->readRawPass(renderpassoutput::Output::ObjectId,ids,error)) return fail("raw pass readback failed");
    const auto pixels=std::size_t(frame->width())*frame->height();
    if (motion.format!=arbitgpu::NativeTexturePixelFormat::Rg16Float
        || depth.format!=arbitgpu::NativeTexturePixelFormat::R32Float
        || coverage.format!=arbitgpu::NativeTexturePixelFormat::R8Unorm
        || ids.format!=arbitgpu::NativeTexturePixelFormat::R32Uint
        || motion.bytes.size()!=pixels*4 || depth.bytes.size()!=pixels*4
        || ids.bytes.size()!=pixels*4 || coverage.bytes.size()!=pixels)
        return fail("raw pass formats or byte sizes differ from RG16F/R32F/R8/R32U storage for "+std::to_string(pixels)+" pixels");
    const auto half=[](std::uint16_t bits) {
        const auto exponent=(bits>>10)&31,fraction=bits&1023;
        const double value=exponent==0?std::ldexp(double(fraction),-24)
            :exponent==31?std::numeric_limits<double>::infinity()
            :std::ldexp(double(1024+fraction),int(exponent)-25);
        return bits&32768?-value:value;
    };
    const auto num=[](double value) {
        char text[64];
        std::snprintf(text,sizeof text,"%.9g",value);
        return std::string(text);
    };
    std::array<std::size_t,2> imported {};std::array<std::size_t,4> procedural {};
    for(std::size_t p=0;p<pixels;++p)
    {
        std::uint16_t x=0,y=0;std::uint32_t id=0;float z=0;
        std::memcpy(&x,motion.bytes.data()+4*p,2);std::memcpy(&y,motion.bytes.data()+4*p+2,2);
        std::memcpy(&id,ids.bytes.data()+4*p,4);std::memcpy(&z,depth.bytes.data()+4*p,4);
        if (neutral || !coverage.bytes[p])
        {if (half(x)!=0 || half(y)!=0)return fail("pixel "+std::to_string(p)+" moved in a neutral or uncovered pixel: motion ("+num(half(x))+","+num(half(y))+") object "+std::to_string(id)+" coverage "+std::to_string(unsigned(coverage.bytes[p]))+" depth "+num(z));continue;}
        bool moves=id<0x80000000u;
        if (moves)
        {
            if(id!=1 && id!=131073)return fail("covered imported object id "+std::to_string(id)+" is not an admitted root");
            ++imported[id==1?0:1];
        }
        else
        {
            const auto authored=id&0x7fffffffu;
            const unsigned group=authored==707 || authored==505 || authored==606?1:authored==808?2:authored==909?3:0;
            if(!group)return fail("covered procedural authored id "+std::to_string(authored)+" has no group");
            const unsigned member=authored==707?0:authored==606?1:authored==808?2:authored==909?3:4;
            if(member==4)return fail("covered procedural authored id "+std::to_string(authored)+" has no member");
            ++procedural[member];
            moves=direction==2 && (mask&(1u<<group));
        }
        const double distance=camera.nearPlane+double(z)*(camera.farPlane-camera.nearPlane);
        const double projection=frame->height()/(2*std::tan(double(camera.verticalFovRadians)/2)*distance);
        const double dx=moves && direction==2?2.0/30:0;
        const double dy=moves && direction==1?.7*2.0/30:0;
        const double expectedX=dx*projection,expectedY=dy*projection;
        // Binary16 rounding envelope plus the established native raster error.
        const double tolerance=.005+std::max(std::abs(expectedX),std::abs(expectedY))/1024;
        if (!std::isfinite(half(x)) || !std::isfinite(half(y))
            || std::abs(half(x)-expectedX)>tolerance || std::abs(half(y)-expectedY)>tolerance)
            return fail("first mismatching pixel "+std::to_string(p)+" actual motion ("+num(half(x))+","+num(half(y))
                +") expected motion ("+num(expectedX)+","+num(expectedY)+") tolerance "+num(tolerance)
                +(moves?", moving":", not moving"));
    }
    if (neutral) return true;
    if (!(imported[0] && imported[1] && procedural[1] && procedural[2] && procedural[3]
        && (direction==1 || procedural[0])))
        return fail("moving coverage counts imported "+std::to_string(imported[0])+","+std::to_string(imported[1])
            +" procedural "+std::to_string(procedural[0])+","+std::to_string(procedural[1])+","
            +std::to_string(procedural[2])+","+std::to_string(procedural[3])+" direction "+std::to_string(direction));
    return true;
}

template <typename ReadColor>
bool verifyParentedImportedProductionJoin(arbitgpu::NativeDeformationBackend& backend,ReadColor readColor,std::string& error)
{
    using namespace modelpayload;
    using namespace HarmonicMIDI::grid;
    const auto refuse=[&](const char* message){error=message;return false;};
    const auto bytes=parentedProductionGlb();
    visualanimationimport::ExactContentAssetKey key {"parented-original-skins",1,
        sha256Bytes(bytes.data(),bytes.size()),"model/gltf-binary",bytes.size()};
    Store store;
    if (!store.begin(key.id,key) || !store.appendBase64(key.id,0,productionCompositionBase64(bytes)) || !store.commit(key.id))
        return refuse("parented production exact GLB transfer failed");
    // Ordinary decoder/adapters create only sources, never native capabilities.
    auto options=nativeImportedAnimationDecodeOptions(bytes.size(),0);
    const auto document=gltf::decodeAnimatedGlbBaseScene(bytes,options.admission,error);
    if (!document)return false;
    auto imported=gltf::adaptAnimatedGlbMeshToVisual3DSceneHeap(*document,error,{},true);
    if (!imported) return refuse("parented fixture adaptation returned no visual scene");
    // Production's animated admit path calls this same multi-draw adapter. It
    // keeps one draw per mesh node, with the node's world TRS baked in and its
    // parent word cleared. Draw 1 is node 0, the nonroot anchor, and draw
    // 131073 is node 2.
    const auto worldTranslationMatches=[](const auto& translation, const auto& world) {
        return std::abs(translation.x-world[12])<=1e-6f && std::abs(translation.y-world[13])<=1e-6f
            && std::abs(translation.z-world[14])<=1e-6f;
    };
    if (document->nodes.size()<3 || imported->objectCount!=2 || imported->objects[0].id.value!=1
        || imported->objects[1].id.value!=131073 || imported->objects[0].parent.isValid()
        || imported->objects[1].parent.isValid()
        || !worldTranslationMatches(imported->objects[0].transform.translation,document->nodes[0].worldTransform)
        || !worldTranslationMatches(imported->objects[1].transform.translation,document->nodes[2].worldTransform))
    {
        std::string detail="parented fixture adapted draws differ from production multi-draw anchors (objectCount="
            +std::to_string(imported->objectCount)+", objects[0].id="+std::to_string(imported->objects[0].id.value)
            +", objects[0].parent="+std::to_string(imported->objects[0].parent.value)+"; first objects:";
        for (std::size_t i=0;i<std::min<std::size_t>(imported->objectCount,6);++i)
            detail+=" ["+std::to_string(i)+"] id="+std::to_string(imported->objects[i].id.value)
                +" parent="+std::to_string(imported->objects[i].parent.value)+";";
        error=detail;
        return false;
    }
    for (const unsigned direction:{1u,2u})
    for (const unsigned mask:{2u,4u,8u,14u})
    {
        auto p=std::make_shared<Visual3DScene>(makeScene());p->id={700};
        p->objects[0].id={707};p->objects[0].parent={505};
        p->objects[0].transform.translation={.4f,0,0};p->objects[0].transform.scale={.7f,.7f,.7f};
        // Zero-geometry original parent is stored after its child.
        p->objects[1]={};p->objects[1].id={505};p->objects[1].transform.translation={-.3f,0,0};
        p->objects[1].transform.rotation={0,0,.70710678f,.70710678f};p->objectCount=2;
        if (direction==2)
        {
            p->objects[0].transform.scale={.4f,.4f,.4f};p->objects[1].transform.rotation={};
            p->objects[1].transform.translation={-1,-2.5f,0};
        }
        else
        {
            // The deterministic nonroot anchor is a transform record. A large
            // mesh at this same placement would occlude its imported children.
            p->objects[0].vertexCount=0;p->objects[0].indexCount=0;
            p->vertexCount=0;p->indexCount=0;
        }
        auto cube=std::make_unique<Visual3DScene>(makeScene());
        for (const auto id:{606u,808u,909u})
        {
            cube->objects[0].id={id};cube->objects[0].transform.scale={.18f,.18f,.18f};
            // All members stay visible before and after the imported .2 scale;
            // flagged procedural cubes must not hide inside the imported skin.
            cube->objects[0].transform.translation={id==606?-1.9f:id==808?1.5f:2.4f,-2,0};
            if(!scenecomposition::appendScene(*cube,*p,false,error))return false;
        }
        p->cameras[0].transform={};p->cameras[0].transform.translation={0,0,6};
        p->cameras[0].verticalFovRadians=.9f;p->cameras[0].nearPlane=.1f;p->cameras[0].farPlane=20;
        if(!validateVisual3DScene(*p).valid())return refuse("parented procedural fixture is invalid");
        visualimportedscenerender::ImportedParentIntentV1 intent;
        intent.composeStableId=700;intent.importedSourceStableId=801;intent.connectedInputMask=15;
        intent.requestedParentMask=mask;intent.direction=direction;
        intent.importedInputIndex=direction==2?0:mask==4?2:mask==8?3:1;
        if(direction==1)intent.proceduralAnchorId=707;
        else {intent.rootMembershipCount=4;intent.rootMembership={{{505,1},{606,1},{808,2},{909,3}}};}
        // In direction 1 the other procedural flags are already resolved by the
        // graph. Preserve those original internal parent words in the helper.
        if(direction==1 && mask==14)
            for(std::size_t x=2;x<p->objectCount;++x)p->objects[x].parent={707};
        visualimportedscenerender::Request leaf;
        leaf.sourceStableId=801;leaf.renderStableId=802;leaf.asset=key;leaf.sceneIndex=0;
        leaf.structuralRevision=31;leaf.evaluationRevision=35;leaf.deformation.emplace();
        auto& a=*leaf.deformation;a.asset=key;a.sourceStableId=801;a.deformationStableId=803;a.schedule={801,803};
        a.sceneIndex=0;a.animationClipStableId=1;a.clipName="Original composition action";
        visualimportedscenerender::Request authored;
        authored.sourceStableId=700;authored.renderStableId=802;authored.structuralRevision=31;authored.evaluationRevision=35;
        authored.sceneSnapshot=p;authored.importedScene=std::make_shared<const visualimportedscenerender::Request>(leaf);
        authored.importedParentIntent=intent;authored.rawExportMask=render3dimage::kRawExportMask;
        const auto wire=visualimportedscenerender::encode(authored);
        visualimportedscenerender::Request decoded;
        if(wire.empty() || !visualimportedscenerender::decodeCanonical(wire,decoded,error)
            || !visualimportedscenerender::sameRequest(authored,decoded))return false;
        ImportedScenePayloadExecution execution(store,arbitgpu::nativeFixtureSceneBackend(),backend);
        videowire::CompiledVisualLayerPlan plan;
        plan.clipId=42;plan.structuralRevision=31;plan.producerValidated=true;
        plan.nodeIds={699,801};plan.nodeKinds={"visual.3d.scene.retained","visual.3d.render"};
        plan.operations={{699,plan.nodeKinds[0],"control-eval",""},{801,plan.nodeKinds[1],"native-gpu",wire}};
        plan.edges={{699,0,801,0}};
        plan.ports={{699,0,1,"out","control","scene3D","unspecified","unspecified"},
            {801,0,1,"in","control","scene3D","unspecified","unspecified"},{801,1,1,"out","frame","image","rgba8","sRGB"}};
        importedscene::VisualImportedScenePlanCache cache;
        videorender::LayerDesc layer;std::vector<ImportedSceneExecutionReceipt> owners;
        const auto prepare=[&](std::int64_t frame,bool exporting,ImportedSceneExecutionReceipt& out) {
            if(importedscene::prepareVisualImportedSceneLayerAtTime({plan},42,96,96,double(frame)/30,30,
                exporting?importedscene::NativeImportedSceneRenderUse::Export:importedscene::NativeImportedSceneRenderUse::Preview,
                &execution,&cache,layer,owners,error,nullptr,{11,12})!=importedscene::VisualImportedScenePreparation::rendered
                || owners.empty())return false;
            out=owners.back();return true;
        };
        ImportedSceneExecutionReceipt first,current,exported,next;
        if(!prepare(0,false,first) || !prepare(1,false,current) || !prepare(1,true,exported))return false;
        if(!current.deformationFrame || !current.deformationFrame->valid() || !current.deformationFrame->evaluation.deformation
            || !current.deformationFrame->evaluation.sceneAnimation
            || current.deformationFrame->evaluation.deformation->jointTransforms().empty()
            || current.deformationFrame->evaluation.deformation->morphWeights().empty())
            return refuse("parented production omitted genuine skin, morph or ancestor sample");
        const auto& r=*current.deformationFrame;const auto& scene=*r.source->scene;
        if(!r.runtimeInputs.previousMotion || r.runtimeInputs.previousMotion->compositionFrame!=first.deformationFrame->runtimeInputs.compositionFrame
            || exported.deformationFrame->runtimeInputs.previousMotion)return refuse("parented production borrowed or omitted native history");
        // Independent full parent-array oracle, including all unflagged roots.
        auto basis=gltf::composeImportedVisual3DScene(*p,*imported,error);if(!basis)return false;
        auto reference=std::make_shared<Visual3DScene>(*basis);
        for(std::size_t x=0;x<reference->objectCount;++x)
        {
            auto& o=reference->objects[x];
            if(direction==1 && !o.parent.isValid() && o.id.value<0x80000000u)o.parent={0x80000000u|707u};
            if(direction==2 && !o.parent.isValid() && o.id.value>=0x80000000u)
            {
                const auto id=o.id.value&0x7fffffffu;const auto group=id==505||id==606?1:id==808?2:3;
                // The anchor is imported objects[0], the node 0 draw checked above.
                if(mask&(1u<<group))o.parent={1u};
            }
            if(o.parent!=scene.objects[x].parent || !scenecomposition::sameTransform(o.transform,scene.objects[x].transform))
                return refuse("parented production changed an internal/unflagged parent or original local TRS");
        }
        const auto get=[&](std::uint32_t id)->const SceneObjectRecord* {
            return visual3d_detail::findById(scene.objects,scene.objectCount,SceneObjectId{id});
        };
        // Production's multi-draw imported roots are draws 1 and 131073, not transform records.
        if(direction==1 && (!get(1) || !get(131073)
            || get(1)->parent.value!=(0x80000000u|707u) || get(131073)->parent.value!=(0x80000000u|707u)))
            return refuse("parented production omitted one original imported root");
        // A native CPU prepare remains a consumer of executor-issued authority.
        // Procedural post translations have an independent literal TRS oracle.
        for(const auto& draw:r.source->draws)
        {
            arbitgpu::NativeDeformationFrameData data;
            if(!arbitgpu::prepareNativeDeformationFrame(*draw,*r.evaluation.deformation,r.runtimeInputs.compositionFrame,data,error))return false;
            if(draw->object.value==(0x80000000u|707u) && direction==2)
            {
                // Static world scale is .2*.4=.08. The native post acts
                // in that static object's coordinates, before raster placement.
                const double expected=(mask&2)?(2.0/30)/.08:0;
                if(std::abs(data.compositionPost.point()[12]-expected)>1.e-5)
                    return refuse("parented procedural world disagrees with literal local TRS and imported motion");
            }
        }
        // Actual raw passes are obtained on Export as well as Preview. Preview
        // requests Color by default; direct payload execution below admits raw
        // attachments while preserving the ordinary compiled handoff above.
        ImportedSceneRequest request;
        const auto* cached=cache.resolve(plan,error);
        if(!cached || !cached->importedSceneRender)return false;
        const auto& expectedRequest=*cached->importedSceneRender;
        request.importedScene=expectedRequest.importedScene;request.asset=key;request.sceneIndex=0;
        request.deformation=expectedRequest.importedScene->deformation;request.sceneSnapshot=expectedRequest.sceneSnapshot;
        request.parentIntent=&*expectedRequest.importedParentIntent;request.structuralRevision=31;request.evaluationRevision=35;
        request.projectGeneration=11;request.helperGeneration=12;request.clipId=42;request.dimensions={96,96};
        request.staticPayloadIdentity=wire;request.runtimeInputs.rawExportMask=render3dimage::kRawExportMask;
        request.frame={1,30,1};request.deformation->playback.timelineSeconds=1.0/30;
        if(!execution.inspectParentReceipt(current,request,ImportedSceneUse::Preview))
        {
            // A different raw-output request must not borrow the Color receipt.
            request.runtimeInputs.rawExportMask=0;
            if(!execution.inspectParentReceipt(current,request,ImportedSceneUse::Preview))return false;
            request.runtimeInputs.rawExportMask=render3dimage::kRawExportMask;
        }
        // Equal public fields without the private association remain refused.
        ImportedSceneExecutionReceipt forged;
        forged.use=current.use;forged.payload=current.payload;forged.deformationFrame.emplace();
        auto& f=*forged.deformationFrame;f.owner=r.owner;f.payload=r.payload;f.source=r.source;f.evaluation=r.evaluation;
        f.frame=r.frame;f.runtimeInputs=r.runtimeInputs;
        if(execution.inspectParentReceipt(forged,request,ImportedSceneUse::Preview))return refuse("public receipt fields minted a parent association");
        ImportedSceneExecutionReceipt rawFirst,rawCurrent,rawExport;
        request.frame={0,30,1};request.deformation->playback.timelineSeconds=0;
        if(!execution.executePreview(request,rawFirst,error))return false;
        request.frame.frame=1;request.deformation->playback.timelineSeconds=1.0/30;
        if(!execution.executePreview(request,rawCurrent,error) || !execution.executeExport(request,rawExport,error))return false;
        // Each pass is checked in the same order as before; only the diagnostic is per pass.
        std::string motionDetail;
        const auto motionRefuse=[&](const char* pass) {
            std::cerr << "parented production raw Motion (" << pass << "): " << motionDetail << '\n';
            return refuse("parented production raw half-float Motion disagrees with independent projection");
        };
        if(!parentedProductionMotion(rawFirst.deformationFrame->frame.nativeFrame,p->cameras[0],direction,mask,true,error,&motionDetail))
            return motionRefuse("neutral first");
        if(!parentedProductionMotion(rawCurrent.deformationFrame->frame.nativeFrame,p->cameras[0],direction,mask,false,error,&motionDetail))
            return motionRefuse("moving current");
        if(!parentedProductionMotion(rawExport.deformationFrame->frame.nativeFrame,p->cameras[0],direction,mask,true,error,&motionDetail))
            return motionRefuse("neutral export");
        // Literal original 0 -> 2 translation moves both genuine skins. The
        // production draws carry each joint-covered node's world TRS, so the
        // reference moves draws 1 and 131073. Root 1's separately authenticated
        // scene channel must not be applied a second time to the node 0 world.
        for(const auto id:{1u,131073u})
        {
            bool found=false;
            for(std::size_t x=0;x<reference->objectCount;++x)
                if(reference->objects[x].id.value==id){reference->objects[x].transform.translation.x=2.0f/30;found=true;}
            if(!found)return refuse("parented reference lost a joint-covered node draw");
        }
        auto& staticBackend=arbitgpu::nativeFixtureSceneBackend();
        const auto prepared=staticBackend.prepare(reference,{});if(!prepared.prepared){error=prepared.error;return false;}
        const auto referenceFrame=staticBackend.render(reference,prepared.resources,96,96,{});
        if(!referenceFrame.rendered){error=referenceFrame.error;return false;}
        const auto actualColor=readColor(rawCurrent.deformationFrame->frame.nativeFrame);
        const auto expectedColor=readColor(referenceFrame.frame);
        if(actualColor.empty() || actualColor.size()!=expectedColor.size()
            || actualColor!=readColor(rawExport.deformationFrame->frame.nativeFrame))
            return refuse("parented production native Color lost Preview/Export parity");
        for(std::size_t x=0;x<actualColor.size();++x)
            if(std::abs(int(actualColor[x])-int(expectedColor[x]))>6)
                return refuse("parented production Color disagrees with independently authored transformed geometry");
        if(!prepare(2,true,next) || !next.deformationFrame->runtimeInputs.previousMotion)
            return refuse("parented production Export omitted its own adjacent predecessor");
        ImportedSceneExecutionReceipt matchedPreview,matchedExport,repeat,seek;
        request.frame.frame=2;request.deformation->playback.timelineSeconds=2.0/30;
        if(!execution.executePreview(request,matchedPreview,error) || !execution.executeExport(request,matchedExport,error)
            || !execution.executePreview(request,repeat,error)
            || !parentedProductionMotion(matchedPreview.deformationFrame->frame.nativeFrame,p->cameras[0],direction,mask,false,error)
            || !parentedProductionMotion(matchedExport.deformationFrame->frame.nativeFrame,p->cameras[0],direction,mask,false,error)
            || !parentedProductionMotion(repeat.deformationFrame->frame.nativeFrame,p->cameras[0],direction,mask,false,error)
            || readColor(matchedPreview.deformationFrame->frame.nativeFrame)!=readColor(matchedExport.deformationFrame->frame.nativeFrame)
            || readColor(repeat.deformationFrame->frame.nativeFrame)!=readColor(matchedPreview.deformationFrame->frame.nativeFrame))
            return refuse("parented matched Preview/Export or same-frame redraw lost its own current/prior geometry");
        request.frame.frame=9;request.deformation->playback.timelineSeconds=9.0/30;
        if(!execution.executePreview(request,seek,error) || seek.deformationFrame->runtimeInputs.previousMotion
            || !parentedProductionMotion(seek.deformationFrame->frame.nativeFrame,p->cameras[0],direction,mask,true,error))
            return refuse("parented nonadjacent seek borrowed a preceding native frame");
        request.frame.frame=1;request.deformation->playback.timelineSeconds=1.0/30;
        auto altered=current;altered.deformationFrame->runtimeInputs.compositionFrame.reset();
        if(altered.valid())return refuse("parent receipt accepted removal of its public typed frame");
        altered=current;altered.deformationFrame->source=exported.deformationFrame->source;
        altered.deformationFrame->frame.source=altered.deformationFrame->source;
        if(altered.valid() && altered.deformationFrame->source!=r.source)return refuse("parent receipt accepted foreign source");
        altered=current;altered.deformationFrame->frame.nativeFrame=exported.deformationFrame->frame.nativeFrame;
        if(altered.valid())return refuse("parent receipt accepted foreign completed native frame");
        altered=current;altered.deformationFrame->sourceCameraNodeId=1;
        if(altered.valid())return refuse("parent receipt accepted a foreign camera selector");
        altered=current;altered.deformationFrame->runtimeInputs.objectTranslationOffset[0]+=.1f;
        if(altered.valid())return refuse("parent receipt accepted mutated current runtime placement");
        if (!rawCurrent.deformationFrame)return refuse("parented incumbent omitted its deformation receipt");
        const auto keep=rawCurrent.deformationFrame->frame.nativeFrame;
        const auto keepSource=rawCurrent.deformationFrame->source;
        const auto keepComposition=rawCurrent.deformationFrame->runtimeInputs.compositionFrame;
        const auto keepDeformation=rawCurrent.deformationFrame->evaluation.deformation;
        const auto keepPayload=rawCurrent.payload;
        const auto sameOwner=[](const auto& a,const auto& b) {
            return a==b && !a.owner_before(b) && !b.owner_before(a);
        };
        const auto incumbentUnchanged=[&] {
            return rawCurrent.deformationFrame && sameOwner(rawCurrent.payload,keepPayload)
                && sameOwner(rawCurrent.deformationFrame->payload,keepPayload)
                && sameOwner(rawCurrent.deformationFrame->source,keepSource)
                && sameOwner(rawCurrent.deformationFrame->frame.source,keepSource)
                && sameOwner(rawCurrent.deformationFrame->runtimeInputs.compositionFrame,keepComposition)
                && sameOwner(rawCurrent.deformationFrame->evaluation.deformation,keepDeformation)
                && sameOwner(rawCurrent.deformationFrame->frame.nativeFrame,keep);
        };
        auto badIntent=intent;
        if(direction==2){--badIntent.rootMembershipCount;badIntent.rootMembership[3]={};}
        else badIntent.proceduralAnchorId=999;
        request.parentIntent=&badIntent;
        if(execution.executePreview(request,rawCurrent,error) || !incumbentUnchanged()
            || execution.executeExport(request,rawCurrent,error) || !incumbentUnchanged())
            return refuse("missing parent root changed the incumbent native output");
        request.parentIntent=&*expectedRequest.importedParentIntent;
        // Missing or contradictory outer companions do not replace the genuine
        // unchanged retained leaf. Both public boundaries conserve every owner.
        for (unsigned malformed=0;malformed<5;++malformed)
        {
            auto companion=request;
            if (malformed==0)companion.deformation.reset();
            if (malformed==1)companion.deformation->sourceStableId=0;
            if (malformed==2)companion.deformation->deformationStableId=0;
            if (malformed==3)++companion.deformation->sourceStableId;
            if (malformed==4)++companion.deformation->deformationStableId;
            if (execution.executePreview(companion,rawCurrent,error) || !incumbentUnchanged()
                || execution.executeExport(companion,rawCurrent,error) || !incumbentUnchanged()
                || readColor(rawCurrent.deformationFrame->frame.nativeFrame)!=actualColor)
                return refuse("malformed outer deformation companion retired its unchanged genuine leaf");
        }
        auto changed=intent;changed.requestedParentMask=mask==14?2:14;request.parentIntent=&changed;
        if(execution.inspectParentReceipt(current,request,ImportedSceneUse::Preview))return refuse("expected-request inspector borrowed a different parent mask");
        request.parentIntent=&*expectedRequest.importedParentIntent;request.evaluationRevision=34;
        if(execution.executePreview(request,rawCurrent,error) || !incumbentUnchanged())
            return refuse("parented stale epoch replaced an incumbent");
        request.evaluationRevision=35;
        if(mask==2)
        {
            const auto rejects=[&](const auto& bad) {
                request.parentIntent=&bad;
                const bool rejected=!execution.executePreview(request,rawCurrent,error)
                    && incumbentUnchanged();
                request.parentIntent=&*expectedRequest.importedParentIntent;
                return rejected;
            };
            if(direction==1)
            {
                auto alternate=intent;alternate.proceduralAnchorId=505;
                if(!rejects(alternate))return refuse("parent issuer replaced the first stored nonroot anchor with a later root");
            }
            else
            {
                auto sibling=intent;sibling.rootMembershipCount=3;
                sibling.rootMembership={{{505,1},{808,2},{909,3}}};
                auto nonroot=intent;nonroot.rootMembership[0]={606,1};nonroot.rootMembership[1]={707,1};
                auto group=intent;group.rootMembershipCount=3;
                group.rootMembership={{{505,1},{606,1},{909,3}}};
                auto extra=intent;extra.rootMembershipCount=5;extra.rootMembership[4]={999,3};
                if(!rejects(sibling) || !rejects(nonroot) || !rejects(group) || !rejects(extra))
                    return refuse("parent issuer accepted a missing sibling/group, nonroot or foreign root in the complete union");
            }
            // A separately executed epoch supplies actual foreign source/prior
            // authorities. Equal public fields cannot replace the sealed join.
            ImportedAnimatedScenePayloadExecution independent(store,backend);
            ImportedAnimatedSceneRequest foreign;
            foreign.operation=*request.deformation;foreign.parentIntent=request.parentIntent;
            foreign.proceduralScene=request.sceneSnapshot;foreign.frame={1,30,1};
            foreign.structuralRevision=31;foreign.evaluationRevision=36;foreign.projectGeneration=11;foreign.helperGeneration=12;
            foreign.renderClipId=42;foreign.width=96;foreign.height=96;
            ImportedAnimatedSceneReceipt issued;
            if(!independent.executePreview(foreign,issued,error))return false;
            altered=current;altered.deformationFrame->source=issued.source;altered.deformationFrame->frame.source=issued.source;
            if(altered.valid())return refuse("parent receipt accepted an independently issued foreign source epoch");
            altered=current;altered.deformationFrame->runtimeInputs.previousMotion.reset();
            if(altered.valid())return refuse("parent receipt accepted removal of its genuine adjacent predecessor");
            auto foreignOperation=*request.deformation;foreignOperation.animationClipStableId=2;
            request.deformation=foreignOperation;
            if(execution.inspectParentReceipt(current,request,ImportedSceneUse::Preview))
                return refuse("parent inspector accepted a different imported animation selection");
            request.deformation=expectedRequest.importedScene->deformation;request.deformation->playback.timelineSeconds=1.0/30;
            // Source camera 2 is genuine and animated through original root 1.
            // The explicit authored procedural override remains authoritative.
            foreign.evaluationRevision=35;foreign.sourceCameraNodeId=2;
            foreign.runtimeInputs.cameraOverride=p->cameras[0];
            independent.reset();
            if(!independent.executePreview(foreign,issued,error) || !issued.valid()
                || !issued.runtimeInputs.animatedCamera || issued.runtimeInputs.animatedCamera->id.value!=2
                || std::abs(issued.runtimeInputs.animatedCamera->transform.translation.x-2.0f/30)>1.e-5
                || !independent.inspectParentReceipt(issued,foreign,ImportedAnimationEvaluationOwner::Preview)
                || readColor(issued.frame.nativeFrame)!=actualColor)
                return refuse("parented selected source camera or explicit override lost genuine native authority");
            auto cameraChanged=issued;cameraChanged.runtimeInputs.cameraOverride->transform.translation.x+=.1f;
            if(cameraChanged.valid())return refuse("parent selected-camera receipt accepted a foreign explicit override");
            foreign.frame={2,24000,1001};foreign.operation.playback.timelineSeconds=2.0*1001/24000;
            if(!independent.executePreview(foreign,issued,error) || issued.runtimeInputs.previousMotion)
                return refuse("parented rate change retained a predecessor from a different rational clock");
            foreign.frame.frame=3;foreign.operation.playback.timelineSeconds=3.0*1001/24000;
            if(!independent.executePreview(foreign,issued,error) || !issued.runtimeInputs.previousMotion)
                return refuse("parented rational clock omitted its own adjacent predecessor");
            foreign.width=80;++foreign.frame.frame;foreign.operation.playback.timelineSeconds=4.0*1001/24000;
            if(!independent.executePreview(foreign,issued,error) || issued.runtimeInputs.previousMotion)
                return refuse("parented extent change retained a predecessor from a different native image");
            independent.reset();++foreign.helperGeneration;++foreign.frame.frame;
            foreign.operation.playback.timelineSeconds=5.0*1001/24000;
            if(!independent.executePreview(foreign,issued,error) || issued.runtimeInputs.previousMotion)
                return refuse("parented reset/helper replacement retained a foreign native history");
            auto oversized=std::make_shared<Visual3DScene>(*request.sceneSnapshot);
            oversized->lights.back().environmentImage.asset.id.assign(NativeSceneAnimationCompositionBudget::kMaximumBytes,'i');
            const auto originalScene=request.sceneSnapshot;request.sceneSnapshot=oversized;
            if(execution.executePreview(request,rawCurrent,error) || !incumbentUnchanged()
                || error.find("capacity")==std::string::npos)
                return refuse("parent inactive light identity overflow changed an incumbent native frame");
            request.sceneSnapshot=originalScene;
            request.evaluationRevision=36;request.parentIntent=&badIntent;
            if(execution.executePreview(request,rawCurrent,error) || !incumbentUnchanged())
                return refuse("failed newer parent epoch replaced an incumbent native output");
            request.parentIntent=&*expectedRequest.importedParentIntent;request.evaluationRevision=35;
            if(execution.executePreview(request,rawCurrent,error) || !incumbentUnchanged())
                return refuse("failed newer parent attempt forgot its stale epoch floor");
        }
        if(direction==2 && mask==2)
        {
            // Genuine execution issues each alias. Reset never supplies a fresh
            // pool, and retaining only individual public owners keeps debits.
            ImportedAnimatedScenePayloadExecution escaping(store,backend);
            ImportedAnimatedSceneRequest moving;
            moving.operation=*request.deformation;moving.parentIntent=request.parentIntent;
            moving.proceduralScene=request.sceneSnapshot;moving.frame={1,30,1};
            moving.structuralRevision=31;moving.evaluationRevision=35;moving.projectGeneration=11;moving.helperGeneration=12;
            moving.renderClipId=42;moving.width=96;moving.height=96;
            ImportedAnimatedSceneReceipt retained;
            if(!escaping.executePreview(moving,retained,error))return false;
            struct Aliases final
            {
                visualmodelassetpayload::PayloadPtr payload;
                std::shared_ptr<const arbitgpu::NativeDeformationScene> source;
                std::shared_ptr<const arbitgpu::NativeDeformationCompositionOwner> owner;
                std::shared_ptr<const arbitgpu::NativeDeformationCompositionFrame> frame;
                std::shared_ptr<const Visual3DScene> scene;
                std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot> snapshot;
                std::shared_ptr<const visualanimation::Sample> sample;
                std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> native;
            };
            std::vector<Aliases> aliases;bool bounded=false;
            for(std::uint64_t epoch=36;epoch<100;++epoch)
            {
                aliases.push_back({retained.payload,retained.source,retained.source->compositionOwner,
                    retained.runtimeInputs.compositionFrame,retained.source->scene,retained.evaluation.deformation,
                    retained.evaluation.sceneAnimation,retained.frame.nativeFrame});
                escaping.reset();moving.evaluationRevision=epoch;
                const auto prior=retained.frame.nativeFrame;
                if(!escaping.executePreview(moving,retained,error))
                {
                    if(retained.frame.nativeFrame!=prior || error.find("logical CPU capacity")==std::string::npos)
                        return refuse("parent escaped aliases lost bounded storage or incumbent conservation");
                    bounded=true;break;
                }
            }
            if(!bounded)return refuse("parent escaping owner/frame/source/payload aliases exceeded finite storage");
            aliases.clear();escaping.reset();
            if(!escaping.executePreview(moving,retained,error))return false;
            bool exhausted=false;
            for(std::size_t attempt=0;attempt<8192;++attempt)
                if(!retained.valid()){exhausted=true;break;}
            if(!exhausted || retained.valid())return refuse("repeated parent inspectors reset or refunded their permanent work");
            const auto old=retained.frame.nativeFrame;escaping.reset();
            if(retained.valid() || escaping.executePreview(moving,retained,error) || retained.frame.nativeFrame!=old
                || escaping.executePreview(moving,retained,error) || retained.frame.nativeFrame!=old)
                return refuse("reset or same-frame retry replaced an exhausted parent work ledger");
        }
        if (mask==2)
        {
            // The public const request aliases a legitimate mutable owner.
            // Native capabilities still come only from ordinary payload execution.
            for (unsigned mutation=0;mutation<7;++mutation)
            {
                auto mutableSource=std::make_shared<Visual3DScene>(*p);
                ImportedScenePayloadExecution sourceExecution(store,arbitgpu::nativeFixtureSceneBackend(),backend);
                auto mutableRequest=request;mutableRequest.sceneSnapshot=mutableSource;
                mutableRequest.evaluationRevision=35;
                mutableRequest.parentIntent=&intent;
                ImportedSceneExecutionReceipt incumbent;
                if (!sourceExecution.executePreview(mutableRequest,incumbent,error) || !incumbent.deformationFrame
                    || !incumbent.valid() || !sourceExecution.inspectParentReceipt(incumbent,mutableRequest,ImportedSceneUse::Preview))
                    return refuse("mutable original procedural source failed genuine parent admission");
                const auto originalSource=incumbent.deformationFrame->source;
                const auto originalNative=incumbent.deformationFrame->frame.nativeFrame;
                const auto originalFrame=incumbent.deformationFrame->runtimeInputs.compositionFrame;
                const auto originalSnapshot=incumbent.deformationFrame->evaluation.deformation;
                const auto originalPayload=incumbent.payload;
                const auto originalColor=readColor(originalNative);
                if (originalColor.empty())return refuse("mutable-source incumbent had no native Color evidence");
                if (mutation==0)mutableSource->objects[0].transform.translation.x+=.01f;
                if (mutation==1)mutableSource->vertices[0].position.x+=.01f;
                if (mutation==2)mutableSource->materials[0].emissive.x+=.01f;
                if (mutation==3)mutableSource->textureTexels[0].red^=1;
                if (mutation==4)mutableSource->cameras[0].transform.translation.x+=.01f;
                if (mutation==5)mutableSource->lights[0].intensity+=.01f;
                if (mutation==6)mutableSource->lights.back().environmentImage.asset.id="inactive original source mutation";
                if (!validateVisual3DScene(*mutableSource).valid())
                    return refuse("mutable-source refusal fixture changed the scene's validity");
                if (incumbent.valid() || sourceExecution.inspectParentReceipt(incumbent,mutableRequest,ImportedSceneUse::Preview))
                    return refuse("parent current inspector ignored mutation of its legitimate original source");
                const auto unchanged=[&] {
                    return incumbent.deformationFrame && sameOwner(incumbent.payload,originalPayload)
                        && sameOwner(incumbent.deformationFrame->payload,originalPayload)
                        && sameOwner(incumbent.deformationFrame->source,originalSource)
                        && sameOwner(incumbent.deformationFrame->frame.source,originalSource)
                        && sameOwner(incumbent.deformationFrame->frame.nativeFrame,originalNative)
                        && sameOwner(incumbent.deformationFrame->runtimeInputs.compositionFrame,originalFrame)
                        && sameOwner(incumbent.deformationFrame->evaluation.deformation,originalSnapshot)
                        && readColor(incumbent.deformationFrame->frame.nativeFrame)==originalColor;
                };
                if (sourceExecution.executePreview(mutableRequest,incumbent,error) || !unchanged()
                    || sourceExecution.executeExport(mutableRequest,incumbent,error) || !unchanged())
                    return refuse("mutated original source was reused or changed its incumbent native output/owners");
                ++mutableRequest.evaluationRevision;
                if (sourceExecution.executePreview(mutableRequest,incumbent,error) || !unchanged())
                    return refuse("a newer epoch re-admitted a mutated original procedural alias");
            }
            // Actual generation retirement must clear rather than restore the
            // prior output, even when the new request then fails root validation.
            for (unsigned retirement=0;retirement<6;++retirement)
            {
                ImportedScenePayloadExecution retiring(store,arbitgpu::nativeFixtureSceneBackend(),backend);
                auto replacement=request;replacement.parentIntent=&intent;replacement.evaluationRevision=35;
                ImportedSceneExecutionReceipt old;
                if (!retiring.executePreview(replacement,old,error) || !old.deformationFrame)return false;
                auto retiredIntent=badIntent;
                replacement.parentIntent=&retiredIntent;
                if (retirement==0)++replacement.projectGeneration;
                if (retirement==1)++replacement.helperGeneration;
                if (retirement==2)replacement.staticPayloadIdentity+="retired exact payload";
                if (retirement==3)replacement.asset.contentSha256.assign(64,'0');
                if (retirement==4)++replacement.structuralRevision;
                if (retirement==5)
                {
                    // Change the actual authored leaf/source/schedule and its
                    // parent/static publication together. An outer-ID-only
                    // contradiction belongs to the conservation cases above.
                    auto retiredLeaf=*replacement.importedScene;
                    retiredLeaf.sourceStableId=804;
                    retiredLeaf.deformation->sourceStableId=804;
                    retiredLeaf.deformation->deformationStableId=805;
                    retiredLeaf.deformation->schedule={804,805};
                    replacement.importedScene=std::make_shared<const visualimportedscenerender::Request>(retiredLeaf);
                    replacement.deformation=retiredLeaf.deformation;
                    retiredIntent.importedSourceStableId=804;
                    auto retiredAuthored=authored;
                    retiredAuthored.importedScene=replacement.importedScene;
                    retiredAuthored.importedParentIntent->importedSourceStableId=804;
                    replacement.staticPayloadIdentity=visualimportedscenerender::encode(retiredAuthored);
                    if (replacement.staticPayloadIdentity.empty())
                        return refuse("true source retirement fixture did not author a coherent static identity");
                }
                if (retiring.executePreview(replacement,old,error) || old.deformationFrame || old.payload)
                    return refuse("retired parent source generation resurrected its prior native output");
            }
            for (const bool exporting:{false,true})
            for (unsigned refusal=0;refusal<2;++refusal)
            for (unsigned generation=0;generation<2;++generation)
            {
                ImportedScenePayloadExecution retiring(store,arbitgpu::nativeFixtureSceneBackend(),backend);
                auto replacement=request;replacement.parentIntent=&intent;replacement.evaluationRevision=35;
                ImportedSceneExecutionReceipt old;
                const auto execute=[&](const auto& incoming) {
                    return exporting?retiring.executeExport(incoming,old,error):retiring.executePreview(incoming,old,error);
                };
                if (!execute(replacement) || !old.deformationFrame)return false;
                const auto escaped=old;
                const auto native=old.deformationFrame->frame.nativeFrame;
                const auto color=readColor(native);
                const auto source=old.deformationFrame->source;
                const auto payload=old.payload;
                const auto composition=old.deformationFrame->runtimeInputs.compositionFrame;
                const auto deformation=old.deformationFrame->evaluation.deformation;
                if (color.empty())return false;
                if (refusal==0)
                {
                    bool exhausted=false;
                    for (std::size_t attempt=0;attempt<8192;++attempt)
                        if (!old.valid()){exhausted=true;break;}
                    if (!exhausted || old.valid())return refuse("public parent receipt inspector never exhausted permanent work");
                    if (execute(replacement) || !old.deformationFrame
                        || !sameOwner(old.payload,payload) || !sameOwner(old.deformationFrame->source,source)
                        || !sameOwner(old.deformationFrame->frame.nativeFrame,native)
                        || !sameOwner(old.deformationFrame->runtimeInputs.compositionFrame,composition)
                        || !sameOwner(old.deformationFrame->evaluation.deformation,deformation)
                        || readColor(native)!=color)
                        return refuse("same-generation exhausted public retry changed its incumbent");
                }
                else
                {
                    replacement.staticPayloadIdentity.reserve(NativeSceneAnimationCompositionBudget::kMaximumBytes);
                    if (execute(replacement) || !old.deformationFrame
                        || !sameOwner(old.payload,payload) || !sameOwner(old.deformationFrame->source,source)
                        || !sameOwner(old.deformationFrame->frame.nativeFrame,native)
                        || !sameOwner(old.deformationFrame->runtimeInputs.compositionFrame,composition)
                        || !sameOwner(old.deformationFrame->evaluation.deformation,deformation)
                        || readColor(native)!=color)
                        return refuse("same-generation early capacity refusal changed its incumbent");
                }
                if (generation==0)++replacement.projectGeneration;
                else ++replacement.helperGeneration;
                if (execute(replacement) || old.deformationFrame || old.payload
                    || !sameOwner(escaped.deformationFrame->frame.nativeFrame,native) || readColor(native)!=color)
                    return refuse("fixed positive retirement was bypassed by exhausted work or early capacity refusal");
                if (refusal==0)
                {
                    replacement.staticPayloadIdentity=wire;
                    replacement.parentIntent=&intent;
                    if (execute(replacement) || old.deformationFrame || escaped.valid())
                        return refuse("retirement replaced the permanently exhausted public ledger");
                }
            }
            {
                ImportedScenePayloadExecution replacementExecution(store,arbitgpu::nativeFixtureSceneBackend(),backend);
                auto original=request;original.parentIntent=&intent;original.evaluationRevision=35;
                ImportedSceneExecutionReceipt old,fresh;
                if (!replacementExecution.executePreview(original,old,error) || !old.deformationFrame)return false;
                const auto guard=replacementExecution.parentPresentationGuard(42);
                const auto native=old.deformationFrame->frame.nativeFrame;
                const auto color=readColor(native);
                if (color.empty())return refuse("static replacement incumbent had no actual native Color");
                auto moved=std::make_shared<Visual3DScene>(*original.sceneSnapshot);
                moved->objects[0].transform.translation.y+=.01f;
                auto published=authored;published.sceneSnapshot=moved;
                auto incoming=original;incoming.sceneSnapshot=moved;
                incoming.staticPayloadIdentity=visualimportedscenerender::encode(published);
                if (incoming.staticPayloadIdentity.empty()
                    || !replacementExecution.executePreview(incoming,fresh,error) || !fresh.deformationFrame
                    || !guard.retired() || !fresh.valid() || !old.valid()
                    || replacementExecution.inspectParentReceipt(old,original,ImportedSceneUse::Preview)
                    || !replacementExecution.inspectParentReceipt(fresh,incoming,ImportedSceneUse::Preview)
                    || !sameOwner(old.deformationFrame->frame.nativeFrame,native) || readColor(native)!=color)
                    return refuse("empty-output static replacement adopted its retired source token or historical native owner");
            }
            for (const bool exporting:{false,true})
            for (unsigned generation=0;generation<2;++generation)
            {
                ImportedScenePayloadExecution exhaustedLayer(store,arbitgpu::nativeFixtureSceneBackend(),backend);
                importedscene::VisualImportedScenePlanCache exhaustedCache;
                videorender::LayerDesc retainedLayer;
                std::vector<ImportedSceneExecutionReceipt> retainedOwners;
                importedscene::ImportedSceneStaticResourceIdentity identity{11,12};
                const auto prepareExhausted=[&](int width) {
                    return importedscene::prepareVisualImportedSceneLayerAtTime({plan},42,width,96,1.0/30,30,
                        exporting?importedscene::NativeImportedSceneRenderUse::Export:importedscene::NativeImportedSceneRenderUse::Preview,
                        &exhaustedLayer,&exhaustedCache,retainedLayer,retainedOwners,error,nullptr,identity);
                };
                if (prepareExhausted(96)!=importedscene::VisualImportedScenePreparation::rendered
                    || retainedOwners.empty() || !retainedLayer.importedParentPresentation)return false;
                const auto saved=retainedLayer;
                const auto escaped=retainedOwners.back();
                const auto native=escaped.deformationFrame->frame.nativeFrame;
                const auto color=readColor(native);
                if (color.empty())return refuse("exhausted Layer incumbent had no actual native Color");
                const auto guard=exhaustedLayer.parentPresentationGuard(saved.importedParentPresentation,42);
                bool exhausted=false;
                for (std::size_t attempt=0;attempt<8192;++attempt)
                    if (!escaped.valid()){exhausted=true;break;}
                if (!exhausted || prepareExhausted(0)!=importedscene::VisualImportedScenePreparation::rejected
                    || guard.retired() || retainedOwners.size()!=1
                    || retainedLayer.texture!=saved.texture || retainedLayer.nativeTextureView!=saved.nativeTextureView
                    || retainedLayer.nativeDepthTextureView!=saved.nativeDepthTextureView
                    || !sameOwner(retainedLayer.importedParentPresentation,saved.importedParentPresentation)
                    || !sameOwner(retainedLayer.rawExportFrame,saved.rawExportFrame)
                    || !sameOwner(retainedOwners.back().deformationFrame->frame.nativeFrame,native) || readColor(native)!=color)
                    return refuse("same-source exhausted cache/layer retry did not conserve its exact presentation");
                if (generation==0)++identity.projectGeneration;
                else ++identity.helperGeneration;
                if (prepareExhausted(0)!=importedscene::VisualImportedScenePreparation::rejected
                    || !guard.retired() || retainedLayer.texture || retainedLayer.nativeTextureView
                    || retainedLayer.nativeDepthTextureView || retainedLayer.rawExportFrame || retainedLayer.importedParentPresentation
                    || retainedOwners.size()!=1 || !sameOwner(retainedOwners.back().deformationFrame->frame.nativeFrame,native)
                    || readColor(native)!=color || escaped.valid())
                    return refuse("exhausted public cache/layer publication resurrected its retired generation or ledger");
            }
            // Ordinary compiled cache/layer publication carries retirement even
            // when no payload executor runs. Historical native owners still live.
            for (const bool exporting:{false,true})
            for (unsigned publication=0;publication<3;++publication)
            for (unsigned failure=0;failure<4;++failure)
            {
                ImportedScenePayloadExecution layered(store,arbitgpu::nativeFixtureSceneBackend(),backend);
                importedscene::VisualImportedScenePlanCache layerCache;
                videorender::LayerDesc publishedLayer;
                std::vector<ImportedSceneExecutionReceipt> layerOwners;
                auto publishedPlan=plan;
                importedscene::ImportedSceneStaticResourceIdentity identity{11,12};
                const auto prepareLayer=[&](int width) {
                    return importedscene::prepareVisualImportedSceneLayerAtTime({publishedPlan},42,width,96,1.0/30,30,
                        exporting?importedscene::NativeImportedSceneRenderUse::Export:importedscene::NativeImportedSceneRenderUse::Preview,
                        &layered,&layerCache,publishedLayer,layerOwners,error,nullptr,identity);
                };
                if (prepareLayer(96)!=importedscene::VisualImportedScenePreparation::rendered
                    || layerOwners.empty() || !layerOwners.back().deformationFrame
                    || !publishedLayer.importedParentPresentation)return false;
                const auto escaped=layerOwners.back();
                const auto native=escaped.deformationFrame->frame.nativeFrame;
                const auto sameDescriptor=[](const auto& a,const auto& b) {
                    return a.backend==b.backend && a.viewKind==b.viewKind && a.format==b.format
                        && a.imageHandle==b.imageHandle && a.textureViewHandle==b.textureViewHandle
                        && a.width==b.width && a.height==b.height && a.sampleCount==b.sampleCount
                        && a.sampledBinding==b.sampledBinding && a.deviceOrContextIdentity==b.deviceOrContextIdentity
                        && a.rendererGeneration==b.rendererGeneration && a.colorSpace==b.colorSpace
                        && a.transfer==b.transfer && a.rowOrder==b.rowOrder;
                };
                const auto color=readColor(native);
                if (color.empty())return refuse("publication Layer incumbent had no actual native Color");
                const auto savedLayer=publishedLayer;
                const auto guard=layered.parentPresentationGuard(42);
                // Same publication plus bad extent restores the exact descriptor,
                // owner and Color. No new container entry is published.
                const auto ownerCount=layerOwners.size();
                if (prepareLayer(0)!=importedscene::VisualImportedScenePreparation::rejected
                    || guard.retired() || layerOwners.size()!=ownerCount
                    || publishedLayer.texture!=savedLayer.texture
                    || publishedLayer.nativeTextureView!=savedLayer.nativeTextureView
                    || publishedLayer.nativeDepthTextureView!=savedLayer.nativeDepthTextureView
                    || !sameDescriptor(publishedLayer.nativeTextureDescriptor,savedLayer.nativeTextureDescriptor)
                    || !sameDescriptor(publishedLayer.nativeDepthTextureDescriptor,savedLayer.nativeDepthTextureDescriptor)
                    || publishedLayer.nativeTextureBackend!=savedLayer.nativeTextureBackend
                    || publishedLayer.nativeDepthTextureBackend!=savedLayer.nativeDepthTextureBackend
                    || publishedLayer.texWidth!=savedLayer.texWidth || publishedLayer.texHeight!=savedLayer.texHeight
                    || publishedLayer.depthWidth!=savedLayer.depthWidth || publishedLayer.depthHeight!=savedLayer.depthHeight
                    || !sameOwner(publishedLayer.importedParentPresentation,savedLayer.importedParentPresentation)
                    || !sameOwner(publishedLayer.rawExportFrame,savedLayer.rawExportFrame)
                    || publishedLayer.rawExportMask!=savedLayer.rawExportMask
                    || publishedLayer.rawExportClipId!=savedLayer.rawExportClipId
                    || publishedLayer.rawExportRenderId!=savedLayer.rawExportRenderId
                    || publishedLayer.rawExportImageOutput!=savedLayer.rawExportImageOutput
                    || !sameOwner(publishedLayer.nativeTextureOwner,savedLayer.nativeTextureOwner)
                    || !sameOwner(publishedLayer.nativeLinearImage,savedLayer.nativeLinearImage)
                    || !sameOwner(layerOwners.back().deformationFrame->frame.nativeFrame,native)
                    || readColor(native)!=color)
                    return refuse("same-generation pre-execute layer refusal lost its exact presentation");
                if (publication==0)++identity.projectGeneration;
                if (publication==1)++identity.helperGeneration;
                if (publication==2)
                {
                    auto publishedAuthored=authored;
                    auto moved=std::make_shared<Visual3DScene>(*p);
                    moved->objects[0].transform.translation.y+=.01f;
                    publishedAuthored.sceneSnapshot=moved;
                    publishedPlan.operations.back().payloadXml=visualimportedscenerender::encode(publishedAuthored);
                    if (publishedPlan.operations.back().payloadXml.empty())return false;
                }
                if (failure==3)
                {
                    // Canonical source camera IDs are admitted by the wire, but
                    // this nonexistent document node refuses in the executor.
                    auto unavailable=authored;
                    auto unavailableLeaf=*unavailable.importedScene;
                    unavailableLeaf.sourceCameraNodeId=9999;
                    unavailable.importedScene=std::make_shared<const visualimportedscenerender::Request>(unavailableLeaf);
                    unavailable.sceneSnapshot=publication==2
                        ? std::make_shared<const Visual3DScene>(*p):authored.sceneSnapshot;
                    publishedPlan.operations.back().payloadXml=visualimportedscenerender::encode(unavailable);
                    if (publishedPlan.operations.back().payloadXml.empty())return false;
                }
                if (failure==1)publishedPlan.operations.back().payloadXml+="malformed compiled replacement";
                std::map<std::string,double> badRuntime;
                if (failure==2)
                {
                    // A genuine compiled plan reaches the ordinary timing
                    // parameter validation after publication and cache resolve.
                    badRuntime["visual803/timelineBeat"]=std::numeric_limits<double>::quiet_NaN();
                }
                const auto prepared=importedscene::prepareVisualImportedSceneLayerAtTime({publishedPlan},42,failure==0?0:96,96,1.0/30,30,
                    exporting?importedscene::NativeImportedSceneRenderUse::Export:importedscene::NativeImportedSceneRenderUse::Preview,
                    &layered,&layerCache,publishedLayer,layerOwners,error,failure==2?&badRuntime:nullptr,identity);
                if (prepared!=importedscene::VisualImportedScenePreparation::rejected || !guard.retired()
                    || publishedLayer.texture || publishedLayer.nativeTextureView || publishedLayer.nativeDepthTextureView
                    || publishedLayer.nativeTextureOwner || publishedLayer.nativeLinearImage
                    || publishedLayer.rawExportFrame || publishedLayer.importedParentPresentation
                    || layerOwners.size()!=ownerCount
                    || !sameOwner(layerOwners.back().deformationFrame->frame.nativeFrame,native)
                    || readColor(native)!=color || !escaped.valid())
                    return refuse("public parent publication resurrected retired layer or destroyed its historical native owner");
                if (failure==0)
                {
                    if (prepareLayer(96)!=importedscene::VisualImportedScenePreparation::rendered
                        || !publishedLayer.importedParentPresentation || layerOwners.size()!=ownerCount+1
                        || !layerOwners.back().deformationFrame)
                        return refuse("replacement parent source did not issue its genuine Layer token");
                    const auto replacementLayer=publishedLayer;
                    const auto replacementOwner=layerOwners.back();
                    const auto replacementNative=replacementOwner.deformationFrame->frame.nativeFrame;
                    const auto replacementColor=readColor(replacementNative);
                    if (replacementColor.empty())return refuse("replacement Layer had no actual native Color");
                    publishedLayer=savedLayer;
                    if (prepareLayer(0)!=importedscene::VisualImportedScenePreparation::rejected
                        || publishedLayer.texture || publishedLayer.nativeTextureView || publishedLayer.nativeDepthTextureView
                        || publishedLayer.rawExportFrame || publishedLayer.importedParentPresentation
                        || layerOwners.size()!=ownerCount+1
                        || layered.parentPresentationGuard(replacementLayer.importedParentPresentation,42).retired()
                        || !sameOwner(layerOwners.back().payload,replacementOwner.payload)
                        || !sameOwner(layerOwners.back().deformationFrame->source,replacementOwner.deformationFrame->source)
                        || !sameOwner(layerOwners.back().deformationFrame->runtimeInputs.compositionFrame,
                            replacementOwner.deformationFrame->runtimeInputs.compositionFrame)
                        || !sameOwner(layerOwners.back().deformationFrame->evaluation.deformation,
                            replacementOwner.deformationFrame->evaluation.deformation)
                        || !sameOwner(layerOwners.back().deformationFrame->frame.nativeFrame,replacementNative)
                        || readColor(replacementNative)!=replacementColor || readColor(native)!=color || !escaped.valid())
                        return refuse("a saved old Layer was restored against a successfully replaced current source");
                }
            }
        }
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
        if (mask==2)
        {
            using Probe=ImportedSceneParentStagingFailureProbe;
            using Site=Probe::Site;
            const auto sites={Site::TokenObject,Site::TokenControlBlock,Site::PayloadAlias,Site::SourceAlias,Site::NativeAlias};
            const auto sameReceipt=[&](const ImportedSceneExecutionReceipt& a,const ImportedSceneExecutionReceipt& b) {
                return a.deformationFrame && b.deformationFrame && a.use==b.use
                    && a.usedLastGoodMaterial==b.usedLastGoodMaterial && a.materialDiagnostic==b.materialDiagnostic
                    && sameOwner(a.payload,b.payload) && sameOwner(a.deformationFrame->payload,b.deformationFrame->payload)
                    && sameOwner(a.canonicalBlockCFrame,b.canonicalBlockCFrame)
                    && sameOwner(a.materialFrameTexture,b.materialFrameTexture)
                    && sameOwner(a.deformationFrame->source,b.deformationFrame->source)
                    && sameOwner(a.deformationFrame->frame.source,b.deformationFrame->frame.source)
                    && sameOwner(a.deformationFrame->frame.nativeFrame,b.deformationFrame->frame.nativeFrame)
                    && sameOwner(a.deformationFrame->runtimeInputs.compositionFrame,b.deformationFrame->runtimeInputs.compositionFrame)
                    && sameOwner(a.deformationFrame->evaluation.deformation,b.deformationFrame->evaluation.deformation)
                    && sameOwner(a.deformationFrame->evaluation.sceneAnimation,b.deformationFrame->evaluation.sceneAnimation)
                    && a.deformationFrame->frame.stats.time==b.deformationFrame->frame.stats.time
                    && ImportedParentBindingFrame::samePrevious(a.deformationFrame->runtimeInputs.previousMotion,
                        b.deformationFrame->runtimeInputs.previousMotion);
            };
            // Real production renders precede each selected throwing allocation.
            // This deterministic fault model tests unwinding, not hardware OOM.
            for (const bool exporting:{false,true})
            for (const bool changedSource:{false,true})
            for (const auto site:sites)
            {
                if (!changedSource && (site==Site::TokenObject || site==Site::TokenControlBlock))continue;
                ImportedScenePayloadExecution staged(store,arbitgpu::nativeFixtureSceneBackend(),backend);
                auto original=request;original.parentIntent=&intent;original.evaluationRevision=35;
                original.frame={0,30,1};original.deformation->playback.timelineSeconds=0;
                ImportedSceneExecutionReceipt preview,offline;
                if (!staged.executePreview(original,preview,error) || !staged.executeExport(original,offline,error))return false;
                original.frame.frame=1;original.deformation->playback.timelineSeconds=1.0/30;
                if (!staged.executePreview(original,preview,error) || !staged.executeExport(original,offline,error))return false;
                auto incumbent=exporting?offline:preview;
                const auto saved=incumbent;
                const auto token=staged.parentPresentationToken(saved);
                const auto guard=staged.parentPresentationGuard(token,42);
                const auto history=Probe::inspect(staged);
                const auto native=saved.deformationFrame->frame.nativeFrame;
                const auto nativeOwner=saved.deformationFrame->runtimeInputs.compositionFrame->owner();
                const auto color=readColor(native);
                if (!token || color.empty())return refuse("staging incumbent had no genuine token or native Color");
                auto incoming=original;incoming.frame.frame=2;incoming.deformation->playback.timelineSeconds=2.0/30;
                if (changedSource)incoming.sceneSnapshot=std::make_shared<const Visual3DScene>(*original.sceneSnapshot);
                bool threw=false;
                {
                    Probe::Scope fault(site);
                    try
                    {
                        if (exporting)staged.executeExport(incoming,incumbent,error);
                        else staged.executePreview(incoming,incumbent,error);
                    }
                    catch (const std::bad_alloc&) { threw=true; }
                    const auto& observed=Probe::state();
                    const bool differentOwner=observed.candidateOwner.owner_before(nativeOwner)
                        || nativeOwner.owner_before(observed.candidateOwner);
                    if (!threw || !observed.hit || differentOwner!=changedSource
                        || (!observed.candidateNative.owner_before(native) && !native.owner_before(observed.candidateNative))
                        || !Probe::conserved(staged,history) || guard.retired() || !sameReceipt(incumbent,saved)
                        || !sameOwner(staged.parentPresentationToken(incumbent),token)
                        || !saved.valid() || readColor(native)!=color)
                        return refuse("post-native outer staging failure changed exact token, caller, source, material or either history");
                }
                incoming.sceneSnapshot=original.sceneSnapshot;
                const auto resumed=exporting?staged.executeExport(incoming,incumbent,error):staged.executePreview(incoming,incumbent,error);
                if (!resumed || !incumbent.deformationFrame || !incumbent.deformationFrame->runtimeInputs.previousMotion
                    || !sameOwner(incumbent.deformationFrame->runtimeInputs.previousMotion->compositionFrame,
                        saved.deformationFrame->runtimeInputs.compositionFrame)
                    || !sameOwner(incumbent.deformationFrame->runtimeInputs.compositionFrame->owner(),nativeOwner)
                    || guard.retired() || !sameOwner(staged.parentPresentationToken(incumbent),token) || readColor(native)!=color)
                    return refuse("staging unwind lost the original actual source or adjacent native predecessor on retry");
            }
            const auto sameDescriptor=[](const auto& a,const auto& b) {
                return a.backend==b.backend && a.viewKind==b.viewKind && a.format==b.format
                    && a.imageHandle==b.imageHandle && a.textureViewHandle==b.textureViewHandle
                    && a.width==b.width && a.height==b.height && a.sampleCount==b.sampleCount
                    && a.sampledBinding==b.sampledBinding && a.deviceOrContextIdentity==b.deviceOrContextIdentity
                    && a.rendererGeneration==b.rendererGeneration && a.colorSpace==b.colorSpace
                    && a.transfer==b.transfer && a.rowOrder==b.rowOrder;
            };
            for (const bool exporting:{false,true})
            for (const auto site:sites)
            {
                ImportedScenePayloadExecution staged(store,arbitgpu::nativeFixtureSceneBackend(),backend);
                importedscene::VisualImportedScenePlanCache stagedCache;
                videorender::LayerDesc stagedLayer;
                std::vector<ImportedSceneExecutionReceipt> stagedOwners;
                const auto prepareStaged=[&](std::int64_t frame) {
                    return importedscene::prepareVisualImportedSceneLayerAtTime({plan},42,96,96,double(frame)/30,30,
                        exporting?importedscene::NativeImportedSceneRenderUse::Export:importedscene::NativeImportedSceneRenderUse::Preview,
                        &staged,&stagedCache,stagedLayer,stagedOwners,error,nullptr,{11,12});
                };
                if (prepareStaged(0)!=importedscene::VisualImportedScenePreparation::rendered
                    || prepareStaged(1)!=importedscene::VisualImportedScenePreparation::rendered
                    || stagedOwners.size()!=2 || !stagedOwners.back().deformationFrame)return false;
                const auto savedLayer=stagedLayer;
                const auto saved=stagedOwners.back();
                const auto history=Probe::inspect(staged);
                const auto native=saved.deformationFrame->frame.nativeFrame;
                const auto nativeOwner=saved.deformationFrame->runtimeInputs.compositionFrame->owner();
                const auto color=readColor(native);
                const auto guard=staged.parentPresentationGuard(savedLayer.importedParentPresentation,42);
                if (color.empty())return refuse("cache-reset staging incumbent had no native Color");
                // Only cache reset causes fresh equal-valued procedural admission.
                // Executor, Layer, publication, frame owners and budgets survive.
                stagedCache.reset();
                bool threw=false;
                {
                    Probe::Scope fault(site);
                    try { prepareStaged(2); }
                    catch (const std::bad_alloc&) { threw=true; }
                    const auto& observed=Probe::state();
                    if (!threw || !observed.hit
                        || (!observed.candidateOwner.owner_before(nativeOwner) && !nativeOwner.owner_before(observed.candidateOwner))
                        || !Probe::conserved(staged,history) || guard.retired() || stagedOwners.size()!=2
                        || !sameOwner(staged.parentPresentationToken(saved),savedLayer.importedParentPresentation)
                        || !sameReceipt(stagedOwners.back(),saved)
                        || stagedLayer.texture!=savedLayer.texture || stagedLayer.nativeTextureView!=savedLayer.nativeTextureView
                        || stagedLayer.nativeDepthTextureView!=savedLayer.nativeDepthTextureView
                        || !sameDescriptor(stagedLayer.nativeTextureDescriptor,savedLayer.nativeTextureDescriptor)
                        || !sameDescriptor(stagedLayer.nativeDepthTextureDescriptor,savedLayer.nativeDepthTextureDescriptor)
                        || stagedLayer.nativeTextureBackend!=savedLayer.nativeTextureBackend
                        || stagedLayer.nativeDepthTextureBackend!=savedLayer.nativeDepthTextureBackend
                        || stagedLayer.texWidth!=savedLayer.texWidth || stagedLayer.texHeight!=savedLayer.texHeight
                        || stagedLayer.depthWidth!=savedLayer.depthWidth || stagedLayer.depthHeight!=savedLayer.depthHeight
                        || !sameOwner(stagedLayer.importedParentPresentation,savedLayer.importedParentPresentation)
                        || !sameOwner(stagedLayer.nativeTextureOwner,savedLayer.nativeTextureOwner)
                        || !sameOwner(stagedLayer.nativeLinearImage,savedLayer.nativeLinearImage)
                        || !sameOwner(stagedLayer.rawExportFrame,savedLayer.rawExportFrame)
                        || stagedLayer.rawExportMask!=savedLayer.rawExportMask || stagedLayer.rawExportClipId!=savedLayer.rawExportClipId
                        || stagedLayer.rawExportRenderId!=savedLayer.rawExportRenderId
                        || stagedLayer.rawExportImageOutput!=savedLayer.rawExportImageOutput
                        || !saved.valid() || readColor(native)!=color)
                        return refuse("same-source cache-reset staging exception changed current Layer, historical owners or inner histories");
                }
                if (prepareStaged(2)!=importedscene::VisualImportedScenePreparation::rendered || stagedOwners.size()!=3
                    || !guard.retired() || !stagedLayer.importedParentPresentation
                    || sameOwner(stagedLayer.importedParentPresentation,savedLayer.importedParentPresentation)
                    || !saved.valid() || readColor(native)!=color)
                    return refuse("successful cache-reset retry failed to replace the real token after whole staging");
            }
            {
                ImportedScenePayloadExecution staged(store,arbitgpu::nativeFixtureSceneBackend(),backend);
                auto original=request;original.parentIntent=&intent;original.evaluationRevision=35;
                original.frame={0,30,1};original.deformation->playback.timelineSeconds=0;
                surfacematerialbinding::ImportedSceneMaterialRequest material;
                material.scene=*visualimportedsceneidentity::sceneId(0);material.sceneRevision=key.version;
                material.structuralRevision=31;material.evaluationRevision=35;material.programRevision=31;
                material.binding.targetKind=surfacematerialbinding::BindingTargetKind::ObjectOverride;
                material.binding.object={1};material.binding.surfaceMaterialRevision=31;
                for (const auto output:surfacematerial::kSurfaceOutputs)
                {
                    surfacematerial::Operation operation;
                    operation.id=static_cast<surfacematerial::ValueId>(material.program.operations.size()+1);
                    operation.resultType=surfacematerial::outputType(output);
                    operation.kind=operation.resultType==surfacematerial::ValueType::UInt
                        ?surfacematerial::OperationKind::UIntConstant:surfacematerial::OperationKind::FloatConstant;
                    if (output==surfacematerial::OutputSemantic::BaseColor)operation.literal={.5f,.4f,.3f,0};
                    if (output==surfacematerial::OutputSemantic::Normal)operation.literal={0,0,1,0};
                    if (output==surfacematerial::OutputSemantic::Opacity)operation.literal[0]=1;
                    if (output==surfacematerial::OutputSemantic::Ior)operation.literal[0]=1.5f;
                    if (output==surfacematerial::OutputSemantic::MaterialId)operation.unsignedLiteral=1;
                    material.program.outputs[static_cast<std::size_t>(output)]=operation.id;
                    material.program.operations.push_back(operation);
                }
                const auto admitted=surfacematerial::admit(material.program,error);
                if (!admitted)return false;
                material.binding.surfaceMaterialDigest=admitted->structuralDigest();
                auto materialLeaf=*original.importedScene;materialLeaf.material=material;
                original.importedScene=std::make_shared<const visualimportedscenerender::Request>(materialLeaf);
                original.material=material;
                auto materialAuthored=authored;materialAuthored.importedScene=original.importedScene;
                original.staticPayloadIdentity=visualimportedscenerender::encode(materialAuthored);
                ImportedSceneExecutionReceipt incumbent;
                if (original.staticPayloadIdentity.empty() || !staged.executePreview(original,incumbent,error)
                    || !incumbent.deformationFrame || incumbent.usedLastGoodMaterial)return false;
                const auto saved=incumbent;
                const auto history=Probe::inspect(staged);
                const auto token=staged.parentPresentationToken(saved);
                const auto guard=staged.parentPresentationGuard(token,42);
                const auto native=saved.deformationFrame->frame.nativeFrame;
                const auto color=readColor(native);
                if (color.empty())return refuse("diagnostic staging incumbent had no native Color");
                auto incoming=original;incoming.frame.frame=1;incoming.deformation->playback.timelineSeconds=1.0/30;
                // A coherent public material request for a missing native target
                // exercises the released Preview last-good-material diagnostic.
                incoming.material->binding.object={999999};
                materialLeaf.material=incoming.material;
                incoming.importedScene=std::make_shared<const visualimportedscenerender::Request>(materialLeaf);
                bool threw=false;
                {
                    Probe::Scope fault(Site::DiagnosticCopy);
                    try { staged.executePreview(incoming,incumbent,error); }
                    catch (const std::bad_alloc&) { threw=true; }
                    if (!threw || !Probe::state().hit || !Probe::state().diagnosticAllocates
                        || !Probe::conserved(staged,history) || guard.retired() || !sameReceipt(incumbent,saved)
                        || !sameOwner(staged.parentPresentationToken(incumbent),token) || !saved.valid() || readColor(native)!=color)
                        return refuse("real post-native diagnostic allocation failure lost incumbent material, source, histories or caller");
                }
                if (!staged.executePreview(incoming,incumbent,error) || !incumbent.usedLastGoodMaterial
                    || incumbent.materialDiagnostic.empty() || !incumbent.deformationFrame
                    || incumbent.materialDiagnostic!=incumbent.deformationFrame->materialDiagnostic
                    || guard.retired() || !sameOwner(staged.parentPresentationToken(incumbent),token)
                    || !incumbent.deformationFrame->runtimeInputs.previousMotion
                    || !sameOwner(incumbent.deformationFrame->runtimeInputs.previousMotion->compositionFrame,
                        saved.deformationFrame->runtimeInputs.compositionFrame) || readColor(native)!=color)
                    return refuse("diagnostic staging retry lost genuine last-good material or original adjacent predecessor");
            }
        }
#endif
#if defined(ARBIT_TEST_IMPORTED_PARENT_STAGING_FAILURES)
        if (mask==2)
        {
            using Probe=ImportedSceneParentStagingFailureProbe;
            const auto sameWeak=[](const auto& a,const auto& b) {
                return a.lock().get()==b.lock().get() && !a.owner_before(b) && !b.owner_before(a);
            };
            const auto sameSample=[&](const Probe::Sample& a,const Probe::Sample& b) {
                return a.present==b.present && (!a.present || (a.translation==b.translation && a.rotation==b.rotation
                    && a.cameraTranslation==b.cameraTranslation && a.scale==b.scale && a.morph==b.morph
                    && a.owner==b.owner && a.camera.has_value()==b.camera.has_value()
                    && (!a.camera || scenecomposition::sameCameraRecord(*a.camera,*b.camera))
                    && sameWeak(a.deformation,b.deformation) && sameWeak(a.composition,b.composition)
                    && sameWeak(a.worlds,b.worlds) && sameWeak(a.notes,b.notes)));
            };
            const auto sameHistory=[&](const Probe::History& a,const Probe::History& b) {
                return a.frame==b.frame && a.numerator==b.numerator && a.denominator==b.denominator
                    && a.width==b.width && a.height==b.height && a.owner==b.owner
                    && a.mappingEmpty==b.mappingEmpty && a.noteOwnerAbsent==b.noteOwnerAbsent
                    && sameSample(a.current,b.current) && sameSample(a.previous,b.previous);
            };
            // Only incumbent owns an allocating diagnostic. Native/token/weak
            // witnesses do not retain its supplementary diagnostic debit.
            for (const int boundary:{0,1,2,3,4})
            {
                ImportedScenePayloadExecution staged(store,arbitgpu::nativeFixtureSceneBackend(),backend);
                auto original=request;original.parentIntent=&intent;original.evaluationRevision=35;
                original.frame={0,30,1};original.deformation->playback.timelineSeconds=0;
                surfacematerialbinding::ImportedSceneMaterialRequest material;
                material.scene=*visualimportedsceneidentity::sceneId(0);material.sceneRevision=key.version;
                material.structuralRevision=31;material.evaluationRevision=35;material.programRevision=31;
                material.binding.targetKind=surfacematerialbinding::BindingTargetKind::ObjectOverride;
                material.binding.object={1};material.binding.surfaceMaterialRevision=31;
                for (const auto output:surfacematerial::kSurfaceOutputs)
                {
                    surfacematerial::Operation operation;
                    operation.id=static_cast<surfacematerial::ValueId>(material.program.operations.size()+1);
                    operation.resultType=surfacematerial::outputType(output);
                    operation.kind=operation.resultType==surfacematerial::ValueType::UInt
                        ?surfacematerial::OperationKind::UIntConstant:surfacematerial::OperationKind::FloatConstant;
                    if (output==surfacematerial::OutputSemantic::BaseColor)operation.literal={.5f,.4f,.3f,0};
                    if (output==surfacematerial::OutputSemantic::Normal)operation.literal={0,0,1,0};
                    if (output==surfacematerial::OutputSemantic::Opacity)operation.literal[0]=1;
                    if (output==surfacematerial::OutputSemantic::Ior)operation.literal[0]=1.5f;
                    if (output==surfacematerial::OutputSemantic::MaterialId)operation.unsignedLiteral=1;
                    material.program.outputs[static_cast<std::size_t>(output)]=operation.id;
                    material.program.operations.push_back(operation);
                }
                const auto admitted=surfacematerial::admit(material.program,error);
                if (!admitted)return false;
                material.binding.surfaceMaterialDigest=admitted->structuralDigest();
                auto materialLeaf=*original.importedScene;materialLeaf.material=material;
                original.importedScene=std::make_shared<const visualimportedscenerender::Request>(materialLeaf);
                original.material=material;
                auto materialAuthored=authored;materialAuthored.importedScene=original.importedScene;
                original.staticPayloadIdentity=visualimportedscenerender::encode(materialAuthored);
                ImportedSceneExecutionReceipt incumbent;
                if (original.staticPayloadIdentity.empty() || !staged.executePreview(original,incumbent,error)
                    || !incumbent.deformationFrame || incumbent.usedLastGoodMaterial)return false;
                {
                    ImportedSceneExecutionReceipt offline;
                    if (!staged.executeExport(original,offline,error))return false;
                }
                auto incoming=original;incoming.frame.frame=1;incoming.deformation->playback.timelineSeconds=1.0/30;
                incoming.material->binding.object={999999};materialLeaf.material=incoming.material;
                incoming.importedScene=std::make_shared<const visualimportedscenerender::Request>(materialLeaf);
                if (!staged.executePreview(incoming,incumbent,error) || !incumbent.usedLastGoodMaterial
                    || !incumbent.deformationFrame || incumbent.materialDiagnostic.empty()
                    || incumbent.materialDiagnostic!=incumbent.deformationFrame->materialDiagnostic)
                    return refuse("successful diagnostic disposal case did not publish genuine Preview last-good material");
                auto diagnostic=Probe::diagnostic(incumbent);
                const auto emptyCapacity=std::string().capacity();
                if (diagnostic.capacity<=emptyCapacity || diagnostic.bytes!=incumbent.deformationFrame->materialDiagnostic.capacity()+1
                    || diagnostic.capacity+1>diagnostic.bytes || diagnostic.retention.expired()
                    || diagnostic.retention.use_count()!=1)
                    return refuse("allocating diagnostic has no sole exact supplementary debit");
                const auto token=staged.parentPresentationToken(incumbent);
                const auto guard=staged.parentPresentationGuard(token,42);
                const auto native=incumbent.deformationFrame->frame.nativeFrame;
                const auto color=readColor(native);
                const auto source=incumbent.deformationFrame->source;
                const auto nativeOwner=incumbent.deformationFrame->runtimeInputs.compositionFrame->owner();
                auto priorComposition=incumbent.deformationFrame->runtimeInputs.compositionFrame;
                const auto history=Probe::inspect(staged);
                if (!token || color.empty())return refuse("diagnostic disposal lacks genuine current token and native Color");
                if (boundary==0)
                {
                    auto restored=original;restored.frame.frame=2;restored.deformation->playback.timelineSeconds=2.0/30;
                    if (!staged.executePreview(restored,incumbent,error) || !incumbent.deformationFrame
                        || incumbent.usedLastGoodMaterial || !incumbent.materialDiagnostic.empty()
                        || !incumbent.deformationFrame->materialDiagnostic.empty()
                        || !staged.inspectParentReceipt(incumbent,restored,ImportedSceneUse::Preview))return false;
                    const auto replacement=Probe::diagnostic(incumbent);
                    const auto after=Probe::inspect(staged);
                    if (!diagnostic.retention.expired() || replacement.retention.expired()
                        || replacement.capacity!=emptyCapacity || replacement.bytes!=replacement.capacity+1
                        || incumbent.deformationFrame->materialDiagnostic.capacity()!=replacement.capacity
                        || !sameWeak(replacement.pool,diagnostic.pool) || guard.retired()
                        || !sameOwner(staged.parentPresentationToken(incumbent),token)
                        || !sameOwner(incumbent.deformationFrame->source,source)
                        || !sameOwner(incumbent.deformationFrame->runtimeInputs.compositionFrame->owner(),nativeOwner)
                        || !sameWeak(after.source,history.source) || !sameWeak(after.parent,history.parent)
                        || !sameWeak(after.material,history.material) || after.materialRevisions!=history.materialRevisions
                        || !sameHistory(after.history[1],history.history[1]) || after.history[0].frame!=2
                        || !sameWeak(after.history[0].previous.composition,history.history[0].current.composition)
                        || !incumbent.deformationFrame->runtimeInputs.previousMotion
                        || !sameOwner(incumbent.deformationFrame->runtimeInputs.previousMotion->compositionFrame,priorComposition)
                        || readColor(incumbent.deformationFrame->frame.nativeFrame).empty() || readColor(native)!=color)
                        return refuse("long-to-empty publication retained old capacity/debit or changed source, material, token or histories");
                    // Publish another genuine allocating diagnostic to check the
                    // same sole-caller disposal at positive retirement next.
                    incoming.frame.frame=3;incoming.deformation->playback.timelineSeconds=3.0/30;
                    if (!staged.executePreview(incoming,incumbent,error) || !incumbent.deformationFrame
                        || !incumbent.usedLastGoodMaterial || incumbent.materialDiagnostic.empty())return false;
                    diagnostic=Probe::diagnostic(incumbent);
                    if (diagnostic.capacity<=emptyCapacity || diagnostic.retention.use_count()!=1)return false;
                }
                auto retired=incoming;retired.dimensions.width=0;
                if (boundary<=2)++retired.projectGeneration;
                if (boundary==1)
                    retired.staticPayloadIdentity.reserve(NativeSceneAnimationCompositionBudget::kMaximumBytes);
                if (boundary==2)
                {
                    while (incumbent.valid()) {}
                    if (Probe::inspect(staged).attemptedWork!=NativeSceneAnimationCompositionBudget::kMaximumWork)
                        return refuse("genuine diagnostic retirement did not exhaust the original inspector ledger");
                }
                if (boundary==3)retired.parentIntent=nullptr;
                if (boundary==4)retired.glassCaustic=visualimportedscenerender::GlassCausticSelection {};
                if (staged.executePreview(retired,incumbent,error) || incumbent.payload || incumbent.deformationFrame
                    || !incumbent.materialDiagnostic.empty() || incumbent.materialDiagnostic.capacity()!=emptyCapacity
                    || !Probe::diagnostic(incumbent).retention.expired() || readColor(native)!=color)
                    return refuse("executor clear retained an allocating diagnostic after removing its stamp");
                if (boundary<=2 && !guard.retired())return refuse("positive retirement did not retire its actual token");
                if (diagnostic.retention.expired())
                    return refuse("retired publication dropped its supplementary debit before explicit reset");
                const auto beforeReset=Probe::inspect(staged);
                staged.reset();
                const auto afterReset=Probe::inspect(staged);
                if (!diagnostic.retention.expired() || incumbent.materialDiagnostic.capacity()!=emptyCapacity
                    || incumbent.payload || incumbent.deformationFrame || !guard.retired()
                    || !sameWeak(beforeReset.retentionBudget,afterReset.retentionBudget)
                    || !sameWeak(beforeReset.workBudget,afterReset.workBudget)
                    || afterReset.attemptedWork<beforeReset.attemptedWork || readColor(native)!=color)
                    return refuse("clear-then-reset refunded a surviving caller buffer or replaced its persistent budget");
            }
        }
#endif
        cache.reset();execution.reset();
        if(!current.valid())return refuse("escaped parent receipt lost its source or permanent ledger after reset");
    }
    // Exercise the actual complete-union function with one-short permanent work.
    // This helper creates no native capability and uses unchanged real sources.
    auto procedural=std::make_shared<Visual3DScene>(makeScene());procedural->id={700};
    visualimportedscenerender::ImportedParentIntentV1 intent;
    intent.direction=2;intent.composeStableId=700;intent.importedSourceStableId=801;
    intent.connectedInputMask=3;intent.requestedParentMask=2;intent.rootMembershipCount=1;
    intent.rootMembership[0]={procedural->objects[0].id.value,1};
    auto basis=gltf::composeImportedVisual3DScene(*procedural,*imported,error);if(!basis)return false;
    auto budget=std::make_shared<NativeSceneAnimationCompositionBudget>();ImportedParentBindingOwner::Parents expected;
    if(!ImportedParentBindingOwner::parents(intent,*imported,*procedural,*basis,*budget,expected))return false;
    const auto exact=budget->logicalWork();
    auto shortWork=std::make_shared<NativeSceneAnimationCompositionBudget>();
    if(!shortWork->consumeWork(NativeSceneAnimationCompositionBudget::kMaximumWork-exact+1,1)
        || ImportedParentBindingOwner::parents(intent,*imported,*procedural,*basis,*shortWork,expected)
        || ImportedParentBindingOwner::parents(intent,*imported,*procedural,*basis,*shortWork,expected)
        || shortWork->logicalWork()!=NativeSceneAnimationCompositionBudget::kMaximumWork)
        return refuse("parented one-short resolution failure refunded attempted work");
    auto parented=std::make_shared<Visual3DScene>(*basis);
    for(std::size_t x=0;x<parented->objectCount;++x)parented->objects[x].parent=expected.values[x];
    auto complete=std::make_shared<NativeSceneAnimationCompositionBudget>();
    if(!ImportedParentBindingOwner::parents(intent,*imported,*procedural,*basis,*complete,expected)
        || !ImportedParentBindingOwner::conservation(*basis,*parented,expected,*complete))return false;
    const auto completeWork=complete->logicalWork();
    auto shortConservation=std::make_shared<NativeSceneAnimationCompositionBudget>();
    if(!shortConservation->consumeWork(NativeSceneAnimationCompositionBudget::kMaximumWork-completeWork+1,1)
        || !ImportedParentBindingOwner::parents(intent,*imported,*procedural,*basis,*shortConservation,expected)
        || ImportedParentBindingOwner::conservation(*basis,*parented,expected,*shortConservation)
        || ImportedParentBindingOwner::conservation(*basis,*parented,expected,*shortConservation)
        || shortConservation->logicalWork()!=NativeSceneAnimationCompositionBudget::kMaximumWork)
        return refuse("parent one-short full conservation refunded failed inspector work");
    for(unsigned mutation=0;mutation<5;++mutation)
    {
        auto changed=std::make_shared<Visual3DScene>(*parented);
        if(mutation==0)changed->vertices.back().position.x+=1;
        if(mutation==1)changed->objects.back().transform.translation.x+=1;
        if(mutation==2)changed->materials.back().emissive.x+=1;
        if(mutation==3)changed->lights.back().environmentImage.asset.id="inactive identity mutation";
        if(mutation==4)changed->objects[0].transform.scale.x+=.1f;
        auto census=std::make_shared<NativeSceneAnimationCompositionBudget>();
        if(ImportedParentBindingOwner::conservation(*basis,*changed,expected,*census))
            return refuse("parent full-scene conservation ignored non-parent or inactive record mutation");
    }
    // Each full frozen-source comparison charges its own real schedule. A
    // rejected attempt permanently consumes the same ledger on repeated retry.
    auto sourceWork=std::make_shared<NativeSceneAnimationCompositionBudget>();
    if (!ImportedParentBindingOwner::unchangedSource(*procedural,*procedural,*sourceWork))return false;
    const auto exactSourceWork=sourceWork->logicalWork();
    auto shortSourceWork=std::make_shared<NativeSceneAnimationCompositionBudget>();
    if (!shortSourceWork->consumeWork(NativeSceneAnimationCompositionBudget::kMaximumWork-exactSourceWork+1,1)
        || ImportedParentBindingOwner::unchangedSource(*procedural,*procedural,*shortSourceWork)
        || ImportedParentBindingOwner::unchangedSource(*procedural,*procedural,*shortSourceWork)
        || shortSourceWork->logicalWork()!=NativeSceneAnimationCompositionBudget::kMaximumWork)
        return refuse("parent one-short frozen-source comparison refunded its failed permanent work");
    // The same private retention pool used by production refuses one-short
    // metadata/clone/scratch reservations. This issues no native capability.
    auto storage=std::make_shared<NativeCompositionRetentionBudget>();
    const auto room=NativeCompositionRetentionBudget::kMaximumBytes-storage->logicalBytes();
    auto occupied=NativeCompositionRetentionBudget::reserve(storage,room-1,error);
    if(!occupied || NativeCompositionRetentionBudget::reserve(storage,2,error))
        return refuse("parent one-short persistent storage admission exceeded its unchanged ceiling");
    NativeCompositionRetentionBudget::Scratch scratch;
    if(scratch.acquire(storage,2))return refuse("parent inspector scratch borrowed nonexistent one-short storage");
    occupied.reset();
    error.clear();return true;
}
} // namespace videohelper::fixture3d
