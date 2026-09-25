#pragma once

#include "reactive_surface_fixture.h"
#include "../src/render_snapshot.h"

inline surfacematerialbinding::ImportedSceneMaterialRequest reactiveFrameRequest(int node = 91, int port = 0)
{
    using namespace surfacematerial;
    auto request = reactiveSurfaceRequest();
    request.program.textureSlotCount = 1;
    const auto append = [&](OperationKind kind, ValueType type, std::initializer_list<ValueId> inputs,
                            InputSemantic semantic = InputSemantic::Invalid, std::uint16_t parameter = 0) {
        Operation operation;
        operation.id = static_cast<ValueId>(request.program.operations.size() + 1);
        operation.kind = kind; operation.resultType = type; operation.semantic = semantic; operation.parameter = parameter;
        operation.inputCount = static_cast<std::uint8_t>(inputs.size());
        std::copy(inputs.begin(),inputs.end(),operation.inputs.begin());
        request.program.operations.push_back(operation); return operation.id;
    };
    const auto uv = append(OperationKind::Input,ValueType::Vec2,{},InputSemantic::TexCoord0);
    const auto sample = append(OperationKind::TextureSample2D,ValueType::Vec4,{uv});
    const auto red = append(OperationKind::Component,ValueType::Scalar,{sample},InputSemantic::Invalid,0);
    const auto green = append(OperationKind::Component,ValueType::Scalar,{sample},InputSemantic::Invalid,1);
    const auto blue = append(OperationKind::Component,ValueType::Scalar,{sample},InputSemantic::Invalid,2);
    const auto rgb = append(OperationKind::ComposeVec3,ValueType::Vec3,{red,green,blue});
    request.program.outputs[0] = append(OperationKind::Multiply,ValueType::Vec3,{rgb,request.program.outputs[0]});
    request.version = surfacematerialbinding::kGraphFrameWireVersion;
    surfacematerialbinding::TextureSlotBinding texture;
    texture.source = surfacematerialbinding::TextureSourceKind::GraphFrame;
    texture.graphFrame = surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint{node,port};
    request.binding.textures = {texture};
    std::string error;
    const auto admitted = admit(request.program,error);
    if (admitted) request.binding.surfaceMaterialDigest = admitted->structuralDigest();
    return request;
}

inline videowire::CompiledVisualLayerPlan reactiveFramePlan(std::string& error)
{
    using namespace videowire::geometry;
    PortContract contract; contract.carrier = CarrierKind::geometry3D; contract.maxVertices = 4; contract.maxIndices = 6;
    const auto geometry = lowerGrid(100,100,1,2,2,1,contract,error);
    const auto value = geometry ? admitValue(*geometry,contract,error) : std::nullopt;
    videowire::CompiledVisualLayerPlan plan;
    if (!value) return plan;
    plan.clipId = 92; plan.structuralRevision = 1;
    plan.nodeIds = {71,91}; plan.nodeKinds = {"geometry.core.runtime","video.source"};
    plan.operations = {{71,"geometry.core.runtime","native-gpu",
        encodeLoweredPlanText(lowerRuntimePlan(contract,*value,geometrysurfacematerial::encode(reactiveFrameRequest())))},
        {91,"video.source","source-decode",{}}};
    plan.edges = {{91,0,71,12}};
    plan.ports = {{71,1,1,"out","frame","image","rgba8","sRGB"},
        {91,0,1,"out","frame","image","rgba8","sRGB"},
        {71,12,1,"in","frame","image","rgba8","sRGB"}};
    return plan;
}

template <typename ReadPixels>
bool verifyNativeReactiveFrame(arbitgpu::NativeFixtureSceneBackend& backend, ReadPixels readPixels,
    const std::shared_ptr<HarmonicMIDI::grid::Visual3DScene>& inputScene, std::string& error)
{
    using namespace arbitgpu;
    using namespace videohelper::geometry;
    if (!inputScene || inputScene->objectCount < 2) { error = "Frame fixture requires two scene objects"; return false; }
    auto scene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*inputScene);
    scene->vertices[0].uv = {0,0}; scene->vertices[1].uv = {1,0}; scene->vertices[2].uv = {0.5f,1};
    const auto target = backend.info().backend == "metal" ? videohelper::materialprogram::BackendTarget::Metal
        : videohelper::materialprogram::BackendTarget::OpenGl;
    const auto binding = [&](const auto& snapshot,auto request, std::size_t object) {
        request.scene = snapshot->id; request.sceneSnapshot = snapshot; request.binding.object = snapshot->objects[object].id;
        for (auto& op : request.program.operations) if (op.id == request.program.outputs[9]) op.unsignedLiteral = snapshot->objects[object].material.value;
        const auto ir = surfacematerial::admit(request.program,error);
        if (ir) request.binding.surfaceMaterialDigest = ir->structuralDigest();
        return videorender::fixture3d::admitSurfaceMaterialBinding(snapshot,request,target,error);
    };
    auto sourceScene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*scene);
    sourceScene->objectCount = 1; sourceScene->objects[0].transform.translation = {};
    sourceScene->vertices[0].position = {-100,-100,0}; sourceScene->vertices[1].position = {100,-100,0};
    sourceScene->vertices[2].position = {0,100,0};
    const auto sourceBinding = binding(sourceScene,reactiveSurfaceRequest(true),0);
    if (!sourceBinding) return false;
    const auto sources = backend.prepare(sourceScene,sourceBinding->nativeProgram());
    if (!sources.prepared) { error = sources.error; return false; }
    NativeFixtureSceneRuntimeInputs inputs;
    const auto first = backend.render(sourceScene,sources.resources,64,64,inputs);
    inputs.timeSeconds = 1;
    const auto later = backend.render(sourceScene,sources.resources,64,64,inputs);
    if (!first.rendered || !later.rendered) { error = "Frame fixture source draw failed"; return false; }
    const auto textureBinding = binding(scene,reactiveFrameRequest(),1);
    if (!textureBinding) return false;
    auto program = std::make_shared<NativeFixtureSurfaceMaterialProgram>(*textureBinding->nativeProgram());
    const auto targetResources = backend.prepare(scene,program);
    if (!targetResources.prepared) { error = targetResources.error; return false; }
    inputs.materialFrameTexture = first.frame;
    const auto a = backend.render(scene,targetResources.resources,64,64,inputs);
    inputs.materialFrameTexture = later.frame;
    const auto b = backend.render(scene,targetResources.resources,64,64,inputs);
    inputs.materialFrameTexture = first.frame;
    const auto sought = backend.render(scene,targetResources.resources,64,64,inputs);
    if (!a.rendered || !b.rendered || !sought.rendered || readPixels(a.frame) == readPixels(b.frame)
        || readPixels(a.frame) != readPixels(sought.frame))
    { error = "Exact-object Frame binding must refresh and replay without mutating prepared resources"; return false; }
    auto reversed = std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*scene);
    std::swap(reversed->objects[0],reversed->objects[1]);
    const auto reversedResources = backend.prepare(reversed,program);
    const auto reordered = reversedResources.prepared ? backend.render(reversed,reversedResources.resources,64,64,inputs)
        : NativeFixtureSceneSubmission{};
    if (!reordered.rendered || readPixels(reordered.frame) != readPixels(a.frame))
    { error = "Frame material target must follow object identity after scene reorder"; return false; }
    visualimportedscenerender::Request retained;
    retained.sceneSnapshot=scene; retained.sourceStableId=scene->id.value; retained.renderStableId=9751;
    retained.structuralRevision=retained.evaluationRevision=1;
    std::array<HarmonicMIDI::grid::SceneObjectRecord,2> objects {scene->objects[0],scene->objects[1]};
    std::sort(objects.begin(),objects.end(),[](const auto& left,const auto& right) { return left.id.value<right.id.value; });
    for (std::size_t index=0;index<objects.size();++index) {
        geometrysurfacematerial::ObjectProgram object;
        object.program.sourceMaterial=9752+index;
        auto& request=object.program.material;
        request=index==0 ? reactiveSurfaceRequest() : reactiveFrameRequest();
        request.scene=scene->id; request.sceneSnapshot=scene; request.binding.object=objects[index].id;
        for (auto& op : request.program.operations) if (op.id==request.program.outputs[9]) op.unsignedLiteral=objects[index].material.value;
        const auto ir=surfacematerial::admit(request.program,error);
        if (!ir) return false;
        request.binding.surfaceMaterialDigest=ir->structuralDigest();
        retained.surfacePrograms.push_back(std::move(object));
    }
    visualimportedscenerender::Request reopened;
    if (!visualimportedscenerender::decode(visualimportedscenerender::encode(retained),reopened))
    { error="Retained mixed Surface Frame collection must survive immutable transport"; return false; }
    const auto collection=videorender::fixture3d::admitSurfaceMaterialCollection(
        reopened.sceneSnapshot,reopened.surfacePrograms,{},target,error);
    if (!collection) return false;
    videorender::fixture3d::FixtureSceneRenderer renderer(backend);
    videorender::fixture3d::RenderedFrame initial,refreshed,replay,offline;
    inputs.materialFrameTexture=first.frame;
    if (!renderer.renderPreview(reopened.sceneSnapshot,collection,{},inputs,{64,64},"native-gpu",initial,error)) return false;
    inputs.materialFrameTexture=later.frame;
    if (!renderer.renderPreview(reopened.sceneSnapshot,collection,{},inputs,{64,64},"native-gpu",refreshed,error)
        || !renderer.renderExport(reopened.sceneSnapshot,collection,{},inputs,{64,64},"native-gpu",offline,error)) return false;
    inputs.materialFrameTexture=first.frame;
    if (!renderer.renderPreview(reopened.sceneSnapshot,collection,{},inputs,{64,64},"native-gpu",replay,error)) return false;
    if (readPixels(initial.nativeFrame).empty() || readPixels(initial.nativeFrame)==readPixels(refreshed.nativeFrame)
        || readPixels(initial.nativeFrame)!=readPixels(replay.nativeFrame)
        || readPixels(refreshed.nativeFrame)!=readPixels(offline.nativeFrame))
    { error="Retained nonfirst Surface Frame must refresh and match reopen, seek and export pixels"; return false; }
    inputs.materialFrameTexture.reset();
    if (renderer.renderPreview(reopened.sceneSnapshot,collection,{},inputs,{64,64},"native-gpu",replay,error))
    { error="Retained Surface collection must reject a missing required Frame"; return false; }
    error.clear();
    if (backend.render(scene,targetResources.resources,64,64,inputs).rendered)
    { error = "Missing required Frame texture must fail the native draw"; return false; }
    const auto secondBinding = binding(scene,reactiveFrameRequest(92),0);
    if (!secondBinding) return false;
    auto multi = std::make_shared<NativeFixtureSurfaceMaterialProgram>(*program);
    multi->objectPrograms.push_back(secondBinding->nativeProgram());
    const auto multiResources = backend.prepare(scene,multi);
    if (!multiResources.prepared) { error = multiResources.error; return false; }
    inputs.materialFrameTextures = {{{91,0},first.frame},{{92,0},later.frame}};
    const auto distinct = backend.render(scene,multiResources.resources,64,64,inputs);
    inputs.materialFrameTextures = {{{91,0},later.frame},{{92,0},first.frame}};
    const auto exchanged = backend.render(scene,multiResources.resources,64,64,inputs);
    inputs.materialFrameTextures = {{{91,0},first.frame},{{92,0},later.frame}};
    const auto multiSeek = backend.render(scene,multiResources.resources,64,64,inputs);
    if (!distinct.rendered || !exchanged.rendered || !multiSeek.rendered
        || readPixels(distinct.frame) == readPixels(exchanged.frame)
        || readPixels(distinct.frame) != readPixels(multiSeek.frame))
    { error = "Independent exact-object Frame leases must remain isolated and deterministic"; return false; }
    inputs.materialFrameTextures.erase({92,0});
    if (backend.render(scene,multiResources.resources,64,64,inputs).rendered)
    { error = "A missing second Frame cannot borrow the first object's lease"; return false; }
    inputs.materialFrameTextures.clear();
    NativeGeometryCoreCapabilitySource capabilities(backend);
    for (bool table : {false,true})
    {
        using namespace videowire::geometry;
        const auto value = table ? instanceMaterialFieldsFixture("materialIndex",true,false,error)
            : instanceAppearanceFixture(0,error);
        if (!value) return false;
        const auto contract = withAttributeContract(appearanceInstanceContract(),*value);
        const auto admitted = admitValue(*value,contract,error);
        if (!admitted) return false;
        geometrysurfacematerial::ProgramSet programs(table ? 2u : 1u);
        if (table)
        {
            programs.front().sourceMaterial = 9521;
            programs.front().material = tableSurfaceRequest(17,false);
            programs.back().sourceMaterial = 9522;
        }
        programs.back().material = reactiveFrameRequest();
        GeometryCorePlanRuntime runtime(capabilities);
        const auto wire = lowerRuntimePlan(contract,*admitted,geometrysurfacematerial::encodeSet(programs));
        const auto preview = runtime.admitPreview({1,1,1,9750,1,PlanUse::preview},wire,{}, {},error);
        const auto exported = runtime.admitExport({1,1,1,9750,1,PlanUse::exportRender},wire,{}, {},error);
        if (!preview || !exported) return false;
        const auto draw = [&](const RuntimeAdmission& admission,const auto& image) {
            NativeFixtureSceneRuntimeInputs frameInputs; frameInputs.materialFrameTexture = image;
            return executeNativeGeometry(backend,admission,64,64,error,false,frameInputs,nullptr,{},0);
        };
        const auto initial = draw(*preview,first.frame), refreshed = draw(*preview,later.frame);
        const auto replay = draw(*preview,first.frame), offline = draw(*exported,later.frame);
        if (!initial || !refreshed || !replay || !offline) return false;
        const auto initialPixels = readPixels(initial->frame), refreshedPixels = readPixels(refreshed->frame);
        if (initialPixels.empty() || initialPixels == refreshedPixels || initialPixels != readPixels(replay->frame)
            || refreshedPixels != readPixels(offline->frame) || initial->drawnInstanceIds != refreshed->drawnInstanceIds)
        { error = "Geometry Surface Frame must refresh exact table/instance targets and match seek/export pixels"; return false; }
    }
    return true;
}
