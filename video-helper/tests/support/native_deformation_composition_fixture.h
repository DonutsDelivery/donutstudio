#pragma once

#include "fixture_scene.h"
#include "gpu_backend/backend.h"
#include "gltf_glb.h"
#include "glb_scene_adapter.h"
#include "sha256.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace arbitgpu
{
// Only this named test friend may mint fixture capabilities. Production minting
// remains with ImportedAnimatedScenePayloadExecution's private admission join.
class NativeDeformationCompositionFixture final
{
public:
    static auto owner(visualmodelassetpayload::PayloadPtr payload,
        std::shared_ptr<const videohelper::gltf::GlbStaticMeshDocument> document,
        std::shared_ptr<const videohelper::gltf::GlbAnimationDocument> animation,
        std::shared_ptr<const NativeDeformationScene> source,
        std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> procedural,
        std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> composed,
        const std::vector<NativeDeformationCompositionMapping>& mappings,std::string& error)
    {
        const auto epoch=source?source->evaluationRevision:0;
        return NativeDeformationCompositionOwner::admit(std::move(payload),std::move(document),std::move(animation),
            0,1,std::move(source),std::move(procedural),std::move(composed),mappings,11,12,error,epoch);
    }
    static auto frame(std::shared_ptr<const NativeDeformationCompositionOwner> owner,
        std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot> snapshot,
        std::string& error,std::uint64_t project=11,std::uint64_t helper=12,
        std::shared_ptr<const visualanimation::Sample> scene={})
    { return snapshot ? NativeDeformationCompositionFrame::admit(std::move(owner),snapshot,std::move(scene),snapshot->time(),project,helper,error)
        : std::shared_ptr<const NativeDeformationCompositionFrame>{}; }
    // Multi-leaf capabilities use the same project and helper generations.
    static auto leafOwner(std::vector<NativeDeformationCompositionLeaf> leaves,
        std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> procedural,
        std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> composed,
        const std::vector<NativeDeformationCompositionMapping>& mappings,std::string& error,std::uint64_t epoch=77)
    { return NativeDeformationCompositionOwner::admit(std::move(leaves),std::move(procedural),std::move(composed),
        mappings,11,12,error,epoch); }
    static auto leafFrame(std::shared_ptr<const NativeDeformationCompositionOwner> owner,
        std::vector<NativeDeformationCompositionLeafSample> samples,visualdeformation::RationalFrameTime time,
        std::string& error)
    { return NativeDeformationCompositionFrame::admit(std::move(owner),std::move(samples),time,11,12,error); }
};
}

namespace videohelper::fixture3d
{
using namespace visualdeformation;
namespace animation=visualanimation;
enum class CompositionSkinCase { Node, IndependentRotation, IndependentReflection, IndependentNonuniform, SingularBlend, NonfiniteInverse };

// Original cube, zero morph deltas and authored channels are encoded into one
// real GLB. Both documents and the native rest draw decode these exact bytes.
inline std::vector<std::uint8_t> compositionGlb(bool animateScale,bool unskinned,CompositionSkinCase skinCase)
{
    using Json=nlohmann::json;
    auto cube=std::make_unique<HarmonicMIDI::grid::Visual3DScene>(makeScene());
    Json root={{"asset",{{"version","2.0"},{"generator","DonutStudio original composition fixture"}}},
        {"scene",0},{"scenes",Json::array({{{"nodes",Json::array({0})}}})},
        {"nodes",Json::array({{{"name","Original cube"},{"mesh",0}}})},
        {"bufferViews",Json::array()},{"accessors",Json::array()}};
    std::vector<std::uint8_t> bin;
    const auto appendWord=[](std::vector<std::uint8_t>& bytes,std::uint32_t value) {
        for (unsigned shift=0;shift<32;shift+=8) bytes.push_back(static_cast<std::uint8_t>(value>>shift));
    };
    const auto accessor=[&](const std::vector<float>& values,std::size_t count,const char* type,bool vertices) {
        const auto offset=bin.size(),view=root["bufferViews"].size(),index=root["accessors"].size();
        for (float value:values) {std::uint32_t bits=0;std::memcpy(&bits,&value,4);appendWord(bin,bits);}
        Json record={{"buffer",0},{"byteOffset",offset},{"byteLength",bin.size()-offset}};
        if (vertices) record["target"]=34962;
        root["bufferViews"].push_back(record);
        root["accessors"].push_back({{"bufferView",view},{"componentType",5126},{"count",count},{"type",type}});
        return index;
    };
    std::vector<float> positions,normals,tangents,uv,weights;
    for (std::size_t v=0;v<24;++v) {
        auto vertex=cube->vertices[v];
        if (skinCase==CompositionSkinCase::IndependentNonuniform)
        {
            const float c=.7071067811865476f;
            const auto rotate=[&](HarmonicMIDI::grid::SceneVec3 value) {
                return HarmonicMIDI::grid::SceneVec3 {c*(value.x-value.y),c*(value.x+value.y),value.z};
            };
            const HarmonicMIDI::grid::SceneVec3 tangent=vertex.normal.x!=0?
                HarmonicMIDI::grid::SceneVec3 {0,-vertex.normal.x,0}:HarmonicMIDI::grid::SceneVec3 {1,0,0};
            vertex.position=rotate(vertex.position);vertex.normal=rotate(vertex.normal);
            const auto rotated=rotate(tangent);vertex.tangent={rotated.x,rotated.y,rotated.z,1};
        }
        const auto& n=vertex.normal;
        positions.insert(positions.end(),{vertex.position.x,vertex.position.y,vertex.position.z});
        normals.insert(normals.end(),{n.x,n.y,n.z});
        // Each tangent is perpendicular to its authored face normal.
        if (skinCase==CompositionSkinCase::IndependentNonuniform)
            tangents.insert(tangents.end(),vertex.tangent.begin(),vertex.tangent.end());
        else
            tangents.insert(tangents.end(),n.x!=0?std::initializer_list<float>{0,0,1,1}:
                std::initializer_list<float>{1,0,0,1});
        uv.insert(uv.end(),{vertex.uv.x,vertex.uv.y});
        weights.insert(weights.end(),(skinCase==CompositionSkinCase::SingularBlend || skinCase==CompositionSkinCase::NonfiniteInverse)?
            std::initializer_list<float>{.5f,.5f,0,0}:std::initializer_list<float>{1,0,0,0});
    }
    Json attributes={{"POSITION",accessor(positions,24,"VEC3",true)},
        {"NORMAL",accessor(normals,24,"VEC3",true)},{"TANGENT",accessor(tangents,24,"VEC4",true)},
        {"TEXCOORD_0",accessor(uv,24,"VEC2",true)}};
    const float extent=skinCase==CompositionSkinCase::IndependentNonuniform?1.4142135623730951f:1.0f;
    root["accessors"][0]["min"]={-extent,-extent,-1.0f};
    root["accessors"][0]["max"]={extent,extent,1.0f};
    if (!unskinned) {
        const auto jointOffset=bin.size(),jointView=root["bufferViews"].size(),jointAccessor=root["accessors"].size();
        for (std::size_t v=0;v<24;++v)
            bin.insert(bin.end(),{0,static_cast<std::uint8_t>(skinCase==CompositionSkinCase::SingularBlend || skinCase==CompositionSkinCase::NonfiniteInverse?1:0),0,0});
        root["bufferViews"].push_back({{"buffer",0},{"byteOffset",jointOffset},{"byteLength",96},{"target",34962}});
        root["accessors"].push_back({{"bufferView",jointView},{"componentType",5121},{"count",24},{"type","VEC4"}});
        attributes["JOINTS_0"]=jointAccessor;attributes["WEIGHTS_0"]=accessor(weights,24,"VEC4",true);
        root["nodes"][0]["skin"]=0;
        const bool independent=skinCase!=CompositionSkinCase::Node;
        if (independent) {
            root["nodes"].push_back({{"name","Independent skin joint"}});
            root["scenes"][0]["nodes"].push_back(1);
        }
        if (skinCase==CompositionSkinCase::NonfiniteInverse) {
            root["nodes"].push_back({{"name","Independent opposing skin joint"}});
            root["scenes"][0]["nodes"].push_back(2);
        }
        root["skins"]=Json::array({{{"joints",skinCase==CompositionSkinCase::SingularBlend?
            Json::array({0,1}):skinCase==CompositionSkinCase::NonfiniteInverse?Json::array({1,2}):Json::array({independent?1:0})}}});
    }
    const auto zero=accessor(std::vector<float>(72,0),24,"VEC3",true);
    const auto indexOffset=bin.size(),indexView=root["bufferViews"].size(),indexAccessor=root["accessors"].size();
    for (std::size_t i=0;i<36;++i) appendWord(bin,cube->indices[i]);
    root["bufferViews"].push_back({{"buffer",0},{"byteOffset",indexOffset},{"byteLength",144},{"target",34963}});
    root["accessors"].push_back({{"bufferView",indexView},{"componentType",5125},{"count",36},{"type","SCALAR"}});
    root["meshes"]=Json::array({{{"weights",Json::array({.125})},{"primitives",Json::array({
        {{"attributes",attributes},{"indices",indexAccessor},{"material",0},
         {"targets",Json::array({{{"POSITION",zero},{"NORMAL",zero}}})}}})}}});
    // Existing original test PNG with two RGBA texels supplies a visibly tilted
    // normal map. Keeping it in the GLB makes the rotated-skin TBN oracle visible.
    const std::vector<std::uint8_t> png {
        137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82,0,0,0,2,0,0,0,1,8,6,0,0,0,244,34,127,138,
        0,0,0,17,73,68,65,84,120,156,99,248,207,192,240,159,225,63,67,3,0,16,121,3,126,33,192,253,141,
        0,0,0,0,73,69,78,68,174,66,96,130};
    const auto imageView=root["bufferViews"].size(),imageOffset=bin.size();bin.insert(bin.end(),png.begin(),png.end());
    root["bufferViews"].push_back({{"buffer",0},{"byteOffset",imageOffset},{"byteLength",png.size()}});
    while (bin.size()%4) bin.push_back(0);
    root["images"]=Json::array({{{"bufferView",imageView},{"mimeType","image/png"}}});
    root["samplers"]=Json::array({{{"magFilter",9728},{"minFilter",9728},{"wrapS",33071},{"wrapT",33071}}});
    root["textures"]=Json::array({{{"source",0},{"sampler",0}}});
    root["materials"]=Json::array({{{"pbrMetallicRoughness",{{"baseColorFactor",{.72,.82,1.0,1.0}},
        {"metallicFactor",.15},{"roughnessFactor",.4}}},{"normalTexture",{{"index",0}}}}});
    const auto time=accessor({0,1},2,"SCALAR",false);
    Json samplers=Json::array(),channels=Json::array();
    if (!unskinned || animateScale) {
        const bool rotate=skinCase==CompositionSkinCase::IndependentRotation;
        const bool reflect=skinCase==CompositionSkinCase::IndependentReflection || skinCase==CompositionSkinCase::SingularBlend;
        const auto output=accessor(rotate?std::vector<float>{0,0,0,1,0,0,.7071067811865476f,.7071067811865476f}:
            reflect?std::vector<float>{1,1,1,-1,1,1}:
            skinCase==CompositionSkinCase::NonfiniteInverse?std::vector<float>{1,1,1,1.0e-35f,1,1}:
            skinCase==CompositionSkinCase::IndependentNonuniform?std::vector<float>{1,1,1,2,1,1}:animateScale?std::vector<float>{1,1,1,-2,3,4}:
            std::vector<float>{0,0,0,2,0,0},2,rotate?"VEC4":"VEC3",false);
        samplers.push_back({{"input",time},{"output",output},{"interpolation","LINEAR"}});
        channels.push_back({{"sampler",0},{"target",{{"node",skinCase==CompositionSkinCase::Node?0:1},
            {"path",rotate?"rotation":(reflect || animateScale || skinCase==CompositionSkinCase::IndependentNonuniform || skinCase==CompositionSkinCase::NonfiniteInverse)?"scale":"translation"}}}});
    }
    if (skinCase==CompositionSkinCase::NonfiniteInverse) {
        const auto sampler=samplers.size();
        samplers.push_back({{"input",time},{"output",accessor({1,1,1,-1.0e-35f+1.0e-41f,1,1},2,"VEC3",false)},
            {"interpolation","LINEAR"}});
        channels.push_back({{"sampler",sampler},{"target",{{"node",2},{"path","scale"}}}});
    }
    const auto morphSampler=samplers.size();
    samplers.push_back({{"input",time},{"output",accessor({0,.5f},2,"SCALAR",false)},{"interpolation","LINEAR"}});
    channels.push_back({{"sampler",morphSampler},{"target",{{"node",0},{"path","weights"}}}});
    root["animations"]=Json::array({{{"name","Original composition action"},{"samplers",samplers},{"channels",channels}}});
    root["buffers"]=Json::array({{{"byteLength",bin.size()}}});
    auto json=root.dump();while (json.size()%4) json.push_back(' ');
    std::vector<std::uint8_t> bytes;
    appendWord(bytes,0x46546c67u);appendWord(bytes,2);appendWord(bytes,static_cast<std::uint32_t>(28+json.size()+bin.size()));
    appendWord(bytes,static_cast<std::uint32_t>(json.size()));appendWord(bytes,0x4e4f534au);bytes.insert(bytes.end(),json.begin(),json.end());
    appendWord(bytes,static_cast<std::uint32_t>(bin.size()));appendWord(bytes,0x004e4942u);bytes.insert(bytes.end(),bin.begin(),bin.end());
    return bytes;
}

struct NativeCompositionFixture
{
    std::shared_ptr<const arbitgpu::NativeDeformationScene> source;
    std::shared_ptr<HarmonicMIDI::grid::Visual3DScene> procedural,composed;
    std::shared_ptr<const videohelper::gltf::GlbStaticMeshDocument> document;
    std::shared_ptr<const videohelper::gltf::GlbAnimationDocument> animationDocument;
    std::shared_ptr<const animation::Clip> clip;
    std::vector<JointAnimationBindingView> bindings;
    std::vector<MorphAnimationBindingView> morphBindings;
    std::vector<animation::TargetId> sceneTargets;
    std::vector<arbitgpu::NativeDeformationCompositionMapping> mappings;
    visualmodelassetpayload::PayloadPtr payload;
    std::shared_ptr<const arbitgpu::NativeDeformationCompositionOwner> owner;
    CompositionSkinCase skinCase;
    std::string error;

    explicit NativeCompositionFixture(bool importedParent,bool animateScale=false,bool unskinned=false,
        CompositionSkinCase kind=CompositionSkinCase::Node):skinCase(kind)
    {
        using namespace HarmonicMIDI::grid;
        auto bytes=compositionGlb(animateScale,unskinned,skinCase);
        visualanimationimport::ExactContentAssetKey key {"original-native-composition",1,
            videohelper::sha256Bytes(bytes.data(),bytes.size()),"model/gltf-binary",bytes.size()};
        payload=visualmodelassetpayload::ImmutablePayload::create(std::move(key),std::move(bytes));
        videohelper::gltf::GlbAdmissionOptions options;options.admitAnimations=true;options.admitSkins=true;options.sceneIndex=0;
        auto decoded=videohelper::gltf::decodeAnimatedGlbBaseScene(payload->bytes(),options,error);
        if (!decoded) return;
        document=std::make_shared<const videohelper::gltf::GlbStaticMeshDocument>(std::move(*decoded));
        videohelper::gltf::GlbAnimationDecodeOptions animationOptions;animationOptions.admission=options;
        // Retain authored skin hierarchy. The morph-only control also exercises
        // the decoder's original unskinned animation contract, without rigid skins.
        animationOptions.retainGeometryHierarchy=!unskinned;
        auto animated=videohelper::gltf::decodeGlbAnimations(payload->bytes(),animationOptions,error);
        if (!animated || animated->clips.empty() || animated->renderBindings.empty()) return;
        animationDocument=std::make_shared<const videohelper::gltf::GlbAnimationDocument>(std::move(*animated));
        auto adapted=videohelper::gltf::adaptAnimatedGlbMeshToVisual3DSceneHeap(*document,error,0);
        if (!adapted) return;
        auto rest=std::shared_ptr<Visual3DScene>(std::move(adapted));
        auto staging=std::make_unique<Visual3DScene>(makeScene());
        rest->cameras[0]=staging->cameras[0];rest->cameraCount=1;rest->activeCamera=staging->activeCamera;
        rest->lights[0]=staging->lights[0];rest->lightCount=1;rest->ambientColor=staging->ambientColor;
        const auto& named=animationDocument->clips[0];const auto& render=animationDocument->renderBindings[0];
        clip=named.clip;
        auto draw=std::make_shared<arbitgpu::NativeDeformationScene>();
        draw->sourceStableId=801;draw->deformationStableId=802;draw->structuralRevision=77;
        draw->evaluationRevision=77;draw->clip=clip->id();
        draw->scene=rest;draw->deformation=animationDocument->deformation;draw->mesh=render.mesh;
        draw->object=rest->objects[0].id;draw->animationNodeStableId=render.nodeIndex+1;draw->batchMember=true;
        draw->retainDeformedGeometry=true;
        if (render.skin.isValid()) draw->skin=render.skin;
        draw->morphBaseWeights=render.morphBaseWeights;
        for (const auto& base:render.jointBaseTransforms)
            draw->jointBaseTransforms.push_back({base.skin,base.joint,base.translation,base.rotation,base.scale,base.matrix});
        source=draw;
        for (const auto& binding:named.jointBindings) bindings.push_back({binding.animationTarget,binding.skin,binding.joint});
        for (const auto& binding:named.morphBindings)
            morphBindings.push_back({binding.animationTarget,binding.mesh,binding.targets.data(),binding.targets.size()});
        sceneTargets=named.sceneTransformTargets;
        if (unskinned && animateScale) sceneTargets.push_back({1});
        procedural=std::make_shared<Visual3DScene>(*rest);
        procedural->objects[0].id={65537}; // Must never be interpreted as GLB node 2.
        procedural->objects[0].transform.translation.x=.8f;
        composed=std::make_shared<Visual3DScene>(*rest);
        composed->objectCount=2;composed->vertexCount=48;composed->indexCount=72;
        composed->materialCount=2;composed->materials[1]=procedural->materials[0];composed->materials[1].id={9000};
        composed->objects[1]=procedural->objects[0];composed->objects[1].material={9000};composed->objects[1].id={900};
        composed->objects[1].firstVertex=24;composed->objects[1].firstIndex=36;
        for (std::size_t i=0;i<24;++i) composed->vertices[24+i]=procedural->vertices[i];
        for (std::size_t i=0;i<36;++i) composed->indices[36+i]=procedural->indices[i];
        if (importedParent) composed->objects[1].parent={1};
        else composed->objects[0].parent={900};
        using Mapping=arbitgpu::NativeDeformationCompositionMapping;
        mappings={{Mapping::Kind::Imported,{1},{1},1},{Mapping::Kind::Procedural,{65537},{900},0}};
        owner=admit(composed,mappings);
    }
    std::shared_ptr<const arbitgpu::NativeDeformationCompositionOwner> admit(std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene,
        const std::vector<arbitgpu::NativeDeformationCompositionMapping>& map)
    { return arbitgpu::NativeDeformationCompositionFixture::owner(payload,document,animationDocument,source,
        procedural,std::move(scene),map,error); }
    // clipSeconds samples the clip at its own time for the same presentation
    // frame, as a leaf with its own speed and trim does.
    std::shared_ptr<const AnimationDeformationSnapshot> snapshot(std::int64_t frame,bool selectedPose=false,bool weighted=false,
        std::optional<double> clipSeconds={})
    {
        if (!source || !source->deformation || !clip) return {};
        const AnimationDeformationBindingView view {bindings.data(),bindings.size(),
            morphBindings.data(),morphBindings.size(),sceneTargets.data(),sceneTargets.size()};
        AnimationDeformationRequest request; request.time={frame,30,1}; request.revision=source->structuralRevision;request.allowNodeTransformAndMorph=true;
        if (selectedPose) {request.pose.nodeStableId=1;request.pose.meshStableId=source->mesh.value;request.pose.boneStableId=1;
            request.pose.boneEnabled=true;request.pose.translation[0]=.5;}
        if (weighted) {request.combinationMode=animation::CombinationMode::WeightedBlend;request.combinationWeight=.5;}
        request.requestedTimeSeconds=clipSeconds;
        // Production consumers attest their exact payload. Multi-leaf frames require it.
        if (payload) visualdeformation::sha256Digest(payload->key().contentSha256,request.sourceContentSha256);
        return evaluateAnimationDeformation(*source->deformation,*clip,view,request,{},error);
    }
    auto batch() const
    {
        auto result=std::make_shared<arbitgpu::NativeDeformationScene>(*source);
        result->scene=composed; result->compositionOwner=owner; result->batchMember=false; result->draws.clear();
        if (owner) for (std::size_t i=0; i<owner->memberCount(); ++i)
        {
            auto member=std::make_shared<arbitgpu::NativeDeformationScene>(*owner->originalMember(i));
            member->object=owner->memberObject(i); member->scene=composed; member->compositionOwner=owner;
            member->batchMember=true; result->draws.push_back(std::move(member));
        }
        return result;
    }
};

// Independent test-only local vertex oracle. The product keeps GPU deformation
// as its sole rendering authority. These fixture morph deltas are all zero.
inline std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> compositionReference(
    const NativeCompositionFixture& fixture,const arbitgpu::NativeDeformationScene& batch,
    const visualdeformation::AnimationDeformationSnapshot& snapshot,
    const std::shared_ptr<const arbitgpu::NativeDeformationCompositionFrame>& frame,std::string& error)
{
    auto scene=std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*fixture.composed);
    const auto transform=[](const auto& matrix,HarmonicMIDI::grid::SceneVec3 v,float w) {
        return HarmonicMIDI::grid::SceneVec3 {matrix[0]*v.x+matrix[4]*v.y+matrix[8]*v.z+matrix[12]*w,
            matrix[1]*v.x+matrix[5]*v.y+matrix[9]*v.z+matrix[13]*w,
            matrix[2]*v.x+matrix[6]*v.y+matrix[10]*v.z+matrix[14]*w};
    };
    const auto normalize=[](HarmonicMIDI::grid::SceneVec3 v) {
        const auto length=std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z);
        return HarmonicMIDI::grid::SceneVec3 {v.x/length,v.y/length,v.z/length};
    };
    const auto determinant3=[](const auto& m) {
        return m[0]*(m[5]*m[10]-m[9]*m[6])-m[4]*(m[1]*m[10]-m[9]*m[2])+m[8]*(m[1]*m[6]-m[5]*m[2]);
    };
    for (const auto& draw:batch.draws)
    {
        arbitgpu::NativeDeformationFrameData data;
        if (!arbitgpu::prepareNativeDeformationFrame(*draw,snapshot,frame,data,error)) return {};
        const auto object=std::find_if(scene->objects.begin(),scene->objects.begin()+scene->objectCount,
            [&](const auto& o){return o.id==draw->object;});
        for (std::size_t i=object->firstVertex;i<object->firstVertex+object->vertexCount;++i)
        {
            auto& vertex=scene->vertices[i];
            HarmonicMIDI::grid::SceneVec3 tangent {vertex.tangent[0],vertex.tangent[1],vertex.tangent[2]};
            float handedness=vertex.tangent[3];
            if (draw->animationNodeStableId==1 && fixture.skinCase!=CompositionSkinCase::Node)
            {
                // Literal independent endpoint oracle. No palette or post table
                // determines the rotation, reflection or nonuniform-scale result.
                if (snapshot.sampleTimeSeconds()!=1.0 || fixture.skinCase==CompositionSkinCase::SingularBlend
                    || fixture.skinCase==CompositionSkinCase::NonfiniteInverse)
                {error="literal skin oracle requires its admitted nonsingular endpoint";return {};}
                const auto literal=[&](HarmonicMIDI::grid::SceneVec3 value) {
                    return fixture.skinCase==CompositionSkinCase::IndependentRotation?
                        HarmonicMIDI::grid::SceneVec3{-value.y,value.x,value.z}:
                        fixture.skinCase==CompositionSkinCase::IndependentReflection?
                        HarmonicMIDI::grid::SceneVec3{-value.x,value.y,value.z}:
                        HarmonicMIDI::grid::SceneVec3{2*value.x,value.y,value.z};
                };
                vertex.position=literal(vertex.position);
                vertex.normal=normalize(fixture.skinCase==CompositionSkinCase::IndependentNonuniform?
                    HarmonicMIDI::grid::SceneVec3{.5f*vertex.normal.x,vertex.normal.y,vertex.normal.z}:literal(vertex.normal));
                tangent=normalize(literal(tangent));
                if (fixture.skinCase==CompositionSkinCase::IndependentReflection) handedness=-handedness;
                vertex.tangent={tangent.x,tangent.y,tangent.z,handedness};
                continue;
            }
            if (data.jointCount) {
                vertex.position=transform(data.jointPalette,vertex.position,1);
                const auto& skin=data.jointPalette;
                const float determinant=skin[0]*(skin[5]*skin[10]-skin[9]*skin[6])
                    -skin[4]*(skin[1]*skin[10]-skin[9]*skin[2])+skin[8]*(skin[1]*skin[6]-skin[5]*skin[2]);
                const auto n=vertex.normal;
                vertex.normal={
                    ((skin[5]*skin[10]-skin[6]*skin[9])*n.x+(skin[9]*skin[2]-skin[10]*skin[1])*n.y+(skin[1]*skin[6]-skin[2]*skin[5])*n.z)/determinant,
                    ((skin[6]*skin[8]-skin[4]*skin[10])*n.x+(skin[10]*skin[0]-skin[8]*skin[2])*n.y+(skin[2]*skin[4]-skin[0]*skin[6])*n.z)/determinant,
                    ((skin[4]*skin[9]-skin[5]*skin[8])*n.x+(skin[8]*skin[1]-skin[9]*skin[0])*n.y+(skin[0]*skin[5]-skin[1]*skin[4])*n.z)/determinant};
                tangent=transform(data.jointPalette,tangent,0);
                if (determinant<0) handedness=-handedness;
            }
            vertex.position=transform(data.compositionPost.point(),vertex.position,1);
            vertex.normal=normalize(transform(data.compositionPost.normal(),vertex.normal,0));
            tangent=normalize(transform(data.compositionPost.point(),tangent,0));
            vertex.tangent={tangent.x,tangent.y,tangent.z,handedness*data.compositionPost.normal()[15]};
        }
        // The static fixture path culls by the objectMatrix sign alone. A baked mirror (the literal
        // reflection, a reflected palette, or a reflected composition point) reverses the screen
        // winding, so its triangles are reversed here to keep the outer faces visible under that rule.
        const bool literalDraw=draw->animationNodeStableId==1 && fixture.skinCase!=CompositionSkinCase::Node;
        const bool mirrored=literalDraw ? fixture.skinCase==CompositionSkinCase::IndependentReflection
            : (data.jointCount!=0 && determinant3(data.jointPalette)<0.0f)!=(determinant3(data.compositionPost.point())<0.0f);
        if (mirrored)
            for (std::size_t t=object->firstIndex;t+2<object->firstIndex+object->indexCount;t+=3)
            {
                const auto swapped=scene->indices[t+1];
                scene->indices[t+1]=scene->indices[t+2];
                scene->indices[t+2]=swapped;
            }
    }
    return scene;
}

inline bool compositionMotion(const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame,
    bool moving,std::string& error,const char* call="composition frame",bool previousPresent=false)
{
    arbitgpu::NativeRawPassPixels motion,mask;
    if (!frame || !frame->readRawPass(renderpassoutput::Output::Motion,motion,error)
        || !frame->readRawPass(renderpassoutput::Output::Mask,mask,error)) return false;
    const auto count=std::size_t(frame->width())*frame->height();
    if (motion.format!=arbitgpu::NativeTexturePixelFormat::Rg16Float || motion.bytes.size()!=count*4
        || mask.format!=arbitgpu::NativeTexturePixelFormat::R8Unorm || mask.bytes.size()!=count)
    {error="composition Motion lacks exact native RG16F/Mask storage";return false;}
    const auto halfMagnitude=[](std::uint16_t bits) {
        const auto exponent=(bits>>10)&31u,fraction=bits&1023u;
        const double value=exponent==0 ? std::ldexp(double(fraction),-24)
            : std::ldexp(double(1024+fraction),int(exponent)-25);
        return std::abs(value);
    };
    bool geometry=false,displaced=false;
    double largest=0;
    std::size_t visible=0,movedPixels=0,firstVisible=count,firstDisplaced=count;
    for (std::size_t i=0;i<count;++i)
        if (mask.bytes[i]) {
            geometry=true;
            ++visible;
            if (firstVisible==count) firstVisible=i;
            bool pixelDisplaced=false;
            for (std::size_t component=0;component<2;++component) {
                std::uint16_t bits=0;std::memcpy(&bits,motion.bytes.data()+4*i+2*component,2);
                if (((bits>>10)&31u)==31u) {error="composition Motion contains a nonfinite displacement";return false;}
                pixelDisplaced=pixelDisplaced || (bits&0x7fffu)!=0;
                displaced=displaced || (bits&0x7fffu)!=0;
                largest=std::max(largest,halfMagnitude(bits));
            }
            if (pixelDisplaced) {++movedPixels; if (firstDisplaced==count) firstDisplaced=i;}
        }
    if (!geometry || displaced!=moving) {
        const auto num=[](double value) {
            char text[64];
            std::snprintf(text,sizeof text,"%.9g",value);
            return std::string(text);
        };
        const auto signedValue=[&](std::uint16_t bits) {
            const double magnitude=halfMagnitude(bits);
            return (bits&0x8000u) ? -magnitude : magnitude;
        };
        const std::size_t width=std::size_t(frame->width());
        // Moving frames report the first visible pixel; neutral frames report the first displaced pixel.
        const std::size_t offending=moving ? firstVisible : firstDisplaced;
        std::string detail=" first offending pixel none";
        if (offending<count) {
            std::uint16_t first=0,second=0;
            std::memcpy(&first,motion.bytes.data()+4*offending,2);
            std::memcpy(&second,motion.bytes.data()+4*offending+2,2);
            detail=" first offending pixel ("+std::to_string(offending%width)+","+std::to_string(offending/width)
                +") motion=("+num(signedValue(first))+","+num(signedValue(second))+")";
        }
        error=std::string("composition Motion violates neutral or moving visible geometry in ")+call
            +" (expected moving="+(moving?"yes":"no")+", visible geometry="+(geometry?"yes":"no")
            +", displaced="+(displaced?"yes":"no")+", largest |displacement|="+num(largest)
            +", visible pixels="+std::to_string(visible)+" of "+std::to_string(count)
            +", displaced pixels="+std::to_string(movedPixels)
            +", previous deformation="+(previousPresent?"yes":"no")+","+detail+")";
        return false;
    }
    return true;
}

inline bool compositionRetainedGeometry(const arbitgpu::NativeDeformationSubmission& rendered,
    const HarmonicMIDI::grid::Visual3DScene& reference,std::string& error)
{
    if (rendered.deformedVertices.size()!=reference.vertexCount*8u)
    {error="composition retained geometry omitted a disjoint member range";return false;}
    for (std::size_t i=0;i<reference.vertexCount;++i)
    {
        const auto& v=reference.vertices[i];
        const std::array<float,8> expected {v.position.x,v.position.y,v.position.z,v.normal.x,v.normal.y,v.normal.z,v.uv.x,v.uv.y};
        for (std::size_t component=0;component<8;++component)
            if (!std::isfinite(rendered.deformedVertices[i*8+component])
                || std::abs(rendered.deformedVertices[i*8+component]-expected[component])>1.0e-4f)
            {error="composition retained geometry differs from the admitted member reference";return false;}
    }
    return true;
}

template <typename ReadColor>
bool verifyNativeComposition(arbitgpu::NativeDeformationBackend& backend,ReadColor readColor,std::string& error)
{
    for (bool parent:{false,true}) for (bool nonuniform:{false,true})
    for (const auto kind:{CompositionSkinCase::Node,CompositionSkinCase::IndependentRotation,CompositionSkinCase::IndependentReflection,CompositionSkinCase::IndependentNonuniform})
    {
        if (nonuniform && kind!=CompositionSkinCase::Node) continue;
        NativeCompositionFixture fixture(parent,nonuniform,false,kind);
        if (!fixture.owner) {error=fixture.error;return false;}
        const auto source=fixture.batch();const auto now=fixture.snapshot(nonuniform || kind!=CompositionSkinCase::Node?30:15),
            before=fixture.snapshot(nonuniform || kind!=CompositionSkinCase::Node?29:14);
        const auto current=arbitgpu::NativeDeformationCompositionFixture::frame(fixture.owner,now,error);
        const auto previous=arbitgpu::NativeDeformationCompositionFixture::frame(fixture.owner,before,error);
        if (!current || !previous) return false;
        const auto prepared=backend.prepare(source,{});
        if (!prepared.prepared) {error=prepared.error;return false;}
        arbitgpu::NativeDeformationRuntimeInputs inputs;inputs.compositionFrame=current;
        inputs.rawExportMask=render3dimage::kRawExportMask;
        auto first=backend.render(source,now,prepared.resources,96,96,inputs);
        if (!first.rendered || first.stats.drawCount!=2 || first.stats.dispatchCount!=2
            || !compositionMotion(first.frame,false,error,"neutral first frame",inputs.previousMotion.has_value())) {if(error.empty())error=first.error;return false;}
        inputs.compositionFrame=previous;
        inputs.previousMotion=arbitgpu::sceneMotionSample(inputs,before);
        inputs.compositionFrame=current;
        auto moved=backend.render(source,now,prepared.resources,96,96,inputs);
        auto repeated=backend.render(source,now,prepared.resources,96,96,inputs);
        if (!moved.rendered || !repeated.rendered || moved.stats.drawCount!=2 || moved.stats.dispatchCount!=4
            || !compositionMotion(moved.frame,true,error,"moved frame",inputs.previousMotion.has_value()) || readColor(moved.frame)!=readColor(repeated.frame))
        {if(error.empty())error=moved.error;return false;}
        const auto reference=compositionReference(fixture,*source,*now,current,error);
        if (!reference || !compositionRetainedGeometry(moved,*reference,error)
            || !compositionRetainedGeometry(repeated,*reference,error)) return false;
        auto& staticBackend=arbitgpu::nativeFixtureSceneBackend();
        const auto staticResources=staticBackend.prepare(reference,{});
        if (!staticResources.prepared) {error=staticResources.error;return false;}
        const auto golden=staticBackend.render(reference,staticResources.resources,96,96,{});
        if (!golden.rendered) {error=golden.error;return false;}
        const auto actual=readColor(moved.frame),expected=readColor(golden.frame);
        if (actual.empty() || actual.size()!=expected.size()) {error="composition reference pixels are incomplete";return false;}
        for (std::size_t i=0;i<actual.size();++i)
            if (std::abs(int(actual[i])-int(expected[i]))>6)
            {error="composition full draw disagrees with the independent position/normal/tangent reference";return false;}
        if (!nonuniform && kind==CompositionSkinCase::Node)
        {
            // Selected imported-node controls affect its composed descendants.
            // Node zero also exercises whole-composition controls when procedural
            // geometry is the root. No procedural ID is used as a GLB node ID.
            for (const std::uint64_t selected:{std::uint64_t(1),std::uint64_t(0)})
            {
                auto controls=inputs;controls.objectNodeStableId=selected;
                controls.objectTranslationOffset={.25f,.1f,0};
                auto priorControls=controls;priorControls.objectTranslationOffset={-.2f,0,0};
                priorControls.compositionFrame=previous;
                controls.previousMotion=arbitgpu::sceneMotionSample(priorControls,before);
                const auto controlled=backend.render(source,now,prepared.resources,96,96,controls);
                auto expectedScene=std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*reference);
                // Literal selected import vs descendant expectation, independent
                // of runtimeTargetsObject and the renderer's uniform builder.
                for (std::size_t object=0;object<2;++object)
                    if (selected==0 || object==0 || parent)
                        for (std::size_t vertex=expectedScene->objects[object].firstVertex;
                            vertex<expectedScene->objects[object].firstVertex+expectedScene->objects[object].vertexCount;++vertex)
                        {expectedScene->vertices[vertex].position.x+=.25f;expectedScene->vertices[vertex].position.y+=.1f;}
                const auto expectedResources=staticBackend.prepare(expectedScene,{});
                const auto expectedControl=expectedResources.prepared?
                    staticBackend.render(expectedScene,expectedResources.resources,96,96,{}):arbitgpu::NativeFixtureSceneSubmission{};
                if (!controlled.rendered || !expectedControl.rendered || !compositionMotion(controlled.frame,true,error,"selected control frame",controls.previousMotion.has_value())
                    || !compositionRetainedGeometry(controlled,*reference,error))
                {if(error.empty())error="selected composition controls failed draw, Motion or local readback";return false;}
                const auto controlPixels=readColor(controlled.frame),expectedPixels=readColor(expectedControl.frame);
                if (controlPixels.empty() || controlPixels==actual || controlPixels.size()!=expectedPixels.size())
                {error="selected composition controls did not change the full draw";return false;}
                for (std::size_t i=0;i<controlPixels.size();++i)
                    if (std::abs(int(controlPixels[i])-int(expectedPixels[i]))>6)
                    {error="selected imported-parent controls affected the wrong composed member";return false;}
            }
        }
        auto foreign=fixture.admit(fixture.composed,fixture.mappings);
        inputs.compositionFrame=arbitgpu::NativeDeformationCompositionFixture::frame(foreign,now,error);
        const auto refused=backend.render(source,now,prepared.resources,96,96,inputs);
        if (refused.rendered || refused.frame) {error="composition GPU accepted a foreign typed frame";return false;}
        // Source failures must not poison the incumbent current/previous pair.
        inputs.compositionFrame=current;
        auto recovered=backend.render(source,now,prepared.resources,96,96,inputs);
        if (!recovered.rendered || readColor(recovered.frame)!=actual) {error="composition GPU refusal changed the incumbent draw";return false;}
    }
    for (const auto kind:{CompositionSkinCase::SingularBlend,CompositionSkinCase::NonfiniteInverse})
    {
        NativeCompositionFixture singular(true,false,false,kind);
        if (!singular.owner) {error=singular.error;return false;}
        const auto singularSource=singular.batch();const auto singularSnapshot=singular.snapshot(30);
        const auto singularFrame=arbitgpu::NativeDeformationCompositionFixture::frame(singular.owner,singularSnapshot,error);
        const auto singularPrepared=backend.prepare(singularSource,{});
        if (!singularFrame || !singularPrepared.prepared) {error=singularPrepared.error;return false;}
        arbitgpu::NativeDeformationRuntimeInputs singularInputs;singularInputs.compositionFrame=singularFrame;
        const auto refusedSkin=backend.render(singularSource,singularSnapshot,singularPrepared.resources,96,96,singularInputs);
        if (refusedSkin.rendered || refusedSkin.frame || !refusedSkin.deformedVertices.empty()
            || refusedSkin.stats.dispatchCount!=0)
        {error="undefined weighted skin reached native dispatch or published geometry";return false;}
    }
    // The newly added compute post block is disabled for every legacy call.
    NativeCompositionFixture fixture(true);
    auto legacy=std::make_shared<arbitgpu::NativeDeformationScene>(*fixture.source);legacy->batchMember=false;
    const auto snapshot=fixture.snapshot(0);const auto prepared=backend.prepare(legacy,{});
    if (!prepared.prepared) {error=prepared.error;return false;}
    const auto rendered=backend.render(legacy,snapshot,prepared.resources,96,96,{});
    const auto goldenResources=arbitgpu::nativeFixtureSceneBackend().prepare(legacy->scene,{});
    if (!goldenResources.prepared) {error=goldenResources.error;return false;}
    const auto golden=arbitgpu::nativeFixtureSceneBackend().render(legacy->scene,goldenResources.resources,96,96,{});
    if (!rendered.rendered || !golden.rendered || readColor(rendered.frame)!=readColor(golden.frame))
    {error="disabled composition post changed the legacy rest draw";return false;}
    for (const auto kind:{CompositionSkinCase::IndependentRotation,CompositionSkinCase::IndependentReflection,
        CompositionSkinCase::IndependentNonuniform})
    {
        NativeCompositionFixture original(true,false,false,kind);
        if (!original.owner) {error=original.error;return false;}
        auto disabled=std::make_shared<arbitgpu::NativeDeformationScene>(*original.source);disabled->batchMember=false;
        const auto sample=original.snapshot(30);const auto resources=backend.prepare(disabled,{});
        if (!sample || !resources.prepared) {error=resources.error;return false;}
        const auto legacyResult=backend.render(disabled,sample,resources.resources,96,96,{});
        auto legacyReference=std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*disabled->scene);
        for (std::size_t i=0;i<legacyReference->vertexCount;++i)
        {
            auto& vertex=legacyReference->vertices[i];
            const auto linear=[&](HarmonicMIDI::grid::SceneVec3 value) {
                return kind==CompositionSkinCase::IndependentRotation?HarmonicMIDI::grid::SceneVec3{-value.y,value.x,value.z}:
                    kind==CompositionSkinCase::IndependentReflection?HarmonicMIDI::grid::SceneVec3{-value.x,value.y,value.z}:
                    HarmonicMIDI::grid::SceneVec3{2*value.x,value.y,value.z};
            };
            vertex.position=linear(vertex.position);vertex.normal=linear(vertex.normal);
            const auto length=std::sqrt(vertex.normal.x*vertex.normal.x+vertex.normal.y*vertex.normal.y+vertex.normal.z*vertex.normal.z);
            vertex.normal={vertex.normal.x/length,vertex.normal.y/length,vertex.normal.z/length};
            // Disabled legacy keeps authored tangent and handedness unchanged.
        }
        // A deformed draw flips its front face for a mirrored skin (commit a94f5c6d3b), while the
        // static fixture path culls by the objectMatrix sign alone. The literal reflection is a baked
        // mirror, so its triangles are reversed here, as the composition reference does.
        if (kind==CompositionSkinCase::IndependentReflection)
            for (std::size_t object=0;object<legacyReference->objectCount;++object)
                for (std::size_t t=legacyReference->objects[object].firstIndex;
                    t+2<legacyReference->objects[object].firstIndex+legacyReference->objects[object].indexCount;t+=3)
                {
                    const auto swapped=legacyReference->indices[t+1];
                    legacyReference->indices[t+1]=legacyReference->indices[t+2];
                    legacyReference->indices[t+2]=swapped;
                }
        const auto staticResources=arbitgpu::nativeFixtureSceneBackend().prepare(legacyReference,{});
        const auto staticResult=staticResources.prepared?arbitgpu::nativeFixtureSceneBackend().render(
            legacyReference,staticResources.resources,96,96,{}):arbitgpu::NativeFixtureSceneSubmission{};
        if (!legacyResult.rendered || !staticResult.rendered
            || !compositionRetainedGeometry(legacyResult,*legacyReference,error))
        {if(error.empty())error="disabled legacy animated geometry changed";return false;}
        const auto pixels=readColor(legacyResult.frame),expected=readColor(staticResult.frame);
        if (pixels.empty() || pixels.size()!=expected.size()) {error="disabled legacy animation has incomplete pixels";return false;}
        for (std::size_t i=0;i<pixels.size();++i)
            if (std::abs(int(pixels[i])-int(expected[i]))>6)
            {error="disabled legacy skin changed its normal, rest tangent or handedness";return false;}
    }
    return true;
}

} // namespace videohelper::fixture3d
