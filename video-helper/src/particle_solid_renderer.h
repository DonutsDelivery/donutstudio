#pragma once

#include "particle_solid_state.h"
#include "geometry_core_backend.h"

namespace videorender
{
struct ParticleSolidFrame
{
    videohelper::geometry::PlanOwnerIdentity owner;
    std::shared_ptr<const ParticleSolidState> state;
    std::shared_ptr<const ParticleGeometryBinding> geometry;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
    std::shared_ptr<const arbitgpu::NativeFixtureSceneResources> resources;
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame;
};

// Scene object IDs are bounded draw slots, never truncated note/Geometry IDs.
// The retained state and binding carry the exact signed note and unsigned
// Geometry identities for every slot throughout the compositor lease.
inline std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> particleSolidScene(
    const ParticleParams& p,const ParticleSolidState& state,std::uint32_t nodeId,std::string& error)
{
    using namespace HarmonicMIDI::grid;
    const int contacts=(state.colliders.meshIdentity ? 1:0)+state.colliders.pointCount;
    if (state.count<0 || state.count>64 || contacts<0 || state.colliders.pointCount>64
        || state.count+contacts>int(Visual3DScene::kMaxObjects) || nodeId==0) {
        error="Solid 3D draw exceeds 64 body and collider objects; reduce Count or collider points";
        return {};
    }
    auto scene=std::make_shared<Visual3DScene>();
    scene->id={nodeId}; scene->activeCamera={1}; scene->ambientColor={0.16f,0.16f,0.16f};
    // A faceted unit sphere keeps the collider visibly three-dimensional even
    // with no mesh input. Rigid Geometry replaces its appearance, not inertia.
    const SceneVec3 sphere[]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    const std::uint32_t sphereIndices[]={0,2,4,4,2,1,1,2,5,5,2,0,0,4,3,4,1,3,1,5,3,5,0,3};
    for (const auto& v:sphere) { auto& out=scene->vertices[scene->vertexCount++]; out.position=out.normal=v; }
    for (auto index:sphereIndices) scene->indices[scene->indexCount++]=index;
    const std::uint32_t sphereVertices=scene->vertexCount,sphereIndexCount=scene->indexCount;
    std::uint32_t bodyFirstVertex=0,bodyFirstIndex=0,bodyVertices=sphereVertices,bodyIndices=sphereIndexCount;
    if (p.geometryBinding && p.geometryBinding->body) {
        const auto& geometry=std::get<videowire::geometry::GeometryData>(p.geometryBinding->body->data);
        if (geometry.positions.empty() || geometry.positions.size()>256 || geometry.indices.empty() || geometry.indices.size()>384) {
            error="Rigid body appearance requires 1 to 256 vertices and 1 to 128 triangles"; return {};
        }
        auto low=geometry.positions.front(),high=low;
        for (const auto& v:geometry.positions) {
            low.x=std::min(low.x,v.x); low.y=std::min(low.y,v.y); low.z=std::min(low.z,v.z);
            high.x=std::max(high.x,v.x); high.y=std::max(high.y,v.y); high.z=std::max(high.z,v.z);
        }
        const SceneVec3 center{(low.x+high.x)*0.5f,(low.y+high.y)*0.5f,(low.z+high.z)*0.5f};
        float radius=0;
        for (const auto& v:geometry.positions) radius=std::max(radius,std::hypot(std::hypot(v.x-center.x,v.y-center.y),v.z-center.z));
        if (!std::isfinite(radius) || radius<=1.0e-7f) { error="Rigid body mesh has no finite bounding sphere"; return {}; }
        bodyFirstVertex=scene->vertexCount; bodyFirstIndex=scene->indexCount;
        bodyVertices=geometry.positions.size(); bodyIndices=geometry.indices.size();
        for (const auto& v:geometry.positions)
            scene->vertices[scene->vertexCount++].position={(v.x-center.x)/radius,(v.y-center.y)/radius,(v.z-center.z)/radius};
        for (auto index:geometry.indices) {
            if (index>=bodyVertices) { error="Rigid body mesh contains an invalid vertex index"; return {}; }
            scene->indices[scene->indexCount++]=index;
        }
        for (std::size_t i=0;i<geometry.indices.size();i+=3) {
            const auto a=geometry.indices[i],b=geometry.indices[i+1],c=geometry.indices[i+2];
            const auto& x=scene->vertices[bodyFirstVertex+a].position;
            const auto& y=scene->vertices[bodyFirstVertex+b].position;
            const auto& z=scene->vertices[bodyFirstVertex+c].position;
            const auto normal=solidCross({y.x-x.x,y.y-x.y,y.z-x.z},{z.x-x.x,z.y-x.y,z.z-x.z});
            for (auto index:{a,b,c}) {
                auto& n=scene->vertices[bodyFirstVertex+index].normal;
                n.x+=normal[0]; n.y+=normal[1]; n.z+=normal[2];
            }
        }
        for (std::uint32_t i=0;i<bodyVertices;++i) {
            auto& n=scene->vertices[bodyFirstVertex+i].normal;
            const float length=std::hypot(std::hypot(n.x,n.y),n.z);
            if (length>1.0e-7f) { n.x/=length; n.y/=length; n.z/=length; } else n={0,1,0};
        }
    }
    scene->materials[0].id={1}; scene->materials[0].baseColor={p.red,p.green,p.blue};
    scene->materials[0].opacity=p.alpha; scene->materials[0].roughness=0.5f;
    scene->materials[0].alphaMode=p.alpha<1 ? SceneAlphaMode::Blend:SceneAlphaMode::Opaque;
    scene->materials[1].id={2}; scene->materials[1].baseColor={0.3f,0.34f,0.4f};
    scene->materialCount=2;
    const auto object=[&](SceneVec3 position,SceneVec3 scale,std::uint32_t firstVertex,std::uint32_t vertices,
                          std::uint32_t firstIndex,std::uint32_t indices,std::uint32_t material)->SceneObjectRecord& {
        auto& out=scene->objects[scene->objectCount++]; out.id={static_cast<std::uint32_t>(scene->objectCount)};
        out.material={material}; out.firstVertex=firstVertex; out.vertexCount=vertices; out.firstIndex=firstIndex; out.indexCount=indices;
        out.transform.translation={position.x-0.5f,position.y-0.5f,position.z-0.5f}; out.transform.scale=scale;
        return out;
    };
    for (int i=0;i<state.count;++i) {
        const auto& body=state.bodies[i];
        if (!std::isfinite(body.x) || !std::isfinite(body.y) || !std::isfinite(body.z)
            || std::abs(body.x)>352 || std::abs(body.y)>352 || std::abs(body.z)>352) {
            error="Solid body exceeds the bounded replay coordinate range"; return {};
        }
        auto& out=object({body.x,body.y,body.z},{p.bodyRadius,p.bodyRadius,p.bodyRadius},bodyFirstVertex,bodyVertices,bodyFirstIndex,bodyIndices,1);
        const auto& q=body.orientation; out.transform.rotation={q[0],q[1],q[2],q[3]};
    }
    if (state.colliders.meshIdentity) {
        const auto& low=state.colliders.minimum; const auto& high=state.colliders.maximum;
        const auto firstVertex=static_cast<std::uint32_t>(scene->vertexCount),firstIndex=static_cast<std::uint32_t>(scene->indexCount);
        for (int i=0;i<8;++i) {
            auto& v=scene->vertices[scene->vertexCount++]; v.position={i&1 ? 1.0f:-1.0f,i&2 ? 1.0f:-1.0f,i&4 ? 1.0f:-1.0f};
            v.normal={v.position.x*0.57735027f,v.position.y*0.57735027f,v.position.z*0.57735027f};
        }
        const std::uint32_t indices[]={0,2,1,1,2,3,4,5,6,5,7,6,0,1,4,1,5,4,2,6,3,3,6,7,0,4,2,2,4,6,1,3,5,3,7,5};
        for (auto index:indices) scene->indices[scene->indexCount++]=index;
        object({(low[0]+high[0])*0.5f,(low[1]+high[1])*0.5f,(low[2]+high[2])*0.5f},
            {std::max(0.00001f,(high[0]-low[0])*0.5f),std::max(0.00001f,(high[1]-low[1])*0.5f),std::max(0.00001f,(high[2]-low[2])*0.5f)},
            firstVertex,8,firstIndex,36,2);
    }
    for (int i=0;i<state.colliders.pointCount;++i) {
        const auto& point=state.colliders.solidPoints[i]; const auto r=std::max(0.00001f,p.colliderThickness);
        object({point[0],point[1],point[2]},{r,r,r},0,sphereVertices,0,sphereIndexCount,2);
    }
    scene->lights[0].id={1}; scene->lights[0].kind=SceneLightKind::Directional;
    scene->lights[0].transform.rotation={-0.25f,0.25f,0,0.9354143f}; scene->lights[0].intensity=1.5f; scene->lightCount=1;
    scene->cameras[0].id={1}; scene->cameras[0].transform.translation={0,0,2.2f};
    scene->cameras[0].verticalFovRadians=0.7853981634f; scene->cameras[0].nearPlane=0.05f; scene->cameras[0].farPlane=400;
    scene->cameraCount=1;
    if (scene->objectCount && !arbitgpu::nativeFixtureSceneTopologySupported(*scene)) {
        error="Solid replay does not satisfy the native retained-scene contract"; return {};
    }
    return scene;
}

template <typename Layer>
inline bool renderParticleSolids(const ParticleParams& p,const videohelper::geometry::PlanOwnerIdentity& owner,
                                int nodeId,int width,int height,Layer& layer,std::string& error,bool linearColor=false)
{
    if (!owner.valid() || !p.preparedSolids || nodeId<=0 || width<=0 || height<=0) {
        error="Solid 3D requires a prepared replay, exact preview/export owner and output extent"; return false;
    }
    auto scene=particleSolidScene(p,*p.preparedSolids,static_cast<std::uint32_t>(nodeId),error);
    if (!scene) return false;
    if (scene->objectCount==0) { layer.opacity=0; layer.particleSource=false; return true; }
    auto& backend=arbitgpu::nativeFixtureSceneBackend();
    auto prepared=backend.prepare(scene);
    if (!prepared.prepared || !prepared.resources) { error=prepared.error; return false; }
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs; inputs.linearColor=linearColor;
    auto rendered=backend.render(scene,prepared.resources,width,height,inputs);
    if (!rendered.rendered || !rendered.frame) { error=rendered.error; return false; }
    auto retained=std::make_shared<const ParticleSolidFrame>(ParticleSolidFrame{
        owner,p.preparedSolids,p.geometryBinding,std::move(scene),std::move(prepared.resources),std::move(rendered.frame)});
    const auto& frame=retained->frame; const auto view=frame->colorTextureViewHandle();
    if (frame->backend()=="opengl" && view>std::numeric_limits<unsigned>::max()) {
        error="Solid 3D texture exceeds compositor handle bounds"; return false;
    }
    layer.texture=frame->backend()=="opengl" ? static_cast<unsigned>(view):0;
    layer.nativeTextureBackend=frame->backend(); layer.nativeTextureView=view;
    layer.nativeTextureDescriptor=frame->colorTextureDescriptor(); layer.nativeTextureOwner=std::move(retained);
    layer.texWidth=width; layer.texHeight=height; layer.particleSource=false;
    return true;
}
}
