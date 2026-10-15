#pragma once

#include "particle_solid_state.h"
#include "particle_body_replay.h"
#include "geometry_core_backend.h"
#include "../../shared/GeometryTimelineSceneBinding.h"
#include "../../shared/Render3DImageOutput.h"

#include <type_traits>
#include <utility>

namespace videorender
{
template <typename Layer>
inline bool renderParticleSolids(const ParticleParams&,const videohelper::geometry::PlanOwnerIdentity&,
    int,int,int,Layer&,std::string&,bool linearColor=false);

class ParticleSolidFrame final
{
    const std::shared_ptr<const ParticleBodyCallerLease> callerLease_;
public:
    ParticleSolidFrame(const ParticleSolidFrame&)=delete;
    ParticleSolidFrame& operator=(const ParticleSolidFrame&)=delete;
    const videohelper::geometry::PlanOwnerIdentity owner;
    const std::shared_ptr<const ParticleSolidState> state;
    const std::shared_ptr<const ParticleGeometryBinding> geometry;
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneResources> resources;
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame;
    const std::shared_ptr<const ParticleBodyReplayReceipt> replay;
    const std::shared_ptr<const ParticleBodyGeometrySample> bodySample;
    static constexpr std::size_t callerLogicalBytes() noexcept
    {
        return sizeof(ParticleSolidFrame)+sizeof(arbitgpu::NativeFixtureScenePreparation)
            +sizeof(arbitgpu::NativeFixtureSceneSubmission)+sizeof(arbitgpu::NativeFixtureSceneRuntimeInputs)
            +2*sizeof(arbitgpu::NativeTextureViewDescriptor);
    }
private:
    ParticleSolidFrame(videohelper::geometry::PlanOwnerIdentity o,std::shared_ptr<const ParticleSolidState> s,
        std::shared_ptr<const ParticleGeometryBinding> g,std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> sceneIn,
        std::shared_ptr<const arbitgpu::NativeFixtureSceneResources> resourcesIn,
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frameIn,std::shared_ptr<const ParticleBodyReplayReceipt> r,
        std::shared_ptr<const ParticleBodyCallerLease> lease)
        : callerLease_(std::move(lease)),owner(o),state(std::move(s)),geometry(std::move(g)),scene(std::move(sceneIn)),
          resources(std::move(resourcesIn)),frame(std::move(frameIn)),replay(std::move(r)),bodySample(state->terminalBodySample) {}
    template <typename Layer>
    friend bool renderParticleSolids(const ParticleParams&,const videohelper::geometry::PlanOwnerIdentity&,
        int,int,int,Layer&,std::string&,bool);
};

// The replay one frame earlier, for Motion. Births come from the same replays
// as the two states (ParticlePointReplayTimes::birthSeconds). A body keeps its
// prior pose only when its note identity, Geometry identity and birth time all
// match. A body respawned by a replay wrap or reset keeps its identities but
// gets a new birth time, so it counts as born this frame. It has zero own
// Motion and still carries the frame camera and object Motion.
// previousBodyRadius is bodyRadius sampled at the prior frame.
struct ParticleSolidMotionPrior final
{
    const ParticleSolidState& previous;
    const std::array<double,kParticleSolidMaxBodies>& previousBirths;
    const std::array<double,kParticleSolidMaxBodies>& currentBirths;
    float previousBodyRadius=0;
};

// The prior body that body i of state continues, or null when it was born
// this frame. Note identity, Geometry identity and birth time must all match.
inline const ParticleRigidBody* particleSolidPriorBody(const ParticleSolidMotionPrior& prior,
    const ParticleSolidState& state,int i)
{
    const auto& body=state.bodies[static_cast<std::size_t>(i)];
    for (int j=0;j<prior.previous.count;++j) {
        const auto& old=prior.previous.bodies[static_cast<std::size_t>(j)];
        if (old.identity==body.identity && old.geometryIdentity==body.geometryIdentity
            && prior.previousBirths[static_cast<std::size_t>(j)]==prior.currentBirths[static_cast<std::size_t>(i)])
            return &old;
    }
    return nullptr;
}

// The owned prior for one production render, kept on ParticleParams as
// preparedSolidMotion. currentBirths come from the current frame's replay.
struct ParticleSolidMotionFrame final
{
    ParticleSolidState previous;
    std::array<double,kParticleSolidMaxBodies> previousBirths {}, currentBirths {};
    float previousBodyRadius=0;
};

// Motion View needs Motion on every render. Export Motion needs it only on
// export renders, so a preview with only that request draws Color as before.
inline bool particleSolidMotionRequested(const ParticleParams& p,bool exportUse) noexcept
{
    return p.solidMotionView || (exportUse && p.solidExportMotion);
}

// The prior replay reproduces bodies and static colliders exactly. A timeline
// body mesh and a coupled Water surface change between frames and neither
// keeps its prior positions, so both are refused before any replay runs.
inline bool admitParticleSolidMotion(const ParticleParams& p,std::string& error)
{
    if (p.simulationSpace!=1) { error="Particle Motion output requires Solid 3D"; return false; }
    if (requiresAuthenticatedParticleBodyReplay(p)) {
        error="Motion output for Solid 3D bodies with timeline body appearance is unavailable: the prior body mesh is not retained";
        return false;
    }
    if (hasBodyCoupledWater(p)) {
        error="Motion output for Solid 3D bodies over moving colliders is unavailable: prior collider positions are not retained";
        return false;
    }
    return true;
}

// Replays Solid 3D one project frame earlier. Replay is a pure function of
// the clip and project times, so preview, export, seeks and loop wraps all
// get the same prior and no frame history is kept. A prior time before the
// clip start has no bodies: every current body counts as born this frame and
// reports zero Motion, and the current static colliders stand in for the
// prior ones. The prior radius is the prior frame's terminal sample, as the
// current frame's is.
inline std::shared_ptr<const ParticleSolidMotionFrame> replayParticleSolidMotionPrior(
    const ParticleParams& parameters,double clipSeconds,double frameSeconds,float aspect,
    const canonicalblockc::CanonicalBlockCFrame* frame,const ParticleSolidState& currentState,
    const ParticlePointReplayTimes& current,float currentBodyRadius,std::string& error)
{
    if (!admitParticleSolidMotion(parameters,error)) return {};
    if (!std::isfinite(clipSeconds) || !std::isfinite(frameSeconds) || frameSeconds<=0 || frameSeconds>1
        || !std::isfinite(currentBodyRadius) || currentBodyRadius<=0) {
        error="Solid 3D Motion requires a finite clip time, a project frame of at most 1 s and a positive body radius";
        return {};
    }
    auto result=std::make_shared<ParticleSolidMotionFrame>();
    result->currentBirths=current.birthSeconds;
    result->previousBodyRadius=currentBodyRadius;
    const double priorSeconds=clipSeconds-frameSeconds;
    if (priorSeconds<0) { result->previous.colliders=currentState.colliders; error.clear(); return result; }
    auto prior=parameters;
    prior.historyProjectSeconds-=frameSeconds;
    prior.preparedBodies.reset(); prior.preparedSolids.reset(); prior.preparedBodyReplay.reset();
    prior.preparedSolidMotion.reset();
    if (!particleHistoryReady(prior,error,priorSeconds)) return {};
    ParticlePointReplayTimes times;
    replayParticleBodies(prior,priorSeconds,aspect,frame,&error,&result->previous,&times);
    if (!error.empty()) return {};
    // The prior body mesh is not retained. The current frame's sample is refused
    // in particleSolidScene, and a prior sample is refused here.
    if (result->previous.terminalBodySample) {
        error="Motion output for Solid 3D bodies with timeline body appearance is unavailable: the prior body mesh is not retained";
        return {};
    }
    result->previousBirths=times.birthSeconds;
    if (result->previous.count>0 && prior.historicalReplay) {
        auto sampled=prior;
        if (!sampleParticleParameters(prior,sampled,prior.historyProjectSeconds,error)) return {};
        const auto window=particleReplayWindow(sampled,priorSeconds);
        auto terminal=prior;
        if (!sampleParticleParameters(prior,terminal,double(window.start+window.age)/kParticleReplayTicksPerSecond,error))
            return {};
        result->previousBodyRadius=terminal.bodyRadius;
    }
    error.clear(); return result;
}

// Collider meshes and points draw as ordinary scene objects with no prior
// positions, so Motion needs them unchanged between the two frames.
inline bool sameParticleSolidColliders(const ParticleGeometryFrame& a,const ParticleGeometryFrame& b)
{
    if ((a.meshIdentity!=0)!=(b.meshIdentity!=0) || a.meshVertexCount!=b.meshVertexCount
        || a.meshTriangleCount!=b.meshTriangleCount || a.pointCount!=b.pointCount
        || a.meshVertexCount>a.meshVertices.size() || a.meshTriangleCount>a.meshTriangles.size()
        || a.pointCount<0 || a.pointCount>static_cast<int>(a.solidPoints.size())) return false;
    return std::equal(a.meshVertices.begin(),a.meshVertices.begin()+a.meshVertexCount,b.meshVertices.begin())
        && std::equal(a.meshTriangles.begin(),a.meshTriangles.begin()+a.meshTriangleCount,b.meshTriangles.begin())
        && std::equal(a.solidPoints.begin(),a.solidPoints.begin()+a.pointCount,b.solidPoints.begin());
}

// Scene object IDs are bounded draw slots, never truncated note/Geometry IDs.
// The retained state and binding carry the exact signed note and unsigned
// Geometry identities for every slot throughout the compositor lease. With a
// prior frame the scene also carries each body's previous instance transform.
inline std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> particleSolidScene(
    const ParticleParams& p,const ParticleSolidState& state,std::uint32_t nodeId,std::string& error,
    const ParticleSolidMotionPrior* prior=nullptr,bool rawMotionExport=false)
{
    using namespace HarmonicMIDI::grid;
    const bool dynamicBody=requiresAuthenticatedParticleBodyReplay(p) || state.terminalBodySample;
    if (dynamicBody && (!p.preparedBodyReplay || !p.bodyReplayContext
        || p.preparedSolids.get()!=&state || !p.preparedBodyReplay->matchesRender(p,p.bodyReplayContext->owner,
            static_cast<int>(nodeId),p.bodyReplayContext->width,p.bodyReplayContext->height)
        || (state.count>0 && !state.terminalBodySample)
        || (state.terminalBodySample && (state.terminalBodySample->absoluteTick()!=p.preparedBodyReplay->terminalTick()
            || state.terminalBodySample->source()!=p.geometryBinding->bodySource)))) {
        error="Dynamic body appearance requires its exact completed terminal replay sample"; return {};
    }
    // All bodies draw as one instanced scene object. Each collider is one more.
    const int contacts=(state.colliders.meshIdentity ? 1:0)+state.colliders.pointCount;
    if (state.count<0 || state.count>kParticleSolidMaxBodies || contacts<0 || state.colliders.pointCount>64
        || (state.count>0 ? 1:0)+contacts>int(Visual3DScene::kMaxObjects) || nodeId==0) {
        error="Solid 3D draw exceeds 128 bodies or 63 collider objects; reduce Count or collider points";
        return {};
    }
    if (prior) {
        if (dynamicBody) {
            error="Motion output for Solid 3D bodies with timeline body appearance is unavailable: the prior body mesh is not retained";
            return {};
        }
        if (!sameParticleSolidColliders(prior->previous.colliders,state.colliders)) {
            error="Motion output for Solid 3D bodies over moving colliders is unavailable: prior collider positions are not retained";
            return {};
        }
        if (prior->previous.count<0 || prior->previous.count>kParticleSolidMaxBodies
            || !std::isfinite(prior->previousBodyRadius) || prior->previousBodyRadius<=0) {
            error="Solid 3D Motion prior frame has an invalid body count or radius"; return {};
        }
    }
    std::shared_ptr<const ParticleBodyCallerLease> sceneLease;
    if (dynamicBody) {
        sceneLease=p.bodyReplayContext->caller->reserve(sizeof(ParticleBodyCallerPayload<Visual3DScene>),
            4*sizeof(Visual3DScene)+sizeof(ParticleSolidState));
        if (!sceneLease) { error="Dynamic body scene prospective caller admission is short"; return {}; }
    }
    auto scene=dynamicBody ? retainParticleBodyCallerPayload<Visual3DScene>(std::move(sceneLease))
        : std::make_shared<Visual3DScene>();
    scene->id={nodeId}; scene->activeCamera={1}; scene->ambientColor={0.16f,0.16f,0.16f};
    // A faceted unit sphere keeps the collider visibly three-dimensional even
    // with no mesh input. Rigid Geometry replaces its appearance, not inertia.
    const SceneVec3 sphere[]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    const std::uint32_t sphereIndices[]={0,2,4,4,2,1,1,2,5,5,2,0,0,4,3,4,1,3,1,5,3,5,0,3};
    for (const auto& v:sphere) { auto& out=scene->vertices[scene->vertexCount++]; out.position=out.normal=v; }
    for (auto index:sphereIndices) scene->indices[scene->indexCount++]=index;
    const std::uint32_t sphereVertices=scene->vertexCount,sphereIndexCount=scene->indexCount;
    std::uint32_t bodyFirstVertex=0,bodyFirstIndex=0,bodyVertices=sphereVertices,bodyIndices=sphereIndexCount;
    if (p.geometryBinding && p.geometryBinding->body && (!dynamicBody || state.terminalBodySample)) {
        const auto& geometry=dynamicBody ? state.terminalBodySample->prototype()
            : std::get<videowire::geometry::GeometryData>(p.geometryBinding->body->data);
        if (geometry.positions.empty() || geometry.positions.size()>256 || geometry.indices.empty() || geometry.indices.size()>384) {
            error="Rigid body appearance requires 1 to 256 vertices and 1 to 128 triangles"; return {};
        }
        SceneVec3 center{};
        float radius=1;
        if (!dynamicBody) {
            auto low=geometry.positions.front(),high=low;
            for (const auto& v:geometry.positions) {
                low.x=std::min(low.x,v.x); low.y=std::min(low.y,v.y); low.z=std::min(low.z,v.z);
                high.x=std::max(high.x,v.x); high.y=std::max(high.y,v.y); high.z=std::max(high.z,v.z);
            }
            center={(low.x+high.x)*0.5f,(low.y+high.y)*0.5f,(low.z+high.z)*0.5f};
            radius=0;
            for (const auto& v:geometry.positions) radius=std::max(radius,std::hypot(std::hypot(v.x-center.x,v.y-center.y),v.z-center.z));
            if (!std::isfinite(radius) || radius<=1.0e-7f) { error="Rigid body mesh has no finite bounding sphere"; return {}; }
        }
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
    scene->materials[1].doubleSided=true;
    scene->materialCount=2;
    const auto object=[&](SceneVec3 position,SceneVec3 scale,std::uint32_t firstVertex,std::uint32_t vertices,
                          std::uint32_t firstIndex,std::uint32_t indices,std::uint32_t material)->SceneObjectRecord& {
        auto& out=scene->objects[scene->objectCount++]; out.id={static_cast<std::uint32_t>(scene->objectCount)};
        out.material={material}; out.firstVertex=firstVertex; out.vertexCount=vertices; out.firstIndex=firstIndex; out.indexCount=indices;
        out.transform.translation={position.x-0.5f,position.y-0.5f,position.z-0.5f}; out.transform.scale=scale;
        return out;
    };
    if (state.count>0) {
        // object() subtracts 0.5, so the shared body object sits at the origin.
        // Each instance carries one body's replay position, radius and spin.
        scene->instancedObject=object({0.5f,0.5f,0.5f},{1,1,1},bodyFirstVertex,bodyVertices,bodyFirstIndex,bodyIndices,1).id;
        for (int i=0;i<state.count;++i) {
            const auto& body=state.bodies[static_cast<std::size_t>(i)];
            if (!std::isfinite(body.x) || !std::isfinite(body.y) || !std::isfinite(body.z)
                || std::abs(body.x)>352 || std::abs(body.y)>352 || std::abs(body.z)>352) {
                error="Solid body exceeds the bounded replay coordinate range"; return {};
            }
            auto& instance=scene->objectInstances[static_cast<std::size_t>(i)];
            instance.translation={body.x-0.5f,body.y-0.5f,body.z-0.5f};
            instance.scale={p.bodyRadius,p.bodyRadius,p.bodyRadius};
            const auto& q=body.orientation; instance.rotation={q[0],q[1],q[2],q[3]};
            if (!prior) continue;
            // A body born this frame repeats its current transform.
            auto& previous=scene->previousObjectInstances[static_cast<std::size_t>(i)];
            previous=instance;
            if (const auto* old=particleSolidPriorBody(*prior,state,i)) {
                if (!std::isfinite(old->x) || !std::isfinite(old->y) || !std::isfinite(old->z)
                    || std::abs(old->x)>352 || std::abs(old->y)>352 || std::abs(old->z)>352) {
                    error="Solid body exceeds the bounded replay coordinate range"; return {};
                }
                const auto r=prior->previousBodyRadius; const auto& o=old->orientation;
                previous.translation={old->x-0.5f,old->y-0.5f,old->z-0.5f};
                previous.scale={r,r,r}; previous.rotation={o[0],o[1],o[2],o[3]};
            }
        }
        scene->objectInstanceCount=static_cast<std::size_t>(state.count);
        if (prior) scene->previousObjectInstanceCount=scene->objectInstanceCount;
    }
    if (state.colliders.meshIdentity) {
        SolidTriangleWorkBudget work;
        if (!validateSolidTriangleFrame(state.colliders,work,error)) return {};
        const auto firstVertex=static_cast<std::uint32_t>(scene->vertexCount),firstIndex=static_cast<std::uint32_t>(scene->indexCount);
        const auto& surface=state.colliders;
        if (surface.meshVertexCount>Visual3DScene::kMaxVertices-scene->vertexCount
            || surface.meshTriangleCount*3>Visual3DScene::kMaxIndices-scene->indexCount)
        { error="Authored triangle collider exceeds the retained scene budget"; return {}; }
        for (std::uint32_t vertex=0;vertex<surface.meshVertexCount;++vertex) {
            const auto& point=surface.meshVertices[vertex];
            scene->vertices[scene->vertexCount++].position={point[0],point[1],point[2]};
        }
        for (std::uint32_t triangle=0;triangle<surface.meshTriangleCount;++triangle) {
            const auto& indices=surface.meshTriangles[triangle];
            for (const auto index:indices) scene->indices[scene->indexCount++]=index;
            const auto& a=surface.meshVertices[indices[0]];
            const auto& b=surface.meshVertices[indices[1]];
            const auto& c=surface.meshVertices[indices[2]];
            const auto normal=solidCross({b[0]-a[0],b[1]-a[1],b[2]-a[2]},
                {c[0]-a[0],c[1]-a[1],c[2]-a[2]});
            for (const auto index:indices) {
                auto& n=scene->vertices[firstVertex+index].normal;
                n.x+=normal[0]; n.y+=normal[1]; n.z+=normal[2];
            }
        }
        for (std::uint32_t vertex=0;vertex<surface.meshVertexCount;++vertex) {
            auto& n=scene->vertices[firstVertex+vertex].normal;
            const float length=std::hypot(std::hypot(n.x,n.y),n.z);
            if (length>0) { n.x/=length; n.y/=length; n.z/=length; } else n={0,1,0};
        }
        object({0,0,0},{1,1,1},firstVertex,surface.meshVertexCount,firstIndex,surface.meshTriangleCount*3,2);
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
    // A raw Motion export needs a native frame even with no body or collider.
    // A transform-only object has no triangles, so nothing draws and the frame
    // stays clear with zero Motion. Without it the scene would have no objects.
    if (scene->objectCount==0 && rawMotionExport) object({0.5f,0.5f,0.5f},{1,1,1},0,0,0,0,1);
    if (scene->objectCount && !arbitgpu::nativeFixtureSceneTopologySupported(*scene)) {
        error="Solid replay does not satisfy the native retained-scene contract"; return {};
    }
    return scene;
}

// Retain the exact body replay and composed scene behind its single native
// frame. This is a request receipt, not authored or frame-order simulation state.
struct ParticleImportedSolidFrame final
{
    videohelper::geometry::PlanOwnerIdentity owner;
    int nodeId = 0;
    double projectSeconds = 0, clipSeconds = 0;
    std::shared_ptr<const ParticleBodyState> bodies;
    std::shared_ptr<const ParticleSolidState> state;
    std::shared_ptr<const ParticleGeometryBinding> geometry;
    std::shared_ptr<const WaterCoupledFrame> water;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> sourceScene, scene;
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame;
};

// Append each replayed body as one scene object under parent. The replay maps
// parent space by center plus scale, so a body's local translation is its
// replay position minus center, over scale. With a Motion prior the scene also
// carries every object's prior local transform: a body takes its pose from the
// prior replay. Scene objects keep the priors the source carries, and a source
// without priors repeats their current transforms. Bodies born this frame repeat
// their own. A body's prior composes with the current parent world, so the
// refusal below stops Motion when an ancestor changed between the two frames.
inline std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> appendParticleSolidBodies(
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& source,
    const ParticleParams& p, const ParticleSolidState& state, std::uint32_t nodeId,
    HarmonicMIDI::grid::SceneObjectId parent, std::string& error,
    const ParticleSolidMotionPrior* prior=nullptr)
{
    using namespace HarmonicMIDI::grid;
    const auto refuse=[&](const char* reason)->std::shared_ptr<const Visual3DScene> { error=reason; return {}; };
    const auto& binding=*p.geometryBinding;
    if (std::size_t(state.count)>Visual3DScene::kMaxObjects-source->objectCount
        || source->materialCount==Visual3DScene::kMaxMaterials)
        return refuse("Imported solid composition exceeds the remaining object or material budget");
    if (prior && (prior->previous.count<0 || prior->previous.count>kParticleSolidMaxBodies
        || !std::isfinite(prior->previousBodyRadius) || prior->previousBodyRadius<=0))
        return refuse("Solid 3D Motion prior frame has an invalid body count or radius");
    // Use the released solid mesh builder, including exact selected rigid
    // appearance. Only its local mesh ranges are borrowed, never its camera.
    ParticleSolidState meshOnly;
    const auto mesh=particleSolidScene(p,meshOnly,nodeId,error);
    if (!mesh) return {};
    const std::uint32_t first=p.geometryBinding->body ? 6u:0u;
    const std::uint32_t firstIndex=p.geometryBinding->body ? 24u:0u;
    const auto vertices=mesh->vertexCount-first, indices=mesh->indexCount-firstIndex;
    if (vertices>Visual3DScene::kMaxVertices-source->vertexCount
        || indices>Visual3DScene::kMaxIndices-source->indexCount)
        return refuse("Imported solid composition exceeds the remaining vertex or index budget");
    auto result=std::make_shared<Visual3DScene>(*source);
    const auto freeId=[](const auto& records,std::size_t count) {
        std::uint32_t id=1;
        while (std::any_of(records.begin(),records.begin()+count,[&](const auto& record) { return record.id.value==id; })) ++id;
        return id;
    };
    auto material=mesh->materials[0];
    material.id={freeId(result->materials,result->materialCount)};
    result->materials[result->materialCount++]=material;
    const auto baseVertex=result->vertexCount, baseIndex=result->indexCount;
    std::copy_n(mesh->vertices.begin()+first,vertices,result->vertices.begin()+baseVertex);
    std::copy_n(mesh->indices.begin()+firstIndex,indices,result->indices.begin()+baseIndex);
    result->vertexCount+=vertices; result->indexCount+=indices;
    for (std::size_t i=0;i<source->objectCount;++i)
        if (source->objects[i].id.value>arbitgpu::kAppendedSolidObjectIdBase)
            return refuse("Imported solid composition source uses the reserved solid body object ID range");
    for (int i=0;i<state.count;++i) {
        const auto& body=state.bodies[i];
        if (!std::isfinite(body.x) || !std::isfinite(body.y) || !std::isfinite(body.z)
            || std::abs(body.x)>352 || std::abs(body.y)>352 || std::abs(body.z)>352)
            return refuse("Imported solid body exceeds the bounded replay coordinate range");
        auto& object=result->objects[result->objectCount];
        // Bodies take IDs from the reserved range so no object-targeted node can reach them.
        object.id={arbitgpu::kAppendedSolidObjectIdBase+1u+static_cast<std::uint32_t>(i)};
        object.parent=parent; object.material=material.id;
        object.firstVertex=baseVertex; object.vertexCount=vertices;
        object.firstIndex=baseIndex; object.indexCount=indices;
        object.transform.translation={(body.x-binding.centerX)/binding.scale,
            (body.y-binding.centerY)/binding.scale,(body.z-binding.centerZ)/binding.scale};
        const auto radius=p.bodyRadius/binding.scale;
        object.transform.scale={radius,radius,radius};
        object.transform.rotation={body.orientation[0],body.orientation[1],body.orientation[2],body.orientation[3]};
        ++result->objectCount;
    }
    if (prior) {
        const auto firstBody=result->objectCount-static_cast<std::size_t>(state.count);
        // The body prior composes with the current parent world. That is exact
        // only while no ancestor moved, and an ancestor's prior is recorded only
        // when the source carries priors. Refuse when a recorded ancestor moved.
        const auto ancestorMoved=[&]() {
            auto ancestor=parent;
            for (std::size_t depth=0; ancestor.isValid() && depth<source->objectCount; ++depth) {
                std::size_t index=0;
                while (index<source->objectCount && !(source->objects[index].id==ancestor)) ++index;
                if (index==source->objectCount) break;
                if (source->previousObjectTransformCount!=0) {
                    const auto& now=source->objects[index].transform;
                    const auto& before=source->previousObjectTransforms[index];
                    if (before.translation.x!=now.translation.x || before.translation.y!=now.translation.y
                        || before.translation.z!=now.translation.z || before.rotation.x!=now.rotation.x
                        || before.rotation.y!=now.rotation.y || before.rotation.z!=now.rotation.z
                        || before.rotation.w!=now.rotation.w || before.scale.x!=now.scale.x
                        || before.scale.y!=now.scale.y || before.scale.z!=now.scale.z) return true;
                }
                ancestor=source->objects[index].parent;
            }
            return false;
        };
        if (ancestorMoved())
            return refuse("Solid 3D Motion is unavailable because a parent or ancestor of the solid bodies changed between frames");
        for (std::size_t i=0;i<result->objectCount;++i)
            result->previousObjectTransforms[i]=source->previousObjectTransformCount!=0 && i<source->objectCount
                ? source->previousObjectTransforms[i] : result->objects[i].transform;
        for (int i=0;i<state.count;++i) {
            const auto* old=particleSolidPriorBody(*prior,state,i);
            if (!old) continue;
            if (!std::isfinite(old->x) || !std::isfinite(old->y) || !std::isfinite(old->z)
                || std::abs(old->x)>352 || std::abs(old->y)>352 || std::abs(old->z)>352)
                return refuse("Imported solid body exceeds the bounded replay coordinate range");
            auto& previous=result->previousObjectTransforms[firstBody+static_cast<std::size_t>(i)];
            previous.translation={(old->x-binding.centerX)/binding.scale,
                (old->y-binding.centerY)/binding.scale,(old->z-binding.centerZ)/binding.scale};
            const auto radius=prior->previousBodyRadius/binding.scale;
            previous.scale={radius,radius,radius};
            previous.rotation={old->orientation[0],old->orientation[1],old->orientation[2],old->orientation[3]};
        }
        result->previousObjectTransformCount=result->objectCount;
    }
    if (!validateVisual3DScene(*result).valid() || !arbitgpu::nativeFixtureSceneTopologySupported(*result))
        return refuse("Composed solid bodies do not satisfy the retained native scene contract");
    error.clear(); return result;
}

inline std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> appendParticleSolidsToScene(
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& source,
    const HarmonicMIDI::grid::Visual3DScene& authored,
    const videowire::geometry::TimelineSceneBinding& timeline,
    const ParticleParams& p, const ParticleSolidState& state, std::uint32_t nodeId,
    std::string& error, const ParticleSolidMotionPrior* prior=nullptr)
{
    using namespace HarmonicMIDI::grid;
    using namespace videowire::geometry;
    const auto refuse=[&](const char* reason)->std::shared_ptr<const Visual3DScene> { error=reason; return {}; };
    if (state.count>kParticleImportedSolidMaxBodies)
        return refuse("Imported Solid 3D supports at most 64 bodies because each body is its own composed scene object; set Count to 64 or less");
    if (!source || !validateVisual3DScene(*source).valid() || !p.geometryBinding || !p.geometryBinding->mesh
        || p.simulationSpace!=1 || (p.bodyShape!=0 && p.bodyShape!=2) || state.count<0 || nodeId==0
        || !std::isfinite(p.bodyRadius) || p.bodyRadius<=0 || p.bodyRadius>.1f
        || p.geometryBinding->projection!=0 || !std::isfinite(p.geometryBinding->scale) || p.geometryBinding->scale<=0)
        return refuse("Imported solid composition requires an exact sphere or oriented box replay and uniform XYZ geometry binding");
    // The box proxy and its appearance share one normalization: body mesh
    // bounds over the enclosing radius, scaled by bodyRadius. The appended
    // object below uses that same mesh and scale, so the drawn box matches
    // the contact box.
    if (p.bodyShape==2 && (!p.geometryBinding->body || requiresAuthenticatedParticleBodyReplay(p)))
        return refuse("Imported oriented box solids require static body Geometry");
    if (p.geometryBinding->points || state.colliders.pointCount!=0)
        return refuse("Imported solid composition does not support additional Points3D colliders");
    const auto bytes=decodeLoweredPlanText(timeline.loweredPlan,error);
    const auto plan=bytes ? decodeLoweredRuntimePlan(*bytes,error):std::nullopt;
    const auto bound=decodeTimelineScenePlan(timeline,error);
    const auto particle=plan ? admitValue(*p.geometryBinding->mesh,plan->contract,error):std::nullopt;
    if (!bound || !particle || encodeRuntimeValue(*particle)!=encodeRuntimeValue(*bound)
        || !validTimelineSceneBinding(timeline,authored,*bound,error))
        return refuse("Imported solids must share the exact retained TimelineScene3D mesh binding");
    const auto* parent=visual3d_detail::findById(source->objects,source->objectCount,timeline.object);
    if (!parent) return refuse("Imported solid composition lost its TimelineScene3D object");
    const auto* ancestor=parent;
    for (std::size_t depth=0;ancestor && depth<source->objectCount;++depth) {
        const auto& scale=ancestor->transform.scale;
        if (scale.x<=0 || scale.x!=scale.y || scale.x!=scale.z)
            return refuse("Imported solid composition requires positive uniform scale throughout the bound hierarchy");
        ancestor=ancestor->parent.isValid() ? visual3d_detail::findById(source->objects,source->objectCount,ancestor->parent):nullptr;
    }
    if (state.count==0) { error.clear(); return source; }
    if (state.colliders.meshIdentity!=p.geometryBinding->mesh->stableId)
        return refuse("Imported solids lost the exact TimelineScene3D collider identity");
    if (parent->vertexCount!=state.colliders.meshVertexCount || parent->indexCount!=state.colliders.meshTriangleCount*3)
        return refuse("Imported solid collider does not match the evaluated scene mesh capacity");
    SolidTriangleWorkBudget work;
    if (!validateSolidTriangleFrame(state.colliders,work,error)) return {};
    const auto& binding=*p.geometryBinding;
    for (std::uint32_t i=0;i<parent->vertexCount;++i) {
        const auto& point=source->vertices[parent->firstVertex+i].position;
        if (state.colliders.meshVertices[i]!=SolidVector{binding.centerX+binding.scale*point.x,
            binding.centerY+binding.scale*point.y,binding.centerZ+binding.scale*point.z})
            return refuse("Imported solid collider time or coordinates differ from the evaluated TimelineScene3D mesh");
    }
    for (std::uint32_t i=0;i<parent->indexCount;++i)
        if (source->indices[parent->firstIndex+i]!=state.colliders.meshTriangles[i/3][i%3])
            return refuse("Imported solid collider lost the exact evaluated triangle order");
    return appendParticleSolidBodies(source,p,state,nodeId,timeline.object,error,prior);
}

// Family form of the composition above. One replay met every child, so each
// child's evaluated mesh must match its own range of the merged collider frame,
// placed by its composed transform in the frame. Bodies attach to the shared
// parent, or to the scene root when children have different parents.
inline std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> appendParticleSolidsToFamilyScene(
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& source,
    const HarmonicMIDI::grid::Visual3DScene& authored,
    const std::vector<videowire::geometry::TimelineSceneBinding>& family,
    const ParticleParams& p, const ParticleSolidState& state, std::uint32_t nodeId,
    std::string& error, const ParticleSolidMotionPrior* prior=nullptr)
{
    using namespace HarmonicMIDI::grid;
    using namespace videowire::geometry;
    const auto refuse=[&](const char* reason)->std::shared_ptr<const Visual3DScene> { error=reason; return {}; };
    if (state.count>kParticleImportedSolidMaxBodies)
        return refuse("Imported Solid 3D supports at most 64 bodies because each body is its own composed scene object; set Count to 64 or less");
    if (!source || !validateVisual3DScene(*source).valid() || !p.geometryBinding || !p.geometryBinding->mesh
        || p.simulationSpace!=1 || (p.bodyShape!=0 && p.bodyShape!=2) || state.count<0 || nodeId==0
        || !std::isfinite(p.bodyRadius) || p.bodyRadius<=0 || p.bodyRadius>.1f
        || p.geometryBinding->projection!=0 || !std::isfinite(p.geometryBinding->scale) || p.geometryBinding->scale<=0)
        return refuse("Imported solid composition requires an exact sphere or oriented box replay and uniform XYZ geometry binding");
    if (p.bodyShape==2 && (!p.geometryBinding->body || requiresAuthenticatedParticleBodyReplay(p)))
        return refuse("Imported oriented box solids require static body Geometry");
    if (p.geometryBinding->points || state.colliders.pointCount!=0)
        return refuse("Imported solid composition does not support additional Points3D colliders");
    const auto& binding=*p.geometryBinding;
    if (family.size()<2 || binding.meshFamily.size()!=family.size())
        return refuse("Imported solid family composition requires one collider plan per timeline child");
    SceneObjectId parent;
    bool sharedParent=true;
    std::vector<const SceneObjectRecord*> children;
    for (std::size_t i=0;i<family.size();++i) {
        const auto& timeline=family[i];
        const auto& child=binding.meshFamily[i];
        if (!child.mesh || child.object!=timeline.object.value)
            return refuse("Imported solid family colliders must name the timeline children in order");
        const auto bytes=decodeLoweredPlanText(timeline.loweredPlan,error);
        const auto plan=bytes ? decodeLoweredRuntimePlan(*bytes,error):std::nullopt;
        const auto bound=decodeTimelineScenePlan(timeline,error);
        const auto particle=plan ? admitValue(*child.mesh,plan->contract,error):std::nullopt;
        if (!bound || !particle || encodeRuntimeValue(*particle)!=encodeRuntimeValue(*bound)
            || !validTimelineSceneBinding(timeline,authored,*bound,error))
            return refuse("Imported solids must share each child's exact retained TimelineScene3D mesh binding");
        const auto* object=visual3d_detail::findById(source->objects,source->objectCount,timeline.object);
        if (!object) return refuse("Imported solid composition lost a timeline family child");
        if (i==0) parent=object->parent;
        else if (object->parent!=parent) sharedParent=false;
        children.push_back(object);
    }
    // Bodies attach to the frame, which is the shared parent or the scene root.
    if (!sharedParent) parent={};
    if (!particleFamilyFrameUniform(*source,parent))
        return refuse("Imported solid composition requires positive uniform scale throughout the bound hierarchy");
    for (std::size_t c=0;c<children.size();++c) {
        std::array<float,3> translation {};
        std::array<float,9> linear {};
        const auto& child=binding.meshFamily[c];
        if (!particleFamilyWorldTransform(*source,*children[c],parent,translation,linear)
            || child.translation!=translation || child.linear!=linear)
            return refuse("Imported solid family children must keep their collider transforms; the collider frame needs static parent transforms, so an animated parent refuses");
    }
    if (state.count==0) { error.clear(); return source; }
    if (state.colliders.meshIdentity!=particleColliderMeshIdentity(binding))
        return refuse("Imported solids lost the exact timeline family collider identity");
    std::uint32_t vertices=0, triangles=0;
    for (const auto* object:children) {
        if (object->indexCount%3!=0) return refuse("Imported solid collider does not match the evaluated scene mesh capacity");
        vertices+=object->vertexCount; triangles+=object->indexCount/3;
    }
    if (vertices!=state.colliders.meshVertexCount || triangles!=state.colliders.meshTriangleCount)
        return refuse("Imported solid collider does not match the evaluated scene mesh capacity");
    SolidTriangleWorkBudget work;
    if (!validateSolidTriangleFrame(state.colliders,work,error)) return {};
    std::uint32_t firstVertex=0, firstTriangle=0;
    for (std::size_t c=0;c<children.size();++c) {
        const auto& object=*children[c];
        for (std::uint32_t i=0;i<object.vertexCount;++i)
            if (state.colliders.meshVertices[firstVertex+i]
                !=particleFamilyColliderPoint(binding,binding.meshFamily[c],source->vertices[object.firstVertex+i].position))
                return refuse("Imported solid collider time or coordinates differ from the evaluated timeline family meshes");
        for (std::uint32_t i=0;i<object.indexCount;++i)
            if (firstVertex+source->indices[object.firstIndex+i]!=state.colliders.meshTriangles[firstTriangle+i/3][i%3])
                return refuse("Imported solid collider lost the exact evaluated triangle order");
        firstVertex+=object.vertexCount; firstTriangle+=object.indexCount/3;
    }
    return appendParticleSolidBodies(source,p,state,nodeId,parent,error,prior);
}

template <typename Layer, typename = void>
struct HasImportedSolidDepth : std::false_type {};
template <typename Layer>
struct HasImportedSolidDepth<Layer,std::void_t<decltype(std::declval<Layer>().nativeDepthTextureDescriptor),
    decltype(std::declval<Layer>().nativeDepthTextureBackend),decltype(std::declval<Layer>().nativeDepthTextureView)>>
    : std::true_type {};

template <typename Layer>
inline bool validateImportedSolidFrame(const ParticleParams& p,
    const videohelper::geometry::PlanOwnerIdentity& owner, int nodeId,
    double projectSeconds, double clipSeconds, int width, int height,
    const Layer& layer, std::string& error)
{
    const auto& receipt=p.preparedImportedSolids;
    if (!receipt || !owner.valid() || !(receipt->owner==owner) || receipt->nodeId!=nodeId
        || receipt->projectSeconds!=projectSeconds || receipt->clipSeconds!=clipSeconds
        || !receipt->bodies || !receipt->state || !receipt->geometry || !receipt->sourceScene || !receipt->scene
        || receipt->bodies!=p.preparedBodies || receipt->state!=p.preparedSolids || receipt->geometry!=p.geometryBinding
        || !receipt->frame || width<=0 || height<=0
        || receipt->frame->width()!=static_cast<unsigned>(width) || receipt->frame->height()!=static_cast<unsigned>(height)
        || layer.texWidth!=width || layer.texHeight!=height
        || layer.nativeTextureBackend!=receipt->frame->backend()
        || layer.nativeTextureView!=receipt->frame->colorTextureViewHandle())
    { error="Imported Solid 3D requires the exact prepared scene, body replay, owner and frame"; return false; }
    if (receipt->water && (receipt->water->projectSeconds!=projectSeconds || clipSeconds!=projectSeconds))
    { error="Imported coupled water receipt lost its exact project-zero body clock"; return false; }
    const auto exact=[](const auto& a,const auto& b) {
        return a.complete() && b.complete() && a.backend==b.backend && a.viewKind==b.viewKind && a.format==b.format
            && a.imageHandle==b.imageHandle && a.textureViewHandle==b.textureViewHandle
            && a.width==b.width && a.height==b.height && a.sampleCount==b.sampleCount
            && a.sampledBinding==b.sampledBinding && a.deviceOrContextIdentity==b.deviceOrContextIdentity
            && a.rendererGeneration==b.rendererGeneration && a.colorSpace==b.colorSpace
            && a.transfer==b.transfer && a.rowOrder==b.rowOrder;
    };
    if (!exact(layer.nativeTextureDescriptor,receipt->frame->colorTextureDescriptor())
        || !receipt->frame->depthTextureDescriptor().complete())
    { error="Imported Solid 3D prepared frame lost its exact native color or depth resource"; return false; }
    if constexpr (!HasImportedSolidDepth<Layer>::value)
    { error="Imported Solid 3D requires the retained native depth descriptor"; return false; }
    else if (layer.nativeDepthTextureBackend!=receipt->frame->backend()
        || layer.nativeDepthTextureView!=receipt->frame->depthTextureViewHandle()
        || !exact(layer.nativeDepthTextureDescriptor,receipt->frame->depthTextureDescriptor()))
    { error="Imported Solid 3D prepared frame lost its exact native depth resource"; return false; }
    error.clear(); return true;
}

// The scene renderParticleSolids draws. With a Motion request each body
// instance also carries its transform from the prepared prior replay.
inline std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> particleSolidRenderScene(
    const ParticleParams& p,std::uint32_t nodeId,bool exportUse,std::string& error)
{
    if (!p.preparedSolids) { error="Solid 3D requires a prepared replay"; return {}; }
    if (!particleSolidMotionRequested(p,exportUse)) return particleSolidScene(p,*p.preparedSolids,nodeId,error);
    if (!admitParticleSolidMotion(p,error)) return {};
    if (!p.preparedSolidMotion) { error="Solid 3D Motion requires the replay one project frame earlier"; return {}; }
    const auto& motion=*p.preparedSolidMotion;
    const ParticleSolidMotionPrior prior{motion.previous,motion.previousBirths,motion.currentBirths,motion.previousBodyRadius};
    return particleSolidScene(p,*p.preparedSolids,nodeId,error,&prior,exportUse && p.solidExportMotion);
}

// True when the replay has no body and no collider. Its scene then holds at
// most the transform-only Motion placeholder, which draws nothing.
inline bool particleSolidStateEmpty(const ParticleSolidState& state)
{
    return state.count==0 && !state.colliders.meshIdentity && state.colliders.pointCount==0;
}

// Historical bodies are born only from note rows or emitter points (see the
// birth loop in particle_body_replay.h). A project with no notes and no emitter
// therefore has no birth source at any time, so an empty replay is a
// configuration problem, not a quiet moment.
inline bool particleSolidBodiesNeedNotes(const ParticleParams& p)
{
    return p.historicalReplay && p.geometryCount==0 && (!p.geometryBinding || !p.geometryBinding->emitters)
        && p.history && p.history->score && p.history->score->notes.empty();
}

// Native inputs for a Solid 3D draw, set as Render 3D sets them. Raw Motion
// export applies to export renders only. The scene camera is fixed, so the
// prior camera is the current one and the prior sample repeats the inputs.
inline bool particleSolidRuntimeInputs(const ParticleParams& p,bool exportUse,bool linearColor,
    arbitgpu::NativeFixtureSceneRuntimeInputs& inputs,std::string& error)
{
    inputs={}; inputs.linearColor=linearColor;
    if (!particleSolidMotionRequested(p,exportUse)) return true;
    if (linearColor && p.solidMotionView)
    { error="HDR Solid 3D requires the Color image, not the Motion View"; return false; }
    inputs.imageOutput=p.solidMotionView ? renderpassoutput::Output::Motion:renderpassoutput::Output::Color;
    if (exportUse && p.solidExportMotion)
        inputs.rawExportMask=1u<<static_cast<unsigned>(renderpassoutput::Output::Motion);
    inputs.previousMotion=arbitgpu::sceneMotionSample(inputs);
    return true;
}

template <typename Layer,typename=void>
struct HasParticleSolidRawExport : std::false_type {};
template <typename Layer>
struct HasParticleSolidRawExport<Layer,std::void_t<decltype(std::declval<Layer>().rawExportFrame),
    decltype(std::declval<Layer>().rawExportMask),decltype(std::declval<Layer>().rawExportClipId),
    decltype(std::declval<Layer>().rawExportRenderId),decltype(std::declval<Layer>().rawExportImageOutput)>>
    : std::true_type {};

template <typename Layer>
inline bool renderParticleSolids(const ParticleParams& p,const videohelper::geometry::PlanOwnerIdentity& owner,
                                int nodeId,int width,int height,Layer& layer,std::string& error,bool linearColor)
{
    if (!owner.valid() || !p.preparedSolids || nodeId<=0 || width<=0 || height<=0) {
        error="Solid 3D requires a prepared replay, exact preview/export owner and output extent"; return false;
    }
    if (requiresAuthenticatedParticleBodyReplay(p) && (!p.preparedBodyReplay
        || !p.preparedBodyReplay->matchesRender(p,owner,nodeId,width,height))) {
        error="Dynamic Solid 3D frame does not match its replay owner, clocks or extent"; return false;
    }
    // Same refusal as the Score Field: an empty frame with no message hides the
    // cause. Colliders still draw, so only a fully empty replay is refused.
    if (particleSolidStateEmpty(*p.preparedSolids) && particleSolidBodiesNeedNotes(p)) {
        error="Solid 3D particles need notes on the timeline: bodies are born from notes or emitters, and this project has no notes";
        return false;
    }
    const bool exportUse=owner.use==videohelper::geometry::PlanUse::exportRender;
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    if (!particleSolidRuntimeInputs(p,exportUse,linearColor,inputs,error)) return false;
    if constexpr (!HasParticleSolidRawExport<Layer>::value)
        if (inputs.rawExportMask!=0)
        { error="Raw Solid 3D Motion export requires a layer that retains the native frame owner"; return false; }
    if (inputs.rawExportMask!=0 && owner.clipIdentity>static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    { error="Raw Solid 3D Motion export clip identity exceeds its manifest range"; return false; }
    auto scene=particleSolidRenderScene(p,static_cast<std::uint32_t>(nodeId),exportUse,error);
    if (!scene) return false;
    if (scene->objectCount==0) { layer.opacity=0; layer.particleSource=false; return true; }
    // The backend's native allocation budget remains independent. Reserve new
    // caller frame/submission/texture metadata before submitting any native work.
    std::shared_ptr<const ParticleBodyCallerLease> frameLease;
    if (requiresAuthenticatedParticleBodyReplay(p)) {
        frameLease=p.bodyReplayContext->caller->reserve(ParticleSolidFrame::callerLogicalBytes(),
            ParticleSolidFrame::callerLogicalBytes());
        if (!frameLease) { error="Dynamic body frame prospective caller admission is short"; return false; }
    }
    auto& backend=arbitgpu::nativeFixtureSceneBackend();
    // Only an export may block. A Preview prepare whose program is still compiling defers the frame.
    const auto prepareUse=exportUse ? arbitgpu::NativeFixturePrepareUse::Blocking
                                    : arbitgpu::NativeFixturePrepareUse::Preview;
    auto prepared=backend.prepare(scene,{},prepareUse);
    if (prepared.pending) {
        if (exportUse) { error="Solid 3D export preparation returned pending"; return false; }
        videohelper::geometry::noteGeometryCorePreviewPendingOnThisThread();
        error.clear();
        return false;
    }
    if (!prepared.prepared || !prepared.resources) { error=prepared.error; return false; }
    auto rendered=backend.render(scene,prepared.resources,width,height,inputs);
    if (!rendered.rendered || !rendered.frame) { error=rendered.error; return false; }
    auto retained=std::shared_ptr<const ParticleSolidFrame>(new ParticleSolidFrame(
        owner,p.preparedSolids,p.geometryBinding,std::move(scene),std::move(prepared.resources),std::move(rendered.frame),p.preparedBodyReplay,std::move(frameLease)));
    const auto& frame=retained->frame; const auto view=frame->colorTextureViewHandle();
    if (frame->backend()=="opengl" && view>std::numeric_limits<unsigned>::max()) {
        error="Solid 3D texture exceeds compositor handle bounds"; return false;
    }
    layer.texture=frame->backend()=="opengl" ? static_cast<unsigned>(view):0;
    layer.nativeTextureBackend=frame->backend(); layer.nativeTextureView=view;
    layer.nativeTextureDescriptor=frame->colorTextureDescriptor(); layer.nativeTextureOwner=std::move(retained);
    layer.texWidth=width; layer.texHeight=height; layer.particleSource=false;
    if constexpr (HasParticleSolidRawExport<Layer>::value)
        if (inputs.rawExportMask!=0) {
            layer.rawExportFrame=frame; layer.rawExportMask=inputs.rawExportMask;
            layer.rawExportClipId=static_cast<int>(owner.clipIdentity);
            // Render 3D keys its raw passes by node value + 1. Using the same rule
            // keeps an adjacent Render 3D node from sharing this frame key.
            layer.rawExportRenderId=static_cast<std::uint32_t>(
                render3dimage::renderStableIdForNodeValue(static_cast<std::uint64_t>(nodeId)));
            layer.rawExportImageOutput=inputs.imageOutput;
        }
    // The placeholder draws nothing. Hide Color as a frame with no request
    // does, so the export does not composite the scene's opaque backdrop.
    if (particleSolidStateEmpty(*p.preparedSolids)) layer.opacity=0;
    return true;
}
}
