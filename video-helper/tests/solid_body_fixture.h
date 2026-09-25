#pragma once

#include "../src/particle_body_replay.h"
#include "../src/particle_solid_renderer.h"
#include "../src/visual_parameter_timeline.h"

inline videorender::ParticleParams solidBodyFixture()
{
    videorender::ParticleParams p;
    p.motionMode=2; p.simulationSpace=1; p.historicalReplay=true;
    auto history=std::make_shared<videorender::ParticleHistorySource>();
    history->score=std::make_shared<const arbitmod::Score>(); p.history=history;
    p.count=2; p.geometryCount=2; p.lifetime=2; p.force=0;
    p.geometryAnchors[0]={0.35f,0.45f,0.35f,0.45f};
    p.geometryAnchors[1]={0.65f,0.55f,0.65f,0.55f};
    p.geometryDepths[0]={0.35f,0.35f}; p.geometryDepths[1]={0.65f,0.65f};
    p.geometryIdentities[0]=9007199254740001ull; p.geometryIdentities[1]=9007199254740002ull;
    p.impulseZ=0.2f; p.bodyRadius=0.09f; p.angularVelocity={0.5f,1,0.25f};
    p.red=0.2f; p.green=0.7f; p.blue=1;
    return p;
}

inline videorender::ParticleSolidState sampleSolidBodyFixture(videorender::ParticleParams& p,double time,std::string& error)
{
    p.historyProjectSeconds=time; p.preparedBodies.reset(); p.preparedSolids.reset();
    videorender::ParticleSolidState state;
    const auto planar=videorender::replayParticleBodies(p,time,1,nullptr,&error,&state);
    if (error.empty()) {
        p.preparedSolids=std::make_shared<const videorender::ParticleSolidState>(state);
        p.preparedBodies=std::make_shared<const videorender::ParticleBodyState>(planar);
    }
    return state;
}

inline bool sameSolidBodies(const videorender::ParticleSolidState& a,const videorender::ParticleSolidState& b)
{
    if (a.count!=b.count) return false;
    for (int i=0;i<a.count;++i) {
        const auto& x=a.bodies[i]; const auto& y=b.bodies[i];
        if (videorender::solidPosition(x)!=videorender::solidPosition(y)
            || videorender::solidVelocity(x)!=videorender::solidVelocity(y)
            || x.orientation!=y.orientation || x.angular!=y.angular
            || x.identity!=y.identity || x.geometryIdentity!=y.geometryIdentity) return false;
    }
    return true;
}

template <typename ReadPixels>
bool verifyNativeSolidBodies(ReadPixels readPixels,std::string& error,
    const std::shared_ptr<const videowire::geometry::ValueDescriptor>& appearance={}, bool automate=false)
{
    struct Layer {
        unsigned texture=0; float opacity=1; bool particleSource=true;
        std::string nativeTextureBackend;
        std::uintptr_t nativeTextureView=0;
        arbitgpu::NativeTextureViewDescriptor nativeTextureDescriptor;
        std::shared_ptr<const void> nativeTextureOwner;
        int texWidth=0,texHeight=0;
    };
    auto p=solidBodyFixture();
    if (automate)
    {
        auto timeline=std::make_shared<videorender::VisualParameterTimeline>();
        if (!timeline->bind({{"clip4/visual7/gravityZ",0,.4},{"clip4/visual7/gravityZ",.25,-.2},
                            {"clip4/visual7/drag",0,.3},{"clip4/visual7/drag",.3,1.5}},error)) return false;
        auto history=std::make_shared<videorender::ParticleHistorySource>(*p.history);
        history->parameterAt=[timeline](const auto& destination,double time,double& value,std::string&)
        { timeline->sample(destination,time,value); return true; };
        p.history=history; p.historyClipId=4; p.historyNodeId=7;
    }
    if (appearance) {
        auto binding=std::make_shared<videorender::ParticleGeometryBinding>(); binding->body=appearance;
        p.geometryBinding=binding;
    }
    const auto expected=sampleSolidBodyFixture(p,0.5,error);
    if (!error.empty()) return false;
    Layer preview,exported,changed;
    videohelper::geometry::PlanOwnerIdentity owner{1,2,3,4,5,videohelper::geometry::PlanUse::preview};
    if (!videorender::renderParticleSolids(p,owner,7,96,96,preview,error)) return false;
    const auto lease=std::static_pointer_cast<const videorender::ParticleSolidFrame>(preview.nativeTextureOwner);
    const auto pixels=readPixels(lease->frame);
    bool visible=false; for (std::size_t i=3;i<pixels.size();i+=4) visible=visible || pixels[i]!=0;
    sampleSolidBodyFixture(p,10000,error); sampleSolidBodyFixture(p,0.125,error);
    const auto sought=sampleSolidBodyFixture(p,0.5,error);
    owner.use=videohelper::geometry::PlanUse::exportRender;
    if (!error.empty() || !videorender::renderParticleSolids(p,owner,7,96,96,exported,error)) return false;
    const auto exportLease=std::static_pointer_cast<const videorender::ParticleSolidFrame>(exported.nativeTextureOwner);
    if (!visible || !sameSolidBodies(expected,sought) || pixels!=readPixels(exportLease->frame)
        || preview.particleSource || exported.particleSource
        || lease->state->bodies[0].geometryIdentity!=9007199254740001ull
        || lease->scene->objects[0].transform.rotation.y==0 || lease->scene->objectCount!=2) {
        error="Solid body preview/export pose, native pixels or exact retained IDs disagree"; return false;
    }
    sampleSolidBodyFixture(p,0.9,error);
    if (!error.empty() || !videorender::renderParticleSolids(p,owner,7,96,96,changed,error)) return false;
    const auto changedLease=std::static_pointer_cast<const videorender::ParticleSolidFrame>(changed.nativeTextureOwner);
    if (pixels==readPixels(changedLease->frame) || pixels!=readPixels(lease->frame)) {
        error="Solid pose did not change native pixels or overwrote a retained frame"; return false;
    }
    return true;
}
