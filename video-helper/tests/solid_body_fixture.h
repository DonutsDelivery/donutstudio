#pragma once

#include "../src/particle_body_replay.h"
#include "../src/particle_solid_renderer.h"
#include "../src/visual_parameter_timeline.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

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

// Each sounding note on the spawn track spawns one Solid 3D body. Distinct
// fractional pitches keep the spawn columns apart, so up to 128 notes replay
// beyond the former 64-body bound without coincident contacts.
inline videorender::ParticleParams manySolidBodyFixture(int notes)
{
    auto p=solidBodyFixture();
    auto score=std::make_shared<arbitmod::Score>(); score->scoreRevision=1;
    for (int i=0;i<notes;++i) {
        arbitmod::Note note; note.id=1001+i; note.midiNote=36+0.45f*float(i);
        note.freqHz=440.0f*std::pow(2.0f,(note.midiNote-69)/12.0f);
        note.velocity=float(64+i%64); note.startBeat=0; note.lengthBeats=64;
        score->notes.push_back(note);
    }
    auto history=std::make_shared<videorender::ParticleHistorySource>(*p.history);
    history->score=score;
    auto& key=history->key;
    key.projectGeneration=key.sourceGeneration=key.helperGeneration=key.backendGeneration=key.deviceGeneration=1;
    key.scoreGeneration=key.beatMapGeneration=key.fpsGeneration=key.loopGeneration=key.seekGeneration=1;
    p.history=history; p.geometryCount=0; p.count=videorender::kParticleSolidMaxBodies;
    p.bodyRadius=0.0025f;
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
        || lease->scene->objectCount!=1 || lease->scene->objectInstanceCount!=2
        || lease->scene->instancedObject!=lease->scene->objects[0].id
        || lease->scene->objectInstances[0].rotation.y==0) {
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

// Replays one Solid 3D frame together with each body's birth time, which the
// Motion prior uses to tell a surviving body from a respawned one.
struct SolidBodyMotionSample
{
    videorender::ParticleSolidState state;
    std::array<double,videorender::kParticleSolidMaxBodies> births {};
};

inline SolidBodyMotionSample sampleSolidBodyMotion(videorender::ParticleParams p,double time,std::string& error)
{
    p.historyProjectSeconds=time; p.preparedBodies.reset(); p.preparedSolids.reset();
    SolidBodyMotionSample sample;
    videorender::ParticlePointReplayTimes times;
    videorender::replayParticleBodies(p,time,1,nullptr,&error,&sample.state,&times);
    sample.births=times.birthSeconds;
    return sample;
}

// Draws 128 Solid 3D bodies, twice the former bound, as one instanced scene
// object. The grid is symmetric about the view center, so each body's center
// pixel is covered whatever the readback row and column order. readMotion
// returns the raw two-channel Motion vectors in pixels.
template <typename ReadPixels,typename ReadMotion>
bool verifyNativeInstancedSolidBodies(ReadPixels readPixels,ReadMotion readMotion,std::string& error)
{
    struct Layer {
        unsigned texture=0; float opacity=1; bool particleSource=true;
        std::string nativeTextureBackend;
        std::uintptr_t nativeTextureView=0;
        arbitgpu::NativeTextureViewDescriptor nativeTextureDescriptor;
        std::shared_ptr<const void> nativeTextureOwner;
        int texWidth=0,texHeight=0;
    };
    constexpr int bodies=videorender::kParticleSolidMaxBodies, size=128;
    auto p=solidBodyFixture(); p.bodyRadius=0.025f;
    auto state=std::make_shared<videorender::ParticleSolidState>(); state->count=bodies;
    for (int i=0;i<bodies;++i) {
        auto& body=state->bodies[static_cast<std::size_t>(i)];
        body.x=0.5f+(float(i%16)-7.5f)*0.06f; body.y=0.5f+(float(i/16)-3.5f)*0.1f; body.z=0.5f;
        body.identity=1001+i;
    }
    p.preparedSolids=state;
    Layer preview,exported;
    videohelper::geometry::PlanOwnerIdentity owner{1,2,3,4,5,videohelper::geometry::PlanUse::preview};
    if (!videorender::renderParticleSolids(p,owner,7,size,size,preview,error)) return false;
    owner.use=videohelper::geometry::PlanUse::exportRender;
    if (!videorender::renderParticleSolids(p,owner,7,size,size,exported,error)) return false;
    const auto lease=std::static_pointer_cast<const videorender::ParticleSolidFrame>(preview.nativeTextureOwner);
    const auto exportLease=std::static_pointer_cast<const videorender::ParticleSolidFrame>(exported.nativeTextureOwner);
    const auto pixels=readPixels(lease->frame);
    if (pixels.size()!=std::size_t(size)*size*4 || pixels!=readPixels(exportLease->frame)
        || lease->scene->objectCount!=1 || lease->scene->objectInstanceCount!=std::size_t(bodies)
        || lease->scene->instancedObject!=lease->scene->objects[0].id) {
        error="Instanced Solid 3D preview/export pixels or the single instanced body object disagree"; return false;
    }
    // Camera at z=2.2 with a 45 degree vertical field of view; bodies sit at z=0.
    const float halfExtent=2.2f*std::tan(0.3926990817f);
    for (int i=0;i<bodies;++i) {
        const auto& body=state->bodies[static_cast<std::size_t>(i)];
        const int column=static_cast<int>(((body.x-0.5f)/halfExtent*0.5f+0.5f)*size);
        const int row=static_cast<int>(((body.y-0.5f)/halfExtent*0.5f+0.5f)*size);
        if (column<0 || column>=size || row<0 || row>=size
            || pixels[(std::size_t(row)*size+std::size_t(column))*4+3]==0) {
            error="Instanced Solid 3D body "+std::to_string(i)+" was not drawn"; return false;
        }
    }
    auto& backend=arbitgpu::nativeFixtureSceneBackend();
    const auto prepared=backend.prepare(lease->scene);
    if (!prepared.prepared) { error=prepared.error; return false; }
    const auto rendered=backend.render(lease->scene,prepared.resources,size,size);
    if (!rendered.rendered || rendered.stats.objectInstanceDrawCount==0
        || rendered.stats.submittedObjectInstanceCount!=rendered.stats.objectInstanceDrawCount*std::uint32_t(bodies)) {
        error="Instanced Solid 3D must submit all 128 bodies in one draw per pass: "+rendered.error; return false;
    }
    // Motion: each scene carries every body's prior transform. Births use
    // equal (zero) birth times, so every body with a prior match keeps it.
    const std::array<double,bodies> births {};
    arbitgpu::NativeFixtureSceneRuntimeInputs motion;
    motion.imageOutput=renderpassoutput::Output::Motion;
    motion.previousMotion=arbitgpu::sceneMotionSample(motion);
    const auto motionScene=[&](const videorender::ParticleSolidState& previous) {
        const videorender::ParticleSolidMotionPrior prior{previous,births,births,p.bodyRadius};
        return videorender::particleSolidScene(p,*state,7,error,&prior);
    };
    const auto renderMotion=[&](const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
                                std::vector<float>& values) {
        if (!scene) return false;
        const auto ready=backend.prepare(scene);
        if (!ready.prepared) { error=ready.error; return false; }
        const auto frame=backend.render(scene,ready.resources,size,size,motion);
        if (!frame.rendered) { error=frame.error; return false; }
        values=readMotion(frame.frame);
        if (values.size()!=std::size_t(size)*size*2) { error="Instanced Solid 3D Motion readback failed"; return false; }
        return true;
    };
    // Raw Motion is the current minus the prior pixel position. Index i of
    // the result is body i's center pixel vector.
    const auto centers=[&](const std::vector<float>& values) {
        std::vector<std::array<float,2>> result;
        for (int i=0;i<bodies;++i) {
            const auto& body=state->bodies[static_cast<std::size_t>(i)];
            const int column=static_cast<int>(((body.x-0.5f)/halfExtent*0.5f+0.5f)*size);
            const int row=static_cast<int>(((body.y-0.5f)/halfExtent*0.5f+0.5f)*size);
            const auto at=(std::size_t(row)*size+std::size_t(column))*2;
            result.push_back({values[at],values[at+1]});
        }
        return result;
    };
    std::vector<float> resting,moving,movingAgain,born;
    if (!renderMotion(motionScene(*state),resting)) return false;
    for (const auto& vector : centers(resting))
        if (std::abs(vector[0])>1e-3f || std::abs(vector[1])>1e-3f) {
            error="Instanced Solid 3D bodies at rest must report zero Motion"; return false;
        }
    // Every body moved +0.05 along x since the prior frame. At the bodies'
    // depth of about 2.2 that is 0.05*64/(tan(22.5 degrees)*2.2), about 3.5 px.
    auto shifted=std::make_unique<videorender::ParticleSolidState>(*state);
    for (int i=0;i<bodies;++i) shifted->bodies[static_cast<std::size_t>(i)].x-=0.05f;
    if (!renderMotion(motionScene(*shifted),moving) || !renderMotion(motionScene(*shifted),movingAgain)) return false;
    if (moving!=movingAgain) { error="Instanced Solid 3D Motion is not repeatable"; return false; }
    for (const auto& vector : centers(moving))
        if (vector[0]<3.4f || vector[0]>3.65f || std::abs(vector[1])>0.05f) {
            error="Instanced Solid 3D bodies moving along +x must report about 3.5 px of +x Motion"; return false;
        }
    // Bodies in the right half of the grid are born this frame: the prior
    // frame holds only the left half, so the right half reports zero.
    auto leftHalf=std::make_unique<videorender::ParticleSolidState>(*shifted); leftHalf->count=0;
    for (int i=0;i<bodies;++i)
        if (i%16<8) leftHalf->bodies[static_cast<std::size_t>(leftHalf->count++)]=shifted->bodies[static_cast<std::size_t>(i)];
    if (!renderMotion(motionScene(*leftHalf),born)) return false;
    const auto bornCenters=centers(born);
    for (int i=0;i<bodies;++i) {
        const auto& vector=bornCenters[static_cast<std::size_t>(i)];
        const bool survivor=i%16<8;
        if (survivor ? (vector[0]<3.4f || vector[0]>3.65f) : (std::abs(vector[0])>1e-3f || std::abs(vector[1])>1e-3f)) {
            error="Instanced Solid 3D bodies born this frame must report zero Motion while survivors move"; return false;
        }
    }
    // A predecessor frame without prior instance transforms is refused. The
    // first frame has no predecessor, so it reports zero Motion.
    const auto refused=backend.render(lease->scene,prepared.resources,size,size,motion);
    if (refused.rendered || refused.error!="Motion output for instanced scene objects requires each instance's prior transform") {
        error="Instanced Solid 3D Motion without prior transforms must be refused with its precise reason"; return false;
    }
    auto first=motion; first.previousMotion.reset();
    const auto firstFrame=backend.render(lease->scene,prepared.resources,size,size,first);
    if (!firstFrame.rendered) { error=firstFrame.error; return false; }
    const auto still=readMotion(firstFrame.frame);
    if (still.size()!=std::size_t(size)*size*2
        || !std::all_of(still.begin(),still.end(),[](float value) { return value==0; })) {
        error="The first instanced Solid 3D Motion frame must report zero vectors"; return false;
    }
    error.clear();
    return true;
}

// The production draw with a Motion request: renderParticleSolids on the
// prior that the executor replays one project frame earlier. Motion View on
// preview and Export Motion Data on export read the same raw Motion, the
// export layer carries the raw pass for the exporter, a seek leaves Motion
// unchanged, and a draw without a request keeps its Color pixels.
template <typename ReadPixels,typename ReadMotion>
bool verifyNativeSolidBodyMotionProduction(ReadPixels readPixels,ReadMotion readMotion,std::string& error)
{
    struct Layer {
        unsigned texture=0; float opacity=1; bool particleSource=true;
        std::string nativeTextureBackend;
        std::uintptr_t nativeTextureView=0;
        arbitgpu::NativeTextureViewDescriptor nativeTextureDescriptor;
        std::shared_ptr<const void> nativeTextureOwner;
        int texWidth=0,texHeight=0;
    };
    struct ExportLayer : Layer {
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> rawExportFrame;
        std::uint32_t rawExportMask=0;
        int rawExportClipId=-1;
        std::uint32_t rawExportRenderId=0;
        renderpassoutput::Output rawExportImageOutput=renderpassoutput::Output::Color;
    };
    constexpr int size=96;
    constexpr double time=0.5, frameSeconds=1.0/30;
    // Lateral gravity moves both bodies across the view between frames.
    auto base=solidBodyFixture(); base.gravityX=1;
    const auto prepare=[&](videorender::ParticleParams p) {
        const auto now=sampleSolidBodyMotion(p,time,error);
        p.historyProjectSeconds=time; p.preparedBodies.reset();
        p.preparedSolids=std::make_shared<const videorender::ParticleSolidState>(now.state);
        videorender::ParticlePointReplayTimes times; times.birthSeconds=now.births;
        if (error.empty() && videorender::particleSolidMotionRequested(p,true))
            p.preparedSolidMotion=videorender::replayParticleSolidMotionPrior(p,time,frameSeconds,1,nullptr,
                now.state,times,p.bodyRadius,error);
        return p;
    };
    const auto frameOf=[](const auto& layer) {
        const auto lease=std::static_pointer_cast<const videorender::ParticleSolidFrame>(layer.nativeTextureOwner);
        return lease ? lease->frame : std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>{};
    };
    videohelper::geometry::PlanOwnerIdentity preview{1,2,3,4,5,videohelper::geometry::PlanUse::preview};
    auto exportOwner=preview; exportOwner.use=videohelper::geometry::PlanUse::exportRender;

    auto viewParams=base; viewParams.solidMotionView=true;
    const auto view=prepare(viewParams);
    Layer viewLayer;
    if (!error.empty() || !view.preparedSolidMotion
        || !videorender::renderParticleSolids(view,preview,7,size,size,viewLayer,error)) return false;
    const auto motion=readMotion(frameOf(viewLayer));
    float largest=0;
    for (float value : motion) largest=std::max(largest,std::abs(value));
    if (motion.size()!=std::size_t(size)*size*2 || largest<0.05f) {
        error="Motion View on moving Solid 3D bodies must report nonzero Motion"; return false;
    }

    // A seek replays both frames again, so Motion is unchanged.
    auto sought=viewParams;
    sampleSolidBodyFixture(sought,1.9,error); sampleSolidBodyFixture(sought,0.125,error);
    Layer soughtLayer;
    const auto replayed=prepare(sought);
    if (!error.empty() || !videorender::renderParticleSolids(replayed,preview,7,size,size,soughtLayer,error)) return false;
    if (readMotion(frameOf(soughtLayer))!=motion) { error="Solid 3D Motion changed after a seek"; return false; }

    // Export Motion Data writes the raw Motion pass from export renders and
    // leaves the Color image as drawn without a request.
    auto exportParams=base; exportParams.solidExportMotion=true;
    const auto exporting=prepare(exportParams);
    const auto color=prepare(base);
    ExportLayer exported,colorExport; Layer exportPreview;
    if (!error.empty() || color.preparedSolidMotion
        || !videorender::renderParticleSolids(exporting,exportOwner,7,size,size,exported,error)
        || !videorender::renderParticleSolids(exporting,preview,7,size,size,exportPreview,error)
        || !videorender::renderParticleSolids(color,exportOwner,7,size,size,colorExport,error)) return false;
    const auto colorPixels=readPixels(frameOf(colorExport));
    bool visible=false;
    for (std::size_t i=3;i<colorPixels.size();i+=4) visible=visible || colorPixels[i]!=0;
    if (!visible || readPixels(frameOf(exported))!=colorPixels || readPixels(frameOf(exportPreview))!=colorPixels) {
        error="Export Motion Data must leave the Solid 3D Color image unchanged"; return false;
    }
    if (exported.rawExportFrame!=frameOf(exported)
        || exported.rawExportMask!=(1u<<static_cast<unsigned>(renderpassoutput::Output::Motion))
        || exported.rawExportClipId!=4
        // Solid node 7 keys its raw passes by stable id 8. Render 3D node 6 keys
        // its passes by 7, so the two never share a frame key.
        || exported.rawExportRenderId!=render3dimage::renderStableIdForNodeValue(7)
        || exported.rawExportRenderId==render3dimage::renderStableIdForNodeValue(6)
        || exported.rawExportImageOutput!=renderpassoutput::Output::Color
        || colorExport.rawExportMask!=0 || colorExport.rawExportFrame) {
        error="Only an export render with Export Motion Data carries the raw Solid 3D Motion pass"; return false;
    }
    if (readMotion(exported.rawExportFrame)!=motion) {
        error="Exported raw Solid 3D Motion must match the Motion View preview"; return false;
    }
    Layer plain;
    if (videorender::renderParticleSolids(exporting,exportOwner,7,size,size,plain,error)
        || error!="Raw Solid 3D Motion export requires a layer that retains the native frame owner") {
        error="Raw Solid 3D Motion export into a layer without raw export fields must be refused"; return false;
    }

    // A frame with no bodies or colliders still writes the raw Motion pass, so
    // the Motion sequence has no hole. The placeholder draws nothing: zero Motion.
    const auto zeroMotion=[size](const std::vector<float>& values) {
        return values.size()==std::size_t(size)*size*2
            && std::all_of(values.begin(),values.end(),[](float value) { return value==0; });
    };
    auto emptyParams=exportParams;
    emptyParams.preparedBodies.reset();
    emptyParams.preparedSolids=std::make_shared<const videorender::ParticleSolidState>();
    auto emptyMotion=std::make_shared<videorender::ParticleSolidMotionFrame>();
    emptyMotion->previousBodyRadius=emptyParams.bodyRadius;
    emptyParams.preparedSolidMotion=emptyMotion;
    ExportLayer emptyExport;
    if (!videorender::renderParticleSolids(emptyParams,exportOwner,7,size,size,emptyExport,error)
        || !emptyExport.rawExportFrame
        || emptyExport.rawExportMask!=(1u<<static_cast<unsigned>(renderpassoutput::Output::Motion))
        || emptyExport.rawExportRenderId!=render3dimage::renderStableIdForNodeValue(7)
        || !zeroMotion(readMotion(emptyExport.rawExportFrame))) {
        error="An export frame with no Solid 3D bodies must still write a zero raw Motion pass"; return false;
    }
    // Its Color stays hidden, as a no-request empty frame is, so the backdrop
    // plate of the placeholder scene never reaches the composite.
    if (emptyExport.opacity!=0) {
        error="An export frame with no Solid 3D bodies must hide Color like a frame with no request"; return false;
    }

    // Frame 0 has no earlier replay. Its prior is empty, so every body is born
    // this frame and the raw Motion pass is zero.
    const auto start=sampleSolidBodyMotion(exportParams,0,error);
    auto startParams=exportParams;
    startParams.historyProjectSeconds=0; startParams.preparedBodies.reset();
    startParams.preparedSolids=std::make_shared<const videorender::ParticleSolidState>(start.state);
    videorender::ParticlePointReplayTimes startTimes; startTimes.birthSeconds=start.births;
    startParams.preparedSolidMotion=videorender::replayParticleSolidMotionPrior(startParams,0,frameSeconds,1,nullptr,
        start.state,startTimes,startParams.bodyRadius,error);
    ExportLayer startExport;
    if (!error.empty() || !startParams.preparedSolidMotion
        || !videorender::renderParticleSolids(startParams,exportOwner,7,size,size,startExport,error)
        || !startExport.rawExportFrame
        || startExport.rawExportMask!=(1u<<static_cast<unsigned>(renderpassoutput::Output::Motion))
        || !zeroMotion(readMotion(startExport.rawExportFrame))) {
        error="Frame 0 of Solid 3D Motion export must write a zero raw Motion pass"; return false;
    }
    error.clear();
    return true;
}

// A static scene with Solid 3D bodies appended as their own objects, as the
// imported Solid 3D overlay draws it. The scene keeps its place, each body
// takes its own prior transform, and both draw in one Motion pass with one
// depth buffer. readMotion returns the raw two-channel Motion in pixels.
template <typename ReadMotion>
bool verifyNativeComposedSolidBodyMotion(ReadMotion readMotion,std::string& error)
{
    constexpr int size=128;
    auto p=solidBodyFixture(); p.colliderThickness=0.3f;
    // Scale 1 and center 0.5 place appended bodies as particleSolidScene does.
    auto binding=std::make_shared<videorender::ParticleGeometryBinding>(); binding->scale=1;
    p.geometryBinding=binding;
    // The static scene: one collider sphere of radius 0.3 at world z -0.5.
    videorender::ParticleSolidState backdrop;
    backdrop.colliders.pointCount=1; backdrop.colliders.solidPoints[0]={0.5f,0.5f,0.0f};
    const auto source=videorender::particleSolidScene(p,backdrop,7,error);
    if (!source || source->objectCount!=1) { if (error.empty()) error="Composed Solid 3D backdrop scene failed"; return false; }
    // Body 1001 moved +0.05 along x in front of the backdrop. Body 1002 is
    // born this frame: the prior frame does not hold it.
    videorender::ParticleSolidState current; current.count=2;
    current.bodies[0].x=0.5f; current.bodies[0].y=0.5f; current.bodies[0].z=0.5f; current.bodies[0].identity=1001;
    current.bodies[1].x=0.5f; current.bodies[1].y=0.8f; current.bodies[1].z=0.5f; current.bodies[1].identity=1002;
    videorender::ParticleSolidState previous; previous.count=1;
    previous.bodies[0]=current.bodies[0]; previous.bodies[0].x-=0.05f;
    const std::array<double,videorender::kParticleSolidMaxBodies> births {};
    const videorender::ParticleSolidMotionPrior prior{previous,births,births,p.bodyRadius};
    const auto scene=videorender::appendParticleSolidBodies(source,p,current,7,{},error,&prior);
    if (!scene) return false;
    if (scene->objectCount!=3 || scene->previousObjectTransformCount!=3
        || !arbitgpu::sameSceneTransform(scene->previousObjectTransforms[0],scene->objects[0].transform)
        || !arbitgpu::sameSceneTransform(scene->previousObjectTransforms[2],scene->objects[2].transform)
        || std::abs(scene->objects[1].transform.translation.x-scene->previousObjectTransforms[1].translation.x-0.05f)>1e-6f) {
        error="Appended Solid 3D bodies must carry their own prior transforms beside the unchanged scene"; return false;
    }
    auto& backend=arbitgpu::nativeFixtureSceneBackend();
    arbitgpu::NativeFixtureSceneRuntimeInputs motion;
    motion.imageOutput=renderpassoutput::Output::Motion;
    motion.previousMotion=arbitgpu::sceneMotionSample(motion);
    const auto renderMotion=[&](const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& drawn,
                                const arbitgpu::NativeFixtureSceneRuntimeInputs& inputs,std::vector<float>& values) {
        const auto ready=backend.prepare(drawn);
        if (!ready.prepared) { error=ready.error; return false; }
        const auto frame=backend.render(drawn,ready.resources,size,size,inputs);
        if (!frame.rendered) { error=frame.error; return false; }
        values=readMotion(frame.frame);
        if (values.size()!=std::size_t(size)*size*2) { error="Composed Solid 3D Motion readback failed"; return false; }
        return true;
    };
    const auto at=[](const std::vector<float>& values,int row,int column) {
        const auto index=(std::size_t(row)*size+std::size_t(column))*2;
        return std::array<float,2>{values[index],values[index+1]};
    };
    // Camera at z=2.2 with a 45 degree vertical field of view. The moving body
    // covers the view center: +0.05 at a depth near 2.2 is about 3.5 px. Column
    // 75 sees only the backdrop, at a depth near 2.6.
    std::vector<float> moving;
    if (!renderMotion(scene,motion,moving)) return false;
    const auto body=at(moving,64,64), still=at(moving,64,75);
    if (body[0]<3.3f || body[0]>3.9f || std::abs(body[1])>0.05f) {
        error="A moving Solid 3D body over a static scene must report its own Motion"; return false;
    }
    if (std::abs(still[0])>1e-3f || std::abs(still[1])>1e-3f) {
        error="The static scene beside a moving Solid 3D body must report zero Motion"; return false;
    }
    // Control: the same pixel reports Motion once the backdrop has its own
    // prior, so the zero above is the drawn static scene, not empty space.
    auto shiftedScene=std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*scene);
    shiftedScene->previousObjectTransforms[0].translation.x-=0.05f;
    std::vector<float> shifted;
    if (!renderMotion(shiftedScene,motion,shifted)) return false;
    if (at(shifted,64,75)[0]<2.0f) { error="A scene object with its own prior transform must report Motion"; return false; }
    // Retained scene positions cover the scene part only. The appended bodies
    // past them keep their current positions and their own prior transforms.
    auto positions=std::make_shared<std::vector<HarmonicMIDI::grid::SceneVec3>>();
    for (std::size_t i=0;i<source->vertexCount;++i) positions->push_back(scene->vertices[i].position);
    auto retained=motion; retained.previousMotion->scenePositions=positions;
    std::vector<float> prefixed;
    if (!renderMotion(scene,retained,prefixed)) return false;
    if (prefixed!=moving) { error="Retained scene positions must not change appended Solid 3D body Motion"; return false; }
    auto straddling=std::make_shared<std::vector<HarmonicMIDI::grid::SceneVec3>>(*positions);
    straddling->push_back(scene->vertices[source->vertexCount].position);
    auto refusedInputs=motion; refusedInputs.previousMotion->scenePositions=straddling;
    const auto ready=backend.prepare(scene);
    const auto refused=ready.prepared ? backend.render(scene,ready.resources,size,size,refusedInputs)
        : arbitgpu::NativeFixtureSceneSubmission{};
    if (refused.rendered || refused.error!="Motion output prior Geometry3D positions do not match the scene vertices") {
        error="Retained positions that split an appended body must be refused"; return false;
    }
    // The first frame has no predecessor, so bodies and scene report zero.
    auto first=motion; first.previousMotion.reset();
    std::vector<float> start;
    if (!renderMotion(scene,first,start)) return false;
    if (!std::all_of(start.begin(),start.end(),[](float value) { return value==0; })) {
        error="The first composed Solid 3D Motion frame must report zero vectors"; return false;
    }
    error.clear();
    return true;
}
