#include "support/diffraction_native_transport_fixture.h"
#include "support/diffraction_reference_oracle.h"
#include "../../shared/DiffractionMaterialPresets.h"
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace
{
namespace tt=triangletransporttests;
namespace dt=diffractiontransport;
int failures=0,checks=0;
void check(bool condition,const char* message)
{
    ++checks;if(!condition){++failures;std::fprintf(stderr,"FAIL: %s\n",message);}
}
bool near(double a,double b,double tolerance=2e-3)
{
    return std::abs(a-b)<=tolerance*std::max(1e-8,std::max(std::abs(a),std::abs(b)));
}
std::optional<dt::Result> evaluate(tt::Fixture& fixture,dt::Failure& failure)
{
    std::string error;
    if(!fixture.prepare(error)){check(false,error.c_str());return std::nullopt;}
    return dt::evaluate(fixture.input,[&](const dt::Ray& ray){return fixture.query(ray);},failure);
}
void compare(const dt::Result& actual,const dt::Result& expected)
{
    check(near(actual.receiverArea,expected.receiverArea,1e-12),"actual receiver area matches independent triangles");
    check(near(actual.formFactor,expected.formFactor),"finite triangle form factor matches independent Duffy quadrature");
    for(unsigned band=0;band<8;++band)
    {
        check(near(actual.receiverIncidentPower[band],expected.receiverIncidentPower[band]),"each authored band reaches receiver with normal/light cosine");
        check(near(actual.meanReceiverRadiance[band],expected.meanReceiverRadiance[band]),"each receiver rho produces Lambertian radiance independently");
        check(near(actual.arrivingPower[band],expected.arrivingPower[band]),"each band reaches grating through actual triangle exchange");
        check(actual.arrivingPower[band]>=0&&actual.arrivingPower[band]<=actual.receiverOutgoingPower[band],"per-band arriving energy never exceeds receiver outgoing energy");
    }
}
void geometryAndVisibility()
{
    dt::Failure failure;
    auto rectangle=tt::fixture();const auto expected=tt::oracle(*rectangle.scene,rectangle.input);
    const auto actual=evaluate(rectangle,failure);check(bool(actual),"opposed actual rectangle triangles admit bounded transport");
    if(!actual)return;
    compare(*actual,expected);
    const double analytic=tt::rectangleFactor(1,.8,2);
    check(near(expected.formFactor,analytic,1e-7),"independent triangle quadrature matches analytic finite-rectangle factor");
    check(near(actual->formFactor,analytic),"production finite-triangle exchange matches analytic rectangle control");
    for(unsigned band=0;band<8;++band)
        check(near(actual->arrivingPower[band],dt::kPi*actual->receiverArea*actual->formFactor*actual->meanReceiverRadiance[band],1e-10),"pi A_R F_R_G L_R matches arriving power for uniform planar receiver");
    auto spectralControl=tt::fixture();spectralControl.input.receiver.reflectance->coefficients[4]=0;
    spectralControl.input.source.radiance[2]*=2;
    const auto changedSpectrum=evaluate(spectralControl,failure);
    check(bool(changedSpectrum),"independent band perturbations retain admitted geometry");
    if(changedSpectrum)for(unsigned band=0;band<8;++band)
    {
        const double factor=band==4?0:band==2?2:1;
        check(near(changedSpectrum->arrivingPower[band],factor*actual->arrivingPower[band],1e-10),
              "source and reflectance edits affect only the authored wavelength band");
    }
    auto moved=tt::fixture(false,.7);const auto movedResult=evaluate(moved,failure);
    check(bool(movedResult)&&movedResult->formFactor<actual->formFactor,"moving actual receiver changes finite geometry exchange");
    if(movedResult)compare(*movedResult,tt::oracle(*moved.scene,moved.input));
    auto bent=tt::fixture(true);const auto bentResult=evaluate(bent,failure);
    check(bool(bentResult)&&!near(bentResult->formFactor,actual->formFactor,1e-3),"nonplanar receiver uses each actual triangle normal and area");
    if(bentResult)compare(*bentResult,tt::oracle(*bent.scene,bent.input));
    auto segmentBlock=tt::fixture(false,0,1),lightBlock=tt::fixture(false,0,2);
    const auto blocked=evaluate(segmentBlock,failure),unlit=evaluate(lightBlock,failure);
    check(bool(blocked)&&blocked->formFactor==0,"opaque intervening triangles block both reciprocal receiver/grating legs");
    check(bool(unlit)&&unlit->formFactor>0,"directional-source blocker leaves receiver/grating segment visible");
    if(blocked)compare(*blocked,tt::oracle(*segmentBlock.scene,segmentBlock.input));
    if(unlit)compare(*unlit,tt::oracle(*lightBlock.scene,lightBlock.input));
    for(unsigned band=0;band<8;++band)
    {
        check(blocked&&blocked->arrivingPower[band]==0&&blocked->receiverIncidentPower[band]>0,"segment blocker removes arriving power in all bands while receiver stays illuminated");
        check(unlit&&unlit->arrivingPower[band]==0&&unlit->receiverIncidentPower[band]==0,"light-leg blocker removes receiver irradiance and arriving power in all bands");
    }
}
void refusals()
{
    auto fixture=tt::fixture();std::string error;check(fixture.prepare(error),"refusal fixture prepares released all-surfaces scene query");
    const auto reject=[&](dt::Input input,dt::Failure expected,const char* message)
    {
        dt::Failure failure;const auto result=dt::evaluate(input,[&](const dt::Ray& ray){return fixture.query(ray);},failure);
        check(!result&&failure==expected,message);
    };
    auto input=fixture.input;input.receiver.programIdentity.object=999;reject(input,dt::Failure::IdentityMismatch,"wrong program/object identity refuses");
    input=fixture.input;input.receiver.parameterMaterial=999;reject(input,dt::Failure::IdentityMismatch,"wrong parameter/material identity refuses");
    input=fixture.input;input.receiver.programRevision=2;reject(input,dt::Failure::RevisionMismatch,"stale program revision refuses");
    input=fixture.input;input.receiver.reflectance.reset();reject(input,dt::Failure::MissingReflectance,"missing authored rho refuses without RGB inference");
    for(double invalid:{-1.0,17.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
    {input=fixture.input;input.source.radiance[3]=invalid;reject(input,dt::Failure::InvalidSpectrum,"invalid independent source radiance refuses");}
    for(float invalid:{-1.0f,1.01f,std::numeric_limits<float>::quiet_NaN()})
    {input=fixture.input;input.receiver.reflectance->coefficients[5]=invalid;reject(input,dt::Failure::InvalidSpectrum,"invalid receiver rho refuses");}
    input=fixture.input;input.receiver.opacity=.5;reject(input,dt::Failure::UnsupportedReceiver,"transparent receiver refuses");
    input=fixture.input;input.receiver.transmission=.1;reject(input,dt::Failure::UnsupportedReceiver,"transmitting receiver refuses");
    input=fixture.input;input.receiver.masked=true;reject(input,dt::Failure::UnsupportedReceiver,"masked receiver refuses");
    input=fixture.input;input.receiver.texturedAlpha=true;reject(input,dt::Failure::UnsupportedReceiver,"textured-alpha receiver refuses");
    input=fixture.input;input.receiver.deformed=true;reject(input,dt::Failure::UnsupportedReceiver,"unevaluated receiver deformation refuses");
    input=fixture.input;input.receiverTriangles[0].normal={};reject(input,dt::Failure::InvalidNormal,"zero actual triangle normal refuses");
    input=fixture.input;input.gratingTriangles[0].normal[0]=std::numeric_limits<double>::quiet_NaN();reject(input,dt::Failure::InvalidNormal,"nonfinite grating normal refuses");
    input=fixture.input;input.receiverTriangles[0].normal={0,0,-1};reject(input,dt::Failure::InvalidNormal,"normal inconsistent with actual winding refuses");
    input=fixture.input;input.source.traceMaximumDistance=std::numeric_limits<double>::infinity();reject(input,dt::Failure::InvalidBounds,"infinite trace bound refuses");
    input=fixture.input;input.rayMinimumDistance=21;reject(input,dt::Failure::InvalidBounds,"inverted ray interval refuses");
    input=fixture.input;input.source.traceMaximumDistance=1;reject(input,dt::Failure::InvalidBounds,"finite trace maximum cannot truncate receiver/grating connection");
    input=fixture.input;input.queryBudget=2;reject(input,dt::Failure::BudgetExhausted,"query budget exhaustion refuses entire result after partial work");
    input=fixture.input;input.subdivisions=17;reject(input,dt::Failure::BudgetExhausted,"oversized quadrature refuses");
    input=fixture.input;input.receiverTriangles.resize(64,input.receiverTriangles[0]);input.gratingTriangles.resize(64,input.gratingTriangles[0]);
    reject(input,dt::Failure::BudgetExhausted,"finite pair-work cap refuses before visibility work");
    for(int kind=0;kind<5;++kind)
    {
        dt::Failure failure;
        const auto result=dt::evaluate(fixture.input,[&](const dt::Ray& ray)
        {
            auto hit=fixture.query(ray);
            if(kind==0)hit.status=-2;
            if(kind==1)hit.sceneRevision=2;
            if(kind==2&&hit.status==1)hit.supportedOpaque=false;
            if(kind==3&&hit.status==1)hit.identity.material=999;
            if(kind==4)hit.status=0;
            return hit;
        },failure);
        const std::array<dt::Failure,5> expected{dt::Failure::QueryRefused,dt::Failure::RevisionMismatch,dt::Failure::UnsupportedHit,dt::Failure::IdentityMismatch,dt::Failure::MissingEndpoint};
        check(!result&&failure==expected[kind],"invalid query, revision, unsupported hit, wrong hit material or missing endpoint refuses full transport");
    }
}
void gratingResponse()
{
    auto fixture=tt::fixture();const auto predicted=tt::oracle(*fixture.scene,fixture.input);
    dt::Failure failure;const auto transported=evaluate(fixture,failure);if(!transported)return;
    auto description=diffractionmaterial::makeAluminiumBinaryGratingPreset();std::string error;
    const auto material=diffractionmaterial::admit(description,error);check(bool(material),"physical grating admits independent transport spectrum");if(!material)return;
    namespace reference=diffractionmaterial::reference;
    reference::EvaluationInput input;input.incidentSpectrum=predicted.meanGratingIrradiance;
    reference::EvaluationFailure gratingFailure;
    const auto expected=reference::evaluate(*material,input,gratingFailure);
    input.incidentSpectrum=transported->meanGratingIrradiance;
    const auto actual=reference::evaluate(*material,input,gratingFailure);
    check(bool(actual)&&bool(expected),"physical discrete grating orders consume eight transported bands");if(!actual||!expected)return;
    for(unsigned band=0;band<8;++band)
    {
        const auto& a=actual->spectralSamples[band];const auto& e=expected->spectralSamples[band];
        check(near(a.incidentEnergy,e.incidentEnergy)&&near(a.resolvedReflectedEnergy,e.resolvedReflectedEnergy),"per-band incident and resolved grating energy matches independently predicted arriving spectrum");
        check(near(a.incidentEnergy,a.resolvedReflectedEnergy+a.unresolvedReflectedEnergy+a.absorbedEnergy,1e-10),"each grating band conserves arriving energy");
    }
    // A moved receiver also changes the actual incoming direction in the local
    // grating frame, independently of the eight-band energy scaling.
    const auto moved=tt::fixture(false,.7);
    const auto centre=[](const std::vector<dt::Triangle>& triangles)
    {
        dt::Point result{};for(const auto& triangle:triangles)for(const auto& p:triangle.vertices)
            for(unsigned axis=0;axis<3;++axis)result[axis]+=p[axis]/(3*triangles.size());
        return result;
    };
    auto incoming=tt::difference(centre(moved.input.receiverTriangles),centre(moved.input.gratingTriangles));
    // The grating's winding gives local tangent +X and normal -Z.
    incoming[2]=-incoming[2];const auto direction=tt::normalized(incoming);
    const auto centred=reference::solveOrder({0,0,1},{1,0},1000,500,1);
    const auto shifted=reference::solveOrder({direction[0],0,direction[2]},{1,0},1000,500,1);
    check(centred.disposition==reference::OrderDisposition::Propagating&&shifted.disposition==reference::OrderDisposition::Propagating&&
          std::abs(centred.outgoingDirection.x-shifted.outgoingDirection.x)>.1,"actual incoming geometry direction changes resolved grating order");
    const auto blue=reference::solveOrder({0,0,1},{1,0},1000,380,1),red=reference::solveOrder({0,0,1},{1,0},1000,700,1);
    check(std::abs(blue.outgoingDirection.x-red.outgoingDirection.x)>.3,"authored wavelength changes physical order direction");
}
// These probes test shared capability routing only; they never render pixels.
class DefaultReceiverCapabilityBackend : public arbitgpu::NativeFixtureSceneBackend
{
public:
    arbitgpu::BackendInfo info() const override { return {}; }
    arbitgpu::NativeFixtureScenePreparation prepare(const std::shared_ptr<const tt::Scene>&,
        const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>&,
        arbitgpu::NativeFixturePrepareUse = arbitgpu::NativeFixturePrepareUse::Blocking) override { return {}; }
    arbitgpu::NativeFixtureSceneSubmission render(const std::shared_ptr<const tt::Scene>&,
        const std::shared_ptr<const arbitgpu::NativeFixtureSceneResources>&,std::uint32_t,std::uint32_t,
        arbitgpu::NativeFixtureSceneRuntimeInputs) override { return {}; }
};
class SelectedReceiverCapabilityBackend final : public DefaultReceiverCapabilityBackend
{
public:
    bool supportsAuthoredSpectralReceiverTransport() const noexcept override
    {
        const arbitgpu::diffractiontransportnative::ConsumerContract consumer{
            arbitgpu::NativeFixtureMaterialBackend::OpenGl,1,8,3,true,true};
        return arbitgpu::diffractiontransportnative::consumerCanRender(consumer,arbitgpu::NativeFixtureMaterialBackend::OpenGl);
    }
};

void nativeTableAdmission()
{
    namespace nt=arbitgpu::diffractiontransportnative;
    namespace ov=arbitgpu::opaquevisibility;
    std::string error;auto f=tt::nativeFixture();nt::Prepared prepared;nt::Frame frame;
    DefaultReceiverCapabilityBackend unavailable;SelectedReceiverCapabilityBackend selected;
    check(!unavailable.supportsAuthoredSpectralReceiverTransport()
        && !arbitgpu::validNativeFixtureMaterialPrograms(f.program,*f.scene,f.program.backend,
            unavailable.supportsAuthoredSpectralReceiverTransport()),"default backend capability keeps mixed v3 admission guarded");
    check(selected.supportsAuthoredSpectralReceiverTransport()
        && arbitgpu::validNativeFixtureMaterialPrograms(f.program,*f.scene,f.program.backend,
            selected.supportsAuthoredSpectralReceiverTransport()),"an exact selected backend capability permits mixed v3 envelope admission only");
    ov::Prepared opaque;
    check(!ov::prepareScene(*f.scene,&f.program,{},opaque,error),"default opaque-shadow preparation keeps v3 grating guarded");
    check(ov::prepareScene(*f.scene,&f.program,{},opaque,error,ov::AdmissionMode::opaqueOnly,false,true)
        && opaque.mode==ov::AdmissionMode::opaqueOnly && opaque.objectMesh[1]>=0,
        "explicit v3 consumer admission preserves the released opaque-only query mode");
    auto wrong=f.program;wrong.programIdentity+="stale";
    check(!ov::prepareScene(*f.scene,&wrong,{},opaque,error,ov::AdmissionMode::opaqueOnly,false,true),
        "explicit opaque-shadow consumer flag cannot admit a stale exact grating receipt");
    check(!arbitgpu::validNativeFixtureMaterialPrograms(wrong,*f.scene,wrong.backend,
            selected.supportsAuthoredSpectralReceiverTransport()),"selected capability cannot admit stale mixed grating identity");
    auto legacy=f.program;diffractionmaterial::LightingDescription legacyLighting;
    legacyLighting.pathCount=1;legacyLighting.paths[0].kind=diffractionmaterial::LightingPathKind::Direct;
    legacyLighting.paths[0].incident.radiance=f.program.diffractionLightingAdmission->description().paths[0].incident.radiance;
    const auto legacyAdmitted=diffractionmaterial::AdmittedLightingPlan::admit(legacyLighting,error);
    if(legacyAdmitted)
    {
        legacy.diffractionLightingAdmission=std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(*legacyAdmitted);
        legacy.diffractionMaximumBounceDepth=0;
        auto parameters=legacy.diffractionPaths[0].material;
        for(unsigned b=0;b<8;++b)parameters.spectral[b].y=legacyLighting.paths[0].incident.radiance[b];
        legacy.diffractionPaths[0]=diffractionmaterial::makeGpuLightingPath(parameters,legacyLighting.paths[0]);
        legacy.programIdentity=arbitgpu::nativeFixtureDiffractionProgramIdentity(legacy.bindingDigest,*legacyAdmitted);
    }
    ov::Prepared defaultLegacy,selectedLegacy;
    check(legacyAdmitted && ov::prepareScene(*f.scene,&legacy,{},defaultLegacy,error)
        && ov::prepareScene(*f.scene,&legacy,{},selectedLegacy,error,ov::AdmissionMode::opaqueOnly,false,true)
        && defaultLegacy.texels==selectedLegacy.texels && defaultLegacy.objectMesh==selectedLegacy.objectMesh,
        "released v1 opaque visibility payload stays byte-identical with consumer flag disabled or enabled");
    nt::ConsumerContract consumer;
    check(!nt::consumerCanRender(consumer,arbitgpu::NativeFixtureMaterialBackend::OpenGl),"shared admission does not imply an OpenGL consumer");
    consumer={arbitgpu::NativeFixtureMaterialBackend::OpenGl,nt::kTableVersion,nt::kSampleStride,3,true,true};
    check(nt::consumerCanRender(consumer,arbitgpu::NativeFixtureMaterialBackend::OpenGl)
        && !nt::consumerCanRender(consumer,arbitgpu::NativeFixtureMaterialBackend::Metal),"GL and Metal consumption contracts select independently");
    consumer.wholeOutputRefusalReadback=false;
    check(!nt::consumerCanRender(consumer,arbitgpu::NativeFixtureMaterialBackend::OpenGl),"shader/table support without whole-output query refusal cannot publish");
    std::vector<float> readback(16,0);check(nt::validateWholeOutput(readback,2,2,error),"complete finite native diagnostic readback admits");
    readback[5]=std::numeric_limits<float>::quiet_NaN();
    check(!nt::validateWholeOutput(readback,2,2,error),"one invalid-query diagnostic refuses the whole output");
    readback.resize(15);check(!nt::validateWholeOutput(readback,2,2,error),"truncated diagnostic readback refuses all AOV publication");
    check(!arbitgpu::validNativeFixtureDiffractionProgram(f.program,f.program.backend,f.program.object),
          "native v3 rendering stays refused until a backend explicitly consumes its table");
    check(arbitgpu::validNativeFixtureDiffractionProgram(f.program,f.program.backend,f.program.object,true),
          "exact native v3 envelope permits table preparation independently of rendering");
    check(nt::prepareScene(*f.scene,f.program,1,prepared,error),"actual receiver/grating table admits exact authored ownership");
    if(!prepared.scene)return;
    const auto instances=tt::nativeInstances(f);
    check(nt::buildFrame(prepared,instances,128,128,frame,error),"bounded native table prepares both reciprocal legs and directional queries");
    if(frame.samples.empty())return;
    check(frame.sampleControl==std::array<float,4>{8,8,8,1},"two actual triangles per role produce eight equal-area samples");
    const auto triangles=tt::worldTriangles(*f.scene);
    double receiverArea=0,gratingArea=0;
    for(std::size_t i=0;i<frame.samples.size();++i)
    {
        const auto& sample=frame.samples[i];(sample.grating?gratingArea:receiverArea)+=sample.area;
        check(sample.object==(sample.grating?102u:101u) && sample.material==(sample.grating?202u:201u),
              "table rows bind stable object/material identity independently of draw order");
        check(near(sample.normal[2],sample.grating?-1:1,1e-7),"table uses actual triangle outward normals");
        const auto offset=i*nt::kSampleStride;
        check(frame.texels[offset+2][0]==float(sample.object) && frame.texels[offset+2][2]==float(sample.material),
              "GPU sample identity words preserve exact scene IDs");
        if(!sample.grating)for(unsigned b=0;b<8;++b)
            check(frame.texels[offset+4+b/4][b%4]==f.input.receiver.reflectance->coefficients[b],
                  "GPU receiver table carries independent raw authored rho bands");
        dt::Ray ray{{sample.position[0],sample.position[1],sample.position[2]},f.input.source.direction,1e-5,20};
        const auto oracleHit=tt::intersection(triangles,ray);
        const auto hit=ov::closestHitCpu(prepared.query,frame.query,sample.position,
            {float(ray.direction[0]),float(ray.direction[1]),float(ray.direction[2])},float(ray.minimum),float(ray.maximum));
        check(hit.status==oracleHit.status && (!hit.status || (hit.objectId==oracleHit.identity.object && hit.materialId==oracleHit.identity.material)),
              "native sample directional leg agrees with independent double world-triangle oracle");
    }
    check(near(receiverArea,.8,1e-7)&&near(gratingArea,.8,1e-7),"native sample areas sum to actual finite receiver and grating areas");
    auto moved=tt::nativeFixture(false,.7);nt::Prepared movedPrepared;nt::Frame movedFrame;
    check(nt::prepareScene(*moved.scene,moved.program,1,movedPrepared,error)
          && nt::buildFrame(movedPrepared,tt::nativeInstances(moved),128,128,movedFrame,error)
          && movedFrame.identity!=frame.identity && movedFrame.samples[0].position!=frame.samples[0].position,
          "moving the actual receiver changes the exact native world sample/query receipt");
    auto reflected=tt::nativeFixture();auto edited=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*reflected.program.objectPrograms[0]);
    edited->spectralReflectance->coefficients[4]=0;reflected.program.objectPrograms[0]=edited;
    nt::Prepared editedPrepared;nt::Frame editedFrame;
    check(nt::prepareScene(*reflected.scene,reflected.program,1,editedPrepared,error)
        && nt::buildFrame(editedPrepared,tt::nativeInstances(reflected),128,128,editedFrame,error)
        && editedFrame.identity!=frame.identity && editedFrame.texels[5][0]==0,
        "raw rho edit changes receiver identity and exact GPU table without RGB inference");
    for(int blocker:{1,2})
    {
        auto blocked=tt::nativeFixture(false,0,blocker);nt::Prepared bp;nt::Frame bf;
        check(nt::prepareScene(*blocked.scene,blocked.program,1,bp,error)
            && nt::buildFrame(bp,tt::nativeInstances(blocked),128,128,bf,error),
            "supported opaque segment and directional-source blockers retain a complete query table");
        if(bf.samples.empty())continue;
        const auto& r=bf.samples[0];const auto& g=bf.samples[8];
        const auto direction=blocker==1?nt::sub(g.position,r.position):nt::Point{bf.sourceDirectionTrace[0],bf.sourceDirectionTrace[1],bf.sourceDirectionTrace[2]};
        const auto hit=ov::closestHitCpu(bp.query,bf.query,r.position,direction,1e-5,20);
        dt::Point doubleDirection{direction[0],direction[1],direction[2]};doubleDirection=tt::normalized(doubleDirection);
        const auto expected=tt::intersection(tt::worldTriangles(*blocked.scene),{{r.position[0],r.position[1],r.position[2]},doubleDirection,1e-5,20});
        check(hit.status==1 && hit.objectId==103 && expected.identity.object==103,
            "each blocker is an exact scene hit verified by the independent triangle oracle");
    }
    const auto rejected=[&](auto mutate,const char* message)
    {
        auto candidate=tt::nativeFixture();mutate(candidate);nt::Prepared p;
        check(!nt::prepareScene(*candidate.scene,candidate.program,1,p,error),message);
    };
    rejected([](auto& candidate){candidate.scene->objects[0].material={202};},"stale receiver material identity refuses the entire table");
    rejected([](auto& candidate){auto p=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*candidate.program.objectPrograms[0]);p->sceneRevision=2;candidate.program.objectPrograms[0]=p;},"stale receiver scene revision refuses");
    rejected([](auto& candidate){auto p=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*candidate.program.objectPrograms[0]);p->spectralReflectance.reset();candidate.program.objectPrograms[0]=p;},"missing authored receiver rho refuses");
    rejected([](auto& candidate){candidate.scene->materials[0].opacity=.5f;},"partial receiver coverage refuses");
    rejected([](auto& candidate){candidate.scene->vertices[0].color[3]=.5f;},"partial vertex coverage refuses");
    auto omitted=instances;omitted.pop_back();nt::Frame discarded;
    check(!nt::buildFrame(prepared,omitted,128,128,discarded,error),"query omission of a composed object refuses");
    auto repeated=instances;
    for(unsigned i=0;i<4;++i)repeated.push_back(instances[0]);
    check(!nt::buildFrame(prepared,repeated,128,128,discarded,error),"expanded receiver triangle instances exceed the table cap before GPU allocation");
    // Change only endpoint metadata, retaining its actual position and finite
    // distance. A different valid scene object at that distance is not a blocker.
    const auto metadata=prepared.query.texels.size();
    for(unsigned endpoint=0;endpoint<2;++endpoint)
    {
        auto stale=frame;
        const unsigned other=1-endpoint;
        const auto& object=f.scene->objects[other];
        stale.query.texels[metadata+endpoint]={float(object.id.value),0,float(object.material.value),0};
        const auto& receiver=stale.samples[0];const auto& grating=stale.samples[8];
        const auto direction=endpoint==1?nt::sub(grating.position,receiver.position):nt::sub(receiver.position,grating.position);
        const auto& origin=endpoint==1?receiver:grating;
        const float distance=std::sqrt(nt::dot(direction,direction));
        const auto hit=ov::closestHitCpu(prepared.query,stale.query,origin.position,direction,1e-5,distance+1e-5);
        check(hit.status==1 && hit.objectId==object.id.value && hit.materialId==object.material.value
              && std::abs(hit.distance-distance)<=1e-5,
              "mutated query retains a complete finite endpoint hit with a different admitted object identity");
        check(!nt::validateQueries(prepared,stale,error),
              "wrong object at either reciprocal endpoint refuses instead of becoming an opaque blocker");
    }
    auto wrongMaterial=frame;wrongMaterial.query.texels[metadata+1][2]=201;
    check(!nt::validateQueries(prepared,wrongMaterial,error),"mutated endpoint material identity refuses the whole query frame");
    auto wrongInstance=frame;wrongInstance.query.texels[metadata+64+4+3][3]=0;
    check(!nt::validateQueries(prepared,wrongInstance,error),"mutated endpoint instance identity refuses the whole query frame");
    auto wrongTriangle=frame;
    const auto& mesh=prepared.query.meshes[std::size_t(prepared.query.objectMesh[1])];
    const auto leaf=std::size_t(wrongTriangle.query.texels[mesh.root+1][3])-1;
    for(unsigned t=0;t<2;++t)wrongTriangle.query.texels[leaf+t*3+1][3]=float(1-t);
    check(!nt::validateQueries(prepared,wrongTriangle,error),"mutated endpoint triangle ordinal refuses the whole query frame");
    auto wrongNormal=frame;for(auto& component:wrongNormal.samples[8].normal)component=-component;
    check(!nt::validateQueries(prepared,wrongNormal,error),"stale endpoint sample normal refuses the whole query frame");
    auto truncated=frame;truncated.query.control[1]=truncated.query.control[0]+1;
    check(!nt::validateQueries(prepared,truncated,error),"truncated native query refuses the whole frame");
    auto exhausted=frame;const auto root=std::size_t(exhausted.query.control[0]);exhausted.query.texels[root-1]={1,0,16,3};
    check(!nt::validateQueries(prepared,exhausted,error),"exhausted closest-hit work refuses the whole frame");
    auto shortRange=tt::nativeFixture();auto lighting=shortRange.program.diffractionLightingAdmission->description();lighting.traceMaximumDistance=1;
    const auto admitted=diffractionmaterial::AdmittedLightingPlan::admit(lighting,error);
    if(admitted){shortRange.program.diffractionLightingAdmission=std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(*admitted);
        shortRange.program.programIdentity=arbitgpu::nativeFixtureDiffractionProgramIdentity(shortRange.program.bindingDigest,*admitted,0,1,{},false,&shortRange.program);}
    nt::Prepared shortPrepared;
    check(nt::prepareScene(*shortRange.scene,shortRange.program,1,shortPrepared,error)
        && !nt::buildFrame(shortPrepared,tt::nativeInstances(shortRange),128,128,discarded,error),"finite trace range cannot truncate an actual reciprocal endpoint");
    check(arbitgpu::nativeFixtureDiffractionQueryWorkWithinBudget(1920,1080,32,32,16),"1080p maximum sample table has an explicit admitted small-topology query budget");
    check(!arbitgpu::nativeFixtureDiffractionQueryWorkWithinBudget(3840,2160,32,32,16),"4K maximum sample table exceeds its explicit query budget");
    check(arbitgpu::nativeFixtureDiffractionQueryWorkWithinBudget(3840,2160,8,8,16),"4K smaller exact table remains eligible within the explicit query budget");
}

}
int main()
{
    geometryAndVisibility();refusals();gratingResponse();nativeTableAdmission();
    std::printf("diffraction triangle transport: %d/%d checks passed\n",checks-failures,checks);
    return failures?EXIT_FAILURE:EXIT_SUCCESS;
}
