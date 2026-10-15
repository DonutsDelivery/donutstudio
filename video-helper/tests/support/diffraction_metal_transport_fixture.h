#pragma once

#include "diffraction_native_transport_fixture.h"
#include <cstring>
#include <limits>

namespace metaltriangletransporttests
{
namespace tt=triangletransporttests;
namespace nt=arbitgpu::diffractiontransportnative;
using Spectrum=std::array<double,8>;
using Rgb=std::array<double,3>;
constexpr unsigned extent=64;

inline diffractionmaterial::Description description()
{
    auto d=diffractionmaterial::makeAluminiumBinaryGratingPreset();
    d.spectrum.firstOrder=d.spectrum.lastOrder=1;
    // Broad enough to compare the bounded native centroid table with an
    // independent six-point Duffy integration, rather than a delta-order probe.
    d.roughness.rmsSlope=.15f;
    return d;
}
inline bool publishLighting(tt::Fixture& f,std::string& error,
                            const diffractionmaterial::Description& optics=description())
{
    auto lighting=f.program.diffractionLightingAdmission->description();
    for(unsigned a=0;a<3;++a) lighting.paths[0].incident.direction[a]=float(f.input.source.direction[a]);
    for(unsigned b=0;b<8;++b) lighting.paths[0].incident.radiance[b]=float(f.input.source.radiance[b]);
    const auto admitted=diffractionmaterial::AdmittedLightingPlan::admit(lighting,error);
    const auto material=diffractionmaterial::admit(optics,error);
    if(!admitted || !material)return false;
    f.program.diffractionLightingAdmission=std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(*admitted);
    f.program.diffractionPaths[0]=diffractionmaterial::makeGpuLightingPath(
        diffractionmaterial::physicalcheckpoint::makeGpuParameters(*material,lighting.paths[0].incident),lighting.paths[0]);
    f.program.diffractionPaths[0].material.incident={0,0,1,0};
    f.program.programIdentity=arbitgpu::nativeFixtureDiffractionProgramIdentity(
        f.program.bindingDigest,*admitted,0,1,{},false,&f.program);
    return true;
}
inline tt::Fixture fixture(bool grating,double move=0,int blocker=0)
{
    auto f=tt::nativeFixture(false,move,blocker);
    f.program.backend=arbitgpu::NativeFixtureMaterialBackend::Metal;
    auto receiver=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*f.program.objectPrograms[0]);
    receiver->backend=f.program.backend;f.program.objectPrograms[0]=receiver;
    f.scene->cameraCount=1;f.scene->activeCamera={301};
    auto& camera=f.scene->cameras[0];camera.id=f.scene->activeCamera;
    camera.transform.translation={0,0,1};
    if(grating)camera.transform.rotation={0,1,0,0};
    f.scene->ambientColor={0,0,0};
    for(std::size_t o=0;o<f.scene->objectCount;++o)
    {
        const auto& object=f.scene->objects[o];
        f.scene->materials[o].doubleSided=true;
        for(std::size_t v=0;v<object.vertexCount;++v)
            f.scene->vertices[object.firstVertex+v].normal={0,0,object.id.value==102?-1.0f:1.0f};
    }
    std::string error;publishLighting(f,error);return f;
}
inline Rgb xyzRgb(const Spectrum& bands,const diffractionmaterial::PhysicalDiffractionGpuParameters& p)
{
    // Only the released wavelength/CIE schedule is shared. Transport and order
    // equations below use double precision and independent geometry queries.
    for(const auto band:bands)if(!std::isfinite(band)) {
        const auto diagnostic=std::numeric_limits<double>::quiet_NaN();
        return {diagnostic,diagnostic,diagnostic};
    }
    std::array<double,3> xyz{};double white=0;
    for(unsigned b=0;b<8;++b) {
        const auto& c=p.spectral[b];const auto& z=p.spectralZ[b];
        xyz[0]+=bands[b]*z.y*c.z;xyz[1]+=bands[b]*z.y*c.w;xyz[2]+=bands[b]*z.y*z.x;white+=z.y*c.w;
    }
    for(auto& c:xyz)c/=white;
    return {std::max(0.0,3.2406*xyz[0]-1.5372*xyz[1]-.4986*xyz[2]),
            std::max(0.0,-.9689*xyz[0]+1.8758*xyz[1]+.0415*xyz[2]),
            std::max(0.0,.0557*xyz[0]-.2040*xyz[1]+1.0570*xyz[2])};
}
inline double binaryEfficiency(int order,unsigned band,tt::D3 incoming,
                               const diffractionmaterial::PhysicalDiffractionGpuParameters& p)
{
    constexpr double pi=3.1415926535897932384626433832795;
    const double lambda=p.spectral[band].x;
    const double x=-incoming[0]+order*lambda/p.geometry.z,y=-incoming[1];
    if(x*x+y*y>=1 || incoming[2]<=1e-6)return 0;
    const double cosine=std::sqrt(1-x*x-y*y);
    const double phase=2*pi*p.geometry.w*(incoming[2]+cosine)/lambda;
    const double duty=p.microstructure.x;
    const double coefficient=order==0
        ? 1-2*duty*(1-duty)*(1-std::cos(phase))
        : 2*(1-std::cos(phase))*std::pow(std::sin(pi*order*duty)/(pi*order),2);
    const double height=2*pi*p.roughness.x*(incoming[2]+cosine)/lambda;
    return coefficient*std::exp(-height*height)*cosine/incoming[2];
}
inline double binaryEfficiencySum(unsigned band,tt::D3 incoming,
                                  const diffractionmaterial::PhysicalDiffractionGpuParameters& p)
{
    double result=0;for(int order:{-1,0,1})result+=binaryEfficiency(order,band,incoming,p);
    return result;
}
inline double binaryBsdf(unsigned band,tt::D3 incoming,tt::D3 outgoing,
                         const diffractionmaterial::PhysicalDiffractionGpuParameters& p)
{
    if(incoming[2]<=1e-6 || outgoing[2]<=0)return 0;
    constexpr double pi=3.1415926535897932384626433832795;
    const double lambda=p.spectral[band].x,sigma=std::max(2.0*p.roughness.y,1e-4);
    const double n=p.microstructure.y,k=p.microstructure.z;
    const double fresnel=((n-1)*(n-1)+k*k)/((n+1)*(n+1)+k*k);
    const double efficiency=binaryEfficiencySum(band,incoming,p);
    if(!std::isfinite(efficiency) || efficiency>1.0002)return std::numeric_limits<double>::quiet_NaN();
    double result=0;
    for(int order:{-1,0,1}) {
        const double x=-incoming[0]+order*lambda/p.geometry.z,y=-incoming[1];
        if(x*x+y*y>=1)continue;
        const double dx=outgoing[0]-x,dy=outgoing[1]-y;
        result+=binaryEfficiency(order,band,incoming,p)
            *std::exp(-(dx*dx+dy*dy)/(2*sigma*sigma))/(2*pi*sigma*sigma);
    }
    return fresnel*result;
}
inline tt::D3 point(bool grating,unsigned x,unsigned y)
{
    const double tangent=std::tan(1.0471975512*.5);
    const double px=(2*(double(x)+.5)/extent-1)*tangent;
    const double py=(1-2*(double(y)+.5)/extent)*tangent;
    return {grating?-px:px,py,grating?2.0:0.0};
}
inline Rgb predicted(const tt::Fixture& f,bool grating,unsigned x,unsigned y)
{
    constexpr double pi=3.1415926535897932384626433832795;
    const auto triangles=tt::worldTriangles(*f.scene);const auto position=point(grating,x,y);
    Spectrum bands{};
    const auto& p=f.program.diffractionPaths[0].material;
    if(!grating) {
        const auto hit=tt::intersection(triangles,{position,f.input.source.direction,1e-5,20});
        for(unsigned b=0;b<8;++b)bands[b]=hit.status ? 0
            : f.input.source.radiance[b]*f.input.source.direction[2]*f.input.receiver.reflectance->coefficients[b]/pi;
    } else {
        auto view=tt::normalized(tt::difference({0,0,1},position));view[2]=-view[2];
        for(const auto& receiver:tt::quadrature(triangles,101)) {
            const auto delta=tt::difference(receiver.p,position),incoming=tt::normalized(delta);
            const double distance=std::sqrt(tt::inner(delta,delta));
            const auto forward=tt::intersection(triangles,{position,incoming,1e-5,distance+1e-5});
            const auto back=tt::intersection(triangles,{receiver.p,tt::normalized(tt::difference(position,receiver.p)),1e-5,distance+1e-5});
            const auto light=tt::intersection(triangles,{receiver.p,f.input.source.direction,1e-5,20});
            if(!forward.status || forward.identity.object!=101 || !back.status || back.identity.object!=102 || light.status)continue;
            auto local=incoming;local[2]=-local[2];
            const double weight=receiver.weight*local[2]*local[2]/(pi*distance*distance)*f.input.source.direction[2];
            for(unsigned b=0;b<8;++b)bands[b]+=f.input.source.radiance[b]*f.input.receiver.reflectance->coefficients[b]
                *weight*binaryBsdf(b,local,view,p);
        }
    }
    return xyzRgb(bands,p);
}
inline std::vector<float> linearPixels(const arbitgpu::NativeRawPassPixels& pixels)
{
    if(pixels.format!=arbitgpu::NativeTexturePixelFormat::Rgba16Float || pixels.bytes.size()!=pixels.width*pixels.height*8u)return {};
    std::vector<float> result(pixels.bytes.size()/2);
    for(std::size_t i=0;i<result.size();++i) {
        std::uint16_t half=0;std::memcpy(&half,pixels.bytes.data()+2*i,2);
        const auto exponent=(half>>10)&31u,fraction=half&1023u;
        const float value=exponent==31 ? std::numeric_limits<float>::quiet_NaN()
            : exponent==0 ? std::ldexp(float(fraction),-24) : std::ldexp(float(1024u+fraction),int(exponent)-25);
        result[i]=(half&32768u)?-value:value;
    }
    return result;
}
inline bool render(arbitgpu::NativeFixtureSceneBackend& backend,tt::Fixture& f,
                   std::vector<float>& pixels,std::string& error,renderpassoutput::Output output=renderpassoutput::Output::Color,
                   const arbitgpu::NativeFixtureSceneRuntimeInputs* request=nullptr)
{
    const auto program=std::make_shared<const arbitgpu::NativeFixtureSurfaceMaterialProgram>(f.program);
    const auto prepared=backend.prepare(f.scene,program);
    if(!prepared.prepared) {error=prepared.error;return false;}
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs=request?*request:arbitgpu::NativeFixtureSceneRuntimeInputs{};inputs.linearColor=true;inputs.imageOutput=output;inputs.timeSeconds=.25f;
    const auto submission=backend.render(f.scene,prepared.resources,extent,extent,inputs);
    if(!submission.rendered || !submission.frame) {error=submission.error;return false;}
    arbitgpu::NativeRawPassPixels raw;
    if(!submission.frame->readRawPass(renderpassoutput::Output::Color,raw,error))return false;
    pixels=linearPixels(raw);return nt::validateWholeOutput(pixels,extent,extent,error);
}
inline bool compare(const tt::Fixture& f,bool grating,const std::vector<float>& pixels,std::string& error)
{
    for(unsigned y:{24u,31u,38u})for(unsigned x:{24u,31u,38u}) {
        const auto expected=predicted(f,grating,x,y);const auto at=(y*extent+x)*4;
        for(unsigned c=0;c<3;++c) {
            const double tolerance=.002+std::abs(expected[c])*(grating?.16:.008);
            if(!std::isfinite(expected[c]) || std::abs(pixels[at+c]-expected[c])>tolerance) {
                error="Metal authored spectral pixel disagrees with independent triangle/Fourier oracle at "
                    +std::to_string(x)+","+std::to_string(y)+" channel "+std::to_string(c)
                    +" actual "+std::to_string(pixels[at+c])+" expected "+std::to_string(expected[c]);return false;
            }
        }
    }
    return true;
}
inline bool verify(arbitgpu::NativeFixtureSceneBackend& backend,std::string& error)
{
    for(bool grating:{false,true}) {
        auto base=fixture(grating);std::vector<float> baseline;
        if(!render(backend,base,baseline,error) || !compare(base,grating,baseline,error))return false;
        // Independent native owners use identical same-time candidate inputs.
        // The helper's preview/export publication gate is a separate check.
        std::vector<float> exportPixels;
        if(!render(backend,base,exportPixels,error) || exportPixels!=baseline) {error="Metal authored spectral independent same-time owner pixels differ";return false;}
        auto changedSource=fixture(grating);changedSource.input.source.radiance[4]=0;
        if(!publishLighting(changedSource,error))return false;
        std::vector<float> sourcePixels;
        if(!render(backend,changedSource,sourcePixels,error) || !compare(changedSource,grating,sourcePixels,error) || sourcePixels==baseline)return false;
        auto changedRho=fixture(grating);
        auto receiver=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*changedRho.program.objectPrograms[0]);
        receiver->spectralReflectance->coefficients[4]=0;changedRho.program.objectPrograms[0]=receiver;
        changedRho.input.receiver.reflectance=receiver->spectralReflectance;
        std::vector<float> rhoPixels;
        if(!render(backend,changedRho,rhoPixels,error) || !compare(changedRho,grating,rhoPixels,error) || rhoPixels!=sourcePixels)return false;
        auto moved=fixture(grating,.3);std::vector<float> movedPixels;
        if(!render(backend,moved,movedPixels,error) || !compare(moved,grating,movedPixels,error))return false;
        if(grating && movedPixels==baseline) {error="Moving actual receiver did not change Metal grating transport";return false;}
        // Background belongs to neither receiver nor grating and must remain
        // byte-identical through source/rho edits and geometry movement.
        for(unsigned c=0;c<4;++c)
            if(sourcePixels[c]!=baseline[c] || rhoPixels[c]!=baseline[c] || movedPixels[c]!=baseline[c]) {error="Unrelated Metal background changed with authored spectral edits";return false;}
        for(int blocker:{1,2}) {
            auto blocked=fixture(grating,0,blocker);std::vector<float> blockedPixels;
            if(!render(backend,blocked,blockedPixels,error) || !compare(blocked,grating,blockedPixels,error))return false;
            if(grating && blockedPixels==baseline) {error="Actual Metal blocker did not change spectral transport";return false;}
        }
    }
    // Triangle identities come from the actual index order, not the first
    // sample belonging to an object. Both orders must retain the same pixels.
    for(bool grating:{false,true}) {
        auto original=fixture(grating),reordered=fixture(grating);
        for(std::size_t o=0;o<reordered.scene->objectCount;++o) {
            const auto first=reordered.scene->objects[o].firstIndex;
            for(unsigned i=0;i<3;++i)std::swap(reordered.scene->indices[first+i],reordered.scene->indices[first+3+i]);
        }
        std::vector<float> before,after;
        if(!render(backend,original,before,error) || !render(backend,reordered,after,error)
            || !compare(reordered,grating,after,error))return false;
        for(std::size_t i=0;i<after.size();++i)
            if(std::abs(before[i]-after[i])>.002f) {error="Metal raster triangle reorder changed spectral pixels";return false;}
        // A constant physical field with exactly the full-frame legacy budget
        // must evaluate every v3 pixel, including rows after the first sample.
        auto field=fixture(grating);
        field.program.diffractionFoilMaximumEvaluations=extent*extent;
        for(auto& corner:field.program.diffractionFoilField)
            corner={field.program.diffractionPaths[0].material.geometry.x,
                    field.program.diffractionPaths[0].material.geometry.y,
                    field.program.diffractionPaths[0].material.geometry.z,1};
        if(!publishLighting(field,error) || !render(backend,field,after,error)
            || !compare(field,grating,after,error))return false;
        for(std::size_t i=0;i<after.size();++i)
            if(std::abs(before[i]-after[i])>.002f) {error="Metal v3 full-frame field schedule changed constant-field pixels";return false;}
    }
    // Two separated native note instances exercise object-major/note-major
    // identity. Both occupy pixels; all reciprocal endpoints must be owned.
    auto score=std::make_shared<arbitmod::Score>();
    score->notationVersion=1;score->scoreRevision=1;score->rootFreq=440;
    for(unsigned i=0;i<2;++i) {
        arbitmod::Note note;note.id=i+1;note.trackId=1;note.startBeat=float(i);
        note.lengthBeats=4;note.velocity=100;note.freqHz=440;note.durationSeconds=2;
        score->notes.push_back(note);
    }
    canonicalblockc::FrameKey key;
    key.projectGeneration=key.sourceGeneration=key.helperGeneration=1;
    key.backendGeneration=key.deviceGeneration=key.scoreGeneration=1;
    key.beatMapGeneration=key.fpsGeneration=key.loopGeneration=key.seekGeneration=1;key.fps=60;
    canonicalblockc::FrameProducer producer;
    arbitgpu::NativeFixtureSceneRuntimeInputs notes;
    notes.canonicalBlockCFrame=producer.evaluate(key,score,0);
    visualnoteinstancing::Mapping mapping;mapping.x=visualnoteinstancing::MappingAxis::Onset;
    mapping.xScale=1;mapping.meshScale=.8f;notes.noteInstanceMapping=mapping;
    if(arbitgpu::prepareNativeNoteInstances(notes).count!=2) {error="Metal spectral identity fixture omitted its second native note instance";return false;}
    for(bool grating:{false,true}) {
        auto instanced=fixture(grating);instanced.scene->cameras[0].transform.translation.x=.5f;
        std::vector<float> pixels,exportPixels;
        if(!render(backend,instanced,pixels,error,renderpassoutput::Output::Color,&notes)
            || !render(backend,instanced,exportPixels,error,renderpassoutput::Output::Color,&notes)
            || pixels!=exportPixels) {error="Metal exact-instance spectral requests differ or refused: "+error;return false;}
        for(unsigned side:{0u,1u}) {
            bool covered=false;
            for(unsigned y=0;y<extent;++y)for(unsigned x=side*extent/2;x<(side+1)*extent/2;++x)
                for(unsigned c=0;c<3;++c)
                    covered=covered || std::abs(pixels[(y*extent+x)*4+c]-pixels[c])>.003f;
            if(!covered) {error="Metal spectral identity fixture did not rasterize both note instances";return false;}
        }
    }
    auto periodEdited=fixture(true);auto optics=description();optics.geometry.grooveSpacingNanometres=800;
    std::vector<float> periodPixels;
    if(!publishLighting(periodEdited,error,optics) || !render(backend,periodEdited,periodPixels,error)
        || !compare(periodEdited,true,periodPixels,error))return false;
    // Older descriptors retain their own lighting behavior. Adding or editing
    // authored rho must not activate the v3 shader for either released version.
    for(unsigned version:{1u,2u}) {
        auto legacy=fixture(true);auto lighting=diffractionmaterial::makeReferenceLighting();
        lighting.version=version;
        if(version==2) {lighting.bouncePlaneHeight=-.1f;lighting.bounceMaximumDistance=20;}
        const auto admitted=diffractionmaterial::AdmittedLightingPlan::admit(lighting,error);
        const auto material=diffractionmaterial::admit(description(),error);
        if(!admitted || !material)return false;
        legacy.program.diffractionLightingAdmission=std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(*admitted);
        legacy.program.diffractionPathCount=lighting.pathCount;
        legacy.program.diffractionMaximumBounceDepth=admitted->hasIndirectBounce()?diffractionmaterial::kMaximumIndirectBounces:0;
        for(unsigned i=0;i<lighting.pathCount;++i)legacy.program.diffractionPaths[i]=diffractionmaterial::makeGpuLightingPath(
            diffractionmaterial::physicalcheckpoint::makeGpuParameters(*material,lighting.paths[i].incident),lighting.paths[i]);
        legacy.program.programIdentity=arbitgpu::nativeFixtureDiffractionProgramIdentity(
            legacy.program.bindingDigest,*admitted,0,1,{},false,&legacy.program);
        std::vector<float> original,edited;
        if(!render(backend,legacy,original,error))return false;
        auto receiver=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*legacy.program.objectPrograms[0]);
        receiver->spectralReflectance->coefficients.fill(0);legacy.program.objectPrograms[0]=receiver;
        if(!render(backend,legacy,edited,error) || original!=edited) {error="Released Metal lighting version changed after an unused rho edit";return false;}
    }
    // A finite authored trace bound can still be too short for these actual
    // receiver endpoints. Every requested output must refuse the complete frame.
    auto shortRange=fixture(true);auto lighting=shortRange.program.diffractionLightingAdmission->description();
    lighting.traceMaximumDistance=1;
    const auto shortLighting=diffractionmaterial::AdmittedLightingPlan::admit(lighting,error);
    if(!shortLighting)return false;
    shortRange.program.diffractionLightingAdmission=std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(*shortLighting);
    shortRange.program.programIdentity=arbitgpu::nativeFixtureDiffractionProgramIdentity(
        shortRange.program.bindingDigest,*shortLighting,0,1,{},false,&shortRange.program);
    // Independently establish an admitted grazing scalar-wave mode whose
    // propagating-order energy exceeds unity. Every AOV must refuse, without
    // renormalizing its physical coefficients or hiding it behind zero light.
    auto overEnergy=fixture(true);overEnergy.input.source.direction=tt::normalized({20,0,-1});
    if(!publishLighting(overEnergy,error))return false;
    auto local=overEnergy.input.source.direction;local[2]=-local[2];
    bool exceeds=false;
    for(unsigned b=0;b<8;++b)
        exceeds=exceeds || binaryEfficiencySum(b,local,overEnergy.program.diffractionPaths[0].material)>1.0002;
    if(!exceeds) {error="Independent Metal spectral refusal fixture has no over-energy propagating mode";return false;}
    for(const auto* refused:std::array<const tt::Fixture*,2>{&shortRange,&overEnergy}) {
        const auto owner=std::make_shared<const arbitgpu::NativeFixtureSurfaceMaterialProgram>(refused->program);
        const auto prepared=backend.prepare(refused->scene,owner);
        if(!prepared.prepared) {error=prepared.error;return false;}
        for(const auto output:renderpassoutput::kOutputs) {
            arbitgpu::NativeFixtureSceneRuntimeInputs inputs;inputs.linearColor=true;inputs.imageOutput=output;
            const auto result=backend.render(refused->scene,prepared.resources,extent,extent,inputs);
            if(result.rendered || result.frame || result.error.empty()) {error="Metal incomplete or over-energy spectral transport published an AOV";return false;}
        }
    }
    // Check the finite positive receiver-energy oracle independently of pixels.
    auto energy=fixture(false);const auto ledger=tt::oracle(*energy.scene,energy.input);
    for(unsigned b=0;b<8;++b)if(ledger.arrivingPower[b]<0 || ledger.arrivingPower[b]>ledger.receiverOutgoingPower[b]+1e-8) {error="Independent authored triangle spectral energy bound failed";return false;}
    const auto p=energy.program.diffractionPaths[0].material;
    const double blue=p.spectral[0].x/p.geometry.z,red=p.spectral[7].x/p.geometry.z;
    if(!(red>blue && red-blue>.2)) {error="Independent physical wavelength order separation failed";return false;}
    // Every channel and the last pixel are part of the refusal boundary.
    std::vector<float> diagnostic(extent*extent*4,0);
    for(unsigned c=0;c<4;++c) {
        diagnostic[diagnostic.size()-4+c]=std::numeric_limits<float>::quiet_NaN();
        if(nt::validateWholeOutput(diagnostic,extent,extent,error))return false;
        diagnostic[diagnostic.size()-4+c]=0;
    }
    diagnostic.pop_back();if(nt::validateWholeOutput(diagnostic,extent,extent,error))return false;
    error.clear();return true;
}
} // namespace metaltriangletransporttests
