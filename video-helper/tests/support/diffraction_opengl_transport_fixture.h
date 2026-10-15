#pragma once

#include "diffraction_native_transport_fixture.h"
// The native geometry executable owns no separately linked reference object.
// Keep the independent test-only physics implementation in this GL fixture TU.
#include "diffraction_reference_oracle.cpp"
#include <GL/gl.h>

namespace triangletransporttests::opengl
{
inline diffractionmaterial::Description materialDescription(const Fixture* f=nullptr)
{
    auto d=diffractionmaterial::makeAluminiumBinaryGratingPreset();
    d.spectrum.firstOrder=d.spectrum.lastOrder=1;
    // Broader lobes make the independently integrated pixel reference suitable
    // for the admitted two-subdivision native quadrature without hiding order changes.
    d.roughness.rmsSlope=.15f;
    if(f) {
        const auto& p=f->program.diffractionPaths[0].material;
        d.geometry.directionUv={p.geometry.x,p.geometry.y};d.geometry.grooveSpacingNanometres=p.geometry.z;
        d.microstructure.grooveDepthNanometres=p.geometry.w;d.roughness.rmsSlope=p.roughness.y;
    }
    return d;
}
inline bool publishPhysics(Fixture& f,std::string& error)
{
    using namespace diffractionmaterial;
    LightingDescription lighting=f.program.diffractionLightingAdmission->description();
    for(unsigned a=0;a<3;++a)lighting.paths[0].incident.direction[a]=float(f.input.source.direction[a]);
    for(unsigned b=0;b<8;++b)lighting.paths[0].incident.radiance[b]=float(f.input.source.radiance[b]);
    const auto admitted=AdmittedLightingPlan::admit(lighting,error);
    const auto physical=admit(materialDescription(&f),error);
    if (!admitted || !physical) return false;
    f.program.diffractionLightingAdmission=std::make_shared<const AdmittedLightingPlan>(*admitted);
    f.program.diffractionPaths[0]=makeGpuLightingPath(physicalcheckpoint::makeGpuParameters(*physical,lighting.paths[0].incident),lighting.paths[0]);
    f.program.diffractionPaths[0].material.incident={0,0,1,0};
    f.program.diffractionPaths[0].kindBounceAndReserved[2]=f.program.diffractionPatternMask;
    if(f.program.diffractionPatternMask)
        std::memcpy(&f.program.diffractionPaths[0].kindBounceAndReserved[3],&f.program.diffractionMaskCoverage,sizeof(float));
    auto receiver=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*f.program.objectPrograms[0]);
    receiver->spectralReflectance=f.input.receiver.reflectance;
    f.program.objectPrograms[0]=std::move(receiver);
    f.program.programIdentity=arbitgpu::nativeFixtureDiffractionProgramIdentity(f.program.bindingDigest,*admitted,f.program.diffractionPatternMask,f.program.diffractionMaskCoverage,f.program.diffractionProductDigest,false,&f.program);
    return true;
}
inline Fixture scene(bool grating,bool nonplanar=false,double move=0,int blocker=0)
{
    auto f=nativeFixture(nonplanar,move,blocker);
    f.program.diffractionPaths[0].material.roughness.y=.15f;
    using namespace HarmonicMIDI::grid;
    f.scene->activeCamera={301}; f.scene->cameraCount=1;
    auto& camera=f.scene->cameras[0];camera.id=f.scene->activeCamera;
    camera.transform.translation={0,0,grating?1.2f:.8f};
    camera.transform.rotation=grating?SceneQuaternion{0,1,0,0}:SceneQuaternion{0,0,0,1};
    camera.verticalFovRadians=1.4f;camera.nearPlane=.01f;camera.farPlane=30;
    f.scene->ambientColor={0,0,0};
    for(std::size_t o=0;o<f.scene->objectCount;++o)
    {
        const auto& object=f.scene->objects[o];
        f.scene->materials[o].doubleSided=true;
        for(std::size_t v=0;v<object.vertexCount;++v)
            f.scene->vertices[object.firstVertex+v].normal={0,0,o==1?-1.0f:1.0f};
    }
    return f;
}
inline diffractionmaterial::reference::CieXyz spectrumXyz(const std::array<double,8>& spectrum)
{
    namespace reference=diffractionmaterial::reference;
    reference::CieXyz xyz;double white=0;
    for(unsigned b=0;b<8;++b)
    {
        const auto& cie=diffractionmaterial::physicalcheckpoint::kCie1931[b];
        const double weight=diffractionmaterial::physicalcheckpoint::kQuadratureWeights[b];
        xyz.x+=spectrum[b]*weight*cie[0];xyz.y+=spectrum[b]*weight*cie[1];xyz.z+=spectrum[b]*weight*cie[2];
        white+=weight*cie[1];
    }
    xyz.x/=white;xyz.y/=white;xyz.z/=white;
    return xyz;
}
inline std::array<double,3> rgb(const std::array<double,8>& spectrum)
{
    const auto color=diffractionmaterial::reference::xyzToLinearSrgbD65(spectrumXyz(spectrum));
    return {color.red,color.green,color.blue};
}
// Independent normalized-UV footprint and double bilinear field coverage.
// This reads authored controls, never the native sample table or shader result.
inline double coverage(const Fixture& f,double u,double v)
{
    double result=f.program.diffractionOccupancyRectangleCount?0:1;
    for(unsigned i=0;i<f.program.diffractionOccupancyRectangleCount;++i) {
        const auto& r=f.program.diffractionOccupancyRectangles[i];
        if(u>=r.x && v>=r.y && u<=r.z && v<=r.w)result=1;
    }
    if(f.program.diffractionOccupancyRectangleCount)result*=f.program.diffractionMaskCoverage;
    if(f.program.diffractionFoilMaximumEvaluations) {
        u=std::clamp(u,0.0,1.0);v=std::clamp(v,0.0,1.0);
        const auto& field=f.program.diffractionFoilField;
        result*=field[0].w*(1-u)*(1-v)+field[1].w*u*(1-v)+field[2].w*(1-u)*v+field[3].w*u*v;
    }
    return std::clamp(result,0.0,1.0);
}
inline std::array<double,3> pixelOracle(const Fixture& f,bool grating,unsigned x,unsigned y,std::string& error)
{
    constexpr double pi=3.14159265358979323846;
    const auto triangles=worldTriangles(*f.scene);
    const double extent=std::tan(double(f.scene->cameras[0].verticalFovRadians)*.5);
    D3 ray=normalized({(2*(x+.5)/64-1)*extent,(2*(y+.5)/64-1)*extent,-1});
    D3 origin{0,0,grating?1.2f:.8f};
    if(grating){ray[0]=-ray[0];ray[2]=-ray[2];}
    const auto hit=intersection(triangles,{origin,ray,.01,30});
    if(hit.status==0)return {7.0/255,10.0/255,18.0/255};
    if(hit.identity.object!=(grating?102u:101u))return {};
    D3 p=origin;for(unsigned a=0;a<3;++a)p[a]+=hit.distance*ray[a];
    if(!grating)
    {
        std::array<double,8> bands{};
        const auto light=intersection(triangles,{p,f.input.source.direction,f.input.rayMinimumDistance,f.input.source.traceMaximumDistance});
        if(light.status==0)for(unsigned b=0;b<8;++b)
            bands[b]=f.input.source.radiance[b]*std::max(0.0,inner(hit.normal,f.input.source.direction))*f.input.receiver.reflectance->coefficients[b]/pi;
        namespace reference=diffractionmaterial::reference;
        auto xyz=spectrumXyz(bands);
        const auto physical=diffractionmaterial::admit(materialDescription(&f),error);if(!physical)return {};
        for(const auto& g:quadrature(triangles,102)) {
            const auto delta=difference(g.p,p);const auto direction=normalized(delta);
            const double squared=inner(delta,delta),distance=std::sqrt(squared);
            const auto toPixel=normalized(difference(p,g.p));
            const double cr=std::max(0.0,inner(hit.normal,direction)),cg=std::max(0.0,inner(g.normal,toPixel));
            if(cr==0 || cg==0 || inner(g.normal,f.input.source.direction)<=1e-6)continue;
            const auto forward=intersection(triangles,{p,direction,f.input.rayMinimumDistance,distance+f.input.rayMinimumDistance});
            const auto backward=intersection(triangles,{g.p,toPixel,f.input.rayMinimumDistance,distance+f.input.rayMinimumDistance});
            const auto illumination=intersection(triangles,{g.p,f.input.source.direction,f.input.rayMinimumDistance,f.input.source.traceMaximumDistance});
            if(forward.status!=1 || backward.status!=1 || forward.identity.object!=102 || backward.identity.object!=101 || illumination.status!=0)continue;
            const double u=g.p[0]+.5,v=(g.p[1]+.4)/.8;
            const double weight=g.weight*cr*cg*coverage(f,u,v)/(pi*squared);
            reference::EvaluationInput input;input.materialUv={u,v};
            input.incidentDirection={f.input.source.direction[0],f.input.source.direction[1],-f.input.source.direction[2]};
            for(unsigned b=0;b<8;++b)input.incidentSpectrum[b]=weight*f.input.source.radiance[b]*f.input.receiver.reflectance->coefficients[b];
            reference::EvaluationFailure failure;
            const auto bsdf=reference::evaluateDirectionalBsdf(*physical,input,{toPixel[0],toPixel[1],-toPixel[2]},failure);
            if(!bsdf){error="OpenGL independent grating-to-receiver reference refused";return {};}
            xyz.x+=bsdf->radianceXyz.x;xyz.y+=bsdf->radianceXyz.y;xyz.z+=bsdf->radianceXyz.z;
        }
        const auto color=reference::xyzToLinearSrgbD65(xyz);return {color.red,color.green,color.blue};
    }
    const auto physical=diffractionmaterial::admit(materialDescription(&f),error);if(!physical)return {};
    namespace reference=diffractionmaterial::reference;
    reference::CieXyz xyz;
    const auto outgoing=normalized(difference(origin,p));
    for(const auto& receiver:quadrature(triangles,101))
    {
        const auto delta=difference(receiver.p,p);const double squared=inner(delta,delta),distance=std::sqrt(squared);
        const auto direction=normalized(delta);
        const auto forward=intersection(triangles,{p,direction,f.input.rayMinimumDistance,distance+f.input.rayMinimumDistance});
        const auto backward=intersection(triangles,{receiver.p,normalized(difference(p,receiver.p)),f.input.rayMinimumDistance,distance+f.input.rayMinimumDistance});
        const auto light=intersection(triangles,{receiver.p,f.input.source.direction,f.input.rayMinimumDistance,f.input.source.traceMaximumDistance});
        if(forward.status!=1 || backward.status!=1 || forward.identity.object!=101 || backward.identity.object!=102 || light.status!=0)continue;
        const double weight=receiver.weight*std::max(0.0,inner(receiver.normal,normalized(difference(p,receiver.p))))/(pi*squared);
        reference::EvaluationInput input;
        // UV tangent +X, bitangent +Y and grating normal -Z.
        input.incidentDirection={direction[0],direction[1],-direction[2]};
        input.materialUv={p[0]+.5,(p[1]+.4)/.8};
        for(unsigned b=0;b<8;++b)input.incidentSpectrum[b]=weight*f.input.source.radiance[b]
            *std::max(0.0,inner(receiver.normal,f.input.source.direction))*f.input.receiver.reflectance->coefficients[b];
        reference::EvaluationFailure failure;
        const auto bsdf=reference::evaluateDirectionalBsdf(*physical,input,{outgoing[0],outgoing[1],-outgoing[2]},failure);
        if(!bsdf){error="OpenGL independent directional grating reference refused";return {};}
        xyz.x+=bsdf->radianceXyz.x;xyz.y+=bsdf->radianceXyz.y;xyz.z+=bsdf->radianceXyz.z;
    }
    const auto color=reference::xyzToLinearSrgbD65(xyz);return {color.red,color.green,color.blue};
}
inline bool nativePixels(arbitgpu::NativeFixtureSceneBackend& backend,Fixture& f,
    std::vector<float>& pixels,std::string& error,renderpassoutput::Output output=renderpassoutput::Output::Color,bool publish=true)
{
    if(publish && !publishPhysics(f,error))return false;
    const auto program=std::make_shared<const arbitgpu::NativeFixtureSurfaceMaterialProgram>(f.program);
    const auto prepared=backend.prepare(f.scene,program);if(!prepared.prepared){error=prepared.error;return false;}
    arbitgpu::NativeFixtureSceneRuntimeInputs runtime;runtime.linearColor=output==renderpassoutput::Output::Color;runtime.imageOutput=output;
    const auto rendered=backend.render(f.scene,prepared.resources,64,64,runtime);
    if(!rendered.rendered || !rendered.frame){error=rendered.error;return false;}
    GLint texture=0;glGetIntegerv(GL_TEXTURE_BINDING_2D,&texture);
    glBindTexture(GL_TEXTURE_2D,static_cast<GLuint>(rendered.frame->colorImageHandle()));
    pixels.resize(64*64*4);glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_FLOAT,pixels.data());
    const auto gpuError=glGetError();glBindTexture(GL_TEXTURE_2D,static_cast<GLuint>(texture));
    if(gpuError!=GL_NO_ERROR){error="OpenGL native spectral Color readback failed";return false;}
    return true;
}
inline bool verify(arbitgpu::NativeFixtureSceneBackend& backend,std::string& error)
{
    const auto refuse=[&](const char* message){error=message;return false;};
    std::array<std::vector<float>,2> baseline;
    for(unsigned view=0;view<2;++view)
    {
        auto f=scene(view!=0);
        if(!nativePixels(backend,f,baseline[view],error))return false;
        for(const auto& point:std::array<std::array<unsigned,2>,3>{{{31,31},{27,30},{35,33}}})
        {
            const auto expected=pixelOracle(f,view!=0,point[0],point[1],error);if(!error.empty())return false;
            for(unsigned c=0;c<3;++c)
            {
                const auto actual=baseline[view][(point[1]*64+point[0])*4+c];
                const auto tolerance=view?.004+.20*expected[c]:.0003+.003*expected[c];
                if(!std::isfinite(actual) || std::abs(actual-expected[c])>tolerance)
                    return refuse("OpenGL authored spectral pixel disagrees with independent triangle/BSDF oracle");
            }
        }
        // Preview and export submit the same intended scene-linear native request.
        std::vector<float> exported;if(!nativePixels(backend,f,exported,error))return false;
        if(exported!=baseline[view])return refuse("OpenGL spectral native preview/export pixels diverged");
        for(unsigned edit=0;edit<5;++edit)
        {
            auto changed=scene(view!=0,false,edit==2?.7:0,edit>=3?int(edit-2):0);
            if(edit==0)changed.input.source.radiance[4]*=2;
            if(edit==1)changed.input.receiver.reflectance->coefficients[2]*=.25f;
            std::vector<float> pixels;if(!nativePixels(backend,changed,pixels,error))return false;
            const auto expected=pixelOracle(changed,view!=0,31,31,error);if(!error.empty())return false;
            for(unsigned c=0;c<3;++c)
            {
                const auto actual=pixels[(31*64+31)*4+c];
                if(std::abs(actual-expected[c])>(view?.004+.20*expected[c]:.0003+.003*expected[c]))
                    return refuse("OpenGL source/rho/move/blocker pixel failed independent prediction");
            }
            if(pixels[0]!=baseline[view][0] || pixels[1]!=baseline[view][1] || pixels[2]!=baseline[view][2])
                return refuse("OpenGL spectral edit changed an unrelated background pixel");
        }
    }
    // Isolate every raw authored wavelength so a normalized or RGB-derived
    // surrogate cannot pass by approximating the broadband total.
    for(unsigned band=0;band<8;++band) for(unsigned view=0;view<2;++view)
    {
        auto isolated=scene(view!=0);
        isolated.input.source.radiance.fill(0);isolated.input.source.radiance[band]=1;
        std::vector<float> pixels;if(!nativePixels(backend,isolated,pixels,error))return false;
        const auto expected=pixelOracle(isolated,view!=0,31,31,error);if(!error.empty())return false;
        for(unsigned c=0;c<3;++c)if(std::abs(pixels[(31*64+31)*4+c]-expected[c])
            > (view?.000005+.20*expected[c]:.000005+.003*expected[c]))
            return refuse("OpenGL isolated authored wavelength failed native spectral prediction");
        isolated.input.receiver.reflectance->coefficients[band]*=.5f;
        std::vector<float> half;if(!nativePixels(backend,isolated,half,error))return false;
        for(unsigned c=0;c<3;++c)if(std::abs(half[(31*64+31)*4+c]-.5*pixels[(31*64+31)*4+c])>.00001)
            return refuse("OpenGL raw receiver reflectance did not scale its isolated wavelength");
    }
    // Legacy local-direct and analytic-plane paths retain their intended native
    // scene-linear pixel. Zero environment bands isolate the v2 direct branch.
    for(unsigned version=1;version<=2;++version)
    {
        auto legacy=scene(true);
        diffractionmaterial::LightingDescription lighting;lighting.version=version;
        lighting.pathCount=version==1?1:3;
        lighting.paths[0].kind=diffractionmaterial::LightingPathKind::Direct;
        lighting.paths[0].incident.direction={0,0,1};lighting.paths[0].incident.radiance.fill(1);
        if(version==2) {
            lighting.bounceMaximumDistance=20;
            lighting.paths[1].kind=diffractionmaterial::LightingPathKind::Environment;
            lighting.paths[2].kind=diffractionmaterial::LightingPathKind::Indirect;
            lighting.paths[2].bounceDepth=diffractionmaterial::kMaximumIndirectBounces;
            lighting.paths[1].incident.direction=lighting.paths[2].incident.direction={0,0,1};
            lighting.paths[1].incident.radiance.fill(0);lighting.paths[2].incident.radiance.fill(0);
            legacy.scene->lightCount=1;legacy.scene->lights[0].id={401};
            legacy.scene->lights[0].kind=HarmonicMIDI::grid::SceneLightKind::Directional;
            legacy.scene->lights[0].transform.rotation={0,1,0,0};legacy.scene->lights[0].intensity=1;
        }
        const auto admitted=diffractionmaterial::AdmittedLightingPlan::admit(lighting,error);
        const auto physical=diffractionmaterial::admit(materialDescription(),error);if(!admitted || !physical)return false;
        legacy.program.diffractionLightingAdmission=std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(*admitted);
        legacy.program.diffractionPaths={};legacy.program.diffractionPathCount=lighting.pathCount;
        legacy.program.diffractionMaximumBounceDepth=version==1?0:diffractionmaterial::kMaximumIndirectBounces;
        for(unsigned i=0;i<lighting.pathCount;++i)
            legacy.program.diffractionPaths[i]=diffractionmaterial::makeGpuLightingPath(
                diffractionmaterial::physicalcheckpoint::makeGpuParameters(*physical,lighting.paths[i].incident),lighting.paths[i]);
        legacy.program.programIdentity=arbitgpu::nativeFixtureDiffractionProgramIdentity(legacy.program.bindingDigest,*admitted,0,1,{},false,&legacy.program);
        std::vector<float> pixels;if(!nativePixels(backend,legacy,pixels,error,renderpassoutput::Output::Color,false))return false;
        const double xy=(2*(31+.5)/64-1)*std::tan(double(legacy.scene->cameras[0].verticalFovRadians)*.5);
        const auto outgoing=normalized({xy,-xy,1});
        diffractionmaterial::reference::EvaluationInput input;
        diffractionmaterial::reference::EvaluationFailure failure;
        const auto expected=diffractionmaterial::reference::evaluateDirectionalBsdf(*physical,input,
            {outgoing[0],outgoing[1],outgoing[2]},failure);if(!expected)return refuse("legacy grating pixel oracle refused");
        const auto& color=expected->outputLinearSrgb;const std::array<double,3> rgbExpected{color.red,color.green,color.blue};
        for(unsigned c=0;c<3;++c)if(std::abs(pixels[(31*64+31)*4+c]-rgbExpected[c])>.0005+.005*rgbExpected[c])
            return refuse("OpenGL legacy v1/v2 native pixel changed");
    }
    std::vector<float> fullCoverage,quarterCoverage;
    for(unsigned row=0;row<5;++row) {
        auto masked=scene(false);masked.input.source.direction=normalized({.6,0,-1});
        masked.program.diffractionPatternMask=1;masked.program.diffractionProductDigest="native-triangle-coverage";
        masked.program.diffractionOccupancyRectangleCount=1;
        masked.program.diffractionOccupancyRectangles[0]={0,0,1,1};
        masked.program.diffractionMaskCoverage=row==0?0:row==1?.25f:1;
        if(row==3)masked.program.diffractionOccupancyRectangles[0]={0,0,.5f,1};
        if(row==4) {
            masked.program.diffractionFoilMaximumEvaluations=4096;
            const auto& geometry=masked.program.diffractionPaths[0].material.geometry;
            const float weights[]={0,.4f,.4f,1};
            for(unsigned i=0;i<4;++i)masked.program.diffractionFoilField[i]={geometry.x,geometry.y,geometry.z,weights[i]};
        }
        std::vector<float> pixels;if(!nativePixels(backend,masked,pixels,error))return false;
        const auto expected=pixelOracle(masked,false,31,31,error);if(!error.empty())return false;
        for(unsigned c=0;c<3;++c)if(!std::isfinite(pixels[(31*64+31)*4+c])
            || std::abs(pixels[(31*64+31)*4+c]-expected[c])>.00001+.25*expected[c])
            return refuse("OpenGL grating-to-receiver coverage failed independent triangle/BSDF prediction");
        if(row==1)quarterCoverage=pixels;
        if(row==2) {
            fullCoverage=pixels;
            for(unsigned c=0;c<3;++c)if(std::abs(quarterCoverage[(31*64+31)*4+c]-.25*pixels[(31*64+31)*4+c])>.00001)
                return refuse("OpenGL partial authored foil coverage did not scale receiver flux");
        }
        if(row==0 && (pixels[(31*64+31)*4]!=0 || pixels[(31*64+31)*4+1]!=0 || pixels[(31*64+31)*4+2]!=0))
            return refuse("OpenGL zero authored foil coverage illuminated the receiver");
        if(row==2 && pixels[(31*64+31)*4]+pixels[(31*64+31)*4+1]+pixels[(31*64+31)*4+2]<=.001f)
            return refuse("OpenGL full authored foil coverage fixture did not illuminate its receiver");
        if(row>2 && pixels==fullCoverage)
            return refuse("OpenGL sampled UV occupancy/field coverage did not change the receiver");
    }
    auto orderEdit=scene(true);
    auto& grooves=orderEdit.program.diffractionPaths[0].material.geometry;
    grooves.x=.8f;grooves.y=.6f;grooves.z=700;
    std::vector<float> orderPixels;if(!nativePixels(backend,orderEdit,orderPixels,error))return false;
    const auto orderExpected=pixelOracle(orderEdit,true,31,31,error);if(!error.empty())return false;
    for(unsigned c=0;c<3;++c)if(std::abs(orderPixels[(31*64+31)*4+c]-orderExpected[c])>.004+.20*orderExpected[c])
        return refuse("OpenGL authored groove/order change failed physical prediction");
    const auto physics=diffractionmaterial::admit(materialDescription(&orderEdit),error);if(!physics)return false;
    diffractionmaterial::reference::EvaluationInput energyInput;energyInput.incidentSpectrum=orderEdit.input.source.radiance;
    diffractionmaterial::reference::EvaluationFailure energyFailure;
    const auto energy=diffractionmaterial::reference::evaluate(*physics,energyInput,energyFailure);if(!energy)return refuse("independent energy/order reference refused");
    for(unsigned b=0;b<8;++b) {
        const auto& sample=energy->spectralSamples[b];
        if(std::abs(sample.incidentEnergy-sample.resolvedReflectedEnergy-sample.unresolvedReflectedEnergy-sample.absorbedEnergy)>1e-9)
            return refuse("independent grating energy/order ledger did not conserve its band");
    }
    auto bent=scene(true,true);std::vector<float> bentPixels;
    if(!nativePixels(backend,bent,bentPixels,error))return false;
    const auto bentExpected=pixelOracle(bent,true,31,31,error);if(!error.empty())return false;
    for(unsigned c=0;c<3;++c)if(std::abs(bentPixels[(31*64+31)*4+c]-bentExpected[c])>.004+.20*bentExpected[c])
        return refuse("OpenGL nonplanar receiver failed world-triangle prediction");
    auto invalid=scene(true);if(!publishPhysics(invalid,error))return false;
    auto stale=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*invalid.program.objectPrograms[0]);
    stale->parameters.identifiers[0]=999;invalid.program.objectPrograms[0]=stale;
    if(backend.prepare(invalid.scene,std::make_shared<const arbitgpu::NativeFixtureSurfaceMaterialProgram>(invalid.program)).prepared)
        return refuse("OpenGL spectral receiver accepted stale material identity");
    auto valid=scene(true);if(!publishPhysics(valid,error))return false;
    const auto prepared=backend.prepare(valid.scene,std::make_shared<const arbitgpu::NativeFixtureSurfaceMaterialProgram>(valid.program));
    if(!prepared.prepared)return refuse("OpenGL authored spectral budget fixture did not prepare");
    if(backend.render(valid.scene,prepared.resources,4096,4096).rendered)
        return refuse("OpenGL authored spectral frame exceeded bounded lobe/query work");
    // A smaller exact triangle table admits 4K without allocating a 4K GPU
    // target here. The conservative default-32 estimate must not gate v3.
    auto small=scene(true);if(!publishPhysics(small,error))return false;
    for(std::size_t o=0;o<small.scene->objectCount;++o)small.scene->objects[o].indexCount=3;
    arbitgpu::diffractiontransportnative::Prepared smallPrepared;
    arbitgpu::diffractiontransportnative::Frame smallFrame;
    if(!arbitgpu::diffractiontransportnative::prepareScene(*small.scene,small.program,small.program.sceneRevision,smallPrepared,error)
        || !arbitgpu::diffractiontransportnative::buildFrame(smallPrepared,nativeInstances(small),4096,4096,smallFrame,error))return false;
    if(smallFrame.sampleControl[0]!=4 || smallFrame.sampleControl[1]!=4
        || arbitgpu::nativeFixtureDiffractionWorkWithinBudget(small.program,4096,4096))
        return refuse("OpenGL actual-count 4K budget fixture lost its default-count boundary");
    std::vector<float> aov;if(!nativePixels(backend,valid,aov,error,renderpassoutput::Output::Normal))return false;
    error.clear();return true;
}
} // namespace triangletransporttests::opengl
