#pragma once

#include "diffraction_triangle_transport_oracle.h"
#include "../../../shared/DiffractionMaterialPresets.h"
#include "../../src/gpu_backend/fixture_diffraction_transport.h"

#include <optional>

namespace triangletransporttests
{
// Reuse actual authored world triangles and the independent double oracle.
// Native backends can change target and consume exactly this table contract.
inline Fixture nativeFixture(bool nonplanar=false,double move=0,int blocker=0,std::optional<double> sourceRadiance={})
{
    auto f=fixture(nonplanar,move,blocker);
    if(sourceRadiance)f.input.source.radiance.fill(*sourceRadiance);
    for(std::size_t o=0;o<f.scene->objectCount;++o)
    {
        const auto first=f.scene->objects[o].firstVertex;
        f.scene->vertices[first].uv={0,0};f.scene->vertices[first+1].uv={1,0};
        f.scene->vertices[first+2].uv={1,1};f.scene->vertices[first+3].uv={0,1};
    }
    // validateVisual3DScene refuses a scene with no active camera or with a zero
    // vertex normal. Production scenes carry both, so the fixture does too: one
    // default camera, and each vertex normal is the sum of its triangles' winding normals.
    f.scene->cameras[0].id=HarmonicMIDI::grid::SceneCameraId{1};
    f.scene->cameraCount=1;f.scene->activeCamera=f.scene->cameras[0].id;
    for(std::size_t o=0;o<f.scene->objectCount;++o)
    {
        const auto& object=f.scene->objects[o];
        std::vector<D3> sums(object.vertexCount,D3{0,0,0});
        for(std::size_t i=0;i<object.indexCount;i+=3)
        {
            std::array<unsigned,3> corner{f.scene->indices[object.firstIndex+i],f.scene->indices[object.firstIndex+i+1],f.scene->indices[object.firstIndex+i+2]};
            std::array<D3,3> p;
            for(unsigned k=0;k<3;++k)
            {
                const auto& position=f.scene->vertices[object.firstVertex+corner[k]].position;
                p[k]={position.x,position.y,position.z};
            }
            const auto n=exterior(difference(p[1],p[0]),difference(p[2],p[0]));
            for(unsigned k=0;k<3;++k)for(unsigned a=0;a<3;++a)sums[corner[k]][a]+=n[a];
        }
        for(std::size_t v=0;v<object.vertexCount;++v)
        {
            const auto n=normalized(sums[v]);
            f.scene->vertices[object.firstVertex+v].normal={float(n[0]),float(n[1]),float(n[2])};
        }
    }
    using P=arbitgpu::NativeFixtureSurfaceMaterialProgram;
    auto receiver=std::make_shared<P>(f.program);
    receiver->backend=arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    receiver->bindingDigest="native-receiver-binding";receiver->programIdentity="native-receiver-program";
    receiver->authoredScene=f.scene->id;receiver->authoredMaterial={201};
    receiver->sceneRevision=receiver->structuralRevision=receiver->evaluationRevision=receiver->materialRevision=1;
    P grating;grating.backend=receiver->backend;grating.kind=arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;
    grating.object={102};grating.authoredScene=f.scene->id;grating.authoredMaterial={202};
    grating.sceneRevision=grating.structuralRevision=grating.evaluationRevision=grating.materialRevision=1;
    grating.bindingDigest="native-grating-binding";grating.diffractionPathCount=1;
    grating.diffractionMaximumBounceDepth=diffractionmaterial::kMaximumIndirectBounces;
    diffractionmaterial::LightingDescription lighting;lighting.version=3;lighting.pathCount=1;
    lighting.sceneReceiverTransport=diffractionmaterial::SceneReceiverTransport::AuthoredSpectralLambertian;
    lighting.traceMaximumDistance=float(f.input.source.traceMaximumDistance);
    lighting.paths[0].kind=diffractionmaterial::LightingPathKind::Direct;
    for(unsigned a=0;a<3;++a)lighting.paths[0].incident.direction[a]=float(f.input.source.direction[a]);
    for(unsigned b=0;b<8;++b)lighting.paths[0].incident.radiance[b]=float(f.input.source.radiance[b]);
    std::string error;
    const auto admitted=diffractionmaterial::AdmittedLightingPlan::admit(lighting,error);
    if(admitted)grating.diffractionLightingAdmission=std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(*admitted);
    auto description=diffractionmaterial::makeAluminiumBinaryGratingPreset();
    description.spectrum.firstOrder=description.spectrum.lastOrder=1;description.roughness.rmsSlope=.01f;
    const auto material=diffractionmaterial::admit(description,error);
    if(material && admitted)
    {
        grating.diffractionPaths[0]=diffractionmaterial::makeGpuLightingPath(
            diffractionmaterial::physicalcheckpoint::makeGpuParameters(*material,lighting.paths[0].incident),lighting.paths[0]);
        grating.diffractionPaths[0].material.incident={0,0,1,0};
        grating.programIdentity=arbitgpu::nativeFixtureDiffractionProgramIdentity(grating.bindingDigest,*admitted,0,1,{},false,&grating);
    }
    grating.objectPrograms.push_back(receiver);f.program=std::move(grating);return f;
}
inline std::vector<arbitgpu::opaquevisibility::Instance> nativeInstances(const Fixture& f)
{
    const arbitgpu::opaquevisibility::Matrix identity{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    std::vector<arbitgpu::opaquevisibility::Instance> result;
    for(std::size_t o=0;o<f.scene->objectCount;++o)result.push_back({identity,o});
    return result;
}
} // namespace triangletransporttests
