#pragma once

#include "thin_interface_water_cases.h"
#include "opengl_rough_glass_cases.h"
#include "environment_secondary_cases.h"

namespace videohelper::tests::openglroughwaterfixture
{
using namespace HarmonicMIDI::grid;
using waterfixture::D3;
using openglroughglassfixture::scalar;
using openglroughglassfixture::identity;
using openglroughglassfixture::write;

inline D3 origin(const Visual3DScene& scene)
{
    const auto& p=scene.cameras[0].transform.translation;
    return {p.x,p.y,p.z};
}
inline D3 ray(const Visual3DScene& scene,std::size_t x,std::size_t y,std::uint32_t extent)
{
    const auto& camera=scene.cameras[0];
    const double angle=2*std::atan2(camera.transform.rotation.y,camera.transform.rotation.w);
    const double tangent=std::tan(camera.verticalFovRadians*.5);
    const double horizontal=(2*(double(x)+.5)/extent-1)*tangent;
    const double vertical=(1-2*(double(y)+.5)/extent)*tangent;
    return waterfixture::unit({std::cos(angle)*horizontal-std::sin(angle),vertical,
        -std::sin(angle)*horizontal-std::cos(angle)});
}
inline Visual3DScene below(float degrees)
{
    auto scene=waterfixture::scene(0);
    const double angle=degrees*std::acos(-1.0)/180;
    const double rotation=std::acos(-1.0)-angle;
    scene.cameras[0].transform.translation={float(3*std::sin(angle)),0,float(-3*std::cos(angle))};
    scene.cameras[0].transform.rotation={0,float(std::sin(rotation*.5)),0,float(std::cos(rotation*.5))};
    return scene;
}
inline double landing(const D3& entry,const D3& direction,double z)
{
    return entry[0]+(z-entry[2])*direction[0]/direction[2];
}
inline void fullReceivers(Visual3DScene& scene)
{
    const auto expected=waterfixture::trace(scene,origin(scene),ray(scene,4,4,9),1.333f);
    const bool entering=origin(scene)[2]>0;
    const double reflectedZ=entering?5:-5,transmittedZ=entering?-3:3;
    waterfixture::receiver(scene,reflectedZ,landing(expected.entry,expected.reflection,reflectedZ),{4,2,1},2);
    if(!expected.tir)
        waterfixture::receiver(scene,transmittedZ,landing(expected.entry,expected.refraction,transmittedZ),{8,3,1},2);
}

inline bool tampered(const arbitgpu::smoothglass::Prepared& prepared,
    const std::array<arbitgpu::NativeRawPassPixels,3>& native,
    const arbitgpu::NativeRawPassPixels& objects,const arbitgpu::NativeRawPassPixels& materials,
    const arbitgpu::NativeRawPassPixels& normal,const arbitgpu::NativeRawPassPixels& color,
    const std::shared_ptr<const arbitgpu::roughglasstiles::Schedule>& schedule,
    std::size_t covered,std::string& error)
{
    for(int kind=0;kind<9;++kind) {
        auto receipts=native;
        auto ids=materials;
        switch(kind) {
            case 0: receipts[0].bytes.pop_back();break;
            case 1: ++receipts[2].height;break;
            case 2: write(receipts[1],covered,0,std::numeric_limits<float>::quiet_NaN());break;
            case 3: write(receipts[0],covered,0,.5f);break;
            case 4: write(receipts[0],covered,3,0);break;
            case 5: write(receipts[1],covered,3,1.5f);break;
            case 6: {
                const std::uint32_t wrong=prepared.material.value+1;
                std::memcpy(ids.bytes.data()+covered*4,&wrong,4);break;
            }
            case 7:
                for(auto& image:receipts) for(std::size_t c=0;c<4;++c) write(image,covered,c,0);
                break;
            case 8:
                write(receipts[0],covered,0,.99f);write(receipts[1],covered,0,.01f);write(receipts[2],covered,0,0);
                break;
        }
        arbitgpu::NativeFixtureSceneStats stats;stats.roughGlassCoveredPixels=987;
        if(arbitgpu::smoothglass::completeRoughFrame(prepared,objects.width,objects.height,
                receipts[0],receipts[1],receipts[2],objects,ids,stats,error)
            ||stats.roughGlassReceiptComplete||stats.roughGlassCoveredPixels!=987)
        {error="GL rough water tamper published a partial energy receipt";return false;}
    }
    using namespace arbitgpu::roughglasstiles;
    const NativeTileReceipt receipt{true,&native[0],&native[1],&native[2],&objects,&materials,&normal,&color};
    for(int kind=0;kind<4;++kind) {
        Completion completion(schedule);
        auto key=schedule->key();
        auto candidate=receipt;
        if(kind==0) candidate.gpuCompleted=false;
        if(kind==1) candidate.tail=nullptr;
        if(kind==2) key.runtime=std::make_shared<const arbitgpu::NativeFixtureSceneRuntimeInputs>(*key.runtime);
        arbitgpu::NativeFixtureSceneStats stats;stats.roughGlassCoveredPixels=987;
        Aggregate aggregate;aggregate.completedPixels=987;
        if((kind!=3 && completion.accept(key,schedule->tiles()[0],candidate,error))
            ||completion.publish(schedule->key(),stats,aggregate,error)
            ||stats.roughGlassReceiptComplete||stats.roughGlassCoveredPixels!=987||aggregate.completedPixels!=987)
        {error="GL rough water incomplete GPU, missing attachment, stale owner or partial tile published";return false;}
    }
    Completion duplicate(schedule);
    if(!duplicate.accept(schedule->key(),schedule->tiles()[0],receipt,error))return false;
    arbitgpu::NativeFixtureSceneStats stats;Aggregate aggregate;
    if(duplicate.accept(schedule->key(),schedule->tiles()[0],receipt,error)
        ||duplicate.publish(schedule->key(),stats,aggregate,error))
    {error="GL rough water duplicate completion published";return false;}
    error.clear();return true;
}

inline bool check(arbitgpu::NativeFixtureSceneBackend& backend,Visual3DScene authored,float roughness,
    bool exerciseOwnership,std::vector<std::uint8_t>& bytes,bool& hdr,bool& occlusion,std::string& error,
    const waterfixture::EnvironmentRadiance& environment={})
{
    constexpr std::uint32_t extent=9;
    const auto snapshot=std::make_shared<const Visual3DScene>(std::move(authored));
    const auto material=waterfixture::binding(snapshot,videohelper::materialprogram::BackendTarget::OpenGl,error,roughness);
    if(!material)return false;
    videorender::fixture3d::FixtureSceneRenderer renderer(backend);
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;inputs.linearColor=true;
    videorender::fixture3d::RenderedFrame preview,exported;
    if(!renderer.renderPreview(snapshot,material,{},inputs,{extent,extent},"native-gpu",preview,error)
        ||!renderer.renderExport(snapshot,material,{},inputs,{extent,extent},"native-gpu",exported,error))
    {error="GL rough water native preview/export failed: "+error;return false;}
    if(!preview.stats.roughGlassReceiptComplete||preview.stats.roughGlassCoveredPixels==0
        ||preview.stats.roughGlassReceiptBytes!=extent*extent*48
        ||preview.stats.roughGlassMaximumEnergyResidual>.00002
        ||preview.stats.roughGlassMaximumUnresolvedPower>.0001)
    {error="GL rough water omitted its complete float32 power/storage receipt";return false;}
    std::array<arbitgpu::NativeRawPassPixels,3> receipt,exportReceipt;
    arbitgpu::NativeRawPassPixels objects,materials,exportObjects,exportMaterials,color,exportColor,normal;
    if(!arbitgpu::readOpenGlRoughGlassReceiptForFixture(*preview.nativeFrame,receipt,objects,materials,error)
        ||!arbitgpu::readOpenGlRoughGlassReceiptForFixture(*exported.nativeFrame,exportReceipt,exportObjects,exportMaterials,error)
        ||!preview.nativeFrame->readRawPass(renderpassoutput::Output::Color,color,error)
        ||!exported.nativeFrame->readRawPass(renderpassoutput::Output::Color,exportColor,error)
        ||!preview.nativeFrame->readRawPass(renderpassoutput::Output::Normal,normal,error))return false;
    if(color.format!=arbitgpu::NativeTexturePixelFormat::Rgba16Float
        ||normal.format!=arbitgpu::NativeTexturePixelFormat::Rgba16Float
        ||objects.format!=arbitgpu::NativeTexturePixelFormat::R32Uint
        ||materials.format!=arbitgpu::NativeTexturePixelFormat::R32Uint
        ||color.bytes!=exportColor.bytes||objects.bytes!=exportObjects.bytes||materials.bytes!=exportMaterials.bytes)
    {error="GL rough water native preview/export lost exact Color or visible raw identity";return false;}
    for(std::size_t i=0;i<receipt.size();++i)
        if(receipt[i].format!=arbitgpu::NativeTexturePixelFormat::Rgba32Float||receipt[i].bytes!=exportReceipt[i].bytes)
        {error="GL rough water native preview/export lost float32 receipt identity";return false;}
    std::size_t firstCovered=extent*extent,coveredCount=0;
    for(std::size_t y=0;y<extent;++y) for(std::size_t x=0;x<extent;++x) {
        const auto pixel=y*extent+x;
        const auto direction=ray(*snapshot,x,y,extent);
        const auto first=waterfixture::hit(*snapshot,origin(*snapshot),direction);
        if(!first.found||identity(objects,pixel)!=snapshot->objects[first.object].id.value
            ||identity(materials,pixel)!=snapshot->objects[first.object].material.value)
        {error="GL rough water raw IDs differ from independent primary triangle visibility";return false;}
        if(first.object!=0) {
            occlusion=true;
            for(const auto& image:receipt) for(std::size_t c=0;c<4;++c)
                if(scalar(image,pixel,c)!=0) {error="GL rough water receipt leaked through an opaque foreground";return false;}
            continue;
        }
        ++coveredCount;firstCovered=std::min(firstCovered,pixel);
        const auto expected=waterfixture::traceRough(*snapshot,origin(*snapshot),direction,1.333f,roughness,environment);
        if(!expected.complete||scalar(receipt[0],pixel,3)!=1||scalar(receipt[2],pixel,3)!=1
            ||scalar(receipt[1],pixel,3)!=expected.closestHitQueries)
        {error="GL rough water completion or query receipt differs from independent binary64 schedule";return false;}
        for(std::size_t c=0;c<3;++c) {
            const double powers[]{expected.energy.terminal[c],expected.energy.masked[c],expected.energy.tail[c]};
            for(std::size_t p=0;p<3;++p)
                if(!std::isfinite(scalar(receipt[p],pixel,c))||std::abs(scalar(receipt[p],pixel,c)-powers[p])>.000004)
                {error="GL rough water per-pixel power differs from independent binary64 interface oracle";return false;}
            std::uint16_t half=0;std::memcpy(&half,color.bytes.data()+pixel*8+c*2,2);
            const double actual=hdrimage::halfToFloat(half);
            hdr|=actual>1;
            if(!std::isfinite(actual)||std::abs(actual-expected.rgb[c])>.003*std::max(1.0,std::abs(expected.rgb[c])))
            {error="GL rough water HDR pixel "+std::to_string(pixel)+" channel "+std::to_string(c)
                +" expected "+std::to_string(expected.rgb[c])+", got "+std::to_string(actual);return false;}
        }
        std::uint16_t marker=0;std::memcpy(&marker,normal.bytes.data()+pixel*8+6,2);
        if(marker!=0x3c00u) {error="GL rough water changed public Normal alpha completion semantics";return false;}
    }
    if(coveredCount!=preview.stats.roughGlassCoveredPixels)
    {error="GL rough water complete coverage differs from raw visibility";return false;}
    bytes=color.bytes;
    if(exerciseOwnership) {
        std::shared_ptr<const arbitgpu::roughglasstiles::Schedule> schedule;
        arbitgpu::roughglasstiles::Aggregate aggregate;
        if(!arbitgpu::readOpenGlRoughGlassTileSummaryForFixture(*preview.nativeFrame,schedule,aggregate,error)
            ||schedule->prepared().transport!=surfacematerial::TransportMode::ThinInterface
            ||!tampered(schedule->prepared(),receipt,objects,materials,normal,color,schedule,firstCovered,error))return false;
        const auto preparation=backend.prepare(snapshot,material->nativeProgram());
        if(!preparation.prepared) {error=preparation.error;return false;}
        const auto tiled=backend.render(snapshot,preparation.resources,100,100,inputs);
        std::shared_ptr<const arbitgpu::roughglasstiles::Schedule> tiledSchedule;
        arbitgpu::roughglasstiles::Aggregate tiledAggregate;
        if(!tiled.rendered||!tiled.frame
            ||!arbitgpu::readOpenGlRoughGlassTileSummaryForFixture(*tiled.frame,tiledSchedule,tiledAggregate,error)
            ||tiledSchedule->tiles().size()<2||tiledAggregate.completedPixels!=10000
            ||tiledAggregate.completedTiles!=tiledSchedule->tiles().size()
            ||tiledAggregate.peakReceiptBytes!=tiled.stats.roughGlassReceiptBytes
            ||tiledAggregate.peakReceiptBytes>arbitgpu::roughglasstiles::kMaximumTilePixels*48)
        {error="GL rough water did not publish a complete bounded tiled frame: "+tiled.error+error;return false;}
        std::array<arbitgpu::NativeRawPassPixels,3> retained;
        arbitgpu::NativeRawPassPixels retainedObjects,retainedMaterials;
        if(arbitgpu::readOpenGlRoughGlassReceiptForFixture(*tiled.frame,retained,retainedObjects,retainedMaterials,error))
        {error="GL rough water tiled scratch was exposed as a complete full-frame receipt";return false;}
        error.clear();
        if(!arbitgpu::readOpenGlRoughGlassReceiptForFixture(*preview.nativeFrame,retained,retainedObjects,retainedMaterials,error))return false;
        for(std::size_t i=0;i<receipt.size();++i) if(retained[i].bytes!=receipt[i].bytes)
        {error="GL rough water later frame overwrote a retained receipt owner";return false;}
        for(unsigned pass=1;pass<=7;++pass) {
            auto aovInputs=inputs;aovInputs.imageOutput=static_cast<renderpassoutput::Output>(pass);
            const auto aov=backend.render(snapshot,preparation.resources,extent,extent,aovInputs);
            if(!aov.rendered||!aov.frame||!aov.stats.roughGlassReceiptComplete)
            {error="GL rough water AOV bypassed or refused complete rough energy admission: "+aov.error;return false;}
        }
        const auto stale=backend.render(std::make_shared<const Visual3DScene>(*snapshot),preparation.resources,extent,extent,inputs);
        if(stale.rendered||stale.frame) {error="GL rough water stale scene owner published";return false;}
    }
    std::cout<<"rough-water-pixels backend=opengl roughness="<<roughness
        <<" camera-z="<<origin(*snapshot)[2]<<" covered="<<coveredCount
        <<" residual="<<preview.stats.roughGlassMaximumEnergyResidual
        <<" unresolved="<<preview.stats.roughGlassMaximumUnresolvedPower<<'\n';
    error.clear();return true;
}

inline bool run(arbitgpu::NativeFixtureSceneBackend& backend,std::string& error)
{
    auto authored=waterfixture::scene(20);
    authored.lights[0].color={};
    const auto narrow=waterfixture::traceRough(authored,origin(authored),ray(authored,4,4,9),1.333f,.04);
    const auto broad=waterfixture::traceRough(authored,origin(authored),ray(authored,4,4,9),1.333f,.08);
    double narrowEdge=-1e30,broadEdge=-1e30;
    for(std::size_t i=0;i<16;++i) {
        narrowEdge=std::max(narrowEdge,landing(narrow.entry,narrow.refraction[i],-3));
        broadEdge=std::max(broadEdge,landing(broad.entry,broad.refraction[i],-3));
    }
    if(!narrow.complete||!broad.complete||!(broadEdge>narrowEdge))
    {error="GL rough water fixture lacks an independently broadened refracted footprint";return false;}
    smoothglassfixture::receiver(authored,float((narrowEdge+broadEdge)*.5),{8,3,1});
    waterfixture::receiver(authored,5,landing(narrow.entry,narrow.reflection[0],5),{.8,.4,.1},2);
    std::vector<std::uint8_t> low,high,changed;
    bool hdr=false,occlusion=false;
    {
        std::vector<std::uint8_t> legacy;D3 legacyRgb{};
        if(!waterfixture::render(backend,authored,0,legacy,legacyRgb,error))return false;
        const auto snapshot=std::make_shared<const Visual3DScene>(authored);
        const auto material=waterfixture::binding(snapshot,videohelper::materialprogram::BackendTarget::OpenGl,error,0);
        if(!material)return false;
        const auto preparation=backend.prepare(snapshot,material->nativeProgram());
        if(!preparation.prepared) {error=preparation.error;return false;}
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;inputs.linearColor=true;
        const auto zero=backend.render(snapshot,preparation.resources,9,9,inputs);
        arbitgpu::NativeRawPassPixels color,objects,materials;
        std::array<arbitgpu::NativeRawPassPixels,3> receipts;
        if(!zero.rendered||!zero.frame||zero.stats.roughGlassReceiptComplete||zero.stats.roughGlassReceiptBytes
            ||!zero.frame->readRawPass(renderpassoutput::Output::Color,color,error)||color.bytes!=legacy
            ||arbitgpu::readOpenGlRoughGlassReceiptForFixture(*zero.frame,receipts,objects,materials,error))
        {error="GL water exact zero changed released smooth pixels or invented a rough receipt";return false;}
        error.clear();
    }
    if(!check(backend,authored,.04f,true,low,hdr,occlusion,error)
        ||!check(backend,authored,.08f,false,high,hdr,occlusion,error)||low==high)
    {error="GL rough water roughness did not change the authored triangle footprint: "+error;return false;}
    auto blocker=authored;
    waterfixture::receiver(blocker,-1,landing(broad.entry,broad.refraction[0],-1),{},.4);
    if(!check(backend,blocker,.08f,false,changed,hdr,occlusion,error)||changed==high)
    {error="GL rough water actual triangle blocker did not change transport: "+error;return false;}
    auto foreground=authored;
    const auto centre=waterfixture::plus(origin(foreground),ray(foreground,4,4,9),1);
    waterfixture::receiver(foreground,centre[2],centre[0]+.0023,{.2,.1,.05},.0025);
    foreground.materials[foreground.materialCount-1].doubleSided=true;
    if(!check(backend,foreground,.04f,false,changed,hdr,occlusion,error)||!occlusion)return false;
    for(float angle:{20.0f,60.0f}) {
        auto internal=below(angle);fullReceivers(internal);
        const auto oracle=waterfixture::traceRough(internal,origin(internal),ray(internal,4,4,9),1.333f,.04);
        if(!oracle.complete||(angle==60 && oracle.tir!=16))
        {error="GL rough water fixture did not exercise the interior medium and TIR";return false;}
        if(!check(backend,internal,.04f,false,changed,hdr,occlusion,error))return false;
    }
    auto normals=authored;
    std::vector<std::uint8_t> flat;
    if(!check(backend,normals,.04f,false,flat,hdr,occlusion,error))return false;
    const auto tilted=waterfixture::unit({-.02,0,1});
    for(std::size_t v=0;v<4;++v) normals.vertices[v].normal={float(tilted[0]),0,float(tilted[2])};
    if(!check(backend,normals,.04f,false,changed,hdr,occlusion,error)||changed==flat)
    {error="GL rough water evaluated wave normals did not change transport: "+error;return false;}
    auto environment=waterfixture::scene(20);
    environment.lights[0].color={1,1,1};environment.lights[0].sky.version=1;
    environment.lights[0].sky.sunAzimuthDegrees=-20;
    environment.lights[0].sky.sunElevationDegrees=0;
    environment.lights[0].sky.sunRadiusDegrees=3;
    environment.lights[0].sky.sunColor={4,3,2};environment.lights[0].sky.sunIntensity=2;
    std::vector<std::uint8_t> sky;
    if(!check(backend,environment,.04f,false,sky,hdr,occlusion,error))return false;
    environment.lights[0].sky.sunIntensity=0;
    if(!check(backend,environment,.04f,false,changed,hdr,occlusion,error)||changed==sky||!hdr)
    {error="GL rough water authored sky or unclamped HDR radiance was not exercised: "+error;return false;}
    const auto image=environmentsecondaryfixture::imageBytes(0);
    const visualenvironmentimage::Identity imageKey{"native-rough-water-hdr",1,modelpayload::detail::digestHex(image)};
    const visualenvironmentimage::Metadata metadata{8,4,visualenvironmentimage::SourcePrimaries::Rec709D65};
    auto& store=videohelper::environmentimage::store();std::uint64_t received=0;
    if(!store.begin("native-rough-water-hdr-upload",imageKey,metadata,image.size(),error)
        ||!store.append("native-rough-water-hdr-upload",0,environmentImageBase64(image),received,error)
        ||!store.commit("native-rough-water-hdr-upload",error))return false;
    auto imageScene=waterfixture::scene(20);imageScene.lights[0].color={1,1,1};
    imageScene.lights[0].environmentImage={1,imageKey,0,0};
    const waterfixture::EnvironmentRadiance imageOracle=[](const auto& light,D3 direction) {
        return environmentsecondaryfixture::radiance(light,direction,0);
    };
    std::vector<std::uint8_t> imageBaseline;
    if(!check(backend,imageScene,.04f,false,imageBaseline,hdr,occlusion,error,imageOracle))return false;
    imageScene.lights[0].environmentImage.orientationDegrees=90;
    if(!check(backend,imageScene,.04f,false,changed,hdr,occlusion,error,imageOracle)||changed==imageBaseline)
    {error="GL rough water did not sample the independently authored HDR image orientation: "+error;return false;}
    imageScene.lights[0].environmentImage.orientationDegrees=0;
    imageScene.lights[0].environmentImage.exposureStops=1;
    if(!check(backend,imageScene,.04f,false,changed,hdr,occlusion,error,imageOracle)||changed==imageBaseline)
    {error="GL rough water HDR image exposure did not change owned linear Color: "+error;return false;}
    auto recursive=waterfixture::scene(0);
    for(int c=0;c<4;++c) {recursive.vertices[4+c]=recursive.vertices[c];recursive.vertices[4+c].position.z=5;recursive.vertices[4+c].normal={0,0,-1};}
    const std::uint32_t indices[]{4,6,5,4,7,6};std::copy_n(indices,6,recursive.indices.begin()+6);
    recursive.vertexCount=recursive.objects[0].vertexCount=8;recursive.indexCount=recursive.objects[0].indexCount=12;
    const auto snapshot=std::make_shared<const Visual3DScene>(recursive);
    const auto material=waterfixture::binding(snapshot,videohelper::materialprogram::BackendTarget::OpenGl,error,.04f);
    if(!material)return false;
    const auto preparation=backend.prepare(snapshot,material->nativeProgram());
    if(!preparation.prepared) {error=preparation.error;return false;}
    for(unsigned pass=0;pass<=7;++pass) {
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;inputs.linearColor=true;
        inputs.imageOutput=static_cast<renderpassoutput::Output>(pass);
        const auto refused=backend.render(snapshot,preparation.resources,9,9,inputs);
        if(refused.rendered||refused.frame||refused.stats.roughGlassReceiptComplete)
        {error="GL rough water re-hit published partial Color or AOV";return false;}
    }
    auto grazing=waterfixture::scene(75);
    const auto unresolved=waterfixture::traceRough(grazing,origin(grazing),ray(grazing,4,4,9),1.333f,.9);
    if(unresolved.complete||unresolved.status!=arbitgpu::smoothglass::RoughTransportStatus::UnresolvedEnergy)
    {error="GL rough water fixture lacks an independently unresolved single-scatter case";return false;}
    const auto grazingSnapshot=std::make_shared<const Visual3DScene>(grazing);
    const auto grazingMaterial=waterfixture::binding(grazingSnapshot,videohelper::materialprogram::BackendTarget::OpenGl,error,.9f);
    if(!grazingMaterial)return false;
    const auto grazingPreparation=backend.prepare(grazingSnapshot,grazingMaterial->nativeProgram());
    if(!grazingPreparation.prepared) {error=grazingPreparation.error;return false;}
    arbitgpu::NativeFixtureSceneRuntimeInputs linear;linear.linearColor=true;
    const auto refused=backend.render(grazingSnapshot,grazingPreparation.resources,9,9,linear);
    if(refused.rendered||refused.frame||refused.stats.roughGlassReceiptComplete)
    {error="GL rough water published a single-scatter case above the unresolved-power bound";return false;}
    error.clear();return true;
}
} // namespace videohelper::tests::openglroughwaterfixture
