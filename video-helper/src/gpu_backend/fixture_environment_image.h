#pragma once

#include "backend.h"
#include "fixture_analytic_sky.h"
#include "fixture_smooth_glass.h"
#include "gpu_residency_ledger.h"
#include "../environment_image_payload_transport.h"

namespace arbitgpu::environmentimage
{
struct GpuBudget final
{
    static inline std::mutex mutex;
    static inline std::uint64_t bytes=0;
    static inline std::size_t count=0;
    std::uint64_t ownedBytes=0;
    // The admitted lease is also the environment-image class in the residency ledger.
    residency::ResidencyCharge residencyCharge;
    ~GpuBudget() { std::lock_guard<std::mutex> lock(mutex); bytes-=ownedBytes; --count; }
    static std::shared_ptr<GpuBudget> acquire(std::uint64_t requested)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (count>=visualenvironmentimagepayload::kMaxResidentPayloads
            || requested>visualenvironmentimagepayload::kMaxResidentDecodedBytes-bytes) return {};
        auto lease=std::make_shared<GpuBudget>(); lease->ownedBytes=requested;
        lease->residencyCharge=residency::ResidencyCharge(residency::BackingClass::environmentImage,requested);
        bytes+=requested; ++count; return lease;
    }
};
struct Prepared
{
    videohelper::environmentimage::ResidentPtr resident;
    std::shared_ptr<GpuBudget> gpuBudget;
    std::vector<std::array<float,4>> texels;
};
inline bool validBinding(const videohelper::environmentimage::Resident& resident,
                         const HarmonicMIDI::grid::SceneLightRecord& light)
{
    if (!HarmonicMIDI::grid::validSceneEnvironmentImage(light)) return false;
    const auto& key=resident.payload->key(); const auto& image=light.environmentImage;
    if (key.id!=image.asset.id || key.version!=image.asset.version || key.contentSha256!=image.asset.contentSha256) return false;
    const auto scale=std::exp2(static_cast<double>(image.exposureStops))*light.intensity;
    const double tint[] {light.color.x*scale,light.color.y*scale,light.color.z*scale};
    for (std::size_t c=0;c<3;++c)
        if (!std::isfinite(tint[c]) || tint[c]<0 || tint[c]*resident.maximumRadiance[c]>visualenvironmentimage::kMaxRadiance) return false;
    return true;
}
// Select only when transportShader(..., true) and its native texture arguments
// are selected together. None preserves the released opaque-only admission.
enum class SecondaryRayConsumer { None, SmoothTransport };
inline bool prepare(const HarmonicMIDI::grid::Visual3DScene& scene,
                    const NativeFixtureSurfaceMaterialProgram* material, Prepared& output,
                    std::string& error, SecondaryRayConsumer consumer = SecondaryRayConsumer::None,
                    const smoothglass::Prepared* transport = nullptr)
{
    using namespace HarmonicMIDI::grid;
    const SceneLightRecord* owner=nullptr;
    for (std::size_t i=0;i<scene.lightCount;++i)
        if (scene.lights[i].environmentImage.version) owner=&scene.lights[i];
    if (!owner) return true;
    if (!validSceneEnvironmentImage(*owner) || !validSceneAnalyticSkySet(scene.lights.data(),scene.lightCount))
    { error="HDR environment requires one valid active sky owner"; return false; }
    const bool secondary = consumer == SecondaryRayConsumer::SmoothTransport && transport && transport->enabled
        && transport->sourceScene.get() == &scene && transport->sourceMaterial.get() == material
        && transport->query.mode == opaquevisibility::AdmissionMode::allSurfaces
        && transport->query.objectCount == scene.objectCount;
    if (consumer != SecondaryRayConsumer::None && !secondary)
    { error="HDR environment secondary consumer requires exact admitted smooth transport"; return false; }
    const auto unsupported=[&](const NativeFixtureSurfaceMaterialProgram& p) {
        const auto mode=static_cast<surfacematerial::TransportMode>(p.parameters.identifiers[1]);
        const bool traced=mode==surfacematerial::TransportMode::ThinInterface
            || mode==surfacematerial::TransportMode::ClosedVolume;
        return p.kind!=NativeFixtureMaterialKind::SurfacePbr
            || (p.parameters.transmissionIorClearcoat[0]>0 && !(secondary && traced && mode==transport->transport))
            || p.parameters.transmissionIorClearcoat[2]>0 || p.parameters.normalOpacity[3]<1;
    };
    if (material && (unsupported(*material) || std::any_of(material->objectPrograms.begin(),material->objectPrograms.end(),
        [&](const auto& p){return !p || unsupported(*p);})))
    { error="HDR environment currently supports opaque base GGX, without glass, clearcoat or diffraction"; return false; }
    for (std::size_t i=0;i<scene.materialCount;++i)
        if (scene.materials[i].alphaMode!=SceneAlphaMode::Opaque || scene.materials[i].opacity<1)
        { error="HDR environment currently requires opaque scene materials"; return false; }
    Prepared candidate;
    candidate.resident=videohelper::environmentimage::store().resolve(owner->environmentImage.asset);
    if (!candidate.resident) { error="HDR environment exact payload is not resident"; return false; }
    if (!validBinding(*candidate.resident,*owner))
    { error="HDR environment exposure and tint exceed native linear radiance range"; return false; }
    const auto bytes=candidate.resident->image.pixels.size()*sizeof(std::array<float,4>);
    try { candidate.gpuBudget=GpuBudget::acquire(bytes); }
    catch (const std::bad_alloc&) { error="HDR environment native budget allocation failed"; return false; }
    if (!candidate.gpuBudget) { error="HDR environment native texture budget is occupied"; return false; }
    try
    {
        candidate.texels.reserve(candidate.resident->image.pixels.size());
        for (const auto& pixel:candidate.resident->image.pixels)
            candidate.texels.push_back({pixel[0],pixel[1],pixel[2],1});
    }
    catch (const std::bad_alloc&) { error="HDR environment native staging allocation failed"; return false; }
    output=std::move(candidate);
    return true;
}
inline analyticsky::Uniforms uniforms(const HarmonicMIDI::grid::SceneLightRecord& light)
{
    const auto scale=std::exp2(light.environmentImage.exposureStops)*light.intensity;
    analyticsky::Uniforms result{};
    result[0]={light.color.x*scale,light.color.y*scale,light.color.z*scale,2};
    result[1]={light.environmentImage.orientationDegrees*0.01745329251994329577f,0,0,0};
    return result;
}
// Bilinear sampling is explicit on both APIs. Longitude wraps, latitude clamps
// at pixel centers, and only mip zero exists. GGX roughness uses the same finite
// visible-normal quadrature as analytic sky. No color transfer or clipping occurs.
template<class String> inline String shaderRecipe(bool metal)
{
    String source=R"shader(
vec3 environmentImageRadiance(vec3 direction,vec4 tint,vec4 control IMAGE_PARAM) {
    vec3 d=normalize(direction);
    float longitude=d.x*d.x+d.z*d.z>1.0e-12 ? atan(d.x,d.z) : 0.0;
    float u=fract((longitude-control.x)/6.283185307179586+0.5);
    float v=acos(clamp(d.y,-1.0,1.0))/3.141592653589793;
    ivec2 size=IMAGE_SIZE;
    vec2 p=vec2(u,v)*vec2(size)-0.5;
    ivec2 a=ivec2(floor(p)); vec2 f=fract(p);
    int x0=(a.x%size.x+size.x)%size.x;
    int x1=(x0+1)%size.x;
    int y0=clamp(a.y,0,size.y-1),y1=clamp(a.y+1,0,size.y-1);
    vec3 c00=IMAGE_READ(x0,y0),c10=IMAGE_READ(x1,y0);
    vec3 c01=IMAGE_READ(x0,y1),c11=IMAGE_READ(x1,y1);
    return mix(mix(c00,c10,f.x),mix(c01,c11,f.x),f.y)*tint.rgb;
}
)shader";
    auto base=analyticsky::shaderRecipe<String>(false);
    base=base.substr(base.find("vec3 analyticSkyBase"));
    const auto replace=[](String& text,const auto& from,const auto& to) { surfacesource::replace(text,from,to); };
    replace(base,"analyticSkyBase","environmentImageBase");
    replace(base,"vec4 zenith,vec4 horizon,\n                     vec4 ground,vec4 sun,vec4 sunDirection","vec4 tint,vec4 control IMAGE_PARAM");
    replace(base,"analyticSkyRadiance(l,zenith,horizon,ground,sun,sunDirection)",
        "environmentImageRadiance(l,tint,control IMAGE_ARG)");
    source+=base;
    if (metal)
    {
        replace(source,"IMAGE_PARAM",",texture2d<float> environmentTexture");
        replace(source,"IMAGE_ARG",",environmentTexture");
        replace(source,"IMAGE_SIZE","int2(environmentTexture.get_width(),environmentTexture.get_height())");
        replace(source,"IMAGE_READ(x0,y0)","environmentTexture.read(uint2(x0,y0)).rgb");
        replace(source,"IMAGE_READ(x1,y0)","environmentTexture.read(uint2(x1,y0)).rgb");
        replace(source,"IMAGE_READ(x0,y1)","environmentTexture.read(uint2(x0,y1)).rgb");
        replace(source,"IMAGE_READ(x1,y1)","environmentTexture.read(uint2(x1,y1)).rgb");
        replace(source,"atan(d.x,d.z)","atan2(d.x,d.z)");
        replace(source,"ivec2","int2"); replace(source,"vec2","float2");
        replace(source,"vec3","float3"); replace(source,"vec4","float4");
    }
    else
    {
        source="uniform sampler2D uEnvironmentImage;\n"+source;
        replace(source,"IMAGE_PARAM",""); replace(source,"IMAGE_ARG","");
        replace(source,"IMAGE_SIZE","textureSize(uEnvironmentImage,0)");
        for (const auto& xy:{"x0,y0","x1,y0","x0,y1","x1,y1"})
            replace(source,String("IMAGE_READ(")+surfacesource::literal<String>(xy)+")",String("texelFetch(uEnvironmentImage,ivec2(")+surfacesource::literal<String>(xy)+"),0).rgb");
    }
    return source;
}
inline std::string shader(bool metal) { return shaderRecipe<std::string>(metal); }
} // namespace arbitgpu::environmentimage
