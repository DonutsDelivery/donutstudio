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
    // The mip levels are charged to the ledger but never count against the decoded image limit,
    // so an image that fits as decoded pixels keeps fitting.
    std::uint64_t limitedBytes=0;
    // The admitted lease is also the environment-image class in the residency ledger.
    residency::ResidencyCharge residencyCharge;
    ~GpuBudget() { std::lock_guard<std::mutex> lock(mutex); bytes-=limitedBytes; --count; }
    static std::shared_ptr<GpuBudget> acquire(std::uint64_t requested,std::uint64_t mipBytes=0)
    {
        std::lock_guard<std::mutex> lock(mutex);
        // Leases only tally residency; a scene uses as many environment images as it binds.
        auto lease=std::make_shared<GpuBudget>(); lease->ownedBytes=requested+mipBytes; lease->limitedBytes=requested;
        lease->residencyCharge=residency::ResidencyCharge(residency::BackingClass::environmentImage,requested+mipBytes);
        bytes+=requested; ++count; return lease;
    }
};
struct Prepared
{
    videohelper::environmentimage::ResidentPtr resident;
    std::shared_ptr<GpuBudget> gpuBudget;
    std::vector<std::array<float,4>> texels;
    // Levels 1 and up, for the GGX and diffuse sample footprints. Level zero is texels.
    std::vector<std::vector<std::array<float,4>>> mips;
};
inline int mipLevelCount(std::uint32_t width,std::uint32_t height)
{
    int levels=1;
    for (auto extent=std::min(width,height);extent>1;extent/=2) ++levels;
    return levels;
}
inline std::uint64_t mipBytes(std::uint32_t width,std::uint32_t height)
{
    std::uint64_t total=0;
    for (int level=1;level<mipLevelCount(width,height);++level)
        total+=std::uint64_t(std::max<std::uint32_t>(width>>level,1))*std::max<std::uint32_t>(height>>level,1)*sizeof(std::array<float,4>);
    return total;
}
// Each level averages the one above it, weighted by the solid angle of the equirectangular rows,
// so a level keeps the image's mean radiance.
inline std::vector<std::vector<std::array<float,4>>> buildMips(const std::vector<std::array<float,4>>& base,
    std::uint32_t width,std::uint32_t height)
{
    constexpr double pi=3.141592653589793;
    std::vector<std::vector<std::array<float,4>>> levels;
    const auto* source=&base;
    auto sourceWidth=width,sourceHeight=height;
    for (int level=1;level<mipLevelCount(width,height);++level)
    {
        const auto w=std::max<std::uint32_t>(sourceWidth/2,1),h=std::max<std::uint32_t>(sourceHeight/2,1);
        std::vector<std::array<float,4>> next(std::size_t(w)*h);
        for (std::uint32_t y=0;y<h;++y)
            for (std::uint32_t x=0;x<w;++x)
            {
                double sum[3]{},weight=0;
                for (auto sy=y*sourceHeight/h;sy<(y+1)*sourceHeight/h;++sy)
                {
                    const auto rowWeight=std::max(std::sin((double(sy)+0.5)/sourceHeight*pi),1.0e-6);
                    for (auto sx=x*sourceWidth/w;sx<(x+1)*sourceWidth/w;++sx)
                    {
                        const auto& texel=(*source)[std::size_t(sy)*sourceWidth+sx];
                        for (int c=0;c<3;++c) sum[c]+=double(texel[std::size_t(c)])*rowWeight;
                        weight+=rowWeight;
                    }
                }
                next[std::size_t(y)*w+x]={float(sum[0]/weight),float(sum[1]/weight),float(sum[2]/weight),1};
            }
        levels.push_back(std::move(next));
        source=&levels.back();
        sourceWidth=w; sourceHeight=h;
    }
    return levels;
}
inline bool validBinding(const videohelper::environmentimage::Resident& resident,
                         const HarmonicMIDI::grid::SceneLightRecord& light)
{
    if (!HarmonicMIDI::grid::validSceneEnvironmentImage(light)) return false;
    const auto& key=resident.payload->key(); const auto& image=light.environmentImage;
    if (key.id!=image.asset.id || key.version!=image.asset.version || key.contentSha256!=image.asset.contentSha256) return false;
    const auto scale=std::exp2(static_cast<double>(image.exposureStops))*light.intensity;
    const double tint[] {light.color.x*scale,light.color.y*scale,light.color.z*scale};
    for (std::size_t c=0;c<3;++c)
        if (!std::isfinite(tint[c]) || tint[c]<0) return false;
    return true;
}
// Select only when transportShader(..., true) and its native texture arguments
// are selected together. None binds the image for every material. Each lobe reads
// the sky through the same uSky branch whether the sky is analytic or an image.
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
    Prepared candidate;
    candidate.resident=videohelper::environmentimage::store().resolve(owner->environmentImage.asset);
    if (!candidate.resident) { error="HDR environment exact payload is not resident"; return false; }
    if (!validBinding(*candidate.resident,*owner))
    { error="HDR environment exposure and tint must be finite and nonnegative"; return false; }
    const auto bytes=candidate.resident->image.pixels.size()*sizeof(std::array<float,4>);
    const auto& imageSize=candidate.resident->image.metadata;
    try { candidate.gpuBudget=GpuBudget::acquire(bytes,mipBytes(imageSize.width,imageSize.height)); }
    catch (const std::bad_alloc&) { error="HDR environment native budget allocation failed"; return false; }
    if (!candidate.gpuBudget) { error="HDR environment native texture budget is occupied"; return false; }
    try
    {
        candidate.texels.reserve(candidate.resident->image.pixels.size());
        for (const auto& pixel:candidate.resident->image.pixels)
            candidate.texels.push_back({pixel[0],pixel[1],pixel[2],1});
        candidate.mips=buildMips(candidate.texels,imageSize.width,imageSize.height);
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
    if (!std::isfinite(result[0][0]) || !std::isfinite(result[0][1]) || !std::isfinite(result[0][2]))
    {
        // A tint past a float keeps its hue at the largest float.
        const double gain=std::exp2(double(light.environmentImage.exposureStops))*light.intensity;
        const double r=light.color.x*gain,g=light.color.y*gain,b=light.color.z*gain;
        const auto fit=double(std::numeric_limits<float>::max())/std::max({r,g,b});
        result[0]={float(r*fit),float(g*fit),float(b*fit),2};
    }
    result[1]={light.environmentImage.orientationDegrees*0.01745329251994329577f,0,0,0};
    return result;
}
// Bilinear sampling is explicit on both APIs. Longitude wraps, latitude clamps
// at pixel centers. GGX roughness uses the same finite visible-normal quadrature as
// analytic sky, but a sample of an image reads the level its solid angle covers (filtered
// importance sampling, GPU Gems 3 chapter 20). Reading level zero through a fixed set of
// directions drew one sharp copy of the image per direction, which showed as petals meeting
// where the tangent frame turns at n=+Z. Direct views still read level zero. No color
// transfer occurs. A radiance past Raster's RGBA16F range clips here keeping its hue, and no
// frame refuses over it.
template<class String> inline String shaderRecipe(bool metal)
{
    String source=R"shader(
vec3 environmentImageLevel(vec3 direction,vec4 tint,vec4 control,int level IMAGE_PARAM) {
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
    vec3 texel=mix(mix(c00,c10,f.x),mix(c01,c11,f.x),f.y);
    vec3 radiance=texel*tint.rgb;
    if(max(max(radiance.x,radiance.y),radiance.z)<=60000.0) return radiance;
    float peak=max(max(texel.x,texel.y),texel.z);
    vec3 hue=texel/peak*tint.rgb;
    return hue*min(peak,60000.0/max(max(hue.x,hue.y),hue.z));
}
vec3 environmentImageRadiance(vec3 direction,vec4 tint,vec4 control IMAGE_PARAM) {
    return environmentImageLevel(direction,tint,control,0 IMAGE_ARG);
}
vec3 environmentImageFiltered(vec3 direction,vec4 tint,vec4 control,float solidAngle IMAGE_PARAM) {
    int level=0;
    ivec2 size=IMAGE_SIZE;
    float lastLevel=floor(log2(float(min(size.x,size.y))));
    float lod=clamp(0.5*log2(max(solidAngle*float(size.x)*float(size.y)/12.566370614359172,1.0)),0.0,lastLevel);
    level=int(lod);
    vec3 fine=environmentImageLevel(direction,tint,control,level IMAGE_ARG);
    if(float(level)>=lastLevel) return fine;
    return mix(fine,environmentImageLevel(direction,tint,control,level+1 IMAGE_ARG),lod-float(level));
}
)shader";
    auto base=analyticsky::shaderRecipe<String>(false);
    base=base.substr(base.find("vec3 analyticSkyBase"));
    const auto replace=[](String& text,const auto& from,const auto& to) { surfacesource::replace(text,from,to); };
    replace(base,"analyticSkyBase","environmentImageBase");
    replace(base,"vec4 zenith,vec4 horizon,\n                     vec4 ground,vec4 sun,vec4 sunDirection","vec4 tint,vec4 control IMAGE_PARAM");
    // Solid angle of one sample: 1/(N*pdf). The GGX pdf is D*G1(v)/(4*n.v), the diffuse pdf n.l/pi.
    replace(base,"result+=analyticSkyRadiance(l,zenith,horizon,ground,sun,sunDirection)\n                *fresnel*g1l/64.0;",
        "float nh=clamp(dot(n,h),0.0,1.0);\n"
        "            float lobe=nh*nh*(a2-1.0)+1.0;\n"
        "            float visibleD=a2/(3.141592653589793*lobe*lobe)*2.0*nv/(nv+sqrt(a2+(1.0-a2)*nv*nv));\n"
        "            result+=environmentImageFiltered(l,tint,control,4.0*nv/(64.0*max(visibleD,1.0e-6)) IMAGE_ARG)\n"
        "                *fresnel*g1l/64.0;");
    replace(base,"result+=analyticSkyRadiance(l,zenith,horizon,ground,sun,sunDirection)\n            *(1.0-fresnel)",
        "result+=environmentImageFiltered(l,tint,control,3.141592653589793/(32.0*max(sqrt(1.0-u),1.0e-3)) IMAGE_ARG)\n            *(1.0-fresnel)");
    source+=base;
    if (metal)
    {
        replace(source,"IMAGE_PARAM",",texture2d<float> environmentTexture");
        replace(source,"IMAGE_ARG",",environmentTexture");
        replace(source,"IMAGE_SIZE","int2(environmentTexture.get_width(uint(level)),environmentTexture.get_height(uint(level)))");
        replace(source,"IMAGE_READ(x0,y0)","environmentTexture.read(uint2(x0,y0),uint(level)).rgb");
        replace(source,"IMAGE_READ(x1,y0)","environmentTexture.read(uint2(x1,y0),uint(level)).rgb");
        replace(source,"IMAGE_READ(x0,y1)","environmentTexture.read(uint2(x0,y1),uint(level)).rgb");
        replace(source,"IMAGE_READ(x1,y1)","environmentTexture.read(uint2(x1,y1),uint(level)).rgb");
        replace(source,"atan(d.x,d.z)","atan2(d.x,d.z)");
        replace(source,"ivec2","int2"); replace(source,"vec2","float2");
        replace(source,"vec3","float3"); replace(source,"vec4","float4");
    }
    else
    {
        source="uniform sampler2D uEnvironmentImage;\n"+source;
        replace(source,"IMAGE_PARAM",""); replace(source,"IMAGE_ARG","");
        replace(source,"IMAGE_SIZE","textureSize(uEnvironmentImage,level)");
        for (const auto& xy:{"x0,y0","x1,y0","x0,y1","x1,y1"})
            replace(source,String("IMAGE_READ(")+surfacesource::literal<String>(xy)+")",String("texelFetch(uEnvironmentImage,ivec2(")+surfacesource::literal<String>(xy)+"),level).rgb");
    }
    return source;
}
inline std::string shader(bool metal) { return shaderRecipe<std::string>(metal); }
} // namespace arbitgpu::environmentimage
