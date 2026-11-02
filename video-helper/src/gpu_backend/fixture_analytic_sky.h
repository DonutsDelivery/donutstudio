#pragma once

#include "../../../shared/Visual3DScene.h"
#include "fixture_surface_source_admission.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace arbitgpu::analyticsky
{
// Five packed vec4s, identical on GL and Metal. Radiance is scene-linear RGB,
// in relative scene units per steradian, not calibrated SI watts. A sky
// color is a radiance, not irradiance. Light RGB/intensity tint the whole field.
// horizon.w packs heading (8 bits), type (6) and coverage (10) as an integer that
// binary32 holds exactly. ground.w is the authored drift. Zero means no clouds.
using Uniforms = std::array<std::array<float,4>,5>;
// Raster's RGBA16F target can't hold a brighter emission or ambient, and a float past this can
// overflow in the shader, so the peak clips here and keeps its hue. Path Trace reads the authored value.
// A light's colour and intensity multiply in the shader, and an infinite product times a zero
// attenuation is NaN. This keeps the product finite with room for attenuation and albedo, far
// above anything the target holds. Below it the stored values stay as they are.
inline void clipRasterLight(float* rgbIntensity) noexcept
{
    const double i=rgbIntensity[3],r=rgbIntensity[0]*i,g=rgbIntensity[1]*i,b=rgbIntensity[2]*i;
    const auto peak=std::max({r,g,b});
    if (!(peak>1.0e30)) return;
    const auto fit=1.0e30/peak;
    rgbIntensity[0]=float(r*fit); rgbIntensity[1]=float(g*fit); rgbIntensity[2]=float(b*fit); rgbIntensity[3]=1.0f;
}
inline void clipRasterRadiance(float* rgb,double gain=1.0) noexcept
{
    // An emission plus an instance's can sum past a float before it gets here.
    const double top=std::numeric_limits<float>::max();
    const double r=std::min(double(rgb[0]),top)*gain,g=std::min(double(rgb[1]),top)*gain,b=std::min(double(rgb[2]),top)*gain;
    const auto peak=std::max({r,g,b});
    const auto fit=peak>60000.0 ? 60000.0/peak : 1.0;
    rgb[0]=float(r*fit); rgb[1]=float(g*fit); rgb[2]=float(b*fit);
}
inline float packedClouds(const HarmonicMIDI::grid::SceneAnalyticSky& s,double headingDegrees)
{
    const auto coverage=std::lround(std::clamp<double>(s.cloudCoverage,0,1)*1023);
    if(coverage==0) return 0;
    const auto kind=std::lround(std::clamp<double>(s.cloudType,0,1)*63);
    const auto turns=headingDegrees/360;
    const auto heading=std::lround((turns-std::floor(turns))*256)%256;
    return static_cast<float>(heading*65536+kind*1024+coverage);
}
inline Uniforms uniforms(const HarmonicMIDI::grid::SceneLightRecord& light)
{
    const auto& s=light.sky;
    constexpr double radians=0.01745329251994329577;
    const auto az=(s.sunAzimuthDegrees+s.orientationDegrees)*radians;
    const auto el=s.sunElevationDegrees*radians;
    // Raster's RGBA16F target can't hold a brighter field, so a term's peak clips here and keeps
    // its hue. Path Trace reads the authored radiance and never sees this.
    const auto scaled=[&](HarmonicMIDI::grid::SceneVec3 c,float w,double limit) {
        const double gain=light.intensity;
        const double r=double(c.x)*light.color.x*gain,g=double(c.y)*light.color.y*gain,b=double(c.z)*light.color.z*gain;
        const auto peak=std::max({r,g,b});
        const auto fit=peak>limit ? limit/peak : 1.0;
        return std::array<float,4>{float(r*fit),float(g*fit),float(b*fit),w};
    };
    const auto color=[&](HarmonicMIDI::grid::SceneVec3 c,float w) { return scaled(c,w,60000.0); };
    // The sun's stored peak is this colour times storedSun, so only that product clips.
    const auto sunColor=scaled(s.sunColor,static_cast<float>(std::cos(s.sunRadiusDegrees*radians)),std::numeric_limits<float>::max());
    const auto sunPeak=std::max({sunColor[0],sunColor[1],sunColor[2]});
    const auto storedSun=sunPeak>0.0f ? std::min(s.sunIntensity,60000.0f/sunPeak) : s.sunIntensity;
    return {color(s.zenith,static_cast<float>(s.version)),
        color(s.horizon,packedClouds(s,static_cast<double>(s.cloudHeadingDegrees)+s.orientationDegrees)),
        color(s.ground,s.cloudCoverage>0 ? s.cloudDrift : 0.0f),sunColor,
        std::array<float,4>{static_cast<float>(std::sin(az)*std::cos(el)),
            static_cast<float>(std::sin(el)),static_cast<float>(std::cos(az)*std::cos(el)),storedSun}};
}

// This is an authored infinite radiance field, not an atmospheric simulation.
// Upper/lower hemispheres interpolate by |direction.y|. The sun is a smooth
// spherical cap: peak sunColor*sunIntensity, zero at the authored radius.
// Background and BRDF integration execute this exact common source. Only vector
// spelling changes for MSL; no backend-specific light model or sample sequence.
// Clouds are five octaves of value noise on a flat layer above the camera, so they
// appear in the background and in every reflection. Type 0 is puffy and type 1
// is streaky along the heading. Coverage 1 closes the sky. A cloud's radiance is the
// brightest ambient channel plus the sun cap's Lambert term (the smooth cap's
// irradiance is pi*(1-cos radius)*peak), so it follows the authored light.
template<class String> inline String shaderRecipe(bool metal)
{
    String source=R"shader(
float analyticSkyCloudHash(vec2 p) {
    vec3 q=fract(vec3(p.x,p.y,p.x)*0.1031);
    q+=dot(q,vec3(q.y,q.z,q.x)+33.33);
    return fract((q.x+q.y)*q.z);
}
float analyticSkyCloudNoise(vec2 p) {
    vec2 cell=floor(p);
    vec2 f=p-cell;
    f=f*f*(3.0-2.0*f);
    return mix(mix(analyticSkyCloudHash(cell),analyticSkyCloudHash(cell+vec2(1.0,0.0)),f.x),
               mix(analyticSkyCloudHash(cell+vec2(0.0,1.0)),analyticSkyCloudHash(cell+vec2(1.0,1.0)),f.x),f.y);
}
float analyticSkyCloudOpacity(vec3 d,vec4 horizon,vec4 ground) {
    if(horizon.w<=0.0 || d.y<=0.0) return 0.0;
    float heading=floor(horizon.w/65536.0);
    float rest=horizon.w-heading*65536.0;
    float kindCode=floor(rest/1024.0);
    float coverage=(rest-kindCode*1024.0)/1023.0;
    float kind=kindCode/63.0;
    float angle=heading*6.283185307179586/256.0;
    vec2 along=vec2(sin(angle),cos(angle));
    vec2 p=vec2(d.x,d.z)/(d.y+0.1);
    vec2 q=vec2(dot(p,along)-ground.w,dot(p,vec2(along.y,-along.x)));
    q=q*vec2(mix(1.6,0.35,kind),mix(1.6,2.8,kind));
    float value=0.0;
    float amplitude=0.5;
    for(int octave=0;octave<5;++octave) {
        value+=amplitude*analyticSkyCloudNoise(q);
        q=q*2.03+vec2(17.0,9.0);
        amplitude*=0.5;
    }
    float edge=mix(0.12,0.3,kind);
    float threshold=mix(0.85,-edge,coverage);
    return smoothstep(threshold,threshold+edge,value)*mix(0.96,0.7,kind)*smoothstep(0.015,0.14,d.y);
}
vec3 analyticSkyRadiance(vec3 direction, vec4 zenith, vec4 horizon,
                        vec4 ground, vec4 sun, vec4 sunDirection) {
    vec3 d=normalize(direction);
    vec3 sky=mix(horizon.rgb,d.y>=0.0 ? zenith.rgb : ground.rgb,abs(d.y));
    float cap=clamp((dot(d,sunDirection.xyz)-sun.w)/(1.0-sun.w),0.0,1.0);
    vec3 clear=sky+sun.rgb*sunDirection.w*cap*cap*(3.0-2.0*cap);
    float cloud=analyticSkyCloudOpacity(d,horizon,ground);
    if(cloud<=0.0) return clear;
    vec3 ambient=mix(horizon.rgb,zenith.rgb,0.5);
    float white=max(max(ambient.x,ambient.y),ambient.z);
    vec3 lit=vec3(white,white,white)+sun.rgb*sunDirection.w*(1.0-sun.w)*max(sunDirection.y,0.0);
    return mix(clear,lit*(1.0-0.3*cloud*cloud),cloud);
}
// Integral Li(w)*f(w,v)*N.w dOmega. 64 GGX visible-normal importance samples
// (p_h=D*G1(v)*v.h/N.v, p_w=p_h/(4*v.h)), plus 32 cosine diffuse samples.
// Same single-scatter separable Smith/Schlick/Lambert BRDF as direct base GGX.
// Visible-normal sampling follows PBRT 4e's projected GGX disk construction:
// https://pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory
// Low discrepancy midpoint/radical-inverse samples are fixed, never frame jitter.
// Specular sample weights F*G1(L) are in [0,1] for physical F0.
// This bounded quadrature may miss very small sun caps at high roughness; it is
// not a converged path tracer, geometry bounce, visibility or clearcoat model.
float analyticSkyRadicalInverse(int i) {
    float result=0.0;
    float scale=0.5;
    for(int bit=0;bit<6;++bit) { result+=float(i%2)*scale; i=i/2; scale*=0.5; }
    return result;
}
vec3 analyticSkyBase(vec3 base,float metallic,float roughness,float dielectric,
                     vec3 normal,vec3 toView,vec4 zenith,vec4 horizon,
                     vec4 ground,vec4 sun,vec4 sunDirection) {
    if (dot(normal,normal)<=1.0e-12 || dot(toView,toView)<=1.0e-12) return vec3(0.0);
    vec3 n=normalize(normal);
    vec3 v=normalize(toView);
    float nv=clamp(dot(n,v),0.0,1.0);
    if(nv<=0.0) return vec3(0.0);
    vec3 tangent=normalize(cross(n,abs(n.z)<0.999 ? vec3(0.0,0.0,1.0) : vec3(1.0,0.0,0.0)));
    vec3 bitangent=cross(n,tangent);
    float alpha=max(roughness*roughness,0.0025);
    float a2=alpha*alpha;
    vec3 f0=mix(vec3(dielectric),base,metallic);
    vec3 result=vec3(0.0);
    vec3 stretched=normalize(vec3(alpha*dot(v,tangent),alpha*dot(v,bitangent),nv));
    vec3 t1=stretched.z<0.99999 ? normalize(vec3(-stretched.y,stretched.x,0.0)) : vec3(1.0,0.0,0.0);
    vec3 t2=cross(stretched,t1);
    for(int i=0;i<64;++i) {
        float u=(float(i)+0.5)/64.0;
        float phi=6.283185307179586*analyticSkyRadicalInverse(i);
        vec2 disk=vec2(cos(phi),sin(phi))*sqrt(u);
        disk.y=mix(sqrt(max(0.0,1.0-disk.x*disk.x)),disk.y,(1.0+stretched.z)*0.5);
        vec3 projected=t1*disk.x+t2*disk.y+stretched*sqrt(max(0.0,1.0-dot(disk,disk)));
        vec3 localH=normalize(vec3(alpha*projected.x,alpha*projected.y,max(0.000001,projected.z)));
        vec3 h=tangent*localH.x+bitangent*localH.y+n*localH.z;
        // A sample can land on the view itself (sample 3 does at n.v near
        // 0.62), and float rounding then puts v.h a few ulps above one. pow()
        // of the negative 1-v.h is undefined and returns NaN on NVIDIA GL.
        float vh=min(dot(v,h),1.0);
        vec3 l=2.0*vh*h-v;
        float nl=clamp(dot(n,l),0.0,1.0);
        if(vh>0.0 && nl>0.0) {
            vec3 fresnel=f0+(1.0-f0)*pow(1.0-vh,5.0);
            float g1l=2.0*nl/(nl+sqrt(a2+(1.0-a2)*nl*nl));
            result+=analyticSkyRadiance(l,zenith,horizon,ground,sun,sunDirection)
                *fresnel*g1l/64.0;
        }
    }
    for(int i=0;i<32;++i) {
        float u=(float(i)+0.5)/32.0;
        float phi=6.283185307179586*analyticSkyRadicalInverse(i);
        vec3 l=tangent*(cos(phi)*sqrt(u))+bitangent*(sin(phi)*sqrt(u))+n*sqrt(1.0-u);
        float vh=clamp(dot(v,normalize(l+v)),0.0,1.0);
        vec3 fresnel=f0+(1.0-f0)*pow(1.0-vh,5.0);
        result+=analyticSkyRadiance(l,zenith,horizon,ground,sun,sunDirection)
            *(1.0-fresnel)*(1.0-metallic)*base/32.0;
    }
    return result;
}
)shader";
    if(metal)
    {
        for(const auto& replacement : {std::pair<const char*,const char*>{"vec3","float3"},
                                      {"vec4","float4"}, {"vec2","float2"}})
        {
            surfacesource::replace(source,replacement.first,replacement.second);
        }
    }
    return source;
}
inline std::string shader(bool metal) { return shaderRecipe<std::string>(metal); }

// Binary32 twin of the recipe above, for CPU engines and tests.
inline float cloudFract(float x) { return x-std::floor(x); }
inline float cloudMix(float a,float b,float t) { return a+(b-a)*t; }
inline float cloudSmoothstep(float low,float high,float x)
{
    const auto t=std::clamp((x-low)/(high-low),0.0f,1.0f);
    return t*t*(3-2*t);
}
inline float cloudHash(float x,float y)
{
    float a=cloudFract(x*0.1031f),b=cloudFract(y*0.1031f),c=a;
    const auto shift=a*(b+33.33f)+b*(c+33.33f)+c*(a+33.33f);
    a+=shift; b+=shift; c+=shift;
    return cloudFract((a+b)*c);
}
inline float cloudNoise(float x,float y)
{
    const auto cellX=std::floor(x),cellY=std::floor(y);
    auto fx=x-cellX,fy=y-cellY;
    fx=fx*fx*(3-2*fx); fy=fy*fy*(3-2*fy);
    return cloudMix(cloudMix(cloudHash(cellX,cellY),cloudHash(cellX+1,cellY),fx),
                    cloudMix(cloudHash(cellX,cellY+1),cloudHash(cellX+1,cellY+1),fx),fy);
}
// d is a unit direction. horizonW and groundW are the packed uniform values.
inline float cloudOpacity(float dx,float dy,float dz,float horizonW,float groundW)
{
    if(horizonW<=0 || dy<=0) return 0;
    const auto heading=std::floor(horizonW/65536);
    const auto rest=horizonW-heading*65536;
    const auto kindCode=std::floor(rest/1024);
    const auto coverage=(rest-kindCode*1024)/1023;
    const auto kind=kindCode/63;
    const auto angle=heading*6.283185307179586f/256;
    const auto alongX=std::sin(angle),alongZ=std::cos(angle);
    const auto px=dx/(dy+0.1f),pz=dz/(dy+0.1f);
    auto qx=(px*alongX+pz*alongZ-groundW)*cloudMix(1.6f,0.35f,kind);
    auto qy=(px*alongZ-pz*alongX)*cloudMix(1.6f,2.8f,kind);
    float value=0,amplitude=0.5f;
    for(int octave=0;octave<5;++octave)
    {
        value+=amplitude*cloudNoise(qx,qy);
        qx=qx*2.03f+17; qy=qy*2.03f+9;
        amplitude*=0.5f;
    }
    const auto edge=cloudMix(0.12f,0.3f,kind);
    const auto threshold=cloudMix(0.85f,-edge,coverage);
    return cloudSmoothstep(threshold,threshold+edge,value)*cloudMix(0.96f,0.7f,kind)
        *cloudSmoothstep(0.015f,0.14f,dy);
}
inline std::array<float,3> radiance(const std::array<float,3>& direction,const Uniforms& u)
{
    const auto length=std::sqrt(direction[0]*direction[0]+direction[1]*direction[1]+direction[2]*direction[2]);
    const std::array<float,3> d{direction[0]/length,direction[1]/length,direction[2]/length};
    const auto& sun=u[3];
    const auto& sunDirection=u[4];
    const auto cap=std::clamp((d[0]*sunDirection[0]+d[1]*sunDirection[1]+d[2]*sunDirection[2]-sun[3])
        /(1-sun[3]),0.0f,1.0f);
    const auto cloud=cloudOpacity(d[0],d[1],d[2],u[1][3],u[2][3]);
    const auto white=std::max({(u[1][0]+u[0][0])/2,(u[1][1]+u[0][1])/2,(u[1][2]+u[0][2])/2});
    std::array<float,3> result{};
    for(int c=0;c<3;++c)
    {
        const auto clear=cloudMix(u[1][c],d[1]>=0 ? u[0][c] : u[2][c],std::abs(d[1]))
            +sun[c]*sunDirection[3]*cap*cap*(3-2*cap);
        const auto lit=white+sun[c]*sunDirection[3]*(1-sun[3])*std::max(sunDirection[1],0.0f);
        result[c]=cloud<=0 ? clear : cloudMix(clear,lit*(1-0.3f*cloud*cloud),cloud);
    }
    return result;
}
} // namespace arbitgpu::analyticsky
