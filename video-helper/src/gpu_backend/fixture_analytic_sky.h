#pragma once

#include "../../../shared/Visual3DScene.h"
#include "fixture_surface_source_admission.h"
#include <array>
#include <cmath>
#include <string>
#include <utility>

namespace arbitgpu::analyticsky
{
// Five packed vec4s, identical on GL and Metal. Radiance is scene-linear RGB,
// in relative scene units per steradian, not calibrated SI watts. A sky
// color is a radiance, not irradiance. Light RGB/intensity tint the whole field.
using Uniforms = std::array<std::array<float,4>,5>;
inline Uniforms uniforms(const HarmonicMIDI::grid::SceneLightRecord& light)
{
    const auto& s=light.sky;
    constexpr double radians=0.01745329251994329577;
    const auto az=(s.sunAzimuthDegrees+s.orientationDegrees)*radians;
    const auto el=s.sunElevationDegrees*radians;
    const auto color=[&](HarmonicMIDI::grid::SceneVec3 c,float w) {
        return std::array<float,4>{c.x*light.color.x*light.intensity,
            c.y*light.color.y*light.intensity,c.z*light.color.z*light.intensity,w};
    };
    return {color(s.zenith,static_cast<float>(s.version)),color(s.horizon,0),
        color(s.ground,0),color(s.sunColor,static_cast<float>(std::cos(s.sunRadiusDegrees*radians))),
        std::array<float,4>{static_cast<float>(std::sin(az)*std::cos(el)),
            static_cast<float>(std::sin(el)),static_cast<float>(std::cos(az)*std::cos(el)),s.sunIntensity}};
}

// This is an authored infinite radiance field, not an atmospheric simulation.
// Upper/lower hemispheres interpolate by |direction.y|. The sun is a smooth
// spherical cap: peak sunColor*sunIntensity, zero at the authored radius.
// Background and BRDF integration execute this exact common source. Only vector
// spelling changes for MSL; no backend-specific light model or sample sequence.
template<class String> inline String shaderRecipe(bool metal)
{
    String source=R"shader(
vec3 analyticSkyRadiance(vec3 direction, vec4 zenith, vec4 horizon,
                        vec4 ground, vec4 sun, vec4 sunDirection) {
    vec3 d=normalize(direction);
    vec3 sky=mix(horizon.rgb,d.y>=0.0 ? zenith.rgb : ground.rgb,abs(d.y));
    float cap=clamp((dot(d,sunDirection.xyz)-sun.w)/(1.0-sun.w),0.0,1.0);
    return sky+sun.rgb*sunDirection.w*cap*cap*(3.0-2.0*cap);
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
        float vh=dot(v,h);
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
} // namespace arbitgpu::analyticsky
