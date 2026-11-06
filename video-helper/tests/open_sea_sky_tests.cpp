// Open-sea sky: the C++ twin of the cloud recipe, the volumetric cloud layer,
// and the sun glint path.
//
// Render-level cloud checks live in support/analytic_sky_cases.h. These cases
// need no GPU. Thresholds come from evaluating the same formulas offline.

#include "../src/gpu_backend/fixture_analytic_sky.h"
#include "../../shared/GeometryCore.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace
{
int checks=0, failures=0;
void check(bool condition,const std::string& message)
{
    ++checks;
    if (condition) return;
    ++failures;
    std::printf("FAIL: %s\n",message.c_str());
}

constexpr double kDegrees=3.14159265358979323846/180;

HarmonicMIDI::grid::SceneLightRecord sunOverDarkSky(float azimuth,float elevation)
{
    HarmonicMIDI::grid::SceneLightRecord light;
    light.kind=HarmonicMIDI::grid::SceneLightKind::Environment;
    light.color={1,1,1}; light.intensity=1;
    light.sky.version=1;
    light.sky.zenith=light.sky.horizon=light.sky.ground={0,0,0};
    light.sky.sunIntensity=10; light.sky.sunRadiusDegrees=2;
    light.sky.sunAzimuthDegrees=azimuth; light.sky.sunElevationDegrees=elevation;
    return light;
}

void cloudTwin()
{
    namespace sky=arbitgpu::analyticsky;
    auto light=sunOverDarkSky(0,35);
    light.sky.zenith=light.sky.horizon={.1f,.2f,.8f};
    light.sky.ground={.1f,.1f,.1f};
    const auto clear=sky::uniforms(light);
    check(clear[1][3]==0 && clear[2][3]==0,"a clear sky leaves both cloud uniform slots at zero");

    auto parked=light; parked.sky.cloudType=1; parked.sky.cloudDrift=3; parked.sky.cloudHeadingDegrees=40;
    check(sky::uniforms(parked)==clear,"zero coverage reaches the shader as a clear sky");

    const auto share=[&](float coverage,float type,float drift,float heading) {
        auto cloudy=light;
        cloudy.sky.cloudCoverage=coverage; cloudy.sky.cloudType=type;
        cloudy.sky.cloudDrift=drift; cloudy.sky.cloudHeadingDegrees=heading;
        const auto u=sky::uniforms(cloudy);
        double covered=0,samples=0;
        for (int a=0;a<72;++a) for (int e=1;e<=8;++e)
        {
            const auto azimuth=a*5*kDegrees, elevation=(10+e*10)*kDegrees;
            const auto opacity=sky::cloudOpacity(float(std::sin(azimuth)*std::cos(elevation)),float(std::sin(elevation)),
                float(std::cos(azimuth)*std::cos(elevation)),u[1][3],u[2][3]);
            covered+=opacity>.5f ? 1 : 0; samples+=1;
        }
        return covered/samples;
    };
    check(share(0,0,0,0)==0,"coverage 0 covers nothing");
    check(share(.35f,0,0,0)>.05 && share(.35f,0,0,0)<.8,"coverage .35 leaves broken cloud");
    check(share(.2f,0,0,0)<=share(.5f,0,0,0) && share(.5f,0,0,0)<=share(1,0,0,0),"more coverage covers more sky");
    check(share(1,0,0,0)==1,"coverage 1 closes the sky above the horizon");

    auto cloudy=light; cloudy.sky.cloudCoverage=1;
    const auto u=sky::uniforms(cloudy);
    check(sky::cloudOpacity(0,0,1,u[1][3],u[2][3])==0 && sky::cloudOpacity(0,-1,0,u[1][3],u[2][3])==0,
          "clouds stop at the horizon");
    const auto top=sky::radiance({0,1,0},u),open=sky::radiance({0,1,0},clear);
    check(top[0]>open[0]*3,"a closed layer whitens the zenith, red "+std::to_string(top[0]));

    // The drift decides which clouds a fixed direction sees, with no hidden clock.
    auto moved=light; moved.sky.cloudCoverage=.35f; moved.sky.cloudDrift=.5f;
    auto still=moved; still.sky.cloudDrift=0;
    const auto movedU=sky::uniforms(moved),stillU=sky::uniforms(still);
    int differs=0;
    for (int a=0;a<72;++a)
    {
        const auto azimuth=a*5*kDegrees;
        const auto x=float(std::sin(azimuth)*.6),z=float(std::cos(azimuth)*.6);
        differs+=sky::cloudOpacity(x,.8f,z,movedU[1][3],movedU[2][3])!=sky::cloudOpacity(x,.8f,z,stillU[1][3],stillU[2][3]);
    }
    check(differs>10,"cloud drift moves the layer");
    check(sky::cloudOpacity(.6f,.8f,0,movedU[1][3],movedU[2][3])==sky::cloudOpacity(.6f,.8f,0,movedU[1][3],movedU[2][3]),
          "the same drift replays the same clouds");
}

// Glint metric: reflect the camera's ray off wind-spectrum water and read the sky
// twin, over a polar patch of the sea around the camera. glintRatio is the mean
// radiance within 5 degrees of the sun's azimuth over the mean at least 20 degrees
// away. pathRows counts the 300 log-spaced ranges from 15 to 600 m with a lit sample
// inside that wedge, so a long streak scores high and a lone highlight scores low.
// Normals use a .5 m difference, like a triangle of a .5 m grid. Water is F0 .02.
struct Glint final { double on=0, off=0; int pathRows=0; int peakAzimuth=0; };
Glint measureGlint(const std::vector<videowire::geometry::WindSpectrumWave>& waves,float sunAzimuth)
{
    namespace sky=arbitgpu::analyticsky;
    using videowire::geometry::windSpectrumHeight;
    const auto u=sky::uniforms(sunOverDarkSky(sunAzimuth,15));
    constexpr int azimuths=360, ranges=300;
    constexpr double step=.5, cameraHeight=8;
    std::vector<double> profile(azimuths,0.0);
    std::vector<char> pathRow(ranges,0);
    for (int a=0;a<azimuths;++a) for (int r=0;r<ranges;++r)
    {
        const auto azimuthDegrees=-90+a*.5;
        const auto range=15*std::pow(40.0,r/double(ranges-1));
        const auto x=range*std::sin(azimuthDegrees*kDegrees),z=range*std::cos(azimuthDegrees*kDegrees);
        const auto height=windSpectrumHeight(waves,x,z,0);
        const auto slopeX=(windSpectrumHeight(waves,x+step,z,0)-height)/step;
        const auto slopeZ=(windSpectrumHeight(waves,x,z+step,0)-height)/step;
        const auto normalLength=std::sqrt(slopeX*slopeX+slopeZ*slopeZ+1);
        const std::array<double,3> n{-slopeX/normalLength,1/normalLength,-slopeZ/normalLength};
        std::array<double,3> v{-x,cameraHeight-height,-z};
        const auto viewLength=std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
        for (auto& value:v) value/=viewLength;
        const auto vn=v[0]*n[0]+v[1]*n[1]+v[2]*n[2];
        if (vn<=0) continue;
        const std::array<float,3> reflected{float(2*vn*n[0]-v[0]),float(2*vn*n[1]-v[1]),float(2*vn*n[2]-v[2])};
        const auto fresnel=.02+.98*std::pow(1-vn,5.0);
        const auto radiance=sky::radiance(reflected,u)[0]*fresnel;
        profile[a]+=radiance/ranges;
        if (radiance>0 && std::abs(azimuthDegrees-sunAzimuth)<=5) pathRow[r]=1;
    }
    Glint result;
    double onCount=0,offCount=0,peak=-1;
    for (int a=0;a<azimuths;++a)
    {
        const auto azimuthDegrees=-90+a*.5;
        const auto distance=std::abs(azimuthDegrees-sunAzimuth);
        if (distance<=5) { result.on+=profile[a]; onCount+=1; }
        if (distance>=20) { result.off+=profile[a]; offCount+=1; }
        if (profile[a]>peak) { peak=profile[a]; result.peakAzimuth=int(std::lround(azimuthDegrees)); }
    }
    result.on/=onCount; result.off/=offCount;
    for (const auto row:pathRow) result.pathRows+=row;
    return result;
}

// Sun radiance has no authored limit. Only Raster's stored peak stops at what RGBA16F holds.
void sunWithoutLimit()
{
    namespace sky=arbitgpu::analyticsky;
    auto light=sunOverDarkSky(0,35);
    for (const float radiance : {64.5f,2000.0f,1.0e7f})
    {
        light.sky.sunIntensity=radiance;
        check(HarmonicMIDI::grid::validSceneAnalyticSky(light),"a bright sun is a valid sky");
    }
    light.sky.sunIntensity=2000;
    check(sky::uniforms(light)[4][3]==2000.0f,"Raster keeps a sun its target can hold");
    light.sky.sunIntensity=1.0e7f;
    const auto stored=sky::uniforms(light);
    const auto peak=std::max({stored[3][0],stored[3][1],stored[3][2]})*stored[4][3];
    check(peak>59000.0f && peak<=60001.0f,"Raster clips only the peak it stores");
    light.sky.sunIntensity=-1;
    check(!HarmonicMIDI::grid::validSceneAnalyticSky(light),"a negative sun stays invalid");
}

// Sky colours and intensity have no authored limit either. Raster clips each term's stored peak.
void skyWithoutLimit()
{
    namespace sky=arbitgpu::analyticsky;
    auto light=sunOverDarkSky(0,35);
    light.sky.zenith={64.5f,32.25f,0.0f};
    light.sky.sunColor={1000.0f,1000.0f,1000.0f};
    light.intensity=2.0e6f;
    light.color={100.0f,100.0f,100.0f};
    check(HarmonicMIDI::grid::validSceneAnalyticSky(light),"a bright sky is a valid sky");
    const auto stored=sky::uniforms(light);
    check(std::abs(stored[0][0]-60000.0f)<1.0f && std::abs(stored[0][1]-30000.0f)<1.0f,
        "Raster clips a sky term's peak and keeps its hue");
    const auto sun=std::max({stored[3][0],stored[3][1],stored[3][2]})*stored[4][3];
    check(sun>59000.0f && sun<=60001.0f,"Raster still clips the sun's stored peak");
    light.sky.sunIntensity=.001f;
    light.intensity=1.0f; light.color={1,1,1};
    check(std::abs(sky::uniforms(light)[3][0]-1000.0f)<1.0f,"a dim sun keeps a colour above the old bound");
    light.sky.zenith.x=-1;
    check(!HarmonicMIDI::grid::validSceneAnalyticSky(light),"a negative sky stays invalid");
}

void glintPath()
{
    using namespace videowire::geometry;
    std::vector<WindSpectrumWave> sea,flat;
    std::string error;
    check(windSpectrumWaves(10,0,.5,1,250,sea,error),"a moderate sea synthesizes");
    check(windSpectrumWaves(10,0,.0001,1,250,flat,error),"a nearly flat sea synthesizes");
    for (const float azimuth:{0.0f,40.0f,-40.0f})
    {
        const auto label="sun azimuth "+std::to_string(azimuth);
        const auto glint=measureGlint(sea,azimuth);
        check(glint.on>.01,label+": the path carries sun radiance, "+std::to_string(glint.on));
        check(glint.on>10*glint.off,label+": glintRatio is at least 10, on "+std::to_string(glint.on)
              +" off "+std::to_string(glint.off));
        check(std::abs(glint.peakAzimuth-azimuth)<=5,label+": the brightest azimuth follows the sun, "
              +std::to_string(glint.peakAzimuth));
        check(glint.pathRows>=60,label+": the glint spreads into a path, rows "+std::to_string(glint.pathRows));
    }
    check(measureGlint(flat,0).pathRows<measureGlint(sea,0).pathRows/2,
          "wave slopes, not the sun cap alone, stretch the highlight into a path");
}
// The volumetric layer: transmittance stays in [0, 1], scattered light is finite
// and non-negative, thicker clouds pass less, and no thickness keeps the flat sky.
void volumetricClouds()
{
    namespace volume=videohelper::pathtrace::cloudvolume;
    namespace sky=arbitgpu::analyticsky;
    auto light=sunOverDarkSky(30,40);
    light.sky.zenith={.1f,.2f,.5f}; light.sky.horizon={.4f,.5f,.7f};
    light.sky.cloudCoverage=.6f; light.sky.cloudType=.2f;
    const auto flatU=sky::uniforms(light);
    const std::array<float,3> up{.3f,.8f,.2f};
    const auto flat=sky::background(up,flatU), reflected=sky::radiance(up,flatU);
    check(flat==reflected,"no cloud thickness draws the flat layer the reflections see");

    volume::Layer thin; thin.coverage=.6f; thin.kind=.2f; thin.thickness=.1f;
    auto thick=thin; thick.thickness=1;
    double thinSum=0, thickSum=0;
    bool bounded=true;
    int samples=0;
    for(int i=0;i<24;++i) for(int j=1;j<12;++j)
    {
        const double azimuth=i*15*kDegrees, elevation=j*7.5*kDegrees;
        const float dx=float(std::sin(azimuth)*std::cos(elevation)),dy=float(std::sin(elevation)),dz=float(std::cos(azimuth)*std::cos(elevation));
        const auto a=volume::march(thin,dx,dy,dz,0,.6f,.8f,5,5,5,.5f);
        const auto b=volume::march(thick,dx,dy,dz,0,.6f,.8f,5,5,5,.5f);
        for(const auto& r:{a,b})
            bounded=bounded && r.transmittance>=0 && r.transmittance<=1 && std::isfinite(r.r) && r.r>=0
                && std::isfinite(r.g) && r.g>=0 && std::isfinite(r.b) && r.b>=0;
        thinSum+=a.transmittance; thickSum+=b.transmittance; ++samples;
    }
    check(bounded,"cloud transmittance stays in [0, 1] and scattered light stays finite");
    check(thickSum<thinSum && thickSum/samples<.95,"a thicker layer passes less light");
    check(volume::march(thick,0,-1,0,0,.6f,.8f,5,5,5,.5f).transmittance==1,"no clouds below the horizon");
    volume::Layer clear=thick; clear.coverage=0;
    check(volume::march(clear,0,1,0,0,.6f,.8f,5,5,5,.5f).transmittance==1,"no coverage draws no clouds");

    light.sky.cloudThickness=1;
    check(HarmonicMIDI::grid::validSceneAnalyticSky(light),"a thickness of one is valid");
    const auto deepU=sky::uniforms(light);
    check(deepU[0][3]>1 && deepU[0][3]<1.5f,"the thickness packs below the image field's slot value");
    int dimmed=0;
    for(int i=0;i<24;++i)
    {
        const double azimuth=i*15*kDegrees, elevation=40*kDegrees;
        const std::array<float,3> d{float(std::sin(azimuth)*std::cos(elevation)),float(std::sin(elevation)),
            float(std::cos(azimuth)*std::cos(elevation))};
        const auto v=sky::background(d,deepU);
        check(std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]),"the volumetric sky is finite");
        dimmed+=v!=sky::radiance(d,deepU);
    }
    check(dimmed>0,"the camera sees the volumetric layer");
    light.sky.cloudThickness=1.5f;
    check(!HarmonicMIDI::grid::validSceneAnalyticSky(light),"a thickness past one is refused");
}
} // namespace

int main()
{
    cloudTwin();
    volumetricClouds();
    sunWithoutLimit();
    skyWithoutLimit();
    glintPath();
    std::printf("open sea sky: %d checks, %d failures\n",checks,failures);
    return failures==0 ? 0 : 1;
}
