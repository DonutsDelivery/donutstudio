#pragma once

#include "VisualEnvironmentImageAssetContract.h"
#include "DiffractionMaterialIR.h"
#include <array>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <vector>

namespace HarmonicMIDI::grid
{

// Shared value contract for admitted 3D scenes. Producers assemble and validate
// a snapshot before publication. Renderers consume that snapshot by const
// reference. It contains no GPU handles, file paths, parser state, or
// backend-specific values.
template <typename Tag>
struct SceneRecordId
{
    std::uint32_t value = 0;

    constexpr bool isValid() const noexcept { return value != 0; }
    constexpr explicit operator bool() const noexcept { return isValid(); }
    constexpr bool operator== (const SceneRecordId& other) const noexcept
    {
        return value == other.value;
    }
    constexpr bool operator!= (const SceneRecordId& other) const noexcept
    {
        return ! (*this == other);
    }
};

struct Scene3DIdTag {};
struct SceneObjectIdTag {};
struct SceneMaterialIdTag {};
struct SceneTextureIdTag {};
struct SceneLightIdTag {};
struct SceneCameraIdTag {};

using Scene3DId = SceneRecordId<Scene3DIdTag>;
using SceneObjectId = SceneRecordId<SceneObjectIdTag>;
using SceneMaterialId = SceneRecordId<SceneMaterialIdTag>;
using SceneTextureId = SceneRecordId<SceneTextureIdTag>;
using SceneLightId = SceneRecordId<SceneLightIdTag>;
using SceneCameraId = SceneRecordId<SceneCameraIdTag>;

struct SceneVec2
{
    float x = 0.0f;
    float y = 0.0f;
};

struct SceneVec3
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

// Components use the glTF x, y, z, w order. Identity is 0, 0, 0, 1.
struct SceneQuaternion
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

// Local transforms apply scale, then rotation, then translation. Parent object
// transforms follow in child-to-root order. The coordinate system is
// right-handed with +Y up. Cameras look down local -Z.
struct SceneTransform3D
{
    SceneVec3 translation {};
    SceneQuaternion rotation {};
    SceneVec3 scale { 1.0f, 1.0f, 1.0f };
};

struct SceneVertex
{
    SceneVec3 position {};
    SceneVec3 normal {};
    // Tangent.xyz is the unit tangent and tangent.w is the bitangent sign.
    std::array<float, 4> tangent { 1.0f, 0.0f, 0.0f, 1.0f };
    SceneVec2 uv {};
    std::array<float, 4> color { 1.0f, 1.0f, 1.0f, 1.0f };
    // A roughness and how much of it replaces the material's: 0 keeps the
    // material's roughness and 1 uses this one. Raster blends it per vertex
    // and Path Trace at each hit, so a mesh attribute such as Ocean foam can
    // roughen part of a surface.
    std::array<float, 2> roughness { 0.0f, 0.0f };
};

struct SceneTexelRgba8
{
    std::uint8_t red = 0;
    std::uint8_t green = 0;
    std::uint8_t blue = 0;
    std::uint8_t alpha = 255;
};

// Schema version 1 uses normalized UVs, repeat wrapping, nearest sampling, and
// row-major texels with (0, 0) at the top left.
struct SceneTextureRecord
{
    SceneTextureId id {};
    std::uint32_t firstTexel = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t magFilter = 9729;
    std::uint32_t minFilter = 9987;
    std::uint32_t wrapS = 10497;
    std::uint32_t wrapT = 10497;
};

enum class SceneAlphaMode : std::uint8_t
{
    Opaque = 0,
    Mask = 1,
    Blend = 2
};

enum class SceneVolumeField : std::uint8_t
{
    Uniform = 0,
    Noise = 1
};

// Participating media inside a mesh or filling the world. Density 0 is no
// medium, which every released scene carries. Scattering colour is the
// single-scatter albedo, absorption colour tints what the medium does not
// scatter, and emission is radiance per unit length. A Noise field scales the
// density by fBm at scale, detail octaves and offset. A volume-only material
// draws no surface at its mesh boundary. Radius bounds the world volume as a
// ball around the scene; 0 picks four scene radii. Materials ignore it.
struct SceneVolume
{
    float density = 0.0f;
    SceneVec3 color { 0.5f, 0.5f, 0.5f };
    SceneVec3 absorptionColor {};
    float anisotropy = 0.0f;
    SceneVec3 emission {};
    SceneVolumeField field = SceneVolumeField::Uniform;
    std::uint32_t fieldDetail = 3;
    float fieldScale = 1.0f;
    SceneVec3 fieldOffset {};
    bool volumeOnly = false;
    float radius = 0.0f;
};

inline bool sceneVolumeActive(const SceneVolume& volume) noexcept
{
    return std::isfinite(volume.density) && volume.density > 0.0f;
}

inline bool sameSceneVolume(const SceneVolume& a, const SceneVolume& b) noexcept
{
    const auto bits = [](float x, float y) { return std::memcmp(&x, &y, sizeof(float)) == 0; };
    const auto same = [&](SceneVec3 x, SceneVec3 y) { return bits(x.x, y.x) && bits(x.y, y.y) && bits(x.z, y.z); };
    return bits(a.density, b.density) && same(a.color, b.color) && same(a.absorptionColor, b.absorptionColor)
        && bits(a.anisotropy, b.anisotropy) && same(a.emission, b.emission) && a.field == b.field
        && a.fieldDetail == b.fieldDetail && bits(a.fieldScale, b.fieldScale)
        && same(a.fieldOffset, b.fieldOffset) && a.volumeOnly == b.volumeOnly && bits(a.radius, b.radius);
}

struct SceneMaterialRecord
{
    SceneMaterialId id {};
    SceneVec3 baseColor { 1.0f, 1.0f, 1.0f };
    SceneTextureId baseColorTexture {};
    SceneTextureId metallicRoughnessTexture {};
    SceneTextureId normalTexture {};
    SceneTextureId occlusionTexture {};
    SceneTextureId emissiveTexture {};
    float opacity = 1.0f;
    float metallic = 0.0f;
    float roughness = 1.0f;
    SceneVec3 emissive {};
    float normalScale = 1.0f;
    float occlusionStrength = 1.0f;
    SceneAlphaMode alphaMode = SceneAlphaMode::Opaque;
    float alphaCutoff = 0.5f;
    bool doubleSided = false;
    SceneVolume volume {};
};

// Ray visibility, as Blender's object and light Ray Visibility panel has it. A
// cleared bit hides the object or light from that kind of ray. Camera rays leave
// the camera, Shadow rays test a light's reach, and Diffuse, Glossy and
// Transmission rays leave a surface by that kind of scatter. Volume Scatter
// rays leave a scattering medium.
namespace sceneRayVisibility
{
inline constexpr std::uint8_t Camera = 1u;
inline constexpr std::uint8_t Shadow = 2u;
inline constexpr std::uint8_t Diffuse = 4u;
inline constexpr std::uint8_t Glossy = 8u;
inline constexpr std::uint8_t Transmission = 16u;
inline constexpr std::uint8_t VolumeScatter = 32u;
inline constexpr std::uint8_t All = 63u;
} // namespace sceneRayVisibility

struct SceneObjectRecord
{
    SceneObjectId id {};
    SceneObjectId parent {};
    SceneMaterialId material {};
    SceneTransform3D transform {};

    // Indices are local to this object's contiguous vertex range.
    std::uint32_t firstVertex = 0;
    std::uint32_t vertexCount = 0;
    std::uint32_t firstIndex = 0;
    std::uint32_t indexCount = 0;

    // A Shade Mesh operation authored these normals. The path tracer then shades
    // through them fully: curved shadow origins and no geometric side rejection.
    bool authoredShading = false;
    std::uint8_t visibility = sceneRayVisibility::All;
};

enum class SceneLightKind : std::uint8_t
{
    Directional = 0,
    Point = 1,
    Environment = 2,
    Spot = 3,
    // A planar emitter: SceneLightRecord::areaShape, areaSizeX and areaSizeY.
    Area = 4
};

// Square and Disk read areaSizeX for both axes. Rectangle and Ellipse read both.
enum class SceneAreaLightShape : std::uint8_t
{
    Square = 0,
    Rectangle = 1,
    Disk = 2,
    Ellipse = 3
};

// Version zero retains the released constant environment fill. Version one is
// an infinite analytic radiance field; it has no image or geometry bounce. Its
// colours are authored, or baked from an atmosphere by the Physical sky model
// (PhysicalSky.h). RGB radiances use the scene's linear working color units.
struct SceneAnalyticSky
{
    std::uint32_t version = 0;
    SceneVec3 zenith {0.12f, 0.3f, 0.7f};
    SceneVec3 horizon {0.6f, 0.7f, 0.9f};
    SceneVec3 ground {0.03f, 0.025f, 0.02f};
    SceneVec3 sunColor {1.0f, 0.9f, 0.7f};
    float sunIntensity = 10.0f;
    float sunAzimuthDegrees = 0.0f;
    float sunElevationDegrees = 35.0f;
    float sunRadiusDegrees = 2.0f;
    float orientationDegrees = 0.0f;
    // Procedural cloud layer. Coverage 0 draws no clouds. Type runs from puffy
    // (0) to streaky (1). Drift moves the layer along the heading, and animating it
    // gives motion that a seek replays exactly.
    float cloudCoverage = 0.0f;
    float cloudType = 0.0f;
    float cloudDrift = 0.0f;
    float cloudHeadingDegrees = 0.0f;
    // Forward Mie scattering around the sun, from 0 (none) to 1. It adds a
    // glow that peaks at the sun, falls off with a Henyey-Greenstein lobe and is
    // strongest towards the horizon, so the sun side of the sky is brighter.
    float sunHalo = 0.0f;
    // Depth of a volumetric cloud layer, as a share of its height, from 0 (the
    // flat layer) to 1. The camera sees it raymarched with sun self-shadowing.
    float cloudThickness = 0.0f;
};

inline bool sameAnalyticSky(const SceneAnalyticSky& a, const SceneAnalyticSky& b) noexcept
{
    const auto bits = [](float x,float y) {return std::memcmp(&x,&y,sizeof(float))==0;};
    const auto same = [&](SceneVec3 x, SceneVec3 y) { return bits(x.x,y.x) && bits(x.y,y.y) && bits(x.z,y.z); };
    return a.version==b.version && same(a.zenith,b.zenith) && same(a.horizon,b.horizon)
        && same(a.ground,b.ground) && same(a.sunColor,b.sunColor)
        && bits(a.sunIntensity,b.sunIntensity) && bits(a.sunAzimuthDegrees,b.sunAzimuthDegrees)
        && bits(a.sunElevationDegrees,b.sunElevationDegrees) && bits(a.sunRadiusDegrees,b.sunRadiusDegrees)
        && bits(a.orientationDegrees,b.orientationDegrees)
        && bits(a.cloudCoverage,b.cloudCoverage) && bits(a.cloudType,b.cloudType)
        && bits(a.cloudDrift,b.cloudDrift) && bits(a.cloudHeadingDegrees,b.cloudHeadingDegrees)
        && bits(a.sunHalo,b.sunHalo) && bits(a.cloudThickness,b.cloudThickness);
}
// The sun halo and the cloud thickness are written only when one is set, so
// skies without them keep their bytes.
inline bool analyticSkyExtrasAuthored(const SceneAnalyticSky& sky) noexcept
{
    const float none = 0.0f;
    return std::memcmp(&sky.sunHalo,&none,sizeof(float))!=0
        || std::memcmp(&sky.cloudThickness,&none,sizeof(float))!=0;
}

// Clear skies keep their released wire bytes because this stays false for them.
inline bool analyticSkyCloudsAuthored(const SceneAnalyticSky& sky) noexcept
{
    const SceneAnalyticSky clear;
    const auto bits = [](float x,float y) {return std::memcmp(&x,&y,sizeof(float))==0;};
    return !bits(sky.cloudCoverage,clear.cloudCoverage) || !bits(sky.cloudType,clear.cloudType)
        || !bits(sky.cloudDrift,clear.cloudDrift) || !bits(sky.cloudHeadingDegrees,clear.cloudHeadingDegrees);
}

// Exact asset identity is serialized with the scene. Decoded pixels and native
// resources stay with the helper, never in this authored value.
struct SceneEnvironmentImage
{
    std::uint32_t version = 0;
    visualenvironmentimage::Identity asset;
    float orientationDegrees = 0.0f;
    float exposureStops = 0.0f;
};
inline bool sameEnvironmentImage(const SceneEnvironmentImage& a, const SceneEnvironmentImage& b) noexcept
{
    return a.version == b.version && a.asset.id == b.asset.id
        && a.asset.version == b.asset.version && a.asset.contentSha256 == b.asset.contentSha256
        && std::memcmp(&a.orientationDegrees, &b.orientationDegrees, sizeof(float)) == 0
        && std::memcmp(&a.exposureStops, &b.exposureStops, sizeof(float)) == 0;
}

// Version zero is the released RGB light. Version one carries raw incident
// radiance, independent of color and intensity, on the fixed production grid.
struct SceneSpectralSource
{
    std::uint32_t version = 0;
    std::array<float, diffractionmaterial::kMaximumSpectralSamples> wavelengthsNanometres
        = diffractionmaterial::kProductionWavelengthsNanometres;
    std::array<float, diffractionmaterial::kMaximumSpectralSamples> radiance {};
    float traceMaximumDistance = 0.0f;
    diffractionmaterial::DirectionalSpectralIrradiance directionalIrradiance {};
};

inline bool sameDirectionalIrradiance(const diffractionmaterial::DirectionalSpectralIrradiance& a,
                                      const diffractionmaterial::DirectionalSpectralIrradiance& b) noexcept
{
    return a.version == b.version && a.measure == b.measure
        && std::memcmp(a.coefficients.data(), b.coefficients.data(), sizeof(a.coefficients)) == 0;
}
inline bool validDirectionalIrradiance(const diffractionmaterial::DirectionalSpectralIrradiance& source) noexcept
{
    using namespace diffractionmaterial;
    if (source.version == 0) return sameDirectionalIrradiance(source, DirectionalSpectralIrradiance{});
    if (source.version != 1 || source.measure != DirectionalSpectralMeasure::IrradianceCoefficient) return false;
    for (float value : source.coefficients)
        if (!std::isfinite(value) || value < 0 || value > 16) return false;
    return true;
}

inline bool sameSpectralSource(const SceneSpectralSource& a, const SceneSpectralSource& b) noexcept
{
    return a.version == b.version
        && sameDirectionalIrradiance(a.directionalIrradiance, b.directionalIrradiance)
        && std::memcmp(a.wavelengthsNanometres.data(), b.wavelengthsNanometres.data(),
                       sizeof(float) * a.wavelengthsNanometres.size()) == 0
        && std::memcmp(a.radiance.data(), b.radiance.data(), sizeof(float) * a.radiance.size()) == 0
        && std::memcmp(&a.traceMaximumDistance, &b.traceMaximumDistance, sizeof(float)) == 0;
}

// What camera rays that miss geometry see when an Environment Light hides the
// world from the camera, apart from the world's lighting and reflections.
// World keeps the world (or black when hidden), Colour shows an opaque colour
// and Transparent leaves the pixels with zero alpha for compositing.
enum class SceneCameraBackgroundMode : std::uint32_t
{
    World = 0,
    Colour = 1,
    Transparent = 2
};

struct SceneCameraBackground
{
    SceneCameraBackgroundMode mode = SceneCameraBackgroundMode::World;
    SceneVec3 color {0.0f, 0.0f, 0.0f};  // scene-linear, Colour only
};

inline bool sameCameraBackground(const SceneCameraBackground& a, const SceneCameraBackground& b) noexcept
{
    const auto bits = [](float x,float y) {return std::memcmp(&x,&y,sizeof(float))==0;};
    return a.mode==b.mode && bits(a.color.x,b.color.x) && bits(a.color.y,b.color.y) && bits(a.color.z,b.color.z);
}

struct SceneLightRecord
{
    SceneLightId id {};
    SceneLightKind kind = SceneLightKind::Directional;
    SceneTransform3D transform {};
    SceneVec3 color { 1.0f, 1.0f, 1.0f };
    float intensity = 1.0f;
    float range = 0.0f;
    float innerConeAngle = 0.0f;
    float outerConeAngle = 0.7853981634f;
    SceneAnalyticSky sky {};
    // Environment lights only. False leaves the environment in lighting,
    // reflection and refraction, but camera rays that miss geometry see black,
    // or the camera background below.
    bool environmentCameraVisible = true;
    SceneCameraBackground cameraBackground {};
    SceneEnvironmentImage environmentImage {};
    SceneSpectralSource spectralSource {};
    // Light size, as Cycles has it. 0 keeps a delta light.
    // Point and Spot: sphere radius in scene units.
    float radius = 0.0f;
    // Directional: angular radius of the sun disk in radians, half of Cycles' Angle.
    float angularRadius = 0.0f;
    // Area: full extent along local X and Y. The light emits along local -Z.
    float areaSizeX = 1.0f;
    float areaSizeY = 1.0f;
    // Area: full spread angle in radians, pi for a Lambertian emitter.
    float spread = 3.14159265358979f;
    SceneAreaLightShape areaShape = SceneAreaLightShape::Square;
    // Environment lights keep environmentCameraVisible and leave this at All.
    std::uint8_t visibility = sceneRayVisibility::All;
};

// The camera background of the first environment light that sets one: what
// camera rays that miss geometry see in place of the hidden world.
inline SceneCameraBackground sceneCameraBackground(const SceneLightRecord* lights, std::size_t count) noexcept
{
    for (std::size_t i = 0; i < count; ++i)
        if (lights[i].kind == SceneLightKind::Environment
            && lights[i].cameraBackground.mode != SceneCameraBackgroundMode::World)
            return lights[i].cameraBackground;
    return {};
}

// The X and Y extents an area light emits over: Square and Disk use X twice.
inline SceneVec2 sceneAreaLightExtent(const SceneLightRecord& light) noexcept
{
    const bool square = light.areaShape == SceneAreaLightShape::Square || light.areaShape == SceneAreaLightShape::Disk;
    return { light.areaSizeX, square ? light.areaSizeX : light.areaSizeY };
}

// Spread of an area light, as a softbox grid narrows it: radiance at angle theta
// from the axis is scaled by max(0, tan(s/2) - tan theta) * N. N keeps the
// emitted power that of a Lambertian emitter: the integral of
// (T - tan theta) cos theta over the cone of half angle h, with T = tan h, is
// pi (T sin^2 h - h + sin 2h / 2). Returns N for a full spread s in radians, or
// 0 when s is pi and the emitter stays Lambertian. For small h the closed form
// cancels, so its series 3 / h^3 (1 - 0.4 h^2) stands in.
inline float sceneAreaSpreadNormalization(float spread) noexcept
{
    if (!std::isfinite(spread) || spread >= 3.14159265f * 0.99999f) return 0.0f;
    const double h = 0.5 * std::max(static_cast<double>(spread), 1.0e-4);
    if (h < 0.01) return static_cast<float>(3.0 / (h * h * h) * (1.0 - 0.4 * h * h));
    const double s = std::sin(h);
    return static_cast<float>(1.0 / (std::tan(h) * s * s - h + 0.5 * std::sin(2.0 * h)));
}

// Bit equality of the size, area and visibility fields, for record comparisons.
inline bool sameLightShapeAndVisibility(const SceneLightRecord& a, const SceneLightRecord& b) noexcept
{
    const auto bits = [](float x, float y) { return std::memcmp(&x, &y, sizeof(float)) == 0; };
    return bits(a.radius, b.radius) && bits(a.angularRadius, b.angularRadius)
        && bits(a.areaSizeX, b.areaSizeX) && bits(a.areaSizeY, b.areaSizeY) && bits(a.spread, b.spread)
        && a.areaShape == b.areaShape && a.visibility == b.visibility;
}

// True when a light carries any size, area or visibility field away from its default.
inline bool sceneLightHasShapeOrVisibility(const SceneLightRecord& light) noexcept
{
    SceneLightRecord defaults;
    defaults.kind = light.kind;
    return light.kind == SceneLightKind::Area || !sameLightShapeAndVisibility(light, defaults);
}

// Range checks on the size, area and visibility fields of any light kind.
inline bool validSceneLightShapeFields(const SceneLightRecord& light) noexcept
{
    const auto finite = [](float v) { return std::isfinite(v); };
    return finite (light.radius) && light.radius >= 0.0f
        && finite (light.angularRadius) && light.angularRadius >= 0.0f
        && light.angularRadius < 1.5707963268f
        && finite (light.areaSizeX) && light.areaSizeX > 0.0f
        && finite (light.areaSizeY) && light.areaSizeY > 0.0f
        && finite (light.spread) && light.spread > 0.0f && light.spread <= 3.1415927f
        && static_cast<unsigned> (light.areaShape) <= static_cast<unsigned> (SceneAreaLightShape::Ellipse)
        && (light.visibility & ~sceneRayVisibility::All) == 0
        && (light.kind != SceneLightKind::Environment || light.visibility == sceneRayVisibility::All);
}

// False when an environment light hides its sky or image from camera rays.
// Both the raster backdrop and the path tracer's primary misses read this.
inline bool sceneEnvironmentVisibleToCamera(const SceneLightRecord* lights, std::size_t count) noexcept
{
    for (std::size_t i = 0; i < count; ++i)
        if (lights[i].kind == SceneLightKind::Environment && !lights[i].environmentCameraVisible) return false;
    return true;
}

// Direction points toward the source. Light propagates along rotated local -Z.
// Translation and scale do not change an infinitely distant directional source.
inline SceneVec3 sceneSpectralSourceDirection(const SceneLightRecord& light) noexcept
{
    const auto& q = light.transform.rotation;
    const double norm = double(q.x)*q.x + double(q.y)*q.y + double(q.z)*q.z + double(q.w)*q.w;
    if (!std::isfinite(norm) || norm <= 0.0) return {};
    const double x = 2.0*(double(q.x)*q.z + double(q.w)*q.y)/norm;
    const double y = 2.0*(double(q.y)*q.z - double(q.w)*q.x)/norm;
    const double z = 1.0 - 2.0*(double(q.x)*q.x + double(q.y)*q.y)/norm;
    const double length = std::sqrt(x*x + y*y + z*z);
    if (!std::isfinite(length) || length <= 0.0) return {};
    return {static_cast<float>(x/length), static_cast<float>(y/length), static_cast<float>(z/length)};
}

inline bool validSceneSpectralSource(const SceneLightRecord& light) noexcept
{
    const auto& source = light.spectralSource;
    if (source.version == 0) return sameSpectralSource(source, SceneSpectralSource{});
    if (!validDirectionalIrradiance(source.directionalIrradiance)
        || source.version != 1 || light.kind != SceneLightKind::Directional
        || source.wavelengthsNanometres != diffractionmaterial::kProductionWavelengthsNanometres
        || !std::isfinite(source.traceMaximumDistance)
        || source.traceMaximumDistance < 0.001f || source.traceMaximumDistance > 10000.0f)
        return false;
    for (const auto value : source.radiance)
        if (!std::isfinite(value) || value < 0.0f) return false;
    const auto direction = sceneSpectralSourceDirection(light);
    const float lengthSquared = direction.x*direction.x + direction.y*direction.y + direction.z*direction.z;
    return std::isfinite(lengthSquared) && std::abs(lengthSquared - 1.0f) <= 1.0e-4f;
}

// A background other than World belongs to an environment light that hides the
// world from the camera. Only Colour carries a colour; it is finite and non-negative.
inline bool validSceneCameraBackground(const SceneLightRecord& light) noexcept
{
    const auto& background = light.cameraBackground;
    if (background.mode == SceneCameraBackgroundMode::World)
        return sameCameraBackground(background, SceneCameraBackground{});
    if (light.kind != SceneLightKind::Environment || light.environmentCameraVisible) return false;
    if (background.mode == SceneCameraBackgroundMode::Transparent)
        return sameCameraBackground(background, SceneCameraBackground{SceneCameraBackgroundMode::Transparent, {0.0f, 0.0f, 0.0f}});
    const auto& c = background.color;
    return background.mode == SceneCameraBackgroundMode::Colour
        && std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z) && c.x>=0 && c.y>=0 && c.z>=0;
}

inline bool validSceneEnvironmentImage(const SceneLightRecord& light) noexcept
{
    const auto& image = light.environmentImage;
    if (image.version == 0) return sameEnvironmentImage(image, SceneEnvironmentImage{});
    return image.version == 1 && light.kind == SceneLightKind::Environment && light.sky.version == 0
        && visualmodelasset::validStableAssetId(image.asset.id)
        && image.asset.version > 0
        && image.asset.version <= std::uint64_t(std::numeric_limits<std::int64_t>::max())
        && visualmodelasset::validSha256(image.asset.contentSha256)
        && std::isfinite(image.orientationDegrees) && std::abs(image.orientationDegrees) <= 360
        && std::isfinite(image.exposureStops) && std::abs(image.exposureStops) <= 16
        && std::isfinite(light.color.x) && light.color.x>=0
        && std::isfinite(light.color.y) && light.color.y>=0
        && std::isfinite(light.color.z) && light.color.z>=0
        && std::isfinite(light.intensity) && light.intensity>=0;
}

inline bool validSceneAnalyticSky(const SceneLightRecord& light) noexcept
{
    const auto& sky = light.sky;
    if (sky.version==0) return sameAnalyticSky(sky,SceneAnalyticSky{});
    const auto colorBound = [](SceneVec3 c) {
        return std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z)
            && c.x>=0 && c.y>=0 && c.z>=0;
    };
    if (sky.version > 1 || (sky.version != 0 && light.kind != SceneLightKind::Environment)
        || !colorBound(sky.zenith) || !colorBound(sky.horizon) || !colorBound(sky.ground)
        || !colorBound(sky.sunColor) || !colorBound(light.color)
        || !std::isfinite(light.intensity) || light.intensity<0
        || !std::isfinite(sky.sunIntensity) || sky.sunIntensity<0
        || !std::isfinite(sky.sunAzimuthDegrees) || std::abs(sky.sunAzimuthDegrees)>360
        || !std::isfinite(sky.sunElevationDegrees) || std::abs(sky.sunElevationDegrees)>90
        || !std::isfinite(sky.orientationDegrees) || std::abs(sky.orientationDegrees)>360
        || !std::isfinite(sky.sunRadiusDegrees) || sky.sunRadiusDegrees<0.25f || sky.sunRadiusDegrees>30
        || !(sky.cloudCoverage>=0 && sky.cloudCoverage<=1) || !(sky.cloudType>=0 && sky.cloudType<=1)
        || !std::isfinite(sky.cloudDrift) || !std::isfinite(sky.cloudHeadingDegrees)
        || !(sky.sunHalo>=0 && sky.sunHalo<=1) || !(sky.cloudThickness>=0 && sky.cloudThickness<=1))
        return false;
    // Radiance has no bound here: Path Trace takes it as authored, and Raster clips what RGBA16F
    // can't store.
    return true;
}


// One analytic or image field owns the visible background and base reflection.
// Released all-constant environment sets remain valid. Multiple active fields
// require separate native descriptors and an aggregate radiance budget.
inline bool validSceneAnalyticSkySet(const SceneLightRecord* lights, std::size_t count) noexcept
{
    std::size_t environments=0;
    bool analytic=false;
    for (std::size_t i=0;i<count;++i)
    {
        environments+=lights[i].kind==SceneLightKind::Environment;
        analytic |= lights[i].sky.version!=0 || lights[i].environmentImage.version!=0;
    }
    return !analytic || environments==1;
}

// The narrowest lens the renderers' float projection takes. At 1.0e-6 degrees and a 16384 x 16384
// film, the path tracer's pixel area over a squared distance of the minimum near plane is still a
// normal float with a finite inverse, so light paths splat. Near 5.0e-10 degrees that inverse overflows.
inline constexpr float kMinimumCameraVerticalFovDegrees = 1.0e-6f;

struct SceneCameraRecord
{
    SceneCameraId id {};
    SceneTransform3D transform {};
    float verticalFovRadians = 1.0471975512f;
    float nearPlane = 0.1f;
    float farPlane = 1000.0f;
    // Zero is a perspective camera. Above zero the camera is orthographic and this
    // is the full height of its view in world units; verticalFovRadians is unused.
    float orthographicHeight = 0.0f;

    bool orthographic() const noexcept { return orthographicHeight > 0.0f; }

    // Thin-lens depth of field. fStop 0 is a pinhole: the default, and what
    // imported cameras get. Otherwise the plane focusDistance in front of the
    // camera is sharp, and sceneCameraApertureRadius gives the lens size.
    float fStop = 0.0f;
    float focusDistance = 1.0f;
    // 0 is a round aperture; 3 or more is a polygon with that many blades.
    std::uint8_t apertureBlades = 0;
    float bladeRotationRadians = 0.0f;
    // Aperture width over height: above 1 stretches bokeh sideways.
    float apertureRatio = 1.0f;
    // Focus on an object: 0 keeps focusDistance. n focuses on the world origin
    // of objects[n - 1], which resolveSceneCameraFocus turns into the focus
    // distance each frame as the object or the camera moves.
    std::uint32_t focusObject = 0;
};

inline bool sceneCameraHasLens (const SceneCameraRecord& camera) noexcept { return camera.fStop > 0.0f; }

// Bit equality of the lens fields, for record comparisons.
inline bool sameCameraLens (const SceneCameraRecord& a, const SceneCameraRecord& b) noexcept
{
    const auto bits = [](float x, float y) { return std::memcmp (&x, &y, sizeof (float)) == 0; };
    return bits (a.fStop, b.fStop) && bits (a.focusDistance, b.focusDistance)
        && a.apertureBlades == b.apertureBlades && bits (a.bladeRotationRadians, b.bladeRotationRadians)
        && bits (a.apertureRatio, b.apertureRatio) && a.focusObject == b.focusObject;
}

inline bool sceneCameraLensAuthored (const SceneCameraRecord& camera) noexcept
{
    return ! sameCameraLens (camera, SceneCameraRecord {});
}

// The most objects a focus reference can name, the stepped control's range.
inline constexpr std::uint32_t kMaximumCameraFocusObject = 65536;

// An orthographic camera has no lens: its rays run parallel, so it keeps fStop 0.
inline bool validSceneCameraLens (const SceneCameraRecord& camera) noexcept
{
    return std::isfinite (camera.fStop) && camera.fStop >= 0.0f
        && (camera.fStop == 0.0f || ! camera.orthographic())
        && std::isfinite (camera.focusDistance) && camera.focusDistance > 0.0f
        && (camera.apertureBlades == 0 || (camera.apertureBlades >= 3 && camera.apertureBlades <= 32))
        && std::isfinite (camera.bladeRotationRadians)
        && std::isfinite (camera.apertureRatio) && camera.apertureRatio >= 0.01f && camera.apertureRatio <= 100.0f
        && camera.focusObject <= kMaximumCameraFocusObject;
}

// Focal length in scene units (metres) of a lens whose 24 mm tall sensor sees
// verticalFovRadians, the full-frame convention.
inline float sceneCameraFocalLength (const SceneCameraRecord& camera) noexcept
{
    return 0.012f / std::tan (0.5f * camera.verticalFovRadians);
}

// Radius of the entrance pupil, focal length / (2 f-stop), or 0 for a pinhole.
inline float sceneCameraApertureRadius (const SceneCameraRecord& camera) noexcept
{
    return sceneCameraHasLens (camera) ? sceneCameraFocalLength (camera) / (2.0f * camera.fStop) : 0.0f;
}

// Diameter, in scene units on the focus plane, of the blur disc a point at
// view depth sees: aperture diameter * |depth - focus| / depth.
inline float sceneCameraCircleOfConfusion (const SceneCameraRecord& camera, float depth) noexcept
{
    if (! sceneCameraHasLens (camera) || ! (depth > 0.0f)) return 0.0f;
    return 2.0f * sceneCameraApertureRadius (camera) * std::fabs (depth - camera.focusDistance) / depth;
}

// Owned record storage that grows on request. Default construction allocates
// DefaultLength records, a starting length and not a ceiling, so
// `scene.lights = {}` still resets to that length. allocateVisual3DSceneBuffers
// grows every buffer and reports a real allocation failure.
template <typename Record, std::size_t DefaultLength>
class SceneBuffer
{
public:
    using value_type = Record;
    using size_type = std::size_t;
    using iterator = typename std::vector<Record>::iterator;
    using const_iterator = typename std::vector<Record>::const_iterator;
    static constexpr std::size_t kDefaultLength = DefaultLength;

    SceneBuffer() : records (DefaultLength) {}
    explicit SceneBuffer (std::size_t length) : records (length) {}
    template <std::size_t Length>
    SceneBuffer& operator= (const std::array<Record, Length>& values)
    {
        records.assign (values.begin(), values.end());
        return *this;
    }

    std::size_t size() const noexcept { return records.size(); }
    bool empty() const noexcept { return records.empty(); }
    Record* data() noexcept { return records.data(); }
    const Record* data() const noexcept { return records.data(); }
    Record& operator[] (std::size_t index) noexcept { return records[index]; }
    const Record& operator[] (std::size_t index) const noexcept { return records[index]; }
    Record& front() noexcept { return records.front(); }
    const Record& front() const noexcept { return records.front(); }
    Record& back() noexcept { return records.back(); }
    const Record& back() const noexcept { return records.back(); }
    iterator begin() noexcept { return records.begin(); }
    iterator end() noexcept { return records.end(); }
    const_iterator begin() const noexcept { return records.begin(); }
    const_iterator end() const noexcept { return records.end(); }
    const_iterator cbegin() const noexcept { return records.cbegin(); }
    const_iterator cend() const noexcept { return records.cend(); }
    void fill (const Record& value) { std::fill (records.begin(), records.end(), value); }
    // Keeps the first min(size, length) records and default-fills the rest.
    void resize (std::size_t length) { records.resize (length); }

    // Equal when both buffers hold the same length and the same records, the
    // whole-storage equality the fixed arrays had. Needs Record::operator==.
    friend bool operator== (const SceneBuffer& a, const SceneBuffer& b) { return a.records == b.records; }
    friend bool operator!= (const SceneBuffer& a, const SceneBuffer& b) { return a.records != b.records; }

private:
    std::vector<Record> records;
};

struct Visual3DScene
{
    static constexpr std::uint32_t kSchemaVersion = 1;
    // Starting length of each buffer. A scene holds any number of records: only
    // a real allocation failure stops it. Nothing may read these as a ceiling.
    static constexpr std::size_t kInitialVertices = 4096;
    static constexpr std::size_t kInitialIndices = 12288;
    static constexpr std::size_t kInitialObjects = 64;
    static constexpr std::size_t kInitialMaterials = 32;
    static constexpr std::size_t kInitialTextures = 16;
    static constexpr std::size_t kInitialLights = 16;
    static constexpr std::size_t kInitialCameras = 8;

    std::uint32_t schemaVersion = kSchemaVersion;
    Scene3DId id {};
    SceneCameraId activeCamera {};
    SceneVec3 ambientColor {};
    // Scene fog. Path Trace renders it as a medium; Realtime maps it to Depth Fog.
    SceneVolume worldVolume {};
    // Smooth-glass camera walk visits per channel. Zero keeps the default walk.
    std::uint32_t smoothGlassCameraVisits = 0;

    SceneBuffer<SceneVertex, kInitialVertices> vertices {};
    SceneBuffer<std::uint32_t, kInitialIndices> indices {};
    SceneBuffer<SceneObjectRecord, kInitialObjects> objects {};
    SceneBuffer<SceneMaterialRecord, kInitialMaterials> materials {};
    SceneBuffer<SceneTextureRecord, kInitialTextures> textures {};
    // Producers size this owned storage only after validating the aggregate
    // count. Scene copies keep value semantics without a multi-MiB stack array.
    std::vector<SceneTexelRgba8> textureTexels;
    SceneBuffer<SceneLightRecord, kInitialLights> lights {};
    SceneBuffer<SceneCameraRecord, kInitialCameras> cameras {};

    std::size_t vertexCount = 0;
    std::size_t indexCount = 0;
    std::size_t objectCount = 0;
    std::size_t materialCount = 0;
    std::size_t textureCount = 0;
    std::size_t textureTexelCount = 0;
    std::size_t lightCount = 0;
    std::size_t cameraCount = 0;

    // At most one object may draw once per instance transform. Each instance
    // applies after the object's own world transform. Consumers that do not
    // read these fields must refuse a scene whose instancedObject is valid.
    static constexpr std::size_t kInitialObjectInstances = 128;
    SceneObjectId instancedObject {};
    SceneBuffer<SceneTransform3D, kInitialObjectInstances> objectInstances {};
    std::size_t objectInstanceCount = 0;
    // Each instance's transform one frame earlier, read only by Motion. The
    // count is zero when the producer has no prior frame. Otherwise it equals
    // objectInstanceCount and entry i belongs to instance i. An instance born
    // this frame repeats its current transform, so its Motion is zero.
    SceneBuffer<SceneTransform3D, kInitialObjectInstances> previousObjectInstances {};
    std::size_t previousObjectInstanceCount = 0;
    // Each object's local transform one frame earlier, read only by Motion.
    // The count is zero when the producer keeps no per-object prior. Otherwise
    // it equals objectCount and entry i belongs to object i. An object with no
    // prior of its own repeats its current transform.
    SceneBuffer<SceneTransform3D, kInitialObjects> previousObjectTransforms {};
    std::size_t previousObjectTransformCount = 0;
    // The active camera one frame earlier, read only by Motion, so a camera
    // move shows in it. Empty when the producer keeps no prior frame; Motion
    // then reads the current camera.
    std::optional<SceneCameraRecord> previousCamera {};
};

enum class Visual3DSceneBackendRequirement : std::uint8_t
{
    NativeGpu = 0
};

struct Visual3DSceneRenderContract
{
    static constexpr bool kAllowsCpuImageFallback = false;
    static constexpr Visual3DSceneBackendRequirement kBackendRequirement
        = Visual3DSceneBackendRequirement::NativeGpu;
};

enum class Visual3DSceneValidationCode : std::uint8_t
{
    Valid = 0,
    UnsupportedSchema,
    MissingSceneIdentity,
    VertexCapacityExceeded,
    IndexCapacityExceeded,
    ObjectCapacityExceeded,
    MaterialCapacityExceeded,
    TextureCapacityExceeded,
    TextureTexelCapacityExceeded,
    LightCapacityExceeded,
    CameraCapacityExceeded,
    InvalidAmbientColor,
    InvalidObjectIdentity,
    DuplicateObjectIdentity,
    InvalidParentObject,
    CyclicObjectHierarchy,
    InvalidObjectTransform,
    InvalidObjectHierarchyTransform,
    InvalidGeometryRange,
    InvalidTriangleIndexCount,
    IndexOutsideObjectVertexRange,
    InvalidVertexValue,
    InvalidMaterialIdentity,
    DuplicateMaterialIdentity,
    InvalidTextureIdentity,
    DuplicateTextureIdentity,
    InvalidTextureRange,
    MissingBaseColorTexture,
    MissingObjectMaterial,
    InvalidMaterialValue,
    InvalidLightIdentity,
    DuplicateLightIdentity,
    InvalidLightKind,
    InvalidLightValue,
    InvalidCameraIdentity,
    DuplicateCameraIdentity,
    InvalidCameraTransform,
    InvalidCameraProjection,
    MissingActiveCamera,
    InvalidObjectInstances
};

struct Visual3DSceneValidation
{
    Visual3DSceneValidationCode code = Visual3DSceneValidationCode::Valid;
    std::size_t recordIndex = 0;

    constexpr bool valid() const noexcept
    {
        return code == Visual3DSceneValidationCode::Valid;
    }
};

namespace visual3d_detail
{
inline constexpr float kMinimumTransformScale = 0.000001f;
inline constexpr float kMaximumTransformComponent = 1000000.0f;

inline bool finite (float value) noexcept
{
    return std::isfinite (value);
}

inline bool finite (const SceneVec2& value) noexcept
{
    return finite (value.x) && finite (value.y);
}

inline bool finite (const SceneVec3& value) noexcept
{
    return finite (value.x) && finite (value.y) && finite (value.z);
}

inline bool nonNegative (const SceneVec3& value) noexcept
{
    return value.x >= 0.0f && value.y >= 0.0f && value.z >= 0.0f;
}

inline bool validTransform (const SceneTransform3D& value) noexcept
{
    if (! finite (value.translation) || ! finite (value.scale)
        || ! finite (value.rotation.x) || ! finite (value.rotation.y)
        || ! finite (value.rotation.z) || ! finite (value.rotation.w)
        || std::abs (value.translation.x) > kMaximumTransformComponent
        || std::abs (value.translation.y) > kMaximumTransformComponent
        || std::abs (value.translation.z) > kMaximumTransformComponent
        || std::abs (value.scale.x) < kMinimumTransformScale
        || std::abs (value.scale.y) < kMinimumTransformScale
        || std::abs (value.scale.z) < kMinimumTransformScale
        || std::abs (value.scale.x) > kMaximumTransformComponent
        || std::abs (value.scale.y) > kMaximumTransformComponent
        || std::abs (value.scale.z) > kMaximumTransformComponent)
        return false;

    const auto lengthSquared = value.rotation.x * value.rotation.x
                             + value.rotation.y * value.rotation.y
                             + value.rotation.z * value.rotation.z
                             + value.rotation.w * value.rotation.w;
    return finite (lengthSquared) && std::abs (lengthSquared - 1.0f) <= 0.001f;
}

template <typename Records, typename Id>
inline const typename Records::value_type* findById (const Records& records,
                                                     std::size_t count, Id id) noexcept
{
    if (! id.isValid())
        return nullptr;
    for (std::size_t index = 0; index < count; ++index)
        if (records[index].id == id)
            return &records[index];
    return nullptr;
}

template <typename Records>
inline bool duplicateIdAt (const Records& records, std::size_t index) noexcept
{
    for (std::size_t earlier = 0; earlier < index; ++earlier)
        if (records[earlier].id == records[index].id)
            return true;
    return false;
}

inline bool cyclicParentAt (const Visual3DScene& scene, std::size_t objectIndex) noexcept
{
    auto next = scene.objects[objectIndex].parent;
    for (std::size_t depth = 0; next.isValid() && depth <= scene.objectCount; ++depth)
    {
        if (next == scene.objects[objectIndex].id)
            return true;
        const auto* parent = findById (scene.objects, scene.objectCount, next);
        if (parent == nullptr)
            return false;
        next = parent->parent;
    }
    return next.isValid();
}

inline bool hierarchyTransformValidAt (const Visual3DScene& scene,
                                       std::size_t objectIndex) noexcept
{
    using Matrix = std::array<double, 16>;
    const auto matrixFor = [] (const SceneTransform3D& transform) noexcept
    {
        const auto x = static_cast<double> (transform.rotation.x);
        const auto y = static_cast<double> (transform.rotation.y);
        const auto z = static_cast<double> (transform.rotation.z);
        const auto w = static_cast<double> (transform.rotation.w);
        const auto xx = x * x, yy = y * y, zz = z * z;
        const auto xy = x * y, xz = x * z, yz = y * z;
        const auto wx = w * x, wy = w * y, wz = w * z;
        return Matrix {{
            (1.0 - 2.0 * (yy + zz)) * transform.scale.x,
            (2.0 * (xy + wz)) * transform.scale.x,
            (2.0 * (xz - wy)) * transform.scale.x, 0.0,
            (2.0 * (xy - wz)) * transform.scale.y,
            (1.0 - 2.0 * (xx + zz)) * transform.scale.y,
            (2.0 * (yz + wx)) * transform.scale.y, 0.0,
            (2.0 * (xz + wy)) * transform.scale.z,
            (2.0 * (yz - wx)) * transform.scale.z,
            (1.0 - 2.0 * (xx + yy)) * transform.scale.z, 0.0,
            transform.translation.x, transform.translation.y,
            transform.translation.z, 1.0 }};
    };
    const auto multiply = [] (const Matrix& left, const Matrix& right) noexcept
    {
        Matrix result {};
        for (std::size_t column = 0; column < 4; ++column)
            for (std::size_t row = 0; row < 4; ++row)
                for (std::size_t inner = 0; inner < 4; ++inner)
                    result[column * 4 + row] += left[inner * 4 + row]
                                                    * right[column * 4 + inner];
        return result;
    };

    auto world = matrixFor (scene.objects[objectIndex].transform);
    auto parentId = scene.objects[objectIndex].parent;
    for (std::size_t depth = 0; parentId.isValid() && depth < scene.objectCount; ++depth)
    {
        const auto* parent = findById (scene.objects, scene.objectCount, parentId);
        if (parent == nullptr)
            return false;
        world = multiply (matrixFor (parent->transform), world);
        parentId = parent->parent;
    }
    if (parentId.isValid())
        return false;

    const auto lengthSquared = [&world] (std::size_t column) noexcept
    {
        return world[column * 4] * world[column * 4]
             + world[column * 4 + 1] * world[column * 4 + 1]
             + world[column * 4 + 2] * world[column * 4 + 2];
    };
    const std::array<double, 3> scaleSquared {{
        lengthSquared (0), lengthSquared (1), lengthSquared (2) }};
    for (const auto value : scaleSquared)
        if (! std::isfinite (value) || value <= 0.0
            || std::sqrt (value) > kMaximumTransformComponent)
            return false;
    for (std::size_t left = 0; left < 3; ++left)
        for (std::size_t right = left + 1; right < 3; ++right)
        {
            const auto dot = world[left * 4] * world[right * 4]
                           + world[left * 4 + 1] * world[right * 4 + 1]
                           + world[left * 4 + 2] * world[right * 4 + 2];
            if (! std::isfinite (dot)
                || std::abs (dot) > 0.0001 * std::sqrt (scaleSquared[left]
                                                       * scaleSquared[right]))
                return false;
        }
    const auto translationRadius = std::sqrt (
        world[12] * world[12] + world[13] * world[13] + world[14] * world[14]);
    return std::isfinite (translationRadius)
        && translationRadius <= kMaximumTransformComponent;
}
inline bool unitColor (const SceneVec3& value) noexcept
{
    return finite (value) && nonNegative (value) && value.x <= 1.0f && value.y <= 1.0f && value.z <= 1.0f;
}

// Any density 0 or above is a valid medium. Released records carry the defaults.
inline bool validVolume (const SceneVolume& volume) noexcept
{
    return std::isfinite (volume.density) && volume.density >= 0.0f
        && unitColor (volume.color) && unitColor (volume.absorptionColor)
        && std::isfinite (volume.anisotropy) && volume.anisotropy >= -1.0f && volume.anisotropy <= 1.0f
        && finite (volume.emission) && nonNegative (volume.emission)
        && static_cast<unsigned> (volume.field) <= static_cast<unsigned> (SceneVolumeField::Noise)
        && volume.fieldDetail >= 1 && volume.fieldDetail <= 8
        && std::isfinite (volume.fieldScale) && volume.fieldScale > 0.0f
        && finite (volume.fieldOffset)
        && std::isfinite (volume.radius) && volume.radius >= 0.0f;
}
} // namespace visual3d_detail

inline Visual3DSceneValidation validateVisual3DScene (const Visual3DScene& scene) noexcept
{
    using Code = Visual3DSceneValidationCode;
    const auto fail = [] (Code code, std::size_t index = 0) noexcept
    {
        return Visual3DSceneValidation { code, index };
    };

    if (scene.schemaVersion != Visual3DScene::kSchemaVersion)
        return fail (Code::UnsupportedSchema);
    if (! scene.id.isValid())
        return fail (Code::MissingSceneIdentity);
    if (! visual3d_detail::validVolume (scene.worldVolume) || scene.worldVolume.volumeOnly)
        return fail (Code::InvalidMaterialValue);
    if (scene.vertexCount > scene.vertices.size())
        return fail (Code::VertexCapacityExceeded);
    if (scene.indexCount > scene.indices.size())
        return fail (Code::IndexCapacityExceeded);
    if (scene.objectCount > scene.objects.size())
        return fail (Code::ObjectCapacityExceeded);
    if (scene.materialCount > scene.materials.size())
        return fail (Code::MaterialCapacityExceeded);
    if (scene.textureCount > scene.textures.size())
        return fail (Code::TextureCapacityExceeded);
    if (scene.textureTexelCount != scene.textureTexels.size())
        return fail (Code::TextureTexelCapacityExceeded);
    if (scene.lightCount > scene.lights.size())
        return fail (Code::LightCapacityExceeded);
    if (scene.cameraCount > scene.cameras.size())
        return fail (Code::CameraCapacityExceeded);
    if (! visual3d_detail::finite (scene.ambientColor)
        || ! visual3d_detail::nonNegative (scene.ambientColor))
        return fail (Code::InvalidAmbientColor);

    for (std::size_t index = 0; index < scene.textureCount; ++index)
    {
        const auto& texture = scene.textures[index];
        if (! texture.id.isValid())
            return fail (Code::InvalidTextureIdentity, index);
        if (visual3d_detail::duplicateIdAt (scene.textures, index))
            return fail (Code::DuplicateTextureIdentity, index);
        const auto texelCount = static_cast<std::uint64_t> (texture.width)
                              * static_cast<std::uint64_t> (texture.height);
        if (texture.width == 0 || texture.height == 0
            || texture.firstTexel > scene.textureTexelCount
            || texelCount > scene.textureTexelCount - texture.firstTexel)
            return fail (Code::InvalidTextureRange, index);
    }

    for (std::size_t index = 0; index < scene.materialCount; ++index)
    {
        const auto& material = scene.materials[index];
        if (! material.id.isValid())
            return fail (Code::InvalidMaterialIdentity, index);
        if (visual3d_detail::duplicateIdAt (scene.materials, index))
            return fail (Code::DuplicateMaterialIdentity, index);
        if (material.baseColorTexture.isValid()
            && visual3d_detail::findById (scene.textures, scene.textureCount,
                                         material.baseColorTexture) == nullptr)
            return fail (Code::MissingBaseColorTexture, index);
        const std::array<SceneTextureId, 4> auxiliaryTextures {{
            material.metallicRoughnessTexture, material.normalTexture,
            material.occlusionTexture, material.emissiveTexture
        }};
        for (const auto texture : auxiliaryTextures)
            if (texture.isValid()
                && visual3d_detail::findById (scene.textures, scene.textureCount, texture) == nullptr)
                return fail (Code::MissingBaseColorTexture, index);
        if (! visual3d_detail::finite (material.baseColor)
            || ! visual3d_detail::finite (material.emissive)
            || ! visual3d_detail::nonNegative (material.baseColor)
            || ! visual3d_detail::nonNegative (material.emissive)
            || ! visual3d_detail::finite (material.opacity)
            || ! visual3d_detail::finite (material.metallic)
            || ! visual3d_detail::finite (material.roughness)
            || material.opacity < 0.0f || material.opacity > 1.0f
            || material.metallic < 0.0f || material.metallic > 1.0f
            || material.roughness < 0.0f || material.roughness > 1.0f
            || ! std::isfinite (material.normalScale)
            || ! std::isfinite (material.occlusionStrength)
            || material.occlusionStrength < 0.0f || material.occlusionStrength > 1.0f
            || ! std::isfinite (material.alphaCutoff)
            || material.alphaCutoff < 0.0f || material.alphaCutoff > 1.0f
            || static_cast<unsigned> (material.alphaMode)
                > static_cast<unsigned> (SceneAlphaMode::Blend)
            || ! visual3d_detail::validVolume (material.volume))
            return fail (Code::InvalidMaterialValue, index);
    }

    for (std::size_t index = 0; index < scene.objectCount; ++index)
    {
        const auto& object = scene.objects[index];
        if (! object.id.isValid())
            return fail (Code::InvalidObjectIdentity, index);
        if (visual3d_detail::duplicateIdAt (scene.objects, index))
            return fail (Code::DuplicateObjectIdentity, index);
        if ((object.visibility & ~sceneRayVisibility::All) != 0)
            return fail (Code::InvalidObjectIdentity, index);
    }

    if (scene.previousObjectTransformCount != 0
        && (scene.previousObjectTransformCount != scene.objectCount
            || scene.previousObjectTransformCount > scene.previousObjectTransforms.size()))
        return fail (Code::InvalidObjectTransform);

    for (std::size_t index = 0; index < scene.objectCount; ++index)
    {
        const auto& object = scene.objects[index];
        if (object.parent.isValid()
            && visual3d_detail::findById (scene.objects, scene.objectCount,
                                         object.parent) == nullptr)
            return fail (Code::InvalidParentObject, index);
        if (visual3d_detail::cyclicParentAt (scene, index))
            return fail (Code::CyclicObjectHierarchy, index);
        if (! visual3d_detail::validTransform (object.transform)
            || (scene.previousObjectTransformCount != 0
                && ! visual3d_detail::validTransform (scene.previousObjectTransforms[index])))
            return fail (Code::InvalidObjectTransform, index);
        if (! visual3d_detail::hierarchyTransformValidAt (scene, index))
            return fail (Code::InvalidObjectHierarchyTransform, index);
        const bool transformOnly = object.vertexCount == 0 && object.indexCount == 0;
        if ((! transformOnly && object.vertexCount == 0)
            || object.firstVertex > scene.vertexCount
            || object.vertexCount > scene.vertexCount - object.firstVertex
            || object.firstIndex > scene.indexCount
            || object.indexCount > scene.indexCount - object.firstIndex)
            return fail (Code::InvalidGeometryRange, index);
        if (! transformOnly && (object.indexCount == 0 || object.indexCount % 3 != 0))
            return fail (Code::InvalidTriangleIndexCount, index);
        if (! transformOnly
            && visual3d_detail::findById (scene.materials, scene.materialCount,
                                         object.material) == nullptr)
            return fail (Code::MissingObjectMaterial, index);
        for (std::size_t item = object.firstIndex;
             item < object.firstIndex + object.indexCount; ++item)
            if (scene.indices[item] >= object.vertexCount)
                return fail (Code::IndexOutsideObjectVertexRange, index);
    }

    for (std::size_t index = 0; index < scene.vertexCount; ++index)
    {
        const auto& vertex = scene.vertices[index];
        const auto normalLengthSquared = vertex.normal.x * vertex.normal.x
                                       + vertex.normal.y * vertex.normal.y
                                       + vertex.normal.z * vertex.normal.z;
        const auto finiteArray = [] (const auto& values) noexcept
        {
            for (const auto value : values)
                if (! std::isfinite (value))
                    return false;
            return true;
        };
        if (! visual3d_detail::finite (vertex.position)
            || ! visual3d_detail::finite (vertex.normal)
            || ! visual3d_detail::finite (vertex.uv)
            || ! finiteArray (vertex.tangent) || ! finiteArray (vertex.color)
            || ! finiteArray (vertex.roughness)
            || ! visual3d_detail::finite (normalLengthSquared)
            || normalLengthSquared <= 0.000001f)
            return fail (Code::InvalidVertexValue, index);
    }

    for (std::size_t index = 0; index < scene.lightCount; ++index)
    {
        const auto& light = scene.lights[index];
        if (! light.id.isValid())
            return fail (Code::InvalidLightIdentity, index);
        if (visual3d_detail::duplicateIdAt (scene.lights, index))
            return fail (Code::DuplicateLightIdentity, index);
        if (light.kind != SceneLightKind::Directional
            && light.kind != SceneLightKind::Point
            && light.kind != SceneLightKind::Environment
            && light.kind != SceneLightKind::Spot
            && light.kind != SceneLightKind::Area)
            return fail (Code::InvalidLightKind, index);
        if (! validSceneLightShapeFields (light))
            return fail (Code::InvalidLightValue, index);
        if (! visual3d_detail::validTransform (light.transform)
            || ! visual3d_detail::finite (light.color)
            || ! visual3d_detail::nonNegative (light.color)
            || ! visual3d_detail::finite (light.intensity)
            || ! visual3d_detail::finite (light.range)
            || ! visual3d_detail::finite (light.innerConeAngle)
            || ! visual3d_detail::finite (light.outerConeAngle)
            || light.intensity < 0.0f || light.range < 0.0f
            || (light.kind == SceneLightKind::Spot
                && (light.innerConeAngle < 0.0f
                    || light.innerConeAngle >= light.outerConeAngle
                    || light.outerConeAngle > 1.5707963268f)))
            return fail (Code::InvalidLightValue, index);
        if (!validSceneAnalyticSky(light) || !validSceneEnvironmentImage(light) || !validSceneSpectralSource(light)) return fail(Code::InvalidLightValue,index);
        if (!light.environmentCameraVisible && light.kind != SceneLightKind::Environment)
            return fail(Code::InvalidLightValue,index);
        if (!validSceneCameraBackground(light)) return fail(Code::InvalidLightValue,index);
    }
    if (!validSceneAnalyticSkySet(scene.lights.data(),scene.lightCount))
        return fail(Code::InvalidLightValue);

    for (std::size_t index = 0; index < scene.cameraCount; ++index)
    {
        const auto& camera = scene.cameras[index];
        if (! camera.id.isValid())
            return fail (Code::InvalidCameraIdentity, index);
        if (visual3d_detail::duplicateIdAt (scene.cameras, index))
            return fail (Code::DuplicateCameraIdentity, index);
        if (! visual3d_detail::validTransform (camera.transform)
            || camera.transform.scale.x != 1.0f
            || camera.transform.scale.y != 1.0f
            || camera.transform.scale.z != 1.0f)
            return fail (Code::InvalidCameraTransform, index);
        if (! visual3d_detail::finite (camera.verticalFovRadians)
            || ! visual3d_detail::finite (camera.nearPlane)
            || ! visual3d_detail::finite (camera.farPlane)
            || camera.verticalFovRadians <= 0.0f
            || camera.verticalFovRadians >= 3.1415926535f
            || camera.nearPlane <= 0.0f
            || camera.farPlane <= camera.nearPlane
            || ! visual3d_detail::finite (camera.orthographicHeight)
            || camera.orthographicHeight < 0.0f
            || ! validSceneCameraLens (camera))
            return fail (Code::InvalidCameraProjection, index);
    }

    if (scene.previousCamera.has_value())
    {
        const auto& camera = *scene.previousCamera;
        if (! visual3d_detail::validTransform (camera.transform)
            || camera.transform.scale.x != 1.0f
            || camera.transform.scale.y != 1.0f
            || camera.transform.scale.z != 1.0f)
            return fail (Code::InvalidCameraTransform, scene.cameraCount);
        if (! visual3d_detail::finite (camera.verticalFovRadians)
            || ! visual3d_detail::finite (camera.nearPlane)
            || ! visual3d_detail::finite (camera.farPlane)
            || camera.verticalFovRadians <= 0.0f
            || camera.verticalFovRadians >= 3.1415926535f
            || camera.nearPlane <= 0.0f
            || camera.farPlane <= camera.nearPlane)
            return fail (Code::InvalidCameraProjection, scene.cameraCount);
    }

    if (visual3d_detail::findById (scene.cameras, scene.cameraCount,
                                   scene.activeCamera) == nullptr)
        return fail (Code::MissingActiveCamera);

    if (! scene.instancedObject.isValid())
    {
        if (scene.objectInstanceCount != 0 || scene.previousObjectInstanceCount != 0)
            return fail (Code::InvalidObjectInstances);
    }
    else
    {
        const auto* instanced = visual3d_detail::findById (
            scene.objects, scene.objectCount, scene.instancedObject);
        if (instanced == nullptr || instanced->indexCount == 0
            || scene.objectInstanceCount == 0
            || scene.objectInstanceCount > scene.objectInstances.size())
            return fail (Code::InvalidObjectInstances);
        for (std::size_t index = 0; index < scene.objectCount; ++index)
            if (scene.objects[index].parent == scene.instancedObject)
                return fail (Code::InvalidObjectInstances, index);
        if (scene.previousObjectInstanceCount != 0
            && (scene.previousObjectInstanceCount != scene.objectInstanceCount
                || scene.previousObjectInstanceCount > scene.previousObjectInstances.size()))
            return fail (Code::InvalidObjectInstances);
        const auto validInstance = [] (const SceneTransform3D& instance)
        {
            return visual3d_detail::validTransform (instance)
                && instance.scale.x > 0.0f && instance.scale.y > 0.0f
                && instance.scale.z > 0.0f;
        };
        for (std::size_t index = 0; index < scene.objectInstanceCount; ++index)
            if (! validInstance (scene.objectInstances[index])
                || (scene.previousObjectInstanceCount != 0
                    && ! validInstance (scene.previousObjectInstances[index])))
                return fail (Code::InvalidObjectInstances, index);
    }

    return {};
}

namespace visual3d_detail
{
inline SceneVec3 rotateByQuaternion (const SceneQuaternion& q, const SceneVec3& v) noexcept
{
    const SceneVec3 u { q.y * v.z - q.z * v.y, q.z * v.x - q.x * v.z, q.x * v.y - q.y * v.x };
    const SceneVec3 w { q.y * u.z - q.z * u.y, q.z * u.x - q.x * u.z, q.x * u.y - q.y * u.x };
    return { v.x + 2.0f * (q.w * u.x + w.x), v.y + 2.0f * (q.w * u.y + w.y), v.z + 2.0f * (q.w * u.z + w.z) };
}
}

// World position of objects[index]'s origin: its transform's translation
// carried through every parent's scale, rotation and translation.
inline SceneVec3 sceneObjectWorldOrigin (const Visual3DScene& scene, std::size_t index) noexcept
{
    if (index >= scene.objectCount) return {};
    const SceneObjectRecord* object = &scene.objects[index];
    SceneVec3 point = object->transform.translation;
    for (std::size_t depth = 0; object->parent.isValid() && depth < scene.objectCount; ++depth)
    {
        object = visual3d_detail::findById (scene.objects, scene.objectCount, object->parent);
        if (object == nullptr) break;
        const auto& t = object->transform;
        const SceneVec3 scaled { point.x * t.scale.x, point.y * t.scale.y, point.z * t.scale.z };
        const SceneVec3 turned = visual3d_detail::rotateByQuaternion (t.rotation, scaled);
        point = { turned.x + t.translation.x, turned.y + t.translation.y, turned.z + t.translation.z };
    }
    return point;
}

// A camera that focuses on an object gets the view depth of that object's
// origin as its focus distance, at least the near plane so an object behind
// or level with the lens keeps a valid focus. Renderers call this on the
// camera they draw with, after any runtime camera motion, every frame.
inline void resolveSceneCameraFocus (const Visual3DScene& scene, SceneCameraRecord& camera) noexcept
{
    if (camera.focusObject == 0 || camera.focusObject > scene.objectCount) return;
    const SceneVec3 target = sceneObjectWorldOrigin (scene, camera.focusObject - 1);
    const SceneVec3 forward = visual3d_detail::rotateByQuaternion (camera.transform.rotation, { 0.0f, 0.0f, -1.0f });
    const SceneVec3 offset { target.x - camera.transform.translation.x, target.y - camera.transform.translation.y,
                             target.z - camera.transform.translation.z };
    const float depth = offset.x * forward.x + offset.y * forward.y + offset.z * forward.z;
    camera.focusDistance = std::isfinite (depth) ? std::max (depth, camera.nearPlane) : camera.nearPlane;
}


// Record counts held by one scene allocation. Each buffer is allocated at its
// full length, so the device cost follows the capacity, not the live counts.
struct Visual3DSceneCapacity
{
    std::size_t vertices = 0;
    std::size_t indices = 0;
    std::size_t objects = 0;
    std::size_t materials = 0;
    std::size_t textures = 0;
    std::size_t textureTexels = 0;
    std::size_t lights = 0;
    std::size_t cameras = 0;
    std::size_t objectInstances = 0;
};

// What a default-constructed scene allocates. Texels start empty.
inline constexpr Visual3DSceneCapacity kVisual3DSceneDefaultCapacity {
    Visual3DScene::kInitialVertices, Visual3DScene::kInitialIndices, Visual3DScene::kInitialObjects,
    Visual3DScene::kInitialMaterials, Visual3DScene::kInitialTextures, 0,
    Visual3DScene::kInitialLights, Visual3DScene::kInitialCameras, Visual3DScene::kInitialObjectInstances };

enum class Visual3DSceneBuffer : std::uint8_t
{
    Vertices = 0,
    Indices,
    Objects,
    Materials,
    Textures,
    TextureTexels,
    Lights,
    Cameras,
    ObjectInstances,
    Count
};

inline const char* visual3DSceneBufferName (Visual3DSceneBuffer buffer) noexcept
{
    switch (buffer)
    {
        case Visual3DSceneBuffer::Vertices: return "vertices";
        case Visual3DSceneBuffer::Indices: return "indices";
        case Visual3DSceneBuffer::Objects: return "objects";
        case Visual3DSceneBuffer::Materials: return "materials";
        case Visual3DSceneBuffer::Textures: return "textures";
        case Visual3DSceneBuffer::TextureTexels: return "texture texels";
        case Visual3DSceneBuffer::Lights: return "lights";
        case Visual3DSceneBuffer::Cameras: return "cameras";
        case Visual3DSceneBuffer::ObjectInstances: return "object instances";
        case Visual3DSceneBuffer::Count: break;
    }
    return "unknown";
}

inline constexpr std::size_t kVisual3DSceneBufferCount
    = static_cast<std::size_t> (Visual3DSceneBuffer::Count);

inline std::array<std::size_t, kVisual3DSceneBufferCount>
    visual3DSceneCapacityRecords (const Visual3DSceneCapacity& capacity) noexcept
{
    return {{ capacity.vertices, capacity.indices, capacity.objects, capacity.materials,
              capacity.textures, capacity.textureTexels, capacity.lights, capacity.cameras,
              capacity.objectInstances }};
}

// Bytes per record of each buffer. Objects carry a prior transform and
// instances a prior instance transform, so both charge two records.
inline constexpr std::array<std::uint64_t, kVisual3DSceneBufferCount> kVisual3DSceneRecordBytes {{
    sizeof (SceneVertex), sizeof (std::uint32_t),
    sizeof (SceneObjectRecord) + sizeof (SceneTransform3D), sizeof (SceneMaterialRecord),
    sizeof (SceneTextureRecord), sizeof (SceneTexelRgba8), sizeof (SceneLightRecord),
    sizeof (SceneCameraRecord), 2 * sizeof (SceneTransform3D) }};

struct Visual3DSceneFootprint
{
    std::array<std::uint64_t, kVisual3DSceneBufferCount> bytes {};
    std::uint64_t total = 0;
    bool overflow = false;
};

inline Visual3DSceneFootprint visual3DSceneFootprint (const Visual3DSceneCapacity& capacity) noexcept
{
    constexpr auto maximum = ~std::uint64_t {};
    Visual3DSceneFootprint result;
    const auto records = visual3DSceneCapacityRecords (capacity);
    for (std::size_t index = 0; index < kVisual3DSceneBufferCount; ++index)
    {
        const std::uint64_t count = records[index];
        const auto unit = kVisual3DSceneRecordBytes[index];
        if (count > maximum / unit)
        {
            result.overflow = true;
            result.bytes[index] = maximum;
            result.total = maximum;
            continue;
        }
        result.bytes[index] = count * unit;
        if (result.bytes[index] > maximum - result.total)
        {
            result.overflow = true;
            result.total = maximum;
        }
        else if (! result.overflow)
            result.total += result.bytes[index];
    }
    return result;
}

enum class Visual3DSceneAllocationCode : std::uint8_t
{
    Allocated = 0,
    AllocationFailed,
    SizeOverflow
};

// The outcome of allocating a scene. A failure names the buffer being allocated,
// its record count and bytes, the scene total and what the OS said, both as
// fields and as one readable message.
struct Visual3DSceneAllocation
{
    Visual3DSceneAllocationCode code = Visual3DSceneAllocationCode::Allocated;
    Visual3DSceneBuffer cause = Visual3DSceneBuffer::Count;
    std::size_t causeRecords = 0;
    std::uint64_t causeBytes = 0;
    std::uint64_t requiredBytes = 0;
    std::string osMessage;
    std::string message;

    bool allocated() const noexcept { return code == Visual3DSceneAllocationCode::Allocated; }
};

namespace visual3d_detail
{
inline std::string byteText (std::uint64_t bytes)
{
    return std::to_string (bytes) + " bytes (" + std::to_string (bytes >> 20) + " MiB)";
}
} // namespace visual3d_detail

// The capacity that holds exactly the scene's live records.
inline Visual3DSceneCapacity visual3DSceneUsedCapacity (const Visual3DScene& scene) noexcept
{
    return { scene.vertexCount, scene.indexCount, scene.objectCount, scene.materialCount,
             scene.textureCount, scene.textureTexelCount, scene.lightCount, scene.cameraCount,
             scene.objectInstanceCount };
}

// The capacity the scene's buffers are allocated at.
inline Visual3DSceneCapacity visual3DSceneAllocatedCapacity (const Visual3DScene& scene) noexcept
{
    return { scene.vertices.size(), scene.indices.size(), scene.objects.size(),
             scene.materials.size(), scene.textures.size(), scene.textureTexels.capacity(),
             scene.lights.size(), scene.cameras.size(), scene.objectInstances.size() };
}

// Resizes every buffer of the scene to the capacity, or to the live count when
// that is larger, so live records are never truncated. Only arithmetic overflow
// and a real allocation failure stop it. After a failure the live records stay
// intact and the buffers allocated before the failing one keep their new length.
inline Visual3DSceneAllocation allocateVisual3DSceneBuffers (Visual3DScene& scene,
                                                             const Visual3DSceneCapacity& capacity)
{
    Visual3DSceneCapacity wanted = capacity;
    const auto used = visual3DSceneUsedCapacity (scene);
    wanted.vertices = std::max (wanted.vertices, used.vertices);
    wanted.indices = std::max (wanted.indices, used.indices);
    wanted.objects = std::max (wanted.objects, used.objects);
    wanted.materials = std::max (wanted.materials, used.materials);
    wanted.textures = std::max (wanted.textures, used.textures);
    wanted.textureTexels = std::max (wanted.textureTexels, used.textureTexels);
    wanted.lights = std::max (wanted.lights, used.lights);
    wanted.cameras = std::max (wanted.cameras, used.cameras);
    wanted.objectInstances = std::max (wanted.objectInstances, used.objectInstances);

    Visual3DSceneAllocation result;
    const auto footprint = visual3DSceneFootprint (wanted);
    const auto records = visual3DSceneCapacityRecords (wanted);
    result.requiredBytes = footprint.total;
    const auto describe = [&result, &records, &footprint] (std::size_t index)
    {
        result.cause = static_cast<Visual3DSceneBuffer> (index);
        result.causeRecords = records[index];
        result.causeBytes = footprint.bytes[index];
        return std::string (visual3DSceneBufferName (result.cause)) + " hold "
            + std::to_string (result.causeRecords) + " records";
    };
    if (footprint.overflow)
    {
        std::size_t largest = 0;
        for (std::size_t index = 1; index < kVisual3DSceneBufferCount; ++index)
            if (footprint.bytes[index] > footprint.bytes[largest])
                largest = index;
        const auto causeText = describe (largest);
        result.code = Visual3DSceneAllocationCode::SizeOverflow;
        result.message = "3D scene size overflows 64 bits; " + causeText + ".";
        return result;
    }
    std::size_t attempting = 0;
    try
    {
        scene.vertices.resize (wanted.vertices);
        attempting = static_cast<std::size_t> (Visual3DSceneBuffer::Indices);
        scene.indices.resize (wanted.indices);
        attempting = static_cast<std::size_t> (Visual3DSceneBuffer::Objects);
        scene.objects.resize (wanted.objects);
        scene.previousObjectTransforms.resize (wanted.objects);
        attempting = static_cast<std::size_t> (Visual3DSceneBuffer::Materials);
        scene.materials.resize (wanted.materials);
        attempting = static_cast<std::size_t> (Visual3DSceneBuffer::Textures);
        scene.textures.resize (wanted.textures);
        attempting = static_cast<std::size_t> (Visual3DSceneBuffer::TextureTexels);
        scene.textureTexels.reserve (wanted.textureTexels);
        attempting = static_cast<std::size_t> (Visual3DSceneBuffer::Lights);
        scene.lights.resize (wanted.lights);
        attempting = static_cast<std::size_t> (Visual3DSceneBuffer::Cameras);
        scene.cameras.resize (wanted.cameras);
        attempting = static_cast<std::size_t> (Visual3DSceneBuffer::ObjectInstances);
        scene.objectInstances.resize (wanted.objectInstances);
        scene.previousObjectInstances.resize (wanted.objectInstances);
    }
    catch (const std::exception& failure) // std::bad_alloc or std::length_error
    {
        const auto causeText = describe (attempting);
        result.code = Visual3DSceneAllocationCode::AllocationFailed;
        result.osMessage = failure.what();
        result.message = "3D scene allocation failed: " + causeText + ", "
            + visual3d_detail::byteText (result.causeBytes) + " of "
            + visual3d_detail::byteText (result.requiredBytes) + " in total. The system said: "
            + result.osMessage + ".";
    }
    return result;
}

// What an append adds to a scene. A zero leaves that buffer alone.
struct Visual3DSceneGrowth
{
    std::size_t vertices = 0, indices = 0, objects = 0, materials = 0, textures = 0, textureTexels = 0,
                lights = 0, cameras = 0, objectInstances = 0;
};

// Makes room for an append. Each buffer that must grow at least doubles, so
// repeated appends do not copy the scene once per record. An addition that
// overflows size_t is reported as overflow. Returns false with the real message.
inline bool growVisual3DScene (Visual3DScene& scene, const Visual3DSceneGrowth& added, std::string& error)
{
    const auto used = visual3DSceneUsedCapacity (scene);
    auto capacity = visual3DSceneAllocatedCapacity (scene);
    bool grows = false, overflow = false;
    const auto room = [&grows, &overflow] (std::size_t& allocated, std::size_t held, std::size_t count)
    {
        if (count > ~std::size_t {} - held)
        {
            overflow = true;
            allocated = ~std::size_t {};
            return;
        }
        if (held + count <= allocated)
            return;
        const auto doubled = allocated > ~std::size_t {} / 2 ? ~std::size_t {} : allocated * 2;
        allocated = std::max (held + count, doubled);
        grows = true;
    };
    room (capacity.vertices, used.vertices, added.vertices);
    room (capacity.indices, used.indices, added.indices);
    room (capacity.objects, used.objects, added.objects);
    room (capacity.materials, used.materials, added.materials);
    room (capacity.textures, used.textures, added.textures);
    room (capacity.textureTexels, used.textureTexels, added.textureTexels);
    room (capacity.lights, used.lights, added.lights);
    room (capacity.cameras, used.cameras, added.cameras);
    room (capacity.objectInstances, used.objectInstances, added.objectInstances);
    if (overflow)
    {
        error = "3D scene append overflows its record counts: "
            + std::to_string (used.vertices) + " vertices + " + std::to_string (added.vertices) + ", "
            + std::to_string (used.indices) + " indices + " + std::to_string (added.indices) + ", "
            + std::to_string (used.objects) + " objects + " + std::to_string (added.objects) + ", "
            + std::to_string (used.materials) + " materials + " + std::to_string (added.materials) + ".";
        return false;
    }
    if (! grows)
        return true;
    const auto allocation = allocateVisual3DSceneBuffers (scene, capacity);
    if (! allocation.allocated())
    {
        error = allocation.message;
        return false;
    }
    return true;
}

} // namespace HarmonicMIDI::grid
