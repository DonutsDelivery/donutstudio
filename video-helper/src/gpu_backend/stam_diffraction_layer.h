#pragma once

#include "stam_diffraction.h"
#include "../../../shared/BaseSurfaceReflection.h"
#include "../../../shared/DiffractionLayerIR.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>

// The Surface Material diffraction layer on the GPU.
//
// Sharing. The fixture shaders run the GLSL in stam_diffraction.h itself:
// glslShader() appends kStamDiffractionGlsl and kStamDiffractionEnvironmentGlsl
// unchanged (minus the unused preset constants), and metalShader() derives the
// MSL from the same two strings by fixed text substitutions. The only code this
// header adds is glue: the uniform unpacking, which is generated from the one
// field list in DiffractionLayerIR.h, the surface frame, the environment
// lookup and the composition with the material's own BSDF. The C++ twin below
// composes the same way with basesurfacereflection::direct, the twin of the
// shaders' baseSurfaceDirect. diffraction_layer_material_tests.cpp checks the
// generated text, the packing round trip and the twin.
//
// Composition. For each direct light the shader adds
//   coatSpecular + diffraction + bsdf * (specularScale - 1)
// to the material's direct term, so the total is the Stam composition
//   coatSpecular + diffraction + bsdf * specularScale
// and a layer with no coat, no glitter and strength 0 adds exactly zero. For the
// sky or HDRI light the base sky term is replaced by
//   coatSpecular + diffraction + skyBase * specularScale
// where the orders come from stamDiffractionLayerEnvironment: for each of 24
// stratified wavelengths and each order it solves the grating equation
// backwards for the incident direction and reads the environment there
// (stamEnvironmentRadiance, which samples the HDRI or the analytic sky).
// The stratum offset is a per-pixel hash, so neighbouring pixels sample
// different wavelengths and the spectrum is continuous over an area.
//
// Surface frame. N is the shading normal, the tangent is the mesh UV tangent,
// surfaceCoord is uv * uvScale, and radial grooves circle the UV point
// (radialCentreU, radialCentreV): the shader places position on the tangent
// plane at the UV offset from that centre, so the centre needs no object
// matrix and is the same for instanced draws. Map grooves take their direction
// from grooveAngle until a groove-direction texture is bound.
namespace arbitgpu::stamlayer
{
inline constexpr std::size_t kUniformVec4Count = 7;
using Uniforms = std::array<std::array<float, 4>, kUniformVec4Count>;

// Lanes are vec4 index * 4 + component. Lanes 0 to 2 hold the integer fields,
// lane 3 is the enable flag, and the float fields follow in the order of
// surfacematerial::forEachDiffractionLayerFloat from lane 4.
inline constexpr std::size_t kModeLane = 0;
inline constexpr std::size_t kBaseModeLane = 1;
inline constexpr std::size_t kMaxOrderLane = 2;
inline constexpr std::size_t kEnabledLane = 3;
inline constexpr std::size_t kFirstFloatLane = 4;
static_assert(kFirstFloatLane + surfacematerial::kDiffractionLayerFloatCount <= kUniformVec4Count * 4,
              "Every layer field has a uniform lane");

inline void setLane(Uniforms& u, std::size_t lane, float value) noexcept
{
    u[lane / 4][lane % 4] = value;
}

inline float lane(const Uniforms& u, std::size_t index) noexcept
{
    return u[index / 4][index % 4];
}

// All zero (enable flag 0) when the program has no layer.
inline Uniforms pack(const surfacematerial::DiffractionLayerDescription* layer) noexcept
{
    Uniforms u {};
    if (layer == nullptr) return u;
    setLane(u, kModeLane, static_cast<float>(layer->grooveMode));
    setLane(u, kBaseModeLane, static_cast<float>(layer->baseMode));
    setLane(u, kMaxOrderLane, static_cast<float>(layer->maxOrder));
    setLane(u, kEnabledLane, 1.0f);
    std::size_t index = kFirstFloatLane;
    surfacematerial::forEachDiffractionLayerFloat(*layer, [&u, &index](const char*, float value) {
        setLane(u, index++, value);
    });
    return u;
}

// Uniform lane of a float field by name, or 0 when the name is unknown.
inline std::size_t floatLane(const char* field) noexcept
{
    std::size_t found = 0, index = kFirstFloatLane;
    const surfacematerial::DiffractionLayerDescription defaults {};
    surfacematerial::forEachDiffractionLayerFloat(defaults, [&](const char* name, float) {
            if (found == 0 && std::strcmp(name, field) == 0) found = index;
            ++index;
        });
    return found;
}

// Uniform array spelling: "uStamLayer" in GLSL, "u.stamLayer" in MSL.
inline std::string laneExpression(const std::string& array, std::size_t index)
{
    static constexpr char components[] = "xyzw";
    return array + "[" + std::to_string(index / 4) + "]." + components[index % 4];
}

// StamLayerParams from the uniform array, generated from the field list so the
// packing and the unpacking cannot drift apart.
inline std::string unpackFunction(const std::string& array, const std::string& parameters,
                                  const std::string& intType)
{
    std::string text = "StamLayerParams stamLayerUniformParams(" + parameters + ") {\n    StamLayerParams P;\n";
    text += "    P.mode = " + intType + "(floor(" + laneExpression(array, kModeLane) + " + 0.5));\n";
    text += "    P.baseMode = " + intType + "(floor(" + laneExpression(array, kBaseModeLane) + " + 0.5));\n";
    text += "    P.maxOrder = " + intType + "(floor(" + laneExpression(array, kMaxOrderLane) + " + 0.5));\n";
    std::size_t index = kFirstFloatLane, field = 0;
    const auto stamFields = surfacematerial::kDiffractionLayerFloatCount
                          - surfacematerial::kDiffractionLayerPlacementCount;
    const surfacematerial::DiffractionLayerDescription defaults {};
    surfacematerial::forEachDiffractionLayerFloat(defaults, [&](const char* name, float) {
            if (field++ < stamFields)
                text += std::string("    P.") + name + " = " + laneExpression(array, index) + ";\n";
            ++index;
        });
    return text + "    return P;\n}\n";
}

inline void replaceAll(std::string& text, const std::string& from, const std::string& to)
{
    for (auto at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size()))
        text.replace(at, from.size(), to);
}

// The core GLSL without the preset constants. The layer reads its values from
// uniforms, and the presets live in DiffractionLayerIR.h through the twin's
// constants, which stam_diffraction_tests.cpp checks against these literals.
inline std::string coreWithoutPresets()
{
    std::string core = kStamDiffractionGlsl;
    const auto begin = core.find("const StamLayerParams stamCdPreset");
    const auto end = core.find("float stamCmfLobe(");
    if (begin == std::string::npos || end == std::string::npos || end < begin) return {};
    return core.erase(begin, end - begin);
}

inline std::string surfaceAndCompositionGlsl(const std::string& array)
{
    return "StamSurface stamLayerSurface(vec3 N, vec3 tangent, float bitangentSign, vec2 uv) {\n"
           "    StamSurface S;\n"
           "    vec3 T = stamTangentAxis(N, tangent);\n"
           "    vec3 B = cross(N, T) * (bitangentSign < 0.0 ? -1.0 : 1.0);\n"
           "    vec2 centred = uv - vec2(" + laneExpression(array, floatLane("radialCentreU")) + ", "
           + laneExpression(array, floatLane("radialCentreV")) + ");\n"
           "    float angle = " + laneExpression(array, floatLane("grooveAngle")) + ";\n"
           "    S.N = N;\n"
           "    S.tangent = T;\n"
           "    S.position = T * centred.x + B * centred.y;\n"
           "    S.centre = vec3(0.0);\n"
           "    S.fieldDirection = vec2(cos(angle), sin(angle));\n"
           "    S.surfaceCoord = uv * " + laneExpression(array, floatLane("uvScale")) + ";\n"
           "    return S;\n"
           "}\n"
           "vec3 stamLayerDirectDelta(StamLayerParams P, StamSurface S, vec3 L, vec3 V, vec3 baseSpecular, vec3 bsdf) {\n"
           "    StamLayerResult R = stamDiffractionLayer(P, S, L, V, baseSpecular);\n"
           "    return R.coatSpecular + R.diffraction + bsdf * (R.specularScale - vec3(1.0));\n"
           "}\n"
           "float stamLayerStratum(vec2 fragCoord) {\n"
           "    return stamHashUnit(stamHash(uvec2(fragCoord)));\n"
           "}\n";
}

inline std::string environmentCompositionGlsl()
{
    return "vec3 stamLayerEnvironmentCompose(StamLayerParams P, StamSurface S, vec3 V, vec3 baseSpecular, vec3 skyBase,\n"
           "                                 float stratumOffset) {\n"
           "    StamLayerResult R = stamDiffractionLayerEnvironment(P, S, V, baseSpecular, stratumOffset);\n"
           "    return R.coatSpecular + R.diffraction + skyBase * R.specularScale;\n"
           "}\n";
}

// OpenGL fragment module. Needs uSky and the analytic sky and environment image
// shaders before it; fixtureFragmentShaderSource appends it after them when the
// feature bit is on.
inline std::string glslShader()
{
    std::string text = "\nuniform vec4 uStamLayer[" + std::to_string(kUniformVec4Count) + "];\n";
    text += coreWithoutPresets();
    text += kStamDiffractionEnvironmentGlsl;
    text += "vec3 stamEnvironmentRadiance(vec3 dir) {\n"
            "    return uSky[0].w > 1.5 ? environmentImageRadiance(dir, uSky[0], uSky[1])\n"
            "        : analyticSkyRadiance(dir, uSky[0], uSky[1], uSky[2], uSky[3], uSky[4]);\n"
            "}\n";
    text += unpackFunction("uStamLayer", "", "int");
    text += surfaceAndCompositionGlsl("uStamLayer");
    text += environmentCompositionGlsl();
    return text;
}

// GLSL to MSL for the Stam strings: program-scope constants, vector types,
// rsqrt, explicit float-to-int lane conversion, and out parameters as thread
// references. Every substitution is checked by the tests.
inline std::string translateToMetal(std::string text)
{
    replaceAll(text, "\nconst ", "\nconstant ");
    replaceAll(text, "uvec2", "uint2");
    replaceAll(text, "ivec2(stamCellIndex(q.x), stamCellIndex(q.y))",
               "int2(int(stamCellIndex(q.x)), int(stamCellIndex(q.y)))");
    replaceAll(text, "vec2", "float2");
    replaceAll(text, "vec3", "float3");
    replaceAll(text, "vec4", "float4");
    replaceAll(text, "inversesqrt(", "rsqrt(");
    replaceAll(text, ", out float3 ", ", thread float3& ");
    replaceAll(text, ", out float ", ", thread float& ");
    replaceAll(text, ", out bool ", ", thread bool& ");
    return text;
}

// Metal fragment module, inserted before _main with the other fixture modules.
// It defines FIXTURE_STAM_LAYER, which turns on the layer code in _main. The
// environment functions take the uniforms and the environment texture as extra
// arguments because MSL has no global uniforms.
inline std::string metalShader()
{
    std::string text = "\n#define FIXTURE_STAM_LAYER 1\n";
    text += translateToMetal(coreWithoutPresets());
    text += "float3 stamEnvironmentRadiance(constant Uniforms& u, texture2d<float> environmentTexture, float3 dir) {\n"
            "    return u.sky[0].w > 1.5f ? environmentImageRadiance(dir, u.sky[0], u.sky[1], environmentTexture)\n"
            "        : analyticSkyRadiance(dir, u.sky[0], u.sky[1], u.sky[2], u.sky[3], u.sky[4]);\n"
            "}\n";
    std::string environment = kStamDiffractionEnvironmentGlsl;
    replaceAll(environment, "vec3 stamEnvironmentRadiance(vec3 dir);\n", "");
    replaceAll(environment, "stamEnvironmentRadiance(", "stamEnvironmentRadiance(u, environmentTexture, ");
    replaceAll(environment, "float stratumOffset) {",
               "float stratumOffset, constant Uniforms& u, texture2d<float> environmentTexture) {");
    replaceAll(environment, "P.maxOrder, stratumOffset)", "P.maxOrder, stratumOffset, u, environmentTexture)");
    text += translateToMetal(environment);
    std::string glue = unpackFunction("u.stamLayer", "constant Uniforms& u", "int");
    glue += surfaceAndCompositionGlsl("u.stamLayer");
    replaceAll(glue, "StamSurface stamLayerSurface(vec3 N,",
               "StamSurface stamLayerSurface(constant Uniforms& u, vec3 N,");
    std::string environmentGlue = environmentCompositionGlsl();
    replaceAll(environmentGlue, "float stratumOffset) {",
               "float stratumOffset, constant Uniforms& u, texture2d<float> environmentTexture) {");
    replaceAll(environmentGlue, "baseSpecular, stratumOffset);", "baseSpecular, stratumOffset, u, environmentTexture);");
    text += translateToMetal("\n" + glue + environmentGlue);
    return text;
}

// ---------------------------------------------------------------------------
// C++ twin of the composition, for tests and CPU references.
// ---------------------------------------------------------------------------
namespace twin
{
namespace stam = arbit::stamdiffraction;

inline stam::Vec3 fromArray(const basesurfacereflection::Color& c) noexcept
{
    return stam::Vec3 {c[0], c[1], c[2]};
}

inline basesurfacereflection::Direction toArray(stam::Vec3 v) noexcept
{
    return {v.x, v.y, v.z};
}

// The shader's surface frame for a UV point.
inline stam::Surface surface(const surfacematerial::DiffractionLayerDescription& layer, stam::Vec3 N,
                             stam::Vec3 tangent, double bitangentSign, double u, double v) noexcept
{
    stam::Surface S;
    const auto T = stam::tangentAxis(N, tangent);
    const auto B = stam::cross(N, T) * (bitangentSign < 0.0 ? -1.0 : 1.0);
    const double cu = u - static_cast<double>(layer.radialCentreU);
    const double cv = v - static_cast<double>(layer.radialCentreV);
    S.N = N;
    S.tangent = T;
    S.position = T * cu + B * cv;
    S.centre = stam::Vec3 {};
    S.fieldDirection = stam::Vec2 {std::cos(static_cast<double>(layer.grooveAngle)),
                                   std::sin(static_cast<double>(layer.grooveAngle))};
    S.surfaceCoord = stam::Vec2 {u * static_cast<double>(layer.uvScale), v * static_cast<double>(layer.uvScale)};
    return S;
}

struct Material
{
    basesurfacereflection::Color base {0.5, 0.5, 0.5};
    double metallic = 1.0;
    double roughness = 0.3;
    double dielectric = 0.04;
    double transmission = 0.0;
};

inline stam::Vec3 baseSpecular(const Material& m) noexcept
{
    return stam::Vec3 {m.dielectric * (1.0 - m.metallic) + m.base[0] * m.metallic,
                       m.dielectric * (1.0 - m.metallic) + m.base[1] * m.metallic,
                       m.dielectric * (1.0 - m.metallic) + m.base[2] * m.metallic};
}

// The material's own direct response, as the shader's baseSurfaceDirect.
inline stam::Vec3 baseDirect(const Material& m, stam::Vec3 N, stam::Vec3 L, stam::Vec3 V) noexcept
{
    return fromArray(basesurfacereflection::direct(m.base, m.metallic, m.roughness, m.dielectric, m.transmission,
                                                   toArray(N), toArray(L), toArray(V)));
}

// Direct response of the layered material per unit light radiance: the base
// term plus the shader's stamLayerDirectDelta.
inline stam::Vec3 direct(const surfacematerial::DiffractionLayerDescription* layer, const Material& m,
                         const stam::Surface& S, stam::Vec3 L, stam::Vec3 V) noexcept
{
    const auto bsdf = baseDirect(m, S.N, L, V);
    if (layer == nullptr) return bsdf;
    const auto R = stam::diffractionLayer(surfacematerial::toStamLayerParams(*layer), S, L, V, baseSpecular(m));
    return bsdf + R.coatSpecular + R.diffraction + bsdf * (R.specularScale - stam::Vec3 {1.0, 1.0, 1.0});
}

// Environment response: skyBase is the material's own sky term.
template <class Radiance>
stam::Vec3 environment(const surfacematerial::DiffractionLayerDescription& layer, const Material& m,
                       const stam::Surface& S, stam::Vec3 V, stam::Vec3 skyBase, double stratumOffset,
                       Radiance radiance)
{
    const auto R = stam::diffractionLayerEnvironment(surfacematerial::toStamLayerParams(layer), S, V,
                                                     baseSpecular(m), stratumOffset, radiance);
    return R.coatSpecular + R.diffraction + skyBase * R.specularScale;
}
} // namespace twin
} // namespace arbitgpu::stamlayer
