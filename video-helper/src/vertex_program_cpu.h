#pragma once

#include "material_program_compiler.h"
#include "../../shared/GeometryCore.h"
#include "../../shared/VertexNoise.h"
#include "../../shared/Visual3DScene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

// The displacement a compiled vertex program applies, run on the host before a
// ray tracer builds its BVH, so the CPU and OptiX devices trace the same moved
// triangles. It follows the OpenGL and Metal vertex shaders: every result is
// clamped to +-1e6, positions move in object space and normals stay as
// authored. Noise is shared/VertexNoise.h, the function Raster's shader text
// spells out. Control reads come from the frame's control bank
// (shared/VertexControlBank.h), the values Raster reads from its texture.
namespace videohelper::vertexprogramcpu
{
using Vec3 = std::array<double, 3>;

struct Inputs final
{
    double timeSeconds = 0;
    std::array<float, 64> spectrum {};
    // Vertex Control Parameter slots, from videowire::vertexcontrol::bank.
    std::vector<float> controls;
};

namespace detail
{
inline double bound (double value) noexcept { return std::clamp (value, -1000000.0, 1000000.0); }
inline Vec3 bound (const Vec3& value) noexcept { return { bound (value[0]), bound (value[1]), bound (value[2]) }; }

// One evaluated value: a scalar in lane 0, or a Vec3.
struct Value final { Vec3 v {}; };

// Per-program state built once per mesh: the Ocean wave tables.
struct Prepared final
{
    std::vector<std::vector<videowire::geometry::WindSpectrumWave>> waves;
};

inline bool prepare (const materialprogram::MaterialProgramRecord& program, Prepared& prepared, std::string& error)
{
    using Operation = videowire::VertexModifierOperation;
    const auto& instructions = program.vertexInstructions();
    prepared.waves.assign (instructions.size(), {});
    for (std::size_t index = 0; index < instructions.size(); ++index)
    {
        const auto& instruction = instructions[index];
        if (instruction.operation != Operation::oceanWaves) continue;
        const auto& p = instruction.parameters;
        videowire::geometry::OceanSettings settings;
        settings.windSpeed = p[0]; settings.headingDegrees = p[1];
        settings.smallestWave = p[2]; settings.size = p[3];
        settings.seed = static_cast<std::uint32_t> (p[4]);
        std::string oceanError;
        if (! videowire::geometry::oceanWaves (settings, prepared.waves[index], oceanError))
        {
            error = "Vertex Ocean Waves refused: " + oceanError;
            return false;
        }
    }
    error.clear();
    return true;
}

inline bool evaluate (const materialprogram::MaterialProgramRecord& program, const Prepared& prepared,
                      const HarmonicMIDI::grid::SceneVertex& vertex, const Inputs& inputs,
                      std::vector<Value>& values, Vec3& output, std::string& error)
{
    using Operation = videowire::VertexModifierOperation;
    const auto& instructions = program.vertexInstructions();
    values.assign (instructions.size(), {});
    for (std::size_t index = 0; index < instructions.size(); ++index)
    {
        const auto& instruction = instructions[index];
        const auto& p = instruction.parameters;
        const auto in = [&] (std::size_t input) -> const Vec3& { return values[instruction.inputs[input]].v; };
        if (instruction.resultSlot != index
            || std::any_of (instruction.inputs.begin(), instruction.inputs.begin() + instruction.inputCount,
                            [&] (std::uint32_t slot) { return slot >= index; }))
        {
            error = "Vertex program instruction order is invalid";
            return false;
        }
        Vec3 r {};
        switch (instruction.operation)
        {
            case Operation::importedPosition: r = { vertex.position.x, vertex.position.y, vertex.position.z }; break;
            case Operation::importedNormal: r = { vertex.normal.x, vertex.normal.y, vertex.normal.z }; break;
            case Operation::importedUv: r = { vertex.uv.x, vertex.uv.y, 0.0 }; break;
            case Operation::importedVertexColor: r = { vertex.color[0], vertex.color[1], vertex.color[2] }; break;
            case Operation::scalarConstant: r[0] = static_cast<float> (p[0]); break;
            case Operation::vec3Constant:
                r = { static_cast<float> (p[0]), static_cast<float> (p[1]), static_cast<float> (p[2]) }; break;
            case Operation::timeSeconds: r[0] = inputs.timeSeconds; break;
            case Operation::audioParameter:
                r[0] = inputs.spectrum[std::min<std::size_t> (static_cast<std::size_t> (p[0]), inputs.spectrum.size() - 1)];
                break;
            case Operation::scalarAdd: r[0] = in (0)[0] + in (1)[0]; break;
            case Operation::scalarSubtract: r[0] = in (0)[0] - in (1)[0]; break;
            case Operation::scalarMultiply: r[0] = in (0)[0] * in (1)[0]; break;
            case Operation::vec3Add: for (int i = 0; i < 3; ++i) r[i] = in (0)[i] + in (1)[i]; break;
            case Operation::vec3Subtract: for (int i = 0; i < 3; ++i) r[i] = in (0)[i] - in (1)[i]; break;
            case Operation::vec3Multiply: for (int i = 0; i < 3; ++i) r[i] = in (0)[i] * in (1)[i]; break;
            case Operation::vec3Scale: for (int i = 0; i < 3; ++i) r[i] = in (0)[i] * in (1)[0]; break;
            case Operation::vec3Compose: r = { in (0)[0], in (1)[0], in (2)[0] }; break;
            case Operation::componentX: r[0] = in (0)[0]; break;
            case Operation::componentY: r[0] = in (0)[1]; break;
            case Operation::componentZ: r[0] = in (0)[2]; break;
            case Operation::scalarRemap:
            case Operation::vec3Remap:
            {
                const auto lanes = instruction.operation == Operation::scalarRemap ? 1 : 3;
                for (int i = 0; i < lanes; ++i)
                {
                    auto t = std::clamp ((in (0)[i] - p[0]) / (p[1] - p[0]), -1000000.0, 1000000.0);
                    if (p[4] != 0.0) t = std::clamp (t, 0.0, 1.0);
                    r[i] = p[2] + t * (p[3] - p[2]);
                }
                break;
            }
            case Operation::boundedDisplacementOutput:
            {
                const auto& d = in (1);
                const auto length = std::sqrt (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                const auto scale = std::min (1.0, p[0] / std::max (length, 0.000001));
                for (int i = 0; i < 3; ++i) r[i] = in (0)[i] + d[i] * scale;
                break;
            }
            case Operation::scalarSin: r[0] = std::sin (in (0)[0]); break;
            case Operation::scalarCos: r[0] = std::cos (in (0)[0]); break;
            case Operation::oceanWaves:
            {
                const auto d = videowire::geometry::oceanDisplacement (prepared.waves[index], in (0)[0], in (0)[1], in (1)[0]);
                r = { d[0] * p[5], d[1] * p[5], d[2] };
                break;
            }
            case Operation::scalarNoise3d:
            case Operation::vec3Noise3d:
            {
                const auto& q = in (0);
                const auto seed = static_cast<std::uint32_t> (p[1]);
                const auto lanes = instruction.operation == Operation::scalarNoise3d ? 1u : 3u;
                for (std::uint32_t i = 0; i < lanes; ++i)
                    r[i] = videowire::vertexnoise::noise (static_cast<float> (q[0]), static_cast<float> (q[1]),
                                                          static_cast<float> (q[2]), static_cast<float> (p[0]), seed + i);
                break;
            }
            case Operation::controlParameter:
            {
                const auto slot = static_cast<std::size_t> (p[0]);
                if (slot >= inputs.controls.size())
                {
                    error = "Vertex Control Parameter slot " + std::to_string (slot) + " has no control bank value";
                    return false;
                }
                r[0] = inputs.controls[slot];
                break;
            }
            case Operation::invalid:
                error = "Vertex program has an invalid instruction";
                return false;
        }
        values[index].v = bound (r);
    }
    if (program.vertexOutputSlot() >= values.size())
    {
        error = "Vertex program has no output";
        return false;
    }
    output = values[program.vertexOutputSlot()].v;
    return true;
}
} // namespace detail

// Moves the object's vertices by the program, in place. The scene stays valid:
// only positions change, each finite and within the clamp.
inline bool displaceObject (HarmonicMIDI::grid::Visual3DScene& scene, HarmonicMIDI::grid::SceneObjectId objectId,
                            const materialprogram::MaterialProgramRecord& program, const Inputs& inputs,
                            std::string& error)
{
    const auto* object = HarmonicMIDI::grid::visual3d_detail::findById (scene.objects, scene.objectCount, objectId);
    if (object == nullptr || ! program.hasVertexProgram())
    {
        error = "Vertex program names no scene object";
        return false;
    }
    if (object->firstVertex > scene.vertexCount || object->vertexCount > scene.vertexCount - object->firstVertex)
    {
        error = "Vertex program object range is outside the scene";
        return false;
    }
    detail::Prepared prepared;
    if (! detail::prepare (program, prepared, error)) return false;
    std::vector<detail::Value> values;
    for (std::uint32_t i = 0; i < object->vertexCount; ++i)
    {
        auto& vertex = scene.vertices[object->firstVertex + i];
        Vec3 moved {};
        if (! detail::evaluate (program, prepared, vertex, inputs, values, moved, error)) return false;
        vertex.position = { static_cast<float> (moved[0]), static_cast<float> (moved[1]), static_cast<float> (moved[2]) };
    }
    error.clear();
    return true;
}
} // namespace videohelper::vertexprogramcpu
