#pragma once

#include "../material_program_compiler.h"
#include "fixture_surface_source_admission.h"
#include "../../../shared/GeometryCore.h"
#include "../../../shared/VertexControlBank.h"
#include "../../../shared/VertexNoise.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace arbitgpu
{
template<class String,std::size_t N> inline String fixtureNullVertexModifierRecipe(const char (&source)[N])
{
    String result(source);
    const auto offset=result.find("// ARBIT_VERTEX_MODIFIER");
    if(offset==String::npos) throw std::length_error("Native fixture shader has no vertex modifier insertion point");
    result.replace(offset,sizeof("// ARBIT_VERTEX_MODIFIER")-1,
        "float3 previousSourcePosition = u.previousNoteValidity[noteIndex / 4u][noteIndex % 4u] > 1u\n"
        "    ? in.previousPosition : in.position;\n"
        "float3 modifiedPosition = in.position;\nfloat3 previousModifiedPosition = previousSourcePosition;\n");
    return result;
}
// The prior pose that attribute 12 carries for skinned meshes and retained
// Geometry3D positions. Without one, the prior pose is the current vertex.
inline const char* fixturePreviousSourcePosition (bool metal) noexcept
{
    return metal
        ? "float3 previousSourcePosition = u.previousNoteValidity[noteIndex / 4u][noteIndex % 4u] > 1u\n"
          "    ? in.previousPosition : in.position;\n"
        : "vec3 previousSourcePosition = uUsePreviousDeformation != 0 ? aPreviousPosition : aPosition;\n";
}
// Only admitted typed instructions become native source. No authored shader text
// or file paths enter this compiler, and vertices remain on the GPU. The program
// is emitted twice: once with the current time, spectrum and position, and once
// with the prior frame's time, spectrum and prior deformed position, so Motion
// projects program(prior pose, prior inputs). Normal, UV and color are read from
// the current vertex in both, so callers refuse programs whose prior evaluation
// would need a prior value of those (vertexModifierPriorGeometryRefusal).
//
// A bounded displacement limit is read from a uniform, not written into the source,
// so one source text serves every limit and the GL program cache hits across a sweep.
// Admission refuses a terminal output anywhere but the root, so a program has exactly
// one bounded displacement and it reads entry 0, which fixtureVertexDisplacementLimits
// fills per draw. The rest of the array only keeps the GL and Metal uniform layouts.
inline constexpr std::size_t kVertexDisplacementLimitCapacity = 64;
static_assert (kVertexDisplacementLimitCapacity == 16 * 4,
               "the vertex displacement limit uniform arrays hold sixteen vec4 values");

inline std::array<float, kVertexDisplacementLimitCapacity> fixtureVertexDisplacementLimits (
    const videohelper::materialprogram::MaterialProgramRecord* program)
{
    std::array<float, kVertexDisplacementLimitCapacity> limits {};
    if (program == nullptr) return limits;
    for (const auto& instruction : program->vertexInstructions())
        if (instruction.operation == videowire::VertexModifierOperation::boundedDisplacementOutput)
            limits[0] = static_cast<float> (instruction.parameters[0]);
    return limits;
}

// The OpenGL Ocean Waves wave table, one RGBA32F row per Ocean Waves instruction
// in program order. Texel 0 holds (wave count, choppiness), and each wave takes
// two texels: (kx, ky, amplitude, omega) and (phase, 1 / k). The shader text
// names only the row, so new wave settings upload new texels without a new
// program. Metal binds the same texels as a vertex-stage texture
// (kFixtureVertexWavesMetalTexture).
// The Metal vertex texture index and Sokol view slot of the wave table. The
// fragment stage uses views 0 to 11.
inline constexpr int kFixtureVertexWavesMetalTexture = 0;
inline constexpr int kFixtureVertexWavesMetalView = 12;
// The control bank's Metal vertex texture index and Sokol view slot.
inline constexpr int kFixtureVertexControlsMetalTexture = 1;
inline constexpr int kFixtureVertexControlsMetalView = 13;
struct FixtureVertexWaveTable final
{
    std::uint32_t width = 0, rows = 0;
    std::vector<std::array<float, 4>> texels;
};

inline bool fixtureVertexWaveTable (const videohelper::materialprogram::MaterialProgramRecord* program,
                                    FixtureVertexWaveTable& table, std::string& error)
{
    table = {};
    if (program == nullptr) return true;
    std::vector<std::vector<videowire::geometry::WindSpectrumWave>> rows;
    std::vector<double> choppiness;
    for (const auto& instruction : program->vertexInstructions())
    {
        if (instruction.operation != videowire::VertexModifierOperation::oceanWaves) continue;
        const auto& p = instruction.parameters;
        videowire::geometry::OceanSettings settings;
        settings.windSpeed = p[0]; settings.headingDegrees = p[1];
        settings.smallestWave = p[2]; settings.size = p[3];
        settings.seed = static_cast<std::uint32_t> (p[4]);
        std::string oceanError;
        rows.emplace_back();
        if (! videowire::geometry::oceanWaves (settings, rows.back(), oceanError))
        {
            error = "Native vertex Ocean Waves refused: " + oceanError;
            return false;
        }
        choppiness.push_back (p[5]);
    }
    if (rows.empty()) return true;
    std::size_t widest = 0;
    for (const auto& row : rows) widest = std::max (widest, row.size());
    table.width = static_cast<std::uint32_t> (1 + 2 * widest);
    table.rows = static_cast<std::uint32_t> (rows.size());
    table.texels.assign (static_cast<std::size_t> (table.width) * table.rows, {});
    for (std::size_t row = 0; row < rows.size(); ++row)
    {
        auto* texel = table.texels.data() + row * table.width;
        texel[0] = { static_cast<float> (rows[row].size()), static_cast<float> (choppiness[row]), 0.0f, 0.0f };
        for (std::size_t wave = 0; wave < rows[row].size(); ++wave)
        {
            const auto& w = rows[row][wave];
            const auto k = std::hypot (w.kx, w.ky);
            texel[1 + 2 * wave] = { static_cast<float> (w.kx), static_cast<float> (w.ky),
                                    static_cast<float> (w.amplitude), static_cast<float> (w.omega) };
            texel[2 + 2 * wave] = { static_cast<float> (w.phase), static_cast<float> (k > 0 ? 1.0 / k : 0.0), 0.0f, 0.0f };
        }
    }
    error.clear();
    return true;
}

inline std::string fixtureVertexModifierShader (
    const char* source, const videohelper::materialprogram::MaterialProgramRecord* program,
    videohelper::materialprogram::BackendTarget target, std::string& error)
{
    const bool metal = target == videohelper::materialprogram::BackendTarget::Metal;
    const std::string vector = metal ? "float3" : "vec3";
    const std::string currentPosition = metal ? "in.position" : "aPosition";
    std::string body = std::string (fixturePreviousSourcePosition (metal))
        + vector + " modifiedPosition = " + currentPosition + ";\n"
        + vector + " previousModifiedPosition = previousSourcePosition;\n";
    if (program != nullptr)
    {
        if (program->capabilities().backend.target != target || ! program->hasVertexProgram())
        {
            error = "Native vertex program does not match the selected backend";
            return {};
        }
        const auto number = [metal] (double value)
        {
            std::ostringstream stream;
            stream.imbue (std::locale::classic());
            stream << std::scientific << std::setprecision (9) << static_cast<float> (value);
            if (metal) stream << 'f';
            return stream.str();
        };
        const auto emit = [&] (const std::string& prefix, const std::string& time,
                               const std::string& spectrum, const std::string& position,
                               const std::string& output, const std::string& controlRow)
        {
            const auto value = [&prefix] (std::uint32_t slot) { return prefix + std::to_string (slot); };
            const std::string limitUniform = metal ? "u.vertexDisplacementLimits[0][0]" : "uVertexDisplacementLimits[0][0]";
            std::uint32_t oceanRow = 0;
            for (const auto& instruction : program->vertexInstructions())
            {
                using Operation = videowire::VertexModifierOperation;
                const auto a = value (instruction.inputs[0]);
                const auto b = value (instruction.inputs[1]);
                const auto c = value (instruction.inputs[2]);
                const auto& p = instruction.parameters;
                std::string expression;
                switch (instruction.operation)
                {
                    case Operation::importedPosition: expression = position; break;
                    case Operation::importedNormal: expression = metal ? "in.normal" : "aNormal"; break;
                    case Operation::importedUv:
                        expression = vector + "(" + (metal ? "in.uv" : "aUv") + ", 0.0)"; break;
                    case Operation::importedVertexColor: expression = metal ? "in.color.rgb" : "aColor.rgb"; break;
                    case Operation::scalarConstant: expression = number (p[0]); break;
                    case Operation::vec3Constant:
                        expression = vector + "(" + number (p[0]) + ", " + number (p[1])
                            + ", " + number (p[2]) + ")"; break;
                    case Operation::timeSeconds: expression = time; break;
                    case Operation::audioParameter:
                    {
                        const auto index = static_cast<unsigned> (p[0]);
                        expression = spectrum + "["
                            + std::to_string (index / 4u) + "][" + std::to_string (index % 4u) + "]";
                        break;
                    }
                    case Operation::scalarAdd: case Operation::vec3Add: expression = a + " + " + b; break;
                    case Operation::scalarSubtract: case Operation::vec3Subtract: expression = a + " - " + b; break;
                    case Operation::scalarMultiply: case Operation::vec3Multiply:
                    case Operation::vec3Scale: expression = a + " * " + b; break;
                    case Operation::vec3Compose: expression = vector + "(" + a + ", " + b + ", " + c + ")"; break;
                    case Operation::componentX: expression = a + ".x"; break;
                    case Operation::componentY: expression = a + ".y"; break;
                    case Operation::componentZ: expression = a + ".z"; break;
                    case Operation::scalarRemap: case Operation::vec3Remap:
                        if (! std::isnormal (static_cast<float> (p[1] - p[0])))
                        {
                            error = "Native vertex remap requires a representable nonzero input range";
                            return false;
                        }
                        expression = "clamp((" + a + " - " + number (p[0]) + ") / "
                            + number (p[1] - p[0]) + ", -1000000.0, 1000000.0)";
                        if (p[4] != 0.0) expression = "clamp(" + expression + ", 0.0, 1.0)";
                        expression = number (p[2]) + " + " + expression + " * " + number (p[3] - p[2]);
                        break;
                    case Operation::boundedDisplacementOutput:
                        expression = a + " + " + b + " * min(1.0, " + limitUniform
                            + " / max(length(" + b + "), 0.000001))";
                        break;
                    case Operation::scalarSin: expression = "sin(" + a + ")"; break;
                    case Operation::scalarCos: expression = "cos(" + a + ")"; break;
                    case Operation::oceanWaves:
                    {
                        // The same waves the CPU Ocean sums, read from the per-draw
                        // table (fixtureVertexWaveTable). Validation stays here so a
                        // refused setting names itself before any draw.
                        videowire::geometry::OceanSettings settings;
                        settings.windSpeed = p[0]; settings.headingDegrees = p[1];
                        settings.smallestWave = p[2]; settings.size = p[3];
                        settings.seed = static_cast<std::uint32_t> (p[4]);
                        std::vector<videowire::geometry::WindSpectrumWave> waves;
                        std::string oceanError;
                        if (! videowire::geometry::validOceanSettings (settings, oceanError)
                            || ! videowire::geometry::oceanWaves (settings, waves, oceanError))
                        {
                            error = "Native vertex Ocean Waves refused: " + oceanError;
                            return false;
                        }
                        const auto name = value (instruction.resultSlot) + "Ocean";
                        const auto row = std::to_string (oceanRow++);
                        // Both backends read the per-draw table: OpenGL through
                        // texelFetch, Metal through a vertex-stage texture read.
                        const auto texel = [&] (const std::string& column)
                        {
                            return metal ? "vertexWaves.read(uint2(" + column + ", " + row + "))"
                                         : "texelFetch(uVertexWaves, ivec2(" + column + ", " + row + "), 0)";
                        };
                        const std::string vec4 = metal ? "float4" : "vec4";
                        body += vector + " " + name + " = " + vector + "(0.0);\n{\n"
                            + vec4 + " header = " + texel ("0") + ";\n"
                            "int count = int(header.x + 0.5);\n"
                            "for (int wave = 0; wave < count; ++wave) {\n"
                            + vec4 + " w = " + texel ("1 + 2 * wave") + ";\n"
                            + vec4 + " q = " + texel ("2 + 2 * wave") + ";\n"
                            "float phi = w.x * " + a + ".x + w.y * " + a + ".y - w.w * " + b + " + q.x;\n"
                            "float c = w.z * cos(phi) * q.y * header.y;\n"
                            + name + " += " + vector + "(w.x * c, w.y * c, w.z * sin(phi));\n}\n}\n";
                        expression = name;
                        break;
                    }
                    // Value noise from shared/VertexNoise.h, the same steps Path Trace runs.
                    case Operation::scalarNoise3d:
                    case Operation::vec3Noise3d:
                    {
                        const auto name = value (instruction.resultSlot) + "Noise";
                        const auto seed = static_cast<std::uint32_t> (p[1]);
                        body += videowire::vertexnoise::source (metal, name + "0", a, number (p[0]), seed);
                        if (instruction.operation == Operation::scalarNoise3d) { expression = name + "0"; break; }
                        body += videowire::vertexnoise::source (metal, name + "1", a, number (p[0]), seed + 1)
                            + videowire::vertexnoise::source (metal, name + "2", a, number (p[0]), seed + 2);
                        expression = vector + "(" + name + "0, " + name + "1, " + name + "2)";
                        break;
                    }
                    // The control bank (shared/VertexControlBank.h): slot i is lane
                    // i % 4 of texel i / 4 in the per-draw control texture, row 0
                    // for this frame and row 1 for the prior frame.
                    case Operation::controlParameter:
                    {
                        const auto slot = static_cast<unsigned> (p[0]);
                        const auto column = std::to_string (slot / 4u);
                        expression = (metal ? "vertexControls.read(uint2(" + column + ", " + controlRow + "))"
                                            : "texelFetch(uVertexControls, ivec2(" + column + ", " + controlRow + "), 0)")
                            + "[" + std::to_string (slot % 4u) + "]";
                        break;
                    }
                    default:
                        error = "Native vertex program has an operation Raster cannot write";
                        return false;
                }
                const bool scalar = instruction.resultType == videowire::VertexModifierValueType::scalar;
                body += (scalar ? "float " : vector + " ") + value (instruction.resultSlot)
                    + " = clamp(" + expression + ", -1000000.0, 1000000.0);\n";
            }
            body += output + " = " + value (program->vertexOutputSlot()) + ";\n";
            return true;
        };
        if (! emit ("vertexValue", metal ? "u.vertexTime.x" : "uVertexTime.x",
                    metal ? "u.vertexSpectrum" : "uVertexSpectrum", currentPosition, "modifiedPosition", "0")
            || ! emit ("previousVertexValue", metal ? "u.vertexTime.y" : "uVertexTime.y",
                       metal ? "u.previousVertexSpectrum" : "uPreviousVertexSpectrum",
                       "previousSourcePosition", "previousModifiedPosition", "1"))
            return {};
    }
    std::string result (source);
    // Metal reads the wave table and the control bank as vertex-stage texture
    // arguments, added after the uniforms only when the program reads them.
    for (const auto& [name, texture] : { std::pair<const char*, int> { "vertexWaves", kFixtureVertexWavesMetalTexture },
                                         std::pair<const char*, int> { "vertexControls", kFixtureVertexControlsMetalTexture } })
    {
        if (! metal || body.find (std::string (name) + ".read") == std::string::npos) continue;
        constexpr const char* uniforms = "constant Uniforms& u [[buffer(0)]],";
        const auto at = result.find (uniforms);
        if (at == std::string::npos || result.find (uniforms, at + 1) != std::string::npos)
        {
            error = "Native fixture shader has no unique vertex uniform argument for its vertex tables";
            return {};
        }
        result.insert (at + std::char_traits<char>::length (uniforms),
                       "\n                       texture2d<float, access::read> " + std::string (name) + " [[texture("
                           + std::to_string (texture) + ")]],");
    }
    constexpr const char* marker = "// ARBIT_VERTEX_MODIFIER";
    const auto offset = result.find (marker);
    if (offset == std::string::npos)
    {
        error = "Native fixture shader has no vertex modifier insertion point";
        return {};
    }
    result.replace (offset, std::char_traits<char>::length (marker), body);
    error.clear();
    return result;
}
} // namespace arbitgpu
