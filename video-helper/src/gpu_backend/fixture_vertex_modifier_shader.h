#pragma once

#include "../material_program_compiler.h"
#include "fixture_surface_source_admission.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>

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
// The Nth bounded displacement in instruction order reads entry N of the uniform
// array, which fixtureVertexDisplacementLimits fills per draw. The GL and Metal
// uniform arrays hold sixteen vec4 values.
inline constexpr std::size_t kVertexDisplacementLimitCapacity = 64;
static_assert (kVertexDisplacementLimitCapacity == 16 * 4,
               "the vertex displacement limit uniform arrays hold sixteen vec4 values");

inline std::array<float, kVertexDisplacementLimitCapacity> fixtureVertexDisplacementLimits (
    const videohelper::materialprogram::MaterialProgramRecord* program)
{
    std::array<float, kVertexDisplacementLimitCapacity> limits {};
    if (program == nullptr) return limits;
    std::size_t index = 0;
    for (const auto& instruction : program->vertexInstructions())
        if (instruction.operation == videowire::VertexModifierOperation::boundedDisplacementOutput
            && index < kVertexDisplacementLimitCapacity)
            limits[index++] = static_cast<float> (instruction.parameters[0]);
    return limits;
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
                               const std::string& output)
        {
            const auto value = [&prefix] (std::uint16_t slot) { return prefix + std::to_string (slot); };
            const auto limitUniform = [metal] (std::size_t index)
            {
                return std::string (metal ? "u.vertexDisplacementLimits[" : "uVertexDisplacementLimits[")
                    + std::to_string (index / 4u) + "][" + std::to_string (index % 4u) + "]";
            };
            std::size_t displacementCount = 0;
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
                    {
                        if (displacementCount >= kVertexDisplacementLimitCapacity)
                        {
                            error = "Native vertex program has too many bounded displacement limits";
                            return false;
                        }
                        expression = a + " + " + b + " * min(1.0, " + limitUniform (displacementCount++)
                            + " / max(length(" + b + "), 0.000001))";
                        break;
                    }
                    default:
                        error = "Native vertex execution does not support external control banks or noise yet";
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
                    metal ? "u.vertexSpectrum" : "uVertexSpectrum", currentPosition, "modifiedPosition")
            || ! emit ("previousVertexValue", metal ? "u.vertexTime.y" : "uVertexTime.y",
                       metal ? "u.previousVertexSpectrum" : "uPreviousVertexSpectrum",
                       "previousSourcePosition", "previousModifiedPosition"))
            return {};
    }
    std::string result (source);
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
