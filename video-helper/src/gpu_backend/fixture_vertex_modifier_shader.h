#pragma once

#include "../material_program_compiler.h"

#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>

namespace arbitgpu
{
// Only admitted typed instructions become native source. No authored shader text
// or file paths enter this compiler, and vertices remain on the GPU.
inline std::string fixtureVertexModifierShader (
    const char* source, const videohelper::materialprogram::MaterialProgramRecord* program,
    videohelper::materialprogram::BackendTarget target, std::string& error)
{
    const bool metal = target == videohelper::materialprogram::BackendTarget::Metal;
    const std::string vector = metal ? "float3" : "vec3";
    const std::string position = metal ? "in.position" : "aPosition";
    std::string body = vector + " modifiedPosition = " + position + ";\n";
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
        const auto value = [] (std::uint16_t slot) { return "vertexValue" + std::to_string (slot); };
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
                case Operation::timeSeconds: expression = metal ? "u.vertexTime.x" : "uVertexTime.x"; break;
                case Operation::audioParameter:
                {
                    const auto index = static_cast<unsigned> (p[0]);
                    expression = std::string (metal ? "u.vertexSpectrum[" : "uVertexSpectrum[")
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
                        return {};
                    }
                    expression = "clamp((" + a + " - " + number (p[0]) + ") / "
                        + number (p[1] - p[0]) + ", -1000000.0, 1000000.0)";
                    if (p[4] != 0.0) expression = "clamp(" + expression + ", 0.0, 1.0)";
                    expression = number (p[2]) + " + " + expression + " * " + number (p[3] - p[2]);
                    break;
                case Operation::boundedDisplacementOutput:
                    expression = a + " + " + b + " * min(1.0, " + number (p[0])
                        + " / max(length(" + b + "), 0.000001))"; break;
                default:
                    error = "Native vertex execution does not support external control banks or noise yet";
                    return {};
            }
            const bool scalar = instruction.resultType == videowire::VertexModifierValueType::scalar;
            body += (scalar ? "float " : vector + " ") + value (instruction.resultSlot)
                + " = clamp(" + expression + ", -1000000.0, 1000000.0);\n";
        }
        body += "modifiedPosition = " + value (program->vertexOutputSlot()) + ";\n";
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
