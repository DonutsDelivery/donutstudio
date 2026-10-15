#pragma once
#include "../../../shared/generated/SurfaceMaterialStarterPrograms.h"

namespace videohelper::tests::smoothglassfixture
{
// SurfaceMaterialGraphLoweringTests compares this exact value to the editable
// Smooth Dispersive Glass starter lowered by the production graph compiler.
inline surfacematerial::ProgramDescription authoredProgram(std::uint32_t materialId = 6,
                                                           float ior = 1.5f, float dispersion = 0.004f)
{
    using namespace surfacematerial;
    auto program = surfacematerialstarterfixture::kPrograms[0].program();
    program.version = kSpectralWireVersion;
    program.transport = TransportMode::ClosedVolume;
    for (auto& operation : program.operations)
    {
        const auto semantic = static_cast<OutputSemantic>(operation.id - 1);
        switch (semantic)
        {
            case OutputSemantic::BaseColor: operation.literal = {1,1,1,0}; break;
            case OutputSemantic::Metallic: case OutputSemantic::Roughness:
            case OutputSemantic::Clearcoat: operation.literal = {}; break;
            case OutputSemantic::Transmission: case OutputSemantic::Opacity: operation.literal = {1,0,0,0}; break;
            case OutputSemantic::Ior: operation.literal = {ior,0,0,0}; break;
            case OutputSemantic::MaterialId: operation.unsignedLiteral = materialId; break;
            default: break;
        }
    }
    Operation cauchy;
    cauchy.id = 11;
    cauchy.kind = OperationKind::FloatConstant;
    cauchy.resultType = ValueType::Scalar;
    cauchy.literal[0] = dispersion;
    program.operations.push_back(cauchy);
    program.outputs[10] = cauchy.id;
    return program;
}
} // namespace videohelper::tests::smoothglassfixture
