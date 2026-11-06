#include "../../shared/SurfaceMaterialProgramBinding.h"
#include "../../shared/Visual3DSceneComposition.h"

#include <cstdio>
#include <cstdlib>

namespace composition = HarmonicMIDI::grid::scenecomposition;
using Scheme = composition::ImportedLeafIdScheme;

namespace
{
int failures = 0;

void check (bool condition, const char* what)
{
    if (condition) return;
    std::fprintf (stderr, "FAIL: %s\n", what);
    ++failures;
}

// A program whose MaterialId output is one unsigned constant.
surfacematerial::ProgramDescription materialProgram (std::uint32_t literal)
{
    surfacematerial::ProgramDescription program;
    surfacematerial::Operation constant;
    constant.id = 1;
    constant.kind = surfacematerial::OperationKind::UIntConstant;
    constant.unsignedLiteral = literal;
    program.operations.push_back (constant);
    program.outputs[static_cast<std::size_t> (surfacematerial::OutputSemantic::MaterialId)] = 1;
    return program;
}

std::uint32_t literalOf (const surfacematerial::ProgramDescription& program)
{
    return program.operations.front().unsignedLiteral;
}

// Saved Scene Compose nodes keep the 2048 stride, byte for byte.
void narrowUnchanged()
{
    check (composition::encodeImportedLeafId (1, 2, 5u) == 2053u
               && composition::encodeImportedLeafId (1, 2, 5u, Scheme::Narrow) == 2053u
               && ! composition::encodeImportedLeafId (1, 2, 2048u, Scheme::Narrow),
           "the Narrow scheme keeps the 2048 stride");
    check (composition::importedLeafCountFits (2, 2047) && ! composition::importedLeafCountFits (2, 2048),
           "Narrow material counts stop below 2048");
    auto program = materialProgram (23u);
    check (geometrysurfacematerial::remapImportedLeafMaterialId (program, 1, 2) && literalOf (program) == 2048u + 23u,
           "a saved leaf material literal moves by 2048");
}

// New Scene Compose nodes give materials and textures a 2^27 stride.
void wideStride()
{
    constexpr auto stride = composition::kImportedLeafWideIdStride;
    check (composition::importedLeafIdStride (Scheme::Wide) == stride
               && composition::importedLeafIdStride (Scheme::Narrow) == composition::kImportedLeafIdStride,
           "each scheme names its stride");
    check (composition::encodeImportedLeafId (1, 2, 5u, Scheme::Wide) == stride + 5u
               && composition::encodeImportedLeafId (1, 2, 9000u, Scheme::Wide) == stride + 9000u
               && composition::encodeImportedLeafId (7, 8, stride - 1u, Scheme::Wide) == 7u * stride + stride - 1u
               && ! composition::encodeImportedLeafId (1, 2, stride, Scheme::Wide)
               && composition::encodeImportedLeafId (0, 1, 9000u, Scheme::Wide) == 9000u,
           "Wide material IDs use the 2^27 stride and one leaf keeps raw IDs");
    const auto encoded = composition::encodeImportedLeafId (3, 8, 70000u, Scheme::Wide);
    const auto decoded = encoded ? composition::decodeImportedLeafId (*encoded, 8, Scheme::Wide) : std::nullopt;
    check (decoded && decoded->leaf == 3 && decoded->raw == 70000u
               && composition::importedLeafOwnsId (*encoded, 3, 8, Scheme::Wide)
               && ! composition::importedLeafOwnsId (*encoded, 4, 8, Scheme::Wide),
           "a Wide ID decodes to its leaf and raw ID");
    check (encoded && *encoded < 0x80000000u, "every Wide ID stays in the imported half");
    check (composition::importedLeafCountFits (2, 100000, Scheme::Wide)
               && ! composition::importedLeafCountFits (2, stride, Scheme::Wide),
           "Wide material counts reach 2^27 - 1 per leaf");
    auto program = materialProgram (9000u);
    check (geometrysurfacematerial::remapImportedLeafMaterialId (program, 2, 3, Scheme::Wide)
               && literalOf (program) == 2u * stride + 9000u,
           "a new leaf material literal past 2048 moves into its Wide range");
    auto narrow = materialProgram (9000u);
    check (! geometrysurfacematerial::remapImportedLeafMaterialId (narrow, 2, 3, Scheme::Narrow)
               && literalOf (narrow) == 9000u,
           "the same literal is refused under Narrow and left unchanged");
    check (composition::validImportedLeafIdScheme (0) && composition::validImportedLeafIdScheme (1)
               && ! composition::validImportedLeafIdScheme (2),
           "only the two schemes decode");
}
}

int main()
{
    narrowUnchanged();
    wideStride();
    if (failures == 0) std::puts ("imported_leaf_wide_material_id: all passed");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
