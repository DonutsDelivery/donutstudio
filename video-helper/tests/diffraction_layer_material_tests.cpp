#include "gpu_backend/stam_diffraction_layer.h"
#include "surface_material_admission.h"
#include "sha256.h"
#include "GeometrySurfaceMaterial.h"
#include "SurfaceMaterialField.h"
#include "VisualImportedSceneRenderOperationContract.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Tests for the Surface Material diffraction layer: the IR companion
// (DiffractionLayerIR.h), its digest and wire form, the GPU module
// (stam_diffraction_layer.h) and its backend wiring, and the optics of the
// composition through the C++ twin of the shader code.
//
// Angle convention as in stam_diffraction_tests.cpp: N = +z, T = +x, angles
// are signed polar angles from N toward +T. With the viewer on the normal,
// order m of a grating with period d sends wavelength lambda from the light at
// theta_i into the viewer when d * (sin(theta_i) + sin(0)) = m * lambda.

#ifndef BACKEND_OPENGL_PATH
#define BACKEND_OPENGL_PATH ""
#endif
#ifndef BACKEND_SOKOL_PATH
#define BACKEND_SOKOL_PATH ""
#endif
#ifndef COMPLETENESS_CONTRACT_PATH
#define COMPLETENESS_CONTRACT_PATH ""
#endif

namespace
{
namespace sm = surfacematerial;
namespace stam = arbit::stamdiffraction;
namespace sl = arbitgpu::stamlayer;
namespace smb = surfacematerialbinding;

constexpr double pi = 3.141592653589793238462643383279502884;

int checks = 0;
int failures = 0;

void check(bool condition, const std::string& message)
{
    ++checks;
    if (!condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

double radians(double degreesValue)
{
    return degreesValue * pi / 180.0;
}

stam::Vec3 inPlane(double thetaDeg)
{
    return stam::Vec3 {std::sin(radians(thetaDeg)), 0.0, std::cos(radians(thetaDeg))};
}

bool finite(stam::Vec3 v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

double sum(stam::Vec3 v)
{
    return v.x + v.y + v.z;
}

std::string readFile(const char* path)
{
    std::ifstream file(path, std::ios::binary);
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

std::size_t count(const std::string& text, const std::string& needle)
{
    std::size_t n = 0;
    for (auto at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size()))
        ++n;
    return n;
}

const std::array<sm::DiffractionLayerPreset, sm::kDiffractionLayerPresetCount> kPresets {
    sm::DiffractionLayerPreset::Cd, sm::DiffractionLayerPreset::Dvd,
    sm::DiffractionLayerPreset::HolographicFoil, sm::DiffractionLayerPreset::Card,
    sm::DiffractionLayerPreset::OilSlick};

// ---------------------------------------------------------------------------
// IR
// ---------------------------------------------------------------------------
sm::ValueId append(sm::ProgramDescription& program, sm::OperationKind kind, sm::ValueType type,
                   std::initializer_list<sm::ValueId> inputs = {})
{
    sm::Operation operation;
    operation.id = static_cast<sm::ValueId>(program.operations.size() + 1);
    operation.kind = kind;
    operation.resultType = type;
    operation.inputCount = static_cast<std::uint8_t>(inputs.size());
    std::copy(inputs.begin(), inputs.end(), operation.inputs.begin());
    program.operations.push_back(operation);
    return operation.id;
}

sm::ProgramDescription minimalProgram()
{
    sm::ProgramDescription program;
    const auto scalar = append(program, sm::OperationKind::FloatConstant, sm::ValueType::Scalar);
    program.operations.back().literal = {0.5f, 0.0f, 0.0f, 0.0f};
    const auto vector = append(program, sm::OperationKind::ComposeVec3, sm::ValueType::Vec3,
                               {scalar, scalar, scalar});
    const auto id = append(program, sm::OperationKind::UIntConstant, sm::ValueType::UInt);
    program.operations.back().unsignedLiteral = 7;
    for (const auto output : sm::kSurfaceOutputs)
    {
        const auto type = sm::outputType(output);
        program.outputs[static_cast<std::size_t>(output)]
            = type == sm::ValueType::Vec3 ? vector : type == sm::ValueType::UInt ? id : scalar;
    }
    return program;
}

void testPresetsAndBounds()
{
    std::set<std::string> tokens;
    for (const auto preset : kPresets)
    {
        const auto layer = sm::diffractionLayerPreset(preset);
        const std::string name(sm::token(preset));
        tokens.insert(name);
        check(sm::validDiffractionLayer(layer), "preset " + name + " is a valid layer");
        // Twin round trip: the layer carries the Stam preset exactly up to float rounding.
        const auto back = sm::fromStamLayerParams(sm::toStamLayerParams(layer));
        auto placed = back;
        placed.uvScale = layer.uvScale;
        placed.radialCentreU = layer.radialCentreU;
        placed.radialCentreV = layer.radialCentreV;
        check(sm::sameDiffractionLayer(placed, layer), "preset " + name + " survives the Stam round trip");
    }
    check(tokens.size() == kPresets.size(), "preset tokens are distinct");
    const auto cd = sm::toStamLayerParams(sm::diffractionLayerPreset(sm::DiffractionLayerPreset::Cd));
    check(cd.mode == stam::kGrooveRadial && cd.periodNm == static_cast<double>(static_cast<float>(stam::kCdTrackPitchNm)),
          "the CD preset is the Stam CD preset");
    const auto foil = sm::toStamLayerParams(sm::diffractionLayerPreset(sm::DiffractionLayerPreset::HolographicFoil));
    check(foil.mode == stam::kGrooveCells && foil.baseMode == stam::kBaseOpaque,
          "the holographic-foil preset is the metallized Stam foil");
    const auto card = sm::toStamLayerParams(sm::diffractionLayerPreset(sm::DiffractionLayerPreset::Card));
    check(card.baseMode == stam::kBaseHri, "the card preset is the HRI foil");

    sm::DiffractionLayerDescription layer;
    check(sm::validDiffractionLayer(layer), "the default layer is valid");
    auto bad = layer;
    bad.periodNm = 50.0f;
    check(!sm::validDiffractionLayer(bad), "a period under 100 nm is rejected");
    bad = layer;
    bad.mask = std::nanf("");
    check(!sm::validDiffractionLayer(bad), "a NaN field is rejected");
    bad = layer;
    bad.grooveMode = 5;
    check(!sm::validDiffractionLayer(bad), "an unknown groove mode is rejected");
    bad = layer;
    bad.version = 2;
    check(!sm::validDiffractionLayer(bad), "an unknown layer version is rejected");
    bad = layer;
    bad.maxOrder = sm::kDiffractionLayerMaximumOrder + 1;
    check(!sm::validDiffractionLayer(bad), "too many orders are rejected");
}

void testAdmissionAndDigest()
{
    std::string error;
    auto program = minimalProgram();
    const auto plain = sm::admit(program, error);
    check(plain.has_value(), "the plain program is admitted: " + error);
    check(plain && !plain->diffractionLayer(), "an old program has no layer");

    program.diffractionLayer = sm::diffractionLayerPreset(sm::DiffractionLayerPreset::HolographicFoil);
    const auto layered = sm::admit(program, error);
    check(layered.has_value(), "the layered program is admitted: " + error);
    check(layered && layered->diffractionLayer()
              && sm::sameDiffractionLayer(*layered->diffractionLayer(), *program.diffractionLayer),
          "admission keeps the layer");
    check(plain && layered && plain->structuralDigest() != layered->structuralDigest(),
          "the layer changes the structural digest");

    auto modulated = program;
    modulated.diffractionLayer->strength = 2.0f;
    const auto changed = sm::admit(modulated, error);
    check(changed && layered && changed->structuralDigest() != layered->structuralDigest(),
          "a layer value changes the structural digest");

    // Back-compat: removing the layer gives the old digest back, byte for byte.
    program.diffractionLayer.reset();
    const auto again = sm::admit(program, error);
    check(again && plain && again->structuralDigest() == plain->structuralDigest(),
          "a program without the layer keeps its old digest");

    program.diffractionLayer = sm::DiffractionLayerDescription {};
    program.diffractionLayer->periodNm = 10.0f;
    check(!sm::admit(program, error).has_value(), "admission rejects an invalid layer");
}

smb::ImportedSceneMaterialRequest request(bool layer, bool reflectance)
{
    smb::ImportedSceneMaterialRequest material;
    material.scene.value = 3;
    material.sceneRevision = 1;
    material.structuralRevision = 2;
    material.evaluationRevision = 3;
    material.programRevision = 4;
    material.program = minimalProgram();
    if (reflectance)
    {
        sm::SpectralReflectanceDescription description;
        description.coefficients.fill(0.5f);
        material.program.spectralReflectance = description;
    }
    if (layer)
    {
        auto description = sm::diffractionLayerPreset(sm::DiffractionLayerPreset::Card);
        description.grooveAngle = -0.0f; // bitwise round trip of a signed zero
        description.strength = 2.75f;
        description.radialCentreU = 0.3f;
        material.program.diffractionLayer = description;
    }
    material.binding.targetKind = smb::BindingTargetKind::MaterialSlot;
    material.binding.materialSlot.value = 1;
    material.binding.surfaceMaterialDigest = std::string(64, 'a');
    material.binding.surfaceMaterialRevision = 1;
    material.version = smb::requestVersion(false, reflectance, 1, layer);
    return material;
}

bool roundTrip(const smb::ImportedSceneMaterialRequest& material, smb::ImportedSceneMaterialRequest& decoded,
               std::string& text)
{
    std::ostringstream encoded;
    visualimportedscenerender::detail::encodeMaterial(encoded, material, false);
    text = encoded.str();
    std::istringstream input(text);
    return visualimportedscenerender::detail::decodeMaterial(input, decoded, false);
}

void testWire()
{
    check(smb::requestVersion(false, false) == smb::kWireVersion, "old requests keep wire version 1");
    check(smb::requestVersion(true, true, 2) == smb::kRoughGlassWireVersion, "rough glass keeps wire version 4");
    check(smb::requestVersion(false, false, 1, true) == smb::kDiffractionLayerWireVersion
              && smb::kDiffractionLayerWireVersion == 5,
          "a layer selects wire version 5");

    for (const bool reflectance : {false, true})
    {
        const auto label = std::string(reflectance ? " with reflectance" : "");
        const auto layered = request(true, reflectance);
        check(visualimportedscenerender::detail::validMaterial(layered), "the layered request is valid" + label);
        smb::ImportedSceneMaterialRequest decoded;
        std::string text;
        check(roundTrip(layered, decoded, text), "the layered request decodes" + label);
        check(decoded.version == smb::kDiffractionLayerWireVersion, "the decoded request keeps version 5" + label);
        check(decoded.program.diffractionLayer
                  && sm::sameDiffractionLayer(*decoded.program.diffractionLayer, *layered.program.diffractionLayer),
              "the layer survives the wire bit for bit" + label);
        check(decoded.program.spectralReflectance.has_value() == reflectance,
              "the reflectance flag survives next to the layer" + label);
        check(visualimportedscenerender::detail::sameProgram(decoded.program, layered.program),
              "the decoded program is the same program" + label);

        // A truncated layer line is refused.
        const auto line = text.find("\n1 4 1 2 ");
        check(line != std::string::npos, "the layer line follows the program" + label);
        if (line != std::string::npos)
        {
            auto truncated = text;
            const auto end = truncated.find('\n', line + 1);
            truncated.erase(line + 1, end - line);
            std::istringstream input(truncated);
            smb::ImportedSceneMaterialRequest refused;
            check(!visualimportedscenerender::detail::decodeMaterial(input, refused, false),
                  "a version 5 request without its layer line is refused" + label);
        }

        // Back-compat: the same request without a layer keeps its old version and
        // wire text and decodes with the layer off.
        const auto old = request(false, reflectance);
        check(old.version == (reflectance ? smb::kReflectanceWireVersion : smb::kWireVersion),
              "a request without a layer keeps its old version" + label);
        smb::ImportedSceneMaterialRequest oldDecoded;
        std::string oldText;
        check(roundTrip(old, oldDecoded, oldText), "the old request decodes" + label);
        check(!oldDecoded.program.diffractionLayer, "an old request decodes with the layer off" + label);
        check(oldText.find("\n1 4 1 2 ") == std::string::npos, "an old request has no layer line" + label);

        // The version and the layer must agree.
        auto mismatched = old;
        mismatched.program.diffractionLayer = sm::DiffractionLayerDescription {};
        check(!visualimportedscenerender::detail::validMaterial(mismatched),
              "a layer under an old wire version is refused" + label);
        auto missing = layered;
        missing.program.diffractionLayer.reset();
        check(!visualimportedscenerender::detail::validMaterial(missing),
              "wire version 5 without a layer is refused" + label);
    }
    auto a = request(true, false);
    auto b = a;
    b.program.diffractionLayer->mask = 0.5f;
    check(!visualimportedscenerender::detail::sameProgram(a.program, b.program), "sameProgram compares layer values");
}

// ---------------------------------------------------------------------------
// Back-compat goldens. Requests and programs without a layer must produce the
// bytes they produced at 9dc9a4edb6, the parent of D1.
// ---------------------------------------------------------------------------
// Turns a minimal version 1 program into a closed-volume version 2 program,
// which rough glass v2 requires. Dispersion gets its own zero constant.
void makeClosedVolume(sm::ProgramDescription& program)
{
    program.version = sm::kSpectralWireVersion;
    program.transport = sm::TransportMode::ClosedVolume;
    const auto dispersion = append(program, sm::OperationKind::FloatConstant, sm::ValueType::Scalar);
    program.outputs[static_cast<std::size_t>(sm::OutputSemantic::Dispersion)] = dispersion;
}

smb::ImportedSceneMaterialRequest roughGlassRequest(bool reflectance, bool layer)
{
    auto material = request(layer, reflectance);
    makeClosedVolume(material.program);
    material.roughGlassScatteringVersion = 2;
    material.version = smb::requestVersion(false, reflectance, 2, layer);
    return material;
}

// The structural digest exactly as surface_material_admission.h computed it at
// 9dc9a4edb6, written out again here so a change to the shared helpers cannot
// move both sides at once.
namespace legacy
{
void u8(std::vector<std::uint8_t>& bytes, std::uint8_t value)
{
    bytes.push_back(value);
}

void u16(std::vector<std::uint8_t>& bytes, std::uint16_t value)
{
    bytes.push_back(static_cast<std::uint8_t>(value >> 8u));
    bytes.push_back(static_cast<std::uint8_t>(value));
}

void u32(std::vector<std::uint8_t>& bytes, std::uint32_t value)
{
    bytes.push_back(static_cast<std::uint8_t>(value >> 24u));
    bytes.push_back(static_cast<std::uint8_t>(value >> 16u));
    bytes.push_back(static_cast<std::uint8_t>(value >> 8u));
    bytes.push_back(static_cast<std::uint8_t>(value));
}

void f32(std::vector<std::uint8_t>& bytes, float value)
{
    if (value == 0.0f) value = 0.0f;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    u32(bytes, bits);
}

std::string structuralDigest(const sm::ProgramDescription& program)
{
    static constexpr char domain[] = "DonutStudio/SurfaceMaterialIR/Structure/v1";
    std::vector<std::uint8_t> bytes(std::begin(domain), std::end(domain) - 1);
    u32(bytes, program.version);
    u16(bytes, program.textureSlotCount);
    u32(bytes, static_cast<std::uint32_t>(program.operations.size()));
    for (const auto& operation : program.operations)
    {
        u16(bytes, operation.id);
        u8(bytes, static_cast<std::uint8_t>(operation.kind));
        u8(bytes, static_cast<std::uint8_t>(operation.resultType));
        u8(bytes, operation.inputCount);
        for (const auto input : operation.inputs) u16(bytes, input);
        for (const auto value : operation.literal) f32(bytes, value);
        u8(bytes, static_cast<std::uint8_t>(operation.semantic));
        u16(bytes, operation.parameter);
        u32(bytes, operation.unsignedLiteral);
    }
    for (std::size_t index = 0; index < sm::outputCount(program.version); ++index)
        u16(bytes, program.outputs[index]);
    if (program.version == sm::kSpectralWireVersion)
        u8(bytes, static_cast<std::uint8_t>(program.transport));
    if (program.spectralReflectance)
    {
        static constexpr char companionDomain[] = "DonutStudio/SurfaceMaterialIR/SpectralReflectance/v1";
        bytes.insert(bytes.end(), std::begin(companionDomain), std::end(companionDomain) - 1);
        u32(bytes, program.spectralReflectance->version);
        for (const auto coefficient : program.spectralReflectance->coefficients) f32(bytes, coefficient);
    }
    videohelper::Sha256 hash;
    hash.update(bytes.data(), bytes.size());
    return hash.finishHex();
}
} // namespace legacy

void testLegacyDigests()
{
    // The oracle's hash is the FIPS 180-2 "abc" vector.
    videohelper::Sha256 abc;
    abc.update("abc", 3);
    check(abc.finishHex() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "the digest oracle hashes the FIPS 180-2 abc vector");

    std::vector<std::pair<std::string, sm::ProgramDescription>> programs;
    programs.emplace_back("v1", minimalProgram());
    auto withReflectance = minimalProgram();
    withReflectance.spectralReflectance = sm::SpectralReflectanceDescription {};
    withReflectance.spectralReflectance->coefficients.fill(0.5f);
    programs.emplace_back("v1 with reflectance", withReflectance);
    auto closed = minimalProgram();
    makeClosedVolume(closed);
    programs.emplace_back("closed-volume v2", closed);
    auto closedReflectance = closed;
    closedReflectance.spectralReflectance = withReflectance.spectralReflectance;
    programs.emplace_back("closed-volume v2 with reflectance", closedReflectance);
    auto signedZero = minimalProgram();
    signedZero.operations[0].literal[1] = -0.0f;
    programs.emplace_back("v1 with a negative zero literal", signedZero);

    for (const auto& entry : programs)
    {
        std::string error;
        const auto admitted = sm::admit(entry.second, error);
        check(admitted.has_value(), entry.first + " is admitted: " + error);
        check(admitted && admitted->structuralDigest() == legacy::structuralDigest(entry.second),
              entry.first + " keeps its 9dc9a4edb6 structural digest");
    }
}

std::string encodeOnly(const smb::ImportedSceneMaterialRequest& material)
{
    std::ostringstream encoded;
    visualimportedscenerender::detail::encodeMaterial(encoded, material, false);
    return encoded.str();
}

// Wire text derived by hand from encodeMaterial at 9dc9a4edb6 for request()
// and roughGlassRequest(). Operation lines are id, kind, result type, input
// count, four inputs, four literal bit patterns, semantic, parameter and
// unsigned literal. 1056964608 is the bit pattern of 0.5f.
const std::string kGoldenHeaderTail = " 3 1 2 3 4\n";
const std::string kGoldenOperations = "1 1 1 0 0 0 0 0 1056964608 0 0 0 0 0 0\n"
                                      "2 20 3 3 1 1 1 0 0 0 0 0 0 0 0\n"
                                      "3 2 5 0 0 0 0 0 0 0 0 0 0 0 7\n";
const std::string kGoldenDispersion = "4 1 1 0 0 0 0 0 0 0 0 0 0 0 0\n";
const std::string kGoldenReflectance = "1 1056964608 1056964608 1056964608 1056964608 1056964608 1056964608"
                                       " 1056964608 1056964608\n";
const std::string kGoldenBinding = "1 1 0 \"" + std::string(64, 'a') + "\" 1\n0\n";

void testLegacyWire()
{
    const auto v1 = "1" + kGoldenHeaderTail + "1 0 3\n" + kGoldenOperations + "2 1 1 2 1 2 1 1 1 3\n" + kGoldenBinding;
    const auto v3 = "3" + kGoldenHeaderTail + "1 0 3\n" + kGoldenOperations + "2 1 1 2 1 2 1 1 1 3\n"
                  + kGoldenReflectance + kGoldenBinding;
    const auto v4Body = "2 0 4\n" + kGoldenOperations + kGoldenDispersion + "2 1 1 2 1 2 1 1 1 3 4 1\n";
    const auto v4 = "4" + kGoldenHeaderTail + "2 0\n" + v4Body + kGoldenBinding;
    const auto v4Reflectance = "4" + kGoldenHeaderTail + "2 1\n" + v4Body + kGoldenReflectance + kGoldenBinding;

    const auto plain = request(false, false);
    const auto reflectance = request(false, true);
    const auto glass = roughGlassRequest(false, false);
    const auto glassReflectance = roughGlassRequest(true, false);
    check(plain.version == smb::kWireVersion && reflectance.version == smb::kReflectanceWireVersion
              && glass.version == smb::kRoughGlassWireVersion && glassReflectance.version == smb::kRoughGlassWireVersion,
          "the golden requests carry wire versions 1, 3, 4 and 4");
    check(encodeOnly(plain) == v1, "a v1 request encodes to its 9dc9a4edb6 bytes");
    check(encodeOnly(reflectance) == v3, "a v3 request encodes to its 9dc9a4edb6 bytes");
    check(encodeOnly(glass) == v4, "a v4 request encodes to its 9dc9a4edb6 bytes");
    check(encodeOnly(glassReflectance) == v4Reflectance, "a v4 request with reflectance encodes to its 9dc9a4edb6 bytes");

    // v4 decode round trip, from the golden bytes.
    const std::array<std::pair<const std::string*, const smb::ImportedSceneMaterialRequest*>, 2> decodeCases{{
        {&v4, &glass},
        {&v4Reflectance, &glassReflectance},
    }};
    for (const auto& entry : decodeCases)
    {
        const bool withReflectance = entry.second->program.spectralReflectance.has_value();
        const auto label = std::string(withReflectance ? " with reflectance" : "");
        std::istringstream input(*entry.first);
        smb::ImportedSceneMaterialRequest decoded;
        check(visualimportedscenerender::detail::decodeMaterial(input, decoded, false), "the v4 golden decodes" + label);
        check(decoded.version == smb::kRoughGlassWireVersion && decoded.roughGlassScatteringVersion == 2,
              "the v4 golden keeps version 4 and rough glass v2" + label);
        check(decoded.program.version == sm::kSpectralWireVersion
                  && decoded.program.transport == sm::TransportMode::ClosedVolume,
              "the v4 golden keeps the closed-volume v2 program" + label);
        check(decoded.program.spectralReflectance.has_value() == withReflectance && !decoded.program.diffractionLayer,
              "the v4 golden keeps its reflectance flag and has no layer" + label);
        check(visualimportedscenerender::detail::sameProgram(decoded.program, entry.second->program),
              "the v4 golden decodes to the same program" + label);
        check(encodeOnly(decoded) == *entry.first, "the decoded v4 golden encodes to the same bytes" + label);
    }
}

// ---------------------------------------------------------------------------
// The Geometry Surface validator and the imported-scene validator agree on the
// version, reflectance, rough-glass and layer rules.
// ---------------------------------------------------------------------------
smb::ImportedSceneMaterialRequest geometryRequest(bool reflectance, bool layer, std::uint32_t roughGlass)
{
    auto material = roughGlass == 2 ? roughGlassRequest(reflectance, layer) : request(layer, reflectance);
    material.scene.value = 1;
    material.sceneRevision = 1;
    material.structuralRevision = 1;
    material.evaluationRevision = 1;
    material.programRevision = 1;
    material.binding.targetKind = smb::BindingTargetKind::ObjectOverride;
    material.binding.materialSlot.value = 0;
    material.binding.object.value = 1;
    material.binding.surfaceMaterialRevision = 1;
    std::string error;
    const auto admitted = sm::admit(material.program, error);
    material.binding.surfaceMaterialDigest = admitted ? admitted->structuralDigest() : std::string(64, '0');
    return material;
}

void testValidatorsAgree()
{
    for (const std::uint32_t roughGlass : {1u, 2u})
        for (const bool reflectance : {false, true})
            for (const bool layer : {false, true})
            {
                const auto label = " (rough glass v" + std::to_string(roughGlass) + (reflectance ? ", reflectance" : "")
                                 + (layer ? ", layer" : "") + ")";
                const auto material = geometryRequest(reflectance, layer, roughGlass);
                std::string error;
                const bool imported = visualimportedscenerender::detail::validMaterial(material);
                const bool geometry = geometrysurfacematerial::valid(material, error);
                check(imported, "the imported-scene validator accepts" + label);
                check(geometry, "the Geometry Surface validator accepts" + label + ": " + error);
                check(!geometry || !geometrysurfacematerial::encode(material).empty(),
                      "Geometry Surface encodes" + label);
            }

    // These used to diverge: Geometry Surface refused reflectance under rough
    // glass v1, with or without a layer, while the imported scene accepted it.
    for (const bool layer : {false, true})
    {
        const auto material = geometryRequest(true, layer, 1);
        std::string error;
        const auto wire = geometrysurfacematerial::encode(material);
        const auto decoded = geometrysurfacematerial::decode(wire, error);
        check(material.version == (layer ? smb::kDiffractionLayerWireVersion : smb::kReflectanceWireVersion),
              std::string("reflectance under rough glass v1 selects wire version ") + (layer ? "5" : "3"));
        check(decoded && visualimportedscenerender::sameMaterialRequest(*decoded, material),
              std::string("Geometry Surface round-trips reflectance under rough glass v1")
                  + (layer ? " with a layer" : "") + ": " + error);
    }

    // Both still refuse what both refused before.
    auto thin = geometryRequest(true, true, 2);
    thin.program.transport = sm::TransportMode::ThinSurface;
    std::string error;
    check(!visualimportedscenerender::detail::validMaterial(thin) && !geometrysurfacematerial::valid(thin, error),
          "both validators refuse rough glass v2 over a thin surface");
    auto stale = geometryRequest(true, true, 1);
    stale.version = smb::kReflectanceWireVersion;
    check(!visualimportedscenerender::detail::validMaterial(stale) && !geometrysurfacematerial::valid(stale, error),
          "both validators refuse a layer under wire version 3");
}

void testMaterialField()
{
    check(std::size(surfacematerialfield::kTargets) == 17, "the Material Field has 17 targets");
    check(std::string(surfacematerialfield::kTargets[12]) == "Dispersion",
          "released targets keep their indices");
    check(std::string(surfacematerialfield::kTargets[surfacematerialfield::kFirstDiffractionTarget]) == "Diffraction Period"
              && std::string(surfacematerialfield::kTargets[16]) == "Diffraction Mask",
          "the diffraction targets follow the released ones");
    sm::DiffractionLayerDescription layer;
    surfacematerialfield::evaluateDiffraction(13, 250.0, layer);
    check(layer.periodNm == 1250.0f && layer.secondPeriodNm == 1250.0f, "the period target adds nanometres to both periods");
    surfacematerialfield::evaluateDiffraction(13, -1.0e9, layer);
    check(layer.periodNm == sm::kDiffractionLayerMinimumPeriodNm, "the period target clamps to the bounds");
    surfacematerialfield::evaluateDiffraction(14, 90.0, layer);
    check(std::fabs(layer.grooveAngle - static_cast<float>(pi / 2.0)) < 1e-6f, "the angle target adds degrees");
    surfacematerialfield::evaluateDiffraction(15, 2.0, layer);
    check(layer.strength == 3.0f, "the strength target adds to the strength");
    surfacematerialfield::evaluateDiffraction(16, 5.0, layer);
    check(layer.mask == 1.0f, "the mask target clamps to 1");
    check(sm::validDiffractionLayer(layer), "modulated layers stay valid");
}

void testFoilMapping()
{
    diffractivefoil::Description foil;
    const float spacings[4] = {800.0f, 900.0f, 1100.0f, 1200.0f};
    for (std::size_t i = 0; i < foil.grooveField.size(); ++i)
    {
        foil.grooveField[i].grooveSpacingNanometres = spacings[i % 4];
        foil.grooveField[i].reciprocalDirectionUv = {0.0f, 2.0f};
        foil.grooveField[i].diffractionCoverage = 0.5f;
    }
    const auto layer = sm::holographicFoilLayerFromFoil(foil);
    check(sm::validDiffractionLayer(layer), "the foil maps to a valid layer");
    check(layer.grooveMode == static_cast<std::uint32_t>(sm::DiffractionGrooveMode::Cells),
          "the foil maps to the holographic-foil preset");
    check(std::fabs(layer.periodNm - 1000.0f) < 1e-3f, "the foil period is the mean spacing");
    check(std::fabs(layer.periodJitter - 0.2f) < 1e-6f, "the foil spread becomes the jitter");
    check(std::fabs(layer.grooveAngle - static_cast<float>(pi / 2.0)) < 1e-6f, "the foil direction becomes the angle");
    check(layer.mask == 0.5f, "the foil coverage becomes the mask");
}

// ---------------------------------------------------------------------------
// GPU module
// ---------------------------------------------------------------------------
void testPacking()
{
    const auto zero = sl::pack(nullptr);
    bool allZero = true;
    for (const auto& row : zero)
        for (const auto value : row) allZero = allZero && value == 0.0f;
    check(allZero, "a program without a layer packs all zero, which is layer off");

    auto layer = sm::diffractionLayerPreset(sm::DiffractionLayerPreset::Cd);
    layer.radialCentreV = 0.25f;
    const auto packed = sl::pack(&layer);
    check(sl::lane(packed, sl::kEnabledLane) == 1.0f, "the enable lane is set");
    check(sl::lane(packed, sl::kModeLane) == static_cast<float>(layer.grooveMode)
              && sl::lane(packed, sl::kBaseModeLane) == static_cast<float>(layer.baseMode)
              && sl::lane(packed, sl::kMaxOrderLane) == static_cast<float>(layer.maxOrder),
          "the integer fields are packed");
    std::size_t index = sl::kFirstFloatLane;
    bool same = true;
    sm::forEachDiffractionLayerFloat(layer, [&](const char* name, float value) {
        same = same && sl::lane(packed, index) == value && sl::floatLane(name) == index;
        ++index;
    });
    check(same, "every float field is packed at its lane");
    check(index <= sl::kUniformVec4Count * 4, "the packing fits the uniform array");
    check(sl::floatLane("unknown") == 0, "unknown fields have no lane");
    check(sl::laneExpression("uStamLayer", sl::kEnabledLane) == "uStamLayer[0].w",
          "the enable flag is uStamLayer[0].w");
}

std::vector<std::string> structFields(const std::string& glsl, const std::string& name)
{
    std::vector<std::string> fields;
    const auto begin = glsl.find("struct " + name + " {");
    const auto end = glsl.find("};", begin);
    if (begin == std::string::npos || end == std::string::npos) return fields;
    const std::regex field(R"(^\s+(?:int|float|vec[234])\s+(\w+);)");
    std::istringstream lines(glsl.substr(begin, end - begin));
    for (std::string line; std::getline(lines, line);)
    {
        std::smatch match;
        if (std::regex_search(line, match, field)) fields.push_back(match[1]);
    }
    return fields;
}

void testGlslContract()
{
    const std::string core = kStamDiffractionGlsl;
    const std::string environment = kStamDiffractionEnvironmentGlsl;
    const auto glsl = sl::glslShader();

    // The module runs the shared Stam text itself, not a copy.
    const auto presets = core.find("const StamLayerParams stamCdPreset");
    const auto body = core.find("float stamCmfLobe(");
    check(presets != std::string::npos && body != std::string::npos, "the core GLSL has its preset block and body");
    check(glsl.find(core.substr(0, presets)) != std::string::npos, "the module carries the Stam constants and structs verbatim");
    check(glsl.find(core.substr(body)) != std::string::npos, "the module carries the Stam functions verbatim");
    check(glsl.find(environment) != std::string::npos, "the module carries the Stam environment GLSL verbatim");
    check(glsl.find("stamCdPreset") == std::string::npos, "the unused preset constants are dropped");

    // Every StamLayerParams field comes from the uniform array, generated from
    // the one field list.
    const auto fields = structFields(core, "StamLayerParams");
    check(fields.size() == 22, "StamLayerParams has 22 fields");
    check(fields.size() == 3 + sm::kDiffractionLayerFloatCount - sm::kDiffractionLayerPlacementCount,
          "the IR field list covers StamLayerParams");
    for (const auto& field : fields)
    {
        const bool integer = field == "mode" || field == "baseMode" || field == "maxOrder";
        check(glsl.find("    P." + field + (integer ? " = int(floor(uStamLayer[" : " = uStamLayer[")) != std::string::npos,
              "the unpack function sets " + field + " from uStamLayer");
    }
    std::size_t lane = sl::kFirstFloatLane;
    const sm::DiffractionLayerDescription defaults {};
    sm::forEachDiffractionLayerFloat(defaults, [&](const char* name, float) {
        const bool stamField = std::find(fields.begin(), fields.end(), name) != fields.end();
        if (stamField)
            check(glsl.find(std::string("    P.") + name + " = " + sl::laneExpression("uStamLayer", lane) + ";")
                      != std::string::npos,
                  std::string("GLSL reads ") + name + " from the lane the C++ packs it into");
        ++lane;
    });
    check(glsl.find("uniform vec4 uStamLayer[7];") != std::string::npos, "the uniform array has 7 vec4");
    check(glsl.find("environmentImageRadiance(dir, uSky[0], uSky[1])") != std::string::npos
              && glsl.find("analyticSkyRadiance(dir, uSky[0], uSky[1], uSky[2], uSky[3], uSky[4])") != std::string::npos,
          "envGrating reads the HDRI or the analytic sky");
    check(glsl.find("vec3 stamEnvironmentRadiance(vec3 dir) {") != std::string::npos,
          "the module defines the radiance the Stam environment code declares");
    check(glsl.find("return R.coatSpecular + R.diffraction + bsdf * (R.specularScale - vec3(1.0));") != std::string::npos,
          "direct lights add coat + orders + bsdf * (specularScale - 1)");
    check(glsl.find("return R.coatSpecular + R.diffraction + skyBase * R.specularScale;") != std::string::npos,
          "the sky becomes coat + orders + sky * specularScale");

    // Metal is the same text through fixed substitutions.
    const auto metal = sl::metalShader();
    for (const char* glslOnly : {"vec2", "vec3", "vec4", "uvec", "ivec", "out float", "out bool", "inversesqrt",
                                 "\nconst ", "uSky", "uStamLayer", "gl_", "texelFetch", "uEnvironmentImage"})
    {
        std::string stripped = metal;
        for (const char* metalType : {"float2", "float3", "float4", "uint2", "int2"})
            for (auto at = stripped.find(metalType); at != std::string::npos; at = stripped.find(metalType, at))
                stripped.replace(at, std::strlen(metalType), "T");
        check(stripped.find(glslOnly) == std::string::npos, std::string("the Metal module has no GLSL ") + glslOnly);
    }
    check(metal.find("\n#define FIXTURE_STAM_LAYER 1\n") == 0, "the Metal module defines FIXTURE_STAM_LAYER first");
    check(metal.find("thread float3& T, thread float& periodNm") != std::string::npos,
          "out parameters become thread references");
    check(metal.find("u.stamLayer[") != std::string::npos, "Metal reads the layer from the uniform block");
    // The environment GLSL declares stamEnvironmentRadiance once and calls it at every lookup.
    check(count(metal, "stamEnvironmentRadiance(u, environmentTexture, ") == count(environment, "stamEnvironmentRadiance(") - 1,
          "every Metal environment lookup passes the uniforms and the HDRI texture");
    check(metal.find("environmentImageRadiance(dir, u.sky[0], u.sky[1], environmentTexture)") != std::string::npos,
          "Metal envGrating samples the HDRI texture");
    const std::regex definition(R"(\n(?:void|float|int|uint|vec[234]|StamLayerParams|StamSurface|StamLayerResult) (stam\w+)\()");
    std::size_t functions = 0;
    for (std::sregex_iterator it(glsl.begin(), glsl.end(), definition), end; it != end; ++it)
    {
        ++functions;
        const std::string name = (*it)[1];
        check(metal.find(" " + name + "(") != std::string::npos, "Metal defines " + name);
    }
    check(functions > 30, "the GLSL module defines the Stam functions");
}

void testBackendSources()
{
    const auto gl = readFile(BACKEND_OPENGL_PATH);
    const auto mm = readFile(BACKEND_SOKOL_PATH);
    check(!gl.empty() && !mm.empty(), "backend sources are readable");
    check(gl.find("kFixtureFragmentStamLayer = 1u << 2,") != std::string::npos, "OpenGL has the layer feature bit");
    check(gl.find("if (program.diffractionLayer) features |= kFixtureFragmentStamLayer;") != std::string::npos,
          "a program with a layer turns the feature on");
    check(gl.find("std::string (\"#define FIXTURE_STAM_LAYER \") + (layered ? \"1\" : \"0\")") != std::string::npos,
          "the OpenGL fragment source defines FIXTURE_STAM_LAYER");
    check(gl.find("if (layered) text += stamlayer::glslShader();") != std::string::npos,
          "OpenGL appends the generated module");
    check(gl.find("uniforms.stamLayer = location (\"uStamLayer[0]\");") != std::string::npos,
          "OpenGL looks up uStamLayer");
    check(count(gl, "stamlayer::pack(") == 2, "OpenGL uploads the layer per frame and per object");
    const auto glMain = gl.substr(gl.find("const char* kFixtureFragmentShaderTail"));
    check(glMain.find("bool stamLayerOn = uStamLayer[0].w > 0.5;") != std::string::npos,
          "OpenGL reads the enable lane");
    check(glMain.find("directBase += radiance * stamLayerDirectDelta(") != std::string::npos,
          "OpenGL composes the layer over every direct light");
    check(glMain.find("stamLayerEnvironmentCompose(stamParams, stamSurface, stamView,") != std::string::npos,
          "OpenGL composes the layer over the sky or HDRI");
    check(count(glMain, "#if FIXTURE_STAM_LAYER") == 4, "every OpenGL layer block is gated");

    check(count(mm, "\n    float4 vertexDisplacementLimits[16];\n};") == 2
              && count(mm, "\n    float4 stamLayer[7];\n") == 0,
          "both released MSL uniform structs end before the layer");
    check(mm.find("\"    float4 vertexDisplacementLimits[16];\\n    float4 stamLayer[7];\\n};\"") != std::string::npos
              && mm.find("            if(layered && !composeLayeredFixtureMain(fragmentSource)) {\n") != std::string::npos
              && mm.find("            else if(layered) declareLayeredFixtureUniform(fragmentSource);\n") != std::string::npos,
          "only a layered Metal fragment declares the layer and composes its _main blocks");
    check(mm.find("float stamLayer[arbitgpu::stamlayer::kUniformVec4Count][4];") != std::string::npos
              && mm.find("offsetof (FixtureUniforms, stamLayer) == 11536") != std::string::npos
              && mm.find("offsetof (FixtureUniforms, glassTransport2) == 11648") != std::string::npos
              && mm.find("sizeof (FixtureUniforms) == 11952") != std::string::npos,
          "the C++ uniform block matches the MSL layout, with the paired partner after the layer");
    check(mm.find("(layered ? arbitgpu::stamlayer::metalShader() : std::string())") != std::string::npos,
          "Metal appends the generated module");
    check(mm.find("&& !layered;") != std::string::npos, "layered programs leave the restricted Metal archive");
    check(count(mm, "storeStamLayer(uniforms,") == 2, "Metal fills the layer per frame and per object");
    check(mm.find("const bool stamLayerOn = u.stamLayer[0].w > 0.5f;") != std::string::npos,
          "Metal reads the enable lane");
    check(count(mm, "#ifdef FIXTURE_STAM_LAYER") == 4, "every Metal layer block is gated");
    check(mm.find("stamBaseSpecular, skyBase - stamSkyBefore, stamStratum, u, environmentTexture);") != std::string::npos,
          "Metal envGrating samples the bound HDRI texture");

    const auto contract = readFile(COMPLETENESS_CONTRACT_PATH);
    check(contract.find("return common && glass && diffractionUniforms && stamLayerUniforms;") != std::string::npos
              && gl.find("return common && glass && diffractionUniforms && stamLayerUniforms;") != std::string::npos,
          "complete() requires uStamLayer only when the layer is compiled in");
}

// ---------------------------------------------------------------------------
// Optics through the C++ twin of the shader composition
// ---------------------------------------------------------------------------
sm::DiffractionLayerDescription testGrating()
{
    sm::DiffractionLayerDescription layer;
    layer.grooveMode = static_cast<std::uint32_t>(sm::DiffractionGrooveMode::Linear);
    layer.baseMode = static_cast<std::uint32_t>(sm::DiffractionBaseMode::Metallized);
    layer.maxOrder = 1;
    layer.periodNm = 1000.0f;
    layer.secondPeriodNm = 1000.0f;
    layer.roughness = 0.03f;
    layer.strength = 1.0f;
    return layer;
}

stam::Surface flatSurface(const sm::DiffractionLayerDescription& layer, double u = 0.7, double v = 0.4)
{
    return sl::twin::surface(layer, stam::Vec3 {0.0, 0.0, 1.0}, stam::Vec3 {1.0, 0.0, 0.0}, 1.0, u, v);
}

// Wavelength of the strongest narrowband response, by a 0.01 nm scan and a
// golden-section refinement.
double peakWavelength(const stam::LayerParams& P, const stam::Surface& S, stam::Vec3 L, stam::Vec3 V)
{
    const auto response = [&](double nm) { return stam::diffractionLayerAtWavelength(P, S, L, V, nm, 0.5); };
    double best = 380.0, bestValue = -1.0;
    for (int i = 0; i <= 40000; ++i)
    {
        const double nm = 380.0 + 0.01 * i;
        const double value = response(nm);
        if (value > bestValue) { bestValue = value; best = nm; }
    }
    double a = best - 0.02, b = best + 0.02;
    const double ratio = 0.5 * (std::sqrt(5.0) - 1.0);
    for (int i = 0; i < 60; ++i)
    {
        const double c = b - ratio * (b - a), d = a + ratio * (b - a);
        if (response(c) > response(d)) b = d; else a = c;
    }
    return bestValue > 0.0 ? 0.5 * (a + b) : -1.0;
}

void testGratingEquation()
{
    const auto layer = testGrating();
    const auto P = sm::toStamLayerParams(layer);
    const auto S = flatSurface(layer);
    const auto V = inPlane(0.0);
    const double d = P.periodNm;

    // Moving the light moves the first-order peak along d (sin theta_i + sin theta_o) = lambda.
    double previous = 0.0;
    for (const double thetaI : {25.0, 33.0, 45.0})
    {
        const double predicted = d * (std::sin(radians(thetaI)) + std::sin(0.0));
        const double peak = peakWavelength(P, S, inPlane(thetaI), V);
        check(std::fabs(peak - predicted) < 0.05,
              "light at " + std::to_string(thetaI) + " deg peaks at " + std::to_string(peak)
                  + " nm, grating equation says " + std::to_string(predicted) + " nm");
        check(peak > previous, "a steeper light moves the peak to longer wavelengths");
        previous = peak;
    }

    // Fixed wavelength: the light angle that lights the viewer follows the equation.
    const double lambda = 550.0;
    double bestDeg = 0.0, bestValue = -1.0;
    for (int i = 0; i <= 44500; ++i)
    {
        const double thetaI = 0.002 * i;
        const double value = stam::diffractionLayerAtWavelength(P, S, inPlane(thetaI), V, lambda, 0.5);
        if (value > bestValue) { bestValue = value; bestDeg = thetaI; }
    }
    const double predictedDeg = std::asin(lambda / d) * 180.0 / pi;
    check(bestValue > 0.0 && std::fabs(bestDeg - predictedDeg) < 0.01,
          "550 nm reaches the viewer from " + std::to_string(bestDeg) + " deg, predicted "
              + std::to_string(predictedDeg) + " deg");

    // The composed colour follows the peak: violet-blue at 25 deg, red at 45 deg.
    sl::twin::Material metal;
    metal.base = {1.0, 1.0, 1.0};
    const auto blue = stam::diffractionLayer(P, S, inPlane(25.0), V, sl::twin::baseSpecular(metal)).diffraction;
    const auto red = stam::diffractionLayer(P, S, inPlane(45.0), V, sl::twin::baseSpecular(metal)).diffraction;
    check(blue.z > blue.x && sum(blue) > 0.0, "the 25 deg order is blue");
    check(red.x > red.z && sum(red) > 0.0, "the 45 deg order is red");

    // Rotating the grooves moves the orders with them: grooves along the plane
    // of incidence send nothing into the normal viewer.
    auto rotated = layer;
    rotated.grooveAngle = static_cast<float>(pi / 2.0);
    const auto R = stam::diffractionLayer(sm::toStamLayerParams(rotated), flatSurface(rotated), inPlane(45.0), V,
                                          sl::twin::baseSpecular(metal));
    check(stam::maxComponent(R.diffraction) < 1e-12, "rotated grooves move the order out of the plane");
}

// Angle between two unit vectors, accurate for tiny angles.
double angleBetween(stam::Vec3 a, stam::Vec3 b)
{
    const auto d = a - b;
    return 2.0 * std::asin(std::min(1.0, 0.5 * std::sqrt(stam::dot(d, d))));
}

// In-plane direction at thetaDeg, tilted out of the T-N plane toward +B (+y).
stam::Vec3 tilted(double thetaDeg, double tiltDeg)
{
    const double c = std::cos(radians(tiltDeg));
    return stam::Vec3 {std::sin(radians(thetaDeg)) * c, std::sin(radians(tiltDeg)), std::cos(radians(thetaDeg)) * c};
}

// envGrating through its C++ twin (diffractionLayerEnvironment ->
// environmentOrders -> environmentGratingDirection) under an environment that
// is black except for one small bright spot. The viewer is on the normal, so
// order 1 at wavelength lambda looks at the direction with
// d * sin(theta) = lambda, toward +T.
void testEnvironmentGratingSpot()
{
    const double stratum = 0.5;
    const int band = 10;
    const double stepNm = (stam::kMaxVisibleNm - stam::kMinVisibleNm) / static_cast<double>(stam::kSpectralSamples);
    const double lambda = stam::kMinVisibleNm + (band + stratum) * stepNm; // 555 nm, a sampled wavelength
    const double spotSigma = radians(0.05);
    sl::twin::Material mirror;
    mirror.base = {1.0, 1.0, 1.0};
    const auto V = inPlane(0.0);

    const auto respond = [&](const sm::DiffractionLayerDescription& layer, stam::Vec3 spot) {
        const auto radiance = [&](stam::Vec3 dir) {
            const double a = angleBetween(dir, spot) / spotSigma;
            const double g = std::exp(-0.5 * a * a);
            return stam::Vec3 {g, g, g};
        };
        return stam::diffractionLayerEnvironment(sm::toStamLayerParams(layer), flatSurface(layer), V,
                                                 sl::twin::baseSpecular(mirror), stratum, radiance)
            .diffraction;
    };
    // In-plane spot position, within 0.5 deg of centreDeg, that lights the green channel most.
    const auto scanPeak = [&](const sm::DiffractionLayerDescription& layer, double centreDeg) {
        double bestDeg = centreDeg, best = -1.0;
        for (int i = -500; i <= 500; ++i)
        {
            const double deg = centreDeg + 0.001 * i;
            const double value = respond(layer, inPlane(deg)).y;
            if (value > best) { best = value; bestDeg = deg; }
        }
        return bestDeg;
    };

    const auto layer = testGrating();
    const double d = static_cast<double>(layer.periodNm);
    const double predictedDeg = std::asin(lambda / d) * 180.0 / pi;
    const auto atSpot = respond(layer, inPlane(predictedDeg));
    const double peak = atSpot.y;
    check(peak > 0.0 && atSpot.y > atSpot.x && atSpot.y > atSpot.z,
          "a spot at the predicted " + std::to_string(predictedDeg) + " deg diffracts 555 nm green into the viewer");
    const double peakDeg = scanPeak(layer, predictedDeg);
    check(std::fabs(peakDeg - predictedDeg) < 0.0015,
          "the environment response peaks at " + std::to_string(peakDeg) + " deg, grating equation says "
              + std::to_string(predictedDeg) + " deg");
    check(respond(layer, inPlane(predictedDeg + 0.5)).y < 1e-6 * peak,
          "moving the spot half a degree along the plane drops the response");
    check(respond(layer, tilted(predictedDeg, 1.0)).y < 1e-6 * peak,
          "moving the spot a degree out of the plane drops the response");
    const auto mirrored = respond(layer, inPlane(-predictedDeg));
    check(std::fabs(mirrored.y - peak) < 1e-9 * peak, "order -1 sees the mirrored spot equally");

    auto wider = layer;
    wider.periodNm = 1200.0f;
    wider.secondPeriodNm = 1200.0f;
    const double widerDeg = std::asin(lambda / 1200.0) * 180.0 / pi;
    check(respond(wider, inPlane(predictedDeg)).y < 1e-6 * peak,
          "with a 1200 nm period the old spot direction goes dark");
    const double widerPeakDeg = scanPeak(wider, widerDeg);
    check(std::fabs(widerPeakDeg - widerDeg) < 0.0015 && widerPeakDeg < peakDeg - 5.0,
          "a 1200 nm period moves the peak to " + std::to_string(widerPeakDeg) + " deg, grating equation says "
              + std::to_string(widerDeg) + " deg");
    check(respond(wider, inPlane(widerDeg)).y > 0.5 * peak, "the moved peak carries the same band");
}

// The Metal module is never compiled here, so check its text for GLSL-only
// spellings that the fixed substitutions could leave behind.
void testMetalHasNoGlslTokens()
{
    const auto metal = sl::metalShader();
    check(!metal.empty(), "the Metal module is generated");
    const std::regex vectorConstructor(R"(\b[uib]?vec[234]\s*\()");
    check(!std::regex_search(metal, vectorConstructor), "the Metal module has no GLSL vector constructor");
    const std::regex vectorType(R"(\b[uib]?vec[234]\b)");
    check(!std::regex_search(metal, vectorType), "the Metal module has no GLSL vector type");
    check(metal.find("uvec") == std::string::npos, "the Metal module has no uvec");
    const std::regex qualifier(R"(\b(inout|in|out)\s+[A-Za-z_])");
    check(!std::regex_search(metal, qualifier), "the Metal module has no in, out or inout parameter qualifier");
    const std::regex functionScopeConst(R"(\n[ \t]+const\s)");
    check(!std::regex_search(metal, functionScopeConst), "the Metal module has no function-scope const");
    const std::regex anyConst(R"(\bconst\b)");
    check(!std::regex_search(metal, anyConst), "every GLSL const became a Metal constant");
}

void testStrengthZeroIsBase()
{
    sl::twin::Material materials[3];
    materials[0].base = {0.9, 0.6, 0.3};
    materials[1].metallic = 0.0;
    materials[1].roughness = 0.6;
    materials[2].metallic = 0.4;
    materials[2].roughness = 0.05;
    for (const auto preset : {sm::DiffractionLayerPreset::Cd, sm::DiffractionLayerPreset::HolographicFoil})
    {
        auto layer = sm::diffractionLayerPreset(preset);
        layer.strength = 0.0f;
        layer.coatIor = 0.0f; // the coat is its own layer; strength drives the grating only
        layer.glitterDensity = 0.0f;
        bool exact = true;
        for (const auto& material : materials)
            for (double thetaI = -80.0; thetaI <= 80.0; thetaI += 10.0)
                for (double thetaO = -80.0; thetaO <= 80.0; thetaO += 10.0)
                    for (const double u : {0.1, 0.55, 0.9})
                    {
                        const auto S = sl::twin::surface(layer, stam::Vec3 {0.0, 0.0, 1.0}, stam::Vec3 {1.0, 0.0, 0.0},
                                                         1.0, u, 0.3);
                        const auto L = inPlane(thetaI), V = stam::Vec3 {-std::sin(radians(thetaO)) * 0.6,
                                                                       std::sin(radians(thetaO)) * 0.8,
                                                                       std::cos(radians(thetaO))};
                        const auto withLayer = sl::twin::direct(&layer, material, S, L, V);
                        const auto base = sl::twin::direct(nullptr, material, S, L, V);
                        exact = exact && withLayer.x == base.x && withLayer.y == base.y && withLayer.z == base.z;
                        const stam::Vec3 sky {0.3, 0.5, 0.7};
                        const auto env = sl::twin::environment(layer, material, S, V, sky, 0.5,
                                                               [](stam::Vec3) { return stam::Vec3 {2.0, 1.0, 0.5}; });
                        exact = exact && env.x == sky.x && env.y == sky.y && env.z == sky.z;
                    }
        check(exact, std::string("strength 0 renders the base exactly for ") + std::string(sm::token(preset)));
    }
}

void testEnergy()
{
    // Uniform white environment, white mirror base: whatever the layer sends
    // into the orders leaves the zero order, so coat + orders + zero order
    // never exceeds the incoming radiance.
    const auto uniform = [](stam::Vec3) { return stam::Vec3 {1.0, 1.0, 1.0}; };
    sl::twin::Material mirror;
    mirror.base = {1.0, 1.0, 1.0};
    double worst = 0.0;
    bool bounded = true;
    for (const auto preset : kPresets)
        for (const float strength : {0.0f, 1.0f, 3.0f, 16.0f})
        {
            auto layer = sm::diffractionLayerPreset(preset);
            layer.strength = strength;
            for (double thetaV = 0.0; thetaV < 89.0; thetaV += 8.0)
                for (const double phi : {0.0, 0.7, 1.9})
                    for (const double stratum : {0.0, 0.37, 0.99})
                        for (const double u : {0.12, 0.61, 0.83})
                        {
                            const auto S = flatSurface(layer, u, 0.27);
                            const stam::Vec3 V {std::sin(radians(thetaV)) * std::cos(phi),
                                                std::sin(radians(thetaV)) * std::sin(phi), std::cos(radians(thetaV))};
                            const auto total = sl::twin::environment(layer, mirror, S, V, stam::Vec3 {1.0, 1.0, 1.0},
                                                                     stratum, uniform);
                            worst = std::max(worst, stam::maxComponent(total));
                            bounded = bounded && finite(total) && stam::maxComponent(total) <= 1.0 + 1e-9;
                            const auto R = stam::diffractionLayerEnvironment(sm::toStamLayerParams(layer), S, V,
                                                                             sl::twin::baseSpecular(mirror), stratum, uniform);
                            bounded = bounded && R.diffractedWeight >= 0.0
                                && stam::maxComponent(R.specularScale) + R.diffractedWeight <= 1.0 + 1e-12;
                        }
        }
    check(bounded, "coat + orders + zero order never exceeds a uniform environment (worst "
                       + std::to_string(worst) + ")");

    // Point lights: the orders only take energy from the zero order.
    bool directBounded = true;
    for (const auto preset : kPresets)
    {
        auto layer = sm::diffractionLayerPreset(preset);
        layer.strength = sm::kDiffractionLayerMaximumStrength;
        const auto P = sm::toStamLayerParams(layer);
        for (double thetaI = -85.0; thetaI <= 85.0; thetaI += 5.0)
            for (double thetaO = -85.0; thetaO <= 85.0; thetaO += 5.0)
            {
                const auto R = stam::diffractionLayer(P, flatSurface(layer), inPlane(thetaI), inPlane(thetaO),
                                                      sl::twin::baseSpecular(mirror));
                directBounded = directBounded && finite(R.diffraction) && finite(R.coatSpecular)
                    && R.diffractedWeight >= 0.0 && stam::maxComponent(R.specularScale) + R.diffractedWeight <= 1.0 + 1e-12;
            }
    }
    check(directBounded, "under point lights the diffracted weight comes out of the zero order");
}

void testPresetsRender()
{
    const auto uniform = [](stam::Vec3) { return stam::Vec3 {1.0, 1.0, 1.0}; };
    sl::twin::Material material;
    for (const auto preset : kPresets)
    {
        const auto layer = sm::diffractionLayerPreset(preset);
        const std::string name(sm::token(preset));
        const auto P = sm::toStamLayerParams(layer);
        stam::Vec3 layerLight {};
        bool allFinite = true;
        for (const double u : {0.2, 0.8})
            for (double thetaV = 0.0; thetaV < 80.0; thetaV += 20.0)
            {
                const auto S = flatSurface(layer, u, 0.5);
                const auto V = inPlane(-thetaV);
                const auto R = stam::diffractionLayerEnvironment(P, S, V, sl::twin::baseSpecular(material), 0.5, uniform);
                layerLight = layerLight + R.diffraction + R.coatSpecular;
                const auto total = sl::twin::environment(layer, material, S, V, stam::Vec3 {0.2, 0.2, 0.2}, 0.5, uniform);
                allFinite = allFinite && finite(total) && finite(R.diffraction) && finite(R.coatSpecular);
                for (double thetaI = 5.0; thetaI < 85.0; thetaI += 10.0)
                    allFinite = allFinite && finite(sl::twin::direct(&layer, material, S, inPlane(thetaI), V));
            }
        check(allFinite, "preset " + name + " renders finite values");
        check(sum(layerLight) > 0.0, "preset " + name + " adds visible light of its own");
    }
}
} // namespace

int main()
{
    testPresetsAndBounds();
    testAdmissionAndDigest();
    testWire();
    testLegacyDigests();
    testLegacyWire();
    testValidatorsAgree();
    testMaterialField();
    testFoilMapping();
    testPacking();
    testGlslContract();
    testMetalHasNoGlslTokens();
    testBackendSources();
    testGratingEquation();
    testEnvironmentGratingSpot();
    testStrengthZeroIsBase();
    testEnergy();
    testPresetsRender();
    std::cout << checks - failures << '/' << checks << " diffraction layer checks passed\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
