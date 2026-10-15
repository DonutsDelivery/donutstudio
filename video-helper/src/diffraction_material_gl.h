#pragma once

#include "diffraction_material_execution.h"
#include "../../shared/DiffractionLightingPlan.h"
#include "gl_loader.h"

#include <array>
#include <cstdint>
#include <string>

namespace diffractionmaterial
{

// The executor owns these OpenGL resources. They remain valid until the next
// execute call or shutdown and contain no CPU-rendered substitute.
struct OpenGlPhysicalDiffractionFrame final
{
    std::uint64_t generation = 0;
    unsigned framebuffer = 0;
    unsigned linearSrgbTexture = 0;
    unsigned energyLedgerTexture = 0;
    unsigned orderLedgerTexture = 0;
    unsigned directionMomentTexture = 0;
    std::array<float, 4> linearSrgb {};
    std::array<float, 4> energyLedger {};
    std::array<float, 4> orderLedger {};
    std::array<float, 4> directionMoment {};
};

// Strict physical backend checkpoint. It supports the canonical eight visible
// samples, one-dimensional and crossed two-dimensional binary-rectangular,
// bounded sinusoidal and blazed-sawtooth gratings, incoherent dielectric coating,
// RMS-height attenuation, RMS-slope broadening, and one direct spectral light.
// Alternate sampling layouts, environments, and indirect transport reject rather
// than approximate.
class OpenGlPhysicalDiffractionExecutor final
{
public:
    bool initialize(arbitgl::GlFuncs* gl, std::string& error);
    bool execute(const AdmittedDiffractionMaterialIR& material,
                 const SpectralIncidentLight& light,
                 OpenGlPhysicalDiffractionFrame& output,
                 std::string& error);
    bool executeAtMaterialUv(const AdmittedDiffractionMaterialIR& material,
                             const SpectralIncidentLight& light,
                             const std::array<float, 2>& materialUv,
                             OpenGlPhysicalDiffractionFrame& output,
                             std::string& error);
    bool executeTransport(const AdmittedDiffractionMaterialIR& material,
                          const AdmittedLightingPlan& lighting,
                          OpenGlPhysicalDiffractionFrame& output,
                          TransportEnergyReceipt& receipt,
                          std::string& error);
    void shutdown() noexcept;

    bool ready() const noexcept { return gl_ != nullptr && program_ != 0; }

private:
    bool validateCheckpoint(const AdmittedDiffractionMaterialIR& material,
                            const SpectralIncidentLight& light,
                            std::string& error) const noexcept;

    arbitgl::GlFuncs* gl_ = nullptr;
    unsigned program_ = 0;
    unsigned vao_ = 0;
    unsigned framebuffer_ = 0;
    std::array<unsigned, 4> textures_ {};
    std::uint64_t generation_ = 0;
};
} // namespace diffractionmaterial
