#pragma once

#include "diffraction_material_execution.h"
#include "../../shared/DiffractionLightingPlan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace diffractionmaterial
{
struct MetalPhysicalDiffractionCapabilities final
{
    bool available = false;
    std::string backend;
    std::string device;
    std::size_t spectralSamples = 0;
    std::uint8_t maximumOrder = 0;
    bool canonicalWavelengthLayout = false;
    bool binaryRectangularProfile = false;
    bool sinusoidalProfile = false;
    bool blazedSawtoothProfile = false;
    bool crossedTwoDimensionalLattice = false;
    bool incoherentCoating = false;
    bool rmsHeightAttenuation = false;
    bool roughnessBroadening = false;
    std::string error;
};

// Backend-local identities and the validation readback for one submitted 1x1
// physical evaluation. The executor owns every image and view identity.
struct MetalPhysicalDiffractionFrame final
{
    std::uint64_t generation = 0;
    std::array<std::uintptr_t, 4> imageHandles {};
    std::array<std::uintptr_t, 4> textureViewHandles {};
    std::array<std::uintptr_t, 4> metalTextureHandles {};
    std::array<float, 4> linearSrgb {};
    std::array<float, 4> energyLedger {};
    std::array<float, 4> orderLedger {};
    std::array<float, 4> directionMoment {};
};

MetalPhysicalDiffractionCapabilities queryMetalPhysicalDiffractionCapabilities();

// Strict Metal/Sokol executor for the same canonical eight-wavelength, binary,
// bounded sinusoidal, blazed-sawtooth, crossed-lattice, coating, and roughness
// checkpoint as OpenGL. Unsupported material data rejects before command
// submission. There is no RGB, HSV, thin-film, ramp, or CPU path.
class MetalPhysicalDiffractionExecutor final
{
public:
    MetalPhysicalDiffractionExecutor();
    ~MetalPhysicalDiffractionExecutor();

    MetalPhysicalDiffractionExecutor(const MetalPhysicalDiffractionExecutor&) = delete;
    MetalPhysicalDiffractionExecutor& operator=(
        const MetalPhysicalDiffractionExecutor&) = delete;

    bool initialize(std::string& error);
    bool execute(const AdmittedDiffractionMaterialIR& material,
                 const SpectralIncidentLight& light,
                 MetalPhysicalDiffractionFrame& output,
                 std::string& error);
    bool executeAtMaterialUv(const AdmittedDiffractionMaterialIR& material,
                             const SpectralIncidentLight& light,
                             const std::array<float, 2>& materialUv,
                             MetalPhysicalDiffractionFrame& output,
                             std::string& error);
    bool executeTransport(const AdmittedDiffractionMaterialIR& material,
                          const AdmittedLightingPlan& lighting,
                          MetalPhysicalDiffractionFrame& output,
                          TransportEnergyReceipt& receipt,
                          std::string& error);
    void shutdown() noexcept;
    bool ready() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace diffractionmaterial
