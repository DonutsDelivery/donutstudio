#pragma once

#include "MaterialFieldBinding.h"
#include "SurfaceMaterialIR.h"

#include <limits>

namespace surfacematerialfield {
// A program carries as many fields as the graph connects.
inline constexpr std::size_t kMaximumBindings = std::numeric_limits<std::size_t>::max();
inline constexpr const char* kTargets[] {"Base Red", "Base Green", "Base Blue", "Metallic", "Roughness",
    "Emission Red", "Emission Green", "Emission Blue", "Opacity", "Transmission", "IOR", "Clearcoat", "Dispersion",
    "Diffraction Period", "Diffraction Groove Angle", "Diffraction Strength", "Diffraction Mask"};
// Targets from here on drive the program's diffraction layer. A program
// without a layer ignores them.
inline constexpr std::uint32_t kFirstDiffractionTarget = 13;
struct Binding final {
  materialfield::Binding sample;
  bool objectIndex = false;
};
inline bool valid(const Binding& binding, std::string& error) {
  if (binding.sample.target >= std::size(kTargets)
      || (binding.objectIndex && (binding.sample.reduction != materialfield::Reduction::index || binding.sample.index != 0))) {
    error = "Surface Material Field has an invalid target or object-index sample"; return false;
  }
  return materialfield::admit(binding.sample,error,true).has_value();
}
// Diffraction targets add the sampled amount to the layer: period in nm,
// groove angle in degrees, strength and mask as plain values, each clamped to
// the layer bounds.
inline void evaluateDiffraction(std::uint32_t target, double amount,
    surfacematerial::DiffractionLayerDescription& layer) {
  using namespace surfacematerial;
  if (target == kFirstDiffractionTarget) {
    layer.periodNm = static_cast<float>(std::clamp(static_cast<double>(layer.periodNm)+amount,
        static_cast<double>(kDiffractionLayerMinimumPeriodNm), static_cast<double>(kDiffractionLayerMaximumPeriodNm)));
    layer.secondPeriodNm = static_cast<float>(std::clamp(static_cast<double>(layer.secondPeriodNm)+amount,
        static_cast<double>(kDiffractionLayerMinimumPeriodNm), static_cast<double>(kDiffractionLayerMaximumPeriodNm)));
  } else if (target == kFirstDiffractionTarget + 1) {
    layer.grooveAngle = static_cast<float>(std::clamp(static_cast<double>(layer.grooveAngle)+amount*0.017453292519943295,
        -static_cast<double>(kDiffractionLayerMaximumAngle), static_cast<double>(kDiffractionLayerMaximumAngle)));
  } else if (target == kFirstDiffractionTarget + 2) {
    layer.strength = static_cast<float>(std::clamp(static_cast<double>(layer.strength)+amount,
        0.0, static_cast<double>(kDiffractionLayerMaximumStrength)));
  } else {
    layer.mask = static_cast<float>(std::clamp(static_cast<double>(layer.mask)+amount, 0.0, 1.0));
  }
}
inline bool evaluate(const Binding& binding, const videowire::geometry::RuntimeFieldEvaluation& runtime,
    std::size_t objectIndex, surfacematerial::SurfacePbrParameterBlock& parameters,
    std::array<float,3>& timeMixEndColor, std::string& error,
    surfacematerial::DiffractionLayerDescription* layer = nullptr) {
  if (!valid(binding,error)) return false;
  auto source = binding.sample;
  if (binding.objectIndex) {
    if (objectIndex > std::numeric_limits<std::uint32_t>::max()) {
      error = "Surface Material Field object index does not fit its 32-bit sample index"; return false;
    }
    source.index = static_cast<std::uint32_t>(objectIndex);
  }
  double amount = 0;
  if (!materialfield::sample(source,runtime,amount,error,true)) return false;
  const auto target = source.target;
  if (target >= kFirstDiffractionTarget) {
    if (layer) evaluateDiffraction(target, amount, *layer);
    return true;
  }
  float* output = target < 4 ? &parameters.baseColorMetallic[target]
      : target == 4 ? &parameters.emissionRoughness[3]
      : target < 8 ? &parameters.emissionRoughness[target-5]
      : target == 8 ? &parameters.normalOpacity[3] : &parameters.transmissionIorClearcoat[target-9];
  // Dispersion stops only at the float32 parameter lane's largest finite value.
  const double low = target == 10 ? surfacematerial::kMinimumMaterialIor : 0;
  const double high = target == 10 ? surfacematerial::kMaximumMaterialIor : target == 12 ? static_cast<double>(std::numeric_limits<float>::max()) : target < 3 || (target >= 5 && target < 8) ? 64 : 1;
  *output = static_cast<float>(std::clamp(static_cast<double>(*output)+amount,low,high));
  if (target < 3) timeMixEndColor[target] = static_cast<float>(std::clamp(static_cast<double>(timeMixEndColor[target])+amount,low,high));
  return true;
}
} // namespace surfacematerialfield
