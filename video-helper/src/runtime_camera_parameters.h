#pragma once

// The Render node's camera runtime parameters (cameraX/Y/Z, cameraRotateX/Y/Z,
// cameraFovDegrees, cameraOrthographicHeight under "visual<renderStableId>/"), read with the clamps the
// draw applies. The same reader serves the current frame and the camera one
// frame earlier, so Motion and motion blur see keyframed or modulated camera moves.

#include "../../shared/Visual3DScene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace videohelper::runtimecamera
{
struct Values final
{
    std::array<float, 3> translationOffset {};
    std::array<float, 3> rotationDegrees {};
    // Zero keeps the authored vertical field of view. A negative value is an
    // orthographic height, as the runtime camera record carries it.
    float verticalFovDegrees = 0.0f;
};

inline constexpr const char* kParameterNames[] = {
    "cameraX", "cameraY", "cameraZ", "cameraRotateX", "cameraRotateY", "cameraRotateZ", "cameraFovDegrees",
    "cameraOrthographicHeight"
};

// The tallest orthographic view the runtime camera admits
// (arbitgpu::kMaximumRuntimeOrthographicHeight).
inline constexpr float kMaximumOrthographicHeight = 10000000.0f;

inline std::string prefix(std::uint64_t renderStableId)
{
    return "visual" + std::to_string(renderStableId) + "/";
}

// An evaluator that returns no camera parameter has no history for the camera.
inline bool containsAny(const std::map<std::string, double>& parameters, const std::string& prefix)
{
    return std::any_of(std::begin(kParameterNames), std::end(kParameterNames),
        [&](const char* name) { return parameters.count(prefix + name) != 0; });
}

inline Values read(const std::map<std::string, double>& parameters, const std::string& prefix)
{
    const auto value = [&](const char* name, double minimum, double maximum)
    {
        const auto found = parameters.find(prefix + name);
        const double candidate = found != parameters.end() && std::isfinite(found->second) ? found->second : 0.0;
        return static_cast<float>(std::clamp(candidate, minimum, maximum));
    };
    Values result;
    result.translationOffset = { value("cameraX", -1000000.0, 1000000.0),
        value("cameraY", -1000000.0, 1000000.0), value("cameraZ", -1000000.0, 1000000.0) };
    result.rotationDegrees = { value("cameraRotateX", -360.0, 360.0),
        value("cameraRotateY", -360.0, 360.0), value("cameraRotateZ", -360.0, 360.0) };
    // Positive values under the narrowest lens round up to it. Zero or negative
    // values keep the authored field of view.
    const float fieldOfView = value("cameraFovDegrees", 0.0, 179.0);
    result.verticalFovDegrees = fieldOfView > 0.0f
        ? std::max(fieldOfView, HarmonicMIDI::grid::kMinimumCameraVerticalFovDegrees) : 0.0f;
    const float orthographicHeight = value("cameraOrthographicHeight", 0.0, kMaximumOrthographicHeight);
    if (orthographicHeight > 0.0f)
        result.verticalFovDegrees = -orthographicHeight;
    return result;
}

// The project time of the frame before the one at projectSeconds.
inline bool previousFrameSeconds(double projectSeconds, double framesPerSecond, double& previous)
{
    if (!std::isfinite(projectSeconds) || !std::isfinite(framesPerSecond) || !(framesPerSecond > 0.0))
        return false;
    previous = projectSeconds - 1.0 / framesPerSecond;
    return true;
}

template <typename Layer, typename = void>
struct HasMaterialInputsAt : std::false_type {};
template <typename Layer>
struct HasMaterialInputsAt<Layer, std::void_t<decltype(std::declval<Layer&>().materialInputsAt)>> : std::true_type {};

// Evaluates the Render node's camera parameters one frame earlier with the
// layer's historical evaluator. An evaluator that cannot reach that time, or
// returns no camera parameter, leaves the camera to the adjacent predecessor.
template <typename Layer>
std::optional<Values> previous(const Layer& layer, const std::string& prefix,
    double projectSeconds, double framesPerSecond)
{
    if constexpr (HasMaterialInputsAt<Layer>::value)
    {
        double at = 0.0;
        if (!layer.materialInputsAt || !previousFrameSeconds(projectSeconds, framesPerSecond, at))
            return std::nullopt;
        Layer sampled;
        std::map<std::string, double> values;
        std::string failure;
        if (!layer.materialInputsAt(at, sampled, values, failure) || !containsAny(values, prefix))
            return std::nullopt;
        return read(values, prefix);
    }
    else
    {
        (void) layer; (void) prefix; (void) projectSeconds; (void) framesPerSecond;
        return std::nullopt;
    }
}
} // namespace videohelper::runtimecamera
