#pragma once

#include "AovInspectionOperationContract.h"
#include "RenderPassCompositeContract.h"
#include "VisualImportedSceneRenderOperationContract.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <string_view>

// Render Passes retains the ordinary Scene3D request. Its typed output and
// authored extent are additional metadata, not a second scene representation.
namespace typedscenepass
{
inline constexpr std::string_view kHeader = "visual.typed-scene-pass.v1";
inline constexpr std::string_view kNodeKind = "visual.3d.render.passes";
inline constexpr std::string_view kSceneKind = "visual.3d.scene.retained";
// The native scene owns eight attachments, totalling 41 bytes per pixel.
// Normal SDR presentation retains one half-float conversion/row-copy target
// and one four-byte display copy. The temporary depth-test target is released
// before those two display targets exist, so this also covers its peak.
inline constexpr std::size_t kFrameSlots = 10;
inline constexpr std::uint64_t kFrameBytesPerPixel = 53;

struct Payload
{
    renderpassoutput::Extent extent;
    renderpassoutput::Output output = renderpassoutput::Output::Color;
    visualimportedscenerender::Request scene;
};

inline bool supported(renderpassoutput::Output output) noexcept
{
    return output == renderpassoutput::Output::Color || output == renderpassoutput::Output::Depth
        || output == renderpassoutput::Output::Normal || output == renderpassoutput::Output::Motion;
}

inline int outputPort(renderpassoutput::Output output) noexcept
{
    return supported(output) ? static_cast<int>(output) + 1 : -1;
}

inline bool valid(const Payload& value)
{
    // Render Passes draw leaf 0 only, so a multi-leaf composition is refused.
    if (!supported(value.output) || !value.scene.additionalImportedScenes.empty()
        || !visualimportedscenerender::valid(value.scene)
        || !value.scene.sceneSnapshot || value.scene.imageOutput != renderpassoutput::Output::Color
        || value.scene.rawExportMask != 0 || value.scene.materialField
        || value.scene.sourceStableId == value.scene.renderStableId
        || value.scene.sourceStableId > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) + 1u
        || value.scene.renderStableId > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) + 1u)
        return false;
    // Frame-dependent materials retain their dedicated prerequisite schedule.
    // A collapsed static scene must never lose that dependency.
    if ((value.scene.material && surfacematerialbinding::hasGraphFrameInput(value.scene.material->binding))
        || (value.scene.diffractionMaterial && value.scene.diffractionMaterial->graphFrame)
        || geometrysurfacematerial::objectFrameEndpoint(value.scene.surfacePrograms)) return false;
    renderpassoutput::Description description;
    description.extent = value.extent;
    const auto requirements = renderpassoutput::requirements(value.output);
    description.attachments.push_back({value.output, requirements.format, requirements.colorSpace, value.extent});
    renderpassoutput::AdmissionFailure failure;
    return renderpassoutput::admit(description, failure).has_value();
}

inline std::string encode(const Payload& value)
{
    if (!valid(value)) return {};
    const auto scene = visualimportedscenerender::encode(value.scene);
    if (scene.empty()) return {};
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << kHeader << '\n' << value.extent.width << ' ' << value.extent.height << ' '
           << static_cast<unsigned>(value.output) << '\n' << scene;
    const auto result = stream.str();
    return result.size() <= visualimportedscenerender::kMaximumV8EncodedBytes ? result : std::string{};
}

inline bool decode(std::string_view text, Payload& destination)
{
    if (text.size() > visualimportedscenerender::kMaximumV8EncodedBytes) return false;
    std::istringstream stream{std::string(text)};
    stream.imbue(std::locale::classic());
    std::string header;
    Payload value;
    unsigned output = 0;
    if (!std::getline(stream, header) || header != kHeader
        || !(stream >> value.extent.width >> value.extent.height >> output) || stream.get() != '\n'
        || output > static_cast<unsigned>(renderpassoutput::Output::Motion)) return false;
    const auto offset = stream.tellg();
    if (offset < 0 || !visualimportedscenerender::decode(
            text.substr(static_cast<std::size_t>(offset)), value.scene)) return false;
    value.output = static_cast<renderpassoutput::Output>(output);
    if (!valid(value) || encode(value) != text) return false;
    destination = std::move(value);
    return true;
}

inline renderpasscomposite::Parameters inspectionParameters(const aovinspection::Payload& inspection)
{
    renderpasscomposite::Parameters result;
    result.mode = inspection.source == aovinspection::Source::Depth
        ? renderpasscomposite::Mode::DepthView : renderpasscomposite::Mode::NormalView;
    result.nearDepth = inspection.depthNear;
    result.farDepth = inspection.depthFar;
    result.inspectionImage = true;
    result.invertDepth = inspection.invertDepth;
    return result;
}
} // namespace typedscenepass
