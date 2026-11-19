#pragma once

#include "SurfaceMaterialIR.h"
#include "Visual3DScene.h"
#include "VertexModifierIr.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace surfacematerialbinding
{
inline constexpr std::uint32_t kWireVersion = 1;
inline constexpr std::uint32_t kGraphFrameWireVersion = 2;
inline constexpr std::uint32_t kReflectanceWireVersion = 3;
inline constexpr std::uint32_t kRoughGlassWireVersion = 4;
// Version 5 carries a program with the diffraction layer companion. It keeps
// version 4's rough-glass and reflectance header, then adds the layer block.
inline constexpr std::uint32_t kDiffractionLayerWireVersion = 5;
// Program, binding, texture and video counts come from the scene. These are the
// type limits, not scene caps; the device's own texture size query governs the
// image dimension, and only a real allocation failure refuses a scene.
inline constexpr std::size_t kMaximumPrograms = std::numeric_limits<std::size_t>::max();
inline constexpr std::size_t kMaximumBindings = std::numeric_limits<std::size_t>::max();
inline constexpr std::size_t kMaximumTextureBindings = std::numeric_limits<std::size_t>::max();
inline constexpr std::size_t kMaximumVideoResources = std::numeric_limits<std::size_t>::max();
// Five modulation semantics exist; this counts the schema, not the scene.
inline constexpr std::size_t kMaximumModulationInputs = 5;
inline constexpr std::uint32_t kMaximumImageDimension = std::numeric_limits<std::uint32_t>::max();
inline constexpr std::uint64_t kMaximumVideoResourceBytes = std::numeric_limits<std::uint64_t>::max();

using VideoResourceIdentity = std::array<std::uint8_t, 16>;

inline constexpr bool hasIdentity(const VideoResourceIdentity& identity) noexcept
{
    for (const auto value : identity)
        if (value != 0)
            return true;
    return false;
}

enum class BindingTargetKind : std::uint8_t
{
    Invalid = 0,
    MaterialSlot = 1,
    ObjectOverride = 2
};

enum class TextureSourceKind : std::uint8_t
{
    Invalid = 0,
    ImportedBaseColor = 1,
    VideoResource = 2,
    GraphFrame = 3
};

enum class UnboundObjectFallback : std::uint8_t
{
    Invalid = 0,
    ImportedMaterial = 1
};

struct TextureSlotBinding
{
    std::uint16_t slot = 0;
    TextureSourceKind source = TextureSourceKind::Invalid;
    VideoResourceIdentity videoResource {};
    // The existing GraphDoc endpoint, after graph flattening. This is an
    // authored dependency, not a decoder identity or a native texture handle.
    struct GraphFrameEndpoint
    {
        std::int32_t node = -1;
        std::int32_t port = -1;
        bool operator==(const GraphFrameEndpoint& other) const noexcept { return node == other.node && port == other.port; }
        bool operator<(const GraphFrameEndpoint& other) const noexcept { return node < other.node || (node == other.node && port < other.port); }
    };
    std::optional<GraphFrameEndpoint> graphFrame {};
};

struct MaterialBinding
{
    BindingTargetKind targetKind = BindingTargetKind::Invalid;
    HarmonicMIDI::grid::SceneMaterialId materialSlot {};
    HarmonicMIDI::grid::SceneObjectId object {};
    std::string surfaceMaterialDigest;
    std::uint64_t surfaceMaterialRevision = 0;
    std::vector<TextureSlotBinding> textures;
};

inline bool hasGraphFrameInput(const MaterialBinding& binding) noexcept
{
    for (const auto& texture : binding.textures)
        if (texture.source == TextureSourceKind::GraphFrame)
            return true;
    return false;
}

inline constexpr std::uint32_t requestVersion(bool hasFrame, bool hasReflectance, std::uint32_t roughGlassScatteringVersion=1,
    bool hasDiffractionLayer=false) noexcept
{
    return hasDiffractionLayer ? kDiffractionLayerWireVersion
                          : roughGlassScatteringVersion!=1 ? kRoughGlassWireVersion : hasReflectance ? kReflectanceWireVersion
                          : hasFrame ? kGraphFrameWireVersion : kWireVersion;
}

// Exact product request emitted from one compiled Visual Graph material terminal.
// It carries no asset path, programmable source, capability grant, or CPU image.
struct ImportedSceneMaterialRequest
{
    std::uint32_t version = kWireVersion;
    std::uint32_t roughGlassScatteringVersion = 1;
    HarmonicMIDI::grid::Scene3DId scene {};
    std::uint64_t sceneRevision = 0;
    std::uint64_t structuralRevision = 0;
    std::uint64_t evaluationRevision = 0;
    std::uint64_t programRevision = 0;
    // Owns the exact immutable scene bytes used to lower and bind this request.
    // Material programs and bindings are value-owned below; imported texture
    // texels remain owned by this shared snapshot rather than borrowed from an
    // editor document or decoder scratch storage.
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> sceneSnapshot;
    surfacematerial::ProgramDescription program;
    // Optional graph-native vertex stage compiled with the surface stage for
    // the same imported-object draw. It contains admitted data only.
    std::optional<videowire::VertexModifierIr> vertexModifier;
    MaterialBinding binding;
};

inline bool validRoughGlassVersion(const ImportedSceneMaterialRequest& request) noexcept
{
    // 0 leaves the choice to the renderer, and only a closed volume has one.
    return request.roughGlassScatteringVersion==1 || (request.roughGlassScatteringVersion<=2
        && request.program.transport==surfacematerial::TransportMode::ClosedVolume);
}

// These are immutable scalar receipts supplied by the established modulation
// owner. Geometry inputs and timeline time remain renderer-owned built-ins.
struct ModulationInputReceipt
{
    surfacematerial::InputSemantic semantic = surfacematerial::InputSemantic::Invalid;
    float value = 0.0f;
    std::uint64_t sourceRevision = 0;
};

struct AdmissionLimits
{
    std::size_t programs = kMaximumPrograms;
    std::size_t bindings = kMaximumBindings;
    std::size_t textureBindings = kMaximumTextureBindings;
    std::size_t videoResources = kMaximumVideoResources;
    std::size_t modulationInputs = kMaximumModulationInputs;
    std::uint32_t maximumImageDimension = kMaximumImageDimension;
    std::uint64_t videoResourceBytes = kMaximumVideoResourceBytes;
};

inline constexpr bool isModulationSemantic(
    surfacematerial::InputSemantic semantic) noexcept
{
    using Semantic = surfacematerial::InputSemantic;
    return semantic == Semantic::AudioLevel || semantic == Semantic::AudioBass
        || semantic == Semantic::AudioMid || semantic == Semantic::AudioTreble
        || semantic == Semantic::AudioBeat;
}

static_assert(static_cast<std::uint8_t>(BindingTargetKind::ObjectOverride) == 2,
              "Material binding target IDs are stable and append-only");
static_assert(static_cast<std::uint8_t>(TextureSourceKind::VideoResource) == 2,
              "Material texture source IDs are stable and append-only");
static_assert(static_cast<std::uint8_t>(TextureSourceKind::GraphFrame) == 3,
              "Authored graph Frame references do not renumber native texture sources");
static_assert(static_cast<std::uint8_t>(UnboundObjectFallback::ImportedMaterial) == 1,
              "Imported-material fallback ID is stable");
} // namespace surfacematerialbinding
