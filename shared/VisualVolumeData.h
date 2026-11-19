#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace videowire
{
constexpr std::uint32_t kVisualVolumeSchemaVersion = 1;

enum class VolumeVoxelFormat : std::uint8_t
{
    densityU8 = 0,
    densityF16 = 1,
    densityF32 = 2
};

enum class VolumeStorage : std::uint8_t
{
    dense = 0,
    sparseBricks = 1
};

struct VolumeDimensions final
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
};

struct VolumeVec3 final
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct VolumeBounds final
{
    VolumeVec3 minimum {};
    VolumeVec3 maximum { 1.0f, 1.0f, 1.0f };
};

// Column-major local-to-world matrix. Admission requires every component to be finite.
struct VolumeTransform final
{
    std::array<float, 16> localToWorld {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
};

struct DenseVolumeDescriptor final
{
    std::uint32_t schemaVersion = kVisualVolumeSchemaVersion;
    VolumeDimensions dimensions {};
    VolumeVoxelFormat format = VolumeVoxelFormat::densityU8;
    VolumeBounds bounds {};
    VolumeTransform transform {};

    // Tightly packed x-major rows, then y slices, then z slices. The descriptor
    // embeds bytes and never grants access to a filesystem path.
    std::vector<std::uint8_t> voxels;
};

struct SparseVolumeBrick final
{
    // Coordinates address the logical brick grid, not individual voxels.
    std::uint32_t brickX = 0;
    std::uint32_t brickY = 0;
    std::uint32_t brickZ = 0;

    // Edge bricks omit cells outside the logical volume. Bytes are tightly
    // packed using the same x, y, z order as dense data.
    std::vector<std::uint8_t> voxels;
};

struct SparseVolumeDescriptor final
{
    std::uint32_t schemaVersion = kVisualVolumeSchemaVersion;
    VolumeDimensions dimensions {};
    VolumeVoxelFormat format = VolumeVoxelFormat::densityU8;
    std::uint32_t brickEdge = 0;
    VolumeBounds bounds {};
    VolumeTransform transform {};
    std::vector<SparseVolumeBrick> bricks;
};

enum class VolumeRendererRequirement : std::uint8_t
{
    nativeGpuRaymarch = 0
};

struct VolumeRenderContract final
{
    static constexpr VolumeRendererRequirement kRequirement
        = VolumeRendererRequirement::nativeGpuRaymarch;
    static constexpr bool kAllowsCpuProductionFallback = false;
};
} // namespace videowire
