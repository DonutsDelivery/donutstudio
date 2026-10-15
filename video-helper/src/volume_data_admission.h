#pragma once

#include "../../shared/VisualVolumeData.h"
#include "sha256.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace videohelper::volume
{
struct VolumeAdmissionLimits final
{
    std::uint32_t maxDimension = 2048;
    std::size_t maxVoxelCount = 134217728;
    std::uint32_t maxBrickEdge = 64;
    std::size_t maxBrickCount = 65536;
    std::size_t maxPayloadBytes = 512u * 1024u * 1024u;
};

// Callers must provide capabilities from the renderer that will own upload and
// raymarch execution. General GPU availability does not satisfy this contract.
struct VolumeRendererCapabilities final
{
    bool nativeGpuAvailable = false;
    bool volumeRaymarch = false;
    bool denseVolumeUpload = false;
    bool sparseBrickUpload = false;
    std::uint32_t maxTexture3DDimension = 0;
    std::uint32_t maxBrickEdge = 0;
    std::size_t maxSparseBrickCount = 0;
    std::size_t maxVolumeBytes = 0;
};

struct AdmittedSparseBrick final
{
    std::uint32_t brickX = 0;
    std::uint32_t brickY = 0;
    std::uint32_t brickZ = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
    std::size_t byteOffset = 0;
    std::size_t byteCount = 0;
};

class AdmittedVolume final
{
public:
    AdmittedVolume (const AdmittedVolume&) = default;
    AdmittedVolume (AdmittedVolume&&) noexcept = default;
    AdmittedVolume& operator= (const AdmittedVolume&) = default;
    AdmittedVolume& operator= (AdmittedVolume&&) noexcept = default;

    videowire::VolumeStorage storage() const noexcept { return storage_; }
    videowire::VolumeVoxelFormat format() const noexcept { return format_; }
    const videowire::VolumeDimensions& dimensions() const noexcept { return dimensions_; }
    const videowire::VolumeBounds& bounds() const noexcept { return bounds_; }
    const videowire::VolumeTransform& transform() const noexcept { return transform_; }
    std::uint32_t brickEdge() const noexcept { return brickEdge_; }
    std::size_t voxelCount() const noexcept { return voxelCount_; }
    const std::vector<AdmittedSparseBrick>& bricks() const noexcept { return bricks_; }
    const std::vector<std::uint8_t>& bytes() const noexcept { return bytes_; }
    const std::string& cacheIdentity() const noexcept { return cacheIdentity_; }

private:
    friend std::optional<AdmittedVolume> admitDenseVolume (
        const videowire::DenseVolumeDescriptor&, const VolumeAdmissionLimits&,
        const VolumeRendererCapabilities&, std::string&);
    friend std::optional<AdmittedVolume> admitSparseVolume (
        const videowire::SparseVolumeDescriptor&, const VolumeAdmissionLimits&,
        const VolumeRendererCapabilities&, std::string&);

    AdmittedVolume (videowire::VolumeStorage storage,
                    videowire::VolumeVoxelFormat format,
                    videowire::VolumeDimensions dimensions,
                    videowire::VolumeBounds bounds,
                    videowire::VolumeTransform transform,
                    std::uint32_t brickEdge,
                    std::size_t voxelCount,
                    std::vector<AdmittedSparseBrick> bricks,
                    std::vector<std::uint8_t> bytes,
                    std::string cacheIdentity)
        : storage_ (storage), format_ (format), dimensions_ (dimensions), bounds_ (bounds),
          transform_ (transform), brickEdge_ (brickEdge), voxelCount_ (voxelCount),
          bricks_ (std::move (bricks)), bytes_ (std::move (bytes)),
          cacheIdentity_ (std::move (cacheIdentity))
    {
    }

    videowire::VolumeStorage storage_ = videowire::VolumeStorage::dense;
    videowire::VolumeVoxelFormat format_ = videowire::VolumeVoxelFormat::densityU8;
    videowire::VolumeDimensions dimensions_ {};
    videowire::VolumeBounds bounds_ {};
    videowire::VolumeTransform transform_ {};
    std::uint32_t brickEdge_ = 0;
    std::size_t voxelCount_ = 0;
    std::vector<AdmittedSparseBrick> bricks_;
    std::vector<std::uint8_t> bytes_;
    std::string cacheIdentity_;
};

namespace detail
{
inline bool checkedMultiply (std::size_t left, std::size_t right, std::size_t& result) noexcept
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) return false;
    result = left * right;
    return true;
}

inline bool checkedAdd (std::size_t left, std::size_t right, std::size_t& result) noexcept
{
    if (right > std::numeric_limits<std::size_t>::max() - left) return false;
    result = left + right;
    return true;
}

inline std::size_t bytesPerVoxel (videowire::VolumeVoxelFormat format) noexcept
{
    switch (format)
    {
        case videowire::VolumeVoxelFormat::densityU8: return 1;
        case videowire::VolumeVoxelFormat::densityF16: return 2;
        case videowire::VolumeVoxelFormat::densityF32: return 4;
    }
    return 0;
}

inline bool validateLimits (const VolumeAdmissionLimits& limits, std::string& error)
{
    if (limits.maxDimension == 0 || limits.maxVoxelCount == 0 || limits.maxBrickEdge == 0
        || limits.maxBrickCount == 0 || limits.maxPayloadBytes == 0)
    {
        error = "volume admission limits are invalid";
        return false;
    }
    return true;
}

inline bool validateCapabilities (const VolumeRendererCapabilities& capabilities,
                                  videowire::VolumeStorage storage,
                                  std::string& error)
{
    if (! capabilities.nativeGpuAvailable || ! capabilities.volumeRaymarch)
    {
        error = "renderer lacks native GPU volume raymarch capability";
        return false;
    }
    if ((storage == videowire::VolumeStorage::dense && ! capabilities.denseVolumeUpload)
        || (storage == videowire::VolumeStorage::sparseBricks
            && ! capabilities.sparseBrickUpload))
    {
        error = storage == videowire::VolumeStorage::dense
            ? "renderer lacks dense volume upload capability"
            : "renderer lacks sparse brick upload capability";
        return false;
    }
    if (capabilities.maxTexture3DDimension == 0 || capabilities.maxVolumeBytes == 0
        || (storage == videowire::VolumeStorage::sparseBricks
            && (capabilities.maxBrickEdge == 0 || capabilities.maxSparseBrickCount == 0)))
    {
        error = "renderer reported invalid volume capability limits";
        return false;
    }
    return true;
}

inline bool validateDimensions (const videowire::VolumeDimensions& dimensions,
                                const VolumeAdmissionLimits& limits,
                                const VolumeRendererCapabilities& capabilities,
                                std::size_t& voxelCount,
                                std::string& error)
{
    if (dimensions.width == 0 || dimensions.height == 0 || dimensions.depth == 0)
    {
        error = "volume dimensions must be non-zero";
        return false;
    }
    const auto maximumDimension = std::min (limits.maxDimension,
                                            capabilities.maxTexture3DDimension);
    if (dimensions.width > maximumDimension || dimensions.height > maximumDimension
        || dimensions.depth > maximumDimension)
    {
        error = "volume dimension budget exceeded";
        return false;
    }
    std::size_t plane = 0;
    if (! checkedMultiply (static_cast<std::size_t> (dimensions.width),
                           static_cast<std::size_t> (dimensions.height), plane)
        || ! checkedMultiply (plane, static_cast<std::size_t> (dimensions.depth), voxelCount))
    {
        error = "volume voxel count overflow";
        return false;
    }
    if (voxelCount > limits.maxVoxelCount)
    {
        error = "volume voxel budget exceeded";
        return false;
    }
    return true;
}

inline bool finiteVec3 (const videowire::VolumeVec3& value) noexcept
{
    return std::isfinite (value.x) && std::isfinite (value.y) && std::isfinite (value.z);
}

inline bool validatePlacement (const videowire::VolumeBounds& bounds,
                               const videowire::VolumeTransform& transform,
                               std::string& error)
{
    if (! finiteVec3 (bounds.minimum) || ! finiteVec3 (bounds.maximum)
        || bounds.minimum.x >= bounds.maximum.x || bounds.minimum.y >= bounds.maximum.y
        || bounds.minimum.z >= bounds.maximum.z)
    {
        error = "volume bounds must be finite and strictly increasing";
        return false;
    }
    if (! std::all_of (transform.localToWorld.begin(), transform.localToWorld.end(),
                       [] (float value) { return std::isfinite (value); }))
    {
        error = "volume transform contains a non-finite value";
        return false;
    }
    return true;
}

inline void hashU8 (Sha256& hash, std::uint8_t value)
{
    hash.update (&value, sizeof (value));
}

inline void hashU32 (Sha256& hash, std::uint32_t value)
{
    const std::uint8_t bytes[] {
        static_cast<std::uint8_t> (value >> 24u),
        static_cast<std::uint8_t> (value >> 16u),
        static_cast<std::uint8_t> (value >> 8u),
        static_cast<std::uint8_t> (value)
    };
    hash.update (bytes, sizeof (bytes));
}

inline void hashU64 (Sha256& hash, std::uint64_t value)
{
    std::uint8_t bytes[8];
    for (int index = 0; index < 8; ++index)
        bytes[index] = static_cast<std::uint8_t> (value >> (56 - index * 8));
    hash.update (bytes, sizeof (bytes));
}

inline void hashFloat (Sha256& hash, float value)
{
    static_assert (sizeof (float) == sizeof (std::uint32_t),
                   "volume cache identity requires 32-bit floats");
    static_assert (std::numeric_limits<float>::is_iec559,
                   "volume cache identity requires IEEE-754 floats");
    if (value == 0.0f) value = 0.0f;
    std::uint32_t bits = 0;
    std::memcpy (&bits, &value, sizeof (bits));
    hashU32 (hash, bits);
}

inline void hashPlacement (Sha256& hash, const videowire::VolumeBounds& bounds,
                           const videowire::VolumeTransform& transform)
{
    hashFloat (hash, bounds.minimum.x);
    hashFloat (hash, bounds.minimum.y);
    hashFloat (hash, bounds.minimum.z);
    hashFloat (hash, bounds.maximum.x);
    hashFloat (hash, bounds.maximum.y);
    hashFloat (hash, bounds.maximum.z);
    for (auto value : transform.localToWorld) hashFloat (hash, value);
}

inline void hashCommon (Sha256& hash, videowire::VolumeStorage storage,
                        videowire::VolumeVoxelFormat format,
                        const videowire::VolumeDimensions& dimensions,
                        const videowire::VolumeBounds& bounds,
                        const videowire::VolumeTransform& transform)
{
    hashU32 (hash, videowire::kVisualVolumeSchemaVersion);
    hashU8 (hash, static_cast<std::uint8_t> (storage));
    hashU8 (hash, static_cast<std::uint8_t> (format));
    hashU32 (hash, dimensions.width);
    hashU32 (hash, dimensions.height);
    hashU32 (hash, dimensions.depth);
    hashPlacement (hash, bounds, transform);
}

inline std::string denseCacheIdentity (videowire::VolumeVoxelFormat format,
                                       const videowire::VolumeDimensions& dimensions,
                                       const videowire::VolumeBounds& bounds,
                                       const videowire::VolumeTransform& transform,
                                       const std::vector<std::uint8_t>& bytes)
{
    Sha256 hash;
    hashCommon (hash, videowire::VolumeStorage::dense, format, dimensions, bounds, transform);
    hashU64 (hash, static_cast<std::uint64_t> (bytes.size()));
    hash.update (bytes.data(), bytes.size());
    return hash.finishHex();
}

inline std::string sparseCacheIdentity (videowire::VolumeVoxelFormat format,
                                        const videowire::VolumeDimensions& dimensions,
                                        const videowire::VolumeBounds& bounds,
                                        const videowire::VolumeTransform& transform,
                                        std::uint32_t brickEdge,
                                        const std::vector<AdmittedSparseBrick>& bricks,
                                        const std::vector<std::uint8_t>& bytes)
{
    Sha256 hash;
    hashCommon (hash, videowire::VolumeStorage::sparseBricks, format,
                dimensions, bounds, transform);
    hashU32 (hash, brickEdge);
    hashU64 (hash, static_cast<std::uint64_t> (bricks.size()));
    for (const auto& brick : bricks)
    {
        hashU32 (hash, brick.brickX);
        hashU32 (hash, brick.brickY);
        hashU32 (hash, brick.brickZ);
        hashU32 (hash, brick.width);
        hashU32 (hash, brick.height);
        hashU32 (hash, brick.depth);
        hashU64 (hash, static_cast<std::uint64_t> (brick.byteCount));
        hash.update (bytes.data() + brick.byteOffset, brick.byteCount);
    }
    return hash.finishHex();
}

inline bool validateSchemaAndFormat (std::uint32_t schemaVersion,
                                     videowire::VolumeVoxelFormat format,
                                     std::string& error)
{
    if (schemaVersion != videowire::kVisualVolumeSchemaVersion)
    {
        error = "volume schema version is unsupported";
        return false;
    }
    if (bytesPerVoxel (format) == 0)
    {
        error = "volume voxel format is unsupported";
        return false;
    }
    return true;
}
} // namespace detail

inline std::optional<AdmittedVolume> admitDenseVolume (
    const videowire::DenseVolumeDescriptor& source,
    const VolumeAdmissionLimits& limits,
    const VolumeRendererCapabilities& capabilities,
    std::string& error)
{
    error.clear();
    if (! detail::validateLimits (limits, error)
        || ! detail::validateCapabilities (capabilities, videowire::VolumeStorage::dense, error)
        || ! detail::validateSchemaAndFormat (source.schemaVersion, source.format, error))
        return std::nullopt;

    std::size_t voxelCount = 0;
    if (! detail::validateDimensions (source.dimensions, limits, capabilities, voxelCount, error)
        || ! detail::validatePlacement (source.bounds, source.transform, error))
        return std::nullopt;

    std::size_t expectedBytes = 0;
    if (! detail::checkedMultiply (voxelCount, detail::bytesPerVoxel (source.format), expectedBytes))
    {
        error = "dense volume byte count overflow";
        return std::nullopt;
    }
    const auto memoryBudget = std::min (limits.maxPayloadBytes, capabilities.maxVolumeBytes);
    if (expectedBytes > memoryBudget)
    {
        error = "dense volume memory budget exceeded";
        return std::nullopt;
    }
    if (source.voxels.size() != expectedBytes)
    {
        error = "dense volume payload size does not match its dimensions and format";
        return std::nullopt;
    }

    try
    {
        auto bytes = source.voxels;
        auto identity = detail::denseCacheIdentity (source.format, source.dimensions,
                                                    source.bounds, source.transform, bytes);
        return AdmittedVolume (videowire::VolumeStorage::dense, source.format,
                               source.dimensions, source.bounds, source.transform, 0,
                               voxelCount, {}, std::move (bytes), std::move (identity));
    }
    catch (const std::bad_alloc&)
    {
        error = "dense volume allocation failed within admitted limits";
        return std::nullopt;
    }
}

inline std::optional<AdmittedVolume> admitSparseVolume (
    const videowire::SparseVolumeDescriptor& source,
    const VolumeAdmissionLimits& limits,
    const VolumeRendererCapabilities& capabilities,
    std::string& error)
{
    error.clear();
    if (! detail::validateLimits (limits, error)
        || ! detail::validateCapabilities (capabilities,
                                           videowire::VolumeStorage::sparseBricks, error)
        || ! detail::validateSchemaAndFormat (source.schemaVersion, source.format, error))
        return std::nullopt;

    std::size_t voxelCount = 0;
    if (! detail::validateDimensions (source.dimensions, limits, capabilities, voxelCount, error)
        || ! detail::validatePlacement (source.bounds, source.transform, error))
        return std::nullopt;

    const auto maximumBrickEdge = std::min (limits.maxBrickEdge, capabilities.maxBrickEdge);
    if (source.brickEdge == 0 || source.brickEdge > maximumBrickEdge)
    {
        error = "sparse volume brick dimension budget exceeded";
        return std::nullopt;
    }
    const auto maximumBricks = std::min (limits.maxBrickCount,
                                         capabilities.maxSparseBrickCount);
    if (source.bricks.size() > maximumBricks)
    {
        error = "sparse volume brick count budget exceeded";
        return std::nullopt;
    }

    const auto gridWidth = 1u + (source.dimensions.width - 1u) / source.brickEdge;
    const auto gridHeight = 1u + (source.dimensions.height - 1u) / source.brickEdge;
    const auto gridDepth = 1u + (source.dimensions.depth - 1u) / source.brickEdge;
    std::vector<const videowire::SparseVolumeBrick*> sorted;
    try
    {
        sorted.reserve (source.bricks.size());
        for (const auto& brick : source.bricks) sorted.push_back (&brick);
        std::sort (sorted.begin(), sorted.end(), [] (const auto* left, const auto* right)
        {
            return std::tie (left->brickZ, left->brickY, left->brickX)
                 < std::tie (right->brickZ, right->brickY, right->brickX);
        });
    }
    catch (const std::bad_alloc&)
    {
        error = "sparse volume descriptor allocation failed within admitted limits";
        return std::nullopt;
    }

    std::vector<AdmittedSparseBrick> admittedBricks;
    std::vector<std::uint8_t> bytes;
    const auto bytesPerVoxel = detail::bytesPerVoxel (source.format);
    const auto memoryBudget = std::min (limits.maxPayloadBytes, capabilities.maxVolumeBytes);
    try
    {
        admittedBricks.reserve (sorted.size());
        for (std::size_t index = 0; index < sorted.size(); ++index)
        {
            const auto& brick = *sorted[index];
            if (brick.brickX >= gridWidth || brick.brickY >= gridHeight || brick.brickZ >= gridDepth)
            {
                error = "sparse volume brick coordinate is outside the logical brick grid";
                return std::nullopt;
            }
            if (index > 0)
            {
                const auto& previous = *sorted[index - 1];
                if (brick.brickX == previous.brickX && brick.brickY == previous.brickY
                    && brick.brickZ == previous.brickZ)
                {
                    error = "sparse volume contains duplicate brick coordinates";
                    return std::nullopt;
                }
            }

            const auto firstX = brick.brickX * source.brickEdge;
            const auto firstY = brick.brickY * source.brickEdge;
            const auto firstZ = brick.brickZ * source.brickEdge;
            const auto width = std::min (source.brickEdge, source.dimensions.width - firstX);
            const auto height = std::min (source.brickEdge, source.dimensions.height - firstY);
            const auto depth = std::min (source.brickEdge, source.dimensions.depth - firstZ);
            std::size_t brickVoxelCount = 0;
            std::size_t plane = 0;
            std::size_t expectedBytes = 0;
            if (! detail::checkedMultiply (width, height, plane)
                || ! detail::checkedMultiply (plane, depth, brickVoxelCount)
                || ! detail::checkedMultiply (brickVoxelCount, bytesPerVoxel, expectedBytes))
            {
                error = "sparse volume brick byte count overflow";
                return std::nullopt;
            }
            if (brick.voxels.size() != expectedBytes)
            {
                error = "sparse volume brick payload size does not match its compact edge extent";
                return std::nullopt;
            }
            std::size_t aggregateBytes = 0;
            if (! detail::checkedAdd (bytes.size(), expectedBytes, aggregateBytes))
            {
                error = "sparse volume aggregate byte count overflow";
                return std::nullopt;
            }
            if (aggregateBytes > memoryBudget)
            {
                error = "sparse volume memory budget exceeded";
                return std::nullopt;
            }

            admittedBricks.push_back ({ brick.brickX, brick.brickY, brick.brickZ,
                                        width, height, depth, bytes.size(), expectedBytes });
            bytes.insert (bytes.end(), brick.voxels.begin(), brick.voxels.end());
        }

        auto identity = detail::sparseCacheIdentity (source.format, source.dimensions,
                                                     source.bounds, source.transform,
                                                     source.brickEdge, admittedBricks, bytes);
        return AdmittedVolume (videowire::VolumeStorage::sparseBricks, source.format,
                               source.dimensions, source.bounds, source.transform,
                               source.brickEdge, voxelCount, std::move (admittedBricks),
                               std::move (bytes), std::move (identity));
    }
    catch (const std::bad_alloc&)
    {
        error = "sparse volume allocation failed within admitted limits";
        return std::nullopt;
    }
}
} // namespace videohelper::volume
