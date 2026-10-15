#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

// Process-wide accounting of GPU memory held by native fixture-scene owners.
// A ResidencyCharge is taken where a GPU object is allocated and released where
// that object is destroyed, so the snapshot reports owned bytes, not requests.
// Owned bytes can undercount driver memory. The charge releases with its owner,
// but the memory can outlive it. A GL object whose owner context is not current
// survives until its context is destroyed. A Metal resource that an in-flight
// command buffer references survives until that buffer completes.
// Accounting only: nothing here admits, refuses, or caps an allocation.
//
// Bytes are nominal. They come from the descriptor values given to the allocator
// (extent, layers, mip levels, texel size, sample count) and exclude driver
// padding and alignment. Only memory-bearing objects, buffers and textures, are
// charged. Views, samplers, shaders, pipelines, programs, and vertex-array objects
// have no descriptor byte size and are not counted.
namespace arbitgpu::residency
{
enum class BackingClass : std::uint8_t
{
    fixtureScene,      // OpenGlFixtureSceneResources and MetalFixtureSceneResources
    deformation,       // MetalDeformationResources
    directGrating,     // MetalDirectGratingSourceView batch buffers
    renderPassOutput,  // MetalRenderPassOutputBackend targets
    roughCausticTile,  // caustic and rough-tile state
    environmentImage,  // HDR environment image lease (environmentimage::GpuBudget)
    other,
    count
};

inline constexpr std::size_t kBackingClassCount = static_cast<std::size_t> (BackingClass::count);

struct ResidencyClassTotals final
{
    std::uint64_t bytes = 0;
    std::uint64_t objects = 0;
};

struct ResidencySnapshot final
{
    std::array<ResidencyClassTotals, kBackingClassCount> classes {};
    std::uint64_t totalBytes = 0;
    std::uint64_t totalObjects = 0;
};

class ResidencyCharge;

class ResidencyLedger final
{
public:
    static ResidencyLedger& instance() noexcept
    {
        static ResidencyLedger ledger;
        return ledger;
    }

    // Each counter is read on its own, so a snapshot is not atomic across classes.
    ResidencySnapshot snapshot() const noexcept
    {
        ResidencySnapshot result;
        for (std::size_t index = 0; index < kBackingClassCount; ++index)
        {
            result.classes[index].bytes = slots_[index].bytes.load (std::memory_order_relaxed);
            result.classes[index].objects = slots_[index].objects.load (std::memory_order_relaxed);
            result.totalBytes += result.classes[index].bytes;
            result.totalObjects += result.classes[index].objects;
        }
        return result;
    }

private:
    friend class ResidencyCharge;

    struct Slot final
    {
        std::atomic<std::uint64_t> bytes { 0 };
        std::atomic<std::uint64_t> objects { 0 };
    };

    void charge (BackingClass backing, std::uint64_t bytes) noexcept
    {
        auto& slot = slots_[static_cast<std::size_t> (backing)];
        slot.bytes.fetch_add (bytes, std::memory_order_relaxed);
        slot.objects.fetch_add (1, std::memory_order_relaxed);
    }

    void release (BackingClass backing, std::uint64_t bytes) noexcept
    {
        auto& slot = slots_[static_cast<std::size_t> (backing)];
        slot.bytes.fetch_sub (bytes, std::memory_order_relaxed);
        slot.objects.fetch_sub (1, std::memory_order_relaxed);
    }

    std::array<Slot, kBackingClassCount> slots_ {};
};

// Move-only charge against the ledger. It releases exactly once: on reset, on
// destruction, or when a move assignment replaces it. A moved-from charge is
// inactive and releases nothing.
class ResidencyCharge final
{
public:
    ResidencyCharge() noexcept = default;

    ResidencyCharge (BackingClass backing, std::uint64_t bytes) noexcept
        : backing_ (backing), bytes_ (bytes), active_ (true)
    {
        ResidencyLedger::instance().charge (backing_, bytes_);
    }

    ResidencyCharge (const ResidencyCharge&) = delete;
    ResidencyCharge& operator= (const ResidencyCharge&) = delete;

    ResidencyCharge (ResidencyCharge&& other) noexcept
        : backing_ (other.backing_), bytes_ (other.bytes_), active_ (std::exchange (other.active_, false))
    {
    }

    ResidencyCharge& operator= (ResidencyCharge&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            backing_ = other.backing_;
            bytes_ = other.bytes_;
            active_ = std::exchange (other.active_, false);
        }
        return *this;
    }

    ~ResidencyCharge() { reset(); }

    void reset() noexcept
    {
        if (! active_)
            return;
        active_ = false;
        ResidencyLedger::instance().release (backing_, bytes_);
    }

    bool active() const noexcept { return active_; }
    std::uint64_t bytes() const noexcept { return active_ ? bytes_ : 0; }

private:
    BackingClass backing_ = BackingClass::other;
    std::uint64_t bytes_ = 0;
    bool active_ = false;
};

// Saturating arithmetic. An impossible descriptor saturates instead of wrapping
// to a small charge.
inline constexpr std::uint64_t saturatingMultiply (std::uint64_t a, std::uint64_t b) noexcept
{
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    return a != 0 && b > maximum / a ? maximum : a * b;
}

inline constexpr std::uint64_t saturatingAdd (std::uint64_t a, std::uint64_t b) noexcept
{
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    return b > maximum - a ? maximum : a + b;
}

inline constexpr std::uint64_t nominalBufferBytes (std::uint64_t length) noexcept
{
    return length;
}

// Extent at one mip level: halved per level, never below one texel.
inline constexpr std::uint64_t mipExtent (std::uint64_t extent, std::uint64_t level) noexcept
{
    const std::uint64_t halved = level < 64 ? extent >> level : 0;
    return halved == 0 ? 1 : halved;
}

// Texture storage for width x height x layers across all mip levels, times the
// texel size and sample count. Layers covers array layers and cube faces, which
// do not shrink per mip. A zero mip or sample count means one. This matches a
// full mip chain halved to one texel, the layout the fixture upload path builds.
inline constexpr std::uint64_t nominalTextureBytes (std::uint64_t width, std::uint64_t height,
                                                    std::uint64_t layers, std::uint64_t mipLevels,
                                                    std::uint64_t bytesPerTexel,
                                                    std::uint64_t sampleCount) noexcept
{
    std::uint64_t texels = 0;
    const std::uint64_t levels = mipLevels == 0 ? std::uint64_t { 1 } : mipLevels;
    const std::uint64_t slices = layers == 0 ? std::uint64_t { 1 } : layers;
    for (std::uint64_t level = 0; level < levels && level < 64; ++level)
    {
        const auto levelTexels = saturatingMultiply (
            saturatingMultiply (mipExtent (width, level), mipExtent (height, level)), slices);
        texels = saturatingAdd (texels, levelTexels);
    }
    const std::uint64_t samples = sampleCount == 0 ? std::uint64_t { 1 } : sampleCount;
    return saturatingMultiply (saturatingMultiply (texels, bytesPerTexel), samples);
}
} // namespace arbitgpu::residency
