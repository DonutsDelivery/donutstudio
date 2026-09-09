#pragma once

#include "sdf_native_render_admission.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace videohelper::sdf
{
inline constexpr std::string_view kNativeGpuCapability = "native-gpu";

enum class NativeSdfRenderUse : std::uint8_t
{
    Preview = 0,
    Export = 1
};

struct NativeSdfCacheIdentity final
{
    std::uint64_t projectGeneration = 0;
    std::uint64_t planRevision = 0;
    std::uint64_t helperGeneration = 0;
    int clipId = 0;
};

struct NativeSdfRenderedFrame final
{
    NativeSdfRenderUse use = NativeSdfRenderUse::Preview;
    std::string structuralDigest;
    NativeSdfRenderDimensions dimensions {};
    NativeSdfRenderControls controls {};
    std::shared_ptr<const arbitgpu::NativeSdfSceneFrame> nativeFrame;
};

// One renderer seam serves preview and export. It accepts only admitted
// immutable SDF IR with the exact native-gpu capability and publishes only
// backend-owned GPU frames.
class NativeSdfRenderer final
{
public:
    explicit NativeSdfRenderer (arbitgpu::NativeSdfExecutionBackend& backend) noexcept;

    bool renderPreview (const std::shared_ptr<const AdmittedSdfIr>& geometry,
                        NativeSdfRenderDimensions dimensions,
                        NativeSdfRenderControls controls,
                        std::string_view backendCapability,
                        NativeSdfRenderedFrame& output,
                        std::string& error, NativeSdfCacheIdentity identity = {});

    bool renderExport (const std::shared_ptr<const AdmittedSdfIr>& geometry,
                       NativeSdfRenderDimensions dimensions,
                       NativeSdfRenderControls controls,
                       std::string_view backendCapability,
                       NativeSdfRenderedFrame& output,
                       std::string& error, NativeSdfCacheIdentity identity = {});

    void invalidateCompiledGeometry() noexcept;
    std::size_t compiledGeometryCount() const noexcept;

private:
    struct CompiledGeometry final
    {
        std::shared_ptr<const AdmittedSdfIr> geometry;
        std::shared_ptr<const arbitgpu::NativeSdfCompiledProgram> program;
        NativeSdfCacheIdentity identity {};
        std::uint64_t lastUse = 0;
    };

    bool render (const std::shared_ptr<const AdmittedSdfIr>& geometry,
                 NativeSdfRenderDimensions dimensions,
                 NativeSdfRenderControls controls,
                 std::string_view backendCapability,
                 NativeSdfRenderUse use,
                 NativeSdfRenderedFrame& output,
                 std::string& error, NativeSdfCacheIdentity identity);

    arbitgpu::NativeSdfExecutionBackend& backend_;
    static constexpr std::size_t kMaximumCompiledGeometryEntries = 32;
    mutable std::mutex compiledGeometryMutex_;
    std::unordered_map<std::string, CompiledGeometry> compiledGeometry_;
    std::uint64_t cacheUse_ = 0;
};

// Process-owned production renderer. Preview and export share its immutable
// geometry/program cache and the process-owned native execution backend.
NativeSdfRenderer& nativeSdfRenderer();
} // namespace videohelper::sdf
