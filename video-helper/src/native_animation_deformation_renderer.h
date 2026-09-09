#pragma once

#include "gpu_backend/backend.h"

#include <memory>
#include <string>
#include <string_view>

namespace videorender::animation3d
{
inline constexpr std::string_view kNativeGpuCapability = "native-gpu";

enum class RenderUse : std::uint8_t { Preview = 0, Export = 1 };

struct RenderedDeformationFrame
{
    RenderUse use = RenderUse::Preview;
    std::shared_ptr<const arbitgpu::NativeDeformationScene> source;
    std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot> deformation;
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> nativeFrame;
    arbitgpu::NativeDeformationStats stats {};
};

class NativeAnimationDeformationRenderer final
{
public:
    static constexpr std::uint32_t kMaxExtent = 4096;
    static constexpr std::uint64_t kMaxPixels = 4096ull * 4096ull;

    explicit NativeAnimationDeformationRenderer (
        arbitgpu::NativeDeformationBackend& backend) noexcept;

    arbitgpu::BackendInfo backendInfo() const { return backend_.info(); }

    bool renderPreview (
        const std::shared_ptr<const arbitgpu::NativeDeformationScene>& source,
        const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>& deformation,
        std::uint32_t width, std::uint32_t height,
        std::string_view backendCapability,
        RenderedDeformationFrame& output,
        std::string& error);

    bool renderPreview (
        const std::shared_ptr<const arbitgpu::NativeDeformationScene>& source,
        const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>& deformation,
        const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>& materialProgram,
        arbitgpu::NativeDeformationRuntimeInputs runtimeInputs,
        std::uint32_t width, std::uint32_t height,
        std::string_view backendCapability,
        RenderedDeformationFrame& output,
        std::string& error);

    bool renderExport (
        const std::shared_ptr<const arbitgpu::NativeDeformationScene>& source,
        const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>& deformation,
        std::uint32_t width, std::uint32_t height,
        std::string_view backendCapability,
        RenderedDeformationFrame& output,
        std::string& error);

    bool renderExport (
        const std::shared_ptr<const arbitgpu::NativeDeformationScene>& source,
        const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>& deformation,
        const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>& materialProgram,
        arbitgpu::NativeDeformationRuntimeInputs runtimeInputs,
        std::uint32_t width, std::uint32_t height,
        std::string_view backendCapability,
        RenderedDeformationFrame& output,
        std::string& error);

private:
    bool render (RenderUse use,
                 const std::shared_ptr<const arbitgpu::NativeDeformationScene>& source,
                 const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>& deformation,
                 const std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>& materialProgram,
                 arbitgpu::NativeDeformationRuntimeInputs runtimeInputs,
                 std::uint32_t width, std::uint32_t height,
                 std::string_view backendCapability,
                 RenderedDeformationFrame& output,
                 std::string& error);

    arbitgpu::NativeDeformationBackend& backend_;
    std::weak_ptr<const arbitgpu::NativeDeformationScene> cachedSource_;
    std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> cachedMaterialProgram_;
    std::shared_ptr<const arbitgpu::NativeDeformationResources> cachedResources_;
    arbitgpu::NativeDeformationStats cachedFootprint_ {};
};
} // namespace videorender::animation3d
