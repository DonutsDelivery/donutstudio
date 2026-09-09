#include "backend.h"

namespace arbitgpu
{

namespace
{
class UnavailableRenderPassOutputBackend final : public RenderPassOutputBackend
{
public:
    RenderPassOutputCapabilities renderPassOutputCapabilities() const override
    {
        RenderPassOutputCapabilities result;
        result.error = "native GPU render-pass outputs are not compiled in";
        return result;
    }

    RenderPassOutputAdmission admitRenderPassOutputs (
        const renderpassoutput::AdmittedOutputs&) override
    {
        RenderPassOutputAdmission result;
        result.error = "native GPU render-pass outputs are not compiled in";
        return result;
    }

    RenderPassColorAovExecution executeColorAovClear (
        RenderPassOutputLifecycleHandle,
        const RenderPassColorAovClear&) override
    {
        RenderPassColorAovExecution result;
        result.error = "native GPU Color AOV execution is not compiled in";
        return result;
    }

    RenderPassMotionAovExecution executeMotionAovClear (
        RenderPassOutputLifecycleHandle,
        const RenderPassMotionAovClear&) override
    {
        RenderPassMotionAovExecution result;
        result.error = "native GPU Motion AOV execution is not compiled in";
        return result;
    }

    RenderPassAovInspectionExecution executeAovInspection (
        RenderPassOutputLifecycleHandle,
        const aovinspection::Payload&) override
    {
        RenderPassAovInspectionExecution result;
        result.error = "native GPU AOV inspection execution is not compiled in";
        return result;
    }

    RenderPassSceneAovExecution executeSceneAov (
        RenderPassOutputLifecycleHandle,
        const sceneaov::Payload&) override
    {
        RenderPassSceneAovExecution result;
        result.error = "native GPU scene AOV execution is not compiled in";
        return result;
    }

    void releaseRenderPassOutputs (RenderPassOutputLifecycleHandle) noexcept override
    {
    }
};

class UnavailableOpticalFlowBackend final : public NativeOpticalFlowExecutionBackend
{
public:
    videoopticalflow::BackendCapabilities opticalFlowCapabilities() const override
    {
        return {};
    }

    NativeOpticalFlowSubmission executeOpticalFlow (
        const videoopticalflow::AdmittedRequest&,
        const NativeOpticalFlowInputResource&,
        const NativeOpticalFlowInputResource&,
        bool) override
    {
        NativeOpticalFlowSubmission result;
        result.error = "native GPU optical-flow execution is not compiled in";
        return result;
    }

    void releaseOpticalFlowOutput (NativeOpticalFlowOutputLifecycleHandle) noexcept override {}
};
} // namespace

BackendInfo queryNativeBackend()
{
    BackendInfo result;
    result.error = "native GPU backend not compiled in";
    return result;
}

BackendSelfTest runNativeBackendSelfTest()
{
    BackendSelfTest result;
    result.error = "native GPU backend not compiled in";
    return result;
}

namespace
{
class StubSdfExecutionBackend final : public NativeSdfExecutionBackend
{
public:
    NativeSdfExecutionCapabilities capabilities() const override
    {
        NativeSdfExecutionCapabilities result;
        result.error = "native GPU SDF execution is not compiled in";
        return result;
    }

    NativeSdfSceneSubmission render (const NativeSdfDrawRequest&) override
    {
        NativeSdfSceneSubmission result;
        result.error = "native GPU SDF execution is not compiled in";
        return result;
    }
};
} // namespace

NativeSdfExecutionBackend& nativeSdfExecutionBackend()
{
    static StubSdfExecutionBackend backend;
    return backend;
}

void invalidateNativeSdfExecutionContext (std::uintptr_t) noexcept
{
}

NativeSdfExecutionCapabilities queryNativeSdfExecution()
{
    return nativeSdfExecutionBackend().capabilities();
}

RenderPassOutputBackend& nativeRenderPassOutputBackend()
{
    static UnavailableRenderPassOutputBackend backend;
    return backend;
}

NativeOpticalFlowExecutionBackend& nativeOpticalFlowExecutionBackend()
{
    static UnavailableOpticalFlowBackend backend;
    return backend;
}

namespace
{
class StubFixtureSceneBackend final : public NativeFixtureSceneBackend
{
public:
    BackendInfo info() const override
    {
        return queryNativeBackend();
    }

    NativeFixtureScenePreparation prepare (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>&,
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>&) override
    {
        NativeFixtureScenePreparation result;
        result.error = "native fixture GPU backend not compiled in";
        return result;
    }

    NativeFixtureSceneSubmission render (
        const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>&,
        const std::shared_ptr<const NativeFixtureSceneResources>&,
        std::uint32_t,
        std::uint32_t,
        NativeFixtureSceneRuntimeInputs) override
    {
        NativeFixtureSceneSubmission result;
        result.error = "native fixture GPU backend not compiled in";
        return result;
    }
};
} // namespace

NativeFixtureSceneBackend& nativeFixtureSceneBackend()
{
    static StubFixtureSceneBackend backend;
    return backend;
}

namespace
{
class StubDeformationBackend final : public NativeDeformationBackend
{
public:
    BackendInfo info() const override
    {
        return queryNativeBackend();
    }

    NativeDeformationPreparation prepare (
        const std::shared_ptr<const NativeDeformationScene>&,
        const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>&) override
    {
        NativeDeformationPreparation result;
        result.error = "native animation deformation GPU backend not compiled in";
        return result;
    }

    NativeDeformationSubmission render (
        const std::shared_ptr<const NativeDeformationScene>&,
        const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>&,
        const std::shared_ptr<const NativeDeformationResources>&,
        std::uint32_t,
        std::uint32_t,
        NativeDeformationRuntimeInputs) override
    {
        NativeDeformationSubmission result;
        result.error = "native animation deformation GPU backend not compiled in";
        return result;
    }
};
} // namespace

NativeDeformationBackend& nativeDeformationBackend()
{
    static StubDeformationBackend backend;
    return backend;
}

} // namespace arbitgpu
