#include "cpu_scene_frame.h"

namespace arbitgpu
{

// Metal and headless builds have no CPU scene upload yet.
std::shared_ptr<const NativeFixtureSceneFrame> uploadCpuSceneFrame (
    const std::string& backend, const CpuSceneImage&, std::string& error)
{
    error = "Path Trace has no " + (backend.empty() ? std::string ("native") : backend)
        + " frame upload in this build";
    return nullptr;
}

} // namespace arbitgpu
