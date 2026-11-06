#include "cpu_scene_frame.h"

namespace arbitgpu
{

// Headless builds have no CPU scene upload.
std::shared_ptr<const NativeFixtureSceneFrame> uploadCpuSceneFrame (
    const std::string& backend, const CpuSceneImage&, std::string& error)
{
    error = "Path Trace has no " + (backend.empty() ? std::string ("native") : backend)
        + " frame upload in this build";
    return nullptr;
}

} // namespace arbitgpu
