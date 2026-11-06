#pragma once

// OIDN for Path Trace frames (Plan 3, Denoising), loaded at run time like the
// vendor runtimes in pathtrace/devices/device_registry.cpp, so the helper has no
// build-time dependency on it and starts without it. Nothing links OIDN: the C
// API's handful of entry points are found with dlopen or LoadLibrary.

#include "pathtrace_denoiser.h"

#include <string>

namespace videorender::fixture3d::pathtracedenoise
{

// Why OIDN cannot run in this process, or empty when its library loaded with
// every entry point. Loads the library on the first call.
std::string oidnLoadProblem();

// The RT filter, HDR for linear frames, with the albedo and normal guides.
// Runs on want's device, and on the CPU when that device does not start; used
// names the device that ran, and note says why it is not want. False with
// error when OIDN failed.
bool oidnDenoise (OidnDevice want, const DenoiseFrame& frame, OidnDevice& used, std::string& note, std::string& error);

// Tests only: the next load tries this library alone, or the platform's names
// again for null. Devices made before stay alive and are not used again.
void setOidnLibraryForTests (const char* path);

} // namespace videorender::fixture3d::pathtracedenoise
