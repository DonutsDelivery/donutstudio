#if ARBIT_HAVE_VIEWPORT

#include "gl_loader.h"

#include <GLFW/glfw3.h>

namespace arbitgl
{

bool loadGlFunctions (GlFuncs& gl, std::string& missingOut)
{
    missingOut.clear();

#define ARBIT_GL_LOAD(type, name)                                             \
    gl.name = reinterpret_cast<decltype (gl.name)> (                          \
        glfwGetProcAddress ("gl" #name));                                     \
    if (gl.name == nullptr)                                                   \
    {                                                                         \
        if (! missingOut.empty()) missingOut += ", ";                         \
        missingOut += "gl" #name;                                             \
    }
    ARBIT_GL33_FUNCS (ARBIT_GL_LOAD)
#undef ARBIT_GL_LOAD

    return missingOut.empty();
}

bool loadGl43Functions (GlFuncs& gl)
{
#if ! defined (__APPLE__)
    bool all = true;
#define ARBIT_GL43_LOAD(type, name)                                           \
    gl.name = reinterpret_cast<decltype (gl.name)> (                          \
        glfwGetProcAddress ("gl" #name));                                     \
    if (gl.name == nullptr) all = false;
    ARBIT_GL43_FUNCS (ARBIT_GL43_LOAD)
#undef ARBIT_GL43_LOAD
    return all;
#else
    (void) gl;
    return false; // macOS desktop GL caps at 4.1 — no compute entry points.
#endif
}

void queryGpuCaps (const GlFuncs& gl, GpuCaps& caps)
{
    caps = GpuCaps {};
    // GL_MAJOR_VERSION / GL_MINOR_VERSION are valid on any 3.0+ core context
    // and glGetIntegerv resolves at link time (GL 1.1).
    glGetIntegerv (GL_MAJOR_VERSION, &caps.glMajor);
    glGetIntegerv (GL_MINOR_VERSION, &caps.glMinor);

    const bool ver42 = caps.glMajor > 4 || (caps.glMajor == 4 && caps.glMinor >= 2);
    const bool ver43 = caps.glMajor > 4 || (caps.glMajor == 4 && caps.glMinor >= 3);

#if ! defined (__APPLE__)
    caps.computeShaders = ver43 && gl.DispatchCompute != nullptr
                                && gl.MemoryBarrier   != nullptr;
    caps.ssbo           = ver43 && gl.BindBufferBase  != nullptr
                                && gl.ShaderStorageBlockBinding != nullptr;
    caps.imageLoadStore = ver42 && gl.BindImageTexture != nullptr;

    if (caps.computeShaders && gl.GetIntegeri_v != nullptr)
    {
        for (int i = 0; i < 3; ++i)
        {
            gl.GetIntegeri_v (GL_MAX_COMPUTE_WORK_GROUP_COUNT, (GLuint) i,
                              &caps.maxComputeWorkGroupCount[i]);
            gl.GetIntegeri_v (GL_MAX_COMPUTE_WORK_GROUP_SIZE,  (GLuint) i,
                              &caps.maxComputeWorkGroupSize[i]);
        }
        glGetIntegerv (GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,
                       &caps.maxComputeWorkGroupInvocations);
    }
    // Off-macOS the particle engine uses the compute path only (the fragment
    // fallback is gated to __APPLE__), so particle availability == compute.
    caps.particles = caps.computeShaders;
#else
    (void) ver42; (void) ver43; // compute/ssbo/imageLoadStore stay false on macOS
    // macOS caps desktop GL at 4.1 (no compute), but ParticleEngine has a GL-4.1
    // ping-pong-FBO fragment fallback. It needs only core 4.1 entry points (FBO,
    // float colour attachments, MRT via glDrawBuffers) — all resolved by
    // loadGlFunctions on any core context — so particles are available there.
    caps.particles = gl.GenFramebuffers != nullptr
                  && gl.FramebufferTexture2D != nullptr
                  && gl.DrawBuffers != nullptr;
#endif
}

} // namespace arbitgl

#endif // ARBIT_HAVE_VIEWPORT
