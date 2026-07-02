// gpu_caps.h — runtime GPU capability descriptor.
//
// Dependency-free (no GL/GLFW headers) so it can be held and serialized by the
// GL-free parts of the helper (main.cpp RPC layer) as well as the GL backends.
// Populated once after context creation by arbitgl::queryGpuCaps (gl_loader.h).
//
// P1 of the visual-engine phased plan: the GL context now requests 4.3 core
// (4.1 on macOS, which caps desktop GL there). Compute-class features query
// this struct and degrade gracefully where they are unavailable.
#pragma once

namespace arbitgl
{

struct GpuCaps
{
    int  glMajor = 0;
    int  glMinor = 0;

    // Derived from the version + whether the matching entry points resolved.
    bool computeShaders = false;   // GL >= 4.3 and glDispatchCompute/glMemoryBarrier present
    bool ssbo           = false;   // GL >= 4.3 shader storage buffer objects
    bool imageLoadStore = false;   // GL >= 4.2 image load/store (glBindImageTexture)

    // True when a particle clip can run at all: real GL 4.3 compute (computeShaders),
    // OR — on macOS, where desktop GL caps at 4.1 — the ParticleEngine's ping-pong-
    // FBO fragment fallback (always available on a 4.1 core context: float colour
    // attachments + MRT are core). The plugin's New-Generator menu gates Particles
    // on THIS, not computeShaders, so macOS offers particles via the fallback while
    // computeShaders stays an honest "real compute present" signal.
    bool particles      = false;

    // Valid only when computeShaders is true (queried via glGetIntegeri_v);
    // left zeroed otherwise.
    int  maxComputeWorkGroupCount[3]    = { 0, 0, 0 };
    int  maxComputeWorkGroupSize[3]     = { 0, 0, 0 };
    int  maxComputeWorkGroupInvocations = 0;
};

} // namespace arbitgl
