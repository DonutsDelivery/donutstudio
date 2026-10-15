#pragma once
//
// Slang → GLSL front-end (visual-engine plan P3).
//
// Slang (shader-slang, Apache-2.0) is a modern modular shading language:
// modules / generics / one source → GLSL/SPIR-V/Metal/WGSL. Arbit's authoring
// story is a shared `arbit_contract.slang` module (shaders/arbit_contract.slang)
// declaring the clock/audio/score uniform contract that user shaders `import`.
//
// In-process compilation needs libslang (the shader-slang compiler), which is a
// large dependency and is NOT a system package on most boxes. So this seam is
// gated behind ARBIT_HAVE_SLANG and, when off, returns guidance pointing at the
// already-working route: compile the Slang offline with `slangc -target glsl`
// (or `-target spirv` then import as lang=spirv — the SPIR-V universal adapter
// in spirv_ingest.h). Wiring an in-process libslang (e.g. a FetchContent of the
// shader-slang prebuilt release) is a drop-in behind this flag — the RPC
// routing, cache, and contract module are already in place.
//
#include <string>

namespace arbitshadercompile
{

// Lowers Slang source to GLSL. Returns false + err when libslang is not
// compiled in (the default), with a pointer to the offline-compile route.
inline bool slangToGlsl (const std::string& source, std::string& outGlsl,
                         std::string& err)
{
#if ARBIT_HAVE_SLANG
    // Placeholder for the libslang IGlobalSession path (createGlobalSession →
    // session with a GLSL TargetDesc → loadModuleFromSourceString →
    // getEntryPointCode). Implemented when ARBIT_HAVE_SLANG is wired to a
    // shader-slang build.
    (void) source; (void) outGlsl;
    err = "in-process Slang compile not yet implemented in this build";
    return false;
#else
    (void) source; (void) outGlsl;
    err = "Slang not built in — compile offline with `slangc -target spirv "
          "shader.slang -o shader.spv` and import with lang=spirv "
          "(the SPIR-V universal adapter), or `-target glsl` and import as glsl";
    return false;
#endif
}

} // namespace arbitshadercompile
