#pragma once
//
// SPIR-V → GLSL ingestion (visual-engine plan P3 — the "universal adapter").
//
// Accepts precompiled SPIR-V and lowers it to the GLSL the renderer already
// runs, via vendored SPIRV-Cross (third_party/spirv_cross/, Apache-2.0). Because
// Slang, HLSL and GLSL all compile to SPIR-V (glslang/dxc/slangc), this one
// path makes the helper language-agnostic: any SPIR-V-targeting front-end works
// for free.
//
// Header-only and guarded by ARBIT_HAVE_SPIRV_CROSS: without the backend the
// function returns false with a clear diagnostic, never a build failure
// (mirrors ARBIT_HAVE_LUA / ARBIT_HAVE_QUICKJS).
//
#include <cstdint>
#include <string>
#include <vector>

#if ARBIT_HAVE_SPIRV_CROSS
#include "spirv_glsl.hpp"
#endif

namespace arbitshadercompile
{

// Lowers a SPIR-V module to a desktop-GL GLSL source string at the requested
// #version (the renderer runs core profile, no ES). Returns false + err on a
// malformed module or when SPIRV-Cross is not compiled in.
inline bool spirvToGlsl (const std::vector<uint32_t>& spirv, int glslVersion,
                         std::string& outGlsl, std::string& err)
{
#if ARBIT_HAVE_SPIRV_CROSS
    if (spirv.empty() || spirv[0] != 0x07230203u)   // SPIR-V magic
    {
        err = "not a SPIR-V module (bad magic / empty)";
        return false;
    }
    try
    {
        spirv_cross::CompilerGLSL glsl (spirv);
        spirv_cross::CompilerGLSL::Options opts;
        opts.version = (uint32_t) glslVersion;
        opts.es = false;
        opts.enable_420pack_extension = false; // keep output portable to 3.3 core
        glsl.set_common_options (opts);
        outGlsl = glsl.compile();
        return true;
    }
    catch (const std::exception& e)
    {
        err = std::string ("SPIRV-Cross: ") + e.what();
        return false;
    }
#else
    (void) spirv; (void) glslVersion; (void) outGlsl;
    err = "SPIR-V ingestion not built in (ARBIT_HAVE_SPIRV_CROSS off)";
    return false;
#endif
}

} // namespace arbitshadercompile
