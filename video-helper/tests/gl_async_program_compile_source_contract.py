#!/usr/bin/env python3
# Contract for the asynchronous OpenGL fixture program compile. A Preview prepare submits
# the scene program without waiting, polls GL_COMPLETION_STATUS_KHR, and returns pending
# until the program is ready. The submit path reads no compile or link status, and every
# pending return comes before the first GL upload of the scene.
from pathlib import Path
import sys

root = Path(sys.argv[1])
loader_header = (root / "src/gl_loader.h").read_text()
loader = (root / "src/gl_loader.cpp").read_text()
backend = (root / "src/gpu_backend/backend_opengl.cpp").read_text()
cmake = (root / "CMakeLists.txt").read_text()

errors = []


def require(text, needle, label):
    if needle not in text:
        errors.append(label + " is missing " + needle)


def region(text, begin, end, label):
    start = text.find(begin)
    if start < 0:
        errors.append(label + " is missing " + begin)
        return ""
    stop = text.find(end, start + len(begin))
    if stop < 0:
        errors.append(label + " is missing " + end)
        return ""
    return text[start:stop]


# Loader header: the KHR tokens, the explicit pointer types, and the two soft-loaded members.
for token in ("GL_MAX_SHADER_COMPILER_THREADS_KHR 0x91B0", "GL_COMPLETION_STATUS_KHR 0x91B1",
              "using GlGetStringiFn", "using GlMaxShaderCompilerThreadsFn",
              "GetStringi = nullptr", "MaxShaderCompilerThreads = nullptr",
              "bool loadGlParallelShaderCompileFunctions (GlFuncs& gl);",
              "bool hasParallelShaderCompile (const GlFuncs& gl);"):
    require(loader_header, token, "gl_loader.h")

# Loader source: glGetStringi is soft-loaded, the KHR entry point is tried before the ARB one,
# and both extension names are recognized.
load_body = region(loader, "bool loadGlParallelShaderCompileFunctions (GlFuncs& gl)",
                   "bool hasParallelShaderCompile (const GlFuncs& gl)", "gl_loader.cpp")
require(load_body, '"glGetStringi"', "gl_loader.cpp load")
khr_at = load_body.find('"glMaxShaderCompilerThreadsKHR"')
arb_at = load_body.find('"glMaxShaderCompilerThreadsARB"')
if khr_at < 0 or arb_at < 0 or khr_at > arb_at:
    errors.append("gl_loader.cpp must try the KHR entry point before the ARB entry point")
has_body = region(loader, "bool hasParallelShaderCompile (const GlFuncs& gl)",
                  "void queryGpuCaps", "gl_loader.cpp")
for name in ("GL_NUM_EXTENSIONS", "GL_KHR_parallel_shader_compile", "GL_ARB_parallel_shader_compile"):
    require(has_body, name, "gl_loader.cpp hasParallelShaderCompile")

# Backend: the loader runs with each prepare, the extension answer is cached per context,
# and the thread limit is set once.
require(backend, "arbitgl::loadGlParallelShaderCompileFunctions (gl);", "backend_opengl.cpp")
require(backend, "arbitgl::hasParallelShaderCompile (gl)", "backend_opengl.cpp")
require(backend, "gl.MaxShaderCompilerThreads (0xFFFFFFFFu);", "backend_opengl.cpp")
require(backend, "int parallelCompile = -1;", "backend_opengl.cpp")

# Submit: starts both compiles and the link without reading any status, and sets the binary
# retrieval hint before the link.
submit = region(backend, "std::string submitFixtureProgram (", "std::string finishFixtureProgram (",
                "backend_opengl.cpp")
for forbidden in ("GL_COMPILE_STATUS", "GL_LINK_STATUS", "GetShaderiv", "GetProgramiv"):
    if forbidden in submit:
        errors.append("submitFixtureProgram must not read " + forbidden)
link_at = submit.find("gl.LinkProgram (entry.program);")
prepare_link_at = submit.find("binaryCache.prepareLink (entry.program);")
if link_at < 0 or prepare_link_at < 0 or prepare_link_at > link_at:
    errors.append("submitFixtureProgram must call prepareLink before LinkProgram")

# Poll: the completion query runs inside the poll scope, and no status is read before it.
poll_scope = backend.find('loopstep::Scope pollScope ("gl:fixture-scene-program-poll")')
completion_at = backend.find("GL_COMPLETION_STATUS_KHR, &complete);")
if poll_scope < 0 or completion_at < poll_scope:
    errors.append("the completion query must follow the poll scope")
for label in ("gl:fixture-scene-program-poll", "gl:fixture-scene-program-submit",
              "gl:fixture-scene-program-binary-load", "gl:fixture-scene-program-compile"):
    require(backend, '("' + label + '")', "backend_opengl.cpp")

# Prepare region: pending returns before the first scene upload, and the first upload is still there.
impl_at = backend.find("NativeFixtureScenePreparation prepareGeometryInstancesImpl(")
end_at = backend.find("NativeFixtureSceneSubmission render (", impl_at) if impl_at >= 0 else -1
impl = backend[impl_at:end_at] if impl_at >= 0 and end_at > impl_at else ""
if not impl:
    errors.append("prepareGeometryInstancesImpl region is missing")
pending_at = impl.find("result.pending = true;")
if pending_at < 0:
    errors.append("the in-flight path never returns pending")
for needle in ("smoothglass::prepare(", "environmentimage::prepare(", "gl.GenVertexArrays (",
               "gl.BufferData (GL_ARRAY_BUFFER", "glTexImage2D (GL_TEXTURE_2D, 0,"):
    at = impl.find(needle)
    if at < 0:
        errors.append("the prepare region no longer contains " + needle)
    elif pending_at >= at:
        errors.append("pending must return before " + needle)

# Blocking path: the synchronous compile keeps its scope, and its link status check follows it.
compile_scope = backend.find('loopstep::Scope sceneCompileScope ("gl:fixture-scene-program-compile")')
sync_link = backend.find("gl.GetProgramiv(shared->id,GL_LINK_STATUS,&linked);")
if compile_scope < 0 or sync_link < compile_scope:
    errors.append("the synchronous compile must keep its link status check after its scope")

# Context release: in-flight GL objects are deleted before the context cache entry is erased.
release = region(backend, "void releaseCachedProgramsForCurrentContext() noexcept override",
                 "BackendInfo info() const override", "backend_opengl.cpp")
delete_at = release.find("deleteInFlightObjects")
erase_at = release.find("programCache_.erase(context);")
if delete_at < 0 or erase_at < 0 or delete_at > erase_at:
    errors.append("in-flight programs must be deleted before the context cache is erased")

# Diagnostic line for an asynchronously finished program.
require(backend, "[gl] fixture program ready after", "backend_opengl.cpp")

# The derived prepare takes the use argument by name, defined outside the class.
require(backend, "NativeFixtureScenePreparation OpenGlFixtureSceneBackend::prepare (", "backend_opengl.cpp")
require(backend, "NativeFixturePrepareUse use)", "backend_opengl.cpp")
require(backend, "NativeFixturePrepareUse = NativeFixturePrepareUse::Blocking) override;", "backend_opengl.cpp")

require(cmake, "add_test(NAME gl_async_program_compile_source_contract", "CMakeLists.txt")
require(cmake, "tests/gl_async_program_compile_source_contract.py", "CMakeLists.txt")

if errors:
    print("gl async program compile contract: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("gl async program compile source contract: PASS")
