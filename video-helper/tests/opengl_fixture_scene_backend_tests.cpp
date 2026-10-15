#include "support/camera_pixel_sign_cases.h"
#include "support/camera_wide_line_predicate_cases.h"
#include "support/camera_line_predicate_cases.h"
#include "support/camera_ray_coefficient_cases.h"
#include "support/exact_line_predicate_cases.h"
#include "../src/gpu_backend/backend.h"
#include "../src/canonical_block_c_frame.h"
#include "../src/diffraction_material_admission.h"
#include "../src/diffraction_material_execution.h"
#include "../src/fixture_scene_renderer.h"
#include "../src/gpu_backend/fixture_environment_image.h"
#include "../src/gpu_backend/gpu_residency_ledger.h"
#include "../src/diffractive_foil_admission.h"
#include "../src/gl_loader.h"
#include "../src/color_transform_gl.h"
#include "../../shared/DecodedFrameColorContract.h"
#include "support/fixture_scene.h"
#include "support/hlg_material_oracle.h"
#include "support/base_surface_reflection_cases.h"
#include "support/analytic_sky_cases.h"
#include "support/environment_image_cases.h"
#include "support/smooth_glass_cases.h"
#include "support/opengl_rough_glass_cases.h"
#include "support/glass_caustic_oracle.h"
#include "support/authored_rough_caustic_renderer_cases.h"
#include "../src/gpu_backend/fixture_glass_caustics_opengl.h"
#include "support/thin_interface_water_cases.h"
#include "support/opengl_rough_water_cases.h"
#include "support/opaque_visibility_cases.h"
#include "support/vertex_modifier_fixture.h"
#include "support/diffraction_environment_cases.h"
#include "support/surface_material_starter_oracle.h"
#include "support/generated_surface_scene.h"
#include "support/render_pass_program_cases.h"
#include "support/material_frame_binding_cases.h"
#include "support/native_scene_orientation_fixture.h"
#include "support/raw_pass_readback_checks.h"
#include "support/linear_scene_hdr_checks.h"
#include "../src/sha256.h"
#include "../../shared/DiffractionMaterialPresets.h"
#include "../../shared/generated/SurfaceMaterialStarterPrograms.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/gl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace
{

bool expect (bool condition, const char* message)
{
    if (! condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}

// Test-local GL4.3 command ownership; arithmetic evidence only. This does not
// create or authenticate any camera parent, native coverage proof or receipt.
bool exactLineNativeProbe()
{
#if defined(__APPLE__)
    return expect(false,"exact integer predicate probe requires GLSL4.30");
#else
    namespace ec=videohelper::tests::exactlinecases;namespace el=arbitgpu::exactline;
    auto* previous=glfwGetCurrentContext();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    auto* window=glfwCreateWindow(1,1,"exact-line-arithmetic-probe",nullptr,nullptr);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    if(!window)return expect(false,"exact-line probe GL4.3 context unavailable");
    glfwMakeContextCurrent(window);
    const bool passed=[&]() {
        arbitgl::GlFuncs gl;std::string error;
        if(!arbitgl::loadGlFunctions(gl,error)||!arbitgl::loadGl43Functions(gl))
            return expect(false,"exact-line probe GL4.3 functions unavailable");
        const auto tests=ec::cases();
        // Fixed test-local admission, separate from any scene/query owner. The
        // derived envelopes, input, output, retained host inputs, native buffers, oracle output and logical thread
        // scratch are known before native buffer allocation/dispatch.
        if(tests.empty()||tests.size()>1024||tests.capacity()>1024)return expect(false,"exact-line probe case admission");
        std::uint64_t admittedWork=0,admittedBytes=tests.capacity()*sizeof(ec::Case);
        for(const auto& test:tests){admittedWork+=el::maximumScalarOperations(static_cast<el::Operation>(test.input[0]));
            admittedBytes+=el::kLogicalScratchBytes+2*sizeof(el::Request)+2*sizeof(el::Result);}
        if(admittedWork>1024*el::maximumScalarOperations(el::Operation::CompareX)
            ||admittedBytes>1024*(el::kLogicalScratchBytes+2*sizeof(el::Request)+2*sizeof(el::Result)+sizeof(ec::Case)))
            return expect(false,"exact-line probe derived envelope exceeded");
        std::vector<el::Request> inputs;inputs.reserve(tests.size());
        for(const auto& test:tests)inputs.push_back(test.input);
        struct Resources {
            arbitgl::GlFuncs& gl;GLuint shader=0,program=0,buffers[2]{};GLsync fence=nullptr;
            ~Resources(){if(fence)gl.DeleteSync(fence);
                gl.UseProgram(0);
                gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,0,0);gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,1,0);
                gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,0);gl.DeleteBuffers(2,buffers);
                if(program)gl.DeleteProgram(program);
                if(shader)gl.DeleteShader(shader);}
        } owned{gl};
        owned.shader=gl.CreateShader(GL_COMPUTE_SHADER);const auto source=el::glslProbeSource();const char* text=source.c_str();
        gl.ShaderSource(owned.shader,1,&text,nullptr);gl.CompileShader(owned.shader);GLint compiled=0;
        gl.GetShaderiv(owned.shader,GL_COMPILE_STATUS,&compiled);
        if(!compiled){char log[4096]{};gl.GetShaderInfoLog(owned.shader,sizeof(log),nullptr,log);
            std::cerr<<"exact-line shader: "<<log<<'\n';return false;}
        owned.program=gl.CreateProgram();gl.AttachShader(owned.program,owned.shader);gl.LinkProgram(owned.program);
        GLint linked=0;gl.GetProgramiv(owned.program,GL_LINK_STATUS,&linked);
        if(!linked){char log[4096]{};gl.GetProgramInfoLog(owned.program,sizeof(log),nullptr,log);
            std::cerr<<"exact-line program: "<<log<<'\n';return false;}
        gl.GenBuffers(2,owned.buffers);
        if(!owned.buffers[0]||!owned.buffers[1])return expect(false,"exact-line native buffer allocation");
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,owned.buffers[0]);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER,inputs.size()*sizeof(el::Request),inputs.data(),GL_STATIC_DRAW);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,owned.buffers[1]);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER,tests.size()*sizeof(el::Result),nullptr,GL_DYNAMIC_READ);
        if(glGetError()!=GL_NO_ERROR)return expect(false,"exact-line native buffer storage");
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,0,owned.buffers[0]);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,1,owned.buffers[1]);gl.UseProgram(owned.program);
        gl.DispatchCompute(static_cast<GLuint>(tests.size()),1,1);
        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
        owned.fence=gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);
        if(!owned.fence)return expect(false,"exact-line native fence creation");
        const auto completion=gl.ClientWaitSync(owned.fence,GL_SYNC_FLUSH_COMMANDS_BIT,5000000000ull);
        if(completion!=GL_ALREADY_SIGNALED&&completion!=GL_CONDITION_SATISFIED)
            return expect(false,"exact-line native command did not complete");
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,owned.buffers[1]);
        const auto* mapped=static_cast<const std::uint8_t*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,
            tests.size()*sizeof(el::Result),GL_MAP_READ_BIT));
        if(!mapped)return expect(false,"exact-line native arithmetic readback unavailable");
        bool ok=true;for(std::size_t k=0;k<tests.size();k++){
            el::Result actual{};std::memcpy(actual.data(),mapped+k*sizeof(el::Result),sizeof(actual));
            if(!ec::compare(tests[k],actual,error)){std::cerr<<"FAIL: GL exact-line arithmetic: "<<error<<'\n';ok=false;}}
        if(!gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER))ok=false;
        return ok;
    }();
    glfwMakeContextCurrent(previous);glfwDestroyWindow(window);return passed;
#endif
}

bool cameraRayCoefficientNativeProbe()
{
#if defined(__APPLE__)
    return expect(false,"camera-ray coefficient probe requires GLSL4.30");
#else
    namespace ec=videohelper::tests::cameraraycases;namespace el=arbitgpu::cameraraycoefficients;
    auto* previous=glfwGetCurrentContext();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    auto* window=glfwCreateWindow(1,1,"camera-ray-coefficient-arithmetic-probe",nullptr,nullptr);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    if(!window)return expect(false,"camera-ray-coefficient probe GL4.3 context unavailable");
    glfwMakeContextCurrent(window);
    const bool passed=[&]() {
        arbitgl::GlFuncs gl;std::string error;
        if(!arbitgl::loadGlFunctions(gl,error)||!arbitgl::loadGl43Functions(gl))
            return expect(false,"camera-ray-coefficient probe GL4.3 functions unavailable");
        const auto tests=ec::cases();
        // Fixed test-local admission, separate from any scene/query owner. The
        // derived envelopes, input, output, retained host inputs, native buffers, oracle output and logical thread
        // scratch are known before native buffer allocation/dispatch.
        if(tests.empty()||tests.size()>1024||tests.capacity()>1024)return expect(false,"camera-ray-coefficient probe case admission");
        std::uint64_t admittedWork=0,admittedBytes=tests.capacity()*sizeof(ec::Case);
        for(const auto& test:tests){admittedWork+=el::maximumScalarOperations(test.input[1]);
            admittedBytes+=el::kLogicalScratchBytes+2*sizeof(el::Request)+2*sizeof(el::Result);}
        if(admittedWork>1024*el::maximumScalarOperations(256u)
            ||admittedBytes>1024*(el::kLogicalScratchBytes+2*sizeof(el::Request)+2*sizeof(el::Result)+sizeof(ec::Case)))
            return expect(false,"camera-ray-coefficient probe derived envelope exceeded");
        std::vector<el::Request> inputs;inputs.reserve(tests.size());
        for(const auto& test:tests)inputs.push_back(test.input);
        struct Resources {
            arbitgl::GlFuncs& gl;GLuint shader=0,program=0,buffers[2]{};GLsync fence=nullptr;
            ~Resources(){if(fence)gl.DeleteSync(fence);
                gl.UseProgram(0);
                gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,0,0);gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,1,0);
                gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,0);gl.DeleteBuffers(2,buffers);
                if(program)gl.DeleteProgram(program);
                if(shader)gl.DeleteShader(shader);}
        } owned{gl};
        owned.shader=gl.CreateShader(GL_COMPUTE_SHADER);const auto source=el::glslProbeSource();const char* text=source.c_str();
        gl.ShaderSource(owned.shader,1,&text,nullptr);gl.CompileShader(owned.shader);GLint compiled=0;
        gl.GetShaderiv(owned.shader,GL_COMPILE_STATUS,&compiled);
        if(!compiled){char log[4096]{};gl.GetShaderInfoLog(owned.shader,sizeof(log),nullptr,log);
            std::cerr<<"camera-ray-coefficient shader: "<<log<<'\n';return false;}
        owned.program=gl.CreateProgram();gl.AttachShader(owned.program,owned.shader);gl.LinkProgram(owned.program);
        GLint linked=0;gl.GetProgramiv(owned.program,GL_LINK_STATUS,&linked);
        if(!linked){char log[4096]{};gl.GetProgramInfoLog(owned.program,sizeof(log),nullptr,log);
            std::cerr<<"camera-ray-coefficient program: "<<log<<'\n';return false;}
        gl.GenBuffers(2,owned.buffers);
        if(!owned.buffers[0]||!owned.buffers[1])return expect(false,"camera-ray-coefficient native buffer allocation");
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,owned.buffers[0]);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER,inputs.size()*sizeof(el::Request),inputs.data(),GL_STATIC_DRAW);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,owned.buffers[1]);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER,tests.size()*sizeof(el::Result),nullptr,GL_DYNAMIC_READ);
        if(glGetError()!=GL_NO_ERROR)return expect(false,"camera-ray-coefficient native buffer storage");
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,0,owned.buffers[0]);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,1,owned.buffers[1]);gl.UseProgram(owned.program);
        gl.DispatchCompute(static_cast<GLuint>(tests.size()),1,1);
        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
        owned.fence=gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);
        if(!owned.fence)return expect(false,"camera-ray-coefficient native fence creation");
        const auto completion=gl.ClientWaitSync(owned.fence,GL_SYNC_FLUSH_COMMANDS_BIT,5000000000ull);
        if(completion!=GL_ALREADY_SIGNALED&&completion!=GL_CONDITION_SATISFIED)
            return expect(false,"camera-ray-coefficient native command did not complete");
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,owned.buffers[1]);
        const auto* mapped=static_cast<const std::uint8_t*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,
            tests.size()*sizeof(el::Result),GL_MAP_READ_BIT));
        if(!mapped)return expect(false,"camera-ray-coefficient native arithmetic readback unavailable");
        bool ok=true;for(std::size_t k=0;k<tests.size();k++){
            el::Result actual{};std::memcpy(actual.data(),mapped+k*sizeof(el::Result),sizeof(actual));
            if(!ec::compare(tests[k],actual,error)){std::cerr<<"FAIL: GL camera-ray-coefficient arithmetic: "<<error<<'\n';ok=false;}}
        if(!gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER))ok=false;
        return ok;
    }();
    glfwMakeContextCurrent(previous);glfwDestroyWindow(window);return passed;
#endif
}

bool cameraLinePredicateNativeProbe()
{
#if defined(__APPLE__)
    return expect(false,"camera-line predicate probe requires GLSL4.30");
#else
    namespace ec=videohelper::tests::cameralinecases;namespace el=arbitgpu::cameraline;
    auto* previous=glfwGetCurrentContext();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    auto* window=glfwCreateWindow(1,1,"camera-line-predicate-arithmetic-probe",nullptr,nullptr);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    if(!window)return expect(false,"camera-line-predicate probe GL4.3 context unavailable");
    glfwMakeContextCurrent(window);
    const bool passed=[&]() {
        arbitgl::GlFuncs gl;std::string error;
        if(!arbitgl::loadGlFunctions(gl,error)||!arbitgl::loadGl43Functions(gl))
            return expect(false,"camera-line-predicate probe GL4.3 functions unavailable");
        const auto tests=ec::cases();
        // Fixed test-local admission, separate from any scene/query owner. The
        // derived envelopes, input, output, retained host inputs, native buffers, oracle output and logical thread
        // scratch are known before native buffer allocation/dispatch.
        if(tests.empty()||tests.size()>1024||tests.capacity()>1024)return expect(false,"camera-line-predicate probe case admission");
        std::uint64_t admittedWork=0,admittedBytes=tests.capacity()*sizeof(ec::Case);
        for(const auto& test:tests){admittedWork+=ec::work(test);
            admittedBytes+=ec::kProbeScratchBytes+2*sizeof(el::ProbeInput)+2*sizeof(el::Result);}
        if(admittedWork>1024*ec::kMaximumProbeWork
            ||admittedBytes>1024*(ec::kProbeScratchBytes+2*sizeof(el::ProbeInput)+2*sizeof(el::Result)+sizeof(ec::Case)))
            return expect(false,"camera-line-predicate probe derived envelope exceeded");
        std::vector<el::ProbeInput> inputs;inputs.reserve(tests.size());
        for(const auto& test:tests)inputs.push_back(test.input);
        struct Resources {
            arbitgl::GlFuncs& gl;GLuint shader=0,program=0,buffers[2]{};GLsync fence=nullptr;
            ~Resources(){if(fence)gl.DeleteSync(fence);
                gl.UseProgram(0);
                gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,0,0);gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,1,0);
                gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,0);gl.DeleteBuffers(2,buffers);
                if(program)gl.DeleteProgram(program);
                if(shader)gl.DeleteShader(shader);}
        } owned{gl};
        owned.shader=gl.CreateShader(GL_COMPUTE_SHADER);const auto source=el::glslProbeSource();const char* text=source.c_str();
        gl.ShaderSource(owned.shader,1,&text,nullptr);gl.CompileShader(owned.shader);GLint compiled=0;
        gl.GetShaderiv(owned.shader,GL_COMPILE_STATUS,&compiled);
        if(!compiled){char log[4096]{};gl.GetShaderInfoLog(owned.shader,sizeof(log),nullptr,log);
            std::cerr<<"camera-line-predicate shader: "<<log<<'\n';return false;}
        owned.program=gl.CreateProgram();gl.AttachShader(owned.program,owned.shader);gl.LinkProgram(owned.program);
        GLint linked=0;gl.GetProgramiv(owned.program,GL_LINK_STATUS,&linked);
        if(!linked){char log[4096]{};gl.GetProgramInfoLog(owned.program,sizeof(log),nullptr,log);
            std::cerr<<"camera-line-predicate program: "<<log<<'\n';return false;}
        gl.GenBuffers(2,owned.buffers);
        if(!owned.buffers[0]||!owned.buffers[1])return expect(false,"camera-line-predicate native buffer allocation");
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,owned.buffers[0]);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER,inputs.size()*sizeof(el::ProbeInput),inputs.data(),GL_STATIC_DRAW);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,owned.buffers[1]);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER,tests.size()*sizeof(el::Result),nullptr,GL_DYNAMIC_READ);
        if(glGetError()!=GL_NO_ERROR)return expect(false,"camera-line-predicate native buffer storage");
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,0,owned.buffers[0]);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,1,owned.buffers[1]);gl.UseProgram(owned.program);
        gl.DispatchCompute(static_cast<GLuint>(tests.size()),1,1);
        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
        owned.fence=gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);
        if(!owned.fence)return expect(false,"camera-line-predicate native fence creation");
        const auto completion=gl.ClientWaitSync(owned.fence,GL_SYNC_FLUSH_COMMANDS_BIT,5000000000ull);
        if(completion!=GL_ALREADY_SIGNALED&&completion!=GL_CONDITION_SATISFIED)
            return expect(false,"camera-line-predicate native command did not complete");
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,owned.buffers[1]);
        const auto* mapped=static_cast<const std::uint8_t*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,
            tests.size()*sizeof(el::Result),GL_MAP_READ_BIT));
        if(!mapped)return expect(false,"camera-line-predicate native arithmetic readback unavailable");
        bool ok=true;for(std::size_t k=0;k<tests.size();k++){
            el::Result actual{};std::memcpy(actual.data(),mapped+k*sizeof(el::Result),sizeof(actual));
            if(!ec::compare(tests[k],actual,error)){std::cerr<<"FAIL: GL camera-line-predicate arithmetic: "<<error<<'\n';ok=false;}}
        if(!gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER))ok=false;
        return ok;
    }();
    glfwMakeContextCurrent(previous);glfwDestroyWindow(window);return passed;
#endif
}

bool cameraWideLinePredicateNativeProbe()
{
#if defined(__APPLE__)
    return expect(false,"camera-wide-line predicate probe requires GLSL4.30");
#else
    namespace ec=videohelper::tests::camerawidelinecases;namespace el=arbitgpu::camerawideline;
    auto* previous=glfwGetCurrentContext();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    auto* window=glfwCreateWindow(1,1,"camera-wide-line-predicate-arithmetic-probe",nullptr,nullptr);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    if(!window)return expect(false,"camera-wide-line-predicate probe GL4.3 context unavailable");
    glfwMakeContextCurrent(window);
    const bool passed=[&]() {
        arbitgl::GlFuncs gl;std::string error;
        if(!arbitgl::loadGlFunctions(gl,error)||!arbitgl::loadGl43Functions(gl))
            return expect(false,"camera-wide-line-predicate probe GL4.3 functions unavailable");
        const auto tests=ec::cases();
        // Test-local logical admission, never a scene/query budget conversion.
        // Include constructor/predicate scratch, retained cases, native input
        // and six readback-sized static payloads for native/oracle/decoded copies.
        // Dynamic CPU oracle and actual compiled resources remain separate.
        if(tests.empty()||tests.size()>1024||tests.capacity()>1024)return expect(false,"camera-wide-line-predicate probe case admission");
        std::uint64_t admittedWork=0,admittedBytes=tests.capacity()*sizeof(ec::Case);
        for(const auto& test:tests){admittedWork+=ec::work(test);
            admittedBytes+=ec::kProbeScratchBytes+2*sizeof(el::ProbeInput)+6*sizeof(el::ProbeResult);}
        if(admittedWork>1024*ec::kMaximumProbeWork
            ||admittedBytes>1024*(ec::kProbeScratchBytes+2*sizeof(el::ProbeInput)+6*sizeof(el::ProbeResult)+sizeof(ec::Case)))
            return expect(false,"camera-wide-line-predicate probe derived envelope exceeded");
        std::vector<el::ProbeInput> inputs;inputs.reserve(tests.size());
        for(const auto& test:tests)inputs.push_back(test.input);
        struct Resources {
            arbitgl::GlFuncs& gl;GLuint shader=0,program=0,buffers[2]{};GLsync fence=nullptr;
            ~Resources(){if(fence)gl.DeleteSync(fence);
                gl.UseProgram(0);
                gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,0,0);gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,1,0);
                gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,0);gl.DeleteBuffers(2,buffers);
                if(program)gl.DeleteProgram(program);
                if(shader)gl.DeleteShader(shader);}
        } owned{gl};
        owned.shader=gl.CreateShader(GL_COMPUTE_SHADER);const auto source=el::glslProbeSource();const char* text=source.c_str();
        gl.ShaderSource(owned.shader,1,&text,nullptr);gl.CompileShader(owned.shader);GLint compiled=0;
        gl.GetShaderiv(owned.shader,GL_COMPILE_STATUS,&compiled);
        if(!compiled){char log[4096]{};gl.GetShaderInfoLog(owned.shader,sizeof(log),nullptr,log);
            std::cerr<<"camera-wide-line-predicate shader: "<<log<<'\n';return false;}
        owned.program=gl.CreateProgram();gl.AttachShader(owned.program,owned.shader);gl.LinkProgram(owned.program);
        GLint linked=0;gl.GetProgramiv(owned.program,GL_LINK_STATUS,&linked);
        if(!linked){char log[4096]{};gl.GetProgramInfoLog(owned.program,sizeof(log),nullptr,log);
            std::cerr<<"camera-wide-line-predicate program: "<<log<<'\n';return false;}
        gl.GenBuffers(2,owned.buffers);
        if(!owned.buffers[0]||!owned.buffers[1])return expect(false,"camera-wide-line-predicate native buffer allocation");
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,owned.buffers[0]);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER,inputs.size()*sizeof(el::ProbeInput),inputs.data(),GL_STATIC_DRAW);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,owned.buffers[1]);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER,tests.size()*sizeof(el::ProbeResult),nullptr,GL_DYNAMIC_READ);
        if(glGetError()!=GL_NO_ERROR)return expect(false,"camera-wide-line-predicate native buffer storage");
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,0,owned.buffers[0]);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,1,owned.buffers[1]);gl.UseProgram(owned.program);
        gl.DispatchCompute(static_cast<GLuint>(tests.size()),1,1);
        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
        owned.fence=gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);
        if(!owned.fence)return expect(false,"camera-wide-line-predicate native fence creation");
        const auto completion=gl.ClientWaitSync(owned.fence,GL_SYNC_FLUSH_COMMANDS_BIT,5000000000ull);
        if(completion!=GL_ALREADY_SIGNALED&&completion!=GL_CONDITION_SATISFIED)
            return expect(false,"camera-wide-line-predicate native command did not complete");
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,owned.buffers[1]);
        const auto* mapped=static_cast<const std::uint8_t*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,
            tests.size()*sizeof(el::ProbeResult),GL_MAP_READ_BIT));
        if(!mapped)return expect(false,"camera-wide-line-predicate native arithmetic readback unavailable");
        bool ok=true;for(std::size_t k=0;k<tests.size();k++){
            el::ProbeResult actual{};std::memcpy(actual.data(),mapped+k*sizeof(el::ProbeResult),sizeof(actual));
            if(!ec::compare(tests[k],actual,error)){std::cerr<<"FAIL: GL camera-wide-line-predicate arithmetic: "<<error<<'\n';ok=false;}}
        if(!gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER))ok=false;
        return ok;
    }();
    glfwMakeContextCurrent(previous);glfwDestroyWindow(window);return passed;
#endif
}

bool cameraPixelSignNativeProbe()
{
#if defined(__APPLE__)
    return expect(false,"camera-pixel-sign probe requires GLSL4.30");
#else
    namespace ec=videohelper::tests::camerapixelsigncases;namespace el=arbitgpu::camerapixelsigns;
    auto* previous=glfwGetCurrentContext();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    auto* window=glfwCreateWindow(1,1,"camera-pixel-sign-arithmetic-probe",nullptr,nullptr);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    if(!window)return expect(false,"camera-pixel-sign probe GL4.3 context unavailable");
    glfwMakeContextCurrent(window);
    const bool passed=[&]() {
        arbitgl::GlFuncs gl;std::string error;
        if(!arbitgl::loadGlFunctions(gl,error)||!arbitgl::loadGl43Functions(gl))
            return expect(false,"camera-pixel-sign probe GL4.3 functions unavailable");
        const auto tests=ec::cases();
        // Test-local logical admission, never a scene/query budget conversion.
        // Include constructor/sign/host-preparation scratch, retained cases, native input
        // and six readback-sized static payloads for native/oracle/decoded copies.
        // Dynamic CPU oracle and actual compiled resources remain separate.
        if(tests.empty()||tests.size()>1024||tests.capacity()>1024)return expect(false,"camera-pixel-sign probe case admission");
        std::uint64_t admittedWork=0,admittedBytes=tests.capacity()*sizeof(ec::Case);
        for(const auto& test:tests){admittedWork+=ec::work(test);
            admittedBytes+=ec::kProbeScratchBytes+2*sizeof(el::ProbeInput)+6*sizeof(el::ProbeResult);}
        if(admittedWork>1024*ec::kMaximumProbeWork
            ||admittedBytes>1024*(ec::kProbeScratchBytes+2*sizeof(el::ProbeInput)+6*sizeof(el::ProbeResult)+sizeof(ec::Case)))
            return expect(false,"camera-pixel-sign probe derived envelope exceeded");
        std::vector<el::ProbeInput> inputs;inputs.reserve(tests.size());
        for(const auto& test:tests)inputs.push_back(test.input);
        struct Resources {
            arbitgl::GlFuncs& gl;GLuint shader=0,program=0,buffers[2]{};GLsync fence=nullptr;
            ~Resources(){if(fence)gl.DeleteSync(fence);
                gl.UseProgram(0);
                gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,0,0);gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,1,0);
                gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,0);gl.DeleteBuffers(2,buffers);
                if(program)gl.DeleteProgram(program);
                if(shader)gl.DeleteShader(shader);}
        } owned{gl};
        owned.shader=gl.CreateShader(GL_COMPUTE_SHADER);const auto source=el::glslProbeSource();const char* text=source.c_str();
        gl.ShaderSource(owned.shader,1,&text,nullptr);gl.CompileShader(owned.shader);GLint compiled=0;
        gl.GetShaderiv(owned.shader,GL_COMPILE_STATUS,&compiled);
        if(!compiled){char log[4096]{};gl.GetShaderInfoLog(owned.shader,sizeof(log),nullptr,log);
            std::cerr<<"camera-pixel-sign shader: "<<log<<'\n';return false;}
        owned.program=gl.CreateProgram();gl.AttachShader(owned.program,owned.shader);gl.LinkProgram(owned.program);
        GLint linked=0;gl.GetProgramiv(owned.program,GL_LINK_STATUS,&linked);
        if(!linked){char log[4096]{};gl.GetProgramInfoLog(owned.program,sizeof(log),nullptr,log);
            std::cerr<<"camera-pixel-sign program: "<<log<<'\n';return false;}
        gl.GenBuffers(2,owned.buffers);
        if(!owned.buffers[0]||!owned.buffers[1])return expect(false,"camera-pixel-sign native buffer allocation");
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,owned.buffers[0]);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER,inputs.size()*sizeof(el::ProbeInput),inputs.data(),GL_STATIC_DRAW);
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,owned.buffers[1]);
        gl.BufferData(GL_SHADER_STORAGE_BUFFER,tests.size()*sizeof(el::ProbeResult),nullptr,GL_DYNAMIC_READ);
        if(glGetError()!=GL_NO_ERROR)return expect(false,"camera-pixel-sign native buffer storage");
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,0,owned.buffers[0]);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER,1,owned.buffers[1]);gl.UseProgram(owned.program);
        gl.DispatchCompute(static_cast<GLuint>(tests.size()),1,1);
        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
        owned.fence=gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);
        if(!owned.fence)return expect(false,"camera-pixel-sign native fence creation");
        const auto completion=gl.ClientWaitSync(owned.fence,GL_SYNC_FLUSH_COMMANDS_BIT,5000000000ull);
        if(completion!=GL_ALREADY_SIGNALED&&completion!=GL_CONDITION_SATISFIED)
            return expect(false,"camera-pixel-sign native command did not complete");
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,owned.buffers[1]);
        const auto* mapped=static_cast<const std::uint8_t*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,
            tests.size()*sizeof(el::ProbeResult),GL_MAP_READ_BIT));
        if(!mapped)return expect(false,"camera-pixel-sign native arithmetic readback unavailable");
        bool ok=true;for(std::size_t k=0;k<tests.size();k++){
            el::ProbeResult actual{};std::memcpy(actual.data(),mapped+k*sizeof(el::ProbeResult),sizeof(actual));
            if(!ec::compare(tests[k],actual,error)){std::cerr<<"FAIL: GL camera-pixel-sign arithmetic: "<<error<<'\n';ok=false;}}
        if(!gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER))ok=false;
        return ok;
    }();
    glfwMakeContextCurrent(previous);glfwDestroyWindow(window);return passed;
#endif
}

bool queryArithmeticSourceChecks()
{
    namespace ov=arbitgpu::opaquevisibility;
    const auto defaultGl=std::string("\nuniform sampler2D uOpaqueVisibility;\nuniform vec4 uOpaqueControl;\n#define SV3 "
        "vec3\n#define SV4 vec4\n#define SC_TEXELS (textureSize(uOpaqueVisibility,0).x*textureSize(uOpaqueVisibility,0).y)\n#define SC_OUT(T,N) , out T N\n#define SC_STACK_ARGS , inout int nodes[SC_STACK_CAPACITY], inout int pending, int capacity\n#define ST_ARGS\n#define ST_PASS\n#define ST_READ "
        "texelFetch(uOpaqueVisibility,ivec2(index%1024,index/1024),0)\n")+ov::shaderBounds()+ov::shader;
    const auto defaultMetal=std::string("\n#define SV3 float3\n#define SV4 float4\n#define SC_TEXELS (opaqueTexture.get_width()*opaqueTexture.get_height())\n#define SC_OUT(T,N) , thread T& N\n#define SC_STACK_ARGS , thread int* nodes, thread int& pending, int capacity\n#define ST_ARGS , texture2d<float> "
        "opaqueTexture\n#define ST_PASS , opaqueTexture\n#define ST_READ opaqueTexture.read(uint2(index%1024,index/1024))\n")+ov::shaderBounds()+ov::shader;
    bool ok=expect(ov::glslShader()==defaultGl&&ov::glslShader(false)==defaultGl&&ov::metalShader()==defaultMetal,
        "query contract preserves default GL330 and Metal source bytes");
    const auto renamed=[](std::string text) {
        const std::pair<const char*,const char*> names[]{{"opaque","glass"},{"Opaque","Glass"},
            {"SceneClosestHit","GlassClosestHit"},{"scene","glassScene"}};
        for(const auto& name:names)for(std::size_t at=0;(at=text.find(name.first,at))!=std::string::npos;at+=std::char_traits<char>::length(name.second))
            text.replace(at,std::char_traits<char>::length(name.first),name.second);
        return "\n#undef SV3\n#undef SV4\n#undef SC_TEXELS\n#undef SC_OUT\n#undef ST_ARGS\n#undef ST_PASS\n#undef ST_READ\n"+text;
    };
    ok&=expect(arbitgpu::smoothglass::queryShader(false)==renamed(defaultGl)
        &&arbitgpu::smoothglass::queryShader(true)==renamed(defaultMetal)
        &&arbitgpu::smoothglass::queryShader(true,true)==renamed(defaultMetal),
        "qualified helper preserves both default emissions and never qualifies Metal");
    const auto qualified=ov::glslShader(true);
    for(const auto* stage:{"precise float nearT=0.0,farT=limit;","SC_OUT(precise float,nearDistance)",
        "SC_OUT(precise float,distance) SC_OUT(precise SV3,normal)","precise SV3 e1=b-a,e2=c-a,p=cross(direction,e2);",
        "precise float det=dot(e1,p);","precise float u=dot(t,p)/det;","precise float v=dot(direction,q)/det;",
        "precise float barycentricSum=u+v;","precise float rayLength=length(rayDirection);",
        "precise SV3 direction=rayDirection/rayLength;","precise SV3 localOrigin=","precise SV3 localDirection=",
        "precise float localDirectionLength=length(localDirection);","precise float normalLength=length(worldNormal);",
        "precise SV3 position=origin+direction*distance;","precise SV3 outwardNormal=worldNormal/normalLength;",
        "precise float enteringDot=dot(direction,outwardNormal);"})
        ok&=expect(qualified.find(stage)!=std::string::npos,"GL430 query stages and predicate operands are precise in their own functions");
    ok&=expect(arbitgpu::glasscaustics::opengl::querySource().find("precise float barycentricSum=u+v;")!=std::string::npos
        &&arbitgpu::smoothglass::queryShader(false,true)==renamed(qualified),
        "actual GL430 assembly and selected-camera helper share the qualified query");
    const auto direct=arbitgpu::glasscaustics::opengl::directGratingComputeSource();
    for(const auto* emitted:{"#version 430 core\n","precise float barycentricSum=u+v;",
        "EndpointInput endpointInput=endpoints[index];",
        "sceneClosestHit(endpointInput.originMinimum.xyz,endpointInput.directionMaximum.xyz,",
        "sceneClosestHit(endpoint.position,uSource.xyz,sourceOrigin.w,uSource.w,uOpaqueControl)",
        "sourceD[bandGroup]*cosine","result[base]=endpointInput.primitive;","result[base+4]=sourceIds;",
        "any(lessThanEqual(barycentric,vec3(1e-5)))"})
        ok&=expect(direct.find(emitted)!=std::string::npos,"direct GLSL430 producer retains native inputs, both production query legs and band arithmetic");
    const auto cameraTrace=arbitgpu::glasscaustics::opengl::cameraGlassGratingComputeSource();
    for(const auto* emitted:{"#version 430 core\n","precise float barycentricSum=u+v;",
        "CameraTraceInput cameraInput=inputs[sampleIndex];",
        "causticCameraRay(vec2(cameraInput.pixel.xy)+vec2(dx,dy),uRotation,uProjection,uExtent)",
        "sceneClosestHit(branch.origin,branch.direction,minimum,maximum,uOpaqueControl)",
        "CAUSTIC_MAX_PENDING","CAUSTIC_MAX_VISITS","CAUSTIC_MAX_INTERFACES","uIor[band/4][band%4]",
        "precise float reflectedFraction=branch.fraction*visit.fresnel;",
        "precise float transmittedFraction=branch.fraction*(1.0-visit.fresnel);",
        "child.token=1u;","child.token=65u;","result[base+12]=path.ancestry[0]",
        "all(greaterThan(barycentric,vec3(1e-5)))","cameraTraceOutside(origin,glassObject,glassMaterial)"})
        ok&=expect(cameraTrace.find(emitted)!=std::string::npos,
            "private camera GLSL430 retains actual rays, both native branches, fixed caps and ancestry");
    ok&=expect(cameraTrace.find("minimumBranchFraction")==std::string::npos,
        "private camera branch traversal has no positive fraction pruning");
    // Codec assertions are not a completed native contribution.
    arbitgpu::glasscaustics::opengl::DirectGratingPacked packed{};
    packed[0]={0xf1234567u,0xe3456789u,0xd56789abu,0xc789abcdu};packed[1]={1,0,0,0};
    packed[25]={1,0,1,2};packed[26]={8,1,0,0};
    arbitgpu::glasscaustics::NativeDirectGratingContributionSample decoded;
    ok&=expect(arbitgpu::glasscaustics::opengl::decodeDirectGratingContribution(packed,1,decoded)
        &&decoded.input.primitive==packed[0],"direct packed input retains all 32 primitive bits without float conversion");
    packed[26][3]=1;
    ok&=expect(!arbitgpu::glasscaustics::opengl::decodeDirectGratingContribution(packed,1,decoded),
        "direct codec refuses nonzero reserved output words");
    // These are emission assertions only. They establish no native arithmetic result.
    return ok;
}

bool readQueryContract(const arbitgpu::NativeFixtureSceneFrame& frame,arbitgpu::OpenGlQueryArithmeticRole role,
    const arbitgpu::glasscaustics::SubmissionIdentity* expected,std::shared_ptr<const arbitgpu::OpenGlQueryArithmeticEvidence>& evidence,
    std::string& error)
{
    if(!arbitgpu::readOpenGlQueryArithmeticEvidenceForFixture(frame,role,evidence,error)||!evidence||!evidence->program
        ||(expected&&!(evidence->submission==*expected))
        ||!arbitgpu::validateOpenGlQueryArithmeticEvidenceForFixture(frame,evidence,evidence->submission,error))return false;
    const auto& identity=*evidence->program;
    if(identity.contractVersion!=1||identity.glslVersion!=430||!identity.contextGeneration||!identity.ownerGeneration
        ||!identity.linkedProgram||identity.context!=reinterpret_cast<std::uintptr_t>(glfwGetCurrentContext())
        ||identity.rendererGeneration!=evidence->submission.rendererGeneration||!identity.stageCount||identity.stageCount>identity.stages.size())return false;
    for(unsigned stage=0;stage<identity.stageCount;++stage) {
        const auto& retained=identity.stages[stage];std::uint64_t fingerprint=14695981039346656037ull;
        for(unsigned char byte:retained.source)fingerprint=(fingerprint^byte)*1099511628211ull;
        if(retained.source.rfind("#version 430 core\n",0)!=0||retained.fingerprint!=fingerprint)return false;
    }
    return true;
}

bool glassGratingTerminalNativeChecks(arbitgpu::NativeFixtureSceneBackend& backend,std::string& error)
{
    namespace gc=arbitgpu::glasscaustics;namespace gt=glasscaustictests;
    int major=0,minor=0;glGetIntegerv(GL_MAJOR_VERSION,&major);glGetIntegerv(GL_MINOR_VERSION,&minor);
    if(major<4||(major==4&&minor<3)) {
        std::cout<<"UNVERIFIED: native glass/grating terminal fixtures require OpenGL 4.3\n";return true;
    }
    const auto fail=[&](const char* message){error=message;return false;};
    // Focused packing negatives must refuse before lossy integral conversion.
    std::array<gc::Texel,gc::kGratingWorkReceiptStride> workWords{};workWords[0]={1,3,4,0};workWords[1][2]=1;
    gc::NativeGratingReadback::WorkReceipt workSentinel;
    if(!gc::opengl::decodeGratingWork(workWords.data(),workSentinel)||workSentinel.incidentCount!=1)return fail("OpenGL dense work packing did not decode");
    workWords[0][0]=1.5f;
    if(gc::opengl::decodeGratingWork(workWords.data(),workSentinel)||workSentinel.incidentCount!=1)return fail("OpenGL dense work accepted a fractional count");
    std::array<gc::Texel,gc::kGratingChildReceiptStride> childWords{};childWords[0]={0,1,1,0};childWords[1]={1,0,4,0};
    gc::NativeGratingReadback::ChildReceipt childSentinel;
    if(!gc::opengl::decodeGratingChild(childWords.data(),childSentinel))return fail("OpenGL child packing did not decode");
    childWords[7][0]=1;
    if(gc::opengl::decodeGratingChild(childWords.data(),childSentinel)||childSentinel.order!=0)return fail("OpenGL child accepted nonzero reserved suffix");
    std::array<gc::Texel,gc::kDepositReceiptStride> depositWords{};depositWords[0][2]=2;depositWords[22]={1,65,0,0};depositWords[30]={1,0,1,0};
    gc::NativeDeposit depositSentinel;
    if(!gc::opengl::decodeGratingDeposit(depositWords.data(),depositSentinel)||depositSentinel.ancestry[1]!=65)return fail("OpenGL smooth child ancestry did not decode");
    depositWords[22][1]=2;
    if(gc::opengl::decodeGratingDeposit(depositWords.data(),depositSentinel)||depositSentinel.ancestry[1]!=65)return fail("OpenGL child accepted a multiple-scatter ancestry token");
    auto fixture=gt::ordinaryFixture();fixture.scene->lights[0].transform.rotation={0,0,0,1};
    fixture.scene->cameras[0].transform.translation={0,0,1};
    fixture.scene->cameras[0].transform.rotation={0,float(std::sqrt(.5)),0,float(std::sqrt(.5))};
    const auto target=fixture.scene->objects[1];
    for(std::size_t v=target.firstVertex;v<target.firstVertex+target.vertexCount;++v) {
        auto& vertex=fixture.scene->vertices[v];vertex.position.z=-6+.3f*vertex.position.x;
        vertex.normal={float(-.3/std::sqrt(1.09)),0,float(1/std::sqrt(1.09))};vertex.tangent={0,1,0,1};
        vertex.uv={vertex.position.y/12,-vertex.position.x/12};
    }
    // The reciprocal direction is along y. All three propagating orders turn
    // away from the glass silhouette toward a real offset ordinary receiver.
    const auto receiverIndex=fixture.scene->objectCount;
    videohelper::tests::smoothglassfixture::receiver(*fixture.scene,-6,{0,0,0});
    auto& receiver=fixture.scene->objects[receiverIndex];
    const HarmonicMIDI::grid::SceneVec3 points[]={{-4,-1,-8},{-4,1,-8},{-4,1,6},{-4,-1,6}};
    for(unsigned v=0;v<4;++v) {
        auto& vertex=fixture.scene->vertices[receiver.firstVertex+v];vertex.position=points[v];
        vertex.normal={1,0,0};vertex.tangent={0,1,0,1};
    }
    auto receiverProgram=std::make_shared<gt::Program>(*fixture.program->objectPrograms[0]);
    receiverProgram->object=receiver.id;receiverProgram->authoredMaterial=receiver.material;
    receiverProgram->parameters.identifiers[0]=receiver.material.value;
    receiverProgram->programIdentity="gl-grating-terminal-receiver";receiverProgram->bindingDigest="gl-grating-terminal-receiver";
    receiverProgram->spectralReflectance->coefficients={.1f,.2f,.3f,.4f,.5f,.6f,.7f,.8f};
    fixture.program->objectPrograms.push_back(receiverProgram);
    fixture.program->backend=arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    for(auto& child:fixture.program->objectPrograms) {
        auto material=std::make_shared<gt::Program>(*child);material->backend=arbitgpu::NativeFixtureMaterialBackend::OpenGl;child=material;
    }
    const auto resources=backend.prepare(fixture.scene,fixture.program);if(!resources.prepared){error=resources.error;return false;}
    if(!fixture.prepare(error))return false;
    gc::Prepared prepared=fixture.prepared;
    arbitgpu::NativeFixtureSceneRuntimeInputs runtime;runtime.linearColor=true;
    gc::SchedulingDescription scheduling;scheduling.entrySubdivisions=1;
    if(!gc::buildFrameImpl(prepared,fixture.frame.instances,nullptr,32,32,fixture.frame,error,scheduling))return false;
    const auto staging=arbitgpu::renderOpenGlGlassCausticScene(fixture.scene,resources.resources,fixture.ordinaryLight,
        fixture.program->object,fixture.program->authoredMaterial,*fixture.policy,scheduling,32,32,runtime);
    if(!staging.rendered||!staging.frame){error=staging.error;return false;}
    auto description=diffractionmaterial::makeAluminiumBinaryGratingPreset();description.spectrum.firstOrder=description.spectrum.lastOrder=1;
    description.microstructure.grooveDepthNanometres=.9f;description.roughness={0,0};
    const auto admitted=diffractionmaterial::admit(description,error);if(!admitted)return false;
    const auto material=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*admitted);
    const auto compileGrating=[&](const diffractionmaterial::Description& physical,const std::string& digest) {
        diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest request;
        request.version=diffractionmaterialbinding::kSceneReceiverWireVersion;request.scene=fixture.scene->id;request.object=target.id;
        request.sceneRevision=request.structuralRevision=request.evaluationRevision=request.materialRevision=1;
        request.material=physical;request.structuralDigest=digest;
        request.lighting.version=3;request.lighting.pathCount=1;request.lighting.traceMaximumDistance=fixture.frame.sourceDirectionTrace[3];
        request.lighting.sceneReceiverTransport=diffractionmaterial::SceneReceiverTransport::AuthoredSpectralLambertian;
        request.lighting.paths[0].kind=diffractionmaterial::LightingPathKind::Direct;
        for(unsigned a=0;a<3;++a)request.lighting.paths[0].incident.direction[a]=fixture.frame.sourceDirectionTrace[a];
        for(unsigned b=0;b<8;++b)request.lighting.paths[0].incident.radiance[b]=fixture.frame.sourceRadiance[b/4][b%4];
        return gc::compileDiscreteGrating(*fixture.scene,request,arbitgpu::NativeFixtureMaterialBackend::OpenGl,error);
    };
    const auto grating=compileGrating(description,material->structuralDigest());if(!grating)return false;
    std::shared_ptr<const gc::GratingIncidentBinding> binding;
    if(!gc::prepareGratingIncidentBinding(prepared,fixture.frame,grating,material,.25,binding,error))return false;
    arbitgpu::NativeRawPassPixels before,after;if(!staging.frame->readRawPass(renderpassoutput::Output::Color,before,error))return false;
    gc::NativeReadback receipt;gc::EnergyLedger ledger;
    if(!arbitgpu::produceOpenGlGlassGratingTerminalReadback(staging.frame,binding,receipt,ledger,error))return false;
    if(!receipt.gratingIncidents)return fail("OpenGL terminal query has no native adjunct submission");
    using QueryRole=arbitgpu::OpenGlQueryArithmeticRole;
    using QueryEvidence=arbitgpu::OpenGlQueryArithmeticEvidence;
    std::shared_ptr<const QueryEvidence> terminalQuery;
    if(!readQueryContract(*staging.frame,QueryRole::GratingPoint,&receipt.gratingIncidents->submission,terminalQuery,error))
        return fail("OpenGL terminal query lacks a completed linked-source contract after temporary cleanup");
    const auto refuseQueryRecord=[&](auto mutate) {
        auto record=std::make_shared<QueryEvidence>(*terminalQuery);
        auto program=std::make_shared<arbitgpu::OpenGlQueryArithmeticProgramIdentity>(*record->program);
        record->program=program;mutate(*record,*program);
        return !arbitgpu::validateOpenGlQueryArithmeticEvidenceForFixture(*staging.frame,record,record->submission,error);
    };
    auto wrongQueryFence=terminalQuery->submission;wrongQueryFence.completedFence++;
    if(arbitgpu::validateOpenGlQueryArithmeticEvidenceForFixture(*staging.frame,terminalQuery,wrongQueryFence,error)
        ||!refuseQueryRecord([](auto&,auto&) {})
        ||!refuseQueryRecord([](auto&,auto& p){p.stages[0].source+="\n";})
        ||!refuseQueryRecord([](auto&,auto& p){p.glslVersion=330;})
        ||!refuseQueryRecord([](auto&,auto& p){p.contextGeneration++;})
        ||!refuseQueryRecord([](auto&,auto& p){p.ownerGeneration++;})
        ||!refuseQueryRecord([](auto&,auto& p){p.linkedProgram++;})
        ||!refuseQueryRecord([](auto& r,auto&){r.submission.completedFence++;}))
        return fail("OpenGL query contract accepted caller records, changed source/profile/program/generation or another fence");
    if(!receipt.gratingIncidents||!receipt.gratingIncidents->childrenComplete||receipt.gratingIncidents->incidents.empty()
        ||receipt.gratingIncidents->paths.size()!=fixture.frame.work.size())return fail("OpenGL terminal grating lacks dense native child proof");
    std::array<std::array<bool,3>,8> orders{};bool deposited=false,visible=false,escaped=false;
    const auto triangles=gt::worldTriangles(*fixture.scene);
    for(const auto& parent:receipt.gratingIncidents->incidents) {
        const double cosine=-gt::inner(gt::doublePoint(parent.direction),gt::doublePoint(parent.normal));
        const double x=-gt::inner(gt::doublePoint(parent.direction),gt::doublePoint(parent.tangent));
        const double y=-gt::inner(gt::doublePoint(parent.direction),gt::doublePoint(parent.bitangent));
        const double wavelength=description.spectrum.wavelengthsNanometres[parent.band];
        const double n=description.substrate.refractiveIndex,k=description.substrate.extinctionCoefficient;
        const double reflectance=((1-n)*(1-n)+k*k)/((1+n)*(1+n)+k*k);
        for(unsigned o=0;o<parent.orderCount;++o) {
            const auto& order=parent.orders[o];const double dx=-x+order.order*wavelength/description.geometry.grooveSpacingNanometres;
            const double dy=-y,z=std::sqrt(1-dx*dx-dy*dy);
            const double phase=2*std::acos(-1.0)*description.microstructure.grooveDepthNanometres*(cosine+z)/wavelength;
            const double duty=description.microstructure.dutyCycle;
            const double efficiency=order.order==0?((1-duty)*(1-duty)+duty*duty+2*duty*(1-duty)*std::cos(phase))*z/cosine
                :std::pow(std::sin(std::acos(-1.0)*order.order*duty)/(std::acos(-1.0)*order.order),2)*2*(1-std::cos(phase))*z/cosine;
            const auto direction=gt::plus(gt::plus(gt::plus(gt::D3{},gt::doublePoint(parent.tangent),dx),gt::doublePoint(parent.bitangent),dy),gt::doublePoint(parent.normal),z);
            for(unsigned a=0;a<3;++a)if(std::abs(direction[a]-order.direction[a])>3e-5)return fail("OpenGL terminal signed vector differs from independent grating equation");
            if(std::abs(order.power-parent.power*reflectance*efficiency)>parent.power*2e-7)
                return fail("OpenGL terminal physical band flux differs from independent binary profile");
        }
    }
    for(const auto& child:receipt.gratingIncidents->children) {
        const auto& parent=receipt.gratingIncidents->incidents[child.incident];orders[parent.band][unsigned(child.order+1)]=true;
        const auto order=std::find_if(parent.orders.begin(),parent.orders.begin()+parent.orderCount,[&](const auto& o){return o.order==child.order;});
        if(order==parent.orders.begin()+parent.orderCount)return fail("OpenGL terminal child lost its signed parent order");
        const auto origin=gt::plus(gt::doublePoint(parent.hit.position),gt::doublePoint(parent.normal),gc::bias(parent.hit.position));
        const auto oracle=gt::closest(triangles,origin,gt::doublePoint(order->direction),1e-5,fixture.frame.sourceDirectionTrace[3]);
        if(child.disposition==gc::NativeGratingReadback::Disposition::Receiver) {
            if(!oracle.found||oracle.object!=child.hit.objectId||oracle.triangle!=child.hit.triangleId
                ||child.hit.objectId!=receiver.id.value)return fail("OpenGL terminal order differs from independent authored receiver query");
            for(unsigned a=0;a<3;++a)if(std::abs(oracle.position[a]-child.hit.position[a])>5e-4)
                return fail("OpenGL terminal receiver point differs from independent authored query");
            const auto d=std::find_if(receipt.deposits.begin(),receipt.deposits.end(),[&](const auto& v){return v.gratingVersion&&v.gratingIncident==child.incident&&v.gratingOrder==child.order;});
            if(d==receipt.deposits.end()||d->ancestry!=parent.ancestry||!gc::nearPower(d->incidentPower,order->power)
                ||!gc::nearPower(d->outgoingPower,order->power*(.1+.1*parent.band)))return fail("OpenGL terminal receiver lost exact ancestry or physical band rho");
            deposited=true;visible|=d->cameraVisible;
        } else {
            if(child.disposition!=gc::NativeGratingReadback::Disposition::Escaped||oracle.found
                ||!gc::nearPower(child.escapedPower,order->power))return fail("OpenGL terminal miss was not an actual independently queried escape");
            escaped=true;
        }
    }
    for(const auto& band:orders)for(bool value:band)if(!value)return fail("OpenGL terminal fixture lacks a nonzero signed order in every band");
    if(!deposited||!visible||!escaped||std::none_of(receipt.spectralRadiance.begin(),receipt.spectralRadiance.end(),[](float value){return value>0;}))
        return fail("OpenGL terminal fixture lacks native receiver deposition and spectral accumulation");
    for(unsigned b=0;b<8;++b)if(!gc::nearPower(ledger.launched[b],ledger.deposited[b]+ledger.absorbed[b]+ledger.escaped[b]+ledger.unresolved[b])
        ||ledger.unresolved[b]>ledger.launched[b]*1e-4*(1+2e-4))return fail("OpenGL terminal original energy ledger did not close");
    auto bound=fixture.frame;bound.gratingIncidentBinding=binding;bound.identity=binding->identity;
    gc::EnergyLedger ignored;
    if(gc::validateNativeReadback(prepared,bound,receipt,ignored,error,&receipt.gratingIncidents->submission))
        return fail("OpenGL terminal prerequisite enabled default combined Color");
    auto broken=receipt;auto proof=std::make_shared<gc::NativeGratingReadback>(*receipt.gratingIncidents);proof->children.pop_back();broken.gratingIncidents=proof;
    if(gc::validateNativeReadback(prepared,bound,broken,ignored,error,&proof->submission,true))return fail("OpenGL terminal accepted incomplete child proof");
    broken=receipt;const auto coupled=std::find_if(broken.deposits.begin(),broken.deposits.end(),[](const auto& value){return value.gratingVersion!=0;});
    if(coupled==broken.deposits.end()){return fail("OpenGL terminal fixture has no exact coupled deposit");}coupled->ancestry[0]^=64;
    if(gc::validateNativeReadback(prepared,bound,broken,ignored,error,&receipt.gratingIncidents->submission,true))return fail("OpenGL terminal accepted wrong parent ancestry");
    broken=receipt;proof=std::make_shared<gc::NativeGratingReadback>(*receipt.gratingIncidents);proof->paths[0].queryWork++;
    broken.gratingIncidents=proof;
    if(gc::validateNativeReadback(prepared,bound,broken,ignored,error,&proof->submission,true))return fail("OpenGL terminal accepted incorrect dense extra query work");
    broken=receipt;broken.deposition.submission.completedFence++;
    if(gc::validateNativeReadback(prepared,bound,broken,ignored,error,&receipt.gratingIncidents->submission,true))return fail("OpenGL terminal accepted a foreign completed deposition fence");
    gc::NativeReadback repeated;gc::EnergyLedger repeatEnergy;
    if(!arbitgpu::produceOpenGlGlassGratingTerminalReadback(staging.frame,binding,repeated,repeatEnergy,error))return false;
    if(receipt.spectralRadiance!=repeated.spectralRadiance)return fail("OpenGL terminal repeated native accumulation changed");
    if(!staging.frame->readRawPass(renderpassoutput::Output::Color,after,error)||before.bytes!=after.bytes)
        return fail("OpenGL terminal prerequisite published partial or completed Color");
    // Prepare the actual mixed root, without relabeling the grating as PBR.
    auto mixedRoot=std::make_shared<gt::Program>(*grating);
    auto glassChild=std::make_shared<gt::Program>(*fixture.program);glassChild->objectPrograms.clear();
    mixedRoot->objectPrograms={glassChild,receiverProgram};
    std::sort(mixedRoot->objectPrograms.begin(),mixedRoot->objectPrograms.end(),[](const auto& a,const auto& b){return a->object.value<b->object.value;});
    const arbitgpu::smoothglass::DistinctGratingAdmission selection{target.id,material};
    auto privateInput=std::make_shared<gt::Visual3DScene>(*fixture.scene);
    if(backend.prepare(privateInput,mixedRoot).prepared)return fail("OpenGL default preparation accepted private zero-slope mixed material");
    const auto mixedResources=arbitgpu::prepareOpenGlGlassGratingResources(privateInput,mixedRoot,selection,grating);
    if(!mixedResources.prepared)return fail(mixedResources.error.c_str());
    // Original editable inputs no longer own native staging authority.
    privateInput->vertices[target.firstVertex].position.x+=.1f;mixedRoot->programIdentity+="edited";
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> mixedFrame;
    std::shared_ptr<const gc::GratingIncidentBinding> mixedBinding;
    if(!arbitgpu::stageOpenGlGlassGratingFrame(mixedResources.resources,fixture.ordinaryLight,*fixture.policy,32,32,runtime,.25,
        mixedFrame,mixedBinding,error,scheduling))return false;
    if(!mixedFrame||!mixedBinding||mixedFrame->colorImageHandle()||mixedFrame->depthImageHandle()
        ||!mixedBinding->frame.gratingStaging||!mixedBinding->prepared.glass.distinctGrating
        ||mixedBinding->prepared.glass.sourceMaterial->kind!=arbitgpu::NativeFixtureMaterialKind::DiffractionReflective
        ||mixedBinding->prepared.glass.sourceMaterial->programIdentity!=grating->programIdentity
        ||mixedBinding->prepared.glass.sourceScene==privateInput
        ||mixedBinding->prepared.glass.sourceScene->vertices[target.firstVertex].position.x==privateInput->vertices[target.firstVertex].position.x)
        return fail("OpenGL private staging lost immutable real mixed ownership or published image handles");
    if(mixedFrame->colorTextureDescriptor().complete()||mixedFrame->depthTextureDescriptor().complete())
        return fail("OpenGL private camera storage exposed unpublished camera attachments");
    for(const auto pass:renderpasscomposite::kInputs) {
        arbitgpu::NativeRawPassPixels unavailable;
        if(mixedFrame->passTextureDescriptor(pass).complete()||mixedFrame->readRawPass(pass,unavailable,error))
            return fail("OpenGL private camera storage exposed an unrendered raw pass");
    }
    if(backend.render(mixedBinding->prepared.glass.sourceScene,mixedResources.resources,32,32,runtime).rendered)
        return fail("OpenGL default renderer published private mixed staging Color");
    std::shared_ptr<const gc::NativeReadback> falsePhysical;gc::EnergyLedger falseEnergy;
    if(arbitgpu::readOpenGlGlassCausticReceiptForFixture(*mixedFrame,falsePhysical,falseEnergy,error))
        return fail("OpenGL staging fence fabricated a physical caustic readback");
    gc::NativeReadback mixedReceipt;gc::EnergyLedger mixedEnergy;
    if(!arbitgpu::produceOpenGlGlassGratingTerminalReadback(mixedFrame,mixedBinding,mixedReceipt,mixedEnergy,error))return false;
    if(!mixedReceipt.gratingIncidents||!mixedReceipt.gratingIncidents->childrenComplete
        ||mixedReceipt.spectralRadiance!=receipt.spectralRadiance||mixedEnergy.launched!=ledger.launched)
        return fail("OpenGL actual mixed native route differs from the independently checked terminal geometry and flux");
    // D is independently authored. Zero raw Li cannot drive this positive
    // native result, and the same geometry provides an exact doubled control.
    auto dimensionalScene=std::make_shared<gt::Visual3DScene>(*mixedBinding->prepared.glass.sourceScene);
    auto& dimensionalSource=dimensionalScene->lights[0].spectralSource;
    dimensionalSource.directionalIrradiance.version=1;
    dimensionalSource.directionalIrradiance.measure=diffractionmaterial::DirectionalSpectralMeasure::IrradianceCoefficient;
    for(unsigned b=0;b<8;++b)dimensionalSource.directionalIrradiance.coefficients[b]=2*fixture.frame.sourceRadiance[b/4][b%4];
    dimensionalSource.radiance.fill(0);
    const auto dimensionalResources=arbitgpu::prepareOpenGlGlassGratingResources(dimensionalScene,
        mixedBinding->prepared.glass.sourceMaterial,selection,grating);
    if(!dimensionalResources.prepared)return fail(dimensionalResources.error.c_str());
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> dimensionalFrame;
    std::shared_ptr<const gc::GratingIncidentBinding> dimensionalBinding;
    if(!arbitgpu::stageOpenGlGlassGratingFrame(dimensionalResources.resources,fixture.ordinaryLight,*fixture.policy,32,32,
        runtime,.25,dimensionalFrame,dimensionalBinding,error,scheduling))return false;
    const auto* sourceBinding=dynamic_cast<const gc::NativeDirectionalSourceBinding*>(dimensionalFrame.get());
    if(!sourceBinding||!sourceBinding->upload().complete||sourceBinding->upload().input.identifiers
        !=std::array<std::uint32_t,4>{1,1,1,fixture.ordinaryLight.value})
        return fail("OpenGL dimensional stage lacks actual typed native source ownership");
    if(arbitgpu::renderOpenGlGlassCausticScene(dimensionalBinding->prepared.glass.sourceScene,dimensionalResources.resources,
        fixture.ordinaryLight,fixture.program->object,fixture.program->authoredMaterial,*fixture.policy,scheduling,32,32,runtime).rendered)
        return fail("OpenGL directional source enabled ordinary Color before its camera consumer");
    gc::NativeReadback dimensionalReceipt;gc::EnergyLedger dimensionalEnergy;
    if(!arbitgpu::produceOpenGlGlassGratingTerminalReadback(dimensionalFrame,dimensionalBinding,dimensionalReceipt,dimensionalEnergy,error))return false;
    if(!dimensionalReceipt.directionalSource||!dimensionalReceipt.gratingIncidents
        ||dimensionalReceipt.directionalSource!=dimensionalReceipt.gratingIncidents->directionalSource
        ||dimensionalReceipt.directionalSource->echo.rawRadiance!=std::array<float,8>{}
        ||dimensionalReceipt.directionalSource->echo.input.coefficients!=dimensionalSource.directionalIrradiance.coefficients
        ||dimensionalReceipt.paths.size()!=mixedReceipt.paths.size()
        ||dimensionalReceipt.spectralRadiance.size()!=mixedReceipt.spectralRadiance.size()
        ||dimensionalFrame->colorTextureDescriptor().complete()||dimensionalFrame->depthTextureDescriptor().complete())
        return fail("OpenGL dimensional receipt lost independent D, zero raw Li or private owner proof");
    for(unsigned b=0;b<8;++b)if(!gc::nearPower(dimensionalEnergy.launched[b],2*mixedEnergy.launched[b]))
        return fail("OpenGL dimensional native flux reused raw Li or multiplied a second incident cosine");
    for(std::size_t p=0;p<dimensionalReceipt.spectralRadiance.size();++p)
        if(!gc::nearPower(dimensionalReceipt.spectralRadiance[p],2*mixedReceipt.spectralRadiance[p]))
            return fail("OpenGL dimensional native spectral accumulation differs from doubled physical source control");
    auto dimensionalBound=dimensionalBinding->frame;dimensionalBound.identity=dimensionalBinding->identity;
    dimensionalBound.gratingIncidentBinding=dimensionalBinding;
    if(gc::validateNativeReadback(dimensionalBinding->prepared,dimensionalBound,dimensionalReceipt,ignored,error,
        &dimensionalReceipt.gratingIncidents->submission,true))
        return fail("OpenGL dimensional readback accepted a missing native source binding");
    const auto refuseSourceProof=[&](auto mutate) {
        auto corrupt=dimensionalReceipt;auto source=std::make_shared<gc::NativeDirectionalSourceReceipt>(*corrupt.directionalSource);
        mutate(*source);auto adjunct=std::make_shared<gc::NativeGratingReadback>(*corrupt.gratingIncidents);
        adjunct->directionalSource=source;corrupt.directionalSource=source;corrupt.gratingIncidents=adjunct;
        return !gc::validateNativeReadback(dimensionalBinding->prepared,dimensionalBound,corrupt,ignored,error,
            &adjunct->submission,true,sourceBinding);
    };
    if(!refuseSourceProof([](auto& r){r.echo.input.identifiers[3]++;})
        ||!refuseSourceProof([](auto& r){r.echo.input.identifiers[2]=0;})
        ||!refuseSourceProof([](auto& r){r.echo.rawRadiance[0]=1;})
        ||!refuseSourceProof([](auto& r){*std::find_if(r.potential.begin(),r.potential.end(),[](float value){return value>0;})*=.5f;})
        ||!refuseSourceProof([](auto& r){r.submission.completedFence++;}))
        return fail("OpenGL dimensional receipt accepted a foreign source, unit, Li, potential or fence");
    if(gc::validateNativeReadback(dimensionalBinding->prepared,dimensionalBound,dimensionalReceipt,ignored,error,
        &dimensionalReceipt.gratingIncidents->submission,true,dynamic_cast<const gc::NativeDirectionalSourceBinding*>(mixedFrame.get())))
        return fail("OpenGL dimensional readback accepted another native frame's source binding");
    {
        // Separate native owner so the new direct ledger cannot affect the
        // later independent cell/camera fixtures' baseline reservations.
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> directFrame;
        std::shared_ptr<const gc::GratingIncidentBinding> directBinding;
        if(!arbitgpu::stageOpenGlGlassGratingFrame(dimensionalResources.resources,fixture.ordinaryLight,*fixture.policy,32,32,
            runtime,.25,directFrame,directBinding,error,scheduling))return false;
        const auto* directOwner=dynamic_cast<const gc::NativeDirectGratingIrradianceBinding*>(directFrame.get());
        if(!directOwner||directOwner->directEndpointUpload().complete)
            return fail("OpenGL staging fabricated a completed direct endpoint input");
        const auto endpoint=[&](const gc::GratingIncidentBinding& owner,gc::Point point) {
            const auto& triangle=owner.triangles.front();gc::NativeDirectGratingEndpointInput input;
            input.primitive={triangle.object,triangle.material,triangle.instance,triangle.triangle};
            const auto origin=gc::advance(point,triangle.normal,.1f);
            input.originMinimum={origin[0],origin[1],origin[2],gc::kRayMinimum};
            input.directionMaximum={-triangle.normal[0],-triangle.normal[1],-triangle.normal[2],1};return input;
        };
        const auto& triangle=directBinding->triangles.front();gc::Point clear{};
        const std::array<float,3> barycentric{.2f,.3f,.5f};
        for(unsigned corner=0;corner<3;++corner)clear=gc::advance(clear,triangle.positions[corner],barycentric[corner]);
        std::vector<gc::NativeDirectGratingEndpointInput> inputs{endpoint(*directBinding,clear)};
        gc::NativeDirectGratingContributionReadback direct;
        if(!arbitgpu::produceOpenGlDirectGratingContribution(directFrame,directBinding,inputs,.25,direct,error))return false;
        if(direct.samples.size()!=1||direct.samples[0].outcome!=gc::NativeDirectGratingContributionSample::Outcome::Positive
            ||direct.samples[0].endpoint.status!=1||direct.samples[0].sourceHit.status!=0
            ||!direct.samples[0].endpoint.work||!direct.samples[0].sourceHit.work||direct.bandWork!=8
            ||direct.queryWork!=direct.samples[0].endpoint.work+direct.samples[0].sourceHit.work
            ||direct.samples[0].source.rawRadiance!=std::array<float,8>{}
            ||!gc::validateDirectGratingContributionReadback(*directBinding,*directOwner,direct.submission,.25,direct,error)
            ||directFrame->colorImageHandle()||directFrame->depthImageHandle()||directFrame->colorTextureDescriptor().complete())
            return fail("OpenGL actual direct contribution lost query legs, D-only bands or private owner");
        for(unsigned band=0;band<8;++band) {
            const auto expected=dimensionalSource.directionalIrradiance.coefficients[band]*triangle.normal[2];
            if(!gc::nearDirectionalSourcePotential(direct.samples[0].surfaceIrradiance[band],expected))
                return fail("OpenGL native direct band differs from independent D times grating cosine");
        }
        std::shared_ptr<const QueryEvidence> directQuery;
        if(!readQueryContract(*directFrame,QueryRole::GratingDirect,&direct.submission,directQuery,error))
            return fail("OpenGL native direct query lacks its actual qualified linked program");
        const auto useProgram=reinterpret_cast<PFNGLUSEPROGRAMPROC>(glfwGetProcAddress("glUseProgram"));
        const auto uniform=reinterpret_cast<PFNGLUNIFORM4FVPROC>(glfwGetProcAddress("glUniform4fv"));
        const auto uniformLocation=reinterpret_cast<PFNGLGETUNIFORMLOCATIONPROC>(glfwGetProcAddress("glGetUniformLocation"));
        if(!useProgram||!uniform||!uniformLocation)return fail("OpenGL direct uniform association fixture lacks native functions");
        int previousProgram=0;glGetIntegerv(GL_CURRENT_PROGRAM,&previousProgram);
        const auto program=directQuery->program->linkedProgram;
        const auto sourceAt=uniformLocation(program,"uSource");
        if(sourceAt<0)return fail("OpenGL direct native source uniform disappeared");
        auto changedSource=directBinding->frame.sourceDirectionTrace;changedSource[0]+=.1f;
        useProgram(program);uniform(sourceAt,1,changedSource.data());
        const bool acceptedChangedUniform=gc::validateDirectGratingContributionReadback(
            *directBinding,*directOwner,direct.submission,.25,direct,error);
        uniform(sourceAt,1,directBinding->frame.sourceDirectionTrace.data());useProgram(unsigned(previousProgram));
        if(acceptedChangedUniform||!gc::validateDirectGratingContributionReadback(
            *directBinding,*directOwner,direct.submission,.25,direct,error))
            return fail("OpenGL direct proof ignored its actual native source uniform or failed to restore it");
        const auto refuseProof=[&](const auto& edit) {
            auto corrupt=direct;edit(corrupt);
            return !directOwner->validateDirectProductionQuery(direct.submission,corrupt,error)
                &&!gc::validateDirectGratingContributionReadback(*directBinding,*directOwner,direct.submission,.25,corrupt,error);
        };
        if(!refuseProof([](auto& r){r.submission.completedFence++;})
            ||!refuseProof([](auto& r){r.endpointGeneration++;})||!refuseProof([](auto& r){r.sourceGeneration++;})
            ||!refuseProof([](auto& r){r.samples[0].input.primitive[3]++;})
            ||!refuseProof([](auto& r){r.samples[0].source.input.coefficients[0]*=.5f;})
            ||!refuseProof([](auto& r){r.samples[0].sourceDirectionMaximum[0]+=.1f;})
            ||!refuseProof([](auto& r){r.samples[0].sourceHit.status=-1;})
            ||!refuseProof([](auto& r){r.samples[0].endpoint.work=0;})
            ||!refuseProof([](auto& r){r.samples[0].barycentric={0,.5f,.5f};})
            ||gc::validateDirectGratingContributionReadback(*directBinding,*directOwner,direct.submission,.5,direct,error))
            return fail("OpenGL direct proof accepted stale inputs, source, time, fence, seam or failed query");
        const auto positiveCopy=direct;
        if(!directOwner->validateDirectProductionQuery(direct.submission,positiveCopy,error)
            ||!gc::validateDirectGratingContributionReadback(*directBinding,*directOwner,direct.submission,.25,positiveCopy,error))
            return fail("OpenGL exact positive caller copy lost its actual retained result association");
        using Mutation=void(*)(gc::NativeDirectGratingContributionReadback&);
        const Mutation mutations[]{
            [](auto& r){r.version++;},[](auto& r){r.complete=false;},[](auto& r){r.overflow=true;},
            [](auto& r){r.queryWork++;},[](auto& r){r.bandWork++;},
            [](auto& r){r.submission.frameIdentity+="x";},[](auto& r){r.submission.rendererGeneration++;},
            [](auto& r){r.submission.submissionSerial++;},[](auto& r){r.submission.rasterGeneration++;},
            [](auto& r){r.submission.queryGeneration++;},[](auto& r){r.submission.scratchGeneration++;},
            [](auto& r){r.samples.clear();},[](auto& r){r.samples.push_back(r.samples[0]);},
            [](auto& r){r.samples[0].input.primitive[0]++;},[](auto& r){r.samples[0].input.control[0]++;},
            [](auto& r){r.samples[0].input.originMinimum[0]+=.01f;},
            [](auto& r){r.samples[0].input.directionMaximum[3]+=.01f;},
            [](auto& r){r.samples[0].source.input.identifiers[3]++;},
            [](auto& r){r.samples[0].source.rawRadiance[0]=1;},
            [](auto& r){r.samples[0].source.directionTrace[0]+=.01f;},
            [](auto& r){r.samples[0].endpoint.status=0;},[](auto& r){r.samples[0].endpoint.distance+=.01f;},
            [](auto& r){r.samples[0].endpoint.position[0]+=.01f;},
            [](auto& r){r.samples[0].endpoint.outwardNormal[0]+=.01f;},
            [](auto& r){r.samples[0].endpoint.objectId++;},[](auto& r){r.samples[0].endpoint.materialId++;},
            [](auto& r){r.samples[0].endpoint.instanceId++;},[](auto& r){r.samples[0].endpoint.triangleId++;},
            [](auto& r){r.samples[0].endpoint.entering=!r.samples[0].endpoint.entering;},
            [](auto& r){r.samples[0].sourceHit.distance*=.5f;},
            [](auto& r){r.samples[0].sourceHit.position[0]=-0.0f;},
            [](auto& r){r.samples[0].sourceHit.outwardNormal[0]=1;},
            [](auto& r){r.samples[0].sourceHit.objectId++;},[](auto& r){r.samples[0].sourceHit.materialId++;},
            [](auto& r){r.samples[0].sourceHit.instanceId++;},[](auto& r){r.samples[0].sourceHit.triangleId++;},
            [](auto& r){r.samples[0].sourceHit.entering=!r.samples[0].sourceHit.entering;},
            [](auto& r){r.samples[0].sourceHit.work++;},
            [](auto& r){r.samples[0].sourceOriginMinimum[3]*=2;},
            [](auto& r){r.samples[0].surfaceIrradiance[0]*=.5f;},[](auto& r){r.samples[0].queryWork++;},
            [](auto& r){r.samples[0].outcome=gc::NativeDirectGratingContributionSample::Outcome::Zero;},
            [](auto& r){r.samples[0].complete=false;}
        };
        for(const auto mutate:mutations)if(!refuseProof(mutate))
            return fail("OpenGL actual native result association accepted changed batch, echo, hit, work or sample fields");
        const auto before=direct;const auto previousUploadGeneration=directOwner->directEndpointUpload().generation;
        auto stale=std::make_shared<gc::GratingIncidentBinding>(*directBinding);
        if(arbitgpu::produceOpenGlDirectGratingContribution(directFrame,stale,inputs,.25,direct,error)
            ||arbitgpu::produceOpenGlDirectGratingContribution(directFrame,directBinding,inputs,.5,direct,error)
            ||arbitgpu::produceOpenGlDirectGratingContribution(directFrame,dimensionalBinding,inputs,.25,direct,error)
            ||!(direct.submission==before.submission)||directOwner->directEndpointUpload().generation!=previousUploadGeneration)
            return fail("OpenGL refused direct request replaced its previous exact native contribution");
        auto miss=inputs;for(unsigned axis=0;axis<3;++axis)miss[0].directionMaximum[axis]*=-1;
        if(arbitgpu::produceOpenGlDirectGratingContribution(directFrame,directBinding,miss,.25,direct,error)
            ||!(direct.submission==before.submission)||directOwner->directEndpointUpload().generation!=previousUploadGeneration)
            return fail("OpenGL actual endpoint miss became supported zero or replaced its incumbent");
        std::vector<gc::NativeDirectGratingEndpointInput> seam{endpoint(*directBinding,
            gc::advance(gc::advance({},triangle.positions[0],.5f),triangle.positions[2],.5f))};
        if(arbitgpu::produceOpenGlDirectGratingContribution(directFrame,directBinding,seam,.25,direct,error)
            ||!(direct.submission==before.submission))
            return fail("OpenGL actual primitive seam acquired a completed direct contribution");
        auto* context=glfwGetCurrentContext();glfwMakeContextCurrent(nullptr);
        const bool foreign=arbitgpu::produceOpenGlDirectGratingContribution(directFrame,directBinding,inputs,.25,direct,error);
        glfwMakeContextCurrent(context);
        if(foreign||!(direct.submission==before.submission)
            ||!gc::validateDirectGratingContributionReadback(*directBinding,*directOwner,direct.submission,.25,direct,error))
            return fail("OpenGL direct context refusal replaced or expired its original native owner");
        const auto previousBytes=directOwner->directEndpointUpload().otherOwnedBytes;
        const auto previousWork=directOwner->directEndpointUpload().otherWork;
        if(!arbitgpu::produceOpenGlDirectGratingContribution(directFrame,directBinding,inputs,.25,direct,error)
            ||directOwner->directEndpointUpload().otherOwnedBytes<=previousBytes
            ||directOwner->directEndpointUpload().otherWork<=previousWork
            ||gc::validateDirectGratingContributionReadback(*directBinding,*directOwner,before.submission,.25,before,error))
            return fail("OpenGL later direct batch omitted prior storage/work or revived a stale endpoint upload");
        // This interior point is under the real glass volume. The endpoint ray
        // starts near the grating; its separate source leg must hit the glass.
        inputs[0]=endpoint(*directBinding,{0,-.5f,-6});
        if(!arbitgpu::produceOpenGlDirectGratingContribution(directFrame,directBinding,inputs,.25,direct,error)
            ||direct.samples[0].sourceHit.status!=1||direct.samples[0].sourceHit.objectId!=fixture.program->object.value
            ||direct.samples[0].outcome!=gc::NativeDirectGratingContributionSample::Outcome::Zero
            ||direct.samples[0].surfaceIrradiance!=std::array<float,8>{})
            return fail("OpenGL real glass blocker did not complete as zero selected-light direct contribution");
        const auto blockedCopy=direct;
        if(!directOwner->validateDirectProductionQuery(direct.submission,blockedCopy,error)
            ||!gc::validateDirectGratingContributionReadback(*directBinding,*directOwner,direct.submission,.25,blockedCopy,error))
            return fail("OpenGL exact blocked-zero caller copy lost its actual retained result association");
        // Edit only the caller copy using the earlier actual clear-query shape
        // and bands. Keep the blocked batch's actual query work and all owners.
        auto forged=blockedCopy;auto& forgedSample=forged.samples[0];const auto shadowWork=forgedSample.sourceHit.work;
        forgedSample.sourceHit=positiveCopy.samples[0].sourceHit;forgedSample.sourceHit.work=shadowWork;
        forgedSample.surfaceIrradiance=positiveCopy.samples[0].surfaceIrradiance;
        forgedSample.outcome=gc::NativeDirectGratingContributionSample::Outcome::Positive;
        if(directOwner->validateDirectProductionQuery(direct.submission,forged,error)
            ||gc::validateDirectGratingContributionReadback(*directBinding,*directOwner,direct.submission,.25,forged,error)
            ||!directOwner->validateDirectProductionQuery(direct.submission,blockedCopy,error))
            return fail("OpenGL copied native blocked result acquired positive irradiance or changed the retained actual result");
        auto zeroScene=std::make_shared<gt::Visual3DScene>(*dimensionalScene);zeroScene->lights[0].transform.rotation={1,0,0,0};
        const auto zeroResources=arbitgpu::prepareOpenGlGlassGratingResources(zeroScene,
            dimensionalBinding->prepared.glass.sourceMaterial,selection,grating);
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> zeroFrame;std::shared_ptr<const gc::GratingIncidentBinding> zeroBinding;
        if(!zeroResources.prepared||!arbitgpu::stageOpenGlGlassGratingFrame(zeroResources.resources,fixture.ordinaryLight,*fixture.policy,
            32,32,runtime,.25,zeroFrame,zeroBinding,error,scheduling))return false;
        std::vector<gc::NativeDirectGratingEndpointInput> zeroInputs{endpoint(*zeroBinding,clear)};
        gc::NativeDirectGratingContributionReadback zero;
        if(!arbitgpu::produceOpenGlDirectGratingContribution(zeroFrame,zeroBinding,zeroInputs,.25,zero,error)
            ||zero.samples[0].sourceHit.status!=0||zero.samples[0].outcome!=gc::NativeDirectGratingContributionSample::Outcome::Zero
            ||zero.samples[0].surfaceIrradiance!=std::array<float,8>{})
            return fail("OpenGL negative grating source cosine did not complete as supported zero");
        gc::DirectGratingContributionBounds cap;
        if(gc::calculateDirectGratingContributionBounds(*directBinding,1,renderpassoutput::kMaximumTotalBytes,0,cap,error)
            ||gc::calculateDirectGratingContributionBounds(*directBinding,1,0,gc::kMaximumQueryWork,cap,error))
            return fail("OpenGL direct native fixture's original shared cumulative caps accepted excess work or storage");
        auto excessive=inputs;excessive.resize(gc::kMaximumEntrySamples+1,inputs[0]);const auto retained=direct.submission;
        if(arbitgpu::produceOpenGlDirectGratingContribution(directFrame,directBinding,excessive,.25,direct,error)
            ||!(direct.submission==retained))
            return fail("OpenGL oversized native endpoint batch bypassed original admission or changed output");
        const auto* stress=std::getenv("ARBIT_GL_DIRECT_BUDGET_STRESS");
        if(stress&&std::string(stress)=="1") {
            direct.samples.reserve(std::size_t(renderpassoutput::kMaximumTotalBytes/sizeof(gc::NativeDirectGratingContributionSample)));
            const auto retainedGeneration=directOwner->directEndpointUpload().generation;
            if(arbitgpu::produceOpenGlDirectGratingContribution(directFrame,directBinding,inputs,.25,direct,error)
                ||!(direct.submission==retained)||directOwner->directEndpointUpload().generation!=retainedGeneration)
                return fail("OpenGL actual near-capacity direct output bypassed pre-dispatch cumulative storage admission");
        }
    }
    {
        // Real two-source resources and one staged scene. Per-source bindings
        // reuse the exact common instances/camera/query, never another frame.
        auto scene=std::make_shared<gt::Visual3DScene>(*dimensionalScene);
        scene->lightCount=2;scene->lights[1]=scene->lights[0];scene->lights[1].id={scene->lights[0].id.value+1000};
        for(unsigned band=0;band<8;++band)
            scene->lights[1].spectralSource.directionalIrradiance.coefficients[band]=
                .25f*scene->lights[0].spectralSource.directionalIrradiance.coefficients[band];
        scene->lights[1].spectralSource.directionalIrradiance.coefficients[0]=1e-20f;
        scene->lights[1].spectralSource.directionalIrradiance.coefficients[7]=0;
        const auto resources=arbitgpu::prepareOpenGlGlassGratingResources(scene,
            dimensionalBinding->prepared.glass.sourceMaterial,selection,grating);
        if(!resources.prepared)return fail(resources.error.c_str());
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame;
        std::shared_ptr<const gc::GratingIncidentBinding> first;
        if(!arbitgpu::stageOpenGlGlassGratingFrame(resources.resources,scene->lights[0].id,*fixture.policy,32,32,
            runtime,.25,frame,first,error,scheduling))return false;
        const auto makeBinding=[&](HarmonicMIDI::grid::SceneLightId id,
            std::shared_ptr<const gc::GratingIncidentBinding>& result) {
            gc::Prepared prepared;gc::Frame scheduled;
            return gc::prepare(first->prepared.glass,id,*fixture.policy,prepared,error)
                &&gc::buildGratingStagingFrame(prepared,first->frame.instances,&runtime,32,32,scheduling,scheduled,error)
                &&gc::prepareGratingIncidentBinding(prepared,scheduled,first->gratingOwner,first->material,.25,result,error,
                    gc::SourceMeasureUse::DirectionalIrradiancePreparation);
        };
        std::shared_ptr<const gc::GratingIncidentBinding> second;if(!makeBinding(scene->lights[1].id,second))return false;
        std::vector<std::shared_ptr<const gc::GratingIncidentBinding>> bindings{first,second};
        const auto endpoint=[&](gc::Point point) {
            const auto& t=first->triangles.front();gc::NativeDirectGratingEndpointInput result;
            result.primitive={t.object,t.material,t.instance,t.triangle};const auto origin=gc::advance(point,t.normal,.1f);
            result.originMinimum={origin[0],origin[1],origin[2],gc::kRayMinimum};
            result.directionMaximum={-t.normal[0],-t.normal[1],-t.normal[2],1};return result;
        };
        const auto& triangle=first->triangles.front();gc::Point clear{};
        for(unsigned v=0;v<3;++v)clear=gc::advance(clear,triangle.positions[v],std::array<float,3>{.2f,.3f,.5f}[v]);
        // Equal rays are distinct ordered, budgeted occurrences.
        std::vector<gc::NativeDirectGratingEndpointInput> inputs{endpoint(clear),endpoint(clear),endpoint({0,-.5f,-6})};
        std::shared_ptr<const gc::DirectGratingIrradianceCohort> cohort;
        if(!gc::prepareDirectGratingIrradianceCohort(bindings,inputs,0,0,cohort,error))return false;
        const auto frozenFirst=cohort->sources()[0];const auto frozenSecond=cohort->sources()[1];
        if(frozenFirst==first||frozenSecond==second)return fail("OpenGL cohort fixture did not freeze source bindings");
        const auto* native=dynamic_cast<const gc::NativeDirectGratingIrradianceCohortBinding*>(frame.get());
        if(!native||native->directSourceCount())return fail("OpenGL cohort staging fabricated source dispatches");
        std::vector<gc::NativeDirectGratingContributionReadback> readbacks;gc::DirectGratingIrradianceCohortReadback sum;
        if(!arbitgpu::produceOpenGlDirectGratingIrradianceCohort(frame,cohort,.25,readbacks,sum,error))return false;
        if(sum.cohort!=cohort||native->directCohort()!=cohort||native->directSourceCount()!=2||readbacks.size()!=2||sum.endpoints.size()!=3
            ||native->directSourceBatch(0)==native->directSourceBatch(1)||native->directSourceBatch(2)
            ||native->directSourceBatch(0)->directGratingBinding()!=frozenFirst
            ||native->directSourceBatch(1)->directGratingBinding()!=frozenSecond
            ||frame->colorImageHandle()||frame->depthImageHandle()||frame->colorTextureDescriptor().complete())
            return fail("OpenGL native cohort lost its distinct frozen source inspectors or private common owner");
        for(std::size_t e=0;e<sum.endpoints.size();++e)for(unsigned band=0;band<8;++band) {
            const float expected=e==2?0:float(double(scene->lights[0].spectralSource.directionalIrradiance.coefficients[band]*triangle.normal[2])
                +double(scene->lights[1].spectralSource.directionalIrradiance.coefficients[band]*triangle.normal[2]));
            if(!gc::nearDirectionalSourcePotential(sum.endpoints[e].surfaceIrradiance[band],expected)
                ||sum.endpoints[e].surfaceIrradiance[band]!=float(double(readbacks[0].samples[e].surfaceIrradiance[band])
                    +double(readbacks[1].samples[e].surfaceIrradiance[band])))
                return fail("OpenGL actual two-source D*cos bands were omitted, duplicated or multiplied by a second cosine");
        }
        if(readbacks[1].samples[0].surfaceIrradiance[0]<=0||readbacks[1].samples[0].surfaceIrradiance[7]!=0
            ||readbacks[0].samples[2].outcome!=gc::NativeDirectGratingContributionSample::Outcome::Zero
            ||readbacks[1].samples[2].outcome!=gc::NativeDirectGratingContributionSample::Outcome::Zero
            ||sum.endpoints[0].surfaceIrradiance!=sum.endpoints[1].surfaceIrradiance)
            return fail("OpenGL cohort lost tiny-positive, zero, blocked or repeated endpoint occurrences");
        std::array<unsigned,4> storage[2]{};unsigned programs[2]{};std::uintptr_t fences[2]{};std::array<std::uint64_t,2> reservation{};
        for(unsigned source=0;source<2;++source)
            if(!arbitgpu::readOpenGlDirectGratingCohortStorageForFixture(*frame,source,storage[source],programs[source],fences[source],error,&reservation))return false;
        if(storage[0][0]==storage[1][0]||storage[0][1]==storage[1][1]||storage[0][2]==storage[1][2]
            ||storage[0][3]!=storage[1][3]||programs[0]==programs[1]||!fences[0]||!fences[1]||fences[0]==fences[1])
            return fail("OpenGL cohort retargeted one source/output/program owner or used independent query textures");
        const auto incumbent=sum;const auto originalReadbacks=readbacks;
        const auto accumulates=[&](const auto& receipts) {
            return gc::accumulateDirectGratingIrradianceCohort(cohort,*frame,sum.submission,.25,receipts,sum,error);
        };
        if(!accumulates(readbacks)||sum.endpoints[0].surfaceIrradiance!=incumbent.endpoints[0].surfaceIrradiance)
            return fail("OpenGL exact actual native cohort copy refused accumulation");
        gc::DirectGratingTerminalOrderLedger terminal;
        const auto terminalOrders=[&](const auto& receipts) {
            return arbitgpu::produceOpenGlDirectGratingTerminalOrders(frame,cohort,incumbent.submission,.25,receipts,terminal,error);
        };
        std::shared_ptr<const gc::NativeDirectGratingTerminalResult> retainedTerminal;
        std::array<std::uint64_t,5> terminalCounters{};
        if(!terminalOrders(readbacks)||!terminal.nativeResult||terminal.nativeResult->owner()!=frame.get()
            ||terminal.nativeResult->retained().nativeResult||terminal.bands.size()!=2*inputs.size()*8
            ||terminal.sources[0].samples.data()==terminal.nativeResult->retained().sources[0].samples.data()
            ||terminal.bands.data()==terminal.nativeResult->retained().bands.data()
            ||terminal.orders.data()==terminal.nativeResult->retained().orders.data()
            ||!arbitgpu::readOpenGlDirectGratingTerminalStateForFixture(*frame,retainedTerminal,terminalCounters,error)
            ||retainedTerminal!=terminal.nativeResult||terminalCounters[4]!=1)
            return fail("OpenGL terminal lease did not publish independent returned and sealed native owners");
        for(const auto& band:terminal.bands) {
            if(band.incidentIrradiance!=readbacks[band.source].samples[band.endpoint].surfaceIrradiance[band.band]
                ||std::abs(band.incidentIrradiance-band.substrateReflectedIrradiance-band.absorbedIrradiance)
                    >2e-10*band.incidentIrradiance
                ||band.orderCount>gc::kGratingOrdersPerIncident)
                return fail("OpenGL terminal native rows lost already-cosined irradiance or conservation");
            for(std::size_t o=band.firstOrder;o<band.firstOrder+band.orderCount;++o) {
                const auto& order=terminal.orders[o];
                if(order.source!=band.source||order.endpoint!=band.endpoint||order.band!=band.band
                    ||order.orderIrradiance!=band.incidentIrradiance*order.order.reflectedPower)
                    return fail("OpenGL terminal native order lost its actual source band or applied another cosine");
            }
        }
        const auto firstTerminal=terminal.nativeResult;const auto firstTerminalCounters=terminalCounters;
        if(!terminalOrders(readbacks)||terminal.nativeResult==firstTerminal
            ||!arbitgpu::readOpenGlDirectGratingTerminalStateForFixture(*frame,retainedTerminal,terminalCounters,error)
            ||retainedTerminal!=terminal.nativeResult||terminalCounters[4]!=2||terminalCounters[0]<=firstTerminalCounters[0]
            ||terminalCounters[2]-firstTerminalCounters[2]!=terminal.bounds.maximumBands+terminal.bounds.maximumOrders
            ||terminalCounters[3]-firstTerminalCounters[3]!=terminal.bounds.maximumOrders
            ||firstTerminal->retained().sources[0].samples[0].surfaceIrradiance!=readbacks[0].samples[0].surfaceIrradiance)
            return fail("OpenGL replacement terminal operation dropped an earlier externally retained sealed owner or work charge");
        const auto acceptedTerminal=terminal.nativeResult;const auto acceptedTerminalCounters=terminalCounters;
        auto corruptTerminalSources=readbacks;corruptTerminalSources[1].samples[0].surfaceIrradiance[0]*=.5f;
        if(terminalOrders(corruptTerminalSources)||terminal.nativeResult!=acceptedTerminal
            ||!arbitgpu::readOpenGlDirectGratingTerminalStateForFixture(*frame,retainedTerminal,terminalCounters,error)
            ||retainedTerminal!=acceptedTerminal||terminalCounters[0]!=acceptedTerminalCounters[0]||terminalCounters[4]!=2
            ||terminalCounters[2]-acceptedTerminalCounters[2]!=terminal.bounds.maximumBands+terminal.bounds.maximumOrders
            ||terminalCounters[3]-acceptedTerminalCounters[3]!=terminal.bounds.maximumOrders)
            return fail("OpenGL refused admitted terminal expansion replaced its incumbent, retained storage or made attempted work free");
        std::array<std::uint64_t,14> trueAccounting{};
        if(!arbitgpu::readOpenGlNativeAccountingForFixture(*frame,trueAccounting,error))return false;
        const auto refusesReducedAccounting=[&](auto poison) {
            const auto before=trueAccounting;poison[13]=before[13];
            if(!arbitgpu::exchangeOpenGlNativeAccountingForFixture(*frame,poison,error))return false;
            gc::NativeDirectGratingContributionReadback single;
            gc::NativeGlassCameraGratingTraceReadback camera;
            gc::NativeGlassCameraGratingTraceInput cameraInput;
            const bool madeCamera=gc::makeGlassCameraGratingTraceInput(first->frame,16,16,0,cameraInput,error);
            const bool sumAccepted=accumulates(readbacks);
            const bool directAccepted=arbitgpu::produceOpenGlDirectGratingContribution(frame,first,inputs,.25,single,error);
            const bool cohortAccepted=arbitgpu::produceOpenGlDirectGratingIrradianceCohort(frame,cohort,.25,readbacks,sum,error);
            const bool cameraAccepted=madeCamera&&arbitgpu::produceOpenGlGlassCameraGratingTrace(frame,first,{cameraInput},.25,camera,error);
            const bool terminalAccepted=terminalOrders(readbacks);
            const bool restored=arbitgpu::exchangeOpenGlNativeAccountingForFixture(*frame,poison,error);
            std::array<std::uint64_t,14> after{};
            return restored&&madeCamera&&!sumAccepted&&!directAccepted&&!cohortAccepted&&!cameraAccepted&&!terminalAccepted
                &&terminal.nativeResult==acceptedTerminal&&sum.submission==incumbent.submission
                &&arbitgpu::readOpenGlNativeAccountingForFixture(*frame,after,error)&&after==before&&accumulates(readbacks);
        };
        auto missingFailedTerminal=trueAccounting;
        const auto failedTerminalWork=terminal.bounds.maximumBands+terminal.bounds.maximumOrders;
        missingFailedTerminal[1]-=failedTerminalWork;missingFailedTerminal[2]-=failedTerminalWork;
        missingFailedTerminal[3]-=terminal.bounds.maximumOrders;
        if(!refusesReducedAccounting(missingFailedTerminal))
            return fail("OpenGL later actual native producers accepted removal of an independently retained failed terminal attempt");
        auto missingSealedStorage=trueAccounting;--missingSealedStorage[0];
        if(!refusesReducedAccounting(missingSealedStorage))
            return fail("OpenGL later actual native producers omitted a retained sealed owner after scalar storage reduction");
        const auto beforeTerminalTime=terminalCounters;
        if(arbitgpu::produceOpenGlDirectGratingTerminalOrders(frame,cohort,incumbent.submission,.5,readbacks,terminal,error)
            ||terminal.nativeResult!=acceptedTerminal
            ||!arbitgpu::readOpenGlDirectGratingTerminalStateForFixture(*frame,retainedTerminal,terminalCounters,error)
            ||terminalCounters!=beforeTerminalTime)
            return fail("OpenGL terminal time refusal changed the frame owner or charged unadmitted work");
        const auto refuses=[&](const auto& mutate) {
            auto corrupt=readbacks;mutate(corrupt);const bool accepted=accumulates(corrupt);
            return !accepted&&sum.cohort==incumbent.cohort&&sum.submission==incumbent.submission
                &&sum.endpoints[0].surfaceIrradiance==incumbent.endpoints[0].surfaceIrradiance;
        };
        if(!refuses([](auto& r){r.pop_back();})||!refuses([](auto& r){r.push_back(r[0]);})||!refuses([](auto& r){r[1]=r[0];})
            ||!refuses([](auto& r){std::swap(r[0],r[1]);})||!refuses([](auto& r){r[1].sourceGeneration++;})
            ||!refuses([](auto& r){r[1].endpointGeneration++;})
            ||!refuses([](auto& r){std::swap(r[1].samples[0],r[1].samples[2]);})
            ||!refuses([](auto& r){r[0].samples[0].input.originMinimum[0]+=.01f;})
            ||!refuses([](auto& r){r[1].samples[2].surfaceIrradiance[0]=1;})
            ||!refuses([](auto& r){r[1].complete=false;})||!refuses([](auto& r){r[1].submission.completedFence++;}))
            return fail("OpenGL actual cohort accepted missing, duplicate, stale, changed or incomplete native contributors");
        // Mutate actual retained native source and endpoint input buffers,
        // then restore them. Supplied receipts remain exact native copies.
        const auto bind=reinterpret_cast<PFNGLBINDBUFFERPROC>(glfwGetProcAddress("glBindBuffer"));
        const auto replace=reinterpret_cast<PFNGLBUFFERSUBDATAPROC>(glfwGetProcAddress("glBufferSubData"));
        const auto read=reinterpret_cast<PFNGLGETBUFFERSUBDATAPROC>(glfwGetProcAddress("glGetBufferSubData"));
        if(!bind||!replace||!read)return fail("OpenGL cohort tamper fixture lacks native buffer functions");
        int oldBuffer=0;glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING,&oldBuffer);
        auto sourceInput=native->directSourceBatch(1)->directSourceBinding().upload().input;
        auto changedSource=sourceInput;changedSource.coefficients[0]*=2;
        bind(GL_SHADER_STORAGE_BUFFER,storage[1][2]);replace(GL_SHADER_STORAGE_BUFFER,0,sizeof(changedSource),&changedSource);
        const bool sourceAccepted=accumulates(readbacks);
        const bool terminalSourceAccepted=terminalOrders(readbacks);
        replace(GL_SHADER_STORAGE_BUFFER,0,sizeof(sourceInput),&sourceInput);
        auto changedEndpoint=inputs[1];changedEndpoint.originMinimum[0]+=.01f;
        bind(GL_SHADER_STORAGE_BUFFER,storage[0][0]);replace(GL_SHADER_STORAGE_BUFFER,sizeof(changedEndpoint),sizeof(changedEndpoint),&changedEndpoint);
        const bool endpointAccepted=accumulates(readbacks);
        replace(GL_SHADER_STORAGE_BUFFER,sizeof(changedEndpoint),sizeof(changedEndpoint),&inputs[1]);
        gc::opengl::DirectGratingPacked packed{};bind(GL_SHADER_STORAGE_BUFFER,storage[1][1]);
        read(GL_SHADER_STORAGE_BUFFER,0,sizeof(packed),&packed);auto changedPacked=packed;changedPacked[0][0]+=1;
        replace(GL_SHADER_STORAGE_BUFFER,0,sizeof(changedPacked),&changedPacked);
        const bool resultAccepted=accumulates(readbacks);
        replace(GL_SHADER_STORAGE_BUFFER,0,sizeof(packed),&packed);bind(GL_SHADER_STORAGE_BUFFER,unsigned(oldBuffer));
        if(sourceAccepted||terminalSourceAccepted||terminal.nativeResult!=acceptedTerminal||endpointAccepted||resultAccepted||glGetError()!=GL_NO_ERROR||!accumulates(readbacks))
            return fail("OpenGL common cohort ignored actual source/endpoint/result buffer mutation or failed restoration");
        const auto use=reinterpret_cast<PFNGLUSEPROGRAMPROC>(glfwGetProcAddress("glUseProgram"));
        const auto uniform=reinterpret_cast<PFNGLUNIFORM4FVPROC>(glfwGetProcAddress("glUniform4fv"));
        const auto location=reinterpret_cast<PFNGLGETUNIFORMLOCATIONPROC>(glfwGetProcAddress("glGetUniformLocation"));
        if(!use||!uniform||!location)return fail("OpenGL cohort uniform fixture lacks native functions");
        int oldProgram=0;glGetIntegerv(GL_CURRENT_PROGRAM,&oldProgram);const auto sourceAt=location(programs[1],"uSource");
        if(sourceAt<0)return fail("OpenGL cohort source uniform is absent");
        auto changedDirection=frozenSecond->frame.sourceDirectionTrace;changedDirection[0]+=.1f;
        use(programs[1]);uniform(sourceAt,1,changedDirection.data());const bool directionAccepted=accumulates(readbacks);
        uniform(sourceAt,1,frozenSecond->frame.sourceDirectionTrace.data());use(unsigned(oldProgram));
        if(directionAccepted||glGetError()!=GL_NO_ERROR||!accumulates(readbacks))
            return fail("OpenGL cohort ignored a source program uniform mutation or failed restoration");
        // Existing selected-source producer keeps its independent head and API.
        gc::NativeDirectGratingContributionReadback single;
        if(!arbitgpu::produceOpenGlDirectGratingContribution(frame,first,inputs,.25,single,error)||!accumulates(readbacks))
            return fail("OpenGL cohort changed the old selected-source API or invalidated retained source programs");
        auto* context=glfwGetCurrentContext();glfwMakeContextCurrent(nullptr);
        const bool foreign=arbitgpu::produceOpenGlDirectGratingIrradianceCohort(frame,cohort,.25,readbacks,sum,error);
        const bool foreignTerminal=terminalOrders(readbacks);
        glfwMakeContextCurrent(context);
        if(foreign||foreignTerminal||terminal.nativeResult!=acceptedTerminal||!accumulates(readbacks)
            ||arbitgpu::produceOpenGlDirectGratingIrradianceCohort(frame,cohort,.5,readbacks,sum,error))
            return fail("OpenGL cohort accepted a foreign context/time or lost incumbent output");
        std::vector<std::shared_ptr<const gc::GratingIncidentBinding>> missing{first},duplicate{first,first},
            reordered{second,first},extra{first,second,first};
        std::shared_ptr<const gc::DirectGratingIrradianceCohort> refused=cohort;
        if(gc::prepareDirectGratingIrradianceCohort(missing,inputs,0,0,refused,error)
            ||gc::prepareDirectGratingIrradianceCohort(duplicate,inputs,0,0,refused,error)
            ||gc::prepareDirectGratingIrradianceCohort(reordered,inputs,0,0,refused,error)
            ||gc::prepareDirectGratingIrradianceCohort(extra,inputs,0,0,refused,error)||refused!=cohort
            ||gc::prepareDirectGratingIrradianceCohort(bindings,inputs,renderpassoutput::kMaximumTotalBytes,0,refused,error)
            ||gc::prepareDirectGratingIrradianceCohort(bindings,inputs,0,gc::kMaximumQueryWork,refused,error))
            return fail("OpenGL cohort bypassed complete roster or original cumulative admission");
        std::vector<gc::NativeDirectGratingEndpointInput> seam{endpoint(gc::advance(gc::advance({},triangle.positions[0],.5f),triangle.positions[2],.5f))};
        std::shared_ptr<const gc::DirectGratingIrradianceCohort> failed;
        if(!gc::prepareDirectGratingIrradianceCohort(bindings,seam,0,0,failed,error))return false;
        if(!arbitgpu::readOpenGlDirectGratingCohortStorageForFixture(*frame,0,storage[0],programs[0],fences[0],error,&reservation))return false;
        const auto beforeFailure=reservation;
        if(arbitgpu::produceOpenGlDirectGratingIrradianceCohort(frame,failed,.25,readbacks,sum,error)
            ||native->directCohort()!=cohort||!(sum.submission==incumbent.submission)
            ||!gc::sameDirectGratingContributionReadback(readbacks[0],originalReadbacks[0])||!accumulates(readbacks)
            ||!arbitgpu::readOpenGlDirectGratingCohortStorageForFixture(*frame,0,storage[0],programs[0],fences[0],error,&reservation)
            ||reservation[0]!=beforeFailure[0]||reservation[1]<=beforeFailure[1])
            return fail("OpenGL refused native cohort prefix changed its incumbent or made dispatched work free");
        gc::NativeReadback refusedPoint;refusedPoint.frameIdentity="unchanged failed point caller";
        gc::EnergyLedger refusedEnergy;gc::NativeGratingCellTransportReadback refusedCells;
        std::array<std::uint64_t,14> beforePoint{},afterPoint{};
        if(!arbitgpu::readOpenGlNativeAccountingForFixture(*frame,beforePoint,error))return false;
        for(unsigned retry=0;retry<2;++retry) {
            if(arbitgpu::produceOpenGlGlassGratingAttemptForFixture(frame,first,{},refusedPoint,refusedEnergy,refusedCells,
                arbitgpu::OpenGlGratingAttemptStopForFixture::PointFence,error)
                ||error.find("after real point dispatch")==std::string::npos||refusedPoint.frameIdentity!="unchanged failed point caller"
                ||!arbitgpu::readOpenGlNativeAccountingForFixture(*frame,afterPoint,error)
                ||afterPoint[5]-beforePoint[5]!=(retry+1)*first->bounds.combinedWork
                ||afterPoint[6]-beforePoint[6]!=(retry+1)*(first->frame.bounds.deposition.depositionWork+first->bounds.orderWork)
                ||afterPoint[13]-beforePoint[13]!=retry+1||afterPoint[0]!=beforePoint[0])
                return fail("OpenGL equal native point refusal retries lost full admitted work or their caller/retained terminal storage");
        }
        trueAccounting=afterPoint;auto missingNativePrefix=afterPoint;
        missingNativePrefix[5]=beforePoint[5];missingNativePrefix[6]=beforePoint[6];
        if(!refusesReducedAccounting(missingNativePrefix))
            return fail("OpenGL later actual native producers accepted reduced failed point-prefix work totals");
        gc::NativeDirectGratingContributionReadback laterSingle;gc::DirectGratingTerminalOrderLedger laterTerminal;
        if(!arbitgpu::produceOpenGlDirectGratingContribution(frame,first,inputs,.25,laterSingle,error)
            ||!arbitgpu::produceOpenGlDirectGratingTerminalOrders(frame,cohort,incumbent.submission,.25,readbacks,laterTerminal,error)
            ||!laterTerminal.nativeResult||firstTerminal->retained().sources[0].samples[0].surfaceIrradiance!=readbacks[0].samples[0].surfaceIrradiance)
            return fail("OpenGL retained failed terminal/native attempts blocked a valid later native admission or dropped an older sealed owner");
        // Refuse inside the actual producer's admitted preparation window,
        // before any copied Frame/source binding or source dispatch exists.
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> preparationFrame;
        std::shared_ptr<const gc::GratingIncidentBinding> preparationFirst;
        if(!arbitgpu::stageOpenGlGlassGratingFrame(resources.resources,scene->lights[0].id,*fixture.policy,32,32,
            runtime,.25,preparationFrame,preparationFirst,error,scheduling))return false;
        const auto* preparationNative=dynamic_cast<const gc::NativeDirectGratingIrradianceCohortBinding*>(preparationFrame.get());
        if(!preparationNative||preparationNative->directSourceCount())return false;
        auto preparationReadbacks=readbacks;auto preparationSum=incumbent;
        std::array<std::uint64_t,14> beforePreparation{},afterPreparation{};
        gc::DirectGratingTerminalNativeBudget preparationBudget,afterPreparationBudget;
        if(!arbitgpu::readOpenGlNativeAccountingForFixture(*preparationFrame,beforePreparation,error,&preparationBudget))return false;
        if(arbitgpu::refuseOpenGlDirectGratingCohortPreparationForFixture(preparationFrame,cohort,.5,preparationReadbacks,preparationSum,error)
            ||!arbitgpu::readOpenGlNativeAccountingForFixture(*preparationFrame,afterPreparation,error,&afterPreparationBudget)
            ||afterPreparation!=beforePreparation||afterPreparationBudget.retainedBytes!=preparationBudget.retainedBytes)
            return fail("OpenGL invalid cohort preparation was admitted or charged");
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> differentCommonFrame;
        std::shared_ptr<const gc::GratingIncidentBinding> differentCommonFirst;
        std::array<std::uint64_t,14> differentBefore{},differentAfter{};
        if(!arbitgpu::stageOpenGlGlassGratingFrame(resources.resources,scene->lights[0].id,*fixture.policy,16,16,
            runtime,.25,differentCommonFrame,differentCommonFirst,error,scheduling)
            ||!arbitgpu::readOpenGlNativeAccountingForFixture(*differentCommonFrame,differentBefore,error))return false;
        if(arbitgpu::refuseOpenGlDirectGratingCohortPreparationForFixture(differentCommonFrame,cohort,.25,
            preparationReadbacks,preparationSum,error)
            ||error.find("actual frozen scene, camera, query or material")==std::string::npos
            ||!arbitgpu::readOpenGlNativeAccountingForFixture(*differentCommonFrame,differentAfter,error)
            ||differentAfter!=differentBefore)
            return fail("OpenGL foreign common-frame cohort was charged as an admitted preparation");
        std::uint64_t preparationWork=0,preparationDeposition=0;
        for(const auto& source:cohort->sources()) {
            gc::DirectGratingContributionBounds bounds;
            if(!gc::calculateDirectGratingContributionBounds(*source,inputs.size(),0,0,bounds,error))return false;
            preparationWork+=bounds.combinedWork;
            preparationDeposition+=source->frame.bounds.deposition.depositionWork+source->bounds.orderWork;
        }
        preparationWork-=preparationFirst->bounds.combinedWork;
        preparationDeposition-=preparationFirst->frame.bounds.deposition.depositionWork+preparationFirst->bounds.orderWork;
        preparationDeposition+=preparationWork;
        const auto preparationRefuses=[&](unsigned attempts) {
            const auto caller=preparationSum;const auto callerSources=preparationReadbacks;
            const bool accepted=arbitgpu::refuseOpenGlDirectGratingCohortPreparationForFixture(
                preparationFrame,cohort,.25,preparationReadbacks,preparationSum,error);
            const bool allocationRefused=error.find("admitted allocation failed")!=std::string::npos;
            return !accepted&&allocationRefused
                &&arbitgpu::readOpenGlNativeAccountingForFixture(*preparationFrame,afterPreparation,error,&afterPreparationBudget)
                &&afterPreparation[1]-beforePreparation[1]==attempts*preparationWork
                &&afterPreparation[4]-beforePreparation[4]==attempts*cohort->bounds().bytes
                &&afterPreparation[13]-beforePreparation[13]==attempts
                &&afterPreparation[0]==beforePreparation[0]&&afterPreparation[12]==beforePreparation[12]
                &&afterPreparationBudget.retainedBytes-preparationBudget.retainedBytes==attempts*cohort->bounds().bytes
                &&afterPreparationBudget.combinedWork-preparationBudget.combinedWork==attempts*preparationWork
                &&afterPreparationBudget.depositionWork-preparationBudget.depositionWork==attempts*preparationDeposition
                &&preparationSum.cohort==caller.cohort&&preparationSum.submission==caller.submission
                &&preparationSum.endpoints.size()==caller.endpoints.size()
                &&preparationSum.endpoints[0].surfaceIrradiance==caller.endpoints[0].surfaceIrradiance
                &&preparationReadbacks.size()==callerSources.size()
                &&gc::sameDirectGratingContributionReadback(preparationReadbacks[0],callerSources[0])
                &&gc::sameDirectGratingContributionReadback(preparationReadbacks[1],callerSources[1]);
        };
        if(!preparationRefuses(1)||!preparationRefuses(2)||preparationNative->directSourceCount()||preparationNative->directCohort())
            return fail("OpenGL refused admitted cohort preparation lost full frozen work/metadata charges or granted partial native completion");
        // A reduced failed-preparation total must refuse a later real producer.
        const auto registeredPreparation=afterPreparation;
        auto preparationPoison=afterPreparation;preparationPoison[1]-=preparationWork;
        if(!arbitgpu::exchangeOpenGlNativeAccountingForFixture(*preparationFrame,preparationPoison,error))return false;
        const bool reducedPreparationAccepted=arbitgpu::produceOpenGlDirectGratingIrradianceCohort(
            preparationFrame,cohort,.25,preparationReadbacks,preparationSum,error);
        if(!arbitgpu::exchangeOpenGlNativeAccountingForFixture(*preparationFrame,preparationPoison,error)
            ||reducedPreparationAccepted||!arbitgpu::readOpenGlNativeAccountingForFixture(*preparationFrame,afterPreparation,error)
            ||afterPreparation!=registeredPreparation
            ||!arbitgpu::produceOpenGlDirectGratingIrradianceCohort(preparationFrame,cohort,.25,preparationReadbacks,preparationSum,error))
            return fail("OpenGL failed preparation witness was bypassed or prevented a restored valid retry");
        if(preparationNative->directCohort()!=cohort||preparationNative->directSourceCount()!=2
            ||preparationReadbacks.size()!=2||preparationSum.endpoints.size()!=incumbent.endpoints.size())
            return fail("OpenGL restored preparation retry omitted a real source or endpoint");
        for(std::size_t i=0;i<preparationSum.endpoints.size();++i)
            if(preparationSum.endpoints[i].surfaceIrradiance!=incumbent.endpoints[i].surfaceIrradiance)
                return fail("OpenGL restored preparation retry changed already-cosined source sums");
        const auto preparedSum=preparationSum;const auto preparedReadbacks=preparationReadbacks;
        gc::DirectGratingTerminalOrderLedger preparationTerminal;
        if(!arbitgpu::produceOpenGlDirectGratingTerminalOrders(preparationFrame,cohort,preparedSum.submission,.25,
            preparationReadbacks,preparationTerminal,error)||!preparationTerminal.nativeResult)return false;
        const auto preparedTerminalOwner=preparationTerminal.nativeResult;
        std::array<unsigned,4> preparedBuffers{};unsigned preparedProgram=0;std::uintptr_t preparedFence=0;
        if(!arbitgpu::readOpenGlDirectGratingCohortStorageForFixture(*preparationFrame,0,preparedBuffers,preparedProgram,preparedFence,error)
            ||!arbitgpu::readOpenGlNativeAccountingForFixture(*preparationFrame,beforePreparation,error,&preparationBudget))return false;
        if(!preparationRefuses(1))return fail("OpenGL failed replacement preparation did not retain its complete attempt charge");
        // Replacement preserves the completed caller and actual buffers/fence.
        std::array<unsigned,4> stillPreparedBuffers{};unsigned stillPreparedProgram=0;std::uintptr_t stillPreparedFence=0;
        std::shared_ptr<const gc::NativeDirectGratingTerminalResult> stillPreparedTerminal;
        std::array<std::uint64_t,5> stillPreparedCounters{};
        if(preparationNative->directCohort()!=cohort||!(preparationSum.submission==preparedSum.submission)
            ||!gc::sameDirectGratingContributionReadback(preparationReadbacks[0],preparedReadbacks[0])
            ||!arbitgpu::readOpenGlDirectGratingCohortStorageForFixture(*preparationFrame,0,stillPreparedBuffers,stillPreparedProgram,stillPreparedFence,error)
            ||stillPreparedBuffers!=preparedBuffers||stillPreparedProgram!=preparedProgram||stillPreparedFence!=preparedFence
            ||!arbitgpu::readOpenGlDirectGratingTerminalStateForFixture(*preparationFrame,stillPreparedTerminal,stillPreparedCounters,error)
            ||stillPreparedTerminal!=preparedTerminalOwner
            ||!arbitgpu::produceOpenGlDirectGratingIrradianceCohort(preparationFrame,cohort,.25,preparationReadbacks,preparationSum,error)
            ||!(preparedTerminalOwner->retained().incident.submission==preparedSum.submission))
            return fail("OpenGL refused replacement preparation changed an incumbent native/terminal owner or prevented a valid retry");
        auto zeroScene=std::make_shared<gt::Visual3DScene>(*scene);
        zeroScene->lights[1].spectralSource.directionalIrradiance.coefficients.fill(0);
        const auto zeroResources=arbitgpu::prepareOpenGlGlassGratingResources(zeroScene,
            dimensionalBinding->prepared.glass.sourceMaterial,selection,grating);
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> zeroFrame;
        std::shared_ptr<const gc::GratingIncidentBinding> zeroFirst,zeroSecond;
        if(!zeroResources.prepared||!arbitgpu::stageOpenGlGlassGratingFrame(zeroResources.resources,zeroScene->lights[0].id,
            *fixture.policy,32,32,runtime,.25,zeroFrame,zeroFirst,error,scheduling))return false;
        gc::Prepared zeroPrepared;gc::Frame zeroScheduled;
        if(!gc::prepare(zeroFirst->prepared.glass,zeroScene->lights[1].id,*fixture.policy,zeroPrepared,error)
            ||!gc::buildGratingStagingFrame(zeroPrepared,zeroFirst->frame.instances,&runtime,32,32,scheduling,zeroScheduled,error)
            ||!gc::prepareGratingIncidentBinding(zeroPrepared,zeroScheduled,zeroFirst->gratingOwner,zeroFirst->material,.25,zeroSecond,error,
                gc::SourceMeasureUse::DirectionalIrradiancePreparation))return false;
        std::shared_ptr<const gc::DirectGratingIrradianceCohort> zeroCohort;
        if(!gc::prepareDirectGratingIrradianceCohort({zeroFirst,zeroSecond},inputs,0,0,zeroCohort,error))return false;
        std::vector<gc::NativeDirectGratingContributionReadback> zeroReadbacks;gc::DirectGratingIrradianceCohortReadback zeroSum;
        if(!arbitgpu::produceOpenGlDirectGratingIrradianceCohort(zeroFrame,zeroCohort,.25,zeroReadbacks,zeroSum,error)
            ||zeroReadbacks[1].samples[0].outcome!=gc::NativeDirectGratingContributionSample::Outcome::Zero
            ||zeroReadbacks[1].samples[0].surfaceIrradiance!=std::array<float,8>{}
            ||zeroSum.endpoints[0].surfaceIrradiance!=zeroReadbacks[0].samples[0].surfaceIrradiance)
            return fail("OpenGL complete zero directional source was omitted, refused or added nonzero irradiance");
        gc::DirectGratingTerminalOrderLedger zeroTerminal;
        if(!arbitgpu::produceOpenGlDirectGratingTerminalOrders(zeroFrame,zeroCohort,zeroSum.submission,.25,zeroReadbacks,zeroTerminal,error)
            ||!zeroTerminal.nativeResult||zeroTerminal.bands.size()!=2*inputs.size()*8
            ||std::any_of(zeroTerminal.bands.begin(),zeroTerminal.bands.end(),[](const auto& band){return band.source==1&&band.incidentIrradiance!=0;}))
            return fail("OpenGL terminal native lease omitted or fabricated the complete all-zero source");
        gc::DirectGratingIrradianceCohortReadback oldOnly;
        if(gc::accumulateDirectGratingIrradianceCohort(cohort,*dimensionalFrame,incumbent.submission,.25,originalReadbacks,oldOnly,error))
            return fail("OpenGL single-source native frame acquired a cohort without native source staging");
        // Retire one actual completed fence after this separate zero frame's
        // positive checks. Exact copied receipts cannot replace its live lease.
        const auto deleteFence=reinterpret_cast<PFNGLDELETESYNCPROC>(glfwGetProcAddress("glDeleteSync"));
        std::array<unsigned,4> zeroStorage{};unsigned zeroProgram=0;std::uintptr_t zeroFence=0;
        if(!deleteFence||!arbitgpu::readOpenGlDirectGratingCohortStorageForFixture(*zeroFrame,1,zeroStorage,zeroProgram,zeroFence,error))return false;
        const auto zeroBefore=zeroSum;deleteFence(reinterpret_cast<GLsync>(zeroFence));
        if(gc::accumulateDirectGratingIrradianceCohort(zeroCohort,*zeroFrame,zeroBefore.submission,.25,zeroReadbacks,zeroSum,error)
            ||!(zeroSum.submission==zeroBefore.submission)||zeroSum.endpoints[0].surfaceIrradiance!=zeroBefore.endpoints[0].surfaceIrradiance)
            return fail("OpenGL cohort accepted a deleted actual native fence or changed its incumbent sum");
        const auto zeroTerminalOwner=zeroTerminal.nativeResult;
        if(arbitgpu::produceOpenGlDirectGratingTerminalOrders(zeroFrame,zeroCohort,zeroBefore.submission,.25,zeroReadbacks,zeroTerminal,error)
            ||zeroTerminal.nativeResult!=zeroTerminalOwner)
            return fail("OpenGL terminal lease registered a deleted real native fence as completed production");
        // Optional real retry exhaustion stays bounded and uses the native
        // producer's original cumulative cap, with no injected completion.
        const auto* stress=std::getenv("ARBIT_GL_COHORT_BUDGET_STRESS");
        if(stress&&std::string(stress)=="1") {
            bool exhausted=false;
            for(unsigned attempt=0;attempt<128;++attempt) {
                const auto before=sum.submission;
                if(!arbitgpu::readOpenGlDirectGratingCohortStorageForFixture(*frame,0,storage[0],programs[0],fences[0],error,&reservation))return false;
                const auto beforeAdmission=reservation;
                if(!arbitgpu::produceOpenGlDirectGratingIrradianceCohort(frame,cohort,.25,readbacks,sum,error)) {
                    const auto reason=error;
                    if(!(sum.submission==before)
                        ||(reason.find("cap")==std::string::npos&&reason.find("admission")==std::string::npos
                            &&reason.find("bounds")==std::string::npos&&reason.find("storage")==std::string::npos)
                        ||!arbitgpu::readOpenGlDirectGratingCohortStorageForFixture(*frame,0,storage[0],programs[0],fences[0],error,&reservation)
                        ||reservation!=beforeAdmission)
                        return fail("OpenGL cohort budget refusal dispatched work, changed incumbent or concealed another native failure");
                    exhausted=true;break;
                }
            }
            if(!exhausted)return fail("OpenGL cohort retries escaped their cumulative storage/work cap");
        }
    }
    {
        using Row=gc::DirectGratingTerminalReceiverRow;
        // Each case uses fresh real private scene/raster/query owners. It never
        // fills a native result with an oracle hit or a caller visibility flag.
        for(unsigned variant=0;variant<9;++variant) {
            auto scene=std::make_shared<gt::Visual3DScene>(*dimensionalScene);
            auto root=std::make_shared<gt::Program>(*dimensionalBinding->prepared.glass.sourceMaterial);
            scene->lights[0].spectralSource.directionalIrradiance.coefficients[7]=0;
            const auto& wall=scene->objects[receiverIndex];
            if(variant==1)for(std::size_t v=wall.firstVertex;v<wall.firstVertex+wall.vertexCount;++v) {
                auto& vertex=scene->vertices[v];vertex.position.x+=.13f*(vertex.position.z+6);
                vertex.normal={float(1/std::sqrt(1.0169)),0,float(-.13/std::sqrt(1.0169))};
            }
            if(variant==2)scene->cameras[0].transform.rotation={0,0,0,1};
            if(variant==3)for(auto& child:root->objectPrograms)if(child->object==wall.id) {
                auto material=std::make_shared<gt::Program>(*child);material->spectralReflectance.reset();child=material;
            }
            if(variant==4)for(std::size_t v=wall.firstVertex;v<wall.firstVertex+wall.vertexCount;++v)scene->vertices[v].position.x-=36;
            if(variant==5) {
                // A source just inside the emitting hemisphere prepares a real
                // grazing specular order. Its normal launch must not self-hit.
                const double theta=std::atan2(1.0,.3)-3e-6;
                scene->lights[0].transform.rotation={0,float(std::sin(theta*.5)),0,float(std::cos(theta*.5))};
            }
            if(variant==6)scene->lights[0].transform.rotation={1,0,0,0};
            if(variant==7)for(std::size_t v=wall.firstVertex;v<wall.firstVertex+wall.vertexCount;++v) {
                auto& vertex=scene->vertices[v];vertex.position.y=-.24f+(vertex.position.y<0?-1e-4f:1e-4f);
            }
            if(variant==8) {
                const auto index=scene->objectCount;
                videohelper::tests::smoothglassfixture::receiver(*scene,-6,{0,0,0});
                const auto& blocker=scene->objects[index];
                const HarmonicMIDI::grid::SceneVec3 points[]={{-2,-2,0},{-2,2,0},{-2,2,2},{-2,-2,2}};
                for(unsigned v=0;v<4;++v) {
                    auto& vertex=scene->vertices[blocker.firstVertex+v];vertex.position=points[v];vertex.normal={1,0,0};vertex.tangent={0,1,0,1};
                }
                auto material=std::make_shared<gt::Program>(*receiverProgram);material->object=blocker.id;material->authoredMaterial=blocker.material;
                material->parameters.identifiers[0]=blocker.material.value;material->bindingDigest="gl-direct-receiver-camera-blocker";
                material->programIdentity="gl-direct-receiver-camera-blocker";root->objectPrograms.push_back(material);
                std::sort(root->objectPrograms.begin(),root->objectPrograms.end(),[](const auto& a,const auto& b){return a->object.value<b->object.value;});
            }
            const auto resources=arbitgpu::prepareOpenGlGlassGratingResources(scene,root,selection,grating);
            if(!resources.prepared)return fail(resources.error.c_str());
            std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame;std::shared_ptr<const gc::GratingIncidentBinding> first;
            if(!arbitgpu::stageOpenGlGlassGratingFrame(resources.resources,scene->lights[0].id,*fixture.policy,32,32,
                runtime,.25,frame,first,error,scheduling))return false;
            const auto& t=first->triangles.front();gc::Point point{};
            for(unsigned v=0;v<3;++v)point=gc::advance(point,t.positions[v],std::array<float,3>{.48f,.04f,.48f}[v]);
            gc::NativeDirectGratingEndpointInput input;input.primitive={t.object,t.material,t.instance,t.triangle};
            const auto origin=gc::advance(point,t.normal,.1f);input.originMinimum={origin[0],origin[1],origin[2],gc::kRayMinimum};
            input.directionMaximum={-t.normal[0],-t.normal[1],-t.normal[2],1};
            std::vector<gc::NativeDirectGratingEndpointInput> inputs{input,input};
            std::shared_ptr<const gc::DirectGratingIrradianceCohort> cohort;
            if(!gc::prepareDirectGratingIrradianceCohort({first},inputs,0,0,cohort,error))return false;
            std::vector<gc::NativeDirectGratingContributionReadback> readbacks;gc::DirectGratingIrradianceCohortReadback sum;
            gc::DirectGratingTerminalOrderLedger terminal;
            if(!arbitgpu::produceOpenGlDirectGratingIrradianceCohort(frame,cohort,.25,readbacks,sum,error)
                ||!arbitgpu::produceOpenGlDirectGratingTerminalOrders(frame,cohort,sum.submission,.25,readbacks,terminal,error))return false;
            gc::DirectGratingTerminalReceiverReadback receiver;
            std::array<std::uint64_t,14> before{},after{};
            if(!arbitgpu::readOpenGlNativeAccountingForFixture(*frame,before,error)
                ||!arbitgpu::produceOpenGlDirectGratingTerminalReceivers(frame,terminal.nativeResult,sum.submission,.25,receiver,error)
                ||!arbitgpu::readOpenGlNativeAccountingForFixture(*frame,after,error)
                ||!arbitgpu::inspectOpenGlDirectGratingTerminalReceiverForFixture(*frame,receiver,error))return false;
            const auto expectedWork=receiver.bounds.queryWork+receiver.bounds.maximumRows*first->frame.triangles.size();
            if(receiver.rows.size()!=terminal.orders.size()||!receiver.nativeResult||receiver.nativeResult->owner()!=frame.get()
                ||receiver.nativeResult->retained().nativeResult||receiver.rows.data()==receiver.nativeResult->retained().rows.data()
                ||after[13]!=before[13]+1||after[5]-before[5]!=expectedWork
                ||after[4]-before[4]!=terminal.bounds.bytes||after[6]!=before[6]
                ||frame->colorImageHandle()||frame->colorTextureDescriptor().complete())
                return fail("OpenGL receiver operation lost exact independent rows, full query/lookup charge or private image ownership");
            unsigned pixel=0,offscreen=0,nonreceiver=0,miss=0,background=0,alternate=0,zero=0,positive=0,grazing=0;bool slanted=false;
            for(std::size_t i=0;i<receiver.rows.size();++i) {
                const auto& row=receiver.rows[i];gc::NativeDirectGratingTerminalReceiverInput exact;
                if(!gc::makeDirectGratingTerminalReceiverInput(terminal,first->frame,i,exact,error)
                    ||!gc::sameDirectGratingTerminalReceiverInput(row.input,exact)
                    ||!gc::sameDirectGratingTerminalEmittedOrder(row.emitted,terminal.orders[i])
                    ||row.input.key[3]!=i||!row.queryWork||row.pixel.worldArea!=0
                    ||row.hit.objectId==selection.object.value)
                    return fail("OpenGL receiver native output changed signed order/occurrence/energy or self-hit the emitting grating");
                pixel+=row.disposition==Row::Disposition::Pixel;offscreen+=row.disposition==Row::Disposition::Offscreen;
                nonreceiver+=row.disposition==Row::Disposition::NonReceiver;miss+=row.disposition==Row::Disposition::Miss;
                background+=row.rasterSample==Row::RasterSample::Background;
                alternate+=row.rasterSample==Row::RasterSample::Surface
                    &&std::tie(row.hit.objectId,row.hit.materialId,row.hit.instanceId,row.hit.triangleId)
                      !=std::tie(row.raster.object,row.raster.material,row.raster.instance,row.raster.triangle);
                zero+=row.emitted.orderIrradiance==0;positive+=row.emitted.orderIrradiance>0;
                if(row.emitted.order.primary==0) {
                    const gc::Point ray{row.input.directionMaximum[0],row.input.directionMaximum[1],row.input.directionMaximum[2]};
                    const float cosine=gc::dot(ray,terminal.incident.endpoints[row.emitted.endpoint].endpoint.outwardNormal);
                    grazing+=cosine>0&&cosine<1e-5f;
                }
                if(row.rasterSample==Row::RasterSample::Surface&&row.raster.object==row.hit.objectId
                    &&std::abs(row.raster.linearDepth-row.pixel.linearDepth)>1e-4f)slanted=true;
            }
            if((variant==0&&(!pixel||!miss||!zero||!positive))||(variant==1&&(!pixel||!slanted))
                ||(variant==2&&!offscreen)||(variant==3&&!nonreceiver)||(variant==4&&!miss)
                ||(variant==5&&!grazing)||(variant==6&&(!receiver.rows.empty()||receiver.queryWork||!terminal.orders.empty()))
                ||(variant==7&&!background)||(variant==8&&!alternate))
                return fail("OpenGL receiver native fixture lacks its positive, miss, nonreceiver, offscreen, slanted, grazing, empty, background or occluder evidence");
            if(variant!=0)continue;
            const auto incumbent=receiver;
            std::shared_ptr<const gc::NativeDirectGratingTerminalReceiverResult> retained;
            std::array<unsigned,2> buffers{};std::uintptr_t fence=0;std::array<std::uint64_t,3> counters{};
            if(!arbitgpu::readOpenGlDirectGratingTerminalReceiverStateForFixture(*frame,retained,buffers,fence,counters,error)
                ||retained!=receiver.nativeResult||!buffers[0]||!buffers[1]||!fence||counters[2]!=1)return false;
            const auto refuses=[&](auto mutate) {
                auto changed=incumbent;mutate(changed);
                return !arbitgpu::inspectOpenGlDirectGratingTerminalReceiverForFixture(*frame,changed,error)
                    &&gc::sameDirectGratingTerminalReceiverReadback(receiver,incumbent);
            };
            if(!refuses([](auto& r){r.inputGeneration++;})||!refuses([](auto& r){r.submission.completedFence++;})
                ||!refuses([](auto& r){r.bounds.queryWork++;})||!refuses([](auto& r){r.rows[0].input.key[3]++;})
                ||!refuses([](auto& r){r.rows[0].emitted.order.primary=-r.rows[0].emitted.order.primary+1;})
                ||!refuses([](auto& r){r.rows[0].emitted.orderIrradiance+=1;})||!refuses([](auto& r){r.rows[0].hit.distance+=.1f;})
                ||!refuses([](auto& r){r.rows[0].raster.linearDepth+=.1f;})
                ||!refuses([](auto& r){std::swap(r.rows[0],r.rows[1]);}))
                return fail("OpenGL receiver native inspector accepted caller order, energy, occurrence, ray, raster or generation tampering");
            auto stale=sum.submission;stale.queryGeneration++;
            if(arbitgpu::produceOpenGlDirectGratingTerminalReceivers(dimensionalFrame,terminal.nativeResult,sum.submission,.25,receiver,error)
                ||arbitgpu::produceOpenGlDirectGratingTerminalReceivers(frame,terminal.nativeResult,stale,.25,receiver,error)
                ||arbitgpu::produceOpenGlDirectGratingTerminalReceivers(frame,terminal.nativeResult,sum.submission,.5,receiver,error)
                ||!gc::sameDirectGratingTerminalReceiverReadback(receiver,incumbent)
                ||!arbitgpu::readOpenGlNativeAccountingForFixture(*frame,before,error)||before!=after)
                return fail("OpenGL receiver stale time/query admission charged work or replaced an incumbent");
            arbitgl::GlFuncs gl;std::string missing;
            if(!arbitgl::loadGlFunctions(gl,missing))return fail("OpenGL receiver fixture lacks native function loading");
            int previous=0;glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING,&previous);
            for(unsigned slot=0;slot<2;++slot) {
                gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[slot]);
                auto* mapped=static_cast<std::uint32_t*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,16,GL_MAP_READ_BIT|GL_MAP_WRITE_BIT));
                if(!mapped){return false;}const auto saved=mapped[3];mapped[3]^=1;
                if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE)return false;
                const bool accepted=arbitgpu::inspectOpenGlDirectGratingTerminalReceiverForFixture(*frame,receiver,error);
                gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[slot]);
                mapped=static_cast<std::uint32_t*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,16,GL_MAP_WRITE_BIT));
                if(!mapped){return false;}mapped[3]=saved;if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE)return false;
                gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,unsigned(previous));
                if(accepted||!arbitgpu::inspectOpenGlDirectGratingTerminalReceiverForFixture(*frame,receiver,error))
                    return fail("OpenGL receiver inspector accepted mutation of actual native input or decoded output storage");
            }
            const auto* native=dynamic_cast<const gc::NativeDirectGratingIrradianceCohortBinding*>(frame.get());
            {
                std::unique_ptr<gc::NativeDirectGratingTerminalReceiverLease> lease;gc::DirectGratingTerminalReceiverNativeBudget budget;
                gc::DirectGratingTerminalReceiverBounds bounds;
                if(!native||!native->beginDirectTerminalReceiverLease(terminal.nativeResult,sum.submission,.25,lease,budget,error))return false;
                budget.retainedBytes=renderpassoutput::kMaximumTotalBytes;
                if(gc::calculateDirectGratingTerminalReceiverBounds(*terminal.nativeResult,lease->frozenFrame(),budget,receiver,bounds,error))
                    return fail("OpenGL receiver bounds accepted actual native storage at the full cap");
            }
            for(const auto stop:{arbitgpu::OpenGlTerminalReceiverStopForFixture::Reserved,arbitgpu::OpenGlTerminalReceiverStopForFixture::Fence}) {
                if(!arbitgpu::readOpenGlNativeAccountingForFixture(*frame,before,error))return false;
                if(arbitgpu::produceOpenGlDirectGratingTerminalReceivers(frame,terminal.nativeResult,sum.submission,.25,receiver,error,stop)
                    ||!gc::sameDirectGratingTerminalReceiverReadback(receiver,incumbent)
                    ||!arbitgpu::readOpenGlNativeAccountingForFixture(*frame,after,error)
                    ||after[5]-before[5]!=expectedWork||after[13]!=before[13]+1||after[4]-before[4]!=terminal.bounds.bytes
                    ||after[0]!=before[0]||!arbitgpu::inspectOpenGlDirectGratingTerminalReceiverForFixture(*frame,incumbent,error))
                    return fail("OpenGL receiver failed admitted allocation/fence attempt dropped work, native owner bytes or an earlier result");
            }
            if(!arbitgpu::produceOpenGlDirectGratingTerminalReceivers(frame,terminal.nativeResult,sum.submission,.25,receiver,error)
                ||receiver.nativeResult==incumbent.nativeResult||receiver.inputGeneration==incumbent.inputGeneration
                ||!arbitgpu::inspectOpenGlDirectGratingTerminalReceiverForFixture(*frame,incumbent,error)
                ||!arbitgpu::readOpenGlDirectGratingTerminalReceiverStateForFixture(*frame,retained,buffers,fence,counters,error)||counters[2]!=2)
                return fail("OpenGL receiver valid retry replaced or invalidated an earlier externally retained result");
            std::shared_ptr<const gc::NativeDirectGratingTerminalReceiverResult> escaped;
            if(!arbitgpu::readOpenGlNativeAccountingForFixture(*frame,before,error))return false;
            {
                std::unique_ptr<gc::NativeDirectGratingTerminalReceiverLease> lease;gc::DirectGratingTerminalReceiverNativeBudget budget;
                gc::DirectGratingTerminalReceiverBounds bounds;
                if(!native->beginDirectTerminalReceiverLease(terminal.nativeResult,sum.submission,.25,lease,budget,error)
                    ||!gc::calculateDirectGratingTerminalReceiverBounds(*terminal.nativeResult,lease->frozenFrame(),budget,receiver,bounds,error)
                    ||!native->reserveDirectTerminalReceiverLease(*lease,bounds,receiver,error))return false;
                gc::DirectGratingTerminalReceiverReadback staged;staged.terminal=terminal.nativeResult;staged.submission=sum.submission;
                staged.evaluationTime=.25;staged.inputGeneration=lease->inputGeneration();staged.bounds=bounds;
                std::vector<gc::NativeDirectGratingTerminalReceiverInput> rays;rays.reserve(bounds.maximumRows);
                for(std::size_t i=0;i<terminal.orders.size();++i) {
                    gc::NativeDirectGratingTerminalReceiverInput ray;
                    if(!gc::makeDirectGratingTerminalReceiverInput(terminal,lease->frozenFrame(),i,ray,error))return false;
                    rays.push_back(ray);
                }
                if(!native->dispatchDirectTerminalReceivers(*lease,rays,staged,error)
                    ||!native->finishDirectTerminalReceiverLease(*lease,staged,staged.nativeResult,error)
                    ||!native->validateDirectTerminalReceiverResult(*lease,staged,error))return false;
                escaped=staged.nativeResult;lease.reset();
            }
            auto unpublished=escaped->retained();unpublished.nativeResult=escaped;
            if(!arbitgpu::readOpenGlNativeAccountingForFixture(*frame,after,error)
                ||after[0]<=before[0]||after[5]-before[5]!=expectedWork||after[13]!=before[13]+1
                ||after[4]-before[4]!=terminal.bounds.bytes
                ||arbitgpu::inspectOpenGlDirectGratingTerminalReceiverForFixture(*frame,unpublished,error)
                ||!arbitgpu::inspectOpenGlDirectGratingTerminalReceiverForFixture(*frame,incumbent,error))
                return fail("OpenGL escaped uncommitted receiver result lost storage or acquired published native authority");
            if(!arbitgpu::produceOpenGlDirectGratingTerminalOrders(frame,cohort,sum.submission,.25,readbacks,terminal,error)
                ||!arbitgpu::inspectOpenGlDirectGratingTerminalReceiverForFixture(*frame,incumbent,error))
                return fail("OpenGL later terminal admission ignored receiver work/storage or dropped its prior owner");
            // Dropping only the scalar charge cannot erase a frozen failed attempt.
            if(!arbitgpu::readOpenGlNativeAccountingForFixture(*frame,before,error))return false;
            auto reduced=before;--reduced[5];
            if(!arbitgpu::exchangeOpenGlNativeAccountingForFixture(*frame,reduced,error))return false;
            const bool admitted=arbitgpu::produceOpenGlDirectGratingTerminalReceivers(frame,terminal.nativeResult,sum.submission,.25,receiver,error);
            if(!arbitgpu::exchangeOpenGlNativeAccountingForFixture(*frame,reduced,error)||admitted
                ||!arbitgpu::inspectOpenGlDirectGratingTerminalReceiverForFixture(*frame,incumbent,error))
                return fail("OpenGL later receiver producer accepted a scalar rollback of retained failed-query work");
            const auto* stress=std::getenv("ARBIT_GL_RECEIVER_BUDGET_STRESS");
            if(stress&&std::string(stress)=="1") {
                bool exhausted=false;const auto caller=receiver;
                for(unsigned attempt=0;attempt<256;++attempt) {
                    if(!arbitgpu::readOpenGlNativeAccountingForFixture(*frame,before,error))return false;
                    if(arbitgpu::produceOpenGlDirectGratingTerminalReceivers(frame,terminal.nativeResult,sum.submission,.25,receiver,error,
                        arbitgpu::OpenGlTerminalReceiverStopForFixture::Reserved))return fail("OpenGL refused receiver fixture published a result");
                    if(!arbitgpu::readOpenGlNativeAccountingForFixture(*frame,after,error)
                        ||!gc::sameDirectGratingTerminalReceiverReadback(receiver,caller))return false;
                    if(after[13]==before[13]) {
                        if(after!=before)return fail("OpenGL over-cap receiver attempt changed unadmitted counters");
                        exhausted=true;break;
                    }
                    if(after[13]!=before[13]+1||after[5]-before[5]!=expectedWork||after[4]-before[4]!=terminal.bounds.bytes
                        ||after[0]!=before[0])return fail("OpenGL repeated admitted receiver failures lost conservative query or owner charges");
                }
                if(!exhausted)return fail("OpenGL actual receiver retries escaped the original cumulative caps");
            }
            if(!arbitgpu::readOpenGlDirectGratingTerminalReceiverStateForFixture(*frame,retained,buffers,fence,counters,error))return false;
            const auto deleteSync=reinterpret_cast<PFNGLDELETESYNCPROC>(glfwGetProcAddress("glDeleteSync"));
            if(!deleteSync||!fence)return false;
            deleteSync(reinterpret_cast<GLsync>(fence));
            if(arbitgpu::inspectOpenGlDirectGratingTerminalReceiverForFixture(*frame,receiver,error)
                ||arbitgpu::produceOpenGlDirectGratingTerminalReceivers(frame,terminal.nativeResult,sum.submission,.25,receiver,error))
                return fail("OpenGL receiver inspector or later admission accepted a deleted actual receiver fence");
        }
    }
    {
        // Author a real quarter-turned scene, then emit zero-order grating rays
        // through its native cohort/terminal/receiver route. No parent row or
        // camera constructor input is supplied as accepted fixture evidence.
        auto physical=description;physical.spectrum.firstOrder=physical.spectrum.lastOrder=0;
        const auto admitted=diffractionmaterial::admit(physical,error);if(!admitted)return false;
        const auto ir=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*admitted);
        const auto leaf=compileGrating(physical,ir->structuralDigest());if(!leaf)return false;
        const arbitgpu::smoothglass::DistinctGratingAdmission diagonalSelection{target.id,ir};
        using Row=gc::DirectGratingTerminalReceiverRow;
        for(unsigned variant=0;variant<7;++variant) {
            auto scene=std::make_shared<gt::Visual3DScene>(*dimensionalScene);
            auto root=std::make_shared<gt::Program>(*dimensionalBinding->prepared.glass.sourceMaterial);
            const float half=float(std::sqrt(.5));
            for(std::size_t v=target.firstVertex;v<target.firstVertex+target.vertexCount;++v) {
                auto& vertex=scene->vertices[v];vertex.position.z=vertex.position.x-6;
                vertex.normal={-half,0,half};vertex.tangent={0,1,0,1};
            }
            for(std::size_t v=0;v<scene->vertexCount;++v) {
                auto& vertex=scene->vertices[v];const auto p=vertex.position,n=vertex.normal;
                vertex.position={-p.z,p.y,p.x};vertex.normal={-n.z,n.y,n.x};
                const auto t=vertex.tangent;vertex.tangent={-t[2],t[1],t[0],t[3]};
            }
            scene->lights[0].transform.rotation={0,-half,0,half};
            // Two repeated emitted occurrences retain pre-dedup charging while
            // the seven supported zero bands keep retry admission bounded.
            scene->lights[0].spectralSource.directionalIrradiance.coefficients.fill(0);
            scene->lights[0].spectralSource.directionalIrradiance.coefficients[0]=variant==6?0:.25f;
            auto& wall=scene->objects[receiverIndex];
            const HarmonicMIDI::grid::SceneVec3 corners[]={{-5,7.75f,-4},{-5,-8.25f,-4},{11,-8.25f,-4},{11,7.75f,-4}};
            for(unsigned v=0;v<4;++v) {
                auto& vertex=scene->vertices[wall.firstVertex+v];vertex.position=corners[v];
                vertex.normal={0,0,1};vertex.tangent={1,0,0,1};
            }
            auto& camera=scene->cameras[0];camera.transform={};camera.transform.translation={3,-.25f,-3};
            camera.verticalFovRadians=.1f;camera.nearPlane=variant==5?1:.01f;camera.farPlane=100;
            if(variant==1) {
                // Separate source endpoint IDs prevent accidental welding.
                const unsigned indices[]{0,2,3};const auto local=wall.vertexCount;
                for(unsigned v=0;v<3;++v) {
                    auto vertex=scene->vertices[wall.firstVertex+indices[v]];
                    if(v==0)vertex.position.x+=.125f;
                    scene->vertices[scene->vertexCount++]=vertex;
                    scene->indices[wall.firstIndex+3+v]=local+v;
                }
                wall.vertexCount+=3;
            }
            // Preserve both shared endpoints, but change the second plane.
            if(variant==4)scene->vertices[wall.firstVertex+3].position.z-=.015625f;
            if(variant==2||variant==3) {
                auto split=wall;split.id={wall.id.value+1000};split.firstIndex+=3;split.indexCount=3;wall.indexCount=3;
                auto program=std::make_shared<gt::Program>(*receiverProgram);program->object=split.id;
                if(variant==3) {
                    auto material=scene->materials[receiverIndex];material.id={split.material.value+1000};
                    scene->materials[scene->materialCount++]=material;split.material=material.id;
                    program->authoredMaterial=material.id;program->parameters.identifiers[0]=material.id.value;
                }
                program->bindingDigest=program->programIdentity="gl-diagonal-split-receiver";
                scene->objects[scene->objectCount++]=split;root->objectPrograms.push_back(program);
                std::sort(root->objectPrograms.begin(),root->objectPrograms.end(),[](const auto& a,const auto& b){return a->object.value<b->object.value;});
            }
            const auto resources=arbitgpu::prepareOpenGlGlassGratingResources(scene,root,diagonalSelection,leaf);
            if(!resources.prepared)return fail(resources.error.c_str());
            std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame;
            std::shared_ptr<const gc::GratingIncidentBinding> first;
            if(!arbitgpu::stageOpenGlGlassGratingFrame(resources.resources,scene->lights[0].id,*fixture.policy,1,1,
                runtime,.25,frame,first,error,scheduling))return false;
            const auto& triangle=first->triangles.front();const gc::Point point{2.984375f,-.25f,3.015625f};
            gc::NativeDirectGratingEndpointInput input;input.primitive={triangle.object,triangle.material,triangle.instance,triangle.triangle};
            const auto origin=gc::advance(point,triangle.normal,.1f);
            input.originMinimum={origin[0],origin[1],origin[2],gc::kRayMinimum};
            input.directionMaximum={-triangle.normal[0],-triangle.normal[1],-triangle.normal[2],1};
            std::shared_ptr<const gc::DirectGratingIrradianceCohort> cohort;
            if(!gc::prepareDirectGratingIrradianceCohort({first},{input,input},0,0,cohort,error))return false;
            std::vector<gc::NativeDirectGratingContributionReadback> sources;gc::DirectGratingIrradianceCohortReadback sum;
            gc::DirectGratingTerminalOrderLedger terminal;gc::DirectGratingTerminalReceiverReadback receiver;
            if(!arbitgpu::produceOpenGlDirectGratingIrradianceCohort(frame,cohort,.25,sources,sum,error)
                ||!arbitgpu::produceOpenGlDirectGratingTerminalOrders(frame,cohort,sum.submission,.25,sources,terminal,error)
                ||!arbitgpu::produceOpenGlDirectGratingTerminalReceivers(frame,terminal.nativeResult,sum.submission,.25,receiver,error)
                ||(variant!=6&&!std::any_of(receiver.rows.begin(),receiver.rows.end(),[](const auto& row){return row.disposition==Row::Disposition::Pixel;})))
                return fail("OpenGL diagonal fixture lacks an actual emitted Pixel parent with all raw triangles retained");
            gc::CameraCellCoverageReadback coverage;
            if(variant==6) {
                std::array<unsigned,2> buffers{};std::uintptr_t fence=0;std::array<std::uint64_t,8> before{},after{};
                if(!receiver.rows.empty()||!arbitgpu::readOpenGlCameraCellCoverageStorageForFixture(*frame,buffers,fence,before,error))return false;
                for(unsigned retry=0;retry<2;++retry) {
                    if(arbitgpu::produceOpenGlCameraCellCoverageForFixture(frame,receiver.nativeResult,sum.submission,.25,coverage,error,
                        arbitgpu::OpenGlCameraCoverageStopForFixture::Reserved)
                        ||!arbitgpu::readOpenGlCameraCellCoverageStorageForFixture(*frame,buffers,fence,after,error)
                        ||after[0]||after[1]||after[2]||after[6]<=before[6]||after[7]<=before[7]
                        ||coverage.nativeResult||buffers!=std::array<unsigned,2>{}||fence)
                        return fail("OpenGL zero-P diagonal refusal lost its permanent fixed setup/provenance work");
                    before=after;
                }
                if(!arbitgpu::produceOpenGlCameraCellCoverageForFixture(frame,receiver.nativeResult,sum.submission,.25,coverage,error)
                    ||!coverage.nativeResult||!coverage.preparation.pixels.empty()||!coverage.proof.regions.empty()
                    ||!coverage.proof.vertices.empty()||!arbitgpu::inspectOpenGlCameraCellCoverageForFixture(*frame,coverage,error)
                    ||!arbitgpu::readOpenGlCameraCellCoverageStorageForFixture(*frame,buffers,fence,after,error)
                    ||after[0]||after[1]||after[2]||after[6]<=before[6]||after[7]<=before[7]
                    ||!buffers[0]||!buffers[1]||!fence)
                    return fail("OpenGL zero-P actual link/fence lost fixed admission or fabricated a pixel region");
                continue;
            }
            if(variant) {
                if(arbitgpu::produceOpenGlCameraCellCoverageForFixture(frame,receiver.nativeResult,sum.submission,.25,coverage,error)
                    ||coverage.nativeResult||!coverage.proof.regions.empty()
                    ||!arbitgpu::inspectOpenGlDirectGratingTerminalReceiverForFixture(*frame,receiver,error))
                    return fail("OpenGL diagonal gap, owner/material split, unequal depth or clip fixture acquired native coverage");
                continue;
            }
            if(!arbitgpu::produceOpenGlCameraCellCoverageForFixture(frame,receiver.nativeResult,sum.submission,.25,coverage,error)
                ||!coverage.nativeResult||coverage.preparation.pixels.size()!=1||coverage.preparation.associations.size()<2
                ||coverage.proof.regions.size()!=2||coverage.proof.vertices.size()!=6
                ||coverage.bounds.triangles!=first->frame.triangles.size()
                ||!arbitgpu::inspectOpenGlCameraCellCoverageForFixture(*frame,coverage,error))
                return fail("OpenGL actual diagonal certificate did not partition the whole captured pixel into two closed native regions");
            const auto incumbent=coverage;
            const auto refuses=[&](auto mutate) {
                auto forged=incumbent;mutate(forged);
                return !arbitgpu::inspectOpenGlCameraCellCoverageForFixture(*frame,forged,error)
                    &&gc::sameCameraCellCoverageReadback(coverage,incumbent);
            };
            if(!refuses([](auto& r){++r.proof.generation;})||!refuses([](auto& r){r.proof.vertices[0][0]=.5;})
                ||!refuses([](auto& r){r.proof.regions[0].projectedArea=.25;})
                ||!refuses([](auto& r){++r.preparation.associations[0].parentRow;})
                ||!refuses([](auto& r){++r.submission.completedFence;}))
                return fail("OpenGL diagonal inspector accepted caller certificate or parent forgery");
            std::array<unsigned,2> buffers{};std::uintptr_t fence=0;std::array<std::uint64_t,8> before{},after{};
            if(!arbitgpu::readOpenGlCameraCellCoverageStorageForFixture(*frame,buffers,fence,before,error)||!buffers[0]||!buffers[1]||!fence)return false;
            arbitgl::GlFuncs gl;std::string missing;if(!arbitgl::loadGlFunctions(gl,missing))return false;
            int previous=0;glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING,&previous);
            for(unsigned slot=0;slot<2;++slot) {
                gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[slot]);
                auto* mapped=static_cast<std::uint32_t*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,4,GL_MAP_READ_BIT|GL_MAP_WRITE_BIT));
                if(!mapped)return false;
                const auto saved=*mapped;*mapped^=1;if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE)return false;
                const bool accepted=arbitgpu::inspectOpenGlCameraCellCoverageForFixture(*frame,coverage,error);
                gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[slot]);mapped=static_cast<std::uint32_t*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,4,GL_MAP_WRITE_BIT));
                if(!mapped)return false;
                *mapped=saved;if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE)return false;
                gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,unsigned(previous));
                if(accepted||!arbitgpu::inspectOpenGlCameraCellCoverageForFixture(*frame,incumbent,error))
                    return fail("OpenGL diagonal inspector accepted actual native input/certificate mutation");
            }
            for(const auto stop:{arbitgpu::OpenGlCameraCoverageStopForFixture::Reserved,arbitgpu::OpenGlCameraCoverageStopForFixture::Fence}) {
                if(!arbitgpu::readOpenGlCameraCellCoverageStorageForFixture(*frame,buffers,fence,before,error))return false;
                if(arbitgpu::produceOpenGlCameraCellCoverageForFixture(frame,receiver.nativeResult,sum.submission,.25,coverage,error,stop)
                    ||!gc::sameCameraCellCoverageReadback(coverage,incumbent)
                    ||!arbitgpu::readOpenGlCameraCellCoverageStorageForFixture(*frame,buffers,fence,after,error)
                    ||after[0]-before[0]!=coverage.bounds.maximumPixels*coverage.bounds.triangles
                    ||after[1]-before[1]!=coverage.bounds.maximumPixels*(coverage.bounds.triangles-1)
                    ||after[2]-before[2]!=coverage.bounds.maximumPixels*(5*coverage.bounds.triangles+3)
                    ||after[6]<=before[6]||after[7]<=before[7]
                    ||!arbitgpu::inspectOpenGlCameraCellCoverageForFixture(*frame,incumbent,error))
                    return fail("OpenGL diagonal failed attempt lost permanent requests/setup work or its incumbent");
            }
            if(!arbitgpu::produceOpenGlCameraCellCoverageForFixture(frame,receiver.nativeResult,sum.submission,.25,coverage,error)
                ||coverage.nativeResult==incumbent.nativeResult||coverage.proof.generation==incumbent.proof.generation
                ||!arbitgpu::inspectOpenGlCameraCellCoverageForFixture(*frame,coverage,error)
                ||!arbitgpu::inspectOpenGlCameraCellCoverageForFixture(*frame,incumbent,error))
                return fail("OpenGL diagonal valid retry reused storage or invalidated an older immutable result");
            const auto published=coverage;
            const auto* native=dynamic_cast<const gc::NativeCameraCellCoverageBinding*>(frame.get());
            std::shared_ptr<const gc::NativeCameraCellCoverageResult> escaped;
            std::array<std::uint64_t,14> nativeBefore{},nativeAfter{};
            if(!native||!arbitgpu::readOpenGlNativeAccountingForFixture(*frame,nativeBefore,error))return false;
            {
                std::unique_ptr<gc::NativeCameraCellCoverageLease> lease;gc::NativeCameraCellCoverageBudget budget;
                gc::CameraCellCoverageBounds bounds;
                if(!native->beginCameraCellCoverageLease(receiver.nativeResult,sum.submission,.25,lease,budget,error)
                    ||!gc::calculateCameraCellCoverageBounds(*receiver.nativeResult,lease->frozenFrame(),budget,coverage,bounds,error)
                    ||!native->reserveCameraCellCoverageLease(*lease,bounds,coverage,error))return false;
                gc::CameraCellCoverageReadback staged;staged.parent=receiver.nativeResult;staged.submission=sum.submission;
                staged.evaluationTime=.25;staged.inputGeneration=lease->inputGeneration();staged.bounds=bounds;
                if(!gc::prepareCameraCellCoverage(*receiver.nativeResult,lease->frozenFrame(),bounds,staged.preparation,error)
                    ||!native->dispatchCameraCellCoverage(*lease,staged.preparation,staged.proof,error)
                    ||!native->validateCameraCellCoverageProof(*lease,staged,error)
                    ||!native->finishCameraCellCoverageLease(*lease,staged,staged.nativeResult,error)
                    ||!native->validateCameraCellCoverageProof(*lease,staged,error))return false;
                escaped=staged.nativeResult;lease.reset();
            }
            auto unpublished=escaped->retained();unpublished.nativeResult=escaped;
            if(!arbitgpu::readOpenGlNativeAccountingForFixture(*frame,nativeAfter,error)||nativeAfter[0]<=nativeBefore[0]
                ||arbitgpu::inspectOpenGlCameraCellCoverageForFixture(*frame,unpublished,error)
                ||!arbitgpu::inspectOpenGlCameraCellCoverageForFixture(*frame,published,error)
                ||!arbitgpu::inspectOpenGlCameraCellCoverageForFixture(*frame,incumbent,error))
                return fail("OpenGL escaped uncommitted diagonal result lost storage or acquired publication authority");
            if(!arbitgpu::readOpenGlCameraCellCoverageStorageForFixture(*frame,buffers,fence,before,error))return false;
            auto stale=sum.submission;++stale.queryGeneration;
            if(arbitgpu::produceOpenGlCameraCellCoverageForFixture(frame,receiver.nativeResult,stale,.25,coverage,error)
                ||arbitgpu::produceOpenGlCameraCellCoverageForFixture(frame,receiver.nativeResult,sum.submission,.5,coverage,error)
                ||!gc::sameCameraCellCoverageReadback(coverage,published)
                ||!arbitgpu::readOpenGlCameraCellCoverageStorageForFixture(*frame,buffers,fence,after,error)||after!=before)
                return fail("OpenGL stale diagonal parent/time admission charged work or replaced its incumbent");
            const auto deleteSync=reinterpret_cast<PFNGLDELETESYNCPROC>(glfwGetProcAddress("glDeleteSync"));
            if(!deleteSync||!fence)return false;
            deleteSync(reinterpret_cast<GLsync>(fence));
            if(arbitgpu::inspectOpenGlCameraCellCoverageForFixture(*frame,published,error)
                ||arbitgpu::produceOpenGlCameraCellCoverageForFixture(frame,receiver.nativeResult,sum.submission,.25,coverage,error)
                ||!gc::sameCameraCellCoverageReadback(coverage,published))
                return fail("OpenGL diagonal deleted native fence was accepted by inspection or later admission");
        }
    }
    std::shared_ptr<const gc::GratingEntryCellBinding> sourceCells;
    // Fresh private camera owners retain the actual closed volume, camera and
    // distinct material. Camera tracing must not borrow the forward receipt.
    const auto cameraTraceStage=[&](bool wedge,bool obstruction,bool seam,
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& cameraFrame,
        std::shared_ptr<const gc::GratingIncidentBinding>& cameraBinding,float cameraZ=3) {
        auto scene=std::make_shared<gt::Visual3DScene>(*dimensionalScene);
        const auto& glass=scene->objects[0];
        for(std::size_t vertex=glass.firstVertex;vertex<glass.firstVertex+glass.vertexCount;++vertex) {
            auto& point=scene->vertices[vertex].position;const bool upper=point.z>0;
            if(upper)point.z-=.25f*point.x;
            point.z+=(wedge?(upper?.75f:0.0f):seam?0.0f:.2f)*point.x;
        }
        auto& camera=scene->cameras[0];camera.transform.translation=seam
            ?HarmonicMIDI::grid::SceneVec3{-.2f,-.2f,cameraZ}:HarmonicMIDI::grid::SceneVec3{-.25f,-.15f,cameraZ};
        const double angle=seam?0:-std::atan(.1);
        camera.transform.rotation={0,float(std::sin(angle/2)),0,float(std::cos(angle/2))};
        auto root=std::make_shared<gt::Program>(*dimensionalBinding->prepared.glass.sourceMaterial);
        if(obstruction) {
            const auto object=scene->objectCount;
            videohelper::tests::smoothglassfixture::receiver(*scene,-2,{0,0,0});
            const auto& blocker=scene->objects[object];
            for(std::size_t vertex=blocker.firstVertex;vertex<blocker.firstVertex+blocker.vertexCount;++vertex)
                scene->vertices[vertex].position.z=-2;
            auto owner=std::make_shared<gt::Program>(*receiverProgram);
            owner->object=blocker.id;owner->authoredMaterial=blocker.material;
            owner->parameters.identifiers[0]=blocker.material.value;
            owner->bindingDigest="gl-camera-grating-blocker";owner->programIdentity="gl-camera-grating-blocker";
            root->objectPrograms.push_back(owner);
            std::sort(root->objectPrograms.begin(),root->objectPrograms.end(),[](const auto& a,const auto& b){return a->object.value<b->object.value;});
        }
        const auto admitted=arbitgpu::prepareOpenGlGlassGratingResources(scene,root,selection,grating);
        return admitted.prepared&&arbitgpu::stageOpenGlGlassGratingFrame(admitted.resources,fixture.ordinaryLight,*fixture.policy,
            32,32,runtime,.25,cameraFrame,cameraBinding,error,scheduling);
    };
    // Actual private production batches only. Binary64 optics below compare
    // their witnesses; they never produce a native trace or completion receipt.
    {
        using Disposition=gc::GlassCameraTraceDisposition;
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> cameraFrame;
        std::shared_ptr<const gc::GratingIncidentBinding> cameraBinding;
        if(!cameraTraceStage(false,false,false,cameraFrame,cameraBinding))return false;
        const auto* owner=dynamic_cast<const gc::NativeGlassCameraGratingTraceBinding*>(cameraFrame.get());
        if(!owner||owner->cameraTraceInputUpload().complete)return fail("OpenGL camera staging fabricated native trace completion");
        std::vector<gc::NativeGlassCameraGratingTraceInput> inputs(5);
        for(unsigned location=0;location<5;++location)
            if(!gc::makeGlassCameraGratingTraceInput(cameraBinding->frame,16,16,location,inputs[location],error))return false;
        gc::NativeGlassCameraGratingTraceReadback trace;
        if(!arbitgpu::produceOpenGlGlassCameraGratingTrace(cameraFrame,cameraBinding,inputs,.25,trace,error)
            ||!trace.dispatchComplete||trace.overflow||trace.bands.size()!=40||!trace.queryWork
            ||!gc::validateGlassCameraGratingTraceReadback(*cameraBinding,*owner,trace.submission,.25,trace,error))return false;
        std::shared_ptr<const arbitgpu::OpenGlQueryArithmeticEvidence> evidence;
        if(!readQueryContract(*cameraFrame,arbitgpu::OpenGlQueryArithmeticRole::GlassCameraGratingTrace,&trace.submission,evidence,error))return false;
        unsigned terminals=0,interfaces=0,reflections=0;bool dispersed=false,cornersDiffer=false;
        const auto near=[](double a,double b,double tolerance=2e-5){return std::abs(a-b)<=tolerance;};
        const auto cameraTriangles=gt::worldTriangles(*cameraBinding->prepared.glass.sourceScene);
        for(const auto& band:trace.bands) {
            const auto& pixel=band.input.pixel;
            const double dx=pixel[2]==0?.5:(pixel[2]==2||pixel[2]==3?1:0),dy=pixel[2]==0?.5:(pixel[2]==3||pixel[2]==4?1:0);
            const double tangent=std::tan(double(cameraBinding->frame.camera.verticalFovRadians)*.5),angle=-std::atan(.1);
            const gt::D3 local{(2*(pixel[0]+dx)/32-1)*tangent,(1-2*(pixel[1]+dy)/32)*tangent,-1};
            const auto projected=gt::normalized(gt::D3{std::cos(angle)*local[0]+std::sin(angle)*local[2],local[1],
                -std::sin(angle)*local[0]+std::cos(angle)*local[2]});
            for(unsigned axis=0;axis<3;++axis)if(!near(projected[axis],band.nativeDirectionMaximum[axis]))
                return fail("OpenGL native camera center/corner ray differs from independent binary64 projection");
            if(band.visits.empty()||band.input.pixel[2]!=band.sample||band.band>=8
                ||band.ior!=cameraBinding->frame.ior[band.band]||band.wavelength!=diffractionmaterial::kProductionWavelengthsNanometres[band.band])
                return fail("OpenGL camera native sample, ray or eight-band owner changed");
            for(const auto& visit:band.visits) {
                if(visit.hit.status>=0) {
                    const gt::D3 origin{visit.originMinimum[0],visit.originMinimum[1],visit.originMinimum[2]},
                        direction{visit.directionFraction[0],visit.directionFraction[1],visit.directionFraction[2]};
                    const auto reference=gt::closest(cameraTriangles,origin,direction,visit.originMinimum[3],cameraBinding->frame.camera.farPlane);
                    if(reference.found!=(visit.hit.status==1)||(reference.found
                        &&(reference.object!=visit.hit.objectId||reference.material!=visit.hit.materialId
                            ||reference.instance!=visit.hit.instanceId||reference.triangle!=visit.hit.triangleId)))
                        return fail("OpenGL camera native interface/exit/terminal query differs from binary64 owning triangles");
                    if(reference.found)for(unsigned axis=0;axis<3;++axis)
                        if(!near(reference.position[axis],visit.hit.position[axis],8e-5))
                            return fail("OpenGL camera native interface/exit point differs from independent binary64 geometry");
                }
                if(visit.disposition==Disposition::Interface) {
                    ++interfaces;
                    const gt::D3 ray{visit.directionFraction[0],visit.directionFraction[1],visit.directionFraction[2]},
                        normal{visit.orientedNormal[0],visit.orientedNormal[1],visit.orientedNormal[2]};
                    const auto optics=videohelper::tests::smoothglassfixture::interface(ray,normal,visit.fromIor,visit.toIor);
                    if(optics.tir!=visit.tir||!near(optics.reflectance,visit.reflection)
                        ||!near(double(visit.directionFraction[3])*optics.reflectance,visit.reflectedFraction)
                        ||!near(double(visit.directionFraction[3])*(1-optics.reflectance),visit.transmittedFraction))
                        return fail("OpenGL camera native Fresnel fractions differ from independent binary64 interfaces");
                    for(unsigned axis=0;axis<3;++axis)if(!near(optics.reflection[axis],visit.reflected[axis])
                        ||(!visit.tir&&!near(optics.transmission[axis],visit.transmitted[axis])))
                        return fail("OpenGL camera native reflected/transmitted directions differ from binary64 interfaces");
                }
                if(visit.parent!=gc::kGlassCameraTraceRoot) {
                    if(visit.parent>=visit.visit)return fail("OpenGL camera child precedes its native parent");
                    const auto& parent=band.visits[visit.parent];const bool reflected=visit.token==1;
                    if(reflected)++reflections;
                    const double offset=std::max(1e-5,2e-6*std::max({std::abs(double(parent.hit.position[0])),
                        std::abs(double(parent.hit.position[1])),std::abs(double(parent.hit.position[2]))}));
                    for(unsigned axis=0;axis<3;++axis)
                        if(!near(double(parent.hit.position[axis])+(reflected?1:-1)*offset*parent.orientedNormal[axis],visit.originMinimum[axis],3e-6))
                            return fail("OpenGL camera native child offset does not follow its actual oriented interface");
                }
                if(visit.disposition==Disposition::Grating) {
                    ++terminals;
                    if(visit.interfaces<2||visit.inside||!visit.hit.entering||visit.barycentric[0]<=1e-5f
                        ||visit.barycentric[1]<=1e-5f||visit.barycentric[2]<=1e-5f)
                        return fail("OpenGL camera grating terminal skipped the closed slab or owning interior primitive");
                    const auto first=std::find_if(trace.bands[0].visits.begin(),trace.bands[0].visits.end(),[](const auto& v){return v.disposition==Disposition::Grating;});
                    if(first!=trace.bands[0].visits.end()) {
                        if(band.sample==0&&band.band==7&&std::abs(visit.hit.position[0]-first->hit.position[0])>1e-5f)dispersed=true;
                        if(band.sample>0&&band.band==0&&std::abs(visit.hit.position[0]-first->hit.position[0])>1e-4f)cornersDiffer=true;
                    }
                    if(visit.interfaces==2) {
                        // The two parallel faces restore the native incoming
                        // direction while retaining the eight distinct offsets.
                        for(unsigned axis=0;axis<3;++axis)
                            if(!near(visit.directionFraction[axis],band.nativeDirectionMaximum[axis],4e-5))
                                return fail("OpenGL parallel camera slab did not restore its incident direction");
                    }
                }
            }
            if(!band.traceComplete&&!band.reasons)return fail("OpenGL camera prefix became unexplained clear zero");
        }
        if(terminals<40||interfaces<80||!reflections||!dispersed||!cornersDiffer)
            return fail("OpenGL native slab fixture lacks eight-band terminals, Fresnel children or corner dispersion");
        const auto exact=trace;
        if(!owner->validateCameraTraceProductionQuery(trace.submission,exact,error))return false;
        using Mutation=void(*)(gc::NativeGlassCameraGratingTraceReadback&);
        const Mutation mutations[]{
            [](auto& r){r.version++;},[](auto& r){r.dispatchComplete=false;},[](auto& r){r.overflow=true;},
            [](auto& r){r.inputGeneration++;},[](auto& r){r.queryWork++;},[](auto& r){r.submission.rendererGeneration++;},
            [](auto& r){r.submission.queryGeneration++;},[](auto& r){r.submission.rasterGeneration++;},
            [](auto& r){r.submission.completedFence++;},[](auto& r){r.submission.scratchGeneration++;},
            [](auto& r){r.bands.pop_back();},[](auto& r){r.bands[0].input.pixel[2]++;},
            [](auto& r){r.bands[0].input.control[0]++;},[](auto& r){r.bands[0].input.originMinimum[0]+=.01f;},
            [](auto& r){r.bands[0].nativeDirectionMaximum[0]+=.01f;},[](auto& r){r.bands[0].band++;},
            [](auto& r){r.bands[0].ior+=.01f;},[](auto& r){r.bands[0].wavelength++;},
            [](auto& r){r.bands[0].traceComplete=!r.bands[0].traceComplete;},[](auto& r){r.bands[0].reasons^=1;},
            [](auto& r){r.bands[0].queryWork++;},[](auto& r){r.bands[0].unresolvedFraction=.5f;},
            [](auto& r){r.bands[0].visits[0].parent=0;},[](auto& r){r.bands[0].visits[0].history++;},
            [](auto& r){r.bands[0].visits[0].inside=true;},[](auto& r){r.bands[0].visits[0].ancestry[0]=1;},
            [](auto& r){r.bands[0].visits[0].hit.status=-1;},[](auto& r){r.bands[0].visits[0].hit.work++;},
            [](auto& r){r.bands[0].visits[0].hit.objectId++;},[](auto& r){r.bands[0].visits[0].hit.triangleId++;},
            [](auto& r){r.bands[0].visits[0].reflection*=.5f;},[](auto& r){r.bands[0].visits[0].transmittedFraction*=.5f;},
            [](auto& r){r.bands[0].visits[0].reflected[0]+=.01f;},[](auto& r){r.bands[0].visits[0].tir=true;},
            [](auto& r){r.bands[0].visits[0].fromIor+=.01f;},[](auto& r){r.bands[0].visits[0].barycentric[0]=-0.0f;}
        };
        for(const auto mutate:mutations) {
            auto changed=trace;mutate(changed);
            if(owner->validateCameraTraceProductionQuery(trace.submission,changed,error)
                ||gc::validateGlassCameraGratingTraceReadback(*cameraBinding,*owner,trace.submission,.25,changed,error))
                return fail("OpenGL camera caller mutation replaced actual retained native output");
        }
        const auto generation=owner->cameraTraceInputUpload().generation;
        auto badInput=inputs;badInput[0].directionMaximum[0]+=.01f;
        auto stale=std::make_shared<gc::GratingIncidentBinding>(*cameraBinding);
        if(arbitgpu::produceOpenGlGlassCameraGratingTrace(cameraFrame,cameraBinding,badInput,.25,trace,error)
            ||arbitgpu::produceOpenGlGlassCameraGratingTrace(cameraFrame,stale,inputs,.25,trace,error)
            ||arbitgpu::produceOpenGlGlassCameraGratingTrace(cameraFrame,cameraBinding,inputs,.5,trace,error)
            ||owner->cameraTraceInputUpload().generation!=generation||!gc::sameGlassCameraGratingTraceReadback(trace,exact))
            return fail("OpenGL camera refused input, owner or time replaced its incumbent");
        auto* context=glfwGetCurrentContext();glfwMakeContextCurrent(nullptr);
        const bool foreign=arbitgpu::produceOpenGlGlassCameraGratingTrace(cameraFrame,cameraBinding,inputs,.25,trace,error);
        const bool foreignProof=owner->validateCameraTraceProductionQuery(exact.submission,exact,error);glfwMakeContextCurrent(context);
        if(foreign||foreignProof||!owner->validateCameraTraceProductionQuery(exact.submission,exact,error))
            return fail("OpenGL camera trace accepted foreign context or lost the restored native owner");
        const auto previousBytes=owner->cameraTraceInputUpload().otherOwnedBytes,previousWork=owner->cameraTraceInputUpload().otherWork;
        if(!arbitgpu::produceOpenGlGlassCameraGratingTrace(cameraFrame,cameraBinding,inputs,.25,trace,error)
            ||owner->cameraTraceInputUpload().otherOwnedBytes<=previousBytes||owner->cameraTraceInputUpload().otherWork<=previousWork
            ||owner->validateCameraTraceProductionQuery(exact.submission,exact,error))
            return fail("OpenGL camera successor omitted cumulative storage/work or revived an old query/fence receipt");
        gc::GlassCameraGratingTraceBounds cap;
        if(gc::calculateGlassCameraGratingTraceBounds(*cameraBinding,5,renderpassoutput::kMaximumTotalBytes,0,cap,error)
            ||gc::calculateGlassCameraGratingTraceBounds(*cameraBinding,5,0,gc::kMaximumQueryWork,cap,error))
            return fail("OpenGL camera trace bypassed original cumulative caps");
        const auto retained=trace;
        const auto& allowance=cameraBinding->frame.query.texels[std::size_t(cameraBinding->frame.query.control[0])-1];
        const auto perQuery=std::uint64_t(allowance[0])+(std::uint64_t(allowance[1])<<16);
        const auto overWork=(gc::kMaximumQueryWork-cameraBinding->bounds.combinedWork)/(8*gc::kMaximumBranchVisits*(perQuery+1))+1;
        if(overWork<=gc::kMaximumEntrySamples) {
            std::vector<gc::NativeGlassCameraGratingTraceInput> tooMuch(std::size_t(overWork),inputs[0]);
            if(arbitgpu::produceOpenGlGlassCameraGratingTrace(cameraFrame,cameraBinding,tooMuch,.25,trace,error)
                ||!gc::sameGlassCameraGratingTraceReadback(trace,retained))
                return fail("OpenGL camera bounded sample count bypassed worst-case original work admission");
        }
        auto excessive=inputs;excessive.resize(gc::kMaximumEntrySamples+1,inputs[0]);
        if(arbitgpu::produceOpenGlGlassCameraGratingTrace(cameraFrame,cameraBinding,excessive,.25,trace,error)
            ||!gc::sameGlassCameraGratingTraceReadback(trace,retained))return fail("OpenGL camera oversized input changed incumbent output");
        const auto checkVariant=[&](bool wedge,bool obstruction,bool seam) {
            std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame;
            std::shared_ptr<const gc::GratingIncidentBinding> binding;
            if(!cameraTraceStage(wedge,obstruction,seam,frame,binding))return false;
            std::vector<gc::NativeGlassCameraGratingTraceInput> samples(wedge?5:1);
            for(unsigned location=0;location<samples.size();++location)
                if(!gc::makeGlassCameraGratingTraceInput(binding->frame,16,seam?15:16,location,samples[location],error))return false;
            gc::NativeGlassCameraGratingTraceReadback actual;
            if(!arbitgpu::produceOpenGlGlassCameraGratingTrace(frame,binding,samples,.25,actual,error))return false;
            bool tir=false,other=false,gratingTerminal=false,unresolved=false;
            for(const auto& band:actual.bands)for(const auto& v:band.visits) {
                tir=tir||v.tir;other=other||v.disposition==Disposition::Other;
                gratingTerminal=gratingTerminal||v.disposition==Disposition::Grating;
                unresolved=unresolved||v.disposition==Disposition::Unresolved;
            }
            if(wedge) {
                const auto terminal=[](const auto& band){return std::find_if(band.visits.begin(),band.visits.end(),[](const auto& v){return v.disposition==Disposition::Grating;});};
                const auto low=terminal(actual.bands[0]),high=terminal(actual.bands[7]);
                if(low==actual.bands[0].visits.end()||high==actual.bands[7].visits.end()
                    ||std::abs(low->directionFraction[0]-high->directionFraction[0])<=1e-5f
                    ||std::abs(actual.bands[8].nativeDirectionMaximum[0]-actual.bands[0].nativeDirectionMaximum[0])<=1e-4f)
                    return fail("OpenGL native nonparallel wedge lacks spectral angular and corner branch divergence");
            }
            if((wedge&&!tir)||(obstruction&&(!other||gratingTerminal))||(seam&&(!unresolved||gratingTerminal)))
                return fail("OpenGL native camera wedge/TIR, obstruction or exact seam control lost its expected witnesses");
            return true;
        };
        if(!checkVariant(true,false,false)||!checkVariant(false,true,false)||!checkVariant(false,false,true))return false;
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> insideFrame;
        std::shared_ptr<const gc::GratingIncidentBinding> insideBinding;
        if(!cameraTraceStage(false,false,false,insideFrame,insideBinding,0))return false;
        if(arbitgpu::produceOpenGlGlassCameraGratingTrace(insideFrame,insideBinding,inputs,.25,trace,error)
            ||!gc::sameGlassCameraGratingTraceReadback(trace,retained))return fail("OpenGL camera initial inside/unknown medium became an outside trace");
        // Authenticate actual retained uniforms, not only caller copies.
        int previousProgram=0;glGetIntegerv(GL_CURRENT_PROGRAM,&previousProgram);
        auto getUniform=reinterpret_cast<PFNGLGETUNIFORMFVPROC>(glfwGetProcAddress("glGetUniformfv"));
        arbitgl::GlFuncs gl;std::string missing;
        if(!arbitgl::loadGlFunctions(gl,missing))return fail("OpenGL camera fixture lacks native function loading");
        std::shared_ptr<const arbitgpu::OpenGlQueryArithmeticEvidence> current;
        if(!readQueryContract(*cameraFrame,arbitgpu::OpenGlQueryArithmeticRole::GlassCameraGratingTrace,&trace.submission,current,error))return false;
        const auto program=current->program->linkedProgram;
        const int at=gl.GetUniformLocation(program,"uIor[0]");std::array<float,4> original{};
        if(!getUniform||at<0)return fail("OpenGL camera fixture lacks actual native uniform inspection");
        getUniform(program,at,original.data());auto changed=original;changed[0]+=.01f;gl.UseProgram(program);gl.Uniform4fv(at,1,changed.data());
        const bool accepted=owner->validateCameraTraceProductionQuery(trace.submission,trace,error);
        gl.Uniform4fv(at,1,original.data());gl.UseProgram(unsigned(previousProgram));
        if(accepted||!owner->validateCameraTraceProductionQuery(trace.submission,trace,error))
            return fail("OpenGL camera proof ignored actual native IOR uniform mutation or failed restoration");
        std::array<unsigned,5> buffers{};std::uintptr_t fence=0;
        if(!arbitgpu::readOpenGlGlassCameraGratingTraceStorageForFixture(*cameraFrame,buffers,fence,error))return false;
        int generic=0;glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING,&generic);
        for(unsigned index=0;index<4;++index) {
            const auto buffer=buffers[index];
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffer);
            auto* row=static_cast<std::uint32_t*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,16,GL_MAP_READ_BIT|GL_MAP_WRITE_BIT));
            if(!row)return fail("OpenGL camera fixture could not map its actual retained native storage");
            const auto saved=row[0];row[0]^=1u;
            if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE)return fail("OpenGL camera fixture native mutation unmap failed");
            const bool changedAccepted=owner->validateCameraTraceProductionQuery(trace.submission,trace,error);
            gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,buffer);
            row=static_cast<std::uint32_t*>(gl.MapBufferRange(GL_SHADER_STORAGE_BUFFER,0,16,GL_MAP_WRITE_BIT));
            if(!row)return fail("OpenGL camera fixture could not restore retained native storage");
            row[0]=saved;
            if(gl.UnmapBuffer(GL_SHADER_STORAGE_BUFFER)!=GL_TRUE||changedAccepted
                ||!owner->validateCameraTraceProductionQuery(trace.submission,trace,error))
                return fail("OpenGL camera actual input/output/witness buffer mutation acquired authority or failed restoration");
        }
        gl.BindBuffer(GL_SHADER_STORAGE_BUFFER,unsigned(generic));
        glBindTexture(GLenum(0),0); // Actual GL error, no manufactured query row.
        if(arbitgpu::produceOpenGlGlassCameraGratingTrace(cameraFrame,cameraBinding,inputs,.25,trace,error)
            ||!gc::sameGlassCameraGratingTraceReadback(trace,retained))
            return fail("OpenGL camera native program/allocation error replaced its incumbent");
        glGetError();
        if(!owner->validateCameraTraceProductionQuery(trace.submission,trace,error))return false;
        // Corrupt actual production query storage before dispatch. Refusal
        // keeps its worst-case attempt charged and never replaces the incumbent.
        int texture=0,unpackBuffer=0;glGetIntegerv(GL_TEXTURE_BINDING_2D,&texture);glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING,&unpackBuffer);
        constexpr std::array<GLenum,6> unpackNames{GL_UNPACK_ALIGNMENT,GL_UNPACK_ROW_LENGTH,GL_UNPACK_SKIP_PIXELS,GL_UNPACK_SKIP_ROWS,GL_UNPACK_SWAP_BYTES,GL_UNPACK_LSB_FIRST};
        std::array<int,6> unpack{};for(unsigned i=0;i<6;++i)glGetIntegerv(unpackNames[i],&unpack[i]);
        gc::GlassCameraGratingTraceBounds attempts;
        const auto beforeFailed=owner->cameraTraceInputUpload().otherWork;
        if(!gc::calculateGlassCameraGratingTraceBounds(*cameraBinding,inputs.size(),owner->cameraTraceInputUpload().otherOwnedBytes,beforeFailed,attempts,error))return false;
        gl.BindBuffer(GL_PIXEL_UNPACK_BUFFER,0);for(unsigned i=0;i<6;++i)glPixelStorei(unpackNames[i],i?0:1);
        glBindTexture(GL_TEXTURE_2D,buffers[4]);auto corrupted=cameraBinding->frame.query.texels.front();corrupted[0]=std::numeric_limits<float>::quiet_NaN();
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,1,1,GL_RGBA,GL_FLOAT,corrupted.data());
        const bool queryAccepted=arbitgpu::produceOpenGlGlassCameraGratingTrace(cameraFrame,cameraBinding,inputs,.25,trace,error);
        glBindTexture(GL_TEXTURE_2D,buffers[4]);glTexSubImage2D(GL_TEXTURE_2D,0,0,0,1,1,GL_RGBA,GL_FLOAT,cameraBinding->frame.query.texels.front().data());
        glBindTexture(GL_TEXTURE_2D,unsigned(texture));gl.BindBuffer(GL_PIXEL_UNPACK_BUFFER,unsigned(unpackBuffer));
        for(unsigned i=0;i<6;++i)glPixelStorei(unpackNames[i],unpack[i]);
        if(queryAccepted||!gc::sameGlassCameraGratingTraceReadback(trace,retained)
            ||!owner->validateCameraTraceProductionQuery(trace.submission,trace,error)
            ||!arbitgpu::produceOpenGlGlassCameraGratingTrace(cameraFrame,cameraBinding,inputs,.25,trace,error)
            ||owner->cameraTraceInputUpload().otherWork<beforeFailed+2*(attempts.queryWork+attempts.branchWork))
            return fail("OpenGL failed native camera query lost its charge, replaced output or became completed clear zero");
        if(!arbitgpu::readOpenGlGlassCameraGratingTraceStorageForFixture(*cameraFrame,buffers,fence,error))return false;
        gl.DeleteSync(reinterpret_cast<GLsync>(fence));
        if(owner->validateCameraTraceProductionQuery(trace.submission,trace,error)
            ||gc::validateGlassCameraGratingTraceReadback(*cameraBinding,*owner,trace.submission,.25,trace,error))
            return fail("OpenGL camera expired actual native fence retained completion authority");
    }
    if(!gc::prepareGratingEntryCellBinding(dimensionalBinding,sourceCells,error))return false;
    const auto* cellOwner=dynamic_cast<const gc::NativeGratingCellSourceBinding*>(dimensionalFrame.get());
    if(!cellOwner||cellOwner->cellUpload().complete)
        return fail("OpenGL staged geometry fabricated completed source-cell transport");
    gc::NativeReadback cellPoint;gc::EnergyLedger cellEnergy;gc::NativeGratingCellTransportReadback cellWitness;
    if(!arbitgpu::produceOpenGlGlassGratingCellTransportReadback(dimensionalFrame,sourceCells,cellPoint,cellEnergy,cellWitness,error))return false;
    if(!cellWitness.binding||!cellWitness.complete||cellWitness.overflow||!cellWitness.queryWork
        ||cellWitness.binding->pointReadback!=cellPoint.gratingIncidents
        ||cellWitness.binding->sourceCells!=sourceCells||cellWitness.binding->nativeOwner!=dimensionalFrame
        ||cellWitness.records.size()!=cellPoint.gratingIncidents->incidents.size()
        ||cellWitness.submission.submissionSerial<=cellPoint.gratingIncidents->submission.submissionSerial
        ||cellWitness.submission.completedFence!=cellWitness.submission.submissionSerial
        ||cellOwner->cellUpload().pointReadback!=cellPoint.gratingIncidents
        ||cellOwner->cellUpload().cellTexels!=sourceCells->cellTexels
        ||cellPoint.gratingIncidents->queryWork!=dimensionalReceipt.gratingIncidents->queryWork
        ||cellPoint.gratingIncidents->orderWork!=dimensionalReceipt.gratingIncidents->orderWork
        ||cellPoint.spectralRadiance!=dimensionalReceipt.spectralRadiance
        ||cellPoint.gratingCamera||dimensionalFrame->colorImageHandle()||dimensionalFrame->colorTextureDescriptor().complete())
        return fail("OpenGL native cell witness lost its input, supplemental fence, original counters or Color guard");
    {
        const auto oldPoint=cellPoint.gratingIncidents;const auto oldCell=cellWitness.binding;
        const auto oldUpload=cellOwner->cellUpload().bindingGeneration;
        std::array<std::uint64_t,14> before{},after{};
        if(!arbitgpu::readOpenGlNativeAccountingForFixture(*dimensionalFrame,before,error))return false;
        const auto query=gc::gratingCellQueryAllowance(dimensionalBinding->frame);
        const auto interfaces=dimensionalBinding->prepared.branchPolicy->description().maximumInterfaces;
        const auto cellAttempt=dimensionalBinding->bounds.maximumIncidents*gc::kGratingCellTracePaths*(interfaces+1)*query;
        for(unsigned retry=0;retry<2;++retry) {
            if(arbitgpu::produceOpenGlGlassGratingAttemptForFixture(dimensionalFrame,dimensionalBinding,sourceCells,cellPoint,cellEnergy,cellWitness,
                arbitgpu::OpenGlGratingAttemptStopForFixture::CellFence,error)
                ||error.find("after real cell dispatch")==std::string::npos||cellPoint.gratingIncidents!=oldPoint||cellWitness.binding!=oldCell
                ||cellOwner->cellUpload().bindingGeneration!=oldUpload
                ||!arbitgpu::readOpenGlNativeAccountingForFixture(*dimensionalFrame,after,error)
                ||after[5]-before[5]!=(retry+1)*dimensionalBinding->bounds.combinedWork||after[9]-before[9]!=(retry+1)*cellAttempt
                ||after[13]-before[13]!=retry+1)
                return fail("OpenGL actual failed cell-prefix retries changed incumbent output or reused a maximum work allowance");
        }
        const auto charged=after;auto missing=after;missing[5]=before[5];missing[6]=before[6];missing[9]=before[9];
        if(!arbitgpu::exchangeOpenGlNativeAccountingForFixture(*dimensionalFrame,missing,error))return false;
        const bool uploadAccepted=cellOwner->cellUpload().complete;
        const bool producerAccepted=arbitgpu::produceOpenGlGlassGratingCellTransportReadback(dimensionalFrame,sourceCells,cellPoint,cellEnergy,cellWitness,error);
        const bool restored=arbitgpu::exchangeOpenGlNativeAccountingForFixture(*dimensionalFrame,missing,error);
        if(!restored||uploadAccepted||producerAccepted||cellPoint.gratingIncidents!=oldPoint||cellWitness.binding!=oldCell
            ||!arbitgpu::readOpenGlNativeAccountingForFixture(*dimensionalFrame,after,error)||after!=charged)
            return fail("OpenGL cell source inspector or later producer accepted reduced independently retained failed native work");
        if(!arbitgpu::produceOpenGlGlassGratingCellTransportReadback(dimensionalFrame,sourceCells,cellPoint,cellEnergy,cellWitness,error)
            ||!arbitgpu::readOpenGlNativeAccountingForFixture(*dimensionalFrame,after,error)
            ||after[5]-charged[5]!=dimensionalBinding->bounds.combinedWork||after[9]-charged[9]!=cellAttempt)
            return fail("OpenGL valid equal-size cell retry failed cumulative native attempt admission");
    }
    std::array<bool,8> cellBands{};unsigned unsupportedCells=0;std::uint64_t cellQueries=0;
    for(const auto& record:cellWitness.records) {
        const auto& incident=cellPoint.gratingIncidents->incidents[record.incident];
        cellBands[record.band]=true;cellQueries+=record.queryWork;
        if(record.paths[0].size()!=incident.interfaces+1||!record.paths[0].back().hit.work
            ||!gc::sameGratingCellHitOwner(record.paths[0].back().hit,incident.hit))
            return fail("OpenGL supplemental center lost the actual original point endpoint");
        if(record.disposition==gc::GratingCellTransportDisposition::Unsupported)++unsupportedCells;
        for(unsigned path=0;path<4;++path)for(std::size_t depth=0;depth<record.paths[path].size();++depth) {
            const auto& step=record.paths[path][depth];
            if(!step.hit.work||step.depth!=depth||step.token!=(depth<incident.interfaces?incident.ancestry[depth]:0u))
                return fail("OpenGL source-cell replay fabricated query work or changed branch ancestry");
        }
    }
    if(cellQueries!=cellWitness.queryWork||!unsupportedCells
        ||std::find(cellBands.begin(),cellBands.end(),false)!=cellBands.end())
        return fail("OpenGL cell fixture lacks separately charged eight-band boundary failures");
    const auto cellIncumbent=cellWitness.binding;const auto pointIncumbent=cellPoint.gratingIncidents;
    const auto* uploadIncumbent=&cellOwner->cellUpload();
    auto corruptCells=std::make_shared<gc::GratingEntryCellBinding>(*sourceCells);corruptCells->cellTexels[0][0]+=.01f;
    if(arbitgpu::produceOpenGlGlassGratingCellTransportReadback(dimensionalFrame,corruptCells,cellPoint,cellEnergy,cellWitness,error)
        ||cellWitness.binding!=cellIncumbent||cellPoint.gratingIncidents!=pointIncumbent
        ||&cellOwner->cellUpload()!=uploadIncumbent)
        return fail("OpenGL rejected cell input replaced its previous native witness or point candidate");
    if(arbitgpu::produceOpenGlGlassGratingCellTransportReadback(mixedFrame,sourceCells,cellPoint,cellEnergy,cellWitness,error)
        ||cellWitness.binding!=cellIncumbent||cellPoint.gratingIncidents!=pointIncumbent
        ||&cellOwner->cellUpload()!=uploadIncumbent)
        return fail("OpenGL foreign frame replaced its previous completed native cell source");
    const auto refuseCellProof=[&](const auto& edit) {
        auto corrupt=cellWitness;edit(corrupt);
        return !gc::validateGratingCellTransportReadback(*cellWitness.binding,corrupt,dimensionalFrame,cellWitness.submission,error);
    };
    if(!refuseCellProof([](auto& r){r.submission.completedFence++;})
        ||!refuseCellProof([](auto& r){r.records[0].incident++;})
        ||!refuseCellProof([](auto& r){auto& token=r.records[0].paths[0][0].token;token=token==1?65:1;})
        ||!refuseCellProof([](auto& r){r.records[0].paths[0][0].signedBias=0;}))
        return fail("OpenGL cell witness accepted another fence, locator, ancestry or applied interface bias");
    gc::NativeReadback afterCells;gc::EnergyLedger afterCellEnergy;
    if(!arbitgpu::produceOpenGlGlassGratingTerminalReadback(dimensionalFrame,dimensionalBinding,afterCells,afterCellEnergy,error))return false;
    if(afterCells.gratingIncidents->queryWork!=pointIncumbent->queryWork
        ||afterCells.gratingIncidents->orderWork!=pointIncumbent->orderWork
        ||afterCells.spectralRadiance!=cellPoint.spectralRadiance||&cellOwner->cellUpload()!=uploadIncumbent)
        return fail("OpenGL ordinary terminal path acquired supplemental queries or replaced its cell owner");
    // Interior quadrature cells can have matching boundary paths. Even this
    // completed native witness leaves camera Color and spectrum unavailable.
    auto refinedScheduling=scheduling;refinedScheduling.entrySubdivisions=4;
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> refinedFrame;
    std::shared_ptr<const gc::GratingIncidentBinding> refinedBinding;
    if(!arbitgpu::stageOpenGlGlassGratingFrame(dimensionalResources.resources,fixture.ordinaryLight,*fixture.policy,32,32,
        runtime,.25,refinedFrame,refinedBinding,error,refinedScheduling))return false;
    std::shared_ptr<const gc::GratingEntryCellBinding> refinedCells;
    if(!gc::prepareGratingEntryCellBinding(refinedBinding,refinedCells,error))return false;
    gc::NativeReadback refinedPoint;gc::EnergyLedger refinedEnergy;gc::NativeGratingCellTransportReadback refinedWitness;
    if(!arbitgpu::produceOpenGlGlassGratingCellTransportReadback(refinedFrame,refinedCells,refinedPoint,refinedEnergy,refinedWitness,error))return false;
    const auto supportedCell=std::find_if(refinedWitness.records.begin(),refinedWitness.records.end(),[](const auto& record) {
        return record.disposition==gc::GratingCellTransportDisposition::BoundaryWitness;
    });
    std::shared_ptr<const arbitgpu::OpenGlGratingCameraSpectrum> cellSpectrum;
    if(supportedCell==refinedWitness.records.end()||refinedPoint.gratingCamera||refinedFrame->colorImageHandle()
        ||refinedFrame->colorTextureDescriptor().complete()
        ||arbitgpu::readOpenGlGratingCameraSpectrumForFixture(*refinedFrame,cellSpectrum,error))
        return fail("OpenGL supported cell boundaries fabricated camera completion or lack a native supported fixture");
    // Reuse the caller output after dropping every external old-frame owner.
    // Replacing this last strong owner must retire its native frame unlocked.
    std::weak_ptr<const arbitgpu::NativeFixtureSceneFrame> retiredCellFrame=refinedFrame;
    refinedFrame.reset();
    if(retiredCellFrame.use_count()!=1||refinedWitness.binding.use_count()!=1)
        return fail("OpenGL cell lifecycle fixture did not isolate the witness as the sole native frame owner");
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> replacementCellFrame;
    std::shared_ptr<const gc::GratingIncidentBinding> replacementCellBinding;
    if(!arbitgpu::stageOpenGlGlassGratingFrame(dimensionalResources.resources,fixture.ordinaryLight,*fixture.policy,32,32,
        runtime,.25,replacementCellFrame,replacementCellBinding,error,scheduling))return false;
    std::shared_ptr<const gc::GratingEntryCellBinding> replacementCells;
    if(!gc::prepareGratingEntryCellBinding(replacementCellBinding,replacementCells,error))return false;
    const auto* oldCellBinding=refinedWitness.binding.get();
    const auto* oldCellPoint=refinedPoint.gratingIncidents.get();
    auto invalidReplacementCells=std::make_shared<gc::GratingEntryCellBinding>(*replacementCells);
    invalidReplacementCells->cellTexels[0][0]+=.01f;
    if(arbitgpu::produceOpenGlGlassGratingCellTransportReadback(replacementCellFrame,invalidReplacementCells,
        refinedPoint,refinedEnergy,refinedWitness,error)||refinedWitness.binding.get()!=oldCellBinding
        ||refinedPoint.gratingIncidents.get()!=oldCellPoint||retiredCellFrame.use_count()!=1)
        return fail("OpenGL refused cell replacement released its sole previous frame owner");
    const auto* cellBudgetStress=std::getenv("ARBIT_GL_CELL_BUDGET_STRESS");
    std::uint64_t budgetBaselineSerial=0;
    if(cellBudgetStress&&std::string(cellBudgetStress)=="1") {
        std::shared_ptr<const QueryEvidence> budgetQuery;
        // Reserve real near-capacity storage without constructing unused rows.
        // This virtual-memory stress is opt-in, never part of ordinary fixtures.
        {
            gc::NativeReadback budgetBaseline;gc::EnergyLedger budgetEnergy;
            arbitgpu::OpenGlGratingIncidentDiagnostics budgetDiagnostic;
            if(!arbitgpu::produceOpenGlGlassGratingTerminalReadback(replacementCellFrame,replacementCellBinding,
                budgetBaseline,budgetEnergy,error,&budgetDiagnostic))return false;
            budgetBaselineSerial=budgetBaseline.gratingIncidents->submission.submissionSerial;
            if(!readQueryContract(*replacementCellFrame,QueryRole::GratingPoint,&budgetBaseline.gratingIncidents->submission,budgetQuery,error))
                return fail("OpenGL cell budget stress lacks its retained query baseline");
            if(!budgetDiagnostic.nativeBytes||budgetDiagnostic.admittedBytes>renderpassoutput::kMaximumTotalBytes)
                return fail("OpenGL cell budget stress lacks its completed native point baseline");
        }
        const auto retainedRecords=refinedWitness.records.size();const auto retainedEnergy=refinedEnergy;
        const auto* replacementSource=dynamic_cast<const gc::NativeGratingCellSourceBinding*>(replacementCellFrame.get());
        if(!replacementSource||replacementSource->cellUpload().complete)
            return fail("OpenGL cell budget stress started with a published replacement source");
        try {
            refinedWitness.records.reserve(std::size_t(renderpassoutput::kMaximumTotalBytes
                /sizeof(gc::NativeGratingCellTransportRecord)));
        } catch(const std::bad_alloc&) {
            return fail("OpenGL opt-in cell budget stress could not reserve its real near-capacity vector");
        }
        const auto reservedRecords=refinedWitness.records.capacity();
        if(arbitgpu::produceOpenGlGlassGratingCellTransportReadback(replacementCellFrame,replacementCells,
            refinedPoint,refinedEnergy,refinedWitness,error)
            ||error!="OpenGL grating incident storage exceeds the existing whole-frame memory cap"
            ||refinedWitness.binding.get()!=oldCellBinding||refinedPoint.gratingIncidents.get()!=oldCellPoint
            ||refinedWitness.records.size()!=retainedRecords||refinedWitness.records.capacity()!=reservedRecords
            ||refinedEnergy.launched!=retainedEnergy.launched||retiredCellFrame.use_count()!=1
            ||replacementSource->cellUpload().complete)
            return fail("OpenGL oversized cross-frame cell witness bypassed initial point admission or changed outputs");
        std::shared_ptr<const QueryEvidence> refusedQuery;
        if(!readQueryContract(*replacementCellFrame,QueryRole::GratingPoint,&budgetQuery->submission,refusedQuery,error)
            ||refusedQuery!=budgetQuery)return fail("OpenGL original-cap refusal changed a completed native query owner");
        auto compactRecords=refinedWitness.records;
        refinedWitness.records.swap(compactRecords);
    }
    if(!arbitgpu::produceOpenGlGlassGratingCellTransportReadback(replacementCellFrame,replacementCells,
        refinedPoint,refinedEnergy,refinedWitness,error))return false;
    if(budgetBaselineSerial&&(refinedPoint.gratingIncidents->submission.submissionSerial!=budgetBaselineSerial+1
        ||refinedWitness.submission.submissionSerial!=budgetBaselineSerial+2))
        return fail("OpenGL initial cell-cap refusal consumed a new native point or supplemental submission");
    if(!retiredCellFrame.expired()||refinedWitness.binding->nativeOwner!=replacementCellFrame
        ||refinedWitness.binding->pointReadback!=refinedPoint.gratingIncidents
        ||refinedWitness.binding->sourceCells!=replacementCells
        ||refinedPoint.gratingIncidents->queryWork!=dimensionalReceipt.gratingIncidents->queryWork
        ||refinedPoint.gratingIncidents->orderWork!=dimensionalReceipt.gratingIncidents->orderWork
        ||refinedPoint.spectralRadiance!=dimensionalReceipt.spectralRadiance
        ||refinedPoint.gratingCamera||replacementCellFrame->colorImageHandle()
        ||replacementCellFrame->colorTextureDescriptor().complete()
        ||!gc::validateGratingCellTransportReadback(*refinedWitness.binding,refinedWitness,replacementCellFrame,
            refinedWitness.submission,error))
        return fail("OpenGL cell replacement did not retire the old frame or preserve exact native point and Color authority");
    // Closed triangular prism: each visited top/back plane has one triangle.
    // Side walls still require genuine native face-normal bundle separation.
    // Its +Z entry plane exercises a negative denominator for the -Z ray.
    auto intervalScene=std::make_shared<gt::Visual3DScene>(*dimensionalScene);
    intervalScene->vertexCount=intervalScene->indexCount=0;
    const auto setIntervalMesh=[&](std::size_t object,std::initializer_list<HarmonicMIDI::grid::SceneVec3> points,
        std::initializer_list<std::uint32_t> indices) {
        auto& mesh=intervalScene->objects[object];mesh.firstVertex=intervalScene->vertexCount;mesh.vertexCount=points.size();
        mesh.firstIndex=intervalScene->indexCount;mesh.indexCount=indices.size();
        for(const auto position:points) {
            auto vertex=dimensionalScene->vertices[dimensionalScene->objects[object].firstVertex];vertex.position=position;
            if(object==1) {vertex.normal={float(-.3/std::sqrt(1.09)),0,float(1/std::sqrt(1.09))};
                vertex.tangent={0,1,0,1};vertex.uv={position.y/12,-position.x/12};}
            intervalScene->vertices[intervalScene->vertexCount++]=vertex;
        }
        for(auto index:indices)intervalScene->indices[intervalScene->indexCount++]=index;
    };
    setIntervalMesh(0,{{-1,-1,1},{1,-1,1},{0,1,1},{-1,-1,0},{1,-1,0},{0,1,0}},
        {0,1,2,3,5,4,0,3,4,0,4,1,1,4,5,1,5,2,2,5,3,2,3,0});
    setIntervalMesh(1,{{-12,-12,-9.6f},{12,-12,-2.4f},{0,12,-6}},{0,1,2});
    setIntervalMesh(2,{{-4,-1,-8},{-4,1,-8},{-4,1,6},{-4,-1,6}},{0,1,2,0,2,3});
    const auto intervalResources=arbitgpu::prepareOpenGlGlassGratingResources(intervalScene,
        dimensionalBinding->prepared.glass.sourceMaterial,selection,grating);
    if(!intervalResources.prepared)return fail(intervalResources.error.c_str());
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> intervalFrame;
    std::shared_ptr<const gc::GratingIncidentBinding> intervalBinding;
    if(!arbitgpu::stageOpenGlGlassGratingFrame(intervalResources.resources,fixture.ordinaryLight,*fixture.policy,32,32,
        runtime,.25,intervalFrame,intervalBinding,error,refinedScheduling))return false;
    std::shared_ptr<const gc::GratingEntryCellBinding> intervalCells;
    if(!gc::prepareGratingEntryCellBinding(intervalBinding,intervalCells,error))return false;
    gc::NativeReadback intervalBaseline;gc::EnergyLedger intervalBaselineEnergy;
    if(!arbitgpu::produceOpenGlGlassGratingTerminalReadback(intervalFrame,intervalBinding,intervalBaseline,intervalBaselineEnergy,error))return false;
    gc::NativeReadback intervalPoint;gc::EnergyLedger intervalEnergy;arbitgpu::OpenGlGratingCellCoverageReadback intervalResult;
    if(!arbitgpu::produceOpenGlGlassGratingCellCoverageReadback(intervalFrame,intervalCells,intervalPoint,intervalEnergy,intervalResult,error,4))return false;
    if(!intervalResult.candidate||!intervalResult.candidate->boundary)return fail("OpenGL interval query has no native boundary lineage");
    std::shared_ptr<const QueryEvidence> intervalQuery,intervalBoundaryQuery,intervalCoverageQuery;
    if(!readQueryContract(*intervalFrame,QueryRole::GratingPoint,&intervalPoint.gratingIncidents->submission,intervalQuery,error)
        ||!readQueryContract(*intervalFrame,QueryRole::CellBoundary,&intervalResult.candidate->boundary->submission,intervalBoundaryQuery,error)
        ||!readQueryContract(*intervalFrame,QueryRole::CellCoverage,&intervalResult.candidate->submission,intervalCoverageQuery,error)
        ||intervalBoundaryQuery->parents[0]!=intervalQuery||intervalCoverageQuery->parents[0]!=intervalQuery
        ||intervalCoverageQuery->parents[1]!=intervalBoundaryQuery
        ||intervalQuery->program->ownerGeneration==intervalBoundaryQuery->program->ownerGeneration
        ||intervalBoundaryQuery->program->ownerGeneration==intervalCoverageQuery->program->ownerGeneration
        ||arbitgpu::validateOpenGlQueryArithmeticEvidenceForFixture(*dimensionalFrame,intervalQuery,intervalQuery->submission,error))
        return fail("OpenGL interval query contract lost distinct linked owners, exact point/boundary lineage or cross-frame refusal");
    const auto* intervalOwner=dynamic_cast<const gc::NativeGratingCellCoverageSource*>(intervalFrame.get());
    if(!intervalResult.candidate||!intervalResult.payload||!intervalOwner||!intervalOwner->coverageEvidence().complete
        ||intervalResult.candidate->boundary->binding->pointReadback!=intervalPoint.gratingIncidents
        ||intervalResult.candidate->submission.submissionSerial<=intervalResult.candidate->boundary->submission.submissionSerial
        ||intervalResult.candidate->submission.completedFence!=intervalResult.candidate->submission.submissionSerial
        ||intervalPoint.gratingIncidents->queryWork!=intervalBaseline.gratingIncidents->queryWork
        ||intervalPoint.gratingIncidents->orderWork!=intervalBaseline.gratingIncidents->orderWork
        ||intervalPoint.spectralRadiance!=intervalBaseline.spectralRadiance||intervalEnergy.launched!=intervalBaselineEnergy.launched
        ||!gc::validateNativeGratingCellCoverage(intervalResult.candidate,intervalResult.payload,intervalFrame,
            intervalResult.candidate->submission,error))
        return fail("OpenGL interval evidence lost native owners, later fence or original point authority");
    unsigned certifiedCells=0,edgeCells=0;bool faceAxis=false,negativePlane=false;std::array<bool,8> intervalBands{};
    for(const auto& record:intervalResult.payload->records) {
        const auto& point=intervalPoint.gratingIncidents->incidents[record.incident];
        if(record.disposition==gc::GratingCellCoverageDisposition::InteriorCertified) {
            ++certifiedCells;intervalBands[point.band]=true;
            for(const auto& leaf:record.leaves)for(const auto& sweep:leaf.sweeps) {
                if(sweep.blockers.size()!=intervalBinding->frame.triangles.size()-1||!sweep.queryWork||sweep.roundingExpansion<=0)
                    return fail("OpenGL certified interval omitted an actual retained blocker or rounding/work evidence");
                negativePlane=negativePlane||(sweep.entering&&gc::dot(sweep.direction,sweep.normal)<0);
                for(const auto& blocker:sweep.blockers)
                    if(std::count_if(blocker.axis.begin(),blocker.axis.end(),[](float c){return c!=0;})>1)faceAxis=true;
            }
        } else ++edgeCells;
    }
    if(!certifiedCells||!edgeCells||!faceAxis||!negativePlane||std::find(intervalBands.begin(),intervalBands.end(),false)!=intervalBands.end()
        ||intervalResult.payload->disposition!=gc::GratingCellCoverageDisposition::Unsupported
        ||intervalPoint.gratingCamera||intervalFrame->colorImageHandle()||intervalFrame->colorTextureDescriptor().complete())
        return fail("OpenGL interval fixture lacks strict interior/edge evidence or enabled finite Color");
    const auto* intervalCandidate=intervalResult.candidate.get();const auto* intervalPayload=intervalResult.payload.get();
    const auto* intervalPointOwner=intervalPoint.gratingIncidents.get();
    if(arbitgpu::produceOpenGlGlassGratingCellCoverageReadback(intervalFrame,intervalCells,intervalPoint,intervalEnergy,intervalResult,error,257)
        ||intervalResult.candidate.get()!=intervalCandidate||intervalResult.payload.get()!=intervalPayload
        ||intervalPoint.gratingIncidents.get()!=intervalPointOwner||intervalOwner->coverageEvidence().payload.get()!=intervalPayload)
        return fail("OpenGL unbounded interval replacement changed completed private outputs");
    auto invalidIntervalCells=std::make_shared<gc::GratingEntryCellBinding>(*intervalCells);invalidIntervalCells->cellTexels[0][0]+=.01f;
    if(arbitgpu::produceOpenGlGlassGratingCellCoverageReadback(intervalFrame,invalidIntervalCells,intervalPoint,intervalEnergy,intervalResult,error,4)
        ||intervalResult.candidate.get()!=intervalCandidate||intervalResult.payload.get()!=intervalPayload
        ||intervalPoint.gratingIncidents.get()!=intervalPointOwner||intervalOwner->coverageEvidence().payload.get()!=intervalPayload)
        return fail("OpenGL malformed interval input replaced completed native evidence");
    std::shared_ptr<const QueryEvidence> preservedCoverageQuery;
    if(!readQueryContract(*intervalFrame,QueryRole::CellCoverage,&intervalCoverageQuery->submission,preservedCoverageQuery,error)
        ||preservedCoverageQuery!=intervalCoverageQuery)return fail("OpenGL refused query replacement changed its incumbent contract");
    // A new point submission must not retire a point program still consumed by
    // the retained older boundary/coverage records. Metadata alone owns no GL ID.
    gc::NativeReadback contractRepeat;gc::EnergyLedger contractRepeatEnergy;
    if(!arbitgpu::produceOpenGlGlassGratingTerminalReadback(intervalFrame,intervalBinding,contractRepeat,contractRepeatEnergy,error))return false;
    std::shared_ptr<const QueryEvidence> repeatedQuery;
    if(!readQueryContract(*intervalFrame,QueryRole::GratingPoint,&contractRepeat.gratingIncidents->submission,repeatedQuery,error)
        ||repeatedQuery==intervalQuery||repeatedQuery->program->ownerGeneration==intervalQuery->program->ownerGeneration
        ||!arbitgpu::validateOpenGlQueryArithmeticEvidenceForFixture(*intervalFrame,intervalQuery,intervalQuery->submission,error)
        ||!arbitgpu::validateOpenGlQueryArithmeticEvidenceForFixture(*intervalFrame,intervalCoverageQuery,intervalCoverageQuery->submission,error)
        ||contractRepeat.spectralRadiance!=intervalPoint.spectralRadiance||contractRepeatEnergy.launched!=intervalEnergy.launched)
        return fail("OpenGL repeated point query lost fresh native lineage or deleted an older consumed query owner");
    const auto useProgram=reinterpret_cast<PFNGLUSEPROGRAMPROC>(glfwGetProcAddress("glUseProgram"));
    const auto deleteProgram=reinterpret_cast<PFNGLDELETEPROGRAMPROC>(glfwGetProcAddress("glDeleteProgram"));
    const auto isProgram=reinterpret_cast<PFNGLISPROGRAMPROC>(glfwGetProcAddress("glIsProgram"));
    if(!useProgram||!deleteProgram||!isProgram)return fail("OpenGL query lifetime fixture lacks native program entry points");
    GLint previousProgram=0;glGetIntegerv(GL_CURRENT_PROGRAM,&previousProgram);
    useProgram(0);deleteProgram(repeatedQuery->program->linkedProgram);
    if(isProgram(repeatedQuery->program->linkedProgram)
        ||arbitgpu::validateOpenGlQueryArithmeticEvidenceForFixture(*intervalFrame,repeatedQuery,repeatedQuery->submission,error))
        return fail("OpenGL query contract accepted a deleted native program owner");
    if(unsigned(previousProgram)!=repeatedQuery->program->linkedProgram)useProgram(unsigned(previousProgram));
    auto querySentinel=intervalCoverageQuery;
    if(arbitgpu::readOpenGlQueryArithmeticEvidenceForFixture(*intervalFrame,QueryRole::GratingPoint,querySentinel,error)
        ||querySentinel!=intervalCoverageQuery)return fail("OpenGL deleted-owner refusal changed the caller's retained record");
    if(!arbitgpu::produceOpenGlGlassGratingTerminalReadback(intervalFrame,intervalBinding,contractRepeat,contractRepeatEnergy,error))return false;
    std::shared_ptr<const QueryEvidence> recoveredQuery;
    if(!readQueryContract(*intervalFrame,QueryRole::GratingPoint,&contractRepeat.gratingIncidents->submission,recoveredQuery,error)
        ||recoveredQuery->program->ownerGeneration==repeatedQuery->program->ownerGeneration
        ||arbitgpu::validateOpenGlQueryArithmeticEvidenceForFixture(*intervalFrame,repeatedQuery,repeatedQuery->submission,error))
        return fail("OpenGL fresh native query revived a retired program generation");
    auto wrongIntervalFence=intervalResult.candidate->submission;wrongIntervalFence.completedFence++;
    if(gc::validateNativeGratingCellCoverage(intervalResult.candidate,intervalResult.payload,intervalFrame,wrongIntervalFence,error))
        return fail("OpenGL native interval evidence accepted another completed fence");
    auto malformedIntervals=*intervalResult.payload;
    auto clearRecord=std::find_if(malformedIntervals.records.begin(),malformedIntervals.records.end(),[](const auto& r) {
        return r.disposition==gc::GratingCellCoverageDisposition::InteriorCertified;
    });
    clearRecord->leaves[0].sweeps[0].blockers.clear();
    if(gc::validateGratingCellCoverageShape(*intervalResult.candidate,malformedIntervals,error))
        return fail("OpenGL interval proof accepted a missing whole-domain blocker");
    if(cellBudgetStress&&std::string(cellBudgetStress)=="1") {
        // Real string reserve is opt-in under the existing native cap stress.
        // Check each accepted candidate string independently, without changing
        // its identity bytes or native frame metrics.
        const auto* intervalSource=dynamic_cast<const gc::NativeGratingCellSourceBinding*>(intervalFrame.get());
        if(!intervalSource)return fail("OpenGL interval string stress has no exact native source owner");
        const auto* retainedUpload=&intervalSource->cellUpload();
        for(unsigned field=0;field<2;++field) {
            gc::NativeReadback stressBaseline;gc::EnergyLedger stressEnergy;
            if(!arbitgpu::produceOpenGlGlassGratingTerminalReadback(intervalFrame,intervalBinding,stressBaseline,stressEnergy,error))return false;
            const auto serial=stressBaseline.gratingIncidents->submission.submissionSerial;
            auto enlarged=std::make_shared<gc::GratingCellCoverageCandidate>(*intervalResult.candidate);
            auto& retained=field==0?enlarged->identity:enlarged->submission.frameIdentity;
            try {retained.reserve(std::size_t(renderpassoutput::kMaximumTotalBytes));}
            catch(const std::bad_alloc&) {return fail("OpenGL opt-in interval string stress could not reserve real capacity");}
            if(!gc::validateGratingCellCoverageCandidate(*enlarged,intervalFrame,enlarged->submission,error))return false;
            arbitgpu::OpenGlGratingCellCoverageReadback stressOutput{enlarged,intervalResult.payload};
            const auto retainedEnergy=intervalEnergy;
            if(arbitgpu::produceOpenGlGlassGratingCellCoverageReadback(intervalFrame,intervalCells,
                intervalPoint,intervalEnergy,stressOutput,error,4)
                ||error!="OpenGL grating incident storage exceeds the existing whole-frame memory cap"
                ||stressOutput.candidate!=enlarged||stressOutput.payload.get()!=intervalPayload
                ||intervalPoint.gratingIncidents.get()!=intervalPointOwner||intervalEnergy.launched!=retainedEnergy.launched
                ||&intervalSource->cellUpload()!=retainedUpload||intervalOwner->coverageEvidence().payload.get()!=intervalPayload
                ||intervalOwner->coverageEvidence().candidate.lock().get()!=intervalCandidate)
                return fail("OpenGL retained interval string capacity bypassed initial admission or changed native evidence");
            stressOutput={};enlarged.reset();
            if(!arbitgpu::produceOpenGlGlassGratingTerminalReadback(intervalFrame,intervalBinding,stressBaseline,stressEnergy,error))return false;
            if(stressBaseline.gratingIncidents->submission.submissionSerial!=serial+1)
                return fail("OpenGL interval string-cap refusal consumed a native point submission");
        }
    }
    // A real interior triangle lies between boundary samples and must deny the
    // affected leaf, without relabeling point power or treating samples as proof.
    auto occludedIntervalScene=std::make_shared<gt::Visual3DScene>(*intervalScene);
    auto occludedPrograms=std::make_shared<gt::Program>(*dimensionalBinding->prepared.glass.sourceMaterial);
    const auto blockerObject=occludedIntervalScene->objectCount++,blockerMaterial=occludedIntervalScene->materialCount++;
    auto& interiorBlocker=occludedIntervalScene->objects[blockerObject];interiorBlocker={};interiorBlocker.id={901};interiorBlocker.material={902};
    interiorBlocker.firstVertex=occludedIntervalScene->vertexCount;interiorBlocker.vertexCount=3;
    interiorBlocker.firstIndex=occludedIntervalScene->indexCount;interiorBlocker.indexCount=3;
    const HarmonicMIDI::grid::SceneVec3 blockerPoints[]={{.0525f,-.4475f,-3},{.0725f,-.4475f,-3},{.0625f,-.4275f,-3}};
    for(unsigned v=0;v<3;++v) {
        auto vertex=occludedIntervalScene->vertices[intervalScene->objects[1].firstVertex];vertex.position=blockerPoints[v];vertex.normal={0,0,1};
        occludedIntervalScene->vertices[occludedIntervalScene->vertexCount++]=vertex;
        occludedIntervalScene->indices[occludedIntervalScene->indexCount++]=v;
    }
    auto& blockerSurface=occludedIntervalScene->materials[blockerMaterial];blockerSurface={};blockerSurface.id=interiorBlocker.material;
    blockerSurface.opacity=1;blockerSurface.roughness=1;
    auto blockerProgram=std::make_shared<gt::Program>(*occludedPrograms->objectPrograms.back());
    blockerProgram->object=interiorBlocker.id;blockerProgram->authoredMaterial=interiorBlocker.material;
    blockerProgram->parameters.identifiers[0]=interiorBlocker.material.value;
    blockerProgram->bindingDigest=blockerProgram->programIdentity="gl-interval-interior-blocker";
    blockerProgram->spectralReflectance->coefficients.fill(0);occludedPrograms->objectPrograms.push_back(blockerProgram);
    const auto blockedIntervalResources=arbitgpu::prepareOpenGlGlassGratingResources(occludedIntervalScene,occludedPrograms,selection,grating);
    if(!blockedIntervalResources.prepared)return fail(blockedIntervalResources.error.c_str());
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> blockedIntervalFrame;
    std::shared_ptr<const gc::GratingIncidentBinding> blockedIntervalBinding;
    if(!arbitgpu::stageOpenGlGlassGratingFrame(blockedIntervalResources.resources,fixture.ordinaryLight,*fixture.policy,32,32,
        runtime,.25,blockedIntervalFrame,blockedIntervalBinding,error,refinedScheduling))return false;
    std::shared_ptr<const gc::GratingEntryCellBinding> blockedIntervalCells;
    if(!gc::prepareGratingEntryCellBinding(blockedIntervalBinding,blockedIntervalCells,error))return false;
    gc::NativeReadback blockedIntervalPoint;gc::EnergyLedger blockedIntervalEnergy;arbitgpu::OpenGlGratingCellCoverageReadback blockedIntervals;
    if(!arbitgpu::produceOpenGlGlassGratingCellCoverageReadback(blockedIntervalFrame,blockedIntervalCells,
        blockedIntervalPoint,blockedIntervalEnergy,blockedIntervals,error,4))return false;
    bool blockedInterior=false;
    for(const auto& record:intervalResult.payload->records)if(record.disposition==gc::GratingCellCoverageDisposition::InteriorCertified) {
        const auto& point=intervalPoint.gratingIncidents->incidents[record.incident];
        for(const auto& other:blockedIntervals.payload->records) {
            const auto& blockedPoint=blockedIntervalPoint.gratingIncidents->incidents[other.incident];
            if(point.work==blockedPoint.work&&point.band==blockedPoint.band&&point.ancestry==blockedPoint.ancestry
                &&other.disposition==gc::GratingCellCoverageDisposition::Unsupported
                &&blockedIntervals.candidate->boundary->records[other.incident].disposition==gc::GratingCellTransportDisposition::BoundaryWitness)
                blockedInterior=true;
        }
    }
    if(!blockedInterior||blockedIntervalPoint.gratingCamera||blockedIntervalFrame->colorTextureDescriptor().complete())
        return fail("OpenGL native interval scan missed an interior blocker between matching boundary samples");
    {
        // Same backend and query, in a genuinely distinct context lifetime.
        // Keep the completed foreign frame alive through its teardown hook.
        auto* originalContext=glfwGetCurrentContext();
        glfwWindowHint(GLFW_VISIBLE,GLFW_FALSE);glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
        auto* foreignContext=glfwCreateWindow(16,16,"query-contract-lifetime",nullptr,nullptr);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
        if(!foreignContext)return fail("OpenGL query context-lifetime fixture could not create a real GL4.3 context");
        int token=0;const auto restore=[&](int*) {
            if(foreignContext) {glfwMakeContextCurrent(foreignContext);backend.releaseCachedProgramsForCurrentContext();glfwDestroyWindow(foreignContext);}
            glfwMakeContextCurrent(originalContext);
        };
        const std::unique_ptr<int,decltype(restore)> contextGuard(&token,restore);
        glfwMakeContextCurrent(foreignContext);
        if(arbitgpu::validateOpenGlQueryArithmeticEvidenceForFixture(*intervalFrame,intervalQuery,intervalQuery->submission,error))
            return fail("OpenGL query contract accepted another current context");
        auto foreignResources=arbitgpu::prepareOpenGlGlassGratingResources(intervalScene,
            dimensionalBinding->prepared.glass.sourceMaterial,selection,grating);
        if(!foreignResources.prepared)return fail(foreignResources.error.c_str());
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> foreignFrame;
        std::shared_ptr<const gc::GratingIncidentBinding> foreignBinding;
        if(!arbitgpu::stageOpenGlGlassGratingFrame(foreignResources.resources,fixture.ordinaryLight,*fixture.policy,32,32,
            runtime,.25,foreignFrame,foreignBinding,error,refinedScheduling))return false;
        gc::NativeReadback foreignReceipt;gc::EnergyLedger foreignEnergy;
        if(!arbitgpu::produceOpenGlGlassGratingTerminalReadback(foreignFrame,foreignBinding,foreignReceipt,foreignEnergy,error))return false;
        std::shared_ptr<const QueryEvidence> foreignQuery;
        if(!readQueryContract(*foreignFrame,QueryRole::GratingPoint,&foreignReceipt.gratingIncidents->submission,foreignQuery,error)
            ||foreignQuery->program->contextGeneration==intervalQuery->program->contextGeneration
            ||arbitgpu::validateOpenGlQueryArithmeticEvidenceForFixture(*foreignFrame,intervalQuery,intervalQuery->submission,error))
            return fail("OpenGL context query identity accepted another frame or reused its context epoch");
        backend.releaseCachedProgramsForCurrentContext();
        if(arbitgpu::validateOpenGlQueryArithmeticEvidenceForFixture(*foreignFrame,foreignQuery,foreignQuery->submission,error)
            ||isProgram(foreignQuery->program->linkedProgram))return fail("OpenGL teardown hook left an expired linked query usable");
        glfwDestroyWindow(foreignContext);foreignContext=nullptr;glfwMakeContextCurrent(originalContext);
        if(arbitgpu::validateOpenGlQueryArithmeticEvidenceForFixture(*foreignFrame,foreignQuery,foreignQuery->submission,error))
            return fail("OpenGL restored context revived a foreign retired query");
        // These final native lease releases occur after the foreign window died.
        foreignFrame.reset();foreignResources.resources.reset();
        if(!arbitgpu::validateOpenGlQueryArithmeticEvidenceForFixture(*intervalFrame,intervalCoverageQuery,intervalCoverageQuery->submission,error))
            return fail("OpenGL foreign teardown deleted the retained original query owner");
    }
    // Coverage owns its boundary and frame through the caller only. Replacing
    // its last old candidate must retire that frame after the mutex unlocks.
    std::weak_ptr<const arbitgpu::NativeFixtureSceneFrame> intervalFrameLifetime=intervalFrame;
    std::weak_ptr<const gc::GratingCellCoverageCandidate> intervalCandidateLifetime=intervalResult.candidate;
    intervalFrame.reset();
    if(intervalFrameLifetime.use_count()!=1||intervalResult.candidate.use_count()!=1)
        return fail("OpenGL interval frame retained a strong coverage cycle or extra candidate owner");
    if(!arbitgpu::produceOpenGlGlassGratingCellCoverageReadback(blockedIntervalFrame,blockedIntervalCells,
        intervalPoint,intervalEnergy,intervalResult,error,4))return false;
    if(isProgram(intervalCoverageQuery->program->linkedProgram)||isProgram(intervalBoundaryQuery->program->linkedProgram)
        ||isProgram(intervalQuery->program->linkedProgram))
        return fail("OpenGL metadata retained native query programs after their last frame lease retired");
    if(!intervalFrameLifetime.expired()||!intervalCandidateLifetime.expired()
        ||intervalResult.candidate->boundary->binding->nativeOwner!=blockedIntervalFrame)
        return fail("OpenGL interval replacement failed to retire its last previous native owner");
    gc::NativeReadback cameraReceipt;gc::EnergyLedger cameraEnergy;arbitgpu::OpenGlGratingCameraDiagnostics cameraDiagnostic;
    std::shared_ptr<const arbitgpu::OpenGlGratingCameraSpectrum> unavailableSpectrum;
    if(arbitgpu::readOpenGlGratingCameraSpectrumForFixture(*dimensionalFrame,unavailableSpectrum,error))
        return fail("OpenGL staged geometry fabricated a private grating spectrum candidate");
    if(!arbitgpu::produceOpenGlGlassGratingCameraReadback(dimensionalFrame,dimensionalBinding,cameraReceipt,cameraEnergy,error,&cameraDiagnostic))return false;
    std::shared_ptr<const QueryEvidence> cameraQuery,cameraPointQuery;
    if(!readQueryContract(*dimensionalFrame,QueryRole::GratingPoint,&cameraReceipt.gratingIncidents->submission,cameraPointQuery,error)
        ||!readQueryContract(*dimensionalFrame,QueryRole::GratingCamera,&cameraReceipt.gratingCamera->submission,cameraQuery,error)
        ||cameraQuery->parents[0]!=cameraPointQuery||cameraQuery->program->stageCount!=1)
        return fail("OpenGL private camera query lost its temporary native program or exact point lineage");
    if(!cameraReceipt.gratingCamera||cameraReceipt.gratingCamera->physicalColor
        ||cameraReceipt.gratingCamera->incidents!=cameraReceipt.gratingIncidents
        ||!(cameraReceipt.gratingCamera->submission==cameraReceipt.gratingIncidents->submission)
        ||!cameraDiagnostic.geometryDrawComplete||!cameraDiagnostic.receiverSpectrumResolveComplete
        ||cameraReceipt.spectralRadiance!=dimensionalReceipt.spectralRadiance
        ||cameraReceipt.gratingCamera->orders.size()!=cameraReceipt.gratingIncidents->children.size()
        ||dimensionalFrame->colorImageHandle()||dimensionalFrame->depthImageHandle())
        return fail("OpenGL private camera primitives lost native owners, receiver spectrum or Color guard");
    unsigned outside=0;
    for(const auto& order:cameraReceipt.gratingCamera->orders)
        if(order.outcome==gc::NativeGratingCameraReadback::Outcome::OutsideCamera) {
            ++outside;
            if(!order.complete||order.x!=UINT32_MAX||order.y!=UINT32_MAX||order.cameraHit.status!=-1
                ||order.cameraHit.work||order.raster.object||order.spectralRadiance!=0)
                return fail("OpenGL native outside-camera order contains fabricated visibility or radiance");
        }
    if(!outside)return fail("OpenGL camera fixture did not exercise angular outside support");
    const auto checkSpectrum=[&](const arbitgpu::NativeFixtureSceneFrame& owner,const gc::NativeReadback& native,
        const arbitgpu::OpenGlGratingCameraDiagnostics& measured,bool positive,
        std::shared_ptr<const arbitgpu::OpenGlGratingCameraSpectrum>& spectrum) {
        if(!measured.gratingSpectrumComplete||!arbitgpu::readOpenGlGratingCameraSpectrumForFixture(owner,spectrum,error)
            ||!spectrum||!spectrum->complete||spectrum->camera!=native.gratingCamera
            ||spectrum->spectralRadiance.size()!=32*32*8||spectrum->resolvedColor.size()!=32*32
            ||spectrum->camera->physicalColor||owner.colorTextureDescriptor().complete())return false;
        std::vector<float> expected(spectrum->spectralRadiance.size(),0);bool foundPositive=false;std::size_t visible=0;
        for(const auto& order:native.gratingCamera->orders)
            if(order.outcome==gc::NativeGratingCameraReadback::Outcome::Visible&&order.spectralRadiance>0) {
                expected[(std::size_t(order.y)*32+order.x)*8+order.band]+=order.spectralRadiance;
                foundPositive=true;++visible;
            }
        for(std::size_t i=0;i<expected.size();++i)
            if(!gc::nearPower(expected[i],spectrum->spectralRadiance[i],4e-6))return false;
        std::size_t keys=0;
        for(std::size_t i=0;i<spectrum->keys.size();++i) {
            if(i&&spectrum->keys[i]<spectrum->keys[i-1])return false;
            if(!gc::depositionSentinel(spectrum->keys[i]))++keys;
        }
        for(const auto& pixel:spectrum->resolvedColor)
            if(pixel[3]!=1||!std::all_of(pixel.begin(),pixel.end(),[](float value){return std::isfinite(value);}))return false;
        return keys==visible&&(!positive||foundPositive);
    };
    std::shared_ptr<const arbitgpu::OpenGlGratingCameraSpectrum> outsideSpectrum;
    if(!checkSpectrum(*dimensionalFrame,cameraReceipt,cameraDiagnostic,false,outsideSpectrum))
        return fail("OpenGL outside-camera native spectrum candidate lacks completed deterministic zero support");
    const auto cameraIncumbent=cameraReceipt.gratingCamera;
    auto wrongCameraBinding=std::make_shared<gc::GratingIncidentBinding>(*dimensionalBinding);
    wrongCameraBinding->evaluationTime+=1;
    if(arbitgpu::produceOpenGlGlassGratingCameraReadback(dimensionalFrame,wrongCameraBinding,cameraReceipt,cameraEnergy,error)
        ||cameraReceipt.gratingCamera!=cameraIncumbent)
        return fail("OpenGL private camera accepted a mismatched retained generation");
    std::shared_ptr<const arbitgpu::OpenGlGratingCameraSpectrum> preservedSpectrum;
    if(!arbitgpu::readOpenGlGratingCameraSpectrumForFixture(*dimensionalFrame,preservedSpectrum,error)
        ||preservedSpectrum!=outsideSpectrum)
        return fail("OpenGL failed private camera request replaced the previous native spectrum candidate");
    auto corruptCamera=cameraReceipt;auto cameraProof=std::make_shared<gc::NativeGratingCameraReadback>(*cameraIncumbent);
    cameraProof->submission.completedFence++;corruptCamera.gratingCamera=cameraProof;
    if(gc::validateGratingCameraReadback(*dimensionalBinding,corruptCamera,cameraReceipt.gratingIncidents->submission,error,sourceBinding))
        return fail("OpenGL private camera accepted a foreign completed fence");
    cameraProof=std::make_shared<gc::NativeGratingCameraReadback>(*cameraIncumbent);
    cameraProof->physicalColor=true;corruptCamera.gratingCamera=cameraProof;
    if(gc::validateGratingCameraReadback(*dimensionalBinding,corruptCamera,cameraReceipt.gratingIncidents->submission,error,sourceBinding))
        return fail("OpenGL local camera derivative witness granted finite-cell Color");
    const auto cameraCase=[&](bool occluded) {
        error.clear();
        auto cameraRuntime=runtime;auto active=dimensionalBinding->frame.camera;
        active.transform.rotation={0,0,0,1};active.verticalFovRadians=2.0f;
        if(occluded)active.transform.translation={-.5f,0,.8f};
        else {
            const auto& event=dimensionalReceipt.gratingIncidents->incidents.front();
            const auto zero=std::find_if(event.orders.begin(),event.orders.begin()+event.orderCount,[](const auto& order){return order.order==0;});
            if(zero==event.orders.begin()+event.orderCount)return false;
            active.transform.translation={event.hit.position[0]+2*zero->direction[0],
                event.hit.position[1]+2*zero->direction[1],event.hit.position[2]+2*zero->direction[2]};
        }
        cameraRuntime.cameraOverride=active;
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> caseFrame;
        std::shared_ptr<const gc::GratingIncidentBinding> caseBinding;
        if(!arbitgpu::stageOpenGlGlassGratingFrame(dimensionalResources.resources,fixture.ordinaryLight,*fixture.policy,32,32,
            cameraRuntime,.25,caseFrame,caseBinding,error,scheduling))return false;
        gc::NativeReadback result;gc::EnergyLedger resultEnergy;arbitgpu::OpenGlGratingCameraDiagnostics measured;
        if(!arbitgpu::produceOpenGlGlassGratingCameraReadback(caseFrame,caseBinding,result,resultEnergy,error,&measured))return false;
        const auto expected=occluded?gc::NativeGratingCameraReadback::Outcome::Occluded:gc::NativeGratingCameraReadback::Outcome::Visible;
        const auto found=std::find_if(result.gratingCamera->orders.begin(),result.gratingCamera->orders.end(),
            [&](const auto& order){return order.complete&&order.outcome==expected;});
        if(found==result.gratingCamera->orders.end()||!found->cameraHit.work||!found->raster.object
            ||found->raster.object!=found->cameraHit.objectId||found->raster.material!=found->cameraHit.materialId
            ||(occluded&&found->spectralRadiance!=0)||(!occluded&&found->spectralRadiance<=0)
            ||caseFrame->colorTextureDescriptor().complete()||result.gratingCamera->physicalColor)return false;
        std::shared_ptr<const arbitgpu::OpenGlGratingCameraSpectrum> firstSpectrum;
        if(!checkSpectrum(*caseFrame,result,measured,!occluded,firstSpectrum))return false;
        gc::NativeReadback repeatedCamera;gc::EnergyLedger repeatedEnergy;arbitgpu::OpenGlGratingCameraDiagnostics repeatedDiagnostic;
        if(!arbitgpu::produceOpenGlGlassGratingCameraReadback(caseFrame,caseBinding,repeatedCamera,repeatedEnergy,error,&repeatedDiagnostic))return false;
        std::shared_ptr<const arbitgpu::OpenGlGratingCameraSpectrum> secondSpectrum;
        if(!checkSpectrum(*caseFrame,repeatedCamera,repeatedDiagnostic,!occluded,secondSpectrum)
            ||secondSpectrum==firstSpectrum||secondSpectrum->spectralRadiance!=firstSpectrum->spectralRadiance
            ||secondSpectrum->resolvedColor!=firstSpectrum->resolvedColor||secondSpectrum->keys!=firstSpectrum->keys)return false;
        auto stale=std::make_shared<gc::GratingIncidentBinding>(*caseBinding);stale->identity+="foreign";
        if(arbitgpu::produceOpenGlGlassGratingCameraReadback(caseFrame,stale,repeatedCamera,repeatedEnergy,error))return false;
        std::shared_ptr<const arbitgpu::OpenGlGratingCameraSpectrum> retainedSpectrum;
        if(!arbitgpu::readOpenGlGratingCameraSpectrumForFixture(*caseFrame,retainedSpectrum,error)||retainedSpectrum!=secondSpectrum)return false;
        return measured.queryWork>0&&measured.geometryDrawComplete&&measured.receiverSpectrumResolveComplete;
    };
    if(!cameraCase(false)) {if(error.empty())error="OpenGL visible camera primitive lacks actual native ray/raster support";return false;}
    if(!cameraCase(true)) {if(error.empty())error="OpenGL occluded camera primitive lacks actual nearer native ray/raster support";return false;}
    auto changedSource=std::make_shared<gc::GratingIncidentBinding>(*dimensionalBinding);
    changedSource->frame.sourceDirectionalIrradiance.coefficients[0]+=.5f;
    const auto previousDimensional=dimensionalReceipt.directionalSource;
    if(arbitgpu::produceOpenGlGlassGratingTerminalReadback(dimensionalFrame,changedSource,dimensionalReceipt,dimensionalEnergy,error)
        ||dimensionalReceipt.directionalSource!=previousDimensional)
        return fail("OpenGL foreign directional coefficient replaced an incumbent native receipt");
    auto wrongBinding=std::make_shared<gc::GratingIncidentBinding>(*mixedBinding);wrongBinding->evaluationTime+=1;
    const auto incumbent=mixedReceipt.spectralRadiance;
    if(arbitgpu::produceOpenGlGlassGratingTerminalReadback(mixedFrame,wrongBinding,mixedReceipt,mixedEnergy,error)
        ||mixedReceipt.spectralRadiance!=incumbent)return fail("OpenGL mixed terminal accepted a foreign staging binding");
    auto badLeaf=std::make_shared<gt::Program>(*grating);badLeaf->materialRevision++;
    if(arbitgpu::prepareOpenGlGlassGratingResources(mixedBinding->prepared.glass.sourceScene,mixedBinding->prepared.glass.sourceMaterial,selection,badLeaf).prepared)
        return fail("OpenGL private mixed preparation accepted a foreign compiled leaf revision");
    const auto originalFrame=mixedFrame;const auto originalBinding=mixedBinding;
    if(arbitgpu::stageOpenGlGlassGratingFrame(mixedResources.resources,{},*fixture.policy,32,32,runtime,.25,
        mixedFrame,mixedBinding,error,scheduling)||mixedFrame!=originalFrame||mixedBinding!=originalBinding)
        return fail("OpenGL failed private staging replaced incumbent native ownership");
    auto foreignRuntime=runtime;foreignRuntime.materialFrameTexture=staging.frame;
    if(arbitgpu::stageOpenGlGlassGratingFrame(mixedResources.resources,fixture.ordinaryLight,*fixture.policy,32,32,
        foreignRuntime,.25,mixedFrame,mixedBinding,error,scheduling)||mixedFrame!=originalFrame||mixedBinding!=originalBinding)
        return fail("OpenGL private camera preparation accepted an unrelated retained image lease");
    if(arbitgpu::stageOpenGlGlassGratingFrame(mixedResources.resources,fixture.ordinaryLight,*fixture.policy,
        arbitgpu::kMaximumNativeFixtureExtent+1,32,runtime,.25,mixedFrame,mixedBinding,error,scheduling)
        ||mixedFrame!=originalFrame||mixedBinding!=originalBinding)
        return fail("OpenGL private camera storage overflow replaced its incumbent owner");

    auto highDescription=description;highDescription.microstructure.grooveDepthNanometres=125;
    const auto high=diffractionmaterial::admit(highDescription,error);if(!high)return false;
    const auto highMaterial=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*high);
    const auto highGrating=compileGrating(highDescription,highMaterial->structuralDigest());if(!highGrating)return false;
    std::shared_ptr<const gc::GratingIncidentBinding> highBinding;
    if(!gc::prepareGratingIncidentBinding(prepared,fixture.frame,highGrating,highMaterial,.25,highBinding,error))return false;
    const auto previous=receipt.spectralRadiance;
    if(arbitgpu::produceOpenGlGlassGratingTerminalReadback(staging.frame,highBinding,receipt,ledger,error)||error.empty()
        ||receipt.spectralRadiance!=previous)return fail("OpenGL high-tail terminal continuation published a partial readback");
    error.clear();return true;
}

bool glassGratingIncidentNativeChecks(arbitgpu::NativeFixtureSceneBackend& backend)
{
    namespace gt=glasscaustictests;namespace gc=arbitgpu::glasscaustics;
    int major=0,minor=0;glGetIntegerv(GL_MAJOR_VERSION,&major);glGetIntegerv(GL_MINOR_VERSION,&minor);
    if(major<4||(major==4&&minor<3)) {
        std::cout<<"UNVERIFIED: native glass/grating incident fixtures require OpenGL 4.3\n";return true;
    }
    bool ok=true;std::string error;
    const auto near=[](double a,double b,double tolerance=8e-4) {
        return std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=tolerance*std::max({std::abs(a),std::abs(b),1e-14});
    };
    const auto staging=[](float lightScale=1) {
        auto fixture=gt::ordinaryFixture();fixture.scene->lights[0].transform.rotation={0,0,0,1};
        fixture.scene->cameras[0].transform.translation={0,0,-2};
        for(auto& band:fixture.scene->lights[0].spectralSource.radiance)band*=lightScale;
        const auto target=fixture.scene->objects[1];
        // Real off-camera receivers make the queried grating instance ordinal9.
        fixture.scene->objects[9]=target;fixture.scene->objectCount=10;
        for(unsigned instance=1;instance<9;++instance) {
            auto& far=fixture.scene->objects[instance];far=target;far.id={900+instance};
            far.transform.translation={100+20*float(instance),0,0};
            auto owner=std::make_shared<gt::Program>(*fixture.program->objectPrograms[0]);owner->object=far.id;
            owner->programIdentity="native-grating-far-receiver-"+std::to_string(instance);
            owner->bindingDigest=owner->programIdentity;fixture.program->objectPrograms.push_back(owner);
        }
        for(std::size_t v=0;v<target.vertexCount;++v) {
            auto& vertex=fixture.scene->vertices[target.firstVertex+v];vertex.uv={vertex.position.x/12,vertex.position.y/12};
        }
        return fixture;
    };
    const auto bind=[&](const gt::Fixture& fixture,float depth,float period,
        std::shared_ptr<const gc::GratingIncidentBinding>& output) {
        auto description=diffractionmaterial::makeAluminiumBinaryGratingPreset();
        description.microstructure.grooveDepthNanometres=depth;description.geometry.grooveSpacingNanometres=period;
        description.spectrum.firstOrder=description.spectrum.lastOrder=1;description.roughness={0,0};
        const auto admitted=diffractionmaterial::admit(description,error);if(!admitted)return false;
        const auto material=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*admitted);
        auto owner=std::make_shared<gt::Program>(*fixture.program->objectPrograms[0]);
        owner->kind=arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;owner->spectralReflectance.reset();
        owner->diffractionPathCount=1;owner->bindingDigest="native-gl-grating-binding";owner->programIdentity="native-gl-grating-program";
        owner->diffractionPaths[0].material=diffractionmaterial::physicalcheckpoint::makeGpuParameters(*material,{});
        return gc::prepareGratingIncidentBinding(fixture.prepared,fixture.frame,owner,material,.25,output,error);
    };
    auto fixture=staging();if(!fixture.prepare(error))return expect(false,error.c_str());
    ok&=expect(fixture.prepared.glass.sourceScene!=fixture.scene&&fixture.prepared.glass.sourceMaterial!=fixture.program,
        "positive native grating fixture retains independently copied scene and material snapshots");
    const auto preparation=backend.prepare(fixture.scene,fixture.program);
    if(!preparation.prepared)return expect(false,preparation.error.c_str());
    arbitgpu::NativeFixtureSceneRuntimeInputs runtime;runtime.linearColor=true;
    const auto native=arbitgpu::renderOpenGlGlassCausticScene(fixture.scene,preparation.resources,fixture.ordinaryLight,
        fixture.program->object,fixture.program->authoredMaterial,*fixture.policy,{gc::kScheduleVersion,4},32,32,runtime);
    if(!native.rendered)return expect(false,native.error.c_str());
    std::shared_ptr<const gc::GratingIncidentBinding> binding;
    if(!bind(fixture,.9f,1000,binding))return expect(false,error.c_str());
    arbitgpu::NativeRawPassPixels before,after;
    if(!native.frame->readRawPass(renderpassoutput::Output::Color,before,error))return expect(false,error.c_str());
    gc::NativeGratingReadback receipt;arbitgpu::OpenGlGratingIncidentDiagnostics diagnostic;
    if(!arbitgpu::produceOpenGlGlassGratingIncidents(native.frame,binding,receipt,error,&diagnostic))return expect(false,error.c_str());
    ok&=expect(receipt.complete&&!receipt.overflow&&!receipt.incidents.empty(),"actual native smooth glass rays produce distinct grating incidents");
    if(receipt.incidents.empty())return false;
    ok&=expect(receipt.submission.frameIdentity==binding->identity&&receipt.submission.completedFence==receipt.submission.submissionSerial
        &&diagnostic.admittedBytes<=renderpassoutput::kMaximumTotalBytes&&diagnostic.nativeBytes>0
        &&diagnostic.baseQueryWork>0&&receipt.queryWork>0&&receipt.orderWork==receipt.incidents.size()*3,
        "native grating producer retains exact completed ownership and measured bounded work");
    const auto triangles=gt::worldTriangles(*fixture.scene);
    std::array<unsigned,8> bands{};bool firstOrders=false,dispersed=false;
    std::array<gc::Point,8> bandDirection{};
    for(const auto& incident:receipt.incidents) {
        ++bands[incident.band];bandDirection[incident.band]=incident.direction;
        ok&=expect(incident.hit.instanceId==9,"real native grating query retains an instance ordinal above eight");
        const auto& work=fixture.frame.work[incident.work];const auto& entry=fixture.frame.entries[work.entry];
        const auto oracle=gt::branchOracle(fixture,entry,incident.band);
        const auto arrival=std::find_if(oracle.arrivals.begin(),oracle.arrivals.end(),[&](const auto& a) {
            return a.history==incident.reflectionHistory&&a.interfaces==incident.interfaces;
        });
        ok&=expect(oracle.valid&&arrival!=oracle.arrivals.end(),"native parent ancestry agrees with independent binary64 glass branches");
        if(arrival!=oracle.arrivals.end()) {
            const auto delta=gt::minus(gt::doublePoint(incident.hit.position),arrival->receiver.position);
            ok&=expect(gt::inner(delta,delta)<.004*.004&&near(incident.fraction,arrival->fraction,3e-3),
                "actual grating endpoint and glass flux agree with independent authored-triangle tracing");
        }
        const auto closest=gt::closest(triangles,gt::doublePoint(incident.origin),gt::doublePoint(incident.direction),1e-5,20);
        ok&=expect(closest.found&&closest.object==incident.hit.objectId&&closest.material==incident.hit.materialId
            &&closest.instance==incident.hit.instanceId&&closest.triangle==incident.hit.triangleId
            &&std::abs(closest.distance-incident.hit.distance)<2e-4,
            "receipt grating identity is the independently verified actual closest hit");
        const auto& description=binding->material->description();
        const double n=description.substrate.refractiveIndex,k=description.substrate.extinctionCoefficient;
        const double substrate=((1-n)*(1-n)+k*k)/((1+n)*(1+n)+k*k);
        const double wavelength=description.spectrum.wavelengthsNanometres[incident.band];
        const double incomingZ=-gt::inner(gt::doublePoint(incident.direction),gt::doublePoint(incident.normal));
        double resolved=0;
        for(unsigned o=0;o<incident.orderCount;++o) {
            const auto& order=incident.orders[o];
            const double tangentX=gt::inner(gt::doublePoint(incident.direction),gt::doublePoint(incident.tangent))
                +order.order*wavelength/description.geometry.grooveSpacingNanometres*description.geometry.directionUv[0];
            const double tangentY=gt::inner(gt::doublePoint(incident.direction),gt::doublePoint(incident.bitangent))
                +order.order*wavelength/description.geometry.grooveSpacingNanometres*description.geometry.directionUv[1];
            const double outgoingZ=std::sqrt(1-tangentX*tangentX-tangentY*tangentY);
            const double phase=2*3.14159265358979323846*description.microstructure.grooveDepthNanometres*(incomingZ+outgoingZ)/wavelength;
            const double duty=description.microstructure.dutyCycle;
            const double efficiency=order.order==0?1-2*duty*(1-duty)*(1-std::cos(phase))
                :std::pow(std::sin(3.14159265358979323846*order.order*duty)/(3.14159265358979323846*order.order),2)
                    *2*(1-std::cos(phase))*outgoingZ/incomingZ;
            const auto direction=gt::plus(gt::plus(gt::plus(gt::D3{},gt::doublePoint(incident.tangent),tangentX),
                gt::doublePoint(incident.bitangent),tangentY),gt::doublePoint(incident.normal),outgoingZ);
            for(unsigned a=0;a<3;++a)ok&=expect(std::abs(order.direction[a]-direction[a])<2e-5,
                "each signed native order follows the independent world-frame grating equation");
            ok&=expect(near(order.power,incident.power*substrate*efficiency,2e-3),
                "native per-band order flux matches independent binary complex-amplitude physics");
            firstOrders|=order.order!=0&&order.power>incident.power*1e-7;resolved+=order.power;
        }
        ok&=expect(incident.orderCount==3&&near(incident.substrateReflectedPower,incident.power*substrate)
            &&near(resolved+incident.unresolvedPower+incident.absorbedPower,incident.power),
            "physical zero and both signed first orders partition actual grating incident flux without renormalization");
        ok&=expect(std::abs(incident.uv[0]-incident.hit.position[0]/12)<1e-5
            &&std::abs(incident.uv[1]-incident.hit.position[1]/12)<1e-5
            &&gc::sameGratingPoint(gc::cross(incident.tangent,incident.bitangent),incident.normal),
            "native actual-hit UV and frame preserve the authored right-handed witness");
    }
    dispersed=!gc::sameGratingPoint(bandDirection[0],bandDirection[7],1e-5f);
    ok&=expect(firstOrders&&dispersed&&std::all_of(bands.begin(),bands.end(),[](auto count){return count>0;}),
        "nonzero signed grating orders retain all eight physical bands and causal Cauchy glass dispersion");
    if(!native.frame->readRawPass(renderpassoutput::Output::Color,after,error))return expect(false,error.c_str());
    ok&=expect(before.bytes==after.bytes,"incident-only producer leaves the completed staging Color unchanged");
    gc::Frame combined=fixture.frame;combined.gratingIncidentBinding=binding;
    ok&=expect(!gc::validateTables(fixture.prepared,combined,error),"combined Color remains refused at the shared prerequisite guard");
    gc::NativeReadback returned;returned.frameIdentity="incumbent";gc::EnergyLedger returnedEnergy;returnedEnergy.launched.fill(29);
    ok&=expect(!arbitgpu::produceOpenGlGlassGratingTerminalReadback(native.frame,binding,returned,returnedEnergy,error)
        &&!error.empty()&&returned.frameIdentity=="incumbent"&&returnedEnergy.launched[0]==29,
        "actual native flat-target glass return refuses the whole terminal batch atomically");
    gc::NativeGratingReadback repeated;
    ok&=expect(arbitgpu::produceOpenGlGlassGratingIncidents(native.frame,binding,repeated,error)
        &&repeated.incidents.size()==receipt.incidents.size()&&repeated.submission.submissionSerial>receipt.submission.submissionSerial,
        "repeated adjunct submissions receive fresh exact completed native fences");
    if(repeated.incidents.size()==receipt.incidents.size())for(std::size_t i=0;i<receipt.incidents.size();++i)
        ok&=expect(repeated.incidents[i].hit.position==receipt.incidents[i].hit.position
            &&repeated.incidents[i].orders[0].power==receipt.incidents[i].orders[0].power,
            "equal-time native adjunct query and order results are deterministic");
    const auto reject=[&](const std::shared_ptr<const gc::GratingIncidentBinding>& bad,const char* message) {
        auto incumbent=receipt;const auto identity=incumbent.submission;const auto count=incumbent.incidents.size();
        ok&=expect(!arbitgpu::produceOpenGlGlassGratingIncidents(native.frame,bad,incumbent,error)
            &&!error.empty()&&incumbent.submission==identity&&incumbent.incidents.size()==count,message);
    };
    auto stale=std::make_shared<gc::GratingIncidentBinding>(*binding);stale->frame.identity+="stale";
    reject(stale,"stale staging identity refuses atomically without replacing completed incident output");
    stale=std::make_shared<gc::GratingIncidentBinding>(*binding);stale->witnessTexels[0][0]+=.1f;
    reject(stale,"forged nearby authored witness cannot replace actual frozen closest-hit geometry");
    auto foreignScene=std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*binding->prepared.glass.sourceScene);
    foreignScene->objects[3].transform.translation.x+=1;
    stale=std::make_shared<gc::GratingIncidentBinding>(*binding);stale->prepared.glass.sourceScene=foreignScene;
    reject(stale,"same-ID scene snapshot with foreign authored transforms cannot bind retained native staging geometry");
    auto foreignStaging=std::make_shared<gt::Program>(*binding->prepared.glass.sourceMaterial);foreignStaging->programIdentity+="foreign";
    stale=std::make_shared<gc::GratingIncidentBinding>(*binding);stale->prepared.glass.sourceMaterial=foreignStaging;
    reject(stale,"same-ID staging material with foreign compiled identity cannot bind retained native resources");
    stale=std::make_shared<gc::GratingIncidentBinding>(*binding);stale->frame.roughness=.05f;
    reject(stale,"rough glass cannot enter the explicitly smooth native grating prerequisite");
    auto foreignOwner=std::make_shared<gt::Program>(*binding->gratingOwner);foreignOwner->materialRevision++;
    stale=std::make_shared<gc::GratingIncidentBinding>(*binding);stale->gratingOwner=foreignOwner;
    reject(stale,"foreign exact grating material revision refuses before native submission");
    foreignOwner=std::make_shared<gt::Program>(*binding->gratingOwner);foreignOwner->object=fixture.prepared.glass.object;
    stale=std::make_shared<gc::GratingIncidentBinding>(*binding);stale->gratingOwner=foreignOwner;
    reject(stale,"same-object grating cannot replace the closed native glass object");
    std::shared_ptr<const gc::GratingIncidentBinding> highTail;
    if(!bind(fixture,125,1000,highTail))return expect(false,error.c_str());
    reject(highTail,"the physical 125nm binary preset honestly refuses its omitted-order tail");
    std::shared_ptr<const gc::GratingIncidentBinding> revised;
    if(!bind(fixture,.9f,1100,revised))return expect(false,error.c_str());
    gc::NativeGratingReadback changed;
    if(!arbitgpu::produceOpenGlGlassGratingIncidents(native.frame,revised,changed,error))return expect(false,error.c_str());
    ok&=expect(changed.submission.frameIdentity!=receipt.submission.frameIdentity
        &&changed.incidents.size()==receipt.incidents.size(),"existing admitted IR period edit changes exact adjunct identity");
    bool directionChanged=false;
    for(std::size_t i=0;i<std::min(changed.incidents.size(),receipt.incidents.size());++i)
        for(unsigned o=0;o<std::min(changed.incidents[i].orderCount,receipt.incidents[i].orderCount);++o)
            directionChanged|=!gc::sameGratingPoint(changed.incidents[i].orders[o].direction,receipt.incidents[i].orders[o].direction,1e-3f);
    ok&=expect(directionChanged,"authored grating period causally changes native signed world directions");
    auto half=staging(.5f);if(!half.prepare(error))return expect(false,error.c_str());
    const auto halfPrepared=backend.prepare(half.scene,half.program);if(!halfPrepared.prepared)return expect(false,halfPrepared.error.c_str());
    const auto halfFrame=arbitgpu::renderOpenGlGlassCausticScene(half.scene,halfPrepared.resources,half.ordinaryLight,
        half.program->object,half.program->authoredMaterial,*half.policy,{gc::kScheduleVersion,4},32,32,runtime);
    if(!halfFrame.rendered)return expect(false,halfFrame.error.c_str());
    std::shared_ptr<const gc::GratingIncidentBinding> halfBinding;if(!bind(half,.9f,1000,halfBinding))return expect(false,error.c_str());
    gc::NativeGratingReadback halfReceipt;
    if(!arbitgpu::produceOpenGlGlassGratingIncidents(halfFrame.frame,halfBinding,halfReceipt,error))return expect(false,error.c_str());
    ok&=expect(halfReceipt.incidents.size()==receipt.incidents.size(),"source spectrum edit preserves geometric incident ancestry");
    for(std::size_t i=0;i<std::min(halfReceipt.incidents.size(),receipt.incidents.size());++i) {
        ok&=expect(halfReceipt.incidents[i].direction==receipt.incidents[i].direction
            &&near(halfReceipt.incidents[i].power*2,receipt.incidents[i].power),"exact Li8 half-power edit scales physical native grating incident flux");
        for(unsigned o=0;o<std::min(halfReceipt.incidents[i].orderCount,receipt.incidents[i].orderCount);++o)
            ok&=expect(near(halfReceipt.incidents[i].orders[o].power*2,receipt.incidents[i].orders[o].power),
                "each native signed-order band scales causally with its authored Li8 power");
    }
    auto malformed=receipt;malformed.submission.completedFence=0;
    ok&=expect(!gc::validateGratingIncidentReadback(*binding,malformed,receipt.submission,error),"native adjunct cannot omit its exact completed fence");
    malformed=receipt;malformed.incidents[0].uv[0]+=.01f;
    ok&=expect(!gc::validateGratingIncidentReadback(*binding,malformed,receipt.submission,error),"native adjunct nearby UV substitution refuses");
    malformed=receipt;malformed.incidents[0].orders[0].order=2;
    ok&=expect(!gc::validateGratingIncidentReadback(*binding,malformed,receipt.submission,error),"native adjunct unknown signed order refuses");
    std::array<gc::Texel,gc::kGratingIncidentStride> packed{};packed[22]={1,1,0,0};packed[14][0]=.5f;
    gc::NativeGratingIncident sentinel;sentinel.work=999;
    ok&=expect(!gc::opengl::decodeGratingIncident(packed.data(),sentinel)&&sentinel.work==999,"fractional native incident ID refuses before integral conversion");
    for(unsigned instance:{9u,unsigned(HarmonicMIDI::grid::Visual3DScene::kMaxObjects-1)}) {
        packed[14][0]=float(instance);
        ok&=expect(gc::opengl::decodeGratingIncident(packed.data(),sentinel)&&sentinel.hit.instanceId==instance,
            "native incident instance ordinals above the grating triangle count remain representable");
    }
    const auto incumbentInstance=sentinel.hit.instanceId;
    packed[14][0]=float(HarmonicMIDI::grid::Visual3DScene::kMaxObjects);
    ok&=expect(!gc::opengl::decodeGratingIncident(packed.data(),sentinel)&&sentinel.hit.instanceId==incumbentInstance,
        "native incident first out-of-range scene instance ordinal refuses atomically");
    sentinel.work=999;packed[14][0]=0;packed[23][0]=1;
    ok&=expect(!gc::opengl::decodeGratingIncident(packed.data(),sentinel)&&sentinel.work==999,"native incident nonzero reserved suffix refuses atomically");
    std::array<gc::Texel,gc::kGratingOrderStride> orderWords{};orderWords[0]={1,1,0,0};orderWords[1]={0,0,1,1};orderWords[3][0]=1;
    gc::NativeGratingOrder orderSentinel;orderSentinel.order=9;
    ok&=expect(!gc::opengl::decodeGratingOrder(orderWords.data(),1,orderSentinel)&&orderSentinel.order==9,
        "native signed-order nonzero reserved suffix refuses atomically");
    return ok;
}

bool glassCausticNativeChecks(arbitgpu::NativeFixtureSceneBackend& backend)
{
    namespace gt=glasscaustictests;
    namespace gc=arbitgpu::glasscaustics;
    using namespace HarmonicMIDI::grid;
    int major=0,minor=0;glGetIntegerv(GL_MAJOR_VERSION,&major);glGetIntegerv(GL_MINOR_VERSION,&minor);
    if(major<4||(major==4&&minor<3))
    {
        std::cout<<"UNVERIFIED: native caustic fixtures require OpenGL 4.3; this context is "<<major<<'.'<<minor<<'\n';
        return true;
    }
    bool ok=true;std::string error;
    auto fixture=gt::ordinaryFixture();fixture.scene->cameras[0].transform.translation={0,0,-2};
    if(!fixture.prepare(error))return expect(false,error.c_str());
    auto preparation=backend.prepare(fixture.scene,fixture.program);
    if(!preparation.prepared)return expect(false,preparation.error.c_str());
    arbitgpu::NativeFixtureSceneRuntimeInputs runtime;runtime.linearColor=true;
    const auto render=[&](const gt::Fixture& f,const arbitgpu::NativeFixtureScenePreparation& p,std::size_t subdivisions=4) {
        return arbitgpu::renderOpenGlGlassCausticScene(f.scene,p.resources,f.ordinaryLight,f.program->object,f.program->authoredMaterial,*f.policy,
            {gc::kScheduleVersion,subdivisions},32,32,runtime);
    };
    auto actual=render(fixture,preparation);
    if(!actual.rendered)return expect(false,actual.error.c_str());
    std::shared_ptr<const gc::NativeReadback> receipt;gc::EnergyLedger energy;
    if(!arbitgpu::readOpenGlGlassCausticReceiptForFixture(*actual.frame,receipt,energy,error))return expect(false,error.c_str());
    ok&=expect(receipt&&!receipt->deposits.empty(),"native caustic has completed GPU branch deposits");
    ok&=expect(receipt->deposition.version==1&&receipt->deposition.complete&&!receipt->deposition.overflow
        &&receipt->deposition.visibleCount>0,"native caustic has completed sorted visible-deposit ownership");
    ok&=expect(std::any_of(receipt->spectralRadiance.begin(),receipt->spectralRadiance.end(),[](float v){return v>0;}),
        "native eight-band accumulation contains visible receiver lighting");
    const auto independent=gt::powerOracle(fixture);
    ok&=expect(independent.complete,"independent binary64 authored-triangle power oracle completes");
    const auto near=[](double a,double b,double tolerance=8e-4) {
        return std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=tolerance*std::max({std::abs(a),std::abs(b),1e-12});
    };
    for(unsigned band=0;band<8;++band)
    {
        ok&=expect(near(energy.launched[band],independent.energy.launched[band]),"native Li*cosine*area launched band power matches binary64");
        ok&=expect(near(energy.deposited[band],independent.energy.deposited[band]),"native Fresnel/rho outgoing band power matches binary64");
        ok&=expect(near(energy.absorbed[band],independent.energy.absorbed[band]),"native receiver absorption matches binary64");
        ok&=expect(near(energy.launched[band],energy.deposited[band]+energy.absorbed[band]+energy.escaped[band]+energy.unresolved[band],3e-4),
            "native deposited/absorbed/escaped/unresolved band powers conserve launched flux");
    }
    std::vector<double> expectedBands(32*32*8);
    const auto triangles=gt::worldTriangles(*fixture.scene);
    for(const auto& deposit:receipt->deposits)
    {
        const auto& work=fixture.frame.work[deposit.work];const auto& entry=fixture.frame.entries[work.entry];
        const auto oracle=gt::branchOracle(fixture,entry,work.band);
        const auto arrival=std::find_if(oracle.arrivals.begin(),oracle.arrivals.end(),[&](const auto& a) {
            return a.history==deposit.reflectionHistory&&a.interfaces==deposit.interfaces;
        });
        if(!expect(arrival!=oracle.arrivals.end(),"every native deposited branch has an independent triangle/Snell/Fresnel arrival")) {ok=false;continue;}
        ok&=expect(arrival->receiver.object==deposit.receiver.objectId&&arrival->receiver.material==deposit.receiver.materialId
            &&arrival->receiver.instance==deposit.receiver.instanceId&&arrival->receiver.triangle==deposit.receiver.triangleId,
            "native branch retains exact independent receiver object/material/instance/triangle IDs");
        ok&=expect(near(deposit.fraction,arrival->fraction,4e-4),"native branch fraction matches independent lossless optics");
        for(unsigned axis=0;axis<3;++axis)
            ok&=expect(std::abs(arrival->receiver.position[axis]-deposit.receiver.position[axis])<3e-4,
                "native receiver arrival position matches independent authored triangles");
        if(!deposit.cameraVisible)continue;
        const auto& p=arrival->receiver.position;const gt::D3 camera{0,0,-2};
        const double depth=-(p[2]-camera[2]),tangent=std::tan(double(fixture.frame.camera.verticalFovRadians)/2);
        const auto x=std::uint32_t((p[0]/(depth*tangent)+1)*16),y=std::uint32_t((1-p[1]/(depth*tangent))*16);
        ok&=expect(x==deposit.footprint.x&&y==deposit.footprint.y,"independent camera projection selects native top-first pixel");
        const auto ray=[&](double px,double py){return gt::normalized(gt::D3{(2*px/32-1)*tangent,(1-2*py/32)*tangent,-1});};
        const auto center=gt::closest(triangles,camera,ray(x+.5,y+.5),0,fixture.frame.camera.farPlane);
        ok&=expect(center.found&&center.object==deposit.raster.object&&center.material==deposit.raster.material
            &&center.instance==deposit.raster.instance&&center.triangle==deposit.raster.triangle,
            "private GPU raster IDs agree with independent center-ray triangle");
        std::array<gt::D3,4> corners;
        for(unsigned corner=0;corner<4;++corner)
        {
            const auto direction=ray(x+(corner==1||corner==2),y+(corner>=2));
            corners[corner]=gt::plus(camera,direction,gt::inner(arrival->receiver.normal,gt::minus(p,camera))/gt::inner(arrival->receiver.normal,direction));
        }
        const auto area=[](gt::D3 a,gt::D3 b,gt::D3 c){const auto n=gt::exterior(gt::minus(b,a),gt::minus(c,a));return .5*std::sqrt(gt::inner(n,n));};
        const double pixelArea=area(corners[0],corners[1],corners[2])+area(corners[0],corners[2],corners[3]);
        ok&=expect(near(pixelArea,deposit.footprint.worldArea,3e-4),"native receiver-plane pixel area matches independent binary64 corners");
        const auto* material=arbitgpu::nativeSurfaceForObject(fixture.program.get(),SceneObjectId{arrival->receiver.object});
        const double power=double(work.incidentPower)*arrival->fraction*material->spectralReflectance->coefficients[work.band];
        if(x<32&&y<32)expectedBands[(y*32+x)*8+work.band]+=power/(std::acos(-1.0)*pixelArea);
    }
    for(std::size_t pixelBand=0;pixelBand<expectedBands.size();++pixelBand)
        ok&=expect(near(receipt->spectralRadiance[pixelBand],expectedBands[pixelBand],1.2e-3),
            "completed GPU per-pixel eight-band radiance agrees with independent physical power/area oracle");
    auto base=backend.render(fixture.scene,preparation.resources,32,32,runtime);
    arbitgpu::NativeRawPassPixels before,after;
    if(!base.rendered||!base.frame->readRawPass(renderpassoutput::Output::Color,before,error)
        ||!actual.frame->readRawPass(renderpassoutput::Output::Color,after,error))return expect(false,error.c_str());
    auto repeated=render(fixture,preparation);arbitgpu::NativeRawPassPixels repeatedColor;
    if(!repeated.rendered||!repeated.frame->readRawPass(renderpassoutput::Output::Color,repeatedColor,error))
        return expect(false,repeated.error.empty()?error.c_str():repeated.error.c_str());
    ok&=expect(after.bytes==repeatedColor.bytes,"equal-time native caustic submissions resolve identical Color bytes");
    const auto weights=gc::resolveWeights();
    for(std::size_t pixel=0;pixel<32*32;++pixel)
    {
        std::array<double,3> xyz{};
        for(unsigned b=0;b<8;++b)for(unsigned a=0;a<3;++a)xyz[a]+=expectedBands[pixel*8+b]*weights[b][a]*weights[b][3];
        const double rgb[]{3.2406*xyz[0]-1.5372*xyz[1]-.4986*xyz[2],-.9689*xyz[0]+1.8758*xyz[1]+.0415*xyz[2],.0557*xyz[0]-.2040*xyz[1]+1.0570*xyz[2]};
        for(unsigned channel=0;channel<3;++channel)
        {
            std::uint16_t a=0,b=0;const auto offset=(pixel*4+channel)*2;
            std::memcpy(&a,before.bytes.data()+offset,2);std::memcpy(&b,after.bytes.data()+offset,2);
            const double expected=hdrimage::halfToFloat(a)+rgb[channel],observed=hdrimage::halfToFloat(b);
            ok&=expect(std::abs(expected-observed)<.003*std::max(1.0,std::abs(expected)),
                "product scene-linear Color adds the validated native spectrum using production CIE resolve");
        }
    }
    for(unsigned perturb=0;perturb<2;++perturb)
    {
        auto changed=gt::ordinaryFixture();changed.scene->cameras[0].transform.translation={0,0,-2};
        if(perturb==0)changed.scene->lights[0].spectralSource.radiance[2]*=2;
        else
        {
            auto receiver=std::make_shared<gt::Program>(*changed.program->objectPrograms[0]);
            receiver->spectralReflectance->coefficients[2]*=2;changed.program->objectPrograms[0]=receiver;
        }
        auto preparedChanged=backend.prepare(changed.scene,changed.program);auto output=render(changed,preparedChanged);
        std::shared_ptr<const gc::NativeReadback> changedReceipt;gc::EnergyLedger changedEnergy;
        if(!output.rendered||!arbitgpu::readOpenGlGlassCausticReceiptForFixture(*output.frame,changedReceipt,changedEnergy,error))
            return expect(false,output.error.empty()?error.c_str():output.error.c_str());
        for(unsigned b=0;b<8;++b)ok&=expect(near(changedEnergy.deposited[b],energy.deposited[b]*(b==2?2:1)),
            "independently authored Li-only/rho-only edit changes exactly its native wavelength power");
    }
    for(unsigned blocked=0;blocked<2;++blocked)
    {
        auto control=gt::fixture(.004f,blocked==0?1:2,false);
        control.scene->lights[0]=fixture.scene->lights[0];control.ordinaryLight=fixture.ordinaryLight;control.policy=fixture.policy;
        control.scene->cameras[0].transform.translation={0,0,-2};
        if(blocked==1)
        {
            const auto& object=control.scene->objects[2];
            for(std::size_t v=object.firstVertex;v<object.firstVertex+object.vertexCount;++v)control.scene->vertices[v].position.z=-8;
        }
        auto preparedControl=backend.prepare(control.scene,control.program);auto output=render(control,preparedControl);
        std::shared_ptr<const gc::NativeReadback> nativeControl;gc::EnergyLedger controlEnergy;
        if(!output.rendered||!arbitgpu::readOpenGlGlassCausticReceiptForFixture(*output.frame,nativeControl,controlEnergy,error))
            return expect(false,output.error.empty()?error.c_str():output.error.c_str());
        for(unsigned b=0;b<8;++b)
            ok&=expect(near(controlEnergy.deposited[b],blocked==0?0:energy.deposited[b]),
                "native source blocker removes caustic power; a blocker beyond the receiver preserves it");
    }
    const auto centroid=[](const gc::NativeReadback& frame) {
        std::array<double,4> moment{};
        for(const auto& d:frame.deposits) {moment[3]+=d.outgoingPower;for(unsigned a=0;a<3;++a)moment[a]+=d.receiver.position[a]*d.outgoingPower;}
        if(moment[3]>0)for(unsigned a=0;a<3;++a)moment[a]/=moment[3];
        return moment;
    };
    const auto originalCentroid=centroid(*receipt);
    for(unsigned movement=0;movement<3;++movement)
    {
        auto moved=gt::ordinaryFixture();moved.scene->cameras[0].transform.translation={0,0,-2};
        if(movement==0)moved.scene->objects[0].transform.translation.x=.3f;
        if(movement==1)moved.scene->objects[1].transform.translation.z=-1;
        if(movement==2) {const double angle=-std::atan(.2);moved.scene->lights[0].transform.rotation={0,float(std::sin(angle/2)),0,float(std::cos(angle/2))};}
        auto preparedMoved=backend.prepare(moved.scene,moved.program);auto output=render(moved,preparedMoved);
        std::shared_ptr<const gc::NativeReadback> nativeMoved;gc::EnergyLedger movedEnergy;
        if(!output.rendered||!arbitgpu::readOpenGlGlassCausticReceiptForFixture(*output.frame,nativeMoved,movedEnergy,error))
            return expect(false,output.error.empty()?error.c_str():output.error.c_str());
        const auto movedCentroid=centroid(*nativeMoved);
        ok&=expect(std::abs(movedCentroid[0]-originalCentroid[0])>.005,
            "native glass/receiver/source movement changes physical receiver caustic centroid");
    }
    auto staleScene=std::make_shared<const Visual3DScene>(*fixture.scene);
    auto stale=arbitgpu::renderOpenGlGlassCausticScene(staleScene,preparation.resources,fixture.ordinaryLight,fixture.program->object,fixture.program->authoredMaterial,*fixture.policy,{},32,32,runtime);
    auto foreign=arbitgpu::renderOpenGlGlassCausticScene(fixture.scene,preparation.resources,SceneLightId{999},fixture.program->object,fixture.program->authoredMaterial,*fixture.policy,{},32,32,runtime);
    auto oversized=arbitgpu::renderOpenGlGlassCausticScene(fixture.scene,preparation.resources,fixture.ordinaryLight,fixture.program->object,fixture.program->authoredMaterial,*fixture.policy,{},8192,8192,runtime);
    ok&=expect(!stale.rendered&&!stale.frame&&!foreign.rendered&&!foreign.frame&&!oversized.rendered&&!oversized.frame,
        "stale scene, foreign source and overflow extent publish no native Color frame");
    gc::EnergyLedger validated;
    ok&=expect(gc::validateNativeReadback(fixture.prepared,fixture.frame,*receipt,validated,error,&receipt->deposition.submission),
        "completed native canonical deposition passes its independent whole-frame owner gate");
    gc::EnergyLedger sentinel;sentinel.launched.fill(123);auto bad=*receipt;bad.frameIdentity+="stale";
    ok&=expect(!gc::validateNativeReadback(fixture.prepared,fixture.frame,bad,sentinel,error,&receipt->deposition.submission)&&sentinel.launched[0]==123,
        "stale completed native readback leaves whole-frame ledger unchanged");
    bad=*receipt;bad.spectralRadiance[0]=std::numeric_limits<float>::infinity();
    ok&=expect(!gc::validateNativeReadback(fixture.prepared,fixture.frame,bad,sentinel,error,&receipt->deposition.submission)&&sentinel.launched[0]==123,
        "nonfinite completed native accumulation refuses atomically");
    const auto visible=std::find_if(receipt->deposits.begin(),receipt->deposits.end(),[](const auto& d){return d.cameraVisible;});
    if(visible!=receipt->deposits.end())
    {
        bad=*receipt;bad.deposits[std::size_t(visible-receipt->deposits.begin())].raster.triangle^=1;
        ok&=expect(!gc::validateNativeReadback(fixture.prepared,fixture.frame,bad,sentinel,error,&receipt->deposition.submission)&&sentinel.launched[0]==123,
            "stale private raster triangle receipt refuses atomically");
    }
    bad=*receipt;bad.paths[0].queryWork=gc::kMaximumQueryWork+1;
    ok&=expect(!gc::validateNativeReadback(fixture.prepared,fixture.frame,bad,sentinel,error,&receipt->deposition.submission)&&sentinel.launched[0]==123,
        "over-budget measured native query receipt refuses atomically");
    bad=*receipt;bad.deposition.complete=false;
    ok&=expect(!gc::validateNativeReadback(fixture.prepared,fixture.frame,bad,sentinel,error,&receipt->deposition.submission)
        &&sentinel.launched[0]==123,"partial sorted native deposition refuses atomically");
    bad=*receipt;bad.deposition.submission.scratchGeneration++;
    ok&=expect(!gc::validateNativeReadback(fixture.prepared,fixture.frame,bad,sentinel,error,&receipt->deposition.submission)
        &&sentinel.launched[0]==123,"foreign native scratch submission refuses atomically");
    if(receipt->deposition.visibleCount>1)
    {
        bad=*receipt;std::swap(bad.deposition.keys[0],bad.deposition.keys[1]);
        ok&=expect(!gc::validateNativeReadback(fixture.prepared,fixture.frame,bad,sentinel,error,&receipt->deposition.submission)
            &&sentinel.launched[0]==123,"reordered native canonical deposit keys refuse atomically");
    }
    auto coarse=render(fixture,preparation,1);auto medium=render(fixture,preparation,2);
    for(const auto* output:{&coarse,&medium})
    {
        std::shared_ptr<const gc::NativeReadback> refined;gc::EnergyLedger refinedEnergy;
        if(!output->rendered||!arbitgpu::readOpenGlGlassCausticReceiptForFixture(*output->frame,refined,refinedEnergy,error))
            return expect(false,output->error.empty()?error.c_str():output->error.c_str());
        double sum=0,momentX=0,momentY=0;
        for(const auto& deposit:refined->deposits)if(deposit.cameraVisible)
        {sum+=deposit.outgoingPower;momentX+=(deposit.footprint.x+.5)*deposit.outgoingPower;momentY+=(deposit.footprint.y+.5)*deposit.outgoingPower;}
        std::cout<<"native-caustic-refinement paths="<<refined->paths.size()<<" visible-power="<<sum
            <<" pixel-centroid="<<(sum?momentX/sum:0)<<','<<(sum?momentY/sum:0)<<'\n';
        for(unsigned b=0;b<8;++b)ok&=expect(near(refinedEnergy.launched[b],energy.launched[b]),
            "native entry refinement preserves integrated launched power independently of pixel centroid");
    }
    // Normal incidence through a parallel slab maps the uniform square source
    // exactly onto the receiver. Its independent normalized second moment is
    // 1/3 on both axes, including every reflected branch that reaches it.
    auto spatial=gt::ordinaryFixture();spatial.scene->cameras[0].transform.translation={0,0,-2};
    spatial.scene->lights[0].transform.rotation={0,0,0,1};
    const auto& glass=spatial.scene->objects[0];
    for(std::size_t v=glass.firstVertex;v<glass.firstVertex+glass.vertexCount;++v)
        if(spatial.scene->vertices[v].position.z>0)
            spatial.scene->vertices[v].position.z-=.25f*spatial.scene->vertices[v].position.x;
    if(!spatial.prepare(error))return expect(false,error.c_str());
    const auto spatialReference=gt::quadraturePowerOracle(spatial);
    ok&=expect(spatialReference.complete,"independent full-branch receiver quadrature completes for the parallel slab");
    auto spatialPreparation=backend.prepare(spatial.scene,spatial.program);
    std::array<std::array<double,8>,3> spatialErrors{};
    for(unsigned level=0;level<3;++level)
    {
        auto output=render(spatial,spatialPreparation,std::size_t(1)<<level);
        std::shared_ptr<const gc::NativeReadback> nativeSpatial;gc::EnergyLedger spatialEnergy;
        if(!output.rendered||!arbitgpu::readOpenGlGlassCausticReceiptForFixture(*output.frame,nativeSpatial,spatialEnergy,error))
            return expect(false,output.error.empty()?error.c_str():output.error.c_str());
        std::array<double,8> power{},x{},y{},xx{},yy{};
        for(const auto& deposit:nativeSpatial->deposits)
        {
            const auto band=deposit.work%8;const double p=deposit.outgoingPower;
            power[band]+=p;x[band]+=p*deposit.receiver.position[0];y[band]+=p*deposit.receiver.position[1];
            xx[band]+=p*deposit.receiver.position[0]*deposit.receiver.position[0];
            yy[band]+=p*deposit.receiver.position[1]*deposit.receiver.position[1];
        }
        for(unsigned b=0;b<8;++b)
        {
            ok&=expect(near(power[b],spatialReference.energy.deposited[b])
                &&std::abs(x[b]/power[b]-spatialReference.centroid[b][0])<3e-4
                &&std::abs(y[b]/power[b]-spatialReference.centroid[b][1])<3e-4,
                "native spatial refinement preserves the independent uniform receiver centroid");
            spatialErrors[level][b]=std::max(std::abs(xx[b]/power[b]-1.0/3),std::abs(yy[b]/power[b]-1.0/3));
            if(level)ok&=expect(spatialErrors[level][b]<=.4*spatialErrors[level-1][b]+1e-6,
                "native 1/2/4 entry refinement converges receiver second moments toward exact square integration");
        }
        std::cout<<"native-caustic-spatial subdivisions="<<(1u<<level)<<" max-second-moment-error="
            <<*std::max_element(spatialErrors[level].begin(),spatialErrors[level].end())<<'\n';
    }
    // Author a separate full-resolution reference, keeping the same directional
    // source, glass optics, receiver and branch policy. The broad camera and Li
    // put even conservative all-power-in-one-pixel bounds below RGBA16F limits.
    auto fullFixture=gt::ordinaryFixture();
    fullFixture.scene->cameras[0].transform.translation={6,0,3};
    fullFixture.scene->cameras[0].verticalFovRadians=2;
    for(auto& value:fullFixture.scene->lights[0].spectralSource.radiance)value*=1e-4f;
    if(!fullFixture.prepare(error))return expect(false,error.c_str());
    const auto fullPreparation=backend.prepare(fullFixture.scene,fullFixture.program);
    if(!fullPreparation.prepared)return expect(false,fullPreparation.error.c_str());
    for(unsigned band=0;band<8;++band)
        ok&=expect(fullFixture.frame.sourceRadiance[band/4][band%4]
            ==fullFixture.scene->lights[0].spectralSource.radiance[band],
            "full-resolution admitted Li retains the independently authored source bands");
    const auto fullIndependent=gt::powerOracle(fullFixture);
    ok&=expect(fullIndependent.complete,"independent full-resolution authored source and optics oracle completes");
    for(const auto extent:{std::array<std::uint32_t,2>{1920,1080},std::array<std::uint32_t,2>{3840,2160}})
    {
        // Even concentrating every launched photon into one receiver pixel must
        // stay inside the unchanged native spectral storage limit.
        const double receiverDepth=double(fullFixture.frame.camera.transform.translation.z)+3;
        const double minimumArea=4*receiverDepth*receiverDepth*std::pow(std::tan(double(fullFixture.frame.camera.verticalFovRadians)/2),2)
            /(double(extent[1])*extent[1]);
        for(unsigned band=0;band<8;++band)
            ok&=expect(fullIndependent.energy.launched[band]/(std::acos(-1.0)*minimumArea)<65504,
                "authored full-resolution total band power fits one linear-storage pixel without saturation");
        auto output=arbitgpu::renderOpenGlGlassCausticScene(fullFixture.scene,fullPreparation.resources,fullFixture.ordinaryLight,
            fullFixture.program->object,fullFixture.program->authoredMaterial,*fullFixture.policy,{gc::kScheduleVersion,4},extent[0],extent[1],runtime);
        std::shared_ptr<const gc::NativeReadback> nativeFull;gc::EnergyLedger fullEnergy;
        arbitgpu::OpenGlGlassCausticDiagnostics timing;arbitgpu::NativeRawPassPixels fullColor;
        if(!output.rendered||!arbitgpu::readOpenGlGlassCausticReceiptForFixture(*output.frame,nativeFull,fullEnergy,error,&timing)
            ||!output.frame->readRawPass(renderpassoutput::Output::Color,fullColor,error))
            return expect(false,output.error.empty()?error.c_str():output.error.c_str());
        const auto pixelCount=std::uint64_t(extent[0])*extent[1];
        ok&=expect(fullColor.format==arbitgpu::NativeTexturePixelFormat::Rgba16Float
            &&nativeFull->spectralRadiance.size()==pixelCount*8,"full-resolution native caustics retain owned linear Color and eight bands");
        ok&=expect(timing.admittedFrameBytes<=renderpassoutput::kMaximumTotalBytes
            &&timing.nativeStorageBytes>pixelCount*gc::kAccumulatorBytesPerPixel
            &&timing.queryWork<=gc::kMaximumQueryWork&&timing.deposition.pixelCount==pixelCount,
            "full-resolution native resources and measured queries match their checked admission");
        ok&=expect(timing.deposition.depositionWork<pixelCount*fullFixture.frame.bounds.maximumDeposits,
            "full-resolution stable deposition work avoids the all-pixel/all-deposit product");
        for(unsigned b=0;b<8;++b)ok&=expect(near(fullEnergy.launched[b],fullIndependent.energy.launched[b])
            &&near(fullEnergy.deposited[b],fullIndependent.energy.deposited[b])
            &&near(fullEnergy.absorbed[b],fullIndependent.energy.absorbed[b]),
            "full-resolution native per-band receiver energy matches independent binary64 optics");
        std::map<std::uint64_t,double> expectedFull;
        const double tangent=std::tan(double(fullFixture.frame.camera.verticalFovRadians)/2),aspect=double(extent[0])/extent[1];
        for(const auto& deposit:nativeFull->deposits)
        {
            const auto& work=fullFixture.frame.work[deposit.work];const auto branch=gt::branchOracle(fullFixture,fullFixture.frame.entries[work.entry],work.band);
            const auto arrival=std::find_if(branch.arrivals.begin(),branch.arrivals.end(),[&](const auto& a) {
                return a.history==deposit.reflectionHistory&&a.interfaces==deposit.interfaces;
            });
            if(!expect(arrival!=branch.arrivals.end(),"full-resolution native receiver retains an independent physical branch")) {ok=false;continue;}
            ok&=expect(arrival->receiver.object==deposit.receiver.objectId&&arrival->receiver.material==deposit.receiver.materialId
                &&arrival->receiver.instance==deposit.receiver.instanceId&&arrival->receiver.triangle==deposit.receiver.triangleId,
                "full-resolution native deposits retain exact semantic receiver triangles");
            ok&=expect(near(deposit.fraction,arrival->fraction,4e-4),
                "full-resolution native branch fraction matches independent lossless optics");
            for(unsigned axis=0;axis<3;++axis)
                ok&=expect(std::abs(arrival->receiver.position[axis]-deposit.receiver.position[axis])<3e-4,
                    "full-resolution receiver arrival matches independent authored triangles");
            if(!deposit.cameraVisible)continue;
            const auto& position=arrival->receiver.position;const double depth=double(fullFixture.frame.camera.transform.translation.z)-position[2];
            const auto px=std::uint32_t(((position[0]-fullFixture.frame.camera.transform.translation.x)/(depth*tangent*aspect)+1)*.5*extent[0]);
            const auto py=std::uint32_t((1-(position[1]-fullFixture.frame.camera.transform.translation.y)/(depth*tangent))*.5*extent[1]);
            ok&=expect(px==deposit.footprint.x&&py==deposit.footprint.y,
                "full-resolution receiver projection matches independent binary64 camera coordinates");
            const double area=4*depth*depth*tangent*tangent/(double(extent[1])*extent[1]);
            ok&=expect(near(area,deposit.footprint.worldArea,3e-4),
                "full-resolution native pixel area matches independent receiver-plane projection");
            const auto* receiver=arbitgpu::nativeSurfaceForObject(fullFixture.program.get(),SceneObjectId{arrival->receiver.object});
            const double outgoing=double(work.incidentPower)*arrival->fraction*receiver->spectralReflectance->coefficients[work.band];
            expectedFull[(std::uint64_t(py)*extent[0]+px)*8+work.band]+=outgoing/(std::acos(-1.0)*area);
        }
        bool spectraMatch=true;
        for(std::size_t bandPixel=0;bandPixel<nativeFull->spectralRadiance.size();++bandPixel)
        {
            const auto found=expectedFull.find(bandPixel);const double expected=found==expectedFull.end()?0:found->second;
            if(!near(nativeFull->spectralRadiance[bandPixel],expected,1.2e-3)) {spectraMatch=false;break;}
        }
        ok&=expect(spectraMatch,"full-resolution GPU spatial eight-band radiance matches independent power and receiver area");
        const auto spectralHash=videohelper::sha256Bytes(nativeFull->spectralRadiance.data(),nativeFull->spectralRadiance.size()*sizeof(float));
        const auto firstSerial=nativeFull->deposition.submission.submissionSerial;
        nativeFull.reset();output.frame.reset();
        auto repeat=arbitgpu::renderOpenGlGlassCausticScene(fullFixture.scene,fullPreparation.resources,fullFixture.ordinaryLight,
            fullFixture.program->object,fullFixture.program->authoredMaterial,*fullFixture.policy,{gc::kScheduleVersion,4},extent[0],extent[1],runtime);
        std::shared_ptr<const gc::NativeReadback> repeatedFull;gc::EnergyLedger repeatedEnergy;arbitgpu::NativeRawPassPixels repeatColor;
        if(!repeat.rendered||!arbitgpu::readOpenGlGlassCausticReceiptForFixture(*repeat.frame,repeatedFull,repeatedEnergy,error)
            ||!repeat.frame->readRawPass(renderpassoutput::Output::Color,repeatColor,error))
            return expect(false,repeat.error.empty()?error.c_str():repeat.error.c_str());
        ok&=expect(fullColor.bytes==repeatColor.bytes&&spectralHash==videohelper::sha256Bytes(
            repeatedFull->spectralRadiance.data(),repeatedFull->spectralRadiance.size()*sizeof(float))
            &&repeatedFull->deposition.submission.submissionSerial>firstSerial,
            "repeated full-resolution native submissions retain equal Color/spectral bytes with fresh ownership");
        const double phases[]{timing.baseMilliseconds,timing.rasterMilliseconds,timing.traceMilliseconds,timing.sortMilliseconds,
            timing.reduceMilliseconds,timing.readbackMilliseconds,timing.resolveMilliseconds,timing.frameMilliseconds};
        ok&=expect(std::all_of(std::begin(phases),std::end(phases),[](double value){return std::isfinite(value)&&value>=0;})
            &&timing.frameMilliseconds>0,"full-resolution phase and frame timings describe completed native work");
        std::cout<<"native-caustic-full-resolution extent="<<extent[0]<<'x'<<extent[1]
            <<" admitted-bytes="<<timing.admittedFrameBytes<<" native-bytes="<<timing.nativeStorageBytes
            <<" query-work="<<timing.queryWork<<" deposition-work="<<timing.deposition.depositionWork
            <<" sort-passes="<<timing.deposition.sortPasses<<" base-ms="<<timing.baseMilliseconds
            <<" raster-ms="<<timing.rasterMilliseconds<<" trace-ms="<<timing.traceMilliseconds
            <<" sort-ms="<<timing.sortMilliseconds<<" reduce-ms="<<timing.reduceMilliseconds
            <<" readback-ms="<<timing.readbackMilliseconds<<" resolve-ms="<<timing.resolveMilliseconds
            <<" frame-ms="<<timing.frameMilliseconds<<'\n';
    }
    std::cout<<"native-caustic-opengl deposits="<<receipt->deposits.size()<<" paths="<<receipt->paths.size()<<'\n';
    return ok;
}

bool roughGlassCausticNativeChecks(arbitgpu::NativeFixtureSceneBackend& backend)
{
    namespace gt=glasscaustictests;
    namespace gc=arbitgpu::glasscaustics;
    using namespace HarmonicMIDI::grid;
    int major=0,minor=0;glGetIntegerv(GL_MAJOR_VERSION,&major);glGetIntegerv(GL_MINOR_VERSION,&minor);
    if(major<4||(major==4&&minor<3))return true;
    bool ok=true;std::string error;
    arbitgpu::NativeFixtureSceneRuntimeInputs runtime;runtime.linearColor=true;
    const auto make=[](float roughness) {
        auto fixture=gt::roughFixture(roughness);fixture.scene->cameras[0].transform.translation={0,0,-2};return fixture;
    };
    const auto schedule=[&](gt::Fixture& fixture,std::uint32_t width,std::uint32_t height) {
        if(!fixture.prepare(error,{arbitgpu::NativeFixtureMaterialBackend::OpenGl,true,true,true}))return false;
        std::vector<arbitgpu::opaquevisibility::Instance> instances;
        for(std::size_t o=0;o<fixture.scene->objectCount;++o) {
            auto matrix=videohelper::tests::sceneclosest::identity();
            const auto& translation=fixture.scene->objects[o].transform.translation;
            matrix[12]=translation.x;matrix[13]=translation.y;matrix[14]=translation.z;instances.push_back({matrix,o});
        }
        return gc::buildScheduledFrame(fixture.prepared,instances,&runtime,width,height,
            {gc::kScheduleVersion,2},fixture.frame,error);
    };
    const auto render=[&](const gt::Fixture& fixture,const arbitgpu::NativeFixtureScenePreparation& preparation,
                          std::uint32_t width,std::uint32_t height) {
        return arbitgpu::renderOpenGlGlassCausticScene(fixture.scene,preparation.resources,fixture.ordinaryLight,
            fixture.program->object,fixture.program->authoredMaterial,*fixture.policy,{gc::kScheduleVersion,2},width,height,runtime);
    };
    const auto near=[](double a,double b,double tolerance=1.5e-3) {
        return std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=tolerance*std::max({std::abs(a),std::abs(b),1e-12});
    };
    auto smooth=make(0);if(!schedule(smooth,32,32))return expect(false,error.c_str());
    auto smoothPreparation=backend.prepare(smooth.scene,smooth.program);auto smoothOutput=render(smooth,smoothPreparation,32,32);
    std::shared_ptr<const gc::NativeReadback> smoothReceipt;gc::EnergyLedger smoothEnergy;
    if(!smoothOutput.rendered||!arbitgpu::readOpenGlGlassCausticReceiptForFixture(*smoothOutput.frame,smoothReceipt,smoothEnergy,error))
        return expect(false,smoothOutput.error.empty()?error.c_str():smoothOutput.error.c_str());
    for(const auto& path:smoothReceipt->paths)ok&=expect(path.maskedPower==0&&path.roughTransportVersion==0&&path.interfaceSample==0,
        "zero roughness preserves legacy smooth native receipt bytes");
    gc::EnergyLedger roughEnergy;
    for(const auto extent:std::array<std::array<std::uint32_t,2>,2>{{{32,32},{113,87}}})
    {
        const auto width=extent[0],height=extent[1];auto fixture=make(.05f);
        if(!schedule(fixture,width,height))return expect(false,error.c_str());
        auto preparation=backend.prepare(fixture.scene,fixture.program);auto output=render(fixture,preparation,width,height);
        std::shared_ptr<const gc::NativeReadback> receipt;gc::EnergyLedger energy;arbitgpu::OpenGlGlassCausticDiagnostics diagnostics;
        if(!output.rendered||!arbitgpu::readOpenGlGlassCausticReceiptForFixture(*output.frame,receipt,energy,error,&diagnostics))
            return expect(false,output.error.empty()?error.c_str():output.error.c_str());
        if(width==32) {
            // The existing positive .05 Smith fixture also admits the actual
            // v2 variant. Keep the older frame through six-program replacement.
            std::shared_ptr<const arbitgpu::OpenGlQueryArithmeticEvidence> oldQuery;
            if(!readQueryContract(*output.frame,arbitgpu::OpenGlQueryArithmeticRole::CausticPath,&receipt->deposition.submission,oldQuery,error))
                return expect(false,"ordinary caustic frame lacks its actual query owner");
            const gc::MultipleScatterPolicy micro;
            const auto replacement=arbitgpu::renderOpenGlGlassCausticScene(fixture.scene,preparation.resources,fixture.ordinaryLight,
                fixture.program->object,fixture.program->authoredMaterial,*fixture.policy,{gc::kScheduleVersion,2},width,height,runtime,&micro);
            if(!replacement.rendered||!replacement.frame)return expect(false,replacement.error.c_str());
            std::shared_ptr<const arbitgpu::OpenGlQueryArithmeticEvidence> newQuery;
            if(!readQueryContract(*replacement.frame,arbitgpu::OpenGlQueryArithmeticRole::CausticPath,nullptr,newQuery,error)
                ||newQuery->program->ownerGeneration==oldQuery->program->ownerGeneration
                ||newQuery->program->linkedProgram==oldQuery->program->linkedProgram
                ||!arbitgpu::validateOpenGlQueryArithmeticEvidenceForFixture(*output.frame,oldQuery,oldQuery->submission,error)
                ||arbitgpu::validateOpenGlQueryArithmeticEvidenceForFixture(*replacement.frame,oldQuery,oldQuery->submission,error))
                return expect(false,"cache replacement lost an older linked query lease or accepted its foreign frame");
        }
        const auto independent=gt::powerOracle(fixture);
        ok&=expect(independent.complete&&fixture.frame.interfaceSamples==16&&receipt->paths.size()==fixture.frame.work.size(),
            "native rough caustics retain every admitted entry/sample/band path");
        ok&=expect(receipt->deposition.complete&&receipt->deposition.visibleCount>0
            &&std::any_of(receipt->spectralRadiance.begin(),receipt->spectralRadiance.end(),[](float value){return value>0;}),
            "native rough caustics publish visible receiver spectrum through complete sorted deposition");
        for(unsigned band=0;band<8;++band) {
            ok&=expect(near(energy.launched[band],independent.energy.launched[band])
                &&near(energy.deposited[band],independent.energy.deposited[band])
                &&near(energy.absorbed[band],independent.energy.absorbed[band])
                &&near(energy.escaped[band],independent.energy.escaped[band]),
                "rough native launched/received/absorbed/escaped band power matches independent VNDF optics");
            ok&=expect(std::abs(energy.masked[band]-independent.energy.masked[band])<8e-7*energy.launched[band]
                &&near(energy.launched[band],energy.deposited[band]+energy.absorbed[band]+energy.escaped[band]
                    +energy.masked[band]+energy.unresolved[band],3e-4),
                "rough native masking is explicit and all band power conserves");
        }
        for(std::size_t w=0;w<receipt->paths.size();++w) {
            const auto& path=receipt->paths[w];
            ok&=expect(path.complete&&path.roughTransportVersion==gc::kRoughTransportVersion
                &&path.interfaceSample==fixture.frame.work[w].interfaceSample
                &&path.maskedPower+path.unresolvedPower<=path.launchedPower*1.0002e-4,
                "every rough GPU path preserves its stratum and refuses excess masked or unresolved flux");
        }
        const auto triangles=gt::worldTriangles(*fixture.scene);const auto light=gt::authoredSource(fixture);
        std::vector<double> expected(std::size_t(width)*height*8);
        bool roughnessMovedArrival=false;
        for(const auto& deposit:receipt->deposits) {
            const auto& work=fixture.frame.work[deposit.work];const auto& entry=fixture.frame.entries[work.entry];
            const auto oracle=gt::branchOracle(fixture,entry,work.band,1e-13,4096,64,{},work.interfaceSample);
            const auto arrival=std::find_if(oracle.arrivals.begin(),oracle.arrivals.end(),[&](const auto& candidate) {
                return candidate.history==deposit.reflectionHistory&&candidate.interfaces==deposit.interfaces;
            });
            if(!expect(arrival!=oracle.arrivals.end(),"rough native branch has an independent sampled authored-triangle arrival")) {ok=false;continue;}
            ok&=expect(arrival->receiver.object==deposit.receiver.objectId&&arrival->receiver.material==deposit.receiver.materialId
                &&arrival->receiver.instance==deposit.receiver.instanceId&&arrival->receiver.triangle==deposit.receiver.triangleId
                &&near(arrival->fraction,deposit.fraction),"rough native arrival retains independent power and exact triangle owners");
            for(unsigned axis=0;axis<3;++axis)ok&=expect(std::abs(arrival->receiver.position[axis]-deposit.receiver.position[axis])<5e-4,
                "rough native footprint follows sampled refraction rather than smooth-ray deposition");
            if(width==32) {
                const auto legacy=std::find_if(smoothReceipt->deposits.begin(),smoothReceipt->deposits.end(),[&](const auto& candidate) {
                    const auto& previous=smooth.frame.work[candidate.work];
                    return previous.entry==work.entry&&previous.band==work.band&&candidate.reflectionHistory==deposit.reflectionHistory;
                });
                if(legacy!=smoothReceipt->deposits.end())roughnessMovedArrival|=std::abs(legacy->receiver.position[0]-deposit.receiver.position[0])>1e-4;
            }
            if(!deposit.cameraVisible)continue;
            const gt::D3 camera{0,0,-2};const auto& point=arrival->receiver.position;
            const double depth=-(point[2]-camera[2]),tangent=std::tan(double(fixture.frame.camera.verticalFovRadians)/2),aspect=double(width)/height;
            const auto x=std::uint32_t((point[0]/(depth*tangent*aspect)+1)*width*.5);
            const auto y=std::uint32_t((1-point[1]/(depth*tangent))*height*.5);
            ok&=expect(x==deposit.footprint.x&&y==deposit.footprint.y,"rough native receiver uses independent full-frame camera pixels");
            const auto ray=gt::normalized(gt::D3{(2*(x+.5)/width-1)*tangent*aspect,(1-2*(y+.5)/height)*tangent,-1});
            const auto center=gt::closest(triangles,camera,ray,0,fixture.frame.camera.farPlane);
            ok&=expect(center.found&&center.object==deposit.raster.object&&center.material==deposit.raster.material
                &&center.instance==deposit.raster.instance&&center.triangle==deposit.raster.triangle,
                "rough caustic private full-frame raster IDs match independent pixel-center triangles");
            const double area=4*depth*depth*tangent*tangent*aspect/(double(width)*height);
            ok&=expect(near(area,deposit.footprint.worldArea),"rough caustic radiance uses independent receiver-plane pixel area");
            const auto* material=arbitgpu::nativeSurfaceForObject(fixture.program.get(),SceneObjectId{arrival->receiver.object});
            const auto triangle=std::find_if(triangles.begin(),triangles.end(),[&](const auto& candidate) {
                return std::tie(candidate.object,candidate.material,candidate.instance,candidate.triangle)
                    ==std::tie(entry.object,entry.material,entry.instance,entry.triangle);
            });
            if(triangle==triangles.end())return expect(false,"rough native entry lacks its independent authored triangle");
            const auto subdivisions=fixture.frame.scheduling.entrySubdivisions;
            const double incident=light.radiance[work.band]*std::max(0.0,gt::inner(triangle->normal,light.direction))
                *triangle->area/(subdivisions*subdivisions*16);
            if(x<width&&y<height)expected[(std::size_t(y)*width+x)*8+work.band]
                +=incident*arrival->fraction*material->spectralReflectance->coefficients[work.band]/(std::acos(-1.0)*area);
        }
        if(width==32)ok&=expect(roughnessMovedArrival,"nonzero authored roughness changes native caustic receiver footprints");
        for(std::size_t band=0;band<expected.size();++band)ok&=expect(near(expected[band],receipt->spectralRadiance[band],2e-3),
            "native rough per-pixel Li8 follows independent sampled branch power and pixel area");
        std::shared_ptr<const arbitgpu::roughglasstiles::Schedule> tiles;arbitgpu::roughglasstiles::Aggregate aggregate;
        if(!arbitgpu::readOpenGlRoughGlassTileSummaryForFixture(*output.frame,tiles,aggregate,error))return expect(false,error.c_str());
        ok&=expect(aggregate.completedPixels==std::uint64_t(width)*height&&aggregate.completedTiles==tiles->tiles().size()
            &&(width==32||tiles->tiles().size()>1)&&diagnostics.admittedFrameBytes<=renderpassoutput::kMaximumTotalBytes,
            "rough caustic Color retains a complete bounded native camera tile schedule");
        auto repeat=render(fixture,preparation,width,height);arbitgpu::NativeRawPassPixels color,repeated;
        if(!repeat.rendered||!output.frame->readRawPass(renderpassoutput::Output::Color,color,error)
            ||!repeat.frame->readRawPass(renderpassoutput::Output::Color,repeated,error))return expect(false,repeat.error.empty()?error.c_str():repeat.error.c_str());
        ok&=expect(color.bytes==repeated.bytes,"rough caustic native Color is byte-repeatable across complete submissions");
        if(width==32) {
            roughEnergy=energy;gc::EnergyLedger checked;
            if(!gc::validateNativeReadback(fixture.prepared,fixture.frame,*receipt,checked,error,&receipt->deposition.submission))
                return expect(false,error.c_str());
            for(unsigned malformed=0;malformed<3;++malformed) {
                auto corrupted=*receipt;
                if(malformed==0)corrupted.paths[0].maskedPower=-1;
                if(malformed==1)corrupted.paths[0].roughTransportVersion=0;
                if(malformed==2)corrupted.paths[0].interfaceSample=16;
                ok&=expect(!gc::validateNativeReadback(fixture.prepared,fixture.frame,corrupted,checked,error,&receipt->deposition.submission),
                    "rough native readback refuses malformed masking/version/stratum on the actual GPU owner");
            }
        }
        auto foreign=std::make_shared<Visual3DScene>(*fixture.scene);
        auto refused=arbitgpu::renderOpenGlGlassCausticScene(foreign,preparation.resources,fixture.ordinaryLight,
            fixture.program->object,fixture.program->authoredMaterial,*fixture.policy,{gc::kScheduleVersion,2},width,height,runtime);
        ok&=expect(!refused.rendered&&!refused.frame,"rough caustics refuse a substituted immutable scene owner without publishing Color");
    }
    auto excessive=make(.35f);auto preparation=backend.prepare(excessive.scene,excessive.program);
    if(!preparation.prepared)return expect(false,preparation.error.c_str());
    auto refused=render(excessive,preparation,32,32);
    ok&=expect(!refused.rendered&&!refused.frame&&!refused.error.empty(),"rough caustics with excess masked flux publish no usable frame");
    for(unsigned edit=0;edit<2;++edit) {
        auto fixture=make(.05f);
        if(edit==0)fixture.scene->lights[0].spectralSource.radiance[2]*=2;
        else {
            auto receiver=std::make_shared<gt::Program>(*fixture.program->objectPrograms[0]);
            receiver->spectralReflectance->coefficients[2]*=2;fixture.program->objectPrograms[0]=receiver;
        }
        auto prepared=backend.prepare(fixture.scene,fixture.program);auto output=render(fixture,prepared,32,32);
        std::shared_ptr<const gc::NativeReadback> receipt;gc::EnergyLedger energy;
        if(!output.rendered||!arbitgpu::readOpenGlGlassCausticReceiptForFixture(*output.frame,receipt,energy,error))
            return expect(false,output.error.empty()?error.c_str():output.error.c_str());
        for(unsigned band=0;band<8;++band)ok&=expect(near(energy.deposited[band],roughEnergy.deposited[band]*(band==2?2:1)),
            "rough native caustics preserve independent per-band Li and receiver reflectance controls");
    }
    for(unsigned blocker=0;blocker<2;++blocker) {
        auto fixture=gt::fixture(.004f,blocker==0?1:2,false);auto source=make(.05f);
        fixture.scene->lights[0]=source.scene->lights[0];fixture.ordinaryLight=source.ordinaryLight;fixture.policy=source.policy;
        fixture.scene->cameras[0].transform.translation={0,0,-2};fixture.program->parameters.emissionRoughness[3]=.05f;
        fixture.scene->materials[0].roughness=.05f;
        if(blocker==1) {
            const auto& object=fixture.scene->objects[2];
            for(std::size_t v=object.firstVertex;v<object.firstVertex+object.vertexCount;++v)fixture.scene->vertices[v].position.z=-8;
        }
        auto prepared=backend.prepare(fixture.scene,fixture.program);auto output=render(fixture,prepared,32,32);
        std::shared_ptr<const gc::NativeReadback> receipt;gc::EnergyLedger energy;
        if(!output.rendered||!arbitgpu::readOpenGlGlassCausticReceiptForFixture(*output.frame,receipt,energy,error))
            return expect(false,output.error.empty()?error.c_str():output.error.c_str());
        for(unsigned band=0;band<8;++band)ok&=expect(near(energy.deposited[band],blocker==0?0:roughEnergy.deposited[band]),
            "rough native source blockers remove transport and behind-receiver blockers preserve it");
    }
    return ok;
}

// Macro companion reference enumerates authored binary64 triangles and calls
// the independent height-walk reference. It never supplies native transport.
struct MsArrival
{
    glasscaustictests::Hit receiver;
    double fraction=0;
    unsigned interfaces=0;
    std::uint64_t history=0;
    std::array<std::uint8_t,32> ancestry{};
};
struct MsPathReference
{
    bool complete=true,blocked=false;
    unsigned events=0,draws=0,visits=0;
    double outgoing=0,absorbed=0,escaped=0,unresolved=0,microResidual=0;
    std::vector<MsArrival> arrivals;
};
MsPathReference multipleScatterPathReference(const glasscaustictests::Fixture& fixture,
    const arbitgpu::glasscaustics::EntrySample& entry,unsigned band,unsigned sample)
{
    namespace gt=glasscaustictests;namespace gc=arbitgpu::glasscaustics;
    using namespace HarmonicMIDI::grid;
    MsPathReference result;
    const auto triangles=gt::worldTriangles(*fixture.scene);const auto source=gt::authoredSource(fixture);
    const auto& policy=fixture.policy->description();const auto& micro=*fixture.frame.multipleScatterPolicy;
    const auto hash=[](std::initializer_list<std::uint32_t> values) {
        std::uint32_t state=2166136261u;for(auto value:values)state=(state^value)*16777619u;return state;
    };
    const auto bias=[](gt::D3 p){return std::max(1e-5,2e-6*std::max({std::abs(p[0]),std::abs(p[1]),std::abs(p[2])}));};
    const auto entryPoint=gt::doublePoint(entry.position);
    result.blocked=gt::closest(triangles,entryPoint,source.direction,1e-5,source.maximum).found;
    if(result.blocked)return result;
    const auto rootKey=hash({entry.object,entry.instance,entry.triangle,entry.quadrature});
    constexpr std::array<double,8> wavelengths{.380,.420,.460,.500,.540,.580,.640,.700};
    const double ior=fixture.referenceIor+fixture.dispersion*(1/(wavelengths[band]*wavelengths[band])-1/(.58756*.58756));
    struct Branch {
        gt::D3 origin,direction;double fraction;unsigned interfaces;bool inside;
        std::uint64_t history;std::array<std::uint8_t,32> ancestry;std::uint32_t samplingHistory;
    };
    std::vector<Branch> pending{{gt::plus(entryPoint,source.direction,2*bias(entryPoint)),
        gt::plus(gt::D3{},source.direction,-1),1,0,false,0,{},2166136261u}};
    while(result.complete&&!pending.empty()&&result.visits<policy.maximumVisits) {
        const auto branch=pending.back();pending.pop_back();++result.visits;
        const auto hit=gt::closest(triangles,branch.origin,branch.direction,branch.interfaces?1e-5:0,source.maximum);
        if(!hit.found) {
            if(branch.inside)result.complete=false;else result.escaped+=branch.fraction;
            continue;
        }
        if(hit.object!=entry.object) {
            const auto* material=arbitgpu::nativeSurfaceForObject(fixture.program.get(),SceneObjectId{hit.object});
            if(branch.inside||!hit.entering||!material||!material->spectralReflectance||!branch.interfaces) {result.complete=false;break;}
            const auto rho=material->spectralReflectance->coefficients[band];
            result.outgoing+=branch.fraction*rho;result.absorbed+=branch.fraction*(1-rho);
            result.arrivals.push_back({hit,branch.fraction,branch.interfaces,branch.history,branch.ancestry});continue;
        }
        if(branch.inside==hit.entering) {result.complete=false;break;}
        if(branch.interfaces>=policy.maximumInterfaces) {result.unresolved+=branch.fraction;continue;}
        const auto normal=gt::plus(gt::D3{},hit.normal,branch.inside?-1:1);
        const auto seed=hash({rootKey,sample,branch.interfaces,branch.samplingHistory});
        const auto walk=gt::microfacetReference(branch.direction,normal,branch.inside?ior:1,branch.inside?1:ior,
            fixture.frame.roughness,seed,micro.maximumEvents,micro.maximumExits,micro.maximumResidual);
        result.events+=walk.events;result.draws+=walk.draws;
        if(walk.status!=1) {result.complete=false;break;}
        result.microResidual+=branch.fraction*walk.residual;result.unresolved+=branch.fraction*walk.residual;
        for(const auto& exit:walk.exits) {
            const auto fraction=branch.fraction*exit.power;
            if(fraction<policy.minimumBranchFraction||pending.size()>=policy.maximumPending) {result.unresolved+=fraction;continue;}
            auto child=branch;child.origin=gt::plus(hit.position,normal,(exit.transmitted?-1:1)*bias(hit.position));
            child.direction=exit.direction;child.fraction=fraction;child.interfaces++;
            child.inside=exit.transmitted?!branch.inside:branch.inside;
            if(!exit.transmitted)child.history|=std::uint64_t(1)<<branch.interfaces;
            child.ancestry[branch.interfaces]=std::uint8_t(exit.order|(exit.transmitted?64:0));
            child.samplingHistory=((branch.samplingHistory^exit.order)*16777619u^std::uint32_t(exit.transmitted))*16777619u;
            pending.push_back(child);
        }
    }
    for(const auto& branch:pending)result.unresolved+=branch.fraction;
    result.complete=result.complete&&result.unresolved<=policy.maximumUnresolvedFraction*(1+2e-4);
    return result;
}

bool multipleScatterGlassCausticNativeChecks(arbitgpu::NativeFixtureSceneBackend& backend)
{
    namespace gt=glasscaustictests;namespace gc=arbitgpu::glasscaustics;
    using namespace HarmonicMIDI::grid;
    int major=0,minor=0;glGetIntegerv(GL_MAJOR_VERSION,&major);glGetIntegerv(GL_MINOR_VERSION,&minor);
    if(major<4||(major==4&&minor<3))return true;
    bool ok=true;std::string error;gc::MultipleScatterPolicy micro;
    arbitgpu::NativeFixtureSceneRuntimeInputs runtime;runtime.linearColor=true;
    const auto make=[](float roughness) {
        auto fixture=gt::reflectionFixture();const auto ordinary=gt::roughFixture(roughness);
        fixture.scene->lights[0]=ordinary.scene->lights[0];fixture.ordinaryLight=ordinary.ordinaryLight;
        fixture.policy=ordinary.policy;fixture.light.reset();fixture.scene->cameras[0].transform.translation={0,0,-2};
        fixture.program->parameters.emissionRoughness[3]=roughness;fixture.scene->materials[0].roughness=roughness;return fixture;
    };
    const auto schedule=[&](gt::Fixture& fixture,std::uint32_t width,std::uint32_t height,const gc::MultipleScatterPolicy& policy) {
        if(!fixture.prepare(error,{arbitgpu::NativeFixtureMaterialBackend::OpenGl,true,true,true}))return false;
        std::vector<arbitgpu::opaquevisibility::Instance> instances;
        for(std::size_t object=0;object<fixture.scene->objectCount;++object) {
            auto world=videohelper::tests::sceneclosest::identity();instances.push_back({world,object});
        }
        return gc::buildMultipleScatterFrame(fixture.prepared,instances,&runtime,width,height,{gc::kScheduleVersion,1},policy,fixture.frame,error);
    };
    const auto render=[&](const gt::Fixture& fixture,const arbitgpu::NativeFixtureScenePreparation& preparation,
        std::uint32_t width,std::uint32_t height,const gc::MultipleScatterPolicy& policy) {
        return arbitgpu::renderOpenGlGlassCausticScene(fixture.scene,preparation.resources,fixture.ordinaryLight,
            fixture.program->object,fixture.program->authoredMaterial,*fixture.policy,{gc::kScheduleVersion,1},width,height,runtime,&policy);
    };
    const auto near=[](double a,double b,double tolerance=2e-3) {
        return std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=tolerance*std::max({std::abs(a),std::abs(b),1e-12});
    };
    for(const auto roughness:{.35f,.75f})for(const auto extent:std::array<std::array<std::uint32_t,2>,2>{{{32,32},{113,87}}}) {
        auto fixture=make(roughness);const auto width=extent[0],height=extent[1];
        if(!schedule(fixture,width,height,micro))return expect(false,error.c_str());
        auto preparation=backend.prepare(fixture.scene,fixture.program);auto output=render(fixture,preparation,width,height,micro);
        std::shared_ptr<const gc::NativeReadback> receipt;gc::EnergyLedger energy;arbitgpu::OpenGlGlassCausticDiagnostics diagnostics;
        if(!output.rendered||!output.frame||!arbitgpu::readOpenGlGlassCausticReceiptForFixture(*output.frame,receipt,energy,error,&diagnostics))
            return expect(false,output.error.empty()?error.c_str():output.error.c_str());
        std::shared_ptr<const arbitgpu::OpenGlQueryArithmeticEvidence> pathQuery,primaryQuery,transportQuery;
        if(!readQueryContract(*output.frame,arbitgpu::OpenGlQueryArithmeticRole::CausticPath,&receipt->deposition.submission,pathQuery,error)
            ||!readQueryContract(*output.frame,arbitgpu::OpenGlQueryArithmeticRole::SelectedCameraPrimary,nullptr,primaryQuery,error)
            ||!readQueryContract(*output.frame,arbitgpu::OpenGlQueryArithmeticRole::SelectedCameraTransport,nullptr,transportQuery,error)
            ||transportQuery->parents[0]!=primaryQuery||!(primaryQuery->submission==transportQuery->submission))
            return expect(false,"selected-camera stages lack completed retained native query owners");
        for(const auto& selected:{primaryQuery,transportQuery}) {
            const auto& source=selected->program->stages[0].source;
            ok&=expect(source.find("glassSceneClosestHit(")!=std::string::npos
                &&source.find("precise float barycentricSum=u+v;")!=std::string::npos,
                "actual linked selected-camera stages retain the qualified renamed glass query");
        }
        ok&=expect(fixture.frame.multipleScatterPolicy&&receipt->paths.size()==fixture.frame.work.size()
            &&receipt->deposition.complete&&receipt->deposition.visibleCount>0,"v2 GPU walks publish complete owned receiver spectrum");
        const auto triangles=gt::worldTriangles(*fixture.scene);const auto source=gt::authoredSource(fixture);
        std::vector<MsPathReference> independent;std::vector<double> inputs;std::array<double,8> launch{},outgoing{},absorbed{},escaped{},tail{};
        std::vector<double> expected(std::size_t(width)*height*8);
        bool laterExit=false,reflectedReceiver=false,transmittedReceiver=false;
        for(std::size_t w=0;w<fixture.frame.work.size();++w) {
            const auto& work=fixture.frame.work[w];const auto& entry=fixture.frame.entries[work.entry];const auto& path=receipt->paths[w];
            independent.push_back(multipleScatterPathReference(fixture,entry,work.band,work.interfaceSample));const auto& reference=independent.back();
            ok&=expect(reference.complete&&path.complete&&path.multipleScatterVersion==2&&path.roughTransportVersion==1
                &&path.interfaceSample==work.interfaceSample&&path.quadrature==entry.quadrature&&path.maskedPower==0
                &&path.microEvents==reference.events&&path.microDraws==reference.draws,
                "native v2 events/draws/quadrature/stratum follow independent height walks");
            const auto triangle=std::find_if(triangles.begin(),triangles.end(),[&](const auto& t){
                return std::tie(t.object,t.material,t.instance,t.triangle)==std::tie(entry.object,entry.material,entry.instance,entry.triangle);
            });
            if(triangle==triangles.end())return expect(false,"v2 entry has no independent authored triangle");
            const double input=reference.blocked?0:source.radiance[work.band]*triangle->area*std::max(0.0,gt::inner(triangle->normal,source.direction))/16;
            inputs.push_back(input);launch[work.band]+=input;outgoing[work.band]+=input*reference.outgoing;absorbed[work.band]+=input*reference.absorbed;
            escaped[work.band]+=input*reference.escaped;tail[work.band]+=input*reference.unresolved;
            ok&=expect(near(path.launchedPower,input)&&near(path.depositedPower,input*reference.outgoing)
                &&near(path.absorbedPower,input*reference.absorbed)&&near(path.escapedPower,input*reference.escaped)
                &&std::abs(path.microResidualPower-input*reference.microResidual)<2e-6*input
                &&path.microResidualPower<=path.unresolvedPower*(1+2e-4)
                &&path.unresolvedPower<=path.launchedPower*1.0002e-4,
                "v2 per-path physical flux includes weighted micro residual exactly once without masking");
        }
        for(const auto& deposit:receipt->deposits) {
            const auto& work=fixture.frame.work[deposit.work];const auto& reference=independent[deposit.work];
            const auto arrival=std::find_if(reference.arrivals.begin(),reference.arrivals.end(),[&](const auto& a){
                return a.interfaces==deposit.interfaces&&a.ancestry==deposit.ancestry;
            });
            if(!expect(arrival!=reference.arrivals.end(),"v2 deposited exact ancestry identifies an independent emitted macro chain")) {ok=false;continue;}
            ok&=expect(arrival->history==deposit.reflectionHistory&&arrival->receiver.object==deposit.receiver.objectId
                &&arrival->receiver.material==deposit.receiver.materialId&&arrival->receiver.instance==deposit.receiver.instanceId
                &&arrival->receiver.triangle==deposit.receiver.triangleId&&near(arrival->fraction,deposit.fraction),
                "v2 deposited ancestry preserves medium parity, reflection history, exact owners and flux");
            for(unsigned axis=0;axis<3;++axis)ok&=expect(std::abs(arrival->receiver.position[axis]-deposit.receiver.position[axis])<4e-3,
                "v2 receiver endpoints follow independent actual multiple-scatter exit directions");
            for(unsigned depth=0;depth<deposit.interfaces;++depth)laterExit|=(deposit.ancestry[depth]&63)>1;
            reflectedReceiver|=deposit.receiver.objectId==fixture.scene->objects[2].id.value;
            transmittedReceiver|=deposit.receiver.objectId==fixture.scene->objects[1].id.value;
            if(!deposit.cameraVisible)continue;
            const auto p=gt::doublePoint(deposit.receiver.position);const double depth=-(p[2]+2);
            const double tangent=std::tan(double(fixture.frame.camera.verticalFovRadians)/2),aspect=double(width)/height;
            const auto x=std::uint32_t((p[0]/(depth*tangent*aspect)+1)*width*.5),y=std::uint32_t((1-p[1]/(depth*tangent))*height*.5);
            const double area=4*depth*depth*tangent*tangent*aspect/(double(width)*height);
            const auto ray=gt::normalized(gt::D3{(2*(x+.5)/width-1)*tangent*aspect,(1-2*(y+.5)/height)*tangent,-1});
            const auto center=gt::closest(triangles,{0,0,-2},ray,0,fixture.frame.camera.farPlane);
            ok&=expect(x==deposit.footprint.x&&y==deposit.footprint.y&&near(area,deposit.footprint.worldArea)
                &&center.found&&center.object==deposit.raster.object&&center.material==deposit.raster.material
                &&center.instance==deposit.raster.instance&&center.triangle==deposit.raster.triangle,
                "v2 splats retain independent full-frame projected area and camera triangle identity");
            const auto* material=arbitgpu::nativeSurfaceForObject(fixture.program.get(),SceneObjectId{arrival->receiver.object});
            if(x<width&&y<height)expected[(std::size_t(y)*width+x)*8+work.band]+=inputs[deposit.work]
                *arrival->fraction*material->spectralReflectance->coefficients[work.band]/(std::acos(-1.0)*area);
        }
        ok&=expect(laterExit&&reflectedReceiver&&transmittedReceiver,"rough v2 has later-event exits and causal reflected plus transmitted receivers");
        for(unsigned band=0;band<8;++band)ok&=expect(near(energy.launched[band],launch[band])&&near(energy.deposited[band],outgoing[band])
            &&near(energy.absorbed[band],absorbed[band])&&near(energy.escaped[band],escaped[band])&&energy.masked[band]==0
            &&std::abs(energy.unresolved[band]-tail[band])<2e-6*launch[band]
            &&near(energy.launched[band],energy.deposited[band]+energy.absorbed[band]+energy.escaped[band]+energy.unresolved[band],3e-4),
            "v2 native per-band energy matches independent emitted branches and conserves launched flux");
        for(std::size_t b=0;b<expected.size();++b)ok&=expect(near(expected[b],receipt->spectralRadiance[b],3e-3),
            "v2 receiver radiance uses independent multi-exit power, rho8 and actual pixel area");
        std::shared_ptr<const arbitgpu::roughglasstiles::Schedule> tiles;arbitgpu::roughglasstiles::Aggregate aggregate;
        if(!arbitgpu::readOpenGlRoughGlassTileSummaryForFixture(*output.frame,tiles,aggregate,error))return expect(false,error.c_str());
        std::uint64_t measured=0;for(const auto& path:receipt->paths)measured+=path.queryWork+path.microEvents+path.microDraws;
        ok&=expect(aggregate.completedPixels==std::uint64_t(width)*height&&aggregate.completedTiles==tiles->tiles().size()
            &&(width==32||tiles->tiles().size()>1)&&measured<=fixture.frame.bounds.combinedWork
            &&diagnostics.admittedFrameBytes<=renderpassoutput::kMaximumTotalBytes
            &&diagnostics.nativeTemporaryBytes==fixture.frame.work.size()*gc::opengl::kMultipleScatterNativeTemporaryBytesPerWork,
            "v2 native camera completion and measured combined work stay within unchanged resource bounds");
        auto repeated=render(fixture,preparation,width,height,micro);arbitgpu::NativeRawPassPixels color,repeatColor;
        if(!repeated.rendered||!output.frame->readRawPass(renderpassoutput::Output::Color,color,error)
            ||!repeated.frame->readRawPass(renderpassoutput::Output::Color,repeatColor,error))return expect(false,repeated.error.empty()?error.c_str():repeated.error.c_str());
        ok&=expect(color.bytes==repeatColor.bytes,"v2 equal-time native submissions have exact repeated Color");
        if(width==32) {
            // Pack the completed GPU owner's typed receipts through the exact
            // backend decoder. These are decoder probes, not rendered evidence.
            const auto words=[](std::uint32_t value){return std::array<float,2>{float(value&65535u),float(value>>16)};};
            const auto hit=[&](gc::Texel* p,const arbitgpu::opaquevisibility::ClosestHit& value) {
                const auto object=words(value.objectId),material=words(value.materialId),instance=words(value.instanceId),triangle=words(value.triangleId),work=words(value.work);
                p[0]={float(value.status),value.distance,float(value.entering),work[0]};p[1]={value.position[0],value.position[1],value.position[2],work[1]};
                p[2]={value.outwardNormal[0],value.outwardNormal[1],value.outwardNormal[2],0};
                p[3]={object[0],object[1],material[0],material[1]};p[4]={instance[0],instance[1],triangle[0],triangle[1]};
            };
            const auto& path=receipt->paths[0];std::array<gc::Texel,16> packed{};
            const auto low=words(std::uint32_t(path.queryWork)),high=words(std::uint32_t(path.queryWork>>32));
            packed[0]={0,0,float(path.depositCount),float(path.visits)};packed[1]={1,float(path.sourceBlocked),0,0};packed[2]={low[0],low[1],high[0],high[1]};
            packed[3]={path.launchedPower,path.depositedPower,path.absorbedPower,path.escapedPower};packed[4]={path.unresolvedPower,0,0,0};
            hit(packed.data()+5,path.sourceHit);hit(packed.data()+10,path.entryHit);
            const auto extension=gc::pathExtensionTexels(path,gc::PathExtensionLayout::OpenGl);packed[14]=extension[0];packed[15]=extension[1];
            gc::NativePathReceipt decoded;
            ok&=expect(gc::opengl::causticDecodePath(packed.data(),fixture.frame,0,decoded)
                &&decoded.entryHit.instanceId==UINT32_MAX&&gc::bindOpenGlMultipleScatterEntry(fixture.frame,decoded,error)
                &&decoded.entryHit.instanceId==path.entryHit.instanceId&&decoded.entryHit.triangleId==path.entryHit.triangleId,
                "v2 packed decoder reads its marker before the overwritten entry IDs and binds exact native world identity");
            for(unsigned malformed=0;malformed<8;++malformed) {
                auto corrupt=packed;gc::NativePathReceipt sentinel;sentinel.work=9999;
                if(malformed==0)corrupt[15][3]=1;
                if(malformed==1)corrupt[14][0]+=.5f;
                if(malformed==2)corrupt[14][1]+=1;
                if(malformed==3)corrupt[14][3]=16;
                if(malformed==4)corrupt[15][0]=.1f;
                if(malformed==5)corrupt[11][0]=std::numeric_limits<float>::quiet_NaN();
                if(malformed==6)corrupt[1][0]=0;
                if(malformed==7)corrupt[4][3]=1;
                ok&=expect(!gc::opengl::causticDecodePath(corrupt.data(),fixture.frame,0,sentinel)&&sentinel.work==9999,
                    "v2 packed path decoder refuses malformed finite/integer/version/reserved fields without mutation");
            }
            const auto& deposit=receipt->deposits[0];std::array<gc::Texel,32> packedDeposit{};
            const auto owner=words(deposit.work),historyLow=words(std::uint32_t(deposit.reflectionHistory)),historyHigh=words(std::uint32_t(deposit.reflectionHistory>>32));
            packedDeposit[0]={owner[0],owner[1],float(deposit.interfaces),float(deposit.cameraVisible)};
            packedDeposit[1]={historyLow[0],historyLow[1],historyHigh[0],historyHigh[1]};
            packedDeposit[2]={deposit.fraction,deposit.incidentPower,deposit.outgoingPower,deposit.absorbedPower};
            packedDeposit[3]={float(deposit.footprint.projected),float(deposit.footprint.x),float(deposit.footprint.y),deposit.footprint.linearDepth};
            packedDeposit[4]={deposit.footprint.worldArea,deposit.raster.linearDepth,0,0};
            const auto object=words(deposit.raster.object),material=words(deposit.raster.material),instance=words(deposit.raster.instance),triangle=words(deposit.raster.triangle);
            packedDeposit[5]={object[0],object[1],material[0],material[1]};packedDeposit[6]={instance[0],instance[1],triangle[0],triangle[1]};
            hit(packedDeposit.data()+7,deposit.receiver);hit(packedDeposit.data()+12,deposit.cameraHit);hit(packedDeposit.data()+17,deposit.pixelHit);
            const auto ancestry=gc::macroAncestryTexels(deposit.ancestry);std::copy(ancestry.begin(),ancestry.end(),packedDeposit.begin()+22);
            gc::NativeDeposit decodedDeposit;
            ok&=expect(gc::opengl::causticDecodeDeposit(packedDeposit.data(),path.multipleScatterVersion,decodedDeposit)
                &&decodedDeposit.ancestry==deposit.ancestry,"v2 packed deposit decoder preserves every exact ancestry byte");
            for(unsigned malformed=0;malformed<5;++malformed) {
                auto corrupt=packedDeposit;gc::NativeDeposit sentinel;sentinel.work=9999;
                if(malformed==0)corrupt[22][0]+=.5f;
                if(malformed==1)corrupt[22][0]=128;
                if(malformed==2)corrupt[30][0]=1;
                if(malformed==3)corrupt[31][3]=std::numeric_limits<float>::infinity();
                const auto version=malformed==4?0u:path.multipleScatterVersion;
                ok&=expect(!gc::opengl::causticDecodeDeposit(corrupt.data(),version,sentinel)&&sentinel.work==9999,
                    "v2 packed deposit decoder rejects wrong owner version, invalid byte tokens and nonzero tails atomically");
            }
        }
        gc::EnergyLedger checked;for(unsigned malformed=0;malformed<8;++malformed) {
            auto corrupted=*receipt;
            if(malformed==0)corrupted.paths[0].multipleScatterVersion=0;
            if(malformed==1)++corrupted.paths[0].microDraws;
            if(malformed==2)corrupted.paths[0].quadrature=16;
            if(malformed==3)corrupted.paths[0].microResidualPower=corrupted.paths[0].launchedPower;
            if(malformed==4)corrupted.deposits[0].ancestry[31]=1;
            if(malformed==5)corrupted.deposits[0].ancestry[0]^=64;
            if(malformed==6)corrupted.deposits[0].ancestry[0]=128;
            if(malformed==7)corrupted.deposits.push_back(corrupted.deposits[0]);
            ok&=expect(!gc::validateNativeReadback(fixture.prepared,fixture.frame,corrupted,checked,error,&receipt->deposition.submission),
                "v2 actual native owner refuses version/count/residual/ancestry/parity/suffix/duplicate corruption atomically");
        }
    }
    // Exact matched media isolate versioned packing and shader-cache controls.
    // They supplement the non-matched positive cases above.
    auto matched=make(.35f);matched.referenceIor=1;matched.dispersion=0;
    matched.program->parameters.transmissionIorClearcoat={1,1,0,0};
    if(!schedule(matched,32,32,micro))return expect(false,error.c_str());
    auto matchedPreparation=backend.prepare(matched.scene,matched.program);
    auto v2=render(matched,matchedPreparation,32,32,micro);
    auto v1=arbitgpu::renderOpenGlGlassCausticScene(matched.scene,matchedPreparation.resources,matched.ordinaryLight,
        matched.program->object,matched.program->authoredMaterial,*matched.policy,{gc::kScheduleVersion,1},32,32,runtime);
    auto v2again=render(matched,matchedPreparation,32,32,micro);
    arbitgpu::NativeRawPassPixels v1Color,v2Color,v2Repeated;
    if(!v2.rendered||!v1.rendered||!v2again.rendered||!v1.frame->readRawPass(renderpassoutput::Output::Color,v1Color,error)
        ||!v2.frame->readRawPass(renderpassoutput::Output::Color,v2Color,error)
        ||!v2again.frame->readRawPass(renderpassoutput::Output::Color,v2Repeated,error))return expect(false,error.c_str());
    ok&=expect(v1Color.bytes==v2Color.bytes&&v2Repeated.bytes==v2Color.bytes,
        "matched-media v1/v2 controls preserve physical Color while switching one immutable owner's cached shader version");
    for(unsigned version=0;version<2;++version) {
        std::shared_ptr<const gc::NativeReadback> receipt;gc::EnergyLedger energy;
        if(!arbitgpu::readOpenGlGlassCausticReceiptForFixture(*(version?v2.frame:v1.frame),receipt,energy,error))return expect(false,error.c_str());
        for(const auto& path:receipt->paths)ok&=expect(path.multipleScatterVersion==(version?2u:0u)
            &&path.microEvents==0&&path.microDraws==0&&path.microResidualPower==0,
            "matched-media controls retain exact separate v1/v2 receipt markers without synthetic micro work");
    }
    for(unsigned failure=0;failure<5;++failure) {
        auto fixture=make(.75f);auto selected=micro;
        if(failure==0)selected.maximumEvents=1;
        if(failure==1)selected.maximumExits=1;
        if(failure==2) {auto limits=fixture.policy->description();limits.maximumPending=1;
            const auto admitted=gc::AdmittedBranchPolicy::admit(limits,error);fixture.policy=std::make_shared<const gc::AdmittedBranchPolicy>(*admitted);}
        if(failure==3) {auto limits=fixture.policy->description();limits.maximumVisits=1;
            const auto admitted=gc::AdmittedBranchPolicy::admit(limits,error);fixture.policy=std::make_shared<const gc::AdmittedBranchPolicy>(*admitted);}
        if(failure==4)selected.version=3;
        auto preparation=backend.prepare(fixture.scene,fixture.program);auto refused=render(fixture,preparation,32,32,selected);
        ok&=expect(!refused.rendered&&!refused.frame&&!refused.error.empty(),"v2 event/exit/pending/visit/unknown-version failures publish no usable Color");
    }
    return ok;
}

bool authoredRoughCausticNativeChecks(arbitgpu::NativeFixtureSceneBackend& backend)
{
    int major=0,minor=0;glGetIntegerv(GL_MAJOR_VERSION,&major);glGetIntegerv(GL_MINOR_VERSION,&minor);
    if(major<4||(major==4&&minor<3))return true;
    const auto read=[](const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame,
        arbitgpu::glasscaustics::NativeReadback& output,arbitgpu::glasscaustics::EnergyLedger& energy,std::string& error) {
        std::shared_ptr<const arbitgpu::glasscaustics::NativeReadback> receipt;
        if(!frame||!arbitgpu::readOpenGlGlassCausticReceiptForFixture(*frame,receipt,energy,error)||!receipt)return false;
        output=*receipt;return true;
    };
    std::string error;
    const bool rendered=videohelper::tests::authoredroughcaustic::nativeCases(backend,
        videohelper::materialprogram::BackendTarget::OpenGl,read,error);
    return expect(rendered,error.c_str());
}

struct CallerState final
{
    GLint framebuffer = 0;
    GLint activeTexture = 0;
    GLint texture = 0;
    GLint texture0 = 0;
    GLint sampler0 = 0;
    GLint pixelUnpackBuffer = 0;
    GLint unpackAlignment = 0;
    GLint unpackRowLength = 0;
    GLint unpackSkipPixels = 0;
    GLint unpackSkipRows = 0;
    std::array<GLint, 4> viewport {};
    std::array<GLboolean, 4> colorMask {};
    std::array<GLfloat, 4> clearColor {};
    GLdouble clearDepth = 0.0;
    GLboolean blend = GL_FALSE;
    GLboolean scissor = GL_FALSE;
};

CallerState captureCallerState (const arbitgl::GlFuncs& gl)
{
    CallerState state;
    glGetIntegerv (GL_FRAMEBUFFER_BINDING, &state.framebuffer);
    glGetIntegerv (GL_ACTIVE_TEXTURE, &state.activeTexture);
    glGetIntegerv (GL_TEXTURE_BINDING_2D, &state.texture);
    gl.ActiveTexture (GL_TEXTURE0);
    glGetIntegerv (GL_TEXTURE_BINDING_2D, &state.texture0);
    glGetIntegerv (GL_SAMPLER_BINDING, &state.sampler0);
    gl.ActiveTexture (static_cast<GLenum> (state.activeTexture));
    glGetIntegerv (GL_PIXEL_UNPACK_BUFFER_BINDING, &state.pixelUnpackBuffer);
    glGetIntegerv (GL_UNPACK_ALIGNMENT, &state.unpackAlignment);
    glGetIntegerv (GL_UNPACK_ROW_LENGTH, &state.unpackRowLength);
    glGetIntegerv (GL_UNPACK_SKIP_PIXELS, &state.unpackSkipPixels);
    glGetIntegerv (GL_UNPACK_SKIP_ROWS, &state.unpackSkipRows);
    glGetIntegerv (GL_VIEWPORT, state.viewport.data());
    glGetBooleanv (GL_COLOR_WRITEMASK, state.colorMask.data());
    glGetFloatv (GL_COLOR_CLEAR_VALUE, state.clearColor.data());
    glGetDoublev (GL_DEPTH_CLEAR_VALUE, &state.clearDepth);
    state.blend = glIsEnabled (GL_BLEND);
    state.scissor = glIsEnabled (GL_SCISSOR_TEST);
    return state;
}

bool sameCallerState (const arbitgl::GlFuncs& gl, const CallerState& expected)
{
    const auto actual = captureCallerState (gl);
    return actual.framebuffer == expected.framebuffer
        && actual.activeTexture == expected.activeTexture
        && actual.texture == expected.texture
        && actual.texture0 == expected.texture0
        && actual.sampler0 == expected.sampler0
        && actual.pixelUnpackBuffer == expected.pixelUnpackBuffer
        && actual.unpackAlignment == expected.unpackAlignment
        && actual.unpackRowLength == expected.unpackRowLength
        && actual.unpackSkipPixels == expected.unpackSkipPixels
        && actual.unpackSkipRows == expected.unpackSkipRows
        && actual.viewport == expected.viewport
        && actual.colorMask == expected.colorMask
        && actual.clearColor == expected.clearColor
        && actual.clearDepth == expected.clearDepth
        && actual.blend == expected.blend
        && actual.scissor == expected.scissor;
}

std::vector<std::uint8_t> readRgba8 (
    const arbitgl::GlFuncs& gl,
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame)
{
    GLint previousFramebuffer = 0;
    GLint previousPackAlignment = 0;
    glGetIntegerv (GL_FRAMEBUFFER_BINDING, &previousFramebuffer);
    glGetIntegerv (GL_PACK_ALIGNMENT, &previousPackAlignment);

    GLuint framebuffer = 0;
    gl.GenFramebuffers (1, &framebuffer);
    gl.BindFramebuffer (GL_FRAMEBUFFER, framebuffer);
    gl.FramebufferTexture2D (
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
        static_cast<GLuint> (frame->colorImageHandle()), 0);

    std::vector<std::uint8_t> pixels (
        static_cast<std::size_t> (frame->width()) * frame->height() * 4u);
    if (gl.CheckFramebufferStatus (GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
    {
        glPixelStorei (GL_PACK_ALIGNMENT, 1);
        glReadPixels (0, 0, static_cast<GLsizei> (frame->width()),
                      static_cast<GLsizei> (frame->height()),
                      GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    }

    glPixelStorei (GL_PACK_ALIGNMENT, previousPackAlignment);
    gl.BindFramebuffer (GL_FRAMEBUFFER, static_cast<GLuint> (previousFramebuffer));
    gl.DeleteFramebuffers (1, &framebuffer);
    return pixels;
}

std::uint64_t fnv1a64 (const std::vector<std::uint8_t>& bytes)
{
    std::uint64_t value = 14695981039346656037ull;
    for (const auto byte : bytes)
    {
        value ^= byte;
        value *= 1099511628211ull;
    }
    return value;
}

void reportCoverage (const char* name, const std::vector<std::uint8_t>& pixels)
{
    std::size_t background = 0;
    std::size_t nonBackground = 0;
    std::array<std::uint8_t, 4> firstDrawn {};
    bool captured = false;
    for (std::size_t index = 0; index + 3 < pixels.size(); index += 4)
    {
        const bool clear = pixels[index] == 7u && pixels[index + 1] == 10u
            && pixels[index + 2] == 18u && pixels[index + 3] == 255u;
        clear ? ++background : ++nonBackground;
        if (! clear && ! captured)
        {
            std::copy_n (pixels.begin() + static_cast<std::ptrdiff_t> (index), 4,
                         firstDrawn.begin());
            captured = true;
        }
    }
    std::cerr << "coverage " << name << " background=" << background
              << " nonBackground=" << nonBackground
              << " first=" << static_cast<unsigned> (firstDrawn[0]) << ','
              << static_cast<unsigned> (firstDrawn[1]) << ','
              << static_cast<unsigned> (firstDrawn[2]) << ','
              << static_cast<unsigned> (firstDrawn[3])
              << " hash=" << std::hex << fnv1a64 (pixels) << std::dec << '\n';
}

// Residency ledger arithmetic and charge ownership. No GL context is needed.
bool residencyLedgerChecks()
{
    namespace rl = arbitgpu::residency;
    const auto before = rl::ResidencyLedger::instance().snapshot();
    // True when the ledger holds exactly the baseline plus the given live charges.
    const auto holds = [&before] (rl::BackingClass backing, std::uint64_t bytes, std::uint64_t objects)
    {
        const auto live = rl::ResidencyLedger::instance().snapshot();
        const auto& base = before.classes[static_cast<std::size_t> (backing)];
        const auto& now = live.classes[static_cast<std::size_t> (backing)];
        return now.bytes == base.bytes + bytes && now.objects == base.objects + objects;
    };
    bool passed = true;

    passed &= expect (rl::nominalBufferBytes (144) == 144,
                      "buffer charges equal the requested length");
    passed &= expect (rl::nominalTextureBytes (64, 32, 1, 1, 16, 1) == 64u * 32u * 16u,
                      "a single-level RGBA32F texture charges width x height x texel bytes");
    passed &= expect (rl::nominalTextureBytes (2, 2, 1, 2, 4, 1) == 20,
                      "a 2x2 RGBA8 mip chain charges both levels");
    passed &= expect (rl::nominalTextureBytes (4, 4, 1, 3, 4, 1) == (16u + 4u + 1u) * 4u,
                      "three mip levels halve each extent and clamp to one texel");
    passed &= expect (rl::nominalTextureBytes (8, 8, 6, 1, 4, 4) == 8u * 8u * 6u * 4u * 4u,
                      "layers and samples multiply the level bytes");
    passed &= expect (rl::nominalTextureBytes (4, 4, 1, 0, 4, 0) == 4u * 4u * 4u,
                      "zero mip and sample counts mean one level and one sample");
    passed &= expect (rl::nominalTextureBytes (~std::uint64_t { 0 }, 2, 1, 1, 4, 1) == ~std::uint64_t { 0 },
                      "an impossible descriptor saturates instead of wrapping");

    {
        rl::ResidencyCharge first (rl::BackingClass::fixtureScene, 100);
        passed &= expect (holds (rl::BackingClass::fixtureScene, 100, 1),
                          "a charge adds its bytes and one object to its class");
        rl::ResidencyCharge moved (std::move (first));
        passed &= expect (! first.active() && first.bytes() == 0 && moved.active() && moved.bytes() == 100
                              && holds (rl::BackingClass::fixtureScene, 100, 1),
                          "a move transfers the charge without creating a second one");
    }
    passed &= expect (holds (rl::BackingClass::fixtureScene, 0, 0),
                      "destroying the moved-from and moved-to charges releases the bytes once");

    {
        rl::ResidencyCharge replaced (rl::BackingClass::deformation, 50);
        replaced = rl::ResidencyCharge (rl::BackingClass::deformation, 70);
        passed &= expect (holds (rl::BackingClass::deformation, 70, 1),
                          "move assignment releases the replaced charge exactly once");
        replaced.reset();
        replaced.reset();
        passed &= expect (holds (rl::BackingClass::deformation, 0, 0),
                          "reset is idempotent and releases the charge once");
    }

    {
        auto lease = arbitgpu::environmentimage::GpuBudget::acquire (4096);
        passed &= expect (lease != nullptr && holds (rl::BackingClass::environmentImage, 4096, 1),
                          "an admitted environment lease is charged as the environment-image class");
        lease.reset();
    }
    passed &= expect (holds (rl::BackingClass::environmentImage, 0, 0),
                      "the environment lease releases its class when the last owner drops");

    const auto totals = rl::ResidencyLedger::instance().snapshot();
    std::uint64_t summedBytes = 0;
    std::uint64_t summedObjects = 0;
    for (const auto& classTotals : totals.classes)
    {
        summedBytes += classTotals.bytes;
        summedObjects += classTotals.objects;
    }
    passed &= expect (totals.totalBytes == summedBytes && totals.totalObjects == summedObjects,
                      "snapshot totals equal the sum of the per-class totals");
    return passed;
}

// A real OpenGL preparation charges its vertex, index, and texture storage at
// descriptor size. Releasing the last owner returns the class to its baseline.
bool residencyFixtureSceneOwnerChecks (arbitgpu::NativeFixtureSceneBackend& backend)
{
    namespace rl = arbitgpu::residency;
    const auto before = rl::ResidencyLedger::instance().snapshot();
    const auto& base = before.classes[static_cast<std::size_t> (rl::BackingClass::fixtureScene)];

    auto sceneValue = videohelper::fixture3d::makeScene();
    sceneValue.lightCount = 0;
    sceneValue.ambientColor = { 0.0f, 0.0f, 0.0f };
    auto scene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (sceneValue);

    auto material = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>();
    material->backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    material->object = videohelper::fixture3d::kCubeObjectId;
    material->bindingDigest = "opengl-fixture-emission-binding";
    material->programIdentity = "opengl-fixture-emission-program";
    material->baseColorSource =
        arbitgpu::NativeFixtureSurfaceMaterialProgram::BaseColorSource::ConstantLinear;
    material->parameters.baseColorMetallic = { 0.0f, 0.0f, 0.0f, 0.0f };
    material->parameters.emissionRoughness = { 0.8f, 0.08f, 0.02f, 0.5f };
    material->parameters.normalOpacity = { 0.0f, 0.0f, 1.0f, 1.0f };
    material->parameters.transmissionIorClearcoat = { 0.0f, 1.5f, 0.0f, 0.0f };
    material->parameters.identifiers = { videohelper::fixture3d::kCubeMaterialId.value, 0u, 0u, 0u };

    auto preparation = backend.prepare (scene, material);
    if (! expect (preparation.prepared && preparation.resources != nullptr, preparation.error.c_str()))
        return false;

    bool passed = true;
    {
        const auto live = rl::ResidencyLedger::instance().snapshot();
        const auto& held = live.classes[static_cast<std::size_t> (rl::BackingClass::fixtureScene)];
        // Two buffers: 24 vertices and 36 indices. Two textures: the white texel
        // and the 2x2 mipmapped cube texture, whose chain is 2x2 plus 1x1.
        const std::uint64_t expectedBytes =
            rl::nominalBufferBytes (scene->vertexCount * sizeof (HarmonicMIDI::grid::SceneVertex))
            + rl::nominalBufferBytes (scene->indexCount * sizeof (std::uint32_t))
            + rl::nominalTextureBytes (1, 1, 1, 1, sizeof (HarmonicMIDI::grid::SceneTexelRgba8), 1)
            + rl::nominalTextureBytes (2, 2, 1, 2, sizeof (HarmonicMIDI::grid::SceneTexelRgba8), 1);
        passed &= expect (held.bytes - base.bytes == expectedBytes,
                          "OpenGL fixture buffers and textures charge their descriptor bytes");
        passed &= expect (held.objects - base.objects == 4,
                          "OpenGL fixture resources charge two buffers and two textures");
    }

    preparation.resources.reset();
    const auto released = rl::ResidencyLedger::instance().snapshot();
    const auto& after = released.classes[static_cast<std::size_t> (rl::BackingClass::fixtureScene)];
    passed &= expect (after.bytes == base.bytes && after.objects == base.objects,
                      "releasing the last fixture owner returns its class to baseline");
    return passed;
}

} // namespace

int main()
{
    if (! residencyLedgerChecks())
        return 1;
    if (! glfwInit())
    {
        std::cerr << "SKIP: GLFW initialization failed\n";
        return 77;
    }

    glfwWindowHint (GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    auto* window = glfwCreateWindow (64, 64, "fixture-scene-backend-test", nullptr, nullptr);
    if (window == nullptr)
    {
        std::cerr << "SKIP: hidden OpenGL 3.3 core context creation failed\n";
        glfwTerminate();
        return 77;
    }
    glfwMakeContextCurrent (window);

    arbitgl::GlFuncs gl;
    std::string missing;
    if (! arbitgl::loadGlFunctions (gl, missing))
    {
        std::cerr << "FAIL: OpenGL function loading failed: " << missing << '\n';
        glfwDestroyWindow (window);
        glfwTerminate();
        return 1;
    }

    {
        // llvmpipe/softpipe JIT-compiles these shaders with unbounded memory growth.
        const auto* rendererRaw = reinterpret_cast<const char*> (glGetString (GL_RENDERER));
        const std::string renderer = rendererRaw != nullptr ? rendererRaw : "";
        std::string lowered = renderer;
        for (auto& c : lowered)
            if (c >= 'A' && c <= 'Z')
                c = static_cast<char> (c - 'A' + 'a');
        const bool softwareRenderer = lowered.find ("llvmpipe") != std::string::npos
            || lowered.find ("softpipe") != std::string::npos
            || lowered.find ("lavapipe") != std::string::npos
            || lowered.find ("swr") != std::string::npos;
        const auto* allowSoftware = std::getenv ("ARBIT_GL_TESTS_ALLOW_SOFTWARE");
        const bool allowed = allowSoftware != nullptr && std::string (allowSoftware) == "1";
        if (softwareRenderer && ! allowed)
        {
            std::cerr << "SKIP: native fixture-scene shaders need a hardware OpenGL driver (renderer: " << renderer << ")\n";
            glfwDestroyWindow (window);
            glfwTerminate();
            return 77;
        }
    }

    bool ok = true;
    ok &= exactLineNativeProbe();
    ok &= cameraRayCoefficientNativeProbe();
    ok &= cameraLinePredicateNativeProbe();
    ok &= cameraWideLinePredicateNativeProbe();
    ok &= cameraPixelSignNativeProbe();
    ok &= queryArithmeticSourceChecks();
    auto& backend = arbitgpu::nativeFixtureSceneBackend();
    ok &= expect(backend.supportsRetainedNoteMotion(),
                 "OpenGL fixture explicitly supports retained stable-note Motion");
    ok &= residencyFixtureSceneOwnerChecks (backend);
    const auto info = backend.info();
    ok &= glassGratingIncidentNativeChecks(backend);
    std::string terminalError;
    ok &= expect(glassGratingTerminalNativeChecks(backend,terminalError),terminalError.c_str());
    ok &= glassCausticNativeChecks(backend);
    ok &= roughGlassCausticNativeChecks(backend);
    ok &= multipleScatterGlassCausticNativeChecks(backend);
    ok &= authoredRoughCausticNativeChecks(backend);
    std::string glassError;
    const bool glassRendered=videohelper::tests::smoothglassfixture::run(backend,glassError);
    ok &= expect(glassRendered,glassError.c_str());
    std::string roughGlassError;
    const bool roughGlassRendered = videohelper::tests::openglroughglassfixture::run(backend,roughGlassError);
    ok &= expect(roughGlassRendered,roughGlassError.c_str());
    std::string waterError;
    const bool waterRendered=videohelper::tests::waterfixture::run(backend,waterError);
    ok &= expect(waterRendered,waterError.c_str());
    std::string roughWaterError;
    const bool roughWaterRendered=videohelper::tests::openglroughwaterfixture::run(backend,roughWaterError);
    ok &= expect(roughWaterRendered,roughWaterError.c_str());
    std::string environmentError;
    const bool environmentRendered=videohelper::tests::environmentImageCases(backend,environmentError);
    ok &= expect(environmentRendered,environmentError.c_str());
    std::string skyError;
    const bool skyRendered=videohelper::tests::analyticSkyCases(backend,skyError);
    ok &= expect(skyRendered,skyError.c_str());
    std::string reflectionError;
    const bool reflected = videohelper::tests::baseSurfaceReflectionCases(backend, reflectionError);
    ok &= expect(reflected, reflectionError.c_str());
    std::string visibilityError;
    const bool visible = videohelper::tests::opaqueVisibilityCases(backend, visibilityError);
    ok &= expect(visible, visibilityError.c_str());
    std::string linearError;
    const bool linearCaptured = linearscenechecks::capture(backend, linearError) != nullptr;
    ok &= expect(linearCaptured, linearError.c_str());
    ok &= expect (info.available && info.backend == "opengl",
                  "the production fixture backend must report the live OpenGL context");
    {
        videorender::ColorTransformGl transform;
        std::string error;
        colortransform::AdmissionFailure failure;
        const auto admitted = decodedframecolor::pqToMaterialLinear(
            {colortransform::ColorSpace::Rec2020,colortransform::TransferFunction::PQ},{2,2},failure);
        unsigned images[2] {};
        glGenTextures(2,images);
        const std::array<float,16> encoded {0.5f,0.5f,0.5f,1, 0.75f,0.75f,0.75f,1,
            0.5f,0,0,1, 0,0,0,1};
        for (int index = 0; index < 2; ++index) {
            glBindTexture(GL_TEXTURE_2D,images[index]);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
            glTexImage2D(GL_TEXTURE_2D,0,index == 0 ? GL_RGBA32F : GL_RGBA16F,2,2,0,
                GL_RGBA,GL_FLOAT,index == 0 ? encoded.data() : nullptr);
        }
        if (expect(admitted && transform.initialize(&gl,error)
            && transform.render(images[0],images[1],2,2,*admitted,error),"native OpenGL PQ material transform executes"))
        {
            std::array<float,16> actual {};
            glBindTexture(GL_TEXTURE_2D,images[1]); glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_FLOAT,actual.data());
            const std::array<float,16> oracle {0.45441236f,0.45441236f,0.45441236f,1,
                4.84422589f,4.84422589f,4.84422589f,1, 0.75454763f,-0.05659728f,-0.00824793f,1, 0,0,0,1};
            for (std::size_t component = 0; component < actual.size(); ++component)
                ok &= expect(std::abs(actual[component]-oracle[component]) < 0.008f,
                    "PQ RGBA16F material transform preserves numeric luminance, negatives and top-first rows");
            ok &= expect(actual[5] > 1 && actual[9] < 0 && actual[10] < 0,
                "native linear output cannot tone-map or clamp signed gamut values");
        } else ok = false;
        glBindTexture(GL_TEXTURE_2D,images[0]);
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,2,2,GL_RGBA,GL_FLOAT,videohelper::tests::hlg::encoded.data());
        for (const auto& sample : videohelper::tests::hlg::cases)
        {
            const auto hlg = decodedframecolor::hdrToMaterialLinear(
                {colortransform::ColorSpace::Rec2020,colortransform::TransferFunction::HLG},
                {2,2},videosourcecolor::Policy{sample.peak},failure);
            if (!expect(hlg && transform.render(images[0],images[1],2,2,*hlg,error),
                "HLG material executes the native reference-display OOTF")) { ok = false; continue; }
            std::array<float,16> actual {};
            glBindTexture(GL_TEXTURE_2D,images[1]); glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_FLOAT,actual.data());
            for (std::size_t i = 0; i < actual.size(); ++i)
                ok &= expect(std::abs(actual[i]-sample.linear[i]) < 0.002f*std::max(1.0f,std::abs(sample.linear[i])),
                    "HLG native RGBA16F values match independent BT.2100 luminance, signed gamut and row oracles");
            ok &= expect(actual[4] > 1 && actual[9] < 0 && actual[10] < 0,
                "HLG material highlights and signed gamut cannot be clipped");
        }
        transform.shutdown(); glDeleteTextures(2,images);
    }
    ok &= videohelper::tests::materialFrameBindingCases(backend,
        [&](const auto& frame) { return readRgba8(gl, frame); });
    std::string orientationError;
    ok &= expect(videohelper::tests::orientation::materialOrientationCases(backend, orientationError),
                 "embedded textures and native Frames preserve four-quadrant text orientation");
    if (!orientationError.empty()) std::cerr << orientationError << '\n';

    auto sceneValue = videohelper::fixture3d::makeScene();
    sceneValue.lightCount = 0;
    sceneValue.ambientColor = { 0.0f, 0.0f, 0.0f };
#if defined(ARBIT_STRICT_PACKAGED_ACCEPTANCE)
    sceneValue.objectCount = 2;
    sceneValue.objects[1] = sceneValue.objects[0];
    sceneValue.objects[1].id = HarmonicMIDI::grid::SceneObjectId { 2 };
    sceneValue.objects[1].transform.translation.x += 1.75f;
#endif
    auto scene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (sceneValue);

    auto material = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>();
    material->backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    material->object = videohelper::fixture3d::kCubeObjectId;
    material->bindingDigest = "opengl-fixture-emission-binding";
    material->programIdentity = "opengl-fixture-emission-program";
    material->baseColorSource =
        arbitgpu::NativeFixtureSurfaceMaterialProgram::BaseColorSource::ConstantLinear;
    material->parameters.baseColorMetallic = { 0.0f, 0.0f, 0.0f, 0.0f };
    material->parameters.emissionRoughness = { 0.8f, 0.08f, 0.02f, 0.5f };
    material->parameters.normalOpacity = { 0.0f, 0.0f, 1.0f, 1.0f };
    material->parameters.transmissionIorClearcoat = { 0.0f, 1.5f, 0.0f, 0.0f };
    material->parameters.identifiers = { videohelper::fixture3d::kCubeMaterialId.value, 0u, 0u, 0u };

    GLuint callerPixelUnpackBuffer = 0;
    std::array<std::uint8_t, 4096> callerPixelUnpackBytes {};
    gl.GenBuffers (1, &callerPixelUnpackBuffer);
    gl.BindBuffer (GL_PIXEL_UNPACK_BUFFER, callerPixelUnpackBuffer);
    gl.BufferData (GL_PIXEL_UNPACK_BUFFER,
                   static_cast<std::ptrdiff_t> (callerPixelUnpackBytes.size()),
                   callerPixelUnpackBytes.data(), GL_STATIC_DRAW);
    glPixelStorei (GL_UNPACK_ALIGNMENT, 8);
    glPixelStorei (GL_UNPACK_ROW_LENGTH, 7);
    glPixelStorei (GL_UNPACK_SKIP_PIXELS, 2);
    glPixelStorei (GL_UNPACK_SKIP_ROWS, 3);
    const auto dirtyUploadState = captureCallerState (gl);

    auto preparation = backend.prepare (scene, material);
    if (! preparation.prepared)
        std::cerr << "OpenGL fixture preparation error: " << preparation.error << '\n';
    ok &= expect (preparation.prepared && preparation.resources != nullptr,
                  "the exact bounded no-light emission scene must prepare on OpenGL");
    ok &= expect (sameCallerState (gl, dirtyUploadState),
                  "fixture preparation must isolate and restore caller pixel-unpack state");
    ok &= expect (preparation.stats.staticUploadCount == 1
                      && preparation.stats.materialProgramUploadCount == 1
                      && preparation.stats.vertexBytes == sceneValue.vertexCount
                                                         * sizeof (HarmonicMIDI::grid::SceneVertex)
                      && preparation.stats.indexBytes == sceneValue.indexCount
                                                        * sizeof (std::uint32_t),
                  "preparation must report exact bounded resource uploads");
    if (! preparation.prepared || preparation.resources == nullptr)
    {
        gl.DeleteBuffers (1, &callerPixelUnpackBuffer);
        glfwDestroyWindow (window);
        glfwTerminate();
        return 1;
    }

    GLuint callerTexture = 0;
    GLuint callerSampler = 0;
    GLuint callerFramebuffer = 0;
    glGenTextures (1, &callerTexture);
    gl.GenSamplers (1, &callerSampler);
    gl.SamplerParameteri (callerSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    gl.SamplerParameteri (callerSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    gl.ActiveTexture (GL_TEXTURE0);
    glBindTexture (GL_TEXTURE_2D, callerTexture);
    gl.BindSampler (0, callerSampler);
    gl.ActiveTexture (GL_TEXTURE1);
    glBindTexture (GL_TEXTURE_2D, callerTexture);
    gl.GenFramebuffers (1, &callerFramebuffer);
    gl.BindFramebuffer (GL_FRAMEBUFFER, callerFramebuffer);
    glViewport (3, 5, 17, 19);
    glEnable (GL_BLEND);
    glEnable (GL_SCISSOR_TEST);
    glScissor (1, 2, 3, 4);
    glColorMask (GL_FALSE, GL_TRUE, GL_FALSE, GL_TRUE);
    glClearColor (0.6f, 0.4f, 0.2f, 0.8f);
    glClearDepth (0.25);
    while (glGetError() != GL_NO_ERROR) {}
    const auto dirtyState = captureCallerState (gl);

    arbitgpu::NativeFixtureSceneRuntimeInputs noteRuntime;
    auto first = backend.render (scene, preparation.resources, 64, 64, noteRuntime);
    if (! first.rendered)
        std::cerr << "OpenGL fixture render error: " << first.error << '\n';
    ok &= expect (first.rendered && first.frame != nullptr,
                  "the exact bounded scene must produce a native OpenGL frame");
    ok &= expect (first.stats.drawCount == sceneValue.objectCount
                      && first.stats.reusedStaticResources
                      && first.stats.reusedMaterialProgram,
                  "the draw must report every object submission and prepared-resource reuse");
    ok &= expect (sameCallerState (gl, dirtyState),
                  "fixture rendering must restore caller OpenGL state exactly");
    if (! first.rendered || first.frame == nullptr)
    {
        preparation.resources.reset();
        gl.DeleteBuffers (1, &callerPixelUnpackBuffer);
        gl.DeleteFramebuffers (1, &callerFramebuffer);
        glDeleteTextures (1, &callerTexture);
        gl.DeleteSamplers (1, &callerSampler);
        glfwDestroyWindow (window);
        glfwTerminate();
        return 1;
    }

    glDisable (GL_SCISSOR_TEST);
    glDisable (GL_BLEND);
    glColorMask (GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    gl.BindFramebuffer (GL_FRAMEBUFFER, 0);
    gl.ActiveTexture (GL_TEXTURE0);
    const auto firstPixels = readRgba8 (gl, first.frame);
    reportCoverage ("constant-emission", firstPixels);

    // Pin the camera and object so the centre pixel has a known world normal.
    auto outputSceneValue = sceneValue;
    outputSceneValue.objectCount = 1;
    outputSceneValue.objects[0].transform = {};
    outputSceneValue.cameras[0].transform = {};
    outputSceneValue.cameras[0].transform.translation.z = 5.0f;
    auto outputScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (outputSceneValue);
    auto outputResources = backend.prepare (outputScene, material);
    ok &= expect (outputResources.prepared, "output inspection scene prepares");
    for (const auto output : { renderpassoutput::Output::Depth, renderpassoutput::Output::Normal,
                              renderpassoutput::Output::Motion,
                              renderpassoutput::Output::Emission, renderpassoutput::Output::Mask,
                              renderpassoutput::Output::MaterialId, renderpassoutput::Output::ObjectId })
    {
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
        inputs.imageOutput = output;
        const auto frame = backend.render (outputScene, outputResources.resources, 64, 64, inputs);
        ok &= expect (frame.rendered && frame.frame != nullptr, "each Render 3D inspection output draws");
        if (!frame.frame) continue;
        const auto pixels = readRgba8 (gl, frame.frame);
        const auto center = (32u * 64u + 32u) * 4u;
        std::array<float, 3> expected {};
        if (output == renderpassoutput::Output::Normal) expected = { 0.5f, 0.5f, 1.0f };
        if (output == renderpassoutput::Output::Motion) expected = { 0.5f, 0.5f, 0.5f };
        if (output == renderpassoutput::Output::Emission) expected = { 0.8f, 0.08f, 0.02f };
        if (output == renderpassoutput::Output::Mask) expected = { 1.0f, 1.0f, 1.0f };
        if (output == renderpassoutput::Output::MaterialId)
            expected = render3dimage::identityColor (outputSceneValue.objects[0].material.value);
        if (output == renderpassoutput::Output::ObjectId)
            expected = render3dimage::identityColor (outputSceneValue.objects[0].id.value);
        if (output == renderpassoutput::Output::Depth)
        {
            const auto& camera = outputSceneValue.cameras[0];
            const auto depth = (4.0f - camera.nearPlane) / (camera.farPlane - camera.nearPlane);
            expected = { depth, depth, depth };
        }
        for (std::size_t channel = 0; channel < 3; ++channel)
            ok &= expect (std::abs (static_cast<int> (pixels[center + channel])
                - static_cast<int> (std::lround (expected[channel] * 255.0f))) <= 1,
                "inspection pixels contain the requested native quantity");
        ok &= expect (pixels[center + 3u] == 255 && pixels[3] == 0,
                      "inspection preserves object coverage and transparent background");
        const auto repeated = backend.render (outputScene, outputResources.resources, 64, 64, inputs);
        ok &= expect (repeated.rendered && readRgba8 (gl, repeated.frame) == pixels
                          && repeated.stats.reusedStaticResources,
                      "inspection is deterministic and reuses native geometry");
    }
    auto rawSceneValue = outputSceneValue;
    rawSceneValue.objects[0].id.value = 0xf1234567u;
    rawSceneValue.materials[0].id.value = 0xe2345678u;
    rawSceneValue.objects[0].material = rawSceneValue.materials[0].id;
    rawSceneValue.materials[0].emissive = { 2.5f, 0.25f, 0.0f };
    auto rawScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(rawSceneValue);
    auto rawResources = backend.prepare(rawScene, nullptr);
    auto rawFrame = backend.render(rawScene, rawResources.resources, 64, 64, {});
    ok &= expect(rawFrame.rendered && rawFrame.frame != nullptr, "raw scene passes render together");
    if (rawFrame.frame)
    {
        const std::size_t center = 32u * 64u + 32u;
        const auto readMotion = [&](const auto& frame)
        {
            std::vector<float> values(64u * 64u * 2u, -99.0f);
            if (!frame) return values;
            const auto descriptor = frame->passTextureDescriptor(renderpassoutput::Output::Motion);
            ok &= expect(descriptor.complete() && descriptor.format == arbitgpu::NativeTexturePixelFormat::Rg16Float,
                "Motion is a retained two-channel half-float data attachment");
            glBindTexture(GL_TEXTURE_2D, static_cast<unsigned>(descriptor.textureViewHandle));
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_FLOAT, values.data());
            return values;
        };
        const auto still = readMotion(rawFrame.frame);
        ok &= expect(std::all_of(still.begin(), still.end(), [](float value) { return value == 0; }),
            "first frame publishes zero vectors including background");
        arbitgpu::NativeFixtureSceneRuntimeInputs moving;
        moving.previousMotion = arbitgpu::NativeSceneMotionSample {};
        moving.objectTranslationOffset[0] = 0.25f;
        moving.passComposite = renderpasscomposite::Parameters {};
        moving.passComposite->mode = renderpasscomposite::Mode::MotionView;
        moving.passComposite->amount = 0;
        const auto moved = backend.render(rawScene, rawResources.resources, 64, 64, moving);
        const auto movedAgain = backend.render(rawScene, rawResources.resources, 64, 64, moving);
        const auto vectors = readMotion(moved.frame);
        std::vector<float> depth(64u * 64u);
        if (moved.frame)
        {
            glBindTexture(GL_TEXTURE_2D, static_cast<unsigned>(moved.frame->depthTextureViewHandle()));
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, depth.data());
        }
        const auto& camera = rawScene->cameras[0];
        const auto distance = camera.nearPlane + depth[center] * (camera.farPlane - camera.nearPlane);
        const auto expectedPixels = 0.25f * 32.0f / (std::tan(camera.verticalFovRadians * 0.5f) * distance);
        ok &= expect(moved.rendered && movedAgain.rendered && moved.stats.drawCount == rawFrame.stats.drawCount
            && vectors == readMotion(movedAgain.frame) && readRgba8(gl, moved.frame) == readRgba8(gl, movedAgain.frame)
            && std::abs(vectors[center * 2u] - expectedPixels) < 0.01f
            && std::abs(vectors[center * 2u + 1u]) < 0.001f,
            "object motion uses signed pixel displacement and repeatable one-render inspection");
        auto directMotion = moving;
        directMotion.passComposite.reset();
        directMotion.imageOutput = renderpassoutput::Output::Motion;
        const auto direct = backend.render(rawScene, rawResources.resources, 64, 64, directMotion);
        ok &= expect(direct.rendered && readRgba8(gl, direct.frame) == readRgba8(gl, moved.frame)
            && readMotion(direct.frame) == vectors,
            "Motion Image selection matches the raw-pass inspector without changing its vectors");
        moving.objectTranslationOffset[0] = 0;
        moving.objectTranslationOffset[1] = 0.25f;
        const auto movedUp = backend.render(rawScene, rawResources.resources, 64, 64, moving);
        const auto upVectors = readMotion(movedUp.frame);
        const auto upImage = readRgba8(gl, movedUp.frame);
        ok &= expect(movedUp.rendered
            && std::abs(upVectors[center * 2u]) < 0.001f
            && std::abs(upVectors[center * 2u + 1u] - expectedPixels) < 0.01f
            && upImage[center * 4u + 1u] > upImage[center * 4u],
            "geometric Y-up Motion remains positive in RG16F and the Motion View image");
        moving.objectTranslationOffset[1] = 0;
        moving.objectTranslationOffset[0] = 0;
        moving.cameraTranslationOffset[0] = 0.25f;
        const auto cameraMoved = backend.render(rawScene, rawResources.resources, 64, 64, moving);
        const auto cameraVectors = readMotion(cameraMoved.frame);
        ok &= expect(cameraMoved.rendered && cameraVectors[center * 2u] < -0.1f,
            "camera translation produces the opposite signed motion");
        moving.cameraTranslationOffset[0] = 0;
        moving.previousMotion->objectTranslationOffset[0] = 1000000.0f;
        const auto extremeFrame = backend.render(rawScene, rawResources.resources, 64, 64, moving);
        const auto extremeMotion = readMotion(extremeFrame.frame);
        ok &= expect(extremeFrame.rendered && extremeMotion[center * 2u] == -65504.0f,
            "motion saturates at the finite RG16F range");
        moving.previousMotion.reset();
        const auto resetFrame = backend.render(rawScene, rawResources.resources, 64, 64, moving);
        const auto resetMotion = readMotion(resetFrame.frame);
        ok &= expect(std::all_of(resetMotion.begin(), resetMotion.end(), [](float value) { return value == 0; }),
            "reset suppresses motion even when the current transform is displaced");
        {
            // Timeline Geometry3D keeps each vertex's prior position. Every
            // vertex sat 0.25 units to the left one frame earlier, so the
            // projected displacement matches the rigid 0.25 offset above.
            const auto positions = [&](float offset)
            {
                auto values = std::make_shared<std::vector<HarmonicMIDI::grid::SceneVec3>>();
                for (std::size_t index = 0; index < rawScene->vertexCount; ++index)
                {
                    auto position = rawScene->vertices[index].position;
                    position.x += offset;
                    values->push_back(position);
                }
                return values;
            };
            arbitgpu::NativeFixtureSceneRuntimeInputs deformed;
            deformed.imageOutput = renderpassoutput::Output::Motion;
            deformed.previousMotion = arbitgpu::NativeSceneMotionSample {};
            deformed.previousMotion->scenePositions = positions(-0.25f);
            const auto deformedFrame = backend.render(rawScene, rawResources.resources, 64, 64, deformed);
            const auto deformedVectors = readMotion(deformedFrame.frame);
            ok &= expect(deformedFrame.rendered
                && std::abs(deformedVectors[center * 2u] - expectedPixels) < 0.01f
                && std::abs(deformedVectors[center * 2u + 1u]) < 0.001f,
                "a deformed vertex moves by the projected displacement from its retained prior position");
            deformed.previousMotion->scenePositions = positions(0.0f);
            const auto staticFrame = backend.render(rawScene, rawResources.resources, 64, 64, deformed);
            const auto staticVectors = readMotion(staticFrame.frame);
            ok &= expect(staticFrame.rendered && std::all_of(staticVectors.begin(), staticVectors.end(),
                    [](float value) { return value == 0; }),
                "a static deformation gives zero motion");
            auto shortened = std::make_shared<std::vector<HarmonicMIDI::grid::SceneVec3>>(*positions(0.0f));
            shortened->pop_back();
            deformed.previousMotion->scenePositions = shortened;
            const auto mismatched = backend.render(rawScene, rawResources.resources, 64, 64, deformed);
            ok &= expect(!mismatched.rendered
                && mismatched.error.find("prior Geometry3D positions") != std::string::npos,
                "prior positions with a different vertex count are refused");
        }
        {
            auto noteSceneValue = rawSceneValue;
            noteSceneValue.objects[0].transform = {};
            auto noteScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(noteSceneValue);
            const auto noteResources = backend.prepare(noteScene);
            auto score = std::make_shared<arbitmod::Score>();
            score->notationVersion = score->scoreRevision = 1; score->rootFreq = 440;
            for (int index=0;index<2;++index)
            {
                arbitmod::Note note;
                note.id = index==0 ? -1 : -102;
                note.trackId = index+1; note.lengthBeats = 4;
                note.durationSeconds = 2; note.velocity = 90; note.freqHz = 440;
                score->notes.push_back(note);
            }
            canonicalblockc::FrameKey key;
            key.projectGeneration=key.sourceGeneration=key.helperGeneration=1;
            key.backendGeneration=key.deviceGeneration=key.scoreGeneration=1;
            key.beatMapGeneration=key.fpsGeneration=key.loopGeneration=key.seekGeneration=1;
            key.fps=30;
            canonicalblockc::FrameProducer producer;
            arbitgpu::NativeFixtureSceneRuntimeInputs notes;
            notes.canonicalBlockCFrame = producer.evaluate(key,score,0);
            notes.noteInstanceMapping.emplace();
            notes.noteInstanceMapping->x=visualnoteinstancing::MappingAxis::Track;
            notes.noteInstanceMapping->xScale=.4f;
            notes.noteInstanceMapping->yScale=notes.noteInstanceMapping->zScale=0;
            notes.noteInstanceMapping->meshScale=.12f;
            notes.imageOutput=renderpassoutput::Output::Motion;
            const auto firstNotes = backend.render(noteScene,noteResources.resources,64,64,notes);
            const auto neutral = [](const auto& values) {
                return !values.empty() && std::all_of(values.begin(),values.end(),[](float value){return value==0;});
            };
            ok &= expect(firstNotes.rendered && neutral(readMotion(firstNotes.frame)),
                "first stable note instances produce neutral Motion");
            const auto batch = arbitgpu::prepareNativeNoteInstances(notes);
            notes.previousMotion = arbitgpu::sceneMotionSample(notes);
            auto reordered = std::make_shared<arbitgpu::NativeSceneNoteMotionTransforms>(*notes.previousMotion->noteTransforms);
            std::swap(reordered->identities[0],reordered->identities[1]);
            std::swap(reordered->transforms[0],reordered->transforms[1]);
            notes.previousMotion->noteTransforms = reordered;
            const auto reorderedFrame = backend.render(noteScene,noteResources.resources,64,64,notes);
            ok &= expect(reorderedFrame.rendered && neutral(readMotion(reorderedFrame.frame)),
                "note Motion matches signed stable identities across table reorders");
            auto priorNotes = std::make_shared<arbitgpu::NativeSceneNoteMotionTransforms>();
            priorNotes->count=1; priorNotes->identities[0]=batch.identities[0];
            priorNotes->transforms[0]=batch.transforms[0]; priorNotes->transforms[0][0]-=.25f;
            notes.previousMotion->noteTransforms=priorNotes;
            notes.objectTranslationOffset[0]=.1f;
            const auto changedNotes = backend.render(noteScene,noteResources.resources,64,64,notes);
            const auto changedVectors = readMotion(changedNotes.frame);
            const auto changedAgain = backend.render(noteScene,noteResources.resources,64,64,notes);
            ok &= expect(changedNotes.rendered && changedAgain.rendered
                && changedVectors==readMotion(changedAgain.frame),"stable note Motion repeats exact native vectors");
            if (changedNotes.frame && changedVectors.size()==64u*64u*2u)
            {
                std::vector<float> coverage(64u*64u);
                glBindTexture(GL_TEXTURE_2D,static_cast<GLuint>(changedNotes.frame->depthTextureViewHandle()));
                glGetTexImage(GL_TEXTURE_2D,0,GL_RED,GL_FLOAT,coverage.data());
                const auto& noteCamera=noteScene->cameras[0];
                const auto focal=32/std::tan(noteCamera.verticalFovRadians*.5f);
                const auto firstCenter=32+focal*(batch.transforms[0][0]+.1f)/(5-.12f);
                const auto secondCenter=32+focal*(batch.transforms[1][0]+.1f)/(5-.12f);
                std::size_t movedCount=0,newCount=0;
                for (std::size_t pixel=0;pixel<coverage.size();++pixel)
                {
                    if (coverage[pixel]>=1) continue;
                    const auto x=float(pixel%64)+.5f;
                    if (std::abs(x-firstCenter)<std::abs(x-secondCenter))
                    {
                        const auto distance=noteCamera.nearPlane+coverage[pixel]*(noteCamera.farPlane-noteCamera.nearPlane);
                        ok &= expect(std::abs(changedVectors[pixel*2]-.35f*focal/distance)<.015f,
                            "note and outer motion match independent signed pixel projection");
                        ++movedCount;
                    }
                    else
                    {
                        ok &= expect(changedVectors[pixel*2]==0 && changedVectors[pixel*2+1]==0,
                            "new note identities stay neutral despite outer object movement");
                        ++newCount;
                    }
                }
                ok &= expect(movedCount>0 && newCount>0,"native note fixture covers retained and new identities");
            }
            notes.previousMotion.reset();
            const auto resetNotes=backend.render(noteScene,noteResources.resources,64,64,notes);
            ok &= expect(resetNotes.rendered && neutral(readMotion(resetNotes.frame)),"note predecessor reset clears Motion");
            notes.previousMotion=arbitgpu::sceneMotionSample(notes);
            auto malformed=std::make_shared<arbitgpu::NativeSceneNoteMotionTransforms>(*notes.previousMotion->noteTransforms);
            malformed->transforms[0][0]=std::numeric_limits<float>::quiet_NaN();
            notes.previousMotion->noteTransforms=malformed;
            const auto refusedNotes=backend.render(noteScene,noteResources.resources,64,64,notes);
            ok &= expect(!refusedNotes.rendered && !refusedNotes.frame && !refusedNotes.error.empty(),
                "malformed predecessor note data fails without publishing Motion");
        }
        rawpasschecks::verify(rawFrame.frame, [&](bool condition, const char* message)
            { ok &= expect(condition, message); });
        for (const auto output : { renderpassoutput::Output::MaterialId, renderpassoutput::Output::ObjectId })
        {
            const auto descriptor = rawFrame.frame->passTextureDescriptor(output);
            std::vector<std::uint32_t> ids(64u * 64u);
            glBindTexture(GL_TEXTURE_2D, static_cast<unsigned>(descriptor.textureViewHandle));
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, ids.data());
            const auto expected = output == renderpassoutput::Output::MaterialId ? 0xe2345678u : 0xf1234567u;
            ok &= expect(descriptor.complete() && descriptor.format == arbitgpu::NativeTexturePixelFormat::R32Uint
                && ids[center] == expected && ids[0] == 0u, "raw IDs retain all 32 bits and zero background");
            arbitgpu::NativeRawPassPixels exported;
            std::string readError;
            ok &= expect(rawFrame.frame->readRawPass(output, exported, readError), "GL IDs export without palette encoding");
            for (std::uint32_t row = 0; row < 64 && exported.bytes.size() == 64u * 64u * 4u; ++row)
                ok &= expect(std::memcmp(exported.bytes.data() + row * 64u * 4u,
                    ids.data() + (63u - row) * 64u, 64u * 4u) == 0,
                    "GL ID readback flips rows into the shared top-first file convention");
        }
        for (const auto output : { renderpassoutput::Output::Normal, renderpassoutput::Output::Emission })
        {
            const auto descriptor = rawFrame.frame->passTextureDescriptor(output);
            std::vector<float> values(64u * 64u * 4u);
            glBindTexture(GL_TEXTURE_2D, static_cast<unsigned>(descriptor.textureViewHandle));
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, values.data());
            const std::array<float, 3> expected = output == renderpassoutput::Output::Normal
                ? std::array<float, 3> { 0, 0, 1 } : std::array<float, 3> { 2.5f, 0.25f, 0 };
            ok &= expect(descriptor.complete() && descriptor.format == arbitgpu::NativeTexturePixelFormat::Rgba16Float,
                "raw normals and emission retain floating-point descriptors");
            for (std::size_t channel = 0; channel < 3; ++channel)
                ok &= expect(std::abs(values[center * 4u + channel] - expected[channel]) < 0.001f,
                    "raw vectors and HDR emission are not display-encoded or clamped");
        }
        for (const auto mode : { renderpasscomposite::Mode::DepthFog, renderpasscomposite::Mode::NormalView,
                                renderpasscomposite::Mode::MaskView, renderpasscomposite::Mode::ObjectMatte })
        {
            arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
            inputs.passComposite = renderpasscomposite::Parameters {};
            inputs.passComposite->mode = mode;
            inputs.passComposite->farDepth = 0.001f;
            inputs.passComposite->fogColor = { 0.2f, 0.4f, 0.6f };
            inputs.passComposite->identity = 0xf1234567u;
            const auto composed = backend.render(rawScene, rawResources.resources, 64, 64, inputs);
            const auto pixels = composed.frame ? readRgba8(gl, composed.frame) : std::vector<std::uint8_t> {};
            ok &= expect(composed.rendered && composed.stats.reusedStaticResources
                && composed.stats.drawCount == rawFrame.stats.drawCount && pixels.size() == 64u * 64u * 4u,
                "typed pass fan-out and compositing reuse one scene draw");
            if (pixels.size() != 64u * 64u * 4u) continue;
            const std::array<float, 3> expected = mode == renderpasscomposite::Mode::DepthFog
                ? std::array<float, 3> { 0.2f, 0.4f, 0.6f } : mode == renderpasscomposite::Mode::NormalView
                ? std::array<float, 3> { 0.5f, 0.5f, 1 } : std::array<float, 3> { 1, 1, 1 };
            for (std::size_t channel = 0; channel < 3; ++channel)
                ok &= expect(std::abs(static_cast<int>(pixels[center * 4u + channel])
                    - static_cast<int>(std::lround(expected[channel] * 255))) <= 1,
                    "pass composite consumes the requested depth, normal or integer matte");
            inputs.passComposite->mode = renderpasscomposite::Mode::ObjectMatte;
            inputs.passComposite->identity = 7;
            const auto wrongId = backend.render(rawScene, rawResources.resources, 64, 64, inputs);
            const auto rejectedPixels = wrongId.frame ? readRgba8(gl, wrongId.frame) : std::vector<std::uint8_t> {};
            ok &= expect(!rejectedPixels.empty() && rejectedPixels[center * 4u] == 0,
                "ID matte compares raw integer identity rather than palette color");
        }
    }
    {
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
        inputs.passProgram = renderpassfixture::hdrBranches();
        inputs.passComposite = inputs.passProgram->steps[inputs.passProgram->output].parameters;
        const auto composed = backend.render(rawScene, rawResources.resources, 64, 64, inputs);
        const auto pixels = composed.frame ? readRgba8(gl, composed.frame) : std::vector<std::uint8_t> {};
        const auto expected = renderpassfixture::expectedSdr();
        ok &= expect(composed.rendered && composed.stats.drawCount == rawFrame.stats.drawCount
            && pixels.size() == 64u * 64u * 4u, "HDR pass branches fuse into the retained scene draw");
        if (pixels.size() == 64u * 64u * 4u)
            for (std::size_t channel = 0; channel < 3; ++channel)
                ok &= expect(std::abs(static_cast<int>(pixels[(32u * 64u + 32u) * 4u + channel])
                    - expected[channel]) <= 1,
                    "linear HDR survives branch fan-out and sRGB fog decoding until final tone mapping");
        if (composed.frame)
        {
            const auto emission = composed.frame->passTextureDescriptor(renderpassoutput::Output::Emission);
            std::vector<float> raw(64u * 64u * 4u);
            glBindTexture(GL_TEXTURE_2D, static_cast<unsigned>(emission.textureViewHandle));
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, raw.data());
            ok &= expect(emission.format == arbitgpu::NativeTexturePixelFormat::Rgba16Float
                && std::abs(raw[(32u * 64u + 32u) * 4u] - 2.5f) < 0.001f,
                "HDR branch output transforms leave the retained raw emission attachment unchanged");
        }
        const auto repeated = backend.render(rawScene, rawResources.resources, 64, 64, inputs);
        ok &= expect(repeated.rendered && readRgba8(gl, repeated.frame) == pixels,
            "HDR pass graph renders deterministically on repeated preview/export samples");
        inputs.passProgram->steps[2].inputA = 3;
        ok &= expect(!backend.render(rawScene, rawResources.resources, 64, 64, inputs).rendered,
            "forward references cannot enter the native pass program");
    }
    rawFrame.frame.reset(); rawResources.resources.reset();
    arbitgpu::NativeFixtureSceneRuntimeInputs unsupportedOutput;
    unsupportedOutput.imageOutput = renderpassoutput::Output::Count;
    ok &= expect (!backend.render (outputScene, outputResources.resources, 64, 64,
                                  unsupportedOutput).rendered,
                  "invalid output cannot masquerade as color");
    outputResources.resources.reset();
    outputSceneValue.materials[0].alphaMode = HarmonicMIDI::grid::SceneAlphaMode::Mask;
    outputSceneValue.materials[0].opacity = 0.25f;
    outputSceneValue.materials[0].alphaCutoff = 0.5f;
    auto cutoutScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(outputSceneValue);
    auto cutoutResources = backend.prepare(cutoutScene, nullptr);
    arbitgpu::NativeFixtureSceneRuntimeInputs cutoutInput;
    cutoutInput.imageOutput = renderpassoutput::Output::Mask;
    const auto cutoutFrame = backend.render(cutoutScene, cutoutResources.resources, 64, 64, cutoutInput);
    const auto cutoutPixels = cutoutFrame.rendered ? readRgba8(gl, cutoutFrame.frame)
                                                  : std::vector<std::uint8_t> {};
    ok &= expect(cutoutPixels.size() == 64u * 64u * 4u
                     && cutoutPixels[(32u * 64u + 32u) * 4u + 3u] == 0,
                 "Mask inspection respects the rendered material alpha cutoff");
    cutoutResources.resources.reset();

    auto second = backend.render (scene, preparation.resources, 64, 64, noteRuntime);
    if (! second.rendered)
        std::cerr << "OpenGL second fixture render error: " << second.error << '\n';
    ok &= expect (second.rendered && second.frame != nullptr,
                  "a second owner-local submission must produce a frame");

    arbitgpu::NativeFixtureSceneRuntimeInputs explicitLight;
    HarmonicMIDI::grid::SceneLightRecord lightOverride;
    lightOverride.id.value = 9001;
    lightOverride.kind = HarmonicMIDI::grid::SceneLightKind::Environment;
    lightOverride.color = { 0.2f, 0.4f, 0.6f };
    lightOverride.intensity = 2.0f;
    explicitLight.lightOverride = lightOverride;
    auto lightBound = backend.render (
        scene, preparation.resources, 64, 64, explicitLight);
    ok &= expect (lightBound.rendered && lightBound.frame != nullptr
                      && lightBound.frame->backend() == "opengl",
                  "an explicit immutable Light binding executes on OpenGL");

    explicitLight.lightOverride->id = {};
    const auto invalidLight = backend.render (
        scene, preparation.resources, 64, 64, explicitLight);
    ok &= expect (! invalidLight.rendered && invalidLight.frame == nullptr,
                  "OpenGL rejects a malformed Light binding before drawing");
    {
        auto value = videohelper::fixture3d::makeScene();
        value.ambientColor = {}; value.lightCount = 0;
        value.materials[0].baseColor = { 1, 1, 1 };
        value.materials[0].emissive = {}; value.materials[0].metallic = 0;
        auto litScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(std::move(value));
        auto lightResources = backend.prepare(litScene, nullptr);
        arbitgpu::NativeFixtureSceneRuntimeInputs dynamicLight;
        dynamicLight.lightOverride.emplace();
        dynamicLight.lightOverride->id.value = 91;
        dynamicLight.lightOverride->intensity = 4;
        const auto lit = backend.render(litScene, lightResources.resources, 64, 64, dynamicLight);
        dynamicLight.lightOverride->intensity = 0;
        const auto dark = backend.render(litScene, lightResources.resources, 64, 64, dynamicLight);
        dynamicLight.lightOverride->intensity = 4;
        const auto replay = backend.render(litScene, lightResources.resources, 64, 64, dynamicLight);
        const auto litPixels = lit.frame ? readRgba8(gl, lit.frame) : std::vector<std::uint8_t> {};
        const auto darkPixels = dark.frame ? readRgba8(gl, dark.frame) : std::vector<std::uint8_t> {};
        const auto replayPixels = replay.frame ? readRgba8(gl, replay.frame) : std::vector<std::uint8_t> {};
        ok &= expect(lit.rendered && dark.rendered && replay.rendered
            && !litPixels.empty() && litPixels != darkPixels && litPixels == replayPixels,
            "per-frame directional light overrides alter pixels and replay without uploading scene resources");
    }
    if (! second.rendered || second.frame == nullptr)
    {
        first.frame.reset();
        preparation.resources.reset();
        gl.DeleteBuffers (1, &callerPixelUnpackBuffer);
        gl.DeleteFramebuffers (1, &callerFramebuffer);
        glDeleteTextures (1, &callerTexture);
        gl.DeleteSamplers (1, &callerSampler);
        glfwDestroyWindow (window);
        glfwTerminate();
        return 1;
    }
    const auto secondPixels = readRgba8 (gl, second.frame);
    ok &= expect (first.frame->colorImageHandle() != second.frame->colorImageHandle(),
                  "simultaneously live submissions must own distinct fresh color textures");
    ok &= expect (firstPixels == secondPixels,
                  "instanced preview/export-equivalent submissions must match exactly");

    const auto center = (32u * 64u + 32u) * 4u;
    std::cerr << "constant-emission center="
              << static_cast<unsigned> (firstPixels[center]) << ','
              << static_cast<unsigned> (firstPixels[center + 1u]) << ','
              << static_cast<unsigned> (firstPixels[center + 2u]) << ','
              << static_cast<unsigned> (firstPixels[center + 3u]) << '\n';
    ok &= expect (firstPixels.size() > center + 3u
                      && firstPixels[center] > firstPixels[center + 1u] * 4u
                      && firstPixels[center] > 100u
                      && firstPixels[center + 3u] == 255u,
                  "the no-light scene must contain the native constant-emission pixels");

    const auto foregroundPixels = [] (const std::vector<std::uint8_t>& pixels)
    {
        const std::array<std::uint8_t, 4> background { 7, 10, 18, 255 };
        std::size_t count = 0;
        for (std::size_t offset = 0; offset + 3 < pixels.size(); offset += 4)
            if (!std::equal (pixels.begin() + static_cast<std::ptrdiff_t> (offset),
                             pixels.begin() + static_cast<std::ptrdiff_t> (offset + 4),
                             background.begin()))
                ++count;
        return count;
    };
    auto frontTriangleValue = sceneValue;
    frontTriangleValue.objectCount = 1;
    frontTriangleValue.objects[0].vertexCount = 3;
    frontTriangleValue.objects[0].indexCount = 3;
    frontTriangleValue.vertexCount = 3;
    frontTriangleValue.indexCount = 3;
    const auto frontTriangle = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
        frontTriangleValue);
    const auto frontPreparation = backend.prepare (frontTriangle, material);
    const auto frontFrame = backend.render (
        frontTriangle, frontPreparation.resources, 64, 64, {});
    const auto frontPixels = frontFrame.rendered
        ? readRgba8 (gl, frontFrame.frame) : std::vector<std::uint8_t> {};
    auto backTriangleValue = frontTriangleValue;
    std::swap (backTriangleValue.indices[1], backTriangleValue.indices[2]);
    const auto backTriangle = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
        backTriangleValue);
    const auto backPreparation = backend.prepare (backTriangle, material);
    const auto backFrame = backend.render (
        backTriangle, backPreparation.resources, 64, 64, {});
    const auto backPixels = backFrame.rendered
        ? readRgba8 (gl, backFrame.frame) : std::vector<std::uint8_t> {};
    auto mirroredTriangleValue = frontTriangleValue;
    mirroredTriangleValue.objects[0].transform.scale.x = -1.0f;
    const auto mirroredTriangle
        = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (mirroredTriangleValue);
    const auto mirroredPreparation = backend.prepare (mirroredTriangle, material);
    const auto mirroredFrame = backend.render (
        mirroredTriangle, mirroredPreparation.resources, 64, 64, {});
    const auto mirroredPixels = mirroredFrame.rendered
        ? readRgba8 (gl, mirroredFrame.frame) : std::vector<std::uint8_t> {};
    ok &= expect (frontFrame.rendered && foregroundPixels (frontPixels) > 0
                      && backFrame.rendered && foregroundPixels (backPixels) == 0
                      && mirroredFrame.rendered && foregroundPixels (mirroredPixels) > 0,
                  "single-sided OpenGL rendering rejects backfaces and preserves mirrored frontfaces");

    const auto oracle = surfacematerialstarteroracle::load (
        SURFACE_MATERIAL_STARTER_ORACLE_PATH);
    auto starterLightingSceneValue = sceneValue;
    starterLightingSceneValue.lightCount = 1;
    starterLightingSceneValue.objects[0].transform.rotation
        = { -0.16773126f, 0.25488700f, 0.04494346f, 0.95125124f };
    starterLightingSceneValue.ambientColor = { 0.12f, 0.12f, 0.12f };
    starterLightingSceneValue.lights[0].kind
        = HarmonicMIDI::grid::SceneLightKind::Directional;
    starterLightingSceneValue.lights[0].transform = {};
    starterLightingSceneValue.lights[0].transform.rotation
        = starterLightingSceneValue.objects[0].transform.rotation;
    starterLightingSceneValue.lights[0].color = { 1.0f, 1.0f, 1.0f };
    starterLightingSceneValue.lights[0].intensity = 0.88f;
    starterLightingSceneValue.materials[0].baseColorTexture = {};
    // Odd extent puts pixel (32,32), at window coordinate (32.5,32.5),
    // exactly on the camera's center ray. The earlier 64-wide frames have
    // a different stride and their center offset is not valid here.
    const auto starterCenter = (32u * 65u + 32u) * 4u;
    std::vector<std::uint8_t> sampledStarterPixels;
    for (std::size_t starterIndex = 0;
         starterIndex < surfacematerialstarterfixture::kPrograms.size(); ++starterIndex)
    {
        const auto& starter = surfacematerialstarterfixture::kPrograms[starterIndex];
        const auto& expected = oracle.materials[starterIndex];
        ok &= expect (expected.id == starter.id,
                      "generated starter programs and independent JSON oracle must share stable IDs");

        std::string generatedError;
        auto generatedScene = videohelper::tests::generatedSurfaceScene(starterLightingSceneValue, generatedError);
        ok &= expect(generatedScene.has_value(), "exact starter uses admitted generated Geometry Core scene transport");
        if (!generatedScene) continue;
        auto starterSceneValue = std::move(*generatedScene);
        starterSceneValue.materials[0].id = { starter.materialIdValue() };
        starterSceneValue.objects[0].material = { starter.materialIdValue() };
        const auto starterScene
            = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
                std::move (starterSceneValue));

        surfacematerialbinding::ImportedSceneMaterialRequest request;
        request.scene = starterScene->id;
        request.sceneRevision = 1;
        request.structuralRevision = 1;
        request.evaluationRevision = 1;
        request.programRevision = 1;
        request.sceneSnapshot = starterScene;
        request.program = starter.program();
        request.binding.targetKind
            = surfacematerialbinding::BindingTargetKind::ObjectOverride;
        request.binding.object = videohelper::fixture3d::kCubeObjectId;
        request.binding.surfaceMaterialRevision = 1;
        std::string admissionError;
        const auto admittedProgram = surfacematerial::admit (request.program, admissionError);
        if (admittedProgram.has_value())
            request.binding.surfaceMaterialDigest = admittedProgram->structuralDigest();
        const auto binding = videorender::fixture3d::admitSurfaceMaterialBinding (
            starterScene, request, videohelper::materialprogram::BackendTarget::OpenGl,
            admissionError);
        ok &= expect (binding != nullptr,
                      "each generated exact starter program must pass OpenGL product admission");
        if (binding == nullptr)
            continue;

        const auto& block = binding->nativeProgram()->parameters;
        const std::array<std::array<float, 4>, 4> actualFloatLanes {{
            block.baseColorMetallic, block.emissionRoughness,
            block.normalOpacity, block.transmissionIorClearcoat
        }};
        bool exactBlock = true;
        for (std::size_t lane = 0; lane < actualFloatLanes.size(); ++lane)
            for (std::size_t channel = 0; channel < 4; ++channel)
                exactBlock &= std::abs (actualFloatLanes[lane][channel]
                    - expected.pbr[lane * 4 + channel]) < 0.000001f;
        exactBlock &= block.identifiers[0]
            == static_cast<std::uint32_t> (expected.pbr[16]);
        ok &= expect (exactBlock,
                      "native OpenGL lowering must match the independent JSON PBR oracle");

        videorender::fixture3d::FixtureSceneRenderer previewOwner (backend);
        videorender::fixture3d::FixtureSceneRenderer exportOwner (backend);
        videorender::fixture3d::RenderedFrame preview;
        videorender::fixture3d::RenderedFrame exportFrame;
        const bool previewRendered = previewOwner.renderPreview (
            starterScene, binding, { 65, 65 },
            videorender::fixture3d::kNativeGpuCapability, preview, admissionError);
        const bool exportRendered = exportOwner.renderExport (
            starterScene, binding, { 65, 65 },
            videorender::fixture3d::kNativeGpuCapability, exportFrame, admissionError);
        const auto previewPixels = previewRendered
            ? readRgba8 (gl, preview.nativeFrame) : std::vector<std::uint8_t> {};
        const auto exportPixels = exportRendered
            ? readRgba8 (gl, exportFrame.nativeFrame) : std::vector<std::uint8_t> {};
        ok &= expect (previewRendered && exportRendered
                          && preview.use == videorender::fixture3d::RenderUse::Preview
                          && exportFrame.use == videorender::fixture3d::RenderUse::Export
                          && preview.nativeFrame != nullptr && exportFrame.nativeFrame != nullptr
                          && preview.nativeFrame->colorImageHandle()
                              != exportFrame.nativeFrame->colorImageHandle(),
                      "distinct production preview and export owners must render fresh OpenGL frames");
        ok &= expect (! previewPixels.empty() && previewPixels == exportPixels,
                      "preview and export owners must produce exact matching OpenGL pixels");
        if (starterIndex == 0)
        {
            auto vertexRequest = request;
            vertexRequest.vertexModifier = videohelper::test::audioNormalDisplacement();
            const auto vertexBinding = videorender::fixture3d::admitSurfaceMaterialBinding(
                starterScene, vertexRequest, videohelper::materialprogram::BackendTarget::OpenGl,
                admissionError);
            if (vertexBinding)
            {
                backend.releaseCachedProgramsForCurrentContext();
                auto changedRequest = vertexRequest;
                changedRequest.vertexModifier = videohelper::test::audioNormalDisplacement(0.5);
                const auto changedBinding = videorender::fixture3d::admitSurfaceMaterialBinding(
                    starterScene, changedRequest, videohelper::materialprogram::BackendTarget::OpenGl,
                    admissionError);
                ok &= expect(changedBinding != nullptr, "changed vertex source admits independently");
                if (changedBinding)
                {
                    const std::array<std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram>, 5> programs {{
                        {}, vertexBinding->nativeProgram(), changedBinding->nativeProgram(),
                        vertexBinding->nativeProgram(), {}
                    }};
                    for (std::size_t index = 0; index < programs.size(); ++index)
                    {
                        // Each preparation is destroyed before the next lookup.
                        const auto preparation = backend.prepare(starterScene, programs[index]);
                        ok &= expect(preparation.prepared
                            && preparation.stats.shaderProgramBuildCount == (index < 3 ? 1u : 0u),
                            "program cache must distinguish exact vertex sources across resource gaps");
                    }
                }
            }
            const auto prepared = vertexBinding ? backend.prepare(starterScene, vertexBinding->nativeProgram())
                                                : arbitgpu::NativeFixtureScenePreparation {};
            ok &= expect(vertexBinding != nullptr && prepared.prepared,
                         "audio vertex material reaches OpenGL shader compilation");
            if (vertexBinding)
            {
                auto mismatched = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(
                    *vertexBinding->nativeProgram());
                mismatched->programIdentity = binding->nativeProgram()->programIdentity;
                ok &= expect(!backend.prepare(starterScene, mismatched).prepared,
                             "vertex shader identity cannot be substituted after material admission");
            }
            if (prepared.prepared)
            {
                arbitgpu::NativeFixtureSceneRuntimeInputs quiet, loud;
                loud.vertexSpectrum[6] = 0.5f;
                const auto quietFrame = backend.render(starterScene, prepared.resources, 65, 65, quiet);
                const auto loudFrame = backend.render(starterScene, prepared.resources, 65, 65, loud);
                const auto replayFrame = backend.render(starterScene, prepared.resources, 65, 65, quiet);
                const auto quietPixels = quietFrame.rendered ? readRgba8(gl, quietFrame.frame) : std::vector<std::uint8_t> {};
                const auto loudPixels = loudFrame.rendered ? readRgba8(gl, loudFrame.frame) : std::vector<std::uint8_t> {};
                ok &= expect(!quietPixels.empty() && quietPixels == previewPixels
                             && !loudPixels.empty() && loudPixels != quietPixels
                             && replayFrame.rendered && readRgba8(gl, replayFrame.frame) == quietPixels,
                             "GPU displacement follows the audio frame and seeking to silence restores original geometry");
                loud.vertexSpectrum[6] = 1.0f;
                const auto boundedFrame = backend.render(starterScene, prepared.resources, 65, 65, loud);
                ok &= expect(boundedFrame.rendered && readRgba8(gl, boundedFrame.frame) == loudPixels,
                             "audio above the displacement bound cannot move GPU vertices farther");
                videorender::fixture3d::RenderedFrame vertexExport;
                const auto exported = exportOwner.renderExport(starterScene, vertexBinding, {}, loud,
                    {65, 65}, videorender::fixture3d::kNativeGpuCapability, vertexExport, admissionError);
                ok &= expect(exported && readRgba8(gl, vertexExport.nativeFrame) == loudPixels,
                             "vertex displacement preview and export produce the same native pixels");
                {
                    // Motion runs the vertex program again at the prior frame's
                    // spectrum. Compare it with the same displacement given as
                    // retained prior positions on a pre-displaced mesh.
                    const auto readMotion = [&](const auto& frame)
                    {
                        std::vector<float> values(65u * 65u * 2u, -99.0f);
                        if (!frame) return values;
                        const auto descriptor = frame->passTextureDescriptor(renderpassoutput::Output::Motion);
                        if (!descriptor.complete()) return values;
                        glBindTexture(GL_TEXTURE_2D, static_cast<unsigned>(descriptor.textureViewHandle));
                        glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_FLOAT, values.data());
                        return values;
                    };
                    auto steady = loud;
                    steady.imageOutput = renderpassoutput::Output::Motion;
                    steady.previousMotion = arbitgpu::sceneMotionSample(loud);
                    const auto steadyFrame = backend.render(starterScene, prepared.resources, 65, 65, steady);
                    const auto steadyMotion = readMotion(steadyFrame.frame);
                    ok &= expect(steadyFrame.rendered && std::all_of(steadyMotion.begin(), steadyMotion.end(),
                            [](float value) { return value == 0; }),
                        "a vertex program with unchanged time and audio gives zero motion");
                    auto rising = steady;
                    rising.previousMotion->vertexSpectrum[6] = 0.0f;
                    const auto risingFrame = backend.render(starterScene, prepared.resources, 65, 65, rising);
                    const auto risingMotion = readMotion(risingFrame.frame);
                    const auto repeatedMotion = readMotion(
                        backend.render(starterScene, prepared.resources, 65, 65, rising).frame);
                    auto displacedValue = *starterScene;
                    std::vector<HarmonicMIDI::grid::SceneVec3> original;
                    for (std::size_t index = 0; index < displacedValue.vertexCount; ++index)
                    {
                        // audioNormalDisplacement at full audio: normal * 0.75,
                        // bounded to length 0.25.
                        auto& vertex = displacedValue.vertices[index];
                        original.push_back(vertex.position);
                        const auto& n = vertex.normal;
                        const float length = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z) * 0.75f;
                        const float scale = 0.75f * std::min(1.0f, 0.25f / std::max(length, 0.000001f));
                        vertex.position = { vertex.position.x + n.x * scale,
                            vertex.position.y + n.y * scale, vertex.position.z + n.z * scale };
                    }
                    const auto displacedScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(displacedValue);
                    const auto displacedPrepared = backend.prepare(displacedScene, nullptr);
                    arbitgpu::NativeFixtureSceneRuntimeInputs retained;
                    retained.imageOutput = renderpassoutput::Output::Motion;
                    retained.previousMotion = arbitgpu::NativeSceneMotionSample {};
                    retained.previousMotion->scenePositions
                        = std::make_shared<const std::vector<HarmonicMIDI::grid::SceneVec3>>(original);
                    const auto retainedFrame = displacedPrepared.prepared
                        ? backend.render(displacedScene, displacedPrepared.resources, 65, 65, retained)
                        : arbitgpu::NativeFixtureSceneSubmission {};
                    const auto retainedMotion = readMotion(retainedFrame.frame);
                    std::size_t movingValues = 0, different = 0;
                    for (std::size_t index = 0; index < risingMotion.size(); ++index)
                    {
                        movingValues += std::abs(risingMotion[index]) > 0.1f ? 1u : 0u;
                        different += std::abs(risingMotion[index] - retainedMotion[index]) > 0.05f ? 1u : 0u;
                    }
                    ok &= expect(risingFrame.rendered && retainedFrame.rendered && risingMotion == repeatedMotion
                            && movingValues > 0 && different * 100u <= risingMotion.size(),
                        "a vertex program's motion equals the projected displacement of its prior evaluation");
                    rising.previousMotion->scenePositions
                        = std::make_shared<const std::vector<HarmonicMIDI::grid::SceneVec3>>(original);
                    const auto combined = backend.render(starterScene, prepared.resources, 65, 65, rising);
                    ok &= expect(!combined.rendered
                            && combined.error.find("reads Normal over deformed geometry") != std::string::npos,
                        "a vertex program that reads Normal over retained deformed positions stays refused");

                    // A position and time program over a deformed mesh moving at a
                    // constant velocity. Motion must project
                    // program(current pose, t) - program(prior pose, t_prev). The
                    // expected motion comes from the retained-position path with
                    // both program results computed on the CPU.
                    auto timeRequest = request;
                    constexpr float rate = 0.6f, currentTime = 0.5f, priorTime = 0.25f;
                    const HarmonicMIDI::grid::SceneVec3 velocity {0.0f, 0.1f, 0.0f};
                    timeRequest.vertexModifier = videohelper::test::timeAxisDisplacement(rate);
                    const auto timeBinding = videorender::fixture3d::admitSurfaceMaterialBinding(
                        starterScene, timeRequest, videohelper::materialprogram::BackendTarget::OpenGl,
                        admissionError);
                    const auto timePrepared = timeBinding
                        ? backend.prepare(starterScene, timeBinding->nativeProgram())
                        : arbitgpu::NativeFixtureScenePreparation {};
                    ok &= expect(timeBinding != nullptr && timePrepared.prepared,
                        "a position and time vertex program reaches OpenGL shader compilation");
                    if (timePrepared.prepared)
                    {
                        std::vector<HarmonicMIDI::grid::SceneVec3> priorPose;
                        for (std::size_t index = 0; index < starterScene->vertexCount; ++index)
                        {
                            const auto& p = starterScene->vertices[index].position;
                            priorPose.push_back({p.x - velocity.x, p.y - velocity.y, p.z - velocity.z});
                        }
                        arbitgpu::NativeFixtureSceneRuntimeInputs deformed;
                        deformed.timeSeconds = currentTime;
                        deformed.imageOutput = renderpassoutput::Output::Motion;
                        deformed.previousMotion = arbitgpu::NativeSceneMotionSample {};
                        deformed.previousMotion->timeSeconds = priorTime;
                        deformed.previousMotion->scenePositions
                            = std::make_shared<const std::vector<HarmonicMIDI::grid::SceneVec3>>(priorPose);
                        const auto programFrame = backend.render(starterScene, timePrepared.resources, 65, 65, deformed);
                        if (!programFrame.rendered) std::cerr << "time program Motion: " << programFrame.error << '\n';
                        const auto programMotion = readMotion(programFrame.frame);

                        const auto retainedMotionOf = [&](float priorPoseWeight)
                        {
                            // priorPoseWeight 1 uses the prior pose, 0 the current pose
                            // as the input of the prior evaluation.
                            auto value = *starterScene;
                            std::vector<HarmonicMIDI::grid::SceneVec3> prior;
                            for (std::size_t index = 0; index < value.vertexCount; ++index)
                            {
                                auto& p = value.vertices[index].position;
                                prior.push_back({p.x - velocity.x * priorPoseWeight + rate * priorTime,
                                    p.y - velocity.y * priorPoseWeight, p.z - velocity.z * priorPoseWeight});
                                p.x += rate * currentTime;
                            }
                            const auto scene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(value);
                            const auto staticPrepared = backend.prepare(scene, nullptr);
                            arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
                            inputs.imageOutput = renderpassoutput::Output::Motion;
                            inputs.previousMotion = arbitgpu::NativeSceneMotionSample {};
                            inputs.previousMotion->scenePositions
                                = std::make_shared<const std::vector<HarmonicMIDI::grid::SceneVec3>>(prior);
                            return readMotion(staticPrepared.prepared
                                ? backend.render(scene, staticPrepared.resources, 65, 65, inputs).frame
                                : nullptr);
                        };
                        const auto expectedMotion = retainedMotionOf(1.0f);
                        const auto currentPoseMotion = retainedMotionOf(0.0f);
                        std::size_t moving = 0, different = 0, fromCurrentPose = 0;
                        for (std::size_t index = 0; index < programMotion.size(); ++index)
                        {
                            moving += std::abs(programMotion[index]) > 0.1f ? 1u : 0u;
                            different += std::abs(programMotion[index] - expectedMotion[index]) > 0.05f ? 1u : 0u;
                            fromCurrentPose += std::abs(expectedMotion[index] - currentPoseMotion[index]) > 0.05f ? 1u : 0u;
                        }
                        ok &= expect(programFrame.rendered && moving > 0 && different * 100u <= programMotion.size()
                                && fromCurrentPose * 100u > programMotion.size(),
                            "a vertex program over deformed geometry moves by the program applied to the prior pose");
                    }
                }
                loud.vertexSpectrum[6] = std::numeric_limits<float>::quiet_NaN();
                ok &= expect(!backend.render(starterScene, prepared.resources, 65, 65, loud).rendered,
                             "non-finite audio cannot reach a native vertex shader");
            }
        }
        if (previewPixels.size() > starterCenter + 3u)
        {
            const std::array<std::uint8_t, 4> actualCenter {{
                previewPixels[starterCenter], previewPixels[starterCenter + 1u],
                previewPixels[starterCenter + 2u], previewPixels[starterCenter + 3u]
            }};
            if (actualCenter != expected.centerRgba8)
            {
                std::cerr << starter.id << " sample (32,32) RGBA actual=";
                for (const auto channel : actualCenter) std::cerr << static_cast<unsigned>(channel) << ' ';
                std::cerr << "expected=";
                for (const auto channel : expected.centerRgba8) std::cerr << static_cast<unsigned>(channel) << ' ';
                std::cerr << '\n';
            }
            ok &= expect (actualCenter == expected.centerRgba8,
                          "OpenGL center pixels must match the pinned independent JSON oracle");
            ok &= expect (std::find (expected.rejectedAlternatives.begin(),
                                    expected.rejectedAlternatives.end(), actualCenter)
                              == expected.rejectedAlternatives.end(),
                          "OpenGL pixels must reject black-output and wrong-transfer shading alternatives");
            sampledStarterPixels.insert (sampledStarterPixels.end(),
                                         actualCenter.begin(), actualCenter.end());

            const auto rejectsControlledMaterialAlternative = [&] (const char* label,
                                                                    auto mutate)
            {
                auto alternative = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
                    *binding->nativeProgram());
                alternative->bindingDigest += label;
                alternative->programIdentity += label;
                mutate (alternative->parameters);
                const auto prepared = backend.prepare (starterScene, alternative);
                if (! prepared.prepared) return false;
                const auto rendered = backend.render (starterScene, prepared.resources, 65, 65, {});
                const auto pixels = rendered.rendered ? readRgba8 (gl, rendered.frame)
                                                      : std::vector<std::uint8_t> {};
                if (pixels.size() <= starterCenter + 3u) return false;
                const std::array<std::uint8_t, 4> value {{ pixels[starterCenter], pixels[starterCenter + 1u],
                                                           pixels[starterCenter + 2u], pixels[starterCenter + 3u] }};
                return value != expected.centerRgba8;
            };
            if (starterIndex == 0)
            {
                ok &= expect (rejectsControlledMaterialAlternative ("/base-color", [] (auto& p)
                    { p.baseColorMetallic[0] = 0.9f; }),
                    "OpenGL lit pixels must detect a wrong base-color binding");
                ok &= expect (rejectsControlledMaterialAlternative ("/roughness", [] (auto& p)
                    { p.emissionRoughness[3] = 0.0f; }),
                    "OpenGL lit pixels must detect a wrong roughness binding");
                ok &= expect (rejectsControlledMaterialAlternative ("/emission", [] (auto& p)
                    { p.emissionRoughness[0] = 0.25f; }),
                    "OpenGL lit pixels must detect a wrong emission binding");
            }
            if (starterIndex == 1)
                ok &= expect (rejectsControlledMaterialAlternative ("/metallic", [] (auto& p)
                    { p.baseColorMetallic[3] = 0.0f; }),
                    "OpenGL lit pixels must detect a wrong metallic binding");
            if (starterIndex == 2)
            {
                ok &= expect (rejectsControlledMaterialAlternative ("/opacity", [] (auto& p)
                    { p.normalOpacity[3] = 1.0f; }),
                    "OpenGL lit pixels must detect a wrong opacity binding");
                ok &= expect (rejectsControlledMaterialAlternative ("/transmission", [] (auto& p)
                    { p.transmissionIorClearcoat[0] = 0.0f; }),
                    "OpenGL lit pixels must detect a wrong transmission binding");
                ok &= expect (rejectsControlledMaterialAlternative ("/ior", [] (auto& p)
                    { p.transmissionIorClearcoat[1] = 2.8f; }),
                    "OpenGL lit pixels must detect a wrong IOR binding");
            }
            if (starterIndex == 3)
                ok &= expect (rejectsControlledMaterialAlternative ("/clearcoat", [] (auto& p)
                    { p.transmissionIorClearcoat[2] = 0.0f; }),
                    "OpenGL lit pixels must detect a wrong clearcoat binding");

            if (starterIndex != 0)
                continue;
            auto wrongNormalSceneValue = *starterScene;
            for (std::size_t vertex = 0; vertex < wrongNormalSceneValue.vertexCount; ++vertex)
                wrongNormalSceneValue.vertices[vertex].normal
                    = { -wrongNormalSceneValue.vertices[vertex].normal.x,
                        -wrongNormalSceneValue.vertices[vertex].normal.y,
                        -wrongNormalSceneValue.vertices[vertex].normal.z };
            const auto wrongNormalScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
                std::move (wrongNormalSceneValue));
            const auto wrongNormalPreparation = backend.prepare (wrongNormalScene, binding->nativeProgram());
            std::vector<std::uint8_t> wrongNormalPixels;
            if (wrongNormalPreparation.prepared)
            {
                const auto wrongNormalRender = backend.render (
                    wrongNormalScene, wrongNormalPreparation.resources, 65, 65, {});
                if (wrongNormalRender.rendered)
                    wrongNormalPixels = readRgba8 (gl, wrongNormalRender.frame);
            }
            ok &= expect (wrongNormalPixels.size() > starterCenter + 3u
                              && ! std::equal (actualCenter.begin(), actualCenter.end(),
                                               wrongNormalPixels.begin() + starterCenter),
                          "OpenGL lit pixels must detect a wrong transformed normal");
        }
    }
    const auto sampledDigest = fnv1a64 (sampledStarterPixels);
    char sampledDigestText[17] {};
    std::snprintf (sampledDigestText, sizeof (sampledDigestText), "%016llx",
                   static_cast<unsigned long long> (sampledDigest));
    ok &= expect (sampledStarterPixels.size() == 20
                      && oracle.sampledRgba8Fnv1a64 == sampledDigestText,
                  "all five OpenGL samples must match the pinned JSON digest");
    videohelper::Sha256 sampledSha256;
    sampledSha256.update (sampledStarterPixels.data(), sampledStarterPixels.size());
    ok &= expect (oracle.sampledRgba8Sha256 == sampledSha256.finishHex(),
                  "all five OpenGL samples must match the pinned SHA-256 digest");

    auto malformedStarter
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>();
    malformedStarter->backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    malformedStarter->object = videohelper::fixture3d::kCubeObjectId;
    malformedStarter->bindingDigest = "visual.material.malformed/binding/v1";
    malformedStarter->programIdentity = "visual.material.malformed/program/v1";
    malformedStarter->parameters = material->parameters;
    malformedStarter->parameters.emissionRoughness[0]
        = std::numeric_limits<float>::quiet_NaN();
    const auto malformedStarterPreparation = backend.prepare (scene, malformedStarter);
    ok &= expect (! malformedStarterPreparation.prepared
                      && malformedStarterPreparation.resources == nullptr
                      && malformedStarterPreparation.stats.staticUploadCount == 0
                      && malformedStarterPreparation.stats.materialProgramUploadCount == 0,
                  "malformed starter values must fail before native OpenGL allocation");

    auto wrongSceneValue = sceneValue;
    wrongSceneValue.objects[0].material = HarmonicMIDI::grid::SceneMaterialId { 999 };
    const auto wrongScene =
        std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (wrongSceneValue);
    const auto wrongPreparation = backend.prepare (wrongScene, material);
    ok &= expect (! wrongPreparation.prepared && wrongPreparation.resources == nullptr,
                  "a mismatched object/material identity must fail closed");

    auto composedSceneValue = sceneValue;
    composedSceneValue.objectCount = 2;
    composedSceneValue.materialCount = 2;
    composedSceneValue.objects[1] = composedSceneValue.objects[0];
    composedSceneValue.objects[1].id.value = 999;
    composedSceneValue.objects[1].parent = composedSceneValue.objects[0].id;
    // The local offset is the parent's inverse rotation of world +X, so the
    // child lands in a separate right-side draw region while exercising hierarchy.
    composedSceneValue.objects[1].transform.translation
        = { 2.5980762f, -0.5130302f, 1.4095389f };
    composedSceneValue.materials[1] = composedSceneValue.materials[0];
    composedSceneValue.materials[1].id.value = 998;
    composedSceneValue.materials[1].emissive = { 0.01f, 0.8f, 0.02f };
    composedSceneValue.objects[1].material = composedSceneValue.materials[1].id;
    auto objectZeroSceneValue = composedSceneValue;
    objectZeroSceneValue.objectCount = 1;
    auto objectZeroScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
        std::move (objectZeroSceneValue));
    const auto objectZeroPreparation = backend.prepare (objectZeroScene);
    const auto objectZeroFrame = backend.render (
        objectZeroScene, objectZeroPreparation.resources, 64, 64, {});
    const auto objectZeroPixels = objectZeroFrame.rendered
        ? readRgba8 (gl, objectZeroFrame.frame) : std::vector<std::uint8_t> {};
    const auto composedScene
        = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (composedSceneValue);
    const auto composedPreparation = backend.prepare (composedScene);
    auto composed = backend.render (
        composedScene, composedPreparation.resources, 64, 64, {});
    const auto composedPixels = composed.rendered
        ? readRgba8 (gl, composed.frame) : std::vector<std::uint8_t> {};
    const auto composedExport = backend.render (
        composedScene, composedPreparation.resources, 64, 64, {});
    const auto composedExportPixels = composedExport.rendered
        ? readRgba8 (gl, composedExport.frame) : std::vector<std::uint8_t> {};
    const std::array<std::uint8_t, 4> background { 7, 10, 18, 255 };
    std::size_t secondObjectPixels = 0;
    std::size_t changedObjectZeroPixels = 0;
    bool objectZeroUnchanged = composedPixels.size() == objectZeroPixels.size();
    for (std::uint32_t y = 0; composedPixels.size() == objectZeroPixels.size() && y < 64u; ++y)
        for (std::uint32_t x = 0; x < 64u; ++x)
        {
            const auto offset = (y * 64u + x) * 4u;
            if (! std::equal (composedPixels.begin() + offset,
                              composedPixels.begin() + offset + 4u,
                              objectZeroPixels.begin() + offset))
            {
                const auto wasBackground = std::equal (
                    objectZeroPixels.begin() + offset, objectZeroPixels.begin() + offset + 4u,
                    background.begin());
                objectZeroUnchanged &= wasBackground;
                changedObjectZeroPixels += wasBackground ? 0u : 1u;
                secondObjectPixels += composedPixels[offset + 1u]
                    > composedPixels[offset] * 4u ? 1u : 0u;
            }
        }
    const auto composedContractHolds = objectZeroPreparation.prepared && objectZeroFrame.rendered
        && objectZeroFrame.stats.drawCount == 1
        && composedPreparation.prepared && composed.rendered
        && composedExport.rendered
        && composedPreparation.stats.staticUploadCount == 1
        && composedPreparation.stats.vertexBytes == objectZeroPreparation.stats.vertexBytes
        && composedPreparation.stats.indexBytes == objectZeroPreparation.stats.indexBytes
        && composed.stats.drawCount == 2
        && composed.stats.reusedStaticResources
        && composedExport.stats.reusedStaticResources
        && composedPixels.size() == 64u * 64u * 4u
        && objectZeroUnchanged && secondObjectPixels > 0
        && composedPixels == composedExportPixels;
    if (! composedContractHolds)
        std::cerr << "OpenGL composed diagnostics: prepared=" << composedPreparation.prepared
                  << " rendered=" << composed.rendered
                  << " uploads=" << composedPreparation.stats.staticUploadCount
                  << " draws=" << composed.stats.drawCount
                  << " object0=" << objectZeroUnchanged
                  << " changedObject0=" << changedObjectZeroPixels
                  << " green=" << secondObjectPixels << '\n';
    ok &= expect (composedContractHolds,
                  "only the second OpenGL object must add distinct green pixels in its own region");

    auto invalidOpacitySceneValue = sceneValue;
    invalidOpacitySceneValue.materials[0].opacity = 1.5f;
    const auto invalidOpacityPreparation = backend.prepare (
        std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (invalidOpacitySceneValue),
        material);
    ok &= expect (! invalidOpacityPreparation.prepared
                      && invalidOpacityPreparation.resources == nullptr
                      && invalidOpacityPreparation.stats.staticUploadCount == 0,
                  "OpenGL rejects out-of-range opacity before GPU upload");

    auto wrongMaterial = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (*material);
    wrongMaterial->parameters.identifiers[0] += 1u;
    const auto wrongMaterialPreparation = backend.prepare (scene, wrongMaterial);
    ok &= expect (! wrongMaterialPreparation.prepared
                      && wrongMaterialPreparation.resources == nullptr
                      && wrongMaterialPreparation.stats.staticUploadCount == 0
                      && wrongMaterialPreparation.stats.materialProgramUploadCount == 0,
                  "a malformed Surface PBR identity must fail before OpenGL allocation");

    auto clippedSceneValue = sceneValue;
    clippedSceneValue.cameras[0].nearPlane = 7.0f;
    auto clippedScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
        clippedSceneValue);
    auto clippedPreparation = backend.prepare (clippedScene, material);
    auto clipped = backend.render (
        clippedScene, clippedPreparation.resources, 64, 64, {});
    const auto clippedPixels = clipped.rendered
        ? readRgba8 (gl, clipped.frame) : std::vector<std::uint8_t> {};
    ok &= expect (clippedPreparation.prepared && clipped.rendered
                      && clippedPixels.size() > center + 3u
                      && clippedPixels[center] == 7u
                      && clippedPixels[center + 1u] == 10u
                      && clippedPixels[center + 2u] == 18u
                      && clippedPixels[center + 3u] == 255u,
                  "the configured OpenGL near plane must clip the whole fixture cube");

    auto texturedSceneValue = videohelper::fixture3d::makeScene();
    texturedSceneValue.lightCount = 0;
    texturedSceneValue.ambientColor = { 1.0f, 1.0f, 1.0f };
    texturedSceneValue.vertexCount = 4;
    texturedSceneValue.indexCount = 6;
    texturedSceneValue.vertices[0] = { { -1.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, {}, { 0.0f, 0.0f } };
    texturedSceneValue.vertices[1] = { { 1.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, {}, { 1.0f, 0.0f } };
    texturedSceneValue.vertices[2] = { { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, {}, { 0.0f, 1.0f } };
    texturedSceneValue.vertices[3] = { { 1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, {}, { 1.0f, 1.0f } };
    texturedSceneValue.indices[0] = 0;
    texturedSceneValue.indices[1] = 2;
    texturedSceneValue.indices[2] = 1;
    texturedSceneValue.indices[3] = 1;
    texturedSceneValue.indices[4] = 2;
    texturedSceneValue.indices[5] = 3;
    texturedSceneValue.objects[0].transform = {};
    texturedSceneValue.objects[0].vertexCount = 4;
    texturedSceneValue.objects[0].indexCount = 6;
    texturedSceneValue.cameras[0].transform = {};
    texturedSceneValue.cameras[0].transform.translation.z = 2.0f;
    texturedSceneValue.textureTexels[0] = { 128, 0, 0, 255 };
    texturedSceneValue.textureTexels[1] = { 0, 128, 0, 255 };
    texturedSceneValue.textureTexels[2] = { 0, 0, 128, 255 };
    texturedSceneValue.textureTexels[3] = { 128, 128, 128, 255 };
    texturedSceneValue.textures[0].minFilter = GL_NEAREST;
    texturedSceneValue.textures[0].magFilter = GL_NEAREST;
    auto texturedScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
        texturedSceneValue);
    auto texturedMaterial = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
        *material);
    texturedMaterial->bindingDigest = "opengl-fixture-texture-binding";
    texturedMaterial->programIdentity = "opengl-fixture-texture-program";
    texturedMaterial->baseColorSource =
        arbitgpu::NativeFixtureSurfaceMaterialProgram::BaseColorSource::ImportedSrgbTexture;
    arbitgpu::NativeFixtureSurfaceMaterialProgram::ImportedSrgbTexture importedTexture;
    importedTexture.width = 2;
    importedTexture.height = 2;
    importedTexture.texelCount = 4;
    importedTexture.texels.resize (importedTexture.texelCount);
    std::copy_n (texturedSceneValue.textureTexels.begin(), 4,
                 importedTexture.texels.begin());
    texturedMaterial->importedBaseColorTexture = std::move (importedTexture);
    texturedMaterial->parameters.baseColorMetallic = { 1.0f, 1.0f, 1.0f, 0.0f };
    texturedMaterial->parameters.emissionRoughness = { 0.0f, 0.0f, 0.0f, 0.0f };
    auto texturedPreparation = backend.prepare (texturedScene, texturedMaterial);
    auto textured = backend.render (
        texturedScene, texturedPreparation.resources, 64, 64, {});
    if (! texturedPreparation.prepared)
        std::cerr << "OpenGL textured preparation error: "
                  << texturedPreparation.error << '\n';
    if (! textured.rendered)
        std::cerr << "OpenGL textured render error: " << textured.error << '\n';
    const auto texturedPixels = textured.rendered
        ? readRgba8 (gl, textured.frame) : std::vector<std::uint8_t> {};
    reportCoverage ("authored-quadrants", texturedPixels);
    const auto channel = [&] (std::uint32_t x, std::uint32_t y, std::uint32_t lane)
    {
        return texturedPixels[(y * 64u + x) * 4u + lane];
    };
    const auto decodedMidtone = [] (std::uint8_t value)
    {
        return value >= 40u && value <= 80u;
    };
    const auto texturedPixelsOk = texturedPreparation.prepared && textured.rendered
        && texturedPixels.size() == 64u * 64u * 4u
        && decodedMidtone (channel (16, 48, 0))
        && channel (16, 48, 1) < 5u && channel (16, 48, 2) < 5u
        && decodedMidtone (channel (48, 48, 1))
        && channel (48, 48, 0) < 5u && channel (48, 48, 2) < 5u
        && decodedMidtone (channel (16, 16, 2))
        && channel (16, 16, 0) < 5u && channel (16, 16, 1) < 5u
        && decodedMidtone (channel (48, 16, 0))
        && decodedMidtone (channel (48, 16, 1))
        && decodedMidtone (channel (48, 16, 2));
    if (! texturedPixelsOk && texturedPixels.size() == 64u * 64u * 4u)
    {
        for (const auto y : { 48u, 16u })
            for (const auto x : { 16u, 48u })
                std::cerr << "quadrant " << x << ',' << y << " = "
                          << static_cast<unsigned> (channel (x, y, 0)) << ','
                          << static_cast<unsigned> (channel (x, y, 1)) << ','
                          << static_cast<unsigned> (channel (x, y, 2)) << '\n';
    }
    ok &= expect (texturedPixelsOk,
                  "the top-left texture rows and sRGB samples must reach their authored quadrants");

    const auto renderStaticPixels = [&] (const HarmonicMIDI::grid::Visual3DScene& value)
    {
        auto snapshot = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (value);
        auto prepared = backend.prepare (snapshot, nullptr);
        if (! prepared.prepared)
        {
            std::cerr << "OpenGL static GLB semantic preparation error: "
                      << prepared.error << '\n';
            return std::vector<std::uint8_t> {};
        }
        auto rendered = backend.render (snapshot, prepared.resources, 64, 64, {});
        if (! rendered.rendered)
        {
            std::cerr << "OpenGL static GLB semantic render error: "
                      << rendered.error << '\n';
            return std::vector<std::uint8_t> {};
        }
        return readRgba8 (gl, rendered.frame);
    };
    const auto nonBackgroundCount = [] (const std::vector<std::uint8_t>& pixels)
    {
        std::size_t count = 0;
        for (std::size_t index = 0; index + 3 < pixels.size(); index += 4)
            if (pixels[index] != 7u || pixels[index + 1] != 10u
                || pixels[index + 2] != 18u || pixels[index + 3] != 255u)
                ++count;
        return count;
    };

    auto semanticScene = texturedSceneValue;
    semanticScene.textureCount = 0;
    semanticScene.textureTexelCount = 0;
    semanticScene.textureTexels.clear();
    semanticScene.materials[0].baseColorTexture = {};
    semanticScene.materials[0].baseColor = { 0.8f, 0.1f, 0.05f };
    semanticScene.materials[0].roughness = 0.2f;
    semanticScene.ambientColor = { 0.3f, 0.3f, 0.3f };
    semanticScene.objectCount = 2;
    semanticScene.vertexCount = 8;
    semanticScene.indexCount = 12;
    semanticScene.materialCount = 2;
    semanticScene.materials[1] = semanticScene.materials[0];
    semanticScene.materials[1].id = HarmonicMIDI::grid::SceneMaterialId { 2 };
    semanticScene.materials[1].baseColor = { 0.05f, 0.7f, 0.2f };
    semanticScene.objects[0].transform.translation.x = -0.55f;
    semanticScene.objects[0].transform.scale = { 0.45f, 0.8f, 1.0f };
    semanticScene.objects[1] = semanticScene.objects[0];
    semanticScene.objects[1].id = HarmonicMIDI::grid::SceneObjectId { 2 };
    semanticScene.objects[1].material = HarmonicMIDI::grid::SceneMaterialId { 2 };
    semanticScene.objects[1].firstVertex = 4;
    semanticScene.objects[1].firstIndex = 6;
    semanticScene.objects[1].transform.translation.x = 0.55f;
    std::copy_n (semanticScene.vertices.begin(), 4, semanticScene.vertices.begin() + 4);
    std::copy_n (semanticScene.indices.begin(), 6, semanticScene.indices.begin() + 6);
    const auto multiPrimitivePixels = renderStaticPixels (semanticScene);
    auto shiftedCameraScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (semanticScene);
    shiftedCameraScene->cameras[0].transform.translation.x += 0.5f;
    const auto shiftedCameraPixels = renderStaticPixels (*shiftedCameraScene);

    auto onePrimitiveScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (semanticScene);
    onePrimitiveScene->objectCount = 1;
    onePrimitiveScene->vertexCount = 4;
    onePrimitiveScene->indexCount = 6;
    const auto onePrimitivePixels = renderStaticPixels (*onePrimitiveScene);

    auto changedMaterialScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (semanticScene);
    changedMaterialScene->materials[1].baseColor = { 0.05f, 0.1f, 0.9f };
    changedMaterialScene->materials[1].metallic = 1.0f;
    changedMaterialScene->materials[1].roughness = 1.0f;
    const auto changedMaterialPixels = renderStaticPixels (*changedMaterialScene);

    auto hierarchyScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (semanticScene);
    hierarchyScene->objects[0].transform.translation.x = -0.8f;
    hierarchyScene->objects[1].parent = hierarchyScene->objects[0].id;
    hierarchyScene->objects[1].transform.translation.x = 1.1f;
    const auto hierarchyPixels = renderStaticPixels (*hierarchyScene);
    auto localOnlyScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*hierarchyScene);
    localOnlyScene->objects[1].parent = {};
    const auto localOnlyPixels = renderStaticPixels (*localOnlyScene);

    auto triangleTopologyScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (semanticScene);
    triangleTopologyScene->objects[1].indexCount = 3;
    triangleTopologyScene->indexCount = 9;
    const auto triangleTopologyPixels = renderStaticPixels (*triangleTopologyScene);

    auto directionalScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (semanticScene);
    directionalScene->ambientColor = {};
    directionalScene->lightCount = 1;
    directionalScene->lights[0].kind = HarmonicMIDI::grid::SceneLightKind::Directional;
    directionalScene->lights[0].transform = {};
    directionalScene->lights[0].intensity = 1.0f;
    const auto directionalPixels = renderStaticPixels (*directionalScene);
    auto pointScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*directionalScene);
    pointScene->lights[0].kind = HarmonicMIDI::grid::SceneLightKind::Point;
    pointScene->lights[0].transform.translation = { 0.0f, 0.0f, 1.0f };
    pointScene->lights[0].range = 5.0f;
    const auto pointPixels = renderStaticPixels (*pointScene);

    auto spotScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*pointScene);
    spotScene->lights[0].kind = HarmonicMIDI::grid::SceneLightKind::Spot;
    spotScene->lights[0].innerConeAngle = 0.15f;
    spotScene->lights[0].outerConeAngle = 0.35f;
    const auto spotInsidePixels = renderStaticPixels (*spotScene);
    auto spotOutsideScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*spotScene);
    // Rotate the cone away while holding position, range, and point-to-surface distance fixed.
    spotOutsideScene->lights[0].transform.rotation = { 0.0f, 1.0f, 0.0f, 0.0f };
    const auto spotOutsidePixels = renderStaticPixels (*spotOutsideScene);
    auto infiniteRangeScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*pointScene);
    infiniteRangeScene->lights[0].range = 0.0f;
    const auto infiniteRangePixels = renderStaticPixels (*infiniteRangeScene);
    auto finiteRangeScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*infiniteRangeScene);
    finiteRangeScene->lights[0].range = 0.25f; // Range is the pair's only changed field.
    const auto finiteRangePixels = renderStaticPixels (*finiteRangeScene);
    auto noLightScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*finiteRangeScene);
    noLightScene->lightCount = 0;
    const auto noLightPixels = renderStaticPixels (*noLightScene);

    auto backFaceScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*onePrimitiveScene);
    std::swap (backFaceScene->indices[0], backFaceScene->indices[1]);
    std::swap (backFaceScene->indices[3], backFaceScene->indices[4]);
    backFaceScene->materials[0].doubleSided = false;
    const auto culledPixels = renderStaticPixels (*backFaceScene);
    backFaceScene->materials[0].doubleSided = true;
    const auto doubleSidedPixels = renderStaticPixels (*backFaceScene);
    auto reflectedScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*onePrimitiveScene);
    reflectedScene->objects[0].transform.scale.x *= -1.0f;
    reflectedScene->materials[0].doubleSided = false;
    const auto reflectedPixels = renderStaticPixels (*reflectedScene);

    auto maskedScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*onePrimitiveScene);
    maskedScene->materials[0].alphaMode = HarmonicMIDI::grid::SceneAlphaMode::Mask;
    maskedScene->materials[0].opacity = 0.25f;
    maskedScene->materials[0].alphaCutoff = 0.5f;
    const auto maskedPixels = renderStaticPixels (*maskedScene);
    auto blendedScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*onePrimitiveScene);
    blendedScene->materials[0].alphaMode = HarmonicMIDI::grid::SceneAlphaMode::Blend;
    blendedScene->materials[0].opacity = 0.25f;
    const auto blendedPixels = renderStaticPixels (*blendedScene);
    auto overlappingBlendScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*onePrimitiveScene);
    overlappingBlendScene->objectCount = 3;
    overlappingBlendScene->materialCount = 3;
    overlappingBlendScene->vertexCount = 12;
    overlappingBlendScene->indexCount = 18;
    for (std::size_t index = 0; index < 3; ++index)
    {
        overlappingBlendScene->objects[index] = overlappingBlendScene->objects[0];
        overlappingBlendScene->objects[index].id
            = HarmonicMIDI::grid::SceneObjectId { static_cast<std::uint32_t>(index + 1) };
        overlappingBlendScene->objects[index].material
            = HarmonicMIDI::grid::SceneMaterialId { static_cast<std::uint32_t>(index + 1) };
        overlappingBlendScene->objects[index].firstVertex = static_cast<std::uint32_t>(index * 4);
        overlappingBlendScene->objects[index].firstIndex = static_cast<std::uint32_t>(index * 6);
        overlappingBlendScene->materials[index] = overlappingBlendScene->materials[0];
        overlappingBlendScene->materials[index].id = overlappingBlendScene->objects[index].material;
        overlappingBlendScene->materials[index].baseColor = {
            index == 0 ? 1.0f : 0.0f, index == 1 ? 1.0f : 0.0f,
            index == 2 ? 1.0f : 0.0f };
        overlappingBlendScene->materials[index].opacity = 0.5f;
        overlappingBlendScene->materials[index].alphaMode
            = HarmonicMIDI::grid::SceneAlphaMode::Blend;
        std::copy_n (overlappingBlendScene->vertices.begin(), 4,
                     overlappingBlendScene->vertices.begin() + index * 4);
        std::copy_n (overlappingBlendScene->indices.begin(), 6,
                     overlappingBlendScene->indices.begin() + index * 6);
    }
    const auto overlappingBlendPixels = renderStaticPixels (*overlappingBlendScene);
    const auto repeatedBlendPixels = renderStaticPixels (*overlappingBlendScene);
    auto reversedBlendScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*overlappingBlendScene);
    std::swap (reversedBlendScene->objects[0], reversedBlendScene->objects[2]);
    const auto reversedBlendPixels = renderStaticPixels (*reversedBlendScene);

    auto packedTextureScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*onePrimitiveScene);
    packedTextureScene->ambientColor = { 0.05f, 0.05f, 0.05f };
    packedTextureScene->lightCount = 1;
    packedTextureScene->lights[0].kind = HarmonicMIDI::grid::SceneLightKind::Directional;
    packedTextureScene->lights[0].transform = {};
    packedTextureScene->lights[0].intensity = 2.0f;
    packedTextureScene->materials[0].metallic = 0.8f;
    packedTextureScene->materials[0].roughness = 1.0f;
    packedTextureScene->textureCount = 5;
    packedTextureScene->textureTexelCount = 5;
    packedTextureScene->textureTexels.resize (packedTextureScene->textureTexelCount);
    for (std::size_t index = 0; index < packedTextureScene->textureCount; ++index)
    {
        packedTextureScene->textures[index].id
            = HarmonicMIDI::grid::SceneTextureId { static_cast<std::uint32_t> (index + 1) };
        packedTextureScene->textures[index].firstTexel = static_cast<std::uint32_t> (index);
        packedTextureScene->textures[index].width = 1;
        packedTextureScene->textures[index].height = 1;
        packedTextureScene->textures[index].minFilter = GL_NEAREST;
        packedTextureScene->textures[index].magFilter = GL_NEAREST;
    }
    packedTextureScene->textureTexels[0] = { 128, 96, 192, 255 };
    packedTextureScene->textureTexels[1] = { 255, 128, 128, 255 };
    packedTextureScene->textureTexels[2] = { 64, 255, 255, 255 };
    packedTextureScene->textureTexels[3] = { 32, 192, 255, 255 };
    packedTextureScene->textureTexels[4] = packedTextureScene->textureTexels[0];
    packedTextureScene->materials[0].metallicRoughnessTexture
        = packedTextureScene->textures[0].id;
    packedTextureScene->materials[0].normalTexture = packedTextureScene->textures[1].id;
    packedTextureScene->materials[0].occlusionTexture = packedTextureScene->textures[2].id;
    packedTextureScene->materials[0].emissiveTexture = packedTextureScene->textures[3].id;
    packedTextureScene->materials[0].emissive = { 0.2f, 0.2f, 0.2f };
    const auto packedTexturePixels = renderStaticPixels (*packedTextureScene);
    auto reflectedNormalScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*packedTextureScene);
    reflectedNormalScene->objects[0].transform.scale.x *= -1.0f;
    const auto reflectedNormalPixels = renderStaticPixels (*reflectedNormalScene);
    auto reflectedWithoutNormal = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*reflectedNormalScene);
    reflectedWithoutNormal->materials[0].normalTexture = {};
    const auto reflectedWithoutNormalPixels = renderStaticPixels (*reflectedWithoutNormal);
    // Independent analytic reflection. It bakes the reflection into geometry and tangent
    // handedness, so it has a positive object determinant but the same mapped surface.
    auto analyticReflectedNormal = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*packedTextureScene);
    for (std::size_t index = 0; index < analyticReflectedNormal->vertexCount; ++index)
    {
        analyticReflectedNormal->vertices[index].position.x *= -1.0f;
        analyticReflectedNormal->vertices[index].normal.x *= -1.0f;
        analyticReflectedNormal->vertices[index].tangent[0] *= -1.0f;
        analyticReflectedNormal->vertices[index].tangent[3] *= -1.0f;
    }
    for (std::size_t index = 0; index + 2 < analyticReflectedNormal->indexCount; index += 3)
        std::swap (analyticReflectedNormal->indices[index], analyticReflectedNormal->indices[index + 1]);
    const auto analyticReflectedNormalPixels = renderStaticPixels (*analyticReflectedNormal);
    auto sharedRoleScene = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*packedTextureScene);
    // The front-facing geometry normal keeps both transfer-role controls
    // distinguishable after RGBA8 quantization. Normal mapping is tested above.
    sharedRoleScene->materials[0].normalTexture = {};
    sharedRoleScene->materials[0].baseColorTexture = sharedRoleScene->textures[0].id;
    sharedRoleScene->materials[0].metallicRoughnessTexture = sharedRoleScene->textures[0].id;
    const auto sharedRolePixels = renderStaticPixels (*sharedRoleScene);
    // The duplicate binding is an independent reference for the required role split.
    auto splitRoleReference = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*sharedRoleScene);
    splitRoleReference->materials[0].baseColorTexture = splitRoleReference->textures[4].id;
    const auto splitRoleReferencePixels = renderStaticPixels (*splitRoleReference);
    auto wronglyLinearBase = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*splitRoleReference);
    // These are sRGB encodings of the original numeric channel values. A correct
    // sRGB base-color view decodes them to the forbidden linear-sampling result.
    wronglyLinearBase->textureTexels[4] = { 188, 165, 225, 255 };
    const auto wronglyLinearBasePixels = renderStaticPixels (*wronglyLinearBase);
    auto wronglySrgbMetallicRoughness = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*splitRoleReference);
    // sRGB decoding maps 128 to about 55. Using those bytes in the linear role models
    // the forbidden reuse of an sRGB image view without depending on backend code.
    wronglySrgbMetallicRoughness->textureTexels[0] = { 55, 30, 134, 255 };
    const auto wronglySrgbMetallicRoughnessPixels = renderStaticPixels (*wronglySrgbMetallicRoughness);
    std::cerr << "semantic reflected=" << std::hex << fnv1a64 (reflectedNormalPixels)
              << " analytic-reflected=" << fnv1a64 (analyticReflectedNormalPixels)
              << " shared-role=" << fnv1a64 (sharedRolePixels)
              << " split-role=" << fnv1a64 (splitRoleReferencePixels)
              << " wrong-linear-base=" << fnv1a64 (wronglyLinearBasePixels)
              << " wrong-srgb-data=" << fnv1a64 (wronglySrgbMetallicRoughnessPixels)
              << std::dec << '\n';
    auto noMetallicRoughnessTexture = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*packedTextureScene);
    noMetallicRoughnessTexture->materials[0].metallicRoughnessTexture = {};
    auto noNormalTexture = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*packedTextureScene);
    noNormalTexture->materials[0].normalTexture = {};
    auto noOcclusionTexture = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*packedTextureScene);
    noOcclusionTexture->materials[0].occlusionTexture = {};
    auto noEmissiveTexture = std::make_unique<HarmonicMIDI::grid::Visual3DScene> (*packedTextureScene);
    noEmissiveTexture->materials[0].emissiveTexture = {};
    auto semanticSnapshot =
        std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (semanticScene);
    auto semanticPreparation = backend.prepare (semanticSnapshot, nullptr);
    const auto generationFrameA = backend.render (
        semanticSnapshot, semanticPreparation.resources, 64, 64, {});
    const auto generationFrameB = backend.render (
        semanticSnapshot, semanticPreparation.resources, 64, 64, {});
    const auto reprepared = backend.prepare (semanticSnapshot, nullptr);
    const auto generationFrameC = backend.render (
        semanticSnapshot, reprepared.resources, 64, 64, {});
    const auto invalidExtent = backend.render (
        semanticSnapshot, semanticPreparation.resources, 0, 64, {});

    const std::array<std::pair<const char*, bool>, 22> semanticChecks {{
        { "primitive count", multiPrimitivePixels != onePrimitivePixels },
        { "camera", shiftedCameraPixels != multiPrimitivePixels },
        { "material", changedMaterialPixels != multiPrimitivePixels },
        { "hierarchy", hierarchyPixels != localOnlyPixels },
        { "topology", triangleTopologyPixels != multiPrimitivePixels },
        { "light kind", directionalPixels != pointPixels },
        { "spot cone", spotInsidePixels != spotOutsidePixels },
        { "finite range", infiniteRangePixels != finiteRangePixels
            && finiteRangePixels == noLightPixels },
        { "single sided", nonBackgroundCount (culledPixels) == 0 },
        { "double sided", nonBackgroundCount (doubleSidedPixels) > 0 },
        { "reflected culling", nonBackgroundCount (reflectedPixels) > 0 },
        { "mask", nonBackgroundCount (maskedPixels) == 0 },
        { "blend", blendedPixels != onePrimitivePixels && nonBackgroundCount (blendedPixels) > 0 },
        { "stable overlapping blend order", overlappingBlendPixels == repeatedBlendPixels
            && overlappingBlendPixels != reversedBlendPixels },
        { "metallic roughness", packedTexturePixels != renderStaticPixels (*noMetallicRoughnessTexture) },
        { "normal", packedTexturePixels != renderStaticPixels (*noNormalTexture) },
        { "reflected tangent", reflectedNormalPixels == analyticReflectedNormalPixels
            && reflectedNormalPixels != reflectedWithoutNormalPixels },
        { "occlusion", packedTexturePixels != renderStaticPixels (*noOcclusionTexture) },
        { "emissive", packedTexturePixels != renderStaticPixels (*noEmissiveTexture) },
        { "shared sRGB and linear image", sharedRolePixels == splitRoleReferencePixels
            && sharedRolePixels != wronglyLinearBasePixels
            && sharedRolePixels != wronglySrgbMetallicRoughnessPixels },
        { "stable generation", generationFrameA.rendered && generationFrameB.rendered
            && generationFrameC.rendered
            && generationFrameA.frame->colorTextureDescriptor().rendererGeneration
                == generationFrameB.frame->colorTextureDescriptor().rendererGeneration
            && generationFrameA.frame->colorTextureDescriptor().rendererGeneration
                != generationFrameC.frame->colorTextureDescriptor().rendererGeneration },
        { "bounded prepare", semanticPreparation.prepared && ! invalidExtent.rendered }
    }};
    for (const auto& semanticCheck : semanticChecks)
        if (! semanticCheck.second)
            std::cerr << "static GLB semantic check failed: " << semanticCheck.first << '\n';

    ok &= expect (! multiPrimitivePixels.empty()
                      && multiPrimitivePixels != onePrimitivePixels
                      && shiftedCameraPixels != multiPrimitivePixels
                      && changedMaterialPixels != multiPrimitivePixels
                      && hierarchyPixels != localOnlyPixels
                      && triangleTopologyPixels != multiPrimitivePixels
                      && directionalPixels != pointPixels
                      && spotInsidePixels != spotOutsidePixels
                      && infiniteRangePixels != finiteRangePixels
                      && finiteRangePixels == noLightPixels
                      && nonBackgroundCount (culledPixels) == 0
                      && nonBackgroundCount (doubleSidedPixels) > 0
                      && nonBackgroundCount (reflectedPixels) > 0
                      && nonBackgroundCount (maskedPixels) == 0
                      && blendedPixels != onePrimitivePixels
                      && nonBackgroundCount (blendedPixels) > 0
                      && overlappingBlendPixels == repeatedBlendPixels
                      && overlappingBlendPixels != reversedBlendPixels
                      && packedTexturePixels != renderStaticPixels (*noMetallicRoughnessTexture)
                      && packedTexturePixels != renderStaticPixels (*noNormalTexture)
                      && reflectedNormalPixels == analyticReflectedNormalPixels
                      && reflectedNormalPixels != reflectedWithoutNormalPixels
                      && packedTexturePixels != renderStaticPixels (*noOcclusionTexture)
                      && packedTexturePixels != renderStaticPixels (*noEmissiveTexture)
                      && sharedRolePixels == splitRoleReferencePixels
                      && sharedRolePixels != wronglyLinearBasePixels
                      && sharedRolePixels != wronglySrgbMetallicRoughnessPixels
                      && generationFrameA.rendered && generationFrameB.rendered
                      && generationFrameC.rendered
                      && generationFrameA.frame->colorTextureDescriptor().rendererGeneration
                          == generationFrameB.frame->colorTextureDescriptor().rendererGeneration
                      && generationFrameA.frame->colorTextureDescriptor().rendererGeneration
                          != generationFrameC.frame->colorTextureDescriptor().rendererGeneration
                      && semanticPreparation.prepared
                      && ! invalidExtent.rendered && invalidExtent.frame == nullptr
                      && invalidExtent.error
                           == "OpenGL fixture render dimensions exceed backend limits",
                  "production OpenGL pixels must independently consume primitive topology, materials, hierarchy, alpha mode, sidedness, and directional or point lights");

    std::string cardLoadError;
    const auto cardAsset = videohelper::fixture3d::loadHolographicTradingCardScene(
        cardLoadError);
    ok &= expect(cardAsset.has_value(),
                 "the exact checked-in card GLB must pass production decode and scene adaptation");
    if (!cardAsset)
    {
        std::cerr << "Card GLB load error: " << cardLoadError << '\n';
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }
    const auto& diffractionScene = cardAsset->scene;
    ok &= expect(cardAsset->assetId == "builtin.visual-model.holographic-trading-card"
                     && cardAsset->contentSha256
                         == "5e3433aa19cc4636c33a1f6dd53a4a05fdffa39d788ad565b530d8dfdcce0287"
                     && cardAsset->sceneName == "Holographic Trading Card Scene"
                     && cardAsset->objectName == "Trading Card"
                     && cardAsset->materialName == "Physical Diffraction Target"
                     && cardAsset->cameraName == "Card Camera"
                     && cardAsset->lightName == "Card Key"
                     && diffractionScene->vertexCount == 24u
                     && diffractionScene->indexCount == 36u,
                 "card asset, content, names, and decoded geometry must survive adaptation");
    auto diffractionDescription = diffractionmaterial::makeAluminiumBinaryGratingPreset();
    std::string diffractionError;
    auto admittedDiffraction = diffractionmaterial::admit (
        diffractionDescription, diffractionError);
    diffractionmaterial::SpectralIncidentLight incident;
    incident.radiance.fill (0.01f);
    diffractionmaterial::SpectralLightingPath lightingPath;
    lightingPath.kind = diffractionmaterial::LightingPathKind::Direct;
    lightingPath.incident = incident;
    auto diffractionProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>();
    diffractionProgram->backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    diffractionProgram->kind
        = arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;
    diffractionProgram->object = diffractionScene->objects[0].id;
    diffractionProgram->bindingDigest = admittedDiffraction
        ? admittedDiffraction->structuralDigest() : std::string {};

    diffractionProgram->diffractionPathCount = 1;
    diffractionmaterial::LightingDescription lightingDescription;
    lightingDescription.pathCount = 1;
    lightingDescription.paths[0] = lightingPath;
    const auto admittedLighting = diffractionmaterial::AdmittedLightingPlan::admit(
        lightingDescription, diffractionError);
    if (admittedLighting)
    {
        diffractionProgram->diffractionLightingAdmission
            = std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(
                *admittedLighting);
        diffractionProgram->programIdentity
            = arbitgpu::nativeFixtureDiffractionProgramIdentity(
                diffractionProgram->bindingDigest, *admittedLighting);
    }
    if (admittedDiffraction)
        diffractionProgram->diffractionPaths[0] = diffractionmaterial::makeGpuLightingPath(
            diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                *admittedDiffraction, incident), lightingPath);


    auto diffractionPreparation = backend.prepare (
        diffractionScene, diffractionProgram);
    auto replayedLightingProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(
            *diffractionProgram);
    auto replayedLightingDescription = lightingDescription;
    replayedLightingDescription.paths[0].incident.radiance[1]
        = std::nextafter(
            replayedLightingDescription.paths[0].incident.radiance[1], 0.0f);
    const auto replayedLighting = diffractionmaterial::AdmittedLightingPlan::admit(
        replayedLightingDescription, diffractionError);
    if (replayedLighting && admittedDiffraction)
    {
        replayedLightingProgram->diffractionLightingAdmission
            = std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(
                *replayedLighting);
        replayedLightingProgram->diffractionPaths[0]
            = diffractionmaterial::makeGpuLightingPath(
                diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                    *admittedDiffraction,
                    replayedLightingDescription.paths[0].incident),
                replayedLightingDescription.paths[0]);
    }
    const auto replayedLightingPreparation = backend.prepare(
        diffractionScene, replayedLightingProgram);
    auto malformedDiffractionProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
            *diffractionProgram);
    malformedDiffractionProgram->programIdentity
        = "opengl-fixture-malformed-diffraction-program";
    malformedDiffractionProgram->diffractionPaths[0].material.spectralZ[7].w
        = std::numeric_limits<float>::quiet_NaN();
    const auto malformedDiffractionPreparation = backend.prepare (
        diffractionScene, malformedDiffractionProgram);
    auto layoutSentinelProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
            *diffractionProgram);
    layoutSentinelProgram->programIdentity
        = "opengl-fixture-layout-sentinel-diffraction-program";
    layoutSentinelProgram->parameters.identifiers[3] = 1u;
    const auto layoutSentinelPreparation = backend.prepare (
        diffractionScene, layoutSentinelProgram);
    auto divergentBlazeProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
            *diffractionProgram);
    divergentBlazeProgram->programIdentity
        = "opengl-fixture-divergent-local-blaze-program";
    divergentBlazeProgram->diffractionPaths[0].material.control.y
        = static_cast<float> (diffractionmaterial::GrooveProfile::BlazedSawtooth);
    divergentBlazeProgram->diffractionPaths[0].material.grooveField = {
        static_cast<float> (diffractionmaterial::GrooveFieldMode::Linear),
        0.5f, 0.5f, 0.0f
    };
    divergentBlazeProgram->diffractionPaths[0].material.grooveVariation
        = { 1.0f, 0.0f, 100.0f, 0.0f };
    const auto divergentBlazePreparation = backend.prepare (
        diffractionScene, divergentBlazeProgram);
    diffractionProgram->diffractionPaths[0].material.roughness.y = 0.0f;
    auto diffractionFrame = backend.render (
        diffractionScene, diffractionPreparation.resources, 96, 96, {});
    const bool linearDiffraction = linearscenechecks::diffraction(backend, diffractionScene,
        diffractionPreparation.resources, linearError);
    ok &= expect(linearDiffraction, linearError.c_str());
    if (! diffractionPreparation.prepared)
        std::cerr << "OpenGL diffraction preparation error: "
                  << diffractionPreparation.error << '\n';
    if (! diffractionFrame.rendered)
        std::cerr << "OpenGL diffraction render error: "
                  << diffractionFrame.error << '\n';
    const auto diffractionPixels = diffractionFrame.rendered
        ? readRgba8 (gl, diffractionFrame.frame) : std::vector<std::uint8_t> {};
    reportCoverage ("diffraction", diffractionPixels);
    if (diffractionFrame.frame)
    {
        for (const auto output : { renderpassoutput::Output::Depth,
                                   renderpassoutput::Output::Normal,
                                   renderpassoutput::Output::Mask,
                                   renderpassoutput::Output::MaterialId,
                                   renderpassoutput::Output::ObjectId })
            ok &= expect(diffractionFrame.frame->passTextureDescriptor(output).complete(),
                "diffraction retains live typed pass descriptors");
        arbitgpu::NativeFixtureSceneRuntimeInputs normalInput;
        normalInput.imageOutput = renderpassoutput::Output::Normal;
        const auto normalFrame = backend.render(diffractionScene,
            diffractionPreparation.resources, 96, 96, normalInput);
        const auto normalPixels = normalFrame.frame
            ? readRgba8(gl, normalFrame.frame) : std::vector<std::uint8_t> {};
        ok &= expect(normalFrame.rendered && normalPixels.size() == 96u * 96u * 4u
            && normalPixels != diffractionPixels,
            "diffraction normal inspection resolves from its retained raw pass");
        for (const auto output : { renderpassoutput::Output::Depth,
                                   renderpassoutput::Output::Motion,
                                   renderpassoutput::Output::Emission,
                                   renderpassoutput::Output::Mask,
                                   renderpassoutput::Output::MaterialId,
                                   renderpassoutput::Output::ObjectId })
        {
            normalInput.imageOutput = output;
            const auto selected = backend.render(diffractionScene,
                diffractionPreparation.resources, 96, 96, normalInput);
            ok &= expect(selected.rendered && selected.frame != nullptr,
                "diffraction typed image outputs resolve from one retained scene draw");
        }

        arbitgpu::NativeFixtureSceneRuntimeInputs matteInput;
        matteInput.passComposite = renderpasscomposite::Parameters {};
        matteInput.passComposite->mode = renderpasscomposite::Mode::ObjectMatte;
        matteInput.passComposite->identity = diffractionScene->objects[0].id.value;
        const auto matteFrame = backend.render(diffractionScene,
            diffractionPreparation.resources, 96, 96, matteInput);
        const auto mattePixels = matteFrame.frame
            ? readRgba8(gl, matteFrame.frame) : std::vector<std::uint8_t> {};
        matteInput.passComposite->identity = diffractionScene->objects[0].id.value + 1u;
        const auto wrongMatteFrame = backend.render(diffractionScene,
            diffractionPreparation.resources, 96, 96, matteInput);
        const auto wrongMattePixels = wrongMatteFrame.frame
            ? readRgba8(gl, wrongMatteFrame.frame) : std::vector<std::uint8_t> {};
        std::size_t coveredPixel = mattePixels.size();
        for (std::size_t pixel = 0; pixel + 3u < mattePixels.size(); pixel += 4u)
            if (mattePixels[pixel] == 255u) { coveredPixel = pixel; break; }
        ok &= expect(matteFrame.rendered && wrongMatteFrame.rendered
            && mattePixels.size() == 96u * 96u * 4u
            && wrongMattePixels.size() == mattePixels.size()
            && coveredPixel < mattePixels.size() && wrongMattePixels[coveredPixel] == 0,
            "diffraction Object Matte compares the retained integer ID after lighting resolves");
    }

    // The authored card reserves three lighting paths at a 1080p ceiling;
    // this smaller draw still reserves one pixel slot per path.
    constexpr auto foilLightingPathCount = 3u;
    constexpr auto foilEvaluationCount = 320u * 180u * foilLightingPathCount;
    constexpr auto foilEvaluationBudget = 1920u * 1080u * foilLightingPathCount;
    const auto& foilRequest = *cardAsset->operation.diffractionMaterial;
    ok &= expect(foilRequest.lighting.pathCount == foilLightingPathCount
                     && foilRequest.spatialFoil
                     && foilRequest.spatialFoil->workBudget.maximumEvaluations == foilEvaluationBudget,
                 "the compiled card retains its three-path 1080p work ceiling");
    ok &= diffractionenvironmenttests::run(backend, diffractionScene, cardAsset->operation,
        videohelper::materialprogram::BackendTarget::OpenGl,
        [&](const auto& frame) { return readRgba8(gl, frame); });

    const auto foilPreviewBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
        diffractionScene, foilRequest,
        videohelper::materialprogram::BackendTarget::OpenGl, diffractionError);
    const auto foilExportBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
        diffractionScene, foilRequest,
        videohelper::materialprogram::BackendTarget::OpenGl, diffractionError);
    auto wrongCardObjectRequest = foilRequest;
    wrongCardObjectRequest.object = HarmonicMIDI::grid::SceneObjectId { 2 };
    const auto wrongCardObjectBinding
        = videorender::fixture3d::admitDiffractionMaterialBinding(
            diffractionScene, wrongCardObjectRequest,
            videohelper::materialprogram::BackendTarget::OpenGl, diffractionError);
    if (foilPreviewBinding == nullptr || foilExportBinding == nullptr)
        std::cerr << "Diffractive foil binding error: " << diffractionError << '\n';
    videorender::fixture3d::FixtureSceneRenderer foilPreviewRenderer(backend);
    videorender::fixture3d::FixtureSceneRenderer foilExportRenderer(backend);
    videorender::fixture3d::RenderedFrame foilPreviewFrame;
    videorender::fixture3d::RenderedFrame foilExportFrame;
    const bool foilRendered = foilPreviewBinding != nullptr && foilExportBinding != nullptr
        && foilPreviewBinding != foilExportBinding
        && foilPreviewBinding->nativeProgram() != foilExportBinding->nativeProgram()
        && foilPreviewBinding->bindingDigest() == foilExportBinding->bindingDigest()
        && wrongCardObjectBinding == nullptr
        && foilPreviewRenderer.renderPreview(diffractionScene, foilPreviewBinding, {},
            { 320, 180 }, videorender::fixture3d::kNativeGpuCapability,
            foilPreviewFrame, diffractionError)
        && foilExportRenderer.renderExport(diffractionScene, foilExportBinding, {},
            { 320, 180 }, videorender::fixture3d::kNativeGpuCapability,
            foilExportFrame, diffractionError);
    if (!foilRendered)
        std::cerr << "Diffractive foil render error: " << diffractionError << '\n';
    ok &= expect(foilRendered
                     && foilPreviewFrame.stats.diffractionEvaluationBudget
                         == foilEvaluationBudget
                     && foilPreviewFrame.stats.diffractionEvaluationCount
                         == foilEvaluationCount
                     && foilExportFrame.stats.diffractionEvaluationBudget
                         == foilEvaluationBudget
                     && foilExportFrame.stats.diffractionEvaluationCount
                         == foilEvaluationCount
                     && foilPreviewFrame.stats.drawCount == foilLightingPathCount
                     && foilExportFrame.stats.drawCount == foilLightingPathCount,
                 "OpenGL reports the exact deterministic pixel-slot and lighting-path evaluation count");
    if (foilPreviewBinding)
    {
        const auto multiPath = backend.prepare(diffractionScene,
            foilPreviewBinding->nativeProgram());
        arbitgpu::NativeFixtureSceneRuntimeInputs fogInput;
        fogInput.passComposite = renderpasscomposite::Parameters {};
        fogInput.passComposite->mode = renderpasscomposite::Mode::DepthFog;
        fogInput.passComposite->farDepth = 0.001f;
        fogInput.passComposite->fogColor = { 0.2f, 0.4f, 0.6f };
        const auto fogFrame = multiPath.prepared
            ? backend.render(diffractionScene, multiPath.resources, 320, 180, fogInput)
            : arbitgpu::NativeFixtureSceneSubmission {};
        const auto fogPixels = fogFrame.frame
            ? readRgba8(gl, fogFrame.frame) : std::vector<std::uint8_t> {};
        std::vector<std::uint8_t> mask(320u * 180u);
        if (fogFrame.frame)
        {
            const auto descriptor = fogFrame.frame->passTextureDescriptor(
                renderpassoutput::Output::Mask);
            if (descriptor.complete())
            {
                glBindTexture(GL_TEXTURE_2D, static_cast<unsigned>(descriptor.textureViewHandle));
                glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_BYTE, mask.data());
            }
        }
        const auto covered = std::find(mask.begin(), mask.end(), std::uint8_t { 255 });
        const auto coveredPixel = static_cast<std::size_t>(covered - mask.begin()) * 4u;
        ok &= expect(fogFrame.rendered && fogFrame.stats.drawCount == foilLightingPathCount
            && fogPixels.size() == 320u * 180u * 4u
            && covered != mask.end()
            && std::abs(static_cast<int>(fogPixels[coveredPixel]) - 51) <= 1
            && std::abs(static_cast<int>(fogPixels[coveredPixel + 1u]) - 102) <= 1
            && std::abs(static_cast<int>(fogPixels[coveredPixel + 2u]) - 153) <= 1,
            "three diffraction lighting paths receive one post-accumulation fog composite");
        fogInput.passProgram = renderpassfixture::hdrBranches();
        fogInput.passComposite = fogInput.passProgram->steps[fogInput.passProgram->output].parameters;
        fogInput.passProgram->steps[2].inputA = 3;
        ok &= expect(multiPath.prepared
            && !backend.render(diffractionScene, multiPath.resources, 320, 180,
                fogInput).rendered,
            "malformed diffraction pass graphs fail before native allocation");
    }

    auto belowCeilingFoilRequest = foilRequest;
    belowCeilingFoilRequest.spatialFoil->workBudget.maximumEvaluations
        = foilEvaluationCount + 1u;
    const auto belowCeilingAdmission = diffractivefoil::admit(
        *belowCeilingFoilRequest.spatialFoil, diffractionError);
    belowCeilingFoilRequest.structuralDigest = belowCeilingAdmission
        ? belowCeilingAdmission->structuralDigest() : std::string {};
    const auto belowCeilingBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
        diffractionScene, belowCeilingFoilRequest,
        videohelper::materialprogram::BackendTarget::OpenGl, diffractionError);
    videorender::fixture3d::FixtureSceneRenderer belowCeilingRenderer(backend);
    videorender::fixture3d::RenderedFrame belowCeilingFrame;
    const bool belowCeilingRendered = belowCeilingBinding != nullptr
        && belowCeilingRenderer.renderPreview(diffractionScene, belowCeilingBinding, {},
            { 320, 180 }, videorender::fixture3d::kNativeGpuCapability,
            belowCeilingFrame, diffractionError);
    ok &= expect(belowCeilingRendered
                     && belowCeilingFrame.stats.diffractionEvaluationBudget
                         == foilEvaluationCount + 1u
                     && belowCeilingFrame.stats.diffractionEvaluationCount
                         == foilEvaluationCount,
                 "OpenGL admits exact work below the immutable evaluation ceiling");

    auto exhaustedFoilRequest = foilRequest;
    exhaustedFoilRequest.spatialFoil->workBudget.maximumEvaluations
        = foilEvaluationCount - 1u;
    const auto exhaustedAdmission = diffractivefoil::admit(
        *exhaustedFoilRequest.spatialFoil, diffractionError);
    exhaustedFoilRequest.structuralDigest = exhaustedAdmission
        ? exhaustedAdmission->structuralDigest() : std::string {};
    const auto exhaustedBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
        diffractionScene, exhaustedFoilRequest,
        videohelper::materialprogram::BackendTarget::OpenGl, diffractionError);
    videorender::fixture3d::FixtureSceneRenderer exhaustedRenderer(backend);
    videorender::fixture3d::RenderedFrame exhaustedFrame;
    const bool exhaustedRendered = exhaustedBinding != nullptr
        && exhaustedRenderer.renderPreview(diffractionScene, exhaustedBinding, {},
            { 320, 180 }, videorender::fixture3d::kNativeGpuCapability,
            exhaustedFrame, diffractionError);
    ok &= expect(!exhaustedRendered && exhaustedFrame.nativeFrame == nullptr
                     && diffractionError
                         == "OpenGL fixture diffraction workload exceeds backend limits",
                 "OpenGL rejects a spatial foil one evaluation over budget before frame allocation");
    const auto foilPreviewPixels = foilRendered
        ? readRgba8(gl, foilPreviewFrame.nativeFrame) : std::vector<std::uint8_t> {};
    const auto foilExportPixels = foilRendered
        ? readRgba8(gl, foilExportFrame.nativeFrame) : std::vector<std::uint8_t> {};
    const auto belowCeilingPixels = belowCeilingRendered
        ? readRgba8(gl, belowCeilingFrame.nativeFrame) : std::vector<std::uint8_t> {};
    const auto disabledFoilPreparation = backend.prepare(diffractionScene, nullptr);
    const auto disabledFoilFrame = backend.render(
        diffractionScene, disabledFoilPreparation.resources, 320, 180, {});
    const auto disabledFoilPixels = disabledFoilFrame.rendered
        ? readRgba8(gl, disabledFoilFrame.frame) : std::vector<std::uint8_t> {};
    std::size_t disabledFoilDifferingPixels = 0;
    if (foilPreviewPixels.size() == disabledFoilPixels.size())
    {
        for (std::size_t offset = 0; offset < foilPreviewPixels.size(); offset += 4)
        {
            if (!std::equal(foilPreviewPixels.begin() + static_cast<std::ptrdiff_t>(offset),
                            foilPreviewPixels.begin() + static_cast<std::ptrdiff_t>(offset + 3),
                            disabledFoilPixels.begin() + static_cast<std::ptrdiff_t>(offset)))
                ++disabledFoilDifferingPixels;
        }
    }
    ok &= expect(disabledFoilPreparation.prepared && disabledFoilFrame.rendered
                     && disabledFoilPixels.size() == 320u * 180u * 4u
                     && fnv1a64(foilPreviewPixels) != fnv1a64(disabledFoilPixels)
                     && disabledFoilDifferingPixels > 100u,
                 "OpenGL card foil must differ from a same-size disabled-diffraction render of the decoded card scene");

    bool everyFoilCornerPerturbsPixels = foilRendered && !foilPreviewPixels.empty()
        && fnv1a64(foilPreviewPixels) == fnv1a64(foilExportPixels)
        && fnv1a64(foilPreviewPixels) == fnv1a64(belowCeilingPixels);
    for (std::size_t corner = 0; corner < diffractivefoil::kFieldSampleCount; ++corner)
    {
        auto changed = foilRequest;
        changed.spatialFoil->grooveField[corner].grooveSpacingNanometres += 113.0f;
        if (corner == 0)
        {
            changed.spatialFoil->physicalBsdf.geometry.grooveSpacingNanometres
                = changed.spatialFoil->grooveField[0].grooveSpacingNanometres;
            changed.material = changed.spatialFoil->physicalBsdf;
        }
        const auto changedAdmission = diffractivefoil::admit(*changed.spatialFoil, diffractionError);
        changed.structuralDigest = changedAdmission
            ? changedAdmission->structuralDigest() : std::string {};
        const auto changedBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
            diffractionScene, changed,
            videohelper::materialprogram::BackendTarget::OpenGl, diffractionError);
        videorender::fixture3d::FixtureSceneRenderer changedRenderer(backend);
        videorender::fixture3d::RenderedFrame changedFrame;
        const bool changedRendered = changedBinding != nullptr
            && changedRenderer.renderPreview(diffractionScene, changedBinding, {}, { 320, 180 },
                videorender::fixture3d::kNativeGpuCapability, changedFrame, diffractionError);
        const auto changedPixels = changedRendered
            ? readRgba8(gl, changedFrame.nativeFrame) : std::vector<std::uint8_t> {};
        const auto changedHash = fnv1a64(changedPixels);
        const auto baselineHash = fnv1a64(foilPreviewPixels);
        const bool grooveChanged = changedRendered && changedHash != baselineHash;
        if (!grooveChanged)
            std::cerr << "Diffractive foil groove corner " << corner
                      << " did not change pixels: " << std::hex << baselineHash
                      << " vs " << changedHash << std::dec << '\n';
        everyFoilCornerPerturbsPixels = everyFoilCornerPerturbsPixels && grooveChanged;

        changed = foilRequest;
        changed.spatialFoil->grooveField[corner].diffractionCoverage = 0.0f;
        const auto occupancyAdmission = diffractivefoil::admit(
            *changed.spatialFoil, diffractionError);
        changed.structuralDigest = occupancyAdmission
            ? occupancyAdmission->structuralDigest() : std::string {};
        const auto occupancyBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
            diffractionScene, changed,
            videohelper::materialprogram::BackendTarget::OpenGl, diffractionError);
        videorender::fixture3d::FixtureSceneRenderer occupancyRenderer(backend);
        videorender::fixture3d::RenderedFrame occupancyFrame;
        const bool occupancyRendered = occupancyBinding != nullptr
            && occupancyRenderer.renderPreview(diffractionScene, occupancyBinding, {}, { 320, 180 },
                videorender::fixture3d::kNativeGpuCapability, occupancyFrame, diffractionError);
        const auto occupancyPixels = occupancyRendered
            ? readRgba8(gl, occupancyFrame.nativeFrame) : std::vector<std::uint8_t> {};
        const auto occupancyHash = fnv1a64(occupancyPixels);
        const bool occupancyChanged = occupancyRendered && occupancyHash != baselineHash;
        if (!occupancyChanged)
            std::cerr << "Diffractive foil occupancy corner " << corner
                      << " did not change pixels: " << std::hex << baselineHash
                      << " vs " << occupancyHash << std::dec << '\n';
        everyFoilCornerPerturbsPixels = everyFoilCornerPerturbsPixels
            && occupancyChanged;
    }
    ok &= expect(everyFoilCornerPerturbsPixels,
                 "graph-authored Diffractive Foil preview and export must share exact immutable semantics and pixels must depend on all four groove and occupancy corners");

    auto malformedFoil = foilRequest;
    malformedFoil.spatialFoil->grooveField[2].reciprocalDirectionUv[0]
        = std::numeric_limits<float>::quiet_NaN();
    const auto malformedFoilBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
        diffractionScene, malformedFoil,
        videohelper::materialprogram::BackendTarget::OpenGl, diffractionError);
    auto staleFoil = foilRequest;
    staleFoil.structuralDigest.push_back('0');
    const auto staleFoilBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
        diffractionScene, staleFoil,
        videohelper::materialprogram::BackendTarget::OpenGl, diffractionError);
    const auto unsupportedFoilBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
        diffractionScene, foilRequest,
        videohelper::materialprogram::BackendTarget::Invalid, diffractionError);
    videorender::fixture3d::RenderedFrame rejectedFoilFrame;
    const bool cpuFallbackRendered = foilPreviewBinding != nullptr
        && foilPreviewRenderer.renderPreview(diffractionScene, foilPreviewBinding, {}, { 320, 180 },
            "cpu-rgb", rejectedFoilFrame, diffractionError);
    ok &= expect(malformedFoilBinding == nullptr && staleFoilBinding == nullptr
                     && unsupportedFoilBinding == nullptr && !cpuFallbackRendered
                     && rejectedFoilFrame.nativeFrame == nullptr
                     && rejectedFoilFrame.stats.vertexBytes == 0
                     && rejectedFoilFrame.stats.indexBytes == 0
                     && rejectedFoilFrame.stats.materialBytes == 0
                     && rejectedFoilFrame.stats.textureBytes == 0,
                 "malformed fields, stale digests, unsupported backends, and CPU/RGB fallback fail closed before OpenGL allocation");

    auto transportProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(
            *diffractionProgram);

    transportProgram->diffractionPathCount = 3;
    transportProgram->diffractionMaximumBounceDepth = 1;
    diffractionmaterial::LightingDescription transportLighting;
    transportLighting.pathCount = 3;
    const std::array<diffractionmaterial::LightingPathKind, 3> transportKinds {
        diffractionmaterial::LightingPathKind::Direct,
        diffractionmaterial::LightingPathKind::Environment,
        diffractionmaterial::LightingPathKind::Indirect };
    const std::array<std::array<float, 3>, 3> transportDirections {{
        {{ 0.0f, 0.0f, 1.0f }},
        {{ 0.36f, 0.0f, 0.9329523f }},
        {{ -0.28f, 0.28f, 0.918259f }} }};
    for (std::size_t pathIndex = 0; pathIndex < transportKinds.size(); ++pathIndex)
    {
        auto path = lightingPath;
        path.kind = transportKinds[pathIndex];
        path.bounceDepth = path.kind == diffractionmaterial::LightingPathKind::Indirect ? 1 : 0;
        path.incident.direction = transportDirections[pathIndex];
        transportLighting.paths[pathIndex] = path;
        transportProgram->diffractionPaths[pathIndex]
            = diffractionmaterial::makeGpuLightingPath(
                diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                    *admittedDiffraction, path.incident), path);
    }
    const auto admittedTransportLighting
        = diffractionmaterial::AdmittedLightingPlan::admit(
            transportLighting, diffractionError);
    if (admittedTransportLighting)
    {
        transportProgram->diffractionLightingAdmission
            = std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(
                *admittedTransportLighting);
        transportProgram->programIdentity
            = arbitgpu::nativeFixtureDiffractionProgramIdentity(
                transportProgram->bindingDigest, *admittedTransportLighting);
    }
    const auto transportPreparation = backend.prepare(diffractionScene, transportProgram);
    const auto transportFrame = backend.render(
        diffractionScene, transportPreparation.resources, 96, 96, {});
    const auto transportPixels = transportFrame.rendered
        ? readRgba8(gl, transportFrame.frame) : std::vector<std::uint8_t> {};
    std::array<std::vector<std::uint8_t>, 3> omittedPathPixels;
    bool everyPathPerturbsPixels = transportPreparation.prepared && transportFrame.rendered;
    for (std::size_t pathIndex = 0; pathIndex < omittedPathPixels.size(); ++pathIndex)
    {
        auto omitted = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(
            *transportProgram);

        auto omittedLighting = transportLighting;
        omittedLighting.paths[pathIndex].incident.radiance.fill(0.0f);
        const auto admittedOmitted = diffractionmaterial::AdmittedLightingPlan::admit(
            omittedLighting, diffractionError);
        if (admittedOmitted)
        {
            omitted->diffractionLightingAdmission
                = std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(
                    *admittedOmitted);
            omitted->programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity(
                omitted->bindingDigest, *admittedOmitted);
            omitted->diffractionPaths[pathIndex]
                = diffractionmaterial::makeGpuLightingPath(
                    diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                        *admittedDiffraction,
                        omittedLighting.paths[pathIndex].incident),
                    omittedLighting.paths[pathIndex]);
        }
        const auto omittedPreparation = backend.prepare(diffractionScene, omitted);
        const auto omittedFrame = backend.render(
            diffractionScene, omittedPreparation.resources, 96, 96, {});
        if (omittedFrame.rendered)
            omittedPathPixels[pathIndex] = readRgba8(gl, omittedFrame.frame);
        everyPathPerturbsPixels = everyPathPerturbsPixels && omittedPreparation.prepared
            && omittedFrame.rendered && omittedPathPixels[pathIndex] != transportPixels;
    }

    // The 63 mm card is 159.65 mm ahead of the camera, with a 0.65 radian
    // vertical field of view. This 8 mm shift keeps its front face in frame:
    // the horizontal NDC bounds at 96x96 are approximately [-0.7343, 0.4369].
    arbitgpu::NativeFixtureSceneRuntimeInputs shiftedCamera;
    shiftedCamera.cameraTranslationOffset[0] = 0.008f;
    auto shiftedDiffractionFrame = backend.render (
        diffractionScene, diffractionPreparation.resources, 96, 96, shiftedCamera);
    const auto shiftedDiffractionPixels = shiftedDiffractionFrame.rendered
        ? readRgba8 (gl, shiftedDiffractionFrame.frame)
        : std::vector<std::uint8_t> {};
    reportCoverage ("diffraction-shifted-camera", shiftedDiffractionPixels);

    auto rotatedDescription = diffractionDescription;
    rotatedDescription.geometry.directionUv = { 0.0f, 1.0f };
    auto admittedRotated = diffractionmaterial::admit (
        rotatedDescription, diffractionError);
    auto rotatedProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
            *diffractionProgram);
    rotatedProgram->bindingDigest = admittedRotated
        ? admittedRotated->structuralDigest() : std::string {};
    if (admittedLighting)
        rotatedProgram->programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity(
            rotatedProgram->bindingDigest, *admittedLighting);
    if (admittedRotated)
        rotatedProgram->diffractionPaths[0] = diffractionmaterial::makeGpuLightingPath(
            diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                *admittedRotated, incident), lightingPath);
    auto rotatedPreparation = backend.prepare (diffractionScene, rotatedProgram);
    auto rotatedFrame = backend.render (
        diffractionScene, rotatedPreparation.resources, 96, 96, {});
    const auto rotatedPixels = rotatedFrame.rendered
        ? readRgba8 (gl, rotatedFrame.frame) : std::vector<std::uint8_t> {};
    reportCoverage ("diffraction-rotated", rotatedPixels);

    auto spatialDescription = diffractionDescription;
    spatialDescription.grooveField.mode = diffractionmaterial::GrooveFieldMode::Linear;
    spatialDescription.grooveField.originUv = { 0.5f, 0.5f };
    spatialDescription.grooveField.axisUv = { 1.0f, 0.0f };
    spatialDescription.grooveField.grooveSpacingDeltaNanometresPerUnit = 500.0f;
    spatialDescription.grooveField.orientationDegreesPerUnit = 90.0f;
    auto admittedSpatial = diffractionmaterial::admit (
        spatialDescription, diffractionError);
    auto spatialProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
            *diffractionProgram);
    spatialProgram->bindingDigest = admittedSpatial
        ? admittedSpatial->structuralDigest() : std::string {};
    if (admittedLighting)
        spatialProgram->programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity(
            spatialProgram->bindingDigest, *admittedLighting);
    if (admittedSpatial)
        spatialProgram->diffractionPaths[0] = diffractionmaterial::makeGpuLightingPath(
            diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                *admittedSpatial, incident), lightingPath);
    auto spatialPreparation = backend.prepare (diffractionScene, spatialProgram);
    auto spatialFrame = backend.render (
        diffractionScene, spatialPreparation.resources, 96, 96, {});
    const auto spatialPixels = spatialFrame.rendered
        ? readRgba8 (gl, spatialFrame.frame) : std::vector<std::uint8_t> {};


    auto crossedDescription
        = diffractionmaterial::makeRealtimeCrossedTwoDimensionalGratingPreset();
    auto admittedCrossed = diffractionmaterial::admit (
        crossedDescription, diffractionError);
    // Aim the measured 540 nm (1,1) order at the centre of the unchanged card.
    // Normal incidence puts most nonzero orders outside its narrow field of
    // view, leaving an invariant zero-order reflection after RGBA8 quantization.
    auto crossedIncident = incident;
    const float crossedX = 540.0f / crossedDescription.geometry.grooveSpacingNanometres;
    const float crossedY = 540.0f / crossedDescription.geometry.secondaryGrooveSpacingNanometres;
    crossedIncident.direction = { crossedX, crossedY,
        std::sqrt(1.0f - crossedX * crossedX - crossedY * crossedY) };
    auto crossedLightPath = lightingPath;
    crossedLightPath.incident = crossedIncident;
    auto crossedLighting = lightingDescription;
    crossedLighting.paths[0] = crossedLightPath;
    const auto admittedCrossedLighting = diffractionmaterial::AdmittedLightingPlan::admit(
        crossedLighting, diffractionError);
    auto crossedProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
            *diffractionProgram);
    crossedProgram->bindingDigest = admittedCrossed
        ? admittedCrossed->structuralDigest() : std::string {};
    if (admittedCrossedLighting)
    {
        crossedProgram->diffractionLightingAdmission
            = std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(*admittedCrossedLighting);
        crossedProgram->programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity(
            crossedProgram->bindingDigest, *admittedCrossedLighting);
    }
    if (admittedCrossed)
        crossedProgram->diffractionPaths[0] = diffractionmaterial::makeGpuLightingPath(
            diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                *admittedCrossed, crossedIncident), crossedLightPath);
    auto unequalCrossedBlazedProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
            *crossedProgram);
    unequalCrossedBlazedProgram->programIdentity
        = "opengl-fixture-unequal-crossed-blazed-program";
    unequalCrossedBlazedProgram->diffractionPaths[0].material.control.y
        = static_cast<float> (diffractionmaterial::GrooveProfile::BlazedSawtooth);
    unequalCrossedBlazedProgram->diffractionPaths[0].material.secondaryGeometry.z
        = unequalCrossedBlazedProgram->diffractionPaths[0].material.geometry.z + 1.0f;
    const auto unequalCrossedBlazedPreparation = backend.prepare (
        diffractionScene, unequalCrossedBlazedProgram);
    auto crossedPreparation = backend.prepare (diffractionScene, crossedProgram);
    auto crossedFrame = backend.render (
        diffractionScene, crossedPreparation.resources, 96, 96, {});
    const auto crossedPixels = crossedFrame.rendered
        ? readRgba8 (gl, crossedFrame.frame) : std::vector<std::uint8_t> {};
    reportCoverage ("diffraction-crossed", crossedPixels);

    auto crossedDirectionDescription = crossedDescription;
    crossedDirectionDescription.geometry.directionUv
        = { 0.70710678f, 0.70710678f };
    crossedDirectionDescription.geometry.secondaryDirectionUv
        = { -0.70710678f, 0.70710678f };
    auto admittedCrossedDirection = diffractionmaterial::admit (
        crossedDirectionDescription, diffractionError);
    auto crossedDirectionProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
            *crossedProgram);
    crossedDirectionProgram->bindingDigest = admittedCrossedDirection
        ? admittedCrossedDirection->structuralDigest() : std::string {};
    if (admittedCrossedLighting)
        crossedDirectionProgram->programIdentity
            = arbitgpu::nativeFixtureDiffractionProgramIdentity(
                crossedDirectionProgram->bindingDigest, *admittedCrossedLighting);
    if (admittedCrossedDirection)
        crossedDirectionProgram->diffractionPaths[0] = diffractionmaterial::makeGpuLightingPath(
            diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                *admittedCrossedDirection, crossedIncident), crossedLightPath);
    auto crossedDirectionPreparation = backend.prepare (
        diffractionScene, crossedDirectionProgram);
    auto crossedDirectionFrame = backend.render (
        diffractionScene, crossedDirectionPreparation.resources, 96, 96, {});
    const auto crossedDirectionPixels = crossedDirectionFrame.rendered
        ? readRgba8 (gl, crossedDirectionFrame.frame)
        : std::vector<std::uint8_t> {};
    reportCoverage ("diffraction-crossed-direction", crossedDirectionPixels);

    auto crossedPeriodDescription = crossedDescription;
    crossedPeriodDescription.geometry.secondaryGrooveSpacingNanometres = 900.0f;
    auto admittedCrossedPeriod = diffractionmaterial::admit (
        crossedPeriodDescription, diffractionError);
    auto crossedPeriodProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (
            *crossedProgram);
    crossedPeriodProgram->bindingDigest = admittedCrossedPeriod
        ? admittedCrossedPeriod->structuralDigest() : std::string {};
    if (admittedCrossedLighting)
        crossedPeriodProgram->programIdentity
            = arbitgpu::nativeFixtureDiffractionProgramIdentity(
                crossedPeriodProgram->bindingDigest, *admittedCrossedLighting);
    if (admittedCrossedPeriod)
        crossedPeriodProgram->diffractionPaths[0] = diffractionmaterial::makeGpuLightingPath(
            diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                *admittedCrossedPeriod, crossedIncident), crossedLightPath);
    auto crossedPeriodPreparation = backend.prepare (
        diffractionScene, crossedPeriodProgram);
    auto crossedPeriodFrame = backend.render (
        diffractionScene, crossedPeriodPreparation.resources, 96, 96, {});
    const auto crossedPeriodPixels = crossedPeriodFrame.rendered
        ? readRgba8 (gl, crossedPeriodFrame.frame)
        : std::vector<std::uint8_t> {};
    reportCoverage ("diffraction-crossed-period", crossedPeriodPixels);

    auto primaryPeriodDescription = crossedDescription;
    primaryPeriodDescription.geometry.grooveSpacingNanometres = 1350.0f;
    auto admittedPrimaryPeriod = diffractionmaterial::admit (
        primaryPeriodDescription, diffractionError);
    auto primaryPeriodProgram
        = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram> (*crossedProgram);
    primaryPeriodProgram->programIdentity
        = "opengl-fixture-diffraction-primary-period-program";
    primaryPeriodProgram->bindingDigest = admittedPrimaryPeriod
        ? admittedPrimaryPeriod->structuralDigest() : std::string {};
    if (admittedCrossedLighting)
        primaryPeriodProgram->programIdentity
            = arbitgpu::nativeFixtureDiffractionProgramIdentity(
                primaryPeriodProgram->bindingDigest, *admittedCrossedLighting);
    if (admittedPrimaryPeriod)
        primaryPeriodProgram->diffractionPaths[0]
            = diffractionmaterial::makeGpuLightingPath(
                diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                    *admittedPrimaryPeriod, crossedIncident), crossedLightPath);
    auto primaryPeriodPreparation = backend.prepare (diffractionScene, primaryPeriodProgram);
    auto primaryPeriodFrame = backend.render (
        diffractionScene, primaryPeriodPreparation.resources, 96, 96, {});
    const auto primaryPeriodPixels = primaryPeriodFrame.rendered
        ? readRgba8 (gl, primaryPeriodFrame.frame) : std::vector<std::uint8_t> {};
    reportCoverage ("diffraction-primary-period", primaryPeriodPixels);

    const auto overBudgetCrossedFrame = backend.render (
        diffractionScene, crossedPreparation.resources, 3840, 2160, {});

    const auto backgroundOnly = [] (const std::vector<std::uint8_t>& pixels)
    {
        for (std::size_t index = 0; index + 3 < pixels.size(); index += 4)
            if (pixels[index] != 7u || pixels[index + 1] != 10u
                || pixels[index + 2] != 18u || pixels[index + 3] != 255u)
                return false;
        return true;
    };
    ok &= expect (admittedDiffraction.has_value()
                      && everyPathPerturbsPixels
                      && diffractionPreparation.prepared
                      && diffractionFrame.rendered && shiftedDiffractionFrame.rendered
                      && rotatedPreparation.prepared && rotatedFrame.rendered
                      && admittedSpatial.has_value()
                      && spatialPreparation.prepared && spatialFrame.rendered
                      && admittedCrossed.has_value()
                      && crossedPreparation.prepared && crossedFrame.rendered
                      && admittedCrossedDirection.has_value()
                      && crossedDirectionPreparation.prepared
                      && crossedDirectionFrame.rendered
                      && admittedCrossedPeriod.has_value()
                      && crossedPeriodPreparation.prepared
                      && crossedPeriodFrame.rendered
                      && admittedPrimaryPeriod.has_value()
                      && primaryPeriodPreparation.prepared
                      && primaryPeriodFrame.rendered
                      && diffractionPixels.size() == 96u * 96u * 4u
                      && ! backgroundOnly (diffractionPixels)
                      && diffractionPixels != shiftedDiffractionPixels
                      // The clear colour is opaque, so alpha cannot prove
                      // that the shifted camera still frames the grating.
                      && nonBackgroundCount(shiftedDiffractionPixels) > 3000
                      && nonBackgroundCount(diffractionPixels) > 3000
                      && diffractionPixels != rotatedPixels
                      && spatialPixels != diffractionPixels
                      && spatialPixels != rotatedPixels

                      && ! backgroundOnly (crossedPixels)
                      && crossedPixels != diffractionPixels
                      && crossedDirectionPixels != crossedPixels
                      && primaryPeriodPixels != crossedPixels
                      && crossedPeriodPixels != crossedPixels,
                  "physical OpenGL diffraction pixels must execute every admitted lighting path and depend on camera, local UV groove fields, crossed-lattice orientation, both periods, and topology");
    ok &= expect (! overBudgetCrossedFrame.rendered
                      && overBudgetCrossedFrame.frame == nullptr
                      && overBudgetCrossedFrame.error
                           == "OpenGL fixture diffraction workload exceeds backend limits",
                  "OpenGL rejects over-budget crossed diffraction before frame allocation");
    ok &= expect (! malformedDiffractionPreparation.prepared
                      && malformedDiffractionPreparation.resources == nullptr
                      && malformedDiffractionPreparation.stats.vertexBytes == 0
                      && malformedDiffractionPreparation.stats.indexBytes == 0
                      && malformedDiffractionPreparation.stats.materialBytes == 0
                      && malformedDiffractionPreparation.stats.textureBytes == 0
                      && malformedDiffractionPreparation.error
                           == "OpenGL fixture preparation requires an exact bounded material binding",
                  "OpenGL rejects malformed diffraction GPU parameters before resource allocation");
    ok &= expect (! replayedLightingPreparation.prepared
                      && replayedLightingPreparation.resources == nullptr
                      && replayedLightingPreparation.stats.vertexBytes == 0
                      && replayedLightingPreparation.stats.indexBytes == 0
                      && replayedLightingPreparation.stats.materialBytes == 0
                      && replayedLightingPreparation.stats.textureBytes == 0
                      && replayedLightingPreparation.error
                           == "OpenGL fixture preparation requires an exact bounded material binding",
                  "OpenGL rejects a replayed spectral receipt before resource allocation");
    ok &= expect (! layoutSentinelPreparation.prepared
                      && layoutSentinelPreparation.resources == nullptr
                      && layoutSentinelPreparation.stats.vertexBytes == 0
                      && layoutSentinelPreparation.stats.indexBytes == 0
                      && layoutSentinelPreparation.stats.materialBytes == 0
                      && layoutSentinelPreparation.stats.textureBytes == 0
                      && layoutSentinelPreparation.error
                           == "OpenGL fixture preparation requires an exact bounded material binding",
                  "OpenGL rejects reused layout-sentinel data before GPU allocation");
    ok &= expect (! divergentBlazePreparation.prepared
                      && divergentBlazePreparation.resources == nullptr
                      && divergentBlazePreparation.stats.vertexBytes == 0
                      && divergentBlazePreparation.stats.indexBytes == 0
                      && divergentBlazePreparation.stats.materialBytes == 0
                      && divergentBlazePreparation.stats.textureBytes == 0
                      && divergentBlazePreparation.error
                           == "OpenGL fixture preparation requires an exact bounded material binding",
                  "OpenGL rejects local period/blaze divergence before GPU allocation");
    ok &= expect (! unequalCrossedBlazedPreparation.prepared
                      && unequalCrossedBlazedPreparation.resources == nullptr
                      && unequalCrossedBlazedPreparation.error
                           == "OpenGL fixture preparation requires an exact bounded material binding",
                  "OpenGL preflight rejects unequal crossed blazed periods before GPU allocation");
    ok &= expect (diffractionFrame.rendered,
                  "OpenGL preparation owns an immutable material snapshot after caller alias mutation");

    auto* alternateContext = glfwCreateWindow (
        16, 16, "fixture-scene-alternate-context", nullptr, nullptr);
    ok &= expect (alternateContext != nullptr,
                  "a second context must be available for cleanup ownership coverage");
    if (alternateContext != nullptr)
        glfwMakeContextCurrent (alternateContext);

    first.frame.reset();
    second.frame.reset();
    preparation.resources.reset();
    clipped.frame.reset();
    clippedPreparation.resources.reset();
    textured.frame.reset();
    texturedPreparation.resources.reset();
    diffractionFrame.frame.reset();
    shiftedDiffractionFrame.frame.reset();
    diffractionPreparation.resources.reset();
    rotatedFrame.frame.reset();
    rotatedPreparation.resources.reset();
    spatialFrame.frame.reset();
    spatialPreparation.resources.reset();
    crossedFrame.frame.reset();
    crossedPreparation.resources.reset();
    crossedDirectionFrame.frame.reset();
    crossedDirectionPreparation.resources.reset();
    crossedPeriodFrame.frame.reset();
    crossedPeriodPreparation.resources.reset();
    ok &= expect (alternateContext == nullptr || glfwGetCurrentContext() == alternateContext,
                  "fixture resource cleanup must restore the caller's OpenGL context");

    glfwMakeContextCurrent (window);
    backend.releaseCachedProgramsForCurrentContext();
    gl.DeleteBuffers (1, &callerPixelUnpackBuffer);
    gl.DeleteFramebuffers (1, &callerFramebuffer);
    glDeleteTextures (1, &callerTexture);
    gl.DeleteSamplers (1, &callerSampler);
    if (alternateContext != nullptr)
        glfwDestroyWindow (alternateContext);
    glfwDestroyWindow (window);
    glfwTerminate();

    if (! ok)
        return 1;
    std::cout << "OpenGL fixture-scene backend: PASS; device="
              << info.device << "; rgba8-fnv1a64=" << std::hex
              << fnv1a64 (firstPixels) << '\n';
    return 0;
}
