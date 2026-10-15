#pragma once
#include "scene_closest_hit_cases.h"
#include "../../src/gl_loader.h"

namespace videohelper::tests::sceneclosest
{
inline bool runGl(std::string& error)
{
    arbitgl::GlFuncs gl;
    if(!arbitgl::loadGlFunctions(gl,error))return false;
    const std::string vertex=R"shader(#version 330 core
void main() { vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.0-1.0,0.0,1.0); }
)shader";
    const std::string fragment=std::string("#version 330 core\n")+glslShader()+outputShader()+R"shader(
uniform vec3 rayOrigin,rayDirection;
uniform vec2 rayBounds;
out vec4 color;
void main() { color=sceneTestOutput(sceneClosestHit(rayOrigin,rayDirection,rayBounds.x,rayBounds.y,uOpaqueControl),int(gl_FragCoord.x)); }
)shader";
    auto compile=[&](GLenum type,const std::string& source) {
        GLuint shader=gl.CreateShader(type);const char* text=source.c_str();gl.ShaderSource(shader,1,&text,nullptr);gl.CompileShader(shader);
        GLint okay=0;gl.GetShaderiv(shader,GL_COMPILE_STATUS,&okay);
        if(!okay) {std::array<char,8192> log{};gl.GetShaderInfoLog(shader,log.size(),nullptr,log.data());error=log.data();gl.DeleteShader(shader);return GLuint(0);}return shader;
    };
    const GLuint vs=compile(GL_VERTEX_SHADER,vertex),fs=compile(GL_FRAGMENT_SHADER,fragment);
    if(!vs||!fs) {if(vs)gl.DeleteShader(vs);if(fs)gl.DeleteShader(fs);return false;}
    GLuint program=gl.CreateProgram();gl.AttachShader(program,vs);gl.AttachShader(program,fs);gl.LinkProgram(program);
    gl.DeleteShader(vs);gl.DeleteShader(fs);GLint okay=0;gl.GetProgramiv(program,GL_LINK_STATUS,&okay);
    if(!okay) {std::array<char,8192> log{};gl.GetProgramInfoLog(program,log.size(),nullptr,log.data());error=log.data();gl.DeleteProgram(program);return false;}
    GLuint texture=0,target=0,fbo=0,vao=0;
    glGenTextures(1,&texture);glGenTextures(1,&target);gl.GenFramebuffers(1,&fbo);gl.GenVertexArrays(1,&vao);
    glBindTexture(GL_TEXTURE_2D,target);glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,4,1,0,GL_RGBA,GL_FLOAT,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    gl.BindFramebuffer(GL_FRAMEBUFFER,fbo);gl.FramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,target,0);
    if(gl.CheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)error="closest-hit GL float target is incomplete";
    gl.UseProgram(program);gl.Uniform1i(gl.GetUniformLocation(program,"uOpaqueVisibility"),0);
    auto execute=[&](const Frame& frame,const Ray& ray,Result& result,std::string& why) {
        gl.ActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,texture);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,kTextureWidth,static_cast<GLsizei>(frame.height),0,GL_RGBA,GL_FLOAT,frame.texels.data());
        gl.Uniform4fv(gl.GetUniformLocation(program,"uOpaqueControl"),1,frame.control.data());
        gl.Uniform3f(gl.GetUniformLocation(program,"rayOrigin"),ray.origin[0],ray.origin[1],ray.origin[2]);
        gl.Uniform3f(gl.GetUniformLocation(program,"rayDirection"),ray.direction[0],ray.direction[1],ray.direction[2]);
        gl.Uniform2f(gl.GetUniformLocation(program,"rayBounds"),ray.minimum,ray.maximum);
        gl.BindFramebuffer(GL_FRAMEBUFFER,fbo);gl.BindVertexArray(vao);glViewport(0,0,4,1);
        glDisable(GL_BLEND);glDisable(GL_DEPTH_TEST);glDisable(GL_CULL_FACE);glDisable(GL_SCISSOR_TEST);
        glDrawArrays(GL_TRIANGLES,0,3);glReadPixels(0,0,4,1,GL_RGBA,GL_FLOAT,result.data());
        if(glGetError()!=GL_NO_ERROR) {why="closest-hit GL execution/readback failed";return false;}return true;
    };
    const bool passed=error.empty()&&run(execute,error);
    gl.UseProgram(0);gl.BindFramebuffer(GL_FRAMEBUFFER,0);gl.BindVertexArray(0);
    gl.DeleteProgram(program);gl.DeleteFramebuffers(1,&fbo);gl.DeleteVertexArrays(1,&vao);
    glDeleteTextures(1,&target);glDeleteTextures(1,&texture);return passed;
}
}
