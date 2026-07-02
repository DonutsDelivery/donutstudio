#include "gpu_export_iosurface.h"

#if ARBIT_HAVE_IOSURFACE

#include "SharedGpuSurfaceProtocol.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurface.h>
#include <OpenGL/OpenGL.h>          // CGL
#include <OpenGL/CGLIOSurface.h>    // CGLTexImageIOSurface2D

namespace gpuexp
{

namespace
{
// CFDictionary helper for IOSurfaceCreate.
void dictSetInt (CFMutableDictionaryRef dict, CFStringRef key, int32_t value)
{
    CFNumberRef num = CFNumberCreate (kCFAllocatorDefault, kCFNumberSInt32Type, &value);
    CFDictionarySetValue (dict, key, num);
    CFRelease (num);
}
} // namespace

bool IOSurfaceExporter::initialize (const arbitgl::GlFuncs* gl, std::string& errorOut)
{
    gl_ = gl;
    ready_ = false;

    cglContext_ = (void*) CGLGetCurrentContext();
    if (cglContext_ == nullptr)
    {
        errorOut = "no current CGL context";
        return false;
    }

    if (readFbo_ == 0)
        gl_->GenFramebuffers (1, &readFbo_);

    ready_ = true;
    return true;
}

bool IOSurfaceExporter::allocate (ExportedBuffer& b, int width, int height,
                                  std::string& errorOut)
{
    if (! ready_)
    {
        errorOut = "exporter not initialized";
        return false;
    }

    b = ExportedBuffer {};
    b.width = width;
    b.height = height;

    CFMutableDictionaryRef props = CFDictionaryCreateMutable (
        kCFAllocatorDefault, 0,
        &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    dictSetInt (props, kIOSurfaceWidth, width);
    dictSetInt (props, kIOSurfaceHeight, height);
    dictSetInt (props, kIOSurfaceBytesPerElement, 4);
    dictSetInt (props, kIOSurfacePixelFormat, (int32_t) 'BGRA');
    // Global surfaces are the only ones IOSurfaceLookup can resolve from
    // another process; deprecated but standard (see header).
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    CFDictionarySetValue (props, kIOSurfaceIsGlobal, kCFBooleanTrue);
#pragma clang diagnostic pop

    IOSurfaceRef surf = IOSurfaceCreate (props);
    CFRelease (props);
    if (surf == nullptr)
    {
        errorOut = "IOSurfaceCreate failed";
        return false;
    }
    b.surface = (void*) surf;
    b.modifier = (uint64_t) IOSurfaceGetID (surf);
    b.fourcc = gpusurf::kHandleIOSurface;
    b.planeCount = 0;

    glGenTextures (1, &b.tex);
    glBindTexture (GL_TEXTURE_RECTANGLE, b.tex);
    const CGLError err = CGLTexImageIOSurface2D (
        (CGLContextObj) cglContext_, GL_TEXTURE_RECTANGLE, GL_RGBA8,
        (GLsizei) width, (GLsizei) height,
        GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, surf, 0);
    if (err != kCGLNoError)
    {
        errorOut = std::string ("CGLTexImageIOSurface2D failed: ")
                 + CGLErrorString (err);
        glBindTexture (GL_TEXTURE_RECTANGLE, 0);
        destroy (b);
        return false;
    }
    glTexParameteri (GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri (GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture (GL_TEXTURE_RECTANGLE, 0);

    gl_->GenFramebuffers (1, &b.fbo);
    gl_->BindFramebuffer (GL_FRAMEBUFFER, b.fbo);
    gl_->FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_RECTANGLE, b.tex, 0);
    if (gl_->CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        errorOut = "IOSurface buffer FBO incomplete";
        gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
        destroy (b);
        return false;
    }
    glViewport (0, 0, width, height);
    glClearColor (0.0f, 0.0f, 0.0f, 1.0f);
    glClear (GL_COLOR_BUFFER_BIT);
    gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
    glFinish(); // defined contents before the consumer can look the ID up

    return true;
}

void IOSurfaceExporter::destroy (ExportedBuffer& b)
{
    if (b.fbo != 0 && gl_ != nullptr)
        gl_->DeleteFramebuffers (1, &b.fbo);
    if (b.tex != 0)
        glDeleteTextures (1, &b.tex);
    if (b.surface != nullptr)
        CFRelease ((IOSurfaceRef) b.surface);
    b = ExportedBuffer {};
}

bool IOSurfaceExporter::blit (unsigned srcTexture, const ExportedBuffer& b)
{
    if (! ready_ || b.fbo == 0)
        return false;

    gl_->BindFramebuffer (GL_READ_FRAMEBUFFER, readFbo_);
    gl_->FramebufferTexture2D (GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, srcTexture, 0);
    gl_->BindFramebuffer (GL_DRAW_FRAMEBUFFER, b.fbo);
    gl_->BlitFramebuffer (0, 0, b.width, b.height, 0, 0, b.width, b.height,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST);
    gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
    return true;
}

void IOSurfaceExporter::shutdown()
{
    if (readFbo_ != 0 && gl_ != nullptr)
    {
        gl_->DeleteFramebuffers (1, &readFbo_);
        readFbo_ = 0;
    }
    ready_ = false;
}

} // namespace gpuexp

#endif // ARBIT_HAVE_IOSURFACE
