#include "gpu_export_iosurface.h"

#if ARBIT_HAVE_IOSURFACE

#include "SharedGpuSurfaceProtocol.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurface.h>
#include <CoreVideo/CVPixelBuffer.h>
#include <climits>

static_assert(gpusurf::kIOSurfaceRGBAHalf == kCVPixelFormatType_64RGBAHalf, "float surface format must match Core Video");

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
    (void) gl;
    ready_ = false;
    errorOut.clear();
    ready_ = true;
    return true;
}

bool IOSurfaceExporter::allocate (ExportedBuffer& b, int width, int height,
                                  std::string& errorOut, IOSurfaceFormat format)
{
    if (! ready_)
    {
        errorOut = "exporter not initialized";
        return false;
    }

    if (width <= 0 || height <= 0 || uint64_t(width) * 8 > INT32_MAX)
    { errorOut = "invalid IOSurface extent"; return false; }
    b = ExportedBuffer {};
    b.format = format;
    const bool linear = format == IOSurfaceFormat::LinearRgba16Float;
    const int bytesPerElement = linear ? 8 : 4;
    b.width = width;
    b.height = height;

    CFMutableDictionaryRef props = CFDictionaryCreateMutable (
        kCFAllocatorDefault, 0,
        &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    dictSetInt (props, kIOSurfaceWidth, width);
    dictSetInt (props, kIOSurfaceHeight, height);
    dictSetInt (props, kIOSurfaceBytesPerElement, bytesPerElement);
    dictSetInt (props, kIOSurfacePixelFormat, static_cast<int32_t>(linear
        ? gpusurf::kIOSurfaceRGBAHalf : gpusurf::kIOSurfaceBGRA));
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
    b.rowBytes = IOSurfaceGetBytesPerRow(surf);
    b.allocationBytes = IOSurfaceGetAllocSize(surf);
    if (IOSurfaceGetWidth(surf) != static_cast<size_t>(width)
        || IOSurfaceGetHeight(surf) != static_cast<size_t>(height)
        || IOSurfaceGetBytesPerElement(surf) != static_cast<size_t>(bytesPerElement)
        || IOSurfaceGetPixelFormat(surf) != (linear ? gpusurf::kIOSurfaceRGBAHalf : gpusurf::kIOSurfaceBGRA)
        || b.rowBytes < uint64_t(width) * bytesPerElement
        || b.rowBytes > UINT64_MAX / static_cast<uint64_t>(height)
        || b.allocationBytes < b.rowBytes * height)
    { CFRelease(surf); b = {}; errorOut = "IOSurface allocation does not match its declared format"; return false; }
    b.surface = (void*) surf;
    b.modifier = (uint64_t) IOSurfaceGetID (surf);
    b.fourcc = gpusurf::kHandleIOSurface;
    b.planeCount = 0;

    errorOut.clear();
    return true;
}

void IOSurfaceExporter::destroy (ExportedBuffer& b)
{
    if (b.surface != nullptr)
        CFRelease ((IOSurfaceRef) b.surface);
    b = ExportedBuffer {};
}

bool IOSurfaceExporter::blit (unsigned srcTexture, const ExportedBuffer& b)
{
    (void) srcTexture;
    (void) b;
    return false; // macOS shared presentation is direct Metal-only.
}

void IOSurfaceExporter::shutdown()
{
    ready_ = false;
}

} // namespace gpuexp

#endif // ARBIT_HAVE_IOSURFACE
