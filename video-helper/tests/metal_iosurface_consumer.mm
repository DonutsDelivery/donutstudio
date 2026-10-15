#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include "../../plugin/Source/SharedGpuSurfaceProtocol.h"
#include <atomic>
#include <array>
#include <cmath>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

int main (int argc, char** argv)
{
    if (argc != 4 && argc != 10)
    {
        std::cerr << "usage: metal-iosurface-consumer ID WIDTH HEIGHT [linear REGISTRY ROWBYTES ALLOCATION RELEASEFD GENERATION]\n";
        return 2;
    }

    const auto surfaceId = static_cast<IOSurfaceID> (std::strtoul (argv[1], nullptr, 10));
    const auto width = static_cast<NSUInteger> (std::strtoul (argv[2], nullptr, 10));
    const auto height = static_cast<NSUInteger> (std::strtoul (argv[3], nullptr, 10));
    IOSurfaceRef surface = IOSurfaceLookup (surfaceId);
    if (surface == nullptr || width == 0 || height == 0)
    {
        std::cerr << "consumer could not look up IOSurface " << surfaceId << '\n';
        return 3;
    }

    const bool linear = argc == 10;
    const auto registry = linear ? std::strtoull(argv[5], nullptr, 10) : 0;
    const auto rowBytes = linear ? std::strtoull(argv[6], nullptr, 10) : 0;
    const auto allocation = linear ? std::strtoull(argv[7], nullptr, 10) : 0;
    const int releaseFd = linear ? std::atoi(argv[8]) : -1;
    const auto generation = linear ? std::strtoull(argv[9], nullptr, 10) : 0;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil || IOSurfaceGetWidth(surface) != width || IOSurfaceGetHeight(surface) != height
        || IOSurfaceGetBytesPerElement(surface) != (linear ? 8u : 4u)
        || IOSurfaceGetPixelFormat(surface) != (linear ? gpusurf::kIOSurfaceRGBAHalf : gpusurf::kIOSurfaceBGRA)
        || (linear && ([device registryID] != registry || rowBytes != IOSurfaceGetBytesPerRow(surface)
            || allocation != IOSurfaceGetAllocSize(surface) || rowBytes < width * 8 || generation == 0)))
    { std::cerr << "consumer rejected storage or registry metadata\n"; return 3; }
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:(linear ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm)
                                     width:width height:height mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor
                                                    iosurface:surface plane:0];
    id<MTLBuffer> readback = [device newBufferWithLength:(linear ? 1024 : 256)
                                                 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
    if (texture == nil || readback == nil || queue == nil || commands == nil || blit == nil)
    { std::cerr << "consumer Metal import failed\n"; return 3; }
    for (int q = 0; q < (linear ? 4 : 1); ++q)
    [blit copyFromTexture:texture
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake (linear ? (q % 2 == 0 ? 1 : 3) * width / 4 : width / 2,
                                      linear ? (q / 2 == 0 ? 1 : 3) * height / 4 : height / 2, 0)
               sourceSize:MTLSizeMake (1, 1, 1)
                 toBuffer:readback
        destinationOffset:q * 256
   destinationBytesPerRow:256
 destinationBytesPerImage:256];
    [blit endEncoding];
    std::atomic<bool> completed {false};
    auto* completedFlag = &completed;
    if (linear)
        [commands addCompletedHandler:^(id<MTLCommandBuffer> finished) {
            if (finished.status != MTLCommandBufferStatusCompleted) return;
            const gpusurf::LinearIOSurfaceFrameReleasePayload release {generation, {0, 0}};
            const bool sent = ::write(releaseFd, &release, sizeof(release)) == static_cast<ssize_t>(sizeof(release));
            completedFlag->store(sent);
            ::close(releaseFd);
        }];
    [commands commit];
    [commands waitUntilCompleted];

    if (linear)
    {
        // Independent IEEE binary16 oracle. Do not use the producer's decoder,
        // color transform, final-composite readback or tone-mapping utilities.
        const auto decode = [](uint16_t bits) -> double {
            const int exponent = (bits >> 10) & 31;
            const int mantissa = bits & 1023;
            const double value = exponent == 0 ? std::ldexp(double(mantissa), -24)
                : exponent == 31 ? std::numeric_limits<double>::infinity()
                : std::ldexp(double(1024 + mantissa), exponent - 25);
            return (bits & 0x8000) != 0 ? -value : value;
        };
        constexpr std::array<std::array<double, 4>, 4> expected {{{1, 1, 1, 1},
            {4.92578125, 4.92578125, 4.92578125, 1}, {2, 0.25, 1, 1}, {0.01, 0.04, 0.125, 1}}};
        bool matched = completed.load() && commands.status == MTLCommandBufferStatusCompleted
            && texture.pixelFormat == MTLPixelFormatRGBA16Float;
        const auto* bytes = static_cast<const uint8_t*>([readback contents]);
        for (int q = 0; q < 4 && bytes != nullptr; ++q)
        {
            const auto* half = reinterpret_cast<const uint16_t*>(bytes + q * 256);
            for (int c = 0; c < 4; ++c)
                matched &= std::isfinite(decode(half[c])) && std::abs(decode(half[c]) - expected[q][c]) < 0.003;
        }
        matched &= bytes != nullptr;
        std::cout << "Cross-process RGBA16F quadrants/top-first/>1/GPU-completion-release=" << matched << '\n';
        [readback release]; [texture release]; [queue release]; [device release]; CFRelease(surface);
        return matched ? 0 : 4;
    }

    const auto* bgra = static_cast<const uint8_t*> ([readback contents]);
    const uint8_t zero[4] = {};
    if (bgra == nullptr) bgra = zero;
    const bool red = bgra[0] < 20 && bgra[1] < 20
                  && bgra[2] > 220 && bgra[3] > 220;
    std::cout << "Cross-process IOSurface center BGRA="
              << static_cast<int> (bgra[0]) << ','
              << static_cast<int> (bgra[1]) << ','
              << static_cast<int> (bgra[2]) << ','
              << static_cast<int> (bgra[3]) << '\n';

    [readback release];
    [texture release];
    [queue release];
    [device release];
    CFRelease (surface);
    return red ? 0 : 4;
}
