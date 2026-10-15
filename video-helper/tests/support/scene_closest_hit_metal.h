#pragma once
#include "scene_closest_hit_cases.h"
#import <Metal/Metal.h>
#include <cstring>

namespace videohelper::tests::sceneclosest
{
inline bool runMetal(std::string& error)
{
    @autoreleasepool {
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        if(device==nil) {error="closest-hit Metal device unavailable";return false;}
        const std::string source=std::string("#include <metal_stdlib>\nusing namespace metal;\n")+metalShader()+outputShader()+R"shader(
kernel void closestTest(texture2d<float> opaqueTexture [[texture(0)]],
    constant float4* rays [[buffer(0)]],device float4* result [[buffer(1)]],uint pixel [[thread_position_in_grid]]) {
    SceneClosestHit hit=sceneClosestHit(rays[0].xyz,rays[1].xyz,rays[2].x,rays[2].y,rays[3],opaqueTexture);
    result[pixel]=sceneTestOutput(hit,int(pixel));
}
)shader";
        NSError* compileError=nil;
        id<MTLLibrary> library=[device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:nil error:&compileError];
        id<MTLFunction> function=library?[library newFunctionWithName:@"closestTest"]:nil;
        id<MTLComputePipelineState> pipeline=function?[device newComputePipelineStateWithFunction:function error:&compileError]:nil;
        id<MTLCommandQueue> queue=[device newCommandQueue];
        if(pipeline==nil||queue==nil)error=compileError?[[compileError localizedDescription] UTF8String]:"closest-hit Metal pipeline unavailable";
        auto execute=[&](const Frame& frame,const Ray& ray,Result& result,std::string& why) {
            MTLTextureDescriptor* desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:kTextureWidth height:frame.height mipmapped:NO];
            desc.storageMode=MTLStorageModeShared;desc.usage=MTLTextureUsageShaderRead;
            id<MTLTexture> texture=[device newTextureWithDescriptor:desc];
            const std::array<Texel,4> uniforms{{{ray.origin[0],ray.origin[1],ray.origin[2],0},
                {ray.direction[0],ray.direction[1],ray.direction[2],0},{ray.minimum,ray.maximum,0,0},frame.control}};
            id<MTLBuffer> input=[device newBufferWithBytes:uniforms.data() length:sizeof(uniforms) options:MTLResourceStorageModeShared];
            id<MTLBuffer> output=[device newBufferWithLength:sizeof(result) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> command=[queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            bool okay=texture!=nil&&input!=nil&&output!=nil&&encoder!=nil;
            if(okay) {
                [texture replaceRegion:MTLRegionMake2D(0,0,kTextureWidth,frame.height) mipmapLevel:0 withBytes:frame.texels.data() bytesPerRow:kTextureWidth*sizeof(Texel)];
                [encoder setComputePipelineState:pipeline];[encoder setTexture:texture atIndex:0];
                [encoder setBuffer:input offset:0 atIndex:0];[encoder setBuffer:output offset:0 atIndex:1];
                [encoder dispatchThreads:MTLSizeMake(4,1,1) threadsPerThreadgroup:MTLSizeMake(4,1,1)];
            }
            if(encoder!=nil)[encoder endEncoding];
            if(okay) {[command commit];[command waitUntilCompleted];okay=command.status==MTLCommandBufferStatusCompleted;}
            if(okay)std::memcpy(result.data(),output.contents,sizeof(result));else why="closest-hit Metal execution failed";
#if !__has_feature(objc_arc)
            [texture release];[input release];[output release];
#endif
            return okay;
        };
        const bool passed=error.empty()&&run(execute,error);
#if !__has_feature(objc_arc)
        [queue release];[pipeline release];[function release];[library release];[device release];
#endif
        return passed;
    }
}
}
