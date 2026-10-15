#include "gpu_backend/backend.h"
#include "sdf_native_renderer.h"
#include "sdf_native_program.h"
#include "support/fixture_scene.h"
#include "support/sdf_native_backend_parity.h"

#import <Metal/Metal.h>

#define SOKOL_METAL
#include "sokol_gfx.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <limits>
#include <string>
#include <vector>

namespace arbitgpu
{
// Read-only fixture seam for the private staged result, without an RPC route.
std::array<std::uint64_t, 2> metalSdfPrimaryAllocationForTests (const NativeSdfSceneFrame&);
}

namespace
{
std::vector<std::uint8_t> readBgra8 (
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame)
{
    const sg_image image { static_cast<std::uint32_t> (frame->colorImageHandle()) };
    const auto metal = sg_mtl_query_image_info (image);
    if (metal.active_slot < 0 || metal.active_slot >= SG_NUM_INFLIGHT_FRAMES)
        return {};
    id<MTLTexture> texture = (__bridge id<MTLTexture>) metal.tex[metal.active_slot];
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
    if (texture == nil || queue == nil)
        return {};
    const auto byteCount = static_cast<NSUInteger> (frame->width()) * frame->height() * 4u;
    id<MTLBuffer> readback = [texture.device newBufferWithLength:byteCount
                                                        options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    [blit copyFromTexture:texture
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake (0, 0, 0)
               sourceSize:MTLSizeMake (frame->width(), frame->height(), 1)
                 toBuffer:readback
        destinationOffset:0
   destinationBytesPerRow:frame->width() * 4u
 destinationBytesPerImage:byteCount];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    std::vector<std::uint8_t> pixels (byteCount);
    if (command.status == MTLCommandBufferStatusCompleted && readback.contents != nullptr)
        std::copy_n (static_cast<const std::uint8_t*> (readback.contents), byteCount,
                     pixels.begin());
    else
        pixels.clear();
#if ! __has_feature(objc_arc)
    [readback release];
#endif
    return pixels;
}

videowire::SdfRecord record (videowire::SdfStableId id,
                             videowire::SdfOperation operation,
                             std::initializer_list<videowire::SdfStableId> inputs,
                             std::initializer_list<double> parameters)
{
    videowire::SdfRecord result;
    result.stableId = id;
    result.operation = operation;
    result.inputCount = static_cast<std::uint8_t> (inputs.size());
    result.parameterCount = static_cast<std::uint8_t> (parameters.size());
    std::copy (inputs.begin(), inputs.end(), result.inputs.begin());
    std::copy (parameters.begin(), parameters.end(), result.parameters.begin());
    return result;
}

std::uint32_t checksum (const std::vector<std::uint8_t>& pixels)
{
    std::uint32_t hash = 2166136261u;
    for (const auto value : pixels)
    {
        hash ^= value;
        hash *= 16777619u;
    }
    return hash;
}
} // namespace

// Private readonly diagnostics from the production Metal translation unit.
namespace arbitgpu
{
std::string metalSdfTextShaderSourceForTests();
std::vector<std::array<std::uint64_t, 3>> metalSdfTextAllocationsForTests (const NativeSdfSceneFrame&);
}

namespace
{
videowire::SdfTextOutline textRectangle (bool counter = true, double extent = 1.0,
    const char* text = "O", const char* font = "InterRegular_ttf")
{
    // These analytic vectors test the consumer. They do not establish font origin.
    videowire::SdfTextOutline outline;
    outline.emSize = extent;
    outline.sourceText = text;
    outline.sourceTextSha256 = videowire::sdfTextSourceSha256 (text);
    outline.embeddedFontToken = font;
    outline.embeddedFontSha256 = videowire::sdfTextEmbeddedFontSha256 (font);
    outline.glyphs.push_back ({ 1, 0, 0, 0, 2, 0 });
    outline.contours.push_back ({ 0, 0, 4, 1 });
    outline.segments = { { -extent,-extent,extent,-extent }, { extent,-extent,extent,extent },
                         { extent,extent,-extent,extent }, { -extent,extent,-extent,-extent } };
    if (counter)
    {
        outline.contours.push_back ({ 0, 4, 4, -1 });
        const videowire::SdfTextSegment hole[] {
            { -.5,-.5,-.5,.5 }, { -.5,.5,.5,.5 }, { .5,.5,.5,-.5 }, { .5,-.5,-.5,-.5 }
        };
        outline.segments.insert (outline.segments.end(), std::begin (hole), std::end (hole));
    }
    outline.stableId = videowire::sdfTextOutlineStableId (outline);
    return outline;
}

arbitgpu::NativeSdfDrawRequest textRequest (videowire::SdfTextOutline outline = textRectangle(),
    double angle = 0.0, double depth = .5)
{
    arbitgpu::NativeSdfDrawRequest request;
    request.geometry.schemaVersion = videowire::kSdfIrTextSchemaVersion;
    auto leaf = record (41, videowire::SdfOperation::textExtrusion, {}, { depth });
    leaf.resourceStableId = outline.stableId;
    request.geometry.rootId = leaf.stableId;
    request.geometry.records.push_back (leaf);
    request.geometry.textOutlines.push_back (std::move (outline));
    if (angle != 0)
    {
        request.geometry.records.push_back (record (42, videowire::SdfOperation::rotate, {41}, {0,1,0,angle}));
        request.geometry.rootId = 42;
    }
    request.width = request.height = 49;
    request.maximumSteps = 256;
    request.epsilon = .00001;
    request.maximumDistance = 10;
    request.adaptiveQuality = request.normalQuality = arbitgpu::NativeSdfQuality::high;
    request.output = arbitgpu::NativeSdfOutput::normal;
    return request;
}

// Closed-form rectangle distances are independent of contour tables, segment
// scans, crossing rules and the shared SDF oracle.
double rectangleField (double x, double y, double extent)
{
    double a = std::abs (x) - extent, b = std::abs (y) - extent;
    if (a <= 0 && b <= 0) return std::max (a,b);
    if (a <= 0) return b;
    if (b <= 0) return a;
    return std::sqrt (a*a+b*b);
}

double ringField (double x, double y, double z)
{
    const double plane = std::max (rectangleField (x,y,1), -rectangleField (x,y,.5));
    const double cap = std::abs (z) - .25;
    if (plane <= 0 && cap <= 0) return std::max (plane,cap);
    if (plane <= 0) return cap;
    if (cap <= 0) return plane;
    return std::sqrt (plane*plane+cap*cap);
}

bool verifyMetalTextField()
{
    const auto request = textRequest();
    std::string error;
    const auto program = videohelper::sdf::compileNativeSdfProgram (request.geometry, error, true);
    if (! program) return false;
    std::vector<std::array<float,4>> points;
    for (const auto xy : std::array<std::array<float,2>, 12> {{
        {{0,0}}, {{.75f,0}}, {{1,0}}, {{.5f,.5f}}, {{1.25f,1.5f}}, {{-.75f,.8f}},
        {{0,.5f}}, {{-.5f,.5f}}, {{.5f,0}}, {{1,1}}, {{-.5f,-.5f}}, {{-1,-1}} }})
        for (float z : { -.75f,-.25f,0.f,.25f,.75f }) points.push_back ({xy[0],xy[1],z,0});
    for(float boundary : {-.5f,.5f,-1.f,1.f})
        for(float adjacent : {std::nextafter(boundary,-std::numeric_limits<float>::infinity()),
                              std::nextafter(boundary,std::numeric_limits<float>::infinity())}) {
            points.push_back({adjacent,0,0,0});
            points.push_back({0,adjacent,0,0});
        }
    const std::string source = arbitgpu::metalSdfTextShaderSourceForTests() + R"metal(
    kernel void textFieldProbe(const device float4* points [[buffer(0)]],device float* output [[buffer(1)]],
        const device uint* indices [[buffer(2)]],const device uint4* resources [[buffer(3)]],
        const device uint4* contours [[buffer(4)]],const device float4* segments [[buffer(5)]],
        constant float& depth [[buffer(6)]],const device TextIndexNode* nodes [[buffer(7)]],
        const device uint* segmentIndices [[buffer(8)]],const device uint* roots [[buffer(9)]],
        uint i [[thread_position_in_grid]]) {
        TextTables tables={indices,resources,contours,segments,nodes,segmentIndices,roots};
        output[i]=textDistance(points[i].xyz,0u,depth,tables);
    })metal";
    id<MTLDevice> device = (__bridge id<MTLDevice>) sg_mtl_device();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
    if (device == nil || queue == nil || queue.device != device) return false;
    NSError* diagnostic = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                options:nil error:&diagnostic];
    id<MTLFunction> function = [library newFunctionWithName:@"textFieldProbe"];
    id<MTLComputePipelineState> pipeline = function != nil
        ? [device newComputePipelineStateWithFunction:function error:&diagnostic] : nil;
    const float depth = .5f;
    const std::array<sg_range,10> ranges {{
        {points.data(),points.size()*sizeof(points[0])}, {nullptr,points.size()*sizeof(float)},
        {program->textResourceIndices().data(),program->textResourceIndices().size()*4},
        {program->textResources().data(),program->textResources().size()*16},
        {program->textContours().data(),program->textContours().size()*16},
        {program->textSegments().data(),program->textSegments().size()*16}, {&depth,sizeof(depth)},
        {program->textIndexNodes().data(),program->textIndexNodes().size()*sizeof(arbitgpu::NativeSdfTextIndexNode)},
        {program->textSegmentIndices().data(),program->textSegmentIndices().size()*4},
        {program->textIndexRoots().data(),program->textIndexRoots().size()*4}
    }};
    std::array<id<MTLBuffer>,10> buffers {};
    bool valid = pipeline != nil;
    for (std::size_t i=0;i<buffers.size();++i)
    {
        buffers[i] = ranges[i].ptr != nullptr
            ? [device newBufferWithBytes:ranges[i].ptr length:ranges[i].size options:MTLResourceStorageModeShared]
            : [device newBufferWithLength:ranges[i].size options:MTLResourceStorageModeShared];
        valid = valid && buffers[i] != nil;
    }
    if (valid)
    {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        valid = command != nil && encoder != nil;
        if (valid)
        {
            [encoder setComputePipelineState:pipeline];
            for (std::size_t i=0;i<buffers.size();++i) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
            [encoder dispatchThreads:MTLSizeMake(points.size(),1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            valid = command.status == MTLCommandBufferStatusCompleted && buffers[1].contents != nullptr;
            if (valid)
                for (std::size_t i=0;i<points.size();++i)
                {
                    const double expected = ringField (points[i][0],points[i][1],points[i][2]);
                    const float actual = static_cast<const float*> (buffers[1].contents)[i];
                    if (! std::isfinite(actual) || std::abs(actual-expected)>2e-6
                        || (expected!=0&&std::signbit(actual)!=std::signbit(expected)))
                    {
                        std::fprintf(stderr,"Metal text field oracle mismatch at %zu: %.9g vs %.17g\n",i,actual,expected);
                        valid = false; break;
                    }
                }
        }
    }
    if (! valid && diagnostic != nil)
        std::fprintf (stderr,"Metal text field native shader diagnostic: %s\n",diagnostic.localizedDescription.UTF8String);
#if ! __has_feature(objc_arc)
    for (id<MTLBuffer> buffer : buffers) [buffer release];
    [pipeline release]; [function release]; [library release];
#endif
    return valid;
}

// Independent slab intersection of four solid boxes forming the ring. This
// oracle never marches the field or reads compiled segments.
struct RingHit { double distance = 10; std::array<double,3> normal {}; int axis = -1; };
RingHit ringRay (std::array<double,3> origin, std::array<double,3> direction, double angle, double depth)
{
    const double c=std::cos(angle), s=std::sin(angle);
    const auto local=[&](std::array<double,3> p) { return std::array<double,3>{c*p[0]-s*p[2],p[1],s*p[0]+c*p[2]}; };
    origin=local(origin); direction=local(direction);
    const std::array<std::array<double,4>,4> boxes {{
        {{-1,-.5,-1,1}}, {{.5,1,-1,1}}, {{-.5,.5,-1,-.5}}, {{-.5,.5,.5,1}} }};
    RingHit closest;
    for (const auto box : boxes)
    {
        const std::array<double,3> low {box[0],box[2],-depth/2}, high {box[1],box[3],depth/2};
        double entry=0,exit=10; int axis=-1; double sign=0;
        for (int k=0;k<3;++k)
        {
            if (std::abs(direction[k])<1e-14)
            { if(origin[k]<low[k]||origin[k]>high[k]) exit=-1; continue; }
            double a=(low[k]-origin[k])/direction[k], b=(high[k]-origin[k])/direction[k];
            const double entrySign=direction[k]>0?-1:1;
            if(a>b)std::swap(a,b);
            if(a>entry){entry=a;axis=k;sign=entrySign;}
            exit=std::min(exit,b);
        }
        if(entry>exit||axis<0||entry>=closest.distance)continue;
        std::array<double,3> normal {}; normal[axis]=sign;
        closest={entry,{c*normal[0]+s*normal[2],normal[1],-s*normal[0]+c*normal[2]},axis};
    }
    return closest;
}

std::vector<float> readTextRawNormals (const arbitgpu::NativeSdfSceneFrame& frame)
{
    const auto view=frame.normalTextureDescriptor();
    if(!view.complete()||view.format!=arbitgpu::NativeTexturePixelFormat::Rgba16Float)return {};
    const auto native=sg_mtl_query_image_info(sg_image{std::uint32_t(view.imageHandle)});
    if(native.active_slot<0||native.active_slot>=SG_NUM_INFLIGHT_FRAMES)return {};
    id<MTLTexture> texture=(__bridge id<MTLTexture>)native.tex[native.active_slot];
    id<MTLCommandQueue> queue=(__bridge id<MTLCommandQueue>)sg_mtl_command_queue();
    if(texture==nil||queue==nil||texture.device!=queue.device||texture.pixelFormat!=MTLPixelFormatRGBA16Float)return {};
    const NSUInteger row=(NSUInteger(frame.width())*8+255)&~NSUInteger(255),bytes=row*frame.height();
    id<MTLBuffer> readback=[texture.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
    std::vector<float> output;
    if(readback!=nil&&command!=nil&&blit!=nil)
    {
        [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
            sourceSize:MTLSizeMake(frame.width(),frame.height(),1) toBuffer:readback destinationOffset:0
            destinationBytesPerRow:row destinationBytesPerImage:bytes];
        [blit endEncoding];[command commit];[command waitUntilCompleted];
        if(command.status==MTLCommandBufferStatusCompleted&&readback.contents!=nullptr)
        {
            output.resize(std::size_t(frame.width())*frame.height()*4);
            for(std::size_t y=0;y<frame.height();++y)
            {
                const auto* source=reinterpret_cast<const std::uint16_t*>(static_cast<const std::uint8_t*>(readback.contents)+y*row);
                for(std::size_t x=0;x<std::size_t(frame.width())*4;++x)
                {
                    const auto bits=source[x];const auto exponent=(bits>>10)&31,fraction=bits&1023;
                    const float magnitude=exponent==0?std::ldexp(float(fraction),-24)
                        :exponent==31?std::numeric_limits<float>::infinity()
                        :std::ldexp(float(1024+fraction),int(exponent)-25);
                    output[y*frame.width()*4+x]=(bits&32768)?-magnitude:magnitude;
                }
            }
        }
    }
    else if(blit!=nil)[blit endEncoding];
#if ! __has_feature(objc_arc)
    [readback release];
#endif
    return output;
}

std::vector<float> readTextPrimary(const arbitgpu::NativeSdfSceneFrame& frame)
{
    const auto allocation=arbitgpu::metalSdfPrimaryAllocationForTests(frame);
    if(allocation[0]==0)return {};
    const auto native=sg_mtl_query_image_info(sg_image{std::uint32_t(allocation[0])});
    if(native.active_slot<0||native.active_slot>=SG_NUM_INFLIGHT_FRAMES)return {};
    id<MTLTexture> texture=(__bridge id<MTLTexture>)native.tex[native.active_slot];
    id<MTLCommandQueue> queue=(__bridge id<MTLCommandQueue>)sg_mtl_command_queue();
    if(texture==nil||queue==nil||texture.device!=queue.device||texture.pixelFormat!=MTLPixelFormatRG32Float)return {};
    const NSUInteger row=(NSUInteger(frame.width())*8+255)&~NSUInteger(255),bytes=row*frame.height();
    id<MTLBuffer> readback=[texture.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
    std::vector<float> output;
    if(readback!=nil&&command!=nil&&blit!=nil) {
        [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
            sourceSize:MTLSizeMake(frame.width(),frame.height(),1) toBuffer:readback destinationOffset:0
            destinationBytesPerRow:row destinationBytesPerImage:bytes];
        [blit endEncoding];[command commit];[command waitUntilCompleted];
        if(command.status==MTLCommandBufferStatusCompleted&&readback.contents!=nullptr) {
            output.resize(std::size_t(frame.width())*frame.height()*2);
            for(std::size_t y=0;y<frame.height();++y)
                std::memcpy(output.data()+y*frame.width()*2,
                    static_cast<const std::uint8_t*>(readback.contents)+y*row,frame.width()*8);
        }
    } else if(blit!=nil)[blit endEncoding];
#if ! __has_feature(objc_arc)
    [readback release];
#endif
    return output;
}

bool verifyMetalTextFrames()
{
    if (! verifyMetalTextField()) return false;
    std::size_t caps=0,sides=0,holes=0;
    for (double angle : {0.0,1.05,3.141592653589793})
    {
        const auto request=textRequest(textRectangle(),angle);
        const auto rendered=arbitgpu::nativeSdfExecutionBackend().render(request);
        std::vector<float> depth,color;
        if(!rendered.rendered||!rendered.frame||!rendered.frame->readDepthFloatPixels(depth)
            ||!rendered.frame->readColorFloatPixels(color)||depth.size()!=49*49||color.size()!=49*49*4)return false;
        const auto rawNormals=readTextRawNormals(*rendered.frame);
        if(rawNormals.size()!=color.size())return false;
        const auto primaryPixels=readTextPrimary(*rendered.frame);
        if(primaryPixels.size()!=depth.size()*2)return false;
        for(std::size_t i=0;i<depth.size();++i) {
            const bool hit=primaryPixels[i*2]!=0;
            const float expectedDepth=hit?std::clamp(primaryPixels[i*2+1]/float(request.maximumDistance),0.f,1.f):1.f;
            if(std::abs(depth[i]-expectedDepth)>2*std::numeric_limits<float>::epsilon()
                ||rawNormals[i*4+3]!=(hit?1.f:0.f))return false;
        }
        const auto& receipt=rendered.frame->sdfResourceReceipt();
        const auto allocations=arbitgpu::metalSdfTextAllocationsForTests(*rendered.frame);
        std::uint64_t logical=0,allocated=0;
        for(auto allocation:allocations)
        {
            if(allocation[0]==0||allocation[0]>allocation[2]||allocation[1]<allocation[0])return false;
            logical+=allocation[0]; allocated+=allocation[1];
        }
        std::string error;
        const auto program=videohelper::sdf::compileNativeSdfProgram(request.geometry,error,true);
        arbitgpu::NativeSdfWorkReceipt stagedWork;
        if(!videohelper::sdf::admitNativeSdfStagedTextWork(request,stagedWork,error)
            ||receipt.distanceQueryCount!=stagedWork.distanceQueryCount
            ||receipt.textIndexNodeVisits!=stagedWork.textIndexNodeVisits)return false;
        if(!program||allocations.size()!=7||logical<program->textTableBytes()
            ||allocated!=receipt.nativeTextTableBytes||receipt.compiledTextTableBytes!=program->textTableBytes()
            ||receipt.segmentVisits!=receipt.distanceQueryCount*8
            ||receipt.recordVisits!=receipt.distanceQueryCount*request.geometry.records.size()
            ||receipt.totalBytes!=program->storageBytes()+receipt.backendProgramBytes+receipt.uniformBytes
                +receipt.attachmentBytes+allocated
            ||receipt.transientUploadBytes!=0||receipt.peakBytes!=receipt.totalBytes)return false;
        const auto primary=arbitgpu::metalSdfPrimaryAllocationForTests(*rendered.frame);
        if(primary[0]==0||primary[1]<49u*49u*8
            ||receipt.attachmentBytes!=49u*49u*20+primary[1])return false;
        const auto colorDescriptor=rendered.frame->colorTextureDescriptor();
        if(!colorDescriptor.complete()||!rendered.frame->depthTextureDescriptor().complete()
            ||!rendered.frame->normalTextureDescriptor().complete()
            ||colorDescriptor.rendererGeneration!=rendered.frame->depthTextureDescriptor().rendererGeneration
            ||colorDescriptor.rendererGeneration!=rendered.frame->normalTextureDescriptor().rendererGeneration)return false;
        for(std::size_t y=0;y<49;++y)for(std::size_t x=0;x<49;++x)
        {
            std::array<double,3> direction {(2*(double(x)+.5)-49)/49,(49-2*(double(y)+.5))/49,-1.8};
            const double norm=std::sqrt(direction[0]*direction[0]+direction[1]*direction[1]+direction[2]*direction[2]);
            for(auto& lane:direction)lane/=norm;
            const auto expected=ringRay({0,0,3},direction,angle,.5);
            const std::size_t i=y*49+x;
            if(expected.axis<0)
            {
                if(depth[i]!=1||rawNormals[i*4+3]!=0)return false;
                if(x==24&&y==24)++holes;
                continue;
            }
            if(std::abs(depth[i]-expected.distance/10)>.00005||rawNormals[i*4+3]!=1)return false;
            // Skip crease pixels where the symmetric normal stencil blends faces.
            std::array<double,3> hit {direction[0]*expected.distance,direction[1]*expected.distance,3+direction[2]*expected.distance};
            const double c=std::cos(angle),s=std::sin(angle);
            hit={c*hit[0]-s*hit[2],hit[1],s*hit[0]+c*hit[2]};
            unsigned nearFaces=0;
            for(int k=0;k<2;++k)
                nearFaces+=std::abs(std::abs(hit[k])-1)<.002||std::abs(std::abs(hit[k])-.5)<.002;
            nearFaces+=std::abs(std::abs(hit[2])-.25)<.002;
            if(nearFaces>1)continue;
            for(int k=0;k<3;++k)
                if(std::abs(color[i*4+k]-(expected.normal[k]*.5+.5))>.004
                    ||std::abs(rawNormals[i*4+k]-expected.normal[k])>.008)return false;
            expected.axis==2?++caps:++sides;
        }
        const auto repeated=arbitgpu::nativeSdfExecutionBackend().render(request);
        std::vector<float> repeatedColor,repeatedDepth;
        if(!repeated.rendered||!repeated.frame||!repeated.frame->readColorFloatPixels(repeatedColor)
            ||!repeated.frame->readDepthFloatPixels(repeatedDepth)||color!=repeatedColor||depth!=repeatedDepth
            ||repeated.frame->colorImageHandle()==rendered.frame->colorImageHandle())return false;
    }
    if(caps==0||sides==0||holes!=3)return false;
    auto base=textRequest();
    std::string error;
    base.compiledProgram=videohelper::sdf::compileNativeSdfProgram(base.geometry,error,true);
    if(!base.compiledProgram)return false;
    const auto baseFrame=arbitgpu::nativeSdfExecutionBackend().render(base);
    std::vector<float> basePixels;
    if(!baseFrame.frame||!baseFrame.frame->readColorFloatPixels(basePixels))return false;
    for(auto outline : {textRectangle(true,1,"A"),textRectangle(true,1,"O","InterBold_ttf"),textRectangle(true,1.125)})
    {
        auto changed=textRequest(outline); changed.compiledProgram=base.compiledProgram;
        const auto refused=arbitgpu::nativeSdfExecutionBackend().render(changed);
        if(refused.rendered||refused.frame||refused.error.empty())return false;
        changed.compiledProgram=videohelper::sdf::compileNativeSdfProgram(changed.geometry,error,true);
        if(!changed.compiledProgram||changed.compiledProgram->structuralDigest()==base.compiledProgram->structuralDigest())return false;
        const auto accepted=arbitgpu::nativeSdfExecutionBackend().render(changed);
        std::vector<float> changedPixels;
        if(!accepted.rendered||!accepted.frame||!accepted.frame->readColorFloatPixels(changedPixels))return false;
        const bool sameVectors=outline.segments[0].x0==-1;
        if(sameVectors!=(changedPixels==basePixels))return false;
    }
    auto paired=base;
    paired.compiledProgram.reset();
    const auto solid=textRectangle(false,.75,"I");
    paired.geometry.textOutlines.push_back(solid);
    std::sort(paired.geometry.textOutlines.begin(),paired.geometry.textOutlines.end(),
        [](const auto& a,const auto& b){return a.stableId<b.stableId;});
    auto second=record(42,videowire::SdfOperation::textExtrusion,{}, {.5});
    second.resourceStableId=solid.stableId;
    paired.geometry.records.push_back(second);
    paired.geometry.records.push_back(record(43,videowire::SdfOperation::translate,{42},{1,0,0}));
    paired.geometry.records.push_back(record(44,videowire::SdfOperation::unionOp,{41,43},{}));
    paired.geometry.rootId=44;
    const auto pairFrame=arbitgpu::nativeSdfExecutionBackend().render(paired);
    std::vector<float> pairPixels;
    if(!pairFrame.frame||!pairFrame.frame->readColorFloatPixels(pairPixels)||pairPixels==basePixels
        ||pairFrame.frame->sdfResourceReceipt().segmentVisits
            !=pairFrame.frame->sdfResourceReceipt().distanceQueryCount*12)return false;
    auto deeper=textRequest(textRectangle(),0,1);
    const auto shallow=arbitgpu::nativeSdfExecutionBackend().render(base);
    const auto deep=arbitgpu::nativeSdfExecutionBackend().render(deeper);
    std::vector<float> firstDepth,secondDepth;
    if(!shallow.frame||!deep.frame||!shallow.frame->readDepthFloatPixels(firstDepth)
        ||!deep.frame->readDepthFloatPixels(secondDepth)||firstDepth==secondDepth)return false;
    auto missing=base; missing.geometry.records[0].resourceStableId^=1;
    auto malformed=base; malformed.geometry.textOutlines[0].segments[0].x0+=.125;
    auto legacy=base; legacy.geometry.schemaVersion=videowire::kSdfIrSchemaVersion;
    auto extent=base; extent.width=4097;
    auto segments=base; segments.geometry.textOutlines[0].segments.resize(videowire::kSdfTextMaximumSegments+1);
    auto cache=base; cache.geometryCacheBytes=std::numeric_limits<std::uint64_t>::max();
    for(const auto& invalid : {missing,malformed,legacy,extent,segments,cache})
    {
        const auto refused=arbitgpu::nativeSdfExecutionBackend().render(invalid);
        if(refused.rendered||refused.frame||refused.error.empty())return false;
    }
    for(const auto extent : {std::array<std::uint32_t,2>{1920,1080},{3840,2160}})
    {
        auto full=base; full.width=extent[0]; full.height=extent[1];
        full.maximumSteps=128; full.output=arbitgpu::NativeSdfOutput::color;
        const auto started=std::chrono::steady_clock::now();
        const auto refused=arbitgpu::nativeSdfExecutionBackend().render(full);
        const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        std::printf("Metal SDF text %ux%u admission refusal seconds=%.6f, no rendered timing receipt\n",
            full.width,full.height,seconds);
        if(refused.rendered||refused.frame||refused.error!="SDF text evaluation work capacity exceeded")return false;
    }
    for(unsigned output=0;output<static_cast<unsigned>(arbitgpu::NativeSdfOutput::count);++output)
    {
        auto diagnostic=base; diagnostic.width=diagnostic.height=17;
        diagnostic.maximumSteps=128; diagnostic.output=static_cast<arbitgpu::NativeSdfOutput>(output);
        const auto frame=arbitgpu::nativeSdfExecutionBackend().render(diagnostic);
        std::vector<float> values;
        if(!frame.rendered||!frame.frame||!frame.frame->readColorFloatPixels(values)
            ||!std::all_of(values.begin(),values.end(),[](float v){return std::isfinite(v);}))return false;
    }
    // Preview/export owners must account for cache storage once and retire
    // stale identity without changing same-time native pixels.
    const auto admitted=videohelper::sdf::admitSdfIr(base.geometry,{},error);
    if(!admitted)return false;
    const auto immutable=std::make_shared<const videohelper::sdf::AdmittedSdfIr>(*admitted);
    videohelper::sdf::NativeSdfRenderedFrame preview,exported;
    videohelper::sdf::NativeSdfRenderControls controls;
    controls.output=arbitgpu::NativeSdfOutput::normal;
    if(!videohelper::sdf::nativeSdfRenderer().renderPreview(immutable,{17,17},controls,
            videohelper::sdf::kNativeGpuCapability,preview,error)
        ||!videohelper::sdf::nativeSdfRenderer().renderExport(immutable,{17,17},controls,
            videohelper::sdf::kNativeGpuCapability,exported,error))return false;
    std::vector<float> previewPixels,exportPixels;
    if(!preview.nativeFrame->readColorFloatPixels(previewPixels)||!exported.nativeFrame->readColorFloatPixels(exportPixels)
        ||previewPixels!=exportPixels)return false;
    const auto& receipt=preview.nativeFrame->sdfResourceReceipt();
    return receipt.geometryCacheBytes>=base.compiledProgram->storageBytes()
        &&receipt.totalBytes==receipt.geometryCacheBytes+receipt.backendProgramBytes+receipt.uniformBytes
            +receipt.attachmentBytes+receipt.nativeTextTableBytes;
}
} // namespace

int main()
{
    const auto info = arbitgpu::queryNativeBackend();
    if (! info.available || info.backend != "metal" || ! info.compute)
    {
        std::fprintf (stderr, "Metal capability probe failed: %s\n", info.error.c_str());
        return 1;
    }

    const auto test = arbitgpu::runNativeBackendSelfTest();
    if (! test.available || ! test.computePassed || ! test.renderPassed)
    {
        std::fprintf (stderr, "Metal self-test failed: %s\n", test.error.c_str());
        return 2;
    }

    renderpassoutput::Description outputDescription;
    outputDescription.extent = { 64, 32 };
    for (const auto output : { renderpassoutput::Output::Color,
                               renderpassoutput::Output::Depth })
    {
        const auto required = renderpassoutput::requirements (output);
        outputDescription.attachments.push_back (
            { output, required.format, required.colorSpace, outputDescription.extent });
    }
    renderpassoutput::AdmissionFailure outputFailure = renderpassoutput::AdmissionFailure::None;
    auto admittedOutputs = renderpassoutput::admit (outputDescription, outputFailure);
    auto& outputBackend = arbitgpu::nativeRenderPassOutputBackend();
    const auto outputCapabilities = outputBackend.renderPassOutputCapabilities();
    auto outputAdmission = admittedOutputs.has_value()
        ? outputBackend.admitRenderPassOutputs (*admittedOutputs)
        : arbitgpu::RenderPassOutputAdmission {};
    if (! outputCapabilities.available || ! outputAdmission.lifecycle
        || outputAdmission.resources.size() != 2
        || outputAdmission.frameMemory.failure
               != arbitgpu::FrameMemoryAdmissionFailure::none
        || outputAdmission.frameMemory.requestedBytes != 64ull * 32ull * 12ull
        || outputAdmission.frameMemory.allocatedSlots != 2)
    {
        std::fprintf (stderr, "Metal frame-memory admission failed: %s\n",
                      outputAdmission.error.c_str());
        return 20;
    }
    outputBackend.releaseRenderPassOutputs (outputAdmission.lifecycle);

    const auto sdf = arbitgpu::queryNativeSdfExecution();
    if (! sdf.available || sdf.backend != "metal"
        || ! sdf.supports (videowire::SdfOperation::sphere)
        || ! sdf.supports (videowire::SdfOperation::box)
        || ! sdf.supports (videowire::SdfOperation::roundedBox)
        || ! sdf.supports (videowire::SdfOperation::plane)
        || ! sdf.supports (videowire::SdfOperation::torus)
        || ! sdf.supports (videowire::SdfOperation::capsule)
        || ! sdf.supports (videowire::SdfOperation::cylinder)
        || ! sdf.supports (videowire::SdfOperation::cone)
        || ! sdf.supports (videowire::SdfOperation::gyroid)
        || ! sdf.supports (videowire::SdfOperation::textExtrusion)
        || ! sdf.supports (arbitgpu::NativeSdfOutput::color)
        || ! sdf.supports (arbitgpu::NativeSdfOutput::depth)
        || ! sdf.supports (arbitgpu::NativeSdfOutput::normal))
    {
        std::fprintf (stderr, "Metal SDF capability probe failed: %s\n",
                      sdf.error.c_str());
        return 3;
    }

    arbitgpu::NativeSdfDrawRequest request;
    request.geometry.rootId = 1;
    videowire::SdfRecord sphere;
    sphere.stableId = request.geometry.rootId;
    sphere.operation = videowire::SdfOperation::sphere;
    sphere.parameterCount = 1;
    sphere.parameters[0] = 1.0;
    request.geometry.records.push_back (sphere);
    request.width = 64;
    request.height = 64;
    request.maximumSteps = 128;
    request.epsilon = 0.001;
    request.maximumDistance = 100.0;
    std::string rendererError;
    auto admitted = videohelper::sdf::admitSdfIr (request.geometry, {}, rendererError);
    videohelper::sdf::NativeSdfRenderedFrame renderedByOwner;
    videohelper::sdf::NativeSdfRenderedFrame exportedByOwner;
    const auto geometry = admitted
        ? std::make_shared<const videohelper::sdf::AdmittedSdfIr> (std::move (*admitted))
        : nullptr;
    const auto ownerRendered = geometry && videohelper::sdf::nativeSdfRenderer().renderPreview (
        geometry, { request.width, request.height }, {},
        videohelper::sdf::kNativeGpuCapability, renderedByOwner, rendererError);
    const auto ownerExported = geometry && videohelper::sdf::nativeSdfRenderer().renderExport (
        geometry, { request.width, request.height }, {},
        videohelper::sdf::kNativeGpuCapability, exportedByOwner, rendererError);
    const auto rendered = ownerRendered
        ? arbitgpu::NativeSdfSceneSubmission { true, renderedByOwner.nativeFrame, {} }
        : arbitgpu::NativeSdfSceneSubmission { false, {}, rendererError };
    if (! rendered.rendered || rendered.frame == nullptr
        || rendered.frame->backend() != "metal"
        || rendered.frame->width() != request.width
        || rendered.frame->height() != request.height
        || rendered.frame->colorImageHandle() == 0
        || rendered.frame->colorTextureViewHandle() == 0
        || rendered.frame->frameMemoryAdmission().failure
               != arbitgpu::FrameMemoryAdmissionFailure::none
        || rendered.frame->frameMemoryAdmission().requestedBytes != 64ull * 64ull * 20ull
        || rendered.frame->frameMemoryAdmission().allocatedSlots != 3
        || renderedByOwner.use != videohelper::sdf::NativeSdfRenderUse::Preview
        || ! ownerExported || exportedByOwner.use != videohelper::sdf::NativeSdfRenderUse::Export
        || exportedByOwner.nativeFrame == nullptr
        || exportedByOwner.nativeFrame->backend() != "metal"
        || exportedByOwner.nativeFrame->colorImageHandle() == 0
        || exportedByOwner.nativeFrame->colorImageHandle()
               == rendered.frame->colorImageHandle())
    {
        std::fprintf (stderr, "Metal SDF draw failed: %s\n", rendered.error.c_str());
        return 4;
    }
    const auto depthView = rendered.frame->depthTextureDescriptor();
    std::vector<float> rawDepth;
    if (! depthView.complete() || depthView.format != arbitgpu::NativeTexturePixelFormat::R32Float
        || depthView.textureViewHandle != rendered.frame->depthTextureViewHandle()
        || depthView.rowOrder != arbitgpu::NativeTextureRowOrder::TopFirst
        || ! rendered.frame->readDepthFloatPixels(rawDepth)
        || rawDepth.size() != 64u * 64u
        || std::fabs(rawDepth[32u * 64u + 32u] - 0.02f) > 0.002f
        || rawDepth[0] != 1.0f)
    {
        std::fprintf (stderr, "Metal SDF raw Depth attachment does not contain analytic hit/miss distance\n");
        return 4;
    }
    const auto& receipt = rendered.frame->sdfResourceReceipt();
    const auto normalView = rendered.frame->normalTextureDescriptor();
    if (! normalView.complete()
        || normalView.format != arbitgpu::NativeTexturePixelFormat::Rgba16Float
        || normalView.textureViewHandle != rendered.frame->normalTextureViewHandle()
        || normalView.rowOrder != arbitgpu::NativeTextureRowOrder::TopFirst
        || normalView.rendererGeneration != depthView.rendererGeneration)
    {
        std::fprintf (stderr, "Metal SDF raw Normal attachment is missing\n");
        return 4;
    }
    if (receipt.compiledRecordCount != 1 || receipt.compiledRecordBytes == 0
        || receipt.geometryCacheBytes < receipt.compiledRecordBytes
        || receipt.backendProgramBytes == 0 || receipt.uniformBytes == 0
        || receipt.attachmentBytes != 64ull * 64ull * 20ull || receipt.totalBytes == 0)
    {
        std::fprintf (stderr, "Metal native SDF resource receipt mismatch\n");
        return 4;
    }

    auto boxRequest = request;
    boxRequest.geometry = {};
    boxRequest.geometry.rootId = 2;
    videowire::SdfRecord box;
    box.stableId = boxRequest.geometry.rootId;
    box.operation = videowire::SdfOperation::box;
    box.parameterCount = 3;
    box.parameters[0] = 0.75;
    box.parameters[1] = 0.5;
    box.parameters[2] = 0.25;
    boxRequest.geometry.records.push_back (box);
    const auto renderedBox = arbitgpu::nativeSdfExecutionBackend().render (boxRequest);
    if (! renderedBox.rendered || renderedBox.frame == nullptr
        || renderedBox.frame->backend() != "metal"
        || renderedBox.frame->colorImageHandle() == 0
        || renderedBox.frame->colorTextureViewHandle() == 0)
    {
        std::fprintf (stderr, "Metal SDF box draw failed: %s\n", renderedBox.error.c_str());
        return 5;
    }

    auto compositeRequest = request;
    compositeRequest.geometry = {};
    compositeRequest.geometry.rootId = 5;
    compositeRequest.geometry.records = {
        record (1, videowire::SdfOperation::sphere, {}, { 0.55 }),
        record (2, videowire::SdfOperation::translate, { 1 }, { -0.7, 0.0, 0.0 }),
        record (3, videowire::SdfOperation::box, {}, { 0.38, 0.5, 0.45 }),
        record (4, videowire::SdfOperation::translate, { 3 }, { 0.7, 0.0, 0.0 }),
        record (5, videowire::SdfOperation::smoothUnion, { 2, 4 }, { 0.18 })
    };
    const auto renderedComposite = arbitgpu::nativeSdfExecutionBackend().render (compositeRequest);
    std::vector<std::uint8_t> spherePixels;
    std::vector<std::uint8_t> compositePixels;
    if (! renderedComposite.rendered || renderedComposite.frame == nullptr
        || ! rendered.frame->readColorPixels (spherePixels)
        || ! renderedComposite.frame->readColorPixels (compositePixels)
        || spherePixels.size() != 64u * 64u * 4u
        || compositePixels.size() != spherePixels.size()
        || checksum (spherePixels) == checksum (compositePixels))
    {
        std::fprintf (stderr, "Metal transformed composite did not affect pixels: %s\n",
                      renderedComposite.error.c_str());
        return 6;
    }

    auto malformed = compositeRequest;
    malformed.geometry.rootId = 2;
    malformed.geometry.records = {
        record (1, videowire::SdfOperation::translate, { 2 }, { 0.0, 0.0, 0.0 }),
        record (2, videowire::SdfOperation::translate, { 1 }, { 0.0, 0.0, 0.0 })
    };
    const auto rejectedCycle = arbitgpu::nativeSdfExecutionBackend().render (malformed);
    malformed.geometry.records[0].inputs[0] = 999;
    const auto rejectedReference = arbitgpu::nativeSdfExecutionBackend().render (malformed);
    if (rejectedCycle.rendered || rejectedCycle.frame != nullptr
        || rejectedReference.rendered || rejectedReference.frame != nullptr)
    {
        std::fprintf (stderr, "Metal malformed SDF request allocated a frame\n");
        return 7;
    }

    videohelper::sdf::NativeSdfRenderControls parityControls;
    parityControls.maximumSteps = 128;
    parityControls.epsilon = 0.001;
    parityControls.maximumDistance = 100.0;
    parityControls.output = arbitgpu::NativeSdfOutput::depth;
    std::string parityError;
    auto parityFixtures = videohelper::sdf::test::nativeOperationFixtures (parityError);
    auto polarFixtures = videohelper::sdf::test::largeOffsetPolarRepeatFixtures (parityError);
    parityFixtures.insert (parityFixtures.end(),
                           std::make_move_iterator (polarFixtures.begin()),
                           std::make_move_iterator (polarFixtures.end()));
    if (parityFixtures.size() != 28)
    {
        std::fprintf (stderr, "Metal SDF parity fixtures failed: %s\n", parityError.c_str());
        return 8;
    }
    for (const auto& fixture : parityFixtures)
    {
        arbitgpu::NativeSdfDrawRequest parityRequest;
        parityRequest.geometry.schemaVersion = fixture.geometry->schemaVersion();
        parityRequest.geometry.rootId = fixture.geometry->rootId();
        parityRequest.geometry.records = fixture.geometry->records();
        parityRequest.width = 17;
        parityRequest.height = 17;
        parityRequest.maximumSteps = parityControls.maximumSteps;
        parityRequest.epsilon = parityControls.epsilon;
        parityRequest.maximumDistance = parityControls.maximumDistance;
        parityRequest.output = parityControls.output;
        const auto parityFrame = arbitgpu::nativeSdfExecutionBackend().render (parityRequest);
        std::vector<std::uint8_t> parityPixels;
        if (! parityFrame.rendered || parityFrame.frame == nullptr
            || ! parityFrame.frame->readColorPixels (parityPixels)
            || ! videohelper::sdf::test::verifyNativeDepthParity (
                   fixture, parityControls, parityPixels, 17, 17, parityError))
        {
            std::fprintf (stderr, "Metal SDF operation parity failed: %s\n",
                          parityError.empty() ? parityFrame.error.c_str() : parityError.c_str());
            return 9;
        }
    }

    if (! videohelper::sdf::test::verifyNativeUtilityOutputs (
            [] (const auto& frame)
            {
                std::vector<float> pixels;
                if (!frame.readColorFloatPixels (pixels)) pixels.clear();
                return pixels;
            }, parityError))
    {
        std::fprintf (stderr, "Metal SDF utility semantic pixels failed: %s\n", parityError.c_str());
        return 10;
    }

    if (! verifyMetalTextFrames())
    {
        std::fprintf (stderr, "Metal native SDF text field/frame/identity/resource fixture failed\n");
        return 11;
    }

    auto fixtureScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
        videohelper::fixture3d::makeScene());
    auto& fixtureBackend = arbitgpu::nativeFixtureSceneBackend();
    auto fixturePreparation = fixtureBackend.prepare (fixtureScene, nullptr);
    arbitgpu::NativeFixtureSceneRuntimeInputs fixtureInputs;
    HarmonicMIDI::grid::SceneLightRecord lightOverride;
    lightOverride.id.value = 9001;
    lightOverride.color = { 0.25f, 0.5f, 0.75f };
    lightOverride.intensity = 8.0f;
    fixtureInputs.lightOverride = lightOverride;
    auto fixtureFrame = fixtureBackend.render (
        fixtureScene, fixturePreparation.resources, 64, 64, fixtureInputs);
    if (! fixturePreparation.prepared || ! fixtureFrame.rendered
        || fixtureFrame.frame == nullptr || fixtureFrame.frame->backend() != "metal"
        || fixtureFrame.frame->colorImageHandle() == 0
        || fixtureFrame.frame->colorTextureViewHandle() == 0)
    {
        std::fprintf (stderr, "Metal fixture Light binding failed: %s %s\n",
                      fixturePreparation.error.c_str(), fixtureFrame.error.c_str());
        return 6;
    }

    const auto fixturePixels = readBgra8 (fixtureFrame.frame);
    fixtureInputs.lightOverride->intensity = 0;
    const auto unlitFrame = fixtureBackend.render(
        fixtureScene, fixturePreparation.resources, 64, 64, fixtureInputs);
    fixtureInputs.lightOverride->intensity = 8;
    const auto replayLightFrame = fixtureBackend.render(
        fixtureScene, fixturePreparation.resources, 64, 64, fixtureInputs);
    if (!unlitFrame.frame || !replayLightFrame.frame
        || readBgra8(unlitFrame.frame) == fixturePixels
        || readBgra8(replayLightFrame.frame) != fixturePixels)
    {
        std::fprintf(stderr, "Metal per-frame light override pixels or replay failed\n");
        return 6;
    }
    auto composedSceneValue = *fixtureScene;
    composedSceneValue.objectCount = 2;
    composedSceneValue.materialCount = 2;
    composedSceneValue.objects[1] = composedSceneValue.objects[0];
    composedSceneValue.objects[1].id.value = 9002;
    composedSceneValue.objects[1].parent = composedSceneValue.objects[0].id;
    // The local offset is the parent's inverse rotation of world +X, so the
    // child lands in a separate right-side draw region while exercising hierarchy.
    composedSceneValue.objects[1].transform.translation
        = { 2.5980762f, -0.5130302f, 1.4095389f };
    composedSceneValue.materials[1] = composedSceneValue.materials[0];
    composedSceneValue.materials[1].id.value = 9003;
    composedSceneValue.materials[1].emissive = { 0.01f, 0.8f, 0.02f };
    composedSceneValue.objects[1].material = composedSceneValue.materials[1].id;
    auto composedScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
        std::move (composedSceneValue));
    auto objectZeroSceneValue = *composedScene;
    objectZeroSceneValue.objectCount = 1;
    auto objectZeroScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (
        std::move (objectZeroSceneValue));
    const auto objectZeroPreparation = fixtureBackend.prepare (objectZeroScene, nullptr);
    const auto objectZeroFrame = fixtureBackend.render (
        objectZeroScene, objectZeroPreparation.resources, 64, 64, fixtureInputs);
    const auto objectZeroPixels = objectZeroFrame.rendered
        ? readBgra8 (objectZeroFrame.frame) : std::vector<std::uint8_t> {};
    const auto composedPreparation = fixtureBackend.prepare (composedScene, nullptr);
    const auto composedFrame = fixtureBackend.render (
        composedScene, composedPreparation.resources, 64, 64, fixtureInputs);
    const auto composedPixels = composedFrame.rendered
        ? readBgra8 (composedFrame.frame) : std::vector<std::uint8_t> {};
    const auto composedExport = fixtureBackend.render (
        composedScene, composedPreparation.resources, 64, 64, fixtureInputs);
    const auto composedExportPixels = composedExport.rendered
        ? readBgra8 (composedExport.frame) : std::vector<std::uint8_t> {};
    const std::array<std::uint8_t, 4> background { 18, 10, 7, 255 };
    std::size_t secondObjectPixels = 0;
    bool objectZeroUnchanged = composedPixels.size() == objectZeroPixels.size();
    for (std::uint32_t y = 0; objectZeroUnchanged && y < 64u; ++y)
        for (std::uint32_t x = 0; x < 64u; ++x)
        {
            const auto offset = (y * 64u + x) * 4u;
            if (! std::equal (composedPixels.begin() + offset,
                              composedPixels.begin() + offset + 4u,
                              objectZeroPixels.begin() + offset))
            {
                objectZeroUnchanged &= std::equal (
                    objectZeroPixels.begin() + offset, objectZeroPixels.begin() + offset + 4u,
                    background.begin());
                secondObjectPixels += composedPixels[offset + 1u]
                    > composedPixels[offset + 2u] * 4u ? 1u : 0u;
            }
        }
    if (! objectZeroPreparation.prepared || ! objectZeroFrame.rendered
        || objectZeroFrame.stats.drawCount != 1
        || ! composedPreparation.prepared || ! composedFrame.rendered
        || ! composedExport.rendered
        || composedPreparation.stats.staticUploadCount != 1
        || composedPreparation.stats.vertexBytes != objectZeroPreparation.stats.vertexBytes
        || composedPreparation.stats.indexBytes != objectZeroPreparation.stats.indexBytes
        || composedFrame.stats.drawCount != 2
        || ! composedFrame.stats.reusedStaticResources
        || ! composedExport.stats.reusedStaticResources
        || composedPixels.size() != 64u * 64u * 4u
        || ! objectZeroUnchanged || secondObjectPixels == 0
        || composedPixels != composedExportPixels)
    {
        std::fprintf (stderr, "Metal composed Scene3D execution failed: %s %s\n",
                      composedPreparation.error.c_str(), composedFrame.error.c_str());
        return 7;
    }

    auto invalidOpacitySceneValue = *fixtureScene;
    invalidOpacitySceneValue.materials[0].opacity = 1.5f;
    const auto invalidOpacityPreparation = fixtureBackend.prepare (
        std::make_shared<const HarmonicMIDI::grid::Visual3DScene> (invalidOpacitySceneValue),
        nullptr);
    if (invalidOpacityPreparation.prepared || invalidOpacityPreparation.resources != nullptr
        || invalidOpacityPreparation.stats.staticUploadCount != 0)
    {
        std::fprintf (stderr, "Metal accepted out-of-range opacity before GPU upload\n");
        return 8;
    }

    std::printf ("backend=%s device=%s compute=%u render=%u sdfImage=%zu sdfBoxImage=%zu fixtureImage=%zu\n",
                 test.backend.c_str(), test.device.c_str(),
                 test.computeChecksum, test.renderChecksum,
                 rendered.frame->colorImageHandle(), renderedBox.frame->colorImageHandle(),
                 fixtureFrame.frame->colorImageHandle());
    return 0;
}
