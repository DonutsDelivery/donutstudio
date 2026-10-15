#include "support/restricted_surface_source_cases.h"
#include "support/camera_pixel_sign_cases.h"
#include "support/camera_wide_line_predicate_cases.h"
#include "support/camera_line_predicate_cases.h"
#include "support/camera_ray_coefficient_cases.h"
#include "support/exact_line_predicate_cases.h"
#include "support/restricted_lambertian_radiance_cases.h"
#include "../src/gpu_backend/metal_lambertian_dispatch.h"
#include "../src/gpu_backend/metal_lambertian_source.h"
#include "../src/gpu_backend/backend.h"
#include "../src/gpu_backend/metal_rough_glass_receipt.h"
#include "../src/gpu_backend/metal_rough_glass_tiles.h"
#include "../src/gpu_backend/metal_glass_caustic_consumer.h"
#include "support/glass_caustic_oracle.h"
#include "../src/diffraction_material_admission.h"
#include "../src/diffraction_material_execution.h"
#include "../src/canonical_block_c_frame.h"
#include "../src/fixture_scene_renderer.h"
#include "support/authored_rough_caustic_renderer_cases.h"
#include "support/fixture_scene.h"
#include "support/base_surface_reflection_cases.h"
#include "support/analytic_sky_cases.h"
#include "support/environment_image_cases.h"
#include "support/environment_secondary_cases.h"
#include "support/smooth_glass_cases.h"
#include "support/thin_interface_water_cases.h"
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

#include "sokol_gfx.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#if defined(ARBIT_HOLOGRAPHIC_TRADING_CARD_STRICT) \
    && ARBIT_HOLOGRAPHIC_TRADING_CARD_STRICT
#include <cstddef>
#endif
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace arbitgpu
{
bool inspectMetalGlassGratingCameraAllocationForFixture(
    const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>&,std::string&);
bool produceMetalDirectGratingFailedQueryForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const glasscaustics::GratingIncidentBinding>&,
    const std::vector<glasscaustics::NativeDirectGratingEndpointInput>&,
    glasscaustics::NativeDirectGratingContributionReadback&,std::string&);
bool inspectMetalDirectGratingRefusalsForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,std::string&);
bool inspectMetalDirectGratingAttemptBudgetForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::vector<glasscaustics::NativeDirectGratingEndpointInput>&,std::uint64_t,std::string&);
bool validateMetalDirectGratingSharedEntryForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const glasscaustics::NativeDirectGratingContributionReadback&,std::string&);
bool validateMetalGlassCameraGratingSharedEntryForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const glasscaustics::NativeGlassCameraGratingTraceReadback&,std::string&);
bool inspectMetalGlassCameraTraceRefusalsForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,std::string&);
bool readMetalDirectGratingCohortForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    std::vector<glasscaustics::NativeDirectGratingContributionReadback>&,std::string&);
bool inspectMetalDirectGratingCohortRefusalsForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    glasscaustics::DirectGratingIrradianceCohortReadback&,std::string&);
bool inspectMetalDirectGratingCohortIncumbentRefusalForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    const std::shared_ptr<const glasscaustics::DirectGratingIrradianceCohort>&,
    glasscaustics::DirectGratingIrradianceCohortReadback&,std::string&);
bool inspectMetalDirectTerminalLeaseRefusalsForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    glasscaustics::DirectGratingTerminalOrderLedger&,std::string&);
bool inspectMetalDirectTerminalReceiverRefusalsForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    glasscaustics::DirectGratingTerminalReceiverReadback&,std::string&);
bool inspectMetalDirectTerminalLaterBudgetRefusalsForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,
    glasscaustics::DirectGratingIrradianceCohortReadback&,std::string&);
bool inspectMetalRasterCaptureForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,bool,std::string&);
}

namespace arbitgpu {
// Read-only production admission guard; no native dispatch or synthetic receipt.
bool metalSelectedRoughBudgetGuardForFixture(const std::shared_ptr<const NativeFixtureSceneFrame>&,std::string&);
}

namespace
{
bool expect(bool condition, const char* message)
{
    if (!condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}

// Test-local native arithmetic only; no camera/coverage authority or receipt.
bool exactLineNativeProbe()
{
    namespace ec=videohelper::tests::exactlinecases;namespace el=arbitgpu::exactline;
    id<MTLCommandQueue> queue=(__bridge id<MTLCommandQueue>)sg_mtl_command_queue();
    if(!queue)return expect(false,"exact-line probe native Metal queue unavailable");
    const auto tests=ec::cases();if(tests.empty()||tests.size()>1024||tests.capacity()>1024)return expect(false,"exact-line probe case admission");
    std::uint64_t work=0,bytes=tests.capacity()*sizeof(ec::Case);
    for(const auto& test:tests){work+=el::maximumScalarOperations(static_cast<el::Operation>(test.input[0]));
        bytes+=el::kLogicalScratchBytes+2*sizeof(el::Request)+2*sizeof(el::Result);}
    if(work>1024*el::maximumScalarOperations(el::Operation::CompareX)
        ||bytes>1024*(el::kLogicalScratchBytes+2*sizeof(el::Request)+2*sizeof(el::Result)+sizeof(ec::Case)))
        return expect(false,"exact-line probe derived envelope exceeded");
    std::vector<el::Request> inputs;inputs.reserve(tests.size());for(const auto& test:tests)inputs.push_back(test.input);
    const auto source=el::metalProbeSource();NSError* error=nil;
    id<MTLLibrary> library=[queue.device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
        options:nil error:&error];
    id<MTLFunction> function=library?[library newFunctionWithName:@"exactLineProbe"]:nil;
    id<MTLComputePipelineState> pipeline=function?[queue.device newComputePipelineStateWithFunction:function error:&error]:nil;
    id<MTLBuffer> inBuffer=nil,outBuffer=nil;
    bool ok=false;
    if(pipeline){
        inBuffer=[queue.device newBufferWithBytes:inputs.data() length:inputs.size()*sizeof(el::Request)
            options:MTLResourceStorageModeShared];
        outBuffer=[queue.device newBufferWithLength:tests.size()*sizeof(el::Result) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=command?[command computeCommandEncoder]:nil;
        if(inBuffer&&outBuffer&&command&&encoder){
            [encoder setComputePipelineState:pipeline];[encoder setBuffer:inBuffer offset:0 atIndex:0];
            [encoder setBuffer:outBuffer offset:0 atIndex:1];
            [encoder dispatchThreadgroups:MTLSizeMake(tests.size(),1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [encoder endEncoding];[command commit];[command waitUntilCompleted];
            if(command.status==MTLCommandBufferStatusCompleted&&outBuffer.contents){
                const auto* result=static_cast<const std::uint8_t*>(outBuffer.contents);ok=true;std::string mismatch;
                for(std::size_t k=0;k<tests.size();k++){
                    el::Result actual{};std::memcpy(actual.data(),result+k*sizeof(el::Result),sizeof(actual));
                    if(!ec::compare(tests[k],actual,mismatch)){std::cerr<<"FAIL: Metal exact-line arithmetic: "<<mismatch<<'\n';ok=false;}}
            }
            else error=command.error;
        }
        else if(encoder)[encoder endEncoding];
    }
    if(!ok){std::cerr<<"FAIL: Metal exact-line arithmetic probe";
        if(error)std::cerr<<": "<<error.localizedDescription.UTF8String;std::cerr<<'\n';}
#if !__has_feature(objc_arc)
    [outBuffer release];[inBuffer release];[pipeline release];[function release];[library release];
#endif
    return ok;
}

bool cameraRayCoefficientNativeProbe()
{
    namespace ec=videohelper::tests::cameraraycases;namespace el=arbitgpu::cameraraycoefficients;
    id<MTLCommandQueue> queue=(__bridge id<MTLCommandQueue>)sg_mtl_command_queue();
    if(!queue)return expect(false,"camera-ray-coefficient probe native Metal queue unavailable");
    const auto tests=ec::cases();if(tests.empty()||tests.size()>1024||tests.capacity()>1024)return expect(false,"camera-ray-coefficient probe case admission");
    std::uint64_t work=0,bytes=tests.capacity()*sizeof(ec::Case);
    for(const auto& test:tests){work+=el::maximumScalarOperations(test.input[1]);
        bytes+=el::kLogicalScratchBytes+2*sizeof(el::Request)+2*sizeof(el::Result);}
    if(work>1024*el::maximumScalarOperations(256u)
        ||bytes>1024*(el::kLogicalScratchBytes+2*sizeof(el::Request)+2*sizeof(el::Result)+sizeof(ec::Case)))
        return expect(false,"camera-ray-coefficient probe derived envelope exceeded");
    std::vector<el::Request> inputs;inputs.reserve(tests.size());for(const auto& test:tests)inputs.push_back(test.input);
    const auto source=el::metalProbeSource();NSError* error=nil;
    id<MTLLibrary> library=[queue.device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
        options:nil error:&error];
    id<MTLFunction> function=library?[library newFunctionWithName:@"cameraRayCoefficientProbe"]:nil;
    id<MTLComputePipelineState> pipeline=function?[queue.device newComputePipelineStateWithFunction:function error:&error]:nil;
    id<MTLBuffer> inBuffer=nil,outBuffer=nil;
    bool ok=false;
    if(pipeline){
        inBuffer=[queue.device newBufferWithBytes:inputs.data() length:inputs.size()*sizeof(el::Request)
            options:MTLResourceStorageModeShared];
        outBuffer=[queue.device newBufferWithLength:tests.size()*sizeof(el::Result) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=command?[command computeCommandEncoder]:nil;
        if(inBuffer&&outBuffer&&command&&encoder){
            [encoder setComputePipelineState:pipeline];[encoder setBuffer:inBuffer offset:0 atIndex:0];
            [encoder setBuffer:outBuffer offset:0 atIndex:1];
            [encoder dispatchThreadgroups:MTLSizeMake(tests.size(),1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [encoder endEncoding];[command commit];[command waitUntilCompleted];
            if(command.status==MTLCommandBufferStatusCompleted&&outBuffer.contents){
                const auto* result=static_cast<const std::uint8_t*>(outBuffer.contents);ok=true;std::string mismatch;
                for(std::size_t k=0;k<tests.size();k++){
                    el::Result actual{};std::memcpy(actual.data(),result+k*sizeof(el::Result),sizeof(actual));
                    if(!ec::compare(tests[k],actual,mismatch)){std::cerr<<"FAIL: Metal camera-ray-coefficient arithmetic: "<<mismatch<<'\n';ok=false;}}
            }
            else error=command.error;
        }
        else if(encoder)[encoder endEncoding];
    }
    if(!ok){std::cerr<<"FAIL: Metal camera-ray-coefficient arithmetic probe";
        if(error)std::cerr<<": "<<error.localizedDescription.UTF8String;
        std::cerr<<'\n';}
#if !__has_feature(objc_arc)
    [outBuffer release];[inBuffer release];[pipeline release];[function release];[library release];
#endif
    return ok;
}


bool cameraLinePredicateNativeProbe()
{
    namespace ec=videohelper::tests::cameralinecases;namespace el=arbitgpu::cameraline;
    id<MTLCommandQueue> queue=(__bridge id<MTLCommandQueue>)sg_mtl_command_queue();
    if(!queue)return expect(false,"camera-line-predicate probe native Metal queue unavailable");
    const auto tests=ec::cases();if(tests.empty()||tests.size()>1024||tests.capacity()>1024)return expect(false,"camera-line-predicate probe case admission");
    std::uint64_t work=0,bytes=tests.capacity()*sizeof(ec::Case);
    for(const auto& test:tests){work+=ec::work(test);
        bytes+=ec::kProbeScratchBytes+2*sizeof(el::ProbeInput)+2*sizeof(el::Result);}
    if(work>1024*ec::kMaximumProbeWork
        ||bytes>1024*(ec::kProbeScratchBytes+2*sizeof(el::ProbeInput)+2*sizeof(el::Result)+sizeof(ec::Case)))
        return expect(false,"camera-line-predicate probe derived envelope exceeded");
    std::vector<el::ProbeInput> inputs;inputs.reserve(tests.size());for(const auto& test:tests)inputs.push_back(test.input);
    const auto source=el::metalProbeSource();NSError* error=nil;
    id<MTLLibrary> library=[queue.device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
        options:nil error:&error];
    id<MTLFunction> function=library?[library newFunctionWithName:@"cameraLinePredicateProbe"]:nil;
    id<MTLComputePipelineState> pipeline=function?[queue.device newComputePipelineStateWithFunction:function error:&error]:nil;
    id<MTLBuffer> inBuffer=nil,outBuffer=nil;
    bool ok=false;
    if(pipeline){
        inBuffer=[queue.device newBufferWithBytes:inputs.data() length:inputs.size()*sizeof(el::ProbeInput)
            options:MTLResourceStorageModeShared];
        outBuffer=[queue.device newBufferWithLength:tests.size()*sizeof(el::Result) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=command?[command computeCommandEncoder]:nil;
        if(inBuffer&&outBuffer&&command&&encoder){
            [encoder setComputePipelineState:pipeline];[encoder setBuffer:inBuffer offset:0 atIndex:0];
            [encoder setBuffer:outBuffer offset:0 atIndex:1];
            [encoder dispatchThreadgroups:MTLSizeMake(tests.size(),1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [encoder endEncoding];[command commit];[command waitUntilCompleted];
            if(command.status==MTLCommandBufferStatusCompleted&&outBuffer.contents){
                const auto* result=static_cast<const std::uint8_t*>(outBuffer.contents);ok=true;std::string mismatch;
                for(std::size_t k=0;k<tests.size();k++){
                    el::Result actual{};std::memcpy(actual.data(),result+k*sizeof(el::Result),sizeof(actual));
                    if(!ec::compare(tests[k],actual,mismatch)){std::cerr<<"FAIL: Metal camera-line-predicate arithmetic: "<<mismatch<<'\n';ok=false;}}
            }
            else error=command.error;
        }
        else if(encoder)[encoder endEncoding];
    }
    if(!ok){std::cerr<<"FAIL: Metal camera-line-predicate arithmetic probe";
        if(error)std::cerr<<": "<<error.localizedDescription.UTF8String;
        std::cerr<<'\n';
    }
#if !__has_feature(objc_arc)
    [outBuffer release];[inBuffer release];[pipeline release];[function release];[library release];
#endif
    return ok;
}

bool cameraWideLinePredicateNativeProbe()
{
    // Test-local logical admission covers constructor/predicate scratch, retained
    // cases, input and six readback-sized native/oracle/decoded static payloads.
    // Dynamic CPU oracle/compiled resources are separate; no scene owner proof.
    namespace ec=videohelper::tests::camerawidelinecases;namespace el=arbitgpu::camerawideline;
    id<MTLCommandQueue> queue=(__bridge id<MTLCommandQueue>)sg_mtl_command_queue();
    if(!queue)return expect(false,"camera-wide-line-predicate probe native Metal queue unavailable");
    const auto tests=ec::cases();if(tests.empty()||tests.size()>1024||tests.capacity()>1024)return expect(false,"camera-wide-line-predicate probe case admission");
    std::uint64_t work=0,bytes=tests.capacity()*sizeof(ec::Case);
    for(const auto& test:tests){work+=ec::work(test);
        bytes+=ec::kProbeScratchBytes+2*sizeof(el::ProbeInput)+6*sizeof(el::ProbeResult);}
    if(work>1024*ec::kMaximumProbeWork
        ||bytes>1024*(ec::kProbeScratchBytes+2*sizeof(el::ProbeInput)+6*sizeof(el::ProbeResult)+sizeof(ec::Case)))
        return expect(false,"camera-wide-line-predicate probe derived envelope exceeded");
    std::vector<el::ProbeInput> inputs;inputs.reserve(tests.size());for(const auto& test:tests)inputs.push_back(test.input);
    const auto source=el::metalProbeSource();NSError* error=nil;
    id<MTLLibrary> library=[queue.device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
        options:nil error:&error];
    id<MTLFunction> function=library?[library newFunctionWithName:@"cameraWideLinePredicateProbe"]:nil;
    id<MTLComputePipelineState> pipeline=function?[queue.device newComputePipelineStateWithFunction:function error:&error]:nil;
    id<MTLBuffer> inBuffer=nil,outBuffer=nil;
    bool ok=false;
    if(pipeline){
        inBuffer=[queue.device newBufferWithBytes:inputs.data() length:inputs.size()*sizeof(el::ProbeInput)
            options:MTLResourceStorageModeShared];
        outBuffer=[queue.device newBufferWithLength:tests.size()*sizeof(el::ProbeResult) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=command?[command computeCommandEncoder]:nil;
        if(inBuffer&&outBuffer&&command&&encoder){
            [encoder setComputePipelineState:pipeline];[encoder setBuffer:inBuffer offset:0 atIndex:0];
            [encoder setBuffer:outBuffer offset:0 atIndex:1];
            [encoder dispatchThreadgroups:MTLSizeMake(tests.size(),1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [encoder endEncoding];[command commit];[command waitUntilCompleted];
            if(command.status==MTLCommandBufferStatusCompleted&&outBuffer.contents){
                const auto* result=static_cast<const std::uint8_t*>(outBuffer.contents);ok=true;std::string mismatch;
                for(std::size_t k=0;k<tests.size();k++){
                    el::ProbeResult actual{};std::memcpy(actual.data(),result+k*sizeof(el::ProbeResult),sizeof(actual));
                    if(!ec::compare(tests[k],actual,mismatch)){std::cerr<<"FAIL: Metal camera-wide-line-predicate arithmetic: "<<mismatch<<'\n';ok=false;}}
            }
            else error=command.error;
        }
        else if(encoder)[encoder endEncoding];
    }
    if(!ok){std::cerr<<"FAIL: Metal camera-wide-line-predicate arithmetic probe";
        if(error)std::cerr<<": "<<error.localizedDescription.UTF8String;
        std::cerr<<'\n';
    }
#if !__has_feature(objc_arc)
    [outBuffer release];[inBuffer release];[pipeline release];[function release];[library release];
#endif
    return ok;
}

bool cameraPixelSignNativeProbe()
{
    // Test-local logical admission covers constructor/sign/host-preparation scratch, retained
    // cases, input and six readback-sized native/oracle/decoded static payloads.
    // Dynamic CPU oracle/compiled resources are separate; no scene owner proof.
    namespace ec=videohelper::tests::camerapixelsigncases;namespace el=arbitgpu::camerapixelsigns;
    id<MTLCommandQueue> queue=(__bridge id<MTLCommandQueue>)sg_mtl_command_queue();
    if(!queue)return expect(false,"camera-pixel-sign probe native Metal queue unavailable");
    const auto tests=ec::cases();if(tests.empty()||tests.size()>1024||tests.capacity()>1024)return expect(false,"camera-pixel-sign probe case admission");
    std::uint64_t work=0,bytes=tests.capacity()*sizeof(ec::Case);
    for(const auto& test:tests){work+=ec::work(test);
        bytes+=ec::kProbeScratchBytes+2*sizeof(el::ProbeInput)+6*sizeof(el::ProbeResult);}
    if(work>1024*ec::kMaximumProbeWork
        ||bytes>1024*(ec::kProbeScratchBytes+2*sizeof(el::ProbeInput)+6*sizeof(el::ProbeResult)+sizeof(ec::Case)))
        return expect(false,"camera-pixel-sign probe derived envelope exceeded");
    std::vector<el::ProbeInput> inputs;inputs.reserve(tests.size());for(const auto& test:tests)inputs.push_back(test.input);
    const auto source=el::metalProbeSource();NSError* error=nil;
    id<MTLLibrary> library=[queue.device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
        options:nil error:&error];
    id<MTLFunction> function=library?[library newFunctionWithName:@"cameraPixelSignProbe"]:nil;
    id<MTLComputePipelineState> pipeline=function?[queue.device newComputePipelineStateWithFunction:function error:&error]:nil;
    id<MTLBuffer> inBuffer=nil,outBuffer=nil;
    bool ok=false;
    if(pipeline){
        inBuffer=[queue.device newBufferWithBytes:inputs.data() length:inputs.size()*sizeof(el::ProbeInput)
            options:MTLResourceStorageModeShared];
        outBuffer=[queue.device newBufferWithLength:tests.size()*sizeof(el::ProbeResult) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=command?[command computeCommandEncoder]:nil;
        if(inBuffer&&outBuffer&&command&&encoder){
            [encoder setComputePipelineState:pipeline];[encoder setBuffer:inBuffer offset:0 atIndex:0];
            [encoder setBuffer:outBuffer offset:0 atIndex:1];
            [encoder dispatchThreadgroups:MTLSizeMake(tests.size(),1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [encoder endEncoding];[command commit];[command waitUntilCompleted];
            if(command.status==MTLCommandBufferStatusCompleted&&outBuffer.contents){
                const auto* result=static_cast<const std::uint8_t*>(outBuffer.contents);ok=true;std::string mismatch;
                for(std::size_t k=0;k<tests.size();k++){
                    el::ProbeResult actual{};std::memcpy(actual.data(),result+k*sizeof(el::ProbeResult),sizeof(actual));
                    if(!ec::compare(tests[k],actual,mismatch)){std::cerr<<"FAIL: Metal camera-pixel-sign arithmetic: "<<mismatch<<'\n';ok=false;}}
            }
            else error=command.error;
        }
        else if(encoder)[encoder endEncoding];
    }
    if(!ok){std::cerr<<"FAIL: Metal camera-pixel-sign arithmetic probe";
        if(error)std::cerr<<": "<<error.localizedDescription.UTF8String;
        std::cerr<<'\n';
    }
#if !__has_feature(objc_arc)
    [outBuffer release];[inBuffer release];[pipeline release];[function release];[library release];
#endif
    return ok;
}

struct MetalMultipleScatterArrival {
    glasscaustictests::Hit receiver;
    double fraction=0;
    std::uint64_t history=0;
    unsigned depth=0;
    std::array<std::uint8_t,32> ancestry{};
};
struct MetalMultipleScatterReference {
    bool complete=true,blocked=false;
    double escaped=0,unresolved=0,microResidual=0;
    unsigned events=0,draws=0;
    std::vector<MetalMultipleScatterArrival> arrivals;
};
// Binary64 authored-triangle macro traversal with the independent Smith walk.
// It retains the native finite stack policy so exact deposited prefixes match.
MetalMultipleScatterReference metalMultipleScatterReference(const glasscaustictests::Fixture& fixture,
    const arbitgpu::glasscaustics::EntrySample& entry,unsigned sample,unsigned band,
    const arbitgpu::glasscaustics::MultipleScatterPolicy& micro)
{
    namespace gt=glasscaustictests;
    MetalMultipleScatterReference result;
    const auto triangles=gt::worldTriangles(*fixture.scene);const auto source=gt::authoredSource(fixture);
    const auto point=gt::doublePoint(entry.position),towardSource=gt::normalized(source.direction);
    if(gt::closest(triangles,point,towardSource,1e-5,source.maximum).found) {result.blocked=true;return result;}
    constexpr std::array<double,8> wavelengths{.380,.420,.460,.500,.540,.580,.640,.700};
    const double ior=fixture.referenceIor+fixture.dispersion*(1/(wavelengths[band]*wavelengths[band])-1/(.58756*.58756));
    const auto hash=[](std::uint32_t value,std::uint32_t word){return (value^word)*16777619u;};
    auto entryKey=hash(hash(hash(hash(2166136261u,entry.object),entry.instance),entry.triangle),entry.quadrature);
    struct Photon {gt::D3 origin,direction;double fraction;std::uint64_t history;unsigned depth;
        bool inside;std::uint32_t samplingHistory;std::array<std::uint8_t,32> ancestry;};
    std::vector<Photon> pending{{gt::plus(point,towardSource,1e-6),gt::plus(gt::D3{},towardSource,-1),1,0,0,false,2166136261u,{}}};
    const auto& policy=fixture.policy->description();unsigned visits=0;
    while(!pending.empty()&&visits++<policy.maximumVisits) {
        const auto photon=pending.back();pending.pop_back();
        const auto hit=gt::closest(triangles,photon.origin,photon.direction,photon.depth?1e-9:0,source.maximum);
        if(!hit.found) {if(photon.inside) {result.complete=false;return result;}result.escaped+=photon.fraction;continue;}
        if(hit.object!=fixture.scene->objects[0].id.value) {
            if(photon.inside||!hit.entering||!photon.depth) {result.complete=false;return result;}
            result.arrivals.push_back({hit,photon.fraction,photon.history,photon.depth,photon.ancestry});continue;
        }
        if(photon.inside==hit.entering) {result.complete=false;return result;}
        if(photon.depth>=policy.maximumInterfaces) {result.unresolved+=photon.fraction;continue;}
        const auto normal=gt::plus(gt::D3{},hit.normal,photon.inside?-1:1);
        auto seed=hash(hash(hash(hash(2166136261u,entryKey),sample),photon.depth),photon.samplingHistory);
        const auto walk=gt::microfacetReference(photon.direction,normal,photon.inside?ior:1,photon.inside?1:ior,
            fixture.program->parameters.emissionRoughness[3],seed,micro.maximumEvents,micro.maximumExits,micro.maximumResidual);
        if(walk.status!=1) {result.complete=false;return result;}
        result.events+=walk.events;result.draws+=walk.draws;
        result.microResidual+=photon.fraction*walk.residual;result.unresolved+=photon.fraction*walk.residual;
        for(const auto& exit:walk.exits) {
            const double fraction=photon.fraction*exit.power;
            if(fraction==0)continue;
            if(fraction<policy.minimumBranchFraction||pending.size()>=policy.maximumPending) {result.unresolved+=fraction;continue;}
            auto child=photon;child.origin=gt::plus(hit.position,normal,exit.transmitted?-1e-8:1e-8);
            child.direction=exit.direction;child.fraction=fraction;
            child.history=photon.history|(exit.transmitted?0ull:(1ull<<photon.depth));
            child.ancestry[photon.depth]=std::uint8_t(exit.order|(exit.transmitted?64u:0u));
            child.samplingHistory=hash(hash(photon.samplingHistory,exit.order),std::uint32_t(exit.transmitted));
            child.depth=photon.depth+1;child.inside=exit.transmitted?!photon.inside:photon.inside;pending.push_back(child);
        }
    }
    for(const auto& photon:pending)result.unresolved+=photon.fraction;
    result.complete=result.unresolved<=policy.maximumUnresolvedFraction*(1+2e-4);
    return result;
}

bool metalMultipleScatterCausticCases(arbitgpu::NativeFixtureSceneBackend& backend,std::string& error)
{
    using namespace HarmonicMIDI::grid;
    namespace gc=arbitgpu::glasscaustics;namespace gt=glasscaustictests;
    const auto fail=[&](const char* message){error=message;return false;};
    gc::MultipleScatterPolicy micro;gc::EnergyLedger baseline;std::vector<float> baselineSpectrum;
    bool reflected=false,transmitted=false,bandResponse=false;
    for(unsigned test=0;test<6;++test) {
        const float roughness=test==1?.75f:.35f;
        auto fixture=gt::reflectionFixture(test==4?1:0);const auto ordinary=gt::roughFixture(roughness);
        fixture.scene->lights[0]=ordinary.scene->lights[0];fixture.ordinaryLight=ordinary.ordinaryLight;
        fixture.policy=ordinary.policy;fixture.light.reset();fixture.scene->cameras[0].transform.translation={0,0,-2};
        fixture.program->parameters.emissionRoughness[3]=roughness;fixture.scene->materials[0].roughness=roughness;
        if(test==2)fixture.scene->lights[0].spectralSource.radiance[3]*=2;
        if(test==3) {
            auto material=std::make_shared<gt::Program>(*fixture.program->objectPrograms[0]);
            material->spectralReflectance->coefficients[5]*=.5f;fixture.program->objectPrograms[0]=material;
        }
        fixture.program->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;
        for(auto& child:fixture.program->objectPrograms) {
            auto material=std::make_shared<gt::Program>(*child);material->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;child=material;
        }
        const auto native=backend.prepare(fixture.scene,fixture.program);
        if(!native.prepared) {error=native.error;return false;}
        gc::Prepared prepared;
        if(!arbitgpu::prepareMetalGlassCaustic(native.resources,fixture.ordinaryLight,*fixture.policy,prepared,error))return false;
        arbitgpu::NativeFixtureSceneRuntimeInputs runtime;runtime.linearColor=true;
        gc::SchedulingDescription scheduling;scheduling.entrySubdivisions=1;
        const auto width=test==5?113u:32u,height=test==5?87u:32u;
        const auto rendered=arbitgpu::renderMetalGlassCausticFrame(fixture.scene,native.resources,width,height,runtime,prepared,scheduling,&micro);
        if(!rendered.rendered||!rendered.frame) {error="Metal multiple scatter native positive failed: "+rendered.error;return false;}
        gc::NativeReadback receipt;gc::EnergyLedger energy;
        if(!arbitgpu::readMetalGlassCausticReceipt(rendered.frame,fixture.frame,receipt,energy,error))return false;
        const auto& schedule=fixture.frame;
        if(!schedule.multipleScatterPolicy||schedule.bounds.combinedWork>gc::kMaximumQueryWork
            ||energy.masked!=std::array<double,8>{})return fail("Metal multiple scatter version or energy admission is missing");
        const std::uint64_t minimumStorage=schedule.work.size()*sizeof(arbitgpu::metalcaustic::Path)
            +schedule.bounds.maximumDeposits*sizeof(arbitgpu::metalcaustic::MultipleScatterDeposit)
            +std::uint64_t(width)*height*8*sizeof(float);
        if(rendered.stats.textureBytes<minimumStorage||rendered.stats.textureBytes>renderpassoutput::kMaximumTotalBytes)
            return fail("Metal multiple scatter larger native storage is not admitted");
        std::uint64_t combinedWork=0;
        gc::EnergyLedger expected;std::vector<double> expectedSpectrum(receipt.spectralRadiance.size());
        const auto triangles=gt::worldTriangles(*fixture.scene);const auto source=gt::authoredSource(fixture);
        for(std::size_t w=0;w<schedule.work.size();++w) {
            const auto& work=schedule.work[w];const auto& entry=schedule.entries[work.entry];const auto& path=receipt.paths[w];
            combinedWork+=path.queryWork+path.microEvents+path.microDraws;
            const auto reference=metalMultipleScatterReference(fixture,entry,work.interfaceSample,work.band,micro);
            if(!reference.complete)return fail("Metal independent multiple scatter macro reference exceeds finite policy");
            const auto triangle=std::find_if(triangles.begin(),triangles.end(),[&](const auto& t) {
                return std::tie(t.object,t.material,t.instance,t.triangle)==std::tie(entry.object,entry.material,entry.instance,entry.triangle);});
            if(triangle==triangles.end())return fail("Metal independent multiple scatter entry triangle is missing");
            const double incident=source.radiance[work.band]*std::max(0.0,gt::inner(triangle->normal,source.direction))*triangle->area/16;
            const auto band=work.band;expected.potential[band]+=incident;
            if(reference.blocked)expected.sourceBlocked[band]+=incident;
            else {expected.launched[band]+=incident;expected.escaped[band]+=incident*reference.escaped;
                expected.unresolved[band]+=incident*reference.unresolved;}
            if(path.microEvents!=reference.events||path.microDraws!=reference.draws||path.sourceBlocked!=reference.blocked
                ||std::abs(double(path.microResidualPower)-incident*reference.microResidual)>incident*2e-6)
                return fail("Metal actual micro work or weighted residual differs from independent reference");
            std::size_t count=0;
            for(const auto& deposit:receipt.deposits)if(deposit.work==w) {
                const auto found=std::find_if(reference.arrivals.begin(),reference.arrivals.end(),[&](const auto& arrival) {
                    return arrival.ancestry==deposit.ancestry;});
                if(found==reference.arrivals.end()||found->history!=deposit.reflectionHistory||found->depth!=deposit.interfaces
                    ||found->receiver.object!=deposit.receiver.objectId||found->receiver.triangle!=deposit.receiver.triangleId
                    ||std::abs(found->fraction-deposit.fraction)>3e-3*std::max(found->fraction,1e-5))
                    return fail("Metal exact multiple scatter ancestry or receiver differs from independent reference");
                for(unsigned axis=0;axis<3;++axis)if(std::abs(found->receiver.position[axis]-deposit.receiver.position[axis])>.004)
                    return fail("Metal multiple scatter receiver endpoint differs from independent reference");
                const auto* material=arbitgpu::nativeSurfaceForObject(fixture.program.get(),SceneObjectId{found->receiver.object});
                if(!material||!material->spectralReflectance)return fail("Metal multiple scatter authored receiver rho is missing");
                const double rho=material->spectralReflectance->coefficients[band],power=incident*found->fraction;
                expected.deposited[band]+=power*rho;expected.absorbed[band]+=power*(1-rho);
                // Independently project the verified native endpoint and integrate
                // its authored receiver plane over the full pixel footprint.
                const gt::D3 camera{0,0,-2},position=gt::doublePoint(deposit.receiver.position);
                const double depth=camera[2]-position[2],tangent=std::tan(double(schedule.camera.verticalFovRadians)/2);
                const double aspect=double(width)/height;
                bool visible=false;
                if(depth>=schedule.camera.nearPlane&&depth<=schedule.camera.farPlane) {
                    const double px=(position[0]/(depth*tangent*aspect)+1)*width/2,py=(1-position[1]/(depth*tangent))*height/2;
                    if(px>=0&&py>=0&&px<width&&py<height) {
                        const auto x=std::uint32_t(px),y=std::uint32_t(py);
                        const auto ray=[&](double x,double y){return gt::normalized(gt::D3{(2*x/width-1)*tangent*aspect,(1-2*y/height)*tangent,-1});};
                        const auto delta=gt::minus(position,camera);const double distance=std::sqrt(gt::inner(delta,delta));
                        const auto endpoint=gt::closest(triangles,camera,gt::normalized(delta),0,distance+1e-3);
                        const auto center=gt::closest(triangles,camera,ray(x+.5,y+.5),0,schedule.camera.farPlane*1e4);
                        const auto same=[&](const gt::Hit& hit){return hit.found&&hit.object==found->receiver.object&&hit.material==found->receiver.material
                            &&hit.instance==found->receiver.instance&&hit.triangle==found->receiver.triangle;};
                        visible=gt::inner(found->receiver.normal,ray(x+.5,y+.5))<-1e-6&&same(endpoint)&&same(center);
                        if(visible) {
                            std::array<gt::D3,4> corners;constexpr std::array<std::array<double,2>,4> offsets{{{0,0},{1,0},{1,1},{0,1}}};
                            for(unsigned c=0;c<4;++c) {
                                const auto direction=ray(x+offsets[c][0],y+offsets[c][1]);
                                corners[c]=gt::plus(camera,direction,gt::inner(found->receiver.normal,delta)/gt::inner(found->receiver.normal,direction));
                            }
                            const auto area=[&](unsigned a,unsigned b,unsigned c) {const auto cross=gt::exterior(gt::minus(corners[b],corners[a]),gt::minus(corners[c],corners[a]));return .5*std::sqrt(gt::inner(cross,cross));};
                            const double footprint=area(0,1,2)+area(0,2,3);
                            if(deposit.footprint.x!=x||deposit.footprint.y!=y||!gc::nearPower(deposit.footprint.worldArea,footprint,3e-4)
                                ||deposit.raster.object!=center.object||deposit.raster.material!=center.material||deposit.raster.instance!=center.instance||deposit.raster.triangle!=center.triangle)
                                return fail("Metal multiple scatter pixel area or raster ownership differs from independent projection");
                            expectedSpectrum[(std::size_t(y)*width+x)*8+band]+=power*rho/(std::acos(-1.0)*footprint);
                        }
                    }
                }
                if(visible!=deposit.cameraVisible)return fail("Metal multiple scatter camera visibility differs from independent authored triangles");

                reflected|=deposit.reflectionHistory!=0;transmitted|=deposit.reflectionHistory==0;++count;
            }
            if(count!=reference.arrivals.size())return fail("Metal multiple scatter omitted an independently deposited branch");
            if(test==0&&w==0) {
                arbitgpu::metalcaustic::Path packed{};packed.complete=1;packed.reserved=2;packed.roughTransportVersion=1;
                packed.interfaceSample=path.interfaceSample;
                const auto extension=gc::pathExtensionTexels(path,gc::PathExtensionLayout::Metal);std::copy(extension[0].begin(),extension[0].end(),packed.padding+17);
                gc::NativePathReceipt decoded;decoded.entryHit=path.entryHit;
                if(!arbitgpu::metalcaustic::decodePathReceiptExtensions(packed,true,decoded)
                    ||decoded.entryHit.instanceId!=path.entryHit.instanceId||decoded.entryHit.triangleId!=path.entryHit.triangleId)
                    return fail("Metal actual path extension did not retain its typed entry IDs");
                packed.padding[17]+=.5f;
                if(arbitgpu::metalcaustic::decodePathReceiptExtensions(packed,true,decoded))return fail("Metal fractional native micro receipt was accepted");
                packed.padding[17]-=.5f;packed.reserved=3;
                if(arbitgpu::metalcaustic::decodePathReceiptExtensions(packed,true,decoded))return fail("Metal unknown native micro version was accepted");
                packed.reserved=2;packed.padding[0]=1;
                if(arbitgpu::metalcaustic::decodePathReceiptExtensions(packed,true,decoded))return fail("Metal nonzero reserved native padding was accepted");
            }
        }
        if(combinedWork>schedule.bounds.combinedWork)return fail("Metal measured combined query and optical work exceeds128Mi admission");
        for(unsigned band=0;band<8;++band) {
            for(const auto pair:{std::pair<double,double>{energy.potential[band],expected.potential[band]},
                {energy.launched[band],expected.launched[band]},{energy.deposited[band],expected.deposited[band]},
                {energy.absorbed[band],expected.absorbed[band]},{energy.escaped[band],expected.escaped[band]}})
                if(!gc::nearPower(pair.first,pair.second,3e-3))return fail("Metal multiple scatter band flux differs from independent physical reference");
            if(energy.unresolved[band]>energy.launched[band]*1.0002e-4
                ||!gc::nearPower(energy.launched[band],energy.deposited[band]+energy.absorbed[band]+energy.escaped[band]+energy.unresolved[band]))
                return fail("Metal multiple scatter frame energy is not conserved within its original tail");
        }
        for(std::size_t at=0;at<expectedSpectrum.size();++at)
            if(!gc::nearPower(receipt.spectralRadiance[at],expectedSpectrum[at],3e-3))
                return fail("Metal multiple scatter spectrum differs from independent flux and pixel area");
        if(test==0) {baseline=energy;baselineSpectrum=receipt.spectralRadiance;}
        if(test==1&&baselineSpectrum==receipt.spectralRadiance)return fail("Metal substantial roughness did not change the receiver footprint");
        if(test==2&&!gc::nearPower(energy.deposited[3],2*baseline.deposited[3],3e-3))return fail("Metal multiple scatter source Li control is not causal");
        if(test==3)bandResponse=energy.deposited[5]<baseline.deposited[5];
        if(test==4&&std::any_of(energy.launched.begin(),energy.launched.end(),[](double p){return p!=0;}))
            return fail("Metal multiple scatter source blocker did not remove launched flux");
        gc::EnergyLedger sentinel;sentinel.launched.fill(19);const auto reject=[&](gc::NativeReadback changed) {
            auto ledger=sentinel;std::string negative;
            return !gc::validateNativeReadback(prepared,schedule,changed,ledger,negative,&receipt.deposition.submission)
                &&!negative.empty()&&ledger.launched==sentinel.launched;
        };
        if(test!=4) {
            auto changed=receipt;changed.paths[0].multipleScatterVersion=1;
            if(!reject(changed))return fail("Metal stale multiple scatter marker published a ledger");
            changed=receipt;changed.paths[0].microDraws++;
            if(!reject(changed))return fail("Metal mismatched random draw receipt published a ledger");
            if(!receipt.deposits.empty()) {
                changed=receipt;changed.deposits[0].ancestry[0]^=64;
                if(!reject(changed))return fail("Metal wrong exact medium ancestry published a ledger");
                changed=receipt;changed.deposits[0].ancestry.back()=128;
                if(!reject(changed))return fail("Metal invalid ancestry tail published a ledger");
            }
        }
        if(test==0) {
            const auto repeat=arbitgpu::renderMetalGlassCausticFrame(fixture.scene,native.resources,width,height,runtime,prepared,scheduling,&micro);
            arbitgpu::NativeRawPassPixels first,second;
            gc::Frame repeatedFrame;gc::NativeReadback repeatedReceipt;gc::EnergyLedger repeatedEnergy;
            if(!repeat.rendered||!repeat.frame||!arbitgpu::readMetalGlassCausticReceipt(repeat.frame,repeatedFrame,repeatedReceipt,repeatedEnergy,error)
                ||!rendered.frame->readRawPass(renderpassoutput::Output::Color,first,error)
                ||!repeat.frame->readRawPass(renderpassoutput::Output::Color,second,error))return false;
            if(first.bytes!=second.bytes||receipt.spectralRadiance!=repeatedReceipt.spectralRadiance
                ||receipt.deposits.size()!=repeatedReceipt.deposits.size())return fail("Metal multiple scatter repeated completed bytes changed");
            for(std::size_t d=0;d<receipt.deposits.size();++d)if(receipt.deposits[d].ancestry!=repeatedReceipt.deposits[d].ancestry)
                return fail("Metal repeated multiple scatter exact ancestry changed");
            auto limited=micro;limited.maximumEvents=1;limited.maximumResidual=1e-12f;
            const auto refused=arbitgpu::renderMetalGlassCausticFrame(fixture.scene,native.resources,width,height,runtime,prepared,scheduling,&limited);
            if(refused.rendered||refused.frame||refused.error.empty())return fail("Metal unfinished micro walk published partial Color");
            auto foreign=std::make_shared<const Visual3DScene>(*fixture.scene);
            const auto stale=arbitgpu::renderMetalGlassCausticFrame(foreign,native.resources,width,height,runtime,prepared,scheduling,&micro);
            if(stale.rendered||stale.frame||stale.error.empty())return fail("Metal foreign multiple scatter scene owner published Color");
        }
    }
    return reflected&&transmitted&&bandResponse?true:fail("Metal multiple scatter fixture lacks reflected, transmitted or spectral rho response");
}

struct MetalCameraSlabReference {
    glasscaustictests::D3 direction {},entry {},exit {},terminal {},inside {};
    double entryFresnel=0,exitFresnel=0;
};
// Independent double plane intersections and optical-admittance Snell oracle.
// This fixture reference does not supply any producer ray, hit or branch data.
MetalCameraSlabReference metalCameraSlabReference(const arbitgpu::glasscaustics::Frame& frame,
    unsigned x,unsigned y,unsigned location,unsigned band)
{
    namespace gt=glasscaustictests;
    const double offsets[5][2]={{.5,.5},{0,0},{1,0},{1,1},{0,1}};
    const auto& camera=frame.camera;const auto& q=camera.transform.rotation;
    const double tangent=std::tan(double(camera.verticalFovRadians)*.5);
    const gt::D3 local{(2*(x+offsets[location][0])/frame.width-1)*tangent*frame.width/frame.height,
        (1-2*(y+offsets[location][1])/frame.height)*tangent,-1};
    const gt::D3 vector{q.x,q.y,q.z};const auto inner=gt::exterior(vector,local);
    const auto rotated=gt::plus(local,gt::plus(gt::exterior(vector,inner),inner,q.w),2);
    MetalCameraSlabReference out;out.direction=gt::normalized(rotated);
    const gt::D3 origin{camera.transform.translation.x,camera.transform.translation.y,camera.transform.translation.z};
    out.entry=gt::plus(origin,out.direction,(.5-origin[2])/out.direction[2]);
    const auto entry=gt::snell(out.direction,{0,0,1},1,frame.ior[band]);
    out.inside=entry.direction;out.entryFresnel=entry.reflection;
    out.exit=gt::plus(out.entry,out.inside,(-.5-out.entry[2])/out.inside[2]);
    const auto exit=gt::snell(out.inside,{0,0,1},frame.ior[band],1);out.exitFresnel=exit.reflection;
    out.terminal=gt::plus(out.exit,exit.direction,(-6-out.exit[2])/exit.direction[2]);
    return out;
}

bool stageMetalCameraGlassGratingForFixture(unsigned scenario,
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame,
    std::shared_ptr<const arbitgpu::glasscaustics::GratingIncidentBinding>& binding,std::string& error)
{
    namespace gc=arbitgpu::glasscaustics;namespace gt=glasscaustictests;
    auto fixture=gt::ordinaryFixture();auto& glassObject=fixture.scene->objects[0];
    for(std::size_t v=glassObject.firstVertex;v<glassObject.firstVertex+glassObject.vertexCount;++v) {
        auto& p=fixture.scene->vertices[v].position;
        if(p.z>0)p.z=.5f+(scenario==1?.45f*p.x:0.0f);
    }
    auto& camera=fixture.scene->cameras[0];camera.transform.translation={.18f,.11f,3};
    camera.transform.rotation={0,float(std::sin(.08)),0,float(std::cos(.08))};camera.verticalFovRadians=.2f;
    if(scenario==3){camera.transform.translation={0,0,3};camera.transform.rotation={0,0,0,1};}
    if(scenario==6){camera.transform.translation={0,0,0};camera.transform.rotation={0,0,0,1};}
    if(scenario==4)fixture.program->parameters.transmissionIorClearcoat={1,1,0,0};
    if(scenario==2) {
        videohelper::tests::smoothglassfixture::receiver(*fixture.scene,-1.5f,{0,0,0});
        auto blocker=std::make_shared<gt::Program>(*fixture.program->objectPrograms[0]);
        blocker->object=fixture.scene->objects[2].id;blocker->authoredMaterial=fixture.scene->objects[2].material;
        blocker->parameters.identifiers[0]=blocker->authoredMaterial.value;
        blocker->programIdentity="metal-camera-glass-obstruction";blocker->bindingDigest=blocker->programIdentity;
        fixture.program->objectPrograms.push_back(blocker);
    }
    auto& target=fixture.scene->objects[1];
    for(std::size_t v=target.firstVertex;v<target.firstVertex+target.vertexCount;++v) {
        auto& vertex=fixture.scene->vertices[v];vertex.uv={vertex.position.x/12,vertex.position.y/12};
    }
    fixture.program->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;
    std::vector<std::shared_ptr<const gt::Program>> ordinary;
    for(std::size_t i=1;i<fixture.program->objectPrograms.size();++i) {
        auto child=std::make_shared<gt::Program>(*fixture.program->objectPrograms[i]);
        child->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;ordinary.push_back(child);
    }
    fixture.program->objectPrograms.clear();
    auto description=diffractionmaterial::makeAluminiumBinaryGratingPreset();description.roughness={0,0};
    description.spectrum.firstOrder=description.spectrum.lastOrder=1;description.microstructure.grooveDepthNanometres=.9f;
    const auto admitted=diffractionmaterial::admit(description,error);if(!admitted)return false;
    const auto material=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*admitted);
    diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest request;
    request.version=diffractionmaterialbinding::kSceneReceiverWireVersion;request.scene=fixture.scene->id;request.object=target.id;
    request.sceneRevision=request.structuralRevision=request.evaluationRevision=request.materialRevision=1;
    request.material=description;request.structuralDigest=material->structuralDigest();
    request.lighting.version=3;request.lighting.pathCount=1;request.lighting.traceMaximumDistance=20;
    request.lighting.sceneReceiverTransport=diffractionmaterial::SceneReceiverTransport::AuthoredSpectralLambertian;
    request.lighting.paths[0].kind=diffractionmaterial::LightingPathKind::Direct;
    request.lighting.paths[0].incident.direction={0,0,1};request.lighting.paths[0].incident.radiance=fixture.scene->lights[0].spectralSource.radiance;
    const auto leaf=gc::compileDiscreteGrating(*fixture.scene,request,arbitgpu::NativeFixtureMaterialBackend::Metal,error);
    if(!leaf)return false;
    auto root=std::make_shared<gt::Program>(*leaf);root->objectPrograms={fixture.program};
    for(const auto& child:ordinary)root->objectPrograms.push_back(child);
    const auto resources=arbitgpu::prepareMetalGlassGratingResources(fixture.scene,root,{target.id,material},leaf);
    if(!resources.prepared){error=resources.error;return false;}
    arbitgpu::NativeFixtureSceneRuntimeInputs runtime;runtime.linearColor=true;runtime.timeSeconds=.25f;
    gc::SchedulingDescription scheduling;scheduling.entrySubdivisions=1;
    return arbitgpu::stageMetalGlassGratingFrame(resources.resources,fixture.ordinaryLight,*fixture.policy,33,33,runtime,.25,
        frame,binding,error,scheduling);
}

bool metalGlassCameraGratingTraceCases(std::string& error)
{
    namespace gc=arbitgpu::glasscaustics;namespace gt=glasscaustictests;
    const auto fail=[&](const char* message){error=message;return false;};
    const auto near=[](const auto& actual,const auto& expected,double tolerance=3e-4) {
        for(unsigned a=0;a<3;++a)if(std::abs(double(actual[a])-expected[a])>tolerance)return false;return true;
    };
    for(unsigned scenario=0;scenario<6;++scenario) {
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame;
        std::shared_ptr<const gc::GratingIncidentBinding> binding;
        if(!stageMetalCameraGlassGratingForFixture(scenario,frame,binding,error))return false;
        if(!arbitgpu::inspectMetalRasterCaptureForFixture(frame,true,error))return false;
        if(!dynamic_cast<const gc::NativeGlassCameraGratingTraceBinding*>(frame.get()))return fail("Metal camera trace optional native binding is absent");
        const unsigned x=scenario==5?17:16,y=scenario==5?15:16;
        std::vector<gc::NativeGlassCameraGratingTraceInput> inputs;inputs.reserve(5);
        for(unsigned location=0;location<5;++location) {
            gc::NativeGlassCameraGratingTraceInput input;
            if(!gc::makeGlassCameraGratingTraceInput(binding->frame,x,y,location,input,error))return false;inputs.push_back(input);
        }
        gc::NativeGlassCameraGratingTraceReadback receipt;
        if(!arbitgpu::produceMetalGlassCameraGratingTrace(frame,binding,inputs,receipt,error)
            ||!arbitgpu::validateMetalGlassCameraGratingTrace(frame,receipt.submission,.25,receipt,error)
            ||!arbitgpu::validateMetalGlassCameraGratingSharedEntryForFixture(frame,receipt,error))return false;
        if(!receipt.dispatchComplete||receipt.overflow||receipt.bands.size()!=40||!receipt.queryWork)
            return fail("Metal camera trace lost its actual five samples or eight native bands");
        const auto triangles=gt::worldTriangles(*binding->prepared.glass.sourceScene);bool tir=false,tirContinued=false,grating=false;
        for(std::size_t index=0;index<receipt.bands.size();++index) {
            const auto& band=receipt.bands[index];
            if(band.sample!=index/8||band.band!=index%8||band.visits.size()<3
                ||!gc::sameGlassCameraGratingTraceInput(band.input,inputs[index/8])||!band.unresolvedFractionKnown
                ||(band.traceComplete?band.unresolvedFraction!=0:band.unresolvedFraction<=0))
                return fail("Metal camera native sample echo, branch prefix or known frontier fraction changed");
            const double lambda=diffractionmaterial::kProductionWavelengthsNanometres[index%8]*.001;
            const double expectedIor=scenario==4?1:1.5+.004*(1/(lambda*lambda)-1/(.58756*.58756));
            if(std::abs(band.ior-expectedIor)>3e-6)return fail("Metal camera trace did not use its authored eight-band Cauchy IOR");
            for(const auto& visit:band.visits) {
                const gt::D3 origin{visit.originMinimum[0],visit.originMinimum[1],visit.originMinimum[2]};
                const gt::D3 direction{visit.directionFraction[0],visit.directionFraction[1],visit.directionFraction[2]};
                const auto hit=gt::closest(triangles,origin,direction,visit.originMinimum[3],binding->frame.camera.farPlane);
                if(visit.hit.status==1&&(!hit.found||!near(visit.hit.position,hit.position)
                    ||!near(visit.hit.outwardNormal,hit.normal)||hit.object!=visit.hit.objectId||hit.material!=visit.hit.materialId))
                    return fail("Metal camera native closest hit differs from independent authored planes");
                if(visit.disposition!=gc::GlassCameraTraceDisposition::Interface
                    &&(visit.tir||visit.orientedNormal!=gc::Point{}||visit.transmitted!=gc::Point{}||visit.reflected!=gc::Point{}
                        ||visit.reflection||visit.transmittedFraction||visit.reflectedFraction||visit.fromIor||visit.toIor))
                    return fail("Metal camera native terminal retained contradictory interface fields");
                if(visit.disposition!=gc::GlassCameraTraceDisposition::Grating&&visit.barycentric!=gc::Point{})
                    return fail("Metal camera native non-grating visit retained grating barycentrics");
                if(visit.disposition==gc::GlassCameraTraceDisposition::Interface) {
                    const auto optics=gt::snell(direction,{visit.orientedNormal[0],visit.orientedNormal[1],visit.orientedNormal[2]},visit.fromIor,visit.toIor);
                    const auto reflected=gt::normalized(gt::plus(direction,{visit.orientedNormal[0],visit.orientedNormal[1],visit.orientedNormal[2]},
                        -2*gt::inner(direction,{visit.orientedNormal[0],visit.orientedNormal[1],visit.orientedNormal[2]})));
                    if(optics.tir!=visit.tir||std::abs(optics.reflection-visit.reflection)>1e-4||!near(visit.reflected,reflected)
                        ||(!optics.tir&&!near(visit.transmitted,optics.direction)))
                        return fail("Metal camera native raw Fresnel, TIR or outgoing directions differ from independent optics");
                    tir|=visit.tir;
                    if(visit.tir)for(const auto& child:band.visits)
                        if(child.parent==visit.visit&&child.token==1&&child.inside==visit.inside)tirContinued=true;
                }
                grating|=visit.disposition==gc::GlassCameraTraceDisposition::Grating;
            }
            const auto& first=band.visits[0];const auto& second=band.visits[1];const auto& terminal=band.visits[2];
            if(first.parent!=gc::kGlassCameraTraceRoot||first.inside||first.interfaces||first.disposition!=gc::GlassCameraTraceDisposition::Interface
                ||second.parent!=0||second.token!=65||!second.inside||second.interfaces!=1
                ||terminal.parent!=1||terminal.token!=65||terminal.inside||terminal.interfaces!=2)
                return fail("Metal camera primary transmitted branch lost actual native ancestry or media");
            if(scenario==0||scenario==4||scenario==5) {
                const auto reference=metalCameraSlabReference(binding->frame,x,y,unsigned(index/8),unsigned(index%8));
                if(!near(band.nativeDirectionMaximum,reference.direction,2e-5)||!near(first.hit.position,reference.entry)
                    ||!near(second.hit.position,reference.exit)||!near(terminal.hit.position,reference.terminal)
                    ||std::abs(first.reflection-reference.entryFresnel)>2e-5||std::abs(second.reflection-reference.exitFresnel)>2e-5
                    ||terminal.disposition!=gc::GlassCameraTraceDisposition::Grating)
                    return fail("Metal camera slab native ray, Fresnel split, lateral shift or grating endpoint differs from the double oracle");
                if(scenario==4&&(!band.traceComplete||band.reasons||band.visits.size()!=3))
                    return fail("Metal camera index-matched slab did not complete its actual three-query path");
                if(scenario!=4&&(band.traceComplete||!band.reasons||band.unresolvedFraction<=0))
                    return fail("Metal camera nonzero reflection cap became a completed clear path");
            }
            if(scenario==2&&terminal.disposition!=gc::GlassCameraTraceDisposition::Other)
                return fail("Metal camera trace passed through actual geometry between glass and grating");
            if(scenario==3&&index/8==0&&(band.traceComplete||!(band.reasons&gc::GlassCameraTopologyFailed)
                ||terminal.disposition!=gc::GlassCameraTraceDisposition::Unresolved))
                return fail("Metal camera strict grating seam became a completed terminal");
        }
        if(scenario==1&&(!tir||!tirContinued||receipt.bands[0].visits[0].transmitted==receipt.bands[7].visits[0].transmitted
            ||receipt.bands[0].visits[0].transmitted==receipt.bands[8].visits[0].transmitted))
            return fail("Metal nonparallel camera glass trace omitted actual TIR continuation or corner/band branch divergence");
        if(scenario!=2&&!grating)return fail("Metal camera native trace omitted every strict grating terminal");
        if(scenario==0) {
            if(receipt.bands[0].visits[1].hit.position==receipt.bands[7].visits[1].hit.position
                ||receipt.bands[0].nativeDirectionMaximum==receipt.bands[8].nativeDirectionMaximum)
                return fail("Metal camera spectral or corner paths were reused across distinct native samples");
        }
        auto changed=receipt;changed.bands[0].nativeDirectionMaximum[0]=std::nextafter(changed.bands[0].nativeDirectionMaximum[0],1.0f);
        if(arbitgpu::validateMetalGlassCameraGratingSharedEntryForFixture(frame,changed,error)
            ||error!="Metal camera glass trace differs from its actual retained native output")return fail("Metal direct shared camera entry accepted a changed caller native ray");
        changed=receipt;changed.bands[0].visits[0].reflection=std::nextafter(changed.bands[0].visits[0].reflection,1.0f);
        if(arbitgpu::validateMetalGlassCameraGratingSharedEntryForFixture(frame,changed,error)
            ||error!="Metal camera glass trace differs from its actual retained native output")return fail("Metal direct shared camera entry accepted changed caller interface optics");
        changed=receipt;changed.bands[0].unresolvedFractionKnown=false;
        if(arbitgpu::validateMetalGlassCameraGratingSharedEntryForFixture(frame,changed,error)
            ||error!="Metal camera glass trace differs from its actual retained native output")
            return fail("Metal direct shared camera entry accepted a changed caller frontier-known flag");
        changed=receipt;changed.bands[0].unresolvedFraction=std::nextafter(changed.bands[0].unresolvedFraction,2.0f);
        if(arbitgpu::validateMetalGlassCameraGratingSharedEntryForFixture(frame,changed,error)
            ||error!="Metal camera glass trace differs from its actual retained native output")
            return fail("Metal direct shared camera entry accepted a changed caller frontier fraction");
        changed=receipt;changed.inputGeneration^=1u;
        if(arbitgpu::validateMetalGlassCameraGratingTrace(frame,receipt.submission,.25,changed,error)
            ||arbitgpu::validateMetalGlassCameraGratingTrace(frame,receipt.submission,.375,receipt,error))
            return fail("Metal camera trace accepted stale input generation or evaluation time");
        auto stale=receipt.submission;stale.completedFence=0;
        if(arbitgpu::validateMetalGlassCameraGratingTrace(frame,stale,.25,receipt,error))return fail("Metal camera trace accepted an incomplete fence");
        if(scenario==4&&!arbitgpu::inspectMetalGlassCameraTraceRefusalsForFixture(frame,error))return false;
        if(!arbitgpu::validateMetalGlassCameraGratingTrace(frame,receipt.submission,.25,receipt,error)
            ||frame->colorImageHandle()||frame->depthImageHandle()||frame->colorTextureViewHandle()||frame->depthTextureViewHandle()
            ||!arbitgpu::inspectMetalGlassGratingCameraAllocationForFixture(frame,binding,error))
            return fail("Metal camera trace lost prior native ownership or exposed a complete public base");
    }
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> insideFrame;
    std::shared_ptr<const gc::GratingIncidentBinding> insideBinding;
    if(!stageMetalCameraGlassGratingForFixture(6,insideFrame,insideBinding,error))return false;
    gc::NativeGlassCameraGratingTraceInput insideInput;
    if(gc::makeGlassCameraGratingTraceInput(insideBinding->frame,16,16,0,insideInput,error))
        return fail("Metal camera trace admitted an unsupported internal or unknown camera medium");
    insideInput.pixel={16,16,0,1};insideInput.control={33,33,0,0};insideInput.originMinimum={0,0,0,gc::kRayMinimum};
    insideInput.directionMaximum={0,0,-1,insideBinding->frame.camera.farPlane};
    gc::NativeGlassCameraGratingTraceReadback untouched;untouched.version=0xdeadbeefu;const auto before=untouched;
    if(arbitgpu::produceMetalGlassCameraGratingTrace(insideFrame,insideBinding,{insideInput},untouched,error)
        ||!gc::sameGlassCameraGratingTraceReadback(untouched,before))return fail("Metal camera initial-medium refusal changed caller output");
    error.clear();return true;
}

bool metalDirectGratingIrradianceCases(std::string& error)
{
    namespace gc=arbitgpu::glasscaustics;namespace gt=glasscaustictests;
    const auto fail=[&](const char* message){error=message;return false;};
    for(unsigned scenario=0;scenario<3;++scenario) {
        auto fixture=gt::ordinaryFixture();auto& light=fixture.scene->lights[0];
        light.id={0xfedcba98u};fixture.ordinaryLight=light.id;
        light.transform.rotation=scenario==2?HarmonicMIDI::grid::SceneQuaternion{0,1,0,0}:HarmonicMIDI::grid::SceneQuaternion{0,0,0,1};
        light.spectralSource.directionalIrradiance={1,diffractionmaterial::DirectionalSpectralMeasure::IrradianceCoefficient,
            {.25f,.5f,.75f,1,1.25f,1.5f,1.75f,2}};
        light.spectralSource.radiance.fill(0);
        auto& target=fixture.scene->objects[1];target.id={0x98765432u};target.material={0x87654321u};
        fixture.scene->materials[1].id=target.material;
        for(std::size_t v=target.firstVertex;v<target.firstVertex+target.vertexCount;++v) {
            auto& vertex=fixture.scene->vertices[v];vertex.uv={vertex.position.x/12,vertex.position.y/12};
        }
        auto glass=std::make_shared<gt::Program>(*fixture.program);glass->objectPrograms.clear();
        glass->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;
        auto description=diffractionmaterial::makeAluminiumBinaryGratingPreset();description.roughness={0,0};
        description.spectrum.firstOrder=description.spectrum.lastOrder=1;
        description.microstructure.grooveDepthNanometres=.9f;
        const auto ir=diffractionmaterial::admit(description,error);if(!ir)return false;
        const auto material=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*ir);
        diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest request;
        request.version=diffractionmaterialbinding::kSceneReceiverWireVersion;request.scene=fixture.scene->id;request.object=target.id;
        request.sceneRevision=request.structuralRevision=request.evaluationRevision=request.materialRevision=1;
        request.material=description;request.structuralDigest=material->structuralDigest();
        request.lighting.version=3;request.lighting.pathCount=1;request.lighting.traceMaximumDistance=20;
        request.lighting.sceneReceiverTransport=diffractionmaterial::SceneReceiverTransport::AuthoredSpectralLambertian;
        request.lighting.paths[0].kind=diffractionmaterial::LightingPathKind::Direct;
        request.lighting.paths[0].incident.direction={0,0,1};
        request.lighting.paths[0].incident.radiance={.25f,.5f,.75f,1,1.25f,1.5f,1.75f,2};
        const auto leaf=gc::compileDiscreteGrating(*fixture.scene,request,arbitgpu::NativeFixtureMaterialBackend::Metal,error);
        if(!leaf)return false;
        auto root=std::make_shared<gt::Program>(*leaf);root->objectPrograms={glass};
        const auto resources=arbitgpu::prepareMetalGlassGratingResources(fixture.scene,root,{target.id,material},leaf);
        if(!resources.prepared){error=resources.error;return false;}
        arbitgpu::NativeFixtureSceneRuntimeInputs runtime;runtime.linearColor=true;runtime.timeSeconds=.25f;
        gc::SchedulingDescription scheduling;scheduling.entrySubdivisions=1;
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame;
        std::shared_ptr<const gc::GratingIncidentBinding> binding;
        if(!arbitgpu::stageMetalGlassGratingFrame(resources.resources,light.id,*fixture.policy,32,32,runtime,.25,
            frame,binding,error,scheduling))return false;
        const auto* native=dynamic_cast<const gc::NativeDirectGratingIrradianceBinding*>(frame.get());
        if(!native||binding->triangles.empty())return fail("Metal direct endpoint binding is absent on real staging frame");
        const auto& t=binding->triangles[0];gc::Point position{};
        const gc::Point bary=scenario==1?gc::Point{.45f,.1f,.45f}:gc::Point{1.0f/3,1.0f/3,1.0f/3};
        for(unsigned v=0;v<3;++v)position=gc::advance(position,t.positions[v],bary[v]);
        const auto inputAt=[&](const gc::Point& p) {
            gc::NativeDirectGratingEndpointInput input;input.primitive={t.object,t.material,t.instance,t.triangle};
            const auto origin=gc::advance(p,t.normal,.25f);input.originMinimum={origin[0],origin[1],origin[2],gc::kRayMinimum};
            input.directionMaximum={-t.normal[0],-t.normal[1],-t.normal[2],1};return input;
        };
        std::vector<gc::NativeDirectGratingEndpointInput> endpoints{inputAt(position)};
        gc::NativeDirectGratingContributionReadback receipt;
        if(!arbitgpu::produceMetalDirectGratingIrradiance(frame,binding,endpoints,receipt,error)
            ||!arbitgpu::validateMetalDirectGratingIrradiance(frame,receipt.submission,.25,receipt,error))return false;
        if(receipt.samples.size()!=1)return fail("Metal direct contribution lost its single native endpoint result");
        const auto& sample=receipt.samples[0];
        if(!receipt.complete||receipt.bandWork!=8||!sample.endpoint.work||!sample.sourceHit.work
            ||receipt.queryWork!=sample.endpoint.work+std::uint64_t(sample.sourceHit.work)
            ||sample.source.input.identifiers[3]!=0xfedcba98u||sample.endpoint.objectId!=0x98765432u
            ||sample.endpoint.materialId!=0x87654321u||sample.input.primitive!=endpoints[0].primitive
            ||sample.outcome!=(scenario==0?gc::NativeDirectGratingContributionSample::Outcome::Positive
                :gc::NativeDirectGratingContributionSample::Outcome::Zero)
            ||sample.sourceHit.status!=(scenario==1?1:0))return fail("Metal direct native positive, actual glass blocker or nonpositive cosine outcome changed");
        const float cosine=std::max(0.0f,gc::dot(t.normal,gc::Point{binding->frame.sourceDirectionTrace[0],
            binding->frame.sourceDirectionTrace[1],binding->frame.sourceDirectionTrace[2]}));
        for(unsigned band=0;band<8;++band)
            if(!gc::nearDirectionalSourcePotential(sample.surfaceIrradiance[band],scenario==0
                ?light.spectralSource.directionalIrradiance.coefficients[band]*cosine:0))
                return fail("Metal native direct eight-band D cosine result changed");
        if(frame->colorImageHandle()||frame->depthImageHandle()||frame->colorTextureViewHandle()||frame->depthTextureViewHandle()
            ||!arbitgpu::inspectMetalGlassGratingCameraAllocationForFixture(frame,binding,error))return fail("Metal direct contribution exposed attachments or completed a private camera base");
        if(!arbitgpu::validateMetalDirectGratingSharedEntryForFixture(frame,receipt,error))return false;
        if(scenario==1) {
            // This edited clear-hit witness and D*cos bands satisfy the local
            // semantic formula. The real completed query still hit glass.
            auto positive=receipt;auto& changed=positive.samples[0];
            const auto work=changed.sourceHit.work;changed.sourceHit={};changed.sourceHit.status=0;
            changed.sourceHit.distance=binding->frame.sourceDirectionTrace[3];changed.sourceHit.work=work;
            changed.outcome=gc::NativeDirectGratingContributionSample::Outcome::Positive;
            for(unsigned band=0;band<8;++band)changed.surfaceIrradiance[band]=changed.source.input.coefficients[band]*cosine;
            if(arbitgpu::validateMetalDirectGratingSharedEntryForFixture(frame,positive,error)||error.empty()
                ||error!="Metal direct contribution differs from its actual retained native output")
                return fail("Metal direct shared validator accepted a semantically valid blocked-to-positive caller edit");
            if(!arbitgpu::validateMetalDirectGratingSharedEntryForFixture(frame,receipt,error))return false;
        }
        if(scenario)continue;
        auto signedZero=receipt;signedZero.samples[0].sourceHit.position[0]=-0.0f;
        if(arbitgpu::validateMetalDirectGratingSharedEntryForFixture(frame,signedZero,error)||error.empty())
            return fail("Metal direct shared validator lost exact native result float bits");
        const auto reject=[&](const auto& changed,const auto& expected,double time) {
            return !arbitgpu::validateMetalDirectGratingIrradiance(frame,expected,time,changed,error)&&!error.empty();
        };
        auto changed=receipt;++changed.endpointGeneration;if(!reject(changed,receipt.submission,.25))return fail("Metal accepted stale endpoint upload");
        changed=receipt;++changed.sourceGeneration;if(!reject(changed,receipt.submission,.25))return fail("Metal accepted stale source upload");
        changed=receipt;changed.submission.completedFence=0;if(!reject(changed,changed.submission,.25))return fail("Metal accepted incomplete direct fence");
        if(!reject(receipt,receipt.submission,.5))return fail("Metal accepted stale endpoint evaluation time");
        auto expected=receipt.submission;++expected.queryGeneration;if(!reject(receipt,expected,.25))return fail("Metal accepted stale native query generation");
        changed=receipt;changed.samples[0].sourceOriginMinimum[0]+=.125f;
        if(!reject(changed,receipt.submission,.25))return fail("Metal accepted an unexecuted source ray");
        auto sentinel=receipt;
        auto foreign=std::make_shared<gc::GratingIncidentBinding>(*binding);
        if(arbitgpu::produceMetalDirectGratingIrradiance(frame,foreign,endpoints,sentinel,error)||error.empty()
            ||!(sentinel.submission==receipt.submission))return fail("Metal accepted replacement endpoint binding ownership");
        const std::vector<gc::NativeDirectGratingEndpointInput> seam{inputAt(t.positions[0])};
        if(arbitgpu::produceMetalDirectGratingIrradiance(frame,binding,seam,sentinel,error)||error.empty()
            ||!(sentinel.submission==receipt.submission))return fail("Metal ambiguous native grating seam became completed zero");
        auto missed=endpoints;missed[0].directionMaximum={t.normal[0],t.normal[1],t.normal[2],1};
        if(arbitgpu::produceMetalDirectGratingIrradiance(frame,binding,missed,sentinel,error)||error.empty())
            return fail("Metal native endpoint miss became completed zero");
        if(arbitgpu::produceMetalDirectGratingFailedQueryForFixture(frame,binding,endpoints,sentinel,error)||error.empty()
            ||!(sentinel.submission==receipt.submission))return fail("Metal failed production query became completed zero");
        if(!arbitgpu::validateMetalDirectGratingIrradiance(frame,receipt.submission,.25,receipt,error))return false;
        gc::NativeDirectGratingContributionReadback repeated;
        if(!arbitgpu::produceMetalDirectGratingIrradiance(frame,binding,endpoints,repeated,error))return false;
        if(native->directEndpointUpload().otherWork<receipt.queryWork+8
            ||native->directEndpointUpload().otherOwnedBytes==0||repeated.submission.submissionSerial<=receipt.submission.submissionSerial
            ||!arbitgpu::metalcaustic::sameDirectGratingSample(receipt.samples[0],repeated.samples[0]))
            return fail("Metal repeated native direct contribution lost exact output or prior reservations");
        if(!reject(receipt,receipt.submission,.25))return fail("Metal stale direct batch acquired current ownership");
        if(!arbitgpu::inspectMetalDirectGratingRefusalsForFixture(frame,error))return false;
        if(!arbitgpu::inspectMetalDirectGratingAttemptBudgetForFixture(frame,endpoints,3,error))return false;
        if(!arbitgpu::validateMetalDirectGratingIrradiance(frame,repeated.submission,.25,repeated,error))return false;
    }
    error.clear();return true;
}

bool metalFiniteGratingSourceCases(std::string& error)
{
    namespace gc=arbitgpu::glasscaustics;namespace gt=glasscaustictests;namespace mc=arbitgpu::metalcaustic;
    const auto fail=[&](const char* message){error=message;return false;};
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> foreignFrame;
    std::shared_ptr<const arbitgpu::MetalFiniteGratingSourceResult> foreign;
    for(unsigned scenario=0;scenario<12;++scenario){
        auto fixture=gt::ordinaryFixture();auto& scene=*fixture.scene;auto& light=scene.lights[0];auto& target=scene.objects[1];
        light.transform.rotation=scenario==4?HarmonicMIDI::grid::SceneQuaternion{0,1,0,0}:HarmonicMIDI::grid::SceneQuaternion{0,0,0,1};
        light.spectralSource.radiance.fill(0);
        light.spectralSource.directionalIrradiance={1,diffractionmaterial::DirectionalSpectralMeasure::IrradianceCoefficient,
            {.25f,.5f,.75f,1,1.25f,1.5f,1.75f,2}};
        if(scenario==1)light.spectralSource.directionalIrradiance.coefficients.fill(0);
        const HarmonicMIDI::grid::SceneVec3 positions[]={{20,20,0},{22,20,0},{20,22,0},{22,22,0}};
        for(unsigned v=0;v<4;++v){auto& vertex=scene.vertices[target.firstVertex+v];vertex.position=positions[v];
            vertex.normal={0,0,1};vertex.tangent={1,0,0,1};
            vertex.uv={float(v&1u),float(v>>1u)};
            if(scenario==5)vertex.uv={float(v>>1u),1-float(v&1u)};
            if(scenario==6)vertex.uv.x*=2;
        }
        const unsigned indices[]{0,1,2,1,3,2};for(unsigned k=0;k<6;++k)scene.indices[target.firstIndex+k]=indices[k];
        if(scenario==9)scene.vertices[target.firstVertex+2].position.x+=.25f;
        if(scenario==11)target.transform.translation.x=.25f;
        auto glass=std::make_shared<gt::Program>(*fixture.program);glass->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;glass->objectPrograms.clear();
        auto description=diffractionmaterial::makeAluminiumBinaryGratingPreset();description.roughness={0,0};
        description.spectrum.firstOrder=description.spectrum.lastOrder=1;description.microstructure.grooveDepthNanometres=.9f;
        if(scenario==10)description.roughness.rmsSlope=.125f;
        const auto admitted=diffractionmaterial::admit(description,error);if(!admitted)return false;
        const auto material=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*admitted);
        diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest request;
        request.version=diffractionmaterialbinding::kSceneReceiverWireVersion;request.scene=scene.id;request.object=target.id;
        request.sceneRevision=request.structuralRevision=request.evaluationRevision=request.materialRevision=1;
        request.material=description;request.structuralDigest=material->structuralDigest();request.lighting.version=3;
        request.lighting.pathCount=1;request.lighting.traceMaximumDistance=20;
        request.lighting.sceneReceiverTransport=diffractionmaterial::SceneReceiverTransport::AuthoredSpectralLambertian;
        request.lighting.paths[0].kind=diffractionmaterial::LightingPathKind::Direct;
        request.lighting.paths[0].incident.direction={0,0,1};request.lighting.paths[0].incident.radiance.fill(1);
        const auto leaf=gc::compileDiscreteGrating(scene,request,arbitgpu::NativeFixtureMaterialBackend::Metal,error);
        if(scenario==10){if(leaf)return fail("Metal finite source rough material widened released compiler authority");continue;}
        if(!leaf)return false;
        auto root=std::make_shared<gt::Program>(*leaf);root->objectPrograms={glass};
        if(scenario==2||scenario==3){
            videohelper::tests::smoothglassfixture::receiver(scene,scenario==2?1e-5f:.5f,{0,0,0});
            auto& blocker=scene.objects[scene.objectCount-1];const float coordinates[4][2]={{20,20},{20.125f,20},{20.125f,20.125f},{20,20.125f}};
            for(unsigned v=0;v<4;++v)scene.vertices[blocker.firstVertex+v].position={coordinates[v][0],coordinates[v][1],scenario==2?1e-5f:.5f};
            auto program=std::make_shared<gt::Program>(*fixture.program->objectPrograms[0]);
            program->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;program->object=blocker.id;program->authoredMaterial=blocker.material;
            program->parameters.identifiers[0]=blocker.material.value;program->bindingDigest=program->programIdentity="metal-finite-source-competitor";
            root->objectPrograms.push_back(program);
        }
        if(scenario==8){scene.lights[1]=light;scene.lights[1].id={light.id.value+1};scene.lightCount=2;}
        const auto resources=arbitgpu::prepareMetalGlassGratingResources(fixture.scene,root,{target.id,material},leaf);
        if(!resources.prepared)return fail(resources.error.c_str());
        arbitgpu::NativeFixtureSceneRuntimeInputs runtime;runtime.linearColor=true;runtime.timeSeconds=.25f;
        gc::SchedulingDescription scheduling;scheduling.entrySubdivisions=1;
        std::vector<arbitgpu::opaquevisibility::Instance> instances;
        for(std::size_t o=0;o<scene.objectCount;++o){auto world=videohelper::tests::sceneclosest::identity();
            world[12]=scene.objects[o].transform.translation.x;world[13]=scene.objects[o].transform.translation.y;
            world[14]=scene.objects[o].transform.translation.z;instances.push_back({world,o});}
        std::vector<std::shared_ptr<const gc::GratingIncidentBinding>> bindings;
        for(unsigned source=0;source<scene.lightCount;++source){gc::Prepared prepared;gc::Frame schedule;
            std::shared_ptr<const gc::GratingIncidentBinding> binding;
            if(!arbitgpu::prepareMetalGlassCaustic(resources.resources,scene.lights[source].id,*fixture.policy,prepared,error)
                ||!gc::buildGratingStagingFrame(prepared,instances,&runtime,32,32,scheduling,schedule,error))return false;
            const bool bound=gc::prepareGratingIncidentBinding(prepared,schedule,leaf,material,.25,binding,error,gc::SourceMeasureUse::DirectionalIrradiancePreparation);
            if(!bound)return false;bindings.push_back(binding);
        }
        const auto& triangle=bindings[0]->triangles[0];gc::Point position{};
        const gc::Point bary{.5f,.25f,.25f};for(unsigned v=0;v<3;++v)position=gc::advance(position,triangle.positions[v],bary[v]);
        gc::NativeDirectGratingEndpointInput input;input.primitive={triangle.object,triangle.material,triangle.instance,triangle.triangle};
        const auto origin=gc::advance(position,triangle.normal,.25f);input.originMinimum={origin[0],origin[1],origin[2],gc::kRayMinimum};
        input.directionMaximum={-triangle.normal[0],-triangle.normal[1],-triangle.normal[2],1};
        std::vector<gc::NativeDirectGratingEndpointInput> inputs{input};if(scenario==7)inputs.push_back(input);
        std::shared_ptr<const gc::DirectGratingIrradianceCohort> cohort;
        if(!gc::prepareDirectGratingIrradianceCohort(bindings,inputs,0,0,cohort,error))return false;
        if(scenario==8){auto duplicate=bindings;duplicate[1]=duplicate[0];auto incumbent=cohort;
            if(gc::prepareDirectGratingIrradianceCohort(duplicate,inputs,0,0,incumbent,error)||incumbent!=cohort)
                return fail("Metal finite source duplicate source identity replaced the prepared cohort");}
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame;
        gc::DirectGratingIrradianceCohortReadback sum;std::vector<gc::NativeDirectGratingContributionReadback> sources;
        gc::DirectGratingTerminalOrderLedger terminal;
        if(!arbitgpu::stageMetalDirectGratingIrradianceCohort(resources.resources,cohort,runtime,frame,error)
            ||!arbitgpu::produceMetalDirectGratingIrradianceCohort(frame,cohort,sum,error)
            ||!arbitgpu::readMetalDirectGratingCohortForFixture(frame,sources,error)
            ||!arbitgpu::produceMetalDirectGratingTerminalOrders(frame,cohort,sum.submission,.25,sources,terminal,error))return false;
        std::shared_ptr<const arbitgpu::MetalFiniteGratingSourceResult> support;
        const auto operation=[&](int stop=-1,std::shared_ptr<const arbitgpu::MetalFiniteGratingSourceResult>* escaped=nullptr){
            return arbitgpu::produceMetalFiniteGratingSourceForFixture(frame,terminal.nativeResult,support,error,stop,escaped);};
        if(scenario>=2){
            if(operation()||support||frame->colorImageHandle()||frame->colorTextureViewHandle())return fail("Metal finite source unsupported scene, axes, UV, material or roster acquired support");
            if((scenario==2||scenario==3)&&error!="Metal finite source native lineage, arithmetic, resource or support changed")
                return fail("Metal finite source contact/blocker did not reach genuine completed native arithmetic");
            continue;
        }
        std::array<std::uintptr_t,4> buffers{};std::array<std::uint64_t,14> before{},after{};
        std::array<std::uint64_t,5> oldStorage{},newStorage{};
        const auto storage=[&](bool next){return arbitgpu::readMetalFiniteGratingSourceStorageForFixture(frame,buffers,
            next?after:before,next?newStorage:oldStorage,error);};
        if(scenario==1){support=foreign;if(!foreignFrame||!support||!storage(false)||operation()||support!=foreign
            ||error!="Metal finite source incumbent is foreign or unregistered"||!storage(true)||before!=after||oldStorage!=newStorage
            ||!arbitgpu::inspectMetalFiniteGratingSourceForFixture(foreignFrame,foreign,error))return fail("Metal finite source credited a foreign incumbent graph");support.reset();}
        for(int stop:{3,0,1,4}){if(!storage(false)||operation(stop)||support||!storage(true)
            ||newStorage[0]!=oldStorage[0]||newStorage[1]!=oldStorage[1]||newStorage[2]!=oldStorage[2]||newStorage[3]!=oldStorage[3]
            ||newStorage[4]!=oldStorage[4]+1||after[0]!=before[0]+1||after[1]!=before[1]+18*bindings[0]->frame.triangles.size()
            ||after[2]<=before[2]||after[3]<=before[3]||after[4]<=before[4]||after[11]<=before[11]||after[12]<=before[12])
                return fail("Metal finite source failed begin/reserve/dispatch/inspection lost permanent work or incumbent storage");}
        if(scenario==0)for(int diagnostic=20;diagnostic<=25;++diagnostic){
            if(operation(diagnostic)||support||error!="Metal finite source fixture completed native arithmetic refusal")
                return fail("Metal finite source invalid direct interval or binary64 precision/sign/exponent/overflow did not execute native refusal");}
        std::shared_ptr<const arbitgpu::MetalFiniteGratingSourceResult> escaped;
        if(!storage(false)||operation(2,&escaped)||!escaped||support||!storage(true)||newStorage[0]<=oldStorage[0]
            ||newStorage[1]<=oldStorage[1]||newStorage[2]<=oldStorage[2]||newStorage[3]<=oldStorage[3]
            ||arbitgpu::inspectMetalFiniteGratingSourceForFixture(frame,escaped,error))return fail("Metal finite source escaped unpublished owner lost its native reservation");
        support=escaped;
        if(!storage(false)||operation(0)||support!=escaped||!storage(true)||newStorage[0]!=oldStorage[0]
            ||newStorage[1]!=oldStorage[1]||newStorage[2]!=oldStorage[2]||newStorage[3]!=oldStorage[3]
            ||after[0]!=before[0]+1||!operation()||support==escaped||!arbitgpu::inspectMetalFiniteGratingSourceForFixture(frame,support,error))
            return fail("Metal finite source successful retry changed escaped credit or failed to publish one support");
        std::array<std::uint32_t,arbitgpu::kMetalFiniteSourceDiagnosticWords> words{};
        if(!arbitgpu::readMetalFiniteGratingSourceWordsForFixture(frame,support,words,error))return false;
        const auto requestWords=mc::finiteSourceRequestWords(bindings[0]->frame.triangles.size(),terminal.orders.size());
        const auto area=8+requestWords;
        // Authored two-metre legs give native exact area2 square metres.
        if(words[area]!=1||words[area+1]!=1||words[area+2]!=1
            ||std::any_of(words.begin()+area+3,words.begin()+area+10,[](auto value){return value!=0;}))
            return fail("Metal finite source genuine native area differs from two square metres");
        for(unsigned band=0;band<8;++band){const auto echo=8+mc::kFiniteSourceHeaderWords+band*mc::kFiniteSourceBandWords+38;
            std::uint64_t density=0;std::memcpy(&density,&terminal.bands[band].incidentIrradiance,8);
            if(words[echo]!=std::uint32_t(density)||words[echo+1]!=std::uint32_t(density>>32))return fail("Metal finite source rounded retained incoming density bits");
            const auto power=area+70+band*10;
            if(scenario==1&&std::any_of(words.begin()+power,words.begin()+power+10,[](auto value){return value!=0;}))
                return fail("Metal finite source zero-D acquired incoming integrated power");}
        for(unsigned order=0;order<terminal.orders.size();++order){const auto echo=8+mc::kFiniteSourceHeaderWords+8*mc::kFiniteSourceBandWords+order*mc::kFiniteSourceOrderWords;
            std::uint64_t density=0;std::memcpy(&density,&terminal.orders[order].orderIrradiance,8);
            if(words[echo+3]!=std::uint32_t(terminal.orders[order].order.primary)||words[echo+4]!=std::uint32_t(terminal.orders[order].order.secondary)
                ||words[echo+6]!=std::uint32_t(density)||words[echo+7]!=std::uint32_t(density>>32))
                return fail("Metal finite source changed signed CPU order identity or binary64 density bits");
            const auto power=area+150+order*10;
            if(scenario==1&&std::any_of(words.begin()+power,words.begin()+power+10,[](auto value){return value!=0;}))
                return fail("Metal finite source zero-D acquired signed-order integrated power");}
        const auto incumbent=support;
        if(!storage(false)||operation()||support!=incumbent||error!="Metal finite source support was already consumed on this frame"
            ||!storage(true)||newStorage[0]!=oldStorage[0]||newStorage[1]!=oldStorage[1]||newStorage[2]!=oldStorage[2]||newStorage[3]!=oldStorage[3]
            ||after[0]!=before[0]+1||!arbitgpu::inspectMetalFiniteGratingSourceRefusalsForFixture(frame,support,error)
            ||!arbitgpu::inspectMetalFiniteGratingSourceForFixture(frame,support,error))return fail("Metal finite source double consumption or resource/provenance mutation was accepted");
        if(frame->colorImageHandle()||frame->depthImageHandle()||frame->colorTextureViewHandle()||frame->depthTextureViewHandle())
            return fail("Metal finite source support granted receiver radiance or public Color");
        if(scenario==0){foreignFrame=frame;foreign=support;}
    }
    error.clear();return true;
}

// Authenticated private arithmetic only. The existing frame remains the owner;
// no receiver support is consumed and no Image/Color is produced.
bool metalLambertianNativeCases(const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame,
    std::shared_ptr<const arbitgpu::MetalLambertianArithmeticResult>& foreign,std::string& error)
{
    namespace lr=arbitgpu::lambertian;namespace nt=lr::native;namespace lt=arbitgpu::lambertiantests;
    const auto fail=[&](const char* message){error=message;return false;};
    nt::Costs before{},after{};nt::Storage prior{},next{};
    const auto storage=[&](bool newer){return arbitgpu::readMetalLambertianStorageForFixture(frame,
        newer?after:before,newer?next:prior,error);};
    std::shared_ptr<const arbitgpu::MetalLambertianArithmeticResult> value;
    // Prepare every fixture arithmetic/source/caller capacity before building
    // requests or invoking the allocating legacy comparison builder.
    if(!arbitgpu::admitMetalLambertianPreparedForFixture(frame,lt::kPreparedChecksCeilings,error))return false;
    nt::Source source;
    if(!nt::assemble(source))return fail("Metal Lambertian bounded fixture source assembly failed");
    const auto legacy=lr::metalArithmeticSource();
    if(legacy.capacity()>2*nt::kSourceCapacity||legacy.size()!=source.arithmeticBytes
        ||std::memcmp(legacy.data(),source.bytes.data()+source.arithmeticOffset,legacy.size()))
        return fail("Metal Lambertian complete legacy arithmetic source changed");
    const auto q=lt::request(lr::kP32Significand,-22);lr::Result sentinel{};
    if(foreign){value=foreign;
        if(!storage(false)||arbitgpu::produceMetalLambertianArithmeticForFixture(frame,q,sentinel,value,error)
            ||value!=foreign||!storage(true)||next.attempts!=prior.attempts||next.frameBytes!=prior.frameBytes
            ||next.attemptBytes!=prior.attemptBytes||next.nativeBytes!=prior.nativeBytes
            ||next.publicInspections<=prior.publicInspections)return fail("Metal Lambertian foreign incumbent received allocation credit");
        return true;
    }
    if(!storage(false)||prior.identityBytes<2ull*512*256*4)
        return fail("Metal Lambertian owning scene did not retain its large texture identity");
    // Independent record-write census and actual owning-frame identity length.
    // Counter deltas include storage probes and every nested registry journal.
    constexpr auto cap=arbitgpu::glasscaustics::kMaximumEntrySamples;
    constexpr auto scanCopies=3*sizeof(lr::Result)+(8*cap+20)*sizeof(nt::Costs)+512;
    constexpr auto inspectionCopies=scanCopies+2*sizeof(lr::Result);
    const auto attemptCopies=scanCopies+1104+6*nt::kSourceCapacity
        +8*(sizeof(lr::Request)+sizeof(lr::Result))+2000+2*(prior.identityBytes+1);
    const auto produce=[&](int stop=-1,std::shared_ptr<const arbitgpu::MetalLambertianArithmeticResult>* escaped=nullptr){
        return arbitgpu::produceMetalLambertianArithmeticForFixture(frame,q,sentinel,value,error,stop,escaped);};
    for(int stop:{3,0,10,11,12,13,5,1,4}){
        if(!storage(false)||produce(stop)||value||!storage(true)||next.attempts!=prior.attempts+1
            ||next.frameBytes!=prior.frameBytes||next.attemptBytes!=prior.attemptBytes||next.nativeBytes!=prior.nativeBytes
            ||after[nt::NativeVisits]-before[nt::NativeVisits]!=lr::kNativeCeilings.limbVisits
            ||after[nt::NativeMoves]-before[nt::NativeMoves]!=lr::kNativeCeilings.wordMoves
            ||next.identityBytes!=prior.identityBytes
            ||next.arithmeticInspections-prior.arithmeticInspections!=(stop==4?1ull:0ull)
            ||after[nt::CopiedBytes]-before[nt::CopiedBytes]!=attemptCopies
                +(next.publicInspections-prior.publicInspections+next.registryInspections-prior.registryInspections)*scanCopies
                +(next.arithmeticInspections-prior.arithmeticInspections)*inspectionCopies
            ||(stop==4&&after[nt::HostVisits]-before[nt::HostVisits]!=lr::kHostCeilings.limbVisits))
            return fail("Metal Lambertian failed native phase lost its permanent journal or retry capacity");
    }
    std::shared_ptr<const arbitgpu::MetalLambertianArithmeticResult> escaped;
    lr::Result actual{};bool evaluated=false;
    if(!storage(false)||produce(2,&escaped)||!escaped||value||!storage(true)
        ||next.attemptBytes<=prior.attemptBytes||next.nativeBytes<=prior.nativeBytes
        ||arbitgpu::readMetalLambertianArithmeticForFixture(frame,escaped,actual,evaluated,error))
        return fail("Metal Lambertian escaped unpublished stage lost exact native ownership");
    value=escaped;
    if(!storage(false)||produce(0)||value!=escaped||!storage(true)||next.attemptBytes!=prior.attemptBytes
        ||next.nativeBytes!=prior.nativeBytes||!produce()||value==escaped
        ||!arbitgpu::readMetalLambertianArithmeticForFixture(frame,value,actual,evaluated,error)||!evaluated
        ||actual[2]!=0x3f800000u)return fail("Metal Lambertian registered escaped incumbent could not retry safely");
    if(!storage(false))return false;
    const auto dispatch=[&](const lr::Request& request,lr::Result& output){
        if(!arbitgpu::produceMetalLambertianArithmeticForFixture(frame,request,output,value,error))return false;
        bool nativeReturn=false;
        if(!arbitgpu::readMetalLambertianArithmeticForFixture(frame,value,output,nativeReturn,error))return false;
        return nativeReturn;
    };
    if(!lt::runPrepared(dispatch,lt::kPreparedChecksCeilings)||!storage(true)
        ||next.attempts-prior.attempts!=57
        ||after[nt::NativeVisits]-before[nt::NativeVisits]!=57*lr::kNativeCeilings.limbVisits
        ||after[nt::NativeMoves]-before[nt::NativeMoves]!=57*lr::kNativeCeilings.wordMoves)
        return fail("Metal Lambertian prepared arithmetic did not use 57 authenticated native attempts");
    if(!arbitgpu::inspectMetalLambertianArithmeticRefusalsForFixture(frame,value,error)
        ||!arbitgpu::readMetalLambertianArithmeticForFixture(frame,value,actual,evaluated,error))
        return fail("Metal Lambertian resource/provenance mutation or permanent inspector journal was accepted");
    if(frame->colorImageHandle()||frame->colorTextureViewHandle())return fail("Metal Lambertian private arithmetic acquired Color");
    foreign=value;error.clear();return true;
}

bool metalFiniteGratingReceiverCases(std::string& error)
{
    namespace gc=arbitgpu::glasscaustics;namespace gt=glasscaustictests;namespace mc=arbitgpu::metalcaustic;
    const auto fail=[&](const char* message){error=message;return false;};
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> foreignFrame;
    std::shared_ptr<const arbitgpu::MetalFiniteGratingReceiverResult> foreign;
    std::shared_ptr<const arbitgpu::MetalLambertianArithmeticResult> foreignArithmetic;
    std::shared_ptr<const arbitgpu::MetalRestrictedCompleteBaseResult> foreignBase;
    std::shared_ptr<const arbitgpu::MetalFiniteGratingRadianceResult> foreignRadiance;
    std::shared_ptr<const arbitgpu::MetalFiniteGratingRgbResult> foreignRgb;
    std::array<float,4> baselineRgb{};
    std::array<std::uint32_t,8> baselineRadiance{};
    std::array<std::uint32_t,arbitgpu::kMetalFiniteReceiverDiagnosticWords> baselineReceiver{};
    for(unsigned scenario=0;scenario<8;++scenario){
        auto fixture=gt::ordinaryFixture();auto& scene=*fixture.scene;auto& light=scene.lights[0];auto& target=scene.objects[1];
        light.transform.rotation=HarmonicMIDI::grid::SceneQuaternion{0,0,0,1};
        light.spectralSource.radiance.fill(0);light.spectralSource.traceMaximumDistance=1;
        light.spectralSource.directionalIrradiance={1,diffractionmaterial::DirectionalSpectralMeasure::IrradianceCoefficient,
            {.25f,.5f,.75f,1,1.25f,1.5f,1.75f,2}};
        if(scenario==1)light.spectralSource.directionalIrradiance.coefficients.fill(0);
        if(scenario==3)for(auto& value:light.spectralSource.directionalIrradiance.coefficients)value*=2;
        if(scenario==6)light.spectralSource.directionalIrradiance.coefficients.fill(std::ldexp(1.0f,-125));
        const HarmonicMIDI::grid::SceneVec3 positions[]={{-1,-1,0},{1,-1,0},{-1,1,0},{1,1,0}};
        for(unsigned v=0;v<4;++v){auto& vertex=scene.vertices[target.firstVertex+v];vertex.position=positions[v];
            vertex.normal={0,0,1};vertex.tangent={1,0,0,1};
            vertex.uv={float(v&1u),float(v>>1u)};
        }
        const unsigned indices[]{0,1,2,1,3,2};for(unsigned k=0;k<6;++k)scene.indices[target.firstIndex+k]=indices[k];
        const auto& glassObject=scene.objects[0];
        for(unsigned v=0;v<glassObject.vertexCount;++v){auto& vertex=scene.vertices[glassObject.firstVertex+v];vertex.position.x+=20;vertex.position.y+=20;}
        auto& camera=scene.cameras[0];camera.transform={};camera.transform.translation={-.25f,-.25f,1};
        camera.transform.rotation={0,1,0,0};camera.verticalFovRadians=.1f;camera.nearPlane=.01f;camera.farPlane=100;
        if(scenario==5)camera.verticalFovRadians=.2f;
        const auto wallIndex=scene.objectCount;videohelper::tests::smoothglassfixture::receiver(scene,2,{0,0,0});
        auto& wall=scene.objects[wallIndex];
        const HarmonicMIDI::grid::SceneVec3 corners[]={{-2,-2,2},{2,-2,2},{2,2,2},{-2,2,2}};
        for(unsigned v=0;v<4;++v){auto& vertex=scene.vertices[wall.firstVertex+v];vertex.position=corners[v];vertex.normal={0,0,-1};vertex.tangent={1,0,0,1};}
        const unsigned wallIndices[]{0,2,1,0,3,2};for(unsigned k=0;k<6;++k)scene.indices[wall.firstIndex+k]=wallIndices[k];
        auto receiverProgram=std::make_shared<gt::Program>(*fixture.program->objectPrograms[0]);
        receiverProgram->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;receiverProgram->object=wall.id;
        receiverProgram->authoredMaterial=wall.material;receiverProgram->parameters.identifiers[0]=wall.material.value;
        receiverProgram->bindingDigest=receiverProgram->programIdentity="metal-finite-receiver";
        if(scenario==2)receiverProgram->parameters.emissionRoughness[0]=1;
        if(scenario==4)for(auto& value:receiverProgram->spectralReflectance->coefficients)value*=.5f;
        auto glass=std::make_shared<gt::Program>(*fixture.program);glass->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;glass->objectPrograms.clear();
        auto description=diffractionmaterial::makeAluminiumBinaryGratingPreset();description.roughness={0,0};
        description.spectrum.firstOrder=description.spectrum.lastOrder=1;description.microstructure.grooveDepthNanometres=1;
        description.geometry.grooveSpacingNanometres=100;description.microstructure.dutyCycle=.5f;
        description.substrate={1.5f,0};description.coating={diffractionmaterial::CoatingModel::Uncoated,0,{0,0}};
        const auto admitted=diffractionmaterial::admit(description,error);if(!admitted)return false;
        const auto material=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*admitted);
        diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest request;
        request.version=diffractionmaterialbinding::kSceneReceiverWireVersion;request.scene=scene.id;request.object=target.id;
        request.sceneRevision=request.structuralRevision=request.evaluationRevision=request.materialRevision=1;
        request.material=description;request.structuralDigest=material->structuralDigest();request.lighting.version=3;
        request.lighting.pathCount=1;request.lighting.traceMaximumDistance=1;
        request.lighting.sceneReceiverTransport=diffractionmaterial::SceneReceiverTransport::AuthoredSpectralLambertian;
        request.lighting.paths[0].kind=diffractionmaterial::LightingPathKind::Direct;
        request.lighting.paths[0].incident.direction={0,0,1};request.lighting.paths[0].incident.radiance.fill(1);
        // Lambertian accounting fixture: an unused, valid owning-scene
        // texture contributes over one MiB of hex identity through production.
        if(scenario==0){
            auto& texture=scene.textures[scene.textureCount++];
            texture.id={918273};texture.firstTexel=std::uint32_t(scene.textureTexelCount);
            texture.width=512;texture.height=256;scene.textureTexelCount+=512*256;
            scene.textureTexels.resize(scene.textureTexelCount,{17,31,47,255});
        }
        const auto leaf=gc::compileDiscreteGrating(scene,request,arbitgpu::NativeFixtureMaterialBackend::Metal,error);
        if(!leaf)return false;
        auto root=std::make_shared<gt::Program>(*leaf);root->objectPrograms={glass,receiverProgram};
        std::sort(root->objectPrograms.begin(),root->objectPrograms.end(),[](const auto& x,const auto& y){return x->object.value<y->object.value;});
        const auto resources=arbitgpu::prepareMetalGlassGratingResources(fixture.scene,root,{target.id,material},leaf);
        if(scenario==2){gc::Prepared refused;
            if(resources.prepared&&arbitgpu::prepareMetalGlassCaustic(resources.resources,light.id,*fixture.policy,refused,error))
                return fail("Metal finite receiver admitted authored emission");continue;}
        if(!resources.prepared)return fail(resources.error.c_str());
        if(!arbitgpu::prepareMetalPrivateSurfaceSourceForFixture(resources.resources,wall.id,error))return false;
        arbitgpu::NativeFixtureSceneRuntimeInputs runtime;runtime.linearColor=true;runtime.timeSeconds=.25f;
        gc::SchedulingDescription scheduling;scheduling.entrySubdivisions=1;
        std::vector<arbitgpu::opaquevisibility::Instance> instances;
        for(std::size_t o=0;o<scene.objectCount;++o){auto world=videohelper::tests::sceneclosest::identity();
            world[12]=scene.objects[o].transform.translation.x;world[13]=scene.objects[o].transform.translation.y;
            world[14]=scene.objects[o].transform.translation.z;instances.push_back({world,o});}
        std::vector<std::shared_ptr<const gc::GratingIncidentBinding>> bindings;
        for(unsigned source=0;source<scene.lightCount;++source){gc::Prepared prepared;gc::Frame schedule;
            std::shared_ptr<const gc::GratingIncidentBinding> binding;
            if(!arbitgpu::prepareMetalGlassCaustic(resources.resources,scene.lights[source].id,*fixture.policy,prepared,error)
                ||!gc::buildGratingStagingFrame(prepared,instances,&runtime,1,1,scheduling,schedule,error))return false;
            const bool bound=gc::prepareGratingIncidentBinding(prepared,schedule,leaf,material,.25,binding,error,gc::SourceMeasureUse::DirectionalIrradiancePreparation);
            if(!bound)return false;
            gc::SourceInputs frozenSource;
            if(!gc::sourceInputs(binding->prepared,binding->frame.activeSource,frozenSource,error))return false;
            if(frozenSource.maximum!=1||binding->frame.sourceDirectionTrace[3]!=1)
                return fail("Metal finite receiver frozen source interval differs from authored one metre");
            bindings.push_back(binding);
        }
        const auto& triangle=bindings[0]->triangles[0];gc::Point position{};
        const gc::Point bary{.25f,.375f,.375f};for(unsigned v=0;v<3;++v)position=gc::advance(position,triangle.positions[v],bary[v]);
        gc::NativeDirectGratingEndpointInput input;input.primitive={triangle.object,triangle.material,triangle.instance,triangle.triangle};
        const auto origin=gc::advance(position,triangle.normal,.25f);input.originMinimum={origin[0],origin[1],origin[2],gc::kRayMinimum};
        input.directionMaximum={-triangle.normal[0],-triangle.normal[1],-triangle.normal[2],1};
        std::vector<gc::NativeDirectGratingEndpointInput> inputs{input};
        std::shared_ptr<const gc::DirectGratingIrradianceCohort> cohort;
        if(!gc::prepareDirectGratingIrradianceCohort(bindings,inputs,0,0,cohort,error))return false;
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame;
        gc::DirectGratingIrradianceCohortReadback sum;std::vector<gc::NativeDirectGratingContributionReadback> sources;
        gc::DirectGratingTerminalOrderLedger terminal;
        if(!arbitgpu::stageMetalDirectGratingIrradianceCohort(resources.resources,cohort,runtime,frame,error)
            ||!arbitgpu::produceMetalDirectGratingIrradianceCohort(frame,cohort,sum,error)
            ||!arbitgpu::readMetalDirectGratingCohortForFixture(frame,sources,error)
            ||!arbitgpu::produceMetalDirectGratingTerminalOrders(frame,cohort,sum.submission,.25,sources,terminal,error))return false;
        if(sources.size()!=bindings.size())return fail("Metal finite receiver native source contribution count differs from bindings");
        for(const auto& contribution:sources){
            if(!contribution.complete||contribution.samples.size()!=inputs.size())
                return fail("Metal finite receiver native source contribution is incomplete");
            for(const auto& sample:contribution.samples)
                if(!sample.complete||sample.source.directionTrace[3]!=1||sample.sourceDirectionMaximum[3]!=1)
                    return fail("Metal finite receiver completed native source interval differs from authored one metre");
        }
        if(terminal.orders.size()!=8||terminal.bands.size()!=8)return fail("Metal finite receiver witness lacks eight physical zero orders");
        for(unsigned band=0;band<8;++band){const auto& row=terminal.orders[band];
            if(row.source||row.endpoint||row.band!=band||row.order.primary||row.order.secondary
                ||row.order.direction!=std::array<double,3>{0,0,1}||terminal.bands[band].firstOrder!=band
                ||terminal.bands[band].orderCount!=1||(scenario==0&&!(row.orderIrradiance>0))
                ||(scenario==1&&row.orderIrradiance!=0))return fail("Metal finite receiver exact positive/zero retained order prerequisite failed");}
        gc::DirectGratingTerminalReceiverReadback emitted;gc::CameraCellCoverageReadback coverage;
        std::shared_ptr<const arbitgpu::MetalFiniteGratingSourceResult> source;
        std::shared_ptr<const arbitgpu::MetalCameraGeometricRangeResult> range;
        gc::NativeGlassCameraGratingTraceInput traceInput;gc::NativeGlassCameraGratingTraceReadback trace;
        if(!arbitgpu::produceMetalDirectTerminalReceivers(frame,terminal.nativeResult,sum.submission,.25,emitted,error)
            ||emitted.rows.size()!=8||!arbitgpu::produceMetalCameraCellCoverageForFixture(frame,emitted.nativeResult,sum.submission,.25,coverage,error)
            ||coverage.preparation.pixels.size()!=1||!gc::makeGlassCameraGratingTraceInput(bindings[0]->frame,0,0,0,traceInput,error)
            ||!arbitgpu::produceMetalGlassCameraGratingTrace(frame,bindings[0],{traceInput},trace,error)
            ||!arbitgpu::produceMetalCameraGeometricRangeForFixture(frame,coverage.nativeResult,range,error)
            ||!arbitgpu::produceMetalFiniteGratingSourceForFixture(frame,terminal.nativeResult,source,error))return false;
        std::shared_ptr<const arbitgpu::MetalFiniteGratingReceiverResult> receiver;
        const auto operation=[&](int stop=-1,std::shared_ptr<const arbitgpu::MetalFiniteGratingReceiverResult>* escape=nullptr){
            return arbitgpu::produceMetalFiniteGratingReceiverForFixture(frame,source,range,receiver,error,stop,escape);};
        std::array<std::uintptr_t,4> buffers{};std::array<std::uint64_t,14> before{},after{};
        std::array<std::uint64_t,5> prior{},next{};
        const auto storage=[&](bool newer){return arbitgpu::readMetalFiniteGratingReceiverStorageForFixture(frame,buffers,newer?after:before,newer?next:prior,error);};
        if(scenario==1){receiver=foreign;
            if(!storage(false)||operation()||receiver!=foreign||error!="Metal finite receiver incumbent is foreign or unregistered"
                ||!storage(true)||!std::equal(before.begin(),before.begin()+5,after.begin())||prior!=next||!arbitgpu::inspectMetalFiniteGratingReceiverForFixture(foreignFrame,foreign,error))
                return fail("Metal finite receiver credited a foreign incumbent");receiver.reset();}
        for(int stop:{3,0,1,4}){
            if(!storage(false)||operation(stop)||receiver||!storage(true)||next[4]!=prior[4]+1
                ||next[0]!=prior[0]||next[1]!=prior[1]||next[2]!=prior[2]||next[3]!=prior[3]
                ||after[0]!=before[0]+1||after[1]!=before[1]+15*bindings[0]->frame.triangles.size()+27
                ||after[2]<=before[2]||after[3]<=before[3]||after[4]<=before[4]
                ||after[5]-before[5]!=(stop==4?2ull:1ull)*mc::finiteReceiverVisits(bindings[0]->frame.triangles.size()))
                return fail("Metal finite receiver failed phase lost admitted work or unconsumed retry storage");}
        if(scenario==0)for(int diagnostic=20;diagnostic<=38;++diagnostic){
            if(operation(diagnostic)||receiver||error!="Metal finite receiver fixture completed native arithmetic refusal")
                return fail("Metal finite receiver diagnostic did not reach its native proof refusal");}
        if(scenario==0&&(operation(39)||receiver||error!="Metal finite receiver fixture rejected coherent reflectance forgery"))
            return fail("Metal finite receiver forged rho did not reach the authored reflectance join");
        std::shared_ptr<const arbitgpu::MetalFiniteGratingReceiverResult> escaped;
        if(!storage(false)||operation(2,&escaped)||!escaped||receiver||!storage(true)||next[0]<=prior[0]||next[3]<=prior[3]
            ||arbitgpu::inspectMetalFiniteGratingReceiverForFixture(frame,escaped,error))
            return fail("Metal finite receiver escaped unpublished owner lost its reservation");
        receiver=escaped;
        if(!storage(false)||operation(0)||receiver!=escaped||!storage(true)||next[0]!=prior[0]||next[3]!=prior[3]
            ||!operation()||receiver==escaped||!arbitgpu::inspectMetalFiniteGratingReceiverForFixture(frame,receiver,error))
            return fail("Metal finite receiver safe retry failed or consumed escaped staging");
        std::array<std::uint32_t,arbitgpu::kMetalFiniteReceiverDiagnosticWords> words{};
        if(!arbitgpu::readMetalFiniteGratingReceiverWordsForFixture(frame,receiver,words,error))return false;
        const auto q=mc::finiteReceiverRequestWords(bindings[0]->frame.triangles.size()),base=8+q;
        if(words[7]!=0x504931u||words[base]!=1||words[base+1]!=1||words[base+2]!=1)
            return fail("Metal finite receiver lost exact source area or symbolic pi relation");
        for(unsigned band=0;band<8;++band){std::uint64_t bits=0;std::memcpy(&bits,&terminal.orders[band].orderIrradiance,8);
            if(words[8+174+2*band]!=std::uint32_t(bits)||words[8+175+2*band]!=std::uint32_t(bits>>32))
                return fail("Metal finite receiver rounded binary64 order density");
            const auto powers=base+40+70*band;
            if(scenario==1&&std::any_of(words.begin()+powers,words.begin()+powers+70,[](auto v){return v!=0;}))
                return fail("Metal finite receiver zero energy gained power, irradiance or exitance");
        }
        const auto incumbent=receiver;
        if(!storage(false)||operation()||receiver!=incumbent||error!="Metal finite receiver support was already consumed on this frame"
            ||!storage(true)||next[0]!=prior[0]||next[1]!=prior[1]||next[2]!=prior[2]||next[3]!=prior[3]
            ||!arbitgpu::inspectMetalFiniteGratingReceiverRefusalsForFixture(frame,receiver,error)
            ||!arbitgpu::inspectMetalFiniteGratingReceiverForFixture(frame,receiver,error)
            ||!arbitgpu::inspectMetalFiniteGratingSourceForFixture(frame,source,error))return fail("Metal finite receiver duplicate or provenance/resource mutation changed incumbent support");
        if(frame->colorImageHandle()||frame->depthImageHandle()||frame->colorTextureViewHandle()||frame->depthTextureViewHandle())
            return fail("Metal finite receiver acquired public Color or publication");
        if(scenario==0&&!videohelper::tests::restrictedsurfacesource::nativeCompleteBaseCases(frame,receiver,error))return false;
        if(!metalLambertianNativeCases(frame,foreignArithmetic,error))return false;
        if(scenario>=6){
            std::shared_ptr<const arbitgpu::MetalRestrictedCompleteBaseResult> baseOwner;
            std::shared_ptr<const arbitgpu::MetalFiniteGratingRadianceResult> refused;
            if(!arbitgpu::produceMetalRestrictedCompleteBaseForFixture(frame,receiver,baseOwner,error))return false;
            std::array<std::uint64_t,8> radianceBefore{},radianceAfter{};arbitgpu::finiteradiance::Costs wrapperBefore{},wrapperAfter{};
            const auto wrapperStorage=[&](bool newer){return arbitgpu::readMetalFiniteGratingRadianceStorageForFixture(frame,
                newer?radianceAfter:radianceBefore,newer?wrapperAfter:wrapperBefore,error);};
            if(scenario==7){
                // Real failed arithmetic attempts consume the existing finite
                // journal. No count mutation or fictitious child registry is used.
                arbitgpu::lambertian::Request q{};q[0]=1;q[1]=arbitgpu::lambertian::kP32Bits;q[2]=128;q[3]=640;q[5]=1;
                arbitgpu::lambertian::putBudgets(q,arbitgpu::lambertian::kNativeCeilings);
                arbitgpu::lambertian::Result sentinel{};std::shared_ptr<const arbitgpu::MetalLambertianArithmeticResult> numeric;
                if(!wrapperStorage(false))return false;
                for(auto count=radianceBefore[5];count<gc::kMaximumEntrySamples-7;++count)
                    if(arbitgpu::produceMetalLambertianArithmeticForFixture(frame,q,sentinel,numeric,error,3)||numeric)return false;
                if(!wrapperStorage(false)||radianceBefore[5]!=gc::kMaximumEntrySamples-7
                    ||arbitgpu::produceMetalFiniteGratingRadianceForFixture(frame,receiver,baseOwner,refused,error)
                    ||refused||!wrapperStorage(true)||radianceAfter[5]!=radianceBefore[5]||radianceAfter[1]||radianceAfter[4])
                    return fail("Finite radiance eight-slot preflight ignored actual previously failed native attempts");
                for(unsigned remaining=0;remaining<7;++remaining)
                    if(arbitgpu::produceMetalLambertianArithmeticForFixture(frame,q,sentinel,numeric,error,3)||numeric)return false;
                if(!wrapperStorage(false)||radianceBefore[5]!=gc::kMaximumEntrySamples
                    ||arbitgpu::produceMetalLambertianArithmeticForFixture(frame,q,sentinel,numeric,error,3)
                    ||arbitgpu::produceMetalFiniteGratingRadianceForFixture(frame,receiver,baseOwner,refused,error)
                    ||!wrapperStorage(true)||radianceAfter[5]!=gc::kMaximumEntrySamples)
                    return fail("Finite radiance or numerical child enlarged the 2048-attempt cap");
            }else {
                for(unsigned retry=0;retry<2;++retry){
                    if(!wrapperStorage(false)||arbitgpu::produceMetalFiniteGratingRadianceForFixture(frame,receiver,baseOwner,refused,error)
                        ||refused||!wrapperStorage(true)||radianceAfter[5]!=radianceBefore[5]+1||radianceAfter[4]
                        ||radianceAfter[3]<=radianceBefore[3])return fail("Finite radiance underflow gained a sample or refunded the authentic refusing child");
                }
            }
            continue;
        }
        std::array<std::uint32_t,8> radiance{};std::array<float,4> rgb{};
        if(!videohelper::tests::restrictedsurfacesource::nativeFiniteRadianceCases(frame,receiver,scenario==1,scenario<=1,
            foreignFrame,foreign,foreignBase,foreignRadiance,radiance,foreignRgb,rgb,error))return false;
        if(scenario==0){foreignFrame=frame;foreign=receiver;baselineRadiance=radiance;baselineReceiver=words;baselineRgb=rgb;}
        if(scenario>=3)for(unsigned channel=0;channel<3;++channel){
            const auto factor=scenario==3?2.0f:scenario==4?.5f:1.0f;
            if(std::abs(rgb[channel]-baselineRgb[channel]*factor)>2e-5f*std::max(1.0f,std::abs(baselineRgb[channel]*factor)))
                return fail("Native RGB source/rho/cell response differs from independently authenticated M scaling");}
        if(scenario>=3)for(unsigned band=0;band<8;++band){
            const auto expected=scenario==3?baselineRadiance[band]+0x00800000u:scenario==4?baselineRadiance[band]-0x00800000u:baselineRadiance[band];
            if(radiance[band]!=expected)return fail("Finite radiance source/rho/cell causal response differs from the exact native normal scaling");
            const auto at=base+100+70*band;
            if(scenario==5&&(!std::equal(words.begin()+at,words.begin()+at+10,baselineReceiver.begin()+at)
                ||std::equal(words.begin()+base+10,words.begin()+base+20,baselineReceiver.begin()+base+10)
                ||std::equal(words.begin()+base+50+70*band,words.begin()+base+60+70*band,baselineReceiver.begin()+base+50+70*band)))
                return fail("Finite radiance cell-area change altered M or failed to change exact cell power");
        }
    }
    error.clear();return true;
}

bool metalCameraDiagonalCoverageCases(std::string& error)
{
    namespace gc=arbitgpu::glasscaustics;namespace gt=glasscaustictests;
    const auto fail=[&](const char* message){error=message;return false;};
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> foreignRangeFrame;
    std::shared_ptr<const arbitgpu::MetalCameraGeometricRangeResult> foreignRange;
    for(unsigned variant=0;variant<10;++variant) {
        auto fixture=gt::ordinaryFixture();auto& scene=*fixture.scene;auto& light=scene.lights[0];
        auto& target=scene.objects[1];const auto targetId=target.id;
        const float half=float(std::sqrt(.5));
        for(std::size_t v=target.firstVertex;v<target.firstVertex+target.vertexCount;++v) {
            auto& vertex=scene.vertices[v];vertex.position.z=vertex.position.x-6;
            vertex.normal={-half,0,half};vertex.tangent={0,1,0,1};vertex.uv={vertex.position.y/12,-vertex.position.x/12};
        }
        // Rotate authored vertices, preserving identity instance transforms.
        // The actual camera is the captured native identity camera below.
        for(std::size_t v=0;v<scene.vertexCount;++v) {
            auto& vertex=scene.vertices[v];const auto p=vertex.position,n=vertex.normal;const auto t=vertex.tangent;
            vertex.position={-p.z,p.y,p.x};vertex.normal={-n.z,n.y,n.x};vertex.tangent={-t[2],t[1],t[0],t[3]};
        }
        light.transform.rotation={0,-half,0,half};light.spectralSource.radiance.fill(0);
        light.spectralSource.directionalIrradiance={1,diffractionmaterial::DirectionalSpectralMeasure::IrradianceCoefficient,{}};
        light.spectralSource.directionalIrradiance.coefficients[0]=variant==6?0:.25f;
        const auto receiverIndex=scene.objectCount;
        videohelper::tests::smoothglassfixture::receiver(scene,-4,{0,0,0});auto& wall=scene.objects[receiverIndex];
        const HarmonicMIDI::grid::SceneVec3 corners[]={{-5,7.75f,-4},{-5,-8.25f,-4},{11,-8.25f,-4},{11,7.75f,-4}};
        for(unsigned v=0;v<4;++v) {auto& vertex=scene.vertices[wall.firstVertex+v];vertex.position=corners[v];
            vertex.normal={0,0,1};vertex.tangent={1,0,0,1};}
        auto& camera=scene.cameras[0];camera.transform={};camera.transform.translation={3,-.25f,-3};
        camera.verticalFovRadians=.1f;camera.nearPlane=variant==5?1:.01f;camera.farPlane=variant==9?1.0005f:100;
        auto receiverProgram=std::make_shared<gt::Program>(*fixture.program->objectPrograms[0]);
        receiverProgram->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;receiverProgram->object=wall.id;
        receiverProgram->authoredMaterial=wall.material;receiverProgram->parameters.identifiers[0]=wall.material.value;
        receiverProgram->programIdentity=receiverProgram->bindingDigest="metal-diagonal-receiver";
        if(variant==1) {
            const unsigned indices[]{0,2,3};const auto local=wall.vertexCount;
            for(unsigned v=0;v<3;++v) {auto vertex=scene.vertices[wall.firstVertex+indices[v]];
                if(v==0)vertex.position.x+=.125f;
                scene.vertices[scene.vertexCount++]=vertex;scene.indices[wall.firstIndex+3+v]=local+v;}
            wall.vertexCount+=3;
        }
        if(variant==4)scene.vertices[wall.firstVertex+3].position.z-=.015625f;
        auto description=diffractionmaterial::makeAluminiumBinaryGratingPreset();description.roughness={0,0};
        description.spectrum.firstOrder=description.spectrum.lastOrder=0;description.microstructure.grooveDepthNanometres=.9f;
        const auto admitted=diffractionmaterial::admit(description,error);if(!admitted)return false;
        const auto material=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*admitted);
        diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest request;
        request.version=diffractionmaterialbinding::kSceneReceiverWireVersion;request.scene=scene.id;request.object=targetId;
        request.sceneRevision=request.structuralRevision=request.evaluationRevision=request.materialRevision=1;
        request.material=description;request.structuralDigest=material->structuralDigest();request.lighting.version=3;
        request.lighting.pathCount=1;request.lighting.traceMaximumDistance=20;
        request.lighting.sceneReceiverTransport=diffractionmaterial::SceneReceiverTransport::AuthoredSpectralLambertian;
        request.lighting.paths[0].kind=diffractionmaterial::LightingPathKind::Direct;
        request.lighting.paths[0].incident.direction={0,0,1};request.lighting.paths[0].incident.radiance.fill(1);
        const auto leaf=gc::compileDiscreteGrating(scene,request,arbitgpu::NativeFixtureMaterialBackend::Metal,error);if(!leaf)return false;
        auto glass=std::make_shared<gt::Program>(*fixture.program);glass->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;glass->objectPrograms.clear();
        auto root=std::make_shared<gt::Program>(*leaf);root->objectPrograms={glass,receiverProgram};
        if(variant==2||variant==3) {
            auto split=wall;split.id={wall.id.value+1000};split.firstIndex+=3;split.indexCount=3;wall.indexCount=3;
            auto program=std::make_shared<gt::Program>(*receiverProgram);program->object=split.id;
            if(variant==3) {auto replacement=scene.materials[receiverIndex];replacement.id={split.material.value+1000};
                scene.materials[scene.materialCount++]=replacement;split.material=replacement.id;program->authoredMaterial=replacement.id;
                program->parameters.identifiers[0]=replacement.id.value;}
            program->bindingDigest=program->programIdentity="metal-diagonal-split-receiver";
            scene.objects[scene.objectCount++]=split;root->objectPrograms.push_back(program);
        }
        if(variant==7) {
            // A nearer competing owner crosses part of the closed pixel while
            // the emitted ray still reaches the nominated wall.
            videohelper::tests::smoothglassfixture::receiver(scene,-3.5f,{0,0,0});
            auto& competitor=scene.objects[scene.objectCount-1];
            const HarmonicMIDI::grid::SceneVec3 points[]={{3,-.27f,-3.5f},{3.02f,-.27f,-3.5f},{3.02f,-.23f,-3.5f},{3,-.23f,-3.5f}};
            for(unsigned v=0;v<4;++v)scene.vertices[competitor.firstVertex+v].position=points[v];
            auto program=std::make_shared<gt::Program>(*receiverProgram);program->object=competitor.id;
            program->authoredMaterial=competitor.material;program->parameters.identifiers[0]=competitor.material.value;
            program->programIdentity=program->bindingDigest="metal-diagonal-nearer-competitor";root->objectPrograms.push_back(program);
        }
        if(variant==8) {
            // The released source authority refuses receiver emission. Keep
            // that refusal before any native capture, rather than fabricating
            // an emitter-containing frozen table to exercise coverage.
            receiverProgram->parameters.emissionRoughness[0]=1;
        }
        std::sort(root->objectPrograms.begin(),root->objectPrograms.end(),[](const auto& a,const auto& b){return a->object.value<b->object.value;});
        const auto resources=arbitgpu::prepareMetalGlassGratingResources(fixture.scene,root,{targetId,material},leaf);
        if(variant==8) {
            gc::Prepared emitter;
            if(resources.prepared&&arbitgpu::prepareMetalGlassCaustic(resources.resources,light.id,*fixture.policy,emitter,error))
                return fail("Metal restricted coverage widened the released receiver-emission authority");
            continue;
        }
        if(!resources.prepared)return fail(resources.error.c_str());
        arbitgpu::NativeFixtureSceneRuntimeInputs runtime;runtime.linearColor=true;runtime.timeSeconds=.25f;
        gc::SchedulingDescription scheduling;scheduling.entrySubdivisions=1;
        std::vector<arbitgpu::opaquevisibility::Instance> instances;
        for(std::size_t o=0;o<scene.objectCount;++o)instances.push_back({videohelper::tests::sceneclosest::identity(),o});
        gc::Prepared prepared;gc::Frame schedule;std::shared_ptr<const gc::GratingIncidentBinding> binding;
        if(!arbitgpu::prepareMetalGlassCaustic(resources.resources,light.id,*fixture.policy,prepared,error)
            ||!gc::buildGratingStagingFrame(prepared,instances,&runtime,1,1,scheduling,schedule,error)
            ||!gc::prepareGratingIncidentBinding(prepared,schedule,leaf,material,.25,binding,error,gc::SourceMeasureUse::DirectionalIrradiancePreparation))return false;
        const auto& triangle=binding->triangles.front();const gc::Point point{2.984375f,-.25f,3.015625f};
        const auto origin=gc::advance(point,triangle.normal,.1f);gc::NativeDirectGratingEndpointInput input;
        input.primitive={triangle.object,triangle.material,triangle.instance,triangle.triangle};
        input.originMinimum={origin[0],origin[1],origin[2],gc::kRayMinimum};
        input.directionMaximum={-triangle.normal[0],-triangle.normal[1],-triangle.normal[2],1};
        std::shared_ptr<const gc::DirectGratingIrradianceCohort> cohort;
        if(!gc::prepareDirectGratingIrradianceCohort({binding},{input,input},0,0,cohort,error))return false;
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame;
        if(!arbitgpu::stageMetalDirectGratingIrradianceCohort(resources.resources,cohort,runtime,frame,error)
            ||!arbitgpu::inspectMetalRasterCaptureForFixture(frame,true,error))return false;
        gc::DirectGratingIrradianceCohortReadback sum;std::vector<gc::NativeDirectGratingContributionReadback> sources;
        gc::DirectGratingTerminalOrderLedger terminal;gc::DirectGratingTerminalReceiverReadback receiver;
        if(!arbitgpu::produceMetalDirectGratingIrradianceCohort(frame,cohort,sum,error)
            ||!arbitgpu::readMetalDirectGratingCohortForFixture(frame,sources,error)
            ||!arbitgpu::produceMetalDirectGratingTerminalOrders(frame,cohort,sum.submission,.25,sources,terminal,error)
            ||!arbitgpu::produceMetalDirectTerminalReceivers(frame,terminal.nativeResult,sum.submission,.25,receiver,error)
            ||(variant!=6&&!std::any_of(receiver.rows.begin(),receiver.rows.end(),[](const auto& row){
                return row.disposition==gc::DirectGratingTerminalReceiverRow::Disposition::Pixel;})))
            return fail("Metal diagonal fixture lacks genuine emitted Pixel rows and captured all-triangle history");
        gc::CameraCellCoverageReadback coverage;std::array<std::uintptr_t,4> buffers{};
        std::array<std::uint64_t,12> before{},after{};std::array<std::uint64_t,5> priorStorage{},nextStorage{};
        const auto operation=[&](int stop=-1,std::shared_ptr<const gc::NativeCameraCellCoverageResult>* escaped=nullptr){
            return arbitgpu::produceMetalCameraCellCoverageForFixture(frame,receiver.nativeResult,sum.submission,.25,coverage,error,stop,escaped);};
        if((variant>=1&&variant<=5)||variant==7) {
            if(operation()||coverage.nativeResult)return fail("Metal diagonal gap, owner/material/depth, clip or nearer competitor conflict acquired coverage");
            continue;
        }
        std::array<std::uint64_t,12> emptyAttempt{};
        const auto sourceBytes=arbitgpu::metalcaustic::cameraDiagonalComputeSource().size();
        const auto fixedBytesCharged=[&]() {
            return after[10]>=before[10]&&after[11]>=before[11]
                &&after[10]-before[10]>=2*sourceBytes+2*4*sizeof(std::uint32_t)
                &&after[11]-before[11]>=2*sourceBytes+2*4*sizeof(std::uint32_t);
        };
        if(variant==6) {
            if(!arbitgpu::readMetalCameraCellCoverageStorageForFixture(frame,buffers,before,priorStorage,error)
                ||operation(3)||!arbitgpu::readMetalCameraCellCoverageStorageForFixture(frame,buffers,after,nextStorage,error)
                ||!fixedBytesCharged()||after[0]||after[1]||after[2]||after[6]<=before[6]||after[7]<=before[7]
                ||nextStorage[0]!=priorStorage[0]||nextStorage[1]!=priorStorage[1]||nextStorage[2]!=priorStorage[2]
                ||nextStorage[3]!=priorStorage[3]||nextStorage[4]!=priorStorage[4]+1||coverage.nativeResult)
                return fail("Metal empty failed begin refunded fixed byte work or retained unescaped setup storage");
            for(unsigned retry=0;retry<2;++retry) {
                if(!receiver.rows.empty()||!arbitgpu::readMetalCameraCellCoverageStorageForFixture(frame,buffers,before,priorStorage,error)
                    ||operation(int(retry))||!arbitgpu::readMetalCameraCellCoverageStorageForFixture(frame,buffers,after,nextStorage,error)
                    ||after[0]||after[1]||after[2]||after[6]<=before[6]||after[7]<=before[7]||!fixedBytesCharged()
                    ||nextStorage[0]!=priorStorage[0]||nextStorage[1]!=priorStorage[1]||nextStorage[2]!=priorStorage[2]
                    ||nextStorage[3]!=priorStorage[3]||nextStorage[4]!=priorStorage[4]+1||coverage.nativeResult)
                    return fail("Metal empty coverage retry lost its fixed source/uniform byte work or unescaped storage unwind");
                for(unsigned k=0;k<emptyAttempt.size();++k) {
                    if(after[k]<before[k]||(retry&&after[k]-before[k]!=emptyAttempt[k]))
                        return fail("Metal empty coverage retry changed its reconstructed permanent envelope");
                    emptyAttempt[k]=after[k]-before[k];
                }
            }
            if(!arbitgpu::readMetalCameraCellCoverageStorageForFixture(frame,buffers,before,priorStorage,error))return false;
        }
        if(!operation()||!coverage.nativeResult||!arbitgpu::inspectMetalCameraCellCoverageForFixture(frame,coverage,error)
            ||coverage.nativeResult->owner()!=frame.get()||coverage.nativeResult->retained().nativeResult
            ||coverage.proof.regions.size()!=2*coverage.preparation.pixels.size()
            ||coverage.proof.vertices.size()!=6*coverage.preparation.pixels.size()
            ||coverage.bounds.triangles!=binding->frame.triangles.size()
            ||(variant==0&&coverage.preparation.pixels.size()!=1))
            return fail("Metal actual diagonal certificate lost closed regions, all raw triangles or independent native ownership");
        if(variant==6) {
            if(!arbitgpu::readMetalCameraCellCoverageStorageForFixture(frame,buffers,after,nextStorage,error)||!fixedBytesCharged()
                ||nextStorage[0]<=priorStorage[0]||nextStorage[1]<=priorStorage[1]||nextStorage[3]<=priorStorage[3]
                ||nextStorage[2]!=priorStorage[2]||nextStorage[4]!=priorStorage[4]+1)
                return fail("Metal empty published coverage omitted fixed bytes or native storage");
            for(unsigned k=0;k<emptyAttempt.size();++k)if(after[k]-before[k]!=emptyAttempt[k])
                return fail("Metal empty publication changed or refunded its attempted envelope");
            if(!arbitgpu::inspectMetalCameraCellCoverageResourceRefusalsForFixture(frame,coverage,error)
                ||!arbitgpu::inspectMetalCameraCellCoverageForFixture(frame,coverage,error))
                return fail("Metal empty coverage accepted fixed or actual-inspection byte journal mutation");
        }
        const auto incumbent=coverage;const auto parent=receiver.nativeResult;
        if(variant==0||variant==6||variant==9) {
            std::shared_ptr<const arbitgpu::MetalCameraGeometricRangeResult> range;
            std::array<std::uintptr_t,4> rangeBuffers{};std::array<std::uint64_t,11> rangeBefore{},rangeAfter{};
            std::array<std::uint64_t,5> rangeStorageBefore{},rangeStorageAfter{};
            const auto rangeOperation=[&](int stop=-1,std::shared_ptr<const arbitgpu::MetalCameraGeometricRangeResult>* escaped=nullptr){
                return arbitgpu::produceMetalCameraGeometricRangeForFixture(frame,coverage.nativeResult,range,error,stop,escaped);};
            if(variant==6){
                range=foreignRange;
                if(!foreignRangeFrame||!range
                    ||!arbitgpu::readMetalCameraGeometricRangeStorageForFixture(frame,rangeBuffers,rangeBefore,rangeStorageBefore,error)
                    ||rangeOperation()||range!=foreignRange
                    ||error!="Metal geometric range incumbent is not an exact registered local result"
                    ||!arbitgpu::readMetalCameraGeometricRangeStorageForFixture(frame,rangeBuffers,rangeAfter,rangeStorageAfter,error)
                    ||rangeBefore!=rangeAfter||rangeStorageBefore!=rangeStorageAfter
                    ||!arbitgpu::inspectMetalCameraGeometricRangeForFixture(foreignRangeFrame,foreignRange,error))
                    return fail("Metal foreign range incumbent retained an uncharged parent or altered either owner");
                range.reset();
            }
            const auto rangeSourceBytes=arbitgpu::metalcaustic::cameraGeometricRangeComputeSource().size();
            const auto rangeFixedBytesCharged=[&](){
                return rangeAfter[7]>=rangeBefore[7]&&rangeAfter[8]>=rangeBefore[8]
                    &&rangeAfter[7]-rangeBefore[7]>=2*rangeSourceBytes+2*4*sizeof(std::uint32_t)
                    &&rangeAfter[8]-rangeBefore[8]>=2*rangeSourceBytes+2*4*sizeof(std::uint32_t);
            };
            // Receiver attribution is not a native camera trace. Missing trace
            // authority refuses only the additional certificate, at zero P too.
            if(!arbitgpu::readMetalCameraGeometricRangeStorageForFixture(frame,rangeBuffers,rangeBefore,rangeStorageBefore,error)
                ||rangeOperation()||range||!arbitgpu::readMetalCameraGeometricRangeStorageForFixture(frame,rangeBuffers,rangeAfter,rangeStorageAfter,error)
                ||rangeStorageAfter[0]!=rangeStorageBefore[0]||rangeStorageAfter[1]!=rangeStorageBefore[1]
                ||rangeStorageAfter[2]!=rangeStorageBefore[2]||rangeStorageAfter[3]!=rangeStorageBefore[3]
                ||rangeStorageAfter[4]!=rangeStorageBefore[4]+1||!rangeFixedBytesCharged()
                ||!arbitgpu::inspectMetalCameraCellCoverageForFixture(frame,coverage,error))
                return fail("Metal missing camera interval changed incumbent coverage or erased admitted fixed work");
            // The existing trace producer supplies a genuine independent
            // completed query owner through its real API. Range production
            // itself never issues a substitute camera-center query.
            gc::NativeGlassCameraGratingTraceInput traceInput;
            if(!gc::makeGlassCameraGratingTraceInput(binding->frame,0,0,0,traceInput,error))return false;
            std::vector<gc::NativeGlassCameraGratingTraceInput> traceInputs;traceInputs.reserve(1);traceInputs.push_back(traceInput);
            gc::NativeGlassCameraGratingTraceReadback trace;
            if(!arbitgpu::produceMetalGlassCameraGratingTrace(frame,binding,traceInputs,trace,error)
                ||!arbitgpu::validateMetalGlassCameraGratingTrace(frame,trace.submission,.25,trace,error))return false;
            if(variant==9){
                // Projector depth 1 is inside far=1.0005, while the complete
                // pixel's oblique Euclidean rays exceed that actual interval.
                if(!arbitgpu::readMetalCameraGeometricRangeStorageForFixture(frame,rangeBuffers,rangeBefore,rangeStorageBefore,error)
                    ||rangeOperation(5)||range
                    ||error!="Metal geometric range registered native arithmetic refused the upper inequality"
                    ||!arbitgpu::readMetalCameraGeometricRangeStorageForFixture(frame,rangeBuffers,rangeAfter,rangeStorageAfter,error)
                    ||rangeStorageAfter[0]<=rangeStorageBefore[0]||rangeStorageAfter[1]<=rangeStorageBefore[1]
                    ||rangeStorageAfter[2]<=rangeStorageBefore[2]||rangeStorageAfter[3]<=rangeStorageBefore[3]
                    ||rangeStorageAfter[4]!=rangeStorageBefore[4]+1||!rangeFixedBytesCharged()
                    ||!arbitgpu::inspectMetalCameraCellCoverageForFixture(frame,coverage,error)
                    ||!arbitgpu::validateMetalGlassCameraGratingTrace(frame,trace.submission,.25,trace,error))
                    return fail("Metal actual Euclidean upper violation erased geometric coverage or obtained a range certificate");
                continue;
            }
            if(variant==6&&(!coverage.preparation.pixels.empty()
                ||!arbitgpu::readMetalCameraGeometricRangeStorageForFixture(frame,rangeBuffers,rangeBefore,rangeStorageBefore,error)))return false;
            if(!rangeOperation()||!range||!arbitgpu::inspectMetalCameraGeometricRangeForFixture(frame,range,error))
                return fail("Metal geometric range lacks genuine retained same-frame interval and diagonal evidence");
            if(variant==6){
                if(!arbitgpu::readMetalCameraGeometricRangeStorageForFixture(frame,rangeBuffers,rangeAfter,rangeStorageAfter,error)
                    ||!rangeFixedBytesCharged()||rangeAfter[0]||rangeAfter[1]||rangeAfter[2]<=rangeBefore[2]||rangeAfter[2]-rangeBefore[2]!=32||rangeAfter[3]<=rangeBefore[3]
                    ||rangeAfter[5]<=rangeBefore[5]||rangeAfter[6]<=rangeBefore[6]
                    ||rangeAfter[9]<=rangeBefore[9]||rangeAfter[10]<=rangeBefore[10]
                    ||rangeStorageAfter[0]<=rangeStorageBefore[0]||rangeStorageAfter[1]<=rangeStorageBefore[1]
                    ||rangeStorageAfter[2]<=rangeStorageBefore[2]||rangeStorageAfter[3]<=rangeStorageBefore[3]
                    ||rangeStorageAfter[4]!=rangeStorageBefore[4]+1)
                    return fail("Metal genuine empty range omitted fixed setup, arithmetic, source or native storage");
            }
            const auto incumbentRange=range;
            for(int stop:{3,0,1,4}) {
                if(!arbitgpu::readMetalCameraGeometricRangeStorageForFixture(frame,rangeBuffers,rangeBefore,rangeStorageBefore,error)
                    ||rangeOperation(stop)||range!=incumbentRange
                    ||!arbitgpu::readMetalCameraGeometricRangeStorageForFixture(frame,rangeBuffers,rangeAfter,rangeStorageAfter,error)
                    ||rangeStorageAfter[0]!=rangeStorageBefore[0]||rangeStorageAfter[1]!=rangeStorageBefore[1]
                    ||rangeStorageAfter[2]!=rangeStorageBefore[2]||rangeStorageAfter[3]!=rangeStorageBefore[3]
                    ||rangeStorageAfter[4]!=rangeStorageBefore[4]+1||!rangeFixedBytesCharged()
                    ||(variant==6&&(rangeAfter[0]||rangeAfter[1]||rangeAfter[2]<=rangeBefore[2]||rangeAfter[2]-rangeBefore[2]!=32||rangeAfter[3]<=rangeBefore[3]
                        ||rangeAfter[5]<=rangeBefore[5]||rangeAfter[6]<=rangeBefore[6]
                        ||rangeAfter[9]<=rangeBefore[9]||rangeAfter[10]<=rangeBefore[10]))
                    ||!arbitgpu::inspectMetalCameraGeometricRangeForFixture(frame,incumbentRange,error))
                    return fail("Metal geometric range failed begin/reserve/dispatch/inspection lost owners or permanent attempted work");
            }
            if(variant==0)for(int diagnostic=10;diagnostic<=16;++diagnostic){
                if(rangeOperation(diagnostic)||error!="Metal geometric range fixture executed native mathematical refusal"
                    ||range!=incumbentRange||!arbitgpu::inspectMetalCameraCellCoverageForFixture(frame,coverage,error))
                    return fail("Metal native lower/upper/loose-bound/zero/sign/precision/exponent refusal was not conserved");
            }
            std::shared_ptr<const arbitgpu::MetalCameraGeometricRangeResult> escapedRange;
            if(!arbitgpu::readMetalCameraGeometricRangeStorageForFixture(frame,rangeBuffers,rangeBefore,rangeStorageBefore,error)
                ||rangeOperation(2,&escapedRange)||!escapedRange||range!=incumbentRange
                ||!arbitgpu::readMetalCameraGeometricRangeStorageForFixture(frame,rangeBuffers,rangeAfter,rangeStorageAfter,error)
                ||rangeStorageAfter[0]<=rangeStorageBefore[0]||rangeStorageAfter[1]<=rangeStorageBefore[1]
                ||rangeStorageAfter[2]<=rangeStorageBefore[2]||rangeStorageAfter[3]<=rangeStorageBefore[3]
                ||arbitgpu::inspectMetalCameraGeometricRangeForFixture(frame,escapedRange,error))
                return fail("Metal geometric range escaped staging lost owners or uncommitted inspection refusal");
            range=escapedRange;
            if(!arbitgpu::readMetalCameraGeometricRangeStorageForFixture(frame,rangeBuffers,rangeBefore,rangeStorageBefore,error)
                ||rangeOperation(0)||range!=escapedRange
                ||!arbitgpu::readMetalCameraGeometricRangeStorageForFixture(frame,rangeBuffers,rangeAfter,rangeStorageAfter,error)
                ||rangeStorageBefore[0]!=rangeStorageAfter[0]||rangeStorageBefore[1]!=rangeStorageAfter[1]
                ||rangeStorageBefore[2]!=rangeStorageAfter[2]||rangeStorageBefore[3]!=rangeStorageAfter[3]
                ||rangeStorageAfter[4]!=rangeStorageBefore[4]+1||!rangeFixedBytesCharged()
                ||!rangeOperation()||range==incumbentRange||range==escapedRange
                ||!arbitgpu::inspectMetalCameraGeometricRangeForFixture(frame,incumbentRange,error)
                ||!arbitgpu::inspectMetalCameraGeometricRangeRefusalsForFixture(frame,range,error)
                ||!arbitgpu::inspectMetalCameraGeometricRangeForFixture(frame,range,error)
                ||!arbitgpu::inspectMetalCameraCellCoverageForFixture(frame,coverage,error)
                ||!arbitgpu::validateMetalGlassCameraGratingTrace(frame,trace.submission,.25,trace,error))
                return fail("Metal geometric range escaped staging, mutation journals or valid retry lost independent authorities");
            if(variant==0){foreignRangeFrame=frame;foreignRange=range;}
        }
        if(variant==6)continue;
        auto forged=coverage;++forged.proof.generation;
        if(arbitgpu::inspectMetalCameraCellCoverageForFixture(frame,forged,error))return fail("Metal coverage accepted a stale certificate generation");
        forged=coverage;forged.proof.vertices[0][0]=.25;
        if(arbitgpu::inspectMetalCameraCellCoverageForFixture(frame,forged,error))return fail("Metal coverage accepted a fabricated region");
        forged=coverage;forged.parent.reset();
        if(arbitgpu::inspectMetalCameraCellCoverageForFixture(frame,forged,error))return fail("Metal coverage accepted parent forgery");
        forged=coverage;++forged.inputGeneration;
        if(arbitgpu::inspectMetalCameraCellCoverageForFixture(frame,forged,error))return fail("Metal coverage accepted changed input generation");
        forged=coverage;forged.evaluationTime=.375;
        if(arbitgpu::inspectMetalCameraCellCoverageForFixture(frame,forged,error))return fail("Metal coverage accepted a changed evaluation time");
        forged=coverage;forged.submission.frameIdentity+="changed";
        if(arbitgpu::inspectMetalCameraCellCoverageForFixture(frame,forged,error))return fail("Metal coverage accepted a changed source identity");
        if(!arbitgpu::readMetalCameraCellCoverageStorageForFixture(frame,buffers,before,priorStorage,error))return false;
        for(unsigned slot=0;slot<buffers.size();++slot) {
            id<MTLBuffer> buffer=(__bridge id<MTLBuffer>)(reinterpret_cast<void*>(buffers[slot]));if(!buffer||!buffer.contents)return false;
            auto* words=static_cast<std::uint32_t*>(buffer.contents);const auto old=words[0];words[0]^=1;
            const bool accepted=arbitgpu::inspectMetalCameraCellCoverageForFixture(frame,coverage,error);words[0]=old;
            if(accepted||!arbitgpu::inspectMetalCameraCellCoverageForFixture(frame,coverage,error))return fail("Metal inspector accepted changed actual input/result/uniform/source bits");
        }
        for(int stop:{0,1}) {
            if(!arbitgpu::readMetalCameraCellCoverageStorageForFixture(frame,buffers,before,priorStorage,error)
                ||operation(stop)||!arbitgpu::readMetalCameraCellCoverageStorageForFixture(frame,buffers,after,nextStorage,error)
                ||!gc::sameCameraCellCoverageReadback(coverage,incumbent)||nextStorage[0]!=priorStorage[0]||nextStorage[1]!=priorStorage[1]
                ||nextStorage[3]!=priorStorage[3]||nextStorage[2]<=priorStorage[2]||nextStorage[4]!=priorStorage[4]+1
                ||after[0]<=before[0]||after[1]<=before[1]||after[2]<=before[2]||after[6]<=before[6]||after[7]<=before[7]
                ||!fixedBytesCharged())
                return fail("Metal coverage failure discarded its incumbent, permanent work or unescaped storage conservation");
        }
        std::shared_ptr<const gc::NativeCameraCellCoverageResult> escaped;
        if(!arbitgpu::readMetalCameraCellCoverageStorageForFixture(frame,buffers,before,priorStorage,error)
            ||operation(2,&escaped)||!escaped||!arbitgpu::readMetalCameraCellCoverageStorageForFixture(frame,buffers,after,nextStorage,error)
            ||!gc::sameCameraCellCoverageReadback(coverage,incumbent)||nextStorage[0]<=priorStorage[0]
            ||nextStorage[1]<=priorStorage[1]||nextStorage[3]<=priorStorage[3]||!fixedBytesCharged())
            return fail("Metal escaped uncommitted coverage lost its charged native owner or fixed byte work");
        auto unpublished=escaped->retained();unpublished.nativeResult=escaped;
        if(arbitgpu::inspectMetalCameraCellCoverageForFixture(frame,unpublished,error)
            ||!arbitgpu::inspectMetalCameraCellCoverageForFixture(frame,incumbent,error)
            ||!arbitgpu::inspectMetalCameraCellCoverageResourceRefusalsForFixture(frame,incumbent,error))
            return fail("Metal escaped coverage acquired publication or old resource/work omissions were accepted");
        if(!operation()||coverage.nativeResult==incumbent.nativeResult||receiver.nativeResult!=parent
            ||!arbitgpu::inspectMetalCameraCellCoverageForFixture(frame,incumbent,error))return fail("Metal valid retry lost the old coverage or emitted parent");
        auto stale=sum.submission;++stale.rasterGeneration;const auto current=coverage;
        if(arbitgpu::produceMetalCameraCellCoverageForFixture(frame,parent,stale,.25,coverage,error)
            ||arbitgpu::produceMetalCameraCellCoverageForFixture(frame,parent,sum.submission,.375,coverage,error)
            ||!gc::sameCameraCellCoverageReadback(coverage,current))return fail("Metal stale owner/generation/time replaced coverage");
        if(frame->colorImageHandle()||frame->depthImageHandle())return fail("Metal finite pixel proof granted guarded Color");
    }
    error.clear();return true;
}

bool metalDirectionalIrradianceCohortCases(std::string& error)
{
    namespace gc=arbitgpu::glasscaustics;namespace gt=glasscaustictests;
    const auto fail=[&](const char* m){error=m;return false;};
    gc::DirectGratingIrradianceCohortReadback distinctIncumbent;
    gc::DirectGratingTerminalOrderLedger distinctTerminalIncumbent;
    gc::DirectGratingTerminalReceiverReadback distinctReceiverIncumbent;
    bool receiverPixel=false,receiverMiss=false,receiverNonReceiver=false,receiverOffscreen=false;
    bool receiverBackground=false,receiverOccluded=false,receiverSlanted=false,receiverZero=false,receiverGrazing=false,receiverEmpty=false;
    for(unsigned scenario=0;scenario<11;++scenario) {
        auto fixture=gt::ordinaryFixture();auto& first=fixture.scene->lights[0];
        first.id={0xfedcba98u};first.transform.rotation={0,0,0,1};
        first.spectralSource.directionalIrradiance={1,diffractionmaterial::DirectionalSpectralMeasure::IrradianceCoefficient,
            {.25f,.5f,.75f,1,1.25f,1.5f,1.75f,2}};first.spectralSource.radiance.fill(0);
        fixture.scene->lights[1]=first;fixture.scene->lightCount=2;auto& second=fixture.scene->lights[1];second.id={0x87654321u};
        second.spectralSource.directionalIrradiance.coefficients={2,1.75f,1.5f,1.25f,1,.75f,.5f,0};
        second.transform.rotation={0,float(std::sin(.2)),0,float(std::cos(.2))};
        if(scenario==1)second.spectralSource.directionalIrradiance.coefficients.fill(0);
        if(scenario==2)second.transform.rotation={0,1,0,0};
        if(scenario==3) {
            const float inverse=1/std::sqrt(11.0f),w=std::sqrt((1+3*inverse)/2);
            second.transform.rotation={-inverse/(2*w),-inverse/(2*w),0,w};
        }
        std::vector<std::shared_ptr<const gt::Program>> receiverPrograms;
        if(scenario>=4&&scenario<=8) {
            const auto addReceiver=[&](float left,float right,float bottom,float top,float z,float slope) {
                videohelper::tests::smoothglassfixture::receiver(*fixture.scene,-6,{0,0,0});
                auto& object=fixture.scene->objects[fixture.scene->objectCount-1];
                const float xy[4][2]={{left,bottom},{right,bottom},{right,top},{left,top}};
                const auto normal=gt::normalized(gt::D3{slope,0,-1});
                for(unsigned v=0;v<4;++v) {
                    auto& vertex=fixture.scene->vertices[object.firstVertex+v];
                    vertex.position={xy[v][0],xy[v][1],z+slope*(xy[v][0]-2)};
                    vertex.normal={float(normal[0]),0,float(normal[2])};
                }
                for(unsigned t=0;t<2;++t)std::swap(fixture.scene->indices[object.firstIndex+3*t+1],fixture.scene->indices[object.firstIndex+3*t+2]);
                auto program=std::make_shared<gt::Program>(*fixture.program->objectPrograms[0]);
                program->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;program->object=object.id;program->authoredMaterial=object.material;
                program->parameters.identifiers[0]=object.material.value;program->bindingDigest="metal-direct-receiver-"+std::to_string(object.id.value);
                program->programIdentity=program->bindingDigest;receiverPrograms.push_back(program);
            };
            if(scenario==8)addReceiver(1.999f,2.001f,-2.001f,-1.999f,-1.5f,0);
            else addReceiver(-6,6,-6,6,-1.5f,scenario==5?.05f:0);
            auto& camera=fixture.scene->cameras[0];camera.transform.translation={2,-2,-2.5f};
            camera.transform.rotation={0,1,0,0};camera.verticalFovRadians=2.5f;
            if(scenario==6)camera.transform.translation.x=30;
            if(scenario==7) {
                camera.transform.translation={0,0,-2.5f};
                addReceiver(-.6f,.6f,-.6f,.6f,-2.4f,0);
            }
            if(scenario==8){camera.transform.translation={2.013f,-1.987f,-2.5f};camera.verticalFovRadians=1;}
        }
        if(scenario==9) {
            const double angle=std::acos(2e-6);first.transform.rotation={0,float(std::sin(angle/2)),0,float(std::cos(angle/2))};
        }
        if(scenario==10){first.transform.rotation={0,1,0,0};second.transform.rotation={0,1,0,0};}
        auto glass=std::make_shared<gt::Program>(*fixture.program);glass->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;
        glass->objectPrograms.clear();
        auto& target=fixture.scene->objects[1];
        for(std::size_t v=target.firstVertex;v<target.firstVertex+target.vertexCount;++v) {
            auto& p=fixture.scene->vertices[v];p.uv={p.position.x/12,p.position.y/12};
        }
        auto description=diffractionmaterial::makeAluminiumBinaryGratingPreset();description.roughness={0,0};
        description.spectrum.firstOrder=description.spectrum.lastOrder=1;description.microstructure.grooveDepthNanometres=.9f;
        const auto admitted=diffractionmaterial::admit(description,error);if(!admitted)return false;
        auto material=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*admitted);
        diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest request;
        request.version=diffractionmaterialbinding::kSceneReceiverWireVersion;request.scene=fixture.scene->id;request.object=target.id;
        request.sceneRevision=request.structuralRevision=request.evaluationRevision=request.materialRevision=1;
        request.material=description;request.structuralDigest=material->structuralDigest();request.lighting.version=3;
        request.lighting.pathCount=1;request.lighting.traceMaximumDistance=20;
        request.lighting.sceneReceiverTransport=diffractionmaterial::SceneReceiverTransport::AuthoredSpectralLambertian;
        request.lighting.paths[0].kind=diffractionmaterial::LightingPathKind::Direct;
        request.lighting.paths[0].incident.direction={0,0,1};request.lighting.paths[0].incident.radiance.fill(1);
        const auto leaf=gc::compileDiscreteGrating(*fixture.scene,request,arbitgpu::NativeFixtureMaterialBackend::Metal,error);if(!leaf)return false;
        auto root=std::make_shared<gt::Program>(*leaf);root->objectPrograms={glass};
        for(const auto& program:receiverPrograms)root->objectPrograms.push_back(program);
        const auto resources=arbitgpu::prepareMetalGlassGratingResources(fixture.scene,root,{target.id,material},leaf);
        if(!resources.prepared){error=resources.error;return false;}
        arbitgpu::NativeFixtureSceneRuntimeInputs runtime;runtime.linearColor=true;runtime.timeSeconds=.25f;
        gc::SchedulingDescription scheduling;scheduling.entrySubdivisions=1;
        std::vector<arbitgpu::opaquevisibility::Instance> instances;
        for(std::size_t o=0;o<fixture.scene->objectCount;++o) {
            auto matrix=videohelper::tests::sceneclosest::identity();const auto& p=fixture.scene->objects[o].transform.translation;
            matrix[12]=p.x;matrix[13]=p.y;matrix[14]=p.z;instances.push_back({matrix,o});
        }
        std::vector<std::shared_ptr<const gc::GratingIncidentBinding>> bindings;
        for(auto id:{first.id,second.id}) {
            gc::Prepared prepared;gc::Frame schedule;std::shared_ptr<const gc::GratingIncidentBinding> binding;
            if(!arbitgpu::prepareMetalGlassCaustic(resources.resources,id,*fixture.policy,prepared,error)
                ||!gc::buildGratingStagingFrame(prepared,instances,&runtime,32,32,scheduling,schedule,error)
                ||!gc::prepareGratingIncidentBinding(prepared,schedule,leaf,material,.25,binding,error,
                    gc::SourceMeasureUse::DirectionalIrradiancePreparation))return false;
            bindings.push_back(std::move(binding));
        }
        const auto& triangle=bindings[0]->triangles[0];
        const auto endpointAt=[&](const gc::Point& bary) {
            gc::Point p{};for(unsigned v=0;v<3;++v)p=gc::advance(p,triangle.positions[v],bary[v]);
            const auto origin=gc::advance(p,triangle.normal,.25f);gc::NativeDirectGratingEndpointInput input;
            input.primitive={triangle.object,triangle.material,triangle.instance,triangle.triangle};
            input.originMinimum={origin[0],origin[1],origin[2],gc::kRayMinimum};
            input.directionMaximum={-triangle.normal[0],-triangle.normal[1],-triangle.normal[2],1};return input;
        };
        // Equal rays are independent ordered, counted occurrences.
        std::vector<gc::NativeDirectGratingEndpointInput> endpoints{endpointAt({1.0f/3,1.0f/3,1.0f/3}),endpointAt({.2f,.3f,.5f})};
        endpoints.push_back(endpoints[0]);
        if(scenario==8)endpoints[1]=endpoints[0];
        std::shared_ptr<const gc::DirectGratingIrradianceCohort> cohort;
        if(!gc::prepareDirectGratingIrradianceCohort(bindings,endpoints,0,0,cohort,error))return false;
        auto retained=cohort;auto malformed=bindings;malformed[1]=malformed[0];
        if(gc::prepareDirectGratingIrradianceCohort(malformed,endpoints,0,0,retained,error)||retained!=cohort)
            return fail("Metal cohort preparation accepted a duplicated source or replaced its incumbent");
        malformed=bindings;std::swap(malformed[0],malformed[1]);
        if(gc::prepareDirectGratingIrradianceCohort(malformed,endpoints,0,0,retained,error))return fail("Metal cohort preparation accepted reordered scene sources");
        malformed=bindings;malformed.pop_back();
        if(gc::prepareDirectGratingIrradianceCohort(malformed,endpoints,0,0,retained,error))return fail("Metal cohort preparation accepted a source prefix");
        if(gc::prepareDirectGratingIrradianceCohort(bindings,endpoints,renderpassoutput::kMaximumTotalBytes,0,retained,error)
            ||gc::prepareDirectGratingIrradianceCohort(bindings,endpoints,0,gc::kMaximumQueryWork,retained,error)||retained!=cohort)
            return fail("Metal cohort preparation exceeded original complete byte or work bounds");
        auto excessive=endpoints;excessive.resize(gc::kMaximumEntrySamples+1,endpoints[0]);
        if(gc::prepareDirectGratingIrradianceCohort(bindings,excessive,0,0,retained,error))return fail("Metal cohort admitted more than 2048 endpoints");
        endpoints[0].originMinimum[0]+=.125f;
        if(gc::sameDirectGratingEndpointInput(endpoints[0],cohort->inputs()[0]))return fail("Metal cohort retained mutable caller endpoint inputs");
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame;
        if(!arbitgpu::stageMetalDirectGratingIrradianceCohort(resources.resources,cohort,runtime,frame,error))return false;
        if(!arbitgpu::inspectMetalRasterCaptureForFixture(frame,true,error))return false;
        gc::DirectGratingIrradianceCohortReadback sum;
        if(scenario==1) {
            sum=distinctIncumbent;
            if(!arbitgpu::inspectMetalDirectGratingCohortIncumbentRefusalForFixture(frame,cohort,sum,error))return false;
        }
        if(!arbitgpu::produceMetalDirectGratingIrradianceCohort(frame,cohort,sum,error))return false;
        std::vector<gc::NativeDirectGratingContributionReadback> readbacks;
        if(!arbitgpu::readMetalDirectGratingCohortForFixture(frame,readbacks,error)||readbacks.size()!=2||sum.endpoints.size()!=3)
            return fail("Metal cohort lost complete actual native source or endpoint occurrences");
        const auto* native=dynamic_cast<const gc::NativeDirectGratingIrradianceCohortBinding*>(frame.get());
        if(!native||native->directCohort()!=cohort||native->directSourceBatch(0)==native->directSourceBatch(1))
            return fail("Metal cohort retargeted one source head instead of retaining distinct views");
        for(std::size_t e=0;e<3;++e)for(unsigned band=0;band<8;++band) {
            double expected=0;
            for(unsigned s=0;s<2;++s) {
                const auto& binding=*cohort->sources()[s];const auto& sample=readbacks[s].samples[e];
                const float cosine=std::max(0.0f,gc::dot(triangle.normal,gc::Point{binding.frame.sourceDirectionTrace[0],
                    binding.frame.sourceDirectionTrace[1],binding.frame.sourceDirectionTrace[2]}));
                const bool blocked=scenario<4?(scenario==3&&s==1&&e!=1):sample.sourceHit.status==1;
                const float contribution=blocked?0:binding.frame.sourceDirectionalIrradiance.coefficients[band]*cosine;
                if(!sample.complete||sample.source.input.identifiers[3]!=binding.prepared.sceneLight.value
                    ||!gc::nearDirectionalSourcePotential(sample.surfaceIrradiance[band],contribution)
                    ||(blocked&&sample.sourceHit.status!=1))return fail("Metal cohort native source echo, visibility or D cosine differs from authored scene");
                expected+=sample.surfaceIrradiance[band];
            }
            if(sum.endpoints[e].surfaceIrradiance[band]!=float(expected))return fail("Metal cohort applied another cosine or lost a physical source contribution");
        }
        const auto before=sum;auto altered=readbacks;altered[1].samples[0].source.input.coefficients[0]+=.125f;
        const auto rejected=[&](const auto& r,const auto& seal,double time) {
            if(arbitgpu::validateMetalDirectGratingIrradianceCohort(frame,seal,time,r,sum,error))return false;
            return sum.cohort==before.cohort&&sum.submission==before.submission&&sum.endpoints[0].surfaceIrradiance==before.endpoints[0].surfaceIrradiance;
        };
        if(!rejected(altered,sum.submission,.25))return fail("Metal cohort accepted changed caller source coefficients or partial output");
        altered=readbacks;std::swap(altered[1].samples[0],altered[1].samples[1]);
        if(!rejected(altered,sum.submission,.25))return fail("Metal cohort accepted reordered endpoint occurrences");
        altered=readbacks;std::swap(altered[0],altered[1]);
        if(!rejected(altered,sum.submission,.25))return fail("Metal cohort accepted reordered actual source batches");
        altered=readbacks;altered[1]=altered[0];
        if(!rejected(altered,sum.submission,.25))return fail("Metal cohort accepted a repeated actual source batch");
        altered=readbacks;altered.pop_back();
        if(!rejected(altered,sum.submission,.25))return fail("Metal cohort accepted a partial actual source roster");
        altered=readbacks;altered.push_back(readbacks[0]);
        if(!rejected(altered,sum.submission,.25))return fail("Metal cohort accepted an extra actual source batch");
        altered=readbacks;++altered[1].sourceGeneration;
        if(!rejected(altered,sum.submission,.25))return fail("Metal cohort accepted a stale actual source generation");
        altered=readbacks;++altered[1].endpointGeneration;
        if(!rejected(altered,sum.submission,.25))return fail("Metal cohort accepted a stale actual endpoint generation");
        altered=readbacks;altered[1].complete=false;
        if(!rejected(altered,sum.submission,.25))return fail("Metal cohort accepted an incomplete source result");
        altered=readbacks;altered[1].samples[0].outcome=gc::NativeDirectGratingContributionSample::Outcome::Unsupported;
        if(!rejected(altered,sum.submission,.25))return fail("Metal cohort accepted an unsupported source result");
        auto stale=sum.submission;stale.completedFence=0;
        if(!rejected(readbacks,stale,.25)||!rejected(readbacks,sum.submission,.375))return fail("Metal cohort accepted stale fence or time");
        if(scenario==0&&!arbitgpu::inspectMetalDirectGratingCohortRefusalsForFixture(frame,sum,error))return false;
        if(scenario==0) {
            const auto previous=sum;
            if(!arbitgpu::produceMetalDirectGratingIrradianceCohort(frame,cohort,sum,error)
                ||sum.submission.submissionSerial<=previous.submission.submissionSerial
                ||!arbitgpu::readMetalDirectGratingCohortForFixture(frame,readbacks,error))
                return fail("Metal repeated cohort lost prior retained storage or reused its common seal");
        }
        if(!arbitgpu::validateMetalDirectGratingIrradianceCohort(frame,sum.submission,.25,readbacks,sum,error)
            ||frame->colorImageHandle()||frame->colorTextureViewHandle()||frame->depthImageHandle()||frame->depthTextureViewHandle())
            return fail("Metal cohort lost accepted receipts or granted public Color/base attachments");
        // Actual Metal owners, buffers, completed commands and reservation
        // lease. Unlike shared mocks, this route must pass the native inspector.
        gc::DirectGratingTerminalOrderLedger terminal=distinctTerminalIncumbent;
        std::unique_ptr<gc::NativeDirectGratingTerminalLease> unregisteredLease;
        gc::DirectGratingTerminalNativeBudget unregisteredBudget{123,456,789};
        if(native->beginDirectCohortTerminalLease(sum.submission,.25,readbacks,unregisteredLease,unregisteredBudget,error)
            ||unregisteredLease||unregisteredBudget.retainedBytes!=123||unregisteredBudget.combinedWork!=456
            ||unregisteredBudget.depositionWork!=789)
            return fail("Metal terminal callback bypassed the actual mutex/context entry registration");
        const auto terminalOperation=[&](const auto& receipts,const auto& seal,double time) {
            return arbitgpu::produceMetalDirectGratingTerminalOrders(frame,cohort,seal,time,receipts,terminal,error);};
        if(!terminalOperation(readbacks,sum.submission,.25)||!terminal.nativeResult||terminal.nativeResult->owner()!=frame.get()
            ||terminal.nativeResult->retained().nativeResult||terminal.bands.size()!=2*3*8
            ||terminal.bands.data()==terminal.nativeResult->retained().bands.data()
            ||(!terminal.orders.empty()&&terminal.orders.data()==terminal.nativeResult->retained().orders.data()))
            return fail("Metal terminal operation lost its independent native-owned copy");
        if(scenario==0&&terminal.bands[0].toSource==terminal.bands[3*8].toSource)
            return fail("Metal terminal operation erased distinct actual source directions");
        for(const auto& row:terminal.bands) {
            if(row.source>=2||row.endpoint>=3||row.band>=8
                ||row.incidentIrradiance!=readbacks[row.source].samples[row.endpoint].surfaceIrradiance[row.band])
                return fail("Metal terminal operation reconstructed directions or energy from a scalar sum");
            if(!row.responseEvaluated) {
                if(row.incidentIrradiance||row.orderCount)return fail("Metal terminal nonincident zero fabricated an order");
                continue;
            }
            diffractionmaterial::DiffractionOrderDistribution expected;
            if(!diffractionmaterial::evaluateDiffractionOrders(*bindings[row.source]->material,row.wavelengthNanometres,
                row.frame,row.toSource,row.uv,expected,error))return false;
            if(row.orderCount!=expected.orderCount||row.substrateReflectedPower!=expected.substrateReflectedPower
                ||row.absorbedIrradiance!=row.incidentIrradiance*(1-expected.substrateReflectedPower))
                return fail("Metal terminal response differs from the released physical evaluator");
            for(std::size_t o=0;o<row.orderCount;++o) {
                const auto& actual=terminal.orders[row.firstOrder+o];const auto& physical=expected.orders[o];
                if(actual.source!=row.source||actual.endpoint!=row.endpoint||actual.band!=row.band
                    ||actual.order.primary!=physical.primary||actual.order.secondary!=physical.secondary
                    ||actual.order.direction!=physical.direction||actual.order.reflectedPower!=physical.reflectedPower
                    ||actual.orderIrradiance!=row.incidentIrradiance*physical.reflectedPower)
                    return fail("Metal terminal direction/order identity or once-cosined per-area energy changed");
            }
        }
        gc::DirectGratingTerminalReceiverReadback receivers=distinctReceiverIncumbent;
        std::unique_ptr<gc::NativeDirectGratingTerminalReceiverLease> unregisteredReceiver;
        gc::DirectGratingTerminalReceiverNativeBudget receiverBudget{123,456,64,128,99};
        if(native->beginDirectTerminalReceiverLease(terminal.nativeResult,sum.submission,.25,unregisteredReceiver,receiverBudget,error)
            ||unregisteredReceiver||receiverBudget.retainedBytes!=123||receiverBudget.combinedWork!=456)
            return fail("Metal receiver callback bypassed its actual native guard");
        const auto receiverOperation=[&](const auto& seal,double time) {
            return arbitgpu::produceMetalDirectTerminalReceivers(frame,terminal.nativeResult,seal,time,receivers,error);};
        if(!receiverOperation(sum.submission,.25)||!receivers.nativeResult||receivers.nativeResult->owner()!=frame.get()
            ||receivers.rows.size()!=terminal.orders.size()||(!receivers.rows.empty()&&receivers.rows.data()==receivers.nativeResult->retained().rows.data())
            ||!arbitgpu::inspectMetalDirectTerminalReceivers(frame,receivers,error))
            return fail("Metal exact-order receivers lack independent native query/raster ownership");
        if(scenario==10){receiverEmpty=terminal.orders.empty()&&receivers.rows.empty()&&receivers.queryWork==0;
            if(!receiverEmpty)return fail("Metal zero-order terminal did not complete an empty native receiver operation");}
        const auto triangles=gt::worldTriangles(*bindings[0]->prepared.glass.sourceScene);
        for(std::size_t i=0;i<receivers.rows.size();++i) {
            const auto& row=receivers.rows[i];const auto& order=terminal.orders[i];
            if(!gc::sameDirectGratingTerminalEmittedOrder(row.emitted,order)||row.input.key[3]!=i
                ||row.pixel.worldArea||row.queryWork!=row.hit.work||!row.queryWork)
                return fail("Metal receiver changed terminal order, zero-energy occurrence or query attribution");
            const gt::D3 origin{row.input.originMinimum[0],row.input.originMinimum[1],row.input.originMinimum[2]};
            const gt::D3 direction{row.input.directionMaximum[0],row.input.directionMaximum[1],row.input.directionMaximum[2]};
            const auto hit=gt::closest(triangles,origin,direction,row.input.originMinimum[3],row.input.directionMaximum[3]);
            if(bool(row.hit.status==1)!=hit.found||(hit.found&&(hit.object!=row.hit.objectId||hit.material!=row.hit.materialId
                ||std::abs(hit.distance-row.hit.distance)>8*gc::bias(row.hit.position))))
                return fail("Metal receiver nearest hit differs from the authored triangle oracle");
            using Row=gc::DirectGratingTerminalReceiverRow;
            receiverMiss|=row.disposition==Row::Disposition::Miss;receiverNonReceiver|=row.disposition==Row::Disposition::NonReceiver;
            receiverOffscreen|=row.disposition==Row::Disposition::Offscreen;
            if(row.disposition==Row::Disposition::Pixel) {
                receiverPixel=true;receiverBackground|=row.rasterSample==Row::RasterSample::Background;
                receiverOccluded|=row.rasterSample==Row::RasterSample::Surface&&row.raster.object!=row.hit.objectId;
                const auto& frozen=bindings[0]->frame;const auto point=gc::cameraPosition(frozen);
                const auto ray=gc::cameraRay(frozen,float(row.pixel.x)+.5f,float(row.pixel.y)+.5f);
                const auto& q=frozen.camera.transform.rotation;
                const auto local=gc::rotateSceneVector({-q.x,-q.y,-q.z,q.w},{ray[0],ray[1],ray[2]});
                const auto raster=gt::closest(triangles,{point[0],point[1],point[2]},{ray[0],ray[1],ray[2]},
                    frozen.camera.nearPlane/-local.z,frozen.camera.farPlane/-local.z);
                if((row.rasterSample==Row::RasterSample::Surface)!=raster.found
                    ||(raster.found&&(row.raster.object!=raster.object||row.raster.material!=raster.material
                        ||std::abs(row.raster.linearDepth-raster.distance*-local.z)>8*gc::bias(row.hit.position))))
                    return fail("Metal receiver raster center differs from its independent camera-plane oracle");
                if(scenario==5&&row.raster.object==row.hit.objectId
                    &&std::abs(row.pixel.linearDepth-row.raster.linearDepth)>1e-4f)receiverSlanted=true;
            }
            receiverZero|=order.orderIrradiance==0&&row.queryWork>0;
            const auto& endpoint=terminal.incident.endpoints[order.endpoint].endpoint;
            if(scenario==9&&gc::dot(gc::Point{float(order.order.direction[0]),float(order.order.direction[1]),float(order.order.direction[2])},
                endpoint.outwardNormal)<1e-5f) {
                receiverGrazing=true;
                const auto launch=gc::advance(endpoint.position,endpoint.outwardNormal,gc::bias(endpoint.position));
                if(row.input.originMinimum[2]!=launch[2]||launch[2]<=endpoint.position[2])
                    return fail("Metal grazing receiver launch lost outward-normal clearance");
            }
        }
        const auto receiverOwner=receivers.nativeResult;
        const auto refusedReceiver=[&](const auto& seal,double time) {
            return !receiverOperation(seal,time)&&receivers.nativeResult==receiverOwner
                &&gc::sameDirectGratingTerminalReceiverReadback(receivers,receiverOwner->retained(),false);};
        auto staleReceiver=sum.submission;++staleReceiver.rasterGeneration;
        if(!refusedReceiver(staleReceiver,.25)||!refusedReceiver(sum.submission,.375))
            return fail("Metal receiver accepted stale raster generation or evaluation time");
        if(!receivers.rows.empty()) {
            auto alteredReceiver=receivers;alteredReceiver.rows.front().input.key[3]^=1;
            if(arbitgpu::inspectMetalDirectTerminalReceivers(frame,alteredReceiver,error))return fail("Metal receiver inspector accepted reordered native input");
            alteredReceiver=receivers;alteredReceiver.rows.front().emitted.orderIrradiance+=1;
            if(arbitgpu::inspectMetalDirectTerminalReceivers(frame,alteredReceiver,error))return fail("Metal receiver inspector accepted changed emitted energy");
            alteredReceiver=receivers;++alteredReceiver.inputGeneration;
            if(arbitgpu::inspectMetalDirectTerminalReceivers(frame,alteredReceiver,error))return fail("Metal receiver inspector accepted stale input generation");
        }
        if(scenario==4&&!arbitgpu::inspectMetalDirectTerminalReceiverRefusalsForFixture(frame,receivers,error))return false;
        if(!receiverOperation(sum.submission,.25)||receivers.nativeResult==receiverOwner
            ||receiverOwner->retained().rows.size()!=terminal.orders.size())
            return fail("Metal receiver retry discarded an externally retained earlier result");
        if(scenario==0)distinctReceiverIncumbent=receivers;
        const auto retainedTerminal=terminal;const auto firstOwner=terminal.nativeResult;
        const auto refusedTerminal=[&](const auto& receipts,const auto& seal,double time) {
            return !terminalOperation(receipts,seal,time)&&terminal.nativeResult==firstOwner
                &&terminal.bounds.bytes==retainedTerminal.bounds.bytes&&terminal.bands.size()==retainedTerminal.bands.size();};
        auto terminalSources=readbacks;std::swap(terminalSources[0],terminalSources[1]);
        if(!refusedTerminal(terminalSources,sum.submission,.25))return fail("Metal terminal accepted reordered actual native source owners");
        terminalSources=readbacks;terminalSources[1].samples[0].source.directionTrace[0]+=.125f;
        if(!refusedTerminal(terminalSources,sum.submission,.25))return fail("Metal terminal accepted changed native source direction");
        auto terminalSeal=sum.submission;terminalSeal.completedFence=0;
        if(!refusedTerminal(readbacks,terminalSeal,.25)||!refusedTerminal(readbacks,sum.submission,.375))
            return fail("Metal terminal accepted stale completed fence or time");
        if(scenario==0&&!arbitgpu::inspectMetalDirectTerminalLeaseRefusalsForFixture(frame,terminal,error))return false;
        if(!terminalOperation(readbacks,sum.submission,.25)||terminal.nativeResult==firstOwner
            ||firstOwner->retained().bands.size()!=retainedTerminal.bands.size())
            return fail("Metal terminal replacement lost an externally retained earlier owner");
        if(scenario==0) {
            // A later native cohort admission must still account for both
            // sealed owners and all refused terminal attempt work. Coherent
            // reductions of both byte totals must refuse all later producers.
            if(!arbitgpu::inspectMetalDirectTerminalLaterBudgetRefusalsForFixture(frame,sum,error))return false;
            if(!arbitgpu::produceMetalDirectGratingIrradianceCohort(frame,cohort,sum,error)
                ||!arbitgpu::readMetalDirectGratingCohortForFixture(frame,readbacks,error)
                ||!terminalOperation(readbacks,sum.submission,.25)||firstOwner->retained().incident.submission==sum.submission)
                return fail("Metal later native admission discarded earlier terminal ownership or reused its fence");
        }
        if(frame->colorImageHandle()||frame->colorTextureViewHandle()||frame->depthImageHandle()||frame->depthTextureViewHandle())
            return fail("Metal terminal math granted public rendering attachments");
        if(scenario==0){distinctIncumbent=sum;distinctTerminalIncumbent=terminal;}
    }
    if(!receiverPixel||!receiverMiss||!receiverNonReceiver||!receiverOffscreen||!receiverBackground||!receiverOccluded
        ||!receiverSlanted||!receiverZero||!receiverGrazing||!receiverEmpty)
        return fail("Metal receiver fixtures omit a native hit/raster, zero-energy, slanted or grazing outcome");
    return metalCameraDiagonalCoverageCases(error)&&metalFiniteGratingSourceCases(error)&&metalFiniteGratingReceiverCases(error);
}

bool metalGlassGratingIncidentCases(arbitgpu::NativeFixtureSceneBackend& backend,std::string& error)
{
    namespace gc=arbitgpu::glasscaustics;namespace gt=glasscaustictests;
    const auto fail=[&](const char* message){error=message;return false;};
    auto fixture=gt::ordinaryFixture();fixture.scene->lights[0].transform.rotation={0,0,0,1};
    fixture.scene->cameras[0].transform.translation={0,0,-2};
    const auto& target=fixture.scene->objects[1];
    for(std::size_t v=0;v<target.vertexCount;++v) {
        auto& vertex=fixture.scene->vertices[target.firstVertex+v];vertex.uv={vertex.position.x/12,vertex.position.y/12};
    }
    fixture.program->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;
    for(auto& child:fixture.program->objectPrograms) {
        auto material=std::make_shared<gt::Program>(*child);material->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;child=material;
    }
    const auto resources=backend.prepare(fixture.scene,fixture.program);
    if(!resources.prepared) {error=resources.error;return false;}
    gc::Prepared prepared;
    if(!arbitgpu::prepareMetalGlassCaustic(resources.resources,fixture.ordinaryLight,*fixture.policy,prepared,error))return false;
    arbitgpu::NativeFixtureSceneRuntimeInputs runtime;runtime.linearColor=true;
    gc::SchedulingDescription scheduling;scheduling.entrySubdivisions=1;
    const auto staging=arbitgpu::renderMetalGlassCausticFrame(fixture.scene,resources.resources,32,32,runtime,prepared,scheduling);
    if(!staging.rendered||!staging.frame) {error=staging.error;return false;}
    gc::NativeReadback legacy;gc::EnergyLedger energy;
    if(!arbitgpu::readMetalGlassCausticReceipt(staging.frame,fixture.frame,legacy,energy,error))return false;
    auto description=diffractionmaterial::makeAluminiumBinaryGratingPreset();
    description.spectrum.firstOrder=description.spectrum.lastOrder=1;description.roughness={0,0};
    description.microstructure.grooveDepthNanometres=.9f;
    const auto admitted=diffractionmaterial::admit(description,error);if(!admitted)return false;
    const auto material=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*admitted);
    auto grating=std::make_shared<gt::Program>(*fixture.program->objectPrograms[0]);
    grating->kind=arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;grating->spectralReflectance.reset();
    grating->diffractionPathCount=1;grating->bindingDigest="metal-queried-grating-binding";grating->programIdentity="metal-queried-grating-program";
    grating->diffractionPaths[0].material=diffractionmaterial::physicalcheckpoint::makeGpuParameters(*material,{});
    std::shared_ptr<const gc::GratingIncidentBinding> binding;
    if(!gc::prepareGratingIncidentBinding(prepared,fixture.frame,grating,material,.25,binding,error))return false;
    arbitgpu::NativeRawPassPixels before,after;
    if(!staging.frame->readRawPass(renderpassoutput::Output::Color,before,error))return false;
    gc::NativeGratingReadback receipt;
    if(!arbitgpu::produceMetalGlassGratingIncidents(staging.frame,binding,receipt,error))return false;
    if(!receipt.complete||receipt.overflow||receipt.incidents.empty()||receipt.orderWork!=3*receipt.incidents.size()
        ||receipt.submission.frameIdentity!=binding->identity||receipt.submission.completedFence!=receipt.submission.submissionSerial)
        return fail("Metal actual grating prerequisite lacks a completed owned native receipt");
    std::array<bool,3> seen{};
    const double n=description.substrate.refractiveIndex,k=description.substrate.extinctionCoefficient;
    const double reflectance=((1-n)*(1-n)+k*k)/((1+n)*(1+n)+k*k);
    for(const auto& event:receipt.incidents) {
        const auto& work=fixture.frame.work[event.work];const auto& entry=fixture.frame.entries[work.entry];
        const auto branches=gt::branchOracle(fixture,entry,work.band,fixture.policy->description().minimumBranchFraction,
            fixture.policy->description().maximumVisits,fixture.policy->description().maximumInterfaces);
        const auto arrival=std::find_if(branches.arrivals.begin(),branches.arrivals.end(),[&](const auto& value) {
            return value.history==event.reflectionHistory&&value.interfaces==event.interfaces&&value.receiver.object==target.id.value;});
        if(!branches.valid||arrival==branches.arrivals.end()||arrival->receiver.triangle!=event.hit.triangleId
            ||!gc::nearPower(arrival->fraction,event.fraction,5e-4))
            return fail("Metal actual glass to grating endpoint differs from authored triangle oracle");
        for(unsigned a=0;a<3;++a)if(std::abs(arrival->receiver.position[a]-event.hit.position[a])>5e-4)
            return fail("Metal actual grating ray landing differs from independent smooth glass trace");
        for(unsigned a=0;a<2;++a)if(std::abs(event.uv[a]-event.hit.position[a]/12)>1e-5)
            return fail("Metal actual closest-hit UV differs from authored barycentric interpolation");
        if(gt::inner(gt::exterior(gt::doublePoint(event.tangent),gt::doublePoint(event.bitangent)),gt::doublePoint(event.normal))<.9999)
            return fail("Metal actual grating witness is not right handed");
        const double cosine=-gt::inner(gt::doublePoint(event.direction),gt::doublePoint(event.normal));
        const double px=gt::inner(gt::doublePoint(event.direction),gt::doublePoint(event.tangent));
        const double py=gt::inner(gt::doublePoint(event.direction),gt::doublePoint(event.bitangent));
        double sum=0;
        for(unsigned o=0;o<event.orderCount;++o) {
            const auto& order=event.orders[o];seen[unsigned(order.order+1)]=true;
            const double wavelength=description.spectrum.wavelengthsNanometres[event.band];
            const double x=px+order.order*wavelength/description.geometry.grooveSpacingNanometres*description.geometry.directionUv[0];
            const double y=py+order.order*wavelength/description.geometry.grooveSpacingNanometres*description.geometry.directionUv[1];
            const double z=std::sqrt(1-x*x-y*y),phase=2*std::acos(-1.0)*description.microstructure.grooveDepthNanometres*(cosine+z)/wavelength;
            const double duty=description.microstructure.dutyCycle;
            const double efficiency=order.order==0
                ?((1-duty)*(1-duty)+duty*duty+2*duty*(1-duty)*std::cos(phase))*z/cosine
                :std::pow(std::sin(std::acos(-1.0)*order.order*duty)/(std::acos(-1.0)*order.order),2)
                    *2*(1-std::cos(phase))*z/cosine;
            const auto expected=gt::plus(gt::plus(gt::plus(gt::D3{},gt::doublePoint(event.tangent),x),gt::doublePoint(event.bitangent),y),gt::doublePoint(event.normal),z);
            for(unsigned a=0;a<3;++a)if(std::abs(expected[a]-order.direction[a])>3e-5)
                return fail("Metal signed order vector differs from independent tangential grating equation");
            if(std::abs(double(order.power)-event.power*reflectance*efficiency)>event.power*2e-7)
                return fail("Metal physical band order flux differs from independent binary profile oracle");
            sum+=order.power;
        }
        if(!gc::nearPower(event.substrateReflectedPower,event.power*reflectance)
            ||!gc::nearPower(sum+event.unresolvedPower,event.substrateReflectedPower)
            ||!gc::nearPower(event.substrateReflectedPower+event.absorbedPower,event.power))
            return fail("Metal grating physical partition lost or normalized order flux");
    }
    if(!std::all_of(seen.begin(),seen.end(),[](bool value){return value;}))return fail("Metal actual grating prerequisite lacks nonzero zero and signed first orders");
    if(!staging.frame->readRawPass(renderpassoutput::Output::Color,after,error)||before.bytes!=after.bytes)
        return fail("Metal incident prerequisite changed staging Color without a full path ledger");
    gc::NativeGratingReadback repeated;
    if(!arbitgpu::produceMetalGlassGratingIncidents(staging.frame,binding,repeated,error))return false;
    if(repeated.incidents.size()!=receipt.incidents.size())return fail("Metal repeated actual grating incident count changed");
    for(std::size_t i=0;i<receipt.incidents.size();++i) {
        const auto& a=receipt.incidents[i];const auto& b=repeated.incidents[i];
        if(a.ancestry!=b.ancestry||a.hit.position!=b.hit.position||a.uv!=b.uv||a.orderCount!=b.orderCount)
            return fail("Metal repeated actual grating lineage or witness changed");
        for(unsigned o=0;o<a.orderCount;++o)if(a.orders[o].order!=b.orders[o].order||a.orders[o].direction!=b.orders[o].direction||a.orders[o].power!=b.orders[o].power)
            return fail("Metal repeated actual signed order bytes changed");
    }
    auto foreignReceipt=receipt;++foreignReceipt.submission.scratchGeneration;
    if(gc::validateGratingIncidentReadback(*binding,foreignReceipt,receipt.submission,error)||error.empty())
        return fail("Metal actual grating readback accepted a foreign native scratch owner");
    foreignReceipt=receipt;foreignReceipt.submission.completedFence=0;
    if(gc::validateGratingIncidentReadback(*binding,foreignReceipt,receipt.submission,error)||error.empty())
        return fail("Metal actual grating readback accepted an incomplete native fence");
    auto foreign=std::make_shared<gc::GratingIncidentBinding>(*binding);foreign->identity+="changed";
    auto sentinel=receipt;
    if(arbitgpu::produceMetalGlassGratingIncidents(staging.frame,foreign,sentinel,error)||error.empty()
        ||sentinel.incidents.size()!=receipt.incidents.size())return fail("Metal foreign frozen incident identity published a batch");
    auto highDescription=description;highDescription.microstructure.grooveDepthNanometres=125;
    const auto high=diffractionmaterial::admit(highDescription,error);if(!high)return false;
    const auto highMaterial=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*high);
    grating->diffractionPaths[0].material=diffractionmaterial::physicalcheckpoint::makeGpuParameters(*highMaterial,{});
    std::shared_ptr<const gc::GratingIncidentBinding> highBinding;
    if(!gc::prepareGratingIncidentBinding(prepared,fixture.frame,grating,highMaterial,.25,highBinding,error))return false;
    if(arbitgpu::produceMetalGlassGratingIncidents(staging.frame,highBinding,sentinel,error)||error.empty())
        return fail("Metal high missing-order tail published an apparently complete grating batch");
    gc::NativeReadback returned;gc::EnergyLedger returnedEnergy;
    if(arbitgpu::produceMetalGlassGratingTerminalReadback(staging.frame,binding,returned,returnedEnergy,error)||error.empty())
        return fail("Metal terminal continuation accepted a queried return into glass");
    auto guarded=fixture.frame;guarded.gratingIncidentBinding=binding;
    if(gc::validateTables(prepared,guarded,error))return fail("Metal prerequisite relaxed combined Color admission");
    return metalDirectGratingIrradianceCases(error);
}

bool metalGlassGratingTerminalCases(arbitgpu::NativeFixtureSceneBackend& backend,std::string& error,
    bool directional=false,unsigned cameraCase=0,bool cellCase=false,bool coverageCase=false)
{
    namespace gc=arbitgpu::glasscaustics;namespace gt=glasscaustictests;
    const auto fail=[&](const char* message){error=message;return false;};
    auto fixture=gt::ordinaryFixture();fixture.scene->lights[0].transform.rotation={0,0,0,1};
    if(coverageCase) {
        // Closed triangular prism: each entry/exit face is one triangle. Side
        // faces have true planar separating axes for strict-interior bundles.
        const auto prism=videohelper::tests::sceneclosest::slab(true,true);
        auto& glass=fixture.scene->objects[0];glass.vertexCount=prism->vertexCount;glass.indexCount=prism->indexCount;
        for(std::size_t v=0;v<prism->vertexCount;++v)fixture.scene->vertices[glass.firstVertex+v].position=prism->vertices[v].position;
        for(std::size_t i=0;i<prism->indexCount;++i)fixture.scene->indices[glass.firstIndex+i]=prism->indices[i];
    }
    if(directional) {
        auto& source=fixture.scene->lights[0].spectralSource;
        source.directionalIrradiance={1,diffractionmaterial::DirectionalSpectralMeasure::IrradianceCoefficient,source.radiance};
        source.radiance.fill(0);
        if(cameraCase)for(auto& value:source.directionalIrradiance.coefficients)value*=1e-5f;
    }
    fixture.scene->cameras[0].transform.translation={0,0,1};
    fixture.scene->cameras[0].transform.rotation={0,float(std::sqrt(.5)),0,float(std::sqrt(.5))};
    if(cameraCase>=2) {
        fixture.scene->cameras[0].transform.translation={-1.3f,0,cameraCase==2?-3.0f:3.0f};
        fixture.scene->cameras[0].transform.rotation={0,0,0,1};
        fixture.scene->cameras[0].verticalFovRadians=1.5f;
    }
    const auto target=fixture.scene->objects[1];
    for(std::size_t v=target.firstVertex;v<target.firstVertex+target.vertexCount;++v) {
        auto& vertex=fixture.scene->vertices[v];vertex.position.z=-6+.3f*vertex.position.x;
        vertex.normal={float(-.3/std::sqrt(1.09)),0,float(1/std::sqrt(1.09))};vertex.tangent={0,1,0,1};
        vertex.uv={vertex.position.y/12,-vertex.position.x/12};
    }
    if(coverageCase) {
        // A single retained grating triangle avoids an inseparable coplanar
        // sibling AABB. Authored cells and branch energy remain unchanged.
        auto& triangle=fixture.scene->objects[1];triangle.vertexCount=3;triangle.indexCount=3;
        const HarmonicMIDI::grid::SceneVec3 p[]={{-6,-6,-7.8f},{6,-6,-4.2f},{0,6,-6}};
        for(unsigned v=0;v<3;++v) {
            auto& vertex=fixture.scene->vertices[triangle.firstVertex+v];vertex.position=p[v];
            vertex.uv={p[v].y/12,-p[v].x/12};fixture.scene->indices[triangle.firstIndex+v]=v;
        }
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
    receiverProgram->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;
    receiverProgram->object=receiver.id;receiverProgram->authoredMaterial=receiver.material;
    receiverProgram->parameters.identifiers[0]=receiver.material.value;
    receiverProgram->programIdentity="metal-grating-terminal-receiver";receiverProgram->bindingDigest="metal-grating-terminal-receiver";
    receiverProgram->spectralReflectance->coefficients={.1f,.2f,.3f,.4f,.5f,.6f,.7f,.8f};
    fixture.program->objectPrograms.push_back(receiverProgram);
    fixture.program->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;
    for(auto& child:fixture.program->objectPrograms) {
        auto material=std::make_shared<gt::Program>(*child);material->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;child=material;
    }
    gc::Prepared prepared;
    arbitgpu::NativeFixtureSceneRuntimeInputs runtime;runtime.linearColor=true;runtime.timeSeconds=.25f;
    gc::SchedulingDescription scheduling;scheduling.entrySubdivisions=cellCase?4:1;
    auto description=diffractionmaterial::makeAluminiumBinaryGratingPreset();description.spectrum.firstOrder=description.spectrum.lastOrder=1;
    description.microstructure.grooveDepthNanometres=.9f;description.roughness={0,0};
    const auto admitted=diffractionmaterial::admit(description,error);if(!admitted)return false;
    const auto material=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*admitted);
    const auto compileGrating=[&](const diffractionmaterial::Description& physical,const std::string& digest) {
        diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest request;
        request.version=diffractionmaterialbinding::kSceneReceiverWireVersion;request.scene=fixture.scene->id;request.object=target.id;
        request.sceneRevision=request.structuralRevision=request.evaluationRevision=request.materialRevision=1;
        request.material=physical;request.structuralDigest=digest;
        request.lighting.version=3;request.lighting.pathCount=1;request.lighting.traceMaximumDistance=fixture.scene->lights[0].spectralSource.traceMaximumDistance;
        request.lighting.sceneReceiverTransport=diffractionmaterial::SceneReceiverTransport::AuthoredSpectralLambertian;
        request.lighting.paths[0].kind=diffractionmaterial::LightingPathKind::Direct;
        request.lighting.paths[0].incident.direction={0,0,1};
        request.lighting.paths[0].incident.radiance=fixture.scene->lights[0].spectralSource.radiance;
        return gc::compileDiscreteGrating(*fixture.scene,request,arbitgpu::NativeFixtureMaterialBackend::Metal,error);
    };
    const auto grating=compileGrating(description,material->structuralDigest());if(!grating)return false;
    // Version-zero identity values remain empty while every retained light,
    // including inactive slots, owns heap capacity outside the scene record.
    for(auto& light:fixture.scene->lights) {
        if(light.environmentImage.version||!light.environmentImage.asset.id.empty()
            ||!light.environmentImage.asset.contentSha256.empty())
            return fail("Metal mixed source fixture requires empty environment identities");
        light.environmentImage.asset.id.reserve(64);
        light.environmentImage.asset.contentSha256.reserve(96);
    }
    const auto prepareMixed=[&](const std::shared_ptr<const gt::Program>& leaf,
        const std::shared_ptr<const diffractionmaterial::AdmittedDiffractionMaterialIR>& ir,
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame,
        std::shared_ptr<const gc::GratingIncidentBinding>& frozen) {
        auto root=std::make_shared<gt::Program>(*leaf);
        auto glass=std::make_shared<gt::Program>(*fixture.program);glass->objectPrograms.clear();
        root->objectPrograms={glass,receiverProgram};
        if(backend.prepare(fixture.scene,root).prepared)return fail("Metal default finite-lobe admission accepted discrete mixed grating");
        arbitgpu::smoothglass::DistinctGratingAdmission selection{target.id,ir};
        const auto resources=arbitgpu::prepareMetalGlassGratingResources(fixture.scene,root,selection,leaf);
        if(!resources.prepared){error=resources.error;return false;}
        if(!arbitgpu::metalPrivateSurfaceSourceCreationCasesForFixture(resources.resources,receiver.id,error)
            ||!arbitgpu::prepareMetalPrivateSurfaceSourceForFixture(resources.resources,receiver.id,error)
            ||!arbitgpu::metalPrivateSurfaceSourceCreationCasesForFixture(resources.resources,receiver.id,error)) return false;
        if(backend.render(fixture.scene,resources.resources,32,32,runtime).rendered)
            return fail("Metal mixed private resources entered the default Color renderer");
        if(!arbitgpu::stageMetalGlassGratingFrame(resources.resources,fixture.ordinaryLight,*fixture.policy,
            32,32,runtime,.25,frame,frozen,error,scheduling))return false;
        if(!arbitgpu::inspectMetalGlassGratingCameraAllocationForFixture(frame,frozen,error))return false;
        const auto savedFrame=frame;const auto savedBinding=frozen;
        auto staleTime=runtime;staleTime.timeSeconds=.5f;
        if(arbitgpu::stageMetalGlassGratingFrame(resources.resources,fixture.ordinaryLight,*fixture.policy,
            32,32,staleTime,.25,frame,frozen,error,scheduling)||error.empty()||frame!=savedFrame||frozen!=savedBinding)
            return fail("Metal pending camera allocation accepted stale time or changed an old owner on refusal");
        auto nonLinear=runtime;nonLinear.linearColor=false;
        if(arbitgpu::stageMetalGlassGratingFrame(resources.resources,fixture.ordinaryLight,*fixture.policy,
            32,32,nonLinear,.25,frame,frozen,error,scheduling)||error.empty()||frame!=savedFrame||frozen!=savedBinding)
            return fail("Metal pending camera allocation accepted non-linear base or changed an old owner on refusal");
        error.clear();return true;
    };
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> staging;
    std::shared_ptr<const gc::GratingIncidentBinding> binding;
    if(!prepareMixed(grating,material,staging,binding))return false;
    prepared=binding->prepared;fixture.frame=binding->frame;
    if(!fixture.frame.gratingStaging||!prepared.glass.distinctGrating
        ||prepared.glass.distinctGrating->object!=target.id
        ||prepared.glass.sourceMaterial->kind!=arbitgpu::NativeFixtureMaterialKind::DiffractionReflective
        ||prepared.glass.sourceScene==fixture.scene||staging->colorImageHandle()!=0||staging->depthImageHandle()!=0)
        return fail("Metal staging did not retain the actual immutable mixed owner without public output");
    gc::Frame unpublishedSchedule;gc::NativeReadback unpublished;gc::EnergyLedger unpublishedEnergy;
    if(arbitgpu::readMetalGlassCausticReceipt(staging,unpublishedSchedule,unpublished,unpublishedEnergy,error))
        return fail("Metal completed staging fence was presented as a physical glass ledger");
    arbitgpu::NativeRawPassPixels absent;
    if(staging->readRawPass(renderpassoutput::Output::Color,absent,error))return fail("Metal staging exposed Color before completed physical proof");
    const auto targetTriangle=std::find_if(fixture.frame.triangles.begin(),fixture.frame.triangles.end(),
        [&](const auto& triangle){return triangle.object==target.id.value;});
    if(targetTriangle==fixture.frame.triangles.end()||targetTriangle->reflectance)
        return fail("Metal mixed grating was replaced with a Surface staging receiver");
    const auto produce=cameraCase?arbitgpu::produceMetalGlassGratingCameraReadback
        :arbitgpu::produceMetalGlassGratingTerminalReadback;
    gc::NativeReadback receipt;gc::EnergyLedger ledger;
    if(!produce(staging,binding,receipt,ledger,error))return false;
    if(!receipt.gratingIncidents||!receipt.gratingIncidents->childrenComplete||receipt.gratingIncidents->incidents.empty()
        ||receipt.gratingIncidents->paths.size()!=fixture.frame.work.size())return fail("Metal terminal grating lacks dense native child proof");
    const auto* sourceBinding=directional?dynamic_cast<const gc::NativeDirectionalSourceBinding*>(staging.get()):nullptr;
    if(directional) {
        if(!sourceBinding||&sourceBinding->owner()!=staging.get()||!sourceBinding->upload().complete
            ||!receipt.directionalSource||receipt.directionalSource!=receipt.gratingIncidents->directionalSource
            ||receipt.directionalSource->echo.input.identifiers!=gc::directionalSourceInput(prepared,fixture.frame).identifiers
            ||std::any_of(receipt.directionalSource->echo.rawRadiance.begin(),receipt.directionalSource->echo.rawRadiance.end(),[](float v){return v!=0;}))
            return fail("Metal directional transport lacks actual raw-zero source upload and shader proof");
        if(!gc::validateDirectionalSourceReadback(prepared,fixture.frame,*sourceBinding,receipt.gratingIncidents->submission,
            *receipt.directionalSource,error,true,binding.get()))return false;
    } else if(receipt.directionalSource||receipt.gratingIncidents->directionalSource)
        return fail("Metal legacy transport acquired a directional source grant");
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
            for(unsigned a=0;a<3;++a)if(std::abs(direction[a]-order.direction[a])>3e-5)return fail("Metal terminal signed vector differs from independent grating equation");
            if(std::abs(order.power-parent.power*reflectance*efficiency)>parent.power*2e-7)
                return fail("Metal terminal physical band flux differs from independent binary profile");
        }
    }
    for(const auto& child:receipt.gratingIncidents->children) {
        const auto& parent=receipt.gratingIncidents->incidents[child.incident];orders[parent.band][unsigned(child.order+1)]=true;
        const auto order=std::find_if(parent.orders.begin(),parent.orders.begin()+parent.orderCount,[&](const auto& o){return o.order==child.order;});
        if(order==parent.orders.begin()+parent.orderCount)return fail("Metal terminal child lost its signed parent order");
        const auto origin=gt::plus(gt::doublePoint(parent.hit.position),gt::doublePoint(parent.normal),gc::bias(parent.hit.position));
        const auto oracle=gt::closest(triangles,origin,gt::doublePoint(order->direction),1e-5,fixture.frame.sourceDirectionTrace[3]);
        if(child.disposition==gc::NativeGratingReadback::Disposition::Receiver) {
            if(!oracle.found||oracle.object!=child.hit.objectId||oracle.triangle!=child.hit.triangleId
                ||child.hit.objectId!=receiver.id.value)return fail("Metal terminal order differs from independent authored receiver query");
            for(unsigned a=0;a<3;++a)if(std::abs(oracle.position[a]-child.hit.position[a])>5e-4)
                return fail("Metal terminal receiver point differs from independent authored query");
            const auto d=std::find_if(receipt.deposits.begin(),receipt.deposits.end(),[&](const auto& v){return v.gratingVersion&&v.gratingIncident==child.incident&&v.gratingOrder==child.order;});
            if(d==receipt.deposits.end()||d->ancestry!=parent.ancestry||!gc::nearPower(d->incidentPower,order->power)
                ||!gc::nearPower(d->outgoingPower,order->power*(.1+.1*parent.band)))return fail("Metal terminal receiver lost exact ancestry or physical band rho");
            deposited=true;visible|=d->cameraVisible;
        } else {
            if(child.disposition!=gc::NativeGratingReadback::Disposition::Escaped||oracle.found
                ||!gc::nearPower(child.escapedPower,order->power))return fail("Metal terminal miss was not an actual independently queried escape");
            escaped=true;
        }
    }
    for(const auto& band:orders)for(bool value:band)if(!value)return fail("Metal terminal fixture lacks a nonzero signed order in every band");
    if(!deposited||(!cameraCase&&!visible)||!escaped||(!cameraCase&&std::none_of(receipt.spectralRadiance.begin(),receipt.spectralRadiance.end(),[](float value){return value>0;})))
        return fail("Metal terminal fixture lacks native receiver deposition and spectral accumulation");
    for(unsigned b=0;b<8;++b)if(!gc::nearPower(ledger.launched[b],ledger.deposited[b]+ledger.absorbed[b]+ledger.escaped[b]+ledger.unresolved[b])
        ||ledger.unresolved[b]>ledger.launched[b]*1e-4*(1+2e-4))return fail("Metal terminal original energy ledger did not close");
    auto bound=fixture.frame;bound.gratingIncidentBinding=binding;bound.identity=binding->identity;
    gc::EnergyLedger ignored;
    if(gc::validateNativeReadback(prepared,bound,receipt,ignored,error,&receipt.gratingIncidents->submission))
        return fail("Metal terminal prerequisite enabled default combined Color");
    auto broken=receipt;auto proof=std::make_shared<gc::NativeGratingReadback>(*receipt.gratingIncidents);proof->children.pop_back();broken.gratingIncidents=proof;
    if(gc::validateNativeReadback(prepared,bound,broken,ignored,error,&proof->submission,true,sourceBinding))return fail("Metal terminal accepted incomplete child proof");
    broken=receipt;const auto coupled=std::find_if(broken.deposits.begin(),broken.deposits.end(),[](const auto& value){return value.gratingVersion!=0;});
    if(coupled==broken.deposits.end())return fail("Metal terminal fixture has no exact coupled deposit");coupled->ancestry[0]^=64;
    if(gc::validateNativeReadback(prepared,bound,broken,ignored,error,&receipt.gratingIncidents->submission,true,sourceBinding))return fail("Metal terminal accepted wrong parent ancestry");
    gc::NativeReadback repeated;gc::EnergyLedger repeatEnergy;
    if(!produce(staging,binding,repeated,repeatEnergy,error))return false;
    if(receipt.spectralRadiance!=repeated.spectralRadiance)return fail("Metal terminal repeated native accumulation changed");
    if(staging->colorImageHandle()!=0||staging->readRawPass(renderpassoutput::Output::Color,absent,error))
        return fail("Metal terminal prerequisite published partial or completed Color");
    if(cameraCase) {
        if(!receipt.gratingCamera||receipt.gratingCameraVersion!=gc::kGratingCameraReceiptVersion
            ||!receipt.gratingCamera->complete||receipt.gratingCamera->physicalColor
            ||receipt.gratingCamera->incidents!=receipt.gratingIncidents
            ||!(receipt.gratingCamera->submission==receipt.gratingIncidents->submission)
            ||receipt.gratingCamera->orders.empty()||!repeated.gratingCamera
            ||receipt.gratingCamera->orders.size()!=repeated.gratingCamera->orders.size())
            return fail("Metal private camera lost native incident/fence ownership or acquired Color");
        const auto wanted=cameraCase==1?gc::NativeGratingCameraReadback::Outcome::OutsideCamera
            :cameraCase==2?gc::NativeGratingCameraReadback::Outcome::Visible:gc::NativeGratingCameraReadback::Outcome::Occluded;
        std::array<bool,8> bandComplete{};
        for(std::size_t i=0;i<receipt.gratingCamera->orders.size();++i) {
            const auto& r=receipt.gratingCamera->orders[i];const auto& again=repeated.gratingCamera->orders[i];
            if(!r.complete||r.band>7||r.order<-1||r.order>1||r.spectralRadiance!=again.spectralRadiance
                ||r.outcome!=again.outcome||r.x!=again.x||r.y!=again.y)
                return fail("Metal private camera incomplete band/order or nondeterministic native draw");
            if(r.outcome==wanted)bandComplete[r.band]=true;
        }
        for(bool complete:bandComplete)if(!complete)return fail("Metal camera fixture omitted expected actual visibility outcome in a band");
        const auto corruptCamera=[&](auto mutate) {
            auto broken=receipt;auto camera=std::make_shared<gc::NativeGratingCameraReadback>(*receipt.gratingCamera);
            mutate(*camera);broken.gratingCamera=camera;
            return !gc::validateNativeReadback(prepared,bound,broken,ignored,error,
                &receipt.gratingIncidents->submission,true,sourceBinding)&&!error.empty();
        };
        if(!corruptCamera([](auto& r){r.submission.completedFence=0;})
            ||!corruptCamera([](auto& r){++r.submission.scratchGeneration;})
            ||!corruptCamera([](auto& r){r.orders.pop_back();})
            ||!corruptCamera([](auto& r){r.orders[0].complete=false;})
            ||!corruptCamera([](auto& r){r.physicalColor=true;})
            ||!corruptCamera([](auto& r){r.incidents=std::make_shared<const gc::NativeGratingReadback>(*r.incidents);}))
            return fail("Metal private camera accepted stale owner/fence, incomplete records or unproved finite-cell Color");
    }
    if(cellCase) {
        gc::NativeReadback cameraIncumbent;gc::EnergyLedger cameraEnergy;
        if(!arbitgpu::produceMetalGlassGratingCameraReadback(staging,binding,cameraIncumbent,cameraEnergy,error)
            ||!arbitgpu::inspectMetalGlassGratingCameraPrimitiveForFixture(staging,cameraIncumbent.gratingCamera,error))return false;
        std::shared_ptr<const gc::GratingEntryCellBinding> cells;
        if(!gc::prepareGratingEntryCellBinding(binding,cells,error))return false;
        gc::NativeReadback boundaryPoint;gc::EnergyLedger boundaryEnergy;gc::NativeGratingCellTransportReadback boundary;
        if(!arbitgpu::produceMetalGlassGratingCellTransportReadback(staging,binding,cells,boundaryPoint,boundaryEnergy,boundary,error))return false;
        if(!boundary.complete||boundary.overflow||!boundary.binding||boundary.records.empty()
            ||boundary.binding->sourceCells!=cells||boundary.binding->pointReadback!=boundaryPoint.gratingIncidents
            ||!gc::gratingCellSubmissionContinuity(boundaryPoint.gratingIncidents->submission,boundary.submission)
            ||boundaryPoint.spectralRadiance!=repeated.spectralRadiance
            ||boundary.records.size()!=boundaryPoint.gratingIncidents->incidents.size())
            return fail("Metal supplemental cell receipt replaced point transport or lost dense native parent/fence linkage");
        if(boundaryPoint.gratingIncidents->queryWork!=receipt.gratingIncidents->queryWork
            ||boundaryPoint.gratingIncidents->orderWork!=receipt.gratingIncidents->orderWork
            ||boundaryPoint.paths.size()!=receipt.paths.size())
            return fail("Metal optional cell queries leaked into the original point counters");
        for(std::size_t w=0;w<receipt.paths.size();++w)if(boundaryPoint.paths[w].queryWork!=receipt.paths[w].queryWork)
            return fail("Metal optional cell queries changed a point terminal path counter");
        if(!arbitgpu::inspectMetalGlassGratingCameraPrimitiveForFixture(staging,cameraIncumbent.gratingCamera,error))return false;
        const auto* nativeCells=dynamic_cast<const gc::NativeGratingCellSourceBinding*>(staging.get());
        if(!nativeCells||&nativeCells->owner()!=staging.get()||!nativeCells->cellUpload().complete
            ||nativeCells->cellUpload().sourceCells!=cells||nativeCells->cellUpload().pointReadback!=boundaryPoint.gratingIncidents
            ||nativeCells->cellUpload().cellTexels!=cells->cellTexels
            ||!(nativeCells->cellUpload().submission==boundary.submission))
            return fail("Metal supplemental cell source lacks actual completed dedicated native echo");
        std::array<bool,8> supportedBands{};bool unsupported=false;
        for(const auto& record:boundary.records) {
            if(!record.complete||record.paths[0].size()!=record.interfaces+1||!record.queryWork)
                return fail("Metal supplemental cell lost actual center interface lineage");
            if(record.disposition==gc::GratingCellTransportDisposition::BoundaryWitness) {
                supportedBands[record.band]=true;
                for(const auto& path:record.paths)if(path.size()!=record.interfaces+1)
                    return fail("Metal native boundary witness omitted a corner endpoint");
            } else if(record.disposition==gc::GratingCellTransportDisposition::Unsupported)unsupported=true;
            else return fail("Metal native cell used an unregistered disposition");
        }
        for(bool supported:supportedBands)if(!supported)return fail("Metal refined native cell fixture lacks a supported boundary in every band");
        if(!unsupported)return fail("Metal native edge-cell mismatch was hidden as a supported finite footprint");
        const auto corruptBoundary=[&](auto mutate) {
            auto broken=boundary;mutate(broken);
            return !gc::validateGratingCellTransportReadback(*boundary.binding,broken,staging,boundary.submission,error)&&!error.empty();
        };
        if(!corruptBoundary([](auto& r){r.submission.completedFence=0;})
            ||!corruptBoundary([](auto& r){r.submission.submissionSerial=r.binding->pointSubmission.submissionSerial;})
            ||!corruptBoundary([](auto& r){r.records.pop_back();})
            ||!corruptBoundary([](auto& r){r.records[0].paths[0][0].origin[0]+=1;})
            ||!corruptBoundary([](auto& r){r.records[0].paths[0][0].signedBias*=-1;})
            ||!corruptBoundary([](auto& r){r.records[0].paths[0][0].token^=64;})
            ||!corruptBoundary([](auto& r){++r.queryWork;})
            ||!corruptBoundary([](auto& r){r.binding=std::make_shared<const gc::GratingCellTransportBinding>(*r.binding);}))
            return fail("Metal cell transport accepted stale owner/fence, incomplete boundary, altered ray/bias/branch or wrong work");
        const auto oldBinding=boundary.binding;const auto oldPoint=boundaryPoint.gratingIncidents;
        auto forgedCells=std::make_shared<gc::GratingEntryCellBinding>(*cells);forgedCells->cellTexels[0][0]+=1;
        if(arbitgpu::produceMetalGlassGratingCellTransportReadback(staging,binding,forgedCells,boundaryPoint,boundaryEnergy,boundary,error)
            ||error.empty()||boundary.binding!=oldBinding||boundaryPoint.gratingIncidents!=oldPoint
            ||nativeCells->cellUpload().pointReadback!=oldPoint)
            return fail("Metal forged cell source mutated its incumbent native boundary/point owners on refusal");
        if(!arbitgpu::inspectMetalGlassGratingCameraPrimitiveForFixture(staging,cameraIncumbent.gratingCamera,error))return false;
        gc::NativeReadback plainAfter;gc::EnergyLedger plainEnergy;
        if(!arbitgpu::produceMetalGlassGratingTerminalReadback(staging,binding,plainAfter,plainEnergy,error)
            ||plainAfter.spectralRadiance!=repeated.spectralRadiance||plainAfter.gratingCamera
            ||nativeCells->cellUpload().pointReadback!=oldPoint)
            return fail("Metal optional cell work changed default terminal counters, pixels or its retained cell source owner");
        if(staging->colorImageHandle()||staging->depthImageHandle()
            ||staging->readRawPass(renderpassoutput::Output::Color,absent,error))
            return fail("Metal native corner boundaries granted unproved interior or base Color");
        // The caller's old receipt is the sole owner of its previous frame.
        // Replacement must release that frame after the backend mutex unlocks.
        std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> priorFrame;
        std::shared_ptr<const gc::GratingIncidentBinding> priorParent;
        std::shared_ptr<const gc::GratingEntryCellBinding> priorCells;
        gc::NativeReadback replacingPoint;gc::EnergyLedger replacingEnergy;gc::NativeGratingCellTransportReadback replacing;
        if(!prepareMixed(grating,material,priorFrame,priorParent)
            ||!gc::prepareGratingEntryCellBinding(priorParent,priorCells,error)
            ||!arbitgpu::produceMetalGlassGratingCellTransportReadback(priorFrame,priorParent,priorCells,
                replacingPoint,replacingEnergy,replacing,error))return false;
        const std::weak_ptr<const arbitgpu::NativeFixtureSceneFrame> retiredFrame=priorFrame;
        priorFrame.reset();priorParent.reset();priorCells.reset();
        const auto priorPoint=replacingPoint.gratingIncidents;
        if(arbitgpu::produceMetalGlassGratingCellTransportReadback(staging,binding,forgedCells,
                replacingPoint,replacingEnergy,replacing,error)||error.empty()||retiredFrame.expired()
            ||replacingPoint.gratingIncidents!=priorPoint||replacing.binding->nativeOwner!=retiredFrame.lock())
            return fail("Metal invalid new cell source destroyed the caller's sole previous frame owner");
        // Explicit real-cap stress only. Ordinary fixtures allocate small
        // native scenes; this selection reserves approximately 2 GiB per case.
        const auto* stressSelection=std::getenv("ARBIT_METAL_CELL_ADMISSION_STRESS");
        const auto stressCopiedOutputs=[&](const gc::NativeReadback& originalPoint,const gc::EnergyLedger& originalEnergy,
            const gc::NativeGratingCellTransportReadback& originalCells) {
            if(stressSelection&&std::strcmp(stressSelection,"1")==0)for(unsigned storage=0;storage<3;++storage) {
                auto stressPoint=originalPoint;auto stressCell=originalCells;auto stressEnergy=originalEnergy;
                const auto maximum=renderpassoutput::kMaximumTotalBytes;
                if(storage==0)stressPoint.spectralRadiance.reserve(maximum/sizeof(float));
                else if(storage==1)stressCell.records.reserve(maximum/sizeof(gc::NativeGratingCellTransportRecord));
                else stressCell.records[0].paths[0].reserve(maximum/sizeof(gc::NativeGratingCellTraceStep));
                const auto* pointStorage=stressPoint.spectralRadiance.data();const auto* cellStorage=stressCell.records.data();
                const auto* pathStorage=stressCell.records[0].paths[0].data();
                const auto pointCapacity=stressPoint.spectralRadiance.capacity(),cellCapacity=stressCell.records.capacity();
                const auto pathCapacity=stressCell.records[0].paths[0].capacity();
                if(arbitgpu::produceMetalGlassGratingCellTransportReadback(staging,binding,cells,
                        stressPoint,stressEnergy,stressCell,error)
                    ||(error!="Metal cell initial simultaneous incumbents exceed the existing byte cap"
                        &&error!="Metal grating complete scratch and retained owners exceed frame memory")
                    ||stressCell.binding!=originalCells.binding
                    ||!(stressCell.submission==originalCells.submission)||stressPoint.gratingIncidents!=originalPoint.gratingIncidents
                    ||stressPoint.directionalSource!=originalPoint.directionalSource
                    ||stressPoint.spectralRadiance!=originalPoint.spectralRadiance
                    ||stressPoint.spectralRadiance.data()!=pointStorage||stressCell.records.data()!=cellStorage
                    ||stressCell.records[0].paths[0].data()!=pathStorage
                    ||stressPoint.spectralRadiance.capacity()!=pointCapacity||stressCell.records.capacity()!=cellCapacity
                    ||stressCell.records[0].paths[0].capacity()!=pathCapacity
                    ||stressEnergy.potential!=originalEnergy.potential||stressEnergy.launched!=originalEnergy.launched
                    ||stressEnergy.sourceBlocked!=originalEnergy.sourceBlocked||stressEnergy.deposited!=originalEnergy.deposited
                    ||stressEnergy.absorbed!=originalEnergy.absorbed||stressEnergy.escaped!=originalEnergy.escaped
                    ||stressEnergy.unresolved!=originalEnergy.unresolved||stressEnergy.cameraVisible!=originalEnergy.cameraVisible
                    ||stressEnergy.masked!=originalEnergy.masked)
                    return fail("Metal near-cap caller storage escaped initial point admission or changed an incumbent on refusal");
                if(!arbitgpu::inspectMetalGlassGratingCameraPrimitiveForFixture(staging,cameraIncumbent.gratingCamera,error))return false;
            }
            return true;
        };
        if(!stressCopiedOutputs(replacingPoint,replacingEnergy,replacing))return false;
        if(!arbitgpu::produceMetalGlassGratingCellTransportReadback(staging,binding,cells,
                replacingPoint,replacingEnergy,replacing,error))return false;
        if(!retiredFrame.expired()||replacing.binding->nativeOwner!=staging
            ||replacing.binding->pointReadback!=replacingPoint.gratingIncidents
            ||!arbitgpu::inspectMetalGlassGratingCameraPrimitiveForFixture(staging,cameraIncumbent.gratingCamera,error))
            return fail("Metal new cell candidate failed to replace and retire its sole previous frame owner");
        // Copies retain parent pointers and allocate independent value storage.
        // Keep originals alive while inspecting the actual production charge.
        auto copiedPoint=replacingPoint;auto copiedCells=replacing;auto copiedEnergy=replacingEnergy;
        if(copiedPoint.gratingIncidents!=replacingPoint.gratingIncidents||copiedCells.binding!=replacing.binding
            ||copiedPoint.spectralRadiance.data()==replacingPoint.spectralRadiance.data()
            ||copiedPoint.paths.data()==replacingPoint.paths.data()||copiedCells.records.data()==replacing.records.data())
            return fail("Metal copied-output fixture did not retain shared parents with independent value storage");
        std::uint64_t valueBytes=copiedCells.records.capacity()*sizeof(gc::NativeGratingCellTransportRecord)
            +copiedPoint.frameIdentity.capacity()+copiedPoint.paths.capacity()*sizeof(gc::NativePathReceipt)
            +copiedPoint.deposits.capacity()*sizeof(gc::NativeDeposit)+copiedPoint.spectralRadiance.capacity()*sizeof(float)
            +copiedPoint.deposition.keys.capacity()*sizeof(gc::DepositionKey);
        for(std::size_t r=0;r<copiedCells.records.size();++r)for(unsigned path=0;path<gc::kGratingCellTracePaths;++path) {
            const auto& copy=copiedCells.records[r].paths[path];const auto& original=replacing.records[r].paths[path];
            if(!copy.empty()&&copy.data()==original.data())return fail("Metal copied cell paths share value storage");
            valueBytes+=copy.capacity()*sizeof(gc::NativeGratingCellTraceStep);
        }
        gc::NativeReadback parentsOnly;parentsOnly.gratingIncidents=copiedPoint.gratingIncidents;
        parentsOnly.directionalSource=copiedPoint.directionalSource;parentsOnly.gratingCamera=copiedPoint.gratingCamera;
        gc::NativeGratingCellTransportReadback cellParentOnly;cellParentOnly.binding=copiedCells.binding;
        std::uint64_t parentBytes=0,copiedBytes=0;
        if(!arbitgpu::inspectMetalGlassGratingCellIncumbentBytesForFixture(staging,cells,parentsOnly,cellParentOnly,parentBytes,error)
            ||!arbitgpu::inspectMetalGlassGratingCellIncumbentBytesForFixture(staging,cells,copiedPoint,copiedCells,copiedBytes,error))return false;
        if(copiedBytes<parentBytes||copiedBytes-parentBytes!=valueBytes-parentsOnly.frameIdentity.capacity())
            return fail("Metal same-frame shared parents credited independent copied point/record/path storage");
        if(!stressCopiedOutputs(copiedPoint,copiedEnergy,copiedCells))return false;
        const auto originalParent=replacing.binding;const auto originalPointParent=replacingPoint.gratingIncidents;
        const auto* originalPointStorage=replacingPoint.spectralRadiance.data();const auto* originalCellStorage=replacing.records.data();
        if(!arbitgpu::produceMetalGlassGratingCellTransportReadback(staging,binding,cells,
                copiedPoint,copiedEnergy,copiedCells,error))return false;
        if(!copiedCells.complete||copiedCells.binding==originalParent||copiedPoint.gratingIncidents==originalPointParent
            ||replacing.binding!=originalParent||replacingPoint.gratingIncidents!=originalPointParent
            ||replacingPoint.spectralRadiance.data()!=originalPointStorage||replacing.records.data()!=originalCellStorage
            ||copiedPoint.spectralRadiance!=replacingPoint.spectralRadiance
            ||!arbitgpu::inspectMetalGlassGratingCameraPrimitiveForFixture(staging,cameraIncumbent.gratingCamera,error))
            return fail("Metal same-frame copied-output replacement changed its still-live original owners");
        if(coverageCase) {
            auto nativeBoundary=std::make_shared<const gc::NativeGratingCellTransportReadback>(copiedCells);
            std::shared_ptr<const gc::GratingCellCoverageCandidate> coverage;
            std::shared_ptr<const gc::GratingCellCoveragePayload> payload;
            if(!arbitgpu::produceMetalGlassGratingCellCoverageReadback(staging,nativeBoundary,4,coverage,payload,error)
                ||!gc::validateNativeGratingCellCoverage(coverage,payload,staging,coverage->submission,error))return false;
            const auto* native=dynamic_cast<const gc::NativeGratingCellCoverageSource*>(staging.get());
            if(!native||native->coverageEvidence().candidate.lock()!=coverage||native->coverageEvidence().boundary.lock()!=nativeBoundary
                ||native->coverageEvidence().payload!=payload||!native->coverageEvidence().complete
                ||!gc::gratingCellSubmissionContinuity(nativeBoundary->submission,coverage->submission)
                ||payload->records.size()!=copiedCells.records.size()
                ||payload->disposition!=gc::GratingCellCoverageDisposition::Unsupported)
                return fail("Metal interval receipt lost exact later native evidence or hid unsupported source edges");
            std::array<bool,8> certifiedBands{};bool unsupported=false,faceAxis=false;
            for(std::size_t i=0;i<payload->records.size();++i) {
                const auto& record=payload->records[i];
                if(record.disposition==gc::GratingCellCoverageDisposition::InteriorCertified) {
                    certifiedBands[copiedCells.records[i].band]=true;
                    for(const auto& leaf:record.leaves)for(const auto& sweep:leaf.sweeps)for(const auto& blocker:sweep.blockers)
                        if((blocker.axis[0]!=0)+(blocker.axis[1]!=0)+(blocker.axis[2]!=0)>1)faceAxis=true;
                } else unsupported=true;
            }
            for(bool certified:certifiedBands)if(!certified)return fail("Metal planar interval fixture lacks strict-interior native proof in every band");
            if(!unsupported||!faceAxis)return fail("Metal interval producer hid edge refusal or omitted retained face-normal separation");
            // Copies retain actual native values and owners; capacity alone
            // must refuse before a new proof allocation or native submission.
            const auto refuseCopiedIncumbent=[&](unsigned storage,bool realCap) {
                auto copiedBoundary=std::make_shared<gc::NativeGratingCellTransportReadback>(*nativeBoundary);
                auto copiedCandidate=std::make_shared<gc::GratingCellCoverageCandidate>(*coverage);
                const auto maximum=renderpassoutput::kMaximumTotalBytes;
                if(storage==0)copiedBoundary->records.reserve(realCap?maximum/sizeof(gc::NativeGratingCellTransportRecord)
                    :nativeBoundary->binding->bounds.maximumRecords+1);
                else if(storage==1)copiedBoundary->records[0].paths[0].reserve(realCap?maximum/sizeof(gc::NativeGratingCellTraceStep)
                    :nativeBoundary->binding->bounds.maximumInterfaces+2);
                else if(storage==2)copiedCandidate->identity.reserve(maximum);
                else if(storage==3)copiedBoundary->submission.frameIdentity.reserve(maximum);
                else {
                    auto copiedBinding=std::make_shared<gc::GratingCellTransportBinding>(*nativeBoundary->binding);
                    copiedBinding->identity.reserve(maximum);copiedBoundary->binding=std::move(copiedBinding);
                }
                copiedCandidate->boundary=copiedBoundary;
                std::shared_ptr<const gc::GratingCellCoverageCandidate> incumbent=copiedCandidate;
                auto incumbentPayload=payload;
                const auto* records=copiedBoundary->records.data();const auto* path=copiedBoundary->records[0].paths[0].data();
                const auto recordCapacity=copiedBoundary->records.capacity(),pathCapacity=copiedBoundary->records[0].paths[0].capacity();
                const auto evidence=native->coverageEvidence();
                const char* expectedErrors[]={"Cell transport native candidate, fence or dense extent changed",
                    "Cell transport trace exceeds its retained depth or host reservation",
                    "Metal coverage incumbent candidate exceeds the existing byte cap",
                    "Metal coverage incumbent boundary exceeds the existing byte cap",
                    "Metal coverage incumbent shared binding exceeds the existing byte cap"};
                if(arbitgpu::produceMetalGlassGratingCellCoverageReadback(staging,nativeBoundary,4,incumbent,incumbentPayload,error)
                    ||error!=expectedErrors[storage]||incumbent!=copiedCandidate||incumbentPayload!=payload
                    ||copiedBoundary->records.data()!=records||copiedBoundary->records[0].paths[0].data()!=path
                    ||copiedBoundary->records.capacity()!=recordCapacity||copiedBoundary->records[0].paths[0].capacity()!=pathCapacity
                    ||native->coverageEvidence().candidate.lock()!=coverage||native->coverageEvidence().boundary.lock()!=nativeBoundary
                    ||native->coverageEvidence().payload!=evidence.payload
                    ||!(native->coverageEvidence().submission==evidence.submission)
                    ||!(native->coverageEvidence().boundarySubmission==evidence.boundarySubmission)
                    ||native->coverageEvidence().nativeBytes!=evidence.nativeBytes
                    ||native->coverageEvidence().bindingGeneration!=evidence.bindingGeneration
                    ||native->coverageEvidence().complete!=evidence.complete||native->coverageEvidence().overflow!=evidence.overflow)
                    return fail("Metal copied coverage incumbent capacity escaped initial proof admission or changed retained owners");
                return true;
            };
            if(!refuseCopiedIncumbent(0,false)||!refuseCopiedIncumbent(1,false))return false;
            // The existing explicit stress selection alone reserves about
            // 2 GiB per case. It supplies no fake metrics or native evidence.
            if(stressSelection&&std::strcmp(stressSelection,"1")==0)
                for(unsigned storage=0;storage<5;++storage)if(!refuseCopiedIncumbent(storage,true))return false;
            {
                const auto before=coverage;const auto beforePayload=payload;
                if(arbitgpu::produceMetalGlassGratingCellCoverageReadback(staging,nativeBoundary,3,coverage,payload,error)
                    ||error.empty()||coverage!=before||payload!=beforePayload||native->coverageEvidence().payload!=beforePayload)
                    return fail("Metal noncanonical interval partition changed its incumbent on refusal");
                auto changed=std::make_shared<gc::NativeGratingCellTransportReadback>(*nativeBoundary);
                auto& position=changed->records[0].paths[0][0].origin[0];position=std::nextafter(position,std::numeric_limits<float>::infinity());
                if(arbitgpu::produceMetalGlassGratingCellCoverageReadback(staging,changed,4,coverage,payload,error)
                    ||error.empty()||coverage!=before||payload!=beforePayload||native->coverageEvidence().payload!=beforePayload)
                    return fail("Metal interval producer accepted host-modified native boundary evidence");
                auto stale=std::make_shared<gc::GratingCellCoverageCandidate>(*coverage);stale->submission.completedFence=0;
                if(gc::validateNativeGratingCellCoverage(stale,payload,staging,coverage->submission,error)||error.empty())
                    return fail("Metal interval evidence accepted a stale candidate fence");
                const auto twin=std::make_shared<const gc::GratingCellCoverageCandidate>(*coverage);
                if(gc::validateNativeGratingCellCoverage(twin,payload,staging,coverage->submission,error)||error.empty())
                    return fail("Metal interval evidence accepted a copied candidate owner");
                auto broken=std::make_shared<gc::GratingCellCoveragePayload>(*payload);bool removed=false;
                for(auto& record:broken->records)for(auto& leaf:record.leaves)for(auto& sweep:leaf.sweeps)
                    if(!removed&&!sweep.blockers.empty()) {sweep.blockers.pop_back();removed=true;}
                if(!removed||gc::validateGratingCellCoverageShape(*coverage,*broken,error)||error.empty())
                    return fail("Metal interval shape accepted an omitted retained-triangle blocker");
            }
            if(!arbitgpu::inspectMetalGlassGratingCameraPrimitiveForFixture(staging,cameraIncumbent.gratingCamera,error))return false;
            const std::weak_ptr<const gc::GratingCellCoverageCandidate> weakCandidate=coverage;
            const std::weak_ptr<const gc::NativeGratingCellTransportReadback> weakBoundary=nativeBoundary;
            coverage.reset();nativeBoundary.reset();
            if(!weakCandidate.expired()||!weakBoundary.expired()||!native->coverageEvidence().candidate.expired()
                ||!native->coverageEvidence().boundary.expired()||native->coverageEvidence().payload!=payload)
                return fail("Metal interval frame retained a candidate/boundary cycle instead of weak lineage");
            if(staging->colorImageHandle()||staging->depthImageHandle()
                ||staging->readRawPass(renderpassoutput::Output::Color,absent,error))
                return fail("Metal interior interval evidence activated incomplete source coverage or Color");
        }
    }
    if(directional&&!cameraCase&&!cellCase) {
        const auto corrupt=[&](auto mutate) {
            auto changed=receipt;auto source=std::make_shared<gc::NativeDirectionalSourceReceipt>(*receipt.directionalSource);
            mutate(*source);changed.directionalSource=source;
            return !gc::validateNativeReadback(prepared,bound,changed,ignored,error,
                &receipt.gratingIncidents->submission,true,sourceBinding)&&!error.empty();
        };
        if(!corrupt([](auto& value){++value.echo.input.identifiers[0];})
            ||!corrupt([](auto& value){value.echo.input.identifiers[1]=0;})
            ||!corrupt([](auto& value){value.echo.input.identifiers[2]=0;})
            ||!corrupt([](auto& value){++value.echo.input.identifiers[3];})
            ||!corrupt([](auto& value){value.echo.rawRadiance[0]=1;})
            ||!corrupt([](auto& value){value.echo.input.coefficients[0]*=.5f;})
            ||!corrupt([](auto& value){value.potential[0]*=.5f;})
            ||!corrupt([](auto& value){value.submission.completedFence=0;})
            ||!corrupt([](auto& value){++value.bindingGeneration;}))
            return fail("Metal directional transport accepted a forged unit, owner, echo, potential or fence");
        const auto originalSource=fixture.scene->lights[0].spectralSource;
        for(unsigned variant=0;variant<2;++variant) {
            auto& source=fixture.scene->lights[0].spectralSource;source=originalSource;
            if(variant==0)source.radiance.fill(.75f);
            else for(auto& coefficient:source.directionalIrradiance.coefficients)coefficient*=.5f;
            const auto changedGrating=compileGrating(description,material->structuralDigest());if(!changedGrating)return false;
            std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> changedFrame;
            std::shared_ptr<const gc::GratingIncidentBinding> changedBinding;
            if(!prepareMixed(changedGrating,material,changedFrame,changedBinding))return false;
            gc::NativeReadback changed;gc::EnergyLedger changedEnergy;
            if(!arbitgpu::produceMetalGlassGratingTerminalReadback(changedFrame,changedBinding,changed,changedEnergy,error))return false;
            if(!changed.directionalSource||changed.spectralRadiance.size()!=receipt.spectralRadiance.size())
                return fail("Metal directional source edit lost native transport proof");
            for(unsigned band=0;band<8;++band)
                if(!gc::nearPower(changedEnergy.launched[band],ledger.launched[band]*(variant==0?1:.5)))
                    return fail("Metal directional flux followed raw Li or failed authored D scaling");
            for(std::size_t pixel=0;pixel<receipt.spectralRadiance.size();++pixel)
                if(!gc::nearPower(changed.spectralRadiance[pixel],receipt.spectralRadiance[pixel]*(variant==0?1:.5)))
                    return fail("Metal directional accumulation followed raw Li or failed authored D scaling");
            if(variant==0&&std::any_of(changed.directionalSource->echo.rawRadiance.begin(),changed.directionalSource->echo.rawRadiance.end(),[](float v){return v!=.75f;}))
                return fail("Metal directional native echo omitted the separately uploaded raw Li");
            auto foreign=changed;gc::EnergyLedger foreignEnergy;
            if(gc::validateNativeReadback(prepared,bound,foreign,foreignEnergy,error,
                &changed.gratingIncidents->submission,true,sourceBinding))return fail("Metal directional receipt accepted a foreign source/frame owner");
        }
        fixture.scene->lights[0].spectralSource=originalSource;
    }
    auto highDescription=description;highDescription.microstructure.grooveDepthNanometres=125;
    const auto high=diffractionmaterial::admit(highDescription,error);if(!high)return false;
    const auto highMaterial=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*high);
    const auto highGrating=compileGrating(highDescription,highMaterial->structuralDigest());if(!highGrating)return false;
    std::shared_ptr<const gc::GratingIncidentBinding> highBinding;
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> highStaging;
    if(!prepareMixed(highGrating,highMaterial,highStaging,highBinding))return false;
    if(arbitgpu::inspectMetalGlassGratingCameraAllocationForFixture(staging,highBinding,error)||error.empty())
        return fail("Metal pending camera allocation accepted a foreign mixed material owner");
    const auto previous=receipt.spectralRadiance;
    if(produce(staging,highBinding,receipt,ledger,error)||error.empty()
        ||receipt.spectralRadiance!=previous)return fail("Metal mixed staging accepted a foreign compiler/native owner");
    if(produce(highStaging,highBinding,receipt,ledger,error)||error.empty()
        ||receipt.spectralRadiance!=previous)return fail("Metal high-tail terminal continuation published a partial readback");
    error.clear();return true;
}

bool metalCausticCases(arbitgpu::NativeFixtureSceneBackend& backend,std::string& error)
{
    namespace gc=arbitgpu::glasscaustics;
    namespace oracle=glasscaustictests;
    const auto fail=[&](const char* message){error=message;return false;};
    std::array<gc::EnergyLedger,3> refined;
    gc::EnergyLedger roughBaseline;
    std::vector<float> roughFootprint;
    for(unsigned caseIndex=0;caseIndex<24;++caseIndex) {
        const unsigned level=std::min(caseIndex,2u);
        auto fixture=oracle::ordinaryFixture();
        if(caseIndex==3||caseIndex==13)fixture.scene->lights[0].spectralSource.radiance[3]*=2;
        if(caseIndex==4||caseIndex==14) {
            auto rho=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*fixture.program->objectPrograms[0]);
            rho->spectralReflectance->coefficients[5]*=.5f;fixture.program->objectPrograms[0]=rho;
        }
        if(caseIndex==5||caseIndex==6||caseIndex==15||caseIndex==16) {
            auto blocker=oracle::fixture(.004f,1,false);
            blocker.scene->lights[0]=fixture.scene->lights[0];blocker.ordinaryLight=fixture.ordinaryLight;blocker.policy=fixture.policy;
            if(caseIndex==6||caseIndex==16) {
                const auto& object=blocker.scene->objects[blocker.scene->objectCount-1];
                for(std::size_t v=object.firstVertex;v<object.firstVertex+object.vertexCount;++v)blocker.scene->vertices[v].position.z=-10;
            }
            fixture=std::move(blocker);
        }
        if(caseIndex==7||caseIndex==17)fixture.scene->objects[0].transform.translation.x+=.2f;
        if(caseIndex==8||caseIndex==18)fixture.scene->objects[1].transform.translation.x+=.2f;
        if(caseIndex==9) {
            const double angle=std::atan(.2)+.04;
            fixture.scene->lights[0].transform.rotation={0,float(std::sin(angle/2)),0,float(std::cos(angle/2))};
        }
        fixture.scene->cameras[0].transform.translation.x=caseIndex==10?6.2f:6;fixture.scene->cameras[0].verticalFovRadians=2;
        if(caseIndex>=11) {
            fixture.program->parameters.emissionRoughness[3]=caseIndex==11?.04f:.05f;
            // Normal source illumination excludes grazing side-entry scattering.
            fixture.scene->lights[0].transform.rotation={0,0,0,1};
            if(caseIndex==19) {
                fixture.dispersion=.006f;
                fixture.program->parameters.transmissionIorClearcoat[3]=fixture.dispersion;
            }
            if(caseIndex==21) {
                fixture.scene->cameras[0].transform.translation.x=0;
                fixture.scene->cameras[0].verticalFovRadians=.25f;
            }
        }
        const std::uint32_t width=caseIndex==22?1920:caseIndex==23?3840:caseIndex>=20?113:32;
        const std::uint32_t height=caseIndex==22?1080:caseIndex==23?2160:caseIndex>=20?87:32;
        fixture.program->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;
        for(auto& child:fixture.program->objectPrograms) {
            auto metal=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*child);
            metal->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;child=metal;
        }
        const auto native=backend.prepare(fixture.scene,fixture.program);
        if(!native.prepared) {error=native.error;return false;}
        gc::Prepared prepared;
        if(!arbitgpu::prepareMetalGlassCaustic(native.resources,fixture.ordinaryLight,*fixture.policy,prepared,error))return false;
        arbitgpu::NativeFixtureSceneRuntimeInputs runtime;runtime.linearColor=true;
        gc::SchedulingDescription scheduling;scheduling.entrySubdivisions=std::size_t(1)<<level;
        const auto result=arbitgpu::renderMetalGlassCausticFrame(fixture.scene,native.resources,width,height,runtime,prepared,scheduling);
        if(!result.rendered||!result.frame) {error=result.error;return false;}
        if(!arbitgpu::inspectMetalRasterCaptureForFixture(result.frame,false,error))return false;
        gc::NativeReadback receipt;gc::EnergyLedger measured;
        if(!arbitgpu::readMetalGlassCausticReceipt(result.frame,fixture.frame,receipt,measured,error))return false;
        const auto& schedule=fixture.frame;
        const std::uint64_t minimumOwned=schedule.workTexels.size()*sizeof(gc::Texel)
            +schedule.triangleTexels.size()*sizeof(gc::Texel)
            +schedule.work.size()*sizeof(arbitgpu::metalcaustic::Path)
            +schedule.bounds.maximumDeposits*sizeof(arbitgpu::metalcaustic::Deposit)
            +schedule.bounds.deposition.paddedKeyCount*sizeof(gc::DepositionKey)
            +std::uint64_t(width)*height*(8*sizeof(float)+20)
            +sizeof(std::uint32_t)
            +std::uint64_t(arbitgpu::opaquevisibility::kTextureWidth)*schedule.query.height*sizeof(gc::Texel);
        if(result.stats.textureBytes<minimumOwned
            ||result.stats.textureBytes>renderpassoutput::kMaximumTotalBytes)
            return fail("Metal caustic storage receipt omits retained native allocations");
        if(caseIndex==0||caseIndex==12) {
            const auto ordinary=backend.render(fixture.scene,native.resources,width,height,runtime);
            if(!ordinary.rendered||(caseIndex==0&&ordinary.stats.textureBytes!=0))
                return fail("Metal ordinary frame changed its prior texture receipt");
            const auto repeated=arbitgpu::renderMetalGlassCausticFrame(fixture.scene,native.resources,width,height,runtime,prepared,scheduling);
            if(!repeated.rendered||repeated.stats.textureBytes!=result.stats.textureBytes)
                return fail("Metal identical caustic submission changed its storage receipt");
            if(!arbitgpu::inspectMetalRasterCaptureForFixture(result.frame,false,error)
                ||!arbitgpu::inspectMetalRasterCaptureForFixture(repeated.frame,false,error))return false;
            gc::Frame repeatedSchedule;gc::NativeReadback repeatedReceipt;gc::EnergyLedger repeatedEnergy;
            if(!arbitgpu::readMetalGlassCausticReceipt(repeated.frame,repeatedSchedule,repeatedReceipt,repeatedEnergy,error))return false;
            arbitgpu::NativeRawPassPixels firstColor,secondColor;
            if(!result.frame->readRawPass(renderpassoutput::Output::Color,firstColor,error)
                ||!repeated.frame->readRawPass(renderpassoutput::Output::Color,secondColor,error))return false;
            if(firstColor.bytes!=secondColor.bytes||receipt.spectralRadiance.size()!=repeatedReceipt.spectralRadiance.size()
                ||std::memcmp(receipt.spectralRadiance.data(),repeatedReceipt.spectralRadiance.data(),receipt.spectralRadiance.size()*sizeof(float))!=0
                ||receipt.deposition.keys.size()!=repeatedReceipt.deposition.keys.size()
                ||std::memcmp(receipt.deposition.keys.data(),repeatedReceipt.deposition.keys.data(),receipt.deposition.keys.size()*sizeof(gc::DepositionKey))!=0)
                return fail("Metal identical caustic submission changed completed bytes");
        }
        if(caseIndex<3)refined[level]=measured;
        if(caseIndex==11)roughFootprint=receipt.spectralRadiance;
        if(caseIndex==12) {
            roughBaseline=measured;
            if(roughFootprint==receipt.spectralRadiance)
                return fail("Metal rough caustic controls did not change the native receiver footprint");
        }
        if(caseIndex>=11) {
            if(fixture.frame.roughness!=fixture.program->parameters.emissionRoughness[3]
                ||fixture.frame.interfaceSamples!=16||!result.stats.roughGlassReceiptComplete)
                return fail("Metal rough caustic omitted its authored estimator or complete camera receipt");
            for(const auto& path:receipt.paths)
                if(path.roughTransportVersion!=gc::kRoughTransportVersion
                    ||path.interfaceSample!=fixture.frame.work[path.work].interfaceSample)
                    return fail("Metal rough caustic lost its native interface sample receipt");
            if(caseIndex>=20) {
                std::shared_ptr<const arbitgpu::roughglasstiles::Schedule> tiles;
                arbitgpu::roughglasstiles::Aggregate aggregate;
                if(!arbitgpu::readMetalRoughGlassTileAggregate(result.frame,tiles,aggregate,error))return false;
                if(!tiles||aggregate.completedPixels!=std::uint64_t(width)*height
                    ||aggregate.completedTiles!=tiles->tiles().size()
                    ||aggregate.peakReceiptBytes>arbitgpu::roughglasstiles::kMaximumTilePixels*48
                    ||(caseIndex==21&&aggregate.coveredPixels==0))
                    return fail("Metal rough caustic camera tiles did not complete exact bounded coverage");
            }
        }
        if(caseIndex==3||caseIndex==4||caseIndex==13||caseIndex==14) for(unsigned b=0;b<8;++b) {
            const double multiplier=((caseIndex==3||caseIndex==13)&&b==3)?2:((caseIndex==4||caseIndex==14)&&b==5)?.5:1;
            const auto& baseline=caseIndex>=11?roughBaseline:refined[2];
            if(!gc::nearPower(measured.deposited[b],baseline.deposited[b]*multiplier,5e-4))
                return fail("Metal caustic Li-only or rho-only perturbation changed the wrong wavelength power");
        }
        if((caseIndex==5||caseIndex==15)&&std::any_of(measured.launched.begin(),measured.launched.end(),[](double p){return p!=0;}))
            return fail("Metal caustic source blocker failed to extinguish launched native flux");
        if(caseIndex==6||caseIndex==16) for(unsigned b=0;b<8;++b)
            if(!gc::nearPower(measured.deposited[b],(caseIndex>=11?roughBaseline:refined[2]).deposited[b],5e-4))
                return fail("Metal caustic blocker behind receiver changed native deposited power");
        fixture.prepared=prepared;
        const auto expected=oracle::powerOracle(fixture);
        if(!expected.complete)return fail("Metal caustic independent binary64 energy oracle is incomplete");
        std::array<oracle::D3,8> moments{};
        for(const auto& deposit:receipt.deposits) {
            const auto band=fixture.frame.work[deposit.work].band;
            moments[band]=oracle::plus(moments[band],oracle::doublePoint(deposit.receiver.position),deposit.outgoingPower);
        }
        for(unsigned band=0;band<8;++band) {
            const auto& actual=measured;const auto& reference=expected.energy;
            if(!gc::nearPower(actual.potential[band],reference.potential[band],5e-5)
                ||!gc::nearPower(actual.launched[band],reference.launched[band],5e-5)
                ||!gc::nearPower(actual.deposited[band],reference.deposited[band],5e-4)
                ||!gc::nearPower(actual.absorbed[band],reference.absorbed[band],5e-4)
                ||std::abs(actual.masked[band]-reference.masked[band])>2e-6*std::max(actual.launched[band],1e-20)
                ||!gc::nearPower(actual.launched[band],actual.deposited[band]+actual.absorbed[band]+actual.escaped[band]+actual.unresolved[band]+actual.masked[band],5e-4))
                return fail("Metal caustic all-band native energy differs from independent binary64 source/triangle/branch oracle");
            if(actual.deposited[band]>0) {
                const auto centroid=oracle::plus(oracle::D3{},moments[band],1/actual.deposited[band]);
                const auto delta=oracle::minus(centroid,expected.centroid[band]);
                if(oracle::inner(delta,delta)>1e-6)
                    return fail("Metal caustic native receiver centroid differs from independent binary64 branch oracle");
                if(caseIndex<3&&band==3) std::cout<<"Metal caustic refinement subdivisions="<<scheduling.entrySubdivisions
                    <<"; deposited="<<actual.deposited[band]<<"; visible="<<actual.cameraVisible[band]
                    <<"; centroid="<<centroid[0]<<','<<centroid[1]<<','<<centroid[2]<<'\n';
            }
        }
        if(caseIndex!=5&&caseIndex!=15&&caseIndex!=21&&std::none_of(measured.cameraVisible.begin(),measured.cameraVisible.end(),[](double power){return power>0;}))
            return fail("Metal caustic native fixture produced no visible receiver flux");
        const auto triangles=oracle::worldTriangles(*fixture.scene);
        const oracle::D3 camera{fixture.frame.camera.transform.translation.x,fixture.frame.camera.transform.translation.y,fixture.frame.camera.transform.translation.z};
        const double tangent=std::tan(double(fixture.frame.camera.verticalFovRadians)/2),aspect=double(fixture.frame.width)/fixture.frame.height;
        std::vector<double> pixels(receipt.spectralRadiance.size());
        for(std::size_t w=0;w<fixture.frame.work.size();++w) {
            const auto& work=fixture.frame.work[w];const auto branches=oracle::branchOracle(fixture,fixture.frame.entries[work.entry],work.band,fixture.policy->description().minimumBranchFraction,
                fixture.policy->description().maximumVisits,fixture.policy->description().maximumInterfaces,{},work.interfaceSample);
            if(!branches.valid)return fail("Metal caustic independent branch oracle refused native work");
            for(const auto& arrival:branches.arrivals) {
                const auto nativeDeposit=std::find_if(receipt.deposits.begin(),receipt.deposits.end(),[&](const auto& d) {
                    return d.work==w&&d.reflectionHistory==arrival.history&&d.interfaces==arrival.interfaces;
                });
                if(nativeDeposit==receipt.deposits.end())return fail("Metal caustic branch receiver lacks native provenance");
                const auto& h=arrival.receiver;
                if(nativeDeposit->receiver.objectId!=h.object||nativeDeposit->receiver.materialId!=h.material
                    ||nativeDeposit->receiver.instanceId!=h.instance||nativeDeposit->receiver.triangleId!=h.triangle
                    ||!gc::nearPower(nativeDeposit->fraction,arrival.fraction,5e-4))
                    return fail("Metal caustic native receiver triangle/history/fraction differs from independent oracle");
                const double depth=camera[2]-h.position[2];
                if(depth<fixture.frame.camera.nearPlane||depth>fixture.frame.camera.farPlane)continue;
                const double px=(double(h.position[0]-camera[0])/(depth*tangent*aspect)+1)*fixture.frame.width/2;
                const double py=(1-double(h.position[1]-camera[1])/(depth*tangent))*fixture.frame.height/2;
                if(px<0||py<0||px>=fixture.frame.width||py>=fixture.frame.height)continue;
                const auto x=std::uint32_t(px),y=std::uint32_t(py);
                const auto ray=[&](double x,double y){return oracle::normalized(oracle::D3{(2*x/fixture.frame.width-1)*tangent*aspect,(1-2*y/fixture.frame.height)*tangent,-1});};
                if(oracle::inner(h.normal,ray(x+.5,y+.5))>=-1e-6)continue;
                const auto delta=oracle::minus(h.position,camera);const double distance=std::sqrt(oracle::inner(delta,delta));
                const auto endpoint=oracle::closest(triangles,camera,oracle::normalized(delta),0,distance+1e-3);
                const auto center=oracle::closest(triangles,camera,ray(x+.5,y+.5),0,fixture.frame.camera.farPlane*1e4);
                const auto same=[&](const oracle::Hit& hit){return hit.found&&hit.object==h.object&&hit.material==h.material&&hit.instance==h.instance&&hit.triangle==h.triangle;};
                if(!same(endpoint)||!same(center))continue;
                std::array<oracle::D3,4> corner;
                const std::array<std::array<double,2>,4> offsets{{{0,0},{1,0},{1,1},{0,1}}};
                for(unsigned c=0;c<4;++c) {
                    const auto direction=ray(x+offsets[c][0],y+offsets[c][1]);
                    const double t=oracle::inner(h.normal,delta)/oracle::inner(h.normal,direction);
                    corner[c]=oracle::plus(camera,direction,t);
                }
                const auto area=[&](unsigned a,unsigned b,unsigned c) {
                    const auto v=oracle::exterior(oracle::minus(corner[b],corner[a]),oracle::minus(corner[c],corner[a]));
                    return .5*std::sqrt(oracle::inner(v,v));
                };
                const double footprint=area(0,1,2)+area(0,2,3);
                const auto* material=arbitgpu::nativeSurfaceForObject(fixture.program.get(),HarmonicMIDI::grid::SceneObjectId{h.object});
                const double power=double(work.incidentPower)*arrival.fraction*material->spectralReflectance->coefficients[work.band];
                pixels[(std::size_t(y)*fixture.frame.width+x)*8+work.band]+=power/(3.14159265358979323846*footprint);
            }
        }
        for(std::size_t pixel=0;pixel<pixels.size();++pixel)
            if(!gc::nearPower(receipt.spectralRadiance[pixel],pixels[pixel],1e-3))
                return fail("Metal caustic native eight-band pixel differs from independent branch/camera/plane-area oracle");
        arbitgpu::NativeRawPassPixels color;
        if(!result.frame->readRawPass(renderpassoutput::Output::Color,color,error)||!arbitgpu::smoothglass::finiteColorFrame(color,error))return false;
        // Atomic negative controls mutate completed native copies only.
        for(unsigned fault=0;caseIndex<20&&fault<13;++fault) {
            auto bad=receipt;auto sentinel=measured;sentinel.potential[0]=-123;
            if(fault==0)bad.frameIdentity+="stale";
            if(fault==1)bad.paths.front().complete=false;
            if(fault==2)bad.spectralRadiance.front()=std::numeric_limits<float>::infinity();
            if(fault==3)bad.paths.front().queryWork=fixture.frame.bounds.queryWork+1;
            if(fault==4) {if(bad.deposits.empty())continue;bad.deposits.front().receiver.triangleId+=1;}
            if(fault==5)bad.deposition.overflow=true;
            if(fault==6)bad.deposition.complete=false;
            if(fault==7)bad.deposition.submission.queryGeneration+=1;
            if(fault==8) {if(bad.deposition.visibleCount<2)continue;std::swap(bad.deposition.keys[0],bad.deposition.keys[1]);}
            if(fault==9) {if(bad.deposition.keys.empty())continue;bad.deposition.keys.back().flatIndex=0;}
            if(fault==10)bad.paths.front().roughTransportVersion+=1;
            if(fault==11)bad.paths.front().interfaceSample+=1;
            if(fault==12)bad.paths.front().maskedPower=bad.paths.front().launchedPower+1;
            std::string rejected;
            if(gc::validateNativeReadback(prepared,fixture.frame,bad,sentinel,rejected,&receipt.deposition.submission)||sentinel.potential[0]!=-123)
                return fail("Metal caustic stale/incomplete/overflow/triangle receipt failed atomic refusal");
        }
        auto stale=prepared;stale.glass.sourceScene=std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(*fixture.scene);
        const auto rejected=arbitgpu::renderMetalGlassCausticFrame(fixture.scene,native.resources,width,height,runtime,stale,scheduling);
        if(rejected.rendered||rejected.frame)return fail("Metal caustic stale owner published a frame");
    }
    {
        auto strong=oracle::ordinaryFixture();
        strong.scene->lights[0].transform.rotation={0,0,0,1};
        strong.scene->cameras[0].transform.translation.x=6;
        strong.scene->cameras[0].verticalFovRadians=2;
        strong.program->parameters.emissionRoughness[3]=.35f;
        strong.program->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;
        for(auto& child:strong.program->objectPrograms) {
            auto metal=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*child);
            metal->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;child=metal;
        }
        const auto native=backend.prepare(strong.scene,strong.program);
        if(!native.prepared) {error=native.error;return false;}
        gc::Prepared prepared;
        if(!arbitgpu::prepareMetalGlassCaustic(native.resources,strong.ordinaryLight,*strong.policy,prepared,error))return false;
        arbitgpu::NativeFixtureSceneRuntimeInputs runtime;runtime.linearColor=true;
        const auto refused=arbitgpu::renderMetalGlassCausticFrame(strong.scene,native.resources,113,87,runtime,prepared);
        if(refused.rendered||refused.frame||refused.error.empty())
            return fail("Metal strongly masked rough caustic published a partial frame");
    }
    for(unsigned level=1;level<3;++level)for(unsigned band=0;band<8;++band)
        if(!gc::nearPower(refined[level].potential[band],refined[0].potential[band],5e-5))
            return fail("Metal caustic native refinement changed integrated projected source power");
    // A normal source through a parallel slab maps the uniform authored square
    // to itself. Its exact outgoing normalized second moment is 1/3 per axis.
    {
        auto spatial=oracle::ordinaryFixture();spatial.scene->cameras[0].transform.translation={0,0,-2};
        spatial.scene->lights[0].transform.rotation={0,0,0,1};
        const auto& glass=spatial.scene->objects[0];
        for(std::size_t v=glass.firstVertex;v<glass.firstVertex+glass.vertexCount;++v)
            if(spatial.scene->vertices[v].position.z>0)spatial.scene->vertices[v].position.z-=.25f*spatial.scene->vertices[v].position.x;
        spatial.program->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;
        for(auto& child:spatial.program->objectPrograms) {
            auto metal=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*child);
            metal->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;child=metal;
        }
        const auto reference=oracle::quadraturePowerOracle(spatial);
        if(!reference.complete)return fail("Metal analytic slab full-branch quadrature is incomplete");
        const auto native=backend.prepare(spatial.scene,spatial.program);
        if(!native.prepared) {error=native.error;return false;}
        gc::Prepared prepared;
        if(!arbitgpu::prepareMetalGlassCaustic(native.resources,spatial.ordinaryLight,*spatial.policy,prepared,error))return false;
        std::array<double,8> previousError{};
        for(unsigned level=0;level<3;++level) {
            arbitgpu::NativeFixtureSceneRuntimeInputs runtime;runtime.linearColor=true;
            gc::SchedulingDescription scheduling;scheduling.entrySubdivisions=std::size_t(1)<<level;
            const auto result=arbitgpu::renderMetalGlassCausticFrame(spatial.scene,native.resources,32,32,runtime,prepared,scheduling);
            if(!result.rendered||!result.frame) {error=result.error;return false;}
            gc::Frame schedule;gc::NativeReadback receipt;gc::EnergyLedger energy;
            if(!arbitgpu::readMetalGlassCausticReceipt(result.frame,schedule,receipt,energy,error))return false;
            std::array<double,8> power{},x{},y{},xx{},yy{};
            for(const auto& deposit:receipt.deposits) {
                const auto band=schedule.work[deposit.work].band;const double p=deposit.outgoingPower;
                const auto& position=deposit.receiver.position;
                power[band]+=p;x[band]+=p*position[0];y[band]+=p*position[1];
                xx[band]+=p*position[0]*position[0];yy[band]+=p*position[1]*position[1];
            }
            for(unsigned band=0;band<8;++band) {
                if(power[band]<=0||!gc::nearPower(power[band],reference.energy.deposited[band],8e-4)
                    ||std::abs(x[band]/power[band]-reference.centroid[band][0])>3e-4
                    ||std::abs(y[band]/power[band]-reference.centroid[band][1])>3e-4)
                    return fail("Metal analytic slab spectral power or centroid differs from independent quadrature");
                const double momentError=std::max(std::abs(xx[band]/power[band]-1.0/3.0),std::abs(yy[band]/power[band]-1.0/3.0));
                if(level&&momentError>.4*previousError[band]+1e-6)
                    return fail("Metal analytic slab spatial second moment failed refinement convergence");
                previousError[band]=momentError;
                std::cout<<"Metal analytic caustic convergence level="<<level<<" band="<<band<<" secondMomentError="<<momentError<<'\n';
            }
        }
    }
    // Sparse physical scenes still require full-canvas storage and resolve.
    // Keep the source below RGBA16F saturation as pixel area shrinks.
    for(const auto extent:std::array<std::array<std::uint32_t,2>,2>{{{1920,1080},{3840,2160}}}) {
        std::array<gc::EnergyLedger,3> energy;
        std::array<oracle::DepositionMoment,3> spatial;
        for(unsigned level=0;level<3;++level) {
            auto fixture=oracle::ordinaryFixture();fixture.scene->cameras[0].transform.translation.x=6;
            fixture.scene->cameras[0].verticalFovRadians=2;
            for(auto& value:fixture.scene->lights[0].spectralSource.radiance)value*=1e-4f;
            fixture.program->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;
            for(auto& child:fixture.program->objectPrograms) {
                auto metal=std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*child);
                metal->backend=arbitgpu::NativeFixtureMaterialBackend::Metal;child=metal;
            }
            const auto native=backend.prepare(fixture.scene,fixture.program);
            if(!native.prepared) {error=native.error;return false;}
            gc::Prepared prepared;
            if(!arbitgpu::prepareMetalGlassCaustic(native.resources,fixture.ordinaryLight,*fixture.policy,prepared,error))return false;
            arbitgpu::NativeFixtureSceneRuntimeInputs runtime;runtime.linearColor=true;
            gc::SchedulingDescription scheduling;scheduling.entrySubdivisions=std::size_t(1)<<level;
            const auto start=std::chrono::steady_clock::now();
            const auto result=arbitgpu::renderMetalGlassCausticFrame(fixture.scene,native.resources,extent[0],extent[1],runtime,prepared,scheduling);
            const auto frameMilliseconds=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if(!result.rendered||!result.frame) {error=result.error;return false;}
            gc::NativeReadback receipt;
            if(!arbitgpu::readMetalGlassCausticReceipt(result.frame,fixture.frame,receipt,energy[level],error))return false;
            fixture.prepared=prepared;const auto reference=oracle::powerOracle(fixture);
            if(!reference.complete)return fail("Metal full-resolution independent transport oracle is incomplete");
            const auto quadrature=oracle::quadraturePowerOracle(fixture);
            if(!quadrature.complete)return fail("Metal independent full-branch spatial quadrature is incomplete");
            std::array<oracle::D3,8> receiverMoments{};
            for(const auto& deposit:receipt.deposits) {
                const auto band=fixture.frame.work[deposit.work].band;
                receiverMoments[band]=oracle::plus(receiverMoments[band],oracle::doublePoint(deposit.receiver.position),deposit.outgoingPower);
            }
            spatial[level]=oracle::depositionMoment(fixture.frame,receipt);
            for(unsigned band=0;band<8;++band) {
                if(!gc::nearPower(energy[level].potential[band],reference.energy.potential[band],5e-5)
                    ||!gc::nearPower(energy[level].deposited[band],reference.energy.deposited[band],5e-4)
                    ||!gc::nearPower(spatial[level].power[band],energy[level].cameraVisible[band],5e-4))
                    return fail("Metal full-resolution native spectral/receiver/spatial power differs from independent oracle");
                if(energy[level].deposited[band]>0) {
                    const auto centroid=oracle::plus(oracle::D3{},receiverMoments[band],1/energy[level].deposited[band]);
                    const auto delta=oracle::minus(centroid,quadrature.centroid[band]);
                    const double spatialError=std::sqrt(oracle::inner(delta,delta));
                    const double powerError=std::abs(energy[level].deposited[band]-quadrature.energy.deposited[band]);
                    if(!std::isfinite(spatialError)||!std::isfinite(powerError))
                        return fail("Metal independent refinement error is non-finite");
                    std::cout<<"Metal caustic independent quadrature "<<extent[0]<<'x'<<extent[1]<<" level="<<level
                        <<" band="<<band<<" receiverCentroidError="<<spatialError<<" spectralPowerError="<<powerError<<'\n';
                }
            }
            for(std::size_t at=0;at<receipt.spectralRadiance.size();++at) {
                const auto expected=spatial[level].radiance.find(at);
                const float value=expected==spatial[level].radiance.end()?0:expected->second;
                if(!gc::nearPower(receipt.spectralRadiance[at],value,1e-6))
                    return fail("Metal full-resolution float32 reduction differs from canonical deposition oracle");
            }
            arbitgpu::MetalGlassCausticMetrics metrics;
            if(!arbitgpu::readMetalGlassCausticMetrics(result.frame,metrics,error))return false;
            if(metrics.admittedBytes>renderpassoutput::kMaximumTotalBytes||metrics.peakNativeBytes==0
                ||metrics.retainedNativeBytes>result.stats.textureBytes||metrics.traceMilliseconds<=0
                ||metrics.depositionMilliseconds<=0||metrics.resolveMilliseconds<=0||frameMilliseconds<=0)
                return fail("Metal full-resolution resource or phase timing receipt is incomplete");
            arbitgpu::NativeRawPassPixels color;
            if(!result.frame->readRawPass(renderpassoutput::Output::Color,color,error)
                ||!arbitgpu::smoothglass::finiteColorFrame(color,error))return false;
            const auto repeated=arbitgpu::renderMetalGlassCausticFrame(fixture.scene,native.resources,extent[0],extent[1],runtime,prepared,scheduling);
            if(!repeated.rendered||!repeated.frame) {error=repeated.error;return false;}
            gc::Frame repeatedSchedule;gc::NativeReadback repeatedReceipt;gc::EnergyLedger repeatedEnergy;
            arbitgpu::NativeRawPassPixels repeatedColor;
            if(!arbitgpu::readMetalGlassCausticReceipt(repeated.frame,repeatedSchedule,repeatedReceipt,repeatedEnergy,error)
                ||!repeated.frame->readRawPass(renderpassoutput::Output::Color,repeatedColor,error))return false;
            if(color.bytes!=repeatedColor.bytes||receipt.spectralRadiance.size()!=repeatedReceipt.spectralRadiance.size()
                ||std::memcmp(receipt.spectralRadiance.data(),repeatedReceipt.spectralRadiance.data(),receipt.spectralRadiance.size()*sizeof(float))!=0)
                return fail("Metal full-resolution equal-time native Color or spectral bytes differ");
            std::cout<<"Metal caustic native "<<extent[0]<<'x'<<extent[1]<<" subdivisions="<<scheduling.entrySubdivisions
                <<" bytes="<<metrics.admittedBytes<<" retained="<<metrics.retainedNativeBytes<<" peakNative="<<metrics.peakNativeBytes
                <<" queryWork="<<fixture.frame.bounds.queryWork<<" depositionWork="<<fixture.frame.bounds.deposition.depositionWork
                <<" resolveWork="<<fixture.frame.bounds.deposition.resolveWork<<" traceMs="<<metrics.traceMilliseconds
                <<" depositionMs="<<metrics.depositionMilliseconds<<" receiptMs="<<metrics.readbackMilliseconds
                <<" resolveMs="<<metrics.resolveMilliseconds<<" frameMs="<<frameMilliseconds<<'\n';
        }
        for(unsigned level=1;level<3;++level)for(unsigned band=0;band<8;++band) {
            if(!gc::nearPower(energy[level].potential[band],energy[0].potential[band],5e-5))
                return fail("Metal full-resolution refinement changed integrated spectral source power");
            // These measured spatial deltas expose refinement behavior. Sparse
            // sample conservation alone does not certify quadrature convergence.
            if(spatial[level].power[band]>0&&spatial[level-1].power[band]>0)
                std::cout<<"Metal caustic spatial refinement "<<extent[0]<<'x'<<extent[1]<<" band="<<band
                    <<" level="<<level<<" centroidDelta="
                    <<std::hypot(spatial[level].x[band]/spatial[level].power[band]-spatial[level-1].x[band]/spatial[level-1].power[band],
                        spatial[level].y[band]/spatial[level].power[band]-spatial[level-1].y[band]/spatial[level-1].power[band])<<'\n';
        }
    }
    error.clear();return true;
}

std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> roughMetalProgram(
    const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
    float roughness, float dispersion, std::string& error)
{
    using namespace HarmonicMIDI::grid;
    using namespace videohelper::tests::smoothglassfixture;
    surfacematerialbinding::ImportedSceneMaterialRequest request;
    request.scene = scene->id;
    request.sceneRevision = request.structuralRevision = request.evaluationRevision = request.programRevision = 1;
    request.sceneSnapshot = scene;
    request.program = authoredProgram(6,1.5f,dispersion);
    for (auto& op : request.program.operations)
        if (op.id == static_cast<unsigned>(surfacematerial::OutputSemantic::Roughness) + 1)
            op.literal = {roughness,0,0,0};
    request.binding.targetKind = surfacematerialbinding::BindingTargetKind::ObjectOverride;
    request.binding.object = scene->objects[0].id;
    request.binding.surfaceMaterialRevision = 1;
    const auto admitted = surfacematerial::admit(request.program,error);
    if (!admitted) return {};
    request.binding.surfaceMaterialDigest = admitted->structuralDigest();
    const auto binding = videorender::fixture3d::admitSurfaceMaterialBinding(
        scene,request,videohelper::materialprogram::BackendTarget::Metal,error);
    return binding ? binding->nativeProgram() : nullptr;
}

bool roughWaterMetalCases(arbitgpu::NativeFixtureSceneBackend& backend, std::string& error)
{
    namespace water = videohelper::tests::waterfixture;
    using namespace HarmonicMIDI::grid;
    using Pixels = arbitgpu::NativeRawPassPixels;
    struct Case { float roughness, angle, normalX; bool below, receivers; int hdrOrientation; float surfaceZ = 0; };
    const Case cases[] = {{.08f,20,0,false,true,-1},{.16f,20,0,false,true,-1},
        {.08f,20,0,true,false,-1},{.08f,60,0,true,false,-1},
        {.08f,20,.01f,false,true,-1},{.08f,20,0,false,false,-1},
        {.08f,20,0,false,false,0},{.08f,20,0,false,false,90},{.08f,20,0,false,true,-1,.1f}};
    namespace environment = videohelper::tests::environmentsecondaryfixture;
    const auto hdrBytes=environment::imageBytes(0);
    const visualenvironmentimage::Identity hdrKey {"metal-rough-water-hdr",1,modelpayload::detail::digestHex(hdrBytes)};
    const visualenvironmentimage::Metadata hdrMetadata {8,4,visualenvironmentimage::SourcePrimaries::Rec709D65};
    auto& hdrStore=videohelper::environmentimage::store();std::uint64_t received=0;
    if (!hdrStore.begin("metal-rough-water-upload",hdrKey,hdrMetadata,hdrBytes.size(),error)
        || !hdrStore.append("metal-rough-water-upload",0,videohelper::tests::environmentImageBase64(hdrBytes),received,error)
        || !hdrStore.commit("metal-rough-water-upload",error)) return false;
    std::vector<std::uint8_t> firstHdrColor;
    std::shared_ptr<const Visual3DScene> baselineScene;
    std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> baselineProgram;
    std::array<Pixels,3> baselineReceipts;
    Pixels baselineObjects,baselineMaterials,baselineColor;
    bool sawTir = false, sawNormalChange = false, sawRoughChange = false, sawReceiverChange = false, sawHdrChange = false, sawGeometryChange = false;
    for (const auto& test : cases) {
        auto authored = water::scene(test.angle);
        if (test.below) {
            auto& camera = authored.cameras[0].transform;
            camera.translation.z = -camera.translation.z;
            const double angle = (180.0-test.angle)*std::acos(-1.0)/180.0;
            camera.rotation = {0,float(std::sin(angle*.5)),0,float(std::cos(angle*.5))};
        }
        for (std::size_t v=0;v<4;++v) authored.vertices[v].normal = {test.normalX,0,1};
        if (test.receivers) {
            const auto& camera=authored.cameras[0].transform.translation;
            const double angle=test.angle*std::acos(-1.0)/180.0;
            const auto smooth=water::trace(authored,{camera.x,camera.y,camera.z},{-std::sin(angle),0,-std::cos(angle)},double(1.333f));
            const auto landing=[&](water::D3 direction,double z) {
                return smooth.entry[0]+(z-smooth.entry[2])/direction[2]*direction[0];
            };
            water::receiver(authored,5,landing(smooth.reflection,5),{2,.4,.1},.04);
            water::receiver(authored,-3,landing(smooth.refraction,-3),{.2,.5,3},.04);
        }
        for (std::size_t v=0;v<4;++v) authored.vertices[v].position.z += test.surfaceZ;
        if (test.hdrOrientation>=0) {
            authored.lights[0].color={1,1,1};
            authored.lights[0].environmentImage={1,hdrKey,float(test.hdrOrientation),0};
        }
        const auto scene=std::make_shared<const Visual3DScene>(authored);
        const auto material=water::binding(scene,videohelper::materialprogram::BackendTarget::Metal,error,test.roughness);
        if (!material) return false;
        videorender::fixture3d::FixtureSceneRenderer renderer(backend);
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;inputs.linearColor=true;
        videorender::fixture3d::RenderedFrame preview,exported;
        if (!renderer.renderPreview(scene,material,{},inputs,{9,9},"native-gpu",preview,error)
            || !renderer.renderExport(scene,material,{},inputs,{9,9},"native-gpu",exported,error)
            || !preview.stats.roughGlassReceiptComplete || !exported.stats.roughGlassReceiptComplete) {
            error="Metal rough water preview/export did not complete: "+error;return false;
        }
        std::array<Pixels,3> receipts,exportReceipts;
        Pixels objects,materials,color,exportObjects,exportMaterials,exportColor,normal;
        if (!arbitgpu::readMetalRoughGlassReceipt(preview.nativeFrame,receipts,objects,materials,error)
            || !arbitgpu::readMetalRoughGlassReceipt(exported.nativeFrame,exportReceipts,exportObjects,exportMaterials,error)
            || !preview.nativeFrame->readRawPass(renderpassoutput::Output::Color,color,error)
            || !exported.nativeFrame->readRawPass(renderpassoutput::Output::Color,exportColor,error)
            || !preview.nativeFrame->readRawPass(renderpassoutput::Output::Normal,normal,error)) return false;
        if (color.format!=arbitgpu::NativeTexturePixelFormat::Rgba16Float || color.bytes!=exportColor.bytes
            || objects.bytes!=exportObjects.bytes || materials.bytes!=exportMaterials.bytes) {
            error="Metal rough water Preview/Export Color or identity differs";return false;
        }
        for (std::size_t attachment=0;attachment<3;++attachment)
            if (receipts[attachment].bytes!=exportReceipts[attachment].bytes) {
                error="Metal rough water Preview/Export receipt differs";return false;
            }
        const auto& camera=authored.cameras[0].transform;
        const double angle=2*std::atan2(camera.rotation.y,camera.rotation.w);
        const double extent=std::tan(double(authored.cameras[0].verticalFovRadians)*.5);
        for (std::size_t y=0;y<9;++y) for (std::size_t x=0;x<9;++x) {
            const std::size_t pixel=y*9+x;
            const double vx=(2*(double(x)+.5)/9-1)*extent,vy=(1-2*(double(y)+.5)/9)*extent;
            const auto ray=water::unit(water::D3{std::cos(angle)*vx-std::sin(angle),vy,-std::sin(angle)*vx-std::cos(angle)});
            const auto expected=water::traceRough(authored,{camera.translation.x,camera.translation.y,camera.translation.z},
                ray,double(1.333f),test.roughness,
                [](const SceneLightRecord& light,water::D3 direction) {
                    return light.environmentImage.version ? environment::radiance(light,direction,0) : water::sky(light,direction);
                });
            if (!expected.complete) {error="Independent binary64 rough water oracle did not complete";return false;}
            sawTir |= expected.tir>0;
            std::uint32_t object=0,mat=0;
            std::memcpy(&object,objects.bytes.data()+pixel*4,4);std::memcpy(&mat,materials.bytes.data()+pixel*4,4);
            std::array<std::array<float,4>,3> values;
            for (std::size_t attachment=0;attachment<3;++attachment)
                std::memcpy(values[attachment].data(),receipts[attachment].bytes.data()+pixel*16,16);
            std::uint16_t normalAlpha=0;std::memcpy(&normalAlpha,normal.bytes.data()+pixel*8+6,2);
            if (object!=authored.objects[0].id.value || mat!=authored.objects[0].material.value
                || values[0][3]!=1 || values[2][3]!=1 || hdrimage::halfToFloat(normalAlpha)!=1
                || values[1][3]!=float(expected.closestHitQueries)) {
                error="Metal rough water native receipt or raw identity is incomplete";return false;
            }
            for (std::size_t c=0;c<3;++c) {
                std::uint16_t half=0;std::memcpy(&half,color.bytes.data()+pixel*8+c*2,2);
                const double actual=hdrimage::halfToFloat(half);
                if (!std::isfinite(actual) || std::abs(actual-expected.rgb[c])>.004
                    || std::abs(values[0][c]-expected.energy.terminal[c])>.00002
                    || std::abs(values[1][c]-expected.energy.masked[c])>.00002
                    || std::abs(values[2][c]-expected.energy.tail[c])>.00002) {
                    error="Metal rough water pixel/energy differs from independent one-interface oracle";return false;
                }
            }
        }
        if (!baselineScene) {
            baselineScene=scene;baselineProgram=material->nativeProgram();baselineReceipts=receipts;
            baselineObjects=objects;baselineMaterials=materials;baselineColor=color;
        } else {
            sawRoughChange |= test.roughness!=cases[0].roughness && color.bytes!=baselineColor.bytes;
            sawNormalChange |= test.normalX!=0 && color.bytes!=baselineColor.bytes;
            sawReceiverChange |= !test.receivers && !test.below && test.hdrOrientation<0 && color.bytes!=baselineColor.bytes;
            sawGeometryChange |= test.surfaceZ!=0 && color.bytes!=baselineColor.bytes;
            if (test.hdrOrientation==0) firstHdrColor=color.bytes;
            if (test.hdrOrientation==90) sawHdrChange=!firstHdrColor.empty() && color.bytes!=firstHdrColor;
        }
        const auto prepared=backend.prepare(scene,material->nativeProgram());
        if (!prepared.prepared) {error=prepared.error;return false;}
        const auto tiled=arbitgpu::renderMetalRoughGlassTiledFrame(scene,prepared.resources,9,9,inputs);
        Pixels tiledColor,tiledObjects,tiledMaterials;
        std::array<Pixels,3> tiledReceipts;
        if (!tiled.rendered || !tiled.stats.roughGlassReceiptComplete
            || !tiled.frame->readRawPass(renderpassoutput::Output::Color,tiledColor,error)
            || tiledColor.bytes!=color.bytes
            || !arbitgpu::readMetalRoughGlassReceipt(tiled.frame,tiledReceipts,tiledObjects,tiledMaterials,error)
            || tiledObjects.bytes!=objects.bytes || tiledMaterials.bytes!=materials.bytes) {
            error="Metal rough water shared tile completion changed Color: "+tiled.error;return false;
        }
        for (std::size_t i=0;i<3;++i) if (tiledReceipts[i].bytes!=receipts[i].bytes) {
            error="Metal rough water shared tile completion changed float32 energy";return false;
        }
    }
    if (!sawTir || !sawNormalChange || !sawRoughChange || !sawReceiverChange || !sawHdrChange || !sawGeometryChange) {
        error="Metal rough water fixtures missed TIR or authored roughness/normal/receiver/HDR causality";return false;
    }
    // Positive roughness consumes the existing evaluated wave mesh at each exact time.
    using namespace videowire::geometry;
    PortContract waveContract;waveContract.carrier=CarrierKind::geometry3D;
    waveContract.maxVertices=4096;waveContract.maxIndices=12288;
    auto grid=lowerGrid(601,601,1,9,9,.35f,waveContract,error);
    auto wave=grid?lowerDirectionalSurfaceWave(*grid,606,10000,.005,2,30,.5,.3,4,waveContract,error):std::nullopt;
    auto admittedWave=wave?admitValue(*wave,waveContract,error):std::nullopt;
    if (!admittedWave) return false;
    auto waveScene=water::scene(0);
    waveScene.vertexCount=waveScene.indexCount=waveScene.objectCount=waveScene.materialCount=0;
    SceneObjectRecord waveObject;waveObject.id={101};
    SceneMaterialRecord waveMaterial;waveMaterial.id={6};waveMaterial.baseColor={1,1,1};
    if (!appendGeometryToScene(*admittedWave,waveObject,waveMaterial,waveScene,error)) return false;
    const TimelineSceneBinding timeline{{101},encodeLoweredPlanText(lowerRuntimePlan(waveContract,*admittedWave))};
    waveScene.lights[0].color={1,1,1};waveScene.lights[0].environmentImage={1,hdrKey,0,0};
    std::vector<std::uint8_t> waveZero,waveHalf;
    for (const double seconds : {0.0,.5,0.0}) {
        auto evaluated=waveScene;
        if (!replaceTimelineSceneMesh(timeline,*admittedWave,seconds,evaluated,error)) return false;
        const auto scene=std::make_shared<const Visual3DScene>(evaluated);
        const auto material=water::binding(scene,videohelper::materialprogram::BackendTarget::Metal,error,.04f);
        if (!material) return false;
        const auto prepared=backend.prepare(scene,material->nativeProgram());
        if (!prepared.prepared) {error=prepared.error;return false;}
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;inputs.linearColor=true;inputs.timeSeconds=float(seconds);
        const auto rendered=backend.render(scene,prepared.resources,9,9,inputs);
        Pixels color;
        if (!rendered.rendered || !rendered.stats.roughGlassReceiptComplete
            || !rendered.frame->readRawPass(renderpassoutput::Output::Color,color,error)) {
            error="Metal rough evaluated wave did not complete: "+rendered.error;return false;
        }
        const auto& camera=evaluated.cameras[0].transform.translation;
        const auto expected=water::traceRough(evaluated,{camera.x,camera.y,camera.z},{0,0,-1},double(1.333f),.04f,
            [](const SceneLightRecord& light,water::D3 direction) {return environment::radiance(light,direction,0);});
        if (!expected.complete) {error="Independent evaluated rough-wave oracle is incomplete";return false;}
        for (std::size_t c=0;c<3;++c) {
            std::uint16_t half=0;std::memcpy(&half,color.bytes.data()+(4*9+4)*8+c*2,2);
            if (std::abs(hdrimage::halfToFloat(half)-expected.rgb[c])>.004) {
                error="Metal rough evaluated wave differs from actual-mesh oracle";return false;
            }
        }
        if (seconds==0 && waveZero.empty()) waveZero=color.bytes;
        else if (seconds==0 && color.bytes!=waveZero) {error="Metal rough-wave reset changed replay";return false;}
        else if (seconds==.5) waveHalf=color.bytes;
    }
    if (waveHalf==waveZero) {error="Metal rough water ignored evaluated wave geometry/normal time";return false;}

    const auto smoothMaterial=water::binding(baselineScene,videohelper::materialprogram::BackendTarget::Metal,error,0);
    if (!smoothMaterial) return false;
    const auto smoothPrepared=backend.prepare(baselineScene,smoothMaterial->nativeProgram());
    if (!smoothPrepared.prepared) {error=smoothPrepared.error;return false;}
    arbitgpu::NativeFixtureSceneRuntimeInputs smoothInputs;smoothInputs.linearColor=true;
    const auto smooth=backend.render(baselineScene,smoothPrepared.resources,9,9,smoothInputs);
    const auto smoothReplay=backend.render(baselineScene,smoothPrepared.resources,9,9,smoothInputs);
    Pixels smoothColor,smoothReplayColor;
    std::array<Pixels,3> absentReceipts;Pixels absentObjects,absentMaterials;
    if (!smooth.rendered || !smoothReplay.rendered || smooth.stats.roughGlassReceiptBytes!=0
        || smooth.stats.roughGlassReceiptComplete
        || !smooth.frame->readRawPass(renderpassoutput::Output::Color,smoothColor,error)
        || !smoothReplay.frame->readRawPass(renderpassoutput::Output::Color,smoothReplayColor,error)
        || smoothColor.bytes!=smoothReplayColor.bytes
        || arbitgpu::readMetalRoughGlassReceipt(smooth.frame,absentReceipts,absentObjects,absentMaterials,error)) {
        error="Metal zero rough water changed the released smooth receipt boundary";return false;
    }
    arbitgpu::smoothglass::Prepared owner;
    if (!arbitgpu::smoothglass::prepare(baselineScene,baselineProgram,{},owner,error,
        {arbitgpu::NativeFixtureMaterialBackend::Metal,true,true,true})) return false;
    const auto refuses=[&](std::array<Pixels,3> receipts,Pixels objects,Pixels materials) {
        arbitgpu::NativeFixtureSceneStats stats;
        return !arbitgpu::smoothglass::completeRoughFrame(owner,9,9,receipts[0],receipts[1],receipts[2],
            objects,materials,stats,error) && !stats.roughGlassReceiptComplete;
    };
    auto broken=baselineReceipts;broken[0].bytes.clear();
    if (!refuses(broken,baselineObjects,baselineMaterials)) return false;
    broken=baselineReceipts;const float incomplete=-1;
    std::memcpy(broken[2].bytes.data()+12,&incomplete,4);
    if (!refuses(broken,baselineObjects,baselineMaterials)) return false;
    broken=baselineReceipts;const float invalid=.5f;
    std::memcpy(broken[0].bytes.data(),&invalid,4);
    if (!refuses(broken,baselineObjects,baselineMaterials)) return false;
    auto stale=baselineObjects;const std::uint32_t id=999;
    std::memcpy(stale.bytes.data(),&id,4);
    if (!refuses(baselineReceipts,stale,baselineMaterials)) return false;
    const auto baselinePrepared=backend.prepare(baselineScene,baselineProgram);
    if (!baselinePrepared.prepared) {error=baselinePrepared.error;return false;}
    const auto staleScene=std::make_shared<const Visual3DScene>(*baselineScene);
    const auto staleFrame=backend.render(staleScene,baselinePrepared.resources,9,9,smoothInputs);
    if (staleFrame.rendered || staleFrame.frame) {
        error="Metal rough water published a stale scene owner";return false;
    }
    const auto grazing=std::make_shared<const Visual3DScene>(water::scene(80));
    const auto grazingMaterial=water::binding(grazing,videohelper::materialprogram::BackendTarget::Metal,error,.9f);
    if (!grazingMaterial) return false;
    const auto grazingPrepared=backend.prepare(grazing,grazingMaterial->nativeProgram());
    if (!grazingPrepared.prepared) {error=grazingPrepared.error;return false;}
    const auto unresolved=backend.render(grazing,grazingPrepared.resources,9,9,smoothInputs);
    if (unresolved.rendered || unresolved.frame || unresolved.stats.roughGlassReceiptComplete) {
        error="Metal strongly masked rough water published unresolved power";return false;
    }
    // Re-entering the same water surface cannot publish even a partial native frame.
    auto recursive=water::scene(0);
    for (int c=0;c<4;++c) {
        recursive.vertices[4+c]=recursive.vertices[c];
        recursive.vertices[4+c].position.z=5;recursive.vertices[4+c].normal={0,0,-1};
    }
    const std::uint32_t secondPlane[]={4,6,5,4,7,6};
    std::copy_n(secondPlane,6,recursive.indices.begin()+6);
    recursive.vertexCount=recursive.objects[0].vertexCount=8;
    recursive.indexCount=recursive.objects[0].indexCount=12;
    const auto recursiveScene=std::make_shared<const Visual3DScene>(recursive);
    const auto recursiveMaterial=water::binding(recursiveScene,videohelper::materialprogram::BackendTarget::Metal,error,.08f);
    if (!recursiveMaterial) return false;
    const auto recursivePrepared=backend.prepare(recursiveScene,recursiveMaterial->nativeProgram());
    if (!recursivePrepared.prepared) {error=recursivePrepared.error;return false;}
    arbitgpu::NativeFixtureSceneRuntimeInputs linear;linear.linearColor=true;
    const auto recursiveFrame=backend.render(recursiveScene,recursivePrepared.resources,9,9,linear);
    if (recursiveFrame.rendered || recursiveFrame.frame || recursiveFrame.stats.roughGlassReceiptComplete) {
        error="Metal rough water re-hit published a partial frame";return false;
    }
    error.clear();return true;
}

bool roughMetalCases(arbitgpu::NativeFixtureSceneBackend& backend, std::string& error)
{
    using namespace HarmonicMIDI::grid;
    namespace oracle = videohelper::tests::smoothglassfixture;
    using Pixels = arbitgpu::NativeRawPassPixels;
    struct Case { float roughness, dispersion, boundary, receiverZ; bool blocker; };
    // Receiver edge controls measure the rough footprint using actual triangles.
    // Dispersion, receiver distance and an intervening blocker change those rays.
    const Case cases[] = {{.04f,.004f,-.435f,-3,false}, {.08f,.004f,-.435f,-3,false},
        {.08f,0,-.435f,-3,false}, {.08f,.004f,-.435f,-4,false},
        {.08f,.004f,-6,-3,true}, {.08f,.004f,-.46f,-3,false}, {.08f,.004f,-.41f,-3,false}};
    bool sawNonzeroTail = false;
    std::shared_ptr<const Visual3DScene> baselineScene;
    std::shared_ptr<const arbitgpu::NativeFixtureSurfaceMaterialProgram> baselineProgram;
    std::shared_ptr<const arbitgpu::NativeFixtureSceneResources> baselineResources;
    std::array<Pixels,3> baselineReceipts;
    Pixels baselineObjects, baselineMaterials, baselineColor;
    for (const auto& test : cases) {
        auto authored = oracle::scene(.25f,1,20);
        oracle::receiver(authored,test.boundary);
        for (std::size_t v = 8; v < 12; ++v) authored.vertices[v].position.z = test.receiverZ;
        if (test.blocker) {
            oracle::receiver(authored,-6,{});
            for (std::size_t v = 12; v < 16; ++v) authored.vertices[v].position.z = -2;
        }
        const auto scene = std::make_shared<const Visual3DScene>(authored);
        const auto program = roughMetalProgram(scene,test.roughness,test.dispersion,error);
        if (!program) return false;
        const auto prepared = backend.prepare(scene,program);
        if (!prepared.prepared) { error = "Metal rough preparation failed: " + prepared.error; return false; }
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs; inputs.linearColor = true;
        const auto rendered = backend.render(scene,prepared.resources,9,9,inputs);
        if (!rendered.rendered || !rendered.frame || !rendered.stats.roughGlassReceiptComplete
            || rendered.stats.roughGlassCoveredPixels != 81 || rendered.stats.roughGlassReceiptBytes != 81*48
            || rendered.stats.roughGlassMaximumEnergyResidual > .00002
            || rendered.stats.roughGlassMaximumUnresolvedPower > .0001) {
            error = "Metal rough frame lacks its bounded native receipt: " + rendered.error; return false;
        }
        std::array<Pixels,3> receipts;
        Pixels objects, materials, color;
        if (!arbitgpu::readMetalRoughGlassReceipt(rendered.frame,receipts,objects,materials,error)
            || !rendered.frame->readRawPass(renderpassoutput::Output::Color,color,error)) return false;
        const double angle = 20*std::acos(-1.0)/180;
        const double extent = std::tan(authored.cameras[0].verticalFovRadians*.5);
        const oracle::D3 camera {authored.cameras[0].transform.translation.x,0,authored.cameras[0].transform.translation.z};
        for (std::size_t y = 0; y < 9; ++y) for (std::size_t x = 0; x < 9; ++x) {
            const std::size_t pixel = y*9+x;
            const double vx = (2*(double(x)+.5)/9-1)*extent;
            const double vy = (1-2*(double(y)+.5)/9)*extent;
            const auto ray = oracle::unit(oracle::D3{std::cos(angle)*vx-std::sin(angle),vy,-std::sin(angle)*vx-std::cos(angle)});
            const auto expected = oracle::traceRough(authored,camera,ray,1.5,test.dispersion,test.roughness);
            if (!expected.optics.complete) { error = "Independent Metal rough triangle oracle is incomplete"; return false; }
            std::uint32_t object=0, material=0;
            std::memcpy(&object,objects.bytes.data()+4*pixel,4);
            std::memcpy(&material,materials.bytes.data()+4*pixel,4);
            if (object != 101 || material != 6) { error = "Metal rough raster identity differs from oracle glass coverage"; return false; }
            std::array<std::array<float,4>,3> values;
            for (std::size_t attachment=0;attachment<3;++attachment)
                std::memcpy(values[attachment].data(),receipts[attachment].bytes.data()+16*pixel,16);
            if (values[0][3]!=1 || values[2][3]!=1
                || values[1][3]<1 || values[1][3]>3073 || std::floor(values[1][3])!=values[1][3]) {
                error = "Metal rough native status/query packing is incomplete"; return false;
            }
            for (std::size_t c=0;c<3;++c) {
                std::uint16_t half=0; std::memcpy(&half,color.bytes.data()+8*pixel+2*c,2);
                const double actual = hdrimage::halfToFloat(half);
                if (!std::isfinite(actual) || std::abs(actual-expected.optics.rgb[c])>.0015
                    || std::abs(values[0][c]-expected.energy.terminal[c])>.00002
                    || std::abs(values[1][c]-expected.energy.masked[c])>.00002
                    || std::abs(values[2][c]-expected.energy.tail[c])>.00002) {
                    error = "Metal rough native pixel/energy differs from the independent triangle optical reference"; return false;
                }
                sawNonzeroTail |= values[2][c]>0;
            }
        }
        const auto replay = backend.render(scene,prepared.resources,9,9,inputs);
        std::array<Pixels,3> replayReceipts; Pixels replayObjects,replayMaterials;
        if (!replay.rendered || !arbitgpu::readMetalRoughGlassReceipt(replay.frame,replayReceipts,replayObjects,replayMaterials,error)) return false;
        for (std::size_t i=0;i<3;++i) if (replayReceipts[i].bytes != receipts[i].bytes) {
            error = "Metal rough deterministic receipt changed on the same owned inputs"; return false;
        }
        if (!baselineScene) {
            baselineScene=scene; baselineProgram=program; baselineResources=prepared.resources;
            baselineReceipts=receipts; baselineObjects=objects; baselineMaterials=materials;baselineColor=color;
        }
    }
    if (!sawNonzeroTail) { error = "Metal rough fixtures did not exercise valid finite nonzero tail energy"; return false; }

    // The validator must reject altered copies of real native receipts.
    arbitgpu::smoothglass::Prepared receiptOwner;
    if (!arbitgpu::smoothglass::prepare(baselineScene,baselineProgram,{},receiptOwner,error,
        {arbitgpu::NativeFixtureMaterialBackend::Metal,true,true,true})) return false;
    const auto refuses = [&](const std::array<Pixels,3>& receipts, const Pixels& objects, const Pixels& materials) {
        arbitgpu::NativeFixtureSceneStats stats;
        return !arbitgpu::smoothglass::completeRoughFrame(receiptOwner,9,9,
            receipts[0],receipts[1],receipts[2],objects,materials,stats,error) && !stats.roughGlassReceiptComplete;
    };
    auto altered = baselineReceipts;
    altered[0].bytes.clear();
    if (!refuses(altered,baselineObjects,baselineMaterials)) return false;
    altered=baselineReceipts; altered[2].width=8;
    if (!refuses(altered,baselineObjects,baselineMaterials)) return false;
    const auto replace = [](Pixels& pixels, std::size_t component, float value) {
        std::memcpy(pixels.bytes.data()+component*4,&value,4);
    };
    for (const float value : {0.0f, -2.0f, std::numeric_limits<float>::quiet_NaN()}) {
        altered=baselineReceipts; replace(altered[0],3,value);
        if (!refuses(altered,baselineObjects,baselineMaterials)) return false;
    }
    altered=baselineReceipts; replace(altered[0],0,std::numeric_limits<float>::quiet_NaN());
    if (!refuses(altered,baselineObjects,baselineMaterials)) return false;
    altered=baselineReceipts; replace(altered[0],0,.5f);
    if (!refuses(altered,baselineObjects,baselineMaterials)) return false;
    altered=baselineReceipts; replace(altered[0],0,.999f); replace(altered[1],0,.001f); replace(altered[2],0,0);
    if (!refuses(altered,baselineObjects,baselineMaterials)) return false;
    altered=baselineReceipts; replace(altered[1],3,1.5f);
    if (!refuses(altered,baselineObjects,baselineMaterials)) return false;
    auto staleMaterial=baselineMaterials; const std::uint32_t wrongMaterial=7;
    std::memcpy(staleMaterial.bytes.data(),&wrongMaterial,4);
    if (!refuses(baselineReceipts,baselineObjects,staleMaterial)) return false;
    auto staleObjects=baselineObjects; const std::uint32_t nonglass=102;
    std::memcpy(staleObjects.bytes.data(),&nonglass,4);
    if (!refuses(baselineReceipts,staleObjects,baselineMaterials)) return false;

    arbitgpu::NativeFixtureSceneRuntimeInputs inputs; inputs.linearColor=true;
    // Force one small tile through the same native path and compare against
    // the unchanged one-job estimator, including its real GPU receipts.
    const auto oneTile=arbitgpu::renderMetalRoughGlassTiledFrame(baselineScene,baselineResources,9,9,inputs);
    std::array<Pixels,3> oneReceipts;Pixels oneObjects,oneMaterials,oneColor;
    if(!oneTile.rendered||!oneTile.frame
        ||!arbitgpu::readMetalRoughGlassReceipt(oneTile.frame,oneReceipts,oneObjects,oneMaterials,error)
        ||!oneTile.frame->readRawPass(renderpassoutput::Output::Color,oneColor,error)) return false;
    if(oneColor.bytes!=baselineColor.bytes||oneObjects.bytes!=baselineObjects.bytes||oneMaterials.bytes!=baselineMaterials.bytes) {
        error="Metal rough single tile changed full-frame Color or owner parity";return false;
    }
    for(unsigned i=0;i<3;++i)if(oneReceipts[i].bytes!=baselineReceipts[i].bytes) {
        error="Metal rough single tile changed the existing 16-stratum native estimator";return false;
    }
    std::shared_ptr<const arbitgpu::roughglasstiles::Schedule> oneSchedule;
    arbitgpu::roughglasstiles::Aggregate oneAggregate;
    if(!arbitgpu::readMetalRoughGlassTileAggregate(oneTile.frame,oneSchedule,oneAggregate,error)
        ||oneAggregate.completedTiles!=1||oneAggregate.completedPixels!=81||oneAggregate.coveredPixels!=81) return false;
    // Just above the released one-job limit, with an exact short edge tile.
    const auto tiled=backend.render(baselineScene,baselineResources,100,100,inputs);
    const auto reverse=arbitgpu::renderMetalRoughGlassTiledFrame(baselineScene,baselineResources,100,100,inputs,true);
    std::shared_ptr<const arbitgpu::roughglasstiles::Schedule> schedule,reverseSchedule;
    arbitgpu::roughglasstiles::Aggregate aggregate,reverseAggregate;
    Pixels tiledColor,reverseColor;
    if(!tiled.rendered||!reverse.rendered||!tiled.frame||!reverse.frame
        ||!arbitgpu::readMetalRoughGlassTileAggregate(tiled.frame,schedule,aggregate,error)
        ||!arbitgpu::readMetalRoughGlassTileAggregate(reverse.frame,reverseSchedule,reverseAggregate,error)
        ||!tiled.frame->readRawPass(renderpassoutput::Output::Color,tiledColor,error)
        ||!reverse.frame->readRawPass(renderpassoutput::Output::Color,reverseColor,error)) return false;
    if(schedule->tiles().size()!=2||aggregate.completedTiles!=2||aggregate.completedPixels!=10000
        ||aggregate.peakReceiptBytes>9708*48||aggregate.reservedQueries!=schedule->reservedQueries()
        ||aggregate.receiptClosestHitQueries>aggregate.reservedQueries
        ||aggregate.terminal!=reverseAggregate.terminal||aggregate.masked!=reverseAggregate.masked
        ||aggregate.tail!=reverseAggregate.tail||aggregate.receiptClosestHitQueries!=reverseAggregate.receiptClosestHitQueries
        ||tiledColor.bytes!=reverseColor.bytes||tiled.frame->width()!=100||tiled.frame->height()!=100
        ||tiled.frame->colorImageHandle()==0||tiled.frame->colorImageHandle()==reverse.frame->colorImageHandle()) {
        error="Metal rough tile order, bounded scratch, coverage or final GPU identity changed";return false;
    }
    std::vector<unsigned> coverage(10000);
    for(const auto& tile:schedule->tiles()) {
        if(tile.pixels()>9708||tile.reservedQueries()>arbitgpu::smoothglass::kMaximumPixelRayQueries) {
            error="Metal rough tile exceeded its unchanged per-job authority";return false;
        }
        for(unsigned y=tile.y;y<tile.y+tile.height;++y)for(unsigned x=tile.x;x<tile.x+tile.width;++x) ++coverage[y*100+x];
    }
    if(std::any_of(coverage.begin(),coverage.end(),[](unsigned n){return n!=1;})) {
        error="Metal rough edge tiles omitted or overlapped full-frame pixels";return false;
    }
    // Check independent full-frame camera rays on both sides of the tile edge.
    const auto& authored=*baselineScene;
    const double angle=20*std::acos(-1.0)/180,extent=std::tan(authored.cameras[0].verticalFovRadians*.5);
    const oracle::D3 camera{authored.cameras[0].transform.translation.x,0,authored.cameras[0].transform.translation.z};
    for(const auto xy:std::array<std::array<unsigned,2>,4>{{{20,0},{50,96},{50,97},{80,99}}}) {
        const double vx=(2*(double(xy[0])+.5)/100-1)*extent,vy=(1-2*(double(xy[1])+.5)/100)*extent;
        const auto ray=oracle::unit(oracle::D3{std::cos(angle)*vx-std::sin(angle),vy,-std::sin(angle)*vx-std::cos(angle)});
        const auto expected=oracle::traceRough(authored,camera,ray,1.5,.004,.04);
        if(!expected.optics.complete) {error="Metal rough edge oracle is incomplete";return false;}
        for(unsigned c=0;c<3;++c) {
            std::uint16_t half=0;std::memcpy(&half,tiledColor.bytes.data()+(xy[1]*100+xy[0])*8+c*2,2);
            if(std::abs(hdrimage::halfToFloat(half)-expected.optics.rgb[c])>.0015) {
                error="Metal rough tile offset changed the full-frame camera footprint";return false;
            }
        }
    }
    // Exact stale owners and a failed native tile publish nothing.
    const auto staleScene=std::make_shared<const Visual3DScene>(*baselineScene);
    const auto stale=arbitgpu::renderMetalRoughGlassTiledFrame(staleScene,baselineResources,100,100,inputs);
    if(stale.rendered||stale.frame) {error="Metal rough tiles published a stale scene owner";return false;}
    const auto failedProgram=roughMetalProgram(baselineScene,.9f,.004f,error);
    if(!failedProgram) return false;
    const auto failedPrepared=backend.prepare(baselineScene,failedProgram);
    const auto failed=arbitgpu::renderMetalRoughGlassTiledFrame(baselineScene,failedPrepared.resources,100,100,inputs,true);
    if(!failedPrepared.prepared||failed.rendered||failed.frame) {error="Metal failed rough tile published a partial frame";return false;}
    // 1080p/4K admission is shared source evidence only; parent owns the actual
    // hardware runs and timings, avoiding a broad native job inside this fixture.
    for(const auto wh:std::array<std::array<unsigned,2>,2>{{{1920,1080},{3840,2160}}}) {
        auto key=schedule->key();key.width=wh[0];key.height=wh[1];
        std::shared_ptr<const arbitgpu::roughglasstiles::Schedule> large;
        if(!arbitgpu::roughglasstiles::Schedule::admit(schedule->prepared(),key,false,large,error)
            ||large->pixels()!=std::uint64_t(wh[0])*wh[1]||large->maximumTilePixels()>9708) return false;
    }
    auto internal=std::make_shared<Visual3DScene>(*baselineScene);
    internal->cameras[0].transform={};
    const auto internalProgram=roughMetalProgram(internal,.08f,.004f,error);
    const auto internalPrepared=backend.prepare(internal,internalProgram);
    const auto internalFrame=backend.render(internal,internalPrepared.resources,9,9,inputs);
    if (!internalPrepared.prepared || internalFrame.rendered || internalFrame.frame
        || internalFrame.error.find("internal camera")==std::string::npos) {
        error="Metal rough glass failed its whole-frame internal-camera refusal"; return false;
    }
    const auto unsupportedProgram=roughMetalProgram(baselineScene,.9f,.004f,error);
    const auto unsupportedPrepared=backend.prepare(baselineScene,unsupportedProgram);
    const auto unsupported=backend.render(baselineScene,unsupportedPrepared.resources,9,9,inputs);
    if (!unsupportedPrepared.prepared || unsupported.rendered || unsupported.frame) {
        error="Metal rough glass silently published unsupported strong scattering"; return false;
    }

    // A foreground ordinary triangle must suppress glass receipts as well as Color.
    auto occluded=std::make_shared<Visual3DScene>(*baselineScene);
    oracle::receiver(*occluded,-6,{});
    for (std::size_t v=12;v<16;++v) occluded->vertices[v].position.z=2;
    const auto occludedProgram=roughMetalProgram(occluded,.08f,.004f,error);
    const auto occludedPrepared=backend.prepare(occluded,occludedProgram);
    const auto occludedFrame=backend.render(occluded,occludedPrepared.resources,9,9,inputs);
    std::array<Pixels,3> occludedReceipts; Pixels occludedObjects,occludedMaterials;
    if (!occludedFrame.rendered || occludedFrame.stats.roughGlassCoveredPixels != 0
        || !arbitgpu::readMetalRoughGlassReceipt(occludedFrame.frame,occludedReceipts,occludedObjects,occludedMaterials,error)) {
        error="Metal rough scratch did not preserve foreground opaque depth coverage"; return false;
    }
    for (const auto& pixels : occludedReceipts) for (const auto byte : pixels.bytes)
        if (byte != 0) { error="Metal nonglass scratch retains stale transport values"; return false; }
    const auto smoothProgram=roughMetalProgram(baselineScene,0,.004f,error);
    const auto smoothPrepared=backend.prepare(baselineScene,smoothProgram);
    const auto smoothFrame=backend.render(baselineScene,smoothPrepared.resources,9,9,inputs);
    if (!smoothFrame.rendered || smoothFrame.stats.roughGlassReceiptBytes != 0
        || smoothFrame.stats.roughGlassReceiptComplete
        || arbitgpu::readMetalRoughGlassReceipt(smoothFrame.frame,occludedReceipts,occludedObjects,occludedMaterials,error)) {
        error="Metal roughness zero changed the released smooth receipt boundary"; return false;
    }
    const auto walk=[](auto incoming,auto normal,double from,double to,double roughness,std::uint32_t seed) {
        return glasscaustictests::microfacetReference(incoming,normal,from,to,roughness,seed,32,16,.0001/128);
    };
    const auto read=[](const auto& frame,auto& receipts,auto& objects,auto& materials,std::string& error) {
        using namespace arbitgpu;
        if(!readMetalRoughGlassReceipt(frame,receipts,objects,materials,error))return false;
        std::shared_ptr<const roughglasstiles::Schedule> schedule;roughglasstiles::Aggregate aggregate;
        if(!readMetalRoughGlassTileAggregate(frame,schedule,aggregate,error))return false;
        std::uint64_t covered=0,queries=0,events=0;
        for(std::size_t pixel=0;pixel<std::uint64_t(objects.width)*objects.height;++pixel) {
            std::uint32_t object=0;std::memcpy(&object,objects.bytes.data()+pixel*4,4);
            if(object!=schedule->prepared().object.value)continue;
            ++covered;float query=0,event=0;
            std::memcpy(&query,receipts[1].bytes.data()+pixel*16+12,4);
            std::memcpy(&event,receipts[2].bytes.data()+pixel*16+12,4);
            queries+=static_cast<std::uint64_t>(query);events+=static_cast<std::uint64_t>(event);
        }
        if(aggregate.completedPixels!=25 ||aggregate.selectionQueries!=25 ||aggregate.transportStarts!=covered
            ||aggregate.coveredPixels!=covered||aggregate.receiptClosestHitQueries!=queries||aggregate.microEvents!=events
            ||aggregate.microWork!=events*smoothglass::kCameraMicroWorkPerEvent
            ||aggregate.completedTiles!=schedule->tiles().size()||aggregate.reservedQueries!=schedule->reservedQueries()) {
            error="Metal selected V2 actual selections, starts, queries and events differ from native pixel receipts";return false;
        }
        return covered==0||metalSelectedRoughBudgetGuardForFixture(frame,error);
    };
    if(!oracle::roughMultipleNativeCases(backend,walk,read,error))return false;
    // A second V2 tile consumes the preceding tile's completed native charge.
    // The original .16/.35 5x5 optical qualification rows remain unchanged.
    auto multiple=oracle::scene(.25f,1,20);multiple.lights[0].color={};oracle::receiver(multiple,-6,{1,1,1});
    const auto multiScene=std::make_shared<const Visual3DScene>(std::move(multiple));
    const auto multiMaterial=oracle::binding(multiScene,videohelper::materialprogram::BackendTarget::Metal,1.5f,.004f,error,.16f,2);
    if(!multiMaterial)return false;
    const auto multiPrepared=backend.prepare(multiScene,multiMaterial->nativeProgram());
    if(!multiPrepared.prepared) {error=multiPrepared.error;return false;}
    const auto forward=backend.render(multiScene,multiPrepared.resources,17,9,inputs);
    const auto reversed=arbitgpu::renderMetalRoughGlassTiledFrame(multiScene,multiPrepared.resources,17,9,inputs,true);
    std::shared_ptr<const arbitgpu::roughglasstiles::Schedule> forwardSchedule,reversedSchedule;
    arbitgpu::roughglasstiles::Aggregate forwardAggregate,reversedAggregate;
    Pixels forwardColor,reversedColor;
    if(!forward.rendered||!reversed.rendered||!forward.frame||!reversed.frame
        ||!arbitgpu::readMetalRoughGlassTileAggregate(forward.frame,forwardSchedule,forwardAggregate,error)
        ||!arbitgpu::readMetalRoughGlassTileAggregate(reversed.frame,reversedSchedule,reversedAggregate,error)
        ||!forward.frame->readRawPass(renderpassoutput::Output::Color,forwardColor,error)
        ||!reversed.frame->readRawPass(renderpassoutput::Output::Color,reversedColor,error))return false;
    if(forwardSchedule->tiles().size()!=2||reversedSchedule->tiles().size()!=2
        ||forwardAggregate.completedTiles!=2||forwardAggregate.completedPixels!=153||forwardAggregate.selectionQueries!=153
        ||forwardAggregate.transportStarts!=forwardAggregate.coveredPixels||forwardAggregate.coveredPixels==0
        ||forwardAggregate.microWork!=forwardAggregate.microEvents*arbitgpu::smoothglass::kCameraMicroWorkPerEvent
        ||forwardAggregate.microWork>arbitgpu::smoothglass::kMaximumCameraMicroWork
        ||forwardAggregate.receiptClosestHitQueries!=reversedAggregate.receiptClosestHitQueries
        ||forwardAggregate.microEvents!=reversedAggregate.microEvents||forwardAggregate.microWork!=reversedAggregate.microWork
        ||forwardAggregate.transportStarts!=reversedAggregate.transportStarts||forwardAggregate.selectionQueries!=reversedAggregate.selectionQueries
        ||forwardColor.bytes!=reversedColor.bytes
        ||!arbitgpu::metalSelectedRoughBudgetGuardForFixture(forward.frame,error)
        ||!arbitgpu::metalSelectedRoughBudgetGuardForFixture(reversed.frame,error)) {
        if(error.empty())error="Metal selected V2 multi-tile work, budget refusal or reverse-order Color changed";
        return false;
    }
    error.clear(); return true;
}

std::vector<std::uint8_t> readBgra8(
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame,
    renderpassoutput::Output output = renderpassoutput::Output::Color)
{
    if (!frame || !frame->passTextureDescriptor(output).complete())
        return {};
    sg_image image {};
    image.id = static_cast<std::uint32_t>(frame->passTextureDescriptor(output).imageHandle);
    const auto imageInfo = sg_mtl_query_image_info(image);
    if (imageInfo.active_slot < 0 || imageInfo.active_slot >= SG_NUM_INFLIGHT_FRAMES)
        return {};
    id<MTLTexture> texture = (__bridge id<MTLTexture>) imageInfo.tex[imageInfo.active_slot];
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>) sg_mtl_command_queue();
    if (texture == nil || queue == nil)
        return {};
    const auto bytesPerPixel = output == renderpassoutput::Output::Normal
        || output == renderpassoutput::Output::Emission ? 8u : output == renderpassoutput::Output::Mask ? 1u : 4u;
    const auto tightRowBytes = static_cast<std::size_t>(frame->width()) * bytesPerPixel;
    const auto rowBytes = (tightRowBytes + 255u) & ~std::size_t(255u);
    id<MTLBuffer> buffer = [texture.device newBufferWithLength:rowBytes * frame->height()
                                                   options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (buffer == nil || command == nil || blit == nil)
    {
#if ! __has_feature(objc_arc)
        [buffer release];
#endif
        return {};
    }
    [blit copyFromTexture:texture
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(frame->width(), frame->height(), 1)
                 toBuffer:buffer
        destinationOffset:0
   destinationBytesPerRow:rowBytes
 destinationBytesPerImage:rowBytes * frame->height()];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
    {
        std::cerr << "Metal fixture readback command failed";
        if (command.error != nil && command.error.localizedDescription != nil)
            std::cerr << ": " << command.error.localizedDescription.UTF8String;
        std::cerr << '\n';
#if ! __has_feature(objc_arc)
        [buffer release];
#endif
        return {};
    }

    std::vector<std::uint8_t> pixels(tightRowBytes * frame->height());
    const auto* source = static_cast<const std::uint8_t*>(buffer.contents);
    if (source != nullptr)
        for (std::uint32_t y = 0; y < frame->height(); ++y)
            std::copy_n(source + y * rowBytes, tightRowBytes,
                        pixels.data() + y * tightRowBytes);
#if ! __has_feature(objc_arc)
    [buffer release];
#endif
    return pixels;
}

std::shared_ptr<arbitgpu::NativeFixtureSurfaceMaterialProgram> makeProgram(
    const diffractionmaterial::Description& description,
    HarmonicMIDI::grid::SceneObjectId object,
    const char* identity)
{
    (void) identity;
    std::string error;
    const auto admitted = diffractionmaterial::admit(description, error);
    diffractionmaterial::SpectralIncidentLight incident;
    incident.radiance.fill(0.01f);
    diffractionmaterial::SpectralLightingPath lightingPath;
    lightingPath.kind = diffractionmaterial::LightingPathKind::Direct;
    lightingPath.incident = incident;
    auto program = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>();
    program->backend = arbitgpu::NativeFixtureMaterialBackend::Metal;
    program->kind = arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;
    program->object = object;
    program->bindingDigest = admitted ? admitted->structuralDigest() : std::string {};

    program->diffractionPathCount = 1;
    diffractionmaterial::LightingDescription lightingDescription;
    lightingDescription.pathCount = 1;
    lightingDescription.paths[0] = lightingPath;
    const auto admittedLighting = diffractionmaterial::AdmittedLightingPlan::admit(
        lightingDescription, error);
    if (admittedLighting)
    {
        program->diffractionLightingAdmission
            = std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(
                *admittedLighting);
        program->programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity(
            program->bindingDigest, *admittedLighting);
    }
    if (admitted)
        program->diffractionPaths[0] = diffractionmaterial::makeGpuLightingPath(
            diffractionmaterial::physicalcheckpoint::makeGpuParameters(*admitted, incident),
            lightingPath);

    return program;
}

bool backgroundOnly(const std::vector<std::uint8_t>& pixels)
{
    for (std::size_t index = 0; index + 3 < pixels.size(); index += 4)
        if (pixels[index] != 18u || pixels[index + 1] != 10u
            || pixels[index + 2] != 7u || pixels[index + 3] != 255u)
            return false;
    return true;
}

std::size_t nonBackgroundPixelCount(const std::vector<std::uint8_t>& pixels)
{
    std::size_t count = 0;
    for (std::size_t index = 0; index + 3 < pixels.size(); index += 4)
        if (pixels[index] != 18u || pixels[index + 1] != 10u
            || pixels[index + 2] != 7u || pixels[index + 3] != 255u)
            ++count;
    return count;
}

std::uint64_t fnv1a64(const std::vector<std::uint8_t>& bytes)
{
    std::uint64_t value = 14695981039346656037ull;
    for (const auto byte : bytes)
    {
        value ^= byte;
        value *= 1099511628211ull;
    }
    return value;
}

#if defined(ARBIT_HOLOGRAPHIC_TRADING_CARD_STRICT) \
    && ARBIT_HOLOGRAPHIC_TRADING_CARD_STRICT
std::size_t differingPixelCount(const std::vector<std::uint8_t>& left,
                                const std::vector<std::uint8_t>& right)
{
    if (left.size() != right.size() || left.size() % 4u != 0)
        return 0;
    std::size_t count = 0;
    for (std::size_t offset = 0; offset < left.size(); offset += 4u)
        if (!std::equal(left.begin() + static_cast<std::ptrdiff_t>(offset),
                        left.begin() + static_cast<std::ptrdiff_t>(offset + 4u),
                        right.begin() + static_cast<std::ptrdiff_t>(offset)))
            ++count;
    return count;
}
#endif
} // namespace

int main()
{
#if ! defined(__aarch64__) && ! defined(__arm64__)
    std::cerr << "FAIL: strict fixture material target requires physical Apple Silicon\n";
    return EXIT_FAILURE;
#else
    @autoreleasepool
    {
        auto& backend = arbitgpu::nativeFixtureSceneBackend();
        const auto info = backend.info();
        std::string linearError;
        const bool linearCaptured = linearscenechecks::capture(backend, linearError) != nullptr;
        if (!expect(linearCaptured, linearError.c_str()))
            return EXIT_FAILURE;
        if (!info.available || info.backend != "metal")
        {
            std::cerr << "FAIL: strict physical Metal fixture backend unavailable: "
                      << info.error << '\n';
            return EXIT_FAILURE;
        }
        std::string surfaceSourceError;
        if(!expect(videohelper::tests::restrictedsurfacesource::run(backend,surfaceSourceError),surfaceSourceError.c_str()))
            return EXIT_FAILURE;
        if(!exactLineNativeProbe())return EXIT_FAILURE;
        if(!cameraRayCoefficientNativeProbe())return EXIT_FAILURE;
        if(!cameraLinePredicateNativeProbe())return EXIT_FAILURE;
        if(!cameraWideLinePredicateNativeProbe())return EXIT_FAILURE;
        if(!cameraPixelSignNativeProbe())return EXIT_FAILURE;
        if (!expect(backend.supportsRetainedNoteMotion(),
            "Metal fixture consumer explicitly supports retained note Motion")) return EXIT_FAILURE;
        std::string glassError;
        const bool glassRendered=videohelper::tests::smoothglassfixture::run(backend,glassError);
        if (!expect(glassRendered,glassError.c_str())) return EXIT_FAILURE;
        std::string roughGlassError;
        const bool roughWaterRendered=roughWaterMetalCases(backend,roughGlassError);
        if (!expect(roughWaterRendered,roughGlassError.c_str())) return EXIT_FAILURE;
        const bool roughGlassRendered=roughMetalCases(backend,roughGlassError);
        if (!expect(roughGlassRendered,roughGlassError.c_str())) return EXIT_FAILURE;
        std::string causticError;
        const bool causticRendered=metalCausticCases(backend,causticError);
        if(!expect(causticRendered,causticError.c_str())) return EXIT_FAILURE;
        const bool multipleScatterRendered=metalMultipleScatterCausticCases(backend,causticError);
        if(!expect(multipleScatterRendered,causticError.c_str())) return EXIT_FAILURE;
        const auto readAuthoredCaustic=[](const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& owner,
            arbitgpu::glasscaustics::NativeReadback& receipt,arbitgpu::glasscaustics::EnergyLedger& energy,std::string& error) {
            arbitgpu::glasscaustics::Frame schedule;
            return arbitgpu::readMetalGlassCausticReceipt(owner,schedule,receipt,energy,error);
        };
        const bool authoredRoughCausticRendered=videohelper::tests::authoredroughcaustic::nativeCases(
            backend,videohelper::materialprogram::BackendTarget::Metal,readAuthoredCaustic,causticError);
        if(!expect(authoredRoughCausticRendered,causticError.c_str())) return EXIT_FAILURE;
        const bool glassGratingIncidentRendered=metalGlassGratingIncidentCases(backend,causticError);
        if(!expect(glassGratingIncidentRendered,causticError.c_str())) return EXIT_FAILURE;
        if(!expect(metalGlassCameraGratingTraceCases(causticError),causticError.c_str())) return EXIT_FAILURE;
        if(!expect(metalDirectionalIrradianceCohortCases(causticError),causticError.c_str())) return EXIT_FAILURE;
        if(!expect(metalGlassGratingTerminalCases(backend,causticError),causticError.c_str())) return EXIT_FAILURE;
        if(!expect(metalGlassGratingTerminalCases(backend,causticError,true),causticError.c_str())) return EXIT_FAILURE;
        for(unsigned cameraCase=1;cameraCase<=3;++cameraCase)
            if(!expect(metalGlassGratingTerminalCases(backend,causticError,true,cameraCase),causticError.c_str())) return EXIT_FAILURE;
        if(!expect(metalGlassGratingTerminalCases(backend,causticError,true,0,true),causticError.c_str())) return EXIT_FAILURE;
        if(!expect(metalGlassGratingTerminalCases(backend,causticError,true,0,true,true),causticError.c_str())) return EXIT_FAILURE;
        std::string waterError;
        const bool waterRendered=videohelper::tests::waterfixture::run(backend,waterError);
        if (!expect(waterRendered,waterError.c_str())) return EXIT_FAILURE;
        std::string environmentError;
        const bool environmentRendered=videohelper::tests::environmentImageCases(backend,environmentError);
        if (!expect(environmentRendered,environmentError.c_str())) return EXIT_FAILURE;
        std::string skyError;
        const bool skyRendered=videohelper::tests::analyticSkyCases(backend,skyError);
        if (!expect(skyRendered,skyError.c_str())) return EXIT_FAILURE;
        std::string reflectionError;
        const bool reflected = videohelper::tests::baseSurfaceReflectionCases(backend, reflectionError);
        if (!expect(reflected, reflectionError.c_str()))
            return EXIT_FAILURE;
        std::string visibilityError;
        const bool visible = videohelper::tests::opaqueVisibilityCases(backend, visibilityError);
        if (!expect(visible, visibilityError.c_str()))
            return EXIT_FAILURE;
        if (!videohelper::tests::materialFrameBindingCases(backend, [](const auto& frame)
            {
                auto pixels = readBgra8(frame);
                for (std::size_t i = 0; i < pixels.size(); i += 4)
                    std::swap(pixels[i], pixels[i + 2]);
                return pixels;
            })) return EXIT_FAILURE;

        std::string orientationError;
        if (!expect(videohelper::tests::orientation::materialOrientationCases(backend, orientationError),
                    "embedded textures and native Frames preserve four-quadrant text orientation"))
        {
            std::cerr << orientationError << '\n';
            return EXIT_FAILURE;
        }

        auto rasterSceneValue = videohelper::fixture3d::makeScene();
        rasterSceneValue.vertexCount = 3;
        rasterSceneValue.indexCount = 3;
        // Metal treats counter-clockwise clip-space triangles as front-facing here.
        // Keep the indexed control winding explicit so either index-order or
        // face-winding mutations break the paired positive/negative controls.
        rasterSceneValue.indices[0] = 0;
        rasterSceneValue.indices[1] = 1;
        rasterSceneValue.indices[2] = 2;
        rasterSceneValue.objects[0].vertexCount = 3;
        rasterSceneValue.objects[0].indexCount = 3;
        rasterSceneValue.objects[0].transform = {};
        rasterSceneValue.materials[0].baseColor = { 1.0f, 0.0f, 0.0f };
        rasterSceneValue.materials[0].baseColorTexture = {};
        rasterSceneValue.materials[0].doubleSided = true;
        rasterSceneValue.textureCount = 0;
        rasterSceneValue.textureTexelCount = 0;
        rasterSceneValue.textureTexels.clear();
        rasterSceneValue.lightCount = 0;
        rasterSceneValue.ambientColor = { 1.0f, 1.0f, 1.0f };
        auto rasterScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(
            rasterSceneValue);
        const auto rasterPreparation = backend.prepare(rasterScene, nullptr);
        const auto rasterFrame = backend.render(
            rasterScene, rasterPreparation.resources, 96, 96, {});
        const auto rasterPixels = readBgra8(rasterFrame.frame);

        rasterSceneValue.materials[0].doubleSided = false;
        auto frontFacingRasterScene
            = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(rasterSceneValue);
        const auto frontFacingRasterPreparation = backend.prepare(frontFacingRasterScene, nullptr);
        const auto frontFacingRasterFrame = backend.render(
            frontFacingRasterScene, frontFacingRasterPreparation.resources, 96, 96, {});
        const auto frontFacingRasterPixels = readBgra8(frontFacingRasterFrame.frame);

        auto backFacingRasterSceneValue = rasterSceneValue;
        backFacingRasterSceneValue.indices[1] = 2;
        backFacingRasterSceneValue.indices[2] = 1;
        auto backFacingRasterScene
            = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(backFacingRasterSceneValue);
        const auto backFacingRasterPreparation = backend.prepare(backFacingRasterScene, nullptr);
        const auto backFacingRasterFrame = backend.render(
            backFacingRasterScene, backFacingRasterPreparation.resources, 96, 96, {});
        const auto backFacingRasterPixels = readBgra8(backFacingRasterFrame.frame);

        const auto rasterControlOk = expect(
            rasterPreparation.prepared && rasterFrame.rendered
                && rasterPixels.size() == 96u * 96u * 4u
                && nonBackgroundPixelCount(rasterPixels) > 0,
            "the double-sided constant-red triangle must prove indexed Metal rasterization and BGRA readback")
            && expect(frontFacingRasterPreparation.prepared && frontFacingRasterFrame.rendered
                && frontFacingRasterPixels.size() == 96u * 96u * 4u
                && nonBackgroundPixelCount(frontFacingRasterPixels) > 0,
                "the counter-clockwise front-facing triangle must survive production back-face culling")
            && expect(backFacingRasterPreparation.prepared && backFacingRasterFrame.rendered
                && backFacingRasterPixels.size() == 96u * 96u * 4u
                && nonBackgroundPixelCount(backFacingRasterPixels) == 0,
                "the reversed-winding back-facing triangle must be fully culled by the production pipeline");

        auto surfaceSceneValue = videohelper::fixture3d::makeScene();
        surfaceSceneValue.objects[0].transform.rotation
            = { -0.16773126f, 0.25488700f, 0.04494346f, 0.95125124f };
        surfaceSceneValue.lightCount = 1;
        surfaceSceneValue.ambientColor = { 0.12f, 0.12f, 0.12f };
        surfaceSceneValue.lights[0].kind = HarmonicMIDI::grid::SceneLightKind::Directional;
        surfaceSceneValue.lights[0].transform = {};
        surfaceSceneValue.lights[0].transform.rotation
            = surfaceSceneValue.objects[0].transform.rotation;
        surfaceSceneValue.lights[0].color = { 1.0f, 1.0f, 1.0f };
        surfaceSceneValue.lights[0].intensity = 0.88f;
        surfaceSceneValue.materials[0].baseColorTexture = {};

        const auto oracle = surfacematerialstarteroracle::load(
            SURFACE_MATERIAL_STARTER_ORACLE_PATH);
        bool starterOk = true;
        auto outputSceneValue = videohelper::fixture3d::makeScene();
        outputSceneValue.objects[0].transform = {};
        outputSceneValue.cameras[0].transform = {};
        outputSceneValue.cameras[0].transform.translation.z = 5.0f;
        outputSceneValue.materials[0].emissive = { 0.8f, 0.08f, 0.02f };
        auto outputScene = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(outputSceneValue);
        auto outputResources = backend.prepare(outputScene, nullptr);
        starterOk &= expect(outputResources.prepared, "Metal output inspection scene prepares");
        const auto rawFrame = backend.render(outputScene, outputResources.resources, 64, 64, {});
        starterOk &= expect(rawFrame.rendered && rawFrame.frame != nullptr, "Metal raw passes share one native scene draw");
        if (rawFrame.frame)
        {
            rawpasschecks::verify(rawFrame.frame, [&](bool condition, const char* message)
                { starterOk &= expect(condition, message); });
            for (const auto output : renderpassoutput::kOutputs)
            {
                auto expected = readBgra8(rawFrame.frame, output);
                if (output == renderpassoutput::Output::Color)
                    for (std::size_t offset = 0; offset + 3 < expected.size(); offset += 4)
                        std::swap(expected[offset], expected[offset + 2]);
                arbitgpu::NativeRawPassPixels exported;
                std::string readError;
                starterOk &= expect(rawFrame.frame->readRawPass(output, exported, readError)
                    && exported.bytes == expected, "Metal export preserves top-first raw bytes and normalizes BGRA Image to RGBA");
            }
            const auto center = 32u * 64u + 32u;
            const auto stationaryMotion = readBgra8(rawFrame.frame, renderpassoutput::Output::Motion);
            starterOk &= expect(stationaryMotion.size() == 64u * 64u * 4u
                && std::all_of(stationaryMotion.begin(), stationaryMotion.end(), [](auto byte) { return byte == 0; }),
                "Metal first-frame Motion is raw zero RG16F, not display gray");
            // Use production canonical note rows and retained transforms, never row-index history.
            canonicalblockc::FrameProducer motionNoteProducer;
            canonicalblockc::FrameKey motionNoteKey;
            motionNoteKey.projectGeneration = motionNoteKey.sourceGeneration = motionNoteKey.helperGeneration = 1;
            motionNoteKey.backendGeneration = motionNoteKey.deviceGeneration = motionNoteKey.scoreGeneration = 1;
            motionNoteKey.beatMapGeneration = motionNoteKey.fpsGeneration = motionNoteKey.loopGeneration = 1;
            motionNoteKey.seekGeneration = 1; motionNoteKey.fps = 60.0;
            auto motionScore = std::make_shared<arbitmod::Score>();
            motionScore->notationVersion = motionScore->scoreRevision = 1;
            motionScore->rootFreq = 440.0f;
            arbitmod::Note motionNote;
            motionNote.id = -7; motionNote.trackId = 1; motionNote.startBeat = 0;
            motionNote.lengthBeats = 4; motionNote.durationSeconds = 2;
            motionNote.velocity = 40; motionNote.freqHz = 440;
            motionScore->notes.push_back(motionNote);
            auto secondNote = motionNote; secondNote.id = 19; secondNote.freqHz = 880;
            motionScore->notes.push_back(secondNote);
            arbitgpu::NativeFixtureSceneRuntimeInputs previousNotes;
            previousNotes.canonicalBlockCFrame = motionNoteProducer.evaluate(motionNoteKey, motionScore, 0.0f);
            visualnoteinstancing::Mapping motionMapping;
            motionMapping.x = visualnoteinstancing::MappingAxis::Velocity;
            motionMapping.xScale = 1; motionMapping.y = visualnoteinstancing::MappingAxis::LogFrequency;
            motionMapping.yScale = 1; motionMapping.zScale = 0; motionMapping.meshScale = 0.3f;
            previousNotes.noteInstanceMapping = motionMapping;
            const auto retainedNotes = arbitgpu::sceneMotionSample(previousNotes);
            auto movedScore = std::make_shared<arbitmod::Score>(*motionScore);
            for (auto& row : movedScore->notes) row.velocity = 80;
            std::reverse(movedScore->notes.begin(), movedScore->notes.end());
            ++movedScore->scoreRevision; ++movedScore->notationVersion;
            ++motionNoteKey.scoreGeneration; motionNoteKey.frame = 1;
            auto currentNotes = previousNotes;
            currentNotes.canonicalBlockCFrame = motionNoteProducer.evaluate(motionNoteKey, movedScore, 0.0f);
            currentNotes.previousMotion = retainedNotes;
            const auto noteMoved = backend.render(outputScene, outputResources.resources, 64, 64, currentNotes);
            const auto noteRepeated = backend.render(outputScene, outputResources.resources, 64, 64, currentNotes);
            const auto noteMotion = readBgra8(noteMoved.frame, renderpassoutput::Output::Motion);
            bool positiveNoteX = false, validNoteVectors = noteMotion.size() == 64u * 64u * 4u;
            for (std::size_t offset = 0; offset + 3 < noteMotion.size(); offset += 4) {
                std::uint16_t nx = 0, ny = 0;
                std::memcpy(&nx, noteMotion.data() + offset, 2);
                std::memcpy(&ny, noteMotion.data() + offset + 2, 2);
                positiveNoteX |= nx > 0 && nx < 0x7c00u;
                validNoteVectors &= ((nx & 0x7fffu) == 0 || (nx & 0x8000u) == 0)
                    && (nx & 0x7c00u) != 0x7c00u && (ny & 0x7fffu) == 0;
            }
            starterOk &= expect(retainedNotes.noteTransforms && retainedNotes.noteTransforms->count == 2
                && noteMoved.rendered && noteRepeated.rendered && positiveNoteX && validNoteVectors
                && noteMotion == readBgra8(noteRepeated.frame, renderpassoutput::Output::Motion),
                "Metal signed stable note identities retain preceding transforms across row reorder");
            auto reversedHistory = std::make_shared<arbitgpu::NativeSceneNoteMotionTransforms>(
                retainedNotes.noteTransforms ? *retainedNotes.noteTransforms : arbitgpu::NativeSceneNoteMotionTransforms {});
            std::swap(reversedHistory->identities[0], reversedHistory->identities[1]);
            std::swap(reversedHistory->transforms[0], reversedHistory->transforms[1]);
            currentNotes.previousMotion->noteTransforms = reversedHistory;
            const auto reorderedHistory = backend.render(outputScene, outputResources.resources, 64, 64, currentNotes);
            starterOk &= expect(reorderedHistory.rendered
                && readBgra8(reorderedHistory.frame, renderpassoutput::Output::Motion) == noteMotion,
                "Metal note Motion is independent of retained table order");
            const auto neutralNotes = [&](arbitgpu::NativeFixtureSceneRuntimeInputs inputs) {
                const auto rendered = backend.render(outputScene, outputResources.resources, 64, 64, inputs);
                const auto bytes = readBgra8(rendered.frame, renderpassoutput::Output::Motion);
                bool zero = bytes.size() == 64u * 64u * 4u;
                for (std::size_t offset = 0; offset + 1 < bytes.size(); offset += 2) {
                    std::uint16_t value = 0; std::memcpy(&value, bytes.data() + offset, 2);
                    zero &= (value & 0x7fffu) == 0;
                }
                return rendered.rendered && zero;
            };
            currentNotes.previousMotion = arbitgpu::sceneMotionSample(currentNotes);
            starterOk &= expect(neutralNotes(currentNotes), "Metal unchanged note transforms produce neutral Motion");
            currentNotes.previousMotion->noteTransforms = std::make_shared<const arbitgpu::NativeSceneNoteMotionTransforms>();
            currentNotes.objectTranslationOffset[0] = 0.25f;
            currentNotes.cameraTranslationOffset[1] = 0.25f;
            starterOk &= expect(neutralNotes(currentNotes), "Metal new note identities neutralize outer object and camera Motion");
            currentNotes.previousMotion.reset();
            starterOk &= expect(neutralNotes(currentNotes), "Metal absent note predecessor produces neutral Motion");
            arbitgpu::NativeSceneMotionHistory notePreview, noteExport;
            starterOk &= expect(notePreview.commit(0,60,1,64,64,previousNotes),
                "Metal note fixture retains only a successful preceding input");
            for (const auto predecessor : {notePreview.predecessor(4,60,1,64,64,{},previousNotes),
                    notePreview.predecessor(1,30,1,64,64,{},previousNotes),
                    notePreview.predecessor(1,60,1,32,64,{},previousNotes),
                    notePreview.predecessor(1,60,1,64,64,{},currentNotes),
                    noteExport.predecessor(1,60,1,64,64,{},previousNotes)}) {
                currentNotes.previousMotion = predecessor;
                starterOk &= expect(!predecessor && neutralNotes(currentNotes),
                    "Metal note seek/rate/extent/source resets and Export isolation render neutral Motion");
            }

            arbitgpu::NativeFixtureSceneRuntimeInputs moving;
            moving.previousMotion = arbitgpu::NativeSceneMotionSample {};
            moving.objectTranslationOffset[0] = 0.25f;
            moving.passComposite = renderpasscomposite::Parameters {};
            moving.passComposite->mode = renderpasscomposite::Mode::MotionView;
            moving.passComposite->amount = 0;
            const auto moved = backend.render(outputScene, outputResources.resources, 64, 64, moving);
            const auto repeated = backend.render(outputScene, outputResources.resources, 64, 64, moving);
            const auto motion = readBgra8(moved.frame, renderpassoutput::Output::Motion);
            std::uint16_t x = 0, y = 1;
            if (motion.size() == 64u * 64u * 4u)
            { std::memcpy(&x, motion.data() + center * 4u, 2); std::memcpy(&y, motion.data() + center * 4u + 2u, 2); }
            starterOk &= expect(moved.rendered && repeated.rendered
                && moved.stats.drawCount == rawFrame.stats.drawCount
                && moved.frame->passTextureDescriptor(renderpassoutput::Output::Motion).format
                    == arbitgpu::NativeTexturePixelFormat::Rg16Float
                && x > 0 && x < 0x7c00u && (y & 0x7fffu) == 0
                && motion == readBgra8(repeated.frame, renderpassoutput::Output::Motion)
                && readBgra8(moved.frame) == readBgra8(repeated.frame),
                "Metal object motion is positive, finite and repeatable without another scene draw");
            auto directMotion = moving;
            directMotion.passComposite.reset();
            directMotion.imageOutput = renderpassoutput::Output::Motion;
            const auto direct = backend.render(outputScene, outputResources.resources, 64, 64, directMotion);
            starterOk &= expect(direct.rendered && readBgra8(direct.frame) == readBgra8(moved.frame)
                && readBgra8(direct.frame, renderpassoutput::Output::Motion) == motion,
                "Metal Motion Image selection matches the raw-pass inspector and preserves raw vectors");
            moving.objectTranslationOffset[0] = 0;
            moving.objectTranslationOffset[1] = 0.25f;
            const auto movedUp = backend.render(outputScene, outputResources.resources, 64, 64, moving);
            const auto upMotion = readBgra8(movedUp.frame, renderpassoutput::Output::Motion);
            const auto upImage = readBgra8(movedUp.frame);
            x = y = 0;
            if (upMotion.size() == 64u * 64u * 4u)
            { std::memcpy(&x, upMotion.data() + center * 4u, 2); std::memcpy(&y, upMotion.data() + center * 4u + 2u, 2); }
            starterOk &= expect(movedUp.rendered && (x & 0x7fffu) == 0
                && y > 0 && y < 0x7c00u && upImage[center * 4u + 1u] > upImage[center * 4u + 2u],
                "Metal geometric Y-up Motion remains positive in RG16F and the Motion View image");
            moving.objectTranslationOffset[1] = 0;
            moving.objectTranslationOffset[0] = 0;
            moving.cameraTranslationOffset[0] = 0.25f;
            const auto cameraMoved = backend.render(outputScene, outputResources.resources, 64, 64, moving);
            const auto cameraMotion = readBgra8(cameraMoved.frame, renderpassoutput::Output::Motion);
            x = 0;
            if (cameraMotion.size() == 64u * 64u * 4u) std::memcpy(&x, cameraMotion.data() + center * 4u, 2);
            starterOk &= expect(cameraMoved.rendered && (x & 0x8000u) != 0 && (x & 0x7fffu) != 0,
                "Metal camera translation produces negative motion");
            {
                // Timeline Geometry3D keeps each vertex's prior position. Every
                // vertex sat 0.25 units to the left one frame earlier, so the
                // projected displacement matches the rigid 0.25 offset above.
                const auto positions = [&](float offset)
                {
                    auto values = std::make_shared<std::vector<HarmonicMIDI::grid::SceneVec3>>();
                    for (std::size_t index = 0; index < outputScene->vertexCount; ++index)
                    {
                        auto position = outputScene->vertices[index].position;
                        position.x += offset;
                        values->push_back(position);
                    }
                    return values;
                };
                const auto zeroMotion = [](const std::vector<std::uint8_t>& bytes)
                {
                    bool zero = bytes.size() == 64u * 64u * 4u;
                    for (std::size_t offset = 0; offset + 1 < bytes.size(); offset += 2) {
                        std::uint16_t value = 0; std::memcpy(&value, bytes.data() + offset, 2);
                        zero &= (value & 0x7fffu) == 0;
                    }
                    return zero;
                };
                arbitgpu::NativeFixtureSceneRuntimeInputs deformed;
                deformed.imageOutput = renderpassoutput::Output::Motion;
                deformed.previousMotion = arbitgpu::NativeSceneMotionSample {};
                deformed.previousMotion->scenePositions = positions(-0.25f);
                const auto deformedFrame = backend.render(outputScene, outputResources.resources, 64, 64, deformed);
                const auto deformedMotion = readBgra8(deformedFrame.frame, renderpassoutput::Output::Motion);
                std::uint16_t rigidX = 0, deformedX = 0, deformedY = 1;
                if (motion.size() == 64u * 64u * 4u) std::memcpy(&rigidX, motion.data() + center * 4u, 2);
                if (deformedMotion.size() == 64u * 64u * 4u)
                {
                    std::memcpy(&deformedX, deformedMotion.data() + center * 4u, 2);
                    std::memcpy(&deformedY, deformedMotion.data() + center * 4u + 2u, 2);
                }
                starterOk &= expect(deformedFrame.rendered && rigidX > 0 && rigidX < 0x7c00u
                    && std::abs(hdrimage::halfToFloat(deformedX) - hdrimage::halfToFloat(rigidX)) < 0.01f
                    && (deformedY & 0x7fffu) == 0,
                    "Metal deformed vertex moves by the projected displacement from its retained prior position");
                deformed.previousMotion->scenePositions = positions(0.0f);
                const auto staticFrame = backend.render(outputScene, outputResources.resources, 64, 64, deformed);
                starterOk &= expect(staticFrame.rendered
                    && zeroMotion(readBgra8(staticFrame.frame, renderpassoutput::Output::Motion)),
                    "Metal static deformation gives zero motion");
                auto shortened = std::make_shared<std::vector<HarmonicMIDI::grid::SceneVec3>>(*positions(0.0f));
                shortened->pop_back();
                deformed.previousMotion->scenePositions = shortened;
                const auto mismatched = backend.render(outputScene, outputResources.resources, 64, 64, deformed);
                starterOk &= expect(!mismatched.rendered
                    && mismatched.error.find("prior Geometry3D positions") != std::string::npos,
                    "Metal refuses prior positions with a different vertex count");
            }
            for (const auto output : { renderpassoutput::Output::MaterialId, renderpassoutput::Output::ObjectId })
            {
                const auto descriptor = rawFrame.frame->passTextureDescriptor(output);
                const auto bytes = readBgra8(rawFrame.frame, output);
                std::uint32_t actual = 0, background = 1;
                if (bytes.size() == 64u * 64u * 4u)
                { std::memcpy(&actual, bytes.data() + center * 4u, 4); std::memcpy(&background, bytes.data(), 4); }
                const auto expected = output == renderpassoutput::Output::MaterialId
                    ? outputSceneValue.objects[0].material.value : outputSceneValue.objects[0].id.value;
                starterOk &= expect(descriptor.format == arbitgpu::NativeTexturePixelFormat::R32Uint
                    && actual == expected && background == 0, "Metal integer attachments preserve scene IDs");
            }
            const auto normal = readBgra8(rawFrame.frame, renderpassoutput::Output::Normal);
            std::array<std::uint16_t, 4> components {};
            if (normal.size() == 64u * 64u * 8u) std::memcpy(components.data(), normal.data() + center * 8u, 8);
            starterOk &= expect(components[0] == 0 && components[1] == 0 && components[2] == 0x3c00u,
                "Metal raw normal contains signed XYZ values before display conversion");
            for (const auto mode : { renderpasscomposite::Mode::DepthFog, renderpasscomposite::Mode::ObjectMatte })
            {
                arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
                inputs.passComposite = renderpasscomposite::Parameters {};
                inputs.passComposite->mode = mode;
                inputs.passComposite->farDepth = 0.001f;
                inputs.passComposite->fogColor = { 0.2f, 0.4f, 0.6f };
                inputs.passComposite->identity = outputSceneValue.objects[0].id.value;
                const auto composed = backend.render(outputScene, outputResources.resources, 64, 64, inputs);
                const auto pixels = composed.frame ? readBgra8(composed.frame) : std::vector<std::uint8_t> {};
                starterOk &= expect(composed.rendered && composed.stats.reusedStaticResources
                    && pixels.size() == 64u * 64u * 4u, "Metal pass composition retains native scene resource reuse");
                if (pixels.size() == 64u * 64u * 4u)
                    starterOk &= expect(std::abs(static_cast<int>(pixels[center * 4u])
                        - (mode == renderpasscomposite::Mode::DepthFog ? 153 : 255)) <= 1,
                        "Metal depth fog and ID matte consume raw scene values");
            }
        }
        {
            arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
            inputs.passProgram = renderpassfixture::hdrBranches();
            inputs.passComposite = inputs.passProgram->steps[inputs.passProgram->output].parameters;
            const auto composed = backend.render(outputScene, outputResources.resources, 64, 64, inputs);
            const auto pixels = composed.frame ? readBgra8(composed.frame) : std::vector<std::uint8_t> {};
            const auto expected = renderpassfixture::expectedSdr();
            starterOk &= expect(composed.rendered && pixels.size() == 64u * 64u * 4u,
                "Metal HDR branches use the native scene compositor");
            if (pixels.size() == 64u * 64u * 4u)
                for (std::size_t channel = 0; channel < 3; ++channel)
                    starterOk &= expect(std::abs(static_cast<int>(pixels[(32u * 64u + 32u) * 4u + 2u - channel])
                        - expected[channel]) <= 1,
                        "Metal linear HDR and sRGB conversion match the shared branch reference");
            const auto repeated = backend.render(outputScene, outputResources.resources, 64, 64, inputs);
            starterOk &= expect(repeated.rendered && readBgra8(repeated.frame) == pixels,
                "Metal pass branching is repeatable for preview and export");
            inputs.passProgram->steps[2].inputA = 3;
            starterOk &= expect(!backend.render(outputScene, outputResources.resources, 64, 64, inputs).rendered,
                "Metal rejects a forward pass reference");
        }
        for (const auto output : { renderpassoutput::Output::Depth, renderpassoutput::Output::Normal,
                                  renderpassoutput::Output::Motion,
                                  renderpassoutput::Output::Emission, renderpassoutput::Output::Mask,
                                  renderpassoutput::Output::MaterialId, renderpassoutput::Output::ObjectId })
        {
            arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
            inputs.imageOutput = output;
            const auto frame = backend.render(outputScene, outputResources.resources, 64, 64, inputs);
            starterOk &= expect(frame.rendered && frame.frame != nullptr,
                                "each Render 3D inspection output draws on Metal");
            if (!frame.frame) continue;
            const auto pixels = readBgra8(frame.frame);
            if (pixels.size() != 64u * 64u * 4u)
            {
                starterOk = false;
                continue;
            }
            const auto centerPixel = (32u * 64u + 32u) * 4u;
            std::array<float, 3> expected {};
            if (output == renderpassoutput::Output::Normal) expected = { 0.5f, 0.5f, 1.0f };
            if (output == renderpassoutput::Output::Motion) expected = { 0.5f, 0.5f, 0.5f };
            if (output == renderpassoutput::Output::Emission) expected = { 0.8f, 0.08f, 0.02f };
            if (output == renderpassoutput::Output::Mask) expected = { 1.0f, 1.0f, 1.0f };
            if (output == renderpassoutput::Output::MaterialId)
                expected = render3dimage::identityColor(outputSceneValue.objects[0].material.value);
            if (output == renderpassoutput::Output::ObjectId)
                expected = render3dimage::identityColor(outputSceneValue.objects[0].id.value);
            if (output == renderpassoutput::Output::Depth)
            {
                const auto& camera = outputSceneValue.cameras[0];
                const auto depth = (4.0f - camera.nearPlane) / (camera.farPlane - camera.nearPlane);
                expected = { depth, depth, depth };
            }
            for (std::size_t channel = 0; channel < 3; ++channel)
                starterOk &= expect(std::abs(static_cast<int>(pixels[centerPixel + 2u - channel])
                    - static_cast<int>(std::lround(expected[channel] * 255.0f))) <= 1,
                    "Metal inspection pixels contain the requested native quantity");
            starterOk &= expect(pixels[centerPixel + 3u] == 255 && pixels[3] == 0,
                                "Metal inspection retains coverage and transparent background");
            const auto repeated = backend.render(outputScene, outputResources.resources, 64, 64, inputs);
            starterOk &= expect(repeated.rendered && readBgra8(repeated.frame) == pixels
                                    && repeated.stats.reusedStaticResources,
                                "Metal output inspection is deterministic and reuses native geometry");
        }
        outputResources.resources.reset();
        std::vector<std::uint8_t> sampledStarterPixels;
        const auto center = (32u * 65u + 32u) * 4u;
        for (std::size_t starterIndex = 0;
             starterIndex < surfacematerialstarterfixture::kPrograms.size(); ++starterIndex)
        {
            const auto& starter = surfacematerialstarterfixture::kPrograms[starterIndex];
            const auto& expected = oracle.materials[starterIndex];
            starterOk = expect(expected.id == starter.id,
                               "generated programs and independent JSON oracle must share stable IDs")
                && starterOk;
            std::string generatedError;
            auto generatedScene = videohelper::tests::generatedSurfaceScene(surfaceSceneValue, generatedError);
            starterOk &= expect(generatedScene.has_value(), "Metal exact starter uses admitted generated Geometry Core transport");
            if (!generatedScene) continue;
            auto starterSceneValue = std::move(*generatedScene);
            starterSceneValue.materials[0].id = {starter.materialIdValue()};
            starterSceneValue.objects[0].material = {starter.materialIdValue()};
            const auto starterScene
                = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(
                    std::move(starterSceneValue));

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
            std::string starterError;
            const auto admittedProgram = surfacematerial::admit(request.program, starterError);
            if (admittedProgram.has_value())
                request.binding.surfaceMaterialDigest = admittedProgram->structuralDigest();
            const auto binding = videorender::fixture3d::admitSurfaceMaterialBinding(
                starterScene, request, videohelper::materialprogram::BackendTarget::Metal,
                starterError);
            starterOk = expect(binding != nullptr,
                               "each generated exact starter program must pass Metal admission")
                && starterOk;
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
                    exactBlock &= std::abs(actualFloatLanes[lane][channel]
                        - expected.pbr[lane * 4 + channel]) < 0.000001f;
            exactBlock &= block.identifiers[0]
                == static_cast<std::uint32_t>(expected.pbr[16]);
            starterOk = expect(exactBlock,
                               "native Metal lowering must match the independent JSON PBR oracle")
                && starterOk;

            videorender::fixture3d::FixtureSceneRenderer previewOwner(backend);
            videorender::fixture3d::FixtureSceneRenderer exportOwner(backend);
            videorender::fixture3d::RenderedFrame preview;
            videorender::fixture3d::RenderedFrame exportFrame;
            const bool previewRendered = previewOwner.renderPreview(
                starterScene, binding, { 65, 65 },
                videorender::fixture3d::kNativeGpuCapability, preview, starterError);
            const bool exportRendered = exportOwner.renderExport(
                starterScene, binding, { 65, 65 },
                videorender::fixture3d::kNativeGpuCapability, exportFrame, starterError);
            const auto previewPixels = previewRendered
                ? readBgra8(preview.nativeFrame) : std::vector<std::uint8_t> {};
            const auto exportPixels = exportRendered
                ? readBgra8(exportFrame.nativeFrame) : std::vector<std::uint8_t> {};
            starterOk = expect(previewRendered && exportRendered
                                   && preview.use == videorender::fixture3d::RenderUse::Preview
                                   && exportFrame.use == videorender::fixture3d::RenderUse::Export
                                   && preview.nativeFrame != nullptr
                                   && exportFrame.nativeFrame != nullptr
                                   && preview.nativeFrame->colorImageHandle()
                                       != exportFrame.nativeFrame->colorImageHandle(),
                               "distinct production preview and export owners must render Metal frames")
                && expect(!previewPixels.empty() && previewPixels == exportPixels,
                          "preview and export owners must produce exact matching Metal pixels")
                && starterOk;
            if (starterIndex == 0)
            {
                auto vertexRequest = request;
                vertexRequest.vertexModifier = videohelper::test::audioNormalDisplacement();
                const auto vertexBinding = videorender::fixture3d::admitSurfaceMaterialBinding(
                    starterScene, vertexRequest, videohelper::materialprogram::BackendTarget::Metal, starterError);
                const auto prepared = vertexBinding ? backend.prepare(starterScene, vertexBinding->nativeProgram())
                                                    : arbitgpu::NativeFixtureScenePreparation {};
                starterOk &= expect(vertexBinding != nullptr && prepared.prepared,
                                    "audio vertex material reaches Metal shader compilation");
                if (vertexBinding)
                {
                    auto mismatched = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(
                        *vertexBinding->nativeProgram());
                    mismatched->programIdentity = binding->nativeProgram()->programIdentity;
                    starterOk &= expect(!backend.prepare(starterScene, mismatched).prepared,
                        "Metal rejects a substituted vertex shader identity");
                }
                if (prepared.prepared)
                {
                    arbitgpu::NativeFixtureSceneRuntimeInputs quiet, loud;
                    loud.vertexSpectrum[6] = 0.5f;
                    const auto quietFrame = backend.render(starterScene, prepared.resources, 65, 65, quiet);
                    const auto loudFrame = backend.render(starterScene, prepared.resources, 65, 65, loud);
                    const auto replayFrame = backend.render(starterScene, prepared.resources, 65, 65, quiet);
                    const auto quietPixels = quietFrame.rendered ? readBgra8(quietFrame.frame) : std::vector<std::uint8_t> {};
                    const auto loudPixels = loudFrame.rendered ? readBgra8(loudFrame.frame) : std::vector<std::uint8_t> {};
                    starterOk &= expect(!quietPixels.empty() && quietPixels == previewPixels
                        && !loudPixels.empty() && loudPixels != quietPixels
                        && replayFrame.rendered && readBgra8(replayFrame.frame) == quietPixels,
                        "Metal vertex displacement follows audio and replay restores original geometry");
                    loud.vertexSpectrum[6] = 1.0f;
                    const auto boundedFrame = backend.render(starterScene, prepared.resources, 65, 65, loud);
                    starterOk &= expect(boundedFrame.rendered && readBgra8(boundedFrame.frame) == loudPixels,
                        "Metal vertex displacement remains inside its authored bound");
                    videorender::fixture3d::RenderedFrame vertexExport;
                    starterOk &= expect(exportOwner.renderExport(starterScene, vertexBinding, {}, loud,
                        {65, 65}, videorender::fixture3d::kNativeGpuCapability, vertexExport, starterError)
                        && readBgra8(vertexExport.nativeFrame) == loudPixels,
                        "Metal vertex deformation preview and export have identical pixels");
                    {
                        // Motion runs the vertex program again at the prior
                        // frame's spectrum. Compare it with the same displacement
                        // given as retained prior positions on a displaced mesh.
                        const auto readMotion = [&](const auto& frame)
                        {
                            std::vector<float> values(65u * 65u * 2u, -99.0f);
                            const auto bytes = frame ? readBgra8(frame, renderpassoutput::Output::Motion)
                                                     : std::vector<std::uint8_t> {};
                            if (bytes.size() != values.size() * 2u) return values;
                            for (std::size_t index = 0; index < values.size(); ++index)
                            {
                                std::uint16_t value = 0;
                                std::memcpy(&value, bytes.data() + index * 2u, 2);
                                values[index] = hdrimage::halfToFloat(value);
                            }
                            return values;
                        };
                        auto steady = loud;
                        steady.imageOutput = renderpassoutput::Output::Motion;
                        steady.previousMotion = arbitgpu::sceneMotionSample(loud);
                        const auto steadyFrame = backend.render(starterScene, prepared.resources, 65, 65, steady);
                        const auto steadyMotion = readMotion(steadyFrame.frame);
                        starterOk &= expect(steadyFrame.rendered && std::all_of(steadyMotion.begin(), steadyMotion.end(),
                                [](float value) { return value == 0; }),
                            "Metal vertex program with unchanged time and audio gives zero motion");
                        auto rising = steady;
                        rising.previousMotion->vertexSpectrum[6] = 0.0f;
                        const auto risingFrame = backend.render(starterScene, prepared.resources, 65, 65, rising);
                        const auto risingMotion = readMotion(risingFrame.frame);
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
                        starterOk &= expect(risingFrame.rendered && retainedFrame.rendered
                                && movingValues > 0 && different * 100u <= risingMotion.size(),
                            "Metal vertex program motion equals the projected displacement of its prior evaluation");
                        rising.previousMotion->scenePositions
                            = std::make_shared<const std::vector<HarmonicMIDI::grid::SceneVec3>>(original);
                        const auto combined = backend.render(starterScene, prepared.resources, 65, 65, rising);
                        starterOk &= expect(!combined.rendered
                                && combined.error.find("reads Normal over deformed geometry") != std::string::npos,
                            "Metal keeps refusing a vertex program that reads Normal over retained deformed positions");

                        // A position and time program over a deformed mesh moving at
                        // a constant velocity. Motion must project
                        // program(current pose, t) - program(prior pose, t_prev). The
                        // expected motion comes from the retained-position path with
                        // both program results computed on the CPU.
                        auto timeRequest = request;
                        constexpr float rate = 0.6f, currentTime = 0.5f, priorTime = 0.25f;
                        const HarmonicMIDI::grid::SceneVec3 velocity {0.0f, 0.1f, 0.0f};
                        timeRequest.vertexModifier = videohelper::test::timeAxisDisplacement(rate);
                        const auto timeBinding = videorender::fixture3d::admitSurfaceMaterialBinding(
                            starterScene, timeRequest, videohelper::materialprogram::BackendTarget::Metal, starterError);
                        const auto timePrepared = timeBinding
                            ? backend.prepare(starterScene, timeBinding->nativeProgram())
                            : arbitgpu::NativeFixtureScenePreparation {};
                        starterOk &= expect(timeBinding != nullptr && timePrepared.prepared,
                            "a position and time vertex program reaches Metal shader compilation");
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
                            if (!programFrame.rendered) std::cerr << "Metal time program Motion: " << programFrame.error << '\n';
                            const auto programMotion = readMotion(programFrame.frame);

                            const auto retainedMotionOf = [&](float priorPoseWeight)
                            {
                                // priorPoseWeight 1 uses the prior pose, 0 the current
                                // pose as the input of the prior evaluation.
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
                            starterOk &= expect(programFrame.rendered && moving > 0 && different * 100u <= programMotion.size()
                                    && fromCurrentPose * 100u > programMotion.size(),
                                "Metal vertex program over deformed geometry moves by the program applied to the prior pose");
                        }
                    }
                }
            }
            if (previewPixels.size() > center + 3u)
            {
                const std::array<std::uint8_t, 4> actualRgba {{
                    previewPixels[center + 2u], previewPixels[center + 1u],
                    previewPixels[center], previewPixels[center + 3u]
                }};
                starterOk = expect(actualRgba == expected.centerRgba8,
                                   "Metal center pixels must match the pinned JSON oracle")
                    && expect(std::find(expected.rejectedAlternatives.begin(),
                                        expected.rejectedAlternatives.end(), actualRgba)
                                  == expected.rejectedAlternatives.end(),
                              "Metal pixels must reject wrong shading alternatives")
                    && starterOk;
                sampledStarterPixels.insert(sampledStarterPixels.end(),
                                            actualRgba.begin(), actualRgba.end());

                const auto rejectsControlledMaterialAlternative = [&] (const char* label,
                                                                        auto mutate)
                {
                    auto alternative = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(
                        *binding->nativeProgram());
                    alternative->bindingDigest += label;
                    alternative->programIdentity += label;
                    mutate(alternative->parameters);
                    const auto prepared = backend.prepare(starterScene, alternative);
                    if (!prepared.prepared) return false;
                    const auto rendered = backend.render(starterScene, prepared.resources, 65, 65, {});
                    const auto pixels = rendered.rendered
                        ? readBgra8(rendered.frame) : std::vector<std::uint8_t> {};
                    if (pixels.size() <= center + 3u) return false;
                    const std::array<std::uint8_t, 4> value {{
                        pixels[center + 2u], pixels[center + 1u],
                        pixels[center], pixels[center + 3u] }};
                    return value != expected.centerRgba8;
                };
                if (starterIndex == 0)
                {
                    starterOk = expect(rejectsControlledMaterialAlternative("/base-color", [] (auto& p)
                        { p.baseColorMetallic[0] = 0.9f; }),
                        "Metal lit pixels must detect a wrong base-color binding") && starterOk;
                    starterOk = expect(rejectsControlledMaterialAlternative("/roughness", [] (auto& p)
                        { p.emissionRoughness[3] = 0.0f; }),
                        "Metal lit pixels must detect a wrong roughness binding") && starterOk;
                    starterOk = expect(rejectsControlledMaterialAlternative("/emission", [] (auto& p)
                        { p.emissionRoughness[0] = 0.25f; }),
                        "Metal lit pixels must detect a wrong emission binding") && starterOk;
                }
                if (starterIndex == 1)
                    starterOk = expect(rejectsControlledMaterialAlternative("/metallic", [] (auto& p)
                        { p.baseColorMetallic[3] = 0.0f; }),
                        "Metal lit pixels must detect a wrong metallic binding") && starterOk;
                if (starterIndex == 2)
                {
                    starterOk = expect(rejectsControlledMaterialAlternative("/opacity", [] (auto& p)
                        { p.normalOpacity[3] = 1.0f; }),
                        "Metal lit pixels must detect a wrong opacity binding") && starterOk;
                    starterOk &= expect(rejectsControlledMaterialAlternative("/transmission", [] (auto& p)
                        { p.transmissionIorClearcoat[0] = 0.0f; }),
                        "Metal lit pixels must detect a wrong transmission binding");
                    starterOk &= expect(rejectsControlledMaterialAlternative("/ior", [] (auto& p)
                        { p.transmissionIorClearcoat[1] = 2.8f; }),
                        "Metal lit pixels must detect a wrong IOR binding");
                }
                if (starterIndex == 3)
                    starterOk &= expect(rejectsControlledMaterialAlternative("/clearcoat", [] (auto& p)
                        { p.transmissionIorClearcoat[2] = 0.0f; }),
                        "Metal lit pixels must detect a wrong clearcoat binding");

                if (starterIndex != 0)
                    continue;
                auto wrongNormalSceneValue = *starterScene;
                for (std::size_t vertex = 0; vertex < wrongNormalSceneValue.vertexCount; ++vertex)
                    wrongNormalSceneValue.vertices[vertex].normal
                        = { -wrongNormalSceneValue.vertices[vertex].normal.x,
                            -wrongNormalSceneValue.vertices[vertex].normal.y,
                            -wrongNormalSceneValue.vertices[vertex].normal.z };
                const auto wrongNormalScene
                    = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(
                        std::move(wrongNormalSceneValue));
                const auto wrongNormalPreparation = backend.prepare(
                    wrongNormalScene, binding->nativeProgram());
                std::vector<std::uint8_t> wrongNormalPixels;
                if (wrongNormalPreparation.prepared)
                {
                    const auto wrongNormalRender = backend.render(
                        wrongNormalScene, wrongNormalPreparation.resources, 65, 65, {});
                    if (wrongNormalRender.rendered)
                        wrongNormalPixels = readBgra8(wrongNormalRender.frame);
                }
                const std::array<std::uint8_t, 4> wrongNormalRgba {{
                    wrongNormalPixels.size() > center + 3u ? wrongNormalPixels[center + 2u] : std::uint8_t { 0 },
                    wrongNormalPixels.size() > center + 3u ? wrongNormalPixels[center + 1u] : std::uint8_t { 0 },
                    wrongNormalPixels.size() > center + 3u ? wrongNormalPixels[center] : std::uint8_t { 0 },
                    wrongNormalPixels.size() > center + 3u ? wrongNormalPixels[center + 3u] : std::uint8_t { 0 } }};
                starterOk = expect(wrongNormalPixels.size() > center + 3u
                                       && wrongNormalRgba != actualRgba,
                                   "Metal lit pixels must detect a wrong transformed normal")
                    && starterOk;
            }
        }
        const auto sampledDigest = fnv1a64(sampledStarterPixels);
        char sampledDigestText[17] {};
        std::snprintf(sampledDigestText, sizeof(sampledDigestText), "%016llx",
                      static_cast<unsigned long long>(sampledDigest));
        starterOk = expect(sampledStarterPixels.size() == 20
                               && oracle.sampledRgba8Fnv1a64 == sampledDigestText,
                           "all five Metal samples must match the pinned JSON digest")
            && starterOk;
        videohelper::Sha256 sampledSha256;
        sampledSha256.update(sampledStarterPixels.data(), sampledStarterPixels.size());
        starterOk = expect(oracle.sampledRgba8Sha256 == sampledSha256.finishHex(),
                           "all five Metal samples must match the pinned SHA-256 digest")
            && starterOk;

        auto malformedSceneValue = surfaceSceneValue;
        malformedSceneValue.lightCount = 0;
        malformedSceneValue.materials[0].id = { 1u };
        malformedSceneValue.objects[0].material = { 1u };
        const auto malformedScene
            = std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(malformedSceneValue);
        surfacematerialbinding::ImportedSceneMaterialRequest malformedRequest;
        malformedRequest.scene = malformedScene->id;
        malformedRequest.sceneRevision = 1;
        malformedRequest.structuralRevision = 1;
        malformedRequest.evaluationRevision = 1;
        malformedRequest.programRevision = 1;
        malformedRequest.sceneSnapshot = malformedScene;
        malformedRequest.program = surfacematerialstarterfixture::kPrograms[0].program();
        malformedRequest.binding.targetKind
            = surfacematerialbinding::BindingTargetKind::ObjectOverride;
        malformedRequest.binding.object = videohelper::fixture3d::kCubeObjectId;
        malformedRequest.binding.surfaceMaterialRevision = 1;
        std::string malformedError;
        const auto validMalformedBase = surfacematerial::admit(
            malformedRequest.program, malformedError);
        if (validMalformedBase.has_value())
            malformedRequest.binding.surfaceMaterialDigest
                = validMalformedBase->structuralDigest();
        malformedRequest.program.operations[0].literal[0]
            = std::numeric_limits<float>::quiet_NaN();
        const auto malformedValueBinding
            = videorender::fixture3d::admitSurfaceMaterialBinding(
                malformedScene, malformedRequest,
                videohelper::materialprogram::BackendTarget::Metal, malformedError);
        starterOk = expect(malformedValueBinding == nullptr,
                           "malformed Surface PBR values must reject before Metal allocation")
            && starterOk;
        malformedRequest.program = surfacematerialstarterfixture::kPrograms[0].program();
        malformedRequest.binding.surfaceMaterialDigest = std::string(64, '0');
        const auto malformedIdentityBinding
            = videorender::fixture3d::admitSurfaceMaterialBinding(
                malformedScene, malformedRequest,
                videohelper::materialprogram::BackendTarget::Metal, malformedError);
        starterOk = expect(malformedIdentityBinding == nullptr,
                           "malformed Surface PBR identity must reject before Metal allocation")
            && starterOk;

        std::string cardLoadError;
        const auto cardAsset = videohelper::fixture3d::loadHolographicTradingCardScene(
            cardLoadError);
        if (!cardAsset)
        {
            std::cerr << "FAIL: exact card GLB production decode/adaptation failed: "
                      << cardLoadError << '\n';
            return EXIT_FAILURE;
        }
        const auto& scene = cardAsset->scene;
        starterOk = diffractionenvironmenttests::run(backend, scene, cardAsset->operation,
            videohelper::materialprogram::BackendTarget::Metal,
            [](const auto& frame) { return readBgra8(frame); }) && starterOk;
        bool ok = expect(cardAsset->assetId
                             == "builtin.visual-model.holographic-trading-card"
                         && cardAsset->contentSha256
                             == "5e3433aa19cc4636c33a1f6dd53a4a05fdffa39d788ad565b530d8dfdcce0287"
                         && cardAsset->sceneName == "Holographic Trading Card Scene"
                         && cardAsset->objectName == "Trading Card"
                         && cardAsset->materialName == "Physical Diffraction Target"
                         && cardAsset->cameraName == "Card Camera"
                         && cardAsset->lightName == "Card Key"
                         && scene->vertexCount == 24u && scene->indexCount == 36u,
                         "strict Metal must retain the exact card asset and named decoded records");

#if defined(ARBIT_HOLOGRAPHIC_TRADING_CARD_STRICT) \
    && ARBIT_HOLOGRAPHIC_TRADING_CARD_STRICT
        const auto& cardRequest = *cardAsset->operation.diffractionMaterial;
        std::string previewError, exportError, wrongObjectError, strictError;
        const auto previewBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
            scene, cardRequest, videohelper::materialprogram::BackendTarget::Metal,
            previewError);
        const auto exportBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
            scene, cardRequest, videohelper::materialprogram::BackendTarget::Metal,
            exportError);
        auto zeroSlope = cardRequest;
        zeroSlope.spatialFoil->physicalBsdf.roughness.rmsSlope = 0.0f;
        zeroSlope.material = zeroSlope.spatialFoil->physicalBsdf;
        const auto admittedZeroSlope = diffractivefoil::admit(*zeroSlope.spatialFoil,
                                                              strictError);
        zeroSlope.structuralDigest = admittedZeroSlope
            ? admittedZeroSlope->structuralDigest() : std::string {};
        const auto zeroSlopeBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
            scene, zeroSlope, videohelper::materialprogram::BackendTarget::Metal,
            strictError);
        auto wrongObject = cardRequest;
        wrongObject.object = HarmonicMIDI::grid::SceneObjectId { 2 };
        const auto wrongObjectBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
            scene, wrongObject, videohelper::materialprogram::BackendTarget::Metal,
            wrongObjectError);
        auto changedGroove = cardRequest;
        changedGroove.spatialFoil->grooveField[2].grooveSpacingNanometres += 113.0f;
        const auto admittedGroove = diffractivefoil::admit(*changedGroove.spatialFoil, strictError);
        changedGroove.structuralDigest = admittedGroove
            ? admittedGroove->structuralDigest() : std::string {};
        const auto grooveBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
            scene, changedGroove, videohelper::materialprogram::BackendTarget::Metal,
            strictError);
        auto changedOccupancy = cardRequest;
        changedOccupancy.spatialFoil->grooveField[2].diffractionCoverage = 0.0f;
        const auto admittedOccupancy = diffractivefoil::admit(
            *changedOccupancy.spatialFoil, strictError);
        changedOccupancy.structuralDigest = admittedOccupancy
            ? admittedOccupancy->structuralDigest() : std::string {};
        const auto occupancyBinding = videorender::fixture3d::admitDiffractionMaterialBinding(
            scene, changedOccupancy, videohelper::materialprogram::BackendTarget::Metal,
            strictError);
        videorender::fixture3d::FixtureSceneRenderer previewOwner(backend);
        videorender::fixture3d::FixtureSceneRenderer exportOwner(backend);
        videorender::fixture3d::FixtureSceneRenderer grooveOwner(backend);
        videorender::fixture3d::FixtureSceneRenderer occupancyOwner(backend);
        videorender::fixture3d::RenderedFrame previewFrame, exportFrame, grooveFrame,
            occupancyFrame;
        const auto capability = videorender::fixture3d::kNativeGpuCapability;
        const bool rendered = previewBinding && exportBinding && grooveBinding
            && occupancyBinding && previewBinding != exportBinding
            && previewBinding->nativeProgram() != exportBinding->nativeProgram()
            && wrongObjectBinding == nullptr
            && previewOwner.renderPreview(scene, previewBinding, {}, { 320, 180 },
                                          capability, previewFrame, strictError)
            && exportOwner.renderExport(scene, exportBinding, {}, { 320, 180 },
                                        capability, exportFrame, strictError)
            && grooveOwner.renderPreview(scene, grooveBinding, {}, { 320, 180 },
                                         capability, grooveFrame, strictError)
            && occupancyOwner.renderPreview(scene, occupancyBinding, {}, { 320, 180 },
                                            capability, occupancyFrame, strictError);
        const auto previewPixels = readBgra8(previewFrame.nativeFrame);
        const auto exportPixels = readBgra8(exportFrame.nativeFrame);
        const auto groovePixels = readBgra8(grooveFrame.nativeFrame);
        const auto occupancyPixels = readBgra8(occupancyFrame.nativeFrame);
        const auto disabledPreparation = backend.prepare(scene, nullptr);
        const auto disabledFrame = backend.render(
            scene, disabledPreparation.resources, 320, 180, {});
        const auto disabledPixels = readBgra8(disabledFrame.frame);
        const auto previewOwnerId = previewFrame.nativeFrame
            ? previewFrame.nativeFrame->nativeResourceCacheIdentity() : 0;
        const auto exportOwnerId = exportFrame.nativeFrame
            ? exportFrame.nativeFrame->nativeResourceCacheIdentity() : 0;
        const auto grooveDifferingPixels = differingPixelCount(previewPixels, groovePixels);
        const auto occupancyDifferingPixels = differingPixelCount(previewPixels, occupancyPixels);
        const auto disabledDifferingPixels = differingPixelCount(previewPixels, disabledPixels);
        const bool strictCardAccepted = rendered && disabledPreparation.prepared
            && disabledFrame.rendered && zeroSlopeBinding == nullptr
            && previewOwnerId != 0 && exportOwnerId != 0 && previewOwnerId != exportOwnerId
            && previewBinding->bindingDigest() == exportBinding->bindingDigest()
            && previewBinding->nativeProgram()->programIdentity
                == exportBinding->nativeProgram()->programIdentity
            && previewPixels.size() == 320u * 180u * 4u
            && exportPixels.size() == previewPixels.size()
            && groovePixels.size() == previewPixels.size()
            && occupancyPixels.size() == previewPixels.size()
            && disabledPixels.size() == previewPixels.size()
            && previewPixels == exportPixels
            && grooveDifferingPixels > 100u
            && occupancyDifferingPixels > 100u
            && disabledDifferingPixels > 100u;
        if (!strictCardAccepted)
        {
            std::cerr << "Strict Metal card operands: rendered=" << rendered
                      << " disabled-prepared=" << disabledPreparation.prepared
                      << " disabled-rendered=" << disabledFrame.rendered
                      << " preview-owner=" << previewOwnerId
                      << " export-owner=" << exportOwnerId
                      << " source-operation=" << cardAsset->operation.sourceStableId
                      << " render-operation=" << cardAsset->operation.renderStableId
                      << " preview-bytes=" << previewPixels.size()
                      << " export-bytes=" << exportPixels.size()
                      << " groove-bytes=" << groovePixels.size()
                      << " occupancy-bytes=" << occupancyPixels.size()
                      << " disabled-bytes=" << disabledPixels.size()
                      << " preview-digest=" << std::hex << fnv1a64(previewPixels)
                      << " export-digest=" << fnv1a64(exportPixels)
                      << " groove-digest=" << fnv1a64(groovePixels)
                      << " occupancy-digest=" << fnv1a64(occupancyPixels)
                      << " disabled-digest=" << fnv1a64(disabledPixels) << std::dec
                      << " groove-differing-pixels=" << grooveDifferingPixels
                      << " occupancy-differing-pixels=" << occupancyDifferingPixels
                      << " disabled-differing-pixels=" << disabledDifferingPixels
                      << " preview-error=" << previewError
                      << " export-error=" << exportError
                      << " wrong-object-error=" << wrongObjectError
                      << " final-error=" << strictError
                      << " preview-program="
                      << (previewBinding ? previewBinding->nativeProgram()->programIdentity
                                         : std::string {})
                      << " export-program="
                      << (exportBinding ? exportBinding->nativeProgram()->programIdentity
                                        : std::string {})
                      << '\n';
        }
        else
        {
            std::cout << "Metal holographic card: PASS; preview-owner=" << previewOwnerId
                      << "; export-owner=" << exportOwnerId
                      << "; source-operation=" << cardAsset->operation.sourceStableId
                      << "; render-operation=" << cardAsset->operation.renderStableId
                      << "; bytes=" << previewPixels.size()
                      << "; preview-export-digest=" << std::hex << fnv1a64(previewPixels)
                      << "; groove-digest=" << fnv1a64(groovePixels)
                      << "; occupancy-digest=" << fnv1a64(occupancyPixels)
                      << "; disabled-digest=" << fnv1a64(disabledPixels) << std::dec
                      << "; groove-differing-pixels=" << grooveDifferingPixels
                      << "; occupancy-differing-pixels=" << occupancyDifferingPixels
                      << "; disabled-differing-pixels=" << disabledDifferingPixels
                      << "; zero-slope-rejected=1\n";
        }
        ok &= expect(strictCardAccepted,
                     "strict Metal card fixture must bind one handler operation to distinct preview/export owners and meaningful same-size alternatives");
        return ok ? 0 : 1;
#endif

        auto description = diffractionmaterial::makeAluminiumBinaryGratingPreset();
        auto program = makeProgram(description, scene->objects[0].id,
                                   "metal-card-diffraction-program");
        auto preparation = backend.prepare(scene, program);
        auto replayedLightingProgram
            = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*program);
        auto replayedLightingDescription
            = replayedLightingProgram->diffractionLightingAdmission->description();
        replayedLightingDescription.paths[0].incident.radiance[1]
            = std::nextafter(
                replayedLightingDescription.paths[0].incident.radiance[1], 0.0f);
        std::string replayedLightingError;
        const auto replayedLighting = diffractionmaterial::AdmittedLightingPlan::admit(
            replayedLightingDescription, replayedLightingError);
        const auto replayedMaterial = diffractionmaterial::admit(
            description, replayedLightingError);
        if (replayedLighting && replayedMaterial)
        {
            replayedLightingProgram->diffractionLightingAdmission
                = std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(
                    *replayedLighting);
            replayedLightingProgram->diffractionPaths[0]
                = diffractionmaterial::makeGpuLightingPath(
                    diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                        *replayedMaterial,
                        replayedLightingDescription.paths[0].incident),
                    replayedLightingDescription.paths[0]);
        }
        const auto replayedLightingPreparation = backend.prepare(
            scene, replayedLightingProgram);
        auto malformedProgram
            = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*program);
        malformedProgram->programIdentity = "metal-fixture-malformed-diffraction-program";
        malformedProgram->diffractionPaths[0].material.spectralZ[7].w
            = std::numeric_limits<float>::quiet_NaN();
        const auto malformedPreparation = backend.prepare(scene, malformedProgram);
        auto layoutSentinelProgram
            = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*program);
        layoutSentinelProgram->programIdentity
            = "metal-fixture-layout-sentinel-diffraction-program";
        layoutSentinelProgram->parameters.identifiers[3] = 1u;
        const auto layoutSentinelPreparation = backend.prepare(
            scene, layoutSentinelProgram);
        auto divergentBlazeProgram
            = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*program);
        divergentBlazeProgram->programIdentity
            = "metal-fixture-divergent-local-blaze-program";
        divergentBlazeProgram->diffractionPaths[0].material.control.y = static_cast<float>(
            diffractionmaterial::GrooveProfile::BlazedSawtooth);
        divergentBlazeProgram->diffractionPaths[0].material.grooveField = {
            static_cast<float>(diffractionmaterial::GrooveFieldMode::Linear),
            0.5f, 0.5f, 0.0f
        };
        divergentBlazeProgram->diffractionPaths[0].material.grooveVariation
            = { 1.0f, 0.0f, 100.0f, 0.0f };
        const auto divergentBlazePreparation = backend.prepare(
            scene, divergentBlazeProgram);
        program->diffractionPaths[0].material.roughness.y = 0.0f;
        if (!preparation.prepared)
            std::cerr << "Metal diffraction preparation error: "
                      << preparation.error << '\n';
        auto frame = backend.render(scene, preparation.resources, 96, 96, {});
        if (!frame.rendered)
            std::cerr << "Metal diffraction render error: " << frame.error << '\n';
        const auto pixels = readBgra8(frame.frame);
        auto repeated = backend.render(scene, preparation.resources, 96, 96, {});
        const auto repeatedPixels = readBgra8(repeated.frame);

        auto transportProgram
            = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(*program);

        transportProgram->diffractionPathCount = 3;
        transportProgram->diffractionMaximumBounceDepth = 1;
        diffractionmaterial::LightingDescription transportLighting;
        transportLighting.pathCount = 3;
        const std::array<diffractionmaterial::LightingPathKind, 3> transportKinds {
            diffractionmaterial::LightingPathKind::Direct,
            diffractionmaterial::LightingPathKind::Environment,
            diffractionmaterial::LightingPathKind::Indirect };
        const std::array<std::array<float, 3>, 3> transportDirections {{
            {{ 0.0f, 0.0f, 1.0f }}, {{ 0.36f, 0.0f, 0.9329523f }},
            {{ -0.28f, 0.28f, 0.918259f }} }};
        std::string transportError;
        const auto admittedTransport = diffractionmaterial::admit(description, transportError);
        for (std::size_t pathIndex = 0; pathIndex < transportKinds.size(); ++pathIndex)
        {
            diffractionmaterial::SpectralLightingPath path;
            path.kind = transportKinds[pathIndex];
            path.bounceDepth = path.kind == diffractionmaterial::LightingPathKind::Indirect ? 1 : 0;
            path.incident.direction = transportDirections[pathIndex];
            path.incident.radiance.fill(0.01f);
            transportLighting.paths[pathIndex] = path;
            if (admittedTransport)
                transportProgram->diffractionPaths[pathIndex]
                    = diffractionmaterial::makeGpuLightingPath(
                        diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                            *admittedTransport, path.incident), path);
        }
        const auto admittedTransportLighting
            = diffractionmaterial::AdmittedLightingPlan::admit(
                transportLighting, transportError);
        if (admittedTransportLighting)
        {
            transportProgram->diffractionLightingAdmission
                = std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(
                    *admittedTransportLighting);
            transportProgram->programIdentity
                = arbitgpu::nativeFixtureDiffractionProgramIdentity(
                    transportProgram->bindingDigest, *admittedTransportLighting);
        }
        const auto transportPreparation = backend.prepare(scene, transportProgram);
        const auto transportFrame = backend.render(
            scene, transportPreparation.resources, 96, 96, {});
        const bool linearDiffraction = linearscenechecks::diffraction(backend, scene,
            transportPreparation.resources, linearError);
        if (!expect(linearDiffraction, linearError.c_str())) return EXIT_FAILURE;
        const bool everyPathSubmitted = transportPreparation.prepared
            && transportFrame.rendered
            && transportFrame.stats.drawCount == transportKinds.size();

        auto zeroLighting = transportLighting;
        for (std::size_t pathIndex = 0; pathIndex < zeroLighting.pathCount; ++pathIndex)
            zeroLighting.paths[pathIndex].incident.radiance.fill(0.0f);
        diffractionmaterial::LightingDescription onePathLighting;
        onePathLighting.pathCount = 1;
        onePathLighting.paths[0] = zeroLighting.paths[0];
        const auto admittedZeroLighting = diffractionmaterial::AdmittedLightingPlan::admit(
            zeroLighting, transportError);
        const auto admittedOnePath = diffractionmaterial::AdmittedLightingPlan::admit(
            onePathLighting, transportError);
        auto zeroProgram = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(
            *transportProgram);
        if (admittedTransport && admittedZeroLighting)
        {
            zeroProgram->diffractionLightingAdmission
                = std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(
                    *admittedZeroLighting);
            for (std::size_t pathIndex = 0; pathIndex < zeroLighting.pathCount; ++pathIndex)
                zeroProgram->diffractionPaths[pathIndex]
                    = diffractionmaterial::makeGpuLightingPath(
                        diffractionmaterial::physicalcheckpoint::makeGpuParameters(
                            *admittedTransport, zeroLighting.paths[pathIndex].incident),
                        zeroLighting.paths[pathIndex]);
            zeroProgram->programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity(
                zeroProgram->bindingDigest, *admittedZeroLighting);
        }
        auto onePathProgram = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(
            *zeroProgram);
        if (admittedOnePath)
        {
            onePathProgram->diffractionPathCount = 1;
            onePathProgram->diffractionMaximumBounceDepth = 0;
            onePathProgram->diffractionLightingAdmission
                = std::make_shared<const diffractionmaterial::AdmittedLightingPlan>(
                    *admittedOnePath);
            onePathProgram->programIdentity = arbitgpu::nativeFixtureDiffractionProgramIdentity(
                onePathProgram->bindingDigest, *admittedOnePath);
        }
        auto noteScore = std::make_shared<arbitmod::Score>();
        noteScore->notationVersion = noteScore->scoreRevision = 1;
        noteScore->rootFreq = 440.0f;
        arbitmod::Note note;
        note.id = -1;
        note.trackId = 1;
        note.startBeat = 0.0f;
        note.lengthBeats = 4.0f;
        note.durationSeconds = 2.0f;
        note.velocity = 100.0f;
        note.freqHz = 440.0f;
        noteScore->notes.push_back(note);
        canonicalblockc::FrameKey noteKey;
        noteKey.projectGeneration = noteKey.sourceGeneration = noteKey.helperGeneration = 1;
        noteKey.backendGeneration = noteKey.deviceGeneration = noteKey.scoreGeneration = 1;
        noteKey.beatMapGeneration = noteKey.fpsGeneration = noteKey.loopGeneration = 1;
        noteKey.seekGeneration = 1;
        noteKey.fps = 60.0;
        canonicalblockc::FrameProducer noteProducer;
        arbitgpu::NativeFixtureSceneRuntimeInputs noteInputs;
        noteInputs.canonicalBlockCFrame = noteProducer.evaluate(noteKey, noteScore, 0.0f);
        visualnoteinstancing::Mapping noteMapping;
        noteMapping.xScale = noteMapping.yScale = noteMapping.zScale = 0.0f;
        noteMapping.meshScale = 1.0f;
        noteMapping.appearanceLow = {0.6f, 0.6f, 0.6f, 0.2f};
        noteMapping.appearanceHigh = noteMapping.appearanceLow;
        noteInputs.noteInstanceMapping = noteMapping;
        const auto zeroPreparation = backend.prepare(scene, zeroProgram);
        const auto onePathPreparation = backend.prepare(scene, onePathProgram);
        const auto zeroFrame = backend.render(scene, zeroPreparation.resources, 96, 96, noteInputs);
        const auto onePathFrame = backend.render(scene, onePathPreparation.resources, 96, 96, noteInputs);
        const auto zeroPixels = readBgra8(zeroFrame.frame);
        const auto onePathPixels = readBgra8(onePathFrame.frame);
        const auto centerPixel = (48u * 96u + 48u) * 4u;
        const bool pathBaseline = expect(admittedZeroLighting && admittedOnePath
            && noteInputs.canonicalBlockCFrame && zeroPreparation.prepared
            && onePathPreparation.prepared && zeroFrame.rendered && onePathFrame.rendered
            && zeroFrame.stats.drawCount == 3 && onePathFrame.stats.drawCount == 1
            && zeroPixels.size() == 96u * 96u * 4u && onePathPixels.size() == zeroPixels.size()
            && onePathPixels[centerPixel] > 20
            && std::equal(zeroPixels.begin(), zeroPixels.end(), onePathPixels.begin(),
                [](std::uint8_t a, std::uint8_t b)
                { return std::abs(static_cast<int>(a) - static_cast<int>(b)) <= 1; }),
            "Metal three-path Color retains one unscattered note baseline");

        arbitgpu::NativeFixtureSceneRuntimeInputs fogInputs;
        fogInputs.passComposite = renderpasscomposite::Parameters {};
        fogInputs.passComposite->mode = renderpasscomposite::Mode::DepthFog;
        fogInputs.passComposite->farDepth = 0.001f;
        fogInputs.passComposite->fogColor = {0.2f, 0.4f, 0.6f};
        const auto fogFrame = backend.render(scene, transportPreparation.resources, 96, 96, fogInputs);
        const auto fogPixels = readBgra8(fogFrame.frame);
        const auto transportPixels = readBgra8(transportFrame.frame);
        const bool resolvedFog = expect(fogFrame.rendered && fogPixels.size() == 96u * 96u * 4u
            && transportPixels.size() == fogPixels.size()
            && std::abs(static_cast<int>(fogPixels[centerPixel]) - 153) <= 1
            && std::abs(static_cast<int>(fogPixels[centerPixel + 1]) - 102) <= 1
            && std::abs(static_cast<int>(fogPixels[centerPixel + 2]) - 51) <= 1
            && fogPixels[centerPixel] != transportPixels[centerPixel],
            "Metal composites diffraction after all physical lighting paths");
        const bool retainedRawPasses = expect(fogFrame.frame
            && readBgra8(fogFrame.frame, renderpassoutput::Output::Depth)
                == readBgra8(transportFrame.frame, renderpassoutput::Output::Depth)
            && readBgra8(fogFrame.frame, renderpassoutput::Output::MaterialId)
                == readBgra8(transportFrame.frame, renderpassoutput::Output::MaterialId)
            && readBgra8(fogFrame.frame, renderpassoutput::Output::ObjectId)
                == readBgra8(transportFrame.frame, renderpassoutput::Output::ObjectId),
            "Metal diffraction resolve retains raw depth and exact IDs");
        bool typedOutputs = true;
        for (const auto output : {renderpassoutput::Output::Depth,
                                  renderpassoutput::Output::Normal,
                                  renderpassoutput::Output::Motion,
                                  renderpassoutput::Output::Emission,
                                  renderpassoutput::Output::Mask,
                                  renderpassoutput::Output::MaterialId,
                                  renderpassoutput::Output::ObjectId})
        {
            arbitgpu::NativeFixtureSceneRuntimeInputs typedInputs;
            typedInputs.imageOutput = output;
            const auto typedFrame = backend.render(
                scene, transportPreparation.resources, 96, 96, typedInputs);
            const auto typedPixels = readBgra8(typedFrame.frame);
            typedOutputs &= typedFrame.rendered && typedPixels.size() == 96u * 96u * 4u
                && typedFrame.frame->passTextureDescriptor(output).complete();
            if (output == renderpassoutput::Output::Depth && typedPixels.size() == 96u * 96u * 4u)
                typedOutputs &= typedPixels[centerPixel] > 0 && typedPixels[0] == 0;
            if (output == renderpassoutput::Output::Mask && typedPixels.size() == 96u * 96u * 4u)
                typedOutputs &= typedPixels[centerPixel] == 255 && typedPixels[0] == 0;
        }
        typedOutputs = expect(typedOutputs,
            "Metal diffraction publishes every typed Render 3D inspection output");
        arbitgpu::NativeFixtureSceneRuntimeInputs branchInputs;
        branchInputs.passProgram = renderpasscomposite::Program {};
        branchInputs.passProgram->count = 3;
        branchInputs.passProgram->output = 2;
        branchInputs.passProgram->steps[0].parameters.mode = renderpasscomposite::Mode::MaskedColor;
        branchInputs.passProgram->steps[1].parameters = *fogInputs.passComposite;
        branchInputs.passProgram->steps[2].parameters.mode = renderpasscomposite::Mode::Mix;
        branchInputs.passProgram->steps[2].parameters.amount = 0.5f;
        branchInputs.passProgram->steps[2].inputA = 0;
        branchInputs.passProgram->steps[2].inputB = 1;
        branchInputs.passComposite = branchInputs.passProgram->steps[2].parameters;
        const auto branchFrame = backend.render(
            scene, transportPreparation.resources, 96, 96, branchInputs);
        const auto branchPixels = readBgra8(branchFrame.frame);
        bool branchResolved = branchFrame.rendered
            && branchPixels.size() == 96u * 96u * 4u
            && transportPixels.size() == branchPixels.size()
            && fogPixels.size() == branchPixels.size();
        if (branchResolved)
            for (std::size_t channel = 0; channel < 3; ++channel)
                branchResolved &= std::abs(static_cast<int>(branchPixels[centerPixel + channel])
                    - (static_cast<int>(transportPixels[centerPixel + channel])
                       + static_cast<int>(fogPixels[centerPixel + channel])) / 2) <= 2;
        branchResolved = expect(branchResolved,
            "Metal diffraction branches consume the resolved physical Color once");
        arbitgpu::NativeFixtureSceneRuntimeInputs inspectionInputs;
        inspectionInputs.passComposite = renderpasscomposite::Parameters {};
        inspectionInputs.passComposite->mode = renderpasscomposite::Mode::DepthView;
        inspectionInputs.passComposite->inspectionImage = true;
        const auto inspectionFrame = backend.render(
            scene, transportPreparation.resources, 96, 96, inspectionInputs);
        const auto inspectionPixels = readBgra8(inspectionFrame.frame);
        const bool inspectionClear = expect(inspectionFrame.rendered
            && inspectionPixels.size() == 96u * 96u * 4u
            && inspectionPixels[0] == 255 && inspectionPixels[centerPixel] < 255,
            "Metal diffraction inspection preserves the uncovered depth clear");
        fogInputs.passComposite->mode = renderpasscomposite::Mode::Count;
        const auto invalidComposite = backend.render(
            scene, transportPreparation.resources, 96, 96, fogInputs);
        const bool invalidCompositeRejected = expect(!invalidComposite.rendered
            && invalidComposite.frame == nullptr,
            "Metal rejects malformed diffraction compositing before frame allocation");

        arbitgpu::NativeFixtureSceneRuntimeInputs shiftedCamera;
        shiftedCamera.cameraTranslationOffset[0] = 0.75f;
        auto shifted = backend.render(
            scene, preparation.resources, 96, 96, shiftedCamera);
        const auto shiftedPixels = readBgra8(shifted.frame);

        auto rotatedDescription = description;
        rotatedDescription.geometry.directionUv = { 0.0f, 1.0f };
        auto rotatedProgram = makeProgram(
            rotatedDescription, scene->objects[0].id,
            "metal-fixture-diffraction-rotated-program");
        auto rotatedPreparation = backend.prepare(scene, rotatedProgram);
        auto rotated = backend.render(
            scene, rotatedPreparation.resources, 96, 96, {});
        const auto rotatedPixels = readBgra8(rotated.frame);

        auto spatialDescription = description;
        spatialDescription.grooveField.mode
            = diffractionmaterial::GrooveFieldMode::Linear;
        spatialDescription.grooveField.originUv = { 0.5f, 0.5f };
        spatialDescription.grooveField.axisUv = { 1.0f, 0.0f };
        spatialDescription.grooveField.grooveSpacingDeltaNanometresPerUnit = 500.0f;
        spatialDescription.grooveField.orientationDegreesPerUnit = 90.0f;
        auto spatialProgram = makeProgram(
            spatialDescription, scene->objects[0].id,
            "metal-fixture-diffraction-spatial-program");
        auto spatialPreparation = backend.prepare(scene, spatialProgram);
        auto spatial = backend.render(
            scene, spatialPreparation.resources, 96, 96, {});
        const auto spatialPixels = readBgra8(spatial.frame);

        auto crossedDescription
            = diffractionmaterial::makeRealtimeCrossedTwoDimensionalGratingPreset();
        auto crossedProgram = makeProgram(
            crossedDescription, scene->objects[0].id,
            "metal-fixture-diffraction-crossed-program");
        auto unequalCrossedBlazedProgram
            = std::make_shared<arbitgpu::NativeFixtureSurfaceMaterialProgram>(
                *crossedProgram);
        unequalCrossedBlazedProgram->programIdentity
            = "metal-fixture-unequal-crossed-blazed-program";
        unequalCrossedBlazedProgram->diffractionPaths[0].material.control.y
            = static_cast<float>(diffractionmaterial::GrooveProfile::BlazedSawtooth);
        unequalCrossedBlazedProgram->diffractionPaths[0].material.secondaryGeometry.z
            = unequalCrossedBlazedProgram->diffractionPaths[0].material.geometry.z + 1.0f;
        const auto unequalCrossedBlazedPreparation = backend.prepare(
            scene, unequalCrossedBlazedProgram);
        auto crossedPreparation = backend.prepare(scene, crossedProgram);
        auto crossed = backend.render(
            scene, crossedPreparation.resources, 96, 96, {});
        const auto crossedPixels = readBgra8(crossed.frame);

        auto crossedDirectionDescription = crossedDescription;
        crossedDirectionDescription.geometry.directionUv
            = { 0.70710678f, 0.70710678f };
        crossedDirectionDescription.geometry.secondaryDirectionUv
            = { -0.70710678f, 0.70710678f };
        auto crossedDirectionProgram = makeProgram(
            crossedDirectionDescription,
            scene->objects[0].id,
            "metal-fixture-diffraction-crossed-direction-program");
        auto crossedDirectionPreparation = backend.prepare(
            scene, crossedDirectionProgram);
        auto crossedDirection = backend.render(
            scene, crossedDirectionPreparation.resources, 96, 96, {});
        const auto crossedDirectionPixels = readBgra8(crossedDirection.frame);

        auto primaryPeriodDescription = crossedDescription;
        primaryPeriodDescription.geometry.grooveSpacingNanometres = 1350.0f;
        auto primaryPeriodProgram = makeProgram(
            primaryPeriodDescription,
            scene->objects[0].id,
            "metal-fixture-diffraction-primary-period-program");
        auto primaryPeriodPreparation = backend.prepare(scene, primaryPeriodProgram);
        auto primaryPeriod = backend.render(
            scene, primaryPeriodPreparation.resources, 96, 96, {});
        const auto primaryPeriodPixels = readBgra8(primaryPeriod.frame);

        auto crossedPeriodDescription = crossedDescription;
        crossedPeriodDescription.geometry.secondaryGrooveSpacingNanometres = 900.0f;
        auto crossedPeriodProgram = makeProgram(
            crossedPeriodDescription,
            scene->objects[0].id,
            "metal-fixture-diffraction-crossed-period-program");
        auto crossedPeriodPreparation = backend.prepare(scene, crossedPeriodProgram);
        auto crossedPeriod = backend.render(
            scene, crossedPeriodPreparation.resources, 96, 96, {});
        const auto crossedPeriodPixels = readBgra8(crossedPeriod.frame);
        const auto overBudgetCrossed = backend.render(
            scene, crossedPreparation.resources, 3840, 2160, {});

        const auto reportDraw = [](const char* name, const auto& prepared,
                                   const auto& rendered)
        {
            if (!prepared.prepared || !rendered.rendered)
                std::cerr << "Metal diffraction draw " << name
                          << " preparation=" << prepared.prepared
                          << " preparationError=" << prepared.error
                          << " rendered=" << rendered.rendered
                          << " renderError=" << rendered.error << '\n';
        };
        reportDraw("base", preparation, frame);
        reportDraw("transport", transportPreparation, transportFrame);
        reportDraw("rotated", rotatedPreparation, rotated);
        reportDraw("spatial", spatialPreparation, spatial);
        reportDraw("crossed", crossedPreparation, crossed);
        reportDraw("crossed-direction", crossedDirectionPreparation, crossedDirection);
        reportDraw("primary-period", primaryPeriodPreparation, primaryPeriod);
        reportDraw("crossed-period", crossedPeriodPreparation, crossedPeriod);

        ok &= starterOk && rasterControlOk && pathBaseline && resolvedFog && retainedRawPasses
            && typedOutputs && branchResolved && inspectionClear && invalidCompositeRejected
            && expect(preparation.prepared && frame.rendered && repeated.rendered
                                   && everyPathSubmitted
                                   && shifted.rendered && rotatedPreparation.prepared
                                   && rotated.rendered && spatialPreparation.prepared
                                   && spatial.rendered && crossedPreparation.prepared
                                   && crossed.rendered
                                   && crossedDirectionPreparation.prepared
                                   && crossedDirection.rendered
                                   && primaryPeriodPreparation.prepared
                                   && primaryPeriod.rendered
                                   && crossedPeriodPreparation.prepared
                                   && crossedPeriod.rendered,
                               "the strict Metal fixture path must submit every diffraction draw")
            && expect(pixels.size() == 96u * 96u * 4u && !backgroundOnly(pixels),
                      "the Metal diffraction draw must write physical surface pixels")
            && expect(repeatedPixels == pixels
                          && frame.frame->nativeResourceCacheIdentity()
                              == repeated.frame->nativeResourceCacheIdentity()
                          && frame.frame->colorTextureDescriptor().rendererGeneration
                              == repeated.frame->colorTextureDescriptor().rendererGeneration,
                      "repeated Metal draws must retain exact cache and renderer-generation identity")
            && expect(pixels != shiftedPixels,
                      "Metal diffraction pixels must change with camera position")
            && expect(pixels != rotatedPixels,
                      "Metal diffraction pixels must change with reciprocal-lattice direction")
            && expect(spatialPixels != pixels && spatialPixels != rotatedPixels,
                      "Metal diffraction pixels must evaluate local UV groove fields")
            && expect(!backgroundOnly(crossedPixels) && crossedPixels != pixels,
                      "Metal diffraction pixels must execute crossed-lattice topology")
            && expect(crossedDirectionPixels != crossedPixels
                          && primaryPeriodPixels != crossedPixels
                          && crossedPeriodPixels != crossedPixels,
                      "Metal crossed diffraction must independently respond to lattice orientation and both periods")
            && expect(!overBudgetCrossed.rendered
                          && overBudgetCrossed.frame == nullptr
                          && overBudgetCrossed.error
                               == "Metal fixture diffraction workload exceeds backend limits",
                      "Metal rejects over-budget crossed diffraction before frame allocation");
        const bool malformedRejected = expect(
            !malformedPreparation.prepared && malformedPreparation.resources == nullptr
                && malformedPreparation.stats.vertexBytes == 0
                && malformedPreparation.stats.indexBytes == 0
                && malformedPreparation.stats.materialBytes == 0
                && malformedPreparation.stats.textureBytes == 0
                && malformedPreparation.error
                    == "Metal fixture preparation requires an exact bounded material binding",
            "Metal rejects malformed diffraction GPU parameters before resource allocation")
            && expect(!replayedLightingPreparation.prepared
                && replayedLightingPreparation.resources == nullptr
                && replayedLightingPreparation.stats.vertexBytes == 0
                && replayedLightingPreparation.stats.indexBytes == 0
                && replayedLightingPreparation.stats.materialBytes == 0
                && replayedLightingPreparation.stats.textureBytes == 0
                && replayedLightingPreparation.error
                    == "Metal fixture preparation requires an exact bounded material binding",
            "Metal rejects a replayed spectral receipt before resource allocation")
            && expect(!layoutSentinelPreparation.prepared
                && layoutSentinelPreparation.resources == nullptr
                && layoutSentinelPreparation.stats.vertexBytes == 0
                && layoutSentinelPreparation.stats.indexBytes == 0
                && layoutSentinelPreparation.stats.materialBytes == 0
                && layoutSentinelPreparation.stats.textureBytes == 0
                && layoutSentinelPreparation.error
                    == "Metal fixture preparation requires an exact bounded material binding",
            "Metal rejects reused layout-sentinel data before GPU allocation")
            && expect(!divergentBlazePreparation.prepared
                && divergentBlazePreparation.resources == nullptr
                && divergentBlazePreparation.stats.vertexBytes == 0
                && divergentBlazePreparation.stats.indexBytes == 0
                && divergentBlazePreparation.stats.materialBytes == 0
                && divergentBlazePreparation.stats.textureBytes == 0
                && divergentBlazePreparation.error
                    == "Metal fixture preparation requires an exact bounded material binding",
            "Metal rejects local period/blaze divergence before GPU allocation")
            && expect(!unequalCrossedBlazedPreparation.prepared
                && unequalCrossedBlazedPreparation.resources == nullptr
                && unequalCrossedBlazedPreparation.error
                    == "Metal fixture preparation requires an exact bounded material binding",
            "Metal preflight rejects unequal crossed blazed periods before GPU allocation")
            && expect(frame.rendered,
                      "Metal preparation owns an immutable material snapshot after caller alias mutation");
        if (!malformedRejected)
            return 1;
        if (!ok)
            return 1;
        std::cout << "Metal fixture diffraction backend: PASS; device="
                  << info.device << "; raster-control-pixels="
                  << nonBackgroundPixelCount(rasterPixels)
                  << "; front-facing-control-pixels="
                  << nonBackgroundPixelCount(frontFacingRasterPixels)
                  << "; culled-control-pixels="
                  << nonBackgroundPixelCount(backFacingRasterPixels)
                  << "; diffraction-pixels=" << nonBackgroundPixelCount(pixels)
                  << '\n';
        return 0;
    }
#endif
}
