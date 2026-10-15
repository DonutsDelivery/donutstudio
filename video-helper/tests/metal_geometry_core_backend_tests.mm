#include "../src/geometry_core_backend.h"
#include "../../shared/HdrImageOutputContract.h"
#include "../src/canonical_block_c_frame.h"
#include "note_instance_appearance_fixture.h"
#include "spectrum_instancer_fixture.h"
#include "spectrum_field_domains_fixture.h"
#include "spectrum_instancer_layouts_fixture.h"
#include "spectral_history_fixture.h"
#include "point_distribution_fixture.h"
#include "particle_points_fixture.h"
#include "constructed_field_fixture.h"
#include "imported_geometry_fixture.h"
#include "instance_appearance_fixture.h"
#include "instance_material_fields_fixture.h"
#include "score_instance_field_fixture.h"
#include "reactive_surface_fixture.h"
#include "reactive_frame_fixture.h"
#include "mixed_surface_diffraction_fixture.h"
#include "material_table_fixture.h"
#include "audio_deformer_fixture.h"
#include "ordered_audio_geometry_fixture.h"
#include "score_field_fixture.h"
#include "timeline_curve_fixture.h"
#include "geometry_scene_composition_fixture.h"
#include "composed_timeline_geometry_fixture.h"
#include "native_geometry_admission_fixture.h"
#include "solid_body_fixture.h"
#include "support/scene_closest_hit_metal.h"
#include "support/diffraction_metal_transport_fixture.h"
#include "support/environment_secondary_cases.h"
#include "sokol_gfx.h"
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <set>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {
std::vector<std::uint8_t> readBgra8(
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> &frame) {
  sg_image image{};
  image.id = static_cast<std::uint32_t>(frame->colorImageHandle());
  const auto info = sg_mtl_query_image_info(image);
  if (info.active_slot < 0 || info.active_slot >= SG_NUM_INFLIGHT_FRAMES) return {};
  id<MTLTexture> texture = (__bridge id<MTLTexture>)info.tex[info.active_slot];
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)sg_mtl_command_queue();
  if (texture == nil || queue == nil || texture.pixelFormat != MTLPixelFormatBGRA8Unorm)
    return {};
  const std::size_t tight = frame->width() * 4u;
  const std::size_t row = (tight + 255u) & ~std::size_t(255u);
  id<MTLBuffer> buffer = [texture.device newBufferWithLength:row * frame->height()
                                                   options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
  if (buffer == nil || command == nil || blit == nil) return {};
  [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(frame->width(), frame->height(), 1)
                 toBuffer:buffer destinationOffset:0 destinationBytesPerRow:row
 destinationBytesPerImage:row * frame->height()];
  [blit endEncoding]; [command commit]; [command waitUntilCompleted];
  std::vector<std::uint8_t> pixels(tight * frame->height());
  const auto *source = static_cast<const std::uint8_t *>(buffer.contents);
  if (source) for (std::uint32_t y = 0; y < frame->height(); ++y)
    std::copy_n(source + y * row, tight, pixels.data() + y * tight);
#if !__has_feature(objc_arc)
  [buffer release];
#endif
  return pixels;
}

// Raw Motion vectors in pixels, two half floats per pixel in row order.
std::vector<float> readMotion(
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> &frame) {
  if (!frame) return {};
  const auto descriptor = frame->passTextureDescriptor(renderpassoutput::Output::Motion);
  if (!descriptor.complete() || descriptor.format != arbitgpu::NativeTexturePixelFormat::Rg16Float)
    return {};
  sg_image image{};
  image.id = static_cast<std::uint32_t>(descriptor.imageHandle);
  const auto info = sg_mtl_query_image_info(image);
  if (info.active_slot < 0 || info.active_slot >= SG_NUM_INFLIGHT_FRAMES) return {};
  id<MTLTexture> texture = (__bridge id<MTLTexture>)info.tex[info.active_slot];
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)sg_mtl_command_queue();
  if (texture == nil || queue == nil || texture.pixelFormat != MTLPixelFormatRG16Float)
    return {};
  const std::size_t tight = frame->width() * 4u;
  const std::size_t row = (tight + 255u) & ~std::size_t(255u);
  id<MTLBuffer> buffer = [texture.device newBufferWithLength:row * frame->height()
                                                   options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
  if (buffer == nil || command == nil || blit == nil) {
#if !__has_feature(objc_arc)
    [buffer release];
#endif
    return {};
  }
  [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(frame->width(), frame->height(), 1)
                 toBuffer:buffer destinationOffset:0 destinationBytesPerRow:row
 destinationBytesPerImage:row * frame->height()];
  [blit endEncoding]; [command commit]; [command waitUntilCompleted];
  std::vector<float> values(std::size_t(frame->width()) * frame->height() * 2u);
  const auto *source = static_cast<const std::uint8_t *>(buffer.contents);
  if (source) for (std::uint32_t y = 0; y < frame->height(); ++y)
    for (std::uint32_t x = 0; x < frame->width() * 2u; ++x) {
      std::uint16_t half = 0;
      std::memcpy(&half, source + y * row + x * 2u, sizeof(half));
      values[std::size_t(y) * frame->width() * 2u + x] = hdrimage::halfToFloat(half);
    }
#if !__has_feature(objc_arc)
  [buffer release];
#endif
  return source ? values : std::vector<float>{};
}

bool fail(const std::string &message) {
  std::cerr << "FAIL: " << message << '\n';
  return false;
}

arbitgpu::NativeFixtureSceneRuntimeInputs noteRuntimeInputs(
    bool includeSecondNote = true, float secondOnset = 1.0f,
    bool duplicateFullTransform = false) {
  auto score = std::make_shared<arbitmod::Score>();
  score->notationVersion = 1; score->scoreRevision = 1; score->rootFreq = 440.0f;
  const int noteCount = includeSecondNote ? 2 : 1;
  for (int index = 0; index < noteCount; ++index) {
    arbitmod::Note note;
    note.id = index + 1; note.trackId = 1;
    note.startBeat = index == 0 ? 0.0f : secondOnset; note.lengthBeats = 4.0f;
    note.velocity = 100.0f;
    note.freqHz = index == 0 || duplicateFullTransform ? 440.0f : 660.0f;
    note.durationSeconds = 2.0f; score->notes.push_back(note);
  }
  canonicalblockc::FrameKey key;
  key.projectGeneration = key.sourceGeneration = key.helperGeneration = 1;
  key.backendGeneration = key.deviceGeneration = key.scoreGeneration = 1;
  key.beatMapGeneration = key.fpsGeneration = key.loopGeneration = 1;
  key.seekGeneration = 1; key.fps = 60.0;
  canonicalblockc::FrameProducer producer;
  arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
  inputs.canonicalBlockCFrame = producer.evaluate(key, score, 0.0f);
  visualnoteinstancing::Mapping mapping;
  mapping.x = visualnoteinstancing::MappingAxis::Onset;
  mapping.xScale = 0.2f; mapping.meshScale = 0.8f;
  inputs.noteInstanceMapping = mapping;
  return inputs;
}

std::shared_ptr<HarmonicMIDI::grid::Visual3DScene> ordinarySharedScene() {
  using namespace HarmonicMIDI::grid;
  auto scene = std::make_shared<Visual3DScene>();
  scene->id = {91};
  scene->vertices[0].position = {-0.25f,-0.25f,0.0f};
  scene->vertices[1].position = {0.25f,-0.25f,0.0f};
  scene->vertices[2].position = {0.0f,0.25f,0.0f};
  for (std::size_t index = 0; index < 3; ++index) {
    scene->vertices[index].normal = {0,0,1};
    scene->vertices[index].color = {1,1,1,1}; scene->indices[index] = index;
  }
  scene->vertexCount = scene->indexCount = 3;
  scene->materials[0].id = {1}; scene->materials[0].baseColor = {0.8f,0.4f,0.2f};
  scene->materialCount = 1;
  for (std::size_t index = 0; index < 2; ++index) {
    scene->objects[index].id = {static_cast<std::uint32_t>(index + 1)};
    scene->objects[index].material = {1}; scene->objects[index].vertexCount = 3;
    scene->objects[index].indexCount = 3;
    scene->objects[index].transform.translation.x = index == 0 ? -0.5f : 0.5f;
  }
  scene->objectCount = 2; scene->cameras[0].id = {1};
  scene->cameras[0].transform.translation = {0,0,5};
  scene->cameras[0].verticalFovRadians = 0.7853981634f;
  scene->cameras[0].nearPlane = 0.1f; scene->cameras[0].farPlane = 1000.0f;
  scene->cameraCount = 1; scene->activeCamera = {1}; return scene;
}

// These fixtures render retained SDF geometry through the product backend.
// Their raw attachments are data; the optional identity colors are diagnostics.
template <typename ReadPixels>
bool verifyNativeRetainedSdfSurface(arbitgpu::NativeFixtureSceneBackend& backend,
    ReadPixels readPixels, std::string& error) {
  using namespace videowire;
  using namespace videowire::geometry;
  using namespace videohelper::geometry;
  using Format = arbitgpu::NativeTexturePixelFormat;
  using Output = renderpassoutput::Output;
  constexpr StableId surfaceId = 9870, sourceId = 9871, primitiveId = 9007199254740001ull;
  constexpr unsigned extent = 96;
  const auto reject = [&](const char* message) { error = message; return false; };
  PortContract meshContract;
  meshContract.carrier = CarrierKind::geometry3D;
  meshContract.maxVertices = HarmonicMIDI::grid::Visual3DScene::kMaxVertices;
  meshContract.maxIndices = HarmonicMIDI::grid::Visual3DScene::kMaxIndices;
  meshContract.maxAttributes = 3; meshContract.attributes = sdfSurfaceAttributes(surfaceId);
  const auto makeSurface = [&](unsigned cells, double radius) {
    sdfsurface::Request request;
    request.cells = {cells,cells,cells}; request.maximumResidual = 0.11;
    SdfRecord sphere;
    sphere.stableId = primitiveId; sphere.operation = SdfOperation::sphere;
    sphere.parameterCount = 1; sphere.parameters[0] = radius;
    request.geometry.rootId = primitiveId; request.geometry.records = {sphere};
    return lowerSdfSurface(surfaceId,sourceId,1,request,meshContract,error);
  };
  const auto coarse = makeSurface(8,1.013), fine = makeSurface(16,1.013);
  const auto smaller = makeSurface(16,0.713);
  if (!coarse || !fine || !smaller) return false;
  const auto radialError = [](const Vec3& p, double radius) {
    return std::abs(std::sqrt(double(p.x)*p.x+double(p.y)*p.y+double(p.z)*p.z)-radius);
  };
  const auto checkMesh = [&](const ValueDescriptor& value, double radius, double& residual) {
    const auto& mesh = std::get<GeometryData>(value.data);
    residual = 0;
    for (std::size_t i=0;i<mesh.positions.size();++i) {
      const auto& p = mesh.positions[i];
      residual = std::max(residual,radialError(p,radius));
      const auto& n = value.attributes[0].elements[i].components;
      const double length = std::sqrt(double(p.x)*p.x+double(p.y)*p.y+double(p.z)*p.z);
      if (std::abs(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]-1)>1e-6
          || (n[0]*p.x+n[1]*p.y+n[2]*p.z)/length<0.99999)
        return reject("retained SDF normal disagrees with the analytic sphere gradient");
    }
    for (std::size_t i=0;i<mesh.indices.size();i+=3) {
      const auto& a=mesh.positions[mesh.indices[i]];
      const auto& b=mesh.positions[mesh.indices[i+1]];
      const auto& c=mesh.positions[mesh.indices[i+2]];
      const double ux=b.x-a.x, uy=b.y-a.y, uz=b.z-a.z;
      const double vx=c.x-a.x, vy=c.y-a.y, vz=c.z-a.z;
      if ((uy*vz-uz*vy)*(a.x+b.x+c.x)+(uz*vx-ux*vz)*(a.y+b.y+c.y)
          +(ux*vy-uy*vx)*(a.z+b.z+c.z)<=0)
        return reject("retained SDF triangles are degenerate or wind inward");
      residual = std::max(residual,radialError({(a.x+b.x+c.x)/3,
          (a.y+b.y+c.y)/3,(a.z+b.z+c.z)/3},radius));
      if (value.attributes[2].elements[i/3].components[0]!=double(primitiveId))
        return reject("retained SDF provenance lost its full primitive integer");
    }
    return residual<0.11;
  };
  double coarseResidual=0, fineResidual=0, smallerResidual=0;
  if (!checkMesh(*coarse,1.013,coarseResidual) || !checkMesh(*fine,1.013,fineResidual)
      || !checkMesh(*smaller,0.713,smallerResidual)) {
    if (error.empty()) error="retained SDF sphere exceeded its independent radial error bound";
    return false;
  }
  if (fineResidual>=coarseResidual*0.65)
    return reject("halving the SDF grid spacing did not converge toward the analytic sphere");
  NativeGeometryCoreCapabilitySource capabilities(backend);
  GeometryCorePlanRuntime runtime(capabilities);
  const auto planFor = [&](const ValueDescriptor& value, const PortContract& contract,
      std::uint64_t revision, bool exportUse=false) -> std::optional<RuntimeAdmission> {
    const auto admitted = admitValue(value,contract,error);
    if (!admitted) return std::nullopt;
    const auto bytes = lowerRuntimePlan(contract,*admitted);
    const PlanOwnerIdentity owner{1,1,1,value.stableId,revision,
        exportUse ? PlanUse::exportRender : PlanUse::preview};
    return exportUse ? runtime.admitExport(owner,bytes,{}, {},error)
                     : runtime.admitPreview(owner,bytes,{}, {},error);
  };
  const auto raw = [&](const NativeGeometryExecution& draw, Output output, Format format,
      arbitgpu::NativeRawPassPixels& result) {
    if (!draw.frame || !draw.frame->readRawPass(output,result,error)) return false;
    if (result.format!=format || result.width!=extent || result.height!=extent
        || result.bytes.size()!=extent*extent*arbitgpu::rawPassChannels(format)
            *arbitgpu::rawPassScalarBytes(format))
      return reject("retained SDF native attachment has the wrong typed layout");
    return true;
  };
  const auto idAt = [](const arbitgpu::NativeRawPassPixels& pixels, std::size_t i) {
    std::uint32_t id=0; std::memcpy(&id,pixels.bytes.data()+i*4,4); return id;
  };
  const auto verifySphereDraw = [&](const NativeGeometryExecution& draw, double radius,
      double meshResidual, double& nativeResidual) {
    arbitgpu::NativeRawPassPixels ids, materials, depths, normals;
    if (!raw(draw,Output::ObjectId,Format::R32Uint,ids)
        || !raw(draw,Output::MaterialId,Format::R32Uint,materials)
        || !raw(draw,Output::Depth,Format::R32Float,depths)
        || !raw(draw,Output::Normal,Format::Rgba16Float,normals)) return false;
    std::size_t interior=0, foreground=0;
    nativeResidual=0;
    for (unsigned y=0;y<extent;++y) for (unsigned x=0;x<extent;++x) {
      const auto i=y*extent+x, id=idAt(ids,i), material=idAt(materials,i);
      if (id==0) {
        if (material!=0) return reject("SDF native background has a nonzero raw material ID");
        continue;
      }
      ++foreground;
      if (id!=1 || material!=1)
        return reject("SDF primitive provenance was substituted for raw scene object/material IDs");
      // The native default camera is at z=5, with a 45-degree vertical FOV.
      float depth=0; std::memcpy(&depth,depths.bytes.data()+i*4,4);
      const double distance=0.1+double(depth)*(1000.0-0.1);
      const double px=(2.0*(x+0.5)/extent-1)*std::tan(0.7853981634*0.5)*distance;
      const double py=(1-2.0*(y+0.5)/extent)*std::tan(0.7853981634*0.5)*distance;
      const double pz=5-distance, length=std::sqrt(px*px+py*py+pz*pz);
      if (px*px+py*py>radius*radius*0.36) continue;
      ++interior;
      // Direct length-radius oracle, independent of the production SDF evaluator.
      nativeResidual=std::max(nativeResidual,std::abs(length-radius));
      std::array<double,3> normal{};
      for (unsigned channel=0;channel<3;++channel) {
        std::uint16_t half=0; std::memcpy(&half,normals.bytes.data()+i*8+channel*2,2);
        normal[channel]=hdrimage::halfToFloat(half);
      }
      if (!std::isfinite(length) || !std::all_of(normal.begin(),normal.end(),
          [](double value) { return std::isfinite(value); })
          || (normal[0]*px+normal[1]*py+normal[2]*pz)/length<0.99)
        return reject("native SDF normals disagree with the analytic outward sphere gradient");
    }
    if (foreground<100 || interior<50 || nativeResidual>meshResidual+0.005
        || draw.stats.ordinaryDrawCount!=1 || draw.stats.instancedDrawCount!=0)
      return reject("retained SDF native pixels disagree with the analytic sphere or ordinary draw route");
    return true;
  };
  std::array<std::vector<std::uint8_t>,3> spherePixels;
  double coarseNativeResidual=0, fineNativeResidual=0;
  for (unsigned variant=0;variant<3;++variant) {
    const auto& value=variant==0 ? *coarse : variant==1 ? *fine : *smaller;
    const auto plan=planFor(value,meshContract,variant+1);
    const auto exported=planFor(value,meshContract,variant+1,true);
    if (!plan || !exported) return false;
    const auto draw=executeNativeGeometry(backend,*plan,extent,extent,error);
    const auto offline=executeNativeGeometry(backend,*exported,extent,extent,error);
    if (!draw || !offline) return false;
    double residual=0;
    if (!verifySphereDraw(*draw,variant==2 ? 0.713 : 1.013,
        variant==0 ? coarseResidual : variant==1 ? fineResidual : smallerResidual,residual)) return false;
    for (const auto output : {Output::Depth,Output::Normal,Output::ObjectId,Output::MaterialId}) {
      arbitgpu::NativeRawPassPixels before, after;
      if (!draw->frame->readRawPass(output,before,error)
          || !offline->frame->readRawPass(output,after,error)) return false;
      if (before.format!=after.format || before.bytes!=after.bytes)
        return reject("retained SDF native preview/export raw geometry or scene identities differ");
    }
    if (variant==0) coarseNativeResidual=residual;
    if (variant==1) fineNativeResidual=residual;
    spherePixels[variant]=readPixels(draw->frame);
    if (spherePixels[variant].size()!=extent*extent*4 || spherePixels[variant]!=readPixels(offline->frame)
        || draw->stats.vertexBytes!=std::get<GeometryData>(value.data).positions.size()
            *sizeof(HarmonicMIDI::grid::SceneVertex)
        || draw->stats.indexBytes!=std::get<GeometryData>(value.data).indices.size()*sizeof(std::uint32_t))
      return reject("retained SDF native upload or preview/export pixels lost the exact mesh");
  }
  if (fineNativeResidual>=coarseNativeResidual*0.8 || spherePixels[1]==spherePixels[2])
    return reject("native SDF pixels did not converge or react to the authored radius edit");
  const auto fineAdmission=admitValue(*fine,meshContract,error);
  if (!fineAdmission) return false;
  const auto wire=encodeRuntimeValue(*fineAdmission);
  const auto reopened=decodeRuntimeValue(wire.data(),wire.size(),meshContract,{}, {},error);
  if (!reopened) return false;
  const auto replayPlan=planFor(reopened->descriptor(),meshContract,4);
  const auto replay=replayPlan ? executeNativeGeometry(backend,*replayPlan,extent,extent,error) : std::nullopt;
  if (!replay || readPixels(replay->frame)!=spherePixels[1]
      || !equal(std::get<GeometryData>(reopened->descriptor().data),std::get<GeometryData>(fine->data))
      || !equalAttributes(reopened->descriptor().attributes,fine->attributes))
    return reject("reopened retained SDF mesh did not replay the same native image and provenance");

  double replayResidual=0;
  if (!verifySphereDraw(*replay,1.013,fineResidual,replayResidual)) return false;

  const auto prototype=lowerMeshGenerator(9872,9872,1,OperationCode::cube,
      {0.3f,0.3f,0.3f,0},constructedFieldMeshContract(),error);
  if (!prototype) return false;
  auto binding=spectrumDomainBinding(); binding.baseScale={1,1,1};
  binding.scaleResponse={0,0,0}; binding.direction={0.12f,0.08f,0};
  const auto contract=appearanceInstanceContract();
  const auto makeInstances = [&](const ValueDescriptor& placement) {
    return lowerSpectrumInstancer(*prototype,9873,binding,{5,2,2,2},contract,error,&placement);
  };
  const auto first=makeInstances(*fine), edited=makeInstances(*smaller);
  if (!first || !edited) return false;
  const auto& a=std::get<InstancesData>(first->data).instances;
  const auto& b=std::get<InstancesData>(edited->data).instances;
  bool moved=false;
  for (std::size_t i=0;i<a.size();++i) {
    const auto& p=a[i].transform.translation; const auto& q=b[i].transform.translation;
    if (a[i].stableId!=i+3 || b[i].stableId!=a[i].stableId
        || radialError(p,1.013)>0.11 || radialError(q,0.713)>0.11)
      return reject("SDF Mesh Surface placements lost analytic surface membership or absolute band IDs");
    moved=moved || std::abs(p.x-q.x)+std::abs(p.y-q.y)+std::abs(p.z-q.z)>0.1;
  }
  if (!moved) return reject("the authored SDF radius edit did not move retained Mesh Surface placements");
  const auto drawInstances = [&](const RuntimeAdmission& plan, bool diagnostic=false) {
    SpectrumEvaluation spectrum; spectrum.bands=spectrumDomainBands(); spectrum.timeSeconds=3.25;
    spectrum.featuresAt=[](double) { return spectrumDomainBands(); };
    return executeNativeGeometry(backend,plan,extent,extent,error,diagnostic,{},&spectrum,{},3.25);
  };
  std::array<std::vector<std::uint8_t>,2> instancePixels;
  for (unsigned variant=0;variant<2;++variant) {
    const auto& value=variant==0 ? *first : *edited;
    const auto admitted=admitValue(value,contract,error);
    if (!admitted) return false;
    const auto bytes=encodeRuntimeValue(*admitted);
    const auto restored=decodeRuntimeValue(bytes.data(),bytes.size(),contract,{}, {},error);
    const auto plan=planFor(value,contract,variant+1);
    const auto exported=planFor(value,contract,variant+1,true);
    const auto restoredPlan=restored ? planFor(restored->descriptor(),contract,variant+3) : std::nullopt;
    if (!plan || !exported || !restoredPlan) return false;
    const auto draw=drawInstances(*plan), offline=drawInstances(*exported);
    const auto again=drawInstances(*restoredPlan), diagnostic=drawInstances(*plan,true);
    if (!draw || !offline || !again || !diagnostic) return false;
    instancePixels[variant]=readPixels(draw->frame);
    if (instancePixels[variant].empty() || instancePixels[variant]!=readPixels(offline->frame)
        || instancePixels[variant]!=readPixels(again->frame)
        || instancePixels[variant]==readPixels(diagnostic->frame)
        || draw->drawnInstanceIds!=std::vector<StableId>{3,4,5,6}
        || diagnostic->drawnInstanceIds!=draw->drawnInstanceIds
        || draw->stats.instancedDrawCount!=1 || draw->stats.ordinaryDrawCount!=0
        || draw->stats.submittedInstanceCount!=4
        || draw->stats.vertexBytes!=std::get<GeometryData>(prototype->data).positions.size()
            *sizeof(HarmonicMIDI::grid::SceneVertex))
      return reject("SDF Mesh Surface native instances lost product pixels, band IDs or retained prototype storage");
    arbitgpu::NativeRawPassPixels ids, diagnosticIds, materials;
    if (!raw(*draw,Output::ObjectId,Format::R32Uint,ids)
        || !raw(*diagnostic,Output::ObjectId,Format::R32Uint,diagnosticIds)
        || !raw(*draw,Output::MaterialId,Format::R32Uint,materials)) return false;
    std::set<std::uint32_t> visible;
    for (std::size_t i=0;i<extent*extent;++i) {
      const auto id=idAt(ids,i);
      if (id==0) continue;
      visible.insert(id);
      if (id<3 || id>6 || idAt(materials,i)!=1)
        return reject("SDF Mesh Surface raw scene IDs were replaced by an RGB diagnostic or primitive ID");
    }
    if (visible.size()<2 || ids.bytes!=diagnosticIds.bytes)
      return reject("SDF Mesh Surface diagnostic colors changed raw IDs or hid the native arrangement");
  }
  if (instancePixels[0]==instancePixels[1])
    return reject("SDF radius edits did not reach native Spectrum Instancer Mesh Surface pixels");
  error.clear(); return true;
}
} // namespace

int main() {
  using namespace videowire::geometry;
#if !defined(__aarch64__)
  return fail("strict Geometry Core Metal test requires physical Apple Silicon") ? 0 : 1;
#endif
  auto &backend = arbitgpu::nativeFixtureSceneBackend();
  videohelper::geometry::NativeGeometryCoreCapabilitySource source(backend);
  PortContract contract;
  contract.carrier = CarrierKind::geometry3D;
  contract.overflow = OverflowPolicy::reject;
  contract.maxVertices = 3; contract.maxIndices = 3;
  ValueDescriptor value;
  value.carrier = CarrierKind::geometry3D; value.stableId = 10;
  value.sourceStableId = 11; value.sourceRevision = 1;
  GeometryData mesh;
  mesh.positions = {{-1.5f,-1.5f,0},{1.5f,-1.5f,0},{0,1.5f,0}};
  mesh.vertexIds = {1,2,3}; mesh.indices = {0,1,2}; value.data = std::move(mesh);
  value.operations = {{10,OperationCode::copy,0,10,{0,0,0,0}}}; value.dispatchCount = 1;
  std::string error;
  if (!videohelper::tests::sceneclosest::runMetal(error)) return fail(error) ? 0 : 1;
  if (!metaltriangletransporttests::verify(backend,error)) return fail(error) ? 0 : 1;
  if (!videohelper::tests::environmentsecondaryfixture::run(backend,error)) return fail(error) ? 0 : 1;
  if (!verifyNativeGeometryAdmission(backend,error)) return fail(error) ? 0 : 1;
  if (!verifyNativeParticlePoints(backend,readBgra8,error)) return fail(error) ? 0 : 1;
  if (!verifyNativeParticlePointsBody(backend,readBgra8,error)) return fail(error) ? 0 : 1;
  if (!verifyNativeDistributedInstances(backend, readBgra8, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeConstructedFields(backend, readBgra8, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeImportedGeometry(backend, readBgra8, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeInstanceAppearance(backend, readBgra8, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeSpectrumDomains(backend,readBgra8,error)) return fail(error) ? 0 : 1;
  if (!verifyNativeSpectrumInstancerLayouts(backend,readBgra8,error)) return fail(error) ? 0 : 1;
  if (!verifyNativeSpectrumInstancerAppearance(backend,readBgra8,error)) return fail(error) ? 0 : 1;
  if (!verifyNativeRetainedSdfSurface(backend,readBgra8,error)) return fail(error) ? 0 : 1;
  if (!verifyNativeInstanceMaterialFields(backend, readBgra8, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeInstanceScoreFields(backend, readBgra8, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeReactiveSurface(backend, readBgra8, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeMixedSurfaceDiffraction(backend, readBgra8, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeReactiveFrame(backend, readBgra8, ordinarySharedScene(), error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeSolidBodies(readBgra8,error)) return fail(error) ? 0 : 1;
  if (!verifyNativeSolidBodies(readBgra8,error,{},true)) return fail(error) ? 0 : 1;
  if (!verifyNativeInstancedSolidBodies(readBgra8,readMotion,error)) return fail(error) ? 0 : 1;
  if (!verifyNativeSolidBodyMotionProduction(readBgra8,readMotion,error)) return fail(error) ? 0 : 1;
  if (!verifyNativeComposedSolidBodyMotion(readMotion,error)) return fail(error) ? 0 : 1;
  {
    const auto imported=videohelper::gltf::adaptGlbMeshToGeometryCore(importedGeometryFixture(),1,9001,9000,1,error);
    if (!imported || !verifyNativeSolidBodies(readBgra8,error,
        std::make_shared<const videowire::geometry::ValueDescriptor>(imported->descriptor()))) return fail(error) ? 0 : 1;
  }
  if (!verifyNativeObjectSurfacePrograms(backend, readBgra8, ordinarySharedScene(), error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeMaterialTables(backend, readBgra8, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeSpectrumInstancer(backend, readBgra8, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeSpectralHistory(backend, readBgra8, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeSpectralHistoryRows(backend, readBgra8, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeAudioDeformer(backend, readBgra8, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeOrderedAudioGeometry(backend, readBgra8, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeScoreFields(backend, readBgra8, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeTimelineCurves(backend, readBgra8, error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeComposedTimelineGeometry(backend,readBgra8,true,error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeDirectionalSurfaceWave(backend,readBgra8,true,error))
    return fail(error) ? 0 : 1;
  if (!verifyNativeGeometryComposition(backend, readBgra8, true, error))
    return fail(error) ? 0 : 1;
  auto admitted = admitValue(value, contract, error);
  if (!admitted) return fail(error) ? 0 : 1;
  videohelper::geometry::GeometryCorePlanRuntime runtime(source);
  auto plan = runtime.admitPreview({1,1,1,1,1,videohelper::geometry::PlanUse::preview},
      lowerRuntimePlan(contract,*admitted), ResourceLimits{}, {}, error);
  if (!plan) return fail(error) ? 0 : 1;
  auto execution = videohelper::geometry::executeNativeGeometry(backend,*plan,64,64,error);
  if (!execution) return fail(error) ? 0 : 1;
  const auto pixels = readBgra8(execution->frame);
  if (pixels.size() != 64u*64u*4u) return fail("Metal readback failed") ? 0 : 1;
  const auto energy = [&](unsigned x,unsigned y) {
    const auto i=(y*64u+x)*4u; return pixels[i]+pixels[i+1]+pixels[i+2];
  };
  if (energy(32,32) <= energy(1,1)+20)
    return fail("Metal pixels disagree with the independent triangle oracle") ? 0 : 1;

  PortContract geometryContract;
  geometryContract.carrier = CarrierKind::geometry3D;
  geometryContract.overflow = OverflowPolicy::reject;
  geometryContract.maxVertices = 4; geometryContract.maxIndices = 6;
  PortContract pointsContract;
  pointsContract.carrier = CarrierKind::points3D;
  pointsContract.overflow = OverflowPolicy::reject; pointsContract.maxPoints = 4;
  PortContract instancesContract;
  instancesContract.carrier = CarrierKind::instances3D;
  instancesContract.overflow = OverflowPolicy::reject; instancesContract.maxInstances = 4;
  auto pointGrid = lowerGrid(10,11,1,2,2,2.0f,geometryContract,error);
  auto points = pointGrid ? lowerPointsFromVertices(*pointGrid,12,pointsContract,error)
                          : std::nullopt;
  auto sourceGrid = lowerGrid(20,11,1,2,2,0.5f,geometryContract,error);
  auto instances = points && sourceGrid
      ? lowerInstanceOnPoints(*points,sourceGrid->stableId,30,instancesContract,error)
      : std::nullopt;
  if (!instances) return fail(error) ? 0 : 1;
  instances->operations.insert(instances->operations.begin(),
      sourceGrid->operations.begin(),sourceGrid->operations.end());
  instances->dispatchCount += sourceGrid->dispatchCount;
  auto admittedInstances = admitValue(*instances,instancesContract,error);
  if (!admittedInstances) return fail(error) ? 0 : 1;
  const auto instanceBytes = lowerRuntimePlan(instancesContract,*admittedInstances);
  auto instancePlan = runtime.admitPreview(
      {1,1,1,2,1,videohelper::geometry::PlanUse::preview},instanceBytes,
      ResourceLimits{}, {}, error);
  if (!instancePlan) return fail(error) ? 0 : 1;
  const auto canonicalInputs = noteRuntimeInputs();
  const auto duplicateInputs = noteRuntimeInputs(true,0.0f,true);
  const auto frequencyMutationInputs = noteRuntimeInputs(true,0.0f,false);
  const auto singleInputs = noteRuntimeInputs(false);
  const auto canonicalBatch = arbitgpu::prepareNativeNoteInstances(canonicalInputs);
  const auto duplicateBatch = arbitgpu::prepareNativeNoteInstances(duplicateInputs);
  const auto frequencyMutationBatch = arbitgpu::prepareNativeNoteInstances(
      frequencyMutationInputs);
  const auto near = [](float left,float right) {
    return std::abs(left-right) <= 0.00001f;
  };
  const float expectedOnsetOffset = 0.2f;
  const float expectedFrequencyOffset = std::log2(660.0f/440.0f);
  const float projectedOnsetPixels = expectedOnsetOffset
      / (std::tan(0.7853981634f*0.5f)*5.0f)*32.0f;
  const float projectedFrequencyPixels = expectedFrequencyOffset
      / (std::tan(0.7853981634f*0.5f)*5.0f)*32.0f;
  if (canonicalBatch.count != 2 || duplicateBatch.count != 2
      || frequencyMutationBatch.count != 2
      || !near(canonicalBatch.transforms[1][0]-canonicalBatch.transforms[0][0],
               expectedOnsetOffset)
      || !near(canonicalBatch.transforms[1][1]-canonicalBatch.transforms[0][1],
               expectedFrequencyOffset)
      || duplicateBatch.transforms[0] != duplicateBatch.transforms[1]
      || !near(frequencyMutationBatch.transforms[1][0]
                   -frequencyMutationBatch.transforms[0][0],0.0f)
      || !near(frequencyMutationBatch.transforms[1][1]
                   -frequencyMutationBatch.transforms[0][1],expectedFrequencyOffset)
      || projectedOnsetPixels < 2.0f || projectedFrequencyPixels < 8.0f)
    return fail("independent CPU projection did not predict two visible Block C transforms") ? 0 : 1;
  std::cout << "cpu-projected-block-c-offset-pixels=" << projectedOnsetPixels
            << ',' << projectedFrequencyPixels << '\n';
  auto identityExecution = videohelper::geometry::executeNativeGeometry(
      backend,*instancePlan,64,64,error,true);
  auto coexistenceExecution = videohelper::geometry::executeNativeGeometry(
      backend,*instancePlan,64,64,error,true,canonicalInputs);
  auto duplicatedNoteExecution = videohelper::geometry::executeNativeGeometry(
      backend,*instancePlan,64,64,error,true,duplicateInputs);
  auto frequencyMutationExecution = videohelper::geometry::executeNativeGeometry(
      backend,*instancePlan,64,64,error,true,frequencyMutationInputs);
  auto ignoredNoteExecution = videohelper::geometry::executeNativeGeometry(
      backend,*instancePlan,64,64,error,true,singleInputs);
  if (!identityExecution || !coexistenceExecution || !duplicatedNoteExecution
      || !frequencyMutationExecution || !ignoredNoteExecution)
    return fail(error) ? 0 : 1;
  const std::vector<StableId> exactIdentities {1,2,3,4};
  if (identityExecution->drawnInstanceIds != exactIdentities
      || coexistenceExecution->drawnInstanceIds != exactIdentities
      || coexistenceExecution->stats.drawCount != 2
      || coexistenceExecution->stats.ordinaryDrawCount != 0
      || coexistenceExecution->stats.instancedDrawCount != 2
      || coexistenceExecution->stats.instanceBufferUploadCount != 1
      || coexistenceExecution->stats.submittedInstanceCount != 8
      || coexistenceExecution->stats.noteInstanceDrawCount != 2
      || coexistenceExecution->stats.noteInstanceTransformUploadCount != 1
      || coexistenceExecution->stats.submittedNoteInstanceCount != 8)
    return fail("same-frame Geometry Core and Block C receipts are not independent") ? 0 : 1;
  const auto identityPixels = readBgra8(identityExecution->frame);
  const auto coexistencePixels = readBgra8(coexistenceExecution->frame);
  const auto duplicatedNotePixels = readBgra8(duplicatedNoteExecution->frame);
  const auto frequencyMutationPixels = readBgra8(frequencyMutationExecution->frame);
  const auto ignoredNotePixels = readBgra8(ignoredNoteExecution->frame);
  if (!verifyNativeNoteAppearance(backend, readBgra8, true, error))
    return fail(error) ? 0 : 1;
  const auto colorAt = [](const std::vector<std::uint8_t>& image,
                          unsigned x,unsigned y) {
    const auto offset=(y*64u+x)*4u;
    return std::array<std::uint8_t,3>{image[offset],image[offset+1],image[offset+2]};
  };
  const std::array<std::array<unsigned,2>,4> probes {{{16,16},{47,16},{16,47},{47,47}}};
  std::array<std::array<std::uint8_t,3>,4> colors {};
  for (std::size_t index=0;index<probes.size();++index)
    colors[index]=colorAt(identityPixels,probes[index][0],probes[index][1]);
  for (std::size_t left=0;left<colors.size();++left)
    for (std::size_t right=left+1;right<colors.size();++right)
      if (colors[left] == colors[right])
        return fail("exact Geometry Core transforms or stable identity pixels collapsed") ? 0 : 1;
  const auto hasDistinctBlockCTransforms = [&](const char* label,
                                                const std::vector<std::uint8_t>& candidate) {
    if (candidate.size() != identityPixels.size()
        || ignoredNotePixels.size() != identityPixels.size()) return false;
    for (const auto& identityColor : colors) {
      std::size_t retainedFirstTransformPixels = 0;
      std::size_t distinctSecondTransformPixels = 0;
      for (std::size_t offset = 0; offset < candidate.size(); offset += 4) {
        const std::array<std::uint8_t,3> candidateColor {
            candidate[offset], candidate[offset+1], candidate[offset+2]};
        const std::array<std::uint8_t,3> firstTransformColor {
            ignoredNotePixels[offset], ignoredNotePixels[offset+1],
            ignoredNotePixels[offset+2]};
        if (candidateColor == identityColor && firstTransformColor == identityColor)
          ++retainedFirstTransformPixels;
        if (candidateColor == identityColor && firstTransformColor != identityColor)
          ++distinctSecondTransformPixels;
      }
      std::cout << label << " identity-bgr="
                << static_cast<unsigned>(identityColor[0]) << ','
                << static_cast<unsigned>(identityColor[1]) << ','
                << static_cast<unsigned>(identityColor[2]) << " retained-pixels="
                << retainedFirstTransformPixels << " second-transform-pixels="
                << distinctSecondTransformPixels << '\n';
      if (retainedFirstTransformPixels < 16 || distinctSecondTransformPixels < 16)
        return false;
    }
    return true;
  };
  if (!hasDistinctBlockCTransforms("canonical", coexistencePixels)
      || hasDistinctBlockCTransforms("duplicate", duplicatedNotePixels)
      || !hasDistinctBlockCTransforms("frequency-mutation", frequencyMutationPixels)
      || hasDistinctBlockCTransforms("single", ignoredNotePixels)
      || duplicatedNotePixels != ignoredNotePixels
      || frequencyMutationPixels == ignoredNotePixels)
    return fail("combined Metal pixels did not preserve four Geometry identities under both distinct Block C transforms") ? 0 : 1;

  auto ordinaryScene = ordinarySharedScene();
  auto ordinaryPreparation = backend.prepare(ordinaryScene);
  auto ordinaryRender = ordinaryPreparation.prepared
      ? backend.render(ordinaryScene,ordinaryPreparation.resources,64,64)
      : arbitgpu::NativeFixtureSceneSubmission{};
  if (!ordinaryRender.rendered || ordinaryRender.stats.ordinaryDrawCount != 2
      || ordinaryRender.stats.instancedDrawCount != 0
      || ordinaryRender.stats.instanceBufferUploadCount != 0)
    return fail("ordinary Scene3D was routed through Geometry Core instancing") ? 0 : 1;
  const auto malformed = backend.prepareGeometryInstances(
      ordinaryScene,{},plan->plan);
  if (malformed.prepared || malformed.resources || malformed.error.empty())
    return fail("Metal accepted malformed or mismatched Geometry Core authority") ? 0 : 1;

  const auto oldGeneration = ignoredNoteExecution->frame->colorTextureDescriptor().rendererGeneration;
  std::shared_ptr<const arbitgpu::NativeFixtureSceneResources> retainedResource;
  {
    std::lock_guard<std::mutex> lock(instancePlan->nativeResources->mutex);
    retainedResource = std::static_pointer_cast<const arbitgpu::NativeFixtureSceneResources>(
        instancePlan->nativeResources->value);
  }
  if (!retainedResource || retainedResource != ignoredNoteExecution->resources)
    return fail("runtime cache did not retain the latest Metal native resource") ? 0 : 1;
  std::weak_ptr<const arbitgpu::NativeFixtureSceneResources> retired = retainedResource;
  auto explicitLease = instancePlan->ownerLease;
  retainedResource.reset();
  identityExecution.reset(); coexistenceExecution.reset();
  duplicatedNoteExecution.reset(); frequencyMutationExecution.reset();
  ignoredNoteExecution.reset(); instancePlan.reset();
  execution.reset(); plan.reset();
  if (!runtime.reconcileGeneration(2,1,1,error) || retired.expired()
      || runtime.admissionsInGeneration() != 0)
    return fail("generation replacement retired a Metal resource with a live lease") ? 0 : 1;
  explicitLease.reset();
  if (!runtime.reconcileGeneration(2,1,1,error) || !retired.expired()
      || runtime.admissionsInGeneration() != 0)
    return fail("generation replacement did not retire the latest lease-free Metal resource") ? 0 : 1;
  auto replacementPlan = runtime.admitPreview(
      {2,1,1,2,1,videohelper::geometry::PlanUse::preview},instanceBytes,
      ResourceLimits{}, {}, error);
  auto replacement = replacementPlan
      ? videohelper::geometry::executeNativeGeometry(backend,*replacementPlan,64,64,error)
      : std::nullopt;
  if (!replacement || runtime.admissionsInGeneration() != 1
      || replacement->frame->colorTextureDescriptor().rendererGeneration <= oldGeneration)
    return fail("replacement generation reused retired Metal resources") ? 0 : 1;
  return 0;
}
