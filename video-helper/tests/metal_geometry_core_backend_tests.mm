#include "../src/geometry_core_backend.h"
#include "../src/canonical_block_c_frame.h"
#include "sokol_gfx.h"
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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
