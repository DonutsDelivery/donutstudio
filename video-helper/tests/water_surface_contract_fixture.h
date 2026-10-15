#pragma once

#include "../../shared/GeometryTimelineSceneBinding.h"
#include <cstring>

// Synthetic frame heights verify the typed contract, not water physics.
template <typename Check>
inline void waterSurfaceTypedContract(Check check) {
  using namespace videowire::geometry;
  using namespace HarmonicMIDI::grid;
  static_assert(static_cast<std::uint8_t>(OperationCode::sdfSurface) == 134);
  static_assert(static_cast<std::uint8_t>(OperationCode::fieldWaterHeight) == 136);
  PortContract contract; contract.carrier=CarrierKind::geometry3D; contract.overflow=OverflowPolicy::reject;
  contract.maxVertices=2048; contract.maxIndices=12288;
  std::string error;
  auto grid=lowerGrid(10101,10101,1,5,4,0.25f,contract,error);
  check(bool(grid),"water rest grid lowers");
  if (!grid) return;
  auto water=lowerWaterSurfaceSimulation(*grid,10102,10103,1,2,0.3,0.4,contract,error);
  check(bool(water),"water authored controls lower through typed geometry");
  if (!water) return;
  const auto value=admitValue(*water,contract,error);
  check(bool(value),"water admission retains authored rest mesh");
  if (!value) return;
  check(equal(std::get<GeometryData>(water->data),std::get<GeometryData>(grid->data)),
        "water admission persists rest positions and exact identities/topology only");
  const auto bytes=encodeRuntimeValue(*value);
  check(bytes.size()>5 && bytes[4]==13 && bytes[5]==0,"water uses unchanged non-text GCRV v13 wire shape");
  // Frozen released water payload, including the reserved zero doubles,
  // source topology and all tail counts. No current encoder builds this oracle.
  const std::string legacyHex=
      "474352560d00000002000000017627000000000000752700000000000001000000000000000100000000000000000000"
      "000400000000000000000000000000000004000000000000007527000000000000057527000000000000752700000000"
      "000000000000000000000000a040000080400000803e0000000077270000000000008875270000000000007727000000"
      "0000000000000000000000000000000000000000000000000000000201000000000000000000000000000000f03f0000"
      "000000000040333333333333d33f9a9999999999d93f0000000000000000000000000000000000000000000000000000"
      "000000000000000078270000000000005d77270000000000007827000000000000000000000000000000000000000000"
      "000000000000000000020201000000000000000000000000000000000000000000000000000000000000f03f00000000"
      "000000000000000000000000000000000000000000000000000000000000000000000000000076270000000000001675"
      "27000000000000762700000000000078270000000000000000803f000000000000000000000000140000000000000048"
      "00000000000000000000bf0000c0be000000000100000000000000000080be0000c0be00000000020000000000000000"
      "0000000000c0be0000000003000000000000000000803e0000c0be0000000004000000000000000000003f0000c0be00"
      "0000000500000000000000000000bf000000be000000000600000000000000000080be000000be000000000700000000"
      "00000000000000000000be0000000008000000000000000000803e000000be0000000009000000000000000000003f00"
      "0000be000000000a00000000000000000000bf0000003e000000000b00000000000000000080be0000003e000000000c"
      "00000000000000000000000000003e000000000d000000000000000000803e0000003e000000000e0000000000000000"
      "00003f0000003e000000000f00000000000000000000bf0000c03e000000001000000000000000000080be0000c03e00"
      "0000001100000000000000000000000000c03e0000000012000000000000000000803e0000c03e000000001300000000"
      "0000000000003f0000c03e00000000140000000000000000000000010000000500000001000000060000000500000001"
      "000000020000000600000002000000070000000600000002000000030000000700000003000000080000000700000003"
      "000000040000000800000004000000090000000800000005000000060000000a000000060000000b0000000a00000006"
      "000000070000000b000000070000000c0000000b00000007000000080000000c000000080000000d0000000c00000008"
      "000000090000000d000000090000000e0000000d0000000a0000000b0000000f0000000b000000100000000f0000000b"
      "0000000c000000100000000c00000011000000100000000c0000000d000000110000000d00000012000000110000000d"
      "0000000e000000120000000e0000001300000012000000000000000000000000000000";
  const auto golden=decodeLoweredPlanText(legacyHex,error);
  check(golden && *golden==bytes,"default water remains byte-identical to the released full GCRV thirteen fixture");
  const auto explicitZero=lowerWaterSurfaceSimulation(*grid,10102,10103,1,2,0.3,0.4,contract,error,0);
  const auto zeroValue=explicitZero ? admitValue(*explicitZero,contract,error):std::nullopt;
  check(zeroValue && encodeRuntimeValue(*zeroValue)==bytes,"absent and explicit zero coupling mass have identical legacy bytes");
  // This fixed descriptor's first grid operation occupies 49 bytes after
  // its 73-byte header. Slot 4 follows the next operation's 49+11-byte prefix.
  constexpr std::size_t massOffset=73+49+49+11+4*8;
  const auto writeDouble=[](std::vector<std::uint8_t>& payload,std::size_t offset,double value) {
    std::uint64_t bits=0; std::memcpy(&bits,&value,sizeof(bits));
    for (std::size_t i=0;i<8;++i) payload[offset+i]=static_cast<std::uint8_t>(bits>>(8*i));
  };
  for (const double mass:{.01,2.5,100.0}) {
    const auto authored=lowerWaterSurfaceSimulation(*grid,10102,10103,1,2,0.3,0.4,contract,error,mass);
    const auto coupled=authored ? admitValue(*authored,contract,error):std::nullopt;
    check(bool(coupled),"bounded positive simulation-space mass per area admits as authored rest");
    if (!coupled) continue;
    auto wire=encodeRuntimeValue(*coupled);
    check(wire.size()==bytes.size() && wire[4]==15 && wire[5]==0
        && coupled->descriptor().operations[1].field.values==std::array<double,8>{1,2,.3,.4,mass,0,0,0},
        "positive mass appends GCRV fifteen with exactly the existing eight-double carrier and payload length");
    const auto roundtrip=decodeRuntimeValue(wire,contract,{}, {},error);
    check(roundtrip && encodeRuntimeValue(*roundtrip)==wire
        && roundtrip->descriptor().operations[1].field.values[4]==mass,
        "GCRV fifteen roundtrips exact receiver mass and retained water identity");
    const auto plan=lowerRuntimePlan(contract,*coupled);
    const auto reopened=decodeLoweredRuntimePlan(plan,error);
    check(plan[4]==1 && plan[5]==0 && reopened
        && reopened->runtimeValue==wire,"GCLP retains its existing version and embeds the gated water payload");
    for (const std::uint8_t version:{13,14}) {
      auto downgraded=wire; downgraded[4]=version;
      check(!decodeRuntimeValue(downgraded,contract,{}, {},error) && error.find("version 15")!=std::string::npos,
          "older GCRV headers refuse positive receiver mass rather than silently accepting it");
    }
    auto legacy=wire; legacy[4]=13; writeDouble(legacy,massOffset,0);
    check(legacy==bytes,"removing only opt-in mass and its version restores every legacy water byte");
    for (const double invalid:{-.01,.001,100.01,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
      auto tampered=wire; writeDouble(tampered,massOffset,invalid);
      check(!decodeRuntimeValue(tampered,contract,{}, {},error),"tampered invalid coupling mass refuses wire admission");
    }
    for (std::size_t slot=5;slot<8;++slot) {
      auto reserved=wire; writeDouble(reserved,massOffset+(slot-4)*8,1);
      check(!decodeRuntimeValue(reserved,contract,{}, {},error),"GCRV fifteen still refuses nonzero water slots five through seven");
    }
  }
  for (const double invalid:{-.01,.001,100.01,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
    check(!lowerWaterSurfaceSimulation(*grid,10102,10103,1,2,0.3,0.4,contract,error,invalid),
        "authoring refuses mass outside zero or the bounded positive range");
  const auto decoded=decodeRuntimeValue(bytes,contract,{}, {},error);
  check(decoded && encodeRuntimeValue(*decoded)==bytes,"water authored operation roundtrip is byte exact");
  auto oldVersion=bytes; oldVersion[4]=12;
  check(!decodeRuntimeValue(oldVersion,contract,{}, {},error),"water opcode refuses transport predating authored water admission");
  check(water->operations.size()==4 && water->operations[1].field.values==std::array<double,8>{1,2,0.3,0.4,0,0,0,0},
        "water retains four authored controls and no solver state");
  RuntimeFieldEvaluation runtime; runtime.timelineSeconds=0.5;
  MaterializedInstancePlan replay;
  check(!validateOperationPlan(*water,{},&replay,error,nullptr,&runtime),
        "water runtime refuses absent historical replay even at authored rest");
  runtime.waterHeightAt=[](const auto&,const auto&,const auto&,double,auto&,std::string& diagnostic) {
    diagnostic="Missing exact historical source"; return false;
  };
  check(!validateOperationPlan(*water,{},&replay,error,nullptr,&runtime),"water runtime propagates historical-source refusal");
  runtime.waterHeightAt=[](const auto&,const auto& mesh,const auto&,double,auto& heights,std::string&) {
    heights.assign(mesh.positions.size()-1,0); return true;
  };
  check(!validateOperationPlan(*water,{},&replay,error,nullptr,&runtime),"water runtime refuses changed sample count");
  runtime.waterHeightAt=[](const auto&,const auto& mesh,const auto&,double,auto& heights,std::string&) {
    heights.assign(mesh.positions.size(),0); heights.back()=std::numeric_limits<double>::quiet_NaN(); return true;
  };
  check(!validateOperationPlan(*water,{},&replay,error,nullptr,&runtime),"water runtime refuses nonfinite heights");
  int calls=0;
  runtime.waterHeightAt=[&](const OperationRecord& op,const GeometryData& mesh,const WaterSurfaceGrid& shape,
                           double seconds,std::vector<double>& heights,std::string& diagnostic) {
    ++calls;
    check(op.code==OperationCode::fieldWaterHeight && shape.columns==5 && shape.rows==4,
          "water producer receives exact typed operation and rest grid");
    heights.resize(mesh.positions.size());
    for (std::size_t i=0;i<heights.size();++i) heights[i]=seconds*mesh.positions[i].x;
    diagnostic.clear(); return true;
  };
  auto scene=std::make_shared<Visual3DScene>(); scene->id={7}; scene->activeCamera={50};
  scene->cameras[0].id={50}; scene->cameraCount=1; scene->cameras[0].transform.translation.z=4;
  SceneObjectRecord object; object.id={10}; SceneMaterialRecord material; material.id={20};
  check(appendGeometryToScene(*value,object,material,*scene,error),"water authored rest mesh appends to scene");
  TimelineSceneBinding binding{{10},encodeLoweredPlanText(lowerRuntimePlan(contract,*value))};
  check(bool(decodeTimelineScenePlan(binding,error)),"water admits existing timeline scene subset");
  const auto original=std::make_shared<Visual3DScene>(*scene);
  check(!replaceTimelineSceneMesh(binding,*value,0.5,*scene,error),"water composed frame refuses missing callback");
  check(scene->vertices[0].position.z==original->vertices[0].position.z,"water refusal leaves scene mesh untouched");
  check(replaceTimelineSceneMesh(binding,*value,0.5,*scene,error,&runtime),"water composed frame consumes runtime heights");
  check(calls==1,"water frame shares one replay result between mesh verification and normal generation");
  const auto& rest=std::get<GeometryData>(grid->data);
  for (std::size_t i=0;i<rest.positions.size();++i) {
    check(scene->vertices[i].position.x==rest.positions[i].x && scene->vertices[i].position.y==rest.positions[i].y
          && scene->vertices[i].position.z==0.5f*rest.positions[i].x,"water displaces only local Z in retained vertex order");
    check(scene->vertices[i].normal.x!=original->vertices[i].normal.x,"water scene rebuilds normals from displaced geometry");
  }
  check(std::equal(rest.indices.begin(),rest.indices.end(),scene->indices.begin()),"water preserves exact triangle order");
  auto again=std::make_shared<Visual3DScene>(*original);
  check(replaceTimelineSceneMesh(binding,*value,0.5,*again,error,&runtime) && calls==2,
        "another frame reconstructs heights without retaining mutable simulation state");
  for (std::size_t i=0;i<rest.positions.size();++i)
    check(sameSceneVertex(scene->vertices[i],again->vertices[i]),"equal-time frame requests produce equal mesh and normals");
  check(encodeRuntimeValue(*value)==bytes,"runtime heights never change authored transport");
  auto combined=*water;
  auto second=combined.operations[1]; second.stableId=second.outputStableId=10160; second.field.values[2]=0.6;
  FieldOperationSettings sumSettings; sumSettings.domain=Domain::vertex; sumSettings.valueType=ValueType::floatValue;
  OperationRecord sum{10161,OperationCode::fieldMath,10103,10161,{},10160,{},sumSettings};
  combined.operations[2].inputStableId=10161;
  combined.operations.insert(combined.operations.begin()+2,second);
  combined.operations.insert(combined.operations.begin()+3,sum);
  combined.dispatchCount=combined.operations.size();
  const auto combinedValue=admitValue(combined,contract,error);
  check(bool(combinedValue),"independent water fields compose through existing typed field math");
  if (combinedValue) {
    TimelineSceneBinding combinedBinding{{10},encodeLoweredPlanText(lowerRuntimePlan(contract,*combinedValue))};
    auto target=std::make_shared<Visual3DScene>(*original);
    int combinedCalls=0;
    runtime.waterHeightAt=[&](const auto& op,const auto& mesh,const auto&,double seconds,auto& heights,std::string&) {
      ++combinedCalls; heights.assign(mesh.positions.size(),op.field.values[2]*seconds); return true;
    };
    check(replaceTimelineSceneMesh(combinedBinding,*combinedValue,0.5,*target,error,&runtime),
          "multiple typed water operations replay in one composed frame");
    check(combinedCalls==2 && std::abs(target->vertices[0].position.z-0.45)<0.000001,
          "frame memo retains separate operation identities/settings and calls each producer once");
  }

  PortContract pointContract; pointContract.carrier=CarrierKind::points3D; pointContract.maxPoints=32;
  PortContract instanceContract; instanceContract.carrier=CarrierKind::instances3D; instanceContract.maxInstances=32;
  auto points=lowerPointsFromVertices(*grid,10171,pointContract,error);
  auto cube=lowerMeshGenerator(10173,10173,1,OperationCode::cube,{0.1f,0.1f,0.1f,0},contract,error);
  auto instances=points ? lowerInstanceOnPoints(*points,10173,10172,instanceContract,error) : std::nullopt;
  if (instances && cube) {
    auto operations=cube->operations;
    check(appendUniqueOperations(operations,instances->operations,error),"water guard fixture retains static prototype");
    instances->operations=std::move(operations); instances->dispatchCount=instances->operations.size();
    PortContract fieldContract; fieldContract.carrier=CarrierKind::field; fieldContract.fieldDomain=Domain::instance;
    fieldContract.fieldValueType=ValueType::vector; fieldContract.fieldInterpolation=Interpolation::constant;
    fieldContract.maxFieldElements=32;
    scorefield::Binding score; score.scoreSourceStableId=10199;
    auto scored=lowerInstanceScoreField(*instances,10174,score,fieldContract,error);
    check(bool(scored),"water guard fixture admits original static instance Score source");
    if (scored) {
      auto& operations=scored->operations;
      const auto pointOp=std::find_if(operations.begin(),operations.end(),[](const auto& op) {
        return op.code==OperationCode::pointsFromVertices;
      });
      if (pointOp!=operations.end()) {
        pointOp->inputStableId=water->stableId;
        const auto cursor=pointOp-operations.begin();
        operations.insert(operations.begin()+cursor,water->operations.begin()+1,water->operations.end());
        scored->dispatchCount=operations.size();
        check(!admitValue(*scored,fieldContract,error)
              && error.find("fixed ordered retained placements")!=std::string::npos,
              "water runtime ancestry cannot enter fixed instance Score placements");
      } else check(false,"water guard fixture requires retained vertex distribution");
    }
  } else check(false,"water guard fixture requires static points and prototype");

  WaterSurfaceGrid shape;
  auto hostile=rest; hostile.indices[1]=hostile.indices[2];
  check(!validateWaterSurfaceGrid(hostile,shape,error),"water refuses reordered grid topology");
  hostile=rest; hostile.positions[2].x+=0.01f;
  check(!validateWaterSurfaceGrid(hostile,shape,error),"water refuses irregular spacing");
  hostile=rest; hostile.positions.back().z=0.1f;
  check(!validateWaterSurfaceGrid(hostile,shape,error),"water refuses nonplanar rest mesh");
  hostile=rest; hostile.vertexIds.back()=hostile.vertexIds.front();
  check(!validateWaterSurfaceGrid(hostile,shape,error),"water refuses duplicated retained identities");
  hostile=rest; hostile.positions.back().x=std::numeric_limits<float>::infinity();
  check(!validateWaterSurfaceGrid(hostile,shape,error),"water refuses nonfinite rest positions");
  auto cap=lowerGrid(10110,10110,1,32,32,0.1f,contract,error);
  check(cap && lowerWaterSurfaceSimulation(*cap,10111,10112,1,1,0,0,contract,error),
        "water accepts exact 32 by 32 cap with finite float spacing");
  auto over=lowerGrid(10120,10120,1,33,2,0.25f,contract,error);
  check(over && !lowerWaterSurfaceSimulation(*over,10121,10122,1,1,0,0,contract,error),"water refuses over-cap axis");
  check(!lowerWaterSurfaceSimulation(*grid,10130,10131,100,2,0,0,contract,error),"water refuses unstable Courant coefficient");
  check(!lowerWaterSurfaceSimulation(*grid,10130,10131,1,121,0,0,contract,error),"water refuses unstable damping");
  check(!lowerWaterSurfaceSimulation(*grid,10130,10131,1,2,std::numeric_limits<double>::infinity(),0,contract,error),
        "water refuses nonfinite authored gain");
  auto named=*grid;
  AttributeDescriptor normal; normal.name="normal"; normal.stableId=10140; normal.domain=Domain::vertex;
  normal.valueType=ValueType::vector; normal.interpolation=Interpolation::normalizedLinear;
  named.attributes.push_back({normal,std::vector<FieldElement>(rest.positions.size(),FieldElement{{0,0,1,0}})});
  check(!lowerWaterSurfaceSimulation(named,10141,10142,1,2,0,0,contract,error),
        "water refuses exact named normal attribute instead of retaining stale authored normals");
}
