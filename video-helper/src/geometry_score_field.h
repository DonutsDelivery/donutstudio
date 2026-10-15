#pragma once

#include "gpu_backend/backend.h"
#include "../../shared/GeometryScoreField.h"

namespace videohelper::scorefield {
struct Evaluation final {
  std::vector<std::array<float, 3>> offsets;
  std::vector<std::int64_t> closestNoteIds, closestLinkIds;
  std::vector<std::array<float, 2>> linkRatios;
};

inline bool evaluate(const videowire::geometry::scorefield::Binding& binding,
                     const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>& frame,
                     Evaluation& result, std::string& error) {
  namespace contract = videowire::geometry::scorefield;
  result = {};
  if (!contract::validate(binding,binding.positions.size(),error)) return false;
  if (!canonicalblockc::valid(frame)) {
    error = "Score Field requires the canonical score frame for this preview or export time";
    return false;
  }
  arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
  inputs.canonicalBlockCFrame=frame; inputs.noteInstanceMapping=binding.mapping;
  const auto notes=arbitgpu::prepareNativeNoteInstances(inputs);
  if (notes.count == 0) {
    error = "Score Field has no resident unmuted notes in the score history/lookahead window";
    return false;
  }
  const auto texel=[&](std::size_t note, std::size_t column, std::size_t channel) {
    return frame->noteTextureValues[(notes.canonicalRows[note]*arbitblockc::kTexelsPerNote+column)*4+channel];
  };
  std::array<int,arbitblockc::kMaxNotes> instanceForRow;
  instanceForRow.fill(-1);
  std::array<double,arbitblockc::kMaxNotes> pitch {}, spacing {};
  for (std::size_t note=0;note<notes.count;++note) {
    instanceForRow[notes.canonicalRows[note]]=static_cast<int>(note);
    if (texel(note,1,3)!=0 && texel(note,1,3)!=1) {
      error="Score Field note root flag is invalid"; return false;
    }
    pitch[note]=12.0*std::log2(static_cast<double>(texel(note,1,0))/frame->rootFrequencyHz);
    for (std::size_t axis=0;axis<3;++axis)
      if (!std::isfinite(notes.transforms[note][axis])) {
        error="Score Field note position is not finite"; return false;
      }
  }
  if (binding.mode == contract::Mode::pitchSpacing) {
    for (std::size_t note=0;note<notes.count;++note) {
      double interval=std::numeric_limits<double>::infinity();
      for (std::size_t other=0;other<notes.count;++other) {
        const auto distance=std::abs(pitch[note]-pitch[other]);
        if (distance>1.0e-6) interval=std::min(interval,distance);
      }
      spacing[note]=std::isfinite(interval) ? interval : 0.0;
    }
  }
  const bool linkMode=binding.mode == contract::Mode::harmonicLinkDistance
      || binding.mode == contract::Mode::harmonicLinkRatio;
  if (linkMode && frame->linkRows() == 0) {
    error="Score Field has no resident harmonic links in the canonical score frame"; return false;
  }
  const auto distanceSquared=[](const auto& a,const auto& b) {
    double value=0;
    for (std::size_t axis=0;axis<3;++axis) {
      const double delta=static_cast<double>(a[axis])-b[axis]; value+=delta*delta;
    }
    return value;
  };
  result.offsets.reserve(binding.positions.size());
  result.closestNoteIds.reserve(binding.positions.size());
  result.closestLinkIds.reserve(binding.positions.size());
  result.linkRatios.reserve(binding.positions.size());
  for (const auto& position : binding.positions) {
    std::size_t nearest=0;
    double distance=std::numeric_limits<double>::infinity(), influence=0;
    for (std::size_t note=0;note<notes.count;++note) {
      const auto squared=distanceSquared(position,notes.transforms[note]);
      if (squared<distance || (squared==distance && notes.identities[note]<notes.identities[nearest])) {
        nearest=note; distance=squared;
      }
      if (binding.mode == contract::Mode::velocityInfluence)
        influence+=texel(note,0,1)*std::max(0.0,1.0-std::sqrt(squared)/binding.radius);
    }
    std::array<double,3> value {};
    std::int64_t closestLink=0;
    std::array<float,2> ratio {};
    if (binding.mode == contract::Mode::primeLattice) {
      for (std::size_t axis=0;axis<3;++axis) {
        const auto prime=binding.primes[axis];
        value[axis]=texel(nearest,prime<4 ? 2 : 3,prime<4 ? prime : prime-4);
      }
    } else {
      double scalar=std::sqrt(distance);
      if (binding.mode == contract::Mode::velocityInfluence) scalar=influence;
      if (binding.mode == contract::Mode::pitchSpacing) scalar=spacing[nearest];
      if (binding.mode == contract::Mode::pitchBendSemitones) scalar=texel(nearest,3,3);
      if (binding.mode == contract::Mode::noteAgeBeats) scalar=texel(nearest,0,2);
      if (binding.mode == contract::Mode::remainingDurationBeats) scalar=texel(nearest,0,3);
      if (binding.mode == contract::Mode::rootFlag) scalar=texel(nearest,1,3);
      if (binding.mode == contract::Mode::midiNoteNumber) scalar=texel(nearest,0,0);
      if (binding.mode == contract::Mode::resolvedFrequencyHz) scalar=texel(nearest,1,0);
      if (linkMode) {
        double best=std::numeric_limits<double>::infinity();
        for (int row=0;row<frame->linkRows();++row) {
          const auto offset=static_cast<std::size_t>(row)*4;
          const auto slaveRow=frame->linkTextureValues[offset], masterRow=frame->linkTextureValues[offset+1];
          const auto numerator=frame->linkTextureValues[offset+2], denominator=frame->linkTextureValues[offset+3];
          if (frame->linkIdentities[static_cast<std::size_t>(row)]==0
              || !std::isfinite(numerator) || !std::isfinite(denominator)
              || numerator<=0 || denominator<=0
              || std::floor(numerator)!=numerator || std::floor(denominator)!=denominator) {
            error="Score Field harmonic link identity or ratio is invalid"; return false;
          }
          if (slaveRow<0 || masterRow<0 || slaveRow>=frame->noteRows() || masterRow>=frame->noteRows()
              || std::floor(slaveRow)!=slaveRow || std::floor(masterRow)!=masterRow) {
            error="Score Field harmonic link references an invalid canonical row"; return false;
          }
          const auto slave=instanceForRow[static_cast<std::size_t>(slaveRow)];
          const auto master=instanceForRow[static_cast<std::size_t>(masterRow)];
          if (slave<0 || master<0) continue;
          const auto& a=notes.transforms[static_cast<std::size_t>(master)];
          const auto& b=notes.transforms[static_cast<std::size_t>(slave)];
          double dot=0;
          const auto length=distanceSquared(a,b);
          for (std::size_t axis=0;axis<3;++axis)
            dot+=(static_cast<double>(position[axis])-a[axis])*(static_cast<double>(b[axis])-a[axis]);
          const auto t=length>0 ? std::clamp(dot/length,0.0,1.0) : 0.0;
          std::array<double,3> closest {};
          for (std::size_t axis=0;axis<3;++axis) closest[axis]=a[axis]+t*(static_cast<double>(b[axis])-a[axis]);
          const auto squared=distanceSquared(position,closest);
          const auto identity=frame->linkIdentities[static_cast<std::size_t>(row)];
          if (squared<best || (squared==best && identity<closestLink)) {
            best=squared; closestLink=identity;
            ratio={numerator,denominator};
          }
        }
        if (closestLink == 0) { error="Score Field has no harmonic links with resident endpoints"; return false; }
        scalar=binding.mode == contract::Mode::harmonicLinkRatio
            ? static_cast<double>(ratio[0])/ratio[1] : std::sqrt(best);
      }
      value.fill(scalar);
    }
    std::array<float,3> displacement {};
    for (std::size_t axis=0;axis<3;++axis) {
      const auto component=(value[axis]*binding.gain+binding.bias)*binding.direction[axis];
      if (!std::isfinite(component) || std::abs(component)>1000000) {
        error="Score Field displacement exceeds its finite geometry bounds"; return false;
      }
      displacement[axis]=static_cast<float>(component);
    }
    result.offsets.push_back(displacement);
    result.closestNoteIds.push_back(notes.identities[nearest]);
    result.closestLinkIds.push_back(closestLink); result.linkRatios.push_back(ratio);
  }
  error.clear(); return true;
}
inline bool evaluateInstanceField(const videowire::geometry::scorefield::Binding& binding,
    const videowire::geometry::FieldData& positions,
    const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>& frame,
    videowire::geometry::FieldData& result, std::string& error,
    bool exactRetainedFrameSamples = false) {
  result={};
  if (binding.positions.size()!=positions.elements.size()) {
    error="Instance Score Field lost its exact retained position samples"; return false;
  }
  for (std::size_t i=0;i<positions.elements.size();++i)
    for (std::size_t axis=0;axis<3;++axis)
      if (!exactRetainedFrameSamples && positions.elements[i].components[axis]!=binding.positions[i][axis]) {
        error="Instance Score Field samples differ from its admitted binding"; return false;
      }
  Evaluation evaluated;
  auto frameBinding=binding;
  // Ordered replay validates the source, count and ordered IDs before supplying
  // current frame positions. The authored binding remains immutable.
  if (exactRetainedFrameSamples)
    for (std::size_t i=0;i<positions.elements.size();++i)
      for (std::size_t axis=0;axis<3;++axis)
        frameBinding.positions[i][axis]=static_cast<float>(positions.elements[i].components[axis]);
  if (!evaluate(frameBinding,frame,evaluated,error)) return false;
  result.elements.resize(evaluated.offsets.size());
  for (std::size_t i=0;i<evaluated.offsets.size();++i) {
    const auto& v=evaluated.offsets[i]; result.elements[i].components={v[0],v[1],v[2],0};
  }
  return true;
}
inline bool evaluateSampleField(const videowire::geometry::OperationRecord& operation,
                                const videowire::geometry::FieldData& positions,
                                const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>& frame,
                                videowire::geometry::FieldData& result, std::string& error) {
  namespace geometry=videowire::geometry;
  if (!canonicalblockc::valid(frame)) {
    error="Score Sample Field requires the canonical score frame for this preview or export time"; return false;
  }
  const auto& p=operation.field.values;
  geometry::scorefield::Binding binding;
  binding.fieldStableId=operation.stableId; binding.sourceGeometryStableId=operation.inputStableId;
  binding.scoreSourceStableId=static_cast<std::uint64_t>(p[4]);
  if (!geometry::scorefield::sampleFieldMode(operation.field.mode,binding.mode)) {
    error="Score Sample Field has an unknown measurement"; return false;
  }
  std::uint8_t primeSlot=0;
  if (geometry::scorefield::samplePrimeLatticeSlot(operation.field.mode,primeSlot)) binding.primes[0]=primeSlot;
  const bool linkMeasure=binding.mode==geometry::scorefield::Mode::harmonicLinkDistance
      || binding.mode==geometry::scorefield::Mode::harmonicLinkRatio;
  binding.radius=static_cast<float>(p[0]); binding.gain=static_cast<float>(p[1]); binding.bias=static_cast<float>(p[2]);
  binding.mapping.xScale=binding.mapping.yScale=binding.mapping.zScale=static_cast<float>(p[3]);
  binding.direction={1,0,0};
  for (const auto& position : positions.elements) {
    binding.positions.push_back({static_cast<float>(position.components[0]),static_cast<float>(position.components[1]),
                                static_cast<float>(position.components[2])});
    binding.elementIds.push_back(binding.elementIds.size()+1);
  }
  if (!geometry::scorefield::validate(binding,positions.elements.size(),error)) return false;
  arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
  inputs.canonicalBlockCFrame=frame; inputs.noteInstanceMapping=binding.mapping;
  // Unlike the vector mode, which errors here, the sample path returns bias when no notes are resident.
  // Empty score windows leave the authored bias. No previous-frame state is retained.
  if (arbitgpu::prepareNativeNoteInstances(inputs).count==0 || (linkMeasure && frame->linkRows()==0)) {
    result.elements.assign(positions.elements.size(),geometry::FieldElement{{p[2],0,0,0}});
    error.clear(); return true;
  }
  Evaluation evaluated;
  if (!evaluate(binding,frame,evaluated,error)) return false;
  result.elements.resize(evaluated.offsets.size());
  for (std::size_t i=0;i<evaluated.offsets.size();++i)
    result.elements[i].components={evaluated.offsets[i][0],0,0,0};
  return true;
}
} // namespace videohelper::scorefield
