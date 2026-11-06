#pragma once

#include "geometry_core_backend.h"
#include "../../shared/GeometryAudioDeformer.h"
#include "../../shared/GeometryCoreScene.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace videohelper::geometry {

// The spectrum readers for a composed Scene3D replay. They read one frame's spectrum
// analysis the way executeNativeGeometry does: the master bands and readers, or the
// selected track or group stream. Each binding keeps its own follower for this frame.
// Audio Deformers, in-graph Audio Spectrum Fields and Set Position displacements all
// run inside the replay, because a composed scene has no scene adapter after it.
inline videowire::geometry::SpectrumFrameBinder
composedSpectrumFrame(std::shared_ptr<SpectrumEvaluation> frame, bool exporting) {
  namespace wire = videowire::geometry;
  namespace spectrum = videowire::geometry::spectrum;
  return [frame, exporting](wire::RuntimeFieldEvaluation& evaluation, const wire::ValueDescriptor& value,
                            std::string& error) {
    if (value.spectrumFields.empty()) return true;
    if (!frame) { error = "Composed audio spectrum requires the shared frame analysis"; return false; }
    struct Input final {
      spectrum::Bands current{};
      spectrum::FeaturesAt featuresAt;
      spectrum::ScalarInput scalars;
    };
    const auto input = [frame](const spectrum::Binding& binding, Input& read, std::string& failure) {
      read.current = frame->bands;
      read.featuresAt = binding.history.enabled ? frame->historyFeaturesAt : frame->featuresAt;
      read.scalars.current = &frame->scalars;
      read.scalars.at = binding.history.enabled ? frame->historyScalarFeaturesAt : frame->scalarFeaturesAt;
      read.scalars.beatDelay = frame->beatDelay;
      if (binding.source == spectrum::Source::master) {
        if (!binding.history.enabled && !frame->liveFrameAvailable && !frame->featuresAt) {
          failure = "Composed audio spectrum requires the shared frame analysis"; return false;
        }
        return true;
      }
      const auto sourceAt = frame->sourceFeaturesAt;
      if (!sourceAt || !sourceAt(binding.source, binding.sourceTrackId, frame->timeSeconds, read.current)) {
        failure = "Selected track/group spectrum is unavailable. Check the saved source ID, use the standalone app and refresh project audio analysis.";
        return false;
      }
      read.featuresAt = [sourceAt, source = binding.source, track = binding.sourceTrackId](double seconds) {
        spectrum::Bands selected{};
        sourceAt(source, track, seconds, selected);
        return selected;
      };
      read.scalars.at = [sourceScalars = frame->sourceScalarFeaturesAt, source = binding.source,
                         track = binding.sourceTrackId](double seconds, spectrum::Scalars& values) {
        return static_cast<bool>(sourceScalars) && sourceScalars(source, track, seconds, values);
      };
      return true;
    };
    auto followers = std::make_shared<std::map<wire::StableId, spectrum::Follower>>();
    const auto follower = [followers](const spectrum::Binding& binding) -> spectrum::Follower& {
      return (*followers)[binding.fieldStableId];
    };
    const auto bands = [frame, input, follower](const spectrum::Binding& binding, spectrum::Bands& followed,
                                                std::string& failure) {
      Input read;
      return input(binding, read, failure)
          && spectrum::evaluate(binding, read.current, frame->timeSeconds, read.featuresAt, follower(binding),
                                followed, failure, read.scalars);
    };
    const auto rows = [frame, exporting, input, follower](const spectrum::Binding& binding,
                                                          std::vector<spectrum::Bands>& history, std::string& failure) {
      Input read;
      return input(binding, read, failure)
          && spectrum::evaluateHistory(binding, read.current, frame->timeSeconds, read.featuresAt,
                                       frame->liveFrameAvailable, exporting, follower(binding), history, failure,
                                       frame->seekGeneration, frame->loopGeneration, read.scalars);
    };
    evaluation.spectrumAt = bands;
    evaluation.spectrumHistoryAt = rows;
    evaluation.audioDeformAt = [frame, bands, rows](wire::GeometryData& mesh, const spectrum::Binding& binding,
                                                    std::string& failure) {
      if (binding.history.enabled) {
        std::vector<spectrum::Bands> history;
        if (!rows(binding, history, failure)) return false;
        if (history.size() != binding.history.depth || binding.history.coordinates.size() != mesh.positions.size()) {
          failure = "Spectral History Audio Deformer rows disagree with its retained vertices"; return false;
        }
        std::vector<double> responses(mesh.positions.size());
        for (std::size_t vertex = 0; vertex < responses.size(); ++vertex)
          responses[vertex] = spectrum::deformerHistoryResponse(binding, history, vertex);
        return wire::applyAudioDeformerResponses(mesh, binding, responses, frame->timeSeconds, failure);
      }
      spectrum::Bands followed{};
      return bands(binding, followed, failure)
          && wire::applyAudioDeformer(mesh, binding, followed, frame->timeSeconds, failure);
    };
    // Set Position displacements without their own Audio Spectrum Field operation.
    std::vector<spectrum::Binding> displacements;
    for (const auto& binding : value.spectrumFields)
      if (wire::adapterSpectrumDisplacement(value, binding)) displacements.push_back(binding);
    evaluation.displaceAdapters = !displacements.empty();
    evaluation.displaceConsumer = [displacements, bands, rows](wire::GeometryData& mesh, wire::StableId consumer,
                                                               std::string& failure) {
      for (const auto& binding : displacements) {
        if (binding.consumerGeometryStableId != consumer) continue;
        std::vector<float> amounts(mesh.positions.size());
        if (binding.history.enabled) {
          std::vector<spectrum::Bands> history;
          if (!rows(binding, history, failure)) return false;
          for (std::size_t vertex = 0; vertex < amounts.size(); ++vertex)
            amounts[vertex] = spectrum::sampleHistory(binding, history, vertex);
        } else {
          if (binding.coordinates.size() != amounts.size()) {
            failure = "Audio Spectrum Field coordinates disagree with its Set Position mesh"; return false;
          }
          spectrum::Bands followed{};
          if (!bands(binding, followed, failure)) return false;
          for (std::size_t vertex = 0; vertex < amounts.size(); ++vertex)
            amounts[vertex] = spectrum::sample(binding, followed, binding.coordinates[vertex]);
        }
        for (std::size_t vertex = 0; vertex < amounts.size(); ++vertex) {
          auto& position = mesh.positions[vertex];
          position.x += amounts[vertex] * binding.direction[0];
          position.y += amounts[vertex] * binding.direction[1];
          position.z += amounts[vertex] * binding.direction[2];
        }
      }
      return true;
    };
    error.clear();
    return true;
  };
}

} // namespace videohelper::geometry
