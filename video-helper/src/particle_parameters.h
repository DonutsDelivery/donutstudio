#pragma once

#include <algorithm>
#include <cmath>
#include "canonical_block_c_frame.h"

namespace videorender
{
struct ParticleHistorySource;
struct ParticleGeometryBinding;
struct ParticleSolidState;
using ParticleBodyState = std::array<std::array<float, 4>, 64>;
struct ParticleParams
{
    int count = 512, spawnTrack = 0, seed = 0;
    float size = 2.0f, gravity = 0.0f, force = 1.0f, lifetime = 0.0f;
    float red = -1.0f, green = -1.0f, blue = -1.0f, alpha = 1.0f;
    // Mode 0 retains accumulated legacy motion. Mode 1 evaluates a damped
    // trajectory at clip time, including while paused or after a seek/loop.
    int motionMode = 0;
    float drag = 0.0f, attraction = 0.0f, resetTime = 0.0f;
    float linkSpring = 0.0f, linkRatioInfluence = 1.0f;
    int collisionMode = 0;
    float bodyRadius = 0.015f, restitution = 0.7f;
    // Coupled replay is bounded to 64 bodies. Existing documents
    // retain snapshot inputs; historicalReplay reads canonical project inputs.
    int bodyShape = 0, linkConstraint = 0, resetMode = 0;
    float bodyMass = 1.0f, friction = 0.0f, linkRestScale = 1.0f;
    float impulseX = 0.0f, impulseY = 0.0f;
    // Opt-in; absent in saved graphs means the unchanged planar solver.
    int simulationSpace = 0;
    float impulseZ = 0, gravityX = 0, gravityZ = 0, spawnZ = 0.5f;
    std::array<float, 3> orientationDegrees {}, angularVelocity {};
    float angularDrag = 0;
    // Negative rate preserves the original immediate source population.
    float emissionRate = -1, reset = 0;
    int historyClipId = -1, historyNodeId = 0;
    bool historicalReplay = false;
    double historyProjectSeconds = 0.0;
    std::shared_ptr<const ParticleHistorySource> history;
    std::shared_ptr<const ParticleGeometryBinding> geometryBinding;
    std::shared_ptr<const ParticleBodyState> preparedBodies;
    std::shared_ptr<const ParticleSolidState> preparedSolids;
    std::uint64_t geometryRevision = 0;
    float colliderThickness = 0.005f;
    float rmsGain = 0.0f, onsetGain = 0.0f;
    float rms = 0.0f, onset = 0.0f, onsetAge = 0.0f;
    // XY emitter/attractor coordinates, projected only in planar mode. Solid
    // mode retains both world Z coordinates separately. Not score rows.
    int geometryCount = 0;
    std::array<std::array<float,4>,64> geometryAnchors {};
    std::array<std::uint64_t,64> geometryIdentities {};
    std::array<std::array<float,2>,64> geometryDepths {};
};

inline std::array<float, canonicalblockc::kNoteTextureValues> particleNoteUpload(
    const canonicalblockc::CanonicalBlockCFrame* notes, bool timelineMotion,
    float linkRatioInfluence = 1.0f)
{
    std::array<float, canonicalblockc::kNoteTextureValues> data {};
    if (notes == nullptr) return data;
    if (!timelineMotion) return notes->noteTextureValues;
    // Residency rows can differ after a seek. Order this private GPU upload by
    // the canonical signed note IDs so a frame does not depend on prior packing.
    std::array<int, arbitblockc::kMaxNotes> rows {};
    const int count = std::clamp(notes->noteRows(), 0, arbitblockc::kMaxNotes);
    for (int i = 0; i < count; ++i) rows[static_cast<std::size_t>(i)] = i;
    std::sort(rows.begin(), rows.begin() + count, [notes](int a, int b)
    { return notes->noteIdentities[static_cast<std::size_t>(a)]
           < notes->noteIdentities[static_cast<std::size_t>(b)]; });
    std::array<int, arbitblockc::kMaxNotes> sortedRows {};
    for (int i = 0; i < count; ++i)
    {
        std::copy_n(notes->noteTextureValues.begin() + rows[static_cast<std::size_t>(i)] * 16,
                    16, data.begin() + i * 16);
        sortedRows[static_cast<std::size_t>(rows[static_cast<std::size_t>(i)])] = i;
        // Texel 2 is private particle data in timeline mode. Canonical Block C
        // remains immutable; its prime exponents are not used by this shader.
        std::fill_n(data.begin() + i * 16 + 8, 4, 0.0f);
    }
    std::array<int, arbitblockc::kMaxLinks> links {};
    const int linkCount = std::clamp(notes->linkRows(), 0, arbitblockc::kMaxLinks);
    for (int i = 0; i < linkCount; ++i) links[static_cast<std::size_t>(i)] = i;
    std::sort(links.begin(), links.begin() + linkCount, [notes](int a, int b)
    { return notes->linkIdentities[static_cast<std::size_t>(a)]
           < notes->linkIdentities[static_cast<std::size_t>(b)]; });
    const float influence = std::isfinite(linkRatioInfluence)
        ? std::clamp(linkRatioInfluence, 0.0f, 1.0f) : 1.0f;
    const auto sounding = [&data](int row)
    {
        const auto offset = static_cast<std::size_t>(row * 16);
        return data[offset + 1] > 0.001f && data[offset + 2] >= 0.0f
            && data[offset + 3] > 0.0f;
    };
    const auto anchorX = [&data](int row)
    { return std::clamp((data[static_cast<std::size_t>(row * 16)] - 36.0f) / 60.0f,
                        0.0f, 1.0f) * 0.8f + 0.1f; };
    for (int index = 0; index < linkCount; ++index)
    {
        const int row = links[static_cast<std::size_t>(index)];
        if (notes->linkIdentities[static_cast<std::size_t>(row)] == 0) continue;
        const auto offset = static_cast<std::size_t>(row * 4);
        const auto& values = notes->linkTextureValues;
        const float source = values[offset], target = values[offset + 1];
        const float numerator = values[offset + 2], denominator = values[offset + 3];
        if (!std::isfinite(source) || !std::isfinite(target)
            || source < 0 || source >= count || target < 0 || target >= count
            || std::floor(source) != source || std::floor(target) != target
            || !std::isfinite(numerator) || !std::isfinite(denominator)
            || numerator <= 0 || denominator <= 0) continue;
        const int a = sortedRows[static_cast<std::size_t>(source)];
        const int b = sortedRows[static_cast<std::size_t>(target)];
        if (a == b || !sounding(a) || !sounding(b)) continue;
        const float ratio = std::clamp(std::max(numerator / denominator,
                                                denominator / numerator), 1.0f, 8.0f);
        const float weight = 1.0f + influence * (ratio - 1.0f);
        const float displacement = anchorX(b) - anchorX(a);
        for (int endpoint = 0; endpoint < 2; ++endpoint)
        {
            const auto field = static_cast<std::size_t>((endpoint == 0 ? a : b) * 16 + 8);
            data[field] += (endpoint == 0 ? displacement : -displacement) * weight;
            data[field + 2] += weight;
            data[field + 3] += 1.0f;
        }
    }
    // Average incident springs so dense link graphs keep stiffness <= 8 times
    // the authored strength. Stable link-ID order also fixes summation order.
    for (int row = 0; row < count; ++row)
    {
        const auto field = static_cast<std::size_t>(row * 16 + 8);
        if (data[field + 3] > 0.0f)
        {
            data[field] /= data[field + 3];
            data[field + 2] /= data[field + 3];
        }
    }
    return data;
}

// Both native compositors consume the same by-value graph parameters. The map
// is only the compatibility path for legacy generator clips.
template <typename Layer>
ParticleParams particleParamsForLayer(const Layer& layer)
{
    ParticleParams params;
    if (layer.particleNodeId != 0)
        params = layer.particleParameters;
    else
    {
        const auto value = [&layer](const char* key, double fallback)
        {
            const auto item = layer.genParams.find(key);
            return item != layer.genParams.end() && std::isfinite(item->second)
                ? item->second : fallback;
        };
        params.count = static_cast<int>(std::clamp(value("count", 512.0), 1.0, 100000.0) + 0.5);
        params.spawnTrack = static_cast<int>(std::clamp(value("spawnTrack", 0.0), 0.0, 65535.0) + 0.5);
        params.size = static_cast<float>(value("size", 2.0));
        params.gravity = static_cast<float>(value("gravity", 0.0));
        params.force = static_cast<float>(value("force", 1.0));
        params.seed = static_cast<int>(std::clamp(value("seed", 0.0), 0.0, 65535.0));
        params.lifetime = static_cast<float>(value("lifetime", 0.0));
        if (value("nativeBuiltin", 0.0) >= 0.5)
        {
            params.force = static_cast<float>(value("speed", 1.0));
            params.red = static_cast<float>(value("red", 0.2));
            params.green = static_cast<float>(value("green", 0.7));
            params.blue = static_cast<float>(value("blue", 1.0));
            params.alpha = static_cast<float>(value("alpha", 1.0));
        }
    }
    params.history = layer.particleHistory;
    params.historyProjectSeconds = layer.particleProjectSeconds;
    if (layer.audioPresent)
    {
        const auto unit = [](float value)
        { return std::isfinite(value) ? std::clamp(value, 0.0f, 1.0f) : 0.0f; };
        params.rms = unit(layer.audioFeatures.rms);
        params.onset = unit(layer.audioFeatures.onset);
        params.onsetAge = std::isfinite(layer.audioFeatures.onsetAge)
            ? std::max(0.0f, layer.audioFeatures.onsetAge) : 0.0f;
    }
    return params;
}
}
