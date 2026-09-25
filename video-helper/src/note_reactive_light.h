#pragma once

#include "canonical_block_c_frame.h"
#include "../../shared/VisualImportedSceneRenderOperationContract.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace videohelper
{
// Produce one per-frame light override. The scene, authored light and canonical
// score frame remain immutable, and the native GL/Metal draw consumes the same value.
inline bool applyNoteReactiveLight(
    HarmonicMIDI::grid::SceneLightRecord& light,
    const visualimportedscenerender::NoteLight& mapping,
    const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>& frame)
{
    if (!canonicalblockc::valid(frame)) return false;
    const auto& data = frame->noteTextureValues;
    std::array<int, arbitblockc::kMaxNotes> rows {};
    const int count = frame->noteRows();
    for (int i = 0; i < count; ++i) rows[static_cast<std::size_t>(i)] = i;
    std::sort(rows.begin(), rows.begin() + count, [&frame](int a, int b)
        { return frame->noteIdentities[static_cast<std::size_t>(a)]
               < frame->noteIdentities[static_cast<std::size_t>(b)]; });
    double velocity = 0, pitch = 0;
    int sounding = 0;
    for (int index = 0; index < count; ++index)
    {
        const auto offset = static_cast<std::size_t>(rows[static_cast<std::size_t>(index)] * 16);
        if (data[offset + 6] != static_cast<float>(mapping.track)
            || data[offset + 1] <= 0.001f || data[offset + 2] < 0 || data[offset + 3] <= 0)
            continue;
        const double strength = std::clamp(data[offset + 1], 0.0f, 1.0f);
        velocity += strength;
        pitch += std::clamp((data[offset] - 60.0f) / 36.0f, -1.0f, 1.0f) * strength;
        ++sounding;
    }
    light.intensity = static_cast<float>(std::min(1000000.0,
        static_cast<double>(light.intensity) * mapping.gain
            * (sounding != 0 ? velocity / sounding : 0.0)));
    const double radians = velocity > 0
        ? pitch / velocity * mapping.pitchDegrees * 0.017453292519943295 : 0;
    const float s = static_cast<float>(std::sin(radians * 0.5));
    const float c = static_cast<float>(std::cos(radians * 0.5));
    const auto q = light.transform.rotation;
    light.transform.rotation = { c * q.x + s * q.z, c * q.y + s * q.w,
                                 c * q.z - s * q.x, c * q.w - s * q.y };
    return true;
}
}
