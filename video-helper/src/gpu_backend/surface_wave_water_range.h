#pragma once

// The water a GPU wave frame may move when no glass query holds it: Realtime
// prepares no smooth glass, so there is no water BLAS to refit, and the wave pass
// only rewrites the water's vertex range. No GL here, so the admission is tested
// without a context (video-helper/tests/surface_wave_hold_test.cpp).

#include "../../../shared/Visual3DScene.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace arbitgpu
{

struct SurfaceWaveWaterRange final
{
    std::size_t object = 0;
    std::uint32_t firstVertex = 0, vertexCount = 0;
};

// Null with the range filled in when the water may move, else why it may not.
// opaqueObjectMesh is the opaque visibility query's mesh per object (empty when
// there is no query): a water in it would cast shadows from its prepared heights.
inline const char* surfaceWaveWaterRange (const HarmonicMIDI::grid::Visual3DScene& scene,
                                          HarmonicMIDI::grid::SceneObjectId water,
                                          std::uint32_t columns, std::uint32_t rows,
                                          const std::vector<int>& opaqueObjectMesh,
                                          SurfaceWaveWaterRange& range)
{
    std::size_t found = scene.objectCount;
    for (std::size_t o = 0; o < scene.objectCount; ++o)
        if (scene.objects[o].id == water)
        {
            if (found != scene.objectCount) return "the water object id is not unique";
            found = o;
        }
    if (found == scene.objectCount) return "the water object is missing";
    const auto& object = scene.objects[found];
    if (columns < 2 || rows < 2 || std::uint64_t (columns) * rows != object.vertexCount)
        return "the water is not the frame's grid";
    if (object.firstVertex > scene.vertexCount || object.vertexCount > scene.vertexCount - object.firstVertex)
        return "the water vertex range is out of bounds";
    if (found < opaqueObjectMesh.size() && opaqueObjectMesh[found] >= 0)
        return "the water is in the opaque visibility query";
    for (std::size_t o = 0; o < scene.objectCount; ++o)
    {
        if (o == found) continue;
        const auto& other = scene.objects[o];
        if (other.vertexCount != 0 && other.firstVertex < object.firstVertex + object.vertexCount
            && object.firstVertex < other.firstVertex + other.vertexCount)
            return "another object shares the water vertex range";
    }
    range = { found, object.firstVertex, object.vertexCount };
    return nullptr;
}

} // namespace arbitgpu
