#pragma once

// Pool sizes for the process-wide sokol_gfx Metal context. This header has no
// Sokol or Metal dependency so the arithmetic is testable on Linux.

#include <cstddef>

namespace arbitgpu::metalpools
{
// The viewport, frame probe, export and Preview warmup can retain compositors
// and fixture preparations at the same time. A fixture preparation of a
// 16-texture imported scene holds 33 images, 33 views and 33 samplers, plus an
// environment image and view. Sokol's defaults (128 images, 256 views, 64
// samplers) leave room for one such scene next to the compositors.
inline constexpr int kBufferPool = 512;
inline constexpr int kImagePool = 512;
inline constexpr int kSamplerPool = 256;
inline constexpr int kShaderPool = 128;
inline constexpr int kPipelinePool = 128;
inline constexpr int kViewPool = 1024;

// Mirrors _sg_mtl_init_pool() in sokol_gfx.h. Every live resource can hold the
// per-type id count, and the factor 2 leaves the same again for resources that
// wait in the deferred release queue until sg_commit() collects them.
constexpr int idPoolSlots (int buffers, int images, int samplers, int shaders,
                           int pipelines, int views) noexcept
{
    return 2 * (2 * buffers + 2 * images + samplers + 6 * shaders + 3 * pipelines + views + 128);
}
inline constexpr int kIdPoolSlots = idPoolSlots (kBufferPool, kImagePool, kSamplerPool,
                                                 kShaderPool, kPipelinePool, kViewPool);

// Upper bound on the Metal ids one fixture preparation creates: vertex and index
// buffers (one id each), the double-buffered instance buffer (two), an image,
// view and sampler for the fallback slot and for both roles of every texture,
// the environment image and view, and the shader and pipelines with their
// receipt and depth variants.
constexpr int fixturePreparationIds (std::size_t textures) noexcept
{
    return 4 + 3 * (1 + 2 * static_cast<int> (textures)) + 2 + 64;
}

// Before a preparation or a compositor initialization creates resources, the
// backend makes sure at least this many id slots are free. Below it, pending
// releases are collected after a queue fence, and if that is not enough the
// creation is refused by name instead of reaching Sokol.
inline constexpr int kMinimumFreeIds = 256;
} // namespace arbitgpu::metalpools
