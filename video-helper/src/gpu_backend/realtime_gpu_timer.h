#pragma once

// GPU time of one Realtime frame on the OpenGL backend: one GL_TIMESTAMP query
// before the frame's submission and one after it, the counters the grating
// submission clock in backend_opengl.cpp uses, without its env var. Each call
// works on the current thread's context. A Preview keeps a small ring of query
// pairs, so it never waits on the GPU for its own time.
//
// Only the OpenGL backend defines these (backend_opengl.cpp).

#include <optional>

namespace arbitgpu
{

// Opens the frame's span. False when the context has no timer queries. blocking
// takes a pair of its own for realtimeGpuTimerResult (true), deleted once read.
bool realtimeGpuTimerBegin (bool blocking) noexcept;
// Closes the span opened by realtimeGpuTimerBegin.
void realtimeGpuTimerEnd() noexcept;
// waitForResult: the span just closed, waiting for the GPU to reach it. The
// export path reads the frame's pixels back right after, so that wait would come
// there anyway. Otherwise the newest earlier span whose result is ready, which
// lags the frame on screen by one to three frames, or nothing yet.
std::optional<double> realtimeGpuTimerResult (bool waitForResult) noexcept;

} // namespace arbitgpu
