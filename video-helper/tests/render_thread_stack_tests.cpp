// RenderThread gives render and export workers an 8 MiB stack on every POSIX
// host. The 15 Oct macOS SIGBUS came from a 512 KiB default; this runs more
// than that much stack on a RenderThread and checks the requested size held.

#include "render_thread.h"

#include <atomic>
#include <cstdio>
#include <cstring>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
  if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}

// Each level holds 64 KiB, so 16 levels need 1 MiB: twice the macOS default.
__attribute__((noinline)) unsigned deepFrames(unsigned depth, volatile unsigned char* sink) {
  volatile unsigned char frame[64 * 1024];
  std::memset(const_cast<unsigned char*>(frame), static_cast<int>(depth), sizeof(frame));
  *sink = static_cast<unsigned char>(*sink + frame[depth * 97u % sizeof(frame)]);
  return depth == 0 ? frame[0] : frame[1] + deepFrames(depth - 1, sink);
}

void deepStackRunsOnRenderThread() {
  std::atomic<bool> finished {false};
  volatile unsigned char sink = 0;
  unsigned total = 0;
  videohelper::RenderThread thread([&] {
    total = deepFrames(16, &sink);
    finished = true;
  });
  check(thread.joinable(), "a started RenderThread is joinable");
  thread.join();
  check(!thread.joinable(), "a joined RenderThread is not joinable");
  check(finished.load(), "1 MiB of nested frames completes on a RenderThread");
  check(total == 136u, "nested frames return their depth sum");
}

#if defined(__linux__)
void requestedStackSizeHolds() {
  std::size_t stackBytes = 0;
  videohelper::RenderThread thread([&] {
    pthread_attr_t attributes;
    if (pthread_getattr_np(pthread_self(), &attributes) == 0) {
      pthread_attr_getstacksize(&attributes, &stackBytes);
      pthread_attr_destroy(&attributes);
    }
  });
  thread.join();
  check(stackBytes >= videohelper::kRenderThreadStackBytes, "RenderThread stack is at least 8 MiB");
}
#endif

void moveAssignmentTransfersOwnership() {
  std::atomic<int> runs {0};
  videohelper::RenderThread empty;
  check(!empty.joinable(), "a default RenderThread is not joinable");
  empty = videohelper::RenderThread([&] { ++runs; });
  check(empty.joinable(), "move assignment transfers the running thread");
  empty.join();
  check(runs.load() == 1, "the moved body runs once");
}

} // namespace

int main() {
  deepStackRunsOnRenderThread();
#if defined(__linux__)
  requestedStackSizeHolds();
#endif
  moveAssignmentTransfersOwnership();
  if (failures != 0) return 1;
  std::puts("PASS: RenderThread runs deep frames on an explicit 8 MiB stack");
  return 0;
}
