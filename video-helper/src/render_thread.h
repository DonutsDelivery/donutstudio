#pragma once

// A std::thread stand-in for render and export workers with an explicit stack.
// macOS gives secondary threads 512 KiB, which a viewport frame that replays
// particles over imported deformation and a caustic fixture render exhausted on
// 15 Oct. Linux already defaults to 8 MiB, so this makes macOS match it.

#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>

#if !defined(_WIN32)
#include <pthread.h>
#endif

namespace videohelper {

inline constexpr std::size_t kRenderThreadStackBytes = 8u * 1024u * 1024u;

class RenderThread final {
public:
  RenderThread() noexcept = default;

  template <typename Body, typename = std::enable_if_t<
      !std::is_same<std::decay_t<Body>, RenderThread>::value>>
  explicit RenderThread(Body&& body, std::size_t stackBytes = kRenderThreadStackBytes) {
    start(std::function<void()>(std::forward<Body>(body)), stackBytes);
  }

  RenderThread(RenderThread&& other) noexcept { swap(other); }
  RenderThread& operator=(RenderThread&& other) noexcept {
    if (joinable()) std::terminate();
    swap(other);
    return *this;
  }
  RenderThread(const RenderThread&) = delete;
  RenderThread& operator=(const RenderThread&) = delete;
  ~RenderThread() { if (joinable()) std::terminate(); }

#if defined(_WIN32)
  bool joinable() const noexcept { return thread_.joinable(); }
  void join() { thread_.join(); }

private:
  void start(std::function<void()> body, std::size_t) { thread_ = std::thread(std::move(body)); }
  void swap(RenderThread& other) noexcept { std::swap(thread_, other.thread_); }
  std::thread thread_;
#else
  bool joinable() const noexcept { return started_; }
  void join() {
    if (!started_) throw std::system_error(std::make_error_code(std::errc::invalid_argument));
    const int result = pthread_join(thread_, nullptr);
    if (result != 0) throw std::system_error(result, std::generic_category());
    started_ = false;
  }

private:
  static void* run(void* argument) noexcept {
    // An escaping exception terminates, as it does on std::thread.
    std::unique_ptr<std::function<void()>> body(static_cast<std::function<void()>*>(argument));
    (*body)();
    return nullptr;
  }

  void start(std::function<void()> body, std::size_t stackBytes) {
    auto owned = std::make_unique<std::function<void()>>(std::move(body));
    pthread_attr_t attributes;
    int result = pthread_attr_init(&attributes);
    if (result != 0) throw std::system_error(result, std::generic_category());
    result = pthread_attr_setstacksize(&attributes, stackBytes);
    if (result == 0) result = pthread_create(&thread_, &attributes, &RenderThread::run, owned.get());
    pthread_attr_destroy(&attributes);
    if (result != 0) throw std::system_error(result, std::generic_category());
    owned.release();
    started_ = true;
  }

  void swap(RenderThread& other) noexcept {
    std::swap(thread_, other.thread_);
    std::swap(started_, other.started_);
  }

  pthread_t thread_ {};
  bool started_ = false;
#endif
};

} // namespace videohelper
