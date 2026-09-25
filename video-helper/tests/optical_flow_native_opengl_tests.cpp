#include "../src/gl_loader.h"
#include "../src/optical_flow_native_executor.h"
#include <GLFW/glfw3.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
namespace {
constexpr uint32_t W = 12, H = 10;
constexpr uint64_t REV = 23, GEN = 7;
videoopticalflow::Description desc() {
  videoopticalflow::Description d;
  d.first.identity.stream[0] = 1;
  d.second.identity.stream = d.first.identity.stream;
  d.first.identity.content[0] = 2;
  d.second.identity.content[0] = 3;
  d.first.identity.frameIndex = 20;
  d.second.identity.frameIndex = 21;
  d.first.timestamp = {20, 30};
  d.second.timestamp = {21, 30};
  d.first.extent = {W, H};
  d.second.extent = d.first.extent;
  d.output = videoopticalflow::canonicalOutput(d.first.extent);
  return d;
}
struct Source final : videoopticalflow::NativeSourceFrame {
  videoopticalflow::FrameDescription d;
  videoopticalflow::ResourceIdentity id{};
  videoopticalflow::BackendIdentity impl{};
  uint32_t implRev = 0;
  unsigned tex = 0;
  ~Source() {
    if (tex)
      glDeleteTextures(1, &tex);
  }
  const videoopticalflow::FrameDescription &frame() const noexcept override {
    return d;
  }
  const videoopticalflow::ResourceIdentity &
  resourceIdentity() const noexcept override {
    return id;
  }
  uint64_t helperGeneration() const noexcept override { return GEN; }
  uint64_t structuralRevision() const noexcept override { return REV; }
  const videoopticalflow::BackendIdentity &
  backendImplementation() const noexcept override {
    return impl;
  }
  uint32_t backendImplementationRevision() const noexcept override {
    return implRev;
  }
  uint64_t byteCount() const noexcept override { return uint64_t(W) * H * 8; }
  uintptr_t imageHandle() const noexcept override { return tex; }
  uintptr_t textureViewHandle() const noexcept override { return tex; }
};
std::shared_ptr<Source> source(const videoopticalflow::FrameDescription &d,
                               int shift,
                               const videoopticalflow::BackendCapabilities &c,
                               uint8_t id) {
  auto s = std::make_shared<Source>();
  s->d = d;
  s->id[0] = id;
  s->impl = c.implementation;
  s->implRev = c.implementationRevision;
  std::vector<float> p(W * H * 4, 0);
  for (size_t i = 0; i < W * H; i++)
    p[i * 4 + 3] = 1;
  for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++) {
      int xx = 5 + x + shift, yy = 5 + y;
      auto o = (yy * W + xx) * 4;
      p[o] = (x + 2) * .25f;
      p[o + 1] = (y + 2) * .25f;
      p[o + 2] = ((x + 1) * 3 + (y + 1) + 1) * .0625f;
    }
  glGenTextures(1, &s->tex);
  glBindTexture(GL_TEXTURE_2D, s->tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, W, H, 0, GL_RGBA, GL_FLOAT,
               p.data());
  return s;
}
bool read(unsigned tex, int x, int y, float &dx, float &dy) {
  unsigned f = 0;
  arbitgl::GlFuncs gl;
  std::string e;
  if (!arbitgl::loadGlFunctions(gl, e))
    return false;
  gl.GenFramebuffers(1, &f);
  gl.BindFramebuffer(GL_FRAMEBUFFER, f);
  gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          tex, 0);
  const bool complete = gl.CheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
  float v[2] = {};
  if (complete)
    glReadPixels(x, y, 1, 1, GL_RG, GL_FLOAT, v);
  const bool succeeded = complete && glGetError() == GL_NO_ERROR;
  gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
  gl.DeleteFramebuffers(1, &f);
  dx = v[0];
  dy = v[1];
  return succeeded && std::isfinite(dx) && std::isfinite(dy);
}
} // namespace
int main() {
  if (!glfwInit())
    return 77;
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  auto *w = glfwCreateWindow(64, 64, "flow", nullptr, nullptr);
  if (!w) {
    glfwTerminate();
    return 77;
  }
  glfwMakeContextCurrent(w);
  auto &b = arbitgpu::nativeOpticalFlowExecutionBackend();
  auto caps = b.opticalFlowCapabilities();
  if (!caps.supportsOpticalFlow)
    return 77;
  auto d = desc();
  videoopticalflow::AdmissionFailure af{};
  auto a = videoopticalflow::admit(d, caps, af);
  if (!a)
    return 1;
  auto first = source(a->first(), 0, caps, 0x31),
       second = source(a->second(), 2, caps, 0x32);
  opticalflowoperation::Payload p;
  p.extent = a->output().extent;
  auto xml = opticalflowoperation::serialize(p);
  videoopticalflow::LoweredOperation op{
      opticalflowoperation::kOperationKind,
      opticalflowoperation::kBackendCapability, xml};
  videoopticalflow::EvaluationContext pc;
  pc.structuralRevision = REV;
  pc.helperGeneration = GEN;
  pc.mode = videoopticalflow::EvaluationMode::Preview;
  auto ec = pc;
  ec.mode = videoopticalflow::EvaluationMode::OfflineSequential;
  videoopticalflow::NativeExecutor preview(
      videoopticalflow::EvaluationMode::Preview, b),
      exporter(videoopticalflow::EvaluationMode::OfflineSequential, b);
  std::string e;
  videoopticalflow::NativeExecutionFailure f{};
  arbitgl::GlFuncs gl;
  if (!arbitgl::loadGlFunctions(gl, e)) return 9;
  unsigned sentinelFbo = 0;
  gl.GenFramebuffers(1, &sentinelFbo);
  gl.BindFramebuffer(GL_FRAMEBUFFER, sentinelFbo);
  gl.ActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, first->tex);
  gl.ActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, second->tex);
  gl.ActiveTexture(GL_TEXTURE3);
  glViewport(3, 4, 7, 8);
  glEnable(GL_BLEND);
  glEnable(GL_SCISSOR_TEST);
  glScissor(0, 0, 0, 0);
  glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  if (!preview.execute(op, *a, first, second, pc, e, &f) ||
      !exporter.execute(op, *a, first, second, ec, e, &f))
    return 2;
  int framebuffer = 0, activeTexture = 0, viewport[4] {}, polygon[2] {};
  GLboolean mask[4] {};
  glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &framebuffer);
  glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTexture);
  glGetIntegerv(GL_VIEWPORT, viewport);
  glGetIntegerv(GL_POLYGON_MODE, polygon);
  glGetBooleanv(GL_COLOR_WRITEMASK, mask);
  if (framebuffer != static_cast<int>(sentinelFbo) || activeTexture != GL_TEXTURE3
      || viewport[0] != 3 || viewport[1] != 4 || viewport[2] != 7 || viewport[3] != 8
      || polygon[0] != GL_LINE || mask[0] || mask[1] || mask[2] || mask[3]
      || !glIsEnabled(GL_BLEND) || !glIsEnabled(GL_SCISSOR_TEST)) return 10;
  glDisable(GL_BLEND);
  glDisable(GL_SCISSOR_TEST);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
  gl.DeleteFramebuffers(1, &sentinelFbo);
  float dx = 0, dy = 0;
  if (!read((unsigned)preview.publication()->imageHandle(), 5, 5, dx, dy) ||
      std::abs(dx - 2) > 0.01 || std::abs(dy) > 0.01)
    return 3;
  if (!read(static_cast<unsigned>(exporter.publication()->imageHandle()), 5, 5, dx, dy)
      || std::abs(dx - 2) > 0.01 || std::abs(dy) > 0.01) return 11;
  auto retained = preview.publication();
  const auto retainedTexture = static_cast<unsigned>(retained->imageHandle());
  preview.reset();
  if (!glIsTexture(retainedTexture)) return 12;
  retained.reset();
  if (glIsTexture(retainedTexture)) return 13;
  auto stationary = source(a->second(), 0, caps, 0x33);
  auto d2 = d;
  d2.second.identity.content[0] = 4;
  videoopticalflow::AdmissionFailure af2{};
  auto a2 = videoopticalflow::admit(d2, caps, af2);
  if (!a2)
    return 4;
  stationary->d = a2->second();
  if (!preview.execute(op, *a2, first, stationary, pc, e, &f) ||
      !read((unsigned)preview.publication()->imageHandle(), 5, 5, dx, dy) ||
      std::abs(dx) > 0.01 || std::abs(dy) > 0.01)
    return 5;
  exporter.reset();
  if (!exporter.execute(op, *a2, first, stationary, ec, e, &f) ||
      !read(static_cast<unsigned>(exporter.publication()->imageHandle()), 5, 5, dx, dy) ||
      std::abs(dx) > 0.01 || std::abs(dy) > 0.01)
    return 20;
  auto alias = second;
  const auto secondTexture = alias->tex;
  alias->tex = first->tex;
  if (preview.execute(op, *a, first, alias, pc, e, &f) ||
      f != videoopticalflow::NativeExecutionFailure::DuplicateSourceResource)
    return 6;
  alias->tex = secondTexture;
  alias.reset();

  // Released publications must free textures during sustained use, without
  // waiting for the window/context to close.
  for (int cycle = 0; cycle < 32; ++cycle) {
    preview.reset();
    if (!preview.execute(op, *a, first, second, pc, e, &f)) return 14;
    const auto texture = static_cast<unsigned>(preview.publication()->imageHandle());
    preview.reset();
    if (glIsTexture(texture)) return 15;
  }
  if (!preview.execute(op, *a, first, second, pc, e, &f)) return 16;
  const auto deferredTexture = static_cast<unsigned>(exporter.publication()->imageHandle());
  auto *w2 = glfwCreateWindow(64, 64, "flow2", nullptr, nullptr);
  if (!w2) return 7;
  glfwMakeContextCurrent(w2);
  exporter.reset(); // A foreign thread/context must not steal the live owner.
  if (glfwGetCurrentContext() != w2) return 17;
  glfwMakeContextCurrent(w);
  if (!glIsTexture(deferredTexture)) return 18;
  first.reset();
  second.reset();
  stationary.reset();
  arbitgpu::invalidateNativeOpticalFlowExecutionContext(
      reinterpret_cast<uintptr_t>(w));
  if (glIsTexture(deferredTexture)) return 19;
  glfwDestroyWindow(w);
  preview.reset();
  glfwMakeContextCurrent(w2);
  if (!b.opticalFlowCapabilities().supportsOpticalFlow)
    return 8;
  arbitgpu::invalidateNativeOpticalFlowExecutionContext(
      reinterpret_cast<uintptr_t>(w2));
  glfwDestroyWindow(w2);
  glfwTerminate();
  std::printf(
      "optical-flow-opengl PASS translated=2,0 stationary=0,0 "
      "preview_export=yes finite_readback=yes invalid=yes state_restored=yes sustained_release=yes "
      "deferred_release=yes retired_release=yes recreate=yes\n");
  return 0;
}
