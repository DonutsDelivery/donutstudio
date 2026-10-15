#include "../src/gl_loader.h"
#include "optical_flow_native_test_fixture.h"
#include "../src/optical_flow_native_executor.h"
#include "../src/optical_flow_native_source_upload.h"
#include <GLFW/glfw3.h>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <thread>
#include <vector>
namespace {
constexpr uint32_t W = opticalflowtest::width, H = opticalflowtest::height;
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
  uintptr_t nativeContextIdentity() const noexcept override {
    return reinterpret_cast<uintptr_t>(glfwGetCurrentContext());
  }
};
std::shared_ptr<Source> source(const videoopticalflow::FrameDescription &d,
                               int shift,
                               const videoopticalflow::BackendCapabilities &c,
                               uint8_t id, int verticalShift = 0) {
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
      int xx = 5 + x + shift, yy = 5 + y + verticalShift;
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
std::shared_ptr<Source> authoredSource(const videoopticalflow::FrameDescription& d,
                                      const std::vector<float>& pixels,
                                      const videoopticalflow::BackendCapabilities& caps,
                                      uint8_t identity) {
  auto s = std::make_shared<Source>();
  s->d = d;
  s->id[0] = identity;
  s->impl = caps.implementation;
  s->implRev = caps.implementationRevision;
  glGenTextures(1, &s->tex);
  glBindTexture(GL_TEXTURE_2D, s->tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, W, H, 0, GL_RGBA, GL_FLOAT, pixels.data());
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
bool readColor(unsigned tex, int x, int y, float& red, float* green = nullptr) {
  unsigned framebuffer = 0;
  arbitgl::GlFuncs gl;
  std::string error;
  if (!arbitgl::loadGlFunctions(gl, error)) return false;
  gl.GenFramebuffers(1, &framebuffer);
  gl.BindFramebuffer(GL_FRAMEBUFFER, framebuffer);
  gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          tex, 0);
  float rgba[4] {};
  const bool complete = gl.CheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
  if (complete) glReadPixels(x, y, 1, 1, GL_RGBA, GL_FLOAT, rgba);
  const bool valid = complete && glGetError() == GL_NO_ERROR;
  gl.BindFramebuffer(GL_FRAMEBUFFER, 0);
  gl.DeleteFramebuffers(1, &framebuffer);
  red = rgba[0];
  if (green != nullptr) *green = rgba[1];
  return valid && std::isfinite(red);
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
  visualtemporalsampling::Payload blur;
  blur.mode = visualtemporalsampling::Mode::motionBlur;
  blur.sampleCount = 3;
  blur.historyWeight = 0.0f;
  blur.jitterSpread = 0.0f;
  blur.shutterAngleDegrees = 360.0f;
  blur.motionScale = 1.0f;
  auto blurred = b.renderMotionBlur(first, preview.publication(), blur, e);
  auto exportBlurred = b.renderMotionBlur(first, exporter.publication(), blur, e);
  float previewRed = 0.0f, exportRed = 0.0f;
  if (!blurred || !exportBlurred
      || !read((unsigned)preview.publication()->imageHandle(), 6, 5, dx, dy)
      || std::abs(dx - 2) > 0.01f
      || !readColor((unsigned)blurred->colorTextureViewHandle(), 6, 5, previewRed)
      || !readColor((unsigned)exportBlurred->colorTextureViewHandle(), 6, 5, exportRed)
      || std::abs(previewRed - exportRed) > 1.0f / 255.0f
      || previewRed < 0.55f || previewRed > 0.8f)
    return 36;
  auto wrongBlur = blur;
  wrongBlur.sampleCount = 0;
  if (b.renderMotionBlur(first, preview.publication(), wrongBlur, e)
      || e != "OpenGL Motion Blur requires the exact retained image and optical field")
    return 37;
  blurred.reset();
  exportBlurred.reset();
  auto checkVertical = [&](int direction) {
    auto vertical = d;
    vertical.first.identity.content[0] = direction > 0 ? 0x51 : 0x53;
    vertical.second.identity.content[0] = direction > 0 ? 0x52 : 0x54;
    vertical.first.identity.frameIndex = direction > 0 ? 30 : 40;
    vertical.second.identity.frameIndex = vertical.first.identity.frameIndex + 1;
    vertical.first.timestamp = {static_cast<int64_t>(vertical.first.identity.frameIndex), 30};
    vertical.second.timestamp = {static_cast<int64_t>(vertical.second.identity.frameIndex), 30};
    videoopticalflow::AdmissionFailure verticalFailure {};
    auto admitted = videoopticalflow::admit(vertical, caps, verticalFailure);
    if (!admitted) return false;
    auto image = source(admitted->first(), 0, caps, direction > 0 ? 0x41 : 0x43);
    auto moved = source(admitted->second(), 0, caps, direction > 0 ? 0x42 : 0x44,
                        direction);
    videoopticalflow::NativeExecutor live(videoopticalflow::EvaluationMode::Preview, b);
    videoopticalflow::NativeExecutor offline(
        videoopticalflow::EvaluationMode::OfflineSequential, b);
    if (!live.execute(op, *admitted, image, moved, pc, e, &f)
        || !offline.execute(op, *admitted, image, moved, ec, e, &f)
        || !read((unsigned)live.publication()->imageHandle(), 5, 5, dx, dy)
        || std::abs(dx) > 0.01f || std::abs(dy - direction) > 0.01f
        || !read((unsigned)offline.publication()->imageHandle(), 5, 5, dx, dy)
        || std::abs(dx) > 0.01f || std::abs(dy - direction) > 0.01f
        || !read((unsigned)live.publication()->imageHandle(), 5, 6, dx, dy)
        || std::abs(dx) > 0.01f || std::abs(dy - direction) > 0.01f
        || !read((unsigned)offline.publication()->imageHandle(), 5, 6, dx, dy)
        || std::abs(dx) > 0.01f || std::abs(dy - direction) > 0.01f)
      return false;
    auto verticalBlur = blur;
    verticalBlur.motionScale = 2.0f;
    auto liveImage = b.renderMotionBlur(image, live.publication(), verticalBlur, e);
    auto offlineImage = b.renderMotionBlur(image, offline.publication(), verticalBlur, e);
    float liveRed = 0.0f, offlineRed = 0.0f;
    float lowerGreen = 0.0f, upperGreen = 0.0f, offlineGreen = 0.0f;
    // Top-first rows 6 and 4 average to sRGB green about 0.678 and 0.537.
    return liveImage && offlineImage
        && readColor((unsigned)liveImage->colorTextureViewHandle(), 5, 6,
                     liveRed, &lowerGreen)
        && readColor((unsigned)offlineImage->colorTextureViewHandle(), 5, 6,
                     offlineRed, &offlineGreen)
        && readColor((unsigned)liveImage->colorTextureViewHandle(), 5, 4,
                     liveRed, &upperGreen)
        && lowerGreen > 0.62f && lowerGreen < 0.74f
        && upperGreen > 0.48f && upperGreen < 0.59f
        && lowerGreen > upperGreen + 0.1f
        && std::abs(lowerGreen - offlineGreen) <= 1.0f / 255.0f;
  };
  if (!checkVertical(1) || !checkVertical(-1)) return 38;
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
  for (const auto& pair : opticalflowtest::pairs()) {
    auto authoredFirst = authoredSource(a->first(), pair.first, caps, 0x81);
    auto authoredSecond = authoredSource(a->second(), pair.second, caps, 0x82);
    videoopticalflow::NativeExecutor live(videoopticalflow::EvaluationMode::Preview, b);
    videoopticalflow::NativeExecutor offline(videoopticalflow::EvaluationMode::OfflineSequential, b);
    if (!live.execute(op, *a, authoredFirst, authoredSecond, pc, e, &f)
        || !offline.execute(op, *a, authoredFirst, authoredSecond, ec, e, &f)) return 39;
    auto verify = [&](const std::shared_ptr<const videoopticalflow::PublishedNativeResult>& publication,
                      const char* mode) {
      std::vector<float> field(W * H * 2u);
      glBindTexture(GL_TEXTURE_2D, (unsigned)publication->imageHandle());
      glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_FLOAT, field.data());
      return glGetError() == GL_NO_ERROR && opticalflowtest::verify(pair, field, mode);
    };
    if (!verify(live.publication(), "opengl-preview")
        || !verify(offline.publication(), "opengl-export")) return 40;
    auto verifyViews = [&](const std::shared_ptr<const videoopticalflow::PublishedNativeResult>& publication,
                           const char* mode) {
      std::vector<float> before(W * H * 2u), after(W * H * 2u);
      glBindTexture(GL_TEXTURE_2D, (unsigned)publication->imageHandle());
      glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_FLOAT, before.data());
      if (glGetError() != GL_NO_ERROR) return false;
      const auto colorBytes = [&](const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& image,
                                  std::vector<std::uint8_t>& pixels) {
        if (!image) return false;
        pixels.resize(W * H * 4u);
        glBindTexture(GL_TEXTURE_2D, (unsigned)image->colorTextureViewHandle());
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        return glGetError() == GL_NO_ERROR;
      };
      std::vector<std::uint8_t> legacy, zero, ranged;
      if (!colorBytes(b.renderMotionView(publication, e), legacy)
          || !colorBytes(b.renderMotionView(publication, motionviewoperation::Payload {}, e), zero)
          || legacy != zero || !opticalflowtest::verifyView(pair, legacy, 0.0, mode, 1)
          || !colorBytes(b.renderMotionView(publication, motionviewoperation::Payload {
              motionviewoperation::Convention::OpticalFirstToSecondTopLeft, 16.0 }, e), ranged)
          || !opticalflowtest::verifyView(pair, ranged, 16.0, mode, 1)) return false;
      // OpenGL retains the existing dithering state; allow one Unorm code.
      if (!colorBytes(b.renderMotionView(publication, motionviewoperation::Payload {
              motionviewoperation::Convention::OpticalFirstToSecondTopLeft, 1.0 }, e), ranged)
          || !opticalflowtest::verifyView(pair, ranged, 1.0, mode, 1)) return false;
      for (const auto invalid : { -1.0, 0.5, 65537.0,
              std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() })
        if (b.renderMotionView(publication, motionviewoperation::Payload {
                motionviewoperation::Convention::OpticalFirstToSecondTopLeft, invalid }, e)) return false;
      if (b.renderMotionView(publication, motionviewoperation::Payload {
              motionviewoperation::Convention::GeometricCurrentMinusPreviousYUp, 16.0 }, e)) return false;
      glBindTexture(GL_TEXTURE_2D, (unsigned)publication->imageHandle());
      glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_FLOAT, after.data());
      return glGetError() == GL_NO_ERROR && before == after && verify(publication, mode);
    };
    if (!verifyViews(live.publication(), "opengl-preview")
        || !verifyViews(offline.publication(), "opengl-export")) return 44;
    auto stale = pc;
    ++stale.structuralRevision;
    const auto retainedHandle = live.publication()->imageHandle();
    if (live.execute(op, *a, authoredFirst, authoredSecond, stale, e, &f)
        || f != videoopticalflow::NativeExecutionFailure::SourceResourceMismatch
        || live.publication()->imageHandle() != retainedHandle) return 41;
    if (live.execute(op, *a, authoredFirst, authoredSecond, ec, e, &f)
        || f != videoopticalflow::NativeExecutionFailure::OwnerModeMismatch
        || live.publication()->imageHandle() != retainedHandle) return 42;
    if (live.execute(op, *a, authoredSecond, authoredFirst, pc, e, &f)
        || f != videoopticalflow::NativeExecutionFailure::SourceResourceMismatch
        || live.publication()->imageHandle() != retainedHandle) return 43;
  }
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
  auto makeDecoded = [] (std::uint32_t clip, std::uint8_t red) {
    videoopticalflow::DecodedSourceReceipt receipt;
    receipt.clipId = clip;
    receipt.sourceGeneration = 2;
    receipt.sourceSha256 = std::string (64, clip == 17 ? 'a' : 'b');
    receipt.presentationOrdinal = clip;
    receipt.frame.width = W;
    receipt.frame.height = H;
    receipt.frame.strideBytes = W * 4;
    receipt.frame.rgba.resize (W * H * 4, 255);
    for (std::size_t pixel = 0; pixel < W * H; ++pixel)
        receipt.frame.rgba[pixel * 4] = red;
    receipt.frame.timeBase = { 1, 24000 };
    receipt.frame.ptsTicks = clip * 1001;
    receipt.frame.color.primaries = colortransform::ColorSpace::SRGB;
    receipt.frame.color.transfer = colortransform::TransferFunction::SRGB;
    receipt.pixelSha256 = videohelper::sha256Bytes (
        receipt.frame.rgba.data(), receipt.frame.rgba.size());
    return receipt;
  };
  auto uploader = videoopticalflow::makeNativeSourceUploader (GEN, REV);
  if (!uploader) return 21;
  auto decodedFirst = makeDecoded (17, 80);
  auto decodedSecond = makeDecoded (18, 150);
  videoopticalflow::SourceFrameV2 expectedFirst, expectedSecond;
  colortransform::Description firstColor, secondColor;
  if (!videoopticalflow::detail::expectedSource (
          decodedFirst, expectedFirst, firstColor, e)
      || !videoopticalflow::detail::expectedSource (
          decodedSecond, expectedSecond, secondColor, e)) return 22;
  colortransform::AdmissionFailure colorFailure {};
  const auto transform = colortransform::admit (
      firstColor, colortransform::BackendCapability::NativeGpu, colorFailure);
  if (!transform) return 23;
  auto firstLease = uploader->upload (decodedFirst, *transform, expectedFirst, e);
  auto secondLease = uploader->upload (decodedSecond, *transform, expectedSecond, e);
  if (!firstLease || !secondLease
      || !videoopticalflow::detail::exactLease (
          *firstLease, expectedFirst, GEN, REV, caps)
      || !videoopticalflow::detail::exactLease (
          *secondLease, expectedSecond, GEN, REV, caps)
      || firstLease->resourceIdentity() == secondLease->resourceIdentity()
      || firstLease->imageHandle() == secondLease->imageHandle()) return 24;
  if (uploader->uploadPreview(decodedFirst, *transform, expectedFirst, false, e)
      || e != videoopticalflow::kPreviewUploadPending) return 31;
  std::shared_ptr<const videoopticalflow::NativeSourceFrame> previewLease;
  for (int attempt = 0; attempt < 500 && !previewLease; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    previewLease = uploader->uploadPreview(
        decodedFirst, *transform, expectedFirst, false, e);
    if (!previewLease && e != videoopticalflow::kPreviewUploadPending) return 32;
  }
  if (!previewLease
      || previewLease->frame().identity.content != firstLease->frame().identity.content
      || !videoopticalflow::detail::exactLease(
          *previewLease, expectedFirst, GEN, REV, caps)) return 33;
  previewLease.reset();
  int priorFramebuffer = 0, priorTexture = 0;
  int format = 0, uploadedWidth = 0, uploadedHeight = 0;
  glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &priorFramebuffer);
  glGetIntegerv(GL_TEXTURE_BINDING_2D, &priorTexture);
  glBindTexture(GL_TEXTURE_2D, static_cast<unsigned>(firstLease->imageHandle()));
  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &format);
  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &uploadedWidth);
  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &uploadedHeight);
  unsigned uploadFramebuffer = 0;
  gl.GenFramebuffers(1, &uploadFramebuffer);
  gl.BindFramebuffer(GL_READ_FRAMEBUFFER, uploadFramebuffer);
  gl.FramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                          GL_TEXTURE_2D,
                          static_cast<unsigned>(firstLease->imageHandle()), 0);
  float linearPixel[4] {};
  if (gl.CheckFramebufferStatus(GL_READ_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, linearPixel);
  gl.BindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<unsigned>(priorFramebuffer));
  gl.DeleteFramebuffers(1, &uploadFramebuffer);
  glBindTexture(GL_TEXTURE_2D, static_cast<unsigned>(priorTexture));
  const float expectedRed = std::pow((80.0f / 255.0f + 0.055f) / 1.055f, 2.4f);
  if (format != GL_RGBA16F || uploadedWidth != W || uploadedHeight != H
      || std::abs(linearPixel[0] - expectedRed) > 0.01f
      || std::abs(linearPixel[3] - 1.0f) > 0.01f) return 30;
  auto wrongExpected = expectedSecond;
  ++wrongExpected.frame.timestamp.ticks;
  if (uploader->upload (decodedSecond, *transform, wrongExpected, e)) return 25;
  if (uploader->uploadPreview(decodedSecond, *transform, wrongExpected, false, e))
    return 34;
  const auto uploadedFirst = static_cast<unsigned> (firstLease->imageHandle());
  const auto uploadedSecond = static_cast<unsigned> (secondLease->imageHandle());
  firstLease.reset();
  secondLease.reset();
  if (glIsTexture (uploadedFirst) || glIsTexture (uploadedSecond)) return 26;
  auto retainedLease = uploader->upload (decodedFirst, *transform, expectedFirst, e);
  if (!retainedLease) return 27;
  const auto retainedSourceTexture = static_cast<unsigned> (retainedLease->imageHandle());
  const auto deferredTexture = static_cast<unsigned>(exporter.publication()->imageHandle());
  auto *w2 = glfwCreateWindow(64, 64, "flow2", nullptr, nullptr);
  if (!w2) return 7;
  glfwMakeContextCurrent(w2);
  if (uploader->upload (decodedFirst, *transform, expectedFirst, e)) return 28;
  exporter.reset(); // A foreign thread/context must not steal the live owner.
  if (glfwGetCurrentContext() != w2) return 17;
  glfwMakeContextCurrent(w);
  if (!glIsTexture(deferredTexture)) return 18;
  first.reset();
  second.reset();
  stationary.reset();
  if (uploader->uploadPreview(decodedSecond, *transform, expectedSecond, false, e)
      || e != videoopticalflow::kPreviewUploadPending) return 35;
  videoopticalflow::invalidateNativeSourceUploads(
      reinterpret_cast<uintptr_t>(w));
  if (retainedLease->imageHandle() != 0 || glIsTexture(retainedSourceTexture)) return 29;
  retainedLease.reset();
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
      "preview_export=yes textured_xy_oracle=yes stationary_exact_zero=yes occluded_background_zero=yes ambiguous_zero=yes "
      "motion_view_range16_pixels=yes motion_view_clamp_pixels=yes legacy_zero_image_bytes=yes raw_field_unchanged=yes optical_blur_xy_pixels=yes finite_readback=yes invalid=yes state_restored=yes sustained_release=yes "
      "deferred_release=yes retired_release=yes recreate=yes\n");
  return 0;
}
