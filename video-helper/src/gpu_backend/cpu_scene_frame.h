#pragma once

// Uploads a CPU-rendered scene image (the path tracer engine) as an owned
// native fixture frame, so the compositor takes it exactly like a raster
// scene draw: Color texture, R32F depth texture, bottom-first rows, a nonzero
// context identity and generation, and sampled single-sample views.
//
// The OpenGL build implements this in cpu_scene_frame_opengl.cpp with the
// calling thread's current context. The Metal build implements it in
// cpu_scene_frame_metal.mm on the sokol Metal device, with top-first rows and
// BGRA8 SDR colour as Metal frames have. Headless builds link
// cpu_scene_frame_stub.cpp, which refuses.
//
// Frame layers finish the image on the GPU. For each layer a fragment pass samples
// the layer's frame texture at the traced uv and adds light * frame.rgb * frame.a,
// plus the background where the frame's alpha uncovers the surface, onto the
// uploaded colour. Nothing reads the frame back.

#include "backend.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace arbitgpu
{

struct CpuSceneFrameLayer final
{
    std::shared_ptr<const NativeFixtureSceneFrame> frame;
    const std::vector<std::uint8_t>* light = nullptr;  // RGBA16F half bits, tight rows, bottom first
    const std::vector<float>* uvCoverage = nullptr;    // u, v, coverage per pixel, bottom first
};

struct CpuSceneImage final
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    // true: RGBA16F half bits, linear sRGB. false: RGBA8 holding clamped
    // linear values with no transfer, like the raster's SDR scene draw.
    bool linearColor = false;
    const std::vector<std::uint8_t>* color = nullptr;  // tight rows, bottom first
    const std::vector<float>* depth = nullptr;         // width * height, bottom first
    std::vector<CpuSceneFrameLayer> frameLayers;
    float background[3] { 0.0f, 0.0f, 0.0f };          // shown through a frame's zero alpha, in color's encoding
    // Render outputs in the raster's raw pass formats, tight rows, bottom first.
    // All six or none: without them the frame has Color and Depth alone.
    const std::vector<std::uint16_t>* normal = nullptr;      // RGBA16F half bits
    const std::vector<std::uint16_t>* emission = nullptr;    // RGBA16F half bits
    const std::vector<std::uint8_t>* mask = nullptr;         // R8
    const std::vector<std::uint32_t>* materialId = nullptr;  // R32UI
    const std::vector<std::uint32_t>* objectId = nullptr;    // R32UI
    const std::vector<std::uint16_t>* motion = nullptr;      // RG16F half bits, pixels, Y up
    // What the Image shows, run on the outputs by the raster's own resolve
    // shader: a view of one output, or the pass composite program. Color with
    // an empty program keeps the traced colour and runs nothing.
    renderpassoutput::Output imageOutput = renderpassoutput::Output::Color;
    renderpasscomposite::Program passProgram {};
    // The resolve target's clear, where the mask is empty: zero, or a pass
    // composite inspection's value (renderpasscomposite::inspectionClear).
    std::array<float, 4> resolveClear { 0.0f, 0.0f, 0.0f, 0.0f };
};

// The raster's pass resolve shader (backend_opengl.cpp): it turns Color, Depth
// and the raw passes into an Image view or a pass composite.
const char* fixturePassResolveVertexSource() noexcept;
std::string fixturePassResolveFragmentSource();

// backend names the fixture backend the frame must match ("opengl" or "metal").
std::shared_ptr<const NativeFixtureSceneFrame> uploadCpuSceneFrame (
    const std::string& backend, const CpuSceneImage& image, std::string& error);

} // namespace arbitgpu
