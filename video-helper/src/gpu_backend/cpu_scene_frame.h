#pragma once

// Uploads a CPU-rendered scene image (the path tracer engine) as an owned
// native fixture frame, so the compositor takes it exactly like a raster
// scene draw: Color texture, R32F depth texture, bottom-first rows, a nonzero
// context identity and generation, and sampled single-sample views.
//
// The OpenGL build implements this in cpu_scene_frame_opengl.cpp with the
// calling thread's current context. Other builds link
// cpu_scene_frame_stub.cpp, which refuses.
//
// Frame layers finish the image on the GPU. For each layer a fragment pass samples
// the layer's frame texture at the traced uv and adds light * frame.rgb * frame.a,
// plus the background where the frame's alpha uncovers the surface, onto the
// uploaded colour. Nothing reads the frame back.

#include "backend.h"

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
};

// backend names the fixture backend the frame must match ("opengl").
std::shared_ptr<const NativeFixtureSceneFrame> uploadCpuSceneFrame (
    const std::string& backend, const CpuSceneImage& image, std::string& error);

} // namespace arbitgpu
