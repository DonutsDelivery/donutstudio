#pragma once

#include "../src/canonical_block_c_frame.h"
#include "../src/gpu_backend/backend.h"

#include <string>
#include <vector>

template <typename ReadPixels>
bool verifyNativeNoteAppearance(arbitgpu::NativeFixtureSceneBackend& backend,
                                ReadPixels readPixels, bool bgra, std::string& error)
{
    using namespace HarmonicMIDI::grid;
    auto scene = std::make_shared<Visual3DScene>();
    scene->id = { 8131 };
    scene->vertices[0].position = { -0.3f, -0.3f, 0.0f };
    scene->vertices[1].position = { 0.3f, -0.3f, 0.0f };
    scene->vertices[2].position = { 0.0f, 0.3f, 0.0f };
    for (std::size_t index = 0; index < 3; ++index)
    {
        scene->vertices[index].normal = { 0.0f, 0.0f, 1.0f };
        scene->vertices[index].color = { 1.0f, 1.0f, 1.0f, 1.0f };
        scene->indices[index] = static_cast<std::uint32_t>(index);
    }
    scene->vertexCount = scene->indexCount = 3;
    scene->materials[0].id = { 1 };
    scene->materials[0].baseColor = { 0.1f, 0.1f, 0.1f };
    scene->materialCount = 1;
    scene->objects[0].id = { 1 };
    scene->objects[0].material = { 1 };
    scene->objects[0].vertexCount = scene->objects[0].indexCount = 3;
    scene->objectCount = 1;
    scene->cameras[0].id = { 1 };
    scene->cameras[0].transform.translation = { 0.9f, 0.0f, 5.0f };
    scene->cameras[0].verticalFovRadians = 0.7853981634f;
    scene->cameras[0].nearPlane = 0.1f;
    scene->cameras[0].farPlane = 1000.0f;
    scene->cameraCount = 1;
    scene->activeCamera = { 1 };

    auto score = std::make_shared<arbitmod::Score>();
    score->notationVersion = score->scoreRevision = 1;
    score->rootFreq = 440.0f;
    for (int index = 0; index < 2; ++index)
    {
        arbitmod::Note note;
        note.id = index == 0 ? -1 : -102;
        note.trackId = 1;
        note.startBeat = static_cast<float>(index);
        note.lengthBeats = 4.0f;
        note.durationSeconds = 2.0f;
        note.velocity = index == 0 ? 1.0f : 126.0f;
        note.freqHz = 440.0f;
        score->notes.push_back(note);
    }
    canonicalblockc::FrameKey key;
    key.projectGeneration = key.sourceGeneration = key.helperGeneration = 1;
    key.backendGeneration = key.deviceGeneration = key.scoreGeneration = 1;
    key.beatMapGeneration = key.fpsGeneration = key.loopGeneration = key.seekGeneration = 1;
    key.fps = 60.0;
    canonicalblockc::FrameProducer producer;
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.canonicalBlockCFrame = producer.evaluate(key, score, 0.0f);
    visualnoteinstancing::Mapping mapping;
    mapping.xScale = 1.8f;
    mapping.yScale = mapping.zScale = 0.0f;
    mapping.meshScale = 1.0f;
    mapping.appearanceLow = { 1.0f, 0.0f, 0.0f, 0.2f };
    mapping.appearanceHigh = { 0.0f, 0.0f, 1.0f, 0.6f };
    inputs.noteInstanceMapping = mapping;
    const auto prepared = backend.prepare(scene);
    const auto render = [&](const auto& runtime)
    {
        const auto result = backend.render(scene, prepared.resources, 64, 64, runtime);
        if (!result.rendered) error = result.error;
        return result.rendered ? readPixels(result.frame) : std::vector<std::uint8_t>{};
    };
    if (!prepared.prepared) { error = prepared.error; return false; }
    const auto pixels = render(inputs);
    if (pixels.size() != 64u * 64u * 4u) return false;
    std::size_t red = 0, blue = 0;
    for (std::size_t offset = 0; offset < pixels.size(); offset += 4)
    {
        const auto r = pixels[offset + (bgra ? 2 : 0)];
        const auto b = pixels[offset + (bgra ? 0 : 2)];
        red += r > 20 && r > 2 * b;
        blue += b > 40 && b > 2 * r;
    }
    if (red < 5 || blue < 5) { error = "native note color ramp did not produce both note colors"; return false; }
    auto noEmission = inputs;
    noEmission.noteInstanceMapping->appearanceLow[3] = 0.0f;
    noEmission.noteInstanceMapping->appearanceHigh[3] = 0.0f;
    if (render(noEmission) == pixels) { error = "native note emission did not change pixels"; return false; }
    ++key.seekGeneration;
    ++key.loopGeneration;
    inputs.canonicalBlockCFrame = producer.evaluate(key, score, 0.0f);
    if (render(inputs) != pixels) { error = "native note appearance changed after timeline reset"; return false; }
    return true;
}
