// Note instancing is sized from the Block C frame: every resident note becomes
// an instance, the GL texel texture wraps at 1024, and the prior-note Motion
// table takes every identity.

#include "canonical_block_c_frame.h"
#include "gpu_backend/backend.h"

#include <cstdio>
#include <cstdlib>
#include <memory>

namespace
{
int failures = 0;

void check (bool condition, const char* what)
{
    if (condition) return;
    std::fprintf (stderr, "FAIL: %s\n", what);
    ++failures;
}

std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> frameFor (int notes)
{
    auto score = std::make_shared<arbitmod::Score>();
    score->notationVersion = 1;
    score->scoreRevision = 1;
    score->rootFreq = 440.0f;
    for (int index = 0; index < notes; ++index)
    {
        arbitmod::Note note;
        note.id = index + 1;
        note.trackId = 1;
        note.startBeat = 0.0f;
        note.lengthBeats = 8.0f;
        note.velocity = 100.0f;
        note.freqHz = 110.0f + static_cast<float> (index);
        note.durationSeconds = 4.0f;
        score->notes.push_back (note);
    }
    canonicalblockc::FrameKey key;
    key.projectGeneration = key.sourceGeneration = key.helperGeneration = 1;
    key.backendGeneration = key.deviceGeneration = key.scoreGeneration = 1;
    key.beatMapGeneration = key.fpsGeneration = key.loopGeneration = 1;
    key.seekGeneration = 1;
    key.fps = 60.0;
    canonicalblockc::FrameProducer producer;
    return producer.evaluate (key, score, 0.0f);
}

void everyNoteIsAnInstance()
{
    const auto frame = frameFor (300);
    check (frame != nullptr && frame->noteRows() == 300, "300 notes take 300 Block C rows");
    if (frame == nullptr) return;
    arbitgpu::NativeImportedSceneRuntimeInputs inputs;
    inputs.canonicalBlockCFrame = frame;
    inputs.noteInstanceMapping = visualnoteinstancing::Mapping {};
    const auto batch = arbitgpu::prepareNativeNoteInstances (inputs);
    check (batch.admitted && batch.count == 300, "all 300 notes become instances");
    check (batch.transforms.size() == 300 && batch.identities.size() == 300
               && batch.canonicalRows.size() == 300 && batch.appearanceValues.size() == 300,
           "every per-instance vector holds 300 entries");
    check (batch.canonicalRows.size() == 300 && batch.canonicalRows[299] == 299,
           "row 299 keeps its canonical row past the old 128 limit");

    const auto packed = arbitgpu::noteInstanceShaderTransforms (batch);
    check (packed.size() == 300 && packed[200][0] == batch.transforms[200][0]
               && packed[200][3] == batch.appearanceValues[200],
           "packed transforms carry appearance in w for every note");

    const auto table = arbitgpu::sceneNoteMotionTransforms (inputs);
    check (table != nullptr && table->count == 300 && table->find (batch.identities[250]).has_value(),
           "the prior-note Motion table finds a note past 128");
}

void textureRowsWrap()
{
    check (arbitgpu::noteInstanceTextureRows (0) == 1, "an empty batch binds one row");
    check (arbitgpu::noteInstanceTextureRows (1024) == 1, "1024 notes fill one row");
    check (arbitgpu::noteInstanceTextureRows (1025) == 2, "note 1024 starts the second row");
}

void unmappedDrawsOnce()
{
    arbitgpu::NativeImportedSceneRuntimeInputs inputs;
    const auto batch = arbitgpu::prepareNativeNoteInstances (inputs);
    check (! batch.admitted && batch.count == 1 && batch.transforms.size() == 1
               && batch.transforms[0][3] == 1.0f,
           "without a mapping one neutral instance is drawn");
}
} // namespace

int main()
{
    everyNoteIsAnInstance();
    textureRowsWrap();
    unmappedDrawsOnce();
    if (failures == 0) std::puts ("note_instancing_content_sized: all passed");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
