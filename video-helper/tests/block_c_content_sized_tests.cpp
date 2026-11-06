#include "canonical_block_c_frame.h"
#include "particle_parameters.h"

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

// Every note sounds at beat 0. Link i joins notes i % (notes - 1) and the next one.
std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> frameFor (int notes, int links)
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
    for (int index = 0; index < links; ++index)
    {
        arbitmod::Link link;
        link.id = index + 1;
        link.slaveNoteId = index % (notes - 1) + 1;
        link.masterNoteId = index % (notes - 1) + 2;
        link.slaveHarmonic = 3;
        link.masterHarmonic = 2;
        score->links.push_back (link);
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

// Texture heights follow the row counts, and an empty score still binds one row.
void textureShapes()
{
    check (arbitblockc::noteTextureRows (0) == 1 && arbitblockc::noteTextureRows (300) == 300,
           "note texture height is one row per note");
    check (arbitblockc::linkTextureRows (0) == 1 && arbitblockc::linkTextureRows (1024) == 1
               && arbitblockc::linkTextureRows (1025) == 2,
           "link texture wraps every kLinkTextureWidth links");
    check (arbitblockc::textureValueBound (300, 1500)
               == arbitblockc::noteTextureValues (300) + arbitblockc::linkTextureValues (1500),
           "the work bound matches the packed sizes");
}

// 300 notes and 1500 links pack without drops, past the former 128 and 256 limits.
void longScorePacks()
{
    const auto frame = frameFor (300, 1500);
    check (canonicalblockc::valid (frame), "the long score packs into a valid frame");
    if (! canonicalblockc::valid (frame)) return;
    check (frame->noteRows() == 300 && frame->linkRows() == 1500, "every note and link has a row");
    check (frame->droppedNotes == 0 && frame->droppedLinks == 0, "nothing is dropped");
    check (frame->noteTextureHeight() == 300 && frame->linkTextureHeight() == 2,
           "texture heights follow the rows");
    check (frame->noteTextureValues.size() == 300u * 16u
               && frame->linkTextureValues.size() == 2u * arbitblockc::kLinkTextureWidth * 4u,
           "texture data is sized from the rows");
    // Links on both sides of the first row wrap read valid endpoint rows.
    for (const int edge : { 0, 1023, 1024, 1499 })
    {
        const auto at = static_cast<std::size_t> (edge) * 4u;
        const float slave = frame->linkTextureValues[at], master = frame->linkTextureValues[at + 1];
        check (slave >= 0 && slave < 300 && master >= 0 && master < 300 && slave != master,
               "a link endpoint row is outside the packed notes");
        check (frame->linkIdentities[static_cast<std::size_t> (edge)] != 0, "a packed link lost its id");
    }
    for (std::size_t edge = 1500u * 4u; edge < frame->linkTextureValues.size(); ++edge)
        check (frame->linkTextureValues[edge] == 0.0f, "the link texture tail is not zero");
    const auto upload = videorender::particleNoteUpload (frame.get(), true);
    check (upload.size() == frame->noteTextureValues.size(), "the particle upload carries every note row");
}
}

int main()
{
    textureShapes();
    longScorePacks();
    if (failures == 0) std::puts ("block_c_content_sized: all passed");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
