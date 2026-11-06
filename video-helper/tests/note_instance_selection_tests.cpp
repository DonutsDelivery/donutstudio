// Note Instanced Mesh selection: the Notes window (sounding, upcoming, recently
// ended) and Filter By keep only the chosen score rows as instances.
#include "canonical_block_c_frame.h"
#include "gpu_backend/backend.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{
int failures = 0;
void check(bool condition, const char* what)
{
    if (condition) return;
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
}

// At beat 6: note -1 ended at beat 2, note -2 sounds until beat 8, note -3
// starts at beat 9 and note -4 starts at beat 13.
std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> frameAtBeat6()
{
    auto score = std::make_shared<arbitmod::Score>();
    score->notationVersion = 1; score->scoreRevision = 1; score->rootFreq = 440;
    score->historyBeats = 8; score->lookaheadBeats = 8;
    const struct { int id; float start, length, midi; } notes[] {
        { -1, 0, 2, 60 }, { -2, 4, 4, 64 }, { -3, 9, 2, 67 }, { -4, 13, 1, 72 } };
    for (const auto& n : notes)
    {
        arbitmod::Note note;
        note.id = n.id; note.startBeat = n.start; note.lengthBeats = n.length; note.midiNote = n.midi;
        note.freqHz = 440.0f; note.velocity = 100; note.durationSeconds = n.length / 2;
        score->notes.push_back(note);
    }
    canonicalblockc::FrameKey key;
    key.projectGeneration = key.sourceGeneration = key.helperGeneration = 1;
    key.backendGeneration = key.deviceGeneration = key.scoreGeneration = 1;
    key.beatMapGeneration = key.fpsGeneration = key.loopGeneration = key.seekGeneration = 1;
    key.fps = 60; key.beat = 6; key.frame = 360;
    canonicalblockc::FrameProducer producer;
    return producer.evaluate(key, score, 6);
}

std::vector<float> selectedMidiNotes(const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>& frame,
                                     const visualnoteinstancing::Mapping& mapping)
{
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.canonicalBlockCFrame = frame;
    inputs.noteInstanceMapping = mapping;
    const auto batch = arbitgpu::prepareNativeNoteInstances(inputs);
    std::vector<float> midi;
    for (std::size_t i = 0; i < batch.count; ++i)
        midi.push_back(frame->noteTextureValues[batch.canonicalRows[i] * arbitblockc::kTexelsPerNote * 4u]);
    std::sort(midi.begin(), midi.end());
    return midi;
}
} // namespace

int main()
{
    using visualnoteinstancing::NoteWindow;
    const auto frame = frameAtBeat6();
    check(frame != nullptr, "the score frame evaluates");
    if (!frame) return 1;

    visualnoteinstancing::Mapping mapping;
    mapping.x = visualnoteinstancing::MappingAxis::MidiNote;
    mapping.schemaVersion = visualnoteinstancing::kDirectAxesMappingSchemaVersion;
    check(selectedMidiNotes(frame, mapping) == std::vector<float>{60, 64, 67, 72}, "All keeps every resident note");

    const auto with = [&](NoteWindow window, float beats)
    {
        auto copy = mapping; copy.window = window; copy.windowBeats = beats;
        return selectedMidiNotes(frame, copy);
    };
    check(with(NoteWindow::Sounding, 0) == std::vector<float>{64}, "Sounding keeps the held note");
    check(with(NoteWindow::Upcoming, 0) == std::vector<float>{67, 72}, "Upcoming keeps notes after the playhead");
    check(with(NoteWindow::Upcoming, 4) == std::vector<float>{67}, "Window Beats limits upcoming notes");
    check(with(NoteWindow::RecentlyEnded, 0) == std::vector<float>{60}, "Recently Ended keeps the finished note");
    check(with(NoteWindow::RecentlyEnded, 3) == std::vector<float>{}, "Window Beats limits ended notes");
    check(with(NoteWindow::SoundingAndUpcoming, 4) == std::vector<float>{64, 67}, "Sounding and Upcoming");
    check(with(NoteWindow::SoundingAndRecentlyEnded, 0) == std::vector<float>{60, 64}, "Sounding and Recently Ended");
    check(with(NoteWindow::All, 4) == std::vector<float>{60, 64, 67}, "All with Window Beats drops far notes");

    auto filtered = mapping;
    filtered.filterAxis = static_cast<std::int32_t>(visualnoteinstancing::MappingAxis::MidiNote);
    filtered.filterMin = 62; filtered.filterMax = 70;
    check(selectedMidiNotes(frame, filtered) == std::vector<float>{64, 67}, "Filter By MIDI Note keeps the range");

    // The selection travels in a trailer; a default mapping keeps the old bytes.
    const auto plain = visualnoteinstancing::encode(mapping);
    check(plain.find("select") == std::string::npos, "default selection writes no trailer");
    filtered.window = NoteWindow::SoundingAndUpcoming; filtered.windowBeats = 2.5f;
    const auto encoded = visualnoteinstancing::encode(filtered);
    visualnoteinstancing::Mapping decoded;
    check(visualnoteinstancing::decode(encoded, decoded), "selection trailer decodes");
    check(decoded.window == filtered.window && decoded.windowBeats == 2.5f && decoded.filterAxis == filtered.filterAxis
          && decoded.filterMin == 62 && decoded.filterMax == 70, "selection round-trips");
    auto inverted = filtered; inverted.filterMin = 80;
    check(!visualnoteinstancing::valid(inverted), "an inverted filter range is invalid");
    auto stray = mapping; stray.filterMax = 5;
    check(!visualnoteinstancing::valid(stray), "a filter range without a filter axis is invalid");

    if (failures == 0) std::cout << "note instance selection tests passed\n";
    return failures == 0 ? 0 : 1;
}
