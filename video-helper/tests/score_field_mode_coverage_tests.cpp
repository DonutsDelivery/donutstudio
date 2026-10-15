// Focused coverage for every Score Sample Field measurement (choices 0 to 13).
//
// The expected values are derived by hand from the small score below. They are
// not read back from the implementation. Sample positions are mapping-space
// coordinates, and the mapping scale is 1 so note positions are exact:
//   x = start - beat (Onset axis), y = log2(bent frequency / root), z = track.
//
// Score, root 440 Hz, history and lookahead 8 beats:
//   N1 id -1   track 0  start 0 length 4  440 Hz  MIDI 69  velocity 127 (1.0), root
//   N2 id -102 track 1  start 2 length 4  base 1760 Hz, constant bend -12 semitones,
//              bent 880 Hz, MIDI 93, velocity 63.5 (0.5)
//   N3 id -103 track 2  start 4 length 2  110 Hz  MIDI 45  velocity 31.75 (0.25)
//   Link -301: slave N2, master N1, ratio 2/1.
// Note positions at beat b: N1 (-b,0,0), N2 (2-b,1,1), N3 (4-b,-2,2).
// Velocity influence uses radius 4. Scalar measures use gain 1 and bias 0.
// Tolerance is 1e-5, which covers float texture and position precision.

#include "canonical_block_c_frame.h"
#include "gpu_backend/backend.h"
#include "score_field_fixture.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{
namespace scoreContract = videowire::geometry::scorefield;
namespace scoreHelper = videohelper::scorefield;
using canonicalblockc::CanonicalBlockCFrame;
using canonicalblockc::FrameKey;
using canonicalblockc::FrameProducer;
using videowire::geometry::FieldData;
using videowire::geometry::FieldElement;

constexpr double kTolerance = 1.0e-5;
constexpr std::uint8_t kMeasureCount = 14;
using Expected = std::array<double, kMeasureCount>;
using FramePointer = std::shared_ptr<const CanonicalBlockCFrame>;

int failures = 0;

void check(bool condition, const std::string& label)
{
    if (!condition)
    {
        ++failures;
        std::cerr << "FAIL " << label << '\n';
    }
}

void checkNear(double actual, double expected, const std::string& label)
{
    if (std::fabs(actual - expected) > kTolerance)
    {
        ++failures;
        std::cerr << "FAIL " << label << ": actual " << actual
                  << ", expected " << expected << '\n';
    }
}

arbitmod::Note makeNote(int id, int track, float start, float length, float durationSeconds,
                        float midi, float freqHz, float velocity, bool root)
{
    arbitmod::Note note;
    note.id = id;
    note.trackId = track;
    note.startBeat = start;
    note.lengthBeats = length;
    note.durationSeconds = durationSeconds;
    note.midiNote = midi;
    note.freqHz = freqHz;
    note.velocity = velocity;
    note.isRoot = root;
    return note;
}

// Primes 2, 3, 5, 7, 11, 13 take the values base, base+1, ... base+5.
void setPrimes(arbitmod::Note& note, float base)
{
    for (std::size_t prime = 0; prime < 6; ++prime)
        note.primes[prime] = base + static_cast<float>(prime);
}

void setScoreHeader(arbitmod::Score& score)
{
    score.notationVersion = 1;
    score.scoreRevision = 1;
    score.rootFreq = 440.0f;
    score.historyBeats = 8.0f;
    score.lookaheadBeats = 8.0f;
}

std::shared_ptr<arbitmod::Score> measureScore(bool withLink = true, bool reversed = false)
{
    auto score = std::make_shared<arbitmod::Score>();
    setScoreHeader(*score);
    auto root = makeNote(-1, 0, 0.0f, 4.0f, 2.0f, 69.0f, 440.0f, 127.0f, true);
    setPrimes(root, 1.0f);
    auto second = makeNote(-102, 1, 2.0f, 4.0f, 2.0f, 93.0f, 1760.0f, 63.5f, false);
    second.pitchBendPoints.push_back(arbitmod::PitchBendPoint{0.0f, -12.0f});
    second.pitchBendPoints.push_back(arbitmod::PitchBendPoint{1.0f, -12.0f});
    setPrimes(second, 11.0f);
    auto third = makeNote(-103, 2, 4.0f, 2.0f, 1.0f, 45.0f, 110.0f, 31.75f, false);
    setPrimes(third, 21.0f);
    score->notes = {root, second, third};
    if (reversed) std::reverse(score->notes.begin(), score->notes.end());
    if (withLink) score->links.push_back(arbitmod::Link{-301, -102, -1, 2, 1, 0});
    return score;
}

// Two coincident notes joined by a zero-length link. Their transforms are equal.
std::shared_ptr<arbitmod::Score> coincidentLinkScore()
{
    auto score = std::make_shared<arbitmod::Score>();
    setScoreHeader(*score);
    auto root = makeNote(-1, 0, 0.0f, 4.0f, 2.0f, 69.0f, 440.0f, 127.0f, true);
    auto twin = makeNote(-104, 0, 0.0f, 4.0f, 2.0f, 69.0f, 440.0f, 127.0f, false);
    score->notes = {root, twin};
    score->links.push_back(arbitmod::Link{-302, -104, -1, 1, 1, 0});
    return score;
}

std::shared_ptr<arbitmod::Score> emptyScore()
{
    auto score = std::make_shared<arbitmod::Score>();
    setScoreHeader(*score);
    return score;
}

FramePointer frameAt(const std::shared_ptr<const arbitmod::Score>& score, float beat)
{
    FrameKey key;
    key.projectGeneration = key.sourceGeneration = key.helperGeneration = 1;
    key.backendGeneration = key.deviceGeneration = key.scoreGeneration = 1;
    key.beatMapGeneration = key.fpsGeneration = key.loopGeneration = key.seekGeneration = 1;
    key.fps = 60.0;
    key.frame = static_cast<std::int64_t>(beat * 60.0f);
    key.beat = static_cast<double>(beat);
    FrameProducer producer;
    return producer.evaluate(key, score, beat);
}

struct Sample
{
    std::array<double, 3> position;
    Expected expected;
};

// Sample path: one operation per choice, every sample checked against the
// hand-derived value and the X-only displacement.
void checkSampleField(const FramePointer& frame, const std::vector<Sample>& samples,
                      const std::string& label)
{
    if (frame == nullptr)
    {
        check(false, label + " canonical frame");
        return;
    }
    FieldData positions;
    for (const auto& sample : samples)
        positions.elements.push_back(FieldElement{{sample.position[0], sample.position[1],
                                                  sample.position[2], 0.0}});
    for (std::uint8_t choice = 0; choice < kMeasureCount; ++choice)
    {
        const auto op = scoreSampleMeasureOperation(choice);
        FieldData result;
        std::string error;
        const std::string name = label + " choice " + std::to_string(static_cast<int>(choice));
        if (!scoreHelper::evaluateSampleField(op, positions, frame, result, error)
            || result.elements.size() != samples.size())
        {
            check(false, name + " evaluates: " + error);
            continue;
        }
        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            const auto& components = result.elements[index].components;
            const std::string sampleLabel = name + " sample " + std::to_string(index);
            checkNear(components[0], samples[index].expected[choice], sampleLabel);
            check(components[1] == 0.0 && components[2] == 0.0 && components[3] == 0.0,
                  sampleLabel + " displaces only along X");
        }
    }
}

// Every choice returns the authored bias for each sample.
void checkBiasOnly(const FramePointer& frame, std::uint8_t choice, const std::string& label)
{
    FieldData positions;
    positions.elements = {FieldElement{{0.0, 0.0, 0.0, 0.0}}, FieldElement{{1.0, 2.0, 3.0, 0.0}}};
    auto op = scoreSampleMeasureOperation(choice);
    op.field.values[2] = 0.5;
    FieldData result;
    std::string error;
    if (!scoreHelper::evaluateSampleField(op, positions, frame, result, error)
        || result.elements.size() != 2)
    {
        check(false, label + " evaluates: " + error);
        return;
    }
    for (std::size_t index = 0; index < result.elements.size(); ++index)
    {
        const auto& components = result.elements[index].components;
        const std::string sampleLabel = label + " sample " + std::to_string(index);
        checkNear(components[0], 0.5, sampleLabel + " bias");
        check(components[1] == 0.0 && components[2] == 0.0 && components[3] == 0.0,
              sampleLabel + " has no other displacement");
    }
}

void checkBiasForAllChoices(const FramePointer& frame, const std::string& label)
{
    for (std::uint8_t choice = 0; choice < kMeasureCount; ++choice)
        checkBiasOnly(frame, choice, label + " choice " + std::to_string(static_cast<int>(choice)));
}

scoreContract::Binding makeBinding(scoreContract::Mode mode, std::array<float, 3> position)
{
    scoreContract::Binding binding;
    binding.fieldStableId = 9902;
    binding.sourceGeometryStableId = 9901;
    binding.consumerGeometryStableId = 9905;
    binding.scoreSourceStableId = 9903;
    binding.mode = mode;
    binding.radius = 4.0f;
    binding.gain = 1.0f;
    binding.bias = 0.0f;
    binding.direction = {1.0f, 0.0f, 0.0f};
    binding.positions.push_back(position);
    binding.elementIds.push_back(1);
    return binding;
}

bool evaluateSingle(const scoreContract::Binding& binding, const FramePointer& frame,
                    scoreHelper::Evaluation& result, const std::string& label)
{
    std::string error;
    const bool ok = scoreHelper::evaluate(binding, frame, result, error);
    if (!ok || result.offsets.size() != 1 || result.closestNoteIds.size() != 1
        || result.closestLinkIds.size() != 1 || result.linkRatios.size() != 1)
    {
        check(false, label + " evaluates one sample: " + error);
        return false;
    }
    return true;
}

void checkChoiceTable()
{
    const scoreContract::Mode expectedModes[kMeasureCount] = {
        scoreContract::Mode::closestNoteDistance, scoreContract::Mode::velocityInfluence,
        scoreContract::Mode::pitchSpacing, scoreContract::Mode::harmonicLinkDistance,
        scoreContract::Mode::pitchBendSemitones, scoreContract::Mode::noteAgeBeats,
        scoreContract::Mode::remainingDurationBeats, scoreContract::Mode::rootFlag,
        scoreContract::Mode::midiNoteNumber, scoreContract::Mode::resolvedFrequencyHz,
        scoreContract::Mode::harmonicLinkRatio, scoreContract::Mode::primeLattice,
        scoreContract::Mode::primeLattice, scoreContract::Mode::primeLattice};
    for (std::uint8_t choice = 0; choice < kMeasureCount; ++choice)
    {
        scoreContract::Mode mode = scoreContract::Mode::velocityInfluence;
        check(scoreContract::sampleFieldMode(choice, mode) && mode == expectedModes[choice],
              "choice " + std::to_string(static_cast<int>(choice)) + " maps to its measure");
    }
    scoreContract::Mode unused = scoreContract::Mode::velocityInfluence;
    check(!scoreContract::sampleFieldMode(kMeasureCount, unused), "choice 14 is refused");

    std::uint8_t slot = 99;
    check(scoreContract::samplePrimeLatticeSlot(11, slot) && slot == 0, "choice 11 reads X slot 0");
    check(scoreContract::samplePrimeLatticeSlot(12, slot) && slot == 1, "choice 12 reads Y slot 1");
    check(scoreContract::samplePrimeLatticeSlot(13, slot) && slot == 2, "choice 13 reads Z slot 2");
    check(!scoreContract::samplePrimeLatticeSlot(10, slot), "choice 10 is not a prime slot");
    check(!scoreContract::samplePrimeLatticeSlot(14, slot), "choice 14 is not a prime slot");
}

void checkSampleMeasures()
{
    const double s3 = std::sqrt(3.0);
    const double s14 = std::sqrt(14.0);
    const double linkA = std::sqrt(1.0 / 3.0);

    // Beat 1 transforms: N1 (-1,0,0), N2 (1,1,1), N3 (3,-2,2).
    // A (origin) is nearest N1 at distance 1.
    // C (2,0,0) is nearest N2 at distance sqrt(3). Its link projection clamps to N2.
    // E (3,-2,2) is exactly on N3, and the link projection clamps to N2.
    checkSampleField(frameAt(measureScore(), 1.0f),
        {
            Sample{{0.0, 0.0, 0.0},
                   {1.0, 1.25 - s3 / 8.0, 12.0, linkA, 0.0, 1.0, 3.0, 1.0, 69.0, 440.0, 2.0, 1.0, 2.0, 3.0}},
            Sample{{2.0, 0.0, 0.0},
                   {s3, 0.8125 - s3 / 8.0, 12.0, s3, -12.0, -1.0, 5.0, 0.0, 93.0, 880.0, 2.0, 11.0, 12.0, 13.0}},
            Sample{{3.0, -2.0, 2.0},
                   {0.0, 0.75 - s14 / 8.0, 24.0, s14, 0.0, -3.0, 5.0, 0.0, 45.0, 110.0, 2.0, 21.0, 22.0, 23.0}},
        },
        "beat 1");

    // Beat 3 transforms: N1 (-3,0,0), N2 (-1,1,1), N3 (1,-2,2).
    // B (origin) is nearest N2 at distance sqrt(3).
    // D (1,-2,2) is exactly on N3.
    checkSampleField(frameAt(measureScore(), 3.0f),
        {
            Sample{{0.0, 0.0, 0.0},
                   {s3, 0.8125 - s3 / 8.0, 12.0, s3, -12.0, 1.0, 3.0, 0.0, 93.0, 880.0, 2.0, 11.0, 12.0, 13.0}},
            Sample{{1.0, -2.0, 2.0},
                   {0.0, 0.75 - s14 / 8.0, 24.0, s14, 0.0, -1.0, 3.0, 0.0, 45.0, 110.0, 2.0, 21.0, 22.0, 23.0}},
        },
        "beat 3");

    // Beat 6: N1 at (-6,0,0), N2 at (-4,1,1), N3 at (-2,-2,2). N2 and N3 have
    // just ended, so remaining is exactly 0. G sits exactly on N3.
    checkSampleField(frameAt(measureScore(), 6.0f),
        {
            Sample{{-2.0, -2.0, 2.0},
                   {0.0, 0.75 - s14 / 8.0, 24.0, s14, 0.0, 2.0, 0.0, 0.0, 45.0, 110.0, 2.0, 21.0, 22.0, 23.0}},
        },
        "beat 6 ended");

    // Beat 7: N3 at (-3,-2,2) is still in the history window, one beat past its
    // end. Remaining is negative, and the sample sits exactly on N3.
    checkSampleField(frameAt(measureScore(), 7.0f),
        {
            Sample{{-3.0, -2.0, 2.0},
                   {0.0, 0.75 - s14 / 8.0, 24.0, s14, 0.0, 3.0, -1.0, 0.0, 45.0, 110.0, 2.0, 21.0, 22.0, 23.0}},
        },
        "beat 7 past end");
}

void checkEmptyScoreDefault()
{
    checkBiasForAllChoices(frameAt(emptyScore(), 1.0f), "empty score");
}

void checkOutOfWindowDefault()
{
    // At beat 15 every note ended more than 8 beats earlier, so nothing is resident.
    checkBiasForAllChoices(frameAt(measureScore(), 15.0f), "beat 15 outside history");
}

void checkMissingLinksDefault()
{
    // The sample path returns bias for link measures when the frame has no links.
    const auto frame = frameAt(measureScore(false), 1.0f);
    checkBiasOnly(frame, 3, "no links choice 3");
    checkBiasOnly(frame, 10, "no links choice 10");
}

void checkUnknownChoiceRefused()
{
    const auto frame = frameAt(measureScore(), 1.0f);
    FieldData positions;
    positions.elements = {FieldElement{{0.0, 0.0, 0.0, 0.0}}};
    const auto op = scoreSampleMeasureOperation(kMeasureCount);
    FieldData result;
    std::string error;
    check(!scoreHelper::evaluateSampleField(op, positions, frame, result, error)
              && error.find("unknown measurement") != std::string::npos,
          "choice 14 refused as an unknown measurement");
}

void checkVectorEvaluation()
{
    const auto frame = frameAt(measureScore(), 1.0f);

    // Prime Lattice in the vector path reads all three axes of the nearest note.
    // Sample A is nearest N1, whose primes are 1 through 6.
    auto lattice = makeBinding(scoreContract::Mode::primeLattice, {0.0f, 0.0f, 0.0f});
    lattice.direction = {1.0f, 1.0f, 1.0f};
    scoreHelper::Evaluation result;
    if (evaluateSingle(lattice, frame, result, "lattice default slots"))
    {
        checkNear(result.offsets[0][0], 1.0, "lattice slots 0,1,2 X (prime 2)");
        checkNear(result.offsets[0][1], 2.0, "lattice slots 0,1,2 Y (prime 3)");
        checkNear(result.offsets[0][2], 3.0, "lattice slots 0,1,2 Z (prime 5)");
        check(result.closestNoteIds[0] == -1, "lattice nearest note is N1");
    }
    lattice.primes[0] = 3;
    lattice.primes[1] = 4;
    lattice.primes[2] = 5;
    if (evaluateSingle(lattice, frame, result, "lattice slots 3,4,5"))
    {
        checkNear(result.offsets[0][0], 4.0, "lattice slot 3 (prime 7)");
        checkNear(result.offsets[0][1], 5.0, "lattice slot 4 (prime 11)");
        checkNear(result.offsets[0][2], 6.0, "lattice slot 5 (prime 13)");
    }

    // Harmonic link distance and ratio at sample A. The only link is N2 to N1,
    // ratio 2/1, and the projection onto the segment sits at t = 1/3.
    auto linkBinding = makeBinding(scoreContract::Mode::harmonicLinkDistance, {0.0f, 0.0f, 0.0f});
    if (evaluateSingle(linkBinding, frame, result, "link ids"))
    {
        checkNear(result.offsets[0][0], std::sqrt(1.0 / 3.0), "link distance at sample A");
        check(result.closestLinkIds[0] == -301, "closest link ID is -301");
        checkNear(result.linkRatios[0][0], 2.0, "link numerator");
        checkNear(result.linkRatios[0][1], 1.0, "link denominator");
    }

    // Coincident link endpoints reduce to a point distance from the shared position.
    const auto twinFrame = frameAt(coincidentLinkScore(), 1.0f);
    auto twinBinding = makeBinding(scoreContract::Mode::harmonicLinkDistance, {0.0f, 0.0f, 0.0f});
    if (evaluateSingle(twinBinding, twinFrame, result, "coincident endpoints"))
    {
        checkNear(result.offsets[0][0], 1.0, "coincident endpoints give point distance");
        check(result.closestLinkIds[0] == -302, "coincident link ID is -302");
    }

    // Exact distance ties choose the smaller signed identity, regardless of row order.
    // P = (-2, 0.5, 0.5) is at squared distance 1.5 from both N1 and N2 at beat 3.
    for (const bool reversed : {false, true})
    {
        const auto tieFrame = frameAt(measureScore(true, reversed), 3.0f);
        auto tie = makeBinding(scoreContract::Mode::closestNoteDistance, {-2.0f, 0.5f, 0.5f});
        if (evaluateSingle(tie, tieFrame, result, reversed ? "tie reversed" : "tie"))
        {
            check(result.closestNoteIds[0] == -102,
                  std::string("tie chooses the smaller signed identity, reversed=")
                      + (reversed ? "true" : "false"));
            checkNear(result.offsets[0][0], std::sqrt(1.5), "tie distance");
        }
    }
}

void checkInvalidPrime()
{
    const std::string expected = "Score Field prime must be one of 2, 3, 5, 7, 11 or 13";
    auto binding = makeBinding(scoreContract::Mode::primeLattice, {0.0f, 0.0f, 0.0f});
    binding.primes[2] = 6;

    std::string error;
    check(!scoreContract::validate(binding, 1, error) && error == expected,
          "prime slot 6 refused by validate");

    const auto frame = frameAt(measureScore(), 1.0f);
    scoreHelper::Evaluation result;
    error.clear();
    check(!scoreHelper::evaluate(binding, frame, result, error) && error == expected,
          "prime slot 6 refused by evaluate");

    binding.primes[2] = 5;
    error.clear();
    check(scoreContract::validate(binding, 1, error) && error.empty(),
          "prime slot 5 (13) is accepted");
}

} // namespace

int main()
{
    checkChoiceTable();
    checkSampleMeasures();
    checkEmptyScoreDefault();
    checkOutOfWindowDefault();
    checkMissingLinksDefault();
    checkUnknownChoiceRefused();
    checkVectorEvaluation();
    checkInvalidPrime();
    if (failures != 0)
    {
        std::cerr << failures << " Score Sample Field coverage check(s) failed\n";
        return 1;
    }
    std::cout << "Score Sample Field mode coverage passed\n";
    return 0;
}
