#include "../src/score_renderer.h"
#include "../src/beat_timeline.h"
#include "../src/block_c_frame_owner.h"

#include <cstdio>
#include <limits>

int main()
{
    int failures = 0;
    int checks = 0;
    const auto check = [&](bool condition, const char* message)
    {
        ++checks;
        if (!condition) { ++failures; std::fprintf(stderr, "FAIL: %s\n", message); }
    };

    arbitmod::Score score;
    score.scoreRevision = 1;
    score.edoStepsPerOctave = 31;
    arbitmod::Note root;
    root.id = 1; root.startBeat = 0.0f; root.lengthBeats = 4.0f;
    root.freqHz = 261.625565f; root.diatonicIndex = 28; root.isRoot = true;
    arbitmod::Note third;
    third.id = 2; third.startBeat = 1.0f; third.lengthBeats = 2.0f;
    third.freqHz = 327.031956f; third.diatonicIndex = 30;
    third.baseAccidental = 1; third.linked = true;
    third.commas[0] = { 5, -1 }; third.commaCount = 1;
    arbitmod::Note edo;
    edo.id = 3; edo.startBeat = 2.0f; edo.lengthBeats = 1.0f;
    edo.freqHz = 392.0f; edo.diatonicIndex = 32;
    edo.edoActive = true; edo.edoInflection = 2; edo.edoDegree = 17;
    score.notes = { root, third, edo };
    score.links = { arbitmod::Link { 9, 2, 1, 5, 4, 0 } };
    const auto immutableScore = std::make_shared<const arbitmod::Score>(score);

    const canonicalblockc::OwnerIdentity identity { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    canonicalblockc::PreviewFrameRequest previewRequest { identity, 8, 4.0 };
    canonicalblockc::ExportFrameRequest exportRequest { identity, 8, 4.0 };
    check(canonicalblockc::makePreviewFrameKey(previewRequest)
              == canonicalblockc::makeExportFrameKey(exportRequest),
          "production preview and export seams construct equal keys for equal inputs");

    videotime::BeatTimeline beatMap;
    beatMap.reset(120.0, 4.0);
    const auto beatAt = [&](std::int64_t frame)
    {
        return beatMap.secondsToBeat(static_cast<double>(frame) / 4.0);
    };
    canonicalblockc::FrameOwner previewOwner;
    canonicalblockc::FrameOwner exportOwner;
    const auto preview = previewOwner.frameAt(previewRequest, immutableScore, beatAt);
    const auto exported = exportOwner.frameAt(exportRequest, immutableScore, beatAt);
    check(preview && exported && preview->key == exported->key,
          "isolated production owners publish equal frame identities");
    check(preview && exported
              && preview->noteTextureValues == exported->noteTextureValues
              && preview->linkTextureValues == exported->linkTextureValues,
          "production owners publish equal carrier note and link bytes");
    check(preview && exported
              && preview->noteIdentities == exported->noteIdentities
              && preview->linkIdentities == exported->linkIdentities,
          "production owners publish equal note and link row arrays");
    check(previewOwner.packCount() == 9 && exportOwner.packCount() == 9,
          "each production owner packs exactly once per warmed frame");

    const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> frozen = preview;
    const auto beforeHold = previewOwner.packCount();
    check(previewOwner.frameAt(previewRequest, immutableScore, beatAt) == frozen
              && previewOwner.packCount() == beforeHold,
          "same-frame production request preserves one frozen carrier");
    previewRequest.targetFrame = 12;
    check(previewOwner.frameAt(previewRequest, immutableScore, beatAt)
              && previewOwner.packCount() == beforeHold + 4,
          "mid-playback warm-forward packs each missing frame once");

    const auto basePacks = exportOwner.packCount();
    const auto mutateIdentity = [&](std::size_t field)
    {
        auto changed = identity;
        switch (field)
        {
            case 0: changed.projectGeneration += 100; break;
            case 1: changed.sourceGeneration += 100; break;
            case 2: changed.helperGeneration += 100; break;
            case 3: changed.backendGeneration += 100; break;
            case 4: changed.deviceGeneration += 100; break;
            case 5: changed.scoreGeneration += 100; break;
            case 6: changed.beatMapGeneration += 100; break;
            case 7: changed.fpsGeneration += 100; break;
            case 8: changed.loopGeneration += 100; break;
            case 9: changed.seekGeneration += 100; break;
            default: break;
        }
        canonicalblockc::FrameOwner liveOwner;
        const auto stale = liveOwner.frameAt(
            canonicalblockc::PreviewFrameRequest { identity, 12, 4.0 }, immutableScore, beatAt);
        const auto packsBeforeMutation = liveOwner.packCount();
        const auto live = liveOwner.frameAt(
            canonicalblockc::PreviewFrameRequest { changed, 12, 4.0 }, immutableScore, beatAt);
        canonicalblockc::FrameOwner freshExportOwner;
        const auto fresh = freshExportOwner.frameAt(
            canonicalblockc::ExportFrameRequest { changed, 12, 4.0 }, immutableScore, beatAt);
        check(stale && live && fresh && live != stale && live->key == fresh->key,
              "mid-playback lifecycle mutation clears stale same-owner output");
        check(live && fresh && live->noteTextureValues == fresh->noteTextureValues
                  && live->linkTextureValues == fresh->linkTextureValues,
              "mutated same owner matches fresh export carrier bytes");
        check(live && fresh && live->noteIdentities == fresh->noteIdentities
                  && live->linkIdentities == fresh->linkIdentities,
              "mutated same owner matches fresh export row arrays");
        check(liveOwner.packCount() == packsBeforeMutation + 13
                  && freshExportOwner.packCount() == 13,
              "lifecycle mutation and fresh export pack each frame exactly once");
    };
    for (std::size_t field = 0; field < 10; ++field) mutateIdentity(field);
    check(exportOwner.packCount() == basePacks, "independent owner mutations do not alter export owner state");

    videorender::ScoreClock clock;
    clock.beat = 1.5f; clock.beatsPerBar = 4.0f;
    std::map<std::string, double> params {
        { "historyBeats", 2.0 }, { "lookaheadBeats", 6.0 }, { "staffScale", 1.0 }
    };
    const auto rendered = videorender::renderScore(frozen, clock, 640, 360, params);
    check(rendered.rgba.size() == 640u * 360u * 4u,
          "frozen carrier renders bounded RGBA without retaining raw score data");

    const auto expectRejectedAndCleared = [&](canonicalblockc::PreviewFrameRequest request,
                                               std::shared_ptr<const arbitmod::Score> candidate,
                                               auto callback, const char* message)
    {
        canonicalblockc::FrameOwner owner;
        check(owner.frameAt(canonicalblockc::PreviewFrameRequest { identity, 1, 4.0 },
                            immutableScore, beatAt) != nullptr,
              "invalid-request fixture starts with cached output");
        const auto packs = owner.packCount();
        check(!owner.frameAt(request, std::move(candidate), callback)
                  && !owner.current() && owner.packedFrame() == -1
                  && owner.packCount() == packs,
              message);
    };

    auto badIdentity = previewRequest; badIdentity.targetFrame = 1;
    badIdentity.identity.backendGeneration = 0;
    expectRejectedAndCleared(badIdentity, immutableScore, beatAt,
                             "zero lifecycle identity clears output and receipts");
    auto badFrame = canonicalblockc::PreviewFrameRequest { identity, -1, 4.0 };
    expectRejectedAndCleared(badFrame, immutableScore, beatAt,
                             "negative frame clears output and receipts");
    auto badFps = canonicalblockc::PreviewFrameRequest { identity, 1,
        std::numeric_limits<double>::infinity() };
    expectRejectedAndCleared(badFps, immutableScore, beatAt,
                             "non-finite FPS clears output and receipts");
    expectRejectedAndCleared(canonicalblockc::PreviewFrameRequest { identity, 1, 4.0 },
                             {}, beatAt, "missing score clears output and receipts");

    const auto rejectScore = [&](auto damage, const char* message)
    {
        auto hostile = std::make_shared<arbitmod::Score>(score);
        damage(*hostile);
        expectRejectedAndCleared(canonicalblockc::PreviewFrameRequest { identity, 1, 4.0 },
                                 hostile, beatAt, message);
    };
    rejectScore([](auto& value) { value.notes.front().id = 0; },
                "zero note identity clears output and receipts");
    rejectScore([](auto& value) { value.notes.front().id = -4; },
                "negative note identity clears output and receipts");
    rejectScore([](auto& value) { value.links.front().id = 0; },
                "zero link identity clears output and receipts");
    rejectScore([](auto& value) { value.links.front().slaveNoteId = -1; },
                "negative link endpoint clears output and receipts");
    rejectScore([](auto& value) { value.notes.front().commaCount = 10; },
                "hostile row count clears output and receipts");
    rejectScore([](auto& value) { value.notes.front().freqHz
                    = std::numeric_limits<float>::quiet_NaN(); },
                "non-finite note number clears output and receipts");
    rejectScore([](auto& value) { value.notes.push_back(value.notes.front()); },
                "duplicate note identity clears output and receipts");
    rejectScore([](auto& value) { value.links.push_back(value.links.front()); },
                "duplicate link identity clears output and receipts");

    const auto nanBeat = [](std::int64_t) { return std::numeric_limits<double>::quiet_NaN(); };
    expectRejectedAndCleared(canonicalblockc::PreviewFrameRequest { identity, 1, 4.0 },
                             immutableScore, nanBeat,
                             "same-frame hostile beat clears output and receipts");
    const auto midWarmHostile = [](std::int64_t frame)
    {
        return frame == 2 ? std::numeric_limits<double>::quiet_NaN()
                          : static_cast<double>(frame) * 0.5;
    };
    expectRejectedAndCleared(canonicalblockc::PreviewFrameRequest { identity, 4, 4.0 },
                             immutableScore, midWarmHostile,
                             "mid-warm hostile beat clears partial output and receipts");

    auto oversized = std::make_shared<arbitmod::Score>(score);
    oversized->notes.resize(arbitblockc::kMaxNotes + 1, root);
    expectRejectedAndCleared(canonicalblockc::PreviewFrameRequest { identity, 1, 4.0 },
                             oversized, beatAt,
                             "oversized note count clears output and receipts");

    const auto noteTail = static_cast<std::size_t>(frozen->noteRowCount) * 16u;
    const auto linkTail = static_cast<std::size_t>(frozen->linkRowCount) * 4u;
    bool zeroTrailing = true;
    for (std::size_t i = noteTail; i < frozen->noteTextureValues.size(); ++i)
        zeroTrailing = zeroTrailing && frozen->noteTextureValues[i] == 0.0f;
    for (std::size_t i = linkTail; i < frozen->linkTextureValues.size(); ++i)
        zeroTrailing = zeroTrailing && frozen->linkTextureValues[i] == 0.0f;
    for (std::size_t i = static_cast<std::size_t>(frozen->noteRowCount);
         i < frozen->noteIdentities.size(); ++i)
        zeroTrailing = zeroTrailing && frozen->noteIdentities[i] == 0
            && frozen->notationRows[i].identity == 0;
    for (std::size_t i = static_cast<std::size_t>(frozen->linkRowCount);
         i < frozen->linkIdentities.size(); ++i)
        zeroTrailing = zeroTrailing && frozen->linkIdentities[i] == 0;
    check(zeroTrailing, "frozen carrier has zero trailing row and texture storage");

    std::printf("score-renderer: %d/%d checks passed\n", checks - failures, checks);
    return failures;
}
