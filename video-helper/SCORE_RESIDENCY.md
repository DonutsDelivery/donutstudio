# Complete scores and resident Block C rows

`HarmonicMIDIProcessor::buildArbitScoreShared()` in `plugin/Source/PluginProcessor.cpp` copies the complete projected note and harmonic-link vectors. `buildScoreVar()` serializes the same score for preview and export. The score includes MIDI clip instances and their structural links even when the project has no freeform notes.

`FrameProducer::admissible()` validates that complete source. Zero or duplicate identities, missing note masters or link endpoints, invalid timing/pitch/notation data and invalid lifecycle metadata still reject. Note and link identity sets replace pairwise duplicate and endpoint scans. Records outside the current visible interval receive the same validation as resident records.

`BlockCPacker` owns the separate GPU residency limit. Its existing selection keeps at most 128 notes, prioritizing sounding notes, then upcoming notes, then recent history. Surviving notes keep their rows. It emits at most 256 links whose endpoints are both resident, in source link order. A valid source master outside the selected set maps to the existing absent-row sentinel. The full source vectors are never resized or truncated. The factory continues to check packed row limits and texture dimensions before publishing a canonical frame.

## Restored-project evidence

The 23 September 2026 investigation used base `c0ec89a3503b5e16dd5a13d444aeb2dfefbc6b8b` and the saved repro at `.tmp/cos-plan3-postcrash/restored-user-project.arbit`. Its SHA-256 is `4a9fc78938c2bed46fbac93c81d5d97da13ea6756543c3bf34e9d3f04f528d17`.

Read-only Python decoding of its JUCE MemoryBlock/zlib processor state found 1,035 `ClipNote` records and 257 `PeriodicLink` records, with no freeform `Note` records. The text clip data independently contains the same 1,035 MIDI notes. Clip 10 alone has 260 notes and a single pass. The legacy text `links` section is empty; the structural links are in the embedded state.

The saved `restored-repro/receipt.json` reports a connected docked viewport, 851 host render entries, zero shown/presented/shared frames and empty renderer errors. On the base revision, the complete source was rejected by comparisons against `kMaxNotes` and `kMaxLinks` before the resident packer ran. The viewport then continued its render loop without publishing the rejection or reaching presentation pacing. These source counts establish the capacity mismatch; they do not constitute a successful render after the fix.

## Invalid-frame handling and cost

The viewport now publishes the canonical admission reason through `rendererError` and logs a reason only when it changes. An invalid nonempty score still prevents frame publication. A 50 ms retry delay runs outside the control mutex, limiting repeated failed attempts to at most 20 per second before other work. The loop continues servicing updates. On recovery or score removal, it clears the matching score error without erasing a different renderer error, and resets the presentation pacing clock to avoid catching up missed ticks.

Identity indexing uses expected linear time and temporary storage in source notes plus links. Other note validation, resident candidate sorting, note lookup and deterministic frame warming retain their existing costs. This change does not claim constant cost for arbitrarily long scores or measured native frame times. The retained GPU textures remain 128 rows of four RGBA32F texels and 256 rows of one RGBA32F texel, 12 KiB total.

## Verification

Prepared C++ coverage is in existing targets, with no new translation unit or CMake change:

| Target | Added coverage |
|---|---|
| `arbit-canonical-block-c-frame-identity-tests` | 320 notes and 480 links across separated chords; late content beyond both source prefixes; preview/export row equality; cache/seek behavior; malformed nonresident records; 160 sounding notes and 560 source links packed to 128/256 resident rows with exact endpoint integrity. |
| `StateSerializationIntegrationTests '[video][score][resident-capacity]'` | The real processor projects a three-note clip through 129 passes to 387 notes and 258 structural links, serializes the full vectors and selects the first and last chord through canonical packing. |

Executed during review:

* `python3 video-helper/tests/block_c_note_instancing_source_contract.py -v`: six tests passed, including error publication, retry pacing and recovery wiring. An existing literal assertion initially failed against the base revision's particle-overlay conditional; it now checks both the overlay and receipt branches explicitly.
* `python3 video-helper/tests/viewport_telemetry_source_contract.py video-helper`: passed.
* An independent Python fixture arithmetic and source-preservation audit passed 34 checks. It checked sparse/dense expected selections, late link identities, loop counts, unchanged resident packing, unchanged frame/lifecycle validation and retained malformed-source rejection conditions.
* `git diff --check`: passed.

No C++ compilation, native renderer, app or export execution was run in this isolated worktree. Prime owns the compiled targets and restored-project preview/export acceptance. The saved zero-frame receipt remains failure evidence until that acceptance runs on the integrated fix.
