# Private Main10 export bridge and pending acceptance

`exporter.cpp` selects `donut-kvazaar10` for explicit software H.265 HDR movie
jobs only when `ARBIT_HAVE_KVAZAAR10_EXPORT=1` and both requested dimensions
are divisible by eight. The adapter encodes those visible dimensions exactly.
Other extents retain the existing x265 admission and encoder path. A build
without an eligible x265 encoder rejects them rather than changing their
extent to fit Kvazaar. Existing odd-dimension normalization belongs to that
older path and is unchanged.

The bridge depends on the adapter header and implementation from checkpoint
`676e31852001c9cec978c48f5f80ef8e9a188129`. CMake must link
`arbit-video-helper` privately to `arbit-kvazaar10-adapter` and define
`ARBIT_HAVE_KVAZAAR10_EXPORT=1` on the helper only when that target is admitted.
The adapter keeps `KVZ_BIT_DEPTH=10` and its matched Kvazaar include directory
private. Neither flag nor include directory belongs on the plugin or exporter.
Without the bridge definition, production export keeps its prior selection.

The unopened FFmpeg context passed to the renderer describes ten-bit frames,
color and the reciprocal rational frame rate. The private adapter owns
encoding. Its codec parameters and Annex B headers reach the existing
FFmpeg stream and container. Packets preserve encoder PTS, DTS and duration,
then use the existing timestamp rescale and interleaved writer. Null-frame
calls continue until the adapter explicitly reports drained. Cancellation
releases buffered pictures and the export owner removes the partial movie.
The existing audio owner and AAC stream remain in that same container.

No mastering-display or content-light metadata is authored by the current
HDR output contract. `peakNits` controls output conversion and admission. The
bridge does not turn it into ST 2086, MaxCLL or MaxFALL.

## Independent linear input

`tests/fixtures/hdr-movie-linear-emission-oracle.json` specifies a 128 by 64,
17-frame calibration at 30000/1001 fps. It uses opaque Scene3D emissive panels
with exact binary16 linear-sRGB values. Interior pixels in the left, center
and right panels supply reference white, a 913.5-nit neutral highlight and a
color with distinct channels. The expected values come from authored
emission, never a captured rendered image. Sample frames 0, 8 and 16 cover
initial, middle and delayed final output.

A matching actual compiled Scene3D helper job must be saved from the product's
normal authoring route. Make the panels fill their sample regions, disable
ambient/light response, textures, post effects and tone mapping, and use the
exact oracle extent and frame rate. Give every segment enough range for all
17 frames. This checkpoint provides the oracle and acceptance driver; it
does not contain or claim an accepted saved scene job.

The runtime driver requires the real native OpenGL or Metal compositor and
fully decodes the production movie through independent FFmpeg tools. It
checks Main10, limited Rec.2020 PQ/HLG VUI, exact visible extents and crop,
frame and packet cardinality, rational cadence, real increasing DTS and all
presentation times, decoded luminance/color tolerance, rejected HDR routes
and cancellation cleanup. With `--with-audio`, a generated 997-Hz tone must
survive AAC muxing and independent decode. It writes a receipt binding the
helper, saved scene job, linear oracle and output movie hashes only after both
transfer routes and cancellation checks pass. Accepted movies are retained beside
the receipt, with decoded sample values and packet PTS/DTS recorded in it.

Run against the integrated and built opt-in helper:

```bash
python3 video-helper/tests/pq_movie_export_runtime_fixture.py \
  /absolute/path/to/arbit-video-helper /absolute/path/to/saved-scene-job.json \
  video-helper/tests/fixtures/hdr-movie-linear-emission-oracle.json \
  --expected-encoder donut-kvazaar10 --with-audio \
  --receipt /absolute/path/to/kvazaar10-hdr-movie-receipt.json
```

Run the same authored scene and oracle against a maintained helper without
the bridge definition using `--expected-encoder libx265` and a separate
receipt. Exercise an even extent not divisible by eight to establish its
existing fallback or explicit rejection in the opt-in build. Adapt and
independently respecify sample coordinates if that changes scene coverage.

Compilation, real exports, saved-job preparation and decoded acceptance are
pending. The adapter's fake API fixture establishes ownership behavior only.
Linear image-sequence exports do not establish HDR HEVC movie acceptance.
