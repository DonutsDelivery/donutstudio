# Raw Render 3D pass files, version 1

Select the `Export ... Data` controls in a Render 3D node, save the graph, and use the ordinary video export dialog. The helper renders the saved selection from the same native scene frame used by the compositor. It writes a sidecar directory named `<video filename>.passes` beside the video. With every pass selection and HDR images off, export creates no sidecar and uses the existing video path.

The sidecar contains one lossless NumPy `.npy` array for each selected pass, clip and output frame, plus `manifest.json`. NPY is written directly; NumPy is not a runtime dependency. A reader may use `numpy.load(path, allow_pickle=False)`.

## File and data conventions

Files are named `clip-<clipId>-render-<stableRenderId>-frame-<frameIndex>-<output>.npy`. Frame indices start at zero for the requested export range. The manifest records the display-timeline time, FPS, clip ID, stable Render 3D ID, backend, Image Output selection, relative filenames, dtypes and shapes. It contains no asset paths. The output tokens are the existing `RenderPassOutputContract` tokens.

Arrays use NPY 1.0, C order and shape `[height, width, channels]`. Scalar values are explicitly little endian; byte arrays have no byte order. Rows run from top to bottom on both OpenGL and Metal. There is no rescaling, display encoding or floating-point conversion during file writing. Metal BGRA Image bytes are reordered to RGBA.

| Output | NPY dtype | Channels | Meaning |
|---|---|---|---|
| `color` | `\|u1` or `<f2` | RGBA | Ordinary SDR Image uses bytes (divide by 255). With a linear-input HDR profile, native Scene3D Image/Emission uses half-float linear sRGB; the file entry declares `primaries: srgb-bt709-d65` and `transfer: linear`. |
| `depth` | `<f4` | Depth | Camera near-to-far normalized linear depth in `[0,1]`, not world-distance units. Background is 1. |
| `normal` | `<f2` | XYZW | Signed world-space XYZ. W is 1 on geometry and 0 on background. |
| `motion` | `<f2` | XY | Current minus previous pixel position. Positive Y points upwards despite top-first row storage. The reset sample is zero. |
| `emission` | `<f2` | RGBA | The native linear emission attachment; values above one survive. |
| `mask` | `\|u1` | Coverage | UNORM8 coverage. Divide by 255. |
| `materialId` | `<u4` | ID | Full unsigned 32-bit scene/material ID. Zero is background. |
| `objectId` | `<u4` | ID | Full unsigned 32-bit scene/object ID. Zero is background. |

Passes describe the scene before clip effects, opacity, transforms, transitions, overlays and final timeline compositing. Only visible scene samples gathered by the export frame loop produce files. A transition's contributing scene can produce its own clip files. A clip rendered more than once at the same timeline frame shares one file set. Intermediate warmup samples and preview/probe requests never write files.

## Completion and cancellation

The helper creates the directory exclusively and refuses an existing destination. It streams the manifest to bound memory use, writes each selected attachment separately and preserves the native frame owner through readback. A duplicate clip/render/frame filename is an error.

The export job uses its existing partial video path for the partial sidecar. After video, audio and trailer completion, the helper closes the manifest and writes `.complete`. The app moves the sidecar and video to their final names. If video promotion fails, it rolls the sidecar back; a rollback failure reports the retained location. The filesystem cannot make both renames one atomic operation.

Cancellation or a failed sample removes the helper-owned partial sidecar. `.donutstudio-raw-pass-export` identifies that ownership for cleanup after a helper interruption. Cleanup does not adopt or remove a foreign sidecar. The final destination is never overwritten.

## Supported frames and remaining limits

The retained native Scene3D renderer supports static imported scenes, retained generated mesh/instance scenes and composed scenes. The exact existing attachment formats above are required. Invalid descriptors, unsupported formats, missing owners and failed GPU readback fail export explicitly.

Reactive `geometry.core.runtime` and harmonic-link renderer paths do not yet transport these selections. Their Render 3D raw export controls diagnose this limitation rather than silently dropping the selection. Diffraction frames reject raw export. Backends that do not expose the requested attachment/readback also reject it, including the separate Metal skin/morph frame implementation.

Motion uses the existing predecessor/reset authority. Previous deforming vertices and per-note transforms are not supplied by this change, so their existing Motion restrictions remain. The video still uses the existing SDR BT.709 encoder path.

## Linear HDR final images

The export dialog's `HDR images` selector adds a final-composite image sequence to the same bundle. The selected profile persists in project state, including the embedded state in `.arbit` files, and export retry settings. RPC callers pass `hdrImageProfile` in `video_export` or the helper export job. Omission means `off`, preserving existing SDR behavior.

| Profile | Declared final-compositor input | Saved linear output primaries |
|---|---|---|
| `linear-srgb` | Linear sRGB | sRGB / BT.709, D65 |
| `linear-p3` | Linear sRGB | Display P3, D65 |
| `linear-rec2020` | Linear sRGB | Rec.2020, D65 |
| `srgb-linear-srgb` | sRGB transfer, sRGB primaries | sRGB / BT.709, D65 |
| `srgb-linear-p3` | sRGB transfer, sRGB primaries | Display P3, D65 |
| `srgb-linear-rec2020` | sRGB transfer, sRGB primaries | Rec.2020, D65 |

These profiles require the user to declare the input interpretation. Legacy compositions may mix differently encoded sources; the export cannot infer their intended working space. A wider output gamut does not invent missing source colors or recover clipped highlights.

OpenGL reads the current RGBA16F compositor texture as float32. Metal reads its owned RGBA16F texture on the renderer's existing command queue and converts half values to float32. Capture happens after composition, bloom and exposure, but before the final display clamp or tone map. No second render advances animation or feedback. The color conversion decodes the selected input transfer and uses the existing Color Transform matrices to change primaries. Signed RGB and values above one survive. The output transfer is always linear; no absolute luminance or HDR display mastering claim is made.

Each `final-frame-<frameIndex>-linear.npy` is little-endian float32 RGBA, top-first, with shape `[height,width,4]`. Its manifest entry declares `stage: timeline-final-composite`, the input interpretation, output primaries, linear transfer, opaque alpha, source precision, timeline time and FPS. Scene pass entries retain their separate `scene-before-clip-composite` stage. Warmup and frame probes never write files.

The companion video is a real SDR preview of the same captured values. It applies per-channel Reinhard in linear sRGB, then the BT.709 output transfer before the existing SDR encoder. This fixed reference transform replaces the ordinary final tone map only for HDR-image exports. The live viewport remains its existing SDR presentation, so it does not establish HDR-monitor preview parity. The image sequence and reference video share the existing one-frame export publication/encoding contract and partial-bundle promotion/cleanup.

Current limits are an opaque canvas, finite pixels, a supported native GPU compositor and dimensions at most 4096 by 4096. Transparent outputs, CPU `minterpolate`, unknown profiles and unsupported formats fail explicitly. PQ/HLG HDR video and direct HDR-monitor presentation remain unavailable.

## Native linear Scene3D capture

The `linear-*` profiles request RGBA16F linear-sRGB color from the existing native imported/generated Scene3D renderer, including its deformation, material, diffraction, reactive Geometry Core and harmonic-link routes. glTF base-color and emission textures undergo their existing sRGB decode exactly once; material factors, lighting and emission stay linear. Render Pass Composite retains its linear working result and bypasses its final display transform during capture. The final image conversion applies the selected output primaries; this is not a metadata-only gamut change.

The default graph and live viewport remain SDR. HDR capture does not change persisted Render 3D port identities or relabel byte textures. Helper frame probes using the same HDR export profile receive the same reference conversion without writing sequence files. An sRGB-input profile rejects a linear native scene. A missing or mismatched native precision/transfer declaration also rejects export.

Multiple explicitly linear native scenes may composite with ordinary source-over, opacity, transforms, crop and alpha mattes. Mixed byte/video/shader/particle/text layers, transitions, adjustment layers, SDR effects/LUTs, graph color/temporal passes and non-normal blend modes reject this working representation until their color contracts support it. The rejection includes combinations that could clip highlights; it is not an assertion that every legacy effect is linear-safe. Data-pass Image visualizations other than Emission do not masquerade as scene color.

Graph material Frames require an explicit sRGB primaries/transfer declaration before the native HDR material sampler accepts them. Current decoded Frame leases do not establish that transfer, so HDR Video on 3D Screen and card imagery reject them; their existing SDR workflows remain available. An explicit verified sRGB Frame uses the existing real shader decode. No guessed BT.709-to-sRGB metadata is added. General mixed-working-space composition and transfer-aware image/video decoding remain open.
