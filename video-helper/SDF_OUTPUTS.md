# Native SDF output maps

`SDF Raymarch` selects these maps through its existing `outputPass` control and Image port. OpenGL and Metal evaluate the same admitted IR, primary ray, camera and light. The IR wire schema and graph port layout are unchanged. The native target is RGBA16F; the existing Image presentation can quantize it to bytes. These are visualizations, not new raw data attachments.

`sdfgraph::RaymarchOutputPass` in `plugin/Source/graph/SdfGraphNodes.h` and `arbitgpu::NativeSdfOutput` share the eight selections below. The wire's `outputPass` carries that selection. The separate `renderpassoutput::Output` enum describes attachment semantics: its value 3 is Motion and its MaterialId is 6. Do not cast a SDF selection to that enum. Every SDF selection here still renders to the existing Color attachment and Image port.

Coordinates and distance keep the contract in `shared/SdfIr.h`. They use right-handed scene units, positive X right, positive Y up and positive Z toward the camera. Negative field values are inside geometry. The current camera is at `[0,0,3]`, with normalized ray `[u,v,-1.8]`. Pixel centers determine `u,v`, scaled by image height. The directional light remains normalized `[-0.45,0.75,0.6]`.

| Selection | Value and interpretation |
|---|---|
| Color, 0 | Existing directional shading, rim response and soft shadow. |
| Depth, 1 | Existing primary-ray travel divided by Maximum Distance, clamped to `[0,1]`. A miss is one. These are normalized ray distances, not Scene3D camera-space depth values. |
| Normal, 2 | Existing world-space finite-difference normal encoded as `0.5 + 0.5*n`. |
| Material ID, 3 | RGB diagnostic of the actual contributing primitive's full 64-bit stable ID. Unary domain operations preserve the leaf identity. Union selects the smaller signed distance; intersection selects the larger; subtraction compares A with negative B, so a cut belongs to B. Smooth Booleans choose the contributor with the larger blend weight. Ordered input A wins equal weights. |
| Curvature, 4 | Signed mean curvature `H = 0.5*div(n)` in inverse scene units, displayed as `0.5 + 0.5*H/(1+abs(H))`. A plane is 0.5, a convex unit sphere is about 0.75, and a concave unit sphere is about 0.25. This compresses the signed measurement into an Image; it is not raw curvature data. |
| Ambient Occlusion, 5 | Local normal-probe visibility. One is open and zero is occluded. Nearby geometry reduces the distance along the outward normal. This is a bounded geometric approximation, not hemisphere path tracing. |
| Soft Shadow, 6 | Visibility toward the same light used by Color. One is unblocked and zero is shadowed. Back-facing surfaces are zero. The map excludes diffuse color and cosine intensity. |
| Edge Distance, 7 | Estimated distance from the primary hit to its nearby rendered silhouette in output pixels, divided by eight. One means no sampled silhouette within eight pixels. The estimate uses eight screen directions and the same bounded ray-hit test as the primary image. |

All five utility maps use black for a missed primary ray and alpha one. Finite-difference normals with a zero or nonfinite gradient use positive Z. Nonfinite ray samples stop the ray; nonfinite occlusion or shadow samples return zero visibility. Native acceptance reads float pixels before clamping or byte conversion and rejects nonfinite or out-of-range map values.

## Material identity visualization

The native compiled record retains the exact stable ID. Its color is derived on the CPU without evaluating geometry. Starting with unsigned 64-bit `id`, apply the following wrapping arithmetic:

```text
id = (id xor (id >> 30)) * 0xbf58476d1ce4e5b9
id = (id xor (id >> 27)) * 0x94d049bb133111eb
id = id xor (id >> 31)
rgb24 = (id & 0xffffff) | 0x202020
```

The low, middle and high bytes are R, G and B, divided by 255. All 24 color bits fit exactly in the OpenGL uniform float. Metal receives the same code as a uint. Colors therefore survive record reordering and include high ID bits. The visualization has possible color collisions and cannot recover a full 64-bit ID. It does not claim to implement a raw integer Material ID attachment or the separate scene-module material table.

## Sampling bounds

The existing quality scale is 4, 2, 1 and 0.5 for Low through Ultra. Normal epsilon is the larger of `epsilon * normalQualityScale` and `1e-6` scene units. Curvature differentiates those normals at six offsets with stencil width `max(4*epsilon*normalQualityScale, 0.002, length(hit)*0.0001)` scene units. The result is bounded to plus or minus the reciprocal stencil width before display mapping. Sharp CSG seams and non-distance-preserving deformations remain finite-difference estimates.

AO uses 4, 8, 12 or 16 normal probes according to Normal Quality. Its radius is `min(maximumDistance, max(0.5, 32*epsilon))` scene units. Samples are equally spaced over that radius; each later sample has 0.75 times the preceding weight. A sample contributes `clamp(1-distance/reach, 0, 1)` to weighted occlusion. Visibility is one minus the weighted average.

Soft Shadow retains Color's existing trace. Shadow Quality allows 8, 16, 32 or 64 steps, up to `min(maximumDistance,8)` scene units. The origin is displaced by `4*epsilon` along the surface normal. The trace starts at `4*epsilon` along the light ray, uses the distance-over-travel visibility estimate with factor 12, and bounds each step between `2*epsilon` and 0.25 scene units.

Edge Distance first probes eight pixels away in each of eight compass directions. A missing endpoint brackets a silhouette, then four bisections narrow that bracket to half a pixel. The smallest upper bracket is returned. The cost is at most 40 additional rays per foreground pixel; each respects the existing Maximum Steps, epsilon, adaptive quality and Maximum Distance. It is a local directional estimate, not a global Euclidean distance transform. A thin gap whose sampled endpoint hits geometry again can be missed. The image border itself is not a silhouette, so an infinite plane remains one even at the border. Resizing changes the pixel distance.

The existing limits of 64 compiled records, depth 32, 256 expanded evaluation nodes and 512 primary steps remain. Unsupported outputs, invalid enums, nonfinite controls and exceeded bounds reject before native frame allocation. No CPU renderer or additional GPU attachment was added.

## Resource and frame cost

Let `S` be Maximum Steps, capped at 512. A field query evaluates the admitted expression once. On a foreground pixel the upper bounds are:

| Selection | Field queries per pixel |
|---|---|
| Depth | `S` |
| Material ID | `S + 1` |
| Normal | `S + 6` |
| Curvature | `S + 36` |
| Ambient Occlusion | `S + 6 + probes`, with 4 to 16 probes |
| Color or Soft Shadow | `S + 6 + shadowSteps`, with 8 to 64 shadow steps |
| Edge Distance | `41*S`, including the primary ray and at most 40 additional rays |

Misses stop after the primary ray. Depth, Material ID, Curvature and Edge Distance skip the otherwise unused normal calculation. The interpreter's 32-entry stacks include one extra integer contributor per level; there is no recursion or image-sized material buffer. A valid expression with `N` expanded nodes requires `2*N-1` dispatcher iterations, at most 511, inside the unchanged 768-iteration shader ceiling. Admission checks the depth starting at one, so the maximum stack index is 31. Child indices and the root are checked against the 64-record array before submission.

At the maximum step setting Edge Distance can issue 20,992 field queries per foreground pixel. Together with the maximum expression this permits 10,726,912 dispatcher iterations per pixel. These bounds prevent unbounded evaluation; they do not establish interactive frame times. A dense scene at the 4096-by-4096 extent cap can be expensive, especially for Edge Distance. Native timing and device watchdog behavior at large extents remain unmeasured in this checkpoint; no quality setting or admission bound is silently reduced.

The shared target remains eight bytes per pixel, at most 128 MiB at 4096 by 4096. Material colors occupy the existing fourth record component, so GPU record storage remains 48 bytes per record. The OpenGL uniform payload is 3,112 bytes (three arrays plus four float and six integer scalar components); the aligned Metal block remains 3,120 bytes. Host compiled records retain a `uint64_t` ID, and receipts and the 32-entry compiled cache account for them using `sizeof(NativeSdfCompiledRecord)`. The existing receipt counts shader source bytes; it does not measure the driver's compiled program or register/private-stack allocation.

Float readback is an explicit diagnostic operation. Metal additionally allocates a staging buffer with `alignUp(width*8,256)*height` bytes and a CPU array with `width*height*16` bytes. These transient diagnostic allocations are outside the retained frame receipt and absent during ordinary composition. Failed readback clears its output; native assertions inspect the float values before any byte clamping.

## Verification targets

The existing reference target `arbit-sdf-reference-evaluator-tests` retains all 26 operation distance goldens and adds analytic plane/sphere/cavity curvature, corner/open AO, occluder/light visibility, contributor ownership, high-ID-bit color goldens and silhouette pixel-unit checks. The scalar reference reads the admitted IR, not the native compiled program or shader output.

`arbit-sdf-native-render-admission-tests` checks positive admission and independent missing-capability rejection for every output. `arbit-opengl-sdf-backend-tests` and physical-Mac `arbit-metal-backend-tests` retain the existing 26-operation and polar-repeat parity cases. They add float-pixel semantic comparisons for all eight outputs, preview/export equality, hard and smooth Boolean contributors, reversed-input ties, record-order stability and invalid-output/control rejection. Negative enum casts, Count and an out-of-range value reject. The scalar oracle computes smooth contributor weights separately from the shaders' distance ordering and rejects invalid controls before quality indexing. `SdfGraphLoweringTests` remains the graph-control regression target.

These are prepared executable checks. The implementation checkpoint does not itself claim a compiled, native or actual-editor passing receipt.

During the 23 September 2026 review, a standalone Python mathematical fixture check passed 82 assertions using analytic sphere, plane and box fields. It checked curvature, open/corner occlusion, blocked/unblocked light visibility, projected silhouettes and resizing, full-width material hash goldens and float transport, smooth blend dominance and the visibility/ties of the added CSG probes. Measurements included sphere curvature `0.999966002`, cavity curvature `-1.999728068`, corner AO `0.778149918` and a near-silhouette estimate of one pixel against an analytic distance of `0.500536` pixels. This executed Python mathematics only, not the C++ oracle or either shader.

After crash recovery, the worktree still had the original `4f8b6b4519979397b6a0af49060565bbbb8f1189` HEAD and all 14 expected modified or new paths, with no staged files or final commit. The recovery review passed `git diff --check` and a Python source audit of the eight graph/native enum values, both shaders' bounded arrays and loops, unchanged primitive/domain/smooth-distance functions, unchanged 26-operation goldens, all 21 semantic cases, invalid-enum coverage and existing CMake source inclusion. The same audit checked the frame-cost arithmetic above. Shader syntax received source review only; C++ and GLSL/MSL compilation, native pixels, application acceptance and timing remain pending with the integration owner.
