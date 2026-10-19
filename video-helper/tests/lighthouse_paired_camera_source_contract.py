#!/usr/bin/env python3
"""Check that both backends route the paired water and caustic volume camera transport.

The shared shader recipe and the per-draw uniform values are covered by
lighthouse_caustic_volume_tests.cpp. This contract covers the backend source that
CPU tests cannot reach: the second dielectric uniform in both upload paths, the
paired fragment text only for a caustic volume preparation, the volume draw
routed by smoothglass::cameraTransport, and the Metal uniform layout.
"""

from pathlib import Path
import re
import sys

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
opengl = (root / "src/gpu_backend/backend_opengl.cpp").read_text(encoding="utf-8")
sokol = (root / "src/gpu_backend/backend_sokol.mm").read_text(encoding="utf-8")
header = (root / "src/gpu_backend/fixture_smooth_glass.h").read_text(encoding="utf-8")
ledger = (root / "src/gpu_backend/fixture_surface_source_admission.h").read_text(encoding="utf-8")

errors = []


def require(text, needle, label):
    if needle not in text:
        errors.append(label + " is missing: " + needle)


def require_count(text, needle, expected, label):
    found = text.count(needle)
    if found != expected:
        errors.append(label + " has " + str(found) + " of " + str(expected) + ": " + needle)


# OpenGL: a feature bit outside AllFeatures, set only for a caustic volume preparation.
# Bit 2 belongs to the Surface Material diffraction layer, which AllFeatures includes.
require(opengl, "kFixtureFragmentStamLayer = 1u << 2,", "OpenGL feature bits")
require(opengl, "kFixtureFragmentCausticVolume = 1u << 3,", "OpenGL feature bits")
require(opengl, "kFixtureFragmentAllFeatures = kFixtureFragmentSmoothGlass | kFixtureFragmentDiffraction | kFixtureFragmentStamLayer,",
        "OpenGL AllFeatures")
all_features = re.search(r"kFixtureFragmentAllFeatures = ([^;]*?),\n", opengl)
if all_features is None or "kFixtureFragmentCausticVolume" in all_features.group(1):
    errors.append("AllFeatures must not include kFixtureFragmentCausticVolume")
bits = re.findall(r"kFixtureFragment\w+ = 1u << (\d+),", opengl)
if len(bits) != len(set(bits)):
    errors.append("OpenGL fragment feature bits collide: " + ", ".join(bits))
require(opengl, "| (causticVolume != nullptr && distinctGrating == nullptr ? static_cast<unsigned> (kFixtureFragmentCausticVolume) : 0u);",
        "OpenGL fragment feature mask")
require(opengl, "const bool causticVolume = smoothGlass && (features & kFixtureFragmentCausticVolume) != 0;",
        "fixtureFragmentShaderSource")
require(opengl, "smoothglass::transportShader (false, true, false, causticVolume)", "fixtureFragmentShaderSource")
require_count(opengl, "smoothglass::transportShader (false, true)", 0, "fixtureFragmentShaderSource unpaired call")
# The uniform is looked up, required only by the paired variant, and uploaded per draw.
require(opengl, 'uniforms.glassTransport2 = location("uGlassTransport2");', "OpenGL uniform lookup")
require(opengl, "&& (! causticVolume || glassTransport2 >= 0));", "OpenGL complete()")
require(opengl, "smoothglass::cameraTransport(resources->smoothGlass, drawObject.id,", "OpenGL per-draw transport")
require(opengl, "smoothglass::cameraPartner(resources->smoothGlass, drawObject.id).data());", "OpenGL per-draw partner")
require_count(opengl, "if (locations.glassTransport2 >= 0)", 2, "OpenGL partner uploads")
require_count(opengl,
              "drawCoating[3], float(resources->smoothGlass.object.value & 65535u), float(resources->smoothGlass.object.value >> 16u));",
              0, "OpenGL per-draw released transport id")

# Metal: the same routing, through both fragment composition paths.
require(sokol, "const bool pairedVolume=causticVolume!=nullptr && distinctGrating==nullptr;", "Metal paired flag")
require(sokol, "restrictedSurfaceSourceRecipe<surfacesource::Source>(pairedVolume);", "Metal restricted source")
require(sokol, "sourceCreation->pairedVolume=pairedVolume;", "Metal source creation")
require(sokol, "+arbitgpu::smoothglass::transportShaderRecipe<String>(true,true,false,pairedVolume)", "Metal restricted recipe")
require(sokol, "smoothglass::transportShader(true,true,false,pairedVolume)", "Metal fragment source")
require(sokol, "smoothglass::transportShader(true,true,false,creation.pairedVolume)", "Metal surface source oracle")
require_count(sokol, "if(pairedVolume) declarePairedFixtureUniform(", 2, "Metal paired fragment declarations")
require(sokol, "if(creation.pairedVolume) declarePairedFixtureUniform(fragment);", "Metal surface source oracle")
require(sokol, "smoothglass::cameraTransport(resources->smoothGlass, drawObject.id,", "Metal per-draw transport")
require(sokol, "smoothglass::cameraPartner(resources->smoothGlass, drawObject.id);", "Metal per-draw partner")
require(sokol, "std::copy(partner.begin(), partner.end(), uniforms.glassTransport2);", "Metal partner upload")
require(sokol, "std::fill_n(uniforms.glassTransport2, 4, 0.0f);", "Metal partner reset")
require_count(sokol, "uniforms.glassTransport[2] = float(resources->smoothGlass.object.value & 65535u);", 1,
              "Metal released transport id (whole-scene path only)")

# Metal layout: glassTransport2 is the last FixtureUniforms member, after the
# diffraction layer block (stamLayer, 7 vec4 rows), and the paired fragment
# declares it last too.
fixture = re.search(r"struct FixtureUniforms\n\{(.*?)\n\};\nstatic_assert", sokol, re.S)
if fixture is None:
    errors.append("FixtureUniforms is missing")
else:
    members = [line.strip() for line in fixture.group(1).splitlines() if line.strip() and not line.strip().startswith("//")]
    if members[-1] != "float glassTransport2[4];":
        errors.append("glassTransport2 must be the last FixtureUniforms member, found " + members[-1])
require(sokol, "&& offsetof (FixtureUniforms, stamLayer) == 11536", "FixtureUniforms layout")
require(sokol, "&& offsetof (FixtureUniforms, glassTransport2) == 11648", "FixtureUniforms layout")
require(sokol, "&& sizeof (FixtureUniforms) == 11664,", "FixtureUniforms layout")
require(sokol, '"    float4 vertexDisplacementLimits[16];\\n    float4 stamLayer[7];\\n    float4 glassTransport2;\\n};"',
        "Metal paired declaration")
# A layered program never takes the restricted archive, and composes with a volume in the general source.
require(sokol, "SurfacePbr) && !layered;", "Metal restricted archive excludes layered programs")
require(sokol, "+(layered ? arbitgpu::stamlayer::metalShader() : std::string()));", "Metal layered module in the general source")

fragment = re.search(r'constexpr char kFixtureFragmentShader\[\] = R"metal\((.*?)\)metal";', sokol, re.S)
if fragment is None:
    errors.append("kFixtureFragmentShader is missing")
else:
    text = fragment.group(1)
    # The released text ends at the pre-layer prefix; the paired declaration
    # adds stamLayer and glassTransport2 (see metal_fixture_released_shader_contract.py).
    require_count(text, "    float4 vertexDisplacementLimits[16];\n};", 1, "Metal fragment Uniforms end")
    require_count(text, "stamLayer", 0, "released Metal fragment")
    require_count(text, "glassTransport2", 0, "released Metal fragment")
    uniforms = re.search(r"struct Uniforms \{\n(.*?)\n\};", text, re.S)
    sizes = {"float4x4": 64, "float4": 16, "uint4": 16}
    offset = 0
    if uniforms is None:
        errors.append("Metal fragment Uniforms is missing")
    else:
        for line in uniforms.group(1).splitlines():
            match = re.fullmatch(r"\s*(float4x4|float4|uint4) (.*);", line)
            if match is None:
                errors.append("unparsed Metal Uniforms member: " + line)
                continue
            for name in match.group(2).split(","):
                count = re.search(r"\[(\d+)\]", name)
                offset += sizes[match.group(1)] * (int(count.group(1)) if count else 1)
        # 11536 + 7 stamLayer rows of 16 bytes puts glassTransport2 at 11648.
        if offset != 11536:
            errors.append("Metal Uniforms ends at " + str(offset) + ", the paired stamLayer must follow at 11536")

# The paired recipe has its own Metal stage budget, and the Lighthouse pair
# admits frames against the closed-volume query ceiling.
require(ledger, "static constexpr std::size_t pairedVolume=8;", "surface source StageBudget")
require(ledger, "+StageBudget::spectral+StageBudget::transport+StageBudget::pairedVolume;", "surface source Plan")
require(header, ": prepared.causticVolume.present ? kMaximumPixelRayQueries/kPairPixelRayQueries", "admitFrame")
require(header, "inline constexpr std::size_t kMaximumPairRayVisits = 16;", "pair walk visit bound")
require(header, "inline constexpr std::uint64_t kPairPixelRayQueries = 2 + 3*kMaximumPairRayVisits;", "pair pixel query count")
require(header, "static_assert(kMaximumPairRayVisits == 16,", "pair walk visit assertion")
require(header, "static_assert(kMaximumRayVisits == 64,", "glassTransport visit assertion")
require_count(header, "for(int visit=0;visit<48;++visit) {", 1, "glassPairWalk visit loop")
require(header, "bool pairedVolume = false)", "transportShaderRecipe")

# The paired walk visits the heaviest pending branch first. Branches left at the
# visit or slot bound end without a query: in air through the environment,
# inside the volume as truncated weight. A channel past the tolerance refuses.
tolerance = re.search(r"inline constexpr float kPairTruncatedEnergyTolerance = ([0-9.]+)f;", header)
define = re.search(r"\n#define GLASS_PAIR_TRUNCATED ([0-9.]+)\n", header)
if tolerance is None or define is None:
    errors.append("the pair truncation tolerance is missing from C++ or the shader")
elif float(tolerance.group(1)) != float(define.group(1)) or float(define.group(1)) != 0.02:
    errors.append("the pair truncation tolerance differs: " + tolerance.group(1) + " and " + define.group(1))
walk_at = header.find("vec3 glassPairWalk(")
walk = header[walk_at:header.find("\n}\n", walk_at)] if walk_at >= 0 else ""
require(walk, "for(int slot=1;slot<pending;++slot) if(weights[slot]>weights[heaviest]) heaviest=slot;",
        "glassPairWalk heaviest-first selection")
require(walk, "float weight=weights[heaviest];", "glassPairWalk heaviest-first selection")
require(walk, "for(int slot=1;slot<16;++slot) if(weights[slot]<weights[lightest]) lightest=slot;",
        "glassPairWalk lightest spill when the slots are full")
require(walk, "if(truncated>GLASS_PAIR_TRUNCATED&&spilled)\n            {status=GLASS_SMOOTH_STATUS_PENDING;return vec3(0.0);}\n        if(truncated>GLASS_PAIR_TRUNCATED) trapped=true;",
        "glassPairWalk spilled refusal and trapped absorption")
require_count(walk, "GLASS_SMOOTH_STATUS_VISITS", 0, "glassPairWalk visit refusal")
require_count(walk, "if(pending!=0) {status=GLASS_SMOOTH_STATUS_VISITS;", 0, "glassPairWalk first-unvisited refusal")
require_count(walk, "glassSceneClosestHit(", 1, "glassPairWalk ray queries")
require_count(walk, "glassPairLeftover(", 4, "glassPairWalk leftover branches")
require(header, "    return inside?vec3(0.0):weight*glassEnvironment(direction GX_PASS);",
        "glassPairLeftover environment for a branch in air")
# Nondispersive glass: one walk, all three channels, the pooled 48 visits.
# Dispersive glass: three walks of 16. The water IOR has no dispersion.
require(walk, "bool sharedPaths=volumeDispersion==0.0;", "glassPairWalk shared paths")
require(walk, "int walks=sharedPaths?1:3,visits=sharedPaths?48:16;", "glassPairWalk visit budgets")
require(walk, "if(visit>=visits||pending==0) break;", "glassPairWalk visit budgets")
require(header, 'error = "Water interface refuses dispersion"', "water dispersion refusal the shared walk relies on")
# Total internal reflection in the cube reaches the depth bound with little
# weight left; that branch ends through the leftover rule.
require_count(walk, "GLASS_SMOOTH_STATUS_DEPTH", 0, "glassPairWalk depth refusal")
require(walk, "if(depth>=GLASS_INTERFACE_DEPTH) {\n                radiance+=lanes*glassPairLeftover(rayDirection,weight,rayInside GX_PASS);\n                if(rayInside) truncated+=weight;",
        "glassPairWalk depth leftover")
require_count(walk, "radiance[channel]", 0, "glassPairWalk per-channel radiance writes")


def pair_walk(panes, reflectance, heaviest, visits=16, slots=16, limit=0.02):
    """Bookkeeping of one glassPairWalk channel along a line of parallel panes.

    Interfaces 0..2*panes-1 split the line into regions; odd regions are glass.
    The camera ray starts left of every pane, and a branch leaving either end
    meets the environment. Each interface reflects `reflectance` of the branch
    weight. heaviest=False is the earlier last-in-first-out walk that refused
    on any branch left unvisited. Returns (status, environment, truncated, queries).
    """
    end = 2 * panes
    pending = [(0, 1, 1.0)]
    environment = truncated = 0.0
    queries = 0
    spilled = False

    def leftover(branch):
        nonlocal environment, truncated
        if branch[0] % 2 == 1:
            truncated += branch[2]
        else:
            environment += branch[2]

    for _ in range(visits):
        if not pending:
            break
        if heaviest:
            at = max(range(len(pending)), key=lambda slot: pending[slot][2])
            branch = pending[at]
            pending[at] = pending[-1]
            pending.pop()
        else:
            branch = pending.pop()
        queries += 1
        region, step, weight = branch
        if (step > 0 and region == end) or (step < 0 and region == 0):
            environment += weight
            continue
        for child in ((region + step, step, weight * (1 - reflectance)), (region, -step, weight * reflectance)):
            if child[2] < 0.00001:
                continue
            if len(pending) >= slots:
                if not heaviest:
                    return ("PENDING", environment, truncated, queries)
                spilled = True
                lightest = min(range(slots), key=lambda slot: pending[slot][2])
                if pending[lightest][2] >= child[2]:
                    leftover(child)
                    continue
                leftover(pending[lightest])
                pending[lightest] = pending[-1]
                pending.pop()
            pending.append(child)
    if pending and not heaviest:
        return ("VISITS", environment, truncated, queries)
    for branch in pending:
        leftover(branch)
    if truncated > limit:
        return ("PENDING" if spilled else "TRAPPED", environment, truncated, queries)
    return ("COMPLETE", environment, truncated, queries)


# Two lantern panes at the 0.04 normal-incidence reflectance of n=1.5 glass:
# four interfaces, too many branches for 16 last-in-first-out visits.
before = pair_walk(2, 0.04, heaviest=False)
after = pair_walk(2, 0.04, heaviest=True)
if before[0] != "VISITS":
    errors.append("the pane model no longer reproduces the VISITS refusal: " + str(before))
if after[0] != "COMPLETE" or after[2] > 0.02 or after[3] > 16 or abs(after[1] + after[2] - 1.0) > 0.0001:
    errors.append("heaviest-first walk does not complete two panes within the tolerance: " + str(after))
# Four panes leave about 0.1 of the power inside the glass after 16 visits,
# five times the tolerance. The pixel is trapped: it keeps the environment it
# gathered, absorbs the rest, and only systemic trapping refuses the frame.
deep = pair_walk(4, 0.04, heaviest=True)
if deep[0] != "TRAPPED" or deep[2] <= 0.05:
    errors.append("four panes past the tolerance must be trapped: " + str(deep))

# The paired walk gives a non-finite final radiance the same status the water draw uses.
require(header,
        "    complete=glassSceneFinite3(radiance);\n    if(!complete) status=GLASS_SMOOTH_STATUS_THIN_RADIANCE;\n    else if(trapped) status=GLASS_SMOOTH_STATUS_TRAPPED;\n    return radiance;\n}\n// Water draw.",
        "glassPairWalk non-finite radiance status")

# SDR presentation. The caustic forces scene-linear RGBA16F Color, and the ordinary
# SDR viewport and export convert that layer to sRGB through nativeLinearImage.
# The imported scene receipt moves into the owner list before the layer publishes
# that image, so the frame must be held by value, not by reference into the receipt.
imported = (root / "src/imported_scene_visual_plan_execution.h").read_text(encoding="utf-8")
renderer_cpp = (root / "src/renderer.cpp").read_text(encoding="utf-8")
renderer_h = (root / "src/renderer.h").read_text(encoding="utf-8")
by_value = "const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> frame = Execution::nativeFrame(receipt);"
require(imported, by_value, "imported scene frame held by value")
require_count(imported, "const auto& frame = Execution::nativeFrame(receipt);", 0, "imported scene frame reference into the moved receipt")
moved = imported.find("frameOwners.push_back(std::move(receipt));")
published = imported.find("layer.nativeLinearImage = frame;")
captured = imported.find(by_value)
if not (0 <= captured < moved < published):
    errors.append("imported scene must capture the frame by value, move the receipt, then publish nativeLinearImage")
require(imported, "if (compiled->importedSceneRender->glassCaustic)", "caustic linear Color request")
for label, text in (("OpenGL", renderer_cpp), ("Metal", sokol)):
    prepare = text.find("displayInputs.prepare(")
    validate = text.find("validateLinearSceneComposite(")
    if not (0 <= prepare < validate):
        errors.append(label + " SDR composite must convert linear images before validateLinearSceneComposite")
require(renderer_h, "if (!source.nativeLinearImage) return true;", "SdrNativeImageLayers skips layers without a linear image")
require(renderer_h, "if (hdrCapture) return true;", "SdrNativeImageLayers keeps HDR capture linear")

# Glass Caustics off. The renderer derives a camera-only selection of the one
# closed volume beside one thin water interface and prepares it through the
# same paired-volume entry points, so Color and the material-ID view both take
# the paired camera walk. The cache key separates it from a caustic selection
# of the same volume, and caustic preparation refuses a camera-only slot.
fixture_renderer = (root / "src/fixture_scene_renderer.cpp").read_text(encoding="utf-8")
fixture_renderer_h = (root / "src/fixture_scene_renderer.h").read_text(encoding="utf-8")
caustics = (root / "src/gpu_backend/fixture_glass_caustics.h").read_text(encoding="utf-8")
require(header, "    bool cameraOnly = false;\n};", "CausticVolumeSelection and CausticVolume camera-only flags")
require(header, "inline std::optional<CausticVolumeSelection> cameraVolumeSelection(", "camera-only derivation")
require(header, "    if (thin != 1 || volumes != 1) return std::nullopt;", "camera-only derivation counts")
require(header, "slot.cameraOnly = causticVolume->cameraOnly;", "camera-only slot flag")
require(header, '"Camera glass volume beside water requires smooth water"', "camera-only rough water refusal")
require(header, '"Camera glass volume requires one thin water interface beside it"', "camera-only lone volume refusal")
require(header, '"Camera glass volume beside water requires a smooth closed volume"', "camera-only rough volume refusal")
require(fixture_renderer, "else if (! sceneInputs.glassCaustic && ! diffractionMaterial)", "renderer camera-only branch")
require(fixture_renderer, "causticVolume = arbitgpu::smoothglass::cameraVolumeSelection (*snapshot, nativeMaterial.get());",
        "renderer camera-only derivation")
require(fixture_renderer, "&& cache_.cameraVolumeOnly == requestedCameraOnly;", "renderer cache key")
require(fixture_renderer, "cache_.cameraVolumeOnly = requestedCameraOnly;", "renderer cache store")
require(fixture_renderer_h, "bool cameraVolumeOnly = false;", "renderer cache member")
derive_at = fixture_renderer.find("cameraVolumeSelection (*snapshot")
color_only = fixture_renderer.find("sceneInputs.imageOutput != renderpassoutput::Output::Color")
if not (0 <= color_only < derive_at):
    errors.append("the Color-only rule must stay inside the caustic branch, before the camera-only derivation")
if "imageOutput" in fixture_renderer[derive_at - 1200:derive_at]:
    errors.append("the camera-only derivation must not depend on the image output")
require(caustics, "if (slot.present && slot.cameraOnly)", "caustic preparation camera-only refusal")
require(opengl, "return backend->prepareGeometryInstancesImpl(scene,material,{},false,nullptr,nullptr,{},0,use,&selection);",
        "OpenGL paired-volume preparation")
require(sokol, "return backend->prepareGeometryInstancesOwned(scene,material,{},false,nullptr,nullptr,{},&selection);",
        "Metal paired-volume preparation")

# Glass Caustics off output encoding. The caustic route forces scene-linear
# RGBA16F Color and the SDR compositor converts it with the sRGB transfer. The
# camera-only route renders the ordinary SDR target, so without its own encode
# the whole frame arrived as the sRGB-decoded caustic frame. Both backends set
# pass control w for a camera-only SDR Color draw, compositePass applies the
# same transfer once when no pass program owns the output transform, and the
# background clear is encoded the same way.
require(imported, "linearColor = true;\n    }\n\n    if constexpr (!HasImportedParentInspector",
        "caustic route forces scene-linear Color")
require(header, "inline bool cameraOnlyDisplayEncode(const Prepared& prepared, bool linearColor,", "camera-only encode rule")
require(header, "return prepared.enabled && prepared.causticVolume.present && prepared.causticVolume.cameraOnly\n"
        "        && !prepared.distinctGrating && !linearColor && plainColorOutput && !resolvedFrame\n"
        "        && !blendedDraw;",
        "camera-only encode applies only to SDR plain Color")
require(header, "return value <= 0.0031308f ? value * 12.92f : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;",
        "camera-only clear encode")
encode_rule = ("smoothglass::cameraOnlyDisplayEncode(resources->smoothGlass,",
               "runtimeInputs.linearColor, runtimeInputs.imageOutput == renderpassoutput::Output::Color",
               "&& !runtimeInputs.passComposite && !runtimeInputs.passProgram,")
for label, text in (("OpenGL", opengl), ("Metal", sokol)):
    for needle in encode_rule:
        require_count(text, needle, 1, label + " camera-only encode flag")
# Resolved frames and frames with any blended draw keep the unencoded output.
# A blended draw would mix in sRGB space after the encode.
require_count(opengl, "            diffractionFrame || roughGlassFrame,\n"
              "            smoothglass::frameHasBlendedDraw(*scene, materialProgram.get(), resources->instanceAppearances));",
              1, "OpenGL resolved and blended frame exclusion")
require_count(sokol, "                diffractionDraw || tiledRough || (resources->smoothGlass.enabled && resources->smoothGlass.roughness > 0.0f),\n"
              "                smoothglass::frameHasBlendedDraw(*scene, materialProgram.get(), resources->instanceAppearances));",
              1, "Metal resolved and blended frame exclusion")
require(header, "inline bool frameHasBlendedDraw(const HarmonicMIDI::grid::Visual3DScene& scene,", "blended draw test")
blend_test = header[header.find("inline bool frameHasBlendedDraw("):]
blend_test = blend_test[:blend_test.find("\n}\n")]
for needle in ("nativeSurfaceRequiresBlend(nativeSurfaceForObject(root, object.id))",
               "instanceAppearances[o].color[3] < 1.0f",
               "material->alphaMode == HarmonicMIDI::grid::SceneAlphaMode::Blend"):
    require(blend_test, needle, "blended draw test mirrors the per-draw blend rule")
# The per-draw rule it mirrors, on both backends.
for label, text in (("OpenGL", opengl), ("Metal", sokol)):
    require(text, "const auto alphaMode = nativeSurfaceRequiresBlend(drawProgram) || (!resources->instanceAppearances.empty()",
            label + " per-draw blend rule")
# Raw pass export reads the same Color texture the compositor samples, so it is not gated.
require(header, "// Raw pass export needs no gate.", "raw export encoding note")
require_count(opengl, "passProgram.control[3] = cameraOnlyEncode ? 1.0f : 0.0f;", 1, "OpenGL pass control w")
require_count(sokol, "uniforms.passProgram.control[3] = cameraOnlyEncode ? 1.0f : 0.0f;", 1, "Metal pass control w")
require(opengl, "    if (count == 0 && uPassProgramControl.w > 0.5) {\n", "OpenGL compositePass camera-only encode")
require(opengl, "        return vec4(mix(1.055 * pow(rgb, vec3(1.0 / 2.4)) - 0.055, rgb * 12.92,\n"
        "            lessThanEqual(rgb, vec3(0.0031308))), clamp(color.a, 0.0, 1.0));",
        "OpenGL compositePass sRGB transfer")
require(sokol, "    if (count == 0 && u.passProgramControl.w > 0.5f) {\n", "Metal compositePass camera-only encode")
require(sokol, "        return float4(select(1.055f * pow(rgb, float3(1.0f / 2.4f)) - 0.055f, rgb * 12.92f,\n"
        "            rgb <= float3(0.0031308f)), clamp(value.color.a, 0.0f, 1.0f));",
        "Metal compositePass sRGB transfer")
for label, text, encode_at in (("OpenGL", opengl, "if (count == 0 && uPassProgramControl.w > 0.5)"),
                               ("Metal", sokol, "if (count == 0 && u.passProgramControl.w > 0.5f)")):
    at = text.find(encode_at)
    passthrough = text.find("return fixtureHalfRange(color);" if label == "OpenGL" else "return fixtureHalfRange(value.color);", at)
    if not (0 <= at < passthrough):
        errors.append(label + " compositePass must encode a camera-only draw before the count-zero passthrough")
require(opengl, "cameraOnlyEncode ? smoothglass::displayEncode(7.0f / 255.0f) : 7.0f / 255.0f,", "OpenGL encoded clear")
require(sokol, "if (cameraOnlyEncode)\n                pass.action.colors[0].clear_value = { smoothglass::displayEncode(7.0f / 255.0f),",
        "Metal encoded clear")
# The flag must be set before the per-frame pass control upload and the clear.
gl_flag = opengl.find("const bool cameraOnlyEncode = smoothglass::cameraOnlyDisplayEncode(")
if not (0 <= gl_flag < opengl.find("cameraOnlyEncode ? smoothglass::displayEncode(7.0f")
        < opengl.find("passProgram.control[3] = cameraOnlyEncode")):
    errors.append("OpenGL camera-only encode flag must precede the clear and the pass control upload")
metal_flag = sokol.find("const bool cameraOnlyEncode = smoothglass::cameraOnlyDisplayEncode(")
if not (0 <= metal_flag < sokol.find("uniforms.passProgram.control[3] = cameraOnlyEncode")
        < sokol.find("if (cameraOnlyEncode)\n                pass.action.colors[0].clear_value")):
    errors.append("Metal camera-only encode flag must precede the pass control and the clear")

if errors:
    for error in errors:
        print("FAIL: " + error, file=sys.stderr)
    raise SystemExit(1)
print("lighthouse paired camera source contract: PASS")
