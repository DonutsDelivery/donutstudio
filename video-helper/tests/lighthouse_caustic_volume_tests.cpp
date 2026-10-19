// Lighthouse caustic volume: a selected closed glass volume beside one thin
// water interface. Admission keeps the water as the camera dielectric and the
// selected volume in the second slot. The CPU photon reference absorbs photons
// that reach the water: no deposit, no refusal and no unresolved charge.
// The camera pass shades both dielectrics through the paired transport text.
// These cases check its per-draw uniforms and shader text against the released
// single-dielectric text. They run CPU admission, tracing and text only. They
// need no GPU.

#include "support/glass_caustic_oracle.h"
#include "../src/gpu_backend/fixture_glass_caustics_opengl.h"
#include "../src/gpu_backend/fixture_smooth_glass.h"
#include "../src/gpu_backend/metal_glass_caustic_consumer.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{
namespace gt = glasscaustictests;
namespace gc = arbitgpu::glasscaustics;
namespace sg = arbitgpu::smoothglass;
namespace ov = arbitgpu::opaquevisibility;
using namespace HarmonicMIDI::grid;

int failures = 0, checks = 0;
void check(bool condition, const std::string& text)
{
    ++checks;
    if (!condition) { ++failures; std::fprintf(stderr, "FAIL: %s\n", text.c_str()); }
}
bool close(double a, double b, double tolerance)
{
    return std::abs(a - b) <= tolerance * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

// The ordinary caustic fixture (glass slab over a diffuse receiver at z=-3)
// with a thin water plane at z=-1.5 between them, covering the receiver.
gt::Fixture lighthouse(SceneObjectId& water, SceneMaterialId& waterMaterial, float waterRoughness = 0)
{
    auto f = gt::ordinaryFixture();
    const auto first = f.scene->vertexCount;
    const auto o = f.scene->objectCount;
    videohelper::tests::smoothglassfixture::receiver(*f.scene, -6, {0, 0, 0});
    for (std::size_t v = first; v < f.scene->vertexCount; ++v) f.scene->vertices[v].position.z = -1.5f;
    water = f.scene->objects[o].id;
    waterMaterial = f.scene->objects[o].material;
    auto p = std::make_shared<gt::Program>();
    p->backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    p->object = water;
    p->bindingDigest = "caustic-fixture-binding-water";
    p->programIdentity = "caustic-fixture-program-water";
    p->authoredScene = f.scene->id;
    p->authoredMaterial = f.scene->objects[o].material;
    p->sceneRevision = p->structuralRevision = p->evaluationRevision = p->materialRevision = 1;
    p->parameters.identifiers[0] = p->authoredMaterial.value;
    p->parameters.identifiers[1] = std::uint32_t(surfacematerial::TransportMode::ThinInterface);
    p->parameters.normalOpacity = {0, 0, 1, 1};
    p->parameters.baseColorMetallic = {1, 1, 1, 0};
    p->parameters.transmissionIorClearcoat = {1, 1.33f, 0, 0};
    p->parameters.emissionRoughness = {0, 0, 0, waterRoughness};
    f.program->objectPrograms.push_back(p);
    return f;
}

std::vector<arbitgpu::opaquevisibility::Instance> instances(const Visual3DScene& scene)
{
    std::vector<arbitgpu::opaquevisibility::Instance> output;
    for (std::size_t o = 0; o < scene.objectCount; ++o)
    {
        // Scale, then rotation, then translation, as SceneTransform3D documents. The starter
        // lens carries all three; the other objects only a translation.
        auto m = videohelper::tests::sceneclosest::identity();
        const auto& transform = scene.objects[o].transform;
        const double x = transform.rotation.x, y = transform.rotation.y, z = transform.rotation.z,
                     w = transform.rotation.w;
        const double norm = x * x + y * y + z * z + w * w;
        const auto& k = transform.scale;
        m[0] = float((1 - 2 * (y * y + z * z) / norm) * k.x);
        m[1] = float(2 * (x * y + w * z) / norm * k.x);
        m[2] = float(2 * (x * z - w * y) / norm * k.x);
        m[4] = float(2 * (x * y - w * z) / norm * k.y);
        m[5] = float((1 - 2 * (x * x + z * z) / norm) * k.y);
        m[6] = float(2 * (y * z + w * x) / norm * k.y);
        m[8] = float(2 * (x * z + w * y) / norm * k.z);
        m[9] = float(2 * (y * z - w * x) / norm * k.z);
        m[10] = float((1 - 2 * (x * x + y * y) / norm) * k.z);
        const auto& p = transform.translation;
        m[12] = p.x; m[13] = p.y; m[14] = p.z;
        output.push_back({m, o});
    }
    return output;
}

void admission()
{
    SceneObjectId water;
    SceneMaterialId waterMaterial;
    auto f = lighthouse(water, waterMaterial);
    const std::shared_ptr<const Visual3DScene> scene = f.scene;
    const std::shared_ptr<const gt::Program> root = f.program;
    const auto volume = f.scene->objects[0].id;
    const sg::CausticVolumeSelection selection {volume, f.scene->objects[0].material};
    std::string error;

    // prepare itself takes no implicit selection: with none it applies the
    // released single-dielectric rules. With Glass Caustics off the renderer
    // derives a camera-only selection of the unique closed volume, and that
    // admits the unselected water-plus-glass scene.
    sg::Prepared unselected;
    check(!sg::prepare(scene, root, {}, unselected, error)
              && error == "Closed glass transport requires one dielectric with unit opacity and transmission",
          "prepare with no selection keeps the single-dielectric rules: " + error);
    const auto derived = sg::cameraVolumeSelection(*scene, root.get());
    check(derived && derived->cameraOnly && derived->object == volume && derived->material == selection.material,
          "the unselected water-plus-glass scene derives a camera-only selection of its closed volume");
    sg::Prepared cameraOnly;
    check(derived && sg::prepare(scene, root, {}, cameraOnly, error, {}, nullptr, &*derived)
              && cameraOnly.causticVolume.present && cameraOnly.causticVolume.cameraOnly,
          "the unselected water-plus-glass scene is admitted through camera-only selection: " + error);

    sg::Prepared prepared;
    const bool admitted = sg::prepare(scene, root, {}, prepared, error, {}, nullptr, &selection);
    check(admitted, "water plus the selected glass volume is admitted: " + error);
    check(prepared.enabled && prepared.transport == surfacematerial::TransportMode::ThinInterface
              && prepared.object == water,
          "the thin water stays the camera dielectric");
    check(prepared.causticVolume.present && prepared.causticVolume.object == volume
              && prepared.causticVolume.material == selection.material
              && prepared.causticVolume.ior == 1.5f && prepared.causticVolume.roughness == 0,
          "the selected closed volume fills the second slot");
    check(sg::causticObject(prepared) == volume && sg::causticVolumeNeedsSlot(root.get(), volume),
          "caustic consumers see the selected volume");

    const sg::CausticVolumeSelection wrong {water, waterMaterial};
    sg::Prepared refused;
    check(!sg::prepare(scene, root, {}, refused, error, {}, nullptr, &wrong)
              && error == "Glass caustic selection requires its exact authored closed-volume object and material",
          "a selection naming the thin water is refused");

    // A selection on a single-volume scene keeps the released result.
    auto single = gt::ordinaryFixture();
    const std::shared_ptr<const Visual3DScene> singleScene = single.scene;
    const std::shared_ptr<const gt::Program> singleRoot = single.program;
    sg::Prepared alone;
    check(!sg::causticVolumeNeedsSlot(singleRoot.get(), volume), "a lone volume needs no slot");
    check(sg::prepare(singleScene, singleRoot, {}, alone, error, {}, nullptr, &selection)
              && !alone.causticVolume.present && alone.object == volume
              && alone.transport == surfacematerial::TransportMode::ClosedVolume,
          "a selected lone volume stays the camera dielectric: " + error);
}

// Rough water has no paired camera walk, so a selection beside it is refused.
// Without a selection the rough water keeps today's behaviour.
void roughWater()
{
    SceneObjectId water;
    SceneMaterialId waterMaterial;
    auto f = lighthouse(water, waterMaterial, 0.3f);
    const std::shared_ptr<const Visual3DScene> scene = f.scene;
    const std::shared_ptr<const gt::Program> root = f.program;
    const auto volume = f.scene->objects[0].id;
    const sg::CausticVolumeSelection selection {volume, f.scene->objects[0].material};
    const sg::RoughNativeConsumer consumer {arbitgpu::NativeFixtureMaterialBackend::OpenGl, true, true, true};
    std::string error;

    sg::Prepared refused;
    check(!sg::prepare(scene, root, {}, refused, error, consumer, nullptr, &selection)
              && error == "Glass caustic volume beside water requires smooth water",
          "rough water beside the selected caustic volume is refused: " + error);

    // With Glass Caustics off the camera-only selection meets the same rule
    // and names the camera volume, not a caustic.
    sg::Prepared unselected;
    const auto derived = sg::cameraVolumeSelection(*scene, root.get());
    check(derived && !sg::prepare(scene, root, {}, unselected, error, consumer, nullptr, &*derived)
              && error == "Camera glass volume beside water requires smooth water",
          "rough water beside a camera-only volume is refused: " + error);

    // Smooth water beside the selected volume is unaffected.
    auto smoothFixture = lighthouse(water, waterMaterial);
    const std::shared_ptr<const Visual3DScene> smoothScene = smoothFixture.scene;
    const std::shared_ptr<const gt::Program> smoothRoot = smoothFixture.program;
    sg::Prepared smooth;
    check(sg::prepare(smoothScene, smoothRoot, {}, smooth, error, consumer, nullptr, &selection)
              && smooth.causticVolume.present && smooth.roughness == 0,
          "smooth water beside the selected volume stays admitted: " + error);
}

void waterAbsorbs()
{
    SceneObjectId water;
    SceneMaterialId waterMaterial;
    auto f = lighthouse(water, waterMaterial);
    const std::shared_ptr<const Visual3DScene> scene = f.scene;
    const std::shared_ptr<const gt::Program> root = f.program;
    const sg::CausticVolumeSelection selection {f.scene->objects[0].id, f.scene->objects[0].material};
    std::string error;
    sg::Prepared glass;
    if (!f.policy || !sg::prepare(scene, root, {}, glass, error, {}, nullptr, &selection))
    { check(false, "lighthouse admission: " + error); return; }
    gc::Prepared prepared;
    if (!gc::prepare(glass, f.ordinaryLight, *f.policy, prepared, error))
    { check(false, "lighthouse caustic preparation: " + error); return; }
    check(prepared.absorber == water && prepared.glass.object == f.scene->objects[0].id
              && prepared.glass.transport == surfacematerial::TransportMode::ClosedVolume,
          "the caustic view traces the volume and names the water as absorber");
    gc::Frame frame;
    if (!gc::buildFrame(prepared, instances(*f.scene), 32, 32, frame, error))
    { check(false, "lighthouse caustic frame: " + error); return; }

    // The same glass and light with no water. Pruned branch tails are the only
    // unresolved power there, and the water must not add to them.
    auto control = gt::ordinaryFixture();
    if (!control.prepare(error)) { check(false, "control caustic frame: " + error); return; }
    gc::EnergyLedger controlLedger;
    check(gc::referencePower(control.prepared, control.frame, controlLedger, error), "control reference power: " + error);

    double absorbed = 0;
    bool waterDeposit = false;
    for (const auto& work : frame.work)
    {
        gc::BranchReceipt receipt;
        const bool complete = gc::traceBranches(prepared, frame, frame.entries[work.entry], work.band,
                                                *f.policy, receipt, error, work.interfaceSample);
        check(complete && receipt.withinErrorBound, "a branch that reaches the water completes: " + error);
        absorbed += receipt.absorberFraction;
        for (const auto& deposit : receipt.deposits) waterDeposit |= deposit.receiver.objectId == water.value;
    }
    check(absorbed > 0, "photons leaving the glass are absorbed by the water");
    check(!waterDeposit, "the water receives no caustic deposit");

    gc::EnergyLedger ledger;
    check(gc::referencePower(prepared, frame, ledger, error), "reference power: " + error);
    for (unsigned band = 0; band < 8; ++band)
    {
        check(ledger.thinAbsorbed[band] > 0 && controlLedger.thinAbsorbed[band] == 0,
              "each band reports thin-absorbed power only when water is present");
        check(ledger.unresolved[band] <= controlLedger.unresolved[band] * (1 + 1e-9) + 1e-15,
              "absorbed photons are never charged as unresolved");
        check(close(ledger.launched[band], ledger.deposited[band] + ledger.absorbed[band] + ledger.escaped[band]
                  + ledger.unresolved[band] + ledger.masked[band] + ledger.thinAbsorbed[band], 1e-6),
              "thin-absorbed power closes the per-band energy balance");
    }
}

void nativeParity()
{
    // The CPU rule is mirrored in both legacy native tracers by owner flag 2.
    const auto gl = gc::opengl::computeSource();
    const auto metal = arbitgpu::metalcaustic::shader(false);
    check(gl.find("if(shape.z==2.0)") != std::string::npos
              && gl.find("paths[base+4].y=thinAbsorbed;") != std::string::npos,
          "OpenGL photon kernel absorbs at the water and reports texel 4.y");
    check(metal.find("triangles[uint(triangle)*8u+5u].z==2.0f") != std::string::npos
              && metal.find("receipt.padding[0]+=power*branch.fraction;") != std::string::npos,
          "Metal photon kernel absorbs at the water and reports padding[0]");
}

std::array<float,4> halves(float x, float y, SceneObjectId id)
{
    return {x, y, float(id.value & 65535u), float(id.value >> 16u)};
}

void cameraUniforms()
{
    SceneObjectId water;
    SceneMaterialId waterMaterial;
    auto f = lighthouse(water, waterMaterial);
    const std::shared_ptr<const Visual3DScene> scene = f.scene;
    const std::shared_ptr<const gt::Program> root = f.program;
    const auto volume = f.scene->objects[0].id;
    const auto receiver = f.scene->objects[1].id;
    const sg::CausticVolumeSelection selection {volume, f.scene->objects[0].material};
    std::string error;
    sg::Prepared prepared;
    if (!sg::prepare(scene, root, {}, prepared, error, {}, nullptr, &selection))
    { check(false, "lighthouse admission for camera uniforms: " + error); return; }
    const auto& slot = prepared.causticVolume;
    check(slot.cameraIor == 1.33f, "the volume slot carries the water IOR");

    // The volume draw shades the volume and names the water as partner.
    check(sg::cameraTransport(prepared, volume, 1, slot.dispersion) == halves(1, slot.dispersion, volume),
          "the caustic volume draw routes into the closed-volume branch with its own id");
    check(sg::cameraPartner(prepared, volume) == halves(1.33f, 0, water),
          "the caustic volume draw names the water and its IOR as partner");
    // The water draw keeps the released id and names the volume as partner.
    check(sg::cameraTransport(prepared, water, 2, 0) == halves(2, 0, water),
          "the water draw keeps the camera dielectric id");
    check(sg::cameraPartner(prepared, water) == halves(slot.ior, slot.dispersion, volume),
          "the water draw names the volume IOR, dispersion and id as partner");
    check(sg::cameraTransport(prepared, receiver, 0, 0) == halves(0, 0, water)
              && sg::cameraPartner(prepared, receiver) == halves(slot.ior, slot.dispersion, volume),
          "a receiver draw keeps the released transport id");

    // Without a volume slot the transport is the released upload and the
    // partner is zero, which the paired text reads as absent.
    auto single = gt::ordinaryFixture();
    const std::shared_ptr<const Visual3DScene> singleScene = single.scene;
    const std::shared_ptr<const gt::Program> singleRoot = single.program;
    sg::Prepared alone;
    if (!sg::prepare(singleScene, singleRoot, {}, alone, error))
    { check(false, "single volume admission: " + error); return; }
    for (const auto draw : {single.scene->objects[0].id, single.scene->objects[1].id})
    {
        check(sg::cameraTransport(alone, draw, 1, .25f) == halves(1, .25f, alone.object),
              "a single-dielectric draw uploads the released transport");
        check(sg::cameraPartner(alone, draw) == std::array<float,4> {},
              "a single-dielectric draw has a zero partner");
    }

    // The pair budget admits 1080p and 4K frames. Its walk is bounded to 16
    // visits per channel, or 48 for one walk that carries all three channels,
    // so the per-pixel count is 2 + 3 * 16 = 50.
    check(sg::admitFrame(prepared, 1920, 1080, false, error), "paired 1080p frame is admitted: " + error);
    check(sg::admitFrame(prepared, 3840, 2160, false, error), "paired 4K frame is admitted: " + error);
    check(sg::kPairPixelRayQueries == 2 + 3 * sg::kMaximumPairRayVisits && sg::kPairPixelRayQueries == 50,
          "the pair budget counts the primary, the sliver probe and 48 walk visits");
    check(sg::kMaximumPairRayVisits < sg::kMaximumRayVisits, "the pair walk is bounded tighter than glassTransport");
    check(sg::kPairTruncatedEnergyTolerance == 0.02f,
          "the pair walk truncates at most 0.02 of unit camera power per channel");
    check(!sg::admitFrame(prepared, 4096, 4096, false, error),
          "a paired frame past the pair query ceiling refuses");
}

std::size_t occurrences(const std::string& text, const std::string& needle)
{
    std::size_t n = 0;
    for (auto at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size())) ++n;
    return n;
}
void replaceAll(std::string& text, const std::string& from, const std::string& to)
{
    for (auto at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size()))
        text.replace(at, from.size(), to);
}
// The Metal spelling the shared recipe produces from GLSL parameter text.
std::string metalSpelling(std::string text)
{
    replaceAll(text, "out bool ", "thread bool& ");
    replaceAll(text, "out int ", "thread int& ");
    replaceAll(text, "vec3", "float3");
    return text;
}

void pairedShaderText()
{
    const std::string prototypes =
        "\nvec3 thinInterfaceTransport(vec3 camera,vec3 entry,vec3 evaluatedNormal,out bool complete,out int status GX_ARGS);\n"
        "vec3 glassTransport(vec3 camera,vec3 entry,out bool complete,out int status GX_ARGS);\n";
    const std::string thinEntry = "vec3 thinInterfaceTransport(vec3 camera,vec3 entry,vec3 evaluatedNormal,out bool complete,out int status GX_ARGS) {";
    const std::string volumeEntry = "vec3 glassTransport(vec3 camera,vec3 entry,out bool complete,out int status GX_ARGS) {";
    for (const bool metal : {false, true})
    {
        const std::string name = metal ? "Metal" : "OpenGL";
        const auto spell = [&](const std::string& text) { return metal ? metalSpelling(text) : text; };
        const auto released = sg::transportShader(metal, true);
        const auto paired = sg::transportShader(metal, true, false, true);
        check(released == sg::transportShader(metal, true, false, false), name + " default is the unpaired text");
        for (const char* token : {"GlassTransport2", "glassTransport2", "glassPair", "SingleTransport", "SingleInterface"})
            check(released.find(token) == std::string::npos,
                  name + " released text has no paired token " + token);

        // The second dielectric uniform exists only in the paired text.
        const std::string partner = metal ? "u.glassTransport2" : "uGlassTransport2";
        check(paired.find(partner + ".x==0.0") != std::string::npos, name + " paired text reads the partner uniform");
        check(paired.find("GU_") == std::string::npos, name + " paired text has no unrenamed uniform");
        check(metal ? paired.find("uniform vec4") == std::string::npos
                    : occurrences(paired, "uniform vec4 uGlassTransport2;\n") == 1,
              name + " partner uniform declaration");

        // Each entry point is defined once; the released bodies keep new names.
        check(occurrences(paired, spell(thinEntry)) == 1 && occurrences(paired, spell(volumeEntry)) == 1,
              name + " paired entry points are defined once");
        check(occurrences(paired, spell("vec3 thinSingleInterfaceTransport(")) == 1
                  && occurrences(paired, spell("vec3 glassSingleTransport(")) == 1
                  && occurrences(paired, spell("vec3 glassPairWalk(")) == 1,
              name + " paired text keeps both released bodies and one walk");
        check(paired.find("return thinSingleInterfaceTransport(camera,entry,evaluatedNormal,complete,status GX_PASS);") != std::string::npos
                  && paired.find("return glassSingleTransport(camera,entry,complete,status GX_PASS);") != std::string::npos,
              name + " paired entry points fall back to the released bodies without a partner");
        check(paired.find("for(int visit=0;visit<48;++visit)") != std::string::npos
                  && paired.find("if(depth>=GLASS_INTERFACE_DEPTH) {status=GLASS_SMOOTH_STATUS_DEPTH;") != std::string::npos
                  && paired.find("if(pending>=16) {status=GLASS_SMOOTH_STATUS_PENDING;") != std::string::npos,
              name + " the released walks keep their depth and pending refusals; the paired walk has its 48 visit loop");
        // A non-finite final radiance carries a status, not UNCLASSIFIED.
        const auto walkAt = paired.find(spell("vec3 glassPairWalk("));
        const auto walkEnd = paired.find(spell("vec3 thinInterfaceTransport("), walkAt);
        const auto walk = walkAt == std::string::npos || walkEnd == std::string::npos
            ? std::string {} : paired.substr(walkAt, walkEnd - walkAt);
        check(walk.find("visit<48") != std::string::npos && walk.find("visit<64") == std::string::npos
                  && paired.find("for(int visit=0;visit<64;++visit)") != std::string::npos,
              name + " only the paired walk takes the 48 visit loop; glassTransport keeps 64");
        // Nondispersive glass walks once for three channels with 48 visits;
        // dispersive glass walks each channel with 16. Both stay within 48.
        check(walk.find("bool sharedPaths=volumeDispersion==0.0;") != std::string::npos
                  && walk.find("int walks=sharedPaths?1:3,visits=sharedPaths?48:16;") != std::string::npos
                  && walk.find("if(channel>=walks) break;") != std::string::npos
                  && walk.find("if(visit>=visits||pending==0) break;") != std::string::npos,
              name + " the paired walk pools 48 visits for nondispersive glass and keeps 16 per dispersive channel");
        // A branch at the interface depth bound ends as a leftover, not a refusal.
        check(walk.find("GLASS_SMOOTH_STATUS_DEPTH") == std::string::npos
                  && walk.find(spell("if(depth>=GLASS_INTERFACE_DEPTH) {\n                radiance+=lanes*glassPairLeftover(rayDirection,weight,rayInside GX_PASS);"))
                         != std::string::npos,
              name + " a paired branch at the depth bound ends through the leftover rule");
        // The heaviest pending branch is visited first. Branches left at the
        // visit or slot bound end without a query, and a channel whose
        // truncated weight passes the tolerance still refuses.
        check(paired.find("#define GLASS_PAIR_TRUNCATED 0.02\n") != std::string::npos
                  && released.find("GLASS_PAIR_TRUNCATED") == std::string::npos,
              name + " only the paired text defines the truncation tolerance");
        check(walk.find("for(int slot=1;slot<pending;++slot) if(weights[slot]>weights[heaviest]) heaviest=slot;") != std::string::npos
                  && walk.find("float weight=weights[heaviest];") != std::string::npos,
              name + " the paired walk visits the heaviest pending branch");
        check(walk.find("if(pending>=16) {status=GLASS_SMOOTH_STATUS_PENDING;") == std::string::npos
                  && walk.find("if(pending!=0) {status=GLASS_SMOOTH_STATUS_VISITS;") == std::string::npos,
              name + " the paired walk no longer refuses on the first unvisited branch");
        check(walk.find(spell("if(truncated>GLASS_PAIR_TRUNCATED&&spilled)\n            {status=GLASS_SMOOTH_STATUS_PENDING;return vec3(0.0);}\n        if(truncated>GLASS_PAIR_TRUNCATED) trapped=true;"))
                  != std::string::npos
                  && walk.find("GLASS_SMOOTH_STATUS_VISITS") == std::string::npos,
              name + " truncated weight past the tolerance refuses as PENDING when spilled, else the pixel is trapped");
        check(occurrences(walk, "glassSceneClosestHit(") == 1,
              name + " the paired walk queries once per visit; leftover branches take no query");
        check(walk.find("complete=glassSceneFinite3(radiance);\n    if(!complete) status=GLASS_SMOOTH_STATUS_THIN_RADIANCE;\n    else if(trapped) status=GLASS_SMOOTH_STATUS_TRAPPED;\n    return radiance;")
                  != std::string::npos,
              name + " paired walk classifies a non-finite final radiance and a trapped one");

        // Removing the paired additions gives the released text byte for byte.
        auto stripped = paired;
        const auto appended = stripped.find(spell("\nuint glassPairId("));
        check(appended != std::string::npos, name + " paired entry points are appended last");
        if (appended == std::string::npos) continue;
        stripped.erase(appended);
        const auto prototypeAt = stripped.find(spell(prototypes));
        check(prototypeAt != std::string::npos, name + " paired prototypes precede the released text");
        if (prototypeAt != std::string::npos) stripped.erase(prototypeAt, spell(prototypes).size());
        if (!metal)
        {
            const std::string declaration = "uniform vec4 uGlassTransport2;\n";
            const auto at = stripped.find(declaration);
            if (at != std::string::npos) stripped.erase(at, declaration.size());
        }
        replaceAll(stripped, spell("vec3 thinSingleInterfaceTransport("), spell("vec3 thinInterfaceTransport("));
        replaceAll(stripped, spell("vec3 glassSingleTransport("), spell("vec3 glassTransport("));
        check(stripped == released, name + " paired text only adds to the released text");
    }

    // The Metal SurfacePbr ledger builds the same paired bytes within its budget.
    namespace ss = arbitgpu::surfacesource;
    const auto census = [&](bool pairedVolume, std::string& text) -> std::size_t {
        try
        {
            const auto archive = ss::prepare([&] {
                return std::make_pair(sg::transportShaderRecipe<ss::Source>(true, true, false, pairedVolume), ss::Source(""));
            });
            text.assign(archive->vertex(), archive->vertexLength);
            return archive->plan.count;
        }
        catch (const std::exception& exception)
        {
            check(false, std::string("paired surface source refused: ") + exception.what());
            return 0;
        }
    };
    std::string releasedText, pairedText;
    const auto releasedStages = census(false, releasedText), pairedStages = census(true, pairedText);
    check(releasedText == sg::transportShader(true, true) && pairedText == sg::transportShader(true, true, false, true),
          "the Metal surface source ledger emits the std::string recipe bytes");
    check(pairedStages > releasedStages && pairedStages - releasedStages + 1 <= ss::StageBudget::pairedVolume,
          "paired transport uses " + std::to_string(pairedStages - releasedStages)
              + " extra stages; one more declares the fragment uniform within StageBudget::pairedVolume");
}

// The real lighthouse-and-ocean starter, rebuilt from its authored values:
// object order water, tower, lantern, glass, rock (ids 101 to 105), a 12 by 12
// water grid at 0.7 spacing laid flat with evaluated wave normals, three cubes at
// the starter's sizes and translations, the starter's sphere lens with its scale,
// tilt and position, IOR 1.5 glass, matte spectral receivers
// with the starter's rho bands, and a spectral directional sun. The water
// rotation is baked into its vertices. The caustic selects the glass.
struct Starter
{
    std::shared_ptr<Visual3DScene> scene;
    std::shared_ptr<gt::Program> root;
    SceneObjectId water, glass;
    SceneMaterialId glassMaterial;
    SceneLightId sun;
};

void starterCube(Visual3DScene& scene, std::size_t o, float w, float h, float d, SceneVec3 at)
{
    struct Face { float n[3], u[3], v[3]; };
    const Face faces[6] = {
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},
        {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}, {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
        {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}}, {{0, 0, -1}, {0, 1, 0}, {1, 0, 0}}};
    const float half[3] = {w / 2, h / 2, d / 2};
    auto& object = scene.objects[o];
    object.firstVertex = std::uint32_t(scene.vertexCount);
    object.vertexCount = 24;
    object.firstIndex = std::uint32_t(scene.indexCount);
    object.indexCount = 36;
    object.transform.translation = at;
    const float corners[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    for (std::size_t f = 0; f < 6; ++f)
    {
        const std::uint32_t base = std::uint32_t(f * 4);
        for (std::size_t c = 0; c < 4; ++c)
        {
            auto& vertex = scene.vertices[scene.vertexCount++];
            float p[3];
            for (std::size_t a = 0; a < 3; ++a)
                p[a] = half[a] * (faces[f].n[a] + corners[c][0] * faces[f].u[a] + corners[c][1] * faces[f].v[a]);
            vertex = {};
            vertex.position = {p[0], p[1], p[2]};
            vertex.normal = {faces[f].n[0], faces[f].n[1], faces[f].n[2]};
            vertex.uv = {corners[c][0] > 0 ? 1.0f : 0.0f, corners[c][1] > 0 ? 1.0f : 0.0f};
        }
        const std::uint32_t order[6] = {0, 1, 2, 0, 2, 3};
        for (const auto i : order) scene.indices[scene.indexCount++] = base + i;
    }
}

// Transform 3D's quaternion from rotationX, Y and Z in degrees (R = Rz Ry Rx).
SceneQuaternion starterRotation(SceneVec3 rotationDegrees)
{
    const double toHalfRadians = 3.14159265358979323846 / 360.0;
    const double sx = std::sin(rotationDegrees.x * toHalfRadians), cx = std::cos(rotationDegrees.x * toHalfRadians);
    const double sy = std::sin(rotationDegrees.y * toHalfRadians), cy = std::cos(rotationDegrees.y * toHalfRadians);
    const double sz = std::sin(rotationDegrees.z * toHalfRadians), cz = std::cos(rotationDegrees.z * toHalfRadians);
    return {float(sx * cy * cz - cx * sy * sz), float(cx * sy * cz + sx * cy * sz),
            float(cx * cy * sz - sx * sy * cz), float(cx * cy * cz + sx * sy * sz)};
}

// The earlier starter lens: the closed cylinder GeometryCore.h generates (bottom and top side
// rings, bottom and top flat cap rings, then the two cap centres, every triangle wound
// outward), scaled, tilted and moved by the object transform. The rotation is the
// quaternion Transform 3D lowers from rotationX, Y and Z in degrees (R = Rz Ry Rx).
// Indices are relative to the object's first vertex, as in starterCube.
void starterCylinder(Visual3DScene& scene, std::size_t o, float radius, float height, std::uint32_t segments,
                     SceneVec3 scale, SceneVec3 rotationDegrees, SceneVec3 at)
{
    auto& object = scene.objects[o];
    object.firstVertex = std::uint32_t(scene.vertexCount);
    object.vertexCount = 4 * segments + 2;
    object.firstIndex = std::uint32_t(scene.indexCount);
    object.indexCount = 12 * segments;
    object.transform.translation = at;
    object.transform.scale = scale;
    object.transform.rotation = starterRotation(rotationDegrees);
    for (std::uint32_t ring = 0; ring < 4; ++ring)
        for (std::uint32_t column = 0; column < segments; ++column)
        {
            const double angle = 2.0 * 3.14159265358979323846 * column / segments;
            const float c = float(std::cos(angle)), s = float(std::sin(angle));
            auto& vertex = scene.vertices[scene.vertexCount++];
            vertex = {};
            // Rounded like the generator, in double and then to float, so the side and cap rings weld exactly.
            vertex.position = {float(double(radius) * std::cos(angle)), (ring % 2 == 0 ? -0.5f : 0.5f) * height,
                               float(double(radius) * std::sin(angle))};
            if (ring < 2)
            {
                vertex.normal = {c, 0, s};
                vertex.tangent = {-s, 0, c, 1};
                vertex.uv = {float(column) / float(segments), float(ring)};
            }
            else
            {
                vertex.normal = {0, ring == 2 ? -1.0f : 1.0f, 0};
                vertex.uv = {0.5f + 0.5f * c, 0.5f + 0.5f * s};
            }
        }
    for (const float y : {-1.0f, 1.0f})
    {
        auto& vertex = scene.vertices[scene.vertexCount++];
        vertex = {};
        vertex.position = {0, 0.5f * y * height, 0};
        vertex.normal = {0, y, 0};
        vertex.uv = {0.5f, 0.5f};
    }
    const std::uint32_t bottom = 4 * segments, top = bottom + 1;
    for (std::uint32_t column = 0; column < segments; ++column)
    {
        const std::uint32_t next = (column + 1) % segments;
        const std::uint32_t triangles[12] = {column, column + segments, next,
                                             next, column + segments, next + segments,
                                             bottom, 2 * segments + column, 2 * segments + next,
                                             top, 3 * segments + next, 3 * segments + column};
        for (const auto i : triangles) scene.indices[scene.indexCount++] = i;
    }
}


// The Geometry Core icosphere generator's own mesh, as the native lowering
// emits it: shared vertices, outward winding, positions relative to the centre.
bool starterIcosphere(Visual3DScene& scene, std::size_t o, float radius, float subdivisions, SceneVec3 at)
{
    namespace geometry = videowire::geometry;
    geometry::OperationRecord operation;
    operation.code = geometry::OperationCode::icosphere;
    operation.parameters = {radius, subdivisions, 0, 0};
    geometry::GeometryData mesh;
    std::string error;
    if (!geometry::materializeMeshGenerator(operation, mesh, Visual3DScene::kMaxVertices, Visual3DScene::kMaxIndices, error)
        || scene.vertexCount + mesh.positions.size() > Visual3DScene::kMaxVertices
        || scene.indexCount + mesh.indices.size() > Visual3DScene::kMaxIndices)
    { check(false, "icosphere generator: " + error); return false; }
    auto& object = scene.objects[o];
    object.firstVertex = std::uint32_t(scene.vertexCount);
    object.vertexCount = std::uint32_t(mesh.positions.size());
    object.firstIndex = std::uint32_t(scene.indexCount);
    object.indexCount = std::uint32_t(mesh.indices.size());
    object.transform.translation = at;
    for (const auto& p : mesh.positions)
    {
        auto& vertex = scene.vertices[scene.vertexCount++];
        vertex = {};
        vertex.position = {p.x, p.y, p.z};
        vertex.normal = {p.x / radius, p.y / radius, p.z / radius};
    }
    for (const auto index : mesh.indices) scene.indices[scene.indexCount++] = index;
    return true;
}

// The starter lens: the Geometry Core sphere generator's own mesh (radius 0.15,
// 10 segments, 6 rings: 52 vertices, 100 triangles, outward winding), flattened
// to scaleY 0.14, turned so its axis lies along the sun and hung in front of the
// lantern, as BuiltInVisualModules.h authors it.
bool starterSphere(Visual3DScene& scene, std::size_t o)
{
    namespace geometry = videowire::geometry;
    const float radius = 0.15f;
    geometry::OperationRecord operation;
    operation.code = geometry::OperationCode::sphere;
    operation.parameters = {radius, 10, 6, 0};
    geometry::GeometryData mesh;
    std::string error;
    if (!geometry::materializeMeshGenerator(operation, mesh, Visual3DScene::kMaxVertices, Visual3DScene::kMaxIndices, error)
        || scene.vertexCount + mesh.positions.size() > Visual3DScene::kMaxVertices
        || scene.indexCount + mesh.indices.size() > Visual3DScene::kMaxIndices)
    { check(false, "sphere generator: " + error); return false; }
    auto& object = scene.objects[o];
    object.firstVertex = std::uint32_t(scene.vertexCount);
    object.vertexCount = std::uint32_t(mesh.positions.size());
    object.firstIndex = std::uint32_t(scene.indexCount);
    object.indexCount = std::uint32_t(mesh.indices.size());
    object.transform.translation = {-1.425f, 2.573f, -0.392f};
    object.transform.scale = {1.0f, 0.14f, 1.0f};
    object.transform.rotation = starterRotation({55.0f, -25.0f, 0.0f});
    for (const auto& p : mesh.positions)
    {
        auto& vertex = scene.vertices[scene.vertexCount++];
        vertex = {};
        vertex.position = {p.x, p.y, p.z};
        vertex.normal = {p.x / radius, p.y / radius, p.z / radius};
    }
    for (const auto index : mesh.indices) scene.indices[scene.indexCount++] = index;
    return true;
}

enum class StarterLens { Sphere, EarlierCylinder };

// lens: null keeps a generated lens, the starter's sphere or, with
// StarterLens::EarlierCylinder, the cylinder the starter shipped before it.
// Otherwise the glass is an icosphere of radius 0.3 and one subdivision centred there.
Starter starterScene(float waterRoughness = 0, const SceneVec3* lens = nullptr,
                     StarterLens shape = StarterLens::Sphere)
{
    Starter s;
    s.scene = std::make_shared<Visual3DScene>();
    auto& scene = *s.scene;
    scene.id = {7};
    const std::uint32_t objectIds[5] = {101, 102, 103, 104, 105};
    const std::uint32_t materialIds[5] = {1001, 1003, 1004, 1002, 1005};
    for (std::size_t o = 0; o < 5; ++o)
    {
        scene.objects[o].id = {objectIds[o]};
        scene.objects[o].material = {materialIds[o]};
        scene.materials[o].id = {materialIds[o]};
        scene.materials[o].baseColor = {1, 1, 1};
        scene.materials[o].roughness = 1;
        scene.materials[o].opacity = 1;
    }
    scene.objectCount = scene.materialCount = 5;

    // Water: 12 by 12 vertices, +y facing triangles, normals from the wave slope.
    auto& water = scene.objects[0];
    water.firstVertex = 0;
    water.vertexCount = 144;
    water.firstIndex = 0;
    water.indexCount = 11 * 11 * 6;
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < 12; ++j)
        {
            const float x = (float(i) - 5.5f) * 0.7f, z = (float(j) - 5.5f) * 0.7f;
            const float phase = 2.0f * 3.14159265f * (x * 0.94f + z * 0.34f) / 2.8f;
            const float slope = 0.09f * 2.0f * 3.14159265f / 2.8f * std::cos(phase);
            const float nx = -slope * 0.94f, nz = -slope * 0.34f;
            const float length = std::sqrt(nx * nx + 1 + nz * nz);
            auto& vertex = scene.vertices[scene.vertexCount++];
            vertex = {};
            vertex.position = {x, 0.09f * std::sin(phase), z};
            vertex.normal = {nx / length, 1 / length, nz / length};
            vertex.uv = {float(i) / 11, float(j) / 11};
        }
    for (std::uint32_t i = 0; i < 11; ++i)
        for (std::uint32_t j = 0; j < 11; ++j)
        {
            const std::uint32_t v00 = i * 12 + j, v01 = v00 + 1, v10 = v00 + 12, v11 = v10 + 1;
            const std::uint32_t quad[6] = {v00, v01, v11, v00, v11, v10};
            for (const auto v : quad) scene.indices[scene.indexCount++] = v;
        }
    starterCube(scene, 1, 0.65f, 2.6f, 0.65f, {-1.2f, 1.3f, -1.2f});
    starterCube(scene, 2, 0.95f, 0.4f, 0.95f, {-1.2f, 2.8f, -1.2f});
    // The sphere lens spans x -1.561 to -1.289, y 2.456 to 2.690 and z -0.490 to -0.295:
    // 0.38 in front of the tower front face (z -0.875) and 0.23 in front of the lantern (z -0.725).
    // The earlier lens is a 12-segment cylinder of radius 0.3 and height 0.3, squeezed to half
    // width, tilted by (118.86, -36.54, 133.50) degrees and hung at (-1.2, 0.9, -0.582): its lowest
    // point is 0.26 above the rock top (y 0.375) and 4 cm in front of the tower front face (z -0.875).
    if (lens) starterIcosphere(scene, 3, 0.3f, 1, *lens);
    else if (shape == StarterLens::Sphere) starterSphere(scene, 3);
    else starterCylinder(scene, 3, 0.3f, 0.3f, 12, {0.5f, 1.0f, 1.0f}, {118.86f, -36.54f, 133.50f}, {-1.2f, 0.9f, -0.582f});
    starterCube(scene, 4, 1.9f, 0.65f, 1.7f, {-1.2f, 0.05f, -1.2f});
    s.water = scene.objects[0].id;
    s.glass = scene.objects[3].id;
    s.glassMaterial = scene.objects[3].material;

    scene.lightCount = 2;
    scene.lights[0].id = {301};
    scene.lights[0].kind = SceneLightKind::Environment;
    scene.lights[0].color = {.125f, .25f, .5f};
    auto& sun = scene.lights[1];
    sun.id = {302};
    sun.kind = SceneLightKind::Directional;
    sun.color = {1, 0.9f, 0.75f};
    sun.intensity = 1.2f;
    const float ax = -35.0f * 3.14159265f / 180, ay = -25.0f * 3.14159265f / 180;
    const float sx = std::sin(ax / 2), cx = std::cos(ax / 2), sy = std::sin(ay / 2), cy = std::cos(ay / 2);
    sun.transform.rotation = {cy * sx, sy * cx, -sy * sx, cy * cx};
    sun.spectralSource.version = 1;
    sun.spectralSource.radiance.fill(1.0f);
    sun.spectralSource.traceMaximumDistance = 1000;
    s.sun = sun.id;
    scene.cameraCount = 1;
    scene.cameras[0].id = {401};
    scene.activeCamera = {401};
    scene.cameras[0].transform.translation = {0, 2.4f, 7};
    scene.cameras[0].verticalFovRadians = 52.0f * 3.14159265f / 180;
    scene.cameras[0].nearPlane = 0.1f;
    scene.cameras[0].farPlane = 100;

    // Surface Material programs. The glass is the root, as in the other cases here.
    const auto program = [&](std::size_t o) {
        auto p = std::make_shared<gt::Program>();
        p->backend = arbitgpu::NativeFixtureMaterialBackend::OpenGl;
        p->object = scene.objects[o].id;
        p->bindingDigest = "starter-binding-" + std::to_string(o);
        p->programIdentity = "starter-program-" + std::to_string(o);
        p->authoredScene = scene.id;
        p->authoredMaterial = scene.objects[o].material;
        p->sceneRevision = p->structuralRevision = p->evaluationRevision = p->materialRevision = 1;
        p->parameters.identifiers[0] = p->authoredMaterial.value;
        p->parameters.normalOpacity = {0, 0, 1, 1};
        return p;
    };
    const auto receiver = [&](std::size_t o, double r, double g, double b) {
        auto p = program(o);
        p->parameters.baseColorMetallic = {float(r), float(g), float(b), 0};
        p->parameters.transmissionIorClearcoat = {0, 1.5f, 0, 0};
        p->parameters.emissionRoughness = {0, 0, 0, 1};
        p->spectralReflectance = surfacematerial::SpectralReflectanceDescription{};
        p->spectralReflectance->coefficients = {float(b), float(b), float(b), float(0.5 * (g + b)), float(g),
                                                float(0.5 * (r + g)), float(r), float(r)};
        return p;
    };
    auto waterProgram = program(0);
    waterProgram->parameters.identifiers[1] = std::uint32_t(surfacematerial::TransportMode::ThinInterface);
    waterProgram->parameters.baseColorMetallic = {1, 1, 1, 0};
    waterProgram->parameters.transmissionIorClearcoat = {1, 1.33f, 0, 0};
    waterProgram->parameters.emissionRoughness = {0, 0, 0, waterRoughness};
    auto glassProgram = program(3);
    glassProgram->parameters.identifiers[1] = std::uint32_t(surfacematerial::TransportMode::ClosedVolume);
    glassProgram->parameters.baseColorMetallic = {1, 1, 1, 0};
    glassProgram->parameters.transmissionIorClearcoat = {1, 1.5f, 0, 0};
    glassProgram->parameters.emissionRoughness = {0, 0, 0, 0};
    glassProgram->objectPrograms = {waterProgram, receiver(1, 0.8, 0.72, 0.58), receiver(2, 0.95, 0.62, 0.2),
                                    receiver(4, 0.12, 0.15, 0.18)};
    s.root = glassProgram;
    return s;
}

// The starter's own values must pass every CPU admission step of the paired
// path: smooth glass prepare, the frame budget at 1920x1080 and 3840x2160, caustic scene
// admission, source preparation and the scheduled frame build.
void starterAdmission()
{
    auto starter = starterScene();
    const std::shared_ptr<const Visual3DScene> scene = starter.scene;
    const std::shared_ptr<const gt::Program> root = starter.root;
    check(validateVisual3DScene(*scene).valid(), "the rebuilt starter scene is a valid snapshot");
    const sg::CausticVolumeSelection selection {starter.glass, starter.glassMaterial};
    std::string error;

    sg::Prepared prepared;
    check(sg::prepare(scene, root, {}, prepared, error, {}, nullptr, &selection),
          "starter prepare admits water plus the selected glass: " + error);
    check(prepared.enabled && prepared.transport == surfacematerial::TransportMode::ThinInterface
              && prepared.object == starter.water && prepared.causticVolume.present
              && prepared.causticVolume.object == starter.glass && prepared.causticVolume.ior == 1.5f,
          "starter water is the camera dielectric and the glass fills the second slot");

    check(sg::admitFrame(prepared, 1920, 1080, false, error), "starter frame at 1920x1080 is admitted: " + error);
    check(sg::admitFrame(prepared, 3840, 2160, false, error), "starter frame at 3840x2160 is admitted: " + error);
    check(!sg::admitFrame(prepared, 4096, 4096, false, error)
              && error.find("4096x4096") != std::string::npos && error.find("10737418") != std::string::npos
              && error.find("dynamic instances") == std::string::npos,
          "an oversized frame names its extent and the limit in the pixel budget refusal: " + error);
    check(!sg::admitFrame(prepared, 1920, 1080, true, error)
              && error == "Closed glass transport refuses dynamic instances",
          "dynamic instances keep their own refusal: " + error);
    check(!sg::admitFrame(prepared, 0, 1080, false, error) && error.find("must not be empty") != std::string::npos,
          "an empty extent is refused as such: " + error);

    gc::Prepared caustic;
    check(gc::prepareScene(prepared, "starter-lighthouse", caustic, error),
          "starter caustic scene admission: " + error);
    check(caustic.absorber == starter.water && caustic.glass.object == starter.glass
              && caustic.glass.transport == surfacematerial::TransportMode::ClosedVolume,
          "the caustic view holds the glass as its volume and the water as absorber");

    const auto policy = gt::ordinaryFixture().policy;
    check(policy != nullptr, "the branch policy admits");
    if (!policy) return;
    gc::Prepared withSource;
    check(gc::prepare(prepared, starter.sun, *policy, withSource, error), "starter source preparation: " + error);
    gc::Frame frame;
    check(gc::buildFrame(withSource, instances(*scene), 1920, 1080, frame, error),
          "starter scheduled frame build at 1920x1080: " + error);
    // 4K: the deposition schedule and the whole-frame ledger scale with pixels.
    // 128 bytes per pixel covers the native frame attachments both backends add.
    gc::Frame frame4k;
    check(gc::buildFrame(withSource, instances(*scene), 3840, 2160, frame4k, error),
          "starter scheduled frame build at 3840x2160: " + error);
    std::uint64_t admittedBytes = 0;
    check(gc::admitFrameMemory(frame4k, std::uint64_t(3840) * 2160 * 128, admittedBytes, error),
          "starter 4K caustic frame memory is admitted: " + error);

    // Audit mode lists reasons on stderr but returns the same first refusal.
    auto broken = starterScene();
    broken.scene->materials[3].alphaMode = SceneAlphaMode::Blend;
    const std::shared_ptr<const Visual3DScene> brokenScene = broken.scene;
    const std::shared_ptr<const gt::Program> brokenRoot = broken.root;
    sg::Prepared refused;
    std::string quiet, audited;
    check(!sg::prepare(brokenScene, brokenRoot, {}, refused, quiet, {}, nullptr, &selection) && !quiet.empty(),
          "a nonopaque glass material is refused");
#ifdef _WIN32
    _putenv_s("ARBIT_GLASS_ADMISSION_AUDIT", "1");
#else
    setenv("ARBIT_GLASS_ADMISSION_AUDIT", "1", 1);
#endif
    check(!sg::prepare(brokenScene, brokenRoot, {}, refused, audited, {}, nullptr, &selection),
          "the audited run refuses the same scene");
#ifdef _WIN32
    _putenv_s("ARBIT_GLASS_ADMISSION_AUDIT", "");
#else
    unsetenv("ARBIT_GLASS_ADMISSION_AUDIT");
#endif
    check(quiet == audited, "audit mode returns the first refusal unchanged: " + audited);
}

// The earlier starter, with its cylinder lens, at 4K with the starter camera
// tilt of -14 degrees and the policy the plugin admits (1e-4 unresolved, 1e-6
// minimum branch). The pixel and work numbers below belong to that lens, so the
// case keeps it. Work 1392 of 3072 deposits at pixel (1580, 861) on the tower front face. That
// pixel centre lies on the diagonal the front face's two triangles share, so
// the rasterizer and the closest-hit query may each name a different one.
// The native walk refused the whole frame there before the shared-edge owner
// rule.
void starterSharedEdgeRaster()
{
    auto starter = starterScene(0, nullptr, StarterLens::EarlierCylinder);
    const float tilt = -14.0f * 3.14159265f / 180;
    starter.scene->cameras[0].transform.rotation = {std::sin(tilt / 2), 0, 0, std::cos(tilt / 2)};
    const std::shared_ptr<const Visual3DScene> scene = starter.scene;
    const std::shared_ptr<const gt::Program> root = starter.root;
    const sg::CausticVolumeSelection selection {starter.glass, starter.glassMaterial};
    std::string error;
    sg::Prepared prepared;
    if (!sg::prepare(scene, root, {}, prepared, error, {}, nullptr, &selection))
    { check(false, "tilted starter prepare: " + error); return; }
    gc::BranchPolicyDescription description;
    description.version = gc::kBranchPolicyVersion;
    description.maximumUnresolvedFraction = 1e-4f;
    description.minimumBranchFraction = 1e-6f;
    const auto policy = gc::AdmittedBranchPolicy::admit(description, error);
    if (!policy) { check(false, "starter branch policy: " + error); return; }
    gc::Prepared caustic;
    if (!gc::prepare(prepared, starter.sun, *policy, caustic, error))
    { check(false, "tilted starter source preparation: " + error); return; }
    gc::Frame frame;
    if (!gc::buildFrame(caustic, instances(*scene), 3840, 2160, frame, error))
    { check(false, "tilted starter 4K frame: " + error); return; }
    check(frame.work.size() == 3072, "the earlier starter schedules 384 lens entries in 8 bands");

    // Every walk completes inside the policy, so the tail is not what refused.
    bool allComplete = true;
    std::size_t maximumVisits = 0;
    for (const auto& work : frame.work)
    {
        if (work.band != 0) continue;
        gc::BranchReceipt receipt;
        const bool complete = gc::traceBranches(caustic, frame, frame.entries[work.entry], work.band, *policy,
                                                receipt, error, work.interfaceSample);
        allComplete &= complete && receipt.withinErrorBound;
        maximumVisits = std::max(maximumVisits, receipt.visits);
    }
    check(allComplete, "every starter lens walk completes within the 1e-4 tail policy: " + error);
    check(maximumVisits < gc::kMaximumBranchVisits, "no starter walk reaches the visit cap");

    // The pixel-centre hit on the tower front face, next to the shared diagonal.
    const auto ray = gc::cameraRay(frame, 1580.5f, 861.5f);
    const auto hit = ov::closestHitCpu(caustic.glass.query, frame.query, gc::cameraPosition(frame), ray,
                                       gc::kRayMinimum, 4 * frame.camera.farPlane);
    const auto tower = starter.scene->objects[1];
    check(hit.status == 1 && hit.objectId == tower.id.value && std::abs(hit.position[2] + 0.875f) < 1e-3f,
          "pixel (1580, 861) sees the tower front face");
    if (hit.status != 1 || hit.objectId != tower.id.value) return;
    std::size_t owners = 0, others = 0;
    bool sameTriangle = false;
    for (const auto& triangle : frame.triangles)
    {
        if (triangle.object != hit.objectId || triangle.material != hit.materialId
            || triangle.instance != hit.instanceId)
            continue;
        const bool owns = gc::rasterOwnsPixelHit(frame, hit, hit, triangle.triangle);
        if (triangle.triangle == hit.triangleId) sameTriangle = owns;
        else if (owns) ++owners;
        else ++others;
    }
    check(sameTriangle, "the hit triangle owns its own pixel");
    check(owners == 1, "exactly one other tower triangle, the front face partner, owns the shared-edge pixel");
    check(others == 10, "the other ten tower triangles do not own the pixel");
    auto missed = hit;
    missed.status = 0;
    check(!gc::rasterOwnsPixelHit(frame, hit, missed, hit.triangleId + 1)
              && gc::rasterOwnsPixelHit(frame, hit, missed, hit.triangleId),
          "a missed pixel hit grants no other triangle the pixel");

    // The OpenGL kernel applies the same rule and names the failed condition.
    const auto shader = gc::opengl::computeSource();
    check(shader.find("!causticRasterOwns(raster,hit,pixelHit)") != std::string::npos
              && shader.find("causticFailReason=10u") != std::string::npos
              && shader.find("paths[base+4].z=float(causticFailReason);") != std::string::npos,
          "the photon kernel checks the shared-edge raster owner and writes its refusal reason");
    std::array<gc::Texel, gc::kPathReceiptStride> receipt {};
    receipt[1][0] = 0;
    receipt[4][2] = float(gc::opengl::CausticWalkRefusal::RasterMismatch);
    const char* reason = nullptr;
    gc::NativePathReceipt decoded;
    check(!gc::opengl::causticDecodePath(receipt.data(), frame, 1392, decoded, &reason) && reason
              && std::string(reason).find("raster owner") != std::string::npos,
          "an incomplete receipt decodes its reason word into the refusal text");
    receipt[4][2] = 99;
    check(!gc::opengl::causticDecodePath(receipt.data(), frame, 1392, decoded, &reason) && reason
              && std::string(reason).find("malformed") != std::string::npos,
          "an out-of-range reason word is reported as malformed");
}

// The same tilted 4K earlier starter with kernel splat deposition. It keeps the
// cylinder lens because its deposits reach the grazing lantern underside, and
// because the sphere lens's 6400 work items need more gather slots than the
// splat schedule admits at 64 visits, so that frame falls back to single-pixel
// deposits (see starterSphereFocus). The CPU traces
// band 0, marks each deposit visible when its pixel-centre hit shows its own
// receiver, and builds the splat reference the native readback is checked
// against. The lantern underside is seen at view cosine 0.025: before the
// grazing spread its deposits refused with a normaliser tolerance near 0.1 of N.
void starterSplatNormalisers()
{
    auto starter = starterScene(0, nullptr, StarterLens::EarlierCylinder);
    const float tilt = -14.0f * 3.14159265f / 180;
    starter.scene->cameras[0].transform.rotation = {std::sin(tilt / 2), 0, 0, std::cos(tilt / 2)};
    const std::shared_ptr<const Visual3DScene> scene = starter.scene;
    const std::shared_ptr<const gt::Program> root = starter.root;
    const sg::CausticVolumeSelection selection {starter.glass, starter.glassMaterial};
    std::string error;
    sg::Prepared prepared;
    if (!sg::prepare(scene, root, {}, prepared, error, {}, nullptr, &selection))
    { check(false, "splat starter prepare: " + error); return; }
    gc::BranchPolicyDescription description;
    description.version = gc::kBranchPolicyVersion;
    description.maximumUnresolvedFraction = 1e-4f;
    description.minimumBranchFraction = 1e-6f;
    const auto policy = gc::AdmittedBranchPolicy::admit(description, error);
    if (!policy) { check(false, "splat starter branch policy: " + error); return; }
    gc::Prepared caustic;
    gc::Frame frame;
    if (!gc::prepare(prepared, starter.sun, *policy, caustic, error)
        || !gc::buildFrame(caustic, instances(*scene), 3840, 2160, frame, error) || !gc::enableSplat(caustic, frame, error))
    { check(false, "splat starter 4K frame: " + error); return; }

    const auto query = gc::splatCpuQuery(caustic, frame);
    std::vector<gc::NativeDeposit> deposits;
    for (std::size_t w = 0; w < frame.work.size(); ++w)
    {
        const auto& work = frame.work[w];
        if (work.band != 0) continue;
        gc::BranchReceipt receipt;
        if (!gc::traceBranches(caustic, frame, frame.entries[work.entry], work.band, *policy, receipt, error,
                               work.interfaceSample))
        { check(false, "splat starter walk: " + error); return; }
        for (const auto& branch : receipt.deposits)
        {
            gc::NativeDeposit deposit;
            deposit.work = std::uint32_t(w);
            deposit.receiver = branch.receiver;
            deposit.outgoingPower = float(work.incidentPower * branch.fraction);
            std::string ignored;
            ov::ClosestHit centre;
            if (gc::projectReceiver(frame, branch.receiver, deposit.footprint, ignored) && deposit.footprint.projected
                && query(deposit.footprint.x, deposit.footprint.y, centre, ignored) && centre.status == 1
                && centre.objectId == branch.receiver.objectId && centre.materialId == branch.receiver.materialId
                && centre.instanceId == branch.receiver.instanceId)
                deposit.cameraVisible = gc::rasterOwnsPixelHit(frame, centre, centre, branch.receiver.triangleId);
            deposits.push_back(deposit);
        }
    }
    gc::SplatReference reference;
    check(gc::buildSplatReference(frame, deposits, query, reference, error) && !reference.capOverflow,
          "the starter splat reference builds without a refusal or a cap fallback: " + error);
    if (reference.normalisers.size() != deposits.size()) return;

    const auto camera = gc::cameraPosition(frame);
    std::size_t visible = 0, grazing = 0, spread = 0;
    bool grazingSingle = true, spreadWide = true, conserved = true;
    for (std::size_t i = 0; i < deposits.size(); ++i)
    {
        const auto& deposit = deposits[i];
        if (!deposit.cameraVisible) continue;
        ++visible;
        const double power = deposit.outgoingPower;
        conserved &= std::abs(reference.splattedFlux[i] - power) <= 1e-9 * power;
        double toward[3], length = 0, facing = 0;
        for (unsigned a = 0; a < 3; ++a)
        {
            toward[a] = double(deposit.receiver.position[a]) - camera[a];
            length += toward[a] * toward[a];
            facing += toward[a] * deposit.receiver.outwardNormal[a];
        }
        const double cosine = std::abs(facing) / std::sqrt(length);
        const auto key = std::uint64_t(deposit.footprint.y) * frame.width + deposit.footprint.x;
        const auto area = reference.pixelAreas.find(key);
        if (area == reference.pixelAreas.end()) { grazingSingle = spreadWide = false; continue; }
        if (cosine < gc::kSplatSpreadCosineLow)
        {
            ++grazing;
            grazingSingle &= reference.normalisers[i] == area->second;
        }
        else if (cosine >= gc::kSplatSpreadCosineHigh)
        {
            ++spread;
            spreadWide &= reference.normalisers[i] > 2 * area->second;
        }
    }
    check(visible > 1000, "the starter has over a thousand visible band-0 deposits (" + std::to_string(visible) + ")");
    check(grazing > 0 && grazingSingle, "deposits on the grazing lantern underside keep their own pixel only ("
          + std::to_string(grazing) + ")");
    check(spread > 900 && spreadWide, "deposits on the tower and rock spread over a kernel (" + std::to_string(spread) + ")");
    check(conserved, "every visible splat deposits exactly its outgoing flux");

    check(gc::splatSpread(0.05) == 0 && std::abs(gc::splatSpread(0.15) - 0.5) < 1e-12 && gc::splatSpread(0.3) == 1,
          "the spread is 0 at view cosine 0.1, 1 at 0.2 and linear between");
    check(gc::nativeSplatMathSource().find("float causticSplatSpread(") != std::string::npos
              && gc::opengl::splatNormaliserSource().find("deposit.spread=causticSplatSpread(") != std::string::npos
              && gc::opengl::splatNormaliserSource().find("deposit.radius)*deposit.spread;") != std::string::npos
              && arbitgpu::metalcaustic::splatShader().find("deposit.spread=causticSplatSpread(") != std::string::npos
              && arbitgpu::metalcaustic::splatShader().find("deposit.radius)*deposit.spread;") != std::string::npos,
          "the GL and Metal splat passes scale neighbour weights by the same spread");

    // At 4K the native raster and the CPU centre ray can name different
    // surfaces at window pixels within a vertex snap of an edge: (1610, 895)
    // sits within 1/2048 pixel of the lens silhouette over the tower front.
    // The reference probes kSplatRasterSnapMargin (1/64 pixel) around such
    // centres, which brackets the 1/256 pixel vertex snap.
    // The edge-aware reference keeps every centre-ray value and admits the
    // other outcome, so a raster that shades the glass there (reading zero)
    // or the tower where the ray met glass no longer refuses the frame.
    check(gc::kSplatRasterSnapMargin == 1.0f / 64, "edge rays sit 1/64 pixel from the centre");
    gc::SplatReference edged;
    check(gc::buildSplatReference(frame, deposits, query, edged, error, gc::splatCpuOffsetQuery(caustic, frame))
              && !edged.capOverflow,
          "the edge-aware starter splat reference builds without a refusal: " + error);
    check(edged.edgePixels > 0, "the 4K starter has window pixels near a surface edge (" + std::to_string(edged.edgePixels) + ")");
    std::size_t zeroAdmitted = 0, shadeAdmitted = 0;
    bool sameValues = edged.pixels.size() >= reference.pixels.size();
    for (const auto& [key, sum] : edged.pixels)
    {
        const auto plain = reference.pixels.find(key);
        const double plainValue = plain == reference.pixels.end() ? 0.0 : plain->second.physical;
        sameValues &= std::abs(sum.physical - plainValue) <= 1e-12 * std::max(1.0, plainValue);
        if (plain != reference.pixels.end() && plainValue > 0 && sum.tolerance + 5e-4 * sum.physical >= sum.physical
            && plain->second.tolerance + 5e-4 * plainValue < plainValue)
            ++zeroAdmitted;
        if (plain == reference.pixels.end() && sum.physical == 0 && sum.tolerance > 0)
            ++shadeAdmitted;
    }
    check(sameValues, "edge outcomes leave every centre-ray pixel value unchanged");
    check(zeroAdmitted + shadeAdmitted > 0, "edge pixels admit the raster's other outcome (zero at "
          + std::to_string(zeroAdmitted) + ", shaded at " + std::to_string(shadeAdmitted) + " pixel bands)");
    for (std::size_t i = 0; i < deposits.size() && i < edged.normalisers.size(); ++i)
        if (deposits[i].cameraVisible)
            sameValues &= edged.normalisers[i] == reference.normalisers[i]
                && edged.normaliserTolerances[i] >= reference.normaliserTolerances[i];
    check(sameValues, "edge outcomes keep each normaliser and only widen its tolerance");

    // Banded native passes: no single 4K gather or normaliser submission.
    const auto rows4k = gc::opengl::splatGatherBandRows(3840), rows1080 = gc::opengl::splatGatherBandRows(1920);
    check(rows4k == 64 && rows1080 == 128 && std::uint64_t(rows4k) * 3840 <= gc::opengl::kSplatGatherBandPixels
              && rows4k % gc::kSplatGatherGroup == 0 && gc::opengl::splatGatherBandRows(100000) == gc::kSplatGatherGroup,
          "4K gathers run in 64-row bands of at most 262144 pixels, whole 16-row groups");
    check(gc::opengl::splatGatherSource().find("uvec2 pixel=gl_GlobalInvocationID.xy+uvec2(0u,uRowOffset);") != std::string::npos
              && gc::opengl::splatNormaliserSource().find("uint index=gl_GlobalInvocationID.x+uSlotOffset;") != std::string::npos,
          "the GL gather and normaliser passes read their band offsets");
}
// The starter's sphere lens focuses the sun onto the tower front face just
// below the lantern. 50 of its 100 triangles face the sun, so 800 entries in 8
// bands. Every band-0 walk completes within the policy, and at least half of
// the incident lens power lands within 5 cm of (-1.2, 2.2) on the front face
// (z -0.875). The model in lighthouse_starter_walk_model.py puts 0.65 there.
void starterSphereFocus()
{
    auto starter = starterScene();
    const float tilt = -14.0f * 3.14159265f / 180;
    starter.scene->cameras[0].transform.rotation = {std::sin(tilt / 2), 0, 0, std::cos(tilt / 2)};
    const std::shared_ptr<const Visual3DScene> scene = starter.scene;
    const std::shared_ptr<const gt::Program> root = starter.root;
    check(scene->objects[3].vertexCount == 52 && scene->objects[3].indexCount == 300,
          "the lens is the 10-segment, 6-ring generator sphere");
    const sg::CausticVolumeSelection selection {starter.glass, starter.glassMaterial};
    std::string error;
    sg::Prepared prepared;
    if (!sg::prepare(scene, root, {}, prepared, error, {}, nullptr, &selection))
    { check(false, "sphere starter prepare: " + error); return; }
    gc::BranchPolicyDescription description;
    description.version = gc::kBranchPolicyVersion;
    description.maximumUnresolvedFraction = 1e-4f;
    description.minimumBranchFraction = 1e-6f;
    const auto policy = gc::AdmittedBranchPolicy::admit(description, error);
    if (!policy) { check(false, "sphere starter branch policy: " + error); return; }
    gc::Prepared caustic;
    gc::Frame frame;
    if (!gc::prepare(prepared, starter.sun, *policy, caustic, error)
        || !gc::buildFrame(caustic, instances(*scene), 3840, 2160, frame, error))
    { check(false, "sphere starter 4K frame: " + error); return; }
    check(frame.work.size() == 6400, "the sphere starter schedules 800 lens entries in 8 bands");

    bool allComplete = true;
    std::size_t maximumVisits = 0;
    double incident = 0, focused = 0;
    const auto tower = starter.scene->objects[1].id.value;
    for (const auto& work : frame.work)
    {
        if (work.band != 0) continue;
        gc::BranchReceipt receipt;
        const bool complete = gc::traceBranches(caustic, frame, frame.entries[work.entry], work.band, *policy,
                                                receipt, error, work.interfaceSample);
        allComplete &= complete && receipt.withinErrorBound;
        maximumVisits = std::max(maximumVisits, receipt.visits);
        incident += work.incidentPower;
        for (const auto& branch : receipt.deposits)
        {
            const auto& at = branch.receiver.position;
            if (branch.receiver.objectId == tower && std::abs(at[2] + 0.875f) < 1e-3f
                && std::hypot(double(at[0]) + 1.2, double(at[1]) - 2.2) <= 0.05)
                focused += work.incidentPower * branch.fraction;
        }
    }
    check(allComplete, "every sphere lens walk completes within the 1e-4 tail policy: " + error);
    check(maximumVisits < gc::kMaximumBranchVisits, "no sphere lens walk reaches the visit cap");
    check(incident > 0 && focused >= 0.5 * incident,
          "the lens focuses at least half its power into the tower-front spot ("
          + std::to_string(incident > 0 ? focused / incident : 0.0) + ")");
}

// Glass Caustics off. A scene of one thin smooth water interface and one
// closed smooth volume gets a camera-only selection: the same second slot,
// per-draw uniforms and pair budget as the caustic selection, and no caustic
// preparation. Any other dielectric count derives nothing and keeps the
// single-dielectric refusal.
void cameraOnlyVolume()
{
    SceneObjectId water;
    SceneMaterialId waterMaterial;
    auto f = lighthouse(water, waterMaterial);
    const std::shared_ptr<const Visual3DScene> scene = f.scene;
    const std::shared_ptr<const gt::Program> root = f.program;
    const auto volume = f.scene->objects[0].id;
    const auto receiver = f.scene->objects[1].id;
    const sg::CausticVolumeSelection caustic {volume, f.scene->objects[0].material};
    std::string error;

    const auto derived = sg::cameraVolumeSelection(*scene, root.get());
    if (!derived) { check(false, "lighthouse derives a camera-only selection"); return; }
    check(derived->cameraOnly && !caustic.cameraOnly, "a caustic selection is not camera-only by default");
    sg::Prepared camera, selected;
    if (!sg::prepare(scene, root, {}, camera, error, {}, nullptr, &*derived)
        || !sg::prepare(scene, root, {}, selected, error, {}, nullptr, &caustic))
    { check(false, "camera-only and caustic admission: " + error); return; }
    check(camera.enabled && camera.transport == surfacematerial::TransportMode::ThinInterface
              && camera.object == water && camera.causticVolume.present && camera.causticVolume.object == volume
              && camera.causticVolume.cameraIor == 1.33f && camera.causticVolume.cameraOnly
              && !selected.causticVolume.cameraOnly,
          "camera-only admission keeps the water as camera dielectric and the volume in the second slot");
    for (const auto draw : {volume, water, receiver})
        check(sg::cameraTransport(camera, draw, 1, 0) == sg::cameraTransport(selected, draw, 1, 0)
                  && sg::cameraPartner(camera, draw) == sg::cameraPartner(selected, draw),
              "a camera-only draw uploads the same paired uniforms as the caustic selection");
    check(sg::admitFrame(camera, 1920, 1080, false, error) && sg::admitFrame(camera, 3840, 2160, false, error),
          "camera-only frames at 1080p and 4K are admitted under the pair budget: " + error);
    std::string cameraRefusal, causticRefusal;
    check(!sg::admitFrame(camera, 4096, 4096, false, cameraRefusal)
              && !sg::admitFrame(selected, 4096, 4096, false, causticRefusal) && cameraRefusal == causticRefusal,
          "camera-only and caustic frames share the pair ceiling: " + cameraRefusal);

    // No caustic resources come from a camera-only slot.
    gc::Prepared refusedCaustic;
    check(!gc::prepareScene(camera, "camera-only", refusedCaustic, error)
              && error == "Glass caustics require a caustic volume selection, not a camera-only volume",
          "caustic preparation refuses a camera-only slot: " + error);

    // A rough volume beside the water is refused without a caustic: the paired
    // walk shades the volume as smooth glass.
    auto roughVolume = lighthouse(water, waterMaterial);
    roughVolume.program->parameters.emissionRoughness[3] = 0.3f;
    const std::shared_ptr<const Visual3DScene> roughScene = roughVolume.scene;
    const std::shared_ptr<const gt::Program> roughRoot = roughVolume.program;
    const auto roughDerived = sg::cameraVolumeSelection(*roughScene, roughRoot.get());
    const sg::RoughNativeConsumer consumer {arbitgpu::NativeFixtureMaterialBackend::OpenGl, true, true, true};
    sg::Prepared roughRefused;
    check(roughDerived && !sg::prepare(roughScene, roughRoot, {}, roughRefused, error, consumer, nullptr, &*roughDerived)
              && error == "Camera glass volume beside water requires a smooth closed volume",
          "a rough closed volume beside water is refused with Glass Caustics off: " + error);

    // A lone volume or a lone water interface derives nothing.
    auto single = gt::ordinaryFixture();
    check(!sg::cameraVolumeSelection(*single.scene, single.program.get()),
          "a lone closed volume derives no camera-only selection");
    sg::CausticVolumeSelection forced {volume, single.scene->objects[0].material, true};
    const std::shared_ptr<const Visual3DScene> singleScene = single.scene;
    const std::shared_ptr<const gt::Program> singleRoot = single.program;
    sg::Prepared alone;
    check(!sg::prepare(singleScene, singleRoot, {}, alone, error, {}, nullptr, &forced)
              && error == "Camera glass volume requires one thin water interface beside it",
          "a camera-only selection without water is refused: " + error);
    check(sg::prepare(singleScene, singleRoot, {}, alone, error) && !alone.causticVolume.present,
          "a lone volume keeps the single-dielectric admission: " + error);
}

// The starter with Glass Caustics off: water plus the glass lens derive the
// camera-only selection and admit at 4K. A second closed volume refuses.
void starterCameraOnly()
{
    auto starter = starterScene();
    const std::shared_ptr<const Visual3DScene> scene = starter.scene;
    const std::shared_ptr<const gt::Program> root = starter.root;
    std::string error;
    const auto derived = sg::cameraVolumeSelection(*scene, root.get());
    check(derived && derived->cameraOnly && derived->object == starter.glass
              && derived->material == starter.glassMaterial,
          "the starter derives a camera-only selection of its glass lens");
    sg::Prepared camera;
    check(derived && sg::prepare(scene, root, {}, camera, error, {}, nullptr, &*derived),
          "the starter is admitted with Glass Caustics off: " + error);
    check(camera.object == starter.water && camera.causticVolume.present
              && camera.causticVolume.object == starter.glass && camera.causticVolume.cameraOnly,
          "the starter camera-only slot holds the glass beside the water");
    check(sg::admitFrame(camera, 3840, 2160, false, error),
          "the starter camera-only frame at 3840x2160 is admitted: " + error);

    // Turn the lighthouse tower (object 1, a closed cube) into a second glass volume.
    auto twoVolumes = starterScene();
    auto tower = std::make_shared<gt::Program>(*twoVolumes.root->objectPrograms[1]);
    tower->parameters.identifiers[1] = std::uint32_t(surfacematerial::TransportMode::ClosedVolume);
    tower->parameters.baseColorMetallic = {1, 1, 1, 0};
    tower->parameters.transmissionIorClearcoat = {1, 1.5f, 0, 0};
    tower->parameters.emissionRoughness = {0, 0, 0, 0};
    tower->spectralReflectance.reset();
    twoVolumes.root->objectPrograms[1] = tower;
    const std::shared_ptr<const Visual3DScene> twoScene = twoVolumes.scene;
    const std::shared_ptr<const gt::Program> twoRoot = twoVolumes.root;
    check(!sg::cameraVolumeSelection(*twoScene, twoRoot.get()),
          "two closed volumes derive no camera-only selection");
    sg::Prepared refused;
    check(!sg::prepare(twoScene, twoRoot, {}, refused, error)
              && error == "Closed glass transport requires one dielectric with unit opacity and transmission",
          "two closed volumes beside water keep the released refusal: " + error);
    const sg::CausticVolumeSelection forced {starter.glass, starter.glassMaterial, true};
    check(!sg::prepare(twoScene, twoRoot, {}, refused, error, {}, nullptr, &forced)
              && error == "Closed glass transport requires one dielectric with unit opacity and transmission",
          "a forced camera-only selection beside a second volume is refused: " + error);
}

float lensDot(const ov::Point& a, const ov::Point& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
ov::Point lensAlong(ov::Point p, const ov::Point& d, float s)
{
    for (int c = 0; c < 3; ++c) p[c] += s * d[c];
    return p;
}
ov::Point lensUnit(ov::Point p)
{
    const float length = std::sqrt(lensDot(p, p));
    for (auto& c : p) c /= length;
    return p;
}
float lensBias(const ov::Point& p)
{
    return std::max(1.0e-5f, 2.0e-6f * std::max({std::abs(p[0]), std::abs(p[1]), std::abs(p[2])}));
}
// glassFresnel in fixture_smooth_glass.h. The normal faces the incoming ray.
float lensFresnel(const ov::Point& incoming, const ov::Point& normal, float from, float to,
                  ov::Point& transmitted, bool& tir)
{
    const float cosine = std::clamp(-lensDot(incoming, normal), 0.0f, 1.0f);
    const float ratio = from / to, sineSquared = ratio * ratio * std::max(0.0f, 1 - cosine * cosine);
    tir = sineSquared > 1;
    transmitted = {};
    if (tir) return 1;
    const float exitCosine = std::sqrt(std::max(0.0f, 1 - sineSquared));
    for (int c = 0; c < 3; ++c) transmitted[c] = ratio * incoming[c] + (ratio * cosine - exitCosine) * normal[c];
    transmitted = lensUnit(transmitted);
    const float s = (from * cosine - to * exitCosine) / (from * cosine + to * exitCosine);
    const float p = (to * cosine - from * exitCosine) / (to * cosine + from * exitCosine);
    return std::clamp(0.5f * (s * s + p * p), 0.0f, 1.0f);
}
// The volume draw's glassPairWalk for nondispersive glass, over the CPU scene
// query: heaviest branch first, the pooled 48 visits, 16 pending slots, the
// 1e-5 branch and 1e-4 discarded power thresholds, GLASS_INTERFACE_DEPTH and
// the truncation tolerance. Returns 0 when the walk completes, else the
// SmoothTransportStatus the shader would write (Trapped is a completed pixel
// whose truncated weight was absorbed). visits narrows the pooled budget.
int lensWalk(const ov::Prepared& query, const ov::Frame& frame, const ov::Point& camera, const ov::Point& incoming,
             std::uint32_t water, std::uint32_t volume, float& truncated, int visits = 48)
{
    using Status = sg::SmoothTransportStatus;
    struct Branch { ov::Point origin, direction; float weight; int depth; bool inside; };
    truncated = 0;
    const auto first = ov::closestHitCpu(query, frame, camera, incoming, 0.0f, 3.402823e38f);
    if (first.status != 1 || first.objectId != volume) return int(Status::FirstHitMismatch);
    std::vector<Branch> pending {{camera, incoming, 1.0f, 0, !first.entering}};
    float discarded = 0;
    bool spilled = false;
    for (int visit = 0; visit < std::min(visits, 48) && !pending.empty(); ++visit)
    {
        std::size_t heaviest = 0;
        for (std::size_t s = 1; s < pending.size(); ++s)
            if (pending[s].weight > pending[heaviest].weight) heaviest = s;
        const auto ray = pending[heaviest];
        pending[heaviest] = pending.back();
        pending.pop_back();
        const auto hit = ov::closestHitCpu(query, frame, ray.origin, ray.direction, 0.0f, 3.402823e38f);
        if (hit.status < 0) return int(Status::QueryRefused);
        const bool thin = hit.status == 1 && hit.objectId == water;
        if (hit.status == 0 || (!thin && hit.objectId != volume))
        {
            if (ray.inside) return int(Status::MediumMismatch);
            continue;
        }
        if (thin && ray.inside) return int(Status::MediumMismatch);
        if (!thin && ray.inside == hit.entering) return int(Status::OrientationMismatch);
        if (ray.depth >= int(sg::kMaximumInterfaceDepth))
        {
            if (ray.inside) truncated += ray.weight;
            continue;
        }
        auto normal = hit.outwardNormal;
        if (!hit.entering) for (auto& c : normal) c = -c;
        const float from = thin ? (hit.entering ? 1.0f : 1.33f) : (ray.inside ? 1.5f : 1.0f);
        const float to = thin ? (hit.entering ? 1.33f : 1.0f) : (ray.inside ? 1.0f : 1.5f);
        ov::Point transmitted;
        bool tir = false;
        const float reflection = lensFresnel(ray.direction, normal, from, to, transmitted, tir);
        const float bias = lensBias(hit.position);
        if (!tir && ray.weight * (1 - reflection) < 1.0e-5f) discarded += ray.weight * (1 - reflection);
        if (ray.weight * reflection < 1.0e-5f) discarded += ray.weight * reflection;
        if (discarded > 1.0e-4f) return int(Status::DiscardedEnergy);
        for (const bool transmission : {true, false})
        {
            const float weight = ray.weight * (transmission ? 1 - reflection : reflection);
            if ((transmission && tir) || weight < 1.0e-5f) continue;
            const auto direction = transmission ? transmitted
                : lensAlong(ray.direction, normal, -2 * lensDot(ray.direction, normal));
            const bool inside = transmission && !thin ? !ray.inside : ray.inside;
            if (pending.size() >= 16)
            {
                spilled = true;
                std::size_t lightest = 0;
                for (std::size_t s = 1; s < pending.size(); ++s)
                    if (pending[s].weight < pending[lightest].weight) lightest = s;
                if (pending[lightest].weight >= weight)
                {
                    if (inside) truncated += weight;
                    continue;
                }
                if (pending[lightest].inside) truncated += pending[lightest].weight;
                pending[lightest] = pending.back();
                pending.pop_back();
            }
            pending.push_back({lensAlong(hit.position, normal, transmission ? -bias : bias), direction, weight,
                               ray.depth + 1, inside});
        }
    }
    for (const auto& branch : pending)
        if (branch.inside) truncated += branch.weight;
    return sg::pairWalkChannelStatus(truncated, spilled);
}

// The starter with its lens replaced by a faceted lens: the Geometry
// Core icosphere, one subdivision (80 faces), radius 0.3, Glass Caustics off.
// The camera transport needs no special case for it. Rays that enter the lens
// meet its far side from inside, and the paired walk completes. A radius-0.3
// lens at the cube's centre (z -0.62) reaches 45 mm into the tower, whose
// front face is at z -0.875. Inside rays then meet the tower paint before any
// glass face, and the walk refuses them as a medium mismatch. That overlap is
// the live "inside-glass ray" refusal, not the transport.
void starterIcosphereLens()
{
    // 1 mm above the rock top (y 0.375) and 5 mm in front of the tower.
    const SceneVec3 clear {-1.2f, 0.676f, -0.57f};
    auto starter = starterScene(0, &clear);
    const std::shared_ptr<const Visual3DScene> scene = starter.scene;
    const std::shared_ptr<const gt::Program> root = starter.root;
    std::string error;
    check(scene->objects[3].indexCount == 240 && scene->objects[3].vertexCount == 42,
          "the lens is the 80-face generator icosphere");
    const auto derived = sg::cameraVolumeSelection(*scene, root.get());
    sg::Prepared camera;
    if (!derived || !sg::prepare(scene, root, {}, camera, error, {}, nullptr, &*derived))
    { check(false, "the icosphere lens passes closed-volume admission: " + error); return; }
    check(camera.causticVolume.present && camera.causticVolume.object == starter.glass,
          "the icosphere lens fills the camera-only volume slot");
    check(sg::admitFrame(camera, 3840, 2160, false, error), "the icosphere lens frame at 3840x2160 is admitted: " + error);
    ov::Frame queryFrame;
    if (!ov::frame(camera.query, instances(*scene), queryFrame, error))
    { check(false, "the icosphere lens scene query frame: " + error); return; }

    const ov::Point eye {0, 2.4f, 7};
    const ov::Point centre {clear.x, clear.y, clear.z};
    const auto view = lensUnit({centre[0] - eye[0], centre[1] - eye[1], centre[2] - eye[2]});
    const auto across = lensUnit({-view[2], 0, view[0]});
    const ov::Point upward {across[1] * view[2] - across[2] * view[1], across[2] * view[0] - across[0] * view[2],
                            across[0] * view[1] - across[1] * view[0]};
    const auto glass = starter.glass.value, water = starter.water.value;
    int farSide = 0, complete = 0;
    float worst = 0;
    // 25 camera rays over the central half of the lens disc.
    for (const float a : {-0.5f, -0.25f, 0.0f, 0.25f, 0.5f})
        for (const float b : {-0.5f, -0.25f, 0.0f, 0.25f, 0.5f})
        {
            const auto target = lensAlong(lensAlong(centre, across, 0.3f * a), upward, 0.3f * b);
            const auto incoming = lensUnit({target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]});
            const auto first = ov::closestHitCpu(camera.query, queryFrame, eye, incoming, 0.0f, 3.402823e38f);
            if (first.status != 1 || first.objectId != glass || !first.entering) continue;
            ov::Point transmitted;
            bool tir = false;
            lensFresnel(incoming, first.outwardNormal, 1.0f, 1.5f, transmitted, tir);
            const auto inside = lensAlong(first.position, first.outwardNormal, -lensBias(first.position));
            const auto far = ov::closestHitCpu(camera.query, queryFrame, inside, transmitted, 0.0f, 3.402823e38f);
            if (!tir && far.status == 1 && far.objectId == glass && !far.entering
                && far.distance > 0 && far.distance <= 0.6f) ++farSide;
            float truncated = 0;
            if (lensWalk(camera.query, queryFrame, eye, incoming, water, glass, truncated) == 0) ++complete;
            worst = std::max(worst, truncated);
        }
    check(farSide == 25, "every refracted inside ray meets the lens far side from inside: "
                             + std::to_string(farSide) + " of 25");
    check(complete == 25 && worst <= sg::kPairTruncatedEnergyTolerance,
          "the paired walk completes through the faceted lens: " + std::to_string(complete) + " of 25");

    // The same inside ray from the lens centre, toward the tower.
    const auto towardTower = lensUnit({0.05f, 0.1f, -1});
    const auto clearHit = ov::closestHitCpu(camera.query, queryFrame, centre, towardTower, 0.0f, 3.402823e38f);
    check(clearHit.status == 1 && clearHit.objectId == glass && !clearHit.entering,
          "clear of the tower, an inside ray leaves through the lens");
    const SceneVec3 live {-1.2f, 0.675f, -0.62f};
    auto overlapping = starterScene(0, &live);
    const std::shared_ptr<const Visual3DScene> overlapScene = overlapping.scene;
    const std::shared_ptr<const gt::Program> overlapRoot = overlapping.root;
    const auto overlapDerived = sg::cameraVolumeSelection(*overlapScene, overlapRoot.get());
    // At the live placement the lens reaches 45 mm into the tower. Admission
    // now refuses that overlap by name instead of leaving it to the walk.
    sg::Prepared overlap;
    check(overlapDerived && !sg::prepare(overlapScene, overlapRoot, {}, overlap, error, {}, nullptr, &*overlapDerived)
              && error.rfind(sg::kOpaqueInterpenetrationReason, 0) == 0
              && error.find("opaque object " + std::to_string(overlapScene->objects[1].id.value)) != std::string::npos,
          "the lens sunk into the tower is refused as interpenetration: " + error);
}
// A camera walk that runs out of visits with power inside the lens ends there.
// The pixel is trapped: the untraced power is absorbed and the frame still
// publishes, until trapped pixels become systemic.
void trappedCameraPixels()
{
    // (a) The icosphere lens walk completes with the full budget. With one
    // visit the entering ray leaves its refracted branch, about 0.96 of the
    // power, inside the lens. That is far past the tolerance, never spilled,
    // so the pixel is trapped rather than refused.
    const SceneVec3 clear {-1.2f, 0.676f, -0.57f};
    auto starter = starterScene(0, &clear);
    const std::shared_ptr<const Visual3DScene> scene = starter.scene;
    const std::shared_ptr<const gt::Program> root = starter.root;
    std::string error;
    const auto derived = sg::cameraVolumeSelection(*scene, root.get());
    sg::Prepared camera;
    if (!derived || !sg::prepare(scene, root, {}, camera, error, {}, nullptr, &*derived))
    { check(false, "the trapped-ray lens passes closed-volume admission: " + error); return; }
    ov::Frame queryFrame;
    if (!ov::frame(camera.query, instances(*scene), queryFrame, error))
    { check(false, "the trapped-ray lens scene query frame: " + error); return; }
    const ov::Point eye {0, 2.4f, 7};
    const auto incoming = lensUnit({clear.x - eye[0], clear.y - eye[1], clear.z - eye[2]});
    float truncated = 0;
    check(lensWalk(camera.query, queryFrame, eye, incoming, starter.water.value, starter.glass.value, truncated) == 0
              && truncated <= sg::kPairTruncatedEnergyTolerance,
          "the centre lens ray completes with the pooled 48 visits");
    const int exhausted = lensWalk(camera.query, queryFrame, eye, incoming, starter.water.value, starter.glass.value,
                                   truncated, 1);
    check(exhausted == int(sg::SmoothTransportStatus::Trapped) && truncated > 0.5f && truncated <= 1.0f,
          "a lens ray out of visits is trapped with its inside power absorbed: status " + std::to_string(exhausted)
              + ", absorbed " + std::to_string(truncated));
    check(sg::pairWalkChannelStatus(0.5f, true) == int(sg::SmoothTransportStatus::PendingBranches)
              && sg::pairWalkChannelStatus(0.5f, false) == int(sg::SmoothTransportStatus::Trapped)
              && sg::pairWalkChannelStatus(sg::kPairTruncatedEnergyTolerance, false) == 0,
          "only an unspilled walk past the tolerance is trapped; a spilled one still refuses");

    // The readback carries trapped pixels as +2.0 normal alpha.
    constexpr std::uint32_t width = 640, height = 360;
    const auto frameWith = [&](std::size_t trappedCount, std::uint16_t extra) {
        arbitgpu::NativeRawPassPixels pixels {arbitgpu::NativeTexturePixelFormat::Rgba16Float, width, height,
                                              std::vector<std::uint8_t>(std::size_t(width) * height * 8)};
        for (std::size_t texel = 0; texel < std::size_t(width) * height; ++texel)
        {
            const std::uint16_t alpha = texel < trappedCount ? sg::kTrappedNormalBits : std::uint16_t(0x3c00u);
            std::memcpy(pixels.bytes.data() + texel * 8 + 6, &alpha, 2);
        }
        if (extra) std::memcpy(pixels.bytes.data() + (std::size_t(width) * height - 1) * 8 + 6, &extra, 2);
        return pixels;
    };
    std::uint64_t trappedPixels = 0;
    error.clear();
    check(sg::completeFrame(frameWith(16, 0), error, &trappedPixels) && trappedPixels == 16,
          "a frame with 16 trapped pixels is admitted and counts them: " + error);
    // A hard refusal still refuses next to trapped pixels.
    error.clear();
    check(!sg::completeFrame(frameWith(16, sg::negativeHalfBits(int(sg::SmoothTransportStatus::InterfaceDepth))), error)
              && error.find("interface depth budget exhausted; 1 pixel") != std::string::npos,
          "an interface-depth pixel still refuses beside trapped pixels: " + error);

    // (b) Systemic trapping refuses: 0.5% of 230400 drawn pixels is 1152.
    error.clear();
    check(sg::completeFrame(frameWith(1100, 0), error), "1100 trapped of 230400 drawn pixels is admitted: " + error);
    error.clear();
    check(!sg::completeFrame(frameWith(1200, 0), error)
              && error.find("Closed glass ray transport did not produce a complete finite result (bounce budget exhausted; 1200 trapped pixels")
                     == 0
              && error.find("first at texel x=0 y=0") != std::string::npos,
          "1200 trapped of 230400 drawn pixels refuses the frame: " + error);
    // At 3840x2160 the absolute cap bites first.
    const std::uint64_t uhd = 3840ull * 2160ull;
    check(sg::admitTrappedCameraPixels(16, uhd) && sg::admitTrappedCameraPixels(4096, uhd)
              && !sg::admitTrappedCameraPixels(4097, uhd) && !sg::admitTrappedCameraPixels(1, 100)
              && sg::admitTrappedCameraPixels(0, 0),
          "the trapped cap is 4096 pixels and 0.5% of drawn pixels");
}
} // namespace

int main()
{
    admission();
    roughWater();
    waterAbsorbs();
    nativeParity();
    cameraUniforms();
    pairedShaderText();
    starterAdmission();
    starterSharedEdgeRaster();
    starterSplatNormalisers();
    starterSphereFocus();
    cameraOnlyVolume();
    starterCameraOnly();
    starterIcosphereLens();
    trappedCameraPixels();
    std::printf("lighthouse caustic volume: %d/%d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
