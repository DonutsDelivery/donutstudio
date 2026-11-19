// Score-driven scenes in Path Trace (slice ptscore). Path Trace used to refuse
// score frames without a mapping, note appearance, note instancing on more than
// one object, and diffraction programs and products, and to tell the user to
// choose Raster. Each case now renders, and each check compares the traced
// frame with what the raster draws for it:
//
//  - coverage: every pixel's object id against the raster's vertex transform,
//    objectMatrix * (position * meshScale + note position), rasterised here;
//  - note appearance: rgb * (shaded + a), with nothing lit, so shaded is 0;
//  - a diffraction product's occupancy: the foil inside its rectangles and the
//    object's own material outside, as the raster's diffractionScatterCoverage;
//  - an advancing frame refits the scene and its BVH, and the image equals a
//    fresh build of the same frame. A changed note count builds again.
//
// ARBIT_PTSCORE_PNG=dir writes the coverage and appearance frames there.
// ARBIT_PTSCORE_OPTIX=1 times advancing frames on OptiX, refit against rebuild.

#include "../src/canonical_block_c_frame.h"
#include "../src/pathtrace_fixture_engine.h"
#include "../src/pathtrace/engine.h"
#include "../src/pathtrace/integrator.h"
#include "DiffractionMaterialPresets.h"
#include "support/fixture_scene.h"
#if defined(ARBIT_HAVE_OPTIX_KERNEL)
#include "../src/pathtrace/devices/device_optix_host.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace
{

namespace ptfixture = videorender::fixture3d::pathtracefixture;
namespace pt = videohelper::pathtrace;
namespace vni = visualnoteinstancing;
using HarmonicMIDI::grid::Visual3DScene;

int checks = 0;
int failures = 0;

void check (bool condition, const std::string& message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::fprintf (stderr, "FAIL: %s\n", message.c_str());
    }
}

float halfToFloat (std::uint16_t bits)
{
    const std::uint32_t sign = (bits >> 15) & 1u, exponent = (bits >> 10) & 31u, mantissa = bits & 1023u;
    float value = exponent == 0 ? std::ldexp (static_cast<float> (mantissa), -24)
                : exponent == 31 ? INFINITY
                : std::ldexp (static_cast<float> (mantissa | 1024u), static_cast<int> (exponent) - 25);
    return sign != 0 ? -value : value;
}

// A stored-block PNG, so the frames need no image library.
void writePng (const std::string& path, std::uint32_t width, std::uint32_t height, const std::vector<std::uint8_t>& rgbTopFirst)
{
    std::vector<std::uint8_t> raw;
    for (std::uint32_t y = 0; y < height; ++y)
    {
        raw.push_back (0);
        raw.insert (raw.end(), rgbTopFirst.begin() + static_cast<std::ptrdiff_t> (y * width * 3),
                    rgbTopFirst.begin() + static_cast<std::ptrdiff_t> ((y + 1) * width * 3));
    }
    std::vector<std::uint8_t> z { 0x78, 0x01 };
    std::uint32_t a = 1, b = 0;
    for (const auto byte : raw) { a = (a + byte) % 65521u; b = (b + a) % 65521u; }
    for (std::size_t at = 0; at < raw.size(); at += 65535)
    {
        const std::size_t n = std::min<std::size_t> (65535, raw.size() - at);
        z.push_back (at + n == raw.size() ? 1 : 0);
        z.push_back (static_cast<std::uint8_t> (n)); z.push_back (static_cast<std::uint8_t> (n >> 8));
        z.push_back (static_cast<std::uint8_t> (~n)); z.push_back (static_cast<std::uint8_t> (~n >> 8));
        z.insert (z.end(), raw.begin() + static_cast<std::ptrdiff_t> (at), raw.begin() + static_cast<std::ptrdiff_t> (at + n));
    }
    const std::uint32_t adler = (b << 16) | a;
    for (int s = 24; s >= 0; s -= 8) z.push_back (static_cast<std::uint8_t> (adler >> s));
    std::FILE* file = std::fopen (path.c_str(), "wb");
    if (file == nullptr) return;
    const auto put32 = [file] (std::uint32_t v) { for (int s = 24; s >= 0; s -= 8) std::fputc (static_cast<int> ((v >> s) & 255u), file); };
    const auto chunk = [&] (const char* type, const std::vector<std::uint8_t>& data) {
        put32 (static_cast<std::uint32_t> (data.size()));
        std::vector<std::uint8_t> body (type, type + 4);
        body.insert (body.end(), data.begin(), data.end());
        std::fwrite (body.data(), 1, body.size(), file);
        std::uint32_t crc = 0xffffffffu;
        for (const auto byte : body)
        {
            crc ^= byte;
            for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
        }
        put32 (~crc);
    };
    std::fwrite ("\x89PNG\r\n\x1a\n", 1, 8, file);
    std::vector<std::uint8_t> header;
    for (const auto v : { width, height }) for (int s = 24; s >= 0; s -= 8) header.push_back (static_cast<std::uint8_t> (v >> s));
    header.insert (header.end(), { 8, 2, 0, 0, 0 });
    chunk ("IHDR", header);
    chunk ("IDAT", z);
    chunk ("IEND", {});
    std::fclose (file);
}

std::string pngDirectory()
{
    const char* dir = std::getenv ("ARBIT_PTSCORE_PNG");
    return dir != nullptr ? std::string (dir) : std::string {};
}

// ---------------------------------------------------------------- score frames

struct ScoreFrame
{
    std::shared_ptr<arbitmod::Score> score = std::make_shared<arbitmod::Score>();
    canonicalblockc::FrameProducer producer;
    canonicalblockc::FrameKey key;

    ScoreFrame (std::initializer_list<std::pair<int, float>> notes)
    {
        score->scoreRevision = 1;
        for (const auto& [midi, velocity] : notes)
        {
            arbitmod::Note note;
            note.id = -midi; note.midiNote = static_cast<float> (midi);
            note.freqHz = 440.0f * std::exp2 ((static_cast<float> (midi) - 69.0f) / 12.0f);
            note.velocity = velocity; note.lengthBeats = 8;
            score->notes.push_back (note);
        }
        key.projectGeneration = key.sourceGeneration = key.helperGeneration = 1;
        key.backendGeneration = key.deviceGeneration = key.scoreGeneration = 1;
        key.beatMapGeneration = key.fpsGeneration = key.loopGeneration = key.seekGeneration = 1;
        key.fps = 24;
    }

    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> at (float beat)
    {
        ++key.frame;
        key.beat = beat;
        return producer.evaluate (key, score, beat);
    }
};

// Notes along x by pitch, rising in y with their age, so a later beat moves every
// instance and keeps the count.
vni::Mapping noteMapping()
{
    vni::Mapping mapping;
    mapping.schemaVersion = vni::kDirectAxesMappingSchemaVersion;
    mapping.x = vni::MappingAxis::MidiNote;
    mapping.y = vni::MappingAxis::NoteAgeBeats;
    mapping.z = vni::MappingAxis::MidiNote;
    mapping.xScale = 0.25f; mapping.yScale = 0.5f; mapping.zScale = 0.0f;
    mapping.meshScale = 0.3f;
    return mapping;
}

// The fixture cube twice, unrotated, one above the other, offset so notes 60 to 67
// sit around the camera axis.
std::shared_ptr<Visual3DScene> twoCubeScene()
{
    auto scene = std::make_shared<Visual3DScene> (videohelper::fixture3d::makeScene());
    scene->objects[0].transform.rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    scene->objects[0].transform.translation = { -15.875f, 0.45f, 0.0f };
    scene->objects[1] = scene->objects[0];
    scene->objects[1].id = HarmonicMIDI::grid::SceneObjectId { 102 };
    scene->objects[1].transform.translation = { -15.875f, -0.85f, 0.0f };
    scene->objectCount = 2;
    return scene;
}

pt::EngineSceneSetup baseSetup (const std::shared_ptr<const Visual3DScene>& scene,
                                const arbitgpu::NativeFixtureSceneRuntimeInputs& inputs, bool lit)
{
    pt::EngineSceneSetup setup;
    setup.scene = scene;
    setup.emissionGain = inputs.emissionGain;
    for (std::size_t index = 0; index < scene->objectCount; ++index)
        setup.objectWorlds.push_back (pt::objectWorldMatrix (*scene, index));
    if (lit)
        for (std::size_t index = 0; index < arbitgpu::effectiveNativeLightCount (*scene, inputs); ++index)
            setup.lights.push_back (arbitgpu::effectiveNativeLightRecord (*scene, inputs, index));
    return setup;
}

pt::EngineView viewOf (const Visual3DScene& scene, std::uint32_t width, std::uint32_t height)
{
    pt::EngineView view;
    view.camera = *HarmonicMIDI::grid::visual3d_detail::findById (scene.cameras, scene.cameraCount, scene.activeCamera);
    view.width = width;
    view.height = height;
    return view;
}

// The raster's coverage: each object once per note at objectMatrix * (position *
// meshScale + note), nearest first, sampled at pixel centres. Top row first.
std::vector<std::uint32_t> rasterObjectIds (const Visual3DScene& scene, const arbitgpu::NativeNoteInstanceBatch& batch,
                                            const pt::EngineView& view)
{
    pt::PinholeCamera camera;
    pt::makePinholeCamera (view.camera, view.width, view.height, camera);
    std::vector<std::uint32_t> ids (static_cast<std::size_t> (view.width) * view.height, 0);
    std::vector<float> depth (ids.size(), INFINITY);
    for (std::size_t object = 0; object < scene.objectCount; ++object)
    {
        const auto& record = scene.objects[object];
        const pt::Mat4 world = pt::objectWorldMatrix (scene, object);
        for (std::size_t note = 0; note < batch.count; ++note)
        {
            const auto& placed = batch.transforms[note];
            for (std::size_t tri = 0; tri + 3 <= record.indexCount; tri += 3)
            {
                float px[3], py[3], pd[3];
                bool visible = true;
                for (int corner = 0; corner < 3; ++corner)
                {
                    const auto& p = scene.vertices[record.firstVertex + scene.indices[record.firstIndex + tri + static_cast<std::size_t> (corner)]].position;
                    const pt::Vec3 local { p.x * placed[3] + placed[0], p.y * placed[3] + placed[1], p.z * placed[3] + placed[2] };
                    visible = camera.project (pt::transformPoint (world, local), px[corner], py[corner], pd[corner]) && visible;
                }
                if (! visible) continue;
                const float area = (px[1] - px[0]) * (py[2] - py[0]) - (px[2] - px[0]) * (py[1] - py[0]);
                if (std::abs (area) < 1.0e-9f) continue;
                for (std::uint32_t y = 0; y < view.height; ++y)
                    for (std::uint32_t x = 0; x < view.width; ++x)
                    {
                        const float cx = static_cast<float> (x) + 0.5f, cy = static_cast<float> (y) + 0.5f;
                        const float w0 = ((px[1] - cx) * (py[2] - cy) - (px[2] - cx) * (py[1] - cy)) / area;
                        const float w1 = ((px[2] - cx) * (py[0] - cy) - (px[0] - cx) * (py[2] - cy)) / area;
                        const float w2 = 1.0f - w0 - w1;
                        if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;
                        const float d = w0 * pd[0] + w1 * pd[1] + w2 * pd[2];
                        const std::size_t at = static_cast<std::size_t> (y) * view.width + x;
                        if (d < depth[at]) { depth[at] = d; ids[at] = record.id.value; }
                    }
            }
        }
    }
    return ids;
}

// ---------------------------------------------------------------- admission

void admissionTests()
{
    std::string error;
    ScoreFrame frame ({ { 60, 127.0f }, { 64, 32.0f }, { 67, 80.0f } });
    arbitgpu::NativeFixtureSceneRuntimeInputs scoreOnly;
    scoreOnly.canonicalBlockCFrame = frame.at (0.25f);
    // The raster's own validity check (validFixtureRuntimeInputs) wants a mapping
    // with every score frame, so neither route draws one alone; Path Trace no
    // longer adds a refusal of its own on top.
    check (! ptfixture::admitInputs (scoreOnly, error) && error == "Path Trace runtime inputs are invalid"
               && ! arbitgpu::validFixtureRuntimeInputs (scoreOnly),
           "a score frame without a note mapping is refused only by the check the raster shares");

    auto tinted = scoreOnly;
    tinted.noteInstanceMapping = noteMapping();
    tinted.noteInstanceMapping->appearanceLow = { 0.0f, 0.2f, 1.0f, 0.5f };
    tinted.noteInstanceMapping->appearanceHigh = { 1.0f, 0.1f, 0.0f, 3.0f };
    check (ptfixture::admitInputs (tinted, error), "note appearance is admitted");

    const auto scene = twoCubeScene();
    pt::EngineSceneSetup setup;
    check (ptfixture::bindNoteInstances (*scene, tinted, false, setup, error),
           "note instancing binds on a scene of two mesh objects: " + error);
    const auto batch = arbitgpu::prepareNativeNoteInstances (tinted);
    check (setup.noteInstanced && setup.noteInstances.size() == batch.count && batch.count == 3,
           "every note is an instance of every object");
    check (setup.noteAppearance.size() == batch.count, "every note carries its appearance");
    for (std::size_t note = 0; note < batch.count && note < setup.noteAppearance.size(); ++note)
    {
        const float v = batch.appearanceValues[note];
        bool mixed = true;
        for (std::size_t c = 0; c < 4; ++c)
        {
            const float expected = tinted.noteInstanceMapping->appearanceLow[c]
                + (tinted.noteInstanceMapping->appearanceHigh[c] - tinted.noteInstanceMapping->appearanceLow[c]) * v;
            mixed = mixed && std::abs (setup.noteAppearance[note][c] - expected) < 1.0e-6f;
        }
        check (mixed, "a note's appearance is the raster's mix (low, high, value)");
        const auto& placed = setup.noteInstances[note];
        check (placed.translation.x == batch.transforms[note][0] && placed.translation.y == batch.transforms[note][1]
                   && placed.scale.x == batch.transforms[note][3] && placed.scale.z == batch.transforms[note][3],
               "a note instance sits at the raster's position and mesh scale");
    }

    // The raster refuses an instanced object under the mapping, and so does Path Trace, in the raster's words.
    auto instanced = std::make_shared<Visual3DScene> (*scene);
    instanced->instancedObject = instanced->objects[0].id;
    instanced->objectInstances.resize (1);
    instanced->objectInstanceCount = 1;
    error.clear();
    check (! ptfixture::bindNoteInstances (*instanced, tinted, false, setup, error)
               && error == "Instanced scene objects do not combine with per-note instancing",
           "an instanced object under a note mapping is refused as the raster refuses it");

    // A diffraction program maps to the object's scene material, which shows
    // where the foil's mask leaves the grating out.
    arbitgpu::NativeFixtureSurfaceMaterialProgram diffraction;
    diffraction.kind = arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;
    HarmonicMIDI::grid::SceneMaterialRecord record;
    record.baseColor = { 0.2f, 0.4f, 0.6f };
    pt::Material material;
    error.clear();
    check (ptfixture::mapSurfaceProgram (diffraction, &record, 0.0f, material, error) && error.empty()
               && std::abs (material.baseColor.y - 0.4f) < 1.0e-6f,
           "a diffraction program maps to the object's own material instead of being refused");
}

// ---------------------------------------------------------------- coverage and appearance

void coverageTests()
{
    std::string error;
    ScoreFrame frame ({ { 60, 127.0f }, { 64, 32.0f }, { 67, 80.0f } });
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.linearColor = true;
    inputs.canonicalBlockCFrame = frame.at (0.25f);
    inputs.noteInstanceMapping = noteMapping();
    const auto scene = twoCubeScene();
    auto setup = baseSetup (scene, inputs, true);
    check (ptfixture::bindNoteInstances (*scene, inputs, false, setup, error), "the notes bind: " + error);
    const auto view = viewOf (*scene, 160, 120);
    pt::SceneEngine engine (2);
    pt::EngineImage image;
    check (engine.renderExport (setup, view, pt::EngineColor::LinearRgba16f, 4, image, error),
           "Path Trace renders two objects instanced per note: " + error);
    const auto raster = rasterObjectIds (*scene, arbitgpu::prepareNativeNoteInstances (inputs), view);
    std::size_t covered = 0, mismatched = 0, edge = 0;
    for (std::uint32_t y = 0; y < view.height; ++y)
        for (std::uint32_t x = 0; x < view.width; ++x)
        {
            const std::uint32_t expected = raster[static_cast<std::size_t> (y) * view.width + x];
            const std::uint32_t traced = image.objectIdPass.empty() ? 0u
                : image.objectIdPass[static_cast<std::size_t> (view.height - 1 - y) * view.width + x];
            covered += expected != 0 ? 1 : 0;
            if (expected == traced) continue;
            ++mismatched;
            // A pixel centre on a raster edge may land either side of it in the tracer.
            bool nearEdge = false;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                {
                    const int nx = static_cast<int> (x) + dx, ny = static_cast<int> (y) + dy;
                    if (nx >= 0 && ny >= 0 && nx < static_cast<int> (view.width) && ny < static_cast<int> (view.height))
                        nearEdge = nearEdge || raster[static_cast<std::size_t> (ny) * view.width + static_cast<std::size_t> (nx)] != expected;
                }
            edge += nearEdge ? 1 : 0;
        }
    std::printf ("[ptscore] coverage: %zu raster pixels, %zu differ, %zu of them on a raster edge\n", covered, mismatched, edge);
    std::size_t first = 0, second = 0;
    for (const auto id : raster) { first += id == 101 ? 1 : 0; second += id == 102 ? 1 : 0; }
    check (first > 300 && second > 300, "the raster frame shows both objects");
    check (mismatched == edge && mismatched * 100 <= covered,
           "every traced pixel shows the raster's object, apart from edge pixels under 1%");
    if (const auto dir = pngDirectory(); ! dir.empty() && image.colorBytes.size() == static_cast<std::size_t> (view.width) * view.height * 8)
    {
        std::vector<std::uint8_t> rgb (static_cast<std::size_t> (view.width) * view.height * 3 * 2);
        for (std::uint32_t y = 0; y < view.height; ++y)
            for (std::uint32_t x = 0; x < view.width; ++x)
            {
                const std::size_t traced = (static_cast<std::size_t> (view.height - 1 - y) * view.width + x) * 8;
                const std::size_t left = (static_cast<std::size_t> (y) * view.width * 2 + x) * 3;
                const std::size_t right = left + static_cast<std::size_t> (view.width) * 3;
                for (int c = 0; c < 3; ++c)
                {
                    std::uint16_t bits = 0;
                    std::memcpy (&bits, &image.colorBytes[traced + static_cast<std::size_t> (c) * 2], 2);
                    const float linear = std::clamp (halfToFloat (bits), 0.0f, 1.0f);
                    rgb[left + static_cast<std::size_t> (c)] = static_cast<std::uint8_t> (std::pow (linear, 1.0f / 2.2f) * 255.0f + 0.5f);
                }
                const std::uint32_t id = raster[static_cast<std::size_t> (y) * view.width + x];
                rgb[right] = id == 101 ? 230 : 20; rgb[right + 1] = id == 102 ? 230 : 20; rgb[right + 2] = 40;
            }
        writePng (dir + "/coverage-traced-vs-raster.png", view.width * 2, view.height, rgb);
    }
}

void appearanceTests()
{
    std::string error;
    ScoreFrame frame ({ { 60, 127.0f }, { 64, 32.0f }, { 67, 80.0f } });
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.linearColor = true;
    inputs.canonicalBlockCFrame = frame.at (0.25f);
    inputs.noteInstanceMapping = noteMapping();
    inputs.noteInstanceMapping->appearanceLow = { 0.0f, 0.2f, 1.0f, 0.5f };
    inputs.noteInstanceMapping->appearanceHigh = { 1.0f, 0.1f, 0.0f, 3.0f };
    auto dark = std::make_shared<Visual3DScene> (*twoCubeScene());
    dark->ambientColor = { 0.0f, 0.0f, 0.0f };
    auto setup = baseSetup (dark, inputs, false);
    check (ptfixture::bindNoteInstances (*dark, inputs, false, setup, error), "the tinted notes bind: " + error);
    const auto view = viewOf (*dark, 160, 120);
    pt::SceneEngine engine (2);
    pt::EngineImage image;
    check (engine.renderExport (setup, view, pt::EngineColor::LinearRgba16f, 2, image, error),
           "Path Trace renders note appearance: " + error);
    if (image.colorBytes.size() != static_cast<std::size_t> (view.width) * view.height * 8)
        return;
    pt::PinholeCamera camera;
    pt::makePinholeCamera (view.camera, view.width, view.height, camera);
    const auto batch = arbitgpu::prepareNativeNoteInstances (inputs);
    for (std::size_t object = 0; object < dark->objectCount; ++object)
        for (std::size_t note = 0; note < batch.count; ++note)
        {
            // The front face's centre: unlit, so the raster shows rgb * a there.
            const auto& placed = batch.transforms[note];
            const pt::Vec3 centre = pt::transformPoint (pt::objectWorldMatrix (*dark, object),
                                                        { placed[0], placed[1], placed[3] + placed[2] });
            float px = 0, py = 0, depth = 0;
            if (! camera.project (centre, px, py, depth)) { check (false, "a note instance projects"); continue; }
            const std::size_t at = (static_cast<std::size_t> (view.height - 1 - static_cast<std::uint32_t> (py)) * view.width
                                    + static_cast<std::uint32_t> (px)) * 8;
            const float v = batch.appearanceValues[note];
            const auto& low = inputs.noteInstanceMapping->appearanceLow;
            const auto& high = inputs.noteInstanceMapping->appearanceHigh;
            const float a = low[3] + (high[3] - low[3]) * v;
            bool matches = true;
            for (std::size_t c = 0; c < 3; ++c)
            {
                std::uint16_t bits = 0;
                std::memcpy (&bits, &image.colorBytes[at + c * 2], 2);
                const float expected = (low[c] + (high[c] - low[c]) * v) * a;
                matches = matches && std::abs (halfToFloat (bits) - expected) <= 0.01f + 0.01f * expected;
            }
            check (matches, "an unlit note instance shows the raster's rgb * a, note " + std::to_string (note));
        }
    if (const auto dir = pngDirectory(); ! dir.empty())
    {
        std::vector<std::uint8_t> rgb (static_cast<std::size_t> (view.width) * view.height * 3);
        for (std::uint32_t y = 0; y < view.height; ++y)
            for (std::uint32_t x = 0; x < view.width; ++x)
                for (int c = 0; c < 3; ++c)
                {
                    std::uint16_t bits = 0;
                    std::memcpy (&bits, &image.colorBytes[(static_cast<std::size_t> (view.height - 1 - y) * view.width + x) * 8 + static_cast<std::size_t> (c) * 2], 2);
                    const float linear = std::clamp (halfToFloat (bits) / 3.0f, 0.0f, 1.0f);
                    rgb[(static_cast<std::size_t> (y) * view.width + x) * 3 + static_cast<std::size_t> (c)]
                        = static_cast<std::uint8_t> (std::pow (linear, 1.0f / 2.2f) * 255.0f + 0.5f);
                }
        writePng (dir + "/note-appearance.png", view.width, view.height, rgb);
    }
}

// ---------------------------------------------------------------- refit

void refitTests()
{
    std::string error;
    ScoreFrame frame ({ { 60, 127.0f }, { 64, 32.0f }, { 67, 80.0f } });
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.linearColor = true;
    inputs.noteInstanceMapping = noteMapping();
    const auto scene = twoCubeScene();
    const auto view = viewOf (*scene, 96, 72);
    const auto setupAt = [&] (float beat) {
        inputs.canonicalBlockCFrame = frame.at (beat);
        auto setup = baseSetup (scene, inputs, true);
        ptfixture::bindNoteInstances (*scene, inputs, false, setup, error);
        return setup;
    };

    pt::SceneEngine advancing (2);
    pt::EngineImage first, second;
    check (advancing.renderExport (setupAt (0.25f), view, pt::EngineColor::LinearRgba16f, 2, first, error)
               && first.sceneRebuilt && ! first.sceneRefitted,
           "the first frame builds the scene");
    const auto moved = setupAt (0.75f);
    check (advancing.renderExport (moved, view, pt::EngineColor::LinearRgba16f, 2, second, error)
               && second.sceneRefitted && ! second.sceneRebuilt && second.restarted,
           "an advancing frame refits the scene, and its accumulation starts again");
    check (advancing.sceneBuilds() == 1 && advancing.sceneRefits() == 1, "one build, then one refit");
    check (second.stats.phases.accelBuild != nullptr && std::string (second.stats.phases.accelBuild) == "refit", "the frame reports its BVH as refit");
    check (second.objectIdPass != first.objectIdPass, "the notes moved between the frames");

    pt::SceneEngine fresh (2);
    pt::EngineImage rebuilt;
    check (fresh.renderExport (moved, view, pt::EngineColor::LinearRgba16f, 2, rebuilt, error), "a fresh engine renders the frame");
    check (rebuilt.objectIdPass == second.objectIdPass && rebuilt.depth == second.depth,
           "the refit frame's coverage and depth equal a fresh build's");
    std::size_t differ = 0;
    for (std::size_t i = 0; i < rebuilt.colorBytes.size() && i < second.colorBytes.size(); i += 2)
    {
        std::uint16_t a = 0, b = 0;
        std::memcpy (&a, &rebuilt.colorBytes[i], 2);
        std::memcpy (&b, &second.colorBytes[i], 2);
        differ += std::abs (halfToFloat (a) - halfToFloat (b)) > 1.0e-3f ? 1 : 0;
    }
    check (rebuilt.colorBytes.size() == second.colorBytes.size() && differ * 1000 <= rebuilt.colorBytes.size() / 2,
           "the refit frame's colour equals a fresh build's: " + std::to_string (differ) + " channels differ");

    // A note that starts sounding changes the topology: the scene builds again.
    frame.score->notes.push_back (frame.score->notes.front());
    frame.score->notes.back().id = -72; frame.score->notes.back().midiNote = 72;
    ++frame.key.scoreGeneration; ++frame.score->scoreRevision;
    pt::EngineImage third;
    check (advancing.renderExport (setupAt (0.75f), view, pt::EngineColor::LinearRgba16f, 2, third, error)
               && third.sceneRebuilt && ! third.sceneRefitted && advancing.sceneBuilds() == 2,
           "a changed note count builds the scene again");
}

// ---------------------------------------------------------------- diffraction product occupancy

void productOccupancyTests()
{
    std::string error;
    const auto admitted = diffractionmaterial::admit (diffractionmaterial::makeAluminiumBinaryGratingPreset(), error);
    check (admitted.has_value(), "the grating preset admits: " + error);
    if (! admitted)
        return;
    // A unit quad facing the camera, u running left to right, glowing green.
    auto quad = std::make_shared<Visual3DScene> (videohelper::fixture3d::makeScene());
    const HarmonicMIDI::grid::SceneVertex corners[4] {
        {{-1,-1, 0}, { 0, 0, 1}, {}, {0,0}}, {{ 1,-1, 0}, { 0, 0, 1}, {}, {1,0}},
        {{ 1, 1, 0}, { 0, 0, 1}, {}, {1,1}}, {{-1, 1, 0}, { 0, 0, 1}, {}, {0,1}} };
    for (std::size_t i = 0; i < 4; ++i) quad->vertices[i] = corners[i];
    const std::uint32_t indices[6] { 0, 1, 2, 0, 2, 3 };
    for (std::size_t i = 0; i < 6; ++i) quad->indices[i] = indices[i];
    quad->vertexCount = 4; quad->indexCount = 6;
    quad->objects[0].transform.rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    quad->objects[0].vertexCount = 4; quad->objects[0].indexCount = 6;
    quad->materials[0].baseColorTexture = {};
    quad->materials[0].baseColor = { 0.0f, 0.0f, 0.0f };
    quad->materials[0].emissive = { 0.0f, 0.8f, 0.0f };
    quad->ambientColor = { 0.0f, 0.0f, 0.0f };
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    auto setup = baseSetup (quad, inputs, false);
    pt::ObjectGratingBinding grating;
    grating.objectId = quad->objects[0].id.value;
    grating.material = std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR> (*admitted);
    grating.occupancy[0] = { 0.0f, 0.0f, 0.5f, 1.0f };
    grating.occupancyCount = 1;
    grating.maskCoverage = 1.0f;
    setup.objectGratings.push_back (grating);
    const auto view = viewOf (*quad, 64, 48);
    pt::SceneEngine engine (2), bare (2);
    pt::EngineImage image, own;
    check (engine.renderExport (setup, view, pt::EngineColor::LinearRgba16f, 4, image, error),
           "Path Trace renders a diffraction product's occupancy: " + error);
    // The object's own material alone, the reference outside the occupancy.
    auto withoutFoil = setup;
    withoutFoil.objectGratings.clear();
    check (bare.renderExport (withoutFoil, view, pt::EngineColor::LinearRgba16f, 4, own, error), "the bare quad renders");
    if (image.colorBytes.size() != static_cast<std::size_t> (view.width) * view.height * 8 || own.colorBytes.size() != image.colorBytes.size())
        return;
    pt::PinholeCamera camera;
    pt::makePinholeCamera (view.camera, view.width, view.height, camera);
    const auto greenIn = [&] (const pt::EngineImage& frame, float u, float v) {
        float px = 0, py = 0, depth = 0;
        camera.project (pt::transformPoint (pt::objectWorldMatrix (*quad, 0), { u * 2.0f - 1.0f, v * 2.0f - 1.0f, 0.0f }), px, py, depth);
        const std::size_t at = (static_cast<std::size_t> (view.height - 1 - static_cast<std::uint32_t> (py)) * view.width
                                + static_cast<std::uint32_t> (px)) * 8;
        std::uint16_t bits = 0;
        std::memcpy (&bits, &frame.colorBytes[at + 2], 2);
        return halfToFloat (bits);
    };
    const auto greenAt = [&] (float u, float v) { return greenIn (image, u, v); };
    std::printf ("[ptscore] occupancy green: inside %.3f %.3f, outside %.3f %.3f\n",
                 greenAt (0.25f, 0.5f), greenAt (0.25f, 0.2f), greenAt (0.75f, 0.5f), greenAt (0.75f, 0.8f));
    if (const auto dir = pngDirectory(); ! dir.empty())
    {
        std::vector<std::uint8_t> rgb (static_cast<std::size_t> (view.width) * view.height * 3);
        for (std::uint32_t y = 0; y < view.height; ++y)
            for (std::uint32_t x = 0; x < view.width; ++x)
                for (int c = 0; c < 3; ++c)
                {
                    std::uint16_t bits = 0;
                    std::memcpy (&bits, &image.colorBytes[(static_cast<std::size_t> (view.height - 1 - y) * view.width + x) * 8 + static_cast<std::size_t> (c) * 2], 2);
                    rgb[(static_cast<std::size_t> (y) * view.width + x) * 3 + static_cast<std::size_t> (c)]
                        = static_cast<std::uint8_t> (std::pow (std::clamp (halfToFloat (bits), 0.0f, 1.0f), 1.0f / 2.2f) * 255.0f + 0.5f);
                }
        writePng (dir + "/product-occupancy.png", view.width, view.height, rgb);
    }
    check (greenAt (0.25f, 0.5f) < 0.05f && greenAt (0.25f, 0.2f) < 0.05f,
           "inside its occupancy rectangle the object is the foil");
    check (greenIn (own, 0.25f, 0.5f) > 0.5f, "without the foil the left half glows too");
    // A grating makes the frame spectral, so its noise differs from the bare
    // frame's; the means over the uncovered half agree.
    const auto meanGreen = [&] (const pt::EngineImage& frame) {
        double sum = 0.0;
        int n = 0;
        for (float u = 0.6f; u < 0.95f; u += 0.05f)
            for (float v = 0.1f; v < 0.95f; v += 0.05f) { sum += greenIn (frame, u, v); ++n; }
        return sum / n;
    };
    std::printf ("[ptscore] uncovered half mean green: with foil %.4f, bare %.4f\n", meanGreen (image), meanGreen (own));
    check (std::abs (meanGreen (image) - meanGreen (own)) < 0.05 * meanGreen (own),
           "outside it the object keeps its own material, as the raster's litBase");
}

// ---------------------------------------------------------------- OptiX timing

#if defined(ARBIT_HAVE_OPTIX_KERNEL)
// A UV sphere of segments x rings quads, as one mesh object.
std::shared_ptr<Visual3DScene> sphereScene (std::uint32_t segments, std::uint32_t rings)
{
    auto scene = std::make_shared<Visual3DScene> (videohelper::fixture3d::makeScene());
    const std::size_t vertices = static_cast<std::size_t> (segments + 1) * (rings + 1);
    const std::size_t indices = static_cast<std::size_t> (segments) * rings * 6;
    scene->vertices.resize (vertices);
    scene->indices.resize (indices);
    std::size_t at = 0;
    for (std::uint32_t r = 0; r <= rings; ++r)
        for (std::uint32_t s = 0; s <= segments; ++s)
        {
            const float theta = 3.14159265f * static_cast<float> (r) / static_cast<float> (rings);
            const float phi = 6.28318531f * static_cast<float> (s) / static_cast<float> (segments);
            const HarmonicMIDI::grid::SceneVec3 n { std::sin (theta) * std::cos (phi), std::cos (theta), std::sin (theta) * std::sin (phi) };
            scene->vertices[at++] = { n, n, {}, { static_cast<float> (s) / static_cast<float> (segments), static_cast<float> (r) / static_cast<float> (rings) } };
        }
    at = 0;
    for (std::uint32_t r = 0; r < rings; ++r)
        for (std::uint32_t s = 0; s < segments; ++s)
        {
            const std::uint32_t a = r * (segments + 1) + s, b = a + segments + 1;
            for (const std::uint32_t index : { a, b, a + 1, a + 1, b, b + 1 })
                scene->indices[at++] = index;
        }
    scene->vertexCount = vertices;
    scene->indexCount = indices;
    scene->objects[0].transform.rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
    scene->objects[0].transform.translation = { -7.5f, -1.5f, -6.0f };
    scene->objects[0].vertexCount = static_cast<std::uint32_t> (vertices);
    scene->objects[0].indexCount = static_cast<std::uint32_t> (indices);
    scene->materials[0].baseColorTexture = {};
    return scene;
}

void optixTiming()
{
    if (std::getenv ("ARBIT_PTSCORE_OPTIX") == nullptr)
    {
        std::printf ("[ptscore] OptiX timing skipped: set ARBIT_PTSCORE_OPTIX=1\n");
        return;
    }
    // 128 notes of a 4096-triangle sphere: 524288 triangles a frame.
    std::vector<std::pair<int, float>> notes;
    ScoreFrame frame ({});
    for (int i = 0; i < 128; ++i)
    {
        arbitmod::Note note;
        note.id = -(i + 1); note.midiNote = static_cast<float> (24 + (i % 64)); note.velocity = 100;
        note.startBeat = static_cast<float> (i / 64) * 0.5f; note.lengthBeats = 64;
        frame.score->notes.push_back (note);
    }
    vni::Mapping mapping = noteMapping();
    mapping.meshScale = 0.1f;
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.linearColor = true;
    inputs.noteInstanceMapping = mapping;
    const auto scene = sphereScene (64, 32);
    const auto view = viewOf (*scene, 960, 540);
    std::string error;

    const auto run = [&] (bool forceRebuild, const char* label) {
        pt::SceneEngine engine (1);
        engine.setDevicePass (pt::device::makeOptixDevicePass ([&engine] { return engine.sceneBuilds(); }),
                              [] { return pt::device::optixPipelineReady(); });
        double total = 0.0, lower = 0.0, bvh = 0.0, upload = 0.0, accel = 0.0;
        int timed = 0;
        std::string accelKind;
        for (int f = 0; f < 13; ++f)
        {
            inputs.canonicalBlockCFrame = frame.at (2.0f + 0.05f * static_cast<float> (f));
            auto setup = baseSetup (scene, inputs, true);
            ptfixture::bindNoteInstances (*scene, inputs, false, setup, error);
            // A gain one ulp off 1 changes nothing visible and makes every frame a new scene.
            if (forceRebuild && (f & 1) != 0)
                setup.emissionGain = std::nextafter (1.0f, 2.0f);
            pt::EngineImage image;
            const auto start = std::chrono::steady_clock::now();
            bool rendered = false;
            for (int wait = 0; wait < 1200; ++wait)
            {
                image = {};
                rendered = engine.renderPreview (setup, view, pt::EngineColor::LinearRgba16f, image, error);
                if (! rendered || ! image.pending)
                    break;
                std::this_thread::sleep_for (std::chrono::milliseconds (100));
            }
            const double ms = std::chrono::duration<double, std::milli> (std::chrono::steady_clock::now() - start).count();
            if (! rendered || image.stats.deviceFailed)
            {
                check (false, std::string ("OptiX renders the note-instanced frame: ") + error);
                return;
            }
            const auto& phases = image.stats.phases;
            const auto phase = [&phases] (pt::FramePhase p) { return phases.has (p) ? phases.get (p) : 0.0; };
            std::printf ("[ptscore] %s frame %d: %.2f ms, lower %.2f, cpu bvh %.2f, upload %.2f, accel %.2f (%s), refit=%d rebuilt=%d\n",
                         label, f, ms, phase (pt::FramePhase::SceneLower), phase (pt::FramePhase::CpuBvhBuild),
                         phase (pt::FramePhase::Upload), phase (pt::FramePhase::AccelBuild), phases.accelBuild != nullptr ? phases.accelBuild : "-",
                         image.sceneRefitted ? 1 : 0, image.sceneRebuilt ? 1 : 0);
            if (f == 0)
                continue;  // the first frame builds in both runs and waits for the kernel
            total += ms; lower += phase (pt::FramePhase::SceneLower); bvh += phase (pt::FramePhase::CpuBvhBuild);
            upload += phase (pt::FramePhase::Upload); accel += phase (pt::FramePhase::AccelBuild);
            accelKind = phases.accelBuild != nullptr ? phases.accelBuild : "";
            ++timed;
        }
        std::printf ("[ptscore] %s mean of %d advancing frames: %.2f ms a frame, lower %.2f, cpu bvh %.2f, upload %.2f, accel %.2f; scene builds %llu refits %llu\n",
                     label, timed, total / timed, lower / timed, bvh / timed, upload / timed, accel / timed,
                     static_cast<unsigned long long> (engine.sceneBuilds()), static_cast<unsigned long long> (engine.sceneRefits()));
        if (! forceRebuild)
            check (engine.sceneBuilds() == 1 && engine.sceneRefits() == 12 && accelKind == "update",
                   "every advancing frame on OptiX refits the BVH and updates the GAS");
    };
    run (false, "refit");
    run (true, "rebuild");
}
#endif

} // namespace

int scoreScenePathTraceTests()
{
    admissionTests();
    coverageTests();
    appearanceTests();
    refitTests();
    productOccupancyTests();
#if defined(ARBIT_HAVE_OPTIX_KERNEL)
    optixTiming();
#endif
    std::printf ("ptscore: %d checks, %d failures\n", checks, failures);
    return failures;
}
