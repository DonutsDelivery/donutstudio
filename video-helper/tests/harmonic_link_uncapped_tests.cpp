#include "harmonic_link_geometry.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

namespace geometry = videowire::geometry;

namespace
{
int failures = 0;

void check (bool condition, const char* what)
{
    if (condition) return;
    std::fprintf (stderr, "FAIL: %s\n", what);
    ++failures;
}

// Every note sounds at beat 0. Each link joins neighbouring notes.
std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> frameFor (int notes, int links)
{
    auto score = std::make_shared<arbitmod::Score>();
    score->notationVersion = 1;
    score->scoreRevision = 1;
    score->rootFreq = 440.0f;
    for (int index = 0; index < notes; ++index)
    {
        arbitmod::Note note;
        note.id = index + 1;
        note.trackId = 1;
        note.startBeat = -static_cast<float> (index % 7) * 0.25f;
        note.lengthBeats = 8.0f;
        note.velocity = 100.0f;
        note.freqHz = 220.0f + static_cast<float> (index) * 3.0f;
        note.durationSeconds = 4.0f;
        score->notes.push_back (note);
    }
    for (int index = 0; index < links; ++index)
    {
        arbitmod::Link link;
        link.id = index + 1;
        link.slaveNoteId = index % (notes - 1) + 1;
        link.masterNoteId = index % (notes - 1) + 2;
        link.slaveHarmonic = 3;
        link.masterHarmonic = 2;
        score->links.push_back (link);
    }
    canonicalblockc::FrameKey key;
    key.projectGeneration = key.sourceGeneration = key.helperGeneration = 1;
    key.backendGeneration = key.deviceGeneration = key.scoreGeneration = 1;
    key.beatMapGeneration = key.fpsGeneration = key.loopGeneration = 1;
    key.seekGeneration = 1;
    key.fps = 60.0;
    canonicalblockc::FrameProducer producer;
    return producer.evaluate (key, score, 0.0f);
}

visualharmonicgeometry::Mapping mapping()
{
    visualharmonicgeometry::Mapping value;
    value.stableId = 11;
    return value;
}

// The packed rows hold all 300 notes and 500 links. The scene mesh reads the score.
void scorePastTheTextures()
{
    const auto frame = frameFor (300, 500);
    check (frame != nullptr, "the long score packs");
    if (frame == nullptr) return;
    check (frame->noteRows() >= 300 && frame->linkRows() == 500, "the frame holds every note and link");
    std::string error;
    const auto value = videohelper::harmonicgeometry::lower (mapping(), frame, error, nullptr, false);
    check (value.has_value(), "the scene route lowers the long score");
    if (! value) return;
    const auto corners = videohelper::harmonicgeometry::linkTubeLayout (
        mapping().shape, visualharmonicgeometry::effectiveLinkLayout (
            mapping().shape, mapping().curvaturePercent, mapping().segments, 1)).vertices;
    check (std::get<geometry::GeometryData> (value->data).positions.size() == 500 * corners,
           "every link of the score reaches the mesh");
}

// Under the texture sizes, both routes draw the same links in the same order.
void routesAgree()
{
    const auto frame = frameFor (20, 19);
    std::string error;
    const auto packed = videohelper::harmonicgeometry::lower (mapping(), frame, error, nullptr, true);
    const auto scored = videohelper::harmonicgeometry::lower (mapping(), frame, error, nullptr, false);
    check (packed.has_value() && scored.has_value(), "both routes lower a small score");
    if (! packed || ! scored) return;
    const auto& a = std::get<geometry::GeometryData> (packed->data);
    const auto& b = std::get<geometry::GeometryData> (scored->data);
    // The row route draws flat strips and the score route round tubes, so compare where each link starts.
    const auto stripCorners = a.positions.size() / 19, tubeCorners = b.positions.size() / 19;
    const auto sides = videohelper::harmonicgeometry::kLinkTubeSides;
    bool same = ! a.positions.empty() && packed->attributes[0].elements.size() / (a.indices.size() / 3 / 19) == 19;
    for (std::size_t link = 0; same && link < 19; ++link)
    {
        geometry::Vec3 ring {};
        for (std::size_t corner = 0; corner < sides; ++corner)
        {
            const auto& point = b.positions[link * tubeCorners + corner];
            ring.x += point.x / static_cast<float> (sides);
            ring.y += point.y / static_cast<float> (sides);
            ring.z += point.z / static_cast<float> (sides);
        }
        const auto& left = a.positions[link * stripCorners];
        const auto& right = a.positions[link * stripCorners + 1];
        same = std::abs (ring.x - (left.x + right.x) * 0.5f) < 1.0e-4f
            && std::abs (ring.y - (left.y + right.y) * 0.5f) < 1.0e-4f
            && std::abs (ring.z - (left.z + right.z) * 0.5f) < 1.0e-4f;
    }
    check (same, "the score route draws what the row route draws");
}

// Coincident notes have no segment, so a score can hold more links than the mesh draws.
void linkCount()
{
    const auto frame = frameFor (20, 19);
    std::string error;
    const auto value = videohelper::harmonicgeometry::lower (mapping(), frame, error, nullptr, false);
    check (value.has_value() && videohelper::harmonicgeometry::linkCount (*value) == 19,
           "the mesh reports each drawn link once");
}

// A flat strip can't show a bent highlight, so a retained scene's links have a round section
// whatever the shape, with faces that point outward.
void roundLinks()
{
    using visualharmonicgeometry::LinkShape;
    const auto frame = frameFor (20, 19);
    check (frame != nullptr, "the score packs");
    if (frame == nullptr) return;
    const auto sides = videohelper::harmonicgeometry::kLinkTubeSides;
    for (const auto shape : { LinkShape::ribbon, LinkShape::beam, LinkShape::curve, LinkShape::spring, LinkShape::strut })
    {
        auto authored = mapping();
        authored.shape = shape;
        std::string error;
        const auto value = videohelper::harmonicgeometry::lower (authored, frame, error, nullptr, false);
        check (value.has_value(), "the scene route lowers every shape");
        if (! value) continue;
        const auto& mesh = std::get<geometry::GeometryData> (value->data);
        const auto layout = videohelper::harmonicgeometry::linkTubeLayout (shape,
            visualharmonicgeometry::effectiveLinkLayout (shape, authored.curvaturePercent, authored.segments, 1));
        check (mesh.positions.size() == 19 * layout.vertices && mesh.indices.size() == 19 * layout.triangles * 3,
               "the mesh matches the round layout");
        check (mesh.vertexIds.size() == mesh.positions.size()
                   && value->attributes[0].elements.size() == mesh.indices.size() / 3,
               "identities and face attributes follow the round layout");
        geometry::Vec3 center {};
        for (std::size_t corner = 0; corner < sides; ++corner)
        {
            center.x += mesh.positions[corner].x / static_cast<float> (sides);
            center.y += mesh.positions[corner].y / static_cast<float> (sides);
            center.z += mesh.positions[corner].z / static_cast<float> (sides);
        }
        const auto radiusOf = [&] (const geometry::Vec3& point)
        { return std::hypot (point.x - center.x, point.y - center.y, point.z - center.z); };
        const auto radius = radiusOf (mesh.positions[0]);
        bool round = radius > 0.0f;
        for (std::size_t corner = 0; corner < sides; ++corner)
            round = round && std::abs (radiusOf (mesh.positions[corner]) - radius) < radius * 0.01f;
        const auto opposite = radiusOf ({ (mesh.positions[0].x + mesh.positions[sides / 2].x) * 0.5f,
                                          (mesh.positions[0].y + mesh.positions[sides / 2].y) * 0.5f,
                                          (mesh.positions[0].z + mesh.positions[sides / 2].z) * 0.5f });
        check (round && opposite < radius * 0.01f, "the section is a circle around the link");
        const auto& p = mesh.positions[mesh.indices[0]];
        const auto& q = mesh.positions[mesh.indices[1]];
        const auto& r = mesh.positions[mesh.indices[2]];
        const geometry::Vec3 normal { (q.y-p.y)*(r.z-p.z)-(q.z-p.z)*(r.y-p.y),
                                      (q.z-p.z)*(r.x-p.x)-(q.x-p.x)*(r.z-p.z),
                                      (q.x-p.x)*(r.y-p.y)-(q.y-p.y)*(r.x-p.x) };
        check (normal.x*(p.x-center.x) + normal.y*(p.y-center.y) + normal.z*(p.z-center.z) > 0.0f,
               "the tube's faces point outward");
    }
}

float span (const std::vector<geometry::Vec3>& positions, float geometry::Vec3::* axis)
{
    auto low = positions.front().*axis, high = low;
    for (const auto& position : positions)
    {
        low = std::min (low, position.*axis);
        high = std::max (high, position.*axis);
    }
    return high - low;
}

// A fixed scale range would leave tiny links as specks and huge ones past the frame.
void fitHasNoRange()
{
    const auto visibleHeight = 2.0f * 10.0f * std::tan (0.7853981634f * 0.5f);
    const auto visibleWidth = visibleHeight * 1920.0f / 1080.0f;
    const auto near = [] (float value, float expected) { return std::abs (value - expected) < expected * 0.001f; };
    std::vector<geometry::Vec3> tiny { { 0, 0, 0 }, { 0.01f, 0.01f, 0 } };
    geometry::fitHarmonicLinksToView (tiny, 1920, 1080);
    check (near (span (tiny, &geometry::Vec3::y), visibleHeight * 0.65f), "tiny links fill 65% of the view's height");

    std::vector<geometry::Vec3> tall { { 0, 0, 0 }, { 1, 4, 0 } };
    geometry::fitHarmonicLinksToView (tall, 1920, 1080);
    check (near (span (tall, &geometry::Vec3::y), visibleHeight * 0.65f), "tall links fill 65% of the view's height");

    std::vector<geometry::Vec3> wide { { 0, 0, 0 }, { 40, 1, 0 } };
    geometry::fitHarmonicLinksToView (wide, 1920, 1080);
    check (near (span (wide, &geometry::Vec3::x), visibleWidth * 0.65f), "wide links fill 65% of the view's width");

    std::vector<geometry::Vec3> huge { { 0, 0, 0 }, { 0, 2000, 0 } };
    geometry::fitHarmonicLinksToView (huge, 1920, 1080);
    check (span (huge, &geometry::Vec3::y) < visibleHeight, "huge links stay in the view");

    std::vector<geometry::Vec3> deep { { 0, 0, 0 }, { 0.01f, 0.01f, 40 } };
    geometry::fitHarmonicLinksToView (deep, 1920, 1080);
    check (span (deep, &geometry::Vec3::z) < 10.0f, "deep links stay in front of the camera");
}
}

int main()
{
    scorePastTheTextures();
    routesAgree();
    linkCount();
    roundLinks();
    fitHasNoRange();
    if (failures == 0) std::puts ("harmonic link uncapped tests passed");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
