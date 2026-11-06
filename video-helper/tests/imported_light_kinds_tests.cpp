// Imported-scene Render 3D Light with Spot and Area lights: the request codec
// carries cones, size, shape and visibility behind a flagged kind, a light
// without them encodes exactly as before, and the validator refuses bad cones.

#include "VisualImportedSceneRenderOperationContract.h"

#include <iostream>
#include <sstream>
#include <string>

namespace
{

namespace grid = HarmonicMIDI::grid;
namespace codec = visualimportedscenerender::detail;

int checks = 0;
int failures = 0;

void check (bool condition, const std::string& message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

grid::SceneLightRecord light (grid::SceneLightKind kind)
{
    grid::SceneLightRecord record;
    record.id = grid::SceneLightId { 7 };
    record.kind = kind;
    record.intensity = 3.0f;
    return record;
}

std::string encoded (const grid::SceneLightRecord& record)
{
    std::ostringstream stream;
    codec::encodeLight (stream, record);
    return stream.str();
}

bool roundTrip (const grid::SceneLightRecord& record, grid::SceneLightRecord& decoded)
{
    std::istringstream stream (encoded (record));
    return codec::decodeLight (stream, decoded);
}

} // namespace

int main()
{
    // Directional and Point lights with default shape fields keep the old wire.
    for (const auto kind : { grid::SceneLightKind::Directional, grid::SceneLightKind::Point })
    {
        const auto record = light (kind);
        const auto text = encoded (record);
        check (text.rfind ("7 " + std::to_string (static_cast<unsigned> (kind)) + ' ', 0) == 0,
               "a plain light keeps its unflagged kind");
        grid::SceneLightRecord decoded;
        check (roundTrip (record, decoded) && decoded.kind == kind, "a plain light round-trips");
    }

    auto spot = light (grid::SceneLightKind::Spot);
    spot.innerConeAngle = 0.2f;
    spot.outerConeAngle = 0.6f;
    spot.radius = 0.25f;
    spot.visibility = grid::sceneRayVisibility::All & ~grid::sceneRayVisibility::Shadow;
    check (encoded (spot).rfind ("7 " + std::to_string (static_cast<unsigned> (grid::SceneLightKind::Spot) | 0x80u) + ' ', 0) == 0,
           "a Spot light flags its kind");
    grid::SceneLightRecord decodedSpot;
    check (roundTrip (spot, decodedSpot) && decodedSpot.kind == grid::SceneLightKind::Spot
               && decodedSpot.innerConeAngle == 0.2f && decodedSpot.outerConeAngle == 0.6f
               && decodedSpot.radius == 0.25f && decodedSpot.visibility == spot.visibility,
           "a Spot light keeps its cones, radius and visibility");

    auto area = light (grid::SceneLightKind::Area);
    area.areaShape = grid::SceneAreaLightShape::Ellipse;
    area.areaSizeX = 2.0f;
    area.areaSizeY = 0.5f;
    area.spread = 1.0f;
    grid::SceneLightRecord decodedArea;
    check (roundTrip (area, decodedArea) && decodedArea.kind == grid::SceneLightKind::Area
               && decodedArea.areaShape == grid::SceneAreaLightShape::Ellipse
               && decodedArea.areaSizeX == 2.0f && decodedArea.areaSizeY == 0.5f && decodedArea.spread == 1.0f,
           "an Area light keeps its shape, extents and spread");

    // A point light hidden from shadow rays needs the tail too.
    auto hidden = light (grid::SceneLightKind::Point);
    hidden.visibility = grid::sceneRayVisibility::Camera;
    grid::SceneLightRecord decodedHidden;
    check (roundTrip (hidden, decodedHidden) && decodedHidden.visibility == grid::sceneRayVisibility::Camera,
           "a Point light keeps a cleared visibility bit");

    // A forged flag on a plain light is refused.
    {
        auto text = encoded (light (grid::SceneLightKind::Point));
        text.replace (0, 3, "7 " + std::to_string (static_cast<unsigned> (grid::SceneLightKind::Point) | 0x80u));
        text.pop_back();
        for (int i = 0; i < 7; ++i) text += " 0";
        text += " 0 63\n";
        // The extra floats are zero sizes, which the shape checks refuse as well.
        std::istringstream stream (text);
        grid::SceneLightRecord decoded;
        check (! codec::decodeLight (stream, decoded), "a flagged plain light is refused");
    }

    auto badCone = spot;
    badCone.innerConeAngle = 0.7f;
    check (! codec::validLight (badCone), "a Spot light with inner past outer is refused");
    auto flatArea = area;
    flatArea.areaSizeY = 0.0f;
    check (! codec::validLight (flatArea), "an Area light with a zero extent is refused");
    check (codec::validLight (spot) && codec::validLight (area), "valid Spot and Area lights are accepted");

    std::cout << checks - failures << '/' << checks << " imported light kinds checks passed\n";
    return failures == 0 ? 0 : 1;
}
