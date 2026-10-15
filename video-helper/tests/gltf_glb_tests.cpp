#include "../src/gltf_glb.h"
#include "../src/glb_geometry_core_adapter.h"
#include "../src/imported_animation_deformation_consumer.h"
#include "../src/sha256.h"
#include "../../shared/VisualImportedAnimationOperationContract.h"
#include "../../shared/VisualStarterModelAssets.h"
#include "../../shared/ReactiveCharacterAsset.h"
#include "../../shared/HolographicTradingCardAsset.h"
#include "../../plugin/Tests/fixtures/visual-model/StaticVisualModelWorkflowFixture.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace fs = std::filesystem;
using videohelper::gltf::GlbAdmissionOptions;
using videohelper::gltf::GlbMetadata;

int failures = 0;

void check(bool value, const char* message)
{
    if (!value)
    {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
}

void appendU32(std::vector<std::uint8_t>& bytes, std::uint32_t value)
{
    bytes.push_back(static_cast<std::uint8_t>(value));
    bytes.push_back(static_cast<std::uint8_t>(value >> 8u));
    bytes.push_back(static_cast<std::uint8_t>(value >> 16u));
    bytes.push_back(static_cast<std::uint8_t>(value >> 24u));
}

void appendU16(std::vector<std::uint8_t>& bytes, std::uint16_t value)
{
    bytes.push_back(static_cast<std::uint8_t>(value));
    bytes.push_back(static_cast<std::uint8_t>(value >> 8u));
}

void appendFloat(std::vector<std::uint8_t>& bytes, float value)
{
    std::uint32_t encoded = 0;
    std::memcpy(&encoded, &value, sizeof(encoded));
    appendU32(bytes, encoded);
}

std::vector<std::uint8_t> makeGlb(nlohmann::json root,
                                  std::vector<std::uint8_t> bin = {})
{
    auto text = root.dump();
    while ((text.size() & 3u) != 0)
        text.push_back(' ');
    while ((bin.size() & 3u) != 0)
        bin.push_back(0);

    std::vector<std::uint8_t> bytes;
    const auto total = 12u + 8u + static_cast<unsigned>(text.size())
                     + (bin.empty() ? 0u : 8u + static_cast<unsigned>(bin.size()));
    appendU32(bytes, 0x46546c67u);
    appendU32(bytes, 2);
    appendU32(bytes, total);
    appendU32(bytes, static_cast<std::uint32_t>(text.size()));
    appendU32(bytes, 0x4e4f534au);
    bytes.insert(bytes.end(), text.begin(), text.end());
    if (!bin.empty())
    {
        appendU32(bytes, static_cast<std::uint32_t>(bin.size()));
        appendU32(bytes, 0x004e4942u);
        bytes.insert(bytes.end(), bin.begin(), bin.end());
    }
    return bytes;
}

nlohmann::json validRoot()
{
    return {
        {"asset", {{"version", "2.0"}, {"generator", "focused fixture"}}},
        {"scene", 0},
        {"scenes", nlohmann::json::array({{{"nodes", nlohmann::json::array({0})}}})},
        {"nodes", nlohmann::json::array({
            {{"children", nlohmann::json::array({1})}, {"mesh", 0}},
            nlohmann::json::object()
        })},
        {"meshes", nlohmann::json::array({
            {{"primitives", nlohmann::json::array({nlohmann::json::object()})}}
        })},
        {"materials", nlohmann::json::array({nlohmann::json::object()})},
        {"textures", nlohmann::json::array({nlohmann::json::object()})},
        {"images", nlohmann::json::array({nlohmann::json::object()})},
        {"samplers", nlohmann::json::array({nlohmann::json::object()})},
        {"cameras", nlohmann::json::array({nlohmann::json::object()})},
        {"accessors", nlohmann::json::array({nlohmann::json::object()})},
        {"bufferViews", nlohmann::json::array({nlohmann::json::object()})}
    };
}

const std::vector<std::uint8_t>& pngImageBytes()
{
    static const std::vector<std::uint8_t> bytes {
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
        0, 0, 0, 2, 0, 0, 0, 1, 8, 6, 0, 0, 0, 244, 34, 127, 138,
        0, 0, 0, 17, 73, 68, 65, 84, 120, 156, 99, 248, 207, 192, 240,
        159, 225, 63, 67, 3, 0, 16, 121, 3, 126, 33, 192, 253, 141, 0,
        0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130
    };
    return bytes;
}

std::vector<std::uint8_t> decodeBase64(std::string_view text)
{
    constexpr std::string_view alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<std::uint8_t> bytes;
    std::uint32_t accumulator = 0;
    int bits = 0;
    for (const auto character : text)
    {
        if (character == '=')
            break;
        const auto value = alphabet.find(character);
        if (value == std::string_view::npos)
            continue;
        accumulator = (accumulator << 6u) | static_cast<std::uint32_t>(value);
        bits += 6;
        if (bits >= 8)
        {
            bits -= 8;
            bytes.push_back(static_cast<std::uint8_t>(accumulator >> bits));
        }
    }
    return bytes;
}

const std::vector<std::uint8_t>& rgb1024ImageBytes()
{
    // Authored four-quadrant 1024x1024 RGB8 PNG, compressed with zlib level 9.
    // Colors: (10,20,30), (40,50,60), (70,80,90), (100,110,120).
    static const auto bytes = decodeBase64(
        "iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAIAAADwf7zUAAAXL0lEQVR42u3XoQ0AIRAAQTQajUaj0Wg0Gv36y6cMxE0yPWw2"
        "5VIBCKj1AUBASQIBDAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAA"
        "YAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAADIAKAhgA"
        "AAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAABgAAAwAAAYAAAMA"
        "gAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAABgAAAwAAAYAAAMAgAEAwAAAYAAA"
        "MAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAA"
        "BgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDA"
        "AABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAAJBAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAM"
        "AAAGAAADAIABAMAAAGAAADAAABgAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIAB"
        "AMAAAGAAADAAABgAAAOgggAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDA"
        "AABgAAAwAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAG"
        "AAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAADAIABAMAA"
        "AGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAADAIABAMAAAGAAADAAABgA"
        "AAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAIABAMAAAGAAADAAABgAAAwAAAYAAAMA"
        "gAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAA"
        "MAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAIABUEEAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAA"
        "ABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYA"
        "AAMAgAEAwAAAYAAAMAAAGAAAA6CCAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAAD"
        "AIABAMAAAGAAADAAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAA"
        "ADAAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAMA"
        "gAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAMAgAEAwAAAYAAA"
        "MAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAMggQAGAAADAIABAMAAAGAAADAA"
        "ABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYA"
        "AAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAGQAUBDAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAAD"
        "AIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAA"
        "ADAAABgAAAwAAAYAAAMAgAEAwAAAYAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwA"
        "AAYAAAMAgAEAwAAAYAAAMAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEA"
        "wAAAYAAAMAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAA"
        "SCCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADACAAQDAAABg"
        "AAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADACAAVBBAAMAgAEAwAAAYAAA"
        "MAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAMAgAEAwAAAYAAAMAAAGAAADAAA"
        "BgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDA"
        "AABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAY"
        "AAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAAD"
        "AIABAMAAAGAAADAAABgAAAwAAAYAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAA"
        "ADAAABgAAAwAAAYAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwA"
        "AAYAwACoIIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAIAB"
        "AMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAIABUEEAAwCAAQDA"
        "AABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAAAwCAAQDAAABgAAAwAAAY"
        "AAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAAD"
        "AIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADACAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAA"
        "ADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADACAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAADwYgLk2"
        "AAGd7wcgIAMAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMA"
        "YAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAYABUEMAAAGAA"
        "ADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAMAAAGAAADAAABgAAAwA"
        "AAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAMAAqCCAAQDAAABgAAAwAAAYAAAMAAAG"
        "AAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADACAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAA"
        "AGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADACAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgA"
        "AAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMA"
        "gAEAwAAAYAAAMAAAGAAADAAABgDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAA"
        "MAAAGAAADAAABgDAAEgggAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAA"
        "ABgAAAwAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAgAFQ"
        "QQADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAADAIABAMAA"
        "AGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAADAIABAMAAAGAAADAAABgA"
        "AAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAIABAMAAAGAAADAAABgAAAwAAAYAAAMA"
        "gAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAA"
        "MAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAASCGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAA"
        "AwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABg"
        "AAAwAAAYAAAMAAAGAAADAGAAVBDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAA"
        "MAAAGAAADAAABgDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAA"
        "BgDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwBgAAAw"
        "AAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwBgAAAwAAAYAAAMAAAG"
        "AAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQAwAAAYAAAMAAAGAAADAIABAMAA"
        "AGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgA"
        "AAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQAwACoIYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAM"
        "AAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIAB"
        "AMAAAGAAADAAABgAAAwAAAYAAAMAYABUEMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDA"
        "AABgAAAwAAAYAAAMAAAGAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAY"
        "AAAMAAAGAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAAD"
        "AGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAGAAADAAABgA"
        "AAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAGAAJBDAAABgAAAwAAAYAAAM"
        "AAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgDAAABgAAAwAAAYAAAMAAAGAAADAIAB"
        "AMAAAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgDAAKgggAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDA"
        "AABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAY"
        "AAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAAD"
        "AIABAMAAAGAAADAAABgAAAwAAAYAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAA"
        "ADAAABgAAAwAAAYAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwA"
        "AAYAAAkEMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAMAAA"
        "GAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABAMAAAGAAADAAABgAAAwAAAYAAAMAgAEAMAAqCGAAADAAABgA"
        "AAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAGAAADAAABgAAAwAAAYAAAMA"
        "gAEAwAAAYAAAMAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAGAAADAAABgAAAwAAAYAAAMAgAEAwAAAYAAA"
        "MAAAGAAADAAABgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAA"
        "BgAAAwCAAQDAAABgAAAwAAAYAAAMAAAGAAADAIABADAAABgAAAwAAAYAAAMAgAEAwAAAYAAAMAAAGAAADAAABgCABy55sLbR"
        "ZhNbUwAAAABJRU5ErkJggg==");
    return bytes;
}

const std::vector<std::uint8_t>& jpegImageBytes()
{
    static const auto bytes = decodeBase64(
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
        "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
        "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDAREA"
        "AhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAA"
        "F9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk"
        "6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6i"
        "pqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwE"
        "BAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJB"
        "UQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVV"
        "ldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6w"
        "sPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDx+v8AVg/zX"
        "P/Z");
    return bytes;
}

std::vector<std::uint8_t> staticMeshBinWithImage(const std::vector<std::uint8_t>& imageBytes,
                                                  bool invalidIndex = false)
{
    std::vector<std::uint8_t> bin;
    for (const auto value : std::vector<float>{
             0.0f, 0.0f, 0.0f,
             1.0f, 0.0f, 0.0f,
             0.0f, 1.0f, 0.0f})
        appendFloat(bin, value);
    for (int vertex = 0; vertex < 3; ++vertex)
        for (const auto value : std::vector<float>{0.0f, 0.0f, 1.0f})
            appendFloat(bin, value);
    for (int vertex = 0; vertex < 3; ++vertex)
        for (const auto value : std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f})
            appendFloat(bin, value);
    for (const auto value : std::vector<float>{0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f})
        appendFloat(bin, value);
    bin.insert(bin.end(), {
        255, 0, 0, 255,
        0, 255, 0, 128,
        0, 0, 255, 64
    });
    appendU16(bin, 0);
    appendU16(bin, 1);
    appendU16(bin, invalidIndex ? 3 : 2);
    bin.insert(bin.end(), {0, 0});
    bin.insert(bin.end(), imageBytes.begin(), imageBytes.end());
    return bin;
}

std::vector<std::uint8_t> staticMeshBin(bool invalidIndex = false)
{
    return staticMeshBinWithImage(pngImageBytes(), invalidIndex);
}

nlohmann::json staticMeshRoot()
{
    return {
        {"asset", {{"version", "2.0"}, {"generator", "static decode fixture"}}},
        {"scene", 0},
        {"scenes", nlohmann::json::array({
            {{"name", "Scene A"}, {"nodes", nlohmann::json::array({0})}},
            {{"name", "Scene B"}, {"nodes", nlohmann::json::array({0})}}
        })},
        {"nodes", nlohmann::json::array({
            {
                {"name", "Root"},
                {"mesh", 0},
                {"camera", 0},
                {"children", nlohmann::json::array({1})},
                {"translation", nlohmann::json::array({1.0, 2.0, 3.0})},
                {"extensions", {{"KHR_lights_punctual", {{"light", 0}}}}}
            },
            {
                {"name", "Child"},
                {"translation", nlohmann::json::array({2.0, 0.0, 0.0})}
            }
        })},
        {"meshes", nlohmann::json::array({
            {
                {"name", "Triangle"},
                {"primitives", nlohmann::json::array({
                    {
                        {"attributes", {
                            {"POSITION", 0}, {"NORMAL", 1}, {"TANGENT", 2},
                            {"TEXCOORD_0", 3}, {"COLOR_0", 4}
                        }},
                        {"indices", 5},
                        {"material", 0}
                    }
                })}
            }
        })},
        {"materials", nlohmann::json::array({
            {
                {"name", "Paint"},
                {"pbrMetallicRoughness", {
                    {"baseColorFactor", nlohmann::json::array({0.5, 0.6, 0.7, 0.8})},
                    {"metallicFactor", 0.25},
                    {"roughnessFactor", 0.75},
                    {"baseColorTexture", {{"index", 0}}},
                    {"metallicRoughnessTexture", {{"index", 0}}}
                }},
                {"normalTexture", {{"index", 0}, {"scale", 0.5}}},
                {"occlusionTexture", {{"index", 0}, {"strength", 0.25}}},
                {"emissiveTexture", {{"index", 0}}},
                {"emissiveFactor", nlohmann::json::array({0.1, 0.2, 0.3})},
                {"alphaMode", "MASK"},
                {"alphaCutoff", 0.4},
                {"doubleSided", true}
            }
        })},
        {"textures", nlohmann::json::array({
            {{"name", "Texture"}, {"source", 0}, {"sampler", 0}}
        })},
        {"images", nlohmann::json::array({
            {{"name", "Embedded"}, {"bufferView", 6}, {"mimeType", "image/png"}}
        })},
        {"samplers", nlohmann::json::array({
            {{"name", "Nearest"}, {"magFilter", 9728}, {"minFilter", 9984},
             {"wrapS", 33071}, {"wrapT", 33648}}
        })},
        {"cameras", nlohmann::json::array({
            {{"name", "Camera"}, {"type", "perspective"},
             {"perspective", {{"aspectRatio", 1.5}, {"yfov", 1.0},
                              {"znear", 0.1}, {"zfar", 100.0}}}},
            {{"name", "Ortho"}, {"type", "orthographic"},
             {"orthographic", {{"xmag", 2.0}, {"ymag", 1.0},
                               {"znear", 0.0}, {"zfar", 10.0}}}}
        })},
        {"extensionsUsed", nlohmann::json::array({"KHR_lights_punctual"})},
        {"extensions", {
            {"KHR_lights_punctual", {
                {"lights", nlohmann::json::array({
                    {{"name", "Key"}, {"type", "point"},
                     {"color", nlohmann::json::array({1.0, 0.5, 0.25})},
                     {"intensity", 2.0}, {"range", 10.0}},
                    {{"name", "Rim"}, {"type", "spot"},
                     {"spot", {{"innerConeAngle", 0.2}, {"outerConeAngle", 0.6}}}}
                })}
            }}
        }},
        {"buffers", nlohmann::json::array({{{"byteLength", 238}}})},
        {"bufferViews", nlohmann::json::array({
            {{"buffer", 0}, {"byteOffset", 0}, {"byteLength", 36}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 36}, {"byteLength", 36}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 72}, {"byteLength", 48}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 120}, {"byteLength", 24}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 144}, {"byteLength", 12}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 156}, {"byteLength", 6}, {"target", 34963}},
            {{"buffer", 0}, {"byteOffset", 164}, {"byteLength", 74}}
        })},
        {"accessors", nlohmann::json::array({
            {{"bufferView", 0}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"},
             {"min", nlohmann::json::array({0.0, 0.0, 0.0})},
             {"max", nlohmann::json::array({1.0, 1.0, 0.0})}},
            {{"bufferView", 1}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"}},
            {{"bufferView", 2}, {"componentType", 5126}, {"count", 3}, {"type", "VEC4"}},
            {{"bufferView", 3}, {"componentType", 5126}, {"count", 3}, {"type", "VEC2"}},
            {{"bufferView", 4}, {"componentType", 5121}, {"normalized", true},
             {"count", 3}, {"type", "VEC4"}},
            {{"bufferView", 5}, {"componentType", 5123}, {"count", 3}, {"type", "SCALAR"}}
        })}
    };
}

nlohmann::json staticMeshRootWithImage(std::string_view mimeType,
                                       std::size_t imageByteLength)
{
    auto root = staticMeshRoot();
    root["images"][0]["mimeType"] = mimeType;
    root["buffers"][0]["byteLength"] = 164u + imageByteLength;
    root["bufferViews"][6]["byteLength"] = imageByteLength;
    return root;
}

std::vector<std::uint8_t> animatedDeformationBin(bool descendingTimes = false)
{
    std::vector<std::uint8_t> bin;
    for (const auto value : {0.0f, 0.0f, 0.0f,
                             1.0f, 0.0f, 0.0f,
                             0.0f, 1.0f, 0.0f})
        appendFloat(bin, value);
    for (int vertex = 0; vertex < 3; ++vertex)
        bin.insert(bin.end(), {0, 0, 0, 0});
    for (int vertex = 0; vertex < 3; ++vertex)
        for (const auto value : {1.0f, 0.0f, 0.0f, 0.0f}) appendFloat(bin, value);
    for (int target = 0; target < 2; ++target)
        for (int value = 0; value < 9; ++value)
            appendFloat(bin, target == 0 && value == 0 ? 0.25f : 0.0f);
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            appendFloat(bin, row == column ? 1.0f : 0.0f);
    appendFloat(bin, descendingTimes ? 1.0f : 0.0f);
    appendFloat(bin, descendingTimes ? 0.0f : 1.0f);
    for (const auto value : {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f}) appendFloat(bin, value);
    for (const auto value : {0.0f, 1.0f, 0.5f, 0.5f}) appendFloat(bin, value);
    return bin;
}

nlohmann::json animatedDeformationRoot()
{
    return {
        {"asset", {{"version", "2.0"}}},
        {"nodes", nlohmann::json::array({
            {{"mesh", 0}, {"skin", 0}, {"children", nlohmann::json::array({1})}},
            nlohmann::json::object()
        })},
        {"meshes", nlohmann::json::array({{{"weights", nlohmann::json::array({0.0, 0.0})},
            {"primitives", nlohmann::json::array({{
                {"attributes", {{"POSITION", 0}, {"JOINTS_0", 1}, {"WEIGHTS_0", 2}}},
                {"targets", nlohmann::json::array({{{"POSITION", 3}}, {{"POSITION", 4}}})}
            }})}}})},
        {"skins", nlohmann::json::array({{{"joints", nlohmann::json::array({1})},
                                            {"inverseBindMatrices", 5}}})},
        {"animations", nlohmann::json::array({{
            {"name", "Walk"},
            {"samplers", nlohmann::json::array({
                {{"input", 6}, {"output", 7}, {"interpolation", "LINEAR"}},
                {{"input", 6}, {"output", 8}, {"interpolation", "STEP"}}
            })},
            {"channels", nlohmann::json::array({
                {{"sampler", 0}, {"target", {{"node", 1}, {"path", "translation"}}}},
                {{"sampler", 1}, {"target", {{"node", 0}, {"path", "weights"}}}}
            })}
        }})},
        {"buffers", nlohmann::json::array({{{"byteLength", 280}}})},
        {"bufferViews", nlohmann::json::array({
            {{"buffer", 0}, {"byteOffset", 0}, {"byteLength", 36}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 36}, {"byteLength", 12}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 48}, {"byteLength", 48}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 96}, {"byteLength", 36}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 132}, {"byteLength", 36}, {"target", 34962}},
            {{"buffer", 0}, {"byteOffset", 168}, {"byteLength", 64}},
            {{"buffer", 0}, {"byteOffset", 232}, {"byteLength", 8}},
            {{"buffer", 0}, {"byteOffset", 240}, {"byteLength", 24}},
            {{"buffer", 0}, {"byteOffset", 264}, {"byteLength", 16}}
        })},
        {"accessors", nlohmann::json::array({
            {{"bufferView", 0}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"}},
            {{"bufferView", 1}, {"componentType", 5121}, {"count", 3}, {"type", "VEC4"}},
            {{"bufferView", 2}, {"componentType", 5126}, {"count", 3}, {"type", "VEC4"}},
            {{"bufferView", 3}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"}},
            {{"bufferView", 4}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"}},
            {{"bufferView", 5}, {"componentType", 5126}, {"count", 1}, {"type", "MAT4"}},
            {{"bufferView", 6}, {"componentType", 5126}, {"count", 2}, {"type", "SCALAR"}},
            {{"bufferView", 7}, {"componentType", 5126}, {"count", 2}, {"type", "VEC3"}},
            {{"bufferView", 8}, {"componentType", 5126}, {"count", 4}, {"type", "SCALAR"}}
        })}
    };
}

bool animationRejected(const nlohmann::json& root,
                       const std::vector<std::uint8_t>& bin,
                       std::string* diagnostic = nullptr,
                       videohelper::gltf::GlbAnimationDecodeOptions options = {})
{
    std::string error;
    const auto decoded = videohelper::gltf::decodeGlbAnimations(makeGlb(root, bin), options, error);
    if (diagnostic != nullptr) *diagnostic = error;
    return !decoded && !error.empty();
}

std::size_t animationFactoryCalls = 0;
std::size_t deformationFactoryCalls = 0;

std::shared_ptr<const visualanimation::Clip> countingAnimationFactory(
    const visualanimation::ClipView& source,
    const visualanimation::Limits& limits,
    std::string& error)
{
    ++animationFactoryCalls;
    return visualanimation::Clip::create(source, limits, error);
}

std::shared_ptr<const visualdeformation::DeformationAsset> countingDeformationFactory(
    const visualdeformation::AssetView& source,
    const visualdeformation::Limits& limits,
    std::string& error)
{
    ++deformationFactoryCalls;
    return visualdeformation::DeformationAsset::create(source, limits, error);
}

bool animationRejectedBeforeFactory(const nlohmann::json& root,
                                    const std::vector<std::uint8_t>& bin)
{
    animationFactoryCalls = 0;
    deformationFactoryCalls = 0;
    std::string error;
    const auto bytes = makeGlb(root, bin);
    const auto decoded = videohelper::gltf::detail::decodeGlbAnimationsWithFactories(
        bytes.data(), bytes.size(), {}, countingAnimationFactory,
        countingDeformationFactory, error);
    return !decoded && !error.empty() && animationFactoryCalls == 0
        && deformationFactoryCalls == 0;
}

bool decodeRejected(const nlohmann::json& root,
                    const std::vector<std::uint8_t>& bin,
                    std::string* diagnostic = nullptr,
                    GlbAdmissionOptions options = {})
{
    std::string error;
    try
    {
        const auto result = videohelper::gltf::decodeStaticGlb(makeGlb(root, bin), options, error);
        if (diagnostic != nullptr)
            *diagnostic = error;
        return !result.has_value() && !error.empty();
    }
    catch (const std::exception& exception)
    {
        if (diagnostic != nullptr)
            *diagnostic = exception.what();
        return false;
    }
}

bool near(float left, float right)
{
    return std::abs(left - right) < 0.0001f;
}

bool rejected(const std::vector<std::uint8_t>& bytes,
              std::string* diagnostic = nullptr,
              GlbAdmissionOptions options = {})
{
    std::string error;
    const auto result = videohelper::gltf::admitGlbMetadata(bytes, options, error);
    if (diagnostic != nullptr)
        *diagnostic = error;
    return !result.has_value() && !error.empty();
}
} // namespace

int main()
{
    {
        std::string characterError;
        GlbAdmissionOptions admission;
        admission.admitAnimations=true;
        admission.admitSkins=true;
        const auto scene=videohelper::gltf::decodeAnimatedGlbBaseScene(
            reactivecharacter::kGlb.data(),reactivecharacter::kGlb.size(),admission,characterError);
        check(scene && scene->decodedVertexCount==144 && scene->decodedIndexCount==216,
              "reactive character has a native-decodable six-part body mesh");
        videohelper::gltf::GlbAnimationDecodeOptions animationOptions;
        animationOptions.admission=admission;
        const auto animation=videohelper::gltf::decodeGlbAnimations(
            reactivecharacter::kGlb.data(),reactivecharacter::kGlb.size(),animationOptions,characterError);
        if (!animation) std::fprintf(stderr,"reactive character: %s\n",characterError.c_str());
        check(animation && animation->clips.size()==1 && animation->clips[0].name=="Body Sway"
              && animation->clips[0].jointBindings.size()==1
              && animation->clips[0].jointBindings[0].joint.value==2
              && animation->deformation && animation->deformation->skins().size()==1
              && animation->deformation->meshes().size()==1
              && animation->deformation->meshes()[0].morphTargets().size()==1,
              "reactive character preserves exact body-bone animation and named morph deformation");
    }
    GlbAdmissionOptions options;
    auto valid = makeGlb(validRoot(), {1, 2, 3, 4});
    std::string error;
    const auto metadata = videohelper::gltf::admitGlbMetadata(valid, options, error);
    check(metadata.has_value(), "valid static GLB metadata is admitted from memory");
    check(metadata && metadata->containerBytes == valid.size()
          && metadata->binBytes == 4 && metadata->scenes == 1
          && metadata->nodes == 2 && metadata->meshes == 1
          && metadata->primitives == 1 && metadata->materials == 1
          && metadata->maxHierarchyDepth == 2
          && metadata->defaultScene == std::optional<std::size_t>(0),
          "admission returns bounded container and scene metadata counts");
    check(metadata && metadata->generator == "focused fixture",
          "asset generator metadata is retained");

    const auto path = fs::temp_directory_path() / "donutstudio-gltf-glb-focused.glb";
    {
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(valid.data()),
                     static_cast<std::streamsize>(valid.size()));
    }
    error.clear();
    const auto fromPath = videohelper::gltf::admitGlbMetadataFile(path.string(), options, error);
    check(fromPath && fromPath->nodes == 2 && fromPath->binBytes == 4,
          "the bounded path entry point returns the same metadata");
    fs::remove(path);

    {
        auto bytes = valid;
        bytes[0] = 'x';
        check(rejected(bytes), "wrong GLB magic is rejected");
    }
    {
        auto bytes = valid;
        bytes[4] = 1;
        check(rejected(bytes), "non-2 GLB container version is rejected");
    }
    {
        auto bytes = valid;
        bytes.push_back(0);
        check(rejected(bytes), "trailing bytes beyond the declared GLB length are rejected");
    }
    {
        auto bytes = valid;
        bytes[12] = 3;
        check(rejected(bytes), "unaligned chunk length is rejected");
    }
    {
        auto bytes = valid;
        bytes[16] = 'B'; bytes[17] = 'I'; bytes[18] = 'N'; bytes[19] = 0;
        check(rejected(bytes), "a non-JSON first chunk is rejected");
    }
    {
        auto bytes = valid;
        appendU32(bytes, 0); appendU32(bytes, 0x004e4942u);
        const auto size = static_cast<std::uint32_t>(bytes.size());
        bytes[8] = static_cast<std::uint8_t>(size);
        bytes[9] = static_cast<std::uint8_t>(size >> 8u);
        bytes[10] = static_cast<std::uint8_t>(size >> 16u);
        bytes[11] = static_cast<std::uint8_t>(size >> 24u);
        check(rejected(bytes), "a second BIN chunk is rejected");
    }
    {
        auto root = validRoot();
        root["asset"]["version"] = "2.1";
        check(rejected(makeGlb(root)), "asset.version other than exactly 2.0 is rejected");
    }
    {
        auto root = validRoot();
        root["nodes"][1]["children"] = nlohmann::json::array({0});
        check(rejected(makeGlb(root)), "cyclic node hierarchy is rejected");
    }
    {
        auto root = validRoot();
        root["scenes"][0]["nodes"] = nlohmann::json::array({9});
        check(rejected(makeGlb(root)), "out-of-range scene node indices are rejected");
    }
    {
        auto root = validRoot();
        root["animations"] = nlohmann::json::array({nlohmann::json::object()});
        std::string diagnostic;
        check(rejected(makeGlb(root), &diagnostic)
              && diagnostic.find("static GLB parser") != std::string::npos,
              "animation metadata fails closed with a static-admission diagnostic");
    }
    {
        auto root = validRoot();
        root["extensionsUsed"] = nlohmann::json::array({"KHR_draco_mesh_compression"});
        root["extensionsRequired"] = root["extensionsUsed"];
        std::string diagnostic;
        check(rejected(makeGlb(root), &diagnostic)
              && diagnostic.find("KHR_draco_mesh_compression") != std::string::npos,
              "unsupported required extensions are named in the rejection diagnostic");

        GlbAdmissionOptions allowed;
        allowed.supportedRequiredExtensions.push_back("KHR_draco_mesh_compression");
        error.clear();
        const auto admitted = videohelper::gltf::admitGlbMetadata(makeGlb(root), allowed, error);
        check(admitted && admitted->extensionsRequired
                           == std::vector<std::string>{"KHR_draco_mesh_compression"},
              "caller-declared required extension support is retained in metadata");
    }
    {
        GlbAdmissionOptions limited;
        limited.limits.maxContainerBytes = valid.size() - 1;
        check(rejected(valid, nullptr, limited), "container byte limit is enforced before parsing");
    }
    {
        GlbAdmissionOptions limited;
        limited.limits.maxNodes = 1;
        check(rejected(valid, nullptr, limited), "top-level metadata count limits are enforced");
    }
    {
        GlbAdmissionOptions limited;
        limited.limits.maxNodeDepth = 1;
        check(rejected(valid, nullptr, limited), "node hierarchy depth limit is enforced");
    }
    {
        GlbAdmissionOptions limited;
        limited.limits.maxJsonNestingDepth = 2;
        check(rejected(valid, nullptr, limited), "JSON nesting is bounded before DOM parsing");
    }

    const auto meshRoot = staticMeshRoot();
    const auto meshBin = staticMeshBin();
    {
        const std::vector<std::uint8_t> bytes(holographictradingcard::kGlb.begin(),
                                             holographictradingcard::kGlb.end());
        const auto geometry = videohelper::gltf::decodeGlbMeshToGeometryCore(
            bytes, 0, 0, 71, 70, 1, error);
        check(geometry.has_value(), "Score-Deformed Imported Mesh starter extracts the exact bundled static card");
        if (geometry)
        {
            const auto& mesh = std::get<videowire::geometry::GeometryData>(geometry->descriptor().data);
            check(!mesh.positions.empty() && mesh.positions.size() <= videowire::geometry::scorefield::kMaximumElements,
                  "bundled card mesh fits the score-field sample limit");
        }
    }
    {
        const auto geometry=videohelper::gltf::decodeGlbMeshToGeometryCore(
            makeGlb(meshRoot,meshBin),std::nullopt,0,71,70,1,error);
        check(geometry.has_value(),"real GLB bytes use the canonical static decoder for Geometry Core extraction");
        auto animated=meshRoot; animated["animations"]=nlohmann::json::array({nlohmann::json::object()});
        check(!videohelper::gltf::decodeGlbMeshToGeometryCore(makeGlb(animated,meshBin),std::nullopt,0,71,70,1,error),
            "animated source bytes cannot silently become a static Geometry Core mesh");
        auto morph=meshRoot; morph["meshes"][0]["primitives"][0]["targets"]=nlohmann::json::array({{{"POSITION",0}}});
        check(!videohelper::gltf::decodeGlbMeshToGeometryCore(makeGlb(morph,meshBin),std::nullopt,0,71,70,1,error),
            "morph source bytes cannot silently discard their deformation");
    }
    {
        error.clear();
        const auto decoded = videohelper::gltf::decodeStaticGlb(
            staticvisualmodelworkflowfixture::kGlb.data(),
            staticvisualmodelworkflowfixture::kGlb.size(), {}, error);
        check(decoded && error.empty()
              && decoded->metadata.extensionsUsed
                    == std::vector<std::string>{"KHR_lights_punctual"}
              && decoded->metadata.extensionsRequired
                    == std::vector<std::string>{"KHR_lights_punctual"}
              && decoded->lights.size() == 1
              && decoded->lights[0].type == videohelper::gltf::GlbLightType::Directional
              && near(decoded->lights[0].intensity, 2049.0f)
              && decoded->nodes.size() > 1
              && decoded->nodes[1].light == std::optional<std::size_t>(0),
              "the production rich GLB fixture decodes its required punctual light without test stripping");
    }
    {
        auto root = meshRoot;
        root["extensionsRequired"] = nlohmann::json::array({"KHR_lights_punctual"});
        error.clear();
        const auto decoded = videohelper::gltf::decodeStaticGlb(
            makeGlb(root, meshBin), {}, error);
        check(decoded && error.empty() && decoded->lights.size() == 2
              && decoded->nodes[0].light == std::optional<std::size_t>(0),
              "a structurally valid required punctual-light extension decodes by default");
    }
    {
        auto root = meshRoot;
        root["extensionsRequired"] = nlohmann::json::array({"KHR_texture_basisu"});
        root["extensionsUsed"].push_back("KHR_texture_basisu");
        std::string diagnostic;
        check(decodeRejected(root, meshBin, &diagnostic)
              && diagnostic.find("KHR_texture_basisu") != std::string::npos,
              "an unknown required extension remains rejected by the decoder");
    }
    {
        auto root = meshRoot;
        root["extensionsRequired"] = nlohmann::json::array({"KHR_lights_punctual"});
        root["extensions"]["KHR_lights_punctual"]["lights"][0]["color"]
            = nlohmann::json::array({1.0, -0.1, 1.0});
        check(decodeRejected(root, meshBin),
              "a required punctual light with a malformed color is rejected");

        root = meshRoot;
        root["extensionsRequired"] = nlohmann::json::array({"KHR_lights_punctual"});
        root["extensions"]["KHR_lights_punctual"]["lights"][0]["intensity"] = "bright";
        check(decodeRejected(root, meshBin),
              "a required punctual light with a malformed intensity is rejected");

        root = meshRoot;
        root["extensionsRequired"] = nlohmann::json::array({"KHR_lights_punctual"});
        root["extensions"]["KHR_lights_punctual"]["lights"][0]["range"] = 0.0;
        check(decodeRejected(root, meshBin),
              "a required punctual light with a non-positive range is rejected");

        root = meshRoot;
        root["extensionsRequired"] = nlohmann::json::array({"KHR_lights_punctual"});
        root["extensions"]["KHR_lights_punctual"]["lights"][1]["spot"]["outerConeAngle"] = 0.1;
        check(decodeRejected(root, meshBin),
              "a required spot light with crossed cone angles is rejected");

        root = meshRoot;
        root["extensionsRequired"] = nlohmann::json::array({"KHR_lights_punctual"});
        root["nodes"][0]["extensions"]["KHR_lights_punctual"]["light"] = 9;
        check(decodeRejected(root, meshBin),
              "an out-of-range node punctual-light reference is rejected");
    }
    {
        GlbAdmissionOptions mixedOptions;
        mixedOptions.supportedRequiredExtensions.push_back("EXT_caller_admitted");
        const auto admitMixed = [&] (nlohmann::json root)
        {
            root["extensionsUsed"].push_back("EXT_caller_admitted");
            root["extensionsRequired"] =
                nlohmann::json::array({"EXT_caller_admitted", "KHR_lights_punctual"});
            root["extensions"]["EXT_caller_admitted"] = nlohmann::json::object();
            error.clear();
            return videohelper::gltf::admitGlbMetadata(
                makeGlb(std::move(root), meshBin), mixedOptions, error).has_value();
        };

        auto root = meshRoot;
        root["extensions"]["KHR_lights_punctual"]["lights"][0]["intensity"] = "bright";
        check(!admitMixed(root),
              "a caller-admitted companion cannot bypass malformed punctual-light values");

        root = meshRoot;
        root.erase("extensions");
        check(!admitMixed(root),
              "a caller-admitted companion cannot bypass a missing punctual-light root payload");

        root = meshRoot;
        root["nodes"][0]["extensions"]["KHR_lights_punctual"]["light"] = 9;
        check(!admitMixed(root),
              "a caller-admitted companion cannot bypass an invalid punctual-light node reference");

        root = meshRoot;
        root["extensions"]["KHR_lights_punctual"]["lights"].erase(1);
        root["nodes"][1]["extensions"] = {{"KHR_lights_punctual", {{"light", 0}}}};
        mixedOptions.limits.maxLights = 1;
        check(!admitMixed(root),
              "a caller-admitted companion cannot bypass the punctual-light instance limit");

        mixedOptions.limits.maxLights = GlbAdmissionOptions{}.limits.maxLights;
        check(admitMixed(meshRoot),
              "valid punctual lights remain admitted with a caller-admitted required companion");
        check(decodeRejected(meshRoot, meshBin) == false,
              "the base punctual-light fixture remains production-decodable");

        root = meshRoot;
        root["extensionsUsed"].push_back("EXT_caller_admitted");
        root["extensionsRequired"] =
            nlohmann::json::array({"EXT_caller_admitted", "KHR_lights_punctual"});
        check(decodeRejected(root, meshBin),
              "production decode remains fail-closed for an unimplemented required companion");
    }
    {
        auto root = meshRoot;
        root["extensionsRequired"] = nlohmann::json::array({"KHR_lights_punctual"});
        root["extensions"]["KHR_lights_punctual"]["lights"][0]["range"] = 1.0e-320;
        check(decodeRejected(root, meshBin),
              "a positive range that underflows to zero as float is rejected before allocation");

        root = meshRoot;
        root["extensionsRequired"] = nlohmann::json::array({"KHR_lights_punctual"});
        root["extensions"]["KHR_lights_punctual"]["lights"][1]["spot"] = {
            {"innerConeAngle", 0.5},
            {"outerConeAngle", std::nextafter(0.5, 1.0)}
        };
        check(decodeRejected(root, meshBin),
              "cone angles that become equal as floats are rejected before allocation");

        root = meshRoot;
        root["extensionsRequired"] = nlohmann::json::array({"KHR_lights_punctual"});
        root["extensions"]["KHR_lights_punctual"]["lights"][1]["spot"]["outerConeAngle"]
            = 1.0e-320;
        check(decodeRejected(root, meshBin),
              "an outer cone angle that underflows to zero as float is rejected before allocation");
    }
    {
        auto root = meshRoot;
        root["extensionsRequired"] = nlohmann::json::array({"KHR_lights_punctual"});
        root["extensions"]["KHR_lights_punctual"]["lights"].erase(1);
        root["nodes"][1]["extensions"] = {{"KHR_lights_punctual", {{"light", 0}}}};
        GlbAdmissionOptions limited;
        limited.limits.maxLights = 1;
        std::string diagnostic;
        check(decodeRejected(root, meshBin, &diagnostic, limited)
              && diagnostic.find("stable identity limit") != std::string::npos,
              "punctual-light instances are identity-bounded before scene allocation");
    }
    {
        GlbAdmissionOptions decodeOptions;
        decodeOptions.sceneIndex = 1;
        error.clear();
        const auto decoded = videohelper::gltf::decodeStaticGlb(
            makeGlb(meshRoot, meshBin), decodeOptions, error);
        check(decoded.has_value() && error.empty(),
              "a bounded static triangle GLB decodes from memory");
        check(decoded && decoded->decodedVertexCount == 3
              && decoded->decodedIndexCount == 3 && decoded->decodedBytes == 212
              && decoded->bufferViews.size() == 7 && decoded->accessors.size() == 6,
              "decoded mesh and accessor allocation bounds are recorded");
        check(decoded && decoded->selectedScene == 1 && decoded->scenes.size() == 2
              && decoded->scenes[1].rootNodes == std::vector<std::size_t>{0},
              "the requested scene is selected without flattening scene roots");
        check(decoded && decoded->nodes.size() == 2
              && decoded->nodes[1].parent == std::optional<std::size_t>(0)
              && near(decoded->nodes[1].worldTransform[12], 3.0f)
              && near(decoded->nodes[1].worldTransform[13], 2.0f)
              && near(decoded->nodes[1].worldTransform[14], 3.0f),
              "node hierarchy and composed world transforms are retained");

        const bool hasPrimitive = decoded && decoded->meshes.size() == 1
                               && decoded->meshes[0].primitives.size() == 1;
        check(hasPrimitive, "the decoded document retains one mesh primitive");
        if (hasPrimitive)
        {
            const auto& primitive = decoded->meshes[0].primitives[0];
            check(primitive.positions.size() == 9 && near(primitive.positions[3], 1.0f)
                  && primitive.normals.size() == 9 && near(primitive.normals[2], 1.0f)
                  && primitive.tangents.size() == 12 && near(primitive.tangents[3], 1.0f)
                  && primitive.texCoords0.size() == 6 && near(primitive.texCoords0[2], 1.0f)
                  && primitive.colors0.size() == 12 && near(primitive.colors0[5], 1.0f)
                  && near(primitive.colors0[7], 128.0f / 255.0f)
                  && primitive.indexAccessor == 5
                  && primitive.indices == std::vector<std::uint32_t>{0, 1, 2},
                  "positions, normals, tangents, UVs, normalized colors, and indices decode");
            check(primitive.material == std::optional<std::size_t>(0),
                  "the primitive retains its material reference");
        }

        check(decoded && decoded->materials.size() == 1
              && decoded->materials[0].baseColorTexture
              && decoded->materials[0].baseColorTexture->texture == 0
              && decoded->materials[0].metallicRoughnessTexture
              && decoded->materials[0].normalTexture
              && decoded->materials[0].occlusionTexture
              && decoded->materials[0].emissiveTexture
              && near(decoded->materials[0].metallicFactor, 0.25f)
              && near(decoded->materials[0].roughnessFactor, 0.75f)
              && near(decoded->materials[0].normalScale, 0.5f)
              && near(decoded->materials[0].occlusionStrength, 0.25f)
              && near(decoded->materials[0].emissiveFactor[2], 0.3f)
              && decoded->materials[0].alphaMode == videohelper::gltf::GlbAlphaMode::Mask
              && near(decoded->materials[0].alphaCutoff, 0.4f)
              && decoded->materials[0].doubleSided,
              "material factors and texture references are retained");
        check(decoded && decoded->images.size() == 1
              && decoded->images[0].mimeType == "image/png"
              && decoded->images[0].width == 2 && decoded->images[0].height == 1
              && decoded->images[0].decodedRgba8
                    == std::vector<std::uint8_t>{255, 0, 0, 255, 0, 255, 0, 128}
              && decoded->samplers.size() == 1
              && decoded->samplers[0].magFilter == 9728
              && decoded->samplers[0].wrapT == 33648,
              "embedded PNG texels and bounded sampler values are retained");
        check(decoded && decoded->cameras.size() == 2
              && near(decoded->cameras[0].verticalFovRadians, 1.0f)
              && decoded->cameras[1].type == videohelper::gltf::GlbCameraType::Orthographic
              && near(decoded->cameras[1].xMagnification, 2.0f)
              && decoded->lights.size() == 2
              && near(decoded->lights[0].intensity, 2.0f)
              && decoded->lights[1].type == videohelper::gltf::GlbLightType::Spot
              && near(decoded->lights[1].innerConeAngle, 0.2f)
              && near(decoded->lights[1].outerConeAngle, 0.6f)
              && decoded->nodes.size() == 2
              && decoded->nodes[0].camera == std::optional<std::size_t>(0)
              && decoded->nodes[0].light == std::optional<std::size_t>(0),
              "bounded camera and punctual-light records remain attached to nodes");
    }
    {
        const auto& jpeg = jpegImageBytes();
        const auto root = staticMeshRootWithImage("image/jpeg", jpeg.size());
        error.clear();
        const auto decoded = videohelper::gltf::decodeStaticGlb(
            makeGlb(root, staticMeshBinWithImage(jpeg)), {}, error);
        const auto* texels = decoded && !decoded->images.empty()
            ? &decoded->images[0].decodedRgba8 : nullptr;
        check(decoded && error.empty() && decoded->images[0].width == 1
              && decoded->images[0].height == 1 && texels && texels->size() == 4
              && std::abs(static_cast<int>((*texels)[0]) - 64) <= 2
              && std::abs(static_cast<int>((*texels)[1]) - 128) <= 2
              && std::abs(static_cast<int>((*texels)[2]) - 192) <= 2
              && (*texels)[3] == 255,
              "embedded JPEG decodes to bounded RGBA8 texels in memory");
    }
    {
        auto unsupportedComponents = pngImageBytes();
        unsupportedComponents[24] = 16;
        std::string diagnostic;
        check(decodeRejected(staticMeshRoot(),
                             staticMeshBinWithImage(unsupportedComponents),
                             &diagnostic)
              && diagnostic == "embedded PNG image must use supported 8-bit color components",
              "unsupported PNG component depth has a deterministic pre-decode diagnostic");

        auto corrupt = pngImageBytes();
        corrupt[45] ^= 0xff;
        diagnostic.clear();
        check(decodeRejected(staticMeshRoot(), staticMeshBinWithImage(corrupt), &diagnostic)
              && diagnostic == "embedded PNG image decode failed",
              "corrupt embedded image payload fails closed with a deterministic diagnostic");

        auto wrongMimeRoot = staticMeshRoot();
        wrongMimeRoot["images"][0]["mimeType"] = "image/jpeg";
        diagnostic.clear();
        check(decodeRejected(wrongMimeRoot, staticMeshBin(), &diagnostic)
              && diagnostic == "embedded JPEG image has a malformed SOI marker",
              "declared image MIME type must match the embedded payload framing");
    }
    {
        auto root = meshRoot;
        root["accessors"][0]["count"] = 4;
        check(decodeRejected(root, meshBin),
              "an accessor that exceeds its bufferView is rejected");
    }
    {
        auto root = meshRoot;
        root["meshes"][0]["primitives"][0].erase("indices");
        std::string diagnostic;
        const auto decoded = videohelper::gltf::decodeStaticGlb(makeGlb(root, meshBin), {}, diagnostic);
        check(decoded && diagnostic.empty()
              && !decoded->meshes[0].primitives[0].indexAccessor
              && decoded->meshes[0].primitives[0].indices == std::vector<std::uint32_t>{0, 1, 2}
              && decoded->decodedIndexCount == 3 && decoded->decodedBytes == 212,
              "non-indexed TRIANGLES receive explicit indices charged to count and byte budgets");

        for (const auto count : {0u, 2u, 4u, std::numeric_limits<std::uint32_t>::max()})
        {
            auto malformed = root;
            malformed["accessors"][0]["count"] = count;
            check(decodeRejected(malformed, meshBin),
                  "non-indexed zero, incomplete, overrun and huge POSITION counts fail closed");
        }
        for (const auto mode : {0, 1, 2, 3, 5, 6})
        {
            auto malformed = root;
            malformed["meshes"][0]["primitives"][0]["mode"] = mode;
            check(decodeRejected(malformed, meshBin),
                  "index synthesis admits only TRIANGLES topology");
        }
        for (const auto& badIndex : {nlohmann::json(nullptr), nlohmann::json(-1),
                                   nlohmann::json(999), nlohmann::json("absent")})
        {
            auto malformed = root;
            malformed["meshes"][0]["primitives"][0]["indices"] = badIndex;
            check(decodeRejected(malformed, meshBin),
                  "a present malformed index reference cannot request sequential synthesis");
        }
        auto malformed = root;
        malformed["accessors"][1]["count"] = 2;
        check(decodeRejected(malformed, meshBin),
              "non-indexed NORMAL count must still match POSITION");
        malformed = root;
        malformed["accessors"][0]["normalized"] = true;
        check(decodeRejected(malformed, meshBin),
              "non-indexed POSITION retains its strict FLOAT accessor contract");
        auto nonfinite = meshBin;
        const auto nan = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(nonfinite.data(), &nan, sizeof(nan));
        check(decodeRejected(root, nonfinite), "non-indexed non-finite vertices fail closed");

        GlbAdmissionOptions limited;
        limited.limits.maxDecodedIndices = 2;
        check(decodeRejected(root, meshBin, &diagnostic, limited)
              && diagnostic == "decoded glTF index count exceeds the admission limit",
              "generated indices are bounded before allocation");
        limited = {};
        limited.limits.maxDecodedVertices = 2;
        check(decodeRejected(root, meshBin, nullptr, limited),
              "non-indexed vertices cannot bypass the vertex budget");
        limited = {};
        limited.limits.maxDecodedBytes = 211;
        check(decodeRejected(root, meshBin, nullptr, limited),
              "generated uint32 index bytes cannot bypass the decoded byte budget");
        root["meshes"][0]["primitives"].push_back(root["meshes"][0]["primitives"][0]);
        limited = {};
        limited.limits.maxDecodedIndices = 5;
        check(decodeRejected(root, meshBin, nullptr, limited),
              "sequential index limits are aggregate across primitives");
    }
    {
        auto root = meshRoot;
        auto& primitive = root["meshes"][0]["primitives"][0];
        primitive.erase("indices");
        primitive["attributes"].erase("NORMAL");
        auto bin = meshBin;
        const float raisedCorner = 1.0f;
        std::memcpy(bin.data() + 8 * sizeof(float), &raisedCorner, sizeof(float));
        const auto nan = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(bin.data() + 72, &nan, sizeof(float)); // Ignored TANGENT data.
        std::string diagnostic;
        const auto decoded = videohelper::gltf::decodeStaticGlb(makeGlb(root, bin), {}, diagnostic);
        check(decoded && diagnostic.empty() && decoded->decodedVertexCount == 3
              && decoded->decodedIndexCount == 3 && decoded->decodedBytes == 164,
              "missing flat normals are generated within the decoded byte budget without vertex expansion");
        if (decoded)
        {
            const auto& values = decoded->meshes[0].primitives[0];
            check(values.generatedFlatNormals && !values.normalAccessor && !values.tangentAccessor
                  && values.tangents.empty() && values.normals.size() == 9
                  && values.indices == std::vector<std::uint32_t>{0, 1, 2}
                  && near(values.normals[0], 0.0f) && near(values.normals[1], -std::sqrt(0.5f))
                  && near(values.normals[2], std::sqrt(0.5f)),
                  "non-axis-aligned face normals derive from positions and ignore provided tangents");
            bool sharedFaceNormal = true;
            for (std::size_t corner = 1; corner < 3; ++corner)
                for (std::size_t axis = 0; axis < 3; ++axis)
                    sharedFaceNormal = sharedFaceNormal
                        && values.normals[corner * 3 + axis] == values.normals[axis];
            check(sharedFaceNormal && values.texCoords0.size() == 6 && values.colors0.size() == 12,
                  "all three corners share one flat normal while UV and color order remains intact");
        }
        GlbAdmissionOptions limited;
        limited.limits.maxDecodedBytes = 163;
        check(decodeRejected(root, bin, &diagnostic, limited)
              && diagnostic == "decoded glTF byte size exceeds the admission limit",
              "generated normal storage cannot evade a one-byte-short aggregate budget");
        auto indexed = root;
        indexed["meshes"][0]["primitives"][0]["indices"] = 5;
        const auto indexedValue = videohelper::gltf::decodeStaticGlb(makeGlb(indexed, bin), {}, diagnostic);
        check(indexedValue && indexedValue->decodedBytes == 176
              && indexedValue->meshes[0].primitives[0].cornerSources == std::vector<std::uint32_t>{0, 1, 2}
              && decoded && indexedValue->meshes[0].primitives[0].normals == decoded->meshes[0].primitives[0].normals,
              "indexed missing normals retain charged source-corner correspondence and flat normals");
        auto invalidReference = root;
        invalidReference["meshes"][0]["primitives"][0]["attributes"]["TANGENT"] = 999;
        check(decodeRejected(invalidReference, bin),
              "ignored tangent data still requires a valid bounded accessor reference");

        for (const auto scale : {1.0e-25f, 1.0e25f})
        {
            auto scaled = bin;
            const std::array<float, 9> positions {{0, 0, 0, scale, 0, 0, 0, scale, scale}};
            std::memcpy(scaled.data(), positions.data(), sizeof(positions));
            const auto value = videohelper::gltf::decodeStaticGlb(makeGlb(root, scaled), {}, diagnostic);
            check(value && near(value->meshes[0].primitives[0].normals[1], -std::sqrt(0.5f))
                  && near(value->meshes[0].primitives[0].normals[2], std::sqrt(0.5f)),
                  "double cross products normalize finite tiny and large triangles without float overflow");
        }
        auto degenerate = bin;
        const std::array<float, 9> collinear {{0, 0, 0, 1, 0, 0, 2, 0, 0}};
        std::memcpy(degenerate.data(), collinear.data(), sizeof(collinear));
        check(decodeRejected(root, degenerate, &diagnostic)
              && diagnostic == "cannot generate flat glTF normals for a degenerate triangle",
              "degenerate faces cannot receive invented fallback normals");
        std::memcpy(degenerate.data(), &nan, sizeof(float));
        check(decodeRejected(root, degenerate), "non-finite missing-normal positions fail admission");
    }
    {
        auto root = meshRoot;
        auto bin = meshBin;
        while (bin.size() % 4 != 0) bin.push_back(0);
        const auto appendAccessor = [&](std::size_t offset, std::size_t byteLength,
                                        std::size_t count, int componentType,
                                        const char* type, bool normalized, int target) {
            const auto view = root["bufferViews"].size();
            root["bufferViews"].push_back({{"buffer", 0}, {"byteOffset", offset},
                                          {"byteLength", byteLength}, {"target", target}});
            const auto accessor = root["accessors"].size();
            root["accessors"].push_back({{"bufferView", view}, {"componentType", componentType},
                {"count", count}, {"type", type}, {"normalized", normalized}});
            return accessor;
        };
        auto offset = bin.size();
        for (const auto value : {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}) appendFloat(bin, value);
        const auto position = appendAccessor(offset, 48, 4, 5126, "VEC3", false, 34962);
        offset = bin.size();
        for (const auto value : {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f}) appendFloat(bin, value);
        const auto uv = appendAccessor(offset, 32, 4, 5126, "VEC2", false, 34962);
        offset = bin.size();
        bin.insert(bin.end(), {255, 0, 0, 255, 0, 255, 0, 128,
                               0, 0, 255, 64, 255, 255, 255, 255});
        const auto color = appendAccessor(offset, 16, 4, 5121, "VEC4", true, 34962);
        offset = bin.size();
        const std::vector<std::uint32_t> sources {0, 1, 2, 0, 3, 1};
        for (const auto index : sources) appendU16(bin, static_cast<std::uint16_t>(index));
        const auto indices = appendAccessor(offset, 12, 6, 5123, "SCALAR", false, 34963);
        auto& primitive = root["meshes"][0]["primitives"][0];
        primitive["attributes"] = {{"POSITION", position}, {"TEXCOORD_0", uv}, {"COLOR_0", color}};
        primitive["indices"] = indices;
        root["buffers"][0]["byteLength"] = bin.size();
        std::string diagnostic;
        const auto decoded = videohelper::gltf::decodeStaticGlb(makeGlb(root, bin), {}, diagnostic);
        // Six corners carry twelve floats each, six indices and six source
        // indices, plus the fixture's eight decoded image bytes.
        check(decoded && decoded->decodedVertexCount == 6 && decoded->decodedIndexCount == 6
              && decoded->decodedBytes == 344 && diagnostic.empty(),
              "indexed sharp faces debit expanded attributes, indices and retained remap before allocation");
        if (decoded)
        {
            const auto& values = decoded->meshes[0].primitives[0];
            check(values.cornerSources == sources && values.indexAccessor == indices
                  && values.positionAccessor == position && values.material == 0
                  && values.indices == std::vector<std::uint32_t>{0, 1, 2, 3, 4, 5}
                  && values.normals[2] == 1 && values.normals[10] == 1
                  && values.normals[11] == 0 && values.positions[14] == 1
                  && values.texCoords0[8] == 0.5f && values.texCoords0[10] == 1
                  && values.colors0[16] == 1 && near(values.colors0[23], 128.0f / 255.0f),
                  "shared source vertices retain sharp face normals and duplicate UV/color by exact original index order");
            const auto scene = videohelper::gltf::adaptStaticGlbToVisual3DScene(*decoded, diagnostic);
            check(scene && scene->vertexCount == 6 && scene->objects[0].vertexCount == 6
                  && scene->objects[0].id.value == 1 && scene->objects[0].material.value == 1,
                  "expanded static geometry retains source object and material identity in the scene adapter");
            const auto geometry = videohelper::gltf::adaptGlbMeshToGeometryCore(*decoded, 0, 71, 70, 3, diagnostic);
            const auto repeated = videohelper::gltf::adaptGlbMeshToGeometryCore(*decoded, 0, 71, 70, 3, diagnostic);
            if (geometry && repeated)
            {
                const auto& mesh = std::get<videowire::geometry::GeometryData>(geometry->descriptor().data);
                check(mesh.vertexIds == std::vector<videowire::geometry::StableId>{65537, 65538, 65539, 65540, 65541, 65542}
                      && mesh.vertexIds == std::get<videowire::geometry::GeometryData>(repeated->descriptor().data).vertexIds
                      && mesh.indices == values.indices
                      && geometry->descriptor().attributes[0].elements[3].components[1] == 1,
                      "Geometry Core assigns distinct repeatable corner IDs with exact topology and sharp normals");
            }
            else check(false, "expanded imported mesh reaches existing Geometry Core admission");
        }
        for (const auto componentType : {5121, 5125})
        {
            auto encodedRoot = root;
            auto encodedBin = bin;
            encodedBin.resize(offset);
            for (const auto index : sources)
                if (componentType == 5121) encodedBin.push_back(static_cast<std::uint8_t>(index));
                else appendU32(encodedBin, index);
            encodedRoot["buffers"][0]["byteLength"] = encodedBin.size();
            encodedRoot["bufferViews"].back()["byteLength"] = encodedBin.size() - offset;
            encodedRoot["accessors"][indices]["componentType"] = componentType;
            const auto value = videohelper::gltf::decodeStaticGlb(makeGlb(encodedRoot, encodedBin), {}, diagnostic);
            check(value && value->decodedBytes == 344
                  && value->meshes[0].primitives[0].cornerSources == sources,
                  "byte and uint index encodings retain the same charged flat corner mapping");
        }
        auto unused = bin;
        // Both faces now use vertices zero, one and two, so vertex three is
        // unreferenced. Its carried attributes must still pass source checks.
        const std::uint16_t sourceTwo = 2;
        std::memcpy(unused.data() + offset + 4 * sizeof(std::uint16_t), &sourceTwo, sizeof(sourceTwo));
        const auto nan = std::numeric_limits<float>::quiet_NaN();
        const auto positionView = root["accessors"][position]["bufferView"].get<std::size_t>();
        const auto positionOffset = root["bufferViews"][positionView]["byteOffset"].get<std::size_t>();
        std::memcpy(unused.data() + positionOffset + 9 * sizeof(float), &nan, sizeof(float));
        check(decodeRejected(root, unused), "unused indexed source positions cannot hide non-finite data");
        GlbAdmissionOptions limited;
        limited.limits.maxDecodedBytes = 343;
        check(decodeRejected(root, bin, &diagnostic, limited)
              && diagnostic == "decoded glTF byte size exceeds the admission limit",
              "one-byte-short expanded geometry invoice refuses before allocation");
        limited = {}; limited.limits.maxDecodedVertices = 5;
        check(decodeRejected(root, bin, &diagnostic, limited)
              && diagnostic == "decoded glTF vertex count exceeds the admission limit",
              "corner expansion cannot bypass the vertex cap with four source vertices");
        limited = {}; limited.limits.maxDecodedIndices = 5;
        check(decodeRejected(root, bin, &diagnostic, limited), "expanded topology retains the index cap");
        auto malformed = bin;
        const std::uint16_t outOfRange = 4;
        std::memcpy(malformed.data() + offset, &outOfRange, sizeof(outOfRange));
        check(decodeRejected(root, malformed), "flat corner mapping rejects an out-of-range original source index");
        const std::uint16_t repeatedCorner = 1;
        std::memcpy(malformed.data() + offset, &repeatedCorner, sizeof(repeatedCorner));
        check(decodeRejected(root, malformed, &diagnostic)
              && diagnostic == "cannot generate flat glTF normals for a degenerate triangle",
              "expanded indexed degenerate faces cannot receive a guessed normal");
    }
    {
        constexpr std::size_t repeats = 576;
        constexpr std::size_t vertices = repeats * 3;
        const auto& image = rgb1024ImageBytes();
        auto root = staticMeshRootWithImage("image/png", image.size());
        root["meshes"][0]["primitives"][0].erase("indices");
        root["meshes"][0]["primitives"][0]["attributes"].erase("NORMAL");
        std::vector<std::uint8_t> bin;
        for (std::size_t attribute = 0; attribute < 5; ++attribute)
        {
            const auto offset = root["bufferViews"][attribute]["byteOffset"].get<std::size_t>();
            const auto length = root["bufferViews"][attribute]["byteLength"].get<std::size_t>();
            root["bufferViews"][attribute]["byteOffset"] = bin.size();
            root["bufferViews"][attribute]["byteLength"] = length * repeats;
            root["accessors"][attribute]["count"] = vertices;
            for (std::size_t repeat = 0; repeat < repeats; ++repeat)
                bin.insert(bin.end(), meshBin.begin() + offset, meshBin.begin() + offset + length);
        }
        root["bufferViews"][5]["byteOffset"] = bin.size();
        bin.insert(bin.end(), {0, 0, 1, 0, 2, 0, 0, 0}); // Unused, valid index accessor.
        root["bufferViews"][6]["byteOffset"] = bin.size();
        bin.insert(bin.end(), image.begin(), image.end());
        root["buffers"][0]["byteLength"] = bin.size();
        const auto bytes = makeGlb(root, bin);
        auto native = videohelper::nativeImportedAnimationDecodeOptions(bytes.size(), {});
        std::string diagnostic;
        const auto decoded = videohelper::gltf::decodeAnimatedGlbBaseScene(bytes, native.admission, diagnostic);
        check(decoded && diagnostic.empty() && decoded->decodedVertexCount == vertices
              && decoded->decodedIndexCount == vertices
              && decoded->decodedBytes == 4194304u + vertices * 52u
              && decoded->meshes[0].primitives[0].generatedFlatNormals,
              "production import bounds admit 1728 non-indexed vertices, generated normals and a full 1024 RGB PNG");
        if (decoded)
        {
            const auto& primitive = decoded->meshes[0].primitives[0];
            bool sequential = !primitive.indexAccessor && primitive.indices.size() == vertices;
            for (std::size_t index = 0; index < primitive.indices.size(); ++index)
                sequential = sequential && primitive.indices[index] == index;
            check(sequential, "all 1728 generated indices preserve source vertex order");
            const auto& rgba = decoded->images[0].decodedRgba8;
            check(decoded->images[0].width == 1024 && decoded->images[0].height == 1024
                  && rgba.size() == 4194304 && rgba[0] == 10 && rgba[1] == 20 && rgba[2] == 30
                  && rgba[3] == 255 && rgba[2048] == 40 && rgba[2097152] == 70
                  && rgba[rgba.size() - 4] == 100 && rgba.back() == 255,
                  "RGB expands to full-size RGBA8 with unchanged corner pixels and opaque alpha");
        }
        auto limited = native.admission;
        limited.limits.maxDecodedImageBytes = 4194303;
        check(decodeRejected(root, bin, &diagnostic, limited)
              && diagnostic == "decoded embedded image bytes exceed the image admission limit",
              "a one-byte-short RGBA budget rejects the full image before allocation");

        const auto imageOffset = root["bufferViews"][6]["byteOffset"].get<std::size_t>();
        auto duplicateImage = root;
        duplicateImage["images"].push_back(duplicateImage["images"][0]);
        auto duplicateImageAdmission = native.admission;
        duplicateImageAdmission.limits.maxContainerBytes = makeGlb(duplicateImage, bin).size();
        const bool duplicateRejected = decodeRejected(
            duplicateImage, bin, &diagnostic, duplicateImageAdmission);
        if (!duplicateRejected || diagnostic != "decoded embedded image bytes exceed the image admission limit")
            std::fprintf(stderr, "repeated image decode: rejected=%d diagnostic=%s\n",
                         duplicateRejected, diagnostic.c_str());
        check(duplicateRejected
              && diagnostic == "decoded embedded image bytes exceed the image admission limit",
              "decoded image bounds remain aggregate across repeated embedded images");
        auto oversized = bin;
        oversized[imageOffset + 19] = 1; // Header width 1025, still within dimension cap.
        check(decodeRejected(root, oversized, &diagnostic, native.admission)
              && diagnostic == "decoded embedded image bytes exceed the image admission limit",
              "an oversized decoded image header is rejected before invoking the codec");
        auto truncated = bin;
        truncated.resize(imageOffset + 33);
        auto truncatedRoot = root;
        truncatedRoot["bufferViews"][6]["byteLength"] = 33;
        truncatedRoot["buffers"][0]["byteLength"] = truncated.size();
        check(decodeRejected(truncatedRoot, truncated, &diagnostic, native.admission)
              && diagnostic == "embedded PNG image decode failed",
              "a valid-size image header without compressed pixels cannot publish a texture");
    }
    {
        auto root = meshRoot;
        root["bufferViews"][0]["target"] = 34963;
        check(decodeRejected(root, meshBin),
              "vertex attribute bufferViews cannot claim an index-buffer target");
        root = meshRoot;
        root["bufferViews"][5]["target"] = 34962;
        check(decodeRejected(root, meshBin),
              "index bufferViews cannot claim a vertex-buffer target");
    }
    {
        check(decodeRejected(meshRoot, staticMeshBin(true)),
              "an index outside the POSITION vertex range is rejected");
    }
    {
        auto root = meshRoot;
        root["meshes"][0]["primitives"][0]["material"] = 1;
        check(decodeRejected(root, meshBin),
              "an out-of-range primitive material reference is rejected");
    }
    {
        GlbAdmissionOptions limited;
        limited.limits.maxDecodedVertices = 2;
        check(decodeRejected(meshRoot, meshBin, nullptr, limited),
              "the decoded vertex limit is enforced before allocation");
        limited = {};
        limited.limits.maxDecodedIndices = 2;
        check(decodeRejected(meshRoot, meshBin, nullptr, limited),
              "the decoded index limit is enforced before allocation");
        limited = {};
        limited.limits.maxDecodedBytes = 211;
        check(decodeRejected(meshRoot, meshBin, nullptr, limited),
              "the decoded byte limit includes mesh and embedded image records");
        limited = {};
        limited.limits.maxEmbeddedImageBytes = pngImageBytes().size() - 1;
        check(decodeRejected(meshRoot, meshBin, nullptr, limited),
              "the embedded image byte limit is enforced");
        limited = {};
        limited.limits.maxImageWidth = 1;
        check(decodeRejected(meshRoot, meshBin, nullptr, limited),
              "embedded image dimensions are enforced from the header before decode");
        limited = {};
        limited.limits.maxDecodedImageBytes = 7;
        check(decodeRejected(meshRoot, meshBin, nullptr, limited),
              "the decoded embedded image byte limit is enforced before texel allocation");
        limited = {};
        limited.limits.maxLights = 0;
        check(decodeRejected(meshRoot, meshBin, nullptr, limited),
              "the punctual-light record limit is enforced");
    }
    {
        GlbAdmissionOptions selected;
        selected.sceneIndex = 2;
        check(decodeRejected(meshRoot, meshBin, nullptr, selected),
              "an out-of-range requested scene is rejected");
    }
    {
        auto root = meshRoot;
        root["animations"] = nlohmann::json::array({nlohmann::json::object()});
        GlbAdmissionOptions callerAllowed;
        callerAllowed.admitAnimations = true;
        check(decodeRejected(root, meshBin, nullptr, callerAllowed),
              "static decode rejects animations even when metadata admission allows them");
        root = meshRoot;
        root["skins"] = nlohmann::json::array({nlohmann::json::object()});
        callerAllowed = {};
        callerAllowed.admitSkins = true;
        check(decodeRejected(root, meshBin, nullptr, callerAllowed),
              "static decode rejects skins even when metadata admission allows them");
    }
    {
        auto root = meshRoot;
        root.erase("extensions");
        for (auto& node : root["nodes"])
            node.erase("extensions");
        root["extensionsUsed"] = nlohmann::json::array({"KHR_draco_mesh_compression"});
        root["extensionsRequired"] = root["extensionsUsed"];
        GlbAdmissionOptions callerAllowed;
        callerAllowed.supportedRequiredExtensions.push_back("KHR_draco_mesh_compression");
        std::string diagnostic;
        check(decodeRejected(root, meshBin, &diagnostic, callerAllowed)
              && diagnostic.find("KHR_draco_mesh_compression") != std::string::npos,
              "static decode rejects a caller-admitted required extension it does not implement");

        root = meshRoot;
        root["extensionsRequired"] = nlohmann::json::array({"KHR_lights_punctual"});
        GlbAdmissionOptions lightsAllowed;
        lightsAllowed.supportedRequiredExtensions.push_back("KHR_lights_punctual");
        error.clear();
        check(videohelper::gltf::decodeStaticGlb(makeGlb(root, meshBin), lightsAllowed, error)
                  .has_value(),
              "KHR_lights_punctual may be required when the caller admits it");

        root = meshRoot;
        root.erase("extensionsUsed");
        check(decodeRejected(root, meshBin),
              "a punctual-light payload must be declared in extensionsUsed");
    }
    {
        auto root = meshRoot;
        root["buffers"][0]["uri"] = "outside.bin";
        check(decodeRejected(root, meshBin),
              "external buffer URIs are rejected without opening them");
        root = meshRoot;
        root["images"][0]["uri"] = "outside.png";
        check(decodeRejected(root, meshBin),
              "external image URIs are rejected without opening them");
    }

    const auto animatedRoot = animatedDeformationRoot();
    const auto animatedBin = animatedDeformationBin();
    {
        // Transform-only geometry has no skin or morph to enter the ordinary
        // deformation path. Its retained rigid skin still needs six corners.
        for (const bool ancestorTrack : {false, true})
        {
            auto root = animatedRoot;
            auto bin = animatedBin;
            root.erase("skins");
            root["nodes"][0].erase("skin");
            auto& primitive = root["meshes"][0]["primitives"][0];
            primitive["attributes"] = {{"POSITION", 0}};
            primitive.erase("targets");
            root["meshes"][0].erase("weights");
            root["nodes"].push_back({{"children", {0}}});
            root["scene"] = 0;
            root["scenes"] = nlohmann::json::array({{{"nodes", {2}}}});
            root["animations"][0]["samplers"].erase(root["animations"][0]["samplers"].begin() + 1);
            root["animations"][0]["channels"] = nlohmann::json::array({{
                {"sampler", 0}, {"target", {{"node", ancestorTrack ? 2 : 0}, {"path", "translation"}}}
            }});
            const auto offset = bin.size();
            for (const auto index : {0u, 1u, 2u, 0u, 1u, 2u}) appendU16(bin, static_cast<std::uint16_t>(index));
            const auto view = root["bufferViews"].size();
            root["bufferViews"].push_back({{"buffer", 0}, {"byteOffset", offset},
                                          {"byteLength", 12}, {"target", 34963}});
            primitive["indices"] = root["accessors"].size();
            root["accessors"].push_back({{"bufferView", view}, {"componentType", 5123},
                                        {"count", 6}, {"type", "SCALAR"}});
            root["buffers"][0]["byteLength"] = bin.size();
            videohelper::gltf::GlbAnimationDecodeOptions options;
            options.retainGeometryHierarchy = true;
            options.admission.admitAnimations = options.admission.admitSkins = true;
            std::string diagnostic;
            const auto bytes = makeGlb(root, bin);
            const auto base = videohelper::gltf::decodeAnimatedGlbBaseScene(bytes, options.admission, diagnostic);
            const auto decoded = videohelper::gltf::decodeGlbAnimations(bytes, options, diagnostic);
            check(base && decoded && decoded->deformation && base->decodedVertexCount == 6
                  && decoded->deformation->meshes()[0].vertexCount() == 6
                  && decoded->deformation->meshes()[0].id().value == 1
                  && decoded->deformation->meshes()[0].jointWeightSets()[0].weights().size() == 24
                  && decoded->deformation->skins()[0].joints().size() == 2
                  && decoded->renderBindings[0].nodeIndex == 0
                  && decoded->clips[0].clip->tracks()[0].target().value == (ancestorTrack ? 3u : 1u),
                  "transform-only node and ancestor tracks retain six base/deformation corners and original IDs");
            // 40 animation bytes, 24 corner-map bytes, 128 inverse-bind bytes,
            // and 192 synthetic influence bytes are charged before growth.
            options.limits.maxDecodedBytes = 384;
            check(videohelper::gltf::decodeGlbAnimations(bytes, options, diagnostic).has_value(),
                  "exact transform-only expanded invoice is admissible");
            options.limits.maxDecodedBytes = 383;
            animationFactoryCalls = deformationFactoryCalls = 0;
            const auto overBudget = videohelper::gltf::detail::decodeGlbAnimationsWithFactories(
                bytes.data(), bytes.size(), options, countingAnimationFactory, countingDeformationFactory, diagnostic);
            check(!overBudget && animationFactoryCalls == 0 && deformationFactoryCalls == 0
                  && diagnostic == "decoded glTF animation or deformation bytes exceed the admission limit",
                  "one-byte-short rigid invoice refuses before immutable publication");
            options.limits = {};
            options.deformationLimits.maxVerticesPerMesh = 5;
            check(animationRejected(root, bin, &diagnostic, options)
                  && diagnostic == "animated Geometry3D vertex capacity exceeded",
                  "rigid per-mesh capacity charges six corners before growth");
            options.deformationLimits = {};
            options.deformationLimits.maxTotalVertices = 5;
            check(animationRejected(root, bin, &diagnostic, options)
                  && diagnostic == "animated Geometry3D vertex capacity exceeded",
                  "rigid aggregate capacity charges output corners before growth");
            options.deformationLimits = {};
            options.admission.limits.maxDecodedIndices = 5;
            check(animationRejected(root, bin, &diagnostic, options),
                  "rigid corner indices retain the existing admission index cap");
            options.admission.limits = {};
            options.deformationLimits.maxTotalJoints = 1;
            check(animationRejected(root, bin, &diagnostic, options)
                  && diagnostic == "animated Geometry3D transform hierarchy exceeds the skin capacity",
                  "rigid ancestors retain the aggregate joint capacity before palette growth");
            options.deformationLimits = {};
            options.deformationLimits.maxMeshes = 0;
            check(animationRejected(root, bin, &diagnostic, options)
                  && diagnostic == "deformed mesh capacity exceeded",
                  "synthetic rigid meshes cannot bypass the existing mesh capacity");
            options.deformationLimits = {};
            auto multiPrimitive = root;
            multiPrimitive["meshes"][0]["primitives"].push_back(primitive);
            options.deformationLimits.maxVerticesPerMesh = 11;
            check(animationRejected(multiPrimitive, bin, &diagnostic, options)
                  && diagnostic == "animated Geometry3D vertex capacity exceeded",
                  "rigid primitive corner totals cannot bypass the mesh cap");
            options.deformationLimits = {};
            options.deformationLimits.maxTotalVertices = 11;
            check(animationRejected(multiPrimitive, bin, &diagnostic, options)
                  && diagnostic == "animated Geometry3D vertex capacity exceeded",
                  "rigid primitive corner totals cannot bypass the aggregate cap");
            options.deformationLimits = {};
            auto invalidBin = bin;
            invalidBin[offset + 10] = 3;
            check(animationRejected(root, invalidBin, &diagnostic, options)
                  && diagnostic == "glTF triangle index references a vertex outside POSITION",
                  "rigid synthesis uses the original index range check");
        }
    }
    {
        // The last authored vertex is valid geometry but not referenced by any
        // triangle. Its influences must not disappear from source validation.
        for (const bool authoredNormals : {false, true})
            for (const bool retainAncestors : {false, true})
                for (const std::size_t setCount : {1u, 2u})
                {
                    auto root = animatedRoot;
                    root.erase("animations");
                    root["meshes"][0].erase("weights");
                    auto& primitive = root["meshes"][0]["primitives"][0];
                    primitive.erase("targets");
                    primitive["attributes"] = nlohmann::json::object();
                    root["skins"][0].erase("inverseBindMatrices");
                    root["nodes"].push_back({{"children", {0}}});
                    root["scene"] = 0;
                    root["scenes"] = nlohmann::json::array({{{"nodes", {2}}}});
                    root["accessors"] = root["bufferViews"] = nlohmann::json::array();
                    std::vector<std::uint8_t> bin;
                    const auto appendAccessor = [&](std::size_t offset, std::size_t byteLength,
                                                    int component, const char* type, std::size_t count, int target) {
                        const auto view = root["bufferViews"].size();
                        root["bufferViews"].push_back({{"buffer", 0}, {"byteOffset", offset},
                                                      {"byteLength", byteLength}, {"target", target}});
                        const auto index = root["accessors"].size();
                        root["accessors"].push_back({{"bufferView", view}, {"componentType", component},
                                                    {"count", count}, {"type", type}});
                        return index;
                    };
                    for (const auto value : {0.f,0.f,0.f, 1.f,0.f,0.f, 0.f,1.f,0.f, 2.f,2.f,2.f})
                        appendFloat(bin, value);
                    primitive["attributes"]["POSITION"] = appendAccessor(0, 48, 5126, "VEC3", 4, 34962);
                    if (authoredNormals)
                    {
                        const auto offset = bin.size();
                        for (int vertex = 0; vertex < 4; ++vertex)
                            for (const auto value : {0.f, 0.f, 1.f}) appendFloat(bin, value);
                        primitive["attributes"]["NORMAL"] = appendAccessor(offset, 48, 5126, "VEC3", 4, 34962);
                    }
                    std::vector<std::size_t> jointOffsets, weightOffsets;
                    for (std::size_t set = 0; set < setCount; ++set)
                    {
                        jointOffsets.push_back(bin.size());
                        bin.insert(bin.end(), 16, 0);
                        primitive["attributes"]["JOINTS_" + std::to_string(set)]
                            = appendAccessor(jointOffsets.back(), 16, 5121, "VEC4", 4, 34962);
                        weightOffsets.push_back(bin.size());
                        for (int vertex = 0; vertex < 4; ++vertex)
                            for (const auto value : {1.0f / static_cast<float>(setCount), 0.f, 0.f, 0.f}) appendFloat(bin, value);
                        primitive["attributes"]["WEIGHTS_" + std::to_string(set)]
                            = appendAccessor(weightOffsets.back(), 64, 5126, "VEC4", 4, 34962);
                    }
                    const auto indexOffset = bin.size();
                    for (const auto index : {0u, 1u, 2u}) appendU16(bin, static_cast<std::uint16_t>(index));
                    primitive["indices"] = appendAccessor(indexOffset, 6, 5123, "SCALAR", 3, 34963);
                    root["buffers"][0]["byteLength"] = bin.size();
                    videohelper::gltf::GlbAnimationDecodeOptions options;
                    options.retainGeometryHierarchy = retainAncestors;
                    std::string diagnostic;
                    const auto decoded = videohelper::gltf::decodeGlbAnimations(makeGlb(root, bin), options, diagnostic);
                    check(decoded && decoded->deformation
                          && decoded->deformation->meshes()[0].vertexCount() == (authoredNormals ? 4u : 3u)
                          && decoded->deformation->meshes()[0].jointWeightSets().size() == setCount
                          && decoded->deformation->skins()[0].joints().size() == (retainAncestors ? 3u : 1u),
                          "unused valid source influences preserve authored/flat topology and contiguous sets with ancestors");
                    for (const auto component : {0u, 3u})
                    {
                        auto invalid = bin;
                        invalid[jointOffsets.back() + 12 + component] = 1;
                        check(animationRejected(root, invalid, &diagnostic, options)
                              && diagnostic == "joint accessor index is outside the original glTF skin",
                              "unused source joints including zero-weight components cannot address appended ancestors");
                    }
                    for (const auto invalidSum : {0.f, 0.75f, 1.25f})
                    {
                        auto invalid = bin;
                        for (const auto offset : weightOffsets)
                        {
                            const float weight = invalidSum / static_cast<float>(setCount);
                            std::memcpy(invalid.data() + offset + 12 * sizeof(float), &weight, sizeof(weight));
                        }
                        check(animationRejected(root, invalid, &diagnostic, options)
                              && diagnostic == "skin weights for a vertex are not normalized",
                              "unused source zero and non-unit sums are refused across all contiguous influence sets");
                    }
                    for (const auto invalidScalar : {-0.25f, std::numeric_limits<float>::infinity()})
                    {
                        auto invalid = bin;
                        std::memcpy(invalid.data() + weightOffsets.back() + 15 * sizeof(float),
                                    &invalidScalar, sizeof(invalidScalar));
                        check(animationRejected(root, invalid, &diagnostic, options)
                              && diagnostic == "glTF WEIGHTS accessor contains a negative or nonfinite value",
                              "unused source scalar weight checks remain strict");
                    }
                    if (!authoredNormals)
                    {
                        auto nearUnit = bin;
                        const float within = 1.0f / static_cast<float>(setCount) + 0.00005f;
                        std::memcpy(nearUnit.data() + weightOffsets[0] + 12 * sizeof(float), &within, sizeof(within));
                        options.deformationLimits.weightSumTolerance = 1.0f;
                        check(videohelper::gltf::decodeGlbAnimations(makeGlb(root, nearUnit), options, diagnostic).has_value(),
                              "source cross-set sums accept the existing bounded tolerance");
                        const float outside = 1.0f / static_cast<float>(setCount) + 0.0002f;
                        std::memcpy(nearUnit.data() + weightOffsets[0] + 12 * sizeof(float), &outside, sizeof(outside));
                        check(animationRejected(root, nearUnit, &diagnostic, options)
                              && diagnostic == "skin weights for a vertex are not normalized",
                              "requested loose tolerance cannot exceed the existing hard tolerance");
                        options.deformationLimits.weightSumTolerance = std::numeric_limits<float>::quiet_NaN();
                        std::memcpy(nearUnit.data() + weightOffsets[0] + 12 * sizeof(float), &within, sizeof(within));
                        check(animationRejected(root, nearUnit, &diagnostic, options),
                              "nonfinite requested weight tolerance becomes zero for source validation");
                        options.deformationLimits = {};
                        // Original skin plus retained ancestors, three-corner map,
                        // four double source sums and three mapped VEC4s per set.
                        const auto invoice = (retainAncestors ? 192u : 64u) + 12u + 32u + setCount * 96u;
                        options.limits.maxDecodedBytes = invoice;
                        check(videohelper::gltf::decodeGlbAnimations(makeGlb(root, bin), options, diagnostic).has_value(),
                              "exact unused-source validation storage invoice is admissible");
                        options.limits.maxDecodedBytes = invoice - 1;
                        check(animationRejected(root, bin, &diagnostic, options)
                              && diagnostic == "decoded glTF animation or deformation bytes exceed the admission limit",
                              "source validation storage cannot bypass a tight decode budget");
                        options.limits = {};
                        options.deformationLimits.maxVerticesPerMesh = 3;
                        check(animationRejected(root, bin, &diagnostic, options)
                              && diagnostic == "deformed mesh vertex capacity exceeded",
                              "unused source POSITION count remains bounded independently of mapped corners");
                    }
                }
    }
    {
        auto root = animatedRoot;
        auto bin = animatedBin;
        const auto indexOffset = bin.size();
        const std::vector<std::uint32_t> sources {0, 1, 2, 0, 2, 1};
        for (const auto index : sources) appendU16(bin, static_cast<std::uint16_t>(index));
        const auto view = root["bufferViews"].size();
        root["bufferViews"].push_back({{"buffer", 0}, {"byteOffset", indexOffset},
                                      {"byteLength", 12}, {"target", 34963}});
        const auto indices = root["accessors"].size();
        root["accessors"].push_back({{"bufferView", view}, {"componentType", 5123},
                                    {"count", 6}, {"type", "SCALAR"}});
        root["meshes"][0]["primitives"][0]["indices"] = indices;
        root["nodes"].push_back(nlohmann::json::object());
        root["scenes"] = nlohmann::json::array({{{"nodes", {0, 2}}}});
        root["scene"] = 0;
        root["skins"][0]["joints"] = {1, 2};
        root["skins"][0].erase("inverseBindMatrices");
        bin[36 + 4] = 1; // Vertex one belongs to the second authored joint.
        const float raisedCorner = 1;
        std::memcpy(bin.data() + 96 + 8 * sizeof(float), &raisedCorner, sizeof(float));
        root["buffers"][0]["byteLength"] = bin.size();
        std::string diagnostic;
        const auto decoded = videohelper::gltf::decodeGlbAnimations(makeGlb(root, bin), {}, diagnostic);
        GlbAdmissionOptions baseOptions;
        baseOptions.admitAnimations = true;
        baseOptions.admitSkins = true;
        const auto base = videohelper::gltf::decodeAnimatedGlbBaseScene(makeGlb(root, bin), baseOptions, diagnostic);
        check(decoded && decoded->deformation && base && diagnostic.empty(),
              "indexed missing-normal skin and morphs share base-scene and deformation admission");
        if (decoded && decoded->deformation && base)
        {
            const auto& mesh = decoded->deformation->meshes()[0];
            const auto& primitive = base->meshes[0].primitives[0];
            const auto& joints = mesh.jointWeightSets()[0].jointIndices();
            const auto& weights = mesh.jointWeightSets()[0].weights();
            const auto& positions = mesh.morphTargets()[0].positionDeltas();
            const auto& normals = mesh.morphTargets()[0].normalDeltas();
            check(mesh.vertexCount() == 6 && primitive.positions.size() == 18
                  && primitive.cornerSources == sources && joints.size() == 24 && weights.size() == 24
                  && positions.size() == 18 && normals.size() == 18
                  && joints[4] == 1 && joints[20] == 1 && joints[16] == 0
                  && weights[0] == 1 && weights[20] == 1
                  && positions[0] == 0.25f && positions[9] == 0.25f
                  && positions[8] == 1 && positions[14] == 1
                  && near(normals[1], -std::sqrt(0.5f))
                  && near(normals[10], std::sqrt(0.5f))
                  && near(normals[11], 1 - std::sqrt(0.5f))
                  && decoded->clips[0].morphBindings[0].mesh.value == 1
                  && decoded->renderBindings[0].nodeIndex == 0
                  && decoded->renderBindings[0].authoredJointCount == 2,
                  "joint, weight and morph arrays duplicate each original index corner and preserve animation identities");
        }
        videohelper::gltf::GlbAnimationDecodeOptions limited;
        // 128 inverse-bind bytes, 24 remap bytes, 192 joint/weight bytes,
        // 24 source-weight-sum bytes, 288 morph position/normal bytes and 48 animation bytes.
        limited.limits.maxDecodedBytes = 704;
        check(videohelper::gltf::decodeGlbAnimations(makeGlb(root, bin), limited, diagnostic).has_value(),
              "the exact expanded animation/deformation invoice remains admissible");
        limited.limits.maxDecodedBytes = 703;
        animationFactoryCalls = deformationFactoryCalls = 0;
        const auto bytes = makeGlb(root, bin);
        const auto overBudget = videohelper::gltf::detail::decodeGlbAnimationsWithFactories(
            bytes.data(), bytes.size(), limited, countingAnimationFactory,
            countingDeformationFactory, diagnostic);
        check(!overBudget && animationFactoryCalls == 0 && deformationFactoryCalls == 0
              && diagnostic == "decoded glTF animation or deformation bytes exceed the admission limit",
              "one-byte-short remapped influence/morph/animation budget refuses before immutable publication");
        limited = {}; limited.deformationLimits.maxVerticesPerMesh = 5;
        check(animationRejected(root, bin, &diagnostic, limited)
              && diagnostic == "deformed mesh vertex capacity exceeded",
              "deformation capacity charges six corners instead of three source vertices");
    }
    {
        auto root = animatedRoot;
        auto bin = animatedBin;
        auto& primitive = root["meshes"][0]["primitives"][0];
        primitive["attributes"]["TANGENT"] = 2; // Structurally valid FLOAT VEC4, ignored without NORMAL.
        primitive["targets"][0]["TANGENT"] = 4;
        const float raisedCorner = 1.0f;
        std::memcpy(bin.data() + 96 + 8 * sizeof(float), &raisedCorner, sizeof(float));
        std::string diagnostic;
        const auto decoded = videohelper::gltf::decodeGlbAnimations(makeGlb(root, bin), {}, diagnostic);
        check(decoded && decoded->deformation && diagnostic.empty(),
              "non-indexed missing-normal morphs enter the immutable deformation contract");
        if (decoded && decoded->deformation)
        {
            const auto& mesh = decoded->deformation->meshes()[0];
            const auto& delta = mesh.morphTargets()[0].normalDeltas();
            // Target positions are (.25,0,0), (1,0,0), (0,1,1).
            check(mesh.vertexCount() == 3 && !mesh.jointWeightSets().empty()
                  && mesh.jointWeightSets()[0].jointIndices().size() == 12
                  && mesh.jointWeightSets()[0].weights().size() == 12
                  && mesh.jointWeightSets()[0].weights()[0] == 1.0f
                  && delta.size() == 9 && near(delta[0], 0.0f)
                  && near(delta[1], -std::sqrt(0.5f))
                  && near(delta[2], std::sqrt(0.5f) - 1.0f)
                  && !mesh.morphTargets()[0].hasTangentDeltas(),
                  "generated morph normals encode target-minus-base before skinning and discard tangent deltas");
            bool sameCorners = delta.size() == 9;
            for (std::size_t corner = 1; sameCorners && corner < 3; ++corner)
                for (std::size_t axis = 0; axis < 3; ++axis)
                    sameCorners = sameCorners && delta[corner * 3 + axis] == delta[axis];
            check(sameCorners, "morph normal generation preserves the flat triangle and skin vertex correspondence");
        }
        videohelper::gltf::GlbAnimationDecodeOptions limited;
        limited.deformationLimits.maxMorphDeltaMagnitude = 0.5f;
        auto smallDelta = bin;
        const float displacement = 0.49f;
        std::memcpy(smallDelta.data() + 96 + 8 * sizeof(float), &displacement, sizeof(float));
        // Flip a very shallow base triangle using admitted small position deltas.
        const float baseHeight = 0.01f, negativeHeight = -0.02f;
        std::memcpy(smallDelta.data() + 7 * sizeof(float), &baseHeight, sizeof(float));
        std::memcpy(smallDelta.data() + 96 + 7 * sizeof(float), &negativeHeight, sizeof(float));
        check(animationRejected(root, smallDelta, &diagnostic, limited),
              "generated morph normal deltas retain the requested component magnitude bound");
        auto indexed = root;
        indexed["meshes"][0]["primitives"][0]["indices"] = 1;
        check(animationRejected(indexed, bin, &diagnostic)
              && diagnostic == "glTF index bufferView target must be ELEMENT_ARRAY_BUFFER",
              "indexed morphs still reject a vertex attribute masquerading as an index accessor");
        auto degenerate = bin;
        const float minusOne = -1.0f, zero = 0.0f;
        std::memcpy(degenerate.data() + 96 + 7 * sizeof(float), &minusOne, sizeof(float));
        std::memcpy(degenerate.data() + 96 + 8 * sizeof(float), &zero, sizeof(float));
        check(animationRejected(root, degenerate, &diagnostic)
              && diagnostic == "cannot generate flat glTF normals for a degenerate triangle",
              "degenerate morph target faces cannot publish fabricated normals");
        auto normalWithoutBase = root;
        normalWithoutBase["meshes"][0]["primitives"][0]["targets"][0]["NORMAL"] = 4;
        check(animationRejected(normalWithoutBase, bin, &diagnostic)
              && diagnostic == "glTF morph NORMAL requires a base NORMAL attribute",
              "morph NORMAL cannot refer to an absent base NORMAL attribute");

        limited = {};
        // One inverse bind is 64 bytes, joint/weight values 96 bytes and the
        // first morph's positions 36 bytes. Its generated normals need 36 more.
        limited.limits.maxDecodedBytes = 231;
        animationFactoryCalls = deformationFactoryCalls = 0;
        const auto bytes = makeGlb(root, bin);
        const auto overBudget = videohelper::gltf::detail::decodeGlbAnimationsWithFactories(
            bytes.data(), bytes.size(), limited, countingAnimationFactory,
            countingDeformationFactory, diagnostic);
        check(!overBudget && animationFactoryCalls == 0 && deformationFactoryCalls == 0
              && diagnostic == "decoded glTF animation or deformation bytes exceed the admission limit",
              "generated morph normal storage is admitted before allocation and immutable publication");

        auto mixedRoot = root;
        auto mixedBin = bin;
        const auto normalOffset = mixedBin.size();
        for (int corner = 0; corner < 3; ++corner)
            for (const auto component : {0.0f, 0.0f, 1.0f}) appendFloat(mixedBin, component);
        const auto normalView = mixedRoot["bufferViews"].size();
        mixedRoot["bufferViews"].push_back({{"buffer", 0}, {"byteOffset", normalOffset},
                                          {"byteLength", 36}, {"target", 34962}});
        const auto normalAccessor = mixedRoot["accessors"].size();
        mixedRoot["accessors"].push_back({{"bufferView", normalView}, {"componentType", 5126},
                                         {"count", 3}, {"type", "VEC3"}});
        mixedRoot["buffers"][0]["byteLength"] = mixedBin.size();
        auto authoredNormal = primitive;
        authoredNormal["attributes"]["NORMAL"] = normalAccessor;
        authoredNormal["attributes"].erase("TANGENT");
        authoredNormal["targets"][0].erase("TANGENT");
        for (const auto generatedFirst : {false, true})
        {
            mixedRoot["meshes"][0]["primitives"] = generatedFirst
                ? nlohmann::json::array({primitive, authoredNormal})
                : nlohmann::json::array({authoredNormal, primitive});
            const auto mixed = videohelper::gltf::decodeGlbAnimations(
                makeGlb(mixedRoot, mixedBin), {}, diagnostic);
            check(mixed && mixed->deformation->meshes()[0].vertexCount() == 6,
                  "mixed authored and generated normals retain concatenated primitive vertices");
            if (mixed)
            {
                const auto& delta = mixed->deformation->meshes()[0].morphTargets()[0].normalDeltas();
                const auto generatedOffset = generatedFirst ? 0u : 9u;
                const auto unchangedOffset = generatedFirst ? 9u : 0u;
                check(delta.size() == 18
                      && near(delta[generatedOffset + 1], -std::sqrt(0.5f))
                      && delta[unchangedOffset] == 0 && delta[unchangedOffset + 1] == 0
                      && delta[unchangedOffset + 2] == 0,
                      "generated morph normals and zero padding retain exact primitive offsets in both orders");
            }
        }
    }
    {
        auto root=animatedRoot;
        root["nodes"].push_back({{"children",{0}}, {"translation",{2.0,0.0,0.0}}});
        root["animations"][0]["channels"].push_back(
            {{"sampler",0},{"target",{{"node",2},{"path","translation"}}}});
        videohelper::gltf::GlbAnimationDecodeOptions options;
        options.retainGeometryHierarchy=true;
        const auto decoded=videohelper::gltf::decodeGlbAnimations(makeGlb(root,animatedBin),options,error);
        check(decoded && error.empty() && decoded->sceneMeshes.empty(),
              "scene-less animated GLB retains its hierarchy without inventing a scene");
        check(decoded && decoded->deformation->skins()[0].joints().size()==3
            && decoded->deformation->skins()[0].joints()[0].id().value==2
            && decoded->deformation->skins()[0].joints()[0].parent().value==1
            && decoded->deformation->skins()[0].joints()[1].parent().value==3,
            "Geometry3D skin decoding retains non-joint ancestors without reindexing vertex influences");
        check(decoded && decoded->clips[0].jointBindings.size()==2
            && decoded->renderBindings[0].jointBaseTransforms.size()==3,
            "Geometry3D parent animation and base transforms share the original node identities");
        auto matrixParent = root;
        matrixParent["nodes"].push_back({{"children", {2}}, {"matrix",
            {1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0, 3.0,0.0,0.0,1.0}}});
        const auto matrixDecoded = videohelper::gltf::decodeGlbAnimations(
            makeGlb(matrixParent, animatedBin), options, error);
        check(matrixDecoded && matrixDecoded->renderBindings.size() == 1
            && matrixDecoded->renderBindings[0].jointBaseTransforms.size() == 4
            && matrixDecoded->renderBindings[0].jointBaseTransforms.back().matrix
            && (*matrixDecoded->renderBindings[0].jointBaseTransforms.back().matrix)[12] == 3.0f,
            "static matrix ancestors retain their exact transforms in the native skin palette");
        root["nodes"][0].erase("skin");
        root.erase("skins");
        root["meshes"][0]["primitives"][0]["attributes"].erase("JOINTS_0");
        root["meshes"][0]["primitives"][0]["attributes"].erase("WEIGHTS_0");
        root["animations"][0]["channels"].erase(root["animations"][0]["channels"].begin());
        const auto rigid=videohelper::gltf::decodeGlbAnimations(makeGlb(root,animatedBin),options,error);
        check(rigid && rigid->deformation->meshes()[0].skin().isValid()
            && rigid->deformation->meshes()[0].jointWeightSets()[0].weights()[0]==1.0f
            && rigid->deformation->skins()[0].joints()[0].id().value==1,
            "Geometry3D retains rigid mesh parent tracks alongside named morph animation");
    }
    {
        error.clear();
        const auto decoded = videohelper::gltf::decodeGlbAnimations(
            makeGlb(animatedRoot, animatedBin), {}, error);
        check(decoded && error.empty() && decoded->metadata.animations == 1
              && decoded->clips.size() == 1 && decoded->clips[0].name == "Walk"
              && decoded->clips[0].clip && decoded->clips[0].clip->id().value == 1
              && decoded->clips[0].clip->tracks().size() == 2,
              "valid transform and morph animation decodes in source order with stable identities");
        check(decoded && decoded->clips[0].clip->tracks()[0].target().value == 2
              && decoded->clips[0].clip->tracks()[1].target().value == 1
              && decoded->clips[0].clip->tracks()[1].valueWidth() == 2
              && decoded->clips[0].jointBindings.size() == 1
              && decoded->clips[0].jointBindings[0].joint.value == 2
              && decoded->clips[0].morphBindings.size() == 1
              && decoded->clips[0].morphBindings[0].targets.size() == 2,
              "joint and morph animation bindings preserve deterministic node and target identities");
        check(decoded && decoded->deformation && decoded->deformation->skins().size() == 1
              && decoded->deformation->meshes().size() == 1
              && decoded->deformation->meshes()[0].vertexCount() == 3
              && decoded->deformation->meshes()[0].morphTargets().size() == 2,
              "valid skin weights, inverse binds, and morph deltas create an immutable deformation asset");
        check(decodeRejected(animatedRoot, animatedBin),
              "the static decoder remains fail-closed for a valid animated and skinned GLB");

        const auto complete = makeGlb(animatedRoot, animatedBin);
        bool everyTruncationRejected = true;
        for (std::size_t prefix = 0; prefix < complete.size(); ++prefix)
        {
            error.clear();
            const auto truncated = videohelper::gltf::decodeGlbAnimations(
                complete.data(), prefix, {}, error);
            everyTruncationRejected = everyTruncationRejected
                                   && !truncated && !error.empty();
        }
        check(everyTruncationRejected,
              "every truncated animated GLB prefix fails closed without partial decode");
    }
    {
        auto root = animatedRoot;
        root["animations"][0]["channels"].push_back(
            {{"sampler", 0}, {"target", {{"node", 1}, {"path", "scale"}}}});
        const auto decoded = videohelper::gltf::decodeGlbAnimations(
            makeGlb(root, animatedBin), {}, error);
        check(decoded && decoded->clips.size() == 1
              && decoded->clips[0].clip->tracks().size() == 3
              && decoded->clips[0].jointBindings.size() == 1,
              "multiple transform channels share one stable joint binding");

        root = animatedRoot;
        root["skins"].push_back(root["skins"][0]);
        check(animationRejected(root, animatedBin),
              "an animated joint shared by multiple skins is rejected by the bounded binding model");
        {
            auto options = videohelper::nativeImportedAnimationDecodeOptions(animatedBin.size());
            options.admission.limits.maxContainerBytes = 512u * 1024u * 1024u;
            const auto shared = videohelper::gltf::decodeGlbAnimations(makeGlb(root, animatedBin), options, error);
            check(shared && shared->clips[0].jointBindings.size() == 2,
                  "native scene animation retains one sampled node binding for each skin ancestor chain");
        }

        root = animatedRoot;
        root["nodes"].push_back({{"mesh", 0}, {"skin", 0}});
        root["animations"][0]["channels"].push_back(
            {{"sampler", 1}, {"target", {{"node", 2}, {"path", "weights"}}}});
        check(animationRejected(root, animatedBin),
              "two animated instances cannot alias one deformation-mesh binding");

        root = animatedRoot;
        root["skins"][0]["joints"] = nlohmann::json::array({0});
        root["animations"][0]["channels"].push_back(
            {{"sampler", 0}, {"target", {{"node", 0}, {"path", "translation"}}}});
        check(animationRejected(root, animatedBin),
              "one animation target cannot own both joint and morph deformation bindings");
    }
    {
        auto root = animatedRoot;
        root["animations"][0]["samplers"][0]["interpolation"] = "CUBICSPLINE";
        check(animationRejected(root, animatedBin),
              "unsupported cubic animation interpolation is rejected");
        root = animatedRoot;
        root["animations"][0]["channels"].push_back(root["animations"][0]["channels"][0]);
        check(animationRejected(root, animatedBin),
              "duplicate animation target-channel tracks are rejected");
        root = animatedRoot;
        root["accessors"][6]["sparse"] = nlohmann::json::object();
        check(animationRejected(root, animatedBin),
              "sparse animation accessors remain outside the bounded subset");
        root = animatedRoot;
        root["buffers"][0]["uri"] = "outside.bin";
        check(animationRejected(root, animatedBin),
              "animation decode rejects external buffers without opening them");
        root = animatedRoot;
        root["bufferViews"][7]["target"] = 34962;
        check(animationRejected(root, animatedBin),
              "animation outputs cannot reuse a vertex-targeted bufferView");
        root = animatedRoot;
        root["accessors"][8]["count"] = 3;
        check(animationRejected(root, animatedBin),
              "morph animation output cardinality must equal keys times targets");
        check(animationRejected(animatedRoot, animatedDeformationBin(true)),
              "animation input times must be strictly increasing");
        root = animatedRoot;
        root["nodes"][1]["matrix"] = nlohmann::json::array(
            {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0});
        check(animationRejected(root, animatedBin),
              "TRS animation cannot target a matrix-authored node");
    }
    {
        auto badJoints = animatedBin;
        badJoints[36] = 1;
        check(animationRejected(animatedRoot, badJoints),
              "joint indices outside the original authored skin are rejected before immutable publication");
        auto nonfinite = animatedBin;
        const float infinity = std::numeric_limits<float>::infinity();
        std::memcpy(nonfinite.data() + 264, &infinity, sizeof(infinity));
        check(animationRejected(animatedRoot, nonfinite),
              "nonfinite morph animation values are rejected");
        auto rotationRoot = animatedRoot;
        auto rotationBin = animatedBin;
        for (const auto value : {0.0f, 0.0f, 0.0f, 1.0f,
                                 0.0f, 0.0f, 1.0f, 0.0f})
            appendFloat(rotationBin, value);
        rotationRoot["buffers"][0]["byteLength"] = rotationBin.size();
        rotationRoot["bufferViews"].push_back(
            {{"buffer", 0}, {"byteOffset", 280}, {"byteLength", 32}});
        rotationRoot["accessors"].push_back(
            {{"bufferView", 9}, {"componentType", 5126}, {"count", 2}, {"type", "VEC4"}});
        rotationRoot["animations"][0]["samplers"][0]["output"] = 9;
        rotationRoot["animations"][0]["channels"][0]["target"]["path"] = "rotation";
        error.clear();
        const auto rotationDocument = videohelper::gltf::decodeGlbAnimations(
            makeGlb(rotationRoot, rotationBin), {}, error);
        const auto boundarySample = rotationDocument
            ? visualanimation::sample(*rotationDocument->clips[0].clip,
                                      {1.0, visualanimation::Playback::Clamp}, error)
            : std::nullopt;
        check(boundarySample && boundarySample->tracks()[0].values()
                                  == std::vector<float>({0.0f, 0.0f, 1.0f, 0.0f}),
              "normalized GLB rotation keys reach the production sampler unchanged at a boundary");

        error.clear();
        const auto midpointSample = rotationDocument
            ? visualanimation::sample(*rotationDocument->clips[0].clip,
                                      {0.5, visualanimation::Playback::Clamp}, error)
            : std::nullopt;
        const auto midpointValues = midpointSample
            ? midpointSample->tracks()[0].values() : std::vector<float>{};
        check(midpointValues.size() == 4 && near(midpointValues[0], 0.0f)
                  && near(midpointValues[1], 0.0f)
                  && near(midpointValues[2], std::sqrt(0.5f))
                  && near(midpointValues[3], std::sqrt(0.5f)),
              "normalized GLB rotation keys use normalized production interpolation");

        auto nonNormalizedRotation = rotationBin;
        const float doubledComponent = 2.0f;
        std::memcpy(nonNormalizedRotation.data() + 308, &doubledComponent,
                    sizeof(doubledComponent));
        std::string diagnostic;
        check(animationRejected(rotationRoot, nonNormalizedRotation, &diagnostic)
                  && diagnostic
                       == "glTF animation rotation output contains a non-normalized quaternion",
              "non-normalized animation rotation keys are rejected by the production decoder");

        bool everyQuaternionComponentRejected = true;
        for (std::size_t key = 0; key < 2; ++key)
            for (std::size_t component = 0; component < 4; ++component)
            {
                auto invalid = rotationBin;
                std::memcpy(invalid.data() + 280 + (key * 4 + component) * sizeof(float),
                            &doubledComponent, sizeof(doubledComponent));
                everyQuaternionComponentRejected = everyQuaternionComponentRejected
                    && animationRejected(rotationRoot, invalid);
            }
        check(everyQuaternionComponentRejected,
              "every component of every GLB rotation key is validated");

        auto invalidRotation = rotationBin;
        const float zeroQuaternion[4]{};
        std::memcpy(invalidRotation.data() + 280, zeroQuaternion, sizeof(zeroQuaternion));
        check(animationRejectedBeforeFactory(rotationRoot, invalidRotation),
              "zero GLB rotation keys are rejected before contract allocation");
        const float nan = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(invalidRotation.data() + 280, &nan, sizeof(nan));
        check(animationRejectedBeforeFactory(rotationRoot, invalidRotation),
              "NaN GLB rotation components are rejected before contract allocation");
        const float rotationInfinity = std::numeric_limits<float>::infinity();
        std::memcpy(invalidRotation.data() + 280, &rotationInfinity,
                    sizeof(rotationInfinity));
        check(animationRejectedBeforeFactory(rotationRoot, invalidRotation),
              "infinite GLB rotation components are rejected before contract allocation");

        const auto setFirstQuaternionW = [&] (float value)
        {
            auto bin = rotationBin;
            std::memcpy(bin.data() + 292, &value, sizeof(value));
            return bin;
        };
        check(!animationRejected(rotationRoot, setFirstQuaternionW(std::sqrt(1.0009f))),
              "GLB rotation tolerance accepts squared-norm error below 0.001");
        check(animationRejectedBeforeFactory(rotationRoot,
                                             setFirstQuaternionW(std::sqrt(1.0011f))),
              "GLB rotation tolerance rejects squared-norm error above 0.001 before allocation");
        videohelper::gltf::GlbAnimationDecodeOptions limited;
        limited.limits.maxDecodedBytes = 31;
        check(animationRejected(animatedRoot, animatedBin, nullptr, limited),
              "the aggregate animation and deformation decoded-byte budget is enforced");
        limited = {};
        limited.deformationLimits.maxJointsPerSkin = 0;
        check(animationRejected(animatedRoot, animatedBin, nullptr, limited),
              "caller skin joint limits are enforced before contract construction");
        limited = {};
        limited.deformationLimits.maxMorphTargetsPerMesh = 1;
        check(animationRejected(animatedRoot, animatedBin, nullptr, limited),
              "caller morph-target limits are enforced before allocation");
    }
    {
        // Both actions address the same joint and two morph targets. The second
        // lasts twice as long, so Beat Sync must map each duration independently.
        auto root = animatedRoot;
        auto bin = animatedBin;
        root["animations"][0]["samplers"][1]["interpolation"] = "LINEAR";
        for (const auto value : {4.0f,5.0f,6.0f, 2.0f,3.0f,4.0f,
                                 0.8f,0.2f,0.2f,0.8f, 0.0f,2.0f}) appendFloat(bin, value);
        root["buffers"][0]["byteLength"] = bin.size();
        for (const auto& view : {nlohmann::json{{"buffer",0},{"byteOffset",280},{"byteLength",24}},
                               nlohmann::json{{"buffer",0},{"byteOffset",304},{"byteLength",16}},
                               nlohmann::json{{"buffer",0},{"byteOffset",320},{"byteLength",8}}})
            root["bufferViews"].push_back(view);
        root["accessors"].push_back({{"bufferView",9},{"componentType",5126},{"count",2},{"type","VEC3"}});
        root["accessors"].push_back({{"bufferView",10},{"componentType",5126},{"count",4},{"type","SCALAR"}});
        root["accessors"].push_back({{"bufferView",11},{"componentType",5126},{"count",2},{"type","SCALAR"}});
        auto second = root["animations"][0];
        second["name"] = "Reach";
        second["samplers"][0]["input"] = second["samplers"][1]["input"] = 11;
        second["samplers"][0]["output"] = 9; second["samplers"][1]["output"] = 10;
        root["animations"].push_back(second);
        const auto bytes = makeGlb(root, bin);
        videohelper::Sha256 hash; hash.update(bytes.data(), bytes.size());
        visualanimationimport::Request request;
        request.sourceStableId=41; request.deformationStableId=42; request.schedule={41,42};
        request.asset={"crossfade-joint-morph",1,hash.finishHex(),"model/gltf-binary",bytes.size()};
        request.animationClipStableId=1; request.clipName="Walk"; request.meshStableId=1;
        request.crossfadeClipStableId=2; request.crossfadeClipName="Reach";
        request.playback.timelineSeconds=0.5;
        videohelper::ImportedAnimationDeformationConsumer consumer;
        check(consumer.admit(request,bytes.data(),bytes.size(),error),
              "same-target two-action GLB is admitted for actual imported crossfade");
        for (const auto progress : {0.0,0.5,1.0})
        {
            request.crossfadeProgress=progress;
            videohelper::ImportedAnimationDeformationEvaluation preview, exported;
            const bool evaluated=consumer.evaluatePreview(request,{12,24,1},9,preview,error)
                && consumer.evaluateExport(request,{12,24,1},9,exported,error);
            check(evaluated && preview.deformation && exported.deformation,
                  "two imported actions evaluate at endpoints and midpoint through both owners");
            if (!evaluated || !preview.deformation || !exported.deformation) continue;
            const auto& joints=preview.deformation->jointTransforms();
            const auto& morphs=preview.deformation->morphWeights();
            const std::array<float,3> expected {float(0.5+3*progress),float(1+3.5*progress),float(1.5+4*progress)};
            check(joints.size()==1 && joints[0].translation()==expected
                && morphs.size()==1 && near(morphs[0].values()[0],float(0.25+0.4*progress))
                && near(morphs[0].values()[1],float(0.75-0.4*progress)),
                "imported crossfade joint and morph channels match independent interpolation math");
            if (joints.size()!=1 || morphs.size()!=1
                || exported.deformation->jointTransforms().size()!=1 || exported.deformation->morphWeights().size()!=1) continue;
            check(joints[0].translation()==exported.deformation->jointTransforms()[0].translation()
                && morphs[0].values()==exported.deformation->morphWeights()[0].values()
                && preview.deformation->time()==exported.deformation->time(),
                "crossfade preview/export retain identical poses and exact rational frame identity");
            visualanimationimport::Request reopened;
            check(visualanimationoperation::decode(visualanimationoperation::encode(request),reopened,error)
                && reopened.crossfadeClipStableId==2 && reopened.crossfadeClipName=="Reach"
                && reopened.crossfadeProgress==progress,
                "v8 operation retains exact paired selectors and authored crossfade amount");
        }
        request.crossfadeProgress=0.5;
        request.playback.timeSource=visualanimation::TimeSource::BeatSync;
        request.playback.timelineBeat=2; request.playback.beatsPerLoop=4;
        videohelper::ImportedAnimationDeformationEvaluation beat;
        check(consumer.evaluatePreview(request,{12,24,1},9,beat,error) && beat.deformation
            && beat.deformation->jointTransforms().size()==1 && beat.deformation->morphWeights().size()==1
            && beat.deformation->jointTransforms()[0].translation()==std::array<float,3>{1.75f,2.5f,3.25f}
            && near(beat.deformation->morphWeights()[0].values()[0],0.375f),
            "Beat Sync resolves each action duration before crossfading joint and morph channels");
        request.crossfadeClipName="Wrong Reach";
        videohelper::ImportedAnimationDeformationEvaluation unchanged; unchanged.sourceStableId=999;
        check(!consumer.evaluatePreview(request,{12,24,1},9,unchanged,error) && unchanged.sourceStableId==999,
              "second clip identity/name disagreement refuses output publication");
        request.crossfadeClipName="Reach";
        request.crossfadeProgress=std::numeric_limits<double>::quiet_NaN();
        check(!consumer.evaluateExport(request,{12,24,1},9,unchanged,error),
              "nonfinite crossfade progress is rejected before sampling");
        request.crossfadeProgress=0.5;
        auto incompatible=root;
        incompatible["animations"][1]["channels"].erase(1);
        const auto incompatibleBytes=makeGlb(incompatible,bin);
        videohelper::Sha256 incompatibleHash; incompatibleHash.update(incompatibleBytes.data(),incompatibleBytes.size());
        request.asset.contentSha256=incompatibleHash.finishHex(); request.asset.sourceByteSize=incompatibleBytes.size();
        check(!consumer.admit(request,incompatibleBytes.data(),incompatibleBytes.size(),error)
            && error=="animation crossfade clips have incompatible channel sets",
            "imported crossfade refuses actions with unequal joint/morph channel topology");
    }
    {
        const auto bytes = makeGlb(animatedRoot, animatedBin);
        videohelper::Sha256 hash;
        hash.update(bytes.data(), bytes.size());

        visualanimationimport::Request request;
        request.sourceStableId = 41;
        request.deformationStableId = 42;
        request.schedule = {request.sourceStableId, request.deformationStableId};
        request.asset = {"animated-suzanne", 7, hash.finishHex(),
                         "model/gltf-binary", bytes.size()};
        request.animationClipStableId = 1;
        request.clipName = "Walk";
        request.playback.timelineSeconds = 0.25;
        request.playback.speed = 2.0;
        request.playback.trimStartSeconds = 0.25;
        request.playback.trimEndSeconds = 0.75;
        request.playback.weight = 0.75;

        videohelper::ImportedAnimationDeformationConsumer consumer;
        error.clear();
        check(consumer.admit(request, bytes.data(), bytes.size(), error)
                  && error.empty(),
              "the helper admits graph-lowered exact-content animation bytes");

        auto requiredExtensionRoot = animatedRoot;
        requiredExtensionRoot["nodes"][1]["extensions"] = {
            {"KHR_lights_punctual", {{"light", 0}}}
        };
        requiredExtensionRoot["extensionsUsed"]
            = nlohmann::json::array({"KHR_lights_punctual"});
        requiredExtensionRoot["extensionsRequired"]
            = nlohmann::json::array({"KHR_lights_punctual"});
        requiredExtensionRoot["extensions"]["KHR_lights_punctual"]["lights"]
            = nlohmann::json::array({{{"type", "directional"}}});
        const auto requiredExtensionBytes
            = makeGlb(requiredExtensionRoot, animatedBin);
        videohelper::Sha256 requiredExtensionHash;
        requiredExtensionHash.update(
            requiredExtensionBytes.data(), requiredExtensionBytes.size());
        auto requiredExtensionRequest = request;
        requiredExtensionRequest.asset.contentSha256
            = requiredExtensionHash.finishHex();
        requiredExtensionRequest.asset.sourceByteSize
            = requiredExtensionBytes.size();
        videohelper::ImportedAnimationDeformationConsumer requiredExtensionConsumer;
        error.clear();
        check(requiredExtensionConsumer.admit(
                  requiredExtensionRequest,
                  requiredExtensionBytes.data(), requiredExtensionBytes.size(), error)
                  && error.empty(),
              "runtime animation admission accepts the same required punctual-light extension as preflight");

        videohelper::ImportedAnimationDeformationEvaluation preview;
        videohelper::ImportedAnimationDeformationEvaluation exportEvaluation;
        const visualdeformation::RationalFrameTime frame {15, 30, 1};
        check(consumer.evaluatePreview(request, frame, 77, preview, error)
                  && consumer.evaluateExport(request, frame, 77, exportEvaluation, error)
                  && preview.owner
                         == videohelper::ImportedAnimationEvaluationOwner::Preview
                  && exportEvaluation.owner
                         == videohelper::ImportedAnimationEvaluationOwner::Export
                  && preview.sourceStableId == request.sourceStableId
                  && preview.deformationStableId == request.deformationStableId
                  && near(static_cast<float>(preview.playback.sample.timeSeconds), 0.75f)
                  && near(static_cast<float>(preview.playback.weight), 0.75f)
                  && preview.deformation && exportEvaluation.deformation
                  && preview.deformation->time() == frame
                  && preview.deformation->revision() == 77
                  && near(static_cast<float>(preview.deformation->requestedTimeSeconds()), 0.75f)
                  && near(static_cast<float>(preview.deformation->sampleTimeSeconds()), 0.75f)
                  && preview.deformation->jointTransforms().size() == 1
                  && exportEvaluation.deformation->jointTransforms().size() == 1
                  && preview.deformation->jointTransforms()[0].translation()
                         == exportEvaluation.deformation->jointTransforms()[0].translation()
                  && preview.deformation->morphWeights().size() == 1
                  && exportEvaluation.deformation->morphWeights().size() == 1
                  && preview.deformation->morphWeights()[0].values()
                         == exportEvaluation.deformation->morphWeights()[0].values(),
              "preview and export consume one playback mapping and deformation snapshot contract");

        auto controlled = request;
        controlled.meshStableId = 1;
        controlled.pose.meshStableId = 1;
        controlled.pose.boneEnabled = true;
        controlled.pose.boneStableId = 2;
        controlled.pose.translation[1] = 0.5;
        controlled.pose.morphEnabled = true;
        controlled.pose.morphTargetStableId = 2;
        controlled.pose.morphTargetIndex = 1;
        controlled.pose.morphWeight = 0.8;
        check(consumer.evaluatePreview(controlled, frame, 77, preview, error)
                  && consumer.evaluateExport(controlled, frame, 77, exportEvaluation, error)
                  && preview.deformation->pose().boneStableId == 2
                  && preview.deformation->pose().translation[1] == 0.5
                  && preview.deformation->pose().morphTargetIndex == 1
                  && preview.deformation->pose().morphWeight == 0.8
                  && exportEvaluation.deformation->pose().morphWeight == 0.8
                  && preview.deformation->time() == exportEvaluation.deformation->time(),
              "selected GLB bone and second named morph retain exact controls for preview and export");
        auto missingBone = controlled;
        auto legacyPose = controlled;
        legacyPose.pose.morphTargetIndex = visualanimation::kUnresolvedMorphTargetIndex;
        check(consumer.evaluatePreview(legacyPose, frame, 77, preview, error)
                  && preview.deformation->pose().morphTargetIndex == 1,
              "an older named morph selector resolves its exact mesh-local index during admission");
        missingBone.pose.boneStableId = 1;
        check(!consumer.evaluatePreview(missingBone, frame, 77, preview, error)
                  && error.find("selected bone") != std::string::npos,
              "the selected bone must belong to the exact GLB mesh skin");
        auto mismatchedMorph = controlled;
        mismatchedMorph.pose.morphTargetStableId = 1;
        check(!consumer.evaluateExport(mismatchedMorph, frame, 77, exportEvaluation, error)
                  && error.find("catalog identity") != std::string::npos,
              "a named morph selection cannot silently address another target index");

        auto mismatched = request;
        mismatched.asset.contentSha256[0] = mismatched.asset.contentSha256[0] == '0' ? '1' : '0';
        videohelper::ImportedAnimationDeformationEvaluation unchanged;
        unchanged.sourceStableId = 999;
        check(! consumer.evaluatePreview(mismatched, frame, 77, unchanged, error)
                  && error == "imported animation/deformation exact content is not admitted"
                  && unchanged.sourceStableId == 999,
              "evaluation rejects an unadmitted fingerprint without mutating output");

        auto unavailableSelector = request;
        unavailableSelector.animationClipStableId = 2;
        videohelper::ImportedAnimationDeformationConsumer rejectedSelector;
        error.clear();
        check(! rejectedSelector.admit(
                  unavailableSelector, bytes.data(), bytes.size(), error)
                  && error == "imported animation/deformation clip selector is unavailable",
              "stable animation selectors reject an unavailable decoded clip identity");
        auto unavailableMesh = request;
        unavailableMesh.meshStableId = 2;
        error.clear();
        check(! rejectedSelector.admit(
                  unavailableMesh, bytes.data(), bytes.size(), error)
                  && error == "imported animation/deformation mesh selector is unavailable",
              "stable mesh selectors reject an unavailable decoded mesh identity");
        auto mismatchedSelectorName = request;
        mismatchedSelectorName.clipName = "Not Walk";
        error.clear();
        check(! rejectedSelector.admit(
                  mismatchedSelectorName, bytes.data(), bytes.size(), error)
                  && error
                       == "imported animation/deformation clip selector and name disagree",
              "stable animation selectors reject stale display-name metadata");

        auto unnamedRoot = animatedRoot;
        unnamedRoot["animations"][0].erase("name");
        const auto unnamedBytes = makeGlb(unnamedRoot, animatedBin);
        videohelper::Sha256 unnamedHash;
        unnamedHash.update(unnamedBytes.data(), unnamedBytes.size());
        auto unnamedClip = request;
        unnamedClip.asset.contentSha256 = unnamedHash.finishHex();
        unnamedClip.asset.sourceByteSize = unnamedBytes.size();
        unnamedClip.clipName.clear();
        error.clear();
        check(rejectedSelector.admit(
                  unnamedClip, unnamedBytes.data(), unnamedBytes.size(), error)
                  && error.empty(),
              "stable animation selectors admit an unnamed clip without synthesizing a name");

        auto repeatedRoot = animatedRoot;
        repeatedRoot["nodes"].push_back({{"mesh", 0}, {"skin", 0}, {"name", "Repeated"}});
        repeatedRoot["nodes"][0]["name"] = "Repeated";
        repeatedRoot["scenes"][0]["nodes"].push_back(2);
        auto repeatedMorph = repeatedRoot["animations"][0]["channels"][1];
        repeatedMorph["target"]["node"] = 2;
        repeatedRoot["animations"][0]["channels"].push_back(repeatedMorph);
        repeatedRoot["animations"].push_back(repeatedRoot["animations"][0]);
        const auto repeatedBytes = makeGlb(repeatedRoot, animatedBin);
        videohelper::Sha256 repeatedHash;
        repeatedHash.update(repeatedBytes.data(), repeatedBytes.size());
        auto repeatedRequest = controlled;
        repeatedRequest.asset.contentSha256 = repeatedHash.finishHex();
        repeatedRequest.asset.sourceByteSize = repeatedBytes.size();
        repeatedRequest.meshStableId = 0;
        repeatedRequest.pose.nodeStableId = 3;
        const auto repeatedDecoded = videohelper::gltf::decodeGlbAnimations(
            repeatedBytes, videohelper::nativeImportedAnimationDecodeOptions(repeatedBytes.size()), error);
        check(repeatedDecoded && repeatedDecoded->renderBindings.size() == 2
            && repeatedDecoded->renderBindings[0].mesh == repeatedDecoded->renderBindings[1].mesh
            && repeatedDecoded->renderBindings[0].skin == repeatedDecoded->renderBindings[1].skin
            && repeatedDecoded->renderBindings[0].nodeIndex == 0
            && repeatedDecoded->renderBindings[1].nodeIndex == 2,
            "repeated animated nodes retain distinct bindings to the shared authored mesh and skin");
        videohelper::ImportedAnimationDeformationConsumer repeatedConsumer;
        check(repeatedConsumer.admit(repeatedRequest, repeatedBytes.data(), repeatedBytes.size(), error)
            && repeatedConsumer.evaluatePreview(repeatedRequest, frame, 77, preview, error)
            && repeatedConsumer.evaluateExport(repeatedRequest, frame, 77, exportEvaluation, error)
            && preview.deformation->morphWeights().size() == 2
            && preview.deformation->morphWeights()[0].mesh() == preview.deformation->morphWeights()[1].mesh()
            && preview.deformation->morphWeights()[0].animationTarget()
                != preview.deformation->morphWeights()[1].animationTarget()
            && preview.deformation->pose().nodeStableId == 3
            && exportEvaluation.deformation->pose().nodeStableId == 3,
            "Whole scene evaluates repeated mesh morph tracks and selected shared-skin pose by exact node");
        auto ambiguousPose = repeatedRequest;
        ambiguousPose.meshStableId = 1;
        ambiguousPose.pose.nodeStableId = 0;
        check(!repeatedConsumer.evaluatePreview(ambiguousPose, frame, 77, preview, error)
            && error.find("select an exact object node ID") != std::string::npos,
            "mesh-only pose selection fails closed when repeated nodes make it ambiguous");
        auto ambiguousClip = repeatedRequest;
        ambiguousClip.animationClipStableId = 0;
        check(!repeatedConsumer.evaluatePreview(ambiguousClip, frame, 77, preview, error)
            && error == "imported animation/deformation named clip is ambiguous",
            "duplicate clip names require a stable clip ID");
        repeatedRequest.animationClipStableId = 2;
        check(repeatedConsumer.evaluatePreview(repeatedRequest, frame, 77, preview, error),
            "an exact stable clip ID disambiguates duplicate names without renaming asset content");

        auto multiMeshRoot = animatedRoot;
        multiMeshRoot["meshes"].push_back(multiMeshRoot["meshes"][0]);
        auto& alternateAttributes
            = multiMeshRoot["meshes"][1]["primitives"][0]["attributes"];
        alternateAttributes.erase("JOINTS_0");
        alternateAttributes.erase("WEIGHTS_0");
        multiMeshRoot["nodes"].push_back({ { "mesh", 1 } });
        const auto multiMeshBytes = makeGlb(multiMeshRoot, animatedBin);
        videohelper::Sha256 multiMeshHash;
        multiMeshHash.update(multiMeshBytes.data(), multiMeshBytes.size());
        auto incompatibleMesh = request;
        incompatibleMesh.asset.contentSha256 = multiMeshHash.finishHex();
        incompatibleMesh.asset.sourceByteSize = multiMeshBytes.size();
        incompatibleMesh.meshStableId = 2;
        error.clear();
        const auto incompatibleAdmitted = rejectedSelector.admit(
            incompatibleMesh, multiMeshBytes.data(), multiMeshBytes.size(), error);
        if (incompatibleAdmitted
            || error != "imported animation/deformation clip does not drive selected mesh")
            std::fprintf(stderr, "incompatible selected mesh: admitted=%d error=%s\n",
                         incompatibleAdmitted ? 1 : 0, error.c_str());
        check(! incompatibleAdmitted
                  && error == "imported animation/deformation clip does not drive selected mesh",
              "stable selectors reject a renderable mesh that the selected clip does not drive");
    }
    {
        videohelper::Sha256 hash;
        hash.update(visualstartermodel::kAnimatedTriangleGlb.data(),
                    visualstartermodel::kAnimatedTriangleGlb.size());
        check(visualstartermodel::kAnimatedTriangleGlb.size() == 3576
                  && visualstartermodel::kAnimatedTriangleGlb.size()
                         <= visualstartermodel::kMaximumEmbeddedBytes
                  && hash.finishHex() == visualstartermodel::kContentSha256,
              "built-in model bytes retain their exact size, bound, and digest");
        check(visualstartermodel::kLicenseSpdx == "CC0-1.0"
                  && visualstartermodel::kBlenderVersion == "5.2.0 LTS"
                  && visualstartermodel::kExporterGenerator
                         == "Khronos glTF Blender I/O v5.2.39"
                  && visualstartermodel::kProvenance.find(
                         visualstartermodel::kSourceScript) != std::string_view::npos,
              "built-in model bytes retain Blender source and license provenance");

        GlbAdmissionOptions admission;
        admission.admitAnimations = true;
        admission.admitSkins = true;
        admission.supportedRequiredExtensions = {"KHR_lights_punctual"};
        error.clear();
        const auto metadata = videohelper::gltf::admitGlbMetadata(
            visualstartermodel::kAnimatedTriangleGlb.data(),
            visualstartermodel::kAnimatedTriangleGlb.size(), admission, error);
        check(metadata && error.empty() && metadata->scenes == 1
                  && metadata->meshes == 1 && metadata->primitives == 1
                  && metadata->materials == 1 && metadata->textures == 1
                  && metadata->images == 1 && metadata->animations == 1
                  && metadata->cameras == 1,
              "real GLB metadata admission accepts the built-in model sample");

        error.clear();
        const auto scene = videohelper::gltf::decodeAnimatedGlbBaseScene(
            visualstartermodel::kAnimatedTriangleGlb.data(),
            visualstartermodel::kAnimatedTriangleGlb.size(), admission, error);
        if (!scene)
            std::fprintf(stderr, "built-in base-scene decode: %s\n", error.c_str());
        check(scene && error.empty() && scene->decodedVertexCount == 3
                  && scene->decodedIndexCount == 3 && scene->meshes.size() == 1
                  && scene->materials.size() == 1 && scene->cameras.size() == 1
                  && scene->lights.size() == 1 && scene->textures.size() == 1
                  && scene->images.size() == 1
                  && scene->materials[0].baseColorTexture
                  && scene->materials[0].baseColorTexture->texture == 0
                  && near(scene->materials[0].baseColorFactor[0], 1.0f)
                  && near(scene->materials[0].baseColorFactor[1], 1.0f)
                  && near(scene->materials[0].baseColorFactor[2], 1.0f)
                  && near(scene->materials[0].roughnessFactor, 0.55f),
              "real base-scene decoder accepts the built-in model sample");

        videohelper::gltf::GlbAnimationDecodeOptions animationOptions;
        animationOptions.admission = admission;
        error.clear();
        const auto animation = videohelper::gltf::decodeGlbAnimations(
            visualstartermodel::kAnimatedTriangleGlb.data(),
            visualstartermodel::kAnimatedTriangleGlb.size(), animationOptions, error);
        if (!animation)
            std::fprintf(stderr, "built-in animation decode: %s\n", error.c_str());
        check(animation && error.empty() && animation->clips.size() == 1
                  && animation->clips[0].name == visualstartermodel::kClipName
                  && animation->clips[0].clip
                  && animation->clips[0].clip->tracks().size() == 1
                  && animation->clips[0].morphBindings.size() == 1
                  && animation->clips[0].morphBindings[0].targets.size() == 1
                  && animation->deformation
                  && animation->deformation->meshes().size() == 1
                  && animation->deformation->meshes()[0].morphTargets().size() == 1,
              "real animation/deformation decoder accepts the built-in model sample");

        auto malformed = visualstartermodel::kAnimatedTriangleGlb;
        malformed[0] ^= 0xff;
        error.clear();
        check(!videohelper::gltf::admitGlbMetadata(
                  malformed.data(), malformed.size(), admission, error)
                  && !error.empty(),
              "malformed built-in model bytes passed real GLB admission");
    }

    std::fprintf(stderr, failures ? "%d GLB parser checks failed\n"
                                  : "GLB admission and static decode checks passed\n",
                 failures);
    return failures == 0 ? 0 : 1;
}
