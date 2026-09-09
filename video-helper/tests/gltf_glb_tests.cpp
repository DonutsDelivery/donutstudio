#include "../src/gltf_glb.h"
#include "../src/imported_animation_deformation_consumer.h"
#include "../src/sha256.h"
#include "../../shared/VisualStarterModelAssets.h"
#include "../../plugin/Tests/fixtures/visual-model/StaticVisualModelWorkflowFixture.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
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
    const auto result = videohelper::gltf::decodeStaticGlb(makeGlb(root, bin), options, error);
    if (diagnostic != nullptr)
        *diagnostic = error;
    return !result.has_value() && !error.empty();
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
        check(decodeRejected(root, meshBin),
              "non-indexed triangle primitives are outside the bounded static subset");
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
              "joint indices outside the bound skin are rejected by the immutable contract");
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
