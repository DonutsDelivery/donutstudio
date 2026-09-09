#include "gltf_glb.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <utility>

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

namespace videohelper::gltf
{
namespace
{
using Json = nlohmann::json;

constexpr std::uint32_t glbMagic = 0x46546c67u;
constexpr std::uint32_t jsonChunkType = 0x4e4f534au;
constexpr std::uint32_t binChunkType = 0x004e4942u;

std::uint32_t readU32(const std::uint8_t* bytes)
{
    return static_cast<std::uint32_t>(bytes[0])
         | (static_cast<std::uint32_t>(bytes[1]) << 8u)
         | (static_cast<std::uint32_t>(bytes[2]) << 16u)
         | (static_cast<std::uint32_t>(bytes[3]) << 24u);
}

bool fail(std::string& error, std::string message)
{
    error = std::move(message);
    return false;
}

bool checkJsonNesting(const std::uint8_t* bytes,
                      std::size_t size,
                      std::size_t maximum,
                      std::string& error)
{
    std::size_t depth = 0;
    bool inString = false;
    bool escaped = false;
    for (std::size_t i = 0; i < size; ++i)
    {
        const auto c = static_cast<char>(bytes[i]);
        if (inString)
        {
            if (escaped)
                escaped = false;
            else if (c == '\\')
                escaped = true;
            else if (c == '"')
                inString = false;
            continue;
        }
        if (c == '"')
        {
            inString = true;
            continue;
        }
        if (c == '{' || c == '[')
        {
            if (depth == maximum)
                return fail(error, "GLB JSON nesting exceeds the admission limit");
            ++depth;
        }
        else if ((c == '}' || c == ']') && depth > 0)
        {
            --depth;
        }
    }
    return true;
}

bool readStringArray(const Json& root,
                     const char* name,
                     std::vector<std::string>& destination,
                     std::string& error)
{
    const auto found = root.find(name);
    if (found == root.end())
        return true;
    if (!found->is_array())
        return fail(error, std::string("glTF ") + name + " must be an array");

    std::unordered_set<std::string> unique;
    for (const auto& item : *found)
    {
        if (!item.is_string() || item.get_ref<const std::string&>().empty())
            return fail(error, std::string("glTF ") + name + " must contain non-empty strings");
        const auto& value = item.get_ref<const std::string&>();
        if (!unique.insert(value).second)
            return fail(error, std::string("glTF ") + name + " contains a duplicate entry: " + value);
        destination.push_back(value);
    }
    return true;
}

bool arrayCount(const Json& root,
                const char* name,
                std::size_t maximum,
                std::size_t& destination,
                std::string& error)
{
    const auto found = root.find(name);
    if (found == root.end())
        return true;
    if (!found->is_array())
        return fail(error, std::string("glTF ") + name + " must be an array");
    if (found->size() > maximum)
        return fail(error, std::string("glTF ") + name + " exceeds the admission limit");
    destination = found->size();
    return true;
}

bool readIndex(const Json& value, std::size_t upperBound, std::size_t& result)
{
    std::uint64_t index = 0;
    if (value.is_number_unsigned())
        index = value.get<std::uint64_t>();
    else if (value.is_number_integer())
    {
        const auto signedValue = value.get<std::int64_t>();
        if (signedValue < 0)
            return false;
        index = static_cast<std::uint64_t>(signedValue);
    }
    else
        return false;
    if (index >= upperBound || index > std::numeric_limits<std::size_t>::max())
        return false;
    result = static_cast<std::size_t>(index);
    return true;
}

bool validateHierarchy(const Json& root,
                       const GlbAdmissionOptions& options,
                       GlbMetadata& metadata,
                       std::string& error)
{
    const auto nodesIt = root.find("nodes");
    const std::size_t nodeCount = metadata.nodes;
    std::vector<std::vector<std::size_t>> children(nodeCount);
    std::vector<std::size_t> parentCount(nodeCount, 0);

    if (nodesIt != root.end())
    {
        for (std::size_t node = 0; node < nodeCount; ++node)
        {
            const auto& record = (*nodesIt)[node];
            if (!record.is_object())
                return fail(error, "glTF nodes must contain objects");
            const auto childIt = record.find("children");
            if (childIt == record.end())
                continue;
            if (!childIt->is_array() || childIt->size() > nodeCount)
                return fail(error, "glTF node children are malformed or exceed the admission limit");
            std::unordered_set<std::size_t> unique;
            for (const auto& childValue : *childIt)
            {
                std::size_t child = 0;
                if (!readIndex(childValue, nodeCount, child))
                    return fail(error, "glTF node child index is out of range");
                if (!unique.insert(child).second)
                    return fail(error, "glTF node contains a duplicate child index");
                if (++parentCount[child] > 1)
                    return fail(error, "glTF node hierarchy gives a node more than one parent");
                children[node].push_back(child);
            }
        }
    }

    const auto scenesIt = root.find("scenes");
    if (scenesIt != root.end())
    {
        for (const auto& scene : *scenesIt)
        {
            if (!scene.is_object())
                return fail(error, "glTF scenes must contain objects");
            const auto roots = scene.find("nodes");
            if (roots == scene.end())
                continue;
            if (!roots->is_array() || roots->size() > nodeCount)
                return fail(error, "glTF scene node roots are malformed or exceed the admission limit");
            std::unordered_set<std::size_t> unique;
            for (const auto& rootValue : *roots)
            {
                std::size_t rootIndex = 0;
                if (!readIndex(rootValue, nodeCount, rootIndex))
                    return fail(error, "glTF scene node index is out of range");
                if (!unique.insert(rootIndex).second)
                    return fail(error, "glTF scene contains a duplicate root node");
            }
        }
    }

    std::vector<std::uint8_t> state(nodeCount, 0);
    struct Frame { std::size_t node; std::size_t nextChild; std::size_t depth; };
    std::vector<Frame> stack;
    stack.reserve(std::min(nodeCount, options.limits.maxNodeDepth + 1));
    for (std::size_t start = 0; start < nodeCount; ++start)
    {
        if (state[start] == 2)
            continue;
        stack.push_back({start, 0, 1});
        while (!stack.empty())
        {
            auto& frame = stack.back();
            if (state[frame.node] == 0)
            {
                state[frame.node] = 1;
                if (frame.depth > options.limits.maxNodeDepth)
                    return fail(error, "glTF node hierarchy exceeds the admission depth limit");
                metadata.maxHierarchyDepth = std::max(metadata.maxHierarchyDepth, frame.depth);
            }
            if (frame.nextChild == children[frame.node].size())
            {
                state[frame.node] = 2;
                stack.pop_back();
                continue;
            }
            const auto child = children[frame.node][frame.nextChild++];
            if (state[child] == 1)
                return fail(error, "glTF node hierarchy contains a cycle");
            if (state[child] == 0)
                stack.push_back({child, 0, frame.depth + 1});
        }
    }
    return true;
}

bool finiteFloatInRange(const Json& value, float minimum, float maximum, float& result)
{
    if (!value.is_number())
        return false;
    const auto number = value.get<double>();
    if (!std::isfinite(number) || number < static_cast<double>(minimum)
        || number > static_cast<double>(maximum))
        return false;
    result = static_cast<float>(number);
    return std::isfinite(result) && result >= minimum && result <= maximum;
}

bool validatePunctualLightsExtension(const Json& root,
                                     const GlbAdmissionOptions& options,
                                     const GlbMetadata& metadata,
                                     std::string& error)
{
    constexpr auto halfPi = 1.57079632679489661923f;
    const auto extensions = root.find("extensions");
    if (extensions != root.end() && !extensions->is_object())
        return fail(error, "glTF root extensions must be an object");
    const std::unordered_set<std::string> callerSupported(
        options.supportedRequiredExtensions.begin(), options.supportedRequiredExtensions.end());
    if (extensions != root.end())
        for (auto extension = extensions->begin(); extension != extensions->end(); ++extension)
            if (extension.key() != "KHR_lights_punctual"
                && callerSupported.count(extension.key()) == 0)
                return fail(error, "unsupported static glTF root extension: " + extension.key());

    const auto used = std::find(metadata.extensionsUsed.begin(), metadata.extensionsUsed.end(),
                                "KHR_lights_punctual") != metadata.extensionsUsed.end();
    const auto required = std::find(metadata.extensionsRequired.begin(), metadata.extensionsRequired.end(),
                                    "KHR_lights_punctual") != metadata.extensionsRequired.end();
    const Json* lights = nullptr;
    if (extensions != root.end())
    {
        const auto punctual = extensions->find("KHR_lights_punctual");
        if (punctual != extensions->end())
        {
            if (!used)
                return fail(error, "KHR_lights_punctual payload is not listed in extensionsUsed");
            if (!punctual->is_object())
                return fail(error, "KHR_lights_punctual root payload must be an object");
            const auto foundLights = punctual->find("lights");
            if (foundLights == punctual->end() || !foundLights->is_array()
                || foundLights->empty() || foundLights->size() > options.limits.maxLights)
                return fail(error, "KHR_lights_punctual lights are missing or exceed the admission limit");
            lights = &*foundLights;
        }
    }
    if (required && lights == nullptr)
        return fail(error, "required KHR_lights_punctual root payload is missing");

    if (lights != nullptr)
        for (const auto& light : *lights)
        {
            if (!light.is_object() || light.find("extensions") != light.end())
                return fail(error, "static GLB punctual lights must be unextended objects");
            const auto name = light.find("name");
            if (name != light.end() && !name->is_string())
                return fail(error, "glTF light name must be a string");
            const auto type = light.find("type");
            if (type == light.end() || !type->is_string()
                || (*type != "directional" && *type != "point" && *type != "spot"))
                return fail(error, "glTF punctual light type is missing or unsupported");
            const auto color = light.find("color");
            if (color != light.end())
            {
                if (!color->is_array() || color->size() != 3)
                    return fail(error, "glTF punctual light color must contain three factors");
                for (const auto& component : *color)
                {
                    float parsed = 0.0f;
                    if (!finiteFloatInRange(component, 0.0f, 1.0f, parsed))
                        return fail(error, "glTF punctual light color must contain finite factors from zero through one");
                }
            }
            const auto intensity = light.find("intensity");
            if (intensity != light.end())
            {
                float parsed = 0.0f;
                if (!finiteFloatInRange(*intensity, 0.0f, std::numeric_limits<float>::max(), parsed))
                    return fail(error, "glTF punctual light intensity must be finite and non-negative");
            }
            const auto range = light.find("range");
            if (range != light.end())
            {
                float parsed = 0.0f;
                if (!finiteFloatInRange(*range, 0.0f, std::numeric_limits<float>::max(), parsed)
                    || parsed <= 0.0f)
                    return fail(error, "glTF punctual light range must be finite and positive");
            }
            const auto spot = light.find("spot");
            if (*type == "spot")
            {
                if (spot != light.end() && !spot->is_object())
                    return fail(error, "glTF spot light payload must be an object");
                const auto& payload = spot == light.end() ? Json::object() : *spot;
                const auto inner = payload.find("innerConeAngle");
                const auto outer = payload.find("outerConeAngle");
                float innerValue = 0.0f;
                float outerValue = 0.7853981634f;
                if ((inner != payload.end() && !finiteFloatInRange(*inner, 0.0f, halfPi, innerValue))
                    || (outer != payload.end() && !finiteFloatInRange(*outer, 0.0f, halfPi, outerValue)))
                    return fail(error, "glTF spot light cone angles are invalid");
                if (innerValue >= outerValue || outerValue <= 0.0f)
                    return fail(error, "glTF spot light cone angles are invalid");
            }
            else if (spot != light.end())
                return fail(error, "glTF non-spot punctual light must not contain a spot payload");
        }

    std::size_t lightInstances = 0;
    const auto nodes = root.find("nodes");
    if (nodes != root.end())
        for (std::size_t nodeIndex = 0; nodeIndex < nodes->size(); ++nodeIndex)
        {
            const auto nodeExtensions = (*nodes)[nodeIndex].find("extensions");
            if (nodeExtensions == (*nodes)[nodeIndex].end())
                continue;
            if (!nodeExtensions->is_object() || nodeExtensions->size() != 1)
                return fail(error, "static GLB nodes admit only KHR_lights_punctual extensions");
            const auto punctual = nodeExtensions->find("KHR_lights_punctual");
            if (punctual == nodeExtensions->end() || !punctual->is_object() || lights == nullptr)
                return fail(error, "static GLB node punctual light extension is unsupported or missing its root payload");
            const auto light = punctual->find("light");
            std::size_t lightIndex = 0;
            if (light == punctual->end() || !readIndex(*light, lights->size(), lightIndex))
                return fail(error, "glTF node punctual light index is missing or out of range");
            if (++lightInstances > options.limits.maxLights
                || nodeIndex >= std::numeric_limits<std::uint32_t>::max())
                return fail(error, "glTF punctual light instances exceed the stable identity limit");
        }
    return true;
}

bool validateMetadata(const Json& root,
                      const GlbAdmissionOptions& options,
                      GlbMetadata& metadata,
                      std::string& error)
{
    if (!root.is_object())
        return fail(error, "glTF JSON root must be an object");
    const auto asset = root.find("asset");
    if (asset == root.end() || !asset->is_object())
        return fail(error, "glTF asset metadata is missing");
    const auto version = asset->find("version");
    if (version == asset->end() || !version->is_string()
        || version->get_ref<const std::string&>() != "2.0")
        return fail(error, "glTF asset.version must be exactly 2.0");
    const auto generator = asset->find("generator");
    if (generator != asset->end())
    {
        if (!generator->is_string())
            return fail(error, "glTF asset.generator must be a string");
        metadata.generator = generator->get<std::string>();
    }

    if (!arrayCount(root, "scenes", options.limits.maxScenes, metadata.scenes, error)
        || !arrayCount(root, "nodes", options.limits.maxNodes, metadata.nodes, error)
        || !arrayCount(root, "meshes", options.limits.maxMeshes, metadata.meshes, error)
        || !arrayCount(root, "materials", options.limits.maxMaterials, metadata.materials, error)
        || !arrayCount(root, "textures", options.limits.maxTextures, metadata.textures, error)
        || !arrayCount(root, "images", options.limits.maxImages, metadata.images, error)
        || !arrayCount(root, "samplers", options.limits.maxSamplers, metadata.samplers, error)
        || !arrayCount(root, "cameras", options.limits.maxCameras, metadata.cameras, error)
        || !arrayCount(root, "accessors", options.limits.maxAccessors, metadata.accessors, error)
        || !arrayCount(root, "bufferViews", options.limits.maxBufferViews, metadata.bufferViews, error)
        || !arrayCount(root, "animations", options.limits.maxNodes, metadata.animations, error)
        || !arrayCount(root, "skins", options.limits.maxNodes, metadata.skins, error))
        return false;

    if (!options.admitAnimations && metadata.animations != 0)
        return fail(error, "glTF animations are not admitted by the static GLB parser");
    if (!options.admitSkins && metadata.skins != 0)
        return fail(error, "glTF skins are not admitted by the static GLB parser");

    const auto defaultScene = root.find("scene");
    if (defaultScene != root.end())
    {
        std::size_t index = 0;
        if (!readIndex(*defaultScene, metadata.scenes, index))
            return fail(error, "glTF default scene index is out of range");
        metadata.defaultScene = index;
    }

    const auto meshes = root.find("meshes");
    if (meshes != root.end())
    {
        for (const auto& mesh : *meshes)
        {
            if (!mesh.is_object())
                return fail(error, "glTF meshes must contain objects");
            const auto primitives = mesh.find("primitives");
            if (primitives == mesh.end() || !primitives->is_array() || primitives->empty())
                return fail(error, "glTF mesh primitives must be a non-empty array");
            if (primitives->size() > options.limits.maxPrimitives - metadata.primitives)
                return fail(error, "glTF mesh primitives exceed the admission limit");
            for (const auto& primitive : *primitives)
                if (!primitive.is_object())
                    return fail(error, "glTF mesh primitives must contain objects");
            metadata.primitives += primitives->size();
        }
    }

    if (!readStringArray(root, "extensionsUsed", metadata.extensionsUsed, error)
        || !readStringArray(root, "extensionsRequired", metadata.extensionsRequired, error))
        return false;
    const std::unordered_set<std::string> used(metadata.extensionsUsed.begin(), metadata.extensionsUsed.end());
    const std::unordered_set<std::string> supported(options.supportedRequiredExtensions.begin(),
                                                     options.supportedRequiredExtensions.end());
    for (const auto& extension : metadata.extensionsRequired)
    {
        if (used.count(extension) == 0)
            return fail(error, "glTF required extension is not listed in extensionsUsed: " + extension);
        if (extension != "KHR_lights_punctual" && supported.count(extension) == 0)
            return fail(error, "unsupported required glTF extension: " + extension);
    }

    return validatePunctualLightsExtension(root, options, metadata, error)
        && validateHierarchy(root, options, metadata, error);
}

bool readSizeValue(const Json& value, std::size_t& result)
{
    std::uint64_t number = 0;
    if (value.is_number_unsigned())
        number = value.get<std::uint64_t>();
    else if (value.is_number_integer())
    {
        const auto signedNumber = value.get<std::int64_t>();
        if (signedNumber < 0)
            return false;
        number = static_cast<std::uint64_t>(signedNumber);
    }
    else
        return false;
    if (number > std::numeric_limits<std::size_t>::max())
        return false;
    result = static_cast<std::size_t>(number);
    return true;
}

bool readRequiredSize(const Json& object,
                      const char* name,
                      std::size_t& result,
                      std::string& error)
{
    const auto found = object.find(name);
    if (found == object.end() || !readSizeValue(*found, result))
        return fail(error, std::string("glTF ") + name + " must be a non-negative integer");
    return true;
}

bool readOptionalSize(const Json& object,
                      const char* name,
                      std::size_t defaultValue,
                      std::size_t& result,
                      std::string& error)
{
    const auto found = object.find(name);
    if (found == object.end())
    {
        result = defaultValue;
        return true;
    }
    if (!readSizeValue(*found, result))
        return fail(error, std::string("glTF ") + name + " must be a non-negative integer");
    return true;
}

bool readOptionalName(const Json& object, std::string& result, std::string& error)
{
    const auto found = object.find("name");
    if (found == object.end())
        return true;
    if (!found->is_string())
        return fail(error, "glTF name must be a string");
    result = found->get<std::string>();
    return true;
}

std::optional<GlbAccessorType> parseAccessorType(const std::string& type)
{
    if (type == "SCALAR") return GlbAccessorType::Scalar;
    if (type == "VEC2") return GlbAccessorType::Vec2;
    if (type == "VEC3") return GlbAccessorType::Vec3;
    if (type == "VEC4") return GlbAccessorType::Vec4;
    if (type == "MAT2") return GlbAccessorType::Mat2;
    if (type == "MAT3") return GlbAccessorType::Mat3;
    if (type == "MAT4") return GlbAccessorType::Mat4;
    return std::nullopt;
}

std::size_t componentByteSize(std::uint32_t componentType)
{
    switch (componentType)
    {
        case 5120:
        case 5121: return 1;
        case 5122:
        case 5123: return 2;
        case 5125:
        case 5126: return 4;
        default: return 0;
    }
}

std::size_t accessorComponentCount(GlbAccessorType type)
{
    switch (type)
    {
        case GlbAccessorType::Scalar: return 1;
        case GlbAccessorType::Vec2: return 2;
        case GlbAccessorType::Vec3: return 3;
        case GlbAccessorType::Vec4:
        case GlbAccessorType::Mat2: return 4;
        case GlbAccessorType::Mat3: return 9;
        case GlbAccessorType::Mat4: return 16;
    }
    return 0;
}

std::size_t accessorElementByteSize(const GlbAccessorRecord& accessor)
{
    const auto componentBytes = componentByteSize(accessor.componentType);
    if (accessor.type == GlbAccessorType::Mat2 || accessor.type == GlbAccessorType::Mat3
        || accessor.type == GlbAccessorType::Mat4)
    {
        const std::size_t dimension = accessor.type == GlbAccessorType::Mat2 ? 2
                                    : accessor.type == GlbAccessorType::Mat3 ? 3 : 4;
        const auto columnBytes = dimension * componentBytes;
        const auto paddedColumnBytes = (columnBytes + 3u) & ~std::size_t(3u);
        return dimension * paddedColumnBytes;
    }
    return accessorComponentCount(accessor.type) * componentBytes;
}

bool rangeFits(std::size_t offset, std::size_t length, std::size_t available)
{
    return offset <= available && length <= available - offset;
}

bool checkedMultiply(std::size_t left, std::size_t right, std::size_t& result)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

bool checkedAdd(std::size_t left, std::size_t right, std::size_t& result)
{
    if (right > std::numeric_limits<std::size_t>::max() - left)
        return false;
    result = left + right;
    return true;
}

bool parseBufferViewsAndAccessors(const Json& root,
                                  const std::uint8_t* binBytes,
                                  std::size_t binSize,
                                  GlbStaticMeshDocument& document,
                                  std::string& error)
{
    const auto buffers = root.find("buffers");
    if (buffers == root.end() || !buffers->is_array() || buffers->size() != 1
        || !(*buffers)[0].is_object())
        return fail(error, "static GLB decode requires exactly one embedded buffer");
    const auto& buffer = (*buffers)[0];
    if (buffer.find("uri") != buffer.end() || buffer.find("extensions") != buffer.end())
        return fail(error, "static GLB decode requires an unextended embedded buffer");
    std::size_t declaredBufferBytes = 0;
    const auto byteLength = buffer.find("byteLength");
    if (byteLength == buffer.end() || !readSizeValue(*byteLength, declaredBufferBytes))
        return fail(error, "glTF buffer byteLength must be a non-negative integer");
    if (declaredBufferBytes == 0 || binBytes == nullptr || binSize < declaredBufferBytes
        || binSize - declaredBufferBytes > 3)
        return fail(error, "GLB BIN chunk does not exactly contain the declared buffer plus alignment padding");

    const auto views = root.find("bufferViews");
    if (views == root.end())
    {
        if (!document.accessors.empty())
            return fail(error, "glTF accessors require bufferViews");
        return true;
    }
    document.bufferViews.reserve(views->size());
    for (const auto& value : *views)
    {
        if (!value.is_object())
            return fail(error, "glTF bufferViews must contain objects");
        if (value.find("extensions") != value.end())
            return fail(error, "static GLB decode does not admit extended bufferViews");
        std::size_t bufferIndex = 0;
        const auto bufferIt = value.find("buffer");
        if (bufferIt == value.end() || !readIndex(*bufferIt, 1, bufferIndex))
            return fail(error, "glTF bufferView buffer index must reference the embedded GLB buffer");

        GlbBufferViewRecord record;
        if (!readOptionalSize(value, "byteOffset", 0, record.byteOffset, error)
            || !readRequiredSize(value, "byteLength", record.byteLength, error))
            return false;
        if (record.byteLength == 0
            || !rangeFits(record.byteOffset, record.byteLength, declaredBufferBytes))
            return fail(error, "glTF bufferView range exceeds the embedded buffer");

        const auto stride = value.find("byteStride");
        if (stride != value.end())
        {
            std::size_t parsedStride = 0;
            if (!readSizeValue(*stride, parsedStride) || parsedStride < 4
                || parsedStride > 252 || (parsedStride & 3u) != 0)
                return fail(error, "glTF bufferView byteStride must be a four-byte multiple from 4 through 252");
            record.byteStride = parsedStride;
        }
        const auto target = value.find("target");
        if (target != value.end())
        {
            std::size_t parsedTarget = 0;
            if (!readSizeValue(*target, parsedTarget)
                || (parsedTarget != 34962 && parsedTarget != 34963))
                return fail(error, "glTF bufferView target must be ARRAY_BUFFER or ELEMENT_ARRAY_BUFFER");
            record.target = static_cast<std::uint32_t>(parsedTarget);
        }
        document.bufferViews.push_back(record);
    }

    const auto accessors = root.find("accessors");
    if (accessors == root.end())
        return true;
    document.accessors.reserve(accessors->size());
    for (const auto& value : *accessors)
    {
        if (!value.is_object())
            return fail(error, "glTF accessors must contain objects");
        if (value.find("sparse") != value.end() || value.find("extensions") != value.end())
            return fail(error, "static GLB decode does not admit sparse or extended accessors");

        GlbAccessorRecord record;
        const auto viewIt = value.find("bufferView");
        if (viewIt == value.end()
            || !readIndex(*viewIt, document.bufferViews.size(), record.bufferView))
            return fail(error, "glTF accessor bufferView index is missing or out of range");
        if (!readOptionalSize(value, "byteOffset", 0, record.byteOffset, error)
            || !readRequiredSize(value, "count", record.count, error))
            return false;
        if (record.count == 0)
            return fail(error, "glTF accessor count must be greater than zero");

        std::size_t parsedComponentType = 0;
        const auto componentType = value.find("componentType");
        if (componentType == value.end() || !readSizeValue(*componentType, parsedComponentType)
            || parsedComponentType > std::numeric_limits<std::uint32_t>::max()
            || componentByteSize(static_cast<std::uint32_t>(parsedComponentType)) == 0)
            return fail(error, "glTF accessor componentType is unsupported or malformed");
        record.componentType = static_cast<std::uint32_t>(parsedComponentType);

        const auto type = value.find("type");
        if (type == value.end() || !type->is_string())
            return fail(error, "glTF accessor type is missing or malformed");
        const auto parsedType = parseAccessorType(type->get_ref<const std::string&>());
        if (!parsedType)
            return fail(error, "glTF accessor type is unsupported");
        record.type = *parsedType;

        const auto normalized = value.find("normalized");
        if (normalized != value.end())
        {
            if (!normalized->is_boolean())
                return fail(error, "glTF accessor normalized must be a boolean");
            record.normalized = normalized->get<bool>();
        }

        const auto componentBytes = componentByteSize(record.componentType);
        const auto elementBytes = accessorElementByteSize(record);
        const auto& view = document.bufferViews[record.bufferView];
        const auto stride = view.byteStride.value_or(elementBytes);
        if (stride < elementBytes || stride % componentBytes != 0)
            return fail(error, "glTF accessor element does not fit its bufferView byteStride");
        if (record.byteOffset > view.byteLength
            || (view.byteOffset + record.byteOffset) % componentBytes != 0)
            return fail(error, "glTF accessor byteOffset is out of range or misaligned");
        const auto remaining = view.byteLength - record.byteOffset;
        if (remaining < elementBytes
            || record.count - 1 > (remaining - elementBytes) / stride)
            return fail(error, "glTF accessor range exceeds its bufferView");
        document.accessors.push_back(record);
    }
    return true;
}

const std::uint8_t* accessorData(const GlbStaticMeshDocument& document,
                                 const std::uint8_t* binBytes,
                                 const GlbAccessorRecord& accessor)
{
    const auto& view = document.bufferViews[accessor.bufferView];
    return binBytes + view.byteOffset + accessor.byteOffset;
}

std::size_t accessorStride(const GlbStaticMeshDocument& document,
                           const GlbAccessorRecord& accessor)
{
    return document.bufferViews[accessor.bufferView].byteStride.value_or(
        accessorElementByteSize(accessor));
}

float readFloat(const std::uint8_t* bytes)
{
    float value = 0.0f;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

bool decodeFloatAttribute(const GlbStaticMeshDocument& document,
                          const std::uint8_t* binBytes,
                          const GlbAccessorRecord& accessor,
                          GlbAccessorType requiredType,
                          std::vector<float>& destination,
                          std::string& error)
{
    if (accessor.componentType != 5126 || accessor.type != requiredType || accessor.normalized)
        return fail(error, "glTF mesh float attribute has an unsupported accessor format");
    const auto components = accessorComponentCount(requiredType);
    if (accessor.count > std::numeric_limits<std::size_t>::max() / components)
        return fail(error, "glTF mesh attribute element count overflows the decode range");
    destination.resize(accessor.count * components);
    const auto* source = accessorData(document, binBytes, accessor);
    const auto stride = accessorStride(document, accessor);
    for (std::size_t element = 0; element < accessor.count; ++element)
    {
        for (std::size_t component = 0; component < components; ++component)
        {
            const auto value = readFloat(source + element * stride + component * sizeof(float));
            if (!std::isfinite(value))
                return fail(error, "glTF mesh attributes must contain finite values");
            destination[element * components + component] = value;
        }
    }
    return true;
}

bool decodeTexCoords(const GlbStaticMeshDocument& document,
                     const std::uint8_t* binBytes,
                     const GlbAccessorRecord& accessor,
                     std::vector<float>& destination,
                     std::string& error)
{
    if (accessor.type != GlbAccessorType::Vec2)
        return fail(error, "glTF TEXCOORD_0 accessor must have type VEC2");
    const bool floatCoordinates = accessor.componentType == 5126 && !accessor.normalized;
    const bool byteCoordinates = accessor.componentType == 5121 && accessor.normalized;
    const bool shortCoordinates = accessor.componentType == 5123 && accessor.normalized;
    if (!floatCoordinates && !byteCoordinates && !shortCoordinates)
        return fail(error, "glTF TEXCOORD_0 accessor must use FLOAT or normalized unsigned integers");

    if (accessor.count > std::numeric_limits<std::size_t>::max() / 2)
        return fail(error, "glTF texture coordinate count overflows the decode range");
    destination.resize(accessor.count * 2);
    const auto* source = accessorData(document, binBytes, accessor);
    const auto stride = accessorStride(document, accessor);
    for (std::size_t element = 0; element < accessor.count; ++element)
    {
        for (std::size_t component = 0; component < 2; ++component)
        {
            float decoded = 0.0f;
            if (floatCoordinates)
                decoded = readFloat(source + element * stride + component * sizeof(float));
            else if (byteCoordinates)
                decoded = static_cast<float>(source[element * stride + component]) / 255.0f;
            else
            {
                std::uint16_t value = 0;
                std::memcpy(&value, source + element * stride + component * sizeof(value), sizeof(value));
                decoded = static_cast<float>(value) / 65535.0f;
            }
            if (!std::isfinite(decoded))
                return fail(error, "glTF texture coordinates must contain finite values");
            destination[element * 2 + component] = decoded;
        }
    }
    return true;
}

bool decodeColors(const GlbStaticMeshDocument& document,
                  const std::uint8_t* binBytes,
                  const GlbAccessorRecord& accessor,
                  std::vector<float>& destination,
                  std::string& error)
{
    const auto components = accessor.type == GlbAccessorType::Vec3 ? 3u
                          : accessor.type == GlbAccessorType::Vec4 ? 4u : 0u;
    const bool floatColors = accessor.componentType == 5126 && !accessor.normalized;
    const bool byteColors = accessor.componentType == 5121 && accessor.normalized;
    const bool shortColors = accessor.componentType == 5123 && accessor.normalized;
    if (components == 0 || (!floatColors && !byteColors && !shortColors))
        return fail(error, "glTF COLOR_0 must be FLOAT or normalized unsigned VEC3/VEC4 data");
    std::size_t decodedValues = 0;
    if (!checkedMultiply(accessor.count, 4, decodedValues))
        return fail(error, "glTF COLOR_0 element count overflows the decode range");
    destination.resize(decodedValues);
    const auto* source = accessorData(document, binBytes, accessor);
    const auto stride = accessorStride(document, accessor);
    for (std::size_t element = 0; element < accessor.count; ++element)
    {
        destination[element * 4 + 3] = 1.0f;
        for (std::size_t component = 0; component < components; ++component)
        {
            float decoded = 0.0f;
            if (floatColors)
                decoded = readFloat(source + element * stride + component * sizeof(float));
            else if (byteColors)
                decoded = static_cast<float>(source[element * stride + component]) / 255.0f;
            else
            {
                std::uint16_t value = 0;
                std::memcpy(&value, source + element * stride + component * sizeof(value), sizeof(value));
                decoded = static_cast<float>(value) / 65535.0f;
            }
            if (!std::isfinite(decoded) || decoded < 0.0f || decoded > 1.0f)
                return fail(error, "glTF COLOR_0 values must be finite and between zero and one");
            destination[element * 4 + component] = decoded;
        }
    }
    return true;
}

bool decodeIndices(const GlbStaticMeshDocument& document,
                   const std::uint8_t* binBytes,
                   const GlbAccessorRecord& accessor,
                   std::size_t vertexCount,
                   std::vector<std::uint32_t>& destination,
                   std::string& error)
{
    if (accessor.type != GlbAccessorType::Scalar || accessor.normalized
        || (accessor.componentType != 5121 && accessor.componentType != 5123
            && accessor.componentType != 5125))
        return fail(error, "glTF triangle indices must be non-normalized unsigned scalar values");
    if (accessor.count % 3 != 0)
        return fail(error, "glTF triangle index count must be divisible by three");

    destination.resize(accessor.count);
    const auto* source = accessorData(document, binBytes, accessor);
    const auto stride = accessorStride(document, accessor);
    for (std::size_t index = 0; index < accessor.count; ++index)
    {
        std::uint32_t decoded = 0;
        if (accessor.componentType == 5121)
            decoded = source[index * stride];
        else if (accessor.componentType == 5123)
        {
            std::uint16_t value = 0;
            std::memcpy(&value, source + index * stride, sizeof(value));
            decoded = value;
        }
        else
            std::memcpy(&decoded, source + index * stride, sizeof(decoded));
        if (decoded >= vertexCount)
            return fail(error, "glTF triangle index references a vertex outside POSITION");
        destination[index] = decoded;
    }
    return true;
}

bool readFloatValue(const Json& value,
                    float minimum,
                    float maximum,
                    float& result,
                    const char* diagnostic,
                    std::string& error)
{
    if (!value.is_number())
        return fail(error, diagnostic);
    const auto parsed = value.get<double>();
    if (!std::isfinite(parsed) || parsed < minimum || parsed > maximum
        || std::abs(parsed) > std::numeric_limits<float>::max())
        return fail(error, diagnostic);
    result = static_cast<float>(parsed);
    return true;
}

template <std::size_t Size>
bool readFactorArray(const Json& object,
                     const char* name,
                     std::array<float, Size>& destination,
                     std::string& error)
{
    const auto found = object.find(name);
    if (found == object.end())
        return true;
    if (!found->is_array() || found->size() != Size)
        return fail(error, std::string("glTF ") + name + " must contain "
                           + std::to_string(Size) + " factors");
    for (std::size_t i = 0; i < Size; ++i)
        if (!readFloatValue((*found)[i], 0.0f, 1.0f, destination[i],
                            "glTF material factors must be finite values from zero through one",
                            error))
            return false;
    return true;
}

bool parseTextureInfo(const Json& owner,
                      const char* name,
                      std::size_t textureCount,
                      std::optional<GlbTextureInfo>& destination,
                      std::string& error)
{
    const auto found = owner.find(name);
    if (found == owner.end())
        return true;
    if (!found->is_object() || found->find("extensions") != found->end())
        return fail(error, std::string("glTF ") + name + " must be an unextended texture reference");
    const auto index = found->find("index");
    GlbTextureInfo info;
    if (index == found->end() || !readIndex(*index, textureCount, info.texture))
        return fail(error, std::string("glTF ") + name + " texture index is missing or out of range");
    if (!readOptionalSize(*found, "texCoord", 0, info.texCoord, error))
        return false;
    if (info.texCoord != 0)
        return fail(error, "static GLB decode admits only TEXCOORD_0 material references");
    destination = info;
    return true;
}

enum class EmbeddedImageFormat
{
    Png,
    Jpeg
};

struct EmbeddedImageSource
{
    const std::uint8_t* bytes = nullptr;
    std::size_t size = 0;
    EmbeddedImageFormat format = EmbeddedImageFormat::Png;
};

std::uint16_t readBigU16(const std::uint8_t* bytes)
{
    return static_cast<std::uint16_t>((static_cast<std::uint16_t>(bytes[0]) << 8u)
                                    | static_cast<std::uint16_t>(bytes[1]));
}

std::uint32_t readBigU32(const std::uint8_t* bytes)
{
    return (static_cast<std::uint32_t>(bytes[0]) << 24u)
         | (static_cast<std::uint32_t>(bytes[1]) << 16u)
         | (static_cast<std::uint32_t>(bytes[2]) << 8u)
         | static_cast<std::uint32_t>(bytes[3]);
}

bool inspectPng(const std::uint8_t* bytes,
                std::size_t size,
                std::uint32_t& width,
                std::uint32_t& height,
                std::string& error)
{
    constexpr std::array<std::uint8_t, 8> signature { 137, 80, 78, 71, 13, 10, 26, 10 };
    if (size < 33 || !std::equal(signature.begin(), signature.end(), bytes)
        || readBigU32(bytes + 8) != 13 || std::memcmp(bytes + 12, "IHDR", 4) != 0)
        return fail(error, "embedded PNG image has a malformed signature or IHDR");

    width = readBigU32(bytes + 16);
    height = readBigU32(bytes + 20);
    const auto bitDepth = bytes[24];
    const auto colorType = bytes[25];
    if (width == 0 || height == 0)
        return fail(error, "embedded PNG image dimensions must be nonzero");
    if (bitDepth != 8
        || (colorType != 0 && colorType != 2 && colorType != 3
            && colorType != 4 && colorType != 6))
        return fail(error, "embedded PNG image must use supported 8-bit color components");
    if (bytes[26] != 0 || bytes[27] != 0 || bytes[28] > 1)
        return fail(error, "embedded PNG image uses unsupported compression, filtering, or interlace");
    return true;
}

bool isJpegSof(std::uint8_t marker)
{
    return marker >= 0xc0 && marker <= 0xcf
        && marker != 0xc4 && marker != 0xc8 && marker != 0xcc;
}

bool inspectJpeg(const std::uint8_t* bytes,
                 std::size_t size,
                 std::uint32_t& width,
                 std::uint32_t& height,
                 std::string& error)
{
    if (size < 4 || bytes[0] != 0xff || bytes[1] != 0xd8)
        return fail(error, "embedded JPEG image has a malformed SOI marker");

    std::size_t offset = 2;
    while (offset < size)
    {
        if (bytes[offset++] != 0xff)
            return fail(error, "embedded JPEG image has malformed marker framing");
        while (offset < size && bytes[offset] == 0xff)
            ++offset;
        if (offset >= size)
            break;
        const auto marker = bytes[offset++];
        if (marker == 0x00)
            return fail(error, "embedded JPEG image has malformed marker framing");
        if (marker == 0xd9 || marker == 0xda)
            break;
        if (marker == 0x01 || (marker >= 0xd0 && marker <= 0xd7))
            continue;
        if (size - offset < 2)
            break;
        const auto segmentLength = static_cast<std::size_t>(readBigU16(bytes + offset));
        if (segmentLength < 2 || segmentLength > size - offset)
            return fail(error, "embedded JPEG image has a malformed marker segment");
        if (isJpegSof(marker))
        {
            if (marker != 0xc0 && marker != 0xc1 && marker != 0xc2)
                return fail(error, "embedded JPEG image uses an unsupported frame encoding");
            if (segmentLength < 8)
                return fail(error, "embedded JPEG image has a malformed frame header");
            const auto* frame = bytes + offset + 2;
            const auto components = frame[5];
            if (frame[0] != 8 || (components != 1 && components != 3)
                || segmentLength < 8u + 3u * components)
                return fail(error, "embedded JPEG image must use 8-bit grayscale or three-component color");
            height = readBigU16(frame + 1);
            width = readBigU16(frame + 3);
            if (width == 0 || height == 0)
                return fail(error, "embedded JPEG image dimensions must be nonzero");
            return true;
        }
        offset += segmentLength;
    }
    return fail(error, "embedded JPEG image has no supported frame header");
}

struct CodecContextDeleter
{
    void operator()(AVCodecContext* value) const
    {
        avcodec_free_context(&value);
    }
};

struct PacketDeleter
{
    void operator()(AVPacket* value) const
    {
        av_packet_free(&value);
    }
};

struct FrameDeleter
{
    void operator()(AVFrame* value) const
    {
        av_frame_free(&value);
    }
};

struct SwsContextDeleter
{
    void operator()(SwsContext* value) const
    {
        sws_freeContext(value);
    }
};

bool decodeEmbeddedImage(const EmbeddedImageSource& source,
                         GlbImageRecord& image,
                         std::string& error)
{
    const auto codecId = source.format == EmbeddedImageFormat::Png
        ? AV_CODEC_ID_PNG : AV_CODEC_ID_MJPEG;
    const auto* codec = avcodec_find_decoder(codecId);
    if (codec == nullptr)
        return fail(error, source.format == EmbeddedImageFormat::Png
            ? "embedded PNG image decoder is unavailable"
            : "embedded JPEG image decoder is unavailable");

    std::unique_ptr<AVCodecContext, CodecContextDeleter> context(avcodec_alloc_context3(codec));
    std::unique_ptr<AVPacket, PacketDeleter> packet(av_packet_alloc());
    std::unique_ptr<AVFrame, FrameDeleter> frame(av_frame_alloc());
    if (!context || !packet || !frame)
        return fail(error, "embedded image decoder allocation failed within admitted limits");

    // FFmpeg validates some still-image codecs against a 64-pixel aligned row
    // width before publishing the exact decoded dimensions. Admit only that
    // bounded alignment overhead, then verify the published extent below.
    const auto alignedWidth = (static_cast<std::int64_t>(image.width) + 63) & ~std::int64_t {63};
    context->max_pixels = alignedWidth * static_cast<std::int64_t>(image.height);
    context->thread_count = 1;
    if (avcodec_open2(context.get(), codec, nullptr) < 0)
        return fail(error, "embedded image decoder initialization failed");

    packet->data = const_cast<std::uint8_t*>(source.bytes);
    packet->size = static_cast<int>(source.size);
    if (avcodec_send_packet(context.get(), packet.get()) < 0
        || avcodec_receive_frame(context.get(), frame.get()) < 0)
        return fail(error, source.format == EmbeddedImageFormat::Png
            ? "embedded PNG image decode failed"
            : "embedded JPEG image decode failed");
    if (frame->width != static_cast<int>(image.width)
        || frame->height != static_cast<int>(image.height))
        return fail(error, "embedded image decoded dimensions differ from its admitted header");

    const auto pixelFormat = static_cast<AVPixelFormat>(frame->format);
    const auto* descriptor = av_pix_fmt_desc_get(pixelFormat);
    if (descriptor == nullptr
        || (descriptor->flags & (AV_PIX_FMT_FLAG_HWACCEL | AV_PIX_FMT_FLAG_BITSTREAM)) != 0)
        return fail(error, "embedded image decoder produced an unsupported pixel format");
    for (std::size_t component = 0; component < descriptor->nb_components; ++component)
        if (descriptor->comp[component].depth > 8)
            return fail(error, "embedded image decoder produced unsupported component depth");

    std::unique_ptr<SwsContext, SwsContextDeleter> converter(sws_getContext(
        frame->width, frame->height, pixelFormat,
        frame->width, frame->height, AV_PIX_FMT_RGBA,
        SWS_POINT, nullptr, nullptr, nullptr));
    if (!converter)
        return fail(error, "embedded image RGBA conversion initialization failed");
    std::uint8_t* destination[] { image.decodedRgba8.data(), nullptr, nullptr, nullptr };
    int strides[] { frame->width * 4, 0, 0, 0 };
    if (sws_scale(converter.get(), frame->data, frame->linesize, 0, frame->height,
                  destination, strides) != frame->height)
        return fail(error, "embedded image RGBA conversion failed");
    return true;
}

bool parseMaterials(const Json& root,
                    const std::uint8_t* binBytes,
                    const GlbAdmissionOptions& options,
                    GlbStaticMeshDocument& document,
                    std::string& error)
{
    std::size_t embeddedImageBytes = 0;
    std::size_t decodedImageBytes = 0;
    std::vector<EmbeddedImageSource> imageSources;
    const auto images = root.find("images");
    if (images != root.end())
    {
        document.images.reserve(images->size());
        imageSources.reserve(images->size());
        for (const auto& value : *images)
        {
            if (!value.is_object() || value.find("uri") != value.end()
                || value.find("extensions") != value.end())
                return fail(error, "static GLB images must use an embedded bufferView without extensions or URIs");
            GlbImageRecord image;
            if (!readOptionalName(value, image.name, error))
                return false;
            const auto view = value.find("bufferView");
            if (view == value.end()
                || !readIndex(*view, document.bufferViews.size(), image.bufferView))
                return fail(error, "glTF image bufferView index is missing or out of range");
            const auto mimeType = value.find("mimeType");
            if (mimeType == value.end() || !mimeType->is_string())
                return fail(error, "embedded glTF image mimeType is missing or malformed");
            image.mimeType = mimeType->get<std::string>();
            EmbeddedImageSource source;
            if (image.mimeType == "image/png")
                source.format = EmbeddedImageFormat::Png;
            else if (image.mimeType == "image/jpeg")
                source.format = EmbeddedImageFormat::Jpeg;
            else
                return fail(error, "static GLB images admit only image/png or image/jpeg payloads");
            const auto& imageView = document.bufferViews[image.bufferView];
            if (imageView.byteStride || imageView.target)
                return fail(error, "embedded glTF image bufferViews must not declare stride or target");
            if (embeddedImageBytes > options.limits.maxEmbeddedImageBytes
                || imageView.byteLength > options.limits.maxEmbeddedImageBytes - embeddedImageBytes)
                return fail(error, "embedded glTF image bytes exceed the admission limit");
            if (imageView.byteLength > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                return fail(error, "embedded glTF image exceeds the decoder packet-size limit");
            embeddedImageBytes += imageView.byteLength;
            source.bytes = binBytes + imageView.byteOffset;
            source.size = imageView.byteLength;
            if (source.format == EmbeddedImageFormat::Png)
            {
                if (!inspectPng(source.bytes, source.size, image.width, image.height, error))
                    return false;
            }
            else if (!inspectJpeg(source.bytes, source.size, image.width, image.height, error))
                return false;
            if (image.width > options.limits.maxImageWidth
                || image.height > options.limits.maxImageHeight
                || image.width > static_cast<std::uint32_t>(std::numeric_limits<int>::max() / 4))
                return fail(error, "embedded glTF image dimensions exceed the admission limit");
            std::size_t pixelCount = 0;
            std::size_t imageBytes = 0;
            if (!checkedMultiply(static_cast<std::size_t>(image.width),
                                 static_cast<std::size_t>(image.height), pixelCount)
                || !checkedMultiply(pixelCount, 4u, imageBytes)
                || decodedImageBytes > options.limits.maxDecodedImageBytes
                || imageBytes > options.limits.maxDecodedImageBytes - decodedImageBytes)
                return fail(error, "decoded embedded image bytes exceed the image admission limit");
            decodedImageBytes += imageBytes;
            document.images.push_back(std::move(image));
            imageSources.push_back(source);
        }
    }
    if (document.decodedBytes > options.limits.maxDecodedBytes
        || decodedImageBytes > options.limits.maxDecodedBytes - document.decodedBytes)
        return fail(error, "decoded glTF bytes exceed the admission limit");
    document.decodedBytes += decodedImageBytes;
    for (std::size_t imageIndex = 0; imageIndex < document.images.size(); ++imageIndex)
    {
        auto& image = document.images[imageIndex];
        const auto imageBytes = static_cast<std::size_t>(image.width)
                              * static_cast<std::size_t>(image.height) * 4u;
        image.decodedRgba8.resize(imageBytes);
        if (!decodeEmbeddedImage(imageSources[imageIndex], image, error))
            return false;
    }

    const auto samplers = root.find("samplers");
    if (samplers != root.end())
    {
        const std::unordered_set<std::size_t> magFilters {9728, 9729};
        const std::unordered_set<std::size_t> minFilters {9728, 9729, 9984, 9985, 9986, 9987};
        const std::unordered_set<std::size_t> wraps {33071, 33648, 10497};
        document.samplers.reserve(samplers->size());
        for (const auto& value : *samplers)
        {
            if (!value.is_object() || value.find("extensions") != value.end())
                return fail(error, "static GLB samplers must be unextended objects");
            GlbSamplerRecord sampler;
            if (!readOptionalName(value, sampler.name, error))
                return false;
            for (const auto* name : {"magFilter", "minFilter", "wrapS", "wrapT"})
            {
                const auto found = value.find(name);
                if (found == value.end())
                    continue;
                std::size_t parsed = 0;
                if (!readSizeValue(*found, parsed))
                    return fail(error, std::string("glTF sampler ") + name + " is malformed");
                const auto& accepted = std::string(name) == "magFilter" ? magFilters
                                     : std::string(name) == "minFilter" ? minFilters : wraps;
                if (accepted.count(parsed) == 0)
                    return fail(error, std::string("glTF sampler ") + name + " is unsupported");
                auto* destination = std::string(name) == "magFilter" ? &sampler.magFilter
                                  : std::string(name) == "minFilter" ? &sampler.minFilter
                                  : std::string(name) == "wrapS" ? &sampler.wrapS : &sampler.wrapT;
                *destination = static_cast<std::uint32_t>(parsed);
            }
            document.samplers.push_back(std::move(sampler));
        }
    }

    const auto textures = root.find("textures");
    if (textures != root.end())
    {
        document.textures.reserve(textures->size());
        for (const auto& value : *textures)
        {
            if (!value.is_object() || value.find("extensions") != value.end())
                return fail(error, "static GLB textures must be unextended objects");
            GlbTextureRecord texture;
            if (!readOptionalName(value, texture.name, error))
                return false;
            const auto source = value.find("source");
            if (source == value.end() || !readIndex(*source, document.images.size(), texture.source))
                return fail(error, "glTF texture image source is missing or out of range");
            const auto sampler = value.find("sampler");
            if (sampler != value.end())
            {
                std::size_t index = 0;
                if (!readIndex(*sampler, document.metadata.samplers, index))
                    return fail(error, "glTF texture sampler index is out of range");
                texture.sampler = index;
            }
            document.textures.push_back(std::move(texture));
        }
    }

    const auto materials = root.find("materials");
    if (materials == root.end())
        return true;
    document.materials.reserve(materials->size());
    for (const auto& value : *materials)
    {
        if (!value.is_object() || value.find("extensions") != value.end())
            return fail(error, "static GLB materials must be unextended objects");
        GlbMaterialRecord material;
        if (!readOptionalName(value, material.name, error)
            || !readFactorArray(value, "emissiveFactor", material.emissiveFactor, error))
            return false;
        const auto pbr = value.find("pbrMetallicRoughness");
        if (pbr != value.end())
        {
            if (!pbr->is_object() || pbr->find("extensions") != pbr->end())
                return fail(error, "glTF pbrMetallicRoughness must be an unextended object");
            if (!readFactorArray(*pbr, "baseColorFactor", material.baseColorFactor, error)
                || !parseTextureInfo(*pbr, "baseColorTexture", document.textures.size(),
                                     material.baseColorTexture, error)
                || !parseTextureInfo(*pbr, "metallicRoughnessTexture", document.textures.size(),
                                     material.metallicRoughnessTexture, error))
                return false;
            const auto metallic = pbr->find("metallicFactor");
            if (metallic != pbr->end()
                && !readFloatValue(*metallic, 0.0f, 1.0f, material.metallicFactor,
                                   "glTF metallicFactor must be finite and between zero and one", error))
                return false;
            const auto roughness = pbr->find("roughnessFactor");
            if (roughness != pbr->end()
                && !readFloatValue(*roughness, 0.0f, 1.0f, material.roughnessFactor,
                                   "glTF roughnessFactor must be finite and between zero and one", error))
                return false;
        }
        if (!parseTextureInfo(value, "normalTexture", document.textures.size(),
                              material.normalTexture, error)
            || !parseTextureInfo(value, "occlusionTexture", document.textures.size(),
                                 material.occlusionTexture, error)
            || !parseTextureInfo(value, "emissiveTexture", document.textures.size(),
                                 material.emissiveTexture, error))
            return false;
        if (const auto normal = value.find("normalTexture"); normal != value.end())
        {
            const auto scale = normal->find("scale");
            if (scale != normal->end()
                && !readFloatValue(*scale, -std::numeric_limits<float>::max(),
                                   std::numeric_limits<float>::max(), material.normalScale,
                                   "glTF normal texture scale must be a finite float", error))
                return false;
        }
        if (const auto occlusion = value.find("occlusionTexture"); occlusion != value.end())
        {
            const auto strength = occlusion->find("strength");
            if (strength != occlusion->end()
                && !readFloatValue(*strength, 0.0f, 1.0f, material.occlusionStrength,
                                   "glTF occlusion texture strength must be between zero and one", error))
                return false;
        }
        const auto alphaMode = value.find("alphaMode");
        if (alphaMode != value.end())
        {
            if (!alphaMode->is_string())
                return fail(error, "glTF material alphaMode must be a string");
            const auto& mode = alphaMode->get_ref<const std::string&>();
            if (mode == "OPAQUE") material.alphaMode = GlbAlphaMode::Opaque;
            else if (mode == "MASK") material.alphaMode = GlbAlphaMode::Mask;
            else if (mode == "BLEND") material.alphaMode = GlbAlphaMode::Blend;
            else return fail(error, "glTF material alphaMode is unsupported");
        }
        const auto alphaCutoff = value.find("alphaCutoff");
        if (alphaCutoff != value.end()
            && !readFloatValue(*alphaCutoff, 0.0f, std::numeric_limits<float>::max(),
                               material.alphaCutoff,
                               "glTF material alphaCutoff must be a finite non-negative float", error))
            return false;
        const auto doubleSided = value.find("doubleSided");
        if (doubleSided != value.end())
        {
            if (!doubleSided->is_boolean())
                return fail(error, "glTF material doubleSided must be a boolean");
            material.doubleSided = doubleSided->get<bool>();
        }
        document.materials.push_back(std::move(material));
    }
    return true;
}

bool addDecodedAllocation(std::size_t vertices,
                          std::size_t indices,
                          std::size_t floatValues,
                          const GlbAdmissionOptions& options,
                          GlbStaticMeshDocument& document,
                          std::string& error)
{
    if (document.decodedVertexCount > options.limits.maxDecodedVertices
        || vertices > options.limits.maxDecodedVertices - document.decodedVertexCount)
        return fail(error, "decoded glTF vertex count exceeds the admission limit");
    if (document.decodedIndexCount > options.limits.maxDecodedIndices
        || indices > options.limits.maxDecodedIndices - document.decodedIndexCount)
        return fail(error, "decoded glTF index count exceeds the admission limit");
    if (floatValues > std::numeric_limits<std::size_t>::max() / sizeof(float)
        || indices > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t))
        return fail(error, "decoded glTF allocation size overflows");
    const auto floatBytes = floatValues * sizeof(float);
    const auto indexBytes = indices * sizeof(std::uint32_t);
    if (document.decodedBytes > options.limits.maxDecodedBytes
        || floatBytes > options.limits.maxDecodedBytes - document.decodedBytes
        || indexBytes > options.limits.maxDecodedBytes - document.decodedBytes - floatBytes)
        return fail(error, "decoded glTF byte size exceeds the admission limit");
    document.decodedVertexCount += vertices;
    document.decodedIndexCount += indices;
    document.decodedBytes += floatBytes + indexBytes;
    return true;
}

bool parseMeshes(const Json& root,
                 const std::uint8_t* binBytes,
                 const GlbAdmissionOptions& options,
                 GlbStaticMeshDocument& document,
                 std::string& error)
{
    const auto meshes = root.find("meshes");
    if (meshes == root.end() || meshes->empty())
        return fail(error, "static GLB decode requires at least one mesh");
    document.meshes.reserve(meshes->size());
    for (const auto& meshValue : *meshes)
    {
        if (meshValue.find("extensions") != meshValue.end())
            return fail(error, "static GLB decode does not admit extended meshes");
        GlbMeshRecord mesh;
        if (!readOptionalName(meshValue, mesh.name, error))
            return false;
        const auto primitives = meshValue.find("primitives");
        mesh.primitives.reserve(primitives->size());
        for (const auto& primitiveValue : *primitives)
        {
            if (primitiveValue.find("extensions") != primitiveValue.end())
                return fail(error, "static GLB decode does not admit extended primitives");
            if (primitiveValue.find("targets") != primitiveValue.end()
                && !options.admitAnimations)
                return fail(error, "static GLB decode does not admit morph targets");

            std::size_t mode = 4;
            if (!readOptionalSize(primitiveValue, "mode", 4, mode, error))
                return false;
            if (mode != 4)
                return fail(error, "static GLB decode admits only TRIANGLES primitives");

            const auto attributes = primitiveValue.find("attributes");
            if (attributes == primitiveValue.end() || !attributes->is_object())
                return fail(error, "glTF mesh primitive attributes must be an object");
            for (auto attribute = attributes->begin(); attribute != attributes->end(); ++attribute)
                if (attribute.key() != "POSITION" && attribute.key() != "NORMAL"
                    && attribute.key() != "TANGENT" && attribute.key() != "TEXCOORD_0"
                    && attribute.key() != "COLOR_0"
                    && !(options.admitSkins
                         && (attribute.key().compare(0, 7, "JOINTS_") == 0
                             || attribute.key().compare(0, 8, "WEIGHTS_") == 0)))
                    return fail(error, "unsupported static glTF vertex attribute: " + attribute.key());

            GlbPrimitiveRecord primitive;
            const auto position = attributes->find("POSITION");
            if (position == attributes->end()
                || !readIndex(*position, document.accessors.size(), primitive.positionAccessor))
                return fail(error, "glTF mesh primitive POSITION accessor is missing or out of range");
            const auto normal = attributes->find("NORMAL");
            if (normal != attributes->end())
            {
                std::size_t index = 0;
                if (!readIndex(*normal, document.accessors.size(), index))
                    return fail(error, "glTF mesh primitive NORMAL accessor is out of range");
                primitive.normalAccessor = index;
            }
            const auto tangent = attributes->find("TANGENT");
            if (tangent != attributes->end())
            {
                std::size_t index = 0;
                if (!readIndex(*tangent, document.accessors.size(), index))
                    return fail(error, "glTF mesh primitive TANGENT accessor is out of range");
                primitive.tangentAccessor = index;
            }
            const auto texCoord = attributes->find("TEXCOORD_0");
            if (texCoord != attributes->end())
            {
                std::size_t index = 0;
                if (!readIndex(*texCoord, document.accessors.size(), index))
                    return fail(error, "glTF mesh primitive TEXCOORD_0 accessor is out of range");
                primitive.texCoord0Accessor = index;
            }
            const auto color = attributes->find("COLOR_0");
            if (color != attributes->end())
            {
                std::size_t index = 0;
                if (!readIndex(*color, document.accessors.size(), index))
                    return fail(error, "glTF mesh primitive COLOR_0 accessor is out of range");
                primitive.color0Accessor = index;
            }
            const auto indices = primitiveValue.find("indices");
            if (indices == primitiveValue.end()
                || !readIndex(*indices, document.accessors.size(), primitive.indexAccessor))
                return fail(error, "static GLB triangle primitives require an in-range indices accessor");
            const auto material = primitiveValue.find("material");
            if (material != primitiveValue.end())
            {
                std::size_t index = 0;
                if (!readIndex(*material, document.materials.size(), index))
                    return fail(error, "glTF mesh primitive material index is out of range");
                primitive.material = index;
            }

            const auto& positionAccessor = document.accessors[primitive.positionAccessor];
            if (positionAccessor.type != GlbAccessorType::Vec3
                || positionAccessor.componentType != 5126 || positionAccessor.normalized)
                return fail(error, "glTF POSITION accessor must be a non-normalized FLOAT VEC3");
            if (const auto target = document.bufferViews[positionAccessor.bufferView].target;
                target && *target != 34962)
                return fail(error, "glTF POSITION bufferView target must be ARRAY_BUFFER");
            const auto vertexCount = positionAccessor.count;
            const auto& indexAccessor = document.accessors[primitive.indexAccessor];
            const auto indexCount = indexAccessor.count;
            const auto& indexView = document.bufferViews[indexAccessor.bufferView];
            if (indexView.byteStride)
                return fail(error, "glTF index accessors must be tightly packed");
            if (indexView.target && *indexView.target != 34963)
                return fail(error, "glTF index bufferView target must be ELEMENT_ARRAY_BUFFER");
            if (indexAccessor.type != GlbAccessorType::Scalar || indexAccessor.normalized
                || (indexAccessor.componentType != 5121 && indexAccessor.componentType != 5123
                    && indexAccessor.componentType != 5125)
                || indexCount % 3 != 0)
                return fail(error, "glTF triangle indices must be unsigned scalar triples");

            std::size_t floatValues = 0;
            if (!checkedMultiply(vertexCount, 3, floatValues))
                return fail(error, "decoded glTF attribute count overflows");
            const auto addAttributeValues = [&](std::size_t components) {
                std::size_t values = 0;
                return checkedMultiply(vertexCount, components, values)
                    && checkedAdd(floatValues, values, floatValues);
            };
            if (primitive.normalAccessor)
            {
                const auto& accessor = document.accessors[*primitive.normalAccessor];
                if (accessor.count != vertexCount)
                    return fail(error, "glTF NORMAL count must match POSITION count");
                if (accessor.type != GlbAccessorType::Vec3
                    || accessor.componentType != 5126 || accessor.normalized)
                    return fail(error, "glTF NORMAL accessor must be a non-normalized FLOAT VEC3");
                if (const auto target = document.bufferViews[accessor.bufferView].target;
                    target && *target != 34962)
                    return fail(error, "glTF NORMAL bufferView target must be ARRAY_BUFFER");
                if (!addAttributeValues(3))
                    return fail(error, "decoded glTF attribute count overflows");
            }
            if (primitive.tangentAccessor)
            {
                const auto& accessor = document.accessors[*primitive.tangentAccessor];
                if (accessor.count != vertexCount)
                    return fail(error, "glTF TANGENT count must match POSITION count");
                if (accessor.type != GlbAccessorType::Vec4
                    || accessor.componentType != 5126 || accessor.normalized)
                    return fail(error, "glTF TANGENT accessor must be a non-normalized FLOAT VEC4");
                if (const auto target = document.bufferViews[accessor.bufferView].target;
                    target && *target != 34962)
                    return fail(error, "glTF TANGENT bufferView target must be ARRAY_BUFFER");
                if (!addAttributeValues(4))
                    return fail(error, "decoded glTF attribute count overflows");
            }
            if (primitive.texCoord0Accessor)
            {
                const auto& accessor = document.accessors[*primitive.texCoord0Accessor];
                if (accessor.count != vertexCount)
                    return fail(error, "glTF TEXCOORD_0 count must match POSITION count");
                const bool floatCoordinates = accessor.componentType == 5126 && !accessor.normalized;
                const bool byteCoordinates = accessor.componentType == 5121 && accessor.normalized;
                const bool shortCoordinates = accessor.componentType == 5123 && accessor.normalized;
                if (accessor.type != GlbAccessorType::Vec2
                    || (!floatCoordinates && !byteCoordinates && !shortCoordinates))
                    return fail(error, "glTF TEXCOORD_0 accessor format is unsupported");
                if (const auto target = document.bufferViews[accessor.bufferView].target;
                    target && *target != 34962)
                    return fail(error, "glTF TEXCOORD_0 bufferView target must be ARRAY_BUFFER");
                if (!addAttributeValues(2))
                    return fail(error, "decoded glTF attribute count overflows");
            }
            if (primitive.color0Accessor)
            {
                const auto& accessor = document.accessors[*primitive.color0Accessor];
                if (accessor.count != vertexCount)
                    return fail(error, "glTF COLOR_0 count must match POSITION count");
                const bool validType = accessor.type == GlbAccessorType::Vec3
                                    || accessor.type == GlbAccessorType::Vec4;
                const bool floatColors = accessor.componentType == 5126 && !accessor.normalized;
                const bool byteColors = accessor.componentType == 5121 && accessor.normalized;
                const bool shortColors = accessor.componentType == 5123 && accessor.normalized;
                if (!validType || (!floatColors && !byteColors && !shortColors))
                    return fail(error, "glTF COLOR_0 accessor format is unsupported");
                if (const auto target = document.bufferViews[accessor.bufferView].target;
                    target && *target != 34962)
                    return fail(error, "glTF COLOR_0 bufferView target must be ARRAY_BUFFER");
                if (!addAttributeValues(4))
                    return fail(error, "decoded glTF attribute count overflows");
            }
            if (!addDecodedAllocation(vertexCount, indexCount, floatValues,
                                      options, document, error))
                return false;

            if (!decodeFloatAttribute(document, binBytes, positionAccessor,
                                      GlbAccessorType::Vec3, primitive.positions, error))
                return false;
            if (primitive.normalAccessor
                && !decodeFloatAttribute(document, binBytes,
                                         document.accessors[*primitive.normalAccessor],
                                         GlbAccessorType::Vec3, primitive.normals, error))
                return false;
            if (primitive.tangentAccessor
                && !decodeFloatAttribute(document, binBytes,
                                         document.accessors[*primitive.tangentAccessor],
                                         GlbAccessorType::Vec4, primitive.tangents, error))
                return false;
            if (primitive.texCoord0Accessor
                && !decodeTexCoords(document, binBytes,
                                    document.accessors[*primitive.texCoord0Accessor],
                                    primitive.texCoords0, error))
                return false;
            if (primitive.color0Accessor
                && !decodeColors(document, binBytes,
                                 document.accessors[*primitive.color0Accessor],
                                 primitive.colors0, error))
                return false;
            if (!decodeIndices(document, binBytes, indexAccessor,
                               vertexCount, primitive.indices, error))
                return false;
            mesh.primitives.push_back(std::move(primitive));
        }
        document.meshes.push_back(std::move(mesh));
    }
    return true;
}

bool readFloatArray(const Json& object,
                    const char* name,
                    std::size_t expectedSize,
                    float* destination,
                    std::string& error)
{
    const auto found = object.find(name);
    if (found == object.end())
        return true;
    if (!found->is_array() || found->size() != expectedSize)
        return fail(error, std::string("glTF node ") + name + " has the wrong element count");
    for (std::size_t i = 0; i < expectedSize; ++i)
    {
        if (!(*found)[i].is_number())
            return fail(error, std::string("glTF node ") + name + " must contain numbers");
        const auto value = (*found)[i].get<double>();
        if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
            return fail(error, std::string("glTF node ") + name + " must contain finite float values");
        destination[i] = static_cast<float>(value);
    }
    return true;
}

std::array<float, 16> composeTransform(const std::array<float, 3>& translation,
                                       const std::array<float, 4>& rotation,
                                       const std::array<float, 3>& scale)
{
    const auto x = rotation[0];
    const auto y = rotation[1];
    const auto z = rotation[2];
    const auto w = rotation[3];
    const auto xx = x * x, yy = y * y, zz = z * z;
    const auto xy = x * y, xz = x * z, yz = y * z;
    const auto wx = w * x, wy = w * y, wz = w * z;
    return {
        (1.0f - 2.0f * (yy + zz)) * scale[0],
        (2.0f * (xy + wz)) * scale[0],
        (2.0f * (xz - wy)) * scale[0], 0.0f,
        (2.0f * (xy - wz)) * scale[1],
        (1.0f - 2.0f * (xx + zz)) * scale[1],
        (2.0f * (yz + wx)) * scale[1], 0.0f,
        (2.0f * (xz + wy)) * scale[2],
        (2.0f * (yz - wx)) * scale[2],
        (1.0f - 2.0f * (xx + yy)) * scale[2], 0.0f,
        translation[0], translation[1], translation[2], 1.0f
    };
}

std::array<float, 16> multiplyTransforms(const std::array<float, 16>& left,
                                         const std::array<float, 16>& right)
{
    std::array<float, 16> result {};
    for (std::size_t column = 0; column < 4; ++column)
        for (std::size_t row = 0; row < 4; ++row)
            for (std::size_t inner = 0; inner < 4; ++inner)
                result[column * 4 + row] += left[inner * 4 + row]
                                          * right[column * 4 + inner];
    return result;
}

bool readOptionalFloat(const Json& object,
                       const char* name,
                       float minimum,
                       float maximum,
                       float& destination,
                       std::string& error)
{
    const auto found = object.find(name);
    return found == object.end()
        || readFloatValue(*found, minimum, maximum, destination,
                          (std::string("glTF ") + name + " is out of range").c_str(), error);
}

bool allFinite(const std::array<float, 16>& values)
{
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

bool parseCamerasAndLights(const Json& root,
                           const GlbAdmissionOptions& options,
                           GlbStaticMeshDocument& document,
                           std::string& error)
{
    const auto cameras = root.find("cameras");
    if (cameras != root.end())
    {
        document.cameras.reserve(cameras->size());
        for (const auto& value : *cameras)
        {
            if (!value.is_object() || value.find("extensions") != value.end())
                return fail(error, "static GLB cameras must be unextended objects");
            GlbCameraRecord camera;
            if (!readOptionalName(value, camera.name, error))
                return false;
            const auto type = value.find("type");
            if (type == value.end() || !type->is_string())
                return fail(error, "glTF camera type is missing or malformed");
            if (*type == "perspective")
            {
                camera.type = GlbCameraType::Perspective;
                const auto perspective = value.find("perspective");
                if (perspective == value.end() || !perspective->is_object())
                    return fail(error, "glTF perspective camera payload is missing");
                if (!readOptionalFloat(*perspective, "aspectRatio", 0.0f,
                                       std::numeric_limits<float>::max(), camera.aspectRatio, error)
                    || !readOptionalFloat(*perspective, "yfov", 0.0f, 3.14159265358979323846f,
                                          camera.verticalFovRadians, error)
                    || !readOptionalFloat(*perspective, "znear", 0.0f,
                                          std::numeric_limits<float>::max(), camera.nearPlane, error))
                    return false;
                if (perspective->find("yfov") == perspective->end()
                    || perspective->find("znear") == perspective->end()
                    || camera.verticalFovRadians <= 0.0f || camera.nearPlane <= 0.0f
                    || (perspective->find("aspectRatio") != perspective->end()
                        && camera.aspectRatio <= 0.0f))
                    return fail(error, "glTF perspective camera requires positive yfov and znear");
                const auto farPlane = perspective->find("zfar");
                if (farPlane != perspective->end())
                {
                    float parsed = 0.0f;
                    if (!readFloatValue(*farPlane, 0.0f, std::numeric_limits<float>::max(), parsed,
                                        "glTF perspective zfar is out of range", error)
                        || parsed <= camera.nearPlane)
                        return fail(error, "glTF perspective zfar must exceed znear");
                    camera.farPlane = parsed;
                }
            }
            else if (*type == "orthographic")
            {
                camera.type = GlbCameraType::Orthographic;
                const auto orthographic = value.find("orthographic");
                if (orthographic == value.end() || !orthographic->is_object())
                    return fail(error, "glTF orthographic camera payload is missing");
                float farPlane = 0.0f;
                if (!readOptionalFloat(*orthographic, "xmag", 0.0f,
                                       std::numeric_limits<float>::max(), camera.xMagnification, error)
                    || !readOptionalFloat(*orthographic, "ymag", 0.0f,
                                          std::numeric_limits<float>::max(), camera.yMagnification, error)
                    || !readOptionalFloat(*orthographic, "znear", 0.0f,
                                          std::numeric_limits<float>::max(), camera.nearPlane, error)
                    || !readOptionalFloat(*orthographic, "zfar", 0.0f,
                                          std::numeric_limits<float>::max(), farPlane, error))
                    return false;
                if (orthographic->find("xmag") == orthographic->end()
                    || orthographic->find("ymag") == orthographic->end()
                    || orthographic->find("znear") == orthographic->end()
                    || orthographic->find("zfar") == orthographic->end()
                    || camera.xMagnification <= 0.0f || camera.yMagnification <= 0.0f
                    || farPlane <= camera.nearPlane)
                    return fail(error, "glTF orthographic camera requires positive magnification and zfar above znear");
                camera.farPlane = farPlane;
            }
            else
                return fail(error, "glTF camera type is unsupported");
            document.cameras.push_back(std::move(camera));
        }
    }

    const auto extensions = root.find("extensions");
    if (extensions == root.end())
        return true;
    if (!extensions->is_object())
        return fail(error, "glTF root extensions must be an object");
    for (auto extension = extensions->begin(); extension != extensions->end(); ++extension)
        if (extension.key() != "KHR_lights_punctual")
            return fail(error, "unsupported static glTF root extension: " + extension.key());
    const auto punctual = extensions->find("KHR_lights_punctual");
    if (punctual == extensions->end())
        return true;
    if (std::find(document.metadata.extensionsUsed.begin(),
                  document.metadata.extensionsUsed.end(), "KHR_lights_punctual")
        == document.metadata.extensionsUsed.end())
        return fail(error, "KHR_lights_punctual payload is not listed in extensionsUsed");
    if (!punctual->is_object())
        return fail(error, "KHR_lights_punctual root payload must be an object");
    const auto lights = punctual->find("lights");
    if (lights == punctual->end() || !lights->is_array()
        || lights->size() > options.limits.maxLights)
        return fail(error, "KHR_lights_punctual lights are missing or exceed the admission limit");
    document.lights.reserve(lights->size());
    for (const auto& value : *lights)
    {
        if (!value.is_object() || value.find("extensions") != value.end())
            return fail(error, "static GLB punctual lights must be unextended objects");
        GlbLightRecord light;
        if (!readOptionalName(value, light.name, error)
            || !readFactorArray(value, "color", light.color, error))
            return false;
        const auto type = value.find("type");
        if (type == value.end() || !type->is_string())
            return fail(error, "glTF punctual light type is missing or malformed");
        if (*type == "directional") light.type = GlbLightType::Directional;
        else if (*type == "point") light.type = GlbLightType::Point;
        else if (*type == "spot") light.type = GlbLightType::Spot;
        else return fail(error, "glTF punctual light type is unsupported");
        if (!readOptionalFloat(value, "intensity", 0.0f,
                               std::numeric_limits<float>::max(), light.intensity, error))
            return false;
        const auto range = value.find("range");
        if (range != value.end())
        {
            float parsed = 0.0f;
            if (!readFloatValue(*range, 0.0f, std::numeric_limits<float>::max(), parsed,
                                "glTF punctual light range is out of range", error)
                || parsed <= 0.0f)
                return fail(error, "glTF punctual light range must be positive");
            light.range = parsed;
        }
        if (light.type == GlbLightType::Spot)
        {
            const auto spot = value.find("spot");
            if (spot != value.end() && !spot->is_object())
                return fail(error, "glTF spot light payload must be an object");
            if (spot != value.end()
                && (!readOptionalFloat(*spot, "innerConeAngle", 0.0f,
                                       1.57079632679489661923f, light.innerConeAngle, error)
                    || !readOptionalFloat(*spot, "outerConeAngle", 0.0f,
                                          1.57079632679489661923f, light.outerConeAngle, error)))
                return false;
            if (light.innerConeAngle >= light.outerConeAngle
                || light.outerConeAngle <= 0.0f)
                return fail(error, "glTF spot light cone angles are invalid");
        }
        document.lights.push_back(std::move(light));
    }
    return true;
}

bool parseNodesAndScenes(const Json& root,
                         const GlbAdmissionOptions& options,
                         GlbStaticMeshDocument& document,
                         std::string& error)
{
    const auto nodes = root.find("nodes");
    if (nodes != root.end())
    {
        document.nodes.reserve(nodes->size());
        for (const auto& value : *nodes)
        {
            GlbNodeRecord node;
            if (!readOptionalName(value, node.name, error))
                return false;
            const auto mesh = value.find("mesh");
            if (mesh != value.end())
            {
                std::size_t index = 0;
                if (!readIndex(*mesh, document.meshes.size(), index))
                    return fail(error, "glTF node mesh index is out of range");
                node.mesh = index;
            }
            const auto camera = value.find("camera");
            if (camera != value.end())
            {
                std::size_t index = 0;
                if (!readIndex(*camera, document.cameras.size(), index))
                    return fail(error, "glTF node camera index is out of range");
                node.camera = index;
            }
            if (value.find("skin") != value.end())
                return fail(error, "static GLB decode does not admit skinned nodes");

            const auto nodeExtensions = value.find("extensions");
            if (nodeExtensions != value.end())
            {
                if (!nodeExtensions->is_object() || nodeExtensions->size() != 1)
                    return fail(error, "static GLB nodes admit only KHR_lights_punctual extensions");
                const auto punctual = nodeExtensions->find("KHR_lights_punctual");
                if (punctual == nodeExtensions->end() || !punctual->is_object())
                    return fail(error, "static GLB node extension is unsupported");
                const auto light = punctual->find("light");
                std::size_t index = 0;
                if (light == punctual->end() || !readIndex(*light, document.lights.size(), index))
                    return fail(error, "glTF node punctual light index is missing or out of range");
                node.light = index;
            }

            const auto children = value.find("children");
            if (children != value.end())
                for (const auto& child : *children)
                {
                    std::size_t index = 0;
                    if (!readIndex(child, document.metadata.nodes, index))
                        return fail(error, "glTF node child index is out of range");
                    node.children.push_back(index);
                }

            const bool hasMatrix = value.find("matrix") != value.end();
            const bool hasTrs = value.find("translation") != value.end()
                             || value.find("rotation") != value.end()
                             || value.find("scale") != value.end();
            if (hasMatrix && hasTrs)
                return fail(error, "glTF node must not combine matrix and TRS transforms");
            if (hasMatrix)
            {
                if (!readFloatArray(value, "matrix", 16, node.localTransform.data(), error))
                    return false;
            }
            else
            {
                std::array<float, 3> translation {0.0f, 0.0f, 0.0f};
                std::array<float, 4> rotation {0.0f, 0.0f, 0.0f, 1.0f};
                std::array<float, 3> scale {1.0f, 1.0f, 1.0f};
                if (!readFloatArray(value, "translation", 3, translation.data(), error)
                    || !readFloatArray(value, "rotation", 4, rotation.data(), error)
                    || !readFloatArray(value, "scale", 3, scale.data(), error))
                    return false;
                const auto normSquared = rotation[0] * rotation[0] + rotation[1] * rotation[1]
                                       + rotation[2] * rotation[2] + rotation[3] * rotation[3];
                if (!std::isfinite(normSquared) || std::abs(normSquared - 1.0f) > 0.001f)
                    return fail(error, "glTF node rotation must be a normalized quaternion");
                node.localTransform = composeTransform(translation, rotation, scale);
            }
            if (!allFinite(node.localTransform))
                return fail(error, "glTF node transform exceeds the finite float range");
            document.nodes.push_back(std::move(node));
        }
    }

    for (std::size_t parent = 0; parent < document.nodes.size(); ++parent)
        for (const auto child : document.nodes[parent].children)
            document.nodes[child].parent = parent;
    std::vector<std::size_t> stack;
    for (std::size_t rootNode = 0; rootNode < document.nodes.size(); ++rootNode)
    {
        if (document.nodes[rootNode].parent)
            continue;
        document.nodes[rootNode].worldTransform = document.nodes[rootNode].localTransform;
        stack.push_back(rootNode);
        while (!stack.empty())
        {
            const auto parent = stack.back();
            stack.pop_back();
            for (const auto child : document.nodes[parent].children)
            {
                document.nodes[child].worldTransform = multiplyTransforms(
                    document.nodes[parent].worldTransform, document.nodes[child].localTransform);
                if (!allFinite(document.nodes[child].worldTransform))
                    return fail(error, "glTF node world transform exceeds the finite float range");
                stack.push_back(child);
            }
        }
    }

    const auto scenes = root.find("scenes");
    if (scenes == root.end() || scenes->empty())
        return fail(error, "static GLB decode requires at least one scene");
    document.scenes.reserve(scenes->size());
    for (const auto& value : *scenes)
    {
        if (value.find("extensions") != value.end())
            return fail(error, "static GLB decode does not admit extended scenes");
        GlbSceneRecord scene;
        if (!readOptionalName(value, scene.name, error))
            return false;
        const auto roots = value.find("nodes");
        if (roots != value.end())
            for (const auto& rootNode : *roots)
            {
                std::size_t index = 0;
                if (!readIndex(rootNode, document.nodes.size(), index))
                    return fail(error, "glTF scene node index is out of range");
                if (document.nodes[index].parent)
                    return fail(error, "glTF scene root node must not have a parent");
                scene.rootNodes.push_back(index);
            }
        document.scenes.push_back(std::move(scene));
    }
    document.selectedScene = options.sceneIndex.value_or(
        document.metadata.defaultScene.value_or(0));
    if (document.selectedScene >= document.scenes.size())
        return fail(error, "requested glTF scene index is out of range");
    return true;
}
} // namespace

std::optional<GlbMetadata> admitGlbMetadata(const std::uint8_t* bytes,
                                             std::size_t size,
                                             const GlbAdmissionOptions& options,
                                             std::string& error)
{
    error.clear();
    if (size > options.limits.maxContainerBytes)
    {
        fail(error, "GLB container exceeds the admission byte limit");
        return std::nullopt;
    }
    if (bytes == nullptr || size < 20)
    {
        fail(error, "GLB container is truncated");
        return std::nullopt;
    }
    if (readU32(bytes) != glbMagic)
    {
        fail(error, "GLB magic must be glTF");
        return std::nullopt;
    }
    if (readU32(bytes + 4) != 2)
    {
        fail(error, "GLB container version must be 2");
        return std::nullopt;
    }
    if (readU32(bytes + 8) != size)
    {
        fail(error, "GLB declared length must exactly match the input length");
        return std::nullopt;
    }

    GlbMetadata metadata;
    metadata.containerBytes = size;
    const std::uint8_t* jsonBytes = nullptr;
    std::size_t offset = 12;
    std::size_t chunkIndex = 0;
    bool sawBin = false;
    while (offset < size)
    {
        if (size - offset < 8)
        {
            fail(error, "GLB chunk header is truncated");
            return std::nullopt;
        }
        const auto chunkLength = static_cast<std::size_t>(readU32(bytes + offset));
        const auto chunkType = readU32(bytes + offset + 4);
        offset += 8;
        if ((chunkLength & 3u) != 0)
        {
            fail(error, "GLB chunk length must be four-byte aligned");
            return std::nullopt;
        }
        if (chunkLength > size - offset)
        {
            fail(error, "GLB chunk exceeds the declared container length");
            return std::nullopt;
        }
        if (chunkIndex == 0)
        {
            if (chunkType != jsonChunkType)
            {
                fail(error, "GLB first chunk must be JSON");
                return std::nullopt;
            }
            if (chunkLength == 0 || chunkLength > options.limits.maxJsonBytes)
            {
                fail(error, "GLB JSON chunk is empty or exceeds the admission byte limit");
                return std::nullopt;
            }
            jsonBytes = bytes + offset;
            metadata.jsonBytes = chunkLength;
        }
        else if (chunkIndex == 1 && chunkType == binChunkType)
        {
            if (sawBin || chunkLength > options.limits.maxBinBytes)
            {
                fail(error, "GLB BIN chunk exceeds the admission contract");
                return std::nullopt;
            }
            sawBin = true;
            metadata.binBytes = chunkLength;
        }
        else
        {
            fail(error, "GLB admission supports only one JSON chunk followed by at most one BIN chunk");
            return std::nullopt;
        }
        offset += chunkLength;
        ++chunkIndex;
    }
    if (offset != size || chunkIndex == 0 || jsonBytes == nullptr)
    {
        fail(error, "GLB chunk table does not exactly cover the container");
        return std::nullopt;
    }
    if (!checkJsonNesting(jsonBytes, metadata.jsonBytes,
                          options.limits.maxJsonNestingDepth, error))
        return std::nullopt;

    const auto root = Json::parse(jsonBytes, jsonBytes + metadata.jsonBytes,
                                  nullptr, false, true);
    if (root.is_discarded())
    {
        fail(error, "GLB JSON chunk is not valid JSON");
        return std::nullopt;
    }
    if (!validateMetadata(root, options, metadata, error))
        return std::nullopt;
    return metadata;
}

std::optional<GlbMetadata> admitGlbMetadataFile(std::string_view path,
                                                 const GlbAdmissionOptions& options,
                                                 std::string& error)
{
    error.clear();
    const std::string pathString(path);
    std::ifstream input(pathString, std::ios::binary | std::ios::ate);
    if (!input)
    {
        fail(error, "could not open GLB file");
        return std::nullopt;
    }
    const auto end = input.tellg();
    if (end < 0 || static_cast<std::uint64_t>(end) > options.limits.maxContainerBytes)
    {
        fail(error, "GLB file exceeds the admission byte limit");
        return std::nullopt;
    }
    const auto size = static_cast<std::size_t>(end);
    std::vector<std::uint8_t> bytes(size);
    input.seekg(0, std::ios::beg);
    if (size != 0)
        input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    if (!input || static_cast<std::size_t>(input.gcount()) != size)
    {
        fail(error, "could not read the complete GLB file");
        return std::nullopt;
    }
    if (input.peek() != std::char_traits<char>::eof())
    {
        fail(error, "GLB file changed while it was being read");
        return std::nullopt;
    }
    return admitGlbMetadata(bytes, options, error);
}

static std::optional<GlbStaticMeshDocument> decodeGlbBaseScene(
    const std::uint8_t* bytes,
    std::size_t size,
    const GlbAdmissionOptions& options,
    bool animatedScene,
    std::string& error)
{
    const auto admitted = admitGlbMetadata(bytes, size, options, error);
    if (!admitted)
        return std::nullopt;
    if (! animatedScene && admitted->animations != 0)
    {
        fail(error, "static GLB decode does not admit animations");
        return std::nullopt;
    }
    if (! animatedScene && admitted->skins != 0)
    {
        fail(error, "static GLB decode does not admit skins");
        return std::nullopt;
    }

    const auto jsonLength = static_cast<std::size_t>(readU32(bytes + 12));
    const auto* jsonBytes = bytes + 20;
    const auto root = Json::parse(jsonBytes, jsonBytes + jsonLength, nullptr, false, true);
    const auto binHeaderOffset = 20u + jsonLength;
    const std::uint8_t* binBytes = nullptr;
    std::size_t binSize = 0;
    if (binHeaderOffset < size)
    {
        binSize = static_cast<std::size_t>(readU32(bytes + binHeaderOffset));
        binBytes = bytes + binHeaderOffset + 8;
    }

    GlbStaticMeshDocument document;
    document.metadata = *admitted;
    for (const auto& extension : document.metadata.extensionsRequired)
        if (extension != "KHR_lights_punctual")
        {
            fail(error, "static GLB decode does not implement required glTF extension: "
                        + extension);
            return std::nullopt;
        }
    if (!parseBufferViewsAndAccessors(root, binBytes, binSize, document, error)
        || !parseMaterials(root, binBytes, options, document, error)
        || !parseMeshes(root, binBytes, options, document, error)
        || !parseCamerasAndLights(root, options, document, error)
        || !parseNodesAndScenes(root, options, document, error))
        return std::nullopt;
    error.clear();
    return document;
}

std::optional<GlbStaticMeshDocument> decodeStaticGlb(
    const std::uint8_t* bytes,
    std::size_t size,
    const GlbAdmissionOptions& options,
    std::string& error)
{
    return decodeGlbBaseScene(bytes, size, options, false, error);
}

std::optional<GlbStaticMeshDocument> decodeAnimatedGlbBaseScene(
    const std::uint8_t* bytes,
    std::size_t size,
    const GlbAdmissionOptions& options,
    std::string& error)
{
    if (! options.admitAnimations || ! options.admitSkins)
    {
        fail(error, "animated GLB base-scene decode requires explicit animation and skin admission");
        return std::nullopt;
    }
    return decodeGlbBaseScene(bytes, size, options, true, error);
}

namespace
{
struct AnimationNodeRecord
{
    std::optional<std::size_t> parent;
    std::optional<std::size_t> mesh;
    std::optional<std::size_t> skin;
    bool hasMatrix = false;
    std::array<float, 3> translation {0.0f, 0.0f, 0.0f};
    std::array<float, 4> rotation {0.0f, 0.0f, 0.0f, 1.0f};
    std::array<float, 3> scale {1.0f, 1.0f, 1.0f};
};

struct DecodeByteBudget
{
    std::size_t used = 0;
    std::size_t limit = 0;

    bool add(std::size_t count, std::size_t elementBytes, std::string& error)
    {
        std::size_t bytes = 0;
        if (!checkedMultiply(count, elementBytes, bytes)
            || used > limit || bytes > limit - used)
            return fail(error, "decoded glTF animation or deformation bytes exceed the admission limit");
        used += bytes;
        return true;
    }
};

struct OwnedSkin
{
    visualdeformation::SkinId id;
    std::vector<visualdeformation::JointView> joints;
    std::vector<float> inverseBindMatrices;
    std::vector<std::size_t> jointNodes;
};

struct OwnedJointWeightSet
{
    std::vector<std::uint32_t> joints;
    std::vector<float> weights;
};

struct OwnedMorphTarget
{
    visualdeformation::MorphTargetId id;
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> tangents;
};

struct OwnedDeformationMesh
{
    visualdeformation::MeshId id;
    visualdeformation::SkinId skin;
    std::size_t vertexCount = 0;
    std::vector<OwnedJointWeightSet> jointWeightSets;
    std::vector<OwnedMorphTarget> morphTargets;
};

struct AnimationSamplerRecord
{
    std::size_t input = 0;
    std::size_t output = 0;
    visualanimation::Interpolation interpolation = visualanimation::Interpolation::Linear;
};

struct OwnedAnimationTrack
{
    visualanimation::TrackId id;
    visualanimation::TargetId target;
    visualanimation::Channel channel = visualanimation::Channel::Translation;
    visualanimation::Interpolation interpolation = visualanimation::Interpolation::Linear;
    std::vector<double> times;
    std::vector<float> values;
    std::size_t morphWeightCount = 0;
};

GlbLimits boundedGlbLimits(const GlbLimits& requested)
{
    const GlbLimits hard;
    GlbLimits result;
    result.maxContainerBytes = std::min(requested.maxContainerBytes, hard.maxContainerBytes);
    result.maxJsonBytes = std::min(requested.maxJsonBytes, hard.maxJsonBytes);
    result.maxBinBytes = std::min(requested.maxBinBytes, hard.maxBinBytes);
    result.maxJsonNestingDepth = std::min(requested.maxJsonNestingDepth,
                                          hard.maxJsonNestingDepth);
    result.maxScenes = std::min(requested.maxScenes, hard.maxScenes);
    result.maxNodes = std::min(requested.maxNodes, hard.maxNodes);
    result.maxNodeDepth = std::min(requested.maxNodeDepth, hard.maxNodeDepth);
    result.maxMeshes = std::min(requested.maxMeshes, hard.maxMeshes);
    result.maxPrimitives = std::min(requested.maxPrimitives, hard.maxPrimitives);
    result.maxMaterials = std::min(requested.maxMaterials, hard.maxMaterials);
    result.maxTextures = std::min(requested.maxTextures, hard.maxTextures);
    result.maxImages = std::min(requested.maxImages, hard.maxImages);
    result.maxSamplers = std::min(requested.maxSamplers, hard.maxSamplers);
    result.maxCameras = std::min(requested.maxCameras, hard.maxCameras);
    result.maxLights = std::min(requested.maxLights, hard.maxLights);
    result.maxAccessors = std::min(requested.maxAccessors, hard.maxAccessors);
    result.maxBufferViews = std::min(requested.maxBufferViews, hard.maxBufferViews);
    result.maxDecodedVertices = std::min(requested.maxDecodedVertices,
                                         hard.maxDecodedVertices);
    result.maxDecodedIndices = std::min(requested.maxDecodedIndices,
                                        hard.maxDecodedIndices);
    result.maxDecodedBytes = std::min(requested.maxDecodedBytes, hard.maxDecodedBytes);
    result.maxEmbeddedImageBytes = std::min(requested.maxEmbeddedImageBytes,
                                            hard.maxEmbeddedImageBytes);
    result.maxImageWidth = std::min(requested.maxImageWidth, hard.maxImageWidth);
    result.maxImageHeight = std::min(requested.maxImageHeight, hard.maxImageHeight);
    result.maxDecodedImageBytes = std::min(requested.maxDecodedImageBytes,
                                           hard.maxDecodedImageBytes);
    return result;
}

GlbAnimationDecodeLimits boundedAnimationDecodeLimits(
    const GlbAnimationDecodeLimits& requested)
{
    const GlbAnimationDecodeLimits hard;
    return {
        std::min(requested.maxAnimations, hard.maxAnimations),
        std::min(requested.maxSamplersPerAnimation, hard.maxSamplersPerAnimation),
        std::min(requested.maxChannelsPerAnimation, hard.maxChannelsPerAnimation),
        std::min(requested.maxTotalSamplers, hard.maxTotalSamplers),
        std::min(requested.maxTotalChannels, hard.maxTotalChannels),
        std::min(requested.maxClipNameBytes, hard.maxClipNameBytes),
        std::min(requested.maxTotalClipNameBytes, hard.maxTotalClipNameBytes),
        std::min(requested.maxDecodedBytes, hard.maxDecodedBytes)
    };
}

visualanimation::Limits boundedAnimationContractLimits(
    const visualanimation::Limits& requested)
{
    const visualanimation::Limits hard;
    return {
        std::min(requested.maxTracks, hard.maxTracks),
        std::min(requested.maxKeysPerTrack, hard.maxKeysPerTrack),
        std::min(requested.maxTotalKeys, hard.maxTotalKeys),
        std::min(requested.maxMorphWeightsPerTrack, hard.maxMorphWeightsPerTrack),
        std::min(requested.maxTotalValues, hard.maxTotalValues)
    };
}

visualdeformation::Limits boundedDeformationContractLimits(
    const visualdeformation::Limits& requested)
{
    const visualdeformation::Limits hard;
    visualdeformation::Limits result;
    result.maxSkins = std::min(requested.maxSkins, hard.maxSkins);
    result.maxMeshes = std::min(requested.maxMeshes, hard.maxMeshes);
    result.maxJointsPerSkin = std::min(requested.maxJointsPerSkin,
                                       hard.maxJointsPerSkin);
    result.maxTotalJoints = std::min(requested.maxTotalJoints, hard.maxTotalJoints);
    result.maxVerticesPerMesh = std::min(requested.maxVerticesPerMesh,
                                         hard.maxVerticesPerMesh);
    result.maxTotalVertices = std::min(requested.maxTotalVertices,
                                       hard.maxTotalVertices);
    result.maxInfluenceSetsPerMesh = std::min(requested.maxInfluenceSetsPerMesh,
                                               hard.maxInfluenceSetsPerMesh);
    result.maxMorphTargetsPerMesh = std::min(requested.maxMorphTargetsPerMesh,
                                              hard.maxMorphTargetsPerMesh);
    result.maxTotalMorphTargets = std::min(requested.maxTotalMorphTargets,
                                            hard.maxTotalMorphTargets);
    result.maxTotalAccessorValues = std::min(requested.maxTotalAccessorValues,
                                              hard.maxTotalAccessorValues);
    const auto boundedFloat = [] (float value, float maximum)
    {
        return std::isfinite(value) && value >= 0.0f ? std::min(value, maximum) : 0.0f;
    };
    result.weightSumTolerance = boundedFloat(requested.weightSumTolerance,
                                             hard.weightSumTolerance);
    result.maxInverseBindElementMagnitude = boundedFloat(
        requested.maxInverseBindElementMagnitude, hard.maxInverseBindElementMagnitude);
    result.maxMorphDeltaMagnitude = boundedFloat(requested.maxMorphDeltaMagnitude,
                                                  hard.maxMorphDeltaMagnitude);
    result.maxMorphWeightMagnitude = boundedFloat(requested.maxMorphWeightMagnitude,
                                                   hard.maxMorphWeightMagnitude);
    return result;
}

std::uint64_t oneBasedId(std::size_t index)
{
    return static_cast<std::uint64_t>(index) + 1u;
}

visualdeformation::MorphTargetId morphTargetId(std::size_t mesh,
                                               std::size_t target)
{
    return { (oneBasedId(mesh) << 32u) | oneBasedId(target) };
}

bool accessorHasVertexTarget(const GlbStaticMeshDocument& storage,
                             const GlbAccessorRecord& accessor)
{
    const auto target = storage.bufferViews[accessor.bufferView].target;
    return !target || *target == 34962;
}

bool validateFloatAccessor(const GlbStaticMeshDocument& storage,
                           const GlbAccessorRecord& accessor,
                           GlbAccessorType type,
                           std::size_t count,
                           const char* diagnostic,
                           std::string& error)
{
    if (accessor.componentType != 5126 || accessor.normalized
        || accessor.type != type || accessor.count != count
        || !accessorHasVertexTarget(storage, accessor))
        return fail(error, diagnostic);
    return true;
}

bool appendFloatAccessor(const GlbStaticMeshDocument& storage,
                         const std::uint8_t* binBytes,
                         const GlbAccessorRecord& accessor,
                         std::size_t components,
                         float magnitude,
                         std::vector<float>& destination,
                         DecodeByteBudget& budget,
                         std::string& error)
{
    std::size_t valueCount = 0;
    if (!checkedMultiply(accessor.count, components, valueCount)
        || !budget.add(valueCount, sizeof(float), error))
        return false;
    const auto oldSize = destination.size();
    if (oldSize > std::numeric_limits<std::size_t>::max() - valueCount)
        return fail(error, "decoded glTF float accessor size overflows");
    destination.resize(oldSize + valueCount);
    const auto* source = accessorData(storage, binBytes, accessor);
    const auto stride = accessorStride(storage, accessor);
    for (std::size_t element = 0; element < accessor.count; ++element)
        for (std::size_t component = 0; component < components; ++component)
        {
            const auto value = readFloat(source + element * stride + component * sizeof(float));
            if (!std::isfinite(value) || std::abs(value) > magnitude)
                return fail(error, "glTF float accessor contains a nonfinite or out-of-range value");
            destination[oldSize + element * components + component] = value;
        }
    return true;
}

bool appendJointAccessor(const GlbStaticMeshDocument& storage,
                         const std::uint8_t* binBytes,
                         const GlbAccessorRecord& accessor,
                         std::vector<std::uint32_t>& destination,
                         DecodeByteBudget& budget,
                         std::string& error)
{
    if (accessor.type != GlbAccessorType::Vec4 || accessor.normalized
        || (accessor.componentType != 5121 && accessor.componentType != 5123)
        || !accessorHasVertexTarget(storage, accessor))
        return fail(error, "glTF JOINTS accessor must be an unsigned non-normalized VEC4");
    std::size_t valueCount = 0;
    if (!checkedMultiply(accessor.count, 4, valueCount)
        || !budget.add(valueCount, sizeof(std::uint32_t), error))
        return false;
    const auto oldSize = destination.size();
    if (oldSize > std::numeric_limits<std::size_t>::max() - valueCount)
        return fail(error, "decoded glTF joint accessor size overflows");
    destination.resize(oldSize + valueCount);
    const auto* source = accessorData(storage, binBytes, accessor);
    const auto stride = accessorStride(storage, accessor);
    for (std::size_t element = 0; element < accessor.count; ++element)
        for (std::size_t component = 0; component < 4; ++component)
        {
            std::uint32_t value = 0;
            if (accessor.componentType == 5121)
                value = source[element * stride + component];
            else
            {
                std::uint16_t encoded = 0;
                std::memcpy(&encoded,
                            source + element * stride + component * sizeof(encoded),
                            sizeof(encoded));
                value = encoded;
            }
            destination[oldSize + element * 4 + component] = value;
        }
    return true;
}

bool appendWeightAccessor(const GlbStaticMeshDocument& storage,
                          const std::uint8_t* binBytes,
                          const GlbAccessorRecord& accessor,
                          std::vector<float>& destination,
                          DecodeByteBudget& budget,
                          std::string& error)
{
    const bool floats = accessor.componentType == 5126 && !accessor.normalized;
    const bool bytes = accessor.componentType == 5121 && accessor.normalized;
    const bool shorts = accessor.componentType == 5123 && accessor.normalized;
    if (accessor.type != GlbAccessorType::Vec4 || (!floats && !bytes && !shorts)
        || !accessorHasVertexTarget(storage, accessor))
        return fail(error, "glTF WEIGHTS accessor must be FLOAT or normalized unsigned VEC4 data");
    std::size_t valueCount = 0;
    if (!checkedMultiply(accessor.count, 4, valueCount)
        || !budget.add(valueCount, sizeof(float), error))
        return false;
    const auto oldSize = destination.size();
    if (oldSize > std::numeric_limits<std::size_t>::max() - valueCount)
        return fail(error, "decoded glTF weight accessor size overflows");
    destination.resize(oldSize + valueCount);
    const auto* source = accessorData(storage, binBytes, accessor);
    const auto stride = accessorStride(storage, accessor);
    for (std::size_t element = 0; element < accessor.count; ++element)
        for (std::size_t component = 0; component < 4; ++component)
        {
            float value = 0.0f;
            if (floats)
                value = readFloat(source + element * stride + component * sizeof(float));
            else if (bytes)
                value = static_cast<float>(source[element * stride + component]) / 255.0f;
            else
            {
                std::uint16_t encoded = 0;
                std::memcpy(&encoded,
                            source + element * stride + component * sizeof(encoded),
                            sizeof(encoded));
                value = static_cast<float>(encoded) / 65535.0f;
            }
            if (!std::isfinite(value) || value < 0.0f)
                return fail(error, "glTF WEIGHTS accessor contains a negative or nonfinite value");
            destination[oldSize + element * 4 + component] = value;
        }
    return true;
}

bool readBoundedWeights(const Json& object,
                        const char* name,
                        std::size_t expected,
                        float magnitude,
                        std::string& error)
{
    const auto found = object.find(name);
    if (found == object.end())
        return true;
    if (!found->is_array() || found->size() != expected)
        return fail(error, std::string("glTF ") + name
                           + " cardinality does not match the mesh morph targets");
    for (const auto& item : *found)
    {
        if (!item.is_number())
            return fail(error, std::string("glTF ") + name + " must contain numbers");
        const auto value = item.get<double>();
        if (!std::isfinite(value) || std::abs(value) > magnitude)
            return fail(error, std::string("glTF ") + name
                               + " contains a nonfinite or out-of-range value");
    }
    return true;
}

bool validateAnimationNodeExtensions(const Json& root,
                                     const Json& node,
                                     std::string& error)
{
    const auto extensions = node.find("extensions");
    if (extensions == node.end())
        return true;
    if (!extensions->is_object() || extensions->size() != 1)
        return fail(error, "animated glTF node extensions are unsupported");
    const auto punctual = extensions->find("KHR_lights_punctual");
    if (punctual == extensions->end() || !punctual->is_object() || punctual->size() != 1)
        return fail(error, "animated glTF node extensions admit only KHR_lights_punctual");
    const auto light = punctual->find("light");
    const auto rootExtensions = root.find("extensions");
    if (light == punctual->end() || rootExtensions == root.end()
        || !rootExtensions->is_object())
        return fail(error, "animated glTF punctual-light node binding is malformed");
    const auto rootPunctual = rootExtensions->find("KHR_lights_punctual");
    if (rootPunctual == rootExtensions->end() || !rootPunctual->is_object())
        return fail(error, "animated glTF punctual-light table is missing");
    const auto lights = rootPunctual->find("lights");
    std::size_t lightIndex = 0;
    if (lights == rootPunctual->end() || !lights->is_array()
        || !readIndex(*light, lights->size(), lightIndex))
        return fail(error, "animated glTF punctual-light index is out of range");
    return true;
}

bool parseAnimationNodes(const Json& root,
                         std::size_t meshCount,
                         std::size_t skinCount,
                         float morphWeightMagnitude,
                         std::vector<AnimationNodeRecord>& nodes,
                         std::vector<std::optional<std::size_t>>& meshSkins,
                         std::string& error)
{
    const auto found = root.find("nodes");
    if (found == root.end())
        return true;
    nodes.resize(found->size());
    meshSkins.resize(meshCount);
    std::vector<bool> sawMeshBinding(meshCount, false);
    for (std::size_t index = 0; index < found->size(); ++index)
    {
        const auto& value = (*found)[index];
        if (!value.is_object())
            return fail(error, "animated glTF nodes must be objects");
        if (!validateAnimationNodeExtensions(root, value, error))
            return false;
        auto& node = nodes[index];
        const auto mesh = value.find("mesh");
        if (mesh != value.end())
        {
            std::size_t meshIndex = 0;
            if (!readIndex(*mesh, meshCount, meshIndex))
                return fail(error, "animated glTF node mesh index is out of range");
            node.mesh = meshIndex;
        }
        const auto skin = value.find("skin");
        if (skin != value.end())
        {
            std::size_t skinIndex = 0;
            if (!readIndex(*skin, skinCount, skinIndex))
                return fail(error, "animated glTF node skin index is out of range");
            if (!node.mesh)
                return fail(error, "animated glTF node declares a skin without a mesh");
            node.skin = skinIndex;
        }

        node.hasMatrix = value.find("matrix") != value.end();
        const bool hasTrs = value.find("translation") != value.end()
                         || value.find("rotation") != value.end()
                         || value.find("scale") != value.end();
        if (node.hasMatrix && hasTrs)
            return fail(error, "animated glTF node must not combine matrix and TRS transforms");
        if (node.hasMatrix)
        {
            std::array<float, 16> matrix {};
            if (!readFloatArray(value, "matrix", matrix.size(), matrix.data(), error))
                return false;
        }
        else
        {
            if (!readFloatArray(value, "translation", node.translation.size(), node.translation.data(), error)
                || !readFloatArray(value, "rotation", node.rotation.size(), node.rotation.data(), error)
                || !readFloatArray(value, "scale", node.scale.size(), node.scale.data(), error))
                return false;
            const double norm = static_cast<double>(node.rotation[0]) * node.rotation[0]
                              + static_cast<double>(node.rotation[1]) * node.rotation[1]
                              + static_cast<double>(node.rotation[2]) * node.rotation[2]
                              + static_cast<double>(node.rotation[3]) * node.rotation[3];
            if (!std::isfinite(norm) || std::abs(norm - 1.0) > 0.001)
                return fail(error, "animated glTF node rotation must be a normalized quaternion");
        }

        if (node.mesh)
        {
            if (sawMeshBinding[*node.mesh] && meshSkins[*node.mesh] != node.skin)
                return fail(error, "glTF mesh instances declare conflicting skin bindings");
            sawMeshBinding[*node.mesh] = true;
            meshSkins[*node.mesh] = node.skin;
        }
    }
    for (std::size_t parent = 0; parent < nodes.size(); ++parent)
    {
        const auto children = (*found)[parent].find("children");
        if (children == (*found)[parent].end())
            continue;
        for (const auto& childValue : *children)
        {
            std::size_t child = 0;
            if (!readIndex(childValue, nodes.size(), child))
                return fail(error, "animated glTF node child index is out of range");
            nodes[child].parent = parent;
        }
    }
    (void)morphWeightMagnitude;
    return true;
}

bool parseSkins(const Json& root,
                const GlbStaticMeshDocument& storage,
                const std::uint8_t* binBytes,
                const std::vector<AnimationNodeRecord>& nodes,
                const visualdeformation::Limits& limits,
                DecodeByteBudget& budget,
                std::vector<OwnedSkin>& skins,
                std::string& error)
{
    const auto found = root.find("skins");
    if (found == root.end())
        return true;
    if (found->size() > limits.maxSkins)
        return fail(error, "skin capacity exceeded");

    std::size_t totalJoints = 0;
    skins.reserve(found->size());
    for (std::size_t skinIndex = 0; skinIndex < found->size(); ++skinIndex)
    {
        const auto& value = (*found)[skinIndex];
        if (!value.is_object() || value.find("extensions") != value.end())
            return fail(error, "glTF skins must be unextended objects");
        const auto joints = value.find("joints");
        if (joints == value.end() || !joints->is_array() || joints->empty()
            || joints->size() > limits.maxJointsPerSkin
            || totalJoints > limits.maxTotalJoints
            || joints->size() > limits.maxTotalJoints - totalJoints)
            return fail(error, "skin joint capacity exceeded");
        totalJoints += joints->size();

        const auto skeleton = value.find("skeleton");
        if (skeleton != value.end())
        {
            std::size_t skeletonNode = 0;
            if (!readIndex(*skeleton, nodes.size(), skeletonNode))
                return fail(error, "glTF skin skeleton node index is out of range");
        }

        OwnedSkin skin;
        skin.id = { oneBasedId(skinIndex) };
        std::vector<std::size_t> jointNodes;
        jointNodes.reserve(joints->size());
        std::unordered_set<std::size_t> unique;
        for (const auto& jointValue : *joints)
        {
            std::size_t jointNode = 0;
            if (!readIndex(jointValue, nodes.size(), jointNode))
                return fail(error, "glTF skin joint node index is out of range");
            if (!unique.insert(jointNode).second)
                return fail(error, "glTF skin contains a duplicate joint node");
            jointNodes.push_back(jointNode);
        }

        skin.jointNodes = jointNodes;
        skin.joints.reserve(jointNodes.size());
        for (const auto jointNode : jointNodes)
        {
            visualdeformation::JointId parent;
            auto ancestor = nodes[jointNode].parent;
            while (ancestor && unique.count(*ancestor) == 0)
                ancestor = nodes[*ancestor].parent;
            if (ancestor)
                parent = { oneBasedId(*ancestor) };
            skin.joints.push_back({ { oneBasedId(jointNode) }, parent });
        }

        std::size_t matrixValues = 0;
        if (!checkedMultiply(jointNodes.size(), 16, matrixValues))
            return fail(error, "inverse bind matrix accessor cardinality overflows");
        const auto matrices = value.find("inverseBindMatrices");
        if (matrices == value.end())
        {
            if (!budget.add(matrixValues, sizeof(float), error))
                return false;
            skin.inverseBindMatrices.assign(matrixValues, 0.0f);
            for (std::size_t joint = 0; joint < jointNodes.size(); ++joint)
                for (std::size_t diagonal = 0; diagonal < 4; ++diagonal)
                    skin.inverseBindMatrices[joint * 16 + diagonal * 5] = 1.0f;
        }
        else
        {
            std::size_t accessorIndex = 0;
            if (!readIndex(*matrices, storage.accessors.size(), accessorIndex))
                return fail(error, "glTF inverseBindMatrices accessor index is out of range");
            const auto& accessor = storage.accessors[accessorIndex];
            if (storage.bufferViews[accessor.bufferView].target
                || !validateFloatAccessor(storage, accessor, GlbAccessorType::Mat4,
                                          jointNodes.size(),
                                          "glTF inverseBindMatrices must be a FLOAT MAT4 accessor matching the joints",
                                          error)
                || !appendFloatAccessor(storage, binBytes, accessor, 16,
                                        limits.maxInverseBindElementMagnitude,
                                        skin.inverseBindMatrices, budget, error))
                return false;
        }
        skins.push_back(std::move(skin));
    }
    return true;
}

bool parseAttributeSetIndex(const std::string& semantic,
                            const char* prefix,
                            std::size_t& index)
{
    const std::string prefixString(prefix);
    if (semantic.compare(0, prefixString.size(), prefixString) != 0)
        return false;
    if (semantic.size() == prefixString.size())
        return false;
    std::size_t value = 0;
    for (std::size_t character = prefixString.size(); character < semantic.size(); ++character)
    {
        const auto digit = semantic[character];
        if (digit < '0' || digit > '9'
            || value > (std::numeric_limits<std::size_t>::max()
                        - static_cast<std::size_t>(digit - '0')) / 10)
            return false;
        value = value * 10 + static_cast<std::size_t>(digit - '0');
    }
    index = value;
    return true;
}

bool appendOptionalMorphAccessor(const Json& target,
                                 const char* semantic,
                                 std::size_t priorVertices,
                                 std::size_t vertexCount,
                                 const GlbStaticMeshDocument& storage,
                                 const std::uint8_t* binBytes,
                                 float magnitude,
                                 std::vector<float>& destination,
                                 DecodeByteBudget& budget,
                                 std::string& error)
{
    const auto found = target.find(semantic);
    if (found == target.end())
    {
        if (destination.empty())
            return true;
        std::size_t zeros = 0;
        if (!checkedMultiply(vertexCount, 3, zeros)
            || !budget.add(zeros, sizeof(float), error))
            return false;
        destination.resize(destination.size() + zeros, 0.0f);
        return true;
    }

    std::size_t accessorIndex = 0;
    if (!readIndex(*found, storage.accessors.size(), accessorIndex))
        return fail(error, std::string("glTF morph ") + semantic
                           + " accessor index is out of range");
    const auto& accessor = storage.accessors[accessorIndex];
    if (!validateFloatAccessor(storage, accessor, GlbAccessorType::Vec3, vertexCount,
                               "glTF morph target accessor must be a FLOAT VEC3 matching POSITION",
                               error))
        return false;
    if (destination.empty() && priorVertices != 0)
    {
        std::size_t zeros = 0;
        if (!checkedMultiply(priorVertices, 3, zeros)
            || !budget.add(zeros, sizeof(float), error))
            return false;
        destination.assign(zeros, 0.0f);
    }
    return appendFloatAccessor(storage, binBytes, accessor, 3, magnitude,
                               destination, budget, error);
}

bool parseDeformationMeshes(const Json& root,
                            const GlbStaticMeshDocument& storage,
                            const std::uint8_t* binBytes,
                            const std::vector<std::optional<std::size_t>>& meshSkins,
                            const std::vector<AnimationNodeRecord>& nodes,
                            const visualdeformation::Limits& limits,
                            DecodeByteBudget& budget,
                            std::vector<OwnedDeformationMesh>& meshes,
                            std::vector<std::size_t>& meshTargetCounts,
                            std::string& error)
{
    const auto found = root.find("meshes");
    if (found == root.end())
        return true;
    meshTargetCounts.assign(found->size(), 0);
    std::size_t totalVertices = 0;
    std::size_t totalTargets = 0;

    for (std::size_t meshIndex = 0; meshIndex < found->size(); ++meshIndex)
    {
        const auto& meshValue = (*found)[meshIndex];
        if (!meshValue.is_object() || meshValue.find("extensions") != meshValue.end())
            return fail(error, "deformed glTF meshes must be unextended objects");
        const auto primitives = meshValue.find("primitives");
        if (primitives == meshValue.end() || !primitives->is_array()
            || primitives->empty())
            return fail(error, "deformed glTF mesh primitives must be a non-empty array");

        const auto firstTargets = (*primitives)[0].find("targets");
        const std::size_t targetCount = firstTargets == (*primitives)[0].end()
            ? 0 : firstTargets->is_array() ? firstTargets->size()
                                           : std::numeric_limits<std::size_t>::max();
        if (targetCount == std::numeric_limits<std::size_t>::max())
            return fail(error, "glTF primitive morph targets must be an array");
        meshTargetCounts[meshIndex] = targetCount;
        if (targetCount > limits.maxMorphTargetsPerMesh
            || totalTargets > limits.maxTotalMorphTargets
            || targetCount > limits.maxTotalMorphTargets - totalTargets)
            return fail(error, "mesh morph target capacity exceeded");
        totalTargets += targetCount;
        if (!readBoundedWeights(meshValue, "weights", targetCount,
                                limits.maxMorphWeightMagnitude, error))
            return false;

        const bool hasSkin = meshIndex < meshSkins.size() && meshSkins[meshIndex].has_value();
        if (!hasSkin && targetCount == 0)
            continue;
        if (meshes.size() >= limits.maxMeshes)
            return fail(error, "deformed mesh capacity exceeded");

        OwnedDeformationMesh mesh;
        mesh.id = { oneBasedId(meshIndex) };
        if (hasSkin)
            mesh.skin = { oneBasedId(*meshSkins[meshIndex]) };
        mesh.morphTargets.resize(targetCount);
        for (std::size_t target = 0; target < targetCount; ++target)
            mesh.morphTargets[target].id = morphTargetId(meshIndex, target);

        std::optional<std::size_t> influenceSetCount;
        for (const auto& primitive : *primitives)
        {
            if (!primitive.is_object() || primitive.find("extensions") != primitive.end())
                return fail(error, "deformed glTF primitives must be unextended objects");
            const auto attributes = primitive.find("attributes");
            if (attributes == primitive.end() || !attributes->is_object())
                return fail(error, "deformed glTF primitive attributes must be an object");
            const auto position = attributes->find("POSITION");
            std::size_t positionAccessorIndex = 0;
            if (position == attributes->end()
                || !readIndex(*position, storage.accessors.size(), positionAccessorIndex))
                return fail(error, "deformed glTF primitive POSITION accessor is missing or out of range");
            const auto& positionAccessor = storage.accessors[positionAccessorIndex];
            if (positionAccessor.componentType != 5126 || positionAccessor.normalized
                || positionAccessor.type != GlbAccessorType::Vec3
                || !accessorHasVertexTarget(storage, positionAccessor))
                return fail(error, "deformed glTF POSITION accessor must be a FLOAT VEC3");
            const auto vertexCount = positionAccessor.count;
            if (vertexCount == 0 || vertexCount > limits.maxVerticesPerMesh
                || mesh.vertexCount > limits.maxVerticesPerMesh - vertexCount
                || totalVertices > limits.maxTotalVertices
                || vertexCount > limits.maxTotalVertices - totalVertices)
                return fail(error, "deformed mesh vertex capacity exceeded");
            const auto priorVertices = mesh.vertexCount;
            mesh.vertexCount += vertexCount;
            totalVertices += vertexCount;

            std::unordered_map<std::size_t, std::size_t> jointAccessors;
            std::unordered_map<std::size_t, std::size_t> weightAccessors;
            for (auto attribute = attributes->begin(); attribute != attributes->end(); ++attribute)
            {
                std::size_t setIndex = 0;
                const auto& semantic = attribute.key();
                if (semantic.compare(0, 7, "JOINTS_") == 0)
                {
                    if (!parseAttributeSetIndex(semantic, "JOINTS_", setIndex)
                        || setIndex >= limits.maxInfluenceSetsPerMesh
                        || !readIndex(attribute.value(), storage.accessors.size(),
                                      jointAccessors[setIndex]))
                        return fail(error, "glTF JOINTS attribute semantic or accessor is malformed");
                }
                else if (semantic.compare(0, 8, "WEIGHTS_") == 0)
                {
                    if (!parseAttributeSetIndex(semantic, "WEIGHTS_", setIndex)
                        || setIndex >= limits.maxInfluenceSetsPerMesh
                        || !readIndex(attribute.value(), storage.accessors.size(),
                                      weightAccessors[setIndex]))
                        return fail(error, "glTF WEIGHTS attribute semantic or accessor is malformed");
                }
            }
            if (jointAccessors.size() != weightAccessors.size())
                return fail(error, "glTF JOINTS and WEIGHTS attribute sets do not match");
            const auto setCount = jointAccessors.size();
            for (std::size_t set = 0; set < setCount; ++set)
                if (jointAccessors.count(set) == 0 || weightAccessors.count(set) == 0)
                    return fail(error, "glTF JOINTS and WEIGHTS attribute sets must be contiguous");
            if (hasSkin && setCount == 0)
                return fail(error, "skinned glTF mesh has no JOINTS and WEIGHTS attributes");
            if (!hasSkin && setCount != 0)
                return fail(error, "unskinned glTF mesh declares JOINTS or WEIGHTS attributes");
            if (influenceSetCount && *influenceSetCount != setCount)
                return fail(error, "glTF mesh primitives have conflicting influence set counts");
            influenceSetCount = setCount;
            if (mesh.jointWeightSets.empty())
                mesh.jointWeightSets.resize(setCount);
            for (std::size_t set = 0; set < setCount; ++set)
            {
                const auto& jointAccessor = storage.accessors[jointAccessors[set]];
                const auto& weightAccessor = storage.accessors[weightAccessors[set]];
                if (jointAccessor.count != vertexCount || weightAccessor.count != vertexCount
                    || !appendJointAccessor(storage, binBytes, jointAccessor,
                                            mesh.jointWeightSets[set].joints, budget, error)
                    || !appendWeightAccessor(storage, binBytes, weightAccessor,
                                             mesh.jointWeightSets[set].weights, budget, error))
                    return false;
            }

            const auto targets = primitive.find("targets");
            const auto primitiveTargetCount = targets == primitive.end() ? 0
                : targets->is_array() ? targets->size()
                                      : std::numeric_limits<std::size_t>::max();
            if (primitiveTargetCount != targetCount)
                return fail(error, "glTF mesh primitives have conflicting morph target counts");
            for (std::size_t targetIndex = 0; targetIndex < targetCount; ++targetIndex)
            {
                const auto& target = (*targets)[targetIndex];
                if (!target.is_object() || target.find("extensions") != target.end())
                    return fail(error, "glTF morph targets must be unextended objects");
                for (auto attribute = target.begin(); attribute != target.end(); ++attribute)
                    if (attribute.key() != "POSITION" && attribute.key() != "NORMAL"
                        && attribute.key() != "TANGENT")
                        return fail(error, "unsupported glTF morph target attribute: "
                                           + attribute.key());
                auto& output = mesh.morphTargets[targetIndex];
                const auto positionDelta = target.find("POSITION");
                std::size_t deltaAccessorIndex = 0;
                if (positionDelta == target.end()
                    || !readIndex(*positionDelta, storage.accessors.size(), deltaAccessorIndex))
                    return fail(error, "glTF morph target POSITION accessor is missing or out of range");
                const auto& deltaAccessor = storage.accessors[deltaAccessorIndex];
                if (!validateFloatAccessor(storage, deltaAccessor, GlbAccessorType::Vec3,
                                           vertexCount,
                                           "glTF morph POSITION must be a FLOAT VEC3 matching POSITION",
                                           error)
                    || !appendFloatAccessor(storage, binBytes, deltaAccessor, 3,
                                            limits.maxMorphDeltaMagnitude,
                                            output.positions, budget, error)
                    || !appendOptionalMorphAccessor(target, "NORMAL", priorVertices,
                                                    vertexCount, storage, binBytes,
                                                    limits.maxMorphDeltaMagnitude,
                                                    output.normals, budget, error)
                    || !appendOptionalMorphAccessor(target, "TANGENT", priorVertices,
                                                    vertexCount, storage, binBytes,
                                                    limits.maxMorphDeltaMagnitude,
                                                    output.tangents, budget, error))
                    return false;
            }
        }
        meshes.push_back(std::move(mesh));
    }

    const auto nodeValues = root.find("nodes");
    if (nodeValues != root.end())
        for (std::size_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex)
            if (nodes[nodeIndex].mesh
                && !readBoundedWeights((*nodeValues)[nodeIndex], "weights",
                                       meshTargetCounts[*nodes[nodeIndex].mesh],
                                       limits.maxMorphWeightMagnitude, error))
                return false;
    return true;
}

bool validateAnimationAccessor(const GlbStaticMeshDocument& storage,
                               const GlbAccessorRecord& accessor,
                               GlbAccessorType type,
                               std::size_t count,
                               const char* diagnostic,
                               std::string& error)
{
    if (accessor.componentType != 5126 || accessor.normalized
        || accessor.type != type || accessor.count != count
        || storage.bufferViews[accessor.bufferView].target)
        return fail(error, diagnostic);
    return true;
}

bool decodeAnimationTimes(const GlbStaticMeshDocument& storage,
                          const std::uint8_t* binBytes,
                          const GlbAccessorRecord& accessor,
                          std::vector<double>& times,
                          DecodeByteBudget& budget,
                          std::string& error)
{
    if (!validateAnimationAccessor(storage, accessor, GlbAccessorType::Scalar,
                                   accessor.count,
                                   "glTF animation input must be an untargeted FLOAT SCALAR accessor",
                                   error)
        || !budget.add(accessor.count, sizeof(double), error))
        return false;
    times.resize(accessor.count);
    const auto* source = accessorData(storage, binBytes, accessor);
    const auto stride = accessorStride(storage, accessor);
    for (std::size_t key = 0; key < accessor.count; ++key)
    {
        const auto value = readFloat(source + key * stride);
        if (!std::isfinite(value) || value < 0.0f
            || (key != 0 && value <= times[key - 1]))
            return fail(error, "glTF animation input times must be finite, nonnegative, and strictly increasing");
        times[key] = value;
    }
    return true;
}

bool decodeAnimationValues(const GlbStaticMeshDocument& storage,
                           const std::uint8_t* binBytes,
                           const GlbAccessorRecord& accessor,
                           GlbAccessorType type,
                           std::size_t count,
                           std::size_t components,
                           float magnitude,
                           std::vector<float>& values,
                           DecodeByteBudget& budget,
                           std::string& error)
{
    if (!validateAnimationAccessor(storage, accessor, type, count,
                                   "glTF animation output accessor has an invalid shape, component type, count, or target",
                                   error))
        return false;
    return appendFloatAccessor(storage, binBytes, accessor, components, magnitude,
                               values, budget, error);
}

bool parseAnimations(const Json& root,
                     const GlbStaticMeshDocument& storage,
                     const std::uint8_t* binBytes,
                     const std::vector<AnimationNodeRecord>& nodes,
                     const std::vector<std::size_t>& meshTargetCounts,
                     const GlbAnimationDecodeLimits& decodeLimits,
                     const visualanimation::Limits& contractLimits,
                     float morphWeightMagnitude,
                     DecodeByteBudget& budget,
                     std::vector<std::string>& names,
                     std::vector<std::vector<OwnedAnimationTrack>>& clips,
                     std::vector<double>& durations,
                     std::string& error)
{
    const auto animations = root.find("animations");
    if (animations == root.end())
        return true;
    if (!animations->is_array() || animations->size() > decodeLimits.maxAnimations)
        return fail(error, "glTF animation capacity exceeded");

    std::size_t totalSamplers = 0;
    std::size_t totalChannels = 0;
    std::size_t totalNameBytes = 0;
    names.reserve(animations->size());
    clips.reserve(animations->size());
    durations.reserve(animations->size());
    for (std::size_t animationIndex = 0; animationIndex < animations->size(); ++animationIndex)
    {
        const auto& animation = (*animations)[animationIndex];
        if (!animation.is_object() || animation.find("extensions") != animation.end())
            return fail(error, "glTF animations must be unextended objects");
        std::string name;
        if (!readOptionalName(animation, name, error))
            return false;
        if (name.size() > decodeLimits.maxClipNameBytes
            || totalNameBytes > decodeLimits.maxTotalClipNameBytes
            || name.size() > decodeLimits.maxTotalClipNameBytes - totalNameBytes)
            return fail(error, "glTF animation name bytes exceed the admission limit");
        totalNameBytes += name.size();

        const auto samplerValues = animation.find("samplers");
        const auto channelValues = animation.find("channels");
        if (samplerValues == animation.end() || !samplerValues->is_array()
            || samplerValues->empty()
            || samplerValues->size() > decodeLimits.maxSamplersPerAnimation
            || totalSamplers > decodeLimits.maxTotalSamplers
            || samplerValues->size() > decodeLimits.maxTotalSamplers - totalSamplers)
            return fail(error, "glTF animation sampler capacity exceeded");
        if (channelValues == animation.end() || !channelValues->is_array()
            || channelValues->empty()
            || channelValues->size() > decodeLimits.maxChannelsPerAnimation
            || channelValues->size() > contractLimits.maxTracks
            || totalChannels > decodeLimits.maxTotalChannels
            || channelValues->size() > decodeLimits.maxTotalChannels - totalChannels)
            return fail(error, "glTF animation channel capacity exceeded");
        totalSamplers += samplerValues->size();
        totalChannels += channelValues->size();

        std::vector<AnimationSamplerRecord> samplers;
        samplers.reserve(samplerValues->size());
        for (const auto& value : *samplerValues)
        {
            if (!value.is_object() || value.find("extensions") != value.end())
                return fail(error, "glTF animation samplers must be unextended objects");
            AnimationSamplerRecord sampler;
            const auto input = value.find("input");
            const auto output = value.find("output");
            if (input == value.end() || !readIndex(*input, storage.accessors.size(), sampler.input)
                || output == value.end() || !readIndex(*output, storage.accessors.size(), sampler.output))
                return fail(error, "glTF animation sampler accessor index is missing or out of range");
            const auto interpolation = value.find("interpolation");
            if (interpolation != value.end())
            {
                if (!interpolation->is_string())
                    return fail(error, "glTF animation sampler interpolation is malformed");
                const auto& mode = interpolation->get_ref<const std::string&>();
                if (mode == "STEP") sampler.interpolation = visualanimation::Interpolation::Step;
                else if (mode == "LINEAR") sampler.interpolation = visualanimation::Interpolation::Linear;
                else return fail(error, "glTF animation sampler interpolation is unsupported");
            }
            const auto& inputAccessor = storage.accessors[sampler.input];
            if (!validateAnimationAccessor(storage, inputAccessor, GlbAccessorType::Scalar,
                                           inputAccessor.count,
                                           "glTF animation input must be an untargeted FLOAT SCALAR accessor",
                                           error))
                return false;
            samplers.push_back(sampler);
        }

        std::vector<bool> usedSamplers(samplers.size(), false);
        std::unordered_set<std::uint64_t> targets;
        std::vector<OwnedAnimationTrack> tracks;
        tracks.reserve(channelValues->size());
        double duration = 0.0;
        for (std::size_t channelIndex = 0; channelIndex < channelValues->size(); ++channelIndex)
        {
            const auto& value = (*channelValues)[channelIndex];
            if (!value.is_object() || value.find("extensions") != value.end())
                return fail(error, "glTF animation channels must be unextended objects");
            std::size_t samplerIndex = 0;
            const auto samplerValue = value.find("sampler");
            if (samplerValue == value.end()
                || !readIndex(*samplerValue, samplers.size(), samplerIndex))
                return fail(error, "glTF animation channel sampler index is missing or out of range");
            usedSamplers[samplerIndex] = true;
            const auto target = value.find("target");
            if (target == value.end() || !target->is_object()
                || target->find("extensions") != target->end())
                return fail(error, "glTF animation channel target is missing, malformed, or extended");
            std::size_t nodeIndex = 0;
            const auto node = target->find("node");
            const auto path = target->find("path");
            if (node == target->end() || !readIndex(*node, nodes.size(), nodeIndex)
                || path == target->end() || !path->is_string())
                return fail(error, "glTF animation channel target node or path is missing or malformed");
            if (nodes[nodeIndex].hasMatrix)
                return fail(error, "glTF animation cannot target a node that declares a matrix transform");

            OwnedAnimationTrack track;
            track.id = { oneBasedId(channelIndex) };
            track.target = { oneBasedId(nodeIndex) };
            track.interpolation = samplers[samplerIndex].interpolation;
            GlbAccessorType outputType = GlbAccessorType::Vec3;
            std::size_t width = 3;
            float magnitude = std::numeric_limits<float>::max();
            const auto& pathName = path->get_ref<const std::string&>();
            if (pathName == "translation") track.channel = visualanimation::Channel::Translation;
            else if (pathName == "rotation")
            {
                track.channel = visualanimation::Channel::Rotation;
                outputType = GlbAccessorType::Vec4;
                width = 4;
            }
            else if (pathName == "scale") track.channel = visualanimation::Channel::Scale;
            else if (pathName == "weights")
            {
                track.channel = visualanimation::Channel::MorphWeights;
                if (!nodes[nodeIndex].mesh
                    || *nodes[nodeIndex].mesh >= meshTargetCounts.size()
                    || meshTargetCounts[*nodes[nodeIndex].mesh] == 0)
                    return fail(error, "glTF morph animation target has no mesh morph targets");
                track.morphWeightCount = meshTargetCounts[*nodes[nodeIndex].mesh];
                if (track.morphWeightCount > contractLimits.maxMorphWeightsPerTrack)
                    return fail(error, "animation morph weight capacity exceeded");
                outputType = GlbAccessorType::Scalar;
                width = track.morphWeightCount;
                magnitude = morphWeightMagnitude;
            }
            else
                return fail(error, "glTF animation channel target path is unsupported");

            const auto targetKey = (oneBasedId(nodeIndex) << 8u)
                                 | static_cast<std::uint64_t>(track.channel);
            if (!targets.insert(targetKey).second)
                return fail(error, "glTF animation has duplicate target-channel tracks");

            const auto& sampler = samplers[samplerIndex];
            const auto& inputAccessor = storage.accessors[sampler.input];
            const auto& outputAccessor = storage.accessors[sampler.output];
            std::size_t expectedOutputCount = inputAccessor.count;
            if (track.channel == visualanimation::Channel::MorphWeights
                && !checkedMultiply(inputAccessor.count, track.morphWeightCount,
                                    expectedOutputCount))
                return fail(error, "glTF morph animation output cardinality overflows");
            if (!decodeAnimationTimes(storage, binBytes, inputAccessor, track.times,
                                      budget, error)
                || !decodeAnimationValues(storage, binBytes, outputAccessor, outputType,
                                          expectedOutputCount,
                                          track.channel == visualanimation::Channel::MorphWeights
                                              ? 1 : width,
                                          magnitude,
                                          track.values, budget, error))
                return false;
            if (track.channel == visualanimation::Channel::Rotation)
                for (std::size_t key = 0; key < track.times.size(); ++key)
                {
                    constexpr double maxQuaternionNormSquaredError = 0.001;
                    double normSquared = 0.0;
                    for (std::size_t component = 0; component < 4; ++component)
                    {
                        const auto componentValue = track.values[key * 4 + component];
                        normSquared += static_cast<double>(componentValue) * componentValue;
                    }
                    if (!std::isfinite(normSquared)
                        || std::abs(normSquared - 1.0) > maxQuaternionNormSquaredError)
                        return fail(error, "glTF animation rotation output contains a non-normalized quaternion");
                }
            duration = std::max(duration, track.times.back());
            tracks.push_back(std::move(track));
        }
        if (std::find(usedSamplers.begin(), usedSamplers.end(), false) != usedSamplers.end())
            return fail(error, "glTF animation contains an unused sampler");
        names.push_back(std::move(name));
        clips.push_back(std::move(tracks));
        durations.push_back(duration);
    }
    return true;
}

bool makeDeformationViews(const std::vector<OwnedSkin>& skins,
                          const std::vector<OwnedDeformationMesh>& meshes,
                          std::vector<visualdeformation::SkinView>& skinViews,
                          std::vector<std::vector<visualdeformation::JointWeightSetView>>& setViews,
                          std::vector<std::vector<visualdeformation::MorphTargetView>>& targetViews,
                          std::vector<visualdeformation::MeshView>& meshViews)
{
    skinViews.reserve(skins.size());
    for (const auto& skin : skins)
        skinViews.push_back({ skin.id, skin.joints.data(), skin.joints.size(),
                              skin.inverseBindMatrices.data(), skin.inverseBindMatrices.size() });
    setViews.resize(meshes.size());
    targetViews.resize(meshes.size());
    meshViews.reserve(meshes.size());
    for (std::size_t meshIndex = 0; meshIndex < meshes.size(); ++meshIndex)
    {
        const auto& mesh = meshes[meshIndex];
        auto& sets = setViews[meshIndex];
        sets.reserve(mesh.jointWeightSets.size());
        for (const auto& set : mesh.jointWeightSets)
            sets.push_back({ set.joints.data(), set.joints.size(),
                             set.weights.data(), set.weights.size() });
        auto& targets = targetViews[meshIndex];
        targets.reserve(mesh.morphTargets.size());
        for (const auto& target : mesh.morphTargets)
            targets.push_back({ target.id, target.positions.data(), target.positions.size(),
                                target.normals.empty() ? nullptr : target.normals.data(),
                                target.normals.size(),
                                target.tangents.empty() ? nullptr : target.tangents.data(),
                                target.tangents.size() });
        meshViews.push_back({ mesh.id, mesh.vertexCount, mesh.skin,
                              sets.empty() ? nullptr : sets.data(), sets.size(),
                              targets.empty() ? nullptr : targets.data(), targets.size() });
    }
    return true;
}

} // namespace

std::optional<GlbAnimationDocument> detail::decodeGlbAnimationsWithFactories(
    const std::uint8_t* bytes,
    std::size_t size,
    const GlbAnimationDecodeOptions& requestedOptions,
    AnimationClipFactory animationFactory,
    DeformationAssetFactory deformationFactory,
    std::string& error)
{
    error.clear();
    try
    {
        GlbAnimationDecodeOptions options = requestedOptions;
        options.admission.limits = boundedGlbLimits(requestedOptions.admission.limits);
        options.admission.admitAnimations = true;
        options.admission.admitSkins = true;
        options.limits = boundedAnimationDecodeLimits(requestedOptions.limits);
        options.animationLimits = boundedAnimationContractLimits(requestedOptions.animationLimits);
        options.deformationLimits = boundedDeformationContractLimits(
            requestedOptions.deformationLimits);
        if (!animationFactory || !deformationFactory)
        {
            fail(error, "glTF animation decode requires valid immutable contract factories");
            return std::nullopt;
        }

        const auto metadata = admitGlbMetadata(bytes, size, options.admission, error);
        if (!metadata)
            return std::nullopt;
        for (const auto& extension : metadata->extensionsRequired)
            if (extension != "KHR_lights_punctual")
            {
                fail(error, "animation GLB decode does not implement required glTF extension: "
                            + extension);
                return std::nullopt;
            }

        const auto jsonLength = static_cast<std::size_t>(readU32(bytes + 12));
        const auto* jsonBytes = bytes + 20;
        const auto root = Json::parse(jsonBytes, jsonBytes + jsonLength, nullptr, false, true);
        if (root.is_discarded())
        {
            fail(error, "GLB JSON chunk is not valid JSON");
            return std::nullopt;
        }
        const auto binHeaderOffset = 20u + jsonLength;
        const std::uint8_t* binBytes = nullptr;
        std::size_t binSize = 0;
        if (binHeaderOffset < size)
        {
            binSize = static_cast<std::size_t>(readU32(bytes + binHeaderOffset));
            binBytes = bytes + binHeaderOffset + 8;
        }

        GlbStaticMeshDocument storage;
        storage.metadata = *metadata;
        if (!parseBufferViewsAndAccessors(root, binBytes, binSize, storage, error))
            return std::nullopt;

        std::vector<AnimationNodeRecord> nodes;
        std::vector<std::optional<std::size_t>> meshSkins;
        if (!parseAnimationNodes(root, metadata->meshes, metadata->skins,
                                 options.deformationLimits.maxMorphWeightMagnitude,
                                 nodes, meshSkins, error))
            return std::nullopt;

        DecodeByteBudget budget;
        budget.limit = std::min(options.limits.maxDecodedBytes,
                                options.admission.limits.maxDecodedBytes);
        std::size_t deformationValueBytes = 0;
        if (checkedMultiply(options.deformationLimits.maxTotalAccessorValues,
                            sizeof(float), deformationValueBytes))
            budget.limit = std::min(budget.limit, deformationValueBytes);

        std::vector<OwnedSkin> skins;
        if (!parseSkins(root, storage, binBytes, nodes, options.deformationLimits,
                        budget, skins, error))
            return std::nullopt;
        std::vector<OwnedDeformationMesh> meshes;
        std::vector<std::size_t> meshTargetCounts;
        if (!parseDeformationMeshes(root, storage, binBytes, meshSkins, nodes,
                                    options.deformationLimits, budget, meshes,
                                    meshTargetCounts, error))
            return std::nullopt;

        std::vector<std::string> names;
        std::vector<std::vector<OwnedAnimationTrack>> ownedClips;
        std::vector<double> durations;
        if (!parseAnimations(root, storage, binBytes, nodes, meshTargetCounts,
                             options.limits, options.animationLimits,
                             options.deformationLimits.maxMorphWeightMagnitude,
                             budget, names, ownedClips, durations, error))
            return std::nullopt;
        if (ownedClips.empty() && meshes.empty())
        {
            fail(error, skins.empty()
                ? "GLB contains no animation, skinning, or morph deformation data"
                : "GLB skins are not bound to any decoded deformation mesh");
            return std::nullopt;
        }

        GlbAnimationDocument result;
        result.metadata = *metadata;
        if (!meshes.empty())
        {
            std::vector<visualdeformation::SkinView> skinViews;
            std::vector<std::vector<visualdeformation::JointWeightSetView>> setViews;
            std::vector<std::vector<visualdeformation::MorphTargetView>> targetViews;
            std::vector<visualdeformation::MeshView> meshViews;
            makeDeformationViews(skins, meshes, skinViews, setViews, targetViews, meshViews);
            visualdeformation::AssetView assetView;
            assetView.skins = skinViews.empty() ? nullptr : skinViews.data();
            assetView.skinCount = skinViews.size();
            assetView.meshes = meshViews.data();
            assetView.meshCount = meshViews.size();
            result.deformation = deformationFactory(assetView, options.deformationLimits, error);
            if (!result.deformation)
            {
                if (error.empty())
                    fail(error, "immutable deformation factory rejected decoded GLB data");
                return std::nullopt;
            }

            const auto meshValues = root.find("meshes");
            const auto nodeValues = root.find("nodes");
            for (const auto& mesh : meshes)
            {
                const auto meshIndex = static_cast<std::size_t>(mesh.id.value - 1u);
                std::optional<std::size_t> nodeIndex;
                for (std::size_t index = 0; index < nodes.size(); ++index)
                    if (nodes[index].mesh == meshIndex)
                    {
                        if (nodeIndex) { nodeIndex.reset(); break; }
                        nodeIndex = index;
                    }
                if (!nodeIndex || nodes[*nodeIndex].hasMatrix) continue;

                GlbDeformationRenderBinding binding;
                binding.mesh = mesh.id;
                binding.nodeIndex = *nodeIndex;
                const auto copyWeights = [&] (const Json& value)
                {
                    const auto weights = value.find("weights");
                    if (weights == value.end()) return;
                    binding.morphBaseWeights.clear();
                    for (const auto& item : *weights)
                        binding.morphBaseWeights.push_back(static_cast<float>(item.get<double>()));
                };
                if (meshValues != root.end()) copyWeights((*meshValues)[meshIndex]);
                if (nodeValues != root.end()) copyWeights((*nodeValues)[*nodeIndex]);

                if (mesh.skin.isValid())
                {
                    const auto skinIndex = static_cast<std::size_t>(mesh.skin.value - 1u);
                    if (skinIndex >= skins.size()) continue;
                    bool supported = true;
                    for (std::size_t joint = 0; joint < skins[skinIndex].joints.size(); ++joint)
                    {
                        const auto baseNode = skins[skinIndex].jointNodes[joint];
                        if (nodes[baseNode].hasMatrix) { supported = false; break; }
                        binding.jointBaseTransforms.push_back({
                            skins[skinIndex].id, skins[skinIndex].joints[joint].id,
                            nodes[baseNode].translation, nodes[baseNode].rotation,
                            nodes[baseNode].scale });
                    }
                    if (!supported) continue;
                }
                result.renderBindings.push_back(std::move(binding));
            }
        }

        result.clips.reserve(ownedClips.size());
        for (std::size_t clipIndex = 0; clipIndex < ownedClips.size(); ++clipIndex)
        {
            const auto& ownedTracks = ownedClips[clipIndex];
            std::vector<visualanimation::TrackView> trackViews;
            trackViews.reserve(ownedTracks.size());
            for (const auto& track : ownedTracks)
                trackViews.push_back({ track.id, track.target, track.channel,
                                       track.interpolation, track.times.data(),
                                       track.values.data(), track.times.size(),
                                       track.values.size(), track.morphWeightCount });
            visualanimation::ClipView clipView;
            clipView.id = { oneBasedId(clipIndex) };
            clipView.durationSeconds = durations[clipIndex];
            clipView.tracks = trackViews.data();
            clipView.trackCount = trackViews.size();
            auto clip = animationFactory(clipView, options.animationLimits, error);
            if (!clip)
            {
                if (error.empty())
                    fail(error, "immutable animation factory rejected decoded GLB data");
                return std::nullopt;
            }
            GlbNamedAnimationClip named;
            named.name = std::move(names[clipIndex]);
            named.clip = std::move(clip);
            std::unordered_map<std::uint64_t, GlbJointAnimationBinding> jointBindings;
            std::unordered_set<std::uint64_t> morphTargets;
            std::unordered_set<std::uint64_t> morphMeshes;
            for (const auto& track : ownedTracks)
            {
                if (track.channel == visualanimation::Channel::MorphWeights)
                {
                    const auto nodeIndex = static_cast<std::size_t>(track.target.value - 1u);
                    const auto meshIndex = *nodes[nodeIndex].mesh;
                    if (!morphTargets.insert(track.target.value).second
                        || !morphMeshes.insert(oneBasedId(meshIndex)).second)
                    {
                        fail(error, "glTF animation binds one morph target or deformation mesh more than once");
                        return std::nullopt;
                    }
                    GlbMorphAnimationBinding binding;
                    binding.animationTarget = track.target;
                    binding.mesh = { oneBasedId(meshIndex) };
                    binding.targets.reserve(meshTargetCounts[meshIndex]);
                    for (std::size_t target = 0; target < meshTargetCounts[meshIndex]; ++target)
                        binding.targets.push_back(morphTargetId(meshIndex, target));
                    named.morphBindings.push_back(std::move(binding));
                }
                else
                {
                    std::optional<GlbJointAnimationBinding> binding;
                    for (const auto& skin : skins)
                        for (const auto& joint : skin.joints)
                            if (joint.id.value == track.target.value)
                            {
                                if (binding)
                                {
                                    fail(error, "glTF animated joint belongs to more than one decoded skin");
                                    return std::nullopt;
                                }
                                binding = GlbJointAnimationBinding {
                                    track.target, skin.id, joint.id
                                };
                            }
                    if (binding)
                    {
                        const auto inserted = jointBindings.emplace(track.target.value, *binding);
                        if (inserted.second)
                            named.jointBindings.push_back(*binding);
                    }
                }
            }
            for (const auto target : morphTargets)
                if (jointBindings.count(target) != 0)
                {
                    fail(error, "glTF animation target cannot own both joint and morph deformation bindings");
                    return std::nullopt;
                }
            result.clips.push_back(std::move(named));
        }
        error.clear();
        return result;
    }
    catch (const std::bad_alloc&)
    {
        fail(error, "GLB animation decode allocation failed within admitted limits");
        return std::nullopt;
    }
}

} // namespace videohelper::gltf
