#pragma once

#include "ExactTokenMemo.h"
#include "RenderPassOutputContract.h"
#include "Visual3DScene.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace sceneaov
{
inline constexpr std::uint32_t kWireVersion = 5;
inline constexpr std::string_view kBackendCapability = "native-gpu";
inline constexpr std::array<renderpassoutput::Output, 6> kOutputs {
    renderpassoutput::Output::Depth,
    renderpassoutput::Output::Normal,
    renderpassoutput::Output::Emission,
    renderpassoutput::Output::Mask,
    renderpassoutput::Output::MaterialId,
    renderpassoutput::Output::ObjectId
};

inline constexpr bool supports(renderpassoutput::Output output) noexcept
{
    for (const auto candidate : kOutputs)
        if (candidate == output) return true;
    return false;
}

inline constexpr std::string_view operationKind(renderpassoutput::Output output) noexcept
{
    switch (output)
    {
        case renderpassoutput::Output::Depth: return "visual.3d.aov.depth";
        case renderpassoutput::Output::Normal: return "visual.3d.aov.normal";
        case renderpassoutput::Output::Emission: return "visual.3d.aov.emission";
        case renderpassoutput::Output::Mask: return "visual.3d.aov.mask";
        case renderpassoutput::Output::MaterialId: return "visual.3d.aov.material-id";
        case renderpassoutput::Output::ObjectId: return "visual.3d.aov.object-id";
        default: return {};
    }
}

struct Payload
{
    std::uint32_t version = kWireVersion;
    renderpassoutput::Output output = renderpassoutput::Output::Count;
    renderpassoutput::Extent extent {};
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
};

namespace detail
{
// Byte widths of the explicit wire records, independent of native padding.
inline constexpr std::size_t kSceneHeaderBytes = 56;
inline constexpr std::array<std::size_t, 8> kSceneRecordBytes {64, 4, 68, 74, 32, 4, 73, 56};
// Every count is a u32 on the wire. This is the format's range, not a scene limit.
inline constexpr std::size_t kSceneWireCount = 0xffffffffu;
inline constexpr std::array<std::size_t, 8> kSceneCountLimits {
    kSceneWireCount, kSceneWireCount, kSceneWireCount, kSceneWireCount,
    kSceneWireCount, kSceneWireCount, kSceneWireCount, kSceneWireCount
};
inline constexpr std::size_t sceneByteCount(const std::array<std::size_t, 8>& counts) noexcept
{
    std::size_t bytes = kSceneHeaderBytes;
    for (std::size_t index = 0; index < counts.size(); ++index)
        bytes += counts[index] * kSceneRecordBytes[index];
    return bytes;
}
inline constexpr std::uint32_t kSkyExtensionMagic = 0x31594b53; // SKY1, little endian
inline constexpr std::size_t kSkyRecordBytes = 72;
inline constexpr std::uint32_t kEnvironmentImageExtensionMagic = 0x314d4945; // EIM1
inline constexpr std::size_t kEnvironmentImageRecordBytes = 216;
inline constexpr std::uint32_t kSpectralSourceExtensionMagic = 0x31534c53; // SLS1
inline constexpr std::size_t kSpectralSourceRecordBytes = 76;
inline constexpr std::uint32_t kDirectionalIrradianceExtensionMagic = 0x31444c53; // SLD1
inline constexpr std::size_t kDirectionalIrradianceRecordBytes = 44;
// Per-object prior transforms, one record per object, written after the light
// extensions. A scene without priors writes none of this trailer.
inline constexpr std::uint32_t kPreviousObjectTransformExtensionMagic = 0x31565250; // PRV1, little endian
inline constexpr std::size_t kTransformRecordBytes = 40;
// One flag per light, written last and only when an environment light hides
// itself from camera rays. Released scenes keep their exact bytes.
inline constexpr std::uint32_t kEnvironmentCameraExtensionMagic = 0x31564345; // ECV1, little endian
// One flag per object, written only when a Shade Mesh operation authored an
// object's normals. Other scenes keep their exact bytes.
inline constexpr std::uint32_t kAuthoredShadingExtensionMagic = 0x31485341; // ASH1, little endian
inline constexpr std::size_t kAuthoredShadingRecordBytes = 4;
// Four floats per light, written only when a sky authors clouds. Clear skies
// keep their exact bytes.
inline constexpr std::uint32_t kSkyCloudExtensionMagic = 0x31444c43; // CLD1, little endian
inline constexpr std::size_t kSkyCloudRecordBytes = 16;
// The active camera one frame earlier (extension 20), one record, written only
// when the producer keeps one. Scenes without it keep their exact bytes.
inline constexpr std::uint32_t kPreviousCameraExtensionMagic = 0x31435650; // PVC1, little endian
inline constexpr std::size_t kPreviousCameraRecordBytes = kTransformRecordBytes + 12;
inline constexpr std::size_t kEnvironmentCameraRecordBytes = 4;
// One medium per material, written only when a material carries a volume, and
// then the world volume as one record, written only when the scene has fog.
// Scenes without media keep their exact bytes.
inline constexpr std::uint32_t kMaterialVolumeExtensionMagic = 0x314c4f56; // VOL1, little endian
inline constexpr std::uint32_t kWorldVolumeExtensionMagic = 0x314c5657; // WVL1, little endian
inline constexpr std::size_t kVolumeRecordBytes = 80;
// Sun halo and cloud thickness, two floats per light, written only when a sky
// sets one. Other scenes keep their exact bytes.
inline constexpr std::uint32_t kSkyExtrasExtensionMagic = 0x31584b53; // SKX1, little endian
inline constexpr std::size_t kSkyExtrasRecordBytes = 8;
// Camera background per light: mode, then the colour. Written only when an
// environment light sets one, so other scenes keep their exact bytes.
inline constexpr std::uint32_t kCameraBackgroundExtensionMagic = 0x31474b42; // BKG1, little endian
inline constexpr std::size_t kCameraBackgroundRecordBytes = 16;
// Per light: radius, angular radius, area X and Y, spread, area shape and ray
// visibility. Written only when a light is an area light or sets one of them.
inline constexpr std::uint32_t kLightShapeExtensionMagic = 0x31485354; // TSH1, little endian
inline constexpr std::size_t kLightShapeRecordBytes = 24;
// One ray visibility mask per object, written only when an object hides from a ray kind.
inline constexpr std::uint32_t kObjectVisibilityExtensionMagic = 0x3153564f; // OVS1, little endian
inline constexpr std::size_t kObjectVisibilityRecordBytes = 4;
// Per camera: f-stop, focus distance, blade rotation, aperture ratio and blade
// count. Written only when a camera leaves the pinhole defaults.
inline constexpr std::uint32_t kCameraLensExtensionMagic = 0x31534e4c; // LNS1, little endian
inline constexpr std::size_t kCameraLensRecordBytes = 20;
// One float per camera, its orthographic view height, written only when a camera
// is orthographic. Perspective scenes keep their exact bytes.
inline constexpr std::uint32_t kOrthographicCameraExtensionMagic = 0x314d4143; // CAM1, little endian
inline constexpr std::size_t kOrthographicCameraRecordBytes = 4;
// Extension 17, per vertex: the roughness lane (roughness, weight). Written only
// when a vertex has a nonzero weight, so scenes without one keep their bytes.
inline constexpr std::uint32_t kVertexRoughnessExtensionMagic = 0x31525456; // VTR1, little endian
inline constexpr std::size_t kVertexRoughnessRecordBytes = 8;
// Extension 18, per camera: the 1-based object it focuses on, 0 for none. Written only when
// a camera focuses on an object.
inline constexpr std::uint32_t kCameraFocusObjectExtensionMagic = 0x31434f46; // FOC1, little endian
inline constexpr std::size_t kCameraFocusObjectRecordBytes = 4;
// Extension 19, one record: the smooth-glass camera walk visits. Written only
// when the scene sets a count, so scenes on the default keep their bytes.
inline constexpr std::uint32_t kSmoothGlassCameraVisitsExtensionMagic = 0x31564753; // SGV1, little endian
inline constexpr std::size_t kSmoothGlassCameraVisitsRecordBytes = 4;
// Extension 21, per object: the vertex and index counts of the one timeline mesh
// whose ranges the wire leaves out. The receiver rebuilds those ranges from the
// object's timeline plan. Written only when the encoder is given that object.
inline constexpr std::uint32_t kReplayedMeshExtensionMagic = 0x314d5052; // RPM1, little endian
inline constexpr std::size_t kReplayedMeshRecordBytes = 8;
inline constexpr std::size_t kSceneExtensionCount = 21;
// The format's own ceiling on an encoded scene: u32 counts over these records
// stay far below it, so only a corrupt length can reach it.
inline constexpr std::size_t kMaximumSceneBytes = std::size_t { 1 } << 61;
// Upper bound on one scene's encoded size, from its live counts. All counts are
// u32 on the wire, so this sum fits in 64 bits.
inline std::size_t sceneWireBytes(const HarmonicMIDI::grid::Visual3DScene& scene) noexcept
{
    const std::array<std::size_t, 8> counts { scene.vertexCount, scene.indexCount, scene.objectCount,
        scene.materialCount, scene.textureCount, scene.textureTexelCount, scene.lightCount, scene.cameraCount };
    return sceneByteCount(counts) + 24 + scene.lightCount
            * (kSkyRecordBytes + kEnvironmentImageRecordBytes + kSpectralSourceRecordBytes)
        + 8 + scene.objectCount * kTransformRecordBytes
        + 8 + scene.lightCount * kEnvironmentCameraRecordBytes
        + 8 + scene.materialCount * kVolumeRecordBytes + 8 + kVolumeRecordBytes
        + 8 + kSmoothGlassCameraVisitsRecordBytes
        + 8 + scene.objectCount * kReplayedMeshRecordBytes;
}

// The trailer number of an extension magic, or 0 for an unknown one.
inline unsigned sceneExtension(std::uint32_t magic) noexcept
{
    constexpr std::array<std::uint32_t, kSceneExtensionCount> magics {
        kSkyExtensionMagic, kEnvironmentImageExtensionMagic, kSpectralSourceExtensionMagic,
        kDirectionalIrradianceExtensionMagic, kPreviousObjectTransformExtensionMagic,
        kEnvironmentCameraExtensionMagic, kAuthoredShadingExtensionMagic, kSkyCloudExtensionMagic,
        kLightShapeExtensionMagic, kObjectVisibilityExtensionMagic, kCameraLensExtensionMagic,
        kOrthographicCameraExtensionMagic, kMaterialVolumeExtensionMagic, kWorldVolumeExtensionMagic,
        kSkyExtrasExtensionMagic, kCameraBackgroundExtensionMagic, kVertexRoughnessExtensionMagic,
        kCameraFocusObjectExtensionMagic, kSmoothGlassCameraVisitsExtensionMagic,
        kPreviousCameraExtensionMagic, kReplayedMeshExtensionMagic };
    for (unsigned index = 0; index < magics.size(); ++index)
        if (magics[index] == magic) return index + 1;
    return 0;
}

// The object whose vertex and index ranges the wire may leave out: a non-empty
// mesh that no other object draws from. No empty object may start inside the
// left-out range or at its end, so the receiver puts every object back exactly.
inline std::optional<std::size_t> replayedMeshObject(const HarmonicMIDI::grid::Visual3DScene& scene,
                                                    HarmonicMIDI::grid::SceneObjectId id)
{
    if (!id.isValid()) return std::nullopt;
    std::optional<std::size_t> found;
    for (std::size_t i = 0; i < scene.objectCount; ++i)
        if (scene.objects[i].id == id)
        {
            if (found) return std::nullopt;
            found = i;
        }
    if (!found) return std::nullopt;
    const auto& mesh = scene.objects[*found];
    if (mesh.vertexCount == 0 || mesh.indexCount == 0) return std::nullopt;
    const auto clear = [](std::uint32_t first, std::uint32_t count, std::uint32_t otherFirst, std::uint32_t otherCount)
    {
        const auto end = std::uint64_t { first } + count;
        if (otherCount == 0) return otherFirst <= first || otherFirst > end;
        return std::uint64_t { otherFirst } + otherCount <= first || otherFirst >= end;
    };
    for (std::size_t i = 0; i < scene.objectCount; ++i)
    {
        if (i == *found) continue;
        const auto& other = scene.objects[i];
        if (!clear(mesh.firstVertex, mesh.vertexCount, other.firstVertex, other.vertexCount)
            || !clear(mesh.firstIndex, mesh.indexCount, other.firstIndex, other.indexCount)) return std::nullopt;
    }
    return found;
}

struct Writer
{
    std::vector<std::uint8_t> bytes;
    void u8(std::uint8_t value) { bytes.push_back(value); }
    void u32(std::uint32_t value)
    {
        for (unsigned shift = 0; shift != 32; shift += 8)
            bytes.push_back(static_cast<std::uint8_t>(value >> shift));
    }
    void f32(float value)
    {
        std::uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "float wire size changed");
        std::memcpy(&bits, &value, sizeof(bits));
        u32(bits);
    }
};

struct Reader
{
    const std::vector<std::uint8_t>& bytes;
    std::size_t offset = 0;
    bool ok = true;
    std::uint8_t u8()
    {
        if (offset >= bytes.size()) { ok = false; return 0; }
        return bytes[offset++];
    }
    std::uint32_t u32()
    {
        std::uint32_t value = 0;
        for (unsigned shift = 0; shift != 32; shift += 8)
            value |= static_cast<std::uint32_t>(u8()) << shift;
        return value;
    }
    float f32()
    {
        const auto bits = u32();
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
};

inline void writeVec2(Writer& writer, const HarmonicMIDI::grid::SceneVec2& value)
{
    writer.f32(value.x); writer.f32(value.y);
}
inline void writeVec3(Writer& writer, const HarmonicMIDI::grid::SceneVec3& value)
{
    writer.f32(value.x); writer.f32(value.y); writer.f32(value.z);
}
inline void writeQuaternion(Writer& writer, const HarmonicMIDI::grid::SceneQuaternion& value)
{
    writer.f32(value.x); writer.f32(value.y); writer.f32(value.z); writer.f32(value.w);
}
inline void writeTransform(Writer& writer, const HarmonicMIDI::grid::SceneTransform3D& value)
{
    writeVec3(writer, value.translation); writeQuaternion(writer, value.rotation);
    writeVec3(writer, value.scale);
}
inline HarmonicMIDI::grid::SceneVec2 readVec2(Reader& reader)
{
    return { reader.f32(), reader.f32() };
}
inline HarmonicMIDI::grid::SceneVec3 readVec3(Reader& reader)
{
    return { reader.f32(), reader.f32(), reader.f32() };
}
inline HarmonicMIDI::grid::SceneQuaternion readQuaternion(Reader& reader)
{
    return { reader.f32(), reader.f32(), reader.f32(), reader.f32() };
}
inline HarmonicMIDI::grid::SceneTransform3D readTransform(Reader& reader)
{
    return { readVec3(reader), readQuaternion(reader), readVec3(reader) };
}

inline void writeVolume(Writer& writer, std::uint32_t owner, const HarmonicMIDI::grid::SceneVolume& volume)
{
    writer.u32(owner); writer.f32(volume.density);
    writeVec3(writer, volume.color); writeVec3(writer, volume.absorptionColor);
    writer.f32(volume.anisotropy); writeVec3(writer, volume.emission);
    writer.u32(static_cast<std::uint32_t>(volume.field)); writer.u32(volume.fieldDetail);
    writer.f32(volume.fieldScale); writeVec3(writer, volume.fieldOffset);
    writer.u32(volume.volumeOnly ? 1u : 0u); writer.f32(volume.radius);
}
inline bool readVolume(Reader& reader, std::uint32_t owner, HarmonicMIDI::grid::SceneVolume& volume)
{
    if (reader.u32() != owner) return false;
    volume.density = reader.f32();
    volume.color = readVec3(reader); volume.absorptionColor = readVec3(reader);
    volume.anisotropy = reader.f32(); volume.emission = readVec3(reader);
    const auto field = reader.u32();
    if (field > 1) return false;
    volume.field = static_cast<HarmonicMIDI::grid::SceneVolumeField>(field);
    volume.fieldDetail = reader.u32();
    volume.fieldScale = reader.f32(); volume.fieldOffset = readVec3(reader);
    const auto volumeOnly = reader.u32();
    if (volumeOnly > 1) return false;
    volume.volumeOnly = volumeOnly != 0;
    volume.radius = reader.f32();
    return true;
}

// replayedMesh names a timeline-bound object whose receiver rebuilds its mesh from
// the object's plan. When replayedMeshObject admits it, its vertices and indices
// are left out and extension 21 records their counts. Other scenes keep their bytes.
inline std::vector<std::uint8_t> encodeScene(const HarmonicMIDI::grid::Visual3DScene& scene,
                                             HarmonicMIDI::grid::SceneObjectId replayedMesh = {})
{
    if (!HarmonicMIDI::grid::validateVisual3DScene(scene).valid()) return {};
    // Wire version 1 stores each record count in 32 bits.
    const std::array<std::size_t, 8> counts { scene.vertexCount, scene.indexCount, scene.objectCount,
        scene.materialCount, scene.textureCount, scene.textureTexelCount,
        scene.lightCount, scene.cameraCount };
    for (std::size_t index = 0; index < counts.size(); ++index)
        if (counts[index] > kSceneCountLimits[index]) return {};
    const auto replayed = replayedMeshObject(scene, replayedMesh);
    const auto* gap = replayed ? &scene.objects[*replayed] : nullptr;
    const std::size_t gapVertex = gap ? gap->firstVertex : 0, gapVertices = gap ? gap->vertexCount : 0;
    const std::size_t gapIndex = gap ? gap->firstIndex : 0, gapIndices = gap ? gap->indexCount : 0;
    const auto written = [](std::size_t i, std::size_t first, std::size_t count) { return i < first || i >= first + count; };
    // A position after the left-out range moves down by its length.
    const auto compact = [](std::uint32_t value, std::size_t first, std::size_t count)
    {
        return value <= first ? value
            : static_cast<std::uint32_t>(value - std::min<std::size_t>(value - first, count));
    };
    Writer writer;
    writer.bytes.reserve(sceneByteCount({scene.vertexCount - gapVertices, scene.indexCount - gapIndices,
        scene.objectCount, scene.materialCount, scene.textureCount, scene.textureTexelCount,
        scene.lightCount, scene.cameraCount}));
    writer.u32(scene.schemaVersion); writer.u32(scene.id.value);
    writer.u32(scene.activeCamera.value); writeVec3(writer, scene.ambientColor);
    for (const auto count : { scene.vertexCount - gapVertices, scene.indexCount - gapIndices, scene.objectCount,
                              scene.materialCount, scene.textureCount,
                              scene.textureTexelCount, scene.lightCount, scene.cameraCount })
        writer.u32(static_cast<std::uint32_t>(count));
    for (std::size_t i = 0; i < scene.vertexCount; ++i)
    {
        if (!written(i, gapVertex, gapVertices)) continue;
        writeVec3(writer, scene.vertices[i].position);
        writeVec3(writer, scene.vertices[i].normal);
        writeVec2(writer, scene.vertices[i].uv);
        for (const auto value : scene.vertices[i].tangent) writer.f32(value);
        for (const auto value : scene.vertices[i].color) writer.f32(value);
    }
    for (std::size_t i = 0; i < scene.indexCount; ++i)
        if (written(i, gapIndex, gapIndices)) writer.u32(scene.indices[i]);
    for (std::size_t i = 0; i < scene.objectCount; ++i)
    {
        const auto& value = scene.objects[i];
        writer.u32(value.id.value); writer.u32(value.parent.value);
        writer.u32(value.material.value); writeTransform(writer, value.transform);
        if (&value == gap)
        {
            // The left-out mesh keeps its start and draws nothing until it is rebuilt.
            writer.u32(value.firstVertex); writer.u32(0); writer.u32(value.firstIndex); writer.u32(0);
            continue;
        }
        writer.u32(compact(value.firstVertex, gapVertex, gapVertices)); writer.u32(value.vertexCount);
        writer.u32(compact(value.firstIndex, gapIndex, gapIndices)); writer.u32(value.indexCount);
    }
    for (std::size_t i = 0; i < scene.materialCount; ++i)
    {
        const auto& value = scene.materials[i];
        writer.u32(value.id.value); writeVec3(writer, value.baseColor);
        writer.u32(value.baseColorTexture.value);
        writer.u32(value.metallicRoughnessTexture.value);
        writer.u32(value.normalTexture.value);
        writer.u32(value.occlusionTexture.value);
        writer.u32(value.emissiveTexture.value);
        writer.f32(value.opacity);
        writer.f32(value.metallic); writer.f32(value.roughness);
        writeVec3(writer, value.emissive);
        writer.f32(value.normalScale); writer.f32(value.occlusionStrength);
        writer.u8(static_cast<std::uint8_t>(value.alphaMode));
        writer.f32(value.alphaCutoff);
        writer.u8(value.doubleSided ? 1u : 0u);
    }
    for (std::size_t i = 0; i < scene.textureCount; ++i)
    {
        const auto& value = scene.textures[i];
        writer.u32(value.id.value); writer.u32(value.firstTexel);
        writer.u32(value.width); writer.u32(value.height);
        writer.u32(value.magFilter); writer.u32(value.minFilter);
        writer.u32(value.wrapS); writer.u32(value.wrapT);
    }
    for (std::size_t i = 0; i < scene.textureTexelCount; ++i)
    {
        const auto& value = scene.textureTexels[i];
        writer.u8(value.red); writer.u8(value.green); writer.u8(value.blue); writer.u8(value.alpha);
    }
    for (std::size_t i = 0; i < scene.lightCount; ++i)
    {
        const auto& value = scene.lights[i];
        writer.u32(value.id.value); writer.u8(static_cast<std::uint8_t>(value.kind));
        writeTransform(writer, value.transform); writeVec3(writer, value.color);
        writer.f32(value.intensity); writer.f32(value.range);
        writer.f32(value.innerConeAngle); writer.f32(value.outerConeAngle);
    }
    for (std::size_t i = 0; i < scene.cameraCount; ++i)
    {
        const auto& value = scene.cameras[i];
        writer.u32(value.id.value); writeTransform(writer, value.transform);
        writer.f32(value.verticalFovRadians); writer.f32(value.nearPlane); writer.f32(value.farPlane);
    }
    bool hasSky = false;
    for (std::size_t i=0;i<scene.lightCount;++i) hasSky |= scene.lights[i].sky.version != 0;
    if (hasSky)
    {
        writer.u32(kSkyExtensionMagic); writer.u32(1);
        for (std::size_t i=0;i<scene.lightCount;++i)
        {
            const auto& sky=scene.lights[i].sky;
            writer.u32(sky.version);
            writeVec3(writer,sky.zenith); writeVec3(writer,sky.horizon);
            writeVec3(writer,sky.ground); writeVec3(writer,sky.sunColor);
            writer.f32(sky.sunIntensity); writer.f32(sky.sunAzimuthDegrees);
            writer.f32(sky.sunElevationDegrees); writer.f32(sky.sunRadiusDegrees);
            writer.f32(sky.orientationDegrees);
        }
    }
    bool hasImage = false;
    for (std::size_t i=0; i<scene.lightCount; ++i) hasImage |= scene.lights[i].environmentImage.version != 0;
    if (hasImage)
    {
        writer.u32(kEnvironmentImageExtensionMagic); writer.u32(1);
        for (std::size_t i=0; i<scene.lightCount; ++i)
        {
            const auto& image=scene.lights[i].environmentImage;
            writer.u32(image.version);
            writer.u32(static_cast<std::uint32_t>(image.asset.id.size()));
            for (std::size_t c=0; c<128; ++c) writer.u8(c<image.asset.id.size() ? image.asset.id[c] : 0);
            writer.u32(static_cast<std::uint32_t>(image.asset.version));
            writer.u32(static_cast<std::uint32_t>(image.asset.version >> 32));
            for (std::size_t c=0; c<64; ++c) writer.u8(c<image.asset.contentSha256.size() ? image.asset.contentSha256[c] : 0);
            writer.f32(image.orientationDegrees); writer.f32(image.exposureStops);
        }
    }
    bool hasSpectralSource = false;
    for (std::size_t i=0; i<scene.lightCount; ++i) hasSpectralSource |= scene.lights[i].spectralSource.version != 0;
    if (hasSpectralSource)
    {
        writer.u32(kSpectralSourceExtensionMagic); writer.u32(1);
        for (std::size_t i=0; i<scene.lightCount; ++i)
        {
            const auto& light = scene.lights[i];
            writer.u32(light.id.value); writer.u32(light.spectralSource.version);
            for (const auto wavelength : light.spectralSource.wavelengthsNanometres) writer.f32(wavelength);
            for (const auto radiance : light.spectralSource.radiance) writer.f32(radiance);
            writer.f32(light.spectralSource.traceMaximumDistance);
        }
    }
    bool hasDirectionalIrradiance = false;
    for (std::size_t i=0; i<scene.lightCount; ++i)
        hasDirectionalIrradiance |= scene.lights[i].spectralSource.directionalIrradiance.version != 0;
    if (hasDirectionalIrradiance)
    {
        writer.u32(kDirectionalIrradianceExtensionMagic); writer.u32(1);
        for (std::size_t i=0; i<scene.lightCount; ++i)
        {
            const auto& light=scene.lights[i];
            const auto& source=light.spectralSource.directionalIrradiance;
            writer.u32(light.id.value); writer.u32(source.version);
            writer.u32(static_cast<std::uint32_t>(source.measure));
            for (float value : source.coefficients) writer.f32(value);
        }
    }
    if (scene.previousObjectTransformCount != 0)
    {
        writer.u32(kPreviousObjectTransformExtensionMagic); writer.u32(1);
        for (std::size_t i = 0; i < scene.objectCount; ++i)
            writeTransform(writer, scene.previousObjectTransforms[i]);
    }
    if (!HarmonicMIDI::grid::sceneEnvironmentVisibleToCamera(scene.lights.data(), scene.lightCount))
    {
        writer.u32(kEnvironmentCameraExtensionMagic); writer.u32(1);
        for (std::size_t i = 0; i < scene.lightCount; ++i)
            writer.u32(scene.lights[i].environmentCameraVisible ? 1u : 0u);
    }
    bool hasAuthoredShading = false;
    for (std::size_t i = 0; i < scene.objectCount; ++i) hasAuthoredShading |= scene.objects[i].authoredShading;
    if (hasAuthoredShading)
    {
        writer.u32(kAuthoredShadingExtensionMagic); writer.u32(1);
        for (std::size_t i = 0; i < scene.objectCount; ++i)
            writer.u32(scene.objects[i].authoredShading ? 1u : 0u);
    }
    bool hasClouds = false;
    for (std::size_t i=0; i<scene.lightCount; ++i)
        hasClouds |= HarmonicMIDI::grid::analyticSkyCloudsAuthored(scene.lights[i].sky);
    if (hasClouds)
    {
        writer.u32(kSkyCloudExtensionMagic); writer.u32(1);
        for (std::size_t i=0; i<scene.lightCount; ++i)
        {
            const auto& sky=scene.lights[i].sky;
            writer.f32(sky.cloudCoverage); writer.f32(sky.cloudType);
            writer.f32(sky.cloudDrift); writer.f32(sky.cloudHeadingDegrees);
        }
    }
    bool hasLightShape = false;
    for (std::size_t i=0; i<scene.lightCount; ++i)
        hasLightShape |= HarmonicMIDI::grid::sceneLightHasShapeOrVisibility(scene.lights[i]);
    if (hasLightShape)
    {
        writer.u32(kLightShapeExtensionMagic); writer.u32(1);
        for (std::size_t i=0; i<scene.lightCount; ++i)
        {
            const auto& light=scene.lights[i];
            writer.f32(light.radius); writer.f32(light.angularRadius);
            writer.f32(light.areaSizeX); writer.f32(light.areaSizeY); writer.f32(light.spread);
            writer.u32(static_cast<std::uint32_t>(light.areaShape) | (std::uint32_t(light.visibility) << 8));
        }
    }
    bool hasObjectVisibility = false;
    for (std::size_t i=0; i<scene.objectCount; ++i)
        hasObjectVisibility |= scene.objects[i].visibility != HarmonicMIDI::grid::sceneRayVisibility::All;
    if (hasObjectVisibility)
    {
        writer.u32(kObjectVisibilityExtensionMagic); writer.u32(1);
        for (std::size_t i=0; i<scene.objectCount; ++i) writer.u32(scene.objects[i].visibility);
    }
    bool hasLens = false;
    for (std::size_t i=0; i<scene.cameraCount; ++i)
        hasLens |= HarmonicMIDI::grid::sceneCameraLensAuthored(scene.cameras[i]);
    if (hasLens)
    {
        writer.u32(kCameraLensExtensionMagic); writer.u32(1);
        for (std::size_t i=0; i<scene.cameraCount; ++i)
        {
            const auto& camera=scene.cameras[i];
            writer.f32(camera.fStop); writer.f32(camera.focusDistance);
            writer.f32(camera.bladeRotationRadians); writer.f32(camera.apertureRatio);
            writer.u32(camera.apertureBlades);
        }
    }
    bool hasOrthographic = false;
    for (std::size_t i=0; i<scene.cameraCount; ++i) hasOrthographic |= scene.cameras[i].orthographic();
    if (hasOrthographic)
    {
        writer.u32(kOrthographicCameraExtensionMagic); writer.u32(1);
        for (std::size_t i=0; i<scene.cameraCount; ++i) writer.f32(scene.cameras[i].orthographicHeight);
    }
    bool hasMaterialVolume = false;
    for (std::size_t i=0; i<scene.materialCount; ++i)
        hasMaterialVolume |= HarmonicMIDI::grid::sceneVolumeActive(scene.materials[i].volume);
    if (hasMaterialVolume)
    {
        writer.u32(kMaterialVolumeExtensionMagic); writer.u32(1);
        for (std::size_t i=0; i<scene.materialCount; ++i)
            writeVolume(writer, scene.materials[i].id.value, scene.materials[i].volume);
    }
    if (HarmonicMIDI::grid::sceneVolumeActive(scene.worldVolume))
    {
        writer.u32(kWorldVolumeExtensionMagic); writer.u32(1);
        writeVolume(writer, 0, scene.worldVolume);
    }
    bool hasExtras = false;
    for (std::size_t i=0; i<scene.lightCount; ++i)
        hasExtras |= HarmonicMIDI::grid::analyticSkyExtrasAuthored(scene.lights[i].sky);
    if (hasExtras)
    {
        writer.u32(kSkyExtrasExtensionMagic); writer.u32(1);
        for (std::size_t i=0; i<scene.lightCount; ++i)
        {
            writer.f32(scene.lights[i].sky.sunHalo); writer.f32(scene.lights[i].sky.cloudThickness);
        }
    }
    bool hasBackground = false;
    for (std::size_t i=0; i<scene.lightCount; ++i)
        hasBackground |= scene.lights[i].cameraBackground.mode != HarmonicMIDI::grid::SceneCameraBackgroundMode::World;
    if (hasBackground)
    {
        writer.u32(kCameraBackgroundExtensionMagic); writer.u32(1);
        for (std::size_t i=0; i<scene.lightCount; ++i)
        {
            const auto& background = scene.lights[i].cameraBackground;
            writer.u32(static_cast<std::uint32_t>(background.mode)); writeVec3(writer, background.color);
        }
    }
    bool hasVertexRoughness = false;
    for (std::size_t i=0; i<scene.vertexCount; ++i)
        hasVertexRoughness |= written(i, gapVertex, gapVertices) && scene.vertices[i].roughness[1] != 0.0f;
    if (hasVertexRoughness)
    {
        writer.u32(kVertexRoughnessExtensionMagic); writer.u32(1);
        for (std::size_t i=0; i<scene.vertexCount; ++i)
            if (written(i, gapVertex, gapVertices))
                for (const auto value : scene.vertices[i].roughness) writer.f32(value);
    }
    bool hasFocusObject = false;
    for (std::size_t i=0; i<scene.cameraCount; ++i)
        hasFocusObject |= scene.cameras[i].focusObject != 0;
    if (hasFocusObject)
    {
        writer.u32(kCameraFocusObjectExtensionMagic); writer.u32(1);
        for (std::size_t i=0; i<scene.cameraCount; ++i) writer.u32(scene.cameras[i].focusObject);
    }
    if (scene.smoothGlassCameraVisits != 0)
    {
        writer.u32(kSmoothGlassCameraVisitsExtensionMagic); writer.u32(1);
        writer.u32(scene.smoothGlassCameraVisits);
    }
    if (scene.previousCamera.has_value())
    {
        writer.u32(kPreviousCameraExtensionMagic); writer.u32(1);
        writeTransform(writer, scene.previousCamera->transform);
        writer.f32(scene.previousCamera->verticalFovRadians); writer.f32(scene.previousCamera->nearPlane);
        writer.f32(scene.previousCamera->farPlane);
    }
    if (gap != nullptr)
    {
        writer.u32(kReplayedMeshExtensionMagic); writer.u32(1);
        for (std::size_t i=0; i<scene.objectCount; ++i)
        {
            writer.u32(i == *replayed ? gap->vertexCount : 0u);
            writer.u32(i == *replayed ? gap->indexCount : 0u);
        }
    }
    // The optional measure consumes the existing scene byte budget.
    if (writer.bytes.size()>kMaximumSceneBytes) return {};
    return writer.bytes;
}

// replayedMesh receives the object whose mesh extension 21 left out. Its ranges
// hold zeroed records until the caller rebuilds them, and a zero normal is not a
// valid vertex, so such a scene is validated by the caller after the rebuild.
// A caller that passes no replayedMesh cannot rebuild a mesh, so a scene with
// extension 21 is refused.
inline bool decodeScene(const std::vector<std::uint8_t>& bytes,
                        HarmonicMIDI::grid::Visual3DScene& scene,
                        std::optional<HarmonicMIDI::grid::SceneObjectId>* replayedMesh = nullptr)
{
    using namespace HarmonicMIDI::grid;
    if (bytes.size() < kSceneHeaderBytes || bytes.size() > kMaximumSceneBytes)
        return false;
    Reader reader { bytes };
    scene = {};
    scene.schemaVersion = reader.u32(); scene.id.value = reader.u32();
    scene.activeCamera.value = reader.u32(); scene.ambientColor = readVec3(reader);
    std::array<std::size_t, 8> counts {};
    for (auto& count : counts) count = reader.u32();
    for (std::size_t index = 0; index < counts.size(); ++index)
        if (counts[index] > kSceneCountLimits[index]) return false;
    // Reject a truncated or forged count before allocating any texture storage.
    const auto legacyBytes = sceneByteCount(counts);
    if (!reader.ok || bytes.size() < legacyBytes) return false;
    constexpr std::array<std::size_t, kSceneExtensionCount> extensionRecordBytes {
        kSkyRecordBytes, kEnvironmentImageRecordBytes, kSpectralSourceRecordBytes,
        kDirectionalIrradianceRecordBytes, kTransformRecordBytes, kEnvironmentCameraRecordBytes,
        kAuthoredShadingRecordBytes, kSkyCloudRecordBytes, kLightShapeRecordBytes,
        kObjectVisibilityRecordBytes, kCameraLensRecordBytes, kOrthographicCameraRecordBytes,
        kVolumeRecordBytes, kVolumeRecordBytes, kSkyExtrasRecordBytes, kCameraBackgroundRecordBytes,
        kVertexRoughnessRecordBytes, kCameraFocusObjectRecordBytes, kSmoothGlassCameraVisitsRecordBytes,
        kPreviousCameraRecordBytes, kReplayedMeshRecordBytes };
    // Light extensions count lights. The prior, shading and visibility trailers
    // count objects, the lens and orthographic trailers cameras, the material volume trailer
    // materials, the world volume is one record, the roughness trailer counts the vertices
    // on the wire, the focus trailer cameras, the camera visits (19) and the prior camera (20)
    // are one record each, and the replayed mesh trailer (21) counts objects.
    const std::array<std::size_t, kSceneExtensionCount> extensionRecords {
        counts[6], counts[6], counts[6], counts[6], counts[2], counts[6], counts[2], counts[6],
        counts[6], counts[2], counts[7], counts[7], counts[3], 1, counts[6], counts[6], counts[0], counts[7], 1, 1,
        counts[2] };
    const auto wordAt = [&bytes](std::size_t offset)
    {
        std::uint32_t value = 0;
        for (unsigned shift = 0; shift != 32; shift += 8)
            value |= static_cast<std::uint32_t>(bytes[offset + shift / 8]) << shift;
        return value;
    };
    // Walk the trailers before allocating anything: each appears at most once, in
    // order, and together they fill the input exactly.
    std::size_t replayedRecords = 0;
    {
        std::size_t offset = legacyBytes;
        unsigned last = 0;
        while (offset < bytes.size())
        {
            if (bytes.size() - offset < 8) return false;
            const auto extension = sceneExtension(wordAt(offset));
            if (extension == 0 || extension <= last || wordAt(offset + 4) != 1) return false;
            const auto length = extensionRecords[extension - 1] * extensionRecordBytes[extension - 1];
            if (bytes.size() - offset - 8 < length) return false;
            if (extension == 21) replayedRecords = offset + 8;
            offset += 8 + length;
            last = extension;
        }
    }
    // The left-out mesh: one object with both counts, which the wire draws as empty.
    std::optional<std::size_t> replayed;
    std::size_t gapVertex = 0, gapVertices = 0, gapIndex = 0, gapIndices = 0;
    if (replayedRecords != 0)
    {
        if (replayedMesh == nullptr) return false;
        for (std::size_t i = 0; i < counts[2]; ++i)
        {
            const auto vertices = wordAt(replayedRecords + i * 8), indices = wordAt(replayedRecords + i * 8 + 4);
            if (vertices == 0 && indices == 0) continue;
            if (replayed || vertices == 0 || indices == 0) return false;
            replayed = i; gapVertices = vertices; gapIndices = indices;
        }
        if (!replayed) return false;
        // The object record: id, parent, material, a 40-byte transform, then its ranges.
        const auto record = kSceneHeaderBytes + counts[0] * kSceneRecordBytes[0]
            + counts[1] * kSceneRecordBytes[1] + *replayed * kSceneRecordBytes[2];
        gapVertex = wordAt(record + 52); gapIndex = wordAt(record + 60);
        if (wordAt(record + 56) != 0 || wordAt(record + 64) != 0
            || gapVertex > counts[0] || gapIndex > counts[1]
            || counts[0] + gapVertices > kSceneWireCount || counts[1] + gapIndices > kSceneWireCount) return false;
    }
    // A wire record past the left-out range lands after it.
    const auto vertexSlot = [&](std::size_t i) { return i < gapVertex ? i : i + gapVertices; };
    const auto indexSlot = [&](std::size_t i) { return i < gapIndex ? i : i + gapIndices; };
    scene.vertexCount = counts[0] + gapVertices; scene.indexCount = counts[1] + gapIndices;
    scene.objectCount = counts[2]; scene.materialCount = counts[3];
    scene.textureCount = counts[4]; scene.textureTexelCount = counts[5];
    scene.lightCount = counts[6]; scene.cameraCount = counts[7];
    // The record lengths were admitted against the input size above, so these
    // lengths are backed by bytes that really arrived. The left-out mesh is the
    // one exception: extension 21 gives its counts, and the caller rebuilds it.
    Visual3DSceneCapacity capacity = visual3DSceneAllocatedCapacity(scene);
    capacity.vertices = std::max(capacity.vertices, scene.vertexCount);
    capacity.indices = std::max(capacity.indices, scene.indexCount);
    capacity.objects = std::max(capacity.objects, scene.objectCount);
    capacity.materials = std::max(capacity.materials, scene.materialCount);
    capacity.textures = std::max(capacity.textures, scene.textureCount);
    capacity.textureTexels = scene.textureTexelCount;
    capacity.lights = std::max(capacity.lights, scene.lightCount);
    capacity.cameras = std::max(capacity.cameras, scene.cameraCount);
    if (!allocateVisual3DSceneBuffers(scene, capacity).allocated()) return false;
    // Allocation only reserves texels. Their length is the wire's count.
    scene.textureTexels.resize(scene.textureTexelCount);
    for (std::size_t i = 0; i < counts[0]; ++i)
    {
        auto& vertex = scene.vertices[vertexSlot(i)];
        vertex.position = readVec3(reader);
        vertex.normal = readVec3(reader);
        vertex.uv = readVec2(reader);
        vertex.tangent = { reader.f32(), reader.f32(), reader.f32(), reader.f32() };
        vertex.color = { reader.f32(), reader.f32(), reader.f32(), reader.f32() };
    }
    for (std::size_t i = 0; i < counts[1]; ++i) scene.indices[indexSlot(i)] = reader.u32();
    // Put each object back around the left-out range, the inverse of the encoder's
    // compaction. An empty object at the range's start stays before it.
    const auto restore = [](std::uint32_t& first, std::uint32_t count, std::size_t gap, std::size_t length)
    {
        if (count != 0 && first < gap && std::uint64_t { first } + count > gap) return false;
        if (count != 0 ? first >= gap : first > gap)
        {
            if (std::uint64_t { first } + length > kSceneWireCount) return false;
            first = static_cast<std::uint32_t>(first + length);
        }
        return true;
    };
    for (std::size_t i = 0; i < scene.objectCount; ++i)
    {
        auto& value = scene.objects[i];
        value.id.value = reader.u32(); value.parent.value = reader.u32();
        value.material.value = reader.u32(); value.transform = readTransform(reader);
        value.firstVertex = reader.u32(); value.vertexCount = reader.u32();
        value.firstIndex = reader.u32(); value.indexCount = reader.u32();
        if (!replayed) continue;
        if (i == *replayed)
        {
            value.vertexCount = static_cast<std::uint32_t>(gapVertices);
            value.indexCount = static_cast<std::uint32_t>(gapIndices);
        }
        else if (!restore(value.firstVertex, value.vertexCount, gapVertex, gapVertices)
                 || !restore(value.firstIndex, value.indexCount, gapIndex, gapIndices)) return false;
    }
    for (std::size_t i = 0; i < scene.materialCount; ++i)
    {
        auto& value = scene.materials[i];
        value.id.value = reader.u32(); value.baseColor = readVec3(reader);
        value.baseColorTexture.value = reader.u32();
        value.metallicRoughnessTexture.value = reader.u32();
        value.normalTexture.value = reader.u32();
        value.occlusionTexture.value = reader.u32();
        value.emissiveTexture.value = reader.u32();
        value.opacity = reader.f32();
        value.metallic = reader.f32(); value.roughness = reader.f32();
        value.emissive = readVec3(reader);
        value.normalScale = reader.f32(); value.occlusionStrength = reader.f32();
        value.alphaMode = static_cast<SceneAlphaMode>(reader.u8());
        value.alphaCutoff = reader.f32();
        value.doubleSided = reader.u8() != 0;
    }
    for (std::size_t i = 0; i < scene.textureCount; ++i)
    {
        auto& value = scene.textures[i];
        value.id.value = reader.u32(); value.firstTexel = reader.u32();
        value.width = reader.u32(); value.height = reader.u32();
        value.magFilter = reader.u32(); value.minFilter = reader.u32();
        value.wrapS = reader.u32(); value.wrapT = reader.u32();
    }
    for (std::size_t i = 0; i < scene.textureTexelCount; ++i)
        scene.textureTexels[i] = { reader.u8(), reader.u8(), reader.u8(), reader.u8() };
    for (std::size_t i = 0; i < scene.lightCount; ++i)
    {
        auto& value = scene.lights[i];
        value.id.value = reader.u32(); value.kind = static_cast<SceneLightKind>(reader.u8());
        value.transform = readTransform(reader); value.color = readVec3(reader);
        value.intensity = reader.f32(); value.range = reader.f32();
        value.innerConeAngle = reader.f32(); value.outerConeAngle = reader.f32();
    }
    for (std::size_t i = 0; i < scene.cameraCount; ++i)
    {
        auto& value = scene.cameras[i];
        value.id.value = reader.u32(); value.transform = readTransform(reader);
        value.verticalFovRadians = reader.f32(); value.nearPlane = reader.f32();
        value.farPlane = reader.f32();
    }
    unsigned lastExtension = 0;
    while (reader.ok && reader.offset < bytes.size())
    {
        if (bytes.size()-reader.offset < 8) return false;
        const auto magic = reader.u32();
        if (reader.u32()!=1) return false;
        const unsigned extension = sceneExtension(magic);
        if (extension==0 || extension<=lastExtension
            || bytes.size()-reader.offset < extensionRecords[extension-1]*extensionRecordBytes[extension-1]) return false;
        lastExtension = extension;
        if (extension==1)
        {
            bool active = false;
            for (std::size_t i=0;i<scene.lightCount;++i)
            {
                auto& sky=scene.lights[i].sky;
                sky.version=reader.u32(); active |= sky.version!=0;
                sky.zenith=readVec3(reader); sky.horizon=readVec3(reader);
                sky.ground=readVec3(reader); sky.sunColor=readVec3(reader);
                sky.sunIntensity=reader.f32(); sky.sunAzimuthDegrees=reader.f32();
                sky.sunElevationDegrees=reader.f32(); sky.sunRadiusDegrees=reader.f32();
                sky.orientationDegrees=reader.f32();
            }
            if (!active) return false;
        }
        else if (extension==2)
        {
            bool active=false;
            for (std::size_t i=0; i<scene.lightCount; ++i)
            {
                auto& image=scene.lights[i].environmentImage;
                image.version=reader.u32(); active |= image.version!=0;
                const auto length=reader.u32();
                if (length>128) return false;
                for (std::size_t c=0; c<128; ++c)
                {
                    const auto value=reader.u8();
                    if (c<length) image.asset.id.push_back(static_cast<char>(value));
                    else if (value!=0) return false;
                }
                const auto low=reader.u32(); const auto high=reader.u32();
                image.asset.version=std::uint64_t(low) | (std::uint64_t(high)<<32);
                for (std::size_t c=0; c<64; ++c)
                {
                    const auto value=reader.u8();
                    if (image.version!=0) image.asset.contentSha256.push_back(static_cast<char>(value));
                    else if (value!=0) return false;
                }
                image.orientationDegrees=reader.f32(); image.exposureStops=reader.f32();
            }
            if (!active) return false;
        }
        else if (extension==3)
        {
            bool active = false;
            for (std::size_t i=0; i<scene.lightCount; ++i)
            {
                auto& light = scene.lights[i];
                if (reader.u32()!=light.id.value) return false;
                auto& source = light.spectralSource;
                source.version=reader.u32(); active |= source.version!=0;
                for (auto& wavelength : source.wavelengthsNanometres) wavelength=reader.f32();
                for (auto& radiance : source.radiance) radiance=reader.f32();
                source.traceMaximumDistance=reader.f32();
            }
            if (!active) return false;
        }
        else if (extension==4)
        {
            bool active=false;
            for (std::size_t i=0; i<scene.lightCount; ++i)
            {
                auto& light=scene.lights[i];
                if (reader.u32()!=light.id.value) return false;
                auto& source=light.spectralSource.directionalIrradiance;
                source.version=reader.u32(); active |= source.version!=0;
                source.measure=static_cast<diffractionmaterial::DirectionalSpectralMeasure>(reader.u32());
                for (auto& value : source.coefficients) value=reader.f32();
            }
            if (!active) return false;
        }
        else if (extension==21)
        {
            // Read before allocation above; only the left-out object's counts are nonzero.
            reader.offset += scene.objectCount * kReplayedMeshRecordBytes;
        }
        else if (extension==20)
        {
            SceneCameraRecord previous;
            previous.transform = readTransform(reader);
            previous.verticalFovRadians = reader.f32(); previous.nearPlane = reader.f32();
            previous.farPlane = reader.f32();
            scene.previousCamera = previous;
        }
        else if (extension==13)
        {
            // Only a material with a medium writes this trailer.
            bool active = false;
            for (std::size_t i=0; i<scene.materialCount; ++i)
            {
                auto& material = scene.materials[i];
                if (!readVolume(reader, material.id.value, material.volume)) return false;
                active |= HarmonicMIDI::grid::sceneVolumeActive(material.volume);
            }
            if (!active) return false;
        }
        else if (extension==16)
        {
            bool authored = false;
            for (std::size_t i=0; i<scene.lightCount; ++i)
            {
                auto& background = scene.lights[i].cameraBackground;
                const auto mode = reader.u32();
                if (mode > 2) return false;
                background.mode = static_cast<SceneCameraBackgroundMode>(mode);
                background.color = readVec3(reader);
                authored |= mode != 0;
            }
            if (!authored) return false;
        }
        else if (extension==15)
        {
            bool authored = false;
            for (std::size_t i=0; i<scene.lightCount; ++i)
            {
                auto& sky=scene.lights[i].sky;
                sky.sunHalo=reader.f32(); sky.cloudThickness=reader.f32();
                authored |= HarmonicMIDI::grid::analyticSkyExtrasAuthored(sky);
            }
            if (!authored) return false;
        }
        else if (extension==14)
        {
            if (!readVolume(reader, 0, scene.worldVolume)
                || !HarmonicMIDI::grid::sceneVolumeActive(scene.worldVolume)) return false;
        }
        else if (extension==9)
        {
            // Only a sized, area or partly hidden light writes this trailer.
            bool authored = false;
            for (std::size_t i=0; i<scene.lightCount; ++i)
            {
                auto& light=scene.lights[i];
                light.radius=reader.f32(); light.angularRadius=reader.f32();
                light.areaSizeX=reader.f32(); light.areaSizeY=reader.f32(); light.spread=reader.f32();
                const auto packed=reader.u32();
                if ((packed & 0xff) > static_cast<std::uint32_t>(SceneAreaLightShape::Ellipse)
                    || (packed >> 8) > sceneRayVisibility::All) return false;
                light.areaShape=static_cast<SceneAreaLightShape>(packed & 0xff);
                light.visibility=static_cast<std::uint8_t>(packed >> 8);
                authored |= sceneLightHasShapeOrVisibility(light);
            }
            if (!authored) return false;
        }
        else if (extension==19)
        {
            // Only a scene that sets its visits writes this trailer.
            scene.smoothGlassCameraVisits = reader.u32();
            if (scene.smoothGlassCameraVisits == 0) return false;
        }
        else if (extension==17)
        {
            // Only a scene with a weighted vertex roughness writes this trailer.
            bool weighted = false;
            for (std::size_t i=0; i<counts[0]; ++i)
            {
                auto& lane=scene.vertices[vertexSlot(i)].roughness;
                lane={reader.f32(), reader.f32()};
                weighted |= lane[1] != 0.0f;
            }
            if (!weighted) return false;
        }
        else if (extension==18)
        {
            // Only a camera that focuses on an object writes this trailer.
            bool authored = false;
            for (std::size_t i=0; i<scene.cameraCount; ++i)
            {
                const auto object=reader.u32();
                if (object > HarmonicMIDI::grid::kMaximumCameraFocusObject) return false;
                scene.cameras[i].focusObject=object;
                authored |= object != 0;
            }
            if (!authored) return false;
        }
        else if (extension==11)
        {
            // Only a camera with lens fields writes this trailer.
            bool authored = false;
            for (std::size_t i=0; i<scene.cameraCount; ++i)
            {
                auto& camera=scene.cameras[i];
                camera.fStop=reader.f32(); camera.focusDistance=reader.f32();
                camera.bladeRotationRadians=reader.f32(); camera.apertureRatio=reader.f32();
                const auto blades=reader.u32();
                if (blades > 255) return false;
                camera.apertureBlades=static_cast<std::uint8_t>(blades);
                authored |= sceneCameraLensAuthored(camera);
            }
            if (!authored) return false;
        }
        else if (extension==10)
        {
            bool hidden = false;
            for (std::size_t i=0; i<scene.objectCount; ++i)
            {
                const auto mask = reader.u32();
                if (mask > sceneRayVisibility::All) return false;
                scene.objects[i].visibility = static_cast<std::uint8_t>(mask);
                hidden |= mask != sceneRayVisibility::All;
            }
            if (!hidden) return false;
        }
        else if (extension==12)
        {
            // Only an orthographic camera writes this trailer, so an all-perspective one is refused.
            bool orthographic = false;
            for (std::size_t i=0; i<scene.cameraCount; ++i)
            {
                const auto height=reader.f32();
                if (!std::isfinite(height) || height<0.0f) return false;
                scene.cameras[i].orthographicHeight=height;
                orthographic |= height>0.0f;
            }
            if (!orthographic) return false;
        }
        else if (extension==8)
        {
            // Only authored clouds write this trailer, so an all-clear one is refused.
            bool authored = false;
            for (std::size_t i=0; i<scene.lightCount; ++i)
            {
                auto& sky=scene.lights[i].sky;
                sky.cloudCoverage=reader.f32(); sky.cloudType=reader.f32();
                sky.cloudDrift=reader.f32(); sky.cloudHeadingDegrees=reader.f32();
                authored |= HarmonicMIDI::grid::analyticSkyCloudsAuthored(sky);
            }
            if (!authored) return false;
        }
        else if (extension==7)
        {
            bool authored = false;
            for (std::size_t i=0; i<scene.objectCount; ++i)
            {
                const auto flag = reader.u32();
                if (flag > 1) return false;
                scene.objects[i].authoredShading = flag != 0;
                authored |= flag != 0;
            }
            if (!authored) return false;
        }
        else if (extension==6)
        {
            // Only a hidden environment writes this trailer, so all-visible is refused.
            bool hidden = false;
            for (std::size_t i=0; i<scene.lightCount; ++i)
            {
                const auto visible = reader.u32();
                if (visible > 1) return false;
                scene.lights[i].environmentCameraVisible = visible != 0;
                hidden |= visible == 0;
            }
            if (!hidden) return false;
        }
        else
        {
            // A prior trailer with no objects is never written, so it is refused.
            if (scene.objectCount==0) return false;
            for (std::size_t i=0; i<scene.objectCount; ++i)
                scene.previousObjectTransforms[i]=readTransform(reader);
            scene.previousObjectTransformCount=scene.objectCount;
        }
    }
    if (!reader.ok || reader.offset != bytes.size()) return false;
    // The left-out ranges are still zeroed here; the caller validates once they are rebuilt.
    if (!replayed && !validateVisual3DScene(scene).valid()) return false;
    if (replayedMesh != nullptr)
        *replayedMesh = replayed ? std::optional { scene.objects[*replayed].id } : std::nullopt;
    return true;
}

inline std::string hex(const std::vector<std::uint8_t>& bytes)
{
    static constexpr char digits[] = "0123456789abcdef";
    std::string result(bytes.size() * 2, '0');
    for (std::size_t i = 0; i < bytes.size(); ++i)
    {
        result[i * 2] = digits[bytes[i] >> 4];
        result[i * 2 + 1] = digits[bytes[i] & 15];
    }
    return result;
}
inline bool unhex(std::string_view text, std::vector<std::uint8_t>& bytes)
{
    if (text.empty() || text.size() % 2 != 0
        || text.size() > kMaximumSceneBytes * 2) return false;
    const auto nibble = [](char value) -> int
    {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        return -1;
    };
    bytes.resize(text.size() / 2);
    for (std::size_t i = 0; i < bytes.size(); ++i)
    {
        const auto high = nibble(text[i * 2]); const auto low = nibble(text[i * 2 + 1]);
        if (high < 0 || low < 0) return false;
        bytes[i] = static_cast<std::uint8_t>((high << 4) | low);
    }
    return true;
}

// A scene's hex token is most of a plan payload, and every reader of the payload decodes it. An
// edit that leaves the scene unchanged produces the same token, so the decoded scene is kept
// against those exact bytes. Decoding is a pure function of the token, and a kept scene is const.
inline constexpr std::size_t kSceneTokenMemoMinimumBytes = std::size_t { 1 } << 20;
// A project with more heavy scenes than this reads the older ones again. Nothing is refused.
inline constexpr std::size_t kSceneTokenMemoEntries = 4;

inline ExactTokenMemo<HarmonicMIDI::grid::Visual3DScene>& sceneTokenMemo()
{
    // Never destroyed: helper threads may still read while the process exits.
    static auto* memo = new ExactTokenMemo<HarmonicMIDI::grid::Visual3DScene> (kSceneTokenMemoEntries);
    return *memo;
}

inline thread_local bool sceneTokenMemoSeeding = false;

// With replayedMesh, a scene whose mesh extension 21 left out decodes too, and is
// never kept: its left-out ranges are empty until the caller rebuilds them.
inline std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> decodeSceneToken(std::string_view token,
    std::optional<HarmonicMIDI::grid::SceneObjectId>* replayedMesh = nullptr)
{
    if (replayedMesh != nullptr) replayedMesh->reset();
    const bool memoized = token.size() >= kSceneTokenMemoMinimumBytes;
    if (memoized)
        if (auto kept = sceneTokenMemo().find(token)) return kept;
    std::vector<std::uint8_t> bytes;
    auto scene = std::make_shared<HarmonicMIDI::grid::Visual3DScene>();
    if (!unhex(token, bytes) || !decodeScene(bytes, *scene, replayedMesh)) return nullptr;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> decoded = std::move(scene);
    if (memoized && sceneTokenMemoSeeding && !(replayedMesh != nullptr && replayedMesh->has_value()))
        sceneTokenMemo().store(token, decoded);
    return decoded;
}

inline std::shared_ptr<const std::string> sceneTokenFor(const HarmonicMIDI::grid::Visual3DScene& scene,
                                                        HarmonicMIDI::grid::SceneObjectId replayedMesh = {})
{
    if (replayedMeshObject(scene, replayedMesh))
        return std::make_shared<const std::string>(hex(encodeScene(scene, replayedMesh)));
    if (auto kept = sceneTokenMemo().tokenOf(&scene)) return kept;
    return std::make_shared<const std::string>(hex(encodeScene(scene)));
}
inline bool attribute(std::string_view xml, std::string_view name, std::string_view& value)
{
    const std::string needle = std::string(" ") + std::string(name) + "=\"";
    const auto begin = xml.find(needle);
    if (begin == std::string_view::npos) return false;
    const auto content = begin + needle.size();
    const auto end = xml.find('"', content);
    if (end == std::string_view::npos) return false;
    value = xml.substr(content, end - content);
    return true;
}
inline bool decimal(std::string_view text, std::uint32_t& value)
{
    if (text.empty()) return false;
    std::uint64_t parsed = 0;
    for (const auto character : text)
    {
        if (character < '0' || character > '9') return false;
        parsed = parsed * 10 + static_cast<unsigned>(character - '0');
        if (parsed > 0xffffffffull) return false;
    }
    value = static_cast<std::uint32_t>(parsed);
    return true;
}
} // namespace detail

inline bool hasAnalyticSky(const HarmonicMIDI::grid::Visual3DScene& scene) noexcept
{
    for (std::size_t i=0;i<scene.lightCount && i<scene.lights.size();++i)
        if (scene.lights[i].sky.version!=0) return true;
    return false;
}

inline bool hasEnvironmentImage(const HarmonicMIDI::grid::Visual3DScene& scene) noexcept
{
    for (std::size_t i=0; i<scene.lightCount && i<scene.lights.size(); ++i)
        if (scene.lights[i].environmentImage.version!=0) return true;
    return false;
}

inline bool hasSpectralSource(const HarmonicMIDI::grid::Visual3DScene& scene) noexcept
{
    for (std::size_t i=0; i<scene.lightCount && i<scene.lights.size(); ++i)
        if (scene.lights[i].spectralSource.version!=0) return true;
    return false;
}

inline bool valid(const Payload& payload) noexcept
{
    return (payload.version==kWireVersion || (payload.scene && !hasEnvironmentImage(*payload.scene) && !hasSpectralSource(*payload.scene)
            && (payload.version==4 || (payload.version==3 && !hasAnalyticSky(*payload.scene))))) && supports(payload.output)
        && payload.extent.width > 0 && payload.extent.height > 0
        && payload.scene != nullptr
        && HarmonicMIDI::grid::validateVisual3DScene(*payload.scene).valid()
        && payload.scene->objectCount == 1 && payload.scene->materialCount == 1
        && payload.scene->cameraCount == 1;
}

inline renderpassoutput::Description description(const Payload& payload)
{
    renderpassoutput::Description result;
    result.extent = payload.extent;
    const auto requirements = renderpassoutput::requirements(payload.output);
    result.attachments.push_back({ payload.output, requirements.format,
                                   requirements.colorSpace, payload.extent });
    return result;
}

// Scene tokens decoded on this thread while this lives are kept for later readers. Only the readers
// of a compiled plan open one. A timeline Geometry3D scene decodes a new token on every sample, and
// keeping those would copy a whole scene per frame and push the plan's own scene out.
class ScopedSceneTokenMemoSeeding final
{
public:
    ScopedSceneTokenMemoSeeding() : previous(detail::sceneTokenMemoSeeding) { detail::sceneTokenMemoSeeding = true; }
    ~ScopedSceneTokenMemoSeeding() { detail::sceneTokenMemoSeeding = previous; }
    ScopedSceneTokenMemoSeeding(const ScopedSceneTokenMemoSeeding&) = delete;
    ScopedSceneTokenMemoSeeding& operator=(const ScopedSceneTokenMemoSeeding&) = delete;

private:
    bool previous;
};

inline std::string serialize(const Payload& payload)
{
    if (!valid(payload)) return {};
    const auto bytes = detail::encodeScene(*payload.scene);
    // Keep legacy RGB XML byte-identical to the released v3 operation.
    const auto version=hasEnvironmentImage(*payload.scene) || hasSpectralSource(*payload.scene) ? kWireVersion : hasAnalyticSky(*payload.scene) ? 4u : 3u;
    return "<SceneAov schemaVersion=\"" + std::to_string(version) + "\" output=\""
        + std::string(renderpassoutput::token(payload.output)) + "\" width=\""
        + std::to_string(payload.extent.width) + "\" height=\""
        + std::to_string(payload.extent.height) + "\" scene=\""
        + detail::hex(bytes) + "\"/>";
}

inline bool parse(std::string_view xml, Payload& destination)
{
    if (xml.size() > detail::kMaximumSceneBytes * 2 + 256
        || xml.rfind("<SceneAov ", 0) != 0 || xml.size() < 3
        || xml.substr(xml.size() - 2) != "/>") return false;
    std::string_view versionText, outputText, widthText, heightText, sceneText;
    if (!detail::attribute(xml, "schemaVersion", versionText)
        || !detail::attribute(xml, "output", outputText)
        || !detail::attribute(xml, "width", widthText)
        || !detail::attribute(xml, "height", heightText)
        || !detail::attribute(xml, "scene", sceneText)) return false;
    std::uint32_t version = 0, width = 0, height = 0;
    if (!detail::decimal(versionText, version) || !detail::decimal(widthText, width)
        || !detail::decimal(heightText, height) || (version != 3 && version != 4 && version != kWireVersion)) return false;
    renderpassoutput::Output output = renderpassoutput::Output::Count;
    for (const auto candidate : kOutputs)
        if (outputText == renderpassoutput::token(candidate)) output = candidate;
    if (!supports(output)) return false;
    const auto decoded = detail::decodeSceneToken(sceneText);
    if (!decoded) return false;
    const auto& scene = *decoded;
    if ((version==3 && (hasAnalyticSky(scene) || hasEnvironmentImage(scene)))
        || (version==4 && (!hasAnalyticSky(scene) || hasEnvironmentImage(scene)))
        || (version==5 && !hasEnvironmentImage(scene))) return false;
    Payload candidate;
    candidate.version = version; candidate.output = output;
    candidate.extent = { width, height };
    candidate.scene = decoded;
    if (!valid(candidate)) return false;
    destination = std::move(candidate);
    return true;
}
} // namespace sceneaov
