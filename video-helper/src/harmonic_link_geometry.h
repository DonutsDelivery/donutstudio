#pragma once

#include "geometry_core_backend.h"
#include "harmonic_probe_diagnostic.h"
#include "harmonic_link_shape.h"
#include "../../shared/VisualHarmonicLinkGeometryContract.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace videohelper::harmonicgeometry
{
struct RenderedFrame final
{
    geometry::NativeGeometryExecution native;
    std::shared_ptr<const geometry::AdmittedPlanValue> admittedGeometry;
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> score;
};

// Light neutral link material. The backend maps each low byte to 0.2 + 0.7 * byte / 255, about 0.86 here.
// Bit 24 marks the material self-lit: the fixed Geometry Core light misses most ribbon faces, so the
// backend adds a tinted emission and the ribbons stay visible without any score or audio.
inline constexpr videowire::geometry::StableId kLinkMaterialId = 0x01F0F0F0ull;

// Frame rows and signed source identities are read from the same immutable
// owner used by Note Instanced Mesh. Geometry IDs identify every emitted corner.
inline std::optional<videowire::geometry::ValueDescriptor> lower(
    const visualharmonicgeometry::Mapping& mapping,
    const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>& frame,
    std::string& error, harmonicprobe::Receipt* receipt = nullptr, bool budgeted = true)
{
    using namespace videowire::geometry;
    error.clear();
    if (receipt != nullptr) ++receipt->lowerCallCount;
    if (!visualharmonicgeometry::valid(mapping)
        || (frame && !canonicalblockc::valid(frame)))
    {
        error = "Harmonic Link Geometry requires a valid mapping and canonical score frame";
        return std::nullopt;
    }
    ValueDescriptor value;
    value.carrier = CarrierKind::geometry3D;
    value.stableId = value.sourceStableId = mapping.stableId;
    value.sourceRevision = frame ? std::max<std::uint64_t>(1, frame->key.scoreGeneration) : 1;
    value.overflow = OverflowPolicy::reject;
    value.dispatchCount = 1;
    value.operations.push_back({mapping.stableId, OperationCode::copy, 0, mapping.stableId, {}});
    // Set Material follows the copy. Without it, the backend falls back to material 1, which is dark.
    if (!lowerSetMaterial(value, kLinkMaterialId, error)) return std::nullopt;
    for (const auto& attribute : visualharmonicgeometry::geometryContract().attributes)
        value.attributes.push_back({attribute, {}});
    if (!frame) return value;

    // Pass one collects the links that have a segment. The link count then fixes the level of detail
    // for the whole set, and pass two emits them.
    struct CollectedLink final
    {
        std::int64_t identity = 0;
        std::int64_t masterIdentity = 0;
        std::int64_t slaveIdentity = 0;
        float slaveHarmonic = 0;
        float masterHarmonic = 0;
        Vec3 start {};
        Vec3 end {};
        Vec3 side {};
    };
    std::vector<CollectedLink> collected;
    const auto collect = [&](std::int64_t identity, std::int64_t masterIdentity, std::int64_t slaveIdentity,
                             float slaveHarmonic, float masterHarmonic, Vec3 start, Vec3 end)
    {
        const Vec3 direction { end.x - start.x, end.y - start.y, end.z - start.z };
        const auto length = std::sqrt(direction.x * direction.x + direction.y * direction.y
                                       + direction.z * direction.z);
        if (!std::isfinite(length))
        {
            error = "Harmonic link endpoint coordinates exceed the geometry bounds";
            return false;
        }
        // Coincident notes have no connecting segment.
        if (length > 0.000001f)
            collected.push_back({ identity, masterIdentity, slaveIdentity, slaveHarmonic, masterHarmonic,
                                  start, end, linkSideHalfWidth(direction, mapping.notes.meshScale) });
        return true;
    };
    if (budgeted)
    {
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
        inputs.canonicalBlockCFrame = frame;
        inputs.noteInstanceMapping = mapping.notes;
        const auto notes = arbitgpu::prepareNativeNoteInstances(inputs);
        if (mapping.notes.schemaVersion == visualnoteinstancing::kDirectAxesMappingSchemaVersion
            && !notes.admitted)
        {
            error = "Harmonic link endpoint coordinates exceed the geometry bounds";
            return std::nullopt;
        }
        std::vector<int> instanceForRow(static_cast<std::size_t>(std::max(frame->noteRows(), 0)), -1);
        for (std::size_t index = 0; index < notes.count; ++index)
            instanceForRow[notes.canonicalRows[index]] = static_cast<int>(index);
        collected.reserve(static_cast<std::size_t>(frame->linkRows()));
        for (int row = 0; row < frame->linkRows(); ++row)
        {
            const auto offset = static_cast<std::size_t>(row) * 4;
            for (std::size_t channel = 0; channel < 2; ++channel)
            {
                const auto endpoint = frame->linkTextureValues[offset + channel];
                if (endpoint < 0 || endpoint >= frame->noteRows() || std::floor(endpoint) != endpoint)
                {
                    error = "Harmonic link references an invalid canonical note row";
                    return std::nullopt;
                }
            }
            const auto slaveRow = static_cast<int>(frame->linkTextureValues[offset]);
            const auto masterRow = static_cast<int>(frame->linkTextureValues[offset + 1]);
            const auto slave = instanceForRow[static_cast<std::size_t>(slaveRow)];
            const auto master = instanceForRow[static_cast<std::size_t>(masterRow)];
            if (slave < 0 || master < 0) continue;
            const auto& a = notes.transforms[static_cast<std::size_t>(master)];
            const auto& b = notes.transforms[static_cast<std::size_t>(slave)];
            if (!collect(frame->linkIdentities[static_cast<std::size_t>(row)],
                         notes.identities[static_cast<std::size_t>(master)],
                         notes.identities[static_cast<std::size_t>(slave)],
                         frame->linkTextureValues[offset + 2], frame->linkTextureValues[offset + 3],
                         { a[0], a[1], a[2] }, { b[0], b[1], b[2] }))
                return std::nullopt;
        }
    }
    else
    {
        // A retained scene reads the whole score. The frame's rows stop at the
        // shader textures' 128 notes and 256 links, and this mesh uses no texture.
        const auto& score = *frame->sourceScore;
        const auto beat = frame->packedBeat;
        std::unordered_map<int, Vec3> residents;
        residents.reserve(score.notes.size());
        for (const auto& note : score.notes)
        {
            if (note.muted || arbitblockc::BlockCPacker::residency(
                    note, beat, frame->historyBeats(), frame->lookaheadBeats()) <= 0)
                continue;
            const auto texels = arbitblockc::BlockCPacker::noteTexels(note, score, beat, -1);
            const auto axis = [&](visualnoteinstancing::MappingAxis mappingAxis)
            {
                return arbitgpu::nativeNoteAxisValue(mappingAxis, texels.data(), frame->rootFrequencyHz);
            };
            residents[note.id] = { axis(mapping.notes.x) * mapping.notes.xScale,
                                   axis(mapping.notes.y) * mapping.notes.yScale,
                                   axis(mapping.notes.z) * mapping.notes.zScale };
        }
        collected.reserve(score.links.size());
        for (const auto& link : score.links)
        {
            const auto master = residents.find(link.masterNoteId);
            const auto slave = residents.find(link.slaveNoteId);
            if (master == residents.end() || slave == residents.end()) continue;
            if (!collect(link.id, link.masterNoteId, link.slaveNoteId,
                         static_cast<float>(link.slaveHarmonic), static_cast<float>(link.masterHarmonic),
                         master->second, slave->second))
                return std::nullopt;
        }
    }
    // The raster route's contract has a vertex budget. A retained scene has none,
    // so it keeps full detail at any link count.
    if (budgeted && !visualharmonicgeometry::admitLinkCount(mapping.shape, mapping.curvaturePercent,
                                                            mapping.segments, collected.size(), error))
        return std::nullopt;
    auto effective = visualharmonicgeometry::effectiveLinkLayout(
        mapping.shape, mapping.curvaturePercent, mapping.segments, budgeted ? collected.size() : 1);
    // A retained scene is lit and path traced, so its links are round. The flat strip stays on the
    // raster route, which draws them self-lit.
    if (!budgeted) effective.layout = linkTubeLayout(mapping.shape, effective);
    // Corner IDs stride by the requested corner count, so each link keeps its ID range when the
    // level of detail steps down.
    const auto stride = budgeted ? static_cast<std::uint64_t>(visualharmonicgeometry::effectiveLinkLayout(
        mapping.shape, mapping.curvaturePercent, mapping.segments, 1).layout.vertices)
        : static_cast<std::uint64_t>(effective.layout.vertices);
    const auto corners = static_cast<std::uint64_t>(effective.layout.vertices);
    auto& mesh = std::get<GeometryData>(value.data);
    mesh.positions.reserve(collected.size() * effective.layout.vertices);
    mesh.vertexIds.reserve(collected.size() * effective.layout.vertices);
    mesh.indices.reserve(collected.size() * effective.layout.triangles * 3);
    for (const auto& link : collected)
    {
        if (budgeted)
            appendLinkShape(mesh, mapping.shape, mapping.curvaturePercent, effective,
                            link.start, link.end, link.side);
        else
            appendLinkTube(mesh, mapping.shape, mapping.curvaturePercent, effective,
                           link.start, link.end, link.side);
        const auto identity = link.identity;
        const auto unsignedIdentity = identity < 0
            ? static_cast<std::uint64_t>(-identity) * 2 - 1
            : static_cast<std::uint64_t>(identity) * 2;
        if (unsignedIdentity > (kMaximumStableId - stride) / stride)
        {
            error = "Harmonic link identity exceeds stable geometry corner bounds";
            return std::nullopt;
        }
        for (std::uint64_t corner = 0; corner < corners; ++corner)
            mesh.vertexIds.push_back(unsignedIdentity * stride + corner + 1);
        const double attributes[] { static_cast<double>(identity),
            static_cast<double>(link.masterIdentity), static_cast<double>(link.slaveIdentity),
            link.slaveHarmonic, link.masterHarmonic };
        for (std::size_t attribute = 0; attribute < 5; ++attribute)
            for (std::size_t triangle = 0; triangle < effective.layout.triangles; ++triangle)
                value.attributes[attribute].elements.push_back({{attributes[attribute], 0, 0, 0}});
    }
    if (receipt != nullptr)
    {
        receipt->emittedVertexCount += mesh.positions.size();
        receipt->emittedIndexCount += mesh.indices.size();
    }
    return value;
}

// A project without links would otherwise draw nothing. These ten ribbons stand in until the score
// supplies real ones. They use the mapping's own shape and detail, and a width of at least 0.34 so
// the sculpture reads at the module's resting state. Identities are fixed, so repeated frames match.
inline std::optional<videowire::geometry::ValueDescriptor> restingSculpture(
    const visualharmonicgeometry::Mapping& mapping, std::string& error, bool round = false)
{
    using namespace videowire::geometry;
    auto value = lower(mapping, nullptr, error);
    if (!value) return std::nullopt;
    static constexpr std::array<std::array<float, 3>, 7> points {{
        { -2.4f, -0.9f, 0.0f }, { -1.2f, 0.5f, 0.3f }, { 0.0f, -0.6f, -0.2f },
        { 1.1f, 0.9f, 0.2f }, { 2.3f, -0.2f, 0.0f }, { 0.2f, 1.1f, -0.3f },
        { -0.4f, 0.1f, 0.4f } }};
    static constexpr std::array<std::array<int, 2>, 10> pairs {{
        { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 4 }, { 1, 5 },
        { 5, 3 }, { 0, 2 }, { 2, 4 }, { 6, 1 }, { 6, 3 } }};
    struct RestingLink final { Vec3 start; Vec3 end; Vec3 side; };
    const auto width = std::max(mapping.notes.meshScale, 0.34f);
    std::vector<RestingLink> links;
    links.reserve(pairs.size());
    for (const auto& pair : pairs)
    {
        const auto& a = points[static_cast<std::size_t>(pair[0])];
        const auto& b = points[static_cast<std::size_t>(pair[1])];
        const Vec3 start { a[0], a[1], a[2] };
        const Vec3 end { b[0], b[1], b[2] };
        const Vec3 direction { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
        links.push_back({ start, end, linkSideHalfWidth(direction, width) });
    }
    if (!visualharmonicgeometry::admitLinkCount(mapping.shape, mapping.curvaturePercent,
                                                mapping.segments, links.size(), error))
        return std::nullopt;
    auto effective = visualharmonicgeometry::effectiveLinkLayout(
        mapping.shape, mapping.curvaturePercent, mapping.segments, links.size());
    if (round) effective.layout = linkTubeLayout(mapping.shape, effective);
    const auto stride = round ? static_cast<std::uint64_t>(effective.layout.vertices)
        : static_cast<std::uint64_t>(visualharmonicgeometry::effectiveLinkLayout(
            mapping.shape, mapping.curvaturePercent, mapping.segments, 1).layout.vertices);
    const auto corners = static_cast<std::uint64_t>(effective.layout.vertices);
    auto& mesh = std::get<GeometryData>(value->data);
    for (std::size_t index = 0; index < links.size(); ++index)
    {
        if (round)
            appendLinkTube(mesh, mapping.shape, mapping.curvaturePercent, effective,
                           links[index].start, links[index].end, links[index].side);
        else
            appendLinkShape(mesh, mapping.shape, mapping.curvaturePercent, effective,
                            links[index].start, links[index].end, links[index].side);
        const auto unsignedIdentity = static_cast<std::uint64_t>(index + 1) * 2;
        for (std::uint64_t corner = 0; corner < corners; ++corner)
            mesh.vertexIds.push_back(unsignedIdentity * stride + corner + 1);
        const double attributes[] { static_cast<double>(index + 1), static_cast<double>(index),
            static_cast<double>(index + 1), 3.0, 2.0 };
        for (std::size_t attribute = 0; attribute < 5; ++attribute)
            for (std::size_t triangle = 0; triangle < effective.layout.triangles; ++triangle)
                value->attributes[attribute].elements.push_back({{attributes[attribute], 0, 0, 0}});
    }
    return value;
}

// How many distinct links a lowered value draws. Every triangle carries its link's identity.
inline std::size_t linkCount(const videowire::geometry::ValueDescriptor& value)
{
    if (value.attributes.empty()) return 0;
    std::unordered_set<double> identities;
    for (const auto& element : value.attributes[0].elements)
        identities.insert(element.components[0]);
    return identities.size();
}

// This frame's mesh for one link object of a retained scene: the score's links,
// or the resting sculpture while the score has none.
inline bool replaceSceneMesh(const videowire::geometry::HarmonicLinkSceneBinding& binding,
    const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>& frame,
    int width, int height, HarmonicMIDI::grid::Visual3DScene& scene, std::string& error)
{
    auto value = lower(binding.mapping, frame, error, harmonicprobe::activeReceipt, false);
    if (!value) return false;
    const bool resting = std::get<videowire::geometry::GeometryData>(value->data).positions.empty();
    if (resting)
    {
        if (harmonicprobe::activeReceipt != nullptr)
            ++harmonicprobe::activeReceipt->emptyLayerCount;
        value = restingSculpture(binding.mapping, error, true);
        if (!value) return false;
    }
    auto& mesh = std::get<videowire::geometry::GeometryData>(value->data);
    // Printed when the counts change, so a frame's links can be told from its pixels.
    {
        const auto drawn = linkCount(*value);
        const auto authored = frame ? frame->sourceScore->links.size() : std::size_t {};
        static std::atomic<std::uint64_t> lastCounts { ~std::uint64_t {} };
        const auto counts = (static_cast<std::uint64_t>(drawn) << 32) | authored;
        if (lastCounts.exchange(counts) != counts)
            std::fprintf(stderr, "[links] scene mesh links=%zu scoreLinks=%zu vertices=%zu resting=%d\n",
                         drawn, authored, mesh.positions.size(), resting ? 1 : 0);
    }
    if (binding.fitToView)
        videowire::geometry::fitHarmonicLinksToView(mesh.positions, width, height);
    return videowire::geometry::replaceHarmonicLinkSceneMesh(binding, mesh, scene, error);
}

template <typename LayerDesc>
inline bool render(const visualharmonicgeometry::Mapping& mapping,
                   const geometry::PlanOwnerIdentity& owner, int width, int height,
                   LayerDesc& layer, std::string& error, bool linearColor,
                   arbitgpu::NativeFixturePrepareUse prepareUse)
{
    if (!owner.valid() || width <= 0 || height <= 0)
    {
        error = "Harmonic Link Geometry requires an exact render owner and output extent";
        return false;
    }
    auto value = lower(mapping, layer.canonicalBlockCFrame, error, harmonicprobe::activeReceipt);
    if (!value) return false;
    if (std::get<videowire::geometry::GeometryData>(value->data).positions.empty())
    {
        if (harmonicprobe::activeReceipt != nullptr)
            ++harmonicprobe::activeReceipt->emptyLayerCount;
        // No links yet. Draw the resting sculpture so a silent project still shows the module.
        value = restingSculpture(mapping, error);
        if (!value) return false;
    }
    // Keep the musical mapping in lower(). Fit the image value before native
    // admission so the scene and its retained draw use the same coordinates.
    videowire::geometry::fitHarmonicLinksToView(
        std::get<videowire::geometry::GeometryData>(value->data).positions, width, height);
    const auto contract = visualharmonicgeometry::geometryContract();
    const auto admittedValue = videowire::geometry::admitValue(std::move(*value), contract, {}, error);
    if (!admittedValue) return false;
    const auto bytes = videowire::geometry::lowerRuntimePlan(contract, *admittedValue);
    geometry::NativeGeometryCoreCapabilitySource source(arbitgpu::nativeFixtureSceneBackend());
    // A frame owns its resources. A later score frame never replaces an image
    // still held by the compositor or another preview/export request.
    geometry::GeometryCorePlanRuntime runtime(source, 1);
    const auto admission = owner.use == geometry::PlanUse::preview
        ? runtime.admitPreview(owner, bytes, {}, {}, error)
        : runtime.admitExport(owner, bytes, {}, {}, error);
    if (!admission) return false;
    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.linearColor = linearColor;
    // The caller's use decides whether a still-compiling program defers (Preview) or blocks (export).
    auto execution = geometry::executeNativeGeometry(arbitgpu::nativeFixtureSceneBackend(),
        *admission, static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), error, false, inputs,
        nullptr, {}, 0.0, {}, nullptr, std::numeric_limits<double>::quiet_NaN(), nullptr, {}, {}, prepareUse);
    if (!execution) return false;
    if (auto* receipt = harmonicprobe::activeReceipt; receipt != nullptr && !linearColor)
    {
        arbitgpu::NativeRawPassPixels pixels;
        receipt->nativeColorRead = execution->frame->readRawPass(
            renderpassoutput::Output::Color, pixels, receipt->nativeColorReadError);
        if (receipt->nativeColorRead)
        {
            const bool rgba = pixels.format == arbitgpu::NativeTexturePixelFormat::Rgba8Unorm;
            const bool bgra = pixels.format == arbitgpu::NativeTexturePixelFormat::Bgra8Unorm;
            receipt->nativeColorRead = (rgba || bgra)
                && pixels.width == static_cast<std::uint32_t>(width)
                && pixels.height == static_cast<std::uint32_t>(height)
                && pixels.bytes.size() == static_cast<std::size_t>(width) * height * 4u;
            if (!receipt->nativeColorRead)
                receipt->nativeColorReadError = "Harmonic native color readback has an invalid format or extent";
            else
            {
                int left = width, top = height, right = -1, bottom = -1;
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x)
                    {
                        const auto offset = (static_cast<std::size_t>(y) * width + x) * 4u;
                        const auto red = rgba ? offset : offset + 2;
                        const auto blue = rgba ? offset + 2 : offset;
                        const auto contrast = std::max({ std::abs(int(pixels.bytes[red]) - 7),
                            std::abs(int(pixels.bytes[offset + 1]) - 10),
                            std::abs(int(pixels.bytes[blue]) - 18) });
                        receipt->nativeMaximumContrast = std::max(receipt->nativeMaximumContrast, contrast);
                        if (contrast <= 8) continue;
                        ++receipt->nativeForegroundPixels;
                        left = std::min(left, x); top = std::min(top, y);
                        right = std::max(right, x); bottom = std::max(bottom, y);
                    }
                if (right >= left)
                    receipt->nativeForegroundBounds = { left, top, right - left + 1, bottom - top + 1 };
            }
        }
    }
    auto retained = std::make_shared<const RenderedFrame>(RenderedFrame {
        std::move(*execution), admission->plan, layer.canonicalBlockCFrame });
    const auto& frame = retained->native.frame;
    const auto view = frame->colorTextureViewHandle();
    if (frame->backend() == "opengl" && view > std::numeric_limits<unsigned>::max())
    {
        error = "Harmonic link texture view exceeds compositor bounds";
        return false;
    }
    layer.texture = frame->backend() == "opengl" ? static_cast<unsigned>(view) : 0;
    layer.nativeTextureBackend = frame->backend();
    layer.nativeTextureView = view;
    layer.nativeTextureDescriptor = frame->colorTextureDescriptor();
    layer.nativeTextureOwner = std::move(retained);
    layer.texWidth = width;
    layer.texHeight = height;
    if (harmonicprobe::activeReceipt != nullptr)
        ++harmonicprobe::activeReceipt->nativeLayerCount;
    return true;
}
} // namespace videohelper::harmonicgeometry
