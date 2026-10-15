#pragma once

#include "geometry_core_backend.h"
#include "harmonic_probe_diagnostic.h"
#include "harmonic_link_shape.h"
#include "../../shared/VisualHarmonicLinkGeometryContract.h"
#include <algorithm>
#include <cmath>
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
inline constexpr videowire::geometry::StableId kLinkMaterialId = 0x00F0F0F0ull;

// Frame rows and signed source identities are read from the same immutable
// owner used by Note Instanced Mesh. Geometry IDs identify every emitted corner.
inline std::optional<videowire::geometry::ValueDescriptor> lower(
    const visualharmonicgeometry::Mapping& mapping,
    const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>& frame,
    std::string& error, harmonicprobe::Receipt* receipt = nullptr)
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
    std::array<int, arbitblockc::kMaxNotes> instanceForRow;
    instanceForRow.fill(-1);
    for (std::size_t index = 0; index < notes.count; ++index)
        instanceForRow[notes.canonicalRows[index]] = static_cast<int>(index);
    // Pass one collects the links that have a segment. The link count then fixes the level of detail
    // for the whole set, and pass two emits them.
    struct CollectedLink final
    {
        int row = 0;
        int master = 0;
        int slave = 0;
        Vec3 start {};
        Vec3 end {};
        Vec3 side {};
    };
    std::vector<CollectedLink> collected;
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
        const Vec3 start { a[0], a[1], a[2] };
        const Vec3 end { b[0], b[1], b[2] };
        const Vec3 direction { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
        const auto length = std::sqrt(direction.x * direction.x + direction.y * direction.y
                                       + direction.z * direction.z);
        if (!std::isfinite(length))
        {
            error = "Harmonic link endpoint coordinates exceed the geometry bounds";
            return std::nullopt;
        }
        // Coincident notes have no connecting segment.
        if (length <= 0.000001f) continue;
        collected.push_back({ row, master, slave, start, end,
                              linkSideHalfWidth(direction, mapping.notes.meshScale) });
    }
    if (!visualharmonicgeometry::admitLinkCount(mapping.shape, mapping.curvaturePercent,
                                                mapping.segments, collected.size(), error))
        return std::nullopt;
    const auto effective = visualharmonicgeometry::effectiveLinkLayout(
        mapping.shape, mapping.curvaturePercent, mapping.segments, collected.size());
    // Corner IDs stride by the requested corner count, so each link keeps its ID range when the
    // level of detail steps down.
    const auto stride = static_cast<std::uint64_t>(visualharmonicgeometry::effectiveLinkLayout(
        mapping.shape, mapping.curvaturePercent, mapping.segments, 1).layout.vertices);
    const auto corners = static_cast<std::uint64_t>(effective.layout.vertices);
    auto& mesh = std::get<GeometryData>(value.data);
    mesh.positions.reserve(collected.size() * effective.layout.vertices);
    mesh.vertexIds.reserve(collected.size() * effective.layout.vertices);
    mesh.indices.reserve(collected.size() * effective.layout.triangles * 3);
    for (const auto& link : collected)
    {
        appendLinkShape(mesh, mapping.shape, mapping.curvaturePercent, effective,
                        link.start, link.end, link.side);
        const auto identity = frame->linkIdentities[static_cast<std::size_t>(link.row)];
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
        const auto offset = static_cast<std::size_t>(link.row) * 4;
        const double attributes[] { static_cast<double>(identity),
            static_cast<double>(notes.identities[static_cast<std::size_t>(link.master)]),
            static_cast<double>(notes.identities[static_cast<std::size_t>(link.slave)]),
            frame->linkTextureValues[offset + 2], frame->linkTextureValues[offset + 3] };
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
        layer.opacity = 0.0f;
        return true;
    }
    auto& positions = std::get<videowire::geometry::GeometryData>(value->data).positions;
    auto minimum = positions.front();
    auto maximum = minimum;
    for (const auto& position : positions)
    {
        minimum.x = std::min(minimum.x, position.x);
        minimum.y = std::min(minimum.y, position.y);
        minimum.z = std::min(minimum.z, position.z);
        maximum.x = std::max(maximum.x, position.x);
        maximum.y = std::max(maximum.y, position.y);
        maximum.z = std::max(maximum.z, position.z);
    }
    // Keep the musical mapping in lower(). Fit the image value before native
    // admission so the scene and its retained draw use the same coordinates.
    const auto visibleHeight = 10.0f * std::tan(0.7853981634f * 0.5f);
    const auto visibleWidth = visibleHeight * static_cast<float>(width) / static_cast<float>(height);
    const auto extentX = std::max(maximum.x - minimum.x, 0.001f);
    const auto extentY = std::max(maximum.y - minimum.y, 0.001f);
    const auto displayScale = std::clamp(std::min(visibleWidth * 0.65f / extentX,
                                                visibleHeight * 0.65f / extentY), 0.01f, 3.0f);
    const videowire::geometry::Vec3 center {
        (minimum.x + maximum.x) * 0.5f,
        (minimum.y + maximum.y) * 0.5f,
        (minimum.z + maximum.z) * 0.5f };
    for (auto& position : positions)
    {
        position.x = (position.x - center.x) * displayScale;
        position.y = (position.y - center.y) * displayScale;
        position.z = (position.z - center.z) * displayScale;
    }
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
