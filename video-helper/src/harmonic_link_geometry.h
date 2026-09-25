#pragma once

#include "geometry_core_backend.h"
#include "../../shared/VisualHarmonicLinkGeometryContract.h"

namespace videohelper::harmonicgeometry
{
struct RenderedFrame final
{
    geometry::NativeGeometryExecution native;
    std::shared_ptr<const geometry::AdmittedPlanValue> admittedGeometry;
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> score;
};

// Frame rows and signed source identities are read from the same immutable
// owner used by Note Instanced Mesh. Geometry IDs only identify ribbon corners.
inline std::optional<videowire::geometry::ValueDescriptor> lower(
    const visualharmonicgeometry::Mapping& mapping,
    const std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame>& frame,
    std::string& error)
{
    using namespace videowire::geometry;
    error.clear();
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
    for (const auto& attribute : visualharmonicgeometry::geometryContract().attributes)
        value.attributes.push_back({attribute, {}});
    if (!frame) return value;

    arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
    inputs.canonicalBlockCFrame = frame;
    inputs.noteInstanceMapping = mapping.notes;
    const auto notes = arbitgpu::prepareNativeNoteInstances(inputs);
    std::array<int, arbitblockc::kMaxNotes> instanceForRow;
    instanceForRow.fill(-1);
    for (std::size_t index = 0; index < notes.count; ++index)
        instanceForRow[notes.canonicalRows[index]] = static_cast<int>(index);
    auto& mesh = std::get<GeometryData>(value.data);
    mesh.positions.reserve(static_cast<std::size_t>(frame->linkRows()) * 4);
    mesh.vertexIds.reserve(static_cast<std::size_t>(frame->linkRows()) * 4);
    mesh.indices.reserve(static_cast<std::size_t>(frame->linkRows()) * 6);
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
        Vec3 side { -direction.y, direction.x, 0.0f };
        if (std::abs(direction.z) > length * 0.9f)
            side = { direction.z, 0.0f, -direction.x };
        const auto sideLength = std::sqrt(side.x * side.x + side.y * side.y + side.z * side.z);
        const auto scale = mapping.notes.meshScale * 0.5f / sideLength;
        side = { side.x * scale, side.y * scale, side.z * scale };
        const auto first = static_cast<std::uint32_t>(mesh.positions.size());
        mesh.positions.insert(mesh.positions.end(), {
            {a[0] - side.x, a[1] - side.y, a[2] - side.z},
            {a[0] + side.x, a[1] + side.y, a[2] + side.z},
            {b[0] - side.x, b[1] - side.y, b[2] - side.z},
            {b[0] + side.x, b[1] + side.y, b[2] + side.z} });
        const auto identity = frame->linkIdentities[static_cast<std::size_t>(row)];
        const auto unsignedIdentity = identity < 0
            ? static_cast<std::uint64_t>(-identity) * 2 - 1
            : static_cast<std::uint64_t>(identity) * 2;
        if (unsignedIdentity > (kMaximumStableId - 4) / 4)
        {
            error = "Harmonic link identity exceeds stable geometry corner bounds";
            return std::nullopt;
        }
        for (std::uint64_t corner = 0; corner < 4; ++corner)
            mesh.vertexIds.push_back(unsignedIdentity * 4 + corner + 1);
        mesh.indices.insert(mesh.indices.end(), {first, first + 2, first + 1,
                                                 first + 1, first + 2, first + 3});
        const double attributes[] { static_cast<double>(identity),
            static_cast<double>(notes.identities[static_cast<std::size_t>(master)]),
            static_cast<double>(notes.identities[static_cast<std::size_t>(slave)]),
            frame->linkTextureValues[offset + 2], frame->linkTextureValues[offset + 3] };
        for (std::size_t attribute = 0; attribute < 5; ++attribute)
            for (int face = 0; face < 2; ++face)
                value.attributes[attribute].elements.push_back({{attributes[attribute], 0, 0, 0}});
    }
    return value;
}

template <typename LayerDesc>
inline bool render(const visualharmonicgeometry::Mapping& mapping,
                   const geometry::PlanOwnerIdentity& owner, int width, int height,
                   LayerDesc& layer, std::string& error, bool linearColor = false)
{
    if (!owner.valid() || width <= 0 || height <= 0)
    {
        error = "Harmonic Link Geometry requires an exact render owner and output extent";
        return false;
    }
    auto value = lower(mapping, layer.canonicalBlockCFrame, error);
    if (!value) return false;
    if (std::get<videowire::geometry::GeometryData>(value->data).positions.empty())
    {
        layer.opacity = 0.0f;
        return true;
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
    auto execution = geometry::executeNativeGeometry(arbitgpu::nativeFixtureSceneBackend(),
        *admission, static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), error, false, inputs);
    if (!execution) return false;
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
    return true;
}
} // namespace videohelper::harmonicgeometry
