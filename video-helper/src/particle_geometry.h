#pragma once

#include "particle_history.h"

namespace videorender {
struct ParticleGeometryBinding
{
    std::shared_ptr<const videowire::geometry::ValueDescriptor> emitters, attractors, mesh, points, body;
    float scale = 0.25f, centerX = 0.5f, centerY = 0.5f, centerZ = 0.5f;
    int projection = 0;
};

struct ParticleGeometryFrame
{
    // Mesh triangle edges are thin projected barriers, not filled solids.
    // A degenerate segment is a point collider. Ordering follows source IDs.
    std::array<std::array<float, 4>, 448> segments {};
    int count = 0;
    // Solid mode uses the mesh's filled AABB and point spheres. These retain
    // source identities; they do not reinterpret triangle edges as solid faces.
    std::array<float,3> minimum {}, maximum {};
    std::uint64_t meshIdentity = 0;
    std::array<std::array<float,3>,64> solidPoints {};
    std::array<std::uint64_t,64> pointIdentities {};
    int pointCount = 0;
};

inline bool sampleParticleGeometry(ParticleParams& p, std::int64_t tick,
                                  ParticleGeometryFrame& frame, std::string& error)
{
    if (!p.geometryBinding) return true;
    using namespace videowire::geometry;
    const auto& binding = *p.geometryBinding;
    const visualdeformation::RationalFrameTime time {std::max<std::int64_t>(0,tick), std::uint32_t(kParticleReplayTicksPerSecond), 1};
    const double seconds = double(tick) / kParticleReplayTicksPerSecond;
    std::map<std::string, std::shared_ptr<const RetainedMeshData>> imported;
    const auto evaluate = [&](const std::shared_ptr<const ValueDescriptor>& source, ValueDescriptor& output)
    {
        if (!source) return true;
        output = *source;
        for (auto& operation : output.operations)
            if (operation.retainedMesh && !operation.retainedMesh->importedAnimation.empty())
            {
                if (!p.history || !p.history->importedGeometry)
                { error = "Animated particle geometry requires the preview/export asset owner"; return false; }
                const auto& original = *operation.retainedMesh;
                auto cached = imported.find(original.importedAnimation);
                if (cached == imported.end())
                {
                    auto animated = std::make_shared<RetainedMeshData>(original);
                    if (!p.history->importedGeometry(*animated, time, p.geometryRevision, error)) return false;
                    cached = imported.emplace(original.importedAnimation, std::move(animated)).first;
                }
                if (cached->second->geometry.vertexIds != original.geometry.vertexIds
                    || cached->second->geometry.indices != original.geometry.indices
                    || cached->second->geometry.positions.size() != original.geometry.positions.size())
                { error = "Animated particle mesh changed its admitted topology"; return false; }
                operation.retainedMesh = cached->second;
            }
        RuntimeFieldEvaluation runtime; runtime.timelineSeconds = seconds;
        ValueDescriptor replayed;
        MaterializedInstancePlan intermediates;
        if (!validateOperationPlan(output, {}, &intermediates, error, &replayed, &runtime)
            || !particleGeometryIntermediatesFit(intermediates,error)
            || !validateParticleGeometry(replayed, source->carrier, error, false)) return false;
        if (source->carrier == CarrierKind::points3D)
        {
            const auto& before = std::get<PointsData>(source->data).points;
            const auto& after = std::get<PointsData>(replayed.data).points;
            if (before.size() != after.size() || !std::equal(before.begin(), before.end(), after.begin(),
                [](const auto& a, const auto& b) { return a.stableId == b.stableId; }))
            { error = "Animated particle points changed their admitted identities"; return false; }
        }
        else
        {
            const auto& before = std::get<GeometryData>(source->data);
            const auto& after = std::get<GeometryData>(replayed.data);
            if (before.vertexIds != after.vertexIds || before.indices != after.indices)
            { error = "Animated particle mesh changed its admitted topology"; return false; }
        }
        output = std::move(replayed);
        return true;
    };
    const auto project = [&](const Vec3& point)
    {
        return std::array<float, 2> {binding.centerX + binding.scale * (binding.projection == 2 ? point.y : point.x),
            binding.centerY + binding.scale * (binding.projection == 0 ? point.y : point.z)};
    };
    const auto world = [&](const Vec3& point)
    { return std::array<float,3>{binding.centerX+binding.scale*point.x,
        binding.centerY+binding.scale*point.y,binding.centerZ+binding.scale*point.z}; };
    ValueDescriptor emission, attraction, mesh, points;
    if (!evaluate(binding.emitters, emission) || !evaluate(binding.attractors, attraction)
        || !evaluate(binding.mesh, mesh) || !evaluate(binding.points, points)) return false;
    if (binding.emitters)
    {
        auto emitters = std::get<PointsData>(emission.data).points;
        auto attractors = binding.attractors ? std::get<PointsData>(attraction.data).points : emitters;
        const auto byIdentity = [](const auto& a, const auto& b) { return a.stableId < b.stableId; };
        std::sort(emitters.begin(), emitters.end(), byIdentity); std::sort(attractors.begin(), attractors.end(), byIdentity);
        p.geometryCount = static_cast<int>(emitters.size());
        for (std::size_t i = 0; i < emitters.size(); ++i)
        {
            const auto a = project(emitters[i].position), b = project(attractors[i % attractors.size()].position);
            p.geometryAnchors[i] = {a[0], a[1], b[0], b[1]};
            p.geometryIdentities[i]=emitters[i].stableId;
            if (p.simulationSpace==1) {
                const auto x=world(emitters[i].position),y=world(attractors[i%attractors.size()].position);
                p.geometryAnchors[i]={x[0],x[1],y[0],y[1]};
                p.geometryDepths[i]={x[2],y[2]};
            }
        }
    }
    frame.count = 0;
    frame.meshIdentity=0; frame.pointCount=0;
    if (binding.mesh)
    {
        const auto& geometry = std::get<GeometryData>(mesh.data);
        if (p.simulationSpace==1) {
            frame.meshIdentity=mesh.stableId;
            frame.minimum=frame.maximum=world(geometry.positions.front());
            for (const auto& point:geometry.positions) {
                const auto v=world(point);
                for (int axis=0;axis<3;++axis) {
                    frame.minimum[axis]=std::min(frame.minimum[axis],v[axis]-p.colliderThickness);
                    frame.maximum[axis]=std::max(frame.maximum[axis],v[axis]+p.colliderThickness);
                }
            }
        }
        std::set<std::pair<StableId, StableId>> seen;
        for (std::size_t i = 0; i < geometry.indices.size(); i += 3)
            for (int edge = 0; edge < 3; ++edge)
            {
                auto a = geometry.indices[i + edge], b = geometry.indices[i + (edge + 1) % 3];
                if (geometry.vertexIds[a] > geometry.vertexIds[b]) std::swap(a, b);
                if (!seen.emplace(geometry.vertexIds[a], geometry.vertexIds[b]).second) continue;
                const auto x = project(geometry.positions[a]), y = project(geometry.positions[b]);
                frame.segments[static_cast<std::size_t>(frame.count++)] = {x[0], x[1], y[0], y[1]};
            }
    }
    if (binding.points)
    {
        auto colliders = std::get<PointsData>(points.data).points;
        std::sort(colliders.begin(), colliders.end(), [](const auto& a, const auto& b) { return a.stableId < b.stableId; });
        for (const auto& collider : colliders) {
            const auto x = project(collider.position); frame.segments[static_cast<std::size_t>(frame.count++)] = {x[0], x[1], x[0], x[1]};
            frame.solidPoints[frame.pointCount]=world(collider.position);
            frame.pointIdentities[frame.pointCount++]=collider.stableId;
        }
    }
    const auto bounded=[](const auto& values)
    { return std::all_of(values.begin(),values.end(),[](float x) { return std::isfinite(x) && std::abs(x)<=32; }); };
    for (int i=0;i<p.geometryCount;++i)
        if (!bounded(p.geometryAnchors[static_cast<std::size_t>(i)]))
        { error="Animated particle geometry exceeds the projected coordinate limit of 32"; return false; }
    for (int i=0;i<frame.count;++i)
        if (!bounded(frame.segments[static_cast<std::size_t>(i)]))
        { error="Animated particle colliders exceed the projected coordinate limit of 32"; return false; }
    if (p.simulationSpace==1) {
        for (int i=0;i<p.geometryCount;++i) if (!bounded(p.geometryDepths[i]))
        { error="Solid emitter depth exceeds the coordinate limit of 32"; return false; }
        if (!bounded(frame.minimum) || !bounded(frame.maximum))
        { error="Solid mesh bounds exceed the coordinate limit of 32"; return false; }
        for (int i=0;i<frame.pointCount;++i) if (!bounded(frame.solidPoints[i]))
        { error="Solid point colliders exceed the coordinate limit of 32"; return false; }
    }
    return true;
}
}
