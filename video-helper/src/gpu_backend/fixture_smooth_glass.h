#pragma once

#include "fixture_opaque_visibility.h"
#include <map>
#include <numeric>
#include <cstring>
#include <set>

namespace arbitgpu::smoothglass
{
inline constexpr std::size_t kMaximumSceneTriangles = 8192;
inline constexpr std::size_t kMaximumVolumeTriangles = 2048;
inline constexpr std::size_t kMaximumRayVisits = 64;
inline constexpr std::size_t kMaximumInterfaceDepth = 32;
// Prerequisite for native rough transport, not a capability admission. Native
// callers must retain the separate terminal, masked and tail power receipts.
inline constexpr std::size_t kRoughInterfaceSamples = 16;
// Versioned Smith-height interface distribution shared with native consumers.
inline constexpr std::uint32_t kMultipleScatterVersion = 2;
inline constexpr std::size_t kMaximumMicrofacetEvents = 32, kMaximumMicrofacetExits = 16;
inline constexpr std::uint32_t kSingleScatterCameraVersion = 1;
inline constexpr int kMultipleScatterCameraComplete = 33;
inline constexpr std::uint64_t kMaximumCameraMicroEvents = 3*kRoughInterfaceSamples*kMaximumRayVisits*kMaximumMicrofacetEvents;
// Retain the conservative two-walk charge. Selected compute shares one walk
// between Color and receipt without increasing the admitted workload.
inline constexpr std::uint64_t kCameraMicroWorkPerEvent = 2*5;
inline constexpr std::uint64_t kMaximumCameraMicroWork = 128ull*1024ull*1024ull;
inline constexpr std::uint64_t kMaximumCameraPixelMicroWork = kCameraMicroWorkPerEvent*kMaximumCameraMicroEvents;
inline constexpr float kMaximumRoughUnresolvedPower = 0.0001f;
inline constexpr float kCameraWalkResidual = kMaximumRoughUnresolvedPower / float(kMaximumInterfaceDepth*4);
inline constexpr float kRoughEnergyConservationTolerance = 0.00002f;
inline constexpr std::uint64_t kMaximumRoughThinClosestHitQueries = 1+2*kRoughInterfaceSamples;
inline constexpr std::uint64_t kMaximumRoughClosestHitQueries = 1+3*kRoughInterfaceSamples*kMaximumRayVisits;
inline constexpr std::uint64_t kMaximumRoughColorRayQueries = 1+3*kRoughInterfaceSamples*kMaximumRayVisits*17;
static_assert(kRoughInterfaceSamples==16, "The rough visible-normal schedule requires four radical-inverse bits");
enum class RoughTransportStatus : int
{
    Complete=1, Smooth=2, InvalidInput=-1, InternalCamera=-2, IncompleteQuery=-3,
    IdentityOrMedium=-4, InterfaceDepth=-5, PendingBranches=-6, RayVisits=-7,
    UnresolvedEnergy=-8, Nonfinite=-9, NonconservingEnergy=-10
};
// Smooth closed-glass refusals are written to the raw-normal alpha channel of
// incomplete pixels. -1 (no reason) and -2 (unrepresentable) keep their
// released meanings, so the codes start at -3 and each has a distinct binary16
// pattern. Every early return in glassTransport sets one of these codes.
// The thin water path adds -15 to -19. Rough refusals map to -20 to -29 through
// roughReasonCode, so every incomplete native pixel names its reason.
enum class SmoothTransportStatus : int
{
    Unclassified=-3, QueryRefused=-4, QueryBudget=-5, FirstHitMismatch=-6,
    MediumMismatch=-7, ReceiverMiss=-8, OrientationMismatch=-9, InterfaceDepth=-10,
    NonfiniteScatter=-11, DiscardedEnergy=-12, PendingBranches=-13, RayVisits=-14,
    ThinReentry=-15, ThinNonfiniteIncoming=-16, ThinFirstHit=-17, ThinReceiver=-18,
    ThinRadiance=-19
};
// Rough status s in [-10,-1] is reported as kRoughReasonBase+s, so -1 to -10
// become -20 to -29 and never collide with the smooth codes or -1 and -2.
inline constexpr int kRoughReasonBase=-19;
constexpr int roughReasonCode(RoughTransportStatus status) { return kRoughReasonBase+int(status); }
inline const char* roughTransportDiagnostic(RoughTransportStatus status)
{
    switch(status) {
        case RoughTransportStatus::Complete: return "Bounded single-scatter rough glass transport completed";
        case RoughTransportStatus::Smooth: return "Released three-wavelength smooth glass transport completed";
        case RoughTransportStatus::InvalidInput: return "Rough glass requires finite admitted closed-volume inputs";
        case RoughTransportStatus::InternalCamera: return "Rough glass currently refuses internal cameras";
        case RoughTransportStatus::IncompleteQuery: return "Rough glass closest-hit query did not complete";
        case RoughTransportStatus::IdentityOrMedium: return "Rough glass path identity or medium orientation is inconsistent";
        case RoughTransportStatus::InterfaceDepth: return "Rough glass interface-depth budget exhausted";
        case RoughTransportStatus::PendingBranches: return "Rough glass pending-branch budget exhausted";
        case RoughTransportStatus::RayVisits: return "Rough glass ray-query work budget exhausted";
        case RoughTransportStatus::UnresolvedEnergy: return "Rough glass requires microfacet multiple scattering beyond its unresolved-power bound";
        case RoughTransportStatus::Nonfinite: return "Rough glass transport could not produce finite supported scattering and radiance";
        case RoughTransportStatus::NonconservingEnergy: return "Rough glass energy receipt is incomplete or nonconserving";
    }
    return "Rough glass completion status is unavailable";
}
struct RoughEnergyReceipt final
{
    std::array<double, 3> terminal {}, masked {}, tail {};
};
inline bool completeRoughEnergy(const RoughEnergyReceipt& receipt, std::string& error)
{
    for (std::size_t c=0;c<3;++c) {
        const double terminal=receipt.terminal[c], masked=receipt.masked[c], tail=receipt.tail[c];
        if (!std::isfinite(terminal) || !std::isfinite(masked) || !std::isfinite(tail)
            || terminal<0 || masked<0 || tail<0 || terminal>1 || masked>1 || tail>1
            || std::abs(terminal+masked+tail-1)>kRoughEnergyConservationTolerance)
        { error="Rough glass energy receipt is incomplete or nonconserving"; return false; }
        if (masked+tail>kMaximumRoughUnresolvedPower)
        { error="Rough glass requires microfacet multiple scattering beyond its unresolved-power bound"; return false; }
    }
    return true;
}
inline constexpr std::uint64_t kMaximumPixelRayQueries = 512ull * 1024ull * 1024ull;
// Per pixel: one primary validation and at most two closest-hit queries.
// Each admitted hit uses the existing receiver loop, capped at 16 shadow tests.
inline constexpr std::uint64_t kMaximumThinPixelRayQueries = 32ull * 1024ull * 1024ull;
// An explicit native implementation contract. Default callers cannot admit
// rough glass. The backend owns shader selection, float32 storage and completed
// readback from the exact same frame; none can be inferred from receipt pixels.
struct RoughNativeConsumer final
{
    NativeFixtureMaterialBackend backend = NativeFixtureMaterialBackend::Invalid;
    bool shader = false, float32Receipt = false, readback = false;
    std::uint32_t version = kSingleScatterCameraVersion;
    bool selectedComputeV2 = false;
    bool complete(NativeFixtureMaterialBackend target) const noexcept
    {
        return (backend == NativeFixtureMaterialBackend::OpenGl || backend == NativeFixtureMaterialBackend::Metal)
            && backend == target && shader && float32Receipt && readback
            && (version==kSingleScatterCameraVersion || (version==kMultipleScatterVersion && selectedComputeV2));
    }
};
static_assert(sizeof(float)==4, "Native rough receipts require float32 scalar storage");
inline constexpr std::uint64_t kRoughReceiptBytesPerPixel = 3*4*sizeof(float);
inline constexpr std::uint64_t kMaximumRoughFrameRayQueries = kMaximumRoughColorRayQueries+kMaximumRoughClosestHitQueries;
struct DistinctGratingAdmission
{
    HarmonicMIDI::grid::SceneObjectId object {};
    std::shared_ptr<const diffractionmaterial::AdmittedDiffractionMaterialIR> material;
};
inline bool validDistinctGrating(const NativeFixtureSurfaceMaterialProgram& owner,const DistinctGratingAdmission& admission)
{
    if(!admission.object.isValid()||owner.object!=admission.object||!admission.material
        ||owner.kind!=NativeFixtureMaterialKind::DiffractionReflective
        ||!validNativeFixtureDiffractionProgram(owner,owner.backend,admission.object,true,
            NativeFixtureDiffractionConsumer::SmoothGlassDiscreteOrders)
        ||owner.bindingDigest!=admission.material->structuralDigest()
        ||owner.layoutVersion!=NativeFixtureSurfaceMaterialProgram::kLayoutVersion||owner.vertexProgram||owner.frameEndpoint
        ||owner.importedBaseColorTexture||owner.diffractionPathCount!=1||owner.diffractionPatternMask
        ||owner.diffractionMaskCoverage!=1||!owner.diffractionProductDigest.empty()
        ||owner.diffractionFoilMaximumEvaluations||owner.diffractionOccupancyRectangleCount)return false;
    const auto& d=admission.material->description();
    if(d.geometry.lattice!=diffractionmaterial::GratingLattice::OneDimensional
        ||d.grooveField.mode!=diffractionmaterial::GrooveFieldMode::Constant
        ||d.microstructure.profile!=diffractionmaterial::GrooveProfile::BinaryRectangular
        ||d.coating.model!=diffractionmaterial::CoatingModel::Uncoated
        ||d.roughness.rmsHeightNanometres||d.roughness.rmsSlope
        ||d.spectrum.firstOrder!=1||d.spectrum.lastOrder!=1
        ||d.spectrum.wavelengthsNanometres!=diffractionmaterial::physicalcheckpoint::kWavelengths)return false;
    auto expected=diffractionmaterial::physicalcheckpoint::makeGpuParameters(*admission.material,{});
    auto actual=owner.diffractionPaths[0].material;actual.incident=expected.incident;
    for(unsigned band=0;band<8;++band)actual.spectral[band].y=expected.spectral[band].y;
    return std::memcmp(&expected,&actual,sizeof(expected))==0;
}
struct Prepared final
{
    bool enabled = false;
    // Only the owning prepare overload supplies secondary-consumer admission.
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> sourceScene;
    std::shared_ptr<const NativeFixtureSurfaceMaterialProgram> sourceMaterial;
    surfacematerial::TransportMode transport = surfacematerial::TransportMode::ThinSurface;
    HarmonicMIDI::grid::SceneObjectId object {};
    opaquevisibility::Prepared query;
    std::array<std::array<float, 4>, 64> receivers {}, emission {};
    float roughness = 0;
    HarmonicMIDI::grid::SceneMaterialId material {};
    RoughNativeConsumer roughConsumer {};
    std::shared_ptr<const DistinctGratingAdmission> distinctGrating;
};
inline bool hasVolume(const NativeFixtureSurfaceMaterialProgram* root)
{
    if (!root) return false;
    const auto volume = [](const auto* program) {
        return program && program->parameters.identifiers[1]
            == static_cast<std::uint32_t>(surfacematerial::TransportMode::ClosedVolume);
    };
    if (volume(root)) return true;
    for (const auto& program : root->objectPrograms) if (volume(program.get())) return true;
    return false;
}
inline bool hasTransport(const NativeFixtureSurfaceMaterialProgram* root)
{
    if (!root) return false;
    const auto traced = [](const auto* program) {
        return program && (program->parameters.identifiers[1]
                == static_cast<std::uint32_t>(surfacematerial::TransportMode::ClosedVolume)
            || program->parameters.identifiers[1]
                == static_cast<std::uint32_t>(surfacematerial::TransportMode::ThinInterface));
    };
    if (traced(root)) return true;
    for (const auto& program : root->objectPrograms) if (traced(program.get())) return true;
    return false;
}
namespace topology
{
using Point = std::array<double, 3>;
inline Point sub(Point a, Point b) { for (int c=0;c<3;++c) a[c]-=b[c]; return a; }
inline double dot(Point a, Point b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
inline Point cross(Point a, Point b) { return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
struct Face { std::array<Point, 3> points; std::array<std::size_t, 3> ids; };
// Every contact must lie on the shared vertex or edge. This also checks faces
// with common vertices; skipping adjacent faces can miss folded intersections.
inline bool intersects(const Face& a, const Face& b)
{
    for (int axis=0;axis<3;++axis) {
        double lowA=a.points[0][axis],highA=lowA,lowB=b.points[0][axis],highB=lowB;
        for (int c=1;c<3;++c) {
            lowA=std::min(lowA,a.points[c][axis]); highA=std::max(highA,a.points[c][axis]);
            lowB=std::min(lowB,b.points[c][axis]); highB=std::max(highB,b.points[c][axis]);
        }
        if (highA<lowB || highB<lowA) return false;
    }
    std::vector<Point> shared;
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) if (a.ids[i]==b.ids[j]) shared.push_back(a.points[i]);
    double scale=0;
    for (const auto& face : {a,b}) for (int e=0;e<3;++e) {
        const auto edge=sub(face.points[(e+1)%3],face.points[e]); scale=std::max(scale,std::sqrt(dot(edge,edge)));
    }
    const double epsilon=scale*1e-10;
    const auto allowed=[&](Point p) {
        for (const auto& vertex:shared) if (dot(sub(p,vertex),sub(p,vertex))<=epsilon*epsilon) return true;
        if (shared.size()!=2) return false;
        const auto edge=sub(shared[1],shared[0]); const auto offset=sub(p,shared[0]);
        const auto length=dot(edge,edge); const auto along=dot(offset,edge)/length;
        return along>=0 && along<=1 && dot(cross(offset,edge),cross(offset,edge))<=epsilon*epsilon*length;
    };
    const auto inside=[&](Point p, const Face& face, Point normal) {
        const double length=std::sqrt(dot(normal,normal));
        for (int e=0;e<3;++e) {
            const auto edge=sub(face.points[(e+1)%3],face.points[e]);
            if (dot(normal,cross(edge,sub(p,face.points[e]))) < -epsilon*length*std::sqrt(dot(edge,edge))) return false;
        }
        return true;
    };
    const auto normalA=cross(sub(a.points[1],a.points[0]),sub(a.points[2],a.points[0]));
    const auto normalB=cross(sub(b.points[1],b.points[0]),sub(b.points[2],b.points[0]));
    const double lengthA=std::sqrt(dot(normalA,normalA)),lengthB=std::sqrt(dot(normalB,normalB));
    bool coplanar=true;
    for (const auto& p:b.points) if (std::abs(dot(normalA,sub(p,a.points[0])))>epsilon*lengthA) coplanar=false;
    if (coplanar) {
        for (const auto& p:a.points) if (inside(p,b,normalB) && !allowed(p)) return true;
        for (const auto& p:b.points) if (inside(p,a,normalA) && !allowed(p)) return true;
        int axis=0; for (int c=1;c<3;++c) if (std::abs(normalA[c])>std::abs(normalA[axis])) axis=c;
        const int x=(axis+1)%3,y=(axis+2)%3;
        const auto det=[&](Point u,Point v) { return u[x]*v[y]-u[y]*v[x]; };
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
            const auto u=sub(a.points[(i+1)%3],a.points[i]),v=sub(b.points[(j+1)%3],b.points[j]);
            const auto delta=sub(b.points[j],a.points[i]); const double denominator=det(u,v);
            if (std::abs(denominator)<=epsilon*scale) continue;
            const double t=det(delta,v)/denominator, q=det(delta,u)/denominator;
            if (t>=0 && t<=1 && q>=0 && q<=1) {
                auto p=a.points[i]; for(int c=0;c<3;++c) p[c]+=t*u[c];
                if (!allowed(p)) return true;
            }
        }
        return false;
    }
    const auto crosses=[&](const Face& from,const Face& to,Point normal,double length) {
        for (int e=0;e<3;++e) {
            const auto& p=from.points[e]; const auto& q=from.points[(e+1)%3];
            const double dp=dot(normal,sub(p,to.points[0])),dq=dot(normal,sub(q,to.points[0]));
            if (std::abs(dp)<=epsilon*length && inside(p,to,normal) && !allowed(p)) return true;
            if ((dp>0 && dq<0) || (dp<0 && dq>0)) {
                auto hit=p; const double t=dp/(dp-dq); for(int c=0;c<3;++c) hit[c]+=t*(q[c]-p[c]);
                if (inside(hit,to,normal) && !allowed(hit)) return true;
            }
        }
        return false;
    };
    return crosses(a,b,normalB,lengthB) || crosses(b,a,normalA,lengthA);
}
} // namespace topology
// Exact position welding supports authored UV/normal seams without altering mesh bytes.
inline bool closedVolume(const HarmonicMIDI::grid::Visual3DScene& scene,
                         const HarmonicMIDI::grid::SceneObjectRecord& object, std::string& error)
{
    if (object.firstVertex > scene.vertexCount || object.vertexCount > scene.vertexCount - object.firstVertex
        || object.firstIndex > scene.indexCount || object.indexCount > scene.indexCount - object.firstIndex)
    { error = "Closed glass volume geometry range is invalid"; return false; }
    const auto triangles = object.indexCount / 3;
    if (object.indexCount == 0 || object.indexCount % 3 || triangles > kMaximumVolumeTriangles
        || object.vertexCount > 3*kMaximumVolumeTriangles)
    { error = "Closed glass volume triangle count is outside its bound"; return false; }
    using Position = std::array<float, 3>;
    std::map<Position, std::size_t> welded;
    std::vector<std::size_t> vertices(object.vertexCount);
    for (std::size_t v = 0; v < object.vertexCount; ++v)
    {
        const auto& p = scene.vertices[object.firstVertex + v].position;
        const Position position {p.x, p.y, p.z};
        if (!std::all_of(position.begin(), position.end(), [](float f) { return std::isfinite(f); }))
        { error = "Closed glass volume has a nonfinite vertex"; return false; }
        vertices[v] = welded.emplace(position, welded.size()).first->second;
    }
    struct Edge { std::size_t face = 0, count = 0; int orientation = 0; };
    std::map<std::pair<std::size_t, std::size_t>, Edge> edges;
    std::vector<std::size_t> parents(triangles);
    std::vector<std::map<std::size_t, std::vector<std::size_t>>> links(welded.size());
    std::set<std::array<std::size_t, 3>> faces;
    std::vector<topology::Face> geometricFaces;
    std::iota(parents.begin(), parents.end(), 0);
    const auto find = [&](std::size_t at) { while (parents[at] != at) at = parents[at]; return at; };
    const auto& anchorVertex = scene.vertices[object.firstVertex].position;
    const std::array<double,3> anchor {anchorVertex.x,anchorVertex.y,anchorVertex.z};
    double volume = 0;
    for (std::size_t t = 0; t < triangles; ++t)
    {
        std::array<std::size_t, 3> ids;
        std::array<std::array<double, 3>, 3> points;
        for (std::size_t c = 0; c < 3; ++c)
        {
            const auto v = scene.indices[object.firstIndex + t * 3 + c];
            if (v >= object.vertexCount)
            { error = "Closed glass volume index is out of range"; return false; }
            ids[c] = vertices[v];
            const auto& p = scene.vertices[object.firstVertex + v].position;
            points[c] = {p.x, p.y, p.z};
        }
        auto canonical = ids;
        std::sort(canonical.begin(), canonical.end());
        if (!faces.insert(canonical).second)
        { error = "Closed glass volume has duplicate faces"; return false; }
        for (std::size_t c = 0; c < 3; ++c) {
            auto& link = links[ids[c]];
            link[ids[(c+1)%3]].push_back(ids[(c+2)%3]);
            link[ids[(c+2)%3]].push_back(ids[(c+1)%3]);
        }
        const auto sub = [](auto a, auto b) { for (int axis = 0; axis < 3; ++axis) a[axis] -= b[axis]; return a; };
        const auto cross = [](auto a, auto b) { return std::array<double, 3> {
            a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]}; };
        const auto dot = [](auto a, auto b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; };
        const auto normal = cross(sub(points[1], points[0]), sub(points[2], points[0]));
        if (!(dot(normal, normal) > 0) || !std::isfinite(dot(normal, normal)))
        { error = "Closed glass volume has a degenerate triangle"; return false; }
        geometricFaces.push_back({points,ids});
        volume += dot(sub(points[0],anchor), cross(sub(points[1],anchor),sub(points[2],anchor))) / 6.0;
        for (std::size_t e = 0; e < 3; ++e)
        {
            const auto a = ids[e], b = ids[(e + 1) % 3];
            auto& edge = edges[{std::min(a,b), std::max(a,b)}];
            if (edge.count) parents[find(t)] = find(edge.face);
            else edge.face = t;
            ++edge.count;
            edge.orientation += a < b ? 1 : -1;
            if (edge.count > 2)
            { error = "Closed glass volume has a nonmanifold edge"; return false; }
        }
    }
    if (!(volume > 0) || !std::isfinite(volume))
    { error = "Closed glass volume winding must point outward and enclose nonzero volume"; return false; }
    for (const auto& [key, edge] : edges)
        if (edge.count != 2 || edge.orientation != 0)
        { error = "Closed glass volume has an open edge or inconsistent winding"; return false; }
    for (std::size_t a=0;a<geometricFaces.size();++a)
        for (std::size_t b=a+1;b<geometricFaces.size();++b)
            if (topology::intersects(geometricFaces[a],geometricFaces[b]))
            { error = "Closed glass volume has intersecting faces"; return false; }
    for (const auto& link : links) {
        if (link.empty()) continue;
        std::set<std::size_t> visited;
        std::vector<std::size_t> pending {link.begin()->first};
        while (!pending.empty()) {
            const auto v = pending.back(); pending.pop_back();
            if (!visited.insert(v).second) continue;
            const auto& neighbours = link.at(v);
            if (neighbours.size() != 2)
            { error = "Closed glass volume has a nonmanifold vertex"; return false; }
            pending.insert(pending.end(), neighbours.begin(), neighbours.end());
        }
        if (visited.size() != link.size())
        { error = "Closed glass volume has disconnected vertex fans"; return false; }
    }
    for (std::size_t t = 1; t < triangles; ++t)
        if (find(t) != find(0))
        { error = "Closed glass volume requires one connected shell"; return false; }
    return true;
}
inline bool prepare(const HarmonicMIDI::grid::Visual3DScene& scene,
                    const NativeFixtureSurfaceMaterialProgram* root,
                    const std::vector<videowire::geometry::InstanceAppearance>& appearances,
                    Prepared& output, std::string& error, RoughNativeConsumer consumer = {},
                    const DistinctGratingAdmission* distinctGrating=nullptr)
{
    using namespace HarmonicMIDI::grid;
    Prepared candidate;
    if(distinctGrating&&(scene.objectCount>Visual3DScene::kMaxObjects
        ||scene.vertexCount>Visual3DScene::kMaxVertices||scene.indexCount>Visual3DScene::kMaxIndices
        ||!root||root->kind!=NativeFixtureMaterialKind::DiffractionReflective
        ||root->object!=distinctGrating->object
        ||!validNativeFixtureMaterialPrograms(*root,scene,root->backend,true,
            NativeFixtureDiffractionConsumer::SmoothGlassDiscreteOrders)))
    {error="Smooth glass grating requires the existing Diffraction-root and flat Surface-child authority";return false;}
    if (!hasTransport(root)) { output = std::move(candidate); return true; }
    if (scene.objectCount > 64 || scene.vertexCount > Visual3DScene::kMaxVertices
        || scene.indexCount > Visual3DScene::kMaxIndices)
    { error = "Closed glass scene exceeds snapshot counts"; return false; }
    if (!appearances.empty() || opaquevisibility::vertexDeformation(root))
    { error = "Closed glass volume refuses instances, vertex deformation or scene triangle budget overflow"; return false; }
    std::size_t count = 0, expandedTriangles = 0, gratingCount=0;
    for (std::size_t o = 0; o < scene.objectCount; ++o)
    {
        const auto& object = scene.objects[o];
        const auto* material = visual3d_detail::findById(scene.materials, scene.materialCount, object.material);
        const auto* surface = nativeSurfaceForObject(root, object.id);
        const bool grating=distinctGrating&&object.id==distinctGrating->object;
        if (object.vertexCount == 0 && object.indexCount == 0) continue;
        if (object.indexCount == 0 || object.indexCount % 3
            || object.firstVertex > scene.vertexCount || object.vertexCount > scene.vertexCount - object.firstVertex
            || object.firstIndex > scene.indexCount || object.indexCount > scene.indexCount - object.firstIndex)
        { error = "Closed glass scene requires valid indexed geometry"; return false; }
        // A reused or overlapping range still contributes geometry for this
        // object. The global index array does not bound composed query work.
        expandedTriangles += object.indexCount / 3;
        if (expandedTriangles > kMaximumSceneTriangles)
        { error = "Closed glass expanded scene exceeds its 8192 triangle budget"; return false; }
        if(grating&&(!surface||!validDistinctGrating(*surface,*distinctGrating)))
        {error="Closed glass distinct grating requires exact compiler-owned planar binary IR";return false;}
        if (!material || (surface && ((!grating&&surface->kind != NativeFixtureMaterialKind::SurfacePbr)
            || (!grating&&!validNativeFixtureSurfaceParameters(surface->parameters))
            || surface->baseColorSource != NativeFixtureSurfaceMaterialProgram::BaseColorSource::ConstantLinear
            || surface->vertexProgram)))
        { error = "Closed glass transport requires constant untextured static surfaces"; return false; }
        if (material->baseColorTexture.isValid() || material->metallicRoughnessTexture.isValid()
            || material->normalTexture.isValid() || material->emissiveTexture.isValid()
            || material->occlusionTexture.isValid() || material->alphaMode != SceneAlphaMode::Opaque)
        { error = "Closed glass transport refuses textured or nonopaque coverage"; return false; }
        for (std::size_t v = 0; v < object.vertexCount; ++v)
        {
            const auto& color = scene.vertices[object.firstVertex + v].color;
            if (color[0] != 1 || color[1] != 1 || color[2] != 1 || color[3] != 1)
            { error = "Closed glass transport requires constant unit vertex color"; return false; }
        }
        if(grating) {
            if(++gratingCount!=1||object.indexCount/3>kMaximumNativeDiffractionTriangles)
            {error="Closed glass distinct grating triangle budget is invalid";return false;}
            topology::Point firstNormal{},firstPosition{};
            for(std::size_t triangle=0;triangle<object.indexCount/3;++triangle) {
                std::array<topology::Point,3> points;std::array<std::array<float,2>,3> uv;
                for(unsigned v=0;v<3;++v) {
                    const auto index=scene.indices[object.firstIndex+triangle*3+v];
                    if(index>=object.vertexCount){error="Closed glass grating triangle index is invalid";return false;}
                    const auto& vertex=scene.vertices[object.firstVertex+index];
                    points[v]={vertex.position.x,vertex.position.y,vertex.position.z};uv[v]={vertex.uv.x,vertex.uv.y};
                }
                auto normal=topology::cross(topology::sub(points[1],points[0]),topology::sub(points[2],points[0]));
                const double length=std::sqrt(topology::dot(normal,normal));
                const double det=double(uv[1][0]-uv[0][0])*(uv[2][1]-uv[0][1])-double(uv[2][0]-uv[0][0])*(uv[1][1]-uv[0][1]);
                if(!std::isfinite(length)||length<=0||!std::isfinite(det)||det<=1e-12)
                {error="Closed glass grating requires finite right-handed triangle UVs";return false;}
                for(auto& value:normal)value/=length;
                if(triangle==0){firstNormal=normal;firstPosition=points[0];}
                else if(topology::dot(normal,firstNormal)<1-1e-5
                    ||std::abs(topology::dot(topology::sub(points[0],firstPosition),firstNormal))>1e-5)
                {error="Closed glass distinct grating must be static planar geometry";return false;}
            }
            continue;
        }
        const auto volume = surface && surface->parameters.identifiers[1]
            == static_cast<std::uint32_t>(surfacematerial::TransportMode::ClosedVolume);
        const auto& p = surface ? surface->parameters : surfacematerial::SurfacePbrParameterBlock{};
        const auto thin = surface && surface->parameters.identifiers[1]
            == static_cast<std::uint32_t>(surfacematerial::TransportMode::ThinInterface);
        if (volume || thin)
        {
            if(surface->roughGlassScatteringVersion!=1
                && (surface->roughGlassScatteringVersion!=2 || !volume
                    || (p.emissionRoughness[3]!=0 && (consumer.version!=2 || !consumer.selectedComputeV2))))
            {error="Rough glass scattering version requires a matching closed-volume native consumer";return false;}
            if (!std::isfinite(p.emissionRoughness[3]) || p.emissionRoughness[3]<0 || p.emissionRoughness[3]>1)
            { error = "Rough dielectric requires roughness within [0, 1]"; return false; }
            auto selectedConsumer=consumer;selectedConsumer.version=surface->roughGlassScatteringVersion;
            if (p.emissionRoughness[3] != 0 && !selectedConsumer.complete(surface->backend))
            { error = "Rough dielectric requires a matching native shader, float32 terminal, masked and tail energy receipts and completed readback consumer"; return false; }
            if (++count != 1 || p.baseColorMetallic[0] != 1 || p.baseColorMetallic[1] != 1 || p.baseColorMetallic[2] != 1
                || p.normalOpacity[0] != 0 || p.normalOpacity[1] != 0 || p.normalOpacity[2] != 1
                || p.normalOpacity[3] != 1 || p.transmissionIorClearcoat[0] != 1
                || p.baseColorMetallic[3] != 0 || p.transmissionIorClearcoat[2] != 0
                || p.emissionRoughness[0] != 0 || p.emissionRoughness[1] != 0 || p.emissionRoughness[2] != 0)
            { error = "Closed glass transport requires one dielectric with unit opacity and transmission"; return false; }
            if (thin && p.transmissionIorClearcoat[3] != 0)
            { error = "Water interface refuses dispersion"; return false; }
            for (const auto wavelength : surfacematerial::kGlassIorAdmissionWavelengths)
            {
                const auto index = surfacematerial::wavelengthIor(p.transmissionIorClearcoat[1], p.transmissionIorClearcoat[3], wavelength);
                if (!std::isfinite(index) || index < 1 || index > 3)
                { error = "Closed glass transport wavelength IOR is outside [1, 3]"; return false; }
            }
            if (volume && !closedVolume(scene, object, error)) return false;
            if (thin) {
                for (std::size_t v=0;v<object.vertexCount;++v) {
                    const auto& normal=scene.vertices[object.firstVertex+v].normal;
                    const double length=double(normal.x)*normal.x+double(normal.y)*normal.y+double(normal.z)*normal.z;
                    if (!(length>0) || !std::isfinite(length))
                    { error = "Smooth water interface requires finite evaluated normals"; return false; }
                }
            }
            candidate.transport = volume ? surfacematerial::TransportMode::ClosedVolume
                                         : surfacematerial::TransportMode::ThinInterface;
            candidate.object = object.id;
            candidate.material = object.material;
            candidate.roughness = p.emissionRoughness[3];
            if (candidate.roughness != 0) {
                candidate.roughConsumer = selectedConsumer;
            }
        }
        else
        {
            if ((surface && (p.normalOpacity[0] != 0 || p.normalOpacity[1] != 0 || p.normalOpacity[2] != 1
                || p.normalOpacity[3] != 1 || p.transmissionIorClearcoat[0] != 0 || p.baseColorMetallic[3] != 0
                || p.emissionRoughness[3] != 1 || p.transmissionIorClearcoat[2] != 0 || p.transmissionIorClearcoat[1] != 1.5f))
                || (!surface && (material->opacity != 1 || material->metallic != 0 || material->roughness != 1)))
            { error = "Closed glass receivers require unit-opacity constant diffuse material"; return false; }
            const std::array<float,3> reflectance = surface
                ? std::array<float,3>{p.baseColorMetallic[0],p.baseColorMetallic[1],p.baseColorMetallic[2]}
                : std::array<float,3>{material->baseColor.x,material->baseColor.y,material->baseColor.z};
            if (!std::all_of(reflectance.begin(),reflectance.end(),[](float value) {
                    return std::isfinite(value) && value>=0 && value<=1;
                }))
            { error = "Closed glass receiver diffuse reflectance must be within [0, 1]"; return false; }
            for (std::size_t t=0;t<object.indexCount/3;++t) {
                std::array<topology::Point,3> points;
                for (std::size_t c=0;c<3;++c) {
                    const auto index=scene.indices[object.firstIndex+t*3+c];
                    if (index>=object.vertexCount)
                    { error = "Closed glass receiver index is out of range"; return false; }
                    const auto& p=scene.vertices[object.firstVertex+index].position;
                    points[c]={p.x,p.y,p.z};
                }
                const auto n=topology::cross(topology::sub(points[1],points[0]),topology::sub(points[2],points[0]));
                const double length=std::sqrt(topology::dot(n,n));
                if (!(length>0) || !std::isfinite(length))
                { error = "Closed glass receiver has a degenerate face"; return false; }
                for (std::size_t c=0;c<3;++c) {
                    const auto& normal=scene.vertices[object.firstVertex+scene.indices[object.firstIndex+t*3+c]].normal;
                    const topology::Point value{normal.x,normal.y,normal.z};
                    for (std::size_t axis=0;axis<3;++axis)
                        if (!std::isfinite(value[axis]) || std::abs(value[axis]-n[axis]/length)>1e-5)
                        { error = "Closed glass receivers require flat geometric vertex normals"; return false; }
                }
            }
            candidate.receivers[o] = surface
                ? std::array<float,4>{p.baseColorMetallic[0],p.baseColorMetallic[1],p.baseColorMetallic[2],float(object.id.value & 65535u)}
                : std::array<float,4>{material->baseColor.x,material->baseColor.y,material->baseColor.z,float(object.id.value & 65535u)};
            candidate.emission[o] = surface
                ? std::array<float,4>{p.emissionRoughness[0],p.emissionRoughness[1],p.emissionRoughness[2],float(object.id.value >> 16u)}
                : std::array<float,4>{material->emissive.x,material->emissive.y,material->emissive.z,float(object.id.value >> 16u)};
        }
    }
    if (count != 1) { error = "Closed glass transport requires exactly one volume"; return false; }
    if(distinctGrating) {
        const auto* volume=nativeSurfaceForObject(root,candidate.object);
        if(gratingCount!=1||candidate.object==distinctGrating->object||candidate.transport!=surfacematerial::TransportMode::ClosedVolume
            ||candidate.roughness!=0||!volume||volume->parameters.transmissionIorClearcoat[3]<=0)
        {error="Distinct grating requires one separate smooth dispersive closed glass owner";return false;}
        candidate.distinctGrating=std::make_shared<const DistinctGratingAdmission>(*distinctGrating);
    }
    if (!opaquevisibility::prepareScene(scene, root, {}, candidate.query, error, opaquevisibility::AdmissionMode::allSurfaces,false,false,
        distinctGrating?NativeFixtureDiffractionConsumer::SmoothGlassDiscreteOrders:NativeFixtureDiffractionConsumer::FiniteLobe)) return false;
    candidate.enabled = true;
    output = std::move(candidate);
    return true;
}
// Retain the immutable inputs already owned by native resources. This invokes
// the same admission once and leaves the released nonowning overload intact.
inline bool prepare(const std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene>& scene,
                    const std::shared_ptr<const NativeFixtureSurfaceMaterialProgram>& root,
                    const std::vector<videowire::geometry::InstanceAppearance>& appearances,
                    Prepared& output, std::string& error, RoughNativeConsumer consumer = {},
                    const DistinctGratingAdmission* distinctGrating=nullptr)
{
    if (!scene) { error = "Smooth glass owning admission requires a scene snapshot"; return false; }
    Prepared candidate;
    if (!prepare(*scene, root.get(), appearances, candidate, error, consumer,distinctGrating)) return false;
    if (candidate.enabled) { candidate.sourceScene = scene; candidate.sourceMaterial = root; }
    output = std::move(candidate);
    return true;
}
// The optional byte output must be reserved in the backend's existing frame
// resource ledger before allocation. It covers three private RGBA32F images.
inline bool multipleScatterCamera(const Prepared& prepared) noexcept
{
    return prepared.roughness>0 && prepared.transport==surfacematerial::TransportMode::ClosedVolume
        && prepared.roughConsumer.version==kMultipleScatterVersion;
}
inline bool admitFrame(const Prepared& prepared, std::uint32_t width, std::uint32_t height,
                       bool instanced, std::string& error, std::uint64_t* roughReceiptBytes = nullptr)
{
    if (roughReceiptBytes) *roughReceiptBytes = 0;
    if (!prepared.enabled) return true;
    const bool rough = prepared.roughness != 0;
    if (rough && (!std::isfinite(prepared.roughness) || prepared.roughness<0 || prepared.roughness>1
        || (prepared.transport != surfacematerial::TransportMode::ClosedVolume
            && prepared.transport != surfacematerial::TransportMode::ThinInterface)
        || !prepared.roughConsumer.complete(prepared.roughConsumer.backend)))
    { error = "Rough dielectric frame requires an admitted native transport consumer"; return false; }
    const auto maximumPixels = multipleScatterCamera(prepared)
        ? std::min(kMaximumPixelRayQueries/kMaximumRoughFrameRayQueries,kMaximumCameraMicroWork/kMaximumCameraPixelMicroWork)
        : rough ? kMaximumPixelRayQueries/kMaximumRoughFrameRayQueries
        : prepared.transport == surfacematerial::TransportMode::ThinInterface
        ? kMaximumThinPixelRayQueries/3 : kMaximumPixelRayQueries/(3*kMaximumRayVisits+1);
    const std::uint64_t pixels = std::uint64_t(width)*height;
    if (instanced || width == 0 || height == 0 || pixels > maximumPixels)
    { error = "Closed glass transport refuses dynamic instances or pixel ray-query budget overflow"; return false; }
    if (roughReceiptBytes && rough) *roughReceiptBytes = pixels*kRoughReceiptBytesPerPixel;
    return true;
}
// Native readers supply owned tight rows, top first, in native scalar byte
// order. Backend frame/scene/program/generation/GPU-completion identity checks
// must precede this scan. Matching pixel values cannot establish ownership.
inline bool completeRoughFrame(const Prepared& prepared, std::uint32_t width, std::uint32_t height,
                               const NativeRawPassPixels& terminal, const NativeRawPassPixels& masked,
                               const NativeRawPassPixels& tail, const NativeRawPassPixels& objectIds,
                               const NativeRawPassPixels& materialIds, NativeFixtureSceneStats& stats,
                               std::string& error)
{
    // Exact zero keeps the released smooth completion path without inventing
    // rough-energy certification or requiring rough scratch allocations.
    if (!prepared.enabled || prepared.roughness == 0) return true;
    std::uint64_t scratchBytes = 0;
    if (!admitFrame(prepared,width,height,false,error,&scratchBytes)) return false;
    const std::uint64_t pixels = std::uint64_t(width)*height;
    const auto layout = [&](const NativeRawPassPixels& value, NativeTexturePixelFormat format, std::uint64_t stride) {
        return value.format==format && value.width==width && value.height==height && value.bytes.size()==pixels*stride;
    };
    if (!layout(terminal,NativeTexturePixelFormat::Rgba32Float,16)
        || !layout(masked,NativeTexturePixelFormat::Rgba32Float,16)
        || !layout(tail,NativeTexturePixelFormat::Rgba32Float,16)
        || !layout(objectIds,NativeTexturePixelFormat::R32Uint,4)
        || !layout(materialIds,NativeTexturePixelFormat::R32Uint,4))
    { error = "Rough glass requires complete matching float32 energy and raw uint32 identity attachments"; return false; }
    const auto maximumQueries = prepared.transport == surfacematerial::TransportMode::ThinInterface
        ? kMaximumRoughThinClosestHitQueries : kMaximumRoughClosestHitQueries;
    std::uint64_t covered = 0;
    double maximumResidual = 0, maximumUnresolved = 0;
    for (std::size_t pixel=0;pixel<pixels;++pixel) {
        std::uint32_t object=0,material=0;
        std::memcpy(&object,objectIds.bytes.data()+pixel*4,4);
        std::memcpy(&material,materialIds.bytes.data()+pixel*4,4);
        std::array<float,4> t{},m{},r{};
        std::memcpy(t.data(),terminal.bytes.data()+pixel*16,16);
        std::memcpy(m.data(),masked.bytes.data()+pixel*16,16);
        std::memcpy(r.data(),tail.bytes.data()+pixel*16,16);
        if (object != prepared.object.value) {
            for (const auto& texel : {t,m,r}) for (const auto value : texel)
                if (value != 0) { error = "Rough glass energy receipt has stale or nonfinite nonglass coverage"; return false; }
            continue;
        }
        if (material != prepared.material.value)
        { error = "Rough glass energy receipt material identity does not match its prepared volume"; return false; }
        const bool multiple=multipleScatterCamera(prepared);
        if (!std::isfinite(t[3]) || (!multiple && t[3]!=r[3]))
        { error = "Rough glass energy receipt completion statuses are unavailable or inconsistent"; return false; }
        if (t[3] != (multiple?kMultipleScatterCameraComplete:static_cast<int>(RoughTransportStatus::Complete))) {
            // Compare known values before conversion, avoiding an invalid cast
            // for malicious/nonfinite float32 status data.
            for (int code=-1;code>=-10;--code) if (t[3]==code) {
                error = roughTransportDiagnostic(static_cast<RoughTransportStatus>(code)); return false;
            }
            error = "Rough glass energy receipt does not certify completed rough transport"; return false;
        }
        if (!std::isfinite(m[3]) || m[3]<1 || m[3]>maximumQueries || std::floor(m[3])!=m[3])
        { error = "Rough glass energy receipt query count is outside its integer work bound"; return false; }
        // Selected compute shares one primary across the three wavelength/sample
        // schedules. These first interfaces consume saved geometry, not new queries.
        const auto sharedPrimaryInterfaces=prepared.roughConsumer.selectedComputeV2?3*kRoughInterfaceSamples:0;
        if (multiple && (!std::isfinite(r[3]) || r[3]<0 || r[3]>kMaximumCameraMicroEvents
            || std::floor(r[3])!=r[3] || r[3]>(m[3]-1+sharedPrimaryInterfaces)*kMaximumMicrofacetEvents))
        { error="Rough glass v2 micro-event receipt is outside its integer scene-query work bound";return false; }
        RoughEnergyReceipt receipt;
        if (multiple && (m[0]!=0 || m[1]!=0 || m[2]!=0))
        { error="Rough glass v2 cannot retain single-scatter masked power";return false; }
        for (std::size_t c=0;c<3;++c) { receipt.terminal[c]=t[c]; receipt.masked[c]=m[c]; receipt.tail[c]=r[c]; }
        if (!completeRoughEnergy(receipt,error)) return false;
        for (std::size_t c=0;c<3;++c) {
            maximumResidual=std::max(maximumResidual,std::abs(receipt.terminal[c]+receipt.masked[c]+receipt.tail[c]-1));
            maximumUnresolved=std::max(maximumUnresolved,receipt.masked[c]+receipt.tail[c]);
        }
        ++covered;
    }
    // Commit stats only after the entire frame passes. Failure leaves the
    // caller's stats unchanged and the native backend must publish no frame.
    stats.roughGlassReceiptComplete = true;
    stats.roughGlassCoveredPixels = covered;
    stats.roughGlassReceiptBytes = scratchBytes;
    stats.roughGlassMaximumEnergyResidual = maximumResidual;
    stats.roughGlassMaximumUnresolvedPower = maximumUnresolved;
    return true;
}
// binary16 bit pattern of a negative integer in [-2048,-1].
constexpr std::uint16_t negativeHalfBits(int value)
{
    const unsigned magnitude=unsigned(-value);
    unsigned exponent=0;
    while((2u<<exponent)<=magnitude) ++exponent;
    return std::uint16_t(0x8000u|((exponent+15u)<<10u)|(((magnitude<<10u)>>exponent)&0x3ffu));
}
struct IncompleteNormalReason { std::uint16_t bits; const char* text; };
// Reason codes written by the glass shaders into the raw-normal alpha channel.
// The last entry catches any other negative or nonfinite encoding.
inline constexpr IncompleteNormalReason kIncompleteNormalReasons[]{
    {negativeHalfBits(-1), "no reason recorded"},
    {negativeHalfBits(int(SmoothTransportStatus::Unclassified)), "unclassified refusal"},
    {negativeHalfBits(int(SmoothTransportStatus::QueryRefused)), "query refused: invalid scene query"},
    {negativeHalfBits(int(SmoothTransportStatus::QueryBudget)), "query refused: work budget"},
    {negativeHalfBits(int(SmoothTransportStatus::FirstHitMismatch)), "first hit is not the glass volume"},
    {negativeHalfBits(int(SmoothTransportStatus::MediumMismatch)), "inside-glass ray reached no glass surface"},
    {negativeHalfBits(int(SmoothTransportStatus::ReceiverMiss)), "receiver lookup missed"},
    {negativeHalfBits(int(SmoothTransportStatus::OrientationMismatch)), "hit orientation disagrees with glass medium"},
    {negativeHalfBits(int(SmoothTransportStatus::InterfaceDepth)), "interface depth budget exhausted"},
    {negativeHalfBits(int(SmoothTransportStatus::NonfiniteScatter)), "nonfinite Fresnel or transmitted direction"},
    {negativeHalfBits(int(SmoothTransportStatus::DiscardedEnergy)), "discarded energy above budget"},
    {negativeHalfBits(int(SmoothTransportStatus::PendingBranches)), "pending branch budget exhausted"},
    {negativeHalfBits(int(SmoothTransportStatus::RayVisits)), "bounce budget exhausted"},
    {negativeHalfBits(int(SmoothTransportStatus::ThinReentry)), "water receiver ray re-entered the water surface"},
    {negativeHalfBits(int(SmoothTransportStatus::ThinNonfiniteIncoming)), "nonfinite incoming ray or normal"},
    {negativeHalfBits(int(SmoothTransportStatus::ThinFirstHit)), "first hit is not the water surface"},
    {negativeHalfBits(int(SmoothTransportStatus::ThinReceiver)), "water receiver shading incomplete"},
    {negativeHalfBits(int(SmoothTransportStatus::ThinRadiance)), "nonfinite water radiance"},
    {negativeHalfBits(roughReasonCode(RoughTransportStatus::InvalidInput)), "rough glass input not admitted"},
    {negativeHalfBits(roughReasonCode(RoughTransportStatus::InternalCamera)), "rough glass refuses internal camera"},
    {negativeHalfBits(roughReasonCode(RoughTransportStatus::IncompleteQuery)), "rough glass query incomplete"},
    {negativeHalfBits(roughReasonCode(RoughTransportStatus::IdentityOrMedium)), "rough glass identity or medium inconsistent"},
    {negativeHalfBits(roughReasonCode(RoughTransportStatus::InterfaceDepth)), "rough glass interface depth budget exhausted"},
    {negativeHalfBits(roughReasonCode(RoughTransportStatus::PendingBranches)), "rough glass pending branch budget exhausted"},
    {negativeHalfBits(roughReasonCode(RoughTransportStatus::RayVisits)), "rough glass ray query budget exhausted"},
    {negativeHalfBits(roughReasonCode(RoughTransportStatus::UnresolvedEnergy)), "rough glass unresolved microfacet energy"},
    {negativeHalfBits(roughReasonCode(RoughTransportStatus::Nonfinite)), "rough glass nonfinite scattering or radiance"},
    {negativeHalfBits(roughReasonCode(RoughTransportStatus::NonconservingEnergy)), "rough glass energy receipt nonconserving"},
    {std::uint16_t(0), "unrecognized normal value"}
};
// Slack for thinEntryOwned, relative to the camera-to-entry distance. One
// 24-bit depth step at distance z with a 0.1 near plane is about z*z*6e-7, so
// this covers a depth tie out to z=160. With a 52 degree vertical field of view
// over 720 rows it stays under a tenth of one pixel's footprint, so a visible
// occluder in front of the water still refuses. The shader writes it as 1.0e-4.
inline constexpr float kThinEntryRelativeTolerance = 1.0e-4f;
// C++ twin of the shader's thinEntryOwned, for deterministic CPU tests of the
// raster/query ownership rule. first comes from opaquevisibility::closestHitCpu
// along the normalized camera-to-entry ray.
inline bool thinEntryOwned(const opaquevisibility::ClosestHit& first, std::uint32_t water,
                           const opaquevisibility::Point& camera, const opaquevisibility::Point& entry)
{
    if (first.status == 1 && first.objectId == water) return true;
    if (first.status == 0) return true;
    if (first.status != 1) return false;
    const float dx = entry[0]-camera[0], dy = entry[1]-camera[1], dz = entry[2]-camera[2];
    const float distance = std::sqrt(dx*dx+dy*dy+dz*dz);
    // Same origin clearance as the shader's glassBias.
    const float bias = std::max(1.0e-5f, 2.0e-6f*std::max(std::max(std::abs(entry[0]), std::abs(entry[1])), std::abs(entry[2])));
    const float tolerance = std::max(4.0f*bias, kThinEntryRelativeTolerance*distance);
    return std::isfinite(first.distance) && first.distance >= distance-tolerance;
}
// The owned normal attachment marks incomplete ray paths with a reason code and
// unrepresentable radiance with -2. Refuse the entire preview/export frame and
// name the first failing reason (raster order) with a pixel count per reason,
// then the readback texel where the first failure sits. Native readbacks are
// row-flipped to top-first, so y counts down from the top row.
inline bool completeFrame(const NativeRawPassPixels& pixels, std::string& error)
{
    if (pixels.format != NativeTexturePixelFormat::Rgba16Float
        || pixels.bytes.size() != std::uint64_t(pixels.width)*pixels.height*8)
    { error = "Closed glass completion attachment is unavailable"; return false; }
    constexpr std::size_t kReasons=sizeof(kIncompleteNormalReasons)/sizeof(kIncompleteNormalReasons[0]);
    std::size_t counts[kReasons]{};
    std::size_t firstReason=kReasons;
    std::size_t firstTexel=0;
    for (std::size_t offset = 6; offset < pixels.bytes.size(); offset += 8) {
        std::uint16_t alpha = 0;
        std::memcpy(&alpha, pixels.bytes.data()+offset, sizeof(alpha));
        if (alpha == 0xc000u)
        { error = "Closed glass scene-linear Color exceeds finite RGBA16F representation"; return false; }
        if (!(alpha & 0x8000u) && (alpha & 0x7c00u) != 0x7c00u) continue;
        std::size_t reason=0;
        while (reason+1<kReasons && kIncompleteNormalReasons[reason].bits!=alpha) ++reason;
        if (firstReason==kReasons) { firstReason=reason; firstTexel=offset/8; }
        ++counts[reason];
    }
    std::size_t failing=0;
    for (std::size_t i=0;i<kReasons;++i) failing+=counts[i];
    if (failing==0) return true;
    const auto describe=[&](std::size_t i) {
        return std::string(kIncompleteNormalReasons[i].text)+"; "+std::to_string(counts[i])
            +(counts[i]==1?" pixel":" pixels");
    };
    std::string text="Closed glass ray transport did not produce a complete finite result (";
    text+=describe(firstReason);
    if (failing>counts[firstReason]) {
        text+="; also ";
        bool listed=false;
        for (std::size_t i=0;i<kReasons;++i) {
            if (i==firstReason||counts[i]==0) continue;
            if (listed) text+=", ";
            text+=describe(i);
            listed=true;
        }
    }
    text+=")";
    // width is nonzero here: a failing texel exists and the byte count matched.
    text+=" first at texel x="+std::to_string(firstTexel%pixels.width)+" y="+std::to_string(firstTexel/pixels.width)
        +" of the "+std::to_string(pixels.width)+"x"+std::to_string(pixels.height)+" readback";
    error=text;
    return false;
}
// Inspect the actual owned linear Color target too. Other visible receivers
// and scene lighting can overflow that target independently of a glass ray.
inline bool finiteColorFrame(const NativeRawPassPixels& pixels, std::string& error)
{
    if (pixels.format != NativeTexturePixelFormat::Rgba16Float
        || pixels.bytes.size() != std::uint64_t(pixels.width)*pixels.height*8)
    { error = "Closed glass scene-linear Color attachment is unavailable"; return false; }
    for (std::size_t offset=0;offset<pixels.bytes.size();offset+=2) {
        std::uint16_t value=0;
        std::memcpy(&value,pixels.bytes.data()+offset,sizeof(value));
        if ((value & 0x7c00u) == 0x7c00u)
        { error = "Closed glass scene-linear Color exceeds finite RGBA16F representation"; return false; }
    }
    return true;
}
template<class String,class From,class To> inline void replace(String& text,const From& old,const To& value)
{
    surfacesource::replace(text,old,value);
}
template<class String> inline String queryShaderRecipe(bool metal,bool precise43=false)
{
    String text;
    if(metal) text=opaquevisibility::metalShaderRecipe<String>();
    else {
        if constexpr(std::is_same_v<String,std::string>) text=opaquevisibility::glslShader(precise43);
        else throw std::length_error("Surface source admission is Metal only");
    }
    if(text.empty())return {};
    smoothglass::replace(text, "opaque", "glass");
    smoothglass::replace(text, "Opaque", "Glass");
    smoothglass::replace(text, "SceneClosestHit", "GlassClosestHit");
    smoothglass::replace(text, "scene", "glassScene");
    return "\n#undef SV3\n#undef SV4\n#undef SC_TEXELS\n#undef SC_OUT\n#undef ST_ARGS\n#undef ST_PASS\n#undef ST_READ\n" + text;
}
inline std::string queryShader(bool metal,bool precise43=false) { return queryShaderRecipe<std::string>(metal,precise43); }
// The three representative wavelengths are physical samples, not a full
// spectrum or a hue lookup. Reflection and refraction branch at every interface.
// Rough helpers require the explicit native shader/float32/readback contract
// and whole-frame energy receipt validation. Smith masking is unresolved
// multiple scattering, never absorption.
template<class String> inline String multipleScatterShaderRecipe();
inline std::string multipleScatterShader();
template<class String> inline String transportShaderRecipe(bool metal, bool imageSecondary = false, bool selectedPrimary = false)
{
    String source = String("\n#define GLASS_INTERFACE_DEPTH ") + surfacesource::number<String>(kMaximumInterfaceDepth)
        + "\n#define GLASS_ROUGH_SAMPLES " + surfacesource::number<String>(kRoughInterfaceSamples)
        + "\n#define GLASS_ROUGH_VISITS " + surfacesource::number<String>(kMaximumRayVisits)
        + "\n#define GLASS_ROUGH_QUERY_LIMIT " + surfacesource::number<String>(kMaximumRoughClosestHitQueries)
        + "\n#define GLASS_ROUGH_UNRESOLVED " + surfacesource::number<String>(kMaximumRoughUnresolvedPower)
        + "\n#define GLASS_ROUGH_CONSERVATION " + surfacesource::number<String>(kRoughEnergyConservationTolerance)
        + "\n#define GLASS_ROUGH_REASON_BASE " + surfacesource::number<String>(kRoughReasonBase) + "\n";
    for (const auto& [name,status] : std::initializer_list<std::pair<const char*,RoughTransportStatus>> {
        {"COMPLETE",RoughTransportStatus::Complete}, {"SMOOTH",RoughTransportStatus::Smooth},
        {"INPUT",RoughTransportStatus::InvalidInput}, {"INTERNAL",RoughTransportStatus::InternalCamera},
        {"QUERY",RoughTransportStatus::IncompleteQuery}, {"IDENTITY",RoughTransportStatus::IdentityOrMedium},
        {"DEPTH",RoughTransportStatus::InterfaceDepth}, {"PENDING",RoughTransportStatus::PendingBranches},
        {"VISITS",RoughTransportStatus::RayVisits}, {"ENERGY",RoughTransportStatus::UnresolvedEnergy},
        {"NONFINITE",RoughTransportStatus::Nonfinite}, {"CLOSURE",RoughTransportStatus::NonconservingEnergy}})
        source += "#define GLASS_ROUGH_STATUS_" + surfacesource::literal<String>(name) + " " + surfacesource::number<String>(static_cast<int>(status)) + "\n";
    for (const auto& [name,status] : std::initializer_list<std::pair<const char*,SmoothTransportStatus>> {
        {"UNCLASSIFIED",SmoothTransportStatus::Unclassified}, {"QUERY_REFUSED",SmoothTransportStatus::QueryRefused},
        {"QUERY_BUDGET",SmoothTransportStatus::QueryBudget}, {"FIRST_HIT",SmoothTransportStatus::FirstHitMismatch},
        {"MEDIUM",SmoothTransportStatus::MediumMismatch}, {"RECEIVER",SmoothTransportStatus::ReceiverMiss},
        {"ORIENTATION",SmoothTransportStatus::OrientationMismatch}, {"DEPTH",SmoothTransportStatus::InterfaceDepth},
        {"NONFINITE_SCATTER",SmoothTransportStatus::NonfiniteScatter}, {"DISCARDED",SmoothTransportStatus::DiscardedEnergy},
        {"PENDING",SmoothTransportStatus::PendingBranches}, {"VISITS",SmoothTransportStatus::RayVisits},
        {"THIN_REENTRY",SmoothTransportStatus::ThinReentry}, {"THIN_NONFINITE_INCOMING",SmoothTransportStatus::ThinNonfiniteIncoming},
        {"THIN_FIRST_HIT",SmoothTransportStatus::ThinFirstHit}, {"THIN_RECEIVER",SmoothTransportStatus::ThinReceiver},
        {"THIN_RADIANCE",SmoothTransportStatus::ThinRadiance}})
        source += "#define GLASS_SMOOTH_STATUS_" + surfacesource::literal<String>(name) + " " + surfacesource::number<String>(static_cast<int>(status)) + "\n";
    source += R"shader(
vec3 baseSurfaceDirect(vec3 base,float metallic,float roughness,float dielectric,float transmission,
                       vec3 normal,vec3 toLight,vec3 toView);
float glassBias(vec3 position) {
    return max(1.0e-5,2.0e-6*max(max(abs(position.x),abs(position.y)),abs(position.z)));
}
float glassFresnel(vec3 incoming,vec3 normal,float fromIor,float toIor,
                   out vec3 transmitted,out bool tir) {
    float cosine=clamp(-dot(incoming,normal),0.0,1.0);
    float ratio=fromIor/toIor;
    float sineSquared=ratio*ratio*max(0.0,1.0-cosine*cosine);
    tir=sineSquared>1.0;
    transmitted=vec3(0.0);
    if(tir) return 1.0;
    float exitCosine=sqrt(max(0.0,1.0-sineSquared));
    transmitted=normalize(ratio*incoming+(ratio*cosine-exitCosine)*normal);
    if(fromIor==toIor) return 0.0;
    float s=(fromIor*cosine-toIor*exitCosine)/(fromIor*cosine+toIor*exitCosine);
    float p=(toIor*cosine-fromIor*exitCosine)/(toIor*cosine+fromIor*exitCosine);
    return clamp(0.5*(s*s+p*p),0.0,1.0);
}
// Isotropic GGX visible normals. Alpha is authored roughness squared, without
// the opaque BRDF's finite minimum. The zero limit belongs to glassFresnel.
// Projected-disk construction and BSDF/PDF cancellation are described in PBRT:
// https://pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory
// https://pbr-book.org/4ed/Reflection_Models/Rough_Dielectric_BSDF
vec3 glassRoughNormal(vec3 incoming,vec3 normal,float alpha,float u,float v) {
    vec3 tangent=normalize(cross(normal,abs(normal.z)<0.999?vec3(0.0,0.0,1.0):vec3(1.0,0.0,0.0)));
    vec3 bitangent=cross(normal,tangent);
    vec3 view=-incoming;
    vec3 stretched=normalize(vec3(alpha*dot(view,tangent),alpha*dot(view,bitangent),dot(view,normal)));
    vec3 t1=stretched.z<0.99999?normalize(vec3(-stretched.y,stretched.x,0.0)):vec3(1.0,0.0,0.0);
    vec3 t2=cross(stretched,t1);
    float radius=sqrt(u),phi=6.283185307179586*v;
    float x=radius*cos(phi),y=radius*sin(phi);
    y=mix(sqrt(max(0.0,1.0-x*x)),y,0.5*(1.0+stretched.z));
    vec3 projected=t1*x+t2*y+stretched*sqrt(max(0.0,1.0-x*x-y*y));
    vec3 local=normalize(vec3(alpha*projected.x,alpha*projected.y,max(0.0,projected.z)));
    return tangent*local.x+bitangent*local.y+normal*local.z;
}
float glassRoughG1(float cosine,float alpha) {
    if(cosine<=0.0) return 0.0;
    return 2.0*cosine/(cosine+sqrt(alpha*alpha+(1.0-alpha*alpha)*cosine*cosine));
}
// Conditional single-scatter power weights after GGX visible-normal sampling.
// Each sample splits both Fresnel branches. Geometric hemispheres determine
// medium changes; a microfacet never changes the query hit's entering flag.
// Bias remains along the geometric normal, not along the sampled facet.
// Radiance transport additionally applies (fromIor/toIor)^2 to transmission.
// The weights here retain power accounting separately from that eta factor.
void glassRoughInterface(vec3 incoming,vec3 normal,float fromIor,float toIor,
                         float roughness,float u,float v,out vec3 reflected,
                         out vec3 transmitted,out vec3 power,out bool complete) {
    complete=false;reflected=vec3(0.0);transmitted=vec3(0.0);power=vec3(0.0);
    if(!glassSceneFinite3(incoming)||!glassSceneFinite3(normal)
        ||!glassSceneFinite(fromIor)||!glassSceneFinite(toIor)||fromIor<=0.0||toIor<=0.0
        ||!glassSceneFinite(u)||!glassSceneFinite(v)
        ||!glassSceneFinite(roughness)||roughness<=0.0||roughness>1.0
        ||dot(-incoming,normal)<=0.0||u<=0.0||u>=1.0||v<0.0||v>=1.0) return;
    // Index-matched media have no optical interface or masking deficit.
    if(fromIor==toIor) {
        transmitted=incoming;power=vec3(0.0,1.0,0.0);
        complete=glassSceneFinite3(incoming);return;
    }
    float alpha=roughness*roughness;
    if(alpha<=0.0) return;
    vec3 facet=glassRoughNormal(incoming,normal,alpha,u,v);
    if(!glassSceneFinite3(facet)||dot(-incoming,facet)<=0.0) return;
    bool tir;
    float fresnel=glassFresnel(incoming,facet,fromIor,toIor,transmitted,tir);
    reflected=normalize(reflect(incoming,facet));
    // Separable Smith agrees with the released opaque/environment GGX model.
    float reflectedPower=fresnel*glassRoughG1(dot(reflected,normal),alpha);
    float transmittedPower=tir?0.0:(1.0-fresnel)*glassRoughG1(-dot(transmitted,normal),alpha);
    power=vec3(reflectedPower,transmittedPower,max(0.0,1.0-reflectedPower-transmittedPower));
    complete=glassSceneFinite(fresnel)&&glassSceneFinite3(reflected)
        &&glassSceneFinite3(transmitted)&&glassSceneFinite3(power);
}
vec3 glassEnvironment(vec3 direction GX_ARGS) {
    if(GU_Sky[0].w>0.5) return analyticSkyRadiance(direction,GU_Sky[0],GU_Sky[1],GU_Sky[2],GU_Sky[3],GU_Sky[4]);
    vec3 color=vec3(0.0);
    for(int i=0;i<16;++i) {
        if(i>=GLASS_LIGHT_COUNT) break;
        if(GLASS_LIGHT_KIND(i)==2) color+=GU_LightColors[i].rgb*GU_LightColors[i].a;
    }
    return color;
}
vec3 glassReceiver(GlassClosestHit hit,vec3 direction,out bool complete GX_ARGS) {
    complete=false;
    vec3 base=vec3(0.0),emission=vec3(0.0);
    for(int object=0;object<64;++object) {
        uint id=uint(GU_GlassReceivers[object].w)|(uint(GU_GlassEmission[object].w)<<uint(16));
        if(id==hit.objectId) {base=GU_GlassReceivers[object].rgb;emission=GU_GlassEmission[object].rgb;complete=true;break;}
    }
    if(!complete) return vec3(0.0);
    vec3 normal=hit.outwardNormal;
    vec3 lighting=GU_Ambient.xyz*0.5;
    vec3 direct=vec3(0.0);
    for(int i=0;i<16;++i) {
        if(i>=GLASS_LIGHT_COUNT) break;
        int kind=GLASS_LIGHT_KIND(i);
        if(kind==2) {
            if(GU_Sky[0].w<0.5) lighting+=GU_LightColors[i].rgb*GU_LightColors[i].a*0.5;
            else direct+=analyticSkyBase(base,0.0,1.0,0.04,normal,-direction,
                GU_Sky[0],GU_Sky[1],GU_Sky[2],GU_Sky[3],GU_Sky[4]);
            continue;
        }
        vec3 delta=GU_LightPositions[i].xyz-hit.position;
        vec3 toLight=kind==0?GLASS_ROTATE(GU_LightRotations[i],vec3(0.0,0.0,1.0)):normalize(delta);
        float distance=length(delta),attenuation=1.0;
        if(kind==1||kind==3) {
            attenuation=1.0/max(distance*distance,1.0);
            float range=GU_LightPositions[i].w;
            if(range>0.0) attenuation*=pow(clamp(1.0-distance/range,0.0,1.0),2.0);
            if(kind==3) {
                vec3 spotDirection=GLASS_ROTATE(GU_LightRotations[i],vec3(0.0,0.0,-1.0));
                attenuation*=smoothstep(GU_LightCones[i].y,GU_LightCones[i].x,dot(spotDirection,normalize(-delta)));
            }
        }
        float visibility=opaqueVisibility(hit.position,normal,toLight,kind==0?3.402823e38:distance,GU_OpaqueControl GX_OPAQUE);
        direct+=GU_LightColors[i].rgb*GU_LightColors[i].a*attenuation
            *baseSurfaceDirect(base,0.0,1.0,0.04,0.0,normal,toLight,-direction)*visibility;
    }
    return base*lighting+direct+emission;
}
// Direct-light GGX reflection at a camera-side rough dielectric interface.
// Kind 2 lights stay in glassEnvironment, which reflected rays already reach.
// Every other kind is an analytic light that no traced ray can hit, so this
// lobe is the only path by which those lights reach the interface. It uses the
// dielectric Fresnel, GGX distribution and separable Smith masking of the
// interface split, with the surface path's 1/(4 n.v) normalisation. Its
// albedo stays below the interface Fresnel. It is outgoing radiance, not path
// power, so the transport receipts (terminal, masked, tail) do not change.
vec3 glassDirectSpecular(vec3 position,vec3 normal,vec3 toView,float roughness,
                         float fromIor,float toIor GX_ARGS) {
    vec3 specular=vec3(0.0);
    float alpha=roughness*roughness,alphaSquared=alpha*alpha;
    float viewCosine=dot(normal,toView);
    if(!glassSceneFinite3(position)||!glassSceneFinite3(normal)||!glassSceneFinite3(toView)
        ||!glassSceneFinite(roughness)||roughness<=0.0||roughness>1.0
        ||!glassSceneFinite(fromIor)||!glassSceneFinite(toIor)||fromIor<=0.0||toIor<=0.0
        ||viewCosine<=0.0) return specular;
    for(int i=0;i<16;++i) {
        if(i>=GLASS_LIGHT_COUNT) break;
        int kind=GLASS_LIGHT_KIND(i);
        if(kind==2) continue;
        vec3 delta=GU_LightPositions[i].xyz-position;
        vec3 toLight=kind==0?GLASS_ROTATE(GU_LightRotations[i],vec3(0.0,0.0,1.0)):normalize(delta);
        float lightCosine=dot(normal,toLight);
        if(lightCosine<=0.0) continue;
        float lightDistance=length(delta),attenuation=1.0;
        if(kind==1||kind==3) {
            attenuation=1.0/max(lightDistance*lightDistance,1.0);
            float range=GU_LightPositions[i].w;
            if(range>0.0) attenuation*=pow(clamp(1.0-lightDistance/range,0.0,1.0),2.0);
            if(kind==3) {
                vec3 spotDirection=GLASS_ROTATE(GU_LightRotations[i],vec3(0.0,0.0,-1.0));
                attenuation*=smoothstep(GU_LightCones[i].y,GU_LightCones[i].x,dot(spotDirection,normalize(-delta)));
            }
        }
        // Both light and view sit on the front side, so n.h is positive.
        vec3 halfVector=normalize(toLight+toView);
        float halfCosine=dot(normal,halfVector);
        float denominator=halfCosine*halfCosine*(alphaSquared-1.0)+1.0;
        float distribution=alphaSquared/(3.14159265358979323846*max(denominator*denominator,1.0e-30));
        vec3 transmittedUnused;bool tirUnused;
        float fresnel=glassFresnel(-toView,halfVector,fromIor,toIor,transmittedUnused,tirUnused);
        float masking=glassRoughG1(lightCosine,alpha)*glassRoughG1(viewCosine,alpha);
        float visibility=opaqueVisibility(position,normal,toLight,kind==0?3.402823e38:lightDistance,GU_OpaqueControl GX_OPAQUE);
        specular+=GU_LightColors[i].rgb*GU_LightColors[i].a*attenuation*visibility
            *fresnel*distribution*masking/(4.0*viewCosine);
    }
    return specular;
}
// One interface only. A secondary hit on the interface refuses recursive
// transport rather than substituting a background or an unfinished path.
vec3 thinInterfaceReceiver(vec3 origin,vec3 direction,uint water,out bool complete,out int status GX_ARGS) {
    complete=false;status=GLASS_SMOOTH_STATUS_UNCLASSIFIED;
    GlassClosestHit hit=glassSceneClosestHit(origin,direction,0.0,3.402823e38,GU_GlassControl GX_QUERY);
    if(hit.status<0) {status=hit.status==-2?GLASS_SMOOTH_STATUS_QUERY_BUDGET:GLASS_SMOOTH_STATUS_QUERY_REFUSED;return vec3(0.0);}
    if(hit.status==1 && hit.objectId==water) {status=GLASS_SMOOTH_STATUS_THIN_REENTRY;return vec3(0.0);}
    if(hit.status==0) { complete=true;return glassEnvironment(direction GX_PASS); }
    vec3 receiver=glassReceiver(hit,direction,complete GX_PASS);
    if(!complete) status=GLASS_SMOOTH_STATUS_THIN_RECEIVER;
    return receiver;
}
// Maps a RoughTransportStatus refusal (-1 to -10) into the shared reason range.
// Other values, including complete and smooth, report as unclassified.
int glassRoughReason(int status) {
    return status>=GLASS_ROUGH_STATUS_CLOSURE&&status<=GLASS_ROUGH_STATUS_INPUT
        ? GLASS_ROUGH_REASON_BASE+status : GLASS_SMOOTH_STATUS_UNCLASSIFIED;
}
// The raster pass owns which surface covers a pixel. The float32 camera query
// can disagree with it by less than the pixel's precision: its Moller-Trumbore
// test can pass between two water triangles at a shared edge, or beside a
// silhouette the fixed-point rasterizer covered, and where water meets a
// receiver a 24-bit depth tie can rank the receiver first by under one depth
// step. Each case refused a whole export frame for one pixel. The raster entry
// stands when the query hits the water, finds nothing, or finds another
// surface no nearer than the entry less this tolerance. A surface clearly in
// front of the entry still refuses. thinEntryOwned in C++ is the same test.
bool thinEntryOwned(GlassClosestHit first,uint water,vec3 camera,vec3 entry) {
    if(first.status==1&&first.objectId==water) return true;
    if(first.status==0) return true;
    if(first.status!=1) return false;
    float distance=length(entry-camera);
    float tolerance=max(4.0*glassBias(entry),1.0e-4*distance);
    return glassSceneFinite(first.distance)&&first.distance>=distance-tolerance;
}
vec3 thinInterfaceTransport(vec3 camera,vec3 entry,vec3 evaluatedNormal,out bool complete,out int status GX_ARGS) {
    complete=false;status=GLASS_SMOOTH_STATUS_UNCLASSIFIED;
    vec3 incoming=normalize(entry-camera);
    vec3 normal=normalize(evaluatedNormal);
    if(!glassSceneFinite3(incoming)||!glassSceneFinite3(normal)) {status=GLASS_SMOOTH_STATUS_THIN_NONFINITE_INCOMING;return vec3(0.0);}
    uint water=uint(GU_GlassTransport.z)|(uint(GU_GlassTransport.w)<<uint(16));
    GlassClosestHit first=glassSceneClosestHit(camera,incoming,0.0,3.402823e38,GU_GlassControl GX_QUERY);
    if(first.status<0) {status=first.status==-2?GLASS_SMOOTH_STATUS_QUERY_BUDGET:GLASS_SMOOTH_STATUS_QUERY_REFUSED;return vec3(0.0);}
    if(!thinEntryOwned(first,water,camera,entry)) {status=GLASS_SMOOTH_STATUS_THIN_FIRST_HIT;return vec3(0.0);}
    bool entering=dot(incoming,normal)<0.0;
    if(!entering) normal=-normal;
    float ior=GU_SurfaceCoating.y;
    vec3 transmitted;bool tir;
    float reflection=glassFresnel(incoming,normal,entering?1.0:ior,entering?ior:1.0,transmitted,tir);
    if(!glassSceneFinite(reflection)||!glassSceneFinite3(transmitted)) {status=GLASS_SMOOTH_STATUS_NONFINITE_SCATTER;return vec3(0.0);}
    float bias=glassBias(entry);
    vec3 radiance=vec3(0.0);
    if(reflection>0.0) {
        bool reflectedComplete;
        vec3 reflected=thinInterfaceReceiver(entry+normal*bias,normalize(reflect(incoming,normal)),water,reflectedComplete,status GX_PASS);
        if(!reflectedComplete) return vec3(0.0);
        radiance+=reflection*reflected;
    }
    if(!tir && reflection<1.0) {
        bool transmittedComplete;
        vec3 refracted=thinInterfaceReceiver(entry-normal*bias,transmitted,water,transmittedComplete,status GX_PASS);
        if(!transmittedComplete) return vec3(0.0);
        radiance+=(1.0-reflection)*refracted;
    }
    complete=glassSceneFinite3(radiance);
    if(!complete) status=GLASS_SMOOTH_STATUS_THIN_RADIANCE;
    return radiance;
}
// A rough water surface has exactly one optical interface. The evaluated wave
// normal defines its microfacet distribution; the actual hit fixes medium side,
// branch hemispheres and origin bias. No closed-shell continuation is admitted.
vec3 roughThinInterfaceTransport(vec3 camera,vec3 entry,vec3 evaluatedNormal,
                                float roughness,bool receiptOnly,out vec3 terminal,
                                out vec3 masked,out vec3 tail,out int status,
                                out int queryCount GX_ARGS) {
    terminal=vec3(0.0);masked=vec3(0.0);tail=vec3(0.0);queryCount=0;
    status=GLASS_ROUGH_STATUS_INPUT;
    if(roughness==0.0) {
        bool smoothComplete;int smoothStatus;
        vec3 smoothColour=thinInterfaceTransport(camera,entry,evaluatedNormal,smoothComplete,smoothStatus GX_PASS);
        status=smoothComplete?GLASS_ROUGH_STATUS_SMOOTH:GLASS_ROUGH_STATUS_QUERY;return smoothColour;
    }
    if(!glassSceneFinite(roughness)||roughness<0.0||roughness>1.0
        ||!glassSceneFinite3(camera)||!glassSceneFinite3(entry)||!glassSceneFinite3(evaluatedNormal)
        ||dot(evaluatedNormal,evaluatedNormal)<=0.0||GU_GlassTransport.x!=2.0
        ||GU_GlassTransport.y!=0.0||!glassSceneFinite(GU_SurfaceCoating.y)
        ||GU_SurfaceCoating.y<1.0||GU_SurfaceCoating.y>3.0) return vec3(0.0);
    vec3 incoming=normalize(entry-camera);
    if(!glassSceneFinite3(incoming)) return vec3(0.0);
    GlassClosestHit first=glassSceneClosestHit(camera,incoming,0.0,3.402823e38,GU_GlassControl GX_QUERY);
    ++queryCount;
    uint water=uint(GU_GlassTransport.z)|(uint(GU_GlassTransport.w)<<uint(16));
    if(first.status<0) {status=GLASS_ROUGH_STATUS_QUERY;return vec3(0.0);}
    if(first.status!=1||first.objectId!=water||length(first.position-entry)>4.0*glassBias(entry))
        {status=GLASS_ROUGH_STATUS_IDENTITY;return vec3(0.0);}
    vec3 geometric=first.entering?first.outwardNormal:-first.outwardNormal;
    vec3 normal=normalize(evaluatedNormal)*(first.entering?1.0:-1.0);
    if(!glassSceneFinite3(geometric)||dot(normal,geometric)<=0.0||dot(-incoming,normal)<=0.0)
        {status=GLASS_ROUGH_STATUS_IDENTITY;return vec3(0.0);}
    float fromIor=first.entering?1.0:GU_SurfaceCoating.y;
    float toIor=first.entering?GU_SurfaceCoating.y:1.0;
    float eta=fromIor/toIor,bias=glassBias(first.position);
    vec3 radiance=vec3(0.0);
    // The same evaluated normal that orients the sampled microfacets also frames this lobe.
    if(!receiptOnly) radiance+=glassDirectSpecular(first.position,normal,-incoming,roughness,fromIor,toIor GX_PASS);
    for(int sampleIndex=0;sampleIndex<GLASS_ROUGH_SAMPLES;++sampleIndex) {
        int digits=sampleIndex;float inverse=0.0,place=0.5;
        for(int bit=0;bit<4;++bit) {inverse+=float(digits%2)*place;digits/=2;place*=0.5;}
        vec3 reflected,transmitted,fractions;bool interfaceComplete;
        glassRoughInterface(incoming,normal,fromIor,toIor,roughness,
            (float(sampleIndex)+0.5)/float(GLASS_ROUGH_SAMPLES),inverse,
            reflected,transmitted,fractions,interfaceComplete);
        if(!interfaceComplete) {status=GLASS_ROUGH_STATUS_NONFINITE;return vec3(0.0);}
        masked+=vec3(fractions.z/float(GLASS_ROUGH_SAMPLES));
        for(int branch=0;branch<2;++branch) {
            bool transmission=branch==0;
            float power=(transmission?fractions.y:fractions.x)/float(GLASS_ROUGH_SAMPLES);
            vec3 direction=transmission?transmitted:reflected;
            if(!glassSceneFinite(power)||power<0.0)
                {status=GLASS_ROUGH_STATUS_NONFINITE;return vec3(0.0);}
            if(power<0.00001/float(GLASS_ROUGH_SAMPLES)) {tail+=vec3(power);continue;}
            // A shading microfacet cannot change the actual triangle's medium.
            float hemisphere=dot(direction,geometric)*(transmission?-1.0:1.0);
            if(hemisphere<=0.0) {masked+=vec3(power);continue;}
            vec3 origin=first.position+geometric*(transmission?-bias:bias);
            GlassClosestHit target=glassSceneClosestHit(origin,direction,0.0,3.402823e38,GU_GlassControl GX_QUERY);
            ++queryCount;
            if(target.status<0) {status=GLASS_ROUGH_STATUS_QUERY;return vec3(0.0);}
            if(target.status==1&&target.objectId==water)
                {status=GLASS_ROUGH_STATUS_IDENTITY;return vec3(0.0);}
            vec3 receiver=vec3(0.0);
            if(target.status==0) {
                if(!receiptOnly) receiver=glassEnvironment(direction GX_PASS);
            } else if(!receiptOnly) {
                bool receiverComplete;
                receiver=glassReceiver(target,direction,receiverComplete GX_PASS);
                if(!receiverComplete) {status=GLASS_ROUGH_STATUS_IDENTITY;return vec3(0.0);}
            } else {
                bool owned=false;
                for(int object=0;object<64;++object) {
                    uint id=uint(GU_GlassReceivers[object].w)|(uint(GU_GlassEmission[object].w)<<uint(16));
                    if(id==target.objectId) {owned=true;break;}
                }
                if(!owned) {status=GLASS_ROUGH_STATUS_IDENTITY;return vec3(0.0);}
            }
            terminal+=vec3(power);
            radiance+=power*(transmission?eta*eta:1.0)*receiver;
        }
    }
    for(int channel=0;channel<3;++channel) {
        float sum=terminal[channel]+masked[channel]+tail[channel];
        if(!glassSceneFinite(sum)||!glassSceneFinite(radiance[channel]))
            {status=GLASS_ROUGH_STATUS_NONFINITE;return vec3(0.0);}
        if(terminal[channel]<0.0||terminal[channel]>1.0||masked[channel]<0.0||masked[channel]>1.0
            ||tail[channel]<0.0||tail[channel]>1.0||abs(sum-1.0)>GLASS_ROUGH_CONSERVATION)
            {status=GLASS_ROUGH_STATUS_CLOSURE;return vec3(0.0);}
        if(masked[channel]+tail[channel]>GLASS_ROUGH_UNRESOLVED)
            {status=GLASS_ROUGH_STATUS_ENERGY;return vec3(0.0);}
    }
    status=GLASS_ROUGH_STATUS_COMPLETE;
    return radiance;
}
vec3 glassTransport(vec3 camera,vec3 entry,out bool complete,out int status GX_ARGS) {
    complete=false;status=GLASS_SMOOTH_STATUS_UNCLASSIFIED;
    vec3 incoming=normalize(entry-camera);
    GlassClosestHit first=glassSceneClosestHit(camera,incoming,0.0,3.402823e38,GU_GlassControl GX_QUERY);
    uint volume=uint(GU_GlassTransport.z)|(uint(GU_GlassTransport.w)<<uint(16));
    if(first.status<0) {status=first.status==-2?GLASS_SMOOTH_STATUS_QUERY_BUDGET:GLASS_SMOOTH_STATUS_QUERY_REFUSED;return vec3(0.0);}
    if(first.status!=1||first.objectId!=volume) {status=GLASS_SMOOTH_STATUS_FIRST_HIT;return vec3(0.0);}
    vec3 radiance=vec3(0.0);
    for(int channel=0;channel<3;++channel) {
        float wavelength=channel==0?0.6563:channel==1?0.5461:0.4358;
        float ior=GU_SurfaceCoating.y+GU_GlassTransport.y*(1.0/(wavelength*wavelength)-1.0/(0.58756*0.58756));
        vec3 origins[16],directions[16];float weights[16];int depths[16];bool media[16];
        int pending=1;
        float discarded=0.0;
        origins[0]=camera;directions[0]=incoming;weights[0]=1.0;depths[0]=0;media[0]=!first.entering;
        for(int visit=0;visit<64;++visit) {
            if(pending==0) break;
            --pending;
            vec3 origin=origins[pending],direction=directions[pending];
            float weight=weights[pending];int depth=depths[pending];bool inside=media[pending];
            GlassClosestHit hit=glassSceneClosestHit(origin,direction,0.0,3.402823e38,GU_GlassControl GX_QUERY);
            if(hit.status<0) {status=hit.status==-2?GLASS_SMOOTH_STATUS_QUERY_BUDGET:GLASS_SMOOTH_STATUS_QUERY_REFUSED;return vec3(0.0);}
            if(hit.status==0) {
                if(inside) {status=GLASS_SMOOTH_STATUS_MEDIUM;return vec3(0.0);}
                radiance[channel]+=weight*glassEnvironment(direction GX_PASS)[channel];continue;
            }
            if(hit.objectId!=volume) {
                if(inside) {status=GLASS_SMOOTH_STATUS_MEDIUM;return vec3(0.0);}
                bool receiverComplete;
                vec3 receiver=glassReceiver(hit,direction,receiverComplete GX_PASS);
                if(!receiverComplete) {status=GLASS_SMOOTH_STATUS_RECEIVER;return vec3(0.0);}
                radiance[channel]+=weight*receiver[channel];continue;
            }
            if(inside==hit.entering) {status=GLASS_SMOOTH_STATUS_ORIENTATION;return vec3(0.0);}
            if(depth>=GLASS_INTERFACE_DEPTH) {status=GLASS_SMOOTH_STATUS_DEPTH;return vec3(0.0);}
            vec3 normal=hit.entering?hit.outwardNormal:-hit.outwardNormal;
            vec3 transmitted;bool tir;
            float reflection=glassFresnel(direction,normal,inside?ior:1.0,inside?1.0:ior,transmitted,tir);
            if(!glassSceneFinite(reflection)||!glassSceneFinite3(transmitted)) {status=GLASS_SMOOTH_STATUS_NONFINITE_SCATTER;return vec3(0.0);}
            float bias=glassBias(hit.position);
            if(!tir&&weight*(1.0-reflection)<0.00001) discarded+=weight*(1.0-reflection);
            if(weight*reflection<0.00001) discarded+=weight*reflection;
            if(discarded>0.0001) {status=GLASS_SMOOTH_STATUS_DISCARDED;return vec3(0.0);}
            if(!tir&&weight*(1.0-reflection)>=0.00001) {
                if(pending>=16) {status=GLASS_SMOOTH_STATUS_PENDING;return vec3(0.0);}
                origins[pending]=hit.position-normal*bias;directions[pending]=transmitted;
                weights[pending]=weight*(1.0-reflection);depths[pending]=depth+1;media[pending]=!inside;++pending;
            }
            if(weight*reflection>=0.00001) {
                if(pending>=16) {status=GLASS_SMOOTH_STATUS_PENDING;return vec3(0.0);}
                origins[pending]=hit.position+normal*bias;directions[pending]=normalize(reflect(direction,normal));
                weights[pending]=weight*reflection;depths[pending]=depth+1;media[pending]=inside;++pending;
            }
        }
        if(pending!=0) {status=GLASS_SMOOTH_STATUS_VISITS;return vec3(0.0);}
    }
    complete=glassSceneFinite3(radiance);
    return radiance;
}
// Native callers must retain float32 terminal/masked/tail outputs and require
// COMPLETE before publishing any rough Color or AOV. SMOOTH carries no rough
// receipt, since zero roughness delegates to the released path verbatim.
// receiptOnly follows the same paths while skipping terminal radiance shading.
vec3 roughGlassTransport(vec3 camera,vec3 entry,float roughness,bool receiptOnly,
                         out vec3 terminal,out vec3 masked,out vec3 tail,
                         out int status,out int queryCount GX_ARGS) {
    terminal=vec3(0.0);masked=vec3(0.0);tail=vec3(0.0);queryCount=0;
    status=GLASS_ROUGH_STATUS_INPUT;
    if(roughness==0.0) {
        bool smoothComplete;int smoothStatus;
        vec3 smoothColour=glassTransport(camera,entry,smoothComplete,smoothStatus GX_PASS);
        status=smoothComplete?GLASS_ROUGH_STATUS_SMOOTH:GLASS_ROUGH_STATUS_QUERY;return smoothColour;
    }
    if(!glassSceneFinite(roughness)||roughness<0.0||roughness>1.0
        ||!glassSceneFinite3(camera)||!glassSceneFinite3(entry)||GU_GlassTransport.x!=1.0) return vec3(0.0);
    vec3 incoming=normalize(entry-camera);
    if(!glassSceneFinite3(incoming)) return vec3(0.0);
    GlassClosestHit first=glassSceneClosestHit(camera,incoming,0.0,3.402823e38,GU_GlassControl GX_QUERY);
    ++queryCount;
    uint volume=uint(GU_GlassTransport.z)|(uint(GU_GlassTransport.w)<<uint(16));
    if(first.status<0) {status=GLASS_ROUGH_STATUS_QUERY;return vec3(0.0);}
    if(first.status!=1||first.objectId!=volume) {status=GLASS_ROUGH_STATUS_IDENTITY;return vec3(0.0);}
    if(!first.entering) {status=GLASS_ROUGH_STATUS_INTERNAL;return vec3(0.0);}
    vec3 radiance=vec3(0.0);
    #pragma optionNV(unroll none)
    for(int channel=0;channel<3;++channel) {
        float wavelength=channel==0?0.6563:channel==1?0.5461:0.4358;
        float ior=GU_SurfaceCoating.y+GU_GlassTransport.y*(1.0/(wavelength*wavelength)-1.0/(0.58756*0.58756));
        if(!glassSceneFinite(ior)||ior<1.0||ior>3.0) {status=GLASS_ROUGH_STATUS_INPUT;return vec3(0.0);}
        // Analytic lights reach the camera-side interface directly, not through the sampled paths.
        if(!receiptOnly) radiance[channel]+=glassDirectSpecular(first.position,first.outwardNormal,-incoming,
            roughness,1.0,ior GX_PASS)[channel];
        #pragma optionNV(unroll none)
        for(int sampleIndex=0;sampleIndex<GLASS_ROUGH_SAMPLES;++sampleIndex) {
            vec3 origins[16],directions[16];float powers[16],throughputs[16];int depths[16];bool media[16];
            int pending=1;
            origins[0]=camera;directions[0]=incoming;powers[0]=1.0/float(GLASS_ROUGH_SAMPLES);
            throughputs[0]=powers[0];depths[0]=0;media[0]=false;
            #pragma optionNV(unroll none)
            for(int visit=0;visit<GLASS_ROUGH_VISITS;++visit) {
                if(pending==0) break;
                --pending;
                vec3 origin=origins[pending],direction=directions[pending];
                float power=powers[pending],throughput=throughputs[pending];
                int depth=depths[pending];bool inside=media[pending];
                if(queryCount>=GLASS_ROUGH_QUERY_LIMIT) {status=GLASS_ROUGH_STATUS_VISITS;return vec3(0.0);}
                GlassClosestHit hit=glassSceneClosestHit(origin,direction,0.0,3.402823e38,GU_GlassControl GX_QUERY);
                ++queryCount;
                if(hit.status<0) {status=GLASS_ROUGH_STATUS_QUERY;return vec3(0.0);}
                if(hit.status==0||hit.objectId!=volume) {
                    if(inside) {status=GLASS_ROUGH_STATUS_IDENTITY;return vec3(0.0);}
                    terminal[channel]+=power;
                    if(hit.status==0) {
                        if(!receiptOnly) radiance[channel]+=throughput*glassEnvironment(direction GX_PASS)[channel];
                    } else if(!receiptOnly) {
                        bool receiverComplete;
                        vec3 receiver=glassReceiver(hit,direction,receiverComplete GX_PASS);
                        if(!receiverComplete) {status=GLASS_ROUGH_STATUS_IDENTITY;return vec3(0.0);}
                        radiance[channel]+=throughput*receiver[channel];
                    } else {
                        bool owned=false;
                        #pragma optionNV(unroll none)
                        for(int object=0;object<64;++object) {
                            uint id=uint(GU_GlassReceivers[object].w)|(uint(GU_GlassEmission[object].w)<<uint(16));
                            if(id==hit.objectId) {owned=true;break;}
                        }
                        if(!owned) {status=GLASS_ROUGH_STATUS_IDENTITY;return vec3(0.0);}
                    }
                    continue;
                }
                if(hit.materialId!=first.materialId||hit.instanceId!=first.instanceId||inside==hit.entering)
                    {status=GLASS_ROUGH_STATUS_IDENTITY;return vec3(0.0);}
                if(depth>=GLASS_INTERFACE_DEPTH) {status=GLASS_ROUGH_STATUS_DEPTH;return vec3(0.0);}
                vec3 normal=hit.entering?hit.outwardNormal:-hit.outwardNormal;
                int stratum=(sampleIndex*(2*depth+1)+5*depth)%GLASS_ROUGH_SAMPLES;
                int digits=stratum;float inverse=0.0,place=0.5;
                for(int bit=0;bit<4;++bit) {inverse+=float(digits%2)*place;digits/=2;place*=0.5;}
                float sampleU=(float(stratum)+0.5)/float(GLASS_ROUGH_SAMPLES);
                float sampleV=fract(inverse+float(depth)*0.6180339887498948482);
                float fromIor=inside?ior:1.0,toIor=inside?1.0:ior;
                vec3 reflected,transmitted,fractions;bool interfaceComplete;
                glassRoughInterface(direction,normal,fromIor,toIor,roughness,sampleU,sampleV,
                    reflected,transmitted,fractions,interfaceComplete);
                if(!interfaceComplete) {status=GLASS_ROUGH_STATUS_NONFINITE;return vec3(0.0);}
                masked[channel]+=power*fractions.z;
                float bias=glassBias(hit.position),eta=fromIor/toIor;
                for(int branch=0;branch<2;++branch) {
                    bool transmission=branch==0;
                    float fraction=transmission?fractions.y:fractions.x;
                    float branchPower=power*fraction;
                    float branchThroughput=throughput*fraction*(transmission?eta*eta:1.0);
                    if(!glassSceneFinite(branchPower)||!glassSceneFinite(branchThroughput))
                        {status=GLASS_ROUGH_STATUS_NONFINITE;return vec3(0.0);}
                    if(branchPower<0.00001/float(GLASS_ROUGH_SAMPLES)) {tail[channel]+=branchPower;continue;}
                    if(pending>=16) {status=GLASS_ROUGH_STATUS_PENDING;return vec3(0.0);}
                    origins[pending]=hit.position+normal*(transmission?-bias:bias);
                    directions[pending]=transmission?transmitted:reflected;
                    powers[pending]=branchPower;throughputs[pending]=branchThroughput;
                    depths[pending]=depth+1;media[pending]=transmission?!inside:inside;++pending;
                }
            }
            if(pending!=0) {status=GLASS_ROUGH_STATUS_VISITS;return vec3(0.0);}
        }
        float sum=terminal[channel]+masked[channel]+tail[channel];
        if(!glassSceneFinite(sum)||!glassSceneFinite(radiance[channel]))
            {status=GLASS_ROUGH_STATUS_NONFINITE;return vec3(0.0);}
        if(terminal[channel]<0.0||terminal[channel]>1.0||masked[channel]<0.0||masked[channel]>1.0
            ||tail[channel]<0.0||tail[channel]>1.0||abs(sum-1.0)>GLASS_ROUGH_CONSERVATION)
            {status=GLASS_ROUGH_STATUS_CLOSURE;return vec3(0.0);}
        if(masked[channel]+tail[channel]>GLASS_ROUGH_UNRESOLVED)
            {status=GLASS_ROUGH_STATUS_ENERGY;return vec3(0.0);}
    }
    status=GLASS_ROUGH_STATUS_COMPLETE;
    return radiance;
}
)shader";
    // The released v1 body above stays unchanged. V2 replaces its interface split.
    source += multipleScatterShaderRecipe<String>();
    source += "\n#define GLASS_CAMERA_RESIDUAL (GLASS_ROUGH_UNRESOLVED/128.0)"
        "\n#define GLASS_CAMERA_V2_COMPLETE " + surfacesource::number<String>(kMultipleScatterCameraComplete) + "\n";
    source += R"shader(vec3 roughGlassTransportV2(vec3 camera,vec3 entry,float roughness,bool receiptOnly,uint pixelKey,
                         out vec3 terminal,out vec3 masked,out vec3 tail,
                         out int status,out int queryCount,out int microEvents GX_ARGS) {
    terminal=vec3(0.0);masked=vec3(0.0);tail=vec3(0.0);queryCount=0;microEvents=0;
    status=GLASS_ROUGH_STATUS_INPUT;
    if(roughness==0.0) {
        bool smoothComplete;int smoothStatus;
        vec3 smoothColour=glassTransport(camera,entry,smoothComplete,smoothStatus GX_PASS);
        status=smoothComplete?GLASS_ROUGH_STATUS_SMOOTH:GLASS_ROUGH_STATUS_QUERY;return smoothColour;
    }
    if(!glassSceneFinite(roughness)||roughness<0.0||roughness>1.0
        ||!glassSceneFinite3(camera)||!glassSceneFinite3(entry)||GU_GlassTransport.x!=1.0) return vec3(0.0);
    vec3 incoming=normalize(entry-camera);
    if(!glassSceneFinite3(incoming)) return vec3(0.0);
    GlassClosestHit first=glassSceneClosestHit(camera,incoming,0.0,3.402823e38,GU_GlassControl GX_QUERY);
    ++queryCount;
    uint volume=uint(GU_GlassTransport.z)|(uint(GU_GlassTransport.w)<<uint(16));
    if(first.status<0) {status=GLASS_ROUGH_STATUS_QUERY;return vec3(0.0);}
    if(first.status!=1||first.objectId!=volume) {status=GLASS_ROUGH_STATUS_IDENTITY;return vec3(0.0);}
    if(!first.entering) {status=GLASS_ROUGH_STATUS_INTERNAL;return vec3(0.0);}
    vec3 radiance=vec3(0.0);
    #pragma optionNV(unroll none)
    for(int channel=0;channel<3;++channel) {
        float wavelength=channel==0?0.6563:channel==1?0.5461:0.4358;
        float ior=GU_SurfaceCoating.y+GU_GlassTransport.y*(1.0/(wavelength*wavelength)-1.0/(0.58756*0.58756));
        if(!glassSceneFinite(ior)||ior<1.0||ior>3.0) {status=GLASS_ROUGH_STATUS_INPUT;return vec3(0.0);}
        // Analytic lights reach the camera-side interface directly, not through the sampled paths.
        if(!receiptOnly) radiance[channel]+=glassDirectSpecular(first.position,first.outwardNormal,-incoming,
            roughness,1.0,ior GX_PASS)[channel];
        #pragma optionNV(unroll none)
        for(int sampleIndex=0;sampleIndex<GLASS_ROUGH_SAMPLES;++sampleIndex) {
            vec3 origins[16],directions[16];float powers[16],throughputs[16];int depths[16];bool media[16];uint histories[16];
            int pending=1;
            origins[0]=camera;directions[0]=incoming;powers[0]=1.0/float(GLASS_ROUGH_SAMPLES);
            throughputs[0]=powers[0];depths[0]=0;media[0]=false;histories[0]=2166136261u;
            #pragma optionNV(unroll none)
            for(int visit=0;visit<GLASS_ROUGH_VISITS;++visit) {
                if(pending==0) break;
                --pending;
                vec3 origin=origins[pending],direction=directions[pending];
                float power=powers[pending],throughput=throughputs[pending];
                int depth=depths[pending];bool inside=media[pending];uint history=histories[pending];
                if(queryCount>=GLASS_ROUGH_QUERY_LIMIT) {status=GLASS_ROUGH_STATUS_VISITS;return vec3(0.0);}
                GlassClosestHit hit=glassSceneClosestHit(origin,direction,0.0,3.402823e38,GU_GlassControl GX_QUERY);
                ++queryCount;
                if(hit.status<0) {status=GLASS_ROUGH_STATUS_QUERY;return vec3(0.0);}
                if(hit.status==0||hit.objectId!=volume) {
                    if(inside) {status=GLASS_ROUGH_STATUS_IDENTITY;return vec3(0.0);}
                    terminal[channel]+=power;
                    if(hit.status==0) {
                        if(!receiptOnly) radiance[channel]+=throughput*glassEnvironment(direction GX_PASS)[channel];
                    } else if(!receiptOnly) {
                        bool receiverComplete;
                        vec3 receiver=glassReceiver(hit,direction,receiverComplete GX_PASS);
                        if(!receiverComplete) {status=GLASS_ROUGH_STATUS_IDENTITY;return vec3(0.0);}
                        radiance[channel]+=throughput*receiver[channel];
                    } else {
                        bool owned=false;
                        #pragma optionNV(unroll none)
                        for(int object=0;object<64;++object) {
                            uint id=uint(GU_GlassReceivers[object].w)|(uint(GU_GlassEmission[object].w)<<uint(16));
                            if(id==hit.objectId) {owned=true;break;}
                        }
                        if(!owned) {status=GLASS_ROUGH_STATUS_IDENTITY;return vec3(0.0);}
                    }
                    continue;
                }
                if(hit.materialId!=first.materialId||hit.instanceId!=first.instanceId||inside==hit.entering)
                    {status=GLASS_ROUGH_STATUS_IDENTITY;return vec3(0.0);}
                if(depth>=GLASS_INTERFACE_DEPTH) {status=GLASS_ROUGH_STATUS_DEPTH;return vec3(0.0);}
                vec3 normal=hit.entering?hit.outwardNormal:-hit.outwardNormal;
                float fromIor=inside?ior:1.0,toIor=inside?1.0:ior;
                uint key=glassMsEntryKey(hit.objectId,hit.instanceId,hit.triangleId,pixelKey);
                uint seed=glassMsSeed(key,uint(sampleIndex),uint(depth),history);
                vec4 exits[GLASS_MS_EXITS];uint orders[GLASS_MS_EXITS],count,events,draws;float residual;
                int walk=glassMsWalk(direction,normal,fromIor,toIor,roughness,seed,
                    uint(GLASS_MS_EVENTS),uint(GLASS_MS_EXITS),GLASS_CAMERA_RESIDUAL,
                    exits,orders,count,events,draws,residual);
                microEvents+=int(events);
                if(walk<0 || draws>4u*events || !glassSceneFinite(residual) || residual<0.0 || residual>1.0)
                    {status=GLASS_ROUGH_STATUS_NONFINITE;return vec3(0.0);}
                // Charge unfinished unit flux once, scaled by this path's power.
                // A bounded walk can retain a tail; whole-pixel policy decides admission.
                tail[channel]+=power*residual;
                float bias=glassBias(hit.position),eta=fromIor/toIor;
                for(uint branch=0u;branch<count;++branch) {
                    bool transmission=exits[branch].w<0.0;
                    float fraction=abs(exits[branch].w);
                    float branchPower=power*fraction;
                    float branchThroughput=throughput*fraction*(transmission?eta*eta:1.0);
                    if(!glassSceneFinite(branchPower)||!glassSceneFinite(branchThroughput)
                        ||!glassSceneFinite3(exits[branch].xyz)
                        ||dot(exits[branch].xyz,normal)*(transmission?-1.0:1.0)<=0.0)
                        {status=GLASS_ROUGH_STATUS_NONFINITE;return vec3(0.0);}
                    if(branchPower<0.00001/float(GLASS_ROUGH_SAMPLES)) {tail[channel]+=branchPower;continue;}
                    if(pending>=16) {status=GLASS_ROUGH_STATUS_PENDING;return vec3(0.0);}
                    origins[pending]=hit.position+normal*(transmission?-bias:bias);
                    directions[pending]=exits[branch].xyz;
                    powers[pending]=branchPower;throughputs[pending]=branchThroughput;
                    depths[pending]=depth+1;media[pending]=transmission?!inside:inside;
                    histories[pending]=glassMsChildHistory(history,orders[branch],transmission);++pending;
                }
            }
            if(pending!=0) {status=GLASS_ROUGH_STATUS_VISITS;return vec3(0.0);}
        }
        float sum=terminal[channel]+masked[channel]+tail[channel];
        if(!glassSceneFinite(sum)||!glassSceneFinite(radiance[channel]))
            {status=GLASS_ROUGH_STATUS_NONFINITE;return vec3(0.0);}
        if(terminal[channel]<0.0||terminal[channel]>1.0||masked[channel]<0.0||masked[channel]>1.0
            ||tail[channel]<0.0||tail[channel]>1.0||abs(sum-1.0)>GLASS_ROUGH_CONSERVATION)
            {status=GLASS_ROUGH_STATUS_CLOSURE;return vec3(0.0);}
        if(masked[channel]+tail[channel]>GLASS_ROUGH_UNRESOLVED)
            {status=GLASS_ROUGH_STATUS_ENERGY;return vec3(0.0);}
    }
    status=GLASS_CAMERA_V2_COMPLETE;
    return radiance;
}
uint glassCameraPixelKey(vec2 pixel) {
    return ((2166136261u^uint(floor(pixel.x)))*16777619u^uint(floor(pixel.y)))*16777619u;
}
vec3 roughCameraTransport(vec3 camera,vec3 entry,float roughness,bool receiptOnly,vec2 pixel,
    out vec3 terminal,out vec3 masked,out vec3 tail,out int status,out int queries,out int events GX_ARGS) {
    events=0;
    if(GU_RoughCameraControl.x==2.0)
        return roughGlassTransportV2(camera,entry,roughness,receiptOnly,glassCameraPixelKey(pixel),
            terminal,masked,tail,status,queries,events GX_PASS);
    return roughGlassTransport(camera,entry,roughness,receiptOnly,terminal,masked,tail,status,queries GX_PASS);
}

)shader";
    if (selectedPrimary) {
        if constexpr(!std::is_same_v<String,std::string>) throw std::length_error("Surface source admission excludes selected primary");
        else {
        const auto begin=source.find("vec3 roughGlassTransportV2(");
        auto selected=source.substr(begin,source.find("uint glassCameraPixelKey",begin)-begin);
        replace(selected,"roughGlassTransportV2(vec3 camera,vec3 entry,", "roughGlassTransportV2Selected(vec3 camera,vec3 incoming,GlassClosestHit first,");
        const auto validation=selected.find("    if(roughness==0.0)");
        const auto volume=selected.find("    uint volume=",validation);
        selected.replace(validation,volume-validation,
            "    if(!glassSceneFinite(roughness)||roughness<=0.0||roughness>1.0\n"
            "        ||!glassSceneFinite3(camera)||!glassSceneFinite3(incoming)||GU_GlassTransport.x!=1.0) return vec3(0.0);\n"
            "    queryCount=1; // The completed native selection already queried the primary.\n");
        replace(selected,
            "GlassClosestHit hit=glassSceneClosestHit(origin,direction,0.0,3.402823e38,GU_GlassControl GX_QUERY);\n                ++queryCount;",
            "GlassClosestHit hit=first;\n                if(depth!=0) {hit=glassSceneClosestHit(origin,direction,0.0,3.402823e38,GU_GlassControl GX_QUERY);++queryCount;}");
        source += selected;
    }
    }
    if (imageSecondary)
    {
        smoothglass::replace(source, "vec3 glassEnvironment(vec3 direction GX_ARGS) {",
            "vec3 glassEnvironment(vec3 direction GX_ARGS) {\n"
            "    if(GU_Sky[0].w>1.5) return environmentImageRadiance(direction,GU_Sky[0],GU_Sky[1] GX_IMAGE);");
        smoothglass::replace(source, "else direct+=analyticSkyBase(base,0.0,1.0,0.04,normal,-direction,",
            "else if(GU_Sky[0].w>1.5) direct+=environmentImageBase(base,0.0,1.0,0.04,normal,-direction,GU_Sky[0],GU_Sky[1] GX_IMAGE);\n"
            "            else direct+=analyticSkyBase(base,0.0,1.0,0.04,normal,-direction,");
    }
    if (metal)
    {
        // The NVIDIA unroll pragmas in the shared text are GLSL-only.
        smoothglass::replace(source, "#pragma optionNV(unroll none)\n", "");
        smoothglass::replace(source, "out vec4 directions[GLASS_MS_EXITS]", "thread float4 (&directions)[GLASS_MS_EXITS]");
        smoothglass::replace(source, "out uint orders[GLASS_MS_EXITS]", "thread uint (&orders)[GLASS_MS_EXITS]");
        smoothglass::replace(source, "out uint ", "thread uint& ");
        smoothglass::replace(source, "out float ", "thread float& ");
        smoothglass::replace(source, "vec4", "float4");
        smoothglass::replace(source, "vec2", "float2");
        smoothglass::replace(source, "out vec3 ", "thread float3& ");
        smoothglass::replace(source, "out bool ", "thread bool& ");
        smoothglass::replace(source, "out int ", "thread int& ");
        smoothglass::replace(source, "vec3", "float3");
        for (const auto& [old, value] : std::initializer_list<std::pair<const char*,const char*>> {
            {"GU_Sky","u.sky"}, {"GU_LightColors","u.lightColors"}, {"GU_LightPositions","u.lightPositions"},
            {"GU_LightRotations","u.lightRotations"}, {"GU_LightCones","u.lightCones"}, {"GU_Ambient","u.ambient"},
            {"GU_GlassReceivers","u.glassReceivers"}, {"GU_GlassEmission","u.glassEmission"},
            {"GU_GlassControl","u.glassControl"}, {"GU_GlassTransport","u.glassTransport"},
            {"GU_RoughCameraControl","u.roughCameraControl"}, {"GU_SurfaceCoating","u.surfaceCoating"}, {"GU_OpaqueControl","u.opaqueControl"}})
            smoothglass::replace(source, old, value);
        const String imageArgument = surfacesource::literal<String>(imageSecondary ? ", texture2d<float> environmentTexture" : "");
        const String imagePass = surfacesource::literal<String>(imageSecondary ? ", environmentTexture" : "");
        return "\n#define GX_ARGS , constant Uniforms& u, texture2d<float> opaqueTexture, texture2d<float> glassTexture" + imageArgument + "\n"
            "#define GX_PASS , u, opaqueTexture, glassTexture" + imagePass + "\n#define GX_QUERY , glassTexture\n#define GX_OPAQUE , opaqueTexture\n"
            + surfacesource::literal<String>(imageSecondary ? "#define GX_IMAGE , environmentTexture\n" : "") +
            "#define GLASS_LIGHT_COUNT int(u.ambient.w)\n#define GLASS_LIGHT_KIND(i) int(u.lightKinds[i].x)\n"
            "#define GLASS_ROTATE glassRotateQ\n"
            "float3 glassRotateQ(float4 q,float3 v) {return v+2.0f*cross(q.xyz,cross(q.xyz,v)+q.w*v);}\n" + source;
    }
    smoothglass::replace(source, "GU_", "u");
    return "\nuniform vec4 uRoughCameraControl;\nuniform vec4 uGlassTransport;\nuniform vec4 uGlassReceivers[64];\nuniform vec4 uGlassEmission[64];\n"
        "#define GX_ARGS\n#define GX_PASS\n#define GX_QUERY\n#define GX_OPAQUE\n"
        + surfacesource::literal<String>(imageSecondary ? "#define GX_IMAGE\n" : "") +
        "#define GLASS_LIGHT_COUNT uLightCount\n#define GLASS_LIGHT_KIND(i) uLightKinds[i]\n#define GLASS_ROTATE glassRotateQ\n"
        "vec3 glassRotateQ(vec4 q,vec3 v) {return v+2.0*cross(q.xyz,cross(q.xyz,v)+q.w*v);}\n" + source;
}
inline std::string transportShader(bool metal,bool imageSecondary=false,bool selectedPrimary=false)
{ return transportShaderRecipe<std::string>(metal,imageSecondary,selectedPrimary); }
// Heitz et al., SIGGRAPH 2016, equations 27/30 and algorithms 3/7, with
// conditional continuation from section 9. Uniform symmetric height CDF q.
// https://jo.dreggn.org/home/2016_microfacets.pdf
// Spherical-cap VNDF for either incident hemisphere, Dupuy/Benyoub 2023.
// https://cdrdv2-public.intel.com/782052/sampling-visible-ggx-normals.pdf
// Original implementation of the published mathematical model. No renderer
// implementation is incorporated. Existing transportShader bytes are intact.
template<class String> inline String multipleScatterShaderRecipe()
{
    return R"GLSL(
#define GLASS_MS_VERSION 2
#define GLASS_MS_EVENTS 32
#define GLASS_MS_EXITS 16
uint glassMsChildHistory(uint parent,uint order,bool transmitted) {
    return ((parent^order)*16777619u^uint(transmitted))*16777619u;
}
uint glassMsEntryKey(uint objectId,uint instance,uint triangle,uint quadrature) {
    uint hash=2166136261u;
    hash=(hash^objectId)*16777619u;hash=(hash^instance)*16777619u;
    hash=(hash^triangle)*16777619u;return (hash^quadrature)*16777619u;
}
uint glassMsSeed(uint entryKey,uint sampleIndex,uint depth,uint branchHistory) {
    uint hash=2166136261u;
    hash=(hash^entryKey)*16777619u;hash=(hash^sampleIndex)*16777619u;
    hash=(hash^depth)*16777619u;return (hash^branchHistory)*16777619u;
}
float glassMsRandom(uint seed,uint dimension) {
    uint value=seed+0x9e3779b9u*(dimension+1u);
    value=(value^(value>>16u))*0x85ebca6bu;
    value=(value^(value>>13u))*0xc2b2ae35u;value^=value>>16u;
    return (float(value>>9u)+0.5)/8388608.0;
}
float glassMsLambda(float cosine,float alpha) {
    if(cosine==0.0) return 0.0;
    float square=alpha*alpha*max(0.0,1.0-cosine*cosine);
    float projected=sqrt(cosine*cosine+square);
    return cosine>0.0?square/(2.0*cosine*(projected+cosine)):
        -0.5*(1.0+projected/(-cosine));
}
float glassMsEscape(float q,float cosine,float alpha) {
    if(cosine<=0.0) return 0.0;
    return pow(q,glassMsLambda(cosine,alpha));
}
vec3 glassMsNormal(vec3 incoming,vec3 normal,float alpha,float u,float v) {
    vec3 tangent=normalize(cross(normal,abs(normal.z)<0.999?vec3(0,0,1):vec3(1,0,0)));
    vec3 bitangent=cross(normal,tangent),view=-incoming;
    vec3 stretched=normalize(vec3(alpha*dot(view,tangent),alpha*dot(view,bitangent),dot(view,normal)));
    float z=1.0-u*(1.0+stretched.z),radius=sqrt(max(0.0,1.0-z*z)),phi=6.283185307179586*v;
    vec3 halfVector=normalize(stretched+vec3(radius*cos(phi),radius*sin(phi),z));
    vec3 facet=normalize(vec3(alpha*halfVector.xy,halfVector.z));
    return tangent*facet.x+bitangent*facet.y+normal*facet.z;
}
// directions.w is signed power: positive reflection, negative transmission.
// orders records actual microfacet phase events, not macro interface depth.
// Status 1 meets the residual policy; 0 retains an excessive unfinished tail;
// -1 rejects invalid/numerically unsupported inputs. No partial Color authority.
int glassMsWalk(vec3 incoming,vec3 normal,float fromIor,float toIor,float roughness,
    uint seed,uint eventLimit,uint exitLimit,float maximumResidual,
    out vec4 directions[GLASS_MS_EXITS],out uint orders[GLASS_MS_EXITS],
    out uint count,out uint events,out uint draws,out float residual) {
    count=0u;events=0u;draws=0u;residual=1.0;
    for(int i=0;i<GLASS_MS_EXITS;++i) {directions[i]=vec4(0);orders[i]=0u;}
    if(!glassSceneFinite3(incoming)||!glassSceneFinite3(normal)
        ||abs(dot(incoming,incoming)-1.0)>1e-4||abs(dot(normal,normal)-1.0)>1e-4
        ||dot(incoming,normal)>=0.0||!glassSceneFinite(roughness)||roughness<=0.0||roughness>1.0
        ||!glassSceneFinite(fromIor)||!glassSceneFinite(toIor)||fromIor<1.0||toIor<1.0||fromIor>3.0||toIor>3.0
        ||eventLimit==0u||eventLimit>32u||exitLimit==0u||exitLimit>16u
        ||!glassSceneFinite(maximumResidual)||maximumResidual<=0.0||maximumResidual>1e-4) return -1;
    if(fromIor==toIor) {directions[0]=vec4(incoming,-1.0);count=1u;residual=0.0;return 1;}
    float alpha=roughness*roughness;if(alpha<=0.0) return -1;
    float q=1.0,weight=1.0;vec3 ray=incoming;bool crossed=false;
    for(uint event=0u;event<eventLimit;++event) {
        vec3 side=crossed?-normal:normal;float cosine=dot(ray,side);
        float escape=glassMsEscape(q,cosine,alpha);
        float u=glassMsRandom(seed,event*4u);++draws;
        float lambda=glassMsLambda(cosine,alpha);
        if(cosine!=0.0) {
            if(lambda==0.0) return -1;
            q=exp(log(q)-log(1.0-u*(1.0-escape))/lambda);
        }
        if(!glassSceneFinite(q)||q<=0.0||q>=1.0) return -1;
        vec3 facet=glassMsNormal(ray,side,alpha,glassMsRandom(seed,event*4u+1u),glassMsRandom(seed,event*4u+2u));
        draws+=2u;
        vec3 transmitted;bool tir;
        if(!glassSceneFinite3(facet)||dot(-ray,facet)<=0.0) return -1;
        ++events;
        float fresnel=glassFresnel(ray,facet,crossed?toIor:fromIor,crossed?fromIor:toIor,transmitted,tir);
        float choice=glassMsRandom(seed,event*4u+3u);++draws;
        if(tir||choice<fresnel) ray=normalize(reflect(ray,facet));
        else {ray=transmitted;crossed=!crossed;q=1.0-q;}
        if(!glassSceneFinite(fresnel)||!glassSceneFinite3(ray)||q<=0.0||q>=1.0) return -1;
        escape=glassMsEscape(q,dot(ray,crossed?-normal:normal),alpha);
        float emitted=weight*escape;
        if(emitted>0.0) {
            if(count>=exitLimit) {residual=weight;return 0;}
            directions[count]=vec4(ray,crossed?-emitted:emitted);orders[count]=events;++count;
        }
        weight*=1.0-escape;residual=weight;
        if(weight<=maximumResidual) return 1;
    }
    return 0;
}
)GLSL";
}
inline std::string multipleScatterShader() { return multipleScatterShaderRecipe<std::string>(); }
} // namespace arbitgpu::smoothglass
