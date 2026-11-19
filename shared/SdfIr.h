#pragma once

#include "SdfTextOutlineContract.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace videowire
{
using SdfStableId = std::uint64_t;

inline constexpr std::uint32_t kSdfIrSchemaVersion = 1; // legacy/default, unchanged
inline constexpr std::uint32_t kSdfIrTextSchemaVersion = 2;
inline constexpr std::size_t kSdfMaximumInputsPerRecord = 2;
inline constexpr std::size_t kSdfMaximumParametersPerRecord = 8;

// Values and distance semantics are part of the backend-neutral wire contract.
// Add new operations at the end and increment the IR schema before changing an
// existing operation or any formula below.
//
// Coordinate and distance convention
// ----------------------------------
// Coordinates are right-handed scene units: +X right, +Y up, and +Z toward a
// camera placed on +Z. Primitive axes are stated below. A returned distance is
// negative inside, zero on the surface, and positive outside. Plane normals and
// rotation axes are normalized before use. Angles are radians. Axis selectors
// are integral 0 = X, 1 = Y, 2 = Z. For axis-oriented domain operations,
// u=(axis+1)%3 and v=(axis+2)%3. Binary inputs are ordered A, B.
//
// Let p be the query point, length(v) the Euclidean norm, abs/max/min component
// wise where a vector is involved, clamp(x) clamp to [0, 1], and R(axis, angle)
// Rodrigues rotation. These formulas are normative:
//
// sphere(r): length(p) - r.
// box(b): q=abs(p)-b; length(max(q,0))+min(max(q.x,q.y,q.z),0).
// roundedBox(b,r): box distance using b-r, then subtract r.
// plane(n,h): dot(p,normalize(n))+h.
// torus(R,r): length(vec2(length(p.xz)-R,p.y))-r. Axis is Y.
// capsule(a,b,r): distance to segment a..b minus r.
// cylinder(r,h): capped Y-axis cylinder with radius r and half-height h.
// cone(r,h): exact capped Y-axis cone with apex at +h and base at -h.
// gyroid(s,t): abs(dot(sin(s*p),cos((s*p).zxy)))/s-t. The historical
//   cellScale name s is reciprocal cell scale in radians per scene unit, so a
//   cell edge is 2*pi/s. It is not a world-space cell length.
// union(A,B)=min(A,B); intersection(A,B)=max(A,B);
// subtraction(A,B)=max(A,-B), meaning A minus B.
// smoothUnion(A,B,k): h=clamp(0.5+0.5*(B-A)/k),
//   mix(B,A,h)-k*h*(1-h). Smooth intersection is -smoothUnion(-A,-B,k).
//   Smooth subtraction is smoothIntersection(A,-B,k).
// translate(o): evaluate the child at p-o.
// rotate(axis,a): evaluate the child at R(axis,-a)*p.
// scale(s): evaluate at p/s and multiply by min(abs(s)); negative components
//   mirror that axis. This is the conservative non-uniform distance estimate.
// repeat(period): evaluate at mod(p+period/2,period)-period/2, where mod uses
//   floor and therefore has a defined half-open boundary.
// polarRepeat(axis,count,offset): fold atan2(v,u)-offset into the centered
//   angular sector 2*pi/count in the plane perpendicular to axis, then evaluate
//   at (radius*cos(folded),radius*sin(folded)) in that plane.
// mirror(flags): replace each selected coordinate by its absolute value.
// twist(axis,a): in the plane perpendicular to axis, rotate the query by
//   -a times its coordinate on that axis.
// bend(axis,a): with u=(axis+1)%3, rotate query coordinates (u,axis) by
//   -a times the original axis coordinate; the remaining coordinate is fixed.
// taper(axis,a): multiply both perpendicular query coordinates by
//   1-a*p[axis], then evaluate the child.
// displacement(a,f): child(p)+a*sin(f*p.x)*sin(f*p.y)*sin(f*p.z).
// textExtrusion(resource,depth): d2=non-zero-winding signed XY contour distance;
//   dz=abs(p.z)-depth/2; hypot(max(d2,0),max(dz,0))+min(max(d2,dz),0).
//   The immutable resource includes exact embedded-font and
//   shaping identities. See SdfTextOutlineContract.h for contour/refusal rules.
// domainWarp(a,f): evaluate the child at q where
//   q.x=p.x+a.x*sin(f.x*p.y), q.y=p.y+a.y*sin(f.y*p.z), and
//   q.z=p.z+a.z*sin(f.z*p.x).
enum class SdfOperation : std::uint8_t
{
    sphere = 1,
    box = 2,
    roundedBox = 3,
    plane = 4,
    torus = 5,
    capsule = 6,
    cylinder = 7,
    cone = 8,
    gyroid = 9,

    unionOp = 10,
    intersection = 11,
    subtraction = 12,
    smoothUnion = 13,
    smoothIntersection = 14,
    smoothSubtraction = 15,

    translate = 16,
    rotate = 17,
    scale = 18,
    repeat = 19,
    polarRepeat = 20,
    mirror = 21,
    twist = 22,
    bend = 23,
    taper = 24,
    displacement = 25,
    domainWarp = 26,
    textExtrusion = 27
};

struct SdfOperationSchema final
{
    std::uint8_t inputCount = 0;
    std::uint8_t parameterCount = 0;
    bool valid = false;
};

constexpr SdfOperationSchema sdfOperationSchema (SdfOperation operation,
    std::uint32_t schemaVersion = kSdfIrSchemaVersion) noexcept
{
    switch (operation)
    {
        // Existing consumers use the v1 default and refuse the new operation.
        case SdfOperation::textExtrusion:
            return schemaVersion == kSdfIrTextSchemaVersion ? SdfOperationSchema { 0, 1, true }
                                                           : SdfOperationSchema {}; // full depth
        case SdfOperation::sphere:             return { 0, 1, true }; // radius
        case SdfOperation::box:                return { 0, 3, true }; // half extents xyz
        case SdfOperation::roundedBox:         return { 0, 4, true }; // half extents xyz, radius
        case SdfOperation::plane:              return { 0, 4, true }; // normal xyz, offset
        case SdfOperation::torus:              return { 0, 2, true }; // major radius, minor radius
        case SdfOperation::capsule:            return { 0, 7, true }; // endpoints a/b xyz, radius
        case SdfOperation::cylinder:           return { 0, 2, true }; // radius, half height
        case SdfOperation::cone:               return { 0, 2, true }; // base radius, half height
        case SdfOperation::gyroid:             return { 0, 2, true }; // reciprocal cell scale, thickness
        case SdfOperation::unionOp:
        case SdfOperation::intersection:
        case SdfOperation::subtraction:        return { 2, 0, true };
        case SdfOperation::smoothUnion:
        case SdfOperation::smoothIntersection:
        case SdfOperation::smoothSubtraction:  return { 2, 1, true }; // smoothing radius
        case SdfOperation::translate:          return { 1, 3, true }; // offset xyz
        case SdfOperation::rotate:             return { 1, 4, true }; // axis xyz, radians
        case SdfOperation::scale:              return { 1, 3, true }; // scale xyz
        case SdfOperation::repeat:             return { 1, 3, true }; // period xyz
        case SdfOperation::polarRepeat:        return { 1, 3, true }; // axis 0..2, count, radians offset
        case SdfOperation::mirror:             return { 1, 3, true }; // mirror x/y/z as 0 or 1
        case SdfOperation::twist:
        case SdfOperation::bend:
        case SdfOperation::taper:              return { 1, 2, true }; // axis 0..2, amount per unit
        case SdfOperation::displacement:       return { 1, 2, true }; // amplitude, frequency
        case SdfOperation::domainWarp:         return { 1, 6, true }; // amplitude xyz, frequency xyz
    }
    return {};
}

constexpr const char* sdfOperationWireToken (SdfOperation operation) noexcept
{
    switch (operation)
    {
        case SdfOperation::textExtrusion: return "textExtrusion";
        case SdfOperation::sphere: return "sphere";
        case SdfOperation::box: return "box";
        case SdfOperation::roundedBox: return "roundedBox";
        case SdfOperation::plane: return "plane";
        case SdfOperation::torus: return "torus";
        case SdfOperation::capsule: return "capsule";
        case SdfOperation::cylinder: return "cylinder";
        case SdfOperation::cone: return "cone";
        case SdfOperation::gyroid: return "gyroid";
        case SdfOperation::unionOp: return "union";
        case SdfOperation::intersection: return "intersection";
        case SdfOperation::subtraction: return "subtraction";
        case SdfOperation::smoothUnion: return "smoothUnion";
        case SdfOperation::smoothIntersection: return "smoothIntersection";
        case SdfOperation::smoothSubtraction: return "smoothSubtraction";
        case SdfOperation::translate: return "translate";
        case SdfOperation::rotate: return "rotate";
        case SdfOperation::scale: return "scale";
        case SdfOperation::repeat: return "repeat";
        case SdfOperation::polarRepeat: return "polarRepeat";
        case SdfOperation::mirror: return "mirror";
        case SdfOperation::twist: return "twist";
        case SdfOperation::bend: return "bend";
        case SdfOperation::taper: return "taper";
        case SdfOperation::displacement: return "displacement";
        case SdfOperation::domainWarp: return "domainWarp";
    }
    return "invalid";
}

struct SdfRecord final
{
    SdfStableId stableId = 0;
    SdfOperation operation = SdfOperation::sphere;
    std::uint8_t inputCount = 0;
    std::array<SdfStableId, kSdfMaximumInputsPerRecord> inputs {};
    std::uint8_t parameterCount = 0;
    std::array<double, kSdfMaximumParametersPerRecord> parameters {};
    SdfTextResourceId resourceStableId = 0; // only textExtrusion; zero in every v1 record
};

struct SdfIr final
{
    std::uint32_t schemaVersion = kSdfIrSchemaVersion;
    SdfStableId rootId = 0;
    std::vector<SdfRecord> records;
    std::vector<SdfTextOutline> textOutlines; // immutable after resource admission
};

// This validates the additive resource envelope, not legacy geometry semantics
// or a native backend capability. Native v1 admission still refuses schema v2.
inline bool validateSdfTextIrResources (const SdfIr& ir, std::string& error)
{
    error.clear();
    const auto fail = [&] (const char* reason) { error = reason; return false; };
    if (ir.schemaVersion != kSdfIrTextSchemaVersion || ir.textOutlines.empty() || ir.records.empty())
        return fail ("SDF text IR version or resources are invalid");
    std::size_t payloadBytes = 0, segmentCount = 0, utf8Bytes = 0, glyphCount = 0;
    SdfTextResourceId previous = 0;
    for (const auto& resource : ir.textOutlines)
    {
        if (! validateSdfTextOutline (resource, error)) return false;
        if (resource.stableId <= previous)
            return fail ("SDF text resources are not in unique canonical stable-ID order");
        previous = resource.stableId;
        const auto bytes = sdfTextOutlinePayloadBytes (resource);
        if (bytes > kSdfTextMaximumPayloadBytes - payloadBytes
            || resource.segments.size() > kSdfTextMaximumSegments - segmentCount
            || resource.sourceText.size() > kSdfTextMaximumUtf8Bytes - utf8Bytes
            || resource.glyphs.size() > kSdfTextMaximumGlyphs - glyphCount)
            return fail ("SDF text graph source, glyph, payload or segment capacity exceeded");
        payloadBytes += bytes;
        segmentCount += resource.segments.size();
        utf8Bytes += resource.sourceText.size();
        glyphCount += resource.glyphs.size();
    }
    std::vector<bool> used (ir.textOutlines.size(), false);
    for (std::size_t index = 0; index < ir.records.size(); ++index)
    {
        const auto& record = ir.records[index];
        const auto schema = sdfOperationSchema (record.operation, ir.schemaVersion);
        if (record.stableId == 0 || record.stableId > kSdfTextMaximumStableId
            || ! schema.valid || record.inputCount != schema.inputCount
            || record.parameterCount != schema.parameterCount)
            return fail ("SDF text IR record identity or shape is invalid");
        for (std::size_t other = 0; other < index; ++other)
            if (ir.records[other].stableId == record.stableId)
                return fail ("SDF text IR record stable ID is duplicated");
        for (std::size_t input = record.inputCount; input < record.inputs.size(); ++input)
            if (record.inputs[input] != 0) return fail ("SDF text IR has undeclared inputs");
        for (std::size_t parameter = 0; parameter < record.parameters.size(); ++parameter)
            if (! std::isfinite (record.parameters[parameter])
                || std::abs (record.parameters[parameter]) > kSdfTextMaximumCoordinate
                || (parameter >= record.parameterCount && record.parameters[parameter] != 0.0))
                return fail ("SDF text IR has non-finite or undeclared parameters");
        if (record.operation != SdfOperation::textExtrusion)
        {
            if (record.resourceStableId != 0) return fail ("Non-text SDF record owns a text resource");
            continue;
        }
        if (record.parameters[0] < kSdfTextMinimumDepth)
            return fail ("SDF text extrusion depth is invalid");
        const auto found = std::lower_bound (ir.textOutlines.begin(), ir.textOutlines.end(), record.resourceStableId,
            [] (const SdfTextOutline& resource, SdfTextResourceId id) { return resource.stableId < id; });
        if (found == ir.textOutlines.end() || found->stableId != record.resourceStableId)
            return fail ("SDF text record references a missing outline resource");
        used[static_cast<std::size_t> (found - ir.textOutlines.begin())] = true;
    }
    if (std::find (used.begin(), used.end(), false) != used.end())
        return fail ("SDF text IR has an unreferenced outline resource");
    return true;
}

// Worst-case recursive evaluation visits, including repeated DAG child edges.
// A retained sampler passes all field/gradient queries. Raymarch passes all
// step/normal/shadow/AO queries. Consumers must admit this before dispatch and
// retain the same record/segment limits in their GPU resource/work receipts.
inline bool chargeSdfTextIrEvaluationWork (const SdfIr& ir, std::uint64_t queryCount,
    SdfTextWorkBudget& budget, std::string& error)
{
    if (! validateSdfTextIrResources (ir, error)) return false;
    const auto fail = [&] (const char* reason) { error = reason; return false; };
    struct Cost { std::uint64_t records = 0, segments = 0; std::size_t depth = 1; };
    std::vector<Cost> costs (ir.records.size());
    std::vector<std::uint8_t> state (ir.records.size(), 0);
    std::vector<bool> reached (ir.records.size(), false);
    const auto indexFor = [&] (SdfStableId id)
    {
        for (std::size_t i = 0; i < ir.records.size(); ++i)
            if (ir.records[i].stableId == id) return i;
        return ir.records.size();
    };
    std::function<bool (std::size_t, std::size_t)> visit = [&] (std::size_t index, std::size_t depth)
    {
        if (index >= ir.records.size() || depth > 64 || state[index] == 1) return false;
        reached[index] = true;
        if (state[index] == 2) return costs[index].depth <= 65 - depth;
        state[index] = 1;
        Cost cost { 1, 0 };
        const auto& record = ir.records[index];
        if (record.operation == SdfOperation::textExtrusion)
            for (const auto& resource : ir.textOutlines)
                if (resource.stableId == record.resourceStableId) cost.segments = resource.segments.size();
        for (std::size_t input = 0; input < record.inputCount; ++input)
        {
            const auto child = indexFor (record.inputs[input]);
            if (! visit (child, depth + 1)) return false;
            cost.records = sdfSaturatingAdd (cost.records, costs[child].records);
            cost.segments = sdfSaturatingAdd (cost.segments, costs[child].segments);
            cost.depth = std::max (cost.depth, costs[child].depth + 1);
        }
        if (cost.records > budget.maximumRecordVisits || cost.segments > budget.maximumSegmentVisits
            || cost.depth > 65 - depth) return false;
        costs[index] = cost;
        state[index] = 2;
        return true;
    };
    const auto root = indexFor (ir.rootId);
    if (! visit (root, 1) || std::find (reached.begin(), reached.end(), false) != reached.end())
        return fail ("SDF text evaluation graph or visit budget is invalid");
    const auto& cost = costs[root];
    if (! budget.charge (sdfSaturatingMultiply (queryCount, cost.records),
                         sdfSaturatingMultiply (queryCount, cost.segments)))
        return fail ("SDF text evaluation work capacity exceeded");
    error.clear();
    return true;
}
} // namespace videowire
