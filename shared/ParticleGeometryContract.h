#pragma once

#include "GeometryCore.h"

namespace videowire::geometry {
// Points, curves and instances size from the plan: particle collider lists grow
// to match, so only a partial triangle is malformed here.
inline bool particleGeometryIntermediatesFit(const MaterializedInstancePlan& plan, std::string& error)
{
    for (const auto& mesh : plan.geometries)
        if (mesh.second.indices.size()%3!=0)
        { error="Particle geometry intermediates need whole triangles"; return false; }
    return true;
}
inline bool particleGeometryRequiresHistory(const ValueDescriptor& value)
{
    return hasRuntimeFields(value) || !value.spectrumFields.empty() || !value.scoreFields.empty()
        || std::any_of(value.operations.begin(),value.operations.end(),[](const auto& operation)
            { return operation.retainedMesh && !operation.retainedMesh->importedAnimation.empty(); });
}
// Descriptor eligibility only. The helper still admits the exact decoded token,
// derives its resource recipe and issues completed samples for replay ticks.
#if defined(__GNUC__)
#pragma GCC diagnostic push
// These switches name the operation codes they accept and reject the rest in default.
#pragma GCC diagnostic ignored "-Wswitch-enum"
#endif
inline bool particleBodyGeometrySupportsTimelineReplay(const ValueDescriptor& value)
{
    const auto* mesh=std::get_if<GeometryData>(&value.data);
    if (value.carrier!=CarrierKind::geometry3D || !mesh || mesh->positions.empty()
        || mesh->indices.empty() || mesh->indices.size()%3!=0
        || value.operations.empty() || !value.attributes.empty()
        || !value.spectrumFields.empty() || !value.scoreFields.empty()) return false;
    unsigned roots=0, timelines=0;
    for (const auto& operation : value.operations)
    {
        if (operation.sdfSurface || !operation.field.attributeName.empty()
            || (operation.retainedMesh && (!operation.retainedMesh->importedAnimation.empty()
                || !operation.retainedMesh->attributes.empty()))) return false;
        switch (operation.code)
        {
        case OperationCode::retainedMesh:
            if (!operation.retainedMesh) return false;
            ++roots;
            break;
        case OperationCode::grid:
            ++roots;
            break;
        case OperationCode::transformTRS:
        case OperationCode::positionField:
        case OperationCode::vectorDisplacement:
            break;
        case OperationCode::fieldVectorMeasure:
        case OperationCode::fieldTimeline:
        case OperationCode::fieldComposeVector:
            if (operation.secondaryInputStableId!=0 || operation.field.auxiliaryInputStableId!=0
                || operation.field.domain!=Domain::vertex
                || (operation.code==OperationCode::fieldVectorMeasure
                    && (operation.field.mode<3 || operation.field.mode>5))
                || (operation.code==OperationCode::fieldComposeVector && operation.field.mode!=1)) return false;
            if (operation.code==OperationCode::fieldTimeline) ++timelines;
            break;
        default:
            return false;
        }
    }
    return roots==1 && timelines!=0;
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
// Meshes size from their source; replay grows its collider lists to match.
inline bool validateParticleGeometry(const ValueDescriptor& value, CarrierKind carrier, std::string& error,
                                     bool validateIntermediates=true)
{
    if (value.carrier != carrier)
    { error = "Particle geometry requires its exact carrier"; return false; }
    for (const auto& binding : value.spectrumFields)
        if ((binding.history.enabled && binding.history.live)
            || binding.target == spectrum::Target::instanceTransform)
        { error = "Particle geometry requires baked spectrum history and does not support Spectrum Instancer transforms"; return false; }
    const auto meshFits = [](const GeometryData& mesh)
    { return !mesh.positions.empty() && mesh.indices.size() % 3 == 0; };
    if (carrier == CarrierKind::points3D)
    {
        const auto& points = std::get<PointsData>(value.data).points;
        if (points.empty())
        { error = "Particle geometry requires at least one stable point"; return false; }
    }
    else if (carrier != CarrierKind::geometry3D || !meshFits(std::get<GeometryData>(value.data)))
    { error = "Particle mesh barriers need vertices and whole triangles"; return false; }
    for (const auto& operation : value.operations)
        if (operation.retainedMesh && !meshFits(operation.retainedMesh->geometry))
        { error = "Particle imported geometry needs vertices and whole triangles"; return false; }
    if (validateIntermediates)
    {
        MaterializedInstancePlan plan;
        if (!validateOperationPlan(value,{},&plan,error) || !particleGeometryIntermediatesFit(plan,error)) return false;
    }
    return true;
}
}
