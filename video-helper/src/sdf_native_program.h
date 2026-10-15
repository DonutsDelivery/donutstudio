#pragma once

#include "gpu_backend/backend.h"
#include "sdf_ir_admission.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace videohelper::sdf
{
// Material ID is an RGB diagnostic on the existing Image port, not a raw ID
// attachment. Hash the full primitive stable ID, never a draw/record index.
// The three low bytes are R,G,B; the fixed bits keep black reserved for misses.
inline std::uint32_t nativeSdfMaterialColorCode (videowire::SdfStableId id) noexcept
{
    id ^= id >> 30u;
    id *= UINT64_C(0xbf58476d1ce4e5b9);
    id ^= id >> 27u;
    id *= UINT64_C(0x94d049bb133111eb);
    id ^= id >> 31u;
    return (static_cast<std::uint32_t> (id) & 0x00ffffffu) | 0x00202020u;
}

inline std::vector<const videowire::SdfRecord*> nativeSdfTopologicalRecords (
    const AdmittedSdfIr& geometry)
{
    std::unordered_map<videowire::SdfStableId, const videowire::SdfRecord*> byId;
    for (const auto& record : geometry.records())
        byId.emplace (record.stableId, &record);
    std::unordered_set<videowire::SdfStableId> visited;
    std::vector<const videowire::SdfRecord*> ordered;
    std::function<void (videowire::SdfStableId)> visit = [&] (auto id)
    {
        if (! visited.insert (id).second) return;
        const auto* record = byId.at (id);
        for (std::size_t input = 0; input < record->inputCount; ++input)
            visit (record->inputs[input]);
        ordered.push_back (record);
    };
    visit (geometry.rootId());
    return ordered;
}

// Stable median splits retain every authored edge once. Boxes enclose packed
// coefficients, not a substitute outline. GPU pruning additionally needs an
// arithmetic lower-bound certificate; a typical visit count is not a bound.
inline void nativeSdfBuildTextIndex (
    const std::vector<arbitgpu::NativeSdfTextSegment>& segments,
    const std::vector<arbitgpu::NativeSdfTextResourceHeader>& resources,
    std::vector<arbitgpu::NativeSdfTextIndexNode>& nodes,
    std::vector<std::uint32_t>& segmentIndices, std::vector<std::uint32_t>& roots)
{
    nodes.clear(); segmentIndices.clear(); roots.clear();
    segmentIndices.reserve (segments.size());
    nodes.reserve (segments.size() * 2);
    roots.reserve (resources.size());
    for (const auto& resource : resources)
    {
        const auto first = static_cast<std::uint32_t> (segmentIndices.size());
        for (std::uint32_t i = 0; i < resource.segmentCount; ++i)
            segmentIndices.push_back (resource.firstSegment + i);
        std::function<void (std::uint32_t, std::uint32_t)> build = [&] (auto begin, auto count)
        {
            const auto nodeIndex = static_cast<std::uint32_t> (nodes.size());
            const auto& firstEdge = segments[segmentIndices[begin]];
            arbitgpu::NativeSdfTextIndexNode node;
            node.minX = std::min (firstEdge.x0, firstEdge.x1);
            node.minY = std::min (firstEdge.y0, firstEdge.y1);
            node.maxX = std::max (firstEdge.x0, firstEdge.x1);
            node.maxY = std::max (firstEdge.y0, firstEdge.y1);
            for (std::uint32_t i = begin + 1; i < begin + count; ++i)
            {
                const auto& edge = segments[segmentIndices[i]];
                node.minX = std::min ({ node.minX, edge.x0, edge.x1 });
                node.minY = std::min ({ node.minY, edge.y0, edge.y1 });
                node.maxX = std::max ({ node.maxX, edge.x0, edge.x1 });
                node.maxY = std::max ({ node.maxY, edge.y0, edge.y1 });
            }
            const bool splitX = double (node.maxX) - node.minX >= double (node.maxY) - node.minY;
            node.minX = std::nextafter (node.minX, -std::numeric_limits<float>::infinity());
            node.minY = std::nextafter (node.minY, -std::numeric_limits<float>::infinity());
            node.maxX = std::nextafter (node.maxX, std::numeric_limits<float>::infinity());
            node.maxY = std::nextafter (node.maxY, std::numeric_limits<float>::infinity());
            nodes.push_back (node);
            if (count <= arbitgpu::kNativeSdfTextIndexLeafSegments)
            {
                nodes[nodeIndex].firstIndex = begin;
                nodes[nodeIndex].indexCount = count;
            }
            else
            {
                std::sort (segmentIndices.begin() + begin, segmentIndices.begin() + begin + count,
                    [&] (auto a, auto b)
                    {
                        const auto& x = segments[a]; const auto& y = segments[b];
                        const auto cx = splitX ? double (x.x0) + x.x1 : double (x.y0) + x.y1;
                        const auto cy = splitX ? double (y.x0) + y.x1 : double (y.y0) + y.y1;
                        return cx == cy ? a < b : cx < cy;
                    });
                build (begin, count / 2);
                build (begin + count / 2, count - count / 2);
            }
            nodes[nodeIndex].escape = static_cast<std::uint32_t> (nodes.size());
        };
        roots.push_back (static_cast<std::uint32_t> (nodes.size()));
        build (first, resource.segmentCount);
    }
}

inline std::optional<arbitgpu::NativeSdfSceneBounds> nativeSdfSceneBounds (
    const arbitgpu::NativeSdfCompiledProgram& program)
{
    using Bounds = arbitgpu::NativeSdfSceneBounds;
    std::vector<std::optional<Bounds>> bounds (program.records().size());
    for (std::size_t i = 0; i < bounds.size(); ++i)
    {
        const auto& record = program.records()[i];
        const auto& p = record.parameters;
        const auto operation = static_cast<videowire::SdfOperation> (record.operation);
        auto& result = bounds[i];
        const auto symmetric = [&] (double x, double y, double z)
        { result = Bounds { { -x, -y, -z }, { x, y, z } }; };
        switch (operation)
        {
            case videowire::SdfOperation::sphere: symmetric (p[0], p[0], p[0]); break;
            case videowire::SdfOperation::box:
            case videowire::SdfOperation::roundedBox: symmetric (p[0], p[1], p[2]); break;
            case videowire::SdfOperation::torus:
                symmetric (double (p[0]) + p[1], p[1], double (p[0]) + p[1]); break;
            case videowire::SdfOperation::cylinder:
            case videowire::SdfOperation::cone: symmetric (p[0], p[1], p[0]); break;
            case videowire::SdfOperation::capsule:
                result = Bounds {};
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    result->minimum[axis] = double (std::min (p[axis], p[axis + 3])) - p[6];
                    result->maximum[axis] = double (std::max (p[axis], p[axis + 3])) + p[6];
                }
                break;
            case videowire::SdfOperation::textExtrusion:
            {
                const auto& node = program.textIndexNodes()[program.textIndexRoots()[program.textResourceIndices()[i]]];
                result = Bounds { { node.minX, node.minY, -double (p[0]) / 2 },
                                  { node.maxX, node.maxY, double (p[0]) / 2 } };
                break;
            }
            case videowire::SdfOperation::unionOp:
                if (bounds[record.input0] && bounds[record.input1])
                {
                    result = *bounds[record.input0];
                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        result->minimum[axis] = std::min (result->minimum[axis], bounds[record.input1]->minimum[axis]);
                        result->maximum[axis] = std::max (result->maximum[axis], bounds[record.input1]->maximum[axis]);
                    }
                }
                break;
            case videowire::SdfOperation::intersection:
                // Either finite operand encloses the intersection, including
                // an empty intersection. Avoid inventing a finite empty box.
                result = bounds[record.input0] ? bounds[record.input0] : bounds[record.input1];
                break;
            case videowire::SdfOperation::subtraction: result = bounds[record.input0]; break;
            case videowire::SdfOperation::translate:
            case videowire::SdfOperation::scale:
                if (bounds[record.input0]
                    && (operation != videowire::SdfOperation::scale
                        || (std::abs (p[0]) == std::abs (p[1]) && std::abs (p[1]) == std::abs (p[2]))))
                {
                    result = *bounds[record.input0];
                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        if (operation == videowire::SdfOperation::translate)
                        {
                            result->minimum[axis] += p[axis]; result->maximum[axis] += p[axis];
                        }
                        else
                        {
                            const auto a = result->minimum[axis] * p[axis], b = result->maximum[axis] * p[axis];
                            result->minimum[axis] = std::min (a, b); result->maximum[axis] = std::max (a, b);
                        }
                    }
                }
                break;
            default: break;
        }
        if (result)
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                result->minimum[axis] = std::nextafter (result->minimum[axis], -std::numeric_limits<double>::infinity());
                result->maximum[axis] = std::nextafter (result->maximum[axis], std::numeric_limits<double>::infinity());
                if (! std::isfinite (result->minimum[axis]) || ! std::isfinite (result->maximum[axis]))
                { result.reset(); break; }
            }
    }
    return bounds[program.rootIndex()];
}

inline std::shared_ptr<const arbitgpu::NativeSdfCompiledProgram> compileNativeSdfProgram (
    const videowire::SdfIr& source, std::string& error, bool textTablesAvailable)
{
    SdfAdmissionLimits limits;
    limits.maxOperations = arbitgpu::kNativeSdfMaximumRecords;
    limits.maxDepth = arbitgpu::kNativeSdfMaximumDepth;
    limits.maxStableId = std::numeric_limits<videowire::SdfStableId>::max();
    auto admitted = admitSdfIr (source, limits, error);
    if (! admitted) return {};

    if (! admitted->textOutlines().empty() && ! textTablesAvailable)
    {
        error = "native GPU SDF text table consumer is unavailable";
        return {};
    }
    const auto ordered = nativeSdfTopologicalRecords (*admitted);
    std::shared_ptr<arbitgpu::NativeSdfCompiledProgram> program (
        new arbitgpu::NativeSdfCompiledProgram());
    program->schemaVersion_ = admitted->schemaVersion();
    program->maximumDepth_ = static_cast<std::uint32_t> (admitted->maximumDepth());
    program->structuralDigest_ = admitted->structuralDigest();
    program->records_.reserve (ordered.size());

    std::unordered_map<videowire::SdfTextResourceId, std::uint32_t> resourceIndices;
    for (const auto& admittedOutline : admitted->textOutlines())
    {
        const auto& outline = admittedOutline.outline();
        resourceIndices.emplace (outline.stableId, static_cast<std::uint32_t> (program->textResources_.size()));
        const auto firstSegment = static_cast<std::uint32_t> (program->textSegments_.size());
        program->textResources_.push_back ({
            static_cast<std::uint32_t> (program->textContours_.size()),
            static_cast<std::uint32_t> (outline.contours.size()), firstSegment,
            static_cast<std::uint32_t> (outline.segments.size()) });
        program->textResourceStableIds_.push_back (outline.stableId);
        program->textResourceIdentities_.push_back (admittedOutline.identitySha256());
        for (const auto& segment : outline.segments)
            program->textSegments_.push_back ({ static_cast<float> (segment.x0),
                static_cast<float> (segment.y0), static_cast<float> (segment.x1),
                static_cast<float> (segment.y1) });
        for (const auto& contour : outline.contours)
        {
            program->textContours_.push_back ({ firstSegment + contour.firstSegment,
                contour.segmentCount, contour.winding, contour.glyphIndex });
            // Test the arithmetic used by the shader after float conversion.
            // Accumulate area relative to the first vertex to avoid large-origin
            // cancellation, while preserving the authored orientation.
            const auto& origin = program->textSegments_[firstSegment + contour.firstSegment];
            double twiceArea = 0;
            float nativeTwiceArea = 0;
            for (std::size_t i = 0; i < contour.segmentCount; ++i)
            {
                const auto index = firstSegment + contour.firstSegment + i;
                const auto& segment = program->textSegments_[index];
                const auto& next = program->textSegments_[firstSegment + contour.firstSegment
                    + (i + 1) % contour.segmentCount];
                const auto dx = segment.x1 - segment.x0, dy = segment.y1 - segment.y0;
                const auto squaredLength = dx * dx + dy * dy;
                if (! std::isfinite (segment.x0) || ! std::isfinite (segment.y0)
                    || ! std::isfinite (segment.x1) || ! std::isfinite (segment.y1)
                    || ! std::isfinite (squaredLength) || squaredLength <= 0
                    || segment.x1 != next.x0 || segment.y1 != next.y0)
                {
                    error = "native GPU SDF text contour is open or degenerate after float conversion";
                    return {};
                }
                const auto ax = segment.x0 - origin.x0, ay = segment.y0 - origin.y0;
                const auto bx = segment.x1 - origin.x0, by = segment.y1 - origin.y0;
                nativeTwiceArea += ax * by - bx * ay;
                twiceArea += (static_cast<double> (segment.x0) - origin.x0)
                    * (static_cast<double> (segment.y1) - origin.y0)
                    - (static_cast<double> (segment.x1) - origin.x0)
                    * (static_cast<double> (segment.y0) - origin.y0);
            }
            if (! std::isfinite (twiceArea) || twiceArea == 0
                || ! std::isfinite (nativeTwiceArea) || nativeTwiceArea == 0
                || (nativeTwiceArea > 0 ? 1 : -1) != contour.winding
                || (twiceArea > 0 ? 1 : -1) != contour.winding)
            {
                error = "native GPU SDF text winding collapses or changes after float conversion";
                return {};
            }
        }
    }
    nativeSdfBuildTextIndex (program->textSegments_, program->textResources_,
        program->textIndexNodes_, program->textSegmentIndices_, program->textIndexRoots_);
    if (! program->textResources_.empty())
        program->textResourceIndices_.reserve (ordered.size());

    std::unordered_map<videowire::SdfStableId, std::uint32_t> indices;
    indices.reserve (ordered.size());
    for (std::size_t index = 0; index < ordered.size(); ++index)
        indices.emplace (ordered[index]->stableId, static_cast<std::uint32_t> (index));

    for (const auto* source : ordered)
    {
        const auto& sourceRecord = *source;
        arbitgpu::NativeSdfCompiledRecord record;
        record.stableId = sourceRecord.stableId;
        record.operation = static_cast<std::uint32_t> (sourceRecord.operation);
        record.parameterCount = sourceRecord.parameterCount;
        if (sourceRecord.inputCount > 0) record.input0 = indices.at (sourceRecord.inputs[0]);
        if (sourceRecord.inputCount > 1) record.input1 = indices.at (sourceRecord.inputs[1]);
        for (std::size_t parameter = 0; parameter < sourceRecord.parameterCount; ++parameter)
        {
            const auto value = static_cast<float> (sourceRecord.parameters[parameter]);
            if (! std::isfinite (value))
            {
                error = "native GPU SDF parameter cannot be represented as a finite float";
                return {};
            }
            record.parameters[parameter] = value;
        }
        if (! program->textResources_.empty())
            program->textResourceIndices_.push_back (
                sourceRecord.operation == videowire::SdfOperation::textExtrusion
                    ? resourceIndices.at (sourceRecord.resourceStableId)
                    : std::numeric_limits<std::uint32_t>::max());
        program->records_.push_back (record);
    }

    std::vector<std::size_t> expandedSteps (program->records_.size(), 0);
    std::function<std::size_t (std::size_t)> countSteps = [&] (std::size_t index)
    {
        if (expandedSteps[index] != 0) return expandedSteps[index];
        const auto& record = program->records_[index];
        const auto inputs = videowire::sdfOperationSchema (
            static_cast<videowire::SdfOperation> (record.operation), program->schemaVersion_).inputCount;
        std::size_t steps = 1;
        const auto addChild = [&] (std::uint32_t child)
        {
            const auto childSteps = countSteps (child);
            if (childSteps > arbitgpu::kNativeSdfMaximumEvaluationSteps - steps)
                steps = arbitgpu::kNativeSdfMaximumEvaluationSteps + 1;
            else
                steps += childSteps;
        };
        if (inputs > 0) addChild (record.input0);
        if (inputs > 1 && steps <= arbitgpu::kNativeSdfMaximumEvaluationSteps)
            addChild (record.input1);
        expandedSteps[index] = steps;
        return steps;
    };
    program->rootIndex_ = indices.at (admitted->rootId());
    const auto evaluationSteps = countSteps (program->rootIndex_);
    if (evaluationSteps > arbitgpu::kNativeSdfMaximumEvaluationSteps)
    {
        error = "native GPU SDF expanded evaluation capacity exceeded";
        return {};
    }
    program->evaluationSteps_ = static_cast<std::uint32_t> (evaluationSteps);
    if (! program->textResources_.empty())
    {
        videowire::SdfTextWorkBudget work;
        if (! videowire::chargeSdfTextIrEvaluationWork (source, 1, work, error)) return {};
        program->segmentVisitsPerQuery_ = work.segmentVisits;
    }
    std::vector<std::uint64_t> nodeVisits (program->records_.size(), 0);
    for (std::size_t i = 0; i < nodeVisits.size(); ++i)
    {
        const auto& record = program->records_[i];
        const auto schema = videowire::sdfOperationSchema (
            static_cast<videowire::SdfOperation> (record.operation), program->schemaVersion_);
        if (record.operation == static_cast<std::uint32_t> (videowire::SdfOperation::textExtrusion))
        {
            const auto root = program->textIndexRoots_[program->textResourceIndices_[i]];
            nodeVisits[i] = program->textIndexNodes_[root].escape - root;
        }
        if (schema.inputCount > 0) nodeVisits[i] += nodeVisits[record.input0];
        if (schema.inputCount > 1) nodeVisits[i] += nodeVisits[record.input1];
    }
    program->textIndexNodeVisitsPerQuery_ = nodeVisits[program->rootIndex_];
    program->sceneBounds_ = nativeSdfSceneBounds (*program);
    error.clear();
    return program;
}

inline bool validateNativeSdfProgram (
    const arbitgpu::NativeSdfCompiledProgram& program, std::string& error,
    bool textTablesAvailable = false)
{
    if (! program.textResources().empty() && ! textTablesAvailable)
    {
        error = "native GPU SDF text table consumer is unavailable";
        return false;
    }
    if (program.records().empty() || program.records().size() > arbitgpu::kNativeSdfMaximumRecords
        || program.rootIndex() >= program.records().size()
        || program.maximumDepth() > arbitgpu::kNativeSdfMaximumDepth
        || program.evaluationSteps() == 0
        || program.evaluationSteps() > arbitgpu::kNativeSdfMaximumEvaluationSteps
        || program.structuralDigest().size() != 64
        || (program.schemaVersion() != videowire::kSdfIrSchemaVersion
            && program.schemaVersion() != videowire::kSdfIrTextSchemaVersion)
        || (program.textResources().empty() != program.textSegments().empty())
        || program.textResources().size() != program.textResourceStableIds().size()
        || program.textResources().size() != program.textResourceIdentities().size()
        || (! program.textResources().empty()
            && (program.schemaVersion() != videowire::kSdfIrTextSchemaVersion
                || program.textResourceIndices().size() != program.records().size()
                || program.textContours().empty() || program.segmentVisitsPerQuery() == 0
                || program.textIndexRoots().size() != program.textResources().size()
                || program.textSegmentIndices().size() != program.textSegments().size()
                || program.textIndexNodes().empty() || program.textIndexNodeVisitsPerQuery() == 0))
        || (program.textResources().empty()
            && (! program.textResourceIndices().empty() || ! program.textContours().empty()
                || program.segmentVisitsPerQuery() != 0 || program.textIndexNodeVisitsPerQuery() != 0
                || ! program.textIndexRoots().empty() || ! program.textSegmentIndices().empty()
                || ! program.textIndexNodes().empty())))
    {
        error = "native GPU SDF compiled payload is invalid";
        return false;
    }
    std::size_t nextContour = 0, nextSegment = 0;
    for (std::size_t resourceIndex = 0; resourceIndex < program.textResources().size(); ++resourceIndex)
    {
        const auto& resource = program.textResources()[resourceIndex];
        if (resource.firstContour != nextContour || resource.firstSegment != nextSegment
            || resource.contourCount == 0 || resource.segmentCount == 0
            || resource.contourCount > program.textContours().size() - nextContour
            || resource.segmentCount > program.textSegments().size() - nextSegment
            || program.textResourceStableIds()[resourceIndex] == 0
            || (resourceIndex > 0 && program.textResourceStableIds()[resourceIndex]
                <= program.textResourceStableIds()[resourceIndex - 1])
            || program.textResourceIdentities()[resourceIndex].size() != 64)
        {
            error = "native GPU SDF text resource table is invalid";
            return false;
        }
        const auto endSegment = nextSegment + resource.segmentCount;
        for (std::size_t i = 0; i < resource.contourCount; ++i)
        {
            const auto& contour = program.textContours()[nextContour++];
            if (contour.firstSegment != nextSegment || contour.segmentCount < 3
                || contour.segmentCount > endSegment - nextSegment
                || (contour.winding != 1 && contour.winding != -1))
            {
                error = "native GPU SDF text contour table is invalid";
                return false;
            }
            nextSegment += contour.segmentCount;
        }
        if (nextSegment != endSegment)
        {
            error = "native GPU SDF text resource has unowned segments";
            return false;
        }
    }
    if (nextContour != program.textContours().size() || nextSegment != program.textSegments().size()
        || nextSegment > videowire::kSdfTextMaximumSegments)
    {
        error = "native GPU SDF text table capacity or ownership is invalid";
        return false;
    }
    for (const auto& edge : program.textSegments())
        if (! std::isfinite (edge.x0) || ! std::isfinite (edge.y0)
            || ! std::isfinite (edge.x1) || ! std::isfinite (edge.y1)
            || (edge.x0 == edge.x1 && edge.y0 == edge.y1))
        {
            error = "native GPU SDF text index has an invalid source segment";
            return false;
        }
    std::vector<arbitgpu::NativeSdfTextIndexNode> expectedNodes;
    std::vector<std::uint32_t> expectedIndices, expectedRoots;
    nativeSdfBuildTextIndex (program.textSegments(), program.textResources(),
        expectedNodes, expectedIndices, expectedRoots);
    bool sameIndex = expectedNodes.size() == program.textIndexNodes().size()
        && expectedIndices == program.textSegmentIndices() && expectedRoots == program.textIndexRoots();
    if (sameIndex)
        for (std::size_t i = 0; i < expectedNodes.size(); ++i)
        {
            const auto& a = expectedNodes[i]; const auto& b = program.textIndexNodes()[i];
            sameIndex = a.minX == b.minX && a.minY == b.minY && a.maxX == b.maxX && a.maxY == b.maxY
                && a.firstIndex == b.firstIndex && a.indexCount == b.indexCount
                && a.escape == b.escape && b.reserved == 0;
            if (! sameIndex) break;
        }
    if (! sameIndex)
    {
        error = "native GPU SDF derived text index does not match packed segments";
        return false;
    }
    for (std::size_t index = 0; index < program.records().size(); ++index)
    {
        const auto& record = program.records()[index];
        if (record.stableId == 0
            || record.operation < static_cast<std::uint32_t> (videowire::SdfOperation::sphere)
            || record.operation > static_cast<std::uint32_t> (videowire::SdfOperation::textExtrusion)
            || ! videowire::sdfOperationSchema (
                static_cast<videowire::SdfOperation> (record.operation), program.schemaVersion()).valid)
        {
            error = "native GPU SDF compiled payload has an invalid operation";
            return false;
        }
        const auto schema = videowire::sdfOperationSchema (
            static_cast<videowire::SdfOperation> (record.operation), program.schemaVersion());
        if (record.parameterCount != schema.parameterCount
            || (schema.inputCount > 0 && record.input0 >= index)
            || (schema.inputCount > 1 && record.input1 >= index)
            || (schema.inputCount == 0
                && (record.input0 != std::numeric_limits<std::uint32_t>::max()
                    || record.input1 != std::numeric_limits<std::uint32_t>::max()))
            || (schema.inputCount == 1
                && record.input1 != std::numeric_limits<std::uint32_t>::max()))
        {
            error = "native GPU SDF compiled payload has an invalid record";
            return false;
        }
        if (record.operation == static_cast<std::uint32_t> (videowire::SdfOperation::textExtrusion))
        {
            if (! textTablesAvailable || program.textResources().empty()
                || program.textResourceIndices()[index] >= program.textResources().size()
                || record.parameters[0] < static_cast<float> (videowire::kSdfTextMinimumDepth))
            {
                error = "native GPU SDF compiled text reference is invalid or unavailable";
                return false;
            }
        }
        else if (! program.textResourceIndices().empty()
                 && program.textResourceIndices()[index] != std::numeric_limits<std::uint32_t>::max())
        {
            error = "native GPU SDF non-text record owns a text reference";
            return false;
        }
        for (std::size_t parameter = 0; parameter < record.parameterCount; ++parameter)
        {
            if (! std::isfinite (record.parameters[parameter]))
            {
                error = "native GPU SDF compiled payload has a non-finite parameter";
                return false;
            }
        }
        for (std::size_t parameter = record.parameterCount;
             parameter < record.parameters.size(); ++parameter)
            if (record.parameters[parameter] != 0.0f)
            {
                error = "native GPU SDF compiled payload has an active unused parameter";
                return false;
            }
        if (static_cast<videowire::SdfOperation> (record.operation)
                == videowire::SdfOperation::roundedBox
            && record.parameters[3] > std::min ({ record.parameters[0], record.parameters[1],
                                                   record.parameters[2] }))
        {
            error = "native GPU SDF rounded-box radius exceeds the minimum half extent";
            return false;
        }
    }
    std::vector<bool> reachable (program.records().size(), false);
    std::function<void (std::uint32_t)> visit = [&] (std::uint32_t index)
    {
        if (reachable[index]) return;
        reachable[index] = true;
        const auto schema = videowire::sdfOperationSchema (
            static_cast<videowire::SdfOperation> (program.records()[index].operation), program.schemaVersion());
        if (schema.inputCount > 0) visit (program.records()[index].input0);
        if (schema.inputCount > 1) visit (program.records()[index].input1);
    };
    visit (program.rootIndex());
    if (std::find (reachable.begin(), reachable.end(), false) != reachable.end())
    {
        error = "native GPU SDF compiled payload has unreachable records";
        return false;
    }
    std::vector<std::uint64_t> nodeVisits (program.records().size(), 0);
    for (std::size_t i = 0; i < nodeVisits.size(); ++i)
    {
        const auto& record = program.records()[i];
        const auto schema = videowire::sdfOperationSchema (
            static_cast<videowire::SdfOperation> (record.operation), program.schemaVersion());
        if (record.operation == static_cast<std::uint32_t> (videowire::SdfOperation::textExtrusion))
        {
            const auto root = program.textIndexRoots()[program.textResourceIndices()[i]];
            nodeVisits[i] = program.textIndexNodes()[root].escape - root;
        }
        if (schema.inputCount > 0) nodeVisits[i] += nodeVisits[record.input0];
        if (schema.inputCount > 1) nodeVisits[i] += nodeVisits[record.input1];
    }
    const auto expectedBounds = nativeSdfSceneBounds (program);
    if (nodeVisits[program.rootIndex()] != program.textIndexNodeVisitsPerQuery()
        || expectedBounds.has_value() != program.sceneBounds().has_value()
        || (expectedBounds && (expectedBounds->minimum != program.sceneBounds()->minimum
            || expectedBounds->maximum != program.sceneBounds()->maximum)))
    {
        error = "native GPU SDF derived bounds or index work are invalid";
        return false;
    }
    error.clear();
    return true;
}

// The compiled type is factory-only. Re-admit the untrusted source and bind it
// to the factory result by the canonical structural digest before GPU use.
inline bool validateNativeSdfProgram (
    const arbitgpu::NativeSdfCompiledProgram& program,
    const videowire::SdfIr& source,
    std::string& error, bool textTablesAvailable = false)
{
    if (! validateNativeSdfProgram (program, error, textTablesAvailable))
        return false;
    SdfAdmissionLimits limits;
    limits.maxOperations = arbitgpu::kNativeSdfMaximumRecords;
    limits.maxDepth = arbitgpu::kNativeSdfMaximumDepth;
    limits.maxStableId = std::numeric_limits<videowire::SdfStableId>::max();
    const auto admitted = admitSdfIr (source, limits, error);
    if (! admitted
        || program.structuralDigest() != admitted->structuralDigest()
        || program.maximumDepth() != admitted->maximumDepth()
        || program.records().size() != admitted->records().size()
        || program.schemaVersion() != admitted->schemaVersion())
    {
        if (error.empty())
            error = "native GPU SDF compiled payload does not match admitted geometry";
        return false;
    }
    if (! admitted->textOutlines().empty())
    {
        const auto expected = compileNativeSdfProgram (source, error, true);
        if (! expected) return false;
        const auto sameSegments = [&]
        {
            if (program.textSegments().size() != expected->textSegments().size()) return false;
            for (std::size_t i = 0; i < program.textSegments().size(); ++i)
            {
                const auto& a = program.textSegments()[i];
                const auto& b = expected->textSegments()[i];
                if (a.x0 != b.x0 || a.y0 != b.y0 || a.x1 != b.x1 || a.y1 != b.y1) return false;
            }
            return true;
        };
        const auto sameHeaders = [&]
        {
            if (program.textResources().size() != expected->textResources().size()
                || program.textContours().size() != expected->textContours().size()) return false;
            for (std::size_t i = 0; i < program.textResources().size(); ++i)
            {
                const auto& a = program.textResources()[i];
                const auto& b = expected->textResources()[i];
                if (a.firstContour != b.firstContour || a.contourCount != b.contourCount
                    || a.firstSegment != b.firstSegment || a.segmentCount != b.segmentCount) return false;
            }
            for (std::size_t i = 0; i < program.textContours().size(); ++i)
            {
                const auto& a = program.textContours()[i];
                const auto& b = expected->textContours()[i];
                if (a.firstSegment != b.firstSegment || a.segmentCount != b.segmentCount
                    || a.winding != b.winding || a.glyphIndex != b.glyphIndex) return false;
            }
            return true;
        };
        if (! expected || program.textResourceStableIds() != expected->textResourceStableIds()
            || program.textResourceIdentities() != expected->textResourceIdentities()
            || program.textResourceIndices() != expected->textResourceIndices()
            || program.segmentVisitsPerQuery() != expected->segmentVisitsPerQuery()
            || ! sameSegments() || ! sameHeaders())
        {
            error = "native GPU SDF compiled text tables do not match exact outline identity";
            return false;
        }
    }
    const auto ordered = nativeSdfTopologicalRecords (*admitted);
    std::unordered_map<videowire::SdfStableId, std::uint32_t> indices;
    for (std::size_t index = 0; index < ordered.size(); ++index)
        indices.emplace (ordered[index]->stableId, static_cast<std::uint32_t> (index));
    std::vector<std::size_t> steps (program.records().size(), 0);
    std::function<std::size_t (std::uint32_t)> countSteps = [&] (std::uint32_t index)
    {
        if (steps[index] != 0) return steps[index];
        const auto schema = videowire::sdfOperationSchema (
            static_cast<videowire::SdfOperation> (program.records()[index].operation), program.schemaVersion());
        std::size_t result = 1;
        if (schema.inputCount > 0) result += countSteps (program.records()[index].input0);
        if (schema.inputCount > 1) result += countSteps (program.records()[index].input1);
        return steps[index] = result;
    };
    for (std::size_t index = 0; index < ordered.size(); ++index)
    {
        const auto& sourceRecord = *ordered[index];
        const auto& record = program.records()[index];
        if (record.stableId != sourceRecord.stableId
            || record.operation != static_cast<std::uint32_t> (sourceRecord.operation)
            || record.parameterCount != sourceRecord.parameterCount
            || (sourceRecord.inputCount > 0 && record.input0 != indices.at (sourceRecord.inputs[0]))
            || (sourceRecord.inputCount > 1 && record.input1 != indices.at (sourceRecord.inputs[1])))
        {
            error = "native GPU SDF compiled payload records do not match admitted geometry";
            return false;
        }
        for (std::size_t parameter = 0; parameter < sourceRecord.parameterCount; ++parameter)
            if (record.parameters[parameter] != static_cast<float> (sourceRecord.parameters[parameter]))
            {
                error = "native GPU SDF compiled payload values do not match admitted geometry";
                return false;
            }
    }
    if (program.rootIndex() != indices.at (admitted->rootId())
        || program.evaluationSteps() != countSteps (program.rootIndex()))
    {
        error = "native GPU SDF compiled payload metadata does not match admitted geometry";
        return false;
    }
    error.clear();
    return true;
}
} // namespace videohelper::sdf
