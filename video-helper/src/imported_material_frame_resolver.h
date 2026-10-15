#pragma once

#include "visual_plan_executor.h"
#include "imported_animation_visual_plan_execution.h"

namespace videohelper::importedscene
{
// Constructed only by the production composition below. A bare decoder callback
// cannot exempt imported Layer Source from the legacy safe refusal.
class ImportedMaterialFrameResolver
{
    using Image = std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>;
    struct Bundle
    {
        MaterialFrameEvaluation identity;
        std::vector<Image> images;
    };
    struct Staging
    {
        std::shared_ptr<const Bundle> bundle;
        bool submitted = false;
    };
    MaterialFrameResolver resolver_;
    std::shared_ptr<std::optional<visualdeformation::RationalFrameTime>> preparedFrame_
        = std::make_shared<std::optional<visualdeformation::RationalFrameTime>>();
    std::function<bool(std::string&)> active_;
    std::shared_ptr<Staging> staging_ = std::make_shared<Staging>();

public:
    bool resolve(const MaterialFrameEvaluation& query, MaterialFrameReceipt& receipt,
                 std::string& error) const
    { return resolveMaterialFrame(query,resolver_,receipt,error); }
    bool resolvePrepared(const MaterialFrameEvaluation& query,
        const visualdeformation::RationalFrameTime& preparedFrame,
        MaterialFrameReceipt& receipt,std::string& error) const
    {
        // Imported preparation owns playback/automation clock admission. Reuse
        // that exact request frame rather than rebuilding those authorities here.
        *preparedFrame_ = preparedFrame;
        struct Reset { std::optional<visualdeformation::RationalFrameTime>& value;
            ~Reset() { value.reset(); } } reset{*preparedFrame_};
        return resolve(query,receipt,error);
    }
    bool active(std::string& error) const
    { return active_ && active_(error); }
    void submitted() const { staging_->submitted = true; }
    void discard() const { staging_->bundle.reset(); staging_->submitted = false; }

    template <NativeImportedSceneRenderUse Use, typename Renderer, typename Layer,
        typename Segment, typename Decode, typename Lease>
    static ImportedMaterialFrameResolver make(Renderer& renderer,
        const videowire::CompiledVisualLayerPlan& plan, const videowire::VisualLayerExecution& compiled,
        Segment current, std::vector<Segment> segments, double projectSeconds, double sourceSeconds,
        double fps, const Layer& inputs, const std::map<std::string,double>* parameters,
        videowire::VisualPlanEvaluationContext context,
        videowire::TemporalSamplingCommitTransaction& transaction,
        std::function<bool(std::string&)> activity, Decode decode, Lease lease)
    {
        ImportedMaterialFrameResolver result;
        result.active_ = activity;
        const auto staging = result.staging_;
        const auto preparedFrame = result.preparedFrame_;
        // The callable borrows the layer and adapters only until scene preparation
        // returns. Publications retain just the immutable image bundle and activity.
        result.resolver_ = [&, current = std::move(current), segments = std::move(segments),
            projectSeconds, sourceSeconds, fps, parameters, context, activity, decode, lease, staging, preparedFrame]
            (const MaterialFrameEvaluation& query,MaterialFrameReceipt& receipt,std::string& error) mutable {
            receipt = {}; staging->bundle.reset(); staging->submitted = false;
            if (!activity(error) || !compiled.importedSceneRender || !compiled.materialFrameSource)
            { if (error.empty()) error = "Imported Frame invocation has no admitted render metadata"; return false; }
            auto expectedFrame = visualdeformation::RationalFrameTime{};
            if (!importedanimation::importedAnimationFrameIdentity(sourceSeconds,fps,expectedFrame,error)) return false;
            if (*preparedFrame)
            {
                if ((*preparedFrame)->rateNumerator != expectedFrame.rateNumerator
                    || (*preparedFrame)->rateDenominator != expectedFrame.rateDenominator)
                { error = "Imported Frame prepared clock differs from its source rate"; return false; }
                expectedFrame = **preparedFrame;
            }
            const auto endpoint = visualimportedscenerender::materialFrameEndpoint(*compiled.importedSceneRender);
            const MaterialFrameEvaluation expected {endpoint ? *endpoint : query.endpoint, expectedFrame,plan.clipId,
                plan.structuralRevision,compiled.importedSceneRender->evaluationRevision,
                context.projectGeneration,context.helperGeneration,Use};
            if (!endpoint || !(query == expected))
            { error = "Imported Frame request differs from its frozen render invocation"; return false; }

            const auto validSegment = [](const Segment& segment) {
                return segment.clipId > 0 && std::isfinite(segment.inSec) && segment.inSec >= 0
                    && std::isfinite(segment.outSec) && segment.outSec > segment.inSec
                    && std::isfinite(segment.rate) && segment.rate > 0
                    && std::isfinite(segment.displayStartSec) && segment.displayStartSec >= 0
                    && std::isfinite((segment.outSec-segment.inSec)/segment.rate);
            };
            const auto inDisplay = [&](const Segment& segment,double at) {
                const double end = segment.displayStartSec + (segment.outSec-segment.inSec)/segment.rate;
                return validSegment(segment) && std::isfinite(end) && std::isfinite(at)
                    && at >= segment.displayStartSec && at < end && at >= 0 && at <= 86400000;
            };
            const bool scheduled = bool(compiled.geometryFrameSchedule);
            if (!validSegment(current) || current.clipId != plan.clipId
                || !std::isfinite(projectSeconds) || projectSeconds < 0 || projectSeconds > 86400000
                || !std::isfinite(sourceSeconds) || sourceSeconds < 0)
            { error = "Imported Frame invocation has an invalid current owner or sample"; return false; }
            // Direct compatibility uses the captured source query. The caller
            // may hold an outgoing clip at its inclusive trim boundary.
            auto display = query.frame;
            auto firstTick = display.frame;
            const auto secondsAt = [&](std::int64_t tick,double& at,std::string& failure) {
                at = tick == display.frame ? projectSeconds
                    : static_cast<double>(static_cast<long double>(projectSeconds)
                        + (static_cast<long double>(tick)-display.frame)*display.rateDenominator/display.rateNumerator);
                if (!inDisplay(current,at))
                { failure = "Imported Frame dependency is outside its owning display interval"; return false; }
                return true;
            };
            if (scheduled)
            {
                const double mappedCurrent = current.inSec + (projectSeconds-current.displayStartSec)*current.rate;
                if (!inDisplay(current,projectSeconds) || sourceSeconds < current.inSec || sourceSeconds >= current.outSec
                    || !std::isfinite(mappedCurrent) || std::abs(mappedCurrent-sourceSeconds) > 1.0e-9)
                { error = "Imported Frame anchor does not belong to its current saved segment"; return false; }
                if (!context.materialFrameClock || !context.materialFrameClock(projectSeconds,display,error)
                    || !display.rateNumerator || !display.rateDenominator)
                { if (error.empty()) error = "Imported Frame requires an exact project clock"; return false; }
                const long double first = std::ceil(static_cast<long double>(display.frame)
                    + (static_cast<long double>(current.displayStartSec)-projectSeconds)
                        * display.rateNumerator/display.rateDenominator - 1.0e-9L);
                if (!std::isfinite(first) || first < std::numeric_limits<std::int64_t>::min()
                    || first > std::numeric_limits<std::int64_t>::max())
                { error = "Imported Frame replay index overflows its bounded clock"; return false; }
                firstTick = static_cast<std::int64_t>(first);
                double firstSeconds = 0;
                // Verify the rounded boundary on the anchored lattice. Do not
                // clamp a dependency's time to the media endpoint.
                if (!secondsAt(firstTick,firstSeconds,error))
                {
                    if (!std::isfinite(firstSeconds) || firstSeconds >= current.displayStartSec
                        || current.displayStartSec-firstSeconds > 1.0e-9
                        || firstTick == std::numeric_limits<std::int64_t>::max()) return false;
                    ++firstTick;
                    if (!secondsAt(firstTick,firstSeconds,error)) return false;
                    error.clear();
                }
                if (firstTick > display.frame || firstTick < 0
                    || static_cast<long double>(display.frame)-firstTick > std::numeric_limits<std::int64_t>::max())
                { error = "Imported Frame replay interval exceeds its signed tick bound"; return false; }
            }
            const auto decodedLeaf = [&](const MaterialFrameEvaluation& leaf,std::uint64_t available,
                MaterialFrameReceipt& resolved,std::string& failure) {
                if (!activity(failure)) return false;
                auto identity = query; identity.endpoint = leaf.endpoint; identity.frame = display;
                identity.frame.frame = leaf.frame.frame;
                if (!(identity == leaf) || leaf.frame.frame < firstTick || leaf.frame.frame > display.frame)
                { failure = "Imported Frame decoded leaf has a foreign invocation or clock"; return false; }
                if (compiled.geometryFrameSchedule)
                {
                    const auto found = compiled.geometryFrameSchedule->operations.find(leaf.endpoint);
                    if (found == compiled.geometryFrameSchedule->operations.end()
                        || found->second.shader || found->second.temporal || !found->second.inputs.empty())
                    { failure = "Imported Frame query is not an admitted decoded leaf"; return false; }
                }
                else if (!(leaf.endpoint == query.endpoint)) return false;
                int clip = 0; videosourcecolor::Policy policy;
                if (!videowire::materialframe::sourceClip(plan,leaf.endpoint,clip,failure)
                    || !videowire::materialframe::sourcePolicy(plan,leaf.endpoint,policy,failure)) return false;
                const auto operation = std::find_if(plan.operations.begin(),plan.operations.end(),
                    [&](const auto& value) { return value.nodeId == leaf.endpoint.node; });
                if (operation == plan.operations.end()) return false;
                Segment selected = current;
                const bool saved = operation->kind == "video.layer.source";
                if (saved)
                {
                    const auto count = std::count_if(segments.begin(),segments.end(),
                        [&](const auto& value) { return value.clipId == clip; });
                    if (count != 1)
                    { failure = "Surface Layer Source requires exactly one saved project segment"; return false; }
                    selected = *std::find_if(segments.begin(),segments.end(),
                        [&](const auto& value) { return value.clipId == clip; });
                }
                else if (operation->kind != "video.source" || clip != current.clipId) return false;
                if (!compiled.geometryFrameSchedule && (saved || leaf.frame.frame != display.frame))
                { failure = "Direct imported Frame compatibility requires the current Video Source"; return false; }
                double media = sourceSeconds;
                if (scheduled)
                {
                    double at = 0;
                    if (!secondsAt(leaf.frame.frame,at,failure) || !inDisplay(selected,at)) return false;
                    media = saved ? selected.inSec + (at-selected.displayStartSec)*selected.rate
                        : leaf.frame.frame == display.frame ? sourceSeconds
                            : sourceSeconds + (at-projectSeconds)*current.rate;
                    if (!std::isfinite(media) || media < selected.inSec || media >= selected.outSec)
                    { failure = "Surface Layer Source is outside its saved media interval"; return false; }
                }
                if (!activity(failure)) return false;
                auto* decoded = decode(selected,media,leaf,failure);
                if (!decoded || !activity(failure))
                { if (failure.empty()) failure = "Material Frame source could not decode its selected exact time"; return false; }
                const bool hdr = decoded->hdrFrame.width > 0;
                const auto width = hdr ? decoded->hdrFrame.width : decoded->texW;
                const auto height = hdr ? decoded->hdrFrame.height : decoded->texH;
                std::uint64_t reservation = 0;
                // HDR leases may return admitted linear input or a declared SDR
                // conversion. Reserve the conservative existing float input size.
                if (!videowire::materialFrameImageBytes(width,height,hdr,available,reservation,failure)
                    || !activity(failure)) return false;
                auto image = lease(*decoded,policy,failure);
                if (!activity(failure) || !arbitgpu::validMaterialFrameTexture(image)
                    || image->width() != std::uint32_t(width) || image->height() != std::uint32_t(height))
                { if (failure.empty()) failure = "Material Frame source lease differs from its decoded owner extent"; return false; }
                std::uint64_t actual = 0;
                if (!videowire::materialFrameImageBytes(image->width(),image->height(),
                    arbitgpu::materialFrameIsLinear(image->colorTextureDescriptor()),available,actual,failure)
                    || actual > reservation) return false;
                resolved = {leaf,std::move(image)}; return true;
            };
            std::vector<Image> owners;
            if (compiled.geometryFrameSchedule)
            {
                if (compiled.geometryFrameSchedule->outputs != std::vector<videowire::materialframe::Endpoint>{query.endpoint}
                    || !compiled.geometryFramePlan)
                { error = "Imported Frame schedule does not own its exact authored output"; return false; }
                videowire::MaterialFrameDagInvocation invocation {query,display,firstTick,secondsAt,activity,decodedLeaf,&owners};
                auto evaluated = context;
                if (!videowire::prepareMaterialFrameDag(renderer,plan,inputs,
                    Use == NativeImportedSceneRenderUse::Preview ? geometry::PlanUse::preview : geometry::PlanUse::exportRender,
                    projectSeconds,parameters,evaluated,error,&compiled,&invocation)
                    || !evaluated.geometryMaterialFrame || !(evaluated.geometryMaterialFrame->evaluation == query)) return false;
                receipt = *evaluated.geometryMaterialFrame;
            }
            else
            {
                auto leaf = query; leaf.frame = display;
                if (!decodedLeaf(leaf,colortransform::kMaximumResidentBytes,receipt,error)) return false;
                owners.push_back(receipt.nativeFrame); receipt.evaluation = query;
            }
            if (!activity(error)) { receipt = {}; return false; }
            staging->bundle = std::make_shared<const Bundle>(Bundle{query,std::move(owners)});
            transaction.addPublication(
                [staging,activity](std::string& failure) {
                    if (!staging->bundle || !staging->submitted)
                    { failure = "Imported Frame dependency was not submitted for publication"; return false; }
                    return activity(failure);
                },
                [staging] { staging->bundle.reset(); },
                [staging] { staging->bundle.reset(); staging->submitted = false; });
            return true;
        };
        return result;
    }
};
} // namespace videohelper::importedscene
