#pragma once

#include "particle_history.h"
#include "water_surface_replay.h"
#include "geometry_score_field.h"
#include "../../shared/GeometryAudioDeformer.h"

namespace videorender {
// One child of a timeline Geometry3D family collider. Its mesh evaluates in
// the child object's local space. The child's composed transform places it in
// the frame its bodies attach to: the parent every child shares, or the scene
// root when children have different parents. translation and linear form that
// affine map, with linear row-major and applied before translation.
struct ParticleColliderFamilyChild
{
    std::uint32_t object = 0;
    std::shared_ptr<const videowire::geometry::ValueDescriptor> mesh;
    std::array<float,3> translation {};
    std::array<float,9> linear {1,0,0,0,1,0,0,0,1};
    bool operator==(const ParticleColliderFamilyChild& other) const
    {
        return object==other.object && mesh==other.mesh && translation==other.translation
            && linear==other.linear;
    }
    bool operator!=(const ParticleColliderFamilyChild& other) const { return !(*this==other); }
};

struct ParticleGeometryBinding
{
    std::shared_ptr<const videowire::geometry::ValueDescriptor> emitters, attractors, mesh, points, body;
    // Keep the decoder-issued token. A descriptor or digest cannot recreate it.
    std::shared_ptr<const videowire::geometry::AdmittedValue> bodySource;
    float scale = 0.25f, centerX = 0.5f, centerY = 0.5f, centerZ = 0.5f;
    int projection = 0;
    // Empty for one collider mesh. A family holds every child in object order,
    // and mesh then aliases the first child's descriptor, so the existing
    // collider rules see one mesh collider. One replay meets every child.
    std::vector<ParticleColliderFamilyChild> meshFamily;
};

// Replay coordinates of one family child vertex. The sampler and the imported
// composition check both call this, so their float results match exactly.
template <typename Point>
inline std::array<float,3> particleFamilyColliderPoint(const ParticleGeometryBinding& binding,
    const ParticleColliderFamilyChild& child, const Point& point)
{
    const auto& l=child.linear;
    const auto& t=child.translation;
    const float ox=t[0]+l[0]*point.x+l[1]*point.y+l[2]*point.z;
    const float oy=t[1]+l[3]*point.x+l[4]*point.y+l[5]*point.z;
    const float oz=t[2]+l[6]*point.x+l[7]*point.y+l[8]*point.z;
    return {binding.centerX+binding.scale*ox,binding.centerY+binding.scale*oy,binding.centerZ+binding.scale*oz};
}

// Local affine of one scene object as Visual3DScene applies it: scale, then
// rotation, then translation. linear is rotation times diagonal scale.
template <typename Transform>
inline void particleLocalAffine(const Transform& transform, double (&linear)[9], double (&translation)[3])
{
    const double x=transform.rotation.x, y=transform.rotation.y, z=transform.rotation.z, w=transform.rotation.w;
    const double rotation[9]={1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                              2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                              2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)};
    const double scale[3]={transform.scale.x, transform.scale.y, transform.scale.z};
    for (std::size_t row=0;row<3;++row)
        for (std::size_t column=0;column<3;++column)
            linear[row*3+column]=rotation[row*3+column]*scale[column];
    translation[0]=transform.translation.x;
    translation[1]=transform.translation.y;
    translation[2]=transform.translation.z;
}

// The composed transform of one family child relative to frame, the parent
// its bodies attach to. A null frame is the scene root. Each object between
// the child and frame multiplies in, child to root, as the scene hierarchy does.
// Zero, negative or non-finite scale on the child or any object it multiplies
// through refuses, as does a parent chain that never reaches frame.
template <typename Scene, typename Record>
inline bool particleFamilyWorldTransform(const Scene& scene, const Record& object,
    decltype(Record::parent) frame, std::array<float,3>& translation, std::array<float,9>& linear)
{
    const auto positive=[](const auto& scale)
    { return scale.x>0 && scale.y>0 && scale.z>0
        && std::isfinite(scale.x) && std::isfinite(scale.y) && std::isfinite(scale.z); };
    if (!positive(object.transform.scale)) return false;
    double world[9], offset[3];
    particleLocalAffine(object.transform,world,offset);
    auto next=object.parent;
    for (std::size_t depth=0; next!=frame && next.isValid(); ++depth)
    {
        if (depth>=scene.objectCount) return false;
        const Record* parent=nullptr;
        for (std::size_t i=0;parent==nullptr && i<scene.objectCount;++i)
            if (scene.objects[i].id==next) parent=&scene.objects[i];
        if (parent==nullptr || !positive(parent->transform.scale)) return false;
        double parentLinear[9], parentOffset[3];
        particleLocalAffine(parent->transform,parentLinear,parentOffset);
        // parent applied to child: linear = parentLinear * world, offset = parentLinear * offset + parentOffset.
        double composed[9], moved[3];
        for (std::size_t row=0;row<3;++row)
        {
            for (std::size_t column=0;column<3;++column)
            {
                composed[row*3+column]=0;
                for (std::size_t inner=0;inner<3;++inner)
                    composed[row*3+column]+=parentLinear[row*3+inner]*world[inner*3+column];
            }
            moved[row]=parentOffset[row];
            for (std::size_t inner=0;inner<3;++inner)
                moved[row]+=parentLinear[row*3+inner]*offset[inner];
        }
        for (std::size_t i=0;i<9;++i) world[i]=composed[i];
        for (std::size_t i=0;i<3;++i) offset[i]=moved[i];
        next=parent->parent;
    }
    if (next!=frame) return false;
    for (const double value : world) if (!std::isfinite(value)) return false;
    for (const double value : offset) if (!std::isfinite(value)) return false;
    for (std::size_t i=0;i<9;++i) linear[i]=static_cast<float>(world[i]);
    for (std::size_t i=0;i<3;++i) translation[i]=static_cast<float>(offset[i]);
    return true;
}

// A shared frame is a parent that bodies attach to, so it and every ancestor
// must keep positive uniform scale. A null frame (scene root) always passes.
template <typename Scene, typename Frame>
inline bool particleFamilyFrameUniform(const Scene& scene, Frame frame)
{
    for (std::size_t depth=0; frame.isValid(); ++depth)
    {
        if (depth>=scene.objectCount) return false;
        decltype(&scene.objects[0]) record=nullptr;
        for (std::size_t i=0;record==nullptr && i<scene.objectCount;++i)
            if (scene.objects[i].id==frame) record=&scene.objects[i];
        if (record==nullptr) return false;
        const auto& scale=record->transform.scale;
        if (!(scale.x>0 && scale.x==scale.y && scale.x==scale.z && std::isfinite(scale.x))) return false;
        frame=record->parent;
    }
    return true;
}

// The merged family frame needs one nonzero identity that changes when any
// child's mesh identity or order changes. A single mesh keeps its own ID.
inline std::uint64_t particleColliderMeshIdentity(const ParticleGeometryBinding& binding)
{
    if (binding.meshFamily.empty()) return binding.mesh ? binding.mesh->stableId:0;
    std::uint64_t identity=0x9e3779b97f4a7c15ULL;
    for (const auto& child:binding.meshFamily) {
        identity^=(child.mesh ? child.mesh->stableId:0)+0x9e3779b97f4a7c15ULL+(identity<<6)+(identity>>2);
        identity=(identity^(identity>>30))*0xbf58476d1ce4e5b9ULL;
        identity=(identity^(identity>>27))*0x94d049bb133111ebULL;
        identity^=identity>>31;
    }
    return identity==0 ? 1:identity;
}

// Children usually reuse local vertex IDs, such as 1 to 4 on every grid, so
// the merged frame keys each vertex by its child object too. The mix is a
// bijection for one object, so IDs stay distinct within each child. A
// collision across children would fail the frame's unique-ID check and refuse.
inline std::uint64_t particleFamilyVertexIdentity(std::uint32_t object, std::uint64_t vertex)
{
    const auto mix=[](std::uint64_t value) {
        value+=0x9e3779b97f4a7c15ULL;
        value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;
        value=(value^(value>>27))*0x94d049bb133111ebULL;
        return value^(value>>31);
    };
    const auto identity=mix(mix(object)^vertex);
    return identity==0 ? 1:identity;
}

inline bool hasDynamicParticleBody(const ParticleParams& p)
{
    return p.preparedBodyReplay || (p.geometryBinding && ((p.geometryBinding->body
        && videowire::geometry::particleGeometryRequiresHistory(*p.geometryBinding->body))
        || (p.geometryBinding->bodySource
            && videowire::geometry::particleGeometryRequiresHistory(p.geometryBinding->bodySource->descriptor()))));
}

struct ParticleGeometryFrame
{
    // Mesh triangle edges are thin projected barriers, not filled solids.
    // A degenerate segment is a point collider. Ordering follows source IDs.
    std::array<std::array<float, 4>, 448> segments {};
    int count = 0;
    // Solid mesh contacts are authored two-sided triangle surfaces. Bounds
    // are only broad-phase metadata, never a filled collider.
    std::array<float,3> minimum {}, maximum {};
    std::uint64_t meshIdentity = 0;
    std::array<std::array<float,3>,256> meshVertices {};
    std::array<std::uint64_t,256> meshVertexIdentities {};
    std::array<std::array<std::uint32_t,3>,128> meshTriangles {};
    std::uint32_t meshVertexCount = 0, meshTriangleCount = 0;
    std::array<std::array<float,3>,64> solidPoints {};
    std::array<std::uint64_t,64> pointIdentities {};
    int pointCount = 0;
};

// Exact completed water endpoints over one global 120 Hz interval.
// The caller owns these immutable inputs and gives each new view a request-local
// identity. Sampling borrows them only for the duration of the synchronous call.
struct ParticlePriorWaterGeometryView final
{
    const videowire::geometry::OperationRecord& operation;
    const videowire::geometry::GeometryData& restMesh;
    const videowire::geometry::WaterSurfaceGrid grid;
    const std::vector<double>& heights;
    const std::int64_t intervalStartTicks, intervalEndTicks;
    const std::uint64_t identity;
    const std::vector<double>* nextHeights = nullptr;

    bool covers(std::int64_t start, std::int64_t end, std::string& error) const
    {
        if (identity==0 || intervalStartTicks<0 || intervalStartTicks%kParticleReplayStepTicks!=0
            || intervalStartTicks>std::numeric_limits<std::int64_t>::max()-kParticleReplayStepTicks
            || intervalEndTicks!=intervalStartTicks+kParticleReplayStepTicks
            || start<intervalStartTicks || end<start || end>intervalEndTicks)
        { error="Particle prior-water view requires exact coverage within one global water interval"; return false; }
        return true;
    }

    bool sample(const videowire::geometry::OperationRecord& requestedOperation,
        const videowire::geometry::GeometryData& requestedMesh,
        const videowire::geometry::WaterSurfaceGrid& requestedGrid, std::int64_t tick,
        double seconds, std::vector<double>& output, std::string& error) const
    {
        if (!covers(tick,tick,error)) return false;
        if (seconds!=double(tick)/kParticleReplayTicksPerSecond
            || requestedOperation.code!=operation.code || requestedOperation.stableId!=operation.stableId
            || requestedOperation.inputStableId!=operation.inputStableId
            || requestedOperation.outputStableId!=operation.outputStableId
            || requestedOperation.secondaryInputStableId!=operation.secondaryInputStableId
            || requestedOperation.parameters!=operation.parameters || !(requestedOperation.field==operation.field)
            || !videowire::geometry::equal(requestedOperation.transform,operation.transform)
            || requestedOperation.retainedMesh!=operation.retainedMesh || requestedOperation.sdfSurface!=operation.sdfSurface
            || requestedGrid.columns!=grid.columns || requestedGrid.rows!=grid.rows
            || requestedGrid.spacingX!=grid.spacingX || requestedGrid.spacingY!=grid.spacingY
            || !videowire::geometry::equal(requestedMesh,restMesh))
        { error="Particle prior-water view does not match the exact Water Height request"; return false; }
        // The shared Water Height consumer validates count and finite offsets.
        if (nextHeights) {
            if (nextHeights->size()!=heights.size())
            { error="Water interval endpoints changed vertex count"; return false; }
            const double fraction=double(tick-intervalStartTicks)/kParticleReplayStepTicks;
            output.resize(heights.size());
            for (std::size_t i=0;i<heights.size();++i)
                output[i]=heights[i]+fraction*((*nextHeights)[i]-heights[i]);
        } else output=heights;
        return true;
    }
};

inline bool sampleParticleGeometry(ParticleParams& p, std::int64_t tick,
                                  ParticleGeometryFrame& frame, std::string& error,
                                  const ParticlePriorWaterGeometryView* priorWater = nullptr)
{
    if (!p.geometryBinding) return true;
    using namespace videowire::geometry;
    const auto& binding = *p.geometryBinding;
    const visualdeformation::RationalFrameTime time {std::max<std::int64_t>(0,tick), std::uint32_t(kParticleReplayTicksPerSecond), 1};
    const double seconds = double(tick) / kParticleReplayTicksPerSecond;
    const auto requiresScore=[](const std::shared_ptr<const ValueDescriptor>& value)
    { return value && (!value->scoreFields.empty() || hasRuntimeScoreFields(*value)); };
    const auto dynamic=[](const std::shared_ptr<const ValueDescriptor>& value)
    { return value && particleGeometryRequiresHistory(*value); };
    const auto familyAny=[&](const auto& predicate)
    { return std::any_of(binding.meshFamily.begin(),binding.meshFamily.end(),[&](const auto& child) { return predicate(child.mesh); }); };
    if (dynamic(binding.emitters) || dynamic(binding.attractors) || dynamic(binding.mesh) || dynamic(binding.points)
        || familyAny(dynamic))
        if (p.motionMode!=2 || !p.historicalReplay || !p.history || !p.history->score
            || tick<0 || !std::isfinite(seconds))
        { error="Reactive particle geometry requires the owned project history at its exact replay tick"; return false; }
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> scoreFrame;
    if (requiresScore(binding.emitters) || requiresScore(binding.attractors) || requiresScore(binding.mesh) || requiresScore(binding.points)
        || familyAny(requiresScore))
    {
        auto key=p.history->key;
        key.frame=time.frame; key.fps=time.rateNumerator;
        const float beat=static_cast<float>(p.history->timeline.secondsToBeat(seconds)); key.beat=beat;
        canonicalblockc::FrameProducer producer;
        scoreFrame=producer.evaluate(key,p.history->score,beat);
        if (!scoreFrame) { error="Particle Score Field requires an admitted canonical score at its replay tick"; return false; }
    }
    const auto sourceAt=videohelper::sourceSpectrumReader(p.history ? p.history->analysisSources : nullptr);
    const auto spectrumAt=[&](const spectrum::Binding& source,double at,spectrum::Bands& bands)
    {
        bands={};
        if (!p.history || !std::isfinite(at) || at<0) return false;
        const arbitblockb::FeatureFrame* frame=nullptr;
        std::size_t index=0;
        if (source.source==spectrum::Source::master)
        {
            if (!p.history->hasAudio() || !std::isfinite(p.history->audioSampleRate)
                || p.history->audioFrames->front().sampleIndex!=arbitblockb::kHop) return false;
            const auto& frames=*p.history->audioFrames;
            const double hop=std::floor(at*p.history->audioSampleRate/arbitblockb::kHop)-1;
            if (!std::isfinite(hop)) return false;
            index=static_cast<std::size_t>(std::clamp(hop,0.0,double(frames.size()-1)));
            frame=&frames[index]; bands=frame->bands;
        }
        else
        {
            arbitmod::Audio audio;
            std::string diagnostic;
            if (!videohelper::sourceScalarAudioAt(p.history->analysisSources.get(),{source.sourceTrackId},at,audio,diagnostic)
                || audio.sources[0].tapPoint!=(source.source==spectrum::Source::group ? 2:1)
                || !sourceAt(source.source,source.sourceTrackId,at,bands)) return false;
            for (const auto& stream:*p.history->analysisSources)
                if (stream.trackId==source.sourceTrackId)
                {
                    if (stream.frames.front().sampleIndex!=arbitblockb::kHop) return false;
                    frame=videohelper::sourceFeatureAt(stream,at);
                    if (frame) index=static_cast<std::size_t>(frame-stream.frames.data());
                }
        }
        return (!frame || frame->sampleIndex==static_cast<std::int64_t>(index+1)*arbitblockb::kHop)
            && std::all_of(bands.begin(),bands.end(),[](float value) { return std::isfinite(value); });
    };
    // Scalar reader for Audio Effector Input. It resolves the frame with the same
    // lookup as spectrumAt, so rms, peak and onset come from the frame that
    // supplies the bands at this time.
    const auto scalarsAt=[&](const spectrum::Binding& source,double at,spectrum::Scalars& scalars)
    {
        scalars={};
        if (!p.history || !std::isfinite(at) || at<0) return false;
        if (source.source==spectrum::Source::master)
        {
            if (!p.history->hasAudio() || !std::isfinite(p.history->audioSampleRate)) return false;
            const auto& frames=*p.history->audioFrames;
            const auto& frame=frames[videohelper::analysisFrameIndex(frames.size(),p.history->audioSampleRate,at)];
            scalars.rms=frame.rms; scalars.peak=frame.peak; scalars.onset=frame.onset;
            return true;
        }
        const auto* stream=videohelper::selectedSourceStream(p.history->analysisSources.get(),source.source,source.sourceTrackId);
        if (!stream) return false;
        if (const auto* frame=videohelper::sourceFeatureAt(*stream,at))
        { scalars.rms=frame->rms; scalars.peak=frame->peak; scalars.onset=frame->onset; }
        return true;
    };
    // The live scalars at this time plus the matching reader for delayed and
    // history samples. The caller keeps `current` alive for the evaluation.
    const auto scalarInputFor=[&scalarsAt](const spectrum::Binding& field,const spectrum::Scalars& current)
    {
        spectrum::ScalarInput input;
        input.current=&current;
        input.at=[&scalarsAt,field](double at,spectrum::Scalars& scalars) { return scalarsAt(field,at,scalars); };
        return input;
    };
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
        if (priorWater)
            runtime.waterHeightAt=[&](const auto& operation,const auto& restMesh,const auto& grid,
                double requestedSeconds,auto& heights,auto& diagnostic)
            { return priorWater->sample(operation,restMesh,grid,tick,requestedSeconds,heights,diagnostic); };
        else if (p.history)
            runtime.waterHeightAt=[&](const auto& operation,const auto& restMesh,const auto& grid,
                double requestedSeconds,auto& heights,auto& diagnostic)
            { return replayWaterSurface(*p.history,operation,restMesh,grid,requestedSeconds,heights,diagnostic); };
        runtime.scoreAt=[&](const auto& operation,const auto& positions,auto& result,auto& diagnostic)
        {
            if (operation.code==OperationCode::evaluateField && operation.inputStableId!=operation.outputStableId)
            {
                const auto field=std::find_if(output.scoreFields.begin(),output.scoreFields.end(),[&](const auto& candidate)
                { return candidate.fieldStableId==operation.stableId && candidate.sourceGeometryStableId==operation.inputStableId
                    && candidate.consumerGeometryStableId==0; });
                if (field==output.scoreFields.end()) { diagnostic="Particle Score Field lost its admitted binding"; return false; }
                return videohelper::scorefield::evaluateInstanceField(*field,positions,scoreFrame,result,diagnostic);
            }
            return videohelper::scorefield::evaluateSampleField(operation,positions,scoreFrame,result,diagnostic);
        };
        bool missingSpectrum=false;
        const auto features=[&](const spectrum::Binding& field)
        {
            return spectrum::FeaturesAt{[&,field](double at)
            { spectrum::Bands bands{}; if (!spectrumAt(field,at,bands)) missingSpectrum=true; return bands; }};
        };
        runtime.spectrumAt=[&](const auto& field,auto& followed,std::string& diagnostic)
        {
            const auto found=std::find_if(output.spectrumFields.begin(),output.spectrumFields.end(),[&](const auto& candidate)
            { return spectrum::sameBinding(candidate,field); });
            spectrum::Bands current{}; spectrum::Follower follower;
            if (found==output.spectrumFields.end() || !spectrumAt(field,seconds,current))
            { diagnostic="Particle Spectrum Field requires its exact baked project source history"; return false; }
            spectrum::Scalars currentScalars{};
            if (field.input!=spectrum::EffectorInput::bandRange && !scalarsAt(field,seconds,currentScalars))
            { diagnostic="Particle Spectrum Field requires its exact baked project source history"; return false; }
            if (!spectrum::evaluate(field,current,seconds,features(field),follower,followed,diagnostic,
                    scalarInputFor(field,currentScalars))) return false;
            if (missingSpectrum) { diagnostic="Particle Spectrum Field is missing required project-time history"; return false; }
            return true;
        };
        const auto deform=[&](auto& geometry,const auto& field,std::string& diagnostic)
        { spectrum::Bands followed{}; return runtime.spectrumAt(field,followed,diagnostic)
            && applyAudioDeformer(geometry,field,followed,seconds,diagnostic); };
        ValueDescriptor replayed;
        MaterializedInstancePlan intermediates;
        if (!validateOperationPlan(output, {}, &intermediates, error, &replayed, &runtime, deform)
            || !particleGeometryIntermediatesFit(intermediates,error)
            || !validateParticleGeometry(replayed, source->carrier, error, false)) return false;
        // Legacy terminal bindings use the same shared consumers as retained draws.
        // Ordered fields and Audio Deformers have already run inside replay.
        for (const auto& field:output.spectrumFields)
        {
            if (field.target==spectrum::Target::geometryDeformation
                || std::any_of(output.operations.begin(),output.operations.end(),[&](const auto& operation)
                    { return operation.code==OperationCode::fieldAudioSpectrum && operation.stableId==field.fieldStableId; })) continue;
            if (replayed.carrier!=CarrierKind::geometry3D)
            { error="Terminal particle Spectrum displacement requires Geometry3D"; return false; }
            auto& geometry=std::get<GeometryData>(replayed.data);
            if (field.coordinates.size()!=geometry.positions.size())
            { error="Particle Spectrum Field lost its exact vertex samples"; return false; }
            spectrum::Bands followed{};
            std::vector<spectrum::Bands> rows;
            if (field.history.enabled)
            {
                spectrum::Bands current{}; spectrum::Follower follower;
                spectrum::Scalars currentScalars{};
                if (!spectrumAt(field,seconds,current)
                    || (field.input!=spectrum::EffectorInput::bandRange && !scalarsAt(field,seconds,currentScalars))
                    || !spectrum::evaluateHistory(field,current,seconds,features(field),false,true,follower,rows,error,
                        0,0,scalarInputFor(field,currentScalars))
                    || missingSpectrum)
                { if (error.empty()) error="Particle Spectral History requires its exact baked project source"; return false; }
            }
            else if (!runtime.spectrumAt(field,followed,error)) return false;
            for (std::size_t vertex=0;vertex<geometry.positions.size();++vertex)
            {
                const float amount=field.history.enabled ? spectrum::sampleHistory(field,rows,vertex)
                    : spectrum::sample(field,followed,field.coordinates[vertex]);
                auto& point=geometry.positions[vertex];
                point.x+=amount*field.direction[0]; point.y+=amount*field.direction[1]; point.z+=amount*field.direction[2];
            }
        }
        for (const auto& field:output.scoreFields)
        {
            if (std::any_of(output.operations.begin(),output.operations.end(),[&](const auto& operation)
                { return operation.code==OperationCode::evaluateField && operation.inputStableId!=operation.outputStableId
                    && operation.stableId==field.fieldStableId; })) continue;
            if (replayed.carrier!=CarrierKind::geometry3D)
            { error="Terminal particle Score displacement requires Geometry3D"; return false; }
            auto& geometry=std::get<GeometryData>(replayed.data);
            videohelper::scorefield::Evaluation evaluated;
            if (!videohelper::scorefield::evaluate(field,scoreFrame,evaluated,error)) return false;
            if (evaluated.offsets.size()!=geometry.positions.size())
            { error="Particle Score Field lost its exact vertex samples"; return false; }
            for (std::size_t vertex=0;vertex<geometry.positions.size();++vertex)
            {
                auto& point=geometry.positions[vertex]; const auto& offset=evaluated.offsets[vertex];
                point.x+=offset[0]; point.y+=offset[1]; point.z+=offset[2];
            }
        }
        if (replayed.carrier==CarrierKind::geometry3D)
        {
            const auto& geometry=std::get<GeometryData>(replayed.data);
            if (!std::all_of(geometry.positions.begin(),geometry.positions.end(),[](const auto& point) { return finite(point); }))
            { error="Reactive particle mesh exceeds the shared finite position bounds"; return false; }
        }
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
        if (emitters.size() > p.geometryAnchors.size())
        { error="Geometry particle emitters support at most 64 points; reduce emitter points"; return false; }
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
    frame.meshIdentity=0; frame.pointCount=0; frame.meshVertexCount=frame.meshTriangleCount=0;
    if (!binding.meshFamily.empty())
    {
        // One frame holds every child, in family order, so the replay's
        // vertex and triangle terms are the sums over children. Planar
        // segments are not written; a family collider is Solid 3D only.
        if (p.simulationSpace!=1 || binding.meshFamily.size()<2 || binding.mesh!=binding.meshFamily.front().mesh)
        { error="A timeline Geometry3D collider family requires Solid 3D and at least two children"; return false; }
        bool bounded=false;
        for (const auto& child:binding.meshFamily) {
            ValueDescriptor evaluated;
            if (!child.mesh || !evaluate(child.mesh,evaluated)) { if (error.empty()) error="Collider family child lost its mesh"; return false; }
            const auto& geometry=std::get<GeometryData>(evaluated.data);
            const auto firstVertex=frame.meshVertexCount;
            if (geometry.positions.empty() || geometry.vertexIds.size()!=geometry.positions.size()
                || geometry.indices.empty() || geometry.indices.size()%3!=0
                || geometry.positions.size()>frame.meshVertices.size()-firstVertex
                || geometry.indices.size()/3>frame.meshTriangles.size()-frame.meshTriangleCount)
            { error="Collider family meshes exceed 256 vertices and 128 triangles together"; return false; }
            for (std::size_t vertex=0;vertex<geometry.positions.size();++vertex) {
                const auto v=particleFamilyColliderPoint(binding,child,geometry.positions[vertex]);
                frame.meshVertices[firstVertex+vertex]=v;
                frame.meshVertexIdentities[firstVertex+vertex]=particleFamilyVertexIdentity(child.object,geometry.vertexIds[vertex]);
                if (!bounded) { frame.minimum=frame.maximum=v; bounded=true; }
                for (int axis=0;axis<3;++axis) {
                    frame.minimum[axis]=std::min(frame.minimum[axis],v[axis]-p.colliderThickness);
                    frame.maximum[axis]=std::max(frame.maximum[axis],v[axis]+p.colliderThickness);
                }
            }
            for (std::size_t triangle=0;triangle<geometry.indices.size()/3;++triangle)
                for (int corner=0;corner<3;++corner) {
                    const auto index=geometry.indices[triangle*3+corner];
                    if (index>=geometry.positions.size()) { error="Solid triangle surface contains an invalid vertex index"; return false; }
                    frame.meshTriangles[frame.meshTriangleCount+triangle][corner]=firstVertex+index;
                }
            frame.meshVertexCount+=static_cast<std::uint32_t>(geometry.positions.size());
            frame.meshTriangleCount+=static_cast<std::uint32_t>(geometry.indices.size()/3);
        }
        frame.meshIdentity=particleColliderMeshIdentity(binding);
    }
    else if (binding.mesh)
    {
        const auto& geometry = std::get<GeometryData>(mesh.data);
        if (p.simulationSpace==1) {
            if (geometry.positions.empty() || geometry.positions.size()>frame.meshVertices.size() || geometry.vertexIds.size()!=geometry.positions.size()
                || geometry.indices.empty() || geometry.indices.size()%3!=0 || geometry.indices.size()/3>frame.meshTriangles.size())
            { error="Solid triangle surfaces exceed their admitted indexed topology bounds"; return false; }
            frame.meshIdentity=mesh.stableId;
            frame.meshVertexCount=static_cast<std::uint32_t>(geometry.positions.size());
            frame.meshTriangleCount=static_cast<std::uint32_t>(geometry.indices.size()/3);
            for (std::size_t vertex=0;vertex<geometry.positions.size();++vertex) {
                frame.meshVertices[vertex]=world(geometry.positions[vertex]);
                frame.meshVertexIdentities[vertex]=geometry.vertexIds[vertex];
            }
            for (std::size_t triangle=0;triangle<frame.meshTriangleCount;++triangle)
                for (int corner=0;corner<3;++corner) {
                    const auto index=geometry.indices[triangle*3+corner];
                    if (index>=frame.meshVertexCount) { error="Solid triangle surface contains an invalid vertex index"; return false; }
                    frame.meshTriangles[triangle][corner]=index;
                }
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
        for (std::size_t vertex=0;vertex<frame.meshVertexCount;++vertex) if (!bounded(frame.meshVertices[vertex]))
        { error="Solid triangle vertex exceeds the coordinate limit of 32"; return false; }
        if (!bounded(frame.minimum) || !bounded(frame.maximum))
        { error="Solid mesh bounds exceed the coordinate limit of 32"; return false; }
        for (int i=0;i<frame.pointCount;++i) if (!bounded(frame.solidPoints[i]))
        { error="Solid point colliders exceed the coordinate limit of 32"; return false; }
    }
    return true;
}
}
