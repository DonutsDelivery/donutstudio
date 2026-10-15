// Included inside arbitgpu, after NativeDeformationScene. No production caller
// can mint these capabilities or supply a matrix as admission evidence.
class NativeDeformationCompositionFixture;
struct NativeDeformationFrameData;

class NativeDeformationComputePostBlock final
{
public:
    NativeDeformationComputePostBlock() = default;
    const std::array<float, 4>& control() const noexcept { return control_; }
    const std::array<float, 16>& point() const noexcept { return point_; }
    const std::array<float, 16>& normal() const noexcept { return normal_; }
private:
    friend class NativeDeformationCompositionFrame;
    std::array<float, 4> control_ {1.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, 16> point_ {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    std::array<float, 16> normal_ {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
};
static_assert(sizeof(NativeDeformationComputePostBlock) == 144,
              "native deformation post block must retain its private layout");

struct NativeDeformationCompositionMapping final
{
    enum class Kind : std::uint8_t { Imported, Procedural };
    Kind kind = Kind::Imported;
    HarmonicMIDI::grid::SceneObjectId sourceObject {}, composedObject {};
    // Explicitly authenticated against the imported source only. Procedural
    // mappings must carry zero, regardless of their numeric object spelling.
    std::uint64_t importedNodeStableId = 0;
};

// One imported leaf of a multi-leaf owner. Animated leaves carry their own
// exact payload, documents and native rest draw, and keep their own clip and
// evaluation. Static leaves carry no animation document and no native draw;
// their objects become rigid members, like procedural objects. Imported
// mappings of a multi-leaf owner address leaf k's node r through the
// k * kImportedLeafIdStride + r identity; a single-leaf owner keeps raw IDs.
struct NativeDeformationCompositionLeaf final
{
    visualmodelassetpayload::PayloadPtr payload;
    std::shared_ptr<const videohelper::gltf::GlbStaticMeshDocument> document;
    std::shared_ptr<const videohelper::gltf::GlbAnimationDocument> animation;
    std::optional<std::size_t> selectedScene;
    std::uint64_t selectedMesh = 0;
    std::shared_ptr<const NativeDeformationScene> imported;
    // The leaf's own rest scene. An animated leaf must name imported->scene.
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> scene;
};

// One frame sample per leaf, in owner leaf order. Static leaves carry neither.
struct NativeDeformationCompositionLeafSample final
{
    std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot> deformation;
    std::shared_ptr<const visualanimation::Sample> sceneAnimation;
};

class NativeDeformationCompositionOwner final
{
public:
    static constexpr std::size_t kMaximumLogicalBytes = 8 * 1024 * 1024;
    static constexpr std::size_t kMaximumLogicalWork = 4096ull * 1024 * 1024;
    const auto& scene() const noexcept { return composed_; }
    std::uint64_t revision() const noexcept { return revision_; }
    std::uint64_t evaluationRevision() const noexcept { return evaluationRevision_; }
    std::uint64_t projectGeneration() const noexcept { return projectGeneration_; }
    std::uint64_t helperGeneration() const noexcept { return helperGeneration_; }
    std::size_t logicalBytes() const noexcept { return logicalBytes_; }
    std::size_t logicalWork() const noexcept { return logicalWork_; }
    std::size_t memberCount() const noexcept { return memberCount_; }
    // Logical storage of one owner without its leaf list. A single leaf keeps
    // the released charge; only several leaves add the list (leafListBytes).
    static constexpr std::size_t singleLeafBytes() noexcept
    { return sizeof(NativeDeformationCompositionOwner)-sizeof(std::vector<Leaf>); }
    static constexpr std::size_t leafListBytes(std::size_t leafCount) noexcept
    { return leafCount<2 ? std::size_t(0) : 2*sizeof(std::vector<Leaf>)+leafCount*sizeof(Leaf); }
    // Sum of every leaf's node count. A single leaf keeps its document count.
    std::size_t importedNodeCount() const noexcept { return importedNodeCount_; }
    std::size_t leafCount() const noexcept { return leaves_.empty() ? std::size_t(1) : leaves_.size(); }
    bool staticLeaf(std::size_t leaf) const noexcept
    { return !leaves_.empty() && leaf<leaves_.size() && !leaves_[leaf].input.animation; }
    // Accepts exactly the node identities this owner publishes in its world table.
    bool importedNode(std::uint64_t id) const noexcept
    {
        if (leaves_.empty()) return id!=0 && id<=importedNodeCount_;
        if (id==0 || id>=0x80000000u) return false;
        const auto decoded=HarmonicMIDI::grid::scenecomposition::decodeImportedLeafId(
            static_cast<std::uint32_t>(id),leaves_.size());
        return decoded && decoded->raw<=leaves_[decoded->leaf].nodeCount;
    }
    // Procedural members report the identity leaf, the first animated leaf.
    std::size_t memberLeaf(std::size_t index) const noexcept { return recordLeaf(members_[index].record); }
    bool memberRigid(std::size_t index) const noexcept
    {
        const auto& record=records_[members_[index].record];
        return record.mapping.kind!=NativeDeformationCompositionMapping::Kind::Imported
            || staticLeaf(recordLeaf(members_[index].record));
    }
    const std::shared_ptr<const NativeDeformationScene>& originalMember(std::size_t index) const noexcept
    { return members_[index].source; }
    HarmonicMIDI::grid::SceneObjectId memberObject(std::size_t index) const noexcept
    { return records_[members_[index].record].mapping.composedObject; }
    bool matches(const NativeDeformationScene&) const noexcept;
    // A multi-leaf query names leaf k's node r as k * kImportedLeafIdStride + r,
    // like the mappings; both sides decode to (leaf, raw) before they compare.
    bool targetsObject(std::uint64_t node, HarmonicMIDI::grid::SceneObjectId object) const noexcept
    {
        using HarmonicMIDI::grid::scenecomposition::decodeImportedLeafId;
        const auto decode=[this](std::uint64_t id) {
            return id<0x80000000u ? decodeImportedLeafId(static_cast<std::uint32_t>(id),leaves_.size()) : std::nullopt;
        };
        const auto query=leaves_.empty() || node==0 ? std::nullopt : decode(node);
        if (!leaves_.empty() && node!=0 && !query) return false;
        for (std::size_t i=0; i<recordCount_; ++i)
            if (records_[i].mapping.composedObject==object)
            {
                if (node==0) return true;
                for (std::size_t step=0; i!=64 && step<recordCount_; ++step)
                {
                    const auto& record=records_[i];
                    if (record.mapping.kind==NativeDeformationCompositionMapping::Kind::Imported)
                    {
                        if (leaves_.empty() ? record.mapping.importedNodeStableId==node : [&] {
                                const auto mapped=decode(record.mapping.importedNodeStableId);
                                return mapped && mapped->leaf==query->leaf && mapped->raw==query->raw;
                            }()) return true;
                    }
                    i=record.parent;
                }
                return false;
            }
        return false;
    }
private:
    friend class videohelper::modelpayload::ImportedAnimatedScenePayloadExecution;
    friend class NativeDeformationCompositionFixture;
    friend class NativeDeformationCompositionFrame;
    friend bool prepareNativeDeformationFrame(const NativeDeformationScene&,
        const visualdeformation::AnimationDeformationSnapshot&,
        const std::shared_ptr<const NativeDeformationCompositionFrame>&,
        NativeDeformationFrameData&, std::string&,
        NativeSceneMotionSample::ImportedNodeWorldMatrices*);
    using Matrix = std::array<float, 16>;
    struct Record final
    {
        NativeDeformationCompositionMapping mapping;
        std::size_t parent = 64, originalParent = 64;
        Matrix local {}, originalWorld {}, staticWorld {}, inverseStaticWorld {};
    };
    struct Member final
    {
        std::size_t record = 0;
        std::shared_ptr<const NativeDeformationScene> source;
    };
    struct Leaf final
    {
        NativeDeformationCompositionLeaf input;
        std::size_t nodeCount = 0, nodeOffset = 0;
    };
    NativeDeformationCompositionOwner() = default;
    NativeDeformationCompositionOwner(const NativeDeformationCompositionOwner&) = delete;
    NativeDeformationCompositionOwner(NativeDeformationCompositionOwner&&) = default;
    std::size_t identityLeaf() const noexcept
    {
        for (std::size_t leaf=0; leaf<leaves_.size(); ++leaf) if (leaves_[leaf].input.animation) return leaf;
        return 0;
    }
    std::size_t recordLeaf(std::size_t record) const noexcept
    {
        const auto& mapping=records_[record].mapping;
        if (leaves_.empty()) return 0;
        if (mapping.kind!=NativeDeformationCompositionMapping::Kind::Imported) return identityLeaf();
        const auto decoded=HarmonicMIDI::grid::scenecomposition::decodeImportedLeafId(
            static_cast<std::uint32_t>(mapping.importedNodeStableId),leaves_.size());
        return decoded ? decoded->leaf : 0;
    }
    // Single leaf: the released owner, unchanged and without leaf storage.
    static std::shared_ptr<const NativeDeformationCompositionOwner> admit(
        visualmodelassetpayload::PayloadPtr,
        std::shared_ptr<const videohelper::gltf::GlbStaticMeshDocument>,
        std::shared_ptr<const videohelper::gltf::GlbAnimationDocument>,
        std::optional<std::size_t> selectedScene, std::uint64_t selectedMesh,
        std::shared_ptr<const NativeDeformationScene> imported,
        std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> procedural,
        std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> composed,
        const std::vector<NativeDeformationCompositionMapping>&,
        std::uint64_t projectGeneration, std::uint64_t helperGeneration,
        std::string&, std::uint64_t evaluationRevision);
    // Two to kMaximumImportedLeaves leaves, at least one animated. The first
    // animated leaf supplies the batch identity, clip and native deformation.
    static std::shared_ptr<const NativeDeformationCompositionOwner> admit(
        std::vector<NativeDeformationCompositionLeaf>,
        std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> procedural,
        std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> composed,
        const std::vector<NativeDeformationCompositionMapping>&,
        std::uint64_t projectGeneration, std::uint64_t helperGeneration,
        std::string&, std::uint64_t evaluationRevision);
    static std::shared_ptr<const NativeDeformationCompositionOwner> admitLeaves(
        Leaf*, std::size_t leafCount,
        std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> procedural,
        std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> composed,
        const std::vector<NativeDeformationCompositionMapping>&,
        std::uint64_t projectGeneration, std::uint64_t helperGeneration,
        std::string&, std::uint64_t evaluationRevision, std::vector<Leaf>* retained);
    visualmodelassetpayload::PayloadPtr payload_;
    std::shared_ptr<const videohelper::gltf::GlbStaticMeshDocument> document_;
    std::shared_ptr<const videohelper::gltf::GlbAnimationDocument> animation_;
    std::optional<std::size_t> selectedScene_;
    std::uint64_t selectedMesh_ = 0;
    std::shared_ptr<const NativeDeformationScene> imported_;
    std::shared_ptr<const HarmonicMIDI::grid::Visual3DScene> procedural_, composed_;
    std::uint64_t revision_ = 0, projectGeneration_ = 0, helperGeneration_ = 0;
    std::uint64_t evaluationRevision_ = 0;
    std::array<Record, 64> records_ {};
    std::array<Member, 64> members_ {};
    std::size_t recordCount_ = 0, memberCount_ = 0, importedNodeCount_ = 0;
    std::size_t logicalBytes_ = 0, logicalWork_ = 0;
    // Empty for a single leaf; otherwise every leaf in admitted order.
    std::vector<Leaf> leaves_;
};

class NativeDeformationCompositionFrame final
{
public:
    const auto& owner() const noexcept { return owner_; }
    const auto& time() const noexcept { return time_; }
    std::size_t logicalBytes() const noexcept { return logicalBytes_; }
    std::size_t logicalWork() const noexcept { return logicalWork_; }
    std::uint64_t evaluationRevision() const noexcept { return owner_->evaluationRevision(); }
    // Logical storage of one frame without its sample list; see the owner.
    static constexpr std::size_t singleLeafBytes() noexcept
    { return sizeof(NativeDeformationCompositionFrame)-sizeof(std::vector<NativeDeformationCompositionLeafSample>); }
    static constexpr std::size_t leafListBytes(std::size_t leafCount) noexcept
    {
        return leafCount<2 ? std::size_t(0)
            : 2*sizeof(std::vector<NativeDeformationCompositionLeafSample>)+leafCount*sizeof(NativeDeformationCompositionLeafSample);
    }
    bool matches(const NativeDeformationScene&,
                 const visualdeformation::AnimationDeformationSnapshot&) const noexcept;
    bool retains(const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>& snapshot) const noexcept
    { return snapshot == deformation_; }
    bool matchesTime(visualdeformation::RationalFrameTime time) const noexcept
    { return time == time_; }
    bool sameOwner(const NativeDeformationCompositionFrame& other) const noexcept
    { return owner_ == other.owner_; }
    // The leaf's own snapshot. Rigid and procedural members use the identity
    // leaf's, which also authenticates the whole frame.
    const visualdeformation::AnimationDeformationSnapshot& memberSnapshot(std::size_t member) const noexcept
    {
        if (leaves_.empty() || owner_->memberRigid(member)) return *deformation_;
        return *leaves_[owner_->memberLeaf(member)].deformation;
    }
private:
    friend class videohelper::modelpayload::ImportedAnimatedScenePayloadExecution;
    friend class NativeDeformationCompositionFixture;
    friend bool prepareNativeDeformationFrame(const NativeDeformationScene&,
        const visualdeformation::AnimationDeformationSnapshot&,
        const std::shared_ptr<const NativeDeformationCompositionFrame>&,
        NativeDeformationFrameData&, std::string&,
        NativeSceneMotionSample::ImportedNodeWorldMatrices*);
    NativeDeformationCompositionFrame() = default;
    NativeDeformationCompositionFrame(const NativeDeformationCompositionFrame&) = delete;
    NativeDeformationCompositionFrame(NativeDeformationCompositionFrame&&) = default;
    static std::shared_ptr<const NativeDeformationCompositionFrame> admit(
        std::shared_ptr<const NativeDeformationCompositionOwner>,
        std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>,
        std::shared_ptr<const visualanimation::Sample>,
        visualdeformation::RationalFrameTime, std::uint64_t projectGeneration,
        std::uint64_t helperGeneration, std::string&);
    // A multi-leaf owner takes exactly one sample per leaf, in leaf order, all
    // at the same presentation time.
    static std::shared_ptr<const NativeDeformationCompositionFrame> admit(
        std::shared_ptr<const NativeDeformationCompositionOwner>,
        std::vector<NativeDeformationCompositionLeafSample>,
        visualdeformation::RationalFrameTime, std::uint64_t projectGeneration,
        std::uint64_t helperGeneration, std::string&);
    static std::shared_ptr<const NativeDeformationCompositionFrame> admitLeaves(
        std::shared_ptr<const NativeDeformationCompositionOwner>,
        const NativeDeformationCompositionLeafSample*, std::size_t sampleCount,
        std::vector<NativeDeformationCompositionLeafSample>* retained,
        visualdeformation::RationalFrameTime, std::uint64_t projectGeneration,
        std::uint64_t helperGeneration, std::string&);
    std::shared_ptr<const NativeDeformationCompositionOwner> owner_;
    std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot> deformation_;
    std::shared_ptr<const visualanimation::Sample> sceneAnimation_;
    visualdeformation::RationalFrameTime time_ {};
    std::array<NativeDeformationComputePostBlock, 64> posts_ {};
    NativeSceneMotionSample::ImportedNodeWorldMatrices rawWorlds_;
    std::size_t logicalBytes_ = 0, logicalWork_ = 0;
    // Empty for a single leaf; otherwise one retained sample per owner leaf.
    std::vector<NativeDeformationCompositionLeafSample> leaves_;
};

inline bool validNativeCompositionRetention(const std::shared_ptr<const NativeDeformationCompositionFrame>& frame,
    const std::shared_ptr<const visualdeformation::AnimationDeformationSnapshot>& snapshot) noexcept
{
    return !frame || (snapshot && frame->retains(snapshot) && frame->matchesTime(snapshot->time())
        && frame->logicalBytes()<=NativeDeformationCompositionOwner::kMaximumLogicalBytes/2);
}
inline bool sameNativeCompositionOwner(const std::shared_ptr<const NativeDeformationCompositionFrame>& a,
    const std::shared_ptr<const NativeDeformationCompositionFrame>& b) noexcept
{ return (!a && !b) || (a && b && a->sameOwner(*b)); }
inline bool nativeCompositionTargetsObject(const std::shared_ptr<const NativeDeformationCompositionFrame>& frame,
    std::uint64_t node,HarmonicMIDI::grid::SceneObjectId object) noexcept
{ return frame && frame->owner()->targetsObject(node,object); }

inline bool validNativeCompositionWorldTable(
    const std::shared_ptr<const NativeDeformationCompositionFrame>& frame,
    const std::shared_ptr<const NativeSceneMotionSample::ImportedNodeWorldMatrices>& table) noexcept
{
    if (!table) return true;
    const auto& owner=*frame->owner();
    const auto nodes=owner.importedNodeCount();
    if (table->size()>nodes || table->capacity()>2*nodes) return false;
    for (const auto& entry:*table) if (!owner.importedNode(entry.nodeStableId)) return false;
    return true;
}
