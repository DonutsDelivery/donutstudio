#pragma once

#include "../../../shared/RenderPassOutputContract.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace arbitgpu::surfacesource
{
// Units describe application-owned calls, metadata bytes and scalar setup
// operations. They do not bound libc, allocator, compiler or driver internals.
struct Setup final
{
    std::uint64_t literals=0, stages=0, sourceMethods=0, arithmetic=0,
        comparisons=0, writes=0, metadataCopies=0, byteControl=0, formattingCalls=0;
    bool within(const Setup& b) const noexcept
    { return literals<=b.literals && stages<=b.stages && sourceMethods<=b.sourceMethods
        && arithmetic<=b.arithmetic && comparisons<=b.comparisons && writes<=b.writes
        && metadataCopies<=b.metadataCopies && byteControl<=b.byteControl && formattingCalls<=b.formattingCalls; }
};
// Private V1 construction only. This is not a complete-base or driver-memory receipt.
struct Stage final
{
    std::uint64_t capacity=0, scans=0, copies=0, comparisons=0, formatting=0;
    bool operator==(const Stage& b) const noexcept
    { return capacity==b.capacity && scans==b.scans && copies==b.copies
        && comparisons==b.comparisons && formatting==b.formatting; }
};
// Stage budget of each part of the restricted Metal SurfacePbr recipe. A part
// that grows past its budget makes every Metal fixture preparation refuse with
// "Surface source setup exceeds admission"; surface_source_recipe_budget_tests
// measures the header-only parts on Linux.
struct StageBudget final
{
    static constexpr std::size_t nullVertex=2, finalFragment=8, opaque=24, sky=10,
        environment=24, renamedQuery=6, spectral=4;
    // 21 define stages, 7 for each of the twelve rough and seventeen smooth
    // status tokens, 3 for each of the thirteen uniform tokens, and 30 for the
    // remaining appends, replacements and the final concatenation (293), plus 10.
    static constexpr std::size_t transport=303;
};
struct Plan final
{
    static constexpr std::uint32_t version=1;
    // The opaque recipe runs three times (directly, in the glass query and in
    // the spectral query) and the renamed query twice.
    static constexpr std::size_t maximumStages=StageBudget::nullVertex+StageBudget::finalFragment
        +3*StageBudget::opaque+StageBudget::sky+StageBudget::environment+2*StageBudget::renamedQuery
        +StageBudget::spectral+StageBudget::transport;
    std::array<Stage,maximumStages> stages{};
    std::size_t count=0;
    std::uint64_t capacity=0, scans=0, copies=0, comparisons=0, formatting=0;
    std::uint64_t vertexLimit=0, fragmentLimit=0, literalCapacity=0, retainedCapacityLimit=0,
        metadataCapacity=0, metadataCopies=0, finalCopies=0;
    Setup setupLimit, censusSetup, emissionSetup;
};
inline std::uint64_t add(std::uint64_t a,std::uint64_t b)
{
    std::uint64_t out=0;
    if(!renderpassoutput::detail::checkedAdd(a,b,out)) throw std::length_error("Surface source addition overflow");
    return out;
}
inline std::uint64_t multiply(std::uint64_t a,std::uint64_t b)
{
    std::uint64_t out=0;
    if(!renderpassoutput::detail::checkedMultiply(a,b,out)) throw std::length_error("Surface source multiplication overflow");
    return out;
}
// At most two literal operands per stage, plus the thirteen standalone recipe
// literals: vertex/fragment, three opaque headers/bounds starts, sky, environment
// and its sky, transport and multiple-scatter starts. Sixteen helper returns
// each allow a return and caller transfer. Each stage allows six Source calls:
// emitting method, result construction/return, +=, assignment and wrapper copy.
inline constexpr std::size_t maximumLiterals=2*Plan::maximumStages+2+3*2+1+2+2;
inline constexpr std::size_t maximumSourceMethods=2*maximumLiterals+6*Plan::maximumStages+2*16+12;
// A ledger field allows sixteen source-level scalar operations, including
// advance/add/checkedAdd parameter transfers and accumulator writes. Allow two
// takes and 64 further entry/body operations per category. This is a C++ source
// schedule, not an instruction count after compiler or library lowering.
inline constexpr std::uint64_t setupEntryOperations=2*9*16+64;
// Per take, seven scalar-parameter transfers per field and four whole-record
// initialization/write/read transfers. Four temporaries/returns plus up to 64
// scalar body transfers need four records and ceil(64/9) records respectively.
// Scalar parameters together occupy one Setup-sized slot.
inline constexpr std::uint64_t setupEntryMetadataRecords=2*(7+4)+4+(64+9-1)/9;
// Seven complete Plan initializations/transfers cover all five fields per
// stage and up to 52 scalar/header fields, including the three Setup records.
// The remaining entries cover prepare/Context/Scope, helper and table control.
inline constexpr std::size_t fixedSetupEntries=
    (7*(5*Plan::maximumStages+52)+setupEntryOperations-1)/setupEntryOperations+32+16+12+13+3;
inline Setup setupEntry(std::uint64_t metadataBytes=0)
{
    Setup work;
    work.arithmetic=work.comparisons=work.writes=setupEntryOperations;
    work.metadataCopies=add(setupEntryMetadataRecords*sizeof(Setup),metadataBytes);
    return work;
}

struct Context final
{
    Plan plan;
    Setup setup;
    const Plan* admitted=nullptr;
    char* arena=nullptr;
    std::size_t used=0;
    static inline thread_local Context* current=nullptr;
    explicit Context(const Setup& limit):plan(),setup()
    {
        plan.setupLimit=limit;
        // This reservation precedes recipe evaluation. It includes the fixed
        // initialization above, Scope writes, helper control and prepare tail.
        auto work=setupEntry(7*sizeof(Plan));
        work.arithmetic=work.comparisons=work.writes=multiply(fixedSetupEntries,setupEntryOperations);
        work.metadataCopies=add(work.metadataCopies,multiply(fixedSetupEntries-1,setupEntryMetadataRecords*sizeof(Setup)));
        take(work);
    }
    void take(const Setup& work)
    {
        Setup next;
        // Subtraction tests refuse a category before its checked addition,
        // before any following method work, and without partial ledger writes.
        const auto advance=[](std::uint64_t used,std::uint64_t amount,std::uint64_t limit) {
            if(used>limit || amount>limit-used) throw std::length_error("Surface source setup exceeds admission");
            return add(used,amount);
        };
        next.literals=advance(setup.literals,work.literals,plan.setupLimit.literals);
        next.stages=advance(setup.stages,work.stages,plan.setupLimit.stages);
        next.sourceMethods=advance(setup.sourceMethods,work.sourceMethods,plan.setupLimit.sourceMethods);
        next.arithmetic=advance(setup.arithmetic,work.arithmetic,plan.setupLimit.arithmetic);
        next.comparisons=advance(setup.comparisons,work.comparisons,plan.setupLimit.comparisons);
        next.writes=advance(setup.writes,work.writes,plan.setupLimit.writes);
        next.metadataCopies=advance(setup.metadataCopies,work.metadataCopies,plan.setupLimit.metadataCopies);
        next.byteControl=advance(setup.byteControl,work.byteControl,plan.setupLimit.byteControl);
        next.formattingCalls=advance(setup.formattingCalls,work.formattingCalls,plan.setupLimit.formattingCalls);
        setup=next;
    }
    void source(std::uint64_t bytes)
    {
        auto work=setupEntry(multiply(2,bytes)); work.sourceMethods=1; take(work);
    }
    struct Scope final
    {
        Context* previous;
        explicit Scope(Context& c):previous(current) { current=&c; }
        ~Scope() { current=previous; }
    };
    void literal(std::uint64_t bytes)
    {
        auto work=setupEntry(); work.literals=1; take(work);
        const auto capacity=add(plan.literalCapacity,bytes);
        if(capacity>renderpassoutput::kMaximumTotalBytes || (admitted && capacity>admitted->literalCapacity))
            throw std::length_error("Surface source literal capacity exceeds admission");
        plan.literalCapacity=capacity;
    }
    char* stage(Stage s)
    {
        auto work=setupEntry(2*sizeof(Stage)); work.stages=1; take(work);
        if(s.scans>renderpassoutput::kMaximumTotalBytes || s.copies>renderpassoutput::kMaximumTotalBytes)
            throw std::length_error("Surface source stage byte work exceeds recipe envelope");
        Setup loopWork; loopWork.byteControl=multiply(add(add(s.scans,s.copies),1),16); take(loopWork);
        if(plan.count==Plan::maximumStages) throw std::length_error("Surface source recipe stage overflow");
        const auto capacity=add(plan.capacity,s.capacity);
        const auto scans=add(plan.scans,s.scans), copies=add(plan.copies,s.copies);
        const auto comparisons=add(plan.comparisons,s.comparisons), formatting=add(plan.formatting,s.formatting);
        if(capacity>renderpassoutput::kMaximumTotalBytes) throw std::length_error("Surface source capacity exceeds admission");
        if(admitted && (plan.count>=admitted->count || !(s==admitted->stages[plan.count])
            || capacity>admitted->capacity)) throw std::length_error("Surface source recipe differs from admission");
        char* result=arena ? arena+used : nullptr;
        plan.stages[plan.count++]=s;
        plan.capacity=capacity; plan.scans=scans; plan.copies=copies;
        plan.comparisons=comparisons; plan.formatting=formatting;
        used=static_cast<std::size_t>(capacity);
        return result;
    }
};

// Each value borrows a literal or one slice of the admitted arena. Intermediates
// never grow, move or allocate. All slices remain retained, so the sum of stage
// capacities bounds construction storage. The final buffers are copied into an
// exact compact allocation before compilation, while both allocations are charged.
class Source final
{
    Context* context;
    const char* bytes;
    std::uint64_t length, limit;
    static Context* begin(Context* context=Context::current)
    { context->source(sizeof(Source)); return context; }
    Source(const char* data,std::uint64_t actual,std::uint64_t bound)
        :context(begin()),bytes(data),length(actual),limit(bound) {}
public:
    static constexpr std::size_t npos=std::string::npos;
    Source():context(begin()),bytes(""),length(0),limit(0) {}
    Source(const Source& b):context(begin(b.context)),bytes(b.bytes),length(b.length),limit(b.limit) {}
    Source(Source&& b):Source(static_cast<const Source&>(b)) {}
    Source& operator=(const Source& b)
    { begin(b.context); context=b.context; bytes=b.bytes; length=b.length; limit=b.limit; return *this; }
    Source& operator=(Source&& b) { return *this=static_cast<const Source&>(b); }
    template<std::size_t N> Source(const char (&text)[N])
        :context(begin()),bytes(text),length(N-1),limit(N-1) { context->literal(N); }
    // Fixed pointer tables contain only status/uniform/vector tokens and the
    // optional image arguments. This is their longest token, including NUL.
    static constexpr std::size_t maximumTokenBytes=sizeof("#define GX_IMAGE , environmentTexture\n");
    static Source token(const char* token)
    {
        begin();
        Source result;
        auto* context=result.context;
        context->literal(maximumTokenBytes);
        context->stage({0,2*maximumTokenBytes,0,0,0});
        std::size_t n=0;
        while(n<maximumTokenBytes && token[n]) ++n;
        if(n==maximumTokenBytes) throw std::length_error("Surface source token exceeds recipe bound");
        result.bytes=token; result.length=result.limit=n; return result;
    }
    std::size_t size() const { begin(context); return static_cast<std::size_t>(length); }
    std::uint64_t upper() const { begin(context); return limit; }
    bool empty() const { begin(context); return limit==0; }
    const char* data() const { begin(context); return bytes; }
    const char* c_str() const { begin(context); return bytes; }
    Source& operator+=(const Source& b) { begin(context); return *this=*this+b; }
    friend Source operator+(const Source& a,const Source& b)
    {
        begin(a.context);
        const auto bound=add(a.limit,b.limit);
        auto* out=a.context->stage({add(bound,1),0,add(bound,1),0,0});
        if(out) { std::memcpy(out,a.bytes,a.length); std::memcpy(out+a.length,b.bytes,b.length); out[a.length+b.length]=0; }
        return Source(out,out ? add(a.length,b.length) : bound,bound);
    }
    std::size_t find(const Source& needle,std::size_t start=0) const
    {
        begin(context);
        context->stage({0,limit,0,multiply(limit,needle.limit),0});
        if(!context->arena) return 0; // A slice retains the whole input ceiling in planning.
        if(start>length || needle.length>length-start) return npos;
        for(auto i=start;i<=length-needle.length;++i)
            if(std::memcmp(bytes+i,needle.bytes,needle.length)==0) return i;
        return npos;
    }
    Source substr(std::size_t start,std::size_t count=npos) const
    {
        begin(context);
        auto* out=context->stage({add(limit,1),0,add(limit,1),0,0});
        if(!out) return Source(nullptr,limit,limit);
        if(start>length) throw std::length_error("Surface source slice marker missing");
        const auto n=std::min<std::uint64_t>(length-start,count);
        std::memcpy(out,bytes+start,n); out[n]=0;
        return Source(out,n,limit);
    }
    void replaceAll(const Source& from,const Source& to)
    {
        begin(context);
        if(from.limit==0) throw std::length_error("Surface source empty replacement");
        const auto growth=to.limit>from.limit ? multiply(limit/from.limit,to.limit-from.limit) : 0;
        const auto bound=add(limit,growth);
        auto* out=context->stage({add(bound,1),limit,add(bound,1),multiply(limit,from.limit),0});
        if(!out) { bytes=nullptr; length=limit=bound; return; }
        std::size_t read=0,write=0;
        while(read<length) {
            if(from.length<=length-read && std::memcmp(bytes+read,from.bytes,from.length)==0) {
                std::memcpy(out+write,to.bytes,to.length); write+=to.length; read+=from.length;
            } else out[write++]=bytes[read++];
        }
        out[write]=0; bytes=out; length=write; limit=bound;
    }
    void replace(std::size_t at,std::size_t count,const Source& text)
    {
        begin(context);
        const auto bound=add(limit,text.limit);
        auto* out=context->stage({add(bound,1),0,add(bound,1),0,0});
        if(!out) { bytes=nullptr; length=limit=bound; return; }
        if(at>length || count>length-at) throw std::length_error("Surface source replacement marker missing");
        std::memcpy(out,bytes,at); std::memcpy(out+at,text.bytes,text.length);
        std::memcpy(out+at+text.length,bytes+at+count,length-at-count);
        length=length-count+text.length; out[length]=0; bytes=out; limit=bound;
    }
    void insert(std::size_t at,const Source& text) { begin(context); replace(at,0,text); }
    template<class T> static Source number(T value)
    {
        auto* context=begin();
        Setup formatting; formatting.formattingCalls=1; context->take(formatting);
        // Integers fit 32 decimal bytes. The recipe's floating macros are in
        // [0,1], and legacy to_string formats exactly six fractional digits.
        constexpr std::size_t width=std::is_integral_v<T> ? 32 : 9;
        // formatting is a conservative scratch/output byte ceiling. One
        // bounded-width library call occurs per pass; its internals are unknown.
        auto* out=context->stage({width,0,3*width,0,2*width});
        char scratch[width]{};
        std::size_t n=0;
        if constexpr(std::is_integral_v<T>) {
            const auto r=std::to_chars(scratch,scratch+width-1,value);
            if(r.ec!=std::errc{}) throw std::length_error("Surface source integer formatting failed");
            n=r.ptr-scratch;
        } else {
            if(!std::isfinite(value)||value<0||value>1) throw std::length_error("Surface source numeric recipe changed");
            const auto r=std::snprintf(scratch,width,"%.6f",static_cast<double>(value));
            if(r<0||static_cast<std::size_t>(r)>=width) throw std::length_error("Surface source floating formatting failed");
            n=static_cast<std::size_t>(r);
        }
        if(out) std::memcpy(out,scratch,n+1);
        return Source(out,n,n);
    }
};
inline Setup restrictedSetupEnvelope()
{
    Setup limit;
    limit.literals=maximumLiterals; limit.stages=Plan::maximumStages;
    limit.sourceMethods=maximumSourceMethods;
    // Reserve both Context/Scope/Plan initialization and final preparation in
    // either pass. Also cover helper branches and the 12/13/3 token-table loops.
    const auto entries=add(add(add(limit.literals,limit.stages),limit.sourceMethods),fixedSetupEntries);
    limit.arithmetic=limit.comparisons=limit.writes=multiply(entries,setupEntryOperations);
    limit.metadataCopies=add(add(multiply(entries,setupEntryMetadataRecords*sizeof(Setup)),multiply(limit.stages,2*sizeof(Stage))),
        add(multiply(limit.sourceMethods,2*sizeof(Source)),7*sizeof(Plan)));
    // Every accepted stage has scans/copies <= the unchanged source-byte cap.
    // Sixteen scalar loop operations per visited/copied byte also cover token,
    // find and replacement loop control. Library byte-work stays in Stage.
    limit.byteControl=multiply(multiply(limit.stages,add(multiply(2,renderpassoutput::kMaximumTotalBytes),1)),16);
    limit.formattingCalls=limit.stages;
    return limit;
}
inline bool setupMatches(const Plan& plan)
{
    const auto limit=restrictedSetupEnvelope();
    if(!plan.setupLimit.within(limit)||!limit.within(plan.setupLimit)
        ||plan.metadataCopies!=multiply(2,limit.metadataCopies)) return false;
    const auto matches=[&](const Setup& used) {
        const auto entries=add(add(add(used.literals,used.stages),used.sourceMethods),fixedSetupEntries);
        const auto operations=multiply(entries,setupEntryOperations);
        const auto metadata=add(add(multiply(entries,setupEntryMetadataRecords*sizeof(Setup)),multiply(used.stages,2*sizeof(Stage))),
            add(multiply(used.sourceMethods,2*sizeof(Source)),7*sizeof(Plan)));
        return used.within(limit) && used.stages==plan.count && used.arithmetic==operations
            && used.comparisons==operations && used.writes==operations && used.metadataCopies==metadata
            && used.byteControl==multiply(add(add(plan.scans,plan.copies),plan.count),16)
            && used.formattingCalls<=used.stages;
    };
    return matches(plan.censusSetup) && matches(plan.emissionSetup)
        && plan.censusSetup.literals==plan.emissionSetup.literals
        && plan.censusSetup.formattingCalls==plan.emissionSetup.formattingCalls;
}

template<class String,class T> String number(T value)
{
    if constexpr(std::is_same_v<String,std::string>) return std::to_string(value);
    else return String::number(value);
}
template<class String,class Text> String literal(const Text& text)
{
    if constexpr(std::is_same_v<String,std::string> || std::is_array_v<Text>) return String(text);
    else if constexpr(std::is_same_v<String,Text>) return text;
    else return String::token(text);
}
template<class String> void replaceValues(String& text,const String& from,const String& to)
{
    if constexpr(std::is_same_v<String,std::string>) {
        std::size_t at=0;
        while((at=text.find(from,at))!=std::string::npos) { text.replace(at,from.size(),to); at+=to.size(); }
    } else text.replaceAll(from,to);
}
template<class String,class From,class To> void replace(String& text,const From& fromText,const To& toText)
{
    const auto from=literal<String>(fromText), to=literal<String>(toText);
    replaceValues(text,from,to);
}
struct Archive final
{
    Plan plan;
    std::unique_ptr<const char[]> arena;
    std::size_t vertexOffset=0,fragmentOffset=0,vertexLength=0,fragmentLength=0;
    std::uint64_t retainedCapacity=0;
    const char* vertex() const noexcept { return arena.get()+vertexOffset; }
    const char* fragment() const noexcept { return arena.get()+fragmentOffset; }
};
template<class Recipe> std::unique_ptr<const Archive> prepare(Recipe&& recipe,std::uint64_t ownerBytes=0,
    const Setup* setupCeilingForFixture=nullptr)
{
    const auto setupLimit=restrictedSetupEnvelope();
    // The private fixture seam can shorten a category, never grant more work.
    // Reject even a one-unit shortage before constructing either Context or
    // invoking the census recipe. Product calls always use the derived limit.
    if(setupCeilingForFixture && (!setupLimit.within(*setupCeilingForFixture)
        ||!setupCeilingForFixture->within(setupLimit)))
        throw std::length_error("Surface source setup envelope differs from restricted recipe");
    const auto metadataCapacity=add(ownerBytes,add(add(sizeof(Archive)+2*sizeof(Context)+2*sizeof(Setup),
        multiply(Plan::maximumStages,sizeof(Source)+sizeof(Stage)+16*sizeof(std::uint64_t)+8*sizeof(Setup))),
        (12+13+3)*sizeof(std::pair<const char*,const char*>)
        +3*sizeof(std::initializer_list<std::pair<const char*,const char*>>)+32));
    if(metadataCapacity>renderpassoutput::kMaximumTotalBytes)
        throw std::length_error("Surface source metadata exceeds admission");
    Context census(setupLimit);
    {
        Context::Scope scope(census);
        const auto sources=recipe();
        census.plan.vertexLimit=sources.first.upper(); census.plan.fragmentLimit=sources.second.upper();
    }
    const auto capacity=census.plan.capacity;
    census.plan.metadataCapacity=metadataCapacity;
    census.plan.metadataCopies=multiply(2,setupLimit.metadataCopies);
    census.plan.retainedCapacityLimit=add(add(census.plan.vertexLimit,census.plan.fragmentLimit),2);
    census.plan.finalCopies=census.plan.retainedCapacityLimit;
    if(add(add(add(capacity,census.plan.metadataCapacity),census.plan.literalCapacity),
        census.plan.retainedCapacityLimit)>renderpassoutput::kMaximumTotalBytes)
        throw std::length_error("Surface source owner exceeds admission");
    std::unique_ptr<char[]> arena(new char[capacity]);
    Context writer(setupLimit); writer.admitted=&census.plan; writer.arena=arena.get();
    Context::Scope scope(writer);
    const auto sources=recipe();
    if(writer.plan.count!=census.plan.count || writer.used!=capacity
        ||writer.plan.literalCapacity!=census.plan.literalCapacity
        ||writer.setup.literals!=census.setup.literals || writer.setup.stages!=census.setup.stages
        ||writer.setup.formattingCalls!=census.setup.formattingCalls
        ||sources.first.size()>census.plan.vertexLimit || sources.second.size()>census.plan.fragmentLimit)
        throw std::length_error("Surface source emission differs from admission");
    auto archive=std::make_unique<Archive>();
    archive->plan=census.plan;
    archive->vertexLength=sources.first.size(); archive->fragmentLength=sources.second.size();
    archive->fragmentOffset=add(archive->vertexLength,1);
    archive->retainedCapacity=add(add(archive->vertexLength,archive->fragmentLength),2);
    std::unique_ptr<char[]> retained(new char[archive->retainedCapacity]);
    std::memcpy(retained.get(),sources.first.data(),archive->vertexLength+1);
    std::memcpy(retained.get()+archive->fragmentOffset,sources.second.data(),archive->fragmentLength+1);
    archive->arena=std::move(retained);
    if(!census.setup.within(setupLimit)||!writer.setup.within(setupLimit)
        ||census.setup.stages!=census.plan.count||writer.setup.stages!=writer.plan.count)
        throw std::length_error("Surface source setup schedule differs from admission");
    archive->plan.censusSetup=census.setup; archive->plan.emissionSetup=writer.setup;
    if(!setupMatches(archive->plan)) throw std::length_error("Surface source retained setup differs from admission");
    return archive;
}

// This is source construction capacity, not a whole-frame reservation or native
// creation receipt. A later private owner must reserve the exposed capacity and
// authenticate the actual source/creation association. Stage equality alone
// cannot detect a same-length replacement of literal bytes.
class Census;
struct ConstructionCost final
{
    std::uint64_t metadataCapacity=0;
    Setup setup;
};
class ConstructionAttempt final
{
    friend class Census;
    friend class FrameAdmission;
public:
    enum class State { fresh, census, ready, emission, complete, failed };
    ConstructionAttempt()=default;
    ConstructionAttempt(const ConstructionAttempt&)=delete;
    ConstructionAttempt& operator=(const ConstructionAttempt&)=delete;
    ConstructionAttempt(ConstructionAttempt&&)=delete;
    ConstructionAttempt& operator=(ConstructionAttempt&&)=delete;
    State state() const noexcept { return state_; }
    const ConstructionCost& cost() const noexcept { return cost_; }
    const Setup& censusSetup() const noexcept { return censusSetup_; }
    const Setup& emissionSetup() const noexcept { return emissionSetup_; }
private:
    State state_=State::fresh;
    ConstructionCost cost_;
    Setup censusSetup_,emissionSetup_;
    const Census* token_=nullptr;
};
class Census final
{
    friend class FrameAdmission;
    const Plan plan_;
    ConstructionAttempt* const attempt_;
    Census(const Plan& plan,ConstructionAttempt& attempt):plan_(plan),attempt_(&attempt) {}
    // Caller-filled records avoid hidden value-result copies. The pointer
    // results still reserve return and caller slots, even without elision.
    static void constructionCost(ConstructionCost& cost)
    {
        cost.metadataCapacity=add(sizeof(Census)+sizeof(ConstructionAttempt),
            add(3*sizeof(std::unique_ptr<const Census>)+3*sizeof(std::unique_ptr<const Archive>),
                2*sizeof(Setup)+sizeof(std::uint64_t)+2*sizeof(const Setup*)+2*sizeof(ConstructionAttempt*)));
        // Attempt initialization, cost derivation, census entry, two fixed
        // reservations, token construction, emission entry and capacity check.
        // Both pass success/failure tails and both pointer return/caller
        // transfers have separate entries. Each uses the existing scalar and
        // Setup-record envelope; allocator and library internals stay unknown.
        constexpr std::uint64_t entries=1+1+1+2+1+1+1+2+2+2+2;
        cost.setup=setupEntry();
        cost.setup.arithmetic=cost.setup.comparisons=cost.setup.writes=multiply(entries,setupEntryOperations);
        // The added immutable Plan copy is read and written once. Four Setup
        // read/write copies cover census Plan retention, the two pass journal
        // checkpoints and the added setup-limit result transfer. Pointer
        // results allow both return and caller moves, each read and written.
        cost.setup.metadataCopies=add(multiply(entries,setupEntryMetadataRecords*sizeof(Setup)),
            add(2*sizeof(Plan),add(2*sizeof(ConstructionAttempt)+2*sizeof(ConstructionCost),
                add(8*sizeof(Setup),4*sizeof(std::unique_ptr<const Census>)+4*sizeof(std::unique_ptr<const Archive>)))));
    }
    static void fixedReservation(Setup& work,const Setup& limit)
    {
        // Charge this before Context initialization. Context records the same
        // reservation internally; the journal retains it even if census fails.
        work=setupEntry(7*sizeof(Plan));
        work.arithmetic=work.comparisons=work.writes=multiply(fixedSetupEntries,setupEntryOperations);
        work.metadataCopies=add(work.metadataCopies,multiply(fixedSetupEntries-1,setupEntryMetadataRecords*sizeof(Setup)));
        if(!work.within(limit)) throw std::length_error("Surface source fixed setup exceeds admission");
    }
public:
    Census(const Census&)=delete;
    Census& operator=(const Census&)=delete;
    Census(Census&&)=delete;
    Census& operator=(Census&&)=delete;
    const Plan& plan() const noexcept { return plan_; }
    std::uint64_t requiredCapacity() const
    { return add(add(add(plan_.capacity,plan_.retainedCapacityLimit),plan_.literalCapacity),plan_.metadataCapacity); }
    // The caller-owned journal must outlive its token and any emit call. It
    // cannot move or be reused, so failed work is never replaced by a new pass.
    template<class Recipe> static std::unique_ptr<const Census> create(Recipe&& recipe,ConstructionAttempt& attempt,
        std::uint64_t ownerBytes=0,const Setup* setupCeilingForFixture=nullptr)
    {
        if(attempt.state_!=ConstructionAttempt::State::fresh)
            throw std::length_error("Surface source construction attempt already used");
        constructionCost(attempt.cost_);
        attempt.state_=ConstructionAttempt::State::failed;
        const auto setupLimit=restrictedSetupEnvelope();
        if(setupCeilingForFixture && (!setupLimit.within(*setupCeilingForFixture)
            ||!setupCeilingForFixture->within(setupLimit)))
            throw std::length_error("Surface source setup envelope differs from restricted recipe");
        const auto metadataCapacity=add(attempt.cost_.metadataCapacity,add(ownerBytes,
            add(add(sizeof(Archive)+2*sizeof(Context)+2*sizeof(Setup),
                multiply(Plan::maximumStages,sizeof(Source)+sizeof(Stage)+16*sizeof(std::uint64_t)+8*sizeof(Setup))),
                (12+13+3)*sizeof(std::pair<const char*,const char*>)
                +3*sizeof(std::initializer_list<std::pair<const char*,const char*>>)+32)));
        if(metadataCapacity>renderpassoutput::kMaximumTotalBytes)
            throw std::length_error("Surface source metadata exceeds admission");
        fixedReservation(attempt.censusSetup_,setupLimit);
        Context context(setupLimit);
        attempt.state_=ConstructionAttempt::State::census;
        try {
            {
                Context::Scope scope(context);
                const auto sources=recipe();
                context.plan.vertexLimit=sources.first.upper(); context.plan.fragmentLimit=sources.second.upper();
            }
            context.plan.metadataCapacity=metadataCapacity;
            context.plan.metadataCopies=multiply(2,setupLimit.metadataCopies);
            context.plan.retainedCapacityLimit=add(add(context.plan.vertexLimit,context.plan.fragmentLimit),2);
            context.plan.finalCopies=context.plan.retainedCapacityLimit;
            context.plan.censusSetup=context.setup;
            if(add(add(add(context.plan.capacity,metadataCapacity),context.plan.literalCapacity),
                context.plan.retainedCapacityLimit)>renderpassoutput::kMaximumTotalBytes)
                throw std::length_error("Surface source owner exceeds admission");
            attempt.censusSetup_=context.setup;
            // Only the immutable token allocates here. No emission arena,
            // compact archive or native object exists before outer capacity.
            std::unique_ptr<const Census> token(new Census(context.plan,attempt));
            attempt.token_=token.get(); attempt.state_=ConstructionAttempt::State::ready;
            return token;
        } catch(...) {
            attempt.censusSetup_=context.setup; attempt.state_=ConstructionAttempt::State::failed;
            throw;
        }
    }
    template<class Recipe> std::unique_ptr<const Archive> emit(Recipe&& recipe,std::uint64_t constructionCapacityLimit) const
    {
        if(attempt_->state_!=ConstructionAttempt::State::ready || attempt_->token_!=this)
            throw std::length_error("Surface source census is not ready for emission");
        attempt_->state_=ConstructionAttempt::State::failed;
        // This numeric limit is only the caller's source storage prerequisite.
        // It grants no whole-frame, native, public Color or driver authority.
        if(constructionCapacityLimit>renderpassoutput::kMaximumTotalBytes || requiredCapacity()>constructionCapacityLimit)
            throw std::length_error("Surface source outer construction capacity exceeds admission");
        const auto setupLimit=restrictedSetupEnvelope();
        fixedReservation(attempt_->emissionSetup_,setupLimit);
        Context writer(setupLimit); writer.admitted=&plan_;
        attempt_->state_=ConstructionAttempt::State::emission;
        try {
            std::unique_ptr<char[]> arena(new char[plan_.capacity]); writer.arena=arena.get();
            Context::Scope scope(writer);
            const auto sources=recipe();
            if(writer.plan.count!=plan_.count || writer.used!=plan_.capacity
                ||writer.plan.literalCapacity!=plan_.literalCapacity
                ||writer.setup.literals!=plan_.censusSetup.literals || writer.setup.stages!=plan_.censusSetup.stages
                ||writer.setup.formattingCalls!=plan_.censusSetup.formattingCalls
                ||sources.first.size()>plan_.vertexLimit || sources.second.size()>plan_.fragmentLimit)
                throw std::length_error("Surface source emission differs from admission");
            auto archive=std::make_unique<Archive>();
            archive->plan=plan_;
            archive->vertexLength=sources.first.size(); archive->fragmentLength=sources.second.size();
            archive->fragmentOffset=add(archive->vertexLength,1);
            archive->retainedCapacity=add(add(archive->vertexLength,archive->fragmentLength),2);
            std::unique_ptr<char[]> retained(new char[archive->retainedCapacity]);
            std::memcpy(retained.get(),sources.first.data(),archive->vertexLength+1);
            std::memcpy(retained.get()+archive->fragmentOffset,sources.second.data(),archive->fragmentLength+1);
            archive->arena=std::move(retained);
            if(!plan_.censusSetup.within(setupLimit)||!writer.setup.within(setupLimit)
                ||plan_.censusSetup.stages!=plan_.count||writer.setup.stages!=writer.plan.count)
                throw std::length_error("Surface source setup schedule differs from admission");
            archive->plan.emissionSetup=writer.setup;
            if(!setupMatches(archive->plan)) throw std::length_error("Surface source retained setup differs from admission");
            attempt_->emissionSetup_=writer.setup; attempt_->state_=ConstructionAttempt::State::complete;
            return archive;
        } catch(...) {
            attempt_->emissionSetup_=writer.setup; attempt_->state_=ConstructionAttempt::State::failed;
            throw;
        }
    }
};
// One ledger follows every escaped creation owner. Storage retires with its
// payload; attempted source and inspection work never retires. The caller must
// include all other current frame owners in parentBytes on every reservation.
struct FrameWork final
{
    Setup setup;
    std::uint64_t scans=0,copies=0,comparisons=0,formatting=0,finalCopies=0,inspections=0;
};
class FrameAdmission final
{
public:
    static constexpr std::size_t maximumAttempts=32;
    struct Record final {
        ConstructionAttempt construction;
        std::uint64_t bytes=0;
        bool admitted=false,live=false;
    };
private:
    std::array<Record,maximumAttempts> records_{};
    FrameWork work_{},allowance_{};
    Stage sourceAllowance_{};
    std::uint64_t finalCopyAllowance_=0,retainedAllowance_=0;
    bool hasSourceAllowance_=false;
    std::uint64_t bytes_=sizeof(FrameAdmission);
    std::size_t count_=0;
    static Setup sum(const Setup& a,const Setup& b)
    {
        return {add(a.literals,b.literals),add(a.stages,b.stages),add(a.sourceMethods,b.sourceMethods),
            add(a.arithmetic,b.arithmetic),add(a.comparisons,b.comparisons),add(a.writes,b.writes),
            add(a.metadataCopies,b.metadataCopies),add(a.byteControl,b.byteControl),add(a.formattingCalls,b.formattingCalls)};
    }
public:
    static constexpr std::uint64_t censusScans=maximumLiterals*Source::maximumTokenBytes;
    static constexpr std::uint64_t censusCopies=Plan::maximumStages*3*32;
    static constexpr std::uint64_t censusFormatting=Plan::maximumStages*2*32;
    FrameAdmission()
    {
        const auto scale=[](Setup b,std::uint64_t n) {
            b.literals=multiply(b.literals,n);b.stages=multiply(b.stages,n);b.sourceMethods=multiply(b.sourceMethods,n);
            b.arithmetic=multiply(b.arithmetic,n);b.comparisons=multiply(b.comparisons,n);b.writes=multiply(b.writes,n);
            b.metadataCopies=multiply(b.metadataCopies,n);b.byteControl=multiply(b.byteControl,n);b.formattingCalls=multiply(b.formattingCalls,n);
            return b;
        };
        auto fixed=setupEntry(add(multiply(4,renderpassoutput::kMaximumTotalBytes),4*sizeof(FrameWork)+4*sizeof(Record)));
        fixed.arithmetic=fixed.comparisons=fixed.writes=multiply(64,setupEntryOperations);
        fixed.metadataCopies=add(fixed.metadataCopies,multiply(63,setupEntryMetadataRecords*sizeof(Setup)));
        ConstructionCost cost;Census::constructionCost(cost);
        auto attempt=sum(sum(scale(restrictedSetupEnvelope(),2),scale(fixed,2)),cost.setup);
        attempt.metadataCopies=add(attempt.metadataCopies,multiply(4,renderpassoutput::kMaximumTotalBytes));
        allowance_.setup=sum(scale(attempt,maximumAttempts),scale(fixed,maximumAttempts*maximumAttempts));
    }
    FrameAdmission(const FrameAdmission&)=delete;
    FrameAdmission& operator=(const FrameAdmission&)=delete;
    static std::uint64_t fixedCapacity(std::uint64_t parentBytes,std::uint64_t metadataBytes)
    { return add(add(parentBytes,sizeof(FrameAdmission)),metadataBytes); }
    std::uint64_t bytes() const noexcept { return bytes_; }
    std::size_t attempts() const noexcept { return count_; }
    const FrameWork& work() const noexcept { return work_; }
    const FrameWork& allowance() const noexcept { return allowance_; }
    const Record& record(std::size_t index) const { return records_.at(index); }
    // Reserve each full Setup pass before Context/census. This retains failed
    // setup calls, metadata transfers and byte control separately from the
    // planned byte operations. It does not count compiler/driver instructions.
    std::size_t begin(std::uint64_t parentBytes,std::uint64_t metadataBytes,std::uint64_t limit)
    {
        if(count_==maximumAttempts) throw std::length_error("Surface source frame attempt journal exhausted");
        const auto index=count_++;
        auto setup=restrictedSetupEnvelope();
        setup=sum(setup,setup);
        // Factory/result/identity and journal control have a fixed source
        // schedule. Both parameter and result/caller transfers are charged.
        auto fixed=setupEntry(add(multiply(4,metadataBytes),4*sizeof(FrameWork)+4*sizeof(Record)));
        fixed.arithmetic=fixed.comparisons=fixed.writes=multiply(64,setupEntryOperations);
        fixed.metadataCopies=add(fixed.metadataCopies,multiply(63,setupEntryMetadataRecords*sizeof(Setup)));
        ConstructionCost constructionCost; Census::constructionCost(constructionCost);
        work_.setup=sum(work_.setup,sum(sum(setup,fixed),constructionCost.setup));
        // Census has no emitting arena. Only bounded token scans and numeric
        // scratch/output work can touch bytes, so reserve those before Context.
        work_.scans=add(work_.scans,censusScans);work_.copies=add(work_.copies,censusCopies);
        work_.formatting=add(work_.formatting,censusFormatting);
        if(!work_.setup.within(allowance_.setup)||limit>renderpassoutput::kMaximumTotalBytes||parentBytes>limit||bytes_>limit-parentBytes
            ||metadataBytes>limit-parentBytes-bytes_) {
            records_[index].construction.state_=ConstructionAttempt::State::failed;
            throw std::length_error("Surface source fixed frame journal exceeds aggregate admission");
        }
        records_[index].bytes=metadataBytes; records_[index].live=true;
        bytes_=add(bytes_,metadataBytes);
        return index;
    }
    ConstructionAttempt& construction(std::size_t index) { return records_.at(index).construction; }
    void reserve(std::size_t index,const Census& census,std::uint64_t parentBytes,std::uint64_t limit,
        std::uint64_t creationSourceCopies=0)
    {
        auto& record=records_.at(index);
        if(census.attempt_!=&record.construction||!record.live||record.admitted) throw std::length_error("Surface source frame attempt is not fresh");
        const auto& p=census.plan();
        // First immutable source cardinalities derive the named allowance for
        // this frame's entire bounded attempt journal. Subsequent attempts
        // cannot enlarge any source-work category. Setup for both passes was
        // already reserved before census, including failed recipe work.
        if(creationSourceCopies>5) throw std::length_error("Surface source native input count exceeds recipe");
        if(!hasSourceAllowance_) {
            sourceAllowance_={p.capacity,p.scans,p.copies,p.comparisons,p.formatting};
            finalCopyAllowance_=p.finalCopies;retainedAllowance_=p.retainedCapacityLimit;
            const auto nativeCopies=multiply(5,p.retainedCapacityLimit);
            allowance_.scans=multiply(maximumAttempts,add(censusScans,add(multiply(2,p.scans),nativeCopies)));
            // Parent identities and owned metadata are bounded by the same
            // whole-frame storage cap. Four prospective copies per attempt.
            allowance_.copies=multiply(maximumAttempts,add(censusCopies,add(add(multiply(2,p.copies),nativeCopies),multiply(4,renderpassoutput::kMaximumTotalBytes))));
            allowance_.comparisons=add(multiply(2*maximumAttempts,p.comparisons),
                multiply(maximumAttempts*maximumAttempts,renderpassoutput::kMaximumTotalBytes));
            allowance_.formatting=multiply(maximumAttempts,add(censusFormatting,multiply(2,p.formatting)));
            allowance_.finalCopies=multiply(maximumAttempts,p.finalCopies);
            allowance_.inspections=maximumAttempts*maximumAttempts;
            hasSourceAllowance_=true;
        }
        work_.scans=add(work_.scans,multiply(2,p.scans));
        work_.copies=add(work_.copies,multiply(2,p.copies));
        work_.comparisons=add(work_.comparisons,multiply(2,p.comparisons));
        work_.formatting=add(work_.formatting,multiply(2,p.formatting));
        work_.finalCopies=add(work_.finalCopies,p.finalCopies);
        const auto nativeSourceBytes=multiply(creationSourceCopies,p.retainedCapacityLimit);
        work_.copies=add(work_.copies,nativeSourceBytes);work_.scans=add(work_.scans,nativeSourceBytes);
        if(p.capacity>sourceAllowance_.capacity||p.scans>sourceAllowance_.scans||p.copies>sourceAllowance_.copies
            ||p.comparisons>sourceAllowance_.comparisons||p.formatting>sourceAllowance_.formatting
            ||p.finalCopies>finalCopyAllowance_||p.retainedCapacityLimit>retainedAllowance_
            ||work_.scans>allowance_.scans||work_.copies>allowance_.copies||work_.comparisons>allowance_.comparisons
            ||work_.formatting>allowance_.formatting||work_.finalCopies>allowance_.finalCopies)
            throw std::length_error("Surface source frame work exceeds immutable census allowance");
        const auto bytes=add(census.requiredCapacity(),nativeSourceBytes);
        if(limit>renderpassoutput::kMaximumTotalBytes||parentBytes>limit||bytes_>limit-parentBytes
            ||bytes>limit-parentBytes-bytes_)
            throw std::length_error("Surface source whole frame reservation exceeds aggregate admission");
        bytes_=add(bytes_,bytes); record.bytes=add(record.bytes,bytes); record.admitted=true;
    }
    bool admitted(std::size_t index,const Census& census) const
    {
        const auto& r=records_.at(index);
        return census.attempt_==&r.construction&&r.live&&r.admitted&&r.bytes>=census.requiredCapacity()
            &&r.construction.state()==ConstructionAttempt::State::ready;
    }
    void factory(const Setup& work)
    {
        work_.setup=sum(work_.setup,work);
        if(!work_.setup.within(allowance_.setup)) throw std::length_error("Surface source factory setup exceeds frame admission");
    }
    void metadata(std::uint64_t copies)
    {
        work_.copies=add(work_.copies,copies);
        work_.setup.metadataCopies=add(work_.setup.metadataCopies,copies);
        if(!work_.setup.within(allowance_.setup)||copies>multiply(4,renderpassoutput::kMaximumTotalBytes)
            ||(hasSourceAllowance_&&work_.copies>allowance_.copies))
            throw std::length_error("Surface source aggregate identity copies exceed admission");
    }
    // Called only after arena/archive/native owners are destroyed. The record
    // and every attempted-work category persist for the ledger's lifetime.
    void fail(std::size_t index) noexcept
    {
        auto& attempt=records_[index].construction;
        if(attempt.state_!=ConstructionAttempt::State::complete) attempt.state_=ConstructionAttempt::State::failed;
    }
    void retire(std::size_t index) noexcept
    {
        auto& r=records_[index];
        if(r.live) { bytes_-=r.bytes; r.live=false; }
    }
    void inspect(std::uint64_t bytes,std::uint64_t metadataBytes=0)
    {
        // An inspection is charged before looking at source/native/parent
        // identity. No reconstruction string or success token is allocated.
        if(!hasSourceAllowance_||work_.inspections==allowance_.inspections)
            throw std::length_error("Surface source frame inspection journal exhausted");
        ++work_.inspections;
        work_.comparisons=add(work_.comparisons,bytes);
        if(bytes>renderpassoutput::kMaximumTotalBytes||work_.comparisons>allowance_.comparisons)
            throw std::length_error("Surface source aggregate inspection work exceeds admission");
        auto fixed=setupEntry(add(4*sizeof(FrameWork),metadataBytes));
        fixed.arithmetic=fixed.comparisons=fixed.writes=multiply(64,setupEntryOperations);
        fixed.metadataCopies=add(fixed.metadataCopies,multiply(63,setupEntryMetadataRecords*sizeof(Setup)));
        work_.setup=sum(work_.setup,fixed);
        if(metadataBytes>multiply(4,renderpassoutput::kMaximumTotalBytes)||!work_.setup.within(allowance_.setup))
            throw std::length_error("Surface source aggregate inspection setup exceeds admission");
    }
};
// This overload is the typed outer seam. Ordinary prepare retains its exact
// local refusal/API. A token from a different ledger/attempt cannot emit here.
template<class Recipe> std::unique_ptr<const Archive> prepare(Recipe&& recipe,FrameAdmission& frame,
    std::size_t attempt,std::uint64_t parentBytes,std::uint64_t aggregateLimit,std::uint64_t ownerBytes=0,
    std::uint64_t creationSourceCopies=0)
{
    const auto census=Census::create(recipe,frame.construction(attempt),ownerBytes);
    try {
        frame.reserve(attempt,*census,parentBytes,aggregateLimit,creationSourceCopies);
        if(!frame.admitted(attempt,*census)) throw std::length_error("Surface source frame admission association changed");
        return census->emit(recipe,census->requiredCapacity());
    } catch(...) { frame.fail(attempt);throw; }
}
} // namespace arbitgpu::surfacesource
