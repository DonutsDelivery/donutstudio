#pragma once

#include "../../src/gpu_backend/fixture_surface_complete_base_admission.h"
#include "../../src/gpu_backend/metal_glass_caustic_consumer.h"
#include "../../src/gpu_backend/metal_lambertian_dispatch.h"

#include "fixture_scene.h"
#include "../../src/gpu_backend/fixture_surface_source_admission.h"
#include "../../src/gpu_backend/fixture_environment_image.h"

#include <limits>

namespace arbitgpu
{
bool prepareMetalPrivateSurfaceSourceForFixture(const std::shared_ptr<const NativeFixtureSceneResources>&,
    HarmonicMIDI::grid::SceneObjectId,std::string&);
bool metalPrivateSurfaceSourceCreationCasesForFixture(const std::shared_ptr<const NativeFixtureSceneResources>&,
    HarmonicMIDI::grid::SceneObjectId,std::string&);
bool inspectMetalSurfaceSourceAdmissionForFixture(const std::shared_ptr<const NativeFixtureSceneResources>&,std::string&);
}
namespace videohelper::tests::restrictedsurfacesource
{
inline bool legacyRecipeCases(std::string& error)
{
    for(const bool metal:{false,true}) {
        const auto sky=arbitgpu::analyticsky::shader(metal);
        const auto image=arbitgpu::environmentimage::shader(metal);
        const auto query=arbitgpu::smoothglass::queryShader(metal);
        const auto vector=metal ? "float3" : "vec3";
        if(sky.find(std::string(vector)+" analyticSkyBase(")==std::string::npos
            ||image.find(std::string(vector)+" environmentImageBase(")==std::string::npos
            ||image.find("analyticSkyRadiance(")!=std::string::npos
            ||query.find("GlassClosestHit glassSceneClosestHit(")==std::string::npos
            ||query.find("opaque")!=std::string::npos)
        { error="Legacy source factoring changed sky/image/query spelling or substitution";return false; }
        for(const bool imageSecondary:{false,true}) for(const bool selected:{false,true}) {
            const auto transport=arbitgpu::smoothglass::transportShader(metal,imageSecondary,selected);
            if(transport.find("#define GLASS_ROUGH_UNRESOLVED 0.000100\n")==std::string::npos
                ||transport.find("#define GLASS_ROUGH_CONSERVATION 0.000020\n")==std::string::npos
                ||transport.find("#define GLASS_ROUGH_STATUS_CLOSURE -10\n")==std::string::npos
                ||transport.find("#define GLASS_CAMERA_V2_COMPLETE 33\n")==std::string::npos
                ||(transport.find("roughGlassTransportV2Selected(")!=std::string::npos)!=selected
                ||(transport.find("environmentImageRadiance(direction,")!=std::string::npos)!=imageSecondary
                ||(transport.find("thread float3& ")!=std::string::npos)!=metal)
            { error="Legacy transport factoring changed numeric macros or configuration";return false; }
        }
    }
    error.clear();return true;
}
inline bool constructionCases(std::string& error)
{
    using namespace arbitgpu::surfacesource;
    try {
        const auto archive=prepare([] {
            Source vertex="aaaa";
            vertex.replaceAll("aa","aaa");
            Source fragment="head:tail";
            fragment=fragment.substr(fragment.find("tail"));
            fragment.insert(0,"prefix:");
            return std::make_pair(vertex,fragment);
        });
        if(std::string(archive->vertex())!="aaaaaa"||std::string(archive->fragment())!="prefix:tail"
            ||archive->vertexLength!=6||archive->fragmentLength!=11
            ||archive->plan.vertexLimit<6||archive->plan.fragmentLimit<11)
        { error="Surface source overlap/slice emission differs from authored bytes"; return false; }
        if(!setupMatches(archive->plan)||archive->plan.censusSetup.literals==0
            ||archive->plan.censusSetup.sourceMethods==0||archive->plan.censusSetup.byteControl==0
            ||archive->plan.metadataCopies<=7*sizeof(Plan))
        { error="Surface source retained setup omits construction metadata or control"; return false; }
        std::uint64_t capacity=0;
        for(std::size_t i=0;i<archive->plan.count;++i) capacity=add(capacity,archive->plan.stages[i].capacity);
        if(capacity!=archive->plan.capacity||archive->vertexOffset+7>archive->retainedCapacity||archive->fragmentOffset+12>archive->retainedCapacity
            ||archive->retainedCapacity!=19||archive->retainedCapacity>archive->plan.retainedCapacityLimit)
        { error="Surface source retained stage capacity does not cover actual bytes"; return false; }
        bool writerVisited=false,refused=false;
        try {
            prepare([&] {
                writerVisited=Context::current->arena!=nullptr;
                Source a="a"; a+=Source("b");
                return std::make_pair(a,a);
            },std::numeric_limits<std::uint64_t>::max());
        } catch(const std::length_error&) { refused=true; }
        if(!refused||writerVisited) { error="Surface source overflow reached allocating emission"; return false; }
        std::size_t passes=0;
        refused=false;
        try {
            prepare([&] {
                ++passes;
                Source a="a";
                a+=passes==1 ? Source("b") : Source("different");
                return std::make_pair(a,a);
            });
        } catch(const std::length_error&) { refused=true; }
        if(!refused||passes!=2) { error="Surface source changed recipe escaped stage admission"; return false; }
        const auto formatted=prepare([] {
            return std::make_pair(Source::number(42),Source::number(0.25));
        });
        if(std::string(formatted->vertex())!="42"||std::string(formatted->fragment())!="0.250000"
            ||!setupMatches(formatted->plan)||formatted->plan.censusSetup.formattingCalls!=2
            ||formatted->plan.emissionSetup.formattingCalls!=2||formatted->plan.formatting!=2*(32+9))
        { error="Surface source formatting call/output-width admission differs"; return false; }
        // A shortage in any derived category refuses before recipe evaluation,
        // including the census pass. The ceiling cannot enlarge admission.
        const std::array<std::uint64_t Setup::*,9> categories{
            &Setup::literals,&Setup::stages,&Setup::sourceMethods,&Setup::arithmetic,
            &Setup::comparisons,&Setup::writes,&Setup::metadataCopies,&Setup::byteControl,&Setup::formattingCalls};
        for(const auto category:categories) {
            auto shortLimit=restrictedSetupEnvelope();
            --(shortLimit.*category);
            bool censusVisited=false;
            refused=false;
            try {
                prepare([&] {
                    censusVisited=true;
                    return std::make_pair(Source("a"),Source("b"));
                },0,&shortLimit);
            } catch(const std::length_error&) { refused=true; }
            if(!refused||censusVisited) { error="Surface source short setup category reached census"; return false; }
        }
        // Exercise method admission as well as the prepare entry check. No
        // failing category reaches literal/stage metadata or a byte operation.
        for(const auto category:categories) {
            Context context{restrictedSetupEnvelope()};
            Context::Scope scope(context);
            context.plan.setupLimit.*category=context.setup.*category;
            refused=false;
            try {
                if(category==&Setup::literals) context.literal(1);
                else if(category==&Setup::sourceMethods||category==&Setup::formattingCalls) Source::number(1);
                else context.stage({1,1,1,1,0});
            } catch(const std::length_error&) { refused=true; }
            if(!refused||context.plan.literalCapacity!=0||context.plan.count!=0||context.used!=0)
            { error="Surface source exhausted setup category reached method work"; return false; }
        }
    } catch(const std::exception& exception) { error=exception.what(); return false; }
    error.clear();return true;
}
inline bool censusCases(std::string& error)
{
    using namespace arbitgpu::surfacesource;
    static_assert(std::is_same_v<decltype(std::declval<const Census&>().plan()),const Plan&>);
    static_assert(!std::is_copy_constructible_v<Census> && !std::is_move_constructible_v<Census>);
    static_assert(!std::is_copy_constructible_v<ConstructionAttempt> && !std::is_move_constructible_v<ConstructionAttempt>);
    try {
        ConstructionAttempt attempt;
        std::size_t passes=0;
        bool writerVisited=false,fixedCharged=true;
        const auto recipe=[&] {
            ++passes;
            writerVisited=writerVisited || Context::current->arena!=nullptr;
            const auto& journal=Context::current->arena ? attempt.emissionSetup() : attempt.censusSetup();
            fixedCharged=fixedCharged && journal.arithmetic>0 && journal.metadataCopies>=7*sizeof(Plan);
            Source vertex="aaaa"; vertex.replaceAll("aa","aaa");
            Source fragment="head:tail";
            fragment=fragment.substr(fragment.find("tail")); fragment.insert(0,"prefix:");
            return std::make_pair(vertex,fragment);
        };
        const auto token=Census::create(recipe,attempt);
        if(passes!=1||writerVisited||!fixedCharged||attempt.state()!=ConstructionAttempt::State::ready
            ||token->plan().emissionSetup.sourceMethods!=0||token->plan().censusSetup.sourceMethods==0)
        { error="Surface source census allocated emission or omitted its prospective fixed journal"; return false; }
        const auto required=token->requiredCapacity();
        const auto& plan=token->plan();
        if(required!=add(add(add(plan.capacity,plan.retainedCapacityLimit),plan.literalCapacity),plan.metadataCapacity)
            ||attempt.cost().metadataCapacity<sizeof(Census)+sizeof(ConstructionAttempt)
            ||attempt.cost().setup.metadataCopies<2*sizeof(Plan)
            ||attempt.cost().setup.arithmetic==0)
        { error="Surface source census omits simultaneous storage or token transfer setup"; return false; }
        const auto archive=token->emit(recipe,required);
        const auto legacy=prepare([] {
            Source vertex="aaaa"; vertex.replaceAll("aa","aaa");
            Source fragment="head:tail";
            fragment=fragment.substr(fragment.find("tail")); fragment.insert(0,"prefix:");
            return std::make_pair(vertex,fragment);
        });
        if(passes!=2||!writerVisited||!fixedCharged||attempt.state()!=ConstructionAttempt::State::complete
            ||std::string(archive->vertex())!=legacy->vertex()||std::string(archive->fragment())!=legacy->fragment()
            ||archive->plan.capacity!=legacy->plan.capacity||archive->retainedCapacity!=legacy->retainedCapacity
            ||archive->plan.metadataCapacity!=add(legacy->plan.metadataCapacity,attempt.cost().metadataCapacity)
            ||archive->plan.metadataCopies!=legacy->plan.metadataCopies||!setupMatches(archive->plan)
            ||!archive->plan.censusSetup.within(legacy->plan.censusSetup)||!legacy->plan.censusSetup.within(archive->plan.censusSetup)
            ||!archive->plan.emissionSetup.within(legacy->plan.emissionSetup)||!legacy->plan.emissionSetup.within(archive->plan.emissionSetup))
        { error="Surface source separated census changed legacy bytes or setup"; return false; }
        const auto expectedCapacity=add(add(add(legacy->plan.capacity,legacy->plan.retainedCapacityLimit),
            legacy->plan.literalCapacity),add(legacy->plan.metadataCapacity,attempt.cost().metadataCapacity));
        if(required!=expectedCapacity||archive->plan.count!=legacy->plan.count)
        { error="Surface source census changed the legacy simultaneous storage recipe"; return false; }
        for(std::size_t i=0;i<archive->plan.count;++i) {
            if(!(archive->plan.stages[i]==legacy->plan.stages[i]))
            { error="Surface source separated census changed a legacy stage"; return false; }
        }
        bool refused=false;
        try { token->emit(recipe,required); } catch(const std::length_error&) { refused=true; }
        if(!refused||passes!=2||attempt.state()!=ConstructionAttempt::State::complete)
        { error="Surface source immutable census allowed a second emission"; return false; }
        for(const bool unsupported:{false,true}) {
            ConstructionAttempt shortAttempt;
            std::size_t shortPasses=0;
            const auto shortRecipe=[&] { ++shortPasses; return std::make_pair(Source("a"),Source("b")); };
            const auto shortToken=Census::create(shortRecipe,shortAttempt);
            refused=false;
            try { shortToken->emit(shortRecipe,unsupported ? std::numeric_limits<std::uint64_t>::max() : shortToken->requiredCapacity()-1); }
            catch(const std::length_error&) { refused=true; }
            if(!refused||shortPasses!=1||shortAttempt.state()!=ConstructionAttempt::State::failed
                ||shortAttempt.censusSetup().sourceMethods==0||shortAttempt.emissionSetup().sourceMethods!=0
                ||shortAttempt.cost().setup.metadataCopies==0)
            { error="Surface source outer shortage reached writer or erased attempted work"; return false; }
        }
        ConstructionAttempt changedAttempt;
        std::size_t changedPasses=0;
        const auto changedRecipe=[&] {
            ++changedPasses;
            Source a="a"; a+=changedPasses==1 ? Source("b") : Source("different");
            return std::make_pair(a,a);
        };
        const auto changed=Census::create(changedRecipe,changedAttempt);
        refused=false;
        try { changed->emit(changedRecipe,changed->requiredCapacity()); } catch(const std::length_error&) { refused=true; }
        if(!refused||changedPasses!=2||changedAttempt.state()!=ConstructionAttempt::State::failed
            ||changedAttempt.censusSetup().sourceMethods==0||changedAttempt.emissionSetup().sourceMethods==0)
        { error="Surface source mismatched census escaped stage admission or lost failed writer work"; return false; }
        for(const bool emission:{false,true}) {
            ConstructionAttempt failedAttempt;
            std::size_t failedPasses=0;
            const auto failedRecipe=[&] {
                ++failedPasses;
                Source a="a"; a+=Source("b");
                if(!emission || failedPasses==2) throw std::length_error("Authored source construction failure");
                return std::make_pair(a,a);
            };
            refused=false;
            try {
                const auto failedToken=Census::create(failedRecipe,failedAttempt);
                failedToken->emit(failedRecipe,failedToken->requiredCapacity());
            } catch(const std::length_error&) { refused=true; }
            if(!refused||failedAttempt.state()!=ConstructionAttempt::State::failed
                ||failedAttempt.censusSetup().stages==0||failedAttempt.cost().setup.metadataCopies==0
                ||(emission && failedAttempt.emissionSetup().stages==0))
            { error="Surface source failed recipe lost its prospective construction journal"; return false; }
            const auto before=failedAttempt.censusSetup();
            try { Census::create(failedRecipe,failedAttempt); refused=false; } catch(const std::length_error&) {}
            if(!refused||failedAttempt.censusSetup().metadataCopies!=before.metadataCopies)
            { error="Surface source reused failure erased the census journal"; return false; }
        }
        // Both arithmetic overflow and the unchanged local 2 GiB cap refuse
        // before census source work, without any large allocation.
        for(const auto ownerBytes:{std::numeric_limits<std::uint64_t>::max(),renderpassoutput::kMaximumTotalBytes}) {
            ConstructionAttempt oversized;
            bool visited=false;
            refused=false;
            try { Census::create([&] { visited=true; return std::make_pair(Source("a"),Source("b")); },oversized,ownerBytes); }
            catch(const std::length_error&) { refused=true; }
            if(!refused||visited||oversized.state()!=ConstructionAttempt::State::failed||oversized.cost().setup.metadataCopies==0)
            { error="Surface source outer metadata overflow reached census"; return false; }
        }
        const std::array<std::uint64_t Setup::*,9> categories{
            &Setup::literals,&Setup::stages,&Setup::sourceMethods,&Setup::arithmetic,
            &Setup::comparisons,&Setup::writes,&Setup::metadataCopies,&Setup::byteControl,&Setup::formattingCalls};
        for(const auto category:categories) {
            ConstructionAttempt shortSetup;
            auto limit=restrictedSetupEnvelope(); --(limit.*category);
            bool visited=false;
            refused=false;
            try { Census::create([&] { visited=true; return std::make_pair(Source("a"),Source("b")); },shortSetup,0,&limit); }
            catch(const std::length_error&) { refused=true; }
            if(!refused||visited||shortSetup.state()!=ConstructionAttempt::State::failed||shortSetup.cost().setup.metadataCopies==0)
            { error="Surface source separated census accepted a short setup category"; return false; }
        }
    } catch(const std::exception& exception) { error=exception.what(); return false; }
    error.clear(); return true;
}
inline bool frameCases(std::string& error)
{
    using namespace arbitgpu::surfacesource;
    try {
        FrameAdmission ledger;
        constexpr std::uint64_t parent=4096,metadata=1024;
        const auto fixed=FrameAdmission::fixedCapacity(parent,metadata);
        if(fixed!=parent+sizeof(FrameAdmission)+metadata)
        {error="Surface frame fresh fixed capacity omits metadata or journal";return false;}
        for(const auto limit:{parent+sizeof(FrameAdmission),fixed-1}) {
            FrameAdmission shortLedger;
            bool refused=false;
            try {shortLedger.begin(parent,metadata,limit);}
            catch(const std::length_error&) {refused=true;}
            if(!refused||shortLedger.bytes()!=sizeof(FrameAdmission)||shortLedger.attempts()!=1
                ||shortLedger.record(0).live||shortLedger.record(0).construction.state()!=ConstructionAttempt::State::failed
                ||shortLedger.record(0).construction.censusSetup().sourceMethods
                ||!shortLedger.work().setup.metadataCopies)
            {error="Surface frame fresh fixed shortage reached census or erased begin work";return false;}
        }
        bool overflow=false;
        try {FrameAdmission::fixedCapacity(std::numeric_limits<std::uint64_t>::max(),metadata);}
        catch(const std::length_error&) {overflow=true;}
        if(!overflow) {error="Surface frame fresh fixed capacity accepted parent overflow";return false;}
        overflow=false;
        try {FrameAdmission::fixedCapacity(parent,std::numeric_limits<std::uint64_t>::max());}
        catch(const std::length_error&) {overflow=true;}
        if(!overflow) {error="Surface frame fresh fixed capacity accepted metadata overflow";return false;}
        std::size_t passes=0;
        const auto recipe=[&] {++passes;Source v="aaaa";v.replaceAll("aa","aaa");return std::make_pair(v,Source("tail"));};
        const auto attempt=ledger.begin(parent,metadata,renderpassoutput::kMaximumTotalBytes);
        auto census=Census::create(recipe,ledger.construction(attempt));
        if(passes!=1||ledger.work().setup.sourceMethods<2*restrictedSetupEnvelope().sourceMethods)
        {error="Surface frame census omitted prospective two-pass setup";return false;}
        const auto exact=add(parent,add(ledger.bytes(),census->requiredCapacity()));
        ledger.reserve(attempt,*census,parent,exact);
        auto archive=census->emit(recipe,census->requiredCapacity());
        if(passes!=2||ledger.admitted(attempt,*census)
            ||ledger.construction(attempt).state()!=ConstructionAttempt::State::complete)
        {error="Surface frame outer admission did not complete its exact census";return false;}
        const auto retained=ledger.bytes(),charge=ledger.record(attempt).bytes;
        const auto failedWork=ledger.work().copies;
        const auto shortAttempt=ledger.begin(parent,metadata,renderpassoutput::kMaximumTotalBytes);
        const auto beforePasses=passes;
        bool refused=false;
        try {prepare(recipe,ledger,shortAttempt,parent,add(parent,add(retained,charge))-1);}
        catch(const std::length_error&) {refused=true;}
        if(!refused||passes!=beforePasses+1||ledger.record(shortAttempt).admitted
            ||ledger.construction(shortAttempt).state()!=ConstructionAttempt::State::failed
            ||ledger.construction(shortAttempt).emissionSetup().sourceMethods||ledger.work().copies<=failedWork)
        {error="Surface frame aggregate shortage reached emission or lost failed work";return false;}
        ledger.retire(shortAttempt);
        const auto work=ledger.work().copies;
        if(ledger.bytes()!=retained) {error="Surface frame failed unescaped payload did not retire exact storage";return false;}
        FrameAdmission other;
        const auto foreign=other.begin(parent,metadata,renderpassoutput::kMaximumTotalBytes);
        refused=false;
        try {other.reserve(foreign,*census,parent,renderpassoutput::kMaximumTotalBytes);}
        catch(const std::length_error&) {refused=true;}
        other.retire(foreign);
        if(!refused) {error="Surface frame accepted another ledger's census token";return false;}
        // The archive remains owned until this explicit destruction. Source
        // storage cannot be reused while an escaped payload survives.
        archive.reset();census.reset();ledger.retire(attempt);
        if(ledger.bytes()!=sizeof(FrameAdmission)||ledger.work().copies!=work)
        {error="Surface frame payload retirement erased work or leaked storage";return false;}
    } catch(const std::exception& exception) {error=exception.what();return false;}
    error.clear();return true;
}
inline bool completeBaseAdmissionCases(std::string& error)
{
    namespace cb=arbitgpu::completebase;
    try {
        cb::Admission ledger(1);const auto fixed=ledger.bytes();
        if(cb::Admission::fixedBytes(4096,1024)!=4096+fixed+1024||!ledger.inspectJournal())return false;
        const cb::Costs work{1,2,3,4,5,1,6,7,8,9,10,11};
        const auto slot=ledger.begin(1,4096,1024,work,4096+fixed+1024);
        ledger.inspect(slot);ledger.registry();ledger.publish(slot);
        if(!ledger.record(slot).live||!ledger.record(slot).published||!ledger.inspectJournal())return false;
        const auto retained=ledger.bytes();const auto before=ledger.work();bool refused=false;
        try {ledger.begin(2,4096,1024,work,4096+retained+1023);}catch(const std::length_error&){refused=true;}
        if(!refused||ledger.bytes()!=retained||ledger.attempts()!=2||!ledger.inspectJournal()
            ||ledger.work()==before||ledger.record(1).live)return false;
        refused=false;
        try {cb::Admission::fixedBytes(UINT64_MAX,1);}catch(const std::length_error&){refused=true;}
        if(!refused)return false;
        refused=false;
        try {ledger.begin(3,UINT64_MAX,1,work,renderpassoutput::kMaximumTotalBytes);}catch(const std::length_error&){refused=true;}
        if(!refused||ledger.attempts()!=3||ledger.bytes()!=retained||!ledger.inspectJournal())return false;
        const auto failedWork=ledger.work();
        // Retiring payload storage preserves failed work and inspection units.
        ledger.retire(slot);
        if(ledger.bytes()!=fixed||!(ledger.work()==failedWork)||!ledger.inspectJournal())return false;
        const auto retry=ledger.begin(4,4096,1024,work,4096+fixed+1024);
        if(retry!=3||ledger.record(retry).serial!=4||!ledger.inspectJournal())return false;
        ledger.retire(retry);
    }catch(const std::exception& exception){error=exception.what();return false;}
    error.clear();return true;
}
inline bool nativeCompleteBaseCases(const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame,
    const std::shared_ptr<const arbitgpu::MetalFiniteGratingReceiverResult>& receiver,std::string& error)
{
    using Result=arbitgpu::MetalRestrictedCompleteBaseResult;
    std::shared_ptr<const Result> result,escaped;
    std::array<std::uint64_t,5> before{},after{};std::array<std::uint64_t,12> work{},next{};
    const auto storage=[&](bool newer){return arbitgpu::readMetalRestrictedCompleteBaseStorageForFixture(frame,
        newer?after:before,newer?next:work,error);};
    const auto run=[&](int stop=-1,std::uint64_t limit=renderpassoutput::kMaximumTotalBytes){
        return arbitgpu::produceMetalRestrictedCompleteBaseForFixture(frame,receiver,result,error,stop,nullptr,limit);};
    if(!storage(false)||run(-1,0)||!storage(true)||after[0]!=before[0]+1||after[1]!=before[1]
        ||after[2]!=before[2]||result||next==work){error="Complete base fixed shortage allocated its attempt registry";return false;}
    for(int stop:{0,1,3}) {
        if(!storage(false)||run(stop)||result||!storage(true)||after[1]!=before[1]+1
            ||after[4]!=0||next==work){error="Complete base failed native preparation lost permanent attempted work";return false;}
    }
    if(!arbitgpu::produceMetalRestrictedCompleteBaseForFixture(frame,receiver,result,error,2,&escaped)
        &&escaped) {
        if(!storage(false)||before[4]!=1||arbitgpu::inspectMetalRestrictedCompleteBaseForFixture(frame,escaped,error))return false;
    }else {error="Complete base refused publication did not retain its escaped native owner";return false;}
    if(!run()||!result||!arbitgpu::inspectMetalRestrictedCompleteBaseForFixture(frame,result,error)
        ||!storage(true)||after[4]!=2)return false;
    const auto saved=result;const auto reservation=after[2]-before[2];
    // Exact simultaneous boundary admits a fresh retry; one byte less refuses
    // with the escaped owner and incumbent still alive and fully charged.
    const auto shortLimit=arbitgpu::surfacesource::add(after[3],arbitgpu::surfacesource::add(after[2],reservation))-1;
    if(!storage(false)||run(-1,shortLimit)||result!=saved||!storage(true)
        ||after[2]!=before[2]||after[1]!=before[1]+1||next==work)return false;
    const auto exactLimit=shortLimit+1;
    if(!run(-1,exactLimit)||result==saved||!arbitgpu::inspectMetalRestrictedCompleteBaseForFixture(frame,saved,error)
        ||!arbitgpu::inspectMetalRestrictedCompleteBaseForFixture(frame,result,error)
        ||!arbitgpu::inspectMetalRestrictedCompleteBaseRefusalsForFixture(frame,result,error))return false;
    if(!storage(false)||before[4]!=3)return false;
    // The payload's last native/parent member dies before reservation refund.
    escaped.reset();
    if(!storage(true)||after[4]!=2||before[2]-after[2]!=reservation)return false;
    if(frame->colorImageHandle()||frame->colorTextureViewHandle()||frame->depthImageHandle()||frame->depthTextureViewHandle())return false;
    if(arbitgpu::inspectMetalRestrictedCompleteBaseForFixture({},result,error))return false;
    error.clear();return true;
}
// These cases consume a genuine published metric radiance carrier. Host
// values below are independent numeric oracles, never native authority.
inline bool nativeFiniteRgbCases(const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame,
    const std::shared_ptr<const arbitgpu::MetalFiniteGratingRadianceResult>& carrier,
    const std::array<std::uint32_t,arbitgpu::kMetalFiniteReceiverDiagnosticWords>& receiverWords,bool zero,bool extensive,
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& foreignFrame,
    const std::shared_ptr<const arbitgpu::MetalFiniteGratingRadianceResult>& foreignCarrier,
    std::shared_ptr<const arbitgpu::MetalFiniteGratingRgbResult>& foreignRgb,std::array<float,4>& rgb,std::string& error)
{
    using namespace arbitgpu;
    const auto fail=[&](const char* text){error=text;return false;};
    std::shared_ptr<const MetalFiniteGratingRgbResult> result,escaped;
    std::array<std::uint64_t,9> before{},after{};finitergb::Costs work{},next{};
    const auto storage=[&](bool newer){return readMetalFiniteGratingRgbStorageForFixture(frame,newer?after:before,newer?next:work,error);};
    const auto run=[&](int stop=-1,std::uint64_t limit=renderpassoutput::kMaximumTotalBytes){
        return produceMetalFiniteGratingRgbForFixture(frame,carrier,result,error,stop,nullptr,limit);};
    if(!storage(false)||before[1]||before[2]||!before[7]||!storage(true)||after!=before||next==work)
        return fail("Finite RGB absent ledger read lost permanent inspection work");
    for(const auto limit:{std::uint64_t(0),surfacesource::add(before[3],surfacesource::add(before[6],before[7]))-1}){
        if(!storage(false)||run(-1,limit)||result||!storage(true)||after[0]!=before[0]+1||after[1]!=before[1]
            ||after[2]!=before[2]||after[8]!=before[8]||next==work)return fail("Finite RGB pre-heap C/W shortage allocated or dispatched");}
    if(!storage(false)||run(9)||result||!storage(true)||after[0]!=before[0]+1||after[1]||after[2]||after[8]||next==work)
        return fail("Finite RGB first ledger allocation failure reset attempted work or gained storage");
    for(const int stop:{0,3,4,1,5,11}){
        const auto frameOwners=frame.use_count(),parentOwners=carrier.use_count();
        if(!storage(false)||run(stop)||result||!storage(true)||after[0]!=before[0]+1||after[1]!=before[1]+1||after[4]
            ||after[8]!=before[8]||after[5]||next==work||frame.use_count()!=frameOwners||carrier.use_count()!=parentOwners)
            return fail("Finite RGB construction/native/pre-dispatch failure lost retirement or permanent work");}
    if(!storage(false)||run(6)||result||!storage(true)||after[8]!=before[8]+1||after[4]||after[5]||next==work)
        return fail("Finite RGB post-completion failure lost native retirement or attempted dispatch work");
    if(foreignFrame){
        result=foreignRgb;
        if(!storage(false)||run()||result!=foreignRgb||!storage(true)||after[1]!=before[1]||after[2]!=before[2]||after[8]!=before[8])
            return fail("Finite RGB foreign incumbent received storage credit");
        result.reset();
        if(produceMetalFiniteGratingRgb(frame,foreignCarrier,result,error)||result
            ||inspectMetalFiniteGratingRgb(frame,foreignRgb,error))return fail("Finite RGB foreign carrier/frame gained authority");
    }
    if(!storage(false)||!metalFiniteGratingRgbNativeRefusalProbesForFixture(frame,carrier,error)
        ||!storage(true)||after[8]!=before[8]+3||after[4]||after[5]||next==work)
        return fail("Finite RGB native control/nonfinite/range refusal lost diagnostics or permanent dispatch work");
    if(!storage(false)||produceMetalFiniteGratingRgbForFixture(frame,carrier,result,error,2,&escaped)||result||!escaped
        ||!storage(true)||after[4]!=before[4]+1||after[5]<=before[5]||after[8]!=before[8]+1
        ||inspectMetalFiniteGratingRgb(frame,escaped,error))return fail("Finite RGB unpublished escape lost native residency or gained authority");
    auto untouched=std::array<float,4>{17,18,19,20};
    if(readMetalFiniteGratingRgbForFixture(frame,escaped,untouched,error)||untouched!=std::array<float,4>{17,18,19,20})
        return fail("Finite RGB failed read changed caller data");
    const auto alias=[&](){const auto incumbent=result;if(!storage(false)
            ||produceMetalFiniteGratingRgbForFixture(frame,carrier,result,error,2,&result)||result!=incumbent
            ||!storage(true)||after!=before)return false;return true;};
    if(!alias())return fail("Finite RGB null alias mutated output or admission");
    const std::weak_ptr<const MetalFiniteGratingRgbResult> prior=escaped;
    if(!storage(false)||produceMetalFiniteGratingRgbForFixture(frame,carrier,result,error,2,&escaped)||result||!escaped
        ||!prior.expired()||!storage(true)||after[4]!=before[4]||after[2]!=before[2]||after[8]!=before[8]+1
        ||inspectMetalFiniteGratingRgb(frame,escaped,error))return fail("Finite RGB populated escape replacement did not retire outside mutex exactly once");
    result=escaped;if(!alias())return fail("Finite RGB populated alias mutated owner or journals");
    const auto incumbent=result;
    if(!storage(false)||run(3)||result!=incumbent||!storage(true)||after[4]!=before[4]||after[2]!=before[2]||after[8]!=before[8])
        return fail("Finite RGB first fallible copy changed incumbent or refunded live parent");
    if(!storage(false)||run(-1,surfacesource::add(before[3],before[6])-1)||result!=incumbent||!storage(true)
        ||after[1]!=before[1]||after[2]!=before[2]||after[8]!=before[8])return fail("Finite RGB escaped retry credited the live incumbent");
    if(!produceMetalFiniteGratingRgb(frame,carrier,result,error)||!inspectMetalFiniteGratingRgb(frame,result,error)
        ||!readMetalFiniteGratingRgbForFixture(frame,result,rgb,error))return false;
    // Reconstruct M from the original native receiver dyadic, independently
    // of the RGB readback. This host approximation is a fixture oracle only.
    const auto requestWords=metalcaustic::finiteReceiverRequestWords(receiverWords[11]);
    double xyz[3]{};const auto weights=glasscaustics::resolveWeights();float p32=0;
    const auto p32Bits=lambertian::kP32Bits;std::memcpy(&p32,&p32Bits,4);
    for(unsigned band=0;band<8;++band){const auto at=8+requestWords+100+70*band;
        long double m=0;for(int limb=3;limb>=0;--limb)m=std::ldexp(m,32)+receiverWords[at+2+unsigned(limb)];
        m=std::ldexp(m,std::int32_t(receiverWords[at]));
        const auto radiance=float(m/p32);
        for(unsigned axis=0;axis<3;++axis)xyz[axis]+=double(radiance)*weights[band][3]*weights[band][axis];}
    const std::array<double,3> expected{3.2406*xyz[0]-1.5372*xyz[1]-0.4986*xyz[2],
        -.9689*xyz[0]+1.8758*xyz[1]+.0415*xyz[2],.0557*xyz[0]-.2040*xyz[1]+1.0570*xyz[2]};
    for(unsigned channel=0;channel<3;++channel){if(!std::isfinite(rgb[channel])||std::abs(rgb[channel])>65504
            ||std::abs(double(rgb[channel])-expected[channel])>2e-5*std::max(1.0,std::abs(expected[channel])))
            return fail("Finite RGB differs from receiver M, production weights or matrix convention");
        std::uint32_t bits=0;std::memcpy(&bits,&rgb[channel],4);
        if(zero&&bits!=0u)return fail("Finite RGB genuine zero source did not resolve literal zero");}
    if(rgb[3]!=1.0f)return fail("Finite RGB native alpha is not one");
    // Accepted one-source zero-order domain has D8<=16, E=D8 and 0<rho<=1,
    // hence L<=16/P32. An absolute-coefficient matrix bound below64 includes
    // ample binary32 rounding slack. No genuine RGB overflow carrier exists
    // in this domain. Native refusal probes above mutate submitted bytes and
    // remain explicitly ineligible as optical RGB authority.
    const double matrix[3][3]={{3.2406,-1.5372,-.4986},{-.9689,1.8758,.0415},{.0557,-.2040,1.0570}};
    for(unsigned channel=0;channel<3;++channel){double bound=0;
        for(unsigned band=0;band<8;++band)for(unsigned axis=0;axis<3;++axis)
            bound+=16.0/p32*weights[band][3]*weights[band][axis]*std::abs(matrix[channel][axis]);
        if(!(bound<64))return fail("Finite RGB source-derived accepted-domain bound changed");}
    if(extensive&&!inspectMetalFiniteGratingRgbRefusalsForFixture(frame,result,error))return false;
    if(!storage(false))return false;const auto reserved=before[6],nativeBefore=before[5];
    escaped.reset();
    // incumbent still aliases the escape above, so its residency must remain.
    if(!storage(true)||after[2]!=before[2]||after[5]!=nativeBefore)return fail("Finite RGB alias release refunded a live payload");
    // A separate escaped attempt has exactly one owner and is retired once.
    std::shared_ptr<const MetalFiniteGratingRgbResult> disposable;
    if(!storage(false)||produceMetalFiniteGratingRgbForFixture(frame,carrier,result,error,2,&disposable)||!disposable
        ||!storage(true)||after[2]-before[2]!=reserved||after[4]!=before[4]+1)return false;
    if(!storage(false))return false;const auto retiringNative=before[5];disposable.reset();
    if(!storage(true)||before[2]-after[2]!=reserved||after[4]+1!=before[4]||after[5]>=retiringNative||after[8]!=before[8])
        return fail("Finite RGB native disposal did not refund exactly once or lost permanent work");
    if(frame->colorImageHandle()||frame->colorTextureViewHandle()||frame->depthImageHandle()||frame->depthTextureViewHandle())
        return fail("Finite RGB private result exposed a public attachment");
    if(!foreignFrame)foreignRgb=result;error.clear();return true;
}

inline bool nativeFiniteRadianceCases(const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& frame,
    const std::shared_ptr<const arbitgpu::MetalFiniteGratingReceiverResult>& receiver,bool zero,bool extensive,
    const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& foreignFrame,
    const std::shared_ptr<const arbitgpu::MetalFiniteGratingReceiverResult>& foreignReceiver,
    std::shared_ptr<const arbitgpu::MetalRestrictedCompleteBaseResult>& foreignBase,
    std::shared_ptr<const arbitgpu::MetalFiniteGratingRadianceResult>& foreignRadiance,
    std::array<std::uint32_t,8>& encodings,std::shared_ptr<const arbitgpu::MetalFiniteGratingRgbResult>& foreignRgb,
    std::array<float,4>& rgb,std::string& error)
{
    using namespace arbitgpu;
    const auto fail=[&](const char* message){error=message;return false;};
    std::shared_ptr<const MetalRestrictedCompleteBaseResult> base;
    if(!produceMetalRestrictedCompleteBaseForFixture(frame,receiver,base,error)||!inspectMetalRestrictedCompleteBaseForFixture(frame,base,error))return false;
    std::shared_ptr<const MetalFiniteGratingRadianceResult> result,escaped;
    std::array<std::uint64_t,8> before{},after{};finiteradiance::Costs work{},next{};
    const auto storage=[&](bool newer){return readMetalFiniteGratingRadianceStorageForFixture(frame,newer?after:before,newer?next:work,error);};
    const auto run=[&](int stop=-1,std::uint64_t limit=renderpassoutput::kMaximumTotalBytes){
        return produceMetalFiniteGratingRadianceForFixture(frame,receiver,base,result,error,stop,nullptr,limit);};
    const auto aliasRefused=[&](){
        const auto incumbent=result;
        if(!storage(false)||!storage(true)||after!=before)return false;
        const auto readWork=next;
        for(unsigned k=0;k<work.size();++k)if(next[k]<work[k])return false;
        auto readDelta=next;for(unsigned k=0;k<work.size();++k)readDelta[k]-=work[k];
        if(produceMetalFiniteGratingRadianceForFixture(frame,receiver,base,result,error,2,&result)
            ||result!=incumbent||!storage(true)||after!=before)return false;
        for(unsigned k=0;k<work.size();++k)if(next[k]<readWork[k]||next[k]-readWork[k]!=readDelta[k])return false;
        return true;
    };
    const auto constructionRefused=[&](){
        const auto incumbent=result;
        const auto frameOwners=frame.use_count(),receiverOwners=receiver.use_count(),baseOwners=base.use_count();
        lambertian::native::Costs arithmeticBefore{},arithmeticAfter{};lambertian::native::Storage previous{},current{};
        if(!readMetalLambertianStorageForFixture(frame,arithmeticBefore,previous,error)||!storage(false)
            ||run(3)||result!=incumbent||!storage(true)
            ||after[0]!=before[0]+1||after[1]!=before[1]+1||after[2]!=before[2]||after[4]!=before[4]
            ||after[5]!=before[5]||next==work
            ||frame.use_count()!=frameOwners||receiver.use_count()!=receiverOwners||base.use_count()!=baseOwners
            ||!readMetalLambertianStorageForFixture(frame,arithmeticAfter,current,error)
            ||current.attempts!=previous.attempts||current.attemptBytes!=previous.attemptBytes||current.nativeBytes!=previous.nativeBytes
            ||arithmeticAfter[lambertian::native::NativeVisits]!=arithmeticBefore[lambertian::native::NativeVisits])return false;
        return true;
    };
    if(!aliasRefused())return fail("Finite radiance aliased null output changed admission or output");
    if(!storage(false)||run(-1,0)||result||!storage(true)||after[0]!=before[0]+1||after[1]!=before[1]
        ||after[2]!=before[2]||after[5]!=before[5]||work==next)return fail("Finite radiance pre-heap shortage allocated or dispatched");
    for(int stop:{0,1})if(!storage(false)||run(stop)||result||!storage(true)||after[1]!=before[1]+1||after[4]||after[5]!=before[5]||work==next)
        return fail("Finite radiance early failure lost its wrapper journal or consumed arithmetic slots");
    if(!constructionRefused())return fail("Finite radiance construction unwind lost member ownership, retirement or permanent failed work");
    if(foreignFrame){
        if(!storage(false))return false;
        result=foreignRadiance;
        if(produceMetalFiniteGratingRadianceForFixture(frame,receiver,base,result,error)||result!=foreignRadiance
            ||!storage(true)||after[1]!=before[1]||after[2]!=before[2]||after[5]!=before[5])return fail("Finite radiance foreign incumbent received storage credit");
        result.reset();
        if(produceMetalFiniteGratingRadianceForFixture(frame,foreignReceiver,base,result,error)
            ||produceMetalFiniteGratingRadianceForFixture(frame,receiver,foreignBase,result,error)
            ||inspectMetalFiniteGratingRadianceForFixture(frame,foreignRadiance,error)
            ||inspectMetalFiniteGratingRadianceForFixture(foreignFrame,result,error)||result)return fail("Finite radiance accepted substituted optical parents or frame");
    }
    if(!storage(false))return false;
    const auto exact=surfacesource::add(before[3],surfacesource::add(before[6],surfacesource::multiply(8,before[7])));
    if(run(-1,exact-1)||result||!storage(true)||after[1]!=before[1]||after[2]!=before[2]||after[5]!=before[5])
        return fail("Finite radiance one-byte-short eight-child envelope dispatched");
    if(extensive)for(int stop:{10,13,17}){
        lambertian::native::Costs arithmeticBefore{},arithmeticAfter{};lambertian::native::Storage previous{},current{};
        if(!readMetalLambertianStorageForFixture(frame,arithmeticBefore,previous,error)||!storage(false)||run(stop)||result||!storage(true)
            ||after[1]!=before[1]+1||after[4]||after[5]!=before[5]+std::uint64_t(stop-9)
            ||!readMetalLambertianStorageForFixture(frame,arithmeticAfter,current,error)
            ||current.attemptBytes<=previous.attemptBytes||current.nativeBytes<=previous.nativeBytes
            ||arithmeticAfter[lambertian::native::NativeVisits]-arithmeticBefore[lambertian::native::NativeVisits]
                !=std::uint64_t(stop-9)*lambertian::kNativeCeilings.limbVisits)
            return fail("Finite radiance between-band failure refunded completed children or their attempted work");
    }
    if(extensive)for(unsigned phase:{0u,2u}){
        lambertian::native::Costs arithmeticBefore{},arithmeticAfter{};lambertian::native::Storage previous{},current{};
        if(!readMetalLambertianStorageForFixture(frame,arithmeticBefore,previous,error)||!storage(false)||run(int(100+2*16+phase))||result
            ||!storage(true)||after[5]!=before[5]+3||after[4]
            ||!readMetalLambertianStorageForFixture(frame,arithmeticAfter,current,error)
            ||current.attemptBytes-previous.attemptBytes!=std::uint64_t(phase==2?3:2)*before[7])
            return fail("Finite radiance child refusal lost sealed or committed ownership, or retained an unsealed allocation");
    }
    if(!storage(false)||produceMetalFiniteGratingRadianceForFixture(frame,receiver,base,result,error,2,&escaped)||result||!escaped
        ||!storage(true)||after[4]!=before[4]+1||after[5]!=before[5]+8
        ||inspectMetalFiniteGratingRadianceForFixture(frame,escaped,error))return fail("Finite radiance unpublished escape gained authority or lost storage");
    // Only the reused escape slot owns the first wrapper. Replacement must
    // retire it after unlock while retaining all frame-owned native children.
    const std::weak_ptr<const MetalFiniteGratingRadianceResult> firstEscape=escaped;
    lambertian::native::Costs reuseWork{},reusedWork{};lambertian::native::Storage reuseStorage{},reusedStorage{};
    if(!readMetalLambertianStorageForFixture(frame,reuseWork,reuseStorage,error)||!storage(false)
        ||produceMetalFiniteGratingRadianceForFixture(frame,receiver,base,result,error,2,&escaped)||result||!escaped
        ||!firstEscape.expired()||!storage(true)||after[0]!=before[0]+1||after[1]!=before[1]+1
        ||after[2]!=before[2]||after[4]!=before[4]||after[5]!=before[5]+8||next==work
        ||!readMetalLambertianStorageForFixture(frame,reusedWork,reusedStorage,error)
        ||reusedStorage.attempts!=reuseStorage.attempts+8
        ||reusedStorage.attemptBytes-reuseStorage.attemptBytes!=8*before[7]
        ||reusedStorage.nativeBytes<=reuseStorage.nativeBytes
        ||reusedWork[lambertian::native::NativeVisits]-reuseWork[lambertian::native::NativeVisits]!=8*lambertian::kNativeCeilings.limbVisits
        ||inspectMetalFiniteGratingRadianceForFixture(frame,escaped,error))
        return fail("Finite radiance reused escape did not retire outside guard or preserve native journals");
    std::array<std::uint32_t,8> untouched{};untouched.fill(0xabcdef01u);
    std::array<std::array<std::uint32_t,250>,8> untouchedCertificates{};for(auto& c:untouchedCertificates)c.fill(0x12345678u);
    if(readMetalFiniteGratingRadianceForFixture(frame,escaped,untouched,untouchedCertificates,error)
        ||!std::all_of(untouched.begin(),untouched.end(),[](auto value){return value==0xabcdef01u;}))return false;
    for(const auto& c:untouchedCertificates)if(!std::all_of(c.begin(),c.end(),[](auto value){return value==0x12345678u;}))return false;
    const auto reservation=before[6];
    result=escaped;
    if(!storage(false))return false;
    if(!aliasRefused())return fail("Finite radiance aliased incumbent changed admission or output");
    if(!constructionRefused())return fail("Finite radiance construction unwind changed its retained incumbent or native children");
    if(!storage(false))return false;
    const auto escapedShort=surfacesource::add(before[3],surfacesource::add(before[6],surfacesource::multiply(8,before[7])))-1;
    if(run(-1,escapedShort)||result!=escaped||!storage(true)||after[1]!=before[1]||after[5]!=before[5])
        return fail("Finite radiance refused retry changed its registered unpublished incumbent");

    lambertian::native::Costs arithmeticBefore{},arithmeticAfter{};lambertian::native::Storage previous{},current{};
    if(!readMetalLambertianStorageForFixture(frame,arithmeticBefore,previous,error)||!run()||!result||result==escaped
        ||!inspectMetalFiniteGratingRadianceForFixture(frame,result,error)
        ||!readMetalLambertianStorageForFixture(frame,arithmeticAfter,current,error)
        ||current.attempts!=previous.attempts+8
        ||arithmeticAfter[lambertian::native::NativeVisits]-arithmeticBefore[lambertian::native::NativeVisits]!=8*lambertian::kNativeCeilings.limbVisits
        ||current.arithmeticInspections<previous.arithmeticInspections+32)
        return fail("Finite radiance retry did not retain eight actual certified commands and charged host checks");
    std::array<std::array<std::uint32_t,250>,8> certificates{};
    if(!readMetalFiniteGratingRadianceForFixture(frame,result,encodings,certificates,error))return false;
    std::array<std::uint32_t,kMetalFiniteReceiverDiagnosticWords> receiverWords{};
    if(!readMetalFiniteGratingReceiverWordsForFixture(frame,receiver,receiverWords,error))return false;
    const auto q=metalcaustic::finiteReceiverRequestWords(receiverWords[8+3]);
    for(unsigned band=0;band<8;++band){const auto at=8+q+100+70*band;const auto& c=certificates[band];
        lambertian::Request expected{};expected[0]=1;expected[1]=lambertian::kP32Bits;expected[2]=128;expected[3]=640;
        expected[4]=receiverWords[at];for(unsigned k=0;k<4;++k)expected[5+k]=receiverWords[at+2+k];
        lambertian::putBudgets(expected,lambertian::kNativeCeilings);
        if(c[0]!=1||c[247]!=1||c[2]!=encodings[band]||!std::equal(expected.begin(),expected.end(),c.begin()+4)
            ||(zero&&(encodings[band]!=0||receiverWords[at]||receiverWords[at+1])))
            return fail("Finite radiance certificate differs from exact receiver exitance, request echo or literal zero");
        for(unsigned k=6;k<10;++k)if(receiverWords[at+k])return false;
    }
    if(extensive&&!inspectMetalFiniteGratingRadianceRefusalsForFixture(frame,result,error))return false;
    if(!nativeFiniteRgbCases(frame,result,receiverWords,zero,extensive,foreignFrame,foreignRadiance,foreignRgb,rgb,error))return false;
    {const auto incumbent=result;
    if(!storage(false)||run()||result!=incumbent||!storage(true)||after[1]!=before[1]||after[5]!=before[5]
        ||!inspectMetalFiniteGratingRadianceForFixture(frame,result,error))return fail("Finite radiance duplicate success dispatched or changed incumbent");}
    if(!storage(false)||before[4]!=2)return false;
    escaped.reset();
    if(!storage(true)||after[4]!=1||before[2]-after[2]!=reservation
        ||!readMetalLambertianStorageForFixture(frame,arithmeticAfter,current,error)||current.attemptBytes<previous.attemptBytes)
        return fail("Finite radiance escaped wrapper disposal refunded frame-owned arithmetic children");
    if(frame->colorImageHandle()||frame->colorTextureViewHandle()||frame->depthImageHandle()||frame->depthTextureViewHandle())return false;
    if(!foreignFrame){foreignBase=base;foreignRadiance=result;}
    else {
        result.reset();
        if(!storage(false)||before[4]||run()||result||!storage(true)||after[1]!=before[1]||after[5]!=before[5])
            return fail("Finite radiance released successful carrier lost its permanent support-cell claim");
    }
    error.clear();return true;
}

inline bool run(arbitgpu::NativeFixtureSceneBackend& backend,std::string& error)
{
    if(!legacyRecipeCases(error)||!constructionCases(error)||!censusCases(error)||!frameCases(error)||!completeBaseAdmissionCases(error)) return false;
    const auto scene=std::make_shared<const HarmonicMIDI::grid::Visual3DScene>(fixture3d::makeScene());
    const auto preparation=backend.prepare(scene);
    if(!preparation.prepared||!preparation.resources) { error=preparation.error; return false; }
    if(!arbitgpu::inspectMetalSurfaceSourceAdmissionForFixture(preparation.resources,error)) return false;
    // An unrelated resource cannot borrow the archive or its shader association.
    if(arbitgpu::inspectMetalSurfaceSourceAdmissionForFixture({},error)) {
        error="Surface source fixture accepted an unowned archive";return false;
    }
    error.clear();return true;
}
} // namespace videohelper::tests::restrictedsurfacesource
