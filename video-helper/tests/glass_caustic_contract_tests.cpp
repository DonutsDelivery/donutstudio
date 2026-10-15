#include "support/camera_wide_line_predicate_cases.h"
#include "support/camera_line_predicate_cases.h"
#include "support/camera_ray_coefficient_cases.h"
#include "support/exact_line_predicate_cases.h"
#include "support/glass_caustic_oracle.h"
#include "../src/gpu_backend/fixture_glass_grating_incident.h"
#include "../src/gpu_backend/fixture_glass_grating_entry_cells.h"
#include "../src/gpu_backend/fixture_glass_grating_cell_transport.h"
#include "../src/gpu_backend/fixture_glass_grating_cell_coverage.h"
#include "../src/gpu_backend/fixture_glass_camera_cell_coverage.h"
#include "../src/gpu_backend/fixture_glass_caustics_opengl.h"
#include "../src/gpu_backend/metal_glass_caustic_consumer.h"
#include <cstdio>
#include <limits>
#include <tuple>
#include <map>

namespace
{
namespace gt=glasscaustictests;
namespace gc=arbitgpu::glasscaustics;
int failures=0,checks=0;
void check(bool condition,const std::string& text)
{++checks;if(!condition){++failures;std::fprintf(stderr,"FAIL: %s\n",text.c_str());}}
bool close(double a,double b,double tolerance=3e-5)
{return std::abs(a-b)<=tolerance*std::max(1.0,std::max(std::abs(a),std::abs(b)));}
// Pure independent mathematical preparation, not emitted-parent acceptance.
// Native GL4.3/Metal constructor/predicate probes are prepared and ALL UNRUN.
void cameraWideLinePredicatePreparation()
{
    namespace ec=videohelper::tests::camerawidelinecases;namespace cw=arbitgpu::camerawideline;
    const auto tests=ec::cases();check(tests.size()<1024,"wide generated probes have bounded case count");
    const auto source=ec::sourceCamera();const auto record=ec::modelRecord(source);
    const auto square=ec::cr::multiply(ec::cr::decode(source[14]),ec::cr::decode(source[14]));
    check(square.numerator.bits.size()==47,"source rounded sqrt-half quaternion square needs47 significand bits");
    check(record[0]==1&&ec::oracleRecordAccepted(record,source),"source-grounded wide camera model roundtrips canonically");
    bool wide=false;
    for(unsigned k=0;k<17;k++)for(unsigned j=0;j<3;j++)for(unsigned limb=1;limb<8;limb++)wide|=record[63+28*k+9*j+limb]!=0;
    check(wide,"rounded sqrt-half camera generates coefficients exceeding the precision30 domain");
    check(ec::oracle(tests[0])[0]==0&&ec::oracle(tests[1])[0]==1&&ec::oracle(tests[1])[2]==3,
        "wide source model preserves opposed seam coincidence; native precision30 refusal has a separate diagnostic probe");
    check(ec::oracle(tests[2])[3]==0&&ec::oracle(tests[3])[2]==4&&ec::oracle(tests[6])[3]==0,
        "closed endpoint, zero depth form and equal event remain distinct arithmetic results");
    check(ec::oracle(tests[10])[3]==2&&ec::oracle(tests[11])[3]==2&&ec::oracle(tests[17])[3]==2,
        "independent rational ordering retains denominator normalization");
    check(ec::oracle(tests[19])[0]==1&&ec::oracle(tests[21])[0]==1&&ec::oracle(tests[25])[0]==1,
        "full256-bit products and actual high limbs remain supported mathematical inputs");
    for(unsigned k=26;k<tests.size();k++)check(ec::oracle(tests[k])[0]==0,
        std::string("prepared wide domain/work refusal: ")+tests[k].label);
    for(const auto& test:tests){const auto result=ec::oracle(test);
        check(result[1]==cw::maximumScalarOperations(static_cast<cw::Operation>(test.input[58])),
            "wide attempted failure retains full required charge");
        check(std::equal(test.input.begin(),test.input.begin()+65,result.begin()+112),"wide result retains exact tags and source-line request");}
    const auto prepared=cw::PreparedRecord::prepare(record,source);
    check(prepared&&prepared->record()==record&&prepared->request()==source,"immutable wide preparation retains exact mathematical lineage");
    auto tampered=record;tampered[8]^=1;
    check(!cw::PreparedRecord::prepare(tampered,source),"wide stale source echo refuses preparation");
    check(cw::maximumVisits(cw::Operation::CompareX)==116175&&cw::maximumWordMoves(cw::Operation::CompareX)==1874860,
        "wide accounting includes fixed35 scans and nested argument copies");
    check(cw::glslProbeSource().find(cw::arithmeticSource())!=std::string::npos
        &&cw::metalProbeSource().find(cw::arithmeticSource())!=std::string::npos,"wide native probes share identical arithmetic body");
}
// Preparation/oracle assertions only. Actual native call sites are prepared in
// the GL4.3/Metal targets; ALL rows remain UNRUN under the execution hold.
void cameraLinePredicatePreparation()
{
    namespace ec=videohelper::tests::cameralinecases;namespace cl=arbitgpu::cameraline;
    const auto tests=ec::cases();check(tests.size()<1024,"generated-line probes have bounded input count");
    const auto endpoint=ec::oracle(tests[0]),coincident=ec::oracle(tests[1]),zero=ec::oracle(tests[2]);
    check(endpoint[0]==1&&endpoint[2]==1&&endpoint[3]==0,"independent closed endpoint incidence is exact zero");
    check(coincident[0]==1&&coincident[2]==3&&zero[0]==1&&zero[2]==4,
        "nonzero seam coincidence differs from equal-depth zero form");
    check(ec::oracle(tests[6])[3]==0&&ec::oracle(tests[17])[3]==0,"equal rational events acquire no tie order");
    check(ec::oracle(tests[8])[3]==2&&ec::oracle(tests[9])[3]==2&&ec::oracle(tests[16])[3]==2,
        "independent rational order survives denominator normalization");
    for(unsigned k=24;k<tests.size();k++)check(ec::oracle(tests[k])[0]==0,
        std::string("prepared domain/work refusal: ")+tests[k].label);
    for(const auto& test:tests){const auto result=ec::oracle(test);
        check(result[1]==cl::maximumScalarOperations(static_cast<cl::Operation>(test.input[58])),
            std::string("failure retains full admission charge: ")+test.label);
        check(std::equal(test.input.begin(),test.input.begin()+65,result.begin()+34),"result preserves exact lineage and operation request");}
    const auto parent=ec::cr::base();const auto record=ec::modelRecord(parent);
    const auto prepared=cl::PreparedRecord::prepare(record,parent);
    check(prepared&&prepared->record()==record&&prepared->request()==parent,"immutable preparation retains exact mathematical record");
    auto tampered=record;tampered[8]^=1;
    check(!cl::PreparedRecord::prepare(tampered,parent),"stale opaque frame echo refuses preparation");
    check(cl::kCrossVisits==177&&cl::kIncidenceVisits==124&&cl::kCompareVisits==113,
        "four-limb arithmetic envelopes include width and canonical-zero scans");
    check(cl::glslProbeSource().find(cl::arithmeticSource())!=std::string::npos
        &&cl::metalProbeSource().find(cl::arithmeticSource())!=std::string::npos,
        "generated-line probes share identical native arithmetic body");
}
// Oracle/source checks only; native arithmetic evidence is in the GL/Metal
// fixture targets. All prepared fixtures remain UNRUN under the execution hold.
void exactLinePredicatePreparation()
{
    namespace ec=videohelper::tests::exactlinecases;
    namespace el=arbitgpu::exactline;
    const auto tests=ec::cases();
    check(tests.size()<1024,"exact arithmetic probe has a bounded input count");
    const unsigned signs[]{0,1,2,0,1,0};
    for(unsigned k=0;k<6;k++){
        const auto result=ec::oracle(tests[k].input);
        check(result[0]==1&&result[1]==signs[k],std::string("independent dyadic oracle: ")+tests[k].label);
    }
    for(unsigned k=10;k<13;k++)check(ec::oracle(tests[k].input)[2]==k-9,"independent finite/parallel/coincident classification");
    check(ec::oracle(tests[13].input)[0]==0&&ec::oracle(tests[16].input)[0]==0,
        "zero line and degenerate incidence refuse");
    check(ec::oracle(tests[14].input)[1]==0&&ec::oracle(tests[15].input)[1]==1,
        "exact seam incidence distinguishes the smallest subnormal displacement");
    check(ec::oracle(tests[17].input)[1]==2&&ec::oracle(tests[18].input)[1]==0
        &&ec::oracle(tests[19].input)[1]==2&&ec::oracle(tests[20].input)[1]==2,
        "rational oracle retains order under denominator reversal and exact equality");
    for(const auto& test:tests){const auto result=ec::oracle(test.input);
        check(result[3]<=el::maximumVisits(static_cast<el::Operation>(test.input[0])),
            std::string("prepared admission bound: ")+test.label);}
    check(el::kCoefficientLimbs*32>=277&&el::kDet2Limbs*32>=555
        &&el::kDet3Limbs*32>=834&&el::kComparisonLimbs*32>=1111,
        "fixed limb capacities cover finite coefficient and intermediate arithmetic bounds");
    // Prepared source-accounting checks only; no shader/native execution claim.
    el::Request invalid{};invalid[0]=7;
    check(el::maximumVisits(static_cast<el::Operation>(7))==0
        &&el::maximumScalarOperations(static_cast<el::Operation>(7))==67072
        &&ec::oracle(invalid)==el::Result{},
        "invalid raw dispatch charges positive clears/setup while mathematical refusal remains zero");
    // Det2 independently expands four decodes, two9x9 products and one sum.
    constexpr std::uint64_t visits=4*37+2*(175+71*9*9)+175+35;
    constexpr std::uint64_t moves=4096+4*81+224+2*155+152+7*80+8*45+3*80+81
        +2*162*82+8*visits+162*8+6*(70*162+70);
    check(el::maximumWordMoves(el::Operation::Det2)==moves
        &&el::maximumScalarOperations(el::Operation::Det2)==34304+64*visits+8*moves,
        "raw det2 schedule includes complete nested copies and attempted carry walks");
    check(el::maximumWordMoves(el::Operation::CompareX)==1934660
        &&el::maximumScalarOperations(el::Operation::CompareY)==23149920,
        "raw event comparison reserves both crosses, products and full work copies");
    check(el::glslProbeSource().find(el::arithmeticSource())!=std::string::npos
        &&el::metalProbeSource().find(el::arithmeticSource())!=std::string::npos,
        "GLSL and Metal probes use the identical integer arithmetic body");
}

// Mathematical/oracle preparation only; these assertions establish no visibility
// or seam-union proof. Native probes live in the GL/Metal fixture targets, UNRUN.
void cameraRayCoefficientPreparation()
{
    namespace cr=videohelper::tests::cameraraycases;
    namespace cc=arbitgpu::cameraraycoefficients;
    const auto tests=cr::cases();check(tests.size()<1024,"coefficient probes have bounded case count");
    const auto first=cr::oracle(tests[0].input);
    check(first.valid,"independent rational same-owner seam model prepares");
    if(first.valid){
        for(unsigned k=0;k<3;k++)check(cr::equal(first.forms[4][k],cr::negative(first.forms[12][k])),
            "opposite ordered shared edge coefficients agree exactly");
        const auto& input=tests[0].input;
        check(input[28]==input[45]&&input[29]==input[47],"shared edge preserves exact endpoint identity tags");
        for(unsigned k=0;k<3;k++)check(input[31+k]==input[48+k]&&input[34+k]==input[54+k],
            "shared edge preserves endpoint position bitwords");
        auto quarter=cr::ratio(cr::number(1),cr::number(4)),threeQuarters=cr::ratio(cr::number(3),cr::number(4));
        auto atPoint=cr::add(cr::add(cr::multiply(first.forms[4][0],quarter),
            cr::multiply(first.forms[4][1],threeQuarters)),first.forms[4][2]);
        check(atPoint.numerator.sign==0,"shared source segment line crosses nominated pixel interior exactly");
        // This exact point is on the retained finite source segment, not merely
        // its infinite line. It is arithmetic incidence, never a cell-union proof.
        auto n=cr::add(cr::add(cr::multiply(first.forms[7][0],quarter),
            cr::multiply(first.forms[7][1],threeQuarters)),first.forms[7][2]);
        auto parameter=cr::ratio(first.heights[0],n);
        for(unsigned axis=0;axis<3;axis++){
            auto ray=cr::add(cr::add(cr::multiply(first.forms[axis][0],quarter),
                cr::multiply(first.forms[axis][1],threeQuarters)),first.forms[axis][2]);
            auto point=cr::multiply(ray,parameter);
            auto segment=cr::add(cr::multiply(cr::decode(input[31+axis]),quarter),
                cr::multiply(cr::decode(input[34+axis]),threeQuarters));
            check(cr::equal(point,segment),"retained endpoint segment contains the exact nominated-cell ray hit");}

        for(const auto& coefficient:first.forms[16])check(coefficient.numerator.sign==0,
            "coplanar pair is equal depth without invented tie order");
    }
    const auto exchange=cr::oracle(tests[2].input),reversed=cr::oracle(tests[3].input);
    check(exchange.valid&&reversed.valid,"independent depth-exchange models prepare");
    if(exchange.valid&&reversed.valid){bool nonzero=false;
        for(unsigned k=0;k<3;k++){nonzero|=exchange.forms[16][k].numerator.sign!=0;
            check(cr::equal(exchange.forms[16][k],cr::negative(reversed.forms[16][k])),
                "winding reverses exchange numerator together with its denominator");}
        check(nonzero,"different planes produce a genuine depth-exchange form");}
    const auto nonUnit=cr::oracle(tests[4].input);
    check(nonUnit.valid,"non-unit captured quaternion remains a valid authored camera model");
    if(nonUnit.valid)check(cr::equal(nonUnit.forms[3][2],cr::number(5)),
        "authored conjugate projector produces depth factor five rather than an assumed inverse factor four");
    const auto generated=cr::oracle(tests[5].input);
    check(generated.valid&&generated.forms[0][0].numerator.bits.size()>24,
        "exact generated ray coefficient contains more than a binary32 significand");
    const auto nearEqual=cr::oracle(tests[7].input),farEqual=cr::oracle(tests[8].input),horizon=cr::oracle(tests[9].input);
    if(nearEqual.valid&&farEqual.valid&&horizon.valid){
        for(unsigned k=0;k<3;k++)check(nearEqual.forms[8][k].numerator.sign==0&&farEqual.forms[9][k].numerator.sign==0,
            "projector clip equality is exact and canonical");
        auto middle=cr::ratio(cr::number(1),cr::number(2));
        check(cr::add(cr::multiply(horizon.forms[7][0],middle),horizon.forms[7][2]).numerator.sign==0,
            "horizon line crosses the nominated pixel at exact one half");}
    else check(false,"independent projector/horizon models prepare");
    check(cc::maximumVisits(30)==210740&&cc::maximumVisits(256)==1208219,
        "constructor admission includes fixed35 scans, alignment, normalization and all film clears");
    // Independent source census: prevalidation, film, rotation, direction,
    // projector, ray-map determinant, two triangles and exchange, respectively.
    constexpr std::uint64_t sums=3+2+12+18+6+5+2*59+3;
    constexpr std::uint64_t products=10+18+27+9+9+2*72+6;
    constexpr std::uint64_t shifts=2*sums+3*17,normalizations=35+products+sums;
    constexpr std::uint64_t failureZeros=35+9+products+sums;
    constexpr std::uint64_t calls=failureZeros+2*shifts+normalizations+35+sums+83+products+5+10+39+9+2+17+1;
    for(unsigned precision:{30u,256u}){
        const std::uint64_t limbProducts=products*(precision==30?1:64);
        const auto visits=cc::maximumVisits(precision);
        const std::uint64_t copies=8192+failureZeros*82+shifts*(45+118)+normalizations*(83+36)
            +35*85+(products+sums+83)*157+5*74+10*453+39*305+9*453+2*1160+17*1235+sums*37
            +products*155+sums*152+(35+9+normalizations+shifts+products+sums)*80
            +(3*products+sums)*45+(products+sums)*80+sums*81+2*limbProducts*82
            +8*visits+limbProducts*8+6*(70*limbProducts+70*sums);
        check(calls==2220&&cc::maximumWordMoves(precision)==copies
            &&cc::maximumScalarOperations(precision)==32768+128*calls+64*visits+8*copies,
            "constructor source census accounts full success and mutually exclusive failure paths");
    }
    check(cc::maximumScalarOperations(0)==98304&&cc::maximumWordMoves(0)==8192,
        "invalid constructor precision retains fixed setup/output charge");
    check(cc::kLogicalScratchBytes==36580,"logical constructor scratch is explicit, excluding actual compiler/native owners");
    check(cc::glslProbeSource().find(cc::constructionSource())!=std::string::npos
        &&cc::metalProbeSource().find(cc::constructionSource())!=std::string::npos,
        "both native probes share exactly one dyadic constructor body");
}

void directionalIrradianceSources()
{
    using namespace HarmonicMIDI::grid;
    using namespace diffractionmaterial;
    auto legacy=gt::ordinaryFixture();std::string error;
    check(legacy.prepare(error),"legacy source fixture prepares: "+error);
    const auto legacyBytes=sceneaov::detail::encodeScene(*legacy.scene);
    const auto legacyIdentity=legacy.frame.identity;const auto legacyWork=legacy.frame.workTexels;
    auto physical=gt::ordinaryFixture();
    auto& source=physical.scene->lights[0].spectralSource;
    source.directionalIrradiance.version=1;
    source.directionalIrradiance.measure=DirectionalSpectralMeasure::IrradianceCoefficient;
    source.directionalIrradiance.coefficients={2,3,4,5,6,7,8,9};
    // All legacy radiance channels can be zero. Only explicit D launches flux.
    source.radiance.fill(0);
    check(physical.prepare(error),"authored directional measure reaches prepared entry schedule: "+error);
    if(physical.frame.work.empty())return;
    gc::SourceInputs inputs;
    check(gc::sourceInputs(physical.prepared,physical.frame.activeSource,inputs,error)
        &&inputs.radiance==source.radiance &&sameDirectionalIrradiance(inputs.directionalIrradiance,source.directionalIrradiance)
        &&inputs.light==physical.ordinaryLight,"source input owns exact D and raw Li separately");
    visualimportedscenerender::GlassCausticSelection selected;
    selected.version=1;selected.sourceLight=physical.ordinaryLight;
    selected.glassObject=physical.prepared.glass.object;
    selected.glassMaterial=physical.prepared.glass.sourceScene->objects[0].material;
    selected.policy={1,1e-4f,1e-5f,32,64,16};
    check(visualimportedscenerender::validGlassCausticScene(selected,*physical.scene),
        "ordinary selected caustic source admits explicit positive D without invented raw radiance");
    auto zeroScene=std::make_shared<Visual3DScene>(*physical.scene);
    zeroScene->lights[0].spectralSource.directionalIrradiance.coefficients.fill(0);
    zeroScene->lights[0].spectralSource.radiance.fill(1);
    check(!visualimportedscenerender::validGlassCausticScene(selected,*zeroScene),
        "legacy Li cannot establish a positive source when explicitly selected D is zero");
    auto foreign=selected;foreign.sourceLight={999};
    check(!visualimportedscenerender::validGlassCausticScene(foreign,*physical.scene),"foreign selected source owner refuses");
    bool positive=false;
    for(const auto& work:physical.frame.work) {
        const auto& entry=physical.frame.entries[work.entry];
        const float expected=float(double(source.directionalIrradiance.coefficients[work.band])
            *std::max(0.0f,gc::dot(entry.normal,inputs.direction))*entry.area/physical.frame.interfaceSamples);
        check(work.incidentPower==expected &&work.sourceMeasureVersion==1
            &&work.sourceMeasure==DirectionalSpectralMeasure::IrradianceCoefficient
            &&work.sourceLight==physical.ordinaryLight
            &&work.directionalCoefficient==source.directionalIrradiance.coefficients[work.band],
            "each band computes W/nm from D times projected metric entry area exactly once");
        positive|=work.incidentPower>0;
    }
    check(positive,"zero raw radiance still has genuinely positive explicitly authored flux");
    const auto preparation=gc::SourceMeasureUse::DirectionalIrradiancePreparation;
    check(gc::validateTables(physical.prepared,physical.frame,error,false,preparation),"typed preparation validates exact source owner");
    check(!gc::validateTables(physical.prepared,physical.frame,error)
        &&!gc::validateQueries(physical.prepared,physical.frame,error),"unimplemented native measure consumer stays refused");
    auto forged=physical.frame;
    forged.work[0].sourceLight={999};forged.identity=gc::frameIdentity(physical.prepared,forged);
    check(!gc::validateTables(physical.prepared,forged,error,false,preparation),"foreign scheduled source owner refuses");
    forged=physical.frame;forged.work[0].directionalCoefficient+=1;forged.identity=gc::frameIdentity(physical.prepared,forged);
    check(!gc::validateTables(physical.prepared,forged,error,false,preparation),"forged per-band coefficient refuses");
    auto active=std::make_shared<SceneLightRecord>(*physical.frame.activeSource);
    active->spectralSource.directionalIrradiance.coefficients[0]+=1;
    gc::SourceInputs sentinel;sentinel.maximum=123;
    check(!gc::sourceInputs(physical.prepared,active,sentinel,error)&&sentinel.maximum==123,
        "runtime coefficient changes need new immutable authored ownership and preserve failed output");
    for(int mode=0;mode<7;++mode) {
        auto malformed=gt::ordinaryFixture();auto& d=malformed.scene->lights[0].spectralSource.directionalIrradiance;
        d=source.directionalIrradiance;
        if(mode==0)d.version=2;
        if(mode==1)d.measure=DirectionalSpectralMeasure::Unspecified;
        if(mode==2)d.coefficients[0]=-1;
        if(mode==3)d.coefficients[0]=std::numeric_limits<float>::infinity();
        if(mode==4)d.version=0;
        if(mode==5)d.coefficients[0]=17;
        if(mode==6)d.measure=static_cast<DirectionalSpectralMeasure>(2);
        check(!malformed.prepare(error),"malformed unit/version/dimension refuses before schedule publication");
    }
    auto rough=gt::roughFixture(.05f);
    rough.scene->lights[0].spectralSource.directionalIrradiance=source.directionalIrradiance;
    rough.scene->lights[0].spectralSource.radiance.fill(0);
    const arbitgpu::smoothglass::RoughNativeConsumer roughGeometry{arbitgpu::NativeFixtureMaterialBackend::OpenGl,true,true,true};
    check(rough.prepare(error,roughGeometry),"physical source prepares rough quadrature without native D activation: "+error);
    double scheduled=0,expected=0;
    for(const auto& entry:rough.frame.entries)
        expected+=double(source.directionalIrradiance.coefficients[0])
            *std::max(0.0f,gc::dot(entry.normal,gc::Point{rough.frame.sourceDirectionTrace[0],rough.frame.sourceDirectionTrace[1],rough.frame.sourceDirectionTrace[2]}))*entry.area;
    for(const auto& work:rough.frame.work)if(work.band==0)scheduled+=work.incidentPower;
    check(rough.frame.interfaceSamples==16 &&scheduled>0 &&close(scheduled,expected),
        "sixteen interface samples divide flux once and conserve the entry quadrature total");
    check(legacy.prepare(error)&&sceneaov::detail::encodeScene(*legacy.scene)==legacyBytes
        &&legacy.frame.identity==legacyIdentity&&legacy.frame.workTexels==legacyWork,
        "absent measure preserves released source/GPU schedule bytes and stable accounted host identity");
}
// Prepared fake backend ownership only. These synthetic byte receipts exercise
// refusal/packing; no native upload, shader execution or pixels are claimed.
class PreparedSourceFrame : public arbitgpu::NativeFixtureSceneFrame,public gc::NativeDirectionalSourceBinding
{
public:
    gc::NativeDirectionalSourceUpload uploaded;
    std::uint32_t widthValue=0,heightValue=0;
    std::uintptr_t cache=1;
    const std::string& backend() const noexcept override {static const std::string value="opengl";return value;}
    std::uint32_t width() const noexcept override{return widthValue;}
    std::uint32_t height() const noexcept override{return heightValue;}
    std::uintptr_t colorImageHandle() const noexcept override{return 0;}
    std::uintptr_t colorTextureViewHandle() const noexcept override{return 0;}
    std::uintptr_t nativeResourceCacheIdentity() const noexcept override{return cache;}
    const arbitgpu::NativeFixtureSceneFrame& owner() const noexcept override{return *this;}
    const gc::NativeDirectionalSourceUpload& upload() const noexcept override{return uploaded;}
};
void directionalSourceNativeReceipts()
{
    auto fixture=gt::ordinaryFixture();auto& measure=fixture.scene->lights[0].spectralSource.directionalIrradiance;
    measure.version=1;measure.measure=diffractionmaterial::DirectionalSpectralMeasure::IrradianceCoefficient;
    measure.coefficients={2,1e-25f,0,5,6,7,8,9};std::string error;
    check(fixture.prepare(error),"native source receipt fixture prepares guarded authored D: "+error);
    if(fixture.frame.work.empty())return;
    PreparedSourceFrame owner;owner.widthValue=fixture.frame.width;owner.heightValue=fixture.frame.height;
    auto& upload=owner.uploaded;
    upload.backend=fixture.program->backend;upload.input=gc::directionalSourceInput(fixture.prepared,fixture.frame);
    upload.bindingGeneration=11;upload.complete=true;
    upload.submission={fixture.frame.identity,1,2,3,4,5,2};
    check(sizeof(upload.input)==48&&sizeof(gc::NativeDirectionalSourceEcho)==96
        &&upload.input.identifiers==std::array<std::uint32_t,4>{1,1,1,fixture.ordinaryLight.value},
        "native uint source records retain exact version/unit/metres/source ID and fixed offsets");
    check(gc::validateDirectionalSourceTables(fixture.prepared,fixture.frame,owner,error),
        "synthetic exact native-owner upload contract accepts at staging only");
    gc::NativeDirectionalSourceReceipt receipt;
    receipt.version=gc::kDirectionalSourceReceiptVersion;receipt.bindingGeneration=upload.bindingGeneration;
    receipt.submission=upload.submission;receipt.submission.submissionSerial=receipt.submission.completedFence=7;
    receipt.submission.scratchGeneration=8;receipt.complete=true;receipt.echo.input=upload.input;
    receipt.echo.directionTrace=fixture.frame.sourceDirectionTrace;
    for(unsigned b=0;b<8;++b)receipt.echo.rawRadiance[b]=fixture.frame.sourceRadiance[b/4][b%4];
    for(const auto& work:fixture.frame.work)receipt.potential.push_back(work.incidentPower);
    const auto validate=[&](const gc::NativeDirectionalSourceReceipt& value) {
        return gc::validateDirectionalSourceReadback(fixture.prepared,fixture.frame,owner,receipt.submission,value,error); };
    check(validate(receipt),"completed later transport retains exact input owner and independent rawLi/direction/potential proof");
    for(int mode=0;mode<10;++mode) {
        auto bad=receipt;
        if(mode==0)bad.echo.input.identifiers[0]=2;
        if(mode==1)bad.echo.input.identifiers[1]=0;
        if(mode==2)bad.echo.input.identifiers[2]=0;
        if(mode==3)bad.echo.input.identifiers[3]+=1;
        if(mode==4)bad.echo.input.coefficients[0]+=1;
        if(mode==5)bad.echo.rawRadiance[0]+=1;
        if(mode==6)bad.echo.directionTrace[0]+=1;
        if(mode==7)bad.bindingGeneration+=1;
        if(mode==8)bad.complete=false;
        if(mode==9)bad.overflow=true;
        check(!validate(bad),"forged unit/owner/native source bytes/generation/completion refuses");
    }
    auto bad=receipt;bad.potential[0]=-1;check(!validate(bad),"negative potential refuses");
    bad=receipt;bad.potential[0]=std::numeric_limits<float>::quiet_NaN();check(!validate(bad),"nonfinite potential refuses");
    bad=receipt;bad.potential.pop_back();check(!validate(bad),"missing per-work potential refuses");
    std::size_t positive=0;while(positive<receipt.potential.size()&&receipt.potential[positive]==0)++positive;
    check(positive<receipt.potential.size(),"source receipt fixture has real positive scheduled flux");
    if(positive<receipt.potential.size()) {bad=receipt;bad.potential[positive]=0;check(!validate(bad),
        "a source-blocked zero launch cannot replace the independent before-occlusion potential");}
    std::size_t tiny=receipt.potential.size(),zero=receipt.potential.size();
    for(std::size_t index=0;index<fixture.frame.work.size();++index) {
        if(fixture.frame.work[index].band==1 &&receipt.potential[index]>0)tiny=index;
        if(fixture.frame.work[index].band==2 &&receipt.potential[index]==0)zero=index;
    }
    check(tiny<receipt.potential.size() &&zero<receipt.potential.size(),
        "authored tiny-positive and zero D bands reach the existing source potential witness");
    if(tiny<receipt.potential.size() &&zero<receipt.potential.size()) {
        check(receipt.potential[tiny]<2e-24,"representable authored tiny potential is below the legacy absolute floor");
        bad=receipt;bad.potential[tiny]=0;check(!validate(bad),"missing tiny physical potential refuses");
        bad=receipt;bad.potential[tiny]*=1.01;check(!validate(bad),"tiny positive relative drift refuses");
        bad=receipt;bad.potential[tiny]*=1.0001;check(validate(bad),"tiny positive rounding within relative tolerance accepts");
        bad=receipt;bad.potential[zero]=receipt.potential[tiny];check(!validate(bad),"phantom positive potential on zero authored D refuses");
    }
    for(int mode=0;mode<5;++mode) {
        auto expected=receipt.submission;
        if(mode==0)expected.rendererGeneration+=1;
        if(mode==1)expected.queryGeneration+=1;
        if(mode==2)expected.rasterGeneration+=1;
        if(mode==3)expected.completedFence=0;
        if(mode==4)expected.submissionSerial=expected.completedFence=1;
        bad=receipt;bad.submission=expected;
        check(!gc::validateDirectionalSourceReadback(fixture.prepared,fixture.frame,owner,expected,bad,error),
            "changed renderer/query/raster, incomplete fence and older transport refuse retained upload proof");
    }
    const auto original=upload;
    upload.input.coefficients[0]+=1;check(!gc::validateDirectionalSourceTables(fixture.prepared,fixture.frame,owner,error),"native upload bytes must match exact authored coefficient");
    upload=original;upload.complete=false;check(!gc::validateDirectionalSourceTables(fixture.prepared,fixture.frame,owner,error),"caller feature selection cannot replace completed upload");
    upload=original;owner.cache=0;check(!gc::validateDirectionalSourceTables(fixture.prepared,fixture.frame,owner,error),"detached native frame owner refuses");
    owner.cache=1;
    check(!gc::validateTables(fixture.prepared,fixture.frame,error),"default/preparation marker still cannot advertise native D support");
    gc::NativeReadback incomplete;incomplete.directionalSource=std::make_shared<const gc::NativeDirectionalSourceReceipt>(receipt);
    gc::EnergyLedger sentinel;sentinel.launched.fill(17);
    check(!gc::validateNativeReadback(fixture.prepared,fixture.frame,incomplete,sentinel,error,&receipt.submission,false,&owner)
        &&sentinel.launched==std::array<double,8>{17,17,17,17,17,17,17,17},"source proof alone cannot publish missing path/deposition/Color receipts");
}
void physics()
{
    for(double from:{1.0,1.5,2.0})for(double to:{1.0,1.5,2.0})for(double angle:{0.0,.2,.6,1.0})
    {
        const gt::D3 incoming{std::sin(angle),0,-std::cos(angle)},normal{0,0,1};
        const auto expected=gt::snell(incoming,normal,from,to);gc::Interface actual;
        check(gc::interface({float(incoming[0]),0,float(incoming[2])},{0,0,1},float(from),float(to),actual),"finite Snell/Fresnel interface admitted");
        check(actual.tir==expected.tir && close(actual.reflection,expected.reflection),"float interface agrees with independent tangential/admittance oracle");
        if(!expected.tir)for(unsigned a=0;a<3;++a)check(close(actual.transmitted[a],expected.direction[a]),"each transmitted direction component matches binary64");
        check(close(actual.reflection+(1-actual.reflection),1,1e-12),"lossless interface conserves reflected/transmitted flux");
    }
    gc::Interface equal;check(gc::interface({0,0,-1},{0,0,1},1,1,equal)&&equal.transmitted==gc::Point{0,0,-1}
        &&equal.reflection==0,"equal-media control preserves exact incoming direction");
}
void gratingAreaDifferentials()
{
    const gc::Point e1{2,0,0},e2{0,3,0},direction{0,0,-1},normal{0,0,1};
    gc::Point du{},dv{};gc::GratingReceivingArea area;
    check(gc::smoothHitDifferential(e1,direction,normal,du)&&gc::smoothHitDifferential(e2,direction,normal,dv)
        &&du==e1&&dv==e2&&gc::gratingReceivingArea(e1,e2,.75f,du,dv,area)&&close(area.jacobian,1)&&close(area.area,.75),
        "parallel plane analytical limit has identity area for arbitrary source quadrature weight");
    const float angle=.4f;const gc::Point tilted{std::sin(angle),0,std::cos(angle)};
    check(gc::smoothHitDifferential(e1,direction,tilted,du)&&gc::smoothHitDifferential(e2,direction,tilted,dv)
        &&gc::gratingReceivingArea(e1,e2,.75f,du,dv,area)&&close(area.jacobian,1/std::cos(double(angle)))
        &&close(du[2],-2*std::tan(double(angle))),"tilted hit agrees with independent secant area magnification");
    gc::GratingReceivingArea half;
    check(gc::gratingReceivingArea(e1,e2,.375f,du,dv,half)&&close(2*half.area,area.area),
        "partitioned source quadrature weights add to the same receiving measure");
    float db=99;const gc::Point scaledHit{8,1,2},derivative{2,3,4};gc::Point reflected{},transmitted{};
    check(gc::smoothBiasDifferential(scaledHit,derivative,db)&&close(db,4e-6,1e-12)
        &&gc::smoothChildOriginDifferential(scaledHit,normal,true,derivative,reflected)
        &&gc::smoothChildOriginDifferential(scaledHit,normal,false,derivative,transmitted)
        &&close(reflected[2],4+double(db),2e-7)&&close(transmitted[2],4-double(db),2e-7),
        "signed reflection and transmission origin derivatives follow actual scaled bias");
    gc::Point initial=gc::advance(e1,{0,0,1},2*db);
    check(gc::smoothHitDifferential(initial,direction,normal,du)&&du==e1,
        "entry plane exactly cancels differentiated directional source offset");
    check(gc::smoothBiasDifferential({1,1,1},derivative,db)&&db==0,"strict floor bias has zero derivative despite coordinate ties");
    check(!gc::smoothBiasDifferential({8,8,0},derivative,db),"scaled bias coordinate tie refuses");
    check(!gc::smoothBiasDifferential({5,0,0},derivative,db),"floor-scale bias transition refuses");
    du={123,0,0};
    check(!gc::smoothHitDifferential(e1,{1,0,0},normal,du)&&du==gc::Point{123,0,0},"parallel singular plane refuses atomically");
    check(!gc::smoothHitDifferential(e1,{0,0,-1e-8f},normal,du),"numerically grazing differential refuses");
    area.area=123;
    check(!gc::gratingReceivingArea(e1,e2,.75f,e1,e1,area)&&area.area==123,"collapsed area refuses atomically");
    check(!gc::gratingReceivingArea(e1,e2,std::numeric_limits<float>::max(),{20,0,0},e2,area),"unrepresentable receiving area refuses");
    const auto legacyGl=gc::opengl::computeSource(),legacyMetal=arbitgpu::metalcaustic::shader(false);
    const auto gl=gc::opengl::gratingIncidentComputeSource("float spectralReflectance(float a,int b,float c,float d){return 1.0;}","float efficiency=0.0;vec2 centre=vec2(0.0);float z=1.0;");
    const auto glTerminal=gc::opengl::gratingTerminalComputeSource("float spectralReflectance(float a,int b,float c,float d){return 1.0;}","float efficiency=0.0;vec2 centre=vec2(0.0);float z=1.0;");
    const auto metal=arbitgpu::metalcaustic::gratingIncidentShader("/*fixture compiler-owned order bodies supplied by native owner*/");
    const auto metalTerminal=arbitgpu::metalcaustic::gratingTerminalShader("/*fixture compiler-owned order bodies supplied by native owner*/");
    check(!gl.empty()&&!glTerminal.empty()&&!metal.empty()&&!metalTerminal.empty(),"both additive incident and terminal generator substitutions remain complete");
    check(gl.find("hitDu.xy")!=std::string::npos&&metal.find("branch.du.xy")!=std::string::npos
        &&metal.find("sizeof(CBranch)==128")!=std::string::npos,"native generators emit exact derivative packing and account aligned pending ABI");
    check(legacyGl.find("gratingPlaneDerivative")==std::string::npos&&legacyMetal.find("gratingPlaneDerivative")==std::string::npos,
        "legacy smooth shader generators contain no differential branch extension");
    std::array<gc::Texel,gc::kGratingIncidentStride> packed{};packed[0]={0,0,2,0};packed[2]={65,65,0,0};
    packed[15][3]=1;packed[22]={float(gc::kGratingIncidentVersion),1,2,0};packed[23]={0,0,3,0};
    gc::NativeGratingIncident decodedGl,decodedMetal;std::array<gc::Texel,12> orders{};
    check(gc::opengl::decodeGratingIncident(packed.data(),decodedGl)
        &&arbitgpu::metalcaustic::decodeGratingIncident(packed.data(),orders.data(),0,decodedMetal)
        &&decodedGl.dHdu==e1&&decodedGl.dHdv==e2&&decodedMetal.dHdu==e1&&decodedMetal.dHdv==e2,
        "unchanged 24-texel GL and Metal receipt decoders agree on six derivative lanes");
    const auto physicalGl=gc::opengl::directionalIrradianceComputeSource(glTerminal);
    const auto physicalMetal=arbitgpu::metalcaustic::directionalIrradianceShader(metalTerminal);
    check(!physicalGl.empty()&&!physicalMetal.empty()&&physicalGl.find("directionalEchoRawRadiance")!=std::string::npos
        &&physicalMetal.find("sourcePotential[w]=computed")!=std::string::npos,
        "differential generators retain separate D/rawLi native echo and pre-occlusion potential seams");
    packed[23][3]=std::numeric_limits<float>::infinity();
    check(!gc::opengl::decodeGratingIncident(packed.data(),decodedGl)
        &&!arbitgpu::metalcaustic::decodeGratingIncident(packed.data(),orders.data(),0,decodedMetal),
        "nonfinite derivative receipt lanes refuse in both decoders");
    packed[23][3]=0;packed[22][0]=1;
    check(!gc::opengl::decodeGratingIncident(packed.data(),decodedGl)
        &&!arbitgpu::metalcaustic::decodeGratingIncident(packed.data(),orders.data(),0,decodedMetal),
        "old private zero-derivative incident version cannot establish area");
}
void gratingIncidentContract()
{
    // Synthetic typed fixtures exercise the prerequisite only. No native query,
    // grating order evaluation or production Color is asserted by these values.
    auto fixture=gt::ordinaryFixture();
    // Use a flat parallel slab for the analytical identity branch witness.
    // The general ordinary fixture has a sloped prism top and oblique light.
    for(std::size_t v=0;v<fixture.scene->vertexCount;++v)if(fixture.scene->vertices[v].position.z>0)
        fixture.scene->vertices[v].position.z-=.25f*fixture.scene->vertices[v].position.x;
    fixture.scene->lights[0].transform.rotation={0,0,0,1};
    const auto& target=fixture.scene->objects[1];
    for(std::size_t v=0;v<target.vertexCount;++v) {
        auto& vertex=fixture.scene->vertices[target.firstVertex+v];
        vertex.uv={vertex.position.x/12,vertex.position.y/12};
    }
    std::string error;check(fixture.prepare(error),"legacy staging scene retains smooth query authority: "+error);
    auto description=diffractionmaterial::makeAluminiumBinaryGratingPreset();
    description.spectrum.lastOrder=1;description.roughness.rmsSlope=0;description.microstructure.grooveDepthNanometres=.9f;
    const auto admitted=diffractionmaterial::admit(description,error);
    check(bool(admitted),"first bounded grating IR is structurally admitted");if(!admitted)return;
    const auto material=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*admitted);
    auto owner=std::make_shared<gt::Program>(*fixture.program->objectPrograms[0]);
    owner->kind=arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;owner->spectralReflectance.reset();
    owner->diffractionPathCount=1;owner->bindingDigest="typed-grating-binding";owner->programIdentity="typed-grating-program";
    owner->diffractionPaths[0].material=diffractionmaterial::physicalcheckpoint::makeGpuParameters(*material,{});
    std::shared_ptr<const gc::GratingIncidentBinding> binding;
    check(gc::prepareGratingIncidentBinding(fixture.prepared,fixture.frame,owner,material,.25,binding,error),
        "distinct exact static grating override freezes only the incident prerequisite: "+error);
    if(!binding)return;
    // D witness preparation happens before allocation. Only the later real
    // backend-owned upload/transport receipts can authorize native acceptance.
    auto directional=gt::ordinaryFixture();*directional.scene=*fixture.scene;
    auto& d=directional.scene->lights[0].spectralSource.directionalIrradiance;
    d.version=1;d.measure=diffractionmaterial::DirectionalSpectralMeasure::IrradianceCoefficient;
    d.coefficients={2,3,4,5,6,7,8,9};
    check(directional.prepare(error),"directional incident preparation retains exact authored scene: "+error);
    std::shared_ptr<const gc::GratingIncidentBinding> dBinding;
    check(!gc::prepareGratingIncidentBinding(directional.prepared,directional.frame,owner,material,.25,dBinding,error),
        "legacy incident binding factory cannot advertise D support");
    check(gc::prepareGratingIncidentBinding(directional.prepared,directional.frame,owner,material,.25,dBinding,error,
        gc::SourceMeasureUse::DirectionalIrradiancePreparation),"explicit D witness preparation precedes native allocation");
    if(dBinding) {
        auto bound=directional.frame;bound.gratingIncidentBinding=dBinding;bound.identity=dBinding->identity;
        PreparedSourceFrame native;native.widthValue=bound.width;native.heightValue=bound.height;
        native.uploaded={directional.program->backend,{dBinding->identity,1,2,3,4,5,2},11,
            gc::directionalSourceInput(directional.prepared,directional.frame),true};
        check(gc::validateDirectionalSourceTables(directional.prepared,bound,native,error,true),
            "synthetic upload pins actual bound stage identity rather than unbound schedule");
        auto sourceReceipt=std::make_shared<gc::NativeDirectionalSourceReceipt>();
        sourceReceipt->version=1;sourceReceipt->bindingGeneration=11;sourceReceipt->complete=true;
        sourceReceipt->submission={dBinding->identity,1,7,3,4,8,7};sourceReceipt->echo.input=native.uploaded.input;
        sourceReceipt->echo.directionTrace=directional.frame.sourceDirectionTrace;
        for(unsigned b=0;b<8;++b)sourceReceipt->echo.rawRadiance[b]=directional.frame.sourceRadiance[b/4][b%4];
        for(const auto& work:directional.frame.work)sourceReceipt->potential.push_back(work.incidentPower);
        gc::NativeGratingReadback subset;subset.submission=sourceReceipt->submission;subset.complete=true;
        subset.directionalSource=sourceReceipt;
        check(!gc::validateGratingIncidentReadback(*dBinding,subset,subset.submission,error),
            "even an empty D subset requires backend binding authority");
        check(gc::validateGratingIncidentReadback(*dBinding,subset,subset.submission,error,&native),
            "synthetic empty D subset accepts exact later source transport under retained stage owner");
        native.uploaded.submission.frameIdentity=directional.frame.identity;
        check(!gc::validateGratingIncidentReadback(*dBinding,subset,subset.submission,error,&native),
            "unbound upload identity cannot prove the distinct grating bound frame");
    }
    check(binding->witnessTexels.size()==binding->triangles.size()*gc::kGratingWitnessStride
        &&binding->bounds.combinedWork<=gc::kMaximumQueryWork,"frozen UV witnesses share original query and memory caps");
    const auto legacyIdentity=gc::frameIdentity(fixture.prepared,fixture.frame);
    const auto legacyQuery=fixture.frame.query.texels;
    gc::Frame marked=fixture.frame;marked.gratingIncidentBinding=binding;
    check(!gc::validateTables(fixture.prepared,marked,error)&&!error.empty(),"combined Color stays refused without exact native consumers");
    check(gc::frameIdentity(fixture.prepared,fixture.frame)==legacyIdentity&&fixture.frame.query.texels==legacyQuery
        &&!fixture.frame.gratingIncidentBinding,"default legacy identity and v3 query bytes stay unchanged");
    gc::NativeReadback guarded;guarded.gratingIncidentVersion=gc::kGratingIncidentVersion;
    gc::EnergyLedger ledger;ledger.launched.fill(29);
    check(!gc::validateNativeReadback(fixture.prepared,fixture.frame,guarded,ledger,error)
        &&ledger.launched==std::array<double,8>{29,29,29,29,29,29,29,29},"private grating readback cannot publish through legacy rho ledger");
    // The explicit mixed admission retains the exact query authority. Default
    // preparation still refuses the distinct non-Surface material.
    diffractionmaterialbinding::ImportedSceneDiffractionMaterialRequest request;
    request.version=diffractionmaterialbinding::kSceneReceiverWireVersion;request.scene=fixture.scene->id;request.object=owner->object;
    request.sceneRevision=request.structuralRevision=request.evaluationRevision=request.materialRevision=1;
    request.material=description;request.structuralDigest=material->structuralDigest();
    request.lighting.version=3;request.lighting.pathCount=1;request.lighting.traceMaximumDistance=20;
    request.lighting.sceneReceiverTransport=diffractionmaterial::SceneReceiverTransport::AuthoredSpectralLambertian;
    request.lighting.paths[0].kind=diffractionmaterial::LightingPathKind::Direct;
    for(unsigned axis=0;axis<3;++axis)request.lighting.paths[0].incident.direction[axis]=fixture.frame.sourceDirectionTrace[axis];
    for(unsigned band=0;band<8;++band)request.lighting.paths[0].incident.radiance[band]=fixture.frame.sourceRadiance[band/4][band%4];
    const auto imported=diffractionmaterialbinding::admit(request,error);
    check(bool(imported),"real imported compiler IR/lighting request admits zero-slope discrete material: "+error);if(!imported)return;
    const auto compiledGrating=gc::compileDiscreteGrating(*fixture.scene,request,fixture.program->backend,error);
    check(bool(compiledGrating),"production lowering emits exact zero-Surface-block grating owner: "+error);if(!compiledGrating)return;
    auto framed=request;framed.graphFrame=surfacematerialbinding::TextureSlotBinding::GraphFrameEndpoint{7,0};
    check(!gc::compileDiscreteGrating(*fixture.scene,framed,fixture.program->backend,error),"private compiler cannot discard an authored frame endpoint");
    const auto discrete=arbitgpu::NativeFixtureDiffractionConsumer::SmoothGlassDiscreteOrders;
    check(!arbitgpu::validNativeFixtureDiffractionProgram(*compiledGrating,compiledGrating->backend,compiledGrating->object,true)
        &&arbitgpu::validNativeFixtureDiffractionProgram(*compiledGrating,compiledGrating->backend,compiledGrating->object,true,discrete),
        "zero-slope compiler program is private discrete transport only, never a finite lobe");
    const arbitgpu::smoothglass::DistinctGratingAdmission selection{compiledGrating->object,material};
    auto wrongDigest=*compiledGrating;wrongDigest.bindingDigest="different-IR";
    wrongDigest.programIdentity=arbitgpu::nativeFixtureDiffractionProgramIdentity(wrongDigest.bindingDigest,
        *wrongDigest.diffractionLightingAdmission,0,1,{},false,&wrongDigest);
    check(!arbitgpu::smoothglass::validDistinctGrating(wrongDigest,selection),"a valid lighting receipt cannot replace the exact compiler material IR digest");
    auto surfaceBits=*compiledGrating;surfaceBits.parameters.transmissionIorClearcoat[1]=1.5f;
    check(!arbitgpu::smoothglass::validDistinctGrating(surfaceBits,selection),"selected grating cannot smuggle Surface parameter blocks");
    auto mixedRoot=std::make_shared<gt::Program>(*compiledGrating);
    auto glassChild=std::make_shared<gt::Program>(*fixture.program);glassChild->objectPrograms.clear();mixedRoot->objectPrograms={glassChild};
    arbitgpu::smoothglass::Prepared mixedGlass;
    const std::shared_ptr<const gt::Visual3DScene> snapshot=fixture.scene;
    const std::shared_ptr<const gt::Program> mixedProgram=mixedRoot;
    check(arbitgpu::validNativeFixtureMaterialPrograms(*mixedRoot,*fixture.scene,mixedRoot->backend,true,discrete),
        "actual Diffraction root with flat closed-glass Surface child satisfies existing mixed authority");
    check(!arbitgpu::smoothglass::prepare(snapshot,mixedProgram,{},mixedGlass,error),"default glass admission refuses private discrete grating");
    const bool mixedAdmitted=arbitgpu::smoothglass::prepare(snapshot,mixedProgram,{},mixedGlass,error,{},&selection);
    check(mixedAdmitted,"explicit real compiler mixed owner admits closed glass and planar grating: "+error);if(!mixedAdmitted)return;
    auto wrongOrientation=std::make_shared<gt::Program>(*fixture.program);wrongOrientation->objectPrograms[0]=compiledGrating;
    check(!arbitgpu::smoothglass::prepare(snapshot,wrongOrientation,{},mixedGlass,error,{},&selection),
        "Surface root with Diffraction child honestly refuses instead of inventing a new mixed authority");
    gc::Prepared mixedPrepared;gc::Frame mixedFrame;
    check(gc::prepare(mixedGlass,fixture.ordinaryLight,*fixture.policy,mixedPrepared,error),"real mixed staging retains source, revisions and exact material identities: "+error);
    check(!gc::buildFrame(mixedPrepared,fixture.frame.instances,32,32,mixedFrame,error),"ordinary frame builder keeps combined Color guarded");
    check(gc::buildGratingStagingFrame(mixedPrepared,fixture.frame.instances,nullptr,32,32,{},mixedFrame,error)
        &&gc::validateTables(mixedPrepared,mixedFrame,error,true),"private mixed staging builds existing query tables with real zero Surface blocks: "+error);
    check(!gc::validateTables(mixedPrepared,mixedFrame,error),"mixed staging cannot enter default Color table gate");
    std::shared_ptr<const gc::GratingIncidentBinding> mixedBinding;
    check(gc::prepareGratingIncidentBinding(mixedPrepared,mixedFrame,compiledGrating,material,.25,mixedBinding,error),
        "actual compiled distinct owner freezes exact IR/UV witness: "+error);
    check(mixedFrame.query.texels==legacyQuery&&mixedFrame.workTexels==fixture.frame.workTexels,
        "mixed material admission preserves existing v3 geometry query and smooth work bytes");
    auto sameSelection=selection;sameSelection.object=fixture.prepared.glass.object;
    check(!arbitgpu::smoothglass::prepare(snapshot,mixedProgram,{},mixedGlass,error,{},&sameSelection),"glass cannot also own the distinct grating IR");
    gc::NativeDeposit extension;
    check(gc::decodeGratingDepositExtension({}, {},extension)&&!extension.gratingVersion&&extension.gratingIncident==UINT32_MAX,
        "legacy deposit reserved bytes preserve null coupled provenance");
    check(gc::decodeGratingDepositExtension({float(gc::kGratingIncidentVersion),0,0,0},{},extension)&&extension.gratingOrder==-1,
        "negative signed order has explicit coupled deposit encoding");
    check(!gc::decodeGratingDepositExtension({float(gc::kGratingIncidentVersion),0,.5f,0},{},extension)
        &&!gc::decodeGratingDepositExtension({float(gc::kGratingIncidentVersion+1),0,1,0},{},extension),"fractional order and unknown coupled deposit version refuse");
    std::array<std::uint8_t,32> decodedAncestry{};std::array<gc::Texel,8> packedAncestry{};
    packedAncestry[0]={65,1,1,65};
    check(gc::decodeGratingDepositAncestry(packedAncestry,decodedAncestry)&&decodedAncestry[3]==65,
        "coupled smooth ancestry retains exact transmitted and reflected decisions");
    packedAncestry[0][1]=2;
    check(!gc::decodeGratingDepositAncestry(packedAncestry,decodedAncestry),"microfacet exit token cannot masquerade as smooth coupled ancestry");
    gc::NativeGratingIncident event;event.version=gc::kGratingIncidentVersion;event.complete=true;
    event.work=0;event.band=fixture.frame.work[0].band;event.interfaces=2;event.ancestry[0]=event.ancestry[1]=65;
    event.fraction=.5f;event.power=fixture.frame.work[0].incidentPower*event.fraction;
    const auto& triangle=binding->triangles.front();event.barycentric={.25f,.25f,.5f};
    event.normal=event.hit.outwardNormal=triangle.normal;event.tangent=triangle.tangent;event.bitangent=triangle.bitangent;
    for(unsigned v=0;v<3;++v) {
        event.hit.position=gc::advance(event.hit.position,triangle.positions[v],event.barycentric[v]);
        for(unsigned a=0;a<2;++a)event.uv[a]+=triangle.uv[v][a]*event.barycentric[v];
    }
    event.hit.status=1;event.hit.entering=true;event.hit.work=1;event.hit.distance=1;
    event.hit.objectId=triangle.object;event.hit.materialId=triangle.material;
    event.hit.instanceId=triangle.instance;event.hit.triangleId=triangle.triangle;
    event.origin=gc::advance(event.hit.position,event.normal,1);event.direction=event.normal;for(float& v:event.direction)v=-v;
    const auto& entry=fixture.frame.entries[fixture.frame.work[event.work].entry];
    const auto entryTriangle=std::find_if(fixture.frame.triangles.begin(),fixture.frame.triangles.end(),[&](const auto& t) {
        return std::tie(t.object,t.material,t.instance,t.triangle)==std::tie(entry.object,entry.material,entry.instance,entry.triangle);});
    check(entryTriangle!=fixture.frame.triangles.end(),"synthetic planar differential retains exact entry owner");if(entryTriangle==fixture.frame.triangles.end())return;
    event.dHdu=gc::sub(entryTriangle->points[1],entryTriangle->points[0]);event.dHdv=gc::sub(entryTriangle->points[2],entryTriangle->points[0]);
    gc::GratingReceivingArea area;
    check(gc::gratingReceivingArea(*binding,event,area)&&close(area.jacobian,1)&&close(area.area,entry.area)&&!area.squareMetres,
        "parallel planar synthetic witness has unit magnification without conferring metric units on legacy radiance");
    if(dBinding) {
        auto physicalEvent=event;physicalEvent.band=dBinding->frame.work[physicalEvent.work].band;
        check(gc::gratingReceivingArea(*dBinding,physicalEvent,area)&&area.squareMetres,
            "only explicit authored metre-world D labels the receiving area m2");
        const double flux=dBinding->frame.work[physicalEvent.work].incidentPower*physicalEvent.fraction;
        check(close((flux/area.area)*area.area,flux,1e-12),"incident W/nm divides by receiving m2 once and conserves flux");
    }
    auto staleArea=*binding;staleArea.frame.entries[0].triangle=UINT32_MAX;
    area.area=123;
    check(!gc::gratingReceivingArea(staleArea,event,area)&&area.area==123,"missing exact entry owner refuses area atomically");
    event.substrateReflectedPower=event.power*.8f;event.absorbedPower=event.power*.2f;event.orderCount=1;
    event.orders[0].direction=event.normal;event.orders[0].power=event.substrateReflectedPower;
    gc::SubmissionIdentity submission{binding->identity,1,2,3,4,5,2};
    gc::NativeGratingReadback empty;empty.complete=true;empty.submission=submission;
    check(gc::validateGratingIncidentReadback(*binding,empty,submission,error),"empty incident subset retains completed native owner");
    gc::SubmissionIdentity missing;missing.frameIdentity=binding->identity;empty.submission=missing;
    check(!gc::validateGratingIncidentReadback(*binding,empty,missing,error),"empty incident subset cannot omit native generations and fence");
    auto incomplete=submission;incomplete.completedFence=0;empty.submission=incomplete;
    check(!gc::validateGratingIncidentReadback(*binding,empty,incomplete,error),"empty incident subset requires completed exact native fence");
    check(gc::validateGratingIncident(*binding,event,submission,submission,error),"synthetic exact ray/triangle/UV/flux incident contract is consistent");
    const auto reject=[&](gc::NativeGratingIncident changed,const char* message) {
        check(!gc::validateGratingIncident(*binding,changed,submission,submission,error)&&!error.empty(),message);
    };
    auto changed=event;changed.band=(event.band+1)%8;reject(changed,"foreign incident band refused");
    changed=event;changed.version++;reject(changed,"unknown incident version refused");
    changed=event;changed.hit.triangleId=UINT32_MAX;reject(changed,"foreign queried triangle refused");
    changed=event;changed.uv[0]+=.1f;reject(changed,"nearby sample UV cannot replace actual-hit UV");
    changed=event;changed.barycentric[0]+=.1f;reject(changed,"inconsistent barycentric witness refused");
    changed=event;changed.direction[0]+=.1f;reject(changed,"nonunit propagation refused");
    changed=event;changed.dHdu={};reject(changed,"collapsed native differential refuses rather than substituting triangle area");
    changed=event;changed.dHdv[2]=.1f;reject(changed,"native hit differential must lie on exact queried grating plane");
    changed=event;changed.barycentric={0,.5f,.5f};reject(changed,"shared triangle edge refuses local branch area");
    changed=event;changed.bitangent=gc::Point{};reject(changed,"invalid handed grating frame refused");
    changed=event;changed.ancestry[0]=1;reject(changed,"exact ancestry and reflection history must agree");
    changed=event;changed.ancestry[31]=65;reject(changed,"unused ancestry suffix must remain zero");
    changed=event;changed.orders[0].power*=2;reject(changed,"resolved reflected flux cannot be renormalized");
    changed=event;changed.orders[0].order=1;reject(changed,"signed order must satisfy existing spectral tangential equation");
    changed=event;changed.orderCount=2;changed.orders[1]=changed.orders[0];reject(changed,"duplicate signed order refused");
    auto stale=submission;stale.queryGeneration++;
    check(!gc::validateGratingIncident(*binding,event,submission,stale,error),"foreign native query generation refused");
    gc::NativeGratingReadback readback;readback.submission=submission;readback.complete=true;
    readback.incidents={event};readback.queryWork=1;readback.orderWork=3;
    check(gc::validateGratingIncidentReadback(*binding,readback,submission,error),"synthetic incident batch uses bounded three-candidate work");
    auto omitted=std::make_shared<gc::NativeGratingReadback>();omitted->submission=submission;
    omitted->complete=omitted->childrenComplete=true;omitted->orderWork=3;
    auto weakParent=event;weakParent.fraction=5e-5f;weakParent.power=fixture.frame.work[0].incidentPower*weakParent.fraction;
    weakParent.substrateReflectedPower=weakParent.power*.8f;weakParent.absorbedPower=weakParent.power*.2f;
    weakParent.orderCount=3;weakParent.orders={};
    const float step=material->description().spectrum.wavelengthsNanometres[weakParent.band]
        /material->description().geometry.grooveSpacingNanometres;
    for(unsigned o=0;o<3;++o) {
        auto& order=weakParent.orders[o];order.order=int(o)-1;const float x=order.order*step;
        order.direction=gc::advance(gc::advance(gc::Point{},weakParent.normal,std::sqrt(1-x*x)),weakParent.tangent,x);
        order.power=weakParent.substrateReflectedPower/3;
    }
    omitted->incidents={weakParent};
    omitted->paths.resize(fixture.frame.work.size());for(auto& path:omitted->paths)path.complete=true;
    omitted->paths[0].incidentCount=1;omitted->paths[0].orderWork=3;
    omitted->paths[0].absorbedPower=weakParent.absorbedPower;omitted->paths[0].unresolvedPower=weakParent.substrateReflectedPower;
    for(const auto& order:weakParent.orders) {
        gc::NativeGratingReadback::ChildReceipt child;child.complete=true;child.incident=0;child.order=order.order;
        child.disposition=gc::NativeGratingReadback::Disposition::Unresolved;child.hit.status=0;
        child.unresolvedPower=order.power;omitted->children.push_back(child);
    }
    gc::NativeReadback omittedNative;omittedNative.gratingIncidentVersion=gc::kGratingIncidentVersion;omittedNative.gratingIncidents=omitted;
    omittedNative.paths.resize(fixture.frame.work.size());omittedNative.paths[0].visits=1;
    check(gc::validateGratingTerminalReadback(*binding,omittedNative,submission,error),
        "weak three-order parent retains bounded unresolved flux without charging nonexistent child queries");
    auto incidentOnly=*omitted;incidentOnly.childrenComplete=false;incidentOnly.paths.clear();
    check(!gc::validateGratingIncidentReadback(*binding,incidentOnly,submission,error),
        "incident-only producer still requires every positive order query");
    readback.incidents.push_back(event);readback.orderWork=6;
    check(!gc::validateGratingIncidentReadback(*binding,readback,submission,error),"duplicate actual parent branch refused");
    readback.incidents={event};readback.orderWork=3;readback.incidents[0].unresolvedPower=event.power*.1f;
    readback.incidents[0].orders[0].power-=readback.incidents[0].unresolvedPower;
    check(!gc::validateGratingIncidentReadback(*binding,readback,submission,error),"unresolved grating tail cannot bypass original frame fraction");
    // Binary64 scalar-relief equations independently audit the weak but real
    // .9nm orders. These synthetic receipts exercise ledger validation only;
    // their missed child queries do not assert native closest-hit results.
    const auto orders=[&](gc::NativeGratingIncident& incident,double height) {
        const auto& d=material->description();constexpr double pi=3.14159265358979323846;
        const double n=d.substrate.refractiveIndex,k=d.substrate.extinctionCoefficient;
        const double reflected=((1-n)*(1-n)+k*k)/((1+n)*(1+n)+k*k);
        const double wavelength=d.spectrum.wavelengthsNanometres[incident.band];
        const double cosine=-gc::dot(incident.direction,incident.normal);
        incident.substrateReflectedPower=float(incident.power*reflected);
        incident.absorbedPower=float(incident.power*(1-reflected));incident.orderCount=0;double resolved=0;
        incident.orders={};
        for(int order=-1;order<=1;++order) {
            const double x=gc::dot(incident.direction,incident.tangent)+order*wavelength/d.geometry.grooveSpacingNanometres;
            const double y=gc::dot(incident.direction,incident.bitangent);
            if(x*x+y*y>=1)continue;
            const double z=std::sqrt(1-x*x-y*y),phase=2*pi*height*(cosine+z)/wavelength;
            const double efficiency=order?4*std::pow(std::sin(phase/2)/pi,2)*z/cosine:std::pow(std::cos(phase/2),2);
            auto& output=incident.orders[incident.orderCount++];output.order=order;
            for(unsigned axis=0;axis<3;++axis)output.direction[axis]=float(x*incident.tangent[axis]+y*incident.bitangent[axis]+z*incident.normal[axis]);
            output.power=float(incident.power*reflected*efficiency);resolved+=output.power;
        }
        incident.unresolvedPower=float(std::max(0.0,double(incident.substrateReflectedPower)-resolved));
    };
    auto largePreset=event;orders(largePreset,125);
    check(largePreset.unresolvedPower>largePreset.power*gc::kMaximumUnresolvedFraction,
        "125nm three-order preset has a real refused tail");
    gc::NativeReadback terminal;const bool terminalPrepared=gt::syntheticReadback(fixture,terminal,error);
    check(terminalPrepared,"legacy synthetic full path prepares grating ledger input: "+error);if(!terminalPrepared)return;
    auto full=std::make_shared<gc::NativeGratingReadback>();full->submission=submission;full->complete=full->childrenComplete=true;
    full->paths.resize(fixture.frame.work.size());for(auto& path:full->paths)path.complete=true;
    double positivePower=0,largestTail=0;
    for(std::size_t w=0;w<fixture.frame.work.size();++w) {
        gc::BranchReceipt branches;
        check(gc::traceBranches(fixture.prepared,fixture.frame,fixture.frame.entries[fixture.frame.work[w].entry],
            fixture.frame.work[w].band,*fixture.policy,branches,error),"original glass tail remains in coupled diagnostic: "+error);
        auto& path=terminal.paths[w];auto& adjunct=full->paths[w];
        path.depositedPower=path.absorbedPower=0;path.depositCount=0;
        for(const auto& deposit:terminal.deposits)if(deposit.work==w&&deposit.footprint.projected)
            path.queryWork-=std::uint64_t(deposit.cameraHit.work)+deposit.pixelHit.work;
        for(const auto& branch:branches.deposits) {
            gc::NativeGratingIncident incident;incident.version=gc::kGratingIncidentVersion;incident.complete=true;incident.work=std::uint32_t(w);
            incident.band=fixture.frame.work[w].band;incident.interfaces=std::uint32_t(branch.interfaces);
            incident.reflectionHistory=branch.reflectionHistory;incident.fraction=float(branch.fraction);
            incident.power=fixture.frame.work[w].incidentPower*incident.fraction;incident.hit=branch.receiver;
            for(unsigned depth=0;depth<incident.interfaces;++depth)incident.ancestry[depth]=(incident.reflectionHistory>>depth)&1?1:65;
            const auto witness=std::find_if(binding->triangles.begin(),binding->triangles.end(),[&](const auto& t){return t.triangle==incident.hit.triangleId;});
            if(witness==binding->triangles.end()){check(false,"synthetic grating triangle has frozen witness");continue;}
            incident.normal=witness->normal;incident.tangent=witness->tangent;incident.bitangent=witness->bitangent;
            incident.direction=branch.incidentDirection;incident.origin=gc::advance(incident.hit.position,incident.direction,-incident.hit.distance);
            const auto a=gc::sub(witness->positions[1],witness->positions[0]),b=gc::sub(witness->positions[2],witness->positions[0]);
            const auto delta=gc::sub(incident.hit.position,witness->positions[0]);
            const double aa=gc::dot(a,a),bb=gc::dot(b,b),ab=gc::dot(a,b),den=aa*bb-ab*ab;
            const float v=float((bb*gc::dot(delta,a)-ab*gc::dot(delta,b))/den),u=float((aa*gc::dot(delta,b)-ab*gc::dot(delta,a))/den);
            incident.barycentric={1-v-u,v,u};
            for(unsigned corner=0;corner<3;++corner)for(unsigned axis=0;axis<2;++axis)
                incident.uv[axis]+=incident.barycentric[corner]*witness->uv[corner][axis];
            // Axis-aligned smooth slab and parallel grating give the exact
            // identity derivative for every reflected/transmitted macro path.
            const auto& entrySample=fixture.frame.entries[fixture.frame.work[w].entry];
            const auto sourceTriangle=std::find_if(fixture.frame.triangles.begin(),fixture.frame.triangles.end(),[&](const auto& t) {
                return std::tie(t.object,t.material,t.instance,t.triangle)==std::tie(entrySample.object,entrySample.material,entrySample.instance,entrySample.triangle);});
            if(sourceTriangle==fixture.frame.triangles.end()){check(false,"synthetic slab differential has exact owner");continue;}
            incident.dHdu=gc::sub(sourceTriangle->points[1],sourceTriangle->points[0]);incident.dHdv=gc::sub(sourceTriangle->points[2],sourceTriangle->points[0]);
            orders(incident,.9);check(incident.orderCount>0,"nonzero .9nm profile has propagating native candidates");
            if(incident.fraction>.5f)check(incident.orderCount==3,"dominant authored glass branch retains both nonzero signed orders");
            ++adjunct.incidentCount;adjunct.orderWork+=3;adjunct.absorbedPower+=incident.absorbedPower;adjunct.unresolvedPower+=incident.unresolvedPower;
            const auto index=std::uint32_t(full->incidents.size());full->incidents.push_back(incident);
            for(unsigned o=0;o<incident.orderCount;++o) {
                const auto& output=incident.orders[o];if(output.order)positivePower+=output.power;
                gc::NativeGratingReadback::ChildReceipt child;child.incident=index;child.order=output.order;child.complete=true;
                child.disposition=gc::NativeGratingReadback::Disposition::Escaped;child.hit.status=0;child.hit.work=1;
                child.hit.distance=fixture.frame.sourceDirectionTrace[3];child.queryWork=1;child.escapedPower=output.power;
                full->children.push_back(child);++adjunct.queryWork;++path.queryWork;++path.visits;path.escapedPower+=output.power;
            }
        }
        path.absorbedPower=adjunct.absorbedPower;path.unresolvedPower+=adjunct.unresolvedPower;
        full->orderWork+=adjunct.orderWork;full->queryWork+=adjunct.queryWork;
        if(path.launchedPower)largestTail=std::max(largestTail,double(path.unresolvedPower)/path.launchedPower);
    }
    terminal.deposits.clear();std::fill(terminal.spectralRadiance.begin(),terminal.spectralRadiance.end(),0);
    marked.identity=binding->identity;terminal.frameIdentity=marked.identity;terminal.gratingIncidentVersion=gc::kGratingIncidentVersion;terminal.gratingIncidents=full;
    terminal.deposition=gt::syntheticDeposition(marked,terminal,submission);
    check(positivePower>0&&largestTail<=gc::kMaximumUnresolvedFraction,"real weak signed orders retain original whole-path glass plus grating tail");
    check(gc::validateNativeReadback(fixture.prepared,marked,terminal,ledger,error,&submission,true),
        "private coupled full-path ledger conserves exact incident/order/absorption/query subsets: "+error);
    if(full->children.empty()){check(false,"bounded positive diagnostic has actual signed children");return;}
    auto duplicate=*full;duplicate.children.push_back(duplicate.children.front());terminal.gratingIncidents=std::make_shared<const gc::NativeGratingReadback>(duplicate);
    check(!gc::validateGratingTerminalReadback(*binding,terminal,submission,error),"duplicate signed order child refuses");
    auto missingChildren=*full;missingChildren.childrenComplete=false;terminal.gratingIncidents=std::make_shared<const gc::NativeGratingReadback>(missingChildren);
    check(!gc::validateNativeReadback(fixture.prepared,marked,terminal,ledger,error,&submission,true),"incident-only native prerequisite cannot satisfy full ledger");
    auto wrongWork=*full;wrongWork.paths[0].queryWork++;terminal.gratingIncidents=std::make_shared<const gc::NativeGratingReadback>(wrongWork);
    check(!gc::validateGratingTerminalReadback(*binding,terminal,submission,error),"extra queries cannot be counted twice or lost between work and children");
    auto badAncestry=*full;badAncestry.incidents[0].ancestry[0]=0;terminal.gratingIncidents=std::make_shared<const gc::NativeGratingReadback>(badAncestry);
    check(!gc::validateGratingTerminalReadback(*binding,terminal,submission,error),"child parent exact ancestry cannot be replaced by an identifier");
    auto overflowing=*full;overflowing.queryWork=binding->bounds.queryWork+1;terminal.gratingIncidents=std::make_shared<const gc::NativeGratingReadback>(overflowing);
    check(!gc::validateGratingTerminalReadback(*binding,terminal,submission,error),"coupled query work shares the unchanged global bound");
    terminal.gratingIncidents=full;auto broken=terminal;broken.paths[0].absorbedPower=0;
    check(!gc::validateNativeReadback(fixture.prepared,marked,broken,ledger,error,&submission,true),"substrate absorption is included exactly once in the original path");
    check(!gc::validateNativeReadback(fixture.prepared,marked,terminal,ledger,error,&submission),"full typed diagnostic still cannot bypass production Color guard");
    gc::GratingIncidentBounds sentinel;sentinel.bytes=123;auto huge=fixture.frame;huge.bounds.transientBytes=renderpassoutput::kMaximumTotalBytes;
    check(!gc::calculateGratingIncidentBounds(huge,2,sentinel,error)&&sentinel.bytes==123,"additional storage overflow refuses atomically");
    auto wrongOwner=std::make_shared<gt::Program>(*owner);wrongOwner->materialRevision++;
    auto preserved=binding;
    check(!gc::prepareGratingIncidentBinding(fixture.prepared,fixture.frame,wrongOwner,material,.25,preserved,error)&&preserved==binding,
        "stale override material preserves incumbent binding");
    wrongOwner=std::make_shared<gt::Program>(*owner);wrongOwner->layoutVersion++;
    check(!gc::prepareGratingIncidentBinding(fixture.prepared,fixture.frame,wrongOwner,material,.25,preserved,error),
        "unknown native material layout cannot freeze an incident owner");
    wrongOwner=std::make_shared<gt::Program>(*owner);wrongOwner->object=fixture.prepared.glass.object;
    check(!gc::prepareGratingIncidentBinding(fixture.prepared,fixture.frame,wrongOwner,material,.25,preserved,error),
        "same-object grating cannot replace the closed volume");
    auto roughFrame=fixture.frame;roughFrame.roughness=.1f;
    check(!gc::prepareGratingIncidentBinding(fixture.prepared,roughFrame,owner,material,.25,preserved,error),
        "rough branch cannot enter the smooth incident prerequisite");
    gc::Point tangent,bitangent,normal;
    auto mirrored=triangle.uv;for(auto& uv:mirrored)uv[0]=-uv[0];
    check(!gc::gratingFrame(triangle.positions,mirrored,tangent,bitangent,normal),"mirrored UV orientation refuses instead of reversing signed orders");
    std::shared_ptr<const gc::GratingIncidentBinding> later;
    const bool laterPrepared=gc::prepareGratingIncidentBinding(fixture.prepared,fixture.frame,owner,material,.5,later,error);
    check(laterPrepared,"later evaluation time prepares with unchanged exact owners: "+error);
    check(laterPrepared&&later->identity!=binding->identity&&later->evaluationTime==.5&&binding->evaluationTime==.25,
        "existing request evaluation time is frozen in incident identity");
    std::shared_ptr<const gc::GratingIncidentBinding> repeated,adjacent;
    check(gc::prepareGratingIncidentBinding(fixture.prepared,fixture.frame,owner,material,.25,repeated,error)
        &&repeated->identity==binding->identity,"same exact evaluation time retains deterministic incident identity");
    const double adjacentTime=std::nextafter(.25,.5);
    check(gc::prepareGratingIncidentBinding(fixture.prepared,fixture.frame,owner,material,adjacentTime,adjacent,error)
        &&adjacent->identity!=binding->identity&&adjacent->evaluationTime==adjacentTime,
        "adjacent binary64 evaluation time retains low bits in incident identity");
    owner->diffractionPaths[0].material.geometry.z+=1;
    check(binding->gratingOwner->diffractionPaths[0].material.geometry.z!=owner->diffractionPaths[0].material.geometry.z,
        "mutable caller grating bytes cannot change frozen material owner");
    check(!gc::prepareGratingIncidentBinding(fixture.prepared,fixture.frame,owner,material,.25,preserved,error),"IR and native material mismatch refuses");
}
class PreparedDirectGratingFrame final : public PreparedSourceFrame,public gc::NativeDirectGratingIrradianceBinding
{
public:
    std::shared_ptr<const gc::GratingIncidentBinding> binding;
    gc::NativeDirectGratingEndpointUpload endpoints;
    gc::SubmissionIdentity productionSubmission;
    gc::NativeDirectGratingContributionReadback productionReadback;
    bool productionComplete=true,contextCurrent=true;
    const gc::NativeDirectionalSourceBinding& directSourceBinding() const noexcept override{return *this;}
    const std::shared_ptr<const gc::GratingIncidentBinding>& directGratingBinding() const noexcept override{return binding;}
    const gc::NativeDirectGratingEndpointUpload& directEndpointUpload() const noexcept override{return endpoints;}
    bool validateDirectProductionQuery(const gc::SubmissionIdentity& expected,
        const gc::NativeDirectGratingContributionReadback& supplied,std::string& error) const override
    {
        if(!productionComplete||!contextCurrent||!(productionSubmission==expected)) {
            error="Synthetic production query/context refuses";return false;
        }
        if(!gc::sameDirectGratingContributionReadback(supplied,productionReadback)) {
            error="Supplied contribution differs from retained synthetic native result";return false;
        }
        error.clear();return true;
    }
};
// Explicitly mocked common-frame/native cohort authority. These retained host
// records exercise the contract; they establish no GPU query or native sum.
bool sameTerminalLedger(const gc::DirectGratingTerminalOrderLedger&,const gc::DirectGratingTerminalOrderLedger&,bool compareNativeResult=true);
class PreparedDirectGratingCohort final : public PreparedSourceFrame,public gc::NativeDirectGratingIrradianceCohortBinding,
    public gc::NativeCameraCellCoverageBinding
{
public:
    std::shared_ptr<const gc::DirectGratingIrradianceCohort> cohort;
    std::vector<std::shared_ptr<PreparedDirectGratingFrame>> batches;
    gc::SubmissionIdentity sealed;
    bool current=true,reserved=true;
    bool terminalBudgetSupported=false,unregisteredLease=false,missingResult=false,failFinish=false,failResultInspection=false,retainedCapacityFault=false;
    bool missingLease=false,reservationSupported=true,finishSupported=true,resultInspectionSupported=true;
    bool retainedPayloadFault=false;
    const PreparedDirectGratingCohort* wrongResultOwner=nullptr;
    const arbitgpu::NativeFixtureSceneFrame* wrongLeaseOwner=nullptr;
    mutable gc::DirectGratingTerminalNativeBudget terminalBudget;
    mutable std::shared_ptr<const gc::NativeDirectGratingTerminalResult> terminalResult,stagedTerminalResult;
    mutable const gc::NativeDirectGratingTerminalLease* activeTerminalLease=nullptr;
    mutable gc::DirectGratingTerminalOrderBounds terminalReservation;
    mutable bool terminalReserved=false;
    // Receiver fixtures model contract shape only, never actual native queries.
    bool receiverSupported=false;
    unsigned receiverFault=0;
    mutable gc::DirectGratingTerminalReceiverNativeBudget receiverBudget;
    mutable gc::DirectGratingTerminalReceiverBounds receiverReservation;
    mutable gc::DirectGratingTerminalReceiverReadback dispatchedReceivers;
    mutable std::shared_ptr<const gc::NativeDirectGratingTerminalReceiverResult> receiverResult,stagedReceiverResult;
    mutable const gc::NativeDirectGratingTerminalReceiverLease* activeReceiverLease=nullptr;
    mutable bool receiverReserved=false;
    // No real backend implements this new trait yet. These are lifecycle mocks.
    bool coverageSupported=false;
    unsigned coverageFault=0;
    mutable gc::NativeCameraCellCoverageBudget coverageBudget;
    mutable gc::CameraCellCoverageBounds coverageReservation;
    mutable gc::CameraCellCoverageReadback dispatchedCoverage;
    mutable std::shared_ptr<const gc::NativeCameraCellCoverageResult> coverageResult,stagedCoverageResult;
    mutable const gc::NativeCameraCellCoverageLease* activeCoverageLease=nullptr;
    mutable bool coverageReserved=false;
    class CoverageLease final : public gc::NativeCameraCellCoverageLease
    {
    public:
        explicit CoverageLease(const PreparedDirectGratingCohort& frame):frame(frame),priorBytes(frame.coverageBudget.retainedBytes){}
        ~CoverageLease() override {
            if(!committed)frame.coverageBudget.retainedBytes=priorBytes;
            frame.stagedCoverageResult.reset();frame.dispatchedCoverage={};
            frame.activeCoverageLease=nullptr;frame.coverageReserved=false;
        }
        const arbitgpu::NativeFixtureSceneFrame& owner() const noexcept override{return frame;}
        const gc::Frame& frozenFrame() const noexcept override{return frame.cohort->sources()[0]->frame;}
        std::uint64_t inputGeneration() const noexcept override{return 101;}
        std::uint64_t proofGeneration() const noexcept override{return 103;}
    protected:
        void commit() noexcept override {frame.coverageResult=frame.stagedCoverageResult;committed=true;}
    private:
        const PreparedDirectGratingCohort& frame;std::uint64_t priorBytes;bool committed=false;
    };
    class ReceiverLease final : public gc::NativeDirectGratingTerminalReceiverLease
    {
    public:
        explicit ReceiverLease(const PreparedDirectGratingCohort& frame):frame(frame),priorBytes(frame.receiverBudget.retainedBytes){}
        ~ReceiverLease() override {
            if(!committed)frame.receiverBudget.retainedBytes=priorBytes;
            frame.dispatchedReceivers={};frame.stagedReceiverResult.reset();
            frame.activeReceiverLease=nullptr;frame.receiverReserved=false;
        }
        const arbitgpu::NativeFixtureSceneFrame& owner() const noexcept override{return frame;}
        const gc::Frame& frozenFrame() const noexcept override{return frame.cohort->sources()[0]->frame;}
        std::uint64_t inputGeneration() const noexcept override{return 97;}
    protected:
        void commit() noexcept override {frame.receiverResult=frame.stagedReceiverResult;committed=true;}
    private:
        const PreparedDirectGratingCohort& frame;std::uint64_t priorBytes;bool committed=false;
    };
    class TerminalLease final : public gc::NativeDirectGratingTerminalLease
    {
    public:
        explicit TerminalLease(const PreparedDirectGratingCohort& frame):frame(frame),priorBytes(frame.terminalBudget.retainedBytes){}
        ~TerminalLease() override {
            if(!committed)frame.terminalBudget.retainedBytes=priorBytes;
            frame.stagedTerminalResult.reset();frame.activeTerminalLease=nullptr;frame.terminalReserved=false;
        }
        const arbitgpu::NativeFixtureSceneFrame& owner() const noexcept override{return frame.wrongLeaseOwner?*frame.wrongLeaseOwner:frame;}
    protected:
        void commit() noexcept override {frame.terminalResult=frame.stagedTerminalResult;committed=true;}
    private:
        const PreparedDirectGratingCohort& frame;std::uint64_t priorBytes;bool committed=false;
    };
    const arbitgpu::NativeFixtureSceneFrame& cohortOwner() const noexcept override{return *this;}
    const std::shared_ptr<const gc::DirectGratingIrradianceCohort>& directCohort() const noexcept override{return cohort;}
    std::size_t directSourceCount() const noexcept override{return batches.size();}
    const gc::NativeDirectGratingIrradianceBinding* directSourceBatch(std::size_t i) const noexcept override
    {return i<batches.size()?batches[i].get():nullptr;}
    bool validateDirectCohortProductionQueries(const gc::SubmissionIdentity& expected,
        const std::vector<gc::NativeDirectGratingContributionReadback>& supplied,std::string& error) const override
    {
        if(!current||!reserved||!(expected==sealed)||supplied.size()!=batches.size())
        {error="Mock native common frame or cumulative reservation refuses";return false;}
        for(std::size_t i=0;i<supplied.size();++i)
            if(!batches[i]||!gc::sameDirectGratingContributionReadback(supplied[i],batches[i]->productionReadback))
            {error="Mock retained cohort native result differs";return false;}
        error.clear();return true;
    }
    bool beginDirectCohortTerminalLease(const gc::SubmissionIdentity& expected,double time,
        const std::vector<gc::NativeDirectGratingContributionReadback>& supplied,
        std::unique_ptr<gc::NativeDirectGratingTerminalLease>& lease,
        gc::DirectGratingTerminalNativeBudget& output,std::string& error) const override
    {
        if(!terminalBudgetSupported)return gc::NativeDirectGratingIrradianceCohortBinding::beginDirectCohortTerminalLease(expected,time,supplied,lease,output,error);
        // Explicitly synthetic budget authority for shared semantic fixtures.
        // These numbers establish no real native retained bytes or work.
        if(activeTerminalLease||time!=cohort->sources()[0]->evaluationTime)
        {error="Mock terminal frame lease or time refuses";return false;}
        if(!validateDirectCohortProductionQueries(expected,supplied,error))return false;
        if(missingLease){output=terminalBudget;return true;}
        lease=std::make_unique<TerminalLease>(*this);if(!unregisteredLease)activeTerminalLease=lease.get();
        output=terminalBudget;return true;
    }
    bool reserveDirectCohortTerminalLease(gc::NativeDirectGratingTerminalLease& lease,
        const gc::DirectGratingTerminalOrderBounds& bounds,const gc::DirectGratingTerminalOrderLedger& incumbent,std::string& error) const override
    {
        if(!reservationSupported)return gc::NativeDirectGratingIrradianceCohortBinding::reserveDirectCohortTerminalLease(lease,bounds,incumbent,error);
        gc::DirectGratingTerminalOrderBounds exact;
        if(activeTerminalLease!=&lease||&lease.owner()!=this||!current||!reserved||terminalReserved
            ||!gc::calculateDirectGratingTerminalOrderBounds(*cohort,currentTerminalSources,terminalBudget,incumbent,exact,error)
            ||std::tie(exact.bytes,exact.combinedWork,exact.depositionWork,exact.maximumBands,exact.maximumOrders)
                !=std::tie(bounds.bytes,bounds.combinedWork,bounds.depositionWork,bounds.maximumBands,bounds.maximumOrders))
        {error="Mock terminal reservation lacks its registered exact native lease";return false;}
        terminalReservation=bounds;terminalReserved=true;
        terminalBudget={bounds.bytes,bounds.combinedWork,bounds.depositionWork};return true;
    }
    bool finishDirectCohortTerminalLease(gc::NativeDirectGratingTerminalLease& lease,
        const gc::DirectGratingTerminalOrderLedger& candidate,std::shared_ptr<const gc::NativeDirectGratingTerminalResult>& result,std::string& error) const override
    {
        if(!finishSupported)return gc::NativeDirectGratingIrradianceCohortBinding::finishDirectCohortTerminalLease(lease,candidate,result,error);
        if(activeTerminalLease!=&lease||!terminalReserved||failFinish)
        {error="Mock terminal finish refuses";return false;}
        if(missingResult){result.reset();return true;}
        result=wrongResultOwner?wrongResultOwner->makeDirectCohortTerminalResult(candidate):makeDirectCohortTerminalResult(candidate);
        stagedTerminalResult=result;
        if(retainedCapacityFault)const_cast<gc::DirectGratingTerminalOrderLedger&>(result->retained()).orders.reserve(2*terminalReservation.maximumOrders+1);
        if(retainedPayloadFault)const_cast<gc::DirectGratingTerminalOrderLedger&>(result->retained()).orders.front().orderIrradiance+=1;
        return true;
    }
    bool validateDirectCohortTerminalResult(const gc::NativeDirectGratingTerminalLease& lease,
        const gc::DirectGratingTerminalOrderLedger& candidate,std::string& error) const override
    {
        if(!resultInspectionSupported)return gc::NativeDirectGratingIrradianceCohortBinding::validateDirectCohortTerminalResult(lease,candidate,error);
        if(activeTerminalLease!=&lease||&lease.owner()!=this||!terminalReserved||!current||!reserved||failResultInspection
            ||!candidate.nativeResult||candidate.nativeResult!=stagedTerminalResult||candidate.nativeResult->owner()!=this
            ||!(candidate.incident.submission==sealed)||candidate.incident.cohort!=cohort)
        {error="Mock terminal result has no current registered staged owner";return false;}
        if(!sameTerminalLedger(candidate,candidate.nativeResult->retained(),false)){error="Mock returned and retained terminal payload differ";return false;}
        return true;
    }
    bool beginDirectTerminalReceiverLease(const std::shared_ptr<const gc::NativeDirectGratingTerminalResult>& terminal,
        const gc::SubmissionIdentity& expected,double time,std::unique_ptr<gc::NativeDirectGratingTerminalReceiverLease>& lease,
        gc::DirectGratingTerminalReceiverNativeBudget& budget,std::string& error) const override
    {
        if(!receiverSupported)return gc::NativeDirectGratingIrradianceCohortBinding::beginDirectTerminalReceiverLease(
            terminal,expected,time,lease,budget,error);
        if(activeReceiverLease||activeTerminalLease||!current||!reserved||terminal!=terminalResult
            ||!(expected==sealed)||time!=cohort->sources()[0]->evaluationTime||receiverFault==7)
        {error="Mock receiver begin lacks exact live scattering result";return false;}
        lease=std::make_unique<ReceiverLease>(*this);activeReceiverLease=lease.get();budget=receiverBudget;return true;
    }
    bool reserveDirectTerminalReceiverLease(gc::NativeDirectGratingTerminalReceiverLease& lease,
        const gc::DirectGratingTerminalReceiverBounds& bounds,const gc::DirectGratingTerminalReceiverReadback& incumbent,
        std::string& error) const override
    {
        gc::DirectGratingTerminalReceiverBounds exact;
        if(activeReceiverLease!=&lease||receiverReserved||!current||!reserved
            ||!gc::calculateDirectGratingTerminalReceiverBounds(*terminalResult,lease.frozenFrame(),receiverBudget,incumbent,exact,error)
            ||std::tie(exact.bytes,exact.combinedWork,exact.queryWork,exact.maximumRows,exact.perRayWork,
                exact.nativeInputBytes,exact.nativeOutputBytes,exact.nativeFixedBytes)
                !=std::tie(bounds.bytes,bounds.combinedWork,bounds.queryWork,bounds.maximumRows,bounds.perRayWork,
                bounds.nativeInputBytes,bounds.nativeOutputBytes,bounds.nativeFixedBytes))
        {error="Mock receiver reservation has no registered attempt";return false;}
        receiverReservation=bounds;receiverReserved=true;
        receiverBudget.retainedBytes=bounds.bytes;receiverBudget.combinedWork=bounds.combinedWork;return true;
    }
    bool dispatchDirectTerminalReceivers(gc::NativeDirectGratingTerminalReceiverLease& lease,
        const std::vector<gc::NativeDirectGratingTerminalReceiverInput>& inputs,
        gc::DirectGratingTerminalReceiverReadback& candidate,std::string& error) const override
    {
        if(activeReceiverLease!=&lease||!receiverReserved||receiverFault==11)
        {error="Mock receiver dispatch refuses";return false;}
        candidate.rows.reserve(receiverReservation.maximumRows);
        for(std::size_t i=0;i<inputs.size();++i) {
            gc::DirectGratingTerminalReceiverRow row;row.input=inputs[i];row.emitted=terminalResult->retained().orders[i];
            row.hit.status=0;row.hit.distance=inputs[i].directionMaximum[3];row.hit.work=1;row.queryWork=1;
            candidate.rows.push_back(row);++candidate.queryWork;
        }
        dispatchedReceivers=candidate;
        if(!candidate.rows.empty()) {
            auto& row=candidate.rows[0];
            if(receiverFault==1)++row.input.key[3];
            if(receiverFault==2)row.hit.status=-1;
            if(receiverFault==3)++candidate.queryWork;
            if(receiverFault==4)candidate.rows.pop_back();
            if(receiverFault==5)candidate.rows.reserve(2*receiverReservation.maximumRows+1);
            if(receiverFault==6)row.emitted.orderIrradiance+=1;
            if(receiverFault==8)row.rasterSample=gc::DirectGratingTerminalReceiverRow::RasterSample::Surface;
            if(receiverFault==9)++candidate.inputGeneration;
            if(receiverFault==10)row.hit.position[0]=std::numeric_limits<float>::quiet_NaN();
        }
        return true;
    }
    bool finishDirectTerminalReceiverLease(gc::NativeDirectGratingTerminalReceiverLease& lease,
        const gc::DirectGratingTerminalReceiverReadback& candidate,
        std::shared_ptr<const gc::NativeDirectGratingTerminalReceiverResult>& result,std::string& error) const override
    {
        if(activeReceiverLease!=&lease||!receiverReserved||receiverFault==12)
        {error="Mock receiver finish refuses";return false;}
        if(receiverFault==15){result.reset();return true;}
        result=makeDirectTerminalReceiverResult(candidate);stagedReceiverResult=result;
        if(receiverFault==14)const_cast<gc::DirectGratingTerminalReceiverReadback&>(result->retained()).rows[0].queryWork+=1;
        return true;
    }
    bool validateDirectTerminalReceiverResult(const gc::NativeDirectGratingTerminalReceiverLease& lease,
        const gc::DirectGratingTerminalReceiverReadback& candidate,std::string& error) const override
    {
        if(activeReceiverLease!=&lease||!receiverReserved||!current||candidate.terminal!=terminalResult
            ||!gc::sameDirectGratingTerminalReceiverReadback(candidate,dispatchedReceivers,false)
            ||(candidate.nativeResult&&(candidate.nativeResult!=stagedReceiverResult||receiverFault==13)))
        {error="Mock receiver result lacks exact registered dispatch";return false;}
        return true;
    }
    const arbitgpu::NativeFixtureSceneFrame& coverageOwner() const noexcept override{return *this;}
    bool beginCameraCellCoverageLease(const std::shared_ptr<const gc::NativeDirectGratingTerminalReceiverResult>& parent,
        const gc::SubmissionIdentity& expected,double time,std::unique_ptr<gc::NativeCameraCellCoverageLease>& lease,
        gc::NativeCameraCellCoverageBudget& budget,std::string& error) const override
    {
        if(!coverageSupported)return gc::NativeCameraCellCoverageBinding::beginCameraCellCoverageLease(parent,expected,time,lease,budget,error);
        if(activeCoverageLease||parent!=receiverResult||!current||!reserved||!(expected==sealed)
            ||time!=cohort->sources()[0]->evaluationTime||coverageFault==1)
        {error="Mock coverage parent/lease is stale";return false;}
        lease=std::make_unique<CoverageLease>(*this);activeCoverageLease=lease.get();budget=coverageBudget;return true;
    }
    bool reserveCameraCellCoverageLease(gc::NativeCameraCellCoverageLease& lease,const gc::CameraCellCoverageBounds& bounds,
        const gc::CameraCellCoverageReadback& incumbent,std::string& error) const override
    {
        gc::CameraCellCoverageBounds exact;
        if(activeCoverageLease!=&lease||coverageReserved||!gc::calculateCameraCellCoverageBounds(*receiverResult,
            lease.frozenFrame(),coverageBudget,incumbent,exact,error)||!gc::sameCameraCellCoverageBounds(bounds,exact))
        {error="Mock coverage admission has no registered exact attempt";return false;}
        coverageReservation=bounds;coverageReserved=true;coverageBudget.retainedBytes=bounds.bytes;
        coverageBudget.combinedWork=bounds.combinedWork;return true;
    }
    bool dispatchCameraCellCoverage(gc::NativeCameraCellCoverageLease& lease,const gc::CameraCellCoveragePreparation& preparation,
        gc::CameraCellCoverageProof& proof,std::string& error) const override
    {
        if(activeCoverageLease!=&lease||!coverageReserved||(coverageFault>=2&&coverageFault<=6))
        {error="Mock predicate refuses unsupported seam/gap/occlusion/depth/arithmetic";return false;}
        // Lifecycle positives have no admitted Pixel rows and certify no area.
        // Native finite-cell proof cannot be simulated by assigning a flag.
        if(!preparation.pixels.empty()){error="Mock has no native positive finite-cell predicate";return false;}
        proof.generation=lease.proofGeneration();proof.work=0;
        if(coverageFault==7)++proof.generation;
        if(coverageFault==8)proof.work=coverageReservation.operationWork+1;
        return true;
    }
    bool validateCameraCellCoverageProof(const gc::NativeCameraCellCoverageLease& lease,
        const gc::CameraCellCoverageReadback& value,std::string& error) const override
    {
        if(activeCoverageLease!=&lease||!coverageReserved||!current||value.parent!=receiverResult
            ||value.proof.generation!=lease.proofGeneration()||!value.preparation.pixels.empty()
            ||(value.nativeResult&&(coverageFault==10||value.nativeResult!=stagedCoverageResult
                ||!gc::sameCameraCellCoverageReadback(value,value.nativeResult->retained(),false))))
        {error="Mock coverage inspector refuses unregistered/stale payload";return false;}
        if(!value.nativeResult)dispatchedCoverage=value;
        else if(!gc::sameCameraCellCoverageReadback(value,dispatchedCoverage,false))
        {error="Mock coverage staged result differs from its dispatch";return false;}
        return true;
    }
    bool finishCameraCellCoverageLease(gc::NativeCameraCellCoverageLease& lease,const gc::CameraCellCoverageReadback& value,
        std::shared_ptr<const gc::NativeCameraCellCoverageResult>& result,std::string& error) const override
    {
        if(activeCoverageLease!=&lease||!coverageReserved||coverageFault==9)
        {error="Mock coverage staging refuses";return false;}
        if(coverageFault==11){result.reset();return true;}
        result=makeCameraCellCoverageResult(value);stagedCoverageResult=result;return true;
    }
    std::shared_ptr<const gc::NativeDirectGratingTerminalReceiverResult> preparedCoverageParent(
        const gc::DirectGratingTerminalReceiverReadback& value) const
    {return makeDirectTerminalReceiverResult(value);} // Geometry-only fixture owner.
    mutable std::vector<gc::NativeDirectGratingContributionReadback> currentTerminalSources;
};
bool sameTerminalLedger(const gc::DirectGratingTerminalOrderLedger& a,const gc::DirectGratingTerminalOrderLedger& b,bool compareNativeResult)
{
    if((compareNativeResult&&a.nativeResult!=b.nativeResult)||a.incident.cohort!=b.incident.cohort||!(a.incident.submission==b.incident.submission)
        ||a.incident.sourceMeasure!=b.incident.sourceMeasure||a.incident.endpoints.size()!=b.incident.endpoints.size()
        ||a.sources.size()!=b.sources.size()||a.bands.size()!=b.bands.size()||a.orders.size()!=b.orders.size()
        ||std::tie(a.bounds.bytes,a.bounds.combinedWork,a.bounds.depositionWork,a.bounds.maximumBands,a.bounds.maximumOrders)
            !=std::tie(b.bounds.bytes,b.bounds.combinedWork,b.bounds.depositionWork,b.bounds.maximumBands,b.bounds.maximumOrders))return false;
    for(std::size_t i=0;i<a.incident.endpoints.size();++i) {
        const auto& x=a.incident.endpoints[i];const auto& y=b.incident.endpoints[i];
        const auto& h=x.endpoint;const auto& k=y.endpoint;
        if(!gc::sameDirectGratingEndpointInput(x.input,y.input)||x.barycentric!=y.barycentric||x.surfaceIrradiance!=y.surfaceIrradiance
            ||std::tie(h.status,h.objectId,h.materialId,h.instanceId,h.triangleId,h.entering,h.work,h.distance,h.position,h.outwardNormal)
                !=std::tie(k.status,k.objectId,k.materialId,k.instanceId,k.triangleId,k.entering,k.work,k.distance,k.position,k.outwardNormal))return false;
    }
    for(std::size_t i=0;i<a.sources.size();++i)if(!gc::sameDirectGratingContributionReadback(a.sources[i],b.sources[i]))return false;
    for(std::size_t i=0;i<a.bands.size();++i) {
        const auto& x=a.bands[i];const auto& y=b.bands[i];
        if(std::tie(x.source,x.endpoint,x.band,x.witness,x.frame.tangent,x.frame.bitangent,x.frame.normal,x.toSource,x.uv,
            x.wavelengthNanometres,x.incidentIrradiance,x.substrateReflectedPower,x.resolvedReflectedPower,x.unresolvedReflectedPower,
            x.substrateReflectedIrradiance,x.resolvedReflectedIrradiance,x.unresolvedReflectedIrradiance,x.absorbedIrradiance,x.firstOrder,x.orderCount,x.responseEvaluated)
            !=std::tie(y.source,y.endpoint,y.band,y.witness,y.frame.tangent,y.frame.bitangent,y.frame.normal,y.toSource,y.uv,
            y.wavelengthNanometres,y.incidentIrradiance,y.substrateReflectedPower,y.resolvedReflectedPower,y.unresolvedReflectedPower,
            y.substrateReflectedIrradiance,y.resolvedReflectedIrradiance,y.unresolvedReflectedIrradiance,y.absorbedIrradiance,y.firstOrder,y.orderCount,y.responseEvaluated))return false;
    }
    for(std::size_t i=0;i<a.orders.size();++i) {
        const auto& x=a.orders[i];const auto& y=b.orders[i];
        if(std::tie(x.source,x.endpoint,x.band,x.order.primary,x.order.secondary,x.order.direction,x.order.reflectedPower,
            x.order.selectionProbability,x.order.angularStandardDeviationRadians,x.orderIrradiance)
            !=std::tie(y.source,y.endpoint,y.band,y.order.primary,y.order.secondary,y.order.direction,y.order.reflectedPower,
            y.order.selectionProbability,y.order.angularStandardDeviationRadians,y.orderIrradiance))return false;
    }
    return true;
}
void directGratingTerminalReceiverContract(PreparedDirectGratingCohort&,const gc::DirectGratingTerminalOrderLedger&);
void cameraCellCoverageContract(PreparedDirectGratingCohort&);
void directGratingTerminalOrderContract(PreparedDirectGratingCohort& common,
    const std::vector<gc::NativeDirectGratingContributionReadback>& receipts,bool differingDirections,
    gc::DirectGratingTerminalOrderLedger* retained=nullptr)
{
    // Real released order evaluator, with explicitly mocked native query and
    // budget boundaries. No native visibility, allocation or GPU proof follows.
    std::string error;const auto cohort=common.cohort;
    gc::DirectGratingTerminalOrderLedger ledger;ledger.bounds.bytes=123;
    const auto evaluate=[&](const auto& values,const auto& seal,double time) {
        return gc::evaluateDirectGratingTerminalOrders(cohort,common,seal,time,values,ledger,error);};
    if(!common.terminalBudgetSupported)
        check(!evaluate(receipts,common.sealed,.25)&&ledger.bounds.bytes==123
            &&error=="Directional terminal native lease is unsupported","old native consumers refuse terminal allocation without a native lease");
    common.terminalBudgetSupported=true;
    if(!common.terminalResult)common.terminalBudget={cohort->bounds().bytes+1024,cohort->bounds().combinedWork,cohort->bounds().depositionWork};
    common.currentTerminalSources=receipts;
    check(evaluate(receipts,common.sealed,.25),"terminal ledger uses released order math behind mock native/budget boundaries: "+error);
    if(ledger.bands.empty())return;
    check(ledger.bands.size()==cohort->sources().size()*cohort->inputs().size()*8
        &&ledger.bounds.maximumOrders==3*ledger.bands.size()&&ledger.orders.size()<=ledger.bounds.maximumOrders
        &&ledger.incident.cohort==cohort&&ledger.sources.size()==receipts.size()&&ledger.nativeResult==common.terminalResult
        &&ledger.nativeResult->owner()==&common&&!common.activeTerminalLease,
        "terminal complete source/endpoint/band identities and distinct retained native payload survive lease closure");
    check(ledger.nativeResult&&!ledger.nativeResult->retained().nativeResult
        &&ledger.nativeResult->retained().orders.data()!=ledger.orders.data()
        &&ledger.nativeResult->retained().bands.data()!=ledger.bands.data(),
        "returned and retained result arrays are independent charged copies without a receipt cycle");
    bool tiny=false,zero=false,expectTiny=false;
    for(const auto& receipt:receipts)for(const auto& sample:receipt.samples)for(float e:sample.surfaceIrradiance)
        expectTiny|=e>0&&e<1e-24;
    for(const auto& band:ledger.bands) {
        const auto& sample=receipts[band.source].samples[band.endpoint];const auto& binding=*cohort->sources()[band.source];
        check(band.incidentIrradiance==sample.surfaceIrradiance[band.band],"terminal uses already-cosined native E without Li projection or a second cosine");
        diffractionmaterial::DiffractionOrderDistribution distribution;
        if(!band.responseEvaluated) {
            check(!band.orderCount&&band.incidentIrradiance==0&&band.absorbedIrradiance==0,"nonincident supported zero retains its band and no invented order direction");zero=true;continue;
        }
        check(diffractionmaterial::evaluateDiffractionOrders(*binding.material,band.wavelengthNanometres,band.frame,band.toSource,band.uv,distribution,error)
            &&distribution.orderCount==band.orderCount,"each source direction uses the existing evaluator and released wavelength: "+error);
        double resolved=0;
        for(std::size_t o=0;o<band.orderCount;++o) {
            const auto& row=ledger.orders[band.firstOrder+o];const auto& expected=distribution.orders[o];
            check(row.source==band.source&&row.endpoint==band.endpoint&&row.band==band.band
                &&row.order.primary==expected.primary&&row.order.secondary==expected.secondary&&row.order.direction==expected.direction
                &&row.order.reflectedPower==expected.reflectedPower&&row.orderIrradiance==band.incidentIrradiance*expected.reflectedPower,
                "source/order/direction identity and per-area E times reflectedPower remain exact");
            resolved+=row.orderIrradiance;tiny|=row.orderIrradiance>0&&row.orderIrradiance<1e-24;
            zero|=band.incidentIrradiance==0&&row.orderIrradiance==0;
        }
        const auto relative=[](double a,double b){return std::abs(a-b)<=2e-10*std::max(a,b);};
        check(relative(resolved+band.unresolvedReflectedIrradiance,band.substrateReflectedIrradiance)
            &&relative(band.substrateReflectedIrradiance+band.absorbedIrradiance,band.incidentIrradiance)
            &&band.absorbedIrradiance==band.incidentIrradiance*(1-distribution.substrateReflectedPower)
            &&band.unresolvedReflectedIrradiance==band.incidentIrradiance*distribution.unresolvedReflectedPower,
            "terminal resolved/unresolved/absorbed shares conserve energy without a tiny-value floor");
    }
    check(tiny==expectTiny&&zero,"terminal ledger preserves tiny positive and supported-zero order energies");
    const auto stride=cohort->inputs().size()*8;
    const auto& first=ledger.bands[0];const auto& second=ledger.bands[stride];
    if(differingDirections&&second.responseEvaluated)check(first.toSource!=second.toSource&&first.substrateReflectedPower!=second.substrateReflectedPower
        &&ledger.orders[first.firstOrder].order.direction!=ledger.orders[second.firstOrder].order.direction,
        "different source directions retain different angular efficiencies and outgoing directions");
    else if(!differingDirections)check(first.toSource==second.toSource&&first.substrateReflectedPower==second.substrateReflectedPower
        &&ledger.orders[first.firstOrder].order.direction==ledger.orders[second.firstOrder].order.direction
        &&ledger.orders[first.firstOrder].source!=ledger.orders[second.firstOrder].source,
        "identical directions preserve separate source/order identities");
    else check(first.toSource!=second.toSource&&second.incidentIrradiance==0&&!second.orderCount,
        "backfacing source remains a distinct complete supported-zero band");
    const auto accepted=ledger;
    const auto rejects=[&](const auto& values,const auto& seal,double time,const char* message) {
        check(!evaluate(values,seal,time)&&!error.empty()&&sameTerminalLedger(ledger,accepted),message);
    };
    PreparedDirectGratingCohort foreignResultOwner;
    for(unsigned fault=0;fault<8;++fault) {
        const auto priorFrameResult=common.terminalResult;const auto priorBudget=common.terminalBudget;
        common.failFinish=fault==0;common.missingResult=fault==1;common.failResultInspection=fault==2;common.retainedCapacityFault=fault==3;
        common.finishSupported=fault!=4;common.resultInspectionSupported=fault!=5;
        common.wrongResultOwner=fault==6?&foreignResultOwner:nullptr;common.retainedPayloadFault=fault==7;
        rejects(receipts,common.sealed,.25,"admitted finish/result/copy failure preserves caller and retained frame result");
        check(common.terminalResult==priorFrameResult&&!common.stagedTerminalResult&&!common.activeTerminalLease
            &&common.terminalBudget.retainedBytes==priorBudget.retainedBytes
            &&common.terminalBudget.combinedWork==priorBudget.combinedWork+accepted.bounds.maximumBands+accepted.bounds.maximumOrders
            &&common.terminalBudget.depositionWork==priorBudget.depositionWork+accepted.bounds.maximumOrders,
            "failed terminal attempt releases reservation bytes but retains whole admitted work through lease destruction");
        common.failFinish=false;common.missingResult=false;common.failResultInspection=false;common.retainedCapacityFault=false;
        common.finishSupported=true;common.resultInspectionSupported=true;
        common.wrongResultOwner=nullptr;common.retainedPayloadFault=false;
    }
    PreparedSourceFrame foreignOwner;const auto beforeLeaseBudget=common.terminalBudget;
    common.wrongLeaseOwner=&foreignOwner;rejects(receipts,common.sealed,.25,"terminal lease naming another frame refuses before admission");common.wrongLeaseOwner=nullptr;
    common.unregisteredLease=true;rejects(receipts,common.sealed,.25,"same-address lifetime token cannot replace a registered native lease");common.unregisteredLease=false;
    common.missingLease=true;rejects(receipts,common.sealed,.25,"begin returning only totals cannot replace an actual native lease");common.missingLease=false;
    common.reservationSupported=false;rejects(receipts,common.sealed,.25,"default unsupported reservation refuses before attempted work");common.reservationSupported=true;
    {
        PreparedDirectGratingCohort::TerminalLease forged(common);
        check(!common.validateDirectCohortTerminalResult(forged,accepted,error),"a result value and unregistered lease cannot authenticate retained completion");
    }
    {
        std::unique_ptr<gc::NativeDirectGratingTerminalLease> live;gc::DirectGratingTerminalNativeBudget budget;
        check(common.beginDirectCohortTerminalLease(common.sealed,.25,receipts,live,budget,error)&&live
            &&!common.validateDirectCohortTerminalResult(*live,accepted,error),
            "a fresh registered lease cannot authenticate an old value without a reserved staged result");
    }
    check(common.terminalBudget.retainedBytes==beforeLeaseBudget.retainedBytes
        &&common.terminalBudget.combinedWork==beforeLeaseBudget.combinedWork&&!common.activeTerminalLease,
        "refused foreign/forged lease leaves frame bytes and work unchanged");
    auto missing=receipts;missing.pop_back();rejects(missing,common.sealed,.25,"terminal missing source refuses atomically");
    auto swapped=receipts;std::swap(swapped[0],swapped[1]);rejects(swapped,common.sealed,.25,"terminal source order refuses atomically");
    auto stale=common.sealed;stale.completedFence=0;rejects(receipts,stale,.25,"terminal stale common fence refuses atomically");
    rejects(receipts,common.sealed,.5,"terminal wrong evaluation time refuses atomically");
    auto edited=receipts;edited[1].samples[0].source.directionTrace[0]=std::nextafter(edited[1].samples[0].source.directionTrace[0],1.0f);
    common.batches[1]->productionReadback=edited[1];rejects(edited,common.sealed,.25,"even matching mock native result cannot change the frozen source direction");
    common.batches[1]->productionReadback=receipts[1];
    edited=receipts;edited[1].samples[0].surfaceIrradiance[0]=std::numeric_limits<float>::infinity();
    common.batches[1]->productionReadback=edited[1];rejects(edited,common.sealed,.25,"terminal nonfinite energy refuses atomically");
    common.batches[1]->productionReadback=receipts[1];
    edited=receipts;++edited[1].samples[0].endpoint.triangleId;
    common.batches[1]->productionReadback=edited[1];rejects(edited,common.sealed,.25,"terminal unknown endpoint primitive cannot supply a grating witness");
    common.batches[1]->productionReadback=receipts[1];
    edited=receipts;edited[1].overflow=true;
    common.batches[1]->productionReadback=edited[1];rejects(edited,common.sealed,.25,"terminal source overflow cannot become a supported complete ledger");
    common.batches[1]->productionReadback=receipts[1];
    edited=receipts;edited[1].samples.reserve(2*edited[1].samples.size()+1);
    rejects(edited,common.sealed,.25,"terminal actual source copy capacity beyond admission refuses atomically");
    for(unsigned mode=0;mode<5;++mode) {
        const auto budget=common.terminalBudget;
        if(mode==0)common.terminalBudget.retainedBytes=renderpassoutput::kMaximumTotalBytes;
        if(mode==1)common.terminalBudget.combinedWork=gc::kMaximumQueryWork;
        if(mode==2)common.terminalBudget.depositionWork=gc::kMaximumDepositionWork;
        if(mode==3)common.terminalBudget.retainedBytes=std::numeric_limits<std::uint64_t>::max();
        if(mode==4)common.terminalBudget={};
        rejects(receipts,common.sealed,.25,"terminal worst whole roster/orders/copies cannot exceed authenticated byte/work/deposition caps");
        common.terminalBudget=budget;
    }
    common.terminalBudgetSupported=false;rejects(receipts,common.sealed,.25,"terminal missing budget proof preserves the entire incumbent");
    common.terminalBudgetSupported=true;common.current=false;rejects(receipts,common.sealed,.25,"terminal lost current common frame refuses atomically");common.current=true;
    common.batches[1]->productionComplete=false;rejects(receipts,common.sealed,.25,"terminal incomplete source inspector refuses atomically");common.batches[1]->productionComplete=true;
    check(evaluate(receipts,common.sealed,.25),"terminal unchanged complete cohort revalidates after refused prefixes: "+error);
    if(retained)directGratingTerminalReceiverContract(common,ledger);
    if(retained)*retained=ledger;
}
void directGratingTerminalReceiverContract(PreparedDirectGratingCohort& common,const gc::DirectGratingTerminalOrderLedger& scattering)
{
    // Prepared semantic/budget fixtures only. All native witnesses and ledger
    // totals below are explicit mocks and establish no native receiver PASS.
    using Row=gc::DirectGratingTerminalReceiverRow;std::string error;
    gc::DirectGratingTerminalReceiverReadback result;result.bounds.bytes=123;
    const auto terminal=scattering.nativeResult;const auto before=scattering;
    const auto bind=[&](const auto& seal,double time) {
        return gc::bindDirectGratingTerminalReceivers(terminal,common,seal,time,result,error);};
    check(!bind(common.sealed,.25)&&result.bounds.bytes==123
        &&error=="Directional terminal native receiver lease is unsupported"
        &&common.terminalResult==terminal&&sameTerminalLedger(scattering,before),
        "unsupported additive receivers preserve valid scattering and prior receiver output");
    common.receiverSupported=true;
    common.receiverBudget={common.terminalBudget.retainedBytes,common.terminalBudget.combinedWork,
        sizeof(gc::NativeDirectGratingTerminalReceiverInput),8*sizeof(gc::Texel),256};
    check(bind(common.sealed,.25),"mock receiver lease stages exact emitted-order misses: "+error);
    if(!result.nativeResult)return;
    const auto accepted=result;
    check(result.terminal==terminal&&result.nativeResult==common.receiverResult
        &&result.rows.size()==scattering.orders.size()&&result.bounds.maximumRows==scattering.bounds.maximumOrders
        &&result.bounds.queryWork==result.bounds.maximumRows*result.bounds.perRayWork
        &&result.queryWork==result.rows.size()&&!result.nativeResult->retained().nativeResult
        &&result.nativeResult->retained().rows.data()!=result.rows.data()
        &&common.terminalResult==terminal&&sameTerminalLedger(scattering,before),
        "receiver query admission uses worst orders and independent sealed rows without replacing scattering");
    bool includesZero=false;
    for(std::size_t i=0;i<result.rows.size();++i) {
        const auto& row=result.rows[i];includesZero|=row.emitted.orderIrradiance==0;
        check(row.input.key[3]==i&&gc::sameDirectGratingTerminalEmittedOrder(row.emitted,scattering.orders[i])
            &&row.disposition==Row::Disposition::Miss&&row.rasterSample==Row::RasterSample::Absent,
            "every emitted signed order retains exact index and identity even when its native query misses");
    }
    check(includesZero,"emitted zero-irradiance orders remain in the receiver query set");
    for(unsigned fault=1;fault<=15;++fault) {
        const auto budget=common.receiverBudget;const auto priorOwner=common.receiverResult;
        common.receiverFault=fault;
        check(!bind(common.sealed,.25)&&!error.empty()
            &&gc::sameDirectGratingTerminalReceiverReadback(result,accepted)&&common.receiverResult==priorOwner
            &&common.terminalResult==terminal&&!common.activeReceiverLease&&!common.stagedReceiverResult,
            "receiver stale/tamper/dispatch/staging refusal preserves both previously published results");
        check(common.receiverBudget.retainedBytes==budget.retainedBytes
            &&common.receiverBudget.combinedWork==budget.combinedWork+(fault==7?0:accepted.bounds.queryWork),
            "failed admitted receiver attempts retain full worst work and release only staged bytes");
    }
    common.receiverFault=0;
    auto stale=common.sealed;stale.completedFence=0;
    const auto budget=common.receiverBudget;
    check(!bind(stale,.25)&&!bind(common.sealed,.5)
        &&gc::sameDirectGratingTerminalReceiverReadback(result,accepted)&&common.receiverBudget.combinedWork==budget.combinedWork,
        "wrong receiver fence/time refuses before admission");
    {
        PreparedDirectGratingCohort::ReceiverLease forged(common);
        check(!common.validateDirectTerminalReceiverResult(forged,accepted,error),
            "pointer association and an unregistered receiver lease cannot authenticate native output");
    }
    const auto& frame=common.cohort->sources()[0]->frame;
    gc::DirectGratingTerminalReceiverBounds bounds;
    check(gc::calculateDirectGratingTerminalReceiverBounds(*terminal,frame,common.receiverBudget,result,bounds,error),
        "later receiver admission reconstructs incumbent and failed-work charges: "+error);
    for(unsigned fault=0;fault<6;++fault) {
        auto invalid=common.receiverBudget;
        if(fault==0)invalid.retainedBytes=renderpassoutput::kMaximumTotalBytes;
        if(fault==1)invalid.combinedWork=gc::kMaximumQueryWork;
        if(fault==2)invalid.nativeInputBytesPerRow=UINT64_MAX;
        if(fault==3)invalid.nativeOutputBytesPerRow=UINT64_MAX;
        if(fault==4)invalid.nativeFixedBytes=UINT64_MAX;
        if(fault==5)invalid={};
        auto sentinel=bounds;
        check(!gc::calculateDirectGratingTerminalReceiverBounds(*terminal,frame,invalid,result,sentinel,error)
            &&sentinel.bytes==bounds.bytes&&sentinel.queryWork==bounds.queryWork,
            "receiver checked multiplication and whole-owner byte/work limits refuse without changing bounds");
    }
    auto callerOnly=result;callerOnly.nativeResult.reset();gc::DirectGratingTerminalReceiverBounds callerBounds;
    check(gc::calculateDirectGratingTerminalReceiverBounds(*terminal,frame,common.receiverBudget,callerOnly,callerBounds,error)
        &&bounds.bytes>callerBounds.bytes,"prior sealed receiver rows remain charged alongside caller rows");
    check(bind(common.sealed,.25)&&common.receiverBudget.combinedWork==budget.combinedWork+result.bounds.queryWork,
        "receiver retry succeeds after refused attempts without erasing prior admitted work");
    cameraCellCoverageContract(common);

    // Pure geometry validation of prepared records, not a native dispatch.
    // A camera below a downward-facing slanted receiver observes an off-center
    // emitted ray. Its raster center is on the same plane at a different depth.
    auto geometric=scattering;geometric.orders.resize(1);geometric.bounds.maximumOrders=1;
    geometric.sources.resize(1);geometric.incident.endpoints.resize(1);
    auto geometryFrame=frame;geometryFrame.width=geometryFrame.height=1;
    geometryFrame.camera.transform.translation={0,0,-4};geometryFrame.camera.transform.rotation={1,0,0,0};
    geometryFrame.camera.verticalFovRadians=float(3.14159265358979323846/2);
    geometryFrame.camera.nearPlane=.5f;geometryFrame.camera.farPlane=10;
    const auto triangle=[](std::array<gc::Point,3> points,std::uint32_t object,std::uint32_t material) {
        gc::Triangle value;value.points=points;value.object=object;value.material=material;
        const auto cross=gc::cross(gc::sub(points[1],points[0]),gc::sub(points[2],points[0]));
        value.area=.5f*std::sqrt(gc::dot(cross,cross));value.normal=cross;gc::normalize(value.normal);
        value.reflectance=surfacematerial::SpectralReflectanceDescription{};return value;
    };
    const auto endpointTriangle=triangle({gc::Point{-3,-3,-6},gc::Point{3,-3,-6},gc::Point{0,3,-6}},10,11);
    const auto receiverTriangle=triangle({gc::Point{-4,-4,-4},gc::Point{0,4,-2},gc::Point{4,-4,0}},20,21);
    geometryFrame.triangles={endpointTriangle,receiverTriangle};
    auto& endpoint=geometric.incident.endpoints[0].endpoint;
    endpoint.status=1;endpoint.distance=1;endpoint.position={.25f,0,-6};endpoint.outwardNormal=endpointTriangle.normal;
    endpoint.objectId=10;endpoint.materialId=11;endpoint.instanceId=endpoint.triangleId=0;endpoint.entering=true;
    auto& emitted=geometric.orders[0];emitted.source=emitted.endpoint=emitted.band=0;
    emitted.order.primary=emitted.order.secondary=0;emitted.order.direction={0,0,1};
    // The incoming segment and camera far plane must not cap the outgoing ray.
    auto& incoming=geometric.incident.endpoints[0].input;incoming.directionMaximum[3]=.001f;
    gc::NativeDirectGratingTerminalReceiverInput input;const bool inputReady=gc::makeDirectGratingTerminalReceiverInput(
        geometric,geometryFrame,0,input,error);
    check(inputReady&&input.directionMaximum[3]>6,"outgoing receiver limit derives all frozen triangle vertices, independent of incoming range");
    if(!inputReady)return;
    auto shortCamera=geometryFrame;shortCamera.camera.farPlane=1;gc::NativeDirectGratingTerminalReceiverInput shortInput;
    check(gc::makeDirectGratingTerminalReceiverInput(geometric,shortCamera,0,shortInput,error)
        &&gc::sameDirectGratingTerminalReceiverInput(input,shortInput),"camera far depth cannot narrow the outgoing scene query");

    // One float step behind the emitting plane is an accepted native hit
    // residual. At grazing incidence the old directional offset rounds away.
    auto grazing=geometric;
    auto& grazingEndpoint=grazing.incident.endpoints[0].endpoint;
    grazingEndpoint.position[2]=std::nextafter(endpoint.position[2],-std::numeric_limits<float>::infinity());
    grazing.orders[0].order.direction={std::sqrt(1.0-4e-12),0,2e-6};
    gc::Point grazingDirection{};
    for(unsigned a=0;a<3;++a)grazingDirection[a]=float(grazing.orders[0].order.direction[a]);
    const auto oldOrigin=gc::advance(grazingEndpoint.position,grazingDirection,2*gc::bias(grazingEndpoint.position));
    const float oldPlaneDistance=gc::dot(gc::sub(endpointTriangle.points[0],oldOrigin),endpointTriangle.normal)
        /gc::dot(grazingDirection,endpointTriangle.normal);
    auto oldSelfHit=grazingEndpoint;oldSelfHit.distance=oldPlaneDistance;
    oldSelfHit.position=gc::advance(oldOrigin,grazingDirection,oldPlaneDistance);oldSelfHit.entering=false;
    check(gc::hitTriangle(geometryFrame,grazingEndpoint)
        &&oldOrigin[2]==grazingEndpoint.position[2]&&oldOrigin[2]<endpointTriangle.points[0][2]
        &&oldPlaneDistance>gc::kRayMinimum&&gc::hitTriangle(geometryFrame,oldSelfHit),
        "old grazing direction offset loses normal clearance and crosses the emitting triangle forward");
    gc::NativeDirectGratingTerminalReceiverInput grazingInput;
    const bool grazingReady=gc::makeDirectGratingTerminalReceiverInput(grazing,geometryFrame,0,grazingInput,error);
    const auto normalOrigin=gc::advance(grazingEndpoint.position,grazingEndpoint.outwardNormal,gc::bias(grazingEndpoint.position));
    check(grazingReady&&grazingInput.originMinimum==gc::Texel{normalOrigin[0],normalOrigin[1],normalOrigin[2],gc::kRayMinimum}
        &&normalOrigin[2]>endpointTriangle.points[0][2]
        &&gc::dot(gc::sub(endpointTriangle.points[0],normalOrigin),endpointTriangle.normal)
            /gc::dot(grazingDirection,endpointTriangle.normal)<0
        &&grazingInput.directionMaximum[0]==grazingDirection[0]
        &&grazingInput.directionMaximum[1]==grazingDirection[1]
        &&grazingInput.directionMaximum[2]==grazingDirection[2]
        &&grazingInput.key==input.key&&grazingInput.order==input.order
        &&oldPlaneDistance<grazingInput.directionMaximum[3],
        "grazing receiver launch matches native outward-normal bias with representable clearance and exact order identity");
    grazing.incident.endpoints[0].input.directionMaximum[3]=1000;
    gc::NativeDirectGratingTerminalReceiverInput grazingShortCamera;
    check(grazingReady&&gc::makeDirectGratingTerminalReceiverInput(grazing,shortCamera,0,grazingShortCamera,error)
        &&gc::sameDirectGratingTerminalReceiverInput(grazingInput,grazingShortCamera),
        "grazing outgoing extent remains independent of incoming segment and camera far plane");
    gc::DirectGratingTerminalReceiverReadback prepared;prepared.bounds.maximumRows=1;
    gc::directGratingTerminalReceiverAllowance(geometryFrame,prepared.bounds.perRayWork);
    prepared.bounds.queryWork=prepared.bounds.perRayWork;prepared.queryWork=1;
    Row row;row.input=input;row.emitted=emitted;row.hit.status=1;row.hit.position={.25f,0,-1.875f};
    row.hit.distance=row.hit.position[2]-input.originMinimum[2];row.hit.outwardNormal=receiverTriangle.normal;
    row.hit.objectId=20;row.hit.materialId=21;row.hit.entering=true;row.hit.work=1;row.queryWork=1;
    row.disposition=Row::Disposition::Pixel;row.rasterSample=Row::RasterSample::Surface;row.raster={20,21,0,0,2};
    check(gc::projectDirectGratingTerminalReceiver(geometryFrame,row.hit,row.pixel,error),"prepared slanted receiver point projects: "+error);
    prepared.rows.push_back(row);
    check(row.pixel.projected&&std::abs(row.pixel.linearDepth-row.raster.linearDepth)>.1f
        &&gc::validateDirectGratingTerminalReceiverRows(geometric,geometryFrame,prepared,error),
        "order-hit depth and native raster-center plane depth differ truthfully on a slanted receiver: "+error);
    auto tampered=prepared;tampered.rows[0].raster.linearDepth=row.pixel.linearDepth;
    check(!gc::validateDirectGratingTerminalReceiverRows(geometric,geometryFrame,tampered,error),
        "copying order-hit depth into raster-center depth refuses against the frozen plane");
    auto occluded=prepared;const auto blocker=triangle({gc::Point{-4,-4,-3},gc::Point{0,4,-3},gc::Point{4,-4,-3}},30,31);
    geometryFrame.triangles.push_back(blocker);occluded.rows[0].raster={30,31,0,0,1};
    check(gc::validateDirectGratingTerminalReceiverRows(geometric,geometryFrame,occluded,error),
        "a different actual raster owner records occlusion without inventing receiver visibility");
    auto background=prepared;background.rows[0].rasterSample=Row::RasterSample::Background;background.rows[0].raster={};
    check(gc::validateDirectGratingTerminalReceiverRows(geometric,geometryFrame,background,error),
        "projected order hit and native raster background remain separate truthful records");
    geometryFrame.triangles.pop_back();
    auto offscreen=prepared;offscreen.rows[0].disposition=Row::Disposition::Offscreen;
    gc::projectDirectGratingTerminalReceiver(shortCamera,row.hit,offscreen.rows[0].pixel,error);
    offscreen.rows[0].rasterSample=Row::RasterSample::Absent;offscreen.rows[0].raster={};
    check(gc::validateDirectGratingTerminalReceiverRows(geometric,shortCamera,offscreen,error),
        "a supported offscreen closest hit retains attribution without raster or visibility claims");
    auto nonreceiver=prepared;geometryFrame.triangles[1].reflectance.reset();
    nonreceiver.rows[0].disposition=Row::Disposition::NonReceiver;nonreceiver.rows[0].pixel={};
    nonreceiver.rows[0].rasterSample=Row::RasterSample::Absent;nonreceiver.rows[0].raster={};
    check(gc::validateDirectGratingTerminalReceiverRows(geometric,geometryFrame,nonreceiver,error),
        "a non-diffuse or grating native hit remains exact attribution without a diffuse receiver grant");
}
void cameraCellCoverageContract(PreparedDirectGratingCohort& original)
{
    // Separate mock frames keep each fault's admitted work journal permanent.
    // Empty-pixel lifecycle successes below establish no positive native area.
    std::string error;
    const auto prepare=[&](PreparedDirectGratingCohort& frame,gc::DirectGratingTerminalReceiverReadback& parent) {
        const auto source=original.cohort->sources()[0];
        if(!gc::prepareDirectGratingIrradianceCohort(original.cohort->sources(),{original.cohort->inputs()[0]},0,0,frame.cohort,error))return false;
        frame.widthValue=source->frame.width;frame.heightValue=source->frame.height;
        frame.sealed={frame.cohort->identity(),1,20,3,4,20,20};
        for(std::size_t s=0;s<original.batches.size();++s) {
            auto batch=std::make_shared<PreparedDirectGratingFrame>(*original.batches[s]);batch->binding=frame.cohort->sources()[s];
            batch->endpoints.inputs.resize(1);batch->productionReadback.samples.resize(1);
            batch->productionReadback.queryWork=2;batch->productionReadback.bandWork=8;
            frame.batches.push_back(batch);frame.currentTerminalSources.push_back(batch->productionReadback);
        }
        frame.terminalBudgetSupported=true;
        frame.terminalBudget={frame.cohort->bounds().bytes+1024,frame.cohort->bounds().combinedWork,frame.cohort->bounds().depositionWork};
        gc::DirectGratingTerminalOrderLedger terminal;
        if(!gc::evaluateDirectGratingTerminalOrders(frame.cohort,frame,frame.sealed,.25,frame.currentTerminalSources,terminal,error))return false;
        frame.receiverSupported=true;frame.receiverBudget={frame.terminalBudget.retainedBytes,frame.terminalBudget.combinedWork,
            sizeof(gc::NativeDirectGratingTerminalReceiverInput),8*sizeof(gc::Texel),256};
        if(!gc::bindDirectGratingTerminalReceivers(terminal.nativeResult,frame,frame.sealed,.25,parent,error))return false;
        frame.coverageBudget={frame.receiverBudget.retainedBytes,frame.receiverBudget.combinedWork,16,16,32,16,512};return true;
    };
    PreparedDirectGratingCohort frame;gc::DirectGratingTerminalReceiverReadback parent;
    check(prepare(frame,parent),"small complete mock cohort prepares separate coverage lifecycle: "+error);if(!parent.nativeResult)return;
    gc::CameraCellCoverageReadback result;result.bounds.bytes=123;
    check(!gc::proveCameraCellCoverage(parent.nativeResult,frame,frame.sealed,.25,result,error)&&result.bounds.bytes==123
        &&frame.receiverResult==parent.nativeResult&&error=="Native camera-cell coverage is unsupported",
        "default Unsupported preserves parent receiver and incumbent coverage");
    PreparedSourceFrame oldNativeConsumer;
    check(!gc::proveCameraCellCoverage(parent.nativeResult,oldNativeConsumer,frame.sealed,.25,result,error)
        &&result.bounds.bytes==123&&error=="Native camera-cell coverage is unsupported",
        "existing native frame without the separate coverage trait cannot authenticate a proof");
    frame.coverageSupported=true;
    check(gc::proveCameraCellCoverage(parent.nativeResult,frame,frame.sealed,.25,result,error),
        "mock empty-domain coverage lifecycle publishes no finite area: "+error);if(!result.nativeResult)return;
    const auto accepted=result;const auto baseline=frame.coverageBudget;
    check(result.preparation.pixels.empty()&&result.proof.regions.empty()&&result.proof.vertices.empty()
        &&result.nativeResult==frame.coverageResult&&!result.nativeResult->retained().nativeResult
        &&frame.receiverResult==parent.nativeResult,
        "independent immutable coverage lifecycle leaves all parent receiver rows/world areas unchanged");
    for(unsigned fault=1;fault<=11;++fault) {
        PreparedDirectGratingCohort trial;gc::DirectGratingTerminalReceiverReadback trialParent;
        check(prepare(trial,trialParent),"fresh fault fixture prepares its own retained journal: "+error);
        if(!trialParent.nativeResult){continue;}trial.coverageSupported=true;
        // Keep the first frame's immutable incumbent live while this distinct
        // native frame admits its first attempt. No work journal is reset.
        auto prior=accepted;const auto priorOwner=trial.coverageResult;
        const auto budget=trial.coverageBudget;const auto old=prior;trial.coverageFault=fault;
        check(!gc::proveCameraCellCoverage(trialParent.nativeResult,trial,trial.sealed,.25,prior,error)&&!error.empty()
            &&gc::sameCameraCellCoverageReadback(prior,old)&&trial.coverageResult==priorOwner
            &&trial.receiverResult==trialParent.nativeResult&&!trial.activeCoverageLease&&!trial.stagedCoverageResult,
            "unsupported seam/gap/occlusion/depth/arithmetic and stage/generation failures preserve both old results");
        check(trial.coverageBudget.retainedBytes==budget.retainedBytes
            &&trial.coverageBudget.combinedWork==budget.combinedWork+(fault==1?0:old.bounds.operationWork),
            "failed full arrangement admission stays permanently charged in the owning mock frame");
        if(fault!=1) {
            gc::CameraCellCoverageBounds later;
            const bool admitted=gc::calculateCameraCellCoverageBounds(*trialParent.nativeResult,
                trial.cohort->sources()[0]->frame,trial.coverageBudget,prior,later,error);
            check(!admitted||later.combinedWork==trial.coverageBudget.combinedWork+old.bounds.operationWork,
                "later admission preserves failed work and either fits honestly or refuses exhausted remaining budget");
        }
    }
    auto stale=frame.sealed;stale.completedFence=0;
    check(!gc::proveCameraCellCoverage(parent.nativeResult,frame,stale,.25,result,error)
        &&!gc::proveCameraCellCoverage(parent.nativeResult,frame,frame.sealed,.5,result,error)
        &&frame.coverageBudget.combinedWork==baseline.combinedWork&&gc::sameCameraCellCoverageReadback(result,accepted),
        "coverage stale submission/time refuses before work admission");
    {
        PreparedDirectGratingCohort::CoverageLease forged(frame);
        check(!frame.validateCameraCellCoverageProof(forged,accepted,error),"coverage pointer and unregistered lease grant no proof");
    }
    const auto& frozen=frame.cohort->sources()[0]->frame;gc::CameraCellCoverageBounds bounds;
    // A separate candidate-only budget, before any geometry/association work.
    auto candidateBudget=frame.coverageBudget;candidateBudget.combinedWork=parent.bounds.combinedWork;
    check(gc::calculateCameraCellCoverageBounds(*parent.nativeResult,frozen,candidateBudget,{},bounds,error),
        "camera arrangement derives conservative depth pairs, regions and shared vertices: "+error);
    const auto triangles=std::uint64_t(frozen.triangles.size());
    check(bounds.depthPairs==triangles*(triangles-1)/2&&bounds.linesPerPixel==6*triangles+bounds.depthPairs
        &&bounds.regionsPerPixel==1+bounds.linesPerPixel*(bounds.linesPerPixel+1)/2
        &&bounds.verticesPerPixel==2*(bounds.linesPerPixel+4)*(bounds.linesPerPixel+1)
        &&bounds.maximumPixels==0&&bounds.maximumPixels<=parent.bounds.maximumRows
        &&bounds.maximumCandidates==bounds.maximumPixels*triangles,
        "camera bounds derive from every frozen triangle and pair rather than source-cell leaf limits");
    for(unsigned mode=0;mode<2;++mode) {
        auto invalid=candidateBudget;
        if(mode==0)invalid.retainedBytes=renderpassoutput::kMaximumTotalBytes;
        if(mode==1)invalid.combinedWork=gc::kMaximumQueryWork;
        auto sentinel=bounds;
        check(!gc::calculateCameraCellCoverageBounds(*parent.nativeResult,frozen,invalid,{},sentinel,error)
            &&gc::sameCameraCellCoverageBounds(sentinel,bounds),"camera arrangement byte/work overflow refuses before allocation");
    }
    gc::CameraCellCoverageBounds withOld,withoutOld;auto callerOnly=accepted;callerOnly.nativeResult.reset();
    check(gc::calculateCameraCellCoverageBounds(*parent.nativeResult,frozen,candidateBudget,accepted,withOld,error)
        &&gc::calculateCameraCellCoverageBounds(*parent.nativeResult,frozen,candidateBudget,callerOnly,withoutOld,error)
        &&withOld.bytes>withoutOld.bytes,"old immutable coverage owner stays charged beside caller storage");

    // Geometry-only synthetic parent association, never a published native hit
    // or positive coverage instance. Use a nominated visible receiver centroid.
    auto preparedParent=parent;const auto receiver=std::find_if(frozen.triangles.begin(),frozen.triangles.end(),[](const auto& t) {
        return !t.glass&&bool(t.reflectance);});
    if(receiver==frozen.triangles.end()||preparedParent.rows.size()<2)return;
    gc::Point position{};for(const auto& p:receiver->points)position=gc::advance(position,p,1.0f/3);
    arbitgpu::opaquevisibility::ClosestHit hit;
    hit.status=1;hit.position=position;hit.outwardNormal=receiver->normal;hit.objectId=receiver->object;
    hit.materialId=receiver->material;hit.instanceId=receiver->instance;hit.triangleId=receiver->triangle;hit.entering=true;hit.work=1;
    hit.distance=1;gc::PixelFootprint pixel;
    check(gc::projectDirectGratingTerminalReceiver(frozen,hit,pixel,error)&&pixel.projected,
        "candidate-only nominated receiver centroid projects: "+error);if(!pixel.projected)return;
    for(unsigned i=0;i<2;++i) {
        auto& row=preparedParent.rows[i];row.hit=hit;row.pixel=pixel;row.disposition=gc::DirectGratingTerminalReceiverRow::Disposition::Pixel;
    }
    const auto geometryParent=frame.preparedCoverageParent(preparedParent);gc::CameraCellCoveragePreparation preparation;
    check(gc::calculateCameraCellCoverageBounds(*geometryParent,frozen,candidateBudget,{},bounds,error)
        &&bounds.maximumPixels==2&&bounds.maximumPixels<=parent.bounds.maximumRows,
        "only immutable Pixel-row cardinality bounds unique inputs before their dedup/geometry allocation");
    for(unsigned mode=0;mode<4;++mode) {
        auto invalid=candidateBudget;
        if(mode==0)invalid.pixelBytes=UINT64_MAX;
        if(mode==1)invalid.regionBytes=UINT64_MAX;
        if(mode==2)invalid.vertexBytes=UINT64_MAX;
        if(mode==3)invalid.fixedBytes=UINT64_MAX;
        auto sentinel=bounds;
        check(!gc::calculateCameraCellCoverageBounds(*geometryParent,frozen,invalid,{},sentinel,error)
            &&gc::sameCameraCellCoverageBounds(sentinel,bounds),"nonempty camera native stride multiplication refuses overflow before work");
    }
    auto dense=frozen;dense.triangles.resize(512,frozen.triangles[0]);auto sentinel=bounds;
    check(!gc::calculateCameraCellCoverageBounds(*geometryParent,dense,candidateBudget,{},sentinel,error)
        &&gc::sameCameraCellCoverageBounds(sentinel,bounds),"quadratic depth-boundary arrangement growth refuses conservatively before its work");
    check(gc::prepareCameraCellCoverage(*geometryParent,frozen,bounds,preparation,error)&&preparation.pixels.size()==1
        &&preparation.associations.size()==2&&preparation.associations[0].parentRow==0&&preparation.associations[1].parentRow==1
        &&gc::sameDirectGratingTerminalReceiverInput(preparation.associations[0].order,preparedParent.rows[0].input)
        &&preparation.candidates.size()==frozen.triangles.size()&&preparedParent.rows[0].pixel.worldArea==0,
        "same camera pixel deduplicates while retaining every exact parent occurrence and all-triangle candidates");
    auto conflicting=preparedParent;conflicting.rows[1].hit.objectId+=1;
    const auto conflictParent=frame.preparedCoverageParent(conflicting);auto unchanged=preparation;
    check(!gc::prepareCameraCellCoverage(*conflictParent,frozen,bounds,unchanged,error)
        &&unchanged.associations.size()==preparation.associations.size(),"conflicting receiver ownership refuses without dropping an associated row");
    gc::CameraCellCoverageReadback shape=accepted;shape.preparation=preparation;shape.bounds=bounds;
    shape.proof.work=1;shape.proof.regions={{0,std::uint32_t(receiver-frozen.triangles.begin()),0,4,1}};
    shape.proof.vertices={{{0,0}},{{1,0}},{{1,1}},{{0,1}}};
    check(gc::validateCameraCellCoverageShape(frozen,shape,error),"decoded full square is shape-valid but grants no native union/visibility");
    auto overlap=shape;overlap.proof.regions={{0,std::uint32_t(receiver-frozen.triangles.begin()),0,4,.5},
        {0,std::uint32_t(receiver-frozen.triangles.begin()),4,4,.5}};
    overlap.proof.vertices={{{0,0}},{{.5,0}},{{.5,1}},{{0,1}},{{0,0}},{{.5,0}},{{.5,1}},{{0,1}}};
    check(gc::validateCameraCellCoverageShape(frozen,overlap,error),
        "decoded area sum can hide an overlap and gap, so shared shape checks are deliberately not proof");
    {
        PreparedDirectGratingCohort::CoverageLease forged(frame);
        check(!frame.validateCameraCellCoverageProof(forged,shape,error)&&!frame.validateCameraCellCoverageProof(forged,overlap,error),
            "well-shaped candidate or area sum never bypasses the registered native conservative predicate");
    }
}
void directGratingContributionContract()
{
    // Synthetic native ownership and readback shape only. No actual production
    // query, source visibility, native arithmetic or pixels are established.
    auto fixture=gt::ordinaryFixture();
    const auto& target=fixture.scene->objects[1];
    for(std::size_t v=0;v<target.vertexCount;++v) {
        auto& vertex=fixture.scene->vertices[target.firstVertex+v];
        vertex.uv={(vertex.position.x+6)/12,(vertex.position.y+6)/12};
    }
    auto& d=fixture.scene->lights[0].spectralSource.directionalIrradiance;
    d.version=1;d.measure=diffractionmaterial::DirectionalSpectralMeasure::IrradianceCoefficient;
    d.coefficients={2,1e-25f,0,5,6,7,8,9};
    std::string error;check(fixture.prepare(error),"direct contribution D preparation: "+error);
    auto description=diffractionmaterial::makeAluminiumBinaryGratingPreset();
    description.spectrum.lastOrder=1;description.roughness.rmsSlope=0;description.microstructure.grooveDepthNanometres=.9f;
    const auto admitted=diffractionmaterial::admit(description,error);
    check(bool(admitted),"direct contribution admitted grating material: "+error);if(!admitted)return;
    auto material=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*admitted);
    auto program=std::make_shared<gt::Program>(*fixture.program->objectPrograms[0]);
    program->kind=arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;program->spectralReflectance.reset();
    program->diffractionPathCount=1;program->bindingDigest="direct-grating-material";program->programIdentity="direct-grating-program";
    program->diffractionPaths[0].material=diffractionmaterial::physicalcheckpoint::makeGpuParameters(*material,{});
    PreparedDirectGratingFrame native;
    check(gc::prepareGratingIncidentBinding(fixture.prepared,fixture.frame,program,material,.25,native.binding,error,
        gc::SourceMeasureUse::DirectionalIrradiancePreparation),"direct contribution retains current grating binding: "+error);
    if(!native.binding)return;
    const auto bindingOwner=native.binding;
    const auto& binding=*bindingOwner;const auto& triangle=binding.triangles.front();
    native.widthValue=binding.frame.width;native.heightValue=binding.frame.height;
    native.uploaded={program->backend,{binding.identity,1,2,3,4,5,2},11,
        gc::directionalSourceInput(binding.prepared,binding.frame),true};
    gc::NativeDirectGratingEndpointInput input;
    input.primitive={triangle.object,triangle.material,triangle.instance,triangle.triangle};
    const gc::Point barycentric{.5f,.025f,.475f};gc::Point position{};
    for(unsigned v=0;v<3;++v)position=gc::advance(position,triangle.positions[v],barycentric[v]);
    const auto origin=gc::advance(position,triangle.normal,1);
    input.originMinimum={origin[0],origin[1],origin[2],gc::kRayMinimum};
    input.directionMaximum={-triangle.normal[0],-triangle.normal[1],-triangle.normal[2],binding.frame.sourceDirectionTrace[3]};
    native.endpoints.submission={binding.identity,1,3,3,4,6,3};native.endpoints.generation=13;
    native.endpoints.inputs={input};native.endpoints.complete=true;
    gc::NativeDirectGratingContributionReadback result;
    result.version=gc::kDirectGratingContributionVersion;result.submission={binding.identity,1,7,3,4,8,7};
    result.endpointGeneration=13;result.sourceGeneration=11;result.complete=true;result.queryWork=2;result.bandWork=8;
    native.productionSubmission=result.submission;
    gc::NativeDirectGratingContributionSample sample;sample.input=input;
    sample.source.input=native.uploaded.input;sample.source.directionTrace=binding.frame.sourceDirectionTrace;
    for(unsigned b=0;b<8;++b)sample.source.rawRadiance[b]=binding.frame.sourceRadiance[b/4][b%4];
    auto& hit=sample.endpoint;hit.status=1;hit.distance=1;hit.position=position;hit.outwardNormal=triangle.normal;
    hit.objectId=triangle.object;hit.materialId=triangle.material;hit.instanceId=triangle.instance;hit.triangleId=triangle.triangle;
    hit.entering=true;hit.work=1;sample.barycentric=barycentric;
    sample.sourceOriginMinimum={position[0],position[1],position[2],gc::kRayMinimum};
    sample.sourceDirectionMaximum=binding.frame.sourceDirectionTrace;
    sample.sourceHit.status=0;sample.sourceHit.distance=binding.frame.sourceDirectionTrace[3];sample.sourceHit.work=1;
    const gc::Point toSource{sample.source.directionTrace[0],sample.source.directionTrace[1],sample.source.directionTrace[2]};
    const float cosine=std::max(0.0f,gc::dot(triangle.normal,toSource));
    for(unsigned b=0;b<8;++b)sample.surfaceIrradiance[b]=d.coefficients[b]*cosine;
    sample.outcome=gc::NativeDirectGratingContributionSample::Outcome::Positive;sample.queryWork=2;sample.complete=true;
    result.samples={sample};
    native.productionReadback=result;
    const auto validate=[&](const auto& value) {
        return gc::validateDirectGratingContributionReadback(binding,native,result.submission,.25,value,error);};
    check(cosine>0&&validate(result),"exact synthetic native endpoint contribution equals visible D times cosine: "+error);
    const auto positiveCopy=result;
    check(validate(positiveCopy),"unchanged copy retains exact synthetic positive result association");
    check(sample.surfaceIrradiance[1]>0&&sample.surfaceIrradiance[2]==0,"tiny positive and authored zero D bands retain distinct values");
    for(int mode=0;mode<35;++mode) {
        auto bad=result;auto& s=bad.samples[0];
        if(mode==0)bad.version=0;
        if(mode==1)bad.complete=false;
        if(mode==2)bad.overflow=true;
        if(mode==3)++bad.endpointGeneration;
        if(mode==4)++bad.sourceGeneration;
        if(mode==5)++bad.submission.queryGeneration;
        if(mode==6)bad.submission.completedFence=0;
        if(mode==7)++bad.queryWork;
        if(mode==8)--bad.bandWork;
        if(mode==9)s.complete=false;
        if(mode==10)s.input.primitive[0]++;
        if(mode==11)s.input.originMinimum[0]+=.1f;
        if(mode==12)s.source.input.identifiers[1]=0;
        if(mode==13)s.source.input.coefficients[0]+=1;
        if(mode==14)s.source.directionTrace[3]+=1;
        if(mode==15)s.source.rawRadiance[0]+=1;
        if(mode==16)s.endpoint.status=0;
        if(mode==17)s.endpoint.objectId++;
        if(mode==18)s.endpoint.position[0]+=.1f;
        if(mode==19)s.endpoint.outwardNormal[0]=.2f;
        if(mode==20)s.barycentric={0,.5f,.5f};
        if(mode==21)s.sourceOriginMinimum[0]+=.1f;
        if(mode==22)s.sourceDirectionMaximum[3]-=1;
        if(mode==23)s.sourceHit.status=-2;
        if(mode==24)s.sourceHit.work=0;
        if(mode==25)s.surfaceIrradiance[0]=-1;
        if(mode==26)s.surfaceIrradiance[0]=std::numeric_limits<float>::quiet_NaN();
        if(mode==27)s.surfaceIrradiance[1]=0;
        if(mode==28)s.outcome=gc::NativeDirectGratingContributionSample::Outcome::Unsupported;
        if(mode==29)bad.submission.frameIdentity+="stale";
        if(mode==30)++bad.submission.rendererGeneration;
        if(mode==31)++bad.submission.rasterGeneration;
        if(mode==32)s.surfaceIrradiance[0]=std::numeric_limits<float>::infinity();
        if(mode==33)s.input.directionMaximum[0]=std::numeric_limits<float>::quiet_NaN();
        if(mode==34)s.input.originMinimum[3]=0;
        check(!validate(bad),"stale, incomplete, unassociated or malformed direct contribution refuses");
    }
    const auto uploaded=native.endpoints;
    native.endpoints.complete=false;check(!validate(result),"missing actual native endpoint input upload refuses");
    native.endpoints=uploaded;native.endpoints.inputs[0].directionMaximum[3]-=1;
    check(!validate(result),"changed native endpoint input buffer refuses");native.endpoints=uploaded;
    native.endpoints.submission.queryGeneration++;
    check(!validate(result),"stale native endpoint query generation refuses");native.endpoints=uploaded;
    native.endpoints.submission.completedFence=0;
    check(!validate(result),"native endpoint upload must have its own completed fence");native.endpoints=uploaded;
    const auto sourceUpload=native.uploaded;
    native.uploaded.input.identifiers[0]=0;check(!validate(result),"legacy source cannot be relabeled as D");native.uploaded=sourceUpload;
    native.uploaded.complete=false;check(!validate(result),"incomplete native D upload refuses");native.uploaded=sourceUpload;
    native.uploaded.submission.rendererGeneration++;
    check(!validate(result),"stale source upload frame generation refuses");native.uploaded=sourceUpload;
    native.productionComplete=false;check(!validate(result),"absent completed endpoint-to-source production query refuses");native.productionComplete=true;
    native.contextCurrent=false;check(!validate(result),"stale production query context refuses");native.contextCurrent=true;
    native.cache=0;check(!validate(result),"lost native frame lifetime refuses");native.cache=1;
    check(!gc::validateDirectGratingContributionReadback(binding,native,result.submission,.5,result,error),"changed evaluation time refuses");
    auto detached=std::make_shared<const gc::GratingIncidentBinding>(binding);native.binding=detached;
    check(!gc::validateDirectGratingContributionReadback(binding,native,result.submission,.25,result,error),
        "byte-identical replacement grating owner cannot replace retained endpoint association");
    native.binding=bindingOwner;
    // A real native blocker makes this direct contribution zero, regardless of
    // its transmission. Post-glass transport is a different input measure.
    auto blocked=result;auto& zero=blocked.samples[0];const auto& blocker=binding.frame.triangles.front();
    zero.sourceHit.status=1;zero.sourceHit.distance=(blocker.points[0][2]-position[2])/toSource[2];
    zero.sourceHit.position=gc::advance(position,toSource,zero.sourceHit.distance);zero.sourceHit.outwardNormal=blocker.normal;
    zero.sourceHit.objectId=blocker.object;zero.sourceHit.materialId=blocker.material;
    zero.sourceHit.instanceId=blocker.instance;zero.sourceHit.triangleId=blocker.triangle;
    zero.sourceHit.entering=gc::dot(toSource,blocker.normal)<0;
    zero.surfaceIrradiance.fill(0);zero.outcome=gc::NativeDirectGratingContributionSample::Outcome::Zero;
    native.productionReadback=blocked;
    check(validate(blocked),"completed synthetic blocked contribution is supported zero: "+error);
    const auto zeroCopy=blocked;
    check(validate(zeroCopy),"unchanged copy retains exact synthetic zero result association");
    auto forged=blocked;
    forged.samples[0].sourceHit=sample.sourceHit;
    forged.samples[0].surfaceIrradiance=sample.surfaceIrradiance;
    forged.samples[0].outcome=sample.outcome;
    check(gc::sameDirectGratingContributionReadback(forged,positiveCopy),
        "blocked-to-positive edit is otherwise the exact previously accepted positive batch");
    check(!validate(forged),"semantically valid blocked-to-positive edit cannot replace actual retained native result");
    // Every mutation below compares against an independent retained zero batch.
    // The helper checks decoded field values, never struct padding or tolerance.
    for(int mode=0;mode<40;++mode) {
        auto supplied=blocked;auto& s=supplied.samples[0];
        if(mode==0)++supplied.version;
        if(mode==1)++supplied.endpointGeneration;
        if(mode==2)++supplied.sourceGeneration;
        if(mode==3)++supplied.queryWork;
        if(mode==4)++supplied.bandWork;
        if(mode==5)supplied.complete=false;
        if(mode==6)supplied.overflow=true;
        if(mode==7)++supplied.submission.scratchGeneration;
        if(mode==8)supplied.samples.push_back(blocked.samples[0]);
        if(mode==9)++s.input.primitive[0];
        if(mode==10)++s.input.control[0];
        if(mode==11)s.input.originMinimum[0]+=1;
        if(mode==12)s.input.directionMaximum[3]+=1;
        if(mode==13)++s.source.input.identifiers[3];
        if(mode==14)s.source.input.coefficients[0]+=1;
        if(mode==15)s.source.rawRadiance[0]+=1;
        if(mode==16)s.source.directionTrace[3]+=1;
        if(mode==17)++s.endpoint.status;
        if(mode==18)s.endpoint.distance+=1;
        if(mode==19)s.endpoint.position[0]+=1;
        if(mode==20)s.endpoint.outwardNormal[0]+=1;
        if(mode==21)++s.endpoint.objectId;
        if(mode==22)++s.endpoint.materialId;
        if(mode==23)++s.endpoint.instanceId;
        if(mode==24)++s.endpoint.triangleId;
        if(mode==25)s.endpoint.entering=!s.endpoint.entering;
        if(mode==26)++s.endpoint.work;
        if(mode==27)++s.sourceHit.status;
        if(mode==28)s.sourceHit.distance+=1;
        if(mode==29)s.sourceHit.position[0]+=1;
        if(mode==30)s.sourceHit.outwardNormal[0]+=1;
        if(mode==31)++s.sourceHit.objectId;
        if(mode==32)++s.sourceHit.work;
        if(mode==33)s.barycentric[0]+=.1f;
        if(mode==34)s.sourceOriginMinimum[0]+=1;
        if(mode==35)s.sourceDirectionMaximum[3]+=1;
        if(mode==36)s.surfaceIrradiance[0]=-0.0f;
        if(mode==37)++s.queryWork;
        if(mode==38)s.outcome=gc::NativeDirectGratingContributionSample::Outcome::Positive;
        if(mode==39)s.complete=false;
        check(!gc::sameDirectGratingContributionReadback(supplied,native.productionReadback),
            "changed decoded batch/sample fields cannot match actual retained result");
        check(!validate(supplied),"supplied edited result refuses native completion");
    }
    zero.sourceHit.status=-1;check(!validate(blocked),"failed source query cannot become supported zero");
    gc::DirectGratingContributionBounds bounds;check(gc::calculateDirectGratingContributionBounds(binding,1,0,0,bounds,error)
        &&bounds.bandWork==8&&bounds.queryWork>0&&bounds.combinedWork<=gc::kMaximumQueryWork,
        "direct native endpoint/source legs and bands share original frame caps");
    bounds.bytes=123;check(!gc::calculateDirectGratingContributionBounds(binding,gc::kMaximumEntrySamples+1,0,0,bounds,error)
        &&bounds.bytes==123,"endpoint cap refusal preserves bounds output");
    auto exhausted=binding;exhausted.bounds.bytes=renderpassoutput::kMaximumTotalBytes-exhausted.frame.bounds.transientBytes;
    check(!gc::calculateDirectGratingContributionBounds(exhausted,1,0,0,bounds,error)&&bounds.bytes==123,"whole-frame storage exhaustion refuses before allocation");
    exhausted=binding;exhausted.bounds.combinedWork=gc::kMaximumQueryWork;
    check(!gc::calculateDirectGratingContributionBounds(exhausted,1,0,0,bounds,error),"changed or exhausted whole-frame work refuses");
    check(!gc::calculateDirectGratingContributionBounds(binding,1,renderpassoutput::kMaximumTotalBytes,0,bounds,error)
        &&!gc::calculateDirectGratingContributionBounds(binding,1,0,gc::kMaximumQueryWork,bounds,error),
        "all other live adjunct storage and work must fit the same frame caps");

    fixture.scene->lightCount=2;fixture.scene->lights[1]=fixture.scene->lights[0];
    fixture.scene->lights[1].id.value+=1;fixture.scene->lights[1].transform.translation.x=3;
    fixture.scene->lights[1].transform.rotation={0,0,0,1};
    fixture.scene->lights[1].spectralSource.directionalIrradiance.coefficients={4,2e-25f,0,1,2,3,4,5};
    check(fixture.prepare(error),"cohort full authored D8 scene prepares: "+error);
    auto secondFixture=fixture;secondFixture.ordinaryLight=fixture.scene->lights[1].id;
    check(secondFixture.prepare(error),"cohort second admitted source prepares: "+error);
    std::vector<std::shared_ptr<const gc::GratingIncidentBinding>> sourceBindings(2);
    check(gc::prepareGratingIncidentBinding(fixture.prepared,fixture.frame,program,material,.25,sourceBindings[0],error,
        gc::SourceMeasureUse::DirectionalIrradiancePreparation),"cohort first binding prepares: "+error);
    check(gc::prepareGratingIncidentBinding(secondFixture.prepared,secondFixture.frame,program,material,.25,sourceBindings[1],error,
        gc::SourceMeasureUse::DirectionalIrradiancePreparation),"cohort second binding prepares: "+error);
    if(!sourceBindings[0]||!sourceBindings[1])return;
    std::vector<gc::NativeDirectGratingEndpointInput> endpointSet{input,input};
    const gc::Point secondBary{.525f,.025f,.45f};gc::Point secondPosition{};
    for(unsigned v=0;v<3;++v)secondPosition=gc::advance(secondPosition,triangle.positions[v],secondBary[v]);
    const auto secondOrigin=gc::advance(secondPosition,triangle.normal,1);
    endpointSet[1].originMinimum={secondOrigin[0],secondOrigin[1],secondOrigin[2],gc::kRayMinimum};
    std::shared_ptr<const gc::DirectGratingIrradianceCohort> cohort;
    check(gc::prepareDirectGratingIrradianceCohort(sourceBindings,endpointSet,0,0,cohort,error),"full ordered D8 cohort freezes: "+error);
    if(!cohort)return;
    check(cohort->sources().size()==2&&cohort->sources()[0].get()!=sourceBindings[0].get()
        &&cohort->inputs().size()==2,"cohort independently freezes source bindings and ordered endpoints");
    check(cohort->bounds().bandWork==32&&cohort->bounds().combinedWork<=gc::kMaximumQueryWork
        &&cohort->bounds().depositionWork<=gc::kMaximumDepositionWork
        &&cohort->bounds().bytes<=renderpassoutput::kMaximumTotalBytes,"cohort charges both source batches and retained sum against existing caps");
    PreparedDirectGratingCohort common;common.cohort=cohort;common.widthValue=fixture.frame.width;common.heightValue=fixture.frame.height;
    common.sealed={cohort->identity(),1,20,3,4,20,20};
    std::vector<gc::NativeDirectGratingContributionReadback> receipts;
    for(std::size_t s=0;s<2;++s) {
        auto batch=std::make_shared<PreparedDirectGratingFrame>();batch->binding=cohort->sources()[s];const auto& frozen=*batch->binding;
        batch->widthValue=frozen.frame.width;batch->heightValue=frozen.frame.height;
        const auto serial=std::uint64_t(3*s);
        batch->uploaded={program->backend,{frozen.identity,1,serial+2,3,4,5,serial+2},11+s,
            gc::directionalSourceInput(frozen.prepared,frozen.frame),true};
        batch->endpoints.submission={frozen.identity,1,serial+3,3,4,6,serial+3};batch->endpoints.generation=13+s;
        batch->endpoints.inputs=endpointSet;batch->endpoints.complete=true;
        auto receipt=result;receipt.submission={frozen.identity,1,serial+7,3,4,8,serial+7};
        receipt.endpointGeneration=13+s;receipt.sourceGeneration=11+s;receipt.queryWork=4;receipt.bandWork=16;receipt.samples.clear();
        for(unsigned e=0;e<2;++e) {
            auto contribution=sample;contribution.input=endpointSet[e];contribution.source.input=batch->uploaded.input;
            contribution.source.directionTrace=frozen.frame.sourceDirectionTrace;
            for(unsigned band=0;band<8;++band)contribution.source.rawRadiance[band]=frozen.frame.sourceRadiance[band/4][band%4];
            if(e){contribution.endpoint.position=secondPosition;contribution.barycentric=secondBary;}
            const auto& p=contribution.endpoint.position;contribution.sourceOriginMinimum={p[0],p[1],p[2],gc::kRayMinimum};
            contribution.sourceDirectionMaximum=frozen.frame.sourceDirectionTrace;
            const gc::Point direction{frozen.frame.sourceDirectionTrace[0],frozen.frame.sourceDirectionTrace[1],frozen.frame.sourceDirectionTrace[2]};
            for(unsigned band=0;band<8;++band)contribution.surfaceIrradiance[band]=batch->uploaded.input.coefficients[band]
                *std::max(0.0f,gc::dot(triangle.normal,direction));
            receipt.samples.push_back(contribution);
        }
        batch->productionSubmission=receipt.submission;batch->productionReadback=receipt;
        common.batches.push_back(batch);receipts.push_back(receipt);
    }
    gc::DirectGratingIrradianceCohortReadback sum;
    const auto accumulate=[&](const auto& values){return gc::accumulateDirectGratingIrradianceCohort(cohort,common,common.sealed,.25,values,sum,error);};
    check(accumulate(receipts),"mock retained native source cohort validates atomically: "+error);
    if(sum.endpoints.empty())return;
    check(sum.cohort==cohort&&sum.endpoints.size()==2&&sum.sourceMeasure==diffractionmaterial::DirectionalSpectralMeasure::IrradianceCoefficient,
        "sum retains its complete frozen cohort and physical transient measure");
    for(unsigned e=0;e<2;++e)for(unsigned band=0;band<8;++band)
        check(sum.endpoints[e].surfaceIrradiance[band]==float(double(receipts[0].samples[e].surfaceIrradiance[band])
            +receipts[1].samples[e].surfaceIrradiance[band]),"sum adds already-physical surface contributions exactly once");
    check(sum.endpoints[0].surfaceIrradiance[1]>0&&sum.endpoints[0].surfaceIrradiance[2]==0
        &&sum.endpoints[0].surfaceIrradiance[0]!=6&&sum.endpoints[0].surfaceIrradiance[0]!=.5f,
        "tiny positive, supported zero, cosine-once and independent D versus Li are retained");
    gc::DirectGratingTerminalOrderLedger terminalIncumbent;
    directGratingTerminalOrderContract(common,receipts,true,&terminalIncumbent);
    const auto accepted=sum;
    const auto refuses=[&](const auto& values,const std::string& message) {
        sum=accepted;check(!accumulate(values)&&sum.cohort==accepted.cohort&&sum.submission==accepted.submission
            &&sum.sourceMeasure==accepted.sourceMeasure&&sum.endpoints.size()==accepted.endpoints.size()
            &&sum.endpoints[0].surfaceIrradiance==accepted.endpoints[0].surfaceIrradiance
            &&sum.endpoints[1].surfaceIrradiance==accepted.endpoints[1].surfaceIrradiance,message);
    };
    auto missing=receipts;missing.pop_back();refuses(missing,"missing source cannot publish a partial sum");
    auto extra=receipts;extra.push_back(receipts[0]);refuses(extra,"extra source cannot publish a cohort sum");
    std::swap(common.batches[0],common.batches[1]);refuses(receipts,"swapped native source owners refuse");
    std::swap(common.batches[0],common.batches[1]);
    const auto originalSecond=common.batches[1];common.batches[1]=common.batches[0];refuses(receipts,"duplicate native source owner refuses");
    common.batches[1]=originalSecond;
    for(unsigned mode=0;mode<13;++mode) {
        auto bad=receipts;auto& value=bad[1];
        if(mode==0)value.complete=false;
        if(mode==1)value.overflow=true;
        if(mode==2)++value.sourceGeneration;
        if(mode==3)++value.endpointGeneration;
        if(mode==4)++value.submission.rendererGeneration;
        if(mode==5)value.samples[0].outcome=gc::NativeDirectGratingContributionSample::Outcome::Unsupported;
        if(mode==6)value.samples[0].surfaceIrradiance[0]=std::numeric_limits<float>::infinity();
        if(mode==7){value.samples[0].endpoint.position[0]=std::nextafter(value.samples[0].endpoint.position[0],100.0f);
            value.samples[0].sourceOriginMinimum[0]=value.samples[0].endpoint.position[0];}
        if(mode==8)value.samples.pop_back();
        if(mode==9)value.samples[0].source.input.identifiers[3]++;
        if(mode==10)value.samples[0].source.input.coefficients[0]+=1;
        if(mode==11)++value.submission.queryGeneration;
        if(mode==12)value.submission.completedFence=0;
        common.batches[1]->productionReadback=value;
        refuses(bad,"even matching retained mock output cannot bypass source/frame/completeness/endpoint semantics");
    }
    common.batches[1]->productionReadback=receipts[1];
    common.batches[1]->cache=2;refuses(receipts,"source view from another native resource owner refuses");common.batches[1]->cache=1;
    const auto exactUploaded=common.batches[1]->uploaded;
    common.batches[1]->uploaded.input.coefficients[0]+=1;auto wrongSourceInput=receipts;
    for(auto& contribution:wrongSourceInput[1].samples)contribution.source.input=common.batches[1]->uploaded.input;
    common.batches[1]->productionReadback=wrongSourceInput[1];
    refuses(wrongSourceInput,"matching retained mock source-input edits still differ from the exact admitted 48-byte owner");
    common.batches[1]->uploaded=exactUploaded;common.batches[1]->productionReadback=receipts[1];
    auto reorderedEndpoints=receipts;std::swap(reorderedEndpoints[1].samples[0],reorderedEndpoints[1].samples[1]);
    std::swap(common.batches[1]->endpoints.inputs[0],common.batches[1]->endpoints.inputs[1]);
    common.batches[1]->productionReadback=reorderedEndpoints[1];refuses(reorderedEndpoints,"valid per-source endpoint reordering differs from the frozen cohort");
    std::swap(common.batches[1]->endpoints.inputs[0],common.batches[1]->endpoints.inputs[1]);
    common.batches[1]->productionReadback=receipts[1];
    common.current=false;refuses(receipts,"missing actual common native context refuses");common.current=true;
    common.reserved=false;refuses(receipts,"missing common cumulative native reservation refuses");common.reserved=true;
    check(!gc::accumulateDirectGratingIrradianceCohort(cohort,native,common.sealed,.25,receipts,sum,error),
        "existing single-source inspector cannot claim native cohort authority");
    auto supportedZero=receipts;
    for(auto& contribution:supportedZero[1].samples) {
        const auto& blocker=cohort->sources()[1]->frame.triangles.front();const auto& direction=contribution.source.directionTrace;
        auto& shadow=contribution.sourceHit;const auto& p=contribution.endpoint.position;
        shadow.status=1;shadow.distance=(blocker.points[0][2]-p[2])/direction[2];
        shadow.position=gc::advance(p,gc::Point{direction[0],direction[1],direction[2]},shadow.distance);
        shadow.outwardNormal=blocker.normal;shadow.objectId=blocker.object;shadow.materialId=blocker.material;
        shadow.instanceId=blocker.instance;shadow.triangleId=blocker.triangle;shadow.entering=gc::dot(gc::Point{direction[0],direction[1],direction[2]},blocker.normal)<0;
        contribution.surfaceIrradiance.fill(0);contribution.outcome=gc::NativeDirectGratingContributionSample::Outcome::Zero;
    }
    common.batches[1]->productionReadback=supportedZero[1];check(accumulate(supportedZero),"complete supported-zero source remains in the cohort: "+error);
    check(sum.endpoints[0].surfaceIrradiance==receipts[0].samples[0].surfaceIrradiance,"supported zero contributes zero without omitting its source");
    directGratingTerminalOrderContract(common,supportedZero,true);
    for(unsigned mode=0;mode<3;++mode) {
        auto trial=fixture;trial.scene=std::make_shared<HarmonicMIDI::grid::Visual3DScene>(*fixture.scene);
        if(mode==0)trial.scene->lights[1].transform.rotation=trial.scene->lights[0].transform.rotation;
        if(mode==1)for(unsigned s=0;s<2;++s)trial.scene->lights[s].spectralSource.directionalIrradiance.coefficients.fill(0);
        if(mode==2)trial.scene->lights[1].transform.rotation={0,1,0,0};
        check(trial.prepare(error),"terminal identical/zero/backfacing trial source prepares: "+error);
        auto trialSecond=trial;trialSecond.ordinaryLight=trial.scene->lights[1].id;
        check(trialSecond.prepare(error),"terminal trial second source prepares: "+error);
        std::vector<std::shared_ptr<const gc::GratingIncidentBinding>> bindings(2);
        check(gc::prepareGratingIncidentBinding(trial.prepared,trial.frame,program,material,.25,bindings[0],error,gc::SourceMeasureUse::DirectionalIrradiancePreparation)
            &&gc::prepareGratingIncidentBinding(trialSecond.prepared,trialSecond.frame,program,material,.25,bindings[1],error,gc::SourceMeasureUse::DirectionalIrradiancePreparation),
            "terminal trial freezes exact source material/scene/time: "+error);
        PreparedDirectGratingCohort trialCommon;
        check(gc::prepareDirectGratingIrradianceCohort(bindings,endpointSet,0,0,trialCommon.cohort,error),"terminal trial complete roster prepares: "+error);
        if(!trialCommon.cohort)continue;
        trialCommon.widthValue=common.widthValue;trialCommon.heightValue=common.heightValue;
        trialCommon.sealed=common.sealed;trialCommon.sealed.frameIdentity=trialCommon.cohort->identity();
        auto trialReceipts=receipts;
        for(unsigned s=0;s<2;++s) {
            auto batch=std::make_shared<PreparedDirectGratingFrame>();batch->binding=trialCommon.cohort->sources()[s];
            const auto& frozen=*batch->binding;batch->widthValue=frozen.frame.width;batch->heightValue=frozen.frame.height;
            batch->uploaded=common.batches[s]->uploaded;batch->uploaded.submission.frameIdentity=frozen.identity;
            batch->uploaded.input=gc::directionalSourceInput(frozen.prepared,frozen.frame);
            batch->endpoints=common.batches[s]->endpoints;batch->endpoints.submission.frameIdentity=frozen.identity;
            auto& r=trialReceipts[s];r.submission.frameIdentity=frozen.identity;
            for(auto& contribution:r.samples) {
                contribution.source.input=batch->uploaded.input;contribution.source.directionTrace=frozen.frame.sourceDirectionTrace;
                for(unsigned b=0;b<8;++b)contribution.source.rawRadiance[b]=frozen.frame.sourceRadiance[b/4][b%4];
                contribution.sourceDirectionMaximum=frozen.frame.sourceDirectionTrace;
                const gc::Point direction{frozen.frame.sourceDirectionTrace[0],frozen.frame.sourceDirectionTrace[1],frozen.frame.sourceDirectionTrace[2]};
                bool positive=false;
                for(unsigned b=0;b<8;++b) {
                    contribution.surfaceIrradiance[b]=batch->uploaded.input.coefficients[b]*std::max(0.0f,gc::dot(triangle.normal,direction));
                    positive|=contribution.surfaceIrradiance[b]>0;
                }
                contribution.outcome=positive?gc::NativeDirectGratingContributionSample::Outcome::Positive:gc::NativeDirectGratingContributionSample::Outcome::Zero;
            }
            batch->productionSubmission=r.submission;batch->productionReadback=r;trialCommon.batches.push_back(batch);
        }
        directGratingTerminalOrderContract(trialCommon,trialReceipts,mode!=0);
        if(mode==0&&!terminalIncumbent.bands.empty()) {
            auto incumbent=terminalIncumbent;auto ownerOnly=incumbent;ownerOnly.nativeResult.reset();
            auto alias=ownerOnly;alias.incident.cohort=trialCommon.cohort;
            gc::DirectGratingTerminalOrderBounds foreignBounds,aliasBounds;
            check(gc::calculateDirectGratingTerminalOrderBounds(*trialCommon.cohort,trialReceipts,trialCommon.terminalBudget,ownerOnly,foreignBounds,error)
                &&gc::calculateDirectGratingTerminalOrderBounds(*trialCommon.cohort,trialReceipts,trialCommon.terminalBudget,alias,aliasBounds,error)
                &&foreignBounds.bytes-aliasBounds.bytes==cohort->bounds().bytes,"distinct retained terminal incumbent charges all frozen source/material owners");
            const auto budget=trialCommon.terminalBudget;
            trialCommon.terminalBudget.retainedBytes+=renderpassoutput::kMaximumTotalBytes-aliasBounds.bytes;
            check(!gc::evaluateDirectGratingTerminalOrders(trialCommon.cohort,trialCommon,trialCommon.sealed,.25,trialReceipts,incumbent,error)
                &&sameTerminalLedger(incumbent,terminalIncumbent),"foreign incumbent owner exceeding remaining authenticated headroom refuses atomically");
            trialCommon.terminalBudget=budget;
            check(gc::evaluateDirectGratingTerminalOrders(trialCommon.cohort,trialCommon,trialCommon.sealed,.25,trialReceipts,incumbent,error)
                &&incumbent.incident.cohort==trialCommon.cohort,"complete mock-boundary terminal replacement retains its new frozen owners");
        }
    }
    for(const auto& badSources:std::vector<std::vector<std::shared_ptr<const gc::GratingIncidentBinding>>>{
        {sourceBindings[0]},{sourceBindings[0],sourceBindings[0]}, {sourceBindings[1],sourceBindings[0]},
        {sourceBindings[0],sourceBindings[1],sourceBindings[0]},std::vector<std::shared_ptr<const gc::GratingIncidentBinding>>(17,sourceBindings[0])}) {
        auto retained=cohort;check(!gc::prepareDirectGratingIrradianceCohort(badSources,endpointSet,0,0,retained,error)&&retained==cohort,
            "roster missing/duplicate/reordered/extra/over-limit source preparation refuses atomically");
    }
    std::shared_ptr<const gc::GratingIncidentBinding> later;
    check(gc::prepareGratingIncidentBinding(secondFixture.prepared,secondFixture.frame,program,material,.5,later,error,
        gc::SourceMeasureUse::DirectionalIrradiancePreparation),"independent later-time source remains individually valid");
    auto retained=cohort;check(!gc::prepareDirectGratingIrradianceCohort({sourceBindings[0],later},endpointSet,0,0,retained,error)&&retained==cohort,
        "individually valid source at another evaluation time refuses common cohort");
    arbitgpu::NativeFixtureSceneRuntimeInputs movedCamera;movedCamera.cameraTranslationOffset[0]=.25f;gc::Frame movedFrame;
    check(gc::buildFrame(secondFixture.prepared,secondFixture.frame.instances,movedCamera,32,32,movedFrame,error),"independent moved-camera frame prepares");
    check(gc::prepareGratingIncidentBinding(secondFixture.prepared,movedFrame,program,material,.25,later,error,
        gc::SourceMeasureUse::DirectionalIrradiancePreparation),"independent moved-camera source binding prepares");
    check(!gc::prepareDirectGratingIrradianceCohort({sourceBindings[0],later},endpointSet,0,0,retained,error)&&retained==cohort,
        "individually valid source at another camera frame refuses common cohort");
    check(!gc::prepareDirectGratingIrradianceCohort(sourceBindings,endpointSet,std::numeric_limits<std::uint64_t>::max(),0,retained,error)
        &&retained==cohort&&!gc::prepareDirectGratingIrradianceCohort(sourceBindings,endpointSet,0,gc::kMaximumQueryWork,retained,error),
        "whole-cohort bytes/work overflow refuses before publication");
    auto mutableBinding=std::make_shared<gc::GratingIncidentBinding>(*sourceBindings[0]);
    std::shared_ptr<const gc::DirectGratingIrradianceCohort> snapshot;
    check(gc::prepareDirectGratingIrradianceCohort({mutableBinding,sourceBindings[1]},endpointSet,0,0,snapshot,error),
        "caller-aliased source binding freezes independently");
    if(snapshot) {
        mutableBinding->frame.ior[0]+=1;
        check(snapshot->sources()[0]->frame.ior[0]==cohort->sources()[0]->frame.ior[0],"mutable caller cannot change the frozen source frame");
        retained=cohort;check(!gc::prepareDirectGratingIrradianceCohort({mutableBinding,sourceBindings[1]},endpointSet,0,0,retained,error)
            &&retained==cohort,"stale source frame cannot prepare a replacement cohort");
    }
    endpointSet[0].originMinimum[0]+=1;
    check(!gc::sameDirectGratingEndpointInput(endpointSet[0],cohort->inputs()[0]),"caller edits cannot change the frozen endpoint roster");
}
class PreparedCameraTraceFrame final : public PreparedSourceFrame,public gc::NativeGlassCameraGratingTraceBinding
{
public:
    std::shared_ptr<const gc::GratingIncidentBinding> binding;
    gc::NativeGlassCameraGratingTraceUpload inputUpload;
    gc::NativeGlassCameraGratingTraceReadback retained;
    bool contextCurrent=true,dispatchCurrent=true;
    const arbitgpu::NativeFixtureSceneFrame& cameraTraceOwner() const noexcept override{return *this;}
    const std::shared_ptr<const gc::GratingIncidentBinding>& cameraTraceGratingBinding() const noexcept override{return binding;}
    const gc::NativeGlassCameraGratingTraceUpload& cameraTraceInputUpload() const noexcept override{return inputUpload;}
    bool validateCameraTraceProductionQuery(const gc::SubmissionIdentity& expected,
        const gc::NativeGlassCameraGratingTraceReadback& supplied,std::string& error) const override
    {
        if(!contextCurrent||!dispatchCurrent||!(expected==retained.submission)
            ||!gc::sameGlassCameraGratingTraceReadback(supplied,retained))
        {error="Synthetic camera trace has no exact retained output/context/dispatch";return false;}
        error.clear();return true;
    }
};
void glassCameraGratingTraceContract()
{
    // The fake native owner supplies fabricated finite misses and interface
    // prefixes. These are contract checks, never optics or visibility proof.
    auto fixture=gt::ordinaryFixture();const auto& target=fixture.scene->objects[1];
    for(std::size_t v=0;v<target.vertexCount;++v) {
        auto& vertex=fixture.scene->vertices[target.firstVertex+v];vertex.uv={vertex.position.x/12,vertex.position.y/12};
    }
    std::string error;check(fixture.prepare(error),"camera trace source fixture prepares: "+error);
    auto description=diffractionmaterial::makeAluminiumBinaryGratingPreset();description.spectrum.lastOrder=1;
    description.roughness.rmsSlope=0;description.microstructure.grooveDepthNanometres=.9f;
    const auto admitted=diffractionmaterial::admit(description,error);
    check(bool(admitted),"camera trace grating material admits: "+error);if(!admitted)return;
    auto material=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*admitted);
    auto program=std::make_shared<gt::Program>(*fixture.program->objectPrograms[0]);
    program->kind=arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;program->spectralReflectance.reset();
    program->diffractionPathCount=1;program->bindingDigest="camera-trace-material";program->programIdentity="camera-trace-program";
    program->diffractionPaths[0].material=diffractionmaterial::physicalcheckpoint::makeGpuParameters(*material,{});
    PreparedCameraTraceFrame native;
    check(gc::prepareGratingIncidentBinding(fixture.prepared,fixture.frame,program,material,.25,native.binding,error),
        "camera trace retains immutable scene/query/grating owner: "+error);if(!native.binding)return;
    const auto bindingOwner=native.binding;const auto& binding=*bindingOwner;const auto& frame=binding.frame;
    native.widthValue=frame.width;native.heightValue=frame.height;
    check(gc::admitGlassCameraGratingTraceCamera(frame,error),"camera has conservative outside-volume admission: "+error);
    auto inside=frame;inside.camera.transform.translation={frame.triangles.front().points[0][0],frame.triangles.front().points[0][1],frame.triangles.front().points[0][2]};
    check(!gc::admitGlassCameraGratingTraceCamera(inside,error),"camera on/in glass world bounds cannot assume initial air");
    inside=frame;for(auto& t:inside.triangles)t.glass=false;
    check(!gc::admitGlassCameraGratingTraceCamera(inside,error),"missing retained glass bounds refuse initial medium");
    std::array<gc::NativeGlassCameraGratingTraceInput,5> rays;
    for(unsigned location=0;location<5;++location) {
        check(gc::makeGlassCameraGratingTraceInput(frame,15,15,location,rays[location],error),"center/corner camera sample prepares: "+error);
        const double dx=location==0?.5:(location==2||location==3?1:0),dy=location==0?.5:(location==3||location==4?1:0);
        const double tangent=std::tan(double(frame.camera.verticalFovRadians)*.5),aspect=double(frame.width)/frame.height;
        const gt::D3 unnormalized{(2*(15+dx)/frame.width-1)*tangent*aspect,(1-2*(15+dy)/frame.height)*tangent,-1};
        const auto expected=gt::normalized(unnormalized);
        for(unsigned axis=0;axis<3;++axis)check(close(rays[location].directionMaximum[axis],expected[axis],2e-6),
            "owning identity-camera ray agrees with independent projection arithmetic");
        check(rays[location].originMinimum==gc::Texel{0,0,3,gc::kRayMinimum}
            &&rays[location].directionMaximum[3]==frame.camera.farPlane,"sample retains actual camera origin and finite trace interval");
    }
    auto preserved=rays[0];
    check(!gc::makeGlassCameraGratingTraceInput(frame,frame.width,0,0,preserved,error)
        &&gc::sameGlassCameraGratingTraceInput(preserved,rays[0]),"foreign pixel refusal preserves input output");
    check(!gc::makeGlassCameraGratingTraceInput(frame,0,0,5,preserved,error),"only center and four corners are admitted");
    inside=frame;inside.camera.transform.rotation={0,float(std::sin(.2)),0,float(std::cos(.2))};
    check(gc::makeGlassCameraGratingTraceInput(inside,15,15,0,preserved,error)
        &&!gc::sameGlassCameraTraceFloats(preserved.directionMaximum,rays[0].directionMaximum),"ray uses owning camera rotation");
    gc::GlassCameraGratingTraceBounds bounds;
    check(gc::calculateGlassCameraGratingTraceBounds(binding,1,0,0,bounds,error)
        &&bounds.maximumBands==8&&bounds.maximumVisits==8*64&&bounds.nativeInputBytes==64
        &&bounds.nativeOutputBytes==8*(160+64*256)&&bounds.queryWork>0,
        "all eight bands and worst-case64 native query visits are reserved together: "+error);
    bounds.bytes=123;
    check(!gc::calculateGlassCameraGratingTraceBounds(binding,gc::kMaximumEntrySamples+1,0,0,bounds,error)&&bounds.bytes==123,
        "sample bound refuses without replacing bounds");
    check(!gc::calculateGlassCameraGratingTraceBounds(binding,1,renderpassoutput::kMaximumTotalBytes,0,bounds,error)
        &&!gc::calculateGlassCameraGratingTraceBounds(binding,1,0,gc::kMaximumQueryWork,bounds,error),
        "camera trace shares original cumulative storage and query budget");
    native.inputUpload.backend=program->backend;native.inputUpload.submission={binding.identity,1,3,3,4,6,3};
    native.inputUpload.generation=13;native.inputUpload.inputs={rays[0]};native.inputUpload.complete=true;
    gc::NativeGlassCameraGratingTraceReadback readback;
    readback.version=1;readback.submission={binding.identity,1,7,3,4,8,7};readback.inputGeneration=13;readback.dispatchComplete=true;
    for(unsigned bandIndex=0;bandIndex<8;++bandIndex) {
        gc::NativeGlassCameraGratingTraceBand band;band.input=rays[0];band.nativeOriginMinimum=rays[0].originMinimum;
        band.nativeDirectionMaximum=rays[0].directionMaximum;band.version=1;band.band=bandIndex;band.ior=frame.ior[bandIndex];
        band.wavelength=diffractionmaterial::kProductionWavelengthsNanometres[bandIndex];band.traceComplete=true;band.unresolvedFractionKnown=true;band.queryWork=1;
        gc::NativeGlassCameraGratingTraceVisit visit;visit.originMinimum=band.nativeOriginMinimum;
        visit.directionFraction={band.nativeDirectionMaximum[0],band.nativeDirectionMaximum[1],band.nativeDirectionMaximum[2],1};
        visit.ior=band.ior;visit.disposition=gc::GlassCameraTraceDisposition::Escape;visit.hit.status=0;
        visit.hit.distance=frame.camera.farPlane;visit.hit.work=1;band.visits={visit};readback.bands.push_back(band);++readback.queryWork;
    }
    native.retained=readback;
    const auto validate=[&](const auto& supplied){return gc::validateGlassCameraGratingTraceReadback(binding,native,readback.submission,.25,supplied,error);};
    check(validate(readback),"synthetic exact native finite-miss witness shape admits: "+error);
    const auto copy=readback;check(validate(copy),"unchanged deep copy preserves exact actual-output association");
    for(unsigned mode=0;mode<11;++mode) {
        auto contradictory=readback;auto& visit=contradictory.bands[0].visits[0];
        if(mode==0){visit.tir=true;visit.reflection=.5f;visit.reflectedFraction=visit.transmittedFraction=.5f;}
        if(mode==1)visit.orientedNormal[0]=1;
        if(mode==2)visit.reflected[0]=1;
        if(mode==3)visit.transmitted[0]=1;
        if(mode==4)visit.fromIor=1;
        if(mode==5)visit.toIor=1;
        if(mode==6)visit.reflection=-0.0f;
        if(mode==7)visit.reflectedFraction=.5f;
        if(mode==8)visit.transmittedFraction=.5f;
        if(mode==9)visit.barycentric[0]=1;
        if(mode==10)contradictory.bands[0].unresolvedFractionKnown=false;
        native.retained=contradictory;
        check(gc::sameGlassCameraGratingTraceReadback(contradictory,native.retained)&&!validate(contradictory),
            "exact-retained completed miss cannot carry contradictory interface/barycentric/unknown witnesses");
    }
    native.retained=readback;
    for(unsigned mode=0;mode<33;++mode) {
        auto supplied=readback;auto& band=supplied.bands[0];auto& visit=band.visits[0];
        if(mode==0)++supplied.version;
        if(mode==1)++supplied.inputGeneration;
        if(mode==2)++supplied.queryWork;
        if(mode==3)++supplied.submission.scratchGeneration;
        if(mode==4)supplied.dispatchComplete=false;
        if(mode==5)supplied.overflow=true;
        if(mode==6)++band.input.pixel[0];
        if(mode==7)++band.input.control[0];
        if(mode==8)band.input.originMinimum[0]=-0.0f;
        if(mode==9)band.input.directionMaximum[3]-=1;
        if(mode==10)band.nativeOriginMinimum[0]=-0.0f;
        if(mode==11)band.nativeDirectionMaximum[3]-=1;
        if(mode==12)++band.version;
        if(mode==13)++band.sample;
        if(mode==14)++band.band;
        if(mode==15)band.reasons=gc::GlassCameraVisitCap;
        if(mode==16)band.ior+=1;
        if(mode==17)band.wavelength+=1;
        if(mode==18)band.unresolvedFraction=1;
        if(mode==19)++band.queryWork;
        if(mode==20)band.traceComplete=false;
        if(mode==21)++visit.parent;
        if(mode==22)++visit.token;
        if(mode==23)++visit.interfaces;
        if(mode==24)++visit.history;
        if(mode==25)visit.inside=true;
        if(mode==26)visit.tir=true;
        if(mode==27)visit.ancestry[0]=1;
        if(mode==28)visit.directionFraction[3]=.5f;
        if(mode==29)visit.hit.distance-=1;
        if(mode==30)visit.reflected[0]=1;
        if(mode==31)visit.barycentric[0]=1;
        if(mode==32)band.unresolvedFractionKnown=false;
        check(!gc::sameGlassCameraGratingTraceReadback(supplied,native.retained)&&!validate(supplied),
            "supplied camera/band/branch edits cannot replace retained native output");
    }
    auto prefix=readback;for(auto& band:prefix.bands) {
        band.traceComplete=false;band.reasons=gc::GlassCameraQueryFailed;band.visits[0].disposition=gc::GlassCameraTraceDisposition::Unresolved;
        band.visits[0].hit.status=-1;band.visits[0].hit.distance=0;
        band.unresolvedFraction=1;
    }
    native.retained=prefix;check(validate(prefix),"completed synthetic dispatch preserves explicitly incomplete query prefix: "+error);
    auto forged=prefix;for(auto& band:forged.bands)band.unresolvedFraction=0;
    native.retained=forged;check(!validate(forged),"exact-retained failed root cannot report zero unresolved incoming fraction");
    forged=prefix;for(auto& band:forged.bands)band.visits[0].reflection=.5f;
    native.retained=forged;check(!validate(forged),"exact-retained Unresolved visit cannot carry an interface witness");
    forged=prefix;for(auto& band:forged.bands){band.traceComplete=true;band.reasons=0;}
    native.retained=forged;check(!validate(forged),"even retained failed query cannot acquire optical trace completion");
    prefix=readback;
    const auto glass=std::find_if(frame.triangles.begin(),frame.triangles.end(),[](const auto& t){return t.glass&&t.normal[2]>.9f;});
    check(glass!=frame.triangles.end(),"synthetic interface prefix has retained glass primitive");
    if(glass!=frame.triangles.end())for(auto& band:prefix.bands) {
        band.traceComplete=false;band.reasons=gc::GlassCameraVisitCap;band.unresolvedFraction=1;auto& visit=band.visits[0];
        const gc::Point direction{visit.directionFraction[0],visit.directionFraction[1],visit.directionFraction[2]};
        visit.disposition=gc::GlassCameraTraceDisposition::Interface;visit.hit.status=1;
        visit.hit.distance=(glass->points[0][2]-3)/direction[2];visit.hit.position=gc::advance(gc::Point{0,0,3},direction,visit.hit.distance);
        visit.hit.outwardNormal=visit.orientedNormal=glass->normal;visit.hit.entering=true;
        visit.hit.objectId=glass->object;visit.hit.materialId=glass->material;visit.hit.instanceId=glass->instance;visit.hit.triangleId=glass->triangle;
        visit.reflection=.04f;visit.reflectedFraction=.04f;visit.transmittedFraction=.96f;
        visit.reflected={direction[0],direction[1],-direction[2]};visit.transmitted=direction;visit.fromIor=1;visit.toIor=band.ior;
    }
    if(glass!=frame.triangles.end()) {
        native.retained=prefix;check(validate(prefix),"retained interface cap prefix admits without claiming followed positive children: "+error);
        const auto stoppedInterface=prefix;
        forged=prefix;for(auto& band:forged.bands)band.unresolvedFraction=0;
        native.retained=forged;check(!validate(forged),"exact-retained omitted positive children cannot report zero unresolved fraction");
        // The represented reflected child is already resolved. Only the absent
        // transmitted frontier contributes, never its incoming root again.
        auto frontier=prefix;
        for(auto& band:frontier.bands) {
            const auto& parent=band.visits[0];gc::NativeGlassCameraGratingTraceVisit child;
            child.visit=1;child.parent=0;child.token=1;child.interfaces=1;child.history=1;child.ancestry[0]=1;
            const auto origin=gc::advance(parent.hit.position,parent.orientedNormal,gc::bias(parent.hit.position));
            child.originMinimum={origin[0],origin[1],origin[2],gc::kRayMinimum};
            child.directionFraction={parent.reflected[0],parent.reflected[1],parent.reflected[2],parent.reflectedFraction};
            child.ior=band.ior;child.disposition=gc::GlassCameraTraceDisposition::Escape;child.hit.status=0;
            child.hit.distance=frame.camera.farPlane;child.hit.work=1;band.unresolvedFraction=parent.transmittedFraction;
            band.visits.push_back(child);band.queryWork=2;
        }
        frontier.queryWork=16;native.retained=frontier;
        check(validate(frontier),"represented terminal child leaves only the missing positive frontier unresolved: "+error);
        forged=frontier;for(auto& band:forged.bands)band.unresolvedFraction=1;
        native.retained=forged;check(!validate(forged),"exact-retained ancestor cannot double-count an already represented terminal child");
        for(auto& band:frontier.bands) {
            const auto& parent=band.visits[0];gc::NativeGlassCameraGratingTraceVisit child;
            child.visit=2;child.parent=0;child.token=65;child.interfaces=1;child.ancestry[0]=65;child.inside=true;
            const auto origin=gc::advance(parent.hit.position,parent.orientedNormal,-gc::bias(parent.hit.position));
            child.originMinimum={origin[0],origin[1],origin[2],gc::kRayMinimum};
            child.directionFraction={parent.transmitted[0],parent.transmitted[1],parent.transmitted[2],parent.transmittedFraction};
            child.ior=band.ior;child.disposition=gc::GlassCameraTraceDisposition::Unresolved;child.hit.status=-1;child.hit.work=1;
            band.visits.push_back(child);band.queryWork=3;band.reasons=gc::GlassCameraQueryFailed;
        }
        frontier.queryWork=24;native.retained=frontier;
        check(validate(frontier),"explicit unresolved child replaces its omitted frontier without double counting: "+error);
        forged=frontier;for(auto& band:forged.bands)band.unresolvedFraction=0;
        native.retained=forged;check(!validate(forged),"exact-retained explicit unresolved child cannot disappear from fraction accounting");
        forged=prefix;for(auto& band:forged.bands){band.traceComplete=true;band.reasons=0;}
        native.retained=forged;check(!validate(forged),"missing strictly positive branches cannot become complete even with fake native equality");
        for(auto& band:prefix.bands){band.reasons=gc::GlassCameraFractionUnderflow;auto& visit=band.visits[0];
            visit.reflection=std::numeric_limits<float>::denorm_min();visit.reflectedFraction=0;visit.transmittedFraction=1;}
        native.retained=prefix;check(validate(prefix),"positive fraction underflow retains explicit incomplete native prefix: "+error);
        auto unknown=prefix;
        for(auto& band:unknown.bands) {
            const auto& parent=band.visits[0];gc::NativeGlassCameraGratingTraceVisit child;
            child.visit=1;child.parent=0;child.token=65;child.interfaces=1;child.ancestry[0]=65;child.inside=true;
            const auto origin=gc::advance(parent.hit.position,parent.orientedNormal,-gc::bias(parent.hit.position));
            child.originMinimum={origin[0],origin[1],origin[2],gc::kRayMinimum};
            child.directionFraction={parent.transmitted[0],parent.transmitted[1],parent.transmitted[2],parent.transmittedFraction};
            child.ior=band.ior;child.disposition=gc::GlassCameraTraceDisposition::Unresolved;child.hit.status=-1;child.hit.work=1;
            band.visits.push_back(child);band.queryWork=2;band.reasons|=gc::GlassCameraQueryFailed;
            band.unresolvedFractionKnown=false;band.unresolvedFraction=0;
        }
        unknown.queryWork=16;native.retained=unknown;
        check(validate(unknown),"partly followed underflow retains explicit unknown amount rather than a zero witness: "+error);
        forged=unknown;for(auto& band:forged.bands){band.unresolvedFractionKnown=true;band.unresolvedFraction=1;}
        native.retained=forged;check(!validate(forged),"exact-retained numerical loss cannot pretend its unresolved amount is known");
        forged=unknown;for(auto& band:forged.bands)band.unresolvedFraction=.5f;
        native.retained=forged;check(!validate(forged),"unknown amount requires canonical sentinel rather than a fabricated fraction");
        forged=stoppedInterface;for(auto& band:forged.bands){band.unresolvedFractionKnown=false;band.unresolvedFraction=0;}
        native.retained=forged;check(!validate(forged),"known whole-parent cap cannot relabel its fraction unknown");
        auto abandoned=prefix;for(auto& band:abandoned.bands) {
            auto& visit=band.visits[0];visit.disposition=gc::GlassCameraTraceDisposition::Unresolved;
            visit.orientedNormal=visit.transmitted=visit.reflected={};
            visit.reflection=visit.transmittedFraction=visit.reflectedFraction=visit.fromIor=visit.toIor=0;
        }
        native.retained=abandoned;check(validate(abandoned),"canonical whole-parent underflow abandonment keeps its known incoming fraction: "+error);
        forged=prefix;for(auto& band:forged.bands){band.reasons=0;band.traceComplete=true;}
        native.retained=forged;check(!validate(forged),"positive underflow cannot silently prune into complete tracing");
    }
    native.retained=readback;native.contextCurrent=false;check(!validate(readback),"stale exact native context refuses");native.contextCurrent=true;
    native.dispatchCurrent=false;check(!validate(readback),"lost native dispatch/fence association refuses");native.dispatchCurrent=true;
    native.cache=0;check(!validate(readback),"lost native frame lifetime refuses");native.cache=1;
    const auto upload=native.inputUpload;native.inputUpload.submission.queryGeneration++;
    check(!validate(readback),"stale uploaded camera query generation refuses");native.inputUpload=upload;
    native.binding=std::make_shared<const gc::GratingIncidentBinding>(binding);
    check(!gc::validateGlassCameraGratingTraceReadback(binding,native,readback.submission,.25,readback,error),"replacement scene/material binding cannot replace exact native owner");
    native.binding=bindingOwner;
    check(!gc::validateGlassCameraGratingTraceReadback(binding,native,readback.submission,.5,readback,error),"changed evaluation time refuses");
    gc::NativeGlassCameraGratingTracePackedHeader header{};std::array<gc::NativeGlassCameraGratingTracePackedVisit,1> packed{};
    const auto floats=[](gc::Texel values){return gc::GlassCameraTraceWords{gc::glassCameraTraceFloatBits(values[0]),gc::glassCameraTraceFloatBits(values[1]),
        gc::glassCameraTraceFloatBits(values[2]),gc::glassCameraTraceFloatBits(values[3])};};
    const auto& band=readback.bands[0];const auto& visit=band.visits[0];
    header[0]=band.input.pixel;header[1]=band.input.control;header[2]=floats(band.input.originMinimum);header[3]=floats(band.input.directionMaximum);
    header[4]=floats(band.nativeOriginMinimum);header[5]=floats(band.nativeDirectionMaximum);header[6]={1,0,0,1};header[7]={1,0,1,0};
    header[8]=floats({band.ior,band.wavelength,0,0});header[8][3]=1;packed[0][0]={0,gc::kGlassCameraTraceRoot,0,0};packed[0][1]={0,0,3,0};
    packed[0][2]=floats(visit.originMinimum);packed[0][3]=floats(visit.directionFraction);packed[0][5]={0,1,0,0};
    packed[0][6]=floats({0,0,0,frame.camera.farPlane});packed[0][11]=floats({0,0,0,band.ior});
    gc::NativeGlassCameraGratingTraceBand decoded;
    check(gc::decodeGlassCameraGratingTraceBand(header,packed.data(),packed.size(),decoded,error),"fixed uint/float-bit camera record decodes: "+error);
    auto decodedReadback=readback;decodedReadback.bands[0]=decoded;
    check(gc::sameGlassCameraGratingTraceReadback(decodedReadback,readback),"decoded layout retains every exact echoed field");
    header[8][3]=0;
    check(gc::decodeGlassCameraGratingTraceBand(header,packed.data(),packed.size(),decoded,error)&&!decoded.unresolvedFractionKnown,
        "native uint unknown flag is explicit and survives decode");
    header[8][3]=2;check(!gc::decodeGlassCameraGratingTraceBand(header,packed.data(),packed.size(),decoded,error),"nonboolean native known flag refuses");header[8][3]=1;
    header[9][0]=1;check(!gc::decodeGlassCameraGratingTraceBand(header,packed.data(),packed.size(),decoded,error),"nonzero native reserved header refuses");
    header[9]={};header[6][3]=65;check(!gc::decodeGlassCameraGratingTraceBand(header,packed.data(),packed.size(),decoded,error),"native visit count cannot exceed64 or available storage");
}
void gratingCameraReceipts()
{
    // Synthetic receipt fixtures prove the private validator, never GPU work.
    struct Packet {
        gt::Fixture fixture;
        std::shared_ptr<const gc::GratingIncidentBinding> binding;
        PreparedSourceFrame source;
        gc::NativeReadback native;
        std::shared_ptr<gc::NativeGratingCameraReadback> camera;
    };
    const auto make=[&](bool blocker,bool outside,bool physical,bool slanted=false,bool rotated=false)->std::unique_ptr<Packet> {
        auto p=std::make_unique<Packet>();auto& fixture=p->fixture;fixture=gt::ordinaryFixture();
        for(std::size_t v=0;v<fixture.scene->vertexCount;++v)if(fixture.scene->vertices[v].position.z>0)
            fixture.scene->vertices[v].position.z-=.25f*fixture.scene->vertices[v].position.x;
        fixture.scene->lights[0].transform.rotation={0,0,0,1};
        fixture.scene->cameras[0].transform.translation={2,-2,-2};
        if(outside)fixture.scene->cameras[0].transform.rotation={0,1,0,0};
        if(rotated)fixture.scene->cameras[0].transform.rotation={0,0,float(std::sin(.15)),float(std::cos(.15))};
        if(blocker) {
            const auto first=fixture.scene->vertexCount;
            videohelper::tests::smoothglassfixture::receiver(*fixture.scene,-4,{0,0,0});
            for(std::size_t v=first;v<fixture.scene->vertexCount;++v)fixture.scene->vertices[v].position.z=-2.5f;
            const auto& object=fixture.scene->objects[fixture.scene->objectCount-1];
            auto owner=std::make_shared<gt::Program>(*fixture.program->objectPrograms[0]);
            owner->object=object.id;owner->authoredMaterial=object.material;owner->parameters.identifiers[0]=object.material.value;
            fixture.program->objectPrograms.push_back(owner);
        }
        const auto& target=fixture.scene->objects[1];
        for(std::size_t v=0;v<target.vertexCount;++v) {
            auto& vertex=fixture.scene->vertices[target.firstVertex+v];vertex.uv={vertex.position.x/12,vertex.position.y/12};
            if(slanted) {
                vertex.position.z+=.2f*vertex.position.x;
                const float scale=1/std::sqrt(1.04f);
                vertex.normal={-.2f*scale,0,scale};vertex.tangent={scale,0,.2f*scale,1};
            }
        }
        if(physical) {
            auto& d=fixture.scene->lights[0].spectralSource.directionalIrradiance;
            d.version=1;d.measure=diffractionmaterial::DirectionalSpectralMeasure::IrradianceCoefficient;d.coefficients={2,3,4,5,6,7,8,9}; }
        std::string error;
        if(!fixture.prepare(error)){check(false,"camera receipt scene prepares: "+error);return {};}
        auto d=diffractionmaterial::makeAluminiumBinaryGratingPreset();d.geometry.grooveSpacingNanometres=3000;
        d.spectrum.lastOrder=1;d.roughness.rmsSlope=0;d.microstructure.grooveDepthNanometres=.9f;
        const auto admitted=diffractionmaterial::admit(d,error);
        if(!admitted){check(false,"camera receipt material admits: "+error);return {};}
        auto material=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*admitted);
        auto owner=std::make_shared<gt::Program>(*fixture.program->objectPrograms[0]);
        owner->kind=arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;owner->spectralReflectance.reset();
        owner->diffractionPathCount=1;owner->diffractionPaths[0].material=diffractionmaterial::physicalcheckpoint::makeGpuParameters(*material,{});
        if(!gc::prepareGratingIncidentBinding(fixture.prepared,fixture.frame,owner,material,.25,p->binding,error,
            physical?gc::SourceMeasureUse::DirectionalIrradiancePreparation:gc::SourceMeasureUse::LegacyNative))
        {check(false,"camera receipt exact binding prepares: "+error);return {};}
        auto incidents=std::make_shared<gc::NativeGratingReadback>();incidents->complete=true;
        incidents->submission={p->binding->identity,1,7,3,4,8,7};
        p->native.frameIdentity=p->binding->identity;p->native.gratingIncidentVersion=gc::kGratingIncidentVersion;p->native.gratingIncidents=incidents;
        if(physical) {
            p->source.widthValue=fixture.frame.width;p->source.heightValue=fixture.frame.height;
            p->source.uploaded={fixture.program->backend,{p->binding->identity,1,2,3,4,5,2},11,
                gc::directionalSourceInput(fixture.prepared,fixture.frame),true};
            auto source=std::make_shared<gc::NativeDirectionalSourceReceipt>();source->version=1;source->bindingGeneration=11;source->complete=true;
            source->submission=incidents->submission;source->echo.input=p->source.uploaded.input;source->echo.directionTrace=fixture.frame.sourceDirectionTrace;
            for(unsigned b=0;b<8;++b)source->echo.rawRadiance[b]=fixture.frame.sourceRadiance[b/4][b%4];
            for(const auto& work:fixture.frame.work)source->potential.push_back(work.incidentPower);
            incidents->directionalSource=source;p->native.directionalSource=source;
        }
        auto camera=std::make_shared<gc::NativeGratingCameraReadback>();p->camera=camera;
        camera->version=gc::kGratingCameraReceiptVersion;camera->complete=true;camera->submission=incidents->submission;camera->incidents=incidents;
        p->native.gratingCameraVersion=gc::kGratingCameraReceiptVersion;p->native.gratingCamera=camera;
        const auto& triangle=p->binding->triangles.front();const auto triangles=gt::worldTriangles(*fixture.scene);
        // Independent binary64 quaternion rotation for the optical/ray fixture.
        // Use the authored camera quaternion, including its float storage error.
        const auto& q=fixture.frame.camera.transform.rotation;
        const double qLength=std::sqrt(double(q.x)*q.x+double(q.y)*q.y+double(q.z)*q.z+double(q.w)*q.w);
        const gt::D3 qVector{q.x/qLength,q.y/qLength,q.z/qLength};
        const auto rotate=[&](gt::D3 point) {
            const auto twiceCross=gt::plus(gt::D3{},gt::exterior(qVector,point),2);
            return gt::plus(gt::plus(point,twiceCross,q.w/qLength),gt::exterior(qVector,twiceCross)); };
        diffractionmaterial::DiffractionLocalFrame cameraAxes;
        cameraAxes.tangent=rotate({1,0,0});cameraAxes.bitangent=rotate({0,1,0});cameraAxes.normal=rotate({0,0,1});
        for(unsigned band=0;band<8;++band) {
            gc::NativeGratingIncident event;event.version=gc::kGratingIncidentVersion;event.complete=true;event.work=band;event.band=band;
            event.interfaces=2;event.ancestry[0]=event.ancestry[1]=65;event.fraction=.5f;event.power=fixture.frame.work[band].incidentPower*.5f;
            event.normal=event.hit.outwardNormal=triangle.normal;event.tangent=triangle.tangent;event.bitangent=triangle.bitangent;
            event.barycentric={1.0f/3,1.0f/3,1.0f/3};
            for(unsigned v=0;v<3;++v) {
                event.hit.position=gc::advance(event.hit.position,triangle.positions[v],event.barycentric[v]);
                for(unsigned a=0;a<2;++a)event.uv[a]+=triangle.uv[v][a]*event.barycentric[v]; }
            event.hit.status=1;event.hit.entering=true;event.hit.work=1;event.hit.distance=1;
            event.hit.objectId=triangle.object;event.hit.materialId=triangle.material;event.hit.instanceId=triangle.instance;event.hit.triangleId=triangle.triangle;
            event.origin=gc::advance(event.hit.position,event.normal,1);event.direction=gc::advance({},event.normal,-1);
            const auto& entry=fixture.frame.entries[fixture.frame.work[band].entry];
            const auto source=std::find_if(fixture.frame.triangles.begin(),fixture.frame.triangles.end(),[&](const auto& t) {
                return std::tie(t.object,t.material,t.instance,t.triangle)==std::tie(entry.object,entry.material,entry.instance,entry.triangle);});
            if(source==fixture.frame.triangles.end()){check(false,"camera receipt retains entry owner");return {};}
            if(!gc::smoothHitDifferential(gc::sub(source->points[1],source->points[0]),event.direction,event.normal,event.dHdu)
                ||!gc::smoothHitDifferential(gc::sub(source->points[2],source->points[0]),event.direction,event.normal,event.dHdv))
            {check(false,"camera receipt uses exact slanted-plane receiving differential");return {};}
            const auto incident=gt::normalized(gt::doublePoint(event.normal));
            diffractionmaterial::DiffractionOrderDistribution distribution;
            diffractionmaterial::DiffractionLocalFrame local;
            // Widen and orthonormalize independently of the camera validator.
            local.normal=gt::normalized(gt::doublePoint(event.normal));
            const auto tangent=gt::doublePoint(event.tangent);
            local.tangent=gt::normalized(gt::plus(tangent,local.normal,-gt::inner(tangent,local.normal)));
            local.bitangent=gt::exterior(local.normal,local.tangent);
            if(!diffractionmaterial::evaluateDiffractionOrders(*material,d.spectrum.wavelengthsNanometres[band],local,incident,{0,0},distribution,error))
            {check(false,"camera receipt orders evaluate: "+error);return {};}
            event.substrateReflectedPower=float(event.power*distribution.substrateReflectedPower);event.absorbedPower=float(event.power*(1-distribution.substrateReflectedPower));
            event.unresolvedPower=float(event.power*distribution.unresolvedReflectedPower);
            for(std::size_t o=0;o<distribution.orderCount;++o)if(distribution.orders[o].reflectedPower>0) {
                const auto& order=distribution.orders[o];auto& native=event.orders[event.orderCount++];native.order=order.primary;
                for(unsigned a=0;a<3;++a)native.direction[a]=float(order.direction[a]);
                native.power=float(event.power*order.reflectedPower); }
            const auto index=std::uint32_t(incidents->incidents.size());incidents->incidents.push_back(event);
            incidents->orderWork+=3;incidents->queryWork+=event.orderCount;
            gc::GratingReceivingArea area;
            if(!gc::gratingReceivingArea(*p->binding,event,area)){check(false,"camera receipt area derives");return {};}
            for(unsigned o=0;o<event.orderCount;++o) {
                const auto& order=event.orders[o];gc::NativeGratingCameraReadback::OrderReceipt r;
                r.incident=index;r.band=band;r.order=order.order;r.complete=true;
                if(outside)r.outcome=gc::NativeGratingCameraReadback::Outcome::OutsideCamera;
                else {
                    const double halfY=std::tan(double(fixture.frame.camera.verticalFovRadians)/2),halfX=halfY*fixture.frame.width/fixture.frame.height;
                    const auto outgoing=gt::doublePoint(order.direction);
                    const double backward=gt::inner(outgoing,cameraAxes.normal);
                    r.x=std::uint32_t((1-gt::inner(outgoing,cameraAxes.tangent)/(backward*halfX))*.5*fixture.frame.width);
                    r.y=std::uint32_t((1+gt::inner(outgoing,cameraAxes.bitangent)/(backward*halfY))*.5*fixture.frame.height);
                    r.cameraHit=gt::nativeHit(gt::closest(triangles,{2,-2,-2},gt::normalized({-order.direction[0],-order.direction[1],-order.direction[2]}),
                        fixture.frame.camera.nearPlane,fixture.frame.camera.farPlane));
                    // Independent native-shaped raster fixture from the true
                    // pixel-center ray, separate from the exact-order query.
                    const auto pixelRay=gt::normalized(rotate({(2*(r.x+.5)/fixture.frame.width-1)*halfX,
                        (1-2*(r.y+.5)/fixture.frame.height)*halfY,-1}));
                    const auto pixelHit=gt::closest(triangles,{2,-2,-2},pixelRay,fixture.frame.camera.nearPlane,fixture.frame.camera.farPlane);
                    r.raster={pixelHit.object,pixelHit.material,pixelHit.instance,pixelHit.triangle,float(gt::inner(gt::minus({2,-2,-2},pixelHit.position),cameraAxes.normal))};
                    const bool visible=gc::sameTriangle(event.hit,r.cameraHit);
                    r.outcome=visible?gc::NativeGratingCameraReadback::Outcome::Visible:gc::NativeGratingCameraReadback::Outcome::Occluded;
                    diffractionmaterial::DiffractionCameraPixelIntegral integral;
                    const diffractionmaterial::DiffractionCameraPixel pixel{cameraAxes,fixture.frame.camera.verticalFovRadians,fixture.frame.width,fixture.frame.height,r.x,r.y};
                    if(!diffractionmaterial::integrateDiscreteDiffractionCameraPixel(*material,band,local,incident,
                        {diffractionmaterial::DiffractionIncidentMeasure::SurfaceSpectralIrradiance,double(event.power)/area.area},pixel,integral,error))
                    {check(false,"camera receipt pixel arithmetic prepares: "+error);return {};}
                    for(std::size_t k=0;k<integral.orderCount;++k)if(integral.orders[k].order.primary==order.order &&visible)
                        r.spectralRadiance=float(integral.orders[k].integratedRadiance/integral.solidAngleSteradians);
                    camera->queryWork+=r.cameraHit.work;
                }
                camera->orders.push_back(r);
            }
        }
        return p;
    };
    auto p=make(false,false,true);if(!p)return;
    std::string error;const auto expected=p->camera->submission;
    const auto validate=[&](const gc::NativeGratingCameraReadback& receipt) {
        auto native=p->native;native.gratingCamera=std::make_shared<const gc::NativeGratingCameraReadback>(receipt);
        return gc::validateGratingCameraReadback(*p->binding,native,expected,error,&p->source); };
    check(validate(*p->camera),"private eight-band native camera contract validates exact-order visibility and P/A once: "+error);
    check(p->camera->orders.size()==24 &&p->camera->queryWork==24,"each incident signed order and band has one camera receipt");
    for(unsigned mode=0;mode<15;++mode) {
        auto bad=*p->camera;
        if(mode==0)bad.orders.pop_back();
        if(mode==1)bad.orders.push_back(bad.orders.front());
        if(mode==2)bad.orders[0].band=7;
        if(mode==3)bad.orders[0].cameraHit.materialId++;
        if(mode==4)bad.orders[0].raster.triangle++;
        if(mode==5)bad.orders[0].cameraHit.position[0]+=.1f;
        if(mode==6)bad.orders[0].spectralRadiance*=2;
        if(mode==7)bad.orders[0].x++;
        if(mode==8)bad.orders[0].outcome=gc::NativeGratingCameraReadback::Outcome::Occluded;
        if(mode==9)bad.queryWork++;
        if(mode==10)bad.orders[0].complete=false;
        if(mode==11)bad.incidents=std::make_shared<const gc::NativeGratingReadback>(*p->native.gratingIncidents);
        if(mode==12)bad.version++;
        if(mode==13)bad.complete=false;
        if(mode==14)bad.overflow=true;
        check(!validate(bad),"missing/duplicate/order/band/visibility/raster/math/completion/native owner refuses");
    }
    for(unsigned mode=0;mode<7;++mode) {
        auto bad=*p->camera;
        if(mode==0)bad.submission.rendererGeneration++;
        if(mode==1)bad.submission.submissionSerial++;
        if(mode==2)bad.submission.rasterGeneration++;
        if(mode==3)bad.submission.queryGeneration++;
        if(mode==4)bad.submission.scratchGeneration++;
        if(mode==5)bad.submission.completedFence=0;
        if(mode==6)bad.submission.frameIdentity+="foreign-time-camera-material";
        check(!validate(bad),"camera candidate cannot change any exact submission generation, identity or completed fence");
    }
    auto claim=*p->camera;claim.physicalColor=true;
    check(!validate(claim)&&error=="Grating camera Color requires exact finite receiving-cell coverage, unavailable in receipt v1",
        "complete native angular visibility cannot grant missing finite receiving-patch Color");
    for(bool blocker:{false,true}) {
        auto alternate=make(blocker,!blocker,true);if(!alternate)continue;
        check(gc::validateGratingCameraReadback(*alternate->binding,alternate->native,alternate->camera->submission,error,&alternate->source),
            "explicit occluded or out-of-camera native outcomes complete private receipt: "+error);
        for(const auto& r:alternate->camera->orders)check(r.spectralRadiance==0 &&r.outcome==(blocker
            ?gc::NativeGratingCameraReadback::Outcome::Occluded:gc::NativeGratingCameraReadback::Outcome::OutsideCamera),
            "occlusion and angular support remain independent zero-radiance outcomes");
    }
    auto slanted=make(false,false,true,true);
    if(slanted) {
        const auto verify=[&](const gc::NativeGratingCameraReadback& receipt) {
            auto native=slanted->native;native.gratingCamera=std::make_shared<const gc::NativeGratingCameraReadback>(receipt);
            return gc::validateGratingCameraReadback(*slanted->binding,native,slanted->camera->submission,error,&slanted->source); };
        check(verify(*slanted->camera),"slanted native pixel-center raster depth validates independently of exact-order query depth: "+error);
        std::size_t different=0;double largest=0;
        for(std::size_t i=0;i<slanted->camera->orders.size();++i) {
            const auto& r=slanted->camera->orders[i];
            const double difference=std::abs(r.raster.linearDepth-(-2-r.cameraHit.position[2]));
            if(difference>largest){largest=difference;different=i;}
        }
        const auto& r=slanted->camera->orders[different];
        check(largest>5e-4*r.raster.linearDepth,"slanted fixture genuinely separates raster and exact-order ray depths beyond tolerance");
        auto bad=*slanted->camera;bad.orders[different].raster.linearDepth=-2-bad.orders[different].cameraHit.position[2];
        check(!verify(bad),"query-derived depth cannot replace authentic slanted raster depth");
        bad=*slanted->camera;bad.orders[different].raster.linearDepth+=.1f;
        check(!verify(bad),"wrong slanted pixel-center raster depth refuses");
        bad=*slanted->camera;bad.orders[different].raster.material++;
        check(!verify(bad),"different raster pixel-center triangle owner refuses");
    }
    auto rotated=make(false,false,true,true,true);
    if(rotated) {
        check(gc::validateGratingCameraReadback(*rotated->binding,rotated->native,rotated->camera->submission,error,&rotated->source),
            "authored rotated camera and slanted grating retain valid binary64 optics: "+error);
        check(std::abs(rotated->binding->frame.camera.transform.rotation.z)>.1f
            &&std::abs(rotated->native.gratingIncidents->incidents.front().normal[0])>.1f,
            "rotation and slanted grating orientations remain authored and nontrivial");
        for(const auto& r:rotated->camera->orders)check(r.outcome==gc::NativeGratingCameraReadback::Outcome::Visible
            &&r.spectralRadiance>0,"rotated optical receipt retains visible nonzero radiance");
    }
    diffractionmaterial::DiffractionLocalFrame corrected,unchanged;
    const gc::Point normal{-.2f/std::sqrt(1.04f),0,1/std::sqrt(1.04f)};
    const gc::Point tangent{normal[2],0,-normal[0]};
    check(gc::gratingCameraOpticalFrame(tangent,{0,1,0},normal,corrected)
        &&std::abs(gt::inner(corrected.normal,corrected.normal)-1)<1e-10
        &&std::abs(gt::inner(corrected.tangent,corrected.normal))<1e-10
        &&gt::inner(gt::exterior(corrected.tangent,corrected.bitangent),corrected.normal)>1-1e-10,
        "native slanted frame converts to unit orthogonal right-handed binary64 axes");
    for(unsigned mode=0;mode<5;++mode) {
        auto t=tangent,b=gc::Point{0,1,0},n=normal;
        if(mode==0)t[0]=std::numeric_limits<float>::quiet_NaN();
        if(mode==1)n={};
        if(mode==2)b={0,-1,0};
        if(mode==3)t=gc::advance(t,n,3e-4f);
        if(mode==4)n=gc::advance({},n,1.001f);
        corrected=unchanged;
        check(!gc::gratingCameraOpticalFrame(t,b,n,corrected)
            &&corrected.tangent==unchanged.tangent&&corrected.bitangent==unchanged.bitangent&&corrected.normal==unchanged.normal,
            "nonfinite degenerate mirrored or excessive native optical correction refuses without output mutation");
    }
    std::array<double,3> direction{1,2,3};
    check(!gc::gratingCameraUnitDirection({0,0,1.001f},direction)&&direction==std::array<double,3>{1,2,3},
        "oversized incident length correction refuses without output mutation");
    auto legacy=make(false,false,false);if(legacy) {
        legacy->camera->physicalColor=true;
        check(!gc::validateGratingCameraReadback(*legacy->binding,legacy->native,legacy->camera->submission,error),
            "legacy scene-unit area cannot acquire a physical camera Color grant"); }
    gc::GratingCameraBounds bounds;
    check(gc::calculateGratingCameraBounds(*p->binding,bounds,error)
        &&bounds.maximumOrders==p->binding->bounds.maximumOrders &&bounds.combinedWork<=gc::kMaximumQueryWork,
        "optional camera storage and queries share the existing whole-frame caps");
    auto exhausted=*p->binding;exhausted.bounds.bytes=renderpassoutput::kMaximumTotalBytes-exhausted.frame.bounds.transientBytes;
    bounds.bytes=123;
    check(!gc::calculateGratingCameraBounds(exhausted,bounds,error)&&bounds.bytes==123,
        "optional camera resource overflow refuses without narrowing terminal-only admission or mutating output");
    auto detached=p->native;detached.gratingCameraVersion=0;
    check(!gc::validateGratingCameraReadback(*p->binding,detached,expected,error,&p->source),"camera receipt cannot omit its version marker");
}
void roughPhysics()
{
    for(unsigned depth:{0u,1u,7u,31u})
    {
        std::array<bool,16> seen{};
        for(unsigned sample=0;sample<16;++sample)
        {
            const auto uv=gc::roughSample(sample,depth);
            const auto stratum=unsigned(uv[0]*16);
            check(stratum<16 && !seen[stratum] && uv[0]>0 && uv[0]<1 && uv[1]>=0 && uv[1]<1,
                "each interface depth permutes all sixteen finite visible-normal strata");
            if(stratum<16)seen[stratum]=true;
            for(double roughness:{.05,.35})for(double angle:{0.0,.3,.65})for(double from:{1.0,1.5})
            {
                const double to=from==1?1.5:1;
                const gt::D3 ray{std::sin(angle),0,-std::cos(angle)};
                const auto reference=gt::roughOptics(ray,{0,0,1},from,to,roughness,sample,depth);
                gc::RoughInterface actual;
                const bool complete=gc::roughInterface({float(ray[0]),0,float(ray[2])},{0,0,1},
                    float(from),float(to),float(roughness),sample,depth,actual);
                check(complete,"finite rough caustic facet optics complete without a pixel fallback");
                if(!complete)continue;
                check(actual.optics.tir==reference.snell.tir,"rough facet total internal reflection matches binary64 admittance optics");
                check(close(actual.reflected,reference.reflected) && close(actual.transmitted,reference.transmitted)
                    && close(actual.masked,reference.masked),"rough branch powers and masking deficit match independent binary64 VNDF");
                check(close(actual.reflected+actual.transmitted+actual.masked,1,2e-7),
                    "rough interface accounts for reflected, transmitted and masked incident flux");
                for(unsigned axis=0;axis<3;++axis)
                {
                    check(close(actual.facet[axis],reference.facet[axis]),"sampled facet direction agrees with independent projected-disk construction");
                    check(close(actual.optics.reflected[axis],reference.reflectedDirection[axis]),"rough reflected direction follows the actual sampled facet");
                    if(!actual.optics.tir)check(close(actual.optics.transmitted[axis],reference.snell.direction[axis]),
                        "rough transmitted direction follows the actual sampled facet and band IOR");
                }
            }
        }
    }
    gc::RoughInterface equal;
    check(gc::roughInterface({0,0,-1},{0,0,1},1,1,.35f,15,31,equal) && equal.transmitted==1
        && equal.masked==0 && equal.optics.transmitted==gc::Point{0,0,-1},
        "index-matched rough control has no optical interface or masking deficit");
    gc::RoughInterface invalid;
    check(!gc::roughInterface({0,0,-1},{0,0,1},1,1.5f,.05f,16,0,invalid)
        && !gc::roughInterface({0,0,-1},{0,0,1},1,1.5f,.05f,0,32,invalid),
        "rough interface sample and depth remain within the released finite caps");
    const auto roughSource=gc::nativeRoughMathSource();
    check(roughSource.find("void causticRoughInterface(")!=std::string::npos
        && roughSource.find("vec2 causticRoughSample(")!=std::string::npos
        && roughSource.find("glassRoughInterface")==std::string::npos,
        "native caustic generators reuse the released rough kernel through their shared namespace");
    check(gc::nativeMathSource().find("causticRoughInterface")==std::string::npos,
        "released smooth native math remains separate from rough shader selection");
}
void roughBranches()
{
    std::string error;
    const arbitgpu::smoothglass::RoughNativeConsumer consumer{arbitgpu::NativeFixtureMaterialBackend::OpenGl,true,true,true};
    auto missing=gt::roughFixture();
    check(!missing.prepare(error),"rough caustic preparation requires an explicit native rough consumer");
    auto fixture=gt::roughFixture();const bool prepared=fixture.prepare(error,consumer);check(prepared,error);if(!prepared)return;
    check(fixture.frame.roughness==.05f && fixture.frame.interfaceSamples==16 && fixture.frame.entries.size()==32
        && fixture.frame.work.size()==fixture.frame.entries.size()*16*8,
        "finite rough prism schedules every entry, interface stratum and authored band");
    check(fixture.frame.bounds.queryWork<=gc::kMaximumQueryWork
        && fixture.frame.bounds.deposition.depositionWork<=gc::kMaximumDepositionWork
        && fixture.frame.bounds.transientBytes<=renderpassoutput::kMaximumTotalBytes,
        "meaningful rough transport fits unchanged query, deposition and memory bounds");
    auto smooth=gt::roughFixture(0);const bool smoothPrepared=smooth.prepare(error);check(smoothPrepared,error);if(!smoothPrepared)return;
    check(smooth.frame.interfaceSamples==1 && smooth.frame.work.size()==smooth.frame.entries.size()*8,
        "zero roughness retains the single released interface schedule");
    double maximumSpread=0,maximumMasked=0;
    for(std::size_t e=0;e<fixture.frame.entries.size();++e)for(unsigned band=0;band<8;++band)
    {
        double sum=0;gc::BranchReceipt smoothPath;
        const bool smoothComplete=gc::traceBranches(smooth.prepared,smooth.frame,smooth.frame.entries[e],band,
            *smooth.policy,smoothPath,error);check(smoothComplete,error);
        for(unsigned sample=0;sample<16;++sample)
        {
            const auto w=(e*16+sample)*8+band;const auto& work=fixture.frame.work[w];
            check(work.entry==e && work.band==band && work.interfaceSample==sample,
                "rough work preserves stable entry, sample, band order");
            sum+=work.incidentPower;
            gc::BranchReceipt actual;const bool complete=gc::traceBranches(fixture.prepared,fixture.frame,
                fixture.frame.entries[e],band,*fixture.policy,actual,error,sample);check(complete,error);
            const auto reference=gt::branchOracle(fixture,fixture.frame.entries[e],band,1e-13,4096,64,{},sample);
            check(reference.valid,"independent authored-triangle rough branch reference completes");
            check(close(actual.depositedFraction,reference.deposited,3e-5)
                && close(actual.escapedFraction,reference.escaped,3e-5)
                && close(actual.maskedFraction,reference.masked,3e-6),
                "bounded rough deposits, escapes and separate masking agree with independent binary64 branches");
            check(actual.withinErrorBound && actual.maskedFraction+actual.unresolvedFraction<=gc::kMaximumUnresolvedFraction
                && close(actual.depositedFraction+actual.escapedFraction+actual.unresolvedFraction+actual.maskedFraction,1,1e-12),
                "each completed finite rough stratum conserves flux and meets the original tail bound");
            check(close(reference.deposited+reference.escaped+reference.unresolved+reference.masked,1,1e-12),
                "independent rough branch ledger accounts for all launched power");
            maximumMasked=std::max(maximumMasked,actual.maskedFraction);
            for(const auto& node:actual.interfaceLedger)
            {
                check(node.interfaceSample==sample && close(node.incidentFraction,
                    node.reflectedFraction+node.transmittedFraction+node.maskedFraction,1e-12),
                    "rough interface ledger retains its stratum and its separate physical deficit");
                const auto match=std::find_if(reference.interfaces.begin(),reference.interfaces.end(),[&](const auto& n) {
                    return n.depth==node.depth && n.history==node.reflectionHistory;
                });
                check(match!=reference.interfaces.end(),"rough interface has independent branch-history provenance");
                if(match==reference.interfaces.end())continue;
                check(match->surface.object==node.surface.objectId && match->surface.material==node.surface.materialId
                    && match->surface.instance==node.surface.instanceId && match->surface.triangle==node.surface.triangleId,
                    "rough facets preserve exact authored geometric interface owners");
                for(unsigned axis=0;axis<3;++axis)check(close(node.facetNormal[axis],match->facet[axis],2e-4),
                    "continued rough branch facets match the independent binary64 normal schedule");
            }
            for(const auto& deposit:actual.deposits)
            {
                const auto match=std::find_if(reference.arrivals.begin(),reference.arrivals.end(),[&](const auto& a) {
                    return a.interfaces==deposit.interfaces && a.history==deposit.reflectionHistory;
                });
                check(match!=reference.arrivals.end(),"rough receiver deposit retains independent finite branch provenance");
                if(match!=reference.arrivals.end())for(unsigned axis=0;axis<3;++axis)
                    check(close(deposit.receiver.position[axis],match->receiver.position[axis],3e-4),
                        "rough receiver position agrees with independent authored-triangle transport");
                const auto smoothMatch=std::find_if(smoothPath.deposits.begin(),smoothPath.deposits.end(),[&](const auto& d) {
                    return d.interfaces==deposit.interfaces && d.reflectionHistory==deposit.reflectionHistory;
                });
                if(smoothMatch!=smoothPath.deposits.end() && deposit.fraction>.5)
                    maximumSpread=std::max(maximumSpread,std::sqrt(double(gc::dot(gc::sub(deposit.receiver.position,smoothMatch->receiver.position),
                        gc::sub(deposit.receiver.position,smoothMatch->receiver.position)))));
            }
        }
        check(close(sum,smooth.frame.work[e*8+band].incidentPower,2e-7),
            "sixteen rough strata divide Li times cosine times area without creating source power");
    }
    check(maximumSpread>.005 && maximumMasked>1e-8,
        "finite roughness changes an actual receiver footprint and records nonzero masking");
    std::string branchIdentity;
    check(gc::validateBranchQueries(fixture.prepared,fixture.frame,*fixture.policy,branchIdentity,error),error);
    gc::EnergyLedger referencePower;check(gc::referencePower(fixture.prepared,fixture.frame,referencePower,error),error);
    const auto independent=gt::powerOracle(fixture);check(independent.complete,"independent complete rough band-power ledger resolves");
    for(unsigned band=0;band<8;++band)
    {
        check(close(referencePower.launched[band],independent.energy.launched[band])
            && close(referencePower.deposited[band],independent.energy.deposited[band])
            && close(referencePower.absorbed[band],independent.energy.absorbed[band])
            && close(referencePower.masked[band],independent.energy.masked[band],4e-6),
            "rough energy applies each authored receiver band once with separate absorption and masking");
    }
    gc::NativeReadback synthetic;const bool made=gt::syntheticReadback(fixture,synthetic,error);check(made,error);
    if(made)
    {
        gc::EnergyLedger ledger;check(gc::validateNativeReadback(fixture.prepared,fixture.frame,synthetic,ledger,error),error);
        for(unsigned band=0;band<8;++band)check(close(ledger.launched[band],ledger.deposited[band]+ledger.absorbed[band]
            +ledger.escaped[band]+ledger.unresolved[band]+ledger.masked[band],2e-6),
            "synthetic rough terminal gate conserves each band without claiming native pixels");
        for(unsigned mode=0;mode<4;++mode)
        {
            auto bad=synthetic;auto& path=bad.paths[0];
            if(mode==0)path.roughTransportVersion=0;
            if(mode==1)path.interfaceSample=1;
            if(mode==2) {const auto extra=path.launchedPower*.01f;path.maskedPower+=extra;path.escapedPower-=extra;}
            if(mode==3) {const auto extra=path.launchedPower*.01f;path.unresolvedPower+=extra;path.escapedPower-=extra;}
            gc::EnergyLedger sentinel;sentinel.masked[0]=77;
            check(!gc::validateNativeReadback(fixture.prepared,fixture.frame,bad,sentinel,error) && sentinel.masked[0]==77,
                "missing rough receipt, wrong stratum or excess masked and unresolved tail refuse atomically");
        }
    }
    for(unsigned mode=0;mode<3;++mode)
    {
        auto bad=fixture.frame;
        if(mode==0)bad.roughness=.1f;
        if(mode==1)bad.interfaceSamples=1;
        if(mode==2)bad.work[0].interfaceSample=1;
        bad.identity=gc::frameIdentity(fixture.prepared,bad);
        check(!gc::validateTables(fixture.prepared,bad,error),"mutated rough owner or schedule cannot survive a recomputed frame identity");
    }
    for(float roughness:{.08f,.35f})
    {
        auto limited=gt::roughFixture(roughness);const bool admitted=limited.prepare(error,consumer);check(admitted,error);if(!admitted)continue;
        bool refused=false,physicalDeficit=false;
        for(const auto& work:limited.frame.work)
        {
            gc::BranchReceipt path;
            if(!gc::traceBranches(limited.prepared,limited.frame,limited.frame.entries[work.entry],work.band,*limited.policy,path,error,work.interfaceSample))
            {
                refused=true;physicalDeficit=path.maskedFraction>gc::kMaximumUnresolvedFraction
                    && error=="Glass caustic rough interfaces require microfacet multiple scattering beyond the admitted energy error bound";
                check(close(path.depositedFraction+path.escapedFraction+path.unresolvedFraction+path.maskedFraction,1,1e-12),
                    "out-of-policy rough transport still identifies all physical and unfinished power");
                break;
            }
        }
        check(refused && physicalDeficit,"large single-scatter deficit explicitly refuses pending microfacet multiple scattering");
    }
    gc::ConsumerContract contract{arbitgpu::NativeFixtureMaterialBackend::OpenGl,gc::kTableVersion,3,gc::kBranchPolicyVersion,
        true,true,true,gc::kScheduleVersion,true,true,true,gc::kDepositionVersion};
    check(gc::consumerCanRender(contract,contract.backend) && !gc::consumerCanRender(contract,contract.backend,.05f),
        "released smooth consumer cannot silently admit rough interface transport");
    contract.roughTransportVersion=gc::kRoughTransportVersion;
    check(gc::consumerCanRender(contract,contract.backend,.05f),"rough consumer names its exact new transport version");
}
void multipleScatterPhysics()
{
    const gc::Point normal{0,0,1};
    const gc::MultipleScatterPolicy policy;
    check(gc::validMultipleScatterPolicy(policy) && policy.maximumEvents==32 && policy.maximumExits==16
        && close(policy.maximumResidual,1e-4/128,1e-12),"private Smith walk reserves an explicit finite residual budget");
    for(float roughness:{.08f,.35f,.75f,1.0f})
    {
        const unsigned seed=roughness==.08f?18:0;
        const gc::Point incoming{float(std::sin(.7)),0,float(-std::cos(.7))};
        const auto actual=gc::multipleScatterInterface(incoming,normal,1,1.5f,roughness,seed);
        const auto expected=gt::microfacetReference(gt::doublePoint(incoming),gt::doublePoint(normal),1,1.5,roughness,seed,
            policy.maximumEvents,policy.maximumExits,policy.maximumResidual);
        check(actual.status==gc::MultipleScatterStatus::Complete && expected.status==1,
            "finite private microfacet case meets its interface residual policy");
        check(actual.count==expected.exits.size() && actual.events==expected.events && actual.draws==4*actual.events,
            "event and exit provenance agrees with independent binary64 height walk");
        double power=actual.residual;bool secondary=false;
        for(unsigned i=0;i<actual.count;++i)
        {
            const auto& exit=actual.exits[i];power+=exit.power;
            secondary|=exit.order>1 && exit.power>1e-5;
            check(exit.power>0 && std::isfinite(exit.power) && close(gc::dot(exit.direction,exit.direction),1)
                && (exit.transmitted?exit.direction[2]<0:exit.direction[2]>0),
                "every emitted physical ray exits into its reported medium");
            if(i<expected.exits.size())
            {
                const auto& reference=expected.exits[i];
                check(exit.transmitted==reference.transmitted && exit.order==reference.order
                    && close(exit.power,reference.power,2e-5),"multiple-scatter flux and medium parity match the independent model");
                // Small alpha amplifies float32 height-CDF cancellation in a
                // secondary ray carrying roughly 1e-5 of incident power.
                for(unsigned a=0;a<3;++a)check(close(exit.direction[a],reference.direction[a],roughness==.08f?2e-3:2e-4),
                    "actual exiting direction agrees with binary64 angular reference");
            }
        }
        check(secondary,"physical model returns a nonzero ray from a later microfacet event");
        check(close(power,1,3e-6) && actual.residual<=policy.maximumResidual,
            "unit incident flux is retained without masked-power renormalization");
    }
    bool multipleParity=false;
    for(unsigned seed=0;seed<128;++seed)
    {
        const gc::Point incoming{float(std::sin(1.2)),0,float(-std::cos(1.2))};
        const auto result=gc::multipleScatterInterface(incoming,normal,1.5f,1,.75f,seed);
        double accounted=result.residual;
        for(unsigned i=0;i<result.count;++i) {
            accounted+=result.exits[i].power;
            if(i && result.exits[i].transmitted!=result.exits[0].transmitted)multipleParity=true;
        }
        check(close(accounted,1,3e-6),"TIR, medium crossings and finite refusal retain all power");
        check(result.events<=32 && result.count<=16 && result.draws<=128,
            "private walk respects fixed event, exit and draw bounds");
    }
    check(multipleParity,"dielectric walks can emit rays on both sides after multiple transmissions");
    const gc::Point grazing{float(std::sin(1.3)),0,float(-std::cos(1.3))};
    auto limited=policy;limited.maximumEvents=1;
    const auto truncated=gc::multipleScatterInterface(grazing,normal,1,1.5f,.75f,7,limited);
    check(truncated.status==gc::MultipleScatterStatus::Unresolved && truncated.residual>.1f
        && truncated.count==1 && close(truncated.residual+truncated.exits[0].power,1,3e-6),
        "event truncation leaves a real excessive unresolved tail rather than reporting completion");
    limited=policy;limited.maximumExits=1;
    const auto capacity=gc::multipleScatterInterface({float(std::sin(.7)),0,float(-std::cos(.7))},normal,1,1.5f,.75f,0,limited);
    check(capacity.status==gc::MultipleScatterStatus::Unresolved && capacity.count==1
        && close(capacity.residual+capacity.exits[0].power,1,3e-6),
        "exit capacity retains unrecorded physical power in residual");
    const auto matched=gc::multipleScatterInterface(grazing,normal,1.5f,1.5f,.75f,7);
    check(matched.status==gc::MultipleScatterStatus::Complete && matched.count==1 && matched.events==0
        && matched.draws==0 && matched.residual==0 && matched.exits[0].direction==grazing
        && matched.exits[0].transmitted && matched.exits[0].power==1,
        "equal-index media preserve exact unit-flux straight-through geometry");
    for(unsigned mode=0;mode<5;++mode)
    {
        auto bad=policy;
        if(mode==0)bad.version=1;
        if(mode==1)bad.maximumEvents=33;
        if(mode==2)bad.maximumExits=17;
        if(mode==3)bad.maximumResidual=2e-4f;
        if(mode==4)bad.maximumResidual=std::numeric_limits<float>::quiet_NaN();
        const auto rejected=gc::multipleScatterInterface({0,0,-1},normal,1,1.5f,.35f,0,bad);
        check(!gc::validMultipleScatterPolicy(bad) && rejected.status==gc::MultipleScatterStatus::Invalid
            && rejected.count==0 && rejected.residual==1,"unsupported private model policy refuses without power loss");
    }
    check(gc::multipleScatterInterface({0,0,-1},normal,1,1.5f,0,0).status==gc::MultipleScatterStatus::Invalid,
        "private rough walk cannot replace the released zero-roughness path");
    std::uint64_t combined=123;
    check(gc::multipleScatterWorkWithinBounds(1000,10,policy,combined) && combined==2600,
        "microfacet event and draw work consumes the existing query-work authority");
    check(!gc::multipleScatterWorkWithinBounds(gc::kMaximumQueryWork-159,1,policy,combined)
        && !gc::multipleScatterWorkWithinBounds(0,std::numeric_limits<std::uint64_t>::max(),policy,combined),
        "extra optical work refuses before overflow or the unchanged 128 Mi cap");
    for(unsigned dimension=0;dimension<128;++dimension) {
        const auto u=gc::multipleScatterRandom(gc::multipleScatterSeed(3,5,7,11),dimension);
        check(u>0 && u<1,"stable path samples avoid singular CDF endpoints");
    }
    check(gc::multipleScatterSeed(3,5,7,11)==1799114855u
        && gc::multipleScatterSeed(3,5,7,11)!=gc::multipleScatterSeed(3,5,7,12),
        "discrete ancestor path identity determines the stable microfacet sample stream");
    check(gc::multipleScatterChildHistory(11,2,false)!=gc::multipleScatterChildHistory(11,3,false)
        && gc::multipleScatterChildHistory(11,2,false)!=gc::multipleScatterChildHistory(11,2,true),
        "different microfacet exit orders and media retain different sample streams");
    check(gc::multipleScatterEntryKey(3,5,7,11)==1799114855u
        && gc::multipleScatterEntryKey(3,5,7,11)!=gc::multipleScatterEntryKey(3,5,7,12),
        "owned triangle and barycentric ordinal retain a key independent of frame entry offsets");
    const auto shader=gc::nativeMultipleScatterMathSource();
    check(shader.find("#define GLASS_MS_VERSION 2")!=std::string::npos
        && shader.find("int causticMsWalk(")!=std::string::npos
        && shader.find("uint causticMsSeed(")!=std::string::npos
        && shader.find("glassMsWalk")==std::string::npos,
        "private native source exposes the same bounded versioned angular contract");
    gc::ConsumerContract consumer;
    consumer.backend=arbitgpu::NativeFixtureMaterialBackend::OpenGl;
    consumer.tableVersion=gc::kTableVersion;consumer.closestHitPayloadVersion=3;consumer.branchPolicyVersion=gc::kBranchPolicyVersion;
    consumer.authoredSourceBinding=consumer.nativeDeposition=consumer.wholeFrameReadback=true;
    consumer.scheduleVersion=gc::kScheduleVersion;consumer.cameraTriangleVisibility=consumer.perBandEnergyReadback=consumer.productionCieResolve=true;
    consumer.depositionVersion=gc::kDepositionVersion;consumer.roughTransportVersion=gc::kPrivateMultipleScatterVersion;
    check(!gc::consumerCanRender(consumer,consumer.backend,.35f),
        "private version 2 cannot admit a frame before both native consumer receipts exist");
}
void multipleScatterReceipts()
{
    std::string error;
    const arbitgpu::smoothglass::RoughNativeConsumer roughConsumer{arbitgpu::NativeFixtureMaterialBackend::OpenGl,true,true,true};
    // Matched media is an exact physical v1/v2 common control: both kernels
    // transmit the same ray with unit flux and zero microfacet evaluations.
    auto fixture=gt::roughFixture(.35f);fixture.referenceIor=1;fixture.dispersion=0;
    fixture.program->parameters.transmissionIorClearcoat[1]=1;
    fixture.program->parameters.transmissionIorClearcoat[3]=0;
    const bool prepared=fixture.prepare(error,roughConsumer);check(prepared,error);if(!prepared)return;
    const auto legacy=fixture.frame;
    gc::NativeReadback synthetic;
    const bool made=gt::syntheticReadback(fixture,synthetic,error);check(made,error);if(!made)return;
    bool nonzeroLegacyEntry=false;
    for(const auto& original:synthetic.paths)
    {
        nonzeroLegacyEntry|=original.entryHit.instanceId!=0 || original.entryHit.triangleId!=0;
        const auto legacyPacked=gc::pathExtensionTexels(original,gc::PathExtensionLayout::OpenGl);
        const gc::Texel actualIndices{float(original.entryHit.instanceId&65535),float(original.entryHit.instanceId>>16),
            float(original.entryHit.triangleId&65535),float(original.entryHit.triangleId>>16)};
        auto retained=original;retained.entryHit.instanceId=retained.entryHit.triangleId=UINT32_MAX;
        check(legacyPacked[0]==actualIndices
            && gc::decodePathExtensions(legacyPacked[0],legacyPacked[1],gc::PathExtensionLayout::OpenGl,retained)
            && retained.entryHit.instanceId==original.entryHit.instanceId && retained.entryHit.triangleId==original.entryHit.triangleId,
            "actual legacy OpenGL pathHit texel14 retains queried entry instance and triangle words");
    }
    check(nonzeroLegacyEntry,"legacy packing fixture includes actual nonzero entry triangle IDs");
    gc::MultipleScatterPolicy policy;gc::Frame frame;
    const bool built=gc::buildMultipleScatterFrame(fixture.prepared,legacy.instances,nullptr,legacy.width,legacy.height,
        legacy.scheduling,policy,frame,error);check(built,error);if(!built)return;
    check(frame.work.size()==legacy.work.size() && frame.entries.size()==legacy.entries.size()
        && frame.interfaceSamples==legacy.interfaceSamples && frame.bounds.queryWork==legacy.bounds.queryWork,
        "private v2 retains one scheduled entry/sample/band work item and the original closest-hit reservation");
    check(frame.identity!=legacy.identity && frame.bounds.combinedWork==frame.bounds.queryWork+frame.bounds.microfacetWork
        && frame.bounds.microfacetWork==frame.work.size()*fixture.policy->description().maximumVisits*policy.maximumEvents*5
        && frame.bounds.microfacetTemporaryBytes==frame.work.size()*(16*gc::kMultipleScatterPendingExtensionBytes+16*20)
        && frame.bounds.transientBytes>legacy.bounds.transientBytes && frame.bounds.combinedWork<=gc::kMaximumQueryWork,
        "private policy, extra optical work and exact ancestry/exit storage are admitted within unchanged bounds");
    for(std::size_t w=0;w<frame.work.size();++w)
    {
        const auto& entry=frame.entries[frame.work[w].entry];
        check(entry.quadrature<16 && frame.workTexels[w*gc::kWorkStride+3][2]==entry.quadrature
            && legacy.workTexels[w*gc::kWorkStride+3][2]==legacy.work[w].entry,
            "only v2 replaces the unused upload entry offset with a stable per-triangle quadrature ordinal");
        auto& path=synthetic.paths[w];path.multipleScatterVersion=2;path.quadrature=entry.quadrature;
    }
    for(auto& deposit:synthetic.deposits) {
        check(deposit.reflectionHistory==0,"matched physical control has no macro reflection bits");
        for(unsigned i=0;i<deposit.interfaces;++i)deposit.ancestry[i]=64;
    }
    synthetic.frameIdentity=frame.identity;
    gc::EnergyLedger ledger;
    check(gc::validateNativeReadback(fixture.prepared,frame,synthetic,ledger,error),
        "synthetic matched-media v2 schema conserves each band without claiming native execution");
    auto overwritten=synthetic;auto& first=overwritten.paths[0];
    const auto nativeEntry=first.entryHit;
    const auto firstPacked=gc::pathExtensionTexels(first,gc::PathExtensionLayout::OpenGl);
    check(gc::decodePathExtensions(firstPacked[0],firstPacked[1],gc::PathExtensionLayout::OpenGl,first)
        && first.entryHit.instanceId==UINT32_MAX && first.entryHit.triangleId==UINT32_MAX,
        "v2 OpenGL metadata is never decoded as native entry indices");
    check(!gc::validateNativeReadback(fixture.prepared,frame,overwritten,ledger,error),
        "overwritten entry indices require explicit frozen-work and world-hit binding");
    check(gc::bindOpenGlMultipleScatterEntry(frame,first,error)
        && first.entryHit.instanceId==nativeEntry.instanceId && first.entryHit.triangleId==nativeEntry.triangleId
        && gc::validateNativeReadback(fixture.prepared,frame,overwritten,ledger,error),
        "v2 schema binds only the overwritten indices after validating the retained queried world hit");
    for(unsigned mode=0;mode<7;++mode)
    {
        auto invalid=synthetic.paths[0];invalid.entryHit.instanceId=invalid.entryHit.triangleId=UINT32_MAX;
        if(mode==0)invalid.complete=false;
        if(mode==1)invalid.entryHit.objectId^=1;
        if(mode==2)invalid.entryHit.materialId^=1;
        if(mode==3)invalid.entryHit.position[0]+=1;
        if(mode==4)invalid.entryHit.outwardNormal={0,0,-1};
        if(mode==5)invalid.entryHit.distance=1;
        if(mode==6)invalid.quadrature=16;
        check(!gc::bindOpenGlMultipleScatterEntry(frame,invalid,error) && invalid.entryHit.instanceId==UINT32_MAX,
            "frozen work cannot manufacture entry proof from incomplete, stale or off-triangle native hit fields");
    }
    gc::BranchReceipt ignored;
    check(!gc::traceBranches(fixture.prepared,frame,frame.entries[0],0,*fixture.policy,ignored,error),
        "private requests cannot silently run the v1 diagnostic macro kernel");
    for(unsigned mode=0;mode<14;++mode)
    {
        auto bad=synthetic;
        if(mode==0)bad.paths[0].multipleScatterVersion=0;
        if(mode==1)bad.paths[0].multipleScatterVersion=3;
        if(mode==2)bad.paths[0].quadrature=16;
        if(mode==3)bad.paths[0].microEvents=2049;
        if(mode==4)bad.paths[0].microDraws=1;
        if(mode==5)bad.paths[0].microResidualPower=std::numeric_limits<float>::infinity();
        if(mode==6)bad.paths[0].microResidualPower=1;
        if(mode==7)bad.paths[0].maskedPower=1e-9f;
        if(mode==8)bad.deposits[0].ancestry[0]|=128;
        if(mode==9)bad.deposits[0].ancestry[0]=65;
        if(mode==10)bad.deposits[0].ancestry[bad.deposits[0].interfaces]=1;
        if(mode==11)bad.deposits[0].reflectionHistory=1;
        if(mode==12)bad.deposits.push_back(bad.deposits[0]);
        if(mode==13)bad.paths[0].unresolvedPower=bad.paths[0].launchedPower;
        gc::EnergyLedger untouched;untouched.launched[0]=123;
        check(!gc::validateNativeReadback(fixture.prepared,frame,bad,untouched,error) && untouched.launched[0]==123,
            "stale version, ancestry, optical work or physical tail refuses the whole receipt atomically");
    }
    auto changed=frame;changed.entries[0].quadrature=16;changed.identity=gc::frameIdentity(fixture.prepared,changed);
    check(!gc::validateTables(fixture.prepared,changed,error),"recomputed identity cannot legitimize a changed quadrature owner");
    changed=frame;--changed.bounds.microfacetWork;changed.identity=gc::frameIdentity(fixture.prepared,changed);
    check(!gc::validateTables(fixture.prepared,changed,error),"recomputed identity cannot reduce admitted optical work");
    gc::NativePathReceipt path;path.roughTransportVersion=1;path.interfaceSample=7;path.multipleScatterVersion=2;
    path.microEvents=3;path.microDraws=12;path.microResidualPower=1e-7f;path.quadrature=11;
    const auto packed=gc::pathExtensionTexels(path,gc::PathExtensionLayout::OpenGl);gc::NativePathReceipt decoded;
    check(gc::decodePathExtensions(packed[0],packed[1],gc::PathExtensionLayout::OpenGl,decoded) && decoded.microEvents==3 && decoded.microDraws==12
        && decoded.microResidualPower==path.microResidualPower && decoded.quadrature==11 && decoded.multipleScatterVersion==2,
        "v2 path texels14/15 preserve exact optical counts and separate weighted residual power");
    for(unsigned mode=0;mode<7;++mode)
    {
        auto bad=packed;
        if(mode==0)bad[0][0]=3.5f;
        if(mode==1)bad[0][1]=11;
        if(mode==2)bad[0][2]=std::numeric_limits<float>::quiet_NaN();
        if(mode==3)bad[0][3]=16;
        if(mode==4)bad[1][3]=1;
        if(mode==5)bad[1][0]=1e-8f;
        if(mode==6)bad[1][1]=0;
        decoded.microEvents=123;
        check(!gc::decodePathExtensions(bad[0],bad[1],gc::PathExtensionLayout::OpenGl,decoded) && decoded.microEvents==123,
            "non-integral or unsupported packed v2 receipts cannot reach a typed path");
    }
    gc::NativeDeposit deposit;deposit.interfaces=2;deposit.ancestry[0]=65;deposit.ancestry[1]=66;
    unsigned minimum=0;
    check(gc::validMacroAncestry(deposit,policy,false,minimum) && minimum==3,
        "exact transmitted event-order tokens retain a complete outside-to-outside macro chain");
    deposit.ancestry[0]=1;deposit.ancestry[1]=2;deposit.reflectionHistory=3;
    check(gc::validMacroAncestry(deposit,policy,false,minimum),"exact reflection tokens cross-check the original bitset");
    deposit.reflectionHistory=0;
    check(!gc::validMacroAncestry(deposit,policy,false,minimum),"sampling hashes cannot replace macro reflection evidence");
    auto ancestry=gc::macroAncestryTexels(deposit.ancestry);std::array<std::uint8_t,32> tokens{};
    check(gc::decodeMacroAncestry(ancestry,2,tokens) && tokens==deposit.ancestry,
        "deposit texels22..29 preserve all32 exact one-byte tokens");
    ancestry[0][0]=1.5f;check(!gc::decodeMacroAncestry(ancestry,2,tokens),"fractional ancestry tokens are refused");
    ancestry[0][0]=128;check(!gc::decodeMacroAncestry(ancestry,2,tokens),"reserved ancestry bit7 is refused");
    check(gc::decodeMacroAncestry({},0,tokens) && tokens==std::array<std::uint8_t,32>{},
        "legacy reserved deposit suffix retains all-zero bytes");
    gc::NativePathReceipt smooth,rough;rough.roughTransportVersion=1;rough.interfaceSample=7;rough.maskedPower=1e-6f;
    check(gc::pathExtensionTexels(smooth,gc::PathExtensionLayout::OpenGl)==std::array<gc::Texel,2>{}
        && gc::pathExtensionTexels(rough,gc::PathExtensionLayout::OpenGl)==std::array<gc::Texel,2>{{gc::Texel{},gc::Texel{1e-6f,1,7,0}}},
        "smooth v0 and released rough v1 path extension bytes remain unchanged");
    for(auto legacyPath:{smooth,rough})
    {
        legacyPath.entryHit.instanceId=0x12345678u;legacyPath.entryHit.triangleId=0xabcdef12u;
        const auto gl=gc::pathExtensionTexels(legacyPath,gc::PathExtensionLayout::OpenGl);
        const auto metal=gc::pathExtensionTexels(legacyPath,gc::PathExtensionLayout::Metal);
        auto parsed=legacyPath;
        check(gl[0]==gc::Texel{0x5678,0x1234,0xef12,0xabcd}
            && gc::decodePathExtensions(gl[0],gl[1],gc::PathExtensionLayout::OpenGl,parsed)
            && parsed.entryHit.instanceId==legacyPath.entryHit.instanceId && parsed.entryHit.triangleId==legacyPath.entryHit.triangleId,
            "v0/v1 OpenGL split words preserve all32 entry ID bits instead of requiring zero metadata");
        check(metal[0]==gc::Texel{} && gc::decodePathExtensions(metal[0],metal[1],gc::PathExtensionLayout::Metal,parsed)
            && parsed.entryHit.instanceId==legacyPath.entryHit.instanceId && parsed.entryHit.triangleId==legacyPath.entryHit.triangleId,
            "v0/v1 Metal padding parsing preserves existing typed entry IDs");
        auto invalid=gl;invalid[0][0]=.5f;
        check(!gc::decodePathExtensions(invalid[0],invalid[1],gc::PathExtensionLayout::OpenGl,parsed),
            "non-integral legacy entry words still refuse exact-ID conversion");
        check(!gc::decodePathExtensions(gl[0],metal[1],gc::PathExtensionLayout::Metal,parsed),
            "OpenGL entry words cannot be mistaken for zero Metal padding");
    }
    auto legacyChanged=legacy;for(auto& entry:legacyChanged.entries)entry.quadrature=15;
    check(gc::frameIdentity(fixture.prepared,legacyChanged)==legacy.identity,
        "new quadrature fields add no bytes to released v1 identities");
    gc::ConsumerContract consumer{arbitgpu::NativeFixtureMaterialBackend::OpenGl,gc::kTableVersion,3,gc::kBranchPolicyVersion,
        true,true,true,gc::kScheduleVersion,true,true,true,gc::kDepositionVersion,gc::kRoughTransportVersion};
    check(gc::consumerCanRender(consumer,consumer.backend,.35f) && !gc::consumerCanRender(consumer,consumer.backend,.35f,2),
        "existing v1 consumer defaults remain stable and explicitly refuse private v2 requests");
    consumer.multipleScatterVersion=2;
    check(gc::consumerCanRender(consumer,consumer.backend,.35f,2) && !gc::consumerCanRender(consumer,consumer.backend,.35f,3)
        && !gc::consumerCanRender(consumer,consumer.backend,0,2),"exact v2 advertisement binds only supported rough requests");
}
void geometry()
{
    auto fixture=gt::fixture();std::string error;
    const bool prepared=fixture.prepare(error);check(prepared,error);if(!prepared)return;
    const auto expected=gt::oracle(fixture);
    check(expected.complete,"independent eight-band prism quadrature completes");
    check(std::abs(expected.landingMean[0][0]-expected.landingMean[7][0])>.001,"prism separates actual receiver landing centroids by wavelength");
    for(unsigned b=0;b<8;++b)
    {
        check(expected.receiverPower[b]>0 && expected.outgoingPower[b]<=expected.receiverPower[b],"ordinary diffuse receiver retains bounded authored band power");
        check(close(expected.incident[b],expected.reflected[b]+expected.escaped[b]+expected.receiverPower[b],1e-10),"independent lossless prism conserves all incident power");
        check(close(expected.outgoingPower[b],expected.receiverPower[b]*fixture.program->objectPrograms[0]->spectralReflectance->coefficients[b],1e-12),"independent rho is applied exactly once after transport");
        check(fixture.frame.sourceRadiance[b/4][b%4]==fixture.light->diffractionLightingAdmission->description().paths[0].incident.radiance[b],"Li remains independently authored in table");
    }
    for(const auto& triangle:fixture.frame.triangles)if(!triangle.glass)
        check(triangle.reflectance->coefficients==fixture.program->objectPrograms[0]->spectralReflectance->coefficients,"rho table selects exact object program");
    const auto world=gt::worldTriangles(*fixture.scene);
    for(const auto& entry:fixture.frame.entries)for(unsigned band=0;band<8;++band)
    {
        gc::PathReceipt actual;
        const bool completed=gc::validatePath(fixture.prepared,fixture.frame,entry,band,actual,error);check(completed,error);
        const auto oracle=gt::trace(world,gt::doublePoint(entry.position),gt::doublePoint(entry.normal),
            gt::doublePoint(fixture.light->diffractionLightingAdmission->description().paths[0].incident.direction),entry.object,
            fixture.frame.ior[band],20);
        check(oracle.complete && actual.sourceBlocked==oracle.sourceBlocked,"source visibility matches independent authored triangle traversal");
        if(actual.sourceBlocked)continue;
        check(close(actual.reflectedFraction,oracle.reflected) && close(actual.transmittedFraction,oracle.transmitted),"finite forward path preserves Snell/Fresnel energy");
        check(close(actual.reflectedFraction+actual.transmittedFraction,1),"complete path conserves power including reflected ledger");
        if(actual.receiverHit)
        {
            check(oracle.receiver.found && actual.receiver.objectId==oracle.receiver.object && actual.receiver.materialId==oracle.receiver.material
                && actual.receiver.instanceId==oracle.receiver.instance && actual.receiver.triangleId==oracle.receiver.triangle,"receiver hit retains exact object/material/triangle identity");
            for(unsigned a=0;a<3;++a)check(close(actual.receiver.position[a],oracle.receiver.position[a],2e-4),"native-query companion receiver position matches independent binary64 triangle oracle");
        }
    }
    auto zero=gt::fixture(0);const bool zeroPrepared=zero.prepare(error);check(zeroPrepared,error);if(!zeroPrepared)return;
    const auto converged=gt::oracle(zero);
    for(unsigned b=1;b<8;++b)for(unsigned a=0;a<3;++a)
        check(close(converged.landingMean[0][a],converged.landingMean[b][a],1e-12),"zero dispersion converges receiver geometry independently of Li/rho");
    for(const auto& entry:zero.frame.entries)
    {
        gc::PathReceipt first;const bool firstComplete=gc::validatePath(zero.prepared,zero.frame,entry,0,first,error);check(firstComplete,error);
        for(unsigned b=1;b<8;++b)
        {
            gc::PathReceipt next;const bool nextComplete=gc::validatePath(zero.prepared,zero.frame,entry,b,next,error);check(nextComplete,error);
            check(first.receiver.position==next.receiver.position && first.transmittedFraction==next.transmittedFraction
                && first.reflectedFraction==next.reflectedFraction,"zero-dispersion query and energy remain bit identical across bands");
        }
    }
    auto lightMoved=gt::fixture();lightMoved.light=gt::source({-.19611613f,0,.98058068f});
    auto glassMoved=gt::fixture();glassMoved.scene->objects[0].transform.translation.x=.3f;
    auto receiverMoved=gt::fixture();receiverMoved.scene->objects[1].transform.translation.z=-1;
    for(auto* moved:{&lightMoved,&glassMoved,&receiverMoved})
    {
        const bool movedPrepared=moved->prepare(error);check(movedPrepared,error);const auto changed=gt::oracle(*moved);
        check(changed.complete && std::abs(changed.landingMean[3][0]-expected.landingMean[3][0])>.005,"authored source direction/glass/receiver movement changes actual landing geometry");
    }
    auto sourceBlock=gt::fixture(.004f,1),exitBlock=gt::fixture(.004f,2);
    const bool sourcePrepared=sourceBlock.prepare(error);check(sourcePrepared,error);
    const bool exitPrepared=exitBlock.prepare(error);check(exitPrepared,error);
    for(auto* blocked:{&sourceBlock,&exitBlock})
    {
        const auto dark=gt::oracle(*blocked);check(dark.complete,"two-leg blocker oracle completes");
        for(double value:dark.receiverPower)check(value==0,"authored triangles on either leg remove main receiver caustic power");
    }
    auto changed=gt::fixture();
    auto li=changed.light->diffractionLightingAdmission->description().paths[0].incident.radiance;li[2]*=2;
    changed.light=gt::source({.19611613f,0,.98058068f},li);
    auto receiver=std::make_shared<gt::Program>(*changed.program->objectPrograms[0]);
    receiver->spectralReflectance->coefficients[4]=0;changed.program->objectPrograms[0]=receiver;
    changed.scene->lights[0].color={9,1,3};
    const auto independent=gt::oracle(changed);
    for(unsigned b=0;b<8;++b)check(close(independent.outgoingPower[b],expected.outgoingPower[b]*(b==2?2:b==4?0:1),1e-10),"source Li/rho perturbations affect only their band; unrelated RGB is never a spectrum");
}

void branches()
{
    auto fixture=gt::reflectionFixture();std::string error;
    const bool prepared=fixture.prepare(error);check(prepared,error);if(!prepared)return;
    gc::BranchPolicyDescription description;description.version=gc::kBranchPolicyVersion;
    description.maximumUnresolvedFraction=1e-4f;description.minimumBranchFraction=1e-5f;
    const auto policy=gc::AdmittedBranchPolicy::admit(description,error);
    check(bool(policy),error);if(!policy)return;
    check(!gc::AdmittedBranchPolicy::admit({},error),"absent explicit error policy cannot silently enable reflected caustics");
    for(unsigned mode=0;mode<4;++mode)
    {
        auto invalid=description;
        if(mode==0)invalid.maximumUnresolvedFraction=std::numeric_limits<float>::quiet_NaN();
        if(mode==1)invalid.minimumBranchFraction=2*description.maximumUnresolvedFraction;
        if(mode==2)invalid.maximumVisits=gc::kMaximumBranchVisits+1;
        if(mode==3)invalid.maximumUnresolvedFraction=2*gc::kMaximumUnresolvedFraction;
        check(!gc::AdmittedBranchPolicy::admit(invalid,error),"unbounded or invalid policy cannot admit discarded branch power");
    }
    const auto reflectedObject=fixture.scene->objects[2].id.value;
    const auto fraction=[&](const auto& receipt,std::uint32_t object) {
        double sum=0;for(const auto& deposit:receipt.deposits)if(deposit.receiver.objectId==object)sum+=deposit.fraction;return sum;
    };
    std::optional<gc::EntrySample> chosen;
    std::size_t reflectedPaths=0,tirEvents=0;bool allComplete=true;
    for(const auto& entry:fixture.frame.entries)
    {
        if(entry.normal[2]>.8f && std::abs(entry.position[0]+1.0f/6)<.01f)chosen=entry;
        for(unsigned band=0;band<8;++band)
        {
            gc::BranchReceipt actual;
            const bool completed=gc::traceBranches(fixture.prepared,fixture.frame,entry,band,*policy,actual,error);
            allComplete=allComplete && completed;
            const auto expected=gt::branchOracle(fixture,entry,band);
            check(expected.valid && actual.sourceBlocked==expected.sourceBlocked,"reflected branch oracle independently validates source and authored triangles");
            if(actual.sourceBlocked)continue;
            tirEvents+=actual.tirEvents;
            check(close(actual.depositedFraction+actual.escapedFraction+actual.unresolvedFraction,1,1e-12),"all reflected/transmitted power remains deposited, escaped or explicitly unresolved");
            check(close(expected.deposited+expected.escaped+expected.unresolved,1,1e-12),"independent high-precision branch ledger conserves incident power");
            check(completed==actual.withinErrorBound && completed==(actual.unresolvedFraction<=description.maximumUnresolvedFraction),"finite prefix is accepted only within explicit unresolved-power bound");
            if(!completed)check(error=="Glass caustic unresolved branch power exceeds its admitted energy error bound","out-of-bound pending power has an honest diagnostic");
            for(const auto& node:actual.interfaceLedger)
            {
                check(close(node.incidentFraction,node.reflectedFraction+node.transmittedFraction,1e-12),"each branch interface conserves its full incoming flux");
                const auto match=std::find_if(expected.interfaces.begin(),expected.interfaces.end(),[&](const auto& reference) {
                    return node.reflectionHistory==reference.history && node.depth==reference.depth;
                });
                check(match!=expected.interfaces.end(),"each bounded interface has independent path-history provenance");
                if(match==expected.interfaces.end())continue;
                check(node.surface.objectId==match->surface.object && node.surface.materialId==match->surface.material
                    && node.surface.instanceId==match->surface.instance && node.surface.triangleId==match->surface.triangle,
                    "branch interface keeps exact authored glass triangle identity");
                check(node.tir==match->tir && close(node.reflectedFraction,match->reflected,2e-5)
                    && close(node.transmittedFraction,match->transmitted,2e-5),"branch interface Snell/Fresnel fractions match independent binary64 traversal");
            }
            if(completed)
            {
                const double difference=std::abs(actual.depositedFraction-expected.deposited);
                check(difference<=actual.unresolvedFraction+expected.unresolved+2e-5,"reflected receiver sum agrees with deeper independent integration and separate float error allowance");
                for(const auto& deposit:actual.deposits)
                {
                    const auto match=std::find_if(expected.arrivals.begin(),expected.arrivals.end(),[&](const auto& reference) {
                        return deposit.reflectionHistory==reference.history && deposit.interfaces==reference.interfaces;
                    });
                    check(match!=expected.arrivals.end(),"each bounded receiver branch has independent path-history provenance");
                    if(match==expected.arrivals.end())continue;
                    check(deposit.receiver.objectId==match->receiver.object && deposit.receiver.materialId==match->receiver.material
                        && deposit.receiver.instanceId==match->receiver.instance && deposit.receiver.triangleId==match->receiver.triangle,
                        "reflected receiver branch retains exact object/material/instance/triangle identity");
                    check(close(deposit.fraction,match->fraction,2e-5),"each reflected/transmitted deposited fraction matches independent Fresnel branch power");
                    if(deposit.reflections && deposit.receiver.objectId==reflectedObject)++reflectedPaths;
                }
            }
        }
    }
    std::string admittedIdentity;
    check(gc::validateBranchQueries(fixture.prepared,fixture.frame,*policy,admittedIdentity,error)==allComplete,
        "whole-frame gate agrees with every independent bounded branch admission");
    check(allComplete?admittedIdentity==fixture.frame.identity+policy->identity():admittedIdentity.empty(),
        "only admitted complete frame binds its exact branch policy identity");
    check(reflectedPaths>0,"actual offset receiver captures bounded reflected caustic paths");
    check(tirEvents>0,"authored prism exercises TIR without labeling its retained flux absorption");
    check(bool(chosen),"reflection blocker control has an independently visible entry sample");if(!chosen)return;
    gc::BranchReceipt baseline;const bool baselineComplete=gc::traceBranches(fixture.prepared,fixture.frame,*chosen,3,*policy,baseline,error);
    check(baselineComplete,error);check(fraction(baseline,reflectedObject)>0,"selected source-visible entry reaches reflected receiver");
    auto sourceBlock=gt::reflectionFixture(1),exitBlock=gt::reflectionFixture(2);
    for(auto* blocked:{&sourceBlock,&exitBlock})
    {
        const bool ready=blocked->prepare(error);check(ready,error);if(!ready)continue;
        gc::BranchReceipt receipt;const bool completed=gc::traceBranches(blocked->prepared,blocked->frame,*chosen,3,*policy,receipt,error);
        check(completed,error);
        const auto oracle=gt::branchOracle(*blocked,*chosen,3);
        check(oracle.valid && receipt.sourceBlocked==oracle.sourceBlocked,"both-leg blockers agree with independent branch traversal");
        check(fraction(receipt,reflectedObject)==0,"actual blocker removes reflected receiver path");
        if(blocked==&sourceBlock)check(receipt.sourceBlocked,"source-leg blocker prevents launch of all branches");
        else check(!receipt.sourceBlocked && receipt.depositedFraction>0,"reflected-leg blocker leaves incoming light visible and captures its flux as an actual receiver");
    }
    auto moved=gt::reflectionFixture();moved.scene->objects[0].transform.translation.x=.5f;
    auto changedLight=gt::reflectionFixture();changedLight.light=gt::source({-.19611613f,0,.98058068f});
    for(auto* changed:{&moved,&changedLight})
    {
        const bool ready=changed->prepare(error);check(ready,error);if(!ready)continue;
        auto entry=*chosen;if(changed==&moved)entry.position[0]+=.5f;
        gc::BranchReceipt receipt;const bool completed=gc::traceBranches(changed->prepared,changed->frame,entry,3,*policy,receipt,error);
        check(completed,error);const auto oracle=gt::branchOracle(*changed,entry,3);
        double reflected=0;for(const auto& arrival:oracle.arrivals)if(arrival.receiver.object==reflectedObject)reflected+=arrival.fraction;
        check(close(fraction(receipt,reflectedObject),reflected,2e-5),"changed glass/source reflected receiver follows independent geometry");
        check(std::abs(fraction(receipt,reflectedObject)-fraction(baseline,reflectedObject))>.001,"moving glass/source changes reflected receiver power rather than a fixed rainbow pattern");
    }
    auto zero=gt::reflectionFixture();zero.dispersion=0;
    zero.program->parameters.transmissionIorClearcoat[3]=0;
    const bool zeroReady=zero.prepare(error);check(zeroReady,error);
    if(zeroReady)for(const auto& entry:zero.frame.entries)
    {
        gc::BranchReceipt first;const bool firstComplete=gc::traceBranches(zero.prepared,zero.frame,entry,0,*policy,first,error);
        for(unsigned band=1;band<8;++band)
        {
            gc::BranchReceipt next;const bool nextComplete=gc::traceBranches(zero.prepared,zero.frame,entry,band,*policy,next,error);
            bool same=first.deposits.size()==next.deposits.size();
            for(std::size_t i=0;same && i<first.deposits.size();++i)
                same=first.deposits[i].receiver.position==next.deposits[i].receiver.position
                    && first.deposits[i].fraction==next.deposits[i].fraction
                    && first.deposits[i].reflectionHistory==next.deposits[i].reflectionHistory;
            check(firstComplete==nextComplete && same && first.depositedFraction==next.depositedFraction
                && first.escapedFraction==next.escapedFraction && first.unresolvedFraction==next.unresolvedFraction,
                "zero dispersion preserves exact branching geometry and all power ledgers across independent bands");
        }
    }
    auto shortDescription=description;shortDescription.maximumVisits=1;
    const auto shortPolicy=gc::AdmittedBranchPolicy::admit(shortDescription,error);
    check(bool(shortPolicy),error);
    if(shortPolicy)
    {
        gc::BranchReceipt receipt;check(!gc::traceBranches(fixture.prepared,fixture.frame,*chosen,3,*shortPolicy,receipt,error),"unfinished reflected/transmitted branch work refuses frame publication");
        check(receipt.unresolvedFraction>description.maximumUnresolvedFraction && receipt.depositedFraction==0
            && close(receipt.escapedFraction+receipt.unresolvedFraction,1,1e-12),"visit truncation retains pending power and never invents absorption");
        check(policy->identity()!=shortPolicy->identity(),"branch limits and explicit error bound belong to exact policy identity");
    }
    auto corrupted=fixture.frame;corrupted.query.texels[std::size_t(corrupted.query.control[0])-1][0]=1;
    corrupted.query.texels[std::size_t(corrupted.query.control[0])-1][1]=0;
    gc::BranchReceipt incomplete;
    check(!gc::traceBranches(fixture.prepared,corrupted,*chosen,3,*policy,incomplete,error),"closest-hit exhaustion refuses branching without publishing cached deposited paths");
}
void ordinarySources()
{
    auto fixture=gt::ordinaryFixture();std::string error;
    const bool ready=fixture.prepare(error);check(ready,error);if(!ready)return;
    check(!fixture.light && !fixture.prepared.sourceOwner && fixture.prepared.sceneLight==fixture.ordinaryLight,
        "ordinary glass source owns its actual selected light without a fabricated diffraction wrapper");
    check(fixture.prepared.branchPolicy && fixture.frame.branchPolicyIdentity==fixture.policy->identity(),
        "ordinary source frame owns its explicitly admitted branch policy identity");
    const auto baseline=gt::oracle(fixture);const auto source=gt::authoredSource(fixture);
    check(baseline.complete,"independent ordinary-source prism integration completes");
    for(unsigned axis=0;axis<3;++axis)check(close(fixture.frame.sourceDirectionTrace[axis],source.direction[axis],1e-7),
        "ordinary source direction matches independent binary64 quaternion-vector rotation");
    for(unsigned band=0;band<8;++band)check(fixture.frame.sourceRadiance[band/4][band%4]==source.radiance[band],
        "ordinary source frame keeps each independently authored Li coefficient");
    auto selected=gt::ordinaryFixture();selected.scene->lightCount=2;
    selected.scene->lights[1]=selected.scene->lights[0];selected.scene->lights[1].id={302};
    selected.scene->lights[0]={};selected.scene->lights[0].id={301};selected.scene->lights[0].kind=gt::SceneLightKind::Environment;
    selected.ordinaryLight={302};
    const bool selectedReady=selected.prepare(error);check(selectedReady,error);
    if(selectedReady)
    {
        check(selected.frame.activeSource->id==selected.ordinaryLight
            && selected.frame.sourceRadiance==fixture.frame.sourceRadiance,"source selection uses exact owning light ID rather than first light or RGB environment");
        gc::Prepared wrong;
        check(!gc::prepare(selected.prepared.glass,gt::SceneLightId{301},*selected.policy,wrong,error),
            "existing RGB light cannot replace the selected authored spectral source");
    }
    auto bands=gt::ordinaryFixture();bands.scene->lights[0].spectralSource.radiance[2]*=2;
    bands.scene->lights[0].color={9,1,3};bands.scene->lights[0].intensity=0;
    const bool bandsReady=bands.prepare(error);check(bandsReady,error);
    if(bandsReady)
    {
        const auto changed=gt::oracle(bands);
        for(unsigned band=0;band<8;++band)check(close(changed.receiverPower[band],baseline.receiverPower[band]*(band==2?2:1),1e-10),
            "ordinary light RGB and intensity never scale Li or invent another band's spectrum");
        check(!gc::validateQueries(bands.prepared,fixture.frame,error),"old frame cannot bind a changed authored Li snapshot");
    }
    auto rotated=gt::ordinaryFixture();
    rotated.scene->lights[0].transform.rotation.y=-rotated.scene->lights[0].transform.rotation.y;
    const bool rotatedReady=rotated.prepare(error);check(rotatedReady,error);
    if(rotatedReady)
    {
        const auto changed=gt::oracle(rotated);
        check(changed.complete && std::abs(changed.landingMean[3][0]-baseline.landingMean[3][0])>.005,
            "authored directional-light rotation changes independent actual receiver landing");
        check(!gc::validateQueries(rotated.prepared,fixture.frame,error),"old frame cannot bind a rotated source snapshot");
    }
    const auto encoded=sceneaov::detail::encodeScene(*fixture.scene);
    auto reopened=gt::ordinaryFixture();reopened.scene=std::make_shared<gt::Visual3DScene>();
    const bool decoded=sceneaov::detail::decodeScene(encoded,*reopened.scene);check(decoded,"reopen-like source snapshot uses existing SLS1 scene codec");
    if(decoded)
    {
        const bool reopenedReady=reopened.prepare(error);check(reopenedReady,error);
        if(reopenedReady)check(reopened.frame.identity==fixture.frame.identity,
            "reopened exact scene/material/source/policy retains identical frame ownership");
    }
    // Mutating producer aliases cannot alter either retained scene or program.
    fixture.scene->lights[0].spectralSource.radiance[0]=3;
    auto mutableReceiver=std::make_shared<gt::Program>(*fixture.program->objectPrograms[0]);
    mutableReceiver->spectralReflectance->coefficients[0]=0;fixture.program->objectPrograms[0]=mutableReceiver;
    check(gc::validateQueries(fixture.prepared,fixture.frame,error),"owning scene and material snapshots survive producer edits");
    check(fixture.prepared.glass.sourceScene->lights[0].spectralSource.radiance[0]==source.radiance[0],
        "retained source coefficients come from immutable authored ownership");
    gc::Prepared rejected;
    check(!gc::prepare(fixture.prepared.glass,gt::SceneLightId{999},*fixture.policy,rejected,error),
        "foreign selected light ID cannot bind spectral power");
    check(!gc::prepare(fixture.prepared.glass,gt::SceneLightId{},*fixture.policy,rejected,error),
        "absent selected light cannot substitute the first active light");
    for(unsigned mode=0;mode<8;++mode)
    {
        auto invalid=gt::ordinaryFixture();
        if(mode==0)invalid.scene->lights[0].id={};
        if(mode==1)invalid.scene->lights[0].spectralSource.version=0;
        if(mode==2)invalid.scene->lights[0].spectralSource.radiance[3]=17;
        if(mode==3)invalid.program->authoredScene={999};
        if(mode==4)invalid.program->sceneRevision=2;
        if(mode==5)invalid.scene->lights[0].transform.rotation={0,0,0,0};
        if(mode==6)invalid.scene->lights[0].spectralSource.traceMaximumDistance=std::numeric_limits<float>::quiet_NaN();
        if(mode==7)invalid.scene->lights[0].spectralSource.wavelengthsNanometres[2]+=.5f;
        check(!invalid.prepare(error),"invalid active source or mismatched scene/material/revision owner refuses admission");
    }
    auto clean=gt::ordinaryFixture();const bool cleanReady=clean.prepare(error);check(cleanReady,error);if(!cleanReady)return;
    auto alternateDescription=clean.policy->description();alternateDescription.minimumBranchFraction*=.5f;
    const auto alternate=gc::AdmittedBranchPolicy::admit(alternateDescription,error);check(bool(alternate),error);
    if(alternate)
    {
        gc::Prepared changed;check(gc::prepare(clean.prepared.glass,clean.ordinaryLight,*alternate,changed,error),error);
        check(!gc::validateQueries(changed,clean.frame,error),"old frame cannot bind a different admitted error policy");
        gc::BranchReceipt receipt;
        check(!gc::traceBranches(clean.prepared,clean.frame,clean.frame.entries[0],0,*alternate,receipt,error),
            "per-path caller cannot replace the owning frame's branch policy");
    }
    auto geometry=gt::ordinaryFixture();geometry.scene->objects[0].transform.translation.x=.25f;
    const bool geometryReady=geometry.prepare(error);check(geometryReady,error);
    if(geometryReady)check(!gc::validateQueries(geometry.prepared,clean.frame,error),
        "same IDs and revision numbers cannot hide a changed owning geometry snapshot");
    auto program=gt::ordinaryFixture();program.program->programIdentity+="-edited";
    const bool programReady=program.prepare(error);check(programReady,error);
    if(programReady)check(!gc::validateQueries(program.prepared,clean.frame,error),"different material program identity cannot reuse an old frame");
    for(unsigned mode=0;mode<4;++mode)
    {
        auto stale=clean.frame;
        if(mode==0)stale.sourceRadiance[0][2]*=2;
        if(mode==1)stale.branchPolicyIdentity+="-edited";
        if(mode==2)
        {
            auto active=std::make_shared<gt::SceneLightRecord>(*stale.activeSource);active->id={999};stale.activeSource=active;
        }
        if(mode==3)stale.sourceDirectionTrace[3]*=.5f;
        stale.identity=gc::frameIdentity(clean.prepared,stale);
        check(!gc::validateQueries(clean.prepared,stale,error),"self-consistent receipt status cannot reconstruct missing source or policy ownership");
    }
    for(unsigned mode=0;mode<3;++mode)
    {
        auto active=gt::ordinaryFixture();active.runtime=arbitgpu::NativeFixtureSceneRuntimeInputs{};
        auto light=active.scene->lights[0];
        if(mode==0)light.transform.rotation.y=-light.transform.rotation.y;
        if(mode==1)light.transform.translation={3,2,1};
        if(mode==2)light.transform.rotation.y=-light.transform.rotation.y;
        if(mode==2)active.runtime->animatedLights={light};else active.runtime->lightOverride=light;
        const bool activeReady=active.prepare(error);check(activeReady,error);if(!activeReady)continue;
        const auto changed=gt::oracle(active);const auto independent=gt::authoredSource(active);
        check(changed.complete,"existing runtime light transform produces independent complete receiver transport");
        for(unsigned axis=0;axis<3;++axis)check(close(active.frame.sourceDirectionTrace[axis],independent.direction[axis],1e-7),
            "runtime selected light direction follows its actual owned transform");
        if(mode==1)check(close(changed.receiverPower[3],baseline.receiverPower[3],1e-10)
            && active.frame.identity!=clean.frame.identity,"directional-source translation retains physical power and exact runtime receipt identity");
        else check(std::abs(changed.landingMean[3][0]-baseline.landingMean[3][0])>.005,
            "runtime override and animation change actual light landing without changing authored Li");
    }
    // The existing per-instance world matrices carry evaluated object motion.
    // The oracle derives that moved mesh directly from authored triangle values.
    auto moved=gt::ordinaryFixture();moved.scene->objects[0].transform.translation.x=.3f;
    std::vector<arbitgpu::opaquevisibility::Instance> instances;
    for(std::size_t o=0;o<clean.scene->objectCount;++o)
    {
        auto world=videohelper::tests::sceneclosest::identity();world[12]=o==0?.3f:0;
        instances.push_back({world,o});
    }
    gc::Frame evaluated;
    const bool movedReady=gc::buildFrame(clean.prepared,instances,32,32,evaluated,error);check(movedReady,error);
    if(movedReady)for(const auto& entry:evaluated.entries)
    {
        gc::PathReceipt receipt;const bool complete=gc::validatePath(clean.prepared,evaluated,entry,3,receipt,error);check(complete,error);
        const auto light=gt::authoredSource(moved);
        const auto expected=gt::trace(gt::worldTriangles(*moved.scene),gt::doublePoint(entry.position),gt::doublePoint(entry.normal),
            light.direction,entry.object,evaluated.ior[3],light.maximum);
        check(expected.complete && receipt.sourceBlocked==expected.sourceBlocked,"runtime object-world source visibility follows independent moved triangles");
        if(receipt.receiverHit)for(unsigned axis=0;axis<3;++axis)
            check(expected.receiver.found && close(receipt.receiver.position[axis],expected.receiver.position[axis],2e-4),
                "runtime object-world receiver endpoint follows actual transformed geometry");
    }
    for(unsigned mode=0;mode<4;++mode)
    {
        auto active=gt::ordinaryFixture();active.runtime=arbitgpu::NativeFixtureSceneRuntimeInputs{};
        active.runtime->lightOverride=active.scene->lights[0];
        if(mode==0)active.runtime->lightOverride->id={999};
        if(mode==1)active.runtime->lightOverride->spectralSource.radiance[0]=2;
        if(mode==2)active.runtime->lightOverride->spectralSource.radiance[0]=17;
        if(mode==3)
        {active.runtime->lightOverride.reset();auto light=active.scene->lights[0];light.id={999};active.runtime->animatedLights={light};}
        check(!active.prepare(error),"invalid, foreign or spectrally changed runtime light refuses rather than rebinding source power");
    }
}

void schedulesAndPower()
{
    std::string error;auto fixture=gt::ordinaryFixture();const bool ready=fixture.prepare(error);check(ready,error);if(!ready)return;
    check(fixture.frame.work.size()==fixture.frame.entries.size()*8
        && fixture.frame.workTexels.size()==fixture.frame.work.size()*gc::kWorkStride,"native work retains one independent task per entry and band");
    check(fixture.frame.bounds.maximumDeposits==fixture.frame.work.size()*fixture.policy->description().maximumVisits,
        "native admission reserves every bounded branch receiver deposit");
    check(fixture.frame.bounds.queryWork<=gc::kMaximumQueryWork,"source, branch and camera queries fit one checked work bound");
    std::uint64_t bytes=0;check(gc::admitFrameMemory(fixture.frame,4096,bytes,error)
        && bytes==4096+fixture.frame.bounds.transientBytes,"retained resources and caustic scratch are charged once by the caller");
    bytes=123;check(!gc::admitFrameMemory(fixture.frame,renderpassoutput::kMaximumTotalBytes,bytes,error)
        && bytes==123,"whole-frame memory overflow refuses atomically before allocating");
    const auto compare=[&](const gt::Fixture& f) {
        gc::EnergyLedger actual;const bool complete=gc::referencePower(f.prepared,f.frame,actual,error);check(complete,error);
        const auto expected=gt::powerOracle(f);check(expected.complete,"independent binary64 branch-power oracle completes");
        if(!complete || !expected.complete)return actual;
        for(unsigned b=0;b<8;++b)
        {
            check(close(actual.potential[b],expected.energy.potential[b],2e-6),"Li times actual projected entry sample area matches binary64");
            check(close(actual.launched[b],expected.energy.launched[b],2e-6),"source-visible launch power matches independent triangles");
            const double tail=actual.unresolved[b]+expected.energy.unresolved[b];
            check(std::abs(actual.deposited[b]-expected.energy.deposited[b])<=tail+2e-4*actual.potential[b],
                "outgoing band power applies exact receiver rho once after wavelength branches");
            check(std::abs(actual.absorbed[b]-expected.energy.absorbed[b])<=tail+2e-4*actual.potential[b],
                "receiver absorption stays separate from escaped and unresolved branch power");
            check(close(actual.launched[b],actual.deposited[b]+actual.absorbed[b]+actual.escaped[b]+actual.unresolved[b],2e-6),
                "each diagnostic band conserves deposited, absorbed, escaped and unresolved flux");
            check(close(actual.potential[b],actual.launched[b]+actual.sourceBlocked[b],2e-6),
                "blocked source work is explicitly unlaunched power");
        }
        return actual;
    };
    const auto baseline=compare(fixture);
    auto li=gt::ordinaryFixture();li.scene->lights[0].spectralSource.radiance[2]*=2;
    auto rho=gt::ordinaryFixture();auto surface=std::make_shared<gt::Program>(*rho.program->objectPrograms[0]);
    surface->spectralReflectance->coefficients[4]=0;rho.program->objectPrograms[0]=surface;
    for(auto* f:{&li,&rho})
    {
        const bool prepared=f->prepare(error);check(prepared,error);if(!prepared)continue;
        const auto energy=compare(*f);
        for(unsigned b=0;b<8;++b)
            check(close(energy.deposited[b],baseline.deposited[b]*(f==&li && b==2?2:f==&rho && b==4?0:1),2e-6),
                "isolated Li-only or rho-only edit changes only the authored output band");
        if(f==&rho)check(close(energy.deposited[4]+energy.absorbed[4],baseline.deposited[4]+baseline.absorbed[4],2e-6),
            "zero-rho receiver absorbs the same actual arriving power");
    }
    for(unsigned mode=0;mode<5;++mode)
    {
        auto f=mode<2?gt::fixture(.004f,mode==0?1:2,false):gt::ordinaryFixture();
        f.scene->lights[0]=fixture.scene->lights[0];f.ordinaryLight=fixture.ordinaryLight;f.policy=fixture.policy;
        if(mode==1)for(std::size_t v=f.scene->objects[2].firstVertex;v<f.scene->vertexCount;++v)f.scene->vertices[v].position.z=-4;
        if(mode==2)f.scene->lights[0].transform.rotation.y*=-1;
        if(mode==3)f.scene->objects[0].transform.translation.x=.3f;
        if(mode==4)f.scene->objects[1].transform.translation.z=-1;
        const bool prepared=f.prepare(error);check(prepared,error);if(!prepared)continue;
        const auto energy=compare(f);
        if(mode==0)for(unsigned b=0;b<8;++b)check(energy.launched[b]==0 && energy.deposited[b]==0,
            "actual source-leg blocker prevents every band from launching");
        if(mode==1)for(unsigned b=0;b<8;++b)check(close(energy.deposited[b],baseline.deposited[b],2e-6),
            "blocker behind the receiver cannot steal a nearer actual receiver deposit");
        if(mode>=2)check(f.frame.identity!=fixture.frame.identity,"source/glass/receiver motion retains a new exact frame identity");
    }
    std::vector<arbitgpu::opaquevisibility::Instance> instances;
    for(std::size_t o=0;o<fixture.scene->objectCount;++o)instances.push_back({videohelper::tests::sceneclosest::identity(),o});
    std::array<gt::PowerOracle,3> refinement;
    for(unsigned level=0;level<3;++level)
    {
        auto f=fixture;gc::SchedulingDescription schedule;schedule.entrySubdivisions=1u<<level;
        const bool built=gc::buildScheduledFrame(f.prepared,instances,nullptr,32,32,schedule,f.frame,error);check(built,error);if(!built)continue;
        check(f.frame.entries.size()==fixture.frame.entries.size()*schedule.entrySubdivisions*schedule.entrySubdivisions/16,
            "spatial refinement changes actual quadrature cardinality under the retained bound");
        compare(f);refinement[level]=gt::powerOracle(f);
        for(unsigned b=0;b<8;++b)check(close(refinement[level].energy.potential[b],baseline.potential[b],2e-6),
            "constant Li projected power is invariant under independent entry refinement");
    }
    // These expose independent measurements for acceptance. A branch tail
    // alone does not establish either a power or a centroid convergence pass.
    for(unsigned b=0;b<8;++b)for(unsigned level=1;level<3;++level)
    {
        const double powerDelta=std::abs(refinement[level].energy.deposited[b]-refinement[level-1].energy.deposited[b]);
        const auto shift=gt::minus(refinement[level].centroid[b],refinement[level-1].centroid[b]);
        check(std::isfinite(powerDelta) && std::isfinite(gt::inner(shift,shift)),"refinement retains separate finite integrated-power and centroid measurements");
    }
    for(unsigned mode=0;mode<3;++mode)
    {
        auto bad=fixture.frame;
        if(mode==0)bad.triangles.back().reflectance->coefficients[2]=0;
        if(mode==1)bad.triangles.back().points[0][0]+=.1f;
        if(mode==2)bad.instances[0].world[12]+=.1f;
        bad.triangleTexels.clear();for(const auto& t:bad.triangles)gc::appendTriangleTexels(t,bad.triangleTexels);
        bad.identity=gc::frameIdentity(fixture.prepared,bad);
        check(!gc::validateTables(fixture.prepared,bad,error),"self-consistent table labels cannot replace immutable rho, geometry or query owners");
    }
    gc::Frame sentinel;sentinel.identity="unchanged";gc::SchedulingDescription invalid;invalid.entrySubdivisions=8;
    check(!gc::buildScheduledFrame(fixture.prepared,instances,nullptr,32,32,invalid,sentinel,error) && sentinel.identity=="unchanged",
        "unsupported spatial refinement refuses before publishing a frame");
    const auto weights=gc::resolveWeights();double whiteY=0;
    for(unsigned b=0;b<8;++b)whiteY+=double(weights[b][1])*weights[b][3];
    check(close(whiteY,1,1e-7),"native resolve retains production CIE quadrature and unit-white luminance normalization");
    check(gc::nativeMathSource().find("causticReceiverPower")!=std::string::npos
        && gc::nativeMathSource().find("sceneClosestHit")==std::string::npos,
        "shared generated optical math does not create another scene-query authority");
}
void stableDeposition()
{
    std::string error;
    for(const auto extent:{std::pair<std::uint32_t,std::uint32_t>{1920,1080},{3840,2160}})
    {
        gc::DepositionSchedule schedule;
        check(gc::calculateDepositionSchedule(extent.first,extent.second,16384,schedule,error),error);
        check(schedule.paddedKeyCount==16384 && schedule.sortLevels==14 && schedule.sortPasses==105,
            "full-resolution bitonic schedule has a bounded power-of-two key capacity");
        check(schedule.depositionWork<=gc::kMaximumDepositionWork && schedule.resolveWork==schedule.pixelCount*8
            && schedule.depositionWork<schedule.pixelCount*schedule.maximumDeposits,
            "full-resolution deposition work replaces quadratic pixel/deposit comparisons");
        gc::DeviceDepositionLimits limits;limits.maximumBufferBytes=128ull*1024*1024;
        limits.maximumDispatchGroups=65535;limits.threadsPerGroup=128;limits.spectralBandsPerBuffer=4;
        check(gc::admitDepositionDevice(schedule,limits,error),error);
        limits.maximumDispatchGroups=1;
        check(!gc::admitDepositionDevice(schedule,limits,error),"actual native dispatch cap refuses before allocation");
        limits.maximumDispatchGroups=65535;limits.maximumBufferBytes=16;
        check(!gc::admitDepositionDevice(schedule,limits,error),"actual native storage cap refuses before allocation");
    }
    gc::DepositionSchedule unchanged;unchanged.paddedKeyCount=123;
    check(!gc::calculateDepositionSchedule(UINT32_MAX,UINT32_MAX,UINT64_MAX,unchanged,error)
        && unchanged.paddedKeyCount==123,"hostile extent and slot multiplication refuse without changing output");
    check(!gc::calculateDepositionSchedule(3840,2160,UINT64_MAX,unchanged,error)
        && unchanged.paddedKeyCount==123,"overflowed deposition slot capacity cannot wrap into a small allocation");
    gc::DepositionSchedule empty;check(gc::calculateDepositionSchedule(1920,1080,0,empty,error)
        && empty.paddedKeyCount==1 && empty.sortPasses==0,"empty visibility still owns one invalid key and a bounded image clear");
    auto fixture=gt::ordinaryFixture();fixture.scene->cameras[0].transform.translation.x=6;
    fixture.scene->cameras[0].verticalFovRadians=2;
    const bool ready=fixture.prepare(error);check(ready,error);if(!ready)return;
    for(const auto extent:{std::pair<std::uint32_t,std::uint32_t>{1920,1080},{3840,2160}})
    {
        gc::Frame full;
        const bool admitted=gc::buildScheduledFrame(fixture.prepared,fixture.frame.instances,nullptr,
            extent.first,extent.second,fixture.frame.scheduling,full,error);
        check(admitted,error);
        if(admitted)
        {
            std::uint64_t owned=0;
            check(gc::admitFrameMemory(full,256ull*1024*1024,owned,error)
                && owned<=renderpassoutput::kMaximumTotalBytes,
                "actual authored full-resolution schedule, key readback and retained frame memory fit two GiB");
            check(full.bounds.queryWork==fixture.frame.bounds.queryWork
                && full.bounds.deposition.scratchBytes==2*full.bounds.deposition.paddedKeyCount*sizeof(gc::DepositionKey),
                "full-resolution deposition storage is explicit and independent of unchanged tracing work");
        }
    }
    gc::NativeReadback synthetic;const bool made=gt::syntheticReadback(fixture,synthetic,error);check(made,error);if(!made)return;
    // Arbitrary synthetic tokens test identity matching only. They never claim
    // a device generation, native completion fence or candidate PASS.
    gc::SubmissionIdentity owner{fixture.frame.identity,1,2,3,4,5,6};
    synthetic.deposition=gt::syntheticDeposition(fixture.frame,synthetic,owner);
    gc::EnergyLedger ledger;
    check(gc::validateNativeReadback(fixture.prepared,fixture.frame,synthetic,ledger,error,&owner),error);
    const auto quadrature=gt::quadraturePowerOracle(fixture);
    check(quadrature.complete,"independent Duffy quadrature traces complete reflected and transmitted branches");
    for(unsigned b=0;b<8;++b)
        check(close(quadrature.energy.launched[b],quadrature.energy.deposited[b]+quadrature.energy.absorbed[b]
            +quadrature.energy.escaped[b]+quadrature.energy.unresolved[b],1e-10),
            "independent full-branch spatial reference conserves each authored band");
    const auto moments=gt::depositionMoment(fixture.frame,synthetic);
    for(unsigned b=0;b<8;++b)
    {
        check(close(moments.power[b],ledger.cameraVisible[b],1e-12),"canonical pixel bins retain every visible band energy");
        if(moments.power[b]>0)
            check(moments.x[b]>0 && moments.x[b]<moments.power[b]*fixture.frame.width
                && moments.y[b]>0 && moments.y[b]<moments.power[b]*fixture.frame.height
                && moments.xx[b]*moments.power[b]>=moments.x[b]*moments.x[b]-1e-8
                && moments.yy[b]*moments.power[b]>=moments.y[b]*moments.y[b]-1e-8,
                "canonical deposition retains pixel centroid and nonnegative spatial variance");
    }
    for(const auto& value:moments.radiance)
        check(close(synthetic.spectralRadiance[value.first],value.second,4e-6),
            "independent canonical float32 segment diagnostic matches synthetic visible pixel bands");
    check(synthetic.deposition.visibleCount>1,"order fixture includes multiple real receiver deposit keys");
    if(synthetic.deposition.visibleCount<2)return;
    for(unsigned mutation=0;mutation<16;++mutation)
    {
        auto bad=synthetic;auto& receipt=bad.deposition;
        if(mutation==0)std::swap(receipt.keys[0],receipt.keys[1]);
        if(mutation==1)receipt.keys[1]=receipt.keys[0];
        if(mutation==2)receipt.keys[0].flatIndex+=1;
        if(mutation==3)receipt.keys[0].pixel+=1;
        if(mutation==4)receipt.keys[0].work=UINT32_MAX;
        if(mutation==5)receipt.keys[0].slot=UINT32_MAX;
        if(mutation==6)receipt.keys.back()=receipt.keys[0];
        if(mutation==7)--receipt.visibleCount;
        if(mutation==8)receipt.overflow=true;
        if(mutation==9)receipt.complete=false;
        if(mutation==10)++receipt.submission.rendererGeneration;
        if(mutation==11)++receipt.submission.submissionSerial;
        if(mutation==12)++receipt.submission.rasterGeneration;
        if(mutation==13)++receipt.submission.queryGeneration;
        if(mutation==14)++receipt.submission.scratchGeneration;
        if(mutation==15)++receipt.submission.completedFence;
        gc::EnergyLedger sentinel;sentinel.deposited.fill(123);
        check(!gc::validateNativeReadback(fixture.prepared,fixture.frame,bad,sentinel,error,&owner)
            && sentinel.deposited[0]==123,"reordered keys, incomplete permutation or stale resource/fence refuses atomic Color gate");
    }
    check(!gc::validateNativeReadback(fixture.prepared,fixture.frame,synthetic,ledger,error),
        "versioned deposition cannot validate without a separately retained expected submission");
    auto staleOwner=owner;staleOwner.frameIdentity+="changed";
    check(!gc::validateNativeReadback(fixture.prepared,fixture.frame,synthetic,ledger,error,&staleOwner),
        "exact expected scene, light, material, camera and query frame owner cannot change");
}

void cameraCellGeometry()
{
    gc::Frame frame;frame.width=frame.height=1;frame.camera.id={17};
    frame.camera.verticalFovRadians=float(3.14159265358979323846/2);frame.camera.nearPlane=.1f;frame.camera.farPlane=100;
    const auto triangle=[](std::array<gc::Point,3> points) {
        gc::Triangle result;result.points=points;result.normal=gc::cross(gc::sub(points[1],points[0]),gc::sub(points[2],points[0]));
        result.area=.5f*std::sqrt(gc::dot(result.normal,result.normal));gc::normalize(result.normal);
        result.object=0xf1234567;result.material=0xe2345678;result.instance=0xd3456789;result.triangle=0xc456789a;return result;
    };
    const auto fullTriangle=triangle({gc::Point{-20,-20,-2},gc::Point{20,-20,-2},gc::Point{0,20,-2}});
    gc::CameraCellGeometry full;std::string error;
    const bool ready=gc::prepareCameraCellGeometry(frame,fullTriangle,0,0,full,error);
    check(ready,"pure camera-cell full pixel prepares: "+error);if(!ready)return;
    // Independent axis-aligned square formulas, with no product ray or clip.
    const double tangent=std::tan(double(frame.camera.verticalFovRadians)/2);
    const double pixelAngle=4*std::atan(tangent*tangent/std::sqrt(1+2*tangent*tangent));
    check(full.vertexCount==4&&close(full.worldArea,16*tangent*tangent,2e-6)
        &&close(full.pixelSolidAngle,pixelAngle,2e-6)&&close(full.clippedSolidAngle,pixelAngle,2e-6),
        "full clipped world area and solid angle agree with independent square measures");
    check(full.camera==17&&full.x==0&&full.y==0&&full.object==fullTriangle.object&&full.material==fullTriangle.material
        &&full.instance==fullTriangle.instance&&full.triangle==fullTriangle.triangle,
        "geometry candidate preserves exact integer camera/pixel/triangle IDs without a native receipt");
    auto rectangularFrame=frame;rectangularFrame.width=4;rectangularFrame.height=2;gc::CameraCellGeometry offCenter;
    check(gc::prepareCameraCellGeometry(rectangularFrame,fullTriangle,1,0,offCenter,error)&&offCenter.x==1&&offCenter.y==0
        &&close(offCenter.worldArea,4*tangent*tangent,2e-6)&&close(offCenter.pixelSolidAngle,.25*pixelAngle,2e-6)
        &&offCenter.vertices[0][0]<0&&offCenter.vertices[0][1]>0,
        "non-square extent and nominated off-center pixel preserve aspect and top-first coordinates");
    gc::CameraCellGeometry partial;
    const auto halfTriangle=triangle({gc::Point{-2,-2,-2},gc::Point{2,-2,-2},gc::Point{-2,2,-2}});
    check(gc::prepareCameraCellGeometry(frame,halfTriangle,0,0,partial,error)&&partial.vertexCount>=3&&partial.vertexCount<=9
        &&close(partial.worldArea,8,2e-6)&&close(partial.clippedSolidAngle,.5*pixelAngle,2e-6)
        &&partial.pixelSolidAngle==full.pixelSolidAngle,
        "triangle edges clip half a pixel while full pixel solid angle stays independent of coverage");
    const auto outside=triangle({gc::Point{10,10,-2},gc::Point{20,10,-2},gc::Point{10,20,-2}});
    auto empty=full;
    check(gc::prepareCameraCellGeometry(frame,outside,0,0,empty,error)&&empty.vertexCount==0&&empty.worldArea==0
        &&empty.clippedSolidAngle==0&&empty.pixelSolidAngle==full.pixelSolidAngle&&empty.vertices==gc::CameraCellGeometry{}.vertices,
        "bounded empty triangle overlap is a valid cleared geometry result with positive full pixel angle");
    // Plane z=x/4-3. Full pixel cuts x in [-4,2.4]; linear depth cuts
    // near=2.5/far=3.5 leave [-2,2], y=+/-depth. Integrate that trapezoid.
    const auto tilted=triangle({gc::Point{-20,-20,-8},gc::Point{20,-20,2},gc::Point{0,20,-3}});
    gc::CameraCellGeometry tiltedFull,nearOnly,farOnly,depthClipped;
    check(gc::prepareCameraCellGeometry(frame,tilted,0,0,tiltedFull,error)
        &&close(tiltedFull.worldArea,40.96*std::sqrt(1.0625),3e-6),
        "tilted full footprint uses its actual world plane area");
    auto clippedFrame=frame;clippedFrame.camera.nearPlane=2.5f;
    check(gc::prepareCameraCellGeometry(clippedFrame,tilted,0,0,nearOnly,error)
        &&close(nearOnly.worldArea,39*std::sqrt(1.0625),3e-6),"actual near plane clips linear depth, not ray distance");
    clippedFrame=frame;clippedFrame.camera.farPlane=3.5f;
    check(gc::prepareCameraCellGeometry(clippedFrame,tilted,0,0,farOnly,error)
        &&close(farOnly.worldArea,25.96*std::sqrt(1.0625),3e-6),"actual far plane clips linear depth, not ray distance");
    clippedFrame.camera.nearPlane=2.5f;
    const double clippedAngle=2*(std::atan(.8/std::sqrt(2+.8*.8))
        +std::atan((4.0/7)/std::sqrt(2+16.0/49)));
    check(gc::prepareCameraCellGeometry(clippedFrame,tilted,0,0,depthClipped,error)&&depthClipped.vertexCount>=3&&depthClipped.vertexCount<=9
        &&close(depthClipped.worldArea,24*std::sqrt(1.0625),3e-6)&&close(depthClipped.clippedSolidAngle,clippedAngle,3e-6)
        &&depthClipped.pixelSolidAngle==full.pixelSolidAngle,
        "both depth planes retain the independently integrated trapezoid area and angular rectangle");
    for(std::size_t i=0;i<depthClipped.vertexCount;++i) {
        const auto& p=depthClipped.vertices[i];
        check(-p[2]>=2.5-1e-8&&-p[2]<=3.5+1e-8&&std::abs(p[2]-.25*p[0]+3)<1e-8,
            "every returned vertex lies on the nominated plane inside actual camera depth limits");
    }
    clippedFrame=frame;clippedFrame.camera.farPlane=2;
    check(gc::prepareCameraCellGeometry(clippedFrame,tilted,0,0,empty,error)&&empty.vertexCount==0
        &&empty.worldArea==0&&empty.clippedSolidAngle==0&&empty.pixelSolidAngle==full.pixelSolidAngle,
        "bounded all-far-clipped coverage is valid empty geometry");
    clippedFrame=frame;clippedFrame.camera.nearPlane=4.5f;clippedFrame.camera.farPlane=5;
    check(gc::prepareCameraCellGeometry(clippedFrame,tilted,0,0,empty,error)&&empty.vertexCount==0
        &&empty.worldArea==0&&empty.clippedSolidAngle==0&&empty.pixelSolidAngle==full.pixelSolidAngle,
        "bounded all-near-clipped coverage is valid empty geometry");
    auto rotatedFrame=frame;rotatedFrame.camera.transform.translation={3,4,5};
    rotatedFrame.camera.transform.rotation={0,float(std::sqrt(.5)),0,float(std::sqrt(.5))};
    auto rotatedPoints=fullTriangle.points;
    for(auto& p:rotatedPoints) {
        const auto r=arbitgpu::rotateSceneVector(rotatedFrame.camera.transform.rotation,{p[0],p[1],p[2]});
        p={r.x+3,r.y+4,r.z+5};
    }
    const auto rotatedTriangle=triangle(rotatedPoints);gc::CameraCellGeometry rotated;
    check(gc::prepareCameraCellGeometry(rotatedFrame,rotatedTriangle,0,0,rotated,error)
        &&close(rotated.worldArea,full.worldArea,3e-6)&&close(rotated.clippedSolidAngle,full.clippedSolidAngle,3e-6)
        &&close(rotated.pixelSolidAngle,full.pixelSolidAngle,3e-6),"rigid camera/triangle rotation and translation preserve geometric measures");
    rotatedPoints=tilted.points;
    for(auto& p:rotatedPoints) {
        const auto r=arbitgpu::rotateSceneVector(rotatedFrame.camera.transform.rotation,{p[0],p[1],p[2]});
        p={r.x+3,r.y+4,r.z+5};
    }
    rotatedFrame.camera.nearPlane=2.5f;rotatedFrame.camera.farPlane=3.5f;
    check(gc::prepareCameraCellGeometry(rotatedFrame,triangle(rotatedPoints),0,0,rotated,error)
        &&close(rotated.worldArea,depthClipped.worldArea,3e-6)&&close(rotated.clippedSolidAngle,depthClipped.clippedSolidAngle,3e-6),
        "actual near/far planes rotate with the camera rather than clipping world Z");
    const auto unchanged=[&](const gc::CameraCellGeometry& value) {
        return value.vertices==full.vertices&&value.vertexCount==full.vertexCount
            &&std::tie(value.camera,value.x,value.y,value.object,value.material,value.instance,value.triangle)
                ==std::tie(full.camera,full.x,full.y,full.object,full.material,full.instance,full.triangle)
            &&value.worldArea==full.worldArea&&value.clippedSolidAngle==full.clippedSolidAngle&&value.pixelSolidAngle==full.pixelSolidAngle;
    };
    const auto refuses=[&](const gc::Frame& camera,const gc::Triangle& receiver,std::uint32_t x=0,std::uint32_t y=0) {
        auto sentinel=full;return !gc::prepareCameraCellGeometry(camera,receiver,x,y,sentinel,error)&&unchanged(sentinel);
    };
    check(refuses(frame,fullTriangle,1,0)&&refuses(frame,fullTriangle,0,1),"out-of-frame pixel refuses without output mutation");
    auto invalidFrame=frame;invalidFrame.width=0;check(refuses(invalidFrame,fullTriangle),"zero extent refuses atomically");
    invalidFrame=frame;invalidFrame.camera.verticalFovRadians=0;check(refuses(invalidFrame,fullTriangle),"invalid camera FOV refuses atomically");
    invalidFrame=frame;invalidFrame.camera.nearPlane=invalidFrame.camera.farPlane;check(refuses(invalidFrame,fullTriangle),"invalid camera depth range refuses atomically");
    invalidFrame=frame;invalidFrame.camera.farPlane=std::numeric_limits<float>::infinity();check(refuses(invalidFrame,fullTriangle),"nonfinite camera refuses atomically");
    invalidFrame=frame;invalidFrame.camera.transform.scale.x=2;check(refuses(invalidFrame,fullTriangle),"scaled camera refuses atomically");
    invalidFrame=frame;invalidFrame.camera.transform.rotation={0,0,0,0};check(refuses(invalidFrame,fullTriangle),"invalid camera quaternion refuses atomically");
    auto invalid=fullTriangle;invalid.points[2]=invalid.points[1];check(refuses(frame,invalid),"degenerate triangle refuses atomically");
    const auto sliver=triangle({gc::Point{0,0,-2},gc::Point{1,0,-2},gc::Point{1,1e-13f,-2}});
    check(refuses(frame,sliver),"relative triangle sliver bound refuses atomically without an absolute area floor");
    invalid=fullTriangle;invalid.normal[2]=-1;check(refuses(frame,invalid),"mismatched oriented normal metadata refuses atomically");
    invalid=fullTriangle;invalid.area*=2;check(refuses(frame,invalid),"stale triangle area metadata refuses atomically");
    invalid=fullTriangle;invalid.points[0][0]=std::numeric_limits<float>::quiet_NaN();check(refuses(frame,invalid),"nonfinite world point refuses atomically");
    invalid=fullTriangle;invalid.normal[0]=std::numeric_limits<float>::infinity();check(refuses(frame,invalid),"nonfinite triangle normal refuses atomically");
    invalid=fullTriangle;std::swap(invalid.points[1],invalid.points[2]);invalid=triangle(invalid.points);
    check(refuses(frame,invalid),"actual reversed winding is backfacing and refuses atomically");
    const auto horizon=triangle({gc::Point{-2,-2,-2},gc::Point{-2,-2,2},gc::Point{-2,2,0}});
    check(refuses(frame,horizon),"pixel boundary rays crossing the plane horizon refuse atomically");
    invalidFrame=frame;invalidFrame.camera.verticalFovRadians=1e-8f;
    const auto grazing=triangle({gc::Point{-2,-2,0},gc::Point{-2+2e-6f,-2,-4},gc::Point{-2,2,0}});
    check(refuses(invalidFrame,grazing),"uniformly grazing boundary rays refuse atomically");
    const auto unbounded=triangle({gc::Point{-20,-20,-2e6f},gc::Point{20,-20,-2e6f},gc::Point{0,20,-2e6f}});
    check(refuses(frame,unbounded),"far-scaled unbounded plane footprint refuses atomically");
}

void cameraAndAtomicReadback()
{
    std::string error;auto fixture=gt::ordinaryFixture();fixture.scene->cameras[0].transform.translation.x=6;
    fixture.scene->cameras[0].verticalFovRadians=2;
    const bool ready=fixture.prepare(error);check(ready,error);if(!ready)return;
    gc::NativeReadback synthetic;const bool made=gt::syntheticReadback(fixture,synthetic,error);check(made,error);if(!made)return;
    gc::EnergyLedger ledger;const bool accepted=gc::validateNativeReadback(fixture.prepared,fixture.frame,synthetic,ledger,error);check(accepted,error);
    check(std::any_of(ledger.cameraVisible.begin(),ledger.cameraVisible.end(),[](double power){return power>0;}),
        "synthetic camera control includes actual visible receiver footprints");
    const auto oracle=gt::powerOracle(fixture);
    for(unsigned b=0;b<8;++b)check(std::abs(ledger.deposited[b]-oracle.energy.deposited[b])<=ledger.unresolved[b]+2e-4*ledger.potential[b],
        "synthetic receipt powers match independent binary64 transport without claiming native pixels");
    const auto visible=std::find_if(synthetic.deposits.begin(),synthetic.deposits.end(),[](const auto& d){return d.cameraVisible;});
    check(visible!=synthetic.deposits.end(),"camera fixture contains one independently visible exact-triangle deposit");if(visible==synthetic.deposits.end())return;
    const std::size_t selected=std::size_t(visible-synthetic.deposits.begin());
    const auto& hit=visible->receiver;const auto& camera=fixture.frame.camera;
    const double distance=double(camera.transform.translation.z)-hit.position[2],tangent=std::tan(double(camera.verticalFovRadians)/2);
    const double px=(double(hit.position[0]-camera.transform.translation.x)/(distance*tangent)+1)*16;
    const double py=(1-double(hit.position[1]-camera.transform.translation.y)/(distance*tangent))*16;
    const double side=2*distance*tangent/32;
    check(visible->footprint.x==std::uint32_t(px) && visible->footprint.y==std::uint32_t(py),
        "exact receiver hit projects to independent top-first camera pixel");
    check(close(visible->footprint.worldArea,side*side,2e-6),"receiver footprint uses independent actual plane pixel area");
    for(unsigned mode=0;mode<17;++mode)
    {
        auto bad=synthetic;auto& d=bad.deposits[selected];
        if(mode==0)bad.frameIdentity+="-stale";
        if(mode==1)bad.paths[0].complete=false;
        if(mode==2)d.receiver.materialId+=1;
        if(mode==3)d.footprint.x+=1;
        if(mode==4)d.outgoingPower*=2;
        if(mode==5)d.raster.linearDepth+=1;
        if(mode==6)d.raster.triangle+=1;
        if(mode==7)d.pixelHit.instanceId+=1;
        if(mode==8)d.cameraHit.position[0]+=.1f;
        if(mode==9)d.cameraVisible=false;
        if(mode==10)bad.spectralRadiance[(std::size_t(d.footprint.y)*32+d.footprint.x)*8+fixture.frame.work[d.work].band]+=1;
        if(mode==11)bad.deposits.push_back(d);
        if(mode==12)bad.paths[d.work].unresolvedPower=bad.paths[d.work].launchedPower*.1f;
        if(mode==13)bad.paths[d.work].sourceHit.status=-1;
        if(mode==14)bad.paths[d.work].entryHit.triangleId+=1;
        if(mode==15)bad.paths[d.work].queryWork=gc::kMaximumQueryWork+1;
        if(mode==16)bad.paths[d.work].visits=gc::kMaximumBranchVisits+1;
        gc::EnergyLedger sentinel;sentinel.deposited.fill(123);
        check(!gc::validateNativeReadback(fixture.prepared,fixture.frame,bad,sentinel,error)
            && std::all_of(sentinel.deposited.begin(),sentinel.deposited.end(),[](double p){return p==123;}),
            "stale owner, endpoint, splat, rho, depth, duplicate or energy error refuses the whole readback atomically");
    }
    auto hidden=gt::ordinaryFixture();const bool hiddenReady=hidden.prepare(error);check(hiddenReady,error);
    if(hiddenReady)
    {
        gc::NativeReadback masked;const bool madeHidden=gt::syntheticReadback(hidden,masked,error);check(madeHidden,error);
        gc::EnergyLedger energy;
        if(madeHidden)check(gc::validateNativeReadback(hidden.prepared,hidden.frame,masked,energy,error),error);
        check(energy.deposited[3]>energy.cameraVisible[3],"opaque camera blocker masks pixels while retaining physical outgoing receiver energy");
    }
    auto moved=fixture;moved.runtime=arbitgpu::NativeFixtureSceneRuntimeInputs{};moved.runtime->cameraTranslationOffset[0]=.2f;
    const bool movedReady=moved.prepare(error);check(movedReady,error);
    if(movedReady)
    {
        gc::PixelFootprint next;check(gc::projectReceiver(moved.frame,hit,next,error),error);
        check(moved.frame.cameraIdentity!=fixture.frame.cameraIdentity && moved.frame.identity!=fixture.frame.identity,
            "runtime camera movement changes both camera and whole-frame identities");
        gc::EnergyLedger sentinel;sentinel.deposited.fill(123);
        check(!gc::validateNativeReadback(moved.prepared,moved.frame,synthetic,sentinel,error),"previous camera receipt cannot publish under a moved camera");
    }
    moved=fixture;moved.runtime=arbitgpu::NativeFixtureSceneRuntimeInputs{};moved.runtime->cameraOverride=moved.scene->cameras[0];
    moved.runtime->cameraOverride->id={999};check(!moved.prepare(error),"foreign runtime camera ID cannot retain exact active ownership");
}
void refusals()
{
    std::string error;auto fixture=gt::fixture();const bool prepared=fixture.prepare(error);check(prepared,error);if(!prepared)return;
    const auto rejectFrame=[&](gc::Frame bad,const char* message){check(!gc::validateQueries(fixture.prepared,bad,error),message);};
    auto bad=fixture.frame;bad.entries.clear();rejectFrame(bad,"missing entry work refuses whole frame");
    bad=fixture.frame;bad.query.texels.resize(2);rejectFrame(bad,"truncated closest-hit payload refuses whole frame");
    bad=fixture.frame;bad.sourceRadiance[1][2]*=2;rejectFrame(bad,"changed Li after admission refuses");
    bad=fixture.frame;bad.ior[0]+=.1f;rejectFrame(bad,"changed wavelength IOR refuses");
    bad=fixture.frame;bad.entries[0].triangle=999;rejectFrame(bad,"stale entry triangle refuses");
    bad=fixture.frame;bad.triangles.back().reflectance->coefficients[2]=0;rejectFrame(bad,"rho changed after admission refuses");
    bad=fixture.frame;bad.query.texels[std::size_t(bad.query.control[0])-1][0]=1;
    bad.query.texels[std::size_t(bad.query.control[0])-1][1]=0;
    bad.identity=gc::frameIdentity(fixture.prepared,bad);rejectFrame(bad,"actual query-work exhaustion refuses whole frame");
    const auto rejectMaterial=[&](std::shared_ptr<gt::Program> root,const char* message)
    {
        arbitgpu::smoothglass::Prepared glass;
        const std::shared_ptr<const gt::Visual3DScene> snapshot=fixture.scene;
        const std::shared_ptr<const gt::Program> program=root;
        gc::Prepared admitted;
        check(!arbitgpu::smoothglass::prepare(snapshot,program,{},glass,error)
            || !gc::prepare(glass,fixture.light,admitted,error),message);
    };
    for(unsigned mode=0;mode<4;++mode)
    {
        auto root=std::make_shared<gt::Program>(*fixture.program);
        auto receiver=std::make_shared<gt::Program>(*root->objectPrograms[0]);root->objectPrograms[0]=receiver;
        if(mode==0)receiver->spectralReflectance.reset();
        if(mode==1)receiver->authoredMaterial={999};
        if(mode==2)receiver->evaluationRevision=2;
        if(mode==3)receiver->spectralReflectance->coefficients[5]=1.1f;
        rejectMaterial(root,"missing/stale/invalid exact receiver rho owner refuses");
    }
    // Rho is an independent amplitude companion. It supplies neither a
    // metallic/coat lobe nor transmitted flux or a changed shading normal.
    // Prove caustic admission re-checks its actual owned material snapshots,
    // even if a caller retains an earlier accepted smooth-glass query.
    for(unsigned mode=0;mode<9;++mode)
    {
        auto root=std::make_shared<gt::Program>(*fixture.program);
        auto receiver=std::make_shared<gt::Program>(*root->objectPrograms[0]);root->objectPrograms[0]=receiver;
        const char* message="";
        switch(mode)
        {
            case 0: receiver->parameters.baseColorMetallic[3]=.25f;message="metallic receiver cannot become rho-only Lambertian";break;
            case 1: receiver->parameters.transmissionIorClearcoat[2]=.25f;message="clearcoat receiver cannot lose its reflected lobe";break;
            case 2: receiver->parameters.transmissionIorClearcoat[0]=.25f;message="transmitting receiver cannot deposit all flux as diffuse";break;
            case 3: receiver->parameters.emissionRoughness[3]=.25f;message="non-diffuse roughness cannot be ignored";break;
            case 4: receiver->parameters.normalOpacity[3]=.5f;message="partial-opacity receiver cannot become fully opaque";break;
            case 5: receiver->parameters.normalOpacity={.6f,0,.8f,1};message="authored tangent normal cannot be replaced with geometric normal";break;
            case 6: receiver->baseColorSource=gt::Program::BaseColorSource::ImportedSrgbTexture;
                receiver->importedBaseColorTexture=gt::Program::ImportedSrgbTexture{1,1,1,{{128,128,128,255}}};
                message="image-driven receiver stays outside constant diffuse admission";break;
            case 7: receiver->baseColorSource=gt::Program::BaseColorSource::TimeLinearMix;
                receiver->timeMixEndColor={.5f,.5f,.5f};message="time-driven receiver stays outside constant diffuse admission";break;
            case 8: receiver->parameters.emissionRoughness[0]=.25f;message="emission cannot be ignored in rho-only transport";break;
        }
        auto glass=fixture.prepared.glass;glass.sourceMaterial=root;
        check(arbitgpu::validNativeFixtureMaterialPrograms(*root,*glass.sourceScene,root->backend),
            "negative receiver has valid ordinary Surface ownership and finite authored fields");
        gc::Prepared candidate;const bool accepted=gc::prepare(glass,fixture.light,candidate,error);
        check(!accepted,message);
        const char* expected=mode==6 || mode==7?"Closed glass transport requires constant untextured static surfaces"
            :mode==8?"Glass caustic receiver emission is unsupported"
            :"Closed glass receivers require unit-opacity constant diffuse material";
        check(error==expected,"unsupported receiver diagnostic names its exact admitted limitation");
    }
    for(unsigned mode=0;mode<2;++mode)
    {
        auto scene=std::make_shared<gt::Visual3DScene>(*fixture.scene);
        if(mode==0)
        {
            scene->textureCount=1;scene->textureTexelCount=1;scene->textureTexels={{128,128,255,255}};
            scene->textures[0].id={251};scene->textures[0].width=scene->textures[0].height=1;
            scene->materials[1].normalTexture={251};
        }
        else scene->vertices[scene->objects[1].firstVertex].normal={.6f,0,.8f};
        check(HarmonicMIDI::grid::validateVisual3DScene(*scene).valid(),"normal-input rejection fixture retains valid authored scene geometry");
        auto glass=fixture.prepared.glass;glass.sourceScene=scene;
        gc::Prepared candidate;const bool accepted=gc::prepare(glass,fixture.light,candidate,error);
        check(!accepted,"normal texture/interpolation cannot silently become flat diffuse geometry");
        check(error==(mode==0?"Closed glass transport refuses textured or nonopaque coverage"
            :"Closed glass receivers require flat geometric vertex normals"),"unsupported normal diagnostic identifies the ignored input");
    }
    auto black=gt::fixture();auto blackReceiver=std::make_shared<gt::Program>(*black.program->objectPrograms[0]);
    blackReceiver->spectralReflectance->coefficients.fill(0);black.program->objectPrograms[0]=blackReceiver;
    const bool blackPrepared=black.prepare(error);check(blackPrepared,error);
    const auto blackOracle=gt::oracle(black);
    for(unsigned band=0;band<8;++band)
        check(blackOracle.receiverPower[band]>0 && blackOracle.outgoingPower[band]==0,
            "ordinary zero-rho receiver admits geometry and absorbs incident power independently");
    auto rough=std::make_shared<gt::Program>(*fixture.program);rough->parameters.emissionRoughness[3]=.1f;
    rejectMaterial(rough,"rough closed glass without an explicit native consumer stays refused");
    check(!gc::consumerCanRender({},arbitgpu::NativeFixtureMaterialBackend::OpenGl)
        && !gc::consumerCanRender({},arbitgpu::NativeFixtureMaterialBackend::Metal),"shared admission cannot claim absent native caustic consumers");
    check(!gc::validateWholeOutput(std::vector<float>(4),1,1,false,error),"incomplete paths refuse even finite Color");
    check(!gc::validateWholeOutput({0,0,std::numeric_limits<float>::infinity(),1},1,1,true,error),"nonfinite Color refuses whole output");
}
void gratingEntryCells()
{
    auto fixture=gt::ordinaryFixture();std::string error;
    for(std::size_t v=0;v<fixture.scene->vertexCount;++v)if(fixture.scene->vertices[v].position.z>0)
        fixture.scene->vertices[v].position.z-=.25f*fixture.scene->vertices[v].position.x;
    fixture.scene->lights[0].transform.rotation={0,0,0,1};
    const auto& target=fixture.scene->objects[1];
    for(std::size_t v=0;v<target.vertexCount;++v) {
        auto& vertex=fixture.scene->vertices[target.firstVertex+v];vertex.uv={vertex.position.x/12,vertex.position.y/12};
    }
    check(fixture.prepare(error),"source cell fixture prepares: "+error);
    auto description=diffractionmaterial::makeAluminiumBinaryGratingPreset();
    description.spectrum.lastOrder=1;description.roughness.rmsSlope=0;description.microstructure.grooveDepthNanometres=.9f;
    const auto admitted=diffractionmaterial::admit(description,error);check(bool(admitted),"source cell material admits");if(!admitted)return;
    auto material=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*admitted);
    auto owner=std::make_shared<gt::Program>(*fixture.program->objectPrograms[0]);
    owner->kind=arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;owner->spectralReflectance.reset();
    owner->diffractionPathCount=1;owner->bindingDigest="entry-cell-grating";owner->programIdentity="entry-cell-program";
    owner->diffractionPaths[0].material=diffractionmaterial::physicalcheckpoint::makeGpuParameters(*material,{});
    // The original centroid formula is evaluated separately from the rounded
    // cell vertices. Affine float evaluation does not commute with averaging.
    gc::Triangle affine;affine.glass=true;affine.object=1;affine.material=2;
    affine.points={{{.13f,.27f,.41f},{1.37f,.22f,.43f},{.09f,1.71f,.53f}}};
    affine.normal=gc::cross(gc::sub(affine.points[1],affine.points[0]),gc::sub(affine.points[2],affine.points[0]));
    affine.area=.5f*std::sqrt(gc::dot(affine.normal,affine.normal));gc::normalize(affine.normal);
    std::vector<gc::EntrySample> affineSamples;gc::appendEntrySamples(affine,affine.normal,{gc::kScheduleVersion,4},affineSamples);
    bool differentAverage=false;
    for(std::size_t i=0;i<affineSamples.size();++i) {
        gc::GratingEntryCell cell;
        check(gc::reconstructGratingEntryCell(affine,affineSamples[i],{gc::kScheduleVersion,4},std::uint32_t(i),cell,error),
            "source cell float consistency admits rounded affine vertices: "+error);
        for(unsigned a=0;a<3;++a)differentAverage|=(double(cell.vertices[0][a])+cell.vertices[1][a]+cell.vertices[2][a])/3!=affineSamples[i].position[a];
    }
    check(differentAverage,"retained source cell does not require bit-identical averaged centroid");
    std::string previousIdentity;
    for(std::size_t n:{1u,2u,4u}) {
        gc::Frame frame;
        check(gc::buildScheduledFrame(fixture.prepared,fixture.frame.instances,nullptr,32,32,{gc::kScheduleVersion,n},frame,error),
            "source cell schedule prepares: "+error);
        std::shared_ptr<const gc::GratingIncidentBinding> parent;
        check(gc::prepareGratingIncidentBinding(fixture.prepared,frame,owner,material,.25,parent,error),"source cell incident parent prepares: "+error);
        if(!parent)continue;
        const auto oldBounds=parent->bounds;const auto oldIdentity=parent->identity;const auto oldWork=parent->frame.workTexels;
        std::shared_ptr<const gc::GratingEntryCellBinding> cells;
        check(gc::prepareGratingEntryCellBinding(parent,cells,error),"optional retained source cells prepare: "+error);if(!cells)continue;
        check(gc::validateGratingEntryCellBinding(*cells,parent,error),"source cells validate against exact strong parent: "+error);
        check(cells->incidentBinding==parent&&cells->cells.size()==frame.entries.size()
            &&cells->cellTexels.size()==frame.entries.size()*gc::kGratingEntryCellStride,"source cells retain one independent upload record per entry");
        check(parent->identity==oldIdentity&&parent->bounds.bytes==oldBounds.bytes&&parent->bounds.queryWork==oldBounds.queryWork
            &&parent->bounds.combinedWork==oldBounds.combinedWork&&parent->frame.workTexels==oldWork,
            "optional source geometry preserves terminal identity, bytes, queries and work upload");
        check(cells->identity!=oldIdentity&&cells->identity!=previousIdentity,"private cell identity includes schedule and encoding");previousIdentity=cells->identity;
        check(cells->bounds.nativeUploadBytes==cells->cellTexels.size()*sizeof(gc::Texel)
            &&cells->bounds.bytes==cells->bounds.hostBytes+cells->bounds.nativeUploadBytes
            &&cells->bounds.bytes<=renderpassoutput::kMaximumTotalBytes-parent->bounds.bytes-frame.bounds.transientBytes,
            "cell host and dedicated native upload bytes share the whole-frame cap");
        std::size_t index=0;
        for(const auto& triangle:frame.triangles) {
            if(!triangle.glass||gc::dot(triangle.normal,{frame.sourceDirectionTrace[0],frame.sourceDirectionTrace[1],frame.sourceDirectionTrace[2]})<=0)continue;
            using Vertex=std::pair<unsigned,unsigned>;
            using Edge=std::pair<Vertex,Vertex>;
            std::map<Edge,unsigned> edgeCounts;std::set<std::array<Vertex,3>> unique;
            unsigned ordinal=0;double totalArea=0,domainArea=0;
            for(unsigned u=0;u<n;++u)for(unsigned v=0;v<n-u;++v)for(unsigned down=0;down<(u+v+1<n?2u:1u);++down) {
                const std::array<Vertex,3> expected=down?std::array<Vertex,3>{{{u+1,v},{u+1,v+1},{u,v+1}}}
                    :std::array<Vertex,3>{{{u,v},{u+1,v},{u,v+1}}};
                const auto& cell=cells->cells[index];const auto& sample=frame.entries[index];
                check(cell.entry==index&&cell.quadrature==ordinal++&&cell.object==triangle.object&&cell.material==triangle.material
                    &&cell.instance==triangle.instance&&cell.triangle==triangle.triangle,"cell ordinal and exact source triangle owner agree");
                const auto e1=gc::sub(triangle.points[1],triangle.points[0]),e2=gc::sub(triangle.points[2],triangle.points[0]);
                for(unsigned corner=0;corner<3;++corner) {
                    const std::array<float,2> bary{float(expected[corner].first)/n,float(expected[corner].second)/n};
                    check(cell.barycentrics[corner]==bary&&cell.vertices[corner]==gc::advance(gc::advance(triangle.points[0],e1,bary[0]),e2,bary[1]),
                        "up/down cell corners follow the exact ordinal affine construction");
                    auto a=expected[corner],b=expected[(corner+1)%3];if(b<a)std::swap(a,b);++edgeCounts[{a,b}];
                }
                auto sorted=expected;std::sort(sorted.begin(),sorted.end());check(unique.insert(sorted).second,"source ordinal cells are unique");
                const auto a=expected[0],b=expected[1],c=expected[2];
                const int determinant=(int(b.first)-int(a.first))*(int(c.second)-int(a.second))
                    -(int(b.second)-int(a.second))*(int(c.first)-int(a.first));
                check(determinant==1,"each source cell has positive exact unit lattice area");domainArea+=double(determinant)/(2*n*n);
                totalArea+=sample.area;
                for(unsigned axis=0;axis<3;++axis)check(close((double(cell.vertices[0][axis])+cell.vertices[1][axis]+cell.vertices[2][axis])/3,sample.position[axis]),
                    "rounded vertex centroid agrees with original sample within float arithmetic tolerance");
                ++index;
            }
            check(unique.size()==n*n&&close(totalArea,triangle.area)&&domainArea==.5,"all ordinal cells cover exactly the source domain and area");
            for(const auto& [edge,count]:edgeCounts) {
                const auto [a,b]=edge;const bool boundary=(a.first==0&&b.first==0)||(a.second==0&&b.second==0)
                    ||(a.first+a.second==n&&b.first+b.second==n);
                check(count==(boundary?1u:2u),"domain tiling has single boundary edges and paired interior edges");
            }
        }
        check(index==frame.entries.size(),"source cell domain coverage consumes all retained entries");
        for(const auto& work:frame.work)check(cells->cells[work.entry].entry==work.entry
            &&cells->cellTexels[work.entry*gc::kGratingEntryCellStride+5][3]==float(work.entry),"native work locates cell by entry index, independent of band or ordinal");
        auto preserved=cells;auto bad=std::make_shared<gc::GratingIncidentBinding>(*parent);
        bad->frame.entries[0].quadrature=unsigned(n*n);
        check(!gc::prepareGratingEntryCellBinding(bad,preserved,error)&&preserved==cells,"invalid ordinal refuses atomically despite legacy identity omitting it");
        bad=std::make_shared<gc::GratingIncidentBinding>(*parent);bad->frame.entries[0].quadrature=1;
        if(n>1)check(!gc::prepareGratingEntryCellBinding(bad,preserved,error)&&preserved==cells,"duplicate or reordered ordinal refuses atomically");
        bad=std::make_shared<gc::GratingIncidentBinding>(*parent);bad->frame.entries[0].object^=1;
        check(!gc::prepareGratingEntryCellBinding(bad,preserved,error)&&preserved==cells,"foreign source owner refuses atomically");
        auto changed=*cells;changed.cells[0].vertices[0][0]+=1;
        check(!gc::validateGratingEntryCellBinding(changed,parent,error),"changed retained world vertex refuses");
        changed=*cells;changed.cells[0].barycentrics[0][0]+=.25f;
        check(!gc::validateGratingEntryCellBinding(changed,parent,error),"changed retained domain refuses");
        changed=*cells;changed.cellTexels[0][0]+=1;
        check(!gc::validateGratingEntryCellBinding(changed,parent,error),"changed native upload refuses");
        changed=*cells;changed.cellTexels[6][0]+=1;
        check(!gc::validateGratingEntryCellBinding(changed,parent,error),"changed private upload version refuses");
        changed=*cells;changed.identity+="stale";
        check(!gc::validateGratingEntryCellBinding(changed,parent,error),"changed private cell identity refuses");
        changed=*cells;--changed.bounds.bytes;
        check(!gc::validateGratingEntryCellBinding(changed,parent,error),"undercharged retained storage refuses");
        auto identicalParent=std::make_shared<const gc::GratingIncidentBinding>(*parent);
        check(!gc::validateGratingEntryCellBinding(*cells,identicalParent,error),"byte-identical replacement parent cannot replace exact strong ownership");
        bad=std::make_shared<gc::GratingIncidentBinding>(*parent);--bad->bounds.bytes;
        check(!gc::prepareGratingEntryCellBinding(bad,preserved,error)&&preserved==cells,"undercharged parent storage cannot fund optional source cells");
        bad=std::make_shared<gc::GratingIncidentBinding>(*parent);bad->frame.sourceIdentity+="foreign";
        check(!gc::prepareGratingEntryCellBinding(bad,preserved,error)&&preserved==cells,"changed source identity cannot mint a retained source domain");
        bad=std::make_shared<gc::GratingIncidentBinding>(*parent);bad->identity+="foreign";
        check(!gc::prepareGratingEntryCellBinding(bad,preserved,error)&&preserved==cells,"changed incident identity cannot mint a retained source domain");
        gc::GratingEntryCellBounds sentinel;sentinel.bytes=42;
        auto capped=*parent;capped.bounds.bytes=renderpassoutput::kMaximumTotalBytes-frame.bounds.transientBytes;
        check(!gc::calculateGratingEntryCellBounds(capped,sentinel,error)&&sentinel.bytes==42,"optional cell upload cannot exceed remaining frame storage");
        capped=*parent;capped.bounds.bytes=renderpassoutput::kMaximumTotalBytes-frame.bounds.transientBytes-cells->bounds.bytes;
        check(gc::calculateGratingEntryCellBounds(capped,sentinel,error)&&sentinel.bytes==cells->bounds.bytes,"exact whole-frame cell storage boundary admits");
        ++capped.bounds.bytes;
        check(!gc::calculateGratingEntryCellBounds(capped,sentinel,error),"one byte beyond cell storage boundary refuses");
        for(std::size_t invalid:std::array<std::size_t,4>{0,3,5,std::numeric_limits<std::size_t>::max()}) {
            bad=std::make_shared<gc::GratingIncidentBinding>(*parent);bad->frame.scheduling.entrySubdivisions=invalid;
            check(!gc::prepareGratingEntryCellBinding(bad,preserved,error)&&preserved==cells,"unsupported source subdivision refuses before any output");
        }
        const auto& triangle=*std::find_if(frame.triangles.begin(),frame.triangles.end(),[&](const auto& t) {
            const auto& sample=frame.entries[0];return std::tie(t.object,t.material,t.instance,t.triangle)==std::tie(sample.object,sample.material,sample.instance,sample.triangle);});
        gc::GratingEntryCell output;output.entry=999;
        auto sample=frame.entries[0];sample.position[0]=std::numeric_limits<float>::quiet_NaN();
        check(!gc::reconstructGratingEntryCell(triangle,sample,frame.scheduling,0,output,error)&&output.entry==999,"nonfinite source centroid refuses atomically");
        auto degenerate=triangle;degenerate.points[1]=degenerate.points[0];
        check(!gc::reconstructGratingEntryCell(degenerate,frame.entries[0],frame.scheduling,0,output,error),"degenerate world source cell refuses");
        sample=frame.entries[0];sample.area=std::numeric_limits<float>::infinity();
        check(!gc::reconstructGratingEntryCell(triangle,sample,frame.scheduling,0,output,error),"nonfinite source area refuses");
        std::shared_ptr<const gc::GratingEntryCellBinding> repeated;
        check(gc::prepareGratingEntryCellBinding(parent,repeated,error)&&repeated->identity==cells->identity,"identical source geometry has stable private identity");
    }
}

// Synthetic native-shaped boundary data only. No ray query or native PASS.
class SyntheticCellFrame final : public arbitgpu::NativeFixtureSceneFrame,public gc::NativeGratingCellSourceBinding
{
public:
    gc::NativeGratingCellSourceUpload uploaded;
    std::uint32_t widthValue=32,heightValue=32;
    const std::string& backend() const noexcept override {static const std::string value="opengl";return value;}
    std::uint32_t width() const noexcept override{return widthValue;}
    std::uint32_t height() const noexcept override{return heightValue;}
    std::uintptr_t colorImageHandle() const noexcept override{return 0;}
    std::uintptr_t colorTextureViewHandle() const noexcept override{return 0;}
    std::uintptr_t nativeResourceCacheIdentity() const noexcept override{return 91;}
    const arbitgpu::NativeFixtureSceneFrame& owner() const noexcept override{return *this;}
    const gc::NativeGratingCellSourceUpload& cellUpload() const noexcept override{return uploaded;}
};
void gratingCellCoverageCases(const std::shared_ptr<const gc::GratingCellTransportBinding>&,
    const gc::NativeGratingCellTransportReadback&,const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>&);
void gratingBoundaryPacketV2(const std::shared_ptr<const gc::GratingCellTransportBinding>&,
    const gc::NativeGratingCellTransportReadback&,const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>&);
void gratingCellTransport()
{
    auto fixture=gt::ordinaryFixture();std::string error;
    for(std::size_t v=0;v<fixture.scene->vertexCount;++v)if(fixture.scene->vertices[v].position.z>0)
        fixture.scene->vertices[v].position.z-=.25f*fixture.scene->vertices[v].position.x;
    fixture.scene->lights[0].transform.rotation={0,0,0,1};
    const auto& target=fixture.scene->objects[1];
    for(std::size_t v=0;v<target.vertexCount;++v) {
        auto& vertex=fixture.scene->vertices[target.firstVertex+v];vertex.uv={vertex.position.x/12,vertex.position.y/12};
    }
    check(fixture.prepare(error),"boundary transport fixture prepares: "+error);
    auto description=diffractionmaterial::makeAluminiumBinaryGratingPreset();
    description.spectrum.lastOrder=1;description.roughness.rmsSlope=0;description.microstructure.grooveDepthNanometres=.9f;
    const auto ir=diffractionmaterial::admit(description,error);check(bool(ir),"boundary transport grating IR admits");if(!ir)return;
    auto material=std::make_shared<const diffractionmaterial::AdmittedDiffractionMaterialIR>(*ir);
    auto grating=std::make_shared<gt::Program>(*fixture.program->objectPrograms[0]);
    grating->kind=arbitgpu::NativeFixtureMaterialKind::DiffractionReflective;grating->spectralReflectance.reset();grating->diffractionPathCount=1;
    grating->diffractionPaths[0].material=diffractionmaterial::physicalcheckpoint::makeGpuParameters(*material,{});
    std::shared_ptr<const gc::GratingIncidentBinding> parent;
    check(gc::prepareGratingIncidentBinding(fixture.prepared,fixture.frame,grating,material,.25,parent,error),"boundary transport original parent prepares: "+error);if(!parent)return;
    std::shared_ptr<const gc::GratingEntryCellBinding> cells;
    check(gc::prepareGratingEntryCellBinding(parent,cells,error),"boundary transport source cells prepare: "+error);if(!cells)return;
    auto nativeOwner=std::make_shared<SyntheticCellFrame>();
    auto point=std::make_shared<gc::NativeGratingReadback>();point->complete=true;point->submission={parent->identity,1,7,3,4,8,7};
    gc::NativeGratingCellTransportRecord record;record.version=gc::kGratingCellTransportVersion;record.complete=true;
    record.incident=0;record.work=0;record.entry=0;record.band=0;record.interfaces=2;record.disposition=gc::GratingCellTransportDisposition::BoundaryWitness;
    const auto& entry=parent->frame.entries[0];const auto& frame=parent->frame;
    const auto top=std::find_if(frame.triangles.begin(),frame.triangles.end(),[&](const auto& t) {return t.glass&&t.triangle==entry.triangle;});
    const auto bottom=std::find_if(frame.triangles.begin(),frame.triangles.end(),[](const auto& t) {return t.glass&&t.triangle==0;});
    const auto receiver=std::find_if(frame.triangles.begin(),frame.triangles.end(),[&](const auto& t) {return t.object==target.id.value&&t.triangle==0;});
    check(top!=frame.triangles.end()&&bottom!=frame.triangles.end()&&receiver!=frame.triangles.end(),"boundary fixture retains three explicit planar owners");
    if(top==frame.triangles.end()||bottom==frame.triangles.end()||receiver==frame.triangles.end())return;
    const std::array<const gc::Triangle*,3> planes{&*top,&*bottom,&*receiver};
    for(unsigned path=0;path<4;++path) {
        const auto start=path?cells->cells[0].vertices[path-1]:entry.position;
        gc::Point origin=gc::advance(start,{0,0,1},2*gc::bias(start));
        for(unsigned depth=0;depth<3;++depth) {
            gc::NativeGratingCellTraceStep step;step.origin=origin;step.direction={0,0,-1};step.depth=depth;step.token=depth<2?65:0;
            const auto& triangle=*planes[depth];step.hit.status=1;step.hit.work=1;step.hit.position={origin[0],origin[1],triangle.points[0][2]};
            step.hit.distance=origin[2]-step.hit.position[2];step.hit.outwardNormal=triangle.normal;step.hit.entering=depth!=1;
            step.hit.objectId=triangle.object;step.hit.materialId=triangle.material;step.hit.instanceId=triangle.instance;step.hit.triangleId=triangle.triangle;
            step.signedBias=depth<2?(depth==0?-1.0f:1.0f)*gc::bias(step.hit.position):0;
            origin=gc::advance(step.hit.position,step.hit.outwardNormal,step.signedBias);record.paths[path].push_back(step);++record.queryWork;
        }
    }
    gc::NativeGratingIncident event;event.version=gc::kGratingIncidentVersion;event.complete=true;event.work=event.band=0;
    event.interfaces=2;event.ancestry[0]=event.ancestry[1]=65;event.fraction=.5f;event.power=frame.work[0].incidentPower*.5f;
    const auto& last=record.paths[0].back();event.origin=last.origin;event.direction=last.direction;event.hit=last.hit;
    const auto& witness=parent->triangles[0];event.normal=witness.normal;event.tangent=witness.tangent;event.bitangent=witness.bitangent;
    const float b2=(event.hit.position[1]+6)/12,b1=(event.hit.position[0]+6)/12-b2;
    event.barycentric={1-b1-b2,b1,b2};
    for(unsigned v=0;v<3;++v)for(unsigned a=0;a<2;++a)event.uv[a]+=event.barycentric[v]*witness.uv[v][a];
    event.dHdu=gc::sub(top->points[1],top->points[0]);event.dHdv=gc::sub(top->points[2],top->points[0]);
    diffractionmaterial::DiffractionOrderDistribution distribution;diffractionmaterial::DiffractionLocalFrame local;
    for(unsigned a=0;a<3;++a){local.tangent[a]=event.tangent[a];local.bitangent[a]=event.bitangent[a];local.normal[a]=event.normal[a];}
    check(diffractionmaterial::evaluateDiffractionOrders(*material,description.spectrum.wavelengthsNanometres[0],local,{0,0,1},{0,0},distribution,error),
        "boundary fixture retains the existing discrete order ledger: "+error);
    event.substrateReflectedPower=float(event.power*distribution.substrateReflectedPower);event.absorbedPower=float(event.power*(1-distribution.substrateReflectedPower));
    event.unresolvedPower=float(event.power*distribution.unresolvedReflectedPower);
    for(std::size_t i=0;i<distribution.orderCount;++i)if(distribution.orders[i].reflectedPower>0) {
        const auto& o=distribution.orders[i];auto& native=event.orders[event.orderCount++];native.order=o.primary;native.power=float(event.power*o.reflectedPower);
        for(unsigned a=0;a<3;++a)native.direction[a]=float(o.direction[a]);
    }
    point->incidents.push_back(event);point->orderWork=3;point->queryWork=event.orderCount;
    const gc::SubmissionIdentity supplemental{parent->identity,1,8,3,4,9,8};
    std::shared_ptr<const gc::GratingCellTransportBinding> binding;
    check(gc::prepareGratingCellTransportBinding(cells,point,nativeOwner,point->submission,supplemental,binding,error),"separate boundary transport binding prepares: "+error);if(!binding)return;
    auto& upload=nativeOwner->uploaded;upload.version=gc::kGratingCellTransportVersion;upload.sourceCells=cells;upload.pointReadback=point;
    upload.pointSubmission=point->submission;upload.submission=supplemental;upload.bindingGeneration=11;upload.cellTexels=cells->cellTexels;upload.complete=true;
    gc::NativeGratingCellTransportReadback readback;readback.version=gc::kGratingCellTransportVersion;readback.binding=binding;
    readback.submission=supplemental;readback.bindingGeneration=11;readback.complete=true;readback.records.push_back(record);readback.queryWork=record.queryWork;
    const auto validate=[&](const auto& value){return gc::validateGratingCellTransportReadback(*binding,value,nativeOwner,supplemental,error);};
    check(validate(readback),"synthetic complete native boundary witness validates without an interior claim: "+error);
    check(binding->bounds.maximumRecords==point->incidents.size()&&binding->bounds.maximumInterfaces==parent->prepared.branchPolicy->description().maximumInterfaces
        &&binding->bounds.recordStride==4+4*(binding->bounds.maximumInterfaces+1)*7,
        "dedicated row uses original dense locator and actual authored maximum interfaces");
    check(binding->bounds.maximumQueries==point->incidents.size()*4*(binding->bounds.maximumInterfaces+1)
        &&binding->bounds.queryWork==binding->bounds.maximumQueries*gc::gratingCellQueryAllowance(frame),"extra reservation charges all center and corner replay queries");
    check(parent->bounds.combinedWork+binding->bounds.queryWork<=gc::kMaximumQueryWork
        &&frame.bounds.transientBytes+parent->bounds.bytes+cells->bounds.bytes+binding->bounds.bytes<=renderpassoutput::kMaximumTotalBytes,
        "supplementary boundary work and storage stay inside existing whole-frame caps");
    auto bad=readback;
    bad.records[0].paths[1].back().hit.triangleId=1;bad.records[0].disposition=gc::GratingCellTransportDisposition::Unsupported;
    check(validate(bad),"a real-shaped foreign corner endpoint is Unsupported while original point remains unchanged");
    bad.records[0].disposition=gc::GratingCellTransportDisposition::BoundaryWitness;
    check(!validate(bad),"foreign corner owner cannot become a complete boundary witness");
    bad=readback;bad.records[0].paths[1].resize(1);bad.records[0].paths[1][0].hit.triangleId=3;bad.records[0].paths[1][0].signedBias=0;
    bad.records[0].disposition=gc::GratingCellTransportDisposition::Unsupported;bad.records[0].queryWork-=2;bad.queryWork-=2;
    check(validate(bad),"corner boundary ownership change retains its failed nearest-hit prefix as Unsupported");
    bad=readback;bad.records[0].paths[2].clear();bad.records[0].disposition=gc::GratingCellTransportDisposition::Unsupported;
    bad.records[0].queryWork-=3;bad.queryWork-=3;
    check(validate(bad),"completed supplemental work may retain an explicitly unsupported unstarted corner");
    bad=readback;bad.records[0].paths[0].pop_back();--bad.records[0].queryWork;--bad.queryWork;
    check(!validate(bad),"supplemental center must complete the original native point lineage");
    bad=readback;bad.records[0].paths[1][0].token=1;check(!validate(bad),"changed exact ancestry token refuses");
    bad=readback;bad.records[0].paths[1][0].signedBias*=-1;check(!validate(bad),"changed signed per-hit interface bias refuses");
    bad=readback;bad.records[0].paths[2][1].origin[0]+=.1f;check(!validate(bad),"broken corner origin continuity refuses");
    bad=readback;bad.records[0].paths[2][1].direction[0]=.5f;check(!validate(bad),"nonunit or changed refracted direction refuses");
    bad=readback;bad.records[0].paths[3][2].hit.position[2]=std::numeric_limits<float>::infinity();check(!validate(bad),"nonfinite native endpoint refuses");
    bad=readback;bad.records[0].incident=1;check(!validate(bad),"foreign original incident locator refuses");
    bad=readback;bad.records[0].entry=1;check(!validate(bad),"foreign original source cell entry refuses");
    bad=readback;++bad.records[0].queryWork;++bad.queryWork;check(!validate(bad),"per-record query total cannot replace actual step work");
    bad=readback;++bad.queryWork;check(!validate(bad),"separate native query total must equal all records");
    bad=readback;bad.complete=false;check(!validate(bad),"unfinished native boundary fence refuses");
    bad=readback;bad.overflow=true;check(!validate(bad),"native output overflow refuses");
    bad=readback;bad.submission.completedFence=7;check(!validate(bad),"older point fence cannot be grafted onto supplemental output");
    const auto savedUpload=upload;upload.complete=false;check(!validate(readback),"unfinished dedicated source upload refuses");upload=savedUpload;
    upload.cellTexels[0][0]+=1;check(!validate(readback),"native source-buffer echo must match exact retained vertices");upload=savedUpload;
    upload.pointReadback=std::make_shared<const gc::NativeGratingReadback>(*point);check(!validate(readback),"byte-identical replacement point parent refuses source completion");upload=savedUpload;
    upload.pointSubmission.completedFence=6;check(!validate(readback),"source input must retain the original completed point fence");upload=savedUpload;
    upload.submission.scratchGeneration=8;check(!validate(readback),"source upload cannot use the point scratch generation");upload=savedUpload;
    auto foreignOwner=std::make_shared<SyntheticCellFrame>();foreignOwner->uploaded=upload;
    check(!gc::validateGratingCellTransportReadback(*binding,readback,foreignOwner,supplemental,error),"byte-identical native frame cannot replace strong original candidate owner");
    for(unsigned change=0;change<6;++change) {
        auto wrong=supplemental;
        if(change==0)wrong.submissionSerial=wrong.completedFence=7;
        if(change==1)wrong.scratchGeneration=8;
        if(change==2)++wrong.rendererGeneration;
        if(change==3)++wrong.rasterGeneration;
        if(change==4)++wrong.queryGeneration;
        if(change==5)wrong.frameIdentity+="foreign";
        auto preserved=binding;
        check(!gc::prepareGratingCellTransportBinding(cells,point,nativeOwner,point->submission,wrong,preserved,error)&&preserved==binding,
            "foreign supplemental generation or shared point fence refuses atomically");
    }
    auto brokenPoint=std::make_shared<gc::NativeGratingReadback>(*point);brokenPoint->complete=false;
    auto preserved=binding;
    check(!gc::prepareGratingCellTransportBinding(cells,brokenPoint,nativeOwner,point->submission,supplemental,preserved,error)&&preserved==binding,
        "unfinished original native point receipt refuses a supplemental binding atomically");
    brokenPoint=std::make_shared<gc::NativeGratingReadback>(*point);brokenPoint->incidents[0].ancestry[1]=1;
    check(!gc::prepareGratingCellTransportBinding(cells,brokenPoint,nativeOwner,point->submission,supplemental,preserved,error)&&preserved==binding,
        "changed original point ancestry refuses before supplementary queries");
    // Independent serialization exercises the separate native layout.
    const auto encode=[&](const gc::NativeGratingCellTransportRecord& r) {
        std::vector<gc::Texel> row(binding->bounds.recordStride);
        row[0]={float(r.version),float(r.incident),float(r.work),float(r.entry)};
        row[1]={float(r.band),float(r.interfaces),float(r.disposition),r.complete?1.0f:0.0f};
        for(unsigned a=0;a<4;++a)row[2][a]=float((r.queryWork>>(a*16))&65535);
        for(unsigned path=0;path<4;++path) {
            row[3][path]=float(r.paths[path].size());
            for(std::size_t d=0;d<r.paths[path].size();++d) {
                const auto& step=r.paths[path][d];const auto& hit=step.hit;
                auto* t=row.data()+4+(path*(binding->bounds.maximumInterfaces+1)+d)*7;
                t[0]={step.origin[0],step.origin[1],step.origin[2],float(step.depth)};
                t[1]={step.direction[0],step.direction[1],step.direction[2],float(step.token)};
                t[2]={float(hit.status),hit.distance,hit.entering?1.0f:0.0f,float(hit.work&65535)};
                t[3]={hit.position[0],hit.position[1],hit.position[2],float(hit.work>>16)};
                t[4]={hit.outwardNormal[0],hit.outwardNormal[1],hit.outwardNormal[2],step.signedBias};
                t[5]={float(hit.objectId&65535),float(hit.objectId>>16),float(hit.materialId&65535),float(hit.materialId>>16)};
                t[6]={float(hit.instanceId&65535),float(hit.instanceId>>16),float(hit.triangleId&65535),float(hit.triangleId>>16)};
            }
        }
        return row;
    };
    auto row=encode(record);gc::NativeGratingCellTransportRecord decoded;
    check(gc::decodeGratingCellTransportRecord(row.data(),*binding,decoded,error)
        &&gc::validateGratingCellTransportRecord(*binding,decoded,0,error),"seven-texel native steps and fixed padded row decode exact boundary ownership");
    for(unsigned change=0;change<5;++change) {
        auto malformed=row;
        if(change==0)malformed[0][1]=.5f;
        if(change==1)malformed[1][3]=0;
        if(change==2)malformed[4][0]=std::numeric_limits<float>::quiet_NaN();
        if(change==3)malformed[4+3*7][0]=1;
        if(change==4)malformed[3][1]=float(binding->bounds.maximumInterfaces+2);
        decoded.incident=999;
        check(!gc::decodeGratingCellTransportRecord(malformed.data(),*binding,decoded,error)&&decoded.incident==999,
            "malformed native row integer, finite value, unused slot or extent refuses atomically");
    }
    gc::GratingCellTransportBounds sentinel;sentinel.bytes=42;
    auto oversizedCells=*cells;auto oversizedParent=std::make_shared<gc::GratingIncidentBinding>(*parent);oversizedCells.incidentBinding=oversizedParent;
    oversizedParent->bounds.combinedWork=gc::kMaximumQueryWork;
    check(!gc::calculateGratingCellTransportBounds(oversizedCells,1,sentinel,error)&&sentinel.bytes==42,"replay budget cannot raise the whole-frame query cap");
    oversizedParent->bounds.combinedWork=parent->bounds.combinedWork;
    oversizedParent->bounds.bytes=renderpassoutput::kMaximumTotalBytes-frame.bounds.transientBytes-cells->bounds.bytes;
    check(!gc::calculateGratingCellTransportBounds(oversizedCells,1,sentinel,error)&&sentinel.bytes==42,"transport output cannot raise the whole-frame byte cap");
    check(gc::kGratingIncidentStride==24&&gc::kGratingOrderStride==4&&gc::kGratingChildReceiptStride==8
        &&point->queryWork==event.orderCount&&point->submission.submissionSerial==7,
        "optional transport preserves released point strides, original query counters and original fence");
    gratingCellCoverageCases(binding,readback,nativeOwner);
    gratingBoundaryPacketV2(binding,readback,nativeOwner);
}

void gratingCellCoveragePartitions()
{
    std::string error;gc::GratingCellDyadicTriangle root;
    check(gc::gratingCellDyadicTriangle(0,0,root)&&root.vertices==std::array<std::array<std::uint32_t,2>,3>{{{0,0},{256,0},{0,256}}},
        "coverage root is the exact integer source-cell triangle");
    check(gc::validateGratingCellDyadicPartition({root},error),"whole source-cell root is a canonical partition");
    std::vector<gc::GratingCellDyadicTriangle> leaves;leaves.reserve(4);
    for(unsigned code=0;code<4;++code) {gc::GratingCellDyadicTriangle leaf;check(gc::gratingCellDyadicTriangle(1,code,leaf),"four dyadic children are representable");leaves.push_back(leaf);}
    check(gc::validateGratingCellDyadicPartition(leaves,error),"four canonical children cover the exact source cell without overlap");
    auto bad=leaves;bad.pop_back();check(!gc::validateGratingCellDyadicPartition(bad,error),"missing leaf leaves a source-domain gap");
    bad=leaves;bad[2]=bad[1];check(!gc::validateGratingCellDyadicPartition(bad,error),"duplicate leaf overlaps the source-domain partition");
    bad=leaves;std::swap(bad[0],bad[1]);check(!gc::validateGratingCellDyadicPartition(bad,error),"noncanonical leaf order refuses");
    bad={root,leaves[0]};check(!gc::validateGratingCellDyadicPartition(bad,error),"ancestor and descendant cannot coexist in a prefix-free partition");
    bad=leaves;++bad[0].vertices[0][0];check(!gc::validateGratingCellDyadicPartition(bad,error),"arbitrary integer triangle vertices cannot replace canonical dyadic domains");
    auto adaptive=leaves;adaptive.pop_back();
    for(unsigned code=12;code<16;++code) {gc::GratingCellDyadicTriangle leaf;gc::gratingCellDyadicTriangle(2,code,leaf);adaptive.push_back(leaf);}
    check(gc::validateGratingCellDyadicPartition(adaptive,error),"adaptive dyadic leaves still tile the exact domain");
    root.code=999;check(!gc::gratingCellDyadicTriangle(9,0,root)&&root.code==999,"excess dyadic depth refuses without changing output");
    check(!gc::gratingCellDyadicTriangle(1,4,root)&&root.code==999,"excess prefix code refuses without changing output");
}
// This fake frame supplies synthetic certificate SHAPE data. Neither these
// bounds nor this fake completion constitute native swept-domain execution.
class SyntheticCoverageFrame final : public arbitgpu::NativeFixtureSceneFrame,
    public gc::NativeGratingCellSourceBinding,public gc::NativeGratingCellCoverageSource
{
public:
    std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame> original;
    gc::NativeGratingCellSourceUpload uploaded;
    gc::NativeGratingCellCoverageEvidence evidence;
    const std::string& backend() const noexcept override{return original->backend();}
    std::uint32_t width() const noexcept override{return original->width();}
    std::uint32_t height() const noexcept override{return original->height();}
    std::uintptr_t colorImageHandle() const noexcept override{return 0;}
    std::uintptr_t colorTextureViewHandle() const noexcept override{return 0;}
    std::uintptr_t nativeResourceCacheIdentity() const noexcept override{return original->nativeResourceCacheIdentity();}
    const arbitgpu::NativeFixtureSceneFrame& owner() const noexcept override{return *this;}
    const gc::NativeGratingCellSourceUpload& cellUpload() const noexcept override{return uploaded;}
    const gc::NativeGratingCellCoverageEvidence& coverageEvidence() const noexcept override{return evidence;}
};
void gratingBoundaryDomainsV2()
{
    std::string error;gc::GratingBoundaryRecordV2 record;record.denominator=6;record.regions.resize(2);
    record.regions[0].vertices={{0,0},{2,0},{0,6}};record.regions[0].twiceArea=12;
    record.regions[1].vertices={{0,6},{2,0},{6,0}};record.regions[1].twiceArea=24;
    gc::GratingDomainIncidenceV2 incidence;
    check(gc::gratingBoundaryPartitionV2(record,4,incidence,error)&&incidence.size()==5,
        "v2 closed non-dyadic rational regions exactly cover source area one: "+error);
    for(unsigned mode=0;mode<8;++mode) {
        auto bad=record;
        if(mode==0)bad.regions.pop_back();
        if(mode==1)bad.regions[1]=bad.regions[0];
        if(mode==2)bad.regions[0].twiceArea=0;
        if(mode==3)bad.regions[0].vertices[1]={3,0};
        if(mode==4)bad.denominator=gc::kGratingBoundaryDenominatorV2+1;
        if(mode==5)bad.regions[0].vertices[1][0]=INT32_MAX;
        if(mode==6)std::swap(bad.regions[0].vertices[1],bad.regions[0].vertices[2]);
        if(mode==7)bad.regions[0].vertices.push_back(bad.regions[0].vertices[0]);
        const auto before=incidence;
        check(!gc::gratingBoundaryPartitionV2(bad,4,incidence,error)&&incidence==before,
            "v2 gap overlap false-zero area width overflow orientation or degenerate polygon refuses atomically");
    }
    // One side has a T-junction at (1,4) on the other's unsplit edge.
    record.regions.resize(3);
    record.regions[0].vertices={{0,0},{3,0},{0,6}};record.regions[0].twiceArea=18;
    record.regions[1].vertices={{0,6},{1,4},{6,0}};record.regions[1].twiceArea=6;
    record.regions[2].vertices={{1,4},{3,0},{6,0}};record.regions[2].twiceArea=12;
    check(gc::gratingBoundaryPartitionV2(record,4,incidence,error)&&incidence.count({gc::GratingDomainPointV2{0,6},{1,4}})==1
        &&incidence.count({gc::GratingDomainPointV2{1,4},{3,0}})==1,
        "v2 edge incidence normalizes exact collinear T-junctions without changing region area");
    check(gc::GratingNativeTriangleKeyV2{1,7,3}<gc::GratingNativeTriangleKeyV2{2,0,0},
        "v2 native tie key follows scene object-array order before instance or triangle; IDs are not keys");
    const std::array<gc::Point,2> exact{{{0,0,0},{1,0,0}}};auto adjacent=exact;
    adjacent[1][0]=std::nextafter(adjacent[1][0],2.0f);
    check(!gc::sameGratingMeshEdgeV2(exact,adjacent),"one-ULP endpoint proximity is not exact mesh seam topology");
}
void gratingBoundaryPacketV2(const std::shared_ptr<const gc::GratingCellTransportBinding>& binding,
    const gc::NativeGratingCellTransportReadback& boundaryValue,const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& owner)
{
    // Synthetic structural records only, never native path/seam execution.
    std::string error;auto boundary=std::make_shared<const gc::NativeGratingCellTransportReadback>(boundaryValue);
    auto expected=boundaryValue.submission;expected.submissionSerial+=2;expected.completedFence=expected.submissionSerial;expected.scratchGeneration+=2;
    std::shared_ptr<const gc::GratingBoundaryCandidateV2> candidate;
    check(gc::prepareGratingBoundaryCandidateV2(boundary,owner,expected,2,1,candidate,error),
        "v2 structural candidate binds exact source/point/boundary and a separate later fence: "+error);if(!candidate)return;
    const auto& parent=*binding->sourceCells->incidentBinding;const auto& frame=parent.frame;
    auto payload=std::make_shared<gc::GratingBoundaryPayloadV2>();payload->records.resize(frame.work.size());
    const auto adjacency=[&](gc::GratingMeshEdgeV2 edge) {
        std::array<gc::Point,2> endpoints{};
        const auto extracted=gc::gratingMeshEdgePointsV2(parent,edge,endpoints,error);
        check(extracted,"v2 structural adjacency extracts actual mesh edge endpoints: "+error);
        std::vector<gc::GratingMeshEdgeV2> result;
        if(!extracted)return result;
        result.reserve(3*frame.triangles.size());
        for(const auto& t:frame.triangles)for(unsigned e=0;e<3;++e)
            if(gc::sameGratingMeshEdgeV2(endpoints,{t.points[e],t.points[(e+1)%3]}))
                result.push_back({{std::uint32_t(frame.instances[t.instance].object),t.instance,t.triangle},e});
        std::sort(result.begin(),result.end());return result;
    };
    for(std::size_t work=0;work<frame.work.size();++work) {
        auto& record=payload->records[work];record.work=std::uint32_t(work);record.entry=frame.work[work].entry;record.band=frame.work[work].band;
        record.denominator=6;record.regions.resize(2);
        record.regions[0].vertices={{0,0},{2,0},{0,6}};record.regions[0].twiceArea=12;
        record.regions[1].vertices={{0,6},{2,0},{6,0}};record.regions[1].twiceArea=24;
        const auto& cell=binding->sourceCells->cells[record.entry];
        for(auto& region:record.regions) {
            region.incidentPower=double(frame.work[work].incidentPower)*region.twiceArea/36;
            region.unresolvedSourcePower=region.incidentPower;
            if(work!=boundaryValue.records[0].work)continue;
            double u=0,v=0;for(auto corner:region.vertices){u+=corner[0];v+=corner[1];}u/=18;v/=18;
            gc::Point sample{};for(unsigned a=0;a<3;++a)sample[a]=float((1-u-v)*cell.vertices[0][a]+u*cell.vertices[1][a]+v*cell.vertices[2][a]);
            gc::GratingBoundaryPathV2 path;path.steps.reserve(boundaryValue.records[0].paths[0].size());
            auto origin=gc::advance(sample,{0,0,1},2*gc::bias(sample));bool inside=false;double fraction=1;
            for(const auto& original:boundaryValue.records[0].paths[0]) {
                gc::GratingBoundaryStepV2 step;step.trace=original;
                step.owner={std::uint32_t(frame.instances[original.hit.instanceId].object),original.hit.instanceId,original.hit.triangleId};
                step.trace.origin=origin;step.trace.hit.position[0]=origin[0];step.trace.hit.position[1]=origin[1];
                step.trace.hit.distance=origin[2]-step.trace.hit.position[2];
                if(step.trace.token) {
                    step.fromIor=inside?frame.ior[record.band]:1;step.toIor=inside?1:frame.ior[record.band];
                    gc::Interface optics;gc::interface(step.trace.direction,inside?gc::advance({},step.trace.hit.outwardNormal,-1):step.trace.hit.outwardNormal,
                        step.fromIor,step.toIor,optics);
                    step.reflection=optics.reflection;step.transmission=1-optics.reflection;fraction*=step.transmission;
                    step.trace.signedBias=(inside?1.0f:-1.0f)*gc::bias(step.trace.hit.position);
                    origin=gc::advance(step.trace.hit.position,step.trace.hit.outwardNormal,step.trace.signedBias);inside=!inside;
                }
                payload->queryWork+=step.trace.hit.work;path.steps.push_back(step);
            }
            path.sourceFraction=fraction;path.unresolvedPower=region.incidentPower*fraction;
            region.unresolvedSourcePower=region.incidentPower*(1-fraction);region.paths.push_back(std::move(path));
        }
        gc::GratingDomainIncidenceV2 edges;gc::gratingBoundaryPartitionV2(record,2,edges,error);record.seams.reserve(edges.size());
        const gc::GratingNativeTriangleKeyV2 sourceKey{std::uint32_t(frame.instances[cell.instance].object),cell.instance,cell.triangle};
        for(const auto& edge:edges) {
            gc::GratingBoundarySeamV2 seam;seam.domain=edge.first;seam.regions={edge.second[0],edge.second.size()==2?edge.second[1]:UINT32_MAX};
            seam.sourceEdge=gc::gratingBoundarySourceEdgeV2(cell,edge.first,record.denominator);
            if(seam.sourceEdge!=UINT32_MAX)seam.sourceAdjacent=adjacency({sourceKey,seam.sourceEdge});
            record.seams.push_back(std::move(seam));
        }
    }
    check(gc::validateGratingBoundaryShapeV2(*candidate,*payload,error),"v2 synthetic exact partition/topology/regional Fresnel power shape validates: "+error);
    check(!gc::validateNativeGratingBoundaryCoverageV2(candidate,payload,owner,expected,error)
        &&error.find("native full-segment float predicate")!=std::string::npos,
        "v2 structural shape and later fence cannot prove native seam winners or grant physical coverage");
    auto unresolved=*payload;unresolved.queryWork=0;
    for(auto& region:unresolved.records[0].regions) {
        region.paths.clear();region.unresolvedSourcePower=region.incidentPower;
    }
    check(gc::validateGratingBoundaryShapeV2(*candidate,unresolved,error),"v2 entirely unresolved regions need no completed path: "+error);
    auto skippedSource=*payload;skippedSource.queryWork=0;
    for(auto& region:skippedSource.records[0].regions) {
        auto& path=region.paths[0];auto endpoint=path.steps.back();
        endpoint.trace.origin=path.steps.front().trace.origin;endpoint.trace.direction=path.steps.front().trace.direction;
        endpoint.trace.hit.distance=endpoint.trace.origin[2]-endpoint.trace.hit.position[2];
        endpoint.trace.depth=0;endpoint.trace.token=0;endpoint.trace.signedBias=0;
        endpoint.fromIor=endpoint.toIor=1;endpoint.reflection=endpoint.transmission=0;
        path.steps.assign(1,endpoint);path.sourceFraction=1;path.reflectedPower=path.absorbedPower=0;
        path.unresolvedPower=region.incidentPower;region.unresolvedSourcePower=0;
    }
    for(const auto& record:skippedSource.records)for(const auto& region:record.regions)
        for(const auto& path:region.paths)for(const auto& step:path.steps)skippedSource.queryWork+=step.trace.hit.work;
    check(!gc::validateGratingBoundaryShapeV2(*candidate,skippedSource,error)
        &&error.find("path omits its source interface")!=std::string::npos,
        "v2 one-step receiver cannot skip the retained source glass interface and claim all regional power");
    for(unsigned mode=0;mode<15;++mode) {
        auto bad=*payload;auto& first=bad.records[0];
        if(mode==0)first.seams.pop_back();
        if(mode==1)first.seams[0].power=1e-20;
        if(mode==2)first.regions[0].incidentPower*=.5;
        if(mode==3)first.regions[0].unresolvedSourcePower=0;
        if(mode==4)first.regions[0].paths[0].steps[0].owner.objectIndex=first.regions[0].paths[0].steps[0].trace.hit.objectId;
        if(mode==5)first.regions[0].paths[0].steps[0].trace.hit.materialId++;
        if(mode==6)first.regions[0].paths[0].steps[0].fromIor=2;
        if(mode==7)first.regions[0].paths[0].steps[0].trace.signedBias*=-1;
        if(mode==8)first.regions[0].paths[0].steps[0].trace.hit.outwardNormal[0]=.1f;
        if(mode==9)bad.queryWork++;
        if(mode==10)for(auto& seam:first.seams)if(!seam.sourceAdjacent.empty()){seam.sourceAdjacent.pop_back();break;}
        if(mode==11)first.regions[0].vertices.reserve(gc::kGratingBoundaryVerticesV2+1);
        if(mode==12)first.seams[0].transitions.push_back({});
        if(mode==13)first.regions[0].paths[0].steps.pop_back();
        if(mode==14)first.regions[0].paths[0].steps[0].trace.token=1;
        check(!gc::validateGratingBoundaryShapeV2(*candidate,bad,error),"v2 omitted seam/topology source power owner material normal IOR bias or work refuses");
    }
    auto stale=*candidate;stale.submission.completedFence=0;
    check(!gc::validateGratingBoundaryCandidateV2(stale,owner,expected,error),"v2 stale native fence refuses");
    auto wrong=expected;wrong.frameIdentity+="foreign";
    check(!gc::validateGratingBoundaryCandidateV2(*candidate,owner,wrong,error),"v2 foreign exact frame/time/material identity refuses");
    auto preserved=candidate;auto reused=expected;reused.scratchGeneration=binding->pointSubmission.scratchGeneration;
    check(!gc::prepareGratingBoundaryCandidateV2(boundary,owner,reused,2,1,preserved,error)&&preserved==candidate,
        "v2 candidate refusal preserves caller ownership and cannot reuse point scratch");
    auto oversizedBinding=std::make_shared<gc::GratingCellTransportBinding>(*binding);
    oversizedBinding->identity.reserve(binding->bounds.hostBytes+1);
    auto oversizedBoundary=std::make_shared<gc::NativeGratingCellTransportReadback>(*boundary);oversizedBoundary->binding=oversizedBinding;
    check(!gc::prepareGratingBoundaryCandidateV2(oversizedBoundary,owner,expected,2,1,preserved,error)&&preserved==candidate,
        "v2 copied parent identity capacity cannot evade the existing reservation or mutate candidate output");
    gc::GratingBoundaryBoundsV2 sentinel;sentinel.bytes=42;auto capped=*binding;capped.bounds.queryWork=gc::kMaximumQueryWork;
    check(!gc::calculateGratingBoundaryBoundsV2(capped,2,1,sentinel,error)&&sentinel.bytes==42,"v2 preserves the original hard query cap atomically");
    capped=*binding;capped.bounds.bytes=renderpassoutput::kMaximumTotalBytes;
    check(!gc::calculateGratingBoundaryBoundsV2(capped,2,1,sentinel,error)&&sentinel.bytes==42,"v2 preserves the original whole-frame byte cap atomically");
}
void gratingCellCoverageCases(const std::shared_ptr<const gc::GratingCellTransportBinding>& boundaryBinding,
    const gc::NativeGratingCellTransportReadback& boundaryValue,const std::shared_ptr<const arbitgpu::NativeFixtureSceneFrame>& nativeOwner)
{
    std::string error;const auto oldBoundary=std::make_shared<const gc::NativeGratingCellTransportReadback>(boundaryValue);
    auto expected=boundaryValue.submission;++expected.submissionSerial;expected.completedFence=expected.submissionSerial;++expected.scratchGeneration;
    std::shared_ptr<const gc::GratingCellCoverageCandidate> candidate;
    check(gc::prepareGratingCellCoverageCandidate(oldBoundary,nativeOwner,boundaryValue.submission,expected,4,candidate,error),
        "coverage candidate retains exact completed boundary and a later separate fence: "+error);if(!candidate)return;
    const auto& frame=boundaryBinding->sourceCells->incidentBinding->frame;
    auto payload=std::make_shared<gc::GratingCellCoveragePayload>();payload->records.resize(boundaryValue.records.size());
    for(std::size_t i=0;i<payload->records.size();++i) {
        auto& record=payload->records[i];const auto& boundary=boundaryValue.records[i];record.incident=std::uint32_t(i);record.entry=boundary.entry;
        record.leaves.reserve(4);
        for(unsigned code=0;code<4;++code) {
            gc::GratingCellCoverageLeaf leaf;gc::gratingCellDyadicTriangle(1,code,leaf.domain);leaf.sweeps.resize(boundary.paths[0].size());
            for(std::size_t depth=0;depth<leaf.sweeps.size();++depth) {
                auto& sweep=leaf.sweeps[depth];const auto& center=boundary.paths[0][depth];const auto& hit=center.hit;
                sweep.depth=std::uint32_t(depth);sweep.token=center.token;sweep.rule=gc::kGratingCellCoverageRule;sweep.entering=hit.entering;
                sweep.object=hit.objectId;sweep.material=hit.materialId;sweep.instance=hit.instanceId;sweep.triangle=hit.triangleId;
                sweep.biasSign=center.token?(center.signedBias>0?1:-1):0;sweep.direction=center.direction;sweep.normal=hit.outwardNormal;
            }
            record.leaves.push_back(std::move(leaf));
        }
    }
    check(gc::validateGratingCellCoverageShape(*candidate,*payload,error),"complete partition with no physical evidence is explicitly Unsupported");
    check(!gc::validateNativeGratingCellCoverage(candidate,payload,nativeOwner,expected,error),
        "existing native boundary frame cannot certify an interior partition without a conservative query interface");
    auto malformed=*payload;malformed.records[0].leaves.pop_back();
    check(!gc::validateGratingCellCoverageShape(*candidate,malformed,error),"coverage payload cannot omit a source-domain leaf");
    malformed=*payload;malformed.records[0].leaves[1].domain=malformed.records[0].leaves[0].domain;
    check(!gc::validateGratingCellCoverageShape(*candidate,malformed,error),"coverage payload cannot overlap domain leaves");
    malformed=*payload;malformed.records[0].leaves[0].sweeps.pop_back();
    check(!gc::validateGratingCellCoverageShape(*candidate,malformed,error),"coverage leaf cannot omit an interface or endpoint");
    malformed=*payload;malformed.records[0].leaves[0].sweeps[0].token=1;
    check(!gc::validateGratingCellCoverageShape(*candidate,malformed,error),"coverage ancestry must equal the original native center branch");
    malformed=*payload;malformed.records[0].leaves[0].sweeps[0].biasSign*=-1;
    check(!gc::validateGratingCellCoverageShape(*candidate,malformed,error),"coverage signed-bias provenance must equal the original interface");
    malformed=*payload;malformed.records[0].leaves[0].sweeps[0].object^=1;
    check(!gc::validateGratingCellCoverageShape(*candidate,malformed,error),"coverage owner cannot cross to another surface");
    malformed=*payload;malformed.records[0].leaves[0].sweeps[0].entering=false;
    check(!gc::validateGratingCellCoverageShape(*candidate,malformed,error),"coverage entering state must preserve exact native interface order");
    malformed=*payload;malformed.records[0].leaves[0].sweeps[0].hitBounds[0].upper=std::numeric_limits<float>::infinity();
    check(!gc::validateGratingCellCoverageShape(*candidate,malformed,error),"nonfinite evidence refuses even an Unsupported row");
    malformed=*payload;malformed.disposition=malformed.records[0].disposition=gc::GratingCellCoverageDisposition::InteriorCertified;
    check(!gc::validateGratingCellCoverageShape(*candidate,malformed,error),"an arbitrary certified flag cannot upgrade Unsupported evidence");
    // Deliberately fabricated numeric intervals exercise shape checks only.
    for(auto& record:payload->records) {
        record.disposition=gc::GratingCellCoverageDisposition::InteriorCertified;
        for(auto& leaf:record.leaves)for(auto& sweep:leaf.sweeps) {
            sweep.disposition=gc::GratingCellCoverageDisposition::InteriorCertified;sweep.roundingExpansion=1e-7f;
            sweep.distance={.01f,1};sweep.originBounds=sweep.hitBounds={{{-3,3},{-3,3},{-3,3}}};
            sweep.barycentricBounds={{{.01f,.98f},{.01f,.98f},{.01f,.98f}}};
            if(sweep.token) {
                sweep.biasMagnitude={gc::kRayMinimum,gc::kRayMinimum};
                sweep.signedBias={sweep.biasSign*gc::kRayMinimum,sweep.biasSign*gc::kRayMinimum};
            }
            sweep.blockers.reserve(frame.triangles.size()-1);
            for(std::size_t triangle=0;triangle<frame.triangles.size();++triangle) {
                const auto& t=frame.triangles[triangle];
                if(std::tie(t.object,t.material,t.instance,t.triangle)==std::tie(sweep.object,sweep.material,sweep.instance,sweep.triangle))continue;
                sweep.blockers.push_back({std::uint32_t(triangle),{1,0,0},{0,1},{2,3}});
            }
            sweep.queryWork=frame.triangles.size();payload->queryWork+=sweep.queryWork;
        }
    }
    payload->disposition=gc::GratingCellCoverageDisposition::InteriorCertified;
    check(gc::validateGratingCellCoverageShape(*candidate,*payload,error),"synthetic conservative certificate shape is internally complete: "+error);
    check(!gc::validateNativeGratingCellCoverage(candidate,payload,nativeOwner,expected,error),
        "synthetic numeric shape still cannot grant coverage through the existing boundary-only backend");
    malformed=*payload;malformed.records[0].leaves[0].sweeps[0].blockers.pop_back();
    check(!gc::validateGratingCellCoverageShape(*candidate,malformed,error),"certificate must cover every retained all-surface blocker");
    malformed=*payload;malformed.records[0].leaves[0].sweeps[0].blockers[0].triangleProjection={.5f,.8f};
    check(!gc::validateGratingCellCoverageShape(*candidate,malformed,error),"possible blocker overlap makes certificate Unsupported");
    malformed=*payload;malformed.records[0].leaves[0].sweeps[0].biasMagnitude.upper=gc::kRayMinimum*.5f;
    check(!gc::validateGratingCellCoverageShape(*candidate,malformed,error),"bias interval must enclose the whole conservative hit box");
    auto fake=std::make_shared<SyntheticCoverageFrame>();fake->original=nativeOwner;
    fake->uploaded=dynamic_cast<const gc::NativeGratingCellSourceBinding&>(*nativeOwner).cellUpload();
    auto fakeBinding=std::make_shared<gc::GratingCellTransportBinding>(*boundaryBinding);fakeBinding->nativeOwner=fake;
    auto fakeBoundary=std::make_shared<gc::NativeGratingCellTransportReadback>(boundaryValue);fakeBoundary->binding=fakeBinding;
    std::shared_ptr<const gc::GratingCellCoverageCandidate> fakeCandidate;
    check(gc::prepareGratingCellCoverageCandidate(fakeBoundary,fake,fakeBoundary->submission,expected,4,fakeCandidate,error),
        "synthetic evidence-owner fixture retains a complete boundary linkage: "+error);if(!fakeCandidate)return;
    auto& evidence=fake->evidence;evidence.version=gc::kGratingCellCoverageVersion;evidence.rule=gc::kGratingCellCoverageRule;
    evidence.candidate=fakeCandidate;evidence.boundary=fakeBoundary;evidence.boundarySubmission=fakeBoundary->submission;evidence.submission=expected;
    evidence.payload=payload;evidence.bindingGeneration=1;evidence.nativeBytes=fakeCandidate->bounds.nativeBytes;evidence.complete=true;
    check(gc::validateNativeGratingCellCoverage(fakeCandidate,payload,fake,expected,error),
        "fake native interface exercises completion identity checks only, not native physical execution");
    const auto savedEvidence=evidence;evidence.complete=false;
    check(!gc::validateNativeGratingCellCoverage(fakeCandidate,payload,fake,expected,error),"unfinished conservative native evidence refuses");evidence=savedEvidence;
    evidence.submission.completedFence=fakeBoundary->submission.completedFence;
    check(!gc::validateNativeGratingCellCoverage(fakeCandidate,payload,fake,expected,error),"boundary fence cannot impersonate the later coverage fence");evidence=savedEvidence;
    auto foreignBoundary=std::make_shared<const gc::NativeGratingCellTransportReadback>(*fakeBoundary);evidence.boundary=foreignBoundary;
    check(!gc::validateNativeGratingCellCoverage(fakeCandidate,payload,fake,expected,error),"byte-identical replacement boundary parent refuses authentication");evidence=savedEvidence;
    evidence.payload=std::make_shared<const gc::GratingCellCoveragePayload>(*payload);
    check(!gc::validateNativeGratingCellCoverage(fakeCandidate,payload,fake,expected,error),"byte-identical replacement native proof payload refuses authentication");evidence=savedEvidence;
    evidence.nativeBytes=1;
    check(!gc::validateNativeGratingCellCoverage(fakeCandidate,payload,fake,expected,error),"native allocation must hold every completed proof row");evidence=savedEvidence;
    auto changedExpected=expected;++changedExpected.queryGeneration;
    check(!gc::validateNativeGratingCellCoverage(fakeCandidate,payload,fake,changedExpected,error),"coverage query generation cannot cross native candidates");
    check(!gc::validateNativeGratingCellCoverage(fakeCandidate,payload,nativeOwner,expected,error),"coverage proof cannot graft onto a different native owner");
    auto preserved=candidate;auto wrongFence=expected;wrongFence.submissionSerial=wrongFence.completedFence=oldBoundary->submission.submissionSerial;
    check(!gc::prepareGratingCellCoverageCandidate(oldBoundary,nativeOwner,oldBoundary->submission,wrongFence,4,preserved,error)&&preserved==candidate,
        "coverage candidate refuses old boundary fence atomically");
    wrongFence=expected;wrongFence.scratchGeneration=boundaryBinding->pointSubmission.scratchGeneration;
    check(!gc::prepareGratingCellCoverageCandidate(oldBoundary,nativeOwner,oldBoundary->submission,wrongFence,4,preserved,error)&&preserved==candidate,
        "coverage candidate cannot reuse the retained original point scratch generation");
    check(!gc::prepareGratingCellCoverageCandidate(oldBoundary,nativeOwner,oldBoundary->submission,expected,257,preserved,error)&&preserved==candidate,
        "coverage candidate refuses unbounded subdivision allocation atomically");
    auto capped=*boundaryBinding;capped.bounds.queryWork=gc::kMaximumQueryWork;
    gc::GratingCellCoverageBounds sentinel;sentinel.bytes=42;
    check(!gc::calculateGratingCellCoverageBounds(capped,4,sentinel,error)&&sentinel.bytes==42,"coverage cannot raise the hard whole-frame query cap");
    capped=*boundaryBinding;capped.bounds.bytes=renderpassoutput::kMaximumTotalBytes;
    check(!gc::calculateGratingCellCoverageBounds(capped,4,sentinel,error)&&sentinel.bytes==42,"coverage cannot raise the hard whole-frame memory cap");
}

}
void legacyLightRecordsAccepted()
{
    using namespace HarmonicMIDI::grid;
    // A legacy diffraction source names no scene light, so modulated Environment and
    // animated records are accepted. A malformed record is still refused.
    auto legacy=gt::fixture(.004f,0,true);std::string error;
    check(legacy.prepare(error),"legacy source prepares without runtime light records: "+error);
    arbitgpu::NativeFixtureSceneRuntimeInputs runtime;
    SceneLightRecord environment=legacy.scene->lights[0];
    environment.color={.75f,.5f,.25f};
    runtime.lightOverride=environment;
    runtime.animatedLights={environment};
    legacy.runtime=runtime;error.clear();
    check(legacy.prepare(error),"legacy source accepts modulated Environment and animated light records: "+error);
    auto malformed=runtime;
    malformed.lightOverride->intensity=std::numeric_limits<float>::quiet_NaN();
    auto refused=gt::fixture(.004f,0,true);refused.runtime=malformed;error.clear();
    check(!refused.prepare(error)&&error.find("Legacy diffraction runtime light inputs are invalid")!=std::string::npos,
        "legacy source refuses a non-finite light record: "+error);
}

int main()
{
    legacyLightRecordsAccepted();
    cameraWideLinePredicatePreparation();cameraLinePredicatePreparation();cameraRayCoefficientPreparation();exactLinePredicatePreparation();gratingBoundaryDomainsV2();gratingCellCoveragePartitions();gratingCellTransport();gratingEntryCells();directionalIrradianceSources();directionalSourceNativeReceipts();physics();gratingAreaDifferentials();gratingIncidentContract();directGratingContributionContract();glassCameraGratingTraceContract();gratingCameraReceipts();roughPhysics();multipleScatterPhysics();multipleScatterReceipts();geometry();branches();ordinarySources();roughBranches();schedulesAndPower();stableDeposition();cameraCellGeometry();cameraAndAtomicReadback();refusals();
    std::printf("glass caustic contract: %d checks, %d failures\n",checks,failures);return failures?1:0;
}
