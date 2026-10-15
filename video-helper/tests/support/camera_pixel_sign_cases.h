#pragma once
#include "camera_wide_line_predicate_cases.h"
#include "../../src/gpu_backend/fixture_native_camera_pixel_signs.h"

namespace videohelper::tests::camerapixelsigncases
{
namespace cp=arbitgpu::camerapixelsigns;
namespace cc=arbitgpu::cameraraycoefficients;
namespace cw=videohelper::tests::camerawidelinecases;
namespace cr=videohelper::tests::cameraraycases;
namespace bits=videohelper::tests::exactlinecases;
struct Case {const char* label;cp::ProbeInput input;bool constructorSupported=true;};
inline cp::Request request(const cc::Request& parent,unsigned form)
{
    cp::Request q{};std::copy(parent.begin(),parent.end(),q.begin());
    q[57]=cp::kVersion;q[58]=cp::kRequestTag;q[59]=0;q[60]=cp::maximumScalarOperations();
    q[61]=form;q[62]=cp::kWireTag;return q;
}
inline Case native(const char* label,const cc::Request& parent,unsigned form,bool supported=true)
{
    Case test{label,{},supported};const auto q=request(parent,form);
    std::copy(q.begin(),q.end(),test.input.begin());return test;
}
inline Case synthetic(const char* label,const cw::Line& line)
{
    auto test=native(label,cr::base(256),0);test.input[65]=1;
    cc::Result r{};r[0]=1;r[1]=cc::maximumVisits(256);r[2]=1;r[3]=256;
    std::copy_n(test.input.begin(),57,r.begin()+4);
    cr::Vec values;for(unsigned k=0;k<3;k++)values[k]={line[k],cr::integer(1)};
    cw::storeForm(r.data()+61,values);r[538]=r[539]=r[548]=r[549]=1;
    std::copy(r.begin(),r.end(),test.input.begin()+66);return test;
}
inline cc::Request diagonal()
{
    auto q=cr::base(256);q[8]=q[9]=0;q[21]=q[22]=0x3f800000u;
    // Nominated ideal bit fixture, not a captured native scene parent. At z=-2
    // the triangle square's +/-2 vertices match this1x1 pixel's ray corners.
    // Its seam connects top-right and bottom-left, preserving endpoint zeros.
    for(unsigned t=0;t<2;t++)for(unsigned v=0;v<3;v++)for(unsigned a=0;a<2;a++){
        auto& word=q[31+17*t+3*v+a];word=(word&0x80000000u)|0x40000000u;}
    return q;
}
inline cp::Result oracle(const Case& test)
{
    cp::Request q{};std::copy_n(test.input.begin(),65,q.begin());cp::Result out{};
    out[1]=cp::maximumScalarOperations();out[2]=cp::kWireTag;std::copy(q.begin(),q.end(),out.begin()+8);
    if(q[57]!=1||q[58]!=cp::kRequestTag||q[59]||q[60]<cp::maximumScalarOperations()
        ||q[61]>=17||q[62]!=cp::kWireTag||q[63]||q[64])return out;
    cc::Request parent{};std::copy_n(q.begin(),57,parent.begin());cc::Result r{};
    if(test.input[65]==0){if(test.constructorSupported)r=cw::modelRecord(parent);}
    else if(test.input[65]==1)std::copy_n(test.input.begin()+66,557,r.begin());
    else return out;
    if(!cw::oracleRecordAccepted(r,parent))return out;
    // Independent dynamic binary digits and reduced rational operations. This
    // evaluates the mathematical affine form, sharing no EL sum/carry/limb code.
    cr::Vec values;for(unsigned k=0;k<3;k++){
        const unsigned at=61+28*q[61];values[k]=cr::readScalar(r[at],r[at+1+9*k],r.data()+at+2+9*k);}
    for(unsigned corner=0;corner<4;corner++){
        const auto u=cr::number(corner&1u),v=cr::number(corner>>1u);
        const auto value=cr::add(cr::add(cr::multiply(values[0],u),cr::multiply(values[1],v)),values[2]);
        out[4+corner]=bits::signWord(value.numerator.sign);}
    out[0]=1;out[3]=cp::maximumVisits();return out;
}
inline std::vector<Case> cases()
{
    const auto n=cw::number;const auto pixel=diagonal();
    std::vector<Case> tests;
    for(unsigned form=0;form<17;form++)tests.push_back(native("native CC all17 affine forms",pixel,form));
    tests.push_back(native("native rotated sqrt-half source coefficients",cw::sourceCamera(),4));
    tests.push_back(native("native non-f32 P256 generated form",cr::cases()[5].input,4));
    tests.push_back(native("native non-power-of-two aspect generated form",cr::cases()[6].input,10));
    tests.push_back(synthetic("synthetic exact diagonal endpoint zeros",{n(1),n(1),n(-1)}));
    tests.push_back(synthetic("synthetic opposed diagonal endpoint zeros",{n(-1),n(-1),n(1)}));
    tests.push_back(synthetic("synthetic all-zero form retains four zeros",{}));
    tests.push_back(synthetic("synthetic uniform positive affine certificate",{n(2),n(-3),n(5)}));
    tests.push_back(synthetic("synthetic uniform negative affine certificate",{n(-2),n(3),n(-5)}));
    bits::Integer m;m.sign=1;m.bits.assign(256,1);const auto minus=bits::negative(m);
    tests.push_back(synthetic("synthetic three extreme P256 magnitudes carry to nine limbs",{m,m,m}));
    tests.push_back(synthetic("synthetic negative nine-limb carry",{minus,minus,minus}));
    tests.push_back(synthetic("synthetic extreme cancellation at two corners",{m,m,minus}));
    tests.push_back(synthetic("synthetic extreme opposite signs",{m,minus,n(0)}));
    auto scaled=synthetic("synthetic positive scale exponent minimum",{n(1),n(1),n(-1)});
    scaled.input[66+61]=static_cast<std::uint32_t>(-512);tests.push_back(scaled);
    scaled.input[66+61]=512;scaled.label="synthetic positive scale exponent maximum";tests.push_back(scaled);
    auto high=synthetic("synthetic non-f32 high limb retains low odd bit",{n(1),n(0),n(-1)});
    high.input[66+70]=0x80000000u;tests.push_back(high);
    const auto seed=synthetic("synthetic base",{n(1),n(1),n(-1)});
    const auto changed=[&](const char* label,unsigned word,std::uint32_t value){auto t=seed;t.label=label;t.input[word]=value;tests.push_back(t);};
    changed("malformed record status refuses",66,0);
    changed("malformed record echo refuses",66+8,0);
    changed("malformed record precision refuses",66+3,30);
    changed("malformed record zero performed work refuses",66+2,0);
    changed("malformed record excessive performed work refuses",66+2,cc::maximumVisits(256)+1);
    changed("malformed record required visits refuses",66+1,cc::maximumVisits(256)-1);
    changed("malformed exponent refuses",66+61,513);
    changed("malformed scalar sign refuses",66+62,3);
    changed("malformed sign-zero magnitude refuses",66+62,0);
    changed("malformed signed zero refuses",66+99,1);
    changed("malformed zero form exponent refuses",66+89,1);
    changed("malformed late form refuses after partial parsing",66+61+28*16,513);
    changed("malformed zero plane height refuses",66+538,0);
    changed("malformed version refuses before parsing",57,2);
    changed("malformed request tag refuses before parsing",58,0);
    changed("malformed operation refuses before parsing",59,1);
    changed("malformed form index refuses before parsing",61,17);
    changed("malformed wire tag refuses before parsing",62,0);
    changed("malformed first reserved word refuses before parsing",63,1);
    changed("malformed second reserved word refuses before parsing",64,1);
    changed("one-short scalar budget retains full charge",60,cp::maximumScalarOperations()-1);
    changed("zero scalar budget retains full charge",60,0);
    changed("raised scalar budget cannot raise required work",60,0xffffffffu);
    changed("invalid probe mode clears diagnostic record",65,2);
    auto even=seed;even.label="malformed all-even canonical form refuses";
    even.input[66+63]=even.input[66+72]=even.input[66+81]=2;tests.push_back(even);
    auto parent=seed;parent.label="malformed expected parent version with matching echo refuses";
    parent.input[0]=parent.input[66+4]=2;tests.push_back(parent);
    parent=seed;parent.label="malformed expected parent reserved with matching echo refuses";
    parent.input[3]=parent.input[66+7]=1;tests.push_back(parent);
    auto bad=cr::base(256);bad[17]=0x7f800000u;tests.push_back(native("native nonfinite CC refuses",bad,0,false));
    bad=cr::base(256);--bad[2];tests.push_back(native("native short CC admission refuses",bad,0,false));
    tests.push_back(native("native precision30 cannot enter P256 parser",cr::base(30),0));
    return tests;
}
inline bool compare(const Case& test,const cp::ProbeResult& actual,std::string& error)
{
    const auto expected=oracle(test);
    for(unsigned k=0;k<73;k++){
        // Refused canonical parsing can stop at any form. Its performed counter
        // is bounded, while full scalar charge, all signs and echo are exact.
        if(k==3&&!expected[0]){if(actual[k]<=cp::maximumVisits())continue;}
        else if(actual[k]==expected[k])continue;
        error=std::string(test.label)+" sign word "+std::to_string(k)+" actual "+std::to_string(actual[k])+
            " expected "+std::to_string(expected[k]);return false;}
    cc::Result record{};std::copy_n(actual.begin()+73,557,record.begin());
    if(test.input[65]==0){
        cc::Request parent{};std::copy_n(test.input.begin(),57,parent.begin());
        if(!cr::compare({test.label,parent,test.constructorSupported},record,error))return false;
    }else if(test.input[65]==1){
        if(!std::equal(record.begin(),record.end(),test.input.begin()+66)){error=std::string(test.label)+" synthetic diagnostic changed";return false;}
    }else if(record!=cc::Result{}){error="invalid mode retained mathematical diagnostic";return false;}
    cp::Request q{};std::copy_n(test.input.begin(),65,q.begin());
    if((q[57]!=1||q[58]!=cp::kRequestTag||q[59]||q[60]<cp::maximumScalarOperations()
        ||q[61]>=17||q[62]!=cp::kWireTag||q[63]||q[64])&&actual[3]!=0){
        error=std::string(test.label)+" invalid closed request performed parser/arithmetic visits";return false;}
    const auto prepared=cp::PreparedForm::prepare(record,q);
    if(bool(prepared)!=bool(expected[0])){error=std::string(test.label)+" host canonical preparation disagrees";return false;}
    if(prepared&&(prepared->record()!=record||prepared->request()!=q)){error="prepared immutable copy changed";return false;}
    // Exact endpoint zeros/opposed corners are certificates of affine behavior,
    // not native ownership, closed union, scene occlusion or full pixel coverage.
    if(test.input[65]==0&&q[21]==0x3f800000u&&q[22]==0x3f800000u&&q[61]==4&&expected[0]){
        if(actual[5]!=0||actual[6]!=0||actual[4]==0||actual[7]!=3u-actual[4]){
            error="nominated pixel diagonal lost exact endpoint zeros/opposite corners";return false;}}
    return true;
}
// Logical test-local admission keeps constructor charges dynamic. Host prepare
// validation/factory copies and diagnostic stores are separate from the kernel.
// Dynamic oracle allocations and actual compiler/pipeline resources stay separate.
constexpr std::uint64_t kHostPrepareScalarOperations=cp::kHostValidationScalarOperations+8*(cp::kFactoryCopyBytes/4);
constexpr std::uint64_t kMaximumProbeWork=cp::maximumScalarOperations()+cc::maximumScalarOperations(256)+
    cp::kProbeDiagnosticScalarOperations+kHostPrepareScalarOperations;
constexpr std::uint64_t kProbeScratchBytes=cp::kLogicalScratchBytes+cc::kLogicalScratchBytes+cp::kFactoryCopyBytes+cp::kHostPrepareWordMoves*4;
inline std::uint64_t work(const Case& test)
{return cp::maximumScalarOperations()+cp::kProbeDiagnosticScalarOperations+kHostPrepareScalarOperations+
    (test.input[65]==0?cc::maximumScalarOperations(test.input[1]):0);}
} // namespace videohelper::tests::camerapixelsigncases
